//! Cohere2 MoE ("North Code"): a fully marker-wrapped assistant turn --
//! `<|START_THINKING|>` reasoning, then either `<|START_TEXT|>` content or
//! `<|START_ACTION|>[...]` JSON tool calls.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/cohere2moe.cpp` (llama.cpp v0.6.0,
//! `d81235049`). The parser is built in the order clang evaluates the C++
//! expressions, left to right, since a parser's id is its position.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("../exception.zig");
const chat_peg = @import("../chat_peg.zig");
const params_mod = @import("../params.zig");
const template = @import("../template.zig");
const util = @import("../util.zig");
const parsers = @import("module.zig");

const Allocator = std.mem.Allocator;
const Error = exception.Error;
const ChatParams = params_mod.ChatParams;
const ChatBuilder = chat_peg.ChatBuilder;
const ParserId = peg.ParserId;
const GenerationParams = template.GenerationParams;

const TURN_START = "<|START_OF_TURN_TOKEN|>";
const TURN_END = "<|END_OF_TURN_TOKEN|>";
const CHATBOT = "<|CHATBOT_TOKEN|>";
const USER = "<|USER_TOKEN|>";
const SYSTEM = "<|SYSTEM_TOKEN|>";
const THINK_START = "<|START_THINKING|>";
const THINK_END = "<|END_THINKING|>";
const TEXT_START = "<|START_TEXT|>";
const TEXT_END = "<|END_TEXT|>";
const ACTION_START = "<|START_ACTION|>";
const ACTION_END = "<|END_ACTION|>";
const RESULT_START = "<|START_TOOL_RESULT|>";
const RESULT_END = "<|END_TOOL_RESULT|>";

// Stable prefix of the generation prompt that precedes the (forced) <|START_THINKING|> marker.
const GEN_PREFIX = TURN_START ++ CHATBOT;

const Build = struct {
    inputs: *const GenerationParams,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        const b = &p.b;
        const generation_prompt = try b.literal(GEN_PREFIX);
        const end = try b.end();

        // The thinking block is always present (the generation prompt forces <|START_THINKING|>).
        // When extracting reasoning, capture its body; otherwise keep the whole block (markers
        // included) inline as content, matching reasoning_format=NONE conventions.
        var reasoning = try b.eps();
        if (self.extract_reasoning) {
            const r = try b.plus(try b.literal(THINK_START), try p.reasoning(try b.untilOneOf(&.{ THINK_END, TEXT_START, ACTION_START })));
            reasoning = try b.optional(try b.plus(r, try b.optional(try b.literal(THINK_END))));
        } else {
            const r = try b.plus(try b.literal(THINK_START), try b.untilOneOf(&.{ THINK_END, TEXT_START, ACTION_START }));
            reasoning = try b.optional(try p.content(try b.plus(r, try b.optional(try b.literal(THINK_END)))));
        }

        const text_content = if (self.has_response_format) blk: {
            const head = try b.plus(try b.literal(TEXT_START), try p.content(try b.schemaJson(try b.json(), "response-format-schema", &self.inputs.json_schema, false)));
            break :blk try b.plus(head, try b.optional(try b.literal(TEXT_END)));
        } else blk: {
            const head = try b.plus(try b.literal(TEXT_START), try p.content(try b.until(TEXT_END)));
            break :blk try b.plus(head, try b.optional(try b.literal(TEXT_END)));
        };

        if (!self.has_tools or self.inputs.tool_choice == .none) {
            const x = try b.plus(try b.plus(generation_prompt, reasoning), text_content);
            return b.plus(try b.plus(x, try b.optional(try b.literal(TURN_END))), end);
        }

        const require_tools = self.inputs.tool_choice == .required;

        // <|START_ACTION|>[ {"tool_call_id": "0", "tool_name": "f", "parameters": {...}}, ... ]<|END_ACTION|>
        const tool_calls = try p.standardJsonTools(ACTION_START, ACTION_END, &self.inputs.tools, self.inputs.parallel_tool_calls, true, .{
            .name_key = "tool_name",
            .args_key = "parameters",
            .array_wrapped = true,
            .function_is_key = false,
            .call_id_key = "",
            .gen_call_id_key = "tool_call_id",
            .parameters_order = &.{ "tool_call_id", "tool_name", "parameters" },
        });

        // Content and tool calls are mutually exclusive in this format.
        const body = if (require_tools) tool_calls else try b.choice(&.{ tool_calls, text_content });

        const x = try b.plus(try b.plus(generation_prompt, reasoning), body);
        return b.plus(try b.plus(x, try b.optional(try b.literal(TURN_END))), end);
    }
};

/// Ports `common_chat_params_init_cohere2moe` (parsers/cohere2moe.cpp:19 @d81235049).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_native;
    data.supports_thinking = true;
    data.thinking_start_tag = THINK_START;
    try data.thinking_end_tags.append(a, THINK_END);
    try data.preserved_tokens.appendSlice(a, &.{
        TURN_START,  TURN_END,     CHATBOT,    USER,     SYSTEM,
        THINK_START, THINK_END,    TEXT_START, TEXT_END, ACTION_START,
        ACTION_END,  RESULT_START, RESULT_END,
    });

    // Declare per-role message delimiters. Tool results are rendered with the
    // system token followed by <|START_TOOL_RESULT|>, so the "tool" delimiter must be listed before
    // the plain "system" one (it is a strict superset, and the role split tries delimiters in order).
    try data.message_delimiters.add(a, .assistant, GEN_PREFIX);
    try data.message_delimiters.add(a, .user, TURN_START ++ USER);
    try data.message_delimiters.add(a, .tool, TURN_START ++ SYSTEM ++ RESULT_START);
    try data.message_delimiters.add(a, .system, TURN_START ++ SYSTEM);

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = parsers.hasResponseFormat(&inputs.json_schema);
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;

        data.generation_prompt = try std.mem.concat(a, u8, &.{ GEN_PREFIX, THINK_START, m.reasoning_content });
        if (inputs.continue_final_message == .content) {
            data.generation_prompt = try std.mem.concat(a, u8, &.{ data.generation_prompt, THINK_END, TEXT_START, try m.renderContent(a, "\n\n") });
        }

        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
    }

    var build: Build = .{
        .inputs = inputs,
        .has_tools = has_tools,
        .has_response_format = has_response_format,
        .extract_reasoning = extract_reasoning,
    };
    var arena = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    defer arena.deinit();

    data.parser = try parsers.save(a, &arena);

    if (include_grammar) {
        data.grammar_lazy = !has_response_format and inputs.tool_choice == .auto;
        data.grammar = try parsers.grammar(a, &arena, data.grammar_lazy);

        data.grammar_triggers = .empty;
        try data.addWordTrigger(a, ACTION_START);
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "Cohere2MoE.jinja applies with a tool and builds a parser and a grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "llama.cpp/models/templates/Cohere2MoE.jinja", a, .limited(1 << 20));
    var t = try template.Template.init(std.testing.allocator, src, "<s>", "</s>");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(.peg_native, data.format);
    try std.testing.expect(data.prompt.len > 0);
    try std.testing.expect(data.parser.len > 0);
    try std.testing.expect(data.grammar.len > 0);
}
