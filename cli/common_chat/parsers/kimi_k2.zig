//! Kimi K2 Thinking: `<think>` reasoning, then tool calls whose id is
//! `functions.<name>:<index>` between `<|tool_call_begin|>` and
//! `<|tool_call_end|>`, inside an optional calls section.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/kimi-k2.cpp` (llama.cpp v0.5.0,
//! `7fe450e19`). The parser is built in the order clang evaluates the C++
//! expressions, left to right, since a parser's id is its position.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("../exception.zig");
const chat_peg = @import("../chat_peg.zig");
const params_mod = @import("../params.zig");
const template = @import("../template.zig");
const util = @import("../util.zig");
const msg_mod = @import("../msg.zig");
const parsers = @import("module.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const ChatParams = params_mod.ChatParams;
const ChatBuilder = chat_peg.ChatBuilder;
const ParserId = peg.ParserId;
const GenerationParams = template.GenerationParams;

const SECTION_BEGIN = "<|tool_calls_section_begin|>";
const SECTION_END = "<|tool_calls_section_end|>";
const CALL_BEGIN = "<|tool_call_begin|>";
const ARGS_BEGIN = "<|tool_call_argument_begin|>";
const CALL_END = "<|tool_call_end|>";

const THINK_START = "<think>";
const THINK_END = "</think>";
const GEN_PROMPT = "<|im_assistant|>assistant<|im_middle|>";

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    extract_reasoning: bool,
    cb: *ChatBuilder = undefined,
    tool_choice: ParserId = undefined,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        // Tool call markers
        const end = try b.end();

        // Note: this model is CRAZY. It can diverge from its supposed tool calling pattern in so many ways it's not funny.
        // For example, it can call tools at the end of reasoning without closing reasoning...
        const reasoning = if (self.extract_reasoning) blk: {
            const r = try b.litPlus(THINK_START, try p.reasoning(try b.untilOneOf(&.{ THINK_END, "<|tool_calls_section_begin|>", "<|tool_call_begin|>" })));
            break :blk try b.optional(try b.plus(r, try b.optional(try b.literal(THINK_END))));
        } else try b.eps();
        const generation_prompt = try b.literal(GEN_PROMPT);

        // Content only parser (no tools)
        if (!self.has_tools or self.inputs.tool_choice == .none) {
            const x = try b.plus(try b.plus(generation_prompt, reasoning), try p.content(try b.rest()));
            return b.plus(x, end);
        }

        // Build tool call parsers for each available function
        // The ID format is: functions.<name>:<index>
        // We need to match: functions.<name>:<digits>
        self.tool_choice = try b.choiceEmpty();
        parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

        // Tool calls section: <|tool_calls_section_begin|> tool_calls <|tool_calls_section_end|>
        const min_calls: c_int = if (self.inputs.tool_choice == .required) 1 else 0;
        const max_calls: c_int = if (self.inputs.parallel_tool_calls) -1 else 1;
        // Use trigger_rule so grammar generator knows where to start generating rules
        const section_begin = try b.optional(try b.literal(SECTION_BEGIN));
        const calls = try b.repeat(try b.litPlus(CALL_BEGIN, self.tool_choice), min_calls, max_calls);
        const trigger = try b.triggerRule("tool-call", try b.plus(calls, try b.optional(try b.literal(SECTION_END))));
        const tool_calls = try b.rule("tool-calls", try b.plus(section_begin, trigger), false);

        const content_before_tools = try p.content(try b.untilOneOf(&.{ SECTION_BEGIN, CALL_BEGIN }));

        const x = try b.plus(try b.plus(try b.plus(generation_prompt, reasoning), content_before_tools), tool_calls);
        return b.plus(x, end);
    }

    fn tool(self: *Build, t: *Value) Error!void {
        self.toolPeg(t) catch |e| return util.fromPeg(e);
    }

    fn toolPeg(self: *Build, t: *Value) chat_peg.Error!void {
        const p = self.cb;
        const b = &p.b;
        const function = t.at("function") catch |e| return util.toPeg(util.fromJson(e));
        const name = parsers.functionName(t) catch |e| return util.toPeg(e);
        const schema = msg_mod.toolParameters(self.a, function) catch |e| return util.toPeg(e);

        // Match: functions.<name>:<digits>
        // Capture the full call id -- functions.<name>:<digits> -- using tool_id tag
        var id = try b.plus(try b.literal("functions."), try p.toolName(try b.literal(name)));
        id = try b.plus(id, try b.literal(":"));
        id = try b.plus(id, try b.chars("[0-9]", 1, -1));
        const tool_id = try p.toolId(id);

        const open = try p.toolOpen(try b.plus(tool_id, try b.literal(ARGS_BEGIN)));
        const schema_name = try std.mem.concat(self.a, u8, &.{ "tool-", name, "-schema" });
        const with_args = try b.plus(open, try p.toolArgs(try b.schemaJson(try b.json(), schema_name, &schema, false)));
        const tool_parser = try p.tool(try b.plus(with_args, try p.toolClose(try b.optional(try b.literal(CALL_END)))));

        try b.orAssign(&self.tool_choice, try b.rule(try std.mem.concat(self.a, u8, &.{ "tool-", name }), tool_parser, false));
    }
};

/// Ports `common_chat_params_init_kimi_k2` (parsers/kimi-k2.cpp:5 @7fe450e19).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_native;
    data.supports_thinking = true;
    try data.preserved_tokens.appendSlice(a, &.{
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<think>",
        "</think>",
    });

    const has_tools = parsers.hasTools(&inputs.tools);
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_tools and inputs.tool_choice != .none;

    data.thinking_start_tag = THINK_START;
    try data.thinking_end_tags.append(a, THINK_END);

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;

        data.generation_prompt = try std.mem.concat(a, u8, &.{ GEN_PROMPT, THINK_START, m.reasoning_content });
        if (inputs.continue_final_message == .content) {
            data.generation_prompt = try std.mem.concat(a, u8, &.{ data.generation_prompt, THINK_END, try m.renderContent(a, "\n\n") });
        }

        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
    }

    var build: Build = .{ .a = a, .inputs = inputs, .has_tools = has_tools, .extract_reasoning = extract_reasoning };
    var arena = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    defer arena.deinit();

    data.parser = try parsers.save(a, &arena);

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == .auto;
        data.grammar = try parsers.grammar(a, &arena, data.grammar_lazy);
        try data.addWordTrigger(a, "<|tool_call_begin|>");
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "Kimi-K2-Thinking.jinja applies with a tool and builds a parser and a grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "llama.cpp/models/templates/Kimi-K2-Thinking.jinja", a, .limited(1 << 20));
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
