//! LFM2 and LFM2.5: almost Python-style, parallel-capable tool calls
//! wrapped in `<|tool_call_start|>[name(args)]<|tool_call_end|>`, with
//! optional `<think>` reasoning.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/lfm2.cpp` (llama.cpp v0.9.0,
//! `cd959bae9`). The parser is built in the order clang evaluates the C++
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
const Value = json.Value;
const Error = exception.Error;
const ChatParams = params_mod.ChatParams;
const ChatBuilder = chat_peg.ChatBuilder;
const ParserId = peg.ParserId;
const GenerationParams = template.GenerationParams;

const tool_call_start = "<|tool_call_start|>";
const tool_call_end = "<|tool_call_end|>";
const tool_list_start = "<|tool_list_start|>";
const tool_list_end = "<|tool_list_end|>";
const think_start = "<think>";
const think_end = "</think>";
const gen_prompt = "<|im_start|>assistant\n";

/// Ports `is_lfm2_template` (parsers/lfm2.cpp:5 @cd959bae9): the tool list wrapped
/// in `<|tool_list_start|>` and `<|tool_list_end|>`.
pub fn isLfm2Template(src: []const u8) bool {
    return util.contains(src, "<|tool_list_start|>") and util.contains(src, "<|tool_list_end|>");
}

const Build = struct {
    inputs: *const GenerationParams,
    extract_reasoning: bool,
    has_tools: bool,
    has_response_format: bool,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        const b = &p.b;
        const generation_prompt = try b.literal(gen_prompt);
        const end = try b.end();

        var reasoning = try b.eps();
        if (self.extract_reasoning) {
            reasoning = try b.optional(try b.plusLit(try b.litPlus(think_start, try p.reasoning(try b.until(think_end))), think_end));
        }

        if (!self.has_tools or self.inputs.tool_choice == .none) {
            if (self.has_response_format) {
                const response_format = try p.content(try b.schemaJson(try b.json(), "response-format-schema", &self.inputs.json_schema, false));
                return b.plus(try b.plus(try b.plus(generation_prompt, reasoning), response_format), end);
            }
            return b.plus(try b.plus(try b.plus(generation_prompt, reasoning), try p.content(try b.rest())), end);
        }
        const calls = try b.plus(try b.plus(try b.literal(tool_call_start), try p.pythonStyleToolCalls(&self.inputs.tools, self.inputs.parallel_tool_calls, true)), try b.literal(tool_call_end));
        const tool_calls = try b.rule("tool-calls", try b.triggerRule("tool-call", calls), false);

        const content = try p.content(try b.until(tool_call_start));

        return b.plus(try b.plus(try b.plus(try b.plus(generation_prompt, reasoning), content), tool_calls), end);
    }
};

/// Ports `common_chat_params_init_lfm2` (parsers/lfm2.cpp:14 @cd959bae9).
/// `tool_list_tokens` preserves LFM2's system tool-list markers; LFM2.5
/// renders without them.
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams, tool_list_tokens: bool) Error!ChatParams {
    var data: ChatParams = .{};

    // Copy reasoning to the "thinking" field the template expects
    var adjusted_messages = Value.emptyArray();
    var it = @constCast(&inputs.messages).items();
    while (it.next()) |item| {
        var msg = try item.value.clone(a);
        if (msg.contains("reasoning_content")) {
            const rc = msg.at("reasoning_content") catch |e| return util.fromJson(e);
            if (rc.isString()) {
                const copy = try rc.clone(a);
                const slot = msg.index(a, "thinking") catch |e| return util.fromJson(e);
                slot.assign(a, copy);
            }
        }
        adjusted_messages.pushBack(a, msg) catch |e| return util.fromJson(e);
    }

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, &adjusted_messages, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, &adjusted_messages, null, null);
    data.format = .peg_native;
    data.supports_thinking = true;
    try data.preserved_tokens.appendSlice(a, &.{ tool_call_start, tool_call_end, think_start, think_end });
    if (tool_list_tokens) {
        try data.preserved_tokens.append(a, tool_list_start);
        try data.preserved_tokens.append(a, tool_list_end);
    }

    data.thinking_start_tag = think_start;
    try data.thinking_end_tags.append(a, think_end);

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = !inputs.json_schema.isNull() and inputs.json_schema.isObject();
    // Gate by reasoning format and whether the template supports <think>
    const extract_reasoning = inputs.reasoning_format != .none and util.contains(tmpl.source(), think_start);
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, gen_prompt ++ think_start);
        try gp.appendSlice(a, m.reasoning_content);
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, think_end);
            try gp.appendSlice(a, try m.renderContent(a, "\n\n"));
        }
        data.generation_prompt = gp.items;
        try data.prompt.append(a, data.generation_prompt, false);
    }

    var build: Build = .{
        .inputs = inputs,
        .extract_reasoning = extract_reasoning,
        .has_tools = has_tools,
        .has_response_format = has_response_format,
    };
    data.parser = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    const arena = &data.parser.?;

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format or (has_tools and inputs.tool_choice == .required));
        data.grammar = try parsers.grammar(a, arena, data.grammar_lazy);
        try data.addWordTrigger(a, tool_call_start);
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "LFM2's template is detected and parses" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = "{% if tools %}<|tool_list_start|>{{ tools | tojson }}<|tool_list_end|>{% endif %}" ++
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}" ++
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    try std.testing.expect(isLfm2Template(src));
    var t = try template.Template.init(std.testing.allocator, src, "<s>", "</s>");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    const data = try init(a, &t, &p, true);
    try std.testing.expectEqual(.peg_native, data.format);
    try std.testing.expect(data.parser != null);
}
