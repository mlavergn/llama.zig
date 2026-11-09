//! LLM-jp-4.1: gpt-oss's Harmony format with two differences -- the
//! tokenizer emits a space after every special token
//! (`<|channel|> analysis<|message|> ...`), and parallel tool calls are
//! consecutive assistant messages, all but the last closed by `<|end|>`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/llm-jp-harmony.cpp` (llama.cpp v0.8.0,
//! `8642541b2`), new at v0.6.0. The parser is built in the order clang
//! evaluates the C++ expressions, left to right, since a parser's id is its
//! position.

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

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,
    cb: *ChatBuilder = undefined,
    channel: ParserId = undefined,
    constraint: ParserId = undefined,
    message: ParserId = undefined,
    tool_choice: ParserId = undefined,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        // tokenizer space after special tokens; not p.space() since GBNF `space` allows one space only
        const sp = try b.chars("[ ]", 0, -1);
        const channel_tag = try b.plus(try b.literal("<|channel|>"), sp);
        // one space only: keep an intentional leading space in the body
        const message = try b.plus(try b.literal("<|message|>"), try b.optional(try b.literal(" ")));
        self.message = message;

        const start = try b.rule("start", try b.plus(try b.plus(try b.literal("<|start|>"), sp), try b.literal("assistant")), false);
        const end = try b.rule("end", try b.literal("<|end|>"), false);
        const content = try b.rule("message-content", try b.until("<|end|>"), false);
        self.channel = try b.plus(channel_tag, try b.orr(try b.literal("commentary"), try b.literal("analysis")));
        const constrain_type = try b.chars("[A-Za-z0-9_-]", 1, -1);
        self.constraint = try b.optional(try b.plus(try b.plus(try b.space(), try b.optional(try b.plus(try b.literal("<|constrain|>"), sp))), constrain_type));
        const constraint = self.constraint;

        const start_analysis = try b.plus(try b.plus(channel_tag, try b.literal("analysis")), message);
        if (self.extract_reasoning) {
            _ = try b.rule("analysis", try b.plus(try b.plus(start_analysis, try p.reasoning(content)), end), false);
        } else {
            _ = try b.rule("analysis", try p.content(try b.plus(try b.plus(start_analysis, content), end)), false);
        }

        const analysis = try b.ref("analysis");
        var preamble_body = try b.plus(try b.plus(channel_tag, try b.literal("commentary")), message);
        preamble_body = try b.plus(try b.plus(preamble_body, try p.content(content)), end);
        const preamble = try b.rule("preamble", preamble_body, false);
        const final_body = try b.plus(try b.plus(try b.plus(channel_tag, try b.literal("final")), message), try p.content(content));
        const final_msg = try b.rule("final", final_body, false);

        const any = try b.rule("any", try b.orr(preamble, analysis), false);

        if (self.has_response_format) {
            var rf = try b.plus(try b.plus(try b.plus(channel_tag, try b.literal("final")), constraint), message);
            rf = try b.plus(rf, try p.content(try b.schemaJson(try b.json(), "response-format-schema", &self.inputs.json_schema, false)));
            const response_format = try b.rule("response-format", rf, false);

            return b.plus(try b.plus(try b.zeroOrMore(try b.plus(start, analysis)), start), response_format);
        }

        if (self.has_tools and self.inputs.tool_choice != .none) {
            self.tool_choice = try b.choiceEmpty();
            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

            // parallel calls are separated by <|end|>; inside the trigger rule so the lazy grammar covers all of them
            const tool_calls = if (self.inputs.parallel_tool_calls)
                try b.plus(self.tool_choice, try b.zeroOrMore(try b.plus(try b.plus(end, start), self.tool_choice)))
            else
                self.tool_choice;
            const tool_call = try b.triggerRule("tool-call", tool_calls);

            if (self.inputs.tool_choice == .required) {
                return b.plus(try b.plus(try b.zeroOrMore(try b.plus(start, any)), start), tool_call);
            }

            return b.plus(try b.plus(try b.zeroOrMore(try b.plus(start, any)), start), try b.orr(tool_call, final_msg));
        }

        return b.plus(try b.plus(try b.zeroOrMore(try b.plus(start, any)), start), final_msg);
    }

    fn tool(self: *Build, tool_index: usize, t: *Value) Error!void {
        self.toolPeg(tool_index, t) catch |e| return util.fromPeg(e);
    }

    fn toolPeg(self: *Build, tool_index: usize, t: *Value) chat_peg.Error!void {
        const p = self.cb;
        const b = &p.b;
        const function = t.at("function") catch |e| return util.toPeg(util.fromJson(e));
        const name = parsers.functionName(t) catch |e| return util.toPeg(e);
        const params = msg_mod.toolParameters(self.a, function) catch |e| return util.toPeg(e);

        const func_name = try b.plus(try b.literal(" to=functions."), try p.toolName(try b.literal(name)));
        const schema_name = try parsers.indexName(self.a, "tool-", tool_index, "-schema");
        const args = try p.toolArgs(try b.schemaJson(try b.json(), schema_name, &params, false));

        // recipient in role header
        //   <|start|>assistant to=functions.NAME<|channel|>(commentary|analysis)[constraint]<|message|>ARGS
        const role_open = try b.plus(try b.plus(try b.plus(func_name, self.channel), self.constraint), self.message);
        const tool_in_role = try p.tool(try b.plus(try p.toolOpen(role_open), args));

        // recipient in channel header
        //   <|channel|>(commentary|analysis) to=functions.NAME[constraint]<|message|>ARGS
        const channel_open = try b.plus(try b.plus(try b.plus(self.channel, func_name), self.constraint), self.message);
        const tool_in_channel = try p.tool(try b.plus(try p.toolOpen(channel_open), args));

        try b.orAssign(&self.tool_choice, try b.rule(try parsers.indexName(self.a, "tool-", tool_index, ""), try b.orr(tool_in_role, tool_in_channel), false));
    }
};

/// The messages with each one's `reasoning_content` also as `thinking`, as
/// the template reads it, and the content of a message with tool calls
/// dropped. Part of `common_chat_params_init_llm_jp_harmony`.
fn adjustMessages(a: Allocator, messages: *const Value) Error!Value {
    var adjusted = Value.emptyArray();
    var it = @constCast(messages).items();
    while (it.next()) |item| {
        var msg = try item.value.clone(a);
        if (msg.contains("reasoning_content") and (msg.at("reasoning_content") catch |err| switch (err) {
            error.OutOfMemory, error.JsonError => unreachable,
        }).isString()) {
            const reasoning = try (msg.at("reasoning_content") catch |err| switch (err) {
                error.OutOfMemory, error.JsonError => unreachable,
            }).clone(a);
            msg.set(a, "thinking", reasoning) catch |e| return util.fromJson(e);
            if (msg.contains("tool_calls")) {
                const tool_calls = msg.at("tool_calls") catch |err| switch (err) {
                    error.OutOfMemory, error.JsonError => unreachable,
                };
                if (tool_calls.isArray() and !tool_calls.empty()) {
                    msg.erase(a, "content") catch |e| return util.fromJson(e);
                }
            }
        }
        adjusted.pushBack(a, msg) catch |e| return util.fromJson(e);
    }
    return adjusted;
}

/// Ports `common_chat_params_init_llm_jp_harmony` (parsers/llm-jp-harmony.cpp:6 @8642541b2).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    // Copy reasoning to the "thinking" field as expected by the template
    const adjusted_messages = try adjustMessages(a, &inputs.messages);

    var prompt = try template.directApplyImpl(a, tmpl, inputs, &adjusted_messages, null, null);

    // Check if we need to replace the return token with end token during
    // inference and without generation prompt. For more details see:
    // https://github.com/ggml-org/llama.cpp/issues/15417
    if (inputs.is_inference and !inputs.add_generation_prompt) {
        const return_token = "<|return|>";
        const end_token = "<|end|>";
        try prompt.replaceLast(a, return_token, end_token);
    }

    data.prompt = prompt;
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, &adjusted_messages, null, null);
    try data.message_delimiters.add(a, .assistant, "<|start|>assistant");
    try data.message_delimiters.add(a, .user, "<|start|>user");
    try data.message_delimiters.add(a, .system, "<|start|>developer");
    try data.message_delimiters.add(a, .system, "<|start|>system");
    try data.message_delimiters.add(a, .tool, "<|start|>functions");

    data.format = .peg_native;
    data.supports_thinking = true;

    data.thinking_start_tag = "<|channel|>analysis<|message|>";
    try data.thinking_end_tags.append(a, "<|end|>");

    // These special tokens are required to parse properly
    try data.preserved_tokens.appendSlice(a, &.{ "<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>" });

    // Adjust prompt for continuation
    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;

        data.generation_prompt = try std.mem.concat(a, u8, &.{ "<|start|>assistant<|channel|>analysis<|message|>", m.reasoning_content });
        if (inputs.continue_final_message == .content) {
            data.generation_prompt = try std.mem.concat(a, u8, &.{ data.generation_prompt, "<|end|><|start|>assistant<|channel|>final<|message|>", try m.renderContent(a, "\n\n") });
        }

        try data.prompt.append(a, data.generation_prompt, false);
    }

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = !inputs.json_schema.isNull() and inputs.json_schema.isObject();
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);
    const extract_reasoning = inputs.reasoning_format != .none;

    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .has_tools = has_tools,
        .has_response_format = has_response_format,
        .extract_reasoning = extract_reasoning,
    };
    data.parser = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    const arena = &data.parser.?;

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format or (has_tools and inputs.tool_choice == .required));
        data.grammar = try parsers.grammar(a, arena, data.grammar_lazy);

        try data.grammar_triggers.appendSlice(a, &.{
            .{ .type = .pattern, .value = "^\\s+to$" },
            .{ .type = .pattern, .value = "^<\\|channel\\|>\\s*(?:commentary|analysis)\\s+to=functions$" },
            .{ .type = .pattern, .value = "<\\|start\\|>\\s*assistant(\\s+to)" },
            .{ .type = .pattern, .value = "<\\|start\\|>\\s*assistant(<\\|channel\\|>\\s*(?:commentary|analysis)\\s+to)" },
        });
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "llm-jp-llm-jp-4.1-8b-thinking.jinja applies with a tool and builds a parser and a grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "llama.cpp/models/templates/llm-jp-llm-jp-4.1-8b-thinking.jinja", a, .limited(1 << 20));
    var t = try template.Template.init(std.testing.allocator, src, "<s>", "</s>");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true, .parallel_tool_calls = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(.peg_native, data.format);
    try std.testing.expect(data.prompt.parts.items.len > 0);
    try std.testing.expect(data.parser != null);
    try std.testing.expect(data.grammar_lazy);
    try std.testing.expect(data.grammar.len > 0);
    try std.testing.expectEqual(@as(usize, 4), data.grammar_triggers.items.len);
}
