//! Gemma 4: `<|channel>thought ... <channel|>` reasoning and
//! `<|tool_call>call:name{key:value,...}<tool_call|>` tool calls, with
//! strings quoted `<|"|>`; and the conversion older Gemma 4 templates need
//! from `tool` messages to an assistant message's `tool_responses`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/gemma4.cpp` (llama.cpp v0.5.0,
//! `7fe450e19`): `workaround::gemma4_model_turn_builder`,
//! `workaround::convert_tool_responses_gemma4` and
//! `common_chat_params_init_gemma4`. The parser is built in the order clang
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
const parsers = @import("module.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const ChatParams = params_mod.ChatParams;
const ChatBuilder = chat_peg.ChatBuilder;
const ParserId = peg.ParserId;
const GenerationParams = template.GenerationParams;

// -----------------------------------------------------------------------------
// workaround

/// `m.value(key, "")`: throws a type error for a member that is not a string.
fn valueString(m: *const Value, key: []const u8) Error![]const u8 {
    return m.value(key, []const u8, "") catch |e| util.fromJson(e);
}

/// `msg.at(key)` of a member known to be there.
fn member(m: *const Value, key: []const u8) Error!*Value {
    return m.at(key) catch |e| util.fromJson(e);
}

/// Ports `workaround`'s `gemma4_model_turn_builder` (parsers/gemma4.cpp:15 @7fe450e19):
/// an assistant turn with tool calls, the tool results after it and the
/// final assistant message, folded into one message.
const ModelTurnBuilder = struct {
    a: Allocator,
    messages: *Value,
    pos: usize,
    tool_calls: Value = Value.emptyArray(),
    tool_responses: Value = Value.emptyArray(),
    content: Value = .null,
    reasoning_content: Value = .null,

    /// Ports `collect` (parsers/gemma4.cpp:25 @7fe450e19).
    fn collect(self: *ModelTurnBuilder) Error!void {
        const a = self.a;
        // Collect the first assistant message
        const msg = self.messages.indexAt(a, self.pos) catch |e| return util.fromJson(e);
        if (msg.contains("reasoning_content") and (try member(msg, "reasoning_content")).isString()) {
            // According to the prompt formatting guide, we need to preserve reasoning_content
            // between function calls. The current chat templates do not support this, but we will do it anyway.
            self.reasoning_content = try (try member(msg, "reasoning_content")).clone(a);
        }
        var it = (try member(msg, "tool_calls")).items();
        while (it.next()) |tc| {
            self.tool_calls.pushBack(a, try tc.value.clone(a)) catch |e| return util.fromJson(e);
        }
        self.pos += 1;

        // Collect tool call results
        while (self.pos < self.messages.size() and std.mem.eql(u8, try valueString(self.messages.indexAt(a, self.pos) catch |e| return util.fromJson(e), "role"), "tool")) {
            try self.collectResult(self.messages.indexAt(a, self.pos) catch |err| switch (err) {
                error.OutOfMemory, error.JsonError => unreachable,
            });
            self.pos += 1;
        }

        // Check if the next assistant message is the final message
        if (self.pos < self.messages.size() and std.mem.eql(u8, try valueString(self.messages.indexAt(a, self.pos) catch |e| return util.fromJson(e), "role"), "assistant")) {
            const next = self.messages.indexAt(a, self.pos) catch |err| switch (err) {
                error.OutOfMemory, error.JsonError => unreachable,
            };
            if (!hasToolCalls(next) and hasContent(next)) {
                self.content = try (try member(next, "content")).clone(a);
                self.pos += 1;
            }
        }
    }

    /// Ports `collect_result` (parsers/gemma4.cpp:54 @7fe450e19).
    fn collectResult(self: *ModelTurnBuilder, curr: *const Value) Error!void {
        const a = self.a;
        var response: Value = .null;
        if (curr.contains("content")) {
            const content = try member(curr, "content");
            if (content.isString()) {
                // Try to parse the content as JSON; fall back to raw string
                response = json.parse(a, content.string) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.JsonError => try content.clone(a),
                };
            } else {
                response = try content.clone(a);
            }
        }

        var name: []const u8 = "";

        // Match name with corresponding tool call
        const idx = self.tool_responses.size();
        if (idx < self.tool_calls.size()) {
            const tc = self.tool_calls.indexAt(a, idx) catch |e| return util.fromJson(e);
            if (tc.contains("function")) {
                name = try valueString(try member(tc, "function"), "name");
            }
        }

        // Fallback to the tool call id
        if (name.len == 0) {
            name = try valueString(curr, "tool_call_id");
        }

        var entry = Value.emptyObject();
        try entry.set(a, "name", try Value.from(a, name));
        try entry.set(a, "response", response);
        self.tool_responses.pushBack(a, entry) catch |e| return util.fromJson(e);
    }

    /// Ports `build` (parsers/gemma4.cpp:89 @7fe450e19).
    fn build(self: *ModelTurnBuilder) Error!Value {
        const a = self.a;
        try self.collect();

        var msg = Value.emptyObject();
        try msg.set(a, "role", try Value.from(a, "assistant"));
        try msg.set(a, "tool_calls", try self.tool_calls.clone(a));
        if (!self.tool_responses.empty()) {
            try msg.set(a, "tool_responses", try self.tool_responses.clone(a));
        }
        if (!self.content.isNull()) {
            try msg.set(a, "content", try self.content.clone(a));
        }
        if (!self.reasoning_content.isNull()) {
            try msg.set(a, "reasoning_content", try self.reasoning_content.clone(a));
        }
        return msg;
    }

    /// Ports `has_content` (parsers/gemma4.cpp:108 @7fe450e19).
    fn hasContent(msg: *const Value) bool {
        if (!msg.contains("content")) return false;
        const content = msg.at("content") catch |err| switch (err) {
            error.OutOfMemory, error.JsonError => unreachable,
        };
        if (content.isNull()) return false;
        if (content.isString() and content.string.len != 0) return true;
        if (content.isArray() and !content.empty()) return true;
        return false;
    }

    /// Ports `has_tool_calls` (parsers/gemma4.cpp:122 @7fe450e19).
    fn hasToolCalls(msg: *const Value) bool {
        if (!msg.contains("tool_calls")) return false;
        const tc = msg.at("tool_calls") catch |err| switch (err) {
            error.OutOfMemory, error.JsonError => unreachable,
        };
        return tc.isArray() and !tc.empty();
    }
};

/// Ports `workaround`'s `convert_tool_responses_gemma4` (parsers/gemma4.cpp:127
/// @7fe450e19): each assistant turn with tool calls, the `tool` messages
/// after it and the final assistant reply, as one assistant message with
/// `tool_responses`, as the older Gemma 4 templates expect.
pub fn convertToolResponses(a: Allocator, messages: *Value) Error!void {
    var result = Value.emptyArray();
    var i: usize = 0;

    while (i < messages.size()) {
        const msg = messages.indexAt(a, i) catch |e| return util.fromJson(e);

        const passes = blk: {
            if (!std.mem.eql(u8, try valueString(msg, "role"), "assistant")) break :blk true;
            if (!msg.contains("tool_calls")) break :blk true;
            const tc = try member(msg, "tool_calls");
            break :blk !tc.isArray() or tc.empty();
        };
        if (passes) {
            result.pushBack(a, try msg.clone(a)) catch |e| return util.fromJson(e);
            i += 1;
            continue;
        }

        var builder: ModelTurnBuilder = .{ .a = a, .messages = messages, .pos = i };
        result.pushBack(a, try builder.build()) catch |e| return util.fromJson(e);
        i = builder.pos;
    }

    messages.assign(a, result);
}

// -----------------------------------------------------------------------------
// The parser

fn gemma4Dict(b: *peg.Builder) peg.Error!ParserId {
    const ws = try b.space();
    const m = try b.ref("gemma4-dict-kv");
    const members = try b.sequence(&.{ m, try b.zeroOrMore(try b.sequence(&.{ try b.literal(","), ws, m })) });
    return b.sequence(&.{
        try b.literal("{"),                                                                            ws,
        try b.choice(&.{ try b.literal("}"), try b.sequence(&.{ members, ws, try b.literal("}") }) }),
    });
}

fn gemma4Array(b: *peg.Builder) peg.Error!ParserId {
    const ws = try b.space();
    const value = try b.ref("gemma4-value");
    const elements = try b.sequence(&.{ value, try b.zeroOrMore(try b.sequence(&.{ try b.literal(","), ws, value })) });
    return b.sequence(&.{
        try b.literal("["),                                                                             ws,
        try b.choice(&.{ try b.literal("]"), try b.sequence(&.{ elements, ws, try b.literal("]") }) }),
    });
}

fn gemma4Value(b: *peg.Builder) peg.Error!ParserId {
    return b.choice(&.{
        try b.ref("gemma4-string"), try b.ref("gemma4-dict"), try b.ref("gemma4-array"),
        try b.ref("gemma4-number"), try b.ref("gemma4-bool"), try b.ref("gemma4-null"),
    });
}

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,
    cb: *ChatBuilder = undefined,
    tool_choice: ParserId = undefined,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const start = try b.rule("start", try b.optional(try b.literal("<|turn>model\n")), false);

        if (self.extract_reasoning) {
            const x = try b.plus(try b.plus(try b.literal("<|channel>thought"), try b.space()), try p.reasoning(try b.until("<channel|>")));
            _ = try b.rule("thought", try b.plus(x, try b.literal("<channel|>")), false);
        } else {
            const x = try b.plus(try b.plus(try b.literal("<|channel>thought"), try b.space()), try b.until("<channel|>"));
            _ = try b.rule("thought", try p.content(try b.plus(x, try b.literal("<channel|>"))), false);
        }

        const consume_empty_channels = try b.gbnf(try b.zeroOrMore(try b.plus(try b.literal("<|channel>"), try b.negate(try b.literal("thought")))), "");
        const peeked = try b.plus(try b.plus(try b.peek(try b.literal("<|channel>")), consume_empty_channels), try b.ref("thought"));
        const thought = try b.orr(peeked, try b.negate(try b.literal("<|channel>")));

        if (self.has_response_format) {
            const head = try b.shl(try b.literal("```json"), try p.content(try b.schemaJson(try b.json(), "response-format-schema", &self.inputs.json_schema, false)));
            const response_format = try b.shl(head, try b.literal("```"));
            return b.plus(try b.plus(start, try b.optional(thought)), response_format);
        }

        if (self.has_tools and self.inputs.tool_choice != .none) {
            // Gemma4 tool calling syntax
            // Rules should match traversal logic in gemma4_to_json()
            _ = try b.rule("gemma4-string-content", try b.until("<|\"|>"), false);
            _ = try b.rule("gemma4-string", try b.plus(try b.plus(try b.literal("<|\"|>"), try b.ref("gemma4-string-content")), try b.literal("<|\"|>")), false);
            _ = try b.rule("gemma4-bool", try b.jsonBool(), false);
            _ = try b.rule("gemma4-null", try b.jsonNull(), false);
            _ = try b.rule("gemma4-number", try b.jsonNumber(), false);
            _ = try b.rule("gemma4-dict-key", try b.plus(try b.rule("gemma4-dict-key-name", try b.chars("[^:}]", 1, -1), false), try b.literal(":")), false);
            _ = try b.rule("gemma4-dict-kv", try b.plus(try b.plus(try b.ref("gemma4-dict-key"), try b.space()), try b.ref("gemma4-value")), false);
            _ = try b.ruleFn("gemma4-dict", b, gemma4Dict, false);
            _ = try b.ruleFn("gemma4-array", b, gemma4Array, false);
            _ = try b.ruleFn("gemma4-value", b, gemma4Value, false);

            self.tool_choice = try b.choiceEmpty();

            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

            const min: c_int = if (self.inputs.tool_choice == .required) 1 else 0;
            const max: c_int = if (self.inputs.parallel_tool_calls) -1 else 1;
            const tool_call = try b.triggerRule("tool-call", try b.repeat(try b.plusLit(try b.litPlus("<|tool_call>call:", self.tool_choice), "<tool_call|>"), min, max));

            if (self.inputs.tool_choice == .required) {
                return b.plus(try b.plus(start, thought), tool_call);
            }

            const scan_to_toolcall = try b.rule("scan-to-toolcall", try b.until("<|tool_call>"), false);
            const content = try b.rule("content", try p.content(try b.untilOneOf(&.{ "<|channel>", "<channel|>", "<|tool_call>" })), false);
            const message = try b.rule("message", try b.plus(thought, content), false);
            return b.plus(try b.plus(try b.plus(start, try b.zeroOrMore(message)), scan_to_toolcall), tool_call);
        }

        // Gemma 4 may emit an extra <|channel>thought\n<channel|> at the end of the content. It may
        // also emit a single trailing <channel|> token. Consume all complete reasoning blocks and
        // then stop at the first unmatched <channel|> token.
        const content = try b.rule("content", try p.content(try b.untilOneOf(&.{ "<|channel>", "<channel|>" })), false);
        const message = try b.rule("message", try b.plus(thought, content), false);
        return b.plus(start, try b.oneOrMore(message));
    }

    fn tool(self: *Build, t: *Value) Error!void {
        _ = t.at("function") catch |e| return util.fromJson(e);
        const name = try parsers.functionName(t);
        self.toolPeg(name) catch |e| return util.fromPeg(e);
    }

    fn toolPeg(self: *Build, name: []const u8) chat_peg.Error!void {
        const p = self.cb;
        const b = &p.b;
        // TODO @aldehir : need to extend json-schema-to-grammar to produce more than JSON rules
        // const auto & params   = function.at("parameters");
        const open = try p.toolOpen(try b.plus(try p.toolName(try b.literal(name)), try b.peek(try b.literal("{"))));
        const args = try p.toolArgs(try b.ref("gemma4-dict"));
        const t = try p.tool(try b.sequence(&.{ open, args }));
        try b.orAssign(&self.tool_choice, try b.rule(try std.mem.concat(self.a, u8, &.{ "tool-", name }), t, false));
    }
};

/// Ports `common_chat_params_init_gemma4` (parsers/gemma4.cpp:151 @7fe450e19).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);

    if (inputs.add_generation_prompt and std.mem.endsWith(u8, data.prompt, "<turn|>\n")) {
        // This may happen if the model generates content + tool_call, the
        // template does not add the model's next turn and confuses the model
        // from emitting its proper reasoning token sequence.
        data.generation_prompt = "<|turn>model\n";
        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
    }

    try data.message_delimiters.add(a, .user, "<|turn>user");
    try data.message_delimiters.add(a, .assistant, "<|turn>model");

    data.format = .peg_gemma4;
    data.supports_thinking = true;
    data.thinking_start_tag = "<|channel>thought";
    try data.thinking_end_tags.append(a, "<channel|>");

    try data.preserved_tokens.appendSlice(a, &.{ "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|turn>" });

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;

        const head: []const u8 = if (std.mem.endsWith(u8, data.prompt, "<turn|>\n")) "<|turn>model\n" else "";
        data.generation_prompt = try std.mem.concat(a, u8, &.{ head, "<|channel>thought\n", m.reasoning_content });
        if (inputs.continue_final_message == .content) {
            data.generation_prompt = try std.mem.concat(a, u8, &.{ data.generation_prompt, "<channel|>", try m.renderContent(a, "\n\n") });
        }

        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
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
    var arena = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    defer arena.deinit();

    data.parser = try parsers.save(a, &arena);

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format or (has_tools and inputs.tool_choice == .required));
        data.grammar = try parsers.grammar(a, &arena, data.grammar_lazy);
        try data.addWordTrigger(a, "<|tool_call>");
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "tool results fold into the assistant turn" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var msgs = try json.parse(a,
        \\[{"role":"user","content":"hi"},
        \\ {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":{}}}]},
        \\ {"role":"tool","content":"{\"x\":1}","tool_call_id":"c1"},
        \\ {"role":"assistant","content":"done"}]
    );
    try convertToolResponses(a, &msgs);
    const out = try json.dump(a, &msgs, -1);
    try std.testing.expectEqualStrings(
        \\[{"role":"user","content":"hi"},{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"f","arguments":{}}}],"tool_responses":[{"name":"f","response":{"x":1}}],"content":"done"}]
    , out);
}

test "google-gemma-4-31B-it.jinja applies with a tool and builds a parser and a grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "llama.cpp/models/templates/google-gemma-4-31B-it.jinja", a, .limited(1 << 20));
    var t = try template.Template.init(std.testing.allocator, src, "<s>", "</s>");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(.peg_gemma4, data.format);
    try std.testing.expect(data.prompt.len > 0);
    try std.testing.expect(data.parser.len > 0);
    try std.testing.expect(data.grammar.len > 0);
}
