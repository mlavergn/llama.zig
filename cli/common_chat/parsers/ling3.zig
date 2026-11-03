//! Ling 3.0 / Bailing V3: `<role>X</role>` sections with tagged tool calls,
//! `<tool_call>name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>...</tool_call>`.
//! The generation prompt pre-opens the think block, so reasoning ends at
//! `</think>` or at the first tool call.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/ling3.cpp` (llama.cpp v0.6.0,
//! `d81235049`), new at v0.5.0. The parser is built in the order clang
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

const ROLE = "<role>ASSISTANT</role>";
const THINK_START = "<think>";
const THINK_END = "</think>";
const CALL_START = "<tool_call>";
const CALL_END = "</tool_call>";
const ARG_KEY = "<arg_key>";
const ARG_KEY_END = "</arg_key>";
const ARG_VAL = "<arg_value>";
const ROLE_END = "<|role_end|>";
const ARG_VAL_END = "</arg_value>";

/// The lambda `build_chat_peg_parser` runs, and the state its
/// `foreach_function` and `foreach_parameter` callbacks share.
const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    think_open: bool,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,

    // per tool, while its parameters are visited
    cb: *ChatBuilder = undefined,
    name: []const u8 = "",
    arg_close: ParserId = undefined,
    arg_string: ParserId = undefined,
    required_args: std.ArrayList(ParserId) = .empty,
    optional_args: std.ArrayList(ParserId) = .empty,
    tool_choices: ParserId = undefined,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const end = try b.end();

        // the effective parse input is generation_prompt + model output, so the
        // assistant opener is optionally consumed here
        const opener = try b.optional(try b.plus(try b.literal(ROLE), try b.optional(try b.space())));

        // the generation prompt pre-opens the think block, so the opening tag
        // is optional; a missing close tag does not swallow a tool call
        const body_end = if (self.think_open)
            try b.untilOneOf(&.{ THINK_END, CALL_START })
        else
            try b.untilOneOf(&.{THINK_END});
        const think_body = if (self.extract_reasoning) try p.reasoning(body_end) else try p.content(body_end);

        const think = try b.plus(try b.optional(try b.literal(THINK_START)), think_body);
        const reasoning = try b.optional(try b.plus(think, try b.optional(try b.literal(THINK_END))));

        // content between the think block and the first tool call, plus any
        // trailing text after the last tool call, are plain content
        const content = try b.optional(try p.content(try b.untilOneOf(&.{CALL_START})));

        // a trailing end-of-turn token is consumed instead of leaking into content
        const tail = try b.plus(try b.optional(try p.content(try b.until(ROLE_END))), try b.optional(try b.literal(ROLE_END)));

        // the think block must close before the JSON, so the turn cannot end inside the reasoning
        if (self.has_response_format) {
            const closed_reasoning = try b.plus(try b.plus(try b.literal(THINK_START), think_body), try b.literal(THINK_END));
            const response_format = try p.content(try b.schemaJson(try b.json(), "response-format", &self.inputs.json_schema, false));
            return b.plus(try b.plus(opener, try b.shl(closed_reasoning, response_format)), end);
        }

        if (!self.has_tools or self.inputs.tool_choice == .none) {
            return b.plus(try b.plus(try b.plus(opener, reasoning), tail), end);
        }

        self.tool_choices = try b.choiceEmpty();
        self.arg_close = try p.toolArgClose(try b.literal(ARG_VAL_END));
        self.arg_string = try b.rule("ling3-arg-string", try b.plus(try p.toolArgStringValue(try b.until(ARG_VAL_END)), self.arg_close), false);

        parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

        const calls = if (self.inputs.parallel_tool_calls)
            try b.plus(self.tool_choices, try b.zeroOrMore(try b.plus(try b.space(), self.tool_choices)))
        else
            self.tool_choices;

        var section = try b.plus(calls, try b.space());
        section = try b.plus(section, try b.optional(try p.content(try b.until(ROLE_END))));
        section = try b.plus(section, try b.optional(try b.literal(ROLE_END)));
        const tools_section = try b.triggerRule("ling3-tool-call", section);

        const tools = if (self.inputs.tool_choice == .required) tools_section else try b.optional(tools_section);

        return b.plus(try b.plus(try b.plus(try b.plus(try b.plus(opener, reasoning), content), tools), tail), end);
    }

    fn tool(self: *Build, t: *Value) Error!void {
        const p = self.cb;
        const b = &p.b;
        const function = t.at("function") catch |e| return util.fromJson(e);
        self.name = try parsers.functionName(t);
        self.required_args = .empty;
        self.optional_args = .empty;

        try parsers.foreachParameter(self.a, function, self, param);

        // required arguments in any order (as Qwen3-Coder does), then
        // optional ones in any order and number
        var args = p.permute(try std.mem.concat(self.a, u8, &.{ "ling3-", self.name, "-args" }), self.required_args.items) catch |e| return util.fromPeg(e);
        if (self.optional_args.items.len != 0) {
            args = b.plus(args, b.zeroOrMore(b.choice(self.optional_args.items) catch |e| return util.fromPeg(e)) catch |e| return util.fromPeg(e)) catch |e| return util.fromPeg(e);
        }

        const call = buildCall(p, self.name, args) catch |e| return util.fromPeg(e);
        const rule = b.rule(try std.mem.concat(self.a, u8, &.{ "ling3-tool-", self.name }), call, false) catch |e| return util.fromPeg(e);
        b.orAssign(&self.tool_choices, rule) catch |e| return util.fromPeg(e);
    }

    fn buildCall(p: *ChatBuilder, name: []const u8, args: ParserId) chat_peg.Error!ParserId {
        const b = &p.b;
        var open = try b.plus(try b.literal(CALL_START), try p.toolName(try b.literal(name)));
        open = try b.plus(open, try b.optional(try b.space()));
        const with_args = try b.plus(try p.toolOpen(open), try p.toolArgs(args));
        const close = try p.toolClose(try b.plus(try b.optional(try b.space()), try b.literal(CALL_END)));
        return p.tool(try b.plus(with_args, close));
    }

    fn param(self: *Build, prm: *const json.schema.Property, doc: *peg.SharedDoc) Error!void {
        const r = self.buildParam(prm, doc) catch |e| return util.fromPeg(e);
        try (if (prm.required) &self.required_args else &self.optional_args).append(self.a, r);
    }

    fn buildParam(self: *Build, prm: *const json.schema.Property, doc: *peg.SharedDoc) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        const rule_name = try std.mem.concat(self.a, u8, &.{ "ling3-arg-", self.name, "-", prm.name });

        const types = prm.schema.valueTypes();

        // string arguments are raw text up to the closing tag, other
        // types parse as JSON per their schema; each alternative
        // consumes the closing tag itself so a JSON prefix can not
        // commit the choice before the tag matches
        var arg_value = try b.eps();
        if (!types.has(.string)) {
            const schema_name = try std.mem.concat(self.a, u8, &.{ rule_name, "-schema" });
            arg_value = try b.plus(try p.toolArgJsonValue(try b.schemaDoc(try b.json(), schema_name, doc, prm.schema, false)), self.arg_close);
        } else if (types.isOnly(.string)) {
            arg_value = self.arg_string;
        } else {
            // the parser tries the JSON alternative first to type the value
            const schema_name = try std.mem.concat(self.a, u8, &.{ rule_name, "-schema" });
            const typed = try b.atomic(try b.plus(try p.toolArgJsonValue(try b.schemaDoc(try b.json(), schema_name, doc, prm.schema, false)), self.arg_close));
            arg_value = try b.gbnf(try b.orr(typed, self.arg_string), "ling3-arg-string");
        }

        const lead = try b.optional(try b.space());
        var key = try b.plus(try b.literal(ARG_KEY), try p.toolArgName(try b.literal(prm.name)));
        key = try b.plus(key, try b.literal(ARG_KEY_END));
        var body = try b.plus(try p.toolArgOpen(key), try b.optional(try b.space()));
        body = try b.plus(body, try b.literal(ARG_VAL));
        body = try b.plus(body, arg_value);
        return b.rule(rule_name, try b.plus(lead, try p.toolArg(body)), false);
    }
};

/// Ports `common_chat_params_init_ling3` (parsers/ling3.cpp:11 @d81235049).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_native;
    data.supports_thinking = true;

    try data.preserved_tokens.appendSlice(a, &.{
        THINK_START, THINK_END,   CALL_START, CALL_END,
        ARG_KEY,     ARG_KEY_END, ARG_VAL,    ARG_VAL_END,
        ROLE_END,
    });

    data.thinking_start_tag = THINK_START;
    // Support both </think> and <tool_call> as reasoning end sequences: a call
    // can be emitted before the think block is closed.
    try data.thinking_end_tags.appendSlice(a, &.{ THINK_END, CALL_START });

    try data.message_delimiters.add(a, .assistant, "<role>ASSISTANT</role>");
    try data.message_delimiters.add(a, .user, "<role>HUMAN</role>");
    try data.message_delimiters.add(a, .tool, "<role>OBSERVATION</role>");
    try data.message_delimiters.add(a, .system, "<role>SYSTEM</role>");

    // the model may spell the end-of-turn control token out as text tokens,
    // which does not stop generation; a literal stop string catches it either
    // way (as the Laguna patch does for its </assistant> token)
    try data.additional_stops.append(a, ROLE_END);

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, ROLE ++ "\n" ++ THINK_START);
        try gp.appendSlice(a, m.reasoning_content);
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, THINK_END);
            try gp.appendSlice(a, try m.renderContent(a, "\n\n"));
        }
        data.generation_prompt = gp.items;
        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
    }

    // The generation prompt pre-opens the think block when thinking is on, so
    // the opening tag is optional here and reasoning runs until </think> or a
    // tool call start; with thinking off the template pre-closes the block and
    // everything the model emits is content.
    var think_open = false;
    if (inputs.hasContinuation()) {
        think_open = inputs.continue_final_message != .content;
    } else {
        const last_open = std.mem.lastIndexOf(u8, data.generation_prompt, THINK_START);
        const last_close = std.mem.lastIndexOf(u8, data.generation_prompt, THINK_END);
        think_open = last_open != null and (last_close == null or last_open.? > last_close.?);
    }

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = parsers.hasResponseFormat(&inputs.json_schema);
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);

    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .think_open = think_open,
        .has_tools = has_tools,
        .has_response_format = has_response_format,
        .extract_reasoning = extract_reasoning,
    };
    var arena = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    defer arena.deinit();

    data.parser = try parsers.save(a, &arena);

    if (include_grammar) {
        data.grammar_lazy = !has_response_format and inputs.tool_choice != .required;
        data.grammar = try parsers.grammar(a, &arena, data.grammar_lazy);
        try data.addWordTrigger(a, CALL_START);
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "Ling 3 builds a lazy tool parser with a role-end stop" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try template.Template.init(std.testing.allocator, "{% for m in messages %}<role>{{ m.role | upper }}</role>{{ m.content }}<|role_end|>{% endfor %}{% if add_generation_prompt %}<role>ASSISTANT</role>\n<think>{% endif %}<arg_key>", "", "");
    defer t.deinit();
    var p: GenerationParams = .{};
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"},\"y\":{\"type\":\"string\"}},\"required\":[\"x\"]}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expect(data.grammar_lazy);
    try std.testing.expect(data.grammar.len != 0);
    try std.testing.expectEqual(@as(usize, 1), data.additional_stops.items.len);
    try std.testing.expectEqualStrings(ROLE_END, data.additional_stops.items[0]);
}
