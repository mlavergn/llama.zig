//! K2 Horizon: `<|ifm|im_start|>` turns, reasoning in one of three think
//! tag pairs picked by `reasoning_effort`, and `<ifm|tool_calls>` sections
//! of `<ifm|tool_call>` calls, as XML arguments (the default) or JSON.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/k2-horizon.cpp` (llama.cpp v0.8.0,
//! `8642541b2`). The parser is built in the order clang evaluates the C++
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

const GEN_PREFIX = "<|ifm|im_start|>assistant\n";
const SECTION_START = "<ifm|tool_calls>";
const SECTION_END = "</ifm|tool_calls>";
const CALL_START = "<ifm|tool_call>";
const CALL_END = "</ifm|tool_call>";
const ARG_KEY = "<ifm|arg_key>";
const ARG_KEY_END = "</ifm|arg_key>";
const ARG_TYPE = "<ifm|arg_type>";
const ARG_TYPE_END = "</ifm|arg_type>";
const ARG_VAL = "<ifm|arg_value>";
const ARG_VAL_END = "</ifm|arg_value>";

/// The lambda `build_chat_peg_parser` runs, and the state its
/// `foreach_function` and `foreach_parameter` callbacks share.
const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    data: *const ChatParams,
    think_start: []const u8,
    call_format: []const u8,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,

    cb: *ChatBuilder = undefined,
    think_end: ParserId = undefined,
    arg_close: ParserId = undefined,
    arg_string: ParserId = undefined,
    arg_type: ParserId = undefined,
    tool_choice: ParserId = undefined,
    // per tool, while its parameters are visited
    tool_index: usize = 0,
    required_args: std.ArrayList(ParserId) = .empty,
    optional_args: std.ArrayList(ParserId) = .empty,

    /// `think_block`: `body` up to an end tag, after the start tag.
    fn thinkBlock(self: *Build, body: ParserId) chat_peg.Error!ParserId {
        const b = &self.cb.b;
        const head = try b.litPlus(self.think_start, try b.space());
        return b.optional(try b.plus(head, try b.ac(try b.plus(body, self.think_end), self.data.thinking_end_tags.items)));
    }

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const generation_prompt = try b.literal(GEN_PREFIX);

        self.think_end = try b.choiceEmpty();
        for (self.data.thinking_end_tags.items) |tag| {
            try b.orAssign(&self.think_end, try b.literal(tag));
        }
        const think_body = try b.untilOneOf(self.data.thinking_end_tags.items);
        var reasoning = if (self.extract_reasoning) try self.thinkBlock(try p.reasoning(think_body)) else try b.eps();

        if (self.has_response_format) {
            // The answer must be bare JSON, so the think block is consumed even when it is not extracted
            const thoughts = if (self.extract_reasoning) reasoning else try self.thinkBlock(think_body);
            const answer = try p.content(try b.schemaJson(try b.json(), "response-format", &self.inputs.json_schema, false));
            return b.plus(generation_prompt, try b.shl(thoughts, answer));
        }

        if (!self.has_tools or self.inputs.tool_choice == .none) {
            return b.plus(generation_prompt, try b.shl(reasoning, try p.content(try b.rest())));
        }

        self.tool_choice = try b.choiceEmpty();
        if (std.mem.eql(u8, self.call_format, "json")) {
            self.tool_choice = try p.standardJsonTools(CALL_START, CALL_END, &self.inputs.tools, false, true, .{});
        } else {
            self.arg_close = try p.toolArgClose(try b.literal(ARG_VAL_END));
            self.arg_string = try b.rule("xml-arg-string", try b.acOne(try b.plus(try p.toolArgStringValue(try b.until(ARG_VAL_END)), self.arg_close), ARG_VAL_END), false);

            // The models leave out <ifm|arg_type> even when asked for xml_typed
            self.arg_type = if (std.mem.eql(u8, self.call_format, "xml_typed"))
                try b.optional(try b.plus(try b.plusLit(try b.litPlus(ARG_TYPE, try b.until(ARG_TYPE_END)), ARG_TYPE_END), try b.space()))
            else
                try b.eps();

            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);
        }

        const required = self.inputs.tool_choice == .required;
        const calls = if (self.inputs.parallel_tool_calls)
            try b.plus(self.tool_choice, try b.zeroOrMore(try b.plus(try b.space(), self.tool_choice)))
        else
            self.tool_choice;
        const section = try b.shlLit(try b.litShl(SECTION_START, calls), SECTION_END);
        const tool_calls = try b.triggerRule("tool-calls", try b.repeat(section, if (required) 1 else 0, 1));

        // Keep thinking inline when required calls bypass the content parser.
        if (required and !self.extract_reasoning) {
            reasoning = try p.content(try self.thinkBlock(think_body));
        }

        // A required call follows the reasoning directly, the models otherwise keep writing content
        const content = if (required) try b.eps() else try p.content(try b.until(SECTION_START));

        return b.plus(generation_prompt, try b.shl(try b.shl(reasoning, content), tool_calls));
    }

    fn tool(self: *Build, tool_index: usize, t: *Value) Error!void {
        self.tool_index = tool_index;
        const function = t.at("function") catch |e| return util.fromJson(e);
        const name = try parsers.functionName(t);
        self.required_args = .empty;
        self.optional_args = .empty;

        try parsers.foreachParameter(self.a, function, self, param);

        self.buildTool(name) catch |e| return util.fromPeg(e);
    }

    fn buildTool(self: *Build, name: []const u8) chat_peg.Error!void {
        const p = self.cb;
        const b = &p.b;
        var args = try p.permute(try parsers.indexName(self.a, "tool-", self.tool_index, "-args"), self.required_args.items);
        if (self.optional_args.items.len != 0) {
            args = try b.plus(args, try b.zeroOrMore(try b.choice(self.optional_args.items)));
        }

        const open = try p.toolOpen(try b.plusLit(try b.litPlus(CALL_START, try p.toolName(try b.literal(name))), "\n"));
        const with_args = try b.plus(open, try p.toolArgs(args));
        const call = try p.tool(try b.shl(with_args, try p.toolClose(try b.literal(CALL_END))));
        try b.orAssign(&self.tool_choice, try b.rule(try parsers.indexName(self.a, "tool-", self.tool_index, ""), call, false));
    }

    fn param(self: *Build, param_index: usize, prm: *const json.schema.Property, doc: *peg.SharedDoc) Error!void {
        const r = self.buildParam(param_index, prm, doc) catch |e| return util.fromPeg(e);
        try (if (prm.required) &self.required_args else &self.optional_args).append(self.a, r);
    }

    fn buildParam(self: *Build, param_index: usize, prm: *const json.schema.Property, doc: *peg.SharedDoc) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        const rule_name = try parsers.argName(self.a, self.tool_index, param_index, "");
        const types = prm.schema.valueTypes() catch |e| return peg.errors.fromSchema(e);
        var arg_value = self.arg_string;
        if (!types.has(.string)) {
            const schema_name = try std.mem.concat(self.a, u8, &.{ rule_name, "-schema" });
            arg_value = try b.plus(try p.toolArgJsonValue(try b.schemaDoc(try b.json(), schema_name, doc, prm.schema, false)), self.arg_close);
        }
        if (types.has(.string) and !types.isOnly(.string)) {
            // The string alternative accepts any text, so only the parser needs the JSON alternatives.
            var json_value = try b.choiceEmpty();
            if (types.has(.object)) {
                try b.orAssign(&json_value, try b.jsonObject());
            }
            if (types.has(.array)) {
                try b.orAssign(&json_value, try b.jsonArray());
            }
            if (types.has(.number) or types.has(.integer)) {
                try b.orAssign(&json_value, try b.jsonNumber());
            }
            if (types.has(.boolean)) {
                try b.orAssign(&json_value, try b.jsonBool());
            }
            if (types.has(.null)) {
                try b.orAssign(&json_value, try b.jsonNull());
            }
            const typed = try b.atomic(try b.plus(try p.toolArgJsonValue(json_value), self.arg_close));
            arg_value = try b.gbnf(try b.orr(typed, self.arg_string), "xml-arg-string");
        }

        const lead = try b.space();
        const key = try b.plusLit(try b.litPlus(ARG_KEY, try p.toolArgName(try b.literal(prm.name))), ARG_KEY_END);
        const open = try p.toolArgOpen(key);
        const value = try b.plus(try b.plusLit(self.arg_type, ARG_VAL), arg_value);
        const arg = try b.plus(lead, try p.toolArg(try b.shl(open, value)));
        return b.rule(rule_name, arg, false);
    }
};

/// Ports `common_chat_params_init_k2_horizon` (parsers/k2-horizon.cpp:8
/// @8642541b2).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    // The template requires a thinking field on every assistant message
    var messages = try inputs.messages.clone(a);
    var it = messages.items();
    while (it.next()) |item| {
        const msg = item.value;
        if (std.mem.eql(u8, msg.value("role", []const u8, "") catch |e| return util.fromJson(e), "assistant") and !msg.contains("reasoning_content")) {
            const slot = msg.index(a, "reasoning_content") catch |e| return util.fromJson(e);
            slot.assign(a, try Value.from(a, ""));
        }
    }

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, &messages, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, &messages, null, null);
    data.format = .peg_native;
    data.supports_thinking = true;

    const effort = inputs.extra_context.value("reasoning_effort", []const u8, "high") catch |e| return util.fromJson(e);
    const call_format = inputs.extra_context.value("tool_call_format", []const u8, "xml") catch |e| return util.fromJson(e);

    // Templates that handle enable_thinking disable it with an empty <ifm|think></ifm|think> block for every effort
    const thinking_off = !inputs.enable_thinking and util.contains(tmpl.source(), "enable_thinking");
    const think: []const u8 = if (thinking_off)
        "ifm|think"
    else if (std.mem.eql(u8, effort, "medium"))
        "ifm|think_fast"
    else if (std.mem.eql(u8, effort, "low"))
        "ifm|think_faster"
    else
        "ifm|think";

    const THINK_START = try std.mem.concat(a, u8, &.{ "<", think, ">" });
    const THINK_END = try std.mem.concat(a, u8, &.{ "</", think, ">" });

    data.thinking_start_tag = THINK_START;
    try data.thinking_end_tags.append(a, THINK_END);

    try data.preserved_tokens.appendSlice(a, data.thinking_end_tags.items);
    try data.preserved_tokens.appendSlice(a, &.{
        THINK_START, SECTION_START, SECTION_END, CALL_START,   CALL_END,
        ARG_KEY,     ARG_KEY_END,   ARG_TYPE,    ARG_TYPE_END, ARG_VAL,
        ARG_VAL_END,
    });

    try data.message_delimiters.add(a, .assistant, "<|ifm|im_start|>assistant");
    try data.message_delimiters.add(a, .user, "<|ifm|im_start|>user");
    try data.message_delimiters.add(a, .tool, "<|ifm|im_start|>tool");
    try data.message_delimiters.add(a, .system, "<|ifm|im_start|>system");

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = parsers.hasResponseFormat(&inputs.json_schema);
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;

        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, GEN_PREFIX);
        try gp.appendSlice(a, THINK_START);
        try gp.appendSlice(a, "\n");
        try gp.appendSlice(a, m.reasoning_content);
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, THINK_END);
            try gp.appendSlice(a, try m.renderContent(a, "\n\n"));
        }
        data.generation_prompt = gp.items;

        try data.prompt.append(a, data.generation_prompt, false);
    }

    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .data = &data,
        .think_start = THINK_START,
        .call_format = call_format,
        .has_tools = has_tools,
        .has_response_format = has_response_format,
        .extract_reasoning = extract_reasoning,
    };
    data.parser = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    const arena = &data.parser.?;

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format or inputs.tool_choice == .required);
        data.grammar = try parsers.grammar(a, arena, data.grammar_lazy);

        if (data.grammar_lazy) {
            try data.addWordTrigger(a, SECTION_START);
        }
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "K2 Horizon picks its think tags by effort and names tool rules by index" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try template.Template.init(std.testing.allocator, "{% for m in messages %}<|ifm|im_start|>{{ m.role }}\n{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|ifm|im_start|>assistant\n{% endif %}<ifm|tool_calls>", "", "");
    defer t.deinit();
    var p: GenerationParams = .{};
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":[\"integer\",\"string\"]}},\"required\":[\"x\"]}}}]");
    p.extra_context = try json.parse(a, "{\"reasoning_effort\":\"low\"}");
    p.add_generation_prompt = true;
    const data = try init(a, &t, &p);
    try std.testing.expectEqualStrings("<ifm|think_faster>", data.thinking_start_tag);
    try std.testing.expect(data.grammar_lazy);
    try std.testing.expect(data.parser.?.hasRule("tool-0"));
    try std.testing.expect(data.parser.?.hasRule("tool-0-arg-0"));
}
