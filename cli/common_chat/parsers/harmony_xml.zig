//! Harmony with XML tool calls:
//! `<|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="KEY">\nVALUE\n</rfl:parameter>`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/harmony-xml.cpp` (llama.cpp v0.9.0,
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
const s2g = json.schema_to_grammar;

/// EOG text must reach the parser: an XML call has no closing tag of its own
const eog_tokens = [_][]const u8{
    "<|call|>", "<|return|>", "<|endoftext|>", "<|im_end|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
};

const param_open = "<rfl:parameter name=\"";
const param_close = "\n</rfl:parameter>";

/// `with_eog`: `delims` followed by the EOG tokens.
fn withEog(a: Allocator, delims: []const []const u8) Error![]const []const u8 {
    return std.mem.concat(a, []const u8, &.{ delims, &eog_tokens });
}

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,
    cb: *ChatBuilder = undefined,
    arg_close: ParserId = undefined,
    arg_string: ParserId = undefined,
    arg_sep: ParserId = undefined,
    call_close: ParserId = undefined,
    tool_choice: ParserId = undefined,
    name: []const u8 = "",
    tool_index: usize = 0,
    arg_rules: std.ArrayList(ParserId) = .empty,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const a = self.a;
        const b = &p.b;
        const start = try b.rule("start", try b.literal("<|start|>assistant"), false);

        var eog = try b.choiceEmpty();
        for (eog_tokens) |tok| try b.orAssign(&eog, try b.literal(tok));

        const analysis_open = try b.literal("<|channel|>analysis<|message|>");
        const analysis_body = try b.untilOneOf(try withEog(a, &.{ "<|end|>", "<|constrain|><|start|>assistant", "<|start|>assistant" }));
        const analysis_close = try b.optional(try b.orr(try b.orr(try b.literal("<|end|>"), try b.literal("<|constrain|>")), eog));
        if (self.extract_reasoning) {
            _ = try b.rule("analysis", try b.plus(try b.plus(analysis_open, try p.reasoning(analysis_body)), analysis_close), false);
        } else {
            _ = try b.rule("analysis", try b.plus(try b.atomic(try p.content(analysis_open)), try p.content(try b.plus(analysis_body, analysis_close))), false);
        }
        const analysis = try b.ref("analysis");

        const final_msg = try b.rule("final", try b.plus(try b.plus(try b.literal("<|channel|>final<|message|>"), try p.content(try b.untilOneOf(try withEog(a, &.{"<|end|>"})))), try b.optional(try b.orr(try b.literal("<|end|>"), eog))), false);

        if (self.has_response_format) {
            const constrain_type = try b.chars("[A-Za-z0-9_-]", 1, -1);
            const constraint = try b.optional(try b.plus(try b.plus(try b.space(), try b.optional(try b.literal("<|constrain|>"))), constrain_type));
            var response_format = try b.plus(try b.literal("<|channel|>final"), constraint);
            response_format = try b.plus(response_format, try b.literal("<|message|>"));
            response_format = try b.plus(response_format, try p.content(try b.schemaJson(try b.json(), "response-format-schema", &self.inputs.json_schema, false)));
            response_format = try b.plus(response_format, try b.optional(try b.orr(try b.literal("<|end|>"), eog)));
            response_format = try b.rule("response-format", response_format, false);

            return b.plus(try b.plus(start, try b.zeroOrMore(try b.plus(analysis, start))), response_format);
        }

        if (self.has_tools and self.inputs.tool_choice != .none) {
            // tolerate a missing closing tag
            const value_ends = try withEog(a, &.{ param_open, "<|end|>" });
            var string_ends: std.ArrayList([]const u8) = .empty;
            try string_ends.append(a, param_close);
            var next_open = try b.choiceEmpty();
            for (value_ends) |e| {
                try string_ends.append(a, try std.mem.concat(a, u8, &.{ "\n", e }));
                try b.orAssign(&next_open, try b.literal(e));
            }
            const close_alts = try b.orr(try b.literal(param_close), try b.plus(try b.literal("\n"), try b.peek(try b.plus(try b.optional(try b.literal("\n")), next_open))));
            const close_gbnf = try std.mem.concat(a, u8, &.{
                try gbnfLiteral(a, "\n"),
                " ",
                try gbnfLiteral(a, param_close[1..]),
                "?",
            });
            self.arg_close = try p.toolArgClose(try b.gbnf(close_alts, close_gbnf));
            self.arg_string = try b.rule("rfl-arg-string", try b.acOne(try b.plus(try p.toolArgStringValue(try b.untilOneOf(string_ends.items)), self.arg_close), param_close), false);
            self.arg_sep = try b.optional(try b.literal("\n"));

            var call_close_gbnf: std.ArrayList(u8) = .empty;
            try call_close_gbnf.appendSlice(a, try gbnfLiteral(a, "<|end|>"));
            for (eog_tokens) |tok| {
                try call_close_gbnf.appendSlice(a, " | ");
                try call_close_gbnf.appendSlice(a, try gbnfLiteral(a, tok));
            }
            const call_alts = try b.orr(try b.orr(try b.literal("<|end|>"), eog), try b.peek(try b.literal("<|start|>assistant")));
            self.call_close = try b.gbnf(call_alts, try std.mem.concat(a, u8, &.{ "(", call_close_gbnf.items, ")?" }));

            self.tool_choice = try b.choiceEmpty();
            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

            const tool_calls = if (self.inputs.parallel_tool_calls)
                try b.plus(self.tool_choice, try b.zeroOrMore(try b.plus(start, self.tool_choice)))
            else
                self.tool_choice;
            const tool_call = try b.triggerRule("tool-call", tool_calls);

            if (self.inputs.tool_choice == .required) {
                return b.plus(try b.plus(start, try b.zeroOrMore(try b.plus(analysis, start))), tool_call);
            }

            const head = try b.plus(start, try b.zeroOrMore(try b.plus(analysis, start)));
            return b.plus(head, try b.orr(try b.orr(tool_call, final_msg), try p.content(try b.rest())));
        }

        const unsolicited = try b.rule("unsolicited", try b.atomic(try b.plus(try b.literal(" to="), try b.rest())), false);

        // output without a channel header is returned as content instead of failing the request
        const head = try b.plus(start, try b.zeroOrMore(try b.plus(analysis, start)));
        return b.plus(head, try b.orr(try b.orr(final_msg, unsolicited), try p.content(try b.rest())));
    }

    /// `gbnf_format_literal`.
    fn gbnfLiteral(a: Allocator, s: []const u8) chat_peg.Error![]u8 {
        return s2g.gbnfFormatLiteral(a, s) catch |e| peg.errors.fromSchema(e);
    }

    fn tool(self: *Build, tool_index: usize, t: *Value) Error!void {
        self.tool_index = tool_index;
        const function = t.at("function") catch |e| return util.fromJson(e);
        self.name = try parsers.functionName(t);
        self.arg_rules = .empty;
        try parsers.foreachParameter(self.a, function, self, param);
        self.buildTool() catch |e| return util.fromPeg(e);
    }

    fn buildTool(self: *Build) chat_peg.Error!void {
        const p = self.cb;
        const b = &p.b;
        const a = self.a;

        if (self.arg_rules.items.len == 0) {
            const any_name = try b.chars("[^\"\\n<]", 1, -1);
            const rule_name = try parsers.indexName(a, "tool-", self.tool_index, "-arg-any");
            const open = try p.toolArgOpen(try b.plus(try b.plus(try b.literal(param_open), try p.toolArgName(any_name)), try b.literal("\">\n")));
            try self.arg_rules.append(a, try b.rule(rule_name, try p.toolArg(try b.plus(open, self.arg_string)), false));
        }

        const args = try b.zeroOrMore(try b.plus(try b.choice(self.arg_rules.items), self.arg_sep));

        var open = try b.plus(try b.literal(" to="), try p.toolName(try b.literal(self.name)));
        open = try b.plus(open, try b.literal("<|channel|>analysis <|constrain|>xml<|message|>"));
        const with_args = try b.plus(try p.toolOpen(open), try p.toolArgs(args));
        const call = try p.tool(try b.plus(with_args, try p.toolClose(self.call_close)));

        try b.orAssign(&self.tool_choice, try b.rule(try parsers.indexName(a, "tool-", self.tool_index, ""), call, false));
    }

    fn param(self: *Build, param_index: usize, prop: *const json.schema.Property, doc: *peg.SharedDoc) Error!void {
        const r = self.buildParam(param_index, prop, doc) catch |e| return util.fromPeg(e);
        try self.arg_rules.append(self.a, r);
    }

    fn buildParam(self: *Build, param_index: usize, prop: *const json.schema.Property, doc: *peg.SharedDoc) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        const rule_name = try parsers.argName(self.a, self.tool_index, param_index, "");
        var open = try b.plus(try b.literal(param_open), try p.toolArgName(try b.literal(prop.name)));
        open = try b.plus(open, try b.literal("\">\n"));
        const arg_open = try p.toolArgOpen(open);

        const types = prop.schema.valueTypes() catch |e| return peg.errors.fromSchema(e);
        var arg_value = try b.eps();
        if (!types.has(.string) and (types.has(.object) or types.has(.array))) {
            // the schema grammar fixes the key order, and the model writes keys in its own order
            arg_value = try b.plus(try p.toolArgJsonValue(try b.json()), self.arg_close);
        } else if (!types.has(.string)) {
            const schema_name = try std.mem.concat(self.a, u8, &.{ rule_name, "-schema" });
            arg_value = try b.plus(try p.toolArgJsonValue(try b.schemaDoc(try b.json(), schema_name, doc, prop.schema, false)), self.arg_close);
        } else if (types.isOnly(.string)) {
            arg_value = self.arg_string;
        } else {
            // grammar keeps only the string rule; the parser tries JSON first to type the value
            var json_value = try b.choiceEmpty();
            if (types.has(.object)) try b.orAssign(&json_value, try b.jsonObject());
            if (types.has(.array)) try b.orAssign(&json_value, try b.jsonArray());
            if (types.has(.number) or types.has(.integer)) try b.orAssign(&json_value, try b.jsonNumber());
            if (types.has(.boolean)) try b.orAssign(&json_value, try b.jsonBool());
            if (types.has(.null)) try b.orAssign(&json_value, try b.jsonNull());
            const typed = try b.atomic(try b.plus(try p.toolArgJsonValue(json_value), self.arg_close));
            arg_value = try b.gbnf(try b.orr(typed, self.arg_string), "rfl-arg-string");
        }

        return b.rule(rule_name, try p.toolArg(try b.plus(arg_open, arg_value)), false);
    }
};

/// Ports `common_chat_params_init_harmony_xml` (parsers/harmony-xml.cpp:4 @cd959bae9).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    try data.message_delimiters.add(a, .assistant, "<|start|>assistant");
    try data.message_delimiters.add(a, .user, "<|start|>user");
    try data.message_delimiters.add(a, .system, "<|start|>developer");
    try data.message_delimiters.add(a, .system, "<|start|>system");

    data.format = .peg_native;
    data.supports_thinking = true;

    data.thinking_start_tag = "<|channel|>analysis<|message|>";
    try data.thinking_end_tags.append(a, "<|end|>");

    try data.preserved_tokens.appendSlice(a, &.{ "<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>" });
    try data.preserved_tokens.appendSlice(a, &eog_tokens);

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

        if (data.grammar_lazy) {
            try data.grammar_triggers.appendSlice(a, &.{
                .{ .type = .pattern, .value = "^\\s+to$" },
                .{ .type = .pattern, .value = "<\\|start\\|>assistant(\\s+to)" },
            });
        }
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "Harmony XML builds a tool parser with lazy triggers" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try template.Template.init(std.testing.allocator, "{% for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endfor %}", "", "");
    defer t.deinit();
    var p: GenerationParams = .{};
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"},\"y\":{\"type\":\"string\"}}}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(@as(usize, 2), data.grammar_triggers.items.len);
    try std.testing.expect(data.grammar_lazy);
    try std.testing.expect(data.grammar.len != 0);
}
