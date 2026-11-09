//! MiniCPM5: optional `<think>` reasoning and XML tool calls,
//! `<function name="foo"><param name="bar">value</param></function>`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/minicpm5.cpp` (llama.cpp v0.8.0,
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

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,
    cb: *ChatBuilder = undefined,
    string_value: ParserId = undefined,
    tool_choice: ParserId = undefined,
    name: []const u8 = "",
    tool_index: usize = 0,
    arg_rules: std.ArrayList(ParserId) = .empty,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const generation_prompt = try b.literal("<|im_start|>assistant\n");

        var reasoning = try b.eps();
        if (self.extract_reasoning) {
            reasoning = try b.plus(try b.shlLit(try b.litShl("<think>", try p.reasoning(try b.until("</think>"))), "</think>"), try b.space());
        }

        // Response format parser
        if (self.has_response_format) {
            const head = try b.plus(generation_prompt, reasoning);
            return b.plus(head, try p.content(try b.schemaJson(try b.json(), "response-format", &self.inputs.json_schema, false)));
        }

        if (self.has_tools and self.inputs.tool_choice != .none) {
            // CDATA lets a value carry characters that would otherwise close the tag (e.g.
            // </param>); capture the inner text only, excluding the CDATA markers.
            var cdata = try b.literal("<![CDATA[");
            cdata = try b.plus(cdata, try b.acOne(try b.plus(try p.toolArgStringValue(try b.until("]]>")), try b.literal("]]>")), "]]>"));
            cdata = try b.plus(cdata, try p.toolArgClose(try b.literal("</param>")));
            const plain = try b.plus(try b.negate(try b.literal("<![CDATA[")), try b.acOne(try b.plus(try p.toolArgStringValue(try b.until("</param>")), try p.toolArgClose(try b.literal("</param>"))), "</param>"));
            self.string_value = try b.choice(&.{ cdata, plain });

            self.tool_choice = try b.choiceEmpty();
            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

            const max_calls: c_int = if (self.inputs.parallel_tool_calls) -1 else 1;
            const tool_calls = try b.triggerRule("tool-call", try b.repeat(try b.plus(self.tool_choice, try b.space()), 1, max_calls));

            const content = try p.content(try b.until("<function"));

            return b.plus(try b.plus(try b.plus(try b.plus(generation_prompt, reasoning), content), tool_calls), try b.end());
        }

        return b.plus(try b.plus(try b.plus(generation_prompt, reasoning), try p.content(try b.rest())), try b.end());
    }

    fn tool(self: *Build, tool_index: usize, t: *Value) Error!void {
        self.tool_index = tool_index;
        const function = t.at("function") catch |e| return util.fromJson(e);
        self.name = try parsers.functionName(t);
        self.arg_rules = .empty;
        try parsers.foreachParameter(self.a, function, self, param);
        const tool_parser = self.buildTool() catch |e| return util.fromPeg(e);
        const b = &self.cb.b;
        const rule = b.rule(try parsers.indexName(self.a, "tool-", self.tool_index, ""), tool_parser, false) catch |e| return util.fromPeg(e);
        b.orAssign(&self.tool_choice, rule) catch |e| return util.fromPeg(e);
    }

    fn buildTool(self: *Build) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        var args = try b.eps();
        if (self.arg_rules.items.len != 0) {
            args = try b.zeroOrMore(try b.plus(try b.choice(self.arg_rules.items), try b.space()));
        }
        var open = try b.plus(try b.literal("<function name=\""), try p.toolName(try b.literal(self.name)));
        open = try b.plus(open, try b.literal("\">"));
        const head = try b.shl(try p.toolOpen(open), try p.toolArgs(args));
        return p.tool(try b.shl(head, try p.toolClose(try b.literal("</function>"))));
    }

    fn param(self: *Build, param_index: usize, prop: *const json.schema.Property, doc: *peg.SharedDoc) Error!void {
        const r = self.buildParam(param_index, prop, doc) catch |e| return util.fromPeg(e);
        try self.arg_rules.append(self.a, r);
    }

    fn buildParam(self: *Build, param_index: usize, prop: *const json.schema.Property, doc: *peg.SharedDoc) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        var value_parser = try b.eps();
        if (prop.schema.mayBeString() catch |e| return peg.errors.fromSchema(e)) {
            value_parser = self.string_value;
        } else {
            const schema_name = try parsers.argName(self.a, self.tool_index, param_index, "-schema");
            value_parser = try b.plus(try p.toolArgJsonValue(try b.schemaDoc(try b.json(), schema_name, doc, prop.schema, false)), try p.toolArgClose(try b.literal("</param>")));
        }
        var open = try b.plus(try b.literal("<param name=\""), try p.toolArgName(try b.literal(prop.name)));
        open = try b.plus(open, try b.literal("\">"));
        return p.toolArg(try b.plus(try p.toolArgOpen(open), value_parser));
    }
};

/// Ports `common_chat_params_init_minicpm5` (parsers/minicpm5.cpp:6 @cd959bae9).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_native;
    data.supports_thinking = true;
    try data.preserved_tokens.appendSlice(a, &.{ "<function", "<param", "</function>", "</param>", "<think>", "</think>" });

    data.thinking_start_tag = "<think>";
    try data.thinking_end_tags.append(a, "</think>");

    try data.message_delimiters.add(a, .assistant, "<|im_start|>assistant");
    try data.message_delimiters.add(a, .tool, "<|im_start|>user\n<tool_response>");
    try data.message_delimiters.add(a, .user, "<|im_start|>user");
    try data.message_delimiters.add(a, .system, "<|im_start|>system");

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = parsers.hasResponseFormat(&inputs.json_schema);
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, "<|im_start|>assistant\n<think>\n");
        try gp.appendSlice(a, m.reasoning_content);
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, "\n</think>\n\n");
            try gp.appendSlice(a, try m.renderContent(a, "\n\n"));
        }
        data.generation_prompt = gp.items;
        try data.prompt.append(a, data.generation_prompt, false);
    }

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
        try data.addWordTrigger(a, "<function");
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "MiniCPM5 builds a tool parser" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try template.Template.init(std.testing.allocator, "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}", "", "");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"},\"y\":{\"type\":\"string\"}}}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqualStrings("<|im_start|>assistant\n", data.generation_prompt);
    try std.testing.expect(data.grammar.len != 0);
}
