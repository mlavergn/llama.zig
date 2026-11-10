//! Muse Spark: Muse Glimmer's recipient protocol with a commentary recipient
//! besides -- `<|start|>assistant to=<recipient><|message|>{content}{END}`,
//! `to=self` for chain of thought and `to=commentary` for a user-facing aside,
//! both ending `<|eom|>`, `to=user` for the answer ending `<|eot|>`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/muse-spark.cpp` (llama.cpp v0.9.0,
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

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    extract_reasoning: bool,
    include_grammar: bool,
    cb: *ChatBuilder = undefined,
    string_value: ParserId = undefined,
    tool_choice: ParserId = undefined,
    name: []const u8 = "",
    tool_index: usize = 0,
    arg_rules: std.ArrayList(ParserId) = .empty,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const start = try b.rule("start", try b.literal("<|start|>assistant"), false);

        if (!self.extract_reasoning and !self.include_grammar) {
            return b.plus(start, try p.content(try b.rest()));
        }

        if (self.extract_reasoning) {
            _ = try b.rule("analysis", try b.plus(try b.plus(try b.literal(" to=self<|message|>"), try p.reasoning(try b.until("<|eom|>"))), try b.literal("<|eom|>")), false);
        } else {
            _ = try b.rule("analysis", try b.plus(try b.plus(try b.literal(" to=self<|message|>"), try p.content(try b.until("<|eom|>"))), try b.literal("<|eom|>")), false);
        }
        const analysis = try b.ref("analysis");

        // Commentary is user facing content emitted mid turn, so it reads as
        // content and always ends on <|eom|>.
        const preamble = try b.rule("preamble", try b.plus(try b.plus(try b.literal(" to=commentary<|message|>"), try p.content(try b.until("<|eom|>"))), try b.literal("<|eom|>")), false);

        const aside = try b.rule("aside", try b.orr(analysis, preamble), false);

        const recipient = try b.optional(try b.literal(" to=user"));
        const final_msg = try b.rule("final", try b.plus(try b.plus(recipient, try b.literal("<|message|>")), try p.content(try b.untilOneOf(&.{ "<|eot|>", "<|eom|>" }))), false);

        if (self.has_tools and self.inputs.tool_choice != .none) {
            self.string_value = try b.acOne(try b.plus(try p.toolArgStringValue(try b.until("</atem:parameter>")), try p.toolArgClose(try b.literal("</atem:parameter>"))), "</atem:parameter>");

            self.tool_choice = try b.choiceEmpty();
            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

            const tool_calls = if (self.inputs.parallel_tool_calls)
                try b.triggerRule("tool-call", try b.plus(self.tool_choice, try b.zeroOrMore(try b.plus(try b.plus(try b.literal("<|eom|>"), start), self.tool_choice))))
            else
                try b.triggerRule("tool-call", self.tool_choice);

            if (self.inputs.tool_choice == .required) {
                return b.plus(try b.plus(try b.zeroOrMore(try b.plus(start, aside)), start), tool_calls);
            }
            const trailing_calls = try b.optional(try b.plus(try b.plus(try b.literal("<|eom|>"), start), tool_calls));
            const head = try b.plus(try b.zeroOrMore(try b.plus(start, aside)), start);
            return b.plus(head, try b.orr(tool_calls, try b.plus(final_msg, trailing_calls)));
        }

        return b.plus(try b.plus(try b.zeroOrMore(try b.plus(start, aside)), start), final_msg);
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
        var open = try b.plus(try b.literal(" to="), try b.until("<|message|>"));
        open = try b.plus(open, try b.literal("<|message|><atem:function_calls>"));
        open = try b.plus(open, try b.space());
        open = try b.plus(open, try b.literal("<atem:invoke name=\""));
        open = try b.plus(open, try p.toolName(try b.literal(self.name)));
        open = try b.plus(open, try b.literal("\">"));
        open = try b.plus(open, try b.space());
        const head = try b.shl(try p.toolOpen(open), try p.toolArgs(args));
        const close = try b.plus(try b.plus(try b.literal("</atem:invoke>"), try b.space()), try b.literal("</atem:function_calls>"));
        return p.tool(try b.shl(head, try p.toolClose(close)));
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
            value_parser = try b.plus(try p.toolArgJsonValue(try b.schemaDoc(try b.json(), schema_name, doc, prop.schema, false)), try p.toolArgClose(try b.literal("</atem:parameter>")));
        }
        var open = try b.plus(try b.literal("<atem:parameter name=\""), try p.toolArgName(try b.literal(prop.name)));
        open = try b.plus(open, try b.literal("\">"));
        return p.toolArg(try b.plus(try p.toolArgOpen(open), value_parser));
    }
};

/// Ports `common_chat_params_init_muse_spark` (parsers/muse-spark.cpp:13 @cd959bae9).
/// The generation prompt is just `<|start|>assistant`; the model emits its
/// own ` to=...<|message|>`.
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = "<|start|>assistant";
    data.format = .peg_native;
    data.supports_thinking = true;

    try data.preserved_tokens.appendSlice(a, &.{
        "<|start|>",             "<|message|>",            "<|eom|>",         "<|eot|>",
        // ATEM tool-call markup emitted on " to=<tool>" turns.
        "<atem:function_calls>", "<atem:invoke",           "<atem:parameter", "</atem:parameter>",
        "</atem:invoke>",        "</atem:function_calls>",
    });

    try data.message_delimiters.add(a, .assistant, "<|start|>assistant");
    try data.message_delimiters.add(a, .user, "<|start|>user");
    try data.message_delimiters.add(a, .system, "<|start|>system");
    try data.message_delimiters.add(a, .tool, "<|start|>tool");

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, "<|start|>assistant to=self<|message|>");
        try gp.appendSlice(a, m.reasoning_content);
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, "<|eom|><|start|>assistant to=user<|message|>");
            try gp.appendSlice(a, try m.renderContent(a, "\n\n"));
        }
        data.generation_prompt = gp.items;
        try data.prompt.append(a, data.generation_prompt, false);
    }

    const extract_reasoning = inputs.reasoning_format != .none;

    const has_tools = parsers.hasTools(&inputs.tools);
    // Constrained grammar whenever tools are offered.
    const include_grammar = has_tools and inputs.tool_choice != .none;

    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .has_tools = has_tools,
        .extract_reasoning = extract_reasoning,
        .include_grammar = include_grammar,
    };
    data.parser = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    const arena = &data.parser.?;

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice != .required;
        data.grammar = try parsers.grammar(a, arena, data.grammar_lazy);
        try data.grammar_triggers.append(a, .{
            .type = .pattern,
            .value = "<\\|start\\|>assistant( to=(?!self<\\|message\\|>)(?!commentary<\\|message\\|>)(?!user<\\|message\\|>)[^<]*?<\\|message\\|>)",
        });
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "Muse Spark builds a tool parser with a lazy pattern trigger" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try template.Template.init(std.testing.allocator, "{% for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|eot|>{% endfor %}", "", "");
    defer t.deinit();
    var p: GenerationParams = .{};
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"},\"y\":{\"type\":\"string\"}}}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(params_mod.TriggerType.pattern, data.grammar_triggers.items[0].type);
    try std.testing.expect(data.grammar_lazy);
    try std.testing.expect(data.grammar.len != 0);
}
