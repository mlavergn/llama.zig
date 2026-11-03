//! Kimi K3's XTML tagged format: `<|open|>t k="v"<|sep|>` ...
//! `<|close|>t<|sep|>`, an assistant turn being an optional think section,
//! a response, a tools section and the message closer.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/kimi-k3.cpp` (llama.cpp v0.6.0,
//! `d81235049`). The parser is built in the order clang evaluates the C++
//! expressions, left to right, since a parser's id is its position.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("../exception.zig");
const msg_mod = @import("../msg.zig");
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

const sep = "<|sep|>";
const msg_start = "<|open|>message role=\"assistant\"<|sep|>";
const think_start = "<|open|>think<|sep|>";
const think_end = "<|close|>think<|sep|>";
const resp_start = "<|open|>response<|sep|>";
const resp_end = "<|close|>response<|sep|>";
const tools_start = "<|open|>tools<|sep|>";
const tools_end = "<|close|>tools<|sep|>";
const call_start = "<|open|>call tool=\"";
const call_end = "<|close|>call<|sep|>";
const arg_start = "<|open|>argument key=\"";
const arg_end = "<|close|>argument<|sep|>";
const msg_end = "<|close|>message<|sep|>";
const eom_token = "<|end_of_msg|>";

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    extract_reasoning: bool,
    cb: *ChatBuilder = undefined,
    tool_choices: ParserId = undefined,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const end = try b.end();

        const start = try b.optional(try b.literal(msg_start));

        // the think section is always consumed, even with reasoning extraction off:
        // the generation prompt ends with open_tag('think'), so it is always present.
        // reasoning stops at its own closer, or at the response opener if the model
        // skips the closer
        const think_body = if (self.extract_reasoning)
            try p.reasoning(try b.untilOneOf(&.{ think_end, resp_start }))
        else
            try p.content(try b.untilOneOf(&.{ think_end, resp_start }));

        const reasoning = try b.optional(try b.plus(try b.plus(try b.optional(try b.literal(think_start)), think_body), try b.optional(try b.literal(think_end))));

        // content runs to the response closer, or to the next section if truncated
        const response = try b.plus(
            try b.plus(try b.optional(try b.literal(resp_start)), try p.content(try b.untilOneOf(&.{ resp_end, tools_start, msg_end }))),
            try b.optional(try b.literal(resp_end)),
        );

        // the EOG token after the message closer reaches the parser as text,
        // so it must be consumed or the parse stays incomplete
        const trailer = try b.plus(try b.optional(try b.literal(msg_end)), try b.optional(try b.literal(eom_token)));

        if (!self.has_tools or self.inputs.tool_choice == .none) {
            return b.plus(try b.plus(try b.plus(try b.plus(start, reasoning), response), trailer), end);
        }

        self.tool_choices = try b.choiceEmpty();
        parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

        // all calls go inside one tools section, then the message is closed. the
        // message closer is part of the trigger rule, or else the lazy grammar
        // rejects it once tool calls have started
        var section = try b.plus(try b.literal(tools_start), try b.oneOrMore(self.tool_choices));
        section = try b.plus(section, try b.literal(tools_end));
        section = try b.plus(section, try b.optional(try b.literal(msg_end)));
        section = try b.plus(section, try b.optional(try b.literal(eom_token)));
        const tools_section = try b.triggerRule("kimi-k3-tool-call", section);

        const tools = if (self.inputs.tool_choice == .required) tools_section else try b.optional(tools_section);

        return b.plus(try b.plus(try b.plus(try b.plus(try b.plus(start, reasoning), response), tools), trailer), end);
    }

    fn tool(self: *Build, t: *Value) Error!void {
        const function = t.at("function") catch |e| return util.fromJson(e);
        const name = try parsers.functionName(t);
        const schema = try msg_mod.toolParameters(self.a, function);
        const call = self.buildTool(name, &schema) catch |e| return util.fromPeg(e);
        const b = &self.cb.b;
        const rule = b.rule(try std.mem.concat(self.a, u8, &.{ "kimi-k3-tool-", name }), call, false) catch |e| return util.fromPeg(e);
        b.orAssign(&self.tool_choices, rule) catch |e| return util.fromPeg(e);
    }

    fn buildTool(self: *Build, name: []const u8, schema: *const Value) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;

        // arguments come one tag per key, with the JSON type in a type="..."
        // attribute. the type is taken from the tool schema instead, as it tells
        // us if the value is JSON or a literal string
        var args = try b.eps();
        if (schema.contains("properties") and !(schema.at("properties") catch |err| switch (err) {
            error.OutOfMemory, error.JsonError => unreachable,
        }).empty()) {
            const props = schema.at("properties") catch |err| switch (err) {
                error.OutOfMemory, error.JsonError => unreachable,
            };
            var arg_choices = try b.choiceEmpty();
            var it = props.items();
            while (it.next()) |prop| {
                const key = try self.a.dupe(u8, prop.key);

                var ty: []const u8 = "string";
                if (prop.value.isObject() and prop.value.contains("type")) {
                    const tv = prop.value.at("type") catch |err| switch (err) {
                        error.OutOfMemory, error.JsonError => unreachable,
                    };
                    if (tv.isString()) ty = tv.string;
                }

                const value = if (std.mem.eql(u8, ty, "string"))
                    try p.toolArgStringValue(try b.until(arg_end))
                else
                    try p.toolArgValue(try b.until(arg_end));

                // skip the trailing type="..." attribute: anything up to <|sep|>
                var body = try b.plus(try p.toolArgOpen(try b.literal(arg_start)), try p.toolArgName(try b.literal(key)));
                body = try b.plus(body, try b.literal("\""));
                body = try b.plus(body, try b.until(sep));
                body = try b.plus(body, try b.literal(sep));
                body = try b.plus(body, value);
                body = try b.plus(body, try p.toolArgClose(try b.literal(arg_end)));
                const rule_name = try std.mem.concat(self.a, u8, &.{ "kimi-k3-arg-", name, "-", key });
                try b.orAssign(&arg_choices, try b.rule(rule_name, try p.toolArg(body), false));
            }
            args = try b.zeroOrMore(arg_choices);
        }

        // skip the trailing index="N" attribute the same way
        var open = try b.plus(try b.literal(call_start), try p.toolName(try b.literal(name)));
        open = try b.plus(open, try b.literal("\""));
        open = try b.plus(open, try b.until(sep));
        open = try b.plus(open, try b.literal(sep));
        const with_args = try b.plus(try p.toolOpen(open), try p.toolArgs(args));
        return p.tool(try b.plus(with_args, try p.toolClose(try b.literal(call_end))));
    }
};

/// Ports `common_chat_params_init_kimi_k3` (parsers/kimi-k3.cpp:8 @d81235049).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_native;
    data.supports_thinking = true;

    // only the markers are special tokens. tag names ("think", "response", ...) are
    // normal tokens and must not be preserved, or prose with those words is broken
    try data.preserved_tokens.appendSlice(a, &.{ "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>" });

    data.thinking_start_tag = think_start;
    try data.thinking_end_tags.append(a, think_end);

    // per-role message-start delimiters. user/assistant messages only have the role
    // attribute, so the full opener is used. system and tool messages have more
    // attributes, so those delimiters stop after the closing quote of the role
    try data.message_delimiters.add(a, .assistant, "<|open|>message role=\"assistant\"<|sep|>");
    try data.message_delimiters.add(a, .user, "<|open|>message role=\"user\"<|sep|>");
    try data.message_delimiters.add(a, .tool, "<|open|>message role=\"tool\"");
    try data.message_delimiters.add(a, .system, "<|open|>message role=\"system\"");

    const has_tools = parsers.hasTools(&inputs.tools);
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_tools and inputs.tool_choice != .none;

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, msg_start ++ think_start);
        try gp.appendSlice(a, m.reasoning_content);
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, think_end ++ resp_start);
            try gp.appendSlice(a, try m.renderContent(a, "\n\n"));
        }
        data.generation_prompt = gp.items;
        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
    }

    var build: Build = .{ .a = a, .inputs = inputs, .has_tools = has_tools, .extract_reasoning = extract_reasoning };
    var arena = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    defer arena.deinit();

    data.parser = try parsers.save(a, &arena);

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice != .required;
        data.grammar = try parsers.grammar(a, &arena, data.grammar_lazy);
        try data.addWordTrigger(a, tools_start);
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "Kimi K3 builds a tool parser" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try template.Template.init(std.testing.allocator, "{% for m in messages %}<|open|>message role=\"{{ m.role }}\"<|sep|>{{ m.content }}<|close|>message<|sep|><|end_of_msg|>{% endfor %}", "", "");
    defer t.deinit();
    var p: GenerationParams = .{};
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"},\"y\":{}}}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expect(data.grammar_lazy);
    try std.testing.expect(data.grammar.len != 0);
}
