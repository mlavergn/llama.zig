//! MiniMax-M3: `<mm:think>` reasoning and XML tool calls whose every tag
//! carries the namespace token `]<]minimax[>[`, argument values expanded
//! recursively in XML as the template's `to_xml()` writes them.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/minimax-m3.cpp` (llama.cpp v0.5.0,
//! `7fe450e19`). The parser is built in the order clang evaluates the C++
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
const Node = json.schema.Node;
const mm3 = chat_peg.Mapper;

// M3 prefixes every tool tag with the namespace token "]<]minimax[>[";
// params use the parameter name as the tag (<file_path>...</file_path>).
const ns = "]<]minimax[>[";
const think_start = "<mm:think>";
const think_end = "</mm:think>";
const fc_start = ns ++ "<tool_call>";
const fc_end = ns ++ "</tool_call>";
const invoke_end = ns ++ "</invoke>";

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    gen_prompt: []const u8,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,
    cb: *ChatBuilder = undefined,
    tool_choice: ParserId = undefined,
    doc: *peg.SharedDoc = undefined,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const generation_prompt = try p.prefix(self.gen_prompt, think_start);
        const end = try b.end();

        var reasoning = try b.eps();
        if (self.extract_reasoning) {
            const block = if (self.inputs.enable_thinking)
                try b.plus(try b.plus(try b.literal(think_start), try b.space()), try b.acOne(try b.plus(try p.reasoning(try b.until(think_end)), try b.literal(think_end)), think_end))
            else
                try b.plus(try b.literal(think_start), try b.acOne(try b.plus(try b.until(think_end), try b.literal(think_end)), think_end));

            // A turn without reasoning is prefixed with a bare </mm:think>, written either by the
            // generation prompt (thinking_mode = "disabled") or by the model itself.
            reasoning = try b.optional(try b.choice(&.{ block, try b.literal(think_end) }));
        }

        if (self.has_response_format) {
            var rf = try b.plus(try b.literal("```json"), try b.space());
            rf = try b.plus(rf, try p.content(try b.schemaJson(try b.json(), "response-format-schema", &self.inputs.json_schema, false)));
            rf = try b.plus(rf, try b.space());
            rf = try b.plus(rf, try b.literal("```"));
            const response_format = try b.rule("response-format", rf, false);
            return b.plus(try b.plus(try b.plus(generation_prompt, reasoning), response_format), end);
        }

        if (!self.has_tools or self.inputs.tool_choice == .none) {
            return b.plus(try b.plus(try b.plus(generation_prompt, reasoning), try p.content(try b.rest())), end);
        }

        self.tool_choice = try b.choiceEmpty();
        parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

        const require_tools = self.inputs.tool_choice == .required;

        var tool_calls = try b.eps();
        if (self.inputs.parallel_tool_calls) {
            var body = try b.plus(try b.plus(try b.literal(fc_start), try b.space()), self.tool_choice);
            body = try b.plus(body, try b.zeroOrMore(try b.plus(try b.space(), self.tool_choice)));
            body = try b.plus(body, try b.space());
            body = try b.plus(body, try b.literal(fc_end));
            tool_calls = try b.triggerRule("tool-call", body);
        } else {
            var body = try b.plus(try b.plus(try b.literal(fc_start), try b.space()), self.tool_choice);
            body = try b.plus(body, try b.space());
            body = try b.plus(body, try b.literal(fc_end));
            tool_calls = try b.triggerRule("tool-call", body);
        }

        if (!require_tools) {
            tool_calls = try b.optional(tool_calls);
        }

        const content_before_tools = try p.content(try b.until(fc_start));
        return b.plus(try b.plus(try b.plus(try b.plus(generation_prompt, reasoning), content_before_tools), tool_calls), end);
    }

    fn tool(self: *Build, t: *Value) Error!void {
        const function = t.at("function") catch |e| return util.fromJson(e);
        const name = try parsers.functionName(t);
        const params = try msg_mod.toolParameters(self.a, function);
        self.doc = peg.SharedDoc.fromJson(self.a, &params) catch |e| return util.fromPeg(e);
        defer self.doc.release();
        const rule = self.buildTool(name) catch |e| return util.fromPeg(e);
        self.cb.b.orAssign(&self.tool_choice, rule) catch |e| return util.fromPeg(e);
    }

    fn buildTool(self: *Build, name: []const u8) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        var invoke_body = try b.eps();
        switch (self.doc.doc.root.*) {
            .object => |*o| invoke_body = try self.membersOf(o, try self.cat(&.{ "tool-", name, "-arg" })),
            else => {},
        }

        const open = try p.toolOpen(try b.plus(try b.plus(try b.literal(ns ++ "<invoke name=\""), try p.toolName(try b.literal(name))), try b.literal("\">")));
        var func = try b.plus(open, try b.space());
        func = try b.plus(func, invoke_body);
        func = try b.plus(func, try b.space());
        func = try b.plus(func, try p.toolClose(try b.literal(invoke_end)));
        const func_parser = try p.tool(func);

        return b.rule(try self.cat(&.{ "tool-", name }), func_parser, false);
    }

    fn cat(self: *Build, parts: []const []const u8) chat_peg.Error![]u8 {
        return std.mem.concat(self.a, u8, parts);
    }

    /// `element_of`: one `<tag>value</tag>` element under `rule_name`.
    fn elementOf(self: *Build, tag: []const u8, schema: *const Node, rule_name: []const u8) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        const close = try self.cat(&.{ ns, "</", tag, ">" });
        const open = try p.toolArgOpen(try b.plus(try b.plus(try b.literal(ns ++ "<"), try p.toolArgName(try b.literal(tag))), try b.literal(">")));
        return b.rule(rule_name, try p.toolArg(try b.plus(open, try self.valueOf(schema, rule_name, close))), false);
    }

    /// `value_of`: a value of `schema` up to its closing tag -- a string
    /// as raw text, a union by its alternatives, an object and an array
    /// expanded in XML, anything else as JSON.
    fn valueOf(self: *Build, schema: *const Node, rule_name: []const u8, close: []const u8) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        const close_tag = try p.toolArgClose(try b.literal(close));

        // A string accepts anything, so a union with a string alternative is a string
        if (schema.mayBeString()) {
            return b.acOne(try b.plus(try p.toolArgStringValue(try b.until(close)), close_tag), close);
        }

        switch (schema.*) {
            .any_of => |children| {
                var choices: std.ArrayList(ParserId) = .empty;
                for (children, 0..) |alternative, index| {
                    const alt_name = try std.fmt.allocPrint(self.a, "{s}-{d}", .{ rule_name, index });

                    // There is a risk that this breaks streaming deltas, but that's a risk we
                    // assume to provide tool arg streaming.
                    try choices.append(self.a, try self.valueOf(alternative, alt_name, close));
                }
                return b.choice(choices.items);
            },
            .object => |*object| if (object.properties.len != 0) {
                return b.plus(try b.plus(try b.tag(mm3.TOOL_ARG_OBJECT, try self.membersOf(object, rule_name)), try b.space()), close_tag);
            },
            .array => |*arr| {
                const item_close = ns ++ "</item>";
                const item_name = try self.cat(&.{ rule_name, "-item" });
                const item = try b.rule(item_name, try b.tag(mm3.TOOL_ARG_ITEM, try b.plus(try b.literal(ns ++ "<item>"), try self.valueOf(arr.items, item_name, item_close))), false);
                return b.plus(try b.plus(try b.tag(mm3.TOOL_ARG_ARRAY, try b.repeat(try b.plus(try b.space(), item), 0, -1)), try b.space()), close_tag);
            },
            else => {},
        }

        return b.plus(try p.toolArgJsonValue(try b.schemaDoc(try b.json(), try self.cat(&.{ rule_name, "-schema" }), self.doc, schema, false)), close_tag);
    }

    /// `members_of`: required properties in schema order, then any number
    /// of optional ones in any order.
    fn membersOf(self: *Build, object: *const json.schema.Object, rule_prefix: []const u8) chat_peg.Error!ParserId {
        const b = &self.cb.b;
        var required_elements: std.ArrayList(ParserId) = .empty;
        var optional_elements: std.ArrayList(ParserId) = .empty;
        for (object.properties) |prop| {
            const element = try self.elementOf(prop.name, prop.schema, try self.cat(&.{ rule_prefix, "-", prop.name }));
            try (if (prop.required) &required_elements else &optional_elements).append(self.a, element);
        }

        var members = try b.eps();
        for (required_elements.items, 0..) |element, i| {
            if (i > 0) members = try b.plus(members, try b.space());
            members = try b.plus(members, element);
        }

        if (optional_elements.items.len != 0) {
            var any_optional = try b.choiceEmpty();
            for (optional_elements.items) |element| try b.orAssign(&any_optional, element);
            members = try b.plus(members, try b.repeat(try b.plus(try b.space(), any_optional), 0, -1));
        }

        return members;
    }
};

/// Ports `common_chat_params_init_minimax_m3` (parsers/minimax-m3.cpp:3 @7fe450e19).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_minimax_m3;
    data.supports_thinking = true;
    data.thinking_start_tag = think_start;
    try data.thinking_end_tags.append(a, think_end);

    try data.preserved_tokens.appendSlice(a, &.{ ns, "<tool_call>", "</tool_call>", think_start, think_end });

    try data.message_delimiters.add(a, .assistant, "]~b]ai");
    try data.message_delimiters.add(a, .user, "]~b]user");
    try data.message_delimiters.add(a, .tool, "]~b]tool");
    try data.message_delimiters.add(a, .system, "]~b]developer");
    try data.message_delimiters.add(a, .system, "]~b]system");

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = !inputs.json_schema.isNull() and inputs.json_schema.isObject();
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);

    const gen_prompt = data.generation_prompt;

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, gen_prompt);
        try gp.appendSlice(a, think_start);
        try gp.appendSlice(a, m.reasoning_content);
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, think_end);
            try gp.appendSlice(a, try m.renderContent(a, "\n\n"));
        }
        data.generation_prompt = gp.items;
        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
    }

    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .gen_prompt = gen_prompt,
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
        try data.addWordTrigger(a, fc_start);
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "MiniMax-M3 expands nested arguments" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try template.Template.init(std.testing.allocator, "{% for m in messages %}]~b]{{ m.role }}\n{{ m.content }}{% endfor %}{% if add_generation_prompt %}]~b]ai\n{% endif %}", "", "");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a,
        \\[{"type":"function","function":{"name":"f","parameters":{"type":"object","required":["o"],"properties":{
        \\"o":{"type":"object","properties":{"k":{"type":"integer"}}},"l":{"type":"array","items":{"type":"number"}},
        \\"u":{"anyOf":[{"type":"integer"},{"type":"boolean"}]},"s":{"type":"string"}}}}}]
    );
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(.peg_minimax_m3, data.format);
    try std.testing.expect(std.mem.indexOf(u8, data.parser, "tool-f-arg-l-item") != null);
    try std.testing.expect(std.mem.indexOf(u8, data.parser, "tool-f-arg-u-1") != null);
}
