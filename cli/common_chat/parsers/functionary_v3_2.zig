//! Functionary v3.2's recipient format: `>>>all\n{content}` for text and
//! `>>>name\n{json args}` for a tool call.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/functionary-v3-2.cpp` (llama.cpp v1.0.0,
//! `b1b770d90`). The parser is built in the order clang evaluates the C++
//! expressions, left to right, since a parser's id is its position.

const std = @import("std");
const common_chat = @import("../module.zig");
const json = @import("json");
const peg = @import("peg");

const Build = struct {
    a: std.mem.Allocator,
    inputs: *const common_chat.template.GenerationParams,
    has_tools: bool,
    cb: *common_chat.chat_peg.ChatBuilder = undefined,
    tool_choice: peg.ParserId = undefined,

    fn parser(self: *Build, p: *common_chat.chat_peg.ChatBuilder) peg.Error!peg.ParserId {
        self.cb = p;
        const b = &p.b;
        // Functionary v3.2 format:
        // - Normal content: >>>all\n{content}
        // - Tool calls: >>>function_name\n{json_args}
        // Generation prompt ends with ">>>" so model outputs recipient immediately

        // Build content parser for >>>all\n{content}
        // When tools are present, content stops before the next ">>>" (tool call)
        // When no tools, content goes until end
        const content_until_tool = try b.plus(try b.literal("all\n"), try p.content(try b.until(">>>")));
        const content_until_end = try b.plus(try b.literal("all\n"), try p.content(try b.rest()));
        const generation_prompt = try b.literal("<|start_header_id|>assistant<|end_header_id|>\n\n>>>");

        // If no tools or tool_choice is NONE, just parse content
        if (!self.has_tools or self.inputs.tool_choice == .none) {
            // When no tools, just match the prefix and capture everything after
            return b.plus(try b.plus(generation_prompt, content_until_end), try b.end());
        }

        // Build tool call parsers for each available function
        self.tool_choice = try b.choiceEmpty();
        common_chat.parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return common_chat.util.toPeg(e);
        const tool_choice = self.tool_choice;

        const content_only = content_until_end;
        const tools_only = try b.triggerRule("tools", try b.oneOrMore(tool_choice));
        const content_and_tools = try b.plus(content_until_tool, tools_only);

        var ret = try b.eps();
        if (self.inputs.tool_choice == .required) {
            if (self.inputs.parallel_tool_calls) {
                ret = try b.plus(try b.choice(&.{ content_and_tools, tools_only }), try b.end());
            } else {
                ret = try b.plus(try b.choice(&.{ try b.plus(content_until_tool, tool_choice), tools_only }), try b.end());
            }
        } else if (self.inputs.parallel_tool_calls) {
            ret = try b.plus(try b.choice(&.{ content_and_tools, content_only, tools_only }), try b.end());
        } else {
            const content_and_tool = try b.plus(content_until_tool, tool_choice);
            ret = try b.plus(try b.choice(&.{ content_and_tool, content_only, tool_choice }), try b.end());
        }
        return b.plus(generation_prompt, ret);
    }

    fn tool(self: *Build, tool_index: usize, t: *json.Value) common_chat.exception.Error!void {
        self.toolPeg(tool_index, t) catch |e| return common_chat.util.fromPeg(e);
    }

    fn toolPeg(self: *Build, tool_index: usize, t: *json.Value) peg.Error!void {
        const p = self.cb;
        const b = &p.b;
        const function = t.at("function") catch |e| return common_chat.util.toPeg(common_chat.util.fromJson(e));
        const name = common_chat.parsers.functionName(t) catch |e| return common_chat.util.toPeg(e);
        const schema = common_chat.msg.toolParameters(self.a, function) catch |e| return common_chat.util.toPeg(e);

        // Tool format: >>>function_name\n{json_args}
        const open = try p.toolOpen(try b.plus(try p.toolName(try b.literal(name)), try b.literal("\n")));
        const schema_name = try common_chat.parsers.indexName(self.a, "tool-", tool_index, "-schema");
        const tool_parser = try p.tool(try b.plus(open, try p.toolArgs(try b.schemaJson(try b.json(), schema_name, &schema, false))));

        try b.orAssign(&self.tool_choice, try b.rule(try common_chat.parsers.indexName(self.a, "tool-", tool_index, ""), tool_parser, false));
    }
};

/// Ports `common_chat_params_init_functionary_v3_2` (parsers/functionary-v3-2.cpp:4 @b1b770d90).
///
/// Return: owned by the arena `a`.
pub fn init(a: std.mem.Allocator, tmpl: *const common_chat.template.Template, inputs: *const common_chat.template.GenerationParams) common_chat.exception.Error!common_chat.params.ChatParams {
    var data: common_chat.params.ChatParams = .{};

    data.prompt = try common_chat.template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try common_chat.template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_native;
    try data.preserved_tokens.append(a, ">>>all");

    const has_tools = common_chat.parsers.hasTools(&inputs.tools);
    const include_grammar = has_tools and inputs.tool_choice != .none;

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        data.generation_prompt = try std.mem.concat(a, u8, &.{ "<|start_header_id|>assistant<|end_header_id|>\n\n>>>all\n", try m.renderContent(a, "\n\n") });
        try data.prompt.append(a, data.generation_prompt, false);
    }

    var build: Build = .{ .a = a, .inputs = inputs, .has_tools = has_tools };
    data.parser = common_chat.chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return common_chat.util.fromPeg(e);
    const arena = &data.parser.?;

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == .auto;
        data.grammar = try common_chat.parsers.grammar(a, arena, data.grammar_lazy);

        // Grammar trigger for when the model starts outputting a tool call
        // (after the initial ">>>" in the generation prompt but recipient other than "all")
        try data.grammar_triggers.append(a, .{ .type = .pattern, .value = ">>>(?!all)" });
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "meetkai-functionary-medium-v3.2.jinja applies with a tool and builds a parser and a grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "llama.cpp/models/templates/meetkai-functionary-medium-v3.2.jinja", a, .limited(1 << 20));
    var t = try common_chat.template.Template.init(std.testing.allocator, src, "<s>", "</s>");
    defer t.deinit();
    var p: common_chat.template.GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(.peg_native, data.format);
    try std.testing.expect(data.prompt.parts.items.len > 0);
    try std.testing.expect(data.parser != null);
    try std.testing.expect(data.grammar.len > 0);
}
