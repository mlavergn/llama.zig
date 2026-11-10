//! GigaChat 3's tool calls: one JSON call after a `function call` role
//! header, `<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": ..., "arguments": {...}}`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/gigachat-v3.cpp` (llama.cpp v0.9.0,
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
const msg_mod = @import("../msg.zig");
const parsers = @import("module.zig");

const Allocator = std.mem.Allocator;
const Error = exception.Error;
const ChatParams = params_mod.ChatParams;
const ChatBuilder = chat_peg.ChatBuilder;
const ParserId = peg.ParserId;
const GenerationParams = template.GenerationParams;

const tool_call_start_prefix = "<|message_sep|>\n\nfunction call<|role_sep|>\n";

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    has_tools: bool,
    include_grammar: bool,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        const b = &p.b;
        var ret = try b.eps();
        if (self.has_tools and self.inputs.tool_choice != .none) {
            // Build a choice of all available tools
            var tool_choice = try b.choiceEmpty();
            for (0..self.inputs.tools.size()) |i| {
                const tool = self.inputs.tools.atIndex(i) catch |e| return util.toPeg(util.fromJson(e));
                const function = tool.at("function") catch |e| return util.toPeg(util.fromJson(e));
                const name = parsers.functionName(tool) catch |e| return util.toPeg(e);
                const schema = msg_mod.toolParameters(self.a, function) catch |e| return util.toPeg(e);

                const quoted_name = try b.plusLit(try b.litPlus("\"", try p.toolName(try b.literal(name))), "\"");
                const tool_name = try b.jsonMember("name", quoted_name);
                const schema_name = try parsers.indexName(self.a, "tool-", i, "-schema");
                const tool_args = try b.jsonMember("arguments", try p.toolArgs(try b.schemaJson(try b.json(), schema_name, &schema, false)));

                const tool_open = try p.toolOpen(try b.shl(try b.literal("{"), tool_name));

                const body = try b.shlLit(try b.shl(try b.shlLit(tool_open, ","), tool_args), "}");
                try b.orAssign(&tool_choice, try b.rule(try parsers.indexName(self.a, "tool-", i, ""), body, false));
            }

            // Define the tool call structure
            const min_calls: c_int = if (self.inputs.tool_choice == .required) 1 else 0;
            const max_calls: c_int = 1; // parallel toolcalls are not supported
            const tool_call = try b.rule("tool-call", try b.plus(try b.literal(tool_call_start_prefix), tool_choice), false);
            const tool_calls = try b.triggerRule("tool-call-root", try b.repeat(tool_call, min_calls, max_calls));

            ret = try b.shl(try p.content(try b.until("<|message_sep|>\n\n")), tool_calls);
        } else {
            // Content only parser
            self.include_grammar = false;
            ret = try p.content(try b.rest());
        }

        return b.plus(try b.literal("assistant<|role_sep|>\n"), ret);
    }
};

/// Ports `common_chat_params_init_gigachat_v3` (parsers/gigachat-v3.cpp:3 @cd959bae9).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_native;
    data.supports_thinking = false;
    try data.preserved_tokens.appendSlice(a, &.{ "<|message_sep|>\n\n", "<|role_sep|>\n" });

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        data.generation_prompt = try std.mem.concat(a, u8, &.{ "assistant<|role_sep|>\n", try m.renderContent(a, "\n\n") });
        try data.prompt.append(a, data.generation_prompt, false);
    }

    const has_tools = parsers.hasTools(&inputs.tools);
    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .has_tools = has_tools,
        .include_grammar = has_tools and inputs.tool_choice != .none,
    };

    data.parser = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    const arena = &data.parser.?;

    if (build.include_grammar) {
        data.grammar_lazy = has_tools and inputs.tool_choice == .auto;
        data.grammar = try parsers.grammar(a, arena, data.grammar_lazy);
        try data.addWordTrigger(a, tool_call_start_prefix);
    }
    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "GigaChat3-10B-A1.8B.jinja applies with a tool and builds a parser and a grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "llama.cpp/models/templates/GigaChat3-10B-A1.8B.jinja", a, .limited(1 << 20));
    var t = try template.Template.init(std.testing.allocator, src, "<s>", "</s>");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(.peg_native, data.format);
    try std.testing.expect(data.prompt.parts.items.len > 0);
    try std.testing.expect(data.parser != null);
    try std.testing.expect(data.grammar.len > 0);
}
