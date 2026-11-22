//! Ministral 3 and Magistral Large 3: `[THINK]...[/THINK]` reasoning and
//! `[TOOL_CALLS]name[ARGS]{...}` tool calls, over messages rebuilt into
//! the typed blocks the template expects.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/ministral3.cpp` (llama.cpp v1.0.0,
//! `b1b770d90`). The parser is built in the order clang evaluates the C++
//! expressions, left to right, since a parser's id is its position.

const std = @import("std");
const common_chat = @import("../module.zig");
const json = @import("json");
const peg = @import("peg");

const Allocator = std.mem.Allocator;

const Build = struct {
    a: Allocator,
    inputs: *const common_chat.template.GenerationParams,
    has_tools: bool,
    has_response_format: bool,
    extract_reasoning: bool,
    include_grammar: bool = true,
    cb: *common_chat.chat_peg.ChatBuilder = undefined,
    tool_choice: peg.ParserId = undefined,

    fn parser(self: *Build, p: *common_chat.chat_peg.ChatBuilder) peg.Error!peg.ParserId {
        self.cb = p;
        const b = &p.b;
        const generation_prompt = try b.eps();
        const reasoning = if (self.extract_reasoning)
            try b.optional(try b.plusLit(try b.litPlus("[THINK]", try p.reasoning(try b.until("[/THINK]"))), "[/THINK]"))
        else
            try b.eps();

        // Response format parser
        if (self.has_response_format) {
            // Ministral wants to emit json surrounded by code fences
            var r = try b.shlLit(reasoning, "```json");
            r = try b.shl(r, try p.content(try b.schemaJson(try b.json(), "response-format", &self.inputs.json_schema, false)));
            r = try b.shlLit(r, "```");
            return b.plus(generation_prompt, r);
        }

        // Tool call parser
        if (self.has_tools and self.inputs.tool_choice != .none) {
            self.tool_choice = try b.choiceEmpty();
            common_chat.parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return common_chat.util.toPeg(e);

            const min_calls: c_int = if (self.inputs.tool_choice == .required) 1 else 0;
            const max_calls: c_int = if (self.inputs.parallel_tool_calls) -1 else 1;
            const tool_calls = try b.triggerRule("tool-call", try b.repeat(try b.litPlus("[TOOL_CALLS]", self.tool_choice), min_calls, max_calls));

            return b.plus(generation_prompt, try b.shl(try b.shl(reasoning, try p.content(try b.until("[TOOL_CALLS]"))), tool_calls));
        }

        // Content only parser
        self.include_grammar = false;
        return b.plus(generation_prompt, try b.shl(reasoning, try p.content(try b.rest())));
    }

    fn tool(self: *Build, tool_index: usize, t: *json.Value) common_chat.exception.Error!void {
        const function = t.at("function") catch |e| return common_chat.util.fromJson(e);
        const name = try common_chat.parsers.functionName(t);
        const schema = try common_chat.msg.toolParameters(self.a, function);
        const rule = self.buildTool(tool_index, name, &schema) catch |e| return common_chat.util.fromPeg(e);
        self.cb.b.orAssign(&self.tool_choice, rule) catch |e| return common_chat.util.fromPeg(e);
    }

    fn buildTool(self: *Build, tool_index: usize, name: []const u8, schema: *const json.Value) peg.Error!peg.ParserId {
        const p = self.cb;
        const b = &p.b;
        const open = try p.toolOpen(try b.plusLit(try p.toolName(try b.literal(name)), "[ARGS]"));
        const schema_name = try common_chat.parsers.indexName(self.a, "tool-", tool_index, "-schema");
        const body = try b.plus(open, try p.toolArgs(try b.schemaJson(try b.json(), schema_name, schema, false)));
        return b.rule(try common_chat.parsers.indexName(self.a, "tool-", tool_index, ""), body, false);
    }
};

/// The `{"type": ..., <key>: <text>}` block pushed onto the rebuilt content.
fn block(a: Allocator, ty: []const u8, key: []const u8, text: []const u8) common_chat.exception.Error!json.Value {
    var v = json.Value.emptyObject();
    try v.set(a, "type", try json.Value.from(a, ty));
    try v.set(a, key, try json.Value.from(a, text));
    return v;
}

/// The messages rebuilt as the template expects: a system or assistant
/// message's reasoning and content as typed blocks.
fn adjustMessages(a: Allocator, messages: *const json.Value) common_chat.exception.Error!json.Value {
    var adjusted_messages = json.Value.emptyArray();
    var it = @constCast(messages).items();
    while (it.next()) |item| {
        const msg = item.value;
        const role = msg.value("role", []const u8, "") catch |e| return common_chat.util.fromJson(e);
        if (!std.mem.eql(u8, role, "system") and !std.mem.eql(u8, role, "assistant")) {
            // Only adjust system and assistant messages. Interestingly, the system message may contain thinking.
            adjusted_messages.pushBack(a, try msg.clone(a)) catch |e| return common_chat.util.fromJson(e);
            continue;
        }

        var content = json.Value.emptyArray();

        // If message contains `reasoning_content`, add it as a block of type `thinking`
        if (msg.contains("reasoning_content")) {
            const rc = msg.at("reasoning_content") catch |err| switch (err) {
                error.OutOfMemory, error.JsonError => unreachable,
            };
            if (rc.isString()) {
                content.pushBack(a, try block(a, "thinking", "thinking", rc.string)) catch |e| return common_chat.util.fromJson(e);
            }
        }

        // If message contains `content`, add it as a block of type `text`
        if (msg.contains("content")) {
            const c = msg.at("content") catch |err| switch (err) {
                error.OutOfMemory, error.JsonError => unreachable,
            };
            if (c.isString()) {
                content.pushBack(a, try block(a, "text", "text", c.string)) catch |e| return common_chat.util.fromJson(e);
            } else if (c.isArray()) {
                const blocks = try c.clone(a);
                content.insert(a, &blocks) catch |e| return common_chat.util.fromJson(e);
            }
        }

        var adjusted = try msg.clone(a);
        const slot = adjusted.index(a, "content") catch |e| return common_chat.util.fromJson(e);
        slot.assign(a, content);
        adjusted.erase(a, "reasoning_content") catch |e| return common_chat.util.fromJson(e);
        adjusted_messages.pushBack(a, adjusted) catch |e| return common_chat.util.fromJson(e);
    }
    return adjusted_messages;
}

/// Ports `common_chat_params_init_ministral_3` (parsers/ministral3.cpp:3 @b1b770d90).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const common_chat.template.Template, inputs: *const common_chat.template.GenerationParams) common_chat.exception.Error!common_chat.params.ChatParams {
    var data: common_chat.params.ChatParams = .{};

    // Build up messages to follow the format: https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512/blob/main/chat_template.jinja
    const adjusted_messages = try adjustMessages(a, &inputs.messages);

    const has_tools = common_chat.parsers.hasTools(&inputs.tools);
    const has_response_format = common_chat.parsers.hasResponseFormat(&inputs.json_schema);
    const extract_reasoning = inputs.reasoning_format != .none;

    data.supports_thinking = true;
    data.thinking_start_tag = "[THINK]";
    try data.thinking_end_tags.append(a, "[/THINK]");
    data.prompt = try common_chat.template.directApplyImpl(a, tmpl, inputs, &adjusted_messages, null, null);
    data.generation_prompt = try common_chat.template.generationPromptImpl(a, tmpl, inputs, &adjusted_messages, null, null);
    data.format = .peg_native;
    try data.preserved_tokens.appendSlice(a, &.{ "[THINK]", "[/THINK]", "[TOOL_CALLS]", "[ARGS]" });

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;
        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, "[THINK]");
        try gp.appendSlice(a, m.reasoning_content);
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, "[/THINK]");
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
    data.parser = common_chat.chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return common_chat.util.fromPeg(e);
    const arena = &data.parser.?;

    if (build.include_grammar) {
        data.grammar_lazy = !has_response_format and has_tools and inputs.tool_choice == .auto;
        data.grammar = try common_chat.parsers.grammar(a, arena, data.grammar_lazy);
        try data.addWordTrigger(a, "[TOOL_CALLS]");
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "Ministral 3 rebuilds assistant content as blocks" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var msgs = try json.parse(a, "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"x\",\"reasoning_content\":\"r\"}]");
    const adj = try adjustMessages(a, &msgs);
    const out = try json.dump(a, &adj, -1);
    try std.testing.expectEqualStrings("[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"r\"},{\"type\":\"text\",\"text\":\"x\"}]}]", out);
}
