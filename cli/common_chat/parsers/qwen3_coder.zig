//! Qwen3-Coder's XML tool calls, shared by Nemotron Nano 3, Qwen3.5 and
//! StepFun-3.5-Flash: `<tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/qwen3-coder.cpp` (llama.cpp v0.8.0,
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

const gen_prefix = "<|im_start|>assistant\n";

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    supports_reasoning: bool,
    extract_reasoning: bool,
    has_response_format: bool,
    has_tools: bool,
    is_qwen3_coder: bool,
    tool_call_starts: []const []const u8,

    // per tool, while its parameters are visited
    cb: *ChatBuilder = undefined,
    name: []const u8 = "",
    tool_index: usize = 0,
    arg_close: ParserId = undefined,
    arg_string: ParserId = undefined,
    required_args: std.ArrayList(ParserId) = .empty,
    optional_args: std.ArrayList(ParserId) = .empty,
    tool_choice: ParserId = undefined,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const generation_prompt = try b.literal(gen_prefix);

        var reasoning = try b.eps();
        if (self.supports_reasoning and self.extract_reasoning) {
            var r = try b.litPlus("<think>", try b.space());
            // The template strips the newline before </think>, so leave it out of the reasoning
            r = try b.plus(r, try p.reasoning(try b.untilOneOf(&.{ "\n</think>", "</think>", "<tool_call>" })));
            r = try b.plus(r, try b.orr(try b.orr(try b.literal("\n</think>"), try b.literal("</think>")), try b.peek(try b.literal("<tool_call>"))));
            reasoning = try b.optional(r);
        }

        // Response format parser
        if (self.has_response_format) {
            const content = try p.content(try b.schemaJson(try b.json(), "response-format", &self.inputs.json_schema, false));
            return b.plus(generation_prompt, try b.shl(reasoning, content));
        }

        // Tool call parser
        if (self.has_tools and self.inputs.tool_choice != .none) {
            self.arg_close = try p.toolArgClose(try b.literal("\n</parameter>\n"));
            self.arg_string = try b.rule("xml-arg-string", try b.acOne(try b.plus(try p.toolArgStringValue(try b.until("\n</parameter>\n")), self.arg_close), "\n</parameter>\n"), false);

            self.tool_choice = try b.choiceEmpty();
            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);

            const min_calls: c_int = if (self.inputs.tool_choice == .required) 1 else 0;

            const tool_call_body = try b.plus(try b.plusLit(self.tool_choice, "</tool_call>"), try b.space());
            const tool_call = try b.rule("tool-call", try b.litPlus("<tool_call>\n", tool_call_body), false);

            // Qwen3-Coder models may occasionally omit the <tool_call> token.
            const tool_call_first = if (self.is_qwen3_coder)
                try b.rule("tool-call-first", try b.plus(try b.optional(try b.literal("<tool_call>\n")), tool_call_body), false)
            else
                tool_call;

            const calls = if (self.inputs.parallel_tool_calls) try b.plus(tool_call_first, try b.zeroOrMore(tool_call)) else tool_call_first;
            const tool_calls = try b.triggerRule("tool-call-root", try b.repeat(calls, min_calls, 1));

            // The templates separate content from the tool calls with one or two newlines,
            // stop the content before them
            var content_ends: std.ArrayList([]const u8) = .empty;
            for (self.tool_call_starts) |start| {
                try content_ends.append(self.a, try std.mem.concat(self.a, u8, &.{ "\n\n", start }));
                try content_ends.append(self.a, try std.mem.concat(self.a, u8, &.{ "\n", start }));
                try content_ends.append(self.a, start);
            }

            const content = try p.content(try b.untilOneOf(content_ends.items));
            return b.plus(generation_prompt, try b.shl(try b.shl(reasoning, content), tool_calls));
        }

        // Content only parser
        return b.plus(generation_prompt, try b.shl(reasoning, try p.content(try b.rest())));
    }

    fn tool(self: *Build, tool_index: usize, t: *Value) Error!void {
        self.tool_index = tool_index;
        const p = self.cb;
        const b = &p.b;
        const function = t.at("function") catch |e| return util.fromJson(e);
        self.name = try parsers.functionName(t);
        self.required_args = .empty;
        self.optional_args = .empty;

        try parsers.foreachParameter(self.a, function, self, param);

        // Accept required arguments in any order, as Qwen does not always adhere to the
        // order provided.
        var args = p.permute(try parsers.indexName(self.a, "tool-", self.tool_index, "-args"), self.required_args.items) catch |e| return util.fromPeg(e);
        if (self.optional_args.items.len != 0) {
            args = b.plus(args, b.zeroOrMore(b.choice(self.optional_args.items) catch |e| return util.fromPeg(e)) catch |e| return util.fromPeg(e)) catch |e| return util.fromPeg(e);
        }

        const func = buildFunc(p, self.name, args) catch |e| return util.fromPeg(e);
        const rule = b.rule(try parsers.indexName(self.a, "tool-", self.tool_index, ""), func, false) catch |e| return util.fromPeg(e);
        b.orAssign(&self.tool_choice, rule) catch |e| return util.fromPeg(e);
    }

    fn buildFunc(p: *ChatBuilder, name: []const u8, args: ParserId) chat_peg.Error!ParserId {
        const b = &p.b;
        const open = try p.toolOpen(try b.plusLit(try b.litPlus("<function=", try p.toolName(try b.literal(name))), ">\n"));
        const with_args = try b.plus(open, try p.toolArgs(args));
        return p.tool(try b.plus(with_args, try p.toolClose(try b.literal("</function>\n"))));
    }

    fn param(self: *Build, param_index: usize, prm: *const json.schema.Property, doc: *peg.SharedDoc) Error!void {
        const r = self.buildParam(param_index, prm, doc) catch |e| return util.fromPeg(e);
        try (if (prm.required) &self.required_args else &self.optional_args).append(self.a, r);
    }

    fn buildParam(self: *Build, param_index: usize, prm: *const json.schema.Property, doc: *peg.SharedDoc) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        const rule_name = try parsers.argName(self.a, self.tool_index, param_index, "");

        const arg_open = try p.toolArgOpen(try b.plusLit(try b.litPlus("<parameter=", try p.toolArgName(try b.literal(prm.name))), ">\n"));

        const types = prm.schema.valueTypes() catch |e| return peg.errors.fromSchema(e);

        var arg_value = try b.eps();
        if (!types.has(.string)) {
            const schema_name = try std.mem.concat(self.a, u8, &.{ rule_name, "-schema" });
            arg_value = try b.plus(try p.toolArgJsonValue(try b.schemaDoc(try b.json(), schema_name, doc, prm.schema, false)), self.arg_close);
        } else if (types.isOnly(.string)) {
            arg_value = self.arg_string;
        } else {
            // The string alternative accepts any text, so the grammar only keeps the raw string
            // rule. The parser still tries the JSON alternatives first to type the value.
            var json_value = try b.choiceEmpty();
            if (types.has(.object)) try b.orAssign(&json_value, try b.jsonObject());
            if (types.has(.array)) try b.orAssign(&json_value, try b.jsonArray());
            if (types.has(.number) or types.has(.integer)) try b.orAssign(&json_value, try b.jsonNumber());
            if (types.has(.boolean)) try b.orAssign(&json_value, try b.jsonBool());
            if (types.has(.null)) try b.orAssign(&json_value, try b.jsonNull());
            const typed = try b.atomic(try b.plus(try p.toolArgJsonValue(json_value), self.arg_close));
            arg_value = try b.gbnf(try b.orr(typed, self.arg_string), "xml-arg-string");
        }

        return b.rule(rule_name, try p.toolArg(try b.plus(arg_open, arg_value)), false);
    }
};

const Starts = struct {
    a: Allocator,
    list: std.ArrayList([]const u8),

    fn add(self: *Starts, _: usize, t: *Value) Error!void {
        const name = try parsers.functionName(t);
        try self.list.append(self.a, try std.mem.concat(self.a, u8, &.{ "<function=", name, ">" }));
    }
};

/// Ports `common_chat_params_init_qwen3_coder` (qwen3-coder.cpp:3 @8642541b2).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, null, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, null, null, null);
    data.format = .peg_native;

    const supports_reasoning = util.contains(tmpl.source(), "<think>");

    data.supports_thinking = supports_reasoning;
    try data.preserved_tokens.appendSlice(a, &.{ "<tool_call>", "</tool_call>" });

    const is_qwen3_coder = !supports_reasoning;

    if (supports_reasoning) {
        data.thinking_start_tag = "<think>";
        // Support both </think> and <tool_call> as reasoning end sequences.
        // The newline variant comes first so it is included in the forced message
        // <function= is omitted, as it is a workaround for Qwen3-Coder which is not a thinking model
        try data.thinking_end_tags.appendSlice(a, &.{ "\n</think>", "</think>", "<tool_call>" });
        try data.preserved_tokens.appendSlice(a, &.{ "<think>", "</think>" });
    }

    try data.message_delimiters.add(a, .assistant, "<|im_start|>assistant");
    try data.message_delimiters.add(a, .tool, "<|im_start|>user\n<tool_response>"); // Qwen3-Coder, Qwen3.5, Nemotron Nano 3
    try data.message_delimiters.add(a, .tool, "<|im_start|>tool_response"); // StepFun-3.5-Flash
    try data.message_delimiters.add(a, .user, "<|im_start|>user");
    try data.message_delimiters.add(a, .system, "<|im_start|>system");

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = parsers.hasResponseFormat(&inputs.json_schema);
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;

        var gp: std.ArrayList(u8) = .empty;
        try gp.appendSlice(a, gen_prefix);
        if (supports_reasoning) {
            try gp.appendSlice(a, "<think>\n");
            try gp.appendSlice(a, m.reasoning_content);
            if (inputs.continue_final_message == .content) {
                try gp.appendSlice(a, "\n</think>\n\n");
            }
        }
        if (inputs.continue_final_message == .content) {
            try gp.appendSlice(a, try m.renderContent(a, "\n\n"));
        }
        data.generation_prompt = gp.items;
        try data.prompt.append(a, data.generation_prompt, false);
    }

    var starts: Starts = .{ .a = a, .list = .empty };
    try starts.list.append(a, "<tool_call>");

    if (is_qwen3_coder) {
        // Match complete <function=name> opener for Qwen3-Coder models that occasionally omit the
        // starting <tool_call>. The model may hallucinate a tool name, but it is preferable over
        // constraining on <function which may occur in valid content generation, e.g. #include <functional>
        try parsers.foreachFunction(&inputs.tools, &starts, Starts.add);
    }

    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .supports_reasoning = supports_reasoning,
        .extract_reasoning = extract_reasoning,
        .has_response_format = has_response_format,
        .has_tools = has_tools,
        .is_qwen3_coder = is_qwen3_coder,
        .tool_call_starts = starts.list.items,
    };
    data.parser = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    const arena = &data.parser.?;

    if (include_grammar) {
        data.grammar_lazy = has_tools and inputs.tool_choice == .auto;
        data.grammar = try parsers.grammar(a, arena, data.grammar_lazy);
        if (data.grammar_lazy) {
            for (starts.list.items) |start| try data.addWordTrigger(a, start);
        }
    }

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
