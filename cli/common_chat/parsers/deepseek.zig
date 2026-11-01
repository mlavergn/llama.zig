//! DeepSeek V3.2 and V4's DSML tool calls:
//! `<｜DSML｜function_calls>` (V4: `tool_calls`) around
//! `<｜DSML｜invoke name="f">` with `<｜DSML｜parameter name="k" string="true">v</｜DSML｜parameter>`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/deepseek.cpp` (llama.cpp v0.4.1,
//! `b29c606e2`). The parser is built in the order clang evaluates the C++
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

const DSML = "｜DSML｜";
const THINK_START = "<think>";
const THINK_END = "</think>";
const INVOKE_START = "<" ++ DSML ++ "invoke";
const INVOKE_END = "</" ++ DSML ++ "invoke>";
const PARAM_START = "<" ++ DSML ++ "parameter";
const PARAM_END = "</" ++ DSML ++ "parameter>";
const GEN_PROMPT = "<｜Assistant｜>";
const TC_SEPARATOR = "\n\n";

/// One entry of `call_order`, a `std::map<std::string, size_t>`.
const CallOrder = struct { id: []const u8, idx: usize };

/// `m.value("role", "")`, which throws a type error for a role that is not
/// a string.
fn role(m: *const Value) Error![]const u8 {
    return m.value("role", []const u8, "") catch |e| util.fromJson(e);
}

/// Ports `deepseek_v4_sort_tool_results` (parsers/deepseek.cpp:6 @b29c606e2): each
/// run of user and tool messages with its tool results put in the order of
/// the preceding assistant message's tool calls, matched by id.
///
/// Return: a copy owned by `a`.
fn deepseekV4SortToolResults(a: Allocator, messages: *const Value) Error!Value {
    var adjusted = try messages.clone(a);
    var call_order: std.ArrayList(CallOrder) = .empty;

    const size = adjusted.size();
    var i: usize = 0;
    while (i < size) {
        const msg = adjusted.atIndex(i) catch |e| return util.fromJson(e);
        const r = try role(msg);

        if (std.mem.eql(u8, r, "assistant") and msg.contains("tool_calls")) {
            const tool_calls = msg.at("tool_calls") catch unreachable;
            if (tool_calls.isArray() and !tool_calls.empty()) {
                call_order.clearRetainingCapacity();
                for (tool_calls.array.items, 0..) |*tc, idx| {
                    const id = tc.value("id", []const u8, "") catch |e| return util.fromJson(e);
                    if (id.len != 0) {
                        for (call_order.items) |*e| {
                            if (std.mem.eql(u8, e.id, id)) {
                                e.idx = idx;
                                break;
                            }
                        } else try call_order.append(a, .{ .id = id, .idx = idx });
                    }
                }
                i += 1;
                continue;
            }
        }

        if (!std.mem.eql(u8, r, "user") and !std.mem.eql(u8, r, "tool")) {
            i += 1;
            continue;
        }

        // collect a maximal run of user/tool messages - they render into one user block
        var tool_positions: std.ArrayList(usize) = .empty;
        var run_end = i;
        while (run_end < size) : (run_end += 1) {
            const rr = try role(adjusted.atIndex(run_end) catch |e| return util.fromJson(e));
            if (std.mem.eql(u8, rr, "tool")) {
                try tool_positions.append(a, run_end);
            } else if (!std.mem.eql(u8, rr, "user")) {
                break;
            }
        }

        if (tool_positions.items.len > 1 and call_order.items.len != 0) {
            const Keyed = struct { order: usize, v: Value };
            const results = try a.alloc(Keyed, tool_positions.items.len);
            for (tool_positions.items, results) |pos, *res| {
                const m = adjusted.atIndex(pos) catch unreachable;
                const id = m.value("tool_call_id", []const u8, "") catch |e| return util.fromJson(e);
                var order: usize = 0;
                for (call_order.items) |e| {
                    if (std.mem.eql(u8, e.id, id)) {
                        order = e.idx;
                        break;
                    }
                }
                res.* = .{ .order = order, .v = m.* };
            }
            // std::stable_sort: equal keys keep their order
            std.sort.insertion(Keyed, results, {}, struct {
                fn lt(_: void, x: Keyed, y: Keyed) bool {
                    return x.order < y.order;
                }
            }.lt);
            for (tool_positions.items, results) |pos, res| {
                (adjusted.atIndex(pos) catch unreachable).* = res.v;
            }
        }

        i = run_end;
    }

    return adjusted;
}

const Build = struct {
    a: Allocator,
    inputs: *const GenerationParams,
    is_v4: bool,
    fc_start: []const u8,
    fc_end: []const u8,
    has_response_format: bool,
    extract_reasoning: bool,
    require_tools: bool,
    has_tool_calls: bool,
    cb: *ChatBuilder = undefined,
    tool_choice: ParserId = undefined,
    name: []const u8 = "",
    required_parsers: std.ArrayList(ParserId) = .empty,
    optional_parsers: std.ArrayList(ParserId) = .empty,

    fn parser(self: *Build, p: *ChatBuilder) chat_peg.Error!ParserId {
        self.cb = p;
        const b = &p.b;
        const generation_prompt = try b.literal(GEN_PROMPT);
        const end = try b.end();

        // build tool call section first since we might need it in reasoning
        self.tool_choice = try b.choiceEmpty();
        if (self.has_tool_calls) {
            parsers.foreachFunction(&self.inputs.tools, self, tool) catch |e| return util.toPeg(e);
        }
        const tool_choice = self.tool_choice;

        var tool_calls = try b.eps();
        if (self.inputs.parallel_tool_calls) {
            var x = try b.plus(try b.plus(try b.literal(self.fc_start), try b.space()), tool_choice);
            x = try b.plus(x, try b.zeroOrMore(try b.plus(try b.space(), tool_choice)));
            x = try b.plus(try b.plus(x, try b.space()), try b.literal(self.fc_end));
            tool_calls = try b.triggerRule("tool-call", x);
        } else {
            var x = try b.plus(try b.plus(try b.literal(self.fc_start), try b.space()), tool_choice);
            x = try b.plus(try b.plus(x, try b.space()), try b.literal(self.fc_end));
            tool_calls = try b.triggerRule("tool-call", x);
        }

        var reasoning = try b.eps();
        var reasoning_with_tc = try b.eps();
        const obligatory_tool_calls = tool_calls;
        var allow_reasoning_with_tc = false;

        if (!self.require_tools) {
            tool_calls = try b.optional(tool_calls);
        }

        if (self.extract_reasoning and self.inputs.enable_thinking) {
            reasoning = try b.optional(try b.plusLit(try b.litPlus(THINK_START, try p.reasoning(try b.until(THINK_END))), THINK_END));
            const sep_fc = try std.mem.concat(self.a, u8, &.{ TC_SEPARATOR, self.fc_start });
            const r = try b.litPlus(THINK_START, try p.reasoning(try b.untilOneOf(&.{ sep_fc, self.fc_start, THINK_END })));
            reasoning_with_tc = try b.plus(try b.plus(r, try b.space()), obligatory_tool_calls);
            allow_reasoning_with_tc = true;
        } else if (self.extract_reasoning) {
            // Thinking disabled but reasoning extraction requested: the generation prompt
            // contains an empty <think></think> pair (V3.2) or a bare </think> (V4) that
            // must still be consumed.
            reasoning = if (self.is_v4)
                try b.optional(try b.literal(THINK_END))
            else
                try b.optional(try b.plus(try b.plus(try b.literal(THINK_START), try b.until(THINK_END)), try b.literal(THINK_END)));
        }

        if (self.has_response_format) {
            var rf = try b.plus(try b.literal("```json"), try b.space());
            rf = try b.plus(rf, try p.content(try b.schemaJson(try b.json(), "response-format-schema", &self.inputs.json_schema, false)));
            rf = try b.plus(try b.plus(rf, try b.space()), try b.literal("```"));
            const response_format = try b.rule("response-format", rf, false);
            return b.plus(try b.plus(try b.plus(generation_prompt, reasoning), response_format), end);
        }

        if (!self.has_tool_calls) {
            return b.plus(try b.plus(try b.plus(generation_prompt, reasoning), try p.content(try b.rest())), end);
        }

        const sep_fc = try std.mem.concat(self.a, u8, &.{ TC_SEPARATOR, self.fc_start });
        const content_before_tools = try b.plus(try b.plus(try b.negate(try b.literal(THINK_START)), try p.content(try b.untilOneOf(&.{ sep_fc, self.fc_start }))), try b.space());
        if (allow_reasoning_with_tc) {
            const alt = try b.plus(try b.plus(reasoning, content_before_tools), tool_calls);
            return b.plus(try b.plus(generation_prompt, try b.orr(reasoning_with_tc, alt)), end);
        }
        return b.plus(try b.plus(try b.plus(try b.plus(generation_prompt, reasoning), content_before_tools), tool_calls), end);
    }

    fn tool(self: *Build, t: *Value) Error!void {
        const function = t.at("function") catch |e| return util.fromJson(e);
        self.name = try parsers.functionName(t);
        self.required_parsers = .empty;
        self.optional_parsers = .empty;
        try parsers.foreachParameter(self.a, function, self, param);
        self.finishTool() catch |e| return util.fromPeg(e);
    }

    fn param(self: *Build, prm: *const json.schema.Property, doc: *peg.SharedDoc) Error!void {
        const named_arg = self.buildParam(prm, doc) catch |e| return util.fromPeg(e);
        if (prm.required) {
            try self.required_parsers.append(self.a, named_arg);
        } else {
            try self.optional_parsers.append(self.a, named_arg);
        }
    }

    fn buildParam(self: *Build, prm: *const json.schema.Property, doc: *peg.SharedDoc) chat_peg.Error!ParserId {
        const p = self.cb;
        const b = &p.b;
        const is_string = prm.schema.mayBeString();

        const open_lit = try b.literal(PARAM_START ++ " name=\"");
        const open_name = try b.plus(open_lit, try p.toolArgName(try b.literal(prm.name)));
        const open = try p.toolArgOpen(try b.plus(open_name, try b.literal(if (is_string) "\" string=\"true\">" else "\" string=\"false\">")));
        const value = if (is_string)
            try p.toolArgStringValue(try b.until(PARAM_END))
        else
            try p.toolArgJsonValue(try b.schemaDoc(try b.json(), try std.mem.concat(self.a, u8, &.{ "tool-", self.name, "-arg-", prm.name, "-schema" }), doc, prm.schema, false));
        const arg = try p.toolArg(try b.plus(try b.plus(open, value), try p.toolArgClose(try b.literal(PARAM_END))));

        return b.rule(try std.mem.concat(self.a, u8, &.{ "tool-", self.name, "-arg-", prm.name }), arg, false);
    }

    fn finishTool(self: *Build) chat_peg.Error!void {
        const p = self.cb;
        const b = &p.b;
        var args_seq = try b.eps();
        for (self.required_parsers.items, 0..) |rp, i| {
            if (i > 0) {
                args_seq = try b.plus(args_seq, try b.space());
            }
            args_seq = try b.plus(args_seq, rp);
        }

        if (self.optional_parsers.items.len != 0) {
            var any_opt = try b.choiceEmpty();
            for (self.optional_parsers.items) |opt| {
                try b.orAssign(&any_opt, opt);
            }
            args_seq = try b.plus(args_seq, try b.repeat(try b.plus(try b.space(), any_opt), 0, -1));
        }

        const invoke_body = args_seq;
        const head = try b.plus(try b.literal(INVOKE_START ++ " name=\""), try p.toolName(try b.literal(self.name)));
        const open = try p.toolOpen(try b.plus(head, try b.literal("\">\n")));
        const body = try b.plus(try b.plus(try b.plus(open, invoke_body), try b.space()), try p.toolClose(try b.literal(INVOKE_END)));
        const func_parser = try p.tool(body);

        try b.orAssign(&self.tool_choice, try b.rule(try std.mem.concat(self.a, u8, &.{ "tool-", self.name }), func_parser, false));
    }
};

/// Ports `common_chat_params_init_deepseek_v3_2` (parsers/deepseek.cpp:69 @b29c606e2).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    // V4 uses the same DSML markup as V3.2, but names the tool call block "tool_calls"
    // instead of "function_calls", renders tool results in tool call order and its
    // non-thinking generation prompt ends with a bare </think> instead of an empty
    // <think></think> pair.
    const is_v4 = !util.contains(tmpl.source(), "function_calls");

    var adjusted_storage: Value = .null;
    var adjusted_messages: ?*const Value = null;
    if (is_v4) {
        adjusted_storage = try deepseekV4SortToolResults(a, &inputs.messages);
        adjusted_messages = &adjusted_storage;
    }

    const has_tools = parsers.hasTools(&inputs.tools);
    const has_response_format = !inputs.json_schema.isNull() and inputs.json_schema.isObject();
    const extract_reasoning = inputs.reasoning_format != .none;
    const include_grammar = has_response_format or (has_tools and inputs.tool_choice != .none);

    var additional_storage: Value = .null;
    var additional_context: ?*const Value = null;
    if (is_v4 and has_response_format) {
        additional_storage = Value.emptyObject();
        try additional_storage.set(a, "response_format", try inputs.json_schema.clone(a));
        additional_context = &additional_storage;
    }

    const tc_block = if (is_v4) "tool_calls" else "function_calls";
    const fc_start = try std.mem.concat(a, u8, &.{ "<", DSML, tc_block, ">" });
    const fc_end = try std.mem.concat(a, u8, &.{ "</", DSML, tc_block, ">" });

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, adjusted_messages, null, additional_context);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, adjusted_messages, null, additional_context);
    data.format = .peg_native;
    data.supports_thinking = true;
    data.thinking_start_tag = THINK_START;
    try data.thinking_end_tags.appendSlice(a, &.{ THINK_END, fc_start });
    try data.preserved_tokens.appendSlice(a, &.{ DSML, THINK_START, THINK_END });

    if (inputs.hasContinuation()) {
        const m = &inputs.continue_msg;

        if (is_v4 and m.reasoning_content.len == 0) {
            data.generation_prompt = GEN_PROMPT ++ THINK_END;
            if (inputs.continue_final_message == .content) {
                data.generation_prompt = try std.mem.concat(a, u8, &.{ data.generation_prompt, try m.renderContent(a, "\n\n") });
            }
        } else {
            data.generation_prompt = try std.mem.concat(a, u8, &.{ GEN_PROMPT, THINK_START, m.reasoning_content });
            if (inputs.continue_final_message == .content) {
                data.generation_prompt = try std.mem.concat(a, u8, &.{ data.generation_prompt, THINK_END, try m.renderContent(a, "\n\n") });
            }
        }

        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
    }

    const require_tools = inputs.tool_choice == .required;
    const has_tool_calls = has_tools and inputs.tool_choice != .none;

    var build: Build = .{
        .a = a,
        .inputs = inputs,
        .is_v4 = is_v4,
        .fc_start = fc_start,
        .fc_end = fc_end,
        .has_response_format = has_response_format,
        .extract_reasoning = extract_reasoning,
        .require_tools = require_tools,
        .has_tool_calls = has_tool_calls,
    };
    var arena = chat_peg.buildChatPegParser(a, &build, Build.parser) catch |e| return util.fromPeg(e);
    defer arena.deinit();

    data.parser = try parsers.save(a, &arena);

    if (include_grammar) {
        data.grammar_lazy = has_tools and !require_tools;
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

test "deepseek-ai-DeepSeek-V3.2.jinja applies with a tool and builds a parser and a grammar" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "llama.cpp/models/templates/deepseek-ai-DeepSeek-V3.2.jinja", a, .limited(1 << 20));
    var t = try template.Template.init(std.testing.allocator, src, "<s>", "</s>");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    p.tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]");
    const data = try init(a, &t, &p);
    try std.testing.expectEqual(.peg_native, data.format);
    try std.testing.expect(data.prompt.len > 0);
    try std.testing.expect(data.parser.len > 0);
    try std.testing.expect(data.grammar.len > 0);
}
