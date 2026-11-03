//! Building a PEG parser, and the grammar and triggers that go with it,
//! from what the differential analysis read off a template.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/chat-auto-parser-generator.cpp` (llama.cpp
//! v0.6.0, `d81235049`): `peg_generator`'s `generate_parser`,
//! `autoparser`'s `build_parser` and the three analyzers' `build_parser`
//! methods with their helpers. Each declaration names what it ports and the
//! line it began at.
//!
//! # Order
//!
//! The parser built here is saved into `common_chat_params`'s `parser`, and a
//! parser's id is its position in the arena, so the builders are called in
//! the order clang evaluates the C++ -- left to right, each operand built
//! before the operator joins it -- exactly as `chat_peg.zig` builds its
//! own.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("../exception.zig");
const msg_mod = @import("../msg.zig");
const params_mod = @import("../params.zig");
const template = @import("../template.zig");
const chat_peg = @import("../chat_peg.zig");
const util = @import("../util.zig");
const log = @import("../log.zig");
const parsers = @import("../parsers/module.zig");
const types = @import("types.zig");
const helpers = @import("helpers.zig");
const analyzer = @import("analyzer.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const PegError = peg.Error;
const ParserId = peg.ParserId;
const ChatBuilder = chat_peg.ChatBuilder;
const ChatParams = params_mod.ChatParams;
const GenerationParams = template.GenerationParams;
const Template = template.Template;
const Autoparser = types.Autoparser;
const trimWhitespace = helpers.trimWhitespace;

/// Ports `peg_generator`'s `generate_parser` (chat-auto-parser-generator.cpp:23
/// @d81235049), the overload that analyses the template first.
pub fn generateParser(a: Allocator, tmpl: *const Template, inputs: *const GenerationParams) Error!ChatParams {
    // Run differential analysis to extract template structure
    var ap: Autoparser = .{};
    try analyzer.analyzeTemplate(&ap, a, tmpl);
    return generateParserAnalyzed(a, tmpl, inputs, &ap);
}

/// Ports `peg_generator`'s `generate_parser` (chat-auto-parser-generator.cpp:31
/// @d81235049): the prompt, the generation prompt (rebuilt for a continued
/// message), the parser, and -- for tools or a response format -- the
/// grammar and its lazy trigger.
pub fn generateParserAnalyzed(a: Allocator, tmpl: *const Template, inputs: *const GenerationParams, ap: *const Autoparser) Error!ChatParams {
    // Create the result structure
    var data: ChatParams = .{};
    data.prompt = try template.directApply(a, tmpl, inputs);
    data.generation_prompt = try template.generationPrompt(a, tmpl, inputs);
    data.format = .peg_native;
    try data.preserved_tokens.appendSlice(a, ap.preserved_tokens.items);
    try data.additional_stops.appendSlice(a, ap.additional_stops.items);

    const parser_generation_prompt = data.generation_prompt;

    if (inputs.continue_final_message != .none and !inputs.continue_msg.empty()) {
        // Build up generation prompt manually
        const msg = &inputs.continue_msg;

        if (ap.reasoning.start.len != 0) {
            const gp = data.generation_prompt;
            var next: std.ArrayList(u8) = .empty;
            try next.appendSlice(a, try util.substr(gp, 0, util.find(gp, ap.reasoning.start)));
            try next.appendSlice(a, ap.reasoning.start);
            try next.appendSlice(a, msg.reasoning_content);
            if (inputs.continue_final_message == .content) {
                try next.appendSlice(a, ap.reasoning.end);
            }
            data.generation_prompt = next.items;
        }

        if (inputs.continue_final_message == .content) {
            data.generation_prompt = try std.mem.concat(a, u8, &.{ data.generation_prompt, try msg.renderContent(a, "\n\n") });
        }

        data.prompt = try std.mem.concat(a, u8, &.{ data.prompt, data.generation_prompt });
    }

    var parser = try buildParser(a, ap, inputs, parser_generation_prompt);
    defer parser.deinit();
    data.parser = try parsers.save(a, &parser);

    // Build grammar if tools are present
    const has_tools = ap.tools.format.mode != .NONE and inputs.tools.isArray() and !inputs.tools.empty();
    const trigger_marker = if (ap.tools.format.section_start.len != 0) ap.tools.format.section_start else ap.tools.format.per_call_start;

    const has_response_format = !inputs.json_schema.empty() and inputs.json_schema.isObject();
    const include_grammar = has_response_format or (has_tools and
        ((inputs.tool_choice == .auto and trigger_marker.len != 0) or
            inputs.tool_choice == .required));

    if (include_grammar) {
        data.grammar_lazy = !has_response_format and inputs.tool_choice == .auto;
        data.grammar = try parsers.grammar(a, &parser, data.grammar_lazy);

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {
            data.grammar_triggers = .empty;
            try data.addWordTrigger(a, trigger_marker);
            if (ap.tools.format.openai_wrapper_trigger) {
                // model emits the OpenAI function wrapper, trigger on it
                try data.addWordTrigger(a, "{\"type\": \"function\",");
            }
        }
    }

    return data;
}

/// Ports `parser_build_context` (chat-auto-parser.h:226 @d81235049) and its
/// constructor (chat-auto-parser-generator.cpp:18 @d81235049), which builds
/// the `eps` the reasoning parser starts as.
const Ctx = struct {
    p: *ChatBuilder,
    inputs: *const GenerationParams,
    reasoning_parser: ParserId,
    extracting_reasoning: bool = false,
    reasoning: ?*const types.AnalyzeReasoning = null,
    content: ?*const types.AnalyzeContent = null,
    a: Allocator,

    fn init(p: *ChatBuilder, inputs: *const GenerationParams, a: Allocator) PegError!Ctx {
        return .{ .p = p, .inputs = inputs, .reasoning_parser = try p.b.eps(), .a = a };
    }
};

/// The state `autoparser`'s `build_parser`'s lambda closes over.
const Root = struct {
    a: Allocator,
    ap: *const Autoparser,
    inputs: *const GenerationParams,
    generation_prompt: []const u8,

    fn build(self: *const Root, p: *ChatBuilder) PegError!ParserId {
        const b = &p.b;
        const ap = self.ap;
        const inputs = self.inputs;
        var ctx = try Ctx.init(p, inputs, self.a);
        const extract_reasoning = inputs.reasoning_format != .none;

        ctx.extracting_reasoning = extract_reasoning and ap.reasoning.mode != .NONE;
        ctx.content = &ap.content;
        ctx.reasoning = &ap.reasoning;

        // Build reasoning parser
        ctx.reasoning_parser = try reasoningBuildParser(&ap.reasoning, &ctx);

        var parser = try b.eps();

        const has_tools = inputs.tools.isArray() and !inputs.tools.empty();
        const has_response_format = inputs.json_schema.isObject() and !inputs.json_schema.empty();
        var pure_content = ap.reasoning.mode == .NONE;

        if (has_response_format) {
            const response_format = try b.rule("response-format", try p.content(try b.schemaJson(try b.json(), "response-format-schema", &inputs.json_schema, false)), false);
            var x = try b.plus(ctx.reasoning_parser, try b.space());
            var fenced = try b.plus(try b.literal("```json"), try b.space());
            fenced = try b.plus(fenced, response_format);
            fenced = try b.plus(fenced, try b.space());
            fenced = try b.plus(fenced, try b.literal("```"));
            var bare = try b.plus(try b.space(), response_format);
            bare = try b.plus(bare, try b.space());
            x = try b.plus(x, try b.choice(&.{ fenced, bare }));
            parser = try b.plus(x, try b.end());
            pure_content = false;
        } else if (has_tools and inputs.tool_choice != .none and ap.jinja_caps.supports_tool_calls) {
            parser = try toolsBuildParser(&ap.tools, &ctx);
            pure_content = false;
        } else {
            parser = try contentBuildParser(&ap.content, &ctx);
        }
        const reasoning_start = trimWhitespace(ap.reasoning.start);
        const prefix = try p.prefix(self.generation_prompt, reasoning_start);
        return if (pure_content) b.plus(prefix, parser) else b.shl(prefix, parser);
    }
};

/// Ports `autoparser`'s `build_parser` (chat-auto-parser-generator.cpp:99
/// @d81235049).
///
/// Return: the arena, its `gpa` `a`; release with `deinit`.
pub fn buildParser(a: Allocator, ap: *const Autoparser, inputs: *const GenerationParams, generation_prompt: []const u8) Error!peg.Arena {
    if (!ap.analysis_complete) {
        return exception.throw(error.InvalidArgument, "Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)", .{});
    }
    const root: Root = .{ .a = a, .ap = ap, .inputs = inputs, .generation_prompt = generation_prompt };
    return chat_peg.buildChatPegParser(a, &root, Root.build) catch |e| util.fromPeg(e);
}

/// Ports `analyze_reasoning`'s `build_parser` (chat-auto-parser-generator.cpp:138
/// @d81235049).
fn reasoningBuildParser(r: *const types.AnalyzeReasoning, ctx: *Ctx) PegError!ParserId {
    const p = ctx.p;
    const b = &p.b;

    if (!ctx.extracting_reasoning) {
        return b.eps();
    }

    if (r.mode == .TAG_BASED or r.mode == .TOOLS_ONLY) {
        if (r.end.len != 0) {
            if (r.start.len != 0) {
                // Standard tag-based: optional(<think>reasoning</think>)
                const x = try b.plus(try p.optspace(r.start), try p.reasoning(try b.until(trimWhitespace(r.end))));
                return b.optional(try b.plus(x, try p.optspace(r.end)));
            }
            // Delimiter-style (empty start)
            return b.optional(try b.plus(try p.reasoning(try b.until(trimWhitespace(r.end))), try p.optspace(r.end)));
        }
    }

    return b.eps();
}

/// Ports `analyze_content`'s `build_parser` (chat-auto-parser-generator.cpp:159
/// @d81235049).
fn contentBuildParser(c: *const types.AnalyzeContent, ctx: *Ctx) PegError!ParserId {
    const p = ctx.p;
    const b = &p.b;

    if (c.isAlwaysWrapped()) {
        if (ctx.extracting_reasoning) {
            var x = try b.plusLit(ctx.reasoning_parser, c.start);
            x = try b.plus(x, try p.content(try b.until(c.end)));
            x = try b.plusLit(x, c.end);
            return b.plus(x, try b.end());
        }
        var x = try b.plusLit(try p.content(try b.until(c.start)), c.start);
        x = try b.plus(x, try p.content(try b.until(c.end)));
        x = try b.plusLit(x, c.end);
        return b.plus(x, try b.end());
    }
    return b.plus(try b.plus(ctx.reasoning_parser, try p.content(try b.rest())), try b.end());
}

/// Ports `analyze_content`'s `build_optional_wrapped` (chat-auto-parser-generator.cpp:171
/// @d81235049).
fn contentBuildOptionalWrapped(c: *const types.AnalyzeContent, ctx: *Ctx) PegError!ParserId {
    const p = ctx.p;
    const b = &p.b;

    if (c.isAlwaysWrapped()) {
        return b.optional(try b.plusLit(try b.litPlus(c.start, try p.content(try b.until(c.end))), c.end));
    }
    return b.eps();
}

/// Ports `analyze_tools`'s `build_parser` (chat-auto-parser-generator.cpp:180
/// @d81235049).
fn toolsBuildParser(t: *const types.AnalyzeTools, ctx: *Ctx) PegError!ParserId {
    switch (t.format.mode) {
        .JSON_NATIVE => return buildToolParserJsonNative(t, ctx),
        .TAG_WITH_JSON => return buildToolParserTagJson(t, ctx),
        .TAG_WITH_TAGGED => return buildToolParserTagTagged(t, ctx),
        else => {
            log.err("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly. " ++
                "Check for a fixed template for your model in the models/templates directory of your llama.cpp installation or " ++
                "report an issue at https://github.com/ggml-org/llama.cpp/issues\n", .{});
            return ctx.p.b.eps();
        },
    }
}

/// Ports `analyze_tools`'s `build_tool_parser_json_native` (chat-auto-parser-generator.cpp:196
/// @d81235049).
fn buildToolParserJsonNative(t: *const types.AnalyzeTools, ctx: *Ctx) PegError!ParserId {
    const p = ctx.p;
    const b = &p.b;
    const inputs = ctx.inputs;
    const format = &t.format;

    // Build effective field names with dot notation if function_field is set
    var name_field = format.name_field;
    var args_field = format.args_field;

    if (format.function_field.len != 0 and !std.mem.eql(u8, format.function_field, "function") and
        std.mem.indexOfScalar(u8, name_field, '.') == null)
    {
        name_field = try std.mem.concat(ctx.a, u8, &.{ format.function_field, ".", name_field });
        args_field = try std.mem.concat(ctx.a, u8, &.{ format.function_field, ".", args_field });
    }

    const opts: ChatBuilder.JsonToolsOptions = .{
        .name_key = name_field,
        .args_key = args_field,
        .array_wrapped = format.tools_array_wrapped,
        .function_is_key = format.fun_name_is_key,
        .call_id_key = format.id_field,
        .gen_call_id_key = format.gen_id_field,
        .parameters_order = format.parameter_order.items,
        .accept_openai_wrapper = format.openai_wrapper_trigger,
    };

    var tools_parser = try b.eps();
    if (format.section_start.len == 0 and format.per_call_start.len != 0) {
        const single_tool_parser = try p.standardJsonTools(format.per_call_start, format.per_call_end, &inputs.tools, inputs.parallel_tool_calls, inputs.tool_choice == .required, opts);
        tools_parser = try b.triggerRule("tool-calls", try b.oneOrMore(try b.plus(single_tool_parser, try b.space())));
    } else {
        tools_parser = try p.standardJsonTools(format.section_start, format.section_end, &inputs.tools, inputs.parallel_tool_calls, inputs.tool_choice == .required, opts);
    }

    // Handle content wrappers if present
    if (ctx.content) |content| {
        if (content.isAlwaysWrapped()) {
            const wrapped_content = try contentBuildOptionalWrapped(content, ctx);
            return b.plus(try b.plus(try b.plus(ctx.reasoning_parser, wrapped_content), tools_parser), try b.end());
        }
    }

    var tool_start: []const u8 = "{";
    if (format.section_start.len != 0) {
        tool_start = format.section_start;
    } else if (format.per_call_start.len != 0) {
        tool_start = format.per_call_start;
    }

    const x = try b.plus(ctx.reasoning_parser, try b.optional(try p.content(try b.until(tool_start))));
    return b.plus(try b.plus(x, tools_parser), try b.end());
}

/// Ports `analyze_tools`'s `build_func_parser` (chat-auto-parser-generator.cpp:240
/// @d81235049): a function's opening with its name, the call id and the
/// arguments, then its closing, atomic where a partial match would
/// mislead.
fn buildFuncParser(t: *const types.AnalyzeTools, p: *ChatBuilder, name: []const u8, call_id_section: ParserId, have_call_id: bool, args: ParserId, atomic_peek: ?ParserId) PegError!ParserId {
    const b = &p.b;
    const function = &t.function;
    var open = try p.toolOpen(try b.plusLit(try b.litPlus(function.name_prefix, try p.toolName(try b.literal(name))), function.name_suffix));
    var matched_atomic = false;
    var func_parser = try b.eps();

    if (function.args_separator.len != 0) {
        open = try b.plus(try b.plus(open, try b.space()), try b.literal(function.args_separator));
    }

    if (function.name_suffix.len != 0) {
        func_parser = try b.plus(try b.plus(try b.plus(open, call_id_section), try b.space()), args);
        matched_atomic = true;
    } else if (have_call_id) {
        func_parser = try b.plus(try b.plus(try b.atomic(try b.plus(open, call_id_section)), try b.space()), args);
        matched_atomic = true;
    } else if (atomic_peek) |peek| {
        const inner = try b.plus(try b.plus(try b.plus(open, call_id_section), try b.space()), peek);
        func_parser = try b.plus(try b.atomic(inner), args);
        matched_atomic = true;
    } else {
        func_parser = try b.plus(try b.plus(try b.plus(open, call_id_section), try b.space()), args);
    }

    if (function.close.len != 0) {
        func_parser = try b.plus(try b.plus(func_parser, try b.space()), try p.toolClose(try b.literal(function.close)));
    } else if (t.format.per_call_end.len != 0) {
        // When there's no func_close but there is a per_call_end marker, use peek() to ensure
        // we only emit tool_close when we can actually see the closing marker. This prevents
        // premature closing during partial parsing when we've seen e.g. "</" which could be
        // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
        // Laguna (v4): the model may emit whitespace between the last </arg_value> and
        // </tool_call> even though the template renders them tight. Tolerate optional
        // leading space in the close lookahead so the tool call still closes.
        const close_peek = if (t.arguments.tolerate_intertag_whitespace)
            try b.peek(try b.plus(try b.space(), try b.literal(t.format.per_call_end)))
        else
            try b.peek(try b.literal(t.format.per_call_end));
        func_parser = try b.plus(func_parser, try p.toolClose(close_peek));
    } else {
        func_parser = try b.plus(func_parser, try p.toolClose(try b.space())); // force this to process tool closing callbacks in mapper
    }
    if (!matched_atomic) {
        func_parser = try b.atomic(func_parser);
    }
    return func_parser;
}

/// The per-tool callback of `build_tool_parser_tag_json`.
const TagJson = struct {
    t: *const types.AnalyzeTools,
    ctx: *Ctx,
    tool_choice: ParserId,

    fn tool(self: *TagJson, tool_def: *Value) Error!void {
        self.build(tool_def) catch |e| return util.fromPeg(e);
    }

    fn build(self: *TagJson, tool_def: *Value) PegError!void {
        const t = self.t;
        const p = self.ctx.p;
        const b = &p.b;
        const a = self.ctx.a;
        const func = tool_def.at("function") catch |e| return peg.errors.fromJson(e);
        const name = parsers.functionName(tool_def) catch |e| return util.toPeg(e);
        const schema = msg_mod.toolParameters(a, func) catch |e| return util.toPeg(e);

        // Build call_id parser based on position (if supported)
        var have_call_id = false;
        var call_id_section = try b.eps();
        if (t.call_id.pos == .BETWEEN_FUNC_AND_ARGS and t.call_id.prefix.len != 0 and
            (t.call_id.suffix.len != 0 or t.arguments.start.len != 0))
        {
            if (t.call_id.suffix.len != 0) {
                call_id_section = try b.plusLit(try b.optional(try b.litPlus(t.call_id.prefix, try p.toolId(try b.until(t.call_id.suffix)))), t.call_id.suffix);
            } else {
                call_id_section = try b.optional(try b.litPlus(t.call_id.prefix, try p.toolId(try b.until(t.arguments.start))));
            }
            have_call_id = true;
        }
        const schema_name = try std.mem.concat(a, u8, &.{ "tool-", name, "-schema" });
        var args_parser = try p.toolArgs(try b.schemaJson(try b.json(), schema_name, &schema, false));
        if (t.arguments.start.len != 0) {
            args_parser = try b.plus(try b.literal(t.arguments.start), args_parser);
        }
        if (t.arguments.end.len != 0) {
            args_parser = try b.plus(args_parser, try b.literal(t.arguments.end));
        }

        const atomic_peek: ?ParserId = if (t.arguments.start.len != 0) try b.peek(try b.literal(t.arguments.start)) else null;
        const func_parser = try buildFuncParser(t, p, name, call_id_section, have_call_id, args_parser, atomic_peek);
        try b.orAssign(&self.tool_choice, try b.rule(try std.mem.concat(a, u8, &.{ "tool-", name }), func_parser, false));
    }
};

/// Ports `analyze_tools`'s `build_tool_parser_tag_json` (chat-auto-parser-generator.cpp:288
/// @d81235049): tagged calls with JSON arguments.
fn buildToolParserTagJson(t: *const types.AnalyzeTools, ctx: *Ctx) PegError!ParserId {
    const p = ctx.p;
    const b = &p.b;
    const inputs = ctx.inputs;
    const format = &t.format;

    var state: TagJson = .{ .t = t, .ctx = ctx, .tool_choice = try b.choiceEmpty() };
    parsers.foreachFunction(&inputs.tools, &state, TagJson.tool) catch |e| return util.toPeg(e);
    const tool_choice = state.tool_choice;

    const require_calls = inputs.tool_choice == .required;

    var tool_calls = try b.eps();

    if (format.per_call_start.len != 0) {
        const wrapped_call = try b.plusLit(try b.litPlus(format.per_call_start, tool_choice), format.per_call_end);
        if (inputs.parallel_tool_calls) {
            tool_calls = try b.triggerRule("tool-call", try b.plus(wrapped_call, try b.zeroOrMore(try b.plus(try b.space(), wrapped_call))));
        } else {
            tool_calls = try b.triggerRule("tool-call", wrapped_call);
        }
        if (format.section_start.len != 0) {
            var x = try b.plus(try b.literal(format.section_start), try b.space());
            x = try b.plus(x, tool_calls);
            x = try b.plus(x, try b.space());
            x = try b.plus(x, if (format.section_end.len == 0) try b.end() else try b.literal(format.section_end));
            tool_calls = try b.triggerRule("tool-calls", x);
        }
    } else {
        const separator = ", "; // Default
        if (inputs.parallel_tool_calls) {
            var x = try b.litPlus(format.section_start, tool_choice);
            x = try b.plus(x, try b.zeroOrMore(try b.litPlus(separator, tool_choice)));
            tool_calls = try b.triggerRule("tool-call", try b.plusLit(x, format.section_end));
        } else {
            tool_calls = try b.triggerRule("tool-call", try b.plusLit(try b.litPlus(format.section_start, tool_choice), format.section_end));
        }
    }

    if (!require_calls) {
        tool_calls = try b.optional(tool_calls);
    }

    const trigger_marker = if (format.section_start.len != 0) format.section_start else format.per_call_start;
    const content_before_tools = if (trigger_marker.len == 0) try b.eps() else try b.until(trigger_marker);
    const x = try b.plus(ctx.reasoning_parser, try b.optional(try p.content(content_before_tools)));
    return b.plus(try b.plus(x, tool_calls), try b.end());
}

/// The per-tool callback of `build_tool_parser_tag_tagged`.
const TagTagged = struct {
    t: *const types.AnalyzeTools,
    ctx: *Ctx,
    until_suffix: ParserId,
    tool_choice: ParserId,
    name: []const u8 = "",
    required_parsers: std.ArrayList(ParserId) = .empty,
    optional_parsers: std.ArrayList(ParserId) = .empty,

    fn tool(self: *TagTagged, tool_def: *Value) Error!void {
        self.build(tool_def) catch |e| return util.fromPeg(e);
    }

    fn param(self: *TagTagged, prm: *const json.schema.Property, doc: *peg.SharedDoc) Error!void {
        self.buildParam(prm, doc) catch |e| return util.fromPeg(e);
    }

    fn buildParam(self: *TagTagged, prm: *const json.schema.Property, doc: *peg.SharedDoc) PegError!void {
        const t = self.t;
        const p = self.ctx.p;
        const b = &p.b;
        const a = self.ctx.a;
        const args = &t.arguments;

        var head = try b.litPlus(args.name_prefix, try p.toolArgName(try b.literal(prm.name)));
        head = try b.plusLit(head, args.name_suffix);
        const open = try p.toolArgOpen(head);
        const with_prefix = try b.plusLit(open, args.value_prefix);
        const value = if (prm.schema.mayBeString())
            try b.acOne(try b.plus(try p.toolArgStringValue(self.until_suffix), try p.toolArgClose(try b.literal(args.value_suffix))), args.value_suffix)
        else blk: {
            const schema_name = try std.mem.concat(a, u8, &.{ "tool-", self.name, "-arg-", prm.name, "-schema" });
            const json_value = try p.toolArgJsonValue(try b.schemaDoc(try b.json(), schema_name, doc, prm.schema, false));
            break :blk try b.plus(json_value, try p.toolArgClose(try b.literal(args.value_suffix)));
        };
        const arg = try p.toolArg(try b.plus(with_prefix, value));

        const named_arg = try b.rule(try std.mem.concat(a, u8, &.{ "tool-", self.name, "-arg-", prm.name }), arg, false);
        if (prm.required) {
            try self.required_parsers.append(a, named_arg);
        } else {
            try self.optional_parsers.append(a, named_arg);
        }
    }

    fn build(self: *TagTagged, tool_def: *Value) PegError!void {
        const t = self.t;
        const p = self.ctx.p;
        const b = &p.b;
        const a = self.ctx.a;
        const func = tool_def.at("function") catch |e| return peg.errors.fromJson(e);
        self.name = parsers.functionName(tool_def) catch |e| return util.toPeg(e);
        const name = self.name;

        // Build parser for each argument, separating required and optional
        self.required_parsers = .empty;
        self.optional_parsers = .empty;
        parsers.foreachParameter(a, func, self, param) catch |e| return util.toPeg(e);

        // Build required arg sequence in definition order
        var args_seq = try b.eps();
        for (self.required_parsers.items, 0..) |rp, i| {
            if (i > 0) {
                args_seq = try b.plus(args_seq, try b.space());
            }
            args_seq = try b.plus(args_seq, rp);
        }

        // Build optional args with flexible ordering
        if (self.optional_parsers.items.len != 0) {
            var any_opt = try b.choiceEmpty();
            for (self.optional_parsers.items) |opt| {
                try b.orAssign(&any_opt, opt);
            }
            args_seq = try b.plus(args_seq, try b.repeat(try b.plus(try b.space(), any_opt), 0, -1));
        }

        if (t.arguments.start.len != 0) {
            args_seq = try b.plus(try b.literal(t.arguments.start), args_seq);
        }
        if (t.arguments.end.len != 0) {
            args_seq = try b.plus(args_seq, try b.literal(t.arguments.end));
        }

        // Build call_id parser based on position (if supported)
        var call_id_section = try b.eps();
        var have_call_id = false;
        if (t.call_id.pos == .BETWEEN_FUNC_AND_ARGS and t.call_id.prefix.len != 0 and
            (t.call_id.suffix.len != 0 or t.arguments.start.len != 0))
        {
            have_call_id = true;
            if (t.call_id.suffix.len != 0) {
                call_id_section = try b.optional(try b.plusLit(try b.litPlus(t.call_id.prefix, try p.toolId(try b.until(t.call_id.suffix))), t.call_id.suffix));
            } else {
                call_id_section = try b.optional(try b.litPlus(t.call_id.prefix, try p.toolId(try b.until(t.arguments.start))));
            }
        }

        // Only peek for an arg tag when there are required args that must follow.
        // When all args are optional, the model may emit no arg tags at all (#20650).
        const atomic_peek: ?ParserId = if (t.arguments.name_prefix.len != 0 and self.required_parsers.items.len != 0)
            try b.peek(try b.literal(t.arguments.name_prefix))
        else
            null;
        const func_parser = try buildFuncParser(t, p, name, call_id_section, have_call_id, args_seq, atomic_peek);
        try b.orAssign(&self.tool_choice, try b.rule(try std.mem.concat(a, u8, &.{ "tool-", name }), func_parser, false));
    }
};

/// Ports `analyze_tools`'s `build_tool_parser_tag_tagged` (chat-auto-parser-generator.cpp:359
/// @d81235049): tagged calls with tagged arguments.
fn buildToolParserTagTagged(t: *const types.AnalyzeTools, ctx: *Ctx) PegError!ParserId {
    const p = ctx.p;
    const b = &p.b;
    const inputs = ctx.inputs;
    const format = &t.format;

    const until_suffix = try b.rule("until-suffix", try b.until(t.arguments.value_suffix), false);

    var state: TagTagged = .{ .t = t, .ctx = ctx, .until_suffix = until_suffix, .tool_choice = try b.choiceEmpty() };
    parsers.foreachFunction(&inputs.tools, &state, TagTagged.tool) catch |e| return util.toPeg(e);
    const tool_choice = state.tool_choice;

    const require_tools = inputs.tool_choice == .required;

    var tool_calls = try b.eps();

    if (format.per_call_start.len != 0) {
        var wrapped_call = try b.plus(try b.litPlus(format.per_call_start, try b.space()), tool_choice);
        wrapped_call = try b.plus(wrapped_call, try b.space());
        wrapped_call = try b.plusLit(wrapped_call, format.per_call_end);
        if (inputs.parallel_tool_calls) {
            const x = try b.plus(wrapped_call, try b.zeroOrMore(try b.plus(try b.space(), wrapped_call)));
            tool_calls = try b.triggerRule("tool-call", try b.plus(x, try b.space()));
        } else {
            tool_calls = try b.triggerRule("tool-call", try b.plus(wrapped_call, try b.space()));
        }
        if (format.section_start.len != 0) {
            var x = try b.plus(try b.literal(format.section_start), try b.space());
            x = try b.plus(x, tool_calls);
            x = try b.plus(x, try b.space());
            x = try b.plus(x, if (format.section_end.len == 0) try b.end() else try b.plus(try b.literal(format.section_end), try b.space()));
            tool_calls = try b.triggerRule("tool-calls", x);
        }
    } else {
        const separator = ", "; // Default

        if (inputs.parallel_tool_calls) {
            var x = try b.plus(try b.litPlus(format.section_start, try b.space()), tool_choice);
            x = try b.plus(x, try b.zeroOrMore(try b.litPlus(separator, tool_choice)));
            x = try b.plus(x, try b.space());
            tool_calls = try b.triggerRule("tool-call", try b.plusLit(x, format.section_end));
        } else {
            var x = try b.plus(try b.litPlus(format.section_start, try b.space()), tool_choice);
            x = try b.plus(x, try b.space());
            tool_calls = try b.triggerRule("tool-call", try b.plusLit(x, format.section_end));
        }
    }

    if (!require_tools) {
        tool_calls = try b.optional(tool_calls);
    }

    const trigger_marker = if (format.section_start.len != 0) format.section_start else format.per_call_start;
    const content_before_tools = if (trigger_marker.len == 0) try b.eps() else try b.until(trigger_marker);
    const x = try b.plus(ctx.reasoning_parser, try b.optional(try p.content(content_before_tools)));
    return b.plus(try b.plus(x, tool_calls), try b.end());
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
