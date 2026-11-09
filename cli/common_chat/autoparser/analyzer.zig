//! The differential analysis: a template rendered over pairs of probe
//! conversations that differ in one thing, and its format read off the
//! differences -- the reasoning and content markers, the tool-call section,
//! the function name and argument markers, the call id.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/chat-diff-analyzer.cpp` (llama.cpp v0.8.0,
//! `8642541b2`): the probe constants and messages, the workarounds,
//! `autoparser`'s `analyze_template` and its helpers, and the three analyzers'
//! constructors and methods. Each declaration names what it ports and the
//! line it began at.
//!
//! # Shape
//!
//! The C++'s analyzers are classes whose constructors run the analysis and
//! fill their own fields; here each is a function over the field struct of
//! `types.zig`. The probes are written as JSON text, member for member in
//! the order the C++ initializer lists insert them, since `ordered_json`
//! keeps that order and templates print it. The tagged parsers that pick
//! markers out of a diff are rebuilt for each use, as the C++'s lambdas
//! are; their parser ids never leave the analysis, so only what they match
//! matters, not the order they are built in.
//!
//! # Ownership
//!
//! Everything lives in the arena `a` the analysis is given.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("../exception.zig");
const template = @import("../template.zig");
const chat_peg = @import("../chat_peg.zig");
const util = @import("../util.zig");
const log = @import("../log.zig");
const types = @import("types.zig");
const helpers = @import("helpers.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const Template = template.Template;
const TemplateParams = types.TemplateParams;
const Autoparser = types.Autoparser;
const ParserId = peg.ParserId;
const B = peg.Builder;
const PegError = peg.Error;

const trimWhitespace = helpers.trimWhitespace;
const trimLeadingWhitespace = helpers.trimLeadingWhitespace;
const compareVariants = helpers.compareVariants;
const npos = util.npos;

// -----------------------------------------------------------------------------
// Probes

/// Ports `FUN_FIRST` and the other probe strings (chat-diff-analyzer.cpp:24
/// @cd959bae9).
const FUN_FIRST = "FFF_FIRST_FUN_F";
const FUN_SECOND = "SSS_SECOND_FUN_S";
const ARG_FIRST = "AA_ARG_FST_AA";
const ARG_SECOND = "BB_ARG_SND_BB";
const USER_MSG = "U_USER_MSG Hello END_U";
const USER_MSG_TWO = "V_USER_MSG Hello END_V";
const ASSISTANT_MSG = "A_ASST_MSG I can help END_A";
const THINKING_CONTENT = "REASON_PART I am thinking END_R";
const CALL_ID_001 = "call00001";
const CALL_ID_002 = "call00002";
const CALL_ID_999 = "call99999";

/// Ports `params_schema` (chat-diff-analyzer.cpp:208 @cd959bae9).
const params_schema = "{\"type\":\"object\",\"properties\":{\"" ++ ARG_FIRST ++ "\":{\"type\":\"string\",\"description\":\"First argument\"}," ++
    "\"" ++ ARG_SECOND ++ "\":{\"type\":\"string\",\"description\":\"Second argument\"}},\"required\":[]}";

/// Ports `tools` (chat-diff-analyzer.cpp:216 @cd959bae9).
const probe_tools = "[{\"type\":\"function\",\"function\":{\"name\":\"" ++ FUN_FIRST ++ "\",\"description\":\"Test function foo\",\"parameters\":" ++ params_schema ++ "}}," ++
    "{\"type\":\"function\",\"function\":{\"name\":\"" ++ FUN_SECOND ++ "\",\"description\":\"Test function bar\",\"parameters\":" ++ params_schema ++ "}}]";

/// Ports `user_msg` (chat-diff-analyzer.cpp:225 @cd959bae9).
const user_msg = "{\"role\":\"user\",\"content\":\"" ++ USER_MSG ++ "\"}";

/// Ports `build_tool_call` (chat-diff-analyzer.cpp:230 @cd959bae9): `args`
/// is JSON text.
fn buildToolCall(comptime name: []const u8, comptime args: []const u8, comptime id: []const u8) []const u8 {
    return "{\"id\":\"" ++ id ++ "\",\"type\":\"function\",\"function\":{\"name\":\"" ++ name ++ "\",\"arguments\":" ++ args ++ "}}";
}

/// Ports `first_tool_call_zero_args` and the other probe calls
/// (chat-diff-analyzer.cpp:238 @cd959bae9).
const first_tool_call_zero_args = buildToolCall(FUN_FIRST, "{}", CALL_ID_001);
const first_tool_call_one_arg = buildToolCall(FUN_FIRST, "{\"" ++ ARG_FIRST ++ "\":\"XXXX\"}", CALL_ID_001);
const first_tool_call_one_arg_other_val = buildToolCall(FUN_FIRST, "{\"" ++ ARG_FIRST ++ "\":\"YYYY\"}", CALL_ID_001);
const first_tool_call_other_arg = buildToolCall(FUN_FIRST, "{\"" ++ ARG_SECOND ++ "\":\"YYYY\"}", CALL_ID_001);
const first_tool_call = buildToolCall(FUN_FIRST, "{\"" ++ ARG_FIRST ++ "\":\"XXXX\",\"" ++ ARG_SECOND ++ "\":\"YYYY\"}", CALL_ID_001);
const second_tool_call = buildToolCall(FUN_SECOND, "{\"" ++ ARG_FIRST ++ "\":\"XXXX\",\"" ++ ARG_SECOND ++ "\":\"YYYY\"}", CALL_ID_002);
const first_tool_call_alt_id = buildToolCall(FUN_FIRST, "{\"" ++ ARG_FIRST ++ "\":\"XXXX\",\"" ++ ARG_SECOND ++ "\":\"YYYY\"}", CALL_ID_999);

/// An assistant message with `content` "" and these tool calls.
fn assistantCalls(comptime calls: []const u8) []const u8 {
    return "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[" ++ calls ++ "]}";
}

const assistant_content = "{\"role\":\"assistant\",\"content\":\"" ++ ASSISTANT_MSG ++ "\"}";

/// A probe's params: these messages, the probe tools when `with_tools`, no
/// generation prompt, thinking on.
fn probe(a: Allocator, comptime messages: []const u8, with_tools: bool) Error!TemplateParams {
    return .{
        .messages = json.parse(a, messages) catch |e| return util.fromJson(e),
        .tools = if (with_tools) json.parse(a, probe_tools) catch |e| return util.fromJson(e) else .null,
        .add_generation_prompt = false,
        .enable_thinking = true,
    };
}

/// `params` with its messages replaced, as the C++'s modifier lambdas do.
fn withMessages(a: Allocator, params: TemplateParams, comptime messages: []const u8) Error!TemplateParams {
    var p = params;
    p.messages = json.parse(a, messages) catch |e| return util.fromJson(e);
    return p;
}

// -----------------------------------------------------------------------------
// Tagged parsers

/// A tagged parse's outcome and its tags.
const Tagged = struct {
    ok: bool,
    tags: chat_peg.TagMap,

    /// `result.tags[key]`: "" for a tag that matched nothing.
    fn get(self: *const Tagged, key: []const u8) []const u8 {
        for (self.tags.entries.items) |e| if (std.mem.eql(u8, e.key, key)) return e.value;
        return "";
    }
};

/// `build_tagged_peg_parser(f).parse_anywhere_and_extract(input)`.
fn anywhere(a: Allocator, ctx: anytype, comptime f: fn (@TypeOf(ctx), *B) PegError!ParserId, input: []const u8) Error!Tagged {
    var tp = chat_peg.buildTaggedPegParser(a, ctx, f) catch |e| return util.fromPeg(e);
    defer tp.deinit();
    const r = tp.parseAnywhereAndExtract(a, input) catch |e| return util.fromPeg(e);
    return .{ .ok = r.result.success(), .tags = r.tags };
}

/// `build_tagged_peg_parser(f).parse_and_extract(input)`.
fn extract(a: Allocator, ctx: anytype, comptime f: fn (@TypeOf(ctx), *B) PegError!ParserId, input: []const u8) Error!Tagged {
    var tp = chat_peg.buildTaggedPegParser(a, ctx, f) catch |e| return util.fromPeg(e);
    defer tp.deinit();
    const r = tp.parseAndExtract(a, input, peg.Flags.none) catch |e| return util.fromPeg(e);
    return .{ .ok = r.result.success(), .tags = r.tags };
}

/// `p.marker() + p.space()`.
fn markerSpace(b: *B) PegError!ParserId {
    return b.plus(try b.marker(), try b.space());
}

/// `p.zero_or_more(p.negate(p.marker()) + p.any())`: everything up to the
/// next marker.
fn upToMarker(b: *B) PegError!ParserId {
    return b.zeroOrMore(try b.plus(try b.negate(try b.marker()), try b.any()));
}

/// `p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end()`:
/// the last marker and the text after it.
fn lastMarker(b: *B) PegError!ParserId {
    return b.plus(try b.plus(try b.marker(), try upToMarker(b)), try b.end());
}

/// `p.zero_or_more(p.negate(last) + p.any()) + p.tag(name, last)`.
fn untilLastMarker(b: *B, comptime name: []const u8) PegError!ParserId {
    const last = try lastMarker(b);
    return b.plus(try b.zeroOrMore(try b.plus(try b.negate(last), try b.any())), try b.tag(name, last));
}

// -----------------------------------------------------------------------------
// Workarounds

/// `std::string::find(x) != npos` on the template's source.
fn srcHas(tmpl: *const Template, needle: []const u8) bool {
    return util.contains(tmpl.src, needle);
}

/// Ports `workarounds` (chat-diff-analyzer.cpp:36 @cd959bae9): the
/// templates the analysis reads wrong, patched by name.
fn applyWorkarounds(tmpl: *const Template, analysis: *Autoparser, a: Allocator) Error!void {
    // Old reasoning Qwen templates - they don't really display reasoning content, but we still want to
    // support reasoning on them
    if (srcHas(tmpl, "content.split('</think>')") and
        !srcHas(tmpl, "reasoning_content") and
        !srcHas(tmpl, "<SPECIAL_12>") and
        analysis.reasoning.mode == .NONE)
    {
        analysis.reasoning.mode = .TAG_BASED;
        analysis.reasoning.start = "<think>";
        analysis.reasoning.end = "</think>";
        try analysis.preserved_tokens.append(a, "<think>");
        try analysis.preserved_tokens.append(a, "</think>");
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: old Qwen/Deepseek thinking template]\n\x1b[0m", .{});
    }
    // Granite 3.3, with separate reasoning and content markers
    if (srcHas(tmpl, "Write your thoughts between <think></think> and write your response between " ++
        "<response></response>"))
    {
        analysis.reasoning.mode = .TAG_BASED;
        analysis.reasoning.start = "<think>";
        analysis.reasoning.end = "</think>";
        try analysis.preserved_tokens.append(a, "<think>");
        try analysis.preserved_tokens.append(a, "</think>");
        analysis.content.mode = .WRAPPED_WITH_REASONING;
        analysis.content.start = "<response>";
        analysis.content.end = "</response>";
        try analysis.preserved_tokens.append(a, "<response>");
        try analysis.preserved_tokens.append(a, "</response>");
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Granite 3.3]\n\x1b[0m", .{});
    }
    // Cohere Command R+ - content wrapped in <|CHATBOT_TOKEN|>...<|END_OF_TURN_TOKEN|>
    if (srcHas(tmpl, "<|CHATBOT_TOKEN|>") and
        srcHas(tmpl, "<|END_OF_TURN_TOKEN|>") and analysis.content.start.len == 0)
    {
        analysis.content.mode = .ALWAYS_WRAPPED;
        analysis.content.start = "<|CHATBOT_TOKEN|>";
        analysis.content.end = "<|END_OF_TURN_TOKEN|>";
        try analysis.preserved_tokens.append(a, "<|CHATBOT_TOKEN|>");
        try analysis.preserved_tokens.append(a, "<|END_OF_TURN_TOKEN|>");
        analysis.user_start = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>";
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Cohere Command R+]\n\x1b[0m", .{});
    }
    // Functionary - no tool call section delimiter
    if (srcHas(tmpl, "set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", " ++
        "\"code_interpreter\") | list | length > 0"))
    {
        analysis.content.mode = .PLAIN;
        analysis.content.end = "";
        analysis.tools.function.name_prefix = "";
        analysis.tools.format.section_start = "";
        analysis.tools.format.section_end = "";
        analysis.tools.format.per_call_start = "<function=";
        analysis.tools.format.per_call_end = "</function>";
        analysis.tools.function.close = "";
        analysis.preserved_tokens.clearRetainingCapacity();
        try analysis.preserved_tokens.append(a, "<|eot_id|>");
        try analysis.preserved_tokens.append(a, "<|eom_id|>");
        try analysis.preserved_tokens.append(a, "<function=");
        try analysis.preserved_tokens.append(a, ">");
        try analysis.preserved_tokens.append(a, "</function>");
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Functionary 3.1]\n\x1b[0m", .{});
    }
    // DeepSeek-R1-Distill-Qwen
    if (srcHas(tmpl, "{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'")) {
        analysis.tools.format.section_start = "<｜tool▁calls▁begin｜>";
        analysis.tools.format.section_end = "<｜tool▁calls▁end｜>";
        analysis.tools.format.per_call_start = "<｜tool▁call▁begin｜>function";
        analysis.tools.function.name_prefix = "<｜tool▁sep｜>";
        analysis.tools.format.per_call_end = "<｜tool▁call▁end｜>";
        analysis.tools.function.close = "```";
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: DeepSeek-R1-Distill-Qwen]\n\x1b[0m", .{});
    }
    // Nemotron Nano v2
    if (srcHas(tmpl, "<SPECIAL_10>") and srcHas(tmpl, "<SPECIAL_11>") and
        srcHas(tmpl, "<SPECIAL_12>") and srcHas(tmpl, "<TOOL_RESPONSE>"))
    {
        analysis.tools.format.mode = .JSON_NATIVE;
        analysis.tools.format.section_start = "";
        analysis.tools.format.section_end = "";
        analysis.tools.format.per_call_start = "<TOOLCALL>";
        analysis.tools.format.per_call_end = "</TOOLCALL>";
        analysis.tools.format.tools_array_wrapped = true;
        analysis.content.mode = .PLAIN;
        analysis.content.start = "";
        analysis.content.end = "";
        analysis.reasoning.mode = .TAG_BASED;
        analysis.reasoning.start = "<think>\n";
        analysis.reasoning.end = "</think>";
        analysis.assistant_start = "<SPECIAL_11>Assistant";
        analysis.user_start = "<SPECIAL_11>User";
        analysis.preserved_tokens.clearRetainingCapacity();
        try analysis.preserved_tokens.append(a, "<SPECIAL_11>");
        try analysis.preserved_tokens.append(a, "</think>");
        try analysis.preserved_tokens.append(a, "<TOOLCALL>");
        try analysis.preserved_tokens.append(a, "</TOOLCALL>");
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Nemotron Nano v2]\n\x1b[0m", .{});
    }
    // Fireworks
    if (srcHas(tmpl, "{%- set system_prompt = '<|start_header_id|>' + 'system' + '<|end_header_id|>\\n\\n'" ++
        " + message['content'] | trim + '\\n' + system_prompt_suffix + '<|eot_id|>' -%}"))
    {
        analysis.assistant_start = "<|start_header_id|>assistant<|end_header_id|>";
        analysis.user_start = "<|start_header_id|>user<|end_header_id|>";
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Fireworks v2]\n\x1b[0m", .{});
    }
    // Solar Open
    if (srcHas(tmpl, "<|begin|>assistant<|think|><|end|>")) {
        analysis.assistant_start = "<|begin|>assistant";
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Solar Open]\n\x1b[0m", .{});
    }
    // Apriel 1.6
    if (srcHas(tmpl, "if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text")) {
        analysis.user_start = "<|begin_user|>";
        analysis.assistant_start = "<|begin_assistant|>";
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Apriel 1.6]\n\x1b[0m", .{});
    }
    // template uses the JSON {name, parameters} tool instruction, emits the OpenAI function wrapper
    if (srcHas(tmpl, "Respond in the format {\"name\": function name") and
        srcHas(tmpl, "Do not use variables."))
    {
        analysis.tools.format.openai_wrapper_trigger = true;
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: JSON name/parameters tool instruction]\n\x1b[0m", .{});
    }
    // Laguna (poolside) - the v4 chat template renders reasoning and tool-arg
    // delimiters with formatting whitespace ("<think>\n", "</arg_value>\n") that
    // the model does not emit, so the inferred delimiters carry a spurious
    // newline and never match the model output. Trim to the bare tag. (v8
    // renders without the whitespace, so this is a no-op there.)
    if (srcHas(tmpl, "laguna_glm_thinking")) {
        analysis.reasoning.start = trimWhitespace(analysis.reasoning.start);
        analysis.reasoning.end = trimWhitespace(analysis.reasoning.end);
        analysis.tools.arguments.value_prefix = trimWhitespace(analysis.tools.arguments.value_prefix);
        analysis.tools.arguments.value_suffix = trimWhitespace(analysis.tools.arguments.value_suffix);
        analysis.tools.arguments.separator = trimWhitespace(analysis.tools.arguments.separator);
        analysis.tools.arguments.tolerate_intertag_whitespace = true;
        // The CONTROL/eot </assistant> token only halts generation when emitted as the
        // single token; after tool calls the model can spell it out as text tokens.
        // A literal stop string catches it either way.
        try analysis.additional_stops.append(a, "</assistant>");
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Laguna]\n\x1b[0m", .{});
    }
    // Bailing V3
    if (srcHas(tmpl, "Bailing V3 chat template")) {
        analysis.tools.arguments.value_suffix = trimWhitespace(analysis.tools.arguments.value_suffix);
        analysis.tools.arguments.tolerate_intertag_whitespace = true;
        log.dbg("\x1b[1m\x1b[38;5;214m[Patch: Bailing V3]\n\x1b[0m", .{});
    }
}

// -----------------------------------------------------------------------------
// autoparser

/// Ports `autoparser`'s `analyze_template` (chat-diff-analyzer.cpp:257
/// @cd959bae9): every phase of the analysis, then the workarounds.
pub fn analyzeTemplate(self: *Autoparser, a: Allocator, tmpl: *const Template) Error!void {
    self.jinja_caps = tmpl.originalCaps();
    self.reasoning = try analyzeReasoning(a, tmpl, self.jinja_caps.supports_tool_calls);
    self.content = try analyzeContent(a, tmpl, &self.reasoning);
    self.tools = if (self.jinja_caps.supports_tool_calls) try analyzeTools(a, tmpl, &self.jinja_caps, &self.reasoning) else .{};
    self.assistant_start = try detectAssistantStartMarker(self, a, tmpl);
    self.user_start = try detectUserStartMarker(a, tmpl);
    try collectPreservedTokens(self, a);

    try applyWorkarounds(tmpl, self, a);

    if (log.debugEnabled()) try logAnalysis(self, a);
    self.analysis_complete = true;
}

/// The `LOG_DBG` report at the end of `analyze_template`.
fn logAnalysis(self: *const Autoparser, a: Allocator) Error!void {
    log.dbg("\n--- Reasoning & Content Structure ---\n", .{});
    log.dbg("user_msg_start: {s}\n", .{self.user_start});
    log.dbg("assistant_msg_start: {s}\n", .{self.assistant_start});
    log.dbg("reasoning_mode: {s}\n", .{self.reasoning.mode.name()});
    log.dbg("reasoning_start: '{s}'\n", .{self.reasoning.start});
    log.dbg("reasoning_end: '{s}'\n", .{self.reasoning.end});
    log.dbg("content_mode: {s}\n", .{self.content.mode.name()});
    log.dbg("content_start: '{s}'\n", .{self.content.start});
    log.dbg("content_end: '{s}'\n", .{self.content.end});

    const t = &self.tools;
    log.dbg("\n--- Tool Call Structure ---\n", .{});
    log.dbg("tool_mode: {s}\n", .{t.format.mode.name()});
    log.dbg("supports_tools: {s}\n", .{if (self.jinja_caps.supports_tools) "true" else "false"});
    log.dbg("supports_parallel_calls: {s}\n", .{if (self.jinja_caps.supports_parallel_tool_calls) "true" else "false"});
    log.dbg("tool_section_start: '{s}'\n", .{t.format.section_start});
    log.dbg("tool_section_end: '{s}'\n", .{t.format.section_end});
    log.dbg("per_call_start: '{s}'\n", .{t.format.per_call_start});
    log.dbg("per_call_end: '{s}'\n", .{t.format.per_call_end});
    log.dbg("func_name_prefix: '{s}'\n", .{t.function.name_prefix});
    log.dbg("func_name_suffix: '{s}'\n", .{t.function.name_suffix});
    log.dbg("func_args_separator: '{s}'\n", .{t.function.args_separator});
    log.dbg("func_close: '{s}'\n", .{t.function.close});
    log.dbg("call_id_prefix: '{s}'\n", .{t.call_id.prefix});
    log.dbg("call_id_suffix: '{s}'\n", .{t.call_id.suffix});
    log.dbg("call_id_pos: '{s}'\n", .{t.call_id.pos.name()});
    log.dbg("args_start: '{s}'\n", .{t.arguments.start});
    log.dbg("args_end: '{s}'\n", .{t.arguments.end});
    log.dbg("arg_name_prefix: '{s}'\n", .{t.arguments.name_prefix});
    log.dbg("arg_name_suffix: '{s}'\n", .{t.arguments.name_suffix});
    log.dbg("arg_value_prefix: '{s}'\n", .{t.arguments.value_prefix});
    log.dbg("arg_value_suffix: '{s}'\n", .{t.arguments.value_suffix});
    log.dbg("name_field: '{s}'\n", .{t.format.name_field});
    log.dbg("args_field: '{s}'\n", .{t.format.args_field});
    log.dbg("id_field: '{s}'\n", .{t.format.id_field});
    log.dbg("gen_id_field: '{s}'\n", .{t.format.gen_id_field});
    var order: std.ArrayList(u8) = .empty;
    for (t.format.parameter_order.items) |p| {
        if (order.items.len != 0) try order.appendSlice(a, ", ");
        try order.appendSlice(a, p);
    }
    log.dbg("parameter_order: '{s}'\n", .{order.items});
    log.dbg("\x1b[1m\x1b[38;5;126m=== Differential analysis complete ===\n\x1b[0m", .{});
}

/// Ports `autoparser`'s `collect_preserved_tokens` (chat-diff-analyzer.cpp:313
/// @cd959bae9): every marker found, trimmed, once.
fn collectPreservedTokens(self: *Autoparser, a: Allocator) Error!void {
    const t = &self.tools;
    const all = [_][]const u8{
        self.reasoning.start,      self.reasoning.end,
        self.content.start,        self.content.end,
        t.format.section_start,    t.format.section_end,
        t.format.per_call_start,   t.format.per_call_end,
        t.function.name_prefix,    t.function.name_suffix,
        t.function.args_separator, t.function.close,
        t.arguments.start,         t.arguments.end,
        t.arguments.name_prefix,   t.arguments.name_suffix,
        t.arguments.separator,     t.arguments.value_prefix,
        t.arguments.value_suffix,  t.call_id.prefix,
        t.call_id.suffix,
    };
    outer: for (all) |org_token| {
        const token = trimWhitespace(org_token);
        if (token.len == 0) continue;
        // Avoid duplicates
        for (self.preserved_tokens.items) |have| if (std.mem.eql(u8, have, token)) continue :outer;
        try self.preserved_tokens.append(a, token);
    }
}

/// Ports `autoparser`'s `detect_assistant_start_marker` (chat-diff-analyzer.cpp:347
/// @cd959bae9): what a template writes before an assistant message's text.
fn detectAssistantStartMarker(self: *const Autoparser, a: Allocator, tmpl: *const Template) Error![]const u8 {
    const params = try probe(a, "[" ++ user_msg ++ "]", false);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ assistant_content ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed, skipping assistant start detection\n\x1b[0m", .{"detect_assistant_start_marker"});
        return "";
    };

    const usermsg = comparison.diff.right;
    if (!util.contains(usermsg, ASSISTANT_MSG)) {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Did not find assistant message in assistant message block, skipping detection\n\x1b[0m", .{"detect_assistant_start_marker"});
    }

    var ast_prefix = try util.substr(usermsg, 0, util.find(usermsg, ASSISTANT_MSG));
    if (self.reasoning.start.len != 0 and util.contains(ast_prefix, trimWhitespace(self.reasoning.start))) {
        ast_prefix = try util.substr(ast_prefix, 0, util.find(ast_prefix, trimWhitespace(self.reasoning.start)));
    }
    if (self.reasoning.end.len != 0 and util.contains(ast_prefix, trimWhitespace(self.reasoning.end))) {
        ast_prefix = try util.substr(ast_prefix, 0, util.find(ast_prefix, trimWhitespace(self.reasoning.end)));
    }
    return trimWhitespace(ast_prefix);
}

/// Ports `autoparser`'s `detect_user_start_marker` (chat-diff-analyzer.cpp:389
/// @cd959bae9): what a template writes before a user message's text, an
/// end marker of the turn before skipped.
fn detectUserStartMarker(a: Allocator, tmpl: *const Template) Error![]const u8 {
    const user_msg_two = "{\"role\":\"user\",\"content\":\"" ++ USER_MSG_TWO ++ "\"}";
    var params = try probe(a, "[]", false);
    var b = try withMessages(a, params, "[" ++ user_msg ++ "]");
    var comparison = try compareVariants(a, tmpl, &params, &b);

    if (comparison == null) {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed, unsupported empty messages? trying complex variant\n\x1b[0m", .{"detect_user_start_marker"});
        params = try withMessages(a, params, "[" ++ user_msg_two ++ "," ++ assistant_content ++ "]");
        b = try withMessages(a, params, "[" ++ user_msg_two ++ "," ++ assistant_content ++ "," ++ user_msg ++ "]");
        comparison = try compareVariants(a, tmpl, &params, &b);
        if (comparison == null) {
            log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed for reserve variant, aborting\n\x1b[0m", .{"detect_user_start_marker"});
            return "";
        }
    }

    var usermsg = comparison.?.diff.right;
    if (!util.contains(usermsg, USER_MSG)) {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Did not find user message in user message block, aborting detection\n\x1b[0m", .{"detect_user_start_marker"});
    }

    if (util.contains(usermsg, ASSISTANT_MSG)) {
        usermsg = try util.substrFrom(usermsg, util.find(usermsg, ASSISTANT_MSG) + ASSISTANT_MSG.len);
    }

    const candidate = try util.substr(usermsg, 0, util.find(usermsg, USER_MSG));
    const candidate_split = try helpers.segmentizeMarkers(a, candidate);
    var result: std.ArrayList(u8) = .empty;
    var encountered_marker = false;
    for (candidate_split) |mrk| {
        const lower_mrk = try std.ascii.allocLowerString(a, mrk.value);
        // heuristic to weed out potential end markers, but only at the start
        if (mrk.type == .MARKER and !encountered_marker and
            (util.contains(lower_mrk, "end") or util.contains(lower_mrk, "close")))
        {
            continue;
        }
        if (mrk.type == .TEXT and !encountered_marker and trimWhitespace(mrk.value).len == 0) {
            continue;
        }
        encountered_marker = encountered_marker or mrk.type == .MARKER;
        try result.appendSlice(a, mrk.value);
    }
    return trimWhitespace(result.items);
}

// -----------------------------------------------------------------------------
// analyze_reasoning

/// Ports `analyze_reasoning`'s constructor (chat-diff-analyzer.cpp:461
/// @cd959bae9).
fn analyzeReasoning(a: Allocator, tmpl: *const Template, supports_tools: bool) Error!types.AnalyzeReasoning {
    log.dbg("\x1b[1m\x1b[38;5;126m=== Starting differential analysis ===\n\x1b[0m", .{});
    log.dbg("\x1b[1m\x1b[38;5;214mPhase 1: Reasoning analysis\n\x1b[0m", .{});
    var r: types.AnalyzeReasoning = .{};
    try compareReasoningPresence(&r, a, tmpl);
    try compareThinkingEnabled(&r, a, tmpl);
    if (supports_tools) {
        try compareReasoningScope(&r, a, tmpl);
    }
    return r;
}

const ReasoningParsers = struct {
    /// `p.literal(reasoning_content) + p.space() + p.optional(p.tag("post", (p.marker() + p.space())) + p.rest())`
    fn delimiter(_: void, b: *B) PegError!ParserId {
        const head = try b.plus(try b.literal(THINKING_CONTENT), try b.space());
        return b.plus(head, try b.optional(try b.plus(try b.tag("post", try markerSpace(b)), try b.rest())));
    }

    /// `p.tag("pre", p.marker() + p.space()) + p.literal(reasoning_content) + p.tag("post", (p.space() + p.marker() + p.space())) + p.rest()`
    fn wrapped(_: void, b: *B) PegError!ParserId {
        var x = try b.plus(try b.tag("pre", try markerSpace(b)), try b.literal(THINKING_CONTENT));
        x = try b.plus(x, try b.tag("post", try b.plus(try b.plus(try b.space(), try b.marker()), try b.space())));
        return b.plus(x, try b.rest());
    }

    /// `p.tag("pre", p.marker() + p.space()) + p.literal(reasoning_content) + p.space() + p.tag("post", (p.marker() + p.space()))`
    fn scopeWrapped(_: void, b: *B) PegError!ParserId {
        var x = try b.plus(try b.tag("pre", try markerSpace(b)), try b.literal(THINKING_CONTENT));
        x = try b.plus(x, try b.space());
        return b.plus(x, try b.tag("post", try markerSpace(b)));
    }

    /// `p.literal(reasoning_content) + p.space() + p.optional(p.tag("post", (p.marker() + p.space())))`
    fn scopeDelimiter(_: void, b: *B) PegError!ParserId {
        const head = try b.plus(try b.literal(THINKING_CONTENT), try b.space());
        return b.plus(head, try b.optional(try b.tag("post", try markerSpace(b))));
    }
};

/// Ports `analyze_reasoning`'s `compare_reasoning_presence` (chat-diff-analyzer.cpp:473
/// @cd959bae9): the markers around reasoning content in an assistant turn.
fn compareReasoningPresence(r: *types.AnalyzeReasoning, a: Allocator, tmpl: *const Template) Error!void {
    const assistant_with_reasoning = "{\"role\":\"assistant\",\"content\":\"" ++ ASSISTANT_MSG ++ "\",\"reasoning_content\":\"" ++ THINKING_CONTENT ++ "\"}";
    const params = try probe(a, "[" ++ user_msg ++ "," ++ assistant_content ++ "]", false);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ assistant_with_reasoning ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed, skipping reasoning detection\n\x1b[0m", .{"compare_reasoning_presence"});
        return;
    };

    const diff = comparison.diff;
    if (diff.right.len != 0 and util.contains(diff.right, THINKING_CONTENT)) {
        // try the more aggressive parse first, if it fails, fall back to the delimiter one
        var result = try anywhere(a, {}, ReasoningParsers.wrapped, comparison.output_B);
        if (!result.ok) {
            result = try anywhere(a, {}, ReasoningParsers.delimiter, comparison.output_B);
        }
        if (result.ok) {
            if (result.get("pre").len != 0 and result.get("post").len != 0) {
                r.mode = .TAG_BASED;
                r.start = result.get("pre");
                r.end = result.get("post");
            } else if (result.get("post").len != 0) {
                r.mode = .TAG_BASED;
                r.end = result.get("post");
            }
        }
    }
}

/// Ports `analyze_reasoning`'s `compare_thinking_enabled` (chat-diff-analyzer.cpp:532
/// @cd959bae9): what `enable_thinking` changes in the generation prompt.
fn compareThinkingEnabled(r: *types.AnalyzeReasoning, a: Allocator, tmpl: *const Template) Error!void {
    var params = try probe(a, "[" ++ user_msg ++ "]", false);
    params.add_generation_prompt = true;
    params.enable_thinking = false;
    var b = params;
    b.enable_thinking = true;
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"compare_thinking_enabled"});
        return;
    };

    const diff = comparison.diff;
    const left_trimmed = trimWhitespace(diff.left);
    const right_trimmed = trimWhitespace(diff.right);

    if (left_trimmed.len == 0 and diff.right.len != 0) {
        if (right_trimmed.len != 0 and std.mem.endsWith(u8, comparison.output_B, right_trimmed)) {
            if (r.start.len == 0) {
                r.start = diff.right;
                r.mode = .TAG_BASED;
            }
        }
    } else if (right_trimmed.len == 0 and diff.left.len != 0) {
        if (left_trimmed.len != 0 and std.mem.endsWith(u8, comparison.output_A, left_trimmed)) {
            if (r.end.len == 0) {
                const seg = try helpers.pruneWhitespaceSegments(a, try helpers.segmentizeMarkers(a, comparison.output_A));
                if (seg.len >= 2 and std.mem.eql(u8, seg[seg.len - 1].value, left_trimmed) and seg[seg.len - 2].type == .MARKER) {
                    r.start = seg[seg.len - 2].value;
                }
                r.end = diff.left;
                r.mode = .TAG_BASED;
            }
        }
    } else if (left_trimmed.len != 0 and right_trimmed.len != 0) {
        // Full-output diff is noisy (e.g., SmolLM3 changes the system message when enable_thinking flips).
        // Try to find reasoning markers by tail-anchoring:
        // one output's generation prompt tail may appear in the other with extra reasoning markers appended.
        const output_A = comparison.output_A;
        const output_B = comparison.output_B;
        const anchor_len: usize = 64;

        for (0..2) |dir| {
            const base = if (dir == 0) output_B else output_A;
            const extended = if (dir == 0) output_A else output_B;

            const len = @min(base.len, anchor_len);
            const anchor = base[base.len - len ..];
            const pos = util.rfind(extended, anchor);
            if (pos == npos or pos + len >= extended.len) continue;

            const extra = trimWhitespace(extended[pos + len ..]);
            if (extra.len == 0) continue;

            const seg = try helpers.pruneWhitespaceSegments(a, try helpers.segmentizeMarkers(a, extra));
            if (seg.len == 2 and seg[0].type == .MARKER and seg[1].type == .MARKER) {
                if (r.start.len == 0) {
                    r.start = seg[0].value;
                }
                if (r.end.len == 0) {
                    r.end = seg[1].value;
                }
                r.mode = .TAG_BASED;
                break;
            }
        }
    }

    if (r.mode == .NONE and r.start.len == 0 and r.end.len != 0) {
        r.mode = .TAG_BASED;
    }
}

/// Ports `analyze_reasoning`'s `compare_reasoning_scope` (chat-diff-analyzer.cpp:616
/// @cd959bae9): reasoning that a template shows only beside tool calls.
fn compareReasoningScope(r: *types.AnalyzeReasoning, a: Allocator, tmpl: *const Template) Error!void {
    const assistant_reasoning_content = "{\"role\":\"assistant\",\"content\":\"" ++ ASSISTANT_MSG ++ "\",\"reasoning_content\":\"" ++ THINKING_CONTENT ++ "\"}";
    const assistant_reasoning_tools = "{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"" ++ THINKING_CONTENT ++ "\",\"tool_calls\":[" ++
        comptime buildToolCall(FUN_FIRST, "{\"" ++ ARG_FIRST ++ "\":\"VVVV\",\"" ++ ARG_SECOND ++ "\":\"XXXX\"}", CALL_ID_001) ++ "]}";

    const params = try probe(a, "[" ++ user_msg ++ "," ++ assistant_reasoning_content ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ assistant_reasoning_tools ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"compare_reasoning_scope"});
        return;
    };

    // Check if reasoning only appears in variant B (with tools)
    const reasoning_in_A = util.contains(comparison.output_A, THINKING_CONTENT);
    const reasoning_in_B = util.contains(comparison.output_B, THINKING_CONTENT);

    if (!reasoning_in_A and reasoning_in_B) {
        r.mode = .TOOLS_ONLY;
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Detected TOOLS_ONLY reasoning mode\n\x1b[0m", .{"compare_reasoning_scope"});

        const result = try anywhere(a, {}, ReasoningParsers.scopeWrapped, comparison.output_B);
        if (result.ok) {
            r.start = result.get("pre");
            r.end = result.get("post");
        } else {
            const delim = try anywhere(a, {}, ReasoningParsers.scopeDelimiter, comparison.output_B);
            if (delim.ok) {
                r.end = delim.get("post");
            } else {
                log.dbg("\x1b[1m\x1b[38;5;214m{s}: Unable to extract reasoning markers, falling back to reasoning = NONE\n\x1b[0m", .{"compare_reasoning_scope"});
                r.mode = .NONE;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// analyze_content

const ContentParsers = struct {
    /// `p.space() + diff_reasoning.left + p.space() + p.optional(p.marker()) + p.space() + p.end()`
    fn plain(left: []const u8, b: *B) PegError!ParserId {
        var x = try b.plusLit(try b.space(), left);
        x = try b.plus(x, try b.space());
        x = try b.plus(x, try b.optional(try b.marker()));
        x = try b.plus(x, try b.space());
        return b.plus(x, try b.end());
    }

    /// `p.literal(reasoning.end) + p.space() + p.literal(response)`
    fn postReasoning(end: []const u8, b: *B) PegError!ParserId {
        return b.plus(try b.plus(try b.literal(end), try b.space()), try b.literal(ASSISTANT_MSG));
    }

    /// `p.tag("pre", p.marker() + p.space()) + p.literal(response) + p.space() + p.tag("post", (p.marker() + p.space())) + p.rest()`
    fn wrapped(_: void, b: *B) PegError!ParserId {
        var x = try b.plus(try b.tag("pre", try markerSpace(b)), try b.literal(ASSISTANT_MSG));
        x = try b.plus(x, try b.space());
        x = try b.plus(x, try b.tag("post", try markerSpace(b)));
        return b.plus(x, try b.rest());
    }
};

/// Ports `analyze_content`'s constructor (chat-diff-analyzer.cpp:677
/// @cd959bae9): the markers around an assistant message's text.
fn analyzeContent(a: Allocator, tmpl: *const Template, reasoning: *const types.AnalyzeReasoning) Error!types.AnalyzeContent {
    log.dbg("\x1b[1m\x1b[38;5;214mPhase 2: Content analysis\n\x1b[0m", .{});
    var c: types.AnalyzeContent = .{};

    const assistant_with_tools = "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[" ++ comptime buildToolCall("test_func", "{\"arg1\":\"value1\"}", CALL_ID_001) ++ "]}";
    const assistant_with_reasoning = "{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"" ++ THINKING_CONTENT ++ "\"}";

    const params_content_only = try probe(a, "[" ++ user_msg ++ "," ++ assistant_content ++ "]", true);
    const with_tools = try withMessages(a, params_content_only, "[" ++ user_msg ++ "," ++ assistant_with_tools ++ "]");
    const comparison_with_tools = try compareVariants(a, tmpl, &params_content_only, &with_tools);
    const with_reasoning = try withMessages(a, params_content_only, "[" ++ user_msg ++ "," ++ assistant_with_reasoning ++ "]");
    const comparison_with_reasoning = try compareVariants(a, tmpl, &params_content_only, &with_reasoning);

    if (comparison_with_tools == null or comparison_with_reasoning == null) {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"analyze_content"});
        return c;
    }

    const diff_tools = comparison_with_tools.?.diff;
    const diff_reasoning = comparison_with_reasoning.?.diff;

    const response = ASSISTANT_MSG;

    var found_plain_content = false;
    if (std.mem.eql(u8, trimWhitespace(diff_tools.left), response)) {
        if ((try extract(a, diff_reasoning.left, ContentParsers.plain, diff_reasoning.left)).ok) {
            // We only have the content text in the diff (possibly with a stray EOG marker), so no markers
            c.mode = .PLAIN;
            found_plain_content = true;
        } else if (reasoning.mode != .NONE and reasoning.end.len != 0) {
            if ((try anywhere(a, reasoning.end, ContentParsers.postReasoning, diff_reasoning.left)).ok) {
                c.mode = .PLAIN;
                found_plain_content = true;
            }
        }
    }
    if (!found_plain_content) {
        var rdiff = diff_reasoning.left;
        if (reasoning.end.len != 0 and util.contains(rdiff, reasoning.end)) {
            rdiff = try util.substrFrom(rdiff, util.find(rdiff, reasoning.end) + reasoning.end.len);
        }
        // Take the more promising diff
        const pure_content = if (rdiff.len > diff_tools.left.len) rdiff else diff_tools.left;
        const result = try anywhere(a, {}, ContentParsers.wrapped, pure_content);
        c.start = result.get("pre");
        c.end = result.get("post");
        // TODO: WRAPPED_WITH_REASONING
    }

    // Determine content mode
    if (c.start.len != 0 or c.end.len != 0) {
        c.mode = .ALWAYS_WRAPPED;
        // TODO: END_DELIMITED content mode - delimited at end but not at start?
    }
    return c;
}

// -----------------------------------------------------------------------------
// analyze_tools

/// Ports `analyze_tools`'s constructor (chat-diff-analyzer.cpp:768
/// @cd959bae9): the tool-call format, then, outside JSON, the function,
/// argument and call-id markers.
fn analyzeTools(a: Allocator, tmpl: *const Template, caps: *const @import("../caps.zig").Caps, reasoning: *const types.AnalyzeReasoning) Error!types.AnalyzeTools {
    log.dbg("\x1b[1m\x1b[38;5;214mPhase 3: Tool call analysis\n\x1b[0m", .{});
    var t: types.AnalyzeTools = .{};

    try analyzeToolCalls(&t, a, tmpl, reasoning, caps.supports_parallel_tool_calls);

    if (t.format.mode != .NONE and t.format.mode != .JSON_NATIVE) {
        if (caps.supports_parallel_tool_calls) {
            try checkPerCallMarkers(&t, a, tmpl);
        }
        log.dbg("\x1b[1m\x1b[38;5;214mPhase 3a: Function call analysis\n\x1b[0m", .{});
        try extractFunctionMarkers(&t, a, tmpl);
        log.dbg("\x1b[1m\x1b[38;5;214mPhase 3b: Argument analysis\n\x1b[0m", .{});
        if (t.format.mode == .TAG_WITH_TAGGED) {
            try analyzeArguments(&t, a, tmpl);
        }
        try extractArgumentSeparator(&t, a, tmpl);
        try extractArgsMarkers(&t, a, tmpl);
        log.dbg("\x1b[1m\x1b[38;5;214mPhase 3c: Call id analysis\n\x1b[0m", .{});
        try extractCallIdMarkers(&t, a, tmpl);
    }
    return t;
}

/// Ports `analyze_tools`'s `analyze_tool_calls` (chat-diff-analyzer.cpp:793
/// @cd959bae9): the text one tool call adds to an assistant turn.
fn analyzeToolCalls(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template, reasoning: *const types.AnalyzeReasoning, supports_parallel_tool_calls: bool) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ assistant_content ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call) ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"analyze_tool_calls"});
        return;
    };

    const tool_section = comparison.diff.right;
    if (tool_section.len == 0) return;

    try analyzeToolCallFormat(t, a, tmpl, tool_section, FUN_FIRST, ARG_FIRST, reasoning, supports_parallel_tool_calls);
}

const ToolParsers = struct {
    /// `p.choice({ p.literal("{"), p.literal(":") }) << p.choice({ p.tag("dq", p.literal("\"") + p.literal(needle) + p.literal("\"")) })`
    fn inJson(needle: []const u8, b: *B) PegError!ParserId {
        const opener = try b.choice(&.{ try b.literal("{"), try b.literal(":") });
        const quoted = try b.plus(try b.plus(try b.literal("\""), try b.literal(needle)), try b.literal("\""));
        return b.shl(opener, try b.choice(&.{try b.tag("dq", quoted)}));
    }
};

/// Ports `analyze_tools`'s `analyze_tool_call_format` (chat-diff-analyzer.cpp:830
/// @cd959bae9): JSON, tagged with JSON arguments, or tagged throughout,
/// decided by whether the names appear quoted after `{` or `:`.
fn analyzeToolCallFormat(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template, haystack: []const u8, fun_name_needle: []const u8, arg_name_needle: []const u8, reasoning: *const types.AnalyzeReasoning, supports_parallel_tool_calls: bool) Error!void {
    if (fun_name_needle.len == 0 or arg_name_needle.len == 0 or haystack.len == 0) return;

    const fun_quote = (try anywhere(a, fun_name_needle, ToolParsers.inJson, haystack)).ok;
    const arg_quote = (try anywhere(a, arg_name_needle, ToolParsers.inJson, haystack)).ok;

    if (fun_quote) {
        // no need to check further, we're in JSON land
        t.format.mode = .JSON_NATIVE;
    } else if (arg_quote) {
        t.format.mode = .TAG_WITH_JSON;
    } else {
        t.format.mode = .TAG_WITH_TAGGED;
    }

    // first, remove any reasoning markers
    var clean_haystack = haystack;
    if (reasoning.start.len != 0) {
        const pos = util.find(haystack, reasoning.start);
        if (pos != npos) {
            clean_haystack = try std.mem.concat(a, u8, &.{ haystack[0..pos], haystack[pos + reasoning.start.len ..] });
        }
    }
    if (reasoning.end.len != 0) {
        const pos = util.find(clean_haystack, reasoning.end);
        if (pos != npos) {
            clean_haystack = try std.mem.concat(a, u8, &.{ clean_haystack[0..pos], clean_haystack[pos + reasoning.end.len ..] });
        }
    }

    if (t.format.mode == .JSON_NATIVE) {
        try analyzeToolCallFormatJsonNative(t, a, clean_haystack, fun_name_needle, arg_name_needle);
        if (supports_parallel_tool_calls) {
            try analyzeJsonNativeParallelCalls(t, a, tmpl);
        }
    } else {
        try analyzeToolCallFormatNonJson(t, a, clean_haystack, fun_name_needle);
    }
    // always relax whitespace requirements on ending markers since they don't influence content
    t.format.section_end = trimWhitespace(t.format.section_end);
    t.format.per_call_end = trimWhitespace(t.format.per_call_end);
}

/// Ports `analyze_tools`'s `analyze_json_native_parallel_calls` (chat-diff-analyzer.cpp:888
/// @cd959bae9): a section marker repeated per call is a per-call marker.
fn analyzeJsonNativeParallelCalls(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call) ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call ++ "," ++ second_tool_call) ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"analyze_json_native_parallel_calls"});
        return;
    };

    const second_call = comparison.diff.right;
    if (t.format.section_start.len != 0 and util.contains(second_call, t.format.section_start)) {
        t.format.per_call_start = t.format.section_start;
        t.format.per_call_end = t.format.section_end;
        t.format.section_start = "";
        t.format.section_end = "";
    }
}

/// `std::string::substr(pos, n)` with C++'s `int` positions converted as
/// the C++ converts them: a negative `pos` is a huge one, which throws, and
/// a negative `n` a huge count, which is the rest of the string.
fn substrInt(s: []const u8, pos: i64, n: i64) Error![]const u8 {
    return util.substr(s, @bitCast(@as(i64, pos)), @bitCast(@as(i64, n)));
}

/// Ports `analyze_tools`'s `analyze_tool_call_format_json_native` (chat-diff-analyzer.cpp:924
/// @cd959bae9): the call's JSON parsed, its fields named by what they hold,
/// and the markers around it.
fn analyzeToolCallFormatJsonNative(t: *types.AnalyzeTools, a: Allocator, clean_haystack: []const u8, fun_name_needle: []const u8, arg_name_needle: []const u8) Error!void {
    // we might not have the typical OpenAI tool calling structure
    var json_start: i64 = if (std.mem.indexOfScalar(u8, clean_haystack, '{')) |i| @intCast(i) else -1;
    var json_end: i64 = if (std.mem.lastIndexOfScalar(u8, clean_haystack, '}')) |i| @intCast(i) else -1;
    const cut = try substrInt(clean_haystack, json_start, json_end - json_start + 1);
    var call_struct = json.parse(a, cut) catch |e| return util.fromJson(e);

    const Reg = struct {
        t: *types.AnalyzeTools,
        a: Allocator,
        fun: []const u8,
        arg: []const u8,

        fn field(self: @This(), prefix: []const u8, key: []const u8) Error![]const u8 {
            return if (prefix.len != 0) try std.mem.concat(self.a, u8, &.{ prefix, ".", key }) else try self.a.dupe(u8, key);
        }

        fn register(self: @This(), prefix: []const u8, key: []const u8, value: *const Value) Error!void {
            if (value.isString() and util.contains(value.string, "call0000")) {
                self.t.format.id_field = try self.field(prefix, key);
            } else if (value.isString() and std.mem.eql(u8, value.string, self.fun)) {
                self.t.format.name_field = try self.field(prefix, key);
            } else if (util.contains(json.dump(self.a, value, -1) catch |e| return util.fromJson(e), self.arg)) { // handle both string and JSON obj variants
                self.t.format.args_field = try self.field(prefix, key);
            } else if (util.contains(key, "id")) {
                // heuristics for generated id field
                self.t.format.gen_id_field = try self.field(prefix, key);
            }
        }
    };
    const reg: Reg = .{ .t = t, .a = a, .fun = fun_name_needle, .arg = arg_name_needle };

    var it = call_struct.items();
    while (it.next()) |el| {
        const key = try a.dupe(u8, el.key);
        if (std.mem.eql(u8, key, fun_name_needle)) {
            t.format.fun_name_is_key = true;
            // When function name is the key, there's no name field and args are direct
            t.format.name_field = "";
            t.format.args_field = "";
            // Don't register this element - the function name IS the key, not a field
        } else {
            if (el.value.isObject() and
                !util.contains(json.dump(a, el.value, -1) catch |e| return util.fromJson(e), arg_name_needle))
            { // not the args object
                t.format.function_field = key;
                var sub = el.value.items();
                while (sub.next()) |subel| {
                    try reg.register(key, try a.dupe(u8, subel.key), subel.value);
                }
            }
            // Register this element as a potential field
            try reg.register("", key, el.value);
        }
    }

    const ArrayParser = struct {
        fn f(c: []const u8, b: *B) PegError!ParserId {
            const pre = try b.tag("pre", try b.plus(try b.literal("["), try b.space()));
            const x = try b.plus(pre, try b.literal(c));
            return b.plus(x, try b.tag("post", try b.plus(try b.space(), try b.literal("]"))));
        }
    };
    const ar_parse_res = try anywhere(a, cut, ArrayParser.f, clean_haystack);
    if (ar_parse_res.ok) {
        t.format.tools_array_wrapped = true;
        json_start -= @intCast(ar_parse_res.get("pre").len);
        json_end += @intCast(ar_parse_res.get("post").len);
    }
    json_end += 1; // we want to move past the closing char for end marker extraction

    const Located = struct { pos: usize, name: []const u8 };
    var located_params: std.ArrayList(Located) = .empty;
    if (t.format.name_field.len != 0) {
        try located_params.append(a, .{ .pos = util.find(clean_haystack, t.format.name_field), .name = t.format.name_field });
    }
    if (t.format.args_field.len != 0) {
        try located_params.append(a, .{ .pos = util.find(clean_haystack, t.format.args_field), .name = t.format.args_field });
    }
    if (t.format.id_field.len != 0) {
        try located_params.append(a, .{ .pos = util.find(clean_haystack, t.format.id_field), .name = t.format.id_field });
    }
    if (t.format.gen_id_field.len != 0) {
        try located_params.append(a, .{ .pos = util.find(clean_haystack, t.format.gen_id_field), .name = t.format.gen_id_field });
    }
    std.mem.sort(Located, located_params.items, {}, struct {
        fn lt(_: void, x: Located, y: Located) bool {
            if (x.pos != y.pos) return x.pos < y.pos;
            return std.mem.order(u8, x.name, y.name) == .lt;
        }
    }.lt);
    for (located_params.items) |pair| {
        try t.format.parameter_order.append(a, pair.name);
    }
    // we can immediately extract tool calling markers too
    t.format.section_start = trimLeadingWhitespace(try substrInt(clean_haystack, 0, json_start));
    t.format.section_end = trimWhitespace(try substrInt(clean_haystack, json_end, -1));
    // When tools_array_wrapped is true, the closing bracket is part of the array structure,
    // not a separate section end marker. Clear tool_section_end to avoid duplicate brackets.
    if (t.format.tools_array_wrapped and std.mem.eql(u8, t.format.section_end, "]")) {
        t.format.section_end = "";
    }
}

/// Ports `analyze_tools`'s `analyze_tool_call_format_non_json` (chat-diff-analyzer.cpp:1003
/// @cd959bae9): the section and per-call markers around a tagged call.
fn analyzeToolCallFormatNonJson(t: *types.AnalyzeTools, a: Allocator, clean_haystack: []const u8, fun_name_needle: []const u8) Error!void {
    const P = struct {
        // first, let's find out if the function is inside a tag or standalone
        fn funMarker(fun: []const u8, b: *B) PegError!ParserId {
            var angle = try b.tag("fun_pre", try b.plus(try b.literal("<"), try b.untilOneOf(&.{ ">", fun })));
            angle = try b.plus(angle, try b.literal(fun));
            var post = try b.plus(try b.negate(try b.plus(try b.space(), try b.literal("<"))), try b.until(">"));
            post = try b.plus(post, try b.literal(">"));
            angle = try b.plus(angle, try b.tag("fun_post", post));
            angle = try b.plus(angle, try b.space());
            var square = try b.tag("fun_pre", try b.plus(try b.literal("["), try b.untilOneOf(&.{ "]", fun })));
            square = try b.plus(square, try b.literal(fun));
            var bracket = try b.plus(try b.plus(try b.space(), try b.literal("[")), try b.until("]"));
            bracket = try b.plus(bracket, try b.literal("]"));
            square = try b.plus(square, try b.tag("fun_post", try b.plus(try b.negate(bracket), try b.space())));
            return b.tag("fun_marker", try b.choice(&.{ angle, square }));
        }

        // now, consume up to two markers, then treat everything up to the function marker as function name prefix
        fn perTool(fun_marker: []const u8, b: *B) PegError!ParserId {
            var x = try b.plus(try b.tag("sec_start", try markerSpace(b)), try b.tag("call_start", try markerSpace(b)));
            x = try b.plus(x, try b.tag("fun_pre", try b.until(fun_marker)));
            x = try b.plusLit(x, fun_marker);
            return b.plus(x, try b.tag("rest", try b.rest()));
        }

        fn section(fun_marker: []const u8, b: *B) PegError!ParserId {
            const x = try b.plusLit(try b.tag("sec_start", try markerSpace(b)), fun_marker);
            return b.plus(x, try b.tag("rest", try b.rest()));
        }

        fn doubleCloser(_: void, b: *B) PegError!ParserId {
            const x = try b.plus(try b.tag("call_end", try markerSpace(b)), try b.tag("sec_end", try markerSpace(b)));
            return b.plus(x, try b.end());
        }

        fn singleCloser(_: void, b: *B) PegError!ParserId {
            return b.plus(try b.tag("sec_end", try markerSpace(b)), try b.end());
        }
    };

    const fun_res = try anywhere(a, fun_name_needle, P.funMarker, clean_haystack);
    var fun_marker = fun_name_needle;
    if (fun_res.ok) {
        fun_marker = fun_res.get("fun_marker");
    }
    var result = try anywhere(a, fun_marker, P.perTool, clean_haystack);
    var result_end: Tagged = undefined;
    if (result.ok) {
        result_end = try anywhere(a, {}, P.doubleCloser, result.get("rest"));
        t.function.name_prefix = try std.mem.concat(a, u8, &.{ fun_res.get("fun_pre"), t.function.name_prefix });
    } else {
        result = try anywhere(a, fun_marker, P.section, clean_haystack);
        result_end = try anywhere(a, {}, P.singleCloser, result.get("rest"));
    }
    t.format.per_call_start = result.get("call_start");
    t.format.per_call_end = result_end.get("call_end");
    t.format.section_start = result.get("sec_start");
    t.format.section_end = result_end.get("sec_end");
}

/// Ports `analyze_tools`'s `check_per_call_markers` (chat-diff-analyzer.cpp:1047
/// @cd959bae9): whether a second call repeats the section markers.
fn checkPerCallMarkers(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call) ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call ++ "," ++ second_tool_call) ++ "]");
    const one_vs_two = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Generating double tool call comparison failed\n\x1b[0m", .{"check_per_call_markers"});
        return;
    };

    const filter_common_call_part = try helpers.calculateDiffSplit(a, one_vs_two.diff.suffix, one_vs_two.diff.right);

    const second_tool_content = trimLeadingWhitespace(filter_common_call_part.right);
    if (t.format.section_start.len != 0 and
        util.find(second_tool_content, t.format.section_start) == 0)
    {
        t.format.per_call_start = t.format.section_start;
        t.format.per_call_end = t.format.section_end;
        t.format.section_start = "";
        t.format.section_end = "";
    }

    if (t.format.per_call_end.len != 0) {
        const calls_one = countOccurrences(one_vs_two.output_A, t.format.per_call_end);
        const calls_two = countOccurrences(one_vs_two.output_B, t.format.per_call_end);
        if (calls_one > 0 and calls_one == calls_two) {
            t.format.section_end = t.format.per_call_end;
            t.format.per_call_end = "";
        }
    }
}

/// `count_occurrences` in `check_per_call_markers`: non-overlapping.
fn countOccurrences(haystack: []const u8, needle: []const u8) usize {
    var count: usize = 0;
    var pos = util.find(haystack, needle);
    while (pos != npos) : (pos = util.findFrom(haystack, needle, pos + needle.len)) count += 1;
    return count;
}

const FunctionParsers = struct {
    fn name(_: void, b: *B) PegError!ParserId {
        const x = try b.plus(try b.tag("pre", try b.until(FUN_FIRST)), try b.literal(FUN_FIRST));
        return b.plus(x, try b.tag("post", try upToMarker(b)));
    }

    fn jsonSuffix(_: void, b: *B) PegError!ParserId {
        const non_json = try b.orr(try b.marker(), try b.plus(try b.plus(try b.negate(try b.literal("{")), try b.negate(try b.literal("["))), try b.any()));
        const after_json = try b.plus(try upToMarker(b), try b.marker());
        return b.plus(try b.tag("ext", try b.zeroOrMore(non_json)), after_json);
    }

    fn taggedSuffix(_: void, b: *B) PegError!ParserId {
        return b.tag("ext", try upToMarker(b));
    }

    fn argStart(b: *B) PegError!ParserId {
        const x = try b.plus(try b.marker(), try b.space());
        return b.plus(x, try b.choice(&.{ try b.literal(ARG_FIRST), try b.literal(ARG_SECOND) }));
    }

    fn sep(_: void, b: *B) PegError!ParserId {
        const body = try b.zeroOrMore(try b.plus(try b.negate(try argStart(b)), try b.any()));
        return b.plus(try b.tag("sep", body), try argStart(b));
    }

    fn closerTagged(_: void, b: *B) PegError!ParserId {
        var x = try b.plus(try b.until("YYYY"), try b.literal("YYYY"));
        x = try b.plus(x, try b.space());
        x = try b.plus(x, try b.marker());
        x = try b.plus(x, try b.space());
        return b.plus(x, try b.tag("close", try b.rest()));
    }

    fn closerJson(_: void, b: *B) PegError!ParserId {
        const x = try b.plus(try b.until("YYYY"), try b.literal("YYYY"));
        return b.plus(x, try b.tag("post_val", try b.rest()));
    }
};

/// Ports `analyze_tools`'s `extract_function_markers` (chat-diff-analyzer.cpp:1103
/// @cd959bae9): the markers around a function's name, before its arguments
/// and after them.
fn extractFunctionMarkers(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call) ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(second_tool_call) ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"extract_function_markers"});
        return;
    };

    const diff = comparison.diff;

    if (util.contains(diff.left, FUN_FIRST) and util.contains(diff.right, FUN_SECOND)) {
        const prefix_marker = if (t.format.per_call_start.len != 0) t.format.per_call_start else t.format.section_start;
        if (prefix_marker.len != 0 and util.rfind(diff.prefix, prefix_marker) != npos) {
            t.function.name_prefix = try util.substrFrom(diff.prefix, util.rfind(diff.prefix, prefix_marker) + prefix_marker.len);
        }

        // Extract name prefix/suffix from diff.left (stop at the next marker boundary)
        const name_result = try extract(a, {}, FunctionParsers.name, diff.left);
        if (name_result.ok) {
            t.function.name_prefix = try std.mem.concat(a, u8, &.{ t.function.name_prefix, name_result.get("pre") });
            t.function.name_suffix = name_result.get("post");
        }

        // Extend name_suffix with content from diff.suffix before args begin
        if (t.format.mode == .TAG_WITH_JSON) {
            // For JSON: name_suffix extends to the first non-marker { or [, including any
            // markers along the way. Only applies if there's at least one marker after
            // the JSON content (matching the original "stop < seg_suf.size() - 1" guard).
            const suf_result = try extract(a, {}, FunctionParsers.jsonSuffix, diff.suffix);
            if (suf_result.ok) {
                t.function.name_suffix = try std.mem.concat(a, u8, &.{ t.function.name_suffix, suf_result.get("ext") });
            }
        } else {
            // For tagged: name_suffix extends to the first marker (arg marker)
            const suf_result = try extract(a, {}, FunctionParsers.taggedSuffix, diff.suffix);
            if (suf_result.ok) {
                t.function.name_suffix = try std.mem.concat(a, u8, &.{ t.function.name_suffix, suf_result.get("ext") });

                const sep_result = try extract(a, {}, FunctionParsers.sep, try util.substrFrom(diff.suffix, suf_result.get("ext").len));
                if (sep_result.ok) {
                    t.function.args_separator = trimWhitespace(sep_result.get("sep"));
                }
            }
        }

        // Extract the closer (between last arg and call/section end marker)
        const suffix_marker = if (t.format.per_call_end.len != 0) t.format.per_call_end else t.format.section_end;
        var closer_suffix: []const u8 = "";
        if (suffix_marker.len == 0) {
            // we'll have to rely on an extra diff with no-calls version
            const nocall = try withMessages(a, params, "[" ++ user_msg ++ "," ++ assistant_content ++ "]");
            if (try compareVariants(a, tmpl, &params, &nocall)) |notool_comp| {
                const nt_diff = notool_comp.diff;
                closer_suffix = try util.substrFrom(nt_diff.left, util.find(nt_diff.left, "YYYY") +% 4);
            }
        } else {
            closer_suffix = try util.substr(diff.suffix, 0, util.find(diff.suffix, suffix_marker));
        }
        if (closer_suffix.len != 0) {
            if (t.format.mode == .TAG_WITH_TAGGED) {
                // After last arg value, skip the closing arg marker, rest is closer
                const close_result = try extract(a, {}, FunctionParsers.closerTagged, closer_suffix);
                if (close_result.ok) {
                    t.function.close = close_result.get("close");
                }
            } else if (t.format.mode == .TAG_WITH_JSON) {
                // After last arg value, find end of JSON args, rest is closer
                const close_result = try extract(a, {}, FunctionParsers.closerJson, closer_suffix);
                if (close_result.ok) {
                    const post = close_result.get("post_val");
                    if (std.mem.lastIndexOfAny(u8, post, "}]")) |pos| {
                        if (pos < post.len -% 1) {
                            t.function.close = trimLeadingWhitespace(post[pos + 1 ..]);
                        }
                    }
                }
            }
        }
        t.function.close = trimLeadingWhitespace(t.function.close);
    }
}

/// Ports `analyze_tools`'s `analyze_arguments` (chat-diff-analyzer.cpp:1246 @cd959bae9).
fn analyzeArguments(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    try extractArgumentNameMarkers(t, a, tmpl);
    try extractArgumentValueMarkers(t, a, tmpl);
}

const ArgumentParsers = struct {
    fn left(_: void, b: *B) PegError!ParserId {
        const x = try b.plus(try b.tag("pre", try b.until(ARG_FIRST)), try b.literal(ARG_FIRST));
        return b.plus(x, try b.tag("suffix", try b.untilOneOf(&.{ "\"", "X" })));
    }

    fn right(_: void, b: *B) PegError!ParserId {
        const x = try b.plus(try b.tag("pre", try b.until(ARG_SECOND)), try b.literal(ARG_SECOND));
        return b.plus(x, try b.tag("suffix", try b.untilOneOf(&.{ "\"", "Y" })));
    }

    fn namePrefix(_: void, b: *B) PegError!ParserId {
        return untilLastMarker(b, "name_prefix");
    }

    fn valPrefix(_: void, b: *B) PegError!ParserId {
        return untilLastMarker(b, "val_prefix");
    }

    fn suffix(_: void, b: *B) PegError!ParserId {
        const x = try b.plus(try upToMarker(b), try b.marker());
        return b.tag("suffix", try b.plus(x, try b.space()));
    }
};

/// Ports `analyze_tools`'s `extract_argument_name_markers` (chat-diff-analyzer.cpp:1251
/// @cd959bae9): the markers around an argument's name.
fn extractArgumentNameMarkers(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call_one_arg) ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call_other_arg) ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"extract_argument_name_markers"});
        return;
    };

    const diff = comparison.diff;

    if (diff.left.len != 0 and diff.right.len != 0) {
        // Parse both sides to find ARG_FIRST/ARG_SECOND and extract the surrounding structure
        const left_result = try anywhere(a, {}, ArgumentParsers.left, diff.left);
        const right_result = try anywhere(a, {}, ArgumentParsers.right, diff.right);

        if (left_result.ok and right_result.ok and
            left_result.get("pre").len != 0 and
            std.mem.eql(u8, left_result.get("pre"), right_result.get("pre")) and
            std.mem.eql(u8, left_result.get("suffix"), right_result.get("suffix")))
        {
            // Name is inside a structure (e.g., JSON key): prefix is the shared wrapper
            t.arguments.name_prefix = left_result.get("pre");
            t.arguments.name_suffix = left_result.get("suffix");
        } else if (std.mem.eql(u8, try util.substr(diff.left, 0, ARG_FIRST.len), ARG_FIRST) and std.mem.eql(u8, try util.substr(diff.right, 0, ARG_SECOND.len), ARG_SECOND)) {
            // Name is directly in the diff: prefix comes from last marker in diff.prefix
            const pre_result = try extract(a, {}, ArgumentParsers.namePrefix, diff.prefix);
            t.arguments.name_prefix = if (pre_result.ok) pre_result.get("name_prefix") else diff.prefix;

            // Suffix extends from after ARG_FIRST to the first marker (+ optional whitespace).
            // The marker could be in diff.left itself or in diff.suffix, so we concatenate.
            const after_first = try std.mem.concat(a, u8, &.{ try util.substrFrom(diff.left, ARG_FIRST.len), diff.suffix });
            const suf_result = try anywhere(a, {}, ArgumentParsers.suffix, after_first);
            if (suf_result.ok) {
                t.arguments.name_suffix = suf_result.get("suffix");
            }
        }
    }
}

/// Ports `analyze_tools`'s `extract_argument_value_markers` (chat-diff-analyzer.cpp:1325
/// @cd959bae9): the markers around an argument's value.
fn extractArgumentValueMarkers(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call_one_arg) ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call_one_arg_other_val) ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"extract_argument_value_markers"});
        return;
    };

    const diff = comparison.diff;

    if (std.mem.eql(u8, diff.left, "XXXX") and std.mem.eql(u8, diff.right, "YYYY")) {
        const arg_name_ending = try std.mem.concat(a, u8, &.{ ARG_FIRST, t.arguments.name_suffix });
        var prefix = diff.prefix;
        if (util.rfind(prefix, arg_name_ending) != npos) {
            prefix = try util.substrFrom(prefix, util.rfind(prefix, arg_name_ending) + arg_name_ending.len);
        }
        if (prefix.len != 0) {
            // Find the last marker + any trailing non-marker text to end
            const pre_result = try extract(a, {}, ArgumentParsers.valPrefix, prefix);
            t.arguments.value_prefix = if (pre_result.ok) pre_result.get("val_prefix") else prefix;
        }

        var value_suffix = diff.suffix;
        if (t.function.close.len != 0) {
            const func_close_pos = util.find(value_suffix, t.function.close);
            if (func_close_pos != npos) {
                value_suffix = value_suffix[0..func_close_pos];
            }
        } else if (t.format.per_call_end.len != 0 or t.format.section_end.len != 0) {
            const end_marker = if (t.format.per_call_end.len != 0) t.format.per_call_end else t.format.section_end;
            const end_marker_pos = util.find(value_suffix, end_marker);
            if (end_marker_pos != npos) {
                value_suffix = value_suffix[0..end_marker_pos];
            }
        }
        if (trimWhitespace(value_suffix).len != 0) {
            t.arguments.value_suffix = value_suffix;
        }
    }
}

/// Ports `analyze_tools`'s `extract_argument_separator` (chat-diff-analyzer.cpp:1390
/// @cd959bae9): what a second argument is preceded by.
fn extractArgumentSeparator(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call_one_arg) ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call) ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"extract_argument_separator"});
        return;
    };

    const diff = comparison.diff;
    if (diff.right.len != 0) {
        const separator = helpers.untilCommonPrefix(diff.right, ARG_FIRST, ARG_SECOND);
        t.arguments.separator = separator;
    }
}

/// Ports `analyze_tools`'s `extract_args_markers` (chat-diff-analyzer.cpp:1425
/// @cd959bae9): markers around a JSON call's arguments.
fn extractArgsMarkers(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call_zero_args) ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call_one_arg) ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed\n\x1b[0m", .{"extract_args_markers"});
        return;
    };

    const diff = comparison.diff;

    if (t.format.mode == .JSON_NATIVE) {
        const prefix_marker = if (t.format.section_start.len != 0) t.format.section_start else t.format.per_call_start;
        const suffix_marker = if (t.format.section_end.len != 0) t.format.section_end else t.format.per_call_end;
        // these might happen earlier in the tools section as an example or somewhere else, so we need to find the closest ones
        var prefix_pos = if (prefix_marker.len == 0) 0 else util.rfind(diff.prefix, prefix_marker);
        var suffix_pos = if (suffix_marker.len == 0) diff.suffix.len else util.find(diff.suffix, suffix_marker);
        if (prefix_pos == npos) prefix_pos = 0;
        if (suffix_pos == npos) suffix_pos = diff.suffix.len;
        const prefix_cut = try util.substrFrom(diff.prefix, prefix_pos + prefix_marker.len);
        const suffix_cut = try util.substr(diff.suffix, 0, suffix_pos);
        var args_start = helpers.untilCommonPrefix(prefix_cut, "{}", "{\"first\":");
        const args_end = helpers.afterCommonSuffix(suffix_cut, "{}", "\"XXXX\"}");

        if (args_start.len != 0 or args_end.len != 0) {
            const find_fun = util.find(args_start, FUN_FIRST);
            if (find_fun != npos) {
                args_start = try util.substr(args_start, find_fun + FUN_FIRST.len, args_start.len - find_fun - FUN_FIRST.len);
            }
            const find_call_id = util.find(args_start, CALL_ID_001);
            if (find_call_id != npos) {
                args_start = try util.substr(args_start, find_call_id + CALL_ID_001.len, args_start.len - find_call_id - CALL_ID_001.len);
            }
            t.arguments.start = args_start;
            t.arguments.end = args_end;
        }
    }
}

const CallIdParsers = struct {
    fn last(_: void, b: *B) PegError!ParserId {
        const l = try lastMarker(b);
        return b.plus(try b.zeroOrMore(try b.plus(try b.negate(l), try b.any())), try b.tag("m", try b.marker()));
    }

    fn first(_: void, b: *B) PegError!ParserId {
        return b.tag("m", try b.marker());
    }

    fn idPrefix(common_id_part: []const u8, b: *B) PegError!ParserId {
        const x = try b.tag("prefix", try b.marker());
        const skip = try b.plus(try b.plus(try b.negate(try b.marker()), try b.negate(try b.literal(common_id_part))), try b.any());
        return b.plus(try b.plus(x, try b.zeroOrMore(skip)), try b.literal(common_id_part));
    }

    fn suffix(_: void, b: *B) PegError!ParserId {
        const skip = try b.plus(try b.plus(try b.negate(try b.marker()), try b.negate(try b.literal("{"))), try b.any());
        return b.plus(try b.zeroOrMore(skip), try b.tag("suffix", try b.marker()));
    }
};

/// `find_last_marker` in `extract_call_id_markers`: the last marker in
/// `str`, without the text after it.
fn findLastMarker(a: Allocator, str: []const u8) Error![]const u8 {
    const res = try anywhere(a, {}, CallIdParsers.last, str);
    return if (res.ok) res.get("m") else "";
}

/// `find_first_marker` in `extract_call_id_markers`.
fn findFirstMarker(a: Allocator, str: []const u8) Error![]const u8 {
    const res = try anywhere(a, {}, CallIdParsers.first, str);
    return if (res.ok) res.get("m") else "";
}

/// Ports `analyze_tools`'s `extract_call_id_markers` (chat-diff-analyzer.cpp:1486
/// @cd959bae9): where a call's id sits, and the markers around it.
fn extractCallIdMarkers(t: *types.AnalyzeTools, a: Allocator, tmpl: *const Template) Error!void {
    const params = try probe(a, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call) ++ "]", true);
    const b = try withMessages(a, params, "[" ++ user_msg ++ "," ++ comptime assistantCalls(first_tool_call_alt_id) ++ "]");
    const comparison = try compareVariants(a, tmpl, &params, &b) orelse {
        log.dbg("\x1b[1m\x1b[38;5;214m{s}: Template application failed for call_id detection\n\x1b[0m", .{"extract_call_id_markers"});
        return;
    };

    const diff = comparison.diff;

    if (diff.left.len == 0 and diff.right.len == 0) return;

    const id_value_1 = CALL_ID_001;
    const id_value_2 = CALL_ID_999;

    var common_id_prefix_len: usize = 0;
    for (0..@min(id_value_1.len, id_value_2.len)) |i| {
        if (id_value_1[i] == id_value_2[i]) {
            common_id_prefix_len += 1;
        } else {
            break;
        }
    }
    const common_id_part = id_value_1[0..common_id_prefix_len];

    // Check if the function name is in the prefix (normal case: BETWEEN_FUNC_AND_ARGS or POST_ARGS)
    // or in the suffix (call_id is PRE_FUNC_NAME)
    const func_name = FUN_FIRST;
    const func_name_in_prefix = util.rfind(diff.prefix, func_name);
    const func_name_in_suffix = util.find(diff.suffix, func_name);

    if (func_name_in_prefix != npos and func_name_in_suffix == npos) {
        // Function name is only in prefix - call_id is BETWEEN_FUNC_AND_ARGS or POST_ARGS
        // Check if args indicator "{" is in prefix or suffix
        const args_in_prefix = if (std.mem.indexOfScalarPos(u8, diff.prefix, func_name_in_prefix, '{')) |i| i else npos;
        const args_in_suffix = if (std.mem.indexOfScalar(u8, diff.suffix, '{')) |i| i else npos;

        if (args_in_suffix != npos and
            (args_in_prefix == npos or args_in_prefix > diff.prefix.len))
        {
            // Args are in suffix, so call_id is BETWEEN_FUNC_AND_ARGS
            t.call_id.pos = .BETWEEN_FUNC_AND_ARGS;

            // Find call_id_prefix: marker immediately preceding common_id_part (no intervening markers)
            const after_func = try util.substrFrom(diff.prefix, func_name_in_prefix + func_name.len);
            const id_res = try anywhere(a, common_id_part, CallIdParsers.idPrefix, after_func);
            if (id_res.ok) {
                t.call_id.prefix = id_res.get("prefix");
            } else {
                // Fallback: use the last marker in after_func
                t.call_id.prefix = try findLastMarker(a, after_func);
            }

            // Extract call_id_suffix: the first marker in the suffix before args "{"
            const suf_res = try anywhere(a, {}, CallIdParsers.suffix, diff.suffix);
            if (suf_res.ok) {
                t.call_id.suffix = suf_res.get("suffix");
            }
        } else if (args_in_prefix != npos) {
            // Args are in prefix, so call_id is POST_ARGS
            t.call_id.pos = .POST_ARGS;

            // Extract last marker between args closing brace and the ID
            const after_args = try util.substrFrom(diff.prefix, args_in_prefix);
            if (std.mem.lastIndexOfScalar(u8, after_args, '}')) |closing_brace| {
                const between_args_and_id = after_args[closing_brace + 1 ..];
                t.call_id.prefix = try findLastMarker(a, between_args_and_id);
            }

            // call_id_suffix: first marker in diff.suffix
            t.call_id.suffix = try findFirstMarker(a, diff.suffix);
        }
    } else if (func_name_in_suffix != npos and func_name_in_prefix == npos) {
        // Function name is only in suffix - call_id is PRE_FUNC_NAME
        t.call_id.pos = .PRE_FUNC_NAME;

        // call_id_prefix: last marker in diff.prefix
        t.call_id.prefix = try findLastMarker(a, diff.prefix);

        // call_id_suffix: first marker in the portion of diff.suffix before func_name
        const before_func = diff.suffix[0..func_name_in_suffix];
        t.call_id.suffix = try findFirstMarker(a, before_func);
    }

    if (std.mem.eql(u8, t.call_id.prefix, t.arguments.end)) {
        t.call_id.prefix = "";
    }

    if (std.mem.eql(u8, t.call_id.suffix, t.arguments.start)) {
        t.call_id.suffix = "";
    }

    // When call_id is detected, per_call_end may have been incorrectly set to include
    // the call_id_suffix and sample args. Clear it if it starts with call_id_suffix.
    if (t.call_id.pos != .NONE and t.call_id.suffix.len != 0 and
        util.find(t.format.per_call_end, t.call_id.suffix) == 0)
    {
        t.format.per_call_end = "";
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the probe JSON parses, in the C++'s member order" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var v = try json.parse(a, probe_tools);
    const s = try json.dump(a, &v, -1);
    try std.testing.expect(std.mem.startsWith(u8, s, "[{\"type\":\"function\",\"function\":{\"name\":\"FFF_FIRST_FUN_F\""));
}
