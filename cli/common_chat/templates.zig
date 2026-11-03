//! A model's chat templates, and applying them: a conversation into a
//! prompt, with the grammar, triggers and parser its reply is read with.
//!
//! # Provenance
//!
//! Ports the template half of `llama.cpp/common/chat.cpp` (llama.cpp v0.6.0,
//! `d81235049`): `common_chat_templates` and its init, free, source and
//! explicit flag; `common_chat_templates_inputs`; `common_chat_verify_template`;
//! `common_chat_format_single` and `common_chat_format_example`;
//! `common_chat_templates_get_caps`;
//! `common_chat_templates_support_enable_thinking`;
//! `common_chat_get_asr_prompt`; the `workaround` namespace;
//! `common_chat_try_specialized_template`; and
//! `common_chat_templates_apply` with its Jinja and legacy arms. From
//! `llama.cpp/common/common.cpp`, `common_token_to_piece`. Each declaration
//! names what it ports and the line it began at.
//!
//! # Ownership
//!
//! `Templates` owns its two templates, released with `deinit`. Every other
//! function takes an arena `a` and returns what it built in it.
//!
//! # libllama
//!
//! The model is reached through libllama's port, `@import("llama").api`,
//! never a `llama_*` C symbol: in a harness the C symbols are upstream's
//! (`CLAUDE.md`, the libmtmd note).

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("exception.zig");
const msg_mod = @import("msg.zig");
const params_mod = @import("params.zig");
const template = @import("template.zig");
const caps_mod = @import("caps.zig");
const chat_peg = @import("chat_peg.zig");
const util = @import("util.zig");
const log = @import("log.zig");
const parsers = @import("parsers/module.zig");
const autoparser = @import("autoparser/module.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const throw = exception.throw;
const Msg = msg_mod.Msg;
const Template = template.Template;
const GenerationParams = template.GenerationParams;
const ChatParams = params_mod.ChatParams;
const Caps = caps_mod.Caps;

/// One `chat_template_kwargs` entry: a name and the JSON text of its value.
pub const KwArg = struct { key: []const u8, value: []const u8 };

/// Ports `common_chat_templates_inputs` (chat.h:249 @d81235049). Everything
/// is borrowed for the duration of the call it is passed to.
pub const TemplatesInputs = struct {
    messages: []const Msg = &.{},
    grammar: []const u8 = "",
    json_schema: []const u8 = "",
    add_generation_prompt: bool = true,
    continue_final_message: msg_mod.Continuation = .none,
    use_jinja: bool = true,
    // Parameters below only supported when use_jinja is true
    tools: []const msg_mod.Tool = &.{},
    tool_choice: msg_mod.ToolChoice = .auto,
    parallel_tool_calls: bool = false,
    reasoning_format: msg_mod.ReasoningFormat = .none, // TODO: refactor this to "bool enable_thinking"
    enable_thinking: bool = true,
    /// A `std`'s `map`: iterated sorted by key, which `applyJinja` does whatever
    /// order this is in. A repeated key is the map's one entry, the last.
    chat_template_kwargs: []const KwArg = &.{},
    add_bos: bool = false,
    add_eos: bool = false,
    force_pure_content: bool = false,
};

/// What `common_chat_templates_init` reads off a model: its templates and
/// its vocabulary's BOS and EOS. A field is null where libllama has none.
pub const ModelInfo = struct {
    /// `llama_model_chat_template(model, nullptr)`
    chat_template: ?[]const u8 = null,
    /// `llama_model_chat_template(model, "tool_use")`
    tool_use_template: ?[]const u8 = null,
    /// `common_token_to_piece(vocab, llama_vocab_bos(vocab), true)`; null
    /// for `LLAMA_TOKEN_NULL`
    bos: ?[]const u8 = null,
    eos: ?[]const u8 = null,
    add_bos: bool = false,
    add_eos: bool = false,
};

/// Ports `common_token_to_piece` (common/common.cpp:1841 @d81235049), the
/// `special` arm: a token's text through libllama's port.
///
/// Return: owned by `a`.
fn tokenToPiece(a: Allocator, vocab: anytype, token: i32) Error![]u8 {
    const api = @import("llama").api;
    var buf = try a.alloc(u8, 16);
    var n = api.vocab.llama_token_to_piece(vocab, token, buf.ptr, @intCast(buf.len), 0, true);
    if (n < 0) {
        buf = try a.alloc(u8, @intCast(-n));
        const check = api.vocab.llama_token_to_piece(vocab, token, buf.ptr, @intCast(buf.len), 0, true);
        util.ggmlAssert(check == -n, "check == -n_chars");
        n = check;
    }
    return buf[0..@intCast(n)];
}

/// Reads a loaded model's `ModelInfo` through libllama's port.
///
/// Parameters:
/// - `a`: owns the strings returned.
/// - `model`: a `llama_model *`, from whichever `llama.h` import the
///   caller has; it is only passed through.
pub fn modelInfo(a: Allocator, model: *const anyopaque) Error!ModelInfo {
    const api = @import("llama").api;
    const ModelPtr = @typeInfo(@TypeOf(api.model.llama_model_chat_template)).@"fn".param_types[0].?;
    const m: ModelPtr = @ptrCast(@alignCast(model));
    var info: ModelInfo = .{};
    if (api.model.llama_model_chat_template(m, null)) |s| info.chat_template = try a.dupe(u8, std.mem.span(@as([*:0]const u8, @ptrCast(s))));
    if (api.model.llama_model_chat_template(m, "tool_use")) |s| info.tool_use_template = try a.dupe(u8, std.mem.span(@as([*:0]const u8, @ptrCast(s))));
    const vocab = api.model.llama_model_get_vocab(m);
    const bos = api.vocab.llama_vocab_bos(vocab);
    const eos = api.vocab.llama_vocab_eos(vocab);
    if (bos != -1) info.bos = try tokenToPiece(a, vocab, bos);
    if (eos != -1) info.eos = try tokenToPiece(a, vocab, eos);
    info.add_bos = api.vocab.llama_vocab_get_add_bos(vocab);
    info.add_eos = api.vocab.llama_vocab_get_add_eos(vocab);
    return info;
}

/// Ports `CHATML_TEMPLATE_SRC` (chat.cpp:715 @d81235049).
pub const chatml_template_src =
    "{%- for message in messages -%}\n" ++
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" ++
    "{%- endfor -%}\n" ++
    "{%- if add_generation_prompt -%}\n" ++
    "  {{- '<|im_start|>assistant\n' -}}\n" ++
    "{%- endif -%}";

/// Ports `common_chat_templates` (chat.cpp:337 @d81235049).
pub const Templates = struct {
    gpa: Allocator,
    add_bos: bool,
    add_eos: bool,
    /// Model had builtin template or template overridden was specified.
    has_explicit_template: bool,
    /// always set (defaults to chatml)
    template_default: Template,
    template_tool_use: ?Template,

    /// Ports `common_chat_templates_free` (chat.cpp:723 @d81235049).
    pub fn deinit(self: *Templates) void {
        self.template_default.deinit();
        if (self.template_tool_use) |*t| t.deinit();
        self.* = undefined;
    }
};

/// Ports `common_chat_templates_init` (chat.cpp:757 @d81235049): the model's
/// templates, or the override, with the vocabulary's BOS and EOS, and two
/// known template bugs patched.
///
/// Parameters:
/// - `gpa`: owns the result; release with `Templates.deinit`.
/// - `model`: what the model holds, or null without one -- which, like
///   upstream's `GGML_ASSERT(model != nullptr)`, needs an override.
/// - `chat_template_override`: a template's source, or "" for the model's.
///
/// Return: `error.RuntimeError` when the default template does not parse,
/// its `what()` "std::exception", as the C++'s `throw e;` of a
/// `const std::exception &` slices it.
pub fn init(gpa: Allocator, model: ?*const ModelInfo, chat_template_override: []const u8, bos_token_override: []const u8, eos_token_override: []const u8) Error!Templates {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var default_template_src: []const u8 = "";
    var template_tool_use_src: []const u8 = "";

    var has_explicit_template = chat_template_override.len != 0;
    if (chat_template_override.len == 0) {
        util.ggmlAssert(model != null, "model != nullptr");
        const info = model.?;
        if (info.chat_template) |s| {
            default_template_src = s;
            has_explicit_template = true;
        }
        if (info.tool_use_template) |s| {
            template_tool_use_src = s;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.len == 0 or std.mem.eql(u8, default_template_src, "chatml")) {
        if (template_tool_use_src.len != 0) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = chatml_template_src;
        }
    }

    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (util.contains(default_template_src, "<|channel|>") and
        // search for the error message and patch it
        util.contains(default_template_src, "in message.content or"))
    {
        default_template_src = try util.replaceAll(a, default_template_src, "{%- if \"<|channel|>analysis<|message|>\" in message.content or " ++
            "\"<|channel|>final<|message|>\" in message.content %}", "{%- if false %}");
    }

    // TODO @aldehir : this is a temporary fix, pending Minja changes
    // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
    if (util.contains(default_template_src, "[TOOL_CALLS]") and
        // search for the error message and patch it
        util.contains(default_template_src, "if (message['content'] is none or"))
    {
        default_template_src = try util.replaceAll(a, default_template_src, "{%- if (message['content'] is none or message['content'] == '' or " ++
            "message['content']|length == 0) and (message['tool_calls'] is not defined or " ++
            "message['tool_calls'] is none or message['tool_calls']|length == 0) %}", "{%- if false %}");
    }

    var token_bos = bos_token_override;
    var token_eos = eos_token_override;
    var add_bos = false;
    var add_eos = false;
    if (model) |info| {
        token_bos = try vocabToken(info.bos, "BOS", "bos_token", default_template_src, template_tool_use_src);
        token_eos = try vocabToken(info.eos, "EOS", "eos_token", default_template_src, template_tool_use_src);
        add_bos = info.add_bos;
        add_eos = info.add_eos;
    }

    const template_default = Template.init(gpa, default_template_src, token_bos, token_eos) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        log.err("{s}: error: {s}\n", .{ "common_chat_templates_init", exception.what(e) });
        log.err("{s}: failed to initialize chat template\n", .{"common_chat_templates_init"});
        log.err("{s}: please consider disabling jinja via --no-jinja, or using another chat template\n", .{"common_chat_templates_init"});
        return throw(error.RuntimeError, "std::exception", .{});
    };
    var tmpls: Templates = .{
        .gpa = gpa,
        .add_bos = add_bos,
        .add_eos = add_eos,
        .has_explicit_template = has_explicit_template,
        .template_default = template_default,
        .template_tool_use = null,
    };
    errdefer tmpls.deinit();
    if (template_tool_use_src.len != 0) {
        tmpls.template_tool_use = Template.init(gpa, template_tool_use_src, token_bos, token_eos) catch |e| blk: {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            log.err("{s}: failed to parse tool use chat template (ignoring it): {s}\n", .{ "common_chat_templates_init", exception.what(e) });
            break :blk null;
        };
    }
    return tmpls;
}

/// `get_token` in `common_chat_templates_init`: the piece, or "" with a
/// warning where a template names a token the vocabulary lacks.
fn vocabToken(piece: ?[]const u8, name: []const u8, jinja_variable_name: []const u8, default_src: []const u8, tool_use_src: []const u8) Error![]const u8 {
    if (piece) |p| return p;
    if (util.contains(default_src, jinja_variable_name) or util.contains(tool_use_src, jinja_variable_name)) {
        log.wrn("common_chat_templates_init: warning: vocab does not have a {s} token, jinja template won't work as intended.\n", .{name});
    }
    return "";
}

/// Ports `common_chat_templates_was_explicit` (chat.cpp:727 @d81235049).
pub fn wasExplicit(tmpls: *const Templates) bool {
    return tmpls.has_explicit_template;
}

/// Ports `common_chat_templates_source` (chat.cpp:744 @d81235049): the
/// default template's source, or the `tool_use` one's ("" without one).
pub fn source(tmpls: *const Templates, variant: []const u8) []const u8 {
    if (variant.len != 0) {
        if (std.mem.eql(u8, variant, "tool_use")) {
            if (tmpls.template_tool_use) |*t| return t.source();
            return "";
        }
        log.dbg("{s}: unknown template variant: {s}\n", .{ "common_chat_templates_source", variant });
    }
    return tmpls.template_default.source();
}

/// Ports `common_chat_templates_get_caps` (chat.cpp:1531 @d81235049): the
/// tool-use template's capabilities when there is one, the default's
/// otherwise, sorted by name as the `std`'s `map` is.
pub fn getCaps(tmpls: *const Templates) [9]Caps.Entry {
    if (tmpls.template_tool_use) |*t| {
        // take the more expressive template when available
        return t.caps.toMap();
    }
    return tmpls.template_default.caps.toMap();
}

/// `caps[name]` of `getCaps`: false for a name it lacks, as `std`'s `map`'s
/// `operator[]` default-inserts.
pub fn capsValue(entries: [9]Caps.Entry, name: []const u8) bool {
    for (entries) |e| if (std.mem.eql(u8, e.name, name)) return e.value;
    return false;
}

/// Ports `common_chat_templates_support_enable_thinking` (chat.cpp:358
/// @d81235049): whether one user turn applied with thinking on reports
/// that the template thinks.
pub fn supportEnableThinking(gpa: Allocator, tmpls: *const Templates) Error!bool {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const msg: Msg = .{ .role = "user", .content = "test" };
    const inputs: TemplatesInputs = .{
        .messages = &.{msg},
        .enable_thinking = true,
        .add_generation_prompt = true,
        .reasoning_format = .deepseek,
    };
    const params = try apply(a, tmpls, &inputs);
    return params.supports_thinking;
}

/// Ports `common_chat_prompt_preset` (chat.h:384 @d81235049).
pub const PromptPreset = struct {
    system: []const u8,
    user: []const u8,
};

/// Ports `common_chat_get_asr_prompt` (chat.cpp:731 @d81235049).
pub fn getAsrPrompt(tmpls: ?*const Templates) PromptPreset {
    if (tmpls) |t| {
        if (parsers.lfm2.isLfm2Template(t.template_default.source())) {
            return .{ .system = "Perform ASR.", .user = "" };
        }
    }
    return .{ .system = "", .user = "Transcribe audio to text" };
}

/// Ports `common_chat_verify_template` (chat.cpp:638 @d81235049): whether
/// one user turn applies, through Jinja or through libllama's built-in
/// template families.
pub fn verifyTemplate(gpa: Allocator, tmpl: []const u8, use_jinja: bool) Error!bool {
    if (use_jinja) {
        verifyJinja(gpa, tmpl) catch |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            log.err("{s}: failed to apply template: {s}\n", .{ "common_chat_verify_template", exception.what(e) });
            return false;
        };
        return true;
    }
    const api = @import("llama").api;
    const Message = std.meta.Child(@typeInfo(@TypeOf(api.llama.llama_chat_apply_template)).@"fn".param_types[1].?);
    const chat = [_]Message{.{ .role = "user", .content = "test" }};
    const z = try gpa.dupeSentinel(u8, tmpl, 0);
    defer gpa.free(z);
    const res = api.llama.llama_chat_apply_template(z.ptr, &chat, 1, true, null, 0);
    return res >= 0;
}

/// The `try` block of `common_chat_verify_template`'s Jinja arm.
fn verifyJinja(gpa: Allocator, tmpl: []const u8) Error!void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var tmpls = try init(gpa, null, tmpl, "", "");
    defer tmpls.deinit();
    const msg: Msg = .{ .role = "user", .content = "test" };
    const inputs: TemplatesInputs = .{ .messages = &.{msg} };
    _ = try apply(arena_state.allocator(), &tmpls, &inputs);
}

/// Ports `common_chat_format_single` (chat.cpp:664 @d81235049): what adding
/// `new_msg` to `past_msg` adds to the rendered conversation.
///
/// Return: owned by the arena `a`.
pub fn formatSingle(a: Allocator, tmpls: *const Templates, past_msg: []const Msg, new_msg: Msg, add_ass: bool, use_jinja: bool) Error![]const u8 {
    var inputs: TemplatesInputs = .{
        .use_jinja = use_jinja,
        .add_bos = tmpls.add_bos,
        .add_eos = tmpls.add_eos,
    };
    var fmt_past_msg: []const u8 = "";
    if (past_msg.len != 0) {
        inputs.messages = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg = (try apply(a, tmpls, &inputs)).prompt;
    }
    var ss: std.ArrayList(u8) = .empty;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass and fmt_past_msg.len != 0 and fmt_past_msg[fmt_past_msg.len - 1] == '\n') {
        try ss.appendSlice(a, "\n");
    }
    // format chat with new_msg
    const all = try a.alloc(Msg, past_msg.len + 1);
    @memcpy(all[0..past_msg.len], past_msg);
    all[past_msg.len] = new_msg;
    inputs.messages = all;
    inputs.add_generation_prompt = add_ass;
    const fmt_new_msg = (try apply(a, tmpls, &inputs)).prompt;
    // get the diff part
    try ss.appendSlice(a, try util.substr(fmt_new_msg, fmt_past_msg.len, fmt_new_msg.len -% fmt_past_msg.len));
    return ss.items;
}

/// Ports `common_chat_format_example` (chat.cpp:694 @d81235049): a short
/// fixed conversation rendered, for the log.
///
/// Return: owned by the arena `a`.
pub fn formatExample(a: Allocator, tmpls: *const Templates, use_jinja: bool, chat_template_kwargs: []const KwArg) Error![]const u8 {
    const msgs = [_]Msg{
        .{ .role = "system", .content = "You are a helpful assistant" },
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there" },
        .{ .role = "user", .content = "How are you?" },
    };
    const inputs: TemplatesInputs = .{
        .use_jinja = use_jinja,
        .add_bos = tmpls.add_bos,
        .add_eos = tmpls.add_eos,
        .chat_template_kwargs = chat_template_kwargs,
        .messages = &msgs,
    };
    return (try apply(a, tmpls, &inputs)).prompt;
}

// -----------------------------------------------------------------------------
// workaround

/// `j == "s"`.
fn isStr(v: *const Value, s: []const u8) bool {
    return v.isString() and std.mem.eql(u8, v.string, s);
}

/// Ports `workaround`'s `map_developer_role_to_system` (chat.cpp:1004 @d81235049).
fn mapDeveloperRoleToSystem(a: Allocator, messages: *Value) Error!void {
    var it = messages.items();
    while (it.next()) |item| {
        const message = item.value;
        if (message.contains("role")) {
            const role = message.index(a, "role") catch |e| return util.fromJson(e);
            if (isStr(role, "developer")) role.assign(a, try Value.from(a, "system"));
        }
    }
}

/// Ports `workaround`'s `system_message_not_supported` (chat.cpp:1016
/// @d81235049): a leading system message merged into the next one, or
/// dropped when it is alone.
fn systemMessageNotSupported(a: Allocator, messages: *Value) Error!void {
    if (messages.empty()) return;
    const first = messages.front() catch |e| return util.fromJson(e);
    const role = first.at("role") catch |e| return util.fromJson(e);
    if (!isStr(role, "system")) return;
    if (messages.size() > 1) {
        log.dbg("Merging system prompt into next message\n", .{});
        const first_content = (first.at("content") catch |e| return util.fromJson(e)).get([]const u8) catch |e| return util.fromJson(e);
        const second = messages.indexAt(a, 1) catch |e| return util.fromJson(e);
        const second_content = (second.at("content") catch |e| return util.fromJson(e)).get([]const u8) catch |e| return util.fromJson(e);
        const merged = try std.mem.concat(a, u8, &.{ first_content, "\n", second_content });
        const slot = second.index(a, "content") catch |e| return util.fromJson(e);
        slot.assign(a, try Value.from(a, merged));
        messages.eraseIndex(a, 0) catch |e| return util.fromJson(e);
    } else {
        log.wrn("Removing system prompt due to template not supporting system role\n", .{});
        messages.eraseIndex(a, 0) catch |e| return util.fromJson(e);
    }
}

/// Ports `workaround`'s `requires_non_null_content` (chat.cpp:1032 @d81235049).
fn requiresNonNullContent(a: Allocator, messages: *Value) Error!void {
    util.ggmlAssert(messages.isArray(), "messages.is_array()");
    var it = messages.items();
    while (it.next()) |item| {
        const message = item.value;
        if (message.contains("tool_calls") and !message.contains("content")) {
            const slot = message.index(a, "content") catch |e| return util.fromJson(e);
            slot.assign(a, try Value.from(a, ""));
        }
    }
}

/// Ports `workaround`'s `func_args_not_string` (chat.cpp:1041 @d81235049): a
/// tool call's string arguments parsed, for a template that reads them as
/// an object.
fn funcArgsNotString(a: Allocator, messages: *Value) Error!void {
    util.ggmlAssert(messages.isArray(), "messages.is_array()");
    var it = messages.items();
    while (it.next()) |item| {
        const message = item.value;
        if (!message.contains("tool_calls")) continue;
        const tool_calls = message.index(a, "tool_calls") catch |e| return util.fromJson(e);
        var calls = tool_calls.items();
        while (calls.next()) |call| {
            const tool_call = call.value;
            if (!tool_call.contains("function")) continue;
            const function = tool_call.index(a, "function") catch |e| return util.fromJson(e);
            if (!function.contains("arguments")) continue;
            const args = function.index(a, "arguments") catch |e| return util.fromJson(e);
            if (args.isString()) {
                const parsed = json.parse(a, args.string) catch |e| {
                    return throw(error.RuntimeError, "Failed to parse tool call arguments as JSON: {s}", .{exception.what(util.fromJson(e))});
                };
                args.assign(a, parsed);
            }
        }
    }
}

/// Ports `workaround`'s `trim_all_content` (chat.cpp:1066 @d81235049): each
/// message's content, reasoning and text parts trimmed.
///
/// Return: copies owned by `a`.
fn trimAllContent(a: Allocator, messages: []const Msg) Error![]Msg {
    const helpers = autoparser.helpers;
    const out = try a.alloc(Msg, messages.len);
    for (messages, out) |m, *o| {
        o.* = m;
        o.content = helpers.trimWhitespace(m.content);
        o.reasoning_content = helpers.trimWhitespace(m.reasoning_content);
        o.content_parts = .empty;
        for (m.content_parts.items) |part| {
            var p = part;
            if (std.mem.eql(u8, part.type, "text")) p.text = helpers.trimWhitespace(part.text);
            try o.content_parts.append(a, p);
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// apply

/// Ports `common_chat_try_specialized_template` (chat.cpp:1090 @d81235049):
/// a hand-written parser for the templates the auto-parser cannot read,
/// chosen by what the source holds.
///
/// Return: null when none applies.
pub fn trySpecializedTemplate(a: Allocator, tmpl: *const Template, src: []const u8, params: *GenerationParams) Error!?ChatParams {
    const has = util.contains;
    // Ministral/Mistral Large 3 - uses special reasoning structure fixes, can't use autoparser
    // Note: Mistral Small 3.2 uses [CALL_ID] which Ministral doesn't have, so we can distinguish them
    if (has(src, "[SYSTEM_PROMPT]") and has(src, "[TOOL_CALLS]") and
        has(src, "[ARGS]") and !has(src, "[CALL_ID]"))
    {
        log.dbg("Using specialized template: Ministral/Magistral Large 3\n", .{});
        return try parsers.ministral3.init(a, tmpl, params);
    }

    // LLM-jp-4.1 - GPT-OSS dialect (spaces after special tokens, <|end|>-separated parallel calls)
    if (has(src, "chat_format=llm-jp-harmony-v1")) {
        log.dbg("Using specialized template: LLM-jp Harmony v1\n", .{});
        return try parsers.llm_jp_harmony.init(a, tmpl, params);
    }

    // GPT-OSS - has unique channel-based structure that needs dedicated handler
    if (has(src, "<|channel|>")) {
        log.dbg("Using specialized template: GPT-OSS\n", .{});
        return try parsers.gpt_oss.init(a, tmpl, params);
    }

    // Muse Glimmer format using " to=<recipient>" recipients and <|eom|>/<|eot|> message terminators.
    if (has(src, "<atem:function_calls>") and has(src, "<|eom|>")) {
        log.dbg("Using specialized template: Muse Glimmer\n", .{});
        return try parsers.muse_glimmer.init(a, tmpl, params);
    }

    // Functionary v3.2 - uses recipient-based format with >>>recipient\n{content}
    // Detection: template has ">>>all" for content and ">>>" prefix for tool calls
    if (has(src, ">>>all") and has(src, ">>>${recipient}")) {
        log.dbg("Using specialized template: Functionary v3.2\n", .{});
        return try parsers.functionary_v3_2.init(a, tmpl, params);
    }

    // Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
    // Detection: template has "<|tool_calls_section_begin|>" and "functions." prefix in tool call IDs
    if (has(src, "<|tool_calls_section_begin|>") and
        has(src, "<|tool_call_begin|>"))
    {
        log.dbg("Using specialized template: Kimi K2 Thinking\n", .{});
        return try parsers.kimi_k2.init(a, tmpl, params);
    }

    // Kimi K3 - the <|open|>/<|close|>/<|end_of_msg|> markers are unique to it
    if (has(src, "<|open|>") and has(src, "<|close|>") and
        has(src, "<|end_of_msg|>"))
    {
        log.dbg("Using specialized template: Kimi K3\n", .{});
        return try parsers.kimi_k3.init(a, tmpl, params);
    }

    // Ling 3.0 / Bailing V3 - <role>X</role> sections with <arg_key>/<arg_value> tagged
    // tool calls. <role> sections are unique to this family among the tagged-arg templates.
    if (has(src, "<role>ASSISTANT</role>") and
        has(src, "<arg_key>"))
    {
        log.dbg("Using specialized template: Ling 3.0 (Bailing V3)\n", .{});
        return try parsers.ling3.init(a, tmpl, params);
    }

    // Cohere2 MoE / North Code - marker-wrapped format with <|START_TEXT|> content and
    // <|START_ACTION|> JSON tool calls. <|START_TEXT|> is unique to this template (the older
    // Command-R templates use <|START_RESPONSE|>).
    if (has(src, "<|START_TEXT|>") and
        has(src, "<|START_ACTION|>"))
    {
        log.dbg("Using specialized template: Cohere2 MoE\n", .{});
        return try parsers.cohere2moe.init(a, tmpl, params);
    }

    if (parsers.lfm2.isLfm2Template(src)) {
        log.dbg("Using specialized template: LFM2\n", .{});
        return try parsers.lfm2.init(a, tmpl, params, true);
    }

    // LFM2.5 format detection: template uses plain "List of tools: [...]" with no special tokens
    if (has(src, "List of tools: [") and
        !has(src, "<|tool_list_start|>"))
    {
        log.dbg("Using specialized template: LFM2.5\n", .{});
        return try parsers.lfm2.init(a, tmpl, params, false);
    }

    // GigaChatV3 format detection
    if (has(src, "<|role_sep|>") and
        has(src, "<|message_sep|>") and
        !has(src, "<|function_call|>"))
    {
        log.dbg("Using specialized template: GigaChatV3\n", .{});
        return try parsers.gigachat_v3.init(a, tmpl, params);
    }

    // MiniMax-M3: the namespace token "]<]minimax[>[" collides with the autoparser's
    // markup delimiters, so detect the template and use a dedicated parser.
    if (has(src, "]<]minimax[>[") and
        has(src, "<tool_call>") and
        has(src, "<invoke name="))
    {
        log.dbg("Using specialized template: MiniMax-M3\n", .{});
        return try parsers.minimax_m3.init(a, tmpl, params);
    }

    // DeepSeek V3.2/V4 format detection: template defines dsml_token and uses it for tool calls.
    // The template source contains the token as a variable assignment, not as a literal in markup.
    // V3.2 names the tool call block "function_calls", V4 names it "tool_calls".
    if (has(src, "dsml_token") and
        has(src, "DSML") and
        (has(src, "function_calls") or
            has(src, "tool_calls")))
    {
        log.dbg("Using specialized template: DeepSeek V3.2/V4\n", .{});
        return try parsers.deepseek.init(a, tmpl, params);
    }

    // Gemma4 format detection
    if (has(src, "'<|tool_call>call:'")) {
        if (!has(src, "{#- OpenAI Chat Completions:")) {
            // apply workarounds if using the older gemma4 templates
            log.wrn("{s}: detected an outdated gemma4 chat template, applying compatibility workarounds. " ++
                "Consider updating to the official template.\n", .{"common_chat_try_specialized_template"});
            try parsers.gemma4.convertToolResponses(a, &params.messages);
        }
        return try parsers.gemma4.init(a, tmpl, params);
    }

    // MiniCPM5 - XML tool calls with <function name="..."><param name="...">...</param></function>
    if (has(src, "Tool usage guidelines:") and
        has(src, "<function name=\"") and
        has(src, "<param name=\""))
    {
        log.dbg("Using specialized template: MiniCPM5\n", .{});
        return try parsers.minicpm5.init(a, tmpl, params);
    }

    // Qwen3-Coder XML tool calls, also used by Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash
    if (has(src, "<tool_call>") and
        has(src, "<function=") and
        has(src, "<parameter=") and
        // Exclude models that don't use \n between tags
        !has(src, "'<tool_call><function=' ~ tool_call.name ~ '>'"))
    {
        log.dbg("Using specialized template: Qwen3-Coder\n", .{});
        return try parsers.qwen3_coder.init(a, tmpl, params);
    }

    return null;
}

/// `std::map<std::string, std::string>`'s iteration: by key, bytewise; a
/// repeated key keeps its last value.
fn sortedKwargs(a: Allocator, kwargs: []const KwArg) Error![]KwArg {
    const out = try a.alloc(KwArg, kwargs.len);
    var n: usize = 0;
    outer: for (kwargs) |kv| {
        for (out[0..n]) |*e| if (std.mem.eql(u8, e.key, kv.key)) {
            e.value = kv.value;
            continue :outer;
        };
        out[n] = kv;
        n += 1;
    }
    const sorted = out[0..n];
    std.mem.sort(KwArg, sorted, {}, struct {
        fn lt(_: void, x: KwArg, y: KwArg) bool {
            return std.mem.order(u8, x.key, y.key) == .lt;
        }
    }.lt);
    return sorted;
}

/// Ports `common_chat_templates_apply_jinja` (chat.cpp:1231 @d81235049).
fn applyJinja(a: Allocator, tmpls: *const Templates, inputs: *const TemplatesInputs) Error!ChatParams {
    var params: GenerationParams = .{};
    params.tools = try msg_mod.toolsToJsonOaicompat(a, inputs.tools);
    const tmpl = if (params.tools.isArray() and tmpls.template_tool_use != null) &tmpls.template_tool_use.? else &tmpls.template_default;
    const src = tmpl.source();
    const caps = tmpl.originalCaps();
    var messages_to_render = inputs.messages;
    if (util.contains(src, "You have access to the following functions in JSONSchema format")) {
        // StepFun: trim message contents (including typed content parts) before rendering,
        // otherwise leftover whitespace drives the model into reasoning loops (issue #24181)
        messages_to_render = try trimAllContent(a, inputs.messages);
    }
    params.messages = try msg_mod.renderMessagesToJson(a, messages_to_render, caps.contentCaps());
    params.tool_choice = inputs.tool_choice;
    params.reasoning_format = inputs.reasoning_format;
    params.enable_thinking = inputs.enable_thinking;
    params.grammar = inputs.grammar;
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.add_bos = tmpls.add_bos;
    params.add_eos = tmpls.add_eos;

    params.continue_final_message = inputs.continue_final_message;
    if (params.continue_final_message != .none) {
        params.add_generation_prompt = false;

        if (inputs.messages.len != 0) {
            // Render messages[:-1] and store continuation message separately
            params.continue_msg = inputs.messages[inputs.messages.len - 1];
            params.messages.eraseIndex(a, params.messages.size() - 1) catch |e| return util.fromJson(e);
        }

        if (params.continue_final_message == .auto and inputs.messages.len != 0) {
            // Resolve based on message content
            params.continue_final_message = .content;
            if (params.continue_msg.reasoning_content.len != 0 and
                params.continue_msg.content.len == 0 and
                params.continue_msg.content_parts.items.len == 0)
            {
                params.continue_final_message = .reasoning;
            }
        }
    }

    if (!util.contains(src, "<|channel|>")) {
        // map developer to system for all models except for GPT-OSS
        try mapDeveloperRoleToSystem(a, &params.messages);
    }

    if (!caps.supports_system_role) {
        try systemMessageNotSupported(a, &params.messages);
    }

    if (caps.supports_tool_calls) {
        // some templates will require the content field in tool call messages
        // to still be non-null, this puts an empty string everywhere where the
        // content field is null
        try requiresNonNullContent(a, &params.messages);
    }

    if (caps.supports_object_arguments) {
        try funcArgsNotString(a, &params.messages);
    }

    params.extra_context = try template.extraContext(a);
    for (try sortedKwargs(a, inputs.chat_template_kwargs)) |el| {
        const slot = params.extra_context.index(a, el.key) catch |e| return util.fromJson(e);
        slot.assign(a, json.parse(a, el.value) catch |e| return util.fromJson(e));
    }

    if (inputs.json_schema.len != 0) {
        params.json_schema = json.parse(a, inputs.json_schema) catch |e| return util.fromJson(e);
    }

    params.parallel_tool_calls = inputs.parallel_tool_calls;

    if (params.tools.isArray()) {
        if (params.tool_choice != .none and params.grammar.len != 0) {
            return throw(error.RuntimeError, "Cannot specify grammar with tools", .{});
        }
        if (caps.supports_tool_calls and !caps.supports_tools) {
            log.wrn("Template supports tool calls but does not natively describe tools. The fallback behaviour used may " ++
                "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n", .{});
        }
    }

    if (inputs.force_pure_content) {
        log.wrn("Forcing pure content template, will not render reasoning or tools separately.", .{});
        // Create the result structure
        var data: ChatParams = .{};
        var params_copy = params;
        params_copy.reasoning_format = .none;
        data.prompt = try template.directApplyImpl(a, tmpl, &params_copy, null, null, null);
        data.generation_prompt = try template.generationPromptImpl(a, tmpl, &params, null, null, null);
        data.format = .peg_native;
        var arena = chat_peg.buildChatPegParser(a, data.generation_prompt, pureContentParser) catch |e| return util.fromPeg(e);
        defer arena.deinit();
        data.parser = try parsers.save(a, &arena);
        return data;
    }

    if (try trySpecializedTemplate(a, tmpl, src, &params)) |result| {
        return result;
    }

    return autoParse(a, tmpl, &params) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        return exception.rethrow(error.InvalidArgument, e, "Unable to generate parser for this template. Automatic parser generation failed: ", "");
    };
}

/// `p.literal(data.generation_prompt) << p.content(p.rest())`.
fn pureContentParser(generation_prompt: []const u8, p: *chat_peg.ChatBuilder) chat_peg.Error!peg.ParserId {
    return p.b.shl(try p.b.literal(generation_prompt), try p.content(try p.b.rest()));
}

/// The `try` block of `common_chat_templates_apply_jinja`: the differential
/// auto-parser, the message delimiters it found and the thinking tags.
fn autoParse(a: Allocator, tmpl: *const Template, params: *const GenerationParams) Error!ChatParams {
    log.dbg("{s}: using differential autoparser\n", .{"common_chat_templates_apply_jinja"});
    var ap: autoparser.Autoparser = .{};
    try autoparser.analyzeTemplate(&ap, a, tmpl);
    var auto_params = try autoparser.generateParserAnalyzed(a, tmpl, params, &ap);

    var delimiters: msg_mod.MsgDelimiters = .{};
    if (ap.assistant_start.len != 0) {
        try delimiters.add(a, .assistant, ap.assistant_start);
    }
    if (ap.user_start.len != 0) {
        try delimiters.add(a, .user, ap.user_start);
    }

    auto_params.message_delimiters = delimiters;

    auto_params.supports_thinking = ap.reasoning.mode != .NONE;
    if (auto_params.supports_thinking) {
        auto_params.thinking_start_tag = autoparser.helpers.trimWhitespace(ap.reasoning.start);
        const end_tag = autoparser.helpers.trimWhitespace(ap.reasoning.end);
        if (end_tag.len != 0) {
            auto_params.thinking_end_tags = .empty;
            try auto_params.thinking_end_tags.append(a, end_tag);
        }
    }
    var arena = peg.Arena.init(a) catch |e| return util.fromPeg(e);
    defer arena.deinit();
    peg.serialize.load(&arena, auto_params.parser) catch |e| return util.fromPeg(e);
    if (log.debugEnabled()) {
        const dumped = arena.dump(a, arena.root) catch |e| return util.fromPeg(e);
        // `%s` of each `c_str()`: up to its first NUL
        log.dbg("{s}: generated parser:\n{s}\n\nparser generation prompt: {s}\n", .{ "common_chat_templates_apply_jinja", std.mem.sliceTo(dumped, 0), std.mem.sliceTo(auto_params.generation_prompt, 0) });
    }
    return auto_params;
}

/// Ports `common_chat_templates_apply_legacy` (chat.cpp:1375 @d81235049):
/// the conversation through libllama's built-in template families, and the
/// grammar from the JSON schema.
fn applyLegacy(a: Allocator, tmpls: *const Templates, inputs: *const TemplatesInputs) Error!ChatParams {
    const api = @import("llama").api;
    const Message = std.meta.Child(@typeInfo(@TypeOf(api.llama.llama_chat_apply_template)).@"fn".param_types[1].?);
    var alloc_size: usize = 0;
    const chat = try a.alloc(Message, inputs.messages.len);
    const contents = try a.alloc([:0]u8, inputs.messages.len);

    for (inputs.messages, 0..) |msg, i| {
        var content: std.ArrayList(u8) = .empty;
        try content.appendSlice(a, msg.content);
        for (msg.content_parts.items) |part| {
            if (!std.mem.eql(u8, part.type, "text") and !std.mem.eql(u8, part.type, "media_marker")) {
                log.wrn("Ignoring non-text content part: {s}\n", .{part.type});
                continue;
            }
            if (content.items.len != 0) {
                try content.appendSlice(a, "\n");
            }
            try content.appendSlice(a, part.text);
        }
        contents[i] = try content.toOwnedSliceSentinel(a, 0);
    }
    for (contents, 0..) |content, i| {
        const msg = inputs.messages[i];
        chat[i] = .{ .role = (try a.dupeSentinel(u8, msg.role, 0)).ptr, .content = content.ptr };
        const msg_size = msg.role.len + content.len;
        alloc_size += msg_size + (msg_size / 4); // == msg_size * 1.25 but avoiding float ops
    }

    var buf = try a.alloc(u8, alloc_size);

    // run the first time to get the total output length
    const src = try a.dupeSentinel(u8, tmpls.template_default.source(), 0);
    var res = api.llama.llama_chat_apply_template(src.ptr, chat.ptr, chat.len, inputs.add_generation_prompt, buf.ptr, @intCast(buf.len));

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        return throw(error.RuntimeError, "this custom template is not supported, try using --jinja", .{});
    }

    // if it turns out that our buffer is too small, we resize it
    if (@as(usize, @intCast(res)) > buf.len) {
        buf = try a.alloc(u8, @intCast(res));
        res = api.llama.llama_chat_apply_template(src.ptr, chat.ptr, chat.len, inputs.add_generation_prompt, buf.ptr, @intCast(buf.len));
    }

    // for safety, we check the result again
    if (res < 0 or @as(usize, @intCast(res)) > buf.len) {
        return throw(error.RuntimeError, "failed to apply chat template, try using --jinja", .{});
    }

    var params: ChatParams = .{};
    params.prompt = buf[0..@intCast(res)];
    if (inputs.json_schema.len != 0) {
        const schema = json.parse(a, inputs.json_schema) catch |e| return util.fromJson(e);
        params.grammar = json.schema_to_grammar.jsonSchemaToGrammar(a, &schema, false) catch |e| return util.fromSchema(e);
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}

/// Ports `common_chat_templates_apply` (chat.cpp:1440 @d81235049): the
/// conversation rendered, and how to read the reply.
///
/// Parameters:
/// - `a`: an arena; the result and every temporary are its.
/// - `tmpls`: from `init`.
/// - `inputs`: borrowed for the call.
pub fn apply(a: Allocator, tmpls: *const Templates, inputs: *const TemplatesInputs) Error!ChatParams {
    return if (inputs.use_jinja) applyJinja(a, tmpls, inputs) else applyLegacy(a, tmpls, inputs);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the kwargs map iterates by key, the last of a repeat winning" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const s = try sortedKwargs(arena_state.allocator(), &.{ .{ .key = "b", .value = "1" }, .{ .key = "a", .value = "2" }, .{ .key = "b", .value = "3" } });
    try std.testing.expectEqual(@as(usize, 2), s.len);
    try std.testing.expectEqualStrings("a", s[0].key);
    try std.testing.expectEqualStrings("3", s[1].value);
}
