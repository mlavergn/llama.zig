//! `common_chat_templates` on its legacy path: the model's template source,
//! applied through libllama's built-in template families.
//!
//! # Provenance
//!
//! Ports `common_chat_templates`, `common_chat_templates_init`,
//! `common_chat_format_single`, `common_chat_format_example` and
//! `common_chat_templates_apply_legacy` from `llama.cpp/common/chat.cpp`,
//! and the source normalization of `jinja::lexer::tokenize` from
//! `llama.cpp/common/jinja/lexer.cpp`, at v0.4.0 (`5266f24da`). Each
//! declaration names the C++ it replaces and the line it began at.
//!
//! # The Jinja path
//!
//! `llama-mtmd-cli` defaults to `use_jinja = false`, and that path is ported
//! whole: the template's source goes to `llama_chat_apply_template`, which
//! recognises its family. The C++ still parses the source with
//! `common/jinja` while building the templates, and a template that does not
//! parse throws there; this port does not parse it, so such a template
//! reaches `llama_chat_apply_template` instead of terminating. With
//! `--jinja`, `apply` refuses: `common_chat_templates_apply_jinja`
//! (capabilities, role workarounds, the auto-parser) is not ported.

const std = @import("std");
const c = @import("tools_h");
const log = @import("log.zig");
const common = @import("common.zig");
const io = @import("io.zig");

/// `CHATML_TEMPLATE_SRC` (chat.cpp:704 @5266f24da).
const chatml_template_src =
    "{%- for message in messages -%}\n" ++
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" ++
    "{%- endfor -%}\n" ++
    "{%- if add_generation_prompt -%}\n" ++
    "  {{- '<|im_start|>assistant\n' -}}\n" ++
    "{%- endif -%}";

/// Ports `common_chat_msg` (chat.h:80 @5266f24da) as far as these paths read
/// it: a role and its text.
pub const Msg = struct {
    role: []const u8,
    content: []const u8,
};

/// What escapes `main` from these paths: libc++'s message for an uncaught
/// `std::runtime_error`, then `abort`.
fn terminate(what: [*:0]const u8) noreturn {
    io.eprintf("libc++abi: terminating due to uncaught exception of type std::runtime_error: %s\n", .{what});
    std.process.abort();
}

/// Ports the normalization of `tokenize` (lexer.cpp:32 @5266f24da), a
/// member of `jinja::lexer`:
/// `\r\n` and `\r` become `\n`, and one trailing newline of the original is
/// dropped. The caller owns the result.
fn lexerSource(a: std.mem.Allocator, source: []const u8) ![]u8 {
    var src: std.ArrayList(u8) = .empty;
    if (source.len == 0) return src.toOwnedSlice(a);
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        const ch = source[i];
        if (ch == '\r') {
            if (i + 1 < source.len and source[i + 1] == '\n') continue; // the \r of \r\n
            try src.append(a, '\n');
        } else {
            try src.append(a, ch);
        }
    }
    if (source[source.len - 1] == '\n') _ = src.pop();
    return src.toOwnedSlice(a);
}

/// Ports `common_chat_templates` (chat.cpp:336 @5266f24da).
pub const Templates = struct {
    add_bos: bool,
    add_eos: bool,
    has_explicit_template: bool, // Model had builtin template or template overridden was specified.
    /// `template_default->source()`, always set (defaults to chatml)
    default_src: []const u8,
    /// `template_tool_use->source()`, or null
    tool_use_src: ?[]const u8,
    bos_tok: []const u8,
    eos_tok: []const u8,

    /// Ports `common_chat_templates_init` (chat.cpp:753 @5266f24da).
    ///
    /// Return: the templates, every string `a`'s.
    pub fn init(a: std.mem.Allocator, model: ?*const c.llama_model, chat_template_override: []const u8) !Templates {
        var default_template_src: []const u8 = "";
        var template_tool_use_src: []const u8 = "";

        var has_explicit_template = chat_template_override.len != 0;
        if (chat_template_override.len == 0) {
            if (model == null) _ = c.ggml_abort("chat.cpp", 762, "GGML_ASSERT(%s) failed", "model != nullptr");
            if (c.llama_model_chat_template(model, null)) |str| {
                default_template_src = std.mem.span(str);
                has_explicit_template = true;
            }
            if (c.llama_model_chat_template(model, "tool_use")) |str| {
                template_tool_use_src = std.mem.span(str);
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
        if (std.mem.indexOf(u8, default_template_src, "<|channel|>") != null and
            std.mem.indexOf(u8, default_template_src, "in message.content or") != null)
        {
            default_template_src = try replaceAll(a, default_template_src, "{%- if \"<|channel|>analysis<|message|>\" in message.content or " ++
                "\"<|channel|>final<|message|>\" in message.content %}", "{%- if false %}");
        }

        // TODO @aldehir : this is a temporary fix, pending Minja changes
        // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
        if (std.mem.indexOf(u8, default_template_src, "[TOOL_CALLS]") != null and
            std.mem.indexOf(u8, default_template_src, "if (message['content'] is none or") != null)
        {
            default_template_src = try replaceAll(a, default_template_src, "{%- if (message['content'] is none or message['content'] == '' or " ++
                "message['content']|length == 0) and (message['tool_calls'] is not defined or " ++
                "message['tool_calls'] is none or message['tool_calls']|length == 0) %}", "{%- if false %}");
        }

        var token_bos: []const u8 = "";
        var token_eos: []const u8 = "";
        var add_bos = false;
        var add_eos = false;
        if (model != null) {
            const vocab = c.llama_model_get_vocab(model);
            token_bos = try getToken(a, vocab, c.llama_vocab_bos(vocab), "BOS", "bos_token", default_template_src, template_tool_use_src);
            token_eos = try getToken(a, vocab, c.llama_vocab_eos(vocab), "EOS", "eos_token", default_template_src, template_tool_use_src);
            add_bos = c.llama_vocab_get_add_bos(vocab);
            add_eos = c.llama_vocab_get_add_eos(vocab);
        }
        return .{
            .add_bos = add_bos,
            .add_eos = add_eos,
            .has_explicit_template = has_explicit_template,
            .default_src = try lexerSource(a, default_template_src),
            .tool_use_src = if (template_tool_use_src.len != 0) try lexerSource(a, template_tool_use_src) else null,
            .bos_tok = token_bos,
            .eos_tok = token_eos,
        };
    }

    /// The lambda of `common_chat_templates_init` named `get_token` (chat.cpp:813
    /// @5266f24da).
    fn getToken(a: std.mem.Allocator, vocab: ?*const c.llama_vocab, token: c.llama_token, comptime name: [*:0]const u8, jinja_variable_name: []const u8, default_src: []const u8, tool_use_src: []const u8) ![]const u8 {
        if (token == c.LLAMA_TOKEN_NULL) {
            if (std.mem.indexOf(u8, default_src, jinja_variable_name) != null or
                std.mem.indexOf(u8, tool_use_src, jinja_variable_name) != null)
            {
                log.wrn("common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't " ++
                    "work as intended.\n", .{name});
            }
            return "";
        }
        return common.tokenToPiece(a, vocab, token, true);
    }

    /// Ports `common_chat_templates_apply_legacy` (chat.cpp:3751 @5266f24da):
    /// the conversation through `llama_chat_apply_template`.
    ///
    /// Return: the prompt, `a`'s.
    fn applyLegacy(self: *const Templates, a: std.mem.Allocator, messages: []const Msg, add_generation_prompt: bool) ![]const u8 {
        var alloc_size: usize = 0;
        const chat = try a.alloc(c.llama_chat_message, messages.len);
        for (messages, 0..) |msg, i| {
            chat[i] = .{
                .role = (try a.dupeSentinel(u8, msg.role, 0)).ptr,
                .content = (try a.dupeSentinel(u8, msg.content, 0)).ptr,
            };
            const msg_size = msg.role.len + msg.content.len;
            alloc_size += msg_size + (msg_size / 4); // == msg_size * 1.25 but avoiding float ops
        }

        var buf = try a.alloc(u8, alloc_size);

        // run the first time to get the total output length
        const src = try a.dupeSentinel(u8, self.default_src, 0);
        var res = c.llama_chat_apply_template(src.ptr, chat.ptr, chat.len, add_generation_prompt, buf.ptr, @intCast(buf.len));

        // error: chat template is not supported
        if (res < 0) {
            // if the custom "tmpl" is not supported, we throw an error
            terminate("this custom template is not supported, try using --jinja");
        }

        // if it turns out that our buffer is too small, we resize it
        if (@as(usize, @intCast(res)) > buf.len) {
            buf = try a.realloc(buf, @intCast(res));
            res = c.llama_chat_apply_template(src.ptr, chat.ptr, chat.len, add_generation_prompt, buf.ptr, @intCast(buf.len));
        }

        // for safety, we check the result again
        if (res < 0 or @as(usize, @intCast(res)) > buf.len) {
            terminate("failed to apply chat template, try using --jinja");
        }

        return buf[0..@intCast(res)];
    }

    /// Ports `common_chat_templates_apply` (chat.cpp:3816 @5266f24da).
    fn apply(self: *const Templates, a: std.mem.Allocator, messages: []const Msg, add_generation_prompt: bool, use_jinja: bool) ![]const u8 {
        if (use_jinja) terminate("--jinja is not supported by this port of llama-mtmd-cli");
        return self.applyLegacy(a, messages, add_generation_prompt);
    }

    /// Ports `common_chat_format_single` (chat.cpp:653 @5266f24da): the text
    /// `new_msg` adds to the rendered history.
    ///
    /// Return: the new text, `a`'s.
    pub fn formatSingle(self: *const Templates, a: std.mem.Allocator, past_msg: []const Msg, new_msg: Msg, add_ass: bool, use_jinja: bool) ![]const u8 {
        var fmt_past_msg: []const u8 = "";
        if (past_msg.len != 0) {
            fmt_past_msg = try self.apply(a, past_msg, false, use_jinja);
        }
        var ss: std.ArrayList(u8) = .empty;
        // if the past_msg ends with a newline, we must preserve it in the formatted version
        if (add_ass and fmt_past_msg.len != 0 and fmt_past_msg[fmt_past_msg.len - 1] == '\n') {
            try ss.append(a, '\n');
        }
        // format chat with new_msg
        const msgs = try a.alloc(Msg, past_msg.len + 1);
        @memcpy(msgs[0..past_msg.len], past_msg);
        msgs[past_msg.len] = new_msg;
        const fmt_new_msg = try self.apply(a, msgs, add_ass, use_jinja);
        // get the diff part
        try ss.appendSlice(a, substr(fmt_new_msg, fmt_past_msg.len, fmt_new_msg.len -% fmt_past_msg.len));
        return ss.toOwnedSlice(a);
    }

    /// Ports `common_chat_format_example` (chat.cpp:683 @5266f24da): a short
    /// conversation rendered, to show the template.
    pub fn formatExample(self: *const Templates, a: std.mem.Allocator, use_jinja: bool) ![]const u8 {
        const msgs = [_]Msg{
            .{ .role = "system", .content = "You are a helpful assistant" },
            .{ .role = "user", .content = "Hello" },
            .{ .role = "assistant", .content = "Hi there" },
            .{ .role = "user", .content = "How are you?" },
        };
        return self.apply(a, &msgs, true, use_jinja);
    }
};

/// `std::string::substr(pos, count)`: `out_of_range` past the end, and a
/// count clipped to what is left.
fn substr(s: []const u8, pos: usize, count: usize) []const u8 {
    if (pos > s.len) {
        io.eprintf("libc++abi: terminating due to uncaught exception of type std::out_of_range: basic_string\n", .{});
        std.process.abort();
    }
    return s[pos..][0..@min(count, s.len - pos)];
}

/// Ports `string_replace_all` (common/common.cpp:507 @5266f24da): every `search`
/// replaced. The caller owns the result.
fn replaceAll(a: std.mem.Allocator, s: []const u8, search: []const u8, replace: []const u8) ![]const u8 {
    if (search.len == 0) return s;
    var o: std.ArrayList(u8) = .empty;
    var last: usize = 0;
    while (std.mem.indexOfPos(u8, s, last, search)) |pos| {
        try o.appendSlice(a, s[last..pos]);
        try o.appendSlice(a, replace);
        last = pos + search.len;
    }
    try o.appendSlice(a, s[last..]);
    return o.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the lexer's normalization" {
    const a = std.testing.allocator;
    const s = try lexerSource(a, "a\r\nb\rc\n");
    defer a.free(s);
    try std.testing.expectEqualStrings("a\nb\nc", s);
    const t = try lexerSource(a, "x\r");
    defer a.free(t);
    try std.testing.expectEqualStrings("x\n", t);
}
