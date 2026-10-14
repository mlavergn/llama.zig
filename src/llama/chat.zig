//! The built-in chat templates: naming, detecting and applying them.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-chat.cpp` -- `LLM_CHAT_TEMPLATES`,
//!   `llm_chat_template_from_str`, `llm_chat_detect_template`,
//!   `llm_chat_apply_template` and `llama_chat_builtin_templates`
//! - `llama.cpp/src/llama-chat.h` -- `llm_chat_template`
//!
//! Both at v0.3.0 (`c1d0e7a00`). Each declaration below names the line it
//! came from.
//!
//! # Not a Jinja engine
//!
//! This is upstream's string-matching fallback: `detectTemplate` looks for
//! marker substrings in a template's source, and `applyTemplate` formats a
//! conversation with hand-written code for the family it found. The CLI's
//! real templating is `cli/chat.zig`.
//!
//! # What the C++ leaves to the C library
//!
//! `trim` asks `isspace`, and the OpenChat and Bailing templates upper-case
//! a role with `toupper`. Both are locale-dependent, and `toupper` is handed
//! a plain `char`, which is signed on this target, so a role starting with a
//! byte at or above 0x80 reaches it negative. Rather than reproduce Apple's
//! `"C"` locale tables, the port calls the same libc functions with the same
//! arguments, as `impl.zig` does for libm.
//!
//! # No exports
//!
//! Nothing here exports a C symbol: `llama_chat_apply_template` and
//! `llama_chat_builtin_templates` are still defined by the C++, and
//! libllama swaps once, at `llama.h` (`PLAN.md`, Decision 42). A shim there
//! calls `detectTemplate`, `applyTemplate` and `builtinTemplates`.

const std = @import("std");
const impl = @import("impl.zig");

const c = impl.c;
const allocator = impl.allocator;

extern "c" fn isspace(ch: c_int) c_int;
extern "c" fn toupper(ch: c_int) c_int;

/// `llama_chat_message`: a role and its content, both NUL-terminated and
/// borrowed for the duration of a call. Neither may be null; the C++ builds
/// a `std::string` from each.
pub const Message = c.llama_chat_message;

/// Ports `llm_chat_template` (llama-chat.h:7 @c1d0e7a00).
///
/// The ordinals are the C++'s: `unknown` is the last, and `dots1` is the one
/// template `detectTemplate` can return that has no name in
/// `LLM_CHAT_TEMPLATES`.
pub const Template = enum(c_int) {
    chatml,
    llama_2,
    llama_2_sys,
    llama_2_sys_bos,
    llama_2_sys_strip,
    mistral_v1,
    mistral_v3,
    mistral_v3_tekken,
    mistral_v7,
    mistral_v7_tekken,
    phi_3,
    phi_4,
    falcon_3,
    zephyr,
    monarch,
    gemma,
    orion,
    openchat,
    vicuna,
    vicuna_orca,
    deepseek,
    deepseek_2,
    deepseek_3,
    deepseek_ocr,
    command_r,
    llama_3,
    chatglm_3,
    chatglm_4,
    glmedge,
    minicpm,
    exaone_3,
    exaone_4,
    exaone_moe,
    rwkv_world,
    granite_3_x,
    granite_4_0,
    granite_4_1,
    gigachat,
    megrez,
    yandex,
    bailing,
    bailing_think,
    bailing2,
    llama4,
    smolvlm,
    dots1,
    hunyuan_moe,
    openai_moe,
    hunyuan_dense,
    hunyuan_vl,
    kimi_k2,
    seed_oss,
    grok_2,
    pangu_embed,
    solar_open,
    unknown,
};

const Entry = struct {
    name: [:0]const u8,
    tmpl: Template,
};

/// Ports `LLM_CHAT_TEMPLATES` (llama-chat.cpp:28 @c1d0e7a00), in its
/// declaration order. `templates` below is the order the C++ actually
/// iterates in.
const declared = [_]Entry{
    .{ .name = "chatml", .tmpl = .chatml },
    .{ .name = "llama2", .tmpl = .llama_2 },
    .{ .name = "llama2-sys", .tmpl = .llama_2_sys },
    .{ .name = "llama2-sys-bos", .tmpl = .llama_2_sys_bos },
    .{ .name = "llama2-sys-strip", .tmpl = .llama_2_sys_strip },
    .{ .name = "mistral-v1", .tmpl = .mistral_v1 },
    .{ .name = "mistral-v3", .tmpl = .mistral_v3 },
    .{ .name = "mistral-v3-tekken", .tmpl = .mistral_v3_tekken },
    .{ .name = "mistral-v7", .tmpl = .mistral_v7 },
    .{ .name = "mistral-v7-tekken", .tmpl = .mistral_v7_tekken },
    .{ .name = "phi3", .tmpl = .phi_3 },
    .{ .name = "phi4", .tmpl = .phi_4 },
    .{ .name = "falcon3", .tmpl = .falcon_3 },
    .{ .name = "zephyr", .tmpl = .zephyr },
    .{ .name = "monarch", .tmpl = .monarch },
    .{ .name = "gemma", .tmpl = .gemma },
    .{ .name = "orion", .tmpl = .orion },
    .{ .name = "openchat", .tmpl = .openchat },
    .{ .name = "vicuna", .tmpl = .vicuna },
    .{ .name = "vicuna-orca", .tmpl = .vicuna_orca },
    .{ .name = "deepseek", .tmpl = .deepseek },
    .{ .name = "deepseek2", .tmpl = .deepseek_2 },
    .{ .name = "deepseek3", .tmpl = .deepseek_3 },
    .{ .name = "deepseek-ocr", .tmpl = .deepseek_ocr },
    .{ .name = "command-r", .tmpl = .command_r },
    .{ .name = "llama3", .tmpl = .llama_3 },
    .{ .name = "chatglm3", .tmpl = .chatglm_3 },
    .{ .name = "chatglm4", .tmpl = .chatglm_4 },
    .{ .name = "glmedge", .tmpl = .glmedge },
    .{ .name = "minicpm", .tmpl = .minicpm },
    .{ .name = "exaone3", .tmpl = .exaone_3 },
    .{ .name = "exaone4", .tmpl = .exaone_4 },
    .{ .name = "exaone-moe", .tmpl = .exaone_moe },
    .{ .name = "rwkv-world", .tmpl = .rwkv_world },
    .{ .name = "granite", .tmpl = .granite_3_x },
    .{ .name = "granite-4.0", .tmpl = .granite_4_0 },
    .{ .name = "granite-4.1", .tmpl = .granite_4_1 },
    .{ .name = "gigachat", .tmpl = .gigachat },
    .{ .name = "megrez", .tmpl = .megrez },
    .{ .name = "yandex", .tmpl = .yandex },
    .{ .name = "bailing", .tmpl = .bailing },
    .{ .name = "bailing-think", .tmpl = .bailing_think },
    .{ .name = "bailing2", .tmpl = .bailing2 },
    .{ .name = "llama4", .tmpl = .llama4 },
    .{ .name = "smolvlm", .tmpl = .smolvlm },
    .{ .name = "hunyuan-moe", .tmpl = .hunyuan_moe },
    .{ .name = "gpt-oss", .tmpl = .openai_moe },
    .{ .name = "hunyuan-dense", .tmpl = .hunyuan_dense },
    .{ .name = "hunyuan-vl", .tmpl = .hunyuan_vl },
    .{ .name = "kimi-k2", .tmpl = .kimi_k2 },
    .{ .name = "seed_oss", .tmpl = .seed_oss },
    .{ .name = "grok-2", .tmpl = .grok_2 },
    .{ .name = "pangu-embedded", .tmpl = .pangu_embed },
    .{ .name = "solar-open", .tmpl = .solar_open },
};

/// `LLM_CHAT_TEMPLATES` in the order a `std::map<std::string, ...>` holds
/// it: by key, `std::less<std::string>`, which is `char_traits<char>::compare`
/// -- `memcmp`, so unsigned bytes -- and then length. That is
/// `std.mem.lessThan(u8, ...)`. `llama_chat_builtin_templates` hands the
/// names out in this order, so it is part of the contract.
const templates: [declared.len]Entry = blk: {
    @setEvalBranchQuota(100_000);
    var sorted = declared;
    // Insertion sort; the keys are unique, so stability is moot.
    for (1..sorted.len) |i| {
        var j = i;
        while (j > 0 and std.mem.lessThan(u8, sorted[j].name, sorted[j - 1].name)) : (j -= 1) {
            const t = sorted[j];
            sorted[j] = sorted[j - 1];
            sorted[j - 1] = t;
        }
    }
    break :blk sorted;
};

/// Ports `trim` (llama-chat.cpp:16 @c1d0e7a00): drops leading and trailing
/// bytes `isspace` accepts, each passed as an `unsigned char`.
///
/// Return: a sub-slice of `s`.
fn trim(s: []const u8) []const u8 {
    var start: usize = 0;
    var end: usize = s.len;
    while (start < end and isspace(s[start]) != 0) start += 1;
    while (end > start and isspace(s[end - 1]) != 0) end -= 1;
    return s[start..end];
}

/// `toupper` applied to a `char`, as `role[0] = toupper(role[0])` and
/// `std::transform(..., ::toupper)` do: the byte is read as a signed `char`,
/// widened to `int`, and the result narrowed back to `char`.
fn upper(b: u8) u8 {
    const r = toupper(@as(i8, @bitCast(b)));
    return @truncate(@as(u32, @bitCast(r)));
}

/// Ports `llm_chat_template_from_str` (llama-chat.cpp:85 @c1d0e7a00).
///
/// The C++ is `LLM_CHAT_TEMPLATES.at(name)`, which throws
/// `std::out_of_range` for a name not in the table; its one caller,
/// `llm_chat_detect_template`, catches exactly that. Here the miss is
/// `null`.
///
/// Parameters:
/// - `name`: the template name, compared byte for byte.
///
/// Return: the template, or `null` where the C++ throws.
pub fn templateFromStr(name: []const u8) ?Template {
    for (templates) |e| {
        if (std.mem.eql(u8, e.name, name)) return e.tmpl;
    }
    return null;
}

fn has(tmpl: []const u8, needle: []const u8) bool {
    return std.mem.find(u8, tmpl, needle) != null;
}

/// Ports `llm_chat_detect_template` (llama-chat.cpp:89 @c1d0e7a00).
///
/// A built-in name first, then a cascade of substring tests on the
/// template's source. The order of the tests is the contract: many
/// templates contain several families' markers, and the first test that
/// matches wins.
///
/// Parameters:
/// - `tmpl`: a template name or a Jinja template's source; borrowed for the
///   call.
///
/// Return: the template, or `.unknown`.
pub fn detectTemplate(tmpl: []const u8) Template {
    if (templateFromStr(tmpl)) |t| return t;

    if (has(tmpl, "<|im_start|>")) {
        return if (has(tmpl, "<|im_sep|>"))
            .phi_4
        else if (has(tmpl, "<end_of_utterance>"))
            .smolvlm // SmolVLM uses <|im_start|> as BOS, but it is NOT chatml
        else
            .chatml;
    } else if (std.mem.startsWith(u8, tmpl, "mistral") or has(tmpl, "[INST]")) {
        if (has(tmpl, "[SYSTEM_PROMPT]")) {
            return .mistral_v7;
        } else if (has(tmpl, "' [INST] ' + system_message") or has(tmpl, "[AVAILABLE_TOOLS]")) {
            // Official mistral 'v1', 'v3' and 'v3-tekken' templates.
            if (has(tmpl, " [INST]")) {
                return .mistral_v1;
            } else if (has(tmpl, "\"[INST]\"")) {
                return .mistral_v3_tekken;
            }
            return .mistral_v3;
        } else {
            // llama2 template and its variants.
            const support_system_message = has(tmpl, "<<SYS>>");
            const add_bos_inside_history = has(tmpl, "bos_token + '[INST]");
            const strip_message = has(tmpl, "content.strip()");
            if (strip_message) {
                return .llama_2_sys_strip;
            } else if (add_bos_inside_history) {
                return .llama_2_sys_bos;
            } else if (support_system_message) {
                return .llama_2_sys;
            } else {
                return .llama_2;
            }
        }
    } else if (has(tmpl, "<|assistant|>") and has(tmpl, "<|end|>")) {
        return .phi_3;
    } else if (has(tmpl, "[gMASK]<sop>")) {
        return .chatglm_4;
    } else if (has(tmpl, "<|assistant|>") and has(tmpl, "<|user|>")) {
        if (has(tmpl, "<|tool_declare|>")) {
            return .exaone_moe;
        }
        return if (has(tmpl, "</s>")) .falcon_3 else .glmedge;
    } else if (has(tmpl, "<|{{ item['role'] }}|>") and has(tmpl, "<|begin_of_image|>")) {
        return .glmedge;
    } else if (has(tmpl, "<|user|>") and has(tmpl, "<|endoftext|>")) {
        return .zephyr;
    } else if (has(tmpl, "bos_token + message['role']")) {
        return .monarch;
    } else if (has(tmpl, "<start_of_turn>")) {
        return .gemma;
    } else if (has(tmpl, "'\\n\\nAssistant: ' + eos_token")) {
        // OrionStarAI/Orion-14B-Chat
        return .orion;
    } else if (has(tmpl, "GPT4 Correct ")) {
        // openchat/openchat-3.5-0106
        return .openchat;
    } else if (has(tmpl, "USER: ") and has(tmpl, "ASSISTANT: ")) {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        if (has(tmpl, "SYSTEM: ")) {
            return .vicuna_orca;
        }
        return .vicuna;
    } else if (has(tmpl, "### Instruction:") and has(tmpl, "<|EOT|>")) {
        // deepseek-ai/deepseek-coder-33b-instruct
        return .deepseek;
    } else if (has(tmpl, "<|START_OF_TURN_TOKEN|>") and has(tmpl, "<|USER_TOKEN|>")) {
        // CohereForAI/c4ai-command-r-plus
        return .command_r;
    } else if (has(tmpl, "<|start_header_id|>") and has(tmpl, "<|end_header_id|>")) {
        return .llama_3;
    } else if (has(tmpl, "[gMASK]sop")) {
        // chatglm3-6b
        return .chatglm_3;
    } else if (has(tmpl, "<用户>")) {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        return .minicpm;
    } else if (has(tmpl, "'Assistant: ' + message['content'] + eos_token")) {
        return .deepseek_2;
    } else if (has(tmpl, "<｜Assistant｜>") and has(tmpl, "<｜User｜>") and has(tmpl, "<｜end▁of▁sentence｜>")) {
        return .deepseek_3;
    } else if (has(tmpl, "[|system|]") and has(tmpl, "[|assistant|]") and has(tmpl, "[|endofturn|]")) {
        if (has(tmpl, "[|tool|]")) {
            return .exaone_4;
        }
        // EXAONE-3.0-7.8B-Instruct
        return .exaone_3;
    } else if (has(tmpl, "rwkv-world") or has(tmpl, "{{- 'User: ' + message['content']|trim + '\\n\\n' -}}")) {
        return .rwkv_world;
    } else if (has(tmpl, "<|start_of_role|>")) {
        if (has(tmpl, "<tool_call>") or has(tmpl, "<tools>")) {
            if (has(tmpl, "g4_default_system_message")) {
                return .granite_4_0;
            }
            return .granite_4_1;
        }
        return .granite_3_x;
    } else if (has(tmpl, "message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]")) {
        return .gigachat;
    } else if (has(tmpl, "<|role_start|>")) {
        return .megrez;
    } else if (has(tmpl, " Ассистент:")) {
        return .yandex;
    } else if (has(tmpl, "<role>ASSISTANT</role>") and has(tmpl, "'HUMAN'")) {
        return .bailing;
    } else if (has(tmpl, "<role>ASSISTANT</role>") and has(tmpl, "\"HUMAN\"") and has(tmpl, "<think>")) {
        return .bailing_think;
    } else if (has(tmpl, "<role>ASSISTANT</role>") and has(tmpl, "<role>HUMAN</role>") and has(tmpl, "<|role_end|>")) {
        return .bailing2;
    } else if (has(tmpl, "<|header_start|>") and has(tmpl, "<|header_end|>")) {
        return .llama4;
    } else if (has(tmpl, "<|endofuserprompt|>")) {
        return .dots1;
    } else if (has(tmpl, "<|extra_0|>") and has(tmpl, "<|extra_4|>")) {
        return .hunyuan_moe;
    } else if (has(tmpl, "<|start|>") and has(tmpl, "<|channel|>")) {
        return .openai_moe;
    } else if (has(tmpl, "<｜hy_Assistant｜>") and has(tmpl, "<｜hy_begin▁of▁sentence｜>")) {
        return .hunyuan_vl;
    } else if (has(tmpl, "<｜hy_Assistant｜>") and has(tmpl, "<｜hy_place▁holder▁no▁3｜>")) {
        return .hunyuan_dense;
    } else if (has(tmpl, "<|im_assistant|>assistant<|im_middle|>")) {
        return .kimi_k2;
    } else if (has(tmpl, "<seed:bos>")) {
        return .seed_oss;
    } else if (has(tmpl, "'Assistant: '  + message['content'] + '<|separator|>")) {
        return .grok_2;
    } else if (has(tmpl, "[unused9]系统：[unused10]")) {
        return .pangu_embed;
    } else if (has(tmpl, "<|begin|>") and has(tmpl, "<|end|>") and has(tmpl, "<|content|>")) {
        return .solar_open;
    }
    return .unknown;
}

/// The `std::stringstream` `llm_chat_apply_template` writes into, appending
/// to the caller's list.
const Stream = struct {
    list: *std.ArrayList(u8),

    fn put(self: Stream, s: []const u8) error{OutOfMemory}!void {
        try self.list.appendSlice(allocator, s);
    }

    /// `ss << a << b << ...`: every element of the tuple, in order.
    fn all(self: Stream, parts: anytype) error{OutOfMemory}!void {
        inline for (parts) |p| try self.put(p);
    }
};

fn str(p: [*c]const u8) []const u8 {
    return std.mem.span(@as([*:0]const u8, @ptrCast(p)));
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// Ports `llm_chat_apply_template` (llama-chat.cpp:244 @c1d0e7a00).
///
/// Formats `chat` for `tmpl` into `out`, replacing what it held. Every
/// template but `.unknown` is supported; for `.unknown` the C++ returns -1
/// before assigning `dest`, so `out` is left as it was.
///
/// The C++ takes a vector of message *pointers* and the Monarch template
/// compares each with `chat.front()` by address. Its only caller,
/// `llama_chat_apply_template`, fills the vector with `&chat[i]`, so the
/// addresses are distinct and the test is `i == 0`, which is what a slice
/// can express.
///
/// Running out of memory aborts, where the C++'s `std::bad_alloc` would
/// escape through the C ABI and terminate.
///
/// Parameters:
/// - `tmpl`: the template.
/// - `chat`: the conversation; borrowed for the call.
/// - `out`: receives the prompt; allocated with `impl.allocator`, owned by
///   the caller.
/// - `add_ass`: whether to end with the prefix of an assistant turn, for
///   the templates that have one.
///
/// Return: the prompt's length in bytes, narrowed to `int32_t` as the C++'s
/// `return dest.size()` does; or -1 for `.unknown`.
pub fn applyTemplate(tmpl: Template, chat: []const Message, out: *std.ArrayList(u8), add_ass: bool) i32 {
    if (tmpl == .unknown) return -1;
    out.clearRetainingCapacity();
    format(tmpl, chat, .{ .list = out }, add_ass) catch impl.abort("llama-chat.cpp", 249, "std::bad_alloc");
    return @bitCast(@as(u32, @truncate(out.items.len)));
}

/// The body of `llm_chat_apply_template`, one branch per template family.
fn format(tmpl: Template, chat: []const Message, ss: Stream, add_ass: bool) error{OutOfMemory}!void {
    switch (tmpl) {
        .unknown => unreachable,
        .chatml => {
            for (chat) |m| try ss.all(.{ "<|im_start|>", str(m.role), "\n", str(m.content), "<|im_end|>\n" });
            if (add_ass) try ss.put("<|im_start|>assistant\n");
        },
        .mistral_v7, .mistral_v7_tekken => {
            // Official mistral 'v7' template.
            const trailing_space: []const u8 = if (tmpl == .mistral_v7) " " else "";
            for (chat) |m| {
                const role = str(m.role);
                const content = str(m.content);
                if (eql(role, "system")) {
                    try ss.all(.{ "[SYSTEM_PROMPT]", trailing_space, content, "[/SYSTEM_PROMPT]" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "[INST]", trailing_space, content, "[/INST]" });
                } else {
                    try ss.all(.{ trailing_space, content, "</s>" });
                }
            }
        },
        .mistral_v1, .mistral_v3, .mistral_v3_tekken => {
            const leading_space: []const u8 = if (tmpl == .mistral_v1) " " else "";
            const trailing_space: []const u8 = if (tmpl == .mistral_v3_tekken) "" else " ";
            const trim_assistant_message = tmpl == .mistral_v3;
            var is_inside_turn = false;
            for (chat) |m| {
                if (!is_inside_turn) {
                    try ss.all(.{ leading_space, "[INST]", trailing_space });
                    is_inside_turn = true;
                }
                const role = str(m.role);
                const content = str(m.content);
                if (eql(role, "system")) {
                    try ss.all(.{ content, "\n\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ content, leading_space, "[/INST]" });
                } else {
                    try ss.all(.{ trailing_space, if (trim_assistant_message) trim(content) else content, "</s>" });
                    is_inside_turn = false;
                }
            }
        },
        .llama_2, .llama_2_sys, .llama_2_sys_bos, .llama_2_sys_strip => {
            const support_system_message = tmpl != .llama_2;
            const add_bos_inside_history = tmpl == .llama_2_sys_bos;
            const strip_message = tmpl == .llama_2_sys_strip;
            var is_inside_turn = true; // skip BOS at the beginning
            try ss.put("[INST] ");
            for (chat) |m| {
                const content = if (strip_message) trim(str(m.content)) else str(m.content);
                const role = str(m.role);
                if (!is_inside_turn) {
                    is_inside_turn = true;
                    try ss.put(if (add_bos_inside_history) "<s>[INST] " else "[INST] ");
                }
                if (eql(role, "system")) {
                    if (support_system_message) {
                        try ss.all(.{ "<<SYS>>\n", content, "\n<</SYS>>\n\n" });
                    } else {
                        // Without <<SYS>>, the system message still leads the first turn.
                        try ss.all(.{ content, "\n" });
                    }
                } else if (eql(role, "user")) {
                    try ss.all(.{ content, " [/INST]" });
                } else {
                    try ss.all(.{ content, "</s>" });
                    is_inside_turn = false;
                }
            }
        },
        .phi_3 => {
            for (chat) |m| try ss.all(.{ "<|", str(m.role), "|>\n", str(m.content), "<|end|>\n" });
            if (add_ass) try ss.put("<|assistant|>\n");
        },
        .phi_4 => {
            for (chat) |m| try ss.all(.{ "<|im_start|>", str(m.role), "<|im_sep|>", str(m.content), "<|im_end|>" });
            if (add_ass) try ss.put("<|im_start|>assistant<|im_sep|>");
        },
        .falcon_3 => {
            for (chat) |m| try ss.all(.{ "<|", str(m.role), "|>\n", str(m.content), "\n" });
            if (add_ass) try ss.put("<|assistant|>\n");
        },
        .zephyr => {
            for (chat) |m| try ss.all(.{ "<|", str(m.role), "|>", "\n", str(m.content), "<|endoftext|>\n" });
            if (add_ass) try ss.put("<|assistant|>\n");
        },
        .monarch => {
            // The <s> is included inside history, skipped for the first message.
            for (chat, 0..) |m, i| {
                const bos: []const u8 = if (i == 0) "" else "<s>";
                try ss.all(.{ bos, str(m.role), "\n", str(m.content), "</s>\n" });
            }
            if (add_ass) try ss.put("<s>assistant\n");
        },
        .gemma => {
            var system_prompt: std.ArrayList(u8) = .empty;
            defer system_prompt.deinit(allocator);
            for (chat) |m| {
                var role = str(m.role);
                if (eql(role, "system")) {
                    // Gemma has no system message; it is merged into the next user turn.
                    try system_prompt.appendSlice(allocator, trim(str(m.content)));
                    continue;
                }
                // in gemma, "assistant" is "model"
                if (eql(role, "assistant")) role = "model";
                try ss.all(.{ "<start_of_turn>", role, "\n" });
                if (system_prompt.items.len != 0 and !eql(role, "model")) {
                    try ss.all(.{ system_prompt.items, "\n\n" });
                    system_prompt.clearRetainingCapacity();
                }
                try ss.all(.{ trim(str(m.content)), "<end_of_turn>\n" });
            }
            if (add_ass) try ss.put("<start_of_turn>model\n");
        },
        .orion => {
            var system_prompt: std.ArrayList(u8) = .empty;
            defer system_prompt.deinit(allocator);
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    // No system message support; it is merged into the next user turn.
                    try system_prompt.appendSlice(allocator, str(m.content));
                    continue;
                } else if (eql(role, "user")) {
                    try ss.put("Human: ");
                    if (system_prompt.items.len != 0) {
                        try ss.all(.{ system_prompt.items, "\n\n" });
                        system_prompt.clearRetainingCapacity();
                    }
                    try ss.all(.{ str(m.content), "\n\nAssistant: </s>" });
                } else {
                    try ss.all(.{ str(m.content), "</s>" });
                }
            }
        },
        .openchat => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ str(m.content), "<|end_of_turn|>" });
                } else {
                    // `role[0] = toupper(role[0])`. For an empty role that
                    // writes `toupper(0) == 0` over the terminator: no change.
                    try ss.put("GPT4 Correct ");
                    if (role.len != 0) {
                        try ss.all(.{ &[1]u8{upper(role[0])}, role[1..] });
                    }
                    try ss.all(.{ ": ", str(m.content), "<|end_of_turn|>" });
                }
            }
            if (add_ass) try ss.put("GPT4 Correct Assistant:");
        },
        .vicuna, .vicuna_orca => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    // Orca-Vicuna variant uses a system prefix
                    if (tmpl == .vicuna_orca) {
                        try ss.all(.{ "SYSTEM: ", str(m.content), "\n" });
                    } else {
                        try ss.all(.{ str(m.content), "\n\n" });
                    }
                } else if (eql(role, "user")) {
                    try ss.all(.{ "USER: ", str(m.content), "\n" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "ASSISTANT: ", str(m.content), "</s>\n" });
                }
            }
            if (add_ass) try ss.put("ASSISTANT:");
        },
        .deepseek => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.put(str(m.content));
                } else if (eql(role, "user")) {
                    try ss.all(.{ "### Instruction:\n", str(m.content), "\n" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "### Response:\n", str(m.content), "\n<|EOT|>\n" });
                }
            }
            if (add_ass) try ss.put("### Response:\n");
        },
        .command_r => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>", trim(str(m.content)), "<|END_OF_TURN_TOKEN|>" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", trim(str(m.content)), "<|END_OF_TURN_TOKEN|>" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", trim(str(m.content)), "<|END_OF_TURN_TOKEN|>" });
                }
            }
            if (add_ass) try ss.put("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        },
        .llama_3 => {
            for (chat) |m| try ss.all(.{ "<|start_header_id|>", str(m.role), "<|end_header_id|>\n\n", trim(str(m.content)), "<|eot_id|>" });
            if (add_ass) try ss.put("<|start_header_id|>assistant<|end_header_id|>\n\n");
        },
        .chatglm_3 => {
            try ss.all(.{ "[gMASK]", "sop" });
            for (chat) |m| try ss.all(.{ "<|", str(m.role), "|>", "\n ", str(m.content) });
            if (add_ass) try ss.put("<|assistant|>");
        },
        .chatglm_4 => {
            try ss.all(.{ "[gMASK]", "<sop>" });
            for (chat) |m| try ss.all(.{ "<|", str(m.role), "|>", "\n", str(m.content) });
            if (add_ass) try ss.put("<|assistant|>\n");
        },
        .glmedge => {
            for (chat) |m| try ss.all(.{ "<|", str(m.role), "|>", "\n", str(m.content) });
            if (add_ass) try ss.put("<|assistant|>");
        },
        .minicpm => {
            for (chat) |m| {
                if (eql(str(m.role), "user")) {
                    try ss.all(.{ "<用户>", trim(str(m.content)), "<AI>" });
                } else {
                    try ss.put(trim(str(m.content)));
                }
            }
        },
        .deepseek_2 => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ str(m.content), "\n\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "User: ", str(m.content), "\n\n" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "Assistant: ", str(m.content), "<｜end▁of▁sentence｜>" });
                }
            }
            if (add_ass) try ss.put("Assistant:");
        },
        .deepseek_3 => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ str(m.content), "\n\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "<｜User｜>", str(m.content) });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "<｜Assistant｜>", str(m.content), "<｜end▁of▁sentence｜>" });
                }
            }
            if (add_ass) try ss.put("<｜Assistant｜>");
        },
        .deepseek_ocr => {
            // no template
            for (chat) |m| try ss.put(str(m.content));
        },
        .exaone_3 => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ "[|system|]", trim(str(m.content)), "[|endofturn|]\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "[|user|]", trim(str(m.content)), "\n" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "[|assistant|]", trim(str(m.content)), "[|endofturn|]\n" });
                }
            }
            if (add_ass) try ss.put("[|assistant|]");
        },
        .exaone_4 => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ "[|system|]", trim(str(m.content)), "[|endofturn|]\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "[|user|]", trim(str(m.content)), "\n" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "[|assistant|]", trim(str(m.content)), "[|endofturn|]\n" });
                } else if (eql(role, "tool")) {
                    try ss.all(.{ "[|tool|]", trim(str(m.content)), "[|endofturn|]\n" });
                }
            }
            if (add_ass) try ss.put("[|assistant|]");
        },
        .exaone_moe => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ "<|system|>\n", trim(str(m.content)), "<|endofturn|>\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "<|user|>\n", trim(str(m.content)), "<|endofturn|>\n" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "<|assistant|>\n", trim(str(m.content)), "<|endofturn|>\n" });
                } else if (eql(role, "tool")) {
                    try ss.all(.{ "<|tool|>\n", trim(str(m.content)), "<|endofturn|>\n" });
                }
            }
            if (add_ass) try ss.put("<|assistant|>\n");
        },
        .rwkv_world => {
            // This template requires the model to have "\n\n" as EOT token.
            for (chat, 0..) |m, i| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ "System: ", trim(str(m.content)), "\n\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "User: ", trim(str(m.content)), "\n\n" });
                    if (i == chat.len - 1) try ss.put("Assistant:");
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "Assistant: ", trim(str(m.content)), "\n\n" });
                }
            }
        },
        .granite_3_x => {
            for (chat) |m| {
                const role = str(m.role);
                try ss.all(.{ "<|start_of_role|>", role, "<|end_of_role|>" });
                if (eql(role, "assistant_tool_call")) try ss.put("<|tool_call|>");
                try ss.all(.{ str(m.content), "<|end_of_text|>\n" });
            }
            if (add_ass) try ss.put("<|start_of_role|>assistant<|end_of_role|>");
        },
        // The C++ has a branch for each; their bodies are identical.
        .granite_4_0, .granite_4_1 => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "assistant_tool_call")) {
                    try ss.put("<|start_of_role|>assistant<|end_of_role|><|tool_call|>");
                } else {
                    try ss.all(.{ "<|start_of_role|>", role, "<|end_of_role|>" });
                }
                try ss.all(.{ str(m.content), "<|end_of_text|>\n" });
            }
            if (add_ass) try ss.put("<|start_of_role|>assistant<|end_of_role|>");
        },
        .gigachat => {
            const has_system = chat.len != 0 and eql(str(chat[0].role), "system");
            if (has_system) {
                try ss.all(.{ "<s>", str(chat[0].content), "<|message_sep|>" });
            } else {
                try ss.put("<s>");
            }
            for (chat[@intFromBool(has_system)..]) |m| {
                const role = str(m.role);
                if (eql(role, "user")) {
                    try ss.all(.{ "user<|role_sep|>", str(m.content), "<|message_sep|>", "available functions<|role_sep|>[]<|message_sep|>" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "assistant<|role_sep|>", str(m.content), "<|message_sep|>" });
                }
            }
            if (add_ass) try ss.put("assistant<|role_sep|>");
        },
        .megrez => {
            for (chat) |m| try ss.all(.{ "<|role_start|>", str(m.role), "<|role_end|>", str(m.content), "<|turn_end|>" });
            if (add_ass) try ss.put("<|role_start|>assistant<|role_end|>");
        },
        .yandex => {
            // "\n\n" is defined as EOT token.
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "user")) {
                    try ss.all(.{ " Пользователь: ", str(m.content), "\n\n" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ " Ассистент: ", str(m.content), "\n\n" });
                }
            }
            if (add_ass) try ss.put(" Ассистент:[SEP]");
        },
        .bailing, .bailing_think => {
            // Bailing (Ling/Ring) template
            for (chat) |m| {
                try ss.put("<role>");
                try putRoleUpper(ss, str(m.role));
                try ss.all(.{ "</role>", str(m.content) });
            }
            if (add_ass) {
                try ss.put("<role>ASSISTANT</role>");
                if (tmpl == .bailing_think) try ss.put("<think>");
            }
        },
        .bailing2 => {
            // Bailing2 (Ling 2.0) template
            const has_system = chat.len != 0 and eql(str(chat[0].role), "system");
            if (!has_system) try ss.put("<role>SYSTEM</role>detailed thinking off<|role_end|>");
            for (chat) |m| {
                try ss.put("<role>");
                try putRoleUpper(ss, str(m.role));
                try ss.all(.{ "</role>", str(m.content), "<|role_end|>" });
            }
            if (add_ass) try ss.put("<role>ASSISTANT</role>");
        },
        .llama4 => {
            for (chat) |m| try ss.all(.{ "<|header_start|>", str(m.role), "<|header_end|>\n\n", trim(str(m.content)), "<|eot|>" });
            if (add_ass) try ss.put("<|header_start|>assistant<|header_end|>\n\n");
        },
        .smolvlm => {
            try ss.put("<|im_start|>"); // uses <|im_start|> as BOS, but the actual content is NOT chatml
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ str(m.content), "\n\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "User: ", str(m.content), "<end_of_utterance>\n" });
                } else {
                    try ss.all(.{ "Assistant: ", str(m.content), "<end_of_utterance>\n" });
                }
            }
            if (add_ass) try ss.put("Assistant:");
        },
        .dots1 => {
            // dots.llm1.inst (DOTS1)
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ "<|system|>", str(m.content), "<|endofsystem|>" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "<|userprompt|>", str(m.content), "<|endofuserprompt|>" });
                } else {
                    try ss.all(.{ "<|response|>", str(m.content), "<|endofresponse|>" });
                }
            }
            if (add_ass) try ss.put("<|response|>");
        },
        .hunyuan_moe => {
            // tencent/Hunyuan-A13B-Instruct
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ "<|startoftext|>", str(m.content), "<|extra_4|>" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ str(m.content), "<|eos|>" });
                } else {
                    try ss.all(.{ "<|startoftext|>", str(m.content), "<|extra_0|>" });
                }
            }
        },
        .openai_moe => {
            // OpenAI MoE (based on Harmony chat template)
            for (chat) |m| {
                const role = str(m.role);
                try ss.all(.{ "<|start|>", role, "<|message|>", str(m.content) });
                try ss.put(if (eql(role, "assistant")) "<|return|>" else "<|end|>");
            }
            if (add_ass) try ss.put("<|start|>assistant");
        },
        .hunyuan_dense => {
            // tencent/Hunyuan-4B-Instruct
            for (chat, 0..) |m, i| {
                const role = str(m.role);
                if (i == 0 and eql(role, "system")) {
                    try ss.all(.{ str(m.content), "<｜hy_place▁holder▁no▁3｜>" });
                }
                if (eql(role, "assistant")) {
                    try ss.all(.{ "<｜hy_Assistant｜>", str(m.content), "<｜hy_place▁holder▁no▁2｜>" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "<｜hy_User｜>", str(m.content), "<｜hy_Assistant｜>" });
                }
            }
        },
        .hunyuan_vl => {
            // tencent/HunyuanOCR & tencent/HunyuanVL
            try ss.put("<｜hy_begin▁of▁sentence｜>");
            for (chat, 0..) |m, i| {
                const role = str(m.role);
                if (i == 0 and eql(role, "system")) {
                    try ss.all(.{ str(m.content), "<｜hy_place▁holder▁no▁3｜>" });
                    continue;
                }
                if (eql(role, "user")) {
                    try ss.all(.{ str(m.content), "<｜hy_User｜>" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ str(m.content), "<｜hy_Assistant｜>" });
                }
            }
        },
        .kimi_k2 => {
            // moonshotai/Kimi-K2-Instruct
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.put("<|im_system|>system<|im_middle|>");
                } else if (eql(role, "user")) {
                    try ss.put("<|im_user|>user<|im_middle|>");
                } else if (eql(role, "assistant")) {
                    try ss.put("<|im_assistant|>assistant<|im_middle|>");
                } else if (eql(role, "tool")) {
                    try ss.put("<|im_system|>tool<|im_middle|>");
                }
                try ss.all(.{ str(m.content), "<|im_end|>" });
            }
            if (add_ass) try ss.put("<|im_assistant|>assistant<|im_middle|>");
        },
        .seed_oss => {
            for (chat) |m| {
                const role = str(m.role);
                const content = if (eql(role, "assistant")) trim(str(m.content)) else str(m.content);
                try ss.all(.{ "<seed:bos>", role, "\n", content, "<seed:eos>" });
            }
            if (add_ass) try ss.put("<seed:bos>assistant\n");
        },
        .grok_2 => {
            for (chat) |m| {
                const role = str(m.role);
                if (eql(role, "system")) {
                    try ss.all(.{ "System: ", trim(str(m.content)), "<|separator|>\n\n" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "Human: ", trim(str(m.content)), "<|separator|>\n\n" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "Assistant: ", str(m.content), "<|separator|>\n\n" });
                }
            }
            if (add_ass) try ss.put("Assistant:");
        },
        .pangu_embed => {
            for (chat, 0..) |m, i| {
                const role = str(m.role);
                const content = str(m.content);
                if (i == 0 and !eql(role, "system")) try ss.put("[unused9]系统：[unused10]");
                if (eql(role, "system")) {
                    try ss.all(.{ "[unused9]系统：", content, "[unused10]" });
                } else if (eql(role, "user")) {
                    try ss.all(.{ "[unused9]用户：", content, "[unused10]" });
                } else if (eql(role, "assistant")) {
                    try ss.all(.{ "[unused9]助手：", content, "[unused10]" });
                } else if (eql(role, "tool")) {
                    try ss.all(.{ "[unused9]工具：", content, "[unused10]" });
                } else if (eql(role, "function")) {
                    try ss.all(.{ "[unused9]方法：", content, "[unused10]" });
                }
            }
            if (add_ass) try ss.put("[unused9]助手：");
        },
        .solar_open => {
            for (chat) |m| try ss.all(.{ "<|begin|>", str(m.role), "<|content|>", str(m.content), "<|end|>" });
            if (add_ass) try ss.put("<|begin|>assistant");
        },
    }
}

/// The Bailing templates' role: `"user"` becomes `"HUMAN"`, anything else
/// is upper-cased byte by byte with `std::transform(..., ::toupper)`.
fn putRoleUpper(ss: Stream, role: []const u8) error{OutOfMemory}!void {
    if (eql(role, "user")) return ss.put("HUMAN");
    for (role) |b| try ss.put(&[1]u8{upper(b)});
}

/// Ports `llama_chat_builtin_templates` (llama-chat.cpp:950 @c1d0e7a00).
///
/// Parameters:
/// - `output`: receives up to `len` names, in `std::map` key order; each
///   points at static storage. Not read when `len` is 0, so it may then be
///   null.
/// - `len`: the capacity of `output`.
///
/// Return: the number of built-in templates, however many were written.
pub fn builtinTemplates(output: ?[*]?[*:0]const u8, len: usize) i32 {
    const n = @min(len, templates.len);
    for (0..n) |i| output.?[i] = templates[i].name.ptr;
    return @intCast(templates.len);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the enum has the C++'s ordinals at both ends" {
    try std.testing.expectEqual(@as(c_int, 0), @backingInt(Template.chatml));
    try std.testing.expectEqual(@as(c_int, 45), @backingInt(Template.dots1));
    try std.testing.expectEqual(@as(c_int, 55), @backingInt(Template.unknown));
}

test "builtin names come out in std::map order" {
    var names: [64]?[*:0]const u8 = @splat(null);
    const n = builtinTemplates(&names, names.len);
    try std.testing.expectEqual(@as(i32, 54), n);
    try std.testing.expectEqualStrings("bailing", std.mem.span(names[0].?));
    try std.testing.expectEqualStrings("bailing-think", std.mem.span(names[1].?));
    try std.testing.expectEqualStrings("bailing2", std.mem.span(names[2].?));
    try std.testing.expectEqualStrings("zephyr", std.mem.span(names[53].?));
    try std.testing.expect(names[54] == null);
    try std.testing.expectEqual(@as(i32, 54), builtinTemplates(null, 0));
}

test "a name wins over the substring cascade" {
    try std.testing.expectEqual(Template.mistral_v7_tekken, detectTemplate("mistral-v7-tekken"));
    try std.testing.expectEqual(Template.llama_2, detectTemplate("mistral"));
    try std.testing.expectEqual(Template.unknown, detectTemplate(""));
    try std.testing.expect(templateFromStr("dots1") == null);
    try std.testing.expectEqual(Template.dots1, detectTemplate("<|endofuserprompt|>"));
}

test "chatml applies, and unknown leaves the output alone" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "kept");
    const chat = [_]Message{
        .{ .role = "system", .content = "be brief" },
        .{ .role = "user", .content = "hi" },
    };
    try std.testing.expectEqual(@as(i32, -1), applyTemplate(.unknown, &chat, &out, true));
    try std.testing.expectEqualStrings("kept", out.items);
    const expected = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    try std.testing.expectEqual(@as(i32, expected.len), applyTemplate(.chatml, &chat, &out, true));
    try std.testing.expectEqualStrings(expected, out.items);
}

test "trim and the openchat role go through the C library" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    const chat = [_]Message{.{ .role = "user", .content = " \t hi \n" }};
    _ = applyTemplate(.openchat, &chat, &out, false);
    try std.testing.expectEqualStrings("GPT4 Correct User:  \t hi \n<|end_of_turn|>", out.items);
    _ = applyTemplate(.llama_3, &chat, &out, false);
    try std.testing.expectEqualStrings("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>", out.items);
}
