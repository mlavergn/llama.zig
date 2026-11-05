//! TranslateGemma: no tools and no reasoning, but each user message's
//! content rewritten into the template's own schema, a text part carrying
//! the source and target language codes.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/parsers/translate-gemma.cpp` (llama.cpp v0.7.0,
//! `620d2225c`).

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const exception = @import("../exception.zig");
const chat_peg = @import("../chat_peg.zig");
const params_mod = @import("../params.zig");
const template = @import("../template.zig");
const util = @import("../util.zig");
const log = @import("../log.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const ChatParams = params_mod.ChatParams;
const GenerationParams = template.GenerationParams;

/// `p.literal(data.generation_prompt) << p.content(p.rest())`.
fn parser(generation_prompt: []const u8, p: *chat_peg.ChatBuilder) chat_peg.Error!peg.ParserId {
    return p.b.shl(try p.b.literal(generation_prompt), try p.content(try p.b.rest()));
}

/// `json{{k, v}, ...}` member by member, in order.
fn obj(a: Allocator, members: []const [2][]const u8) Error!Value {
    var v = Value.emptyObject();
    for (members) |m| v.set(a, m[0], try Value.from(a, m[1])) catch |e| return util.fromJson(e);
    return v;
}

/// Ports `common_chat_params_init_translate_gemma` (parsers/translate-gemma.cpp:5
/// @620d2225c).
///
/// Return: owned by the arena `a`.
pub fn init(a: Allocator, tmpl: *const template.Template, inputs: *const GenerationParams) Error!ChatParams {
    var data: ChatParams = .{};

    // default to chat_template_kwargs, or en-GB if not specified
    const src_lang = inputs.extra_context.value("source_lang_code", []const u8, "en-GB") catch |e| return util.fromJson(e);
    const tgt_lang = inputs.extra_context.value("target_lang_code", []const u8, "en-GB") catch |e| return util.fromJson(e);
    for ([_][]const u8{ "source_lang_code", "target_lang_code" }) |key| {
        if (!inputs.extra_context.contains(key)) {
            log.wrn("TranslateGemma: {s} not set in chat_template_kwargs, defaulting to en-GB\n", .{key});
        }
    }

    var messages = try inputs.messages.clone(a);
    var it = messages.items();
    while (it.next()) |item| {
        const message = item.value;
        if (!std.mem.eql(u8, message.value("role", []const u8, "") catch |e| return util.fromJson(e), "user")) {
            continue;
        }
        var text: std.ArrayList(u8) = .empty;
        const null_value: Value = .null;
        const content: *const Value = if (message.contains("content")) message.at("content") catch |e| return util.fromJson(e) else &null_value;
        if (content.isString()) {
            try text.appendSlice(a, content.string);
        } else if (content.isArray()) {
            var parts = @constCast(content).items();
            while (parts.next()) |part| {
                if (text.items.len != 0) {
                    try text.appendSlice(a, "\n");
                }
                try text.appendSlice(a, part.value.value("text", []const u8, "") catch |e| return util.fromJson(e));
            }
        }
        var arr = Value.emptyArray();
        arr.pushBack(a, try obj(a, &.{
            .{ "type", "text" },
            .{ "text", text.items },
            .{ "source_lang_code", src_lang },
            .{ "target_lang_code", tgt_lang },
        })) catch |e| return util.fromJson(e);
        const slot = message.index(a, "content") catch |e| return util.fromJson(e);
        slot.assign(a, arr);
    }

    data.prompt = try template.directApplyImpl(a, tmpl, inputs, &messages, null, null);
    data.generation_prompt = try template.generationPromptImpl(a, tmpl, inputs, &messages, null, null);
    data.format = .peg_native;
    data.supports_thinking = false;

    if (inputs.hasContinuation()) {
        data.generation_prompt = try std.mem.concat(a, u8, &.{ "<start_of_turn>model\n", try inputs.continue_msg.renderContent(a, "\n\n") });
        try data.prompt.append(a, data.generation_prompt, false);
    }

    data.parser = chat_peg.buildChatPegParser(a, data.generation_prompt, parser) catch |e| return util.fromPeg(e);

    return data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a user message's content becomes one text part with both language codes" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try template.Template.init(std.testing.allocator, "{% for m in messages %}{% if m.content is string %}{{ m.content }}{% else %}{{ m.content[0].source_lang_code }}>{{ m.content[0].target_lang_code }}:{{ m.content[0].text }}{% endif %}{% endfor %}[source_lang_code][target_lang_code]", "", "");
    defer t.deinit();
    var p: GenerationParams = .{};
    p.messages = try json.parse(a, "[{\"role\":\"system\",\"content\":\"s\"},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\",\"text\":\"b\"}]}]");
    p.extra_context = try json.parse(a, "{\"target_lang_code\":\"fr\"}");
    const data = try init(a, &t, &p);
    try std.testing.expectEqualStrings("sen-GB>fr:a\nb[source_lang_code][target_lang_code]", try data.prompt.str(a));
    try std.testing.expect(data.parser != null);
}
