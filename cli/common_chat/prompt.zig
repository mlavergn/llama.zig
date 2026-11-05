//! A rendered chat template, split into the runs the template wrote and the
//! runs that came from user input: special tokens inside the latter must
//! not be parsed.
//!
//! # Provenance
//!
//! Ports `common_chat_prompt_part` and `common_chat_prompt` from
//! `llama.cpp/common/chat.h` and the members of the latter from
//! `llama.cpp/common/chat.cpp` (llama.cpp v0.7.0, `620d2225c`). Each
//! declaration names what it ports and the line it began at.
//!
//! # Ownership
//!
//! A `Prompt`'s list and texts belong to the allocator each call takes, an
//! arena in practice -- the one its `ChatParams` was built in: nothing here
//! is freed singly.

const std = @import("std");

const Allocator = std.mem.Allocator;

/// Ports `common_chat_prompt_part` (chat.h:276 @620d2225c).
pub const PromptPart = struct {
    text: []const u8,
    /// from user input, special tokens inside must not be parsed
    is_input: bool = false,
};

/// Ports `common_chat_prompt` (chat.h:282 @620d2225c): the rendered chat
/// template, split into template parts and user input parts.
pub const Prompt = struct {
    parts: std.ArrayList(PromptPart) = .empty,

    /// Ports `common_chat_prompt` (chat.h:288 @620d2225c), the constructor
    /// from text: a single part of template text, borrowed.
    pub fn fromText(a: Allocator, text: []const u8) Allocator.Error!Prompt {
        var self: Prompt = .{};
        try self.parts.append(a, .{ .text = text, .is_input = false });
        return self;
    }

    /// Ports `str` (chat.cpp:1503 @620d2225c): the parts joined.
    ///
    /// Return: the one part's text when there is one, else owned by `a`.
    pub fn str(self: *const Prompt, a: Allocator) Allocator.Error![]const u8 {
        if (self.parts.items.len == 1) return self.parts.items[0].text;
        var len: usize = 0;
        for (self.parts.items) |part| len += part.text.len;
        const out = try a.alloc(u8, len);
        var at: usize = 0;
        for (self.parts.items) |part| {
            @memcpy(out[at..][0..part.text.len], part.text);
            at += part.text.len;
        }
        return out;
    }

    /// Ports `append` (chat.cpp:1519 @620d2225c): `text` joined onto the
    /// last part when it is of the same kind, else a new part. Empty text
    /// adds nothing.
    pub fn append(self: *Prompt, a: Allocator, text: []const u8, is_input: bool) Allocator.Error!void {
        if (text.len == 0) return;
        if (self.parts.items.len != 0 and self.parts.items[self.parts.items.len - 1].is_input == is_input) {
            const last = &self.parts.items[self.parts.items.len - 1];
            last.text = try std.mem.concat(a, u8, &.{ last.text, text });
        } else {
            try self.parts.append(a, .{ .text = text, .is_input = is_input });
        }
    }

    /// Ports `replace_last` (chat.cpp:1530 @620d2225c): the last occurrence
    /// of `from` within one template part, searching the parts from the
    /// end, replaced by `to`. User input parts are skipped, and an
    /// occurrence spanning two parts is not one.
    pub fn replaceLast(self: *Prompt, a: Allocator, from: []const u8, to: []const u8) Allocator.Error!void {
        var i = self.parts.items.len;
        while (i > 0) {
            i -= 1;
            const part = &self.parts.items[i];
            if (part.is_input) continue;
            if (std.mem.lastIndexOf(u8, part.text, from)) |pos| {
                part.text = try std.mem.concat(a, u8, &.{ part.text[0..pos], to, part.text[pos + from.len ..] });
                return;
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a prompt joins a run of one kind and replaces only in template text" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var p: Prompt = .{};
    try p.append(a, "<s>", false);
    try p.append(a, "x<|return|>", true);
    try p.append(a, "", false);
    try p.append(a, "<|return|>", false);
    try p.append(a, "!", false);
    try std.testing.expectEqual(@as(usize, 3), p.parts.items.len);
    try p.replaceLast(a, "<|return|>", "<|end|>");
    try std.testing.expectEqualStrings("<s>x<|return|><|end|>!", try p.str(a));
    try p.replaceLast(a, "<|return|>", "?");
    try std.testing.expectEqualStrings("<s>x<|return|><|end|>!", try p.str(a));
}
