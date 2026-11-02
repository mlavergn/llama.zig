//! What applying a chat template produces: the prompt, and the grammar,
//! triggers and parser a model's reply is constrained and read with.
//!
//! # Provenance
//!
//! Ports `common_chat_params` from `llama.cpp/common/chat.h` and
//! `common_grammar_trigger` with its type from `llama.cpp/common/common.h`
//! (llama.cpp v0.5.0, `7fe450e19`). Each declaration names what it ports and
//! the line it began at.
//!
//! # Ownership
//!
//! Every string and list in a `ChatParams` belongs to the allocator the call
//! that built it took, an arena: nothing here is freed singly.

const std = @import("std");
const msg_mod = @import("msg.zig");

const Allocator = std.mem.Allocator;

/// Ports `common_grammar_trigger_type` (common/common.h:143 @7fe450e19).
pub const TriggerType = enum(c_int) { token, word, pattern, pattern_full };

/// Ports `common_grammar_trigger` (common/common.h:150 @7fe450e19).
pub const GrammarTrigger = struct {
    type: TriggerType,
    value: []const u8,
    /// `LLAMA_TOKEN_NULL` unless `type` is `token`.
    token: i32 = -1,
};

/// Ports `common_chat_params` (chat.h:269 @7fe450e19).
pub const ChatParams = struct {
    format: msg_mod.Format = .content_only,
    prompt: []const u8 = "",
    grammar: []const u8 = "",
    grammar_lazy: bool = false,
    generation_prompt: []const u8 = "",
    supports_thinking: bool = false,
    /// e.g. `<think>`
    thinking_start_tag: []const u8 = "",
    /// e.g. `</think>`
    thinking_end_tags: std.ArrayList([]const u8) = .empty,
    grammar_triggers: std.ArrayList(GrammarTrigger) = .empty,
    preserved_tokens: std.ArrayList([]const u8) = .empty,
    additional_stops: std.ArrayList([]const u8) = .empty,
    /// The PEG parser, serialized (`common_peg_arena`'s `save`).
    parser: []const u8 = "",
    message_delimiters: msg_mod.MsgDelimiters = .{},

    /// `push_back` of each of `items` onto `list`.
    pub fn appendAll(a: Allocator, list: *std.ArrayList([]const u8), items: []const []const u8) !void {
        try list.appendSlice(a, items);
    }

    /// `data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word })`.
    pub fn addWordTrigger(self: *ChatParams, a: Allocator, word: []const u8) !void {
        try self.grammar_triggers.append(a, .{ .type = .word, .value = word });
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
