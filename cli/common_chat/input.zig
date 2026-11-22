//! A model's reply as the chat parser reads it: the text, and beside each
//! byte the token that began there.
//!
//! # Provenance
//!
//! Ports `common_chat_input` from `llama.cpp/common/chat.h` and its members
//! and `common_chat_input_tokenize` from `llama.cpp/common/chat.cpp`
//! (llama.cpp v1.0.0, `b1b770d90`). Each declaration names what it ports
//! and the line it began at.
//!
//! # Ownership
//!
//! An `Input` owns its text and tokens through the allocator it was made
//! with; `deinit` frees them.
//!
//! # libllama
//!
//! `tokenize` reaches the vocabulary through libllama's port,
//! `@import("llama").api`, never a `llama_*` C symbol: in a harness
//! those are upstream's.

const std = @import("std");
const mod = @import("module.zig");

/// `LLAMA_TOKEN_NULL`.
pub const token_null: i32 = -1;

/// Ports `common_chat_input` (chat.h:320 @b1b770d90): `tokens` has one
/// entry per byte of `text`, the token that begins there or
/// `token_null`.
pub const Input = struct {
    gpa: std.mem.Allocator,
    text: std.ArrayList(u8) = .empty,
    tokens: std.ArrayList(i32) = .empty,

    /// `common_chat_input()`: empty.
    pub fn init(gpa: std.mem.Allocator) Input {
        return .{ .gpa = gpa };
    }

    /// Ports `common_chat_input` (chat.h:327 @b1b770d90), the constructor
    /// from text: plain text, with no tokens.
    ///
    /// Return: owned by `gpa`; release with `deinit`.
    pub fn fromText(gpa: std.mem.Allocator, text: []const u8) mod.exception.Error!Input {
        var self: Input = .init(gpa);
        errdefer self.deinit();
        try self.text.appendSlice(gpa, text);
        try self.tokens.appendNTimes(gpa, token_null, text.len);
        return self;
    }

    pub fn deinit(self: *Input) void {
        self.text.deinit(self.gpa);
        self.tokens.deinit(self.gpa);
        self.* = undefined;
    }

    /// The C++ copy.
    ///
    /// Return: owned by `gpa`.
    pub fn clone(self: *const Input, gpa: std.mem.Allocator) mod.exception.Error!Input {
        var out: Input = .init(gpa);
        errdefer out.deinit();
        try out.text.appendSlice(gpa, self.text.items);
        try out.tokens.appendSlice(gpa, self.tokens.items);
        return out;
    }

    /// Ports `size` (chat.h:329 @b1b770d90).
    pub fn size(self: *const Input) usize {
        return self.text.items.len;
    }

    /// Ports `empty` (chat.h:330 @b1b770d90).
    pub fn empty(self: *const Input) bool {
        return self.text.items.len == 0;
    }

    /// Ports `append` (chat.cpp:1745 @b1b770d90), the overload taking a
    /// piece and its token: the token on the piece's first byte, null on
    /// the rest. An empty piece adds nothing, not even its token.
    pub fn append(self: *Input, piece: []const u8, token: i32) mod.exception.Error!void {
        if (piece.len == 0) return;
        try self.tokens.append(self.gpa, token);
        try self.tokens.appendNTimes(self.gpa, token_null, piece.len - 1);
        try self.text.appendSlice(self.gpa, piece);
    }

    /// Ports `append` (chat.cpp:1754 @b1b770d90), the overload taking a
    /// chunk.
    pub fn appendInput(self: *Input, chunk: *const Input) mod.exception.Error!void {
        try self.tokens.appendSlice(self.gpa, chunk.tokens.items);
        try self.text.appendSlice(self.gpa, chunk.text.items);
    }

    /// Ports `truncate` (chat.cpp:1759 @b1b770d90).
    pub fn truncate(self: *Input, pos: usize) void {
        if (pos < self.text.items.len) {
            self.text.shrinkRetainingCapacity(pos);
            self.tokens.shrinkRetainingCapacity(pos);
        }
    }

    /// Ports `substr` (chat.cpp:1766 @b1b770d90): `std::string::substr`,
    /// throwing `out_of_range` past the end, with the tokens beside it.
    ///
    /// Parameters:
    /// - `n`: `std::string::npos` (`util.npos`) for the rest.
    ///
    /// Return: owned by `gpa`.
    pub fn substr(self: *const Input, gpa: std.mem.Allocator, pos: usize, n: usize) mod.exception.Error!Input {
        const text = try mod.util.substr(self.text.items, pos, n);
        var out: Input = .init(gpa);
        errdefer out.deinit();
        try out.text.appendSlice(gpa, text);
        try out.tokens.appendSlice(gpa, self.tokens.items[pos..][0..text.len]);
        return out;
    }

    /// Ports `prepend` (chat.cpp:1773 @b1b770d90), the overload taking
    /// text: no tokens for it.
    pub fn prepend(self: *Input, prefix: []const u8) mod.exception.Error!void {
        try self.tokens.ensureUnusedCapacity(self.gpa, prefix.len);
        const old = self.tokens.items.len;
        self.tokens.items.len += prefix.len;
        std.mem.copyBackwards(i32, self.tokens.items[prefix.len..], self.tokens.items[0..old]);
        @memset(self.tokens.items[0..prefix.len], token_null);
        try self.text.insertSlice(self.gpa, 0, prefix);
    }

    /// Ports `prepend` (chat.cpp:1778 @b1b770d90), the overload taking a
    /// chunk.
    pub fn prependInput(self: *Input, prefix: *const Input) mod.exception.Error!void {
        try self.tokens.insertSlice(self.gpa, 0, prefix.tokens.items);
        try self.text.insertSlice(self.gpa, 0, prefix.text.items);
    }
};

/// The `llama_vocab *` of libllama's port that `vocab` is.
fn vocabPtr(vocab: *const anyopaque) VocabPtr {
    return @ptrCast(@alignCast(vocab));
}

const VocabPtr = @typeInfo(@TypeOf(@import("llama").api.vocab.llama_tokenize)).@"fn".param_types[0].?;

/// Ports `common_tokenize` (common/common.cpp:1866 @b1b770d90), the
/// overload taking a vocabulary, through libllama's port.
///
/// Parameters:
/// - `vocab`: a `llama_vocab *` from whichever `llama.h` import the caller
///   has.
///
/// Return: owned by `gpa`. `error.RuntimeError` when the count overflows
/// `int32_t`, or when libllama throws (a vocabulary with no byte token for a
/// byte of the text), with its `what()`.
pub fn commonTokenize(gpa: std.mem.Allocator, vocab: *const anyopaque, text: []const u8, add_special: bool, parse_special: bool) mod.exception.Error![]i32 {
    const api = @import("llama").api;
    const v = vocabPtr(vocab);
    // upper limit for the number of tokens
    var n_tokens: c_int = @intCast(text.len + 2 * @as(usize, @intFromBool(add_special)));
    var result = try gpa.alloc(i32, @intCast(n_tokens));
    errdefer gpa.free(result);
    // libllama's throw reaches the caller as an exception, as upstream's
    // does; `llama_tokenize` itself would terminate.
    var exc: @import("llama").impl.Exception = .{};
    n_tokens = api.vocab.tokenize(v, text, result, add_special, parse_special, &exc) catch
        return mod.exception.throw(error.RuntimeError, "{s}", .{exc.what()});
    if (n_tokens == std.math.minInt(i32)) {
        return mod.exception.throw(error.RuntimeError, "Tokenization failed: input text too large, tokenization result exceeds int32_t limit", .{});
    }
    if (n_tokens < 0) {
        result = try gpa.realloc(result, @intCast(-n_tokens));
        const check = api.vocab.tokenize(v, text, result, add_special, parse_special, &exc) catch
            return mod.exception.throw(error.RuntimeError, "{s}", .{exc.what()});
        @import("llama").impl.assert(check == -n_tokens, "common/common.cpp", 1881, "check == -n_tokens");
        return result;
    }
    return gpa.realloc(result, @intCast(n_tokens));
}

/// Ports `common_chat_input_tokenize` (chat.cpp:1783 @b1b770d90): `text`
/// tokenized with special tokens parsed, each token's piece beside it. A
/// leading space a tokenizer adds before the first special token is
/// dropped with its token; if the pieces do not give `text` back, the text
/// alone, without tokens.
///
/// Parameters:
/// - `vocab`: a `llama_vocab *` from whichever `llama.h` import the caller
///   has; only passed through to libllama's port.
///
/// Return: owned by `gpa`.
pub fn tokenize(gpa: std.mem.Allocator, vocab: *const anyopaque, text: []const u8) mod.exception.Error!Input {
    const api = @import("llama").api;
    const v = vocabPtr(vocab);

    var input: Input = .init(gpa);
    errdefer input.deinit();

    const tokens = try commonTokenize(gpa, vocab, text, false, true);
    defer gpa.free(tokens);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    for (tokens, 0..) |tok, i| {
        // common_token_to_piece(vocab, tokens[i], true)
        try buf.resize(gpa, 22);
        var n = api.vocab.llama_token_to_piece(v, tok, buf.items.ptr, @intCast(buf.items.len), 0, true);
        if (n < 0) {
            try buf.resize(gpa, @intCast(-n));
            n = api.vocab.llama_token_to_piece(v, tok, buf.items.ptr, @intCast(buf.items.len), 0, true);
        }
        const piece = buf.items[0..@intCast(n)];
        // `piece[0]` and `text[0]` of an empty string are its NUL
        const p0: u8 = if (piece.len > 0) piece[0] else 0;
        const t0: u8 = if (text.len > 0) text[0] else 0;
        if (i == 0 and std.ascii.isWhitespace(p0) and !std.ascii.isWhitespace(t0)) {
            // Some tokenizers will add a space before the first special token, need to exclude
            continue;
        }
        try input.append(piece, tok);
    }
    if (!std.mem.eql(u8, input.text.items, text)) {
        // the pieces do not give back the same text, keep the text without tokens
        input.deinit();
        return Input.fromText(gpa, text);
    }
    return input;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an input keeps one token slot per byte through every edit" {
    const a = std.testing.allocator;
    var in: Input = .init(a);
    defer in.deinit();
    try in.append("ab", 7);
    try in.append("", 9);
    try in.append("c", 8);
    try std.testing.expectEqualSlices(i32, &.{ 7, -1, 8 }, in.tokens.items);
    try in.prepend("xy");
    try std.testing.expectEqualStrings("xyabc", in.text.items);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 7, -1, 8 }, in.tokens.items);
    var sub = try in.substr(a, 2, 2);
    defer sub.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 7, -1 }, sub.tokens.items);
    try in.prependInput(&sub);
    try std.testing.expectEqualStrings("abxyabc", in.text.items);
    in.truncate(3);
    try std.testing.expectEqualStrings("abx", in.text.items);
    try std.testing.expectEqualSlices(i32, &.{ 7, -1, -1 }, in.tokens.items);
    try std.testing.expectError(error.RuntimeError, in.substr(a, 4, 1));
}
