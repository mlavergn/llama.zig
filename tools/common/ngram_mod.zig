//! `common_ngram_mod`: a fixed-size table from the hash of an n-gram to the
//! token that followed it, for `ngram-mod` self-speculation.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/ngram-mod.h` and `llama.cpp/common/ngram-mod.cpp`
//! at v0.8.0 (`8642541b2`). Each declaration names the C++ it replaces and
//! the line it began at.
//!
//! # Collisions
//!
//! The table keeps one token per slot and never checks that the n-gram in
//! a slot is the one asked about: `add` overwrites and `get` answers for
//! whatever hashed there last. The hash is `size_t` arithmetic, so it wraps
//! here as it does in the C++.

const std = @import("std");

/// Ports `common_ngram_mod` (ngram-mod.h:13 @8642541b2).
pub const Mod = struct {
    /// `entry_t`: a token.
    pub const Entry = i32;

    /// Ports `EMPTY` (ngram-mod.h:16 @8642541b2), a member of `common_ngram_mod`.
    pub const empty_entry: Entry = -1;

    allocator: std.mem.Allocator,
    /// ngram size to hash
    n: usize,
    used: usize,
    entries: []Entry,

    /// Ports `common_ngram_mod` (ngram-mod.cpp:9 @8642541b2), the
    /// constructor: `size` empty slots.
    ///
    /// Parameters:
    /// - `allocator`: owns the table.
    /// - `n`: the n-gram size hashed.
    /// - `size`: the number of slots.
    ///
    /// Return: the table, freed with `deinit`.
    pub fn init(allocator: std.mem.Allocator, n: u16, size_: usize) error{OutOfMemory}!Mod {
        var self: Mod = .{ .allocator = allocator, .n = n, .used = 0, .entries = try allocator.alloc(Entry, size_) };
        self.reset();
        return self;
    }

    /// Frees the table.
    pub fn deinit(self: *Mod) void {
        self.allocator.free(self.entries);
    }

    /// A copy, as the C++ copies the struct.
    ///
    /// Return: the copy, freed with `deinit`.
    pub fn clone(self: *const Mod, allocator: std.mem.Allocator) error{OutOfMemory}!Mod {
        return .{ .allocator = allocator, .n = self.n, .used = self.used, .entries = try allocator.dupe(Entry, self.entries) };
    }

    /// Ports `idx` (ngram-mod.cpp:15 @8642541b2), a member of
    /// `common_ngram_mod`: each token is widened to `size_t` with its sign,
    /// as the C++ converts it.
    ///
    /// Parameters:
    /// - `tokens`: at least `n` tokens.
    ///
    /// Return: the slot.
    pub fn idx(self: *const Mod, tokens: [*]const Entry) usize {
        var res: u64 = 0;

        for (0..self.n) |i| {
            res = res *% 6364136223846793005 +% @as(u64, @bitCast(@as(i64, tokens[i])));
        }

        res = res % self.entries.len;

        return @intCast(res);
    }

    /// Ports `add` (ngram-mod.cpp:27 @8642541b2), a member of
    /// `common_ngram_mod`: the token after the n-gram, `tokens[n]`.
    ///
    /// Parameters:
    /// - `tokens`: at least `n + 1` tokens.
    pub fn add(self: *Mod, tokens: [*]const Entry) void {
        const i = self.idx(tokens);

        if (self.entries[i] == empty_entry) {
            self.used += 1;
        }

        self.entries[i] = tokens[self.n];
    }

    /// Ports `get` (ngram-mod.cpp:37 @8642541b2), a member of
    /// `common_ngram_mod`.
    ///
    /// Parameters:
    /// - `tokens`: at least `n` tokens.
    ///
    /// Return: the token stored for the n-gram's slot, `empty_entry` if none.
    pub fn get(self: *const Mod, tokens: [*]const Entry) Entry {
        const i = self.idx(tokens);

        return self.entries[i];
    }

    /// Ports `reset` (ngram-mod.cpp:43 @8642541b2), a member of
    /// `common_ngram_mod`.
    pub fn reset(self: *Mod) void {
        @memset(self.entries, empty_entry);
        self.used = 0;
    }

    /// Ports `get_n` (ngram-mod.cpp:48 @8642541b2), a member of
    /// `common_ngram_mod`.
    pub fn getN(self: *const Mod) usize {
        return self.n;
    }

    /// Ports `get_used` (ngram-mod.cpp:52 @8642541b2), a member of
    /// `common_ngram_mod`.
    pub fn getUsed(self: *const Mod) usize {
        return self.used;
    }

    /// Ports `size` (ngram-mod.cpp:56 @8642541b2), a member of
    /// `common_ngram_mod`.
    pub fn size(self: *const Mod) usize {
        return self.entries.len;
    }

    /// Ports `size_bytes` (ngram-mod.cpp:60 @8642541b2), a member of
    /// `common_ngram_mod`.
    pub fn sizeBytes(self: *const Mod) usize {
        return self.entries.len * @sizeOf(Entry);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the hash wraps in size_t and widens negative tokens with their sign" {
    var m = try Mod.init(std.testing.allocator, 2, 1000003);
    defer m.deinit();
    const toks = [_]i32{ -1, 7, 42 };
    // (0 * K + 2^64 - 1) * K + 7, all mod 2^64, then mod size
    const k: u64 = 6364136223846793005;
    const want = ((@as(u64, std.math.maxInt(u64)) *% k) +% 7) % 1000003;
    try std.testing.expectEqual(@as(usize, @intCast(want)), m.idx(&toks));
    m.add(&toks);
    try std.testing.expectEqual(@as(usize, 1), m.getUsed());
    try std.testing.expectEqual(@as(i32, 42), m.get(&toks));
    m.add(&toks);
    try std.testing.expectEqual(@as(usize, 1), m.getUsed());
    m.reset();
    try std.testing.expectEqual(Mod.empty_entry, m.get(&toks));
    try std.testing.expectEqual(@as(usize, 4 * 1000003), m.sizeBytes());
}
