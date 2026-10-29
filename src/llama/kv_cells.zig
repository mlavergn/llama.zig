//! The bookkeeping behind a KV cache: which cells are used, at which
//! position, by which sequences.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-kv-cells.h` at v0.4.1 (`b29c606e2`):
//! `llama_kv_cell_ext` and `llama_kv_cells`.
//!
//! # The ordered containers
//!
//! The C++ keeps `used` in a `std::set<uint32_t>` and each sequence's cells
//! in a `std::set<std::pair<llama_pos, uint32_t>>` ordered by position, then
//! cell, and asks of them their size, smallest and largest key, and an
//! `upper_bound`. Here `used` is a bit per cell with a count, and each
//! sequence's cells a sorted array of (position, cell) pairs: positions
//! arrive mostly in increasing order, so an insertion is almost always an
//! append. The answers are the same; only the asymptotics of a removal from
//! the middle differ.

const std = @import("std");
const impl = @import("impl.zig");

const allocator = impl.allocator;

/// `LLAMA_MAX_SEQ` (llama-cparams.h:8 @b29c606e2).
pub const max_seq = 256;

/// `std::bitset<LLAMA_MAX_SEQ>`: which sequences hold a cell.
pub const SeqSet = std.bit_set.StaticBitSet(max_seq);

fn oom() noreturn {
    impl.abort("llama-kv-cells.h", 0, "out of memory");
}

/// Mirrors `llama_kv_cell_ext` (llama-kv-cells.h:13 @b29c606e2): the 2D
/// position M-RoPE models give a cell, and the token it holds. It travels
/// in a saved state as its raw bytes.
pub const CellExt = extern struct {
    x: i32 = 0,
    y: i32 = 0,
    /// `LLAMA_TOKEN_NULL` when the cell came from an embedding (multimodal)
    /// input; the n-gram embeddings hash it.
    tok: i32 = -1,

    /// Ports `llama_kv_cell_ext`'s `is_2d_gt` (llama-kv-cells.h:23 @b29c606e2).
    pub fn is2dGt(self: CellExt, ox: i32, oy: i32) bool {
        return (self.y > oy) or (self.y == oy and self.x > ox);
    }
};

comptime {
    // `static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>)`: the
    // state format is these 12 bytes.
    std.debug.assert(@sizeOf(CellExt) == 12);
}

/// A sequence's cells as (position, cell) pairs, ordered by position, then
/// cell: `std::set<std::pair<llama_pos, uint32_t>>`.
const PosSet = struct {
    const Key = struct {
        pos: i32,
        cell: u32,

        fn lessThan(a: Key, b: Key) bool {
            return a.pos < b.pos or (a.pos == b.pos and a.cell < b.cell);
        }
    };

    items: std.ArrayList(Key) = .empty,

    /// The first index whose key is not less than `k`: `lower_bound`.
    fn lowerBound(self: *const PosSet, k: Key) usize {
        const keys = self.items.items;
        // Fast path: positions mostly arrive in order.
        if (keys.len > 0 and Key.lessThan(keys[keys.len - 1], k)) return keys.len;
        var lo: usize = 0;
        var hi: usize = keys.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (Key.lessThan(keys[mid], k)) lo = mid + 1 else hi = mid;
        }
        return lo;
    }

    /// The first index whose key is greater than `k`: `upper_bound`.
    fn upperBound(self: *const PosSet, k: Key) usize {
        const keys = self.items.items;
        var lo: usize = 0;
        var hi: usize = keys.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (!Key.lessThan(k, keys[mid])) lo = mid + 1 else hi = mid;
        }
        return lo;
    }

    /// `insert`: a key already present is left alone.
    fn insert(self: *PosSet, k: Key) void {
        const at = self.lowerBound(k);
        const keys = self.items.items;
        if (at < keys.len and keys[at].pos == k.pos and keys[at].cell == k.cell) return;
        self.items.insert(allocator, at, k) catch oom();
    }

    /// `erase`: a key that is absent is left alone, as the C++'s is once
    /// its `assert(n == 1)` compiles out.
    fn erase(self: *PosSet, k: Key) void {
        const at = self.lowerBound(k);
        const keys = self.items.items;
        if (at < keys.len and keys[at].pos == k.pos and keys[at].cell == k.cell) _ = self.items.orderedRemove(at);
    }
};

/// Ports `llama_kv_cells` (llama-kv-cells.h:36 @b29c606e2).
pub const KvCells = struct {
    has_shift: bool = false,
    /// `used`: a bit per cell, and how many are set.
    used: std.DynamicBitSetUnmanaged = .{},
    n_used: u32 = 0,
    pos: std.ArrayList(i32) = .empty,
    ext: std.ArrayList(CellExt) = .empty,
    /// The shifts applied since the last `resetShift`.
    shift: std.ArrayList(i32) = .empty,
    /// Which sequences occupy each cell.
    seq: std.ArrayList(SeqSet) = .empty,
    /// One (position, cell) pair per cell holding the sequence, ordered by
    /// position: the cell is part of the key because a position can occur
    /// more than once in a sequence, through a cache reuse (rm + add) or
    /// the repeated positions of some vision models' embeddings.
    seq_pos: [max_seq]PosSet = @splat(.{}),

    /// Return: nothing.
    pub fn deinit(self: *KvCells) void {
        self.used.deinit(allocator);
        self.pos.deinit(allocator);
        self.ext.deinit(allocator);
        self.shift.deinit(allocator);
        self.seq.deinit(allocator);
        for (&self.seq_pos) |*sp| sp.items.deinit(allocator);
    }

    fn usedInsert(self: *KvCells, i: u32) void {
        if (!self.used.isSet(i)) {
            self.used.set(i);
            self.n_used += 1;
        }
    }

    fn usedErase(self: *KvCells, i: u32) void {
        if (self.used.isSet(i)) {
            self.used.unset(i);
            self.n_used -= 1;
        }
    }

    /// Ports `llama_kv_cells`'s `reset` (llama-kv-cells.h:40 @b29c606e2).
    pub fn reset(self: *KvCells) void {
        @memset(self.pos.items, -1);
        @memset(self.ext.items, .{});
        @memset(self.shift.items, 0);
        @memset(self.seq.items, SeqSet.empty);
        self.has_shift = false;
        self.used.unsetAll();
        self.n_used = 0;
        for (&self.seq_pos) |*sp| sp.items.clearRetainingCapacity();
    }

    /// Ports `llama_kv_cells`'s `reset_shift` (llama-kv-cells.h:57 @b29c606e2).
    pub fn resetShift(self: *KvCells) void {
        self.has_shift = false;
        @memset(self.shift.items, 0);
    }

    /// Ports `llama_kv_cells`'s `size` (llama-kv-cells.h:65 @b29c606e2).
    pub fn size(self: *const KvCells) u32 {
        return @intCast(self.pos.items.len);
    }

    /// Ports `llama_kv_cells`'s `resize` (llama-kv-cells.h:69 @b29c606e2).
    pub fn resize(self: *KvCells, n: u32) void {
        self.pos.resize(allocator, n) catch oom();
        self.ext.resize(allocator, n) catch oom();
        self.shift.resize(allocator, n) catch oom();
        self.seq.resize(allocator, n) catch oom();
        self.used.resize(allocator, n, false) catch oom();
        self.reset();
    }

    /// Ports `llama_kv_cells`'s `is_empty` (llama-kv-cells.h:78 @b29c606e2).
    pub fn isEmpty(self: *const KvCells, i: u32) bool {
        return self.pos.items[i] == -1;
    }

    /// Ports `llama_kv_cells`'s `get_used` (llama-kv-cells.h:85 @b29c606e2).
    pub fn getUsed(self: *const KvCells) u32 {
        return self.n_used;
    }

    /// Ports `llama_kv_cells`'s `used_min` (llama-kv-cells.h:91 @b29c606e2):
    /// the first used cell, or 0 when none is.
    pub fn usedMin(self: *const KvCells) u32 {
        if (self.n_used == 0) return 0;
        return @intCast(self.used.findFirstSet().?);
    }

    /// Ports `llama_kv_cells`'s `used_max_p1` (llama-kv-cells.h:97 @b29c606e2):
    /// one past the last used cell, or 0 when none is.
    pub fn usedMaxP1(self: *const KvCells) u32 {
        if (self.n_used == 0) return 0;
        return @intCast(self.used.findLastSet().? + 1);
    }

    /// Ports `llama_kv_cells`'s `get_has_shift` (llama-kv-cells.h:101 @b29c606e2).
    pub fn getHasShift(self: *const KvCells) bool {
        return self.has_shift;
    }

    /// Ports `llama_kv_cells`'s `cp` over a range (llama-kv-cells.h:126
    /// @b29c606e2): cells `[i, i + n)`, for saving the state.
    ///
    /// Return: the copy, the caller's to `deinit`.
    pub fn cpRange(self: *const KvCells, i: u32, n: u32) KvCells {
        var res: KvCells = .{};
        res.resize(n);
        for (0..n) |j| {
            const idx = i + j;
            res.pos.items[j] = self.pos.items[idx];
            res.ext.items[j] = self.ext.items[idx];
            res.seq.items[j] = self.seq.items[idx];
            std.debug.assert(self.shift.items[idx] == 0);
        }
        return res;
    }

    /// Ports `llama_kv_cells`'s `cp` over indices (llama-kv-cells.h:147
    /// @b29c606e2).
    ///
    /// Return: the copy, the caller's to `deinit`.
    pub fn cpIdxs(self: *const KvCells, idxs: []const u32) KvCells {
        var res: KvCells = .{};
        res.resize(@intCast(idxs.len));
        for (idxs, 0..) |idx, j| {
            res.pos.items[j] = self.pos.items[idx];
            res.ext.items[j] = self.ext.items[idx];
            res.seq.items[j] = self.seq.items[idx];
            std.debug.assert(self.shift.items[idx] == 0);
        }
        return res;
    }

    fn setOne(self: *KvCells, idx: u32, other: *const KvCells, j: usize) void {
        if (self.pos.items[idx] == -1 and other.pos.items[j] != -1) self.usedInsert(idx);
        if (self.pos.items[idx] != -1 and other.pos.items[j] == -1) self.usedErase(idx);
        if (self.pos.items[idx] != -1) self.seqPosRm(idx);
        self.pos.items[idx] = other.pos.items[j];
        self.ext.items[idx] = other.ext.items[j];
        self.seq.items[idx] = other.seq.items[j];
        if (self.pos.items[idx] != -1) self.seqPosAdd(idx);
        std.debug.assert(self.shift.items[idx] == 0);
    }

    /// Ports `llama_kv_cells`'s `set` over a range (llama-kv-cells.h:166
    /// @b29c606e2): cells `[i, i + other.size())` take `other`'s state.
    pub fn setRange(self: *KvCells, i: u32, other: *const KvCells) void {
        for (0..other.pos.items.len) |j| self.setOne(i + @as(u32, @intCast(j)), other, j);
    }

    /// Ports `llama_kv_cells`'s `set` over indices (llama-kv-cells.h:197
    /// @b29c606e2).
    pub fn setIdxs(self: *KvCells, idxs: []const u32, other: *const KvCells) void {
        std.debug.assert(idxs.len == other.pos.items.len);
        for (idxs, 0..) |idx, j| self.setOne(idx, other, j);
    }

    /// Ports `llama_kv_cells`'s `rm` (llama-kv-cells.h:228 @b29c606e2): clear a
    /// non-empty cell.
    pub fn rm(self: *KvCells, i: u32) void {
        std.debug.assert(self.pos.items[i] != -1);
        self.seqPosRm(i);
        self.seq.items[i] = SeqSet.empty;
        self.pos.items[i] = -1;
        self.ext.items[i] = .{};
        self.shift.items[i] = 0;
        self.usedErase(i);
    }

    /// Ports `llama_kv_cells`'s `seq_rm` (llama-kv-cells.h:244 @b29c606e2).
    ///
    /// Return: true when the cell becomes empty.
    pub fn seqRm(self: *KvCells, i: u32, seq_id: i32) bool {
        const s: usize = @intCast(seq_id);
        std.debug.assert(self.seq.items[i].isSet(s));
        std.debug.assert(self.pos.items[i] != -1);
        self.seq.items[i].unset(s);
        self.seqPosDec(s, i);
        if (self.seq.items[i].count() == 0) {
            self.pos.items[i] = -1;
            self.ext.items[i] = .{};
            self.shift.items[i] = 0;
            self.usedErase(i);
            return true;
        }
        return false;
    }

    /// Ports `llama_kv_cells`'s `seq_keep` (llama-kv-cells.h:267 @b29c606e2).
    ///
    /// Return: true when the cell becomes empty, i.e. did not hold `seq_id`.
    pub fn seqKeep(self: *KvCells, i: u32, seq_id: i32) bool {
        const s: usize = @intCast(seq_id);
        if (self.seq.items[i].isSet(s)) {
            self.seqPosRm(i);
            self.seq.items[i] = SeqSet.empty;
            self.seq.items[i].set(s);
            self.seqPosInc(s, i);
            return false;
        }
        if (self.seq.items[i].count() != 0) {
            self.seqPosRm(i);
            self.seq.items[i] = SeqSet.empty;
            self.pos.items[i] = -1;
            self.ext.items[i] = .{};
            self.shift.items[i] = 0;
            self.usedErase(i);
            return true;
        }
        std.debug.assert(self.pos.items[i] == -1);
        return false;
    }

    /// Ports `llama_kv_cells`'s `seq_count` (llama-kv-cells.h:299 @b29c606e2).
    pub fn seqCount(self: *const KvCells, i: u32) c_int {
        std.debug.assert(self.pos.items[i] != -1);
        return @intCast(self.seq.items[i].count());
    }

    /// Ports `llama_kv_cells`'s `seq_get_all` (llama-kv-cells.h:307
    /// @b29c606e2): every sequence the cell is visible to.
    pub fn seqGetAll(self: *const KvCells, i: u32) SeqSet {
        return self.seq.items[i];
    }

    /// Ports `llama_kv_cells`'s `seq_has` (llama-kv-cells.h:314 @b29c606e2).
    pub fn seqHas(self: *const KvCells, i: u32, seq_id: i32) bool {
        std.debug.assert(seq_id >= 0);
        return self.seq.items[i].isSet(@intCast(seq_id));
    }

    /// Ports `llama_kv_cells`'s `seq_pos_tok_le` (llama-kv-cells.h:325
    /// @b29c606e2): the token of the sequence's cell at the largest
    /// position at or before `p`, the highest cell among equal positions;
    /// `LLAMA_TOKEN_NULL` when there is none. The n-gram input embeddings
    /// recover a ubatch's preceding tokens with it.
    pub fn seqPosTokLe(self: *const KvCells, seq_id: i32, p: i32) i32 {
        std.debug.assert(seq_id >= 0 and seq_id < max_seq);
        const sp = &self.seq_pos[@intCast(seq_id)];
        const it = sp.upperBound(.{ .pos = p, .cell = std.math.maxInt(u32) });
        if (it == 0) return -1;
        return self.ext.items[sp.items.items[it - 1].cell].tok;
    }

    /// Ports `llama_kv_cells`'s `seq_add` (llama-kv-cells.h:340 @b29c606e2).
    pub fn seqAdd(self: *KvCells, i: u32, seq_id: i32) void {
        const s: usize = @intCast(seq_id);
        std.debug.assert(self.pos.items[i] != -1);
        std.debug.assert(!self.seq.items[i].isSet(s));
        self.seq.items[i].set(s);
        self.seqPosInc(s, i);
    }

    /// Ports `llama_kv_cells`'s `seq_get` (llama-kv-cells.h:351 @b29c606e2): the
    /// one sequence of a cell with exactly one.
    pub fn seqGet(self: *const KvCells, i: u32) i32 {
        std.debug.assert(self.seq.items[i].count() == 1);
        if (self.seq.items[i].findFirstSet()) |s| return @intCast(s);
        return -1;
    }

    /// Ports `llama_kv_cells`'s `seq_pos_min` (llama-kv-cells.h:365 @b29c606e2):
    /// -1 when the sequence is absent.
    pub fn seqPosMin(self: *const KvCells, seq_id: i32) i32 {
        const sp = &self.seq_pos[@intCast(seq_id)];
        if (sp.items.items.len == 0) return -1;
        return sp.items.items[0].pos;
    }

    /// Ports `llama_kv_cells`'s `seq_pos_max` (llama-kv-cells.h:378 @b29c606e2):
    /// -1 when the sequence is absent.
    pub fn seqPosMax(self: *const KvCells, seq_id: i32) i32 {
        const sp = &self.seq_pos[@intCast(seq_id)];
        if (sp.items.items.len == 0) return -1;
        return sp.items.items[sp.items.items.len - 1].pos;
    }

    /// Ports `llama_kv_cells`'s `pos_get` (llama-kv-cells.h:390 @b29c606e2).
    pub fn posGet(self: *const KvCells, i: u32) i32 {
        std.debug.assert(self.pos.items[i] != -1);
        return self.pos.items[i];
    }

    /// Ports `llama_kv_cells`'s `ext_get` (llama-kv-cells.h:397 @b29c606e2).
    pub fn extGet(self: *const KvCells, i: u32) CellExt {
        std.debug.assert(self.pos.items[i] != -1);
        return self.ext.items[i];
    }

    /// Ports `llama_kv_cells`'s `get_shift` (llama-kv-cells.h:405 @b29c606e2).
    pub fn getShift(self: *const KvCells, i: u32) i32 {
        std.debug.assert(self.pos.items[i] != -1);
        return self.shift.items[i];
    }

    /// Ports `llama_kv_cells`'s `pos_in` (llama-kv-cells.h:413 @b29c606e2):
    /// whether the cell's position is in `[p0, p1)`.
    pub fn posIn(self: *const KvCells, i: u32, p0: i32, p1: i32) bool {
        return self.pos.items[i] >= p0 and self.pos.items[i] < p1;
    }

    /// Ports `llama_kv_cells`'s `pos_set` (llama-kv-cells.h:422 @b29c606e2): the
    /// position of an empty cell.
    pub fn posSet(self: *KvCells, i: u32, p: i32) void {
        std.debug.assert(self.pos.items[i] == -1);
        std.debug.assert(self.seq.items[i].count() == 0);
        self.pos.items[i] = p;
        self.usedInsert(i);
    }

    /// Ports `llama_kv_cells`'s `ext_set` (llama-kv-cells.h:432 @b29c606e2).
    pub fn extSet(self: *KvCells, i: u32, p: CellExt) void {
        self.ext.items[i] = p;
    }

    /// Ports `llama_kv_cells`'s `pos_add` (llama-kv-cells.h:440 @b29c606e2):
    /// shift a cell's position, emptying it when it goes negative.
    ///
    /// Return: true when the cell becomes empty.
    pub fn posAdd(self: *KvCells, i: u32, d: i32) bool {
        std.debug.assert(self.pos.items[i] != -1);
        self.seqPosRm(i);
        self.pos.items[i] +%= d;
        self.shift.items[i] +%= d;
        self.has_shift = true;
        if (self.pos.items[i] < 0) {
            self.seq.items[i] = SeqSet.empty;
            self.pos.items[i] = -1;
            self.shift.items[i] = 0;
            self.usedErase(i);
            return true;
        }
        self.seqPosAdd(i);
        return false;
    }

    /// Ports `llama_kv_cells`'s `pos_div` (llama-kv-cells.h:469 @b29c606e2).
    pub fn posDiv(self: *KvCells, i: u32, d: c_int) void {
        std.debug.assert(self.pos.items[i] != -1);
        const p_old = self.pos.items[i];
        self.seqPosRm(i);
        self.pos.items[i] = @divTrunc(self.pos.items[i], d);
        self.shift.items[i] +%= p_old -% self.pos.items[i];
        self.seqPosAdd(i);
        self.has_shift = true;
    }

    /// Ports `llama_kv_cells`'s `seq_pos_dec` (llama-kv-cells.h:528 @b29c606e2).
    fn seqPosDec(self: *KvCells, s: usize, i: u32) void {
        self.seq_pos[s].erase(.{ .pos = self.pos.items[i], .cell = i });
    }

    /// Ports `llama_kv_cells`'s `seq_pos_inc` (llama-kv-cells.h:534 @b29c606e2).
    fn seqPosInc(self: *KvCells, s: usize, i: u32) void {
        self.seq_pos[s].insert(.{ .pos = self.pos.items[i], .cell = i });
    }

    /// Ports `llama_kv_cells`'s `seq_pos_rm` (llama-kv-cells.h:539 @b29c606e2).
    fn seqPosRm(self: *KvCells, i: u32) void {
        var it = self.seq.items[i].iterator(.{});
        while (it.next()) |s| self.seqPosDec(s, i);
    }

    /// Ports `llama_kv_cells`'s `seq_pos_add` (llama-kv-cells.h:548 @b29c606e2).
    fn seqPosAdd(self: *KvCells, i: u32) void {
        var it = self.seq.items[i].iterator(.{});
        while (it.next()) |s| self.seqPosInc(s, i);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(CellExt);
    std.testing.refAllDecls(KvCells);
}

test "positions count per sequence and min/max follow removals" {
    var cells: KvCells = .{};
    defer cells.deinit();
    cells.resize(8);
    cells.posSet(0, 5);
    cells.seqAdd(0, 1);
    cells.posSet(3, 2);
    cells.seqAdd(3, 1);
    cells.posSet(4, 5);
    cells.seqAdd(4, 1);
    try std.testing.expectEqual(@as(i32, 2), cells.seqPosMin(1));
    try std.testing.expectEqual(@as(i32, 5), cells.seqPosMax(1));
    try std.testing.expectEqual(@as(u32, 0), cells.usedMin());
    try std.testing.expectEqual(@as(u32, 5), cells.usedMaxP1());
    _ = cells.seqRm(0, 1);
    try std.testing.expectEqual(@as(i32, 5), cells.seqPosMax(1));
    _ = cells.seqRm(4, 1);
    try std.testing.expectEqual(@as(i32, 2), cells.seqPosMax(1));
    try std.testing.expectEqual(@as(u32, 1), cells.getUsed());
    try std.testing.expectEqual(@as(i32, -1), cells.seqPosMin(0));
}

test "pos_add past zero empties the cell; pos_div truncates" {
    var cells: KvCells = .{};
    defer cells.deinit();
    cells.resize(2);
    cells.posSet(0, 3);
    cells.seqAdd(0, 0);
    try std.testing.expect(cells.posAdd(0, -4));
    try std.testing.expect(cells.isEmpty(0));
    cells.posSet(1, 7);
    cells.seqAdd(1, 0);
    cells.posDiv(1, 2);
    try std.testing.expectEqual(@as(i32, 3), cells.posGet(1));
    try std.testing.expectEqual(@as(i32, 4), cells.getShift(1));
}

test "seq_pos_tok_le finds the nearest cell at or before a position" {
    var cells: KvCells = .{};
    defer cells.deinit();
    cells.resize(6);
    // Two cells share position 4; the higher index wins.
    for ([_]struct { u32, i32, i32 }{ .{ 0, 1, 10 }, .{ 2, 4, 11 }, .{ 5, 4, 12 }, .{ 3, 7, 13 } }) |e| {
        cells.posSet(e[0], e[1]);
        cells.extSet(e[0], .{ .tok = e[2] });
        cells.seqAdd(e[0], 0);
    }
    try std.testing.expectEqual(@as(i32, -1), cells.seqPosTokLe(0, 0));
    try std.testing.expectEqual(@as(i32, 10), cells.seqPosTokLe(0, 3));
    try std.testing.expectEqual(@as(i32, 12), cells.seqPosTokLe(0, 4));
    try std.testing.expectEqual(@as(i32, 13), cells.seqPosTokLe(0, 100));
    // Removing one of the shared positions keeps the other.
    _ = cells.seqRm(5, 0);
    try std.testing.expectEqual(@as(i32, 11), cells.seqPosTokLe(0, 6));
    try std.testing.expectEqual(@as(i32, 7), cells.seqPosMax(0));
    try std.testing.expectEqual(@as(i32, -1), cells.seqPosTokLe(1, 6));
}
