//! The bookkeeping behind a KV cache: which cells are used, at which
//! position, by which sequences.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-kv-cells.h` at v0.3.0 (`c1d0e7a00`):
//! `llama_kv_cell_ext` and `llama_kv_cells`.
//!
//! # The ordered containers
//!
//! The C++ keeps `used` in a `std::set<uint32_t>` and each sequence's
//! positions in a `std::map<llama_pos, int>` of counts, and asks of them
//! only their size, smallest and largest key. Here `used` is a bit per cell
//! with a count, and each sequence's positions a sorted array of
//! (position, count) pairs: positions arrive mostly in increasing order, so
//! an insertion is almost always an append. The answers are the same; only
//! the asymptotics of a removal from the middle differ.

const std = @import("std");
const impl = @import("impl.zig");

const allocator = impl.allocator;

/// `LLAMA_MAX_SEQ` (llama-cparams.h:8 @c1d0e7a00).
pub const max_seq = 256;

/// `std::bitset<LLAMA_MAX_SEQ>`: which sequences hold a cell.
pub const SeqSet = std.bit_set.StaticBitSet(max_seq);

fn oom() noreturn {
    impl.abort("llama-kv-cells.h", 0, "out of memory");
}

/// Mirrors `llama_kv_cell_ext` (llama-kv-cells.h:13 @c1d0e7a00): the 2D
/// position M-RoPE models give a cell.
pub const CellExt = extern struct {
    x: i32 = 0,
    y: i32 = 0,

    /// Ports `llama_kv_cell_ext`'s `is_2d_gt` (llama-kv-cells.h:19 @c1d0e7a00).
    pub fn is2dGt(self: CellExt, ox: i32, oy: i32) bool {
        return (self.y > oy) or (self.y == oy and self.x > ox);
    }
};

/// A sequence's positions and how often each occurs, smallest first.
const PosCounts = struct {
    items: std.ArrayList(struct { pos: i32, n: i32 }) = .empty,

    fn find(self: *const PosCounts, p: i32) struct { usize, bool } {
        var lo: usize = 0;
        var hi: usize = self.items.items.len;
        // Fast path: positions mostly arrive in order.
        if (hi > 0 and self.items.items[hi - 1].pos < p) return .{ hi, false };
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.items.items[mid].pos < p) lo = mid + 1 else hi = mid;
        }
        return .{ lo, lo < self.items.items.len and self.items.items[lo].pos == p };
    }

    fn inc(self: *PosCounts, p: i32) void {
        const at, const found = self.find(p);
        if (found) {
            self.items.items[at].n += 1;
        } else {
            self.items.insert(allocator, at, .{ .pos = p, .n = 1 }) catch oom();
        }
    }

    fn dec(self: *PosCounts, p: i32) void {
        const at, const found = self.find(p);
        std.debug.assert(found);
        self.items.items[at].n -= 1;
        if (self.items.items[at].n == 0) _ = self.items.orderedRemove(at);
    }
};

/// Ports `llama_kv_cells` (llama-kv-cells.h:32 @c1d0e7a00).
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
    /// How often each position is present, per sequence.
    seq_pos: [max_seq]PosCounts = @splat(.{}),

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

    /// Ports `llama_kv_cells`'s `reset` (llama-kv-cells.h:34 @c1d0e7a00).
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

    /// Ports `llama_kv_cells`'s `reset_shift` (llama-kv-cells.h:51 @c1d0e7a00).
    pub fn resetShift(self: *KvCells) void {
        self.has_shift = false;
        @memset(self.shift.items, 0);
    }

    /// Ports `llama_kv_cells`'s `size` (llama-kv-cells.h:59 @c1d0e7a00).
    pub fn size(self: *const KvCells) u32 {
        return @intCast(self.pos.items.len);
    }

    /// Ports `llama_kv_cells`'s `resize` (llama-kv-cells.h:63 @c1d0e7a00).
    pub fn resize(self: *KvCells, n: u32) void {
        self.pos.resize(allocator, n) catch oom();
        self.ext.resize(allocator, n) catch oom();
        self.shift.resize(allocator, n) catch oom();
        self.seq.resize(allocator, n) catch oom();
        self.used.resize(allocator, n, false) catch oom();
        self.reset();
    }

    /// Ports `llama_kv_cells`'s `is_empty` (llama-kv-cells.h:72 @c1d0e7a00).
    pub fn isEmpty(self: *const KvCells, i: u32) bool {
        return self.pos.items[i] == -1;
    }

    /// Ports `llama_kv_cells`'s `get_used` (llama-kv-cells.h:79 @c1d0e7a00).
    pub fn getUsed(self: *const KvCells) u32 {
        return self.n_used;
    }

    /// Ports `llama_kv_cells`'s `used_min` (llama-kv-cells.h:85 @c1d0e7a00):
    /// the first used cell, or 0 when none is.
    pub fn usedMin(self: *const KvCells) u32 {
        if (self.n_used == 0) return 0;
        return @intCast(self.used.findFirstSet().?);
    }

    /// Ports `llama_kv_cells`'s `used_max_p1` (llama-kv-cells.h:91 @c1d0e7a00):
    /// one past the last used cell, or 0 when none is.
    pub fn usedMaxP1(self: *const KvCells) u32 {
        if (self.n_used == 0) return 0;
        return @intCast(self.used.findLastSet().? + 1);
    }

    /// Ports `llama_kv_cells`'s `get_has_shift` (llama-kv-cells.h:95 @c1d0e7a00).
    pub fn getHasShift(self: *const KvCells) bool {
        return self.has_shift;
    }

    /// Ports `llama_kv_cells`'s `cp` over a range (llama-kv-cells.h:120
    /// @c1d0e7a00): cells `[i, i + n)`, for saving the state.
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

    /// Ports `llama_kv_cells`'s `cp` over indices (llama-kv-cells.h:141
    /// @c1d0e7a00).
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

    /// Ports `llama_kv_cells`'s `set` over a range (llama-kv-cells.h:160
    /// @c1d0e7a00): cells `[i, i + other.size())` take `other`'s state.
    pub fn setRange(self: *KvCells, i: u32, other: *const KvCells) void {
        for (0..other.pos.items.len) |j| self.setOne(i + @as(u32, @intCast(j)), other, j);
    }

    /// Ports `llama_kv_cells`'s `set` over indices (llama-kv-cells.h:191
    /// @c1d0e7a00).
    pub fn setIdxs(self: *KvCells, idxs: []const u32, other: *const KvCells) void {
        std.debug.assert(idxs.len == other.pos.items.len);
        for (idxs, 0..) |idx, j| self.setOne(idx, other, j);
    }

    /// Ports `llama_kv_cells`'s `rm` (llama-kv-cells.h:222 @c1d0e7a00): clear a
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

    /// Ports `llama_kv_cells`'s `seq_rm` (llama-kv-cells.h:238 @c1d0e7a00).
    ///
    /// Return: true when the cell becomes empty.
    pub fn seqRm(self: *KvCells, i: u32, seq_id: i32) bool {
        const s: usize = @intCast(seq_id);
        std.debug.assert(self.seq.items[i].isSet(s));
        std.debug.assert(self.pos.items[i] != -1);
        self.seq.items[i].unset(s);
        self.seq_pos[s].dec(self.pos.items[i]);
        if (self.seq.items[i].count() == 0) {
            self.pos.items[i] = -1;
            self.ext.items[i] = .{};
            self.shift.items[i] = 0;
            self.usedErase(i);
            return true;
        }
        return false;
    }

    /// Ports `llama_kv_cells`'s `seq_keep` (llama-kv-cells.h:261 @c1d0e7a00).
    ///
    /// Return: true when the cell becomes empty, i.e. did not hold `seq_id`.
    pub fn seqKeep(self: *KvCells, i: u32, seq_id: i32) bool {
        const s: usize = @intCast(seq_id);
        if (self.seq.items[i].isSet(s)) {
            self.seqPosRm(i);
            self.seq.items[i] = SeqSet.empty;
            self.seq.items[i].set(s);
            self.seq_pos[s].inc(self.pos.items[i]);
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

    /// Ports `llama_kv_cells`'s `seq_count` (llama-kv-cells.h:293 @c1d0e7a00).
    pub fn seqCount(self: *const KvCells, i: u32) c_int {
        std.debug.assert(self.pos.items[i] != -1);
        return @intCast(self.seq.items[i].count());
    }

    /// Ports `llama_kv_cells`'s `seq_has` (llama-kv-cells.h:301 @c1d0e7a00).
    pub fn seqHas(self: *const KvCells, i: u32, seq_id: i32) bool {
        std.debug.assert(seq_id >= 0);
        return self.seq.items[i].isSet(@intCast(seq_id));
    }

    /// Ports `llama_kv_cells`'s `seq_add` (llama-kv-cells.h:309 @c1d0e7a00).
    pub fn seqAdd(self: *KvCells, i: u32, seq_id: i32) void {
        const s: usize = @intCast(seq_id);
        std.debug.assert(self.pos.items[i] != -1);
        std.debug.assert(!self.seq.items[i].isSet(s));
        self.seq.items[i].set(s);
        self.seq_pos[s].inc(self.pos.items[i]);
    }

    /// Ports `llama_kv_cells`'s `seq_get` (llama-kv-cells.h:320 @c1d0e7a00): the
    /// one sequence of a cell with exactly one.
    pub fn seqGet(self: *const KvCells, i: u32) i32 {
        std.debug.assert(self.seq.items[i].count() == 1);
        if (self.seq.items[i].findFirstSet()) |s| return @intCast(s);
        return -1;
    }

    /// Ports `llama_kv_cells`'s `seq_pos_min` (llama-kv-cells.h:334 @c1d0e7a00):
    /// -1 when the sequence is absent.
    pub fn seqPosMin(self: *const KvCells, seq_id: i32) i32 {
        const sp = &self.seq_pos[@intCast(seq_id)];
        if (sp.items.items.len == 0) return -1;
        return sp.items.items[0].pos;
    }

    /// Ports `llama_kv_cells`'s `seq_pos_max` (llama-kv-cells.h:349 @c1d0e7a00):
    /// -1 when the sequence is absent.
    pub fn seqPosMax(self: *const KvCells, seq_id: i32) i32 {
        const sp = &self.seq_pos[@intCast(seq_id)];
        if (sp.items.items.len == 0) return -1;
        return sp.items.items[sp.items.items.len - 1].pos;
    }

    /// Ports `llama_kv_cells`'s `pos_get` (llama-kv-cells.h:363 @c1d0e7a00).
    pub fn posGet(self: *const KvCells, i: u32) i32 {
        std.debug.assert(self.pos.items[i] != -1);
        return self.pos.items[i];
    }

    /// Ports `llama_kv_cells`'s `ext_get` (llama-kv-cells.h:370 @c1d0e7a00).
    pub fn extGet(self: *const KvCells, i: u32) CellExt {
        std.debug.assert(self.pos.items[i] != -1);
        return self.ext.items[i];
    }

    /// Ports `llama_kv_cells`'s `get_shift` (llama-kv-cells.h:378 @c1d0e7a00).
    pub fn getShift(self: *const KvCells, i: u32) i32 {
        std.debug.assert(self.pos.items[i] != -1);
        return self.shift.items[i];
    }

    /// Ports `llama_kv_cells`'s `pos_in` (llama-kv-cells.h:386 @c1d0e7a00):
    /// whether the cell's position is in `[p0, p1)`.
    pub fn posIn(self: *const KvCells, i: u32, p0: i32, p1: i32) bool {
        return self.pos.items[i] >= p0 and self.pos.items[i] < p1;
    }

    /// Ports `llama_kv_cells`'s `pos_set` (llama-kv-cells.h:395 @c1d0e7a00): the
    /// position of an empty cell.
    pub fn posSet(self: *KvCells, i: u32, p: i32) void {
        std.debug.assert(self.pos.items[i] == -1);
        std.debug.assert(self.seq.items[i].count() == 0);
        self.pos.items[i] = p;
        self.usedInsert(i);
    }

    /// Ports `llama_kv_cells`'s `ext_set` (llama-kv-cells.h:405 @c1d0e7a00).
    pub fn extSet(self: *KvCells, i: u32, p: CellExt) void {
        self.ext.items[i] = p;
    }

    /// Ports `llama_kv_cells`'s `pos_add` (llama-kv-cells.h:413 @c1d0e7a00):
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

    /// Ports `llama_kv_cells`'s `pos_div` (llama-kv-cells.h:442 @c1d0e7a00).
    pub fn posDiv(self: *KvCells, i: u32, d: c_int) void {
        std.debug.assert(self.pos.items[i] != -1);
        const p_old = self.pos.items[i];
        self.seqPosRm(i);
        self.pos.items[i] = @divTrunc(self.pos.items[i], d);
        self.shift.items[i] +%= p_old -% self.pos.items[i];
        self.seqPosAdd(i);
        self.has_shift = true;
    }

    /// Ports `llama_kv_cells`'s `seq_pos_rm` (llama-kv-cells.h:517 @c1d0e7a00).
    fn seqPosRm(self: *KvCells, i: u32) void {
        var it = self.seq.items[i].iterator(.{});
        while (it.next()) |s| self.seq_pos[s].dec(self.pos.items[i]);
    }

    /// Ports `llama_kv_cells`'s `seq_pos_add` (llama-kv-cells.h:526 @c1d0e7a00).
    fn seqPosAdd(self: *KvCells, i: u32) void {
        var it = self.seq.items[i].iterator(.{});
        while (it.next()) |s| self.seq_pos[s].inc(self.pos.items[i]);
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
