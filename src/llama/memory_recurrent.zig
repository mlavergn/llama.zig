//! The recurrent memory: one state per sequence, for Mamba-, RWKV- and
//! gated-delta-net layers.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-memory-recurrent.h` -- `llama_memory_recurrent`,
//!   its `mem_cell`, and `llama_memory_recurrent_context`
//! - `llama.cpp/src/llama-memory-recurrent.cpp` -- their bodies
//!
//! All at v0.4.1 (`b29c606e2`).
//!
//! # A cell's sequences
//!
//! `mem_cell`'s `seq_id` is a `std::set<llama_seq_id>`; every id is below
//! `LLAMA_MAX_SEQ`, so here it is a bitset, whose ascending iteration is the
//! set's order.
//!
//! # What the reference compiles out
//!
//! `find_slot` cross-checks every tail under `#ifndef NDEBUG`, logging what
//! it finds wrong. The reference is built with `NDEBUG`, so the check is
//! not here.

const std = @import("std");
const impl = @import("impl.zig");
const cparams_mod = @import("cparams.zig");
const hparams_mod = @import("hparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const io = @import("io.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;
const Ubatch = batch.Ubatch;
const Tensor = c.ggml_tensor;

fn oom() noreturn {
    impl.abort("llama-memory-recurrent.cpp", 0, "out of memory");
}

const SeqSet = std.bit_set.StaticBitSet(cparams_mod.max_seq);

/// Mirrors `llama_memory_recurrent`'s `mem_cell` (llama-memory-recurrent.h:88 @b29c606e2).
pub const MemCell = struct {
    pos: i32 = -1,
    /// Where the state is copied from.
    src: i32 = -1,
    /// Like `src`, but only for setting the inputs, so a state copies once.
    src0: i32 = -1,
    tail: i32 = -1,
    seq_id: SeqSet = SeqSet.empty,

    pub fn hasSeqId(self: *const MemCell, id: i32) bool {
        if (id < 0 or id >= cparams_mod.max_seq) return false;
        return self.seq_id.isSet(@intCast(id));
    }
    pub fn isEmpty(self: *const MemCell) bool {
        return self.seq_id.count() == 0;
    }
};

/// Ports `llama_memory_recurrent` (llama-memory-recurrent.h:17 @b29c606e2).
pub const MemoryRecurrent = struct {
    /// Where the batch will be placed in the cache.
    head: u32 = 0,
    /// Cells, shared across all sequences.
    size: u32 = 0,
    /// Cells with at least one sequence.
    used: u32 = 0,
    n_rs_seq: u32 = 0,
    rs_idx: std.ArrayList(u32) = .empty,
    n: u32 = 0,
    rs_z: i32 = -1,

    cells: std.ArrayList(MemCell) = .empty,
    r_l: std.ArrayList(?*Tensor) = .empty,
    s_l: std.ArrayList(?*Tensor) = .empty,
    /// A second conv history, PLE's, which must stay replicated across
    /// devices and so cannot share the r row.
    p_l: std.ArrayList(?*Tensor) = .empty,

    hparams: *const Hparams,
    n_seq_max: u32 = 1,
    ctxs_bufs: std.ArrayList(struct { ctx: *c.ggml_context, buf: c.ggml_backend_buffer_t }) = .empty,

    /// Ports the constructor of `llama_memory_recurrent`
    /// (llama-memory-recurrent.cpp:20 @b29c606e2).
    ///
    /// Return: the memory, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        type_r: c.enum_ggml_type,
        type_s: c.enum_ggml_type,
        offload: bool,
        mem_size: u32,
        n_seq_max: u32,
        n_rs_seq: u32,
        filter: memory.LayerFilter,
        exc: *Exception,
    ) Thrown!*MemoryRecurrent {
        const func = "llama_memory_recurrent";
        const self = allocator.create(MemoryRecurrent) catch oom();
        self.* = .{ .hparams = &model.hparams, .n_seq_max = n_seq_max };
        errdefer self.destroy();
        const hparams = self.hparams;
        const n_layer: i32 = @intCast(hparams.nLayer());

        self.head = 0;
        self.size = mem_size;
        self.used = 0;
        self.n_rs_seq = n_rs_seq;
        self.rs_idx.appendNTimes(allocator, 0, n_seq_max) catch oom();
        self.cells.appendNTimes(allocator, .{}, mem_size) catch oom();

        // One context per buffer type, ordered by the buffer type's name.
        var ctx_map: std.ArrayList(struct { buft: c.ggml_backend_buffer_type_t, ctx: *c.ggml_context }) = .empty;
        var n_moved: usize = 0;
        defer {
            for (ctx_map.items[n_moved..]) |e| c.ggml_free(e.ctx);
            ctx_map.deinit(allocator);
        }

        self.r_l.appendNTimes(allocator, null, @intCast(n_layer)) catch oom();
        self.s_l.appendNTimes(allocator, null, @intCast(n_layer)) catch oom();
        self.p_l.appendNTimes(allocator, null, @intCast(n_layer)) catch oom();

        var i: i32 = 0;
        while (i < n_layer) : (i += 1) {
            if (filter.func != null and !filter.call(i)) {
                impl.logDebug("%s: layer %3d: skipped\n", .{ func, i });
                continue;
            }
            var dev_name: [*:0]const u8 = "CPU";
            var buft = c.ggml_backend_cpu_buffer_type();
            if (offload) {
                const dev = model.devLayer(@intCast(i));
                buft = c.ggml_backend_dev_buffer_type(dev);
                dev_name = c.ggml_backend_dev_name(dev);
            }
            impl.logDebug("%s, layer %3d: dev = %s\n", .{ func, i, dev_name });

            const ctx = ctxForBuft(&ctx_map, buft, n_layer, hparams.pleConvState() > 0) orelse
                return exc.throw(.runtime_error, "failed to create ggml context for rs cache", .{});

            const n_rows: i64 = @as(i64, mem_size) * (1 + n_rs_seq);
            const r = c.ggml_new_tensor_2d(ctx, type_r, hparams.nEmbdR(), n_rows);
            const s = c.ggml_new_tensor_2d(ctx, type_s, hparams.nEmbdS(), n_rows);
            _ = c.ggml_format_name(r, "cache_r_l%d", i);
            _ = c.ggml_format_name(s, "cache_s_l%d", i);
            self.r_l.items[@intCast(i)] = r;
            self.s_l.items[@intCast(i)] = s;

            // The PLE history needs its own row: the meta backend must
            // mirror it while the delta-net conv state beside it is split.
            if (hparams.pleConvState() > 0 and hparams.isPle(@intCast(i))) {
                const p = c.ggml_new_tensor_2d(ctx, type_r, hparams.pleConvState(), n_rows);
                _ = c.ggml_format_name(p, "cache_ple_r_l%d", i);
                self.p_l.items[@intCast(i)] = p;
            }
        }

        // Allocate the tensors and clear the buffers, so the padding holds
        // no NaNs.
        for (ctx_map.items) |e| {
            const buf = c.ggml_backend_alloc_ctx_tensors_from_buft(e.ctx, e.buft) orelse
                return exc.throw(.runtime_error, "failed to allocate buffer for rs cache", .{});
            c.ggml_backend_buffer_clear(buf, 0);
            impl.logInfo("%s: %10s RS buffer size = %8.2f MiB\n", .{ func, c.ggml_backend_buffer_name(buf), @as(f64, @floatFromInt(c.ggml_backend_buffer_get_size(buf))) / 1024.0 / 1024.0 });
            self.ctxs_bufs.append(allocator, .{ .ctx = e.ctx, .buf = buf }) catch oom();
            n_moved += 1;
        }

        {
            const memory_size_r = self.sizeRBytes();
            const memory_size_s = self.sizeSBytes();
            const memory_size_p = self.sizePBytes();
            impl.logInfo("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB, P (%s): %7.2f MiB\n", .{
                func,
                @as(f64, @as(f32, @floatFromInt(memory_size_r + memory_size_s + memory_size_p)) / (1024.0 * 1024.0)),
                mem_size,
                n_layer,
                n_seq_max,
                n_rs_seq,
                c.ggml_type_name(type_r),
                @as(f64, @as(f32, @floatFromInt(memory_size_r)) / (1024.0 * 1024.0)),
                c.ggml_type_name(type_s),
                @as(f64, @as(f32, @floatFromInt(memory_size_s)) / (1024.0 * 1024.0)),
                c.ggml_type_name(type_r),
                @as(f64, @as(f32, @floatFromInt(memory_size_p)) / (1024.0 * 1024.0)),
            });
        }
        return self;
    }

    /// `ctx_for_buft`: r and s per layer, plus the separate PLE conv row
    /// where the model has one.
    fn ctxForBuft(map: anytype, buft: c.ggml_backend_buffer_type_t, n_layer: i32, has_ple: bool) ?*c.ggml_context {
        const name = std.mem.span(c.ggml_backend_buft_name(buft));
        var pos: usize = 0;
        while (pos < map.items.len) : (pos += 1) {
            const order = std.mem.order(u8, std.mem.span(c.ggml_backend_buft_name(map.items[pos].buft)), name);
            if (order == .eq) return map.items[pos].ctx;
            if (order == .gt) break;
        }
        const n_per_layer: usize = if (has_ple) 3 else 2;
        const ctx = c.ggml_init(.{ .mem_size = n_per_layer * @as(usize, @intCast(n_layer)) * c.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = true }) orelse return null;
        map.insert(allocator, pos, .{ .buft = buft, .ctx = ctx }) catch oom();
        return ctx;
    }

    /// Return: nothing.
    pub fn destroy(self: *MemoryRecurrent) void {
        for (self.ctxs_bufs.items) |e| {
            c.ggml_backend_buffer_free(e.buf);
            c.ggml_free(e.ctx);
        }
        self.ctxs_bufs.deinit(allocator);
        self.rs_idx.deinit(allocator);
        self.cells.deinit(allocator);
        self.r_l.deinit(allocator);
        self.s_l.deinit(allocator);
        self.p_l.deinit(allocator);
        allocator.destroy(self);
    }

    /// Ports `llama_memory_recurrent`'s `clear` (llama-memory-recurrent.cpp:141 @b29c606e2).
    pub fn clear(self: *MemoryRecurrent, data: bool) void {
        for (self.cells.items[0..self.size]) |*cell| {
            cell.pos = -1;
            cell.seq_id = SeqSet.empty;
            cell.src = -1;
            cell.tail = -1;
        }
        self.head = 0;
        self.used = 0;
        if (data) {
            for (self.ctxs_bufs.items) |e| c.ggml_backend_buffer_clear(e.buf, 0);
        }
        @memset(self.rs_idx.items, 0);
    }

    /// Ports `llama_memory_recurrent`'s `seq_rm` (llama-memory-recurrent.cpp:161 @b29c606e2).
    pub fn seqRm(self: *MemoryRecurrent, seq_id: i32, p0_: i32, p1_: i32) bool {
        var new_head = self.size;
        const p0 = if (p0_ < 0) 0 else p0_;
        const p1 = if (p1_ < 0) std.math.maxInt(i32) else p1_;

        if (@as(u32, @bitCast(seq_id)) >= self.n_seq_max) {
            impl.logError("%s: invalid seq_id (%d) - larger than n_seq_max (%d)\n", .{ "seq_rm", seq_id, self.n_seq_max });
            return false;
        }

        const rm_all = p0 == 0 and p1 == std.math.maxInt(i32);
        if (rm_all) self.setRsIdx(seq_id, 0);

        // A Mamba or RWKV state cannot be partly erased at the end: the
        // states of earlier tokens are not kept.
        if (seq_id >= @as(i64, self.size)) return false;
        if (0 <= seq_id) {
            const tail_id = &self.cells.items[@intCast(seq_id)].tail;
            if (tail_id.* >= 0) {
                const cell = &self.cells.items[@intCast(tail_id.*)];
                // A partial rollback through a snapshot, bounded by n_rs_seq.
                if (0 < p0 and p0 <= cell.pos and p1 > cell.pos) {
                    const rollback = cell.pos - (p0 - 1);
                    // A pending rollback is single-use.
                    const pending = self.rs_idx.items[@intCast(seq_id)] != 0;
                    if (!pending and rollback >= 1 and rollback <= @as(i32, @intCast(self.n_rs_seq))) {
                        self.setRsIdx(seq_id, @intCast(rollback));
                        cell.pos = p0 - 1;
                        return true;
                    }
                    return false;
                }
                // Invalidate the tails about to be cleared.
                if (p0 <= cell.pos and cell.pos < p1) tail_id.* = -1;
            }
        } else {
            // A negative seq_id: the range covers everything or nothing.
            if (p0 != p1 and (p0 != 0 or p1 != std.math.maxInt(i32))) return false;
        }

        for (self.cells.items[0..self.size], 0..) |*cell, i| {
            if (cell.pos >= p0 and cell.pos < p1) {
                if (seq_id < 0) {
                    cell.seq_id = SeqSet.empty;
                } else if (cell.hasSeqId(seq_id)) {
                    cell.seq_id.unset(@intCast(seq_id));
                } else {
                    continue;
                }
                if (cell.isEmpty()) {
                    // Keep count of the used cells.
                    if (cell.pos >= 0) self.used -%= 1;
                    cell.pos = -1;
                    cell.src = -1;
                    if (new_head == self.size) new_head = @intCast(i);
                }
            }
        }
        // A freed slot is where the next search starts.
        if (new_head != self.size and new_head < self.head) self.head = new_head;
        return true;
    }

    /// Ports `llama_memory_recurrent`'s `seq_cp` (llama-memory-recurrent.cpp:249 @b29c606e2).
    pub fn seqCp(self: *MemoryRecurrent, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) void {
        _ = p0;
        _ = p1;
        if (seq_id_src == seq_id_dst) return;
        if (@as(u32, @bitCast(seq_id_dst)) < self.size and @as(u32, @bitCast(seq_id_src)) < self.size) {
            const tail_src = &self.cells.items[@intCast(seq_id_src)];
            const tail_dst = &self.cells.items[@intCast(seq_id_dst)];
            if (tail_dst.tail >= 0) {
                // Clear the destination sequence.
                const cell_dst = &self.cells.items[@intCast(tail_dst.tail)];
                cell_dst.seq_id.unset(@intCast(seq_id_dst));
                tail_dst.tail = -1;
                if (cell_dst.isEmpty()) {
                    cell_dst.pos = -1;
                    cell_dst.src = -1;
                    self.used -%= 1;
                }
            }
            if (tail_src.tail >= 0) {
                const cell_src = &self.cells.items[@intCast(tail_src.tail)];
                cell_src.seq_id.set(@intCast(seq_id_dst));
                tail_dst.tail = tail_src.tail;
            }
        }
    }

    /// Ports `llama_memory_recurrent`'s `seq_keep` (llama-memory-recurrent.cpp:286 @b29c606e2).
    pub fn seqKeep(self: *MemoryRecurrent, seq_id: i32) void {
        var new_head = self.size;
        for (self.cells.items[0..self.size], 0..) |*cell, i| {
            if (@as(i32, @intCast(i)) != seq_id) cell.tail = -1;
            if (!cell.hasSeqId(seq_id)) {
                if (cell.pos >= 0) self.used -%= 1;
                cell.pos = -1;
                cell.src = -1;
                cell.seq_id = SeqSet.empty;
                if (new_head == self.size) new_head = @intCast(i);
            } else {
                cell.seq_id = SeqSet.empty;
                cell.seq_id.set(@intCast(seq_id));
            }
        }
        if (new_head != self.size and new_head < self.head) self.head = new_head;
    }

    /// Ports `llama_memory_recurrent`'s `seq_add` (llama-memory-recurrent.cpp:318
    /// @b29c606e2): only the position moves.
    pub fn seqAdd(self: *MemoryRecurrent, seq_id: i32, p0_: i32, p1_: i32, shift: i32) void {
        if (shift == 0) return;
        const p0 = if (p0_ < 0) 0 else p0_;
        const p1 = if (p1_ < 0) std.math.maxInt(i32) else p1_;
        if (p0 == p1) return;
        if (0 <= seq_id and seq_id < @as(i64, self.size)) {
            const tail_id = self.cells.items[@intCast(seq_id)].tail;
            if (tail_id >= 0) {
                const cell = &self.cells.items[@intCast(tail_id)];
                if (cell.hasSeqId(seq_id) and p0 <= cell.pos and cell.pos < p1) cell.pos +%= shift;
            }
        }
    }

    /// Ports `llama_memory_recurrent`'s `seq_div` (llama-memory-recurrent.cpp:348 @b29c606e2).
    pub fn seqDiv(self: *MemoryRecurrent, seq_id: i32, p0_: i32, p1_: i32, d: c_int) void {
        if (d == 1) return;
        const p0 = if (p0_ < 0) 0 else p0_;
        const p1 = if (p1_ < 0) std.math.maxInt(i32) else p1_;
        if (p0 == p1) return;
        if (0 <= seq_id and seq_id < @as(i64, self.size)) {
            const tail_id = self.cells.items[@intCast(seq_id)].tail;
            if (tail_id >= 0) {
                const cell = &self.cells.items[@intCast(tail_id)];
                if (cell.hasSeqId(seq_id) and p0 <= cell.pos and cell.pos < p1) cell.pos = @divTrunc(cell.pos, d);
            }
        }
    }

    /// Ports `llama_memory_recurrent`'s `seq_pos_min` (llama-memory-recurrent.cpp:378 @b29c606e2).
    pub fn seqPosMin(self: *const MemoryRecurrent, seq_id: i32) i32 {
        var result: i32 = std.math.maxInt(i32);
        for (self.cells.items[0..self.size]) |*cell| {
            if (cell.hasSeqId(seq_id)) result = @min(result, cell.pos);
        }
        if (result == std.math.maxInt(i32)) result = -1;
        return result;
    }

    /// Ports `llama_memory_recurrent`'s `seq_pos_max` (llama-memory-recurrent.cpp:394 @b29c606e2).
    pub fn seqPosMax(self: *const MemoryRecurrent, seq_id: i32) i32 {
        var result: i32 = -1;
        for (self.cells.items[0..self.size]) |*cell| {
            if (cell.hasSeqId(seq_id)) result = @max(result, cell.pos);
        }
        return result;
    }

    /// Ports `llama_memory_recurrent`'s `set_rs_idx` (llama-memory-recurrent.cpp:406 @b29c606e2).
    pub fn setRsIdx(self: *MemoryRecurrent, seq_id: i32, idx: u32) void {
        if (seq_id < 0) {
            @memset(self.rs_idx.items, 0);
            return;
        }
        std.debug.assert(self.n_seq_max == self.rs_idx.items.len);
        impl.assert(@as(u32, @bitCast(seq_id)) < self.n_seq_max, "llama-memory-recurrent.cpp", 414, "(uint32_t) seq_id < n_seq_max");
        impl.assert(idx <= self.n_rs_seq, "llama-memory-recurrent.cpp", 415, "idx <= n_rs_seq");
        self.rs_idx.items[@intCast(seq_id)] = idx;
    }

    /// Ports `llama_memory_recurrent`'s `memory_breakdown` (llama-memory-recurrent.cpp:420 @b29c606e2).
    pub fn memoryBreakdown(self: *const MemoryRecurrent, out: *std.ArrayList(memory.BuftSize)) void {
        for (self.ctxs_bufs.items) |e| memory.addBuftSize(out, c.ggml_backend_buffer_get_type(e.buf), c.ggml_backend_buffer_get_size(e.buf));
    }

    /// Ports `llama_memory_recurrent`'s `init_batch` (llama-memory-recurrent.cpp:428 @b29c606e2).
    pub fn initBatch(self: *MemoryRecurrent, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *RecurrentContext {
        blk: {
            balloc.splitReset();
            var ubatches: std.ArrayList(Ubatch) = .empty;
            while (true) {
                // All outputs: split by sequence. Otherwise a sequential
                // equal split, keeping each sequence's last 1 + n_rs_seq
                // tokens together so the rollback snapshots stay valid.
                const ub = if (embd_all)
                    balloc.splitSeq(n_ubatch)
                else
                    balloc.splitEqual(n_ubatch, true, if (self.n_rs_seq > 0) self.n_rs_seq + 1 else 0);
                if (ub.n_tokens == 0) break;
                ubatches.append(allocator, ub) catch oom();
            }
            if (balloc.getNUsed() < balloc.getNTokens() or !self.prepare(ubatches.items)) {
                for (ubatches.items) |*u| u.release();
                ubatches.deinit(allocator);
                break :blk;
            }
            return RecurrentContext.createBatch(self, ubatches);
        }
        return RecurrentContext.createStatus(.failed_prepare);
    }

    /// Ports `llama_memory_recurrent`'s `init_full` (llama-memory-recurrent.cpp:470 @b29c606e2).
    pub fn initFull(self: *MemoryRecurrent) *RecurrentContext {
        return RecurrentContext.createFull(self);
    }

    /// Ports `llama_memory_recurrent`'s `init_update` (llama-memory-recurrent.cpp:474
    /// @b29c606e2): a recurrent memory never updates.
    pub fn initUpdate(self: *MemoryRecurrent, lctx: *anyopaque, optimize: bool) *RecurrentContext {
        _ = self;
        _ = lctx;
        _ = optimize;
        return RecurrentContext.createStatus(.no_update);
    }

    /// Ports `llama_memory_recurrent`'s `prepare` (llama-memory-recurrent.cpp:481
    /// @b29c606e2): try every ubatch on a copy of the state, then restore it.
    pub fn prepare(self: *MemoryRecurrent, ubatches: []const Ubatch) bool {
        const org_cells = allocator.dupe(MemCell, self.cells.items) catch oom();
        defer allocator.free(org_cells);
        const org_used = self.used;
        const org_head = self.head;
        var success = true;
        for (ubatches) |*ub| {
            if (!self.findSlot(ub)) {
                success = false;
                break;
            }
        }
        @memcpy(self.cells.items, org_cells);
        self.used = org_used;
        self.head = org_head;
        return success;
    }

    /// Ports `llama_memory_recurrent`'s `find_slot` (llama-memory-recurrent.cpp:505
    /// @b29c606e2): every sequence gets a cell of its own, contiguous for
    /// the ubatch, and each cell records where its state comes from.
    pub fn findSlot(self: *MemoryRecurrent, ubatch: *const Ubatch) bool {
        const n_seq_tokens = ubatch.n_seq_tokens;
        const n_seqs = ubatch.n_seqs;
        const size = self.size;
        const cells = self.cells.items;

        // With enough unused cells before the head, search from the start.
        if (self.head > self.used + 2 * n_seqs) self.head = 0;

        // Each cell stores a whole sequence's state; a slot is always
        // contiguous, and only equal-length sequences can share a ubatch.
        impl.assert(ubatch.equalSeqs(), "llama-memory-recurrent.cpp", 520, "ubatch.equal_seqs()");

        var min: i32 = @as(i32, @intCast(size)) - 1;
        var max: i32 = 0;

        // Everything fits if every seq_id is below the size.
        for (0..n_seqs) |s| {
            const i = s * n_seq_tokens;
            const n_seq_id: u32 = @intCast(ubatch.n_seq_id[i]);
            for (0..n_seq_id) |j| {
                const seq_id = ubatch.seq_id[i][j];
                if (seq_id < 0 or @as(u32, @bitCast(seq_id)) >= size) {
                    impl.logError("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", .{ "find_slot", seq_id, self.n_seq_max });
                    return false;
                }
                if (j > 0) {
                    const seq = &cells[@intCast(seq_id)];
                    if (seq.tail >= 0) {
                        // Clear cells of sequences that become shared.
                        const cell = &cells[@intCast(seq.tail)];
                        cell.seq_id.unset(@intCast(seq_id));
                        seq.tail = -1;
                        if (cell.isEmpty()) {
                            cell.pos = -1;
                            cell.src = -1;
                            self.used -%= 1;
                        }
                    }
                }
            }
        }

        // The next empty cell.
        var next_empty_cell: u32 = self.head;
        for (0..size) |_| {
            if (next_empty_cell >= size) next_empty_cell -= size;
            if (cells[next_empty_cell].isEmpty()) break;
            next_empty_cell += 1;
        }

        // The usable cell range.
        for (0..n_seqs) |s| {
            const i = s * n_seq_tokens;
            const seq_id = ubatch.seq_id[i][0];
            const seq_meta = &cells[@intCast(seq_id)];
            var has_cell = false;
            if (seq_meta.tail >= 0) {
                const cell = &cells[@intCast(seq_meta.tail)];
                impl.assert(cell.hasSeqId(seq_id), "llama-memory-recurrent.cpp", 596, "cell.has_seq_id(seq_id)");
                // Does this sequence own the cell?
                if (cell.seq_id.count() == 1) has_cell = true;
            }
            if (!has_cell) {
                const empty_cell = &cells[next_empty_cell];
                impl.assert(empty_cell.isEmpty(), "llama-memory-recurrent.cpp", 602, "empty_cell.is_empty()");
                // Copy the old tail into the empty cell.
                if (seq_meta.tail >= 0) {
                    const orig_cell = &cells[@intCast(seq_meta.tail)];
                    empty_cell.pos = orig_cell.pos;
                    empty_cell.src = orig_cell.src;
                    orig_cell.seq_id.unset(@intCast(seq_id));
                    empty_cell.seq_id.set(@intCast(seq_id)); // overwritten below
                    impl.assert(!orig_cell.isEmpty(), "llama-memory-recurrent.cpp", 610, "!orig_cell.is_empty()");
                }
                seq_meta.tail = @intCast(next_empty_cell);
                // The next empty cell.
                if (s + 1 < n_seqs) {
                    for (0..size) |_| {
                        next_empty_cell += 1;
                        if (next_empty_cell >= size) next_empty_cell -= size;
                        if (cells[next_empty_cell].isEmpty()) break;
                    }
                }
            }
            if (min > seq_meta.tail) min = seq_meta.tail;
            if (max < seq_meta.tail) max = seq_meta.tail;
        }

        // Gather and reorder.
        for (0..n_seqs) |s| {
            const i = s * n_seq_tokens;
            const dst_id: i32 = @as(i32, @intCast(s)) + min;
            const src_id = cells[@intCast(ubatch.seq_id[i][0])].tail;
            if (dst_id != src_id) {
                const dst_cell = &cells[@intCast(dst_id)];
                const src_cell = &cells[@intCast(src_id)];
                std.mem.swap(i32, &dst_cell.pos, &src_cell.pos);
                std.mem.swap(i32, &dst_cell.src, &src_cell.src);
                std.mem.swap(SeqSet, &dst_cell.seq_id, &src_cell.seq_id);
                // Swap the tails.
                for (cells[0..size]) |*cl| {
                    if (cl.tail == src_id) {
                        cl.tail = dst_id;
                    } else if (cl.tail == dst_id) {
                        cl.tail = src_id;
                    }
                }
            }
        }

        // Update the used sequences' positions.
        for (0..n_seqs) |s| {
            const i = s * n_seq_tokens;
            const last_pos = ubatch.pos[i + n_seq_tokens - 1];
            const cell_id: i32 = @as(i32, @intCast(s)) + min;
            const cell = &cells[@intCast(cell_id)];
            if (cell.pos >= 0 and last_pos != cell.pos + @as(i32, @intCast(n_seq_tokens))) {
                // A position that backtracks or skips cannot clear the
                // state mid-batch.
                impl.logWarn("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n", .{ "find_slot", last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens });
            }
            cell.pos = last_pos;
            cell.seq_id = SeqSet.empty;
            for (0..@intCast(ubatch.n_seq_id[i])) |j| {
                const seq_id = ubatch.seq_id[i][j];
                cell.seq_id.set(@intCast(seq_id));
                cells[@intCast(seq_id)].tail = cell_id;
            }
        }

        // The first cell no source refers to, used as the zeroed state.
        {
            const refcounts = allocator.alloc(i32, size) catch oom();
            defer allocator.free(refcounts);
            @memset(refcounts, 0);
            for (cells[0..size]) |*cl| {
                if (cl.src >= 0) refcounts[@intCast(cl.src)] += 1;
            }
            self.rs_z = -1;
            var i: i32 = min;
            while (i <= max) : (i += 1) {
                if (refcounts[@intCast(i)] == 0) {
                    self.rs_z = i;
                    break;
                }
            }
            i = min;
            while (i <= max) : (i += 1) {
                const cl = &cells[@intCast(i)];
                if (cl.src < 0) {
                    impl.assert(self.rs_z >= 0, "llama-memory-recurrent.cpp", 695, "rs_z >= 0");
                    cl.src0 = self.rs_z;
                } else {
                    // Stage the sources, so seq_* still behave and the
                    // inputs can still read them.
                    cl.src0 = cl.src;
                }
                // Avoid moving or clearing twice.
                cl.src = i;
            }
        }

        // The used range is head to head + n.
        self.head = @intCast(min);
        self.n = @intCast(max - min + 1);
        var used: u32 = 0;
        for (cells) |*cl| {
            if (!cl.isEmpty()) used += 1;
        }
        self.used = used;

        // A sanity check.
        return self.n >= n_seqs;
    }

    /// Ports `llama_memory_recurrent`'s `get_can_shift` (llama-memory-recurrent.cpp:716
    /// @b29c606e2): shifting a position is trivial here.
    pub fn getCanShift(_: *const MemoryRecurrent) bool {
        return true;
    }

    /// Ports `llama_memory_recurrent`'s `total_size` (llama-memory-recurrent.cpp:721 @b29c606e2).
    pub fn totalSize(self: *const MemoryRecurrent) usize {
        var size: usize = 0;
        for (self.ctxs_bufs.items) |e| size += c.ggml_backend_buffer_get_size(e.buf);
        return size;
    }

    /// Ports `llama_memory_recurrent`'s `size_r_bytes` (llama-memory-recurrent.cpp:730 @b29c606e2).
    fn sizeRBytes(self: *const MemoryRecurrent) usize {
        var size: usize = 0;
        for (self.r_l.items) |r| if (r) |t| {
            size += c.ggml_nbytes(t);
        };
        return size;
    }

    /// Ports `llama_memory_recurrent`'s `size_s_bytes` (llama-memory-recurrent.cpp:742 @b29c606e2).
    fn sizeSBytes(self: *const MemoryRecurrent) usize {
        var size: usize = 0;
        for (self.s_l.items) |s| if (s) |t| {
            size += c.ggml_nbytes(t);
        };
        return size;
    }

    /// Ports `llama_memory_recurrent`'s `size_p_bytes` (llama-memory-recurrent.cpp:754 @b29c606e2).
    fn sizePBytes(self: *const MemoryRecurrent) usize {
        var size: usize = 0;
        for (self.p_l.items) |p| if (p) |t| {
            size += c.ggml_nbytes(t);
        };
        return size;
    }

    /// Ports `llama_memory_recurrent`'s `state_write` (llama-memory-recurrent.cpp:766 @b29c606e2).
    pub fn stateWrite(self: *const MemoryRecurrent, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        var cell_ranges: std.ArrayList([2]u32) = .empty;
        defer cell_ranges.deinit(allocator);
        // The logical source row ranges.
        var cell_ranges_data: std.ArrayList([2]u32) = .empty;
        defer cell_ranges_data.deinit(allocator);
        var cell_count: u32 = 0;
        const size = self.size;

        // The ranges of cells holding the sequence, or any when -1.
        var cell_range_begin: u32 = size;
        for (self.cells.items[0..size], 0..) |*cell, i_| {
            const i: u32 = @intCast(i_);
            if ((seq_id == -1 and !cell.isEmpty()) or cell.hasSeqId(seq_id)) {
                cell_count += 1;
                var rs_idx_cur: u32 = 0;
                if (self.n_rs_seq != 0) {
                    if (seq_id != -1) {
                        impl.assert(seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.rs_idx.items.len, "llama-memory-recurrent.cpp", 785, "seq_id >= 0 && (size_t) seq_id < rs_idx.size()");
                        rs_idx_cur = self.rs_idx.items[@intCast(seq_id)];
                    } else {
                        var has_rs_idx = false;
                        var it = cell.seq_id.iterator(.{});
                        while (it.next()) |cell_seq_id| {
                            impl.assert(cell_seq_id < self.rs_idx.items.len, "llama-memory-recurrent.cpp", 790, "cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size()");
                            const seq_rs_idx = self.rs_idx.items[cell_seq_id];
                            if (!has_rs_idx) {
                                rs_idx_cur = seq_rs_idx;
                                has_rs_idx = true;
                            } else if (rs_idx_cur != seq_rs_idx) {
                                impl.abort("llama-memory-recurrent.cpp", 797, "cannot write shared recurrent state with different rollback indices");
                            }
                        }
                    }
                }

                const cell_id: u32 = rs_idx_cur *% size +% @as(u32, @bitCast(if (cell.src >= 0) cell.src else @as(i32, @intCast(i))));
                const n_data = cell_ranges_data.items.len;
                if (n_data == 0 or cell_ranges_data.items[n_data - 1][1] != cell_id) {
                    cell_ranges_data.append(allocator, .{ cell_id, cell_id + 1 }) catch oom();
                } else {
                    cell_ranges_data.items[n_data - 1][1] += 1;
                }
                if (cell_range_begin == size) cell_range_begin = i;
            } else if (cell_range_begin != size) {
                cell_ranges.append(allocator, .{ cell_range_begin, i }) catch oom();
                cell_range_begin = size;
            }
        }
        if (cell_range_begin != size) cell_ranges.append(allocator, .{ cell_range_begin, size }) catch oom();

        if (flags & c.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0 and cell_ranges.items.len > 1) {
            impl.abort("llama-memory-recurrent.cpp", 825, "cannot save/load multiple ranges of cells to/from device memory\n");
        }

        var check: u32 = 0;
        for (cell_ranges.items) |r| check += r[1] - r[0];
        impl.assert(cell_count == check, "llama-memory-recurrent.cpp", 833, "cell_count == cell_count_check");
        check = 0;
        for (cell_ranges_data.items) |r| check += r[1] - r[0];
        impl.assert(cell_count == check, "llama-memory-recurrent.cpp", 839, "cell_count == cell_count_check");

        try w.writeValue(cell_count, exc);
        try self.stateWriteMeta(w, cell_ranges.items, seq_id, exc);
        try self.stateWriteData(w, cell_ranges_data.items, exc);
    }

    /// Ports `llama_memory_recurrent`'s `state_read` (llama-memory-recurrent.cpp:847 @b29c606e2).
    pub fn stateRead(self: *MemoryRecurrent, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        _ = flags;
        const cell_count = try r.readValue(u32, exc);
        var res = try self.stateReadMeta(r, cell_count, seq_id, exc);
        if (res) {
            // The C++ catches any exception from reading the data.
            var inner: Exception = .{};
            res = self.stateReadData(r, cell_count, &inner) catch false;
        }
        if (!res) {
            if (seq_id == -1) self.clear(true) else _ = self.seqRm(seq_id, -1, -1);
            return exc.throw(.runtime_error, "failed to restore kv cache", .{});
        }
        if (self.n_rs_seq != 0) self.setRsIdx(seq_id, 0);
    }

    /// Ports `llama_memory_recurrent`'s `state_write_meta` (llama-memory-recurrent.cpp:878 @b29c606e2).
    fn stateWriteMeta(self: *const MemoryRecurrent, w: io.Writer, cell_ranges: []const [2]u32, seq_id: i32, exc: *Exception) Thrown!void {
        for (cell_ranges) |range| {
            var i = range[0];
            while (i < range[1]) : (i += 1) {
                const cell = &self.cells.items[i];
                const n_seq_id: u32 = if (seq_id == -1) @intCast(cell.seq_id.count()) else 0;
                try w.writeValue(cell.pos, exc);
                try w.writeValue(n_seq_id, exc);
                if (n_seq_id != 0) {
                    var it = cell.seq_id.iterator(.{});
                    while (it.next()) |sid| try w.writeValue(@as(i32, @intCast(sid)), exc);
                }
            }
        }
    }

    /// Ports `llama_memory_recurrent`'s `state_write_data` (llama-memory-recurrent.cpp:897
    /// @b29c606e2). The S tensor is never transposed, so that arm of the
    /// C++ is unreachable and not here. A layer's PLE conv history, a
    /// second recurrent row, travels right after its R rows.
    fn stateWriteData(self: *const MemoryRecurrent, w: io.Writer, cell_ranges: []const [2]u32, exc: *Exception) Thrown!void {
        const s_trans: u32 = 0;
        const n_layer: u32 = self.hparams.nLayer();
        try w.writeValue(s_trans, exc);
        try w.writeValue(n_layer, exc);

        // All the R tensors first, a row per cell.
        for (self.r_l.items[0..n_layer], self.p_l.items[0..n_layer]) |rl, pl| {
            const r = rl orelse continue;
            const r_type_i: i32 = @intCast(r.type);
            try w.writeValue(r_type_i, exc);
            const r_size_row: u64 = c.ggml_row_size(r.type, self.hparams.nEmbdR());
            try w.writeValue(r_size_row, exc);
            for (cell_ranges) |range| {
                const range_size: usize = range[1] - range[0];
                try w.writeTensor(r, range[0] * r_size_row, range_size * r_size_row, exc);
            }

            if (pl) |p| {
                const p_size_row: u64 = c.ggml_row_size(p.type, self.hparams.pleConvState());
                try w.writeValue(p_size_row, exc);
                for (cell_ranges) |range| {
                    const range_size: usize = range[1] - range[0];
                    try w.writeTensor(p, range[0] * p_size_row, range_size * p_size_row, exc);
                }
            }
        }
        for (self.s_l.items[0..n_layer]) |sl| {
            const s = sl orelse continue;
            const s_type_i: i32 = @intCast(s.type);
            try w.writeValue(s_type_i, exc);
            const s_size_row: u64 = c.ggml_row_size(s.type, self.hparams.nEmbdS());
            try w.writeValue(s_size_row, exc);
            for (cell_ranges) |range| {
                const range_size: usize = range[1] - range[0];
                try w.writeTensor(s, range[0] * s_size_row, range_size * s_size_row, exc);
            }
        }
    }

    /// Ports `llama_memory_recurrent`'s `state_read_meta` (llama-memory-recurrent.cpp:992 @b29c606e2).
    fn stateReadMeta(self: *MemoryRecurrent, r: io.Reader, cell_count: u32, dest_seq_id_: i32, exc: *Exception) Thrown!bool {
        const func = "state_read_meta";
        var dest_seq_id = dest_seq_id_;
        if (dest_seq_id != -1) {
            // A single sequence.
            _ = self.seqRm(dest_seq_id, -1, -1);
            if (cell_count == 0) return true;

            var balloc = batch.BatchAllocr.init(self.hparams.nPosPerEmbd());
            defer balloc.deinit();
            var ubatch = balloc.ubatchReserve(cell_count, 1);
            defer ubatch.release();

            for (0..cell_count) |i| {
                const pos = try r.readValue(i32, exc);
                const n_seq_id = try r.readValue(u32, exc);
                if (n_seq_id != 0) {
                    impl.logError("%s: invalid seq_id-agnostic kv cell\n", .{func});
                    return false;
                }
                ubatch.pos[i] = pos;
            }
            ubatch.n_seq_id[0] = 1;
            ubatch.seq_id[0] = @ptrCast(&dest_seq_id);

            if (!self.findSlot(&ubatch)) {
                impl.logError("%s: failed to find available cells in kv cache\n", .{func});
                return false;
            }

            // One contiguous block from head, with the right positions.
            impl.assert(self.head + cell_count <= self.size, "llama-memory-recurrent.cpp", 1029, "head + cell_count <= size");
            impl.assert(self.cells.items[self.head].pos == ubatch.pos[0], "llama-memory-recurrent.cpp", 1030, "cells[head].pos == ubatch.pos[0]");
            impl.assert(self.cells.items[self.head + cell_count - 1].pos == ubatch.pos[cell_count - 1], "llama-memory-recurrent.cpp", 1031, "cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]");
            impl.assert(self.cells.items[self.head].hasSeqId(dest_seq_id), "llama-memory-recurrent.cpp", 1032, "cells[head].has_seq_id(dest_seq_id)");
            impl.assert(self.cells.items[self.head + cell_count - 1].hasSeqId(dest_seq_id), "llama-memory-recurrent.cpp", 1033, "cells[head + cell_count - 1].has_seq_id(dest_seq_id)");
        } else {
            // The whole cache.
            if (cell_count > self.size) {
                impl.logError("%s: not enough cells in kv cache\n", .{func});
                return false;
            }
            self.clear(true);
            for (0..cell_count) |i| {
                const cell = &self.cells.items[i];
                const pos = try r.readValue(i32, exc);
                const n_seq_id = try r.readValue(u32, exc);
                cell.pos = pos;
                for (0..n_seq_id) |_| {
                    const seq_id = try r.readValue(i32, exc);
                    if (seq_id < 0 or @as(u32, @bitCast(seq_id)) >= self.n_seq_max) {
                        impl.logError("%s: invalid seq_id, %d is out of range [0, %u)\n", .{ func, seq_id, self.n_seq_max });
                        return false;
                    }
                    cell.seq_id.set(@intCast(seq_id));
                    const tail = &self.cells.items[@intCast(seq_id)].tail;
                    if (tail.* != -1) {
                        impl.logError("%s: duplicate tail for seq_id %d in cell %d and %d\n", .{ func, seq_id, @as(c_int, @intCast(i)), tail.* });
                        return false;
                    }
                    tail.* = @intCast(i);
                }
            }
            self.head = 0;
            self.used = cell_count;
        }

        for (0..cell_count) |i| {
            const cell_id = self.head + @as(u32, @intCast(i));
            // The restored states keep themselves.
            self.cells.items[cell_id].src = @intCast(cell_id);
        }
        return true;
    }

    /// Ports `llama_memory_recurrent`'s `state_read_data` (llama-memory-recurrent.cpp:1088 @b29c606e2).
    fn stateReadData(self: *MemoryRecurrent, r: io.Reader, cell_count: u32, exc: *Exception) Thrown!bool {
        const func = "state_read_data";
        const s_trans = try r.readValue(u32, exc);
        const n_layer = try r.readValue(u32, exc);

        if (n_layer != self.hparams.nLayer()) {
            impl.logError("%s: mismatched layer count (%u instead of %u)\n", .{ func, n_layer, self.hparams.nLayer() });
            return false;
        }
        if (cell_count > self.size) {
            impl.logError("%s: not enough cells in kv cache to restore state (%u > %u)\n", .{ func, cell_count, self.size });
            return false;
        }
        if (s_trans != 0) {
            impl.logError("%s: incompatible s transposition\n", .{func});
            return false;
        }

        // Each layer's R rows, one contiguous block.
        for (self.r_l.items[0..n_layer], self.p_l.items[0..n_layer], 0..) |rl, pl, il| {
            const t = rl orelse continue;
            const r_type_i_ref = try r.readValue(i32, exc);
            const r_type_i: i32 = @intCast(t.type);
            if (r_type_i != r_type_i_ref) {
                impl.logError("%s: mismatched r type (%d != %d, layer %d)\n", .{ func, r_type_i, r_type_i_ref, @as(c_int, @intCast(il)) });
                return false;
            }
            const r_size_row_ref = try r.readValue(u64, exc);
            const r_size_row = c.ggml_row_size(t.type, self.hparams.nEmbdR());
            if (r_size_row != r_size_row_ref) {
                impl.logError("%s: mismatched r row size (%zu != %zu, layer %d)\n", .{ func, r_size_row, @as(usize, @intCast(r_size_row_ref)), @as(c_int, @intCast(il)) });
                return false;
            }
            if (cell_count != 0) try r.readTensor(t, self.head * r_size_row, cell_count * r_size_row, exc);

            if (pl) |p| {
                const p_size_row_ref = try r.readValue(u64, exc);
                const p_size_row = c.ggml_row_size(p.type, self.hparams.pleConvState());
                if (p_size_row != p_size_row_ref) {
                    impl.logError("%s: mismatched ple row size (%zu != %zu, layer %d)\n", .{ func, p_size_row, @as(usize, @intCast(p_size_row_ref)), @as(c_int, @intCast(il)) });
                    return false;
                }
                if (cell_count != 0) try r.readTensor(p, self.head * p_size_row, cell_count * p_size_row, exc);
            }
        }

        for (self.s_l.items[0..n_layer], 0..) |sl, il| {
            const t = sl orelse continue;
            const s_type_i_ref = try r.readValue(i32, exc);
            const s_type_i: i32 = @intCast(t.type);
            if (s_type_i != s_type_i_ref) {
                impl.logError("%s: mismatched s type (%d != %d, layer %d)\n", .{ func, s_type_i, s_type_i_ref, @as(c_int, @intCast(il)) });
                return false;
            }
            const s_size_row_ref = try r.readValue(u64, exc);
            const s_size_row = c.ggml_row_size(t.type, self.hparams.nEmbdS());
            if (s_size_row != s_size_row_ref) {
                impl.logError("%s: mismatched s row size (%zu != %zu, layer %d)\n", .{ func, s_size_row, @as(usize, @intCast(s_size_row_ref)), @as(c_int, @intCast(il)) });
                return false;
            }
            if (cell_count != 0) try r.readTensor(t, self.head * s_size_row, cell_count * s_size_row, exc);
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *MemoryRecurrent) memory.Memory {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: memory.Memory.VTable = .{
        .initBatch = struct {
            fn f(p: *anyopaque, b: *batch.BatchAllocr, n: u32, e: bool) memory.Context {
                return from(p).initBatch(b, n, e).contextOf();
            }
        }.f,
        .initFull = struct {
            fn f(p: *anyopaque) memory.Context {
                return from(p).initFull().contextOf();
            }
        }.f,
        .initUpdate = struct {
            fn f(p: *anyopaque, l: *anyopaque, o: bool) memory.Context {
                return from(p).initUpdate(l, o).contextOf();
            }
        }.f,
        .getCanShift = struct {
            fn f(p: *anyopaque) bool {
                return from(p).getCanShift();
            }
        }.f,
        .clear = struct {
            fn f(p: *anyopaque, d: bool) void {
                from(p).clear(d);
            }
        }.f,
        .seqRm = struct {
            fn f(p: *anyopaque, s: i32, p0: i32, p1: i32) bool {
                return from(p).seqRm(s, p0, p1);
            }
        }.f,
        .seqCp = struct {
            fn f(p: *anyopaque, a: i32, b: i32, p0: i32, p1: i32) void {
                from(p).seqCp(a, b, p0, p1);
            }
        }.f,
        .seqKeep = struct {
            fn f(p: *anyopaque, s: i32) void {
                from(p).seqKeep(s);
            }
        }.f,
        .seqAdd = struct {
            fn f(p: *anyopaque, s: i32, p0: i32, p1: i32, sh: i32) void {
                from(p).seqAdd(s, p0, p1, sh);
            }
        }.f,
        .seqDiv = struct {
            fn f(p: *anyopaque, s: i32, p0: i32, p1: i32, d: c_int) void {
                from(p).seqDiv(s, p0, p1, d);
            }
        }.f,
        .seqPosMin = struct {
            fn f(p: *anyopaque, s: i32) i32 {
                return from(p).seqPosMin(s);
            }
        }.f,
        .seqPosMax = struct {
            fn f(p: *anyopaque, s: i32) i32 {
                return from(p).seqPosMax(s);
            }
        }.f,
        .memoryBreakdown = struct {
            fn f(p: *anyopaque, out: *std.ArrayList(memory.BuftSize)) void {
                from(p).memoryBreakdown(out);
            }
        }.f,
        .stateWrite = struct {
            fn f(p: *anyopaque, w: io.Writer, s: i32, fl: u32, e: *Exception) Thrown!void {
                return from(p).stateWrite(w, s, fl, e);
            }
        }.f,
        .stateRead = struct {
            fn f(p: *anyopaque, r: io.Reader, s: i32, fl: u32, e: *Exception) Thrown!void {
                return from(p).stateRead(r, s, fl, e);
            }
        }.f,
        .destroy = struct {
            fn f(p: *anyopaque) void {
                from(p).destroy();
            }
        }.f,
    };

    fn from(p: *anyopaque) *MemoryRecurrent {
        return @ptrCast(@alignCast(p));
    }
};

/// Ports `llama_memory_recurrent_context` (llama-memory-recurrent.h:139 @b29c606e2).
pub const RecurrentContext = struct {
    status: memory.Status,
    mem: ?*MemoryRecurrent = null,
    i_next: usize = 0,
    ubatches: std.ArrayList(Ubatch) = .empty,
    is_full: bool = false,

    /// Ports the status constructor of `llama_memory_recurrent_context` (llama-memory-recurrent.cpp:1230 @b29c606e2).
    pub fn createStatus(status: memory.Status) *RecurrentContext {
        const self = allocator.create(RecurrentContext) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-memory constructor of `llama_memory_recurrent_context` (llama-memory-recurrent.cpp:1232 @b29c606e2).
    pub fn createFull(mem: *MemoryRecurrent) *RecurrentContext {
        const self = allocator.create(RecurrentContext) catch oom();
        self.* = .{ .status = .success, .mem = mem, .is_full = true };
        return self;
    }

    /// Ports the batch constructor of `llama_memory_recurrent_context` (llama-memory-recurrent.cpp:1236 @b29c606e2).
    pub fn createBatch(mem: *MemoryRecurrent, ubatches: std.ArrayList(Ubatch)) *RecurrentContext {
        const self = allocator.create(RecurrentContext) catch oom();
        self.* = .{ .status = .success, .mem = mem, .ubatches = ubatches };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *RecurrentContext) void {
        for (self.ubatches.items) |*u| u.release();
        self.ubatches.deinit(allocator);
        allocator.destroy(self);
    }

    /// Ports `llama_memory_recurrent_context`'s `next` (llama-memory-recurrent.cpp:1242 @b29c606e2).
    pub fn next(self: *RecurrentContext) bool {
        std.debug.assert(self.status == .success);
        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_memory_recurrent_context`'s `apply` (llama-memory-recurrent.cpp:1252 @b29c606e2).
    pub fn apply(self: *RecurrentContext) bool {
        std.debug.assert(!memory.statusIsFail(self.status));
        // No ubatches: an update, which a recurrent memory never performs.
        if (self.ubatches.items.len == 0) {
            std.debug.assert(self.status == .no_update);
            return true;
        }
        _ = self.mem.?.findSlot(&self.ubatches.items[self.i_next]);
        return true;
    }

    /// Ports `llama_memory_recurrent_context`'s `get_status` (llama-memory-recurrent.cpp:1268 @b29c606e2).
    pub fn getStatus(self: *const RecurrentContext) memory.Status {
        return self.status;
    }

    /// Ports `llama_memory_recurrent_context`'s `get_ubatch` (llama-memory-recurrent.cpp:1272 @b29c606e2).
    pub fn getUbatch(self: *const RecurrentContext) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_memory_recurrent_context`'s `get_n_rs` (llama-memory-recurrent.cpp:1278 @b29c606e2).
    pub fn getNRs(self: *const RecurrentContext) u32 {
        return if (self.is_full) self.mem.?.size else self.mem.?.n;
    }

    /// Ports `llama_memory_recurrent_context`'s `get_head` (llama-memory-recurrent.cpp:1282 @b29c606e2).
    pub fn getHead(self: *const RecurrentContext) u32 {
        return if (self.is_full) 0 else self.mem.?.head;
    }

    /// Ports `llama_memory_recurrent_context`'s `get_rs_z` (llama-memory-recurrent.cpp:1286 @b29c606e2).
    pub fn getRsZ(self: *const RecurrentContext) i32 {
        return if (self.is_full) 0 else self.mem.?.rs_z;
    }

    /// Ports `llama_memory_recurrent_context`'s `get_size` (llama-memory-recurrent.cpp:1290 @b29c606e2).
    pub fn getSize(self: *const RecurrentContext) u32 {
        return self.mem.?.size;
    }

    /// Ports `llama_memory_recurrent_context`'s `get_r_l` (llama-memory-recurrent.cpp:1294 @b29c606e2).
    pub fn getRL(self: *const RecurrentContext, il: i32) *Tensor {
        return self.mem.?.r_l.items[@intCast(il)].?;
    }

    /// Ports `llama_memory_recurrent_context`'s `get_s_l` (llama-memory-recurrent.cpp:1298 @b29c606e2).
    pub fn getSL(self: *const RecurrentContext, il: i32) *Tensor {
        return self.mem.?.s_l.items[@intCast(il)].?;
    }

    /// Ports `llama_memory_recurrent_context`'s `get_p_l` (llama-memory-recurrent.cpp:1302
    /// @b29c606e2): the layer's PLE conv history row, null where it has none.
    pub fn getPL(self: *const RecurrentContext, il: i32) ?*Tensor {
        return self.mem.?.p_l.items[@intCast(il)];
    }

    /// Ports `llama_memory_recurrent_context`'s `s_copy` (llama-memory-recurrent.cpp:1306
    /// @b29c606e2): where cell `i` of the range copies its state from. A
    /// pending rollback is consumed here, so the C++'s `const` method writes.
    pub fn sCopy(self: *const RecurrentContext, i: c_int) i32 {
        const mem = self.mem.?;
        const cell_idx: u32 = @as(u32, @intCast(i)) + mem.head;
        const src0 = mem.cells.items[cell_idx].src0;
        if (mem.n_rs_seq == 0) return src0;
        var idx: u32 = 0;
        if (mem.cells.items[cell_idx].seq_id.findFirstSet()) |seq| {
            if (seq < mem.rs_idx.items.len) {
                idx = mem.rs_idx.items[seq];
                // Reset the rollback index.
                mem.rs_idx.items[seq] = 0;
            }
        }
        return @as(i32, @bitCast(idx *% mem.size)) +% src0;
    }

    pub fn contextOf(self: *RecurrentContext) memory.Context {
        return .{ .ptr = self, .vtable = &ctx_vtable };
    }

    const ctx_vtable: memory.Context.VTable = .{
        .next = struct {
            fn f(p: *anyopaque) bool {
                return cast(p).next();
            }
        }.f,
        .apply = struct {
            fn f(p: *anyopaque) bool {
                return cast(p).apply();
            }
        }.f,
        .getUbatch = struct {
            fn f(p: *anyopaque) *const Ubatch {
                return cast(p).getUbatch();
            }
        }.f,
        .getStatus = struct {
            fn f(p: *anyopaque) memory.Status {
                return cast(p).getStatus();
            }
        }.f,
        .destroy = struct {
            fn f(p: *anyopaque) void {
                cast(p).destroy();
            }
        }.f,
    };

    fn cast(p: *anyopaque) *RecurrentContext {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(MemCell);
    std.testing.refAllDecls(MemoryRecurrent);
    std.testing.refAllDecls(RecurrentContext);
}
