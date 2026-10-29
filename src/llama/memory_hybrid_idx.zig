//! A hybrid memory with a third cache: one indexer key per token, for
//! qwen4exp's block-sparse attention (QSA).
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-memory-hybrid-idx.h` -- `llama_memory_hybrid_idx`
//!   and `llama_memory_hybrid_idx_context`
//! - `llama.cpp/src/llama-memory-hybrid-idx.cpp` -- their bodies and
//!   `llama_memory_hybrid_idx_ns`
//!
//! All at v0.4.1 (`b29c606e2`).
//!
//! # A side buffer over the attention cells
//!
//! The indexer cache has the attention cache's size, padding, streams and
//! slots, so cell `j` is one token in both. It never searches for cells of
//! its own: a batch takes the attention cache's slot infos, and a restore
//! takes the cells the attention restore chose (`[TAG_HYBRID_IDX_SINFO]`).
//!
//! # Inheritance
//!
//! The C++ derives both classes from `llama_memory_hybrid` and
//! `llama_memory_hybrid_context`, and the graph `static_cast`s the context
//! to its base. Here the memory holds the hybrid memory it extends, and the
//! context embeds a `HybridContext` as `base`: the `memory.Context` it hands
//! out points at `base`, so the graph's `cast(HybridContext)` works
//! unchanged, and `fromBase` recovers the extension, the C++'s downcast.

const std = @import("std");
const impl = @import("impl.zig");
const hparams_mod = @import("hparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const io = @import("io.zig");
const kv_cache = @import("kv_cache.zig");
const kv_cells = @import("kv_cells.zig");
const memory_hybrid = @import("memory_hybrid.zig");
const memory_recurrent = @import("memory_recurrent.zig");
const model_mod = @import("model.zig");
const libcxx = @import("libcxx.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;
const Ubatch = batch.Ubatch;
const Tensor = c.ggml_tensor;
const KvCache = kv_cache.KvCache;
const KvCells = kv_cells.KvCells;
const SlotInfo = kv_cache.SlotInfo;
const MemoryHybrid = memory_hybrid.MemoryHybrid;
const HybridContext = memory_hybrid.HybridContext;

/// `LLAMA_MAX_SEQ` (llama-cparams.h:8 @b29c606e2).
const max_seq = kv_cells.max_seq;

fn oom() noreturn {
    impl.abort("llama-memory-hybrid-idx.cpp", 0, "out of memory");
}

fn deinitSinfos(list: *std.ArrayList(SlotInfo)) void {
    for (list.items) |*si| si.deinit();
    list.deinit(allocator);
}

fn cloneSinfos(list: []const SlotInfo) std.ArrayList(SlotInfo) {
    var out: std.ArrayList(SlotInfo) = .empty;
    out.ensureTotalCapacity(allocator, list.len) catch oom();
    for (list) |*si| out.appendAssumeCapacity(si.clone());
    return out;
}

/// Ports `llama_memory_hybrid_idx` (llama-memory-hybrid-idx.h:15 @b29c606e2).
pub const MemoryHybridIdx = struct {
    /// The `llama_memory_hybrid` this extends.
    base: *MemoryHybrid,
    /// The indexer cache holds one key head per layer, so it needs its own
    /// hparams, and the cache keeps a pointer to what it is given: this
    /// struct is heap-allocated and does not move.
    hparams_idx: Hparams,
    /// Null when the model carries no indexer.
    mem_idx: ?*KvCache = null,

    /// Ports the constructor of `llama_memory_hybrid_idx`
    /// (llama-memory-hybrid-idx.cpp:19 @b29c606e2): the hybrid memory, and
    /// the indexer cache when `filter_idx` is given (`func != null`). The
    /// indexer is MQA with one key head of `indexer_head_size`, as
    /// `llama_kv_cache_dsa` shapes its own, and has no RoPE: its keys are
    /// cached raw and rotated after pooling, so a K-shift must not rotate
    /// them, and it is made to look like an MLA cache so no V is stored.
    ///
    /// Return: the memory, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        type_k: c.enum_ggml_type,
        type_v: c.enum_ggml_type,
        v_trans: bool,
        kv_size: u32,
        n_pad: u32,
        n_swa: u32,
        swa_type: hparams_mod.SwaType,
        type_r: c.enum_ggml_type,
        type_s: c.enum_ggml_type,
        rs_size: u32,
        n_seq_max: u32,
        n_rs_seq: u32,
        offload: bool,
        unified: bool,
        filter_attn: memory.LayerFilter,
        filter_recr: memory.LayerFilter,
        filter_idx: memory.LayerFilter,
        exc: *Exception,
    ) Thrown!*MemoryHybridIdx {
        const base = try MemoryHybrid.create(model, type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type, type_r, type_s, rs_size, n_seq_max, n_rs_seq, offload, unified, filter_attn, filter_recr, exc);
        errdefer base.destroy();

        const self = allocator.create(MemoryHybridIdx) catch oom();
        errdefer allocator.destroy(self);
        self.* = .{ .base = base, .hparams_idx = model.hparams };

        if (filter_idx.func != null) {
            @memset(&self.hparams_idx.n_head_kv_arr, 1);
            self.hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;
            self.hparams_idx.rope_type = c.LLAMA_ROPE_TYPE_NONE;
            // MLA-shaped, so the cache stores no V tensors.
            self.hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
            self.hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

            impl.logInfo("%s: creating indexer KV cache, size = %u cells\n", .{ "operator()", kv_size });

            self.mem_idx = try KvCache.createTagged(model, &self.hparams_idx, type_k, type_v, v_trans, offload, unified, kv_size, n_seq_max, n_pad, n_swa, swa_type, null, filter_idx, .{}, .{}, "idx_", exc);
        }
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *MemoryHybridIdx) void {
        if (self.mem_idx) |m| m.destroy();
        self.base.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_memory_hybrid_idx`'s `init_batch`
    /// (llama-memory-hybrid-idx.cpp:70 @b29c606e2): `llama_memory_hybrid`'s,
    /// repeated because the indexer needs the attention slot infos the base
    /// context hides.
    pub fn initBatch(self: *MemoryHybridIdx, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *HybridIdxContext {
        const func = "init_batch";
        blk: {
            balloc.splitReset();
            var ubatches: std.ArrayList(Ubatch) = .empty;
            while (true) {
                var ub: Ubatch = undefined;
                if (embd_all) {
                    // All outputs: split by sequence.
                    ub = balloc.splitSeq(n_ubatch);
                } else {
                    // A unified KV cache splits non-sequentially; each
                    // sequence's last 1 + n_rs_seq tokens stay together
                    // ([TAG_RECURRENT_ROLLBACK_SPLITS]).
                    const unified = self.getMemAttn().getNStream() == 1;
                    const n_rs_seq = self.getMemRecr().n_rs_seq;
                    ub = balloc.splitEqual(n_ubatch, !unified, if (n_rs_seq > 0) n_rs_seq + 1 else 0);
                }
                if (ub.n_tokens == 0) break;
                ubatches.append(allocator, ub) catch oom();
            }

            if (balloc.getNUsed() < balloc.getNTokens()) {
                // No suitable split.
                memory_hybrid.releaseAll(&ubatches);
                break :blk;
            }

            // The recurrent batches first.
            if (!self.getMemRecr().prepare(ubatches.items)) {
                impl.logError("%s: failed to prepare recurrent ubatches\n", .{func});
                memory_hybrid.releaseAll(&ubatches);
                return HybridIdxContext.createStatus(.failed_prepare);
            }

            // Then the attention cache.
            var heads_attn = self.getMemAttn().prepare(ubatches.items) orelse {
                impl.logError("%s: failed to prepare attention ubatches\n", .{func});
                memory_hybrid.releaseAll(&ubatches);
                return HybridIdxContext.createStatus(.failed_prepare);
            };
            if (heads_attn.items.len == 0) {
                heads_attn.deinit(allocator);
                impl.logError("%s: failed to prepare attention ubatches\n", .{func});
                memory_hybrid.releaseAll(&ubatches);
                return HybridIdxContext.createStatus(.failed_prepare);
            }

            // The indexer takes the attention cache's slot layout: one of
            // its own could drift from it.
            const heads_idx: std.ArrayList(SlotInfo) = if (self.mem_idx != null) cloneSinfos(heads_attn.items) else .empty;

            return HybridIdxContext.createBatch(self, heads_attn, heads_idx, ubatches);
        }
        return HybridIdxContext.createStatus(.failed_prepare);
    }

    /// Ports `llama_memory_hybrid_idx`'s `init_full` (llama-memory-hybrid-idx.cpp:135 @b29c606e2).
    pub fn initFull(self: *MemoryHybridIdx) *HybridIdxContext {
        return HybridIdxContext.createFull(self);
    }

    /// Ports `llama_memory_hybrid_idx`'s `init_update` (llama-memory-hybrid-idx.cpp:139 @b29c606e2).
    pub fn initUpdate(self: *MemoryHybridIdx, lctx: *anyopaque, optimize: bool) *HybridIdxContext {
        return HybridIdxContext.createUpdate(self, lctx, optimize);
    }

    /// `llama_memory_hybrid`'s `get_can_shift`, which this does not override.
    pub fn getCanShift(self: *const MemoryHybridIdx) bool {
        return self.base.getCanShift();
    }

    /// Ports `llama_memory_hybrid_idx`'s `clear` (llama-memory-hybrid-idx.cpp:143 @b29c606e2).
    pub fn clear(self: *MemoryHybridIdx, data: bool) void {
        self.base.clear(data);
        if (self.mem_idx) |m| m.clear(data);
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_rm` (llama-memory-hybrid-idx.cpp:151
    /// @b29c606e2): in `llama_memory_hybrid`'s order, the recurrent cache
    /// first, since it may refuse.
    pub fn seqRm(self: *MemoryHybridIdx, seq_id: i32, p0: i32, p1: i32) bool {
        if (!self.getMemRecr().seqRm(seq_id, p0, p1)) return false;
        if (self.mem_idx) |m| _ = m.seqRm(seq_id, p0, p1);
        return self.getMemAttn().seqRm(seq_id, p0, p1);
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_cp` (llama-memory-hybrid-idx.cpp:164 @b29c606e2).
    pub fn seqCp(self: *MemoryHybridIdx, a: i32, b: i32, p0: i32, p1: i32) void {
        self.base.seqCp(a, b, p0, p1);
        if (self.mem_idx) |m| m.seqCp(a, b, p0, p1);
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_keep` (llama-memory-hybrid-idx.cpp:172 @b29c606e2).
    pub fn seqKeep(self: *MemoryHybridIdx, seq_id: i32) void {
        self.base.seqKeep(seq_id);
        if (self.mem_idx) |m| m.seqKeep(seq_id);
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_add` (llama-memory-hybrid-idx.cpp:180 @b29c606e2).
    pub fn seqAdd(self: *MemoryHybridIdx, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.base.seqAdd(seq_id, p0, p1, shift);
        if (self.mem_idx) |m| m.seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_div` (llama-memory-hybrid-idx.cpp:188 @b29c606e2).
    pub fn seqDiv(self: *MemoryHybridIdx, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.base.seqDiv(seq_id, p0, p1, d);
        if (self.mem_idx) |m| m.seqDiv(seq_id, p0, p1, d);
    }

    /// `llama_memory_hybrid`'s `seq_pos_min`, which this does not override.
    pub fn seqPosMin(self: *const MemoryHybridIdx, seq_id: i32) i32 {
        return self.base.seqPosMin(seq_id);
    }

    /// `llama_memory_hybrid`'s `seq_pos_max`, which this does not override.
    pub fn seqPosMax(self: *const MemoryHybridIdx, seq_id: i32) i32 {
        return self.base.seqPosMax(seq_id);
    }

    /// Ports `llama_memory_hybrid_idx`'s `memory_breakdown`
    /// (llama-memory-hybrid-idx.cpp:196 @b29c606e2): the indexer's bytes
    /// added to the hybrid memory's, per buffer type.
    pub fn memoryBreakdown(self: *const MemoryHybridIdx, out: *std.ArrayList(memory.BuftSize)) void {
        self.base.memoryBreakdown(out);
        if (self.mem_idx) |m| m.memoryBreakdown(out);
    }

    /// Ports `llama_memory_hybrid_idx`'s `state_write`
    /// (llama-memory-hybrid-idx.cpp:208 @b29c606e2): the indexer section
    /// goes last, a pure suffix, under the attention cache's PARTIAL_ONLY
    /// gate ([TAG_HYBRID_IDX_STATE]).
    pub fn stateWrite(self: *const MemoryHybridIdx, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        try self.base.stateWrite(w, seq_id, flags, exc);
        if (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 0) {
            if (self.mem_idx) |m| try m.stateWrite(w, seq_id, flags, exc);
        }
    }

    /// Ports `llama_memory_hybrid_idx`'s `state_read`
    /// (llama-memory-hybrid-idx.cpp:221 @b29c606e2): `llama_memory_hybrid`'s,
    /// with the indexer restored into the cells the attention restore chose
    /// ([TAG_HYBRID_IDX_SINFO]). A failure drops what was being restored
    /// from all three caches, a state they agree on, and rethrows.
    pub fn stateRead(self: *MemoryHybridIdx, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        var sinfos_attn: std.ArrayList(SlotInfo) = .empty;
        defer deinitSinfos(&sinfos_attn);

        self.stateReadInner(r, seq_id, flags, &sinfos_attn, exc) catch |err| {
            self.stateDrop(seq_id);
            return err;
        };
    }

    fn stateReadInner(self: *MemoryHybridIdx, r: io.Reader, seq_id: i32, flags: u32, sinfos_attn: *std.ArrayList(SlotInfo), exc: *Exception) Thrown!void {
        const partial = flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY != 0;
        if (!partial) {
            try self.getMemAttn().stateReadSinfo(r, seq_id, flags, if (self.mem_idx != null) sinfos_attn else null, null, exc);
        }

        try self.getMemRecr().stateRead(r, seq_id, flags, exc);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order.
        if (!partial) {
            if (self.mem_idx) |m| try m.stateReadSinfo(r, seq_id, flags, null, sinfos_attn.items, exc);
        }
    }

    /// Ports `llama_memory_hybrid_idx`'s `state_drop`
    /// (llama-memory-hybrid-idx.cpp:253 @b29c606e2): forget `seq_id` in
    /// every cache at once, or the whole context when it is negative --
    /// directly, not through `seqRm`, which the recurrent cache may refuse
    /// and leave only the other two cleared.
    fn stateDrop(self: *MemoryHybridIdx, seq_id: i32) void {
        if (seq_id < 0) {
            self.clear(true);
            return;
        }
        _ = self.getMemAttn().seqRm(seq_id, -1, -1);
        _ = self.getMemRecr().seqRm(seq_id, -1, -1);
        if (self.mem_idx) |m| _ = m.seqRm(seq_id, -1, -1);
    }

    /// `llama_memory_hybrid`'s `get_mem_attn`.
    pub fn getMemAttn(self: *const MemoryHybridIdx) *KvCache {
        return self.base.getMemAttn();
    }

    /// `llama_memory_hybrid`'s `get_mem_recr`.
    pub fn getMemRecr(self: *const MemoryHybridIdx) *memory_recurrent.MemoryRecurrent {
        return self.base.getMemRecr();
    }

    /// Ports `llama_memory_hybrid_idx`'s `get_mem_idx` (llama-memory-hybrid-idx.cpp:269
    /// @b29c606e2): null when the model carries no indexer.
    pub fn getMemIdx(self: *const MemoryHybridIdx) ?*KvCache {
        return self.mem_idx;
    }

    /// Ports `llama_memory_hybrid_idx`'s `set_input_qsa`
    /// (llama-memory-hybrid-idx.cpp:273 @b29c606e2): block-compressed
    /// sparse attention over the indexer cache's cells. Blocks cut the
    /// position line, not the cell array, so no caller assumes a contiguous
    /// layout:
    ///
    /// - `cell_blk`  I32 `[n_kv, ns]`: the block each cell belongs to
    /// - `blk_cells` I32 `[ratio*n_blocks, ns]`: the cells making up each block
    /// - `blk_pos`   I32 `[4*n_blocks*ns]`: the M-RoPE position rows of each
    ///   block's first token
    /// - `bias`      F32 `[n_kv, n_tokens/ns, ns]`: -inf where invisible,
    ///   large where always visible; with `blk_bias`, per block instead,
    ///   `[n_blocks, n_tokens/ns, ns]`, and the caller adds the attention
    ///   mask, the only part of the bias that varies within a block.
    ///
    /// A block is keyed on (sequence set, index bucket): a unified cache
    /// counts every sequence from zero, so the bucket alone would pool two
    /// sequences into one block.
    pub fn setInputQsa(self: *const MemoryHybridIdx, cell_blk: *Tensor, blk_cells: *Tensor, blk_pos: *Tensor, bias: *Tensor, ubatch: *const Ubatch, ratio: u32, blk_bias: bool) void {
        const file = "llama-memory-hybrid-idx.cpp";
        impl.assert(ratio > 0, file, 277, "ratio > 0");
        impl.assert(self.getMemIdx() != null, file, 278, "get_mem_idx() != nullptr");
        impl.assert(c.ggml_backend_buffer_is_host(cell_blk.buffer), file, 280, "ggml_backend_buffer_is_host(cell_blk->buffer)");

        const mem_idx = self.getMemIdx().?;

        const n_kv: i64 = cell_blk.ne[0];
        // The streams in this ubatch.
        const n_ns: i64 = cell_blk.ne[1];
        const n_blocks: i64 = @divTrunc(blk_pos.ne[0], 4 * n_ns);
        const n_tokens: i64 = ubatch.n_tokens;
        const r: i64 = ratio;

        impl.assert(@rem(n_tokens, n_ns) == 0, file, 288, "n_tokens % n_ns == 0");
        // Tokens per stream.
        const n_tps: i64 = @divTrunc(n_tokens, n_ns);

        const dst_cell_blk: [*]i32 = @ptrCast(@alignCast(cell_blk.data));
        const dst_blk_cells: [*]i32 = @ptrCast(@alignCast(blk_cells.data));
        const dst_blk_pos: [*]i32 = @ptrCast(@alignCast(blk_pos.data));
        const dst_bias: [*]f32 = @ptrCast(@alignCast(bias.data));

        impl.assert(r <= 64, file, 298, "r <= 64");
        const slots_full: u64 = if (r == 64) ~@as(u64, 0) else ((@as(u64, 1) << @intCast(r)) - 1);

        const nkv: usize = @intCast(n_kv);
        const nb: usize = @intCast(n_blocks);
        const ru: usize = @intCast(r);

        var st: QsaScratch = .{};
        defer st.deinit();
        st.blk_of.resize(allocator, nkv) catch oom();
        st.cell_grp.resize(allocator, nkv) catch oom();
        st.grp_head.resize(allocator, nb) catch oom();

        @memset(dst_blk_pos[0 .. 4 * nb * @as(usize, @intCast(n_ns))], 0);

        var s: i64 = 0;
        while (s < n_ns) : (s += 1) {
            const su: usize = @intCast(s);
            // The ubatch's token s*n_tps belongs to this stream: ask which
            // cells it uses.
            const seq_of_stream = ubatch.seq_id[@intCast(s * n_tps)][0];
            const cells = mem_idx.getCells(seq_of_stream);

            const cur_cell_blk = dst_cell_blk + su * nkv;
            const cur_blk_cells = dst_blk_cells + su * (ru * nb);

            @memset(cur_blk_cells[0 .. ru * nb], 0);

            st.bid_idx.clearRetainingCapacity();
            st.bid_cell.clearRetainingCapacity();
            st.bid_slot0.clearRetainingCapacity();

            var n_seq_present: c_int = 0;
            var sq: i32 = 0;
            while (sq < max_seq and n_seq_present < 2) : (sq += 1) {
                if (cells.seqPosMin(sq) >= 0) n_seq_present += 1;
            }

            const one_seq = n_seq_present <= 1;

            // A cell no block covers needs its own -inf, which a per-block
            // bias cannot carry; every cache path keeps the position below
            // the cell window, so this stays false.
            var flags: GroupFlags = .{};
            var ranked = false;

            st.groupCells(cells, n_kv, n_blocks, r, one_seq, ranked, &flags);

            // M-RoPE repeats one position across an image: rank the cells
            // instead of using the position.
            if (flags.dup and ubatch.isPos2d() and one_seq) {
                st.order.clearRetainingCapacity();
                st.order.ensureTotalCapacity(allocator, nkv) catch oom();
                for (0..nkv) |j| {
                    if (!cells.isEmpty(@intCast(j))) st.order.appendAssumeCapacity(@intCast(j));
                }

                // The M-RoPE causal mask's total order: pos, then ext.y,
                // then ext.x. libc++'s `std::sort`, whose order among equal
                // cells decides their ranks.
                libcxx.sort(i32, st.order.items, cells, rankLess);

                st.rank.clearRetainingCapacity();
                st.rank.appendNTimes(allocator, -1, nkv) catch oom();
                for (st.order.items, 0..) |cell, k| st.rank.items[@intCast(cell)] = @intCast(k);

                ranked = true;

                st.groupCells(cells, n_kv, n_blocks, r, one_seq, ranked, &flags);
            }

            impl.assert(!blk_bias or !flags.oor, file, 452, "(!blk_bias || !oor) && \"qsa: cell position runs past the cell window\"");

            var n_bid: i32 = 0;
            for (0..nb) |pb| {
                var g = st.grp_head.items[pb];
                while (g >= 0) : (g = st.grp_next.items[@intCast(g)]) {
                    const gu: usize = @intCast(g);
                    if (st.grp_slots.items[gu] != slots_full) continue;

                    st.grp_bid.items[gu] = n_bid;
                    n_bid += 1;

                    st.bid_idx.append(allocator, @intCast(@as(i64, @intCast(pb)) * r)) catch oom();
                    st.bid_cell.append(allocator, st.grp_first.items[gu]) catch oom();
                    st.bid_slot0.append(allocator, st.grp_slot0.items[gu]) catch oom();
                }
            }

            impl.assert(n_bid <= n_blocks, file, 470, "n_bid <= n_blocks");

            for (0..@intCast(n_bid)) |b| {
                var sec_pos: [4]i32 = @splat(st.bid_idx.items[b]);
                if (ranked) {
                    const cell: u32 = @intCast(st.bid_slot0.items[b]);
                    const p = cells.posGet(cell);
                    const e = cells.extGet(cell);
                    sec_pos = .{ p, e.y, e.x, p };
                }
                for (0..4) |sec| {
                    dst_blk_pos[sec * (nb * @as(usize, @intCast(n_ns))) + su * nb + b] = sec_pos[sec];
                }
            }

            // Unpooled cells all point at one spare block, which exists
            // only when some cell is unpooled: n_bid == n_blocks means every
            // cell sits in a full block.
            const have_dead = n_bid < n_blocks;
            const dead_bid: i32 = if (have_dead) n_bid else @intCast(n_blocks - 1);

            for (0..nkv) |j| {
                const g = st.cell_grp.items[j];
                st.blk_of.items[j] = if (g < 0) -1 else st.grp_bid.items[@intCast(g)];
                if (st.blk_of.items[j] >= 0) {
                    const idx: i64 = if (ranked) st.rank.items[j] else cells.posGet(@intCast(j));
                    cur_blk_cells[@as(usize, @intCast(st.blk_of.items[j])) * ru + @as(usize, @intCast(@rem(idx, r)))] = @intCast(j);
                }
                cur_cell_blk[j] = if (st.blk_of.items[j] < 0) dead_bid else st.blk_of.items[j];
            }

            var ii: i64 = 0;
            while (ii < n_tps) : (ii += 1) {
                const i: usize = @intCast(s * n_tps + ii);
                const seq_id = ubatch.seq_id[i][0];

                var q: i64 = ubatch.pos[i];

                if (ranked) {
                    const qt = ubatch.pos[i];
                    const qy = ubatch.pos[i + @as(usize, @intCast(n_tokens))];
                    const qx = ubatch.pos[i + @as(usize, @intCast(n_tokens)) * 2];

                    var lo: i64 = 0;
                    var hi: i64 = @intCast(st.order.items.len);
                    while (lo < hi) {
                        const mid = @divTrunc(lo + hi, 2);
                        const cell: u32 = @intCast(st.order.items[@intCast(mid)]);
                        const pc = cells.posGet(cell);
                        if (pc < qt or (pc == qt and !cells.extGet(cell).is2dGt(qx, qy))) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    q = lo - 1;
                }

                // The tail is an incomplete block, always visible, as in the
                // reference.
                const tail_start: i64 = @divTrunc(q + 1, r) * r;

                if (blk_bias) {
                    // A block sits wholly inside or outside the tail, so one
                    // value covers it; the caller's attention mask drops
                    // empty, foreign and future cells.
                    const cur_blk_bias = dst_bias + i * nb;
                    for (0..nb) |b| {
                        if (b >= @as(usize, @intCast(n_bid)) or !cells.seqHas(@intCast(st.bid_cell.items[b]), seq_id)) {
                            cur_blk_bias[b] = -std.math.inf(f32);
                            continue;
                        }
                        // Finite, so it can never meet a -inf and make a nan.
                        cur_blk_bias[b] = if (st.bid_idx.items[b] >= tail_start) 1e9 else 0.0;
                    }

                    // The spare block holds the unpooled cells, the
                    // incomplete tail, so it takes the tail's value; finite,
                    // since a sequence with fewer than `ratio` cells owns no
                    // full block and a row of -inf gives only a nan.
                    if (have_dead) cur_blk_bias[@intCast(dead_bid)] = 1e9;
                    continue;
                }

                const cur_bias = dst_bias + i * nkv;
                for (0..nkv) |j| {
                    var v: f32 = -std.math.inf(f32);
                    const ju: u32 = @intCast(j);
                    if (!cells.isEmpty(ju) and cells.seqHas(ju, seq_id)) {
                        const idx: i64 = if (ranked) st.rank.items[j] else cells.posGet(ju);
                        if (idx <= q) {
                            // Finite, so it can never meet a -inf and make a nan.
                            v = if (idx >= tail_start) 1e9 else (if (st.blk_of.items[j] < 0) -std.math.inf(f32) else 0.0);
                        }
                    }
                    cur_bias[j] = v;
                }
            }
        }
    }

    fn rankLess(cells: *const KvCells, a: i32, b: i32) bool {
        const pa = cells.posGet(@intCast(a));
        const pb = cells.posGet(@intCast(b));
        if (pa != pb) return pa < pb;
        const ea = cells.extGet(@intCast(a));
        return cells.extGet(@intCast(b)).is2dGt(ea.x, ea.y);
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *MemoryHybridIdx) memory.Memory {
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

    fn from(p: *anyopaque) *MemoryHybridIdx {
        return @ptrCast(@alignCast(p));
    }
};

/// What `group_cells` sets besides the groups.
const GroupFlags = struct {
    /// A cell's block runs past the cell window.
    oor: bool = false,
    /// Two cells of a group share a slot.
    dup: bool = false,
};

/// The per-call vectors of `set_input_qsa`.
const QsaScratch = struct {
    blk_of: std.ArrayList(i32) = .empty,
    cell_grp: std.ArrayList(i32) = .empty,
    grp_head: std.ArrayList(i32) = .empty,
    grp_next: std.ArrayList(i32) = .empty,
    grp_first: std.ArrayList(i32) = .empty,
    grp_slot0: std.ArrayList(i32) = .empty,
    grp_slots: std.ArrayList(u64) = .empty,
    grp_bid: std.ArrayList(i32) = .empty,
    bid_idx: std.ArrayList(i32) = .empty,
    bid_cell: std.ArrayList(i32) = .empty,
    bid_slot0: std.ArrayList(i32) = .empty,
    order: std.ArrayList(i32) = .empty,
    rank: std.ArrayList(i32) = .empty,

    fn deinit(self: *QsaScratch) void {
        inline for (@typeInfo(QsaScratch).@"struct".field_names) |name| @field(self, name).deinit(allocator);
    }

    /// Ports `group_cells` (llama-memory-hybrid-idx.cpp:356 @b29c606e2), a
    /// lambda inside `set_input_qsa`: each non-empty cell
    /// into the group of its block and sequence set; -1 where no usable
    /// block, an incomplete or short group, takes it.
    fn groupCells(self: *QsaScratch, cells: *const KvCells, n_kv: i64, n_blocks: i64, r: i64, one_seq: bool, ranked: bool, flags: *GroupFlags) void {
        @memset(self.blk_of.items, -1);
        @memset(self.cell_grp.items, -1);
        @memset(self.grp_head.items, -1);

        self.grp_next.clearRetainingCapacity();
        self.grp_first.clearRetainingCapacity();
        self.grp_slot0.clearRetainingCapacity();
        self.grp_slots.clearRetainingCapacity();
        self.grp_bid.clearRetainingCapacity();

        flags.* = .{};

        var j: i64 = 0;
        while (j < n_kv) : (j += 1) {
            const ju: u32 = @intCast(j);
            if (cells.isEmpty(ju)) continue;

            const idx: i64 = if (ranked) self.rank.items[ju] else cells.posGet(ju);
            const pb = @divTrunc(idx, r);

            if (pb >= n_blocks) {
                flags.oor = true;
                continue;
            }

            var g: i32 = -1;
            var cand = self.grp_head.items[@intCast(pb)];
            while (cand >= 0) : (cand = self.grp_next.items[@intCast(cand)]) {
                if (one_seq or cells.seqGetAll(@intCast(self.grp_first.items[@intCast(cand)])).eql(cells.seqGetAll(ju))) {
                    g = cand;
                    break;
                }
            }

            if (g < 0) {
                g = @intCast(self.grp_first.items.len);
                self.grp_next.append(allocator, self.grp_head.items[@intCast(pb)]) catch oom();
                self.grp_first.append(allocator, @intCast(j)) catch oom();
                self.grp_slot0.append(allocator, -1) catch oom();
                self.grp_slots.append(allocator, 0) catch oom();
                self.grp_bid.append(allocator, -1) catch oom();
                self.grp_head.items[@intCast(pb)] = g;
            }

            const gu: usize = @intCast(g);
            const bit: u64 = @as(u64, 1) << @intCast(@rem(idx, r));

            flags.dup = flags.dup or (self.grp_slots.items[gu] & bit) != 0;

            self.cell_grp.items[ju] = g;
            self.grp_slots.items[gu] |= bit;

            if (@rem(idx, r) == 0) self.grp_slot0.items[gu] = @intCast(j);
        }
    }
};

/// Ports `llama_memory_hybrid_idx_ns` (llama-memory-hybrid-idx.cpp:596
/// @b29c606e2): the streams in each ubatch's slot info, `get_k`/`get_v`'s
/// `ns`.
fn streamsPerUbatch(sinfos: []const SlotInfo) std.ArrayList(u32) {
    var res: std.ArrayList(u32) = .empty;
    res.ensureTotalCapacity(allocator, sinfos.len) catch oom();
    for (sinfos) |*si| res.appendAssumeCapacity(si.s1 - si.s0 + 1);
    return res;
}

/// Ports `llama_memory_hybrid_idx_context` (llama-memory-hybrid-idx.h:102
/// @b29c606e2).
pub const HybridIdxContext = struct {
    /// The `llama_memory_hybrid_idx_context`'s base. The `memory.Context`
    /// points here.
    base: HybridContext,
    mem: ?*const MemoryHybridIdx = null,
    /// Streams per ubatch, read from the slot infos before `ctx_idx` takes
    /// them.
    ns_ubatch: std.ArrayList(u32) = .empty,
    /// Null unless the model has an indexer.
    ctx_idx: ?*kv_cache.KvCacheContext = null,
    /// The base's ubatch cursor, private there.
    i_cur: usize = 0,

    /// Ports the status constructor of `llama_memory_hybrid_idx_context`
    /// (llama-memory-hybrid-idx.cpp:607 @b29c606e2).
    pub fn createStatus(status: memory.Status) *HybridIdxContext {
        const self = allocator.create(HybridIdxContext) catch oom();
        self.* = .{ .base = HybridContext.initStatus(status) };
        return self;
    }

    /// Ports the full-memory constructor of `llama_memory_hybrid_idx_context`
    /// (llama-memory-hybrid-idx.cpp:610 @b29c606e2). Graph reservation walks
    /// a full context, and qwen4exp builds the sparse attention only when
    /// the indexer context is set: without it the reserved worst case
    /// would be the dense graph.
    pub fn createFull(mem: *MemoryHybridIdx) *HybridIdxContext {
        const self = allocator.create(HybridIdxContext) catch oom();
        self.* = .{ .base = HybridContext.initFull(mem.base), .mem = mem };
        if (mem.getMemIdx()) |m| {
            self.ns_ubatch.append(allocator, m.getNStream()) catch oom();
            self.ctx_idx = kv_cache.KvCacheContext.createFull(m);
        }
        return self;
    }

    /// Ports the update constructor of `llama_memory_hybrid_idx_context`
    /// (llama-memory-hybrid-idx.cpp:620 @b29c606e2): `update` applies a
    /// pending cross-stream `seq_cp`, without which the copy keeps stale
    /// indexer keys.
    pub fn createUpdate(mem: *MemoryHybridIdx, lctx: *anyopaque, optimize: bool) *HybridIdxContext {
        const self = allocator.create(HybridIdxContext) catch oom();
        self.* = .{ .base = HybridContext.initUpdate(mem.base, lctx, optimize), .mem = mem };
        if (mem.getMemIdx()) |m| self.ctx_idx = m.initUpdate(lctx, optimize);
        return self;
    }

    /// Ports the batch constructor of `llama_memory_hybrid_idx_context`
    /// (llama-memory-hybrid-idx.cpp:630 @b29c606e2): takes the three lists;
    /// the base and the indexer each get their own copy of the ubatches.
    pub fn createBatch(mem: *MemoryHybridIdx, sinfos_attn: std.ArrayList(SlotInfo), sinfos_idx: std.ArrayList(SlotInfo), ubatches: std.ArrayList(Ubatch)) *HybridIdxContext {
        const self = allocator.create(HybridIdxContext) catch oom();
        const ns = streamsPerUbatch(sinfos_idx.items);
        var ctx_idx: ?*kv_cache.KvCacheContext = null;
        if (mem.getMemIdx()) |m| {
            ctx_idx = kv_cache.KvCacheContext.createBatch(m, sinfos_idx, memory_hybrid.retainedCopy(ubatches.items));
        } else {
            var unused = sinfos_idx;
            deinitSinfos(&unused);
        }
        self.* = .{ .base = HybridContext.initBatch(mem.base, sinfos_attn, ubatches), .mem = mem, .ns_ubatch = ns, .ctx_idx = ctx_idx };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *HybridIdxContext) void {
        if (self.ctx_idx) |x| x.destroy();
        self.ns_ubatch.deinit(allocator);
        self.base.deinit();
        allocator.destroy(self);
    }

    /// The extension behind a hybrid context, the C++'s
    /// `static_cast<const llama_memory_hybrid_idx_context *>`: valid only
    /// for a context `MemoryHybridIdx` made, as qwen4exp's always is.
    pub fn fromBase(base: *const HybridContext) *const HybridIdxContext {
        return @alignCast(@fieldParentPtr("base", base));
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `next` (llama-memory-hybrid-idx.cpp:642 @b29c606e2).
    pub fn next(self: *HybridIdxContext) bool {
        if (self.ctx_idx) |x| _ = x.next();
        self.i_cur += 1;
        return self.base.next();
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `apply`
    /// (llama-memory-hybrid-idx.cpp:652 @b29c606e2): `&`, so the indexer
    /// applies whatever the base answered.
    pub fn apply(self: *HybridIdxContext) bool {
        var res = self.base.apply();
        if (self.ctx_idx) |x| res = @intFromBool(res) & @intFromBool(x.apply()) != 0;
        return res;
    }

    /// `llama_memory_hybrid_context`'s `get_status`.
    pub fn getStatus(self: *const HybridIdxContext) memory.Status {
        return self.base.getStatus();
    }

    /// `llama_memory_hybrid_context`'s `get_ubatch`.
    pub fn getUbatch(self: *const HybridIdxContext) *const Ubatch {
        return self.base.getUbatch();
    }

    /// `llama_memory_hybrid_context`'s `get_attn`.
    pub fn getAttn(self: *const HybridIdxContext) *const kv_cache.KvCacheContext {
        return self.base.getAttn();
    }

    /// `llama_memory_hybrid_context`'s `get_recr`.
    pub fn getRecr(self: *const HybridIdxContext) *const memory_recurrent.RecurrentContext {
        return self.base.getRecr();
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `get_idx`
    /// (llama-memory-hybrid-idx.cpp:662 @b29c606e2): null with no indexer.
    pub fn getIdx(self: *const HybridIdxContext) ?*const kv_cache.KvCacheContext {
        return self.ctx_idx;
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `get_n_stream`
    /// (llama-memory-hybrid-idx.cpp:666 @b29c606e2): the streams in the
    /// current slot info, `get_k`/`get_v`'s `ns`; 1 when unified.
    pub fn getNStream(self: *const HybridIdxContext) u32 {
        impl.assert(self.i_cur < self.ns_ubatch.items.len, "llama-memory-hybrid-idx.cpp", 667, "i_cur < ns_ubatch.size()");
        return self.ns_ubatch.items[self.i_cur];
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `set_input_qsa`
    /// (llama-memory-hybrid-idx.cpp:672 @b29c606e2); see
    /// `MemoryHybridIdx.setInputQsa`.
    pub fn setInputQsa(self: *const HybridIdxContext, cell_blk: *Tensor, blk_cells: *Tensor, blk_pos: *Tensor, bias: *Tensor, ubatch: *const Ubatch, ratio: u32, blk_bias: bool) void {
        const mem = self.mem orelse impl.abort("llama-memory-hybrid-idx.cpp", 680, "GGML_ASSERT(mem != nullptr) failed");
        mem.setInputQsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
    }

    /// The `memory.Context`, pointing at `base` so the graph's
    /// `cast(HybridContext)` sees the base.
    pub fn contextOf(self: *HybridIdxContext) memory.Context {
        return .{ .ptr = &self.base, .vtable = &ctx_vtable };
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

    fn cast(p: *anyopaque) *HybridIdxContext {
        const base: *HybridContext = @ptrCast(@alignCast(p));
        return @alignCast(@fieldParentPtr("base", base));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(MemoryHybridIdx);
    std.testing.refAllDecls(HybridIdxContext);
}

test "a context's base pointer recovers the extension" {
    const ctx = HybridIdxContext.createStatus(.failed_prepare);
    const mc = ctx.contextOf();
    try std.testing.expect(mc.cast(HybridContext) == &ctx.base);
    try std.testing.expect(HybridIdxContext.fromBase(mc.cast(HybridContext)) == ctx);
    try std.testing.expectEqual(memory.Status.failed_prepare, mc.getStatus());
    mc.destroy();
}
