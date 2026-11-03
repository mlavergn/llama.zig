//! A hybrid memory with a third cache: one indexer row per token, for the
//! k-pool DSA indexers of glm5-next and qwen4exp.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-memory-hybrid-idx.h` -- `llama_memory_hybrid_idx`
//!   and `llama_memory_hybrid_idx_context`
//! - `llama.cpp/src/llama-memory-hybrid-idx.cpp` -- their bodies,
//!   `kpool_layout`, `kpool_state`, `kpool_access`, `kpool_pad`,
//!   `kpool_rank` and `llama_memory_hybrid_idx_ns`
//!
//! All at v0.6.0 (`d81235049`).
//!
//! # A side buffer over the attention cells
//!
//! The indexer cache has the attention cache's size, padding, streams and
//! slots, so cell `j` is one token in both. It never searches for cells of
//! its own: a batch takes the attention cache's slot infos, and a restore
//! takes the cells the attention restore chose (`[TAG_HYBRID_IDX_SINFO]`).
//!
//! # Pools
//!
//! A k-pool indexer caches, per token, its key (and glm5-next's gate) and,
//! in the last member's row of each complete pool of `kpool` tokens, the
//! pooled key. The memory keeps which cells make up which pool
//! (`KpoolLayout`), extended as ubatches append and rebuilt for a sequence
//! after an edit staled it (`mem_idx_stale`); each batch context works out
//! which pools its current ubatch must re-pool (`KpoolState`) and fills the
//! graph's inputs from both (`setInputKpool`).
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

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;
const Ubatch = batch.Ubatch;
const Tensor = c.ggml_tensor;
const KvCache = kv_cache.KvCache;
const KvCells = kv_cells.KvCells;
const PosCell = kv_cells.PosCell;
const SlotInfo = kv_cache.SlotInfo;
const MemoryHybrid = memory_hybrid.MemoryHybrid;
const HybridContext = memory_hybrid.HybridContext;

/// `LLAMA_MAX_SEQ` (llama-cparams.h:8 @d81235049).
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

/// Ports `llama_memory_hybrid_idx`'s `POS_CLEAN` (llama-memory-hybrid-idx.h:99
/// @d81235049): no pooled key is stale.
pub const pos_clean: i32 = std.math.maxInt(i32);

/// Ports `llama_memory_hybrid_idx`'s `stale_pos_t`: per sequence, the
/// position its pooled keys are stale from.
pub const StalePos = [max_seq]i32;

/// Ports `llama_memory_hybrid_idx`'s `stale_pos_clean`
/// (llama-memory-hybrid-idx.h:101 @d81235049).
pub fn stalePosClean() StalePos {
    return @splat(pos_clean);
}

fn lessPosCell(a: PosCell, b: PosCell) bool {
    return a.pos < b.pos or (a.pos == b.pos and a.cell < b.cell);
}

/// `std::lower_bound` over (position, cell) pairs.
fn lowerBoundPosCell(cells: []const PosCell, k: PosCell) usize {
    var lo: usize = 0;
    var hi: usize = cells.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (lessPosCell(cells[mid], k)) lo = mid + 1 else hi = mid;
    }
    return lo;
}

/// `std::upper_bound` over (position, cell) pairs.
fn upperBoundPosCell(cells: []const PosCell, k: PosCell) usize {
    var lo: usize = 0;
    var hi: usize = cells.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (!lessPosCell(k, cells[mid])) lo = mid + 1 else hi = mid;
    }
    return lo;
}

/// Ports `kpool_pad` (llama-memory-hybrid-idx.cpp:386 @d81235049): the
/// padded pool count, whose last pool is always unused.
fn kpoolPad(n_pool: u32) u32 {
    return @max(64, (n_pool +% 1 +% 63) / 64 * 64);
}

/// Ports `kpool_rank` (llama-memory-hybrid-idx.cpp:392 @d81235049): the rank
/// of (pos, cell) among a sequence's cells, ordered by position then cell,
/// or -1 when absent. In order mode the rank alone places a token: cells
/// sharing a position (an M-RoPE image) have distinct ranks.
fn kpoolRank(cells: []const PosCell, pos: i32, cell: u32) i64 {
    const it = lowerBoundPosCell(cells, .{ .pos = pos, .cell = cell });
    return if (it != cells.len and cells[it].cell == cell and cells[it].pos == pos) @intCast(it) else -1;
}

/// Ports `llama_memory_hybrid_idx`'s `kpool_layout` (llama-memory-hybrid-idx.cpp:352
/// @d81235049): which cells of each sequence make up which pool of `kpool`
/// consecutive positions (or cells, in order mode), for the whole cache. It
/// outlives the batch: pools are fixed by positions relative to the
/// sequence's first, so a ubatch only ever appends to it.
pub const KpoolLayout = struct {
    pub const Seq = struct {
        pos_min: i32 = 0,
        /// The stream holding this sequence's cells.
        strm: u32 = 0,
        /// (position, stream-local cell) pairs, ordered by position.
        cells: std.ArrayList(PosCell) = .empty,
        /// Each pool's first index into `cells`.
        pools: std.ArrayList(u32) = .empty,
        /// Where the pool scan stopped, so an append resumes there.
        j_next: usize = 0,
        /// Whether any cell carries another sequence too, which rules out
        /// caching this sequence's pooled keys.
        shared: bool = false,

        fn deinit(self: *Seq) void {
            self.cells.deinit(allocator);
            self.pools.deinit(allocator);
        }

        /// `seq()`: back to the default.
        fn reset(self: *Seq) void {
            self.deinit();
            self.* = .{};
        }
    };

    seqs: [max_seq]Seq = @splat(.{}),
    n_pool_real: u32 = 0,
    cache_safe: bool = true,

    fn deinit(self: *KpoolLayout) void {
        for (&self.seqs) |*sq| sq.deinit();
    }
};

/// Ports `llama_memory_hybrid_idx_context`'s `kpool_state`
/// (llama-memory-hybrid-idx.cpp:373 @d81235049): which pools of the layout
/// the current ubatch must re-pool, in the layout's pool order.
const KpoolState = struct {
    is_new: std.ArrayList(u32) = .empty,
    generation: u32 = 0,
    n_pool_real: u32 = 0,
    n_new: u32 = 0,
    /// The new pool list's size in the graph, stable across decode steps.
    n_new_g: u32 = 1,
    cache_safe: bool = true,

    fn deinit(self: *KpoolState) void {
        self.is_new.deinit(allocator);
    }
};

/// Ports `llama_memory_hybrid_idx` (llama-memory-hybrid-idx.h:17 @d81235049).
pub const MemoryHybridIdx = struct {
    /// The `llama_memory_hybrid` this extends.
    base: *MemoryHybrid,
    /// The indexer cache holds one key head per layer, so it needs its own
    /// hparams, and the cache keeps a pointer to what it is given: this
    /// struct is heap-allocated and does not move.
    hparams_idx: Hparams,
    /// Null when the model carries no indexer.
    mem_idx: ?*KvCache = null,

    /// Null until the first `kpoolLayoutUpdate`.
    kpool_lay: ?*KpoolLayout = null,

    /// Per sequence, the position its pooled keys are stale from.
    mem_idx_stale: StalePos = stalePosClean(),

    /// Ports the constructor of `llama_memory_hybrid_idx`
    /// (llama-memory-hybrid-idx.cpp:23 @d81235049): the hybrid memory, and
    /// the indexer cache when `filter_idx` is given (`func != null`). The
    /// indexer is MQA with one key head of `indexer_head_size`, as
    /// `llama_kv_cache_dsa` shapes its own -- `indexer_kpool_row` heads
    /// wide for a k-pool indexer, which caches its per-token rows and the
    /// pooled key side by side (glm5-next: key | gate | pooled, qwen4exp:
    /// key | pooled) -- and has no RoPE: its keys are cached raw and rotated
    /// after pooling, so a K-shift must not rotate them. It is made to look
    /// like an MLA cache so no V is stored.
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
            const hp = &model.hparams;
            @memset(&self.hparams_idx.n_head_kv_arr, 1);
            self.hparams_idx.n_embd_head_k_full = hp.indexer_head_size *% (if (hp.indexer_kpool > 0) hp.indexer_kpool_row else 1);
            self.hparams_idx.rope_type = c.LLAMA_ROPE_TYPE_NONE;
            // MLA-shaped, so the cache stores no V tensors.
            self.hparams_idx.n_embd_head_k_mla_impl = hp.indexer_head_size;
            self.hparams_idx.n_embd_head_v_mla_impl = hp.indexer_head_size;

            impl.logInfo("%s: creating indexer KV cache, size = %u cells\n", .{ "operator()", kv_size });

            self.mem_idx = try KvCache.createTagged(model, &self.hparams_idx, type_k, type_v, v_trans, offload, unified, kv_size, n_seq_max, n_pad, n_swa, swa_type, null, filter_idx, .{}, .{}, "idx_", exc);
        }
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *MemoryHybridIdx) void {
        if (self.kpool_lay) |l| {
            l.deinit();
            allocator.destroy(l);
        }
        if (self.mem_idx) |m| m.destroy();
        self.base.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_memory_hybrid_idx`'s `init_batch`
    /// (llama-memory-hybrid-idx.cpp:76 @d81235049): `llama_memory_hybrid`'s,
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

    /// Ports `llama_memory_hybrid_idx`'s `init_full` (llama-memory-hybrid-idx.cpp:141 @d81235049).
    pub fn initFull(self: *MemoryHybridIdx) *HybridIdxContext {
        return HybridIdxContext.createFull(self);
    }

    /// Ports `llama_memory_hybrid_idx`'s `init_update` (llama-memory-hybrid-idx.cpp:145 @d81235049).
    pub fn initUpdate(self: *MemoryHybridIdx, lctx: *anyopaque, optimize: bool) *HybridIdxContext {
        return HybridIdxContext.createUpdate(self, lctx, optimize);
    }

    /// `llama_memory_hybrid`'s `get_can_shift`, which this does not override.
    pub fn getCanShift(self: *const MemoryHybridIdx) bool {
        return self.base.getCanShift();
    }

    /// Ports `llama_memory_hybrid_idx`'s `clear` (llama-memory-hybrid-idx.cpp:149 @d81235049).
    pub fn clear(self: *MemoryHybridIdx, data: bool) void {
        self.base.clear(data);
        if (self.mem_idx) |m| {
            m.clear(data);
            self.memIdxStaleSet(-1, 0);
        }
    }

    /// Ports `llama_memory_hybrid_idx`'s `mem_idx_stale_set`
    /// (llama-memory-hybrid-idx.cpp:160 @d81235049): a pooled key is valid
    /// only while the grouping that produced it holds, and grouping is
    /// relative to the sequence, so an edit at `p0` leaves every pool ending
    /// before it alone. `seq_id < 0` stales every sequence.
    fn memIdxStaleSet(self: *MemoryHybridIdx, seq_id: i32, p0_: i32) void {
        const p0 = @max(p0_, 0);

        if (seq_id < 0) {
            for (&self.mem_idx_stale) |*p| p.* = @min(p.*, p0);
            return;
        }

        impl.assert(seq_id < max_seq, "llama-memory-hybrid-idx.cpp", 171, "seq_id < (llama_seq_id) LLAMA_MAX_SEQ");

        const s: usize = @intCast(seq_id);
        self.mem_idx_stale[s] = @min(self.mem_idx_stale[s], p0);
    }

    /// Ports `llama_memory_hybrid_idx`'s `mem_idx_stale_pos`
    /// (llama-memory-hybrid-idx.cpp:177 @d81235049): the position an edit at
    /// `p0` stales the sequence from -- 0 at or below its first position,
    /// which moves `pos_min` and regroups the whole of it.
    fn memIdxStalePos(self: *const MemoryHybridIdx, seq_id: i32, p0: i32) i32 {
        if (seq_id < 0 or p0 <= self.mem_idx.?.seqPosMin(seq_id)) return 0;
        return p0;
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_rm` (llama-memory-hybrid-idx.cpp:185
    /// @d81235049): in `llama_memory_hybrid`'s order, the recurrent cache
    /// first, since it may refuse.
    pub fn seqRm(self: *MemoryHybridIdx, seq_id: i32, p0: i32, p1: i32) bool {
        if (!self.getMemRecr().seqRm(seq_id, p0, p1)) return false;

        if (self.mem_idx) |m| {
            const stale = self.memIdxStalePos(seq_id, p0);
            _ = m.seqRm(seq_id, p0, p1);
            self.memIdxStaleSet(seq_id, stale);

            // Removing a sequence can free cells another one shared, and
            // only this one is marked stale, so the survivor would keep
            // `shared` and pin `cache_safe` off: stale every sequence to
            // re-derive it.
            if (self.kpoolLayoutShared()) self.memIdxStaleSet(-1, 0);
        }

        return self.getMemAttn().seqRm(seq_id, p0, p1);
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_cp` (llama-memory-hybrid-idx.cpp:206
    /// @d81235049): the copy shares cells, which cannot hold two groupings,
    /// so both sides drop their cached keys.
    pub fn seqCp(self: *MemoryHybridIdx, a: i32, b: i32, p0: i32, p1: i32) void {
        self.base.seqCp(a, b, p0, p1);
        if (self.mem_idx) |m| {
            m.seqCp(a, b, p0, p1);
            self.memIdxStaleSet(a, 0);
            self.memIdxStaleSet(b, 0);
        }
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_keep` (llama-memory-hybrid-idx.cpp:217
    /// @d81235049): cells shared with the dropped sequences become exclusive
    /// again, and their keys were never cached.
    pub fn seqKeep(self: *MemoryHybridIdx, seq_id: i32) void {
        self.base.seqKeep(seq_id);
        if (self.mem_idx) |m| {
            m.seqKeep(seq_id);
            self.memIdxStaleSet(-1, 0);
        }
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_add` (llama-memory-hybrid-idx.cpp:227
    /// @d81235049): a negative shift moves cells below `p0`, which regroup
    /// as well.
    pub fn seqAdd(self: *MemoryHybridIdx, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.base.seqAdd(seq_id, p0, p1, shift);
        if (self.mem_idx) |m| {
            const stale = self.memIdxStalePos(seq_id, if (shift < 0) p0 +% shift else p0);
            m.seqAdd(seq_id, p0, p1, shift);
            self.memIdxStaleSet(seq_id, stale);
        }
    }

    /// Ports `llama_memory_hybrid_idx`'s `seq_div` (llama-memory-hybrid-idx.cpp:238 @d81235049).
    pub fn seqDiv(self: *MemoryHybridIdx, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.base.seqDiv(seq_id, p0, p1, d);
        if (self.mem_idx) |m| {
            m.seqDiv(seq_id, p0, p1, d);
            self.memIdxStaleSet(seq_id, 0);
        }
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
    /// (llama-memory-hybrid-idx.cpp:247 @d81235049): the indexer's bytes
    /// added to the hybrid memory's, per buffer type.
    pub fn memoryBreakdown(self: *const MemoryHybridIdx, out: *std.ArrayList(memory.BuftSize)) void {
        self.base.memoryBreakdown(out);
        if (self.mem_idx) |m| m.memoryBreakdown(out);
    }

    /// Ports `llama_memory_hybrid_idx`'s `state_write`
    /// (llama-memory-hybrid-idx.cpp:259 @d81235049): the indexer section
    /// goes last, a pure suffix, under the attention cache's PARTIAL_ONLY
    /// gate ([TAG_HYBRID_IDX_STATE]).
    pub fn stateWrite(self: *const MemoryHybridIdx, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        try self.base.stateWrite(w, seq_id, flags, exc);
        if (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 0) {
            if (self.mem_idx) |m| try m.stateWrite(w, seq_id, flags, exc);
        }
    }

    /// Ports `llama_memory_hybrid_idx`'s `state_read`
    /// (llama-memory-hybrid-idx.cpp:272 @d81235049): `llama_memory_hybrid`'s,
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
            if (self.mem_idx) |m| {
                try m.stateReadSinfo(r, seq_id, flags, null, sinfos_attn.items, exc);
                // the restore rewrites the cells behind the pool layout's back
                self.memIdxStaleSet(seq_id, 0);
                // and can change which cells are shared: re-derive sharing
                // for every sequence, as `seqRm` does
                if (self.kpoolLayoutShared()) self.memIdxStaleSet(-1, 0);
            }
        }
    }

    /// Ports `llama_memory_hybrid_idx`'s `state_drop`
    /// (llama-memory-hybrid-idx.cpp:310 @d81235049): forget `seq_id` in
    /// every cache at once, or the whole context when it is negative --
    /// directly, not through `seqRm`, which the recurrent cache may refuse
    /// and leave only the other two cleared.
    fn stateDrop(self: *MemoryHybridIdx, seq_id: i32) void {
        if (seq_id < 0) {
            self.clear(true);
            return;
        }
        self.getMemAttn().stateClear(seq_id);
        _ = self.getMemRecr().seqRm(seq_id, -1, -1);
        if (self.mem_idx) |m| {
            m.stateClear(seq_id);
            self.memIdxStaleSet(seq_id, 0);
            // clearing this sequence can end a sharing the survivor would
            // otherwise keep flagged (see `seqRm`)
            if (self.kpoolLayoutShared()) self.memIdxStaleSet(-1, 0);
        }
    }

    /// `llama_memory_hybrid`'s `get_mem_attn`.
    pub fn getMemAttn(self: *const MemoryHybridIdx) *KvCache {
        return self.base.getMemAttn();
    }

    /// `llama_memory_hybrid`'s `get_mem_recr`.
    pub fn getMemRecr(self: *const MemoryHybridIdx) *memory_recurrent.MemoryRecurrent {
        return self.base.getMemRecr();
    }

    /// Ports `llama_memory_hybrid_idx`'s `get_mem_idx` (llama-memory-hybrid-idx.cpp:331
    /// @d81235049): null when the model carries no indexer.
    pub fn getMemIdx(self: *const MemoryHybridIdx) ?*KvCache {
        return self.mem_idx;
    }

    /// Ports `llama_memory_hybrid_idx`'s `get_kpool` (llama-memory-hybrid-idx.h:82
    /// @d81235049): the model's indexer pool size.
    pub fn getKpool(self: *const MemoryHybridIdx) u32 {
        return self.hparams_idx.indexer_kpool;
    }

    /// Ports `llama_memory_hybrid_idx`'s `get_kpool_by_order`
    /// (llama-memory-hybrid-idx.h:85 @d81235049): whether a pool is `kpool`
    /// consecutive cells in sequence order (qwen4exp) rather than `kpool`
    /// consecutive positions.
    pub fn getKpoolByOrder(self: *const MemoryHybridIdx) bool {
        return self.hparams_idx.indexer_kpool_by_order;
    }

    /// Ports `llama_memory_hybrid_idx`'s `mem_idx_stale_get` (llama-memory-hybrid-idx.h:107 @d81235049).
    pub fn memIdxStaleGet(self: *const MemoryHybridIdx) *const StalePos {
        return &self.mem_idx_stale;
    }

    /// Ports `llama_memory_hybrid_idx`'s `mem_idx_stale_clear` (llama-memory-hybrid-idx.h:108 @d81235049).
    pub fn memIdxStaleClear(self: *MemoryHybridIdx) void {
        self.mem_idx_stale = stalePosClean();
    }

    /// Ports `llama_memory_hybrid_idx`'s `kpool_layout_get`
    /// (llama-memory-hybrid-idx.cpp:401 @d81235049).
    pub fn kpoolLayoutGet(self: *const MemoryHybridIdx) *const KpoolLayout {
        impl.assert(self.kpool_lay != null, "llama-memory-hybrid-idx.cpp", 402, "kpool_lay != nullptr");
        return self.kpool_lay.?;
    }

    /// Ports `llama_memory_hybrid_idx`'s `kpool_layout_shared`
    /// (llama-memory-hybrid-idx.cpp:407 @d81235049): whether the current
    /// layout has cells shared between sequences.
    fn kpoolLayoutShared(self: *const MemoryHybridIdx) bool {
        const l = self.kpool_lay orelse return false;
        return !l.cache_safe;
    }

    /// Ports `llama_memory_hybrid_idx`'s `kpool_layout_update`
    /// (llama-memory-hybrid-idx.cpp:413 @d81235049): extend each sequence's
    /// pools with the cells appended since, or rebuild them when an edit
    /// staled the sequence or its cells no longer add up.
    ///
    /// Return: the layout, borrowed from the memory.
    pub fn kpoolLayoutUpdate(self: *MemoryHybridIdx) *const KpoolLayout {
        impl.assert(self.mem_idx != null, "llama-memory-hybrid-idx.cpp", 414, "mem_idx != nullptr");
        const m = self.mem_idx.?;

        if (self.kpool_lay == null) {
            const l = allocator.create(KpoolLayout) catch oom();
            l.* = .{};
            self.kpool_lay = l;
        }
        const lay = self.kpool_lay.?;

        const kpool = self.getKpool();
        const n_stream_kv = m.getNStream();
        const unified = n_stream_kv == 1;

        lay.n_pool_real = 0;
        lay.cache_safe = true;

        for (0..max_seq) |su| {
            const s: i32 = @intCast(su);
            const sq = &lay.seqs[su];

            // A cache that is not unified gives each sequence its own
            // stream, with stream-local cell indices.
            if (!unified and su >= n_stream_kv) {
                sq.reset();
                continue;
            }

            const cells = m.getCells(if (unified) 0 else s);
            const sp = cells.seqPosGet(s);

            sq.strm = if (unified) 0 else m.getStream(s);

            var n_kept: usize = 0;
            if (self.mem_idx_stale[su] == pos_clean and sq.cells.items.len != 0 and sp.len != 0 and sq.pos_min == sp[0].pos) {
                n_kept = sq.cells.items.len;
                const back = sq.cells.items[sq.cells.items.len - 1];
                sq.cells.appendSlice(allocator, sp[upperBoundPosCell(sp, back)..]) catch oom();
            }

            // The appended tail accounts for every cell only when nothing
            // before it was dropped, and an edit can regroup a sequence
            // without changing its cell count: a stale sequence rebuilds
            // regardless.
            if (sq.cells.items.len != sp.len or self.mem_idx_stale[su] != pos_clean) {
                sq.cells.clearRetainingCapacity();
                sq.cells.appendSlice(allocator, sp) catch oom();
                sq.pools.clearRetainingCapacity();
                sq.j_next = 0;
                sq.shared = false;
                sq.pos_min = if (sp.len == 0) 0 else sp[0].pos;
                n_kept = 0;
            }

            // Sharing starts with a seq_cp and ends with an edit, or a
            // seq_rm/state_drop/state_read freeing the shared cells, each of
            // which stales every sequence so the rebuild above re-derives
            // it: once set it holds until then, and the rescan is skipped.
            if (unified and !sq.shared) {
                for (sq.cells.items[n_kept..]) |pc| {
                    if (cells.seqCount(pc.cell) > 1) {
                        sq.shared = true;
                        break;
                    }
                }
            }

            // Pools start at the first valid token.
            var j = sq.j_next;
            const n = sq.cells.items.len;
            if (self.hparams_idx.indexer_kpool_by_order) {
                // Consecutive cells in sequence order, whatever their positions.
                while (j + kpool <= n) : (j += kpool) {
                    sq.pools.append(allocator, @intCast(j)) catch oom();
                }
            } else {
                while (j + kpool <= n) {
                    const p0 = sq.cells.items[j].pos;
                    if (@rem(p0 -% sq.pos_min, @as(i32, @bitCast(kpool))) != 0) {
                        j += 1;
                        continue;
                    }
                    var ok = true;
                    var k: u32 = 1;
                    while (k < kpool) : (k += 1) {
                        if (sq.cells.items[j + k].pos != p0 +% @as(i32, @bitCast(k))) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        sq.pools.append(allocator, @intCast(j)) catch oom();
                        j += kpool;
                    } else {
                        j += 1;
                    }
                }
            }
            sq.j_next = j;

            lay.n_pool_real +%= @intCast(sq.pools.items.len);
            lay.cache_safe = lay.cache_safe and !sq.shared;
        }

        return lay;
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

/// Ports `llama_memory_hybrid_idx_ns` (llama-memory-hybrid-idx.cpp:340
/// @d81235049): the streams in each ubatch's slot info, `get_k`/`get_v`'s
/// `ns`.
fn streamsPerUbatch(sinfos: []const SlotInfo) std.ArrayList(u32) {
    var res: std.ArrayList(u32) = .empty;
    res.ensureTotalCapacity(allocator, sinfos.len) catch oom();
    for (sinfos) |*si| res.appendAssumeCapacity(si.s1 - si.s0 + 1);
    return res;
}

/// Ports `llama_memory_hybrid_idx_context`'s `kpool_access`
/// (llama-memory-hybrid-idx.h:138 @d81235049): the indexer storage of one
/// layer seen two ways -- the per-token part of each row (glm5-next: key |
/// gate, qwen4exp: key) and the pooled key after it.
pub const KpoolAccess = struct {
    ctx: *c.ggml_context,
    key_gate: *Tensor,
    pooled: *Tensor,

    /// Ports the constructor of `kpool_access`
    /// (llama-memory-hybrid-idx.cpp:617 @d81235049).
    fn init(ctx: *c.ggml_context, k: *Tensor, n_embd: i64) KpoolAccess {
        const n_tok = k.ne[0] - n_embd;
        impl.assert(n_tok > 0 and @rem(n_tok, n_embd) == 0, "llama-memory-hybrid-idx.cpp", 620, "n_tok > 0 && n_tok % n_embd == 0");

        const n_cells = k.ne[1] * k.ne[2];

        // Pool indices can refer to other streams, so these are views of
        // the whole storage.
        return .{
            .ctx = ctx,
            .key_gate = c.ggml_view_2d(ctx, k, n_tok, n_cells, k.nb[1], 0),
            .pooled = c.ggml_view_2d(ctx, k, n_embd, n_cells, k.nb[1], c.ggml_row_size(k.type, n_tok)),
        };
    }

    /// Ports `kpool_access`'s `gather_key_gate` (llama-memory-hybrid-idx.cpp:630 @d81235049).
    pub fn gatherKeyGate(self: KpoolAccess, idxs: *Tensor) *Tensor {
        return c.ggml_get_rows(self.ctx, self.key_gate, idxs);
    }

    /// Ports `kpool_access`'s `scatter_pooled` (llama-memory-hybrid-idx.cpp:634 @d81235049).
    pub fn scatterPooled(self: KpoolAccess, values: *Tensor, idxs: *Tensor) *Tensor {
        return c.ggml_set_rows(self.ctx, self.pooled, values, idxs);
    }

    /// Ports `kpool_access`'s `gather_pooled` (llama-memory-hybrid-idx.cpp:638 @d81235049).
    pub fn gatherPooled(self: KpoolAccess, idxs: *Tensor) *Tensor {
        return c.ggml_get_rows(self.ctx, self.pooled, idxs);
    }
};

/// Ports `llama_memory_hybrid_idx_context` (llama-memory-hybrid-idx.h:136
/// @d81235049).
pub const HybridIdxContext = struct {
    /// The `llama_memory_hybrid_idx_context`'s base. The `memory.Context`
    /// points here.
    base: HybridContext,
    mem: ?*MemoryHybridIdx = null,
    /// Streams per ubatch, read from the slot infos before `ctx_idx` takes
    /// them.
    ns_ubatch: std.ArrayList(u32) = .empty,
    /// Each ubatch's indexer cells, kept for pools in cache order
    /// (qwen4exp): token `s*n + i` of ubatch `u` sits in cell `idxs[s][i]`
    /// of stream `strm[s]` of `sinfos_kpool[u]`, and several cells can
    /// share a position.
    sinfos_kpool: std.ArrayList(SlotInfo) = .empty,
    /// Null unless the model has an indexer.
    ctx_idx: ?*kv_cache.KvCacheContext = null,
    /// The base's ubatch cursor, private there.
    i_cur: usize = 0,

    /// Which pools this ubatch must re-pool; null until built.
    kpool_st: ?*KpoolState = null,
    /// The ubatch `kpool_st` was built for, against reads before `apply`.
    i_kpool: usize = std.math.maxInt(usize),

    /// The positions each sequence must re-pool from, cleared only after
    /// the first ubatch succeeds.
    mem_idx_stale_batch: StalePos = stalePosClean(),

    /// Ports the status constructor of `llama_memory_hybrid_idx_context`
    /// (llama-memory-hybrid-idx.cpp:513 @d81235049).
    pub fn createStatus(status: memory.Status) *HybridIdxContext {
        const self = allocator.create(HybridIdxContext) catch oom();
        self.* = .{ .base = HybridContext.initStatus(status) };
        return self;
    }

    /// Ports the full-memory constructor of `llama_memory_hybrid_idx_context`
    /// (llama-memory-hybrid-idx.cpp:516 @d81235049). Graph reservation walks
    /// a full context, and qwen4exp builds the sparse attention only when
    /// the indexer context is set: without it the reserved worst case would
    /// be the dense graph. The pool counts are sized for every pool the
    /// cache can hold.
    pub fn createFull(mem: *MemoryHybridIdx) *HybridIdxContext {
        const self = allocator.create(HybridIdxContext) catch oom();
        self.* = .{ .base = HybridContext.initFull(mem.base), .mem = mem };
        if (mem.getMemIdx()) |m| {
            self.ns_ubatch.append(allocator, m.getNStream()) catch oom();
            self.ctx_idx = kv_cache.KvCacheContext.createFull(m);
        }
        if (self.kpoolTrack()) {
            _ = mem.kpoolLayoutUpdate();
            var st = self.kpoolBuildSizes();
            const idx = mem.getMemIdx().?;
            const n_pool_max: u64 = @as(u64, idx.getSize() / mem.getKpool()) * idx.getNSeqMax();
            impl.assert(n_pool_max <= std.math.maxInt(u32) - 64, "llama-memory-hybrid-idx.cpp", 530, "n_pool_max <= UINT32_MAX - 64");
            st.n_pool_real = @max(st.n_pool_real, @as(u32, @intCast(n_pool_max)));
            st.n_new = st.n_pool_real;
            st.n_new_g = @max(st.n_new, 1);
            const p = allocator.create(KpoolState) catch oom();
            p.* = st;
            self.kpool_st = p;
            self.i_kpool = 0;
        }
        return self;
    }

    /// Ports the update constructor of `llama_memory_hybrid_idx_context`
    /// (llama-memory-hybrid-idx.cpp:539 @d81235049): `update` applies a
    /// pending cross-stream `seq_cp`, without which the copy keeps stale
    /// indexer keys.
    pub fn createUpdate(mem: *MemoryHybridIdx, lctx: *anyopaque, optimize: bool) *HybridIdxContext {
        const self = allocator.create(HybridIdxContext) catch oom();
        self.* = .{ .base = HybridContext.initUpdate(mem.base, lctx, optimize), .mem = mem };
        if (mem.getMemIdx()) |m| self.ctx_idx = m.initUpdate(lctx, optimize);
        return self;
    }

    /// Ports the batch constructor of `llama_memory_hybrid_idx_context`
    /// (llama-memory-hybrid-idx.cpp:549 @d81235049): takes the three lists;
    /// the base and the indexer each get their own copy of the ubatches,
    /// and a sequence edit since the last batch forces its touched
    /// positions to re-pool.
    pub fn createBatch(mem: *MemoryHybridIdx, sinfos_attn: std.ArrayList(SlotInfo), sinfos_idx: std.ArrayList(SlotInfo), ubatches: std.ArrayList(Ubatch)) *HybridIdxContext {
        const self = allocator.create(HybridIdxContext) catch oom();
        const ns = streamsPerUbatch(sinfos_idx.items);
        const sinfos_kpool: std.ArrayList(SlotInfo) = if (mem.getMemIdx() != null and mem.getKpool() > 0 and mem.getKpoolByOrder()) cloneSinfos(sinfos_idx.items) else .empty;
        var ctx_idx: ?*kv_cache.KvCacheContext = null;
        if (mem.getMemIdx()) |m| {
            ctx_idx = kv_cache.KvCacheContext.createBatch(m, sinfos_idx, memory_hybrid.retainedCopy(ubatches.items));
        } else {
            var unused = sinfos_idx;
            deinitSinfos(&unused);
        }
        self.* = .{
            .base = HybridContext.initBatch(mem.base, sinfos_attn, ubatches),
            .mem = mem,
            .ns_ubatch = ns,
            .sinfos_kpool = sinfos_kpool,
            .ctx_idx = ctx_idx,
            .mem_idx_stale_batch = mem.memIdxStaleGet().*,
        };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *HybridIdxContext) void {
        if (self.ctx_idx) |x| x.destroy();
        if (self.kpool_st) |st| {
            st.deinit();
            allocator.destroy(st);
        }
        deinitSinfos(&self.sinfos_kpool);
        self.ns_ubatch.deinit(allocator);
        self.base.deinit();
        allocator.destroy(self);
    }

    /// The extension behind a hybrid context, the C++'s
    /// `static_cast<const llama_memory_hybrid_idx_context *>`: valid only
    /// for a context `MemoryHybridIdx` made, as qwen4exp's and glm5-next's
    /// always are.
    pub fn fromBase(base: *const HybridContext) *const HybridIdxContext {
        return @alignCast(@fieldParentPtr("base", base));
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `next`
    /// (llama-memory-hybrid-idx.cpp:567 @d81235049): the staleness is
    /// cleared only after a successful first ubatch.
    pub fn next(self: *HybridIdxContext) bool {
        if (self.i_cur == 0) {
            if (self.mem) |m| m.memIdxStaleClear();
        }
        if (self.ctx_idx) |x| _ = x.next();
        self.i_cur += 1;
        return self.base.next();
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `apply`
    /// (llama-memory-hybrid-idx.cpp:582 @d81235049): `&`, so the indexer
    /// applies whatever the base answered; then the pool layout takes this
    /// ubatch's cells, and the state what it must re-pool.
    pub fn apply(self: *HybridIdxContext) bool {
        var res = self.base.apply();
        if (self.ctx_idx) |x| res = @intFromBool(res) & @intFromBool(x.apply()) != 0;

        if (res and self.kpoolTrack()) {
            _ = self.mem.?.kpoolLayoutUpdate();
            if (self.kpool_st == null) {
                const p = allocator.create(KpoolState) catch oom();
                p.* = .{};
                self.kpool_st = p;
            }
            self.kpoolBuildState(self.getUbatch());
            self.i_kpool = self.i_cur;
        }

        return res;
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `kpool_track`
    /// (llama-memory-hybrid-idx.cpp:602 @d81235049): whether this context
    /// tracks k-pool states, derived from the memory rather than cached.
    fn kpoolTrack(self: *const HybridIdxContext) bool {
        const m = self.mem orelse return false;
        return m.getMemIdx() != null and m.getKpool() > 0 and self.ns_ubatch.items.len != 0;
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
    /// (llama-memory-hybrid-idx.cpp:607 @d81235049): null with no indexer.
    pub fn getIdx(self: *const HybridIdxContext) ?*const kv_cache.KvCacheContext {
        return self.ctx_idx;
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `get_n_stream`
    /// (llama-memory-hybrid-idx.cpp:611 @d81235049): the streams in the
    /// current slot info, `get_k`/`get_v`'s `ns`; 1 when unified.
    pub fn getNStream(self: *const HybridIdxContext) u32 {
        impl.assert(self.i_cur < self.ns_ubatch.items.len, "llama-memory-hybrid-idx.cpp", 612, "i_cur < ns_ubatch.size()");
        return self.ns_ubatch.items[self.i_cur];
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `get_kpool_access`
    /// (llama-memory-hybrid-idx.cpp:642 @d81235049).
    pub fn getKpoolAccess(self: *const HybridIdxContext, ctx: *c.ggml_context, il: i32, n_embd: i64) KpoolAccess {
        impl.assert(self.mem != null and self.mem.?.getMemIdx() != null, "llama-memory-hybrid-idx.cpp", 644, "mem != nullptr && mem->get_mem_idx() != nullptr");
        return KpoolAccess.init(ctx, self.mem.?.getMemIdx().?.getKStorage(il), n_embd);
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `gather_mla_rows`
    /// (llama-memory-hybrid-idx.cpp:649 @d81235049): rows of the attention
    /// cache's K storage of layer `il`, across every stream.
    pub fn gatherMlaRows(self: *const HybridIdxContext, ctx: *c.ggml_context, idxs: *Tensor, n_rows: i64, n_embd: i64, il: i32) *Tensor {
        impl.assert(self.mem != null, "llama-memory-hybrid-idx.cpp", 651, "mem != nullptr");
        const k = self.mem.?.getMemAttn().getKStorage(il);
        impl.assert(k.ne[0] == n_embd, "llama-memory-hybrid-idx.cpp", 653, "k->ne[0] == n_embd");

        const rows = c.ggml_view_2d(ctx, k, k.ne[0], k.ne[1] * k.ne[2], k.nb[1], 0);
        return c.ggml_get_rows(ctx, rows, c.ggml_reshape_1d(ctx, idxs, n_rows));
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `kpool_build_sizes`
    /// (llama-memory-hybrid-idx.cpp:662 @d81235049): the sizes alone, for
    /// the full context, so `getNKpool` works during graph reserve.
    fn kpoolBuildSizes(self: *const HybridIdxContext) KpoolState {
        const lay = self.mem.?.kpoolLayoutGet();
        return .{ .n_pool_real = lay.n_pool_real };
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `kpool_build_state`
    /// (llama-memory-hybrid-idx.cpp:680 @d81235049): which pools this ubatch
    /// must re-pool. `cpy_k` writes each token's key | gate into its row,
    /// pooled slots zeroed; this marks the pools the ubatch touches or
    /// completes as new -- one pool every `kpool` tokens during decode; the
    /// graph pools only those and scatters each result into the pooled
    /// slot of its pool's last member, then gathers every pool through
    /// `pool_cells`. A sequence edit stales the pools from the edited
    /// position on, and the next batch's first ubatch rebuilds them from
    /// the key | gate rows, which are still valid.
    fn kpoolBuildState(self: *HybridIdxContext, ubatch: *const Ubatch) void {
        const mem = self.mem.?;
        const lay = mem.kpoolLayoutGet();
        const st = self.kpool_st.?;

        st.n_pool_real = lay.n_pool_real;
        st.cache_safe = lay.cache_safe;
        st.n_new = 0;
        st.generation +%= 1;
        if (st.generation == 0) {
            @memset(st.is_new.items, 0);
            st.generation = 1;
        }
        if (st.is_new.items.len < lay.n_pool_real) {
            st.is_new.appendNTimes(allocator, 0, lay.n_pool_real - st.is_new.items.len) catch oom();
        } else {
            st.is_new.shrinkRetainingCapacity(lay.n_pool_real);
        }

        const Mark = struct {
            fn mark(s: *KpoolState, ip: u32) void {
                if (s.is_new.items[ip] != s.generation) {
                    s.is_new.items[ip] = s.generation;
                    s.n_new += 1;
                }
            }
        };

        const kpool = mem.getKpool();
        var pool_start: [max_seq]u32 = undefined;
        var ip: u32 = 0;
        for (0..max_seq) |su| {
            const sq = &lay.seqs[su];
            pool_start[su] = ip;
            ip += @intCast(sq.pools.items.len);

            if (!st.cache_safe) continue;

            // A sequence edit stales only the pools ending after it.
            const stale_from: i32 = if (self.i_cur == 0) self.mem_idx_stale_batch[su] else pos_clean;
            if (stale_from == pos_clean) continue;

            // `lower_bound`: the first pool whose last member is not before it
            var lo: usize = 0;
            var hi: usize = sq.pools.items.len;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (sq.cells.items[sq.pools.items[mid] + kpool - 1].pos < stale_from) lo = mid + 1 else hi = mid;
            }
            for (lo..sq.pools.items.len) |pi| Mark.mark(st, pool_start[su] + @as(u32, @intCast(pi)));
        }
        impl.assert(ip == st.is_new.items.len, "llama-memory-hybrid-idx.cpp", 725, "ip == st.is_new.size()");

        if (!st.cache_safe) {
            @memset(st.is_new.items, st.generation);
            st.n_new = st.n_pool_real;
            st.n_new_g = @max(st.n_new, 1);
            return;
        }

        // In order mode a token's cell gives its rank, and the rank its
        // pool: positions cannot, as an image shares one.
        const by_order = mem.getKpoolByOrder();
        const sinfo: ?*const SlotInfo = if (by_order) &self.sinfos_kpool.items[self.i_cur] else null;
        const n_tps: u32 = if (sinfo) |si| @intCast(si.size()) else 0;

        for (0..ubatch.n_tokens) |i| {
            const p = ubatch.pos[i];
            var k: usize = 0;
            while (k < ubatch.n_seq_id[i]) : (k += 1) {
                const s: usize = @intCast(ubatch.seq_id[i][k]);
                const sq = &lay.seqs[s];
                if (by_order) {
                    const r = kpoolRank(sq.cells.items, p, sinfo.?.idxs.items[i / n_tps].items[i % n_tps]);
                    impl.assert(r >= 0, "llama-memory-hybrid-idx.cpp", 746, "r >= 0");
                    const pool: usize = @as(usize, @intCast(r)) / kpool;
                    if (pool < sq.pools.items.len) Mark.mark(st, pool_start[s] + @as(u32, @intCast(pool)));
                    continue;
                }
                // `upper_bound`: the first pool starting after p
                var lo: usize = 0;
                var hi: usize = sq.pools.items.len;
                while (lo < hi) {
                    const mid = lo + (hi - lo) / 2;
                    if (!(p < sq.cells.items[sq.pools.items[mid]].pos)) lo = mid + 1 else hi = mid;
                }
                if (lo == 0) continue;
                const it = lo - 1;
                if (p <= sq.cells.items[sq.pools.items[it] + kpool - 1].pos) {
                    Mark.mark(st, pool_start[s] + @as(u32, @intCast(it)));
                }
            }
        }

        // A ubatch touches at most t_s/kpool + 1 pools per sequence: pad to
        // that bound so the graph keeps its shape as the count moves, but
        // never past n_pool_max, the most the reserve sized for.
        const idx = mem.getMemIdx().?;
        const n_pool_max: u32 = idx.getSize() / kpool *% idx.getNSeqMax();
        const bound: u32 = ubatch.n_tokens / kpool +% ubatch.n_seqs_unq;
        st.n_new_g = @max(@max(st.n_new, 1), @min(@min(bound, kpoolPad(st.n_pool_real) - 1), n_pool_max));
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `kpool_cur`
    /// (llama-memory-hybrid-idx.cpp:772 @d81235049).
    fn kpoolCur(self: *const HybridIdxContext) *const KpoolState {
        impl.assert(self.kpool_st != null and self.i_kpool == self.i_cur, "llama-memory-hybrid-idx.cpp", 773, "kpool_st != nullptr && i_kpool == i_cur && \"k-pool state read before apply()\"");
        return self.kpool_st.?;
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `get_n_kpool`
    /// (llama-memory-hybrid-idx.cpp:778 @d81235049): the padded pool count,
    /// whose last pool is always unused.
    pub fn getNKpool(self: *const HybridIdxContext) u32 {
        return kpoolPad(self.kpoolCur().n_pool_real);
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `get_n_kpool_new`
    /// (llama-memory-hybrid-idx.cpp:782 @d81235049): the pools to re-pool
    /// this ubatch, padded to a stable bound and never below 1.
    pub fn getNKpoolNew(self: *const HybridIdxContext) u32 {
        return self.kpoolCur().n_new_g;
    }

    /// Ports `llama_memory_hybrid_idx_context`'s `set_input_kpool`
    /// (llama-memory-hybrid-idx.cpp:786 @d81235049): the k-pool indexer's
    /// inputs for the current ubatch.
    ///
    /// - `pool_cells` I32 `[n_pool]`: each pool's rep cell, where its
    ///   pooled key lives
    /// - `pool_idxs` I32 `[kpool, n_pool]`: each pool's member cells
    /// - `pool_mask` F32/F16 `[n_pool, n_tokens]`: 0 where a pool is visible
    ///   to a token -- of its sequence and ending at or before it
    /// - `tail_idxs` I32 `[kpool - 1, n_tokens]`: the cells of each token's
    ///   incomplete trailing pool
    /// - `gather_mask` F32 `[n_sel, 1, 1, n_tokens]`, optional: the
    ///   selection's mask
    /// - `gather`: indices into every stream's storage, padding mapped to a
    ///   real cell and masked; otherwise stream-local, padding at `n_kv`
    /// - `new_pool_idxs` I32 `[kpool, n_new_g]`, `new_pool_rep` I64
    ///   `[n_new_g]`: the pools to re-pool and the rows to scatter them to
    /// - `new_pool_pos` I32 `[4*n_new_g]`, optional: each new pool's first
    ///   member's M-RoPE position, for keys rotated at pooling time
    pub fn setInputKpool(
        self: *const HybridIdxContext,
        pool_cells: *Tensor,
        pool_idxs: *Tensor,
        pool_mask: *Tensor,
        tail_idxs: *Tensor,
        gather_mask: ?*Tensor,
        gather: bool,
        new_pool_idxs: *Tensor,
        new_pool_rep: ?*Tensor,
        ubatch: *const Ubatch,
        new_pool_pos: ?*Tensor,
    ) void {
        const f = "llama-memory-hybrid-idx.cpp";
        impl.assert(self.mem != null and self.mem.?.getMemIdx() != null, f, 789, "mem != nullptr && mem->get_mem_idx() != nullptr");
        impl.assert(c.ggml_backend_buffer_is_host(pool_cells.buffer), f, 790, "ggml_backend_buffer_is_host(pool_cells->buffer)");
        impl.assert(c.ggml_backend_buffer_is_host(pool_idxs.buffer), f, 791, "ggml_backend_buffer_is_host(pool_idxs->buffer)");
        impl.assert(c.ggml_backend_buffer_is_host(pool_mask.buffer), f, 792, "ggml_backend_buffer_is_host(pool_mask->buffer)");
        impl.assert(c.ggml_backend_buffer_is_host(tail_idxs.buffer), f, 793, "ggml_backend_buffer_is_host(tail_idxs->buffer)");

        const mem = self.mem.?;
        const m_idx = mem.getMemIdx().?;
        const kpool = mem.getKpool();
        const n_kv = self.getIdx().?.getNKv();

        const st = self.kpoolCur();
        const lay = mem.kpoolLayoutGet();

        const n_tokens: u32 = ubatch.n_tokens;
        const n_pool: u32 = @intCast(pool_cells.ne[0]);
        const n_new = st.n_new;
        // the graph always pools at least one entry, padded to a stable
        // bound (see `kpoolBuildState`)
        const n_new_g = st.n_new_g;

        const by_order = mem.getKpoolByOrder();

        impl.assert(n_pool == kpoolPad(st.n_pool_real), f, 809, "n_pool == kpool_pad(st.n_pool_real)");
        impl.assert(st.is_new.items.len == st.n_pool_real, f, 810, "st.is_new.size() == st.n_pool_real");
        impl.assert(pool_mask.ne[0] == n_pool and pool_mask.ne[1] == n_tokens, f, 811, "pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens");
        impl.assert(tail_idxs.ne[0] == @as(i64, kpool) - 1 and tail_idxs.ne[1] == n_tokens, f, 812, "tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens");
        impl.assert(pool_idxs.ne[0] == kpool and pool_idxs.ne[1] == n_pool, f, 813, "pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool");
        impl.assert(c.ggml_backend_buffer_is_host(new_pool_idxs.buffer), f, 814, "ggml_backend_buffer_is_host(new_pool_idxs->buffer)");
        impl.assert(new_pool_idxs.ne[0] == kpool and new_pool_idxs.ne[1] == n_new_g, f, 815, "new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g");
        // the graph always scatters the fresh pooled keys back into the cache
        impl.assert(new_pool_rep != null and c.ggml_backend_buffer_is_host(new_pool_rep.?.buffer), f, 817, "new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer)");
        impl.assert(new_pool_rep.?.ne[0] == n_new_g, f, 818, "new_pool_rep->ne[0] == (int64_t) n_new_g");
        if (new_pool_pos) |npp| {
            impl.assert(c.ggml_backend_buffer_is_host(npp.buffer), f, 820, "ggml_backend_buffer_is_host(new_pool_pos->buffer)");
            impl.assert(npp.ne[0] == 4 * @as(i64, n_new_g), f, 821, "new_pool_pos->ne[0] == 4*(int64_t) n_new_g");
        }

        const kv_size: u32 = m_idx.getSize();
        const n_stream_kv: u32 = m_idx.getNStream();

        const Gcell = struct {
            fn of(sq: *const KpoolLayout.Seq, size: u32, cell: u32) i64 {
                return @as(i64, sq.strm) * size + cell;
            }
        };

        // The sequences of this ubatch: an absent one's pools must fall on
        // the scatter's sentinel row.
        var seq_in_ub: [max_seq]bool = @splat(false);
        for (0..n_tokens) |i| {
            var k: usize = 0;
            while (k < ubatch.n_seq_id[i]) : (k += 1) seq_in_ub[@intCast(ubatch.seq_id[i][k])] = true;
        }

        // The first ubatch cell, for padded gathers.
        var dummy_cell: i64 = 0;
        {
            const s: usize = @intCast(ubatch.seq_id[0][0]);
            const sq = &lay.seqs[s];
            const it = lowerBoundPosCell(sq.cells.items, .{ .pos = ubatch.pos[0], .cell = 0 });
            impl.assert(it != sq.cells.items.len and sq.cells.items[it].pos == ubatch.pos[0], f, 845, "it != sq.cells.end() && it->first == ubatch->pos[0]");
            dummy_cell = Gcell.of(sq, kv_size, sq.cells.items[it].cell);
        }

        // In order mode a token sees the pools and the tail up to its own
        // rank in the sequence, which its cell pins down.
        var rank: std.ArrayList(i64) = .empty;
        defer rank.deinit(allocator);
        if (by_order) {
            const sinfo = &self.sinfos_kpool.items[self.i_cur];
            const n_tps: u32 = @intCast(sinfo.size());

            rank.resize(allocator, n_tokens) catch oom();
            for (0..n_tokens) |i| {
                rank.items[i] = kpoolRank(lay.seqs[@intCast(ubatch.seq_id[i][0])].cells.items, ubatch.pos[i], sinfo.idxs.items[i / n_tps].items[i % n_tps]);
                impl.assert(rank.items[i] >= 0, f, 858, "rank[i] >= 0");
            }
        }

        // A gather maps padding to a real cell and masks it separately.
        const sentinel: i32 = if (gather) @intCast(dummy_cell) else @bitCast(n_kv);

        var gm: ?[*]f32 = null;
        var n_sel: u32 = 0;
        // Pools per token in the selection.
        var n_top: u32 = 0;
        if (gather_mask) |g| {
            impl.assert(c.ggml_backend_buffer_is_host(g.buffer), f, 869, "ggml_backend_buffer_is_host(gather_mask->buffer)");
            impl.assert(g.type == c.GGML_TYPE_F32, f, 870, "gather_mask->type == GGML_TYPE_F32");
            impl.assert(g.ne[3] == n_tokens and g.ne[1] == 1 and g.ne[2] == 1, f, 871, "gather_mask->ne[3] == (int64_t) n_tokens && gather_mask->ne[1] == 1 && gather_mask->ne[2] == 1");
            n_sel = @intCast(g.ne[0]);
            // the tail slots, when selected, are the n_sel % kpool != 0 remainder
            n_top = n_sel / kpool;
            impl.assert(n_sel % kpool == 0 or n_sel % kpool == kpool - 1, f, 875, "n_sel % kpool == 0 || n_sel % kpool == kpool - 1");
            gm = @ptrCast(@alignCast(g.data));
        }

        // pools are laid out per sequence
        var seq_pool_start: [max_seq]u32 = @splat(0);
        var pool_end: std.ArrayList(i32) = .empty;
        defer pool_end.deinit(allocator);
        pool_end.ensureTotalCapacity(allocator, n_pool) catch oom();

        const pcell: [*]i32 = @ptrCast(@alignCast(pool_cells.data));
        const pidx: [*]i32 = @ptrCast(@alignCast(pool_idxs.data));
        const nidx: [*]i32 = @ptrCast(@alignCast(new_pool_idxs.data));
        const nrep: [*]i64 = @ptrCast(@alignCast(new_pool_rep.?.data));
        const npos: ?[*]i32 = if (new_pool_pos) |npp| @ptrCast(@alignCast(npp.data)) else null;

        if (npos) |np| @memset(np[0 .. 4 * @as(usize, n_new_g)], 0);

        var i_new: u32 = 0;
        for (0..max_seq) |su| {
            const sq = &lay.seqs[su];
            seq_pool_start[su] = @intCast(pool_end.items.len);

            const inert = !gather and n_stream_kv > 1 and !seq_in_ub[su];

            for (sq.pools.items) |j| {
                const ipu: u32 = @intCast(pool_end.items.len);
                impl.assert(ipu + 1 < n_pool, f, 904, "ip + 1 < n_pool");

                // The pooled key lives in the last member's row.
                const rep = sq.cells.items[j + kpool - 1].cell;
                pcell[ipu] = @intCast(Gcell.of(sq, kv_size, rep));

                for (0..kpool) |k| {
                    const cell = sq.cells.items[j + k].cell;
                    pidx[@as(usize, ipu) * kpool + k] = if (inert) sentinel else @intCast(if (gather) Gcell.of(sq, kv_size, cell) else @as(i64, cell));
                }

                if (st.is_new.items[ipu] == st.generation) {
                    impl.assert(i_new < n_new, f, 916, "i_new < n_new");
                    for (0..kpool) |k| {
                        nidx[@as(usize, i_new) * kpool + k] = @intCast(Gcell.of(sq, kv_size, sq.cells.items[j + k].cell));
                    }
                    nrep[i_new] = Gcell.of(sq, kv_size, rep);
                    if (npos) |np| {
                        // a pooled key is rotated to its first member's M-RoPE position
                        const cl = sq.cells.items[j].cell;
                        const e = m_idx.getCells(@intCast(su)).extGet(cl);
                        np[0 * @as(usize, n_new_g) + i_new] = sq.cells.items[j].pos;
                        np[1 * @as(usize, n_new_g) + i_new] = e.y;
                        np[2 * @as(usize, n_new_g) + i_new] = e.x;
                        np[3 * @as(usize, n_new_g) + i_new] = sq.cells.items[j].pos;
                    }
                    i_new += 1;
                }

                pool_end.appendAssumeCapacity(sq.cells.items[j + kpool - 1].pos);
            }
        }
        impl.assert(i_new == n_new, f, 936, "i_new == n_new");

        // Padded entries re-pool cells whose pooled slot is never read (only
        // complete pools' reps are), each its own cell: entries sharing one
        // would write it from several threads in the scatter.
        if (n_new_g > n_new) {
            const reps = allocator.alloc(i64, pool_end.items.len) catch oom();
            defer allocator.free(reps);
            for (reps, 0..) |*r, k| r.* = pcell[k];
            std.sort.pdq(i64, reps, {}, std.sort.asc(i64));

            var pad_cell: i64 = 0;
            var i = n_new;
            while (i < n_new_g) : ({
                i += 1;
                pad_cell += 1;
            }) {
                while (std.sort.binarySearch(i64, reps, pad_cell, orderI64) != null) pad_cell += 1;
                impl.assert(pad_cell < @as(i64, kv_size) * n_stream_kv, f, 949, "pad_cell < (int64_t) kv_size*n_stream_kv");
                for (0..kpool) |k| nidx[@as(usize, i) * kpool + k] = @intCast(pad_cell);
                nrep[i] = pad_cell;
            }
        }

        const n_pool_real: u32 = @intCast(pool_end.items.len);
        {
            var ipu = n_pool_real;
            while (ipu < n_pool) : (ipu += 1) {
                // pool_cells always addresses the K storage
                pcell[ipu] = @intCast(dummy_cell);
                for (0..kpool) |k| pidx[@as(usize, ipu) * kpool + k] = sentinel;
            }
        }

        // A pool is visible when it belongs to the token's sequence and
        // ends at or before it.
        const Fill = struct {
            fn run(comptime T: type, data: [*]T, ctx: anytype) void {
                const keep: T = if (T == u16) c.ggml_fp32_to_fp16(0.0) else 0.0;
                const drop: T = if (T == u16) c.ggml_fp32_to_fp16(-std.math.inf(f32)) else -std.math.inf(f32);

                for (0..ctx.n_tokens) |i| {
                    const s: usize = @intCast(ctx.ubatch.seq_id[i][0]);
                    const p = ctx.ubatch.pos[i];

                    const row = data + i * ctx.n_pool;
                    @memset(row[0..ctx.n_pool], drop);

                    const p0 = ctx.seq_pool_start[s];
                    const p1 = p0 + @as(u32, @intCast(ctx.lay.seqs[s].pools.items.len));
                    var nv: u32 = undefined;
                    if (ctx.by_order) {
                        nv = @min(p1 - p0, @as(u32, @intCast(@divTrunc(ctx.rank[i] + 1, ctx.kpool))));
                    } else {
                        // `upper_bound` of p over pool_end[p0, p1)
                        var lo: usize = p0;
                        var hi: usize = p1;
                        while (lo < hi) {
                            const mid = lo + (hi - lo) / 2;
                            if (!(p < ctx.pool_end[mid])) lo = mid + 1 else hi = mid;
                        }
                        nv = @intCast(lo - p0);
                    }
                    @memset(row[p0 .. p0 + nv], keep);

                    // The finite visible pools take the first min(nv, n_top)
                    // ranked slots.
                    if (ctx.gm) |g| {
                        const nvc = @min(nv, ctx.n_top);
                        const grow = g + i * ctx.n_sel;
                        @memset(grow[0 .. @as(usize, nvc) * ctx.kpool_u], 0.0);
                        @memset(grow[@as(usize, nvc) * ctx.kpool_u .. @as(usize, ctx.n_top) * ctx.kpool_u], -std.math.inf(f32));
                    }
                }
            }
        };
        const fill_ctx = .{
            .n_tokens = @as(usize, n_tokens),
            .ubatch = ubatch,
            .n_pool = @as(usize, n_pool),
            .seq_pool_start = &seq_pool_start,
            .lay = lay,
            .by_order = by_order,
            .rank = rank.items,
            .kpool = @as(i64, kpool),
            .kpool_u = @as(usize, kpool),
            .pool_end = pool_end.items,
            .gm = gm,
            .n_top = n_top,
            .n_sel = @as(usize, n_sel),
        };
        if (pool_mask.type == c.GGML_TYPE_F16) {
            Fill.run(u16, @ptrCast(@alignCast(pool_mask.data)), fill_ctx);
        } else {
            Fill.run(f32, @ptrCast(@alignCast(pool_mask.data)), fill_ctx);
        }

        const tidx: [*]i32 = @ptrCast(@alignCast(tail_idxs.data));
        for (0..n_tokens) |i| {
            const s: usize = @intCast(ubatch.seq_id[i][0]);
            const p = ubatch.pos[i];
            const sq = &lay.seqs[s];

            const n_tail: u32 = if (by_order)
                @intCast(@rem(rank.items[i] + 1, @as(i64, kpool)))
            else
                @bitCast(@rem(p -% sq.pos_min +% 1, @as(i32, @bitCast(kpool))));

            for (0..kpool - 1) |k| {
                var cell: i32 = sentinel;
                var real = false;
                if (k < n_tail and by_order) {
                    const cl = sq.cells.items[@intCast(rank.items[i] - @as(i64, @intCast(k)))].cell;
                    cell = @intCast(if (gather) Gcell.of(sq, kv_size, cl) else @as(i64, cl));
                    real = true;
                } else if (k < n_tail) {
                    const pt = p -% @as(i32, @intCast(k));
                    const it = lowerBoundPosCell(sq.cells.items, .{ .pos = pt, .cell = 0 });
                    if (it != sq.cells.items.len and sq.cells.items[it].pos == pt) {
                        const cl = sq.cells.items[it].cell;
                        cell = @intCast(if (gather) Gcell.of(sq, kv_size, cl) else @as(i64, cl));
                        real = true;
                    }
                }
                tidx[i * (kpool - 1) + k] = cell;

                if (gm) |g| {
                    if (n_sel % kpool != 0) {
                        g[i * n_sel + @as(usize, n_top) * kpool + k] = if (real) 0.0 else -std.math.inf(f32);
                    }
                }
            }
        }
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

fn orderI64(a: i64, b: i64) std.math.Order {
    return std.math.order(a, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(MemoryHybridIdx);
    std.testing.refAllDecls(HybridIdxContext);
    std.testing.refAllDecls(KpoolAccess);
}

test "a context's base pointer recovers the extension" {
    const ctx = HybridIdxContext.createStatus(.failed_prepare);
    const mc = ctx.contextOf();
    try std.testing.expect(mc.cast(HybridContext) == &ctx.base);
    try std.testing.expect(HybridIdxContext.fromBase(mc.cast(HybridContext)) == ctx);
    try std.testing.expectEqual(memory.Status.failed_prepare, mc.getStatus());
    mc.destroy();
}

test "a pool count pads to 64 past one spare pool" {
    try std.testing.expectEqual(@as(u32, 64), kpoolPad(0));
    try std.testing.expectEqual(@as(u32, 64), kpoolPad(63));
    try std.testing.expectEqual(@as(u32, 128), kpoolPad(64));
}

test "a rank is the index of the exact pair, or -1" {
    const cells = [_]PosCell{ .{ .pos = 0, .cell = 3 }, .{ .pos = 1, .cell = 1 }, .{ .pos = 1, .cell = 4 } };
    try std.testing.expectEqual(@as(i64, 2), kpoolRank(&cells, 1, 4));
    try std.testing.expectEqual(@as(i64, -1), kpoolRank(&cells, 1, 2));
}
