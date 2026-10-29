//! The MiniMax Sparse Attention (MSA) KV cache: a `llama_kv_cache` for the
//! K/V tensors and a second one for the MSA indexer's keys, driven as one.
//!
//! Both receive identical sequence operations and identical ubatches, so
//! their cell layouts stay in sync. The per-batch context also exposes
//! position-to-cell translation maps, populated from the base cache's
//! cells, which the model graph uses to run MSA block selection in
//! position space.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-kv-cache-msa.h` -- `llama_kv_cache_msa` and
//!   `llama_kv_cache_msa_context`
//! - `llama.cpp/src/llama-kv-cache-msa.cpp` -- their bodies
//!
//! All at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("impl.zig");
const hparams_mod = @import("hparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const io = @import("io.zig");
const kv_cache = @import("kv_cache.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;
const Ubatch = batch.Ubatch;
const KvCache = kv_cache.KvCache;
const KvCacheContext = kv_cache.KvCacheContext;
const SlotInfo = kv_cache.SlotInfo;
const Tensor = c.ggml_tensor;

fn oom() noreturn {
    impl.abort("llama-kv-cache-msa.cpp", 0, "out of memory");
}

/// Ports `llama_kv_cache_msa` (llama-kv-cache-msa.h:14 @b29c606e2): two
/// `llama_kv_cache`s, the first for K/V and the second for the indexer
/// keys.
pub const KvCacheMsa = struct {
    /// The indexer cache's hyperparameters, kept here because a
    /// `llama_kv_cache` stores only a reference to its own.
    hparams_idx: Hparams,

    n_stream: u32 = 1,
    n_seq_max: u32 = 1,
    n_pad: u32 = 1,

    n_swa: u32 = 0,
    swa_type: hparams_mod.SwaType = .none,

    kv_base: *KvCache = undefined,
    kv_idx: *KvCache = undefined,

    /// Ports the constructor of `llama_kv_cache_msa`
    /// (llama-kv-cache-msa.cpp:13 @b29c606e2).
    ///
    /// Parameters:
    /// - `filter`: the base cache's layers; empty selects every layer.
    /// - `filter_idx`: the indexer cache's layers.
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        type_k: c.enum_ggml_type,
        type_v: c.enum_ggml_type,
        v_trans: bool,
        offload: bool,
        unified: bool,
        kv_size: u32,
        n_seq_max: u32,
        n_pad: u32,
        n_swa: u32,
        swa_type: hparams_mod.SwaType,
        filter: memory.LayerFilter,
        filter_idx: memory.LayerFilter,
        reuse: memory.LayerMap,
        exc: *Exception,
    ) Thrown!*KvCacheMsa {
        const func = "llama_kv_cache_msa";

        // Allocated first: the indexer cache keeps a pointer to
        // `hparams_idx`.
        const self = allocator.create(KvCacheMsa) catch oom();
        errdefer allocator.destroy(self);
        self.* = .{
            .hparams_idx = model.hparams,
            .n_stream = if (unified) 1 else n_seq_max,
            .n_seq_max = n_seq_max,
            .n_pad = n_pad,
            .n_swa = n_swa,
            .swa_type = swa_type,
        };

        impl.logInfo("%s: creating main KV cache, size = %u cells\n", .{ func, kv_size });

        self.kv_base = try KvCache.create(model, &model.hparams, type_k, type_v, v_trans, offload, unified, kv_size, n_seq_max, n_pad, n_swa, swa_type, null, filter, reuse, .{}, exc);
        errdefer self.kv_base.destroy();

        // The MSA indexer uses a single key head per layer.
        self.hparams_idx.n_head_kv_arr = @splat(1);
        self.hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;
        // The RoPE parameters are kept identical to the main cache.

        impl.logInfo("%s: creating indexer KV cache, size = %u cells\n", .{ func, kv_size });

        self.kv_idx = try KvCache.create(model, &self.hparams_idx, type_k, type_v, v_trans, offload, unified, kv_size, n_seq_max, n_pad, n_swa, swa_type, null, filter_idx, reuse, .{}, exc);

        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *KvCacheMsa) void {
        self.kv_idx.destroy();
        self.kv_base.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_msa`'s `clear` (llama-kv-cache-msa.cpp:52 @b29c606e2).
    pub fn clear(self: *KvCacheMsa, data: bool) void {
        self.kv_base.clear(data);
        self.kv_idx.clear(data);
    }

    /// Ports `llama_kv_cache_msa`'s `seq_rm` (llama-kv-cache-msa.cpp:57
    /// @b29c606e2): both caches, whatever the first answers.
    pub fn seqRm(self: *KvCacheMsa, seq_id: i32, p0: i32, p1: i32) bool {
        var res = true;
        res = self.kv_base.seqRm(seq_id, p0, p1) and res;
        res = self.kv_idx.seqRm(seq_id, p0, p1) and res;
        return res;
    }

    /// Ports `llama_kv_cache_msa`'s `seq_cp` (llama-kv-cache-msa.cpp:66 @b29c606e2).
    pub fn seqCp(self: *KvCacheMsa, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) void {
        self.kv_base.seqCp(seq_id_src, seq_id_dst, p0, p1);
        self.kv_idx.seqCp(seq_id_src, seq_id_dst, p0, p1);
    }

    /// Ports `llama_kv_cache_msa`'s `seq_keep` (llama-kv-cache-msa.cpp:71 @b29c606e2).
    pub fn seqKeep(self: *KvCacheMsa, seq_id: i32) void {
        self.kv_base.seqKeep(seq_id);
        self.kv_idx.seqKeep(seq_id);
    }

    /// Ports `llama_kv_cache_msa`'s `seq_add` (llama-kv-cache-msa.cpp:76 @b29c606e2).
    pub fn seqAdd(self: *KvCacheMsa, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.kv_base.seqAdd(seq_id, p0, p1, shift);
        self.kv_idx.seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_kv_cache_msa`'s `seq_div` (llama-kv-cache-msa.cpp:81 @b29c606e2).
    pub fn seqDiv(self: *KvCacheMsa, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.kv_base.seqDiv(seq_id, p0, p1, d);
        self.kv_idx.seqDiv(seq_id, p0, p1, d);
    }

    /// Ports `llama_kv_cache_msa`'s `seq_pos_min` (llama-kv-cache-msa.cpp:86
    /// @b29c606e2): the base cache answers.
    pub fn seqPosMin(self: *const KvCacheMsa, seq_id: i32) i32 {
        return self.kv_base.seqPosMin(seq_id);
    }

    /// Ports `llama_kv_cache_msa`'s `seq_pos_max` (llama-kv-cache-msa.cpp:90 @b29c606e2).
    pub fn seqPosMax(self: *const KvCacheMsa, seq_id: i32) i32 {
        return self.kv_base.seqPosMax(seq_id);
    }

    /// Ports `llama_kv_cache_msa`'s `memory_breakdown` (llama-kv-cache-msa.cpp:94
    /// @b29c606e2): the indexer cache's sizes added to the base cache's.
    pub fn memoryBreakdown(self: *const KvCacheMsa, out: *std.ArrayList(memory.BuftSize)) void {
        self.kv_base.memoryBreakdown(out);
        self.kv_idx.memoryBreakdown(out);
    }

    /// Ports `llama_kv_cache_msa`'s `init_batch` (llama-kv-cache-msa.cpp:102
    /// @b29c606e2): a simple split for one stream, else an equal split; every
    /// ubatch must fit in both caches.
    pub fn initBatch(self: *KvCacheMsa, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *MsaContext {
        _ = embd_all;

        blk: {
            balloc.splitReset();

            var ubatches: std.ArrayList(Ubatch) = .empty;
            while (true) {
                const ub = if (self.n_stream == 1) balloc.splitSimple(n_ubatch) else balloc.splitEqual(n_ubatch, true, 0);
                if (ub.n_tokens == 0) break;
                ubatches.append(allocator, ub) catch oom();
            }

            if (balloc.getNUsed() < balloc.getNTokens()) {
                // Failed to find a suitable split.
                releaseAll(&ubatches);
                break :blk;
            }

            var sinfos_base = self.kv_base.prepare(ubatches.items) orelse {
                releaseAll(&ubatches);
                break :blk;
            };
            if (sinfos_base.items.len == 0) {
                sinfos_base.deinit(allocator);
                releaseAll(&ubatches);
                break :blk;
            }

            var sinfos_idx = self.kv_idx.prepare(ubatches.items) orelse {
                freeSlotInfos(&sinfos_base);
                releaseAll(&ubatches);
                break :blk;
            };
            if (sinfos_idx.items.len == 0) {
                sinfos_idx.deinit(allocator);
                freeSlotInfos(&sinfos_base);
                releaseAll(&ubatches);
                break :blk;
            }

            std.debug.assert(sinfos_base.items.len == sinfos_idx.items.len);

            return MsaContext.createBatch(self, sinfos_base, sinfos_idx, ubatches);
        }

        return MsaContext.createStatus(.failed_prepare);
    }

    /// Ports `llama_kv_cache_msa`'s `init_full` (llama-kv-cache-msa.cpp:146 @b29c606e2).
    pub fn initFull(self: *KvCacheMsa) *MsaContext {
        return MsaContext.createFull(self);
    }

    /// Ports `llama_kv_cache_msa`'s `init_update` (llama-kv-cache-msa.cpp:150 @b29c606e2).
    pub fn initUpdate(self: *KvCacheMsa, lctx: *anyopaque, optimize: bool) *MsaContext {
        return MsaContext.createUpdate(self, lctx, optimize);
    }

    /// Ports `llama_kv_cache_msa`'s `get_can_shift` (llama-kv-cache-msa.cpp:154
    /// @b29c606e2): only when both can and the two are the same size.
    pub fn getCanShift(self: *const KvCacheMsa) bool {
        return self.kv_base.getCanShift() and
            self.kv_idx.getCanShift() and
            self.kv_base.getSize() == self.kv_idx.getSize();
    }

    /// Ports `llama_kv_cache_msa`'s `state_write` (llama-kv-cache-msa.cpp:160
    /// @b29c606e2): both caches, whatever the flags.
    pub fn stateWrite(self: *const KvCacheMsa, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        try self.kv_base.stateWrite(w, seq_id, flags, exc);
        try self.kv_idx.stateWrite(w, seq_id, flags, exc);
    }

    /// Ports `llama_kv_cache_msa`'s `state_read` (llama-kv-cache-msa.cpp:165 @b29c606e2).
    pub fn stateRead(self: *KvCacheMsa, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        try self.kv_base.stateRead(r, seq_id, flags, exc);
        try self.kv_idx.stateRead(r, seq_id, flags, exc);
    }

    /// Ports `llama_kv_cache_msa`'s `get_base` (llama-kv-cache-msa.cpp:170 @b29c606e2).
    pub fn getBase(self: *const KvCacheMsa) *KvCache {
        return self.kv_base;
    }

    /// Ports `llama_kv_cache_msa`'s `get_idx` (llama-kv-cache-msa.cpp:174 @b29c606e2).
    pub fn getIdx(self: *const KvCacheMsa) *KvCache {
        return self.kv_idx;
    }

    /// Ports `llama_kv_cache_msa`'s `get_n_pad` (llama-kv-cache-msa.h:70 @b29c606e2).
    pub fn getNPad(self: *const KvCacheMsa) u32 {
        return self.n_pad;
    }

    /// Ports `llama_kv_cache_msa`'s `get_n_seq_max` (llama-kv-cache-msa.h:71 @b29c606e2).
    pub fn getNSeqMax(self: *const KvCacheMsa) u32 {
        return self.n_seq_max;
    }

    /// Ports `llama_kv_cache_msa`'s `get_n_swa` (llama-kv-cache-msa.h:72 @b29c606e2).
    pub fn getNSwa(self: *const KvCacheMsa) u32 {
        return self.n_swa;
    }

    /// Ports `llama_kv_cache_msa`'s `get_swa_type` (llama-kv-cache-msa.h:73 @b29c606e2).
    pub fn getSwaType(self: *const KvCacheMsa) hparams_mod.SwaType {
        return self.swa_type;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *KvCacheMsa) memory.Memory {
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

    fn from(p: *anyopaque) *KvCacheMsa {
        return @ptrCast(@alignCast(p));
    }
};

/// `GGML_PAD` (ggml.h:267 @b29c606e2) over `uint32_t`.
fn padU32(x: u32, n: u32) u32 {
    return (x +% n -% 1) & ~(n -% 1);
}

fn releaseAll(list: *std.ArrayList(Ubatch)) void {
    for (list.items) |*u| u.release();
    list.deinit(allocator);
}

fn freeSlotInfos(list: *std.ArrayList(SlotInfo)) void {
    for (list.items) |*si| si.deinit();
    list.deinit(allocator);
}

fn retainedCopy(list: []const Ubatch) std.ArrayList(Ubatch) {
    var out: std.ArrayList(Ubatch) = .empty;
    out.ensureTotalCapacity(allocator, list.len) catch oom();
    for (list) |u| {
        u.retain();
        out.appendAssumeCapacity(u);
    }
    return out;
}

/// Ports `llama_kv_cache_msa_context` (llama-kv-cache-msa.h:90 @b29c606e2).
pub const MsaContext = struct {
    kv: ?*KvCacheMsa = null,
    /// The index of the next ubatch to process.
    i_next: usize = 0,
    ubatches: std.ArrayList(Ubatch) = .empty,
    ctx_base: ?*KvCacheContext = null,
    ctx_idx: ?*KvCacheContext = null,
    status: memory.Status,

    /// Ports the error constructor of `llama_kv_cache_msa_context`
    /// (llama-kv-cache-msa.cpp:180 @b29c606e2).
    pub fn createStatus(status: memory.Status) *MsaContext {
        const self = allocator.create(MsaContext) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-cache constructor of `llama_kv_cache_msa_context`
    /// (llama-kv-cache-msa.cpp:183 @b29c606e2).
    pub fn createFull(kv: *KvCacheMsa) *MsaContext {
        const self = allocator.create(MsaContext) catch oom();
        const b = kv.getBase().initFull();
        const i = kv.getIdx().initFull();
        self.* = .{ .kv = kv, .ctx_base = b, .ctx_idx = i, .status = memory.statusCombine(b.getStatus(), i.getStatus()) };
        return self;
    }

    /// Ports the update constructor of `llama_kv_cache_msa_context`
    /// (llama-kv-cache-msa.cpp:191 @b29c606e2).
    pub fn createUpdate(kv: *KvCacheMsa, lctx: *anyopaque, optimize: bool) *MsaContext {
        const self = allocator.create(MsaContext) catch oom();
        const b = kv.getBase().initUpdate(lctx, optimize);
        const i = kv.getIdx().initUpdate(lctx, optimize);
        self.* = .{ .kv = kv, .ctx_base = b, .ctx_idx = i, .status = memory.statusCombine(b.getStatus(), i.getStatus()) };
        return self;
    }

    /// Ports the batch constructor of `llama_kv_cache_msa_context`
    /// (llama-kv-cache-msa.cpp:201 @b29c606e2): each cache gets its own copy
    /// of the ubatches, as the C++ copies them.
    pub fn createBatch(kv: *KvCacheMsa, sinfos_base: std.ArrayList(SlotInfo), sinfos_idx: std.ArrayList(SlotInfo), ubatches: std.ArrayList(Ubatch)) *MsaContext {
        const self = allocator.create(MsaContext) catch oom();
        const b = KvCacheContext.createBatch(kv.getBase(), sinfos_base, retainedCopy(ubatches.items));
        const i = KvCacheContext.createBatch(kv.getIdx(), sinfos_idx, retainedCopy(ubatches.items));
        self.* = .{ .kv = kv, .ubatches = ubatches, .ctx_base = b, .ctx_idx = i, .status = memory.statusCombine(b.getStatus(), i.getStatus()) };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *MsaContext) void {
        if (self.ctx_idx) |i| i.destroy();
        if (self.ctx_base) |b| b.destroy();
        releaseAll(&self.ubatches);
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_msa_context`'s `next` (llama-kv-cache-msa.cpp:216 @b29c606e2).
    pub fn next(self: *MsaContext) bool {
        std.debug.assert(self.status == .success);
        _ = self.ctx_base.?.next();
        _ = self.ctx_idx.?.next();
        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_kv_cache_msa_context`'s `apply` (llama-kv-cache-msa.cpp:229
    /// @b29c606e2): both caches apply, whatever the first answers.
    pub fn apply(self: *MsaContext) bool {
        std.debug.assert(!memory.statusIsFail(self.status));
        var res = true;
        res = self.ctx_base.?.apply() and res;
        res = self.ctx_idx.?.apply() and res;
        return res;
    }

    /// Ports `llama_kv_cache_msa_context`'s `get_status` (llama-kv-cache-msa.cpp:240 @b29c606e2).
    pub fn getStatus(self: *const MsaContext) memory.Status {
        return self.status;
    }

    /// Ports `llama_kv_cache_msa_context`'s `get_ubatch` (llama-kv-cache-msa.cpp:244 @b29c606e2).
    pub fn getUbatch(self: *const MsaContext) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_kv_cache_msa_context`'s `get_base` (llama-kv-cache-msa.cpp:250 @b29c606e2).
    pub fn getBase(self: *const MsaContext) *const KvCacheContext {
        std.debug.assert(self.status == .success);
        return self.ctx_base.?;
    }

    /// Ports `llama_kv_cache_msa_context`'s `get_idx` (llama-kv-cache-msa.cpp:256 @b29c606e2).
    pub fn getIdx(self: *const MsaContext) *const KvCacheContext {
        std.debug.assert(self.status == .success);
        return self.ctx_idx.?;
    }

    /// Ports `llama_kv_cache_msa_context`'s `get_n_pos` (llama-kv-cache-msa.cpp:262
    /// @b29c606e2): the largest position in the cache plus one, padded. MSA
    /// blocks are defined over token positions, so the block-selection
    /// tensors are sized by this rather than by the number of cells.
    pub fn getNPos(self: *const MsaContext) u32 {
        const kv = self.kv.?;
        // Pad the value so that the graph stays constant across batches
        // and can be reused.
        const n_pad_cur: u32 = @max(kv.getNPad(), 256);

        var pos_max: i32 = -1;

        var seq_id: i32 = 0;
        while (seq_id < @as(i32, @bitCast(kv.getNSeqMax()))) : (seq_id += 1) {
            pos_max = @max(pos_max, kv.seqPosMax(seq_id));
        }

        return @max(n_pad_cur, padU32(@bitCast(pos_max +% 1), n_pad_cur));
    }

    /// Ports `llama_kv_cache_msa_context`'s `set_input_cell_pos`
    /// (llama-kv-cache-msa.cpp:275 @b29c606e2): each cell's position divided
    /// by `div`, per stream. The value for an empty or another sequence's
    /// cell is 0; consumers mask those.
    pub fn setInputCellPos(self: *const MsaContext, dst: *Tensor, ubatch: *const Ubatch, div: i32) void {
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache-msa.cpp", 276, "ggml_backend_buffer_is_host(dst->buffer)");
        impl.assert(dst.type == c.GGML_TYPE_I32, "llama-kv-cache-msa.cpp", 277, "dst->type == GGML_TYPE_I32");
        impl.assert(div > 0, "llama-kv-cache-msa.cpp", 278, "div > 0");

        const n_tokens: i64 = ubatch.n_tokens;
        const n_kv: i64 = dst.ne[0];
        const n_stream_ub: i64 = dst.ne[1];

        impl.assert(@rem(n_tokens, n_stream_ub) == 0, "llama-kv-cache-msa.cpp", 284, "n_tokens % n_stream_ub == 0");
        const n_tps = @divTrunc(n_tokens, n_stream_ub);

        const data: [*]i32 = @ptrCast(@alignCast(dst.data));

        var s: i64 = 0;
        while (s < n_stream_ub) : (s += 1) {
            const seq_id = ubatch.seq_id[@intCast(s * n_tps)][0];

            const cells = self.kv.?.getBase().getCells(seq_id);

            var j: i64 = 0;
            while (j < n_kv) : (j += 1) {
                const ju: u32 = @intCast(j);
                // The value for empty or other-sequence cells is irrelevant,
                // as consumers mask them.
                data[@intCast(s * n_kv + j)] =
                    if (cells.isEmpty(ju) or !cells.seqHas(ju, seq_id))
                        0
                    else
                        @divTrunc(cells.posGet(ju), div);
            }
        }
    }

    /// Ports `llama_kv_cache_msa_context`'s `set_input_pos_slot`
    /// (llama-kv-cache-msa.cpp:304 @b29c606e2): each position's cell, per
    /// stream. A position without a cell maps to cell 0, which consumers
    /// must mask; it assumes one sequence per stream.
    pub fn setInputPosSlot(self: *const MsaContext, dst: *Tensor, ubatch: *const Ubatch) void {
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache-msa.cpp", 305, "ggml_backend_buffer_is_host(dst->buffer)");
        impl.assert(dst.type == c.GGML_TYPE_I32 or dst.type == c.GGML_TYPE_F32, "llama-kv-cache-msa.cpp", 306, "dst->type == GGML_TYPE_I32 || dst->type == GGML_TYPE_F32");

        const n_tokens: i64 = ubatch.n_tokens;
        const n_pos: i64 = dst.ne[0];
        const n_stream_ub: i64 = dst.ne[1];

        impl.assert(@rem(n_tokens, n_stream_ub) == 0, "llama-kv-cache-msa.cpp", 312, "n_tokens % n_stream_ub == 0");
        const n_tps = @divTrunc(n_tokens, n_stream_ub);

        const map = allocator.alloc(i32, @intCast(n_pos)) catch oom();
        defer allocator.free(map);

        var s: i64 = 0;
        while (s < n_stream_ub) : (s += 1) {
            const seq_id = ubatch.seq_id[@intCast(s * n_tps)][0];

            const cells = self.kv.?.getBase().getCells(seq_id);

            @memset(map, 0);

            var j: u32 = 0;
            while (j < cells.size()) : (j += 1) {
                if (cells.isEmpty(j) or !cells.seqHas(j, seq_id)) continue;

                const p0 = cells.posGet(j);

                if (p0 < 0 or p0 >= n_pos) continue;

                map[@intCast(p0)] = @bitCast(j);
            }

            const off: usize = @intCast(s * n_pos);
            if (dst.type == c.GGML_TYPE_I32) {
                const data: [*]i32 = @ptrCast(@alignCast(dst.data));
                @memcpy(data[off .. off + map.len], map);
            } else {
                const data: [*]f32 = @ptrCast(@alignCast(dst.data));
                for (map, 0..) |m, p| data[off + p] = @floatFromInt(m);
            }
        }
    }

    /// Ports `llama_kv_cache_msa_context`'s `set_input_pos_mask`
    /// (llama-kv-cache-msa.cpp:348 @b29c606e2): 0 where a token sees a
    /// position, -inf elsewhere -- causal, and windowed when the cache is.
    pub fn setInputPosMask(self: *const MsaContext, dst: *Tensor, ubatch: *const Ubatch) void {
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache-msa.cpp", 349, "ggml_backend_buffer_is_host(dst->buffer)");
        impl.assert(dst.type == c.GGML_TYPE_F32, "llama-kv-cache-msa.cpp", 350, "dst->type == GGML_TYPE_F32");

        const n_tokens: i64 = ubatch.n_tokens;
        const n_pos: i64 = dst.ne[0];

        impl.assert(dst.ne[1] == n_tokens, "llama-kv-cache-msa.cpp", 355, "dst->ne[1] == n_tokens");

        const kv = self.kv.?;
        const n_swa = kv.getNSwa();
        const swa_type = kv.getSwaType();

        const data: [*]f32 = @ptrCast(@alignCast(dst.data));

        @memset(data[0..@intCast(n_pos * n_tokens)], -std.math.inf(f32));

        var i: i64 = 0;
        while (i < n_tokens) : (i += 1) {
            const iu: usize = @intCast(i);
            const seq_id = ubatch.seq_id[iu][0];

            const cells = kv.getBase().getCells(seq_id);

            const p1 = ubatch.pos[iu];

            var j: u32 = 0;
            while (j < cells.size()) : (j += 1) {
                if (cells.isEmpty(j) or !cells.seqHas(j, seq_id)) continue;

                const p0 = cells.posGet(j);

                if (p0 < 0 or p0 >= n_pos) continue;

                // Causal mask.
                if (p0 > p1) continue;

                // Apply SWA, if any.
                if (Hparams.isMaskedSwa(n_swa, swa_type, p0, p1)) continue;

                data[@intCast(i * n_pos + p0)] = 0.0;
            }
        }
    }

    pub fn contextOf(self: *MsaContext) memory.Context {
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

    fn cast(p: *anyopaque) *MsaContext {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(KvCacheMsa);
    std.testing.refAllDecls(MsaContext);
}

test "the position axis pads to 256" {
    try std.testing.expectEqual(@as(u32, 0), padU32(0, 256));
    try std.testing.expectEqual(@as(u32, 256), padU32(1, 256));
    try std.testing.expectEqual(@as(u32, 512), padU32(257, 256));
}
