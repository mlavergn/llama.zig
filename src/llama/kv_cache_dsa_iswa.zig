//! DeepSeek sparse attention with an interleaved sliding window: a DSA
//! cache (`kv_cache_dsa.zig`) for the layers that attend globally and a
//! window-sized `llama_kv_cache` for the layers that attend within the
//! window, driven as one.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-kv-cache-dsa-iswa.h` -- `llama_kv_cache_dsa_iswa`
//!   and `llama_kv_cache_dsa_iswa_context`
//! - `llama.cpp/src/llama-kv-cache-dsa-iswa.cpp` -- their bodies
//!
//! All at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("impl.zig");
const hparams_mod = @import("hparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const io = @import("io.zig");
const kv_cache = @import("kv_cache.zig");
const kv_cache_dsa = @import("kv_cache_dsa.zig");
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
const KvCacheDsa = kv_cache_dsa.KvCacheDsa;
const DsaContext = kv_cache_dsa.DsaContext;

fn oom() noreturn {
    impl.abort("llama-kv-cache-dsa-iswa.cpp", 0, "out of memory");
}

/// The chained filter of the constructor's two lambdas: the caller's MLA
/// filter, then the layer's side of the window. It lives on the
/// constructor's stack, as the C++'s by-reference captures do; the caches
/// read their filter only while they are built.
const ChainedFilter = struct {
    hparams: *const Hparams,
    outer: memory.LayerFilter,
    swa: bool,

    fn call(ctx: ?*const anyopaque, il: i32) bool {
        const self: *const ChainedFilter = @ptrCast(@alignCast(ctx.?));
        if (self.outer.func != null and !self.outer.call(il)) return false;
        return self.hparams.isSwa(@intCast(il)) == self.swa;
    }

    fn filter(self: *const ChainedFilter) memory.LayerFilter {
        return .{ .ctx = self, .func = call };
    }
};

/// Ports `llama_kv_cache_dsa_iswa` (llama-kv-cache-dsa-iswa.h:13 @c1d0e7a00):
/// a `llama_kv_cache_dsa` for the full-attention layers and a
/// `llama_kv_cache` for the SWA layers.
pub const KvCacheDsaIswa = struct {
    unified: bool,
    kv_dsa: *KvCacheDsa,
    kv_swa: *KvCache,

    /// Ports the constructor of `llama_kv_cache_dsa_iswa`
    /// (llama-kv-cache-dsa-iswa.cpp:14 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `filter_mla`: chained before the split by `is_swa`; empty selects every layer.
    /// - `filter_lid`: the layers of the indexer cache, unchained.
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        type_k: c.enum_ggml_type,
        type_v: c.enum_ggml_type,
        v_trans: bool,
        offload: bool,
        swa_full: bool,
        unified: bool,
        kv_size: u32,
        n_seq_max: u32,
        n_ubatch: u32,
        n_pad: u32,
        filter_mla: memory.LayerFilter,
        filter_lid: memory.LayerFilter,
        reuse: memory.LayerMap,
        exc: *Exception,
    ) Thrown!*KvCacheDsaIswa {
        const func = "llama_kv_cache_dsa_iswa";
        const hparams = &model.hparams;

        // Chain the filters.
        const filter_dsa: ChainedFilter = .{ .hparams = hparams, .outer = filter_mla, .swa = false };
        const filter_swa: ChainedFilter = .{ .hparams = hparams, .outer = filter_mla, .swa = true };

        const size_dsa = kv_size;

        // The SWA cache is always padded to 256, for performance
        // (https://github.com/ggml-org/llama.cpp/issues/17037).
        const want_swa: u32 = hparams.n_swa *% (if (unified) n_seq_max else 1) +% n_ubatch;
        var size_swa: u32 = padTo(@min(size_dsa, want_swa), 256);

        // A full-size SWA cache is as large as the DSA one.
        if (swa_full) {
            impl.logWarn("%s: using full-size SWA cache (ref: %s)\n", .{ func, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055" });
            size_swa = size_dsa;
        }

        impl.logInfo("%s: creating DSA KV cache, size = %u cells\n", .{ func, size_dsa });

        const kv_dsa = try KvCacheDsa.create(model, type_k, type_v, v_trans, offload, unified, size_dsa, n_seq_max, n_pad, 0, .none, filter_dsa.filter(), filter_lid, reuse, exc);
        errdefer kv_dsa.destroy();

        impl.logInfo("%s: creating SWA KV cache, size = %u cells\n", .{ func, size_swa });

        const swa_type: hparams_mod.SwaType = @fromBackingInt(@intCast(hparams.swa_type));
        const kv_swa = try KvCache.create(model, hparams, type_k, type_v, v_trans, offload, unified, size_swa, n_seq_max, n_pad, hparams.n_swa, swa_type, null, filter_swa.filter(), reuse, .{}, exc);

        const self = allocator.create(KvCacheDsaIswa) catch oom();
        self.* = .{ .unified = unified, .kv_dsa = kv_dsa, .kv_swa = kv_swa };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *KvCacheDsaIswa) void {
        self.kv_dsa.destroy();
        self.kv_swa.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `clear` (llama-kv-cache-dsa-iswa.cpp:78 @c1d0e7a00).
    pub fn clear(self: *KvCacheDsaIswa, data: bool) void {
        self.kv_dsa.clear(data);
        self.kv_swa.clear(data);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `seq_rm` (llama-kv-cache-dsa-iswa.cpp:83
    /// @c1d0e7a00): both halves, whatever the first answers.
    pub fn seqRm(self: *KvCacheDsaIswa, seq_id: i32, p0: i32, p1: i32) bool {
        var res = true;
        res = self.kv_dsa.seqRm(seq_id, p0, p1) and res;
        res = self.kv_swa.seqRm(seq_id, p0, p1) and res;
        return res;
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `seq_cp` (llama-kv-cache-dsa-iswa.cpp:92 @c1d0e7a00).
    pub fn seqCp(self: *KvCacheDsaIswa, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) void {
        self.kv_dsa.seqCp(seq_id_src, seq_id_dst, p0, p1);
        self.kv_swa.seqCp(seq_id_src, seq_id_dst, p0, p1);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `seq_keep` (llama-kv-cache-dsa-iswa.cpp:97 @c1d0e7a00).
    pub fn seqKeep(self: *KvCacheDsaIswa, seq_id: i32) void {
        self.kv_dsa.seqKeep(seq_id);
        self.kv_swa.seqKeep(seq_id);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `seq_add` (llama-kv-cache-dsa-iswa.cpp:102 @c1d0e7a00).
    pub fn seqAdd(self: *KvCacheDsaIswa, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.kv_dsa.seqAdd(seq_id, p0, p1, shift);
        self.kv_swa.seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `seq_div` (llama-kv-cache-dsa-iswa.cpp:107 @c1d0e7a00).
    pub fn seqDiv(self: *KvCacheDsaIswa, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.kv_dsa.seqDiv(seq_id, p0, p1, d);
        self.kv_swa.seqDiv(seq_id, p0, p1, d);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `seq_pos_min`
    /// (llama-kv-cache-dsa-iswa.cpp:112 @c1d0e7a00): the DSA cache is a
    /// superset of the SWA cache, so the SWA cache answers.
    pub fn seqPosMin(self: *const KvCacheDsaIswa, seq_id: i32) i32 {
        return self.kv_swa.seqPosMin(seq_id);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `seq_pos_max` (llama-kv-cache-dsa-iswa.cpp:117 @c1d0e7a00).
    pub fn seqPosMax(self: *const KvCacheDsaIswa, seq_id: i32) i32 {
        return self.kv_swa.seqPosMax(seq_id);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `memory_breakdown`
    /// (llama-kv-cache-dsa-iswa.cpp:121 @c1d0e7a00).
    pub fn memoryBreakdown(self: *const KvCacheDsaIswa, out: *std.ArrayList(memory.BuftSize)) void {
        self.kv_dsa.memoryBreakdown(out);
        self.kv_swa.memoryBreakdown(out);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `init_batch`
    /// (llama-kv-cache-dsa-iswa.cpp:129 @c1d0e7a00): a simple split when the
    /// cache is unified, then an equal split; each must place every ubatch
    /// in all three caches.
    pub fn initBatch(self: *KvCacheDsaIswa, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *DsaIswaContext {
        _ = embd_all;

        // First the simple split, which needs a unified cache.
        if (self.unified) {
            if (self.tryPrepare(balloc, n_ubatch, false)) |ctx| return ctx;
        }

        // Then the equal split.
        if (self.tryPrepare(balloc, n_ubatch, true)) |ctx| return ctx;

        return DsaIswaContext.createStatus(.failed_prepare);
    }

    /// One `do { ... } while (false)` block of `init_batch`.
    fn tryPrepare(self: *KvCacheDsaIswa, balloc: *batch.BatchAllocr, n_ubatch: u32, equal: bool) ?*DsaIswaContext {
        balloc.splitReset();

        var ubatches: std.ArrayList(Ubatch) = .empty;
        while (true) {
            const ub = if (equal) balloc.splitEqual(n_ubatch, !self.unified, 0) else balloc.splitSimple(n_ubatch);
            if (ub.n_tokens == 0) break;
            ubatches.append(allocator, ub) catch oom();
        }

        if (balloc.getNUsed() < balloc.getNTokens()) {
            // No suitable split.
            releaseAll(&ubatches);
            return null;
        }

        var sinfos_mla = prepareIn(self.kv_dsa.getMla(), &ubatches) orelse return null;

        var sinfos_lid = prepareIn(self.kv_dsa.getLid(), &ubatches) orelse {
            freeSlotInfos(&sinfos_mla);
            return null;
        };

        const sinfos_swa = prepareIn(self.kv_swa, &ubatches) orelse {
            freeSlotInfos(&sinfos_lid);
            freeSlotInfos(&sinfos_mla);
            return null;
        };

        std.debug.assert(sinfos_mla.items.len == sinfos_swa.items.len);

        return DsaIswaContext.createBatch(self, sinfos_mla, sinfos_lid, sinfos_swa, ubatches);
    }

    /// `prepare` on one cache, as `init_batch` checks it: null, with the
    /// ubatches released, when it found no slots.
    fn prepareIn(kv: *KvCache, ubatches: *std.ArrayList(Ubatch)) ?std.ArrayList(SlotInfo) {
        var sinfos = kv.prepare(ubatches.items) orelse {
            releaseAll(ubatches);
            return null;
        };
        if (sinfos.items.len == 0) {
            sinfos.deinit(allocator);
            releaseAll(ubatches);
            return null;
        }
        return sinfos;
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `init_full` (llama-kv-cache-dsa-iswa.cpp:222 @c1d0e7a00).
    pub fn initFull(self: *KvCacheDsaIswa) *DsaIswaContext {
        return DsaIswaContext.createFull(self);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `init_update` (llama-kv-cache-dsa-iswa.cpp:226 @c1d0e7a00).
    pub fn initUpdate(self: *KvCacheDsaIswa, lctx: *anyopaque, optimize: bool) *DsaIswaContext {
        return DsaIswaContext.createUpdate(self, lctx, optimize);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `get_can_shift`
    /// (llama-kv-cache-dsa-iswa.cpp:230 @c1d0e7a00): only when both can and
    /// the main and the SWA caches are the same size.
    pub fn getCanShift(self: *const KvCacheDsaIswa) bool {
        return self.kv_dsa.getCanShift() and
            self.kv_swa.getCanShift() and
            self.kv_dsa.getMla().getSize() == self.kv_swa.getSize();
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `state_write`
    /// (llama-kv-cache-dsa-iswa.cpp:236 @c1d0e7a00): a partial state holds
    /// the SWA cache alone.
    pub fn stateWrite(self: *const KvCacheDsaIswa, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        if (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 0) try self.kv_dsa.stateWrite(w, seq_id, flags, exc);
        try self.kv_swa.stateWrite(w, seq_id, flags, exc);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `state_read` (llama-kv-cache-dsa-iswa.cpp:244 @c1d0e7a00).
    pub fn stateRead(self: *KvCacheDsaIswa, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        if (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 0) try self.kv_dsa.stateRead(r, seq_id, flags, exc);
        try self.kv_swa.stateRead(r, seq_id, flags, exc);
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `get_dsa` (llama-kv-cache-dsa-iswa.cpp:252 @c1d0e7a00).
    pub fn getDsa(self: *const KvCacheDsaIswa) *KvCacheDsa {
        return self.kv_dsa;
    }

    /// Ports `llama_kv_cache_dsa_iswa`'s `get_swa` (llama-kv-cache-dsa-iswa.cpp:256 @c1d0e7a00).
    pub fn getSwa(self: *const KvCacheDsaIswa) *KvCache {
        return self.kv_swa;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *KvCacheDsaIswa) memory.Memory {
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

    fn from(p: *anyopaque) *KvCacheDsaIswa {
        return @ptrCast(@alignCast(p));
    }
};

/// `GGML_PAD` (ggml.h:267 @c1d0e7a00) for a power-of-two `n`.
fn padTo(x: u32, comptime n: u32) u32 {
    return (x +% (n - 1)) & ~@as(u32, n - 1);
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

/// Ports `llama_kv_cache_dsa_iswa_context` (llama-kv-cache-dsa-iswa.h:80 @c1d0e7a00).
pub const DsaIswaContext = struct {
    /// The index of the next ubatch to process.
    i_next: usize = 0,
    ubatches: std.ArrayList(Ubatch) = .empty,
    ctx_dsa: ?*DsaContext = null,
    ctx_swa: ?*KvCacheContext = null,
    status: memory.Status,

    /// Ports the error constructor of `llama_kv_cache_dsa_iswa_context`
    /// (llama-kv-cache-dsa-iswa.cpp:264 @c1d0e7a00).
    pub fn createStatus(status: memory.Status) *DsaIswaContext {
        const self = allocator.create(DsaIswaContext) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-cache constructor of `llama_kv_cache_dsa_iswa_context`
    /// (llama-kv-cache-dsa-iswa.cpp:266 @c1d0e7a00).
    pub fn createFull(kv: *KvCacheDsaIswa) *DsaIswaContext {
        const self = allocator.create(DsaIswaContext) catch oom();
        const d = kv.getDsa().initFull();
        const s = kv.getSwa().initFull();
        self.* = .{ .ctx_dsa = d, .ctx_swa = s, .status = memory.statusCombine(d.getStatus(), s.getStatus()) };
        return self;
    }

    /// Ports the update constructor of `llama_kv_cache_dsa_iswa_context`
    /// (llama-kv-cache-dsa-iswa.cpp:273 @c1d0e7a00).
    pub fn createUpdate(kv: *KvCacheDsaIswa, lctx: *anyopaque, optimize: bool) *DsaIswaContext {
        const self = allocator.create(DsaIswaContext) catch oom();
        const d = kv.getDsa().initUpdate(lctx, optimize);
        const s = kv.getSwa().initUpdate(lctx, optimize);
        self.* = .{ .ctx_dsa = d, .ctx_swa = s, .status = memory.statusCombine(d.getStatus(), s.getStatus()) };
        return self;
    }

    /// Ports the batch constructor of `llama_kv_cache_dsa_iswa_context`
    /// (llama-kv-cache-dsa-iswa.cpp:282 @c1d0e7a00): each half gets its own
    /// copy of the ubatches, as the C++ copies them.
    pub fn createBatch(kv: *KvCacheDsaIswa, sinfos_mla: std.ArrayList(SlotInfo), sinfos_lid: std.ArrayList(SlotInfo), sinfos_swa: std.ArrayList(SlotInfo), ubatches: std.ArrayList(Ubatch)) *DsaIswaContext {
        const self = allocator.create(DsaIswaContext) catch oom();
        const d = DsaContext.createBatch(kv.getDsa(), sinfos_mla, sinfos_lid, retainedCopy(ubatches.items));
        const s = KvCacheContext.createBatch(kv.getSwa(), sinfos_swa, retainedCopy(ubatches.items));
        self.* = .{ .ubatches = ubatches, .ctx_dsa = d, .ctx_swa = s, .status = memory.statusCombine(d.getStatus(), s.getStatus()) };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *DsaIswaContext) void {
        releaseAll(&self.ubatches);
        if (self.ctx_dsa) |d| d.destroy();
        if (self.ctx_swa) |s| s.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_dsa_iswa_context`'s `next` (llama-kv-cache-dsa-iswa.cpp:297 @c1d0e7a00).
    pub fn next(self: *DsaIswaContext) bool {
        std.debug.assert(self.status == .success);
        _ = self.ctx_dsa.?.next();
        _ = self.ctx_swa.?.next();
        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_kv_cache_dsa_iswa_context`'s `apply`
    /// (llama-kv-cache-dsa-iswa.cpp:310 @c1d0e7a00): both halves apply,
    /// whatever the first answers.
    pub fn apply(self: *DsaIswaContext) bool {
        std.debug.assert(!memory.statusIsFail(self.status));
        var res = true;
        res = self.ctx_dsa.?.apply() and res;
        res = self.ctx_swa.?.apply() and res;
        return res;
    }

    /// Ports `llama_kv_cache_dsa_iswa_context`'s `get_status` (llama-kv-cache-dsa-iswa.cpp:321 @c1d0e7a00).
    pub fn getStatus(self: *const DsaIswaContext) memory.Status {
        return self.status;
    }

    /// Ports `llama_kv_cache_dsa_iswa_context`'s `get_ubatch` (llama-kv-cache-dsa-iswa.cpp:325 @c1d0e7a00).
    pub fn getUbatch(self: *const DsaIswaContext) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_kv_cache_dsa_iswa_context`'s `get_dsa` (llama-kv-cache-dsa-iswa.cpp:331 @c1d0e7a00).
    pub fn getDsa(self: *const DsaIswaContext) *const DsaContext {
        std.debug.assert(self.status == .success);
        return self.ctx_dsa.?;
    }

    /// Ports `llama_kv_cache_dsa_iswa_context`'s `get_swa` (llama-kv-cache-dsa-iswa.cpp:337 @c1d0e7a00).
    pub fn getSwa(self: *const DsaIswaContext) *const KvCacheContext {
        std.debug.assert(self.status == .success);
        return self.ctx_swa.?;
    }

    pub fn contextOf(self: *DsaIswaContext) memory.Context {
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

    fn cast(p: *anyopaque) *DsaIswaContext {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(KvCacheDsaIswa);
    std.testing.refAllDecls(DsaIswaContext);
}

test "the SWA cache rounds up to 256 cells" {
    try std.testing.expectEqual(@as(u32, 256), padTo(1, 256));
    try std.testing.expectEqual(@as(u32, 512), padTo(257, 256));
}
