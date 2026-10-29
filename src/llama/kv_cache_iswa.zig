//! An interleaved sliding-window KV cache: a full-size cache for the
//! layers that attend globally and a window-sized one for the layers that
//! attend within the window, driven as one.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-kv-cache-iswa.h` -- `llama_kv_cache_iswa` and
//!   `llama_kv_cache_iswa_context`
//! - `llama.cpp/src/llama-kv-cache-iswa.cpp` -- their bodies
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

fn oom() noreturn {
    impl.abort("llama-kv-cache-iswa.cpp", 0, "out of memory");
}

/// The chained filter of the constructor's two lambdas: the caller's
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

/// Ports `llama_kv_cache_iswa` (llama-kv-cache-iswa.h:14 @b29c606e2): two
/// `llama_kv_cache`s, the first for the non-SWA layers and the second for
/// the SWA layers.
pub const KvCacheIswa = struct {
    unified: bool,
    kv_base: *KvCache,
    kv_swa: *KvCache,

    /// Ports the constructor of `llama_kv_cache_iswa` that takes the
    /// hyperparameters (llama-kv-cache-iswa.cpp:34 @b29c606e2); the one
    /// that does not (llama-kv-cache-iswa.cpp:14 @b29c606e2) passes the
    /// model's, which is `hparams = null` here.
    ///
    /// Parameters:
    /// - `mem_other`: another iSWA cache whose cells this one shares, or null.
    /// - `filter`: chained before the split by `is_swa`; empty selects every layer.
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        hparams_: ?*const Hparams,
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
        mem_other: ?memory.Memory,
        filter: memory.LayerFilter,
        reuse: memory.LayerMap,
        share: memory.LayerMap,
        exc: *Exception,
    ) Thrown!*KvCacheIswa {
        const func = "llama_kv_cache_iswa";
        const hparams = hparams_ orelse &model.hparams;

        // Chain the filters. Both test `is_swa` on the model's own
        // hyperparameters, as the C++'s lambdas do.
        const filter_base: ChainedFilter = .{ .hparams = &model.hparams, .outer = filter, .swa = false };
        const filter_swa: ChainedFilter = .{ .hparams = &model.hparams, .outer = filter, .swa = true };

        const size_base = kv_size;

        // The SWA cache is always padded to 256, for performance
        // (https://github.com/ggml-org/llama.cpp/issues/17037).
        const want_swa: u32 = hparams.n_swa *% (if (unified) n_seq_max else 1) +% n_ubatch;
        var size_swa: u32 = padTo(@min(size_base, want_swa), 256);

        // A full-size SWA cache is as large as the base one.
        if (swa_full) {
            impl.logWarn("%s: using full-size SWA cache (ref: %s)\n", .{ func, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055" });
            size_swa = size_base;
        }

        impl.logInfo("%s: creating non-SWA KV cache, size = %u cells\n", .{ func, size_base });

        var mem_other_base: ?*KvCache = null;
        var mem_other_swa: ?*KvCache = null;
        if (mem_other) |o| {
            mem_other_base = o.cast(KvCacheIswa).getBase();
            mem_other_swa = o.cast(KvCacheIswa).getSwa();
        }

        const kv_base = try KvCache.create(model, hparams, type_k, type_v, v_trans, offload, unified, size_base, n_seq_max, n_pad, 0, .none, mem_other_base, filter_base.filter(), reuse, share, exc);
        errdefer kv_base.destroy();

        impl.logInfo("%s: creating     SWA KV cache, size = %u cells\n", .{ func, size_swa });

        const swa_type: hparams_mod.SwaType = @fromBackingInt(@intCast(hparams.swa_type));
        const kv_swa = try KvCache.create(model, hparams, type_k, type_v, v_trans, offload, unified, size_swa, n_seq_max, n_pad, hparams.n_swa, swa_type, mem_other_swa, filter_swa.filter(), reuse, share, exc);

        const self = allocator.create(KvCacheIswa) catch oom();
        self.* = .{ .unified = unified, .kv_base = kv_base, .kv_swa = kv_swa };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *KvCacheIswa) void {
        self.kv_base.destroy();
        self.kv_swa.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_iswa`'s `clear` (llama-kv-cache-iswa.cpp:108 @b29c606e2).
    pub fn clear(self: *KvCacheIswa, data: bool) void {
        self.kv_base.clear(data);
        self.kv_swa.clear(data);
    }

    /// Ports `llama_kv_cache_iswa`'s `seq_rm` (llama-kv-cache-iswa.cpp:113
    /// @b29c606e2): both halves, whatever the first answers.
    pub fn seqRm(self: *KvCacheIswa, seq_id: i32, p0: i32, p1: i32) bool {
        var res = true;
        res = res and self.kv_base.seqRm(seq_id, p0, p1);
        res = self.kv_swa.seqRm(seq_id, p0, p1) and res;
        return res;
    }

    /// Ports `llama_kv_cache_iswa`'s `seq_cp` (llama-kv-cache-iswa.cpp:122 @b29c606e2).
    pub fn seqCp(self: *KvCacheIswa, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) void {
        self.kv_base.seqCp(seq_id_src, seq_id_dst, p0, p1);
        self.kv_swa.seqCp(seq_id_src, seq_id_dst, p0, p1);
    }

    /// Ports `llama_kv_cache_iswa`'s `seq_keep` (llama-kv-cache-iswa.cpp:127 @b29c606e2).
    pub fn seqKeep(self: *KvCacheIswa, seq_id: i32) void {
        self.kv_base.seqKeep(seq_id);
        self.kv_swa.seqKeep(seq_id);
    }

    /// Ports `llama_kv_cache_iswa`'s `seq_add` (llama-kv-cache-iswa.cpp:132 @b29c606e2).
    pub fn seqAdd(self: *KvCacheIswa, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.kv_base.seqAdd(seq_id, p0, p1, shift);
        self.kv_swa.seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_kv_cache_iswa`'s `seq_div` (llama-kv-cache-iswa.cpp:137 @b29c606e2).
    pub fn seqDiv(self: *KvCacheIswa, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.kv_base.seqDiv(seq_id, p0, p1, d);
        self.kv_swa.seqDiv(seq_id, p0, p1, d);
    }

    /// Ports `llama_kv_cache_iswa`'s `seq_pos_min` (llama-kv-cache-iswa.cpp:142
    /// @b29c606e2): the base cache is a superset of the SWA cache, so the
    /// SWA cache answers.
    pub fn seqPosMin(self: *const KvCacheIswa, seq_id: i32) i32 {
        return self.kv_swa.seqPosMin(seq_id);
    }

    /// Ports `llama_kv_cache_iswa`'s `seq_pos_max` (llama-kv-cache-iswa.cpp:147 @b29c606e2).
    pub fn seqPosMax(self: *const KvCacheIswa, seq_id: i32) i32 {
        return self.kv_swa.seqPosMax(seq_id);
    }

    /// Ports `llama_kv_cache_iswa`'s `memory_breakdown` (llama-kv-cache-iswa.cpp:151 @b29c606e2).
    pub fn memoryBreakdown(self: *const KvCacheIswa, out: *std.ArrayList(memory.BuftSize)) void {
        self.kv_base.memoryBreakdown(out);
        self.kv_swa.memoryBreakdown(out);
    }

    /// Ports `llama_kv_cache_iswa`'s `init_batch` (llama-kv-cache-iswa.cpp:159
    /// @b29c606e2): a simple split when the cache is unified, then an equal
    /// split; each must place every ubatch in both caches.
    pub fn initBatch(self: *KvCacheIswa, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *IswaContext {
        _ = embd_all;

        // First the simple split, which needs a unified cache.
        if (self.unified) {
            if (self.tryPrepare(balloc, n_ubatch, false)) |ctx| return ctx;
        }

        // Then the equal split.
        if (self.tryPrepare(balloc, n_ubatch, true)) |ctx| return ctx;

        // The C++'s TODO: try other splitting strategies, once the batches
        // are flexible enough for it.
        return IswaContext.createStatus(.failed_prepare);
    }

    /// One `do { ... } while (false)` block of `init_batch`.
    fn tryPrepare(self: *KvCacheIswa, balloc: *batch.BatchAllocr, n_ubatch: u32, equal: bool) ?*IswaContext {
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

        var sinfos_base = self.kv_base.prepare(ubatches.items) orelse {
            releaseAll(&ubatches);
            return null;
        };
        if (sinfos_base.items.len == 0) {
            sinfos_base.deinit(allocator);
            releaseAll(&ubatches);
            return null;
        }

        var sinfos_swa = self.kv_swa.prepare(ubatches.items) orelse {
            freeSlotInfos(&sinfos_base);
            releaseAll(&ubatches);
            return null;
        };
        if (sinfos_swa.items.len == 0) {
            sinfos_swa.deinit(allocator);
            freeSlotInfos(&sinfos_base);
            releaseAll(&ubatches);
            return null;
        }

        std.debug.assert(sinfos_base.items.len == sinfos_swa.items.len);

        return IswaContext.createBatch(self, sinfos_base, sinfos_swa, ubatches);
    }

    /// Ports `llama_kv_cache_iswa`'s `init_full` (llama-kv-cache-iswa.cpp:245 @b29c606e2).
    pub fn initFull(self: *KvCacheIswa) *IswaContext {
        return IswaContext.createFull(self);
    }

    /// Ports `llama_kv_cache_iswa`'s `init_update` (llama-kv-cache-iswa.cpp:249 @b29c606e2).
    pub fn initUpdate(self: *KvCacheIswa, lctx: *anyopaque, optimize: bool) *IswaContext {
        return IswaContext.createUpdate(self, lctx, optimize);
    }

    /// Ports `llama_kv_cache_iswa`'s `get_can_shift` (llama-kv-cache-iswa.cpp:253
    /// @b29c606e2): only when both can and the two are the same size.
    pub fn getCanShift(self: *const KvCacheIswa) bool {
        return self.kv_base.getCanShift() and
            self.kv_swa.getCanShift() and
            self.kv_base.getSize() == self.kv_swa.getSize();
    }

    /// Ports `llama_kv_cache_iswa`'s `state_write` (llama-kv-cache-iswa.cpp:259 @b29c606e2).
    pub fn stateWrite(self: *const KvCacheIswa, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        if (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 0) try self.kv_base.stateWrite(w, seq_id, flags, exc);
        try self.kv_swa.stateWrite(w, seq_id, flags, exc);
    }

    /// Ports `llama_kv_cache_iswa`'s `state_read` (llama-kv-cache-iswa.cpp:267 @b29c606e2).
    pub fn stateRead(self: *KvCacheIswa, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        if (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 0) try self.kv_base.stateRead(r, seq_id, flags, exc);
        try self.kv_swa.stateRead(r, seq_id, flags, exc);
    }

    /// Ports `llama_kv_cache_iswa`'s `get_base` (llama-kv-cache-iswa.cpp:275 @b29c606e2).
    pub fn getBase(self: *const KvCacheIswa) *KvCache {
        return self.kv_base;
    }

    /// Ports `llama_kv_cache_iswa`'s `get_swa` (llama-kv-cache-iswa.cpp:279 @b29c606e2).
    pub fn getSwa(self: *const KvCacheIswa) *KvCache {
        return self.kv_swa;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *KvCacheIswa) memory.Memory {
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

    fn from(p: *anyopaque) *KvCacheIswa {
        return @ptrCast(@alignCast(p));
    }
};

/// `GGML_PAD` (ggml.h:267 @b29c606e2) for a power-of-two `n`.
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

/// Ports `llama_kv_cache_iswa_context` (llama-kv-cache-iswa.h:100 @b29c606e2).
pub const IswaContext = struct {
    /// The index of the next ubatch to process.
    i_next: usize = 0,
    ubatches: std.ArrayList(Ubatch) = .empty,
    ctx_base: ?*KvCacheContext = null,
    ctx_swa: ?*KvCacheContext = null,
    status: memory.Status,

    /// Ports the error constructor of `llama_kv_cache_iswa_context`
    /// (llama-kv-cache-iswa.cpp:287 @b29c606e2).
    pub fn createStatus(status: memory.Status) *IswaContext {
        const self = allocator.create(IswaContext) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-cache constructor of `llama_kv_cache_iswa_context`
    /// (llama-kv-cache-iswa.cpp:296 @b29c606e2).
    pub fn createFull(kv: *KvCacheIswa) *IswaContext {
        const self = allocator.create(IswaContext) catch oom();
        const b = kv.getBase().initFull();
        const s = kv.getSwa().initFull();
        self.* = .{ .ctx_base = b, .ctx_swa = s, .status = memory.statusCombine(b.getStatus(), s.getStatus()) };
        return self;
    }

    /// Ports the update constructor of `llama_kv_cache_iswa_context`
    /// (llama-kv-cache-iswa.cpp:289 @b29c606e2).
    pub fn createUpdate(kv: *KvCacheIswa, lctx: *anyopaque, optimize: bool) *IswaContext {
        const self = allocator.create(IswaContext) catch oom();
        const b = kv.getBase().initUpdate(lctx, optimize);
        const s = kv.getSwa().initUpdate(lctx, optimize);
        self.* = .{ .ctx_base = b, .ctx_swa = s, .status = memory.statusCombine(b.getStatus(), s.getStatus()) };
        return self;
    }

    /// Ports the batch constructor of `llama_kv_cache_iswa_context`
    /// (llama-kv-cache-iswa.cpp:305 @b29c606e2): each half gets its own copy
    /// of the ubatches, as the C++ copies them.
    pub fn createBatch(kv: *KvCacheIswa, sinfos_base: std.ArrayList(SlotInfo), sinfos_swa: std.ArrayList(SlotInfo), ubatches: std.ArrayList(Ubatch)) *IswaContext {
        const self = allocator.create(IswaContext) catch oom();
        const b = KvCacheContext.createBatch(kv.getBase(), sinfos_base, retainedCopy(ubatches.items));
        const s = KvCacheContext.createBatch(kv.getSwa(), sinfos_swa, retainedCopy(ubatches.items));
        self.* = .{ .ubatches = ubatches, .ctx_base = b, .ctx_swa = s, .status = memory.statusCombine(b.getStatus(), s.getStatus()) };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *IswaContext) void {
        releaseAll(&self.ubatches);
        if (self.ctx_base) |b| b.destroy();
        if (self.ctx_swa) |s| s.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_iswa_context`'s `next` (llama-kv-cache-iswa.cpp:319 @b29c606e2).
    pub fn next(self: *IswaContext) bool {
        std.debug.assert(self.status == .success);
        _ = self.ctx_base.?.next();
        _ = self.ctx_swa.?.next();
        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_kv_cache_iswa_context`'s `apply` (llama-kv-cache-iswa.cpp:332
    /// @b29c606e2): both halves apply, whatever the first answers.
    pub fn apply(self: *IswaContext) bool {
        std.debug.assert(!memory.statusIsFail(self.status));
        var res = true;
        res = res and self.ctx_base.?.apply();
        res = self.ctx_swa.?.apply() and res;
        return res;
    }

    /// Ports `llama_kv_cache_iswa_context`'s `get_status` (llama-kv-cache-iswa.cpp:343 @b29c606e2).
    pub fn getStatus(self: *const IswaContext) memory.Status {
        return self.status;
    }

    /// Ports `llama_kv_cache_iswa_context`'s `get_ubatch` (llama-kv-cache-iswa.cpp:347 @b29c606e2).
    pub fn getUbatch(self: *const IswaContext) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_kv_cache_iswa_context`'s `get_base` (llama-kv-cache-iswa.cpp:353 @b29c606e2).
    pub fn getBase(self: *const IswaContext) *const KvCacheContext {
        std.debug.assert(self.status == .success);
        return self.ctx_base.?;
    }

    /// Ports `llama_kv_cache_iswa_context`'s `get_swa` (llama-kv-cache-iswa.cpp:359 @b29c606e2).
    pub fn getSwa(self: *const IswaContext) *const KvCacheContext {
        std.debug.assert(self.status == .success);
        return self.ctx_swa.?;
    }

    pub fn contextOf(self: *IswaContext) memory.Context {
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

    fn cast(p: *anyopaque) *IswaContext {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(KvCacheIswa);
    std.testing.refAllDecls(IswaContext);
}

test "the SWA cache rounds up to 256 cells" {
    try std.testing.expectEqual(@as(u32, 256), padTo(1, 256));
    try std.testing.expectEqual(@as(u32, 256), padTo(256, 256));
    try std.testing.expectEqual(@as(u32, 512), padTo(257, 256));
}
