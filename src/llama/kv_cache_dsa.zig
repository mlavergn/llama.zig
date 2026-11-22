//! The KV cache of DeepSeek sparse attention: an MLA cache for the model's
//! keys and a second cache for the lightning indexer's keys, driven as one.
//!
//! The indexer's cache is an ordinary `llama_kv_cache` over a copy of the
//! hyperparameters edited to describe it: the indexer is MQA with one key
//! head of `indexer_head_size`, roped NeoX-style.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-kv-cache-dsa.h` -- `llama_kv_cache_dsa` and
//!   `llama_kv_cache_dsa_context`
//! - `llama.cpp/src/llama-kv-cache-dsa.cpp` -- their bodies
//!
//! All at v1.0.0 (`b1b770d90`).

const std = @import("std");
const mod = @import("module.zig");

fn oom() noreturn {
    mod.impl.abort("llama-kv-cache-dsa.cpp", 0, "out of memory");
}

/// Ports `llama_kv_cache_dsa` (llama-kv-cache-dsa.h:15 @b1b770d90): two
/// `llama_kv_cache`s, the first for the model's keys and the second for
/// the lightning indexer's.
pub const KvCacheDsa = struct {
    /// The indexer cache's hyperparameters. `llama_kv_cache` keeps only a
    /// pointer to its hyperparameters, so they live here, as long as it.
    hparams_lid: mod.hparams.Hparams,
    n_stream: u32 = 1,

    kv_mla: *mod.kv_cache.KvCache,
    kv_lid: *mod.kv_cache.KvCache,

    /// Ports the constructor of `llama_kv_cache_dsa` (llama-kv-cache-dsa.cpp:15
    /// @b1b770d90).
    ///
    /// Parameters:
    /// - `filter_mla`: the layers of the main cache; empty selects every layer.
    /// - `filter_lid`: the layers of the indexer cache; empty selects every layer.
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const mod.model.Model,
        no_alloc: bool,
        type_k: mod.impl.c.enum_ggml_type,
        type_v: mod.impl.c.enum_ggml_type,
        v_trans: bool,
        offload: bool,
        unified: bool,
        kv_size: u32,
        n_seq_max: u32,
        n_swa: u32,
        swa_type: mod.hparams.SwaType,
        filter_mla: mod.memory.LayerFilter,
        filter_lid: mod.memory.LayerFilter,
        reuse: mod.memory.LayerMap,
        exc: *mod.impl.Exception,
    ) mod.impl.Thrown!*KvCacheDsa {
        const func = "llama_kv_cache_dsa";

        const self = std.heap.c_allocator.create(KvCacheDsa) catch oom();
        errdefer std.heap.c_allocator.destroy(self);
        self.* = .{
            .hparams_lid = model.hparams,
            .n_stream = if (unified) 1 else n_seq_max,
            .kv_mla = undefined,
            .kv_lid = undefined,
        };

        mod.impl.logInfo("%s: creating main KV cache, size = %u cells\n", .{ func, kv_size });

        self.kv_mla = try mod.kv_cache.KvCache.create(model, no_alloc, &model.hparams, type_k, type_v, v_trans, offload, unified, kv_size, n_seq_max, n_swa, swa_type, null, filter_mla, reuse, .{}, exc);
        errdefer self.kv_mla.destroy();

        // `llama_kv_cache` caches the indexer keys too: hand-tweaked
        // hyperparameters fool it into creating indexer key tensors of the
        // right dimensions
        // (https://github.com/ggml-org/llama.cpp/pull/21149#discussion_r3015940823).

        // The DSA lightning indexer uses MQA with a single key head.
        @memset(&self.hparams_lid.n_head_kv_arr, 1);
        self.hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;

        // KV sharing applies to the MLA cache only: each indexer owns its keys.
        for (&self.hparams_lid.kv_source_impl, 0..) |*src, i| src.* = @intCast(i);

        mod.impl.logInfo("%s: creating indexer KV cache, size = %u cells\n", .{ func, kv_size });

        self.kv_lid = try mod.kv_cache.KvCache.createTagged(model, no_alloc, &self.hparams_lid, type_k, type_v, v_trans, offload, unified, kv_size, n_seq_max, n_swa, swa_type, null, filter_lid, .{}, .{}, "", true, exc);

        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *KvCacheDsa) void {
        self.kv_mla.destroy();
        self.kv_lid.destroy();
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_dsa`'s `clear` (llama-kv-cache-dsa.cpp:59 @b1b770d90).
    pub fn clear(self: *KvCacheDsa, data: bool) void {
        self.kv_mla.clear(data);
        self.kv_lid.clear(data);
    }

    /// Ports `llama_kv_cache_dsa`'s `seq_rm` (llama-kv-cache-dsa.cpp:64
    /// @b1b770d90): both caches, whatever the first answers.
    pub fn seqRm(self: *KvCacheDsa, seq_id: i32, p0: i32, p1: i32) bool {
        var res = true;
        res = self.kv_mla.seqRm(seq_id, p0, p1) and res;
        res = self.kv_lid.seqRm(seq_id, p0, p1) and res;
        return res;
    }

    /// Ports `llama_kv_cache_dsa`'s `seq_cp` (llama-kv-cache-dsa.cpp:73 @b1b770d90).
    pub fn seqCp(self: *KvCacheDsa, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) void {
        self.kv_mla.seqCp(seq_id_src, seq_id_dst, p0, p1);
        self.kv_lid.seqCp(seq_id_src, seq_id_dst, p0, p1);
    }

    /// Ports `llama_kv_cache_dsa`'s `seq_keep` (llama-kv-cache-dsa.cpp:78 @b1b770d90).
    pub fn seqKeep(self: *KvCacheDsa, seq_id: i32) void {
        self.kv_mla.seqKeep(seq_id);
        self.kv_lid.seqKeep(seq_id);
    }

    /// Ports `llama_kv_cache_dsa`'s `seq_add` (llama-kv-cache-dsa.cpp:83 @b1b770d90).
    pub fn seqAdd(self: *KvCacheDsa, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.kv_mla.seqAdd(seq_id, p0, p1, shift);
        self.kv_lid.seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_kv_cache_dsa`'s `seq_div` (llama-kv-cache-dsa.cpp:88 @b1b770d90).
    pub fn seqDiv(self: *KvCacheDsa, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.kv_mla.seqDiv(seq_id, p0, p1, d);
        self.kv_lid.seqDiv(seq_id, p0, p1, d);
    }

    /// Ports `llama_kv_cache_dsa`'s `seq_pos_min` (llama-kv-cache-dsa.cpp:93
    /// @b1b770d90): the main cache answers.
    pub fn seqPosMin(self: *const KvCacheDsa, seq_id: i32) i32 {
        return self.kv_mla.seqPosMin(seq_id);
    }

    /// Ports `llama_kv_cache_dsa`'s `seq_pos_max` (llama-kv-cache-dsa.cpp:97 @b1b770d90).
    pub fn seqPosMax(self: *const KvCacheDsa, seq_id: i32) i32 {
        return self.kv_mla.seqPosMax(seq_id);
    }

    /// Ports `llama_kv_cache_dsa`'s `memory_breakdown` (llama-kv-cache-dsa.cpp:101 @b1b770d90).
    pub fn memoryBreakdown(self: *const KvCacheDsa, out: *std.ArrayList(mod.memory.BuftSize)) void {
        self.kv_mla.memoryBreakdown(out);
        self.kv_lid.memoryBreakdown(out);
    }

    /// Ports `llama_kv_cache_dsa`'s `init_batch` (llama-kv-cache-dsa.cpp:109
    /// @b1b770d90): a simple split for a single stream, an equal one
    /// otherwise; every ubatch must find a slot in both caches.
    pub fn initBatch(self: *KvCacheDsa, balloc: *mod.batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *DsaContext {
        _ = embd_all;

        blk: {
            balloc.splitReset();

            var ubatches: std.ArrayList(mod.batch.Ubatch) = .empty;
            while (true) {
                const ub = if (self.n_stream == 1) balloc.splitSimple(n_ubatch) else balloc.splitEqual(n_ubatch, true, 0);
                if (ub.n_tokens == 0) break;
                ubatches.append(std.heap.c_allocator, ub) catch oom();
            }

            if (balloc.getNUsed() < balloc.getNTokens()) {
                // No suitable split.
                releaseAll(&ubatches);
                break :blk;
            }

            var sinfos_mla = self.kv_mla.prepare(ubatches.items) orelse {
                releaseAll(&ubatches);
                break :blk;
            };
            if (sinfos_mla.items.len == 0) {
                sinfos_mla.deinit(std.heap.c_allocator);
                releaseAll(&ubatches);
                break :blk;
            }

            var sinfos_lid = self.kv_lid.prepare(ubatches.items) orelse {
                freeSlotInfos(&sinfos_mla);
                releaseAll(&ubatches);
                break :blk;
            };
            if (sinfos_lid.items.len == 0) {
                sinfos_lid.deinit(std.heap.c_allocator);
                freeSlotInfos(&sinfos_mla);
                releaseAll(&ubatches);
                break :blk;
            }

            std.debug.assert(sinfos_mla.items.len == sinfos_lid.items.len);

            return DsaContext.createBatch(self, sinfos_mla, sinfos_lid, ubatches);
        }

        return DsaContext.createStatus(.failed_prepare);
    }

    /// Ports `llama_kv_cache_dsa`'s `init_full` (llama-kv-cache-dsa.cpp:153 @b1b770d90).
    pub fn initFull(self: *KvCacheDsa) *DsaContext {
        return DsaContext.createFull(self);
    }

    /// Ports `llama_kv_cache_dsa`'s `init_update` (llama-kv-cache-dsa.cpp:157 @b1b770d90).
    pub fn initUpdate(self: *KvCacheDsa, lctx: *anyopaque, optimize: bool) *DsaContext {
        return DsaContext.createUpdate(self, lctx, optimize);
    }

    /// Ports `llama_kv_cache_dsa`'s `get_can_shift` (llama-kv-cache-dsa.cpp:161
    /// @b1b770d90): only when both can and the two are the same size.
    pub fn getCanShift(self: *const KvCacheDsa) bool {
        return self.kv_mla.getCanShift() and
            self.kv_lid.getCanShift() and
            self.kv_mla.getSize() == self.kv_lid.getSize();
    }

    /// Ports `llama_kv_cache_dsa`'s `state_write` (llama-kv-cache-dsa.cpp:167 @b1b770d90).
    pub fn stateWrite(self: *const KvCacheDsa, w: mod.io.Writer, seq_id: i32, flags: u32, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        try self.kv_mla.stateWrite(w, seq_id, flags, exc);
        try self.kv_lid.stateWrite(w, seq_id, flags, exc);
    }

    /// Ports `llama_kv_cache_dsa`'s `state_read` (llama-kv-cache-dsa.cpp:172
    /// @b1b770d90): an indexer restore that fails undoes the MLA restore, so
    /// a failed restore leaves nothing behind.
    pub fn stateRead(self: *KvCacheDsa, r: mod.io.Reader, seq_id: i32, flags: u32, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        try self.kv_mla.stateRead(r, seq_id, flags, exc);
        self.kv_lid.stateRead(r, seq_id, flags, exc) catch |err| {
            self.kv_mla.stateClear(seq_id);
            return err;
        };
    }

    /// Ports `llama_kv_cache_dsa`'s `get_mla` (llama-kv-cache-dsa.cpp:185 @b1b770d90).
    pub fn getMla(self: *const KvCacheDsa) *mod.kv_cache.KvCache {
        return self.kv_mla;
    }

    /// Ports `llama_kv_cache_dsa`'s `get_lid` (llama-kv-cache-dsa.cpp:189 @b1b770d90).
    pub fn getLid(self: *const KvCacheDsa) *mod.kv_cache.KvCache {
        return self.kv_lid;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *KvCacheDsa) mod.memory.Memory {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: mod.memory.Memory.VTable = .{
        .initBatch = struct {
            fn f(p: *anyopaque, b: *mod.batch.BatchAllocr, n: u32, e: bool) mod.memory.Context {
                return from(p).initBatch(b, n, e).contextOf();
            }
        }.f,
        .initFull = struct {
            fn f(p: *anyopaque) mod.memory.Context {
                return from(p).initFull().contextOf();
            }
        }.f,
        .initUpdate = struct {
            fn f(p: *anyopaque, l: *anyopaque, o: bool) mod.memory.Context {
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
            fn f(p: *anyopaque, out: *std.ArrayList(mod.memory.BuftSize)) void {
                from(p).memoryBreakdown(out);
            }
        }.f,
        .stateWrite = struct {
            fn f(p: *anyopaque, w: mod.io.Writer, s: i32, fl: u32, e: *mod.impl.Exception) mod.impl.Thrown!void {
                return from(p).stateWrite(w, s, fl, e);
            }
        }.f,
        .stateRead = struct {
            fn f(p: *anyopaque, r: mod.io.Reader, s: i32, fl: u32, e: *mod.impl.Exception) mod.impl.Thrown!void {
                return from(p).stateRead(r, s, fl, e);
            }
        }.f,
        .destroy = struct {
            fn f(p: *anyopaque) void {
                from(p).destroy();
            }
        }.f,
    };

    fn from(p: *anyopaque) *KvCacheDsa {
        return @ptrCast(@alignCast(p));
    }
};

fn releaseAll(list: *std.ArrayList(mod.batch.Ubatch)) void {
    for (list.items) |*u| u.release();
    list.deinit(std.heap.c_allocator);
}

fn freeSlotInfos(list: *std.ArrayList(mod.kv_cache.SlotInfo)) void {
    for (list.items) |*si| si.deinit();
    list.deinit(std.heap.c_allocator);
}

fn retainedCopy(list: []const mod.batch.Ubatch) std.ArrayList(mod.batch.Ubatch) {
    var out: std.ArrayList(mod.batch.Ubatch) = .empty;
    out.ensureTotalCapacity(std.heap.c_allocator, list.len) catch oom();
    for (list) |u| {
        u.retain();
        out.appendAssumeCapacity(u);
    }
    return out;
}

/// Ports `llama_kv_cache_dsa_context` (llama-kv-cache-dsa.h:84 @b1b770d90).
pub const DsaContext = struct {
    /// The index of the next ubatch to process.
    i_next: usize = 0,
    ubatches: std.ArrayList(mod.batch.Ubatch) = .empty,
    ctx_mla: ?*mod.kv_cache.KvCacheContext = null,
    ctx_lid: ?*mod.kv_cache.KvCacheContext = null,
    status: mod.memory.Status,

    /// Ports the error constructor of `llama_kv_cache_dsa_context`
    /// (llama-kv-cache-dsa.cpp:197 @b1b770d90).
    pub fn createStatus(status: mod.memory.Status) *DsaContext {
        const self = std.heap.c_allocator.create(DsaContext) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-cache constructor of `llama_kv_cache_dsa_context`
    /// (llama-kv-cache-dsa.cpp:199 @b1b770d90).
    pub fn createFull(kv: *KvCacheDsa) *DsaContext {
        const self = std.heap.c_allocator.create(DsaContext) catch oom();
        const m = kv.getMla().initFull();
        const l = kv.getLid().initFull();
        self.* = .{ .ctx_mla = m, .ctx_lid = l, .status = mod.memory.statusCombine(m.getStatus(), l.getStatus()) };
        return self;
    }

    /// Ports the update constructor of `llama_kv_cache_dsa_context`
    /// (llama-kv-cache-dsa.cpp:206 @b1b770d90).
    pub fn createUpdate(kv: *KvCacheDsa, lctx: *anyopaque, optimize: bool) *DsaContext {
        const self = std.heap.c_allocator.create(DsaContext) catch oom();
        const m = kv.getMla().initUpdate(lctx, optimize);
        const l = kv.getLid().initUpdate(lctx, optimize);
        self.* = .{ .ctx_mla = m, .ctx_lid = l, .status = mod.memory.statusCombine(m.getStatus(), l.getStatus()) };
        return self;
    }

    /// Ports the batch constructor of `llama_kv_cache_dsa_context`
    /// (llama-kv-cache-dsa.cpp:215 @b1b770d90): each cache gets its own copy
    /// of the ubatches, as the C++ copies them.
    pub fn createBatch(kv: *KvCacheDsa, sinfos_mla: std.ArrayList(mod.kv_cache.SlotInfo), sinfos_lid: std.ArrayList(mod.kv_cache.SlotInfo), ubatches: std.ArrayList(mod.batch.Ubatch)) *DsaContext {
        const self = std.heap.c_allocator.create(DsaContext) catch oom();
        const m = mod.kv_cache.KvCacheContext.createBatch(kv.getMla(), sinfos_mla, retainedCopy(ubatches.items));
        const l = mod.kv_cache.KvCacheContext.createBatch(kv.getLid(), sinfos_lid, retainedCopy(ubatches.items));
        self.* = .{ .ubatches = ubatches, .ctx_mla = m, .ctx_lid = l, .status = mod.memory.statusCombine(m.getStatus(), l.getStatus()) };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *DsaContext) void {
        releaseAll(&self.ubatches);
        if (self.ctx_mla) |m| m.destroy();
        if (self.ctx_lid) |l| l.destroy();
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_dsa_context`'s `next` (llama-kv-cache-dsa.cpp:229 @b1b770d90).
    pub fn next(self: *DsaContext) bool {
        std.debug.assert(self.status == .success);
        _ = self.ctx_mla.?.next();
        _ = self.ctx_lid.?.next();
        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_kv_cache_dsa_context`'s `apply` (llama-kv-cache-dsa.cpp:242
    /// @b1b770d90): both caches apply, whatever the first answers.
    pub fn apply(self: *DsaContext) bool {
        std.debug.assert(!mod.memory.statusIsFail(self.status));
        var res = true;
        res = self.ctx_mla.?.apply() and res;
        res = self.ctx_lid.?.apply() and res;
        return res;
    }

    /// Ports `llama_kv_cache_dsa_context`'s `get_status` (llama-kv-cache-dsa.cpp:253 @b1b770d90).
    pub fn getStatus(self: *const DsaContext) mod.memory.Status {
        return self.status;
    }

    /// Ports `llama_kv_cache_dsa_context`'s `get_ubatch` (llama-kv-cache-dsa.cpp:257 @b1b770d90).
    pub fn getUbatch(self: *const DsaContext) *const mod.batch.Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_kv_cache_dsa_context`'s `get_mla` (llama-kv-cache-dsa.cpp:263 @b1b770d90).
    pub fn getMla(self: *const DsaContext) *const mod.kv_cache.KvCacheContext {
        std.debug.assert(self.status == .success);
        return self.ctx_mla.?;
    }

    /// Ports `llama_kv_cache_dsa_context`'s `get_lid` (llama-kv-cache-dsa.cpp:269 @b1b770d90).
    pub fn getLid(self: *const DsaContext) *const mod.kv_cache.KvCacheContext {
        std.debug.assert(self.status == .success);
        return self.ctx_lid.?;
    }

    pub fn contextOf(self: *DsaContext) mod.memory.Context {
        return .{ .ptr = self, .vtable = &ctx_vtable };
    }

    const ctx_vtable: mod.memory.Context.VTable = .{
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
            fn f(p: *anyopaque) *const mod.batch.Ubatch {
                return cast(p).getUbatch();
            }
        }.f,
        .getStatus = struct {
            fn f(p: *anyopaque) mod.memory.Status {
                return cast(p).getStatus();
            }
        }.f,
        .destroy = struct {
            fn f(p: *anyopaque) void {
                cast(p).destroy();
            }
        }.f,
    };

    fn cast(p: *anyopaque) *DsaContext {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(KvCacheDsa);
    std.testing.refAllDecls(DsaContext);
}
