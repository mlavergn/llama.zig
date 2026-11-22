//! A hybrid memory: a KV cache for the attention layers and a recurrent
//! memory for the others, driven as one.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-memory-hybrid.h` -- `llama_memory_hybrid` and
//!   `llama_memory_hybrid_context`
//! - `llama.cpp/src/llama-memory-hybrid.cpp` -- their bodies
//!
//! All at v1.0.0 (`b1b770d90`).
//!
//! # A context other contexts extend
//!
//! `llama_memory_hybrid_idx_context` derives from
//! `llama_memory_hybrid_context`, and the graph `static_cast`s either to the
//! base. So the context's state can be built in place (`init*`) and embedded
//! as a field, with the extending context handing out a pointer to that
//! field; `create*` and `destroy` are the standalone, heap-owned forms.

const std = @import("std");
const mod = @import("module.zig");

fn oom() noreturn {
    mod.impl.abort("llama-memory-hybrid.cpp", 0, "out of memory");
}

fn isRecr(ctx: ?*const anyopaque, il: i32) bool {
    const hp: *const mod.hparams.Hparams = @ptrCast(@alignCast(ctx.?));
    return hp.isRecr(@intCast(il));
}

fn isNotRecr(ctx: ?*const anyopaque, il: i32) bool {
    return !isRecr(ctx, il);
}

/// Ports `llama_memory_hybrid` (llama-memory-hybrid.h:19 @b1b770d90).
pub const MemoryHybrid = struct {
    hparams: *const mod.hparams.Hparams,
    mem_attn: *mod.kv_cache.KvCache,
    mem_recr: *mod.memory_recurrent.MemoryRecurrent,

    /// Ports the constructor of `llama_memory_hybrid`
    /// (llama-memory-hybrid.cpp:11 @b1b770d90). An empty filter selects the
    /// layers by `is_recr`.
    ///
    /// Return: the memory, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const mod.model.Model,
        no_alloc: bool,
        type_k: mod.impl.llama_c.enum_ggml_type,
        type_v: mod.impl.llama_c.enum_ggml_type,
        v_trans: bool,
        kv_size: u32,
        n_swa: u32,
        swa_type: mod.hparams.SwaType,
        type_r: mod.impl.llama_c.enum_ggml_type,
        type_s: mod.impl.llama_c.enum_ggml_type,
        rs_size: u32,
        n_seq_max: u32,
        n_rs_seq: u32,
        offload: bool,
        unified: bool,
        filter_attn_: mod.memory.LayerFilter,
        filter_recr_: mod.memory.LayerFilter,
        exc: *mod.impl.Exception,
    ) mod.impl.Thrown!*MemoryHybrid {
        const hp = &model.hparams;
        const filter_attn: mod.memory.LayerFilter = if (filter_attn_.func == null) .{ .ctx = hp, .func = isNotRecr } else filter_attn_;
        const filter_recr: mod.memory.LayerFilter = if (filter_recr_.func == null) .{ .ctx = hp, .func = isRecr } else filter_recr_;

        const mem_attn = try mod.kv_cache.KvCache.create(model, no_alloc, hp, type_k, type_v, v_trans, offload, unified, kv_size, n_seq_max, n_swa, swa_type, null, filter_attn, .{}, .{}, exc);
        errdefer mem_attn.destroy();
        const mem_recr = try mod.memory_recurrent.MemoryRecurrent.create(model, no_alloc, type_r, type_s, offload, rs_size, n_seq_max, n_rs_seq, filter_recr, exc);

        const self = std.heap.c_allocator.create(MemoryHybrid) catch oom();
        self.* = .{ .hparams = hp, .mem_attn = mem_attn, .mem_recr = mem_recr };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *MemoryHybrid) void {
        self.mem_attn.destroy();
        self.mem_recr.destroy();
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `llama_memory_hybrid`'s `init_batch` (llama-memory-hybrid.cpp:68
    /// @b1b770d90): split as the recurrent memory needs, then prepare both.
    pub fn initBatch(self: *MemoryHybrid, balloc: *mod.batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *HybridContext {
        const func = "init_batch";
        blk: {
            balloc.splitReset();
            var ubatches: std.ArrayList(mod.batch.Ubatch) = .empty;
            while (true) {
                var ub: mod.batch.Ubatch = undefined;
                if (embd_all) {
                    // All outputs: split by sequence.
                    ub = balloc.splitSeq(n_ubatch);
                } else {
                    // A unified KV cache splits non-sequentially; each
                    // sequence's last 1 + n_rs_seq tokens stay together.
                    const unified = self.mem_attn.getNStream() == 1;
                    const n_rs_seq = self.mem_recr.n_rs_seq;
                    ub = balloc.splitEqual(n_ubatch, !unified, if (n_rs_seq > 0) n_rs_seq + 1 else 0);
                }
                if (ub.n_tokens == 0) break;
                ubatches.append(std.heap.c_allocator, ub) catch oom();
            }

            if (balloc.getNUsed() < balloc.getNTokens()) {
                releaseAll(&ubatches);
                break :blk;
            }

            // The recurrent batches first.
            if (!self.mem_recr.prepare(ubatches.items)) {
                mod.impl.logError("%s: failed to prepare recurrent ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridContext.createStatus(.failed_prepare);
            }

            // Then the attention cache.
            const heads_attn = self.mem_attn.prepare(ubatches.items) orelse {
                mod.impl.logError("%s: failed to prepare attention ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridContext.createStatus(.failed_prepare);
            };
            if (heads_attn.items.len == 0) {
                var h = heads_attn;
                h.deinit(std.heap.c_allocator);
                mod.impl.logError("%s: failed to prepare attention ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridContext.createStatus(.failed_prepare);
            }
            return HybridContext.createBatch(self, heads_attn, ubatches);
        }
        return HybridContext.createStatus(.failed_prepare);
    }

    /// Ports `llama_memory_hybrid`'s `init_full` (llama-memory-hybrid.cpp:126 @b1b770d90).
    pub fn initFull(self: *MemoryHybrid) *HybridContext {
        return HybridContext.createFull(self);
    }

    /// Ports `llama_memory_hybrid`'s `init_update` (llama-memory-hybrid.cpp:130 @b1b770d90).
    pub fn initUpdate(self: *MemoryHybrid, lctx: *anyopaque, optimize: bool) *HybridContext {
        return HybridContext.createUpdate(self, lctx, optimize);
    }

    /// Ports `llama_memory_hybrid`'s `get_can_shift` (llama-memory-hybrid.cpp:134
    /// @b1b770d90): shifting is trivial for the recurrent half.
    pub fn getCanShift(self: *const MemoryHybrid) bool {
        return self.mem_attn.getCanShift();
    }

    /// Ports `llama_memory_hybrid`'s `clear` (llama-memory-hybrid.cpp:139 @b1b770d90).
    pub fn clear(self: *MemoryHybrid, data: bool) void {
        self.mem_attn.clear(data);
        self.mem_recr.clear(data);
    }

    /// Ports `llama_memory_hybrid`'s `seq_rm` (llama-memory-hybrid.cpp:144
    /// @b1b770d90): the recurrent half first, since it may refuse without
    /// changing anything.
    pub fn seqRm(self: *MemoryHybrid, seq_id: i32, p0: i32, p1: i32) bool {
        if (!self.mem_recr.seqRm(seq_id, p0, p1)) return false;
        return self.mem_attn.seqRm(seq_id, p0, p1);
    }

    /// Ports `llama_memory_hybrid`'s `seq_cp` (llama-memory-hybrid.cpp:153 @b1b770d90).
    pub fn seqCp(self: *MemoryHybrid, a: i32, b: i32, p0: i32, p1: i32) void {
        self.mem_attn.seqCp(a, b, p0, p1);
        self.mem_recr.seqCp(a, b, p0, p1);
    }

    /// Ports `llama_memory_hybrid`'s `seq_keep` (llama-memory-hybrid.cpp:158 @b1b770d90).
    pub fn seqKeep(self: *MemoryHybrid, seq_id: i32) void {
        self.mem_attn.seqKeep(seq_id);
        self.mem_recr.seqKeep(seq_id);
    }

    /// Ports `llama_memory_hybrid`'s `seq_add` (llama-memory-hybrid.cpp:163 @b1b770d90).
    pub fn seqAdd(self: *MemoryHybrid, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.mem_attn.seqAdd(seq_id, p0, p1, shift);
        self.mem_recr.seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_memory_hybrid`'s `seq_div` (llama-memory-hybrid.cpp:168 @b1b770d90).
    pub fn seqDiv(self: *MemoryHybrid, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.mem_attn.seqDiv(seq_id, p0, p1, d);
        self.mem_recr.seqDiv(seq_id, p0, p1, d);
    }

    /// Ports `llama_memory_hybrid`'s `seq_pos_min` (llama-memory-hybrid.cpp:173
    /// @b1b770d90): the larger of the two minimums.
    pub fn seqPosMin(self: *const MemoryHybrid, seq_id: i32) i32 {
        return @max(self.mem_attn.seqPosMin(seq_id), self.mem_recr.seqPosMin(seq_id));
    }

    /// Ports `llama_memory_hybrid`'s `seq_pos_max` (llama-memory-hybrid.cpp:178
    /// @b1b770d90): the smaller of the two maximums.
    pub fn seqPosMax(self: *const MemoryHybrid, seq_id: i32) i32 {
        return @min(self.mem_attn.seqPosMax(seq_id), self.mem_recr.seqPosMax(seq_id));
    }

    /// Ports `llama_memory_hybrid`'s `memory_breakdown` (llama-memory-hybrid.cpp:183 @b1b770d90).
    pub fn memoryBreakdown(self: *const MemoryHybrid, out: *std.ArrayList(mod.memory.BuftSize)) void {
        self.mem_attn.memoryBreakdown(out);
        self.mem_recr.memoryBreakdown(out);
    }

    /// Ports `llama_memory_hybrid`'s `state_write` (llama-memory-hybrid.cpp:191
    /// @b1b770d90): the attention cache, which skips itself for a partial
    /// state unless it is SWA, then the recurrent one.
    pub fn stateWrite(self: *const MemoryHybrid, w: mod.io.Writer, seq_id: i32, flags: u32, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        try self.mem_attn.stateWrite(w, seq_id, flags, exc);
        try self.mem_recr.stateWrite(w, seq_id, flags, exc);
    }

    /// Ports `llama_memory_hybrid`'s `state_read` (llama-memory-hybrid.cpp:196
    /// @b1b770d90): a failed restore leaves the sequence empty in both
    /// caches.
    pub fn stateRead(self: *MemoryHybrid, r: mod.io.Reader, seq_id: i32, flags: u32, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        try self.mem_attn.stateRead(r, seq_id, flags, exc);
        self.mem_recr.stateRead(r, seq_id, flags, exc) catch |err| {
            self.mem_attn.stateClear(seq_id);
            return err;
        };
    }

    /// Ports `llama_memory_hybrid`'s `get_mem_attn` (llama-memory-hybrid.cpp:209 @b1b770d90).
    pub fn getMemAttn(self: *const MemoryHybrid) *mod.kv_cache.KvCache {
        return self.mem_attn;
    }

    /// Ports `llama_memory_hybrid`'s `get_mem_recr` (llama-memory-hybrid.cpp:213 @b1b770d90).
    pub fn getMemRecr(self: *const MemoryHybrid) *mod.memory_recurrent.MemoryRecurrent {
        return self.mem_recr;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *MemoryHybrid) mod.memory.Memory {
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

    fn from(p: *anyopaque) *MemoryHybrid {
        return @ptrCast(@alignCast(p));
    }
};

/// Releases every ubatch of the list, and the list.
pub fn releaseAll(list: *std.ArrayList(mod.batch.Ubatch)) void {
    for (list.items) |*u| u.release();
    list.deinit(std.heap.c_allocator);
}

/// A copy of the list holding its own reference to every ubatch, as a
/// `std::vector<llama_ubatch>` copy shares the ubatches' data.
pub fn retainedCopy(list: []const mod.batch.Ubatch) std.ArrayList(mod.batch.Ubatch) {
    var out: std.ArrayList(mod.batch.Ubatch) = .empty;
    out.ensureTotalCapacity(std.heap.c_allocator, list.len) catch oom();
    for (list) |u| {
        u.retain();
        out.appendAssumeCapacity(u);
    }
    return out;
}

/// Ports `llama_memory_hybrid_context` (llama-memory-hybrid.h:93 @b1b770d90).
pub const HybridContext = struct {
    i_next: usize = 0,
    ubatches: std.ArrayList(mod.batch.Ubatch) = .empty,
    ctx_attn: ?*mod.kv_cache.KvCacheContext = null,
    ctx_recr: ?*mod.memory_recurrent.RecurrentContext = null,
    status: mod.memory.Status,

    /// Ports the status constructor of `llama_memory_hybrid_context` (llama-memory-hybrid.cpp:217 @b1b770d90).
    pub fn initStatus(status: mod.memory.Status) HybridContext {
        return .{ .status = status };
    }

    /// Ports the full-memory constructor of `llama_memory_hybrid_context` (llama-memory-hybrid.cpp:219 @b1b770d90).
    pub fn initFull(mem: *MemoryHybrid) HybridContext {
        const a = mem.getMemAttn().initFull();
        const r = mem.getMemRecr().initFull();
        return .{ .ctx_attn = a, .ctx_recr = r, .status = mod.memory.statusCombine(a.getStatus(), r.getStatus()) };
    }

    /// Ports the update constructor of `llama_memory_hybrid_context` (llama-memory-hybrid.cpp:225 @b1b770d90).
    pub fn initUpdate(mem: *MemoryHybrid, lctx: *anyopaque, optimize: bool) HybridContext {
        const a = mem.getMemAttn().initUpdate(lctx, optimize);
        const r = mem.getMemRecr().initUpdate(lctx, optimize);
        return .{ .ctx_attn = a, .ctx_recr = r, .status = mod.memory.statusCombine(a.getStatus(), r.getStatus()) };
    }

    /// Ports the batch constructor of `llama_memory_hybrid_context` (llama-memory-hybrid.cpp:234 @b1b770d90):
    /// each half gets its own copy of the ubatches, as the C++ copies them.
    /// Takes both lists.
    pub fn initBatch(mem: *MemoryHybrid, sinfos_attn: std.ArrayList(mod.kv_cache.SlotInfo), ubatches: std.ArrayList(mod.batch.Ubatch)) HybridContext {
        const a = mod.kv_cache.KvCacheContext.createBatch(mem.getMemAttn(), sinfos_attn, retainedCopy(ubatches.items));
        const r = mod.memory_recurrent.RecurrentContext.createBatch(mem.getMemRecr(), retainedCopy(ubatches.items));
        return .{ .ubatches = ubatches, .ctx_attn = a, .ctx_recr = r, .status = mod.memory.statusCombine(a.getStatus(), r.getStatus()) };
    }

    fn boxed(ctx: HybridContext) *HybridContext {
        const self = std.heap.c_allocator.create(HybridContext) catch oom();
        self.* = ctx;
        return self;
    }

    /// `initStatus` on the heap.
    pub fn createStatus(status: mod.memory.Status) *HybridContext {
        return boxed(initStatus(status));
    }

    /// `initFull` on the heap.
    pub fn createFull(mem: *MemoryHybrid) *HybridContext {
        return boxed(initFull(mem));
    }

    /// `initUpdate` on the heap.
    pub fn createUpdate(mem: *MemoryHybrid, lctx: *anyopaque, optimize: bool) *HybridContext {
        return boxed(initUpdate(mem, lctx, optimize));
    }

    /// `initBatch` on the heap.
    pub fn createBatch(mem: *MemoryHybrid, sinfos_attn: std.ArrayList(mod.kv_cache.SlotInfo), ubatches: std.ArrayList(mod.batch.Ubatch)) *HybridContext {
        return boxed(initBatch(mem, sinfos_attn, ubatches));
    }

    /// Frees what the context holds, leaving the context itself.
    ///
    /// Return: nothing.
    pub fn deinit(self: *HybridContext) void {
        releaseAll(&self.ubatches);
        if (self.ctx_attn) |a| a.destroy();
        if (self.ctx_recr) |r| r.destroy();
        self.* = undefined;
    }

    /// Return: nothing.
    pub fn destroy(self: *HybridContext) void {
        self.deinit();
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `llama_memory_hybrid_context`'s `next` (llama-memory-hybrid.cpp:245 @b1b770d90).
    pub fn next(self: *HybridContext) bool {
        std.debug.assert(self.status == .success);
        _ = self.ctx_attn.?.next();
        _ = self.ctx_recr.?.next();
        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_memory_hybrid_context`'s `apply` (llama-memory-hybrid.cpp:258 @b1b770d90).
    pub fn apply(self: *HybridContext) bool {
        std.debug.assert(!mod.memory.statusIsFail(self.status));
        var res = true;
        res = res and self.ctx_attn.?.apply();
        // `&`, not `&&`: both halves apply.
        res = @intFromBool(res) & @intFromBool(self.ctx_recr.?.apply()) != 0;
        return res;
    }

    /// Ports `llama_memory_hybrid_context`'s `get_status` (llama-memory-hybrid.cpp:269 @b1b770d90).
    pub fn getStatus(self: *const HybridContext) mod.memory.Status {
        return self.status;
    }

    /// Ports `llama_memory_hybrid_context`'s `get_ubatch` (llama-memory-hybrid.cpp:273 @b1b770d90).
    pub fn getUbatch(self: *const HybridContext) *const mod.batch.Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_memory_hybrid_context`'s `get_attn` (llama-memory-hybrid.cpp:278 @b1b770d90).
    pub fn getAttn(self: *const HybridContext) *const mod.kv_cache.KvCacheContext {
        return self.ctx_attn.?;
    }

    /// Ports `llama_memory_hybrid_context`'s `get_recr` (llama-memory-hybrid.cpp:282 @b1b770d90).
    pub fn getRecr(self: *const HybridContext) *const mod.memory_recurrent.RecurrentContext {
        return self.ctx_recr.?;
    }

    pub fn contextOf(self: *HybridContext) mod.memory.Context {
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

    fn cast(p: *anyopaque) *HybridContext {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(MemoryHybrid);
    std.testing.refAllDecls(HybridContext);
}
