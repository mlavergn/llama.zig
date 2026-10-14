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
//! All at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("impl.zig");
const hparams_mod = @import("hparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const io = @import("io.zig");
const kv_cache = @import("kv_cache.zig");
const memory_recurrent = @import("memory_recurrent.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;
const Ubatch = batch.Ubatch;

fn oom() noreturn {
    impl.abort("llama-memory-hybrid.cpp", 0, "out of memory");
}

fn isRecr(ctx: ?*const anyopaque, il: i32) bool {
    const hp: *const Hparams = @ptrCast(@alignCast(ctx.?));
    return hp.isRecr(@intCast(il));
}

fn isNotRecr(ctx: ?*const anyopaque, il: i32) bool {
    return !isRecr(ctx, il);
}

/// Ports `llama_memory_hybrid` (llama-memory-hybrid.h:19 @c1d0e7a00).
pub const MemoryHybrid = struct {
    hparams: *const Hparams,
    mem_attn: *kv_cache.KvCache,
    mem_recr: *memory_recurrent.MemoryRecurrent,

    /// Ports the constructor of `llama_memory_hybrid`
    /// (llama-memory-hybrid.cpp:11 @c1d0e7a00). An empty filter selects the
    /// layers by `is_recr`.
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
        filter_attn_: memory.LayerFilter,
        filter_recr_: memory.LayerFilter,
        exc: *Exception,
    ) Thrown!*MemoryHybrid {
        const hp = &model.hparams;
        const filter_attn: memory.LayerFilter = if (filter_attn_.func == null) .{ .ctx = hp, .func = isNotRecr } else filter_attn_;
        const filter_recr: memory.LayerFilter = if (filter_recr_.func == null) .{ .ctx = hp, .func = isRecr } else filter_recr_;

        const mem_attn = try kv_cache.KvCache.create(model, hp, type_k, type_v, v_trans, offload, unified, kv_size, n_seq_max, n_pad, n_swa, swa_type, null, filter_attn, .{}, .{}, exc);
        errdefer mem_attn.destroy();
        const mem_recr = try memory_recurrent.MemoryRecurrent.create(model, type_r, type_s, offload, rs_size, n_seq_max, n_rs_seq, filter_recr, exc);

        const self = allocator.create(MemoryHybrid) catch oom();
        self.* = .{ .hparams = hp, .mem_attn = mem_attn, .mem_recr = mem_recr };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *MemoryHybrid) void {
        self.mem_attn.destroy();
        self.mem_recr.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_memory_hybrid`'s `init_batch` (llama-memory-hybrid.cpp:67
    /// @c1d0e7a00): split as the recurrent memory needs, then prepare both.
    pub fn initBatch(self: *MemoryHybrid, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *HybridContext {
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
                    // sequence's last 1 + n_rs_seq tokens stay together.
                    const unified = self.mem_attn.getNStream() == 1;
                    const n_rs_seq = self.mem_recr.n_rs_seq;
                    ub = balloc.splitEqual(n_ubatch, !unified, if (n_rs_seq > 0) n_rs_seq + 1 else 0);
                }
                if (ub.n_tokens == 0) break;
                ubatches.append(allocator, ub) catch oom();
            }

            if (balloc.getNUsed() < balloc.getNTokens()) {
                releaseAll(&ubatches);
                break :blk;
            }

            // The recurrent batches first.
            if (!self.mem_recr.prepare(ubatches.items)) {
                impl.logError("%s: failed to prepare recurrent ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridContext.createStatus(.failed_prepare);
            }

            // Then the attention cache.
            const heads_attn = self.mem_attn.prepare(ubatches.items) orelse {
                impl.logError("%s: failed to prepare attention ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridContext.createStatus(.failed_prepare);
            };
            if (heads_attn.items.len == 0) {
                var h = heads_attn;
                h.deinit(allocator);
                impl.logError("%s: failed to prepare attention ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridContext.createStatus(.failed_prepare);
            }
            return HybridContext.createBatch(self, heads_attn, ubatches);
        }
        return HybridContext.createStatus(.failed_prepare);
    }

    /// Ports `llama_memory_hybrid`'s `init_full` (llama-memory-hybrid.cpp:125 @c1d0e7a00).
    pub fn initFull(self: *MemoryHybrid) *HybridContext {
        return HybridContext.createFull(self);
    }

    /// Ports `llama_memory_hybrid`'s `init_update` (llama-memory-hybrid.cpp:129 @c1d0e7a00).
    pub fn initUpdate(self: *MemoryHybrid, lctx: *anyopaque, optimize: bool) *HybridContext {
        return HybridContext.createUpdate(self, lctx, optimize);
    }

    /// Ports `llama_memory_hybrid`'s `get_can_shift` (llama-memory-hybrid.cpp:133
    /// @c1d0e7a00): shifting is trivial for the recurrent half.
    pub fn getCanShift(self: *const MemoryHybrid) bool {
        return self.mem_attn.getCanShift();
    }

    /// Ports `llama_memory_hybrid`'s `clear` (llama-memory-hybrid.cpp:138 @c1d0e7a00).
    pub fn clear(self: *MemoryHybrid, data: bool) void {
        self.mem_attn.clear(data);
        self.mem_recr.clear(data);
    }

    /// Ports `llama_memory_hybrid`'s `seq_rm` (llama-memory-hybrid.cpp:143
    /// @c1d0e7a00): the recurrent half first, since it may refuse without
    /// changing anything.
    pub fn seqRm(self: *MemoryHybrid, seq_id: i32, p0: i32, p1: i32) bool {
        if (!self.mem_recr.seqRm(seq_id, p0, p1)) return false;
        return self.mem_attn.seqRm(seq_id, p0, p1);
    }

    /// Ports `llama_memory_hybrid`'s `seq_cp` (llama-memory-hybrid.cpp:152 @c1d0e7a00).
    pub fn seqCp(self: *MemoryHybrid, a: i32, b: i32, p0: i32, p1: i32) void {
        self.mem_attn.seqCp(a, b, p0, p1);
        self.mem_recr.seqCp(a, b, p0, p1);
    }

    /// Ports `llama_memory_hybrid`'s `seq_keep` (llama-memory-hybrid.cpp:157 @c1d0e7a00).
    pub fn seqKeep(self: *MemoryHybrid, seq_id: i32) void {
        self.mem_attn.seqKeep(seq_id);
        self.mem_recr.seqKeep(seq_id);
    }

    /// Ports `llama_memory_hybrid`'s `seq_add` (llama-memory-hybrid.cpp:162 @c1d0e7a00).
    pub fn seqAdd(self: *MemoryHybrid, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.mem_attn.seqAdd(seq_id, p0, p1, shift);
        self.mem_recr.seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_memory_hybrid`'s `seq_div` (llama-memory-hybrid.cpp:167 @c1d0e7a00).
    pub fn seqDiv(self: *MemoryHybrid, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.mem_attn.seqDiv(seq_id, p0, p1, d);
        self.mem_recr.seqDiv(seq_id, p0, p1, d);
    }

    /// Ports `llama_memory_hybrid`'s `seq_pos_min` (llama-memory-hybrid.cpp:172
    /// @c1d0e7a00): the larger of the two minimums.
    pub fn seqPosMin(self: *const MemoryHybrid, seq_id: i32) i32 {
        return @max(self.mem_attn.seqPosMin(seq_id), self.mem_recr.seqPosMin(seq_id));
    }

    /// Ports `llama_memory_hybrid`'s `seq_pos_max` (llama-memory-hybrid.cpp:177
    /// @c1d0e7a00): the smaller of the two maximums.
    pub fn seqPosMax(self: *const MemoryHybrid, seq_id: i32) i32 {
        return @min(self.mem_attn.seqPosMax(seq_id), self.mem_recr.seqPosMax(seq_id));
    }

    /// Ports `llama_memory_hybrid`'s `memory_breakdown` (llama-memory-hybrid.cpp:182 @c1d0e7a00).
    pub fn memoryBreakdown(self: *const MemoryHybrid, out: *std.ArrayList(memory.BuftSize)) void {
        self.mem_attn.memoryBreakdown(out);
        self.mem_recr.memoryBreakdown(out);
    }

    /// Ports `llama_memory_hybrid`'s `state_write` (llama-memory-hybrid.cpp:190 @c1d0e7a00).
    pub fn stateWrite(self: *const MemoryHybrid, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        if (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 0) try self.mem_attn.stateWrite(w, seq_id, flags, exc);
        try self.mem_recr.stateWrite(w, seq_id, flags, exc);
    }

    /// Ports `llama_memory_hybrid`'s `state_read` (llama-memory-hybrid.cpp:197 @c1d0e7a00).
    pub fn stateRead(self: *MemoryHybrid, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        if (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 0) try self.mem_attn.stateRead(r, seq_id, flags, exc);
        try self.mem_recr.stateRead(r, seq_id, flags, exc);
    }

    /// Ports `llama_memory_hybrid`'s `get_mem_attn` (llama-memory-hybrid.cpp:204 @c1d0e7a00).
    pub fn getMemAttn(self: *const MemoryHybrid) *kv_cache.KvCache {
        return self.mem_attn;
    }

    /// Ports `llama_memory_hybrid`'s `get_mem_recr` (llama-memory-hybrid.cpp:208 @c1d0e7a00).
    pub fn getMemRecr(self: *const MemoryHybrid) *memory_recurrent.MemoryRecurrent {
        return self.mem_recr;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *MemoryHybrid) memory.Memory {
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

    fn from(p: *anyopaque) *MemoryHybrid {
        return @ptrCast(@alignCast(p));
    }
};

fn releaseAll(list: *std.ArrayList(Ubatch)) void {
    for (list.items) |*u| u.release();
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

/// Ports `llama_memory_hybrid_context` (llama-memory-hybrid.h:93 @c1d0e7a00).
pub const HybridContext = struct {
    i_next: usize = 0,
    ubatches: std.ArrayList(Ubatch) = .empty,
    ctx_attn: ?*kv_cache.KvCacheContext = null,
    ctx_recr: ?*memory_recurrent.RecurrentContext = null,
    status: memory.Status,

    /// Ports the status constructor of `llama_memory_hybrid_context` (llama-memory-hybrid.cpp:212 @c1d0e7a00).
    pub fn createStatus(status: memory.Status) *HybridContext {
        const self = allocator.create(HybridContext) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-memory constructor of `llama_memory_hybrid_context` (llama-memory-hybrid.cpp:214 @c1d0e7a00).
    pub fn createFull(mem: *MemoryHybrid) *HybridContext {
        const self = allocator.create(HybridContext) catch oom();
        const a = mem.getMemAttn().initFull();
        const r = mem.getMemRecr().initFull();
        self.* = .{ .ctx_attn = a, .ctx_recr = r, .status = memory.statusCombine(a.getStatus(), r.getStatus()) };
        return self;
    }

    /// Ports the update constructor of `llama_memory_hybrid_context` (llama-memory-hybrid.cpp:220 @c1d0e7a00).
    pub fn createUpdate(mem: *MemoryHybrid, lctx: *anyopaque, optimize: bool) *HybridContext {
        const self = allocator.create(HybridContext) catch oom();
        const a = mem.getMemAttn().initUpdate(lctx, optimize);
        const r = mem.getMemRecr().initUpdate(lctx, optimize);
        self.* = .{ .ctx_attn = a, .ctx_recr = r, .status = memory.statusCombine(a.getStatus(), r.getStatus()) };
        return self;
    }

    /// Ports the batch constructor of `llama_memory_hybrid_context` (llama-memory-hybrid.cpp:229 @c1d0e7a00):
    /// each half gets its own copy of the ubatches, as the C++ copies them.
    pub fn createBatch(mem: *MemoryHybrid, sinfos_attn: std.ArrayList(kv_cache.SlotInfo), ubatches: std.ArrayList(Ubatch)) *HybridContext {
        const self = allocator.create(HybridContext) catch oom();
        const a = kv_cache.KvCacheContext.createBatch(mem.getMemAttn(), sinfos_attn, retainedCopy(ubatches.items));
        const r = memory_recurrent.RecurrentContext.createBatch(mem.getMemRecr(), retainedCopy(ubatches.items));
        self.* = .{ .ubatches = ubatches, .ctx_attn = a, .ctx_recr = r, .status = memory.statusCombine(a.getStatus(), r.getStatus()) };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *HybridContext) void {
        releaseAll(&self.ubatches);
        if (self.ctx_attn) |a| a.destroy();
        if (self.ctx_recr) |r| r.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_memory_hybrid_context`'s `next` (llama-memory-hybrid.cpp:240 @c1d0e7a00).
    pub fn next(self: *HybridContext) bool {
        std.debug.assert(self.status == .success);
        _ = self.ctx_attn.?.next();
        _ = self.ctx_recr.?.next();
        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_memory_hybrid_context`'s `apply` (llama-memory-hybrid.cpp:253 @c1d0e7a00).
    pub fn apply(self: *HybridContext) bool {
        std.debug.assert(!memory.statusIsFail(self.status));
        var res = true;
        res = res and self.ctx_attn.?.apply();
        // `&`, not `&&`: both halves apply.
        res = @intFromBool(res) & @intFromBool(self.ctx_recr.?.apply()) != 0;
        return res;
    }

    /// Ports `llama_memory_hybrid_context`'s `get_status` (llama-memory-hybrid.cpp:264 @c1d0e7a00).
    pub fn getStatus(self: *const HybridContext) memory.Status {
        return self.status;
    }

    /// Ports `llama_memory_hybrid_context`'s `get_ubatch` (llama-memory-hybrid.cpp:268 @c1d0e7a00).
    pub fn getUbatch(self: *const HybridContext) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_memory_hybrid_context`'s `get_attn` (llama-memory-hybrid.cpp:273 @c1d0e7a00).
    pub fn getAttn(self: *const HybridContext) *const kv_cache.KvCacheContext {
        return self.ctx_attn.?;
    }

    /// Ports `llama_memory_hybrid_context`'s `get_recr` (llama-memory-hybrid.cpp:277 @c1d0e7a00).
    pub fn getRecr(self: *const HybridContext) *const memory_recurrent.RecurrentContext {
        return self.ctx_recr.?;
    }

    pub fn contextOf(self: *HybridContext) memory.Context {
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
