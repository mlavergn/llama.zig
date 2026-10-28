//! A hybrid memory whose attention half is an iSWA cache: a recurrent
//! memory for the recurrent layers, and for the attention layers one KV
//! cache for the full-attention layers and one for the sliding-window
//! layers, driven as one.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-memory-hybrid-iswa.h` -- `llama_memory_hybrid_iswa`
//!   and `llama_memory_hybrid_iswa_context`
//! - `llama.cpp/src/llama-memory-hybrid-iswa.cpp` -- their bodies
//!
//! All at v0.4.0 (`5266f24da`). The shape follows `memory_hybrid.zig`,
//! with `kv_cache_iswa.zig` in place of the single KV cache.

const std = @import("std");
const impl = @import("impl.zig");
const hparams_mod = @import("hparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const io = @import("io.zig");
const kv_cache = @import("kv_cache.zig");
const kv_cache_iswa = @import("kv_cache_iswa.zig");
const memory_recurrent = @import("memory_recurrent.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;
const Ubatch = batch.Ubatch;
const SlotInfo = kv_cache.SlotInfo;

fn oom() noreturn {
    impl.abort("llama-memory-hybrid-iswa.cpp", 0, "out of memory");
}

fn isRecr(ctx: ?*const anyopaque, il: i32) bool {
    const hp: *const Hparams = @ptrCast(@alignCast(ctx.?));
    return hp.isRecr(@intCast(il));
}

fn isNotRecr(ctx: ?*const anyopaque, il: i32) bool {
    return !isRecr(ctx, il);
}

/// Ports `llama_memory_hybrid_iswa` (llama-memory-hybrid-iswa.h:19 @5266f24da).
pub const MemoryHybridIswa = struct {
    hparams: *const Hparams,
    mem_attn: *kv_cache_iswa.KvCacheIswa,
    mem_recr: *memory_recurrent.MemoryRecurrent,

    /// Ports the constructor of `llama_memory_hybrid_iswa`
    /// (llama-memory-hybrid-iswa.cpp:11 @5266f24da). An empty filter selects
    /// the layers by `is_recr`; the iSWA cache then splits the attention
    /// layers by `is_swa`.
    ///
    /// Return: the memory, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        type_k: c.enum_ggml_type,
        type_v: c.enum_ggml_type,
        v_trans: bool,
        swa_full: bool,
        kv_size: u32,
        n_ubatch: u32,
        n_pad: u32,
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
    ) Thrown!*MemoryHybridIswa {
        const hp = &model.hparams;
        const filter_attn: memory.LayerFilter = if (filter_attn_.func == null) .{ .ctx = hp, .func = isNotRecr } else filter_attn_;
        const filter_recr: memory.LayerFilter = if (filter_recr_.func == null) .{ .ctx = hp, .func = isRecr } else filter_recr_;

        const mem_attn = try kv_cache_iswa.KvCacheIswa.create(model, null, type_k, type_v, v_trans, offload, swa_full, unified, kv_size, n_seq_max, n_ubatch, n_pad, null, filter_attn, .{}, .{}, exc);
        errdefer mem_attn.destroy();
        const mem_recr = try memory_recurrent.MemoryRecurrent.create(model, type_r, type_s, offload, rs_size, n_seq_max, n_rs_seq, filter_recr, exc);

        const self = allocator.create(MemoryHybridIswa) catch oom();
        self.* = .{ .hparams = hp, .mem_attn = mem_attn, .mem_recr = mem_recr };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *MemoryHybridIswa) void {
        self.mem_attn.destroy();
        self.mem_recr.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `init_batch`
    /// (llama-memory-hybrid-iswa.cpp:66 @5266f24da): split as the recurrent
    /// memory needs, then prepare the recurrent memory and both halves of
    /// the iSWA cache.
    pub fn initBatch(self: *MemoryHybridIswa, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *HybridIswaContext {
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
                    // sequence's last 1 + n_rs_seq tokens stay together so
                    // the rollback snapshots remain valid.
                    const unified = self.mem_attn.getBase().getNStream() == 1;
                    const n_rs_seq = self.mem_recr.n_rs_seq;
                    ub = balloc.splitEqual(n_ubatch, !unified, if (n_rs_seq > 0) n_rs_seq + 1 else 0);
                }
                if (ub.n_tokens == 0) break;
                ubatches.append(allocator, ub) catch oom();
            }

            if (balloc.getNUsed() < balloc.getNTokens()) {
                // No suitable split.
                releaseAll(&ubatches);
                break :blk;
            }

            // The recurrent batches first.
            if (!self.mem_recr.prepare(ubatches.items)) {
                impl.logError("%s: failed to prepare recurrent ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridIswaContext.createStatus(.failed_prepare);
            }

            // Then the attention cache, both halves.
            var sinfos_base = self.mem_attn.getBase().prepare(ubatches.items) orelse {
                impl.logError("%s: failed to prepare attention base ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridIswaContext.createStatus(.failed_prepare);
            };
            if (sinfos_base.items.len == 0) {
                sinfos_base.deinit(allocator);
                impl.logError("%s: failed to prepare attention base ubatches\n", .{func});
                releaseAll(&ubatches);
                return HybridIswaContext.createStatus(.failed_prepare);
            }

            var sinfos_swa = self.mem_attn.getSwa().prepare(ubatches.items) orelse {
                impl.logError("%s: failed to prepare attention swa ubatches\n", .{func});
                freeSlotInfos(&sinfos_base);
                releaseAll(&ubatches);
                return HybridIswaContext.createStatus(.failed_prepare);
            };
            if (sinfos_swa.items.len == 0) {
                sinfos_swa.deinit(allocator);
                impl.logError("%s: failed to prepare attention swa ubatches\n", .{func});
                freeSlotInfos(&sinfos_base);
                releaseAll(&ubatches);
                return HybridIswaContext.createStatus(.failed_prepare);
            }

            return HybridIswaContext.createBatch(self, sinfos_base, sinfos_swa, ubatches);
        }
        return HybridIswaContext.createStatus(.failed_prepare);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `init_full` (llama-memory-hybrid-iswa.cpp:130 @5266f24da).
    pub fn initFull(self: *MemoryHybridIswa) *HybridIswaContext {
        return HybridIswaContext.createFull(self);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `init_update` (llama-memory-hybrid-iswa.cpp:134 @5266f24da).
    pub fn initUpdate(self: *MemoryHybridIswa, lctx: *anyopaque, optimize: bool) *HybridIswaContext {
        return HybridIswaContext.createUpdate(self, lctx, optimize);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `get_can_shift`
    /// (llama-memory-hybrid-iswa.cpp:138 @5266f24da): shifting is trivial
    /// for the recurrent half.
    pub fn getCanShift(self: *const MemoryHybridIswa) bool {
        return self.mem_attn.getCanShift();
    }

    /// Ports `llama_memory_hybrid_iswa`'s `clear` (llama-memory-hybrid-iswa.cpp:143 @5266f24da).
    pub fn clear(self: *MemoryHybridIswa, data: bool) void {
        self.mem_attn.clear(data);
        self.mem_recr.clear(data);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `seq_rm` (llama-memory-hybrid-iswa.cpp:148
    /// @5266f24da): the recurrent half first, since it may refuse without
    /// changing anything.
    pub fn seqRm(self: *MemoryHybridIswa, seq_id: i32, p0: i32, p1: i32) bool {
        if (!self.mem_recr.seqRm(seq_id, p0, p1)) return false;
        return self.mem_attn.seqRm(seq_id, p0, p1);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `seq_cp` (llama-memory-hybrid-iswa.cpp:157 @5266f24da).
    pub fn seqCp(self: *MemoryHybridIswa, a: i32, b: i32, p0: i32, p1: i32) void {
        self.mem_attn.seqCp(a, b, p0, p1);
        self.mem_recr.seqCp(a, b, p0, p1);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `seq_keep` (llama-memory-hybrid-iswa.cpp:162 @5266f24da).
    pub fn seqKeep(self: *MemoryHybridIswa, seq_id: i32) void {
        self.mem_attn.seqKeep(seq_id);
        self.mem_recr.seqKeep(seq_id);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `seq_add` (llama-memory-hybrid-iswa.cpp:167 @5266f24da).
    pub fn seqAdd(self: *MemoryHybridIswa, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.mem_attn.seqAdd(seq_id, p0, p1, shift);
        self.mem_recr.seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `seq_div` (llama-memory-hybrid-iswa.cpp:172 @5266f24da).
    pub fn seqDiv(self: *MemoryHybridIswa, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.mem_attn.seqDiv(seq_id, p0, p1, d);
        self.mem_recr.seqDiv(seq_id, p0, p1, d);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `seq_pos_min`
    /// (llama-memory-hybrid-iswa.cpp:177 @5266f24da): the larger of the two
    /// minimums.
    pub fn seqPosMin(self: *const MemoryHybridIswa, seq_id: i32) i32 {
        return @max(self.mem_attn.seqPosMin(seq_id), self.mem_recr.seqPosMin(seq_id));
    }

    /// Ports `llama_memory_hybrid_iswa`'s `seq_pos_max`
    /// (llama-memory-hybrid-iswa.cpp:182 @5266f24da): the smaller of the two
    /// maximums.
    pub fn seqPosMax(self: *const MemoryHybridIswa, seq_id: i32) i32 {
        return @min(self.mem_attn.seqPosMax(seq_id), self.mem_recr.seqPosMax(seq_id));
    }

    /// Ports `llama_memory_hybrid_iswa`'s `memory_breakdown`
    /// (llama-memory-hybrid-iswa.cpp:187 @5266f24da): the attention half's
    /// sizes, with the recurrent half's added per buffer type.
    pub fn memoryBreakdown(self: *const MemoryHybridIswa, out: *std.ArrayList(memory.BuftSize)) void {
        self.mem_attn.memoryBreakdown(out);
        self.mem_recr.memoryBreakdown(out);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `state_write`
    /// (llama-memory-hybrid-iswa.cpp:195 @5266f24da). The iSWA cache applies
    /// the partial-only flag itself.
    pub fn stateWrite(self: *const MemoryHybridIswa, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        try self.mem_attn.stateWrite(w, seq_id, flags, exc);
        try self.mem_recr.stateWrite(w, seq_id, flags, exc);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `state_read` (llama-memory-hybrid-iswa.cpp:200 @5266f24da).
    pub fn stateRead(self: *MemoryHybridIswa, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        try self.mem_attn.stateRead(r, seq_id, flags, exc);
        try self.mem_recr.stateRead(r, seq_id, flags, exc);
    }

    /// Ports `llama_memory_hybrid_iswa`'s `get_mem_attn` (llama-memory-hybrid-iswa.cpp:205 @5266f24da).
    pub fn getMemAttn(self: *const MemoryHybridIswa) *kv_cache_iswa.KvCacheIswa {
        return self.mem_attn;
    }

    /// Ports `llama_memory_hybrid_iswa`'s `get_mem_recr` (llama-memory-hybrid-iswa.cpp:209 @5266f24da).
    pub fn getMemRecr(self: *const MemoryHybridIswa) *memory_recurrent.MemoryRecurrent {
        return self.mem_recr;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *MemoryHybridIswa) memory.Memory {
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

    fn from(p: *anyopaque) *MemoryHybridIswa {
        return @ptrCast(@alignCast(p));
    }
};

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

/// Ports `llama_memory_hybrid_iswa_context` (llama-memory-hybrid-iswa.h:93 @5266f24da).
pub const HybridIswaContext = struct {
    i_next: usize = 0,
    ubatches: std.ArrayList(Ubatch) = .empty,
    ctx_attn: ?*kv_cache_iswa.IswaContext = null,
    ctx_recr: ?*memory_recurrent.RecurrentContext = null,
    status: memory.Status,

    /// Ports the status constructor of `llama_memory_hybrid_iswa_context`
    /// (llama-memory-hybrid-iswa.cpp:217 @5266f24da).
    pub fn createStatus(status: memory.Status) *HybridIswaContext {
        const self = allocator.create(HybridIswaContext) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-memory constructor of `llama_memory_hybrid_iswa_context`
    /// (llama-memory-hybrid-iswa.cpp:219 @5266f24da).
    pub fn createFull(mem: *MemoryHybridIswa) *HybridIswaContext {
        const self = allocator.create(HybridIswaContext) catch oom();
        const a = mem.getMemAttn().initFull();
        const r = mem.getMemRecr().initFull();
        self.* = .{ .ctx_attn = a, .ctx_recr = r, .status = memory.statusCombine(a.getStatus(), r.getStatus()) };
        return self;
    }

    /// Ports the update constructor of `llama_memory_hybrid_iswa_context`
    /// (llama-memory-hybrid-iswa.cpp:225 @5266f24da).
    pub fn createUpdate(mem: *MemoryHybridIswa, lctx: *anyopaque, optimize: bool) *HybridIswaContext {
        const self = allocator.create(HybridIswaContext) catch oom();
        const a = mem.getMemAttn().initUpdate(lctx, optimize);
        const r = mem.getMemRecr().initUpdate(lctx, optimize);
        self.* = .{ .ctx_attn = a, .ctx_recr = r, .status = memory.statusCombine(a.getStatus(), r.getStatus()) };
        return self;
    }

    /// Ports the batch constructor of `llama_memory_hybrid_iswa_context`
    /// (llama-memory-hybrid-iswa.cpp:234 @5266f24da): each half gets its own
    /// copy of the ubatches, as the C++ copies them.
    pub fn createBatch(mem: *MemoryHybridIswa, sinfos_base: std.ArrayList(SlotInfo), sinfos_swa: std.ArrayList(SlotInfo), ubatches: std.ArrayList(Ubatch)) *HybridIswaContext {
        const self = allocator.create(HybridIswaContext) catch oom();
        const a = kv_cache_iswa.IswaContext.createBatch(mem.getMemAttn(), sinfos_base, sinfos_swa, retainedCopy(ubatches.items));
        const r = memory_recurrent.RecurrentContext.createBatch(mem.getMemRecr(), retainedCopy(ubatches.items));
        self.* = .{ .ubatches = ubatches, .ctx_attn = a, .ctx_recr = r, .status = memory.statusCombine(a.getStatus(), r.getStatus()) };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *HybridIswaContext) void {
        releaseAll(&self.ubatches);
        if (self.ctx_attn) |a| a.destroy();
        if (self.ctx_recr) |r| r.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_memory_hybrid_iswa_context`'s `next` (llama-memory-hybrid-iswa.cpp:246 @5266f24da).
    pub fn next(self: *HybridIswaContext) bool {
        std.debug.assert(self.status == .success);
        _ = self.ctx_attn.?.next();
        _ = self.ctx_recr.?.next();
        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_memory_hybrid_iswa_context`'s `apply`
    /// (llama-memory-hybrid-iswa.cpp:259 @5266f24da): `&`, not `&&`, so both
    /// halves apply.
    pub fn apply(self: *HybridIswaContext) bool {
        std.debug.assert(!memory.statusIsFail(self.status));
        var res = true;
        res = @intFromBool(res) & @intFromBool(self.ctx_attn.?.apply()) != 0;
        res = @intFromBool(res) & @intFromBool(self.ctx_recr.?.apply()) != 0;
        return res;
    }

    /// Ports `llama_memory_hybrid_iswa_context`'s `get_status` (llama-memory-hybrid-iswa.cpp:270 @5266f24da).
    pub fn getStatus(self: *const HybridIswaContext) memory.Status {
        return self.status;
    }

    /// Ports `llama_memory_hybrid_iswa_context`'s `get_ubatch` (llama-memory-hybrid-iswa.cpp:274 @5266f24da).
    pub fn getUbatch(self: *const HybridIswaContext) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_memory_hybrid_iswa_context`'s `get_attn` (llama-memory-hybrid-iswa.cpp:279 @5266f24da).
    pub fn getAttn(self: *const HybridIswaContext) *const kv_cache_iswa.IswaContext {
        return self.ctx_attn.?;
    }

    /// Ports `llama_memory_hybrid_iswa_context`'s `get_recr` (llama-memory-hybrid-iswa.cpp:283 @5266f24da).
    pub fn getRecr(self: *const HybridIswaContext) *const memory_recurrent.RecurrentContext {
        return self.ctx_recr.?;
    }

    pub fn contextOf(self: *HybridIswaContext) memory.Context {
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

    fn cast(p: *anyopaque) *HybridIswaContext {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(MemoryHybridIswa);
    std.testing.refAllDecls(HybridIswaContext);
}
