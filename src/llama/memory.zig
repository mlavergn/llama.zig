//! What a context remembers between decodes, and the per-batch views of it.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-memory.h` -- `llama_memory_params`,
//!   `llama_memory_status`, `llama_memory_context_i` and `llama_memory_i`
//! - `llama.cpp/src/llama-memory.cpp` -- `llama_memory_status_combine` and
//!   `llama_memory_status_is_fail`
//!
//! All at v0.3.0 (`c1d0e7a00`).
//!
//! # Interfaces, not classes
//!
//! The C++ makes `llama_memory_i` and `llama_memory_context_i` abstract
//! bases with one subclass per kind of memory. Here each is a pointer and a
//! table of functions, the shape `std.mem.Allocator` has; the graph code
//! that needs the concrete type recovers it with `cast`, where the C++
//! `static_cast`s.

const std = @import("std");
const impl = @import("impl.zig");
const batch = @import("batch.zig");
const io = @import("io.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;

/// Mirrors `llama_memory_params` (llama-memory.h:17 @c1d0e7a00).
pub const Params = struct {
    type_k: c.enum_ggml_type,
    type_v: c.enum_ggml_type,
    swa_full: bool,
    ctx_type: c.enum_llama_context_type,
    /// `mem_other`: the memory this one shares cells with, or null.
    mem_other: ?Memory,
};

/// Mirrors `llama_memory_status` (llama-memory.h:30 @c1d0e7a00).
pub const Status = enum(c_int) {
    success = 0,
    no_update,
    failed_prepare,
    failed_compute,
};

/// Ports `llama_memory_status_combine` (llama-memory.cpp:3 @c1d0e7a00): a
/// failure wins, else an update in either is an update.
pub fn statusCombine(s0: Status, s1: Status) Status {
    var has_update = false;
    switch (s0) {
        .success => has_update = true,
        .no_update => {},
        .failed_prepare, .failed_compute => return s0,
    }
    switch (s1) {
        .success => has_update = true,
        .no_update => {},
        .failed_prepare, .failed_compute => return s1,
    }
    return if (has_update) .success else .no_update;
}

/// Ports `llama_memory_status_is_fail` (llama-memory.cpp:44 @c1d0e7a00).
pub fn statusIsFail(status: Status) bool {
    return switch (status) {
        .success, .no_update => false,
        .failed_prepare, .failed_compute => true,
    };
}

/// One entry of `memory_breakdown`: bytes per buffer type, in the
/// `std`'s `map`'s pointer order.
pub const BuftSize = struct {
    buft: c.ggml_backend_buffer_type_t,
    size: usize,
};

/// Adds `size` to `buft`'s entry, keeping the list in pointer order, as
/// `std::map<ggml_backend_buffer_type_t, size_t>::operator[]` does.
///
/// Return: nothing.
pub fn addBuftSize(list: *std.ArrayList(BuftSize), buft: c.ggml_backend_buffer_type_t, size: usize) void {
    var pos: usize = 0;
    while (pos < list.items.len) : (pos += 1) {
        if (list.items[pos].buft == buft) {
            list.items[pos].size += size;
            return;
        }
        if (@intFromPtr(list.items[pos].buft) > @intFromPtr(buft)) break;
    }
    list.insert(impl.allocator, pos, .{ .buft = buft, .size = size }) catch impl.abort("llama-memory.h", 119, "out of memory");
}

/// Mirrors `llama_memory_context_i` (llama-memory.h:51 @c1d0e7a00): the
/// memory's view for one batch -- its ubatches, in order, and the state
/// each needs.
pub const Context = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Advances to the next ubatch; false when there is none.
        next: *const fn (ptr: *anyopaque) bool,
        /// Applies the memory state for the current ubatch; false on error.
        apply: *const fn (ptr: *anyopaque) bool,
        getUbatch: *const fn (ptr: *anyopaque) *const batch.Ubatch,
        getStatus: *const fn (ptr: *anyopaque) Status,
        destroy: *const fn (ptr: *anyopaque) void,
    };

    /// Return: false when there is no next ubatch.
    pub fn next(self: Context) bool {
        return self.vtable.next(self.ptr);
    }

    /// Return: false on error.
    pub fn apply(self: Context) bool {
        return self.vtable.apply(self.ptr);
    }

    /// Return: the current ubatch, borrowed from the context.
    pub fn getUbatch(self: Context) *const batch.Ubatch {
        return self.vtable.getUbatch(self.ptr);
    }

    /// Return: the context's status.
    pub fn getStatus(self: Context) Status {
        return self.vtable.getStatus(self.ptr);
    }

    /// Frees the context, the `unique_ptr`'s destructor.
    ///
    /// Return: nothing.
    pub fn destroy(self: Context) void {
        self.vtable.destroy(self.ptr);
    }

    /// The concrete context, where the C++ `static_cast`s.
    pub fn cast(self: Context, comptime T: type) *T {
        return @ptrCast(@alignCast(self.ptr));
    }
};

/// Mirrors `llama_memory_i` (llama-memory.h:73 @c1d0e7a00).
pub const Memory = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        initBatch: *const fn (ptr: *anyopaque, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) Context,
        initFull: *const fn (ptr: *anyopaque) Context,
        /// `lctx` is the `llama_context`, opaque here to keep this module
        /// below the context in the import graph.
        initUpdate: *const fn (ptr: *anyopaque, lctx: *anyopaque, optimize: bool) Context,
        getCanShift: *const fn (ptr: *anyopaque) bool,
        clear: *const fn (ptr: *anyopaque, data: bool) void,
        seqRm: *const fn (ptr: *anyopaque, seq_id: i32, p0: i32, p1: i32) bool,
        seqCp: *const fn (ptr: *anyopaque, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) void,
        seqKeep: *const fn (ptr: *anyopaque, seq_id: i32) void,
        seqAdd: *const fn (ptr: *anyopaque, seq_id: i32, p0: i32, p1: i32, shift: i32) void,
        seqDiv: *const fn (ptr: *anyopaque, seq_id: i32, p0: i32, p1: i32, d: c_int) void,
        seqPosMin: *const fn (ptr: *anyopaque, seq_id: i32) i32,
        seqPosMax: *const fn (ptr: *anyopaque, seq_id: i32) i32,
        /// Appends to `out`, keeping it in pointer order.
        memoryBreakdown: *const fn (ptr: *anyopaque, out: *std.ArrayList(BuftSize)) void,
        stateWrite: *const fn (ptr: *anyopaque, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void,
        stateRead: *const fn (ptr: *anyopaque, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void,
        destroy: *const fn (ptr: *anyopaque) void,
    };

    pub fn initBatch(self: Memory, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) Context {
        return self.vtable.initBatch(self.ptr, balloc, n_ubatch, embd_all);
    }
    pub fn initFull(self: Memory) Context {
        return self.vtable.initFull(self.ptr);
    }
    pub fn initUpdate(self: Memory, lctx: *anyopaque, optimize: bool) Context {
        return self.vtable.initUpdate(self.ptr, lctx, optimize);
    }
    pub fn getCanShift(self: Memory) bool {
        return self.vtable.getCanShift(self.ptr);
    }
    pub fn clear(self: Memory, data: bool) void {
        self.vtable.clear(self.ptr, data);
    }
    pub fn seqRm(self: Memory, seq_id: i32, p0: i32, p1: i32) bool {
        return self.vtable.seqRm(self.ptr, seq_id, p0, p1);
    }
    pub fn seqCp(self: Memory, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) void {
        self.vtable.seqCp(self.ptr, seq_id_src, seq_id_dst, p0, p1);
    }
    pub fn seqKeep(self: Memory, seq_id: i32) void {
        self.vtable.seqKeep(self.ptr, seq_id);
    }
    pub fn seqAdd(self: Memory, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.vtable.seqAdd(self.ptr, seq_id, p0, p1, shift);
    }
    pub fn seqDiv(self: Memory, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.vtable.seqDiv(self.ptr, seq_id, p0, p1, d);
    }
    pub fn seqPosMin(self: Memory, seq_id: i32) i32 {
        return self.vtable.seqPosMin(self.ptr, seq_id);
    }
    pub fn seqPosMax(self: Memory, seq_id: i32) i32 {
        return self.vtable.seqPosMax(self.ptr, seq_id);
    }
    pub fn memoryBreakdown(self: Memory, out: *std.ArrayList(BuftSize)) void {
        self.vtable.memoryBreakdown(self.ptr, out);
    }
    pub fn stateWrite(self: Memory, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        return self.vtable.stateWrite(self.ptr, w, seq_id, flags, exc);
    }
    pub fn stateRead(self: Memory, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        return self.vtable.stateRead(self.ptr, r, seq_id, flags, exc);
    }
    pub fn destroy(self: Memory) void {
        self.vtable.destroy(self.ptr);
    }

    /// The concrete memory, where the C++ `static_cast`s.
    pub fn cast(self: Memory, comptime T: type) *T {
        return @ptrCast(@alignCast(self.ptr));
    }

    /// The batch allocator's view: the memory's position bounds.
    pub fn posView(self: *const Memory) batch.MemoryPos {
        return .{ .ctx = self.ptr, .seq_pos_min = self.vtable.seqPosMin, .seq_pos_max = self.vtable.seqPosMax };
    }
};

/// The `std::function<bool(int32_t il)>` filters a memory takes, as a
/// context pointer and a function; `func == null` is an empty function.
pub const LayerFilter = struct {
    ctx: ?*const anyopaque = null,
    func: ?*const fn (ctx: ?*const anyopaque, il: i32) bool = null,

    pub fn call(self: LayerFilter, il: i32) bool {
        return self.func.?(self.ctx, il);
    }
};

/// The `std::function<int32_t(int32_t il)>` reuse and share callbacks.
pub const LayerMap = struct {
    ctx: ?*const anyopaque = null,
    func: ?*const fn (ctx: ?*const anyopaque, il: i32) i32 = null,

    pub fn call(self: LayerMap, il: i32) i32 {
        return self.func.?(self.ctx, il);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Params);
    std.testing.refAllDecls(BuftSize);
    std.testing.refAllDecls(Context);
    std.testing.refAllDecls(Memory);
    std.testing.refAllDecls(LayerFilter);
    std.testing.refAllDecls(LayerMap);
}

test "a failure wins a combination, else any update is an update" {
    try std.testing.expectEqual(Status.failed_compute, statusCombine(.success, .failed_compute));
    try std.testing.expectEqual(Status.failed_prepare, statusCombine(.failed_prepare, .failed_compute));
    try std.testing.expectEqual(Status.success, statusCombine(.no_update, .success));
    try std.testing.expectEqual(Status.no_update, statusCombine(.no_update, .no_update));
}
