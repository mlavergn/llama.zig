//! What a context remembers between decodes, and the per-batch views of it.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-memory.h` -- `llama_memory_params`,
//!   `llama_memory_bufs`, `llama_memory_status`, `llama_memory_context_i`
//!   and `llama_memory_i`
//! - `llama.cpp/src/llama-memory.cpp` -- `llama_memory_status_combine`,
//!   `llama_memory_status_is_fail`, `llama_memory_bufs`'s bodies,
//!   `llama_memory_buffers_writer` and `llama_memory_get_buffers`
//!
//! All at v0.9.0 (`cd959bae9`).
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

fn oom() noreturn {
    impl.abort("llama-memory.cpp", 0, "out of memory");
}

/// Mirrors `llama_memory_params` (llama-memory.h:18 @cd959bae9).
pub const Params = struct {
    type_k: c.enum_ggml_type,
    type_v: c.enum_ggml_type,
    swa_full: bool,
    ctx_type: c.enum_llama_context_type,
    /// `mem_other`: the memory this one shares cells with, or null.
    mem_other: ?Memory,
    /// Only simulate the memory allocations.
    no_alloc: bool,
};

/// Ports `llama_memory_bufs` (llama-memory.h:36 @cd959bae9): the tensor
/// contexts and backend buffers of a memory instance. With `no_alloc`, dummy
/// buffers are attached instead of allocating the tensor data, so the memory
/// use can still be measured.
pub const Bufs = struct {
    pub const Entry = struct { ctx: *c.ggml_context, buf: c.ggml_backend_buffer_t };

    bufs: std.ArrayList(Entry) = .empty,
    no_alloc: bool,

    /// Ports `llama_memory_bufs`'s `alloc_buf` (llama-memory.cpp:73
    /// @cd959bae9): allocate a single context's buffer, for the memory
    /// instances that keep their own buffer layout.
    ///
    /// Parameters:
    /// - `name`: what the failure names.
    /// - `exc`: receives the failure.
    ///
    /// Return: the buffer, the caller's; or `error.Thrown`.
    pub fn allocBuf(ctx: *c.ggml_context, buft: c.ggml_backend_buffer_type_t, no_alloc: bool, name: [*:0]const u8, exc: *Exception) Thrown!c.ggml_backend_buffer_t {
        var buf: c.ggml_backend_buffer_t = null;
        if (no_alloc) {
            buf = c.ggml_backend_buft_alloc_buffer(buft, 0); // dummy buffer
            var t: ?*c.ggml_tensor = c.ggml_get_first_tensor(ctx);
            // a dummy buffer, so that the backend scheduler won't try to
            // allocate the tensors
            while (t) |cur| : (t = c.ggml_get_next_tensor(ctx, cur)) cur.buffer = buf;
        } else {
            buf = c.ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
        }
        if (buf == null) return exc.throw(.runtime_error, "failed to allocate buffer for {s}", .{name});
        return buf;
    }

    /// Ports `llama_memory_bufs`'s `emplace` (llama-memory.cpp:89
    /// @cd959bae9): allocate the tensors of `ctx` on `buft`, or attach a
    /// dummy buffer if `no_alloc` is set. Takes `ctx` on success; on
    /// `error.Thrown` it stays the caller's.
    ///
    /// Return: the buffer, for logging; borrowed.
    pub fn emplace(self: *Bufs, ctx: *c.ggml_context, buft: c.ggml_backend_buffer_type_t, name: [*:0]const u8, exc: *Exception) Thrown!c.ggml_backend_buffer_t {
        const buf = try allocBuf(ctx, buft, self.no_alloc, name, exc);

        // initialize the buffer to avoid NaNs in the padding
        c.ggml_backend_buffer_clear(buf, 0);

        self.bufs.append(impl.allocator, .{ .ctx = ctx, .buf = buf }) catch oom();

        return buf;
    }

    /// Ports `llama_memory_bufs`'s `memory_breakdown` (llama-memory.cpp:100
    /// @cd959bae9): the memory used by the tensors, the buffer sizes or the
    /// simulated sizes if `no_alloc` is set.
    pub fn memoryBreakdown(self: *const Bufs, out: *std.ArrayList(BuftSize)) void {
        for (self.bufs.items) |e| {
            const buft = c.ggml_backend_buffer_get_type(e.buf);
            if (self.no_alloc) {
                impl.assert(c.ggml_backend_buffer_get_base(e.buf) == null, "llama-memory.cpp", 106, "ggml_backend_buffer_get_base(buf.get()) == nullptr");
                addBuftSize(out, buft, c.ggml_backend_alloc_ctx_tensors_from_buft_size(e.ctx, buft));
            } else {
                addBuftSize(out, buft, c.ggml_backend_buffer_get_size(e.buf));
            }
        }
    }

    /// Frees every buffer and context.
    pub fn deinit(self: *Bufs) void {
        for (self.bufs.items) |e| {
            c.ggml_backend_buffer_free(e.buf);
            c.ggml_free(e.ctx);
        }
        self.bufs.deinit(impl.allocator);
    }
};

/// Mirrors `llama_memory_status` (llama-memory.h:61 @cd959bae9).
pub const Status = enum(c_int) {
    success = 0,
    no_update,
    failed_prepare,
    failed_compute,
};

/// Ports `llama_memory_status_combine` (llama-memory.cpp:11 @cd959bae9): a
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

/// Ports `llama_memory_status_is_fail` (llama-memory.cpp:52 @cd959bae9).
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
    list.insert(impl.allocator, pos, .{ .buft = buft, .size = size }) catch impl.abort("llama-memory.h", 151, "out of memory");
}

/// Mirrors `llama_memory_context_i` (llama-memory.h:82 @cd959bae9): the
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

/// Mirrors `llama_memory_i` (llama-memory.h:104 @cd959bae9).
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

/// Ports `llama_memory_buffers_writer` (llama-memory.cpp:116 @cd959bae9): a
/// state sink that keeps no data, only the buffers of the tensors a state
/// write visits.
const BuffersWriter = struct {
    /// `std::set<ggml_backend_buffer_t>`: kept sorted by address.
    buffers: std.ArrayList(c.ggml_backend_buffer_t) = .empty,
    size: usize = 0,

    fn write(ptr: *anyopaque, src: []const u8, _: *Exception) Thrown!void {
        const self: *BuffersWriter = @ptrCast(@alignCast(ptr));
        self.size += src.len;
    }

    fn writeTensor(ptr: *anyopaque, tensor: *const c.ggml_tensor, _: usize, n: usize, _: *Exception) Thrown!void {
        const self: *BuffersWriter = @ptrCast(@alignCast(ptr));
        const buf = tensor.buffer;
        var at: usize = 0;
        while (at < self.buffers.items.len and @intFromPtr(self.buffers.items[at]) < @intFromPtr(buf)) at += 1;
        if (at == self.buffers.items.len or self.buffers.items[at] != buf) {
            self.buffers.insert(impl.allocator, at, buf) catch impl.abort("llama-memory.cpp", 0, "out of memory");
        }
        self.size += n;
    }

    fn nBytes(ptr: *anyopaque) usize {
        const self: *BuffersWriter = @ptrCast(@alignCast(ptr));
        return self.size;
    }

    const vtable: io.Writer.VTable = .{ .write = write, .writeTensor = writeTensor, .nBytes = nBytes };
};

/// Ports `llama_memory_get_buffers` (llama-memory.cpp:134 @cd959bae9), from
/// `llama-ext.h`: the backend buffers a state write of this memory would
/// touch; an empty cache reports none. The null memory the C++ answers with
/// an empty list is `api/context.zig`'s to answer, `mem` being non-null here.
///
/// Parameters:
/// - `mem`: the memory.
/// - `exc`: receives what the state write throws.
///
/// Return: the buffers in address order, the caller's to free with
/// `impl.allocator`; or `error.Thrown`.
pub fn getBuffers(mem: Memory, exc: *Exception) Thrown![]c.ggml_backend_buffer_t {
    var writer: BuffersWriter = .{};
    errdefer writer.buffers.deinit(impl.allocator);

    try mem.stateWrite(.{ .ptr = &writer, .vtable = &BuffersWriter.vtable }, -1, 0, exc);

    return writer.buffers.toOwnedSlice(impl.allocator) catch impl.abort("llama-memory.cpp", 0, "out of memory");
}

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
