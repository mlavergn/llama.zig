//! Saving a context's state and restoring it: to a buffer, to a file, and
//! for one sequence to a copy kept on the device.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-context.h` -- `llama_memory_buffer` and
//!   `llama_memory_buffers`
//! - `llama.cpp/src/llama-context.cpp` -- the seven `llama_io_*` stream
//!   classes, and `llama_context`'s `state_*` and `state_seq_*` methods
//!
//! All at v0.3.0 (`c1d0e7a00`).
//!
//! # Destructors
//!
//! The host and device streams do their real work in their destructors:
//! the tensor copies are queued as the state is walked and flushed when the
//! stream goes out of scope, on the exception path too. Here each is a
//! `finish` called through `defer`, at the point the C++ destroys it.

const std = @import("std");
const impl = @import("impl.zig");
const io = @import("io.zig");
const mmap = @import("mmap.zig");
const arch_mod = @import("arch.zig");
const context_mod = @import("context.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Context = context_mod.Context;
const Tensor = c.ggml_tensor;

fn oom() noreturn {
    impl.abort("llama-context.cpp", 0, "out of memory");
}

fn endOfBuffer(exc: *Exception) Thrown {
    return exc.throw(.runtime_error, "unexpectedly reached end of buffer", .{});
}

/// Ports `llama_memory_buffer` (llama-context.h:28 @c1d0e7a00): one buffer
/// type's copies of a sequence's state tensors.
pub const MemoryBuffer = struct {
    n_tensors: c_int = 0,
    total_size: usize = 0,
    buf: c.ggml_backend_buffer_t = null,
    ctx: ?*c.ggml_context = null,
    org: std.ArrayList(*Tensor) = .empty,
    cpy: std.ArrayList(*Tensor) = .empty,

    fn deinit(self: *MemoryBuffer) void {
        if (self.buf != null) c.ggml_backend_buffer_free(self.buf);
        if (self.ctx) |ctx| c.ggml_free(ctx);
        self.org.deinit(allocator);
        self.cpy.deinit(allocator);
        self.* = .{};
    }
};

/// Ports `llama_memory_buffers` (llama-context.h:40 @c1d0e7a00), a
/// `std`'s `map` from buffer type to its buffer: iterated in pointer order.
pub const MemoryBuffers = struct {
    entries: std.ArrayList(Entry) = .empty,

    pub const Entry = struct { buft: c.ggml_backend_buffer_type_t, mbuf: MemoryBuffer };

    /// `operator[]`: the entry, inserted empty if absent.
    fn at(self: *MemoryBuffers, buft: c.ggml_backend_buffer_type_t) *MemoryBuffer {
        var pos: usize = 0;
        while (pos < self.entries.items.len) : (pos += 1) {
            const k = self.entries.items[pos].buft;
            if (k == buft) return &self.entries.items[pos].mbuf;
            if (@intFromPtr(k) > @intFromPtr(buft)) break;
        }
        self.entries.insert(allocator, pos, .{ .buft = buft, .mbuf = .{} }) catch oom();
        return &self.entries.items[pos].mbuf;
    }

    fn find(self: *const MemoryBuffers, buft: c.ggml_backend_buffer_type_t) ?*const MemoryBuffer {
        for (self.entries.items) |*e| if (e.buft == buft) return &e.mbuf;
        return null;
    }

    pub fn deinit(self: *MemoryBuffers) void {
        for (self.entries.items) |*e| e.mbuf.deinit();
        self.entries.deinit(allocator);
    }
};

/// `mem_storage` (llama-context.h:374 @c1d0e7a00): each sequence's
/// on-device copy, a `std`'s `map` by sequence.
pub const MemStorage = struct {
    entries: std.ArrayList(Entry) = .empty,

    pub const Entry = struct { seq_id: i32, bufs: MemoryBuffers };

    fn at(self: *MemStorage, seq_id: i32) *MemoryBuffers {
        var pos: usize = 0;
        while (pos < self.entries.items.len) : (pos += 1) {
            const k = self.entries.items[pos].seq_id;
            if (k == seq_id) return &self.entries.items[pos].bufs;
            if (k > seq_id) break;
        }
        self.entries.insert(allocator, pos, .{ .seq_id = seq_id, .bufs = .{} }) catch oom();
        return &self.entries.items[pos].bufs;
    }

    fn contains(self: *const MemStorage, seq_id: i32) bool {
        for (self.entries.items) |e| if (e.seq_id == seq_id) return true;
        return false;
    }

    pub fn deinit(self: *MemStorage) void {
        for (self.entries.items) |*e| e.bufs.deinit();
        self.entries.deinit(allocator);
    }
};

const TensorIo = struct {
    tensor: *Tensor,
    ptr: [*]u8,
    size: usize,
    offset: usize,
};

/// Ports `llama_io_write_dummy` (llama-context.cpp:2535 @c1d0e7a00): counts
/// what would be written.
const WriteDummy = struct {
    skip_tensors: bool,
    size_written: usize = 0,

    fn writer(self: *WriteDummy) io.Writer {
        return .{ .ptr = self, .vtable = &.{ .write = write, .writeTensor = writeTensor, .nBytes = nBytes } };
    }
    fn write(ptr: *anyopaque, src: []const u8, _: *Exception) Thrown!void {
        const self: *WriteDummy = @ptrCast(@alignCast(ptr));
        self.size_written += src.len;
    }
    fn writeTensor(ptr: *anyopaque, _: *const Tensor, _: usize, size: usize, _: *Exception) Thrown!void {
        const self: *WriteDummy = @ptrCast(@alignCast(ptr));
        if (self.skip_tensors) return;
        self.size_written += size;
    }
    fn nBytes(ptr: *anyopaque) usize {
        const self: *WriteDummy = @ptrCast(@alignCast(ptr));
        return self.size_written;
    }
};

/// Ports `llama_io_write_host` (llama-context.cpp:2561 @c1d0e7a00): into a
/// host buffer, the tensors read back when the stream finishes.
const WriteHost = struct {
    ptr: [*]u8,
    buf_size: usize,
    size_written: usize = 0,
    winfos: std.ArrayList(TensorIo) = .empty,

    fn writer(self: *WriteHost) io.Writer {
        return .{ .ptr = self, .vtable = &.{ .write = write, .writeTensor = writeTensor, .nBytes = nBytes } };
    }

    /// The destructor of `llama_io_write_host` (llama-context.cpp:2566 @c1d0e7a00).
    fn finish(self: *WriteHost) void {
        for (self.winfos.items) |w| c.ggml_backend_tensor_get(w.tensor, w.ptr, w.offset, w.size);
        self.winfos.deinit(allocator);
    }

    fn write(p: *anyopaque, src: []const u8, exc: *Exception) Thrown!void {
        const self: *WriteHost = @ptrCast(@alignCast(p));
        if (src.len > self.buf_size) return endOfBuffer(exc);
        @memcpy(self.ptr[0..src.len], src);
        self.ptr += src.len;
        self.size_written += src.len;
        self.buf_size -= src.len;
    }
    fn writeTensor(p: *anyopaque, tensor: *const Tensor, offset: usize, size: usize, exc: *Exception) Thrown!void {
        const self: *WriteHost = @ptrCast(@alignCast(p));
        if (size > self.buf_size) return endOfBuffer(exc);
        self.winfos.append(allocator, .{ .tensor = @constCast(tensor), .ptr = self.ptr, .size = size, .offset = offset }) catch oom();
        self.ptr += size;
        self.size_written += size;
        self.buf_size -= size;
    }
    fn nBytes(p: *anyopaque) usize {
        const self: *WriteHost = @ptrCast(@alignCast(p));
        return self.size_written;
    }
};

/// Ports `llama_io_read_host` (llama-context.cpp:2614 @c1d0e7a00): from a
/// host buffer, the tensors written when the stream finishes.
const ReadHost = struct {
    ptr: [*]const u8,
    buf_size: usize,
    size_read: usize = 0,
    rinfos: std.ArrayList(TensorIo) = .empty,

    fn reader(self: *ReadHost) io.Reader {
        return .{ .ptr = self, .vtable = &.{ .read = read, .readTensor = readTensor, .nBytes = nBytes } };
    }

    /// The destructor of `llama_io_read_host` (llama-context.cpp:2618 @c1d0e7a00).
    fn finish(self: *ReadHost) void {
        for (self.rinfos.items) |r| c.ggml_backend_tensor_set(r.tensor, r.ptr, r.offset, r.size);
        self.rinfos.deinit(allocator);
    }

    fn read(p: *anyopaque, dst: []u8, exc: *Exception) Thrown!void {
        const self: *ReadHost = @ptrCast(@alignCast(p));
        if (dst.len > self.buf_size) return endOfBuffer(exc);
        @memcpy(dst, self.ptr[0..dst.len]);
        self.ptr += dst.len;
        self.size_read += dst.len;
        self.buf_size -= dst.len;
    }
    fn readTensor(p: *anyopaque, tensor: *Tensor, offset: usize, size: usize, exc: *Exception) Thrown!void {
        const self: *ReadHost = @ptrCast(@alignCast(p));
        if (size > self.buf_size) return endOfBuffer(exc);
        self.rinfos.append(allocator, .{ .tensor = tensor, .ptr = @constCast(self.ptr), .size = size, .offset = offset }) catch oom();
        self.ptr += size;
        self.size_read += size;
        self.buf_size -= size;
    }
    fn nBytes(p: *anyopaque) usize {
        const self: *ReadHost = @ptrCast(@alignCast(p));
        return self.size_read;
    }
};

/// Ports `llama_io_write_file` (llama-context.cpp:2666 @c1d0e7a00).
const WriteFile = struct {
    file: *mmap.File,
    size_written: usize = 0,
    temp_buffer: std.ArrayList(u8) = .empty,

    fn writer(self: *WriteFile) io.Writer {
        return .{ .ptr = self, .vtable = &.{ .write = write, .writeTensor = writeTensor, .nBytes = nBytes } };
    }
    fn deinit(self: *WriteFile) void {
        self.temp_buffer.deinit(allocator);
    }

    fn write(p: *anyopaque, src: []const u8, exc: *Exception) Thrown!void {
        const self: *WriteFile = @ptrCast(@alignCast(p));
        try self.file.writeRaw(src.ptr, src.len, exc);
        self.size_written += src.len;
    }
    fn writeTensor(p: *anyopaque, tensor: *const Tensor, offset: usize, size: usize, exc: *Exception) Thrown!void {
        const self: *WriteFile = @ptrCast(@alignCast(p));
        self.temp_buffer.resize(allocator, size) catch oom();
        c.ggml_backend_tensor_get(tensor, self.temp_buffer.items.ptr, offset, size);
        try write(p, self.temp_buffer.items, exc);
    }
    fn nBytes(p: *anyopaque) usize {
        const self: *WriteFile = @ptrCast(@alignCast(p));
        return self.size_written;
    }
};

/// Ports `llama_io_read_file` (llama-context.cpp:2691 @c1d0e7a00).
const ReadFile = struct {
    file: *mmap.File,
    size_read: usize = 0,
    temp_buffer: std.ArrayList(u8) = .empty,

    fn reader(self: *ReadFile) io.Reader {
        return .{ .ptr = self, .vtable = &.{ .read = read, .readTensor = readTensor, .nBytes = nBytes } };
    }
    fn deinit(self: *ReadFile) void {
        self.temp_buffer.deinit(allocator);
    }

    fn read(p: *anyopaque, dst: []u8, exc: *Exception) Thrown!void {
        const self: *ReadFile = @ptrCast(@alignCast(p));
        try self.file.readRaw(dst.ptr, dst.len, exc);
        self.size_read += dst.len;
    }
    fn readTensor(p: *anyopaque, tensor: *Tensor, offset: usize, size: usize, exc: *Exception) Thrown!void {
        const self: *ReadFile = @ptrCast(@alignCast(p));
        self.temp_buffer.resize(allocator, size) catch oom();
        try read(p, self.temp_buffer.items, exc);
        c.ggml_backend_tensor_set(tensor, self.temp_buffer.items.ptr, offset, size);
    }
    fn nBytes(p: *anyopaque) usize {
        const self: *ReadFile = @ptrCast(@alignCast(p));
        return self.size_read;
    }
};

/// Ports `llama_io_write_device` (llama-context.cpp:2716 @c1d0e7a00): the
/// metadata into a host buffer, the tensors copied into buffers kept on
/// their own device for the sequence.
const WriteDevice = struct {
    ptr: [*]u8,
    buf_size: usize,
    size_written: usize = 0,
    winfos: std.ArrayList(TensorIo) = .empty,
    mbufs: *MemoryBuffers,

    fn writer(self: *WriteDevice) io.Writer {
        return .{ .ptr = self, .vtable = &.{ .write = write, .writeTensor = writeTensor, .nBytes = nBytes } };
    }

    /// The destructor of `llama_io_write_device` (llama-context.cpp:2721 @c1d0e7a00): lay out one
    /// copy per tensor range, reuse the sequence's buffers where the
    /// layout is unchanged, then copy on the device.
    fn finish(self: *WriteDevice) void {
        defer self.winfos.deinit(allocator);
        var mbufs_new: MemoryBuffers = .{};
        defer mbufs_new.deinit();

        for (self.winfos.items) |w| {
            const buft = c.ggml_backend_buffer_get_type(w.tensor.buffer);
            const mb = mbufs_new.at(buft);
            mb.n_tensors += 1;
            mb.total_size += w.size;
        }
        for (mbufs_new.entries.items) |*e| {
            const n: usize = @intCast(e.mbuf.n_tensors);
            e.mbuf.ctx = c.ggml_init(.{ .mem_size = 2 * n * c.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = true });
            e.mbuf.org.ensureTotalCapacity(allocator, n) catch oom();
            e.mbuf.cpy.ensureTotalCapacity(allocator, n) catch oom();
        }
        for (self.winfos.items) |w| {
            const buft = c.ggml_backend_buffer_get_type(w.tensor.buffer);
            const n: i64 = @intCast(w.size / c.ggml_element_size(w.tensor));
            const mb = mbufs_new.at(buft);
            mb.org.append(allocator, c.ggml_view_1d(mb.ctx, w.tensor, n, w.offset)) catch oom();
            mb.cpy.append(allocator, c.ggml_new_tensor_1d(mb.ctx, w.tensor.type, n)) catch oom();
        }

        for (mbufs_new.entries.items) |*e| {
            const buft = e.buft;
            const mbuf = &e.mbuf;
            const mbuf_cur = self.mbufs.at(buft);

            var need_alloc = false;
            need_alloc = need_alloc or mbuf_cur.buf == null;
            need_alloc = need_alloc or mbuf_cur.org.items.len != mbuf.org.items.len;
            need_alloc = need_alloc or mbuf_cur.total_size != mbuf.total_size;

            if (!need_alloc) {
                for (mbuf_cur.org.items, mbuf.org.items) |org0, org1| {
                    if (!c.ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }
                    if (org0.view_src != org1.view_src or org0.view_offs != org1.view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (mbuf_cur.buf == null or mbuf_cur.total_size != mbuf.total_size) {
                    const total_size = mbuf.total_size;
                    mbuf_cur.deinit();
                    mbuf_cur.* = mbuf.*;
                    mbuf.* = .{};
                    mbuf_cur.buf = c.ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx, buft);
                    // The C++ logs the moved-from total, which a move
                    // leaves in place.
                    impl.logInfo("%s: allocated '%s' buffer %.3f MiB\n", .{ "~llama_io_write_device", c.ggml_backend_buft_name(buft), @as(f64, @floatFromInt(total_size)) / 1024.0 / 1024.0 });
                } else {
                    // The old buffer, with the new tensors in it.
                    const buf = mbuf_cur.buf;
                    mbuf_cur.buf = null;
                    mbuf_cur.deinit();
                    mbuf_cur.* = mbuf.*;
                    mbuf.* = .{};
                    var talloc = c.ggml_tallocr_new(buf);
                    for (mbuf_cur.org.items, mbuf_cur.cpy.items) |org, cpy| {
                        _ = c.ggml_backend_view_init(org);
                        _ = c.ggml_tallocr_alloc(&talloc, cpy);
                    }
                    mbuf_cur.buf = buf;
                }
            }

            for (mbuf_cur.org.items, mbuf_cur.cpy.items) |org, cpy| c.ggml_backend_tensor_copy(org, cpy);
        }
    }

    fn write(p: *anyopaque, src: []const u8, exc: *Exception) Thrown!void {
        const self: *WriteDevice = @ptrCast(@alignCast(p));
        if (src.len > self.buf_size) return endOfBuffer(exc);
        @memcpy(self.ptr[0..src.len], src);
        self.ptr += src.len;
        self.size_written += src.len;
        self.buf_size -= src.len;
    }
    fn writeTensor(p: *anyopaque, tensor: *const Tensor, offset: usize, size: usize, _: *Exception) Thrown!void {
        const self: *WriteDevice = @ptrCast(@alignCast(p));
        self.winfos.append(allocator, .{ .tensor = @constCast(tensor), .ptr = self.ptr, .size = size, .offset = offset }) catch oom();
    }
    fn nBytes(p: *anyopaque) usize {
        const self: *WriteDevice = @ptrCast(@alignCast(p));
        return self.size_written;
    }
};

/// Ports `llama_io_read_device` (llama-context.cpp:2848 @c1d0e7a00): the
/// metadata from a host buffer, the tensors copied back from the
/// sequence's device buffers.
const ReadDevice = struct {
    ptr: [*]const u8,
    buf_size: usize,
    size_read: usize = 0,
    rinfos: std.ArrayList(TensorIo) = .empty,
    mbufs: *const MemoryBuffers,

    fn reader(self: *ReadDevice) io.Reader {
        return .{ .ptr = self, .vtable = &.{ .read = read, .readTensor = readTensor, .nBytes = nBytes } };
    }

    /// The destructor of `llama_io_read_device` (llama-context.cpp:2853 @c1d0e7a00).
    fn finish(self: *ReadDevice) void {
        defer self.rinfos.deinit(allocator);
        var mbufs_new: MemoryBuffers = .{};
        defer mbufs_new.deinit();

        for (self.rinfos.items) |r| {
            const buft = c.ggml_backend_buffer_get_type(r.tensor.buffer);
            const mb = mbufs_new.at(buft);
            mb.n_tensors += 1;
            mb.total_size += r.size;
        }
        for (mbufs_new.entries.items) |*e| {
            const n: usize = @intCast(e.mbuf.n_tensors);
            e.mbuf.ctx = c.ggml_init(.{ .mem_size = n * c.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = true });
            e.mbuf.org.ensureTotalCapacity(allocator, n) catch oom();
        }
        for (self.rinfos.items) |r| {
            const buft = c.ggml_backend_buffer_get_type(r.tensor.buffer);
            const n: i64 = @intCast(r.size / c.ggml_element_size(r.tensor));
            const mb = mbufs_new.at(buft);
            const view = c.ggml_view_1d(mb.ctx, r.tensor, n, r.offset);
            mb.org.append(allocator, view) catch oom();
            _ = c.ggml_backend_view_init(view);
        }
        for (mbufs_new.entries.items) |*e| {
            // `mbufs.at()` throws out of a destructor where there is no
            // entry, which terminates.
            const mbuf_cur = self.mbufs.find(e.buft) orelse impl.abort("llama-context.cpp", 2885, "map::at:  key not found");
            if (mbuf_cur.buf == null or mbuf_cur.n_tensors != e.mbuf.n_tensors or mbuf_cur.total_size != e.mbuf.total_size) {
                c.ggml_abort("llama-context.cpp", 2888, "%s: memory buffer mismatch\n", "~llama_io_read_device");
            }
            for (mbuf_cur.cpy.items, 0..) |cpy, i| c.ggml_backend_tensor_copy(cpy, e.mbuf.org.items[i]);
        }

        impl.assert(self.buf_size == 0, "llama-context.cpp", 2899, "buf_size == 0");
    }

    fn read(p: *anyopaque, dst: []u8, exc: *Exception) Thrown!void {
        const self: *ReadDevice = @ptrCast(@alignCast(p));
        if (dst.len > self.buf_size) return endOfBuffer(exc);
        @memcpy(dst, self.ptr[0..dst.len]);
        self.ptr += dst.len;
        self.size_read += dst.len;
        self.buf_size -= dst.len;
    }
    fn readTensor(p: *anyopaque, tensor: *Tensor, offset: usize, size: usize, _: *Exception) Thrown!void {
        const self: *ReadDevice = @ptrCast(@alignCast(p));
        self.rinfos.append(allocator, .{ .tensor = tensor, .ptr = @constCast(self.ptr), .size = size, .offset = offset }) catch oom();
    }
    fn nBytes(p: *anyopaque) usize {
        const self: *ReadDevice = @ptrCast(@alignCast(p));
        return self.size_read;
    }
};

/// Ports `io_magic` (llama-context.cpp:2967 @c1d0e7a00).
const io_magic: u32 = 0xaf143cd8;

/// Ports `llama_context`'s `state_get_size` (llama-context.cpp:2937 @c1d0e7a00).
pub fn stateGetSize(ctx: *Context) usize {
    var dummy: WriteDummy = .{ .skip_tensors = false };
    var exc: Exception = .{};
    return stateWriteData(ctx, dummy.writer(), &exc) catch {
        impl.logError("%s: error getting state size: %s\n", .{ "state_get_size", exc.whatZ() });
        return 0;
    };
}

/// Ports `llama_context`'s `state_get_data` (llama-context.cpp:2947 @c1d0e7a00).
pub fn stateGetData(ctx: *Context, dst: [*]u8, size: usize) usize {
    var host: WriteHost = .{ .ptr = dst, .buf_size = size };
    defer host.finish();
    var exc: Exception = .{};
    return stateWriteData(ctx, host.writer(), &exc) catch {
        impl.logError("%s: error saving state: %s\n", .{ "state_get_data", exc.whatZ() });
        return 0;
    };
}

/// Ports `llama_context`'s `state_set_data` (llama-context.cpp:2957 @c1d0e7a00).
pub fn stateSetData(ctx: *Context, src: [*]const u8, size: usize) usize {
    var host: ReadHost = .{ .ptr = src, .buf_size = size };
    defer host.finish();
    var exc: Exception = .{};
    return stateReadData(ctx, host.reader(), &exc) catch {
        impl.logError("%s: error loading state: %s\n", .{ "state_set_data", exc.whatZ() });
        return 0;
    };
}

/// Ports `llama_context`'s `state_seq_get_size` (llama-context.cpp:2969 @c1d0e7a00).
pub fn stateSeqGetSize(ctx: *Context, seq_id: i32, flags: u32) usize {
    var dummy: WriteDummy = .{ .skip_tensors = flags & c.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0 };
    const w = dummy.writer();
    var exc: Exception = .{};
    return seqPrelude(ctx, w, seq_id, flags, &exc) catch {
        impl.logError("%s: error getting state size: %s\n", .{ "state_seq_get_size", exc.whatZ() });
        return 0;
    };
}

fn seqPrelude(ctx: *Context, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!usize {
    try w.writeValue(io_magic, exc);
    try w.writeValue(seq_id, exc);
    return stateSeqWriteData(ctx, w, seq_id, flags, exc);
}

/// Ports `llama_context`'s `state_seq_get_data` (llama-context.cpp:2982 @c1d0e7a00).
pub fn stateSeqGetData(ctx: *Context, seq_id: i32, dst: [*]u8, size: usize, flags: u32) usize {
    var exc: Exception = .{};
    if (flags & c.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0) {
        var dev: WriteDevice = .{ .ptr = dst, .buf_size = size, .mbufs = ctx.mem_storage.at(seq_id) };
        defer dev.finish();
        return seqPrelude(ctx, dev.writer(), seq_id, flags, &exc) catch {
            impl.logError("%s: error saving state: %s\n", .{ "state_seq_get_data", exc.whatZ() });
            return 0;
        };
    }
    var host: WriteHost = .{ .ptr = dst, .buf_size = size };
    defer host.finish();
    return seqPrelude(ctx, host.writer(), seq_id, flags, &exc) catch {
        impl.logError("%s: error saving state: %s\n", .{ "state_seq_get_data", exc.whatZ() });
        return 0;
    };
}

fn seqReadPrelude(ctx: *Context, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!usize {
    const magic_read = try r.readValue(u32, exc);
    if (magic_read != io_magic) return exc.throw(.runtime_error, "wrong sequence state magic", .{});
    _ = try r.readValue(i32, exc);
    return stateSeqReadData(ctx, r, seq_id, flags, exc);
}

/// Ports `llama_context`'s `state_seq_set_data` (llama-context.cpp:3001 @c1d0e7a00).
pub fn stateSeqSetData(ctx: *Context, seq_id: i32, src: [*]const u8, size: usize, flags: u32) usize {
    var exc: Exception = .{};
    if (flags & c.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0) {
        // A first stream for the magic and the source sequence, outside the
        // C++'s try: what it throws escapes.
        var probe: ReadHost = .{ .ptr = src, .buf_size = size };
        const pr = probe.reader();
        const magic_read = pr.readValue(u32, &exc) catch exc.terminate("llama-context.cpp", 3010);
        if (magic_read != io_magic) {
            exc.set(.runtime_error, "wrong sequence state magic", .{});
            exc.terminate("llama-context.cpp", 3012);
        }
        const seq_id_read = pr.readValue(i32, &exc) catch exc.terminate("llama-context.cpp", 3016);
        probe.finish();
        impl.assert(ctx.mem_storage.contains(seq_id_read), "llama-context.cpp", 3016, "mem_storage.find(seq_id_read) != mem_storage.end()");

        var dev: ReadDevice = .{ .ptr = src, .buf_size = size, .mbufs = ctx.mem_storage.at(seq_id_read) };
        defer dev.finish();
        return seqReadPrelude(ctx, dev.reader(), seq_id, flags, &exc) catch {
            impl.logError("%s: error loading state: %s\n", .{ "state_seq_set_data", exc.whatZ() });
            return 0;
        };
    }
    var host: ReadHost = .{ .ptr = src, .buf_size = size };
    defer host.finish();
    return seqReadPrelude(ctx, host.reader(), seq_id, flags, &exc) catch {
        impl.logError("%s: error loading state: %s\n", .{ "state_seq_set_data", exc.whatZ() });
        return 0;
    };
}

/// Ports `llama_context`'s `state_load_file` (llama-context.cpp:3040 @c1d0e7a00).
///
/// Return: whether it loaded, or `error.Thrown` for the caller to catch.
pub fn stateLoadFile(ctx: *Context, filepath: [:0]const u8, tokens_out: [*]c.llama_token, n_token_capacity: usize, n_token_count_out: *usize, exc: *Exception) Thrown!bool {
    const func = "state_load_file";
    var file = try mmap.File.open(filepath, "rb", false, exc);
    defer file.close();

    {
        const magic = try file.readU32(exc);
        const version = try file.readU32(exc);
        if (magic != c.LLAMA_SESSION_MAGIC or version != c.LLAMA_SESSION_VERSION) {
            impl.logError("%s: unknown (magic, version) for session file: %08x, %08x\n", .{ func, magic, version });
            return false;
        }
    }
    {
        const n_token_count = try file.readU32(exc);
        if (n_token_count > n_token_capacity) {
            impl.logError("%s: token count in session file exceeded capacity! %u > %zu\n", .{ func, n_token_count, n_token_capacity });
            return false;
        }
        try file.readRaw(tokens_out, @sizeOf(c.llama_token) * n_token_count, exc);
        n_token_count_out.* = n_token_count;
    }
    {
        const n_state_size_cur = file.size - try file.tell(exc);
        var rf: ReadFile = .{ .file = &file };
        defer rf.deinit();
        const n_read = try stateReadData(ctx, rf.reader(), exc);
        if (n_read != n_state_size_cur) {
            impl.logError("%s: did not read all of the session file data! size %zu, got %zu\n", .{ func, n_state_size_cur, n_read });
            return false;
        }
    }
    return true;
}

/// Ports `llama_context`'s `state_save_file` (llama-context.cpp:3083 @c1d0e7a00).
pub fn stateSaveFile(ctx: *Context, filepath: [:0]const u8, tokens: [*]const c.llama_token, n_token_count: usize, exc: *Exception) Thrown!bool {
    var file = try mmap.File.open(filepath, "wb", false, exc);
    defer file.close();

    try file.writeU32(c.LLAMA_SESSION_MAGIC, exc);
    try file.writeU32(c.LLAMA_SESSION_VERSION, exc);

    try file.writeU32(@truncate(n_token_count), exc);
    try file.writeRaw(tokens, @sizeOf(c.llama_token) * n_token_count, exc);

    var wf: WriteFile = .{ .file = &file };
    defer wf.deinit();
    _ = try stateWriteData(ctx, wf.writer(), exc);
    return true;
}

/// Ports `llama_context`'s `state_seq_load_file` (llama-context.cpp:3100 @c1d0e7a00).
pub fn stateSeqLoadFile(ctx: *Context, seq_id: i32, filepath: [:0]const u8, tokens_out: ?[*]c.llama_token, n_token_capacity: usize, n_token_count_out: *usize, exc: *Exception) Thrown!usize {
    const func = "state_seq_load_file";
    var file = try mmap.File.open(filepath, "rb", false, exc);
    defer file.close();

    {
        const magic = try file.readU32(exc);
        const version = try file.readU32(exc);
        if (magic != c.LLAMA_STATE_SEQ_MAGIC or version != c.LLAMA_STATE_SEQ_VERSION) {
            impl.logError("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", .{ func, magic, version });
            return 0;
        }
    }
    {
        const n_token_count = try file.readU32(exc);
        const out = tokens_out orelse {
            const n_token_max = (file.size - try file.tell(exc)) / @sizeOf(c.llama_token);
            if (n_token_count > n_token_max) {
                impl.logError("%s: token count in sequence state file exceeds the file size! %u > %zu\n", .{ func, n_token_count, n_token_max });
                return 0;
            }
            n_token_count_out.* = n_token_count;
            return file.tell(exc);
        };
        if (n_token_count > n_token_capacity) {
            impl.logError("%s: token count in sequence state file exceeded capacity! %u > %zu\n", .{ func, n_token_count, n_token_capacity });
            return 0;
        }
        try file.readRaw(out, @sizeOf(c.llama_token) * n_token_count, exc);
        n_token_count_out.* = n_token_count;
    }
    {
        const state_size = file.size - try file.tell(exc);
        var rf: ReadFile = .{ .file = &file };
        defer rf.deinit();
        const nread = try stateSeqReadData(ctx, rf.reader(), seq_id, 0, exc);
        if (nread == 0) {
            impl.logError("%s: failed to restore sequence state\n", .{func});
            return 0;
        }
        impl.assert(nread <= state_size, "llama-context.cpp", 3147, "nread <= state_size");
        impl.assert(nread + @sizeOf(u32) * 3 + @sizeOf(c.llama_token) * n_token_count_out.* == try file.tell(exc), "llama-context.cpp", 3148, "nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell()");
    }
    return file.tell(exc);
}

/// Ports `llama_context`'s `state_seq_save_file` (llama-context.cpp:3154 @c1d0e7a00).
pub fn stateSeqSaveFile(ctx: *Context, seq_id: i32, filepath: [:0]const u8, tokens: [*]const c.llama_token, n_token_count: usize, exc: *Exception) Thrown!usize {
    var file = try mmap.File.open(filepath, "wb", false, exc);
    defer file.close();

    try file.writeU32(c.LLAMA_STATE_SEQ_MAGIC, exc);
    try file.writeU32(c.LLAMA_STATE_SEQ_VERSION, exc);

    try file.writeU32(@truncate(n_token_count), exc);
    try file.writeRaw(tokens, @sizeOf(c.llama_token) * n_token_count, exc);

    var wf: WriteFile = .{ .file = &file };
    defer wf.deinit();
    _ = try stateSeqWriteData(ctx, wf.writer(), seq_id, 0, exc);

    const res = try file.tell(exc);
    impl.assert(res == @sizeOf(u32) * 3 + @sizeOf(c.llama_token) * n_token_count + wf.size_written, "llama-context.cpp", 3169, "res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes()");
    return res;
}

/// Ports `llama_context`'s `state_write_data` (llama-context.cpp:3174 @c1d0e7a00).
pub fn stateWriteData(ctx: *Context, w: io.Writer, exc: *Exception) Thrown!usize {
    const func = "state_write_data";
    impl.logDebug("%s: writing state\n", .{func});
    {
        impl.logDebug("%s: - writing model info\n", .{func});
        try w.writeString(arch_mod.archName(ctx.model.arch), exc);
    }
    if (ctx.mem) |m| {
        impl.logDebug("%s: - writing memory module\n", .{func});
        try m.stateWrite(w, -1, 0, exc);
    }
    return w.nBytes();
}

/// Ports `llama_context`'s `state_read_data` (llama-context.cpp:3194 @c1d0e7a00).
pub fn stateReadData(ctx: *Context, r: io.Reader, exc: *Exception) Thrown!usize {
    const func = "state_read_data";
    impl.logDebug("%s: reading state\n", .{func});
    {
        impl.logDebug("%s: - reading model info\n", .{func});
        const cur_arch_str = arch_mod.archName(ctx.model.arch);
        var arch_str: std.ArrayList(u8) = .empty;
        defer arch_str.deinit(allocator);
        try r.readString(&arch_str, exc);
        if (!std.mem.eql(u8, cur_arch_str, arch_str.items)) {
            return exc.throw(.runtime_error, "wrong model arch: '{s}' instead of '{s}'", .{ arch_str.items, cur_arch_str });
        }
    }
    if (ctx.mem) |m| {
        impl.logDebug("%s: - reading memory module\n", .{func});
        try m.stateRead(r, -1, 0, exc);
    }
    return r.nBytes();
}

/// Ports `llama_context`'s `state_seq_write_data` (llama-context.cpp:3220 @c1d0e7a00).
pub fn stateSeqWriteData(ctx: *Context, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!usize {
    if (ctx.mem) |m| try m.stateWrite(w, seq_id, flags, exc);
    return w.nBytes();
}

/// Ports `llama_context`'s `state_seq_read_data` (llama-context.cpp:3228 @c1d0e7a00).
pub fn stateSeqReadData(ctx: *Context, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!usize {
    if (ctx.mem) |m| try m.stateRead(r, seq_id, flags, exc);
    return r.nBytes();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the host streams refuse past the end and count what they moved" {
    var buf: [8]u8 = undefined;
    var host: WriteHost = .{ .ptr = &buf, .buf_size = buf.len };
    defer host.finish();
    var exc: Exception = .{};
    const w = host.writer();
    try w.writeValue(@as(u32, 7), &exc);
    try std.testing.expectError(error.Thrown, w.write("12345", &exc));
    try std.testing.expectEqualStrings("unexpectedly reached end of buffer", exc.what());
    try std.testing.expectEqual(@as(usize, 4), w.nBytes());

    var rh: ReadHost = .{ .ptr = &buf, .buf_size = 4 };
    defer rh.finish();
    try std.testing.expectEqual(@as(u32, 7), try rh.reader().readValue(u32, &exc));
}
