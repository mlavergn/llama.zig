//! The backend abstraction: buffer types, buffers, backends, devices,
//! registries, events, and the utilities that copy a graph between backends.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-backend.cpp` (v0.3.0, `c1d0e7a00`),
//! together with `src/ggml/backend_sched.zig`, which takes the scheduler. Each
//! declaration below names the C++ it replaces and the line it began at. This
//! file exports the same C symbols with the same signatures, so the C and C++
//! that call into it link unchanged.
//!
//! # Split across two files
//!
//! `ggml-backend.cpp` is 2,443 lines and divides cleanly in two: roughly 1,100
//! lines of thin dispatch over the `iface` vtables — which is what this file
//! is — and roughly 1,300 lines of graph scheduler, which is
//! `backend_sched.zig`. The C marks the boundary itself with a `// scheduler`
//! comment at line 750. Keeping them together would make one 2,500-line Zig
//! file whose two halves have nothing to say to each other.
//!
//! # The contract is 102 symbols, not 82
//!
//! `PLAN.md` records 82. Measured against `zig c++ -c` at the pinned commit,
//! the unmangled export count across both halves is **102**; of 243 total
//! exports the rest are template instantiations and inline functions. **None
//! of the mangled names is referenced from any other object** in `libggml.a`
//! or `libllama.a` — checked directly, not assumed — so the Stage 3 swap
//! mechanism applies without qualification.
//!
//! # It is C in a `.cpp` file
//!
//! The first line of the C++ is `// Note: porting this file to C++ is a work
//! in progress`, and it means it. The whole translation unit contains **zero
//! `throw`, zero `catch`**, no `std::string`, no `std::map`, no
//! `std::function`, no `std::unique_ptr`, no templates and no virtual
//! functions. There are exactly two `std::vector`s, both local to
//! `ggml_backend_sched_compute_splits`, and one lambda beside them. So despite
//! being the largest C++ unit in this group and the one the whole inference
//! path runs through, it ports more like `ggml-alloc.c` did than like
//! `gguf.cpp`.
//!
//! What it does use is C++ `new`/`delete` for one struct
//! (`ggml_backend_buffer`), designated-initialiser vtable constants, and
//! function-local `static` storage for the two CPU buffer types.

const std = @import("std");
const impl = @import("impl.zig");
const c = impl.c;

/// The heap the rest of ggml uses. A function entered through a C ABI has no
/// allocator to be handed one, the same reasoning as `backend_reg.zig` and
/// `gguf.zig`.
const allocator = std.heap.c_allocator;

// -----------------------------------------------------------------------------
// Backend buffer type
//
// Every entry point here is a null check and a jump through `buft->iface`. The
// only interest is which members are optional and what they default to.

/// Ports `ggml_backend_buft_name` (src/ggml-backend.cpp:33 @c1d0e7a00).
pub export fn ggml_backend_buft_name(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    impl.assert(buft != null, "buft");
    return buft.*.iface.get_name.?(buft);
}

/// Ports `ggml_backend_buft_alloc_buffer` (src/ggml-backend.cpp:38 @c1d0e7a00).
///
/// Return: a buffer of `size` bytes, or a dummy zero-sized buffer when `size`
/// is 0 — the C returns one rather than asking the backend for nothing.
pub export fn ggml_backend_buft_alloc_buffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    impl.assert(buft != null, "buft");
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, std.mem.zeroes(c.ggml_backend_buffer_i), null, 0);
    }
    return buft.*.iface.alloc_buffer.?(buft, size);
}

/// Ports `ggml_backend_buft_get_alignment` (src/ggml-backend.cpp:47 @c1d0e7a00).
pub export fn ggml_backend_buft_get_alignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    impl.assert(buft != null, "buft");
    return buft.*.iface.get_alignment.?(buft);
}

/// Ports `ggml_backend_buft_get_max_size` (src/ggml-backend.cpp:52 @c1d0e7a00).
///
/// Return: the backend's limit, or `SIZE_MAX` — `get_max_size` is optional.
pub export fn ggml_backend_buft_get_max_size(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    impl.assert(buft != null, "buft");
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft.*.iface.get_max_size) |f| {
        return f(buft);
    }
    return std.math.maxInt(usize);
}

/// Ports `ggml_backend_buft_get_alloc_size` (src/ggml-backend.cpp:61 @c1d0e7a00).
///
/// Return: the bytes this buffer type needs for `tensor`, which may exceed
/// `ggml_nbytes` for a type that pads. Defaults to `ggml_nbytes`.
pub export fn ggml_backend_buft_get_alloc_size(buft: c.ggml_backend_buffer_type_t, tensor: [*c]const c.ggml_tensor) callconv(.c) usize {
    impl.assert(buft != null, "buft");
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft.*.iface.get_alloc_size) |f| {
        const size = f(buft, tensor);
        // The C uses plain `assert`, not GGML_ASSERT, so this one is compiled
        // out under NDEBUG on the C side. `impl.assert` is always live; the
        // condition is a backend-implementation invariant and holds.
        impl.assert(size >= c.ggml_nbytes(tensor), "size >= ggml_nbytes(tensor)");
        return size;
    }
    return c.ggml_nbytes(tensor);
}

/// Ports `ggml_backend_buft_is_host` (src/ggml-backend.cpp:72 @c1d0e7a00).
pub export fn ggml_backend_buft_is_host(buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    impl.assert(buft != null, "buft");
    if (buft.*.iface.is_host) |f| {
        return f(buft);
    }
    return false;
}

/// Ports `ggml_backend_buft_get_device` (src/ggml-backend.cpp:80 @c1d0e7a00).
pub export fn ggml_backend_buft_get_device(buft: c.ggml_backend_buffer_type_t) callconv(.c) c.ggml_backend_dev_t {
    impl.assert(buft != null, "buft");
    return buft.*.device;
}

// -----------------------------------------------------------------------------
// Backend buffer

/// Ports `ggml_backend_buffer_init` (src/ggml-backend.cpp:87 @c1d0e7a00).
///
/// The C++ uses `new ggml_backend_buffer{...}`; this is that allocation, and
/// `ggml_backend_buffer_free` is the matching `delete`.
///
/// Return: a buffer owned by the caller, released with
/// `ggml_backend_buffer_free`.
pub export fn ggml_backend_buffer_init(
    buft: c.ggml_backend_buffer_type_t,
    iface: c.ggml_backend_buffer_i,
    context: ?*anyopaque,
    size: usize,
) callconv(.c) c.ggml_backend_buffer_t {
    const buffer = allocator.create(c.struct_ggml_backend_buffer) catch impl.abort("backend: out of memory");
    buffer.* = .{
        .iface = iface,
        .buft = buft,
        .context = context,
        .size = size,
        .usage = c.GGML_BACKEND_BUFFER_USAGE_ANY,
    };
    return buffer;
}

/// Ports `ggml_backend_buffer_name` (src/ggml-backend.cpp:103 @c1d0e7a00).
pub export fn ggml_backend_buffer_name(buffer: c.ggml_backend_buffer_t) callconv(.c) [*c]const u8 {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

/// Ports `ggml_backend_buffer_free` (src/ggml-backend.cpp:107 @c1d0e7a00).
///
/// Return: nothing. Null is accepted, as the C++'s `delete` is.
pub export fn ggml_backend_buffer_free(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    if (buffer == null) {
        return;
    }
    if (buffer.*.iface.free_buffer) |f| {
        f(buffer);
    }
    allocator.destroy(@as(*c.struct_ggml_backend_buffer, @ptrCast(buffer)));
}

/// Ports `ggml_backend_buffer_get_size` (src/ggml-backend.cpp:118 @c1d0e7a00).
pub export fn ggml_backend_buffer_get_size(buffer: c.ggml_backend_buffer_t) callconv(.c) usize {
    impl.assert(buffer != null, "buffer");
    return buffer.*.size;
}

/// Ports `ggml_backend_buffer_get_base` (src/ggml-backend.cpp:123 @c1d0e7a00).
///
/// Return: the buffer's base address, or null for a zero-sized non-meta buffer
/// or a buffer type with no `get_base`. The C carries a `FIXME` here about
/// multi-buffers having a non-zero size; it is reproduced as written.
pub export fn ggml_backend_buffer_get_base(buffer: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    impl.assert(buffer != null, "buffer");
    // get_base is optional if the buffer is zero-sized
    if (!c.ggml_backend_buffer_is_meta(buffer) and buffer.*.size == 0) {
        return null;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above
    // comment get_base is not optional, I don't know whether the above comment
    // is correct
    const get_base = buffer.*.iface.get_base orelse return null;

    const base = get_base(buffer);

    impl.assert(base != null, "backend buffer base cannot be NULL");

    return base;
}

/// Ports `ggml_backend_buffer_init_tensor` (src/ggml-backend.cpp:143 @c1d0e7a00).
pub export fn ggml_backend_buffer_init_tensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor) callconv(.c) c.enum_ggml_status {
    impl.assert(buffer != null, "buffer");
    // init_tensor is optional
    if (buffer.*.iface.init_tensor) |f| {
        return f(buffer, tensor);
    }
    return c.GGML_STATUS_SUCCESS;
}

/// Ports `ggml_backend_buffer_clear` (src/ggml-backend.cpp:152 @c1d0e7a00).
pub export fn ggml_backend_buffer_clear(buffer: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    impl.assert(buffer != null, "buffer");
    // clear is optional if the buffer is zero-sized
    if (buffer.*.size == 0) {
        return;
    }
    buffer.*.iface.clear.?(buffer, value);
}

/// Ports `ggml_backend_buffer_get_alignment` (src/ggml-backend.cpp:162 @c1d0e7a00).
pub export fn ggml_backend_buffer_get_alignment(buffer: c.ggml_backend_buffer_t) callconv(.c) usize {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

/// Ports `ggml_backend_buffer_get_max_size` (src/ggml-backend.cpp:166 @c1d0e7a00).
pub export fn ggml_backend_buffer_get_max_size(buffer: c.ggml_backend_buffer_t) callconv(.c) usize {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

/// Ports `ggml_backend_buffer_get_alloc_size` (src/ggml-backend.cpp:170 @c1d0e7a00).
pub export fn ggml_backend_buffer_get_alloc_size(buffer: c.ggml_backend_buffer_t, tensor: [*c]const c.ggml_tensor) callconv(.c) usize {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

/// Ports `ggml_backend_buffer_is_host` (src/ggml-backend.cpp:174 @c1d0e7a00).
pub export fn ggml_backend_buffer_is_host(buffer: c.ggml_backend_buffer_t) callconv(.c) bool {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

/// Ports `ggml_backend_buffer_set_usage` (src/ggml-backend.cpp:178 @c1d0e7a00).
///
/// A multi-buffer forwards the usage to every buffer it owns. The C marks that
/// special case `FIXME: add a generic callback to the buffer interface`.
pub export fn ggml_backend_buffer_set_usage(buffer: c.ggml_backend_buffer_t, usage: c.enum_ggml_backend_buffer_usage) callconv(.c) void {
    impl.assert(buffer != null, "buffer");
    buffer.*.usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

/// Ports `ggml_backend_buffer_get_usage` (src/ggml-backend.cpp:188 @c1d0e7a00).
pub export fn ggml_backend_buffer_get_usage(buffer: c.ggml_backend_buffer_t) callconv(.c) c.enum_ggml_backend_buffer_usage {
    impl.assert(buffer != null, "buffer");
    return buffer.*.usage;
}

/// Ports `ggml_backend_buffer_get_type` (src/ggml-backend.cpp:193 @c1d0e7a00).
pub export fn ggml_backend_buffer_get_type(buffer: c.ggml_backend_buffer_t) callconv(.c) c.ggml_backend_buffer_type_t {
    impl.assert(buffer != null, "buffer");
    return buffer.*.buft;
}

/// Ports `ggml_backend_buffer_reset` (src/ggml-backend.cpp:198 @c1d0e7a00).
pub export fn ggml_backend_buffer_reset(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    impl.assert(buffer != null, "buffer");
    if (buffer.*.iface.reset) |f| {
        f(buffer);
    }
}

/// Ports `ggml_backend_buffer_copy_tensor` (src/ggml-backend.cpp:205 @c1d0e7a00).
///
/// Return: true if the destination's buffer performed the copy; false if it
/// has no `cpy_tensor`, leaving the caller to fall back.
pub export fn ggml_backend_buffer_copy_tensor(src: [*c]const c.ggml_tensor, dst: [*c]c.ggml_tensor) callconv(.c) bool {
    const dst_buf = if (dst.*.view_src != null) dst.*.view_src.*.buffer else dst.*.buffer;
    if (dst_buf.*.iface.cpy_tensor) |f| {
        return f(dst_buf, src, dst);
    }
    return false;
}

// -----------------------------------------------------------------------------
// Backend

/// Ports `ggml_backend_guid` (src/ggml-backend.cpp:215 @c1d0e7a00).
pub export fn ggml_backend_guid(backend: c.ggml_backend_t) callconv(.c) c.ggml_guid_t {
    if (backend == null) {
        return null;
    }
    return backend.*.guid;
}

/// Ports `ggml_backend_name` (src/ggml-backend.cpp:222 @c1d0e7a00).
///
/// Return: the backend's name, or the literal `"NULL"` for a null backend —
/// this is a logging helper and does not assert.
pub export fn ggml_backend_name(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    if (backend == null) {
        return "NULL";
    }
    return backend.*.iface.get_name.?(backend);
}

/// Ports `ggml_backend_free` (src/ggml-backend.cpp:229 @c1d0e7a00).
pub export fn ggml_backend_free(backend: c.ggml_backend_t) callconv(.c) void {
    if (backend == null) {
        return;
    }
    backend.*.iface.free.?(backend);
}

/// Ports `ggml_backend_get_default_buffer_type` (src/ggml-backend.cpp:237 @c1d0e7a00).
pub export fn ggml_backend_get_default_buffer_type(backend: c.ggml_backend_t) callconv(.c) c.ggml_backend_buffer_type_t {
    impl.assert(backend != null, "backend");
    return ggml_backend_dev_buffer_type(backend.*.device);
}

/// Ports `ggml_backend_alloc_buffer` (src/ggml-backend.cpp:242 @c1d0e7a00).
pub export fn ggml_backend_alloc_buffer(backend: c.ggml_backend_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

/// Ports `ggml_backend_get_alignment` (src/ggml-backend.cpp:246 @c1d0e7a00).
pub export fn ggml_backend_get_alignment(backend: c.ggml_backend_t) callconv(.c) usize {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

/// Ports `ggml_backend_get_max_size` (src/ggml-backend.cpp:250 @c1d0e7a00).
pub export fn ggml_backend_get_max_size(backend: c.ggml_backend_t) callconv(.c) usize {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

/// Ports `ggml_backend_tensor_set_async` (src/ggml-backend.cpp:254 @c1d0e7a00).
///
/// Falls back to a synchronize plus a blocking set when the backend has no
/// async path.
pub export fn ggml_backend_tensor_set_async(
    backend: c.ggml_backend_t,
    tensor: [*c]c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    impl.assert(backend != null, "backend");
    impl.assert(tensor != null, "tensor");
    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + size <= c.ggml_nbytes(tensor), "tensor write out of bounds");

    if (backend.*.iface.set_tensor_async) |f| {
        f(backend, tensor, data, offset, size);
    } else {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    }
}

/// Ports `ggml_backend_tensor_get_async` (src/ggml-backend.cpp:268 @c1d0e7a00).
pub export fn ggml_backend_tensor_get_async(
    backend: c.ggml_backend_t,
    tensor: [*c]const c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    impl.assert(backend != null, "backend");
    impl.assert(tensor != null, "tensor");
    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + size <= c.ggml_nbytes(tensor), "tensor read out of bounds");

    if (backend.*.iface.get_tensor_async) |f| {
        f(backend, tensor, data, offset, size);
    } else {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    }
}

/// Ports `ggml_backend_tensor_set_2d_async` (src/ggml-backend.cpp:282 @c1d0e7a00).
///
/// A strided batch of `n_copies` writes. Falls back to a loop of 1-D async
/// writes when the backend has no 2-D path or there is at most one copy —
/// note the bounds assertion is only reached on the strided path, as in the C.
pub export fn ggml_backend_tensor_set_2d_async(
    backend: c.ggml_backend_t,
    tensor: [*c]c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) callconv(.c) void {
    impl.assert(backend != null, "backend");
    impl.assert(tensor != null, "tensor");
    impl.assert(tensor.*.data != null, "tensor not allocated");

    if (n_copies <= 1 or backend.*.iface.set_tensor_2d_async == null) {
        for (0..n_copies) |i| {
            ggml_backend_tensor_set_async(
                backend,
                tensor,
                @as([*]const u8, @ptrCast(data.?)) + i * stride_data,
                offset + i * stride_tensor,
                size,
            );
        }
        return;
    }
    if (size == 0) {
        return;
    }

    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + (n_copies - 1) * stride_tensor + size <= c.ggml_nbytes(tensor), "tensor write out of bounds");
    backend.*.iface.set_tensor_2d_async.?(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

/// Ports `ggml_backend_tensor_get_2d_async` (src/ggml-backend.cpp:303 @c1d0e7a00).
pub export fn ggml_backend_tensor_get_2d_async(
    backend: c.ggml_backend_t,
    tensor: [*c]const c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) callconv(.c) void {
    impl.assert(backend != null, "backend");
    impl.assert(tensor != null, "tensor");
    impl.assert(tensor.*.data != null, "tensor not allocated");

    if (n_copies <= 1 or backend.*.iface.get_tensor_2d_async == null) {
        for (0..n_copies) |i| {
            ggml_backend_tensor_get_async(
                backend,
                tensor,
                @as([*]u8, @ptrCast(data.?)) + i * stride_data,
                offset + i * stride_tensor,
                size,
            );
        }
        return;
    }
    if (size == 0) {
        return;
    }

    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + (n_copies - 1) * stride_tensor + size <= c.ggml_nbytes(tensor), "tensor read out of bounds");
    backend.*.iface.get_tensor_2d_async.?(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

/// Ports `ggml_backend_tensor_set` (src/ggml-backend.cpp:324 @c1d0e7a00).
///
/// Writes through the *view source's* buffer when the tensor is a view, which
/// is the rule everywhere in this file.
pub export fn ggml_backend_tensor_set(tensor: [*c]c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
    impl.assert(buf != null, "tensor buffer not set");

    if (size == 0) {
        return;
    }

    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + size <= c.ggml_nbytes(tensor), "tensor write out of bounds");

    buf.*.iface.set_tensor.?(buf, tensor, data, offset, size);
}

/// Ports `ggml_backend_tensor_get` (src/ggml-backend.cpp:339 @c1d0e7a00).
pub export fn ggml_backend_tensor_get(tensor: [*c]const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
    impl.assert(buf != null, "tensor buffer not set");

    if (size == 0) {
        return;
    }

    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + size <= c.ggml_nbytes(tensor), "tensor read out of bounds");

    buf.*.iface.get_tensor.?(buf, tensor, data, offset, size);
}

/// Ports `ggml_backend_tensor_set_2d` (src/ggml-backend.cpp:354 @c1d0e7a00).
pub export fn ggml_backend_tensor_set_2d(
    tensor: [*c]c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) callconv(.c) void {
    impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
    impl.assert(buf != null, "tensor buffer not set");

    if (n_copies <= 1 or buf.*.iface.set_tensor_2d == null) {
        for (0..n_copies) |i| {
            ggml_backend_tensor_set(
                tensor,
                @as([*]const u8, @ptrCast(data.?)) + i * stride_data,
                offset + i * stride_tensor,
                size,
            );
        }
        return;
    }
    if (size == 0) {
        return;
    }

    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + (n_copies - 1) * stride_tensor + size <= c.ggml_nbytes(tensor), "tensor write out of bounds");

    buf.*.iface.set_tensor_2d.?(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

/// Ports `ggml_backend_tensor_get_2d` (src/ggml-backend.cpp:376 @c1d0e7a00).
pub export fn ggml_backend_tensor_get_2d(
    tensor: [*c]const c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) callconv(.c) void {
    impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
    impl.assert(buf != null, "tensor buffer not set");

    if (n_copies <= 1 or buf.*.iface.get_tensor_2d == null) {
        for (0..n_copies) |i| {
            ggml_backend_tensor_get(
                tensor,
                @as([*]u8, @ptrCast(data.?)) + i * stride_data,
                offset + i * stride_tensor,
                size,
            );
        }
        return;
    }
    if (size == 0) {
        return;
    }

    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + (n_copies - 1) * stride_tensor + size <= c.ggml_nbytes(tensor), "tensor read out of bounds");

    buf.*.iface.get_tensor_2d.?(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

/// Ports `ggml_backend_tensor_memset` (src/ggml-backend.cpp:398 @c1d0e7a00).
///
/// Note the assertion order: unlike its neighbours this one tests `size == 0`
/// *before* asserting the buffer is set, so a zero-length memset on an
/// unallocated tensor is allowed. Reproduced as written.
pub export fn ggml_backend_tensor_memset(tensor: [*c]c.ggml_tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;

    if (size == 0) {
        return;
    }

    impl.assert(buf != null, "tensor buffer not set");
    impl.assert(tensor.*.data != null, "tensor not allocated");
    impl.assert(offset + size <= c.ggml_nbytes(tensor), "tensor write out of bounds");
    impl.assert(buf.*.iface.memset_tensor != null, "memset not implemented by backend buffer");

    buf.*.iface.memset_tensor.?(buf, tensor, value, offset, size);
}

/// Ports `ggml_backend_synchronize` (src/ggml-backend.cpp:414 @c1d0e7a00).
pub export fn ggml_backend_synchronize(backend: c.ggml_backend_t) callconv(.c) void {
    impl.assert(backend != null, "backend");
    const f = backend.*.iface.synchronize orelse return;
    f(backend);
}

/// Ports `ggml_backend_graph_plan_create` (src/ggml-backend.cpp:423 @c1d0e7a00).
pub export fn ggml_backend_graph_plan_create(backend: c.ggml_backend_t, cgraph: *impl.CGraph) callconv(.c) c.ggml_backend_graph_plan_t {
    impl.assert(backend != null, "backend");
    impl.assert(backend.*.iface.graph_plan_create != null, "backend->iface.graph_plan_create != NULL");
    return backend.*.iface.graph_plan_create.?(backend, @ptrCast(cgraph));
}

/// Ports `ggml_backend_graph_plan_free` (src/ggml-backend.cpp:430 @c1d0e7a00).
pub export fn ggml_backend_graph_plan_free(backend: c.ggml_backend_t, plan: c.ggml_backend_graph_plan_t) callconv(.c) void {
    impl.assert(backend != null, "backend");
    impl.assert(backend.*.iface.graph_plan_free != null, "backend->iface.graph_plan_free != NULL");
    backend.*.iface.graph_plan_free.?(backend, plan);
}

/// Ports `ggml_backend_graph_plan_compute` (src/ggml-backend.cpp:437 @c1d0e7a00).
pub export fn ggml_backend_graph_plan_compute(backend: c.ggml_backend_t, plan: c.ggml_backend_graph_plan_t) callconv(.c) c.enum_ggml_status {
    impl.assert(backend != null, "backend");
    impl.assert(backend.*.iface.graph_plan_compute != null, "backend->iface.graph_plan_compute != NULL");
    return backend.*.iface.graph_plan_compute.?(backend, plan);
}

/// Ports `ggml_backend_graph_compute` (src/ggml-backend.cpp:444 @c1d0e7a00).
///
/// Return: the compute status. Synchronizes before returning, and returns the
/// status from *before* the synchronize, as the C does.
pub export fn ggml_backend_graph_compute(backend: c.ggml_backend_t, cgraph: *impl.CGraph) callconv(.c) c.enum_ggml_status {
    const err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

/// Ports `ggml_backend_graph_compute_async` (src/ggml-backend.cpp:450 @c1d0e7a00).
pub export fn ggml_backend_graph_compute_async(backend: c.ggml_backend_t, cgraph: *impl.CGraph) callconv(.c) c.enum_ggml_status {
    impl.assert(backend != null, "backend");
    return backend.*.iface.graph_compute.?(backend, @ptrCast(cgraph));
}

/// Ports `ggml_backend_supports_op` (src/ggml-backend.cpp:455 @c1d0e7a00).
pub export fn ggml_backend_supports_op(backend: c.ggml_backend_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    impl.assert(backend != null, "backend");
    return ggml_backend_dev_supports_op(backend.*.device, op);
}

/// Ports `ggml_backend_supports_buft` (src/ggml-backend.cpp:460 @c1d0e7a00).
pub export fn ggml_backend_supports_buft(backend: c.ggml_backend_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    impl.assert(backend != null, "backend");
    return ggml_backend_dev_supports_buft(backend.*.device, buft);
}

/// Ports `ggml_backend_offload_op` (src/ggml-backend.cpp:465 @c1d0e7a00).
pub export fn ggml_backend_offload_op(backend: c.ggml_backend_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    impl.assert(backend != null, "backend");
    return ggml_backend_dev_offload_op(backend.*.device, op);
}

/// Ports `ggml_backend_get_device` (src/ggml-backend.cpp:470 @c1d0e7a00).
pub export fn ggml_backend_get_device(backend: c.ggml_backend_t) callconv(.c) c.ggml_backend_dev_t {
    impl.assert(backend != null, "backend");
    return backend.*.device;
}

// -----------------------------------------------------------------------------
// Backend copy

/// Ports `ggml_backend_tensor_copy` (src/ggml-backend.cpp:477 @c1d0e7a00).
///
/// Three paths in priority order: a host source reads directly, a host
/// destination writes directly, and otherwise the destination buffer is asked
/// to do it. The last resort is a malloc'd bounce buffer, which the C logs as
/// a slow copy under `NDEBUG`.
pub export fn ggml_backend_tensor_copy(src: [*c]const c.ggml_tensor, dst: [*c]c.ggml_tensor) callconv(.c) void {
    impl.assert(impl.areSameLayout(&src.*, &dst.*), "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src.*.buffer)) {
        ggml_backend_tensor_set(dst, src.*.data, 0, c.ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst.*.buffer)) {
        ggml_backend_tensor_get(src, dst.*.data, 0, c.ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
        if (!debug_off) {
            impl.logDebug(
                "%s: warning: slow copy from %s to %s\n",
                .{ "ggml_backend_tensor_copy", ggml_backend_buffer_name(src.*.buffer), ggml_backend_buffer_name(dst.*.buffer) },
            );
        }
        const nbytes = c.ggml_nbytes(src);
        const data = std.c.malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        std.c.free(data);
    }
}

/// Mirrors `NDEBUG`, which gates the slow-copy warning above and several
/// scheduler diagnostics. `build/llamacpp.zig` compiles the reference tree
/// with `-DNDEBUG` only in release, matching Zig's own optimize mode.
const debug_off = @import("builtin").mode != .Debug;

/// Ports `ggml_backend_tensor_copy_async` (src/ggml-backend.cpp:500 @c1d0e7a00).
///
/// The fallback synchronizes *both* backends before a blocking copy, because
/// an async copy would otherwise have to wait for work already queued on each.
pub export fn ggml_backend_tensor_copy_async(
    backend_src: c.ggml_backend_t,
    backend_dst: c.ggml_backend_t,
    src: [*c]const c.ggml_tensor,
    dst: [*c]c.ggml_tensor,
) callconv(.c) void {
    impl.assert(impl.areSameLayout(&src.*, &dst.*), "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    impl.assert(backend_dst != null, "backend_dst");
    if (backend_dst.*.iface.cpy_tensor_async) |f| {
        if (f(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on
    // both backends are completed; to simulate the same behavior, synchronize
    // both backends first and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// -----------------------------------------------------------------------------
// Events

/// Ports `ggml_backend_event_new` (src/ggml-backend.cpp:523 @c1d0e7a00).
///
/// Return: a new event, or null when the device is null or has no events —
/// the C notes a null device is allowed during the transition to the device
/// interface.
pub export fn ggml_backend_event_new(device: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_event_t {
    // null device is allowed for the transition period to the device interface
    if (device == null or device.*.iface.event_new == null) {
        return null;
    }
    return device.*.iface.event_new.?(device);
}

/// Ports `ggml_backend_event_free` (src/ggml-backend.cpp:531 @c1d0e7a00).
pub export fn ggml_backend_event_free(event: c.ggml_backend_event_t) callconv(.c) void {
    if (event == null) {
        return;
    }
    event.*.device.*.iface.event_free.?(event.*.device, event);
}

/// Ports `ggml_backend_event_record` (src/ggml-backend.cpp:538 @c1d0e7a00).
pub export fn ggml_backend_event_record(event: c.ggml_backend_event_t, backend: c.ggml_backend_t) callconv(.c) void {
    impl.assert(backend != null, "backend");
    impl.assert(backend.*.iface.event_record != null, "backend->iface.event_record != NULL");
    backend.*.iface.event_record.?(backend, event);
}

/// Ports `ggml_backend_event_synchronize` (src/ggml-backend.cpp:545 @c1d0e7a00).
pub export fn ggml_backend_event_synchronize(event: c.ggml_backend_event_t) callconv(.c) void {
    impl.assert(event != null, "event");
    impl.assert(event.*.device.*.iface.event_synchronize != null, "event->device->iface.event_synchronize");
    event.*.device.*.iface.event_synchronize.?(event.*.device, event);
}

/// Ports `ggml_backend_event_wait` (src/ggml-backend.cpp:552 @c1d0e7a00).
pub export fn ggml_backend_event_wait(backend: c.ggml_backend_t, event: c.ggml_backend_event_t) callconv(.c) void {
    impl.assert(backend != null, "backend");
    impl.assert(backend.*.iface.event_wait != null, "backend->iface.event_wait != NULL");
    backend.*.iface.event_wait.?(backend, event);
}

/// Ports `ggml_backend_graph_optimize` (src/ggml-backend.cpp:559 @c1d0e7a00).
///
/// File-private in the C++ and used only by the scheduler, so it is `pub`
/// here rather than exported: `backend_sched.zig` is the one caller.
pub fn graphOptimize(backend: c.ggml_backend_t, cgraph: *impl.CGraph) void {
    impl.assert(backend != null, "backend");
    if (backend.*.iface.graph_optimize) |f| {
        f(backend, @ptrCast(cgraph));
    }
}

// -----------------------------------------------------------------------------
// Backend device

/// Ports `ggml_backend_dev_name` (src/ggml-backend.cpp:568 @c1d0e7a00).
pub export fn ggml_backend_dev_name(device: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    impl.assert(device != null, "device");
    return device.*.iface.get_name.?(device);
}

/// Ports `ggml_backend_dev_description` (src/ggml-backend.cpp:573 @c1d0e7a00).
pub export fn ggml_backend_dev_description(device: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    impl.assert(device != null, "device");
    return device.*.iface.get_description.?(device);
}

/// Ports `ggml_backend_dev_memory` (src/ggml-backend.cpp:578 @c1d0e7a00).
pub export fn ggml_backend_dev_memory(device: c.ggml_backend_dev_t, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    impl.assert(device != null, "device");
    device.*.iface.get_memory.?(device, free, total);
}

/// Ports `ggml_backend_dev_type` (src/ggml-backend.cpp:583 @c1d0e7a00).
pub export fn ggml_backend_dev_type(device: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    impl.assert(device != null, "device");
    return device.*.iface.get_type.?(device);
}

/// Ports `ggml_backend_dev_get_props` (src/ggml-backend.cpp:588 @c1d0e7a00).
///
/// Zeroes `props` first, so a device that fills only some fields leaves the
/// rest defined.
pub export fn ggml_backend_dev_get_props(device: c.ggml_backend_dev_t, props: [*c]c.struct_ggml_backend_dev_props) callconv(.c) void {
    impl.assert(device != null, "device");
    props.* = std.mem.zeroes(c.struct_ggml_backend_dev_props);
    device.*.iface.get_props.?(device, props);
}

/// Ports `ggml_backend_dev_backend_reg` (src/ggml-backend.cpp:594 @c1d0e7a00).
pub export fn ggml_backend_dev_backend_reg(device: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_reg_t {
    impl.assert(device != null, "device");
    return device.*.reg;
}

/// Ports `ggml_backend_dev_init` (src/ggml-backend.cpp:599 @c1d0e7a00).
pub export fn ggml_backend_dev_init(device: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    impl.assert(device != null, "device");
    return device.*.iface.init_backend.?(device, params);
}

/// Ports `ggml_backend_dev_buffer_type` (src/ggml-backend.cpp:604 @c1d0e7a00).
pub export fn ggml_backend_dev_buffer_type(device: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    impl.assert(device != null, "device");
    return device.*.iface.get_buffer_type.?(device);
}

/// Ports `ggml_backend_dev_host_buffer_type` (src/ggml-backend.cpp:609 @c1d0e7a00).
///
/// Return: the device's pinned-host buffer type, or null when it has none.
pub export fn ggml_backend_dev_host_buffer_type(device: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    impl.assert(device != null, "device");
    const f = device.*.iface.get_host_buffer_type orelse return null;
    return f(device);
}

/// Ports `ggml_backend_dev_buffer_from_host_ptr` (src/ggml-backend.cpp:618 @c1d0e7a00).
pub export fn ggml_backend_dev_buffer_from_host_ptr(
    device: c.ggml_backend_dev_t,
    ptr: ?*anyopaque,
    size: usize,
    max_tensor_size: usize,
) callconv(.c) c.ggml_backend_buffer_t {
    impl.assert(device != null, "device");
    return device.*.iface.buffer_from_host_ptr.?(device, ptr, size, max_tensor_size);
}

/// Ports `ggml_backend_dev_supports_op` (src/ggml-backend.cpp:623 @c1d0e7a00).
pub export fn ggml_backend_dev_supports_op(device: c.ggml_backend_dev_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    impl.assert(device != null, "device");
    return device.*.iface.supports_op.?(device, op);
}

/// Ports `ggml_backend_dev_supports_buft` (src/ggml-backend.cpp:628 @c1d0e7a00).
pub export fn ggml_backend_dev_supports_buft(device: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    impl.assert(device != null, "device");
    return device.*.iface.supports_buft.?(device, buft);
}

/// Ports `ggml_backend_dev_offload_op` (src/ggml-backend.cpp:633 @c1d0e7a00).
///
/// Return: whether a higher-priority device wants to take this op off the CPU.
/// Optional; false when the device does not implement it.
pub export fn ggml_backend_dev_offload_op(device: c.ggml_backend_dev_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    impl.assert(device != null, "device");
    if (device.*.iface.offload_op) |f| {
        return f(device, op);
    }
    return false;
}

// -----------------------------------------------------------------------------
// Backend registry

/// Ports `ggml_backend_reg_name` (src/ggml-backend.cpp:644 @c1d0e7a00).
pub export fn ggml_backend_reg_name(reg: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    impl.assert(reg != null, "reg");
    return reg.*.iface.get_name.?(reg);
}

/// Ports `ggml_backend_reg_dev_count` (src/ggml-backend.cpp:649 @c1d0e7a00).
pub export fn ggml_backend_reg_dev_count(reg: c.ggml_backend_reg_t) callconv(.c) usize {
    impl.assert(reg != null, "reg");
    return reg.*.iface.get_device_count.?(reg);
}

/// Ports `ggml_backend_reg_dev_get` (src/ggml-backend.cpp:654 @c1d0e7a00).
pub export fn ggml_backend_reg_dev_get(reg: c.ggml_backend_reg_t, index: usize) callconv(.c) c.ggml_backend_dev_t {
    impl.assert(reg != null, "reg");
    return reg.*.iface.get_device.?(reg, index);
}

/// Ports `ggml_backend_reg_get_proc_address` (src/ggml-backend.cpp:659 @c1d0e7a00).
///
/// Return: the named extension entry point, or null when the registry has no
/// `get_proc_address` or does not know the name.
pub export fn ggml_backend_reg_get_proc_address(reg: c.ggml_backend_reg_t, name: [*c]const u8) callconv(.c) ?*anyopaque {
    impl.assert(reg != null, "reg");
    const f = reg.*.iface.get_proc_address orelse return null;
    return f(reg, name);
}

// -----------------------------------------------------------------------------
// Multi-buffer
//
// A buffer that owns several others and fans free, clear and usage changes out
// to all of them. Its `iface` deliberately leaves everything else null, so a
// multi-buffer cannot be read from or written to directly.

/// Ports `struct ggml_backend_multi_buffer_context` (src/ggml-backend.cpp:669 @c1d0e7a00).
const MultiBufferContext = extern struct {
    buffers: [*c]c.ggml_backend_buffer_t,
    n_buffers: usize,
};

/// Ports `ggml_backend_multi_buffer_free_buffer` (src/ggml-backend.cpp:674 @c1d0e7a00).
///
/// Its address is also the tag `ggml_backend_buffer_is_multi_buffer` compares
/// against, so this function must stay a distinct symbol.
fn multiBufferFreeBuffer(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    impl.assert(buffer != null, "buffer");
    const ctx: *MultiBufferContext = @ptrCast(@alignCast(buffer.*.context.?));
    for (0..ctx.n_buffers) |i| {
        ggml_backend_buffer_free(ctx.buffers[i]);
    }
    std.c.free(@ptrCast(ctx.buffers));
    std.c.free(ctx);
}

/// Ports `ggml_backend_multi_buffer_clear` (src/ggml-backend.cpp:685 @c1d0e7a00).
fn multiBufferClear(buffer: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    impl.assert(buffer != null, "buffer");
    const ctx: *MultiBufferContext = @ptrCast(@alignCast(buffer.*.context.?));
    for (0..ctx.n_buffers) |i| {
        ggml_backend_buffer_clear(ctx.buffers[i], value);
    }
}

/// Ports `ggml_backend_multi_buffer_i` (src/ggml-backend.cpp:693 @c1d0e7a00).
const multi_buffer_i: c.ggml_backend_buffer_i = .{
    .free_buffer = multiBufferFreeBuffer,
    .get_base = null,
    .init_tensor = null,
    .memset_tensor = null,
    .set_tensor = null,
    .get_tensor = null,
    .set_tensor_2d = null,
    .get_tensor_2d = null,
    .cpy_tensor = null,
    .clear = multiBufferClear,
    .reset = null,
};

/// Ports `ggml_backend_multi_buffer_alloc_buffer` (src/ggml-backend.cpp:707 @c1d0e7a00).
///
/// Parameters:
/// - `buffers`: the buffers to take ownership of; copied into the context.
/// - `n_buffers`: how many.
///
/// Return: a buffer whose size is the sum of theirs, and whose type is taken
/// from the first.
pub export fn ggml_backend_multi_buffer_alloc_buffer(buffers: [*c]c.ggml_backend_buffer_t, n_buffers: usize) callconv(.c) c.ggml_backend_buffer_t {
    const ctx: *MultiBufferContext = @ptrCast(@alignCast(std.c.malloc(@sizeOf(MultiBufferContext)).?));
    ctx.n_buffers = n_buffers;
    ctx.buffers = @ptrCast(@alignCast(std.c.malloc(n_buffers * @sizeOf(c.ggml_backend_buffer_t))));

    impl.assert(ctx.buffers != null, "ctx->buffers != NULL");

    var total_size: usize = 0;
    for (0..n_buffers) |i| {
        ctx.buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0].*.buft, multi_buffer_i, ctx, total_size);
}

/// Ports `ggml_backend_buffer_is_multi_buffer` (src/ggml-backend.cpp:723 @c1d0e7a00).
///
/// Identified by the address of its `free_buffer`, which is how the C tells
/// them apart — there is no tag field.
pub export fn ggml_backend_buffer_is_multi_buffer(buffer: c.ggml_backend_buffer_t) callconv(.c) bool {
    impl.assert(buffer != null, "buffer");
    return buffer.*.iface.free_buffer == @as(?*const fn (c.ggml_backend_buffer_t) callconv(.c) void, multiBufferFreeBuffer);
}

/// Ports `ggml_backend_multi_buffer_set_usage` (src/ggml-backend.cpp:728 @c1d0e7a00).
pub export fn ggml_backend_multi_buffer_set_usage(buffer: c.ggml_backend_buffer_t, usage: c.enum_ggml_backend_buffer_usage) callconv(.c) void {
    impl.assert(buffer != null, "buffer");
    impl.assert(ggml_backend_buffer_is_multi_buffer(buffer), "ggml_backend_buffer_is_multi_buffer(buffer)");
    const ctx: *MultiBufferContext = @ptrCast(@alignCast(buffer.*.context.?));
    for (0..ctx.n_buffers) |i| {
        ggml_backend_buffer_set_usage(ctx.buffers[i], usage);
    }
}

/// Ports `ggml_dup_tensor_layout` (src/ggml-backend.cpp:738 @c1d0e7a00).
///
/// `ggml_dup_tensor` gives the same shape and type but contiguous strides;
/// this copies the source's `nb` over the top so the layout matches too.
/// Shared with `backend_sched.zig`, which is the heavier user.
pub fn dupTensorLayout(ctx: ?*c.ggml_context, tensor: [*c]const c.ggml_tensor) [*c]c.ggml_tensor {
    const dup = c.ggml_dup_tensor(ctx, tensor);
    for (0..c.GGML_MAX_DIMS) |i| {
        dup.*.nb[i] = tensor.*.nb[i];
    }
    return dup;
}

/// Ports `ggml_is_view_op` (src/ggml-backend.cpp:746 @c1d0e7a00).
///
/// The four ops that produce a view rather than compute anything; the
/// scheduler skips them everywhere it walks a graph.
pub fn isViewOp(op: c.enum_ggml_op) bool {
    return op == c.GGML_OP_VIEW or op == c.GGML_OP_RESHAPE or op == c.GGML_OP_PERMUTE or op == c.GGML_OP_TRANSPOSE;
}

// -----------------------------------------------------------------------------
// Utils

/// Ports `ggml_backend_view_init` (src/ggml-backend.cpp:2052 @c1d0e7a00).
///
/// Points a view at its source's buffer and offset, then lets the buffer
/// initialise it. Asserts the view is not already attached.
pub export fn ggml_backend_view_init(tensor: [*c]c.ggml_tensor) callconv(.c) c.enum_ggml_status {
    impl.assert(tensor != null, "tensor");
    impl.assert(tensor.*.buffer == null, "tensor->buffer == NULL");
    impl.assert(tensor.*.view_src != null, "tensor->view_src != NULL");
    impl.assert(tensor.*.view_src.*.buffer != null, "tensor->view_src->buffer != NULL");
    impl.assert(tensor.*.view_src.*.data != null, "tensor->view_src->data != NULL");

    tensor.*.buffer = tensor.*.view_src.*.buffer;
    tensor.*.data = @as([*]u8, @ptrCast(tensor.*.view_src.*.data)) + tensor.*.view_offs;
    return ggml_backend_buffer_init_tensor(tensor.*.buffer, tensor);
}

/// Ports `ggml_backend_tensor_alloc` (src/ggml-backend.cpp:2064 @c1d0e7a00).
///
/// Parameters:
/// - `buffer`: the buffer the tensor will live in.
/// - `tensor`: must be unattached and not a view.
/// - `addr`: where inside `buffer` it goes; bounds-checked unless the buffer
///   is a meta buffer, which has no real storage to overrun.
pub export fn ggml_backend_tensor_alloc(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor, addr: ?*anyopaque) callconv(.c) c.enum_ggml_status {
    impl.assert(tensor != null, "tensor");
    impl.assert(tensor.*.buffer == null, "tensor->buffer == NULL");
    impl.assert(tensor.*.data == null, "tensor->data == NULL");
    impl.assert(tensor.*.view_src == null, "tensor->view_src == NULL");
    impl.assert(@intFromPtr(addr) >= @intFromPtr(ggml_backend_buffer_get_base(buffer)), "addr >= ggml_backend_buffer_get_base(buffer)");
    impl.assert(
        c.ggml_backend_buffer_is_meta(buffer) or
            @intFromPtr(addr) + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                @intFromPtr(ggml_backend_buffer_get_base(buffer)) + ggml_backend_buffer_get_size(buffer),
        "tensor does not fit in the buffer",
    );

    tensor.*.buffer = buffer;
    tensor.*.data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

/// Ports `graph_copy_dup_tensor` (src/ggml-backend.cpp:2079 @c1d0e7a00).
///
/// Recursively duplicates `src` and everything it depends on into one of two
/// contexts: allocated tensors go to `ctx_allocated`, views and unallocated
/// ones to `ctx_unallocated`, so a single buffer can later back the first set.
///
/// Note the C takes `hash_set` **by value** while `graph_copy_init_tensor`
/// takes it by pointer. `ggml_hash_insert` can only set bits in a table it
/// does not resize, so the copy shares the same storage and the difference
/// does not matter — but it is the C's shape and is kept.
fn graphCopyDupTensor(
    hash_set: impl.HashSet,
    node_copies: [*c][*c]c.ggml_tensor,
    ctx_allocated: ?*c.ggml_context,
    ctx_unallocated: ?*c.ggml_context,
    src: *c.ggml_tensor,
) *c.ggml_tensor {
    impl.assert(src.data != null, "graph must be allocated");

    var hs = hash_set;
    const id = impl.hashInsert(&hs, src);
    if (id == impl.hashset_already_exists) {
        return impl.one(c.ggml_tensor, node_copies[impl.hashFind(&hs, src)]);
    }

    const dst = impl.one(c.ggml_tensor, dupTensorLayout(
        if (src.data != null and src.view_src == null) ctx_allocated else ctx_unallocated,
        src,
    ));
    if (src.view_src != null) {
        dst.view_src = graphCopyDupTensor(hs, node_copies, ctx_allocated, ctx_unallocated, impl.one(c.ggml_tensor, src.view_src));
        dst.view_offs = src.view_offs;
    }
    dst.op = src.op;
    dst.flags = src.flags;
    @memcpy(std.mem.asBytes(&dst.op_params), std.mem.asBytes(&src.op_params));
    _ = c.ggml_set_name(dst, &src.name);

    // copy src
    for (0..c.GGML_MAX_SRC) |i| {
        const s = src.src[i];
        if (s == null) {
            continue;
        }
        dst.src[i] = graphCopyDupTensor(hs, node_copies, ctx_allocated, ctx_unallocated, impl.one(c.ggml_tensor, s));
    }

    node_copies[id] = dst;
    return dst;
}

/// Ports `graph_copy_init_tensor` (src/ggml-backend.cpp:2113 @c1d0e7a00).
///
/// Walks the same graph a second time, now that storage exists: views are
/// wired to their source and everything else has its data copied across.
fn graphCopyInitTensor(
    hash_set: [*c]impl.HashSet,
    node_copies: [*c][*c]c.ggml_tensor,
    node_init: [*c]bool,
    src: *c.ggml_tensor,
) void {
    const id = impl.hashFind(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    const dst = impl.one(c.ggml_tensor, node_copies[id]);
    if (dst.view_src != null) {
        graphCopyInitTensor(hash_set, node_copies, node_init, impl.one(c.ggml_tensor, src.view_src));
        const status = ggml_backend_view_init(dst);
        impl.assert(status == c.GGML_STATUS_SUCCESS, "status == GGML_STATUS_SUCCESS");
    } else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (0..c.GGML_MAX_SRC) |i| {
        const s = src.src[i];
        if (s == null) {
            continue;
        }
        graphCopyInitTensor(hash_set, node_copies, node_init, impl.one(c.ggml_tensor, s));
    }
}

/// Ports `ggml_backend_graph_copy` (src/ggml-backend.cpp:2140 @c1d0e7a00).
///
/// Parameters:
/// - `backend`: the backend the copy's tensors are allocated on.
/// - `graph`: the graph to duplicate; must already be allocated.
///
/// Return: the copy, with an all-null struct on failure. Released with
/// `ggml_backend_graph_copy_free`.
pub export fn ggml_backend_graph_copy(backend: c.ggml_backend_t, graph: *impl.CGraph) callconv(.c) c.struct_ggml_backend_graph_copy {
    // The C asserts `graph` is non-null; `*impl.CGraph` says the same thing
    // in the signature.
    var hash_set = ggml_hash_set_new(graph.*.visited_hash_set.size);
    const node_copies: [*c][*c]c.ggml_tensor = @ptrCast(@alignCast(std.c.calloc(hash_set.size, @sizeOf([*c]c.ggml_tensor))));
    const node_init: [*c]bool = @ptrCast(std.c.calloc(hash_set.size, @sizeOf(bool)));

    const params: c.ggml_init_params = .{
        .mem_size = c.ggml_tensor_overhead() * hash_set.size + c.ggml_graph_overhead_custom(@intCast(graph.*.size), false),
        .mem_buffer = null,
        .no_alloc = true,
    };

    const ctx_allocated = c.ggml_init(params);
    const ctx_unallocated = c.ggml_init(params);

    const failed: c.struct_ggml_backend_graph_copy = .{
        .buffer = null,
        .ctx_allocated = null,
        .ctx_unallocated = null,
        .graph = null,
    };

    if (ctx_allocated == null or ctx_unallocated == null) {
        impl.logError("%s: failed to allocate context for graph copy\n", .{"ggml_backend_graph_copy"});
        ggml_hash_set_free(&hash_set);
        std.c.free(@ptrCast(node_copies));
        std.c.free(@ptrCast(node_init));
        c.ggml_free(ctx_allocated);
        c.ggml_free(ctx_unallocated);
        return failed;
    }

    // dup nodes
    for (0..@intCast(graph.*.n_nodes)) |i| {
        const node = graph.*.nodes[i];
        _ = graphCopyDupTensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node.?);
    }

    // allocate nodes
    const buffer = c.ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == null) {
        impl.logError("%s: failed to allocate buffer for graph copy\n", .{"ggml_backend_graph_copy"});
        ggml_hash_set_free(&hash_set);
        std.c.free(@ptrCast(node_copies));
        std.c.free(@ptrCast(node_init));
        c.ggml_free(ctx_allocated);
        c.ggml_free(ctx_unallocated);
        return failed;
    }

    // copy data and init views
    for (0..@intCast(graph.*.n_nodes)) |i| {
        const node = graph.*.nodes[i];
        graphCopyInitTensor(&hash_set, node_copies, node_init, node.?);
    }

    // build graph copy
    const graph_copy: *impl.CGraph = @ptrCast(@alignCast(c.ggml_new_graph_custom(ctx_allocated, @intCast(graph.*.size), false)));
    for (0..@intCast(graph.*.n_nodes)) |i| {
        const node = graph.*.nodes[i].?;
        graph_copy.nodes[i] = node_copies[impl.hashFind(&hash_set, node)];
    }
    graph_copy.n_nodes = graph.*.n_nodes;

    ggml_hash_set_free(&hash_set);
    std.c.free(@ptrCast(node_copies));
    std.c.free(@ptrCast(node_init));

    return .{
        .buffer = buffer,
        .ctx_allocated = ctx_allocated,
        .ctx_unallocated = ctx_unallocated,
        .graph = @ptrCast(graph_copy),
    };
}

/// Ports `ggml_backend_graph_copy_free` (src/ggml-backend.cpp:2222 @c1d0e7a00).
pub export fn ggml_backend_graph_copy_free(copy: c.struct_ggml_backend_graph_copy) callconv(.c) void {
    ggml_backend_buffer_free(copy.buffer);
    c.ggml_free(copy.ctx_allocated);
    c.ggml_free(copy.ctx_unallocated);
}

/// Ports `ggml_backend_compare_graph_backend` (src/ggml-backend.cpp:2228 @c1d0e7a00).
///
/// Runs `graph` on two backends and hands each pair of results to `callback`.
/// With `test_nodes` it computes both graphs whole and compares only those
/// nodes; without, it steps node by node so a divergence is attributed to the
/// op that caused it. This is what `test-backend-ops --diff` drives.
///
/// Return: false if the graph copy could not be allocated.
pub export fn ggml_backend_compare_graph_backend(
    backend1: c.ggml_backend_t,
    backend2: c.ggml_backend_t,
    graph: *impl.CGraph,
    callback: c.ggml_backend_eval_callback,
    user_data: ?*anyopaque,
    test_nodes: [*c]const [*c]const c.ggml_tensor,
    num_test_nodes: usize,
) callconv(.c) bool {
    const copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == null) {
        return false;
    }

    const g1 = graph;
    const g2: *impl.CGraph = @ptrCast(@alignCast(copy.graph.?));

    impl.assert(g1.*.n_nodes == g2.*.n_nodes, "g1->n_nodes == g2->n_nodes");

    if (num_test_nodes != 0) {
        impl.assert(test_nodes != null, "test_nodes");
        // Compute the whole graph and only test the output for specific tensors
        _ = ggml_backend_graph_compute(backend1, g1);
        _ = ggml_backend_graph_compute(backend2, g2);

        var verified = false;
        for (0..@intCast(g1.*.n_nodes)) |i| {
            for (0..num_test_nodes) |j| {
                if (g1.*.nodes[i] == test_nodes[j]) {
                    _ = callback.?(@intCast(i), g1.*.nodes[i], g2.*.nodes[i], user_data);
                    verified = true;
                }
            }
        }
        impl.assert(verified, "verified");
    } else {
        for (0..@intCast(g1.*.n_nodes)) |i| {
            const t1 = g1.*.nodes[i].?;
            const t2 = g2.*.nodes[i].?;

            impl.assert(t1.*.op == t2.*.op and impl.areSameLayout(&t1.*, &t2.*), "t1->op == t2->op && ggml_are_same_layout(t1, t2)");

            var g1v = graphView(g1, @intCast(i), @intCast(i + 1));
            var g2v = graphView(g2, @intCast(i), @intCast(i + 1));

            _ = ggml_backend_graph_compute(backend1, &g1v);
            _ = ggml_backend_graph_compute(backend2, &g2v);

            if (isViewOp(t1.*.op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback.?(@intCast(i), t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// -----------------------------------------------------------------------------
// CPU backend buffer
//
// Defined here rather than in the CPU backend so that every backend can reach
// it — the C says so in a comment above the buffer type.

/// Ports `ggml_backend_cpu_buffer_get_base` (src/ggml-backend.cpp:2285 @c1d0e7a00).
///
/// The context *is* the allocation, rounded up to `TENSOR_ALIGNMENT`. A
/// from-pointer buffer is already aligned and the rounding is a no-op.
fn cpuBufferGetBase(buffer: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    impl.assert(buffer != null, "buffer");
    var data = @intFromPtr(buffer.*.context);

    // align the buffer
    if (data % impl.tensor_alignment != 0) {
        data = impl.pad(data, impl.tensor_alignment);
    }

    return @ptrFromInt(data);
}

/// Ports `ggml_backend_cpu_buffer_free_buffer` (src/ggml-backend.cpp:2297 @c1d0e7a00).
fn cpuBufferFreeBuffer(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    impl.assert(buffer != null, "buffer");
    ggml_aligned_free(buffer.*.context, buffer.*.size);
}

/// Ports `ggml_backend_cpu_buffer_memset_tensor` (src/ggml-backend.cpp:2302 @c1d0e7a00).
fn cpuBufferMemsetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    _ = buffer;
    impl.assert(tensor != null, "tensor");
    @memset((@as([*]u8, @ptrCast(tensor.*.data)) + offset)[0..size], value);
}

/// Ports `ggml_backend_cpu_buffer_set_tensor` (src/ggml-backend.cpp:2309 @c1d0e7a00).
fn cpuBufferSetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    _ = buffer;
    impl.assert(tensor != null, "tensor");
    @memcpy(
        (@as([*]u8, @ptrCast(tensor.*.data)) + offset)[0..size],
        @as([*]const u8, @ptrCast(data.?))[0..size],
    );
}

/// Ports `ggml_backend_cpu_buffer_get_tensor` (src/ggml-backend.cpp:2316 @c1d0e7a00).
fn cpuBufferGetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    _ = buffer;
    impl.assert(tensor != null, "tensor");
    @memcpy(
        @as([*]u8, @ptrCast(data.?))[0..size],
        (@as([*]const u8, @ptrCast(tensor.*.data)) + offset)[0..size],
    );
}

/// Ports `ggml_backend_cpu_buffer_cpy_tensor` (src/ggml-backend.cpp:2323 @c1d0e7a00).
///
/// Return: true only when the source is host memory, which is the one case a
/// plain `memcpy` can serve.
fn cpuBufferCpyTensor(buffer: c.ggml_backend_buffer_t, src: [*c]const c.ggml_tensor, dst: [*c]c.ggml_tensor) callconv(.c) bool {
    _ = buffer;
    impl.assert(src != null, "src");
    if (ggml_backend_buffer_is_host(src.*.buffer)) {
        const n = c.ggml_nbytes(src);
        @memcpy(@as([*]u8, @ptrCast(dst.*.data))[0..n], @as([*]const u8, @ptrCast(src.*.data))[0..n]);
        return true;
    }
    return false;
}

/// Ports `ggml_backend_cpu_buffer_clear` (src/ggml-backend.cpp:2334 @c1d0e7a00).
///
/// Clears from the *context*, not from `get_base`, so the alignment padding
/// is cleared too. That is what the C does.
fn cpuBufferClear(buffer: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    impl.assert(buffer != null, "buffer");
    @memset(@as([*]u8, @ptrCast(buffer.*.context))[0..buffer.*.size], value);
}

/// Ports `ggml_backend_cpu_buffer_i` (src/ggml-backend.cpp:2339 @c1d0e7a00).
const cpu_buffer_i: c.ggml_backend_buffer_i = .{
    .free_buffer = cpuBufferFreeBuffer,
    .get_base = cpuBufferGetBase,
    .init_tensor = null, // no initialization required
    .memset_tensor = cpuBufferMemsetTensor,
    .set_tensor = cpuBufferSetTensor,
    .get_tensor = cpuBufferGetTensor,
    .set_tensor_2d = null,
    .get_tensor_2d = null,
    .cpy_tensor = cpuBufferCpyTensor,
    .clear = cpuBufferClear,
    .reset = null,
};

/// Ports `ggml_backend_cpu_buffer_from_ptr_i` (src/ggml-backend.cpp:2353 @c1d0e7a00).
///
/// Identical to `cpu_buffer_i` but for `free_buffer`: the pointer is the
/// caller's and must not be freed.
const cpu_buffer_from_ptr_i: c.ggml_backend_buffer_i = .{
    .free_buffer = null, // ptr is not owned by the buffer, so it does not need to be freed
    .get_base = cpuBufferGetBase,
    .init_tensor = null, // no initialization required
    .memset_tensor = cpuBufferMemsetTensor,
    .set_tensor = cpuBufferSetTensor,
    .get_tensor = cpuBufferGetTensor,
    .set_tensor_2d = null,
    .get_tensor_2d = null,
    .cpy_tensor = cpuBufferCpyTensor,
    .clear = cpuBufferClear,
    .reset = null,
};

// -----------------------------------------------------------------------------
// CPU backend buffer type

/// Ports `ggml_backend_cpu_buffer_type_get_name` (src/ggml-backend.cpp:2371 @c1d0e7a00).
fn cpuBufferTypeGetName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    _ = buft;
    return "CPU";
}

/// Ports `ggml_backend_cpu_buffer_type_alloc_buffer` (src/ggml-backend.cpp:2377 @c1d0e7a00).
fn cpuBufferTypeAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    const data = ggml_aligned_malloc(size);

    if (data == null) {
        impl.logError("%s: failed to allocate buffer of size %zu\n", .{ "ggml_backend_cpu_buffer_type_alloc_buffer", size });
        return null;
    }

    return ggml_backend_buffer_init(buft, cpu_buffer_i, data, size);
}

/// Ports `ggml_backend_cpu_buffer_type_get_alignment` (src/ggml-backend.cpp:2388 @c1d0e7a00).
fn cpuBufferTypeGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return impl.tensor_alignment;
}

/// Ports `ggml_backend_cpu_buffer_type_is_host` (src/ggml-backend.cpp:2394 @c1d0e7a00).
fn cpuBufferTypeIsHost(buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = buft;
    return true;
}

/// Backing storage for `ggml_backend_cpu_buffer_type`.
///
/// The C++ puts this in a function-local `static`, so its address is stable
/// and it is shared by every caller; a file-scope `var` is the same thing.
/// `device` is null, with a `FIXME` in the C noting it should be the CPU
/// registry's device.
var cpu_buffer_type_storage: c.ggml_backend_buffer_type = .{
    .iface = .{
        .get_name = cpuBufferTypeGetName,
        .alloc_buffer = cpuBufferTypeAllocBuffer,
        .get_alignment = cpuBufferTypeGetAlignment,
        .get_max_size = null, // defaults to SIZE_MAX
        .get_alloc_size = null, // defaults to ggml_nbytes
        .is_host = cpuBufferTypeIsHost,
    },
    .device = null, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0)
    .context = null,
};

/// Ports `ggml_backend_cpu_buffer_type` (src/ggml-backend.cpp:2400 @c1d0e7a00).
///
/// Return: the shared CPU buffer type. Not owned by the caller.
pub export fn ggml_backend_cpu_buffer_type() callconv(.c) c.ggml_backend_buffer_type_t {
    return &cpu_buffer_type_storage;
}

/// Ports `ggml_backend_cpu_buffer_from_ptr_type_get_name` (src/ggml-backend.cpp:2417 @c1d0e7a00).
fn cpuBufferFromPtrTypeGetName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    _ = buft;
    return "CPU_Mapped";
}

/// Backing storage for `cpuBufferFromPtrType`.
///
/// A second, distinct object from `cpu_buffer_type_storage`: the two differ
/// only in `get_name`, but `ggml_backend_sched` compares buffer types by
/// address, so they must not be merged.
var cpu_buffer_from_ptr_type_storage: c.ggml_backend_buffer_type = .{
    .iface = .{
        .get_name = cpuBufferFromPtrTypeGetName,
        .alloc_buffer = cpuBufferTypeAllocBuffer,
        .get_alignment = cpuBufferTypeGetAlignment,
        .get_max_size = null, // defaults to SIZE_MAX
        .get_alloc_size = null, // defaults to ggml_nbytes
        .is_host = cpuBufferTypeIsHost,
    },
    .device = null, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0)
    .context = null,
};

/// Ports `ggml_backend_cpu_buffer_from_ptr_type` (src/ggml-backend.cpp:2423 @c1d0e7a00).
///
/// File-private in the C++, and only `ggml_backend_cpu_buffer_from_ptr` calls
/// it.
fn cpuBufferFromPtrType() c.ggml_backend_buffer_type_t {
    return &cpu_buffer_from_ptr_type_storage;
}

/// Ports `ggml_backend_cpu_buffer_from_ptr` (src/ggml-backend.cpp:2440 @c1d0e7a00).
///
/// Wraps memory the caller owns. The buffer never frees it.
pub export fn ggml_backend_cpu_buffer_from_ptr(ptr: ?*anyopaque, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    impl.assert(@intFromPtr(ptr) % impl.tensor_alignment == 0, "buffer pointer must be aligned");
    return ggml_backend_buffer_init(cpuBufferFromPtrType(), cpu_buffer_from_ptr_i, ptr, size);
}

// -----------------------------------------------------------------------------
// Imported from the sibling ported files
//
// `ggml-impl.h` declares these but cannot be imported, so they are redeclared
// here exactly as `context.zig` does. `ggml_graph_view` is `graph.zig`'s,
// taking and returning `impl.CGraph` rather than the opaque import type.

extern fn ggml_aligned_malloc(size: usize) ?*anyopaque;
extern fn ggml_hash_set_new(size: usize) impl.HashSet;
extern fn ggml_hash_set_free(hash_set: *impl.HashSet) void;
extern fn ggml_aligned_free(ptr: ?*anyopaque, size: usize) void;

/// Calls `graph.zig`'s `ggml_graph_view` under its C name.
inline fn graphView(cgraph0: *impl.CGraph, first: c_int, last: c_int) impl.CGraph {
    return @call(.auto, @extern(*const fn (*impl.CGraph, c_int, c_int) callconv(.c) impl.CGraph, .{ .name = "ggml_graph_view" }), .{ cgraph0, first, last });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the CPU buffer type reports its name, alignment and hostness" {
    const buft = ggml_backend_cpu_buffer_type();
    try std.testing.expectEqualStrings("CPU", std.mem.span(ggml_backend_buft_name(buft)));
    try std.testing.expectEqual(@as(usize, impl.tensor_alignment), ggml_backend_buft_get_alignment(buft));
    try std.testing.expect(ggml_backend_buft_is_host(buft));
    // `get_max_size` is null on this type, so the default must come through.
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), ggml_backend_buft_get_max_size(buft));
    // Calling it twice must give the same object: callers compare by address.
    try std.testing.expectEqual(buft, ggml_backend_cpu_buffer_type());
}

test "a CPU buffer round-trips a tensor through set and get" {
    const buft = ggml_backend_cpu_buffer_type();
    const buffer = ggml_backend_buft_alloc_buffer(buft, 1024);
    defer ggml_backend_buffer_free(buffer);

    try std.testing.expectEqual(@as(usize, 1024), ggml_backend_buffer_get_size(buffer));
    try std.testing.expect(ggml_backend_buffer_is_host(buffer));
    try std.testing.expectEqual(buft, ggml_backend_buffer_get_type(buffer));

    const gctx = c.ggml_init(.{ .mem_size = 1024 * 1024, .mem_buffer = null, .no_alloc = true });
    defer c.ggml_free(gctx);
    const t = c.ggml_new_tensor_1d(gctx, c.GGML_TYPE_F32, 8);

    const status = ggml_backend_tensor_alloc(buffer, t, ggml_backend_buffer_get_base(buffer));
    try std.testing.expectEqual(@as(c.enum_ggml_status, c.GGML_STATUS_SUCCESS), status);

    const want = [_]f32{ 1.0, -2.0, 3.5, 0.25, 0.0, 7.0, -0.5, 100.0 };
    ggml_backend_tensor_set(t, &want, 0, @sizeOf(@TypeOf(want)));

    var got: [8]f32 = undefined;
    ggml_backend_tensor_get(t, &got, 0, @sizeOf(@TypeOf(got)));
    try std.testing.expectEqualSlices(f32, &want, &got);

    // memset writes bytes, so 0x00 gives +0.0 across the board.
    ggml_backend_tensor_memset(t, 0, 0, @sizeOf(@TypeOf(got)));
    ggml_backend_tensor_get(t, &got, 0, @sizeOf(@TypeOf(got)));
    for (got) |v| try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)));
}

test "a zero-sized allocation yields a dummy buffer rather than reaching the backend" {
    // `ggml_backend_buft_alloc_buffer` short-circuits at size 0, so the
    // result has an all-null iface and `get_base` must not be called on it.
    const buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 0);
    defer ggml_backend_buffer_free(buffer);

    try std.testing.expectEqual(@as(usize, 0), ggml_backend_buffer_get_size(buffer));
    try std.testing.expect(ggml_backend_buffer_get_base(buffer) == null);
    // `clear` is optional for a zero-sized buffer; this must not dereference
    // the null `iface.clear`.
    ggml_backend_buffer_clear(buffer, 0xAA);
}

test "a multi-buffer is recognised by its free_buffer and fans usage out" {
    const buft = ggml_backend_cpu_buffer_type();
    var parts = [_]c.ggml_backend_buffer_t{
        ggml_backend_buft_alloc_buffer(buft, 256),
        ggml_backend_buft_alloc_buffer(buft, 512),
    };

    const multi = ggml_backend_multi_buffer_alloc_buffer(&parts, parts.len);
    // Frees the parts too, through `multiBufferFreeBuffer`.
    defer ggml_backend_buffer_free(multi);

    try std.testing.expect(ggml_backend_buffer_is_multi_buffer(multi));
    try std.testing.expectEqual(@as(usize, 256 + 512), ggml_backend_buffer_get_size(multi));

    // A plain buffer must not be mistaken for one.
    const plain = ggml_backend_buft_alloc_buffer(buft, 64);
    defer ggml_backend_buffer_free(plain);
    try std.testing.expect(!ggml_backend_buffer_is_multi_buffer(plain));

    // Setting usage on the parent reaches every child.
    ggml_backend_buffer_set_usage(multi, c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (parts) |p| {
        try std.testing.expectEqual(
            @as(c.enum_ggml_backend_buffer_usage, c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS),
            ggml_backend_buffer_get_usage(p),
        );
    }
}

test "a from-ptr buffer wraps caller memory and never frees it" {
    var storage: [256]u8 align(impl.tensor_alignment) = @splat(0);
    const buffer = ggml_backend_cpu_buffer_from_ptr(&storage, storage.len);
    defer ggml_backend_buffer_free(buffer);

    // Distinct from the plain CPU type -- the scheduler compares by address.
    try std.testing.expect(ggml_backend_buffer_get_type(buffer) != ggml_backend_cpu_buffer_type());
    try std.testing.expectEqualStrings("CPU_Mapped", std.mem.span(ggml_backend_buffer_name(buffer)));
    try std.testing.expectEqual(@as(?*anyopaque, @ptrCast(&storage)), ggml_backend_buffer_get_base(buffer));

    ggml_backend_buffer_clear(buffer, 0x5A);
    for (storage) |b| try std.testing.expectEqual(@as(u8, 0x5A), b);
    // `free_buffer` is null on this type, so `storage` is still ours and the
    // deferred free above must not touch it.
}

test "isViewOp names exactly the four view ops" {
    try std.testing.expect(isViewOp(c.GGML_OP_VIEW));
    try std.testing.expect(isViewOp(c.GGML_OP_RESHAPE));
    try std.testing.expect(isViewOp(c.GGML_OP_PERMUTE));
    try std.testing.expect(isViewOp(c.GGML_OP_TRANSPOSE));
    try std.testing.expect(!isViewOp(c.GGML_OP_ADD));
    try std.testing.expect(!isViewOp(c.GGML_OP_MUL_MAT));
    try std.testing.expect(!isViewOp(c.GGML_OP_CPY));
}

test "dupTensorLayout keeps the source's strides, not contiguous ones" {
    const gctx = c.ggml_init(.{ .mem_size = 1024 * 1024, .mem_buffer = null, .no_alloc = true });
    defer c.ggml_free(gctx);

    const base = c.ggml_new_tensor_2d(gctx, c.GGML_TYPE_F32, 4, 8);
    // A transpose has non-contiguous strides, which `ggml_dup_tensor` alone
    // would discard -- that is the whole reason this helper exists.
    const t = c.ggml_transpose(gctx, base);
    const dup = dupTensorLayout(gctx, t);

    for (0..c.GGML_MAX_DIMS) |i| {
        try std.testing.expectEqual(t.*.nb[i], dup.*.nb[i]);
        try std.testing.expectEqual(t.*.ne[i], dup.*.ne[i]);
    }
    try std.testing.expectEqual(t.*.type, dup.*.type);
}
