//! The backend abstraction: buffer types, buffers, backends, devices,
//! registries, events, and the utilities that copy a graph between backends.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-backend.cpp` (v1.0.0, `b1b770d90`),
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
//! An early survey recorded 82. Measured against `zig c++ -c` at the pinned commit,
//! the unmangled export count across both halves is **102**; of 243 total
//! exports the rest are template instantiations and inline functions. **None
//! of the mangled names is referenced from any other object** in `libggml.a`
//! or `libllama.a` — checked directly, not assumed — so the C-ABI swap
//! applies without qualification.
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
const mod = @import("module.zig");
const stdlib = @import("stdlib");

/// The heap the rest of ggml uses. A function entered through a C ABI has no
/// allocator to be handed one, the same reasoning as `backend_reg.zig` and
/// `gguf.zig`.

// -----------------------------------------------------------------------------
// Backend buffer type
//
// Every entry point here is a null check and a jump through `buft->iface`. The
// only interest is which members are optional and what they default to.

/// Ports `ggml_backend_buft_name` (src/ggml-backend.cpp:34 @b1b770d90).
pub fn ggml_backend_buft_name(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    mod.impl.assert(buft != null, "buft");
    return buft.*.iface.get_name.?(buft);
}

/// Ports `ggml_backend_buft_alloc_buffer` (src/ggml-backend.cpp:39 @b1b770d90).
///
/// Return: a buffer of `size` bytes, or a dummy zero-sized buffer when `size`
/// is 0 — the C returns one rather than asking the backend for nothing.
pub fn ggml_backend_buft_alloc_buffer(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t, size: usize) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    mod.impl.assert(buft != null, "buft");
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, std.mem.zeroes(mod.impl.ggml_c.ggml_backend_buffer_i), null, 0);
    }
    return buft.*.iface.alloc_buffer.?(buft, size);
}

/// Ports `ggml_backend_buft_alloc_buffer_n_plan_item` (src/ggml-backend.cpp:49 @b1b770d90):
/// one buffer of a plan, and the tensors placed in it.
const AllocBufferNPlanItem = struct {
    /// Total bytes for this buffer.
    size: usize,
    /// First tensor index (inclusive).
    first: usize,
    /// Last tensor index (exclusive).
    last: usize,
};

/// Ports `ggml_backend_buft_alloc_buffer_n_plan` (src/ggml-backend.cpp:57 @b1b770d90).
///
/// Splits `tensors` into runs that each fit the buffer type's maximum size,
/// the shared planning logic of the two defaults below.
///
/// Return: the plan, owned by the caller and freed with `deinit(allocator)`.
fn allocBufferNPlan(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t, tensors: [*c]?*mod.impl.ggml_c.ggml_tensor, n_tensors: c_int) std.ArrayList(AllocBufferNPlanItem) {
    var plan: std.ArrayList(AllocBufferNPlanItem) = .empty;

    const alignment = ggml_backend_buft_get_alignment(buft);
    const max_size = ggml_backend_buft_get_max_size(buft);

    var cur_buf_size: usize = 0;
    var first: usize = 0;

    const n: usize = if (n_tensors > 0) @intCast(n_tensors) else 0;
    for (0..n) |i| {
        var this_size: usize = 0;
        const t = tensors[i].?;
        if (t.data == null and t.view_src == null) {
            this_size = stdlib.num.pad(usize, ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 and (cur_buf_size + this_size) > max_size) {
            plan.append(std.heap.c_allocator, .{ .size = cur_buf_size, .first = first, .last = i }) catch mod.impl.abort("backend: out of memory");
            cur_buf_size = this_size;
            first = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.append(std.heap.c_allocator, .{ .size = cur_buf_size, .first = first, .last = n }) catch mod.impl.abort("backend: out of memory");
    }

    return plan;
}

/// Ports `ggml_backend_buft_alloc_buffer_n_default` (src/ggml-backend.cpp:93 @b1b770d90).
///
/// The default `alloc_buffer_n`: allocates tensors from a list into one or
/// more buffers of the given type, as `allocBufferNPlan` splits them.
///
/// Return: one buffer, a multi-buffer over several, or null when nothing
/// needed memory or an allocation failed.
fn allocBufferNDefault(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t, tensors: [*c]?*mod.impl.ggml_c.ggml_tensor, n_tensors: c_int) mod.impl.ggml_c.ggml_backend_buffer_t {
    var plan = allocBufferNPlan(buft, tensors, n_tensors);
    defer plan.deinit(std.heap.c_allocator);

    var buffers: std.ArrayList(mod.impl.ggml_c.ggml_backend_buffer_t) = .empty;
    defer buffers.deinit(std.heap.c_allocator);
    buffers.ensureTotalCapacity(std.heap.c_allocator, plan.items.len) catch mod.impl.abort("backend: out of memory");

    for (plan.items) |item| {
        const buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == null) {
            mod.impl.logError("%s: failed to allocate %s buffer of size %zu\n", .{ "ggml_backend_buft_alloc_buffer_n_default", ggml_backend_buft_name(buft), item.size });
            for (buffers.items) |b| ggml_backend_buffer_free(b);
            return null;
        }

        var tallocr = mod.impl.ggml_c.ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        var t_failed: ?*mod.impl.ggml_c.ggml_tensor = null;
        for (item.first..item.last) |j| {
            const t = tensors[j].?;
            if (t.data == null) {
                if (t.view_src == null) {
                    if (mod.impl.ggml_c.ggml_tallocr_alloc(&tallocr, t) != mod.impl.ggml_c.GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t.buffer == null) {
                    if (mod.impl.ggml_c.ggml_backend_view_init(t) != mod.impl.ggml_c.GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t.view_src != null and t.buffer == null) {
                    // view of a pre-allocated tensor
                    if (mod.impl.ggml_c.ggml_backend_view_init(t) != mod.impl.ggml_c.GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed) |t| {
            mod.impl.logError("%s: failed to initialize tensor %s\n", .{ "ggml_backend_buft_alloc_buffer_n_default", &t.name });
            for (buffers.items) |b| ggml_backend_buffer_free(b);
            ggml_backend_buffer_free(buffer);
            return null;
        }

        buffers.appendAssumeCapacity(buffer);
    }

    if (buffers.items.len == 0) return null;

    if (buffers.items.len == 1) return buffers.items[0];

    return ggml_backend_multi_buffer_alloc_buffer(buffers.items.ptr, buffers.items.len);
}

/// Ports `ggml_backend_buft_get_alloc_size_n_default` (src/ggml-backend.cpp:162 @b1b770d90).
///
/// Return: the total size `allocBufferNDefault` would allocate for `tensors`.
fn getAllocSizeNDefault(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t, tensors: [*c]?*mod.impl.ggml_c.ggml_tensor, n_tensors: c_int) usize {
    var plan = allocBufferNPlan(buft, tensors, n_tensors);
    defer plan.deinit(std.heap.c_allocator);

    var total: usize = 0;
    for (plan.items) |item| total += item.size;
    return total;
}

/// Ports `ggml_backend_buft_alloc_buffer_n` (src/ggml-backend.cpp:172 @b1b770d90).
///
/// Return: the buffer type's own `alloc_buffer_n` when it has one, else the
/// default's answer.
pub fn ggml_backend_buft_alloc_buffer_n(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t, tensors: [*c]?*mod.impl.ggml_c.ggml_tensor, n_tensors: c_int) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    mod.impl.assert(buft != null, "buft");
    if (buft.*.iface.alloc_buffer_n) |f| {
        return f(buft, @ptrCast(tensors), n_tensors);
    }
    return allocBufferNDefault(buft, tensors, n_tensors);
}

/// Ports `ggml_backend_buft_get_alignment` (src/ggml-backend.cpp:180 @b1b770d90).
pub fn ggml_backend_buft_get_alignment(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) usize {
    mod.impl.assert(buft != null, "buft");
    return buft.*.iface.get_alignment.?(buft);
}

/// Ports `ggml_backend_buft_get_max_size` (src/ggml-backend.cpp:185 @b1b770d90).
///
/// Return: the backend's limit, or `SIZE_MAX` — `get_max_size` is optional.
pub fn ggml_backend_buft_get_max_size(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) usize {
    mod.impl.assert(buft != null, "buft");
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft.*.iface.get_max_size) |f| {
        return f(buft);
    }
    return std.math.maxInt(usize);
}

/// Ports `ggml_backend_buft_get_alloc_size` (src/ggml-backend.cpp:194 @b1b770d90).
///
/// Return: the bytes this buffer type needs for `tensor`, which may exceed
/// `ggml_nbytes` for a type that pads. Defaults to `ggml_nbytes`.
pub fn ggml_backend_buft_get_alloc_size(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t, tensor: [*c]const mod.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    mod.impl.assert(buft != null, "buft");
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft.*.iface.get_alloc_size) |f| {
        const size = f(buft, tensor);
        // The C uses plain `assert`, not GGML_ASSERT, so this one is compiled
        // out under NDEBUG on the C side. `impl.assert` is always live; the
        // condition is a backend-implementation invariant and holds.
        mod.impl.assert(size >= mod.impl.ggml_c.ggml_nbytes(tensor), "size >= ggml_nbytes(tensor)");

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        mod.impl.assert(size <= mod.impl.ggml_c.ggml_nbytes(tensor) or
            mod.impl.opIsEmpty(tensor.*.op) or
            mod.impl.ggml_c.ggml_is_quantized(tensor.*.type) or // [TAG_ALLOC_SIZE_EXPAND]
            ggml_op_alloc_size_may_expand(tensor.*.op), "size <= ggml_nbytes(tensor) || ggml_op_is_empty(tensor->op) || ggml_is_quantized(tensor->type) || ggml_op_alloc_size_may_expand(tensor->op)");

        return size;
    }
    return mod.impl.ggml_c.ggml_nbytes(tensor);
}

/// Ports `ggml_backend_buft_get_alloc_size_n` (src/ggml-backend.cpp:213 @b1b770d90).
///
/// Return: the buffer type's own `get_alloc_size_n` when it has one, else the
/// default's answer.
pub fn ggml_backend_buft_get_alloc_size_n(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t, tensors: [*c]?*mod.impl.ggml_c.ggml_tensor, n_tensors: c_int) callconv(.c) usize {
    mod.impl.assert(buft != null, "buft");
    if (buft.*.iface.get_alloc_size_n) |f| {
        return f(buft, @ptrCast(tensors), n_tensors);
    }
    return getAllocSizeNDefault(buft, tensors, n_tensors);
}

/// Ports `ggml_backend_buft_is_host` (src/ggml-backend.cpp:221 @b1b770d90).
pub fn ggml_backend_buft_is_host(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) bool {
    mod.impl.assert(buft != null, "buft");
    if (buft.*.iface.is_host) |f| {
        return f(buft);
    }
    return false;
}

/// Ports `ggml_backend_buft_get_device` (src/ggml-backend.cpp:229 @b1b770d90).
pub fn ggml_backend_buft_get_device(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) mod.impl.ggml_c.ggml_backend_dev_t {
    mod.impl.assert(buft != null, "buft");
    return buft.*.device;
}

// -----------------------------------------------------------------------------
// Backend buffer

/// Ports `ggml_backend_buffer_init` (src/ggml-backend.cpp:236 @b1b770d90).
///
/// The C++ uses `new ggml_backend_buffer{...}`; this is that allocation, and
/// `ggml_backend_buffer_free` is the matching `delete`.
///
/// Return: a buffer owned by the caller, released with
/// `ggml_backend_buffer_free`.
pub fn ggml_backend_buffer_init(
    buft: mod.impl.ggml_c.ggml_backend_buffer_type_t,
    iface: mod.impl.ggml_c.ggml_backend_buffer_i,
    context: ?*anyopaque,
    size: usize,
) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    const buffer = std.heap.c_allocator.create(mod.impl.ggml_c.struct_ggml_backend_buffer) catch mod.impl.abort("backend: out of memory");
    buffer.* = .{
        .iface = iface,
        .buft = buft,
        .context = context,
        .size = size,
        .usage = mod.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_ANY,
    };
    return buffer;
}

/// Ports `ggml_backend_buffer_name` (src/ggml-backend.cpp:252 @b1b770d90).
pub fn ggml_backend_buffer_name(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) [*c]const u8 {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

/// Ports `ggml_backend_buffer_free` (src/ggml-backend.cpp:256 @b1b770d90).
///
/// Return: nothing. Null is accepted, as the C++'s `delete` is.
pub fn ggml_backend_buffer_free(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) void {
    if (buffer == null) {
        return;
    }
    if (buffer.*.iface.free_buffer) |f| {
        f(buffer);
    }
    std.heap.c_allocator.destroy(@as(*mod.impl.ggml_c.struct_ggml_backend_buffer, @ptrCast(buffer)));
}

/// Ports `ggml_backend_buffer_get_size` (src/ggml-backend.cpp:267 @b1b770d90).
pub fn ggml_backend_buffer_get_size(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) usize {
    mod.impl.assert(buffer != null, "buffer");
    return buffer.*.size;
}

/// Ports `ggml_backend_buffer_get_base` (src/ggml-backend.cpp:272 @b1b770d90).
///
/// Return: the buffer's base address, or null for a zero-sized non-meta buffer
/// or a buffer type with no `get_base`. The C carries a `FIXME` here about
/// multi-buffers having a non-zero size; it is reproduced as written.
pub fn ggml_backend_buffer_get_base(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    mod.impl.assert(buffer != null, "buffer");
    // get_base is optional if the buffer is zero-sized
    if (!mod.impl.ggml_c.ggml_backend_buffer_is_meta(buffer) and buffer.*.size == 0) {
        return null;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above
    // comment get_base is not optional, I don't know whether the above comment
    // is correct
    const get_base = buffer.*.iface.get_base orelse return null;

    const base = get_base(buffer);

    mod.impl.assert(base != null, "backend buffer base cannot be NULL");

    return base;
}

/// Ports `ggml_backend_buffer_init_tensor` (src/ggml-backend.cpp:292 @b1b770d90).
pub fn ggml_backend_buffer_init_tensor(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]mod.impl.ggml_c.ggml_tensor) callconv(.c) mod.impl.ggml_c.enum_ggml_status {
    mod.impl.assert(buffer != null, "buffer");
    // init_tensor is optional
    if (buffer.*.iface.init_tensor) |f| {
        return f(buffer, tensor);
    }
    return mod.impl.ggml_c.GGML_STATUS_SUCCESS;
}

/// Ports `ggml_backend_buffer_clear` (src/ggml-backend.cpp:301 @b1b770d90).
pub fn ggml_backend_buffer_clear(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    mod.impl.assert(buffer != null, "buffer");
    // clear is optional if the buffer is zero-sized
    if (buffer.*.size == 0) {
        return;
    }
    buffer.*.iface.clear.?(buffer, value);
}

/// Ports `ggml_backend_buffer_get_alignment` (src/ggml-backend.cpp:311 @b1b770d90).
pub fn ggml_backend_buffer_get_alignment(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) usize {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

/// Ports `ggml_backend_buffer_get_max_size` (src/ggml-backend.cpp:315 @b1b770d90).
pub fn ggml_backend_buffer_get_max_size(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) usize {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

/// Ports `ggml_backend_buffer_get_alloc_size` (src/ggml-backend.cpp:319 @b1b770d90).
pub fn ggml_backend_buffer_get_alloc_size(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]const mod.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

/// Ports `ggml_backend_buffer_is_host` (src/ggml-backend.cpp:323 @b1b770d90).
pub fn ggml_backend_buffer_is_host(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) bool {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

/// Ports `ggml_backend_buffer_set_usage` (src/ggml-backend.cpp:327 @b1b770d90).
///
/// A multi-buffer or a meta buffer forwards the usage to every buffer it owns.
/// The C marks those special cases `FIXME: add a generic callback to the buffer
/// interface`.
pub fn ggml_backend_buffer_set_usage(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, usage: mod.impl.ggml_c.enum_ggml_backend_buffer_usage) callconv(.c) void {
    mod.impl.assert(buffer != null, "buffer");
    buffer.*.usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (mod.impl.ggml_c.ggml_backend_buffer_is_meta(buffer)) {
        mod.impl.ggml_c.ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

/// Ports `ggml_backend_buffer_get_usage` (src/ggml-backend.cpp:339 @b1b770d90).
pub fn ggml_backend_buffer_get_usage(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) mod.impl.ggml_c.enum_ggml_backend_buffer_usage {
    mod.impl.assert(buffer != null, "buffer");
    return buffer.*.usage;
}

/// Ports `ggml_backend_buffer_get_type` (src/ggml-backend.cpp:344 @b1b770d90).
pub fn ggml_backend_buffer_get_type(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_type_t {
    mod.impl.assert(buffer != null, "buffer");
    return buffer.*.buft;
}

/// Ports `ggml_backend_buffer_reset` (src/ggml-backend.cpp:349 @b1b770d90).
pub fn ggml_backend_buffer_reset(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) void {
    mod.impl.assert(buffer != null, "buffer");
    if (buffer.*.iface.reset) |f| {
        f(buffer);
    }
}

/// Ports `ggml_backend_buffer_copy_tensor` (src/ggml-backend.cpp:356 @b1b770d90).
///
/// Return: true if the destination's buffer performed the copy; false if it
/// has no `cpy_tensor`, leaving the caller to fall back.
pub fn ggml_backend_buffer_copy_tensor(src: [*c]const mod.impl.ggml_c.ggml_tensor, dst: [*c]mod.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    const dst_buf = if (dst.*.view_src != null) dst.*.view_src.*.buffer else dst.*.buffer;
    if (dst_buf.*.iface.cpy_tensor) |f| {
        return f(dst_buf, src, dst);
    }
    return false;
}

// -----------------------------------------------------------------------------
// Backend

/// Ports `ggml_backend_guid` (src/ggml-backend.cpp:366 @b1b770d90).
pub fn ggml_backend_guid(backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) mod.impl.ggml_c.ggml_guid_t {
    if (backend == null) {
        return null;
    }
    return backend.*.guid;
}

/// Ports `ggml_backend_name` (src/ggml-backend.cpp:373 @b1b770d90).
///
/// Return: the backend's name, or the literal `"NULL"` for a null backend —
/// this is a logging helper and does not assert.
pub fn ggml_backend_name(backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) [*c]const u8 {
    if (backend == null) {
        return "NULL";
    }
    return backend.*.iface.get_name.?(backend);
}

/// Ports `ggml_backend_free` (src/ggml-backend.cpp:380 @b1b770d90).
pub fn ggml_backend_free(backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) void {
    if (backend == null) {
        return;
    }
    backend.*.iface.free.?(backend);
}

/// Ports `ggml_backend_get_default_buffer_type` (src/ggml-backend.cpp:388 @b1b770d90).
pub fn ggml_backend_get_default_buffer_type(backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_type_t {
    mod.impl.assert(backend != null, "backend");
    return ggml_backend_dev_buffer_type(backend.*.device);
}

/// Ports `ggml_backend_alloc_buffer` (src/ggml-backend.cpp:393 @b1b770d90).
pub fn ggml_backend_alloc_buffer(backend: mod.impl.ggml_c.ggml_backend_t, size: usize) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

/// Ports `ggml_backend_get_alignment` (src/ggml-backend.cpp:397 @b1b770d90).
pub fn ggml_backend_get_alignment(backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) usize {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

/// Ports `ggml_backend_get_max_size` (src/ggml-backend.cpp:401 @b1b770d90).
pub fn ggml_backend_get_max_size(backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) usize {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

/// Ports `ggml_backend_tensor_set_async` (src/ggml-backend.cpp:461 @b1b770d90).
///
/// Falls back to a synchronize plus a blocking set when the backend has no
/// async path.
pub fn ggml_backend_tensor_set_async(
    backend: mod.impl.ggml_c.ggml_backend_t,
    tensor: [*c]mod.impl.ggml_c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(tensor != null, "tensor");
    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor write out of bounds");

    if (backend.*.iface.set_tensor_async) |f| {
        f(backend, tensor, data, offset, size);
    } else {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    }
}

/// Ports `ggml_backend_tensor_get_async` (src/ggml-backend.cpp:475 @b1b770d90).
pub fn ggml_backend_tensor_get_async(
    backend: mod.impl.ggml_c.ggml_backend_t,
    tensor: [*c]const mod.impl.ggml_c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(tensor != null, "tensor");
    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor read out of bounds");

    if (backend.*.iface.get_tensor_async) |f| {
        f(backend, tensor, data, offset, size);
    } else {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    }
}

/// Ports `ggml_backend_tensor_set_2d_async` (src/ggml-backend.cpp:489 @b1b770d90).
///
/// A strided batch of `n_copies` writes. Falls back to a loop of 1-D async
/// writes when the backend has no 2-D path or there is at most one copy —
/// note the bounds assertion is only reached on the strided path, as in the C.
pub fn ggml_backend_tensor_set_2d_async(
    backend: mod.impl.ggml_c.ggml_backend_t,
    tensor: [*c]mod.impl.ggml_c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) callconv(.c) void {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(tensor != null, "tensor");
    mod.impl.assert(tensor.*.data != null, "tensor not allocated");

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

    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + (n_copies - 1) * stride_tensor + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor write out of bounds");
    backend.*.iface.set_tensor_2d_async.?(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

/// Ports `ggml_backend_tensor_get_2d_async` (src/ggml-backend.cpp:510 @b1b770d90).
pub fn ggml_backend_tensor_get_2d_async(
    backend: mod.impl.ggml_c.ggml_backend_t,
    tensor: [*c]const mod.impl.ggml_c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) callconv(.c) void {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(tensor != null, "tensor");
    mod.impl.assert(tensor.*.data != null, "tensor not allocated");

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

    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + (n_copies - 1) * stride_tensor + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor read out of bounds");
    backend.*.iface.get_tensor_2d_async.?(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

/// Ports `ggml_backend_tensor_set` (src/ggml-backend.cpp:531 @b1b770d90).
///
/// Writes through the *view source's* buffer when the tensor is a view, which
/// is the rule everywhere in this file.
pub fn ggml_backend_tensor_set(tensor: [*c]mod.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    mod.impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
    mod.impl.assert(buf != null, "tensor buffer not set");

    if (size == 0) {
        return;
    }

    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor write out of bounds");

    buf.*.iface.set_tensor.?(buf, tensor, data, offset, size);
}

/// Ports `ggml_backend_tensor_get` (src/ggml-backend.cpp:546 @b1b770d90).
pub fn ggml_backend_tensor_get(tensor: [*c]const mod.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    mod.impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
    mod.impl.assert(buf != null, "tensor buffer not set");

    if (size == 0) {
        return;
    }

    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor read out of bounds");

    buf.*.iface.get_tensor.?(buf, tensor, data, offset, size);
}

/// Ports `ggml_backend_tensor_set_2d` (src/ggml-backend.cpp:561 @b1b770d90).
pub fn ggml_backend_tensor_set_2d(
    tensor: [*c]mod.impl.ggml_c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) callconv(.c) void {
    mod.impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
    mod.impl.assert(buf != null, "tensor buffer not set");

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

    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + (n_copies - 1) * stride_tensor + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor write out of bounds");

    buf.*.iface.set_tensor_2d.?(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

/// Ports `ggml_backend_tensor_get_2d` (src/ggml-backend.cpp:583 @b1b770d90).
pub fn ggml_backend_tensor_get_2d(
    tensor: [*c]const mod.impl.ggml_c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) callconv(.c) void {
    mod.impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
    mod.impl.assert(buf != null, "tensor buffer not set");

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

    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + (n_copies - 1) * stride_tensor + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor read out of bounds");

    buf.*.iface.get_tensor_2d.?(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

/// Ports `ggml_backend_tensor_buft_groups` (src/ggml-backend.cpp:406 @b1b770d90).
///
/// Splits `0..n_tensors` by the buffer type of each tensor's buffer, the
/// groups in the order their buffer type first appears and each group's
/// indices ascending. The C keys an `unordered_map` on the buffer type only
/// to find a group's slot, so nothing reads its order; a scan does the same.
///
/// Parameters:
/// - `n_tensors`: how many indices to split.
/// - `ctx`, `getBuffer`: the buffer of index `i` is `getBuffer(ctx, i)`.
///
/// Return: the groups, owned by the caller; free with `freeBuftGroups`.
fn tensorBuftGroups(n_tensors: usize, ctx: anytype, comptime getBuffer: fn (@TypeOf(ctx), usize) mod.impl.ggml_c.ggml_backend_buffer_t) []std.ArrayList(usize) {
    var bufts: std.ArrayList(mod.impl.ggml_c.ggml_backend_buffer_type_t) = .empty;
    defer bufts.deinit(std.heap.c_allocator);
    var groups: std.ArrayList(std.ArrayList(usize)) = .empty;
    for (0..n_tensors) |i| {
        const buft = getBuffer(ctx, i).*.buft;
        const g = for (bufts.items, 0..) |b, k| {
            if (b == buft) break k;
        } else blk: {
            bufts.append(std.heap.c_allocator, buft) catch mod.impl.abort("backend: out of memory");
            groups.append(std.heap.c_allocator, .empty) catch mod.impl.abort("backend: out of memory");
            break :blk groups.items.len - 1;
        };
        groups.items[g].append(std.heap.c_allocator, i) catch mod.impl.abort("backend: out of memory");
    }
    return groups.toOwnedSlice(std.heap.c_allocator) catch mod.impl.abort("backend: out of memory");
}

/// Frees what `tensorBuftGroups` returned.
fn freeBuftGroups(groups: []std.ArrayList(usize)) void {
    for (groups) |*g| g.deinit(std.heap.c_allocator);
    std.heap.c_allocator.free(groups);
}

/// The buffer a tensor's data lives in: its view source's for a view.
fn tensorBuffer(tensor: [*c]const mod.impl.ggml_c.ggml_tensor) mod.impl.ggml_c.ggml_backend_buffer_t {
    return if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;
}

/// Ports `ggml_backend_tensor_batch_is_2d` (src/ggml-backend.cpp:420 @b1b770d90).
///
/// True when a batch is one tensor written or read `n_tensors` times at a
/// fixed stride on both sides with one size, which `set_2d`/`get_2d` can do
/// in one call; the four outputs are then that call's arguments.
fn tensorBatchIs2d(
    tensors: [*c]const [*c]const mod.impl.ggml_c.ggml_tensor,
    offsets: [*c]const usize,
    sizes: [*c]const usize,
    data_offsets: [*c]const usize,
    n_tensors: usize,
    base_offset: *usize,
    size: *usize,
    stride_tensor: *usize,
    stride_data: *usize,
) bool {
    if (n_tensors <= 1) {
        return false;
    }

    const tensor = tensors[0];
    const size0 = sizes[0];

    if (size0 == 0 or offsets[1] < offsets[0] or data_offsets[1] < data_offsets[0]) {
        return false;
    }

    const stride_tensor0 = offsets[1] - offsets[0];
    const stride_data0 = data_offsets[1] - data_offsets[0];

    for (1..n_tensors) |i| {
        if (tensors[i] != tensor or sizes[i] != size0) {
            return false;
        }
        if (offsets[i] != offsets[0] +% i *% stride_tensor0 or data_offsets[i] != data_offsets[0] +% i *% stride_data0) {
            return false;
        }
    }

    base_offset.* = offsets[0];
    size.* = size0;
    stride_tensor.* = stride_tensor0;
    stride_data.* = stride_data0;

    return true;
}

/// The body `ggml_backend_tensor_set_n` and `ggml_backend_tensor_get_n`
/// share, which in the C differ only in direction, constness and the bounds
/// message. `is_set` picks the direction.
fn tensorBatchIo(
    comptime is_set: bool,
    tensors: [*c]const [*c]mod.impl.ggml_c.ggml_tensor,
    data: ?*anyopaque,
    offsets: [*c]const usize,
    sizes: [*c]const usize,
    data_offsets: [*c]const usize,
    n_tensors: usize,
) void {
    if (n_tensors == 0) {
        return;
    }

    const data_ptrs = std.heap.c_allocator.alloc(?*anyopaque, n_tensors) catch mod.impl.abort("backend: out of memory");
    defer std.heap.c_allocator.free(data_ptrs);
    for (0..n_tensors) |i| {
        data_ptrs[i] = @as([*]u8, @ptrCast(data.?)) + data_offsets[i];
    }

    var buf: mod.impl.ggml_c.ggml_backend_buffer_t = null;
    var same_buft = true;

    for (0..n_tensors) |i| {
        mod.impl.assert(tensors[i] != null, "tensors[i] != NULL");
        const tensor_buf = tensorBuffer(tensors[i]);
        mod.impl.assert(tensor_buf != null, "tensor buffer not set");
        mod.impl.assert(tensors[i].*.data != null, "tensor not allocated");
        mod.impl.assert(offsets[i] + sizes[i] <= mod.impl.ggml_c.ggml_nbytes(tensors[i]), if (is_set) "tensor write out of bounds" else "tensor read out of bounds");

        if (buf == null) {
            buf = tensor_buf;
        } else if (buf.*.buft != tensor_buf.*.buft) {
            same_buft = false;
        }
    }

    if (is_set) {
        if (same_buft and buf.*.iface.set_tensor_n != null) {
            buf.*.iface.set_tensor_n.?(buf, @ptrCast(tensors), @ptrCast(data_ptrs.ptr), offsets, sizes, n_tensors);
            return;
        }
    } else {
        if (same_buft and buf.*.iface.get_tensor_n != null) {
            buf.*.iface.get_tensor_n.?(buf, @ptrCast(tensors), @ptrCast(data_ptrs.ptr), offsets, sizes, n_tensors);
            return;
        }
    }

    if (!same_buft) {
        const groups = tensorBuftGroups(n_tensors, tensors, struct {
            fn get(ts: [*c]const [*c]mod.impl.ggml_c.ggml_tensor, i: usize) mod.impl.ggml_c.ggml_backend_buffer_t {
                return tensorBuffer(ts[i]);
            }
        }.get);
        defer freeBuftGroups(groups);

        for (groups) |group| {
            const n = group.items.len;
            const g_tensors = std.heap.c_allocator.alloc([*c]mod.impl.ggml_c.ggml_tensor, n) catch mod.impl.abort("backend: out of memory");
            defer std.heap.c_allocator.free(g_tensors);
            const g_sizes = std.heap.c_allocator.alloc(usize, 3 * n) catch mod.impl.abort("backend: out of memory");
            defer std.heap.c_allocator.free(g_sizes);
            const g_offsets = g_sizes[0..n];
            const g_size = g_sizes[n .. 2 * n];
            const g_data_offsets = g_sizes[2 * n .. 3 * n];

            for (group.items, 0..) |i, k| {
                g_tensors[k] = tensors[i];
                g_offsets[k] = offsets[i];
                g_size[k] = sizes[i];
                g_data_offsets[k] = data_offsets[i];
            }

            tensorBatchIo(is_set, g_tensors.ptr, data, g_offsets.ptr, g_size.ptr, g_data_offsets.ptr, n);
        }
        return;
    }

    var base_offset: usize = 0;
    var size: usize = 0;
    var stride_tensor: usize = 0;
    var stride_data: usize = 0;
    if (tensorBatchIs2d(@ptrCast(tensors), offsets, sizes, data_offsets, n_tensors, &base_offset, &size, &stride_tensor, &stride_data)) {
        if (is_set) {
            ggml_backend_tensor_set_2d(tensors[0], data_ptrs[0], base_offset, size, n_tensors, stride_tensor, stride_data);
        } else {
            ggml_backend_tensor_get_2d(tensors[0], data_ptrs[0], base_offset, size, n_tensors, stride_tensor, stride_data);
        }
        return;
    }

    for (0..n_tensors) |i| {
        if (is_set) {
            ggml_backend_tensor_set(tensors[i], data_ptrs[i], offsets[i], sizes[i]);
        } else {
            ggml_backend_tensor_get(tensors[i], data_ptrs[i], offsets[i], sizes[i]);
        }
    }
}

/// Ports `ggml_backend_tensor_set_n` (src/ggml-backend.cpp:605 @b1b770d90).
///
/// Writes `sizes[i]` bytes from `data + data_offsets[i]` to `tensors[i]` at
/// `offsets[i]`, for every `i`: through the buffer's own `set_tensor_n` when
/// every tensor shares one buffer type and the buffer has it, split by buffer
/// type when they do not, as one `set_2d` when the batch is a fixed stride
/// over one tensor, and one `set` at a time otherwise.
pub fn ggml_backend_tensor_set_n(
    tensors: [*c]const [*c]mod.impl.ggml_c.ggml_tensor,
    data: ?*const anyopaque,
    offsets: [*c]const usize,
    sizes: [*c]const usize,
    data_offsets: [*c]const usize,
    n_tensors: usize,
) callconv(.c) void {
    tensorBatchIo(true, tensors, @constCast(data), offsets, sizes, data_offsets, n_tensors);
}

/// Ports `ggml_backend_tensor_get_n` (src/ggml-backend.cpp:679 @b1b770d90).
///
/// `ggml_backend_tensor_set_n` in the other direction.
pub fn ggml_backend_tensor_get_n(
    tensors: [*c]const [*c]const mod.impl.ggml_c.ggml_tensor,
    data: ?*anyopaque,
    offsets: [*c]const usize,
    sizes: [*c]const usize,
    data_offsets: [*c]const usize,
    n_tensors: usize,
) callconv(.c) void {
    tensorBatchIo(false, @ptrCast(tensors), data, offsets, sizes, data_offsets, n_tensors);
}

/// Ports `ggml_backend_tensor_memset` (src/ggml-backend.cpp:753 @b1b770d90).
///
/// Note the assertion order: unlike its neighbours this one tests `size == 0`
/// *before* asserting the buffer is set, so a zero-length memset on an
/// unallocated tensor is allowed. Reproduced as written.
pub fn ggml_backend_tensor_memset(tensor: [*c]mod.impl.ggml_c.ggml_tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    mod.impl.assert(tensor != null, "tensor");
    const buf = if (tensor.*.view_src != null) tensor.*.view_src.*.buffer else tensor.*.buffer;

    if (size == 0) {
        return;
    }

    mod.impl.assert(buf != null, "tensor buffer not set");
    mod.impl.assert(tensor.*.data != null, "tensor not allocated");
    mod.impl.assert(offset + size <= mod.impl.ggml_c.ggml_nbytes(tensor), "tensor write out of bounds");
    mod.impl.assert(buf.*.iface.memset_tensor != null, "memset not implemented by backend buffer");

    buf.*.iface.memset_tensor.?(buf, tensor, value, offset, size);
}

/// Ports `ggml_backend_synchronize` (src/ggml-backend.cpp:769 @b1b770d90).
pub fn ggml_backend_synchronize(backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) void {
    mod.impl.assert(backend != null, "backend");
    const f = backend.*.iface.synchronize orelse return;
    f(backend);
}

/// Ports `ggml_backend_graph_plan_create` (src/ggml-backend.cpp:778 @b1b770d90).
pub fn ggml_backend_graph_plan_create(backend: mod.impl.ggml_c.ggml_backend_t, cgraph: *mod.impl.CGraph) callconv(.c) mod.impl.ggml_c.ggml_backend_graph_plan_t {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(backend.*.iface.graph_plan_create != null, "backend->iface.graph_plan_create != NULL");
    return backend.*.iface.graph_plan_create.?(backend, @ptrCast(cgraph));
}

/// Ports `ggml_backend_graph_plan_free` (src/ggml-backend.cpp:785 @b1b770d90).
pub fn ggml_backend_graph_plan_free(backend: mod.impl.ggml_c.ggml_backend_t, plan: mod.impl.ggml_c.ggml_backend_graph_plan_t) callconv(.c) void {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(backend.*.iface.graph_plan_free != null, "backend->iface.graph_plan_free != NULL");
    backend.*.iface.graph_plan_free.?(backend, plan);
}

/// Ports `ggml_backend_graph_plan_compute` (src/ggml-backend.cpp:792 @b1b770d90).
pub fn ggml_backend_graph_plan_compute(backend: mod.impl.ggml_c.ggml_backend_t, plan: mod.impl.ggml_c.ggml_backend_graph_plan_t) callconv(.c) mod.impl.ggml_c.enum_ggml_status {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(backend.*.iface.graph_plan_compute != null, "backend->iface.graph_plan_compute != NULL");
    return backend.*.iface.graph_plan_compute.?(backend, plan);
}

/// Ports `ggml_backend_graph_compute` (src/ggml-backend.cpp:799 @b1b770d90).
///
/// Return: the compute status. Synchronizes before returning, and returns the
/// status from *before* the synchronize, as the C does.
pub fn ggml_backend_graph_compute(backend: mod.impl.ggml_c.ggml_backend_t, cgraph: *mod.impl.CGraph) callconv(.c) mod.impl.ggml_c.enum_ggml_status {
    const err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

/// Ports `ggml_backend_graph_compute_async` (src/ggml-backend.cpp:805 @b1b770d90).
pub fn ggml_backend_graph_compute_async(backend: mod.impl.ggml_c.ggml_backend_t, cgraph: *mod.impl.CGraph) callconv(.c) mod.impl.ggml_c.enum_ggml_status {
    mod.impl.assert(backend != null, "backend");
    return backend.*.iface.graph_compute.?(backend, @ptrCast(cgraph));
}

/// Ports `ggml_backend_supports_op` (src/ggml-backend.cpp:810 @b1b770d90).
pub fn ggml_backend_supports_op(backend: mod.impl.ggml_c.ggml_backend_t, op: [*c]const mod.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    mod.impl.assert(backend != null, "backend");
    return ggml_backend_dev_supports_op(backend.*.device, op);
}

/// Ports `ggml_backend_supports_buft` (src/ggml-backend.cpp:815 @b1b770d90).
pub fn ggml_backend_supports_buft(backend: mod.impl.ggml_c.ggml_backend_t, buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) bool {
    mod.impl.assert(backend != null, "backend");
    return ggml_backend_dev_supports_buft(backend.*.device, buft);
}

/// Ports `ggml_backend_offload_op` (src/ggml-backend.cpp:820 @b1b770d90).
pub fn ggml_backend_offload_op(backend: mod.impl.ggml_c.ggml_backend_t, op: [*c]const mod.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    mod.impl.assert(backend != null, "backend");
    return ggml_backend_dev_offload_op(backend.*.device, op);
}

/// Ports `ggml_backend_get_device` (src/ggml-backend.cpp:825 @b1b770d90).
pub fn ggml_backend_get_device(backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) mod.impl.ggml_c.ggml_backend_dev_t {
    mod.impl.assert(backend != null, "backend");
    return backend.*.device;
}

// -----------------------------------------------------------------------------
// Backend copy

/// Ports `ggml_backend_tensor_copy` (src/ggml-backend.cpp:832 @b1b770d90).
///
/// Three paths in priority order: a host source reads directly, a host
/// destination writes directly, and otherwise the destination buffer is asked
/// to do it. The last resort is a malloc'd bounce buffer, which the C logs as
/// a slow copy under `NDEBUG`.
pub fn ggml_backend_tensor_copy(src: [*c]const mod.impl.ggml_c.ggml_tensor, dst: [*c]mod.impl.ggml_c.ggml_tensor) callconv(.c) void {
    mod.impl.assert(mod.impl.areSameLayout(&src.*, &dst.*), "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src.*.buffer)) {
        ggml_backend_tensor_set(dst, src.*.data, 0, mod.impl.ggml_c.ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst.*.buffer)) {
        ggml_backend_tensor_get(src, dst.*.data, 0, mod.impl.ggml_c.ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
        if (!debug_off) {
            mod.impl.logDebug(
                "%s: warning: slow copy from %s to %s\n",
                .{ "ggml_backend_tensor_copy", ggml_backend_buffer_name(src.*.buffer), ggml_backend_buffer_name(dst.*.buffer) },
            );
        }
        const nbytes = mod.impl.ggml_c.ggml_nbytes(src);
        const data = std.c.malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        std.c.free(data);
    }
}

/// Ports `ggml_backend_tensor_copy_n` (src/ggml-backend.cpp:855 @b1b770d90).
///
/// `ggml_backend_tensor_copy` for `n_tensors` pairs: through the destination
/// buffer's `cpy_tensor_n` when every destination shares one buffer type and
/// the buffer does it -- a `false` from it means it copied nothing -- split
/// by destination buffer type when they do not, and one copy at a time
/// otherwise.
pub fn ggml_backend_tensor_copy_n(
    srcs: [*c]const [*c]const mod.impl.ggml_c.ggml_tensor,
    dsts: [*c]const [*c]mod.impl.ggml_c.ggml_tensor,
    n_tensors: usize,
) callconv(.c) void {
    if (n_tensors == 0) {
        return;
    }

    for (0..n_tensors) |i| {
        mod.impl.assert(mod.impl.areSameLayout(&srcs[i].*, &dsts[i].*), "cannot copy tensors with different layouts");
    }

    const buf = tensorBuffer(dsts[0]);
    var same_buft = true;
    for (0..n_tensors) |i| {
        const dst_buf = tensorBuffer(dsts[i]);
        if (dst_buf.*.buft != buf.*.buft) {
            same_buft = false;
            break;
        }
    }

    // if cpy_tensor_n returns false, it must not have performed any copies
    if (same_buft and buf.*.iface.cpy_tensor_n != null) {
        if (buf.*.iface.cpy_tensor_n.?(buf, @ptrCast(srcs), @ptrCast(dsts), n_tensors)) {
            return;
        }
    }

    if (!same_buft) {
        const groups = tensorBuftGroups(n_tensors, dsts, struct {
            fn get(ds: [*c]const [*c]mod.impl.ggml_c.ggml_tensor, i: usize) mod.impl.ggml_c.ggml_backend_buffer_t {
                return tensorBuffer(ds[i]);
            }
        }.get);
        defer freeBuftGroups(groups);

        for (groups) |group| {
            const n = group.items.len;
            const g_srcs = std.heap.c_allocator.alloc([*c]const mod.impl.ggml_c.ggml_tensor, n) catch mod.impl.abort("backend: out of memory");
            defer std.heap.c_allocator.free(g_srcs);
            const g_dsts = std.heap.c_allocator.alloc([*c]mod.impl.ggml_c.ggml_tensor, n) catch mod.impl.abort("backend: out of memory");
            defer std.heap.c_allocator.free(g_dsts);

            for (group.items, 0..) |i, k| {
                g_srcs[k] = srcs[i];
                g_dsts[k] = dsts[i];
            }

            ggml_backend_tensor_copy_n(g_srcs.ptr, g_dsts.ptr, n);
        }
        return;
    }

    for (0..n_tensors) |i| {
        ggml_backend_tensor_copy(srcs[i], dsts[i]);
    }
}

/// Mirrors `NDEBUG`, which gates the slow-copy warning above and several
/// scheduler diagnostics. `build/llamacpp.zig` compiles the reference tree
/// with `-DNDEBUG` only in release, matching Zig's own optimize mode.
const debug_off = @import("builtin").mode != .debug;

/// Ports `ggml_backend_tensor_copy_async` (src/ggml-backend.cpp:908 @b1b770d90).
///
/// The fallback synchronizes *both* backends before a blocking copy, because
/// an async copy would otherwise have to wait for work already queued on each.
pub fn ggml_backend_tensor_copy_async(
    backend_src: mod.impl.ggml_c.ggml_backend_t,
    backend_dst: mod.impl.ggml_c.ggml_backend_t,
    src: [*c]const mod.impl.ggml_c.ggml_tensor,
    dst: [*c]mod.impl.ggml_c.ggml_tensor,
) callconv(.c) void {
    mod.impl.assert(mod.impl.areSameLayout(&src.*, &dst.*), "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    mod.impl.assert(backend_dst != null, "backend_dst");
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

/// Ports `ggml_backend_event_new` (src/ggml-backend.cpp:931 @b1b770d90).
///
/// Return: a new event, or null when the device is null or has no events —
/// the C notes a null device is allowed during the transition to the device
/// interface.
pub fn ggml_backend_event_new(device: mod.impl.ggml_c.ggml_backend_dev_t) callconv(.c) mod.impl.ggml_c.ggml_backend_event_t {
    // null device is allowed for the transition period to the device interface
    if (device == null or device.*.iface.event_new == null) {
        return null;
    }
    return device.*.iface.event_new.?(device);
}

/// Ports `ggml_backend_event_free` (src/ggml-backend.cpp:939 @b1b770d90).
pub fn ggml_backend_event_free(event: mod.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    if (event == null) {
        return;
    }
    event.*.device.*.iface.event_free.?(event.*.device, event);
}

/// Ports `ggml_backend_event_record` (src/ggml-backend.cpp:946 @b1b770d90).
pub fn ggml_backend_event_record(event: mod.impl.ggml_c.ggml_backend_event_t, backend: mod.impl.ggml_c.ggml_backend_t) callconv(.c) void {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(backend.*.iface.event_record != null, "backend->iface.event_record != NULL");
    backend.*.iface.event_record.?(backend, event);
}

/// Ports `ggml_backend_event_synchronize` (src/ggml-backend.cpp:953 @b1b770d90).
pub fn ggml_backend_event_synchronize(event: mod.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    mod.impl.assert(event != null, "event");
    mod.impl.assert(event.*.device.*.iface.event_synchronize != null, "event->device->iface.event_synchronize");
    event.*.device.*.iface.event_synchronize.?(event.*.device, event);
}

/// Ports `ggml_backend_event_wait` (src/ggml-backend.cpp:960 @b1b770d90).
pub fn ggml_backend_event_wait(backend: mod.impl.ggml_c.ggml_backend_t, event: mod.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    mod.impl.assert(backend != null, "backend");
    mod.impl.assert(backend.*.iface.event_wait != null, "backend->iface.event_wait != NULL");
    backend.*.iface.event_wait.?(backend, event);
}

/// Ports `ggml_backend_graph_optimize` (src/ggml-backend.cpp:967 @b1b770d90).
///
/// File-private in the C++ and used only by the scheduler, so it is `pub`
/// here rather than exported: `backend_sched.zig` is the one caller.
/// `params` carries the scheduler's `add_alloc_dep` callback, through which a
/// backend that runs nodes out of order keeps tensors alive longer.
pub fn graphOptimize(backend: mod.impl.ggml_c.ggml_backend_t, cgraph: *mod.impl.CGraph, params: *mod.impl.ggml_c.struct_ggml_backend_graph_optimize_params) void {
    mod.impl.assert(backend != null, "backend");
    if (backend.*.iface.graph_optimize) |f| {
        f(backend, @ptrCast(cgraph), params);
    }
}

// -----------------------------------------------------------------------------
// Backend device

/// Ports `ggml_backend_dev_name` (src/ggml-backend.cpp:976 @b1b770d90).
pub fn ggml_backend_dev_name(device: mod.impl.ggml_c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    mod.impl.assert(device != null, "device");
    return device.*.iface.get_name.?(device);
}

/// Ports `ggml_backend_dev_description` (src/ggml-backend.cpp:981 @b1b770d90).
pub fn ggml_backend_dev_description(device: mod.impl.ggml_c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    mod.impl.assert(device != null, "device");
    return device.*.iface.get_description.?(device);
}

/// Ports `ggml_backend_dev_memory` (src/ggml-backend.cpp:986 @b1b770d90).
pub fn ggml_backend_dev_memory(device: mod.impl.ggml_c.ggml_backend_dev_t, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    mod.impl.assert(device != null, "device");
    device.*.iface.get_memory.?(device, free, total);
}

/// Ports `ggml_backend_dev_type` (src/ggml-backend.cpp:991 @b1b770d90).
pub fn ggml_backend_dev_type(device: mod.impl.ggml_c.ggml_backend_dev_t) callconv(.c) mod.impl.ggml_c.enum_ggml_backend_dev_type {
    mod.impl.assert(device != null, "device");
    return device.*.iface.get_type.?(device);
}

/// Ports `ggml_backend_dev_get_props` (src/ggml-backend.cpp:996 @b1b770d90).
///
/// Zeroes `props` first, so a device that fills only some fields leaves the
/// rest defined.
pub fn ggml_backend_dev_get_props(device: mod.impl.ggml_c.ggml_backend_dev_t, props: [*c]mod.impl.ggml_c.struct_ggml_backend_dev_props) callconv(.c) void {
    mod.impl.assert(device != null, "device");
    props.* = std.mem.zeroes(mod.impl.ggml_c.struct_ggml_backend_dev_props);
    device.*.iface.get_props.?(device, props);
}

/// Ports `ggml_backend_dev_backend_reg` (src/ggml-backend.cpp:1002 @b1b770d90).
pub fn ggml_backend_dev_backend_reg(device: mod.impl.ggml_c.ggml_backend_dev_t) callconv(.c) mod.impl.ggml_c.ggml_backend_reg_t {
    mod.impl.assert(device != null, "device");
    return device.*.reg;
}

/// Ports `ggml_backend_dev_init` (src/ggml-backend.cpp:1007 @b1b770d90).
pub fn ggml_backend_dev_init(device: mod.impl.ggml_c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) mod.impl.ggml_c.ggml_backend_t {
    mod.impl.assert(device != null, "device");
    return device.*.iface.init_backend.?(device, params);
}

/// Ports `ggml_backend_dev_buffer_type` (src/ggml-backend.cpp:1012 @b1b770d90).
pub fn ggml_backend_dev_buffer_type(device: mod.impl.ggml_c.ggml_backend_dev_t) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_type_t {
    mod.impl.assert(device != null, "device");
    return device.*.iface.get_buffer_type.?(device);
}

/// Ports `ggml_backend_dev_host_buffer_type` (src/ggml-backend.cpp:1017 @b1b770d90).
///
/// Return: the device's pinned-host buffer type, or null when it has none.
pub fn ggml_backend_dev_host_buffer_type(device: mod.impl.ggml_c.ggml_backend_dev_t) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_type_t {
    mod.impl.assert(device != null, "device");
    const f = device.*.iface.get_host_buffer_type orelse return null;
    return f(device);
}

/// Ports `ggml_backend_dev_buffer_from_host_ptr` (src/ggml-backend.cpp:1026 @b1b770d90).
pub fn ggml_backend_dev_buffer_from_host_ptr(
    device: mod.impl.ggml_c.ggml_backend_dev_t,
    ptr: ?*anyopaque,
    size: usize,
    max_tensor_size: usize,
) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    mod.impl.assert(device != null, "device");
    return device.*.iface.buffer_from_host_ptr.?(device, ptr, size, max_tensor_size);
}

/// Ports `ggml_backend_dev_supports_op` (src/ggml-backend.cpp:1031 @b1b770d90).
pub fn ggml_backend_dev_supports_op(device: mod.impl.ggml_c.ggml_backend_dev_t, op: [*c]const mod.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    mod.impl.assert(device != null, "device");
    return device.*.iface.supports_op.?(device, op);
}

/// Ports `ggml_backend_dev_supports_buft` (src/ggml-backend.cpp:1036 @b1b770d90).
pub fn ggml_backend_dev_supports_buft(device: mod.impl.ggml_c.ggml_backend_dev_t, buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) bool {
    mod.impl.assert(device != null, "device");
    return device.*.iface.supports_buft.?(device, buft);
}

/// Ports `ggml_backend_dev_offload_op` (src/ggml-backend.cpp:1041 @b1b770d90).
///
/// Return: whether a higher-priority device wants to take this op off the CPU.
/// Optional; false when the device does not implement it.
pub fn ggml_backend_dev_offload_op(device: mod.impl.ggml_c.ggml_backend_dev_t, op: [*c]const mod.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    mod.impl.assert(device != null, "device");
    if (device.*.iface.offload_op) |f| {
        return f(device, op);
    }
    return false;
}

// -----------------------------------------------------------------------------
// Backend registry

/// Ports `ggml_backend_reg_name` (src/ggml-backend.cpp:1052 @b1b770d90).
pub fn ggml_backend_reg_name(reg: mod.impl.ggml_c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    mod.impl.assert(reg != null, "reg");
    return reg.*.iface.get_name.?(reg);
}

/// Ports `ggml_backend_reg_dev_count` (src/ggml-backend.cpp:1057 @b1b770d90).
pub fn ggml_backend_reg_dev_count(reg: mod.impl.ggml_c.ggml_backend_reg_t) callconv(.c) usize {
    mod.impl.assert(reg != null, "reg");
    return reg.*.iface.get_device_count.?(reg);
}

/// Ports `ggml_backend_reg_dev_get` (src/ggml-backend.cpp:1062 @b1b770d90).
pub fn ggml_backend_reg_dev_get(reg: mod.impl.ggml_c.ggml_backend_reg_t, index: usize) callconv(.c) mod.impl.ggml_c.ggml_backend_dev_t {
    mod.impl.assert(reg != null, "reg");
    return reg.*.iface.get_device.?(reg, index);
}

/// Ports `ggml_backend_reg_get_proc_address` (src/ggml-backend.cpp:1067 @b1b770d90).
///
/// Return: the named extension entry point, or null when the registry has no
/// `get_proc_address` or does not know the name.
pub fn ggml_backend_reg_get_proc_address(reg: mod.impl.ggml_c.ggml_backend_reg_t, name: [*c]const u8) callconv(.c) ?*anyopaque {
    mod.impl.assert(reg != null, "reg");
    const f = reg.*.iface.get_proc_address orelse return null;
    return f(reg, name);
}

// -----------------------------------------------------------------------------
// Multi-buffer
//
// A buffer that owns several others and fans free, clear and usage changes out
// to all of them. Its `iface` deliberately leaves everything else null, so a
// multi-buffer cannot be read from or written to directly.

/// Ports `struct ggml_backend_multi_buffer_context` (src/ggml-backend.cpp:1077 @b1b770d90).
const MultiBufferContext = extern struct {
    buffers: [*c]mod.impl.ggml_c.ggml_backend_buffer_t,
    n_buffers: usize,
};

/// Ports `ggml_backend_multi_buffer_free_buffer` (src/ggml-backend.cpp:1082 @b1b770d90).
///
/// Its address is also the tag `ggml_backend_buffer_is_multi_buffer` compares
/// against, so this function must stay a distinct symbol.
fn multiBufferFreeBuffer(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) void {
    mod.impl.assert(buffer != null, "buffer");
    const ctx: *MultiBufferContext = @ptrCast(@alignCast(buffer.*.context.?));
    for (0..ctx.n_buffers) |i| {
        ggml_backend_buffer_free(ctx.buffers[i]);
    }
    std.c.free(@ptrCast(ctx.buffers));
    std.c.free(ctx);
}

/// Ports `ggml_backend_multi_buffer_clear` (src/ggml-backend.cpp:1093 @b1b770d90).
fn multiBufferClear(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    mod.impl.assert(buffer != null, "buffer");
    const ctx: *MultiBufferContext = @ptrCast(@alignCast(buffer.*.context.?));
    for (0..ctx.n_buffers) |i| {
        ggml_backend_buffer_clear(ctx.buffers[i], value);
    }
}

/// Ports `ggml_backend_multi_buffer_i` (src/ggml-backend.cpp:1101 @b1b770d90).
const multi_buffer_i: mod.impl.ggml_c.ggml_backend_buffer_i = .{
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
    .set_tensor_n = null,
    .get_tensor_n = null,
    .cpy_tensor_n = null,
};

/// Ports `ggml_backend_multi_buffer_alloc_buffer` (src/ggml-backend.cpp:1118 @b1b770d90).
///
/// Parameters:
/// - `buffers`: the buffers to take ownership of; copied into the context.
/// - `n_buffers`: how many.
///
/// Return: a buffer whose size is the sum of theirs, and whose type is taken
/// from the first.
pub fn ggml_backend_multi_buffer_alloc_buffer(buffers: [*c]mod.impl.ggml_c.ggml_backend_buffer_t, n_buffers: usize) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    const ctx: *MultiBufferContext = @ptrCast(@alignCast(std.c.malloc(@sizeOf(MultiBufferContext)).?));
    ctx.n_buffers = n_buffers;
    ctx.buffers = @ptrCast(@alignCast(std.c.malloc(n_buffers * @sizeOf(mod.impl.ggml_c.ggml_backend_buffer_t))));

    mod.impl.assert(ctx.buffers != null, "ctx->buffers != NULL");

    var total_size: usize = 0;
    for (0..n_buffers) |i| {
        ctx.buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0].*.buft, multi_buffer_i, ctx, total_size);
}

/// Ports `ggml_backend_buffer_is_multi_buffer` (src/ggml-backend.cpp:1134 @b1b770d90).
///
/// Identified by the address of its `free_buffer`, which is how the C tells
/// them apart — there is no tag field.
pub fn ggml_backend_buffer_is_multi_buffer(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) bool {
    mod.impl.assert(buffer != null, "buffer");
    return buffer.*.iface.free_buffer == @as(?*const fn (mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) void, multiBufferFreeBuffer);
}

/// Ports `ggml_backend_multi_buffer_set_usage` (src/ggml-backend.cpp:1139 @b1b770d90).
pub fn ggml_backend_multi_buffer_set_usage(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, usage: mod.impl.ggml_c.enum_ggml_backend_buffer_usage) callconv(.c) void {
    mod.impl.assert(buffer != null, "buffer");
    mod.impl.assert(ggml_backend_buffer_is_multi_buffer(buffer), "ggml_backend_buffer_is_multi_buffer(buffer)");
    const ctx: *MultiBufferContext = @ptrCast(@alignCast(buffer.*.context.?));
    for (0..ctx.n_buffers) |i| {
        ggml_backend_buffer_set_usage(ctx.buffers[i], usage);
    }
}

/// Ports `ggml_dup_tensor_layout` (src/ggml-backend.cpp:1149 @b1b770d90).
///
/// `ggml_dup_tensor` gives the same shape and type but contiguous strides;
/// this copies the source's `nb` over the top so the layout matches too.
/// Shared with `backend_sched.zig`, which is the heavier user.
///
/// **Do not write this as `dup.*.nb[i] = tensor.*.nb[i]`.** Zig 0.16 types
/// `p.*.arr[i]` as the whole array when `p` is a C pointer (`[*c]T`), and
/// indexes in steps of the array's size: that loop copied 128 bytes starting
/// at `nb`, so every scheduler input copy inherited its source's `op`,
/// `op_params`, `flags` and first three `src` pointers. Outputs stayed right,
/// which is why nothing caught it until `scripts/node-diff --gpu` printed the
/// copies' ops. `p[0].arr[i]`, or a `*T`, indexes correctly.
pub fn dupTensorLayout(ctx: ?*mod.impl.ggml_c.ggml_context, tensor: [*c]const mod.impl.ggml_c.ggml_tensor) [*c]mod.impl.ggml_c.ggml_tensor {
    const dup = mod.impl.one(mod.impl.ggml_c.ggml_tensor, mod.impl.ggml_c.ggml_dup_tensor(ctx, tensor));
    const src = mod.impl.one(mod.impl.ggml_c.ggml_tensor, @constCast(tensor));
    for (0..mod.impl.ggml_c.GGML_MAX_DIMS) |i| {
        dup.nb[i] = src.nb[i];
    }
    return dup;
}

/// Ports `ggml_is_view_op` (src/ggml-backend.cpp:1157 @b1b770d90).
///
/// The four ops that produce a view rather than compute anything; the
/// scheduler skips them everywhere it walks a graph.
pub fn isViewOp(op: mod.impl.ggml_c.enum_ggml_op) bool {
    return op == mod.impl.ggml_c.GGML_OP_VIEW or op == mod.impl.ggml_c.GGML_OP_RESHAPE or op == mod.impl.ggml_c.GGML_OP_PERMUTE or op == mod.impl.ggml_c.GGML_OP_TRANSPOSE;
}

// -----------------------------------------------------------------------------
// Utils

/// Ports `ggml_op_alloc_size_may_expand` (src/ggml-backend.cpp:2575 @b1b770d90).
///
/// The ops whose output a backend's `get_alloc_size` may size beyond
/// `ggml_nbytes`, for scratch it keeps beside the result
/// ([TAG_ALLOC_SIZE_EXPAND]). `ggml_backend_buft_get_alloc_size` asserts
/// every other op stays within it.
pub fn ggml_op_alloc_size_may_expand(op: mod.impl.ggml_c.enum_ggml_op) callconv(.c) bool {
    return switch (op) {
        mod.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT,
        mod.impl.ggml_c.GGML_OP_MUL_MAT,
        mod.impl.ggml_c.GGML_OP_MUL_MAT_ID,
        mod.impl.ggml_c.GGML_OP_CUMSUM,
        mod.impl.ggml_c.GGML_OP_ARGSORT,
        mod.impl.ggml_c.GGML_OP_TOP_K,
        mod.impl.ggml_c.GGML_OP_ARGMAX,
        => true,
        else => false,
    };
}

/// Ports `ggml_backend_view_init` (src/ggml-backend.cpp:2590 @b1b770d90).
///
/// Points a view at its source's buffer and offset, then lets the buffer
/// initialise it. Asserts the view is not already attached.
pub fn ggml_backend_view_init(tensor: [*c]mod.impl.ggml_c.ggml_tensor) callconv(.c) mod.impl.ggml_c.enum_ggml_status {
    mod.impl.assert(tensor != null, "tensor");
    mod.impl.assert(tensor.*.buffer == null, "tensor->buffer == NULL");
    mod.impl.assert(tensor.*.view_src != null, "tensor->view_src != NULL");
    mod.impl.assert(tensor.*.view_src.*.buffer != null, "tensor->view_src->buffer != NULL");
    mod.impl.assert(tensor.*.view_src.*.data != null, "tensor->view_src->data != NULL");

    tensor.*.buffer = tensor.*.view_src.*.buffer;
    tensor.*.data = @as([*]u8, @ptrCast(tensor.*.view_src.*.data)) + tensor.*.view_offs;
    return ggml_backend_buffer_init_tensor(tensor.*.buffer, tensor);
}

/// Ports `ggml_backend_tensor_alloc` (src/ggml-backend.cpp:2602 @b1b770d90).
///
/// Parameters:
/// - `buffer`: the buffer the tensor will live in.
/// - `tensor`: must be unattached and not a view.
/// - `addr`: where inside `buffer` it goes; bounds-checked unless the buffer
///   is a meta buffer, which has no real storage to overrun.
pub fn ggml_backend_tensor_alloc(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]mod.impl.ggml_c.ggml_tensor, addr: ?*anyopaque) callconv(.c) mod.impl.ggml_c.enum_ggml_status {
    mod.impl.assert(tensor != null, "tensor");
    mod.impl.assert(tensor.*.buffer == null, "tensor->buffer == NULL");
    mod.impl.assert(tensor.*.data == null, "tensor->data == NULL");
    mod.impl.assert(tensor.*.view_src == null, "tensor->view_src == NULL");
    mod.impl.assert(@intFromPtr(addr) >= @intFromPtr(ggml_backend_buffer_get_base(buffer)), "addr >= ggml_backend_buffer_get_base(buffer)");
    mod.impl.assert(
        mod.impl.ggml_c.ggml_backend_buffer_is_meta(buffer) or
            @intFromPtr(addr) + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                @intFromPtr(ggml_backend_buffer_get_base(buffer)) + ggml_backend_buffer_get_size(buffer),
        "tensor does not fit in the buffer",
    );

    tensor.*.buffer = buffer;
    tensor.*.data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

/// Ports `graph_copy_dup_tensor` (src/ggml-backend.cpp:2617 @b1b770d90).
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
    hash_set: mod.impl.HashSet,
    node_copies: [*c][*c]mod.impl.ggml_c.ggml_tensor,
    ctx_allocated: ?*mod.impl.ggml_c.ggml_context,
    ctx_unallocated: ?*mod.impl.ggml_c.ggml_context,
    src: *mod.impl.ggml_c.ggml_tensor,
) *mod.impl.ggml_c.ggml_tensor {
    mod.impl.assert(src.data != null, "graph must be allocated");

    var hs = hash_set;
    const id = mod.impl.hashInsert(&hs, src);
    if (id == mod.impl.hashset_already_exists) {
        return mod.impl.one(mod.impl.ggml_c.ggml_tensor, node_copies[mod.impl.hashFind(&hs, src)]);
    }

    const dst = mod.impl.one(mod.impl.ggml_c.ggml_tensor, dupTensorLayout(
        if (src.data != null and src.view_src == null) ctx_allocated else ctx_unallocated,
        src,
    ));
    if (src.view_src != null) {
        dst.view_src = graphCopyDupTensor(hs, node_copies, ctx_allocated, ctx_unallocated, mod.impl.one(mod.impl.ggml_c.ggml_tensor, src.view_src));
        dst.view_offs = src.view_offs;
    }
    dst.op = src.op;
    dst.flags = src.flags;
    @memcpy(std.mem.asBytes(&dst.op_params), std.mem.asBytes(&src.op_params));
    _ = mod.impl.ggml_c.ggml_set_name(dst, &src.name);

    // copy src
    for (0..mod.impl.ggml_c.GGML_MAX_SRC) |i| {
        const s = src.src[i];
        if (s == null) {
            continue;
        }
        dst.src[i] = graphCopyDupTensor(hs, node_copies, ctx_allocated, ctx_unallocated, mod.impl.one(mod.impl.ggml_c.ggml_tensor, s));
    }

    node_copies[id] = dst;
    return dst;
}

/// Ports `graph_copy_init_tensor` (src/ggml-backend.cpp:2651 @b1b770d90).
///
/// Walks the same graph a second time, now that storage exists: views are
/// wired to their source and everything else has its data copied across.
fn graphCopyInitTensor(
    hash_set: [*c]mod.impl.HashSet,
    node_copies: [*c][*c]mod.impl.ggml_c.ggml_tensor,
    node_init: [*c]bool,
    src: *mod.impl.ggml_c.ggml_tensor,
) void {
    const id = mod.impl.hashFind(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    const dst = mod.impl.one(mod.impl.ggml_c.ggml_tensor, node_copies[id]);
    if (dst.view_src != null) {
        graphCopyInitTensor(hash_set, node_copies, node_init, mod.impl.one(mod.impl.ggml_c.ggml_tensor, src.view_src));
        const status = ggml_backend_view_init(dst);
        mod.impl.assert(status == mod.impl.ggml_c.GGML_STATUS_SUCCESS, "status == GGML_STATUS_SUCCESS");
    } else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (0..mod.impl.ggml_c.GGML_MAX_SRC) |i| {
        const s = src.src[i];
        if (s == null) {
            continue;
        }
        graphCopyInitTensor(hash_set, node_copies, node_init, mod.impl.one(mod.impl.ggml_c.ggml_tensor, s));
    }
}

/// Ports `ggml_backend_graph_copy` (src/ggml-backend.cpp:2678 @b1b770d90).
///
/// Parameters:
/// - `backend`: the backend the copy's tensors are allocated on.
/// - `graph`: the graph to duplicate; must already be allocated.
///
/// Return: the copy, with an all-null struct on failure. Released with
/// `ggml_backend_graph_copy_free`.
pub fn ggml_backend_graph_copy(backend: mod.impl.ggml_c.ggml_backend_t, graph: *mod.impl.CGraph) callconv(.c) mod.impl.ggml_c.struct_ggml_backend_graph_copy {
    // The C asserts `graph` is non-null; `*impl.CGraph` says the same thing
    // in the signature.
    var hash_set = mod.graph.ggml_hash_set_new(graph.*.visited_hash_set.size);
    const node_copies: [*c][*c]mod.impl.ggml_c.ggml_tensor = @ptrCast(@alignCast(std.c.calloc(hash_set.size, @sizeOf([*c]mod.impl.ggml_c.ggml_tensor))));
    const node_init: [*c]bool = @ptrCast(std.c.calloc(hash_set.size, @sizeOf(bool)));

    const params: mod.impl.ggml_c.ggml_init_params = .{
        .mem_size = mod.impl.ggml_c.ggml_tensor_overhead() * hash_set.size + mod.impl.ggml_c.ggml_graph_overhead_custom(@intCast(graph.*.size), false),
        .mem_buffer = null,
        .no_alloc = true,
    };

    const ctx_allocated = mod.impl.ggml_c.ggml_init(params);
    const ctx_unallocated = mod.impl.ggml_c.ggml_init(params);

    const failed: mod.impl.ggml_c.struct_ggml_backend_graph_copy = .{
        .buffer = null,
        .ctx_allocated = null,
        .ctx_unallocated = null,
        .graph = null,
    };

    if (ctx_allocated == null or ctx_unallocated == null) {
        mod.impl.logError("%s: failed to allocate context for graph copy\n", .{"ggml_backend_graph_copy"});
        mod.graph.ggml_hash_set_free(&hash_set);
        std.c.free(@ptrCast(node_copies));
        std.c.free(@ptrCast(node_init));
        mod.impl.ggml_c.ggml_free(ctx_allocated);
        mod.impl.ggml_c.ggml_free(ctx_unallocated);
        return failed;
    }

    // dup nodes
    for (0..@intCast(graph.*.n_nodes)) |i| {
        const node = graph.*.nodes[i];
        _ = graphCopyDupTensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node.?);
    }

    // allocate nodes
    const buffer = mod.impl.ggml_c.ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == null) {
        mod.impl.logError("%s: failed to allocate buffer for graph copy\n", .{"ggml_backend_graph_copy"});
        mod.graph.ggml_hash_set_free(&hash_set);
        std.c.free(@ptrCast(node_copies));
        std.c.free(@ptrCast(node_init));
        mod.impl.ggml_c.ggml_free(ctx_allocated);
        mod.impl.ggml_c.ggml_free(ctx_unallocated);
        return failed;
    }

    // copy data and init views
    for (0..@intCast(graph.*.n_nodes)) |i| {
        const node = graph.*.nodes[i];
        graphCopyInitTensor(&hash_set, node_copies, node_init, node.?);
    }

    // build graph copy
    const graph_copy: *mod.impl.CGraph = @ptrCast(@alignCast(mod.impl.ggml_c.ggml_new_graph_custom(ctx_allocated, @intCast(graph.*.size), false)));
    for (0..@intCast(graph.*.n_nodes)) |i| {
        const node = graph.*.nodes[i].?;
        graph_copy.nodes[i] = node_copies[mod.impl.hashFind(&hash_set, node)];
    }
    graph_copy.n_nodes = graph.*.n_nodes;

    mod.graph.ggml_hash_set_free(&hash_set);
    std.c.free(@ptrCast(node_copies));
    std.c.free(@ptrCast(node_init));

    return .{
        .buffer = buffer,
        .ctx_allocated = ctx_allocated,
        .ctx_unallocated = ctx_unallocated,
        .graph = @ptrCast(graph_copy),
    };
}

/// Ports `ggml_backend_graph_copy_free` (src/ggml-backend.cpp:2760 @b1b770d90).
pub fn ggml_backend_graph_copy_free(copy: mod.impl.ggml_c.struct_ggml_backend_graph_copy) callconv(.c) void {
    ggml_backend_buffer_free(copy.buffer);
    mod.impl.ggml_c.ggml_free(copy.ctx_allocated);
    mod.impl.ggml_c.ggml_free(copy.ctx_unallocated);
}

/// Ports `ggml_backend_compare_graph_backend` (src/ggml-backend.cpp:2766 @b1b770d90).
///
/// Runs `graph` on two backends and hands each pair of results to `callback`.
/// With `test_nodes` it computes both graphs whole and compares only those
/// nodes; without, it steps node by node so a divergence is attributed to the
/// op that caused it. This is what `test-backend-ops --diff` drives.
///
/// Return: false if the graph copy could not be allocated.
pub fn ggml_backend_compare_graph_backend(
    backend1: mod.impl.ggml_c.ggml_backend_t,
    backend2: mod.impl.ggml_c.ggml_backend_t,
    graph: *mod.impl.CGraph,
    callback: mod.impl.ggml_c.ggml_backend_eval_callback,
    user_data: ?*anyopaque,
    test_nodes: [*c]const [*c]const mod.impl.ggml_c.ggml_tensor,
    num_test_nodes: usize,
) callconv(.c) bool {
    const copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == null) {
        return false;
    }

    const g1 = graph;
    const g2: *mod.impl.CGraph = @ptrCast(@alignCast(copy.graph.?));

    mod.impl.assert(g1.*.n_nodes == g2.*.n_nodes, "g1->n_nodes == g2->n_nodes");

    if (num_test_nodes != 0) {
        mod.impl.assert(test_nodes != null, "test_nodes");
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
        mod.impl.assert(verified, "verified");
    } else {
        for (0..@intCast(g1.*.n_nodes)) |i| {
            const t1 = g1.*.nodes[i].?;
            const t2 = g2.*.nodes[i].?;

            mod.impl.assert(t1.*.op == t2.*.op and mod.impl.areSameLayout(&t1.*, &t2.*), "t1->op == t2->op && ggml_are_same_layout(t1, t2)");

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

/// Ports `ggml_backend_cpu_buffer_get_base` (src/ggml-backend.cpp:2823 @b1b770d90).
///
/// The context *is* the allocation, rounded up to `TENSOR_ALIGNMENT`. A
/// from-pointer buffer is already aligned and the rounding is a no-op.
fn cpuBufferGetBase(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    mod.impl.assert(buffer != null, "buffer");
    var data = @intFromPtr(buffer.*.context);

    // align the buffer
    if (data % mod.impl.tensor_alignment != 0) {
        data = stdlib.num.pad(usize, data, mod.impl.tensor_alignment);
    }

    return @ptrFromInt(data);
}

/// Ports `ggml_backend_cpu_buffer_free_buffer` (src/ggml-backend.cpp:2835 @b1b770d90).
fn cpuBufferFreeBuffer(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) void {
    mod.impl.assert(buffer != null, "buffer");
    mod.runtime.ggml_aligned_free(buffer.*.context, buffer.*.size);
}

/// Ports `ggml_backend_cpu_buffer_memset_tensor` (src/ggml-backend.cpp:2840 @b1b770d90).
fn cpuBufferMemsetTensor(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]mod.impl.ggml_c.ggml_tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    _ = buffer;
    mod.impl.assert(tensor != null, "tensor");
    @memset((@as([*]u8, @ptrCast(tensor.*.data)) + offset)[0..size], value);
}

/// Ports `ggml_backend_cpu_buffer_set_tensor` (src/ggml-backend.cpp:2847 @b1b770d90).
fn cpuBufferSetTensor(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]mod.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    _ = buffer;
    mod.impl.assert(tensor != null, "tensor");
    @memcpy(
        (@as([*]u8, @ptrCast(tensor.*.data)) + offset)[0..size],
        @as([*]const u8, @ptrCast(data.?))[0..size],
    );
}

/// Ports `ggml_backend_cpu_buffer_get_tensor` (src/ggml-backend.cpp:2854 @b1b770d90).
fn cpuBufferGetTensor(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]const mod.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    _ = buffer;
    mod.impl.assert(tensor != null, "tensor");
    @memcpy(
        @as([*]u8, @ptrCast(data.?))[0..size],
        (@as([*]const u8, @ptrCast(tensor.*.data)) + offset)[0..size],
    );
}

/// Ports `ggml_backend_cpu_buffer_cpy_tensor` (src/ggml-backend.cpp:2861 @b1b770d90).
///
/// Return: true only when the source is host memory, which is the one case a
/// plain `memcpy` can serve.
fn cpuBufferCpyTensor(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, src: [*c]const mod.impl.ggml_c.ggml_tensor, dst: [*c]mod.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    _ = buffer;
    mod.impl.assert(src != null, "src");
    if (ggml_backend_buffer_is_host(src.*.buffer)) {
        const n = mod.impl.ggml_c.ggml_nbytes(src);
        @memcpy(@as([*]u8, @ptrCast(dst.*.data))[0..n], @as([*]const u8, @ptrCast(src.*.data))[0..n]);
        return true;
    }
    return false;
}

/// Ports `ggml_backend_cpu_buffer_clear` (src/ggml-backend.cpp:2872 @b1b770d90).
///
/// Clears from the *context*, not from `get_base`, so the alignment padding
/// is cleared too. That is what the C does.
fn cpuBufferClear(buffer: mod.impl.ggml_c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    mod.impl.assert(buffer != null, "buffer");
    @memset(@as([*]u8, @ptrCast(buffer.*.context))[0..buffer.*.size], value);
}

/// Ports `ggml_backend_cpu_buffer_i` (src/ggml-backend.cpp:2877 @b1b770d90).
const cpu_buffer_i: mod.impl.ggml_c.ggml_backend_buffer_i = .{
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
    .set_tensor_n = null,
    .get_tensor_n = null,
    .cpy_tensor_n = null,
};

/// Ports `ggml_backend_cpu_buffer_from_ptr_i` (src/ggml-backend.cpp:2894 @b1b770d90).
///
/// Identical to `cpu_buffer_i` but for `free_buffer`: the pointer is the
/// caller's and must not be freed.
const cpu_buffer_from_ptr_i: mod.impl.ggml_c.ggml_backend_buffer_i = .{
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
    .set_tensor_n = null,
    .get_tensor_n = null,
    .cpy_tensor_n = null,
};

// -----------------------------------------------------------------------------
// CPU backend buffer type

/// Ports `ggml_backend_cpu_buffer_type_get_name` (src/ggml-backend.cpp:2915 @b1b770d90).
fn cpuBufferTypeGetName(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    _ = buft;
    return "CPU";
}

/// Ports `ggml_backend_cpu_buffer_type_alloc_buffer` (src/ggml-backend.cpp:2921 @b1b770d90).
fn cpuBufferTypeAllocBuffer(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t, size: usize) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    const data = mod.runtime.ggml_aligned_malloc(size);

    if (data == null) {
        mod.impl.logError("%s: failed to allocate buffer of size %zu\n", .{ "ggml_backend_cpu_buffer_type_alloc_buffer", size });
        return null;
    }

    return ggml_backend_buffer_init(buft, cpu_buffer_i, data, size);
}

/// Ports `ggml_backend_cpu_buffer_type_get_alignment` (src/ggml-backend.cpp:2932 @b1b770d90).
fn cpuBufferTypeGetAlignment(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return mod.impl.tensor_alignment;
}

/// Ports `ggml_backend_cpu_buffer_type_is_host` (src/ggml-backend.cpp:2938 @b1b770d90).
fn cpuBufferTypeIsHost(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = buft;
    return true;
}

/// Backing storage for `ggml_backend_cpu_buffer_type`.
///
/// The C++ puts this in a function-local `static`, so its address is stable
/// and it is shared by every caller; a file-scope `var` is the same thing.
/// `device` is null, with a `FIXME` in the C noting it should be the CPU
/// registry's device.
var cpu_buffer_type_storage: mod.impl.ggml_c.ggml_backend_buffer_type = .{
    .iface = .{
        .get_name = cpuBufferTypeGetName,
        .alloc_buffer = cpuBufferTypeAllocBuffer,
        .alloc_buffer_n = null,
        .get_alignment = cpuBufferTypeGetAlignment,
        .get_max_size = null, // defaults to SIZE_MAX
        .get_alloc_size = null, // defaults to ggml_nbytes
        .get_alloc_size_n = null,
        .is_host = cpuBufferTypeIsHost,
    },
    .device = null, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0)
    .context = null,
};

/// Ports `ggml_backend_cpu_buffer_type` (src/ggml-backend.cpp:2944 @b1b770d90).
///
/// Return: the shared CPU buffer type. Not owned by the caller.
pub fn ggml_backend_cpu_buffer_type() callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_type_t {
    return &cpu_buffer_type_storage;
}

/// Ports `ggml_backend_cpu_buffer_from_ptr_type_get_name` (src/ggml-backend.cpp:2963 @b1b770d90).
fn cpuBufferFromPtrTypeGetName(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    _ = buft;
    return "CPU_Mapped";
}

/// Backing storage for `cpuBufferFromPtrType`.
///
/// A second, distinct object from `cpu_buffer_type_storage`: the two differ
/// only in `get_name`, but `ggml_backend_sched` compares buffer types by
/// address, so they must not be merged.
var cpu_buffer_from_ptr_type_storage: mod.impl.ggml_c.ggml_backend_buffer_type = .{
    .iface = .{
        .get_name = cpuBufferFromPtrTypeGetName,
        .alloc_buffer = cpuBufferTypeAllocBuffer,
        .alloc_buffer_n = null,
        .get_alignment = cpuBufferTypeGetAlignment,
        .get_max_size = null, // defaults to SIZE_MAX
        .get_alloc_size = null, // defaults to ggml_nbytes
        .get_alloc_size_n = null,
        .is_host = cpuBufferTypeIsHost,
    },
    .device = null, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0)
    .context = null,
};

/// Ports `ggml_backend_cpu_buffer_from_ptr_type` (src/ggml-backend.cpp:2969 @b1b770d90).
///
/// File-private in the C++, and only `ggml_backend_cpu_buffer_from_ptr` calls
/// it.
fn cpuBufferFromPtrType() mod.impl.ggml_c.ggml_backend_buffer_type_t {
    return &cpu_buffer_from_ptr_type_storage;
}

/// Ports `ggml_backend_cpu_buffer_from_ptr` (src/ggml-backend.cpp:2988 @b1b770d90).
///
/// Wraps memory the caller owns. The buffer never frees it.
pub fn ggml_backend_cpu_buffer_from_ptr(ptr: ?*anyopaque, size: usize) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    mod.impl.assert(@intFromPtr(ptr) % mod.impl.tensor_alignment == 0, "buffer pointer must be aligned");
    return ggml_backend_buffer_init(cpuBufferFromPtrType(), cpu_buffer_from_ptr_i, ptr, size);
}

// -----------------------------------------------------------------------------
// Imported from the sibling ported files
//
// `ggml-impl.h` declares these but cannot be imported, so they are redeclared
// here exactly as `context.zig` does. `ggml_graph_view` is `graph.zig`'s,
// taking and returning `impl.CGraph` rather than the opaque import type.

/// Calls `graph.zig`'s `ggml_graph_view` under its C name.
inline fn graphView(cgraph0: *mod.impl.CGraph, first: c_int, last: c_int) mod.impl.CGraph {
    return @call(.auto, @extern(*const fn (*mod.impl.CGraph, c_int, c_int) callconv(.c) mod.impl.CGraph, .{ .name = "ggml_graph_view" }), .{ cgraph0, first, last });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the CPU buffer type reports its name, alignment and hostness" {
    const buft = ggml_backend_cpu_buffer_type();
    try std.testing.expectEqualStrings("CPU", std.mem.span(ggml_backend_buft_name(buft)));
    try std.testing.expectEqual(@as(usize, mod.impl.tensor_alignment), ggml_backend_buft_get_alignment(buft));
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

    const gctx = mod.impl.ggml_c.ggml_init(.{ .mem_size = 1024 * 1024, .mem_buffer = null, .no_alloc = true });
    defer mod.impl.ggml_c.ggml_free(gctx);
    const t = mod.impl.ggml_c.ggml_new_tensor_1d(gctx, mod.impl.ggml_c.GGML_TYPE_F32, 8);

    const status = ggml_backend_tensor_alloc(buffer, t, ggml_backend_buffer_get_base(buffer));
    try std.testing.expectEqual(@as(mod.impl.ggml_c.enum_ggml_status, mod.impl.ggml_c.GGML_STATUS_SUCCESS), status);

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
    var parts = [_]mod.impl.ggml_c.ggml_backend_buffer_t{
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
    ggml_backend_buffer_set_usage(multi, mod.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (parts) |p| {
        try std.testing.expectEqual(
            @as(mod.impl.ggml_c.enum_ggml_backend_buffer_usage, mod.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS),
            ggml_backend_buffer_get_usage(p),
        );
    }
}

test "a from-ptr buffer wraps caller memory and never frees it" {
    var storage: [256]u8 align(mod.impl.tensor_alignment) = @splat(0);
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
    try std.testing.expect(isViewOp(mod.impl.ggml_c.GGML_OP_VIEW));
    try std.testing.expect(isViewOp(mod.impl.ggml_c.GGML_OP_RESHAPE));
    try std.testing.expect(isViewOp(mod.impl.ggml_c.GGML_OP_PERMUTE));
    try std.testing.expect(isViewOp(mod.impl.ggml_c.GGML_OP_TRANSPOSE));
    try std.testing.expect(!isViewOp(mod.impl.ggml_c.GGML_OP_ADD));
    try std.testing.expect(!isViewOp(mod.impl.ggml_c.GGML_OP_MUL_MAT));
    try std.testing.expect(!isViewOp(mod.impl.ggml_c.GGML_OP_CPY));
}

test "dupTensorLayout keeps the source's strides, not contiguous ones" {
    const gctx = mod.impl.ggml_c.ggml_init(.{ .mem_size = 1024 * 1024, .mem_buffer = null, .no_alloc = true });
    defer mod.impl.ggml_c.ggml_free(gctx);

    const base = mod.impl.ggml_c.ggml_new_tensor_2d(gctx, mod.impl.ggml_c.GGML_TYPE_F32, 4, 8);
    // A transpose has non-contiguous strides, which `ggml_dup_tensor` alone
    // would discard -- that is the whole reason this helper exists.
    const t = mod.impl.one(mod.impl.ggml_c.ggml_tensor, mod.impl.ggml_c.ggml_transpose(gctx, base));
    const dup = mod.impl.one(mod.impl.ggml_c.ggml_tensor, dupTensorLayout(gctx, t));

    // Through single-item pointers: `t.*.nb[i]` on a `[*c]` is the whole
    // array, and this test used to pass by comparing two wrong reads.
    for (0..mod.impl.ggml_c.GGML_MAX_DIMS) |i| {
        try std.testing.expectEqual(t.nb[i], dup.nb[i]);
        try std.testing.expectEqual(t.ne[i], dup.ne[i]);
    }
    try std.testing.expectEqual(t.type, dup.type);
}

test "dupTensorLayout touches nothing past the strides" {
    const gctx = mod.impl.ggml_c.ggml_init(.{ .mem_size = 1024 * 1024, .mem_buffer = null, .no_alloc = true });
    defer mod.impl.ggml_c.ggml_free(gctx);

    const a = mod.impl.ggml_c.ggml_new_tensor_2d(gctx, mod.impl.ggml_c.GGML_TYPE_F32, 4, 8);
    const b = mod.impl.ggml_c.ggml_new_tensor_2d(gctx, mod.impl.ggml_c.GGML_TYPE_F32, 4, 8);
    const src = mod.impl.one(mod.impl.ggml_c.ggml_tensor, mod.impl.ggml_c.ggml_add(gctx, a, b));
    _ = mod.impl.ggml_c.ggml_set_output(src);

    const dup = mod.impl.one(mod.impl.ggml_c.ggml_tensor, dupTensorLayout(gctx, src));

    // A fresh tensor: no op, no sources, no flags. The C pointer-indexing bug
    // above copied all three from `src`.
    try std.testing.expectEqual(@as(mod.impl.ggml_c.enum_ggml_op, mod.impl.ggml_c.GGML_OP_NONE), dup.op);
    try std.testing.expectEqual(@as(i32, 0), dup.flags);
    try std.testing.expect(dup.src[0] == null and dup.src[1] == null);
}
