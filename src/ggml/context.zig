//! Contexts, and the tensors allocated inside them.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml.c` (v0.3.0, `c1d0e7a00`), roughly
//! lines 1267-1283 and 1608-2026. Each function names the C function it
//! replaces and the line it began at.
//!
//! # How a context allocates
//!
//! A context is one flat buffer plus a linked list of `Object` headers laid
//! out inside it, each aligned to `GGML_MEM_ALIGN`. Allocation only ever
//! appends: there is no free, and `ggml_reset` drops the whole list at once.
//! That is why building a graph is cheap and why a context has to be sized up
//! front.
//!
//! A tensor is an `Object` header, then the `ggml_tensor` struct, then
//! optionally its data in the same allocation. When the data is inline it sits
//! immediately after the struct, which is what `result + 1` means in the C.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("impl.zig");
const types = @import("types.zig");
const c = impl.c;

extern fn vsnprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ap: std.builtin.VaList) c_int;
extern fn printf(fmt: [*:0]const u8, ...) c_int;

/// True when the C build would leave `NDEBUG` unset. `ggml_new_object` aborts
/// rather than returning null in that case, and the difference is observable.
const debug_build = builtin.mode == .Debug;

/// Ports `struct ggml_context` (ggml.c:974 @c1d0e7a00).
///
/// Opaque to callers -- only a pointer crosses the ABI -- but the layout still
/// has to match, because a context created by C code may be freed by this code
/// and the other way round.
pub const Context = extern struct {
    mem_size: usize,
    mem_buffer: ?*anyopaque,
    mem_buffer_owned: bool,
    no_alloc: bool,

    n_objects: c_int,

    objects_begin: ?*impl.Object,
    objects_end: ?*impl.Object,
};

/// Ports `GGML_ASSERT_ALIGNED` (ggml.c:1605 @c1d0e7a00).
inline fn assertAligned(ptr: ?*const anyopaque) void {
    std.debug.assert(@intFromPtr(ptr) % c.GGML_MEM_ALIGN == 0);
}

/// Guards the one-time initialisation in `ggml_init`.
var is_first_call = true;

// -----------------------------------------------------------------------------
// Context lifecycle

/// Ports `ggml_init` (ggml.c:1610 @c1d0e7a00).
///
/// A caller may supply its own buffer, in which case the context does not own
/// it and will not free it. Passing `mem_size` of 0 is allowed and rounds up to
/// one alignment unit, so a context always has somewhere to put its first
/// object header.
pub export fn ggml_init(params: c.struct_ggml_init_params) ?*Context {
    // The C guards this with a critical section because the time system needs
    // initialising exactly once, on Windows.
    c.ggml_critical_section_start();
    if (is_first_call) {
        ggml_time_init();
        is_first_call = false;
    }
    c.ggml_critical_section_end();

    const ctx: *Context = @ptrCast(@alignCast(std.c.malloc(@sizeOf(Context)) orelse
        impl.abort("fatal error")));

    var mem_size = params.mem_size;
    if (mem_size == 0) mem_size = c.GGML_MEM_ALIGN;

    // A caller-supplied buffer is taken at its stated size; one we allocate is
    // rounded up so every object header inside stays aligned.
    const owned = params.mem_buffer == null;
    if (owned) mem_size = impl.pad(mem_size, c.GGML_MEM_ALIGN);

    ctx.* = .{
        .mem_size = mem_size,
        .mem_buffer = if (owned) ggml_aligned_malloc(mem_size) else params.mem_buffer,
        .mem_buffer_owned = owned,
        .no_alloc = params.no_alloc,
        .n_objects = 0,
        .objects_begin = null,
        .objects_end = null,
    };

    impl.assert(ctx.mem_buffer != null, "ctx->mem_buffer != NULL");
    assertAligned(ctx.mem_buffer);

    return ctx;
}

/// Ports `ggml_reset` (ggml.c:1652 @c1d0e7a00).
///
/// Drops every object without freeing memory, so the context can be refilled.
/// Any tensor pointer handed out before this is invalid afterwards.
export fn ggml_reset(ctx_opt: ?*Context) void {
    const ctx = ctx_opt orelse return;
    ctx.n_objects = 0;
    ctx.objects_begin = null;
    ctx.objects_end = null;
}

/// Ports `ggml_free` (ggml.c:1662 @c1d0e7a00).
pub export fn ggml_free(ctx_opt: ?*Context) void {
    const ctx = ctx_opt orelse return;
    if (ctx.mem_buffer_owned) ggml_aligned_free(ctx.mem_buffer, ctx.mem_size);
    std.c.free(ctx);
}

/// Ports `ggml_used_mem` (ggml.c:1674 @c1d0e7a00).
pub export fn ggml_used_mem(ctx: *const Context) usize {
    const end = ctx.objects_end orelse return 0;
    return end.offs + end.size;
}

/// Ports `ggml_get_no_alloc` (ggml.c:1678 @c1d0e7a00).
pub export fn ggml_get_no_alloc(ctx: *Context) bool {
    return ctx.no_alloc;
}

/// Ports `ggml_set_no_alloc` (ggml.c:1682 @c1d0e7a00).
pub export fn ggml_set_no_alloc(ctx: *Context, no_alloc: bool) void {
    ctx.no_alloc = no_alloc;
}

/// Ports `ggml_get_mem_buffer` (ggml.c:1686 @c1d0e7a00).
export fn ggml_get_mem_buffer(ctx: *const Context) ?*anyopaque {
    return ctx.mem_buffer;
}

/// Ports `ggml_get_mem_size` (ggml.c:1690 @c1d0e7a00).
export fn ggml_get_mem_size(ctx: *const Context) usize {
    return ctx.mem_size;
}

/// Ports `ggml_get_max_tensor_size` (ggml.c:1694 @c1d0e7a00).
export fn ggml_get_max_tensor_size(ctx: *const Context) usize {
    var max_size: usize = 0;
    var tensor = ggml_get_first_tensor(ctx);
    while (tensor) |t| : (tensor = ggml_get_next_tensor(ctx, t)) {
        max_size = @max(max_size, types.ggml_nbytes(t));
    }
    return max_size;
}

// -----------------------------------------------------------------------------
// Object allocation

/// Ports `ggml_new_object` (ggml.c:1707 @c1d0e7a00).
///
/// Appends an object header at the end of the pool. The overflow checks are
/// not theoretical: `size` comes from tensor dimensions, which callers compute
/// from untrusted model metadata.
///
/// Return: the new header, or null when the pool is exhausted or the
/// arithmetic would overflow. In a debug build the exhausted case aborts
/// instead, matching the C.
pub fn newObject(ctx: *Context, obj_type: c.enum_ggml_object_type, size: usize) ?*impl.Object {
    const obj_cur = ctx.objects_end;

    const cur_offs = if (obj_cur) |o| o.offs else 0;
    const cur_size = if (obj_cur) |o| o.size else 0;
    const cur_end = cur_offs + cur_size;

    impl.assert(size <= std.math.maxInt(usize) - (@as(usize, c.GGML_MEM_ALIGN) - 1), "size <= SIZE_MAX - (GGML_MEM_ALIGN - 1)");
    const size_needed = impl.pad(size, c.GGML_MEM_ALIGN);

    const mem_buffer: [*]u8 = @ptrCast(ctx.mem_buffer.?);

    if (cur_end > std.math.maxInt(usize) - size_needed) {
        impl.logWarn("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", .{ "ggml_new_object", cur_end, size_needed });
        return null;
    }
    if (cur_end + size_needed > std.math.maxInt(usize) - types.object_size) {
        impl.logWarn("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", .{ "ggml_new_object", cur_end, size_needed, types.object_size });
        return null;
    }

    if (cur_end + size_needed + types.object_size > ctx.mem_size) {
        impl.logWarn("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", .{ "ggml_new_object", cur_end + size_needed + types.object_size, ctx.mem_size });
        if (debug_build) impl.abort("not enough space in the context's memory pool");
        return null;
    }

    const obj_new: *impl.Object = @ptrCast(@alignCast(mem_buffer + cur_end));
    obj_new.* = .{
        .offs = cur_end + types.object_size,
        .size = size_needed,
        .next = null,
        .type = obj_type,
        .padding = @splat(0),
    };

    assertAligned(mem_buffer + obj_new.offs);

    if (obj_cur) |o| {
        o.next = obj_new;
    } else {
        ctx.objects_begin = obj_new;
    }
    ctx.objects_end = obj_new;

    return obj_new;
}

/// Ports `ggml_new_buffer` (ggml.c:1888 @c1d0e7a00).
pub export fn ggml_new_buffer(ctx: *Context, nbytes: usize) ?*anyopaque {
    const obj = newObject(ctx, c.GGML_OBJECT_TYPE_WORK_BUFFER, nbytes) orelse return null;
    const mem_buffer: [*]u8 = @ptrCast(ctx.mem_buffer.?);
    return mem_buffer + obj.offs;
}

// -----------------------------------------------------------------------------
// Tensor creation

/// Ports `ggml_new_tensor_impl` (ggml.c:1765 @c1d0e7a00).
///
/// The single place tensors come from. A view shares its base tensor's data
/// rather than allocating; views of views are collapsed onto the ultimate base
/// so `view_src` is never itself a view.
///
/// Parameters:
/// - `ctx`: context to allocate in.
/// - `t`: element type.
/// - `n_dims`: 1 to `GGML_MAX_DIMS`; trailing dimensions default to 1.
/// - `ne`: extent per dimension, `n_dims` entries.
/// - `view_src`: tensor whose data to share, or null to allocate.
/// - `view_offs`: byte offset into `view_src`'s data.
///
/// Return: the tensor, owned by the context.
pub fn newTensorImpl(
    ctx: *Context,
    t: c.enum_ggml_type,
    n_dims: c_int,
    ne: [*]const i64,
    view_src_in: ?*c.ggml_tensor,
    view_offs_in: usize,
) *c.ggml_tensor {
    impl.assert(t >= 0 and t < c.GGML_TYPE_COUNT, "type >= 0 && type < GGML_TYPE_COUNT");
    impl.assert(n_dims >= 1 and n_dims <= c.GGML_MAX_DIMS, "n_dims >= 1 && n_dims <= GGML_MAX_DIMS");

    var view_src = view_src_in;
    var view_offs = view_offs_in;

    // Collapse a view of a view, so offsets are always relative to real data.
    if (view_src) |vs| {
        if (vs.view_src != null) {
            view_offs += vs.view_offs;
            view_src = impl.one(c.ggml_tensor, vs.view_src);
        }
    }

    var data_size = types.ggml_row_size(t, ne[0]);
    for (1..@intCast(n_dims)) |i| {
        data_size *= @intCast(ne[i]);
    }

    impl.assert(
        view_src == null or data_size == 0 or data_size + view_offs <= types.ggml_nbytes(view_src.?),
        "view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)",
    );

    var data: ?*anyopaque = if (view_src) |vs| vs.data else null;
    if (data) |d| {
        data = @as([*]u8, @ptrCast(d)) + view_offs;
    }

    // Data goes in the same allocation as the struct, unless this is a view or
    // the context has been told not to allocate.
    var obj_alloc_size: usize = 0;
    if (view_src == null and !ctx.no_alloc) obj_alloc_size = data_size;

    const tensor_size = @sizeOf(c.ggml_tensor);
    impl.assert(tensor_size <= std.math.maxInt(usize) - obj_alloc_size, "GGML_TENSOR_SIZE <= SIZE_MAX - obj_alloc_size");

    const obj_new = newObject(ctx, c.GGML_OBJECT_TYPE_TENSOR, tensor_size + obj_alloc_size);
    impl.assert(obj_new != null, "obj_new");

    const mem_buffer: [*]u8 = @ptrCast(ctx.mem_buffer.?);
    const result: *c.ggml_tensor = @ptrCast(@alignCast(mem_buffer + obj_new.?.offs));

    result.* = std.mem.zeroes(c.ggml_tensor);
    result.type = t;
    result.ne = .{ 1, 1, 1, 1 };
    result.op = c.GGML_OP_NONE;
    result.view_src = view_src;
    result.view_offs = view_offs;
    // Inline data begins immediately after the struct: `result + 1` in the C.
    result.data = if (obj_alloc_size > 0)
        @ptrCast(@as([*]c.ggml_tensor, @ptrCast(result)) + 1)
    else
        data;

    for (0..@intCast(n_dims)) |i| {
        result.ne[i] = ne[i];
    }

    // Strides are contiguous by construction. Note nb[1] divides by the block
    // size, so a quantized row is measured in blocks rather than elements.
    result.nb[0] = types.ggml_type_size(t);
    result.nb[1] = result.nb[0] * @as(usize, @intCast(@divTrunc(result.ne[0], types.ggml_blck_size(t))));
    for (2..c.GGML_MAX_DIMS) |i| {
        result.nb[i] = result.nb[i - 1] * @as(usize, @intCast(result.ne[i - 1]));
    }

    ctx.n_objects += 1;

    return result;
}

/// Ports `ggml_new_tensor` (ggml.c:1843 @c1d0e7a00).
pub export fn ggml_new_tensor(ctx: *Context, t: c.enum_ggml_type, n_dims: c_int, ne: [*]const i64) *c.ggml_tensor {
    return newTensorImpl(ctx, t, n_dims, ne, null, 0);
}

/// Ports `ggml_new_tensor_1d` (ggml.c:1851 @c1d0e7a00).
pub export fn ggml_new_tensor_1d(ctx: *Context, t: c.enum_ggml_type, ne0: i64) *c.ggml_tensor {
    var ne = [_]i64{ne0};
    return ggml_new_tensor(ctx, t, 1, &ne);
}

/// Ports `ggml_new_tensor_2d` (ggml.c:1858 @c1d0e7a00).
pub export fn ggml_new_tensor_2d(ctx: *Context, t: c.enum_ggml_type, ne0: i64, ne1: i64) *c.ggml_tensor {
    const ne = [_]i64{ ne0, ne1 };
    return ggml_new_tensor(ctx, t, 2, &ne);
}

/// Ports `ggml_new_tensor_3d` (ggml.c:1867 @c1d0e7a00).
pub export fn ggml_new_tensor_3d(ctx: *Context, t: c.enum_ggml_type, ne0: i64, ne1: i64, ne2: i64) *c.ggml_tensor {
    const ne = [_]i64{ ne0, ne1, ne2 };
    return ggml_new_tensor(ctx, t, 3, &ne);
}

/// Ports `ggml_new_tensor_4d` (ggml.c:1877 @c1d0e7a00).
pub export fn ggml_new_tensor_4d(ctx: *Context, t: c.enum_ggml_type, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *c.ggml_tensor {
    const ne = [_]i64{ ne0, ne1, ne2, ne3 };
    return ggml_new_tensor(ctx, t, 4, &ne);
}

/// Ports `ggml_dup_tensor` (ggml.c:1894 @c1d0e7a00).
///
/// Same type and shape, fresh storage. Not a copy: the data is uninitialised.
pub export fn ggml_dup_tensor(ctx: *Context, src: *const c.ggml_tensor) *c.ggml_tensor {
    return ggml_new_tensor(ctx, src.type, c.GGML_MAX_DIMS, &src.ne);
}

/// Ports `ggml_view_tensor` (ggml.c:1962 @c1d0e7a00).
///
/// Shares `src`'s data and copies its strides, so the view sees the same
/// layout rather than a contiguous reinterpretation.
pub export fn ggml_view_tensor(ctx: *Context, src: *c.ggml_tensor) *c.ggml_tensor {
    const result = newTensorImpl(ctx, src.type, c.GGML_MAX_DIMS, &src.ne, src, 0);
    _ = ggml_format_name(result, "%s (view)", &src.name);
    for (0..c.GGML_MAX_DIMS) |i| {
        result.nb[i] = src.nb[i];
    }
    return result;
}

// -----------------------------------------------------------------------------
// Walking a context

/// Ports `ggml_get_first_tensor` (ggml.c:1975 @c1d0e7a00).
pub export fn ggml_get_first_tensor(ctx: *const Context) ?*c.ggml_tensor {
    const mem_buffer: [*]u8 = @ptrCast(ctx.mem_buffer.?);
    var obj = ctx.objects_begin;
    while (obj) |o| : (obj = o.next) {
        if (o.type == c.GGML_OBJECT_TYPE_TENSOR) {
            return @ptrCast(@alignCast(mem_buffer + o.offs));
        }
    }
    return null;
}

/// Ports `ggml_get_next_tensor` (ggml.c:1991 @c1d0e7a00).
///
/// Steps back from the tensor to its object header, which sits immediately
/// before it, then follows the list. Only valid for a tensor this context
/// allocated.
pub export fn ggml_get_next_tensor(ctx: *const Context, tensor: *c.ggml_tensor) ?*c.ggml_tensor {
    const header: *impl.Object = @ptrCast(@alignCast(@as([*]u8, @ptrCast(tensor)) - types.object_size));
    const mem_buffer: [*]u8 = @ptrCast(ctx.mem_buffer.?);
    var obj = header.next;
    while (obj) |o| : (obj = o.next) {
        if (o.type == c.GGML_OBJECT_TYPE_TENSOR) {
            return @ptrCast(@alignCast(mem_buffer + o.offs));
        }
    }
    return null;
}

/// Ports `ggml_get_tensor` (ggml.c:2008 @c1d0e7a00).
pub export fn ggml_get_tensor(ctx: *Context, name: [*:0]const u8) ?*c.ggml_tensor {
    const mem_buffer: [*]u8 = @ptrCast(ctx.mem_buffer.?);
    var obj = ctx.objects_begin;
    while (obj) |o| : (obj = o.next) {
        if (o.type != c.GGML_OBJECT_TYPE_TENSOR) continue;
        const cur: *c.ggml_tensor = @ptrCast(@alignCast(mem_buffer + o.offs));
        if (std.mem.orderZ(u8, @ptrCast(&cur.name), name) == .eq) return cur;
    }
    return null;
}

// -----------------------------------------------------------------------------
// Tensor accessors

/// Ports `ggml_unravel_index` (ggml.c:1898 @c1d0e7a00).
///
/// Turns a flat element index into per-dimension indices. Any output pointer
/// may be null, which callers use to ask for only the dimensions they need.
pub export fn ggml_unravel_index(
    tensor: *const c.ggml_tensor,
    i: i64,
    out0: ?*i64,
    out1: ?*i64,
    out2: ?*i64,
    out3: ?*i64,
) void {
    const ne0 = tensor.ne[0];
    const ne1 = tensor.ne[1];
    const ne2 = tensor.ne[2];

    const idx3 = @divTrunc(i, ne2 * ne1 * ne0);
    const idx2 = @divTrunc(i - idx3 * ne2 * ne1 * ne0, ne1 * ne0);
    const idx1 = @divTrunc(i - idx3 * ne2 * ne1 * ne0 - idx2 * ne1 * ne0, ne0);
    const idx0 = i - idx3 * ne2 * ne1 * ne0 - idx2 * ne1 * ne0 - idx1 * ne0;

    if (out0) |p| p.* = idx0;
    if (out1) |p| p.* = idx1;
    if (out2) |p| p.* = idx2;
    if (out3) |p| p.* = idx3;
}

/// Ports `ggml_get_data` (ggml.c:1922 @c1d0e7a00).
pub export fn ggml_get_data(tensor: *const c.ggml_tensor) ?*anyopaque {
    return tensor.data;
}

/// Ports `ggml_get_data_f32` (ggml.c:1926 @c1d0e7a00).
pub export fn ggml_get_data_f32(tensor: *const c.ggml_tensor) ?[*]f32 {
    std.debug.assert(tensor.type == c.GGML_TYPE_F32);
    return @ptrCast(@alignCast(tensor.data));
}

/// Ports `ggml_get_name` (ggml.c:1941 @c1d0e7a00).
pub export fn ggml_get_name(tensor: *const c.ggml_tensor) [*:0]const u8 {
    return @ptrCast(&tensor.name);
}

/// Ports `ggml_set_name` (ggml.c:1945 @c1d0e7a00).
///
/// Truncates rather than failing when the name does not fit, and always
/// terminates.
///
/// Return: `tensor`, so calls can be chained onto a constructor.
pub export fn ggml_set_name(tensor: *c.ggml_tensor, name: [*:0]const u8) *c.ggml_tensor {
    var i: usize = 0;
    while (i < tensor.name.len - 1 and name[i] != 0) : (i += 1) {
        tensor.name[i] = @bitCast(name[i]);
    }
    tensor.name[i] = 0;
    return tensor;
}

/// Ports `ggml_format_name` (ggml.c:1954 @c1d0e7a00).
pub export fn ggml_format_name(tensor: *c.ggml_tensor, fmt: [*:0]const u8, ...) callconv(.c) *c.ggml_tensor {
    var ap = @cVaStart();
    defer @cVaEnd(&ap);
    _ = vsnprintf(@ptrCast(&tensor.name), tensor.name.len, fmt, ap);
    return tensor;
}

// -----------------------------------------------------------------------------
// Debug printing

/// Ports `ggml_print_object` (ggml.c:1267 @c1d0e7a00).
export fn ggml_print_object(obj: *const impl.Object) void {
    _ = printf(
        "%s: type = %d, offset = %zu, size = %zu, next = %p\n",
        "ggml_print_object",
        obj.type,
        obj.offs,
        obj.size,
        obj.next,
    );
}

/// Ports `ggml_print_objects` (ggml.c:1272 @c1d0e7a00).
export fn ggml_print_objects(ctx: *const Context) void {
    _ = printf("%s: objects in context %p:\n", "ggml_print_objects", ctx);
    var obj = ctx.objects_begin;
    while (obj) |o| : (obj = o.next) {
        ggml_print_object(o);
    }
    _ = printf("%s: --- end ---\n", "ggml_print_objects");
}

// -----------------------------------------------------------------------------
// Imported from the sibling ported files

extern fn ggml_time_init() void;
extern fn ggml_aligned_malloc(size: usize) ?*anyopaque;
extern fn ggml_aligned_free(ptr: ?*anyopaque, size: usize) void;

// -----------------------------------------------------------------------------
// Unit Tests

extern fn ggml_log_get(callback: *c.ggml_log_callback, user_data: *?*anyopaque) void;
extern fn ggml_log_set(callback: c.ggml_log_callback, user_data: ?*anyopaque) void;

/// Swallows log output, for tests that deliberately trigger a warning.
fn quietLog(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user_data;
}

/// Builds a context backed by a heap buffer this file owns, so tests do not
/// depend on the Mach allocator path.
fn testContext(buf: []u8) Context {
    return .{
        .mem_size = buf.len,
        .mem_buffer = buf.ptr,
        .mem_buffer_owned = false,
        .no_alloc = false,
        .n_objects = 0,
        .objects_begin = null,
        .objects_end = null,
    };
}

test "a fresh context has no objects and no used memory" {
    var buf: [4096]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    try std.testing.expectEqual(@as(usize, 0), ggml_used_mem(&ctx));
    try std.testing.expect(ggml_get_first_tensor(&ctx) == null);
}

test "tensors are laid out contiguously with block-aware strides" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const t = ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    try std.testing.expectEqual(@as(i64, 8), t.ne[0]);
    try std.testing.expectEqual(@as(i64, 4), t.ne[1]);
    try std.testing.expectEqual(@as(i64, 1), t.ne[2]);
    try std.testing.expectEqual(@as(usize, 4), t.nb[0]);
    try std.testing.expectEqual(@as(usize, 32), t.nb[1]);
    try std.testing.expect(types.ggml_is_contiguous(t));
    try std.testing.expectEqual(@as(i64, 32), types.ggml_nelements(t));

    // A quantized row measures nb[1] in blocks, not elements: 256 Q4_K
    // elements are one 144-byte block.
    const q = ggml_new_tensor_2d(&ctx, c.GGML_TYPE_Q4_K, 256, 2);
    try std.testing.expectEqual(@as(usize, 144), q.nb[1]);
}

test "walking a context finds every tensor in order" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 4);
    _ = ggml_set_name(a, "a");
    const b = ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 4);
    _ = ggml_set_name(b, "b");

    try std.testing.expectEqual(a, ggml_get_first_tensor(&ctx).?);
    try std.testing.expectEqual(b, ggml_get_next_tensor(&ctx, a).?);
    try std.testing.expect(ggml_get_next_tensor(&ctx, b) == null);
    try std.testing.expectEqual(b, ggml_get_tensor(&ctx, "b").?);
    try std.testing.expect(ggml_get_tensor(&ctx, "nope") == null);
    try std.testing.expectEqual(@as(c_int, 2), ctx.n_objects);
}

test "a view shares data and strides with its source" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const src = ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    const view = ggml_view_tensor(&ctx, src);

    try std.testing.expectEqual(src.data, view.data);
    try std.testing.expectEqual(@as(?*c.ggml_tensor, src), view.view_src);
    for (0..c.GGML_MAX_DIMS) |i| {
        try std.testing.expectEqual(src.nb[i], view.nb[i]);
    }
    try std.testing.expect(types.ggml_is_view(view));
}

test "a view of a view collapses onto the base tensor" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const base = ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 64);
    const first = newTensorImpl(&ctx, c.GGML_TYPE_F32, 1, &[_]i64{16}, base, 16);
    const second = newTensorImpl(&ctx, c.GGML_TYPE_F32, 1, &[_]i64{8}, first, 8);

    // view_src must be the base, never another view, and the offsets add.
    try std.testing.expectEqual(@as(?*c.ggml_tensor, base), second.view_src);
    try std.testing.expectEqual(@as(usize, 24), second.view_offs);
}

test "names truncate rather than overflow" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const t = ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 1);

    const long = "x" ** 128;
    _ = ggml_set_name(t, long);
    try std.testing.expectEqual(@as(u8, 0), @as(u8, @bitCast(t.name[t.name.len - 1])));
    try std.testing.expectEqual(t.name.len - 1, std.mem.len(@as([*:0]const u8, @ptrCast(&t.name))));
}

test "unravel index inverts the flat layout" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const t = ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 2, 3, 4, 5);

    var d0: i64 = 0;
    var d1: i64 = 0;
    var d2: i64 = 0;
    var d3: i64 = 0;
    // Element 1 + 2*2 + 3*6 + 4*24 = 119.
    ggml_unravel_index(t, 119, &d0, &d1, &d2, &d3);
    try std.testing.expectEqual(@as(i64, 1), d0);
    try std.testing.expectEqual(@as(i64, 2), d1);
    try std.testing.expectEqual(@as(i64, 3), d2);
    try std.testing.expectEqual(@as(i64, 4), d3);

    // Null outputs are allowed and must not be written.
    ggml_unravel_index(t, 0, null, null, null, null);
}

test "an exhausted pool returns null rather than corrupting the context" {
    // In a debug build this path aborts instead of returning, matching the C,
    // so only the release behaviour can be asserted.
    if (debug_build) return error.SkipZigTest;

    // This deliberately exhausts the pool, and ggml logs when that happens.
    // Silenced so a passing run does not print what looks like a failure.
    var saved: c.ggml_log_callback = undefined;
    var saved_data: ?*anyopaque = undefined;
    ggml_log_get(&saved, &saved_data);
    ggml_log_set(quietLog, null);
    defer ggml_log_set(saved, saved_data);

    // Room for one tensor but not for the oversized request that follows. A
    // ggml_tensor is 336 bytes, so a buffer under that would fail on the first
    // allocation and test nothing.
    var buf: [1024]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const first = ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 1);
    try std.testing.expectEqual(@as(c_int, 1), ctx.n_objects);

    try std.testing.expect(newObject(&ctx, c.GGML_OBJECT_TYPE_TENSOR, 4096) == null);

    // A failed allocation must leave the list intact, not half-linked.
    try std.testing.expectEqual(@as(c_int, 1), ctx.n_objects);
    try std.testing.expectEqual(first, ggml_get_first_tensor(&ctx).?);
    try std.testing.expect(ggml_get_next_tensor(&ctx, first) == null);
}
