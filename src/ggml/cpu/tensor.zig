//! Reading and writing single tensor elements from the host.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! the section beginning at line 748. Each function names the C it replaces
//! and the line it began at.
//!
//! # Why these live in the CPU backend
//!
//! They dereference `tensor->data`, so they only work for a tensor whose
//! buffer is host memory. That is why `ggml.c` does not have them and
//! `ggml-cpu.c` does. Nothing on the inference path calls them -- the graph
//! does its reading through the ops -- but `test-backend-ops` and the
//! reference drivers do, which is what makes them worth getting exactly right.
//!
//! # Element indices are 64-bit here
//!
//! The C narrows `ggml_nrows` and `ne[0]` to `int` before looping, then passes
//! them to `ggml_vec_set_*`, which also takes `int`. This port keeps them
//! `i64`. The two differ only for a tensor with more than 2^31 rows or
//! columns, where the C already overflows and produces nonsense; Zig would
//! trap on the narrowing instead, which is a worse way to say the same thing.

const std = @import("std");
const impl = @import("../impl.zig");
const types = @import("../types.zig");
const context = @import("../context.zig");
const convert = @import("convert.zig");
const c = impl.c;

const Tensor = c.ggml_tensor;
const Context = context.Context;

/// A byte pointer into a tensor's data, with no alignment promise.
///
/// The C reaches elements as `((float *)(tensor->data))[i]` and as
/// `(char *) tensor->data + i0*nb0 + ...`. The second form can land anywhere,
/// so every access here goes through `align(1)`: unaligned loads cost nothing
/// on AArch64 and asserting an alignment the C never guarantees would turn a
/// working read into a panic.
fn bytes(tensor: *const Tensor) [*]u8 {
    return @ptrCast(tensor.data.?);
}

/// The element at a byte offset, read as `T` without an alignment assumption.
inline fn load(comptime T: type, p: [*]const u8) T {
    return @as(*align(1) const T, @ptrCast(p)).*;
}

/// Writes `v` at a byte offset as `T`, without an alignment assumption.
inline fn store(comptime T: type, p: [*]u8, v: T) void {
    @as(*align(1) T, @ptrCast(p)).* = v;
}

/// Ports the `ggml_vec_set_i8` family (vec.h:80 @c1d0e7a00), all six at once.
///
/// The C has one `inline static` per element type, each a scalar loop. Zig's
/// `@memset` covers the lot.
inline fn vecSet(comptime T: type, n: i64, p: [*]u8, v: T) void {
    const dst: [*]align(1) T = @ptrCast(p);
    @memset(dst[0..@intCast(n)], v);
}

/// The byte offset of element `(d0, d1, d2, d3)` — the C calls them `i0`..`i3`,
/// which Zig reserves as integer type names. The `nd` accessors' shared first line.
/// first line.
inline fn offsetOf(tensor: *const Tensor, d0: c_int, d1: c_int, d2: c_int, d3: c_int) usize {
    return @as(usize, @intCast(d0)) * tensor.nb[0] +
        @as(usize, @intCast(d1)) * tensor.nb[1] +
        @as(usize, @intCast(d2)) * tensor.nb[2] +
        @as(usize, @intCast(d3)) * tensor.nb[3];
}

// -----------------------------------------------------------------------------
// Constructors

/// Ports `ggml_new_i32` (ggml-cpu.c:748 @c1d0e7a00).
///
/// Parameters:
/// - `ctx`: context to allocate in; must not be in no-alloc mode, since the
///   value is written immediately.
/// - `value`: the scalar to store.
///
/// Return: a 1-element I32 tensor owned by `ctx`.
pub export fn ggml_new_i32(ctx: *Context, value: i32) *Tensor {
    impl.assert(!context.ggml_get_no_alloc(ctx), "!ggml_get_no_alloc(ctx)");

    const result = context.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, 1);
    _ = ggml_set_i32(result, value);
    return result;
}

/// Ports `ggml_new_f32` (ggml-cpu.c:758 @c1d0e7a00).
///
/// Parameters:
/// - `ctx`: context to allocate in; must not be in no-alloc mode.
/// - `value`: the scalar to store.
///
/// Return: a 1-element F32 tensor owned by `ctx`.
pub export fn ggml_new_f32(ctx: *Context, value: f32) *Tensor {
    impl.assert(!context.ggml_get_no_alloc(ctx), "!ggml_get_no_alloc(ctx)");

    const result = context.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 1);
    _ = ggml_set_f32(result, value);
    return result;
}

// -----------------------------------------------------------------------------
// Filling a whole tensor

/// Ports `ggml_set_i32` (ggml-cpu.c:768 @c1d0e7a00).
///
/// Fills every element with `value`, row by row, so a tensor with row padding
/// keeps its padding untouched.
///
/// Parameters:
/// - `tensor`: the tensor to fill; must be one of the six scalar types.
/// - `value`: the value, converted to the tensor's type.
///
/// Return: `tensor`, for chaining, as the C does.
pub export fn ggml_set_i32(tensor: *Tensor, value: i32) *Tensor {
    const n = types.ggml_nrows(tensor);
    const nc = tensor.ne[0];
    const n1 = tensor.nb[1];
    const data = bytes(tensor);

    switch (tensor.type) {
        c.GGML_TYPE_I8 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(i8));
            for (0..@intCast(n)) |i| vecSet(i8, nc, data + i * n1, @truncate(value));
        },
        c.GGML_TYPE_I16 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(i16));
            for (0..@intCast(n)) |i| vecSet(i16, nc, data + i * n1, @truncate(value));
        },
        c.GGML_TYPE_I32 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(i32));
            for (0..@intCast(n)) |i| vecSet(i32, nc, data + i * n1, value);
        },
        c.GGML_TYPE_F16 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(c.ggml_fp16_t));
            const v = convert.cpuFp32ToFp16(@floatFromInt(value));
            for (0..@intCast(n)) |i| vecSet(c.ggml_fp16_t, nc, data + i * n1, v);
        },
        c.GGML_TYPE_BF16 => {
            // The C asserts `sizeof(ggml_fp16_t)` here rather than
            // `sizeof(ggml_bf16_t)`. They are the same two bytes, so the
            // assertion still holds; kept as the C wrote it.
            std.debug.assert(tensor.nb[0] == @sizeOf(c.ggml_fp16_t));
            const v = c.ggml_bf16_t{ .bits = impl.fp32ToBf16(@floatFromInt(value)) };
            for (0..@intCast(n)) |i| vecSet(c.ggml_bf16_t, nc, data + i * n1, v);
        },
        c.GGML_TYPE_F32 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(f32));
            const v: f32 = @floatFromInt(value);
            for (0..@intCast(n)) |i| vecSet(f32, nc, data + i * n1, v);
        },
        else => impl.abort("fatal error"),
    }

    return tensor;
}

/// Ports `ggml_set_f32` (ggml-cpu.c:827 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to fill; must be one of the six scalar types.
/// - `value`: the value, converted to the tensor's type.
///
/// Return: `tensor`, for chaining.
pub export fn ggml_set_f32(tensor: *Tensor, value: f32) *Tensor {
    const n = types.ggml_nrows(tensor);
    const nc = tensor.ne[0];
    const n1 = tensor.nb[1];
    const data = bytes(tensor);

    switch (tensor.type) {
        c.GGML_TYPE_I8 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(i8));
            const v = impl.truncTo(i8, value);
            for (0..@intCast(n)) |i| vecSet(i8, nc, data + i * n1, v);
        },
        c.GGML_TYPE_I16 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(i16));
            const v = impl.truncTo(i16, value);
            for (0..@intCast(n)) |i| vecSet(i16, nc, data + i * n1, v);
        },
        c.GGML_TYPE_I32 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(i32));
            const v = impl.truncTo(i32, value);
            for (0..@intCast(n)) |i| vecSet(i32, nc, data + i * n1, v);
        },
        c.GGML_TYPE_F16 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(c.ggml_fp16_t));
            const v = convert.cpuFp32ToFp16(value);
            for (0..@intCast(n)) |i| vecSet(c.ggml_fp16_t, nc, data + i * n1, v);
        },
        c.GGML_TYPE_BF16 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(c.ggml_bf16_t));
            const v = c.ggml_bf16_t{ .bits = impl.fp32ToBf16(value) };
            for (0..@intCast(n)) |i| vecSet(c.ggml_bf16_t, nc, data + i * n1, v);
        },
        c.GGML_TYPE_F32 => {
            std.debug.assert(tensor.nb[0] == @sizeOf(f32));
            for (0..@intCast(n)) |i| vecSet(f32, nc, data + i * n1, value);
        },
        else => impl.abort("fatal error"),
    }

    return tensor;
}

// -----------------------------------------------------------------------------
// Single elements, by flat index
//
// A flat index only means anything for a contiguous tensor, so each of these
// opens by unravelling the index and handing off to the `nd` form when it is
// not.

/// Ports `ggml_get_i32_1d` (ggml-cpu.c:886 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to read.
/// - `i`: flat element index.
///
/// Return: the element, converted to `i32` by truncation toward zero.
pub export fn ggml_get_i32_1d(tensor: *const Tensor, i: c_int) i32 {
    if (!types.ggml_is_contiguous(tensor)) {
        var id: [4]i64 = @splat(0);
        context.ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, @intCast(id[0]), @intCast(id[1]), @intCast(id[2]), @intCast(id[3]));
    }

    const data = bytes(tensor);
    const idx: usize = @intCast(i);

    return switch (tensor.type) {
        c.GGML_TYPE_I8 => blk: {
            impl.assert(tensor.nb[0] == @sizeOf(i8), "tensor->nb[0] == sizeof(int8_t)");
            break :blk load(i8, data + idx * @sizeOf(i8));
        },
        c.GGML_TYPE_I16 => blk: {
            impl.assert(tensor.nb[0] == @sizeOf(i16), "tensor->nb[0] == sizeof(int16_t)");
            break :blk load(i16, data + idx * @sizeOf(i16));
        },
        c.GGML_TYPE_I32 => blk: {
            impl.assert(tensor.nb[0] == @sizeOf(i32), "tensor->nb[0] == sizeof(int32_t)");
            break :blk load(i32, data + idx * @sizeOf(i32));
        },
        c.GGML_TYPE_F16 => blk: {
            impl.assert(tensor.nb[0] == @sizeOf(c.ggml_fp16_t), "tensor->nb[0] == sizeof(ggml_fp16_t)");
            break :blk impl.truncTo(i32, convert.cpuFp16ToFp32(load(c.ggml_fp16_t, data + idx * 2)));
        },
        c.GGML_TYPE_BF16 => blk: {
            impl.assert(tensor.nb[0] == @sizeOf(c.ggml_bf16_t), "tensor->nb[0] == sizeof(ggml_bf16_t)");
            break :blk impl.truncTo(i32, impl.bf16ToFp32(load(c.ggml_bf16_t, data + idx * 2).bits));
        },
        c.GGML_TYPE_F32 => blk: {
            impl.assert(tensor.nb[0] == @sizeOf(f32), "tensor->nb[0] == sizeof(float)");
            break :blk impl.truncTo(i32, load(f32, data + idx * @sizeOf(f32)));
        },
        else => impl.abort("fatal error"),
    };
}

/// Ports `ggml_set_i32_1d` (ggml-cpu.c:930 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to write.
/// - `i`: flat element index.
/// - `value`: the value, converted to the tensor's type.
pub export fn ggml_set_i32_1d(tensor: *const Tensor, i: c_int, value: i32) void {
    if (!types.ggml_is_contiguous(tensor)) {
        var id: [4]i64 = @splat(0);
        context.ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, @intCast(id[0]), @intCast(id[1]), @intCast(id[2]), @intCast(id[3]), value);
        return;
    }

    const data = bytes(tensor);
    const idx: usize = @intCast(i);

    switch (tensor.type) {
        c.GGML_TYPE_I8 => {
            impl.assert(tensor.nb[0] == @sizeOf(i8), "tensor->nb[0] == sizeof(int8_t)");
            store(i8, data + idx * @sizeOf(i8), @truncate(value));
        },
        c.GGML_TYPE_I16 => {
            impl.assert(tensor.nb[0] == @sizeOf(i16), "tensor->nb[0] == sizeof(int16_t)");
            store(i16, data + idx * @sizeOf(i16), @truncate(value));
        },
        c.GGML_TYPE_I32 => {
            impl.assert(tensor.nb[0] == @sizeOf(i32), "tensor->nb[0] == sizeof(int32_t)");
            store(i32, data + idx * @sizeOf(i32), value);
        },
        c.GGML_TYPE_F16 => {
            impl.assert(tensor.nb[0] == @sizeOf(c.ggml_fp16_t), "tensor->nb[0] == sizeof(ggml_fp16_t)");
            store(c.ggml_fp16_t, data + idx * 2, convert.cpuFp32ToFp16(@floatFromInt(value)));
        },
        c.GGML_TYPE_BF16 => {
            impl.assert(tensor.nb[0] == @sizeOf(c.ggml_bf16_t), "tensor->nb[0] == sizeof(ggml_bf16_t)");
            store(c.ggml_bf16_t, data + idx * 2, .{ .bits = impl.fp32ToBf16(@floatFromInt(value)) });
        },
        c.GGML_TYPE_F32 => {
            impl.assert(tensor.nb[0] == @sizeOf(f32), "tensor->nb[0] == sizeof(float)");
            store(f32, data + idx * @sizeOf(f32), @floatFromInt(value));
        },
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_get_f32_1d` (ggml-cpu.c:1029 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to read.
/// - `i`: flat element index.
///
/// Return: the element, widened to `f32`.
pub export fn ggml_get_f32_1d(tensor: *const Tensor, i: c_int) f32 {
    if (!types.ggml_is_contiguous(tensor)) {
        var id: [4]i64 = @splat(0);
        context.ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, @intCast(id[0]), @intCast(id[1]), @intCast(id[2]), @intCast(id[3]));
    }

    const data = bytes(tensor);
    const idx: usize = @intCast(i);

    // Unlike the `i32` reader, the C asserts nothing here.
    return switch (tensor.type) {
        c.GGML_TYPE_I8 => @floatFromInt(load(i8, data + idx * @sizeOf(i8))),
        c.GGML_TYPE_I16 => @floatFromInt(load(i16, data + idx * @sizeOf(i16))),
        c.GGML_TYPE_I32 => @floatFromInt(load(i32, data + idx * @sizeOf(i32))),
        c.GGML_TYPE_F16 => convert.cpuFp16ToFp32(load(c.ggml_fp16_t, data + idx * 2)),
        c.GGML_TYPE_BF16 => impl.bf16ToFp32(load(c.ggml_bf16_t, data + idx * 2).bits),
        c.GGML_TYPE_F32 => load(f32, data + idx * @sizeOf(f32)),
        else => impl.abort("fatal error"),
    };
}

/// Ports `ggml_set_f32_1d` (ggml-cpu.c:1067 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to write.
/// - `i`: flat element index.
/// - `value`: the value, converted to the tensor's type.
pub export fn ggml_set_f32_1d(tensor: *const Tensor, i: c_int, value: f32) void {
    if (!types.ggml_is_contiguous(tensor)) {
        var id: [4]i64 = @splat(0);
        context.ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, @intCast(id[0]), @intCast(id[1]), @intCast(id[2]), @intCast(id[3]), value);
        return;
    }

    const data = bytes(tensor);
    const idx: usize = @intCast(i);

    switch (tensor.type) {
        c.GGML_TYPE_I8 => store(i8, data + idx * @sizeOf(i8), impl.truncTo(i8, value)),
        c.GGML_TYPE_I16 => store(i16, data + idx * @sizeOf(i16), impl.truncTo(i16, value)),
        c.GGML_TYPE_I32 => store(i32, data + idx * @sizeOf(i32), impl.truncTo(i32, value)),
        c.GGML_TYPE_F16 => store(c.ggml_fp16_t, data + idx * 2, convert.cpuFp32ToFp16(value)),
        c.GGML_TYPE_BF16 => store(c.ggml_bf16_t, data + idx * 2, .{ .bits = impl.fp32ToBf16(value) }),
        c.GGML_TYPE_F32 => store(f32, data + idx * @sizeOf(f32), value),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Single elements, by coordinate

/// Ports `ggml_get_i32_nd` (ggml-cpu.c:975 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to read.
/// - `d0`..`d3`: coordinates along each dimension.
///
/// Return: the element, converted to `i32` by truncation toward zero.
pub export fn ggml_get_i32_nd(tensor: *const Tensor, d0: c_int, d1: c_int, d2: c_int, d3: c_int) i32 {
    const p = bytes(tensor) + offsetOf(tensor, d0, d1, d2, d3);

    return switch (tensor.type) {
        c.GGML_TYPE_I8 => load(i8, p),
        c.GGML_TYPE_I16 => load(i16, p),
        c.GGML_TYPE_I32 => load(i32, p),
        c.GGML_TYPE_F16 => impl.truncTo(i32, convert.cpuFp16ToFp32(load(c.ggml_fp16_t, p))),
        c.GGML_TYPE_BF16 => impl.truncTo(i32, impl.bf16ToFp32(load(c.ggml_bf16_t, p).bits)),
        c.GGML_TYPE_F32 => impl.truncTo(i32, load(f32, p)),
        else => impl.abort("fatal error"),
    };
}

/// Ports `ggml_set_i32_nd` (ggml-cpu.c:995 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to write.
/// - `d0`..`d3`: coordinates along each dimension.
/// - `value`: the value, converted to the tensor's type.
pub export fn ggml_set_i32_nd(tensor: *const Tensor, d0: c_int, d1: c_int, d2: c_int, d3: c_int, value: i32) void {
    const p = bytes(tensor) + offsetOf(tensor, d0, d1, d2, d3);

    switch (tensor.type) {
        c.GGML_TYPE_I8 => store(i8, p, @truncate(value)),
        c.GGML_TYPE_I16 => store(i16, p, @truncate(value)),
        c.GGML_TYPE_I32 => store(i32, p, value),
        c.GGML_TYPE_F16 => store(c.ggml_fp16_t, p, convert.cpuFp32ToFp16(@floatFromInt(value))),
        c.GGML_TYPE_BF16 => store(c.ggml_bf16_t, p, .{ .bits = impl.fp32ToBf16(@floatFromInt(value)) }),
        c.GGML_TYPE_F32 => store(f32, p, @floatFromInt(value)),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_get_f32_nd` (ggml-cpu.c:1106 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to read.
/// - `d0`..`d3`: coordinates along each dimension.
///
/// Return: the element, widened to `f32`.
pub export fn ggml_get_f32_nd(tensor: *const Tensor, d0: c_int, d1: c_int, d2: c_int, d3: c_int) f32 {
    const p = bytes(tensor) + offsetOf(tensor, d0, d1, d2, d3);

    return switch (tensor.type) {
        c.GGML_TYPE_I8 => @floatFromInt(load(i8, p)),
        c.GGML_TYPE_I16 => @floatFromInt(load(i16, p)),
        c.GGML_TYPE_I32 => @floatFromInt(load(i32, p)),
        c.GGML_TYPE_F16 => convert.cpuFp16ToFp32(load(c.ggml_fp16_t, p)),
        c.GGML_TYPE_BF16 => impl.bf16ToFp32(load(c.ggml_bf16_t, p).bits),
        c.GGML_TYPE_F32 => load(f32, p),
        else => impl.abort("fatal error"),
    };
}

/// Ports `ggml_set_f32_nd` (ggml-cpu.c:1126 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: the tensor to write.
/// - `d0`..`d3`: coordinates along each dimension.
/// - `value`: the value, converted to the tensor's type.
pub export fn ggml_set_f32_nd(tensor: *const Tensor, d0: c_int, d1: c_int, d2: c_int, d3: c_int, value: f32) void {
    const p = bytes(tensor) + offsetOf(tensor, d0, d1, d2, d3);

    switch (tensor.type) {
        c.GGML_TYPE_I8 => store(i8, p, impl.truncTo(i8, value)),
        c.GGML_TYPE_I16 => store(i16, p, impl.truncTo(i16, value)),
        c.GGML_TYPE_I32 => store(i32, p, impl.truncTo(i32, value)),
        c.GGML_TYPE_F16 => store(c.ggml_fp16_t, p, convert.cpuFp32ToFp16(value)),
        c.GGML_TYPE_BF16 => store(c.ggml_bf16_t, p, .{ .bits = impl.fp32ToBf16(value) }),
        c.GGML_TYPE_F32 => store(f32, p, value),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

/// A context big enough for the small tensors below, with its own buffer.
fn testContext() *Context {
    return context.ggml_init(.{
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = null,
        .no_alloc = false,
    }).?;
}

test "new_i32 and new_f32 round trip a scalar" {
    const ctx = testContext();
    defer context.ggml_free(ctx);

    const a = ggml_new_i32(ctx, -7);
    try std.testing.expectEqual(@as(i32, -7), ggml_get_i32_1d(a, 0));

    const b = ggml_new_f32(ctx, 2.5);
    try std.testing.expectEqual(@as(f32, 2.5), ggml_get_f32_1d(b, 0));
}

test "set_f32 fills every element of every scalar type" {
    const ctx = testContext();
    defer context.ggml_free(ctx);

    const cases = [_]c.enum_ggml_type{
        c.GGML_TYPE_I8,  c.GGML_TYPE_I16,  c.GGML_TYPE_I32,
        c.GGML_TYPE_F16, c.GGML_TYPE_BF16, c.GGML_TYPE_F32,
    };

    for (cases) |t| {
        const a = context.ggml_new_tensor_1d(ctx, t, 8);
        _ = ggml_set_f32(a, 3.0);
        for (0..8) |i| {
            try std.testing.expectEqual(@as(f32, 3.0), ggml_get_f32_1d(a, @intCast(i)));
        }
    }
}

test "set_i32 fills every element of every scalar type" {
    const ctx = testContext();
    defer context.ggml_free(ctx);

    const cases = [_]c.enum_ggml_type{
        c.GGML_TYPE_I8,  c.GGML_TYPE_I16,  c.GGML_TYPE_I32,
        c.GGML_TYPE_F16, c.GGML_TYPE_BF16, c.GGML_TYPE_F32,
    };

    for (cases) |t| {
        const a = context.ggml_new_tensor_1d(ctx, t, 8);
        _ = ggml_set_i32(a, -5);
        for (0..8) |i| {
            try std.testing.expectEqual(@as(i32, -5), ggml_get_i32_1d(a, @intCast(i)));
        }
    }
}

test "the nd accessors walk the strides" {
    const ctx = testContext();
    defer context.ggml_free(ctx);

    const a = context.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 3, 2);
    _ = ggml_set_f32(a, 0.0);

    ggml_set_f32_nd(a, 2, 1, 0, 0, 9.0);
    try std.testing.expectEqual(@as(f32, 9.0), ggml_get_f32_nd(a, 2, 1, 0, 0));

    // Flat index 5 is the same element for a contiguous 3x2.
    try std.testing.expectEqual(@as(f32, 9.0), ggml_get_f32_1d(a, 5));
    try std.testing.expectEqual(@as(f32, 0.0), ggml_get_f32_1d(a, 4));
}

test "a non-contiguous view is read through unravel_index" {
    const ctx = testContext();
    defer context.ggml_free(ctx);

    const a = context.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 4, 2);
    _ = ggml_set_f32(a, 0.0);
    for (0..8) |i| ggml_set_f32_1d(a, @intCast(i), @floatFromInt(i));

    // Two columns of the four, so rows are strided.
    const v = ctx_view: {
        const t = @import("../ops.zig");
        break :ctx_view t.ggml_view_2d(ctx, a, 2, 2, a.nb[1], 0);
    };
    try std.testing.expect(!types.ggml_is_contiguous(v));

    // Flat 2 is (0,1) in the view, which is element 4 of the source.
    try std.testing.expectEqual(@as(f32, 4.0), ggml_get_f32_1d(v, 2));
    try std.testing.expectEqual(@as(f32, 1.0), ggml_get_f32_1d(v, 1));
}

test "truncation toward zero, not rounding" {
    const ctx = testContext();
    defer context.ggml_free(ctx);

    const a = context.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 2);
    ggml_set_f32_1d(a, 0, 1.9);
    ggml_set_f32_1d(a, 1, -1.9);

    try std.testing.expectEqual(@as(i32, 1), ggml_get_i32_1d(a, 0));
    try std.testing.expectEqual(@as(i32, -1), ggml_get_i32_1d(a, 1));
}
