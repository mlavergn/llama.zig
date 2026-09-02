//! Graph-building operations: the functions that add a node to a graph.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml.c` (v0.3.0, `c1d0e7a00`), the section
//! beginning at line 2027. Each function names the C function it replaces and
//! the line it began at.
//!
//! # These do not compute anything
//!
//! Every function here allocates a result tensor, records an op and its
//! sources, and returns. No arithmetic happens: the backends do that later,
//! when the graph is evaluated. So what matters is the shape of the result, the
//! contents of `op_params`, and the assertions -- those assertions are ggml's
//! only type checking, and a wrong one shows up as a wrong answer rather than
//! a crash.
//!
//! # Shared shapes
//!
//! The C repeats a handful of patterns across two hundred functions, each
//! spelled out longhand with an `_impl` helper and thin `x`/`x_inplace`
//! wrappers. Those patterns are factored into `dupOrView`, `unaryOp`, and
//! `binaryOp` here. The wrappers remain, because they are the C ABI.

const std = @import("std");
const impl = @import("impl.zig");
const types = @import("types.zig");
const context = @import("context.zig");
const c = impl.c;

const Context = context.Context;
const Tensor = c.ggml_tensor;

// -----------------------------------------------------------------------------
// Shared shapes

/// The `inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a)` that
/// opens almost every op in the C.
///
/// An in-place op writes over its input, so the result is a view of it rather
/// than fresh storage. Only ops listed in `ggml_op_can_inplace` may do this.
inline fn dupOrView(ctx: *Context, a: *Tensor, inplace: bool) *Tensor {
    return if (inplace)
        context.ggml_view_tensor(ctx, a)
    else
        context.ggml_dup_tensor(ctx, a);
}

/// One input, same shape out.
inline fn unaryOp(ctx: *Context, a: *Tensor, op: c.enum_ggml_op, inplace: bool) *Tensor {
    const result = dupOrView(ctx, a, inplace);
    result.op = op;
    result.src[0] = a;
    return result;
}

/// Two inputs, shape of the first out, with `b` broadcastable over `a`.
inline fn binaryOp(ctx: *Context, a: *Tensor, b: *Tensor, op: c.enum_ggml_op, inplace: bool) *Tensor {
    impl.assert(types.ggml_can_repeat(b, a), "ggml_can_repeat(b, a)");
    const result = dupOrView(ctx, a, inplace);
    result.op = op;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

// -----------------------------------------------------------------------------
// ggml_dup

/// Ports `ggml_dup` (ggml.c:2043 @c1d0e7a00).
pub export fn ggml_dup(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_DUP, false);
}

/// Ports `ggml_dup_inplace` (ggml.c:2049 @c1d0e7a00).
pub export fn ggml_dup_inplace(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_DUP, true);
}

// -----------------------------------------------------------------------------
// ggml_add

/// Ports `ggml_add` (ggml.c:2073 @c1d0e7a00).
pub export fn ggml_add(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_ADD, false);
}

/// Ports `ggml_add_inplace` (ggml.c:2080 @c1d0e7a00).
pub export fn ggml_add_inplace(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_ADD, true);
}

/// Ports `ggml_add_cast` (ggml.c:2112 @c1d0e7a00).
///
/// Adds and changes type in one node, so a quantized weight can accumulate
/// into a wider gradient without a separate cast. Only quantized, f16, and
/// bf16 inputs are supported, which is what the backends implement.
pub export fn ggml_add_cast(ctx: *Context, a: *Tensor, b: *Tensor, t: c.enum_ggml_type) *Tensor {
    // The C notes this constraint is stricter than it needs to be.
    impl.assert(types.canRepeatRows(b, a), "ggml_can_repeat_rows(b, a)");
    impl.assert(
        types.ggml_is_quantized(a.type) or a.type == c.GGML_TYPE_F16 or a.type == c.GGML_TYPE_BF16,
        "ggml_is_quantized(a->type) || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16",
    );

    const result = context.ggml_new_tensor(ctx, t, c.GGML_MAX_DIMS, &a.ne);
    result.op = c.GGML_OP_ADD;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_add_id` (ggml.c:2120 @c1d0e7a00).
///
/// Adds a row of `b` selected per position by `ids`, which is how mixture-of-
/// experts biases are applied without materialising the full bias tensor.
pub export fn ggml_add_id(ctx: *Context, a: *Tensor, b: *Tensor, ids: *Tensor) *Tensor {
    impl.assert(a.ne[0] == b.ne[0], "a->ne[0] == b->ne[0]");
    impl.assert(a.ne[1] == ids.ne[0], "a->ne[1] == ids->ne[0]");
    impl.assert(a.ne[2] == ids.ne[1], "a->ne[2] == ids->ne[1]");
    impl.assert(ids.type == c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    const result = context.ggml_dup_tensor(ctx, a);
    result.op = c.GGML_OP_ADD_ID;
    result.src[0] = a;
    result.src[1] = b;
    result.src[2] = ids;
    return result;
}

// -----------------------------------------------------------------------------
// ggml_add1

/// Ports `ggml_add1_impl` (ggml.c:2143 @c1d0e7a00).
pub fn add1Impl(ctx: *Context, a: *Tensor, b: *Tensor, inplace: bool) *Tensor {
    impl.assert(types.ggml_is_scalar(b), "ggml_is_scalar(b)");
    impl.assert(types.isPadded1d(a), "ggml_is_padded_1d(a)");

    const result = dupOrView(ctx, a, inplace);
    result.op = c.GGML_OP_ADD1;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_add1` (ggml.c:2160 @c1d0e7a00).
pub export fn ggml_add1(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return add1Impl(ctx, a, b, false);
}

/// Ports `ggml_add1_inplace` (ggml.c:2167 @c1d0e7a00).
pub export fn ggml_add1_inplace(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return add1Impl(ctx, a, b, true);
}

// -----------------------------------------------------------------------------
// ggml_acc

/// Ports `ggml_acc_impl` (ggml.c:2176 @c1d0e7a00).
///
/// Accumulates `b` into a strided window of `a`. The strides and offset go
/// into `op_params` as i32, which is what the C writes even though the
/// arguments are `size_t`.
pub fn accImpl(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
    inplace: bool,
) *Tensor {
    impl.assert(types.ggml_nelements(b) <= types.ggml_nelements(a), "ggml_nelements(b) <= ggml_nelements(a)");
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
    impl.assert(a.type == c.GGML_TYPE_F32, "a->type == GGML_TYPE_F32");
    impl.assert(b.type == c.GGML_TYPE_F32, "b->type == GGML_TYPE_F32");

    const result = dupOrView(ctx, a, inplace);

    const params = [_]i32{
        @intCast(nb1),         @intCast(nb2), @intCast(nb3), @intCast(offset),
        if (inplace) 1 else 0,
    };
    impl.setOpParamsValue(result, params);

    result.op = c.GGML_OP_ACC;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_acc` (ggml.c:2202 @c1d0e7a00).
pub export fn ggml_acc(ctx: *Context, a: *Tensor, b: *Tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize) *Tensor {
    return accImpl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

/// Ports `ggml_acc_inplace` (ggml.c:2213 @c1d0e7a00).
pub export fn ggml_acc_inplace(ctx: *Context, a: *Tensor, b: *Tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize) *Tensor {
    return accImpl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// -----------------------------------------------------------------------------
// Elementwise binaries

/// Ports `ggml_add_impl` (ggml.c:2057 @c1d0e7a00) and `ggml_sub_impl` (ggml.c:2226 @c1d0e7a00).
///
/// The C keeps these as separate static helpers; both are `binaryOp` with a
/// different op, and the wrappers above already go through it. They exist
/// under their own names because the autodiff pass in `graph.zig` calls them
/// with a computed `inplace`, which no exported wrapper offers.
pub fn addImpl(ctx: *Context, a: *Tensor, b: *Tensor, inplace: bool) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_ADD, inplace);
}

/// Ports `ggml_sub_impl` (ggml.c:2226 @c1d0e7a00). See `addImpl`.
pub fn subImpl(ctx: *Context, a: *Tensor, b: *Tensor, inplace: bool) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_SUB, inplace);
}

/// Ports `ggml_sub` (ggml.c:2242 @c1d0e7a00).
pub export fn ggml_sub(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_SUB, false);
}

/// Ports `ggml_sub_inplace` (ggml.c:2249 @c1d0e7a00).
pub export fn ggml_sub_inplace(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_SUB, true);
}

/// Ports `ggml_mul` (ggml.c:2274 @c1d0e7a00).
pub export fn ggml_mul(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_MUL, false);
}

/// Ports `ggml_mul_inplace` (ggml.c:2281 @c1d0e7a00).
pub export fn ggml_mul_inplace(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_MUL, true);
}

/// Ports `ggml_div` (ggml.c:2306 @c1d0e7a00).
pub export fn ggml_div(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_DIV, false);
}

/// Ports `ggml_div_inplace` (ggml.c:2313 @c1d0e7a00).
pub export fn ggml_div_inplace(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return binaryOp(ctx, a, b, c.GGML_OP_DIV, true);
}

// -----------------------------------------------------------------------------
// Elementwise unaries with their own op codes
//
// These predate `GGML_OP_UNARY` and keep dedicated op codes, unlike the
// activations further down which share one.

/// Ports `ggml_sqr` (ggml.c:2334 @c1d0e7a00).
pub export fn ggml_sqr(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_SQR, false);
}

/// Ports `ggml_sqr_inplace` (ggml.c:2340 @c1d0e7a00).
pub export fn ggml_sqr_inplace(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_SQR, true);
}

/// Ports `ggml_sqrt` (ggml.c:2360 @c1d0e7a00).
pub export fn ggml_sqrt(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_SQRT, false);
}

/// Ports `ggml_sqrt_inplace` (ggml.c:2366 @c1d0e7a00).
pub export fn ggml_sqrt_inplace(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_SQRT, true);
}

/// Ports `ggml_log` (ggml.c:2386 @c1d0e7a00).
pub export fn ggml_log(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_LOG, false);
}

/// Ports `ggml_log_inplace` (ggml.c:2392 @c1d0e7a00).
pub export fn ggml_log_inplace(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_LOG, true);
}

/// Ports `ggml_sin` (ggml.c:2436 @c1d0e7a00).
pub export fn ggml_sin(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_SIN, false);
}

/// Ports `ggml_sin_inplace` (ggml.c:2442 @c1d0e7a00).
pub export fn ggml_sin_inplace(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_SIN, true);
}

/// Ports `ggml_cos` (ggml.c:2462 @c1d0e7a00).
pub export fn ggml_cos(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_COS, false);
}

/// Ports `ggml_cos_inplace` (ggml.c:2468 @c1d0e7a00).
pub export fn ggml_cos_inplace(ctx: *Context, a: *Tensor) *Tensor {
    return unaryOp(ctx, a, c.GGML_OP_COS, true);
}

// -----------------------------------------------------------------------------
// Reductions

/// Ports `ggml_sum` (ggml.c:2476 @c1d0e7a00).
///
/// Collapses every dimension to a single element, keeping the input's type.
pub export fn ggml_sum(ctx: *Context, a: *Tensor) *Tensor {
    const result = context.ggml_new_tensor_1d(ctx, a.type, 1);
    result.op = c.GGML_OP_SUM;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_sum_rows` (ggml.c:2489 @c1d0e7a00).
///
/// Collapses only dimension 0, so a matrix becomes a column.
pub export fn ggml_sum_rows(ctx: *Context, a: *Tensor) *Tensor {
    var ne = [_]i64{ 1, 1, 1, 1 };
    for (1..c.GGML_MAX_DIMS) |i| ne[i] = a.ne[i];

    const result = context.ggml_new_tensor(ctx, a.type, c.GGML_MAX_DIMS, &ne);
    result.op = c.GGML_OP_SUM_ROWS;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_cumsum` (ggml.c:2507 @c1d0e7a00).
pub export fn ggml_cumsum(ctx: *Context, a: *Tensor) *Tensor {
    impl.assert(a.type == c.GGML_TYPE_F32, "a->type == GGML_TYPE_F32");
    const result = context.ggml_dup_tensor(ctx, a);
    result.op = c.GGML_OP_CUMSUM;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_mean` (ggml.c:2522 @c1d0e7a00).
///
/// Always f32 out, whatever went in, because the division needs a float.
pub export fn ggml_mean(ctx: *Context, a: *Tensor) *Tensor {
    const ne = [_]i64{ 1, a.ne[1], a.ne[2], a.ne[3] };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    result.op = c.GGML_OP_MEAN;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_argmax` (ggml.c:2536 @c1d0e7a00).
///
/// Index of the largest element in each row, so the result is i32 and one
/// element per row.
pub export fn ggml_argmax(ctx: *Context, a: *Tensor) *Tensor {
    impl.assert(types.ggml_is_matrix(a), "ggml_is_matrix(a)");
    impl.assert(a.ne[0] <= std.math.maxInt(i32), "a->ne[0] <= INT32_MAX");

    const result = context.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, a.ne[1]);
    result.op = c.GGML_OP_ARGMAX;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_count_equal` (ggml.c:2552 @c1d0e7a00).
///
/// i64 out, because the count can exceed what i32 holds for a large tensor.
pub export fn ggml_count_equal(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(types.ggml_are_same_shape(a, b), "ggml_are_same_shape(a, b)");

    const result = context.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I64, 1);
    result.op = c.GGML_OP_COUNT_EQUAL;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

// -----------------------------------------------------------------------------
// Broadcasting and concatenation

/// Ports `ggml_repeat` (ggml.c:2569 @c1d0e7a00).
///
/// Tiles `a` up to `b`'s shape. `b` supplies only the shape; it is not a source
/// of the node, so it does not create a graph dependency.
pub export fn ggml_repeat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(types.ggml_can_repeat(a, b), "ggml_can_repeat(a, b)");

    const result = context.ggml_new_tensor(ctx, a.type, c.GGML_MAX_DIMS, &b.ne);
    result.op = c.GGML_OP_REPEAT;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_repeat_4d` (ggml.c:2583 @c1d0e7a00).
///
/// Same as `ggml_repeat` with the target shape given directly. The check is
/// spelled out rather than reusing `ggml_can_repeat`, since there is no tensor
/// to compare against.
pub export fn ggml_repeat_4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
    const can_repeat = types.ggml_is_empty(a) or
        (@rem(ne0, a.ne[0]) == 0 and
            @rem(ne1, a.ne[1]) == 0 and
            @rem(ne2, a.ne[2]) == 0 and
            @rem(ne3, a.ne[3]) == 0);
    impl.assert(can_repeat, "can_repeat");

    const result = context.ggml_new_tensor_4d(ctx, a.type, ne0, ne1, ne2, ne3);
    result.op = c.GGML_OP_REPEAT;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_repeat_back` (ggml.c:2605 @c1d0e7a00).
///
/// The gradient of `ggml_repeat`: sums the tiles back down.
pub export fn ggml_repeat_back(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(types.ggml_can_repeat(b, a), "ggml_can_repeat(b, a)");

    const result = context.ggml_new_tensor(ctx, a.type, c.GGML_MAX_DIMS, &b.ne);
    result.op = c.GGML_OP_REPEAT_BACK;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_concat` (ggml.c:2621 @c1d0e7a00).
///
/// Joins along `dim`; every other dimension must match exactly.
pub export fn ggml_concat(ctx: *Context, a: *Tensor, b: *Tensor, dim: c_int) *Tensor {
    impl.assert(dim >= 0 and dim < c.GGML_MAX_DIMS, "dim >= 0 && dim < GGML_MAX_DIMS");
    impl.assert(a.type == b.type, "a->type == b->type");

    var ne: [c.GGML_MAX_DIMS]i64 = undefined;
    for (0..c.GGML_MAX_DIMS) |d| {
        if (d == @as(usize, @intCast(dim))) {
            ne[d] = a.ne[d] + b.ne[d];
            continue;
        }
        impl.assert(a.ne[d] == b.ne[d], "a->ne[d] == b->ne[d]");
        ne[d] = a.ne[d];
    }

    const result = context.ggml_new_tensor(ctx, a.type, c.GGML_MAX_DIMS, &ne);
    impl.setOpParamsI32(result, 0, dim);

    result.op = c.GGML_OP_CONCAT;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests

/// A context over a caller-owned buffer, so tests avoid the Mach allocator.
fn testContext(buf: []u8) Context {
    return .{
        .mem_size = buf.len,
        .mem_buffer = buf.ptr,
        .mem_buffer_owned = false,
        .no_alloc = true, // shapes only; these ops never touch data
        .n_objects = 0,
        .objects_begin = null,
        .objects_end = null,
    };
}

test "elementwise ops keep the shape of their first input" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    const b = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    inline for (.{
        .{ ggml_add, c.GGML_OP_ADD },
        .{ ggml_sub, c.GGML_OP_SUB },
        .{ ggml_mul, c.GGML_OP_MUL },
        .{ ggml_div, c.GGML_OP_DIV },
    }) |case| {
        const r = case[0](&ctx, a, b);
        try std.testing.expectEqual(@as(c.enum_ggml_op, case[1]), r.op);
        try std.testing.expect(types.ggml_are_same_shape(r, a));
        try std.testing.expectEqual(@as(?*Tensor, a), r.src[0]);
        try std.testing.expectEqual(@as(?*Tensor, b), r.src[1]);
    }
}

test "in-place ops return a view of their input" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    const b = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const out_of_place = ggml_add(&ctx, a, b);
    try std.testing.expect(!types.ggml_is_view(out_of_place));

    const in_place = ggml_add_inplace(&ctx, a, b);
    try std.testing.expect(types.ggml_is_view(in_place));
    try std.testing.expectEqual(@as(?*Tensor, a), in_place.view_src);
}

test "broadcasting is allowed only when shapes tile" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    const row = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 1);
    // A single row tiles over four rows, so this is a legal add.
    const r = ggml_add(&ctx, a, row);
    try std.testing.expect(types.ggml_are_same_shape(r, a));

    // 3 does not divide 8, so ggml_can_repeat rejects it.
    const bad = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 3, 4);
    try std.testing.expect(!types.ggml_can_repeat(bad, a));
}

test "reductions collapse the dimensions they claim to" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 8, 4, 2);

    const sum = ggml_sum(&ctx, a);
    try std.testing.expectEqual(@as(i64, 1), types.ggml_nelements(sum));

    const rows = ggml_sum_rows(&ctx, a);
    try std.testing.expectEqual(@as(i64, 1), rows.ne[0]);
    try std.testing.expectEqual(@as(i64, 4), rows.ne[1]);
    try std.testing.expectEqual(@as(i64, 2), rows.ne[2]);

    const mean = ggml_mean(&ctx, a);
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F32), mean.type);
    try std.testing.expectEqual(@as(i64, 1), mean.ne[0]);

    // argmax needs a matrix and reports one i32 index per row.
    const m = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    const am = ggml_argmax(&ctx, m);
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_I32), am.type);
    try std.testing.expectEqual(@as(i64, 4), am.ne[0]);

    // count_equal is i64, because the count can exceed i32.
    const ce = ggml_count_equal(&ctx, m, m);
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_I64), ce.type);
}

test "concat extends only the chosen dimension" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 8, 4, 2);
    const b = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 8, 4, 3);

    const r = ggml_concat(&ctx, a, b, 2);
    try std.testing.expectEqual(@as(i64, 8), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 4), r.ne[1]);
    try std.testing.expectEqual(@as(i64, 5), r.ne[2]);
    // The chosen dimension is recorded for the backend to read back.
    try std.testing.expectEqual(@as(i32, 2), impl.getOpParamsI32(r, 0));
}

test "repeat_4d accepts only whole-number tilings" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 4, 2);
    const r = ggml_repeat_4d(&ctx, a, 8, 4, 1, 1);
    try std.testing.expectEqual(@as(i64, 8), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 4), r.ne[1]);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_REPEAT), r.op);
    // The shape source is not a graph dependency.
    try std.testing.expectEqual(@as(?*Tensor, a), r.src[0]);
    try std.testing.expect(r.src[1] == null);
}

test "acc records its strides and offset in op_params" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    const b = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 8);

    const r = ggml_acc(&ctx, a, b, 32, 64, 96, 16);
    try std.testing.expectEqual(@as(i32, 32), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i32, 64), impl.getOpParamsI32(r, 1));
    try std.testing.expectEqual(@as(i32, 96), impl.getOpParamsI32(r, 2));
    try std.testing.expectEqual(@as(i32, 16), impl.getOpParamsI32(r, 3));
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(r, 4));

    const ri = ggml_acc_inplace(&ctx, a, b, 32, 64, 96, 16);
    try std.testing.expectEqual(@as(i32, 1), impl.getOpParamsI32(ri, 4));
}

// -----------------------------------------------------------------------------
// Unary activations
//
// These share one op code, `GGML_OP_UNARY`, with the specific activation in
// `op_params[0]`. The C writes out forty near-identical wrappers; here the
// bodies are generated from a table and exported under the C names, so the
// mapping from symbol to activation is stated once and cannot drift.

/// Ports `ggml_unary_impl` (ggml.c:5935 @c1d0e7a00).
fn unaryActivation(ctx: *Context, a: *Tensor, op: c.enum_ggml_unary_op, inplace: bool) *Tensor {
    impl.assert(types.ggml_is_contiguous_rows(a), "ggml_is_contiguous_rows(a)");

    const result = dupOrView(ctx, a, inplace);
    impl.setOpParamsI32(result, 0, @intCast(op));

    result.op = c.GGML_OP_UNARY;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_unary` (ggml.c:5952 @c1d0e7a00).
pub export fn ggml_unary(ctx: *Context, a: *Tensor, op: c.enum_ggml_unary_op) *Tensor {
    return unaryActivation(ctx, a, op, false);
}

/// Ports `ggml_unary_inplace` (ggml.c:5959 @c1d0e7a00).
pub export fn ggml_unary_inplace(ctx: *Context, a: *Tensor, op: c.enum_ggml_unary_op) *Tensor {
    return unaryActivation(ctx, a, op, true);
}

/// Builds one activation wrapper. `op` and `inplace` are comptime, so each
/// instantiation is the same single tail call the C wrapper compiles to.
fn activationWrapper(
    comptime op: c.enum_ggml_unary_op,
    comptime inplace: bool,
) fn (*Context, *Tensor) callconv(.c) *Tensor {
    return struct {
        fn f(ctx: *Context, a: *Tensor) callconv(.c) *Tensor {
            return unaryActivation(ctx, a, op, inplace);
        }
    }.f;
}

/// The activation wrappers ggml exports, each naming its `GGML_UNARY_OP_*`.
///
/// Taken from ggml.c lines 2650-2900. Note `hardswish` and `hardsigmoid` have
/// no in-place form there, and so have none here.
const activations = .{
    .{ "ggml_abs", c.GGML_UNARY_OP_ABS, false },
    .{ "ggml_abs_inplace", c.GGML_UNARY_OP_ABS, true },
    .{ "ggml_sgn", c.GGML_UNARY_OP_SGN, false },
    .{ "ggml_sgn_inplace", c.GGML_UNARY_OP_SGN, true },
    .{ "ggml_neg", c.GGML_UNARY_OP_NEG, false },
    .{ "ggml_neg_inplace", c.GGML_UNARY_OP_NEG, true },
    .{ "ggml_step", c.GGML_UNARY_OP_STEP, false },
    .{ "ggml_step_inplace", c.GGML_UNARY_OP_STEP, true },
    .{ "ggml_tanh", c.GGML_UNARY_OP_TANH, false },
    .{ "ggml_tanh_inplace", c.GGML_UNARY_OP_TANH, true },
    .{ "ggml_elu", c.GGML_UNARY_OP_ELU, false },
    .{ "ggml_elu_inplace", c.GGML_UNARY_OP_ELU, true },
    .{ "ggml_relu", c.GGML_UNARY_OP_RELU, false },
    .{ "ggml_relu_inplace", c.GGML_UNARY_OP_RELU, true },
    .{ "ggml_sigmoid", c.GGML_UNARY_OP_SIGMOID, false },
    .{ "ggml_sigmoid_inplace", c.GGML_UNARY_OP_SIGMOID, true },
    .{ "ggml_gelu", c.GGML_UNARY_OP_GELU, false },
    .{ "ggml_gelu_inplace", c.GGML_UNARY_OP_GELU, true },
    .{ "ggml_gelu_erf", c.GGML_UNARY_OP_GELU_ERF, false },
    .{ "ggml_gelu_erf_inplace", c.GGML_UNARY_OP_GELU_ERF, true },
    .{ "ggml_gelu_quick", c.GGML_UNARY_OP_GELU_QUICK, false },
    .{ "ggml_gelu_quick_inplace", c.GGML_UNARY_OP_GELU_QUICK, true },
    .{ "ggml_silu", c.GGML_UNARY_OP_SILU, false },
    .{ "ggml_silu_inplace", c.GGML_UNARY_OP_SILU, true },
    .{ "ggml_hardswish", c.GGML_UNARY_OP_HARDSWISH, false },
    .{ "ggml_hardsigmoid", c.GGML_UNARY_OP_HARDSIGMOID, false },
    .{ "ggml_exp", c.GGML_UNARY_OP_EXP, false },
    .{ "ggml_exp_inplace", c.GGML_UNARY_OP_EXP, true },
    .{ "ggml_expm1", c.GGML_UNARY_OP_EXPM1, false },
    .{ "ggml_expm1_inplace", c.GGML_UNARY_OP_EXPM1, true },
    .{ "ggml_softplus", c.GGML_UNARY_OP_SOFTPLUS, false },
    .{ "ggml_softplus_inplace", c.GGML_UNARY_OP_SOFTPLUS, true },
    .{ "ggml_floor", c.GGML_UNARY_OP_FLOOR, false },
    .{ "ggml_floor_inplace", c.GGML_UNARY_OP_FLOOR, true },
    .{ "ggml_ceil", c.GGML_UNARY_OP_CEIL, false },
    .{ "ggml_ceil_inplace", c.GGML_UNARY_OP_CEIL, true },
    .{ "ggml_round", c.GGML_UNARY_OP_ROUND, false },
    .{ "ggml_round_inplace", c.GGML_UNARY_OP_ROUND, true },
    .{ "ggml_trunc", c.GGML_UNARY_OP_TRUNC, false },
    .{ "ggml_trunc_inplace", c.GGML_UNARY_OP_TRUNC, true },
};

comptime {
    for (activations) |entry| {
        @export(&activationWrapper(entry[1], entry[2]), .{ .name = entry[0] });
    }
}

/// Named handles on the generated wrappers above, for callers inside Zig.
///
/// The exports are produced in a loop, so each exists as a symbol but not as a
/// declaration -- there is no `ops.ggml_neg` to call. The autodiff pass in
/// `graph.zig` needs exactly these six by name.
///
/// They are namespaced rather than declared as `ggml_neg` and friends because
/// the tests below deliberately reach the same functions through `extern`
/// declarations, so that a mis-wired entry in the `activations` table is
/// caught. A Zig binding under the C's own name would collide with that, and
/// resolving the collision the other way would weaken the test into re-
/// deriving the mapping from the table it is checking.
pub const unary = struct {
    pub const neg = activationWrapper(c.GGML_UNARY_OP_NEG, false);
    pub const sgn = activationWrapper(c.GGML_UNARY_OP_SGN, false);
    pub const step = activationWrapper(c.GGML_UNARY_OP_STEP, false);
    pub const silu = activationWrapper(c.GGML_UNARY_OP_SILU, false);
    pub const exp = activationWrapper(c.GGML_UNARY_OP_EXP, false);
    pub const sigmoid = activationWrapper(c.GGML_UNARY_OP_SIGMOID, false);
};

/// Ports `ggml_xielu` (ggml.c:2837 @c1d0e7a00).
///
/// The only activation with parameters of its own. `alpha_n` and `alpha_p`
/// arrive raw and are passed through softplus here, at graph-build time, so
/// the backend reads values it can use directly.
pub export fn ggml_xielu(ctx: *Context, a: *Tensor, alpha_n: f32, alpha_p: f32, beta: f32, eps: f32) *Tensor {
    const result = context.ggml_dup_tensor(ctx, a);

    impl.setOpParamsI32(result, 0, @intCast(c.GGML_UNARY_OP_XIELU));
    impl.setOpParamsF32(result, 1, beta + impl.softplus(alpha_n));
    impl.setOpParamsF32(result, 2, impl.softplus(alpha_p));
    impl.setOpParamsF32(result, 3, beta);
    impl.setOpParamsF32(result, 4, eps);

    result.op = c.GGML_OP_UNARY;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_silu_back` (ggml.c:2860 @c1d0e7a00).
pub export fn ggml_silu_back(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    const result = context.ggml_dup_tensor(ctx, a);
    result.op = c.GGML_OP_SILU_BACK;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

// -----------------------------------------------------------------------------
// Gated linear units
//
// A GLU splits its input in half along dimension 0, gates one half by the
// other, and so halves that dimension. The `_split` forms take the two halves
// as separate tensors instead, in which case the shape is unchanged.

/// Ports `ggml_glu_impl` (ggml.c:2905 @c1d0e7a00).
fn gluImpl(ctx: *Context, a: *Tensor, b: ?*Tensor, op: c.enum_ggml_glu_op, swapped: bool) *Tensor {
    impl.assert(types.ggml_is_contiguous_1(a), "ggml_is_contiguous_1(a)");

    if (b) |bt| {
        impl.assert(types.ggml_is_contiguous_1(bt), "ggml_is_contiguous_1(b)");
        impl.assert(types.ggml_are_same_shape(a, bt), "ggml_are_same_shape(a, b)");
        impl.assert(a.type == bt.type, "a->type == b->type");
    }

    // Halved along dimension 0 when gating within one tensor; unchanged when
    // the halves arrive separately.
    var ne = [_]i64{ @divTrunc(a.ne[0], 2), 1, 1, 1 };
    for (1..c.GGML_MAX_DIMS) |i| ne[i] = a.ne[i];
    const shape: [*]const i64 = if (b != null) &a.ne else &ne;

    const result = context.newTensorImpl(ctx, a.type, c.GGML_MAX_DIMS, shape, null, 0);

    impl.setOpParamsI32(result, 0, @intCast(op));
    impl.setOpParamsI32(result, 1, if (swapped) 1 else 0);

    result.op = c.GGML_OP_GLU;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_glu` (ggml.c:2988 @c1d0e7a00).
pub export fn ggml_glu(ctx: *Context, a: *Tensor, op: c.enum_ggml_glu_op, swapped: bool) *Tensor {
    return gluImpl(ctx, a, null, op, swapped);
}

/// Ports `ggml_glu_split` (ggml.c:2996 @c1d0e7a00).
pub export fn ggml_glu_split(ctx: *Context, a: *Tensor, b: *Tensor, op: c.enum_ggml_glu_op) *Tensor {
    return gluImpl(ctx, a, b, op, false);
}

/// Builds a single-tensor GLU wrapper.
fn gluWrapper(
    comptime op: c.enum_ggml_glu_op,
    comptime swapped: bool,
) fn (*Context, *Tensor) callconv(.c) *Tensor {
    return struct {
        fn f(ctx: *Context, a: *Tensor) callconv(.c) *Tensor {
            return gluImpl(ctx, a, null, op, swapped);
        }
    }.f;
}

/// Builds a split GLU wrapper, taking the two halves separately.
fn gluSplitWrapper(comptime op: c.enum_ggml_glu_op) fn (*Context, *Tensor, *Tensor) callconv(.c) *Tensor {
    return struct {
        fn f(ctx: *Context, a: *Tensor, b: *Tensor) callconv(.c) *Tensor {
            return gluImpl(ctx, a, b, op, false);
        }
    }.f;
}

/// The GLU wrappers, from ggml.c lines 3004-3120. A `_swapped` form gates on
/// the other half.
const glus = .{
    .{ "ggml_reglu", c.GGML_GLU_OP_REGLU, false },
    .{ "ggml_reglu_swapped", c.GGML_GLU_OP_REGLU, true },
    .{ "ggml_geglu", c.GGML_GLU_OP_GEGLU, false },
    .{ "ggml_geglu_swapped", c.GGML_GLU_OP_GEGLU, true },
    .{ "ggml_swiglu", c.GGML_GLU_OP_SWIGLU, false },
    .{ "ggml_swiglu_swapped", c.GGML_GLU_OP_SWIGLU, true },
    .{ "ggml_geglu_erf", c.GGML_GLU_OP_GEGLU_ERF, false },
    .{ "ggml_geglu_erf_swapped", c.GGML_GLU_OP_GEGLU_ERF, true },
    .{ "ggml_geglu_quick", c.GGML_GLU_OP_GEGLU_QUICK, false },
    .{ "ggml_geglu_quick_swapped", c.GGML_GLU_OP_GEGLU_QUICK, true },
};

const glu_splits = .{
    .{ "ggml_reglu_split", c.GGML_GLU_OP_REGLU },
    .{ "ggml_geglu_split", c.GGML_GLU_OP_GEGLU },
    .{ "ggml_swiglu_split", c.GGML_GLU_OP_SWIGLU },
    .{ "ggml_geglu_erf_split", c.GGML_GLU_OP_GEGLU_ERF },
    .{ "ggml_geglu_quick_split", c.GGML_GLU_OP_GEGLU_QUICK },
};

comptime {
    for (glus) |entry| {
        @export(&gluWrapper(entry[1], entry[2]), .{ .name = entry[0] });
    }
    for (glu_splits) |entry| {
        @export(&gluSplitWrapper(entry[1]), .{ .name = entry[0] });
    }
}

// -----------------------------------------------------------------------------
// Unit Tests for activations and GLU

// The generated wrappers exist as symbols, not as Zig identifiers, so the
// tests reach them the way C does. This is the only way to check that the
// export table wired each name to the activation it claims: calling
// `activationWrapper` directly would re-derive the mapping from the same table
// under test.
extern fn ggml_relu(ctx: *Context, a: *Tensor) *Tensor;
extern fn ggml_gelu(ctx: *Context, a: *Tensor) *Tensor;
extern fn ggml_silu(ctx: *Context, a: *Tensor) *Tensor;
extern fn ggml_relu_inplace(ctx: *Context, a: *Tensor) *Tensor;
extern fn ggml_hardswish(ctx: *Context, a: *Tensor) *Tensor;
extern fn ggml_swiglu(ctx: *Context, a: *Tensor) *Tensor;
extern fn ggml_swiglu_swapped(ctx: *Context, a: *Tensor) *Tensor;
extern fn ggml_swiglu_split(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor;
extern fn ggml_geglu(ctx: *Context, a: *Tensor) *Tensor;

test "exported activation names map to the right activation" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 8);

    // Called through the real symbols, so a mis-wired table entry shows up.
    try std.testing.expectEqualStrings("RELU", std.mem.span(types.ggml_op_desc(ggml_relu(&ctx, a))));
    try std.testing.expectEqualStrings("GELU", std.mem.span(types.ggml_op_desc(ggml_gelu(&ctx, a))));
    try std.testing.expectEqualStrings("SILU", std.mem.span(types.ggml_op_desc(ggml_silu(&ctx, a))));
    try std.testing.expectEqualStrings("HARDSWISH", std.mem.span(types.ggml_op_desc(ggml_hardswish(&ctx, a))));
    try std.testing.expectEqualStrings("SWIGLU", std.mem.span(types.ggml_op_desc(ggml_swiglu(&ctx, a))));
    try std.testing.expectEqualStrings("GEGLU", std.mem.span(types.ggml_op_desc(ggml_geglu(&ctx, a))));

    // The in-place symbol must actually be in-place.
    try std.testing.expect(!types.ggml_is_view(ggml_relu(&ctx, a)));
    try std.testing.expect(types.ggml_is_view(ggml_relu_inplace(&ctx, a)));
}

test "activations share one op code and differ in op_params" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const r = ggml_relu(&ctx, a);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_UNARY), r.op);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_UNARY_OP_RELU)), impl.getOpParamsI32(r, 0));
    try std.testing.expect(types.ggml_are_same_shape(r, a));

    const g = ggml_gelu(&ctx, a);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_UNARY_OP_GELU)), impl.getOpParamsI32(g, 0));

    // ggml_op_desc must report the activation, not "UNARY".
    try std.testing.expectEqualStrings("RELU", std.mem.span(types.ggml_op_desc(r)));
    try std.testing.expectEqualStrings("GELU", std.mem.span(types.ggml_op_desc(g)));
}

test "every generated activation wrapper reports its own op" {
    var buf: [1 << 20]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 8);

    // Walk the table so a wrapper wired to the wrong activation is caught.
    inline for (activations) |entry| {
        const f = activationWrapper(entry[1], entry[2]);
        const r = f(&ctx, a);
        try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_UNARY), r.op);
        try std.testing.expectEqual(@as(i32, @intCast(entry[1])), impl.getOpParamsI32(r, 0));
        try std.testing.expectEqual(entry[2], types.ggml_is_view(r));
    }
}

test "xielu passes its alphas through softplus" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 8);

    const r = ggml_xielu(&ctx, a, 0.5, 1.5, 0.25, 1e-6);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_UNARY_OP_XIELU)), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(0.25 + impl.softplus(0.5), impl.getOpParamsF32(r, 1));
    try std.testing.expectEqual(impl.softplus(1.5), impl.getOpParamsF32(r, 2));
    try std.testing.expectEqual(@as(f32, 0.25), impl.getOpParamsF32(r, 3));
    try std.testing.expectEqual(@as(f32, 1e-6), impl.getOpParamsF32(r, 4));

    // Above 20 softplus is the identity, since expf would overflow.
    try std.testing.expectEqual(@as(f32, 25.0), impl.softplus(25.0));
}

test "glu halves dimension 0 unless the halves arrive separately" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    const gated = ggml_swiglu(&ctx, a);
    try std.testing.expectEqual(@as(i64, 4), gated.ne[0]);
    try std.testing.expectEqual(@as(i64, 4), gated.ne[1]);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_GLU_OP_SWIGLU)), impl.getOpParamsI32(gated, 0));
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(gated, 1));

    const swapped = ggml_swiglu_swapped(&ctx, a);
    try std.testing.expectEqual(@as(i32, 1), impl.getOpParamsI32(swapped, 1));

    // Split form keeps the shape, because nothing is being divided.
    const b = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    const split = ggml_swiglu_split(&ctx, a, b);
    try std.testing.expectEqual(@as(i64, 8), split.ne[0]);
    try std.testing.expectEqual(@as(?*Tensor, b), split.src[1]);
    try std.testing.expectEqualStrings("SWIGLU", std.mem.span(types.ggml_op_desc(split)));
}

// -----------------------------------------------------------------------------
// Normalisations
//
// Each takes an epsilon and stores it in `op_params`. The variants differ in
// what they normalise over: `norm` over a row's mean and variance, `rms_norm`
// over the root mean square only, `group_norm` over channel groups, `l2_norm`
// to unit length.

/// Builds a normalisation wrapper taking only an epsilon.
fn normWrapper(
    comptime op: c.enum_ggml_op,
    comptime inplace: bool,
) fn (*Context, *Tensor, f32) callconv(.c) *Tensor {
    return struct {
        fn f(ctx: *Context, a: *Tensor, eps: f32) callconv(.c) *Tensor {
            const result = dupOrView(ctx, a, inplace);
            impl.setOpParamsValue(result, eps);
            result.op = op;
            result.src[0] = a;
            return result;
        }
    }.f;
}

/// The epsilon-only normalisations, from ggml.c lines 3120-3265.
///
/// `l2_norm` writes its epsilon with `set_op_params_f32(result, 0, eps)` in the
/// C rather than `set_op_params(&eps)`; both put a float at index 0, so the
/// result is identical.
const norms = .{
    .{ "ggml_norm", c.GGML_OP_NORM, false },
    .{ "ggml_norm_inplace", c.GGML_OP_NORM, true },
    .{ "ggml_rms_norm", c.GGML_OP_RMS_NORM, false },
    .{ "ggml_rms_norm_inplace", c.GGML_OP_RMS_NORM, true },
    .{ "ggml_l2_norm", c.GGML_OP_L2_NORM, false },
    .{ "ggml_l2_norm_inplace", c.GGML_OP_L2_NORM, true },
};

comptime {
    for (norms) |entry| {
        @export(&normWrapper(entry[1], entry[2]), .{ .name = entry[0] });
    }
}

/// Ports `ggml_rms_norm_back` (ggml.c:3186 @c1d0e7a00).
pub export fn ggml_rms_norm_back(ctx: *Context, a: *Tensor, b: *Tensor, eps: f32) *Tensor {
    const result = context.ggml_dup_tensor(ctx, a);
    impl.setOpParamsValue(result, eps);
    result.op = c.GGML_OP_RMS_NORM_BACK;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_group_norm_impl` (ggml.c:3204 @c1d0e7a00).
fn groupNormImpl(ctx: *Context, a: *Tensor, n_groups: c_int, eps: f32, inplace: bool) *Tensor {
    const result = dupOrView(ctx, a, inplace);
    impl.setOpParamsI32(result, 0, n_groups);
    impl.setOpParamsF32(result, 1, eps);
    result.op = c.GGML_OP_GROUP_NORM;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_group_norm` (ggml.c:3221 @c1d0e7a00).
pub export fn ggml_group_norm(ctx: *Context, a: *Tensor, n_groups: c_int, eps: f32) *Tensor {
    return groupNormImpl(ctx, a, n_groups, eps, false);
}

/// Ports `ggml_group_norm_inplace` (ggml.c:3229 @c1d0e7a00).
pub export fn ggml_group_norm_inplace(ctx: *Context, a: *Tensor, n_groups: c_int, eps: f32) *Tensor {
    return groupNormImpl(ctx, a, n_groups, eps, true);
}

// -----------------------------------------------------------------------------
// Matrix multiplication

/// Ports `ggml_can_mul_mat` (ggml.c:3270 @c1d0e7a00).
///
/// The shared dimension must match exactly; the batch dimensions of `t0` need
/// only divide `t1`'s, so a single weight matrix broadcasts over a batch.
fn canMulMat(t0: *const Tensor, t1: *const Tensor) bool {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return t0.ne[0] == t1.ne[0] and
        @rem(t1.ne[2], t0.ne[2]) == 0 and
        @rem(t1.ne[3], t0.ne[3]) == 0;
}

/// Ports `ggml_mul_mat` (ggml.c:3278 @c1d0e7a00).
///
/// Always f32 out, whatever the input types, because that is what the backends
/// accumulate in. Note `a` is the weight and is indexed by its second
/// dimension, so the result's first dimension is `a->ne[1]`.
pub export fn ggml_mul_mat(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(canMulMat(a, b), "ggml_can_mul_mat(a, b)");
    impl.assert(!types.ggml_is_transposed(a), "!ggml_is_transposed(a)");

    const ne = [_]i64{ a.ne[1], b.ne[1], b.ne[2], b.ne[3] };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    result.op = c.GGML_OP_MUL_MAT;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_mul_mat_set_prec` (ggml.c:3295 @c1d0e7a00).
///
/// Mutates an existing node rather than building one, so it must be called on
/// a tensor that is already a matmul.
pub export fn ggml_mul_mat_set_prec(a: *Tensor, prec: c.enum_ggml_prec) void {
    impl.assert(a.op == c.GGML_OP_MUL_MAT, "a->op == GGML_OP_MUL_MAT");
    impl.setOpParamsI32(a, 0, @intCast(prec));
}

/// Ports `ggml_mul_mat_set_hint` (ggml.c:3305 @c1d0e7a00).
pub export fn ggml_mul_mat_set_hint(a: *Tensor, hint: c.enum_ggml_op_hint) void {
    impl.assert(a.op == c.GGML_OP_MUL_MAT, "a->op == GGML_OP_MUL_MAT");
    impl.setOpParamsI32(a, 1, @intCast(hint));
}

/// Ports `ggml_mul_mat_id` (ggml.c:3329 @c1d0e7a00).
///
/// Mixture-of-experts matmul: `ids` selects which expert matrix in `as` each
/// row of `b` goes through, so the experts are never materialised as one big
/// tensor. From the C's own sketch:
///
///     as  -> [cols, rows, n_expert]
///     b   -> [cols, n_expert_used, n_tokens]
///     ids -> [n_expert_used, n_tokens] (i32)
///     c   -> [rows, n_expert_used, n_tokens]
pub export fn ggml_mul_mat_id(ctx: *Context, as: *Tensor, b: *Tensor, ids: *Tensor) *Tensor {
    impl.assert(!types.ggml_is_transposed(as), "!ggml_is_transposed(as)");
    impl.assert(ids.type == c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    impl.assert(as.ne[3] == 1, "as->ne[3] == 1"); // one matrix per expert
    impl.assert(b.ne[3] == 1, "b->ne[3] == 1");
    impl.assert(ids.ne[2] == 1 and ids.ne[3] == 1, "ids->ne[2] == 1 && ids->ne[3] == 1");
    impl.assert(ids.ne[1] == b.ne[2], "ids->ne[1] == b->ne[2]"); // an expert list per row
    impl.assert(as.ne[0] == b.ne[0], "as->ne[0] == b->ne[0]");
    impl.assert(@rem(ids.ne[0], b.ne[1]) == 0, "ids->ne[0] % b->ne[1] == 0");

    const ne = [_]i64{ as.ne[1], ids.ne[0], b.ne[2], 1 };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    result.op = c.GGML_OP_MUL_MAT_ID;
    result.src[0] = as;
    result.src[1] = b;
    result.src[2] = ids;
    return result;
}

/// Ports `ggml_can_out_prod` (ggml.c:3357 @c1d0e7a00).
fn canOutProd(t0: *const Tensor, t1: *const Tensor) bool {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return t0.ne[1] == t1.ne[1] and
        @rem(t1.ne[2], t0.ne[2]) == 0 and
        @rem(t1.ne[3], t0.ne[3]) == 0;
}

/// Ports `ggml_out_prod` (ggml.c:3365 @c1d0e7a00).
pub export fn ggml_out_prod(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(canOutProd(a, b), "ggml_can_out_prod(a, b)");
    impl.assert(!types.ggml_is_transposed(a), "!ggml_is_transposed(a)");

    // `a` broadcasts over `b`'s batch dimensions, so `b`'s are the result's.
    const ne = [_]i64{ a.ne[0], b.ne[0], b.ne[2], b.ne[3] };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    result.op = c.GGML_OP_OUT_PROD;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

// -----------------------------------------------------------------------------
// Scaling

/// Ports `ggml_scale_impl` (ggml.c:3385 @c1d0e7a00).
///
/// Computes `s * x + b`, so the bias is part of the same node rather than a
/// separate add.
pub fn scaleImpl(ctx: *Context, a: *Tensor, s: f32, b: f32, inplace: bool) *Tensor {
    impl.assert(types.isPadded1d(a), "ggml_is_padded_1d(a)");

    const result = dupOrView(ctx, a, inplace);
    impl.setOpParamsValue(result, [_]f32{ s, b });

    result.op = c.GGML_OP_SCALE;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_scale` (ggml.c:3404 @c1d0e7a00).
pub export fn ggml_scale(ctx: *Context, a: *Tensor, s: f32) *Tensor {
    return scaleImpl(ctx, a, s, 0.0, false);
}

/// Ports `ggml_scale_inplace` (ggml.c:3411 @c1d0e7a00).
pub export fn ggml_scale_inplace(ctx: *Context, a: *Tensor, s: f32) *Tensor {
    return scaleImpl(ctx, a, s, 0.0, true);
}

/// Ports `ggml_scale_bias` (ggml.c:3418 @c1d0e7a00).
pub export fn ggml_scale_bias(ctx: *Context, a: *Tensor, s: f32, b: f32) *Tensor {
    return scaleImpl(ctx, a, s, b, false);
}

/// Ports `ggml_scale_bias_inplace` (ggml.c:3426 @c1d0e7a00).
pub export fn ggml_scale_bias_inplace(ctx: *Context, a: *Tensor, s: f32, b: f32) *Tensor {
    return scaleImpl(ctx, a, s, b, true);
}

// -----------------------------------------------------------------------------
// Unit Tests for norms and matmul

extern fn ggml_rms_norm(ctx: *Context, a: *Tensor, eps: f32) *Tensor;
extern fn ggml_norm(ctx: *Context, a: *Tensor, eps: f32) *Tensor;
extern fn ggml_l2_norm(ctx: *Context, a: *Tensor, eps: f32) *Tensor;
extern fn ggml_rms_norm_inplace(ctx: *Context, a: *Tensor, eps: f32) *Tensor;

test "normalisations record their epsilon and keep shape" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const r = ggml_rms_norm(&ctx, a, 1e-5);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_RMS_NORM), r.op);
    try std.testing.expectEqual(@as(f32, 1e-5), impl.getOpParamsF32(r, 0));
    try std.testing.expect(types.ggml_are_same_shape(r, a));

    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_NORM), ggml_norm(&ctx, a, 1e-5).op);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_L2_NORM), ggml_l2_norm(&ctx, a, 1e-5).op);
    try std.testing.expect(types.ggml_is_view(ggml_rms_norm_inplace(&ctx, a, 1e-5)));

    const g = ggml_group_norm(&ctx, a, 4, 1e-6);
    try std.testing.expectEqual(@as(i32, 4), impl.getOpParamsI32(g, 0));
    try std.testing.expectEqual(@as(f32, 1e-6), impl.getOpParamsF32(g, 1));
}

test "mul_mat contracts the shared dimension" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // Weight [k=8, n=16] times activations [k=8, m=4] gives [n=16, m=4].
    const w = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 16);
    const x = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const r = ggml_mul_mat(&ctx, w, x);
    try std.testing.expectEqual(@as(i64, 16), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 4), r.ne[1]);
    // f32 out regardless of input type, which is what the backends accumulate in.
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F32), r.type);

    const q = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_Q4_K, 256, 16);
    const xq = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 256, 4);
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F32), ggml_mul_mat(&ctx, q, xq).type);

    // The shared dimension must match exactly; batch dimensions may broadcast.
    try std.testing.expect(canMulMat(w, x));
    const mismatched = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 7, 4);
    try std.testing.expect(!canMulMat(w, mismatched));

    const batched = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 8, 4, 6);
    try std.testing.expect(canMulMat(w, batched)); // w has ne[2]==1, divides 6
}

test "mul_mat precision and hint land in distinct op_params slots" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const w = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 16);
    const x = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const r = ggml_mul_mat(&ctx, w, x);
    ggml_mul_mat_set_prec(r, c.GGML_PREC_F32);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_PREC_F32)), impl.getOpParamsI32(r, 0));
    // Setting the hint must not disturb the precision.
    ggml_mul_mat_set_hint(r, c.GGML_HINT_SRC0_IS_HADAMARD);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_PREC_F32)), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_HINT_SRC0_IS_HADAMARD)), impl.getOpParamsI32(r, 1));
}

test "scale carries both factor and bias" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 8);

    const plain = ggml_scale(&ctx, a, 2.5);
    try std.testing.expectEqual(@as(f32, 2.5), impl.getOpParamsF32(plain, 0));
    try std.testing.expectEqual(@as(f32, 0.0), impl.getOpParamsF32(plain, 1));

    const biased = ggml_scale_bias(&ctx, a, 2.5, -1.25);
    try std.testing.expectEqual(@as(f32, 2.5), impl.getOpParamsF32(biased, 0));
    try std.testing.expectEqual(@as(f32, -1.25), impl.getOpParamsF32(biased, 1));
    try std.testing.expect(types.ggml_is_view(ggml_scale_inplace(&ctx, a, 2.0)));
}

test "out_prod widens rather than contracts" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 4, 8);
    const b = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 6, 8);

    const r = ggml_out_prod(&ctx, a, b);
    try std.testing.expectEqual(@as(i64, 4), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 6), r.ne[1]);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_OUT_PROD), r.op);
}

// -----------------------------------------------------------------------------
// Writing into a tensor

/// Ports `ggml_set_impl` (ggml.c:3436 @c1d0e7a00).
///
/// Writes `b` into a strided window of `a`. The offset is capped at 2^30
/// because it is stored as an i32 in `op_params`, and the C asserts that
/// rather than silently truncating.
fn setImpl(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
    inplace: bool,
) *Tensor {
    impl.assert(types.ggml_nelements(a) >= types.ggml_nelements(b), "ggml_nelements(a) >= ggml_nelements(b)");

    const result = dupOrView(ctx, a, inplace);

    impl.assert(offset < (1 << 30), "offset < (size_t)(1 << 30)");
    const params = [_]i32{
        @intCast(nb1),         @intCast(nb2), @intCast(nb3), @intCast(offset),
        if (inplace) 1 else 0,
    };
    impl.setOpParamsValue(result, params);

    result.op = c.GGML_OP_SET;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_set` (ggml.c:3461 @c1d0e7a00).
pub export fn ggml_set(ctx: *Context, a: *Tensor, b: *Tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize) *Tensor {
    return setImpl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

/// Ports `ggml_set_inplace` (ggml.c:3472 @c1d0e7a00).
pub export fn ggml_set_inplace(ctx: *Context, a: *Tensor, b: *Tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize) *Tensor {
    return setImpl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

/// Ports `ggml_set_1d` (ggml.c:3483 @c1d0e7a00).
///
/// Takes `a`'s own strides, so the window is a contiguous run.
pub export fn ggml_set_1d(ctx: *Context, a: *Tensor, b: *Tensor, offset: usize) *Tensor {
    return setImpl(ctx, a, b, a.nb[1], a.nb[2], a.nb[3], offset, false);
}

/// Ports `ggml_set_1d_inplace` (ggml.c:3491 @c1d0e7a00).
pub export fn ggml_set_1d_inplace(ctx: *Context, a: *Tensor, b: *Tensor, offset: usize) *Tensor {
    return setImpl(ctx, a, b, a.nb[1], a.nb[2], a.nb[3], offset, true);
}

/// Ports `ggml_set_2d` (ggml.c:3499 @c1d0e7a00).
pub export fn ggml_set_2d(ctx: *Context, a: *Tensor, b: *Tensor, nb1: usize, offset: usize) *Tensor {
    return setImpl(ctx, a, b, nb1, a.nb[2], a.nb[3], offset, false);
}

/// Ports `ggml_set_2d_inplace` (ggml.c:3508 @c1d0e7a00).
pub export fn ggml_set_2d_inplace(ctx: *Context, a: *Tensor, b: *Tensor, nb1: usize, offset: usize) *Tensor {
    return setImpl(ctx, a, b, nb1, a.nb[2], a.nb[3], offset, true);
}

// -----------------------------------------------------------------------------
// Copying and layout changes
//
// `cpy` and `cast` both produce `GGML_OP_CPY`; `cont` produces a contiguous
// copy. All three exist because a view can have any strides, and most kernels
// need contiguous input.

/// Ports `ggml_cpy_impl` (ggml.c:3519 @c1d0e7a00).
///
/// The result is a view of the destination, not of the source: the node writes
/// into `b`'s storage.
fn cpyImpl(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(types.ggml_nelements(a) == types.ggml_nelements(b), "ggml_nelements(a) == ggml_nelements(b)");

    const result = context.ggml_view_tensor(ctx, b);
    if (b.name[0] != 0) {
        _ = context.ggml_format_name(result, "%s (copy of %s)", &b.name, &a.name);
    } else {
        _ = context.ggml_format_name(result, "%s (copy)", &a.name);
    }

    result.op = c.GGML_OP_CPY;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_cpy` (ggml.c:3540 @c1d0e7a00).
pub export fn ggml_cpy(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return cpyImpl(ctx, a, b);
}

/// Ports `ggml_cast` (ggml.c:3547 @c1d0e7a00).
///
/// Note `src[1]` points at the result itself. The C flags this as looking
/// redundant but required: some backends read `src[1]` as the destination, and
/// for a cast the destination is the new tensor.
pub export fn ggml_cast(ctx: *Context, a: *Tensor, t: c.enum_ggml_type) *Tensor {
    const result = context.ggml_new_tensor(ctx, t, c.GGML_MAX_DIMS, &a.ne);
    _ = context.ggml_format_name(result, "%s (copy)", &a.name);

    result.op = c.GGML_OP_CPY;
    result.src[0] = a;
    result.src[1] = result;
    return result;
}

/// Ports `ggml_cont_impl` (ggml.c:3564 @c1d0e7a00).
fn contImpl(ctx: *Context, a: *Tensor) *Tensor {
    const result = context.ggml_dup_tensor(ctx, a);
    _ = context.ggml_format_name(result, "%s (cont)", &a.name);

    result.op = c.GGML_OP_CONT;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_cont` (ggml.c:3576 @c1d0e7a00).
pub export fn ggml_cont(ctx: *Context, a: *Tensor) *Tensor {
    return contImpl(ctx, a);
}

/// Ports `ggml_cont_4d` (ggml.c:3607 @c1d0e7a00).
///
/// Makes contiguous and reshapes in one node. The element count must be
/// preserved; only the layout changes.
pub export fn ggml_cont_4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
    impl.assert(types.ggml_nelements(a) == ne0 * ne1 * ne2 * ne3, "ggml_nelements(a) == (ne0*ne1*ne2*ne3)");

    const result = context.ggml_new_tensor_4d(ctx, a.type, ne0, ne1, ne2, ne3);
    _ = context.ggml_format_name(result, "%s (cont)", &a.name);

    result.op = c.GGML_OP_CONT;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_cont_1d` (ggml.c:3583 @c1d0e7a00).
pub export fn ggml_cont_1d(ctx: *Context, a: *Tensor, ne0: i64) *Tensor {
    return ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

/// Ports `ggml_cont_2d` (ggml.c:3590 @c1d0e7a00).
pub export fn ggml_cont_2d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64) *Tensor {
    return ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

/// Ports `ggml_cont_3d` (ggml.c:3598 @c1d0e7a00).
pub export fn ggml_cont_3d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64) *Tensor {
    return ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

// -----------------------------------------------------------------------------
// Reshaping
//
// A reshape is a view with different extents, so it requires contiguous input
// and costs nothing at run time. `cont_*` is the version that copies.

/// Ports `ggml_reshape` (ggml.c:3627 @c1d0e7a00).
///
/// `b` supplies only the shape, and may itself be non-contiguous; it is not a
/// source of the node.
pub export fn ggml_reshape(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
    impl.assert(types.ggml_nelements(a) == types.ggml_nelements(b), "ggml_nelements(a) == ggml_nelements(b)");

    const result = context.newTensorImpl(ctx, a.type, c.GGML_MAX_DIMS, &b.ne, a, 0);
    _ = context.ggml_format_name(result, "%s (reshaped)", &a.name);

    result.op = c.GGML_OP_RESHAPE;
    result.src[0] = a;
    return result;
}

/// Builds a reshape wrapper for a fixed number of dimensions.
fn reshapeWrapper(comptime n: usize) fn (*Context, *Tensor, [*]const i64) *Tensor {
    return struct {
        fn f(ctx: *Context, a: *Tensor, ne: [*]const i64) *Tensor {
            impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
            var count: i64 = 1;
            for (0..n) |i| count *= ne[i];
            impl.assert(types.ggml_nelements(a) == count, "ggml_nelements(a) == product of ne");

            const result = context.newTensorImpl(ctx, a.type, n, ne, a, 0);
            _ = context.ggml_format_name(result, "%s (reshaped)", &a.name);

            result.op = c.GGML_OP_RESHAPE;
            result.src[0] = a;
            return result;
        }
    }.f;
}

/// Ports `ggml_reshape_1d` (ggml.c:3644 @c1d0e7a00).
pub export fn ggml_reshape_1d(ctx: *Context, a: *Tensor, ne0: i64) *Tensor {
    const ne = [_]i64{ne0};
    return reshapeWrapper(1)(ctx, a, &ne);
}

/// Ports `ggml_reshape_2d` (ggml.c:3661 @c1d0e7a00).
pub export fn ggml_reshape_2d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64) *Tensor {
    const ne = [_]i64{ ne0, ne1 };
    return reshapeWrapper(2)(ctx, a, &ne);
}

/// Ports `ggml_reshape_3d` (ggml.c:3679 @c1d0e7a00).
pub export fn ggml_reshape_3d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64) *Tensor {
    const ne = [_]i64{ ne0, ne1, ne2 };
    return reshapeWrapper(3)(ctx, a, &ne);
}

/// Ports `ggml_reshape_4d` (ggml.c:3698 @c1d0e7a00).
pub export fn ggml_reshape_4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *Tensor {
    const ne = [_]i64{ ne0, ne1, ne2, ne3 };
    return reshapeWrapper(4)(ctx, a, &ne);
}

// -----------------------------------------------------------------------------
// Views
//
// Unlike `ggml_view_tensor`, which copies the source's strides, these take the
// strides explicitly, so a view can stride over a subset of its source.

/// Ports `ggml_view_impl` (ggml.c:3718 @c1d0e7a00).
///
/// The byte offset goes into `op_params` as a `size_t`, not an i32, so views
/// are not subject to the 2^30 limit `ggml_set` has.
fn viewImpl(ctx: *Context, a: *Tensor, n_dims: c_int, ne: [*]const i64, offset: usize) *Tensor {
    const result = context.newTensorImpl(ctx, a.type, n_dims, ne, a, offset);
    _ = context.ggml_format_name(result, "%s (view)", &a.name);

    impl.setOpParamsValue(result, offset);

    result.op = c.GGML_OP_VIEW;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_view_1d` (ggml.c:3737 @c1d0e7a00).
pub export fn ggml_view_1d(ctx: *Context, a: *Tensor, ne0: i64, offset: usize) *Tensor {
    const ne = [_]i64{ne0};
    return viewImpl(ctx, a, 1, &ne, offset);
}

/// Ports `ggml_view_2d` (ggml.c:3749 @c1d0e7a00).
///
/// Only `nb1` is given; the higher strides follow from it, which is what makes
/// this a plain 2-D window rather than an arbitrary layout.
pub export fn ggml_view_2d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) *Tensor {
    const ne = [_]i64{ ne0, ne1 };
    const result = viewImpl(ctx, a, 2, &ne, offset);
    result.nb[1] = nb1;
    result.nb[2] = result.nb[1] * @as(usize, @intCast(ne1));
    result.nb[3] = result.nb[2];
    return result;
}

/// Ports `ggml_view_3d` (ggml.c:3769 @c1d0e7a00).
pub export fn ggml_view_3d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) *Tensor {
    const ne = [_]i64{ ne0, ne1, ne2 };
    const result = viewImpl(ctx, a, 3, &ne, offset);
    result.nb[1] = nb1;
    result.nb[2] = nb2;
    result.nb[3] = result.nb[2] * @as(usize, @intCast(ne2));
    return result;
}

/// Ports `ggml_view_4d` (ggml.c:3791 @c1d0e7a00).
pub export fn ggml_view_4d(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize) *Tensor {
    const ne = [_]i64{ ne0, ne1, ne2, ne3 };
    const result = viewImpl(ctx, a, 4, &ne, offset);
    result.nb[1] = nb1;
    result.nb[2] = nb2;
    result.nb[3] = nb3;
    return result;
}

// -----------------------------------------------------------------------------
// Reordering dimensions

/// Ports `ggml_permute` (ggml.c:3815 @c1d0e7a00).
///
/// The axis arguments say where each of the source's dimensions *goes*, not
/// where it comes from, which is the easy thing to get backwards. All four must
/// be distinct, so this is a permutation and not a projection.
pub export fn ggml_permute(ctx: *Context, a: *Tensor, axis0: c_int, axis1: c_int, axis2: c_int, axis3: c_int) *Tensor {
    const axes = [_]c_int{ axis0, axis1, axis2, axis3 };
    for (axes) |axis| {
        impl.assert(axis >= 0 and axis < c.GGML_MAX_DIMS, "axis >= 0 && axis < GGML_MAX_DIMS");
    }
    for (0..axes.len) |i| {
        for (i + 1..axes.len) |j| {
            impl.assert(axes[i] != axes[j], "axes must be distinct");
        }
    }

    const result = context.ggml_view_tensor(ctx, a);
    _ = context.ggml_format_name(result, "%s (permuted)", &a.name);

    var ne: [c.GGML_MAX_DIMS]i64 = undefined;
    var nb: [c.GGML_MAX_DIMS]usize = undefined;
    for (axes, 0..) |axis, src_dim| {
        ne[@intCast(axis)] = a.ne[src_dim];
        nb[@intCast(axis)] = a.nb[src_dim];
    }
    result.ne = ne;
    result.nb = nb;

    result.op = c.GGML_OP_PERMUTE;
    result.src[0] = a;

    // The axes are recorded as well as applied. The strides above already
    // describe the permutation, but backends read these back -- so omitting
    // them produces a node that looks right and is not.
    impl.setOpParamsValue(result, [_]i32{ axis0, axis1, axis2, axis3 });

    return result;
}

/// Ports `ggml_transpose` (ggml.c:3871 @c1d0e7a00).
///
/// Swaps the first two dimensions by swapping their strides, so no data moves
/// and the result is not contiguous.
pub export fn ggml_transpose(ctx: *Context, a: *Tensor) *Tensor {
    const result = context.ggml_view_tensor(ctx, a);
    _ = context.ggml_format_name(result, "%s (transposed)", &a.name);

    result.ne[0] = a.ne[1];
    result.ne[1] = a.ne[0];
    result.nb[0] = a.nb[1];
    result.nb[1] = a.nb[0];

    result.op = c.GGML_OP_TRANSPOSE;
    result.src[0] = a;
    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests for views and reshaping

test "reshape is a view and preserves the element count" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const r = ggml_reshape_2d(&ctx, a, 4, 8);
    try std.testing.expectEqual(@as(i64, 4), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[1]);
    try std.testing.expectEqual(types.ggml_nelements(a), types.ggml_nelements(r));
    // A reshape shares storage, so it must be a view of its source.
    try std.testing.expectEqual(@as(?*Tensor, a), r.view_src);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_RESHAPE), r.op);

    const flat = ggml_reshape_1d(&ctx, a, 32);
    try std.testing.expectEqual(@as(i64, 32), flat.ne[0]);
    const r4 = ggml_reshape_4d(&ctx, a, 2, 2, 2, 4);
    try std.testing.expectEqual(@as(i64, 32), types.ggml_nelements(r4));
}

test "cont copies rather than viewing" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const r = ggml_cont(&ctx, a);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_CONT), r.op);
    // Unlike reshape, cont produces fresh contiguous storage.
    try std.testing.expect(!types.ggml_is_view(r));
    try std.testing.expect(types.ggml_is_contiguous(r));

    const reshaped = ggml_cont_2d(&ctx, a, 4, 8);
    try std.testing.expectEqual(@as(i64, 4), reshaped.ne[0]);
    try std.testing.expect(types.ggml_is_contiguous(reshaped));
}

test "views take their strides explicitly" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    // A 2-row window starting one row in.
    const v = ggml_view_2d(&ctx, a, 8, 2, a.nb[1], a.nb[1]);
    try std.testing.expectEqual(@as(i64, 8), v.ne[0]);
    try std.testing.expectEqual(@as(i64, 2), v.ne[1]);
    try std.testing.expectEqual(a.nb[1], v.nb[1]);
    try std.testing.expectEqual(@as(usize, 32), v.view_offs);
    // The offset is recorded for the backend, as a full size_t.
    try std.testing.expectEqual(@as(usize, 32), std.mem.bytesAsValue(usize, std.mem.asBytes(&v.op_params)[0..@sizeOf(usize)]).*);

    // A strided view can skip: every other row.
    const strided = ggml_view_2d(&ctx, a, 8, 2, a.nb[1] * 2, 0);
    try std.testing.expectEqual(a.nb[1] * 2, strided.nb[1]);
    try std.testing.expect(!types.ggml_is_contiguous(strided));
}

test "permute moves each dimension to the axis it names" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 2, 3, 4, 5);

    // Send dim0 -> axis1, dim1 -> axis2, dim2 -> axis0, dim3 -> axis3.
    const p = ggml_permute(&ctx, a, 1, 2, 0, 3);
    try std.testing.expectEqual(a.ne[2], p.ne[0]);
    try std.testing.expectEqual(a.ne[0], p.ne[1]);
    try std.testing.expectEqual(a.ne[1], p.ne[2]);
    try std.testing.expectEqual(a.ne[3], p.ne[3]);
    // Strides travel with their dimensions.
    try std.testing.expectEqual(a.nb[2], p.nb[0]);
    try std.testing.expectEqual(a.nb[0], p.nb[1]);
    try std.testing.expect(types.ggml_is_view(p));

    // The identity permutation must change nothing.
    const same = ggml_permute(&ctx, a, 0, 1, 2, 3);
    for (0..c.GGML_MAX_DIMS) |i| {
        try std.testing.expectEqual(a.ne[i], same.ne[i]);
        try std.testing.expectEqual(a.nb[i], same.nb[i]);
    }
}

test "transpose swaps the first two dimensions without moving data" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const t = ggml_transpose(&ctx, a);
    try std.testing.expectEqual(@as(i64, 4), t.ne[0]);
    try std.testing.expectEqual(@as(i64, 8), t.ne[1]);
    try std.testing.expectEqual(a.nb[1], t.nb[0]);
    try std.testing.expectEqual(a.nb[0], t.nb[1]);
    // nb[0] > nb[1] is exactly what ggml_is_transposed looks for.
    try std.testing.expect(types.ggml_is_transposed(t));
    try std.testing.expect(!types.ggml_is_contiguous(t));
    // Transposing twice returns the original layout.
    const tt = ggml_transpose(&ctx, t);
    try std.testing.expectEqual(a.ne[0], tt.ne[0]);
    try std.testing.expectEqual(a.nb[0], tt.nb[0]);
}

test "cast keeps shape, changes type, and self-references src[1]" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);

    const r = ggml_cast(&ctx, a, c.GGML_TYPE_F16);
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F16), r.type);
    try std.testing.expect(types.ggml_are_same_shape(r, a));
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_CPY), r.op);
    // Deliberate in the C: some backends read src[1] as the destination.
    try std.testing.expectEqual(@as(?*Tensor, r), r.src[1]);
}

test "set records strides and rejects an offset that would not fit i32" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 64, 4);
    const b = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 8);

    const r = ggml_set_1d(&ctx, a, b, 16);
    try std.testing.expectEqual(@as(i32, @intCast(a.nb[1])), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i32, 16), impl.getOpParamsI32(r, 3));
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(r, 4));
    try std.testing.expect(types.ggml_is_view(ggml_set_1d_inplace(&ctx, a, b, 16)));
}

// -----------------------------------------------------------------------------
// Row gathering
//
// The embedding lookup and its inverse: `get_rows` indexes a table, `set_rows`
// scatters back into one.

/// Ports `ggml_get_rows` (ggml.c:3891 @c1d0e7a00).
///
/// Indexes rows of `a` by the i32 indices in `b`. This is how token embeddings
/// are looked up.
///
/// The result is f32 whatever `a`'s type, so a quantized embedding table is
/// dequantized on the way out. The one exception is an i32 table, which stays
/// i32 -- the C notes non-f32 returns are otherwise unimplemented.
pub export fn ggml_get_rows(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(a.ne[2] == b.ne[1], "a->ne[2] == b->ne[1]");
    impl.assert(a.ne[3] == b.ne[2], "a->ne[3] == b->ne[2]");
    impl.assert(b.ne[3] == 1, "b->ne[3] == 1");
    impl.assert(b.type == c.GGML_TYPE_I32, "b->type == GGML_TYPE_I32");

    const t: c.enum_ggml_type = if (a.type == c.GGML_TYPE_I32) a.type else c.GGML_TYPE_F32;
    const result = context.ggml_new_tensor_4d(ctx, t, a.ne[0], b.ne[0], b.ne[1], b.ne[2]);

    result.op = c.GGML_OP_GET_ROWS;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_get_rows_back` (ggml.c:3916 @c1d0e7a00).
///
/// The gradient of `get_rows`: scatters rows back to where they came from.
/// `c` supplies the output shape only and is not a source, matching the C.
pub export fn ggml_get_rows_back(ctx: *Context, a: *Tensor, b: *Tensor, shape: *Tensor) *Tensor {
    impl.assert(
        types.ggml_is_matrix(a) and types.ggml_is_vector(b) and b.type == c.GGML_TYPE_I32,
        "ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32",
    );
    impl.assert(
        types.ggml_is_matrix(shape) and a.ne[0] == shape.ne[0],
        "ggml_is_matrix(c) && (a->ne[0] == c->ne[0])",
    );

    const result = context.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, shape.ne[0], shape.ne[1]);
    result.op = c.GGML_OP_GET_ROWS_BACK;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_set_rows` (ggml.c:3937 @c1d0e7a00).
///
/// Writes rows of `b` into `a` at the indices in `idx`. Used to fill the KV
/// cache in place.
///
/// Note the source order: `src[0]` is the data, `src[1]` the indices, and
/// `src[2]` the destination. The C flags this as weird and legacy, and it must
/// be preserved because the backends read those slots positionally.
pub export fn ggml_set_rows(ctx: *Context, a: *Tensor, b: *Tensor, idx: *Tensor) *Tensor {
    impl.assert(a.ne[0] == b.ne[0], "a->ne[0] == b->ne[0]");
    impl.assert(a.ne[2] == b.ne[2], "a->ne[2] == b->ne[2]");
    impl.assert(a.ne[3] == b.ne[3], "a->ne[3] == b->ne[3]");
    impl.assert(b.ne[1] == idx.ne[0], "b->ne[1] == c->ne[0]");
    impl.assert(@rem(b.ne[2], idx.ne[1]) == 0, "b->ne[2] % c->ne[1] == 0");
    impl.assert(@rem(b.ne[3], idx.ne[2]) == 0, "b->ne[3] % c->ne[2] == 0");
    impl.assert(idx.ne[3] == 1, "c->ne[3] == 1");
    impl.assert(
        b.type == c.GGML_TYPE_F32 or b.type == c.GGML_TYPE_F16,
        "b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16",
    );
    impl.assert(
        idx.type == c.GGML_TYPE_I64 or idx.type == c.GGML_TYPE_I32,
        "c->type == GGML_TYPE_I64 || c->type == GGML_TYPE_I32",
    );
    impl.assert(types.ggml_is_contiguous_rows(a), "ggml_is_contiguous_rows(a)");
    impl.assert(types.ggml_is_contiguous_rows(b), "ggml_is_contiguous_rows(b)");

    const result = context.ggml_view_tensor(ctx, a);
    result.op = c.GGML_OP_SET_ROWS;
    result.src[0] = b;
    result.src[1] = idx;
    result.src[2] = a;
    return result;
}

// -----------------------------------------------------------------------------
// Diagonals and masking

/// Ports `ggml_diag` (ggml.c:3967 @c1d0e7a00).
///
/// Turns a row vector into a square matrix with that vector on the diagonal.
pub export fn ggml_diag(ctx: *Context, a: *Tensor) *Tensor {
    impl.assert(a.ne[1] == 1, "a->ne[1] == 1");

    const ne = [_]i64{ a.ne[0], a.ne[0], a.ne[2], a.ne[3] };
    const result = context.ggml_new_tensor(ctx, a.type, 4, &ne);
    result.op = c.GGML_OP_DIAG;
    result.src[0] = a;
    return result;
}

/// Builds a causal-masking wrapper. `n_past` is how many positions before the
/// current window are always visible.
fn diagMaskWrapper(
    comptime op: c.enum_ggml_op,
    comptime inplace: bool,
) fn (*Context, *Tensor, c_int) callconv(.c) *Tensor {
    return struct {
        fn f(ctx: *Context, a: *Tensor, n_past: c_int) callconv(.c) *Tensor {
            const result = dupOrView(ctx, a, inplace);
            impl.setOpParamsValue(result, [_]i32{n_past});
            result.op = op;
            result.src[0] = a;
            return result;
        }
    }.f;
}

/// The masking wrappers, from ggml.c lines 4001-4066. `_INF` masks with
/// negative infinity so the positions vanish under softmax; `_ZERO` masks with
/// zero.
const diag_masks = .{
    .{ "ggml_diag_mask_inf", c.GGML_OP_DIAG_MASK_INF, false },
    .{ "ggml_diag_mask_inf_inplace", c.GGML_OP_DIAG_MASK_INF, true },
    .{ "ggml_diag_mask_zero", c.GGML_OP_DIAG_MASK_ZERO, false },
    .{ "ggml_diag_mask_zero_inplace", c.GGML_OP_DIAG_MASK_ZERO, true },
};

comptime {
    for (diag_masks) |entry| {
        @export(&diagMaskWrapper(entry[1], entry[2]), .{ .name = entry[0] });
    }
}

/// Ports `ggml_diag_mask_zero_impl` (ggml.c:4015 @c1d0e7a00) with `inplace` false.
///
/// The exported wrappers are generated above and so have no Zig-visible name;
/// this is the one variant the autodiff pass calls directly.
pub const diagMaskZero = diagMaskWrapper(c.GGML_OP_DIAG_MASK_ZERO, false);

// -----------------------------------------------------------------------------
// Clamping

/// Ports `ggml_clamp_impl` (ggml.c:4047 @c1d0e7a00).
fn clampImpl(ctx: *Context, a: *Tensor, min: f32, max: f32, inplace: bool) *Tensor {
    const result = dupOrView(ctx, a, inplace);
    impl.setOpParamsValue(result, [_]f32{ min, max });
    result.op = c.GGML_OP_CLAMP;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_clamp` (ggml.c:4064 @c1d0e7a00).
pub export fn ggml_clamp(ctx: *Context, a: *Tensor, min: f32, max: f32) *Tensor {
    return clampImpl(ctx, a, min, max, false);
}

/// Ports `ggml_clamp_inplace` (ggml.c:4072 @c1d0e7a00).
pub export fn ggml_clamp_inplace(ctx: *Context, a: *Tensor, min: f32, max: f32) *Tensor {
    return clampImpl(ctx, a, min, max, true);
}

// -----------------------------------------------------------------------------
// Softmax
//
// The attention softmax. `scale` divides the logits and `max_bias` drives ALiBi
// position bias; both live in `op_params`, and a mask can be supplied as a
// second source.

/// Ports `ggml_soft_max_impl` (ggml.c:4082 @c1d0e7a00).
fn softMaxImpl(
    ctx: *Context,
    a: *Tensor,
    mask: ?*Tensor,
    scale: f32,
    max_bias: f32,
    inplace: bool,
) *Tensor {
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");

    if (mask) |m| {
        impl.assert(
            m.type == c.GGML_TYPE_F16 or m.type == c.GGML_TYPE_F32,
            "mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32",
        );
        impl.assert(types.ggml_is_contiguous(m), "ggml_is_contiguous(mask)");
        impl.assert(m.ne[0] == a.ne[0], "mask->ne[0] == a->ne[0]");
        // The mask may be longer than the query, which is how a shared causal
        // mask is reused across shorter batches.
        impl.assert(m.ne[1] >= a.ne[1], "mask->ne[1] >= a->ne[1]");
        impl.assert(@rem(a.ne[2], m.ne[2]) == 0, "a->ne[2]%mask->ne[2] == 0");
        impl.assert(@rem(a.ne[3], m.ne[3]) == 0, "a->ne[3]%mask->ne[3] == 0");
    }

    // ALiBi needs a mask to know the positions it is biasing.
    if (max_bias > 0.0) impl.assert(mask != null, "mask");

    const result = dupOrView(ctx, a, inplace);
    impl.setOpParamsValue(result, [_]f32{ scale, max_bias });

    result.op = c.GGML_OP_SOFT_MAX;
    result.src[0] = a;
    result.src[1] = mask;
    return result;
}

/// Ports `ggml_soft_max` (ggml.c:4116 @c1d0e7a00).
pub export fn ggml_soft_max(ctx: *Context, a: *Tensor) *Tensor {
    return softMaxImpl(ctx, a, null, 1.0, 0.0, false);
}

/// Ports `ggml_soft_max_inplace` (ggml.c:4122 @c1d0e7a00).
pub export fn ggml_soft_max_inplace(ctx: *Context, a: *Tensor) *Tensor {
    return softMaxImpl(ctx, a, null, 1.0, 0.0, true);
}

/// Ports `ggml_soft_max_ext` (ggml.c:4128 @c1d0e7a00).
pub export fn ggml_soft_max_ext(ctx: *Context, a: *Tensor, mask: ?*Tensor, scale: f32, max_bias: f32) *Tensor {
    return softMaxImpl(ctx, a, mask, scale, max_bias, false);
}

/// Ports `ggml_soft_max_ext_inplace` (ggml.c:4137 @c1d0e7a00).
pub export fn ggml_soft_max_ext_inplace(ctx: *Context, a: *Tensor, mask: ?*Tensor, scale: f32, max_bias: f32) *Tensor {
    return softMaxImpl(ctx, a, mask, scale, max_bias, true);
}

/// Ports `ggml_soft_max_add_sinks` (ggml.c:4146 @c1d0e7a00).
///
/// Attaches attention sinks to an existing softmax node, in `src[2]`. Mutates
/// rather than building, so it must be called on a node that is already a
/// softmax and does not already have sinks.
///
/// Passing null clears the slot and skips the checks, as the C does.
pub export fn ggml_soft_max_add_sinks(a: *Tensor, sinks: ?*Tensor) void {
    const s = sinks orelse {
        a.src[2] = null;
        return;
    };

    impl.assert(a.op == c.GGML_OP_SOFT_MAX, "a->op == GGML_OP_SOFT_MAX");
    impl.assert(a.src[2] == null, "a->src[2] == NULL");
    impl.assert(impl.one(c.ggml_tensor, a.src[0]).ne[2] == s.ne[0], "a->src[0]->ne[2] == sinks->ne[0]");
    impl.assert(s.type == c.GGML_TYPE_F32, "sinks->type == GGML_TYPE_F32");

    a.src[2] = s;
}

/// Ports `ggml_soft_max_ext_back_impl` (ggml.c:4164 @c1d0e7a00).
fn softMaxBackImpl(ctx: *Context, a: *Tensor, b: *Tensor, scale: f32, max_bias: f32, inplace: bool) *Tensor {
    const result = dupOrView(ctx, a, inplace);

    result.op = c.GGML_OP_SOFT_MAX_BACK;
    result.src[0] = a;
    result.src[1] = b;

    // The C writes these after setting the op, via two memcpys rather than
    // ggml_set_op_params. Same bytes either way.
    impl.setOpParamsF32(result, 0, scale);
    impl.setOpParamsF32(result, 1, max_bias);
    return result;
}

/// Ports `ggml_soft_max_ext_back` (ggml.c:4183 @c1d0e7a00).
pub export fn ggml_soft_max_ext_back(ctx: *Context, a: *Tensor, b: *Tensor, scale: f32, max_bias: f32) *Tensor {
    return softMaxBackImpl(ctx, a, b, scale, max_bias, false);
}

/// Ports `ggml_soft_max_ext_back_inplace` (ggml.c:4192 @c1d0e7a00).
pub export fn ggml_soft_max_ext_back_inplace(ctx: *Context, a: *Tensor, b: *Tensor, scale: f32, max_bias: f32) *Tensor {
    return softMaxBackImpl(ctx, a, b, scale, max_bias, true);
}

// -----------------------------------------------------------------------------
// Unit Tests for gathering, masking, and softmax

extern fn ggml_diag_mask_inf(ctx: *Context, a: *Tensor, n_past: c_int) *Tensor;
extern fn ggml_diag_mask_zero(ctx: *Context, a: *Tensor, n_past: c_int) *Tensor;
extern fn ggml_diag_mask_inf_inplace(ctx: *Context, a: *Tensor, n_past: c_int) *Tensor;

test "get_rows dequantizes to f32 but keeps i32 tables" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // A quantized embedding table indexed by token ids.
    const table = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_Q4_K, 256, 100);
    const ids = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_I32, 7);

    const r = ggml_get_rows(&ctx, table, ids);
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F32), r.type);
    try std.testing.expectEqual(@as(i64, 256), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 7), r.ne[1]);

    // An i32 table is the one case that keeps its type.
    const i32_table = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_I32, 8, 100);
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_I32), ggml_get_rows(&ctx, i32_table, ids).type);
}

test "set_rows keeps the legacy source order" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const dst = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 64, 100);
    const data = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 64, 4);
    const idx = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_I64, 4);

    const r = ggml_set_rows(&ctx, dst, data, idx);
    // src[0]=data, src[1]=indices, src[2]=destination. Backends read these
    // positionally, so the order is load-bearing.
    try std.testing.expectEqual(@as(?*Tensor, data), r.src[0]);
    try std.testing.expectEqual(@as(?*Tensor, idx), r.src[1]);
    try std.testing.expectEqual(@as(?*Tensor, dst), r.src[2]);
    // Writes in place, so the result views the destination.
    try std.testing.expect(types.ggml_is_view(r));
    try std.testing.expectEqual(@as(?*Tensor, dst), r.view_src);
}

test "diag squares a row vector" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const v = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 8);

    const r = ggml_diag(&ctx, v);
    try std.testing.expectEqual(@as(i64, 8), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[1]);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_DIAG), r.op);
}

test "diag masks record n_past and differ only in op" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 8);

    const inf = ggml_diag_mask_inf(&ctx, a, 3);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_DIAG_MASK_INF), inf.op);
    try std.testing.expectEqual(@as(i32, 3), impl.getOpParamsI32(inf, 0));

    const zero = ggml_diag_mask_zero(&ctx, a, 3);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_DIAG_MASK_ZERO), zero.op);
    try std.testing.expectEqual(@as(i32, 3), impl.getOpParamsI32(zero, 0));

    try std.testing.expect(types.ggml_is_view(ggml_diag_mask_inf_inplace(&ctx, a, 0)));
}

test "clamp stores both bounds" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 8);

    const r = ggml_clamp(&ctx, a, -1.5, 2.5);
    try std.testing.expectEqual(@as(f32, -1.5), impl.getOpParamsF32(r, 0));
    try std.testing.expectEqual(@as(f32, 2.5), impl.getOpParamsF32(r, 1));
    try std.testing.expect(types.ggml_is_view(ggml_clamp_inplace(&ctx, a, 0, 1)));
}

test "softmax carries scale and mask" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    // Attention logits [n_kv, n_q, n_head].
    const logits = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 64, 8, 4);

    const plain = ggml_soft_max(&ctx, logits);
    try std.testing.expectEqual(@as(f32, 1.0), impl.getOpParamsF32(plain, 0));
    try std.testing.expectEqual(@as(f32, 0.0), impl.getOpParamsF32(plain, 1));
    try std.testing.expect(plain.src[1] == null);

    // A mask longer than the query is allowed: a shared causal mask reused
    // across shorter batches.
    const mask = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F16, 64, 32);
    const ext = ggml_soft_max_ext(&ctx, logits, mask, 0.125, 8.0);
    try std.testing.expectEqual(@as(f32, 0.125), impl.getOpParamsF32(ext, 0));
    try std.testing.expectEqual(@as(f32, 8.0), impl.getOpParamsF32(ext, 1));
    try std.testing.expectEqual(@as(?*Tensor, mask), ext.src[1]);
    try std.testing.expect(types.ggml_are_same_shape(ext, logits));
}

test "attention sinks attach to src[2] and clear" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const logits = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 64, 8, 4);
    const sm = ggml_soft_max(&ctx, logits);

    // One sink per head: sinks->ne[0] must match src[0]->ne[2].
    const sinks = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 4);
    ggml_soft_max_add_sinks(sm, sinks);
    try std.testing.expectEqual(@as(?*Tensor, sinks), sm.src[2]);

    // Null clears without checking, so it works on any node.
    ggml_soft_max_add_sinks(sm, null);
    try std.testing.expect(sm.src[2] == null);
}

// -----------------------------------------------------------------------------
// Rotary position embedding
//
// RoPE rotates each pair of dimensions by an angle derived from the token's
// position. Its configuration is unusually large -- fourteen values -- so
// `op_params` is filled as a fixed 16-slot layout that the backends index
// positionally. That layout is reproduced exactly below; a slot in the wrong
// place produces plausible-looking garbage rather than a crash.

/// The `op_params` layout shared by `GGML_OP_ROPE` and `GGML_OP_ROPE_BACK`,
/// from `ggml_rope_impl` (ggml.c:4203 @c1d0e7a00).
///
/// Slots 0 and 3 are `n_past` and `n_ctx`, both retired but still occupying
/// their positions. Slot 15 is set later by `ggml_rope_set_offset`.
const RopeParams = extern struct {
    n_past: i32 = 0,
    n_dims: i32,
    mode: i32,
    n_ctx: i32 = 0,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    sections: [c.GGML_MROPE_SECTIONS]i32,
    n_offs: i32 = 0,

    comptime {
        // The C builds this as int32_t params[16]; if the struct is any other
        // size the slot indices below have drifted.
        std.debug.assert(@sizeOf(RopeParams) == 16 * @sizeOf(i32));
    }
};

/// Ports `ggml_rope_impl` (ggml.c:4203 @c1d0e7a00).
///
/// Parameters:
/// - `a`: the tensor to rotate.
/// - `b`: i32 vector of positions, one per token, or four per token for mrope.
/// - `freqs`: optional per-dimension frequency factors (`c` in the C).
/// - `sections`: mrope section widths; ignored unless the mode selects mrope.
///
/// Return: the rotated tensor, sharing `a`'s shape.
fn ropeImpl(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    freqs: ?*Tensor,
    n_dims: c_int,
    sections: ?[*]const c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    inplace: bool,
) *Tensor {
    impl.assert((mode & 1) == 0, "mode & 1 == 1 is no longer supported");
    impl.assert(types.ggml_is_vector(b), "ggml_is_vector(b)");
    impl.assert(b.type == c.GGML_TYPE_I32, "b->type == GGML_TYPE_I32");

    const mrope_used = (mode & c.GGML_ROPE_TYPE_MROPE) != 0;
    if (mrope_used) {
        // mrope carries four position ids per token: time plus three spatial.
        impl.assert(a.ne[2] * 4 == b.ne[0], "a->ne[2] * 4 == b->ne[0]");
    } else {
        impl.assert(a.ne[2] == b.ne[0], "a->ne[2] == b->ne[0]");
    }

    if (freqs) |f| {
        impl.assert(f.type == c.GGML_TYPE_F32, "c->type == GGML_TYPE_F32");
        // One factor per rotated pair.
        impl.assert(f.ne[0] >= @divTrunc(n_dims, 2), "c->ne[0] >= n_dims / 2");
    }

    const result = dupOrView(ctx, a, inplace);

    var params: RopeParams = .{
        .n_dims = n_dims,
        .mode = mode,
        .n_ctx_orig = n_ctx_orig,
        .freq_base = freq_base,
        .freq_scale = freq_scale,
        .ext_factor = ext_factor,
        .attn_factor = attn_factor,
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .sections = @splat(0),
    };
    // Sections are only meaningful for mrope; otherwise the slots stay zero.
    if (mrope_used) {
        if (sections) |s| {
            for (0..c.GGML_MROPE_SECTIONS) |i| params.sections[i] = s[i];
        }
    }
    impl.setOpParamsValue(result, params);

    result.op = c.GGML_OP_ROPE;
    result.src[0] = a;
    result.src[1] = b;
    result.src[2] = freqs;
    return result;
}

/// Ports `ggml_rope` (ggml.c:4262 @c1d0e7a00).
///
/// The plain form, with the defaults the original paper used.
pub export fn ggml_rope(ctx: *Context, a: *Tensor, b: *Tensor, n_dims: c_int, mode: c_int) *Tensor {
    return ropeImpl(ctx, a, b, null, n_dims, null, mode, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0, false);
}

/// Ports `ggml_rope_inplace` (ggml.c:4315 @c1d0e7a00).
pub export fn ggml_rope_inplace(ctx: *Context, a: *Tensor, b: *Tensor, n_dims: c_int, mode: c_int) *Tensor {
    return ropeImpl(ctx, a, b, null, n_dims, null, mode, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0, true);
}

/// Ports `ggml_rope_ext` (ggml.c:4326 @c1d0e7a00).
///
/// The form models actually use: every scaling knob exposed, plus optional
/// per-dimension frequency factors for YaRN-style context extension.
pub export fn ggml_rope_ext(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    freqs: ?*Tensor,
    n_dims: c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    return ropeImpl(ctx, a, b, freqs, n_dims, null, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, false);
}

/// Ports `ggml_rope_ext_inplace` (ggml.c:4346 @c1d0e7a00).
pub export fn ggml_rope_ext_inplace(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    freqs: ?*Tensor,
    n_dims: c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    return ropeImpl(ctx, a, b, freqs, n_dims, null, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, true);
}

/// Ports `ggml_rope_multi` (ggml.c:4273 @c1d0e7a00).
///
/// Multimodal rope: the head dimension is split into sections rotated by
/// different position components, so image patches carry spatial position
/// alongside sequence position.
pub export fn ggml_rope_multi(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    freqs: ?*Tensor,
    n_dims: c_int,
    sections: ?[*]const c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    return ropeImpl(ctx, a, b, freqs, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, false);
}

/// Ports `ggml_rope_multi_inplace` (ggml.c:4294 @c1d0e7a00).
pub export fn ggml_rope_multi_inplace(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    freqs: ?*Tensor,
    n_dims: c_int,
    sections: ?[*]const c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    return ropeImpl(ctx, a, b, freqs, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, true);
}

/// Ports `ggml_rope_custom` (ggml.c:4366 @c1d0e7a00).
///
/// Deprecated alias kept for the C ABI: `ggml_rope_ext` without frequency
/// factors.
pub export fn ggml_rope_custom(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    n_dims: c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    return ropeImpl(ctx, a, b, null, n_dims, null, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, false);
}

/// Ports `ggml_rope_custom_inplace` (ggml.c:4385 @c1d0e7a00).
pub export fn ggml_rope_custom_inplace(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    n_dims: c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    return ropeImpl(ctx, a, b, null, n_dims, null, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, true);
}

/// Ports `ggml_rope_ext_back` (ggml.c:4422 @c1d0e7a00).
///
/// Builds the forward node and flips the op, as the C does: the shape and
/// parameters are identical, only the direction differs.
pub export fn ggml_rope_ext_back(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    freqs: ?*Tensor,
    n_dims: c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    const result = ggml_rope_ext(ctx, a, b, freqs, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result.op = c.GGML_OP_ROPE_BACK;
    return result;
}

/// Ports `ggml_rope_multi_back` (ggml.c:4442 @c1d0e7a00).
pub export fn ggml_rope_multi_back(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    freqs: ?*Tensor,
    n_dims: c_int,
    sections: ?[*]const c_int,
    mode: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) *Tensor {
    const result = ggml_rope_multi(ctx, a, b, freqs, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result.op = c.GGML_OP_ROPE_BACK;
    return result;
}

/// Ports `ggml_rope_set_offset` (ggml.c:4463 @c1d0e7a00).
///
/// Sets the position offset on an existing rope node, in the last params slot.
/// Vision mode has no offset, and the C rejects it.
pub export fn ggml_rope_set_offset(a: *Tensor, n_offs: c_int) *Tensor {
    impl.assert(
        a.op == c.GGML_OP_ROPE or a.op == c.GGML_OP_ROPE_BACK,
        "a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK",
    );
    impl.assert(n_offs >= 0, "n_offs >= 0");

    const mode = impl.getOpParamsI32(a, 2);
    impl.assert(mode != c.GGML_ROPE_TYPE_VISION, "mode != GGML_ROPE_TYPE_VISION");

    impl.setOpParamsI32(a, 15, n_offs);
    return a;
}

/// Ports `ggml_rope_yarn_corr_dim` (ggml.c:4406 @c1d0e7a00).
fn ropeYarnCorrDim(n_dims: c_int, n_ctx_orig: c_int, n_rot: f32, base: f32) f32 {
    const dims: f32 = @floatFromInt(n_dims);
    const orig: f32 = @floatFromInt(n_ctx_orig);
    return dims * @log(orig / (n_rot * 2 * std.math.pi)) / (2 * @log(base));
}

/// Ports `ggml_rope_yarn_corr_dims` (ggml.c:4410 @c1d0e7a00).
///
/// The dimension range over which YaRN blends between interpolated and
/// extrapolated frequencies. Below `dims[0]` the frequency is extrapolated,
/// above `dims[1]` interpolated, and between them the two are mixed.
///
/// Parameters:
/// - `dims`: two floats, written as start and end.
pub export fn ggml_rope_yarn_corr_dims(
    n_dims: c_int,
    n_ctx_orig: c_int,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
    dims: *[2]f32,
) void {
    const start = @floor(ropeYarnCorrDim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const end = @ceil(ropeYarnCorrDim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = @max(0, start);
    dims[1] = @min(@as(f32, @floatFromInt(n_dims - 1)), end);
}

// -----------------------------------------------------------------------------
// Unit Tests for rope

test "rope fills the sixteen-slot parameter layout" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // [head_dim, n_head, n_tokens]
    const q = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 128, 32, 6);
    const pos = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_I32, 6);

    const r = ggml_rope(&ctx, q, pos, 128, c.GGML_ROPE_TYPE_NEOX);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_ROPE), r.op);
    try std.testing.expect(types.ggml_are_same_shape(r, q));

    // Slot order is what the backends index by, so check it positionally.
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(r, 0)); // n_past, retired
    try std.testing.expectEqual(@as(i32, 128), impl.getOpParamsI32(r, 1)); // n_dims
    try std.testing.expectEqual(@as(i32, c.GGML_ROPE_TYPE_NEOX), impl.getOpParamsI32(r, 2));
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(r, 3)); // n_ctx, retired
    try std.testing.expectEqual(@as(f32, 10000.0), impl.getOpParamsF32(r, 5));
    try std.testing.expectEqual(@as(f32, 1.0), impl.getOpParamsF32(r, 6)); // freq_scale
    try std.testing.expectEqual(@as(f32, 1.0), impl.getOpParamsF32(r, 8)); // attn_factor
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(r, 15)); // n_offs
}

test "rope_ext records every scaling knob" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const q = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 128, 32, 6);
    const pos = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_I32, 6);
    // One frequency factor per rotated pair.
    const freqs = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 64);

    const r = ggml_rope_ext(&ctx, q, pos, freqs, 128, c.GGML_ROPE_TYPE_NEOX, 4096, 500000.0, 0.5, 1.0, 0.75, 32.0, 1.0);
    try std.testing.expectEqual(@as(i32, 4096), impl.getOpParamsI32(r, 4)); // n_ctx_orig
    try std.testing.expectEqual(@as(f32, 500000.0), impl.getOpParamsF32(r, 5));
    try std.testing.expectEqual(@as(f32, 0.5), impl.getOpParamsF32(r, 6));
    try std.testing.expectEqual(@as(f32, 1.0), impl.getOpParamsF32(r, 7)); // ext_factor
    try std.testing.expectEqual(@as(f32, 0.75), impl.getOpParamsF32(r, 8));
    try std.testing.expectEqual(@as(f32, 32.0), impl.getOpParamsF32(r, 9)); // beta_fast
    try std.testing.expectEqual(@as(f32, 1.0), impl.getOpParamsF32(r, 10)); // beta_slow
    try std.testing.expectEqual(@as(?*Tensor, freqs), r.src[2]);
}

test "mrope carries four positions per token and its sections" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const q = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 128, 32, 6);
    // Four ids per token: time plus three spatial.
    const pos = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_I32, 24);
    const sections = [_]c_int{ 16, 24, 24, 0 };

    const r = ggml_rope_multi(&ctx, q, pos, null, 128, &sections, c.GGML_ROPE_TYPE_MROPE, 4096, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    for (sections, 0..) |want, i| {
        try std.testing.expectEqual(@as(i32, want), impl.getOpParamsI32(r, 11 + i));
    }

    // Without an mrope mode the section slots must stay zero, even if given.
    const plain = ggml_rope_multi(&ctx, q, context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_I32, 6), null, 128, &sections, c.GGML_ROPE_TYPE_NEOX, 4096, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    for (0..c.GGML_MROPE_SECTIONS) |i| {
        try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(plain, 11 + i));
    }
}

test "rope_set_offset writes the last slot without disturbing the rest" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const q = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 128, 32, 6);
    const pos = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_I32, 6);

    const r = ggml_rope(&ctx, q, pos, 128, c.GGML_ROPE_TYPE_NEOX);
    _ = ggml_rope_set_offset(r, 17);
    try std.testing.expectEqual(@as(i32, 17), impl.getOpParamsI32(r, 15));
    try std.testing.expectEqual(@as(i32, 128), impl.getOpParamsI32(r, 1));
    try std.testing.expectEqual(@as(f32, 10000.0), impl.getOpParamsF32(r, 5));
}

test "rope back flips the op but keeps the parameters" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const q = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 128, 32, 6);
    const pos = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_I32, 6);

    const r = ggml_rope_ext_back(&ctx, q, pos, null, 128, c.GGML_ROPE_TYPE_NEOX, 4096, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_ROPE_BACK), r.op);
    try std.testing.expectEqual(@as(i32, 128), impl.getOpParamsI32(r, 1));
    // set_offset accepts ROPE_BACK too.
    _ = ggml_rope_set_offset(r, 3);
    try std.testing.expectEqual(@as(i32, 3), impl.getOpParamsI32(r, 15));
}

test "yarn correction dims stay inside the dimension range" {
    var dims: [2]f32 = undefined;
    ggml_rope_yarn_corr_dims(128, 4096, 10000.0, 32.0, 1.0, &dims);

    // Clamped to [0, n_dims-1], and start must not exceed end.
    try std.testing.expect(dims[0] >= 0);
    try std.testing.expect(dims[1] <= 127);
    try std.testing.expect(dims[0] <= dims[1]);

    // A tiny context drives the correction dim negative, which must clamp to 0.
    ggml_rope_yarn_corr_dims(128, 1, 10000.0, 32.0, 1.0, &dims);
    try std.testing.expectEqual(@as(f32, 0), dims[0]);
}

// -----------------------------------------------------------------------------
// Convolution
//
// Most of these are composites: they build a subgraph out of ops already
// defined above rather than introducing an op code of their own. `ggml_conv_1d`
// for instance is im2col, then a matmul, then two reshapes. Only `im2col`,
// `col2im_1d`, `conv_transpose_1d` and the direct variants are real op codes.

/// Ports `ggml_calc_conv_output_size` (ggml.c:4476 @c1d0e7a00).
fn convOutputSize(ins: i64, ks: i64, s: c_int, p: c_int, d: c_int) i64 {
    return @divTrunc(ins + 2 * p - @as(i64, d) * (ks - 1) - 1, @as(i64, s)) + 1;
}

/// Ports `ggml_calc_conv_transpose_1d_output_size` (ggml.c:4651 @c1d0e7a00).
fn convTranspose1dOutputSize(ins: i64, ks: i64, s: c_int, p: c_int, d: c_int) i64 {
    return (ins - 1) * s - 2 * p + @as(i64, d) * (ks - 1) + 1;
}

/// The intermediate type im2col produces for a convolution.
///
/// f16 normally, but bf16 kernels go through f32: there is no bf16 matmul path
/// to feed. Mirrors the ternary the C repeats at every conv site.
fn im2colType(a: *const Tensor) c.enum_ggml_type {
    return if (a.type == c.GGML_TYPE_BF16) c.GGML_TYPE_F32 else c.GGML_TYPE_F16;
}

/// Ports `ggml_im2col` (ggml.c:4484 @c1d0e7a00).
///
/// Rearranges overlapping patches into rows so a convolution becomes a matrix
/// multiply:
///
///     a (kernel) [OC, IC, KH, KW], b (image) [N, IC, IH, IW]
///     result                       [N, OH, OW, IC*KH*KW]
///
/// Parameters:
/// - `s0`, `s1`: stride, `p0`, `p1`: padding, `d0`, `d1`: dilation.
/// - `is_2D`: false collapses the vertical dimension for 1-D convolution.
/// - `dst_type`: the intermediate type, chosen by the caller.
pub export fn ggml_im2col(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    s0: c_int,
    s1: c_int,
    p0: c_int,
    p1: c_int,
    d0: c_int,
    d1: c_int,
    is_2D: bool,
    dst_type: c.enum_ggml_type,
) *Tensor {
    if (is_2D) {
        impl.assert(a.ne[2] == b.ne[2], "a->ne[2] == b->ne[2]");
    } else {
        impl.assert(b.ne[1] == a.ne[1], "b->ne[1] == a->ne[1]");
        impl.assert(b.ne[3] == 1, "b->ne[3] == 1");
    }

    const oh: i64 = if (is_2D) convOutputSize(b.ne[1], a.ne[1], s1, p1, d1) else 0;
    const ow: i64 = convOutputSize(b.ne[0], a.ne[0], s0, p0, d0);

    impl.assert(!is_2D or oh > 0, "b too small compared to a");
    impl.assert(ow > 0, "b too small compared to a");

    const ne = [_]i64{
        if (is_2D) a.ne[2] * a.ne[1] * a.ne[0] else a.ne[1] * a.ne[0],
        ow,
        if (is_2D) oh else b.ne[2],
        if (is_2D) b.ne[3] else 1,
    };

    const result = context.ggml_new_tensor(ctx, dst_type, 4, &ne);
    impl.setOpParamsValue(result, [_]i32{ s0, s1, p0, p1, d0, d1, if (is_2D) 1 else 0 });

    result.op = c.GGML_OP_IM2COL;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_im2col_back` (ggml.c:4528 @c1d0e7a00).
///
/// The gradient of `im2col`. The output shape cannot be derived from the
/// inputs, so the caller supplies it.
pub export fn ggml_im2col_back(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    ne: [*]const i64,
    s0: c_int,
    s1: c_int,
    p0: c_int,
    p1: c_int,
    d0: c_int,
    d1: c_int,
    is_2D: bool,
) *Tensor {
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, ne);
    impl.setOpParamsValue(result, [_]i32{ s0, s1, p0, p1, d0, d1, if (is_2D) 1 else 0 });

    result.op = c.GGML_OP_IM2COL_BACK;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_conv_1d` (ggml.c:4553 @c1d0e7a00).
///
/// im2col, then one matmul, then a reshape back to [N, OC, OL]. No op code of
/// its own.
pub export fn ggml_conv_1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: c_int, p0: c_int, d0: c_int) *Tensor {
    // [N, OL, IC * K]
    const im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, im2colType(a));

    const result = ggml_mul_mat(
        ctx,
        // [N, OL, IC*K] => [N*OL, IC*K]
        ggml_reshape_2d(ctx, im2col, im2col.ne[0], im2col.ne[2] * im2col.ne[1]),
        // [OC, IC, K] => [OC, IC*K]
        ggml_reshape_2d(ctx, a, a.ne[0] * a.ne[1], a.ne[2]),
    );

    return ggml_reshape_3d(ctx, result, im2col.ne[1], a.ne[2], im2col.ne[2]);
}

/// Ports `ggml_conv_1d_ph` (ggml.c:4574 @c1d0e7a00).
///
/// "Padded half": padding chosen so the output length matches the input.
pub export fn ggml_conv_1d_ph(ctx: *Context, a: *Tensor, b: *Tensor, s: c_int, d: c_int) *Tensor {
    return ggml_conv_1d(ctx, a, b, s, @intCast(@divTrunc(a.ne[0], 2)), d);
}

/// Ports `ggml_conv_1d_dw` (ggml.c:4585 @c1d0e7a00).
///
/// Depthwise: each channel is convolved by its own kernel, so there is no
/// summation across channels and the matmul is smaller.
pub export fn ggml_conv_1d_dw(ctx: *Context, a: *Tensor, b: *Tensor, s0: c_int, p0: c_int, d0: c_int) *Tensor {
    const new_b = ggml_reshape_4d(ctx, b, b.ne[0], 1, b.ne[1], b.ne[2]);
    const im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, im2colType(a));
    const result = ggml_mul_mat(ctx, im2col, a);
    return ggml_reshape_3d(ctx, result, result.ne[0], result.ne[2], 1);
}

/// Ports `ggml_conv_1d_dw_ph` (ggml.c:4605 @c1d0e7a00).
pub export fn ggml_conv_1d_dw_ph(ctx: *Context, a: *Tensor, b: *Tensor, s0: c_int, d0: c_int) *Tensor {
    return ggml_conv_1d_dw(ctx, a, b, s0, @intCast(@divTrunc(a.ne[0], 2)), d0);
}

/// Ports `ggml_col2im_1d` (ggml.c:4616 @c1d0e7a00).
///
/// Folds overlapping columns back into a signal, summing where they overlap.
/// The input's first dimension packs `oc` blocks of kernel taps, so it must
/// divide evenly.
pub export fn ggml_col2im_1d(ctx: *Context, a: *Tensor, s0: c_int, oc: c_int, p0: c_int) *Tensor {
    impl.assert(types.ggml_is_matrix(a), "ggml_is_matrix(a)");
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
    impl.assert(
        a.type == c.GGML_TYPE_F32 or a.type == c.GGML_TYPE_F16 or a.type == c.GGML_TYPE_BF16,
        "a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16",
    );
    impl.assert(s0 > 0, "s0 > 0");
    impl.assert(oc > 0, "oc > 0");
    impl.assert(p0 >= 0, "p0 >= 0");

    const k_oc = a.ne[0];
    const t_in = a.ne[1];
    const k = @divTrunc(k_oc, oc);
    const t_out = (t_in - 1) * s0 + k - 2 * p0;

    impl.assert(k_oc == k * oc, "a->ne[0] must be a whole number of oc blocks");
    impl.assert(k > 0 and t_out > 0, "K > 0 && T_out > 0");

    const ne = [_]i64{ t_out, oc, 1, 1 };
    const result = context.ggml_new_tensor(ctx, a.type, 2, &ne);
    impl.setOpParamsValue(result, [_]i32{ s0, oc, p0 });

    result.op = c.GGML_OP_COL2IM_1D;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_conv_transpose_1d` (ggml.c:4655 @c1d0e7a00).
///
/// Padding and dilation are asserted away rather than supported; the C takes
/// the arguments only to keep the signature stable.
pub export fn ggml_conv_transpose_1d(ctx: *Context, a: *Tensor, b: *Tensor, s0: c_int, p0: c_int, d0: c_int) *Tensor {
    impl.assert(types.ggml_is_matrix(b), "ggml_is_matrix(b)");
    impl.assert(a.ne[2] == b.ne[1], "a->ne[2] == b->ne[1]");
    impl.assert(a.ne[3] == 1, "a->ne[3] == 1");
    impl.assert(p0 == 0, "p0 == 0");
    impl.assert(d0 == 1, "d0 == 1");

    const ne = [_]i64{
        convTranspose1dOutputSize(b.ne[0], a.ne[0], s0, 0, 1),
        a.ne[1],
        b.ne[2],
        1,
    };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    impl.setOpParamsValue(result, [_]i32{ s0, p0, d0 });

    result.op = c.GGML_OP_CONV_TRANSPOSE_1D;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests for convolution

test "im2col shape follows the convolution output formula" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // kernel [KW, KH, IC, OC] = [3, 3, 4, 8]; image [IW, IH, IC, N] = [16, 16, 4, 1]
    const kernel = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F16, 3, 3, 4, 8);
    const image = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 16, 4, 1);

    // stride 1, pad 1, dilation 1 keeps the spatial size: (16 + 2 - 2 - 1)/1 + 1 = 16.
    const r = ggml_im2col(&ctx, kernel, image, 1, 1, 1, 1, 1, 1, true, c.GGML_TYPE_F16);
    try std.testing.expectEqual(@as(i64, 3 * 3 * 4), r.ne[0]); // IC*KH*KW
    try std.testing.expectEqual(@as(i64, 16), r.ne[1]); // OW
    try std.testing.expectEqual(@as(i64, 16), r.ne[2]); // OH

    // stride 2 halves it: (16 + 2 - 2 - 1)/2 + 1 = 8.
    const strided = ggml_im2col(&ctx, kernel, image, 2, 2, 1, 1, 1, 1, true, c.GGML_TYPE_F16);
    try std.testing.expectEqual(@as(i64, 8), strided.ne[1]);
    try std.testing.expectEqual(@as(i64, 8), strided.ne[2]);

    // All six geometry values plus the 2-D flag are recorded for the backend.
    try std.testing.expectEqual(@as(i32, 2), impl.getOpParamsI32(strided, 0));
    try std.testing.expectEqual(@as(i32, 1), impl.getOpParamsI32(strided, 2));
    try std.testing.expectEqual(@as(i32, 1), impl.getOpParamsI32(strided, 6));
}

test "conv_1d builds a subgraph rather than its own op" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // kernel [K, IC, OC] = [3, 4, 8]; signal [L, IC, N] = [32, 4, 1]
    const kernel = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F16, 3, 4, 8);
    const signal = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 32, 4, 1);

    const r = ggml_conv_1d(&ctx, kernel, signal, 1, 1, 1);
    // The last node is the reshape, not a conv op: this is a composite.
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_RESHAPE), r.op);
    try std.testing.expectEqual(@as(i64, 32), r.ne[0]); // OL, padding preserves length
    try std.testing.expectEqual(@as(i64, 8), r.ne[1]); // OC

    // Padded-half picks the padding that preserves length.
    const ph = ggml_conv_1d_ph(&ctx, kernel, signal, 1, 1);
    try std.testing.expectEqual(@as(i64, 32), ph.ne[0]);
}

test "bf16 kernels go through f32 rather than f16" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const f16_kernel = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F16, 3, 4, 8);
    const bf16_kernel = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_BF16, 3, 4, 8);

    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F16), im2colType(f16_kernel));
    // There is no bf16 matmul path to feed, so the intermediate widens.
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F32), im2colType(bf16_kernel));
}

test "col2im_1d inverts the column packing" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // 24 = K(3) * oc(8) taps per column, 10 columns.
    const cols = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 24, 10);
    const r = ggml_col2im_1d(&ctx, cols, 2, 8, 1);

    // T_out = (10-1)*2 + 3 - 2*1 = 19
    try std.testing.expectEqual(@as(i64, 19), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[1]);
    try std.testing.expectEqual(@as(i32, 2), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i32, 8), impl.getOpParamsI32(r, 1));
    try std.testing.expectEqual(@as(i32, 1), impl.getOpParamsI32(r, 2));
}

test "conv_transpose_1d grows the signal" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // kernel [K, OC, IC] = [4, 8, 4]; signal [L, IC] = [10, 4]
    const kernel = context.ggml_new_tensor_3d(&ctx, c.GGML_TYPE_F32, 4, 8, 4);
    const signal = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 10, 4);

    // (10-1)*2 - 0 + 1*(4-1) + 1 = 22
    const r = ggml_conv_transpose_1d(&ctx, kernel, signal, 2, 0, 1);
    try std.testing.expectEqual(@as(i64, 22), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[1]);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_CONV_TRANSPOSE_1D), r.op);
}

test "convolution output size matches the formula" {
    // The identity every conv shape here depends on.
    try std.testing.expectEqual(@as(i64, 16), convOutputSize(16, 3, 1, 1, 1));
    try std.testing.expectEqual(@as(i64, 14), convOutputSize(16, 3, 1, 0, 1));
    try std.testing.expectEqual(@as(i64, 8), convOutputSize(16, 3, 2, 1, 1));
    // Dilation widens the effective kernel: 3 taps at dilation 2 spans 5.
    try std.testing.expectEqual(@as(i64, 12), convOutputSize(16, 3, 1, 0, 2));
    try std.testing.expectEqual(@as(i64, 22), convTranspose1dOutputSize(10, 4, 2, 0, 1));
}

// -----------------------------------------------------------------------------
// Two- and three-dimensional convolution
//
// Two families here. The `_direct` variants have op codes of their own and let
// a backend implement the convolution however it likes. Everything else is a
// composite over im2col and matmul, which is what runs when a backend has no
// direct kernel.

/// Ports `ggml_conv_2d` (ggml.c:4690 @c1d0e7a00).
///
/// The matmul leaves the result as [OC, N, OH, OW], so it ends with a permute
/// and a `cont` to reach [N, OC, OH, OW]. That copy is why the `_direct`
/// variant exists.
pub export fn ggml_conv_2d(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    s0: c_int,
    s1: c_int,
    p0: c_int,
    p1: c_int,
    d0: c_int,
    d1: c_int,
) *Tensor {
    // [N, OH, OW, IC*KH*KW]
    const im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, im2colType(a));

    var result = ggml_mul_mat(
        ctx,
        ggml_reshape_2d(ctx, im2col, im2col.ne[0], im2col.ne[3] * im2col.ne[2] * im2col.ne[1]),
        ggml_reshape_2d(ctx, a, a.ne[0] * a.ne[1] * a.ne[2], a.ne[3]),
    );

    result = ggml_reshape_4d(ctx, result, im2col.ne[1], im2col.ne[2], im2col.ne[3], a.ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
}

/// Ports `ggml_conv_2d_sk_p0` (ggml.c:4801 @c1d0e7a00).
///
/// Stride equal to the kernel, no padding: tiles the input into
/// non-overlapping patches. This is how vision transformers embed patches.
pub export fn ggml_conv_2d_sk_p0(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return ggml_conv_2d(ctx, a, b, @intCast(a.ne[0]), @intCast(a.ne[1]), 0, 0, 1, 1);
}

/// Ports `ggml_conv_2d_s1_ph` (ggml.c:4810 @c1d0e7a00).
///
/// Stride one, padding half the kernel: preserves the spatial size.
pub export fn ggml_conv_2d_s1_ph(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    return ggml_conv_2d(ctx, a, b, 1, 1, @intCast(@divTrunc(a.ne[0], 2)), @intCast(@divTrunc(a.ne[1], 2)), 1, 1);
}

/// Ports `ggml_conv_2d_dw` (ggml.c:4819 @c1d0e7a00).
///
/// Depthwise: one kernel per channel, no summation across channels. Both
/// kernel and input are reshaped to fold the channel dimension into the batch,
/// so a single grouped matmul does the work.
pub export fn ggml_conv_2d_dw(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    s0: c_int,
    s1: c_int,
    p0: c_int,
    p1: c_int,
    d0: c_int,
    d1: c_int,
) *Tensor {
    var new_a = ggml_reshape_4d(ctx, a, a.ne[0], a.ne[1], 1, a.ne[2] * a.ne[3]);
    const im2col = ggml_im2col(
        ctx,
        new_a,
        ggml_reshape_4d(ctx, b, b.ne[0], b.ne[1], 1, b.ne[2] * b.ne[3]),
        s0,
        s1,
        p0,
        p1,
        d0,
        d1,
        true,
        im2colType(a),
    );
    // [N*IC, OH, OW, KH*KW] => [N, IC, OH*OW, KH*KW]
    const new_b = ggml_reshape_4d(ctx, im2col, im2col.ne[0], im2col.ne[2] * im2col.ne[1], b.ne[2], b.ne[3]);

    // [OC, 1, KH, KW] => [1, OC, 1, KH*KW]
    new_a = ggml_reshape_4d(ctx, new_a, new_a.ne[0] * new_a.ne[1], new_a.ne[2], new_a.ne[3], 1);
    const result = ggml_mul_mat(ctx, new_a, new_b);
    return ggml_reshape_4d(ctx, result, im2col.ne[1], im2col.ne[2], b.ne[2], b.ne[3]);
}

/// Ports `ggml_conv_2d_dw_direct` (ggml.c:4844 @c1d0e7a00).
///
/// A real op code, so no im2col buffer is materialised.
///
/// When the input is in channels-first (CWHN) order the result is given the
/// same permuted strides, so the layout survives the op rather than forcing a
/// copy on either side.
pub export fn ggml_conv_2d_dw_direct(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    stride0: c_int,
    stride1: c_int,
    pad0: c_int,
    pad1: c_int,
    dilation0: c_int,
    dilation1: c_int,
) *Tensor {
    impl.assert(a.ne[2] == 1, "a->ne[2] == 1");
    impl.assert(a.ne[3] == b.ne[2], "a->ne[3] == b->ne[2]");

    const ne = [_]i64{
        convOutputSize(b.ne[0], a.ne[0], stride0, pad0, dilation0),
        convOutputSize(b.ne[1], a.ne[1], stride1, pad1, dilation1),
        b.ne[2],
        b.ne[3],
    };
    const result = context.ggml_new_tensor(ctx, b.type, 4, &ne);

    if (types.ggml_is_contiguous_channels(b)) {
        const type_size = types.ggml_type_size(result.type);
        impl.assert(types.ggml_blck_size(result.type) == 1, "ggml_blck_size(result->type) == 1");
        result.nb[0] = @as(usize, @intCast(result.ne[2])) * type_size;
        result.nb[1] = @as(usize, @intCast(result.ne[0])) * result.nb[0];
        result.nb[2] = type_size;
    }

    impl.setOpParamsValue(result, [_]i32{ stride0, stride1, pad0, pad1, dilation0, dilation1 });

    result.op = c.GGML_OP_CONV_2D_DW;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_conv_2d_direct` (ggml.c:4884 @c1d0e7a00).
///
/// Note the result takes `b`'s type, not f32: a direct kernel writes in the
/// input's precision.
pub export fn ggml_conv_2d_direct(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    s0: c_int,
    s1: c_int,
    p0: c_int,
    p1: c_int,
    d0: c_int,
    d1: c_int,
) *Tensor {
    impl.assert(a.ne[2] == b.ne[2], "a->ne[2] == b->ne[2]");

    const ne = [_]i64{
        convOutputSize(b.ne[0], a.ne[0], s0, p0, d0),
        convOutputSize(b.ne[1], a.ne[1], s1, p1, d1),
        a.ne[3],
        b.ne[3],
    };
    const result = context.ggml_new_tensor(ctx, b.type, 4, &ne);

    impl.setOpParamsI32(result, 0, s0);
    impl.setOpParamsI32(result, 1, s1);
    impl.setOpParamsI32(result, 2, p0);
    impl.setOpParamsI32(result, 3, p1);
    impl.setOpParamsI32(result, 4, d0);
    impl.setOpParamsI32(result, 5, d1);

    result.op = c.GGML_OP_CONV_2D;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_im2col_3d` (ggml.c:4717 @c1d0e7a00).
///
/// The 3-D counterpart of `im2col`. Channels are folded into the batch
/// dimension of both operands, so `IC` has to be passed explicitly to recover
/// the real batch and output-channel counts.
pub export fn ggml_im2col_3d(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    ic: i64,
    s0: c_int,
    s1: c_int,
    s2: c_int,
    p0: c_int,
    p1: c_int,
    p2: c_int,
    d0: c_int,
    d1: c_int,
    d2: c_int,
    dst_type: c.enum_ggml_type,
) *Tensor {
    const n = @divTrunc(b.ne[3], ic);
    const od = convOutputSize(b.ne[2], a.ne[2], s2, p2, d2);
    const oh = convOutputSize(b.ne[1], a.ne[1], s1, p1, d1);
    const ow = convOutputSize(b.ne[0], a.ne[0], s0, p0, d0);

    impl.assert(od > 0, "b too small compared to a");
    impl.assert(oh > 0, "b too small compared to a");
    impl.assert(ow > 0, "b too small compared to a");

    const ne = [_]i64{ a.ne[0] * a.ne[1] * a.ne[2] * ic, ow, oh, od * n };
    const result = context.ggml_new_tensor(ctx, dst_type, 4, &ne);
    impl.setOpParamsValue(result, [_]i32{ s0, s1, s2, p0, p1, p2, d0, d1, d2, @intCast(ic) });

    result.op = c.GGML_OP_IM2COL_3D;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_conv_3d` (ggml.c:4767 @c1d0e7a00).
///
/// Composite, like `ggml_conv_2d`: im2col_3d, matmul, then reshapes and a
/// permute to get the depth and channel dimensions back in order.
pub export fn ggml_conv_3d(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    ic: i64,
    s0: c_int,
    s1: c_int,
    s2: c_int,
    p0: c_int,
    p1: c_int,
    p2: c_int,
    d0: c_int,
    d1: c_int,
    d2: c_int,
) *Tensor {
    const im2col = ggml_im2col_3d(ctx, a, b, ic, s0, s1, s2, p0, p1, p2, d0, d1, d2, im2colType(a));

    const oc = @divTrunc(a.ne[3], ic);
    const n = @divTrunc(b.ne[3], ic);

    var result = ggml_mul_mat(
        ctx,
        ggml_reshape_2d(ctx, im2col, im2col.ne[0], im2col.ne[3] * im2col.ne[2] * im2col.ne[1]),
        ggml_reshape_2d(ctx, a, a.ne[0] * a.ne[1] * a.ne[2] * ic, oc),
    );

    const od = @divTrunc(im2col.ne[3], n);
    result = ggml_reshape_4d(ctx, result, im2col.ne[1] * im2col.ne[2], od, n, oc);
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
    return ggml_reshape_4d(ctx, result, im2col.ne[1], im2col.ne[2], od, oc * n);
}

/// Ports `ggml_conv_3d_direct` (ggml.c:4922 @c1d0e7a00).
pub export fn ggml_conv_3d_direct(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    s0: c_int,
    s1: c_int,
    s2: c_int,
    p0: c_int,
    p1: c_int,
    p2: c_int,
    d0: c_int,
    d1: c_int,
    d2: c_int,
    n_channels: c_int,
    n_batch: c_int,
    n_channels_out: c_int,
) *Tensor {
    impl.assert(a.ne[3] == @as(i64, n_channels) * n_channels_out, "a->ne[3] == c * oc");
    impl.assert(b.ne[3] == @as(i64, n_channels) * n_batch, "b->ne[3] == c * n");

    const ne = [_]i64{
        convOutputSize(b.ne[0], a.ne[0], s0, p0, d0),
        convOutputSize(b.ne[1], a.ne[1], s1, p1, d1),
        convOutputSize(b.ne[2], a.ne[2], s2, p2, d2),
        @as(i64, n_channels_out) * n_batch,
    };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);

    const params = [_]i32{ s0, s1, s2, p0, p1, p2, d0, d1, d2, n_channels, n_batch, n_channels_out };
    for (params, 0..) |v, i| impl.setOpParamsI32(result, i, v);

    result.op = c.GGML_OP_CONV_3D;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

/// Ports `ggml_calc_conv_transpose_output_size` (ggml.c:4972 @c1d0e7a00).
fn convTransposeOutputSize(ins: i64, ks: i64, s: c_int, p: c_int) i64 {
    return (ins - 1) * s - 2 * p + ks;
}

/// Ports `ggml_conv_transpose_2d_p0` (ggml.c:4976 @c1d0e7a00).
///
/// Zero padding only, which is what the `_p0` names.
pub export fn ggml_conv_transpose_2d_p0(ctx: *Context, a: *Tensor, b: *Tensor, stride: c_int) *Tensor {
    impl.assert(a.ne[3] == b.ne[2], "a->ne[3] == b->ne[2]");

    const ne = [_]i64{
        convTransposeOutputSize(b.ne[0], a.ne[0], stride, 0),
        convTransposeOutputSize(b.ne[1], a.ne[1], stride, 0),
        a.ne[2],
        b.ne[3],
    };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    impl.setOpParamsI32(result, 0, stride);

    result.op = c.GGML_OP_CONV_TRANSPOSE_2D;
    result.src[0] = a;
    result.src[1] = b;
    return result;
}

// -----------------------------------------------------------------------------
// Pooling

/// Ports `ggml_calc_pool_output_size` (ggml.c:5002 @c1d0e7a00).
///
/// Padding is a float here, not an int, because `ggml_pool_2d` accepts
/// fractional padding. The division still truncates, matching the C.
fn poolOutputSize(ins: i64, ks: c_int, s: c_int, p: f32) i64 {
    const padded = @as(f32, @floatFromInt(ins)) + 2 * p - @as(f32, @floatFromInt(ks));
    return @as(i64, @intFromFloat(padded / @as(f32, @floatFromInt(s)))) + 1;
}

/// Ports `ggml_pool_1d` (ggml.c:5008 @c1d0e7a00).
pub export fn ggml_pool_1d(ctx: *Context, a: *Tensor, op: c.enum_ggml_op_pool, k0: c_int, s0: c_int, p0: c_int) *Tensor {
    const ne = [_]i64{
        poolOutputSize(a.ne[0], k0, s0, @floatFromInt(p0)),
        a.ne[1],
        a.ne[2],
        a.ne[3],
    };
    impl.assert(ne[0] > 0, "ne[0] > 0");

    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    impl.setOpParamsValue(result, [_]i32{ @intCast(op), k0, s0, p0 });

    result.op = c.GGML_OP_POOL_1D;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_pool_2d` (ggml.c:5036 @c1d0e7a00).
///
/// The padding arguments are floats but land in `op_params` as i32, truncated,
/// which is what the C's brace-initialised `int32_t params[]` does.
pub export fn ggml_pool_2d(
    ctx: *Context,
    a: *Tensor,
    op: c.enum_ggml_op_pool,
    k0: c_int,
    k1: c_int,
    s0: c_int,
    s1: c_int,
    p0: f32,
    p1: f32,
) *Tensor {
    const ne = [_]i64{
        poolOutputSize(a.ne[0], k0, s0, p0),
        poolOutputSize(a.ne[1], k1, s1, p1),
        a.ne[2],
        a.ne[3],
    };
    impl.assert(ne[0] > 0, "ne[0] > 0");
    impl.assert(ne[1] > 0, "ne[1] > 0");

    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    impl.setOpParamsValue(result, [_]i32{
        @intCast(op),      k0,                k1, s0, s1,
        @intFromFloat(p0), @intFromFloat(p1),
    });

    result.op = c.GGML_OP_POOL_2D;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_pool_2d_back` (ggml.c:5067 @c1d0e7a00).
///
/// `af` is the forward pass's input, needed because max pooling has to know
/// which element won in order to route the gradient to it.
pub export fn ggml_pool_2d_back(
    ctx: *Context,
    a: *Tensor,
    af: *Tensor,
    op: c.enum_ggml_op_pool,
    k0: c_int,
    k1: c_int,
    s0: c_int,
    s1: c_int,
    p0: f32,
    p1: f32,
) *Tensor {
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &af.ne);
    impl.setOpParamsValue(result, [_]i32{
        @intCast(op),      k0,                k1, s0, s1,
        @intFromFloat(p0), @intFromFloat(p1),
    });

    result.op = c.GGML_OP_POOL_2D_BACK;
    result.src[0] = a;
    result.src[1] = af;
    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests for 2-D convolution and pooling

test "conv_2d composite ends in a contiguous permute" {
    var buf: [1 << 17]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // kernel [KW, KH, IC, OC]; image [IW, IH, IC, N]
    const kernel = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F16, 3, 3, 4, 8);
    const image = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 16, 4, 2);

    const r = ggml_conv_2d(&ctx, kernel, image, 1, 1, 1, 1, 1, 1);
    // The matmul leaves [OC, N, OH, OW], so the graph ends with a cont over a
    // permute rather than a conv op.
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_CONT), r.op);
    try std.testing.expectEqual(@as(i64, 16), r.ne[0]); // OW
    try std.testing.expectEqual(@as(i64, 16), r.ne[1]); // OH
    try std.testing.expectEqual(@as(i64, 8), r.ne[2]); // OC
    try std.testing.expectEqual(@as(i64, 2), r.ne[3]); // N
}

test "conv_2d_direct has its own op and keeps the input type" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const kernel = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F16, 3, 3, 4, 8);
    const image = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F16, 16, 16, 4, 2);

    const r = ggml_conv_2d_direct(&ctx, kernel, image, 1, 1, 1, 1, 1, 1);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_CONV_2D), r.op);
    // Direct kernels write in the input's precision, not f32.
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F16), r.type);
    try std.testing.expectEqual(@as(i64, 16), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[2]);
    // All six geometry values, in order.
    for ([_]i32{ 1, 1, 1, 1, 1, 1 }, 0..) |want, i| {
        try std.testing.expectEqual(want, impl.getOpParamsI32(r, i));
    }
}

test "patch embedding tiles the image" {
    var buf: [1 << 17]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // A 14x14 patch kernel over a 224x224 image gives a 16x16 grid.
    const kernel = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F16, 14, 14, 3, 768);
    const image = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 224, 224, 3, 1);

    const r = ggml_conv_2d_sk_p0(&ctx, kernel, image);
    try std.testing.expectEqual(@as(i64, 16), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 16), r.ne[1]);
    try std.testing.expectEqual(@as(i64, 768), r.ne[2]);

    // Stride-1 half-padding preserves the spatial size instead.
    const k3 = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F16, 3, 3, 3, 16);
    const small = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 32, 32, 3, 1);
    const same = ggml_conv_2d_s1_ph(&ctx, k3, small);
    try std.testing.expectEqual(@as(i64, 32), same.ne[0]);
    try std.testing.expectEqual(@as(i64, 32), same.ne[1]);
}

test "depthwise direct conv preserves a channels-first layout" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const kernel = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 3, 3, 1, 8);
    const plain = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 16, 8, 1);

    const r = ggml_conv_2d_dw_direct(&ctx, kernel, plain, 1, 1, 1, 1, 1, 1);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_CONV_2D_DW), r.op);
    try std.testing.expectEqual(@as(i64, 8), r.ne[2]); // channels pass through
    // A contiguous input yields a contiguous result.
    try std.testing.expect(types.ggml_is_contiguous(r));

    // A CWHN input makes the result carry the same permuted strides, so the
    // layout survives instead of forcing a copy. Built by hand rather than via
    // permute, because ggml_is_contiguous_channels wants this exact stride
    // shape: channels innermost, nb[2] equal to one element.
    const permuted = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 16, 8, 1);
    permuted.nb[0] = 8 * 4;
    permuted.nb[1] = 16 * permuted.nb[0];
    permuted.nb[2] = 4;
    try std.testing.expect(types.ggml_is_contiguous_channels(permuted));
    const rp = ggml_conv_2d_dw_direct(&ctx, kernel, permuted, 1, 1, 1, 1, 1, 1);
    try std.testing.expectEqual(@as(usize, 4), rp.nb[2]);
    try std.testing.expect(types.ggml_is_contiguous_channels(rp));
}

test "pooling shrinks by the pool formula" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 32, 32, 8, 1);

    // 2x2 stride 2 halves both spatial dimensions and leaves channels alone.
    const p = ggml_pool_2d(&ctx, a, c.GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    try std.testing.expectEqual(@as(i64, 16), p.ne[0]);
    try std.testing.expectEqual(@as(i64, 16), p.ne[1]);
    try std.testing.expectEqual(@as(i64, 8), p.ne[2]);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_OP_POOL_MAX)), impl.getOpParamsI32(p, 0));

    const p1 = ggml_pool_1d(&ctx, a, c.GGML_OP_POOL_AVG, 4, 4, 0);
    try std.testing.expectEqual(@as(i64, 8), p1.ne[0]);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_OP_POOL_AVG)), impl.getOpParamsI32(p1, 0));

    // The backward pass takes the forward input's shape, not the pooled one.
    const back = ggml_pool_2d_back(&ctx, p, a, c.GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    try std.testing.expect(types.ggml_are_same_shape(back, a));
    try std.testing.expectEqual(@as(?*Tensor, a), back.src[1]);
}

test "conv_transpose_2d grows the image" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // kernel [KW, KH, OC, IC]; image [IW, IH, IC, N]
    const kernel = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F16, 4, 4, 8, 16);
    const image = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 8, 8, 16, 1);

    // (8-1)*2 - 0 + 4 = 18
    const r = ggml_conv_transpose_2d_p0(&ctx, kernel, image, 2);
    try std.testing.expectEqual(@as(i64, 18), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 18), r.ne[1]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[2]); // OC
    try std.testing.expectEqual(@as(i32, 2), impl.getOpParamsI32(r, 0));
}

test "3d im2col folds channels into the batch dimension" {
    var buf: [1 << 17]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // IC=2: kernel [KW, KH, KD, OC*IC], volume [IW, IH, ID, N*IC]
    const kernel = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F16, 3, 3, 3, 8);
    const volume = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 16, 16, 2);

    const r = ggml_im2col_3d(&ctx, kernel, volume, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, c.GGML_TYPE_F16);
    try std.testing.expectEqual(@as(i64, 3 * 3 * 3 * 2), r.ne[0]); // KW*KH*KD*IC
    try std.testing.expectEqual(@as(i64, 16), r.ne[1]); // OW
    try std.testing.expectEqual(@as(i64, 16), r.ne[2]); // OH
    // IC is recorded so the backend can recover the real batch count.
    try std.testing.expectEqual(@as(i32, 2), impl.getOpParamsI32(r, 9));
}

// -----------------------------------------------------------------------------
// Resampling and padding

/// Ports `ggml_interpolate_impl` (ggml.c:5093 @c1d0e7a00).
///
/// `mode` packs a scale mode in its low byte and flags above it, so it is
/// masked before the range check.
fn interpolateImpl(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, mode: u32) *Tensor {
    impl.assert((mode & 0xFF) < c.GGML_SCALE_MODE_COUNT, "(mode & 0xFF) < GGML_SCALE_MODE_COUNT");
    // Antialiasing is only implemented for bilinear.
    impl.assert(
        (mode & c.GGML_SCALE_FLAG_ANTIALIAS) == 0 or (mode & 0xFF) == c.GGML_SCALE_MODE_BILINEAR,
        "!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR",
    );
    impl.assert(a.type == c.GGML_TYPE_F32, "a->type == GGML_TYPE_F32");

    const result = context.ggml_new_tensor_4d(ctx, a.type, ne0, ne1, ne2, ne3);
    impl.setOpParamsI32(result, 0, @bitCast(mode));

    // Interpolation and upscaling share one op code.
    result.op = c.GGML_OP_UPSCALE;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_upscale` (ggml.c:5116 @c1d0e7a00).
///
/// Scales the two spatial dimensions by an integer factor, leaving channels
/// and batch alone.
pub export fn ggml_upscale(ctx: *Context, a: *Tensor, scale_factor: c_int, mode: c.enum_ggml_scale_mode) *Tensor {
    impl.assert(scale_factor > 1, "scale_factor > 1");
    return interpolateImpl(ctx, a, a.ne[0] * scale_factor, a.ne[1] * scale_factor, a.ne[2], a.ne[3], @intCast(mode));
}

/// Ports `ggml_upscale_ext` (ggml.c:5125 @c1d0e7a00).
pub export fn ggml_upscale_ext(ctx: *Context, a: *Tensor, ne0: c_int, ne1: c_int, ne2: c_int, ne3: c_int, mode: c.enum_ggml_scale_mode) *Tensor {
    return interpolateImpl(ctx, a, ne0, ne1, ne2, ne3, @intCast(mode));
}

/// Ports `ggml_interpolate` (ggml.c:5136 @c1d0e7a00).
///
/// The general form: any target shape, and `mode` may carry flags as well as a
/// scale mode.
pub export fn ggml_interpolate(ctx: *Context, a: *Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, mode: u32) *Tensor {
    return interpolateImpl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

/// Ports `ggml_pad_ext` (ggml.c:5171 @c1d0e7a00).
///
/// Independent left and right padding on all four dimensions. Slot 8 records
/// whether the padding wraps; `ggml_pad_ext_circular` flips it afterwards.
pub export fn ggml_pad_ext(
    ctx: *Context,
    a: *Tensor,
    lp0: c_int,
    rp0: c_int,
    lp1: c_int,
    rp1: c_int,
    lp2: c_int,
    rp2: c_int,
    lp3: c_int,
    rp3: c_int,
) *Tensor {
    const result = context.ggml_new_tensor_4d(
        ctx,
        a.type,
        a.ne[0] + lp0 + rp0,
        a.ne[1] + lp1 + rp1,
        a.ne[2] + lp2 + rp2,
        a.ne[3] + lp3 + rp3,
    );

    const pads = [_]i32{ lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3 };
    for (pads, 0..) |v, i| impl.setOpParamsI32(result, i, v);
    impl.setOpParamsI32(result, 8, 0); // not circular by default

    result.op = c.GGML_OP_PAD;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_pad` (ggml.c:5149 @c1d0e7a00).
///
/// Right-hand padding only, which is the common case.
pub export fn ggml_pad(ctx: *Context, a: *Tensor, p0: c_int, p1: c_int, p2: c_int, p3: c_int) *Tensor {
    return ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

/// Ports `ggml_pad_ext_circular` (ggml.c:5208 @c1d0e7a00).
///
/// Same shape as `ggml_pad_ext`, but the padding wraps around from the
/// opposite edge rather than being filled with zeros.
pub export fn ggml_pad_ext_circular(
    ctx: *Context,
    a: *Tensor,
    lp0: c_int,
    rp0: c_int,
    lp1: c_int,
    rp1: c_int,
    lp2: c_int,
    rp2: c_int,
    lp3: c_int,
    rp3: c_int,
) *Tensor {
    const result = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    impl.setOpParamsI32(result, 8, 1);
    return result;
}

/// Ports `ggml_pad_circular` (ggml.c:5161 @c1d0e7a00).
pub export fn ggml_pad_circular(ctx: *Context, a: *Tensor, p0: c_int, p1: c_int, p2: c_int, p3: c_int) *Tensor {
    return ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

/// Ports `ggml_pad_reflect_1d` (ggml.c:5227 @c1d0e7a00).
///
/// Mirrors the signal at each edge instead of filling. The padding must be
/// shorter than the dimension, since there would otherwise be nothing to
/// reflect.
pub export fn ggml_pad_reflect_1d(ctx: *Context, a: *Tensor, p0: c_int, p1: c_int) *Tensor {
    impl.assert(p0 >= 0, "p0 >= 0");
    impl.assert(p1 >= 0, "p1 >= 0");
    impl.assert(p0 < a.ne[0], "p0 < a->ne[0]");
    impl.assert(p1 < a.ne[0], "p1 < a->ne[0]");
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
    impl.assert(a.type == c.GGML_TYPE_F32, "a->type == GGML_TYPE_F32");

    const result = context.ggml_new_tensor_4d(ctx, a.type, a.ne[0] + p0 + p1, a.ne[1], a.ne[2], a.ne[3]);
    impl.setOpParamsValue(result, [_]i32{ p0, p1 });

    result.op = c.GGML_OP_PAD_REFLECT_1D;
    result.src[0] = a;
    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests for resampling and padding

test "upscale scales only the spatial dimensions" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 16, 8, 2);

    const r = ggml_upscale(&ctx, a, 2, c.GGML_SCALE_MODE_NEAREST);
    try std.testing.expectEqual(@as(i64, 32), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 32), r.ne[1]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[2]); // channels unchanged
    try std.testing.expectEqual(@as(i64, 2), r.ne[3]);
    // Upscale and interpolate share one op code.
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_UPSCALE), r.op);

    // The extended form takes an arbitrary target shape.
    const ext = ggml_upscale_ext(&ctx, a, 24, 20, 8, 2, c.GGML_SCALE_MODE_BILINEAR);
    try std.testing.expectEqual(@as(i64, 24), ext.ne[0]);
    try std.testing.expectEqual(@as(i64, 20), ext.ne[1]);
}

test "interpolate mode packs flags above the scale mode" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 16, 3, 1);

    // Antialiasing is a flag above the low byte, and only bilinear accepts it.
    const mode: u32 = c.GGML_SCALE_MODE_BILINEAR | c.GGML_SCALE_FLAG_ANTIALIAS;
    const r = ggml_interpolate(&ctx, a, 8, 8, 3, 1, mode);
    try std.testing.expectEqual(@as(i32, @bitCast(mode)), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i64, 8), r.ne[0]);

    // The scale mode survives the round trip through op_params intact.
    const stored: u32 = @bitCast(impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(u32, c.GGML_SCALE_MODE_BILINEAR), stored & 0xFF);
    try std.testing.expect((stored & c.GGML_SCALE_FLAG_ANTIALIAS) != 0);
}

test "padding grows each dimension by both margins" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 8, 4, 2);

    // ggml_pad is right-hand only.
    const r = ggml_pad(&ctx, a, 2, 3, 0, 0);
    try std.testing.expectEqual(@as(i64, 18), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 11), r.ne[1]);
    try std.testing.expectEqual(@as(i64, 4), r.ne[2]);
    // Left margins stay zero, right margins carry the values.
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i32, 2), impl.getOpParamsI32(r, 1));
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(r, 8)); // not circular

    // The extended form pads both sides independently.
    const ext = ggml_pad_ext(&ctx, a, 1, 2, 3, 4, 0, 0, 0, 0);
    try std.testing.expectEqual(@as(i64, 19), ext.ne[0]);
    try std.testing.expectEqual(@as(i64, 15), ext.ne[1]);
}

test "circular padding differs only in the wrap flag" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 16, 8, 4, 2);

    const plain = ggml_pad_ext(&ctx, a, 1, 1, 0, 0, 0, 0, 0, 0);
    const wrap = ggml_pad_ext_circular(&ctx, a, 1, 1, 0, 0, 0, 0, 0, 0);

    try std.testing.expect(types.ggml_are_same_shape(plain, wrap));
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(plain, 8));
    try std.testing.expectEqual(@as(i32, 1), impl.getOpParamsI32(wrap, 8));
    try std.testing.expectEqual(@as(i32, 1), impl.getOpParamsI32(ggml_pad_circular(&ctx, a, 1, 0, 0, 0), 8));
}

test "reflect padding grows only the first dimension" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 16, 4);

    const r = ggml_pad_reflect_1d(&ctx, a, 3, 5);
    try std.testing.expectEqual(@as(i64, 24), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 4), r.ne[1]);
    try std.testing.expectEqual(@as(i32, 3), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i32, 5), impl.getOpParamsI32(r, 1));
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_PAD_REFLECT_1D), r.op);
}

// -----------------------------------------------------------------------------
// Sorting and selection

/// Ports `ggml_argsort` (ggml.c:5360 @c1d0e7a00).
///
/// Returns indices, not values, so the result is i32 whatever went in.
pub export fn ggml_argsort(ctx: *Context, a: *Tensor, order: c.enum_ggml_sort_order) *Tensor {
    impl.assert(a.ne[0] <= std.math.maxInt(i32), "a->ne[0] <= INT32_MAX");

    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_I32, c.GGML_MAX_DIMS, &a.ne);
    impl.setOpParamsI32(result, 0, @intCast(order));

    result.op = c.GGML_OP_ARGSORT;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_argsort_top_k` (ggml.c:5378 @c1d0e7a00).
///
/// A full descending sort, then a view of its first `k` columns. Cheaper than
/// it looks only when the backend fuses the two; `ggml_top_k` is the dedicated
/// op that avoids sorting the tail.
pub export fn ggml_argsort_top_k(ctx: *Context, a: *Tensor, k: c_int) *Tensor {
    impl.assert(a.ne[0] >= k, "a->ne[0] >= k");

    const sorted = ggml_argsort(ctx, a, c.GGML_SORT_ORDER_DESC);
    return ggml_view_4d(
        ctx,
        sorted,
        k,
        sorted.ne[1],
        sorted.ne[2],
        sorted.ne[3],
        sorted.nb[1],
        sorted.nb[2],
        sorted.nb[3],
        0,
    );
}

/// Ports `ggml_top_k` (ggml.c:5396 @c1d0e7a00).
pub export fn ggml_top_k(ctx: *Context, a: *Tensor, k: c_int) *Tensor {
    impl.assert(a.ne[0] >= k, "a->ne[0] >= k");

    const result = context.ggml_new_tensor_4d(ctx, c.GGML_TYPE_I32, k, a.ne[1], a.ne[2], a.ne[3]);
    result.op = c.GGML_OP_TOP_K;
    result.src[0] = a;
    return result;
}

// -----------------------------------------------------------------------------
// Generators and rearrangement

/// Ports `ggml_arange` (ggml.c:5412 @c1d0e7a00).
///
/// The only op with no source at all: it generates rather than transforms.
pub export fn ggml_arange(ctx: *Context, start: f32, stop: f32, step: f32) *Tensor {
    impl.assert(stop > start, "stop > start");

    const steps: i64 = @intFromFloat(@ceil((stop - start) / step));
    const result = context.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, steps);

    impl.setOpParamsF32(result, 0, start);
    impl.setOpParamsF32(result, 1, stop);
    impl.setOpParamsF32(result, 2, step);

    result.op = c.GGML_OP_ARANGE;
    return result;
}

/// Ports `ggml_timestep_embedding` (ggml.c:5286 @c1d0e7a00).
///
/// Sinusoidal position encoding for diffusion timesteps: one `dim`-wide
/// embedding per timestep.
pub export fn ggml_timestep_embedding(ctx: *Context, timesteps: *Tensor, dim: c_int, max_period: c_int) *Tensor {
    const result = context.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, dim, timesteps.ne[0]);

    impl.setOpParamsI32(result, 0, dim);
    impl.setOpParamsI32(result, 1, max_period);

    result.op = c.GGML_OP_TIMESTEP_EMBEDDING;
    result.src[0] = timesteps;
    return result;
}

/// Ports `ggml_roll` (ggml.c:5258 @c1d0e7a00).
///
/// Circular shift along each dimension. Requires a contiguous first dimension,
/// and each shift must be smaller than the dimension it moves.
pub export fn ggml_roll(ctx: *Context, a: *Tensor, shift0: c_int, shift1: c_int, shift2: c_int, shift3: c_int) *Tensor {
    impl.assert(a.nb[0] == types.ggml_type_size(a.type), "a->nb[0] == ggml_type_size(a->type)");

    const shifts = [_]c_int{ shift0, shift1, shift2, shift3 };
    for (shifts, 0..) |shift, i| {
        impl.assert(@abs(shift) < a.ne[i], "abs(shift) < a->ne[i]");
    }

    const result = context.ggml_dup_tensor(ctx, a);
    for (shifts, 0..) |shift, i| impl.setOpParamsI32(result, i, shift);

    result.op = c.GGML_OP_ROLL;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_tri` (ggml.c:5305 @c1d0e7a00).
///
/// Masks a square matrix to one triangle. `type` selects which triangle and
/// whether the diagonal is kept.
pub export fn ggml_tri(ctx: *Context, a: *Tensor, tri_type: c.enum_ggml_tri_type) *Tensor {
    impl.assert(a.type == c.GGML_TYPE_F32, "a->type == GGML_TYPE_F32");
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
    impl.assert(a.ne[0] == a.ne[1], "a->ne[0] == a->ne[1]");

    const result = context.ggml_dup_tensor(ctx, a);
    impl.setOpParamsI32(result, 0, @intCast(tri_type));

    result.op = c.GGML_OP_TRI;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_fill_impl` (ggml.c:5326 @c1d0e7a00).
fn fillImpl(ctx: *Context, a: *Tensor, value: f32, inplace: bool) *Tensor {
    impl.assert(
        a.type == c.GGML_TYPE_F32 or a.type == c.GGML_TYPE_F16,
        "a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16",
    );
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");

    const result = dupOrView(ctx, a, inplace);
    impl.setOpParamsF32(result, 0, value);

    result.op = c.GGML_OP_FILL;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_fill` (ggml.c:5344 @c1d0e7a00).
pub export fn ggml_fill(ctx: *Context, a: *Tensor, value: f32) *Tensor {
    return fillImpl(ctx, a, value, false);
}

/// Ports `ggml_fill_inplace` (ggml.c:5351 @c1d0e7a00).
pub export fn ggml_fill_inplace(ctx: *Context, a: *Tensor, value: f32) *Tensor {
    return fillImpl(ctx, a, value, true);
}

/// Ports `ggml_leaky_relu` (ggml.c:2750 @c1d0e7a00).
///
/// Its own op code rather than a `GGML_UNARY_OP`, because it carries a
/// parameter and the unary ops do not.
pub export fn ggml_leaky_relu(ctx: *Context, a: *Tensor, negative_slope: f32, inplace: bool) *Tensor {
    const result = dupOrView(ctx, a, inplace);
    impl.setOpParamsValue(result, negative_slope);

    result.op = c.GGML_OP_LEAKY_RELU;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_swiglu_oai` (ggml.c:3109 @c1d0e7a00).
///
/// SwiGLU with OpenAI's clamping. Slots 0 and 1 are the op and swap flag that
/// `gluImpl` writes; the two extra parameters go above them.
pub export fn ggml_swiglu_oai(ctx: *Context, a: *Tensor, b: *Tensor, alpha: f32, limit: f32) *Tensor {
    const result = gluImpl(ctx, a, b, c.GGML_GLU_OP_SWIGLU_OAI, false);
    impl.setOpParamsF32(result, 2, alpha);
    impl.setOpParamsF32(result, 3, limit);
    return result;
}

// -----------------------------------------------------------------------------
// Windowed attention
//
// Used by vision models that attend within local windows rather than globally.
// The C marks these as internal-only and subject to removal.

/// Ports `ggml_win_part` (ggml.c:5686 @c1d0e7a00).
///
/// Splits an image into `w`-by-`w` windows, padding up to a whole number of
/// them, and stacks the windows along the batch dimension.
pub export fn ggml_win_part(ctx: *Context, a: *Tensor, w: c_int) *Tensor {
    impl.assert(a.ne[3] == 1, "a->ne[3] == 1");
    impl.assert(a.type == c.GGML_TYPE_F32, "a->type == GGML_TYPE_F32");

    // Padding needed to reach a whole number of windows on each axis.
    const px = @rem(w - @rem(a.ne[1], w), w);
    const py = @rem(w - @rem(a.ne[2], w), w);
    const npx = @divTrunc(px + a.ne[1], w);
    const npy = @divTrunc(py + a.ne[2], w);
    const np = npx * npy;

    const ne = [_]i64{ a.ne[0], w, w, np };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
    impl.setOpParamsValue(result, [_]i32{ @intCast(npx), @intCast(npy), w });

    result.op = c.GGML_OP_WIN_PART;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_win_unpart` (ggml.c:5715 @c1d0e7a00).
///
/// The inverse. The original size cannot be recovered from the windows, so the
/// caller supplies it.
pub export fn ggml_win_unpart(ctx: *Context, a: *Tensor, w0: c_int, h0: c_int, w: c_int) *Tensor {
    impl.assert(a.type == c.GGML_TYPE_F32, "a->type == GGML_TYPE_F32");

    const ne = [_]i64{ a.ne[0], w0, h0, 1 };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 3, &ne);
    impl.setOpParamsValue(result, [_]i32{w});

    result.op = c.GGML_OP_WIN_UNPART;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_get_rel_pos` (ggml.c:5737 @c1d0e7a00).
///
/// Expands a table of relative position embeddings into a query-by-key matrix.
/// The table holds `2*max(qh,kh) - 1` entries, one per possible offset.
pub export fn ggml_get_rel_pos(ctx: *Context, a: *Tensor, qh: c_int, kh: c_int) *Tensor {
    impl.assert(qh == kh, "qh == kh");
    impl.assert(2 * @max(qh, kh) - 1 == a.ne[1], "2*MAX(qh, kh) - 1 == a->ne[1]");

    const ne = [_]i64{ a.ne[0], kh, qh, 1 };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F16, 3, &ne);

    result.op = c.GGML_OP_GET_REL_POS;
    result.src[0] = a;
    return result;
}

/// Ports `ggml_add_rel_pos_impl` (ggml.c:5756 @c1d0e7a00).
fn addRelPosImpl(ctx: *Context, a: *Tensor, pw: *Tensor, ph: *Tensor, inplace: bool) *Tensor {
    impl.assert(types.ggml_are_same_shape(pw, ph), "ggml_are_same_shape(pw, ph)");
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
    impl.assert(types.ggml_is_contiguous(pw), "ggml_is_contiguous(pw)");
    impl.assert(types.ggml_is_contiguous(ph), "ggml_is_contiguous(ph)");
    impl.assert(ph.type == c.GGML_TYPE_F32, "ph->type == GGML_TYPE_F32");
    impl.assert(pw.type == c.GGML_TYPE_F32, "pw->type == GGML_TYPE_F32");
    impl.assert(pw.ne[3] == a.ne[2], "pw->ne[3] == a->ne[2]");
    impl.assert(pw.ne[0] * pw.ne[0] == a.ne[0], "pw->ne[0]*pw->ne[0] == a->ne[0]");
    impl.assert(pw.ne[1] * pw.ne[2] == a.ne[1], "pw->ne[1]*pw->ne[2] == a->ne[1]");

    const result = dupOrView(ctx, a, inplace);
    impl.setOpParamsI32(result, 0, if (inplace) 1 else 0);

    result.op = c.GGML_OP_ADD_REL_POS;
    result.src[0] = a;
    result.src[1] = pw;
    result.src[2] = ph;
    return result;
}

/// Ports `ggml_add_rel_pos` (ggml.c:5783 @c1d0e7a00).
pub export fn ggml_add_rel_pos(ctx: *Context, a: *Tensor, pw: *Tensor, ph: *Tensor) *Tensor {
    return addRelPosImpl(ctx, a, pw, ph, false);
}

/// Ports `ggml_add_rel_pos_inplace` (ggml.c:5791 @c1d0e7a00).
pub export fn ggml_add_rel_pos_inplace(ctx: *Context, a: *Tensor, pw: *Tensor, ph: *Tensor) *Tensor {
    return addRelPosImpl(ctx, a, pw, ph, true);
}

// -----------------------------------------------------------------------------
// Tensor flags
//
// These mark a tensor's role for the graph allocator and the autodiff pass.

/// Ports `ggml_set_input` (ggml.c:7894 @c1d0e7a00).
pub export fn ggml_set_input(tensor: *Tensor) void {
    tensor.flags |= c.GGML_TENSOR_FLAG_INPUT;
}

/// Ports `ggml_set_output` (ggml.c:7898 @c1d0e7a00).
///
/// Walks the view chain, because the allocator must not reuse the storage a
/// view points into either.
pub export fn ggml_set_output(tensor: *Tensor) void {
    var cur: ?*Tensor = tensor;
    while (cur) |t| : (cur = if (t.view_src != null) impl.one(c.ggml_tensor, t.view_src) else null) {
        t.flags |= c.GGML_TENSOR_FLAG_OUTPUT;
    }
}

/// Ports `ggml_set_param` (ggml.c:7904 @c1d0e7a00).
///
/// Only a leaf can be a parameter: something computed has a gradient path of
/// its own.
pub export fn ggml_set_param(tensor: *Tensor) void {
    impl.assert(tensor.op == c.GGML_OP_NONE, "tensor->op == GGML_OP_NONE");
    tensor.flags |= c.GGML_TENSOR_FLAG_PARAM;
}

/// Ports `ggml_set_loss` (ggml.c:7909 @c1d0e7a00).
pub export fn ggml_set_loss(tensor: *Tensor) void {
    impl.assert(types.ggml_is_scalar(tensor), "ggml_is_scalar(tensor)");
    impl.assert(tensor.type == c.GGML_TYPE_F32, "tensor->type == GGML_TYPE_F32");
    tensor.flags |= c.GGML_TENSOR_FLAG_LOSS;
}

/// Ports `ggml_set_zero` (ggml.c:7497 @c1d0e7a00).
///
/// Zeroes a tensor's data now, rather than adding a node. Goes through the
/// backend when the tensor lives on one, since the memory may not be host
/// addressable.
pub export fn ggml_set_zero(tensor: *Tensor) *Tensor {
    if (types.ggml_is_empty(tensor)) return tensor;

    const nbytes = types.ggml_nbytes(tensor);
    if (tensor.buffer != null) {
        c.ggml_backend_tensor_memset(tensor, 0, 0, nbytes);
    } else {
        impl.assert(tensor.data != null, "tensor->data");
        @memset(@as([*]u8, @ptrCast(tensor.data.?))[0..nbytes], 0);
    }
    return tensor;
}

// -----------------------------------------------------------------------------
// Flash attention
//
// One node standing in for the whole scaled-dot-product attention block, so a
// backend can fuse it and never materialise the [n_kv, n_q] score matrix.

/// Ports `ggml_flash_attn_ext` (ggml.c:5434 @c1d0e7a00).
///
/// The result is the attention output already permuted to `(0, 2, 1, 3)`:
/// `[head_dim, n_head, n_tokens, n_batch]`. That transposition is folded in
/// here rather than left to a separate node, which is why the shape does not
/// look like `q`'s.
///
/// Parameters:
/// - `q`, `k`, `v`: query, key, and value.
/// - `mask`: optional f16 additive mask, which must be contiguous.
/// - `scale`: multiplies the scores before the softmax.
/// - `max_bias`: ALiBi slope ceiling; zero disables it, and any non-zero value
///   requires a mask.
/// - `logit_softcap`: tanh soft-cap on the scores; zero disables it.
///
/// Return: a new f32 tensor owned by `ctx`.
pub export fn ggml_flash_attn_ext(
    ctx: *Context,
    q: *Tensor,
    k: *Tensor,
    v: *Tensor,
    mask: ?*Tensor,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) *Tensor {
    impl.assert(canMulMat(k, q), "ggml_can_mul_mat(k, q)");

    impl.assert(q.ne[3] == k.ne[3], "q->ne[3] == k->ne[3]");
    impl.assert(q.ne[3] == v.ne[3], "q->ne[3] == v->ne[3]");

    if (mask) |m| {
        impl.assert(m.type == c.GGML_TYPE_F16, "mask->type == GGML_TYPE_F16");
        impl.assert(types.ggml_is_contiguous(m), "ggml_is_contiguous(mask)");

        impl.assert(@rem(q.ne[2], m.ne[2]) == 0, "q->ne[2] % mask->ne[2] == 0");
        impl.assert(@rem(q.ne[3], m.ne[3]) == 0, "q->ne[3] % mask->ne[3] == 0");
    }

    // ALiBi needs the mask to hang its slopes on.
    if (max_bias > 0.0) impl.assert(mask != null, "mask");

    // permute(0, 2, 1, 3)
    const ne = [4]i64{ v.ne[0], q.ne[2], q.ne[1], q.ne[3] };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);

    impl.setOpParamsValue(result, [_]f32{ scale, max_bias, logit_softcap });

    result.op = c.GGML_OP_FLASH_ATTN_EXT;
    result.src[0] = q;
    result.src[1] = k;
    result.src[2] = v;
    result.src[3] = mask;

    return result;
}

/// Ports `ggml_flash_attn_ext_set_prec` (ggml.c:5479 @c1d0e7a00).
///
/// Slot 3, immediately after the three floats `ggml_flash_attn_ext` wrote.
pub export fn ggml_flash_attn_ext_set_prec(a: *Tensor, prec: c.enum_ggml_prec) void {
    impl.assert(a.op == c.GGML_OP_FLASH_ATTN_EXT, "a->op == GGML_OP_FLASH_ATTN_EXT");

    impl.setOpParamsI32(a, 3, @intCast(prec));
}

/// Ports `ggml_flash_attn_ext_get_prec` (ggml.c:5489 @c1d0e7a00).
pub export fn ggml_flash_attn_ext_get_prec(a: *const Tensor) c.enum_ggml_prec {
    impl.assert(a.op == c.GGML_OP_FLASH_ATTN_EXT, "a->op == GGML_OP_FLASH_ATTN_EXT");

    return @intCast(impl.getOpParamsI32(a, 3));
}

/// Ports `ggml_flash_attn_ext_add_sinks` (ggml.c:5498 @c1d0e7a00).
///
/// Attaches per-head attention sinks as `src[4]`. A null `sinks` clears the
/// slot and returns before the assertions, so it is valid on any tensor --
/// which is what lets a caller unconditionally clear.
pub export fn ggml_flash_attn_ext_add_sinks(a: *Tensor, sinks: ?*Tensor) void {
    const s = sinks orelse {
        a.src[4] = null;
        return;
    };

    impl.assert(a.op == c.GGML_OP_FLASH_ATTN_EXT, "a->op == GGML_OP_FLASH_ATTN_EXT");
    impl.assert(a.src[4] == null, "a->src[4] == NULL");
    impl.assert(impl.one(c.ggml_tensor, a.src[0]).ne[2] == s.ne[0], "a->src[0]->ne[2] == sinks->ne[0]");
    impl.assert(s.type == c.GGML_TYPE_F32, "sinks->type == GGML_TYPE_F32");

    a.src[4] = s;
}

/// Ports `ggml_flash_attn_back` (ggml.c:5516 @c1d0e7a00).
///
/// **Aborts unconditionally.** The C opens with
/// `GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes")`, which is
/// `noreturn`, so every line after it is dead. The body is not reproduced
/// here: it cannot run, and transcribing unreachable code would invite a
/// reader to trust it. The symbol exists because the C exports it.
pub export fn ggml_flash_attn_back(
    ctx: *Context,
    q: *Tensor,
    k: *Tensor,
    v: *Tensor,
    d: *Tensor,
    masked: bool,
) *Tensor {
    _ = .{ ctx, q, k, v, d, masked };
    impl.abort("TODO: adapt to ggml_flash_attn_ext() changes");
}

// -----------------------------------------------------------------------------
// State-space models
//
// Mamba and its relatives. These return a single flat tensor holding the
// output and the updated recurrent state end to end, because the state has to
// survive the call and ggml has no second return value.

/// Ports `ggml_ssm_conv` (ggml.c:5587 @c1d0e7a00).
///
/// The depthwise causal convolution at the front of a Mamba block. `sx` holds
/// the `d_conv - 1` tokens of left context followed by the tokens themselves,
/// so the output is shorter than the input by exactly that context.
pub export fn ggml_ssm_conv(ctx: *Context, sx: *Tensor, conv: *Tensor) *Tensor {
    impl.assert(types.ggml_is_3d(sx), "ggml_is_3d(sx)");
    impl.assert(types.ggml_is_matrix(conv), "ggml_is_matrix(c)");

    const d_conv = conv.ne[0];
    const d_inner = conv.ne[1];
    const n_t = sx.ne[0] - d_conv + 1; // tokens per sequence
    const n_s = sx.ne[2];

    impl.assert(sx.ne[0] == d_conv - 1 + n_t, "sx->ne[0] == d_conv - 1 + n_t");
    impl.assert(sx.ne[1] == d_inner, "sx->ne[1] == d_inner");
    impl.assert(n_t >= 0, "n_t >= 0");

    const result = context.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, d_inner, n_t, n_s);

    result.op = c.GGML_OP_SSM_CONV;
    result.src[0] = sx;
    result.src[1] = conv;

    return result;
}

/// Ports `ggml_ssm_scan` (ggml.c:5615 @c1d0e7a00).
///
/// The selective scan. Seven sources -- more than any other op -- and a flat
/// 1d result holding `y` followed by `K` snapshots of the state per sequence.
///
/// Parameters:
/// - `s`: initial state, `[d_state, head_dim, n_head, n_seqs]`.
/// - `x`: input, `[head_dim, n_head, n_seq_tokens, n_seqs]`.
/// - `dt`: per-head timestep.
/// - `A`: decay. A first dimension of 1 is Mamba-2; `d_state` is Mamba-1,
///   which has per-state decay and therefore only supports `K == 1`.
/// - `B`, `C`: the selective projections, which must have the same shape.
/// - `ids`: i32 sequence indices, one per sequence.
/// - `K`: how many state snapshots to write.
///
/// Return: a new f32 1d tensor owned by `ctx`.
pub export fn ggml_ssm_scan(
    ctx: *Context,
    s: *Tensor,
    x: *Tensor,
    dt: *Tensor,
    a_decay: *Tensor,
    b_proj: *Tensor,
    c_proj: *Tensor,
    ids: *Tensor,
    k_snapshots: i64,
) *Tensor {
    impl.assert(k_snapshots >= 1, "K >= 1");
    impl.assert(k_snapshots <= std.math.maxInt(i32), "K <= INT32_MAX");
    impl.assert(types.ggml_is_contiguous(s), "ggml_is_contiguous(s)");
    impl.assert(types.ggml_is_contiguous(dt), "ggml_is_contiguous(dt)");
    impl.assert(types.ggml_is_contiguous(a_decay), "ggml_is_contiguous(A)");
    impl.assert(x.nb[0] == types.ggml_type_size(x.type), "x->nb[0] == ggml_type_size(x->type)");
    impl.assert(b_proj.nb[0] == types.ggml_type_size(b_proj.type), "B->nb[0] == ggml_type_size(B->type)");
    impl.assert(c_proj.nb[0] == types.ggml_type_size(c_proj.type), "C->nb[0] == ggml_type_size(C->type)");
    impl.assert(x.nb[1] == @as(usize, @intCast(x.ne[0])) * x.nb[0], "x->nb[1] == x->ne[0]*x->nb[0]");
    impl.assert(b_proj.nb[1] == @as(usize, @intCast(b_proj.ne[0])) * b_proj.nb[0], "B->nb[1] == B->ne[0]*B->nb[0]");
    impl.assert(c_proj.nb[1] == @as(usize, @intCast(c_proj.ne[0])) * c_proj.nb[0], "C->nb[1] == C->ne[0]*C->nb[0]");
    impl.assert(types.ggml_are_same_shape(b_proj, c_proj), "ggml_are_same_shape(B, C)");
    impl.assert(ids.type == c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    {
        const d_state = s.ne[0];
        const head_dim = x.ne[0];
        const n_head = x.ne[1];
        const n_seq_tokens = x.ne[2];
        const n_seqs = x.ne[3];

        impl.assert(dt.ne[0] == n_head, "dt->ne[0] == n_head");
        impl.assert(dt.ne[1] == n_seq_tokens, "dt->ne[1] == n_seq_tokens");
        impl.assert(dt.ne[2] == n_seqs, "dt->ne[2] == n_seqs");
        impl.assert(types.ggml_is_3d(dt), "ggml_is_3d(dt)");
        impl.assert(s.ne[1] == head_dim, "s->ne[1] == head_dim");
        impl.assert(s.ne[2] == n_head, "s->ne[2] == n_head");
        impl.assert(b_proj.ne[0] == d_state, "B->ne[0] == d_state");
        impl.assert(b_proj.ne[2] == n_seq_tokens, "B->ne[2] == n_seq_tokens");
        impl.assert(b_proj.ne[3] == n_seqs, "B->ne[3] == n_seqs");
        impl.assert(ids.ne[0] == n_seqs, "ids->ne[0] == n_seqs");
        impl.assert(types.ggml_is_vector(ids), "ggml_is_vector(ids)");
        impl.assert(a_decay.ne[1] == n_head, "A->ne[1] == n_head");
        impl.assert(types.ggml_is_matrix(a_decay), "ggml_is_matrix(A)");

        if (a_decay.ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            impl.assert(a_decay.ne[0] == d_state, "A->ne[0] == d_state");
            impl.assert(k_snapshots == 1, "K == 1");
        }
    }

    // concatenated y + ssm_states
    const n = types.ggml_nelements(x) + k_snapshots * s.ne[0] * s.ne[1] * s.ne[2] * ids.ne[0];
    const result = context.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, n);

    result.op = c.GGML_OP_SSM_SCAN;
    result.src[0] = s;
    result.src[1] = x;
    result.src[2] = dt;
    result.src[3] = a_decay;
    result.src[4] = b_proj;
    result.src[5] = c_proj;
    result.src[6] = ids;

    impl.setOpParamsI32(result, 0, @intCast(k_snapshots));

    return result;
}

// -----------------------------------------------------------------------------
// Linear-attention recurrences
//
// RWKV and the gated variants. All three share a result shape: the output rows
// and the new state rows concatenated into one 2d tensor, for the same reason
// the SSM ops do.

/// The `[S*H, n_tokens + S*n_seqs, 1, 1]` result these three share.
inline fn wkvResult(ctx: *Context, s: i64, h: i64, n_tokens: i64, n_seqs: i64) *Tensor {
    const ne = [4]i64{ s * h, n_tokens + s * n_seqs, 1, 1 };
    return context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);
}

/// Ports `ggml_rwkv_wkv6` (ggml.c:5801 @c1d0e7a00).
pub export fn ggml_rwkv_wkv6(
    ctx: *Context,
    k: *Tensor,
    v: *Tensor,
    r: *Tensor,
    tf: *Tensor,
    td: *Tensor,
    state: *Tensor,
) *Tensor {
    impl.assert(types.ggml_is_contiguous(k), "ggml_is_contiguous(k)");
    impl.assert(types.ggml_is_contiguous(v), "ggml_is_contiguous(v)");
    impl.assert(types.ggml_is_contiguous(r), "ggml_is_contiguous(r)");
    impl.assert(types.ggml_is_contiguous(tf), "ggml_is_contiguous(tf)");
    impl.assert(types.ggml_is_contiguous(td), "ggml_is_contiguous(td)");
    impl.assert(types.ggml_is_contiguous(state), "ggml_is_contiguous(state)");

    const s = k.ne[0];
    const h = k.ne[1];
    const n_tokens = k.ne[2];
    const n_seqs = state.ne[1];
    {
        impl.assert(v.ne[0] == s and v.ne[1] == h and v.ne[2] == n_tokens, "v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens");
        impl.assert(r.ne[0] == s and r.ne[1] == h and r.ne[2] == n_tokens, "r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens");
        impl.assert(td.ne[0] == s and td.ne[1] == h and td.ne[2] == n_tokens, "td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens");
        impl.assert(types.ggml_nelements(state) == s * s * h * n_seqs, "ggml_nelements(state) == S * S * H * n_seqs");
    }

    // concat output and new_state
    const result = wkvResult(ctx, s, h, n_tokens, n_seqs);

    result.op = c.GGML_OP_RWKV_WKV6;
    result.src[0] = k;
    result.src[1] = v;
    result.src[2] = r;
    result.src[3] = tf;
    result.src[4] = td;
    result.src[5] = state;

    return result;
}

/// Ports `ggml_gated_linear_attn` (ggml.c:5844 @c1d0e7a00).
pub export fn ggml_gated_linear_attn(
    ctx: *Context,
    k: *Tensor,
    v: *Tensor,
    q: *Tensor,
    g: *Tensor,
    state: *Tensor,
    scale: f32,
) *Tensor {
    impl.assert(types.ggml_is_contiguous(k), "ggml_is_contiguous(k)");
    impl.assert(types.ggml_is_contiguous(v), "ggml_is_contiguous(v)");
    impl.assert(types.ggml_is_contiguous(q), "ggml_is_contiguous(q)");
    impl.assert(types.ggml_is_contiguous(g), "ggml_is_contiguous(g)");
    impl.assert(types.ggml_is_contiguous(state), "ggml_is_contiguous(state)");

    const s = k.ne[0];
    const h = k.ne[1];
    const n_tokens = k.ne[2];
    const n_seqs = state.ne[1];
    {
        impl.assert(v.ne[0] == s and v.ne[1] == h and v.ne[2] == n_tokens, "v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens");
        impl.assert(q.ne[0] == s and q.ne[1] == h and q.ne[2] == n_tokens, "q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens");
        impl.assert(g.ne[0] == s and g.ne[1] == h and g.ne[2] == n_tokens, "g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens");
        impl.assert(types.ggml_nelements(state) == s * s * h * n_seqs, "ggml_nelements(state) == S * S * H * n_seqs");
    }

    // concat output and new_state
    const result = wkvResult(ctx, s, h, n_tokens, n_seqs);

    impl.setOpParamsF32(result, 0, scale);

    result.op = c.GGML_OP_GATED_LINEAR_ATTN;
    result.src[0] = k;
    result.src[1] = v;
    result.src[2] = q;
    result.src[3] = g;
    result.src[4] = state;

    return result;
}

/// Ports `ggml_rwkv_wkv7` (ggml.c:5887 @c1d0e7a00).
///
/// Note the source order: `r, w, k, v, a, b, state`, which is not the order
/// the shapes are checked in and not `wkv6`'s order either.
pub export fn ggml_rwkv_wkv7(
    ctx: *Context,
    r: *Tensor,
    w: *Tensor,
    k: *Tensor,
    v: *Tensor,
    a: *Tensor,
    b: *Tensor,
    state: *Tensor,
) *Tensor {
    impl.assert(types.ggml_is_contiguous(r), "ggml_is_contiguous(r)");
    impl.assert(types.ggml_is_contiguous(w), "ggml_is_contiguous(w)");
    impl.assert(types.ggml_is_contiguous(k), "ggml_is_contiguous(k)");
    impl.assert(types.ggml_is_contiguous(v), "ggml_is_contiguous(v)");
    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
    impl.assert(types.ggml_is_contiguous(b), "ggml_is_contiguous(b)");
    impl.assert(types.ggml_is_contiguous(state), "ggml_is_contiguous(state)");

    const s = k.ne[0];
    const h = k.ne[1];
    const n_tokens = k.ne[2];
    const n_seqs = state.ne[1];
    {
        impl.assert(w.ne[0] == s and w.ne[1] == h and w.ne[2] == n_tokens, "w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens");
        impl.assert(k.ne[0] == s and k.ne[1] == h and k.ne[2] == n_tokens, "k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens");
        impl.assert(v.ne[0] == s and v.ne[1] == h and v.ne[2] == n_tokens, "v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens");
        impl.assert(a.ne[0] == s and a.ne[1] == h and a.ne[2] == n_tokens, "a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens");
        impl.assert(b.ne[0] == s and b.ne[1] == h and b.ne[2] == n_tokens, "b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens");
        impl.assert(types.ggml_nelements(state) == s * s * h * n_seqs, "ggml_nelements(state) == S * S * H * n_seqs");
    }

    // concat output and new_state
    const result = wkvResult(ctx, s, h, n_tokens, n_seqs);

    result.op = c.GGML_OP_RWKV_WKV7;
    result.src[0] = r;
    result.src[1] = w;
    result.src[2] = k;
    result.src[3] = v;
    result.src[4] = a;
    result.src[5] = b;
    result.src[6] = state;

    return result;
}

/// Ports `ggml_gated_delta_net` (ggml.c:6294 @c1d0e7a00).
///
/// `state` holds only the initial state; `K` snapshot slots are appended to
/// the result, as in `ggml_ssm_scan`.
pub export fn ggml_gated_delta_net(
    ctx: *Context,
    q: *Tensor,
    k: *Tensor,
    v: *Tensor,
    g: *Tensor,
    beta: *Tensor,
    state: *Tensor,
    k_snapshots: i64,
) *Tensor {
    impl.assert(types.ggml_is_contiguous_rows(q), "ggml_is_contiguous_rows(q)");
    impl.assert(types.ggml_is_contiguous_rows(k), "ggml_is_contiguous_rows(k)");
    impl.assert(types.ggml_is_contiguous_rows(v), "ggml_is_contiguous_rows(v)");
    impl.assert(types.ggml_is_contiguous(g), "ggml_is_contiguous(g)");
    impl.assert(types.ggml_is_contiguous(beta), "ggml_is_contiguous(beta)");
    impl.assert(types.ggml_is_contiguous(state), "ggml_is_contiguous(state)");

    impl.assert(q.type == c.GGML_TYPE_F32, "q->type == GGML_TYPE_F32");
    impl.assert(k.type == c.GGML_TYPE_F32, "k->type == GGML_TYPE_F32");
    impl.assert(v.type == c.GGML_TYPE_F32, "v->type == GGML_TYPE_F32");
    impl.assert(g.type == c.GGML_TYPE_F32, "g->type == GGML_TYPE_F32");
    impl.assert(beta.type == c.GGML_TYPE_F32, "beta->type == GGML_TYPE_F32");
    impl.assert(state.type == c.GGML_TYPE_F32, "state->type == GGML_TYPE_F32");

    const s_v = v.ne[0];
    const h = v.ne[1];
    const n_tokens = v.ne[2];
    const n_seqs = v.ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    impl.assert(g.ne[0] == 1 or g.ne[0] == s_v, "g->ne[0] == 1 || g->ne[0] == S_v");
    impl.assert(beta.ne[0] == 1, "beta->ne[0] == 1");

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs].
    impl.assert(state.ne[0] == s_v, "state->ne[0] == S_v");
    impl.assert(state.ne[1] == s_v, "state->ne[1] == S_v");
    impl.assert(state.ne[2] == h, "state->ne[2] == H");
    impl.assert(state.ne[3] == n_seqs, "state->ne[3] == n_seqs");
    impl.assert(k_snapshots >= 1, "K >= 1");

    const state_rows = k_snapshots * s_v * n_seqs;
    const ne = [4]i64{ s_v * h, n_tokens * n_seqs + state_rows, 1, 1 };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);

    impl.setOpParamsI32(result, 0, @intCast(k_snapshots));

    result.op = c.GGML_OP_GATED_DELTA_NET;
    result.src[0] = q;
    result.src[1] = k;
    result.src[2] = v;
    result.src[3] = g;
    result.src[4] = beta;
    result.src[5] = state;

    return result;
}

/// Ports `ggml_lightning_indexer` (ggml.c:6351 @c1d0e7a00).
///
/// Scores every key against every query for sparse-attention selection, so the
/// result is `[n_kv, n_tokens, 1, n_batch]` rather than an attention output.
pub export fn ggml_lightning_indexer(
    ctx: *Context,
    q: *Tensor,
    k: *Tensor,
    weights: *Tensor,
    mask: *Tensor,
) *Tensor {
    impl.assert(q.type == c.GGML_TYPE_F32, "q->type == GGML_TYPE_F32");
    impl.assert(weights.type == c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    impl.assert(mask.type == c.GGML_TYPE_F16, "mask->type == GGML_TYPE_F16");
    impl.assert(q.ne[0] == k.ne[0], "q->ne[0] == k->ne[0]");
    impl.assert(mask.ne[0] == k.ne[2], "mask->ne[0] == k->ne[2]");
    impl.assert(q.ne[1] == weights.ne[0], "q->ne[1] == weights->ne[0]");
    impl.assert(k.ne[1] == 1, "k->ne[1] == 1");
    impl.assert(mask.ne[1] == q.ne[2], "mask->ne[1] == q->ne[2]");
    impl.assert(q.ne[2] == weights.ne[1], "q->ne[2] == weights->ne[1]");
    impl.assert(weights.ne[2] == 1, "weights->ne[2] == 1");
    impl.assert(mask.ne[2] == 1, "mask->ne[2] == 1");
    impl.assert(q.ne[3] == k.ne[3], "q->ne[3] == k->ne[3]");
    impl.assert(k.ne[3] == weights.ne[3], "k->ne[3] == weights->ne[3]");
    impl.assert(@rem(weights.ne[3], mask.ne[3]) == 0, "weights->ne[3] % mask->ne[3] == 0");

    const ne = [4]i64{ k.ne[2], q.ne[2], 1, q.ne[3] };
    const result = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, 4, &ne);

    result.op = c.GGML_OP_LIGHTNING_INDEXER;
    result.src[0] = q;
    result.src[1] = k;
    result.src[2] = weights;
    result.src[3] = mask;

    return result;
}

// -----------------------------------------------------------------------------
// DeepSeek-V4 hyper-connections
//
// A residual stream carried at `hc` parallel widths instead of one. `pre`
// collapses the widths going into a block, `comb` builds the mixing matrix,
// and `post` expands back out.

/// Ports `ggml_dsv4_hc_comb` (ggml.c:6387 @c1d0e7a00).
///
/// `hc` is recovered from the input width by solving `(2 + i)*i == hc_mix_dim`
/// -- the mixes tensor packs an `hc x hc` matrix plus two `hc`-wide vectors --
/// and the C then asserts the answer is 4, so only that width works today.
pub export fn ggml_dsv4_hc_comb(
    ctx: *Context,
    mixes: *Tensor,
    scale: *Tensor,
    base: *Tensor,
    eps: f32,
    n_iter: i32,
) *Tensor {
    impl.assert(mixes.type == c.GGML_TYPE_F32, "mixes->type == GGML_TYPE_F32");
    impl.assert(scale.type == c.GGML_TYPE_F32, "scale->type == GGML_TYPE_F32");
    impl.assert(base.type == c.GGML_TYPE_F32, "base->type == GGML_TYPE_F32");
    impl.assert(n_iter > 0, "n_iter > 0");

    const hc_mix_dim = mixes.ne[0];
    const n_tokens = mixes.ne[1];

    var hc: i64 = 0;
    var i: i64 = 1;
    while (i * i + 2 * i <= hc_mix_dim) : (i += 1) {
        if ((2 + i) * i == hc_mix_dim) {
            hc = i;
            break;
        }
    }

    impl.assert(hc > 0, "hc > 0");
    impl.assert(hc == 4, "hc == 4");
    impl.assert(mixes.ne[2] == 1, "mixes->ne[2] == 1");
    impl.assert(mixes.ne[3] == 1, "mixes->ne[3] == 1");
    impl.assert(scale.ne[0] >= 3, "scale->ne[0] >= 3");
    impl.assert(scale.ne[1] == 1, "scale->ne[1] == 1");
    impl.assert(scale.ne[2] == 1, "scale->ne[2] == 1");
    impl.assert(scale.ne[3] == 1, "scale->ne[3] == 1");
    impl.assert(base.ne[0] == hc_mix_dim, "base->ne[0] == hc_mix_dim");
    impl.assert(base.ne[1] == 1, "base->ne[1] == 1");
    impl.assert(base.ne[2] == 1, "base->ne[2] == 1");
    impl.assert(base.ne[3] == 1, "base->ne[3] == 1");

    const result = context.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, hc, hc, n_tokens);

    impl.setOpParamsF32(result, 0, eps);
    impl.setOpParamsI32(result, 1, n_iter);

    result.op = c.GGML_OP_DSV4_HC_COMB;
    result.src[0] = mixes;
    result.src[1] = scale;
    result.src[2] = base;

    return result;
}

/// Ports `ggml_dsv4_hc_pre` (ggml.c:6438 @c1d0e7a00).
pub export fn ggml_dsv4_hc_pre(ctx: *Context, x: *Tensor, weights: *Tensor) *Tensor {
    impl.assert(x.type == c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    impl.assert(weights.type == c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");

    const n_embd = x.ne[0];
    const hc = x.ne[1];
    const n_tokens = x.ne[2];

    impl.assert(hc > 0, "hc > 0");
    impl.assert(x.ne[3] == 1, "x->ne[3] == 1");
    impl.assert(weights.ne[0] == hc, "weights->ne[0] == hc");
    impl.assert(weights.ne[1] == n_tokens, "weights->ne[1] == n_tokens");
    impl.assert(weights.ne[2] == 1, "weights->ne[2] == 1");
    impl.assert(weights.ne[3] == 1, "weights->ne[3] == 1");

    const result = context.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd, n_tokens);

    result.op = c.GGML_OP_DSV4_HC_PRE;
    result.src[0] = x;
    result.src[1] = weights;

    return result;
}

/// Ports `ggml_dsv4_hc_post` (ggml.c:6467 @c1d0e7a00).
pub export fn ggml_dsv4_hc_post(
    ctx: *Context,
    x: *Tensor,
    residual: *Tensor,
    post: *Tensor,
    comb: *Tensor,
) *Tensor {
    impl.assert(x.type == c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    impl.assert(residual.type == c.GGML_TYPE_F32, "residual->type == GGML_TYPE_F32");
    impl.assert(post.type == c.GGML_TYPE_F32, "post->type == GGML_TYPE_F32");
    impl.assert(comb.type == c.GGML_TYPE_F32, "comb->type == GGML_TYPE_F32");

    const n_embd = x.ne[0];
    const n_tokens = x.ne[1];
    const hc = residual.ne[1];

    impl.assert(hc > 0, "hc > 0");
    impl.assert(x.ne[2] == 1, "x->ne[2] == 1");
    impl.assert(x.ne[3] == 1, "x->ne[3] == 1");

    impl.assert(residual.ne[0] == n_embd, "residual->ne[0] == n_embd");
    impl.assert(residual.ne[2] == n_tokens, "residual->ne[2] == n_tokens");
    impl.assert(residual.ne[3] == 1, "residual->ne[3] == 1");

    impl.assert(post.ne[0] == hc, "post->ne[0] == hc");
    impl.assert(post.ne[1] == n_tokens, "post->ne[1] == n_tokens");
    impl.assert(post.ne[2] == 1, "post->ne[2] == 1");
    impl.assert(post.ne[3] == 1, "post->ne[3] == 1");

    impl.assert(comb.ne[0] == hc, "comb->ne[0] == hc");
    impl.assert(comb.ne[1] == hc, "comb->ne[1] == hc");
    impl.assert(comb.ne[2] == n_tokens, "comb->ne[2] == n_tokens");
    impl.assert(comb.ne[3] == 1, "comb->ne[3] == 1");

    const result = context.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, n_embd, hc, n_tokens);

    result.op = c.GGML_OP_DSV4_HC_POST;
    result.src[0] = x;
    result.src[1] = residual;
    result.src[2] = post;
    result.src[3] = comb;

    return result;
}

// -----------------------------------------------------------------------------
// Caller-supplied kernels
//
// An escape hatch: the caller hands over a function pointer and the CPU
// backend calls it during evaluation. The pointer, a task count, and an opaque
// userdata pointer are stashed in `op_params`, which is why these are the only
// ops whose parameters are not plain numbers.

/// The shared body of `ggml_map_custom1`, `_2`, and `_3` (ggml.c:5968, 6012,
/// 6060).
///
/// The three C functions differ only in how many sources they wire up and
/// which `ggml_map_customN_op_params` they write. Those three structs have
/// identical layout -- a function pointer, an `int`, and a `void *` -- so one
/// generic body covers all three, with `srcs` supplying the difference.
inline fn mapCustomImpl(
    ctx: *Context,
    srcs: []const *Tensor,
    op: c.enum_ggml_op,
    fun: ?*const anyopaque,
    n_tasks: c_int,
    userdata: ?*anyopaque,
    inplace: bool,
) *Tensor {
    impl.assert(n_tasks == c.GGML_N_TASKS_MAX or n_tasks > 0, "n_tasks == GGML_N_TASKS_MAX || n_tasks > 0");

    const result = dupOrView(ctx, srcs[0], inplace);

    impl.setOpParamsValue(result, impl.CustomOpParams{
        .fun = fun,
        .n_tasks = n_tasks,
        .userdata = userdata,
    });

    result.op = op;
    for (srcs, 0..) |src, i| result.src[i] = src;

    return result;
}

/// Ports `ggml_map_custom1` (ggml.c:5992 @c1d0e7a00).
pub export fn ggml_map_custom1(
    ctx: *Context,
    a: *Tensor,
    fun: c.ggml_custom1_op_t,
    n_tasks: c_int,
    userdata: ?*anyopaque,
) *Tensor {
    return mapCustomImpl(ctx, &.{a}, c.GGML_OP_MAP_CUSTOM1, @ptrCast(fun), n_tasks, userdata, false);
}

/// Ports `ggml_map_custom1_inplace` (ggml.c:6001 @c1d0e7a00).
pub export fn ggml_map_custom1_inplace(
    ctx: *Context,
    a: *Tensor,
    fun: c.ggml_custom1_op_t,
    n_tasks: c_int,
    userdata: ?*anyopaque,
) *Tensor {
    return mapCustomImpl(ctx, &.{a}, c.GGML_OP_MAP_CUSTOM1, @ptrCast(fun), n_tasks, userdata, true);
}

/// Ports `ggml_map_custom2` (ggml.c:6038 @c1d0e7a00).
pub export fn ggml_map_custom2(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    fun: c.ggml_custom2_op_t,
    n_tasks: c_int,
    userdata: ?*anyopaque,
) *Tensor {
    return mapCustomImpl(ctx, &.{ a, b }, c.GGML_OP_MAP_CUSTOM2, @ptrCast(fun), n_tasks, userdata, false);
}

/// Ports `ggml_map_custom2_inplace` (ggml.c:6048 @c1d0e7a00).
pub export fn ggml_map_custom2_inplace(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    fun: c.ggml_custom2_op_t,
    n_tasks: c_int,
    userdata: ?*anyopaque,
) *Tensor {
    return mapCustomImpl(ctx, &.{ a, b }, c.GGML_OP_MAP_CUSTOM2, @ptrCast(fun), n_tasks, userdata, true);
}

/// Ports `ggml_map_custom3` (ggml.c:6088 @c1d0e7a00).
pub export fn ggml_map_custom3(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    c_arg: *Tensor,
    fun: c.ggml_custom3_op_t,
    n_tasks: c_int,
    userdata: ?*anyopaque,
) *Tensor {
    return mapCustomImpl(ctx, &.{ a, b, c_arg }, c.GGML_OP_MAP_CUSTOM3, @ptrCast(fun), n_tasks, userdata, false);
}

/// Ports `ggml_map_custom3_inplace` (ggml.c:6099 @c1d0e7a00).
pub export fn ggml_map_custom3_inplace(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    c_arg: *Tensor,
    fun: c.ggml_custom3_op_t,
    n_tasks: c_int,
    userdata: ?*anyopaque,
) *Tensor {
    return mapCustomImpl(ctx, &.{ a, b, c_arg }, c.GGML_OP_MAP_CUSTOM3, @ptrCast(fun), n_tasks, userdata, true);
}

/// Ports `ggml_custom_4d` (ggml.c:6110 @c1d0e7a00).
///
/// The general form: the caller states the result shape outright rather than
/// inheriting it from an input, and passes its sources as an array.
///
/// Parameters:
/// - `type`, `ne0`..`ne3`: the result's type and shape.
/// - `args`: `n_args` sources, wired to `src[0..n_args]`.
/// - `fun`, `n_tasks`, `userdata`: the kernel and how to schedule it.
///
/// Return: a new tensor owned by `ctx`.
pub export fn ggml_custom_4d(
    ctx: *Context,
    t: c.enum_ggml_type,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    args: [*c]?*Tensor,
    n_args: c_int,
    fun: c.ggml_custom_op_t,
    n_tasks: c_int,
    userdata: ?*anyopaque,
) *Tensor {
    impl.assert(n_args < c.GGML_MAX_SRC, "n_args < GGML_MAX_SRC");

    const result = context.ggml_new_tensor_4d(ctx, t, ne0, ne1, ne2, ne3);

    impl.setOpParamsValue(result, impl.CustomOpParams{
        .fun = @ptrCast(fun),
        .n_tasks = n_tasks,
        .userdata = userdata,
    });

    result.op = c.GGML_OP_CUSTOM;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n_args))) : (i += 1) {
        result.src[i] = args[i];
    }

    return result;
}

/// Ports `ggml_custom_inplace` (ggml.c:6142 @c1d0e7a00).
///
/// Note the source shift: `a` takes `src[0]` and the extra arguments start at
/// `src[1]`, which is why the bound is one lower than `ggml_custom_4d`'s.
pub export fn ggml_custom_inplace(
    ctx: *Context,
    a: *Tensor,
    args: [*c]?*Tensor,
    n_args: c_int,
    fun: c.ggml_custom_op_t,
    n_tasks: c_int,
    userdata: ?*anyopaque,
) *Tensor {
    impl.assert(n_args < c.GGML_MAX_SRC - 1, "n_args < GGML_MAX_SRC - 1");

    const result = context.ggml_view_tensor(ctx, a);

    impl.setOpParamsValue(result, impl.CustomOpParams{
        .fun = @ptrCast(fun),
        .n_tasks = n_tasks,
        .userdata = userdata,
    });

    result.op = c.GGML_OP_CUSTOM;
    result.src[0] = a;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n_args))) : (i += 1) {
        result.src[i + 1] = args[i];
    }

    return result;
}

// -----------------------------------------------------------------------------
// Training
//
// Loss and optimiser steps. These exist for ggml's own training support and
// are not reached by inference.

/// Ports `ggml_cross_entropy_loss` (ggml.c:6172 @c1d0e7a00).
///
/// Reduces to a single scalar, which is what `ggml_set_loss` then marks.
pub export fn ggml_cross_entropy_loss(ctx: *Context, a: *Tensor, b: *Tensor) *Tensor {
    impl.assert(types.ggml_are_same_shape(a, b), "ggml_are_same_shape(a, b)");

    const result = context.ggml_new_tensor_1d(ctx, a.type, 1);

    result.op = c.GGML_OP_CROSS_ENTROPY_LOSS;
    result.src[0] = a;
    result.src[1] = b;

    return result;
}

/// Ports `ggml_cross_entropy_loss_back` (ggml.c:6189 @c1d0e7a00).
///
/// `a` is the incoming scalar gradient; the result has `b`'s shape.
pub export fn ggml_cross_entropy_loss_back(ctx: *Context, a: *Tensor, b: *Tensor, c_arg: *Tensor) *Tensor {
    impl.assert(types.ggml_is_scalar(a), "ggml_is_scalar(a)");
    impl.assert(types.ggml_are_same_shape(b, c_arg), "ggml_are_same_shape(b, c)");

    const result = context.ggml_dup_tensor(ctx, b);

    result.op = c.GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result.src[0] = a;
    result.src[1] = b;
    result.src[2] = c_arg;

    return result;
}

/// Ports `ggml_opt_step_adamw` (ggml.c:6209 @c1d0e7a00).
///
/// Updates `a` in place, so the result is a view of it rather than new
/// storage. `adamw_params` is a 7-element f32 vector holding the learning
/// rate, the two betas, epsilon, weight decay, and the two bias corrections.
pub export fn ggml_opt_step_adamw(
    ctx: *Context,
    a: *Tensor,
    grad: *Tensor,
    m: *Tensor,
    v: *Tensor,
    adamw_params: *Tensor,
) *Tensor {
    impl.assert((a.flags & c.GGML_TENSOR_FLAG_PARAM) != 0, "a->flags & GGML_TENSOR_FLAG_PARAM");
    impl.assert(types.ggml_are_same_shape(a, grad), "ggml_are_same_shape(a, grad)");
    impl.assert(types.ggml_are_same_shape(a, m), "ggml_are_same_shape(a, m)");
    impl.assert(types.ggml_are_same_shape(a, v), "ggml_are_same_shape(a, v)");
    impl.assert(adamw_params.type == c.GGML_TYPE_F32, "adamw_params->type == GGML_TYPE_F32");
    impl.assert(types.ggml_nelements(adamw_params) == 7, "ggml_nelements(adamw_params) == 7");

    const result = context.ggml_view_tensor(ctx, a);

    result.op = c.GGML_OP_OPT_STEP_ADAMW;
    result.src[0] = a;
    result.src[1] = grad;
    result.src[2] = m;
    result.src[3] = v;
    result.src[4] = adamw_params;

    return result;
}

/// Ports `ggml_opt_step_sgd` (ggml.c:6237 @c1d0e7a00).
///
/// `params` is a 2-element f32 vector: learning rate and weight decay.
pub export fn ggml_opt_step_sgd(ctx: *Context, a: *Tensor, grad: *Tensor, params: *Tensor) *Tensor {
    impl.assert((a.flags & c.GGML_TENSOR_FLAG_PARAM) != 0, "a->flags & GGML_TENSOR_FLAG_PARAM");
    impl.assert(types.ggml_are_same_shape(a, grad), "ggml_are_same_shape(a, grad)");
    impl.assert(params.type == c.GGML_TYPE_F32, "params->type == GGML_TYPE_F32");
    impl.assert(types.ggml_nelements(params) == 2, "ggml_nelements(params) == 2");

    const result = context.ggml_view_tensor(ctx, a);

    result.op = c.GGML_OP_OPT_STEP_SGD;
    result.src[0] = a;
    result.src[1] = grad;
    result.src[2] = params;

    return result;
}

// -----------------------------------------------------------------------------
// Triangular solve

/// Ports `ggml_solve_tri` (ggml.c:6259 @c1d0e7a00).
///
/// Solves `A X = B` for `X` with `A` triangular. The three booleans describe
/// which variant, but the C asserts `lower && left && !uni`, so only the
/// lower-triangular left-sided non-unit case is actually supported -- the
/// other combinations abort. The parameters are not written to `op_params`:
/// there is only one legal setting, so a backend has nothing to read.
pub export fn ggml_solve_tri(
    ctx: *Context,
    a: *Tensor,
    b: *Tensor,
    left: bool,
    lower: bool,
    uni: bool,
) *Tensor {
    impl.assert(a.type == c.GGML_TYPE_F32, "a->type == GGML_TYPE_F32");
    impl.assert(b.type == c.GGML_TYPE_F32, "b->type == GGML_TYPE_F32");

    // A must be square and lower diagonal
    impl.assert(a.ne[0] == a.ne[1], "a->ne[0] == a->ne[1]");
    // B must have same outer dimension as A
    impl.assert(a.ne[1] == b.ne[1], "a->ne[1] == b->ne[1]");

    // batch dimensions must be equal
    impl.assert(a.ne[2] == b.ne[2], "a->ne[2] == b->ne[2]");
    impl.assert(a.ne[3] == b.ne[3], "a->ne[3] == b->ne[3]");

    impl.assert(types.ggml_is_contiguous(a), "ggml_is_contiguous(a)");
    impl.assert(types.ggml_is_contiguous(b), "ggml_is_contiguous(b)");

    impl.assert(lower and left and !uni, "lower && left && !uni");

    const result = context.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, b.ne[0], b.ne[1], b.ne[2], b.ne[3]);

    result.op = c.GGML_OP_SOLVE_TRI;
    result.src[0] = a;
    result.src[1] = b;

    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests for sorting, generation, and flags

test "argsort returns indices, top_k narrows them" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const logits = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 32000, 4);

    const sorted = ggml_argsort(&ctx, logits, c.GGML_SORT_ORDER_DESC);
    // Indices, so i32 regardless of input type, and the same shape.
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_I32), sorted.type);
    try std.testing.expect(types.ggml_are_same_shape(sorted, logits));
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_SORT_ORDER_DESC)), impl.getOpParamsI32(sorted, 0));

    // The dedicated op produces a narrow result directly.
    const top = ggml_top_k(&ctx, logits, 40);
    try std.testing.expectEqual(@as(i64, 40), top.ne[0]);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_TOP_K), top.op);

    // The composite sorts everything, then views the head of each row.
    const via_sort = ggml_argsort_top_k(&ctx, logits, 40);
    try std.testing.expectEqual(@as(i64, 40), via_sort.ne[0]);
    try std.testing.expect(types.ggml_is_view(via_sort));
    // A view, so the row stride is still the full sorted row.
    try std.testing.expectEqual(sorted.nb[1], via_sort.nb[1]);
}

test "arange generates without a source" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // ceil((10 - 0) / 2.5) = 4
    const r = ggml_arange(&ctx, 0.0, 10.0, 2.5);
    try std.testing.expectEqual(@as(i64, 4), r.ne[0]);
    try std.testing.expect(r.src[0] == null); // the only op with no input
    try std.testing.expectEqual(@as(f32, 0.0), impl.getOpParamsF32(r, 0));
    try std.testing.expectEqual(@as(f32, 10.0), impl.getOpParamsF32(r, 1));
    try std.testing.expectEqual(@as(f32, 2.5), impl.getOpParamsF32(r, 2));

    // A non-dividing step rounds up rather than truncating.
    try std.testing.expectEqual(@as(i64, 4), ggml_arange(&ctx, 0.0, 10.0, 3.0).ne[0]);
}

test "roll and tri keep their shape" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 16, 8);
    const rolled = ggml_roll(&ctx, a, 3, -2, 0, 0);
    try std.testing.expect(types.ggml_are_same_shape(rolled, a));
    try std.testing.expectEqual(@as(i32, 3), impl.getOpParamsI32(rolled, 0));
    // Negative shifts are legal and stored as given.
    try std.testing.expectEqual(@as(i32, -2), impl.getOpParamsI32(rolled, 1));

    // tri needs a square matrix.
    const sq = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 8);
    const t = ggml_tri(&ctx, sq, c.GGML_TRI_TYPE_LOWER);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_TRI), t.op);
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_TRI_TYPE_LOWER)), impl.getOpParamsI32(t, 0));
}

test "fill and leaky_relu carry a float parameter" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 16);

    const f = ggml_fill(&ctx, a, -1.5);
    try std.testing.expectEqual(@as(f32, -1.5), impl.getOpParamsF32(f, 0));
    try std.testing.expect(types.ggml_is_view(ggml_fill_inplace(&ctx, a, 0)));

    // leaky_relu has its own op code, unlike the parameterless activations.
    const lr = ggml_leaky_relu(&ctx, a, 0.01, false);
    try std.testing.expectEqual(@as(c.enum_ggml_op, c.GGML_OP_LEAKY_RELU), lr.op);
    try std.testing.expectEqual(@as(f32, 0.01), impl.getOpParamsF32(lr, 0));
}

test "swiglu_oai adds its parameters above the glu slots" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);
    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 16, 4);
    const b = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 16, 4);

    const r = ggml_swiglu_oai(&ctx, a, b, 1.702, 7.0);
    // Slots 0 and 1 belong to the GLU op and swap flag.
    try std.testing.expectEqual(@as(i32, @intCast(c.GGML_GLU_OP_SWIGLU_OAI)), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i32, 0), impl.getOpParamsI32(r, 1));
    try std.testing.expectEqual(@as(f32, 1.702), impl.getOpParamsF32(r, 2));
    try std.testing.expectEqual(@as(f32, 7.0), impl.getOpParamsF32(r, 3));
}

test "win_part pads up to whole windows" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // [C, H, W, 1] = [64, 14, 14, 1] split into 8x8 windows needs 2x2 of them.
    const a = context.ggml_new_tensor_4d(&ctx, c.GGML_TYPE_F32, 64, 14, 14, 1);
    const r = ggml_win_part(&ctx, a, 8);
    try std.testing.expectEqual(@as(i64, 64), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[1]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[2]);
    try std.testing.expectEqual(@as(i64, 4), r.ne[3]); // 2x2 windows
    try std.testing.expectEqual(@as(i32, 2), impl.getOpParamsI32(r, 0));
    try std.testing.expectEqual(@as(i32, 8), impl.getOpParamsI32(r, 2));

    // The original size cannot be derived from the windows, so it is passed in.
    const back = ggml_win_unpart(&ctx, r, 14, 14, 8);
    try std.testing.expectEqual(@as(i64, 14), back.ne[1]);
    try std.testing.expectEqual(@as(i64, 14), back.ne[2]);
}

test "relative position embeddings expand into a query-key matrix" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    // A table of 2*8-1 = 15 offsets, each 64 wide.
    const table = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F16, 64, 15);
    const r = ggml_get_rel_pos(&ctx, table, 8, 8);
    try std.testing.expectEqual(@as(i64, 64), r.ne[0]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[1]);
    try std.testing.expectEqual(@as(i64, 8), r.ne[2]);
    try std.testing.expectEqual(@as(c.enum_ggml_type, c.GGML_TYPE_F16), r.type);
}

test "flags mark a tensor's role, and output walks the view chain" {
    var buf: [1 << 16]u8 align(16) = undefined;
    var ctx = testContext(&buf);

    const a = context.ggml_new_tensor_2d(&ctx, c.GGML_TYPE_F32, 8, 4);
    ggml_set_input(a);
    try std.testing.expect((a.flags & c.GGML_TENSOR_FLAG_INPUT) != 0);

    // Marking a view as output must also protect what it views into,
    // or the allocator would reuse the underlying storage.
    const view = ggml_reshape_2d(&ctx, a, 4, 8);
    ggml_set_output(view);
    try std.testing.expect((view.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0);
    try std.testing.expect((a.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0);

    // Only a leaf can be a parameter.
    const leaf = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 4);
    ggml_set_param(leaf);
    try std.testing.expect((leaf.flags & c.GGML_TENSOR_FLAG_PARAM) != 0);

    // A loss must be a scalar f32.
    const loss = context.ggml_new_tensor_1d(&ctx, c.GGML_TYPE_F32, 1);
    ggml_set_loss(loss);
    try std.testing.expect((loss.flags & c.GGML_TENSOR_FLAG_LOSS) != 0);
}
