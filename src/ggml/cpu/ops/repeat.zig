//! `repeat`, `repeat_back` and `concat`: tiling a tensor, summing the tiles
//! back, and joining two tensors along one dimension.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # These are byte moves, typed only by width
//!
//! `repeat` sends F16, BF16 and I16 through its `_f16` kernel and F32 and I32
//! through its `_f32` one; `concat` does the same and adds I8. Nothing is
//! converted, so the element types below are chosen by width alone — `u16`
//! where the C says `ggml_fp16_t`, `f32` where it says `float` — and a value
//! is only ever loaded and stored.
//!
//! # Loop index names
//!
//! `i0`…`i3`, `i00`…`i13` are Zig integer type names. Renamed `j0`…`j3`,
//! `j00`…`j13`, digit for digit — the `cpu/mulmat.zig` convention. Do not
//! renumber them.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

/// A non-negative index times a byte stride: the C's `int64_t * size_t`,
/// which converts the index to unsigned. Every index these kernels form is
/// non-negative, so the `@intCast` checks what the C assumes.
inline fn at(i: i64, nb: usize) usize {
    return @as(usize, @intCast(i)) * nb;
}

/// Ports `ggml_compute_forward_repeat_f32` (ops.cpp:1698 @c1d0e7a00) and
/// `ggml_compute_forward_repeat_f16` (ops.cpp:1742 @c1d0e7a00).
///
/// The two are the same loop nest over a different element width; the `f16`
/// one inlines the row copy where the `f32` one calls `ggml_vec_cpy_f32`. Both
/// are a plain element copy, so one comptime function covers them.
fn repeatT(comptime T: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(c.ggml_can_repeat(src0, dst), "ggml_can_repeat(src0, dst)");

    const l = common.UnaryLocals.of(src0, dst);

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const nr0: i64 = @divTrunc(l.ne0, l.ne00);
    const nr1: i64 = @divTrunc(l.ne1, l.ne01);
    const nr2: i64 = @divTrunc(l.ne2, l.ne02);
    const nr3: i64 = @divTrunc(l.ne3, l.ne03);

    // TODO: support for transposed / permuted tensors
    impl.assert(l.nb0 == @sizeOf(T), "nb0 == sizeof(T)");
    impl.assert(l.nb00 == @sizeOf(T), "nb00 == sizeof(T)");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    // TODO: maybe this is not optimal?
    var j3: i64 = 0;
    while (j3 < nr3) : (j3 += 1) {
        var k3: i64 = 0;
        while (k3 < l.ne03) : (k3 += 1) {
            var j2: i64 = 0;
            while (j2 < nr2) : (j2 += 1) {
                var k2: i64 = 0;
                while (k2 < l.ne02) : (k2 += 1) {
                    var j1: i64 = 0;
                    while (j1 < nr1) : (j1 += 1) {
                        var k1: i64 = 0;
                        while (k1 < l.ne01) : (k1 += 1) {
                            var j0: i64 = 0;
                            while (j0 < nr0) : (j0 += 1) {
                                const y: [*]T = @ptrCast(@alignCast(dd + at(j3 * l.ne03 + k3, l.nb3) + at(j2 * l.ne02 + k2, l.nb2) + at(j1 * l.ne01 + k1, l.nb1) + at(j0 * l.ne00, l.nb0)));
                                const x: [*]const T = @ptrCast(@alignCast(s0 + at(k3, l.nb03) + at(k2, l.nb02) + at(k1, l.nb01)));
                                if (T == f32) {
                                    vec.cpy_f32(l.ne00, y, x);
                                } else {
                                    // ggml_vec_cpy_f16(ne00, y, x)
                                    var i: usize = 0;
                                    while (i < @as(usize, @intCast(l.ne00))) : (i += 1) y[i] = x[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_repeat` (ops.cpp:1789 @c1d0e7a00).
pub export fn ggml_compute_forward_repeat(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F16, c.GGML_TYPE_BF16, c.GGML_TYPE_I16 => repeatT(c.ggml_fp16_t, params, dst),
        c.GGML_TYPE_F32, c.GGML_TYPE_I32 => repeatT(f32, params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_repeat_back_f32` (ops.cpp:1822 @c1d0e7a00).
fn repeatBackF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(c.ggml_can_repeat(dst, src0), "ggml_can_repeat(dst, src0)");

    const l = common.UnaryLocals.of(src0, dst);

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const nr0: i64 = @divTrunc(l.ne00, l.ne0);
    const nr1: i64 = @divTrunc(l.ne01, l.ne1);
    const nr2: i64 = @divTrunc(l.ne02, l.ne2);
    const nr3: i64 = @divTrunc(l.ne03, l.ne3);

    // TODO: support for transposed / permuted tensors
    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    if (c.ggml_is_contiguous(dst)) {
        vec.set_f32(l.ne0 * l.ne1 * l.ne2 * l.ne3, @ptrCast(@alignCast(dd)), 0);
    } else {
        var k3: i64 = 0;
        while (k3 < l.ne3) : (k3 += 1) {
            var k2: i64 = 0;
            while (k2 < l.ne2) : (k2 += 1) {
                var k1: i64 = 0;
                while (k1 < l.ne1) : (k1 += 1) {
                    vec.set_f32(l.ne0, @ptrCast(@alignCast(dd + at(k1, l.nb1) + at(k2, l.nb2) + at(k3, l.nb3))), 0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    var j3: i64 = 0;
    while (j3 < nr3) : (j3 += 1) {
        var k3: i64 = 0;
        while (k3 < l.ne3) : (k3 += 1) {
            var j2: i64 = 0;
            while (j2 < nr2) : (j2 += 1) {
                var k2: i64 = 0;
                while (k2 < l.ne2) : (k2 += 1) {
                    var j1: i64 = 0;
                    while (j1 < nr1) : (j1 += 1) {
                        var k1: i64 = 0;
                        while (k1 < l.ne1) : (k1 += 1) {
                            var j0: i64 = 0;
                            while (j0 < nr0) : (j0 += 1) {
                                vec.acc_f32(
                                    l.ne0,
                                    @ptrCast(@alignCast(dd + at(k3, l.nb3) + at(k2, l.nb2) + at(k1, l.nb1))),
                                    @ptrCast(@alignCast(s0 + at(j3 * l.ne3 + k3, l.nb03) + at(j2 * l.ne2 + k2, l.nb02) + at(j1 * l.ne1 + k1, l.nb01) + at(j0 * l.ne0, l.nb00))),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_repeat_back` (ops.cpp:1880 @c1d0e7a00).
pub export fn ggml_compute_forward_repeat_back(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => repeatBackF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_concat_any` (ops.cpp:1899 @c1d0e7a00): whole
/// rows by `memcpy`, for any type including quantized ones.
fn concatAny(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.BinaryLocals.of(src0, src1, dst);

    const dim = impl.getOpParamsI32(dst, 0);

    impl.assert(dim >= 0 and dim < 4, "dim >= 0 && dim < 4");
    impl.assert(c.ggml_is_contiguous_rows(src0), "ggml_is_contiguous_rows(src0)");
    impl.assert(c.ggml_is_contiguous_rows(src1), "ggml_is_contiguous_rows(src1)");

    var o = [4]i64{ 0, 0, 0, 0 };
    const d: usize = @intCast(dim);

    if (dim == 0) {
        impl.assert(@rem(src0.ne[0], c.ggml_blck_size(src0.type)) == 0, "src0->ne[0] % ggml_blck_size(src0->type) == 0");
        impl.assert(@rem(src1.ne[0], c.ggml_blck_size(src1.type)) == 0, "src1->ne[0] % ggml_blck_size(src1->type) == 0");

        o[d] = @divTrunc(src0.ne[d], c.ggml_blck_size(src0.type));
    } else {
        o[d] = src0.ne[d];
    }

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    // Region 1: copy rows from src0
    const row0 = c.ggml_row_size(src0.type, l.ne00);
    var j3: i64 = 0;
    while (j3 < l.ne03) : (j3 += 1) {
        var j2: i64 = ith;
        while (j2 < l.ne02) : (j2 += nth) {
            var j1: i64 = 0;
            while (j1 < l.ne01) : (j1 += 1) {
                const x = s0 + at(j1, l.nb01) + at(j2, l.nb02) + at(j3, l.nb03);
                const y = dd + at(j1, l.nb1) + at(j2, l.nb2) + at(j3, l.nb3);
                @memcpy(y[0..row0], x[0..row0]);
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    const row1 = c.ggml_row_size(src1.type, l.ne10);
    j3 = 0;
    while (j3 < l.ne13) : (j3 += 1) {
        var j2: i64 = ith;
        while (j2 < l.ne12) : (j2 += nth) {
            var j1: i64 = 0;
            while (j1 < l.ne11) : (j1 += 1) {
                const x = s1 + at(j1, l.nb11) + at(j2, l.nb12) + at(j3, l.nb13);
                const y = dd + at(j1 + o[1], l.nb1) + at(j2 + o[2], l.nb2) + at(j3 + o[3], l.nb3) + at(o[0], l.nb0);
                @memcpy(y[0..row1], x[0..row1]);
            }
        }
    }
}

/// Ports `ggml_compute_forward_concat_i8` (ops.cpp:1951 @c1d0e7a00),
/// `ggml_compute_forward_concat_f16` (ops.cpp:1994 @c1d0e7a00) and
/// `ggml_compute_forward_concat_f32` (ops.cpp:2037 @c1d0e7a00).
///
/// Three copies of one element-at-a-time loop, differing only in the element
/// width they assert and move.
fn concatT(comptime T: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(c.ggml_type_size(src0.type) == @sizeOf(T), "ggml_type_size(src0->type) == sizeof(T)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.BinaryLocals.of(src0, src1, dst);

    const dim = impl.getOpParamsI32(dst, 0);

    impl.assert(dim >= 0 and dim < 4, "dim >= 0 && dim < 4");

    var o = [4]i64{ 0, 0, 0, 0 };
    o[@intCast(dim)] = src0.ne[@intCast(dim)];

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    // TODO: smarter multi-theading
    var j3: i64 = 0;
    while (j3 < l.ne3) : (j3 += 1) {
        var j2: i64 = ith;
        while (j2 < l.ne2) : (j2 += nth) {
            var j1: i64 = 0;
            while (j1 < l.ne1) : (j1 += 1) {
                var j0: i64 = 0;
                while (j0 < l.ne0) : (j0 += 1) {
                    const x: *const T = if (j0 < l.ne00 and j1 < l.ne01 and j2 < l.ne02 and j3 < l.ne03)
                        @ptrCast(@alignCast(s0 + at(j0, l.nb00) + at(j1, l.nb01) + at(j2, l.nb02) + at(j3, l.nb03)))
                    else
                        @ptrCast(@alignCast(s1 + at(j0 - o[0], l.nb10) + at(j1 - o[1], l.nb11) + at(j2 - o[2], l.nb12) + at(j3 - o[3], l.nb13)));

                    const y: *T = @ptrCast(@alignCast(dd + at(j0, l.nb0) + at(j1, l.nb1) + at(j2, l.nb2) + at(j3, l.nb3)));

                    y.* = x.*;
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_concat` (ops.cpp:2080 @c1d0e7a00).
pub export fn ggml_compute_forward_concat(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F16, c.GGML_TYPE_BF16, c.GGML_TYPE_I16 => concatT(c.ggml_fp16_t, params, dst),
        c.GGML_TYPE_I8 => concatT(i8, params, dst),
        c.GGML_TYPE_F32, c.GGML_TYPE_I32 => concatT(f32, params, dst),
        else => concatAny(params, dst),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
