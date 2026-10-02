//! `norm`, `rms_norm`, `rms_norm_mul_fused`, `rms_norm_back`, `group_norm`
//! and `l2_norm`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Accumulator types are not uniform, and each is reproduced
//!
//! `norm` sums in `f32` — through `vDSP_sve` when the row is contiguous, by a
//! plain loop when it is not. `rms_norm`, `rms_norm_back`, `group_norm` and
//! `l2_norm` sum in `ggml_float`, a `double`, but square in `f32` first:
//! `(ggml_float)(x*x)` rounds the product before widening it, so the cast
//! sits between the multiply and the add and nothing fuses.
//!
//! # Accelerate is live in `norm`
//!
//! `GGML_USE_ACCELERATE` is defined for this target, so `norm`'s contiguous
//! path centres with `vDSP_vsadd` and takes the variance from `vDSP_measqv`,
//! not from `ggml_vec_cvar_f32`. The `#else` arm is not ported.
//!
//! # Loop index names
//!
//! `i00`…`i13` are Zig integer type names. Renamed `j00`…`j13`, digit for
//! digit — the `cpu/mulmat.zig` convention. Do not renumber them.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

/// Mirrors `ggml_float` (vec.h:15 @c1d0e7a00).
const GgmlFloat = f64;

const Stride = isize;
const Length = c_ulong;
extern fn vDSP_vsadd(a: [*]const f32, ia: Stride, b: *const f32, cc: [*]f32, ic: Stride, n: Length) void;
extern fn vDSP_measqv(a: [*]const f32, ia: Stride, cc: *f32, n: Length) void;
extern fn fmaxf(x: f32, y: f32) f32;

/// A non-negative index times a byte stride: the C's `int64_t * size_t`.
inline fn at(i: i64, nb: usize) usize {
    return @as(usize, @intCast(i)) * nb;
}

/// Ports `ggml_compute_forward_norm_f32` (ops.cpp:3694 @c1d0e7a00), the
/// `GGML_USE_ACCELERATE` arm of its contiguous path.
fn normF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.UnaryLocals.of(src0, dst);

    const eps = impl.getOpParamsF32(dst, 0);

    impl.assert(eps >= 0.0, "eps >= 0.0f");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);
    const ne00f: f32 = @floatFromInt(l.ne00);

    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            var j01: i64 = ith;
            while (j01 < l.ne01) : (j01 += nth) {
                const x = s0 + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03);
                const y = dd + at(j01, l.nb1) + at(j02, l.nb2) + at(j03, l.nb3);

                if (l.nb00 == @sizeOf(f32) and l.nb0 == @sizeOf(f32)) {
                    const xf: [*]const f32 = @ptrCast(@alignCast(x));

                    var sum: f32 = 0.0;
                    vec.sum_f32(l.ne00, &sum, xf);
                    var mean: f32 = sum / ne00f;

                    const yf: [*]f32 = @ptrCast(@alignCast(y));
                    var variance: f32 = 0;

                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, @intCast(l.ne00));
                    vDSP_measqv(yf, 1, &variance, @intCast(l.ne00));

                    const scale = 1.0 / @sqrt(variance + eps);
                    vec.scale_f32(l.ne00, yf, scale);
                } else {
                    var sum: f32 = 0.0;
                    var j00: i64 = 0;
                    while (j00 < l.ne00) : (j00 += 1) {
                        sum += @as(*const f32, @ptrCast(@alignCast(x + at(j00, l.nb00)))).*;
                    }
                    const mean = sum / ne00f;

                    var variance: f32 = 0.0;
                    j00 = 0;
                    while (j00 < l.ne00) : (j00 += 1) {
                        const v = @as(*const f32, @ptrCast(@alignCast(x + at(j00, l.nb00)))).* - mean;
                        @as(*f32, @ptrCast(@alignCast(y + at(j00, l.nb0)))).* = v;
                        // `variance += v * v` is one expression; clang fuses it.
                        variance = @mulAdd(f32, v, v, variance);
                    }
                    variance /= ne00f;

                    const scale = 1.0 / @sqrt(variance + eps);
                    j00 = 0;
                    while (j00 < l.ne00) : (j00 += 1) {
                        @as(*f32, @ptrCast(@alignCast(y + at(j00, l.nb0)))).* *= scale;
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_norm` (ops.cpp:3763 @c1d0e7a00).
pub export fn ggml_compute_forward_norm(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => normF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_rms_norm_fuse_op` (ops.cpp:3785 @c1d0e7a00): the fusions that
/// can ride along with the `rms_norm` pass.
const RmsNormFuseOp = enum { none, mul };

/// Ports `ggml_compute_forward_rms_norm_f32` (ops.cpp:3791 @c1d0e7a00), the
/// `template <ggml_rms_norm_fuse_op FUSE_OP>` form.
///
/// With `.mul`, `src1` is whichever operand of `dst_fused` is not the
/// `rms_norm` node, and results land in `dst_fused`. Without it the C still
/// expands `GGML_TENSOR_BINARY_OP_LOCALS` over a null `src1` and relies on
/// the loads being dead; here the `src1` locals are only read when fused.
///
/// The C's `assert(scale > 0.0f)` is the `<cassert>` one, compiled out under
/// `NDEBUG`, so it is not reproduced.
fn rmsNormF32(comptime fuse_op: RmsNormFuseOp, params: *const ComputeParams, dst_rms_norm: *Tensor, dst_fused: ?*Tensor) void {
    const src0 = impl.one(Tensor, dst_rms_norm.src[0]);
    var src1: ?*const Tensor = null;
    var dst: *Tensor = dst_rms_norm;

    if (fuse_op == .mul) {
        const f = dst_fused.?;
        src1 = if (f.src[0] == dst_rms_norm) f.src[1] else f.src[0];
        dst = f;
    }

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.UnaryLocals.of(src0, dst);

    const eps = impl.getOpParamsF32(dst_rms_norm, 0);
    impl.assert(eps >= 0.0, "eps >= 0.0f");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);
    const ne00d: GgmlFloat = @floatFromInt(l.ne00);

    // TODO: optimize
    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            var j01: i64 = ith;
            while (j01 < l.ne01) : (j01 += nth) {
                const x: [*]const f32 = @ptrCast(@alignCast(s0 + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03)));

                var sum: GgmlFloat = 0.0;
                // worth switching to explicit SIMD?
                var j00: usize = 0;
                while (j00 < @as(usize, @intCast(l.ne00))) : (j00 += 1) {
                    sum += @as(GgmlFloat, x[j00] * x[j00]);
                }

                const mean: f32 = @floatCast(sum / ne00d);
                const scale = 1.0 / @sqrt(mean + eps);

                const y: [*]f32 = @ptrCast(@alignCast(dd + at(j01, l.nb1) + at(j02, l.nb2) + at(j03, l.nb3)));

                if (fuse_op == .mul) {
                    const s1 = src1.?;
                    const j11 = @rem(j01, s1.ne[1]);
                    const j12 = @rem(j02, s1.ne[2]);
                    const j13 = @rem(j03, s1.ne[3]);
                    const w: [*]const f32 = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(s1.data.?)) + at(j11, s1.nb[1]) + at(j12, s1.nb[2]) + at(j13, s1.nb[3])));

                    j00 = 0;
                    while (j00 < @as(usize, @intCast(l.ne00))) : (j00 += 1) {
                        y[j00] = x[j00] * scale * w[j00];
                    }
                } else {
                    const n = @as(usize, @intCast(l.ne00));
                    @memcpy(y[0..n], x[0..n]);
                    vec.scale_f32(l.ne00, y, scale);
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_rms_norm` (ops.cpp:3856 @c1d0e7a00).
pub export fn ggml_compute_forward_rms_norm(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => rmsNormF32(.none, params, dst, null),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_rms_norm_mul_fused` (ops.cpp:3876 @c1d0e7a00):
/// `dst_mul = rms_norm(src0) * src1` in one pass, without materialising the
/// `rms_norm` result.
pub export fn ggml_compute_forward_rms_norm_mul_fused(params: *const ComputeParams, dst_rms_norm: *Tensor, dst_mul: ?*Tensor) callconv(.c) void {
    impl.assert(dst_mul != null, "dst_mul != nullptr");
    const m = dst_mul.?;
    impl.assert(m.src[0] == dst_rms_norm or m.src[1] == dst_rms_norm, "dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm");

    const src0 = impl.one(Tensor, dst_rms_norm.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => rmsNormF32(.mul, params, dst_rms_norm, m),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_rms_norm_back_f32` (ops.cpp:3898 @c1d0e7a00).
///
/// The C's long derivation comment ends at
/// `dx = (dz + x*(-sum_xdz/sum_eps)) * rrms`; see ggml-org/ggml#1491 for why
/// `sum_eps` rather than `mean_eps` is the divisor.
fn rmsNormBackF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // gradients from forward pass output
    const src1 = impl.one(Tensor, dst.src[1]); // src1 from forward pass

    impl.assert(c.ggml_are_same_shape(src0, dst) and c.ggml_are_same_shape(src0, src1), "ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1)");

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(src1.nb[0] == @sizeOf(f32), "src1->nb[0] == sizeof(float)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.BinaryLocals.of(src0, src1, dst);

    const eps = impl.getOpParamsF32(dst, 0);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);
    const ne00f: f32 = @floatFromInt(l.ne00);

    // TODO: optimize
    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            var j01: i64 = ith;
            while (j01 < l.ne01) : (j01 += nth) {
                // src1 is same shape as src0 => same indices
                const j11 = j01;
                const j12 = j02;
                const j13 = j03;

                const dz: [*]const f32 = @ptrCast(@alignCast(s0 + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03)));
                const x: [*]const f32 = @ptrCast(@alignCast(s1 + at(j11, l.nb11) + at(j12, l.nb12) + at(j13, l.nb13)));

                var sum_xx: GgmlFloat = 0.0;
                var sum_xdz: GgmlFloat = 0.0;

                var j00: usize = 0;
                while (j00 < @as(usize, @intCast(l.ne00))) : (j00 += 1) {
                    sum_xx += @as(GgmlFloat, x[j00] * x[j00]);
                    sum_xdz += @as(GgmlFloat, x[j00] * dz[j00]);
                }

                const sum_xx_f: f32 = @floatCast(sum_xx);
                const mean_eps = sum_xx_f / ne00f + eps;
                // `(float)(sum_xx) + eps*ne00` is one expression; clang fuses
                // the product into the add.
                const sum_eps = @mulAdd(f32, eps, ne00f, sum_xx_f);
                const rrms = 1.0 / @sqrt(mean_eps);

                const dx: [*]f32 = @ptrCast(@alignCast(dd + at(j01, l.nb1) + at(j02, l.nb2) + at(j03, l.nb3)));

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                const scale_x = @as(f32, @floatCast(-sum_xdz)) / sum_eps;
                j00 = 0;
                while (j00 < @as(usize, @intCast(l.ne00))) : (j00 += 1) {
                    // `dz + x * scale_x` is one expression; clang fuses it.
                    dx[j00] = @mulAdd(f32, x[j00], scale_x, dz[j00]) * rrms;
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_rms_norm_back` (ops.cpp:4055 @c1d0e7a00).
pub export fn ggml_compute_forward_rms_norm_back(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => rmsNormBackF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_group_norm_f32` (ops.cpp:4075 @c1d0e7a00).
///
/// Threads split over groups, not rows. The mean and variance are divided in
/// `double` and only then narrowed, as the C's `float = double / int64_t` is.
fn groupNormF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");

    const ith: c_int = params.ith;
    const nth: c_int = params.nth;

    const l = common.UnaryLocals.of(src0, dst);

    // TODO: optimize

    const eps = impl.getOpParamsF32(dst, 1);

    const n_channels: c_int = @intCast(src0.ne[2]);
    const n_groups: c_int = impl.getOpParamsI32(dst, 0);
    const n_channels_per_group = @divTrunc(n_channels + n_groups - 1, n_groups);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var i: c_int = ith;
    while (i < n_groups) : (i += nth) {
        const start = i * n_channels_per_group;
        var end = start + n_channels_per_group;
        if (end > n_channels) {
            end = n_channels;
        }
        const step = end - start;
        const count: GgmlFloat = @floatFromInt(l.ne00 * l.ne01 * step);

        var j03: i64 = 0;
        while (j03 < l.ne03) : (j03 += 1) {
            var sum: GgmlFloat = 0.0;
            var j02: i64 = start;
            while (j02 < end) : (j02 += 1) {
                var j01: i64 = 0;
                while (j01 < l.ne01) : (j01 += 1) {
                    const x: [*]const f32 = @ptrCast(@alignCast(s0 + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03)));

                    var sumr: GgmlFloat = 0.0;
                    var j00: usize = 0;
                    while (j00 < @as(usize, @intCast(l.ne00))) : (j00 += 1) {
                        sumr += @as(GgmlFloat, x[j00]);
                    }
                    sum += sumr;
                }
            }
            const mean: f32 = @floatCast(sum / count);

            var sum2: GgmlFloat = 0.0;
            j02 = start;
            while (j02 < end) : (j02 += 1) {
                var j01: i64 = 0;
                while (j01 < l.ne01) : (j01 += 1) {
                    const x: [*]const f32 = @ptrCast(@alignCast(s0 + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03)));

                    const y: [*]f32 = @ptrCast(@alignCast(dd + at(j01, l.nb1) + at(j02, l.nb2) + at(j03, l.nb3)));

                    var sumr: GgmlFloat = 0.0;
                    var j00: usize = 0;
                    while (j00 < @as(usize, @intCast(l.ne00))) : (j00 += 1) {
                        const v = x[j00] - mean;
                        y[j00] = v;
                        sumr += @as(GgmlFloat, v * v);
                    }
                    sum2 += sumr;
                }
            }
            const variance: f32 = @floatCast(sum2 / count);
            const scale = 1.0 / @sqrt(variance + eps);

            j02 = start;
            while (j02 < end) : (j02 += 1) {
                var j01: i64 = 0;
                while (j01 < l.ne01) : (j01 += 1) {
                    const y: [*]f32 = @ptrCast(@alignCast(dd + at(j01, l.nb1) + at(j02, l.nb2) + at(j03, l.nb3)));
                    vec.scale_f32(l.ne00, y, scale);
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_group_norm` (ops.cpp:4150 @c1d0e7a00).
pub export fn ggml_compute_forward_group_norm(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => groupNormF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_l2_norm_f32` (ops.cpp:4170 @c1d0e7a00).
///
/// `sqrtf(sum)` takes a `float`, so the `double` sum is narrowed **before**
/// the square root, not after.
fn l2NormF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.UnaryLocals.of(src0, dst);

    const eps = impl.getOpParamsF32(dst, 0);

    impl.assert(eps >= 0.0, "eps >= 0.0f");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    // TODO: optimize
    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            var j01: i64 = ith;
            while (j01 < l.ne01) : (j01 += nth) {
                const x = s0 + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03);

                var sum: GgmlFloat = 0.0;
                var j00: i64 = 0;
                while (j00 < l.ne00) : (j00 += 1) {
                    const xi = @as(*const f32, @ptrCast(@alignCast(x + at(j00, l.nb00)))).*;
                    sum += @as(GgmlFloat, xi * xi);
                }

                const scale = 1.0 / fmaxf(@sqrt(@as(f32, @floatCast(sum))), eps);

                const y = dd + at(j01, l.nb1) + at(j02, l.nb2) + at(j03, l.nb3);

                if (l.nb00 == @sizeOf(f32) and l.nb0 == @sizeOf(f32)) {
                    const n = @as(usize, @intCast(l.ne00)) * @sizeOf(f32);
                    @memcpy(y[0..n], x[0..n]);
                    vec.scale_f32(l.ne00, @ptrCast(@alignCast(y)), scale);
                } else {
                    j00 = 0;
                    while (j00 < l.ne00) : (j00 += 1) {
                        const xi = @as(*const f32, @ptrCast(@alignCast(x + at(j00, l.nb00)))).*;
                        @as(*f32, @ptrCast(@alignCast(y + at(j00, l.nb0)))).* = xi * scale;
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_l2_norm` (ops.cpp:4218 @c1d0e7a00).
pub export fn ggml_compute_forward_l2_norm(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => l2NormF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
