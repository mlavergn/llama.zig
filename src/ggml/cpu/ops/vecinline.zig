//! The `vec.h` inlines the `ops.cpp` kernels call.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/vec.h` at v0.3.0 (`c1d0e7a00`).
//!
//! **These are `inline static` in the C, so they have no symbols** and
//! `cpu/vec.zig` — which ports `vec.cpp`'s *exported* contract — does not
//! provide them. They are compiled into each caller, and here that caller is
//! the `ops.cpp` port.
//!
//! Every one of them has an `#if defined(__AVX2__)` or similar vector arm that
//! is **dead on this target**, so what is ported is the scalar loop the C
//! falls through to. Where that is the case the doc comment says so, per the
//! `#if`-arm rule in `CLAUDE.md`.

const std = @import("std");
const impl = @import("../../impl.zig");

const c = impl.c;

/// Ports `ggml_vec_add_f32` (vec.h:89 @c1d0e7a00), the scalar fallthrough —
/// its only other arm is `__AVX2__`, which this target does not compile.
pub inline fn add_f32(n: i64, z: [*]f32, x: [*]const f32, y: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) z[i] = x[i] + y[i];
}

/// Ports `ggml_vec_add_f16` (vec.h:104 @c1d0e7a00).
pub inline fn add_f16(n: i64, z: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, y: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        z[i] = impl.fp32ToFp16(impl.fp16ToFp32(x[i]) + impl.fp16ToFp32(y[i]));
    }
}

/// Ports `ggml_vec_add1_f32` (vec.h:109 @c1d0e7a00).
pub inline fn add1_f32(n: i64, z: [*]f32, x: [*]const f32, v: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) z[i] = x[i] + v;
}

/// Ports `ggml_vec_acc_f32` (vec.h:110 @c1d0e7a00).
pub inline fn acc_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] += x[i];
}

/// Ports `ggml_vec_acc1_f32` (vec.h:111 @c1d0e7a00).
pub inline fn acc1_f32(n: i64, y: [*]f32, v: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] += v;
}

/// Ports `ggml_vec_set_f32` (vec.h:118 @c1d0e7a00).
pub inline fn set_f32(n: i64, x: [*]f32, v: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) x[i] = v;
}

/// Ports `ggml_vec_cpy_f32` (vec.h:119 @c1d0e7a00).
pub inline fn cpy_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = x[i];
}

// -----------------------------------------------------------------------------
// Reductions
//
// `ggml_vec_sum_f32` and `ggml_vec_max_f32` have an Accelerate arm, and
// `GGML_USE_ACCELERATE` is defined for this target, so the vDSP call is what
// compiles and the `#ifndef` scalar loop is dead. The `_ggf` variants have no
// Accelerate arm at all -- that is the whole reason they exist alongside
// `ggml_vec_sum_f32`, which on this target cannot accumulate in `double`.

/// Mirrors `vDSP_Length`, the `unsigned long` element count.
const Length = c_ulong;
const Stride = isize;

extern fn vDSP_sve(a: [*]const f32, ia: Stride, s: *f32, n: Length) void;
extern fn vDSP_maxv(a: [*]const f32, ia: Stride, s: *f32, n: Length) void;

/// Ports `ggml_vec_sum_f32` (vec.h:1495 @c1d0e7a00), the `GGML_USE_ACCELERATE`
/// arm.
///
/// **Not interchangeable with `sum_f32_ggf`.** This one sums in `f32` through
/// vDSP; that one sums in `double`. `mean` and `sum_rows` call this, `sum`
/// calls the other, and swapping them changes the result.
pub inline fn sum_f32(n: i64, s: *f32, x: [*]const f32) void {
    vDSP_sve(x, 1, s, @intCast(n));
}

/// Ports `ggml_vec_sum_f32_ggf` (vec.h:1517 @c1d0e7a00). Accumulates in
/// `ggml_float`, a `double`.
pub inline fn sum_f32_ggf(n: i64, s: *f64, x: [*]const f32) void {
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) sum += @as(f64, x[i]);
    s.* = sum;
}

/// Ports `ggml_vec_sum_f16_ggf` (vec.h:1525 @c1d0e7a00). Despite the `_ggf`,
/// this one accumulates in `f32`; only the `f32` variant above widens.
pub inline fn sum_f16_ggf(n: i64, s: *f32, x: [*]const c.ggml_fp16_t) void {
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) sum += impl.fp16ToFp32(x[i]);
    s.* = sum;
}

/// Ports `ggml_vec_sum_bf16_ggf` (vec.h:1533 @c1d0e7a00).
pub inline fn sum_bf16_ggf(n: i64, s: *f32, x: [*]const c.ggml_bf16_t) void {
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) sum += impl.bf16ToFp32(x[i].bits);
    s.* = sum;
}

/// Ports `ggml_vec_cumsum_f32` (vec.h:1507 @c1d0e7a00).
pub inline fn cumsum_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        y[i] = if (i == 0) x[i] else y[i - 1] + x[i];
    }
}

/// Ports `ggml_vec_max_f32` (vec.h:1541 @c1d0e7a00), the
/// `GGML_USE_ACCELERATE` arm.
pub inline fn max_f32(n: i64, s: *f32, x: [*]const f32) void {
    vDSP_maxv(x, 1, s, @intCast(n));
}

/// Ports `ggml_vec_argmax_f32` (vec.h:1558 @c1d0e7a00).
///
/// Note the C's tie-breaking: it reassigns `max` first and then compares
/// `max == x[i]`, so the **last** index attaining the maximum wins, not the
/// first. Written the same way here rather than as the `>` test it looks like.
pub inline fn argmax_f32(n: i64, s: *i32, x: [*]const f32) void {
    var max: f32 = -std.math.inf(f32);
    var idx: i32 = 0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        max = @max(max, x[i]);
        if (max == x[i]) idx = @intCast(i);
    }
    s.* = idx;
}

// -----------------------------------------------------------------------------
// Elementwise arithmetic

/// Ports `ggml_vec_cpy_i32` (vec.h:84 @c1d0e7a00).
pub inline fn cpy_i32(n: i64, y: [*]i32, x: [*]const i32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = x[i];
}

/// Ports `ggml_vec_set_f16` (vec.h:86 @c1d0e7a00).
pub inline fn set_f16(n: i64, x: [*]c.ggml_fp16_t, v: c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) x[i] = v;
}

/// Ports `ggml_vec_sub_f32` (vec.h:112 @c1d0e7a00).
pub inline fn sub_f32(n: i64, z: [*]f32, x: [*]const f32, y: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) z[i] = x[i] - y[i];
}

/// Ports `ggml_vec_mul_f32` (vec.h:127 @c1d0e7a00).
pub inline fn mul_f32(n: i64, z: [*]f32, x: [*]const f32, y: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) z[i] = x[i] * y[i];
}

// -----------------------------------------------------------------------------
// Multiply-accumulate and scale
//
// `GGML_SIMD` is defined on this target and SVE is not, so `mad` and `scale`
// compile their NEON arms. `mad1` and `scale_f32` have an Accelerate arm ahead
// of those, and it wins.
//
// Every NEON arm here is elementwise: lanes never meet, so the step-16 body
// and the scalar tail compute the same function of each element *provided*
// the tail fuses the way the body does. It does — `y[i] += x[i]*v` is one
// expression at `-ffp-contract=on`, and clang emits `fmadd` for it — so the
// whole range is one fused multiply-add per element.
//
// The `f16` arms are different: with `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`
// the body computes **in half precision** (`vfmaq_f16`, `vmulq_f16`) and the
// tail widens to `f32`. That split is real and is reproduced.

/// Ports `ggml_vec_mad_f32` (vec.h:319 @c1d0e7a00), the `GGML_SIMD` NEON arm:
/// `y += x * v`, fused per element. See the section comment for why body and
/// tail are one loop here.
pub inline fn mad_f32(n: i64, y: [*]f32, x: [*]const f32, v: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = @mulAdd(f32, x[i], v, y[i]);
}

/// Ports `ggml_vec_mad_f16` (vec.h:439 @c1d0e7a00), the `GGML_SIMD` NEON arm
/// with `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`.
///
/// Elements below `n & ~31` go through `vfmaq_f16` with `v` rounded to `f16`
/// first — a fused multiply-add **in half precision**. The rest widen to
/// `f32`, fuse there, and narrow once.
pub inline fn mad_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, v: f32) void {
    const nn: usize = @intCast(n);
    const np = nn & ~@as(usize, f16_step - 1);
    const vx: f16 = @floatCast(v);

    var i: usize = 0;
    while (i < np) : (i += 1) {
        const ay: f16 = @bitCast(y[i]);
        const ax: f16 = @bitCast(x[i]);
        y[i] = @bitCast(@mulAdd(f16, ax, vx, ay));
    }
    while (i < nn) : (i += 1) {
        y[i] = impl.fp32ToFp16(@mulAdd(f32, impl.fp16ToFp32(x[i]), v, impl.fp16ToFp32(y[i])));
    }
}

/// `GGML_F16_STEP` (simd-mappings.h:264 @c1d0e7a00), the
/// `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` arm: four `float16x8_t` per step.
const f16_step: usize = 32;

/// `GGML_VEC_MAD_UNROLL` (vec.h:51 @c1d0e7a00).
pub const mad_unroll: usize = 32;

/// Ports `ggml_vec_mad_f32_unroll` (vec.h:585 @c1d0e7a00), the `GGML_SIMD`
/// NEON arm.
///
/// The body loads `y` once and folds all 32 `x[k] * v[k]` into it with
/// `vfmaq_f32`, `k` ascending; the tail runs `k` outermost. Per element both
/// are the same 32 fused steps in the same order, so they are one loop here.
///
/// Parameters:
/// - `xs`, `vs`: **byte** strides between the 32 rows of `xv` and `vv`.
pub inline fn mad_f32_unroll(n: i64, xs: i64, vs: i64, y: [*]f32, xv: [*]const f32, vv: [*]const f32) void {
    var x: [mad_unroll][*]const f32 = undefined;
    var v: [mad_unroll][*]const f32 = undefined;
    const xb: [*]const u8 = @ptrCast(xv);
    const vb: [*]const u8 = @ptrCast(vv);
    for (0..mad_unroll) |k| {
        x[k] = @ptrCast(@alignCast(xb + k * @as(usize, @intCast(xs))));
        v[k] = @ptrCast(@alignCast(vb + k * @as(usize, @intCast(vs))));
    }

    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        var acc = y[i];
        for (0..mad_unroll) |k| acc = @mulAdd(f32, x[k][i], v[k][0], acc);
        y[i] = acc;
    }
}

extern fn vDSP_vsmsa(a: [*]const f32, ia: Stride, b: *const f32, cc: *const f32, d: [*]f32, id: Stride, n: Length) void;
extern fn vDSP_vsmul(a: [*]const f32, ia: Stride, b: *const f32, cc: [*]f32, ic: Stride, n: Length) void;

/// Ports `ggml_vec_mad1_f32` (vec.h:655 @c1d0e7a00), the `GGML_USE_ACCELERATE`
/// arm: `y = x * s + b` through `vDSP_vsmsa`.
pub inline fn mad1_f32(n: i64, y: [*]f32, x: [*]const f32, s: f32, b: f32) void {
    vDSP_vsmsa(x, 1, &s, &b, y, 1, @intCast(n));
}

/// Ports `ggml_vec_scale_f32` (vec.h:703 @c1d0e7a00), the
/// `GGML_USE_ACCELERATE` arm: `y *= v` through `vDSP_vsmul`.
pub inline fn scale_f32(n: i64, y: [*]f32, v: f32) void {
    vDSP_vsmul(y, 1, &v, y, 1, @intCast(n));
}

/// Ports `ggml_vec_scale_f16` (vec.h:769 @c1d0e7a00), the `GGML_SIMD` NEON arm
/// with `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`: the body multiplies **in
/// half precision** by `v` rounded to `f16`, the tail in `f32`.
pub inline fn scale_f16(n: i64, y: [*]c.ggml_fp16_t, v: f32) void {
    const nn: usize = @intCast(n);
    const np = nn & ~@as(usize, f16_step - 1);
    const vx: f16 = @floatCast(v);

    var i: usize = 0;
    while (i < np) : (i += 1) {
        const ay: f16 = @bitCast(y[i]);
        y[i] = @bitCast(ay * vx);
    }
    while (i < nn) : (i += 1) y[i] = impl.fp32ToFp16(impl.fp16ToFp32(y[i]) * v);
}

// -----------------------------------------------------------------------------
// Activations
//
// `GGML_GELU_FP16` and `GGML_GELU_QUICK_FP16` are both defined in `vec.h`, so
// the `f32` gelu kernels compile their table-lookup arms. The tables are
// `cpu/vec.zig`'s, filled by `ggml_cpu_init`.

const tables = @import("../vec.zig");

extern fn tanhf(x: f32) f32;
extern fn expf(x: f32) f32;
extern fn erff(x: f32) f32;

/// `SQRT_2_INV` (vec.h:966 @c1d0e7a00).
const sqrt_2_inv: f32 = 0.70710678118654752440084436210484;

/// Ports `ggml_vec_tanh_f32` (vec.h:909 @c1d0e7a00).
pub inline fn tanh_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = tanhf(x[i]);
}

/// Ports `ggml_vec_leaky_relu_f32` (vec.h:929 @c1d0e7a00).
///
/// `pos + ns * neg` is one expression and clang fuses it. Named here to read
/// as the C compiles, though it cannot change a bit: one of `pos` and `neg`
/// is always zero, so either the product or the addend is exact.
pub inline fn leaky_relu_f32(n: i64, y: [*]f32, x: [*]const f32, ns: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const pos: f32 = if (x[i] > 0.0) x[i] else 0.0;
        const neg: f32 = if (x[i] < 0.0) x[i] else 0.0;
        y[i] = @mulAdd(f32, ns, neg, pos);
    }
}

/// Ports `ggml_vec_leaky_relu_f16` (vec.h:930 @c1d0e7a00).
pub inline fn leaky_relu_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, ns: f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const v = impl.fp16ToFp32(x[i]);
        const pos: f32 = if (v > 0.0) v else 0.0;
        const neg: f32 = if (v < 0.0) v else 0.0;
        y[i] = impl.fp32ToFp16(@mulAdd(f32, ns, neg, pos));
    }
}

/// Ports `ggml_vec_gelu_f16` (vec.h:972 @c1d0e7a00): a straight table lookup.
pub inline fn gelu_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = tables.ggml_table_gelu_f16[x[i]];
}

/// Ports `ggml_vec_gelu_erf_f16` (vec.h:979 @c1d0e7a00).
pub inline fn gelu_erf_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const xi = impl.fp16ToFp32(x[i]);
        const res = 0.5 * xi * (1.0 + erff(xi * sqrt_2_inv));
        y[i] = impl.fp32ToFp16(res);
    }
}

/// Ports `ggml_vec_gelu_f32` (vec.h:988 @c1d0e7a00), the `GGML_GELU_FP16`
/// arm: clamp outside `[-10, 10]`, otherwise look up the `f16` table.
pub inline fn gelu_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        if (x[i] <= -10.0) {
            y[i] = 0.0;
        } else if (x[i] >= 10.0) {
            y[i] = x[i];
        } else {
            const t = impl.fp32ToFp16(x[i]);
            y[i] = impl.fp16ToFp32(tables.ggml_table_gelu_f16[t]);
        }
    }
}

/// Ports `ggml_vec_gelu_erf_f32` (vec.h:1010 @c1d0e7a00).
pub inline fn gelu_erf_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const xi = x[i];
        y[i] = 0.5 * xi * (1.0 + erff(xi * sqrt_2_inv));
    }
}

/// Ports `ggml_vec_gelu_quick_f16` (vec.h:1021 @c1d0e7a00).
pub inline fn gelu_quick_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = tables.ggml_table_gelu_quick_f16[x[i]];
}

/// Ports `ggml_vec_gelu_quick_f32` (vec.h:1029 @c1d0e7a00), the
/// `GGML_GELU_QUICK_FP16` arm. Unlike `gelu_f32` it has no clamp.
pub inline fn gelu_quick_f32(n: i64, y: [*]f32, x: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const t = impl.fp32ToFp16(x[i]);
        y[i] = impl.fp16ToFp32(tables.ggml_table_gelu_quick_f16[t]);
    }
}

/// Ports `ggml_silu_f16` (vec.h:1049 @c1d0e7a00).
inline fn siluF16(x: c.ggml_fp16_t) c.ggml_fp16_t {
    const v = impl.fp16ToFp32(x);
    return impl.fp32ToFp16(v / (1.0 + expf(-v)));
}

/// Ports `ggml_vec_silu_f16` (vec.h:1372 @c1d0e7a00).
pub inline fn silu_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = siluF16(x[i]);
}

/// Ports `ggml_silu_backward_f32` (vec.h:1378 @c1d0e7a00).
///
/// `1.0f + x*(1.0f - s)` is one expression and clang fuses it into
/// `fma(x, 1 - s, 1)`; the outer products are not adds and stay plain.
inline fn siluBackwardF32(x: f32, dy: f32) f32 {
    const s = 1.0 / (1.0 + expf(-x));
    return dy * s * @mulAdd(f32, x, 1.0 - s, 1.0);
}

/// Ports `ggml_silu_backward_f16` (vec.h:1383 @c1d0e7a00), fused as
/// `siluBackwardF32` is.
inline fn siluBackwardF16(x: c.ggml_fp16_t, dy: c.ggml_fp16_t) c.ggml_fp16_t {
    const v = impl.fp16ToFp32(x);
    const s = 1.0 / (1.0 + expf(-v));
    return impl.fp32ToFp16(impl.fp16ToFp32(dy) * s * @mulAdd(f32, v, 1.0 - s, 1.0));
}

/// Ports `ggml_vec_silu_backward_f32` (vec.h:1389 @c1d0e7a00).
pub inline fn silu_backward_f32(n: i64, dx: [*]f32, x: [*]const f32, dy: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) dx[i] = siluBackwardF32(x[i], dy[i]);
}

/// Ports `ggml_vec_silu_backward_f16` (vec.h:1395 @c1d0e7a00).
pub inline fn silu_backward_f16(n: i64, dx: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, dy: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) dx[i] = siluBackwardF16(x[i], dy[i]);
}

// -----------------------------------------------------------------------------
// Gated linear units

/// Ports `ggml_vec_reglu_f32` (vec.h:1401 @c1d0e7a00).
pub inline fn reglu_f32(n: i64, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) y[i] = if (x[i] > 0.0) x[i] * g[i] else 0.0;
}

/// Ports `ggml_vec_reglu_f16` (vec.h:1407 @c1d0e7a00).
pub inline fn reglu_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, g: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const v = impl.fp16ToFp32(x[i]);
        y[i] = impl.fp32ToFp16(if (v > 0.0) v * impl.fp16ToFp32(g[i]) else 0.0);
    }
}

/// Ports `ggml_vec_geglu_f32` (vec.h:1415 @c1d0e7a00), the `GGML_GELU_FP16`
/// arm.
pub inline fn geglu_f32(n: i64, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        if (x[i] <= -10.0) {
            y[i] = 0.0;
        } else if (x[i] >= 10.0) {
            y[i] = x[i] * g[i];
        } else {
            const t = impl.fp32ToFp16(x[i]);
            y[i] = impl.fp16ToFp32(tables.ggml_table_gelu_f16[t]) * g[i];
        }
    }
}

/// Ports `ggml_vec_geglu_f16` (vec.h:1437 @c1d0e7a00).
pub inline fn geglu_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, g: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const v = impl.fp16ToFp32(g[i]);
        y[i] = impl.fp32ToFp16(impl.fp16ToFp32(tables.ggml_table_gelu_f16[x[i]]) * v);
    }
}

/// Ports `ggml_vec_swiglu_f16` (vec.h:1447 @c1d0e7a00).
pub inline fn swiglu_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, g: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const xi = impl.fp16ToFp32(x[i]);
        const gi = impl.fp16ToFp32(g[i]);
        y[i] = impl.fp32ToFp16((xi / (1.0 + expf(-xi))) * gi);
    }
}

/// Ports `ggml_vec_geglu_erf_f32` (vec.h:1455 @c1d0e7a00).
pub inline fn geglu_erf_f32(n: i64, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const xi = x[i];
        y[i] = 0.5 * xi * (1.0 + erff(xi * sqrt_2_inv)) * g[i];
    }
}

/// Ports `ggml_vec_geglu_erf_f16` (vec.h:1462 @c1d0e7a00).
pub inline fn geglu_erf_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, g: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const xi = impl.fp16ToFp32(x[i]);
        const gi = impl.fp16ToFp32(g[i]);
        y[i] = impl.fp32ToFp16(0.5 * xi * (1.0 + erff(xi * sqrt_2_inv)) * gi);
    }
}

/// Ports `ggml_vec_geglu_quick_f32` (vec.h:1471 @c1d0e7a00), the
/// `GGML_GELU_QUICK_FP16` arm.
pub inline fn geglu_quick_f32(n: i64, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const t = impl.fp32ToFp16(x[i]);
        y[i] = impl.fp16ToFp32(tables.ggml_table_gelu_quick_f16[t]) * g[i];
    }
}

/// Ports `ggml_vec_geglu_quick_f16` (vec.h:1487 @c1d0e7a00).
pub inline fn geglu_quick_f16(n: i64, y: [*]c.ggml_fp16_t, x: [*]const c.ggml_fp16_t, g: [*]const c.ggml_fp16_t) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const v = impl.fp16ToFp32(g[i]);
        y[i] = impl.fp32ToFp16(impl.fp16ToFp32(tables.ggml_table_gelu_quick_f16[x[i]]) * v);
    }
}

// -----------------------------------------------------------------------------
// The exported half of `vec.h`
//
// These have symbols — `cpu/vec.zig` ports them from `vec.cpp` — so the
// kernels call them across the C ABI exactly as the C++ does. Re-exported
// here so every `ops.cpp` kernel reaches all of `vec.h` through one import.

pub const dot_f32 = tables.ggml_vec_dot_f32;
pub const dot_f16 = tables.ggml_vec_dot_f16;
pub const silu_f32 = tables.ggml_vec_silu_f32;
pub const swiglu_f32 = tables.ggml_vec_swiglu_f32;
pub const cvar_f32 = tables.ggml_vec_cvar_f32;
pub const soft_max_f32 = tables.ggml_vec_soft_max_f32;
pub const log_soft_max_f32 = tables.ggml_vec_log_soft_max_f32;

/// A plain C reduction loop `acc += x[i] * y[i]` over contiguous memory, as
/// the reference compiler emits it: rounded groups of four, then a fused
/// remainder. See `vectorizedTail` in `cpu/vec.zig` for the measurement.
///
/// **Only for loops the compiler actually vectorizes**, which is a cost-model
/// decision, not something the source states. Each call site is one that
/// `make ops-diff` showed needs it.
pub const strictDot = tables.vectorizedTail;

// -----------------------------------------------------------------------------
// Unit Tests
//
// These are `inline`, so nothing analyses a body until something calls it —
// `refAllDecls` alone does not. Every helper is called at least once below,
// which is what makes a type error in an unused one visible.

const testing = std.testing;

fn h(v: f32) c.ggml_fp16_t {
    return impl.fp32ToFp16(v);
}

test {
    testing.refAllDecls(@This());
}

test "every f32 helper compiles and computes elementwise" {
    var x = [_]f32{ -2, -0.5, 0, 0.5, 2, 12, -12, 3 };
    var g = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var y: [8]f32 = undefined;
    const n: i64 = x.len;

    sub_f32(n, &y, &x, &g);
    try testing.expectEqual(@as(f32, -3), y[0]);
    mul_f32(n, &y, &x, &g);
    try testing.expectEqual(@as(f32, -2), y[0]);
    tanh_f32(n, &y, &x);
    leaky_relu_f32(n, &y, &x, 0.1);
    try testing.expectEqual(@as(f32, -0.2), y[0]);
    try testing.expectEqual(@as(f32, 2), y[4]);
    gelu_f32(n, &y, &x);
    try testing.expectEqual(@as(f32, 12), y[5]);
    try testing.expectEqual(@as(f32, 0), y[6]);
    gelu_erf_f32(n, &y, &x);
    gelu_quick_f32(n, &y, &x);
    silu_backward_f32(n, &y, &x, &g);
    reglu_f32(n, &y, &x, &g);
    try testing.expectEqual(@as(f32, 0), y[0]);
    try testing.expectEqual(@as(f32, 10), y[4]);
    geglu_f32(n, &y, &x, &g);
    try testing.expectEqual(@as(f32, 72), y[5]);
    geglu_erf_f32(n, &y, &x, &g);
    geglu_quick_f32(n, &y, &x, &g);

    @memcpy(&y, &g);
    scale_f32(n, &y, 2);
    try testing.expectEqual(@as(f32, 16), y[7]);
    mad1_f32(n, &y, &x, 2, 1);
    try testing.expectEqual(@as(f32, -3), y[0]);

    var iy = [_]i32{ 0, 0 };
    cpy_i32(2, &iy, &[_]i32{ 4, 5 });
    try testing.expectEqual(@as(i32, 5), iy[1]);
}

test "every f16 helper compiles" {
    var x: [40]c.ggml_fp16_t = undefined;
    var g: [40]c.ggml_fp16_t = undefined;
    var y: [40]c.ggml_fp16_t = undefined;
    for (0..40) |i| {
        x[i] = h(@as(f32, @floatFromInt(i)) * 0.25 - 4);
        g[i] = h(0.5);
    }
    const n: i64 = 40;

    set_f16(n, &y, h(1));
    leaky_relu_f16(n, &y, &x, 0.1);
    gelu_f16(n, &y, &x);
    gelu_erf_f16(n, &y, &x);
    gelu_quick_f16(n, &y, &x);
    silu_f16(n, &y, &x);
    silu_backward_f16(n, &y, &x, &g);
    reglu_f16(n, &y, &x, &g);
    geglu_f16(n, &y, &x, &g);
    swiglu_f16(n, &y, &x, &g);
    geglu_erf_f16(n, &y, &x, &g);
    geglu_quick_f16(n, &y, &x, &g);
}

test "mad_f32 fuses, in body and tail alike" {
    // 1 + 2^-12 squared needs 25 significand bits; only a fused multiply-add
    // keeps the 2^-24 term when the addend cancels the leading 1.
    const e: f32 = 1.0 + 0x1p-12;
    var x = [_]f32{e} ** 17;
    var y = [_]f32{-(1.0 + 0x1p-11)} ** 17;
    mad_f32(17, &y, &x, e);
    for (y) |v| try testing.expectEqual(@as(f32, 0x1p-24), v);
}

test "mad_f16 and scale_f16 compute the first 32 in half precision" {
    // v = 1 + 0.75 * 2^-10. In f16 it rounds to 1 + 2^-10, and 3 times that
    // is a tie on the f16 grid that goes to even, 3 + 2^-8. In f32, 3v is
    // exact and narrows to 3 + 2^-9. So elements 0..31 and element 32 differ,
    // and that difference is the body/tail split.
    const v: f32 = 1.0 + 0x1.8p-11;
    const body: f32 = 3.0 + 0x1p-8;
    const tail: f32 = 3.0 + 0x1p-9;

    var y: [33]c.ggml_fp16_t = undefined;
    var x: [33]c.ggml_fp16_t = undefined;
    for (0..33) |i| y[i] = h(3);
    scale_f16(33, &y, v);
    try testing.expectEqual(body, impl.fp16ToFp32(y[31]));
    try testing.expectEqual(tail, impl.fp16ToFp32(y[32]));

    for (0..33) |i| {
        y[i] = h(0);
        x[i] = h(3);
    }
    mad_f16(33, &y, &x, v);
    try testing.expectEqual(body, impl.fp16ToFp32(y[31]));
    try testing.expectEqual(tail, impl.fp16ToFp32(y[32]));
}

test "mad_f32_unroll folds 32 rows in order" {
    var rows: [mad_unroll][3]f32 = undefined;
    var vs: [mad_unroll]f32 = undefined;
    for (0..mad_unroll) |k| {
        rows[k] = .{ 1, 2, 3 };
        vs[k] = @floatFromInt(k);
    }
    var y = [_]f32{ 0, 0, 0 };
    mad_f32_unroll(3, @sizeOf([3]f32), @sizeOf(f32), &y, &rows[0], &vs);
    // sum k for k < 32 is 496.
    try testing.expectEqual([3]f32{ 496, 992, 1488 }, y);
}
