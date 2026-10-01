//! The float vector kernels: three dot products, the SiLU family, and the
//! softmax reductions.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-cpu/vec.cpp` — the ten exported symbols
//! - `llama.cpp/ggml/src/ggml-cpu/vec.h`   — `ggml_v_expf`, `ggml_v_silu` and
//!                                           `ggml_silu_f32`, which are
//!                                           `inline static` there and have no
//!                                           symbols of their own
//!
//! Both at v0.3.0 (`c1d0e7a00`). Each declaration below names the C it
//! replaces and the line it began at.
//!
//! # Only the NEON arm is ported
//!
//! `vec.cpp` is 613 lines of which **169 compile on this target** — the rest
//! is behind `__ARM_FEATURE_SVE`, `__AVX512F__` and friends. Every citation
//! below names the arm this build takes, per `CLAUDE.md`'s rule for a symbol
//! with several `#if` arms. `ggml_vec_dot_bf16` is the notable one: it has
//! **no SIMD arm at all here** and is a plain scalar loop accumulating in
//! `double`.
//!
//! # The gate is `src/ggml/cpu/vec_golden.zig`, and nothing else
//!
//! These are accumulating reductions, so the answer depends on the accumulator
//! structure and the summation order, not just the arithmetic. No other check
//! in this project can see that:
//!
//! - `make parity-cli` and `make port` never reach them — on a Metal machine
//!   the CPU kernels do not run during inference, measured by aborting in one
//!   and watching generation finish.
//! - `make backend-ops` compares with NMSE at `1e-7`, which catches a kernel
//!   wrong across the domain and misses one wrong in a band of it.
//!
//! So `scripts/vec-golden` captures the exact bits the C produces and
//! `vec_testing.zig` compares on bits. Two of its six patterns — `random` and
//! `lopsided` — actually discriminate summation order; that was measured, not
//! assumed.
//!
//! # Three different accumulator shapes
//!
//! Reproduced exactly, because each is visible in the last bit:
//!
//! | kernel | body | reduction | tail |
//! |---|---|---|---|
//! | `f32`  | 4 × `f32x4`, `vfmaq_f32`, step 16 | tree 2→1, then pairwise `vaddvq_f32` | `f32` scalar |
//! | `f16`  | 4 × `f16x8`, `vfmaq_f16`, step 32 | tree 2→1, widen to `f32x4`, pairwise | `double` scalar |
//! | `bf16` | none | none | `double` scalar |
//!
//! The `f16` body accumulates in **half precision**. The two tails accumulate
//! in different types from each other, which is why `vec_golden.zig` captures
//! a second length that leaves a remainder.

const std = @import("std");
const impl = @import("../impl.zig");
const neon = @import("quants/arm/neon.zig");
const c = impl.c;

const f32x4 = neon.f32x4;

/// Mirrors `ggml_float` (vec.h:15 @c1d0e7a00), the `double` the
/// reductions accumulate in. Named rather than written as `f64` so the C's
/// casts stay readable at each site.
const GgmlFloat = f64;

// -----------------------------------------------------------------------------
// The NEON intrinsics this file needs beyond `quants/arm/neon.zig`
//
// That module is scoped to `arch/arm/quants.c` and says so, so `vec.cpp`'s
// additions live here rather than widening it. The two whose semantics are
// subtle — the pairwise `addvq_f32` and the fused `fma_f32` — are imported
// from it rather than duplicated, because getting either wrong is a last-bit
// bug and it is already documented there.

/// Eight `f16` lanes, matching `float16x8_t`.
const f16x8 = @Vector(8, f16);

/// Four `f16` lanes, matching `float16x4_t`.
const f16x4 = @Vector(4, f16);

/// Ports `vdupq_n_f32`.
inline fn dup_n_f32(x: f32) f32x4 {
    return @splat(x);
}

/// Ports `vaddq_f32`.
inline fn add_f32(a: f32x4, b: f32x4) f32x4 {
    return a + b;
}

/// Ports `vsubq_f32`.
inline fn sub_f32(a: f32x4, b: f32x4) f32x4 {
    return a - b;
}

/// Ports `vdivq_f32`.
inline fn div_f32(a: f32x4, b: f32x4) f32x4 {
    return a / b;
}

/// Ports `vfmsq_f32`: `a - b * cc`, **fused**.
///
/// The negated counterpart of `vfmaq_f32`, and fused the same way — clang
/// emits `fmls`, one rounding. `@mulAdd` with a negated multiplier is the
/// same operation.
inline fn fms_f32(a: f32x4, b: f32x4, cc: f32x4) f32x4 {
    return @mulAdd(f32x4, -b, cc, a);
}

/// Ports `vfmaq_f16`: `a + b * cc` over eight half-precision lanes, fused.
///
/// The accumulation really is in `f16`; widening it would change the result
/// everywhere this is used.
inline fn fma_f16(a: f16x8, b: f16x8, cc: f16x8) f16x8 {
    return @mulAdd(f16x8, b, cc, a);
}

/// Ports `vaddq_f16`.
inline fn add_f16(a: f16x8, b: f16x8) f16x8 {
    return a + b;
}

/// Ports `vget_low_f16` and `vget_high_f16` followed by `vcvt_f32_f16`:
/// the lower or upper four `f16` lanes widened to `f32`.
inline fn cvt_f32_f16_half(v: f16x8, comptime upper: bool) f32x4 {
    const base: usize = if (upper) 4 else 0;
    const half: f16x4 = .{ v[base], v[base + 1], v[base + 2], v[base + 3] };
    return @floatCast(half);
}

// -----------------------------------------------------------------------------
// The gelu lookup tables
//
// Ports `ggml_table_gelu_f16` and `ggml_table_gelu_quick_f16`
// (vec.cpp:6, 9 @c1d0e7a00). Definitions, not declarations: `cpu/convert.zig`
// fills them from `ggml_cpu_init` and until now declared them `extern var`
// against `vec.o`. 65,536 entries each, one per `f16` bit pattern.

/// Ports `ggml_table_gelu_f16` (vec.cpp:6 @c1d0e7a00).
pub export var ggml_table_gelu_f16: [1 << 16]c.ggml_fp16_t = undefined;

/// Ports `ggml_table_gelu_quick_f16` (vec.cpp:9 @c1d0e7a00).
pub export var ggml_table_gelu_quick_f16: [1 << 16]c.ggml_fp16_t = undefined;

// -----------------------------------------------------------------------------
// Dot products

/// Ports `ggml_vec_dot_f32` (vec.cpp:11 @c1d0e7a00), the `GGML_SIMD` NEON arm.
///
/// Four `f32x4` accumulators over a step of 16, reduced by a halving tree and
/// then the **pairwise** `vaddvq_f32`, with a scalar `f32` remainder. Every
/// part of that shape is load-bearing: see the table in the file header.
///
/// Parameters:
/// - `n`: element count.
/// - `s`: the result.
/// - `bs`, `bx`, `by`, `nrc`: unused on this path; `nrc` must be 1.
pub export fn ggml_vec_dot_f32(
    n: c_int,
    s: *f32,
    bs: usize,
    x: [*]const f32,
    bx: usize,
    y: [*]const f32,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    impl.assert(nrc == 1, "nrc == 1");
    _ = bs;
    _ = bx;
    _ = by;

    var sumf: f32 = 0.0;
    const np = n & ~@as(c_int, 16 - 1);

    var sum: [4]f32x4 = .{@as(f32x4, @splat(0.0))} ** 4;
    var ax: [4]f32x4 = undefined;
    var ay: [4]f32x4 = undefined;

    var i: c_int = 0;
    while (i < np) : (i += 16) {
        for (0..4) |j| {
            const off: usize = @intCast(i + @as(c_int, @intCast(j)) * 4);
            ax[j] = x[off..][0..4].*;
            ay[j] = y[off..][0..4].*;
            sum[j] = neon.fma_f32(sum[j], ax[j], ay[j]);
        }
    }

    // GGML_F32_VEC_REDUCE: halve, halve, then one pairwise horizontal add.
    sum[0] = add_f32(sum[0], sum[2]);
    sum[1] = add_f32(sum[1], sum[3]);
    sum[0] = add_f32(sum[0], sum[1]);
    sumf = neon.addvq_f32(sum[0]);

    var k: usize = @intCast(np);
    while (k < @as(usize, @intCast(n))) : (k += 1) {
        sumf += x[k] * y[k];
    }

    s.* = sumf;
}

/// Ports `ggml_vec_dot_bf16` (vec.cpp:139 @c1d0e7a00).
///
/// **No SIMD arm compiles here**, so this is the plain scalar loop the C
/// falls back to, accumulating in `ggml_float` — a `double`, not an `f32`.
pub export fn ggml_vec_dot_bf16(
    n: c_int,
    s: *f32,
    bs: usize,
    x: [*]c.ggml_bf16_t,
    bx: usize,
    y: [*]c.ggml_bf16_t,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    impl.assert(nrc == 1, "nrc == 1");
    _ = bs;
    _ = bx;
    _ = by;

    var sumf: GgmlFloat = 0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        sumf += @as(GgmlFloat, impl.bf16ToFp32(x[i].bits) * impl.bf16ToFp32(y[i].bits));
    }

    s.* = @floatCast(sumf);
}

/// Ports `ggml_vec_dot_f16` (vec.cpp:264 @c1d0e7a00), the `GGML_SIMD` NEON arm.
///
/// Four `f16x8` accumulators over a step of 32 — the body really does
/// accumulate in **half precision** — reduced by a halving tree, then widened
/// to two `f32x4`, added, and summed pairwise. The remainder accumulates in
/// `double`, unlike the `f32` kernel's `f32` tail.
pub export fn ggml_vec_dot_f16(
    n: c_int,
    s: *f32,
    bs: usize,
    x: [*]c.ggml_fp16_t,
    bx: usize,
    y: [*]c.ggml_fp16_t,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    impl.assert(nrc == 1, "nrc == 1");
    _ = bs;
    _ = bx;
    _ = by;

    var sumf: GgmlFloat = 0.0;
    const np = n & ~@as(c_int, 32 - 1);

    var sum: [4]f16x8 = .{@as(f16x8, @splat(0.0))} ** 4;

    var i: c_int = 0;
    while (i < np) : (i += 32) {
        for (0..4) |j| {
            const off: usize = @intCast(i + @as(c_int, @intCast(j)) * 8);
            // `ggml_fp16_t` is a `u16` after the import, so the eight
            // lanes are reinterpreted rather than converted -- a load, as in
            // the C's `vld1q_f16`, not a widening.
            const ax: f16x8 = @bitCast(@as(@Vector(8, u16), x[off..][0..8].*));
            const ay: f16x8 = @bitCast(@as(@Vector(8, u16), y[off..][0..8].*));
            sum[j] = fma_f16(sum[j], ax, ay);
        }
    }

    // GGML_F16_VEC_REDUCE: halve, halve, widen both halves, add, pairwise.
    sum[0] = add_f16(sum[0], sum[2]);
    sum[1] = add_f16(sum[1], sum[3]);
    sum[0] = add_f16(sum[0], sum[1]);
    const t0 = cvt_f32_f16_half(sum[0], false);
    const t1 = cvt_f32_f16_half(sum[0], true);
    sumf = neon.addvq_f32(add_f32(t0, t1));

    var k: usize = @intCast(np);
    while (k < @as(usize, @intCast(n))) : (k += 1) {
        sumf += @as(GgmlFloat, impl.fp16ToFp32(x[k]) * impl.fp16ToFp32(y[k]));
    }

    s.* = @floatCast(sumf);
}

// -----------------------------------------------------------------------------
// The vectorized exponential
//
// Ports `ggml_v_expf` (vec.h:1133 @c1d0e7a00), the `__ARM_NEON &&
// __aarch64__` arm. `inline static` in the C, so it has no symbol and is
// private here.
//
// This is ARM's optimised `expf` routine, not a generic approximation: the C's
// own comment upstream records a maximum error of 1.45358 + 0.5 ulp. Every
// constant is a hex float and every fused site is an explicit `vfmaq_f32` or
// `vfmsq_f32`, so the port names the same fusions with `@mulAdd` and the
// result is bit-identical rather than merely close. That is the condition
// `CLAUDE.md` sets for naming a fusion, and `vec_golden.zig` is the something
// that can say it is wrong.

/// Ports `ggml_v_expf` (vec.h:1133 @c1d0e7a00), the NEON arm.
inline fn vExpf(x: f32x4) f32x4 {
    const u32x4 = @Vector(4, u32);
    const u64x2 = @Vector(2, u64);

    const r = dup_n_f32(0x1.8p23);
    const z = neon.fma_f32(r, x, dup_n_f32(0x1.715476p+0));
    const nn = sub_f32(z, r);
    const b = fms_f32(fms_f32(x, nn, dup_n_f32(0x1.62e4p-1)), nn, dup_n_f32(0x1.7f7d1cp-20));

    const e: u32x4 = @as(u32x4, @bitCast(z)) << @splat(23);
    const k: f32x4 = @bitCast(e +% @as(u32x4, @bitCast(dup_n_f32(1))));

    // `vcagtq_f32`: |n| > 126, lane-wise, as an all-ones mask.
    const cmask = @abs(nn) > dup_n_f32(126);
    const cc: u32x4 = @select(u32, cmask, @as(u32x4, @splat(0xFFFFFFFF)), @as(u32x4, @splat(0)));

    const u = neon.mul_f32(b, b);
    const j = neon.fma_f32(
        neon.mul_f32(dup_n_f32(0x1.ffffecp-1), b),
        neon.fma_f32(
            neon.fma_f32(dup_n_f32(0x1.fffdb6p-2), dup_n_f32(0x1.555e66p-3), b),
            neon.fma_f32(dup_n_f32(0x1.573e2ep-5), dup_n_f32(0x1.0e4020p-7), b),
            u,
        ),
        u,
    );

    // `vpaddd_u64(vreinterpretq_u64_u32(c))`: zero only when no lane is set,
    // which is the fast path.
    const cc64: u64x2 = @bitCast(cc);
    if (cc64[0] +% cc64[1] == 0) {
        return neon.fma_f32(k, j, k);
    }

    // `vclezq_f32(n)`: n <= 0, as a mask, then `& 0x82000000`.
    const lez = nn <= dup_n_f32(0);
    const d: u32x4 = @select(u32, lez, @as(u32x4, @splat(0x82000000)), @as(u32x4, @splat(0)));

    const s1: f32x4 = @bitCast(d +% @as(u32x4, @splat(0x7f000000)));
    const s2: f32x4 = @bitCast(e -% d);

    const big = @abs(nn) > dup_n_f32(192);
    const inner = @select(
        f32,
        cmask,
        neon.mul_f32(neon.fma_f32(s2, s2, j), s1),
        neon.fma_f32(k, k, j),
    );
    return @select(f32, big, neon.mul_f32(s1, s1), inner);
}

/// Ports `ggml_v_silu` (vec.h:1157 @c1d0e7a00), the NEON arm:
/// `x / (1 + exp(-x))`.
inline fn vSilu(x: f32x4) f32x4 {
    const one = dup_n_f32(1.0);
    const zero = dup_n_f32(0.0);
    const neg_x = sub_f32(zero, x);
    const exp_neg_x = vExpf(neg_x);
    const one_plus_exp_neg_x = add_f32(one, exp_neg_x);
    return div_f32(x, one_plus_exp_neg_x);
}

/// Ports `ggml_silu_f32` (vec.h:1046 @c1d0e7a00), the scalar tail's version.
///
/// Note this is **not** the same computation as `vSilu`: it divides by
/// `1 + expf(-x)` using libm's `expf`, where the vector path uses ARM's
/// polynomial. The two differ in the last bit, and the C uses each where it
/// uses it.
inline fn siluF32(x: f32) f32 {
    return x / (1.0 + expf(-x));
}

extern fn expf(x: f32) f32;
extern fn logf(x: f32) f32;

// -----------------------------------------------------------------------------
// The SiLU family

/// Ports `ggml_vec_silu_f32` (vec.cpp:380 @c1d0e7a00), the
/// `__ARM_NEON && __aarch64__` arm.
pub export fn ggml_vec_silu_f32(n: c_int, y: [*]f32, x: [*]const f32) callconv(.c) void {
    var i: c_int = 0;
    while (i + 3 < n) : (i += 4) {
        const off: usize = @intCast(i);
        const v = vSilu(x[off..][0..4].*);
        y[off..][0..4].* = v;
    }
    while (i < n) : (i += 1) {
        const off: usize = @intCast(i);
        y[off] = siluF32(x[off]);
    }
}

/// Ports `ggml_vec_swiglu_f32` (vec.cpp:417 @c1d0e7a00), the NEON arm.
pub export fn ggml_vec_swiglu_f32(n: c_int, y: [*]f32, x: [*]const f32, g: [*]const f32) callconv(.c) void {
    var i: c_int = 0;
    while (i + 3 < n) : (i += 4) {
        const off: usize = @intCast(i);
        const v = neon.mul_f32(vSilu(x[off..][0..4].*), g[off..][0..4].*);
        y[off..][0..4].* = v;
    }
    while (i < n) : (i += 1) {
        const off: usize = @intCast(i);
        y[off] = siluF32(x[off]) * g[off];
    }
}

// -----------------------------------------------------------------------------
// Reductions
//
// All three accumulate in `ggml_float` (double) and add the vector partials
// through the **pairwise** `vaddvq_f32`, one group of four at a time — not a
// single reduction at the end. The order is the C's and is visible in the
// last bit.

/// Ports `ggml_vec_cvar_f32` (vec.cpp:455 @c1d0e7a00), the NEON arm.
///
/// Writes `x - mean` into `y` and returns the mean of its squares.
pub export fn ggml_vec_cvar_f32(n: c_int, y: [*]f32, x: [*]const f32, mean: f32) callconv(.c) GgmlFloat {
    var i: c_int = 0;
    var sum: GgmlFloat = 0;

    while (i + 3 < n) : (i += 4) {
        const off: usize = @intCast(i);
        var val = sub_f32(x[off..][0..4].*, dup_n_f32(mean));
        y[off..][0..4].* = val;
        val = neon.mul_f32(val, val);
        sum += @as(GgmlFloat, neon.addvq_f32(val));
    }
    while (i < n) : (i += 1) {
        const off: usize = @intCast(i);
        var val = x[off] - mean;
        y[off] = val;
        val *= val;
        sum += @as(GgmlFloat, val);
    }

    return sum / @as(GgmlFloat, @floatFromInt(n));
}

/// Ports `ggml_vec_soft_max_f32` (vec.cpp:531 @c1d0e7a00), the NEON arm.
///
/// Writes `exp(x - max)` into `y` and returns the sum. The vector path uses
/// `vExpf` and the tail uses libm's `expf`; they differ in the last bit and
/// the C uses each where it uses it.
pub export fn ggml_vec_soft_max_f32(n: c_int, y: [*]f32, x: [*]const f32, max: f32) callconv(.c) GgmlFloat {
    var i: c_int = 0;
    var sum: GgmlFloat = 0;

    while (i + 3 < n) : (i += 4) {
        const off: usize = @intCast(i);
        const val = vExpf(sub_f32(x[off..][0..4].*, dup_n_f32(max)));
        y[off..][0..4].* = val;
        sum += @as(GgmlFloat, neon.addvq_f32(val));
    }
    while (i < n) : (i += 1) {
        const off: usize = @intCast(i);
        const val = expf(x[off] - max);
        sum += @as(GgmlFloat, val);
        y[off] = val;
    }

    return sum;
}

/// Ports `ggml_vec_log_soft_max_f32` (vec.cpp:602 @c1d0e7a00).
///
/// Scalar throughout — the C has no vector arm for this one on any target.
pub export fn ggml_vec_log_soft_max_f32(n: c_int, y: [*]f32, x: [*]const f32, max: f32) callconv(.c) GgmlFloat {
    var sum: GgmlFloat = 0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        const val = x[i] - max;
        y[i] = val;
        sum += @as(GgmlFloat, expf(val));
    }
    return @floatCast(logf(@floatCast(sum)));
}
