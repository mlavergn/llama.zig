//! The CPU backend's SIMD layer: `simd-mappings.h`'s `GGML_F32_VEC` and
//! `GGML_F16_VEC` macros, and the reference compiler's loop vectorizer.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/simd-mappings.h` at v0.3.0
//! (`c1d0e7a00`): the `GGML_SIMD` block, each arm chosen at comptime on
//! `builtin.cpu` as the C's `#if` chooses it. Each declaration names the macro
//! it replaces and the line it is defined on.
//!
//! `StrictLoop` and the three functions after it are **not a port**: they are
//! a property of the clang that compiles the reference, which the C's plain
//! loops inherit. They were measured, and say where.
//!
//! # Why this layer exists
//!
//! `vec.cpp`, `vec.h` and `ops.cpp` write their vector bodies once, against
//! these macros, and `simd-mappings.h` decides per target what a vector is,
//! how many make a step and how a step is reduced. The port does the same:
//! a kernel written against `F32` and `F16` is every arm's kernel, and a new
//! target is a new arm here rather than a new copy of each kernel.
//!
//! What a kernel must not do is reach past this layer for a constant: four
//! lanes, a step of 16 and the pairwise `vaddvq_f32` are NEON's answers, and
//! x86 gives different ones -- `GGML_F32_EPR` is 8 under AVX and 16 under
//! AVX-512, and the reduction ends in `_mm_hadd_ps`.
//!
//! # What is here
//!
//! - `F32` -- `GGML_F32_VEC`: `float32x4_t` on NEON.
//! - `F16` -- `GGML_F16_VEC`, which is **two different things** on NEON:
//!   `float16x8_t` computing in half precision with
//!   `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (every Apple core), and
//!   `float32x4_t` over widened halves without it (`generic`, Zig's
//!   `aarch64-linux` baseline). `F16.half_precision` says which.
//! - `strict_loop`, `strictDot`, `strictLoopVectorized`, `strictSplit` --
//!   how clang vectorizes a plain `acc += a[i] * b[i]` loop, which decides
//!   which of its products are rounded and which fused.
//! - `neon` -- the ACLE intrinsics, for the arms the C writes in intrinsics
//!   directly rather than through these macros: `vec.h`'s `ggml_v_expf` and
//!   its callers, and `sgemm.cpp`'s `tinyBLAS`. Those are per-architecture
//!   code in the C too, and gain an x86 arm beside the NEON one.
//!
//! # Arms
//!
//! - **NEON** (`__ARM_NEON && __ARM_FEATURE_FMA && __ARM_FP16_FORMAT_IEEE`),
//!   with both of its `F16` variants. SVE's is not ported: no Apple core has
//!   SVE, and Zig's Linux baseline does not enable it.
//! - **AVX-512** (`__AVX512F__`) -- x86-64-v4, Decision 53's second level,
//!   with its `GGML_F32Cx16` fallback for `F16`. The `__AVX512FP16__` arm is
//!   a compile error: no CPU this project gates on has it.
//! - **AVX** with FMA and F16C -- x86-64-v3, Decision 53's first level. AVX
//!   without FMA, and SSE, are compile errors: no target this project builds
//!   needs them.
//!
//! Any other architecture is a compile error.

const std = @import("std");
const builtin = @import("builtin");

/// The NEON intrinsics, as Zig vector code; see `quants/arm/neon.zig`.
/// Only meaningful on `aarch64`, and only for code the C writes in NEON
/// intrinsics itself: kernels written against `GGML_F32_VEC` use `F32`.
pub const neon = @import("quants/arm/neon.zig");

/// `GGML_F32_VEC` and its operations, for this build's target.
pub const F32 = switch (builtin.cpu.arch) {
    .aarch64 => NeonF32,
    .x86_64 => x86Arm(Avx512F32, AvxF32),
    else => @compileError("no SIMD arm for " ++ @tagName(builtin.cpu.arch)),
};

/// `GGML_F16_VEC` and its operations, for this build's target.
pub const F16 = switch (builtin.cpu.arch) {
    .aarch64 => if (neon.has_fp16_va) NeonF16x8 else NeonF32Cx4,
    .x86_64 => x86Arm(Avx512F32Cx16, AvxF32Cx8),
    else => @compileError("no SIMD arm for " ++ @tagName(builtin.cpu.arch)),
};

/// Whether this build's x86 CPU has `feature`: the C's `__AVX2__` and
/// friends, which clang defines from the same feature set.
fn x86Has(comptime feature: std.Target.x86.Feature) bool {
    return builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, feature);
}

/// The x86 arm `simd-mappings.h` would take, in its order -- `__AVX512F__`
/// before `__AVX__`, which is before `__SSE3__` -- refusing the ones not
/// ported rather than falling through to a different one.
fn x86Arm(comptime Avx512: type, comptime Avx: type) type {
    if (x86Has(.avx512f)) {
        if (x86Has(.avx512fp16)) @compileError("the __AVX512FP16__ arm of simd-mappings.h is not ported");
        return Avx512;
    }
    if (!x86Has(.avx)) @compileError("the SSE arm of simd-mappings.h is not ported");
    if (!x86Has(.fma) or !x86Has(.f16c)) @compileError("the AVX arm is ported only with FMA and F16C (x86-64-v3)");
    return Avx;
}

/// The halving loops every `GGML_*_REDUCE` opens with: three passes, each
/// folding the upper half of the registers into the lower, so that `x[0]`
/// holds the sum. With four registers the third pass is empty.
inline fn halve(comptime V: type, comptime n: usize, x: *[n]V) void {
    inline for (.{ n >> 1, n >> 2, n >> 3 }) |offset| {
        inline for (0..offset) |i| x[i] = x[i] + x[offset + i];
    }
}

// -----------------------------------------------------------------------------
// NEON

/// `GGML_F32_VEC` on NEON: `GGML_F32x4`, `float32x4_t`.
const NeonF32 = struct {
    /// Ports `GGML_F32_STEP`, `GGML_F32_EPR` (simd-mappings.h:337, 338
    /// @c1d0e7a00): four registers of four lanes per step.
    pub const step: usize = 16;
    pub const epr: usize = 4;

    /// Ports `GGML_F32_ARR` (simd-mappings.h:1313 @c1d0e7a00).
    pub const arr: usize = step / epr;

    /// `GGML_F32x4`.
    pub const Vec = neon.f32x4;

    /// Ports `GGML_F32x4_ZERO` (simd-mappings.h:341 @c1d0e7a00).
    pub const zero: Vec = @splat(0.0);

    /// Ports `GGML_F32x4_SET1` (simd-mappings.h:342 @c1d0e7a00).
    pub inline fn set1(x: f32) Vec {
        return @splat(x);
    }

    /// Ports `GGML_F32x4_LOAD` (simd-mappings.h:343 @c1d0e7a00).
    pub inline fn load(p: [*]const f32) Vec {
        return p[0..epr].*;
    }

    /// Ports `GGML_F32x4_STORE` (simd-mappings.h:344 @c1d0e7a00).
    pub inline fn store(p: [*]f32, v: Vec) void {
        p[0..epr].* = v;
    }

    /// Ports `GGML_F32x4_FMA` (simd-mappings.h:345 @c1d0e7a00): `a + b * cc`,
    /// fused -- `vfmaq_f32`.
    pub inline fn fma(a: Vec, b: Vec, cc: Vec) Vec {
        return neon.fma_f32(a, b, cc);
    }

    /// Ports `GGML_F32x4_ADD` (simd-mappings.h:346 @c1d0e7a00).
    pub inline fn add(a: Vec, b: Vec) Vec {
        return a + b;
    }

    /// Ports `GGML_F32x4_MUL` (simd-mappings.h:347 @c1d0e7a00).
    pub inline fn mul(a: Vec, b: Vec) Vec {
        return neon.mul_f32(a, b);
    }

    /// Ports `GGML_F32x4_REDUCE` (simd-mappings.h:349 @c1d0e7a00): halve,
    /// halve, then one **pairwise** `vaddvq_f32`.
    ///
    /// The pairwise step is load-bearing. `neon.addvq_f32` reduces as
    /// `(l0+l1)+(l2+l3)`; `@reduce(.Add, ...)` is ordered and differs on
    /// 23.5% of random lane vectors. See `CLAUDE.md`, "Porting notes".
    ///
    /// The C widens the result to `ggml_float` and every caller narrows it
    /// straight back to `float`, which is lossless, so the round trip is
    /// left out. `x` is overwritten, as the macro overwrites it.
    pub inline fn reduce(x: *[arr]Vec) f32 {
        halve(Vec, arr, x);
        return neon.addvq_f32(x[0]);
    }
};

/// `GGML_F16_VEC` on NEON with `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`:
/// `GGML_F16x8`, `float16x8_t`, and the arithmetic **is** half precision.
const NeonF16x8 = struct {
    /// Ports `GGML_F16_STEP`, `GGML_F16_EPR` (simd-mappings.h:379, 380
    /// @c1d0e7a00), the `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` arm.
    pub const step: usize = 32;
    pub const epr: usize = 8;

    /// Ports `GGML_F16_ARR` (simd-mappings.h:1314 @c1d0e7a00).
    pub const arr: usize = step / epr;

    /// Whether `Vec`'s arithmetic rounds to half precision. True here; see
    /// `NeonF32Cx4` for the arm where it does not.
    pub const half_precision = true;

    /// `GGML_F16x8`.
    pub const Vec = neon.f16x8;

    /// Ports `GGML_F16x8_ZERO` (simd-mappings.h:383 @c1d0e7a00).
    pub const zero: Vec = @splat(0.0);

    /// Ports `GGML_F16x8_LOAD` (simd-mappings.h:385 @c1d0e7a00).
    ///
    /// `ggml_fp16_t` is a `u16` after the import, so the eight lanes are
    /// reinterpreted rather than converted -- a load, as `vld1q_f16` is, not
    /// a widening.
    pub inline fn load(p: [*]const u16) Vec {
        return @bitCast(@as(@Vector(epr, u16), p[0..epr].*));
    }

    /// Ports `GGML_F16x8_FMA` (simd-mappings.h:387 @c1d0e7a00): `a + b * cc`
    /// in half precision, fused -- `vfmaq_f16`.
    pub inline fn fma(a: Vec, b: Vec, cc: Vec) Vec {
        return neon.fma_f16(a, b, cc);
    }

    /// Ports `GGML_F16x8_ADD` (simd-mappings.h:388 @c1d0e7a00).
    pub inline fn add(a: Vec, b: Vec) Vec {
        return neon.add_f16(a, b);
    }

    /// Ports `GGML_F16x8_REDUCE` (simd-mappings.h:390 @c1d0e7a00): halve,
    /// halve **in half precision**, then widen both halves of `x[0]` to
    /// `f32x4`, add, and one pairwise `vaddvq_f32`.
    pub inline fn reduce(x: *[arr]Vec) f32 {
        halve(Vec, arr, x);
        const t0 = neon.cvt_f32_f16_half(x[0], false);
        const t1 = neon.cvt_f32_f16_half(x[0], true);
        return neon.addvq_f32(t0 + t1);
    }
};

/// `GGML_F16_VEC` on NEON without `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`:
/// `GGML_F32Cx4`, a `float32x4_t` over four widened halves, so the
/// arithmetic is `f32`'s.
const NeonF32Cx4 = struct {
    /// Ports `GGML_F16_STEP`, `GGML_F16_EPR` (simd-mappings.h:422, 423
    /// @c1d0e7a00), the arm without `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`.
    pub const step: usize = 16;
    pub const epr: usize = 4;

    /// Ports `GGML_F16_ARR` (simd-mappings.h:1314 @c1d0e7a00).
    pub const arr: usize = step / epr;

    /// Whether `Vec`'s arithmetic rounds to half precision. False: each load
    /// widens exactly, `f32` computes, and a store narrows once -- which is
    /// what the C's scalar `f16` tails do, so a kernel's body and tail agree
    /// element for element here.
    pub const half_precision = false;

    /// `GGML_F32Cx4`.
    pub const Vec = neon.f32x4;

    /// Ports `GGML_F32Cx4_ZERO` (simd-mappings.h:426 @c1d0e7a00).
    pub const zero: Vec = @splat(0.0);

    /// Ports `GGML_F32Cx4_LOAD` (simd-mappings.h:428 @c1d0e7a00): `vld1_f16`
    /// then `vcvt_f32_f16`, an exact widening.
    pub inline fn load(p: [*]const u16) Vec {
        return neon.cvt_f32_f16(neon.load_f16x4(p));
    }

    /// Ports `GGML_F32Cx4_FMA` (simd-mappings.h:430 @c1d0e7a00): `a + b * cc`
    /// in `f32`, fused -- `vfmaq_f32`.
    pub inline fn fma(a: Vec, b: Vec, cc: Vec) Vec {
        return neon.fma_f32(a, b, cc);
    }

    /// Ports `GGML_F32Cx4_ADD` (simd-mappings.h:431 @c1d0e7a00).
    pub inline fn add(a: Vec, b: Vec) Vec {
        return a + b;
    }

    /// Ports `GGML_F32Cx4_REDUCE` (simd-mappings.h:433 @c1d0e7a00), which is
    /// `GGML_F32x4_REDUCE`.
    pub inline fn reduce(x: *[arr]Vec) f32 {
        return NeonF32.reduce(x);
    }
};

// -----------------------------------------------------------------------------
// AVX

/// `GGML_F32_VEC` on AVX: `GGML_F32x8`, `__m256`.
const AvxF32 = struct {
    /// Ports `GGML_F32_STEP`, `GGML_F32_EPR` (simd-mappings.h:587, 588
    /// @c1d0e7a00): four registers of eight lanes per step.
    pub const step: usize = 32;
    pub const epr: usize = 8;

    /// Ports `GGML_F32_ARR` (simd-mappings.h:1313 @c1d0e7a00).
    pub const arr: usize = step / epr;

    /// `GGML_F32x8`.
    pub const Vec = @Vector(epr, f32);

    /// Ports `GGML_F32x8_ZERO` (simd-mappings.h:591 @c1d0e7a00).
    pub const zero: Vec = @splat(0.0);

    /// Ports `GGML_F32x8_SET1` (simd-mappings.h:592 @c1d0e7a00).
    pub inline fn set1(x: f32) Vec {
        return @splat(x);
    }

    /// Ports `GGML_F32x8_LOAD` (simd-mappings.h:593 @c1d0e7a00).
    pub inline fn load(p: [*]const f32) Vec {
        return p[0..epr].*;
    }

    /// Ports `GGML_F32x8_STORE` (simd-mappings.h:594 @c1d0e7a00).
    pub inline fn store(p: [*]f32, v: Vec) void {
        p[0..epr].* = v;
    }

    /// Ports `GGML_F32x8_FMA` (simd-mappings.h:596 @c1d0e7a00), the
    /// `__FMA__` arm: `_mm256_fmadd_ps(b, c, a)`, so `a + b * c`, fused.
    pub inline fn fma(a: Vec, b: Vec, cc: Vec) Vec {
        return @mulAdd(Vec, b, cc, a);
    }

    /// Ports `GGML_F32x8_ADD` (simd-mappings.h:600 @c1d0e7a00).
    pub inline fn add(a: Vec, b: Vec) Vec {
        return a + b;
    }

    /// Ports `GGML_F32x8_MUL` (simd-mappings.h:601 @c1d0e7a00).
    pub inline fn mul(a: Vec, b: Vec) Vec {
        return a * b;
    }

    /// Ports `GGML_F32x8_REDUCE` (simd-mappings.h:602 @c1d0e7a00): halve,
    /// halve, then fold the upper 128 bits onto the lower and two
    /// `_mm_hadd_ps`.
    ///
    /// The two horizontal adds are pairwise, like NEON's `vaddvq_f32`: the
    /// first gives `(l0+l1, l2+l3, …)`, the second their sum, so the result
    /// is `(l0+l1)+(l2+l3)` of the folded four lanes. Measured on the
    /// x86-64-v3 reference's `ggml_vec_dot_f32`: `vextractf128`, `vaddps`,
    /// `vhaddps`, `vhaddps`.
    pub inline fn reduce(x: *[arr]Vec) f32 {
        halve(Vec, arr, x);
        const lo: @Vector(4, f32) = .{ x[0][0], x[0][1], x[0][2], x[0][3] };
        const hi: @Vector(4, f32) = .{ x[0][4], x[0][5], x[0][6], x[0][7] };
        const t0 = lo + hi;
        return (t0[0] + t0[1]) + (t0[2] + t0[3]);
    }
};

// `.x86_64_sysv` for the reason `cpu/convert.zig` gives: Windows x64 would
// otherwise pass the vector by reference.
extern fn @"llvm.x86.vcvtps2ph.256"(@Vector(8, f32), i32) callconv(.{ .x86_64_sysv = .{} }) @Vector(8, i16);

/// `GGML_F16_VEC` on AVX: `GGML_F32Cx8`, a `__m256` over eight halves widened
/// with F16C, so the arithmetic is `f32`'s -- AVX has no half-precision
/// arithmetic.
const AvxF32Cx8 = struct {
    /// Ports `GGML_F16_STEP`, `GGML_F16_EPR` (simd-mappings.h:635, 636
    /// @c1d0e7a00).
    pub const step: usize = 32;
    pub const epr: usize = 8;

    /// Ports `GGML_F16_ARR` (simd-mappings.h:1314 @c1d0e7a00).
    pub const arr: usize = step / epr;

    /// Whether `Vec`'s arithmetic rounds to half precision. False, as for
    /// NEON's `GGML_F32Cx4`: a kernel's body and its scalar `f16` tail agree
    /// element for element.
    pub const half_precision = false;

    /// `GGML_F32Cx8`.
    pub const Vec = @Vector(epr, f32);

    /// Ports `GGML_F32Cx8_ZERO` (simd-mappings.h:641 @c1d0e7a00).
    pub const zero: Vec = @splat(0.0);

    /// Ports `GGML_F32Cx8_SET1` (simd-mappings.h:642 @c1d0e7a00).
    pub inline fn set1(x: f32) Vec {
        return @splat(x);
    }

    /// Ports `GGML_F32Cx8_LOAD` (simd-mappings.h:646 @c1d0e7a00), the
    /// `__F16C__` arm: `_mm256_cvtph_ps`, an exact widening.
    pub inline fn load(p: [*]const u16) Vec {
        const h: @Vector(epr, f16) = @bitCast(@as(@Vector(epr, u16), p[0..epr].*));
        return @floatCast(h);
    }

    /// Ports `GGML_F32Cx8_STORE` (simd-mappings.h:647 @c1d0e7a00), the
    /// `__F16C__` arm: `_mm256_cvtps_ph(y, 0)`, through LLVM's own intrinsic.
    ///
    /// **Not** `@floatCast`. The instruction rounds to nearest-even exactly
    /// as the software narrowing does, but keeps a NaN's payload, which
    /// ggml's software routine does not; a Zig cast leaves NaN payloads
    /// unspecified. So the vector body and the scalar tail of a C kernel can
    /// disagree on a NaN, and the port has to take the same one each does.
    pub inline fn store(p: [*]u16, v: Vec) void {
        p[0..epr].* = @bitCast(@"llvm.x86.vcvtps2ph.256"(v, 0));
    }

    /// Ports `GGML_F32Cx8_FMA` (simd-mappings.h:670 @c1d0e7a00), which is
    /// `GGML_F32x8_FMA`.
    pub inline fn fma(a: Vec, b: Vec, cc: Vec) Vec {
        return AvxF32.fma(a, b, cc);
    }

    /// Ports `GGML_F32Cx8_ADD` (simd-mappings.h:671 @c1d0e7a00).
    pub inline fn add(a: Vec, b: Vec) Vec {
        return a + b;
    }

    /// Ports `GGML_F32Cx8_MUL` (simd-mappings.h:672 @c1d0e7a00).
    pub inline fn mul(a: Vec, b: Vec) Vec {
        return a * b;
    }

    /// Ports `GGML_F32Cx8_REDUCE` (simd-mappings.h:673 @c1d0e7a00), which is
    /// `GGML_F32x8_REDUCE`.
    pub inline fn reduce(x: *[arr]Vec) f32 {
        return AvxF32.reduce(x);
    }
};

// -----------------------------------------------------------------------------
// AVX-512

/// The tree `_mm512_reduce_add_ps` reduces in. clang lowers it to a
/// reassociable `llvm.vector.reduce.fadd`, whose order is the backend's;
/// measured on the x86-64-v4 reference at `-O2`: fold the upper 256 bits onto
/// the lower, then the upper 128, then `vshufpd` (lanes 2 and 3 onto 0 and 1),
/// then `vmovshdup` (lane 1 onto 0).
pub inline fn reduceAdd512(v: @Vector(16, f32)) f32 {
    var t8: @Vector(8, f32) = undefined;
    inline for (0..8) |i| t8[i] = v[i] + v[i + 8];
    var t4: @Vector(4, f32) = undefined;
    inline for (0..4) |i| t4[i] = t8[i] + t8[i + 4];
    const t2: @Vector(2, f32) = .{ t4[0] + t4[2], t4[1] + t4[3] };
    return t2[0] + t2[1];
}

// `.x86_64_sysv` for the reason `cpu/convert.zig` gives. This is the
// intrinsic `_mm512_cvtps_ph` compiles to: the masked form, unmasked.
extern fn @"llvm.x86.avx512.mask.vcvtps2ph.512"(@Vector(16, f32), i32, @Vector(16, i16), u16) callconv(.{ .x86_64_sysv = .{} }) @Vector(16, i16);

/// `GGML_F32_VEC` on AVX-512: `GGML_F32x16`, `__m512`.
const Avx512F32 = struct {
    /// Ports `GGML_F32_STEP`, `GGML_F32_EPR` (simd-mappings.h:452, 453
    /// @c1d0e7a00): four registers of sixteen lanes per step.
    pub const step: usize = 64;
    pub const epr: usize = 16;

    /// Ports `GGML_F32_ARR` (simd-mappings.h:1313 @c1d0e7a00).
    pub const arr: usize = step / epr;

    /// `GGML_F32x16`.
    pub const Vec = @Vector(epr, f32);

    /// Ports `GGML_F32x16_ZERO` (simd-mappings.h:456 @c1d0e7a00).
    pub const zero: Vec = @splat(0.0);

    /// Ports `GGML_F32x16_SET1` (simd-mappings.h:457 @c1d0e7a00).
    pub inline fn set1(x: f32) Vec {
        return @splat(x);
    }

    /// Ports `GGML_F32x16_LOAD` (simd-mappings.h:458 @c1d0e7a00).
    pub inline fn load(p: [*]const f32) Vec {
        return p[0..epr].*;
    }

    /// Ports `GGML_F32x16_STORE` (simd-mappings.h:459 @c1d0e7a00).
    pub inline fn store(p: [*]f32, v: Vec) void {
        p[0..epr].* = v;
    }

    /// Ports `GGML_F32x16_FMA` (simd-mappings.h:461 @c1d0e7a00):
    /// `_mm512_fmadd_ps(b, c, a)`, so `a + b * c`, fused.
    pub inline fn fma(a: Vec, b: Vec, cc: Vec) Vec {
        return @mulAdd(Vec, b, cc, a);
    }

    /// Ports `GGML_F32x16_ADD` (simd-mappings.h:462 @c1d0e7a00).
    pub inline fn add(a: Vec, b: Vec) Vec {
        return a + b;
    }

    /// Ports `GGML_F32x16_MUL` (simd-mappings.h:463 @c1d0e7a00).
    pub inline fn mul(a: Vec, b: Vec) Vec {
        return a * b;
    }

    /// Ports `GGML_F32x16_REDUCE` (simd-mappings.h:464 @c1d0e7a00): halve,
    /// halve, then `_mm512_reduce_add_ps`, in `reduceAdd512`'s tree.
    pub inline fn reduce(x: *[arr]Vec) f32 {
        halve(Vec, arr, x);
        return reduceAdd512(x[0]);
    }
};

/// `GGML_F16_VEC` on AVX-512 without `__AVX512FP16__`: `GGML_F32Cx16`, a
/// `__m512` over sixteen widened halves, so the arithmetic is `f32`'s.
const Avx512F32Cx16 = struct {
    /// Ports `GGML_F16_STEP`, `GGML_F16_EPR` (simd-mappings.h:537, 538
    /// @c1d0e7a00), the arm without `__AVX512FP16__`.
    pub const step: usize = 64;
    pub const epr: usize = 16;

    /// Ports `GGML_F16_ARR` (simd-mappings.h:1314 @c1d0e7a00).
    pub const arr: usize = step / epr;

    /// Whether `Vec`'s arithmetic rounds to half precision. False, as for
    /// AVX's `GGML_F32Cx8`.
    pub const half_precision = false;

    /// `GGML_F32Cx16`.
    pub const Vec = @Vector(epr, f32);

    /// Ports `GGML_F32Cx16_ZERO` (simd-mappings.h:541 @c1d0e7a00).
    pub const zero: Vec = @splat(0.0);

    /// Ports `GGML_F32Cx16_SET1` (simd-mappings.h:542 @c1d0e7a00).
    pub inline fn set1(x: f32) Vec {
        return @splat(x);
    }

    /// Ports `GGML_F32Cx16_LOAD` (simd-mappings.h:546 @c1d0e7a00):
    /// `_mm512_cvtph_ps`, an exact widening -- an `fpext` in the
    /// reference's IR.
    pub inline fn load(p: [*]const u16) Vec {
        const h: @Vector(epr, f16) = @bitCast(@as(@Vector(epr, u16), p[0..epr].*));
        return @floatCast(h);
    }

    /// Ports `GGML_F32Cx16_STORE` (simd-mappings.h:547 @c1d0e7a00):
    /// `_mm512_cvtps_ph(y, 0)`, through LLVM's own intrinsic, for the NaN
    /// payloads `AvxF32Cx8.store` explains.
    pub inline fn store(p: [*]u16, v: Vec) void {
        p[0..epr].* = @bitCast(@"llvm.x86.avx512.mask.vcvtps2ph.512"(v, 0, @splat(0), 0xffff));
    }

    /// Ports `GGML_F32Cx16_FMA` (simd-mappings.h:549 @c1d0e7a00).
    pub inline fn fma(a: Vec, b: Vec, cc: Vec) Vec {
        return Avx512F32.fma(a, b, cc);
    }

    /// Ports `GGML_F32Cx16_ADD` (simd-mappings.h:550 @c1d0e7a00).
    pub inline fn add(a: Vec, b: Vec) Vec {
        return a + b;
    }

    /// Ports `GGML_F32Cx16_MUL` (simd-mappings.h:551 @c1d0e7a00).
    pub inline fn mul(a: Vec, b: Vec) Vec {
        return a * b;
    }

    /// Ports `GGML_F32Cx16_REDUCE` (simd-mappings.h:552 @c1d0e7a00), which
    /// halves over `GGML_F32_ARR` -- the same count here -- and ends in
    /// `_mm512_reduce_add_ps`.
    pub inline fn reduce(x: *[arr]Vec) f32 {
        return Avx512F32.reduce(x);
    }
};

// -----------------------------------------------------------------------------
// The reference compiler's loop vectorizer

/// How clang's loop vectorizer shapes a strict `acc += a[i] * b[i]` loop for
/// the CPU this binary targets. Not a port of anything in the C: a property
/// of the compiler that builds the reference, which the C's arithmetic
/// inherits.
///
/// At `-ffp-contract=on` such a loop becomes a strict in-order reduction.
/// Whole vector groups have their products **rounded** (`fmul.4s`) and added
/// one lane at a time; only what is left over runs scalar and **fused**
/// (`fmadd`). Which elements fall in which part depends on the vector body's
/// width, and that is `VF × interleave`, where the interleave factor comes
/// from the target CPU's cost model -- so **the CPU model, not its feature
/// set, decides the reference's bits**.
pub const StrictLoop = struct {
    /// Products rounded per group: elements before the last `t % group` of a
    /// `t`-element loop are rounded, the rest fused. Also the fewest
    /// iterations that take the vector form at all. Null where the cost
    /// model never vectorizes such a loop, so every product is fused.
    group: ?usize,
    /// The vector body's footprint in bytes: a loaded array starting less
    /// than this far below the stored one sends the loop down the scalar,
    /// all-fused path.
    footprint: usize,
};

/// `StrictLoop` for this build's CPU model.
///
/// Measured from the reference `ops.o`'s LLVM IR and disassembly at `-O3`,
/// `ggml_compute_forward_ssm_scan` and `ssm_conv`:
///
/// - **Apple A14 and later** (`apple_a14`-`a19`, `apple_m1`-`m5`; LLVM's
///   interleave factor 4, its `max_interleave_factor_4` feature): a body of
///   four `fmul.4s`, 16 elements, entered at 16 iterations, then a vectorized
///   epilogue of one `fmul.4s` entered at 4. So groups of 4 are rounded,
///   `t % 4` fused, and nothing vectorizes below 4. Footprint 64 bytes.
/// - **`generic`** (Zig's `aarch64-linux` baseline; interleave 2): a body of
///   two `fmul.4s`, 8 elements, entered at 8, and **no** vectorized
///   epilogue. Groups of 8, `t % 8` fused. Footprint 32 bytes.
/// - **Apple A7 to A13** (`apple_a7`-`a13`; interleave 2, the feature
///   absent): `generic`'s shape exactly, measured on the `apple_a7` and
///   `apple_a12` references -- two `fmul.4s`, an 8-iteration guard, a
///   32-byte overlap guard. The CPU model is not what decides: LLVM gives
///   every Apple core the same cost model and only the A14 and later the
///   wider interleave. Keying this on an `apple_` prefix failed `ops-diff`
///   at `apple_a7` on `ssm_conv`, `ssm_scan` and `conv_3d`, which the
///   xcframework's device slices build for.
/// - **`x86_64_v3`**: **never vectorized.** The reference's
///   `ggml_vec_dot_f32` tail is a scalar `vfmadd231ss` loop, `ssm_conv` has
///   no vector instruction at all, and `ssm_scan`'s only `vmulps` are its
///   `GGML_SIMD` body's -- the x86 cost model does not find an in-order
///   reduction worth it. Every product fused.
/// - **`x86_64_v4`**: the same, measured the same way on the v4 reference:
///   `ggml_vec_dot_f32`'s tail is scalar `vfmadd231ss`, `ssm_conv` has no
///   vector instruction, and `ssm_scan`'s eight `vmulps` are its `GGML_SIMD`
///   body's, on `zmm`. **`tigerlake`** likewise, measured on its reference.
///
/// Any other model is a compile error rather than a guess: these sites are
/// bit-exact or wrong, and `NOTES.md`, "Stage 5: the measurement", has how
/// to measure one.
pub const strict_loop: StrictLoop = blk: {
    const model = builtin.cpu.model.name;
    if (builtin.cpu.arch == .aarch64 and std.mem.startsWith(u8, model, "apple_"))
        break :blk if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .max_interleave_factor_4))
            .{ .group = 4, .footprint = 64 }
        else
            .{ .group = 8, .footprint = 32 };
    if (builtin.cpu.arch == .aarch64 and std.mem.eql(u8, model, "generic"))
        break :blk .{ .group = 8, .footprint = 32 };
    if (builtin.cpu.arch == .x86_64 and (std.mem.eql(u8, model, "x86_64_v3") or std.mem.eql(u8, model, "x86_64_v4") or std.mem.eql(u8, model, "tigerlake")))
        break :blk .{ .group = null, .footprint = 0 };
    @compileError("the loop vectorizer's shape is not measured for CPU model '" ++ model ++ "'");
};

/// The loop `for (; i < n; ++i) sumf += x[i]*y[i];` **as the reference
/// compiler emits it**, which is not as it reads.
///
/// `sumf += x[i]*y[i]` is one expression, so at `-ffp-contract=on` clang
/// marks it `llvm.fmuladd` -- "fused or not, the backend's choice". The loop
/// vectorizer then turns the loop into a strict, in-order reduction: a group
/// of products per step with `fmul.4s`, **unfused**, added into `sumf` one
/// lane at a time with scalar `fadd` so the order is kept. Only what is left
/// after whole groups runs scalar, and that part is `fmadd`, **fused**.
///
/// So over `t` elements the first `t - t % strict_loop.group` are rounded
/// product then add, and the rest are fused. Measured, not read off the
/// source: the disassembly of the reference `vec.o` shows the split, and
/// this model matches the reference bit for bit at every `n` from 1 to 47,
/// 3,000 random rows each. Purely fused was 1 ULP out on the conv kernels'
/// two-element dot products; purely unfused was out at `n = 11`.
///
/// **Only for loops the compiler actually vectorizes**, which is a
/// cost-model decision, not something the source states. Each call site is
/// one that `make ops-diff` or the goldens showed needs it.
///
/// Parameters:
/// - `x`, `y`: the operands.
/// - `start`, `end`: the half-open range.
/// - `acc`: the running sum.
///
/// Return: the sum with the range folded in.
pub inline fn strictDot(x: [*]const f32, y: [*]const f32, start: usize, end: usize, acc: f32) f32 {
    var sumf = acc;
    const t = end - start;
    const group = strict_loop.group orelse t + 1;
    const split = start + (t - t % group);
    var k = start;
    while (k < split) : (k += 1) sumf += x[k] * y[k];
    while (k < end) : (k += 1) sumf = @mulAdd(f32, x[k], y[k], sumf);
    return sumf;
}

/// Whether the reference compiler runs the **vectorized** form of a plain C
/// loop that reduces `acc += a[i] * b[i]` while storing to `store[i]`.
///
/// It only takes that form when two runtime guards pass, read off the LLVM
/// IR of the same loop compiled by the same toolchain:
///
/// - at least `strict_loop.group` iterations -- four on an Apple A14 or
///   later, eight at `generic` and on earlier Apple cores -- else the scalar
///   loop runs throughout; and
/// - for every array the loop loads, `(store_row - load_row)` as an
///   **unsigned** byte difference is at least `strict_loop.footprint`. A
///   load that starts closer below the store could be overwritten before it
///   is read, so the loop falls back to scalar.
///
/// When it does vectorize, products are rounded in groups of
/// `strict_loop.group` and the remainder fused; `strictSplit` gives where
/// that changes.
///
/// The second guard is not academic. `ssm_scan` reads the previous state from
/// the buffer it is writing for every token after the first (`s0 = s`), so
/// the difference is zero and those tokens take the all-fused scalar loop
/// while the first one does not.
///
/// Parameters:
/// - `trip`: the loop's iteration count.
/// - `store`: the first address the loop stores to.
/// - `loads`: the first address of each array the loop loads.
///
/// Return: true when the vectorized form runs.
pub inline fn strictLoopVectorized(trip: i64, store: *const anyopaque, loads: []const *const anyopaque) bool {
    const group = strict_loop.group orelse return false;
    if (trip < group) return false;
    const st = @intFromPtr(store);
    for (loads) |l| {
        if (st -% @intFromPtr(l) < strict_loop.footprint) return false;
    }
    return true;
}

/// Whether the reference compiler treats libm calls as writing `errno`:
/// clang's `-fmath-errno`, its default on Linux and off on Darwin. Not a port
/// of anything; a property of the reference build that changes its bits in
/// two ways, both measured on the `aarch64-linux-gnu` reference against the
/// `aarch64-macos` one at the same CPU:
///
/// - a `sinf`/`cosf` pair on one argument is **not** folded into one
///   combined call (`ops/common.sinCos`); and
/// - a strict loop that calls into libm is **not** vectorized, so it runs
///   scalar and all fused. `ssm_scan`'s Mamba-1 loop calls `expf`: on Darwin
///   its `ssm_scan` has twelve `expf` call sites, eight of them one per
///   lane, and on Linux four.
///
/// The other Linux-only libm imports were checked and do not move a bit:
/// `sqrtf` beside an `fsqrt` for the `errno` path, `lroundf` where Darwin
/// inlines `fcvtas`.
pub const math_errno = !builtin.os.tag.isDarwin();

/// Where a vectorized strict loop of `trip` iterations starting at `start`
/// stops rounding and starts fusing: past whole groups of
/// `strict_loop.group`. See `strictLoopVectorized`.
pub inline fn strictSplit(start: i64, trip: i64) i64 {
    const group: i64 = strict_loop.group orelse return start;
    return start + trip - @rem(trip, group);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the F32 reduce halves, then adds pairwise" {
    // Pairwise: (1e8 + 1) + (-1e8 + 1); each inner sum rounds back to +-1e8,
    // so the result is 0. An ordered sum would give ((1e8 + 1) - 1e8) + 1 = 1.
    var x: [F32.arr]F32.Vec = @splat(F32.zero);
    x[0] = .{ 1e8, 1, -1e8, 1 };
    try std.testing.expectEqual(@as(f32, 0), F32.reduce(&x));

    // The halving order: x[0] + x[2] and x[1] + x[3] first. With the
    // registers below, (x0 + x2) + (x1 + x3) is exact in every lane while
    // the ordered fold ((x0 + x1) + x2) + x3 loses the ones.
    x = .{ @splat(1e8), @splat(1), @splat(-1e8), @splat(1) };
    try std.testing.expectEqual(@as(f32, 8), F32.reduce(&x));
}

test "F16 loads the halves it is given, whichever arm" {
    const h = [_]u16{ 0x3c00, 0x4000, 0x4200, 0x4400, 0x3c00, 0x4000, 0x4200, 0x4400 };
    var x: [F16.arr]F16.Vec = @splat(F16.zero);
    x[0] = F16.load(&h);
    // 1 + 2 + 3 + 4, once per load: exact in either precision.
    try std.testing.expectEqual(@as(f32, @floatFromInt(10 * (F16.epr / 4))), F16.reduce(&x));
}

test "strictSplit leaves the remainder of whole groups" {
    const g: i64 = strict_loop.group orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(i64, 5 + 2 * g), strictSplit(5, 2 * g + 1));
    try std.testing.expectEqual(@as(i64, 0), strictSplit(0, g - 1));
}
