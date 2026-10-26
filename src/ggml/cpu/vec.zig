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
//! | `f32`  | `GGML_F32_ARR` × `GGML_F32_VEC`, `GGML_F32_VEC_FMA` | `GGML_F32_VEC_REDUCE` | `f32` scalar, as the vectorizer emits it |
//! | `f16`  | `GGML_F16_ARR` × `GGML_F16_VEC`, `GGML_F16_VEC_FMA` | `GGML_F16_VEC_REDUCE` | `double` scalar |
//! | `bf16` | none | none | `double` scalar |
//!
//! The two dot products are written against `simd.zig`, as `vec.cpp` writes
//! them against `simd-mappings.h`, so each is one body for every arm. On
//! NEON that is four `f32x4` over a step of 16 for `f32`, and for `f16`
//! either four `f16x8` over 32 accumulating in **half precision** or four
//! widened `f32x4` over 16 -- `simd.F16.half_precision` says which. The two
//! tails accumulate in different types from each other, which is why
//! `vec_golden.zig` captures a second length that leaves a remainder.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const simd = @import("simd.zig");
const neon = simd.neon;
const c = impl.c;

const F32 = simd.F32;
const F16 = simd.F16;

/// Mirrors `ggml_float` (vec.h:15 @c1d0e7a00), the `double` the
/// reductions accumulate in. Named rather than written as `f64` so the C's
/// casts stay readable at each site.
const GgmlFloat = f64;

// -----------------------------------------------------------------------------
// The vector width of `vec.cpp`'s own arms
//
// `vec.cpp`'s SiLU and softmax arms and `vec.h`'s `ggml_v_expf` are written
// in each architecture's intrinsics directly, not through `simd-mappings.h`:
// the NEON arm four lanes at a time, the `__AVX2__ && __FMA__` arm eight, the
// `__AVX512F__ && __AVX512DQ__` arm sixteen. The NEON and AVX2 arithmetic is
// the same lane by lane -- the AVX2 `ggml_v_expf` is the NEON one with every
// fused term in the same place -- so it is written once here over `VF`, and
// only the width and the horizontal sum differ. The AVX-512 `ggml_v_expf` is
// a different routine (`vExpf512`); its callers are the same loops.

/// Which of `vec.cpp`'s x86 arms this build takes, refusing an x86 level
/// whose arm is not ported.
const X86Arm = enum { none, avx2, avx512 };
const x86_arm: X86Arm = blk: {
    if (builtin.cpu.arch != .x86_64) break :blk .none;
    const has = std.Target.x86.featureSetHas;
    if (has(builtin.cpu.features, .avx512bf16))
        @compileError("vec.cpp's __AVX512BF16__ arm is not ported");
    if (has(builtin.cpu.features, .avx512f)) {
        if (!has(builtin.cpu.features, .avx512dq))
            @compileError("vec.cpp's AVX-512 arms are ported only with AVX512DQ (x86-64-v4)");
        break :blk .avx512;
    }
    if (!has(builtin.cpu.features, .avx2) or !has(builtin.cpu.features, .fma))
        @compileError("vec.cpp's x86 arms are ported only for AVX2 with FMA (x86-64-v3)");
    break :blk .avx2;
};

/// Lanes per step of the SiLU and softmax arms: `float32x4_t`, `__m256` or
/// `__m512`.
const lanes = switch (builtin.cpu.arch) {
    .aarch64 => 4,
    .x86_64 => switch (x86_arm) {
        .avx2 => 8,
        .avx512 => 16,
        .none => unreachable,
    },
    else => @compileError("vec.cpp has no vector arm ported for " ++ @tagName(builtin.cpu.arch)),
};

/// `float32x4_t`, `__m256` or `__m512`.
const VF = @Vector(lanes, f32);
const VU = @Vector(lanes, u32);

/// Ports `vdupq_n_f32` and `_mm256_set1_ps`.
inline fn dup_n_f32(x: f32) VF {
    return @splat(x);
}

/// Ports `vaddq_f32` and `_mm256_add_ps`.
inline fn add_f32(a: VF, b: VF) VF {
    return a + b;
}

/// Ports `vsubq_f32` and `_mm256_sub_ps`.
inline fn sub_f32(a: VF, b: VF) VF {
    return a - b;
}

/// Ports `vmulq_f32` and `_mm256_mul_ps`.
inline fn mul_f32(a: VF, b: VF) VF {
    return a * b;
}

/// Ports `vdivq_f32` and `_mm256_div_ps`.
inline fn div_f32(a: VF, b: VF) VF {
    return a / b;
}

/// Ports `vfmaq_f32(a, b, cc)` and `_mm256_fmadd_ps(b, cc, a)`: `a + b * cc`,
/// **fused**.
inline fn fma_f32(a: VF, b: VF, cc: VF) VF {
    return @mulAdd(VF, b, cc, a);
}

/// Ports `vfmsq_f32(a, b, cc)` and `_mm256_fnmadd_ps(b, cc, a)`: `a - b * cc`,
/// **fused**.
///
/// The negated counterpart of `vfmaq_f32`, and fused the same way -- clang
/// emits `fmls` and `vfnmadd`, one rounding. `@mulAdd` with a negated
/// multiplier is the same operation.
inline fn fms_f32(a: VF, b: VF, cc: VF) VF {
    return @mulAdd(VF, -b, cc, a);
}

/// One step's horizontal sum, as each arm writes it.
///
/// NEON: `vaddvq_f32`, pairwise, `(v0+v1)+(v2+v3)`. AVX2: fold the upper
/// 128 bits onto the lower (`h = hi + lo`), then `_mm_movehl_ps` and
/// `_mm_movehdup_ps` give `(h0+h2)+(h1+h3)` -- a different pairing, which
/// is visible in the last bit. AVX-512: `_mm512_reduce_add_ps`, in
/// `simd.reduceAdd512`'s tree.
inline fn hsum(v: VF) f32 {
    if (comptime lanes == 4) return neon.addvq_f32(v);
    if (comptime lanes == 16) return simd.reduceAdd512(v);
    const lo: @Vector(4, f32) = .{ v[0], v[1], v[2], v[3] };
    const hi: @Vector(4, f32) = .{ v[4], v[5], v[6], v[7] };
    const h = hi + lo;
    return (h[0] + h[2]) + (h[1] + h[3]);
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

/// Ports `ggml_vec_dot_f32` (vec.cpp:11 @c1d0e7a00), the `GGML_SIMD` arm
/// without SVE or RVV.
///
/// `GGML_F32_ARR` accumulators over steps of `GGML_F32_STEP`, reduced by
/// `GGML_F32_VEC_REDUCE`, with a scalar `f32` remainder that the reference
/// compiler vectorizes (`simd.strictDot`). Every part of that shape is
/// load-bearing: see the table in the file header.
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
    const np = n & ~@as(c_int, F32.step - 1);

    var sum: [F32.arr]F32.Vec = @splat(F32.zero);
    var ax: [F32.arr]F32.Vec = undefined;
    var ay: [F32.arr]F32.Vec = undefined;

    var i: c_int = 0;
    while (i < np) : (i += F32.step) {
        for (0..F32.arr) |j| {
            const off: usize = @as(usize, @intCast(i)) + j * F32.epr;
            ax[j] = F32.load(x + off);
            ay[j] = F32.load(y + off);
            sum[j] = F32.fma(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    sumf = F32.reduce(&sum);

    // leftovers
    sumf = simd.strictDot(x, y, @intCast(np), @intCast(n), sumf);

    s.* = sumf;
}

/// Ports `ggml_vec_dot_bf16` (vec.cpp:139 @c1d0e7a00), its `__AVX512F__`
/// and `__AVX2__` arms and the scalar loop.
///
/// On ARM **no SIMD arm compiles**, so this is the plain scalar loop the C
/// falls back to, accumulating in `ggml_float` -- a `double`, not an `f32`.
/// On x86-64-v3 the `__AVX2__` arm runs first: four `__m256` accumulators
/// over steps of 32, each `c = x*y + c` as a separate `_mm256_mul_ps` and
/// `_mm256_add_ps` -- **unfused**, read off the reference's `vmulps` and
/// `vaddps` -- reduced `(c1+c3)+(c2+c4)`, then `hsum`, into the same
/// `double` the scalar tail continues. On x86-64-v4 the `__AVX512F__` arm
/// runs instead: two `__m512` accumulators over steps of 32, unfused the same
/// way, each reduced by `_mm512_reduce_add_ps` and added to the `double` in
/// turn.
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
    const nn: usize = @intCast(n);
    if (comptime x86_arm == .avx512) {
        const V16 = @Vector(16, f32);
        // `LOAD`: widen sixteen `bf16` by a 16-bit shift, exactly.
        const load = struct {
            inline fn f(p: [*]c.ggml_bf16_t) V16 {
                const p16: [*]const u16 = @ptrCast(p);
                const h: @Vector(16, u16) = p16[0..16].*;
                return @bitCast(@as(@Vector(16, u32), h) << @splat(16));
            }
        }.f;
        var c1: V16 = @splat(0);
        var c2: V16 = @splat(0);
        while (i + 32 <= nn) : (i += 32) {
            c1 = load(x + i) * load(y + i) + c1;
            c2 = load(x + i + 16) * load(y + i + 16) + c2;
        }
        sumf += @as(GgmlFloat, simd.reduceAdd512(c1));
        sumf += @as(GgmlFloat, simd.reduceAdd512(c2));
    }
    if (comptime x86_arm == .avx2) {
        const V8 = @Vector(8, f32);
        // `LOAD`: widen eight `bf16` by a 16-bit shift, exactly.
        const load = struct {
            inline fn f(p: [*]c.ggml_bf16_t) V8 {
                const p16: [*]const u16 = @ptrCast(p);
                const h: @Vector(8, u16) = p16[0..8].*;
                return @bitCast(@as(@Vector(8, u32), h) << @splat(16));
            }
        }.f;
        var c1: V8 = @splat(0);
        var c2: V8 = @splat(0);
        var c3: V8 = @splat(0);
        var c4: V8 = @splat(0);
        while (i + 32 <= nn) : (i += 32) {
            c1 = load(x + i) * load(y + i) + c1;
            c2 = load(x + i + 8) * load(y + i + 8) + c2;
            c3 = load(x + i + 16) * load(y + i + 16) + c3;
            c4 = load(x + i + 24) * load(y + i + 24) + c4;
        }
        c1 = (c1 + c3) + (c2 + c4);
        sumf += @as(GgmlFloat, hsum(c1));
    }
    while (i < nn) : (i += 1) {
        sumf += @as(GgmlFloat, impl.bf16ToFp32(x[i].bits) * impl.bf16ToFp32(y[i].bits));
    }

    s.* = @floatCast(sumf);
}

/// Ports `ggml_vec_dot_f16` (vec.cpp:264 @c1d0e7a00), the `GGML_SIMD` arm
/// without SVE or RVV.
///
/// `GGML_F16_ARR` accumulators over steps of `GGML_F16_STEP`, reduced by
/// `GGML_F16_VEC_REDUCE`. With `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` --
/// every Apple core -- that is four `f16x8` over 32 and the body really does
/// accumulate in **half precision**; without it -- `generic`, Zig's
/// `aarch64-linux` baseline -- four `f32x4` over widened halves, step 16,
/// accumulating in `f32`. Either way the remainder accumulates in `double`,
/// unlike the `f32` kernel's `f32` tail.
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
    const np = n & ~@as(c_int, F16.step - 1);

    var sum: [F16.arr]F16.Vec = @splat(F16.zero);
    var ax: [F16.arr]F16.Vec = undefined;
    var ay: [F16.arr]F16.Vec = undefined;

    var i: c_int = 0;
    while (i < np) : (i += F16.step) {
        for (0..F16.arr) |j| {
            const off: usize = @as(usize, @intCast(i)) + j * F16.epr;
            ax[j] = F16.load(x + off);
            ay[j] = F16.load(y + off);
            sum[j] = F16.fma(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    sumf = F16.reduce(&sum);

    // leftovers
    var k: usize = @intCast(np);
    while (k < @as(usize, @intCast(n))) : (k += 1) {
        sumf += @as(GgmlFloat, impl.fp16ToFp32(x[k]) * impl.fp16ToFp32(y[k]));
    }

    s.* = @floatCast(sumf);
}

// -----------------------------------------------------------------------------
// The vectorized exponential
//
// Ports `ggml_v_expf`, `ggml_v_expf` (vec.h:1133, 1215 @c1d0e7a00), the `__ARM_NEON &&
// __aarch64__` arm and the `__AVX2__ && __FMA__` one. `inline static` in the
// C, so it has no symbol and is private here.
//
// This is ARM's optimised `expf` routine, not a generic approximation: the C's
// own comment upstream records a maximum error of 1.45358 + 0.5 ulp. Every
// constant is a hex float and every fused site is an explicit `vfmaq_f32` /
// `vfmsq_f32` or `_mm256_fmadd_ps` / `_mm256_fnmadd_ps`, so the port names the
// same fusions with `@mulAdd` and the result is bit-identical rather than
// merely close. That is the condition `CLAUDE.md` sets for naming a fusion,
// and `vec_golden.zig` is the something that can say it is wrong.
//
// The two arms were read side by side: the AVX2 one computes `z`, `b`, the
// polynomial `j` and both slow-path selects with the same operands in the
// same order, so one body over `VF` is both. Only the fast-path test is
// spelled differently -- `vpaddd_u64` of the mask against `_mm256_movemask_ps`
// -- and both ask whether any lane is set.

// `.x86_64_sysv` for the reason `cpu/convert.zig` gives. What
// `_mm512_scalef_ps` compiles to: unmasked, current rounding mode (4).
extern fn @"llvm.x86.avx512.mask.scalef.ps.512"(@Vector(16, f32), @Vector(16, f32), @Vector(16, f32), u16, i32) callconv(.{ .x86_64_sysv = .{} }) @Vector(16, f32);

/// Ports `ggml_v_expf` (vec.h:1172 @c1d0e7a00), the `__AVX512F__ &&
/// __AVX512DQ__` arm.
///
/// Not the NEON and AVX2 routine at sixteen lanes. The polynomial's last
/// term is `fma(0x1.ffffecp-1, b, 1)` rather than a product, the scale is
/// applied by `_mm512_scalef_ps(j, n)` rather than by building `2^n` in the
/// exponent bits, and the slow path only distinguishes `|n| > 192`, giving
/// zero or infinity. Every fusion is an explicit `_mm512_fmadd_ps` or
/// `_mm512_fnmadd_ps`; read against the reference's IR at x86-64-v4.
inline fn vExpf512(x: VF) VF {
    const r = dup_n_f32(0x1.8p23);
    const z = fma_f32(r, x, dup_n_f32(0x1.715476p+0));
    const nn = sub_f32(z, r);
    const b = fms_f32(fms_f32(x, nn, dup_n_f32(0x1.62e4p-1)), nn, dup_n_f32(0x1.7f7d1cp-20));
    // `_mm512_cmp_ps_mask(_mm512_abs_ps(n), 192, _CMP_GT_OQ)`.
    const d = @abs(nn) > dup_n_f32(192);
    const u = mul_f32(b, b);
    const j = fma_f32(
        fma_f32(dup_n_f32(1.0), dup_n_f32(0x1.ffffecp-1), b),
        fma_f32(
            fma_f32(dup_n_f32(0x1.fffdb6p-2), dup_n_f32(0x1.555e66p-3), b),
            fma_f32(dup_n_f32(0x1.573e2ep-5), dup_n_f32(0x1.0e4020p-7), b),
            u,
        ),
        u,
    );
    const res = @"llvm.x86.avx512.mask.scalef.ps.512"(j, nn, @splat(0), 0xffff, 4);
    // `_mm512_kortestz(d, d)`: no lane set.
    if (!@reduce(.Or, d)) return res;
    const alt = @select(f32, nn <= dup_n_f32(0), dup_n_f32(0), dup_n_f32(std.math.inf(f32)));
    return @select(f32, d, alt, res);
}

/// Ports `ggml_v_expf`, `ggml_v_expf` (vec.h:1133, 1215 @c1d0e7a00), the
/// NEON and AVX2 arms; the AVX-512 arm is `vExpf512`.
inline fn vExpf(x: VF) VF {
    if (comptime lanes == 16) return vExpf512(x);
    const r = dup_n_f32(0x1.8p23);
    const z = fma_f32(r, x, dup_n_f32(0x1.715476p+0));
    const nn = sub_f32(z, r);
    const b = fms_f32(fms_f32(x, nn, dup_n_f32(0x1.62e4p-1)), nn, dup_n_f32(0x1.7f7d1cp-20));

    const e: VU = @as(VU, @bitCast(z)) << @splat(23);
    const k: VF = @bitCast(e +% @as(VU, @bitCast(dup_n_f32(1))));

    // `vcagtq_f32` / `_mm256_cmp_ps(|n|, 126, _CMP_GT_OQ)`: |n| > 126,
    // lane-wise, as an all-ones mask.
    const cmask = @abs(nn) > dup_n_f32(126);
    const cc: VU = @select(u32, cmask, @as(VU, @splat(0xFFFFFFFF)), @as(VU, @splat(0)));

    const u = mul_f32(b, b);
    const j = fma_f32(
        mul_f32(dup_n_f32(0x1.ffffecp-1), b),
        fma_f32(
            fma_f32(dup_n_f32(0x1.fffdb6p-2), dup_n_f32(0x1.555e66p-3), b),
            fma_f32(dup_n_f32(0x1.573e2ep-5), dup_n_f32(0x1.0e4020p-7), b),
            u,
        ),
        u,
    );

    // The fast path, when no lane is set: `vpaddd_u64(vreinterpretq_u64_u32(c))`
    // is zero, or `_mm256_movemask_ps(c)` is.
    const any_set = if (comptime lanes == 4) blk: {
        const cc64: @Vector(2, u64) = @bitCast(cc);
        break :blk cc64[0] +% cc64[1] != 0;
    } else @reduce(.Or, cmask);
    if (!any_set) {
        return fma_f32(k, j, k);
    }

    // `vclezq_f32(n)` / `_CMP_LE_OQ`: n <= 0, as a mask, then `& 0x82000000`.
    const lez = nn <= dup_n_f32(0);
    const d: VU = @select(u32, lez, @as(VU, @splat(0x82000000)), @as(VU, @splat(0)));

    const s1: VF = @bitCast(d +% @as(VU, @splat(0x7f000000)));
    const s2: VF = @bitCast(e -% d);

    const big = @abs(nn) > dup_n_f32(192);
    const inner = @select(
        f32,
        cmask,
        mul_f32(fma_f32(s2, s2, j), s1),
        fma_f32(k, k, j),
    );
    return @select(f32, big, mul_f32(s1, s1), inner);
}

/// Ports `ggml_v_silu`, `ggml_v_silu`, `ggml_v_silu` (vec.h:1157, 1200,
/// 1255 @c1d0e7a00), the NEON, AVX-512 and AVX2 arms: `x / (1 + exp(-x))`.
inline fn vSilu(x: VF) VF {
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

/// `lanes` floats from `p`.
inline fn loadVF(p: [*]const f32) VF {
    return p[0..lanes].*;
}

// -----------------------------------------------------------------------------
// The SiLU family

/// Ports `ggml_vec_silu_f32` (vec.cpp:380 @c1d0e7a00), the
/// `__ARM_NEON && __aarch64__` arm, the `__AVX512F__ && __AVX512DQ__` one
/// and the `__AVX2__ && __FMA__` one.
pub export fn ggml_vec_silu_f32(n: c_int, y: [*]f32, x: [*]const f32) callconv(.c) void {
    var i: c_int = 0;
    while (i + (lanes - 1) < n) : (i += lanes) {
        const off: usize = @intCast(i);
        const v = vSilu(loadVF(x + off));
        y[off..][0..lanes].* = v;
    }
    while (i < n) : (i += 1) {
        const off: usize = @intCast(i);
        y[off] = siluF32(x[off]);
    }
}

/// Ports `ggml_vec_swiglu_f32` (vec.cpp:417 @c1d0e7a00), the NEON, AVX2 and
/// AVX-512 arms.
pub export fn ggml_vec_swiglu_f32(n: c_int, y: [*]f32, x: [*]const f32, g: [*]const f32) callconv(.c) void {
    var i: c_int = 0;
    while (i + (lanes - 1) < n) : (i += lanes) {
        const off: usize = @intCast(i);
        const v = mul_f32(vSilu(loadVF(x + off)), loadVF(g + off));
        y[off..][0..lanes].* = v;
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
// one step at a time through `hsum` -- not a single reduction at the end. The
// order is the C's and is visible in the last bit.

/// Ports `ggml_vec_cvar_f32` (vec.cpp:455 @c1d0e7a00), the NEON, AVX2 and
/// AVX-512 arms.
///
/// Writes `x - mean` into `y` and returns the mean of its squares.
pub export fn ggml_vec_cvar_f32(n: c_int, y: [*]f32, x: [*]const f32, mean: f32) callconv(.c) GgmlFloat {
    var i: c_int = 0;
    var sum: GgmlFloat = 0;

    while (i + (lanes - 1) < n) : (i += lanes) {
        const off: usize = @intCast(i);
        var val = sub_f32(loadVF(x + off), dup_n_f32(mean));
        y[off..][0..lanes].* = val;
        val = mul_f32(val, val);
        sum += @as(GgmlFloat, hsum(val));
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

/// Ports `ggml_vec_soft_max_f32` (vec.cpp:531 @c1d0e7a00), the NEON, AVX2 and
/// AVX-512 arms.
///
/// Writes `exp(x - max)` into `y` and returns the sum. The vector path uses
/// `vExpf` and the tail uses libm's `expf`; they differ in the last bit and
/// the C uses each where it uses it.
pub export fn ggml_vec_soft_max_f32(n: c_int, y: [*]f32, x: [*]const f32, max: f32) callconv(.c) GgmlFloat {
    var i: c_int = 0;
    var sum: GgmlFloat = 0;

    while (i + (lanes - 1) < n) : (i += lanes) {
        const off: usize = @intCast(i);
        const val = vExpf(sub_f32(loadVF(x + off), dup_n_f32(max)));
        y[off..][0..lanes].* = val;
        sum += @as(GgmlFloat, hsum(val));
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
