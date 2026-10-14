//! The i8mm arms of the NEON interleaved gemms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`), the `__ARM_FEATURE_MATMUL_INT8` arms of five gemms. Each
//! declaration below names the C++ it replaces and the line it began at.
//!
//! # Where these sit
//!
//! Each of the five exports in `fallback.zig` is, in the C, an `#if
//! __ARM_FEATURE_MATMUL_INT8` body followed by a call to its `_generic`
//! form. A baseline (M1) build compiles only the call; a build for a CPU
//! with i8mm -- `-Dcpu=apple_m3` -- compiles only the body, and that body
//! is here. `fallback.zig` picks between them at comptime, so the baseline
//! never references this file.
//!
//! These are also exactly the shapes `ggml_repack_get_optimal_repack_type`
//! starts choosing once i8mm is on (`repack/dispatch.zig`): `q4_0` 4x8,
//! `q4_K`/`q5_K`/`q6_K` 8x8 and `q8_0` 4x8. Their gemvs already had NEON
//! bodies behind `__ARM_FEATURE_DOTPROD` and are ported beside the 8x4
//! shapes; only the gemms needed i8mm.
//!
//! # `smmla` computes a 2x2 tile, which is why every kernel reorders
//!
//! `vmmlaq_s32(acc, a, b)` treats `a` and `b` as two rows of eight bytes
//! each and accumulates `a * transpose(b)`: `[a0.b0, a0.b1, a1.b0, a1.b1]`.
//! With weights as `a` and activations as `b` -- the K-quant kernels --
//! that is two columns by two rows, column-major; the K-quant epilogue zips
//! each accumulator's halves and recombines them into whole output rows
//! (`reorderTile`). `q8_0` passes the activations as `a` instead, so its
//! tiles come out row-major and it needs only the recombine.
//!
//! # The float arithmetic is the C's
//!
//! All the block-scale work is integer (`vmlaq_s32` into `acc`), so the
//! only float per super-block is the epilogue: `d * q8_d` and `dmin * q8_d`
//! as plain `vmulq_f32`, then a fused `vmlsq_f32` for the mins and a fused
//! `vmlaq_f32` for the sum, in that order. `q8_0` closes on `vfmaq_f32` with
//! `vmulq_laneq_f32` -- the same shape. Each product the C writes as its own
//! statement is its own rounding here.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../blocks.zig");
const neon = @import("../../quants/arm/neon.zig");
const kscales = @import("kscales.zig");

const c = impl.c;
const f32x4 = neon.f32x4;
const i32x4 = neon.i32x4;
const i16x8 = neon.i16x8;
const i16x4 = neon.i16x4;
const i8x16 = neon.i8x16;
const u8x16 = neon.u8x16;

/// Whether these arms exist in this build: the C's
/// `__ARM_FEATURE_MATMUL_INT8`.
pub const enabled = neon.has_i8mm;

const QK_K = 256;

inline fn loadq(p: [*]const i8) i8x16 {
    return @bitCast(@as(@Vector(16, i8), p[0..16].*));
}
inline fn loadqu(p: [*]const u8) u8x16 {
    return p[0..16].*;
}
inline fn loadq16(p: [*]const i16) i16x8 {
    return p[0..8].*;
}

/// `vcombine_s32(vget_low_s32(a), vget_low_s32(b))`.
inline fn combineLow(a: i32x4, b: i32x4) i32x4 {
    return .{ a[0], a[1], b[0], b[1] };
}

/// `vcombine_s32(vget_high_s32(a), vget_high_s32(b))`.
inline fn combineHigh(a: i32x4, b: i32x4) i32x4 {
    return .{ a[2], a[3], b[2], b[3] };
}

/// Four `f16` deltas from an interleaved block, widened: `vcvt_f32_f16(
/// vld1_f16(p + 4 * j))`.
inline fn deltas(p: *const [8]c.ggml_half, j: usize) f32x4 {
    return neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(p)) + 4 * j));
}

// -----------------------------------------------------------------------------
// q4_0: hand-written aarch64

/// Ports `ggml_gemm_q4_0_4x8_q8_0` (arch/arm/repack.cpp:2307 @c1d0e7a00),
/// the `__ARM_FEATURE_MATMUL_INT8` arm at line 2327.
///
/// Like its 4x4 sibling in `q4_0_gemm.zig`, this is **hand-written
/// aarch64** in the C, and is reproduced instruction for instruction with
/// the same two mechanical changes: `%x[name]` becomes `%[name]` with the
/// 32-bit operands widened to `usize`, and the clobbers become a typed
/// struct. The template was converted by script, and the script checked by
/// converting the 4x4 kernel and diffing it against `q4_0_gemm.zig`:
/// identical, all 445 lines. The C writes `smmla` as `.inst` encodings, so
/// the assembler needs no i8mm of its own.
///
/// Parameters:
/// - `n`: the reduction length; `nb = n / 32` blocks.
/// - `s`, `bs`: the result and its row stride in elements.
/// - `vx`, `vy`: the interleaved weights and the quantized activations.
/// - `nr`, `nc`: rows and columns of the product.
pub fn gemmQ4_0_4x8(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    const nb: usize = @intCast(@divTrunc(n, 32));

    const b_ptr: [*]const u8 = @ptrCast(vx);
    var a_ptr: [*]const u8 = @ptrCast(vy);
    var res_ptr: [*]f32 = s;
    const res_stride: usize = bs * @sizeOf(f32);
    const nr_u: usize = @intCast(nr);
    const nc_u: usize = @intCast(nc);

    asm volatile (
        \\mov x10, %[nr]
        \\mov x9, #0x88
        \\cmp x10, #0x10
        \\mul x9, %[nb], x9
        \\blt 4f
        \\1:  // Row loop
        \\add x28, %[b_ptr], #0x8
        \\mov x27, %[nc]
        \\add x26, %[res_ptr], %[res_stride], LSL #4
        \\2:  // Column loop
        \\add x25, %[a_ptr], #0x8
        \\movi v2.16b, #0x0
        \\movi v10.16b, #0x0
        \\mov x24, %[nb]
        \\add x23, x25, x9
        \\movi v12.16b, #0x0
        \\movi v28.16b, #0x0
        \\add x22, x23, x9
        \\movi v11.16b, #0x0
        \\movi v13.16b, #0x0
        \\add x21, x22, x9
        \\movi v22.16b, #0x0
        \\movi v23.16b, #0x0
        \\movi v25.16b, #0x0
        \\movi v5.16b, #0x0
        \\movi v7.16b, #0x0
        \\movi v4.16b, #0x0
        \\movi v6.16b, #0x0
        \\movi v30.16b, #0x0
        \\movi v24.16b, #0x0
        \\movi v14.16b, #0x0
        \\3:  // Block loop
        \\ldr q21, [x28, #0x0]
        \\ldr q16, [x28, #0x10]
        \\movi v1.16b, #0x4
        \\movi v19.4s, #0x0
        \\ldr q27, [x25, #0x0]
        \\ldr q15, [x25, #0x10]
        \\movi v26.4s, #0x0
        \\movi v18.4s, #0x0
        \\ldr q29, [x28, #0x20]
        \\ldr q3, [x28, #0x30]
        \\movi v17.4s, #0x0
        \\movi v0.16b, #0xf0
        \\ldr d20, [x25, #-0x8]
        \\ldr d9, [x23, #-0x8]
        \\sshl v8.16b, v21.16b, v1.16b
        \\sshl v31.16b, v16.16b, v1.16b
        \\and v21.16b, v21.16b, v0.16b
        \\and v16.16b, v16.16b, v0.16b
        \\sub x20, x28, #0x8
        \\subs x24, x24, #0x1
        \\add x28, x28, #0x48
        \\.inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b
        \\.inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b
        \\ldr q27, [x25, #0x20]
        \\.inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b
        \\.inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b
        \\sshl v15.16b, v29.16b, v1.16b
        \\sshl v1.16b, v3.16b, v1.16b
        \\and v29.16b, v29.16b, v0.16b
        \\and v3.16b, v3.16b, v0.16b
        \\ldr q0, [x25, #0x30]
        \\fcvtl v20.4s, v20.4h
        \\.inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b
        \\fcvtl v9.4s, v9.4h
        \\.inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b
        \\ldr q27, [x25, #0x40]
        \\.inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b
        \\.inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b
        \\ldr q0, [x25, #0x50]
        \\.inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b
        \\.inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b
        \\ldr q27, [x25, #0x60]
        \\.inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b
        \\.inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b
        \\ldr q0, [x25, #0x70]
        \\add x25, x25, #0x88
        \\.inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b
        \\.inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b
        \\ldr d27, [x20, #0x0]
        \\.inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b
        \\.inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b
        \\fcvtl v27.4s, v27.4h
        \\uzp1 v0.2d, v19.2d, v26.2d
        \\uzp2 v26.2d, v19.2d, v26.2d
        \\fmul v19.4s, v27.4s, v20.s[0]
        \\scvtf v0.4s, v0.4s, #0x4
        \\scvtf v26.4s, v26.4s, #0x4
        \\fmla v2.4s, v0.4s, v19.4s
        \\ldr q19, [x23, #0x0]
        \\uzp1 v0.2d, v18.2d, v17.2d
        \\uzp2 v18.2d, v18.2d, v17.2d
        \\fmul v17.4s, v27.4s, v20.s[1]
        \\scvtf v0.4s, v0.4s, #0x4
        \\scvtf v18.4s, v18.4s, #0x4
        \\fmla v10.4s, v26.4s, v17.4s
        \\ldr q17, [x23, #0x10]
        \\fmul v26.4s, v27.4s, v20.s[2]
        \\fmul v20.4s, v27.4s, v20.s[3]
        \\fmla v12.4s, v0.4s, v26.4s
        \\ldr d0, [x22, #-0x8]
        \\ldr d26, [x21, #-0x8]
        \\fcvtl v0.4s, v0.4h
        \\fmla v28.4s, v18.4s, v20.4s
        \\movi v20.4s, #0x0
        \\movi v18.4s, #0x0
        \\.inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b
        \\.inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b
        \\ldr q19, [x23, #0x20]
        \\fcvtl v26.4s, v26.4h
        \\.inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b
        \\.inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b
        \\ldr q19, [x23, #0x40]
        \\.inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b
        \\.inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b
        \\ldr q19, [x23, #0x60]
        \\.inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b
        \\.inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b
        \\uzp1 v19.2d, v20.2d, v18.2d
        \\scvtf v19.4s, v19.4s, #0x4
        \\uzp2 v20.2d, v20.2d, v18.2d
        \\fmul v18.4s, v27.4s, v9.s[0]
        \\scvtf v20.4s, v20.4s, #0x4
        \\fmla v11.4s, v19.4s, v18.4s
        \\ldr q18, [x22, #0x0]
        \\fmul v19.4s, v27.4s, v9.s[1]
        \\fmla v13.4s, v20.4s, v19.4s
        \\movi v19.4s, #0x0
        \\movi v20.4s, #0x0
        \\.inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b
        \\.inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b
        \\ldr q17, [x23, #0x30]
        \\.inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b
        \\.inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b
        \\ldr q17, [x23, #0x50]
        \\.inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b
        \\.inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b
        \\ldr q17, [x23, #0x70]
        \\add x23, x23, #0x88
        \\.inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b
        \\.inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b
        \\uzp1 v17.2d, v19.2d, v20.2d
        \\scvtf v17.4s, v17.4s, #0x4
        \\uzp2 v20.2d, v19.2d, v20.2d
        \\fmul v19.4s, v27.4s, v9.s[2]
        \\fmul v9.4s, v27.4s, v9.s[3]
        \\scvtf v20.4s, v20.4s, #0x4
        \\fmla v22.4s, v17.4s, v19.4s
        \\ldr q17, [x22, #0x10]
        \\movi v19.4s, #0x0
        \\.inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b
        \\fmla v23.4s, v20.4s, v9.4s
        \\movi v20.4s, #0x0
        \\movi v9.4s, #0x0
        \\.inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b
        \\ldr q18, [x22, #0x20]
        \\.inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b
        \\.inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b
        \\.inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b
        \\ldr q18, [x22, #0x40]
        \\.inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b
        \\.inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b
        \\ldr q18, [x22, #0x60]
        \\.inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b
        \\.inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b
        \\movi v18.4s, #0x0
        \\.inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b
        \\ldr q17, [x22, #0x30]
        \\.inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b
        \\.inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b
        \\ldr q17, [x22, #0x50]
        \\.inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b
        \\.inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b
        \\ldr q17, [x22, #0x70]
        \\add x22, x22, #0x88
        \\.inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b
        \\.inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b
        \\uzp1 v17.2d, v19.2d, v20.2d
        \\uzp2 v20.2d, v19.2d, v20.2d
        \\fmul v19.4s, v27.4s, v0.s[0]
        \\scvtf v17.4s, v17.4s, #0x4
        \\scvtf v20.4s, v20.4s, #0x4
        \\fmla v25.4s, v17.4s, v19.4s
        \\ldr q19, [x21, #0x0]
        \\fmul v17.4s, v27.4s, v0.s[1]
        \\fmla v5.4s, v20.4s, v17.4s
        \\ldr q17, [x21, #0x10]
        \\uzp1 v20.2d, v9.2d, v18.2d
        \\uzp2 v9.2d, v9.2d, v18.2d
        \\fmul v18.4s, v27.4s, v0.s[2]
        \\fmul v0.4s, v27.4s, v0.s[3]
        \\scvtf v20.4s, v20.4s, #0x4
        \\scvtf v9.4s, v9.4s, #0x4
        \\fmla v7.4s, v20.4s, v18.4s
        \\movi v20.4s, #0x0
        \\movi v18.4s, #0x0
        \\.inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b
        \\.inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b
        \\ldr q19, [x21, #0x20]
        \\fmla v4.4s, v9.4s, v0.4s
        \\movi v9.4s, #0x0
        \\movi v0.4s, #0x0
        \\.inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b
        \\fmul v8.4s, v27.4s, v26.s[0]
        \\.inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b
        \\ldr q17, [x21, #0x30]
        \\.inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b
        \\fmul v31.4s, v27.4s, v26.s[1]
        \\.inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b
        \\ldr q19, [x21, #0x40]
        \\.inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b
        \\fmul v15.4s, v27.4s, v26.s[2]
        \\fmul v27.4s, v27.4s, v26.s[3]
        \\.inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b
        \\ldr q1, [x21, #0x50]
        \\.inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b
        \\.inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b
        \\ldr q26, [x21, #0x60]
        \\.inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b
        \\.inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b
        \\ldr q21, [x21, #0x70]
        \\add x21, x21, #0x88
        \\.inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b
        \\.inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b
        \\.inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b
        \\.inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b
        \\uzp1 v29.2d, v20.2d, v18.2d
        \\uzp2 v21.2d, v20.2d, v18.2d
        \\scvtf v29.4s, v29.4s, #0x4
        \\uzp1 v18.2d, v9.2d, v0.2d
        \\uzp2 v16.2d, v9.2d, v0.2d
        \\scvtf v21.4s, v21.4s, #0x4
        \\fmla v6.4s, v29.4s, v8.4s
        \\scvtf v18.4s, v18.4s, #0x4
        \\scvtf v16.4s, v16.4s, #0x4
        \\fmla v30.4s, v21.4s, v31.4s
        \\fmla v24.4s, v18.4s, v15.4s
        \\fmla v14.4s, v16.4s, v27.4s
        \\bgt 3b
        \\mov x20, %[res_ptr]
        \\subs x27, x27, #0x4
        \\add %[res_ptr], %[res_ptr], #0x10
        \\str q2, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q10, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q12, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q28, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q11, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q13, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q22, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q23, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q25, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q5, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q7, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q4, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q6, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q30, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q24, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q14, [x20, #0x0]
        \\bne 2b
        \\mov x20, #0x4
        \\sub x10, x10, #0x10
        \\cmp x10, #0x10
        \\mov %[res_ptr], x26
        \\madd %[a_ptr], x20, x9, %[a_ptr]
        \\bge 1b
        \\4:  // Row loop skip
        \\cbz x10, 9f
        \\5:  // Row tail: Row loop
        \\add x24, %[b_ptr], #0x8
        \\mov x23, %[nc]
        \\add x22, %[res_ptr], %[res_stride], LSL #2
        \\6:  // Row tail: Column loop
        \\movi v2.16b, #0x0
        \\movi v10.16b, #0x0
        \\add x25, %[a_ptr], #0x8
        \\mov x21, %[nb]
        \\movi v12.16b, #0x0
        \\movi v28.16b, #0x0
        \\7:  // Row tail: Block loop
        \\ldr q6, [x24, #0x0]
        \\ldr q5, [x24, #0x10]
        \\movi v17.16b, #0x4
        \\movi v8.4s, #0x0
        \\ldr q4, [x25, #0x0]
        \\ldr q13, [x25, #0x10]
        \\movi v27.4s, #0x0
        \\movi v0.4s, #0x0
        \\ldr q31, [x24, #0x20]
        \\ldr q14, [x24, #0x30]
        \\movi v29.4s, #0x0
        \\movi v22.16b, #0xf0
        \\ldr q11, [x25, #0x20]
        \\ldr q23, [x25, #0x30]
        \\sshl v21.16b, v6.16b, v17.16b
        \\sshl v16.16b, v5.16b, v17.16b
        \\ldr q20, [x25, #0x40]
        \\ldr q26, [x25, #0x50]
        \\and v6.16b, v6.16b, v22.16b
        \\and v5.16b, v5.16b, v22.16b
        \\ldr q25, [x25, #0x60]
        \\ldr q3, [x25, #0x70]
        \\sshl v19.16b, v31.16b, v17.16b
        \\sshl v18.16b, v14.16b, v17.16b
        \\ldr d17, [x25, #-0x8]
        \\.inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b
        \\.inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b
        \\and v31.16b, v31.16b, v22.16b
        \\.inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b
        \\.inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b
        \\and v14.16b, v14.16b, v22.16b
        \\sub x20, x24, #0x8
        \\ldr d16, [x20, #0x0]
        \\subs x21, x21, #0x1
        \\add x25, x25, #0x88
        \\fcvtl v17.4s, v17.4h
        \\add x24, x24, #0x48
        \\.inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b
        \\.inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b
        \\.inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b
        \\.inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b
        \\fcvtl v16.4s, v16.4h
        \\.inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b
        \\.inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b
        \\fmul v23.4s, v16.4s, v17.s[0]
        \\fmul v21.4s, v16.4s, v17.s[1]
        \\fmul v1.4s, v16.4s, v17.s[2]
        \\fmul v20.4s, v16.4s, v17.s[3]
        \\.inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b
        \\.inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b
        \\.inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b
        \\.inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b
        \\.inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b
        \\.inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b
        \\uzp1 v19.2d, v8.2d, v27.2d
        \\uzp2 v18.2d, v8.2d, v27.2d
        \\scvtf v19.4s, v19.4s, #0x4
        \\uzp1 v17.2d, v0.2d, v29.2d
        \\uzp2 v16.2d, v0.2d, v29.2d
        \\scvtf v18.4s, v18.4s, #0x4
        \\fmla v2.4s, v19.4s, v23.4s
        \\scvtf v17.4s, v17.4s, #0x4
        \\scvtf v16.4s, v16.4s, #0x4
        \\fmla v10.4s, v18.4s, v21.4s
        \\fmla v12.4s, v17.4s, v1.4s
        \\fmla v28.4s, v16.4s, v20.4s
        \\bgt 7b
        \\mov x20, %[res_ptr]
        \\cmp x10, #0x1
        \\str q2, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\ble 8f
        \\cmp x10, #0x2
        \\str q10, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\ble 8f
        \\cmp x10, #0x3
        \\str q12, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\ble 8f
        \\str q28, [x20, #0x0]
        \\8:  // Row tail: Accumulator store skip
        \\subs x23, x23, #0x4
        \\add %[res_ptr], %[res_ptr], #0x10
        \\bne 6b
        \\subs x10, x10, #0x4
        \\add %[a_ptr], %[a_ptr], x9
        \\mov %[res_ptr], x22
        \\bgt 5b
        \\9:  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr),
          [res_ptr] "+&r" (res_ptr),
        : [b_ptr] "r" (b_ptr),
          [nr] "r" (nr_u),
          [nb] "r" (nb),
          [res_stride] "r" (res_stride),
          [nc] "r" (nc_u),
        : .{
          .nzcv = true,
          .memory = true,
          .z0 = true,
          .z1 = true,
          .z2 = true,
          .z3 = true,
          .z4 = true,
          .z5 = true,
          .z6 = true,
          .z7 = true,
          .z8 = true,
          .z9 = true,
          .z10 = true,
          .z11 = true,
          .z12 = true,
          .z13 = true,
          .z14 = true,
          .z15 = true,
          .z16 = true,
          .z17 = true,
          .z18 = true,
          .z19 = true,
          .z20 = true,
          .z21 = true,
          .z22 = true,
          .z23 = true,
          .z24 = true,
          .z25 = true,
          .z26 = true,
          .z27 = true,
          .z28 = true,
          .z29 = true,
          .z30 = true,
          .z31 = true,
          .x9 = true,
          .x10 = true,
          .x20 = true,
          .x21 = true,
          .x22 = true,
          .x23 = true,
          .x24 = true,
          .x25 = true,
          .x26 = true,
          .x27 = true,
          .x28 = true,
        });
}

// -----------------------------------------------------------------------------
// The K-quant 8x8 gemms: shared pieces

/// Rows of the `q8_Kx4` activation block, and the number of `f32x4`
/// accumulators a tile needs: four rows by two groups of four columns.
const q8_k_blocklen = 4;
const tile_accs = 8;

/// Each row's sixteen `bsums` folded to eight, pairwise: `vpaddq_s16` and
/// the `vst1q_s16` into `bsums_arr`.
inline fn pairedBsums(q8: *const blocks.block_q8_Kx4) [4][8]i16 {
    var arr: [4][8]i16 = undefined;
    inline for (0..4) |q8_row| {
        const bp: [*]const i16 = @as([*]const i16, @ptrCast(&q8.bsums)) + 16 * q8_row;
        arr[q8_row] = neon.paddq_s16(loadq16(bp), loadq16(bp + 8));
    }
    return arr;
}

/// The "Reorder of i8mm output" block: each accumulator's 2x2 tiles,
/// column-major, into four whole rows of eight.
///
/// `vzip_s32(vget_low_s32(a), vget_high_s32(a))` recombined is
/// `[a0, a2, a1, a3]`; the recombines then take row 0 from the low halves
/// of `acc[0..4]` and row 1 from the high, and rows 2 and 3 likewise from
/// `acc[4..8]`.
inline fn reorderTile(acc_in: [tile_accs]i32x4) [tile_accs]i32x4 {
    var acc: [tile_accs]i32x4 = undefined;
    inline for (0..tile_accs) |i| acc[i] = @shuffle(i32, acc_in[i], undefined, [4]i32{ 0, 2, 1, 3 });
    return .{
        combineLow(acc[0], acc[1]),  combineLow(acc[2], acc[3]),
        combineHigh(acc[0], acc[1]), combineHigh(acc[2], acc[3]),
        combineLow(acc[4], acc[5]),  combineLow(acc[6], acc[7]),
        combineHigh(acc[4], acc[5]), combineHigh(acc[6], acc[7]),
    };
}

/// A block scale for one column pair: `vcombine_s32(vdup_n_s32(a),
/// vdup_n_s32(b))`, matching `smmla`'s `[c0r0, c0r1, c1r0, c1r1]`.
inline fn pairScale(a: i32, b: i32) i32x4 {
    return .{ a, a, b, b };
}

/// The tile's eight `f32x4` results to `s`, `vst1q_f32` each: row `i` of
/// four, columns `x * 8 + 4 * j`.
inline fn storeTile(s: [*]f32, bs: usize, y: usize, x: usize, acc_f32: *const [tile_accs]f32x4) void {
    for (0..q8_k_blocklen) |i| {
        const row = y * q8_k_blocklen + i;
        for (0..2) |j| {
            const col = x * 8 + j * 4;
            s[row * bs + col ..][0..4].* = acc_f32[2 * i + j];
        }
    }
}

/// The two formats `gemmQ45K8x8` serves.
const Q45 = enum { q4_K, q5_K };

/// Ports `ggml_gemm_q4_K_8x8_q8_K` (arch/arm/repack.cpp:3752 @c1d0e7a00),
/// the `__ARM_FEATURE_MATMUL_INT8` arm at line 4083, and
/// `ggml_gemm_q5_K_8x8_q8_K` (arch/arm/repack.cpp:4272 @c1d0e7a00), the arm
/// at line 4293.
///
/// One body for both because the C's two arms differ only in how a column
/// pair's sixteen bytes become eight-bit weights -- its own comment says
/// "This is the only part of the algorithm that differs with Q4_K". `q5_K`
/// interleaves its `smmla` calls with that decode where `q4_K` decodes
/// first, but every one lands in an integer accumulator in the same order
/// per accumulator, so the sums are identical.
fn gemmQ45K8x8(comptime kind: Q45, n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    const Block = switch (kind) {
        .q4_K => blocks.block_q4_Kx8,
        .q5_K => blocks.block_q5_Kx8,
    };
    const nb: usize = @intCast(@divTrunc(n, QK_K));
    const ncols_interleaved = 8;
    const col_pairs = ncols_interleaved / 2;
    const m4b: u8x16 = @splat(0x0f);
    const mone: u8x16 = @splat(1);
    const mtwo: u8x16 = @splat(2);

    const ny: usize = @intCast(@divTrunc(nr, q8_k_blocklen));
    const nx: usize = @intCast(@divTrunc(nc, ncols_interleaved));

    for (0..ny) |y| {
        const q8_ptr = @as([*]const blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + y * nb;

        for (0..nx) |x| {
            const q_ptr = @as([*]const Block, @ptrCast(@alignCast(vx))) + x * nb;

            var acc_f32: [tile_accs]f32x4 = @splat(@splat(0));

            for (0..nb) |b| {
                const bsums_arr = pairedBsums(&q8_ptr[b]);

                // Rows 01 in `acc[0..4]`, rows 23 in `acc[4..8]`; `bias_acc`
                // interleaved as `[2 * row + col_group]`.
                var acc: [tile_accs]i32x4 = @splat(@splat(0));
                var bias_acc: [tile_accs]i32x4 = @splat(@splat(0));

                // `q5_K` loads `qh` once per block and shifts it down two
                // bits after each sub-block.
                var qh: [col_pairs][4]u8x16 = undefined;
                if (kind == .q5_K) {
                    const qh_base: [*]const u8 = &q_ptr[b].qh;
                    for (0..col_pairs) |cp| {
                        inline for (0..4) |k| qh[cp][k] = loadqu(qh_base + 16 * cp + 64 * k);
                    }
                }

                for (0..QK_K / 64) |sb| {
                    // Scales for the low and high nibbles: 24 bytes per
                    // sub-block.
                    var sb_scales: [2][8]i8 = undefined;
                    var sb_mins: [2]i16x8 = undefined;
                    inline for (0..2) |i| {
                        const offset = sb * 24 + i * 12;
                        kscales.decodeQKx86BitScales(@as([*]const u8, &q_ptr[b].scales) + offset, &sb_mins[i], &sb_scales[i]);
                    }

                    // `q8_ptr[b].qs` interleaves rows 01 and 23, 16 bytes each.
                    const q8_base: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + sb * 256;
                    var q8s: [2][8]i8x16 = undefined;
                    inline for (0..8) |i| {
                        q8s[0][i] = loadq(q8_base + i * 32);
                        q8s[1][i] = loadq(q8_base + i * 32 + 16);
                    }

                    // Column pairs 01, 23, 45, 67.
                    for (0..col_pairs) |cp| {
                        const qs: [*]const u8 = @as([*]const u8, &q_ptr[b].qs) + sb * QK_K + 16 * cp;
                        const qs_cp = [4]u8x16{ loadqu(qs), loadqu(qs + 64), loadqu(qs + 128), loadqu(qs + 192) };

                        // `[0]` the low nibbles' sub-block, `[1]` the high's.
                        var w: [2][4]i8x16 = undefined;
                        inline for (0..4) |k| {
                            switch (kind) {
                                .q4_K => {
                                    w[0][k] = @bitCast(neon.@"and"(qs_cp[k], m4b));
                                    w[1][k] = @bitCast(neon.shrN(qs_cp[k], 4));
                                },
                                .q5_K => {
                                    // The high bits packed in as the fifth.
                                    const hbit_lo = neon.@"and"(qh[cp][k], mone);
                                    const hbit_hi = neon.shlN(neon.@"and"(qh[cp][k], mtwo), 3);
                                    qh[cp][k] = neon.shrN(qh[cp][k], 2);
                                    w[0][k] = @bitCast(neon.sli_n_u8(neon.@"and"(qs_cp[k], m4b), hbit_lo, 4));
                                    w[1][k] = @bitCast(neon.orr(neon.shrN(qs_cp[k], 4), hbit_hi));
                                },
                            }
                        }

                        // Row pair `rp` against each 32-weight half `blk`.
                        var sb_acc: [4]i32x4 = @splat(@splat(0));
                        inline for (0..2) |rp| {
                            inline for (0..2) |blk| {
                                var a = sb_acc[2 * rp + blk];
                                inline for (0..4) |qs_offset| {
                                    a = neon.mmlaq_s32(a, w[blk][qs_offset], q8s[rp][4 * blk + qs_offset]);
                                }
                                sb_acc[2 * rp + blk] = a;
                            }
                        }

                        // `scales[i]` is column `i`'s.
                        const so = cp * 2;
                        const block_scale_0 = pairScale(sb_scales[0][so], sb_scales[0][so + 1]);
                        const block_scale_1 = pairScale(sb_scales[1][so], sb_scales[1][so + 1]);

                        acc[cp] = neon.mlaq_s32(acc[cp], sb_acc[0], block_scale_0);
                        acc[cp + 4] = neon.mlaq_s32(acc[cp + 4], sb_acc[2], block_scale_0);
                        acc[cp] = neon.mlaq_s32(acc[cp], sb_acc[1], block_scale_1);
                        acc[cp + 4] = neon.mlaq_s32(acc[cp + 4], sb_acc[3], block_scale_1);
                    }

                    // The mins against the bsums both halves share.
                    inline for (0..4) |q8_row| {
                        const bsums_vec_lo: i16x4 = @splat(bsums_arr[sb][q8_row * 2]);
                        const bsums_vec_hi: i16x4 = @splat(bsums_arr[sb][q8_row * 2 + 1]);
                        bias_acc[2 * q8_row] = neon.mlal_s16(bias_acc[2 * q8_row], bsums_vec_lo, neon.low(sb_mins[0]));
                        bias_acc[2 * q8_row] = neon.mlal_s16(bias_acc[2 * q8_row], bsums_vec_hi, neon.low(sb_mins[1]));
                        bias_acc[2 * q8_row + 1] = neon.mlal_s16(bias_acc[2 * q8_row + 1], bsums_vec_lo, neon.high(sb_mins[0]));
                        bias_acc[2 * q8_row + 1] = neon.mlal_s16(bias_acc[2 * q8_row + 1], bsums_vec_hi, neon.high(sb_mins[1]));
                    }
                }

                const reorder_acc = reorderTile(acc);

                for (0..q8_k_blocklen) |i| {
                    for (0..2) |j| {
                        const q8_d: f32x4 = @splat(q8_ptr[b].d[i]);
                        const dmins = deltas(&q_ptr[b].dmin, j) * q8_d;
                        const scale = deltas(&q_ptr[b].d, j) * q8_d;

                        acc_f32[2 * i + j] = neon.mlsq_f32(acc_f32[2 * i + j], neon.cvt_f32_s32(bias_acc[2 * i + j]), dmins);
                        acc_f32[2 * i + j] = neon.mlaq_f32(acc_f32[2 * i + j], neon.cvt_f32_s32(reorder_acc[2 * i + j]), scale);
                    }
                }
            }

            // The reorder has already put the tile in memory order.
            storeTile(s, bs, y, x, &acc_f32);
        }
    }
}

/// `ggml_gemm_q4_K_8x8_q8_K`'s i8mm arm; see `gemmQ45K8x8`.
pub fn gemmQ4_K_8x8(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    gemmQ45K8x8(.q4_K, n, s, bs, vx, vy, nr, nc);
}

/// `ggml_gemm_q5_K_8x8_q8_K`'s i8mm arm; see `gemmQ45K8x8`.
pub fn gemmQ5_K_8x8(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    gemmQ45K8x8(.q5_K, n, s, bs, vx, vy, nr, nc);
}

/// Ports `ggml_gemm_q6_K_8x8_q8_K` (arch/arm/repack.cpp:4721 @c1d0e7a00),
/// the `__ARM_FEATURE_MATMUL_INT8` arm at line 4742.
///
/// Unlike `q4_K` and `q5_K`, the bias of 32 comes off each weight
/// (`vsubq_s8`) rather than out of the sum through `bsums`, and there are
/// no mins -- so the epilogue is the scaled sum alone. The scales are one
/// signed byte per sixteen weights, widened through `vmovl_s8` in the C,
/// which for a value read as `i32` is the identity.
pub fn gemmQ6_K_8x8(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    const nb: usize = @intCast(@divTrunc(n, QK_K));
    const ncols_interleaved = 8;
    const m4b: u8x16 = @splat(0x0f);
    const mask_lo: u8x16 = @splat(0x03);
    const mask_hi: u8x16 = @splat(0x30);
    const m32s: i8x16 = @splat(32);

    const ny: usize = @intCast(@divTrunc(nr, q8_k_blocklen));
    const nx: usize = @intCast(@divTrunc(nc, ncols_interleaved));

    for (0..ny) |y| {
        const q8_ptr = @as([*]const blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + y * nb;

        for (0..nx) |x| {
            const q6_ptr = @as([*]const blocks.block_q6_Kx8, @ptrCast(@alignCast(vx))) + x * nb;

            var acc_f32: [tile_accs]f32x4 = @splat(@splat(0));

            for (0..nb) |b| {
                // Rows 01 in `acc[0..4]`, rows 23 in `acc[4..8]`.
                var acc: [tile_accs]i32x4 = @splat(@splat(0));

                const q6_scales: *const [QK_K / 16 * 8]i8 = &q6_ptr[b].scales;

                // Two 128-weight halves per super-block.
                for (0..2) |half| {
                    const ql_base: [*]const u8 = @as([*]const u8, &q6_ptr[b].ql) + half * 512;
                    const qh_base: [*]const u8 = @as([*]const u8, &q6_ptr[b].qh) + half * 256;

                    // A sub-block shares one scale; q6_K's index advances by
                    // 64, so the low and high nibbles read separate q8
                    // regions.
                    for (0..QK_K / 64) |sb| {
                        const q8_qs: [*]const i8 = @ptrCast(&q8_ptr[b].qs);
                        const q8_base_l = q8_qs + half * 512 + sb * 64;
                        const q8_base_h = q8_qs + half * 512 + 256 + sb * 64;

                        var q8_l_01: [2]i8x16 = undefined;
                        var q8_l_23: [2]i8x16 = undefined;
                        var q8_h_01: [2]i8x16 = undefined;
                        var q8_h_23: [2]i8x16 = undefined;
                        inline for (0..2) |i| {
                            q8_l_01[i] = loadq(q8_base_l + i * 32);
                            q8_l_23[i] = loadq(q8_base_l + i * 32 + 16);
                            q8_h_01[i] = loadq(q8_base_h + i * 32);
                            q8_h_23[i] = loadq(q8_base_h + i * 32 + 16);
                        }

                        const ql_off_base = sb * QK_K / 2;
                        var q6_ql_0: [4]u8x16 = undefined;
                        var q6_ql_1: [4]u8x16 = undefined;
                        inline for (0..4) |k| {
                            q6_ql_0[k] = loadqu(ql_base + ql_off_base + 16 * k);
                            q6_ql_1[k] = loadqu(ql_base + ql_off_base + 64 + 16 * k);
                        }

                        // Wraps after 256 bytes: sub-blocks 2 and 3 reread
                        // 0 and 1's high bits, two places further down.
                        const qh_off_base = (sb * QK_K / 2) & 255;
                        var q6_qh_0: [4]u8x16 = undefined;
                        var q6_qh_1: [4]u8x16 = undefined;
                        inline for (0..4) |k| {
                            q6_qh_0[k] = loadqu(qh_base + qh_off_base + 16 * k);
                            q6_qh_1[k] = loadqu(qh_base + qh_off_base + 64 + 16 * k);
                        }
                        if (sb > 1) {
                            inline for (0..4) |k| {
                                q6_qh_0[k] = neon.shrN(q6_qh_0[k], 2);
                                q6_qh_1[k] = neon.shrN(q6_qh_1[k], 2);
                            }
                        }

                        // Column pairs 01, 23, 45, 67.
                        for (0..ncols_interleaved / 2) |cp| {
                            const l0_bits = q6_ql_0[cp];
                            const l1_bits = q6_ql_1[cp];
                            const h0_bits = q6_qh_0[cp];
                            const h1_bits = q6_qh_1[cp];

                            // q6 = (low4 | high2 << 4) - 32.
                            const q6_l0 = neon.sub(@as(i8x16, @bitCast(neon.sli_n_u8(neon.@"and"(l0_bits, m4b), neon.@"and"(h0_bits, mask_lo), 4))), m32s);
                            const q6_l1 = neon.sub(@as(i8x16, @bitCast(neon.sli_n_u8(neon.@"and"(l1_bits, m4b), neon.@"and"(h1_bits, mask_lo), 4))), m32s);
                            const q6_h0 = neon.sub(@as(i8x16, @bitCast(neon.orr(neon.shrN(l0_bits, 4), neon.@"and"(h0_bits, mask_hi)))), m32s);
                            const q6_h1 = neon.sub(@as(i8x16, @bitCast(neon.orr(neon.shrN(l1_bits, 4), neon.@"and"(h1_bits, mask_hi)))), m32s);

                            const zero: i32x4 = @splat(0);
                            const sb_acc_0l = neon.mmlaq_s32(neon.mmlaq_s32(zero, q6_l0, q8_l_01[0]), q6_l1, q8_l_01[1]);
                            const sb_acc_0h = neon.mmlaq_s32(neon.mmlaq_s32(zero, q6_h0, q8_h_01[0]), q6_h1, q8_h_01[1]);
                            const sb_acc_1l = neon.mmlaq_s32(neon.mmlaq_s32(zero, q6_l0, q8_l_23[0]), q6_l1, q8_l_23[1]);
                            const sb_acc_1h = neon.mmlaq_s32(neon.mmlaq_s32(zero, q6_h0, q8_h_23[0]), q6_h1, q8_h_23[1]);

                            const scale_idx_l = half * 8 + sb;
                            const scale_idx_h = half * 8 + sb + 4;
                            const scale_vec_l = pairScale(q6_scales[scale_idx_l * 8 + cp * 2], q6_scales[scale_idx_l * 8 + cp * 2 + 1]);
                            const scale_vec_h = pairScale(q6_scales[scale_idx_h * 8 + cp * 2], q6_scales[scale_idx_h * 8 + cp * 2 + 1]);

                            acc[cp] = neon.mlaq_s32(acc[cp], sb_acc_0l, scale_vec_l);
                            acc[cp] = neon.mlaq_s32(acc[cp], sb_acc_0h, scale_vec_h);
                            acc[cp + 4] = neon.mlaq_s32(acc[cp + 4], sb_acc_1l, scale_vec_l);
                            acc[cp + 4] = neon.mlaq_s32(acc[cp + 4], sb_acc_1h, scale_vec_h);
                        }
                    }
                }

                const reorder_acc = reorderTile(acc);

                // The super-block scale; q6_K has no mins.
                for (0..q8_k_blocklen) |i| {
                    for (0..2) |j| {
                        const q8_d: f32x4 = @splat(q8_ptr[b].d[i]);
                        const scale = deltas(&q6_ptr[b].d, j) * q8_d;
                        acc_f32[2 * i + j] = neon.mlaq_f32(acc_f32[2 * i + j], neon.cvt_f32_s32(reorder_acc[2 * i + j]), scale);
                    }
                }
            }

            storeTile(s, bs, y, x, &acc_f32);
        }
    }
}

// -----------------------------------------------------------------------------
// q8_0

/// Ports `ggml_gemm_q8_0_4x8_q8_0` (arch/arm/repack.cpp:5006 @c1d0e7a00),
/// the `__ARM_FEATURE_MATMUL_INT8` arm at line 5091.
///
/// Activations go in as `smmla`'s first operand here, so each accumulator
/// is a row-major 2x2 tile -- `[r0c0, r0c1, r1c0, r1c1]` -- and four
/// recombines make whole rows, with no zip.
pub fn gemmQ8_0_4x8(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    const nb: usize = @intCast(@divTrunc(n, c.QK8_0));
    const ncols_interleaved = 4;

    const b_ptr_base: [*]const blocks.block_q8_0x4 = @ptrCast(@alignCast(vx));
    const nr_u: usize = @intCast(nr);
    const nc_u: usize = @intCast(nc);

    var y: usize = 0;
    while (y < nr_u) : (y += 4) {
        const a_ptr_base = @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + (y / 4) * nb;

        var x: usize = 0;
        while (x < nc_u) : (x += ncols_interleaved) {
            const b_ptr = b_ptr_base + (x / 4) * nb;
            const a_ptr = a_ptr_base;

            var acc_f32: [4]f32x4 = @splat(@splat(0));

            for (0..nb) |b| {
                var acc: [4]i32x4 = @splat(@splat(0));

                // Four chunks of eight positions each.
                inline for (0..4) |chunk| {
                    const a01 = loadq(@as([*]const i8, &a_ptr[b].qs) + chunk * 32);
                    const a23 = loadq(@as([*]const i8, &a_ptr[b].qs) + chunk * 32 + 16);
                    const b01 = loadq(@as([*]const i8, &b_ptr[b].qs) + chunk * 32);
                    const b23 = loadq(@as([*]const i8, &b_ptr[b].qs) + chunk * 32 + 16);

                    acc[0] = neon.mmlaq_s32(acc[0], a01, b01);
                    acc[1] = neon.mmlaq_s32(acc[1], a01, b23);
                    acc[2] = neon.mmlaq_s32(acc[2], a23, b01);
                    acc[3] = neon.mmlaq_s32(acc[3], a23, b23);
                }

                // 2x2 tiles to rows.
                const rows = [4]i32x4{
                    combineLow(acc[0], acc[1]),
                    combineHigh(acc[0], acc[1]),
                    combineLow(acc[2], acc[3]),
                    combineHigh(acc[2], acc[3]),
                };

                const a_d = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&a_ptr[b].d)));
                const b_d = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&b_ptr[b].d)));

                // `vfmaq_f32(acc, row, vmulq_laneq_f32(b_d, a_d, r))`: the
                // scale product rounds, the accumulate fuses.
                inline for (0..4) |r| {
                    const lane: f32x4 = @splat(a_d[r]);
                    acc_f32[r] = neon.fma_f32(acc_f32[r], neon.cvt_f32_s32(rows[r]), b_d * lane);
                }
            }

            for (0..4) |row| {
                s[(y + row) * bs + x ..][0..4].* = acc_f32[row];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
