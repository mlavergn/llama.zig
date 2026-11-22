//! AVX2 dot products for the block-per-32 formats, `q1_0`, and the two
//! four-bit codebooks.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (v1.0.0,
//! `b1b770d90`), the `__AVX2__` arms. Each function names the C it replaces
//! and the line it began at.
//!
//! # Which scalar tails are live
//!
//! Several of these end on a scalar loop over the blocks the vector loop did
//! not take. Where the vector loop steps one block at a time to `nb`, that
//! loop never runs and is not ported -- `q4_0`, `q8_0`, `nvfp4`. Where it
//! steps two at a time, `mxfp4` and `iq4_nl`, an odd final block takes it,
//! and it is ported with the fusion the reference compiles: `sumf += d *
//! sumi` is one `vfmadd231ss`.
//!
//! # The fusions were read, not reasoned
//!
//! From the reference `ggml.lib` disassembly: `summs += m * s` in `q4_1` and
//! `q5_1` is one `vfmadd231ss`; every `_mm256_fmadd_ps` is what it says; and
//! no other scalar product feeds an add.

const std = @import("std");
const ggml = @import("../../../module.zig");

/// `GGML_CPU_E8M0_TO_FP32_HALF`: on x86 a lookup (simd-mappings.h:136 @b1b770d90).
inline fn e8m0Half(x: u8) f32 {
    return ggml.cpu.convert.ggml_table_f32_e8m0_half[x];
}

/// `GGML_CPU_UE4M3_TO_FP32`: a lookup on x86 too (simd-mappings.h:143 @b1b770d90).
inline fn ue4m3(x: u8) f32 {
    return ggml.cpu.convert.ggml_table_f32_ue4m3[x];
}

/// Ports `ggml_vec_dot_q1_0_q8_0` (arch/x86/quants.c:555 @b1b770d90), the
/// `__AVX2__` arm.
///
/// One sign bit per weight. Each bit is spread to a byte mask by a shuffle
/// and a compare, and the mask negates `q8_0` -- `(qy ^ sm) - sm` -- so
/// the dot product is a plain sum. A `q1_0` block spans four `q8_0` blocks,
/// summed into `acc_block` (a multiply, then three fused adds) before the
/// block's own scale is fused in.
pub fn ggml_vec_dot_q1_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q1_0, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vy);

    const ones_8 = ggml.cpu.quants.arch.avx2.mm256_set1_epi8(1);
    const ones_16 = ggml.cpu.quants.arch.avx2.mm256_set1_epi16(1);
    const byte_shuf = ggml.cpu.quants.arch.avx2.mm256_setr_epi8(.{
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    });
    const bit_masks = ggml.cpu.quants.arch.avx2.mm256_setr_epi8(.{
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    });
    const zero = ggml.cpu.quants.arch.avx2.mm256_setzero_si256();
    var acc = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();

    for (0..nb) |ib| {
        const d0 = ggml.cpu.quants.arch.common.f(x[ib].d);
        const y_ptr = y + ib * 4;

        var acc_block: ggml.cpu.quants.arch.avx2.m256 = undefined;
        inline for (0..4) |K| {
            const qs32 = std.mem.readInt(u32, x[ib].qs[4 * K ..][0..4], .little);
            const qy = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y_ptr[K].qs);
            const sm = ggml.cpu.quants.arch.avx2.mm256_cmpeq_epi8(
                ggml.cpu.quants.arch.avx2.mm256_and_si256(ggml.cpu.quants.arch.avx2.mm256_shuffle_epi8(ggml.cpu.quants.arch.avx2.mm256_set1_epi32(@bitCast(qs32)), byte_shuf), bit_masks),
                zero,
            );
            const sy = ggml.cpu.quants.arch.avx2.mm256_sub_epi8(ggml.cpu.quants.arch.avx2.mm256_xor_si256(qy, sm), sm);
            const s32 = ggml.cpu.quants.arch.avx2.mm256_madd_epi16(ggml.cpu.quants.arch.avx2.mm256_maddubs_epi16(ones_8, sy), ones_16);
            const dy = ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(y_ptr[K].d));
            acc_block = if (K == 0)
                ggml.cpu.quants.arch.avx2.mm256_mul_ps(dy, ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(s32))
            else
                ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(dy, ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(s32), acc_block);
        }
        acc = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(ggml.cpu.quants.arch.avx2.mm256_set1_ps(d0), acc_block, acc);
    }

    s[0] = ggml.cpu.quants.arch.common.hsum_float_8(acc);
}

/// Ports `q2_0_unpack_32_avx2` (arch/x86/quants.c:703 @b1b770d90): four
/// consecutive Q2_0 weights per byte. Byte `b` becomes `b & 3`, `(b >> 2) & 3`,
/// `(b >> 4) & 3`, `(b >> 6) & 3` -- OR, never add: the overlapping copies only
/// touch masked-off bits.
inline fn q2_0_unpack_32_avx2(qs: [*]const u8) ggml.cpu.quants.arch.avx2.m256i {
    var v = ggml.cpu.quants.arch.avx2.mm256_cvtepu8_epi32(ggml.cpu.quants.arch.avx2.mm_loadl_epi64(qs));
    v = ggml.cpu.quants.arch.avx2.mm256_or_si256(v, ggml.cpu.quants.arch.avx2.mm256_slli_epi32(v, 12));
    v = ggml.cpu.quants.arch.avx2.mm256_or_si256(v, ggml.cpu.quants.arch.avx2.mm256_slli_epi32(v, 6));
    return ggml.cpu.quants.arch.avx2.mm256_and_si256(v, ggml.cpu.quants.arch.avx2.mm256_set1_epi32(0x03030303));
}

/// Ports `q2_0_dot_32_avx2` (arch/x86/quants.c:711 @b1b770d90): 32 weights
/// against one Q8_0 block, as `sum(v * y) - sum(y)`, which is
/// `sum((v - 1) * y)` exactly for every `int8` `y`.
inline fn q2_0_dot_32_avx2(qs: [*]const u8, qy: [*]const i8) ggml.cpu.quants.arch.avx2.m256 {
    const v = q2_0_unpack_32_avx2(qs);
    const y = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(qy);
    const p = ggml.cpu.quants.arch.avx2.mm256_sub_epi16(ggml.cpu.quants.arch.avx2.mm256_maddubs_epi16(v, y), ggml.cpu.quants.arch.avx2.mm256_maddubs_epi16(ggml.cpu.quants.arch.avx2.mm256_set1_epi8(1), y));
    return ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(ggml.cpu.quants.arch.avx2.mm256_madd_epi16(p, ggml.cpu.quants.arch.avx2.mm256_set1_epi16(1)));
}

/// Ports `q2_0_acc_block_avx2` (arch/x86/quants.c:718 @b1b770d90), at its
/// `__FMA__` arm, which x86-64-v3 compiles: one Q2_0 block over its two Q8_0
/// blocks into `acc`.
inline fn q2_0_acc_block_avx2(x: *const ggml.quants.blocks.Q2_0, y: [*]const ggml.quants.blocks.Q8_0, acc_in: ggml.cpu.quants.arch.avx2.m256) ggml.cpu.quants.arch.avx2.m256 {
    const dx = ggml.cpu.quants.arch.common.f(x.d);
    const d0 = ggml.cpu.quants.arch.avx2.mm256_set1_ps(dx * ggml.cpu.quants.arch.common.f(y[0].d));
    const d1 = ggml.cpu.quants.arch.avx2.mm256_set1_ps(dx * ggml.cpu.quants.arch.common.f(y[1].d));
    const p0 = q2_0_dot_32_avx2(x.qs[0..].ptr, &y[0].qs);
    const p1 = q2_0_dot_32_avx2(x.qs[8..].ptr, &y[1].qs);
    const acc = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(d0, p0, acc_in);
    return ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(d1, p1, acc);
}

/// Ports `ggml_vec_dot_q2_0_q8_0` (arch/x86/quants.c:734 @b1b770d90), its
/// `__AVX2__` arm (#30447). The generic body is dead on x86 as on ARM.
///
/// Two Q2_0 blocks per step, each into its own accumulator (two FMA chains),
/// an odd final block into the first. The `#else` arm, the generic, is for
/// x86 without AVX2, which is not a target.
pub fn ggml_vec_dot_q2_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK2_0;
    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q2_0, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vy);
    const nb: usize = @intCast(@divTrunc(n, qk));

    // two Q2_0 blocks per step, each into its own accumulator (two FMA chains)
    var acc0 = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();
    var acc1 = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();
    var i: usize = 0;
    while (i + 1 < nb) : (i += 2) {
        acc0 = q2_0_acc_block_avx2(&x[i + 0], y + 2 * i + 0, acc0);
        acc1 = q2_0_acc_block_avx2(&x[i + 1], y + 2 * i + 2, acc1);
    }
    if (i < nb) {
        acc0 = q2_0_acc_block_avx2(&x[i], y + 2 * i, acc0);
    }

    s[0] = ggml.cpu.quants.arch.common.hsum_float_8(ggml.cpu.quants.arch.avx2.mm256_add_ps(acc0, acc1));
}

/// Ports `ggml_vec_dot_q4_0_q8_0` (arch/x86/quants.c:765 @b1b770d90), the
/// `__AVX2__` arm.
pub fn ggml_vec_dot_q4_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q4_0, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vy);

    // Initialize accumulator with zeros
    var acc = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();

    // Main loop
    for (0..nb) |ib| {
        // Compute combined scale for the block
        const d = ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(x[ib].d) * ggml.cpu.quants.arch.common.f(y[ib].d));

        var qx = ggml.cpu.quants.arch.common.bytes_from_nibbles_32(&x[ib].qs);

        // Bytes in [0, 15], offset into [-8, 7].
        const off = ggml.cpu.quants.arch.avx2.mm256_set1_epi8(8);
        qx = ggml.cpu.quants.arch.avx2.mm256_sub_epi8(qx, off);

        const qy = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib].qs);

        const q = ggml.cpu.quants.arch.common.mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(d, q, acc);
    }

    s[0] = ggml.cpu.quants.arch.common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_q4_1_q8_1` (arch/x86/quants.c:923 @b1b770d90), the
/// `__AVX2__ || __AVX__` arm with its `__AVX2__` fused accumulate.
///
/// The weights are unsigned, so no sign trick: `maddubs` takes them
/// directly. The minimum comes in through `q8_1`'s precomputed `s`, one
/// fused `summs += m * s` per block.
pub fn ggml_vec_dot_q4_1_q8_1(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK8_1;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q4_1, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_1, vy);

    // Initialize accumulator with zeros
    var acc = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();

    var summs: f32 = 0;

    // Main loop
    for (0..nb) |ib| {
        const d0 = ggml.cpu.quants.arch.common.f(x[ib].d);
        const d1 = ggml.cpu.quants.arch.common.f(y[ib].d);

        summs = @mulAdd(f32, ggml.cpu.quants.arch.common.f(x[ib].m), ggml.cpu.quants.arch.common.f(y[ib].s), summs);

        const d0v = ggml.cpu.quants.arch.avx2.mm256_set1_ps(d0);
        const d1v = ggml.cpu.quants.arch.avx2.mm256_set1_ps(d1);

        // Compute combined scales
        const d0d1 = ggml.cpu.quants.arch.avx2.mm256_mul_ps(d0v, d1v);

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const qx = ggml.cpu.quants.arch.common.bytes_from_nibbles_32(&x[ib].qs);
        const qy = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib].qs);

        const xy = ggml.cpu.quants.arch.common.mul_sum_us8_pairs_float(qx, qy);

        // Accumulate d0*d1*x*y
        acc = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(d0d1, xy, acc);
    }

    s[0] = ggml.cpu.quants.arch.common.hsum_float_8(acc) + summs;
}

/// Ports `ggml_vec_dot_mxfp4_q8_0` (arch/x86/quants.c:982 @b1b770d90), the
/// `__AVX2__` arm and the scalar tail after it.
///
/// Two blocks a step into two accumulators, summed only at the end; the
/// four-bit codes go through `kvalues_mxfp4` with a 16-entry `pshufb`. An
/// odd last block takes the scalar tail.
pub fn ggml_vec_dot_mxfp4_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };
    std.debug.assert(@rem(n, ggml.impl.ggml_c.QK_MXFP4) == 0);
    comptime std.debug.assert(ggml.impl.ggml_c.QK_MXFP4 == ggml.impl.ggml_c.QK8_0);

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.MXFP4, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vy);

    const nb: usize = @intCast(@divTrunc(n, ggml.impl.ggml_c.QK_MXFP4));

    var ib: usize = 0;
    var sumf: f32 = 0;

    const values128 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(&ggml.impl.ggml_c.kvalues_mxfp4);
    const m4b = ggml.cpu.quants.arch.avx2.mm_set1_epi8(0x0f);
    const mone = ggml.cpu.quants.arch.avx2.mm256_set1_epi16(1);

    var accum1 = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();
    var accum2 = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();

    while (ib + 1 < nb) : (ib += 2) {
        const q4bits_1 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(&x[ib + 0].qs);
        const q4bits_2 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(&x[ib + 1].qs);
        const q8b_1 = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib + 0].qs);
        const q8b_2 = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib + 1].qs);
        const q4b_1 = ggml.cpu.quants.arch.avx2.mm256_set_m128i(
            ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(ggml.cpu.quants.arch.avx2.mm_srli_epi16(q4bits_1, 4), m4b)),
            ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(q4bits_1, m4b)),
        );
        const q4b_2 = ggml.cpu.quants.arch.avx2.mm256_set_m128i(
            ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(ggml.cpu.quants.arch.avx2.mm_srli_epi16(q4bits_2, 4), m4b)),
            ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(q4bits_2, m4b)),
        );
        const p16_1 = ggml.cpu.quants.arch.common.mul_add_epi8(q4b_1, q8b_1);
        const p16_2 = ggml.cpu.quants.arch.common.mul_add_epi8(q4b_2, q8b_2);
        const p_1 = ggml.cpu.quants.arch.avx2.mm256_madd_epi16(p16_1, mone);
        const p_2 = ggml.cpu.quants.arch.avx2.mm256_madd_epi16(p16_2, mone);
        const scale0 = ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(y[ib + 0].d) * e8m0Half(x[ib + 0].e));
        const scale1 = ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(y[ib + 1].d) * e8m0Half(x[ib + 1].e));
        accum1 = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(scale0, ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(p_1), accum1);
        accum2 = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(scale1, ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = ggml.cpu.quants.arch.common.hsum_float_8(ggml.cpu.quants.arch.avx2.mm256_add_ps(accum1, accum2));

    while (ib < nb) : (ib += 1) {
        const d = ggml.cpu.quants.arch.common.f(y[ib].d) * e8m0Half(x[ib].e);
        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        for (0..ggml.impl.ggml_c.QK_MXFP4 / 2) |j| {
            sumi1 += @as(i32, y[ib].qs[j + 0]) * ggml.impl.ggml_c.kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += @as(i32, y[ib].qs[j + ggml.impl.ggml_c.QK_MXFP4 / 2]) * ggml.impl.ggml_c.kvalues_mxfp4[x[ib].qs[j] >> 4];
        }
        sumf = @mulAdd(f32, d, @floatFromInt(sumi1 + sumi2), sumf);
    }
    s[0] = sumf;
}

/// Ports `ggml_vec_dot_nvfp4_q8_0` (arch/x86/quants.c:1068 @b1b770d90), the
/// `__AVX2__` arm.
///
/// A 64-weight block with four UE4M3 sub-block scales over two `q8_0`
/// blocks. The nibbles come out of `pshufb` low-then-high per sixteen bytes
/// and are put back in weight order by the two `unpack`s.
pub fn ggml_vec_dot_nvfp4_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };
    std.debug.assert(@rem(n, ggml.impl.ggml_c.QK_NVFP4) == 0);

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.NVFP4, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vy);

    const nb: usize = @intCast(@divTrunc(n, ggml.impl.ggml_c.QK_NVFP4));

    const values128 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(&ggml.impl.ggml_c.kvalues_mxfp4);
    const m4b = ggml.cpu.quants.arch.avx2.mm_set1_epi8(0x0f);
    const mone = ggml.cpu.quants.arch.avx2.mm256_set1_epi16(1);

    var accum = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();
    for (0..nb) |ib| {
        const q4bits_01 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(x[ib].qs[0..].ptr);
        const q4bits_23 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(x[ib].qs[16..].ptr);

        const q8_01 = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[2 * ib + 0].qs);
        const q8_23 = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[2 * ib + 1].qs);

        const q4_01_lo = ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(q4bits_01, m4b));
        const q4_01_hi = ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(ggml.cpu.quants.arch.avx2.mm_srli_epi16(q4bits_01, 4), m4b));
        const q4_23_lo = ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(q4bits_23, m4b));
        const q4_23_hi = ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(ggml.cpu.quants.arch.avx2.mm_srli_epi16(q4bits_23, 4), m4b));

        // reordering
        const q4_01 = ggml.cpu.quants.arch.avx2.mm256_set_m128i(ggml.cpu.quants.arch.avx2.mm_unpackhi_epi64(q4_01_lo, q4_01_hi), ggml.cpu.quants.arch.avx2.mm_unpacklo_epi64(q4_01_lo, q4_01_hi));
        const q4_23 = ggml.cpu.quants.arch.avx2.mm256_set_m128i(ggml.cpu.quants.arch.avx2.mm_unpackhi_epi64(q4_23_lo, q4_23_hi), ggml.cpu.quants.arch.avx2.mm_unpacklo_epi64(q4_23_lo, q4_23_hi));

        const p01 = ggml.cpu.quants.arch.common.mul_add_epi8(q4_01, q8_01);
        const p_1 = ggml.cpu.quants.arch.avx2.mm256_madd_epi16(p01, mone);

        const p23 = ggml.cpu.quants.arch.common.mul_add_epi8(q4_23, q8_23);
        const p_2 = ggml.cpu.quants.arch.avx2.mm256_madd_epi16(p23, mone);

        const dy0 = ggml.cpu.quants.arch.common.f(y[2 * ib].d);
        const dy1 = ggml.cpu.quants.arch.common.f(y[2 * ib + 1].d);

        const s0 = ue4m3(x[ib].d[0]) * dy0;
        const s1 = ue4m3(x[ib].d[1]) * dy0;
        const s2 = ue4m3(x[ib].d[2]) * dy1;
        const s3 = ue4m3(x[ib].d[3]) * dy1;

        const scales01 = ggml.cpu.quants.arch.avx2.mm256_set_m128(ggml.cpu.quants.arch.avx2.mm_set1_ps(s1), ggml.cpu.quants.arch.avx2.mm_set1_ps(s0));
        const scales23 = ggml.cpu.quants.arch.avx2.mm256_set_m128(ggml.cpu.quants.arch.avx2.mm_set1_ps(s3), ggml.cpu.quants.arch.avx2.mm_set1_ps(s2));

        accum = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(scales01, ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(p_1), accum);
        accum = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(scales23, ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(p_2), accum);
    }
    s[0] = ggml.cpu.quants.arch.common.hsum_float_8(accum);
}

/// Ports `ggml_vec_dot_q5_0_q8_0` (arch/x86/quants.c:1206 @b1b770d90), the
/// `__AVX2__` arm.
///
/// The fifth bit is folded in as a sign: `andnot` leaves `0xF0` where the
/// bit is clear, so `qx | bxhi` is the nibble minus 16 as a signed byte.
pub fn ggml_vec_dot_q5_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    comptime std.debug.assert(qk == ggml.impl.ggml_c.QK5_0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q5_0, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vy);

    // Initialize accumulator with zeros
    var acc = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();

    // Main loop
    for (0..nb) |ib| {
        // Compute combined scale for the block
        const d = ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(x[ib].d) * ggml.cpu.quants.arch.common.f(y[ib].d));

        var qx = ggml.cpu.quants.arch.common.bytes_from_nibbles_32(&x[ib].qs);
        var bxhi = ggml.cpu.quants.arch.common.bytes_from_bits_32(&x[ib].qh);
        bxhi = ggml.cpu.quants.arch.avx2.mm256_andnot_si256(bxhi, ggml.cpu.quants.arch.avx2.mm256_set1_epi8(@bitCast(@as(u8, 0xF0))));
        qx = ggml.cpu.quants.arch.avx2.mm256_or_si256(qx, bxhi);

        const qy = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib].qs);

        const q = ggml.cpu.quants.arch.common.mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(d, q, acc);
    }

    s[0] = ggml.cpu.quants.arch.common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_q5_1_q8_1` (arch/x86/quants.c:1286 @b1b770d90), the
/// `__AVX2__` arm.
///
/// Unsigned five-bit weights, so the fifth bit is OR-ed in as `0x10`.
/// Note the operand order of the fused accumulate, `q * (dx * dy) + acc`.
pub fn ggml_vec_dot_q5_1_q8_1(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK8_1;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    comptime std.debug.assert(qk == ggml.impl.ggml_c.QK5_1);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q5_1, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_1, vy);

    // Initialize accumulator with zeros
    var acc = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();

    var summs: f32 = 0;

    // Main loop
    for (0..nb) |ib| {
        const dx = ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(x[ib].d));

        summs = @mulAdd(f32, ggml.cpu.quants.arch.common.f(x[ib].m), ggml.cpu.quants.arch.common.f(y[ib].s), summs);

        var qx = ggml.cpu.quants.arch.common.bytes_from_nibbles_32(&x[ib].qs);
        var bxhi = ggml.cpu.quants.arch.common.bytes_from_bits_32(&x[ib].qh);
        bxhi = ggml.cpu.quants.arch.avx2.mm256_and_si256(bxhi, ggml.cpu.quants.arch.avx2.mm256_set1_epi8(0x10));
        qx = ggml.cpu.quants.arch.avx2.mm256_or_si256(qx, bxhi);

        const dy = ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(y[ib].d));
        const qy = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib].qs);

        const q = ggml.cpu.quants.arch.common.mul_sum_us8_pairs_float(qx, qy);

        acc = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(q, ggml.cpu.quants.arch.avx2.mm256_mul_ps(dx, dy), acc);
    }

    s[0] = ggml.cpu.quants.arch.common.hsum_float_8(acc) + summs;
}

/// Ports `ggml_vec_dot_q8_0_q8_0` (arch/x86/quants.c:1372 @b1b770d90), the
/// `__AVX2__` arm.
pub fn ggml_vec_dot_q8_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vy);

    // Initialize accumulator with zeros
    var acc = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();

    // Main loop
    for (0..nb) |ib| {
        // Compute combined scale for the block
        const d = ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(x[ib].d) * ggml.cpu.quants.arch.common.f(y[ib].d));
        const qx = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&x[ib].qs);
        const qy = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib].qs);

        const q = ggml.cpu.quants.arch.common.mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(d, q, acc);
    }

    s[0] = ggml.cpu.quants.arch.common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_iq4_nl_q8_0` (arch/x86/quants.c:3984 @b1b770d90), the
/// `__AVX2__` arm and the scalar tail after it.
///
/// `mxfp4`'s shape over `kvalues_iq4nl`, with an fp16 delta in place of the
/// E8M0 exponent.
pub fn ggml_vec_dot_iq4_nl_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };
    std.debug.assert(@rem(n, ggml.impl.ggml_c.QK4_NL) == 0);
    comptime std.debug.assert(ggml.impl.ggml_c.QK4_NL == ggml.impl.ggml_c.QK8_0);

    const x = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.IQ4_NL, vx);
    const y = ggml.cpu.quants.arch.common.as(ggml.quants.blocks.Q8_0, vy);

    const nb: usize = @intCast(@divTrunc(n, ggml.impl.ggml_c.QK4_NL));

    var ib: usize = 0;
    var sumf: f32 = 0;

    const values128 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(&ggml.impl.ggml_c.kvalues_iq4nl);
    const m4b = ggml.cpu.quants.arch.avx2.mm_set1_epi8(0x0f);
    const mone = ggml.cpu.quants.arch.avx2.mm256_set1_epi16(1);

    var accum1 = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();
    var accum2 = ggml.cpu.quants.arch.avx2.mm256_setzero_ps();
    while (ib + 1 < nb) : (ib += 2) {
        const q4bits_1 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(&x[ib + 0].qs);
        const q4bits_2 = ggml.cpu.quants.arch.avx2.mm_loadu_si128(&x[ib + 1].qs);
        const q8b_1 = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib + 0].qs);
        const q8b_2 = ggml.cpu.quants.arch.avx2.mm256_loadu_si256(&y[ib + 1].qs);
        const q4b_1 = ggml.cpu.quants.arch.avx2.mm256_set_m128i(
            ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(ggml.cpu.quants.arch.avx2.mm_srli_epi16(q4bits_1, 4), m4b)),
            ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(q4bits_1, m4b)),
        );
        const q4b_2 = ggml.cpu.quants.arch.avx2.mm256_set_m128i(
            ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(ggml.cpu.quants.arch.avx2.mm_srli_epi16(q4bits_2, 4), m4b)),
            ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(values128, ggml.cpu.quants.arch.avx2.mm_and_si128(q4bits_2, m4b)),
        );
        const p16_1 = ggml.cpu.quants.arch.common.mul_add_epi8(q4b_1, q8b_1);
        const p16_2 = ggml.cpu.quants.arch.common.mul_add_epi8(q4b_2, q8b_2);
        const p_1 = ggml.cpu.quants.arch.avx2.mm256_madd_epi16(p16_1, mone);
        const p_2 = ggml.cpu.quants.arch.avx2.mm256_madd_epi16(p16_2, mone);
        accum1 = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(y[ib + 0].d) * ggml.cpu.quants.arch.common.f(x[ib + 0].d)), ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(p_1), accum1);
        accum2 = ggml.cpu.quants.arch.avx2.mm256_fmadd_ps(ggml.cpu.quants.arch.avx2.mm256_set1_ps(ggml.cpu.quants.arch.common.f(y[ib + 1].d) * ggml.cpu.quants.arch.common.f(x[ib + 1].d)), ggml.cpu.quants.arch.avx2.mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = ggml.cpu.quants.arch.common.hsum_float_8(ggml.cpu.quants.arch.avx2.mm256_add_ps(accum1, accum2));

    while (ib < nb) : (ib += 1) {
        const d = ggml.cpu.quants.arch.common.f(y[ib].d) * ggml.cpu.quants.arch.common.f(x[ib].d);
        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        for (0..ggml.impl.ggml_c.QK4_NL / 2) |j| {
            sumi1 += @as(i32, y[ib].qs[j + 0]) * ggml.impl.ggml_c.kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += @as(i32, y[ib].qs[j + ggml.impl.ggml_c.QK4_NL / 2]) * ggml.impl.ggml_c.kvalues_iq4nl[x[ib].qs[j] >> 4];
        }
        sumf = @mulAdd(f32, d, @floatFromInt(sumi1 + sumi2), sumf);
    }
    s[0] = sumf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
