//! AVX2 dot products for the block-per-32 formats, `q1_0`, and the two
//! four-bit codebooks.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (v0.3.0,
//! `c1d0e7a00`), the `__AVX2__` arms. Each function names the C it replaces
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
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const blocks = @import("../../../quants/blocks.zig");
const avx = @import("avx2.zig");
const common = @import("common.zig");
const c = impl.c;

const f = common.f;
const as = common.as;

/// `GGML_CPU_E8M0_TO_FP32_HALF`: on x86 a lookup (simd-mappings.h:131 @c1d0e7a00).
inline fn e8m0Half(x: u8) f32 {
    return convert.ggml_table_f32_e8m0_half[x];
}

/// `GGML_CPU_UE4M3_TO_FP32`: a lookup on x86 too (simd-mappings.h:138 @c1d0e7a00).
inline fn ue4m3(x: u8) f32 {
    return convert.ggml_table_f32_ue4m3[x];
}

/// Ports `ggml_vec_dot_q1_0_q8_0` (arch/x86/quants.c:555 @c1d0e7a00), the
/// `__AVX2__` arm.
///
/// One sign bit per weight. Each bit is spread to a byte mask by a shuffle
/// and a compare, and the mask negates `q8_0` -- `(qy ^ sm) - sm` -- so
/// the dot product is a plain sum. A `q1_0` block spans four `q8_0` blocks,
/// summed into `acc_block` (a multiply, then three fused adds) before the
/// block's own scale is fused in.
pub export fn ggml_vec_dot_q1_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q1_0, vx);
    const y = as(blocks.Q8_0, vy);

    const ones_8 = avx.mm256_set1_epi8(1);
    const ones_16 = avx.mm256_set1_epi16(1);
    const byte_shuf = avx.mm256_setr_epi8(.{
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    });
    const bit_masks = avx.mm256_setr_epi8(.{
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    });
    const zero = avx.mm256_setzero_si256();
    var acc = avx.mm256_setzero_ps();

    for (0..nb) |ib| {
        const d0 = f(x[ib].d);
        const y_ptr = y + ib * 4;

        var acc_block: avx.m256 = undefined;
        inline for (0..4) |K| {
            const qs32 = std.mem.readInt(u32, x[ib].qs[4 * K ..][0..4], .little);
            const qy = avx.mm256_loadu_si256(&y_ptr[K].qs);
            const sm = avx.mm256_cmpeq_epi8(
                avx.mm256_and_si256(avx.mm256_shuffle_epi8(avx.mm256_set1_epi32(@bitCast(qs32)), byte_shuf), bit_masks),
                zero,
            );
            const sy = avx.mm256_sub_epi8(avx.mm256_xor_si256(qy, sm), sm);
            const s32 = avx.mm256_madd_epi16(avx.mm256_maddubs_epi16(ones_8, sy), ones_16);
            const dy = avx.mm256_set1_ps(f(y_ptr[K].d));
            acc_block = if (K == 0)
                avx.mm256_mul_ps(dy, avx.mm256_cvtepi32_ps(s32))
            else
                avx.mm256_fmadd_ps(dy, avx.mm256_cvtepi32_ps(s32), acc_block);
        }
        acc = avx.mm256_fmadd_ps(avx.mm256_set1_ps(d0), acc_block, acc);
    }

    s[0] = common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_q2_0_q8_0_generic` (ggml-cpu/quants.c:177 @c1d0e7a00)
/// as x86 ships it: `arch-fallback.h:86` renames the generic body to the
/// real name, so on x86 it is the live kernel.
///
/// **Two fusions the shared generic port does not make.** That port,
/// `quants/legacy.zig`, matches goldens captured with contraction off,
/// because on ARM the generic is dead code. Here it is compiled as shipped,
/// and the reference's disassembly fuses both scalar statements:
/// `sumi += d1 * sumi_block` is `vfmadd132ss` and `sumf += d0 * sumi` is
/// `vfmadd231ss`. `scripts/x86-ops-diff` found it: `mul_mat q2_0` was the one
/// op out of 356 that differed once every AVX2 kernel was in.
pub export fn ggml_vec_dot_q2_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK2_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q2_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumf: f32 = 0.0;

    for (0..nb) |i| {
        const d0 = f(x[i].d);

        var sumi: f32 = 0.0;

        // One q2_0 block (64 weights) maps to two q8_0 blocks.
        for (0..2) |k| {
            const yb = &y[i * 2 + k];
            const d1 = f(yb.d);
            var sumi_block: i32 = 0;

            const qs = x[i].qs[k * 8 ..];
            const qy = &yb.qs;

            for (0..8) |b| {
                const byte = qs[b];
                // Four two-bit values, {0,1,2,3} -> {-1,0,1,2}
                inline for (0..4) |t| {
                    const w: i32 = @as(i32, (byte >> (2 * t)) & 3) - 1;
                    sumi_block += w * qy[b * 4 + t];
                }
            }

            sumi = @mulAdd(f32, d1, @floatFromInt(sumi_block), sumi);
        }

        sumf = @mulAdd(f32, d0, sumi, sumf);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q4_0_q8_0` (arch/x86/quants.c:701 @c1d0e7a00), the
/// `__AVX2__` arm.
pub export fn ggml_vec_dot_q4_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q4_0, vx);
    const y = as(blocks.Q8_0, vy);

    // Initialize accumulator with zeros
    var acc = avx.mm256_setzero_ps();

    // Main loop
    for (0..nb) |ib| {
        // Compute combined scale for the block
        const d = avx.mm256_set1_ps(f(x[ib].d) * f(y[ib].d));

        var qx = common.bytes_from_nibbles_32(&x[ib].qs);

        // Bytes in [0, 15], offset into [-8, 7].
        const off = avx.mm256_set1_epi8(8);
        qx = avx.mm256_sub_epi8(qx, off);

        const qy = avx.mm256_loadu_si256(&y[ib].qs);

        const q = common.mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = avx.mm256_fmadd_ps(d, q, acc);
    }

    s[0] = common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_q4_1_q8_1` (arch/x86/quants.c:859 @c1d0e7a00), the
/// `__AVX2__ || __AVX__` arm with its `__AVX2__` fused accumulate.
///
/// The weights are unsigned, so no sign trick: `maddubs` takes them
/// directly. The minimum comes in through `q8_1`'s precomputed `s`, one
/// fused `summs += m * s` per block.
pub export fn ggml_vec_dot_q4_1_q8_1(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_1;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q4_1, vx);
    const y = as(blocks.Q8_1, vy);

    // Initialize accumulator with zeros
    var acc = avx.mm256_setzero_ps();

    var summs: f32 = 0;

    // Main loop
    for (0..nb) |ib| {
        const d0 = f(x[ib].d);
        const d1 = f(y[ib].d);

        summs = @mulAdd(f32, f(x[ib].m), f(y[ib].s), summs);

        const d0v = avx.mm256_set1_ps(d0);
        const d1v = avx.mm256_set1_ps(d1);

        // Compute combined scales
        const d0d1 = avx.mm256_mul_ps(d0v, d1v);

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const qx = common.bytes_from_nibbles_32(&x[ib].qs);
        const qy = avx.mm256_loadu_si256(&y[ib].qs);

        const xy = common.mul_sum_us8_pairs_float(qx, qy);

        // Accumulate d0*d1*x*y
        acc = avx.mm256_fmadd_ps(d0d1, xy, acc);
    }

    s[0] = common.hsum_float_8(acc) + summs;
}

/// Ports `ggml_vec_dot_mxfp4_q8_0` (arch/x86/quants.c:918 @c1d0e7a00), the
/// `__AVX2__` arm and the scalar tail after it.
///
/// Two blocks a step into two accumulators, summed only at the end; the
/// four-bit codes go through `kvalues_mxfp4` with a 16-entry `pshufb`. An
/// odd last block takes the scalar tail.
pub export fn ggml_vec_dot_mxfp4_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };
    std.debug.assert(@rem(n, c.QK_MXFP4) == 0);
    comptime std.debug.assert(c.QK_MXFP4 == c.QK8_0);

    const x = as(blocks.MXFP4, vx);
    const y = as(blocks.Q8_0, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_MXFP4));

    var ib: usize = 0;
    var sumf: f32 = 0;

    const values128 = avx.mm_loadu_si128(&c.kvalues_mxfp4);
    const m4b = avx.mm_set1_epi8(0x0f);
    const mone = avx.mm256_set1_epi16(1);

    var accum1 = avx.mm256_setzero_ps();
    var accum2 = avx.mm256_setzero_ps();

    while (ib + 1 < nb) : (ib += 2) {
        const q4bits_1 = avx.mm_loadu_si128(&x[ib + 0].qs);
        const q4bits_2 = avx.mm_loadu_si128(&x[ib + 1].qs);
        const q8b_1 = avx.mm256_loadu_si256(&y[ib + 0].qs);
        const q8b_2 = avx.mm256_loadu_si256(&y[ib + 1].qs);
        const q4b_1 = avx.mm256_set_m128i(
            avx.mm_shuffle_epi8(values128, avx.mm_and_si128(avx.mm_srli_epi16(q4bits_1, 4), m4b)),
            avx.mm_shuffle_epi8(values128, avx.mm_and_si128(q4bits_1, m4b)),
        );
        const q4b_2 = avx.mm256_set_m128i(
            avx.mm_shuffle_epi8(values128, avx.mm_and_si128(avx.mm_srli_epi16(q4bits_2, 4), m4b)),
            avx.mm_shuffle_epi8(values128, avx.mm_and_si128(q4bits_2, m4b)),
        );
        const p16_1 = common.mul_add_epi8(q4b_1, q8b_1);
        const p16_2 = common.mul_add_epi8(q4b_2, q8b_2);
        const p_1 = avx.mm256_madd_epi16(p16_1, mone);
        const p_2 = avx.mm256_madd_epi16(p16_2, mone);
        const scale0 = avx.mm256_set1_ps(f(y[ib + 0].d) * e8m0Half(x[ib + 0].e));
        const scale1 = avx.mm256_set1_ps(f(y[ib + 1].d) * e8m0Half(x[ib + 1].e));
        accum1 = avx.mm256_fmadd_ps(scale0, avx.mm256_cvtepi32_ps(p_1), accum1);
        accum2 = avx.mm256_fmadd_ps(scale1, avx.mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = common.hsum_float_8(avx.mm256_add_ps(accum1, accum2));

    while (ib < nb) : (ib += 1) {
        const d = f(y[ib].d) * e8m0Half(x[ib].e);
        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        for (0..c.QK_MXFP4 / 2) |j| {
            sumi1 += @as(i32, y[ib].qs[j + 0]) * c.kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += @as(i32, y[ib].qs[j + c.QK_MXFP4 / 2]) * c.kvalues_mxfp4[x[ib].qs[j] >> 4];
        }
        sumf = @mulAdd(f32, d, @floatFromInt(sumi1 + sumi2), sumf);
    }
    s[0] = sumf;
}

/// Ports `ggml_vec_dot_nvfp4_q8_0` (arch/x86/quants.c:1004 @c1d0e7a00), the
/// `__AVX2__` arm.
///
/// A 64-weight block with four UE4M3 sub-block scales over two `q8_0`
/// blocks. The nibbles come out of `pshufb` low-then-high per sixteen bytes
/// and are put back in weight order by the two `unpack`s.
pub export fn ggml_vec_dot_nvfp4_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };
    std.debug.assert(@rem(n, c.QK_NVFP4) == 0);

    const x = as(blocks.NVFP4, vx);
    const y = as(blocks.Q8_0, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_NVFP4));

    const values128 = avx.mm_loadu_si128(&c.kvalues_mxfp4);
    const m4b = avx.mm_set1_epi8(0x0f);
    const mone = avx.mm256_set1_epi16(1);

    var accum = avx.mm256_setzero_ps();
    for (0..nb) |ib| {
        const q4bits_01 = avx.mm_loadu_si128(x[ib].qs[0..].ptr);
        const q4bits_23 = avx.mm_loadu_si128(x[ib].qs[16..].ptr);

        const q8_01 = avx.mm256_loadu_si256(&y[2 * ib + 0].qs);
        const q8_23 = avx.mm256_loadu_si256(&y[2 * ib + 1].qs);

        const q4_01_lo = avx.mm_shuffle_epi8(values128, avx.mm_and_si128(q4bits_01, m4b));
        const q4_01_hi = avx.mm_shuffle_epi8(values128, avx.mm_and_si128(avx.mm_srli_epi16(q4bits_01, 4), m4b));
        const q4_23_lo = avx.mm_shuffle_epi8(values128, avx.mm_and_si128(q4bits_23, m4b));
        const q4_23_hi = avx.mm_shuffle_epi8(values128, avx.mm_and_si128(avx.mm_srli_epi16(q4bits_23, 4), m4b));

        // reordering
        const q4_01 = avx.mm256_set_m128i(avx.mm_unpackhi_epi64(q4_01_lo, q4_01_hi), avx.mm_unpacklo_epi64(q4_01_lo, q4_01_hi));
        const q4_23 = avx.mm256_set_m128i(avx.mm_unpackhi_epi64(q4_23_lo, q4_23_hi), avx.mm_unpacklo_epi64(q4_23_lo, q4_23_hi));

        const p01 = common.mul_add_epi8(q4_01, q8_01);
        const p_1 = avx.mm256_madd_epi16(p01, mone);

        const p23 = common.mul_add_epi8(q4_23, q8_23);
        const p_2 = avx.mm256_madd_epi16(p23, mone);

        const dy0 = f(y[2 * ib].d);
        const dy1 = f(y[2 * ib + 1].d);

        const s0 = ue4m3(x[ib].d[0]) * dy0;
        const s1 = ue4m3(x[ib].d[1]) * dy0;
        const s2 = ue4m3(x[ib].d[2]) * dy1;
        const s3 = ue4m3(x[ib].d[3]) * dy1;

        const scales01 = avx.mm256_set_m128(avx.mm_set1_ps(s1), avx.mm_set1_ps(s0));
        const scales23 = avx.mm256_set_m128(avx.mm_set1_ps(s3), avx.mm_set1_ps(s2));

        accum = avx.mm256_fmadd_ps(scales01, avx.mm256_cvtepi32_ps(p_1), accum);
        accum = avx.mm256_fmadd_ps(scales23, avx.mm256_cvtepi32_ps(p_2), accum);
    }
    s[0] = common.hsum_float_8(accum);
}

/// Ports `ggml_vec_dot_q5_0_q8_0` (arch/x86/quants.c:1142 @c1d0e7a00), the
/// `__AVX2__` arm.
///
/// The fifth bit is folded in as a sign: `andnot` leaves `0xF0` where the
/// bit is clear, so `qx | bxhi` is the nibble minus 16 as a signed byte.
pub export fn ggml_vec_dot_q5_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    comptime std.debug.assert(qk == c.QK5_0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q5_0, vx);
    const y = as(blocks.Q8_0, vy);

    // Initialize accumulator with zeros
    var acc = avx.mm256_setzero_ps();

    // Main loop
    for (0..nb) |ib| {
        // Compute combined scale for the block
        const d = avx.mm256_set1_ps(f(x[ib].d) * f(y[ib].d));

        var qx = common.bytes_from_nibbles_32(&x[ib].qs);
        var bxhi = common.bytes_from_bits_32(&x[ib].qh);
        bxhi = avx.mm256_andnot_si256(bxhi, avx.mm256_set1_epi8(@bitCast(@as(u8, 0xF0))));
        qx = avx.mm256_or_si256(qx, bxhi);

        const qy = avx.mm256_loadu_si256(&y[ib].qs);

        const q = common.mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = avx.mm256_fmadd_ps(d, q, acc);
    }

    s[0] = common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_q5_1_q8_1` (arch/x86/quants.c:1222 @c1d0e7a00), the
/// `__AVX2__` arm.
///
/// Unsigned five-bit weights, so the fifth bit is OR-ed in as `0x10`.
/// Note the operand order of the fused accumulate, `q * (dx * dy) + acc`.
pub export fn ggml_vec_dot_q5_1_q8_1(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_1;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    comptime std.debug.assert(qk == c.QK5_1);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q5_1, vx);
    const y = as(blocks.Q8_1, vy);

    // Initialize accumulator with zeros
    var acc = avx.mm256_setzero_ps();

    var summs: f32 = 0;

    // Main loop
    for (0..nb) |ib| {
        const dx = avx.mm256_set1_ps(f(x[ib].d));

        summs = @mulAdd(f32, f(x[ib].m), f(y[ib].s), summs);

        var qx = common.bytes_from_nibbles_32(&x[ib].qs);
        var bxhi = common.bytes_from_bits_32(&x[ib].qh);
        bxhi = avx.mm256_and_si256(bxhi, avx.mm256_set1_epi8(0x10));
        qx = avx.mm256_or_si256(qx, bxhi);

        const dy = avx.mm256_set1_ps(f(y[ib].d));
        const qy = avx.mm256_loadu_si256(&y[ib].qs);

        const q = common.mul_sum_us8_pairs_float(qx, qy);

        acc = avx.mm256_fmadd_ps(q, avx.mm256_mul_ps(dx, dy), acc);
    }

    s[0] = common.hsum_float_8(acc) + summs;
}

/// Ports `ggml_vec_dot_q8_0_q8_0` (arch/x86/quants.c:1308 @c1d0e7a00), the
/// `__AVX2__` arm.
pub export fn ggml_vec_dot_q8_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q8_0, vx);
    const y = as(blocks.Q8_0, vy);

    // Initialize accumulator with zeros
    var acc = avx.mm256_setzero_ps();

    // Main loop
    for (0..nb) |ib| {
        // Compute combined scale for the block
        const d = avx.mm256_set1_ps(f(x[ib].d) * f(y[ib].d));
        const qx = avx.mm256_loadu_si256(&x[ib].qs);
        const qy = avx.mm256_loadu_si256(&y[ib].qs);

        const q = common.mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = avx.mm256_fmadd_ps(d, q, acc);
    }

    s[0] = common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_iq4_nl_q8_0` (arch/x86/quants.c:3920 @c1d0e7a00), the
/// `__AVX2__` arm and the scalar tail after it.
///
/// `mxfp4`'s shape over `kvalues_iq4nl`, with an fp16 delta in place of the
/// E8M0 exponent.
pub export fn ggml_vec_dot_iq4_nl_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };
    std.debug.assert(@rem(n, c.QK4_NL) == 0);
    comptime std.debug.assert(c.QK4_NL == c.QK8_0);

    const x = as(blocks.IQ4_NL, vx);
    const y = as(blocks.Q8_0, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK4_NL));

    var ib: usize = 0;
    var sumf: f32 = 0;

    const values128 = avx.mm_loadu_si128(&c.kvalues_iq4nl);
    const m4b = avx.mm_set1_epi8(0x0f);
    const mone = avx.mm256_set1_epi16(1);

    var accum1 = avx.mm256_setzero_ps();
    var accum2 = avx.mm256_setzero_ps();
    while (ib + 1 < nb) : (ib += 2) {
        const q4bits_1 = avx.mm_loadu_si128(&x[ib + 0].qs);
        const q4bits_2 = avx.mm_loadu_si128(&x[ib + 1].qs);
        const q8b_1 = avx.mm256_loadu_si256(&y[ib + 0].qs);
        const q8b_2 = avx.mm256_loadu_si256(&y[ib + 1].qs);
        const q4b_1 = avx.mm256_set_m128i(
            avx.mm_shuffle_epi8(values128, avx.mm_and_si128(avx.mm_srli_epi16(q4bits_1, 4), m4b)),
            avx.mm_shuffle_epi8(values128, avx.mm_and_si128(q4bits_1, m4b)),
        );
        const q4b_2 = avx.mm256_set_m128i(
            avx.mm_shuffle_epi8(values128, avx.mm_and_si128(avx.mm_srli_epi16(q4bits_2, 4), m4b)),
            avx.mm_shuffle_epi8(values128, avx.mm_and_si128(q4bits_2, m4b)),
        );
        const p16_1 = common.mul_add_epi8(q4b_1, q8b_1);
        const p16_2 = common.mul_add_epi8(q4b_2, q8b_2);
        const p_1 = avx.mm256_madd_epi16(p16_1, mone);
        const p_2 = avx.mm256_madd_epi16(p16_2, mone);
        accum1 = avx.mm256_fmadd_ps(avx.mm256_set1_ps(f(y[ib + 0].d) * f(x[ib + 0].d)), avx.mm256_cvtepi32_ps(p_1), accum1);
        accum2 = avx.mm256_fmadd_ps(avx.mm256_set1_ps(f(y[ib + 1].d) * f(x[ib + 1].d)), avx.mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = common.hsum_float_8(avx.mm256_add_ps(accum1, accum2));

    while (ib < nb) : (ib += 1) {
        const d = f(y[ib].d) * f(x[ib].d);
        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        for (0..c.QK4_NL / 2) |j| {
            sumi1 += @as(i32, y[ib].qs[j + 0]) * c.kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += @as(i32, y[ib].qs[j + c.QK4_NL / 2]) * c.kvalues_iq4nl[x[ib].qs[j] >> 4];
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
