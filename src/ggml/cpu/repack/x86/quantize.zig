//! The AVX2 `ggml_quantize_mat_*_4x8`: quantize four rows at a time into an
//! interleaved block.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` at v1.0.0
//! (`b1b770d90`), the `__AVX2__` arms. Each declaration names the C++ it
//! replaces and the line it began at.
//!
//! # The same arithmetic as the row quantizers
//!
//! `q8_0` is `quants/x86/rows.zig`'s body four rows at once: `maxps` over
//! the absolute values, `127 / max`, round-to-nearest-even, and the two
//! saturating packs with their permute. `q8_K` keeps the sign of the
//! largest element -- a mask tracks which lanes ever held it -- and stores
//! `bsums` already interleaved for the gemm.

const std = @import("std");
const cpu = @import("../../module.zig");

const QK8_0 = 32;
const QK_K = 256;

/// The rounding, the packs and the permute both
/// `ggml_quantize_mat_q8_0_4x8` and `ggml_quantize_mat_q8_K_4x8` end on
/// (arch/x86/repack.cpp:178, 290 @b1b770d90): four registers of 32-bit
/// integers to 32 bytes in order.
inline fn packBytes(v0: cpu.quants.arch.avx2.m256, v1: cpu.quants.arch.avx2.m256, v2: cpu.quants.arch.avx2.m256, v3: cpu.quants.arch.avx2.m256) cpu.quants.arch.avx2.m256i {
    // Round to nearest integer, then convert. The C's `i0`..`i3`, renamed
    // digit for digit: Zig reserves them as integer types.
    var j0 = cpu.quants.arch.avx2.mm256_cvtps_epi32(cpu.quants.arch.avx2.mm256_round_ps(v0, cpu.quants.arch.avx2.MM_ROUND_NEAREST));
    const j1 = cpu.quants.arch.avx2.mm256_cvtps_epi32(cpu.quants.arch.avx2.mm256_round_ps(v1, cpu.quants.arch.avx2.MM_ROUND_NEAREST));
    var j2 = cpu.quants.arch.avx2.mm256_cvtps_epi32(cpu.quants.arch.avx2.mm256_round_ps(v2, cpu.quants.arch.avx2.MM_ROUND_NEAREST));
    const j3 = cpu.quants.arch.avx2.mm256_cvtps_epi32(cpu.quants.arch.avx2.mm256_round_ps(v3, cpu.quants.arch.avx2.MM_ROUND_NEAREST));

    // Convert int32 to int16
    j0 = cpu.quants.arch.avx2.mm256_packs_epi32(j0, j1);
    j2 = cpu.quants.arch.avx2.mm256_packs_epi32(j2, j3);
    // Convert int16 to int8
    j0 = cpu.quants.arch.avx2.mm256_packs_epi16(j0, j2);

    // Permute the quantized weights into order after the pack.
    const perm = cpu.quants.arch.avx2.mm256_setr_epi32(.{ 0, 4, 1, 5, 2, 6, 3, 7 });
    return cpu.quants.arch.avx2.mm256_permutevar8x32_epi32(j0, perm);
}

/// `max(|v0|, ..., |v3|)` reduced to a scalar, as the C writes it.
inline fn absMax4(v: [4]cpu.quants.arch.avx2.m256) cpu.quants.arch.avx2.m256 {
    const signBit = cpu.quants.arch.avx2.mm256_set1_ps(-0.0);
    var maxAbs = cpu.quants.arch.avx2.mm256_andnot_ps(signBit, v[0]);
    maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, cpu.quants.arch.avx2.mm256_andnot_ps(signBit, v[1]));
    maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, cpu.quants.arch.avx2.mm256_andnot_ps(signBit, v[2]));
    maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, cpu.quants.arch.avx2.mm256_andnot_ps(signBit, v[3]));
    return maxAbs;
}

inline fn reduceMax(maxAbs: cpu.quants.arch.avx2.m256) f32 {
    var max4 = cpu.quants.arch.avx2.mm_max_ps(cpu.quants.arch.avx2.mm256_extractf128_ps(maxAbs, 1), cpu.quants.arch.avx2.mm256_castps256_ps128(maxAbs));
    max4 = cpu.quants.arch.avx2.mm_max_ps(max4, cpu.quants.arch.avx2.mm_movehl_ps(max4, max4));
    max4 = cpu.quants.arch.avx2.mm_max_ss(max4, cpu.quants.arch.avx2.mm_movehdup_ps(max4));
    return cpu.quants.arch.avx2.mm_cvtss_f32(max4);
}

/// Ports `ggml_quantize_mat_q8_0_4x8` (arch/x86/repack.cpp:178 @b1b770d90),
/// the `__AVX2__ || __AVX__` arm with its `__AVX2__` pack.
pub fn ggml_quantize_mat_q8_0_4x8(x: [*]const f32, vy: *anyopaque, k: i64) callconv(.c) void {
    std.debug.assert(@rem(k, QK8_0) == 0);
    const nb: usize = @intCast(@divTrunc(k, QK8_0));
    const ku: usize = @intCast(k);

    const y: [*]cpu.repack.blocks.block_q8_0x4 = @ptrCast(@alignCast(vy));

    var srcv: [4][4]cpu.quants.arch.avx2.m256 = undefined;
    var idvec: [4]cpu.quants.arch.avx2.m256 = undefined;

    for (0..nb) |i| {
        for (0..4) |row_iter| {
            // Load elements into 4 AVX vectors
            const base = x + row_iter * ku + i * 32;
            const v = [4]cpu.quants.arch.avx2.m256{
                cpu.quants.arch.avx2.mm256_loadu_ps(base),
                cpu.quants.arch.avx2.mm256_loadu_ps(base + 8),
                cpu.quants.arch.avx2.mm256_loadu_ps(base + 16),
                cpu.quants.arch.avx2.mm256_loadu_ps(base + 24),
            };

            const maxScalar = reduceMax(absMax4(v));

            // Divided by 127.f to mirror results in quantize_row_q8_0
            const d = maxScalar / 127.0;
            const id: f32 = if (maxScalar != 0.0) 127.0 / maxScalar else 0.0;

            // Store the scale for the individual block
            y[i].d[row_iter] = cpu.convert.cpuFp32ToFp16(d);

            srcv[row_iter] = v;
            idvec[row_iter] = cpu.quants.arch.avx2.mm256_set1_ps(id);
        }

        // Four chunks of eight bytes from each row's block, interleaved.
        for (0..4) |j| {
            const q = packBytes(
                cpu.quants.arch.avx2.mm256_mul_ps(srcv[0][j], idvec[0]),
                cpu.quants.arch.avx2.mm256_mul_ps(srcv[1][j], idvec[1]),
                cpu.quants.arch.avx2.mm256_mul_ps(srcv[2][j], idvec[2]),
                cpu.quants.arch.avx2.mm256_mul_ps(srcv[3][j], idvec[3]),
            );
            cpu.quants.arch.avx2.mm256_storeu_si256(y[i].qs[32 * j ..].ptr, q);
        }
    }
}

/// Ports `ggml_quantize_mat_q8_K_4x8` (arch/x86/repack.cpp:290 @b1b770d90),
/// the `__AVX2__` arm.
///
/// The scale's sign follows the element of largest magnitude: `maskAbs`
/// marks every lane that held the running maximum as a positive value, and
/// a lane still marked at the end means the maximum was positive -- so
/// `iscale` is negated, as `quantize_row_q8_K_ref`'s `-127 / max` does with
/// the signed max. The `bsums` are summed from the interleaved quants with
/// shuffles and blends that pick two values from each of four sub-blocks at
/// a time.
pub fn ggml_quantize_mat_q8_K_4x8(x: [*]const f32, vy: *anyopaque, k: i64) callconv(.c) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divTrunc(k, QK_K));
    const ku: usize = @intCast(k);

    const y: [*]cpu.repack.blocks.block_q8_Kx4 = @ptrCast(@alignCast(vy));

    var iscale: [4]f32 = undefined;
    var srcv: [4][32]cpu.quants.arch.avx2.m256 = undefined;
    var iscale_vec: [4]cpu.quants.arch.avx2.m256 = undefined;

    for (0..nb) |i| {
        for (0..4) |row_iter| {
            const signBit = cpu.quants.arch.avx2.mm256_set1_ps(-0.0);

            const base = x + row_iter * ku + i * 256;
            const v0 = cpu.quants.arch.avx2.mm256_loadu_ps(base);
            const v1 = cpu.quants.arch.avx2.mm256_loadu_ps(base + 8);
            const v2 = cpu.quants.arch.avx2.mm256_loadu_ps(base + 16);
            const v3 = cpu.quants.arch.avx2.mm256_loadu_ps(base + 24);

            // Compute max(abs(e)) for the block
            const abs0 = cpu.quants.arch.avx2.mm256_andnot_ps(signBit, v0);
            const abs1 = cpu.quants.arch.avx2.mm256_andnot_ps(signBit, v1);
            const abs2 = cpu.quants.arch.avx2.mm256_andnot_ps(signBit, v2);
            const abs3 = cpu.quants.arch.avx2.mm256_andnot_ps(signBit, v3);

            var maxAbs = cpu.quants.arch.avx2.mm256_max_ps(abs0, abs1);
            maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, abs2);
            maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, abs3);

            var mask0 = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxAbs, v0);
            var mask1 = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxAbs, v1);
            var mask2 = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxAbs, v2);
            var mask3 = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxAbs, v3);

            var maskAbs = cpu.quants.arch.avx2.mm256_or_ps(cpu.quants.arch.avx2.mm256_or_ps(mask0, mask1), cpu.quants.arch.avx2.mm256_or_ps(mask2, mask3));

            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;

            for (1..8) |sb| {
                // Temporarily stores absolute quant values
                const tempAbs = maxAbs;

                const b = base + sb * 32;
                const w0 = cpu.quants.arch.avx2.mm256_loadu_ps(b);
                const w1 = cpu.quants.arch.avx2.mm256_loadu_ps(b + 8);
                const w2 = cpu.quants.arch.avx2.mm256_loadu_ps(b + 16);
                const w3 = cpu.quants.arch.avx2.mm256_loadu_ps(b + 24);

                maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, cpu.quants.arch.avx2.mm256_andnot_ps(signBit, w0));
                maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, cpu.quants.arch.avx2.mm256_andnot_ps(signBit, w1));
                maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, cpu.quants.arch.avx2.mm256_andnot_ps(signBit, w2));
                maxAbs = cpu.quants.arch.avx2.mm256_max_ps(maxAbs, cpu.quants.arch.avx2.mm256_andnot_ps(signBit, w3));

                const mask_prev = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(tempAbs, maxAbs);
                maskAbs = cpu.quants.arch.avx2.mm256_and_ps(maskAbs, mask_prev);

                mask0 = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxAbs, w0);
                mask1 = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxAbs, w1);
                mask2 = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxAbs, w2);
                mask3 = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxAbs, w3);

                const mask_curr = cpu.quants.arch.avx2.mm256_or_ps(cpu.quants.arch.avx2.mm256_or_ps(mask0, mask1), cpu.quants.arch.avx2.mm256_or_ps(mask2, mask3));
                maskAbs = cpu.quants.arch.avx2.mm256_or_ps(maskAbs, mask_curr);

                srcv[row_iter][sb * 4] = w0;
                srcv[row_iter][sb * 4 + 1] = w1;
                srcv[row_iter][sb * 4 + 2] = w2;
                srcv[row_iter][sb * 4 + 3] = w3;
            }

            const maxScalar = reduceMax(maxAbs);

            const maxScalarVec = cpu.quants.arch.avx2.mm256_set1_ps(maxScalar);

            const mask_next = cpu.quants.arch.avx2.mm256_cmpeq_oq_ps(maxScalarVec, maxAbs);
            const finalMask = cpu.quants.arch.avx2.mm256_and_ps(maskAbs, mask_next);

            const mask = cpu.quants.arch.avx2.mm256_movemask_ps(finalMask);
            iscale[row_iter] = if (maxScalar != 0.0) 127.0 / maxScalar else 0.0;

            if (mask != 0) {
                iscale[row_iter] = if (maxScalar != 0.0) -127.0 / maxScalar else 0.0;
            }

            y[i].d[row_iter] = if (maxScalar != 0) 1 / iscale[row_iter] else 0;
            iscale_vec[row_iter] = cpu.quants.arch.avx2.mm256_set1_ps(iscale[row_iter]);
        }

        var quants_interleaved: [32]cpu.quants.arch.avx2.m256i = undefined;
        for (0..32) |j| {
            const q = packBytes(
                cpu.quants.arch.avx2.mm256_mul_ps(srcv[0][j], iscale_vec[0]),
                cpu.quants.arch.avx2.mm256_mul_ps(srcv[1][j], iscale_vec[1]),
                cpu.quants.arch.avx2.mm256_mul_ps(srcv[2][j], iscale_vec[2]),
                cpu.quants.arch.avx2.mm256_mul_ps(srcv[3][j], iscale_vec[3]),
            );
            cpu.quants.arch.avx2.mm256_storeu_si256(y[i].qs[32 * j ..].ptr, q);
            quants_interleaved[j] = q;
        }

        // Masks to shuffle the quants of corresponding sub blocks for
        // rearranging quants for vectorized bsums computation
        var shuffle_mask_sb2 = cpu.quants.arch.avx2.mm256_castsi128_si256(cpu.quants.arch.avx2.mm_setr_epi8(.{ 0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15 }));
        shuffle_mask_sb2 = cpu.quants.arch.avx2.mm256_permute2f128_si256(shuffle_mask_sb2, shuffle_mask_sb2, 0);
        var shuffle_mask_sb3 = cpu.quants.arch.avx2.mm256_castsi128_si256(cpu.quants.arch.avx2.mm_setr_epi8(.{ 0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15 }));
        shuffle_mask_sb3 = cpu.quants.arch.avx2.mm256_permute2f128_si256(shuffle_mask_sb3, shuffle_mask_sb3, 0);
        var shuffle_mask_sb4 = cpu.quants.arch.avx2.mm256_castsi128_si256(cpu.quants.arch.avx2.mm_setr_epi8(.{ 0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9 }));
        shuffle_mask_sb4 = cpu.quants.arch.avx2.mm256_permute2f128_si256(shuffle_mask_sb4, shuffle_mask_sb4, 0);

        const one = cpu.quants.arch.avx2.mm256_set1_epi8(1);

        for (0..4) |kk| {
            // Quants from four different sub blocks are taken
            var q: [8]cpu.quants.arch.avx2.m256i = undefined;
            inline for (0..8) |m| q[m] = quants_interleaved[kk * 8 + m];

            var bsums_r: [2]cpu.quants.arch.avx2.m256i = undefined;
            // The first half of each sub-block (q0, q2, q4, q6), then the
            // second (q1, q3, q5, q7), two values from each at a time.
            inline for (0..2) |h| {
                var interleaved = blend4(q[h], q[h + 2], q[h + 4], q[h + 6], shuffle_mask_sb2, shuffle_mask_sb3, shuffle_mask_sb4);
                var acc = cpu.quants.arch.avx2.mm256_maddubs_epi16(one, interleaved);

                for (0..3) |_| {
                    // Quants value shifted to process next two values from each sub block
                    q[h] = cpu.quants.arch.avx2.mm256_srli_epi64(q[h], 16);
                    q[h + 2] = cpu.quants.arch.avx2.mm256_srli_epi64(q[h + 2], 16);
                    q[h + 4] = cpu.quants.arch.avx2.mm256_srli_epi64(q[h + 4], 16);
                    q[h + 6] = cpu.quants.arch.avx2.mm256_srli_epi64(q[h + 6], 16);

                    interleaved = blend4(q[h], q[h + 2], q[h + 4], q[h + 6], shuffle_mask_sb2, shuffle_mask_sb3, shuffle_mask_sb4);
                    acc = cpu.quants.arch.avx2.mm256_add_epi16(acc, cpu.quants.arch.avx2.mm256_maddubs_epi16(one, interleaved));
                }
                bsums_r[h] = acc;
            }

            // Overall bsums in interleaved fashion computed by adding results of both halves
            const bsums = cpu.quants.arch.avx2.mm256_add_epi16(bsums_r[0], bsums_r[1]);
            cpu.quants.arch.avx2.mm256_storeu_si256(y[i].bsums[16 * kk ..].ptr, bsums);
        }
    }
}

/// The `bsums` step of `ggml_quantize_mat_q8_K_4x8` (arch/x86/repack.cpp:290
/// @b1b770d90): two values from each of four sub-blocks into one register,
/// the first as is and the other three shuffled into the second, third and
/// fourth sixteen-bit lane of each group and blended in.
inline fn blend4(a: cpu.quants.arch.avx2.m256i, b: cpu.quants.arch.avx2.m256i, c: cpu.quants.arch.avx2.m256i, d: cpu.quants.arch.avx2.m256i, sb2: cpu.quants.arch.avx2.m256i, sb3: cpu.quants.arch.avx2.m256i, sb4: cpu.quants.arch.avx2.m256i) cpu.quants.arch.avx2.m256i {
    var r = cpu.quants.arch.avx2.mm256_blend_epi16(a, cpu.quants.arch.avx2.mm256_shuffle_epi8(b, sb2), 34);
    r = cpu.quants.arch.avx2.mm256_blend_epi16(r, cpu.quants.arch.avx2.mm256_shuffle_epi8(c, sb3), 68);
    return cpu.quants.arch.avx2.mm256_blend_epi16(r, cpu.quants.arch.avx2.mm256_shuffle_epi8(d, sb4), 136);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
