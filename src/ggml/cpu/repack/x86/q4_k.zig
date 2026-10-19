//! The AVX2 gemv and gemm for `q4_K` interleaved eight columns at a time.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` at v0.3.0
//! (`c1d0e7a00`): `ggml_gemv_q4_K_8x8_q8_K` and `ggml_gemm_q4_K_8x8_q8_K`,
//! their `__AVX2__` arms. The gemm's `__AVX512BW__ && __AVX512DQ__` arm is
//! not compiled at x86-64-v3 and not ported.
//!
//! # The weights are unsigned; the minima come in through `bsums`
//!
//! A `q4_K` nibble is `0..15`, so `maddubs` takes it directly as its
//! unsigned operand and the activation as its signed one. The products of
//! eight such calls are summed in **sixteen** bits, wrapping, before the
//! 6-bit scales widen them; the minima are `bsums` against the 6-bit mins,
//! accumulated separately and subtracted only when the row is stored.
//!
//! # The gemm's two row loops share one body
//!
//! As in `q4_b32.zig`: sixteen rows a pass, then a four-row remainder, the
//! C's per-group body written once as `accumulate4`. The two bodies were
//! diffed line by line after normalising the row index, and differ in
//! nothing else.

const std = @import("std");
const impl = @import("../../../impl.zig");
const qblocks = @import("../../../quants/blocks.zig");
const blocks = @import("../blocks.zig");
const avx = @import("../../quants/x86/avx2.zig");
const qcommon = @import("../../quants/x86/common.zig");
const common = @import("common.zig");

const c = impl.c;
const m256i = avx.m256i;
const m128i = avx.m128i;
const m256 = avx.m256;

const QK_K = 256;

/// One sub-block pair's 6-bit scales and mins, unpacked as the C's `utmp`
/// shuffle does, as the `_mm_set_epi32` the kernels build from it.
inline fn minsAndScales(scales: *const [96]u8, off: usize) m128i {
    const utmp = qcommon.unpackScales(scales[off..][0..12]);
    return avx.mm_set_epi32(@bitCast(utmp[3]), @bitCast(utmp[2]), @bitCast(utmp[1]), @bitCast(utmp[0]));
}

/// The low or high nibbles of a register.
inline fn nibbles(raw: m256i, comptime hi: bool) m256i {
    const m4b = avx.mm256_set1_epi8(0x0F);
    return avx.mm256_and_si256(if (hi) avx.mm256_srli_epi16(raw, 4) else raw, m4b);
}

/// The eight multiply-adds `ggml_gemv_q4_K_8x8_q8_K` takes over one
/// sub-block (arch/x86/repack.cpp:1464 @c1d0e7a00): four pairs of column
/// registers against an activation broadcast four bytes at a time, summed in
/// sixteen bits.
fn gemvDot(r0123: [4]m256i, r4567: [4]m256i, lhs_a: m256i, lhs_b: m256i) m256i {
    var iacc = avx.mm256_setzero_si256();
    inline for (0..4) |k| {
        const lhs = if (k < 2) lhs_a else lhs_b;
        const imm0: u8 = if (k % 2 == 0) 0 else 170;
        const imm1: u8 = if (k % 2 == 0) 85 else 255;
        iacc = avx.mm256_add_epi16(iacc, avx.mm256_maddubs_epi16(avx.mm256_blend_epi32(r0123[k], avx.mm256_shuffle_epi32(r4567[k], 177), 170), avx.mm256_shuffle_epi32(lhs, imm0)));
        iacc = avx.mm256_add_epi16(iacc, avx.mm256_maddubs_epi16(avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(r0123[k], 177), r4567[k], 170), avx.mm256_shuffle_epi32(lhs, imm1)));
    }
    return iacc;
}

/// Ports `ggml_gemv_q4_K_8x8_q8_K` (arch/x86/repack.cpp:1464 @c1d0e7a00), the
/// `__AVX2__` arm.
///
/// The row is stored as `permute(acc_row) - acc_min_rows`: the minima's
/// accumulator is already in column order and is not permuted, and the store
/// is at `y * nr`, as in the `q4_0` gemv.
pub export fn ggml_gemv_q4_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    const qk = QK_K;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(@rem(nc, 8) == 0);
    _ = bs;

    // Shuffle masks to rearrange delta and scale values to multiply with
    // appropriate scales
    const deltamask = avx.mm_set_epi8(.{ 15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0 });
    const scalemask = avx.mm_set_epi8(.{ 7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0 });
    // Permute mask used for easier vector processing at later stages
    const finalpermutemask = avx.mm256_set_epi32(.{ 7, 5, 3, 1, 6, 4, 2, 0 });

    const b_nb: usize = @intCast(@divTrunc(n, QK_K));

    const b_ptr_start: [*]const blocks.block_q4_Kx8 = @ptrCast(@alignCast(vx));
    const a_ptr_start: [*]const qblocks.Q8_K = @ptrCast(@alignCast(vy));

    // Process Q8_K blocks one by one
    for (0..@intCast(nr)) |y| {
        const a_ptr = a_ptr_start + (y * nb);

        // Eight interleaved block_q4_K at each pass of the loop
        for (0..@intCast(@divTrunc(nc, 8))) |x| {
            const b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            var acc_row = avx.mm256_setzero_ps();
            var acc_min_rows = avx.mm256_setzero_ps();

            for (0..nb) |b| {
                // Load and convert to FP32 scale from block_q8_K
                const row_scale_f32 = avx.mm256_set1_ps(a_ptr[b].d);

                // The eight columns' scales, rearranged to meet the quants
                const col_scale_f32 = common.f32cx8RearrangeLoad(&b_ptr[b].d, deltamask);
                const col_dmin_f32 = common.f32cx8Load(&b_ptr[b].dmin);

                var iacc_b = avx.mm256_setzero_si256();
                var iacc_min_b = avx.mm256_setzero_si256();

                const q8sums = avx.mm256_loadu_si256(&a_ptr[b].bsums);
                var q8s = common.repeat128(avx.mm_hadd_epi16(avx.mm256_castsi256_si128(q8sums), avx.mm256_extracti128_si256(q8sums, 1)));

                // Two sub blocks from each Q4_K in each iteration
                for (0..QK_K / 64) |sb| {
                    const qs = b_ptr[b].qs[sb * 256 ..].ptr;
                    var raw0123: [4]m256i = undefined;
                    var raw4567: [4]m256i = undefined;
                    inline for (0..4) |k| {
                        raw0123[k] = avx.mm256_loadu_si256(qs + 64 * k);
                        raw4567[k] = avx.mm256_loadu_si256(qs + 64 * k + 32);
                    }

                    // 4-bit -> 8-bit: the first sub block in the low nibbles,
                    // the second in the high
                    var r0123_0: [4]m256i = undefined;
                    var r4567_0: [4]m256i = undefined;
                    var r0123_1: [4]m256i = undefined;
                    var r4567_1: [4]m256i = undefined;
                    inline for (0..4) |k| {
                        r0123_0[k] = nibbles(raw0123[k], false);
                        r4567_0[k] = nibbles(raw4567[k], false);
                        r0123_1[k] = nibbles(raw0123[k], true);
                        r4567_1[k] = nibbles(raw4567[k], true);
                    }

                    const mins_and_scales_0 = minsAndScales(&b_ptr[b].scales, 24 * sb);
                    const mins_and_scales_1 = minsAndScales(&b_ptr[b].scales, 12 + sb * 24);

                    // Scales of the two sub blocks
                    const scales_0 = avx.mm256_cvtepu8_epi16(avx.mm_shuffle_epi8(mins_and_scales_0, scalemask));
                    const scales_1 = avx.mm256_cvtepu8_epi16(avx.mm_shuffle_epi8(mins_and_scales_1, scalemask));

                    // Mins of the two sub blocks arranged side by side
                    const mins_01 = avx.mm256_cvtepu8_epi16(avx.mm_unpacklo_epi8(avx.mm_shuffle_epi32(mins_and_scales_0, 78), avx.mm_shuffle_epi32(mins_and_scales_1, 78)));

                    // The two sub blocks of block_q8_K, 16 bytes at a time,
                    // repeated across the vector
                    const aq = a_ptr[b].qs[sb * 64 ..].ptr;
                    const lhs_vec_00 = common.repeat128(avx.mm_loadu_si128(aq));
                    const lhs_vec_01 = common.repeat128(avx.mm_loadu_si128(aq + 16));
                    const lhs_vec_10 = common.repeat128(avx.mm_loadu_si128(aq + 32));
                    const lhs_vec_11 = common.repeat128(avx.mm_loadu_si128(aq + 48));

                    const iacc_0 = avx.mm256_madd_epi16(gemvDot(r0123_0, r4567_0, lhs_vec_00, lhs_vec_01), scales_0);
                    const iacc_1 = avx.mm256_madd_epi16(gemvDot(r0123_1, r4567_1, lhs_vec_10, lhs_vec_11), scales_1);

                    // Accumulate the iacc value for one sb
                    const iacc_sb = avx.mm256_add_epi32(iacc_0, iacc_1);

                    // This iteration's two bsums against the mins
                    const q8s_sb = avx.mm256_shuffle_epi32(q8s, 0);
                    const iacc_min_sb = avx.mm256_madd_epi16(q8s_sb, mins_01);
                    q8s = avx.mm256_bsrli_epi128(q8s, 4);

                    // Accumulate for the complete block
                    iacc_b = avx.mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = avx.mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                // Multiply-Add with scale values for the complete super block
                acc_row = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc_b), avx.mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc_min_b), avx.mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
            }

            // Accumulated output values permuted into storage order
            acc_row = avx.mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            avx.mm256_storeu_ps(s + (y * @as(usize, @intCast(nr)) + x * 8), avx.mm256_sub_ps(acc_row, acc_min_rows));
        }
    }
}

/// One super-block's sub-block pair for the gemm: eight columns' nibbles in
/// both shuffle patterns, their scales and mins.
const Rhs = struct {
    /// `[s][k]`: sub-block `s` of the pair, weights `8k..8k+7`.
    m0145_sp1: [2][4]m256i,
    m2367_sp1: [2][4]m256i,
    m0145_sp2: [2][4]m256i,
    m2367_sp2: [2][4]m256i,
    scale_0145: [2]m256i,
    scale_2367: [2]m256i,
    mins_01: m256i,
};

/// The gemm's per-`sb` column preparation (arch/x86/repack.cpp:2853 @c1d0e7a00).
fn prepareRhs(b: *const blocks.block_q4_Kx8, sb: usize) Rhs {
    // Permute mask used for easier vector processing at later stages
    const requiredOrder = avx.mm256_set_epi32(.{ 3, 2, 1, 0, 7, 6, 5, 4 });

    const qs = b.qs[sb * 256 ..].ptr;
    var r: Rhs = undefined;
    inline for (0..4) |k| {
        const raw0123 = avx.mm256_loadu_si256(qs + 64 * k);
        const raw4567 = avx.mm256_loadu_si256(qs + 64 * k + 32);
        // In the formats B0B1B4B5, B2B3B6B7
        const raw0145 = avx.mm256_blend_epi32(raw0123, avx.mm256_permutevar8x32_epi32(raw4567, requiredOrder), 240);
        const raw2367 = avx.mm256_blend_epi32(avx.mm256_permutevar8x32_epi32(raw0123, requiredOrder), raw4567, 240);
        inline for (0..2) |sub| {
            const m0145 = nibbles(raw0145, sub == 1);
            const m2367 = nibbles(raw2367, sub == 1);
            r.m0145_sp1[sub][k] = avx.mm256_shuffle_epi32(m0145, 136);
            r.m2367_sp1[sub][k] = avx.mm256_shuffle_epi32(m2367, 136);
            r.m0145_sp2[sub][k] = avx.mm256_shuffle_epi32(m0145, 221);
            r.m2367_sp2[sub][k] = avx.mm256_shuffle_epi32(m2367, 221);
        }
    }

    const mins_and_scales_0 = minsAndScales(&b.scales, 24 * sb);
    const mins_and_scales_1 = minsAndScales(&b.scales, 12 + sb * 24);

    const scales_0 = avx.mm256_cvtepu8_epi16(avx.mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));
    const scales_1 = avx.mm256_cvtepu8_epi16(avx.mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

    r.mins_01 = avx.mm256_cvtepu8_epi16(avx.mm_unpacklo_epi8(avx.mm_shuffle_epi32(mins_and_scales_0, 78), avx.mm_shuffle_epi32(mins_and_scales_1, 78)));

    r.scale_0145 = .{ avx.mm256_shuffle_epi32(scales_0, 68), avx.mm256_shuffle_epi32(scales_1, 68) };
    r.scale_2367 = .{ avx.mm256_shuffle_epi32(scales_0, 238), avx.mm256_shuffle_epi32(scales_1, 238) };
    return r;
}

/// Four `maddubs`, columns unsigned first, summed in sixteen bits from the
/// fourth weight group down, as the C nests them.
fn dot4(rhs: [4]m256i, lhs: [4]m256i) m256i {
    return avx.mm256_add_epi16(
        avx.mm256_add_epi16(
            avx.mm256_add_epi16(avx.mm256_maddubs_epi16(rhs[3], lhs[3]), avx.mm256_maddubs_epi16(rhs[2], lhs[2])),
            avx.mm256_maddubs_epi16(rhs[1], lhs[1]),
        ),
        avx.mm256_maddubs_epi16(rhs[0], lhs[0]),
    );
}

/// Two 2x2 blocks of row-by-column sums straightened into four rows.
inline fn rows(m00: m256i, m01: m256i, m10: m256i, m11: m256i) [4]m256i {
    return .{
        avx.mm256_blend_epi32(m00, avx.mm256_shuffle_epi32(m01, 78), 204),
        avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(m00, 78), m01, 204),
        avx.mm256_blend_epi32(m10, avx.mm256_shuffle_epi32(m11, 78), 204),
        avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(m10, 78), m11, 204),
    };
}

/// The per-group body of `ggml_gemm_q4_K_8x8_q8_K` (arch/x86/repack.cpp:2042
/// @c1d0e7a00): one interleaved block of four activation rows against one
/// prepared sub-block pair, into four row accumulators and four minima
/// accumulators.
fn accumulate4(acc: *[4]m256, acc_min: *[4]m256, rhs: *const Rhs, col_scale_f32: m256, col_dmin_f32: m256, a: *const blocks.block_q8_Kx4, sb: usize) void {
    const qs = a.qs[256 * sb ..].ptr;
    // `[sub][k]`, the activations of sub-block `sub`, weights `8k..8k+7`
    var lhs_01_sp1: [2][4]m256i = undefined;
    var lhs_23_sp1: [2][4]m256i = undefined;
    var lhs_01_sp2: [2][4]m256i = undefined;
    var lhs_23_sp2: [2][4]m256i = undefined;
    inline for (0..2) |sub| {
        inline for (0..4) |k| {
            const lhs_mat_0123 = avx.mm256_loadu_si256(qs + 128 * sub + 32 * k);
            const lhs_mat_01 = avx.mm256_permute2f128_si256(lhs_mat_0123, lhs_mat_0123, 0);
            const lhs_mat_23 = avx.mm256_permute2f128_si256(lhs_mat_0123, lhs_mat_0123, 17);
            lhs_01_sp1[sub][k] = avx.mm256_shuffle_epi32(lhs_mat_01, 160);
            lhs_23_sp1[sub][k] = avx.mm256_shuffle_epi32(lhs_mat_23, 160);
            lhs_01_sp2[sub][k] = avx.mm256_shuffle_epi32(lhs_mat_01, 245);
            lhs_23_sp2[sub][k] = avx.mm256_shuffle_epi32(lhs_mat_23, 245);
        }
    }

    // The four rows' bsums for this sub-block pair, summed pairwise
    const lhs_bsums_0123_01 = avx.mm256_loadu_si256(a.bsums[16 * sb ..].ptr);
    const lhs_bsums_hsum_0123_01 = common.repeat128(avx.mm_hadd_epi16(avx.mm256_castsi256_si128(lhs_bsums_0123_01), avx.mm256_extractf128_si256(lhs_bsums_0123_01, 1)));

    var iacc_row: [4]m256i = undefined;
    inline for (0..2) |sub| {
        // Both shuffle patterns summed in sixteen bits, then the scales
        const iacc_mat_00 = avx.mm256_madd_epi16(avx.mm256_add_epi16(dot4(rhs.m0145_sp1[sub], lhs_01_sp1[sub]), dot4(rhs.m0145_sp2[sub], lhs_01_sp2[sub])), rhs.scale_0145[sub]);
        const iacc_mat_01 = avx.mm256_madd_epi16(avx.mm256_add_epi16(dot4(rhs.m2367_sp1[sub], lhs_01_sp1[sub]), dot4(rhs.m2367_sp2[sub], lhs_01_sp2[sub])), rhs.scale_2367[sub]);
        const iacc_mat_10 = avx.mm256_madd_epi16(avx.mm256_add_epi16(dot4(rhs.m0145_sp1[sub], lhs_23_sp1[sub]), dot4(rhs.m0145_sp2[sub], lhs_23_sp2[sub])), rhs.scale_0145[sub]);
        const iacc_mat_11 = avx.mm256_madd_epi16(avx.mm256_add_epi16(dot4(rhs.m2367_sp1[sub], lhs_23_sp1[sub]), dot4(rhs.m2367_sp2[sub], lhs_23_sp2[sub])), rhs.scale_2367[sub]);
        const r = rows(iacc_mat_00, iacc_mat_01, iacc_mat_10, iacc_mat_11);
        inline for (0..4) |i| iacc_row[i] = if (sub == 0) r[i] else avx.mm256_add_epi32(iacc_row[i], r[i]);
    }

    // The four rows' f32 scales, repeated in both halves
    const row_scale_f32_sse = avx.mm_load_ps(&a.d);
    const row_scale_f32 = avx.mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);

    const lanes = [4]u8{ 0, 85, 170, 255 };
    inline for (0..4) |i| {
        const rs = avx.mm256_shuffle_ps(row_scale_f32, row_scale_f32, lanes[i]);
        acc[i] = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc_row[i]), avx.mm256_mul_ps(col_scale_f32, rs), acc[i]);
    }
    inline for (0..4) |i| {
        const iacc_row_min = avx.mm256_madd_epi16(avx.mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, lanes[i]), rhs.mins_01);
        const rs = avx.mm256_shuffle_ps(row_scale_f32, row_scale_f32, lanes[i]);
        acc_min[i] = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc_row_min), avx.mm256_mul_ps(col_dmin_f32, rs), acc_min[i]);
    }
}

/// Ports `ggml_gemm_q4_K_8x8_q8_K` (arch/x86/repack.cpp:2042 @c1d0e7a00), the
/// arm without `__AVX512BW__ && __AVX512DQ__`.
pub export fn ggml_gemm_q4_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    const qk = QK_K;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(@rem(nr, 4) == 0);
    std.debug.assert(@rem(nc, 8) == 0);

    const b_ptr_start: [*]const blocks.block_q4_Kx8 = @ptrCast(@alignCast(vx));
    const a_ptr_start: [*]const blocks.block_q8_Kx4 = @ptrCast(@alignCast(vy));
    const b_nb: usize = @intCast(@divTrunc(n, QK_K));
    var y: usize = 0;

    const xstart: usize = 0;
    const nr_u: usize = @intCast(nr);
    const anr = nr_u - nr_u % 16; // Used to align nr with boundary of 16
    const ncb: usize = @intCast(@divTrunc(nc, 8));

    while (y < anr / 4) : (y += 4) {
        var a_ptrs: [4][*]const blocks.block_q8_Kx4 = undefined;

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (0..3) |i| a_ptrs[i + 1] = a_ptrs[i] + nb;

        for (xstart..ncb) |x| {
            const b_ptr = b_ptr_start + (x * b_nb);

            var acc_rows: [16]m256 = @splat(avx.mm256_setzero_ps());
            var acc_min_rows: [16]m256 = @splat(avx.mm256_setzero_ps());

            for (0..nb) |b| {
                const col_scale_f32 = common.f32cx8Load(&b_ptr[b].d);
                const col_dmin_f32 = common.f32cx8Load(&b_ptr[b].dmin);

                for (0..QK_K / 64) |sb| {
                    const rhs = prepareRhs(&b_ptr[b], sb);
                    for (0..4) |rp| accumulate4(acc_rows[rp * 4 ..][0..4], acc_min_rows[rp * 4 ..][0..4], &rhs, col_scale_f32, col_dmin_f32, &a_ptrs[rp][b], sb);
                }
            }
            for (0..16) |i| avx.mm256_storeu_ps(s + ((y * 4 + i) * bs + x * 8), avx.mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
        }
    }
    while (y < nr_u / 4) : (y += 1) {
        const a_ptr = a_ptr_start + (y * nb);

        for (xstart..ncb) |x| {
            const b_ptr = b_ptr_start + (x * b_nb);

            var acc_rows: [4]m256 = @splat(avx.mm256_setzero_ps());
            var acc_min_rows: [4]m256 = @splat(avx.mm256_setzero_ps());

            for (0..nb) |b| {
                const col_scale_f32 = common.f32cx8Load(&b_ptr[b].d);
                const col_dmin_f32 = common.f32cx8Load(&b_ptr[b].dmin);

                for (0..QK_K / 64) |sb| {
                    const rhs = prepareRhs(&b_ptr[b], sb);
                    accumulate4(&acc_rows, &acc_min_rows, &rhs, col_scale_f32, col_dmin_f32, &a_ptr[b], sb);
                }
            }
            for (0..4) |i| avx.mm256_storeu_ps(s + ((y * 4 + i) * bs + x * 8), avx.mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
