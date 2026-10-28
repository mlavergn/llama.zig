//! The AVX2 gemv and gemm for `q4_K` interleaved eight columns at a time.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` at v0.4.0
//! (`5266f24da`): `ggml_gemv_q4_K_8x8_q8_K` and `ggml_gemm_q4_K_8x8_q8_K`,
//! their `__AVX2__` arms, and the gemm's `__AVX512BW__ && __AVX512DQ__` arm,
//! which is the same body at 512 bits over two blocks (`wide.zig`).
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
const wide = @import("wide.zig");

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
/// sub-block (arch/x86/repack.cpp:1464 @5266f24da): four pairs of column
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

/// Ports `ggml_gemv_q4_K_8x8_q8_K` (arch/x86/repack.cpp:1464 @5266f24da), the
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
/// both shuffle patterns, their scales and mins -- at 512 bits, two blocks'
/// sixteen columns, the second block in the high half.
fn Rhs(comptime L: type) type {
    return struct {
        /// `[s][k]`: sub-block `s` of the pair, weights `8k..8k+7`.
        m0145_sp1: [2][4]L.I,
        m2367_sp1: [2][4]L.I,
        m0145_sp2: [2][4]L.I,
        m2367_sp2: [2][4]L.I,
        scale_0145: [2]L.I,
        scale_2367: [2]L.I,
        mins_01: L.I,
    };
}

/// One block's 256-bit inputs to the gemm's per-`sb` column preparation:
/// the raw nibbles in the formats `B0B1B4B5` and `B2B3B6B7`, and the two
/// sub-blocks' scales and mins widened to sixteen bits.
const RawRhs = struct {
    raw0145: [4]m256i,
    raw2367: [4]m256i,
    scales_0: m256i,
    scales_1: m256i,
    mins_01: m256i,
};

fn rawRhs(b: *const blocks.block_q4_Kx8, sb: usize) RawRhs {
    // Permute mask used for easier vector processing at later stages
    const requiredOrder = avx.mm256_set_epi32(.{ 3, 2, 1, 0, 7, 6, 5, 4 });

    const qs = b.qs[sb * 256 ..].ptr;
    var r: RawRhs = undefined;
    inline for (0..4) |k| {
        const raw0123 = avx.mm256_loadu_si256(qs + 64 * k);
        const raw4567 = avx.mm256_loadu_si256(qs + 64 * k + 32);
        // In the formats B0B1B4B5, B2B3B6B7
        r.raw0145[k] = avx.mm256_blend_epi32(raw0123, avx.mm256_permutevar8x32_epi32(raw4567, requiredOrder), 240);
        r.raw2367[k] = avx.mm256_blend_epi32(avx.mm256_permutevar8x32_epi32(raw0123, requiredOrder), raw4567, 240);
    }

    const mins_and_scales_0 = minsAndScales(&b.scales, 24 * sb);
    const mins_and_scales_1 = minsAndScales(&b.scales, 12 + sb * 24);

    r.scales_0 = avx.mm256_cvtepu8_epi16(avx.mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));
    r.scales_1 = avx.mm256_cvtepu8_epi16(avx.mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

    r.mins_01 = avx.mm256_cvtepu8_epi16(avx.mm_unpacklo_epi8(avx.mm_shuffle_epi32(mins_and_scales_0, 78), avx.mm_shuffle_epi32(mins_and_scales_1, 78)));
    return r;
}

/// The low or high nibbles of a register at `L`'s width.
inline fn nibblesWide(comptime L: type, raw: L.I, comptime hi: bool) L.I {
    const m4b = L.set1_epi8(0x0F);
    return L.and_si(if (hi) L.srli_epi16(raw, 4) else raw, m4b);
}

/// The gemm's per-`sb` column preparation (arch/x86/repack.cpp:2853
/// @5266f24da) -- the 512-bit arm's opens at line 2117 -- for `b0` and, at
/// 512 bits, `b1` in the high half.
///
/// The 512-bit arm builds its scales as `_mm512_cvtepu8_epi16` of one
/// `_mm256_unpacklo_epi8` over both blocks' `utmp` and its mins from
/// `_mm256_shuffle_epi32(.., 78)` of each: per 128-bit lane, each block's
/// AVX2 value. So both are `join`s of `rawRhs`.
fn prepareRhs(comptime L: type, b0: *const blocks.block_q4_Kx8, b1: *const blocks.block_q4_Kx8, sb: usize) Rhs(L) {
    const r0 = rawRhs(b0, sb);
    const r1 = if (L.width == 512) rawRhs(b1, sb) else r0;
    var r: Rhs(L) = undefined;
    inline for (0..4) |k| {
        const raw0145 = L.join(r0.raw0145[k], r1.raw0145[k]);
        const raw2367 = L.join(r0.raw2367[k], r1.raw2367[k]);
        inline for (0..2) |sub| {
            const m0145 = nibblesWide(L, raw0145, sub == 1);
            const m2367 = nibblesWide(L, raw2367, sub == 1);
            r.m0145_sp1[sub][k] = L.shuffle_epi32(m0145, 136);
            r.m2367_sp1[sub][k] = L.shuffle_epi32(m2367, 136);
            r.m0145_sp2[sub][k] = L.shuffle_epi32(m0145, 221);
            r.m2367_sp2[sub][k] = L.shuffle_epi32(m2367, 221);
        }
    }

    const scales_0 = L.join(r0.scales_0, r1.scales_0);
    const scales_1 = L.join(r0.scales_1, r1.scales_1);

    r.mins_01 = L.join(r0.mins_01, r1.mins_01);

    r.scale_0145 = .{ L.shuffle_epi32(scales_0, 68), L.shuffle_epi32(scales_1, 68) };
    r.scale_2367 = .{ L.shuffle_epi32(scales_0, 238), L.shuffle_epi32(scales_1, 238) };
    return r;
}

/// Four `maddubs`, columns unsigned first, summed in sixteen bits from the
/// fourth weight group down, as the C nests them.
fn dot4(comptime L: type, rhs: [4]L.I, lhs: [4]L.I) L.I {
    return L.add_epi16(
        L.add_epi16(
            L.add_epi16(L.maddubs_epi16(rhs[3], lhs[3]), L.maddubs_epi16(rhs[2], lhs[2])),
            L.maddubs_epi16(rhs[1], lhs[1]),
        ),
        L.maddubs_epi16(rhs[0], lhs[0]),
    );
}

/// Two 2x2 blocks of row-by-column sums straightened into four rows.
inline fn rows(comptime L: type, m00: L.I, m01: L.I, m10: L.I, m11: L.I) [4]L.I {
    return .{
        L.blend_epi32(m00, L.shuffle_epi32(m01, 78), 204),
        L.blend_epi32(L.shuffle_epi32(m00, 78), m01, 204),
        L.blend_epi32(m10, L.shuffle_epi32(m11, 78), 204),
        L.blend_epi32(L.shuffle_epi32(m10, 78), m11, 204),
    };
}

/// The per-group body of `ggml_gemm_q4_K_8x8_q8_K` (arch/x86/repack.cpp:2042
/// @5266f24da), either arm: one interleaved block of four activation rows
/// against one prepared sub-block pair, into four row accumulators and four
/// minima accumulators.
fn accumulate4(comptime L: type, acc: *[4]L.F, acc_min: *[4]L.F, rhs: *const Rhs(L), col_scale_f32: L.F, col_dmin_f32: L.F, a: *const blocks.block_q8_Kx4, sb: usize) void {
    const qs = a.qs[256 * sb ..].ptr;
    // `[sub][k]`, the activations of sub-block `sub`, weights `8k..8k+7`
    var lhs_01_sp1: [2][4]L.I = undefined;
    var lhs_23_sp1: [2][4]L.I = undefined;
    var lhs_01_sp2: [2][4]L.I = undefined;
    var lhs_23_sp2: [2][4]L.I = undefined;
    inline for (0..2) |sub| {
        inline for (0..4) |k| {
            const lhs_mat_0123 = avx.mm256_loadu_si256(qs + 128 * sub + 32 * k);
            const lhs_mat_01 = L.rep(avx.mm256_permute2f128_si256(lhs_mat_0123, lhs_mat_0123, 0));
            const lhs_mat_23 = L.rep(avx.mm256_permute2f128_si256(lhs_mat_0123, lhs_mat_0123, 17));
            lhs_01_sp1[sub][k] = L.shuffle_epi32(lhs_mat_01, 160);
            lhs_23_sp1[sub][k] = L.shuffle_epi32(lhs_mat_23, 160);
            lhs_01_sp2[sub][k] = L.shuffle_epi32(lhs_mat_01, 245);
            lhs_23_sp2[sub][k] = L.shuffle_epi32(lhs_mat_23, 245);
        }
    }

    // The four rows' bsums for this sub-block pair, summed pairwise
    const lhs_bsums_0123_01 = avx.mm256_loadu_si256(a.bsums[16 * sb ..].ptr);
    const lhs_bsums_hsum_0123_01 = L.rep(common.repeat128(avx.mm_hadd_epi16(avx.mm256_castsi256_si128(lhs_bsums_0123_01), avx.mm256_extractf128_si256(lhs_bsums_0123_01, 1))));

    var iacc_row: [4]L.I = undefined;
    inline for (0..2) |sub| {
        // Both shuffle patterns summed in sixteen bits, then the scales
        const iacc_mat_00 = L.madd_epi16(L.add_epi16(dot4(L, rhs.m0145_sp1[sub], lhs_01_sp1[sub]), dot4(L, rhs.m0145_sp2[sub], lhs_01_sp2[sub])), rhs.scale_0145[sub]);
        const iacc_mat_01 = L.madd_epi16(L.add_epi16(dot4(L, rhs.m2367_sp1[sub], lhs_01_sp1[sub]), dot4(L, rhs.m2367_sp2[sub], lhs_01_sp2[sub])), rhs.scale_2367[sub]);
        const iacc_mat_10 = L.madd_epi16(L.add_epi16(dot4(L, rhs.m0145_sp1[sub], lhs_23_sp1[sub]), dot4(L, rhs.m0145_sp2[sub], lhs_23_sp2[sub])), rhs.scale_0145[sub]);
        const iacc_mat_11 = L.madd_epi16(L.add_epi16(dot4(L, rhs.m2367_sp1[sub], lhs_23_sp1[sub]), dot4(L, rhs.m2367_sp2[sub], lhs_23_sp2[sub])), rhs.scale_2367[sub]);
        const r = rows(L, iacc_mat_00, iacc_mat_01, iacc_mat_10, iacc_mat_11);
        inline for (0..4) |i| iacc_row[i] = if (sub == 0) r[i] else L.add_epi32(iacc_row[i], r[i]);
    }

    // The four rows' f32 scales, repeated in every half
    const row_scale_f32_sse = avx.mm_load_ps(&a.d);
    const row_scale_f32 = L.repF(avx.mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse));

    const lanes = [4]u8{ 0, 85, 170, 255 };
    inline for (0..4) |i| {
        const rs = L.shuffle_ps(row_scale_f32, lanes[i]);
        acc[i] = L.fmadd_ps(L.cvtepi32_ps(iacc_row[i]), L.mul_ps(col_scale_f32, rs), acc[i]);
    }
    inline for (0..4) |i| {
        const iacc_row_min = L.madd_epi16(L.shuffle_epi32(lhs_bsums_hsum_0123_01, lanes[i]), rhs.mins_01);
        const rs = L.shuffle_ps(row_scale_f32, lanes[i]);
        acc_min[i] = L.fmadd_ps(L.cvtepi32_ps(iacc_row_min), L.mul_ps(col_dmin_f32, rs), acc_min[i]);
    }
}

/// A column block's `d` or `dmin` as floats: `GGML_F32Cx8_LOAD`, or
/// `GGML_F32Cx8x2_LOAD` over two blocks at 512 bits.
inline fn colLoad(comptime L: type, d0: *const [8]c.ggml_half, d1: *const [8]c.ggml_half) L.F {
    if (L.width == 256) return common.f32cx8Load(d0);
    return L.joinF(common.f32cx8Load(d0), common.f32cx8Load(d1));
}

/// Both row loops of `ggml_gemm_q4_K_8x8_q8_K` at `L`'s width, over the
/// eight-column blocks `x_begin..x_end`: sixteen rows a pass, then four.
fn rowLoops(comptime L: type, y: *usize, x_begin: usize, x_end: usize, nb: usize, b_nb: usize, nr_u: usize, s: [*]f32, bs: usize, b_ptr_start: [*]const blocks.block_q4_Kx8, a_ptr_start: [*]const blocks.block_q8_Kx4) void {
    const anr = nr_u - nr_u % 16; // Used to align nr with boundary of 16
    const xstep = L.cols / 8;

    while (y.* < anr / 4) : (y.* += 4) {
        var a_ptrs: [4][*]const blocks.block_q8_Kx4 = undefined;

        a_ptrs[0] = a_ptr_start + (y.* * nb);
        for (0..3) |i| a_ptrs[i + 1] = a_ptrs[i] + nb;

        var x = x_begin;
        while (x < x_end) : (x += xstep) {
            const b_ptr_0 = b_ptr_start + (x * b_nb);
            const b_ptr_1 = b_ptr_start + ((x + xstep - 1) * b_nb);

            var acc_rows: [16]L.F = @splat(L.setzero_ps());
            var acc_min_rows: [16]L.F = @splat(L.setzero_ps());

            for (0..nb) |b| {
                const col_scale_f32 = colLoad(L, &b_ptr_0[b].d, &b_ptr_1[b].d);
                const col_dmin_f32 = colLoad(L, &b_ptr_0[b].dmin, &b_ptr_1[b].dmin);

                for (0..QK_K / 64) |sb| {
                    const rhs = prepareRhs(L, &b_ptr_0[b], &b_ptr_1[b], sb);
                    for (0..4) |rp| accumulate4(L, acc_rows[rp * 4 ..][0..4], acc_min_rows[rp * 4 ..][0..4], &rhs, col_scale_f32, col_dmin_f32, &a_ptrs[rp][b], sb);
                }
            }
            for (0..16) |i| L.storeu_ps(s + ((y.* * 4 + i) * bs + x * 8), L.sub_ps(acc_rows[i], acc_min_rows[i]));
        }
    }
    while (y.* < nr_u / 4) : (y.* += 1) {
        const a_ptr = a_ptr_start + (y.* * nb);

        var x = x_begin;
        while (x < x_end) : (x += xstep) {
            const b_ptr_0 = b_ptr_start + (x * b_nb);
            const b_ptr_1 = b_ptr_start + ((x + xstep - 1) * b_nb);

            var acc_rows: [4]L.F = @splat(L.setzero_ps());
            var acc_min_rows: [4]L.F = @splat(L.setzero_ps());

            for (0..nb) |b| {
                const col_scale_f32 = colLoad(L, &b_ptr_0[b].d, &b_ptr_1[b].d);
                const col_dmin_f32 = colLoad(L, &b_ptr_0[b].dmin, &b_ptr_1[b].dmin);

                for (0..QK_K / 64) |sb| {
                    const rhs = prepareRhs(L, &b_ptr_0[b], &b_ptr_1[b], sb);
                    accumulate4(L, &acc_rows, &acc_min_rows, &rhs, col_scale_f32, col_dmin_f32, &a_ptr[b], sb);
                }
            }
            for (0..4) |i| L.storeu_ps(s + ((y.* * 4 + i) * bs + x * 8), L.sub_ps(acc_rows[i], acc_min_rows[i]));
        }
    }
}

/// Ports `ggml_gemm_q4_K_8x8_q8_K` (arch/x86/repack.cpp:2042 @5266f24da).
///
/// With `__AVX512BW__ && __AVX512DQ__` the 512-bit arm, from line 2077,
/// runs first over the columns up to the last multiple of
/// sixteen; the AVX2 loops then take any left, from the first row again.
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

    var xstart: usize = 0;
    const nr_u: usize = @intCast(nr);
    const nc_u: usize = @intCast(nc);

    if (comptime wide.has_avx512bw_dq) {
        const anc = nc_u - nc_u % 16; // Used to align nc with boundary of 16
        rowLoops(wide.Wide(512), &y, 0, anc / 8, nb, b_nb, nr_u, s, bs, b_ptr_start, a_ptr_start);
        if (anc != nc_u) {
            xstart = anc / 8;
            y = 0;
        }
    }

    rowLoops(wide.Wide(256), &y, xstart, nc_u / 8, nb, b_nb, nr_u, s, bs, b_ptr_start, a_ptr_start);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
