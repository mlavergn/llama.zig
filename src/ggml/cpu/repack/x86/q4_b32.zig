//! The AVX2 gemv and gemm for the three 4-bit, 32-weight-block formats that
//! interleave eight columns: `q4_0`, `iq4_nl` and `mxfp4`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` at v0.3.0
//! (`c1d0e7a00`): the templates `gemv_q4_b32_8x8_q8_0_lut_avx` and
//! `gemm_q4_b32_8x8_q8_0_lut_avx` and the six entry points that instantiate
//! them. Each declaration names the C++ it replaces and the line it began
//! at. The gemm's `__AVX512BW__ && __AVX512DQ__` arm, 433 lines, is not
//! compiled at x86-64-v3 and not ported.
//!
//! # One template, three tables
//!
//! The formats differ in two things only: the 16-entry table a nibble is
//! looked up in -- passed in as `signextendlut`, a `pshufb` control -- and
//! where a column's scale comes from, an fp16 `d` or an E8M0 `e`.
//!
//! # The gemm's two row loops share one body
//!
//! The C writes the same per-block body twice: once for sixteen rows a pass
//! (four `block_q8_0x4`), once for the four-row remainder. Here it is
//! `accumulate4`, called four times or once; every float operation and its
//! order within a row is the C's.
//!
//! # The gemv stores at `y * nr`
//!
//! `_mm256_storeu_ps(s + (y * nr + x * 8), acc_row)`: the row stride is `nr`
//! where every other kernel uses `bs`. With `nr == 1`, which is how the
//! dispatch calls a gemv, the two agree; the port keeps the C's expression.

const std = @import("std");
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const qblocks = @import("../../../quants/blocks.zig");
const blocks = @import("../blocks.zig");
const avx = @import("../../quants/x86/avx2.zig");
const common = @import("common.zig");

const c = impl.c;
const m256i = avx.m256i;
const m256 = avx.m256;

const QK8_0 = 32;

/// The column scales of one interleaved block, in the lane order the gemv's
/// `blend`s leave the columns in: `0, 4, 1, 5, 2, 6, 3, 7`.
inline fn colScaleGemv(comptime B: type, b: *const B) m256 {
    if (B == blocks.block_mxfp4x8) {
        // Load 8 E8M0 exponents and convert to float via LUT, rearranged to
        // match the fp16 path's order.
        return avx.mm256_set_ps(.{
            e8m0(b.e[7]), e8m0(b.e[3]), e8m0(b.e[6]), e8m0(b.e[2]),
            e8m0(b.e[5]), e8m0(b.e[1]), e8m0(b.e[4]), e8m0(b.e[0]),
        });
    }
    const changemask = avx.mm_set_epi8(.{ 15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0 });
    return common.f32cx8RearrangeLoad(&b.d, changemask);
}

/// The column scales in plain order, as the gemm wants them.
inline fn colScaleGemm(comptime B: type, b: *const B) m256 {
    if (B == blocks.block_mxfp4x8) {
        return avx.mm256_set_ps(.{
            e8m0(b.e[7]), e8m0(b.e[6]), e8m0(b.e[5]), e8m0(b.e[4]),
            e8m0(b.e[3]), e8m0(b.e[2]), e8m0(b.e[1]), e8m0(b.e[0]),
        });
    }
    return common.f32cx8Load(&b.d);
}

/// `GGML_CPU_E8M0_TO_FP32_HALF`: a lookup on x86.
inline fn e8m0(x: u8) f32 {
    return convert.ggml_table_f32_e8m0_half[x];
}

/// The nibbles of `raw` through the table: low nibbles, or high with `hi`.
inline fn lut(signextendlut: m256i, raw: m256i, comptime hi: bool) m256i {
    const m4b = avx.mm256_set1_epi8(0x0F);
    const v = if (hi) avx.mm256_srli_epi16(raw, 4) else raw;
    return avx.mm256_shuffle_epi8(signextendlut, avx.mm256_and_si256(v, m4b));
}

/// Ports `gemv_q4_b32_8x8_q8_0_lut_avx` (arch/x86/repack.cpp:522 @c1d0e7a00).
fn gemvLut(comptime B: type, n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int, signextendlut: m256i) void {
    const qk = QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    _ = bs;

    const finalpermutemask = avx.mm256_set_epi32(.{ 7, 5, 3, 1, 6, 4, 2, 0 });

    const b_nb: usize = @intCast(@divTrunc(n, 32));

    const b_ptr_start: [*]const B = @ptrCast(@alignCast(vx));
    const a_ptr_start: [*]const qblocks.Q8_0 = @ptrCast(@alignCast(vy));

    // Process Q8_0 blocks one by one
    for (0..@intCast(nr)) |y| {
        // Pointers to LHS blocks of block_q8_0 format
        const a_ptr = a_ptr_start + y * nb;

        // Take group of eight blocks at each pass of the loop
        for (0..@intCast(@divTrunc(nc, 8))) |x| {
            // Pointers to RHS blocks
            const b_ptr = b_ptr_start + x * b_nb;

            // Master FP accumulator
            var acc_row = avx.mm256_setzero_ps();

            for (0..nb) |b| {
                const qs: [*]const u8 = @ptrCast(&b_ptr[b].qs);
                // Load 8 blocks of 32 interleaved as 8 bytes (B0 - B7)
                const rhs_raw_vec_0123_0 = avx.mm256_loadu_si256(qs);
                const rhs_raw_vec_4567_0 = avx.mm256_loadu_si256(qs + 32);
                const rhs_raw_vec_0123_1 = avx.mm256_loadu_si256(qs + 64);
                const rhs_raw_vec_4567_1 = avx.mm256_loadu_si256(qs + 96);

                // 4-bit -> 8-bit - Sign is maintained
                const rhs_vec_0123_0 = lut(signextendlut, rhs_raw_vec_0123_0, false);
                const rhs_vec_4567_0 = lut(signextendlut, rhs_raw_vec_4567_0, false);
                const rhs_vec_0123_1 = lut(signextendlut, rhs_raw_vec_0123_1, false);
                const rhs_vec_4567_1 = lut(signextendlut, rhs_raw_vec_4567_1, false);

                const rhs_vec_0123_2 = lut(signextendlut, rhs_raw_vec_0123_0, true);
                const rhs_vec_4567_2 = lut(signextendlut, rhs_raw_vec_4567_0, true);
                const rhs_vec_0123_3 = lut(signextendlut, rhs_raw_vec_0123_1, true);
                const rhs_vec_4567_3 = lut(signextendlut, rhs_raw_vec_4567_1, true);

                // Load the scale values for the 8 blocks interleaved in block_tx8
                const col_scale_f32 = colScaleGemv(B, &b_ptr[b]);

                // Load and convert to FP32 scale from block_q8_0
                const row_scale_f32 = avx.mm256_set1_ps(common.f(a_ptr[b].d));

                // The block's values 16 bytes at a time, repeated across the
                // 256-bit vector
                const lhs_vec_0 = common.repeat128(avx.mm_loadu_si128(&a_ptr[b].qs));
                const lhs_vec_1 = common.repeat128(avx.mm_loadu_si128(a_ptr[b].qs[16..].ptr));

                var iacc = avx.mm256_setzero_si256();

                // Dot product within 32-bit lanes, accumulated in one vector
                iacc = common.mul_sum_i8_pairs_acc_int32x8(iacc, avx.mm256_blend_epi32(rhs_vec_0123_0, avx.mm256_shuffle_epi32(rhs_vec_4567_0, 177), 170), avx.mm256_shuffle_epi32(lhs_vec_0, 0));
                iacc = common.mul_sum_i8_pairs_acc_int32x8(iacc, avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(rhs_vec_0123_0, 177), rhs_vec_4567_0, 170), avx.mm256_shuffle_epi32(lhs_vec_0, 85));

                iacc = common.mul_sum_i8_pairs_acc_int32x8(iacc, avx.mm256_blend_epi32(rhs_vec_0123_1, avx.mm256_shuffle_epi32(rhs_vec_4567_1, 177), 170), avx.mm256_shuffle_epi32(lhs_vec_0, 170));
                iacc = common.mul_sum_i8_pairs_acc_int32x8(iacc, avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(rhs_vec_0123_1, 177), rhs_vec_4567_1, 170), avx.mm256_shuffle_epi32(lhs_vec_0, 255));

                iacc = common.mul_sum_i8_pairs_acc_int32x8(iacc, avx.mm256_blend_epi32(rhs_vec_0123_2, avx.mm256_shuffle_epi32(rhs_vec_4567_2, 177), 170), avx.mm256_shuffle_epi32(lhs_vec_1, 0));
                iacc = common.mul_sum_i8_pairs_acc_int32x8(iacc, avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(rhs_vec_0123_2, 177), rhs_vec_4567_2, 170), avx.mm256_shuffle_epi32(lhs_vec_1, 85));

                iacc = common.mul_sum_i8_pairs_acc_int32x8(iacc, avx.mm256_blend_epi32(rhs_vec_0123_3, avx.mm256_shuffle_epi32(rhs_vec_4567_3, 177), 170), avx.mm256_shuffle_epi32(lhs_vec_1, 170));
                iacc = common.mul_sum_i8_pairs_acc_int32x8(iacc, avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(rhs_vec_0123_3, 177), rhs_vec_4567_3, 170), avx.mm256_shuffle_epi32(lhs_vec_1, 255));

                // Accumulated values multiplied with appropriate scales
                acc_row = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc), avx.mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
            }

            // Accumulated output values permuted into storage order
            acc_row = avx.mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            avx.mm256_storeu_ps(s + (y * @as(usize, @intCast(nr)) + x * 8), acc_row);
        }
    }
}

/// One interleaved block's eight columns, decoded and arranged for the
/// gemm's two shuffle patterns: `[k]` is weights `8k..8k+7` of each column,
/// in `B0 B1 B4 B5` and `B2 B3 B6 B7` order.
const Rhs = struct {
    m0145_sp1: [4]m256i,
    m2367_sp1: [4]m256i,
    m0145_sp2: [4]m256i,
    m2367_sp2: [4]m256i,
};

fn prepareRhs(comptime B: type, b: *const B, signextendlut: m256i) Rhs {
    // Permute mask used for easier vector processing at later stages
    const requiredOrder = avx.mm256_set_epi32(.{ 3, 2, 1, 0, 7, 6, 5, 4 });

    const qs: [*]const u8 = @ptrCast(&b.qs);
    // The eight blocks of quantized values interleaved in chunks of eight
    const rhs_raw_mat_0123_0 = avx.mm256_loadu_si256(qs);
    const rhs_raw_mat_4567_0 = avx.mm256_loadu_si256(qs + 32);
    const rhs_raw_mat_0123_1 = avx.mm256_loadu_si256(qs + 64);
    const rhs_raw_mat_4567_1 = avx.mm256_loadu_si256(qs + 96);

    // In the formats B0B1B4B5, B2B3B6B7 for further processing
    const rhs_raw_mat_0145_0 = avx.mm256_blend_epi32(rhs_raw_mat_0123_0, avx.mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
    const rhs_raw_mat_2367_0 = avx.mm256_blend_epi32(avx.mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
    const rhs_raw_mat_0145_1 = avx.mm256_blend_epi32(rhs_raw_mat_0123_1, avx.mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
    const rhs_raw_mat_2367_1 = avx.mm256_blend_epi32(avx.mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

    // 4-bit -> 8-bit - Sign is maintained
    const m0145 = [4]m256i{
        lut(signextendlut, rhs_raw_mat_0145_0, false),
        lut(signextendlut, rhs_raw_mat_0145_1, false),
        lut(signextendlut, rhs_raw_mat_0145_0, true),
        lut(signextendlut, rhs_raw_mat_0145_1, true),
    };
    const m2367 = [4]m256i{
        lut(signextendlut, rhs_raw_mat_2367_0, false),
        lut(signextendlut, rhs_raw_mat_2367_1, false),
        lut(signextendlut, rhs_raw_mat_2367_0, true),
        lut(signextendlut, rhs_raw_mat_2367_1, true),
    };

    var r: Rhs = undefined;
    inline for (0..4) |k| {
        // Shuffle pattern one - right side input
        r.m0145_sp1[k] = avx.mm256_shuffle_epi32(m0145[k], 136);
        r.m2367_sp1[k] = avx.mm256_shuffle_epi32(m2367[k], 136);
        // Shuffle pattern two - right side input
        r.m0145_sp2[k] = avx.mm256_shuffle_epi32(m0145[k], 221);
        r.m2367_sp2[k] = avx.mm256_shuffle_epi32(m2367[k], 221);
    }
    return r;
}

/// `mul_sum_i8_pairs_acc_int32x8` over the four weight groups, innermost
/// first from group 3 as the C nests it.
fn dot4(lhs: [4]m256i, rhs: [4]m256i) m256i {
    var acc = avx.mm256_setzero_si256();
    acc = common.mul_sum_i8_pairs_acc_int32x8(acc, lhs[3], rhs[3]);
    acc = common.mul_sum_i8_pairs_acc_int32x8(acc, lhs[2], rhs[2]);
    acc = common.mul_sum_i8_pairs_acc_int32x8(acc, lhs[1], rhs[1]);
    return common.mul_sum_i8_pairs_acc_int32x8(acc, lhs[0], rhs[0]);
}

/// The per-block body of `gemm_q4_b32_8x8_q8_0_lut_avx` (arch/x86/repack.cpp:641
/// @c1d0e7a00): one interleaved block of four activation rows against one
/// prepared block of eight columns, into four row accumulators.
fn accumulate4(acc: *[4]m256, rhs: *const Rhs, col_scale_f32: m256, a: *const blocks.block_q8_0x4) void {
    const qs: [*]const i8 = &a.qs;
    var lhs_01_sp1: [4]m256i = undefined;
    var lhs_23_sp1: [4]m256i = undefined;
    var lhs_01_sp2: [4]m256i = undefined;
    var lhs_23_sp2: [4]m256i = undefined;
    inline for (0..4) |k| {
        // Four rows' values interleaved in chunks of eight, as 128-bit
        // halves repeated across the vector
        const lhs_mat_0123 = avx.mm256_loadu_si256(qs + 32 * k);
        const lhs_mat_01 = avx.mm256_permute2f128_si256(lhs_mat_0123, lhs_mat_0123, 0);
        const lhs_mat_23 = avx.mm256_permute2f128_si256(lhs_mat_0123, lhs_mat_0123, 17);
        // Shuffle pattern one - left side input
        lhs_01_sp1[k] = avx.mm256_shuffle_epi32(lhs_mat_01, 160);
        lhs_23_sp1[k] = avx.mm256_shuffle_epi32(lhs_mat_23, 160);
        // Shuffle pattern two - left side input
        lhs_01_sp2[k] = avx.mm256_shuffle_epi32(lhs_mat_01, 245);
        lhs_23_sp2[k] = avx.mm256_shuffle_epi32(lhs_mat_23, 245);
    }

    // Dot products within 32-bit lanes, resembling MMLAs into 2x2
    // matrices in the ARM version; both shuffle patterns summed.
    const iacc_mat_00 = avx.mm256_add_epi32(dot4(lhs_01_sp1, rhs.m0145_sp1), dot4(lhs_01_sp2, rhs.m0145_sp2));
    const iacc_mat_01 = avx.mm256_add_epi32(dot4(lhs_01_sp1, rhs.m2367_sp1), dot4(lhs_01_sp2, rhs.m2367_sp2));
    const iacc_mat_10 = avx.mm256_add_epi32(dot4(lhs_23_sp1, rhs.m0145_sp1), dot4(lhs_23_sp2, rhs.m0145_sp2));
    const iacc_mat_11 = avx.mm256_add_epi32(dot4(lhs_23_sp1, rhs.m2367_sp1), dot4(lhs_23_sp2, rhs.m2367_sp2));

    // Straighten out to make 4 row vectors
    const iacc_row = [4]m256i{
        avx.mm256_blend_epi32(iacc_mat_00, avx.mm256_shuffle_epi32(iacc_mat_01, 78), 204),
        avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204),
        avx.mm256_blend_epi32(iacc_mat_10, avx.mm256_shuffle_epi32(iacc_mat_11, 78), 204),
        avx.mm256_blend_epi32(avx.mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204),
    };

    // The four rows' scales, repeated across lanes
    const row_scale_f32 = common.f32cx8RepeatLoad(&a.d);

    // Multiply with appropriate scales and accumulate
    acc[0] = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc_row[0]), avx.mm256_mul_ps(col_scale_f32, avx.mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc[0]);
    acc[1] = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc_row[1]), avx.mm256_mul_ps(col_scale_f32, avx.mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc[1]);
    acc[2] = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc_row[2]), avx.mm256_mul_ps(col_scale_f32, avx.mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc[2]);
    acc[3] = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(iacc_row[3]), avx.mm256_mul_ps(col_scale_f32, avx.mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc[3]);
}

/// Ports `gemm_q4_b32_8x8_q8_0_lut_avx` (arch/x86/repack.cpp:641 @c1d0e7a00),
/// the arm without `__AVX512BW__ && __AVX512DQ__`.
fn gemmLut(comptime B: type, n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int, signextendlut: m256i) void {
    const qk = QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    const b_ptr_start: [*]const B = @ptrCast(@alignCast(vx));
    const a_ptr_start: [*]const blocks.block_q8_0x4 = @ptrCast(@alignCast(vy));

    const b_nb: usize = @intCast(@divTrunc(n, 32));
    var y: usize = 0;
    const xstart: usize = 0;
    const nr_u: usize = @intCast(nr);
    const anr = nr_u - nr_u % 16; // Used to align nr with boundary of 16
    const ncb: usize = @intCast(@divTrunc(nc, 8));

    // Four block_q8_0x4 -- sixteen rows -- at each pass of the loop
    while (y < anr / 4) : (y += 4) {
        var a_ptrs: [4][*]const blocks.block_q8_0x4 = undefined;

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (0..3) |i| a_ptrs[i + 1] = a_ptrs[i] + nb;

        // Eight columns at each pass of the loop
        for (xstart..ncb) |x| {
            const b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            var acc_rows: [16]m256 = @splat(avx.mm256_setzero_ps());

            for (0..nb) |b| {
                const rhs = prepareRhs(B, &b_ptr[b], signextendlut);
                const col_scale_f32 = colScaleGemm(B, &b_ptr[b]);

                // Process LHS in groups of four
                for (0..4) |rp| accumulate4(acc_rows[rp * 4 ..][0..4], &rhs, col_scale_f32, &a_ptrs[rp][b]);
            }

            // Store the accumulated values
            for (0..16) |i| avx.mm256_storeu_ps(s + ((y * 4 + i) * bs + x * 8), acc_rows[i]);
        }
    }

    // One block_q8_0x4 -- four rows -- at each pass of the loop
    while (y < nr_u / 4) : (y += 1) {
        const a_ptr = a_ptr_start + (y * nb);

        for (xstart..ncb) |x| {
            const b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            var acc_rows: [4]m256 = @splat(avx.mm256_setzero_ps());

            for (0..nb) |b| {
                const rhs = prepareRhs(B, &b_ptr[b], signextendlut);
                const col_scale_f32 = colScaleGemm(B, &b_ptr[b]);
                accumulate4(&acc_rows, &rhs, col_scale_f32, &a_ptr[b]);
            }

            // Store the accumulated values
            for (0..4) |i| avx.mm256_storeu_ps(s + ((y * 4 + i) * bs + x * 8), acc_rows[i]);
        }
    }
}

/// The table `ggml_gemv_q4_0_8x8_q8_0` builds (arch/x86/repack.cpp:1448
/// @c1d0e7a00): signed nibbles to signed bytes, 0..7 then -8..-1.
inline fn q4_0Lut() m256i {
    return common.repeat128(avx.mm_set_epi8(.{ -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0 }));
}

/// A 16-byte value table repeated into both halves.
inline fn tableLut(table: *const [16]i8) m256i {
    return common.repeat128(avx.mm_loadu_si128(table));
}

/// Ports `ggml_gemv_q4_0_8x8_q8_0` (arch/x86/repack.cpp:1448 @c1d0e7a00).
pub export fn ggml_gemv_q4_0_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemvLut(blocks.block_q4_0x8, n, s, bs, vx, vy, nr, nc, q4_0Lut());
}

/// Ports `ggml_gemv_iq4_nl_8x8_q8_0` (arch/x86/repack.cpp:1687 @c1d0e7a00).
pub export fn ggml_gemv_iq4_nl_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemvLut(blocks.block_iq4_nlx8, n, s, bs, vx, vy, nr, nc, tableLut(&c.kvalues_iq4nl));
}

/// Ports `ggml_gemv_mxfp4_8x8_q8_0` (arch/x86/repack.cpp:1700 @c1d0e7a00).
pub export fn ggml_gemv_mxfp4_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemvLut(blocks.block_mxfp4x8, n, s, bs, vx, vy, nr, nc, tableLut(&c.kvalues_mxfp4));
}

/// Ports `ggml_gemm_q4_0_8x8_q8_0` (arch/x86/repack.cpp:2026 @c1d0e7a00).
pub export fn ggml_gemm_q4_0_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemmLut(blocks.block_q4_0x8, n, s, bs, vx, vy, nr, nc, q4_0Lut());
}

/// Ports `ggml_gemm_iq4_nl_8x8_q8_0` (arch/x86/repack.cpp:3496 @c1d0e7a00).
pub export fn ggml_gemm_iq4_nl_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemmLut(blocks.block_iq4_nlx8, n, s, bs, vx, vy, nr, nc, tableLut(&c.kvalues_iq4nl));
}

/// Ports `ggml_gemm_mxfp4_8x8_q8_0` (arch/x86/repack.cpp:3511 @c1d0e7a00).
pub export fn ggml_gemm_mxfp4_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemmLut(blocks.block_mxfp4x8, n, s, bs, vx, vy, nr, nc, tableLut(&c.kvalues_mxfp4));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
