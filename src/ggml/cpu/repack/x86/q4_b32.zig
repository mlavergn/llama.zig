//! The AVX2 gemv and gemm for the three 4-bit, 32-weight-block formats that
//! interleave eight columns: `q4_0`, `iq4_nl` and `mxfp4`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` at v0.3.0
//! (`c1d0e7a00`): the templates `gemv_q4_b32_8x8_q8_0_lut_avx` and
//! `gemm_q4_b32_8x8_q8_0_lut_avx` and the six entry points that instantiate
//! them. Each declaration names the C++ it replaces and the line it began
//! at. The gemm's `__AVX512BW__ && __AVX512DQ__` arm is the same body at
//! 512 bits, over two blocks at once (`wide.zig`).
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
//! order within a row is the C's. The same holds for the 512-bit arm, so
//! `rowLoops` is generic over the width and runs once per arm.
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
const wide = @import("wide.zig");

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

/// One interleaved block's eight columns -- or, at 512 bits, two blocks'
/// sixteen -- decoded and arranged for the gemm's two shuffle patterns: `[k]`
/// is weights `8k..8k+7` of each column, in `B0 B1 B4 B5` and `B2 B3 B6 B7`
/// order (`B8 B9 BC BD` and `BA BB BE BF` in the high half).
fn Rhs(comptime L: type) type {
    return struct {
        m0145_sp1: [4]L.I,
        m2367_sp1: [4]L.I,
        m0145_sp2: [4]L.I,
        m2367_sp2: [4]L.I,
    };
}

/// One block's raw nibbles in the formats `B0B1B4B5` and `B2B3B6B7`, the
/// 256-bit `blend`s both arms open with: `[0]` weights 0-7 and 16-23, `[1]`
/// 8-15 and 24-31.
fn rawRhs(comptime B: type, b: *const B) struct { m0145: [2]m256i, m2367: [2]m256i } {
    // Permute mask used for easier vector processing at later stages
    const requiredOrder = avx.mm256_set_epi32(.{ 3, 2, 1, 0, 7, 6, 5, 4 });

    const qs: [*]const u8 = @ptrCast(&b.qs);
    // The eight blocks of quantized values interleaved in chunks of eight
    const rhs_raw_mat_0123_0 = avx.mm256_loadu_si256(qs);
    const rhs_raw_mat_4567_0 = avx.mm256_loadu_si256(qs + 32);
    const rhs_raw_mat_0123_1 = avx.mm256_loadu_si256(qs + 64);
    const rhs_raw_mat_4567_1 = avx.mm256_loadu_si256(qs + 96);

    return .{
        .m0145 = .{
            avx.mm256_blend_epi32(rhs_raw_mat_0123_0, avx.mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240),
            avx.mm256_blend_epi32(rhs_raw_mat_0123_1, avx.mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240),
        },
        .m2367 = .{
            avx.mm256_blend_epi32(avx.mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240),
            avx.mm256_blend_epi32(avx.mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240),
        },
    };
}

/// The nibbles of `raw` through the table at `L`'s width: low nibbles, or
/// high with `hi`.
inline fn lutWide(comptime L: type, signextendlut: m256i, raw: L.I, comptime hi: bool) L.I {
    const m4b = L.set1_epi8(0x0F);
    const v = if (hi) L.srli_epi16(raw, 4) else raw;
    return L.shuffle_epi8(L.rep(signextendlut), L.and_si(v, m4b));
}

/// The columns of `b0` -- and of `b1` in the high half at 512 bits -- for one
/// block row.
fn prepareRhs(comptime L: type, comptime B: type, b0: *const B, b1: *const B, signextendlut: m256i) Rhs(L) {
    const r0 = rawRhs(B, b0);
    const r1 = if (L.width == 512) rawRhs(B, b1) else r0;
    const raw0145 = [2]L.I{ L.join(r0.m0145[0], r1.m0145[0]), L.join(r0.m0145[1], r1.m0145[1]) };
    const raw2367 = [2]L.I{ L.join(r0.m2367[0], r1.m2367[0]), L.join(r0.m2367[1], r1.m2367[1]) };

    // 4-bit -> 8-bit - Sign is maintained
    const m0145 = [4]L.I{
        lutWide(L, signextendlut, raw0145[0], false),
        lutWide(L, signextendlut, raw0145[1], false),
        lutWide(L, signextendlut, raw0145[0], true),
        lutWide(L, signextendlut, raw0145[1], true),
    };
    const m2367 = [4]L.I{
        lutWide(L, signextendlut, raw2367[0], false),
        lutWide(L, signextendlut, raw2367[1], false),
        lutWide(L, signextendlut, raw2367[0], true),
        lutWide(L, signextendlut, raw2367[1], true),
    };

    var r: Rhs(L) = undefined;
    inline for (0..4) |k| {
        // Shuffle pattern one - right side input
        r.m0145_sp1[k] = L.shuffle_epi32(m0145[k], 136);
        r.m2367_sp1[k] = L.shuffle_epi32(m2367[k], 136);
        // Shuffle pattern two - right side input
        r.m0145_sp2[k] = L.shuffle_epi32(m0145[k], 221);
        r.m2367_sp2[k] = L.shuffle_epi32(m2367[k], 221);
    }
    return r;
}

/// The column scales of `b0`, and of `b1` above them at 512 bits:
/// `GGML_F32Cx8x2_LOAD`, or two `_mm256_set_ps` of E8M0 values.
inline fn colScaleWide(comptime L: type, comptime B: type, b0: *const B, b1: *const B) L.F {
    if (L.width == 256) return colScaleGemm(B, b0);
    return L.joinF(colScaleGemm(B, b0), colScaleGemm(B, b1));
}

/// `mul_sum_i8_pairs_acc_int32x8` or `..._int32x16` over the four weight
/// groups, innermost first from group 3 as the C nests it.
fn dot4(comptime L: type, lhs: [4]L.I, rhs: [4]L.I) L.I {
    var acc = L.setzero();
    acc = L.mul_sum_i8_pairs_acc(acc, lhs[3], rhs[3]);
    acc = L.mul_sum_i8_pairs_acc(acc, lhs[2], rhs[2]);
    acc = L.mul_sum_i8_pairs_acc(acc, lhs[1], rhs[1]);
    return L.mul_sum_i8_pairs_acc(acc, lhs[0], rhs[0]);
}

/// The per-block body of `gemm_q4_b32_8x8_q8_0_lut_avx` (arch/x86/repack.cpp:641
/// @c1d0e7a00), either arm: one interleaved block of four activation rows
/// against one prepared block row, into four row accumulators.
fn accumulate4(comptime L: type, acc: *[4]L.F, rhs: *const Rhs(L), col_scale_f32: L.F, a: *const blocks.block_q8_0x4) void {
    const qs: [*]const i8 = &a.qs;
    var lhs_01_sp1: [4]L.I = undefined;
    var lhs_23_sp1: [4]L.I = undefined;
    var lhs_01_sp2: [4]L.I = undefined;
    var lhs_23_sp2: [4]L.I = undefined;
    inline for (0..4) |k| {
        // Four rows' values interleaved in chunks of eight, as 128-bit
        // halves repeated across the vector
        const lhs_mat_0123 = avx.mm256_loadu_si256(qs + 32 * k);
        const lhs_mat_01 = L.rep(avx.mm256_permute2f128_si256(lhs_mat_0123, lhs_mat_0123, 0));
        const lhs_mat_23 = L.rep(avx.mm256_permute2f128_si256(lhs_mat_0123, lhs_mat_0123, 17));
        // Shuffle pattern one - left side input
        lhs_01_sp1[k] = L.shuffle_epi32(lhs_mat_01, 160);
        lhs_23_sp1[k] = L.shuffle_epi32(lhs_mat_23, 160);
        // Shuffle pattern two - left side input
        lhs_01_sp2[k] = L.shuffle_epi32(lhs_mat_01, 245);
        lhs_23_sp2[k] = L.shuffle_epi32(lhs_mat_23, 245);
    }

    // Dot products within 32-bit lanes, resembling MMLAs into 2x2
    // matrices in the ARM version; both shuffle patterns summed.
    const iacc_mat_00 = L.add_epi32(dot4(L, lhs_01_sp1, rhs.m0145_sp1), dot4(L, lhs_01_sp2, rhs.m0145_sp2));
    const iacc_mat_01 = L.add_epi32(dot4(L, lhs_01_sp1, rhs.m2367_sp1), dot4(L, lhs_01_sp2, rhs.m2367_sp2));
    const iacc_mat_10 = L.add_epi32(dot4(L, lhs_23_sp1, rhs.m0145_sp1), dot4(L, lhs_23_sp2, rhs.m0145_sp2));
    const iacc_mat_11 = L.add_epi32(dot4(L, lhs_23_sp1, rhs.m2367_sp1), dot4(L, lhs_23_sp2, rhs.m2367_sp2));

    // Straighten out to make 4 row vectors
    const iacc_row = [4]L.I{
        L.blend_epi32(iacc_mat_00, L.shuffle_epi32(iacc_mat_01, 78), 204),
        L.blend_epi32(L.shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204),
        L.blend_epi32(iacc_mat_10, L.shuffle_epi32(iacc_mat_11, 78), 204),
        L.blend_epi32(L.shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204),
    };

    // The four rows' scales, repeated across lanes: `GGML_F32Cx8_REPEAT_LOAD`
    // or `GGML_F32Cx16_REPEAT_LOAD`
    const row_scale_f32 = L.repF(common.f32cx8RepeatLoad(&a.d));

    // Multiply with appropriate scales and accumulate
    const lanes = [4]u8{ 0, 85, 170, 255 };
    inline for (0..4) |i| {
        acc[i] = L.fmadd_ps(L.cvtepi32_ps(iacc_row[i]), L.mul_ps(col_scale_f32, L.shuffle_ps(row_scale_f32, lanes[i])), acc[i]);
    }
}

/// Both row loops of `gemm_q4_b32_8x8_q8_0_lut_avx` at `L`'s width, over the
/// eight-column blocks `x_begin..x_end`: sixteen rows a pass, then four.
/// `y` is left where the C leaves it.
fn rowLoops(comptime L: type, comptime B: type, y: *usize, x_begin: usize, x_end: usize, nb: usize, b_nb: usize, nr_u: usize, s: [*]f32, bs: usize, b_ptr_start: [*]const B, a_ptr_start: [*]const blocks.block_q8_0x4, signextendlut: m256i) void {
    const anr = nr_u - nr_u % 16; // Used to align nr with boundary of 16
    const xstep = L.cols / 8;

    // Four block_q8_0x4 -- sixteen rows -- at each pass of the loop
    while (y.* < anr / 4) : (y.* += 4) {
        var a_ptrs: [4][*]const blocks.block_q8_0x4 = undefined;

        a_ptrs[0] = a_ptr_start + (y.* * nb);
        for (0..3) |i| a_ptrs[i + 1] = a_ptrs[i] + nb;

        // Eight columns -- sixteen at 512 bits -- at each pass of the loop
        var x = x_begin;
        while (x < x_end) : (x += xstep) {
            const b_ptr_0 = b_ptr_start + (x * b_nb);
            const b_ptr_1 = b_ptr_start + ((x + xstep - 1) * b_nb);

            // Master FP accumulators
            var acc_rows: [16]L.F = @splat(L.setzero_ps());

            for (0..nb) |b| {
                const rhs = prepareRhs(L, B, &b_ptr_0[b], &b_ptr_1[b], signextendlut);
                const col_scale_f32 = colScaleWide(L, B, &b_ptr_0[b], &b_ptr_1[b]);

                // Process LHS in groups of four
                for (0..4) |rp| accumulate4(L, acc_rows[rp * 4 ..][0..4], &rhs, col_scale_f32, &a_ptrs[rp][b]);
            }

            // Store the accumulated values
            for (0..16) |i| L.storeu_ps(s + ((y.* * 4 + i) * bs + x * 8), acc_rows[i]);
        }
    }

    // One block_q8_0x4 -- four rows -- at each pass of the loop
    while (y.* < nr_u / 4) : (y.* += 1) {
        const a_ptr = a_ptr_start + (y.* * nb);

        var x = x_begin;
        while (x < x_end) : (x += xstep) {
            const b_ptr_0 = b_ptr_start + (x * b_nb);
            const b_ptr_1 = b_ptr_start + ((x + xstep - 1) * b_nb);

            // Master FP accumulators
            var acc_rows: [4]L.F = @splat(L.setzero_ps());

            for (0..nb) |b| {
                const rhs = prepareRhs(L, B, &b_ptr_0[b], &b_ptr_1[b], signextendlut);
                const col_scale_f32 = colScaleWide(L, B, &b_ptr_0[b], &b_ptr_1[b]);
                accumulate4(L, &acc_rows, &rhs, col_scale_f32, &a_ptr[b]);
            }

            // Store the accumulated values
            for (0..4) |i| L.storeu_ps(s + ((y.* * 4 + i) * bs + x * 8), acc_rows[i]);
        }
    }
}

/// Ports `gemm_q4_b32_8x8_q8_0_lut_avx` (arch/x86/repack.cpp:641 @c1d0e7a00).
///
/// With `__AVX512BW__ && __AVX512DQ__` the C first runs its 512-bit arm,
/// from line 663, over the columns up to the last
/// multiple of sixteen, two blocks a pass; if any are left, the AVX2 loops
/// then run over those alone, **from the first row again**.
fn gemmLut(comptime B: type, n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int, signextendlut: m256i) void {
    const qk = QK8_0;
    const nb: usize = @intCast(@divTrunc(n, qk));

    const b_ptr_start: [*]const B = @ptrCast(@alignCast(vx));
    const a_ptr_start: [*]const blocks.block_q8_0x4 = @ptrCast(@alignCast(vy));

    const b_nb: usize = @intCast(@divTrunc(n, 32));
    var y: usize = 0;
    var xstart: usize = 0;
    const nr_u: usize = @intCast(nr);
    const nc_u: usize = @intCast(nc);

    if (comptime wide.has_avx512bw_dq) {
        const anc = nc_u - nc_u % 16; // Used to align nc with boundary of 16
        rowLoops(wide.Wide(512), B, &y, 0, anc / 8, nb, b_nb, nr_u, s, bs, b_ptr_start, a_ptr_start, signextendlut);
        if (anc != nc_u) {
            xstart = anc / 8;
            y = 0;
        }
    }

    rowLoops(wide.Wide(256), B, &y, xstart, nc_u / 8, nb, b_nb, nr_u, s, bs, b_ptr_start, a_ptr_start, signextendlut);
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
