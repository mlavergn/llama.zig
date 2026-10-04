//! The NEON `q4_K` × `q8_K` interleaved gemv.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Three sums, not one
//!
//! Each sub-block contributes through three separate accumulators:
//!
//! - `acc_lo` / `acc_hi` — the low and high nibbles' dot products, kept
//!   apart because they carry **different** sub-block scales.
//! - `bias_acc` — the mins, weighted by the activation's `bsums`, which is
//!   subtracted once at the end of the block with `vmlsq_f32`.
//!
//! Collapsing any pair of them changes the scaling.
//!
//! # `vmlsq_f32` is a fused *subtract*
//!
//! `acc - a * b` in one rounding. Writing it as `acc - a * b` in Zig would
//! round twice; `neon.mlsq_f32` folds the negation into the `@mulAdd`.
//!
//! # The bsums are pre-paired
//!
//! `vpaddq_s16` folds the sixteen `bsums` into eight before the sub-block
//! loop, and each sub-block then takes two of them. That is a pairwise
//! add across two vectors, not a plain add — see `neon.paddq_s16`.

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

inline fn loadq(p: [*]const i8) i8x16 {
    return @bitCast(@as(@Vector(16, i8), p[0..16].*));
}
inline fn loadqu(p: [*]const u8) u8x16 {
    return p[0..16].*;
}
inline fn loadq16(p: [*]const i16) i16x8 {
    return p[0..8].*;
}

const m4b: u8x16 = @splat(0x0f);

/// Ports `ggml_gemv_q4_K_8x4_q8_K` (arch/arm/repack.cpp:576 @c1d0e7a00).
pub export fn ggml_gemv_q4_K_8x4_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const col_groups: usize = 2;

    var acc_f32: [col_groups]f32x4 = undefined;
    const q8_ptr: [*]const c.block_q8_K = @ptrCast(@alignCast(vy));

    var x: c_int = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const q4_ptr: [*]const blocks.block_q4_Kx8 =
            @as([*]const blocks.block_q4_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..col_groups) |i| acc_f32[i] = @splat(0);

        var b: usize = 0;
        while (b < @as(usize, @intCast(nb))) : (b += 1) {
            const q4_d_0 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q4_ptr[b].d)));
            const q4_d_1 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q4_ptr[b].d)) + 4));
            const q8_d: f32x4 = @splat(q8_ptr[b].d);
            const sb_scale_0123 = q4_d_0 * q8_d;
            const sb_scale_4567 = q4_d_1 * q8_d;
            const q4_dmin_0 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q4_ptr[b].dmin)));
            const q4_dmin_1 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q4_ptr[b].dmin)) + 4));
            const sb_min_0123 = q4_dmin_0 * q8_d;
            const sb_min_4567 = q4_dmin_1 * q8_d;

            var bias_acc: [2]i32x4 = .{ @splat(0), @splat(0) };
            var acc_lo: [col_groups]i32x4 = undefined;
            var acc_hi: [col_groups]i32x4 = undefined;

            const bsums_ptr: [*]const i16 = @ptrCast(&q8_ptr[b].bsums);
            const bsums = neon.paddq_s16(loadq16(bsums_ptr), loadq16(bsums_ptr + 8));
            const bsums_arr: [8]i16 = bsums;

            for (0..4) |sb| {
                for (0..col_groups) |i| {
                    acc_lo[i] = @splat(0);
                    acc_hi[i] = @splat(0);
                }

                var q4sb_mins: [2]i16x8 = undefined;
                var q4sb_scales: [2]i16x8 = undefined;
                for (0..2) |i| {
                    var aux_q4sb: [8]i8 = undefined;
                    const offset = sb * 24 + i * 12;
                    kscales.decodeQKx86BitScales(@as([*]const u8, &q4_ptr[b].scales) + offset, &q4sb_mins[i], &aux_q4sb);
                    q4sb_scales[i] = neon.movl_s8(@as(@Vector(8, i8), aux_q4sb));
                }

                var q8_qs: [4]i8x16 = undefined;
                for (0..4) |i| {
                    q8_qs[i] = loadq(@as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + sb * 64 + i * 16);
                }

                for (0..col_groups) |cg| {
                    var q4_cols: [8]u8x16 = undefined;
                    for (0..8) |i| {
                        q4_cols[i] = loadqu(@as([*]const u8, &q4_ptr[b].qs) + sb * 256 + i * 32 + 16 * cg);
                    }

                    // Low nibbles against q8_qs[0..1], high against [2..3];
                    // the lane index cycles every four columns.
                    inline for (0..8) |i| {
                        const lo: i8x16 = @bitCast(q4_cols[i] & m4b);
                        acc_lo[cg] = neon.dotq_laneq_s32(acc_lo[cg], lo, q8_qs[i / 4], i % 4);
                    }
                    inline for (0..8) |i| {
                        const hi: i8x16 = @bitCast(q4_cols[i] >> @as(@Vector(16, u3), @splat(4)));
                        acc_hi[cg] = neon.dotq_laneq_s32(acc_hi[cg], hi, q8_qs[2 + i / 4], i % 4);
                    }
                }

                const sc_0123_lo = neon.low(q4sb_scales[0]);
                const sc_0123_hi = neon.low(q4sb_scales[1]);
                const sumf_0123 = neon.cvt_f32_s32((neon.movl_s16(sc_0123_lo) *% acc_lo[0]) +%
                    (neon.movl_s16(sc_0123_hi) *% acc_hi[0]));
                acc_f32[0] = neon.fma_f32(acc_f32[0], sb_scale_0123, sumf_0123);

                const sc_4567_lo = neon.high(q4sb_scales[0]);
                const sc_4567_hi = neon.high(q4sb_scales[1]);
                const sumf_4567 = neon.cvt_f32_s32((neon.movl_s16(sc_4567_lo) *% acc_lo[1]) +%
                    (neon.movl_s16(sc_4567_hi) *% acc_hi[1]));
                acc_f32[1] = neon.fma_f32(acc_f32[1], sb_scale_4567, sumf_4567);

                const bsums_vec_lo: i16x4 = @splat(bsums_arr[2 * sb + 0]);
                const bsums_vec_hi: i16x4 = @splat(bsums_arr[2 * sb + 1]);

                bias_acc[0] = neon.mlal_s16(bias_acc[0], bsums_vec_lo, neon.low(q4sb_mins[0]));
                bias_acc[0] = neon.mlal_s16(bias_acc[0], bsums_vec_hi, neon.low(q4sb_mins[1]));
                bias_acc[1] = neon.mlal_s16(bias_acc[1], bsums_vec_lo, neon.high(q4sb_mins[0]));
                bias_acc[1] = neon.mlal_s16(bias_acc[1], bsums_vec_hi, neon.high(q4sb_mins[1]));
            }

            acc_f32[0] = neon.mlsq_f32(acc_f32[0], neon.cvt_f32_s32(bias_acc[0]), sb_min_0123);
            acc_f32[1] = neon.mlsq_f32(acc_f32[1], neon.cvt_f32_s32(bias_acc[1]), sb_min_4567);
        }

        const base: usize = @intCast(x * ncols_interleaved);
        s[base..][0..4].* = acc_f32[0];
        s[base + 4 ..][0..4].* = acc_f32[1];
    }
}

/// Ports `ggml_gemv_q4_K_8x8_q8_K` (arch/arm/repack.cpp:709 @c1d0e7a00).
///
/// Not the `8x4` kernel with a constant changed. It works in **column
/// pairs**: four accumulators rather than two, activations loaded as a
/// 64-bit value duplicated into both halves rather than lane-indexed, and
/// the pairs folded together at the end by the **pairwise** `vpaddq_s32`.
/// The mins half is identical, so only the dot-product half differs.
pub export fn ggml_gemv_q4_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const col_pairs: usize = 4;

    var acc_f32: [2]f32x4 = undefined;
    const q8_ptr: [*]const c.block_q8_K = @ptrCast(@alignCast(vy));

    var x: c_int = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const q4_ptr: [*]const blocks.block_q4_Kx8 =
            @as([*]const blocks.block_q4_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..2) |i| acc_f32[i] = @splat(0);

        var b: usize = 0;
        while (b < @as(usize, @intCast(nb))) : (b += 1) {
            const q4_d_0 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q4_ptr[b].d)));
            const q4_d_1 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q4_ptr[b].d)) + 4));
            const q8_d: f32x4 = @splat(q8_ptr[b].d);
            const sb_scale_0 = q4_d_0 * q8_d;
            const sb_scale_1 = q4_d_1 * q8_d;
            const q4_dmin_0 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q4_ptr[b].dmin)));
            const q4_dmin_1 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q4_ptr[b].dmin)) + 4));
            const sb_min_0 = q4_dmin_0 * q8_d;
            const sb_min_1 = q4_dmin_1 * q8_d;

            var bias_acc: [2]i32x4 = .{ @splat(0), @splat(0) };
            var acc_lo: [col_pairs]i32x4 = undefined;
            var acc_hi: [col_pairs]i32x4 = undefined;

            const bsums_ptr: [*]const i16 = @ptrCast(&q8_ptr[b].bsums);
            const bsums = neon.paddq_s16(loadq16(bsums_ptr), loadq16(bsums_ptr + 8));
            const bsums_arr: [8]i16 = bsums;

            for (0..4) |sb| {
                for (0..col_pairs) |i| {
                    acc_lo[i] = @splat(0);
                    acc_hi[i] = @splat(0);
                }

                var q4sb_mins: [2]i16x8 = undefined;
                var q4sb_scales: [2]i16x8 = undefined;
                for (0..2) |i| {
                    var aux_q4sb: [8]i8 = undefined;
                    const offset = sb * 24 + i * 12;
                    kscales.decodeQKx86BitScales(@as([*]const u8, &q4_ptr[b].scales) + offset, &q4sb_mins[i], &aux_q4sb);
                    q4sb_scales[i] = neon.movl_s8(@as(@Vector(8, i8), aux_q4sb));
                }

                // Eight bytes duplicated into both halves, not a lane
                // broadcast: each `q8_qs[i]` serves one 16-byte weight load.
                const q8_base: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + sb * 64;
                var q8_qs: [8]i8x16 = undefined;
                for (0..8) |i| q8_qs[i] = neon.dupq_i8x16_from8(q8_base + i * 8);

                const q4_base: [*]const u8 = @as([*]const u8, &q4_ptr[b].qs) + sb * 256;
                for (0..col_pairs) |cp| {
                    var q4: [4]u8x16 = undefined;
                    inline for (0..4) |i| q4[i] = loadqu(q4_base + 16 * cp + 64 * i);

                    inline for (0..4) |i| {
                        const lo: i8x16 = @bitCast(q4[i] & m4b);
                        acc_lo[cp] = neon.dotq_s32(acc_lo[cp], lo, q8_qs[i]);
                    }
                    inline for (0..4) |i| {
                        const hi: i8x16 = @bitCast(q4[i] >> @as(@Vector(16, u3), @splat(4)));
                        acc_hi[cp] = neon.dotq_s32(acc_hi[cp], hi, q8_qs[4 + i]);
                    }
                }

                // Two column pairs per output group, merged pairwise.
                inline for (0..2) |i| {
                    const p = i * 2;
                    const group_scales_lo = if (p == 0) neon.low(q4sb_scales[0]) else neon.high(q4sb_scales[0]);
                    const group_scales_hi = if (p == 0) neon.low(q4sb_scales[1]) else neon.high(q4sb_scales[1]);
                    const sb_scale = if (p == 0) sb_scale_0 else sb_scale_1;

                    const sumf_0 = neon.cvt_f32_s32(neon.movl_s16(group_scales_lo) *%
                        neon.paddq_s32(acc_lo[p], acc_lo[p + 1]));
                    acc_f32[i] = neon.fma_f32(acc_f32[i], sb_scale, sumf_0);

                    const sumf_1 = neon.cvt_f32_s32(neon.movl_s16(group_scales_hi) *%
                        neon.paddq_s32(acc_hi[p], acc_hi[p + 1]));
                    acc_f32[i] = neon.fma_f32(acc_f32[i], sb_scale, sumf_1);
                }

                const bsums_vec_lo: i16x4 = @splat(bsums_arr[2 * sb + 0]);
                const bsums_vec_hi: i16x4 = @splat(bsums_arr[2 * sb + 1]);

                bias_acc[0] = neon.mlal_s16(bias_acc[0], bsums_vec_lo, neon.low(q4sb_mins[0]));
                bias_acc[0] = neon.mlal_s16(bias_acc[0], bsums_vec_hi, neon.low(q4sb_mins[1]));
                bias_acc[1] = neon.mlal_s16(bias_acc[1], bsums_vec_lo, neon.high(q4sb_mins[0]));
                bias_acc[1] = neon.mlal_s16(bias_acc[1], bsums_vec_hi, neon.high(q4sb_mins[1]));
            }

            acc_f32[0] = neon.mlsq_f32(acc_f32[0], neon.cvt_f32_s32(bias_acc[0]), sb_min_0);
            acc_f32[1] = neon.mlsq_f32(acc_f32[1], neon.cvt_f32_s32(bias_acc[1]), sb_min_1);
        }

        const base: usize = @intCast(x * ncols_interleaved);
        s[base..][0..4].* = acc_f32[0];
        s[base + 4 ..][0..4].* = acc_f32[1];
    }
}

/// Ports `ggml_gemm_q4_K_8x4_q8_K` (arch/arm/repack.cpp:3323 @c1d0e7a00).
///
/// Eight accumulators, laid out as `2 * row + col_group`: four rows of the
/// `q8_Kx4` block against two groups of four columns. The lane index of
/// `vdotq_laneq_s32` selects the row, so one weight load feeds all four.
///
/// The per-row scale is the column scale times `q8_d[row]`, broadcast —
/// the gemv's single `q8_d` splat becomes four lane splats here.
///
/// Unlike `q6_K`, this one *does* use `vfmaq_f32` for the combine.
pub export fn ggml_gemm_q4_K_8x4_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const q8_k_blocklen: usize = 4;
    const acc_size: usize = 8;

    var acc_f32: [acc_size]f32x4 = undefined;

    var y: c_int = 0;
    while (y < @divTrunc(nr, @as(c_int, q8_k_blocklen))) : (y += 1) {
        const q8_ptr: [*]const blocks.block_q8_Kx4 =
            @as([*]const blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));

        var x: c_int = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const q4_ptr: [*]const blocks.block_q4_Kx8 =
                @as([*]const blocks.block_q4_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

            for (0..acc_size) |i| acc_f32[i] = @splat(0);

            var b: usize = 0;
            while (b < @as(usize, @intCast(nb))) : (b += 1) {
                const q4_d_0123 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q4_ptr[b].d)));
                const q4_d_4567 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q4_ptr[b].d)) + 4));
                const q4_dmin_0123 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q4_ptr[b].dmin)));
                const q4_dmin_4567 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q4_ptr[b].dmin)) + 4));
                const q8_d_0123: f32x4 = @as([*]const f32, &q8_ptr[b].d)[0..4].*;

                var sbd_scale_0123: [4]f32x4 = undefined;
                var sbd_scale_4567: [4]f32x4 = undefined;
                var sbd_min_0123: [4]f32x4 = undefined;
                var sbd_min_4567: [4]f32x4 = undefined;
                inline for (0..4) |row| {
                    const lane: f32x4 = @splat(q8_d_0123[row]);
                    sbd_scale_0123[row] = q4_d_0123 * lane;
                    sbd_scale_4567[row] = q4_d_4567 * lane;
                    sbd_min_0123[row] = q4_dmin_0123 * lane;
                    sbd_min_4567[row] = q4_dmin_4567 * lane;
                }

                // Each row's sixteen bsums folded to eight, pairwise.
                var bsums_arr: [4][8]i16 = undefined;
                inline for (0..4) |q8_row| {
                    const bp: [*]const i16 = @as([*]const i16, @ptrCast(&q8_ptr[b].bsums)) + 16 * q8_row;
                    bsums_arr[q8_row] = neon.paddq_s16(loadq16(bp), loadq16(bp + 8));
                }

                var bias_acc: [acc_size]i32x4 = .{@as(i32x4, @splat(0))} ** acc_size;

                for (0..4) |sb| {
                    var acc_lo: [acc_size]i32x4 = .{@as(i32x4, @splat(0))} ** acc_size;
                    var acc_hi: [acc_size]i32x4 = .{@as(i32x4, @splat(0))} ** acc_size;

                    var q4sb_scales: [2]i16x8 = undefined;
                    var q4sb_mins: [2]i16x8 = undefined;
                    for (0..2) |i| {
                        var aux: [8]i8 = undefined;
                        const offset = sb * 24 + i * 12;
                        kscales.decodeQKx86BitScales(@as([*]const u8, &q4_ptr[b].scales) + offset, &q4sb_mins[i], &aux);
                        q4sb_scales[i] = neon.movl_s8(@as(@Vector(8, i8), aux));
                    }

                    for (0..8) |k| {
                        const q8_qs: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + sb * 256;
                        const q8_blk0 = loadq(q8_qs + 16 * k);
                        const q8_blk1 = loadq(q8_qs + 16 * k + 128);
                        const q4_qs: [*]const u8 = @as([*]const u8, &q4_ptr[b].qs) + sb * 256;
                        const q4_0123 = loadqu(q4_qs + 32 * k);
                        const q4_4567 = loadqu(q4_qs + 32 * k + 16);

                        const q4_0123_lo: i8x16 = @bitCast(q4_0123 & m4b);
                        const q4_0123_hi: i8x16 = @bitCast(q4_0123 >> @as(@Vector(16, u3), @splat(4)));
                        inline for (0..4) |r| {
                            acc_lo[r] = neon.dotq_laneq_s32(acc_lo[r], q4_0123_lo, q8_blk0, r);
                            acc_hi[r] = neon.dotq_laneq_s32(acc_hi[r], q4_0123_hi, q8_blk1, r);
                        }

                        const q4_4567_lo: i8x16 = @bitCast(q4_4567 & m4b);
                        const q4_4567_hi: i8x16 = @bitCast(q4_4567 >> @as(@Vector(16, u3), @splat(4)));
                        inline for (0..4) |r| {
                            acc_lo[4 + r] = neon.dotq_laneq_s32(acc_lo[4 + r], q4_4567_lo, q8_blk0, r);
                            acc_hi[4 + r] = neon.dotq_laneq_s32(acc_hi[4 + r], q4_4567_hi, q8_blk1, r);
                        }
                    }

                    const sc_0123_lo = neon.low(q4sb_scales[0]);
                    const sc_4567_lo = neon.high(q4sb_scales[0]);
                    const sc_0123_hi = neon.low(q4sb_scales[1]);
                    const sc_4567_hi = neon.high(q4sb_scales[1]);

                    inline for (0..4) |row| {
                        const sumf_0123 = neon.cvt_f32_s32((neon.movl_s16(sc_0123_lo) *% acc_lo[row]) +%
                            (neon.movl_s16(sc_0123_hi) *% acc_hi[row]));
                        acc_f32[2 * row] = neon.fma_f32(acc_f32[2 * row], sbd_scale_0123[row], sumf_0123);

                        const sumf_4567 = neon.cvt_f32_s32((neon.movl_s16(sc_4567_lo) *% acc_lo[row + 4]) +%
                            (neon.movl_s16(sc_4567_hi) *% acc_hi[row + 4]));
                        acc_f32[2 * row + 1] = neon.fma_f32(acc_f32[2 * row + 1], sbd_scale_4567[row], sumf_4567);

                        const bsums_vec_lo: i16x4 = @splat(bsums_arr[sb][row * 2]);
                        const bsums_vec_hi: i16x4 = @splat(bsums_arr[sb][row * 2 + 1]);
                        bias_acc[2 * row] = neon.mlal_s16(bias_acc[2 * row], bsums_vec_lo, neon.low(q4sb_mins[0]));
                        bias_acc[2 * row] = neon.mlal_s16(bias_acc[2 * row], bsums_vec_hi, neon.low(q4sb_mins[1]));
                        bias_acc[2 * row + 1] = neon.mlal_s16(bias_acc[2 * row + 1], bsums_vec_lo, neon.high(q4sb_mins[0]));
                        bias_acc[2 * row + 1] = neon.mlal_s16(bias_acc[2 * row + 1], bsums_vec_hi, neon.high(q4sb_mins[1]));
                    }
                }

                inline for (0..4) |row| {
                    acc_f32[2 * row] = neon.mlsq_f32(acc_f32[2 * row], neon.cvt_f32_s32(bias_acc[2 * row]), sbd_min_0123[row]);
                    acc_f32[2 * row + 1] = neon.mlsq_f32(acc_f32[2 * row + 1], neon.cvt_f32_s32(bias_acc[2 * row + 1]), sbd_min_4567[row]);
                }
            }

            for (0..q8_k_blocklen) |i| {
                const row: usize = @intCast(y * @as(c_int, q8_k_blocklen) + @as(c_int, @intCast(i)));
                for (0..2) |j| {
                    const col: usize = @intCast(x * ncols_interleaved + @as(c_int, @intCast(j)) * 4);
                    const offset = row * bs + col;
                    s[offset..][0..4].* = acc_f32[2 * i + j];
                }
            }
        }
    }
}

comptime {
    _ = std;
}
