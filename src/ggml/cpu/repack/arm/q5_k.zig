//! The NEON `q5_K` × `q8_K` interleaved gemv.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # `q4_k.zig` plus a fifth bit, and the `qh` state is *running*
//!
//! The scale decode, the mins accumulation and the two output shapes are
//! `q4_K`'s. What is new is `qh`, and the way it is consumed is the trap:
//! it is loaded **once per block**, outside the sub-block loop, and each
//! sub-block **shifts it right by two in place**. Re-loading it per
//! sub-block, or shifting a copy, reads the wrong pair of bits for every
//! sub-block after the first.
//!
//! Per sub-block each `qh` byte yields two bits:
//!
//! - `hbit_lo = qh & 1`, inserted at bit 4 of the low nibble.
//! - `hbit_hi = (qh & 2) << 3`, OR-ed onto the high nibble.
//!
//! `vsliq_n_u8` does the first; the operand is already `& 0x0f`, so it is
//! an OR in this use, but see `neon.sli_n_u8` for why it is written as the
//! instruction rather than as an OR.

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
const mone: u8x16 = @splat(1);
const mtwo: u8x16 = @splat(2);
const sh3: @Vector(16, u3) = @splat(3);
const sh4: @Vector(16, u3) = @splat(4);
const sh2: @Vector(16, u3) = @splat(2);

/// Ports `ggml_gemv_q5_K_8x4_q8_K` (arch/arm/repack.cpp:863 @c1d0e7a00).
pub export fn ggml_gemv_q5_K_8x4_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const col_groups: usize = 2;

    var acc_f32: [col_groups]f32x4 = undefined;
    const q8_ptr: [*]const c.block_q8_K = @ptrCast(@alignCast(vy));

    var x: c_int = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const q5_ptr: [*]const blocks.block_q5_Kx8 =
            @as([*]const blocks.block_q5_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..col_groups) |i| acc_f32[i] = @splat(0);

        var b: usize = 0;
        while (b < @as(usize, @intCast(nb))) : (b += 1) {
            const q5_d_0 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q5_ptr[b].d)));
            const q5_d_1 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q5_ptr[b].d)) + 4));
            const q8_d: f32x4 = @splat(q8_ptr[b].d);
            const sb_scale_0123 = q5_d_0 * q8_d;
            const sb_scale_4567 = q5_d_1 * q8_d;
            const q5_dmin_0 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q5_ptr[b].dmin)));
            const q5_dmin_1 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q5_ptr[b].dmin)) + 4));
            const sb_min_0123 = q5_dmin_0 * q8_d;
            const sb_min_4567 = q5_dmin_1 * q8_d;

            var bias_acc: [2]i32x4 = .{ @splat(0), @splat(0) };
            var acc_lo: [col_groups]i32x4 = undefined;
            var acc_hi: [col_groups]i32x4 = undefined;

            // Loaded once per block and shifted in place below; this is
            // running state across the sub-block loop.
            var qh: [col_groups][8]u8x16 = undefined;
            for (0..col_groups) |cg| {
                for (0..8) |i| {
                    qh[cg][i] = loadqu(@as([*]const u8, &q5_ptr[b].qh) + i * 32 + 16 * cg);
                }
            }

            const bsums_ptr: [*]const i16 = @ptrCast(&q8_ptr[b].bsums);
            const bsums = neon.paddq_s16(loadq16(bsums_ptr), loadq16(bsums_ptr + 8));
            const bsums_arr: [8]i16 = bsums;

            for (0..4) |sb| {
                for (0..col_groups) |i| {
                    acc_lo[i] = @splat(0);
                    acc_hi[i] = @splat(0);
                }

                var q5sb_mins: [2]i16x8 = undefined;
                var q5sb_scales: [2]i16x8 = undefined;
                for (0..2) |i| {
                    var aux: [8]i8 = undefined;
                    const offset = sb * 24 + i * 12;
                    kscales.decodeQKx86BitScales(@as([*]const u8, &q5_ptr[b].scales) + offset, &q5sb_mins[i], &aux);
                    q5sb_scales[i] = neon.movl_s8(@as(@Vector(8, i8), aux));
                }

                var q8_qs: [4]i8x16 = undefined;
                for (0..4) |i| {
                    q8_qs[i] = loadq(@as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + sb * 64 + i * 16);
                }

                for (0..col_groups) |cg| {
                    var q5_lo: [8]i8x16 = undefined;
                    var q5_hi: [8]i8x16 = undefined;
                    for (0..8) |i| {
                        const q5_col = loadqu(@as([*]const u8, &q5_ptr[b].qs) + sb * 256 + i * 32 + 16 * cg);
                        const hbit_lo = qh[cg][i] & mone;
                        const hbit_hi = (qh[cg][i] & mtwo) << sh3;
                        // In place: the next sub-block reads the next pair.
                        qh[cg][i] = qh[cg][i] >> sh2;

                        q5_lo[i] = @bitCast(neon.sli_n_u8(q5_col & m4b, hbit_lo, 4));
                        q5_hi[i] = @bitCast((q5_col >> sh4) | hbit_hi);
                    }

                    inline for (0..8) |i| {
                        acc_lo[cg] = neon.dotq_laneq_s32(acc_lo[cg], q5_lo[i], q8_qs[i / 4], i % 4);
                    }
                    inline for (0..8) |i| {
                        acc_hi[cg] = neon.dotq_laneq_s32(acc_hi[cg], q5_hi[i], q8_qs[2 + i / 4], i % 4);
                    }
                }

                const sc_0123_lo = neon.low(q5sb_scales[0]);
                const sc_0123_hi = neon.low(q5sb_scales[1]);
                const sumf_0123 = neon.cvt_f32_s32((neon.movl_s16(sc_0123_lo) *% acc_lo[0]) +%
                    (neon.movl_s16(sc_0123_hi) *% acc_hi[0]));
                acc_f32[0] = neon.fma_f32(acc_f32[0], sb_scale_0123, sumf_0123);

                const sc_4567_lo = neon.high(q5sb_scales[0]);
                const sc_4567_hi = neon.high(q5sb_scales[1]);
                const sumf_4567 = neon.cvt_f32_s32((neon.movl_s16(sc_4567_lo) *% acc_lo[1]) +%
                    (neon.movl_s16(sc_4567_hi) *% acc_hi[1]));
                acc_f32[1] = neon.fma_f32(acc_f32[1], sb_scale_4567, sumf_4567);

                const bsums_vec_lo: i16x4 = @splat(bsums_arr[2 * sb + 0]);
                const bsums_vec_hi: i16x4 = @splat(bsums_arr[2 * sb + 1]);

                bias_acc[0] = neon.mlal_s16(bias_acc[0], bsums_vec_lo, neon.low(q5sb_mins[0]));
                bias_acc[0] = neon.mlal_s16(bias_acc[0], bsums_vec_hi, neon.low(q5sb_mins[1]));
                bias_acc[1] = neon.mlal_s16(bias_acc[1], bsums_vec_lo, neon.high(q5sb_mins[0]));
                bias_acc[1] = neon.mlal_s16(bias_acc[1], bsums_vec_hi, neon.high(q5sb_mins[1]));
            }

            acc_f32[0] = neon.mlsq_f32(acc_f32[0], neon.cvt_f32_s32(bias_acc[0]), sb_min_0123);
            acc_f32[1] = neon.mlsq_f32(acc_f32[1], neon.cvt_f32_s32(bias_acc[1]), sb_min_4567);
        }

        const base: usize = @intCast(x * ncols_interleaved);
        s[base..][0..4].* = acc_f32[0];
        s[base + 4 ..][0..4].* = acc_f32[1];
    }
}

/// Ports `ggml_gemv_q5_K_8x8_q8_K` (arch/arm/repack.cpp:1022 @c1d0e7a00).
///
/// `q4_K`'s 8x8 shape — column pairs, 64-bit-dup activations, pairwise
/// merge — with this file's fifth-bit folding. The C unrolls the column
/// pairs fully; the loads are at `16 * cp + 64 * i`, which is the same
/// addressing the rolled form gives.
///
/// `qh` here is `[4][4]`, not `[2][8]`, and is shifted in place per
/// sub-block exactly as in the 8x4 kernel.
pub export fn ggml_gemv_q5_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const col_pairs: usize = 4;

    var acc_f32: [2]f32x4 = undefined;
    const q8_ptr: [*]const c.block_q8_K = @ptrCast(@alignCast(vy));

    var x: c_int = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const q5_ptr: [*]const blocks.block_q5_Kx8 =
            @as([*]const blocks.block_q5_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..2) |i| acc_f32[i] = @splat(0);

        var b: usize = 0;
        while (b < @as(usize, @intCast(nb))) : (b += 1) {
            const q5_d_0 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q5_ptr[b].d)));
            const q5_d_1 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q5_ptr[b].d)) + 4));
            const q8_d: f32x4 = @splat(q8_ptr[b].d);
            const sb_scale_0 = q5_d_0 * q8_d;
            const sb_scale_1 = q5_d_1 * q8_d;
            const q5_dmin_0 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q5_ptr[b].dmin)));
            const q5_dmin_1 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q5_ptr[b].dmin)) + 4));
            const sb_min_0 = q5_dmin_0 * q8_d;
            const sb_min_1 = q5_dmin_1 * q8_d;

            var acc_lo: [col_pairs]i32x4 = undefined;
            var acc_hi: [col_pairs]i32x4 = undefined;

            const qh_base: [*]const u8 = @as([*]const u8, &q5_ptr[b].qh);
            var qh: [col_pairs][4]u8x16 = undefined;
            for (0..col_pairs) |cp| {
                inline for (0..4) |i| qh[cp][i] = loadqu(qh_base + 16 * cp + 64 * i);
            }

            const bsums_ptr: [*]const i16 = @ptrCast(&q8_ptr[b].bsums);
            const bsums = neon.paddq_s16(loadq16(bsums_ptr), loadq16(bsums_ptr + 8));
            const bsums_arr: [8]i16 = bsums;

            for (0..4) |sb| {
                for (0..col_pairs) |i| {
                    acc_lo[i] = @splat(0);
                    acc_hi[i] = @splat(0);
                }

                var q5sb_mins: [2]i16x8 = undefined;
                var q5sb_scales: [2]i16x8 = undefined;
                for (0..2) |i| {
                    var aux: [8]i8 = undefined;
                    const offset = sb * 24 + i * 12;
                    kscales.decodeQKx86BitScales(@as([*]const u8, &q5_ptr[b].scales) + offset, &q5sb_mins[i], &aux);
                    q5sb_scales[i] = neon.movl_s8(@as(@Vector(8, i8), aux));
                }

                const q8_base: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + sb * 64;
                var q8_qs: [8]i8x16 = undefined;
                for (0..8) |i| q8_qs[i] = neon.dupq_i8x16_from8(q8_base + i * 8);

                const qs_base: [*]const u8 = @as([*]const u8, &q5_ptr[b].qs) + sb * 256;
                for (0..col_pairs) |cp| {
                    inline for (0..4) |i| {
                        const qs = loadqu(qs_base + 16 * cp + 64 * i);
                        const hbit_lo = qh[cp][i] & mone;
                        const hbit_hi = (qh[cp][i] & mtwo) << sh3;
                        qh[cp][i] = qh[cp][i] >> sh2;

                        const lo: i8x16 = @bitCast(neon.sli_n_u8(qs & m4b, hbit_lo, 4));
                        const hi: i8x16 = @bitCast((qs >> sh4) | hbit_hi);
                        acc_lo[cp] = neon.dotq_s32(acc_lo[cp], lo, q8_qs[i]);
                        acc_hi[cp] = neon.dotq_s32(acc_hi[cp], hi, q8_qs[4 + i]);
                    }
                }

                // Each pair of sub-blocks shares one bsum.
                const bsums_vec_lo: i16x4 = @splat(bsums_arr[2 * sb + 0]);
                const bsums_vec_hi: i16x4 = @splat(bsums_arr[2 * sb + 1]);

                inline for (0..2) |i| {
                    const p = i * 2;
                    const group_scales_lo = if (p == 0) neon.low(q5sb_scales[0]) else neon.high(q5sb_scales[0]);
                    const group_scales_hi = if (p == 0) neon.low(q5sb_scales[1]) else neon.high(q5sb_scales[1]);
                    const group_mins_lo = if (p == 0) neon.low(q5sb_mins[0]) else neon.high(q5sb_mins[0]);
                    const group_mins_hi = if (p == 0) neon.low(q5sb_mins[1]) else neon.high(q5sb_mins[1]);
                    const sb_scale = if (p == 0) sb_scale_0 else sb_scale_1;
                    const sb_min = if (p == 0) sb_min_0 else sb_min_1;

                    const sumf_0 = neon.cvt_f32_s32(neon.movl_s16(group_scales_lo) *%
                        neon.paddq_s32(acc_lo[p], acc_lo[p + 1]));
                    acc_f32[i] = neon.fma_f32(acc_f32[i], sb_scale, sumf_0);

                    const sumf_1 = neon.cvt_f32_s32(neon.movl_s16(group_scales_hi) *%
                        neon.paddq_s32(acc_hi[p], acc_hi[p + 1]));
                    acc_f32[i] = neon.fma_f32(acc_f32[i], sb_scale, sumf_1);

                    // The C's "FUSED BIAS": the bias is formed and
                    // subtracted here, inside the sub-block loop, not
                    // accumulated in an integer and subtracted once after
                    // it. `sb_min` is constant across `sb`, so the two are
                    // the same arithmetic -- and not the same rounding.
                    // `scripts/repack-diff` caught the difference.
                    var bias = neon.mull_s16(bsums_vec_lo, group_mins_lo);
                    bias = neon.mlal_s16(bias, bsums_vec_hi, group_mins_hi);
                    acc_f32[i] = neon.mlsq_f32(acc_f32[i], sb_min, neon.cvt_f32_s32(bias));
                }
            }
        }

        const base: usize = @intCast(x * ncols_interleaved);
        s[base..][0..4].* = acc_f32[0];
        s[base + 4 ..][0..4].* = acc_f32[1];
    }
}

/// Ports `ggml_gemm_q5_K_8x4_q8_K` (arch/arm/repack.cpp:3523 @c1d0e7a00).
///
/// `q4_k.zig`'s gemm with this file's fifth-bit fold. `qh` is again
/// per-block running state, shifted right by two once per `k` read —
/// note that is per **read**, not per sub-block as in the gemv, because
/// the gemm walks eight reads per sub-block and each consumes one bit
/// pair.
pub export fn ggml_gemm_q5_K_8x4_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const q8_k_blocklen: usize = 4;
    const acc_size: usize = 8;
    const col_groups: usize = 2;

    var acc_f32: [acc_size]f32x4 = undefined;

    var y: c_int = 0;
    while (y < @divTrunc(nr, @as(c_int, q8_k_blocklen))) : (y += 1) {
        const q8_ptr: [*]const blocks.block_q8_Kx4 =
            @as([*]const blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));

        var x: c_int = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const q5_ptr: [*]const blocks.block_q5_Kx8 =
                @as([*]const blocks.block_q5_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

            for (0..acc_size) |i| acc_f32[i] = @splat(0);

            var b: usize = 0;
            while (b < @as(usize, @intCast(nb))) : (b += 1) {
                const q5_d_0123 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q5_ptr[b].d)));
                const q5_d_4567 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q5_ptr[b].d)) + 4));
                const q5_dmin_0123 = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&q5_ptr[b].dmin)));
                const q5_dmin_4567 = neon.cvt_f32_f16(neon.load_f16x4(@as([*]const u16, @ptrCast(&q5_ptr[b].dmin)) + 4));
                const q8_d_0123: f32x4 = @as([*]const f32, &q8_ptr[b].d)[0..4].*;

                var sbd_scale_0123: [4]f32x4 = undefined;
                var sbd_scale_4567: [4]f32x4 = undefined;
                var sbd_min_0123: [4]f32x4 = undefined;
                var sbd_min_4567: [4]f32x4 = undefined;
                inline for (0..4) |row| {
                    const lane: f32x4 = @splat(q8_d_0123[row]);
                    sbd_scale_0123[row] = q5_d_0123 * lane;
                    sbd_scale_4567[row] = q5_d_4567 * lane;
                    sbd_min_0123[row] = q5_dmin_0123 * lane;
                    sbd_min_4567[row] = q5_dmin_4567 * lane;
                }

                var bsums_arr: [4][8]i16 = undefined;
                inline for (0..4) |q8_row| {
                    const bp: [*]const i16 = @as([*]const i16, @ptrCast(&q8_ptr[b].bsums)) + 16 * q8_row;
                    bsums_arr[q8_row] = neon.paddq_s16(loadq16(bp), loadq16(bp + 8));
                }

                var qh: [col_groups][8]u8x16 = undefined;
                for (0..col_groups) |cg| {
                    for (0..8) |i| qh[cg][i] = loadqu(@as([*]const u8, &q5_ptr[b].qh) + i * 32 + 16 * cg);
                }

                var bias_acc: [acc_size]i32x4 = .{@as(i32x4, @splat(0))} ** acc_size;

                for (0..4) |sb| {
                    var acc_lo: [acc_size]i32x4 = .{@as(i32x4, @splat(0))} ** acc_size;
                    var acc_hi: [acc_size]i32x4 = .{@as(i32x4, @splat(0))} ** acc_size;

                    var q5sb_scales: [2]i16x8 = undefined;
                    var q5sb_mins: [2]i16x8 = undefined;
                    for (0..2) |i| {
                        var aux: [8]i8 = undefined;
                        const offset = sb * 24 + i * 12;
                        kscales.decodeQKx86BitScales(@as([*]const u8, &q5_ptr[b].scales) + offset, &q5sb_mins[i], &aux);
                        q5sb_scales[i] = neon.movl_s8(@as(@Vector(8, i8), aux));
                    }

                    for (0..8) |k| {
                        const q8_qs: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + sb * 256;
                        const q8_blk0 = loadq(q8_qs + 16 * k);
                        const q8_blk1 = loadq(q8_qs + 16 * k + 128);
                        const q5_qs: [*]const u8 = @as([*]const u8, &q5_ptr[b].qs) + sb * 256;
                        const q5_0123 = loadqu(q5_qs + 32 * k);
                        const q5_4567 = loadqu(q5_qs + 32 * k + 16);

                        const hbit_lo_0123 = qh[0][k] & mone;
                        const hbit_hi_0123 = (qh[0][k] & mtwo) << sh3;
                        qh[0][k] = qh[0][k] >> sh2;
                        const hbit_lo_4567 = qh[1][k] & mone;
                        const hbit_hi_4567 = (qh[1][k] & mtwo) << sh3;
                        qh[1][k] = qh[1][k] >> sh2;

                        const q5_0123_lo: i8x16 = @bitCast(neon.sli_n_u8(q5_0123 & m4b, hbit_lo_0123, 4));
                        const q5_0123_hi: i8x16 = @bitCast((q5_0123 >> sh4) | hbit_hi_0123);
                        inline for (0..4) |r| {
                            acc_lo[r] = neon.dotq_laneq_s32(acc_lo[r], q5_0123_lo, q8_blk0, r);
                            acc_hi[r] = neon.dotq_laneq_s32(acc_hi[r], q5_0123_hi, q8_blk1, r);
                        }

                        const q5_4567_lo: i8x16 = @bitCast(neon.sli_n_u8(q5_4567 & m4b, hbit_lo_4567, 4));
                        const q5_4567_hi: i8x16 = @bitCast((q5_4567 >> sh4) | hbit_hi_4567);
                        inline for (0..4) |r| {
                            acc_lo[4 + r] = neon.dotq_laneq_s32(acc_lo[4 + r], q5_4567_lo, q8_blk0, r);
                            acc_hi[4 + r] = neon.dotq_laneq_s32(acc_hi[4 + r], q5_4567_hi, q8_blk1, r);
                        }
                    }

                    const sc_0123_lo = neon.low(q5sb_scales[0]);
                    const sc_4567_lo = neon.high(q5sb_scales[0]);
                    const sc_0123_hi = neon.low(q5sb_scales[1]);
                    const sc_4567_hi = neon.high(q5sb_scales[1]);

                    inline for (0..4) |row| {
                        const sumf_0123 = neon.cvt_f32_s32((neon.movl_s16(sc_0123_lo) *% acc_lo[row]) +%
                            (neon.movl_s16(sc_0123_hi) *% acc_hi[row]));
                        acc_f32[2 * row] = neon.fma_f32(acc_f32[2 * row], sbd_scale_0123[row], sumf_0123);

                        const sumf_4567 = neon.cvt_f32_s32((neon.movl_s16(sc_4567_lo) *% acc_lo[row + 4]) +%
                            (neon.movl_s16(sc_4567_hi) *% acc_hi[row + 4]));
                        acc_f32[2 * row + 1] = neon.fma_f32(acc_f32[2 * row + 1], sbd_scale_4567[row], sumf_4567);

                        const bsums_vec_lo: i16x4 = @splat(bsums_arr[sb][row * 2]);
                        const bsums_vec_hi: i16x4 = @splat(bsums_arr[sb][row * 2 + 1]);
                        bias_acc[2 * row] = neon.mlal_s16(bias_acc[2 * row], bsums_vec_lo, neon.low(q5sb_mins[0]));
                        bias_acc[2 * row] = neon.mlal_s16(bias_acc[2 * row], bsums_vec_hi, neon.low(q5sb_mins[1]));
                        bias_acc[2 * row + 1] = neon.mlal_s16(bias_acc[2 * row + 1], bsums_vec_lo, neon.high(q5sb_mins[0]));
                        bias_acc[2 * row + 1] = neon.mlal_s16(bias_acc[2 * row + 1], bsums_vec_hi, neon.high(q5sb_mins[1]));
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
                    s[row * bs + col ..][0..4].* = acc_f32[2 * i + j];
                }
            }
        }
    }
}

comptime {
    _ = std;
}
