//! The K-quant super-block formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), the
//! 2- through 6-bit super-block sections at lines 889-2313 plus `q8_K` at
//! 2766. Each function names the C function it replaces and the line it began
//! at.
//!
//! # Two levels of scale
//!
//! A super-block is 256 weights. One f16 scale across all of them is too
//! coarse, and an f16 per 16 weights costs more than the weights do. So each
//! sub-block gets its own scale, *quantized* against a super-block scale:
//! `q2_K` packs a 4-bit scale and a 4-bit min per sub-block, `q4_K` and `q5_K`
//! use 6 bits of each, `q6_K` a full signed byte.
//!
//! Dequantizing is therefore two multiplies: `d * sub_scale * q`, minus
//! `dmin * sub_min` for the asymmetric formats.
//!
//! # Bit packing is where these go wrong
//!
//! Every format here interleaves. `q2_K` and `q3_K` pack weights `j`, `j+32`,
//! `j+64` and `j+96` into the four 2-bit fields of one byte -- a stride of 32,
//! not adjacent elements. `q4_K` pairs `j` with `j+32`. `q3_K` scatters its
//! 6-bit scales across 12 bytes in two pieces, and its third weight bit lives
//! in a separate `hmask` where bit `n` covers weights `8n` to `8n+7`.
//!
//! None of that is checkable by round-tripping through our own dequantizer:
//! matching mistakes cancel. The tests compare bytes against the C.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const h = @import("helpers.zig");
const c = impl.c;

const fp16 = impl.fp32ToFp16;
const unfp16 = impl.fp16ToFp32;
const QK_K = c.QK_K;

// -----------------------------------------------------------------------------
// q2_K -- 2 bits per weight, 16 sub-blocks of 16

/// Ports `quantize_row_q2_K_ref` (ggml-quants.c:891 @c1d0e7a00).
///
/// Three passes: fit each sub-block's scale and min, quantize those 16 pairs
/// against the block maxima into 4 bits each, then requantize the weights
/// against the *rounded* scales -- not the fitted ones, so the error the
/// rounding introduced is absorbed rather than compounded.
pub export fn quantize_row_q2_K_ref(x_in: [*c]const f32, y: [*c]blocks.Q2_K, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var x = x_in;

    var l_buf: [QK_K]u8 = undefined;
    var l_aux: [16]u8 = undefined;
    var weights: [16]f32 = undefined;
    var mins: [QK_K / 16]f32 = undefined;
    var scales: [QK_K / 16]f32 = undefined;

    // The sub-block scales get 4 bits, so they are quantized against 15.
    const q4scale: f32 = 15.0;

    for (0..nb) |i| {
        // Deducting the min makes every scale positive, so the maxima start at
        // zero rather than -inf.
        var max_scale: f32 = 0;
        var max_min: f32 = 0;
        for (0..QK_K / 16) |j| {
            for (0..16) |l| weights[l] = @abs(x[16 * j + l]);
            scales[j] = h.makeQkx2Quants(16, 3, x + 16 * j, &weights, @ptrCast(&l_buf[16 * j]), &mins[j], &l_aux, -0.5, 0.1, 15, true);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        if (max_scale > 0) {
            const iscale = q4scale / max_scale;
            for (0..QK_K / 16) |j| {
                y[i].scales[j] = @intCast(h.nearestInt(iscale * scales[j]));
            }
            y[i].d = fp16(max_scale / q4scale);
        } else {
            for (0..QK_K / 16) |j| y[i].scales[j] = 0;
            y[i].d = fp16(0.0);
        }
        if (max_min > 0) {
            const iscale = q4scale / max_min;
            for (0..QK_K / 16) |j| {
                const l = h.nearestInt(iscale * mins[j]);
                y[i].scales[j] |= @as(u8, @intCast(l)) << 4;
            }
            y[i].dmin = fp16(max_min / q4scale);
        } else {
            y[i].dmin = fp16(0.0);
        }

        // Requantize against the rounded scales.
        for (0..QK_K / 16) |j| {
            const d = unfp16(y[i].d) * @as(f32, @floatFromInt(y[i].scales[j] & 0xF));
            if (d == 0) continue;
            const dm = unfp16(y[i].dmin) * @as(f32, @floatFromInt(y[i].scales[j] >> 4));
            for (0..16) |ii| {
                var l = h.nearestInt((x[16 * j + ii] + dm) / d);
                l = @max(0, @min(3, l));
                l_buf[16 * j + ii] = @intCast(l);
            }
        }

        // Weights j, j+32, j+64 and j+96 share a byte -- a stride of 32.
        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                y[i].qs[j / 4 + l] = l_buf[j + l] |
                    (l_buf[j + l + 32] << 2) |
                    (l_buf[j + l + 64] << 4) |
                    (l_buf[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

/// Ports `dequantize_row_q2_K` (ggml-quants.c:961 @c1d0e7a00).
pub export fn dequantize_row_q2_K(x: [*c]const blocks.Q2_K, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        const min = unfp16(x[i].dmin);

        var q: [*]const u8 = &x[i].qs;
        var is: usize = 0;

        var n: usize = 0;
        while (n < QK_K) : (n += 128) {
            var shift: u8 = 0;
            for (0..4) |_| {
                var sc = x[i].scales[is];
                is += 1;
                var dl = d * @as(f32, @floatFromInt(sc & 0xF));
                var ml = min * @as(f32, @floatFromInt(sc >> 4));
                for (0..16) |l| {
                    y[0] = dl * @as(f32, @floatFromInt((q[l] >> @as(u3, @intCast(shift))) & 3)) - ml;
                    y += 1;
                }

                sc = x[i].scales[is];
                is += 1;
                dl = d * @as(f32, @floatFromInt(sc & 0xF));
                ml = min * @as(f32, @floatFromInt(sc >> 4));
                for (0..16) |l| {
                    y[0] = dl * @as(f32, @floatFromInt((q[l + 16] >> @as(u3, @intCast(shift))) & 3)) - ml;
                    y += 1;
                }

                shift += 2;
            }
            q += 32;
        }
    }
}

// -----------------------------------------------------------------------------
// q3_K -- 3 bits per weight, 16 sub-blocks of 16

/// Ports `quantize_row_q3_K_ref` (ggml-quants.c:1229 @c1d0e7a00).
///
/// Symmetric, so there is no min. The 16 sub-block scales are signed 6-bit
/// values biased by 32, split across `scales[0..12]`: the low four bits in the
/// first eight bytes (packed two per byte) and the top two bits in the last
/// four, at `2*(j/4)`.
pub export fn quantize_row_q3_K_ref(x_in: [*c]const f32, y: [*c]blocks.Q3_K, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var x = x_in;

    var l_buf: [QK_K]i8 = undefined;
    var scales: [QK_K / 16]f32 = undefined;

    for (0..nb) |i| {
        var max_scale: f32 = 0;
        var amax: f32 = 0;
        for (0..QK_K / 16) |j| {
            scales[j] = h.makeQ3Quants(16, 4, x + 16 * j, @ptrCast(&l_buf[16 * j]), true);
            const scale = @abs(scales[j]);
            if (scale > amax) {
                amax = scale;
                max_scale = scales[j];
            }
        }

        @memset(&y[i].scales, 0);
        if (max_scale != 0) {
            const iscale = -32.0 / max_scale;
            for (0..QK_K / 16) |j| {
                var l: i8 = @truncate(@as(i32, @intCast(h.nearestInt(iscale * scales[j]))));
                l = @max(-32, @min(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = @as(u8, @bitCast(l)) & 0xF;
                } else {
                    y[i].scales[j - 8] |= (@as(u8, @bitCast(l)) & 0xF) << 4;
                }
                const hi = @as(u8, @bitCast(l)) >> 4;
                y[i].scales[j % 4 + 8] |= hi << @intCast(2 * (j / 4));
            }
            y[i].d = fp16(1 / iscale);
        } else {
            y[i].d = fp16(0.0);
        }

        // Requantize against the rounded, reassembled scales.
        for (0..QK_K / 16) |j| {
            var sc: i8 = if (j < 8)
                @bitCast(y[i].scales[j] & 0xF)
            else
                @bitCast(y[i].scales[j - 8] >> 4);
            sc = @bitCast(@as(u8, @bitCast(sc)) | (((y[i].scales[8 + j % 4] >> @intCast(2 * (j / 4))) & 3) << 4));
            sc -%= 32;
            const d = unfp16(y[i].d) * @as(f32, @floatFromInt(sc));
            if (d == 0) continue;
            for (0..16) |ii| {
                var l = h.nearestInt(x[16 * j + ii] / d);
                l = @max(-4, @min(3, l));
                l_buf[16 * j + ii] = @intCast(l + 4);
            }
        }

        // The third bit goes to hmask, bit n covering weights 8n..8n+7.
        @memset(&y[i].hmask, 0);
        var m: usize = 0;
        var hm: u8 = 1;
        for (0..QK_K) |j| {
            if (l_buf[j] > 3) {
                y[i].hmask[m] |= hm;
                l_buf[j] -= 4;
            }
            m += 1;
            if (m == QK_K / 8) {
                m = 0;
                hm <<= 1;
            }
        }

        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                y[i].qs[j / 4 + l] = @as(u8, @bitCast(l_buf[j + l])) |
                    (@as(u8, @bitCast(l_buf[j + l + 32])) << 2) |
                    (@as(u8, @bitCast(l_buf[j + l + 64])) << 4) |
                    (@as(u8, @bitCast(l_buf[j + l + 96])) << 6);
            }
        }

        x += QK_K;
    }
}

/// Ports `dequantize_row_q3_K` (ggml-quants.c:1305 @c1d0e7a00).
///
/// Reassembles the 6-bit scales with the same word-at-a-time trick the C uses:
/// the twelve bytes are read as four `u32`s and the high bits shuffled in
/// four masked operations rather than sixteen scalar ones.
pub export fn dequantize_row_q3_K(x: [*c]const blocks.Q3_K, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;

    var aux: [4]u32 = undefined;
    const scales: [*]const i8 = @ptrCast(&aux);

    for (0..nb) |i| {
        const d_all = unfp16(x[i].d);

        var q: [*]const u8 = &x[i].qs;
        const hm: [*]const u8 = &x[i].hmask;
        var m: u8 = 1;

        @memcpy(std.mem.asBytes(&aux)[0..12], &x[i].scales);
        const tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        var is: usize = 0;
        var n: usize = 0;
        while (n < QK_K) : (n += 128) {
            var shift: u8 = 0;
            for (0..4) |_| {
                var dl = d_all * @as(f32, @floatFromInt(@as(i32, scales[is]) - 32));
                is += 1;
                for (0..16) |l| {
                    // The high bit being *set* means "do not subtract 4" --
                    // inverted from what the packing suggests.
                    const bias: i32 = if ((hm[l] & m) != 0) 0 else 4;
                    y[0] = dl * @as(f32, @floatFromInt(@as(i32, @as(i8, @intCast((q[l] >> @as(u3, @intCast(shift))) & 3))) - bias));
                    y += 1;
                }

                dl = d_all * @as(f32, @floatFromInt(@as(i32, scales[is]) - 32));
                is += 1;
                for (0..16) |l| {
                    const bias: i32 = if ((hm[l + 16] & m) != 0) 0 else 4;
                    y[0] = dl * @as(f32, @floatFromInt(@as(i32, @as(i8, @intCast((q[l + 16] >> @as(u3, @intCast(shift))) & 3))) - bias));
                    y += 1;
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// -----------------------------------------------------------------------------
// q4_K / q5_K -- 8 sub-blocks of 32, asymmetric
//
// Both fit their sub-block scales the same way and pack them identically into
// twelve bytes of 6-bit fields; they differ only in how many bits each weight
// gets. The shared parts are factored out below rather than written twice.

/// The scale and min fitting `quantize_row_q4_K_ref` and
/// `quantize_row_q5_K_ref` share (ggml-quants.c:1457, 1644 @c1d0e7a00) --
/// the `for (int j = 0; j < QK_K/32; ++j)` loop in each, at :1470 and
/// :1657.
///
/// The weighting is not the plain magnitude the other formats use: it is
/// `sqrt(mean(x^2)) + |x|`, which stops a sub-block of uniformly tiny weights
/// from being fitted as though its largest element mattered.
fn fitScalesK(
    comptime nmax: i32,
    comptime rmin: f32,
    comptime nstep: i32,
    comptime use_mad: bool,
    x: [*]const f32,
    l_buf: [*]u8,
    scales: *[QK_K / 32]f32,
    mins: *[QK_K / 32]f32,
) struct { max_scale: f32, max_min: f32 } {
    var weights: [32]f32 = undefined;
    var l_aux: [32]u8 = undefined;
    var max_scale: f32 = 0;
    var max_min: f32 = 0;

    for (0..QK_K / 32) |j| {
        var sum_x2: f32 = 0;
        for (0..32) |l| sum_x2 += x[32 * j + l] * x[32 * j + l];
        const av_x = @sqrt(sum_x2 / 32);
        for (0..32) |l| weights[l] = av_x + @abs(x[32 * j + l]);
        scales[j] = h.makeQkx2Quants(32, nmax, x + 32 * j, &weights, l_buf + 32 * j, &mins[j], &l_aux, rmin, 0.1, nstep, use_mad);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }
    return .{ .max_scale = max_scale, .max_min = max_min };
}

/// Packs eight 6-bit scale/min pairs into twelve bytes, as
/// `quantize_row_q4_K_ref` and `quantize_row_q5_K_ref` both do
/// (ggml-quants.c:1457, 1644 @c1d0e7a00).
///
/// The first four pairs go in whole; the last four have their low nibble in
/// `scales[j+4]` and their top two bits stolen into the *high* bits of an
/// earlier byte. `getScaleMinK4` is the inverse.
fn packScalesK(scales_out: [*]u8, scales: *const [QK_K / 32]f32, mins: *const [QK_K / 32]f32, max_scale: f32, max_min: f32) void {
    const inv_scale: f32 = if (max_scale > 0) 63.0 / max_scale else 0.0;
    const inv_min: f32 = if (max_min > 0) 63.0 / max_min else 0.0;

    for (0..QK_K / 32) |j| {
        var ls: u8 = @truncate(@as(u32, @bitCast(h.nearestInt(inv_scale * scales[j]))));
        var lm: u8 = @truncate(@as(u32, @bitCast(h.nearestInt(inv_min * mins[j]))));
        ls = @min(63, ls);
        lm = @min(63, lm);
        if (j < 4) {
            scales_out[j] = ls;
            scales_out[j + 4] = lm;
        } else {
            scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales_out[j - 4] |= (ls >> 4) << 6;
            scales_out[j - 0] |= (lm >> 4) << 6;
        }
    }
}

/// Ports `quantize_row_q4_K_ref` (ggml-quants.c:1457 @c1d0e7a00).
pub export fn quantize_row_q4_K_ref(x_in: [*c]const f32, y: [*c]blocks.Q4_K, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var x = x_in;

    var l_buf: [QK_K]u8 = undefined;
    var mins: [QK_K / 32]f32 = undefined;
    var scales: [QK_K / 32]f32 = undefined;

    for (0..nb) |i| {
        const fit = fitScalesK(15, -1.0, 20, false, x, &l_buf, &scales, &mins);

        @memset(&y[i].scales, 0);
        packScalesK(&y[i].scales, &scales, &mins, fit.max_scale, fit.max_min);
        y[i].d = fp16(fit.max_scale / 63.0);
        y[i].dmin = fp16(fit.max_min / 63.0);

        for (0..QK_K / 32) |j| {
            var sc: u8 = undefined;
            var m: u8 = undefined;
            h.getScaleMinK4(j, &y[i].scales, &sc, &m);
            const d = unfp16(y[i].d) * @as(f32, @floatFromInt(sc));
            if (d == 0) continue;
            const dm = unfp16(y[i].dmin) * @as(f32, @floatFromInt(m));
            for (0..32) |ii| {
                var l = h.nearestInt((x[32 * j + ii] + dm) / d);
                l = @max(0, @min(15, l));
                l_buf[32 * j + ii] = @intCast(l);
            }
        }

        // Weight j pairs with j+32 -- the two halves of a 64-weight span.
        var q: [*]u8 = &y[i].qs;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            for (0..32) |l| q[l] = l_buf[j + l] | (l_buf[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

/// Ports `dequantize_row_q4_K` (ggml-quants.c:1529 @c1d0e7a00).
pub export fn dequantize_row_q4_K(x: [*c]const blocks.Q4_K, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    for (0..nb) |i| {
        var q: [*]const u8 = &x[i].qs;
        const d = unfp16(x[i].d);
        const min = unfp16(x[i].dmin);

        var is: usize = 0;
        var sc: u8 = undefined;
        var m: u8 = undefined;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            h.getScaleMinK4(is + 0, &x[i].scales, &sc, &m);
            const d1 = d * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m));
            h.getScaleMinK4(is + 1, &x[i].scales, &sc, &m);
            const d2 = d * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m));
            for (0..32) |l| {
                y[0] = d1 * @as(f32, @floatFromInt(q[l] & 0xF)) - m1;
                y += 1;
            }
            for (0..32) |l| {
                y[0] = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
                y += 1;
            }
            q += 32;
            is += 2;
        }
    }
}

/// Ports `quantize_row_q5_K_ref` (ggml-quants.c:1644 @c1d0e7a00).
pub export fn quantize_row_q5_K_ref(x_in: [*c]const f32, y: [*c]blocks.Q5_K, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var x = x_in;

    var l_buf: [QK_K]u8 = undefined;
    var mins: [QK_K / 32]f32 = undefined;
    var scales: [QK_K / 32]f32 = undefined;

    for (0..nb) |i| {
        // Note the different search parameters from q4_K: -0.5 and 15 steps
        // rather than -1.0 and 20. More levels to aim at, so a narrower search.
        const fit = fitScalesK(31, -0.5, 15, false, x, &l_buf, &scales, &mins);

        @memset(&y[i].scales, 0);
        packScalesK(&y[i].scales, &scales, &mins, fit.max_scale, fit.max_min);
        y[i].d = fp16(fit.max_scale / 63.0);
        y[i].dmin = fp16(fit.max_min / 63.0);

        for (0..QK_K / 32) |j| {
            var sc: u8 = undefined;
            var m: u8 = undefined;
            h.getScaleMinK4(j, &y[i].scales, &sc, &m);
            const d = unfp16(y[i].d) * @as(f32, @floatFromInt(sc));
            if (d == 0) continue;
            const dm = unfp16(y[i].dmin) * @as(f32, @floatFromInt(m));
            for (0..32) |ii| {
                var l = h.nearestInt((x[32 * j + ii] + dm) / d);
                l = @max(0, @min(31, l));
                l_buf[32 * j + ii] = @intCast(l);
            }
        }

        // The fifth bit goes to qh, two bits of it per 64-weight span, so the
        // masks advance by two rather than one.
        var qh: [*]u8 = &y[i].qh;
        var ql: [*]u8 = &y[i].qs;
        @memset(y[i].qh[0..], 0);

        var m1: u8 = 1;
        var m2: u8 = 2;
        var n: usize = 0;
        while (n < QK_K) : (n += 64) {
            for (0..32) |j| {
                var l1: i32 = l_buf[n + j];
                if (l1 > 15) {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                var l2: i32 = l_buf[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[j] = @as(u8, @intCast(l1)) | (@as(u8, @intCast(l2)) << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

/// Ports `dequantize_row_q5_K` (ggml-quants.c:1731 @c1d0e7a00).
pub export fn dequantize_row_q5_K(x: [*c]const blocks.Q5_K, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    for (0..nb) |i| {
        var ql: [*]const u8 = &x[i].qs;
        const qh: [*]const u8 = &x[i].qh;

        const d = unfp16(x[i].d);
        const min = unfp16(x[i].dmin);

        var is: usize = 0;
        var sc: u8 = undefined;
        var m: u8 = undefined;
        // Named `u1`/`u2` in the C; those are integer *types* in Zig.
        var msk1: u8 = 1;
        var msk2: u8 = 2;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            h.getScaleMinK4(is + 0, &x[i].scales, &sc, &m);
            const d1 = d * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m));
            h.getScaleMinK4(is + 1, &x[i].scales, &sc, &m);
            const d2 = d * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m));
            for (0..32) |l| {
                const hi: i32 = if ((qh[l] & msk1) != 0) 16 else 0;
                y[0] = d1 * @as(f32, @floatFromInt(@as(i32, ql[l] & 0xF) + hi)) - m1;
                y += 1;
            }
            for (0..32) |l| {
                const hi: i32 = if ((qh[l] & msk2) != 0) 16 else 0;
                y[0] = d2 * @as(f32, @floatFromInt(@as(i32, ql[l] >> 4) + hi)) - m2;
                y += 1;
            }
            ql += 32;
            is += 2;
            msk1 <<= 2;
            msk2 <<= 2;
        }
    }
}

// -----------------------------------------------------------------------------
// q6_K -- 6 bits per weight, 16 sub-blocks of 16, symmetric

/// Ports `quantize_row_q6_K_ref` (ggml-quants.c:1869 @c1d0e7a00).
///
/// The most accurate K-quant: no min, and the sub-block scales are full signed
/// bytes rather than packed 6-bit fields. Each weight is four low bits in `ql`
/// and two high bits in `qh`.
pub export fn quantize_row_q6_K_ref(x_in: [*c]const f32, y: [*c]blocks.Q6_K, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var x = x_in;

    var l_buf: [QK_K]i8 = undefined;
    var scales: [QK_K / 16]f32 = undefined;

    for (0..nb) |i| {
        var max_scale: f32 = 0;
        var max_abs_scale: f32 = 0;

        for (0..QK_K / 16) |ib| {
            const scale = h.makeQxQuants(16, 32, x + 16 * ib, @ptrCast(&l_buf[16 * ib]), 1, null);
            scales[ib] = scale;

            const abs_scale = @abs(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < h.group_max_eps) {
            // Zeroes the whole block, not just the scale -- the quants must
            // read back as zero too.
            @memset(std.mem.asBytes(&y[i]), 0);
            y[i].d = fp16(0.0);
            x += QK_K;
            continue;
        }

        const iscale = -128.0 / max_scale;
        y[i].d = fp16(1 / iscale);
        for (0..QK_K / 16) |ib| {
            y[i].scales[ib] = @intCast(@min(127, h.nearestInt(iscale * scales[ib])));
        }

        for (0..QK_K / 16) |j| {
            const d = unfp16(y[i].d) * @as(f32, @floatFromInt(y[i].scales[j]));
            if (d == 0) continue;
            for (0..16) |ii| {
                var l = h.nearestInt(x[16 * j + ii] / d);
                l = @max(-32, @min(31, l));
                l_buf[16 * j + ii] = @intCast(l + 32);
            }
        }

        var ql: [*]u8 = &y[i].ql;
        var qh: [*]u8 = &y[i].qh;
        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                const q1: u8 = @as(u8, @bitCast(l_buf[j + l + 0])) & 0xF;
                const q2: u8 = @as(u8, @bitCast(l_buf[j + l + 32])) & 0xF;
                const q3: u8 = @as(u8, @bitCast(l_buf[j + l + 64])) & 0xF;
                const q4: u8 = @as(u8, @bitCast(l_buf[j + l + 96])) & 0xF;
                // Note the pairing: 1 with 3, and 2 with 4 -- not 1 with 2.
                ql[l + 0] = q1 | (q3 << 4);
                ql[l + 32] = q2 | (q4 << 4);
                qh[l] = (@as(u8, @bitCast(l_buf[j + l])) >> 4) |
                    ((@as(u8, @bitCast(l_buf[j + l + 32])) >> 4) << 2) |
                    ((@as(u8, @bitCast(l_buf[j + l + 64])) >> 4) << 4) |
                    ((@as(u8, @bitCast(l_buf[j + l + 96])) >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

/// Ports `dequantize_row_q6_K` (ggml-quants.c:1939 @c1d0e7a00).
pub export fn dequantize_row_q6_K(x: [*c]const blocks.Q6_K, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);

        var ql: [*]const u8 = &x[i].ql;
        var qh: [*]const u8 = &x[i].qh;
        var sc: [*]const i8 = &x[i].scales;

        var n: usize = 0;
        while (n < QK_K) : (n += 128) {
            for (0..32) |l| {
                const is = l / 16;
                const q1: i8 = @as(i8, @bitCast((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4))) -% 32;
                const q2: i8 = @as(i8, @bitCast((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) -% 32;
                const q3: i8 = @as(i8, @bitCast((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4))) -% 32;
                const q4: i8 = @as(i8, @bitCast((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) -% 32;
                y[l + 0] = d * @as(f32, @floatFromInt(sc[is + 0])) * @as(f32, @floatFromInt(q1));
                y[l + 32] = d * @as(f32, @floatFromInt(sc[is + 2])) * @as(f32, @floatFromInt(q2));
                y[l + 64] = d * @as(f32, @floatFromInt(sc[is + 4])) * @as(f32, @floatFromInt(q3));
                y[l + 96] = d * @as(f32, @floatFromInt(sc[is + 6])) * @as(f32, @floatFromInt(q4));
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// -----------------------------------------------------------------------------
// q8_K -- the activation side of the K-quant dot products

/// Ports `quantize_row_q8_K_ref` (ggml-quants.c:2768 @c1d0e7a00).
///
/// Never a stored weight format: this is what an activation is quantized into
/// before a dot product against a K-quant weight. Hence the `f32` scale rather
/// than an f16, and `bsums` -- the per-16 sums, cached so the dot product can
/// apply a sub-block min without walking the quants twice.
///
/// The scale divides by 127, not 128, and the C says why: 128 makes the AVX
/// implementation of `iq2_xxs` awkward.
pub export fn quantize_row_q8_K_ref(x_in: [*c]const f32, y: [*c]blocks.Q8_K, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var x = x_in;

    for (0..nb) |i| {
        var max: f32 = 0;
        var amax: f32 = 0;
        for (0..QK_K) |j| {
            const ax = @abs(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0) {
            y[i].d = 0;
            @memset(&y[i].qs, 0);
            // bsums are left as they were, which is what the C does: it
            // memsets qs and nothing else. A caller reading bsums on an
            // all-zero block reads stale data in both implementations.
            x += QK_K;
            continue;
        }

        const iscale = -127.0 / max;
        for (0..QK_K) |j| {
            const v = h.nearestInt(iscale * x[j]);
            y[i].qs[j] = @intCast(@min(127, v));
        }
        for (0..QK_K / 16) |j| {
            var sum: i32 = 0;
            for (0..16) |ii| sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = @intCast(sum);
        }
        y[i].d = 1 / iscale;
        x += QK_K;
    }
}

/// Ports `dequantize_row_q8_K` (ggml-quants.c:2807 @c1d0e7a00).
pub export fn dequantize_row_q8_K(x: [*c]const blocks.Q8_K, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    for (0..nb) |i| {
        for (0..QK_K) |j| {
            y[0] = x[i].d * @as(f32, @floatFromInt(x[i].qs[j]));
            y += 1;
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

const t = @import("testing.zig");

test {
    std.testing.refAllDecls(@This());
}

/// Same shape as `legacy.zig`'s, checking every format against every pattern.
fn checkK(
    comptime name: []const u8,
    comptime Block: type,
    quant: *const fn ([*c]const f32, [*c]Block, i64) callconv(.c) void,
    dequant: *const fn ([*c]const Block, [*c]f32, i64) callconv(.c) void,
) !void {
    for (t.all_patterns) |pattern| {
        const g = t.find(name, pattern);

        var src: [t.n_elem]f32 = undefined;
        t.fillSrc(pattern, &src);

        var buf: [t.n_elem * 4]u8 align(16) = undefined;
        @memset(&buf, 0);

        quant(&src, @ptrCast(@alignCast(&buf)), @intCast(t.n_elem));

        const used = g.row_size * t.n_rows;
        std.testing.expectEqual(g.ref.?, t.fnv(buf[0..used])) catch |e| {
            std.debug.print("{s}: quantize differs on pattern '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        var out: [t.n_elem]f32 = undefined;
        @memset(&out, 0);
        dequant(@ptrCast(@alignCast(&buf)), &out, @intCast(t.n_elem));
        std.testing.expectEqual(g.deq.?, t.fnv(std.mem.sliceAsBytes(out[0..]))) catch |e| {
            std.debug.print("{s}: dequantize differs on pattern '{s}'\n", .{ name, pattern.name() });
            return e;
        };
    }
}

test "q2_K matches the C on every pattern" {
    try checkK("Q2_K", blocks.Q2_K, quantize_row_q2_K_ref, dequantize_row_q2_K);
}

test "q3_K matches the C on every pattern" {
    try checkK("Q3_K", blocks.Q3_K, quantize_row_q3_K_ref, dequantize_row_q3_K);
}

test "q4_K matches the C on every pattern" {
    try checkK("Q4_K", blocks.Q4_K, quantize_row_q4_K_ref, dequantize_row_q4_K);
}

test "q5_K matches the C on every pattern" {
    try checkK("Q5_K", blocks.Q5_K, quantize_row_q5_K_ref, dequantize_row_q5_K);
}

test "q6_K matches the C on every pattern" {
    try checkK("Q6_K", blocks.Q6_K, quantize_row_q6_K_ref, dequantize_row_q6_K);
}

test "q8_K matches the C on every pattern" {
    try checkK("Q8_K", blocks.Q8_K, quantize_row_q8_K_ref, dequantize_row_q8_K);
}
