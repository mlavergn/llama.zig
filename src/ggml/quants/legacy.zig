//! The non-super-block quantization formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 40-618: the ten formats whose blocks stand alone rather than being grouped
//! into 256-element super-blocks. Each function names the C function it
//! replaces and the line it began at.
//!
//! # The shapes these share
//!
//! Every format here is a block of `QK` weights sharing one or two f16 scales.
//! Two families:
//!
//! - **Symmetric** (`q4_0`, `q5_0`, `q8_0`, `q1_0`, `q2_0`): one scale `d`,
//!   derived from the block's extreme value. Quantized values are signed
//!   offsets around zero.
//! - **Asymmetric** (`q4_1`, `q5_1`): a scale `d` and a minimum `m`, so the
//!   representable range is `[m, m + 15d]`. Better for weights that are not
//!   centred on zero, at the cost of a second f16 per block.
//!
//! `mxfp4` and `nvfp4` are different again: the values come from a fixed
//! 16-entry codebook and only the exponent is stored per block.
//!
//! # Halves, not pairs
//!
//! The 4-bit and 5-bit formats pack element `j` and element `j + qk/2` into one
//! byte -- the two *halves* of the block, not adjacent elements. Reading it as
//! `2j` and `2j+1` produces a plausible-looking block whose weights are
//! scrambled, and the dequantizer would have to be wrong in the same way for a
//! round-trip test to notice. That is why the tests here compare against
//! checksums taken from the C.

const std = @import("std");
const impl = @import("../impl.zig");
const c = impl.c;
const blocks = @import("blocks.zig");

const fp16 = impl.fp32ToFp16;
const unfp16 = impl.fp16ToFp32;

// -----------------------------------------------------------------------------
// q1_0 -- one sign bit per weight

/// Ports `quantize_row_q1_0_ref` (ggml-quants.c:40 @c1d0e7a00).
///
/// The scale is the block's *mean* absolute value, not its maximum: with a
/// single bit per weight there is no magnitude information to preserve, so the
/// average minimises expected error where the max would not.
pub export fn quantize_row_q1_0_ref(x: [*c]const f32, y: [*c]blocks.Q1_0, k: i64) void {
    const qk = c.QK1_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var sum_abs: f32 = 0.0;
        for (0..qk) |j| sum_abs += @abs(x[i * qk + j]);
        const d = sum_abs / @as(f32, qk);

        y[i].d = fp16(d);

        for (0..qk / 8) |j| y[i].qs[j] = 0;

        // Sign only, stored directly -- no normalisation.
        for (0..qk) |j| {
            if (x[i * qk + j] >= 0.0) {
                y[i].qs[j / 8] |= @as(u8, 1) << @intCast(j % 8);
            }
        }
    }
}

/// Ports `dequantize_row_q1_0` (ggml-quants.c:419 @c1d0e7a00).
pub export fn dequantize_row_q1_0(x: [*c]const blocks.Q1_0, y: [*c]f32, k: i64) void {
    const qk = c.QK1_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        const neg_d = -d;

        for (0..qk) |j| {
            const bit = (x[i].qs[j / 8] >> @intCast(j % 8)) & 1;
            y[i * qk + j] = if (bit != 0) d else neg_d;
        }
    }
}

// -----------------------------------------------------------------------------
// q2_0 -- two bits per weight, asymmetric codebook

/// Ports `quantize_row_q2_0_ref` (ggml-quants.c:74 @c1d0e7a00).
///
/// The two-bit code is deliberately lopsided: `00 = -1`, `01 = 0`, `10 = +1`,
/// `11 = +2`. Three of the four values are the usual signed set, and the
/// fourth extends upward rather than downward.
pub export fn quantize_row_q2_0_ref(x: [*c]const f32, y: [*c]blocks.Q2_0, k: i64) void {
    const qk = c.QK2_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..qk) |j| {
            const a = @abs(x[i * qk + j]);
            if (a > amax) amax = a;
        }
        const d = amax;
        const id: f32 = if (d > 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);

        for (0..qk / 4) |j| y[i].qs[j] = 0;

        for (0..qk) |j| {
            const w = x[i * qk + j];
            var q: i32 = impl.truncTo(i32, @round(w * id)) + 1;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            y[i].qs[j / 4] |= @as(u8, @intCast(q)) << @intCast((j % 4) * 2);
        }
    }
}

/// Ports `dequantize_row_q2_0` (ggml-quants.c:439 @c1d0e7a00).
pub export fn dequantize_row_q2_0(x: [*c]const blocks.Q2_0, y: [*c]f32, k: i64) void {
    const qk = c.QK2_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        for (0..qk) |j| {
            const q = (x[i].qs[j / 4] >> @intCast((j % 4) * 2)) & 0x03;
            // 00 = -1, 01 = 0, 10 = +1, 11 = +2
            y[i * qk + j] = @as(f32, @floatFromInt(@as(i32, q) - 1)) * d;
        }
    }
}

// -----------------------------------------------------------------------------
// q4_0 / q4_1 -- four bits per weight

/// Ports `quantize_row_q4_0_ref` (ggml-quants.c:113 @c1d0e7a00).
///
/// `d = max / -8`, where `max` is the value with the largest magnitude, sign
/// included. Dividing by *negative* eight is not a typo: it puts the extreme
/// value at code 0 and zero at code 8, so the unsigned nibble covers
/// `[-8, 7] * d` after the bias is subtracted on the way out.
pub export fn quantize_row_q4_0_ref(x: [*c]const f32, y: [*c]blocks.Q4_0, k: i64) void {
    const qk = c.QK4_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var amax: f32 = 0.0; // absolute max
        var max: f32 = 0.0;

        for (0..qk) |j| {
            const v = x[i * qk + j];
            if (amax < @abs(v)) {
                amax = @abs(v);
                max = v;
            }
        }

        const d = max / -8.0;
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);

        for (0..qk / 2) |j| {
            const x0 = x[i * qk + 0 + j] * id;
            const x1 = x[i * qk + qk / 2 + j] * id;

            const xi0: u8 = @intCast(@min(15, impl.truncTo(i8, x0 + 8.5)));
            const xi1: u8 = @intCast(@min(15, impl.truncTo(i8, x1 + 8.5)));

            y[i].qs[j] = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

/// Ports `dequantize_row_q4_0` (ggml-quants.c:459 @c1d0e7a00).
pub export fn dequantize_row_q4_0(x: [*c]const blocks.Q4_0, y: [*c]f32, k: i64) void {
    const qk = c.QK4_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        for (0..qk / 2) |j| {
            const x0: i32 = @as(i32, x[i].qs[j] & 0x0F) - 8;
            const x1: i32 = @as(i32, x[i].qs[j] >> 4) - 8;

            y[i * qk + j + 0] = @as(f32, @floatFromInt(x0)) * d;
            y[i * qk + j + qk / 2] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// Ports `quantize_row_q4_1_ref` (ggml-quants.c:150 @c1d0e7a00).
pub export fn quantize_row_q4_1_ref(x: [*c]const f32, y: [*c]blocks.Q4_1, k: i64) void {
    const qk = c.QK4_1;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var min: f32 = std.math.floatMax(f32);
        var max: f32 = -std.math.floatMax(f32);

        for (0..qk) |j| {
            const v = x[i * qk + j];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const d = (max - min) / @as(f32, (1 << 4) - 1);
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);
        y[i].m = fp16(min);

        for (0..qk / 2) |j| {
            const x0 = (x[i * qk + 0 + j] - min) * id;
            const x1 = (x[i * qk + qk / 2 + j] - min) * id;

            const xi0: u8 = @intCast(@min(15, impl.truncTo(i8, x0 + 0.5)));
            const xi1: u8 = @intCast(@min(15, impl.truncTo(i8, x1 + 0.5)));

            y[i].qs[j] = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

/// Ports `dequantize_row_q4_1` (ggml-quants.c:479 @c1d0e7a00).
pub export fn dequantize_row_q4_1(x: [*c]const blocks.Q4_1, y: [*c]f32, k: i64) void {
    const qk = c.QK4_1;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        const m = unfp16(x[i].m);

        for (0..qk / 2) |j| {
            const x0: i32 = @intCast(x[i].qs[j] & 0x0F);
            const x1: i32 = @intCast(x[i].qs[j] >> 4);

            y[i * qk + j + 0] = @as(f32, @floatFromInt(x0)) * d + m;
            y[i * qk + j + qk / 2] = @as(f32, @floatFromInt(x1)) * d + m;
        }
    }
}

// -----------------------------------------------------------------------------
// q5_0 / q5_1 -- five bits per weight
//
// The low four bits live in `qs` exactly as in the 4-bit formats; the fifth bit
// of all `qk` weights is gathered into a single 32-bit `qh`, bit `j` for the
// first half and bit `j + qk/2` for the second.

/// Ports `quantize_row_q5_0_ref` (ggml-quants.c:187 @c1d0e7a00).
pub export fn quantize_row_q5_0_ref(x: [*c]const f32, y: [*c]blocks.Q5_0, k: i64) void {
    const qk = c.QK5_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        var max: f32 = 0.0;

        for (0..qk) |j| {
            const v = x[i * qk + j];
            if (amax < @abs(v)) {
                amax = @abs(v);
                max = v;
            }
        }

        const d = max / -16.0;
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);

        var qh: u32 = 0;

        for (0..qk / 2) |j| {
            const x0 = x[i * qk + 0 + j] * id;
            const x1 = x[i * qk + qk / 2 + j] * id;

            const xi0: u8 = @intCast(@min(31, impl.truncTo(i8, x0 + 16.5)));
            const xi1: u8 = @intCast(@min(31, impl.truncTo(i8, x1 + 16.5)));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // The fifth bit, moved to its place in qh.
            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j + 0);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + qk / 2);
        }

        @memcpy(&y[i].qh, std.mem.asBytes(&qh));
    }
}

/// Ports `dequantize_row_q5_0` (ggml-quants.c:500 @c1d0e7a00).
pub export fn dequantize_row_q5_0(x: [*c]const blocks.Q5_0, y: [*c]f32, k: i64) void {
    const qk = c.QK5_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = unfp16(x[i].d);

        var qh: u32 = undefined;
        @memcpy(std.mem.asBytes(&qh), &x[i].qh);

        for (0..qk / 2) |j| {
            // Shifted straight into bit 4, which is where the fifth bit
            // belongs once it rejoins the low nibble.
            const xh_0: u8 = @truncate((qh >> @intCast(j + 0)) << 4);
            const xh_1: u8 = @truncate(qh >> @intCast(j + 12));

            const x0: i32 = @as(i32, (x[i].qs[j] & 0x0F) | (xh_0 & 0x10)) - 16;
            const x1: i32 = @as(i32, (x[i].qs[j] >> 4) | (xh_1 & 0x10)) - 16;

            y[i * qk + j + 0] = @as(f32, @floatFromInt(x0)) * d;
            y[i * qk + j + qk / 2] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// Ports `quantize_row_q5_1_ref` (ggml-quants.c:231 @c1d0e7a00).
pub export fn quantize_row_q5_1_ref(x: [*c]const f32, y: [*c]blocks.Q5_1, k: i64) void {
    const qk = c.QK5_1;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var min: f32 = std.math.floatMax(f32);
        var max: f32 = -std.math.floatMax(f32);

        for (0..qk) |j| {
            const v = x[i * qk + j];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const d = (max - min) / @as(f32, (1 << 5) - 1);
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);
        y[i].m = fp16(min);

        var qh: u32 = 0;

        for (0..qk / 2) |j| {
            const x0 = (x[i * qk + 0 + j] - min) * id;
            const x1 = (x[i * qk + qk / 2 + j] - min) * id;

            // Note this one casts to unsigned, not int8 as q5_0 does: the
            // values are already non-negative because `min` was subtracted.
            const xi0 = impl.truncTo(u8, x0 + 0.5);
            const xi1 = impl.truncTo(u8, x1 + 0.5);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j + 0);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + qk / 2);
        }

        @memcpy(&y[i].qh, std.mem.asBytes(&qh));
    }
}

/// Ports `dequantize_row_q5_1` (ggml-quants.c:526 @c1d0e7a00).
pub export fn dequantize_row_q5_1(x: [*c]const blocks.Q5_1, y: [*c]f32, k: i64) void {
    const qk = c.QK5_1;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        const m = unfp16(x[i].m);

        var qh: u32 = undefined;
        @memcpy(std.mem.asBytes(&qh), &x[i].qh);

        for (0..qk / 2) |j| {
            const xh_0: u8 = @truncate((qh >> @intCast(j + 0)) << 4);
            const xh_1: u8 = @truncate(qh >> @intCast(j + 12));

            const x0: i32 = @intCast((x[i].qs[j] & 0x0F) | (xh_0 & 0x10));
            const x1: i32 = @intCast((x[i].qs[j] >> 4) | (xh_1 & 0x10));

            y[i * qk + j + 0] = @as(f32, @floatFromInt(x0)) * d + m;
            y[i * qk + j + qk / 2] = @as(f32, @floatFromInt(x1)) * d + m;
        }
    }
}

// -----------------------------------------------------------------------------
// q8_0 / q8_1 -- eight bits per weight

/// Ports `quantize_row_q8_0_ref` (ggml-quants.c:276 @c1d0e7a00).
pub export fn quantize_row_q8_0_ref(x: [*c]const f32, y: [*c]blocks.Q8_0, k: i64) void {
    const qk = c.QK8_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..qk) |j| amax = @max(amax, @abs(x[i * qk + j]));

        const d = amax / @as(f32, (1 << 7) - 1);
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);

        for (0..qk) |j| {
            // `roundf` then an implicit narrowing to int8_t in the C.
            y[i].qs[j] = impl.truncTo(i8, @round(x[i * qk + j] * id));
        }
    }
}

/// Ports `dequantize_row_q8_0` (ggml-quants.c:553 @c1d0e7a00).
pub export fn dequantize_row_q8_0(x: [*c]const blocks.Q8_0, y: [*c]f32, k: i64) void {
    const qk = c.QK8_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        for (0..qk) |j| {
            y[i * qk + j] = @as(f32, @floatFromInt(x[i].qs[j])) * d;
        }
    }
}

/// Ports `quantize_row_q8_1_ref` (ggml-quants.c:302 @c1d0e7a00).
///
/// Same quantization as `q8_0` plus `s`, the sum of the quantized values times
/// the scale. That sum lets a dot product against an asymmetric format recover
/// the `m * sum(q)` term without a second pass. There is no
/// `dequantize_row_q8_1`: this format is an intermediate the dot-product
/// kernels produce, never a stored weight.
pub export fn quantize_row_q8_1_ref(x: [*c]const f32, y: [*c]blocks.Q8_1, k: i64) void {
    const qk = c.QK8_1;
    comptime std.debug.assert(qk == 32);
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..qk) |j| amax = @max(amax, @abs(x[i * qk + j]));

        const d = amax / @as(f32, (1 << 7) - 1);
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);

        var sum: i32 = 0;

        // Walks the two halves together, as the C does. The result is the same
        // as a single pass here, but the loop shape is kept so a reader
        // comparing the two files does not have to prove that.
        for (0..qk / 2) |j| {
            const v0 = x[i * qk + j] * id;
            const v1 = x[i * qk + qk / 2 + j] * id;

            y[i].qs[j] = impl.truncTo(i8, @round(v0));
            y[i].qs[qk / 2 + j] = impl.truncTo(i8, @round(v1));

            sum += y[i].qs[j];
            sum += y[i].qs[qk / 2 + j];
        }

        y[i].s = fp16(@as(f32, @floatFromInt(sum)) * d);
    }
}

// -----------------------------------------------------------------------------
// mxfp4 / nvfp4 -- codebook formats
//
// The quantized value is an index into `kvalues_mxfp4`, a fixed 16-entry table
// of E2M1 values doubled. Only an exponent is stored per block, so the scale
// costs one byte rather than an f16.

/// Ports `best_index_mxfp4` (ggml-quants.c:337 @c1d0e7a00).
///
/// Linear search over all sixteen codebook entries. The table is not monotonic
/// in a way that would let a comparison shortcut work, so the C searches, and
/// so does this.
fn bestIndexMxfp4(x: f32, e: f32) u8 {
    var best_index: u8 = 0;
    var best_err = @abs(@as(f32, @floatFromInt(c.kvalues_mxfp4[0])) * e - x);
    for (1..16) |i| {
        const err = @abs(@as(f32, @floatFromInt(c.kvalues_mxfp4[i])) * e - x);
        if (err < best_err) {
            best_index = @intCast(i);
            best_err = err;
        }
    }
    return best_index;
}

/// Ports `quantize_row_mxfp4_ref` (ggml-quants.c:350 @c1d0e7a00).
pub export fn quantize_row_mxfp4_ref(x: [*c]const f32, y: [*c]blocks.MXFP4, k: i64) void {
    const qk = c.QK_MXFP4;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..qk) |j| {
            const v = x[i * qk + j];
            if (amax < @abs(v)) amax = @abs(v);
        }

        // The `- 2` biases the exponent so the codebook's largest value (6,
        // doubled to 12) covers `amax` rather than clipping it.
        const e: u8 = if (amax > 0.0)
            impl.truncTo(u8, @floor(@log2(amax)) - 2 + 127)
        else
            0;

        const d = impl.e8m0ToFp32Half(e);

        y[i].e = e;

        for (0..qk / 2) |j| {
            const x0 = bestIndexMxfp4(x[i * qk + 0 + j], d);
            const x1 = bestIndexMxfp4(x[i * qk + qk / 2 + j], d);

            y[i].qs[j] = x0;
            y[i].qs[j] |= x1 << 4;
        }
    }
}

/// Ports `dequantize_row_mxfp4` (ggml-quants.c:569 @c1d0e7a00).
pub export fn dequantize_row_mxfp4(x: [*c]const blocks.MXFP4, y: [*c]f32, k: i64) void {
    const qk = c.QK_MXFP4;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = impl.e8m0ToFp32Half(x[i].e);

        for (0..qk / 2) |j| {
            const x0 = c.kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const x1 = c.kvalues_mxfp4[x[i].qs[j] >> 4];

            y[i * qk + j + 0] = @as(f32, @floatFromInt(x0)) * d;
            y[i * qk + j + qk / 2] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// Ports `quantize_row_nvfp4_ref` (ggml-quants.c:384 @c1d0e7a00).
///
/// Same codebook as `mxfp4`, but the 64-element block is split into four
/// 16-element sub-blocks each with its own UE4M3 scale, so a block with one
/// large outlier does not flatten the rest.
pub export fn quantize_row_nvfp4_ref(x: [*c]const f32, y: [*c]blocks.NVFP4, k: i64) void {
    const qk = c.QK_NVFP4;
    const qk_sub = c.QK_NVFP4_SUB;
    const n_sub = qk / qk_sub;

    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        for (0..n_sub) |s| {
            const xb = x + i * qk + s * qk_sub;

            var amax: f32 = 0.0;
            for (0..qk_sub) |j| {
                if (amax < @abs(xb[j])) amax = @abs(xb[j]);
            }

            // amax / 6 maps the codebook's top E2M1 value onto amax.
            const ue = impl.fp32ToUe4m3(amax / 6.0);
            y[i].d[s] = ue;
            const d = impl.ue4m3ToFp32(ue);

            for (0..qk_sub / 2) |j| {
                const x0 = bestIndexMxfp4(xb[0 + j], d);
                const x1 = bestIndexMxfp4(xb[qk_sub / 2 + j], d);

                y[i].qs[s * (qk_sub / 2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

/// Ports `dequantize_row_nvfp4` (ggml-quants.c:589 @c1d0e7a00).
pub export fn dequantize_row_nvfp4(x: [*c]const blocks.NVFP4, y: [*c]f32, k: i64) void {
    const qk = c.QK_NVFP4;
    const qk_sub = c.QK_NVFP4_SUB;
    const n_sub = qk / qk_sub;

    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        for (0..n_sub) |s| {
            const d = impl.ue4m3ToFp32(x[i].d[s]);
            const yb = y + i * qk + s * qk_sub;

            for (0..qk_sub / 2) |j| {
                const v0 = c.kvalues_mxfp4[x[i].qs[s * (qk_sub / 2) + j] & 0x0F];
                const v1 = c.kvalues_mxfp4[x[i].qs[s * (qk_sub / 2) + j] >> 4];

                yb[j + 0] = @as(f32, @floatFromInt(v0)) * d;
                yb[j + qk_sub / 2] = @as(f32, @floatFromInt(v1)) * d;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests
//
// Every case compares against a checksum captured from the C by
// `scripts/quants-golden`. A round-trip test would be easier to write and
// would pass with the quantizer and dequantizer broken in matching ways.

const t = @import("testing.zig");
const golden = @import("golden.zig");

test {
    std.testing.refAllDecls(@This());
}

/// Runs one format's reference quantizer and dequantizer against its goldens,
/// for every input pattern.
///
/// The patterns matter more than the count of them: `zeros` is the only case
/// that reaches the `id = d ? 1/d : 0` guard every symmetric quantizer opens
/// with, and `constant` the only one where an asymmetric format sees
/// `min == max`. Random input alone never takes either branch, and deleting
/// the guard from `quantize_row_q4_0_ref` is caught by `zeros` alone.
///
/// **`constant` cannot prove the same for the asymmetric formats**, and that
/// is worth knowing rather than assuming. With `min == max` the scale is zero,
/// so a missing guard computes `(x - min) * inf`, which is `0 * inf` = NaN;
/// `impl.truncTo` maps NaN to 0, and the correct `id = 0` path also yields 0.
/// The two coincide for every input, so no golden can separate them. The guard
/// is still right to have -- it is what keeps the value defined rather than
/// relying on that coincidence -- but this test does not defend it.
///
/// Parameters:
/// - `name`: the type's name in `golden.zig`.
/// - `Block`: the C block struct.
/// - `quant`: the ported `quantize_row_*_ref`.
/// - `dequant`: the ported `dequantize_row_*`, or null when the C has none.
fn checkFormat(
    comptime name: []const u8,
    comptime Block: type,
    quant: *const fn ([*c]const f32, [*c]Block, i64) callconv(.c) void,
    comptime dequant: ?*const fn ([*c]const Block, [*c]f32, i64) callconv(.c) void,
) !void {
    for (t.all_patterns) |pattern| {
        const g = t.find(name, pattern);

        var src: [t.n_elem]f32 = undefined;
        t.fillSrc(pattern, &src);

        // Generous: the widest format here is under 2 bytes per weight.
        var buf: [t.n_elem * 4]u8 align(16) = undefined;
        @memset(&buf, 0);

        quant(&src, @ptrCast(@alignCast(&buf)), @intCast(t.n_elem));

        // Row size comes from the golden record rather than the traits table,
        // so this test links without `ggml-quants.c` and can therefore run
        // before the whole translation unit is ported.
        const used = g.row_size * t.n_rows;
        std.testing.expectEqual(g.ref.?, t.fnv(buf[0..used])) catch |e| {
            std.debug.print("{s}: quantize differs on pattern '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        if (dequant) |dq| {
            var out: [t.n_elem]f32 = undefined;
            @memset(&out, 0);
            dq(@ptrCast(@alignCast(&buf)), &out, @intCast(t.n_elem));
            std.testing.expectEqual(g.deq.?, t.fnv(std.mem.sliceAsBytes(out[0..]))) catch |e| {
                std.debug.print("{s}: dequantize differs on pattern '{s}'\n", .{ name, pattern.name() });
                return e;
            };
        }
    }
}

test "q1_0 matches the C, quantize and dequantize" {
    try checkFormat("Q1_0", blocks.Q1_0, quantize_row_q1_0_ref, dequantize_row_q1_0);
}

test "q2_0 matches the C, quantize and dequantize" {
    try checkFormat("Q2_0", blocks.Q2_0, quantize_row_q2_0_ref, dequantize_row_q2_0);
}

test "q4_0 matches the C, quantize and dequantize" {
    try checkFormat("Q4_0", blocks.Q4_0, quantize_row_q4_0_ref, dequantize_row_q4_0);
}

test "q4_1 matches the C, quantize and dequantize" {
    try checkFormat("Q4_1", blocks.Q4_1, quantize_row_q4_1_ref, dequantize_row_q4_1);
}

test "q5_0 matches the C, quantize and dequantize" {
    try checkFormat("Q5_0", blocks.Q5_0, quantize_row_q5_0_ref, dequantize_row_q5_0);
}

test "q5_1 matches the C, quantize and dequantize" {
    try checkFormat("Q5_1", blocks.Q5_1, quantize_row_q5_1_ref, dequantize_row_q5_1);
}

test "q8_0 matches the C, quantize and dequantize" {
    try checkFormat("Q8_0", blocks.Q8_0, quantize_row_q8_0_ref, dequantize_row_q8_0);
}

test "q8_1 matches the C -- quantize only, the format has no dequantizer" {
    try checkFormat("Q8_1", blocks.Q8_1, quantize_row_q8_1_ref, null);
}

test "mxfp4 matches the C, quantize and dequantize" {
    try checkFormat("MXFP4", blocks.MXFP4, quantize_row_mxfp4_ref, dequantize_row_mxfp4);
}

test "nvfp4 matches the C, quantize and dequantize" {
    try checkFormat("NVFP4", blocks.NVFP4, quantize_row_nvfp4_ref, dequantize_row_nvfp4);
}

test "the four-bit formats pack halves of the block, not adjacent elements" {
    // A direct check of the packing rule, independent of the checksums, because
    // getting it wrong is the single easiest way to produce a file that looks
    // right and decodes to scrambled weights.
    var src: [c.QK4_0]f32 = @splat(0.0);
    // Element 0 large positive, element qk/2 large negative: after packing they
    // must land in the low and high nibble of the *same* byte.
    src[0] = 1.0;
    src[c.QK4_0 / 2] = -1.0;

    var block: blocks.Q4_0 = undefined;
    quantize_row_q4_0_ref(&src, @ptrCast(&block), c.QK4_0);

    var out: [c.QK4_0]f32 = undefined;
    dequantize_row_q4_0(@ptrCast(&block), &out, c.QK4_0);

    try std.testing.expect(out[0] > 0.0);
    try std.testing.expect(out[c.QK4_0 / 2] < 0.0);
}
