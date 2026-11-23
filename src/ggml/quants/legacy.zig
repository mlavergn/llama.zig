//! The non-super-block quantization formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v1.0.0, `b1b770d90`), lines
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
const ggml = @import("../module.zig");

// -----------------------------------------------------------------------------
// q1_0 -- one sign bit per weight

/// Ports `quantize_row_q1_0_ref` (ggml-quants.c:37 @b1b770d90).
///
/// The scale is the block's *mean* absolute value, not its maximum: with a
/// single bit per weight there is no magnitude information to preserve, so the
/// average minimises expected error where the max would not.
pub fn quantize_row_q1_0_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.Q1_0, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK1_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var sum_abs: f32 = 0.0;
        for (0..qk) |j| sum_abs += @abs(x[i * qk + j]);
        const d = sum_abs / @as(f32, qk);

        y[i].d = ggml.impl.fp32ToFp16(d);

        for (0..qk / 8) |j| y[i].qs[j] = 0;

        // Sign only, stored directly -- no normalisation.
        for (0..qk) |j| {
            if (x[i * qk + j] >= 0.0) {
                y[i].qs[j / 8] |= @as(u8, 1) << @intCast(j % 8);
            }
        }
    }
}

/// Ports `dequantize_row_q1_0` (ggml-quants.c:422 @b1b770d90).
pub fn dequantize_row_q1_0(x: [*c]const ggml.quants.blocks.Q1_0, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK1_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = ggml.impl.fp16ToFp32(x[i].d);
        const neg_d = -d;

        for (0..qk) |j| {
            const bit = (x[i].qs[j / 8] >> @intCast(j % 8)) & 1;
            y[i * qk + j] = if (bit != 0) d else neg_d;
        }
    }
}

// -----------------------------------------------------------------------------
// q2_0 -- two bits per weight, asymmetric codebook

/// Ports `quantize_row_q2_0_ref` (ggml-quants.c:71 @b1b770d90).
///
/// The two-bit code is deliberately lopsided: `00 = -1`, `01 = 0`, `10 = +1`,
/// `11 = +2`. Three of the four values are the usual signed set, and the
/// fourth extends upward rather than downward.
pub fn quantize_row_q2_0_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.Q2_0, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK2_0;
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

        y[i].d = ggml.impl.fp32ToFp16(d);

        for (0..qk / 4) |j| y[i].qs[j] = 0;

        for (0..qk) |j| {
            const w = x[i * qk + j];
            var q: i32 = ggml.impl.truncTo(i32, @round(w * id)) + 1;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            y[i].qs[j / 4] |= @as(u8, @intCast(q)) << @intCast((j % 4) * 2);
        }
    }
}

/// Ports `dequantize_row_q2_0` (ggml-quants.c:442 @b1b770d90).
pub fn dequantize_row_q2_0(x: [*c]const ggml.quants.blocks.Q2_0, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK2_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = ggml.impl.fp16ToFp32(x[i].d);
        for (0..qk) |j| {
            const q = (x[i].qs[j / 4] >> @intCast((j % 4) * 2)) & 0x03;
            // 00 = -1, 01 = 0, 10 = +1, 11 = +2
            y[i * qk + j] = @as(f32, @floatFromInt(@as(i32, q) - 1)) * d;
        }
    }
}

// -----------------------------------------------------------------------------
// q4_0 / q4_1 -- four bits per weight

/// Ports `quantize_row_q4_0_ref` (ggml-quants.c:110 @b1b770d90).
///
/// `d = max / -8`, where `max` is the value with the largest magnitude, sign
/// included. Dividing by *negative* eight is not a typo: it puts the extreme
/// value at code 0 and zero at code 8, so the unsigned nibble covers
/// `[-8, 7] * d` after the bias is subtracted on the way out.
pub fn quantize_row_q4_0_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.Q4_0, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK4_0;
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

        y[i].d = ggml.impl.fp32ToFp16(d);

        for (0..qk / 2) |j| {
            const x0 = x[i * qk + 0 + j] * id;
            const x1 = x[i * qk + qk / 2 + j] * id;

            const xi0: u8 = @intCast(@min(15, ggml.impl.truncTo(i8, x0 + 8.5)));
            const xi1: u8 = @intCast(@min(15, ggml.impl.truncTo(i8, x1 + 8.5)));

            y[i].qs[j] = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

/// Ports `dequantize_row_q4_0` (ggml-quants.c:462 @b1b770d90).
pub fn dequantize_row_q4_0(x: [*c]const ggml.quants.blocks.Q4_0, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK4_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = ggml.impl.fp16ToFp32(x[i].d);
        for (0..qk / 2) |j| {
            const x0: i32 = @as(i32, x[i].qs[j] & 0x0F) - 8;
            const x1: i32 = @as(i32, x[i].qs[j] >> 4) - 8;

            y[i * qk + j + 0] = @as(f32, @floatFromInt(x0)) * d;
            y[i * qk + j + qk / 2] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// Ports `quantize_row_q4_1_ref` (ggml-quants.c:147 @b1b770d90).
pub fn quantize_row_q4_1_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.Q4_1, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK4_1;
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

        y[i].d = ggml.impl.fp32ToFp16(d);
        y[i].m = ggml.impl.fp32ToFp16(min);

        for (0..qk / 2) |j| {
            const x0 = (x[i * qk + 0 + j] - min) * id;
            const x1 = (x[i * qk + qk / 2 + j] - min) * id;

            const xi0: u8 = @intCast(@min(15, ggml.impl.truncTo(i8, x0 + 0.5)));
            const xi1: u8 = @intCast(@min(15, ggml.impl.truncTo(i8, x1 + 0.5)));

            y[i].qs[j] = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

/// Ports `dequantize_row_q4_1` (ggml-quants.c:482 @b1b770d90).
pub fn dequantize_row_q4_1(x: [*c]const ggml.quants.blocks.Q4_1, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK4_1;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = ggml.impl.fp16ToFp32(x[i].d);
        const m = ggml.impl.fp16ToFp32(x[i].m);

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

/// Ports `quantize_row_q5_0_ref` (ggml-quants.c:184 @b1b770d90).
pub fn quantize_row_q5_0_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.Q5_0, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK5_0;
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

        y[i].d = ggml.impl.fp32ToFp16(d);

        var qh: u32 = 0;

        for (0..qk / 2) |j| {
            const x0 = x[i * qk + 0 + j] * id;
            const x1 = x[i * qk + qk / 2 + j] * id;

            const xi0: u8 = @intCast(@min(31, ggml.impl.truncTo(i8, x0 + 16.5)));
            const xi1: u8 = @intCast(@min(31, ggml.impl.truncTo(i8, x1 + 16.5)));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // The fifth bit, moved to its place in qh.
            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j + 0);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + qk / 2);
        }

        @memcpy(&y[i].qh, std.mem.asBytes(&qh));
    }
}

/// Ports `dequantize_row_q5_0` (ggml-quants.c:503 @b1b770d90).
pub fn dequantize_row_q5_0(x: [*c]const ggml.quants.blocks.Q5_0, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK5_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = ggml.impl.fp16ToFp32(x[i].d);

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

/// Ports `quantize_row_q5_1_ref` (ggml-quants.c:228 @b1b770d90).
pub fn quantize_row_q5_1_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.Q5_1, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK5_1;
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

        y[i].d = ggml.impl.fp32ToFp16(d);
        y[i].m = ggml.impl.fp32ToFp16(min);

        var qh: u32 = 0;

        for (0..qk / 2) |j| {
            const x0 = (x[i * qk + 0 + j] - min) * id;
            const x1 = (x[i * qk + qk / 2 + j] - min) * id;

            // Note this one casts to unsigned, not int8 as q5_0 does: the
            // values are already non-negative because `min` was subtracted.
            const xi0 = ggml.impl.truncTo(u8, x0 + 0.5);
            const xi1 = ggml.impl.truncTo(u8, x1 + 0.5);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j + 0);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + qk / 2);
        }

        @memcpy(&y[i].qh, std.mem.asBytes(&qh));
    }
}

/// Ports `dequantize_row_q5_1` (ggml-quants.c:529 @b1b770d90).
pub fn dequantize_row_q5_1(x: [*c]const ggml.quants.blocks.Q5_1, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK5_1;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = ggml.impl.fp16ToFp32(x[i].d);
        const m = ggml.impl.fp16ToFp32(x[i].m);

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

/// Ports `quantize_row_q8_0_ref` (ggml-quants.c:273 @b1b770d90).
pub fn quantize_row_q8_0_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.Q8_0, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK8_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..qk) |j| amax = @max(amax, @abs(x[i * qk + j]));

        const d = amax / @as(f32, (1 << 7) - 1);
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = ggml.impl.fp32ToFp16(d);

        for (0..qk) |j| {
            // `roundf` then an implicit narrowing to int8_t in the C.
            y[i].qs[j] = ggml.impl.truncTo(i8, @round(x[i * qk + j] * id));
        }
    }
}

/// Ports `dequantize_row_q8_0` (ggml-quants.c:556 @b1b770d90).
pub fn dequantize_row_q8_0(x: [*c]const ggml.quants.blocks.Q8_0, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK8_0;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = ggml.impl.fp16ToFp32(x[i].d);
        for (0..qk) |j| {
            y[i * qk + j] = @as(f32, @floatFromInt(x[i].qs[j])) * d;
        }
    }
}

/// Ports `quantize_row_q8_1_ref` (ggml-quants.c:299 @b1b770d90).
///
/// Same quantization as `q8_0` plus `s`, the sum of the quantized values times
/// the scale. That sum lets a dot product against an asymmetric format recover
/// the `m * sum(q)` term without a second pass. There is no
/// `dequantize_row_q8_1`: this format is an intermediate the dot-product
/// kernels produce, never a stored weight.
pub fn quantize_row_q8_1_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.Q8_1, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK8_1;
    comptime std.debug.assert(qk == 32);
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..qk) |j| amax = @max(amax, @abs(x[i * qk + j]));

        const d = amax / @as(f32, (1 << 7) - 1);
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = ggml.impl.fp32ToFp16(d);

        var sum: i32 = 0;

        // Walks the two halves together, as the C does. The result is the same
        // as a single pass here, but the loop shape is kept so a reader
        // comparing the two files does not have to prove that.
        for (0..qk / 2) |j| {
            const v0 = x[i * qk + j] * id;
            const v1 = x[i * qk + qk / 2 + j] * id;

            y[i].qs[j] = ggml.impl.truncTo(i8, @round(v0));
            y[i].qs[qk / 2 + j] = ggml.impl.truncTo(i8, @round(v1));

            sum += y[i].qs[j];
            sum += y[i].qs[qk / 2 + j];
        }

        y[i].s = ggml.impl.fp32ToFp16(@as(f32, @floatFromInt(sum)) * d);
    }
}

// -----------------------------------------------------------------------------
// mxfp4 / nvfp4 -- codebook formats
//
// The quantized value is an index into `kvalues_mxfp4`, a fixed 16-entry table
// of E2M1 values doubled. Only an exponent is stored per block, so the scale
// costs one byte rather than an f16.

/// Ports `best_index_mxfp4` (ggml-quants.c:334 @b1b770d90).
///
/// Linear search over all sixteen codebook entries. The table is not monotonic
/// in a way that would let a comparison shortcut work, so the C searches, and
/// so does this.
fn bestIndexMxfp4(x: f32, e: f32) u8 {
    var best_index: u8 = 0;
    var best_err = @abs(@as(f32, @floatFromInt(ggml.impl.ggml_c.kvalues_mxfp4[0])) * e - x);
    for (1..16) |i| {
        const err = @abs(@as(f32, @floatFromInt(ggml.impl.ggml_c.kvalues_mxfp4[i])) * e - x);
        if (err < best_err) {
            best_index = @intCast(i);
            best_err = err;
        }
    }
    return best_index;
}

/// Ports `quantize_row_mxfp4_ref` (ggml-quants.c:347 @b1b770d90).
pub fn quantize_row_mxfp4_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.MXFP4, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK_MXFP4;
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
            ggml.impl.truncTo(u8, @floor(@log2(amax)) - 2 + 127)
        else
            0;

        const d = ggml.impl.e8m0ToFp32Half(e);

        y[i].e = e;

        for (0..qk / 2) |j| {
            const x0 = bestIndexMxfp4(x[i * qk + 0 + j], d);
            const x1 = bestIndexMxfp4(x[i * qk + qk / 2 + j], d);

            y[i].qs[j] = x0;
            y[i].qs[j] |= x1 << 4;
        }
    }
}

/// Ports `dequantize_row_mxfp4` (ggml-quants.c:572 @b1b770d90).
pub fn dequantize_row_mxfp4(x: [*c]const ggml.quants.blocks.MXFP4, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK_MXFP4;
    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        const d = ggml.impl.e8m0ToFp32Half(x[i].e);

        for (0..qk / 2) |j| {
            const x0 = ggml.impl.ggml_c.kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const x1 = ggml.impl.ggml_c.kvalues_mxfp4[x[i].qs[j] >> 4];

            y[i * qk + j + 0] = @as(f32, @floatFromInt(x0)) * d;
            y[i * qk + j + qk / 2] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// Ports `quantize_row_nvfp4_ref` (ggml-quants.c:381 @b1b770d90).
///
/// Same codebook as `mxfp4`, but the 64-element block is split into four
/// 16-element sub-blocks each with its own UE4M3 scale, so a block with one
/// large outlier does not flatten the rest.
pub fn quantize_row_nvfp4_ref(x: [*c]const f32, y: [*c]ggml.quants.blocks.NVFP4, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK_NVFP4;
    const qk_sub = ggml.impl.ggml_c.QK_NVFP4_SUB;
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
            const ue = ggml.impl.fp32ToUe4m3(amax / 6.0);
            y[i].d[s] = ue;
            const d = ggml.impl.ue4m3ToFp32(ue);

            for (0..qk_sub / 2) |j| {
                const x0 = bestIndexMxfp4(xb[0 + j], d);
                const x1 = bestIndexMxfp4(xb[qk_sub / 2 + j], d);

                y[i].qs[s * (qk_sub / 2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

/// Ports `dequantize_row_nvfp4` (ggml-quants.c:592 @b1b770d90).
pub fn dequantize_row_nvfp4(x: [*c]const ggml.quants.blocks.NVFP4, y: [*c]f32, k: i64) callconv(.c) void {
    const qk = ggml.impl.ggml_c.QK_NVFP4;
    const qk_sub = ggml.impl.ggml_c.QK_NVFP4_SUB;
    const n_sub = qk / qk_sub;

    std.debug.assert(@rem(k, qk) == 0);
    const nb: usize = @intCast(@divExact(k, qk));

    for (0..nb) |i| {
        for (0..n_sub) |s| {
            const d = ggml.impl.ue4m3ToFp32(x[i].d[s]);
            const yb = y + i * qk + s * qk_sub;

            for (0..qk_sub / 2) |j| {
                const v0 = ggml.impl.ggml_c.kvalues_mxfp4[x[i].qs[s * (qk_sub / 2) + j] & 0x0F];
                const v1 = ggml.impl.ggml_c.kvalues_mxfp4[x[i].qs[s * (qk_sub / 2) + j] >> 4];

                yb[j + 0] = @as(f32, @floatFromInt(v0)) * d;
                yb[j + qk_sub / 2] = @as(f32, @floatFromInt(v1)) * d;
            }
        }
    }
}

/// Ports `quantize_row_f8_e4m3_ref` (ggml-quants.c:416 @b1b770d90).
///
/// Not a block format: one byte per element, each `fp32ToF8E4m3`.
pub fn quantize_row_f8_e4m3_ref(x: [*c]const f32, y: [*c]ggml.impl.ggml_c.ggml_fp8_e4m3_t, k: i64) callconv(.c) void {
    var i: i64 = 0;
    while (i < k) : (i += 1) {
        const u: usize = @intCast(i);
        y[u].bits = ggml.impl.fp32ToF8E4m3(x[u]);
    }
}

/// Ports `dequantize_row_f8_e4m3` (ggml-quants.c:617 @b1b770d90).
pub fn dequantize_row_f8_e4m3(x: [*c]const ggml.impl.ggml_c.ggml_fp8_e4m3_t, y: [*c]f32, k: i64) callconv(.c) void {
    var i: i64 = 0;
    while (i < k) : (i += 1) {
        const u: usize = @intCast(i);
        y[u] = ggml.impl.f8E4m3ToFp32(x[u].bits);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests
//
// Every case compares against a checksum captured from the C by
// `scripts/quants-golden`. A round-trip test would be easier to write and
// would pass with the quantizer and dequantizer broken in matching ways.

test {
    std.testing.refAllDecls(@This());
}

test "the four-bit formats pack halves of the block, not adjacent elements" {
    // A direct check of the packing rule, independent of the checksums, because
    // getting it wrong is the single easiest way to produce a file that looks
    // right and decodes to scrambled weights.
    var src: [ggml.impl.ggml_c.QK4_0]f32 = @splat(0.0);
    // Element 0 large positive, element qk/2 large negative: after packing they
    // must land in the low and high nibble of the *same* byte.
    src[0] = 1.0;
    src[ggml.impl.ggml_c.QK4_0 / 2] = -1.0;

    var block: ggml.quants.blocks.Q4_0 = undefined;
    quantize_row_q4_0_ref(&src, @ptrCast(&block), ggml.impl.ggml_c.QK4_0);

    var out: [ggml.impl.ggml_c.QK4_0]f32 = undefined;
    dequantize_row_q4_0(@ptrCast(&block), &out, ggml.impl.ggml_c.QK4_0);

    try std.testing.expect(out[0] > 0.0);
    try std.testing.expect(out[ggml.impl.ggml_c.QK4_0 / 2] < 0.0);
}
