//! Checking quantized data for infinities and NaNs.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 5330-5700. Each function names the C function it replaces and the line it
//! began at.
//!
//! # What it checks, and what it does not
//!
//! Only the **scales**. A block's quantized values are integers and cannot be
//! NaN; its scale is an f16 (or an f32, or an E8M0 exponent) and can be. One
//! bad scale poisons every weight in its block, and a NaN reaching the model
//! turns the whole output to NaN with nothing to say where it came from -- so
//! this is checked at load, once, rather than debugged later.
//!
//! The quantized values themselves are never examined. Neither is anything
//! about whether the data is *sensible*: a well-formed block of noise passes.
//!
//! # The SIMD paths are not ported
//!
//! The C has AVX2 and NEON fast paths for the `F16` and `F32` cases that scan
//! sixteen or eight values at a time and fall back to the scalar loop when a
//! candidate is found. They compute the same answer; only the speed differs,
//! and a vectorised rewrite would be the sort of thing that is wrong in the
//! last lane and never noticed. The scalar loop the C ends with is what is
//! ported. Revisit if load times ever make it worth measuring.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const c = impl.c;

extern fn fprintf(stream: *anyopaque, fmt: [*:0]const u8, ...) c_int;
extern var __stderrp: *anyopaque;

const types = @import("../types.zig");

/// The C writes these to stderr directly rather than through ggml's logger,
/// so this does too -- a caller that has redirected ggml logging still wants
/// to know its model file is corrupt.
fn warn(comptime fmt: [*:0]const u8, args: anytype) void {
    _ = @call(.auto, fprintf, .{ __stderrp, fmt } ++ args);
}

/// Ports `validate_float` (ggml-quants.c:5330 @c1d0e7a00).
fn validateFloat(f: f32, i: usize) bool {
    if (std.math.isInf(f)) {
        warn("ggml_validate_row_data: found inf value at block %zu\n", .{i});
        return false;
    }
    if (std.math.isNan(f)) {
        warn("ggml_validate_row_data: found nan value at block %zu\n", .{i});
        return false;
    }
    return true;
}

/// Ports `isinf_fp16` and `isnan_fp16` (ggml-quants.c:5344, 5348 @c1d0e7a00), plus
/// `validate_fp16` (ggml-quants.c:5352 @c1d0e7a00).
///
/// Tests the bit pattern rather than widening to f32 first: exponent all ones
/// with a zero mantissa is infinity, with a non-zero mantissa a NaN.
fn validateFp16(f: blocks.Half, i: usize) bool {
    const exp_all_ones = (f & 0x7c00) == 0x7c00;
    if (exp_all_ones and (f & 0x03ff) == 0) {
        warn("ggml_validate_row_data: found inf value at block %zu\n", .{i});
        return false;
    }
    if (exp_all_ones and (f & 0x03ff) != 0) {
        warn("ggml_validate_row_data: found nan value at block %zu\n", .{i});
        return false;
    }
    return true;
}

/// Ports `validate_e_e8m0` (ggml-quants.c:5366 @c1d0e7a00).
///
/// An E8M0 exponent has no mantissa, so 0xFF is its only invalid encoding --
/// the equivalent of a NaN.
fn validateE8m0(e: u8, i: usize) bool {
    if (e == 0xff) {
        warn("ggml_validate_row_data: found invalid e value %d at block %zu\n", .{ @as(c_int, e), i });
        return false;
    }
    return true;
}

/// Ports `VALIDATE_ROW_DATA_D_F16_IMPL` (ggml-quants.c:5375 @c1d0e7a00): one f16 scale
/// per block.
fn checkD(comptime Block: type, data: [*]const u8, nb: usize) bool {
    const q: [*]const Block = @ptrCast(@alignCast(data));
    for (0..nb) |i| {
        if (!validateFp16(q[i].d, i)) return false;
    }
    return true;
}

/// Ports `VALIDATE_ROW_DATA_DM_F16_IMPL` (ggml-quants.c:5383 @c1d0e7a00): a scale and a
/// second f16, named `m` on the legacy formats and `dmin` on the K-quants.
fn checkDM(comptime Block: type, comptime second: []const u8, data: [*]const u8, nb: usize) bool {
    const q: [*]const Block = @ptrCast(@alignCast(data));
    for (0..nb) |i| {
        if (!validateFp16(q[i].d, i)) return false;
        if (!validateFp16(@field(q[i], second), i)) return false;
    }
    return true;
}

/// Ports `VALIDATE_ROW_DATA_E_E8M0_IMPL` (ggml-quants.c:5391 @c1d0e7a00).
fn checkE(comptime Block: type, data: [*]const u8, nb: usize) bool {
    const q: [*]const Block = @ptrCast(@alignCast(data));
    for (0..nb) |i| {
        if (!validateE8m0(q[i].e, i)) return false;
    }
    return true;
}

/// Ports `VALIDATE_ROW_DATA_DVEC_F16_IMPL` (ggml-quants.c:5399 @c1d0e7a00): a vector of
/// scales per block, which only `nvfp4` has.
fn checkDVec(comptime Block: type, comptime nr: usize, data: [*]const u8, nb: usize) bool {
    const q: [*]const Block = @ptrCast(@alignCast(data));
    for (0..nb) |i| {
        for (0..nr) |j| {
            if (!validateFp16(q[i].d[j], i)) return false;
        }
    }
    return true;
}

/// Ports `ggml_validate_row_data` (ggml-quants.c:5409 @c1d0e7a00).
///
/// Parameters:
/// - `t`: the format `data` is in.
/// - `data`: the raw bytes.
/// - `nbytes`: their length; must be a whole number of blocks.
///
/// Return: false, with a message on stderr, if the data is not a whole number
/// of blocks or any scale is an infinity or a NaN.
pub export fn ggml_validate_row_data(t: c.enum_ggml_type, data: ?*const anyopaque, nbytes: usize) bool {
    if (t < 0 or t >= c.GGML_TYPE_COUNT) {
        warn("%s: invalid type %d\n", .{ "ggml_validate_row_data", @as(c_int, @intCast(t)) });
        return false;
    }

    const type_size = types.ggml_type_size(t);
    if (nbytes % type_size != 0) {
        warn("%s: invalid size %zu for type %s (type size = %zu)\n", .{
            "ggml_validate_row_data", nbytes, types.ggml_type_name(t), type_size,
        });
        return false;
    }

    const nb = nbytes / type_size;
    const bytes: [*]const u8 = @ptrCast(data.?);

    return switch (t) {
        c.GGML_TYPE_BF16 => blk: {
            // Counted rather than short-circuited, so the message can say how
            // many -- the C reports totals here and per-block elsewhere.
            var nans: usize = 0;
            var infs: usize = 0;
            const f: [*]const u16 = @ptrCast(@alignCast(bytes));
            for (0..nb) |i| {
                if ((f[i] & 0x7fff) > 0x7f80) nans += 1;
                if ((f[i] & 0x7fff) == 0x7f80) infs += 1;
            }
            if (nans != 0) {
                warn("%s: found %d NaNs in row of %zu BF16 values\n", .{ "ggml_validate_row_data", @as(c_int, @intCast(nans)), nb });
                break :blk false;
            }
            if (infs != 0) {
                warn("%s: found %d infinities in row of %zu BF16 values\n", .{ "ggml_validate_row_data", @as(c_int, @intCast(infs)), nb });
                break :blk false;
            }
            break :blk true;
        },
        c.GGML_TYPE_F16 => blk: {
            const f: [*]const blocks.Half = @ptrCast(@alignCast(bytes));
            for (0..nb) |i| {
                if (!validateFp16(f[i], i)) break :blk false;
            }
            break :blk true;
        },
        c.GGML_TYPE_F32 => blk: {
            const f: [*]const f32 = @ptrCast(@alignCast(bytes));
            for (0..nb) |i| {
                if (!validateFloat(f[i], i)) break :blk false;
            }
            break :blk true;
        },
        c.GGML_TYPE_F64 => blk: {
            const f: [*]const f64 = @ptrCast(@alignCast(bytes));
            for (0..nb) |i| {
                if (!validateFloat(@floatCast(f[i]), i)) break :blk false;
            }
            break :blk true;
        },

        c.GGML_TYPE_Q1_0 => checkD(blocks.Q1_0, bytes, nb),
        c.GGML_TYPE_Q2_0 => checkD(blocks.Q2_0, bytes, nb),
        c.GGML_TYPE_Q4_0 => checkD(blocks.Q4_0, bytes, nb),
        c.GGML_TYPE_Q4_1 => checkDM(blocks.Q4_1, "m", bytes, nb),
        c.GGML_TYPE_Q5_0 => checkD(blocks.Q5_0, bytes, nb),
        c.GGML_TYPE_Q5_1 => checkDM(blocks.Q5_1, "m", bytes, nb),
        c.GGML_TYPE_Q8_0 => checkD(blocks.Q8_0, bytes, nb),
        c.GGML_TYPE_MXFP4 => checkE(blocks.MXFP4, bytes, nb),
        // NVFP4 is deliberately not checked: its scales are UE4M3 bytes and
        // every byte value is a valid one, so there is nothing that could be
        // a NaN. `checkDVec` exists for the C's macro of the same shape, which
        // no format currently reaches.
        c.GGML_TYPE_NVFP4 => true,

        c.GGML_TYPE_Q2_K => checkDM(blocks.Q2_K, "dmin", bytes, nb),
        c.GGML_TYPE_Q3_K => checkD(blocks.Q3_K, bytes, nb),
        c.GGML_TYPE_Q4_K => checkDM(blocks.Q4_K, "dmin", bytes, nb),
        c.GGML_TYPE_Q5_K => checkDM(blocks.Q5_K, "dmin", bytes, nb),
        c.GGML_TYPE_Q6_K => checkD(blocks.Q6_K, bytes, nb),
        c.GGML_TYPE_Q8_K => blk: {
            // The only block format whose scale is a full f32.
            const q: [*]const blocks.Q8_K = @ptrCast(@alignCast(bytes));
            for (0..nb) |i| {
                if (!validateFloat(q[i].d, i)) break :blk false;
            }
            break :blk true;
        },

        c.GGML_TYPE_TQ1_0 => checkD(blocks.TQ1_0, bytes, nb),
        c.GGML_TYPE_TQ2_0 => checkD(blocks.TQ2_0, bytes, nb),

        c.GGML_TYPE_IQ1_S => checkD(blocks.IQ1_S, bytes, nb),
        c.GGML_TYPE_IQ1_M => blk: {
            // No `d` field: the scale is scattered across the `scales` shorts
            // and has to be reassembled before it can be checked.
            const q: [*]const blocks.IQ1_M = @ptrCast(@alignCast(bytes));
            for (0..nb) |i| {
                const sc: [*]const u16 = @ptrCast(@alignCast(&q[i].scales));
                if (!validateFp16(blocks.iq1mScale(sc), i)) break :blk false;
            }
            break :blk true;
        },
        c.GGML_TYPE_IQ2_XXS => checkD(blocks.IQ2_XXS, bytes, nb),
        c.GGML_TYPE_IQ2_XS => checkD(blocks.IQ2_XS, bytes, nb),
        c.GGML_TYPE_IQ2_S => checkD(blocks.IQ2_S, bytes, nb),
        c.GGML_TYPE_IQ3_XXS => checkD(blocks.IQ3_XXS, bytes, nb),
        c.GGML_TYPE_IQ3_S => checkD(blocks.IQ3_S, bytes, nb),
        c.GGML_TYPE_IQ4_XS => checkD(blocks.IQ4_XS, bytes, nb),
        c.GGML_TYPE_IQ4_NL => checkD(blocks.IQ4_NL, bytes, nb),

        // Integer formats have no scale to be NaN.
        c.GGML_TYPE_I8, c.GGML_TYPE_I16, c.GGML_TYPE_I32, c.GGML_TYPE_I64 => true,

        else => blk: {
            warn("%s: invalid type %d\n", .{ "ggml_validate_row_data", @as(c_int, @intCast(t)) });
            break :blk false;
        },
    };
}
