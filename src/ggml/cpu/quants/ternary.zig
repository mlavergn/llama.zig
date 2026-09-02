//! Reference dot products for the ternary formats, BitNet b1.58 and TriLM.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/quants.c` (v0.3.0, `c1d0e7a00`),
//! the `_generic` dot products at lines 481 and 533. Each function names the C
//! it replaces and the line it began at.
//!
//! # Unreachable on this target
//!
//! As with `legacy.zig` and `k.zig`: `arch/arm/quants.c` supplies the real
//! entry points, so these `_generic` names are exported and never called.
//! `golden.zig` is their only gate.
//!
//! # Base-three unpacking, done in fixed point
//!
//! `tq1_0` does not pack five ternary digits into a byte directly. The
//! quantizer builds `d0*81 + d1*27 + d2*9 + d3*3 + d4`, a value in 0..242,
//! then **rescales it into the full byte range** with a ceiling division:
//! `q = (q * 256 + 242) / 243`. So the byte is `0.d0 d1 d2 d3 d4` in base
//! three, read as a fraction of 256.
//!
//! Extraction is therefore the fixed-point trick for taking the next base-3
//! fractional digit: `q = byte * pow3[l]` shifts past the digits already read,
//! and `(q * 3) >> 8` takes the next one. Both steps rely on **eight-bit
//! truncation** -- `q` is a `uint8_t`, and the multiply is what discards the
//! digits above `l`. Written in Zig with an `i32` or a `u16` in place of that
//! `u8`, the answer is silently wrong, so the truncation is spelled out rather
//! than inferred.

const std = @import("std");
const impl = @import("../../impl.zig");
const convert = @import("../convert.zig");
const blocks = @import("../../quants/blocks.zig");
const c = impl.c;

inline fn f(h: u16) f32 {
    return convert.cpuFp16ToFp32(h);
}

inline fn as(comptime Block: type, p: ?*const anyopaque) [*]const Block {
    return @ptrCast(@alignCast(p.?));
}

/// Ports `pow3` (ggml-cpu/quants.c:493 @c1d0e7a00).
const pow3 = [6]u8{ 1, 3, 9, 27, 81, 243 };

/// One base-three digit, as `ggml_vec_dot_tq1_0_q8_K_generic` extracts it
/// (ggml-cpu/quants.c:481 @c1d0e7a00).
///
/// Parameters:
/// - `byte`: the packed byte, holding five digits as a base-3 fraction of 256.
/// - `l`: which digit, 0 the **most** significant -- the order the quantizer
///   wrote them in.
///
/// Return: the digit's ternary value, 0, 1 or 2 -- **not** yet shifted to
/// -1, 0, +1.
inline fn digit(byte: u8, l: usize) u16 {
    // `q` is a `uint8_t` in the C and the truncation is load-bearing: it is
    // what discards the digits already read.
    const q: u8 = byte *% pow3[l];
    return (@as(u16, q) * 3) >> 8;
}

/// Ports `ggml_vec_dot_tq1_0_q8_K_generic` (ggml-cpu/quants.c:481 @c1d0e7a00).
///
/// The 256 weights of a super-block are stored as 5 digits per byte across
/// `qs`, plus a 4-digit tail in `qh`. `qs` is 48 bytes, so the first loop
/// covers the 32-byte-aligned part and the second the 16-byte remainder --
/// which is why the two loops differ only in their stride.
pub export fn ggml_vec_dot_tq1_0_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.TQ1_0, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    const qs_len = @typeInfo(@FieldType(blocks.TQ1_0, "qs")).array.len;
    const qh_len = @typeInfo(@FieldType(blocks.TQ1_0, "qh")).array.len;

    var sumf: f32 = 0.0;

    for (0..@intCast(nb)) |i| {
        var sum: i32 = 0;

        // The part of `qs` that divides into 32-byte groups.
        const head = qs_len - qs_len % 32;
        var j: usize = 0;
        while (j < head) : (j += 32) {
            for (0..5) |l| {
                for (0..32) |m| {
                    const xi = digit(x[i].qs[j + m], l);
                    sum += (@as(i32, xi) - 1) * y[i].qs[j * 5 + l * 32 + m];
                }
            }
        }

        // The 16-byte remainder.
        j = head;
        while (j < qs_len) : (j += 16) {
            for (0..5) |l| {
                for (0..16) |m| {
                    const xi = digit(x[i].qs[j + m], l);
                    sum += (@as(i32, xi) - 1) * y[i].qs[j * 5 + l * 16 + m];
                }
            }
        }

        // The tail: four digits per byte rather than five.
        for (0..4) |l| {
            for (0..qh_len) |jj| {
                const xi = digit(x[i].qh[jj], l);
                sum += (@as(i32, xi) - 1) * y[i].qs[qs_len * 5 + l * qh_len + jj];
            }
        }

        sumf += @as(f32, @floatFromInt(sum)) * (f(x[i].d) * y[i].d);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_tq2_0_q8_K_generic` (ggml-cpu/quants.c:533 @c1d0e7a00).
///
/// Four two-bit digits per byte, mapped `{0,1,2} -> {-1,0,1}` by subtracting
/// one. Simpler than `tq1_0` because two bits waste a quarter of the range in
/// exchange for a plain shift-and-mask.
pub export fn ggml_vec_dot_tq2_0_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.TQ2_0, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);
    const qs_len = @typeInfo(@FieldType(blocks.TQ2_0, "qs")).array.len;

    var sumf: f32 = 0.0;

    for (0..@intCast(nb)) |i| {
        var sumi: i32 = 0;

        var j: usize = 0;
        while (j < qs_len) : (j += 32) {
            for (0..4) |l| {
                for (0..32) |k| {
                    const shift: u3 = @intCast(l * 2);
                    const w: i32 = @as(i32, (x[i].qs[j + k] >> shift) & 3) - 1;
                    sumi += @as(i32, y[i].qs[j * 4 + l * 32 + k]) * w;
                }
            }
        }

        const d = y[i].d * f(x[i].d);

        sumf += @as(f32, @floatFromInt(sumi)) * d;
    }

    s[0] = sumf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const testing = @import("testing.zig");
const golden = @import("golden.zig");

test "tq1_0 dot matches the C" {
    try testing.check(ggml_vec_dot_tq1_0_q8_K_generic, c.GGML_TYPE_TQ1_0, c.GGML_TYPE_Q8_K, golden.tq1_0);
}

test "tq2_0 dot matches the C" {
    try testing.check(ggml_vec_dot_tq2_0_q8_K_generic, c.GGML_TYPE_TQ2_0, c.GGML_TYPE_Q8_K, golden.tq2_0);
}

/// Packs five ternary digits exactly as `quantize_row_tq1_0_ref` does
/// (ggml-quants.c:2316 @c1d0e7a00), for the round trip below.
fn pack5(digits: [5]u8) u8 {
    var q: u16 = 0;
    for (digits) |d| {
        q = q * 3 + d;
    }
    // Ceiling division into the byte range: 243 == 3^5.
    return @intCast((q * 256 + (243 - 1)) / 243);
}

test "every five-digit group survives the pack and extract round trip" {
    // The whole 3^5 space, not a sample: the fixed-point extraction is only
    // exact because of the quantizer's ceiling division, and an off-by-one
    // there would show up on a handful of the 243 combinations rather than
    // all of them.
    var d: [5]u8 = @splat(0);
    var count: usize = 0;

    for (0..3) |d0| {
        for (0..3) |d1| {
            for (0..3) |d2| {
                for (0..3) |d3| {
                    for (0..3) |d4| {
                        d = .{ @intCast(d0), @intCast(d1), @intCast(d2), @intCast(d3), @intCast(d4) };
                        const byte = pack5(d);
                        for (0..5) |l| {
                            try std.testing.expectEqual(@as(u16, d[l]), digit(byte, l));
                        }
                        count += 1;
                    }
                }
            }
        }
    }

    try std.testing.expectEqual(@as(usize, 243), count);
}

test "digit extraction depends on eight-bit truncation" {
    // Widening `q` past eight bits keeps the digits that should have been
    // discarded, so digit 2 of this byte comes out wrong. Asserted so the
    // `*%` above cannot be "tidied" into a wider type.
    const byte = pack5(.{ 1, 0, 2, 1, 0 });
    try std.testing.expectEqual(@as(u16, 2), digit(byte, 2));

    const widened = (@as(u16, byte) * pow3[2] * 3) >> 8;
    try std.testing.expect(widened != 2);
}

test "no byte yields a digit out of range" {
    // The `- 1` below the extraction assumes 0, 1 or 2; anything else is a
    // weight outside {-1, 0, 1}.
    var b: u16 = 0;
    while (b < 256) : (b += 1) {
        for (0..5) |l| {
            try std.testing.expect(digit(@intCast(b), l) <= 2);
        }
    }
}
