//! NEON dot products for the ternary formats, BitNet b1.58 and TriLM.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (v0.3.0,
//! `c1d0e7a00`), the `__ARM_NEON` arms of the dot products at lines 1397 and
//! 1574. Each function names the C it replaces and the line it began at.
//!
//! # Numerically the simplest pair in the file
//!
//! Both accumulate entirely in `i32` lanes and touch floating point exactly
//! once per super-block: `sumf += d * (float) vaddvq_s32(sumi0)`. There is no
//! fusion question and no accumulation order to pin down, because the integer
//! sum is exact however it is grouped.
//!
//! # The base-three digit, extracted without a divide
//!
//! `tq1_0`'s scalar form takes digit `l` as `((byte * pow3[l]) * 3) >> 8` in
//! eight-bit arithmetic. The NEON form computes the same thing as
//! `(halving_add(q, q >> 1)) >> 6`:
//!
//! - `q >> 1` is `q/2`, and `halving_add(a, b)` is `(a + b) >> 1` computed
//!   without overflowing the lane, so the pair gives `(q + q/2) / 2 == 3q/4`
//!   to within the truncation;
//! - `>> 6` then divides by 64, giving `3q/256`.
//!
//! Which is `(q * 3) >> 8`. The halving add is what keeps `q + q/2` from
//! overflowing eight bits, and is why this cannot be written as a plain add.

const std = @import("std");
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const blocks = @import("../../../quants/blocks.zig");
const neon = @import("neon.zig");
const c = impl.c;

const i8x16 = neon.i8x16;
const u8x16 = neon.u8x16;
const i16x8 = neon.i16x8;
const i32x4 = neon.i32x4;

inline fn f(h: u16) f32 {
    return convert.cpuFp16ToFp32(h);
}

inline fn as(comptime Block: type, p: ?*const anyopaque) [*]const Block {
    return @ptrCast(@alignCast(p.?));
}

/// The base-three digit of every lane, as `tq1_0`'s NEON arm computes it.
///
/// `(halving_add(q, q >> 1)) >> 6`, which equals `(q * 3) >> 8`. See the note
/// at the top of this file for why it is written this way.
inline fn digit(q: u8x16) i8x16 {
    return @bitCast(neon.shrN(neon.hadd_u8(q, neon.shrN(q, 1)), 6));
}

/// The per-super-block epilogue both kernels share (arch/arm/quants.c:1487 and
/// arch/arm/quants.c:1719).
///
/// Folds the two integer accumulators, subtracts the right operand's group
/// sums -- which is the `-1` of the `{0,1,2} -> {-1,0,1}` mapping, applied
/// once per element rather than per lane -- and scales it into `sumf`.
///
/// The C's last line is `sumf += d * (float) vaddvq_s32(sumi0)`: one
/// expression, so clang fuses it. This used to return `d * isum` for the
/// caller to add, which rounds the product first. The golden patterns could
/// not tell the two apart; a CPU-only Qwen3.5 decode against the reference
/// could, and `make ops-diff` now has the cases that do.
///
/// Parameters:
/// - `sumi0`, `sumi1`: the two accumulators.
/// - `y`: the `q8_K` block, for its `bsums`.
/// - `d`: the product of the two blocks' deltas.
/// - `sumf`: the running sum.
///
/// Return: `sumf` with this super-block's contribution added, fused.
inline fn epilogue(sumi0_in: i32x4, sumi1: i32x4, y: *const blocks.Q8_K, d: f32, sumf: f32) f32 {
    const ysum0 = neon.loadFrom(i16x8, &y.bsums);
    const ysum1 = neon.loadFrom(i16x8, y.bsums[8..]);

    var sumi0 = neon.add(sumi0_in, sumi1);
    sumi0 = neon.sub(sumi0, neon.paddlq_s16(neon.add(ysum0, ysum1)));

    return @mulAdd(f32, d, @as(f32, @floatFromInt(neon.addvq_s32(sumi0))), sumf);
}

/// Ports `k_shift` (arch/arm/quants.c:3952 @c1d0e7a00).
///
/// Four lanes each of 1, 3, 9 and 27: the `qh` tail holds four digits per
/// byte, and one broadcast `u32` times these powers puts a different digit in
/// each lane's high bits.
const k_shift: u8x16 = .{ 1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27 };

/// Ports `ggml_vec_dot_tq1_0_q8_K` (arch/arm/quants.c:1397 @c1d0e7a00).
///
/// Five digits per byte across `qs`, then four per byte in the `qh` tail. The
/// two blocks in the C are two braced scopes; they are two loops' worth of
/// straight-line code here for the same reason -- the strides differ.
pub export fn ggml_vec_dot_tq1_0_q8_K(
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

    var sumf: f32 = 0.0;

    for (0..@intCast(nb)) |i| {
        var sumi0: i32x4 = @splat(0);
        var sumi1: i32x4 = @splat(0);

        // The first 32 bytes: five digits each, so ten groups of sixteen
        // weights, alternating between the two accumulators.
        {
            const qx0 = neon.loadFrom(u8x16, &x[i].qs);
            const qx1 = neon.loadFrom(u8x16, x[i].qs[16..]);

            const powers = [5]u8{ 1, 3, 9, 27, 81 };
            inline for (powers, 0..) |p, l| {
                const three: u8x16 = @splat(p);
                const a = digit(neon.mul(qx0, three));
                const b = digit(neon.mul(qx1, three));
                const qy_a = neon.load(i8x16, x_qy(y, i, 2 * l * 16));
                const qy_b = neon.load(i8x16, x_qy(y, i, (2 * l + 1) * 16));
                sumi0 = neon.dotq_s32(sumi0, a, qy_a);
                sumi1 = neon.dotq_s32(sumi1, b, qy_b);
            }
        }

        // The remaining 16 bytes of `qs`, plus the four-digit `qh` tail.
        {
            const qx0 = neon.loadFrom(u8x16, x[i].qs[32..]);

            const qh = std.mem.readInt(u32, &x[i].qh, .little);
            const broadcast: u8x16 = @bitCast(@as(neon.u32x4, @splat(qh)));

            var sq: [6]i8x16 = undefined;
            const powers = [5]u8{ 1, 3, 9, 27, 81 };
            inline for (powers, 0..) |p, l| {
                const m: u8x16 = @splat(p);
                sq[l] = digit(neon.mul(qx0, m));
            }
            // The tail's five digits per byte become four, selected by
            // `k_shift` rather than by a power of three per group.
            sq[5] = digit(neon.mul(broadcast, k_shift));

            inline for (0..6) |l| {
                const qy = neon.load(i8x16, x_qy(y, i, 160 + l * 16));
                if (l % 2 == 0) {
                    sumi0 = neon.dotq_s32(sumi0, sq[l], qy);
                } else {
                    sumi1 = neon.dotq_s32(sumi1, sq[l], qy);
                }
            }
        }

        sumf = epilogue(sumi0, sumi1, &y[i], f(x[i].d) * y[i].d, sumf);
    }

    s[0] = sumf;
}

/// The `q8_K` operand's bytes at an offset, as a pointer the loader accepts.
inline fn x_qy(y: [*]const blocks.Q8_K, i: usize, off: usize) [*]const u8 {
    return @as([*]const u8, @ptrCast(&y[i].qs)) + off;
}

/// Ports `ggml_vec_dot_tq2_0_q8_K` (arch/arm/quants.c:1574 @c1d0e7a00).
///
/// Four two-bit digits per byte, extracted with a shift and a mask rather than
/// the base-three trick `tq1_0` needs.
pub export fn ggml_vec_dot_tq2_0_q8_K(
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

    const m3: u8x16 = @splat(3);

    var sumf: f32 = 0.0;

    for (0..@intCast(nb)) |i| {
        var sumi0: i32x4 = @splat(0);
        var sumi1: i32x4 = @splat(0);

        var j: usize = 0;
        while (j < qs_len) : (j += 32) {
            const qx0 = neon.loadFrom(u8x16, x[i].qs[j..].ptr);
            const qx1 = neon.loadFrom(u8x16, x[i].qs[j + 16 ..].ptr);

            // Eight groups: the two loaded vectors at four shifts each,
            // alternating accumulators so the two chains stay independent.
            inline for (0..4) |l| {
                const sh = 2 * l;
                const a: i8x16 = @bitCast(neon.@"and"(neon.shrN(qx0, sh), m3));
                const b: i8x16 = @bitCast(neon.@"and"(neon.shrN(qx1, sh), m3));
                const qy_a = neon.load(i8x16, x_qy(y, i, j * 4 + (2 * l) * 16));
                const qy_b = neon.load(i8x16, x_qy(y, i, j * 4 + (2 * l + 1) * 16));
                sumi0 = neon.dotq_s32(sumi0, a, qy_a);
                sumi1 = neon.dotq_s32(sumi1, b, qy_b);
            }
        }

        sumf = epilogue(sumi0, sumi1, &y[i], f(x[i].d) * y[i].d, sumf);
    }

    s[0] = sumf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const testing = @import("../testing.zig");
const golden = @import("golden.zig");

test "tq1_0 dot matches the C" {
    try testing.check(ggml_vec_dot_tq1_0_q8_K, c.GGML_TYPE_TQ1_0, c.GGML_TYPE_Q8_K, golden.tq1_0);
}

test "tq2_0 dot matches the C" {
    try testing.check(ggml_vec_dot_tq2_0_q8_K, c.GGML_TYPE_TQ2_0, c.GGML_TYPE_Q8_K, golden.tq2_0);
}

test "the vector digit extraction agrees with the scalar form" {
    // `(halving_add(q, q >> 1)) >> 6` must equal `(q * 3) >> 8` in eight-bit
    // arithmetic, for every byte -- that equivalence is the whole reason the
    // NEON version can avoid a widening multiply.
    var b: u16 = 0;
    while (b < 256) : (b += 1) {
        const q: u8 = @intCast(b);
        const vec: [16]i8 = digit(@as(u8x16, @splat(q)));
        const scalar: u8 = @intCast((@as(u16, q) * 3) >> 8);
        try std.testing.expectEqual(@as(i8, @bitCast(scalar)), vec[0]);
    }
}
