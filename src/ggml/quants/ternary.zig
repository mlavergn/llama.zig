//! The ternary formats, for BitNet b1.58 and TriLM.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 2314-2485. Each function names the C function it replaces and the line it
//! began at.
//!
//! # Base 3, not base 2
//!
//! Every weight is -1, 0 or +1. Two bits per weight would waste a quarter of
//! the space, so `tq1_0` packs **five trits into a byte** base-3: 3^5 = 243,
//! which fits in 256 with room to spare. That is where the 1.6875 bpw comes
//! from, and why `qs` has the odd length it does.
//!
//! `tq2_0` takes the simpler two-bits-per-weight route at 2.0625 bpw, trading
//! space for a decode that is a shift and a mask.
//!
//! # The multiply-and-shift trick
//!
//! Extracting trit `n` from a base-3 byte would need a division. Instead both
//! directions use a fixed-point reciprocal:
//!
//! - Packing ends with `q = (q * 256 + 242) / 243` -- a *ceiling* division that
//!   maps the base-3 digit string into the full byte range.
//! - Unpacking is `((q * pow3[n]) * 3) >> 8`, relying on the `uint8_t`
//!   multiply wrapping to drop the digits above the one wanted.
//!
//! The `u8` wraparound is load-bearing: `q * pow3[n]` is deliberately truncated
//! to eight bits, and a wider intermediate gives a different answer.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const c = impl.c;

const fp16 = impl.fp32ToFp16;
const unfp16 = impl.fp16ToFp32;
const QK_K = c.QK_K;

/// Rounds to nearest, halfway away from zero -- C's `lroundf`.
///
/// Not `helpers.nearestInt`, which the K-quants use: that rounds half to
/// **even**, and the ternary quantizers call `lroundf` instead. With weights
/// this coarse the ties are common, so the two would disagree often.
inline fn lround(v: f32) i32 {
    return impl.truncTo(i32, @round(v));
}

/// Ports `quantize_row_tq1_0_ref` (ggml-quants.c:2316 @c1d0e7a00).
///
/// Three passes over the block because the trit count does not divide evenly:
/// 32 bytes at a time while a full 5x32 span remains, then 16 at a time, then
/// the four-trit tail in `qh`.
pub export fn quantize_row_tq1_0_ref(x_in: [*c]const f32, y: [*c]blocks.TQ1_0, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var x = x_in;

    const qs_len = @typeInfo(@FieldType(blocks.TQ1_0, "qs")).array.len;
    const qh_len = @typeInfo(@FieldType(blocks.TQ1_0, "qh")).array.len;

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..QK_K) |j| amax = @max(amax, @abs(x[j]));

        const d = amax;
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);

        // 5 elements per byte, along 32 bytes
        var j: usize = 0;
        while (j < qs_len - qs_len % 32) : (j += 32) {
            for (0..32) |m| {
                var q: u8 = 0;
                for (0..5) |n| {
                    const xi: u8 = @intCast(lround(x[m + n * 32] * id) + 1); // -1,0,1 -> 0,1,2
                    q *%= 3;
                    q +%= xi;
                }
                y[i].qs[j + m] = ceilDiv243(q);
            }
            x += 5 * 32;
        }
        // along 16 bytes
        j = qs_len - qs_len % 32;
        while (j < qs_len) : (j += 16) {
            for (0..16) |m| {
                var q: u8 = 0;
                for (0..5) |n| {
                    const xi: u8 = @intCast(lround(x[m + n * 16] * id) + 1);
                    q *%= 3;
                    q +%= xi;
                }
                y[i].qs[j + m] = ceilDiv243(q);
            }
            x += 5 * 16;
        }
        // 4 elements per byte
        for (0..qh_len) |jj| {
            var q: u8 = 0;
            for (0..4) |m| {
                const xi: u8 = @intCast(lround(x[jj + m * qh_len] * id) + 1);
                q *%= 3;
                q +%= xi;
            }
            // Only four trits here, so shift the first up to the most
            // significant position the five-trit decoder expects.
            q *%= 3;
            y[i].qh[jj] = ceilDiv243(q);
        }
        x += 4 * qh_len;
    }
}

/// The `(q * 256 + 242) / 243` ceiling division both packing loops end with.
inline fn ceilDiv243(q: u8) u8 {
    return @intCast((@as(u16, q) * 256 + (243 - 1)) / 243);
}

/// Ports `dequantize_row_tq1_0` (ggml-quants.c:2428 @c1d0e7a00).
pub export fn dequantize_row_tq1_0(x: [*c]const blocks.TQ1_0, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    const pow3 = [6]u8{ 1, 3, 9, 27, 81, 243 };
    const qs_len = @typeInfo(@FieldType(blocks.TQ1_0, "qs")).array.len;
    const qh_len = @typeInfo(@FieldType(blocks.TQ1_0, "qh")).array.len;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);

        var j: usize = 0;
        while (j < qs_len - qs_len % 32) : (j += 32) {
            for (0..5) |n| {
                for (0..32) |m| {
                    // The u8 multiply wraps on purpose: it discards the trits
                    // above the one being read.
                    const q: u8 = x[i].qs[j + m] *% pow3[n];
                    const xi: i16 = @intCast((@as(u16, q) * 3) >> 8);
                    y[0] = @as(f32, @floatFromInt(xi - 1)) * d;
                    y += 1;
                }
            }
        }
        j = qs_len - qs_len % 32;
        while (j < qs_len) : (j += 16) {
            for (0..5) |n| {
                for (0..16) |m| {
                    const q: u8 = x[i].qs[j + m] *% pow3[n];
                    const xi: i16 = @intCast((@as(u16, q) * 3) >> 8);
                    y[0] = @as(f32, @floatFromInt(xi - 1)) * d;
                    y += 1;
                }
            }
        }

        for (0..4) |n| {
            for (0..qh_len) |jj| {
                const q: u8 = x[i].qh[jj] *% pow3[n];
                const xi: i16 = @intCast((@as(u16, q) * 3) >> 8);
                y[0] = @as(f32, @floatFromInt(xi - 1)) * d;
                y += 1;
            }
        }
    }
}

/// Ports `quantize_row_tq2_0_ref` (ggml-quants.c:2382 @c1d0e7a00).
pub export fn quantize_row_tq2_0_ref(x_in: [*c]const f32, y: [*c]blocks.TQ2_0, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var x = x_in;

    const qs_len = @typeInfo(@FieldType(blocks.TQ2_0, "qs")).array.len;

    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..QK_K) |j| amax = @max(amax, @abs(x[j]));

        const d = amax;
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        y[i].d = fp16(d);

        var j: usize = 0;
        while (j < qs_len) : (j += 32) {
            for (0..32) |m| {
                var q: u8 = 0;
                for (0..4) |n| {
                    const xi: i32 = lround(x[m + n * 32] * id) + 1;
                    q +%= @as(u8, @intCast(xi & 3)) << @intCast(2 * n);
                }
                y[i].qs[j + m] = q;
            }
            x += 4 * 32;
        }
    }
}

/// Ports `dequantize_row_tq2_0` (ggml-quants.c:2467 @c1d0e7a00).
pub export fn dequantize_row_tq2_0(x: [*c]const blocks.TQ2_0, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    const qs_len = @typeInfo(@FieldType(blocks.TQ2_0, "qs")).array.len;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);

        var j: usize = 0;
        while (j < qs_len) : (j += 32) {
            for (0..4) |l| {
                for (0..32) |m| {
                    const q: i8 = @intCast((x[i].qs[j + m] >> @intCast(l * 2)) & 3);
                    y[0] = @as(f32, @floatFromInt(q - 1)) * d;
                    y += 1;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

const t = @import("testing.zig");

test {
    std.testing.refAllDecls(@This());
}

fn checkTernary(
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

test "tq1_0 matches the C on every pattern" {
    try checkTernary("TQ1_0", blocks.TQ1_0, quantize_row_tq1_0_ref, dequantize_row_tq1_0);
}

test "tq2_0 matches the C on every pattern" {
    try checkTernary("TQ2_0", blocks.TQ2_0, quantize_row_tq2_0_ref, dequantize_row_tq2_0);
}

test "a ternary round trip yields only -d, 0 and +d" {
    // Independent of the checksums: whatever the packing does, the values that
    // come back must be exactly three, or the format is not ternary.
    var src: [QK_K]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = switch (i % 3) {
        0 => -0.75,
        1 => 0.0,
        else => 0.75,
    };

    var b: blocks.TQ2_0 = undefined;
    @memset(std.mem.asBytes(&b), 0);
    quantize_row_tq2_0_ref(&src, @ptrCast(&b), QK_K);

    var out: [QK_K]f32 = undefined;
    dequantize_row_tq2_0(@ptrCast(&b), &out, QK_K);

    const d = unfp16(b.d);
    for (out) |v| {
        try std.testing.expect(v == -d or v == 0.0 or v == d);
    }
}
