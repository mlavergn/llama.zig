//! Dequantizing the i-quant (codebook) formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 2488-2765. Each function names the C function it replaces and the line it
//! began at. The quantizers are in `iq_quant.zig`; these are separate because
//! decoding a codebook index is trivial and choosing one is not.
//!
//! # How a codebook format decodes
//!
//! A stored index selects an 8-element vector from a grid, and a separate sign
//! byte flips some of its elements. So the inner loop is always the same
//! shape:
//!
//! ```
//! grid  = grid_table[index]        // 8 values, packed as bytes in a u64
//! signs = ksigns_iq2xs[sign_index] // one bit per element
//! y[j]  = scale * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1)
//! ```
//!
//! `ksigns_iq2xs` and `kmask_iq2xs` come from `ggml-common.h`, as do the grids.
//! The grids are `u64` arrays reinterpreted as eight bytes -- `gridBytes`
//! below does that reinterpretation once rather than at every call site.
//!
//! # Where the bits actually live
//!
//! These formats hide fields inside other fields, and the C reaches them by
//! pointer arithmetic past the end of a declared array:
//!
//! - `iq2_s` keeps its sign bytes in the tail of `qs`, at `qs + QK_K/8`.
//! - `iq3_xxs` keeps scales *and* signs in the tail of `qs`, at `qs + QK_K/4`.
//! - `iq1_m` has no scale field at all; its f16 is scattered four bits at a
//!   time across the top of the four `scales` shorts.
//!
//! None of that is visible in the struct definitions, which is why each is
//! called out where it is used.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const c = impl.c;

const unfp16 = impl.fp16ToFp32;
const QK_K = c.QK_K;

/// Reinterprets one grid entry as the eight bytes it packs.
///
/// The grids are declared `uint64_t[]` and every use immediately casts to
/// `const uint8_t *`. Doing it here keeps the byte order question in one place:
/// this is a little-endian reinterpretation, matching the C's cast.
inline fn gridBytes(grid: [*]const u64, index: usize) [8]u8 {
    return @bitCast(grid[index]);
}

/// The same, for the **32-bit** grids.
///
/// `iq3xxs_grid` and `iq3s_grid` are `uint32_t[]`, not `uint64_t[]` -- four
/// values per entry rather than eight, which is why the 3-bit dequantizers
/// need two grid lookups per group of eight where the 2-bit ones need one.
///
/// Reading them through the 64-bit helper compiles, runs, and silently indexes
/// every second entry. It is the kind of mistake a round-trip test cannot see
/// and a golden checksum catches immediately.
inline fn gridBytes4(grid: [*]const u32, index: usize) [4]u8 {
    return @bitCast(grid[index]);
}

/// Same, for the grids whose values are signed.
inline fn gridBytesSigned(grid: [*]const u64, index: usize) [8]i8 {
    return @bitCast(grid[index]);
}

/// The sign flip every one of these formats applies.
inline fn signOf(signs: u8, j: usize) f32 {
    return if ((signs & c.kmask_iq2xs[j]) != 0) -1.0 else 1.0;
}

/// Ports `dequantize_row_iq2_xxs` (ggml-quants.c:2488 @c1d0e7a00).
pub export fn dequantize_row_iq2_xxs(x: [*c]const blocks.IQ2_XXS, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    var aux32: [2]u32 = undefined;
    const aux8: [*]const u8 = @ptrCast(&aux32);

    for (0..nb) |i| {
        const d = unfp16(x[i].d);

        for (0..QK_K / 32) |ib32| {
            @memcpy(std.mem.asBytes(&aux32), std.mem.sliceAsBytes(x[i].qs[4 * ib32 ..][0..4]));
            // The top nibble of the second word is the sub-block scale.
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32[1] >> 28))) * 0.25;
            for (0..4) |l| {
                const grid = gridBytes(@ptrCast(&c.iq2xxs_grid), aux8[l]);
                const signs = c.ksigns_iq2xs[(aux32[1] >> @intCast(7 * l)) & 127];
                for (0..8) |j| {
                    y[j] = db * @as(f32, @floatFromInt(grid[j])) * signOf(signs, j);
                }
                y += 8;
            }
        }
    }
}

/// Ports `dequantize_row_iq2_xs` (ggml-quants.c:2516 @c1d0e7a00).
pub export fn dequantize_row_iq2_xs(x: [*c]const blocks.IQ2_XS, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    var db: [2]f32 = undefined;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);

        for (0..QK_K / 32) |ib32| {
            db[0] = d * (0.5 + @as(f32, @floatFromInt(x[i].scales[ib32] & 0xf))) * 0.25;
            db[1] = d * (0.5 + @as(f32, @floatFromInt(x[i].scales[ib32] >> 4))) * 0.25;
            for (0..4) |l| {
                // Nine bits of index, seven of sign, in one u16.
                const grid = gridBytes(@ptrCast(&c.iq2xs_grid), x[i].qs[4 * ib32 + l] & 511);
                const signs = c.ksigns_iq2xs[x[i].qs[4 * ib32 + l] >> 9];
                for (0..8) |j| {
                    y[j] = db[l / 2] * @as(f32, @floatFromInt(grid[j])) * signOf(signs, j);
                }
                y += 8;
            }
        }
    }
}

/// Ports `dequantize_row_iq2_s` (ggml-quants.c:2543 @c1d0e7a00).
pub export fn dequantize_row_iq2_s(x: [*c]const blocks.IQ2_S, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    var db: [2]f32 = undefined;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        var qs: [*]const u8 = &x[i].qs;
        const qh: [*]const u8 = &x[i].qh;
        // The sign bytes live in the tail of `qs`; there is no field for them.
        var signs: [*]const u8 = qs + QK_K / 8;

        for (0..QK_K / 32) |ib32| {
            db[0] = d * (0.5 + @as(f32, @floatFromInt(x[i].scales[ib32] & 0xf))) * 0.25;
            db[1] = d * (0.5 + @as(f32, @floatFromInt(x[i].scales[ib32] >> 4))) * 0.25;
            for (0..4) |l| {
                const dl = db[l / 2];
                // Two more index bits per element come from qh, shifted into
                // place by 8-2l so each l takes a different pair.
                const hi = (@as(usize, qh[ib32]) << @intCast(8 - 2 * l)) & 0x300;
                const grid = gridBytes(@ptrCast(&c.iq2s_grid), @as(usize, qs[l]) | hi);
                for (0..8) |j| {
                    y[j] = dl * @as(f32, @floatFromInt(grid[j])) * signOf(signs[l], j);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

/// Ports `dequantize_row_iq3_xxs` (ggml-quants.c:2575 @c1d0e7a00).
pub export fn dequantize_row_iq3_xxs(x: [*c]const blocks.IQ3_XXS, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    var aux32: u32 = undefined;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        var qs: [*]const u8 = &x[i].qs;
        // Scales and signs share the tail of `qs`, past the grid indices.
        const scales_and_signs: [*]const u8 = qs + QK_K / 4;

        for (0..QK_K / 32) |ib32| {
            @memcpy(std.mem.asBytes(&aux32), (scales_and_signs + 4 * ib32)[0..4]);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32 >> 28))) * 0.5;
            for (0..4) |l| {
                const signs = c.ksigns_iq2xs[(aux32 >> @intCast(7 * l)) & 127];
                // Two grid entries per 8 outputs here, each contributing 4.
                const grid1 = gridBytes4(@ptrCast(&c.iq3xxs_grid), qs[2 * l + 0]);
                const grid2 = gridBytes4(@ptrCast(&c.iq3xxs_grid), qs[2 * l + 1]);
                for (0..4) |j| {
                    y[j + 0] = db * @as(f32, @floatFromInt(grid1[j])) * signOf(signs, j + 0);
                    y[j + 4] = db * @as(f32, @floatFromInt(grid2[j])) * signOf(signs, j + 4);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

/// Ports `dequantize_row_iq3_s` (ggml-quants.c:2607 @c1d0e7a00).
///
/// Two sub-blocks per iteration, because one `scales` byte covers both.
pub export fn dequantize_row_iq3_s(x: [*c]const blocks.IQ3_S, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        var qs: [*]const u8 = &x[i].qs;
        var qh: [*]const u8 = &x[i].qh;
        var signs: [*]const u8 = &x[i].signs;

        var ib32: usize = 0;
        while (ib32 < QK_K / 32) : (ib32 += 2) {
            // Note `1 + 2*scale`, not the `0.5 + scale` the iq2 formats use.
            const db1 = d * @as(f32, @floatFromInt(1 + 2 * @as(u32, x[i].scales[ib32 / 2] & 0xf)));
            const db2 = d * @as(f32, @floatFromInt(1 + 2 * @as(u32, x[i].scales[ib32 / 2] >> 4)));

            for (0..4) |l| {
                const g1 = @as(usize, qs[2 * l + 0]) | ((@as(usize, qh[0]) << @intCast(8 - 2 * l)) & 256);
                const g2 = @as(usize, qs[2 * l + 1]) | ((@as(usize, qh[0]) << @intCast(7 - 2 * l)) & 256);
                const grid1 = gridBytes4(@ptrCast(&c.iq3s_grid), g1);
                const grid2 = gridBytes4(@ptrCast(&c.iq3s_grid), g2);
                for (0..4) |j| {
                    y[j + 0] = db1 * @as(f32, @floatFromInt(grid1[j])) * signOf(signs[l], j + 0);
                    y[j + 4] = db1 * @as(f32, @floatFromInt(grid2[j])) * signOf(signs[l], j + 4);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;

            for (0..4) |l| {
                const g1 = @as(usize, qs[2 * l + 0]) | ((@as(usize, qh[1]) << @intCast(8 - 2 * l)) & 256);
                const g2 = @as(usize, qs[2 * l + 1]) | ((@as(usize, qh[1]) << @intCast(7 - 2 * l)) & 256);
                const grid1 = gridBytes4(@ptrCast(&c.iq3s_grid), g1);
                const grid2 = gridBytes4(@ptrCast(&c.iq3s_grid), g2);
                for (0..4) |j| {
                    y[j + 0] = db2 * @as(f32, @floatFromInt(grid1[j])) * signOf(signs[l], j + 0);
                    y[j + 4] = db2 * @as(f32, @floatFromInt(grid2[j])) * signOf(signs[l], j + 4);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

/// Ports `dequantize_row_iq1_s` (ggml-quants.c:2650 @c1d0e7a00).
///
/// The grid values are *signed* here, and every one is shifted by a constant
/// delta before scaling -- so a grid zero does not dequantize to zero.
pub export fn dequantize_row_iq1_s(x: [*c]const blocks.IQ1_S, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    for (0..nb) |i| {
        const d = unfp16(x[i].d);
        var qs: [*]const u8 = &x[i].qs;
        const qh: [*]const u16 = &x[i].qh;

        for (0..QK_K / 32) |ib| {
            const dl = d * @as(f32, @floatFromInt(2 * ((qh[ib] >> 12) & 7) + 1));
            const delta: f32 = if ((qh[ib] & 0x8000) != 0) -blocks.iq1s_delta else blocks.iq1s_delta;
            for (0..4) |l| {
                const idx = @as(usize, qs[l]) | (@as(usize, (qh[ib] >> @intCast(3 * l)) & 7) << 8);
                const grid = gridBytesSigned(@ptrCast(&c.iq1s_grid), idx);
                for (0..8) |j| {
                    y[j] = dl * (@as(f32, @floatFromInt(grid[j])) + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

/// Ports `dequantize_row_iq1_m` (ggml-quants.c:2675 @c1d0e7a00).
///
/// The only format with no scale field: its f16 is reassembled from four
/// nibbles scattered across the `scales` shorts. See `blocks.iq1mScale`.
pub export fn dequantize_row_iq1_m(x: [*c]const blocks.IQ1_M, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    var delta: [4]f32 = undefined;
    var idx: [4]u16 = undefined;

    for (0..nb) |i| {
        const sc: [*]const u16 = @ptrCast(@alignCast(&x[i].scales));
        const d = unfp16(blocks.iq1mScale(sc));

        var qs: [*]const u8 = &x[i].qs;
        var qh: [*]const u8 = &x[i].qh;

        for (0..QK_K / 32) |ib| {
            // Two scales per 32 weights, three bits each, packed six bits
            // apart and alternating which short they come from.
            const dl1 = d * @as(f32, @floatFromInt(2 * ((sc[ib / 2] >> @intCast(6 * (ib % 2) + 0)) & 0x7) + 1));
            const dl2 = d * @as(f32, @floatFromInt(2 * ((sc[ib / 2] >> @intCast(6 * (ib % 2) + 3)) & 0x7) + 1));

            idx[0] = qs[0] | ((@as(u16, qh[0]) << 8) & 0x700);
            idx[1] = qs[1] | ((@as(u16, qh[0]) << 4) & 0x700);
            idx[2] = qs[2] | ((@as(u16, qh[1]) << 8) & 0x700);
            idx[3] = qs[3] | ((@as(u16, qh[1]) << 4) & 0x700);
            delta[0] = if ((qh[0] & 0x08) != 0) -blocks.iq1m_delta else blocks.iq1m_delta;
            delta[1] = if ((qh[0] & 0x80) != 0) -blocks.iq1m_delta else blocks.iq1m_delta;
            delta[2] = if ((qh[1] & 0x08) != 0) -blocks.iq1m_delta else blocks.iq1m_delta;
            delta[3] = if ((qh[1] & 0x80) != 0) -blocks.iq1m_delta else blocks.iq1m_delta;

            for (0..2) |l| {
                const grid = gridBytesSigned(@ptrCast(&c.iq1s_grid), idx[l]);
                for (0..8) |j| {
                    y[j] = dl1 * (@as(f32, @floatFromInt(grid[j])) + delta[l]);
                }
                y += 8;
            }
            for (2..4) |l| {
                const grid = gridBytesSigned(@ptrCast(&c.iq1s_grid), idx[l]);
                for (0..8) |j| {
                    y[j] = dl2 * (@as(f32, @floatFromInt(grid[j])) + delta[l]);
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

/// Ports `dequantize_row_iq4_nl` (ggml-quants.c:2725 @c1d0e7a00).
///
/// Not a codebook of vectors like the others: `kvalues_iq4nl` is 16 scalar
/// levels, spaced non-uniformly to match the distribution of weights rather
/// than the number line.
pub export fn dequantize_row_iq4_nl(x: [*c]const blocks.IQ4_NL, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, c.QK4_NL) == 0);
    const nb: usize = @intCast(@divExact(k, c.QK4_NL));
    var y = y_in;

    for (0..nb) |i| {
        const qs: [*]const u8 = &x[i].qs;
        const d = unfp16(x[i].d);
        for (0..c.QK4_NL / 2) |j| {
            y[j + 0] = d * @as(f32, @floatFromInt(c.kvalues_iq4nl[qs[j] & 0xf]));
            y[j + c.QK4_NL / 2] = d * @as(f32, @floatFromInt(c.kvalues_iq4nl[qs[j] >> 4]));
        }
        y += c.QK4_NL;
    }
}

/// Ports `dequantize_row_iq4_xs` (ggml-quants.c:2743 @c1d0e7a00).
pub export fn dequantize_row_iq4_xs(x: [*c]const blocks.IQ4_XS, y_in: [*c]f32, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    const nb: usize = @intCast(@divExact(k, QK_K));
    var y = y_in;

    for (0..nb) |i| {
        var qs: [*]const u8 = &x[i].qs;
        const d = unfp16(x[i].d);

        for (0..QK_K / 32) |ib| {
            // Six bits of scale: four from scales_l, two more from scales_h.
            const ls = (@as(i32, (x[i].scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0xf)) |
                (@as(i32, (x[i].scales_h >> @intCast(2 * ib)) & 3) << 4);
            const dl = d * @as(f32, @floatFromInt(ls - 32));
            for (0..16) |j| {
                y[j + 0] = dl * @as(f32, @floatFromInt(c.kvalues_iq4nl[qs[j] & 0xf]));
                y[j + 16] = dl * @as(f32, @floatFromInt(c.kvalues_iq4nl[qs[j] >> 4]));
            }
            y += 32;
            qs += 16;
        }
    }
}
