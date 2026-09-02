//! Reference dot products for the codebook formats: `iq2_*`, `iq3_*`, `iq1_*`
//! and `iq4_xs`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/quants.c` (v0.3.0, `c1d0e7a00`),
//! the `_generic` dot products at lines 906-1330. Each function names the C it
//! replaces and the line it began at.
//!
//! # Unreachable on this target
//!
//! As with `legacy.zig`, `k.zig` and `ternary.zig`: `arch/arm/quants.c`
//! supplies the real entry points, so these `_generic` names are exported and
//! never called. `golden.zig` is their only gate.
//!
//! # Grid width is the trap
//!
//! `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid` and `iq1s_grid` are `uint64_t[]`,
//! eight packed bytes per entry. `iq3xxs_grid` and `iq3s_grid` are
//! `uint32_t[]`, **four**. Reading a 32-bit grid through a 64-bit accessor
//! compiles, runs, and silently indexes every second entry -- which is why the
//! two widths get separate helpers here, as they do in
//! `src/ggml/quants/iq_dequant.zig`.
//!
//! # The trailing scale factors are not cosmetic
//!
//! `iq2_xxs`, `iq2_xs` and `iq2_s` end in `*s = 0.125f * sumf`, `iq3_xxs` in
//! `0.25f * sumf`, and `iq3_s`, `iq1_s`, `iq1_m` and `iq4_xs` in plain `sumf`.
//! They fold the codebook's fixed-point convention into the result. Dropping
//! one is an eight- or four-fold error in that format alone, invisible to
//! everything except a value captured from the C.

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

/// One entry of a 64-bit grid, as the eight bytes it packs.
inline fn grid8(grid: [*]const u64, index: usize) [8]u8 {
    return @bitCast(grid[index]);
}

/// One entry of a 64-bit grid, signed. `iq1s_grid` is read this way.
inline fn grid8s(grid: [*]const u64, index: usize) [8]i8 {
    return @bitCast(grid[index]);
}

/// One entry of a **32-bit** grid, as four bytes. See the note above.
inline fn grid4(grid: [*]const u32, index: usize) [4]u8 {
    return @bitCast(grid[index]);
}

/// The `signs & kmask_iq2xs[j] ? -1 : 1` every codebook format applies.
inline fn signOf(signs: u8, j: usize) i32 {
    return if ((signs & c.kmask_iq2xs[j]) != 0) -1 else 1;
}

/// Ports `ggml_vec_dot_iq2_xxs_q8_K_generic` (ggml-cpu/quants.c:906 @c1d0e7a00).
///
/// Each group of 32 packs four 8-bit grid indices and a 28-bit sign field into
/// two words; the top four bits are the group's scale.
pub export fn ggml_vec_dot_iq2_xxs_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ2_XXS, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;
        var q2: usize = 0;
        var q8: usize = 0;
        var bsum: i32 = 0;

        for (0..c.QK_K / 32) |_| {
            // Two words read as eight bytes: the low four are grid indices,
            // the high word carries the signs and the scale.
            var aux32: [2]u32 = undefined;
            @memcpy(std.mem.asBytes(&aux32), std.mem.sliceAsBytes(x[i].qs[q2..][0..4]));
            const aux8: *const [8]u8 = @ptrCast(&aux32);
            q2 += 4;

            const ls: i32 = @intCast(2 * (aux32[1] >> 28) + 1);
            var sumi: i32 = 0;
            for (0..4) |l| {
                const g = grid8(@ptrCast(&c.iq2xxs_grid), aux8[l]);
                const signs = c.ksigns_iq2xs[(aux32[1] >> @intCast(7 * l)) & 127];
                for (0..8) |j| {
                    sumi += @as(i32, g[j]) * y[i].qs[q8 + j] * signOf(signs, j);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * @as(f32, @floatFromInt(bsum));
    }

    s[0] = 0.125 * sumf;
}

/// Ports `ggml_vec_dot_iq2_xs_q8_K_generic` (ggml-cpu/quants.c:948 @c1d0e7a00).
///
/// Nine-bit grid indices with the sign index in the top seven bits, and two
/// four-bit scales per group of 32 -- so the group is processed in two halves
/// with different scales.
pub export fn ggml_vec_dot_iq2_xs_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ2_XS, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;
        var q2: usize = 0;
        var q8: usize = 0;
        var bsum: i32 = 0;

        for (0..c.QK_K / 32) |ib32| {
            const sc = x[i].scales[ib32];
            const ls1: i32 = 2 * @as(i32, sc & 0xf) + 1;
            const ls2: i32 = 2 * @as(i32, sc >> 4) + 1;

            var sumi: i32 = 0;
            for (0..2) |l| {
                const q = x[i].qs[q2 + l];
                const g = grid8(@ptrCast(&c.iq2xs_grid), q & 511);
                const signs = c.ksigns_iq2xs[q >> 9];
                for (0..8) |j| {
                    sumi += @as(i32, g[j]) * y[i].qs[q8 + j] * signOf(signs, j);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;

            sumi = 0;
            for (2..4) |l| {
                const q = x[i].qs[q2 + l];
                const g = grid8(@ptrCast(&c.iq2xs_grid), q & 511);
                const signs = c.ksigns_iq2xs[q >> 9];
                for (0..8) |j| {
                    sumi += @as(i32, g[j]) * y[i].qs[q8 + j] * signOf(signs, j);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * @as(f32, @floatFromInt(bsum));
    }

    s[0] = 0.125 * sumf;
}

/// Ports `ggml_vec_dot_iq2_s_q8_K_generic` (ggml-cpu/quants.c:998 @c1d0e7a00).
///
/// Ten-bit grid indices: eight bits in `qs` and two more taken from `qh` by a
/// shift that *decreases* with `l` -- `qh[ib32] << (8 - 2*l) & 0x300`. The
/// signs live in a second half of `qs`, `QK_K/8` bytes in.
pub export fn ggml_vec_dot_iq2_s_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ2_S, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;
        var q8: usize = 0;
        var qs: usize = 0;
        // The C aliases `signs = qs + QK_K/8`, an offset into the same array.
        var signs: usize = c.QK_K / 8;

        var bsum: i32 = 0;
        for (0..c.QK_K / 32) |ib32| {
            const sc = x[i].scales[ib32];
            const ls1: i32 = 1 + 2 * @as(i32, sc & 0xf);
            const ls2: i32 = 1 + 2 * @as(i32, sc >> 4);

            var sumi1: i32 = 0;
            var sumi2: i32 = 0;

            for (0..2) |l| {
                const hi = (@as(u32, x[i].qh[ib32]) << @intCast(8 - 2 * l)) & 0x300;
                const g = grid8(@ptrCast(&c.iq2s_grid), x[i].qs[qs + l] | hi);
                for (0..8) |j| {
                    sumi1 += @as(i32, y[i].qs[q8 + j]) * g[j] * signOf(x[i].qs[signs + l], j);
                }
                q8 += 8;
            }
            for (2..4) |l| {
                const hi = (@as(u32, x[i].qh[ib32]) << @intCast(8 - 2 * l)) & 0x300;
                const g = grid8(@ptrCast(&c.iq2s_grid), x[i].qs[qs + l] | hi);
                for (0..8) |j| {
                    sumi2 += @as(i32, y[i].qs[q8 + j]) * g[j] * signOf(x[i].qs[signs + l], j);
                }
                q8 += 8;
            }

            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * @as(f32, @floatFromInt(bsum));
    }

    s[0] = 0.125 * sumf;
}

/// Ports `ggml_vec_dot_iq3_xxs_q8_K_generic` (ggml-cpu/quants.c:1050 @c1d0e7a00).
///
/// Two 32-bit grid lookups per group of eight, because `iq3xxs_grid` holds
/// four bytes per entry rather than eight. The sign field and scale come from
/// a word packed after the indices, at `QK_K/4`.
pub export fn ggml_vec_dot_iq3_xxs_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ3_XXS, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;
        var q3: usize = 0;
        var gas: usize = c.QK_K / 4;
        var q8: usize = 0;
        var bsum: i32 = 0;

        for (0..c.QK_K / 32) |_| {
            const aux32 = std.mem.readInt(u32, x[i].qs[gas..][0..4], .little);
            gas += @sizeOf(u32);

            const ls: i32 = @intCast(2 * (aux32 >> 28) + 1);
            var sumi: i32 = 0;
            for (0..4) |l| {
                const g1 = grid4(@ptrCast(&c.iq3xxs_grid), x[i].qs[q3 + 2 * l + 0]);
                const g2 = grid4(@ptrCast(&c.iq3xxs_grid), x[i].qs[q3 + 2 * l + 1]);
                const signs = c.ksigns_iq2xs[(aux32 >> @intCast(7 * l)) & 127];
                for (0..4) |j| {
                    sumi += @as(i32, g1[j]) * y[i].qs[q8 + j + 0] * signOf(signs, j + 0);
                    sumi += @as(i32, g2[j]) * y[i].qs[q8 + j + 4] * signOf(signs, j + 4);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * @as(f32, @floatFromInt(bsum));
    }

    s[0] = 0.25 * sumf;
}

/// Ports `ggml_vec_dot_iq3_s_q8_K_generic` (ggml-cpu/quants.c:1094 @c1d0e7a00).
///
/// Nine-bit indices into the 32-bit `iq3s_grid`. The two lookups in a pair use
/// shifts of `8 - 2*l` and `7 - 2*l`, one apart -- the high bit for the odd
/// index sits one place further along in `qh`. Groups are handled two at a
/// time so the two nibbles of one scale byte can be applied.
pub export fn ggml_vec_dot_iq3_s_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ3_S, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;
        var qs: usize = 0;
        var signs: usize = 0;
        var q8: usize = 0;
        var bsum: i32 = 0;

        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const sc = x[i].scales[ib32 / 2];
            const ls1: i32 = 2 * @as(i32, sc & 0xf) + 1;
            const ls2: i32 = 2 * @as(i32, sc >> 4) + 1;

            // The two halves differ only in which `qh` byte they read, so the
            // body is written once and run twice.
            inline for (0..2) |half| {
                const qh = x[i].qh[ib32 + half];
                var sumi: i32 = 0;
                for (0..4) |l| {
                    const hi1 = (@as(u32, qh) << @intCast(8 - 2 * l)) & 256;
                    const hi2 = (@as(u32, qh) << @intCast(7 - 2 * l)) & 256;
                    const g1 = grid4(@ptrCast(&c.iq3s_grid), x[i].qs[qs + 2 * l + 0] | hi1);
                    const g2 = grid4(@ptrCast(&c.iq3s_grid), x[i].qs[qs + 2 * l + 1] | hi2);
                    for (0..4) |j| {
                        sumi += @as(i32, g1[j]) * y[i].qs[q8 + j + 0] * signOf(x[i].signs[signs + l], j + 0);
                        sumi += @as(i32, g2[j]) * y[i].qs[q8 + j + 4] * signOf(x[i].signs[signs + l], j + 4);
                    }
                    q8 += 8;
                }
                qs += 8;
                signs += 4;
                bsum += sumi * (if (half == 0) ls1 else ls2);
            }
        }
        sumf += d * @as(f32, @floatFromInt(bsum));
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_iq1_s_q8_K_generic` (ggml-cpu/quants.c:1150 @c1d0e7a00).
///
/// One-bit weights whose grid values are already signed, so there is no sign
/// field. Instead each group carries a `delta` of +/-1 applied to the right
/// operand's group sums -- the `IQ1S_DELTA` offset that makes a "zero" weight
/// not exactly zero.
pub export fn ggml_vec_dot_iq1_s_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ1_S, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        var q8: usize = 0;
        var qs: usize = 0;

        var sumi: i32 = 0;
        var sumi1: i32 = 0;

        for (0..c.QK_K / 32) |ib| {
            const qh = x[i].qh[ib];
            const ls: i32 = 2 * @as(i32, (qh >> 12) & 7) + 1;
            const delta: i32 = if (qh & 0x8000 != 0) -1 else 1;

            var lsum: i32 = 0;
            for (0..4) |l| {
                const idx = @as(u32, x[i].qs[qs + l]) | ((@as(u32, (qh >> @intCast(3 * l)) & 7)) << 8);
                const g = grid8s(@ptrCast(&c.iq1s_grid), idx);
                for (0..8) |j| {
                    lsum += @as(i32, y[i].qs[q8 + j]) * g[j];
                }
                q8 += 8;
            }
            sumi += ls * lsum;
            sumi1 += ls * delta * (@as(i32, y[i].bsums[2 * ib + 0]) + y[i].bsums[2 * ib + 1]);
            qs += 4;
        }

        sumf += f(x[i].d) * y[i].d *
            (@as(f32, @floatFromInt(sumi)) + blocks.iq1s_delta * @as(f32, @floatFromInt(sumi1)));
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_iq1_m_q8_K_generic` (ggml-cpu/quants.c:1193 @c1d0e7a00).
///
/// Like `iq1_s`, but the block has no `d` field: the f16 scale is scattered
/// across four nibbles of `scales` and reassembled by `blocks.iq1mScale`. The
/// delta is per *quarter*-group rather than per group, which is why `sum2`
/// accumulates the right operand's raw sum separately from `sum1`.
pub export fn ggml_vec_dot_iq1_m_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ1_M, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sum1: [2]i32 = undefined;
    var sum2: [2]i32 = undefined;
    var delta: [4]i32 = undefined;

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |i| {
        var q8: usize = 0;
        var qs: usize = 0;
        var qh: usize = 0;

        // `scales` is eight bytes read as four `uint16_t` in the C.
        const sc: *const [4]u16 = @ptrCast(@alignCast(&x[i].scales));
        const scale = blocks.iq1mScale(sc);

        var sumi1: i32 = 0;
        var sumi2: i32 = 0;

        for (0..c.QK_K / 32) |ib| {
            delta[0] = if (x[i].qh[qh + 0] & 0x08 != 0) -1 else 1;
            delta[1] = if (x[i].qh[qh + 0] & 0x80 != 0) -1 else 1;
            delta[2] = if (x[i].qh[qh + 1] & 0x08 != 0) -1 else 1;
            delta[3] = if (x[i].qh[qh + 1] & 0x80 != 0) -1 else 1;

            sum1 = .{ 0, 0 };
            sum2 = .{ 0, 0 };

            for (0..4) |l| {
                const hi = (@as(u32, x[i].qh[qh + l / 2]) << @intCast(8 - 4 * (l % 2))) & 0x700;
                const g = grid8s(@ptrCast(&c.iq1s_grid), @as(u32, x[i].qs[qs + l]) | hi);
                var lsum1: i32 = 0;
                var lsum2: i32 = 0;
                for (0..8) |j| {
                    lsum1 += @as(i32, y[i].qs[q8 + j]) * g[j];
                    lsum2 += y[i].qs[q8 + j];
                }
                q8 += 8;
                sum1[l / 2] += lsum1;
                sum2[l / 2] += lsum2 * delta[l];
            }

            const shift1: u4 = @intCast(6 * (ib % 2) + 0);
            const shift2: u4 = @intCast(6 * (ib % 2) + 3);
            const ls1: i32 = 2 * @as(i32, (sc[ib / 2] >> shift1) & 0x7) + 1;
            const ls2: i32 = 2 * @as(i32, (sc[ib / 2] >> shift2) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += f(scale) * y[i].d *
            (@as(f32, @floatFromInt(sumi1)) + blocks.iq1m_delta * @as(f32, @floatFromInt(sumi2)));
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_iq4_xs_q8_K_generic` (ggml-cpu/quants.c:1283 @c1d0e7a00).
///
/// Four-bit indices into `kvalues_iq4nl`, with six-bit scales biased by 32 and
/// split between `scales_l` and `scales_h`. `h` is consumed four bits at a
/// time, and the shift differs between the two halves of the pair -- `h << 4`
/// then `h << 2` -- which is easy to transpose.
pub export fn ggml_vec_dot_iq4_xs_q8_K_generic(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ4_XS, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    var sumf: f32 = 0;
    for (0..@intCast(nb)) |ibl| {
        const d4d8 = f(x[ibl].d) * y[ibl].d;
        var h = x[ibl].scales_h;
        var qs: usize = 0;
        var q8: usize = 0;

        var ib: usize = 0;
        while (ib < c.QK_K / 32) : (ib += 2) {
            const sl = x[ibl].scales_l[ib / 2];
            const ls1: u8 = (sl & 0xf) | @as(u8, @truncate((h << 4) & 0x30));
            const ls2: u8 = (sl >> 4) | @as(u8, @truncate((h << 2) & 0x30));
            h >>= 4;

            const d1 = d4d8 * @as(f32, @floatFromInt(@as(i32, ls1) - 32));
            const d2 = d4d8 * @as(f32, @floatFromInt(@as(i32, ls2) - 32));

            inline for ([_]usize{ 0, 1 }) |pass| {
                var sumi1: i32 = 0;
                var sumi2: i32 = 0;
                for (0..16) |j| {
                    sumi1 += @as(i32, y[ibl].qs[q8 + j + 0]) * c.kvalues_iq4nl[x[ibl].qs[qs + j] & 0xf];
                    sumi2 += @as(i32, y[ibl].qs[q8 + j + 16]) * c.kvalues_iq4nl[x[ibl].qs[qs + j] >> 4];
                }
                sumf += (if (pass == 0) d1 else d2) * @as(f32, @floatFromInt(sumi1 + sumi2));
                qs += 16;
                q8 += 32;
            }
        }
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

test "iq2_xxs dot matches the C" {
    try testing.check(ggml_vec_dot_iq2_xxs_q8_K_generic, c.GGML_TYPE_IQ2_XXS, c.GGML_TYPE_Q8_K, golden.iq2_xxs);
}

test "iq2_xs dot matches the C" {
    try testing.check(ggml_vec_dot_iq2_xs_q8_K_generic, c.GGML_TYPE_IQ2_XS, c.GGML_TYPE_Q8_K, golden.iq2_xs);
}

test "iq2_s dot matches the C" {
    try testing.check(ggml_vec_dot_iq2_s_q8_K_generic, c.GGML_TYPE_IQ2_S, c.GGML_TYPE_Q8_K, golden.iq2_s);
}

test "iq3_xxs dot matches the C" {
    try testing.check(ggml_vec_dot_iq3_xxs_q8_K_generic, c.GGML_TYPE_IQ3_XXS, c.GGML_TYPE_Q8_K, golden.iq3_xxs);
}

test "iq3_s dot matches the C" {
    try testing.check(ggml_vec_dot_iq3_s_q8_K_generic, c.GGML_TYPE_IQ3_S, c.GGML_TYPE_Q8_K, golden.iq3_s);
}

test "iq1_s dot matches the C" {
    try testing.check(ggml_vec_dot_iq1_s_q8_K_generic, c.GGML_TYPE_IQ1_S, c.GGML_TYPE_Q8_K, golden.iq1_s);
}

test "iq1_m dot matches the C" {
    try testing.check(ggml_vec_dot_iq1_m_q8_K_generic, c.GGML_TYPE_IQ1_M, c.GGML_TYPE_Q8_K, golden.iq1_m);
}

test "iq4_xs dot matches the C" {
    try testing.check(ggml_vec_dot_iq4_xs_q8_K_generic, c.GGML_TYPE_IQ4_XS, c.GGML_TYPE_Q8_K, golden.iq4_xs);
}

test "the 32-bit grids are half the width of the 64-bit ones" {
    // Reading `iq3xxs_grid` through the 64-bit accessor compiles and silently
    // indexes every second entry, so the two widths are asserted apart.
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(@TypeOf(c.iq2xxs_grid[0])));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(@TypeOf(c.iq3xxs_grid[0])));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(@TypeOf(c.iq3s_grid[0])));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(@TypeOf(c.iq1s_grid[0])));
}
