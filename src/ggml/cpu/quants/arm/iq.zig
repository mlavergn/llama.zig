//! NEON dot products for the codebook formats: `iq2_*`, `iq3_*`, `iq1_*` and
//! `iq4_xs`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (v0.3.0,
//! `c1d0e7a00`), the `__ARM_NEON` arms of the dot products at lines 3631,
//! 3693, 3767, 3864, 3926, 4036, 4102 and 4256. Each function names the C it
//! replaces and the line it began at.
//!
//! # The grids are loaded, not computed
//!
//! Every kernel here indexes a codebook and loads eight bytes -- or four, for
//! the 32-bit `iq3` grids. `vld1_s8((const void *)(iq2xs_grid + idx))` reads
//! one `u64` entry as eight `i8` lanes; `vcombine_s8` then pairs two of them
//! into a 16-lane vector. Reading a 32-bit grid through the 64-bit path
//! silently indexes every second entry, which is why the two have separate
//! helpers.
//!
//! # Signs come from a table whose eighth bit is a parity
//!
//! `keven_signs_q2xs` holds 128 eight-byte entries. Bytes 0..6 are `-1` where
//! the index's bit is set, and byte 7 is chosen so the product of all eight is
//! `+1`. That parity is what lets a seven-bit field encode eight signs.
//! Generated here rather than transcribed, and checked against the C's literal
//! for all 128 entries.
//!
//! # The float epilogues are fused
//!
//! As in `k.zig` and `legacy.zig`: `-ffp-contract=on` turns every
//! `sumf += d * x` into one `fmla`. Each is `@mulAdd` here, and
//! `scripts/vecdot-prefix` is what settles the shape when it is ambiguous.

const std = @import("std");
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const blocks = @import("../../../quants/blocks.zig");
const neon = @import("neon.zig");
const c = impl.c;

const i8x8 = neon.i8x8;
const i8x16 = neon.i8x16;
const u8x8 = neon.u8x8;
const u8x16 = neon.u8x16;
const u16x8 = neon.u16x8;
const i16x8 = neon.i16x8;
const i32x4 = neon.i32x4;
const u32x4 = neon.u32x4;

inline fn f(h: u16) f32 {
    return convert.cpuFp16ToFp32(h);
}

inline fn as(comptime Block: type, p: ?*const anyopaque) [*]const Block {
    return @ptrCast(@alignCast(p.?));
}

/// The `q8_K` operand's quantized bytes at an offset.
inline fn q8At(y: *const blocks.Q8_K, off: usize) [*]const u8 {
    return @as([*]const u8, @ptrCast(&y.qs)) + off;
}

/// Four 16-lane `q8` vectors, the `ggml_vld1q_s8_x4` every kernel opens with.
inline fn q8x4(y: *const blocks.Q8_K, off: usize) [4]i8x16 {
    var out: [4]i8x16 = undefined;
    inline for (0..4) |k| out[k] = neon.load(i8x16, q8At(y, off + k * 16));
    return out;
}

/// Ports `vld1_s8((const void *)(grid + idx))`: one 64-bit grid entry as eight
/// signed lanes.
inline fn grid8(grid: [*]const u64, idx: usize) i8x8 {
    return @bitCast(grid[idx]);
}

/// Two grid entries combined into one 16-lane vector, `a` low.
inline fn grid16(grid: [*]const u64, a: usize, b: usize) i8x16 {
    return neon.combine(grid8(grid, a), grid8(grid, b));
}

/// Ports `ggml_vld1q_u32(w,x,y,z)` over the **32-bit** `iq3` grids: four
/// entries into one 16-lane vector.
inline fn grid4x4(grid: [*]const u32, g0: usize, g1: usize, g2: usize, g3: usize) i8x16 {
    const v: u32x4 = .{ grid[g0], grid[g1], grid[g2], grid[g3] };
    return @bitCast(v);
}

/// Ports `keven_signs_q2xs` (arch/arm/quants.c:3595 @c1d0e7a00), read as `uint64_t *`.
///
/// Entry `i` has `-1` in byte `j` where bit `j` of `i` is set, for `j` in
/// 0..6, and byte 7 set so the eight bytes multiply to `+1`.
const keven_signs = blk: {
    @setEvalBranchQuota(20000);
    var t: [128]u64 = undefined;
    for (0..128) |i| {
        var bytes: [8]i8 = undefined;
        var prod: i32 = 1;
        for (0..7) |j| {
            bytes[j] = if ((i >> j) & 1 != 0) -1 else 1;
            prod *= bytes[j];
        }
        bytes[7] = if (prod == 1) 1 else -1;
        t[i] = @bitCast(bytes);
    }
    break :blk t;
};

/// The sign vector for a seven-bit index.
inline fn signs16(a: usize, b: usize) i8x16 {
    return neon.combine(
        @as(i8x8, @bitCast(keven_signs[a])),
        @as(i8x8, @bitCast(keven_signs[b])),
    );
}

/// Ports `k_mask1` (arch/arm/quants.c:3782 @c1d0e7a00) and `k_mask2`
/// (arch/arm/quants.c:3786 @c1d0e7a00), shared by `iq2_s`
/// and `iq3_s`.
///
/// `mask1` broadcasts each of four sign bytes across eight lanes; `mask2`
/// selects one bit per lane. Together they turn a 32-bit sign field into a
/// per-lane `0x00`/`0xFF` mask, which `vorrq_u8(.., 1)` then makes `+1`/`-1`.
const k_mask1_lo: u8x16 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
const k_mask1_hi: u8x16 = .{ 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 };
const k_mask2: u8x16 = .{ 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };

/// Expands a 32-bit sign field into two `+1`/`-1` lane vectors.
///
/// Parameters:
/// - `bits`: the packed signs, four bytes.
///
/// Return: `{ low 16 lanes, high 16 lanes }`, each lane `+1` or `-1`.
inline fn expandSigns(bits: u32) [2]i8x16 {
    const m1: u8x16 = @splat(1);
    const v: u8x16 = @bitCast(@as(u32x4, @splat(bits)));

    const lo_sel = neon.@"and"(neon.qtbl1q_u8(v, k_mask1_lo), k_mask2);
    const hi_sel = neon.@"and"(neon.qtbl1q_u8(v, k_mask1_hi), k_mask2);

    return .{
        @bitCast(neon.orr(neon.ceq(lo_sel, k_mask2), m1)),
        @bitCast(neon.orr(neon.ceq(hi_sel, k_mask2), m1)),
    };
}

/// `ggml_vdotq_s32(zero, a, b)` reduced to a scalar.
inline fn dotv(a: i8x16, b: i8x16) i32 {
    const zero: i32x4 = @splat(0);
    return neon.addvq_s32(neon.dotq_s32(zero, a, b));
}

/// Two chained dot products reduced to a scalar.
inline fn dotv2(a0: i8x16, b0: i8x16, a1: i8x16, b1: i8x16) i32 {
    const zero: i32x4 = @splat(0);
    return neon.addvq_s32(neon.dotq_s32(neon.dotq_s32(zero, a0, b0), a1, b1));
}

/// Ports `ggml_vec_dot_iq2_xxs_q8_K` (arch/arm/quants.c:3631 @c1d0e7a00).
pub export fn ggml_vec_dot_iq2_xxs_q8_K(
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
    const grid: [*]const u64 = @ptrCast(&c.iq2xxs_grid);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;
        var q2: usize = 0;
        var q8: usize = 0;

        // Two scalar accumulators, summed at the end -- the scale is a float
        // here (`0.5f + (aux32 >> 28)`), so this is not an integer sum.
        var sumf1: f32 = 0;
        var sumf2: f32 = 0;

        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const b = q8x4(&y[i], q8);
            q8 += 64;

            var aux32: [4]u32 = undefined;
            @memcpy(std.mem.asBytes(&aux32), std.mem.sliceAsBytes(x[i].qs[q2..][0..8]));
            const aux8: [16]u8 = @bitCast(aux32);
            q2 += 8;

            const w0 = grid16(grid, aux8[0], aux8[1]);
            const w1 = grid16(grid, aux8[2], aux8[3]);
            const w2 = grid16(grid, aux8[8], aux8[9]);
            const w3 = grid16(grid, aux8[10], aux8[11]);

            const s0 = signs16((aux32[1] >> 0) & 127, (aux32[1] >> 7) & 127);
            const s1 = signs16((aux32[1] >> 14) & 127, (aux32[1] >> 21) & 127);
            const s2 = signs16((aux32[3] >> 0) & 127, (aux32[3] >> 7) & 127);
            const s3 = signs16((aux32[3] >> 14) & 127, (aux32[3] >> 21) & 127);

            const p1 = dotv2(neon.mul(w0, s0), b[0], neon.mul(w1, s1), b[1]);
            const p2 = dotv2(neon.mul(w2, s2), b[2], neon.mul(w3, s3), b[3]);

            sumf1 = @mulAdd(f32, @floatFromInt(p1), 0.5 + @as(f32, @floatFromInt(aux32[1] >> 28)), sumf1);
            sumf2 = @mulAdd(f32, @floatFromInt(p2), 0.5 + @as(f32, @floatFromInt(aux32[3] >> 28)), sumf2);
        }

        sumf = @mulAdd(f32, d, sumf1 + sumf2, sumf);
    }

    s[0] = 0.25 * sumf;
}

/// Ports `ggml_vec_dot_iq2_xs_q8_K` (arch/arm/quants.c:3693 @c1d0e7a00).
///
/// Unlike `iq2_xxs`, the scales are integers, so the whole super-block
/// accumulates in `i32` lanes with `vmlaq_s32` and only the epilogue is float.
pub export fn ggml_vec_dot_iq2_xs_q8_K(
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
    const grid: [*]const u64 = @ptrCast(&c.iq2xs_grid);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;

        // The eight scale bytes hold sixteen four-bit fields; zipping the low
        // and high nibbles interleaves them back into block order, then
        // `2*v + 1` gives the odd scale.
        const scales8 = neon.loadFrom(u8x8, &x[i].scales);
        const nib: u8x8 = @splat(0xf);
        const scales_l = neon.@"and"(scales8, nib);
        const scales_h = neon.shrN(scales8, 4);
        var scales = neon.combine(
            neon.zip1_u8(scales_l, scales_h),
            neon.zip2_u8(scales_l, scales_h),
        );
        const ones: u8x16 = @splat(1);
        scales = neon.add(neon.shlN(scales, 1), ones);

        const scales1: u16x8 = neon.movl_u8(neon.low(scales));
        const scales2: u16x8 = neon.movl_u8(neon.high(scales));
        const scales32 = [4]i32x4{
            @bitCast(neon.movl_u16(neon.low(scales1))),
            @bitCast(neon.movl_u16(neon.high(scales1))),
            @bitCast(neon.movl_u16(neon.low(scales2))),
            @bitCast(neon.movl_u16(neon.high(scales2))),
        };

        var sumi: i32x4 = @splat(0);
        var q2: usize = 0;
        var q8: usize = 0;

        for (0..c.QK_K / 64) |ib64| {
            const b = q8x4(&y[i], q8);
            q8 += 64;

            var u: [4]i8x16 = undefined;
            inline for (0..4) |k| {
                const a = x[i].qs[q2 + 2 * k];
                const bb = x[i].qs[q2 + 2 * k + 1];
                const g = grid16(grid, a & 511, bb & 511);
                const sg = signs16(a >> 9, bb >> 9);
                u[k] = neon.mul(g, sg);
            }

            const zero: i32x4 = @splat(0);
            const p1 = neon.dotq_s32(zero, u[0], b[0]);
            const p2 = neon.dotq_s32(zero, u[1], b[1]);
            const p3 = neon.dotq_s32(zero, u[2], b[2]);
            const p4 = neon.dotq_s32(zero, u[3], b[3]);
            const p = neon.paddq_s32(neon.paddq_s32(p1, p2), neon.paddq_s32(p3, p4));

            sumi = neon.mla_s32(sumi, p, scales32[ib64]);
            q2 += 8;
        }

        sumf = @mulAdd(f32, d, @as(f32, @floatFromInt(neon.addvq_s32(sumi))), sumf);
    }

    s[0] = 0.125 * sumf;
}

/// Ports `ggml_vec_dot_iq2_s_q8_K` (arch/arm/quants.c:3767 @c1d0e7a00).
///
/// Ten-bit grid indices -- eight in `qs`, two more shifted out of `qh` -- and
/// signs from a bit field rather than the parity table.
pub export fn ggml_vec_dot_iq2_s_q8_K(
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
    const grid: [*]const u64 = @ptrCast(&c.iq2s_grid);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;

        var qs: usize = 0;
        // The signs alias the second half of `qs`, read as `u16`.
        var sgn: usize = c.QK_K / 8;
        var q8: usize = 0;

        var sumi1: i32 = 0;
        var sumi2: i32 = 0;

        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const b = q8x4(&y[i], q8);
            q8 += 64;

            // The high-bit shift *decreases* with the index: 8, 6, 4, 2.
            var u: [4]i8x16 = undefined;
            inline for (0..2) |half| {
                const h = x[i].qh[ib32 + half];
                const lo_i = x[i].qs[qs + 4 * half + 0] |
                    ((@as(usize, h) << 8) & 0x300);
                const hi_i = x[i].qs[qs + 4 * half + 1] |
                    ((@as(usize, h) << 6) & 0x300);
                const lo_j = x[i].qs[qs + 4 * half + 2] |
                    ((@as(usize, h) << 4) & 0x300);
                const hi_j = x[i].qs[qs + 4 * half + 3] |
                    ((@as(usize, h) << 2) & 0x300);
                u[2 * half + 0] = grid16(grid, lo_i, hi_i);
                u[2 * half + 1] = grid16(grid, lo_j, hi_j);
            }
            qs += 8;

            inline for (0..2) |half| {
                const w0 = std.mem.readInt(u16, x[i].qs[sgn + 4 * half ..][0..2], .little);
                const w1 = std.mem.readInt(u16, x[i].qs[sgn + 4 * half + 2 ..][0..2], .little);
                const vs = expandSigns(@as(u32, w0) | (@as(u32, w1) << 16));
                u[2 * half + 0] = neon.mul(vs[0], u[2 * half + 0]);
                u[2 * half + 1] = neon.mul(vs[1], u[2 * half + 1]);
            }
            sgn += 8;

            const p1 = dotv(u[0], b[0]);
            const p2 = dotv(u[1], b[1]);
            const p3 = dotv(u[2], b[2]);
            const p4 = dotv(u[3], b[3]);

            sumi1 += p1 * (1 + 2 * @as(i32, x[i].scales[ib32 + 0] & 0xf));
            sumi2 += p2 * (1 + 2 * @as(i32, x[i].scales[ib32 + 0] >> 4));
            sumi1 += p3 * (1 + 2 * @as(i32, x[i].scales[ib32 + 1] & 0xf));
            sumi2 += p4 * (1 + 2 * @as(i32, x[i].scales[ib32 + 1] >> 4));
        }

        sumf = @mulAdd(f32, d, @as(f32, @floatFromInt(sumi1 + sumi2)), sumf);
    }

    s[0] = 0.125 * sumf;
}

/// Ports `ggml_vec_dot_iq3_xxs_q8_K` (arch/arm/quants.c:3864 @c1d0e7a00).
///
/// The grid is 32-bit here, so four entries fill one vector -- see `grid4x4`.
pub export fn ggml_vec_dot_iq3_xxs_q8_K(
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
    const grid: [*]const u32 = @ptrCast(&c.iq3xxs_grid);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;

        var q3: usize = 0;
        var gas: usize = c.QK_K / 4;
        var q8: usize = 0;

        var sumf1: f32 = 0;
        var sumf2: f32 = 0;

        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const b = q8x4(&y[i], q8);
            q8 += 64;

            var aux32: [2]u32 = undefined;
            @memcpy(std.mem.asBytes(&aux32), std.mem.sliceAsBytes(x[i].qs[gas..][0..8]));
            gas += 8;

            var g: [4]i8x16 = undefined;
            inline for (0..4) |k| {
                g[k] = grid4x4(
                    grid,
                    x[i].qs[q3 + 4 * k + 0],
                    x[i].qs[q3 + 4 * k + 1],
                    x[i].qs[q3 + 4 * k + 2],
                    x[i].qs[q3 + 4 * k + 3],
                );
            }
            q3 += 16;

            const s0 = signs16((aux32[0] >> 0) & 127, (aux32[0] >> 7) & 127);
            const s1 = signs16((aux32[0] >> 14) & 127, (aux32[0] >> 21) & 127);
            const s2 = signs16((aux32[1] >> 0) & 127, (aux32[1] >> 7) & 127);
            const s3 = signs16((aux32[1] >> 14) & 127, (aux32[1] >> 21) & 127);

            const p1 = dotv2(neon.mul(s0, g[0]), b[0], neon.mul(s1, g[1]), b[1]);
            const p2 = dotv2(neon.mul(s2, g[2]), b[2], neon.mul(s3, g[3]), b[3]);

            sumf1 = @mulAdd(f32, @floatFromInt(p1), 0.5 + @as(f32, @floatFromInt(aux32[0] >> 28)), sumf1);
            sumf2 = @mulAdd(f32, @floatFromInt(p2), 0.5 + @as(f32, @floatFromInt(aux32[1] >> 28)), sumf2);
        }

        sumf = @mulAdd(f32, d, sumf1 + sumf2, sumf);
    }

    s[0] = 0.5 * sumf;
}

/// Ports `k_shift` (arch/arm/quants.c:3952 @c1d0e7a00), `iq3_s`'s index builder.
///
/// Eight *descending* shifts, so `qh`'s bit `j` lands at position 8 for lane
/// `j` -- which the `& 256` then keeps.
const k_shift_iq3s: i16x8 = .{ 8, 7, 6, 5, 4, 3, 2, 1 };

/// Ports `ggml_vec_dot_iq3_s_q8_K` (arch/arm/quants.c:3926 @c1d0e7a00).
pub export fn ggml_vec_dot_iq3_s_q8_K(
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
    const grid: [*]const u32 = @ptrCast(&c.iq3s_grid);

    const m256: u16x8 = @splat(256);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = f(x[i].d) * y[i].d;

        var qs: usize = 0;
        var sgn: usize = 0;
        var q8: usize = 0;

        var scales32: [2]u32 = undefined;
        @memcpy(std.mem.asBytes(&scales32)[0..4], x[i].scales[0..4]);
        scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
        scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;
        const scales8: [8]u8 = @bitCast(scales32);

        var sumi1: i32 = 0;
        var sumi2: i32 = 0;

        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const b = q8x4(&y[i], q8);
            q8 += 64;

            const idx_l = neon.loadFrom(u8x16, x[i].qs[qs..].ptr);
            qs += 16;

            var g: [4]i8x16 = undefined;
            inline for (0..2) |half| {
                const qh: u16x8 = @splat(x[i].qh[ib32 + half]);
                const hi = neon.@"and"(@as(u16x8, @bitCast(neon.shlq(qh, k_shift_iq3s))), m256);
                const base: u16x8 = if (half == 0)
                    neon.movl_u8(neon.low(idx_l))
                else
                    neon.movl_u8(neon.high(idx_l));
                const index: [8]u16 = neon.orr(base, hi);

                g[2 * half + 0] = grid4x4(grid, index[0], index[1], index[2], index[3]);
                g[2 * half + 1] = grid4x4(grid, index[4], index[5], index[6], index[7]);
            }

            inline for (0..2) |half| {
                const w0 = std.mem.readInt(u16, x[i].signs[sgn + 4 * half ..][0..2], .little);
                const w1 = std.mem.readInt(u16, x[i].signs[sgn + 4 * half + 2 ..][0..2], .little);
                const vs = expandSigns(@as(u32, w0) | (@as(u32, w1) << 16));
                g[2 * half + 0] = neon.mul(vs[0], g[2 * half + 0]);
                g[2 * half + 1] = neon.mul(vs[1], g[2 * half + 1]);
            }
            sgn += 8;

            const p1 = dotv2(g[0], b[0], g[1], b[1]);
            const p2 = dotv2(g[2], b[2], g[3], b[3]);

            sumi1 += p1 * scales8[ib32 / 2 + 0];
            sumi2 += p2 * scales8[ib32 / 2 + 4];
        }

        sumf = @mulAdd(f32, d, @as(f32, @floatFromInt(sumi1 + sumi2)), sumf);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_iq1_s_q8_K` (arch/arm/quants.c:4036 @c1d0e7a00).
///
/// The grid values are already signed, so there is no sign field. Instead a
/// per-group `delta` of `+/-1` weights the right operand's group sums --
/// `sumi3` -- which the `IQ1S_DELTA` offset then scales.
pub export fn ggml_vec_dot_iq1_s_q8_K(
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
    const grid: [*]const u64 = @ptrCast(&c.iq1s_grid);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        var qs: usize = 0;
        var q8: usize = 0;

        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        var sumi3: i32 = 0;

        var ib: usize = 0;
        while (ib < c.QK_K / 32) : (ib += 2) {
            // The four shifts are 8, 5, 2 and -1: the last is a *right* shift,
            // which is why it reads `>> 1` in the C rather than `<< -1`.
            var g: [4]i8x16 = undefined;
            inline for (0..2) |half| {
                const h: usize = x[i].qh[ib + half];
                const k0 = x[i].qs[qs + 4 * half + 0] | ((h << 8) & 0x700);
                const k1 = x[i].qs[qs + 4 * half + 1] | ((h << 5) & 0x700);
                const k2 = x[i].qs[qs + 4 * half + 2] | ((h << 2) & 0x700);
                const k3 = x[i].qs[qs + 4 * half + 3] | ((h >> 1) & 0x700);
                g[2 * half + 0] = grid16(grid, k0, k1);
                g[2 * half + 1] = grid16(grid, k2, k3);
            }
            qs += 8;

            const b = q8x4(&y[i], q8);
            q8 += 64;

            const p1 = dotv2(g[0], b[0], g[1], b[1]);
            const p2 = dotv2(g[2], b[2], g[3], b[3]);

            const ls1: i32 = 2 * @as(i32, (x[i].qh[ib + 0] >> 12) & 7) + 1;
            const ls2: i32 = 2 * @as(i32, (x[i].qh[ib + 1] >> 12) & 7) + 1;

            sumi1 += p1 * ls1;
            sumi2 += p2 * ls2;
            sumi3 += (@as(i32, y[i].bsums[2 * ib + 0]) + y[i].bsums[2 * ib + 1]) * ls1 *
                (if (x[i].qh[ib + 0] & 0x8000 != 0) @as(i32, -1) else 1) +
                (@as(i32, y[i].bsums[2 * ib + 2]) + y[i].bsums[2 * ib + 3]) * ls2 *
                    (if (x[i].qh[ib + 1] & 0x8000 != 0) @as(i32, -1) else 1);
        }

        const inner = @as(f32, @floatFromInt(sumi1 + sumi2)) +
            blocks.iq1s_delta * @as(f32, @floatFromInt(sumi3));
        sumf = @mulAdd(f32, y[i].d * f(x[i].d), inner, sumf);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_iq1_m_q8_K` (arch/arm/quants.c:4102 @c1d0e7a00).
///
/// Like `iq1_s`, but the delta is per *quarter*-group and comes from a table of
/// four sign patterns indexed by two bits of `qh`. The scale is scattered
/// across four nibbles of `scales` and reassembled by `blocks.iq1mScale`.
pub export fn ggml_vec_dot_iq1_m_q8_K(
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
    const grid: [*]const u64 = @ptrCast(&c.iq1s_grid);

    const mask: i32x4 = @splat(0x7);
    const mone: i32x4 = @splat(1);
    const mzero: i32x4 = @splat(0);

    // The four sign patterns: each half of the vector is all `+1` or all `-1`.
    const deltas = [4]i8x16{
        neon.combine(@as(i8x8, @splat(1)), @as(i8x8, @splat(1))),
        neon.combine(@as(i8x8, @splat(-1)), @as(i8x8, @splat(1))),
        neon.combine(@as(i8x8, @splat(1)), @as(i8x8, @splat(-1))),
        neon.combine(@as(i8x8, @splat(-1)), @as(i8x8, @splat(-1))),
    };

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        var qs: usize = 0;
        var qh: usize = 0;
        var q8: usize = 0;

        const sc: *const [4]u16 = @ptrCast(@alignCast(&x[i].scales));
        const scale = blocks.iq1mScale(sc);

        var sumi1: i32x4 = mzero;
        var sumi2: i32x4 = mzero;

        var ib: usize = 0;
        while (ib < c.QK_K / 32) : (ib += 2) {
            var g: [4]i8x16 = undefined;
            inline for (0..4) |k| {
                const h: usize = x[i].qh[qh + k];
                const k0 = x[i].qs[qs + 2 * k + 0] | ((h << 8) & 0x700);
                const k1 = x[i].qs[qs + 2 * k + 1] | ((h << 4) & 0x700);
                g[k] = grid16(grid, k0, k1);
            }

            const b = q8x4(&y[i], q8);
            q8 += 64;

            const p1 = neon.paddq_s32(
                neon.dotq_s32(mzero, g[0], b[0]),
                neon.dotq_s32(mzero, g[1], b[1]),
            );
            const p2 = neon.paddq_s32(
                neon.dotq_s32(mzero, g[2], b[2]),
                neon.dotq_s32(mzero, g[3], b[3]),
            );
            const p12 = neon.paddq_s32(p1, p2);

            // Two bits per quarter-group, gathered from `qh` four bytes at a
            // time, indexing `deltas`.
            const qh32 = std.mem.readInt(u32, x[i].qh[qh..][0..4], .little);
            const aux32 = ((qh32 >> 3) & 0x01010101) | ((qh32 >> 6) & 0x02020202);
            const aux8: [4]u8 = @bitCast(aux32);

            const p3 = neon.paddq_s32(
                neon.dotq_s32(mzero, deltas[aux8[0]], b[0]),
                neon.dotq_s32(mzero, deltas[aux8[1]], b[1]),
            );
            const p4 = neon.paddq_s32(
                neon.dotq_s32(mzero, deltas[aux8[2]], b[2]),
                neon.dotq_s32(mzero, deltas[aux8[3]], b[3]),
            );
            const p34 = neon.paddq_s32(p3, p4);

            const raw: i32x4 = .{
                @as(i32, sc[ib / 2] >> 0),
                @as(i32, sc[ib / 2] >> 3),
                @as(i32, sc[ib / 2] >> 6),
                @as(i32, sc[ib / 2] >> 9),
            };
            const scales_4 = neon.add(neon.shlN(neon.@"and"(raw, mask), 1), mone);

            sumi1 = neon.mla_s32(sumi1, scales_4, p12);
            sumi2 = neon.mla_s32(sumi2, scales_4, p34);

            qs += 8;
            qh += 4;
        }

        const inner = @as(f32, @floatFromInt(neon.addvq_s32(sumi1))) +
            blocks.iq1m_delta * @as(f32, @floatFromInt(neon.addvq_s32(sumi2)));
        sumf = @mulAdd(f32, y[i].d * f(scale), inner, sumf);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_iq4_xs_q8_K` (arch/arm/quants.c:4256 @c1d0e7a00).
///
/// Four-bit indices into `kvalues_iq4nl`, with six-bit scales biased by 32 and
/// split between `scales_l` and `scales_h`. Note the two halves of a pair take
/// `h << 4` and `h << 2`, which is easy to transpose.
pub export fn ggml_vec_dot_iq4_xs_q8_K(
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
    std.debug.assert(@rem(n, c.QK_K) == 0);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ4_XS, vx);
    const y = as(blocks.Q8_K, vy);
    const nb = @divTrunc(n, c.QK_K);

    const values = neon.loadFrom(i8x16, @ptrCast(&c.kvalues_iq4nl));
    const m4b: u8x16 = @splat(0x0f);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |ibl| {
        var q4: usize = 0;
        var q8: usize = 0;
        var h = x[ibl].scales_h;

        var sumi1: i32 = 0;
        var sumi2: i32 = 0;

        for (0..c.QK_K / 64) |ib| {
            const q4bits0 = neon.loadFrom(u8x16, x[ibl].qs[q4..].ptr);
            const q4bits1 = neon.loadFrom(u8x16, x[ibl].qs[q4 + 16 ..].ptr);
            q4 += 32;

            const b = q8x4(&y[ibl], q8);
            q8 += 64;

            const w0 = neon.qtbl1q_s8(values, neon.@"and"(q4bits0, m4b));
            const w1 = neon.qtbl1q_s8(values, neon.shrN(q4bits0, 4));
            const w2 = neon.qtbl1q_s8(values, neon.@"and"(q4bits1, m4b));
            const w3 = neon.qtbl1q_s8(values, neon.shrN(q4bits1, 4));

            const prod_1 = dotv2(w0, b[0], w1, b[1]);
            const prod_2 = dotv2(w2, b[2], w3, b[3]);

            const sl = x[ibl].scales_l[ib];
            const ls1: i32 = (@as(i32, sl & 0xf) | (@as(i32, @as(u8, @truncate((h << 4) & 0x30))))) - 32;
            const ls2: i32 = (@as(i32, sl >> 4) | (@as(i32, @as(u8, @truncate((h << 2) & 0x30))))) - 32;
            h >>= 4;

            sumi1 += prod_1 * ls1;
            sumi2 += prod_2 * ls2;
        }

        sumf = @mulAdd(f32, f(x[ibl].d) * y[ibl].d, @as(f32, @floatFromInt(sumi1 + sumi2)), sumf);
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

test "iq2_xxs dot matches the C" {
    try testing.check(ggml_vec_dot_iq2_xxs_q8_K, c.GGML_TYPE_IQ2_XXS, c.GGML_TYPE_Q8_K, golden.iq2_xxs);
}

test "iq2_xs dot matches the C" {
    try testing.check(ggml_vec_dot_iq2_xs_q8_K, c.GGML_TYPE_IQ2_XS, c.GGML_TYPE_Q8_K, golden.iq2_xs);
}

test "iq2_s dot matches the C" {
    try testing.check(ggml_vec_dot_iq2_s_q8_K, c.GGML_TYPE_IQ2_S, c.GGML_TYPE_Q8_K, golden.iq2_s);
}

test "iq3_xxs dot matches the C" {
    try testing.check(ggml_vec_dot_iq3_xxs_q8_K, c.GGML_TYPE_IQ3_XXS, c.GGML_TYPE_Q8_K, golden.iq3_xxs);
}

test "iq3_s dot matches the C" {
    try testing.check(ggml_vec_dot_iq3_s_q8_K, c.GGML_TYPE_IQ3_S, c.GGML_TYPE_Q8_K, golden.iq3_s);
}

test "iq1_s dot matches the C" {
    try testing.check(ggml_vec_dot_iq1_s_q8_K, c.GGML_TYPE_IQ1_S, c.GGML_TYPE_Q8_K, golden.iq1_s);
}

test "iq1_m dot matches the C" {
    try testing.check(ggml_vec_dot_iq1_m_q8_K, c.GGML_TYPE_IQ1_M, c.GGML_TYPE_Q8_K, golden.iq1_m);
}

test "iq4_xs dot matches the C" {
    try testing.check(ggml_vec_dot_iq4_xs_q8_K, c.GGML_TYPE_IQ4_XS, c.GGML_TYPE_Q8_K, golden.iq4_xs);
}

test "the even-signs table matches the C's literal" {
    // Bytes 0..6 are -1 where the index's bit is set; byte 7 is the parity
    // that makes the product +1. Getting that eighth byte wrong is a sign
    // error on one element in eight.
    for (0..128) |i| {
        const bytes: [8]i8 = @bitCast(keven_signs[i]);
        var prod: i32 = 1;
        for (bytes) |b| {
            try std.testing.expect(b == 1 or b == -1);
            prod *= b;
        }
        try std.testing.expectEqual(@as(i32, 1), prod);
        for (0..7) |j| {
            const want: i8 = if ((i >> @intCast(j)) & 1 != 0) -1 else 1;
            try std.testing.expectEqual(want, bytes[j]);
        }
    }
}

test "the sign expansion turns a bit field into plus and minus one" {
    // Every lane must come out exactly +1 or -1, and bit `k` of the field must
    // control lane `k`.
    const vs = expandSigns(0b0000_0000_0000_0000_0000_0000_0000_0101);
    const lo: [16]i8 = vs[0];
    for (lo) |v| try std.testing.expect(v == 1 or v == -1);
    try std.testing.expectEqual(@as(i8, -1), lo[0]);
    try std.testing.expectEqual(@as(i8, 1), lo[1]);
    try std.testing.expectEqual(@as(i8, -1), lo[2]);
}
