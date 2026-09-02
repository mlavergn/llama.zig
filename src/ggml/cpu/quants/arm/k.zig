//! NEON dot products for the K-quant super-block formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (v0.3.0,
//! `c1d0e7a00`), the `__ARM_NEON` arms of the dot products at lines 1685,
//! 2018, 2334, 2864 and 2964. Each function names the C it replaces and the
//! line it began at.
//!
//! # The minima are handled in integers, before the weights
//!
//! Each of these formats carries a per-group minimum as well as a scale. The
//! NEON kernels apply the minima in one pass against `q8_K`'s `bsums` --
//! `vmull_s16` widening products summed with `vaddvq_s32` -- and only then walk
//! the weights. That split is why `sum` takes two contributions per super-block
//! and why they go in that order.
//!
//! # Where the accumulation is integer, order does not matter
//!
//! `isum` is an `i32` and every product feeding it is exact, so the many
//! `isum += vaddvq_s32(...) * scale[k]` lines can be grouped freely.
//!
//! # The float epilogues are fused
//!
//! Every `sum += d * isum` and `sumf -= dmin * mins` in these kernels is one
//! `fmla` in the shipped code: `-ffp-contract=on` fuses a multiply into the
//! add it feeds. Written as `sum += d * x` in Zig that is two roundings, and
//! all five kernels came out one ULP wrong until each was named with
//! `@mulAdd`.
//!
//! `scripts/vecdot-prefix` is what settles these when the shape is ambiguous
//! -- see the note on `iq4_nl` in `legacy.zig` for why an oracle rather than a
//! guess is the only sound way to place a fusion.

const std = @import("std");
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const blocks = @import("../../../quants/blocks.zig");
const neon = @import("neon.zig");
const c = impl.c;

const i8x16 = neon.i8x16;
const u8x16 = neon.u8x16;
const i16x8 = neon.i16x8;
const u16x8 = neon.u16x8;
const i32x4 = neon.i32x4;

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

/// Ports `ggml_vdotq_s32(vzero, a, b)` followed by `vaddvq_s32`, the scalar
/// dot product these kernels take a hundred times over.
inline fn dotv(a: i8x16, b: i8x16) i32 {
    const zero: i32x4 = @splat(0);
    return neon.addvq_s32(neon.dotq_s32(zero, a, b));
}

/// The widening product of eight `i16` pairs, summed into four `i32` lanes,
/// as `vmull_s16` is used throughout the NEON arm (arch/arm/quants.c:1968 @c1d0e7a00).
///
/// `vmull_s16` on each half and `vaddq_s32` between them. The widening is what
/// keeps a `mins * bsums` product from overflowing sixteen bits.
inline fn mullSum(a: i16x8, b: i16x8) i32x4 {
    return neon.add(
        neon.mull_s16(neon.low(a), neon.low(b)),
        neon.mull_s16(neon.high(a), neon.high(b)),
    );
}

/// Ports `ggml_vec_dot_q2_K_q8_K` (arch/arm/quants.c:1685 @c1d0e7a00).
///
/// Two-bit weights with a four-bit scale and four-bit minimum per group of
/// sixteen, both packed into the same byte -- `scales & 0xF` and
/// `scales >> 4`.
pub export fn ggml_vec_dot_q2_K_q8_K(
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

    const x = as(blocks.Q2_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    const m3: u8x16 = @splat(0x3);
    const m4: u8x16 = @splat(0xF);

    var sum: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = y[i].d * f(x[i].d);
        const dmin = -y[i].d * f(x[i].dmin);

        const mins_and_scales = neon.loadFrom(u8x16, &x[i].scales);
        const scales_v = neon.@"and"(mins_and_scales, m4);
        // The C spills the scales to a byte array and indexes it; the lanes
        // are read one at a time either way.
        const aux: [16]u8 = scales_v;
        const mins = neon.shrN(mins_and_scales, 4);

        const q8sums0 = neon.loadFrom(i16x8, &y[i].bsums);
        const q8sums1 = neon.loadFrom(i16x8, y[i].bsums[8..]);

        const mins16_0: i16x8 = @bitCast(@as(u16x8, neon.low(mins)));
        const mins16_1: i16x8 = @bitCast(@as(u16x8, neon.high(mins)));

        const s0 = mullSum(mins16_0, q8sums0);
        const s1 = mullSum(mins16_1, q8sums1);
        sum = @mulAdd(f32, dmin, @as(f32, @floatFromInt(neon.addvq_s32(neon.add(s0, s1)))), sum);

        var isum: i32 = 0;
        var is: usize = 0;
        var q2: usize = 0;
        var q8: usize = 0;

        for (0..c.QK_K / 128) |_| {
            const q2bits0 = neon.loadFrom(u8x16, x[i].qs[q2..].ptr);
            const q2bits1 = neon.loadFrom(u8x16, x[i].qs[q2 + 16 ..].ptr);
            q2 += 32;

            // Four shifts of the same two loaded vectors, each against a
            // freshly loaded pair of `q8` vectors. The C spells this as
            // `MULTIPLY_ACCUM_WITH_SCALE` plus three
            // `SHIFT_MULTIPLY_ACCUM_WITH_SCALE`s.
            inline for (0..4) |g| {
                const shift = 2 * g;
                const b0 = neon.load(i8x16, q8At(&y[i], q8));
                const b1 = neon.load(i8x16, q8At(&y[i], q8 + 16));
                q8 += 32;

                const w0: i8x16 = @bitCast(neon.@"and"(neon.shrN(q2bits0, shift), m3));
                const w1: i8x16 = @bitCast(neon.@"and"(neon.shrN(q2bits1, shift), m3));

                isum += dotv(w0, b0) * aux[is + 2 * g];
                isum += dotv(w1, b1) * aux[is + 1 + 2 * g];
            }
            is += 8;
        }

        sum = @mulAdd(f32, d, @as(f32, @floatFromInt(isum)), sum);
    }

    s[0] = sum;
}

/// Ports `ggml_vec_dot_q3_K_q8_K` (arch/arm/quants.c:2018 @c1d0e7a00).
///
/// Three-bit weights: two bits in `qs` and an **inverted** high bit in
/// `hmask`. The inversion is why the kernel uses `vbicq_u8(mask, qhbits)` --
/// `mask AND NOT qhbits` -- and then *subtracts*: a clear mask bit contributes
/// 4, and a set one contributes nothing.
pub export fn ggml_vec_dot_q3_K_q8_K(
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

    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;

    const x = as(blocks.Q3_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    const m3b: u8x16 = @splat(0x3);
    const m0: u8x16 = @splat(1);
    const m1 = neon.shlN(m0, 1);
    const m2 = neon.shlN(m0, 2);
    const m3 = neon.shlN(m0, 3);

    var sum: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = y[i].d * f(x[i].d);

        var qhbits0 = neon.loadFrom(u8x16, &x[i].hmask);
        var qhbits1 = neon.loadFrom(u8x16, x[i].hmask[16..]);

        var isum: i32 = 0;

        var aux: [3]u32 = undefined;
        @memcpy(std.mem.asBytes(&aux)[0..12], &x[i].scales);

        var utmp: [4]u32 = undefined;
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        // Read back as signed bytes, biased by 32, as the C's `scale` alias
        // does.
        var scale: [16]i8 = @bitCast(utmp);
        for (&scale) |*v| v.* -%= 32;

        var q3: usize = 0;
        var q8: usize = 0;
        var sc: usize = 0;

        for (0..c.QK_K / 128) |j| {
            const q3bits0 = neon.loadFrom(u8x16, x[i].qs[q3..].ptr);
            const q3bits1 = neon.loadFrom(u8x16, x[i].qs[q3 + 16 ..].ptr);
            q3 += 32;

            // Eight `q8` vectors, loaded as two groups of four.
            var b1: [4]i8x16 = undefined;
            var b2: [4]i8x16 = undefined;
            inline for (0..4) |k| b1[k] = neon.load(i8x16, q8At(&y[i], q8 + k * 16));
            q8 += 64;
            inline for (0..4) |k| b2[k] = neon.load(i8x16, q8At(&y[i], q8 + k * 16));
            q8 += 64;

            // First half: shifts 0 and 2, high bits from masks 1 and 2.
            {
                const h0: i8x16 = @bitCast(neon.shlN(neon.bic(m0, qhbits0), 2));
                const h1: i8x16 = @bitCast(neon.shlN(neon.bic(m0, qhbits1), 2));
                const h2: i8x16 = @bitCast(neon.shlN(neon.bic(m1, qhbits0), 1));
                const h3: i8x16 = @bitCast(neon.shlN(neon.bic(m1, qhbits1), 1));

                const w0 = neon.sub(@as(i8x16, @bitCast(neon.@"and"(q3bits0, m3b))), h0);
                const w1 = neon.sub(@as(i8x16, @bitCast(neon.@"and"(q3bits1, m3b))), h1);
                const w2 = neon.sub(@as(i8x16, @bitCast(neon.@"and"(neon.shrN(q3bits0, 2), m3b))), h2);
                const w3 = neon.sub(@as(i8x16, @bitCast(neon.@"and"(neon.shrN(q3bits1, 2), m3b))), h3);

                isum += dotv(w0, b1[0]) * scale[sc + 0];
                isum += dotv(w1, b1[1]) * scale[sc + 1];
                isum += dotv(w2, b1[2]) * scale[sc + 2];
                isum += dotv(w3, b1[3]) * scale[sc + 3];
                sc += 4;
            }

            // Second half: shifts 4 and 6, masks 4 and 8. Note the last two
            // shift *right* by one rather than left.
            {
                const h0: i8x16 = @bitCast(neon.bic(m2, qhbits0));
                const h1: i8x16 = @bitCast(neon.bic(m2, qhbits1));
                const h2: i8x16 = @bitCast(neon.shrN(neon.bic(m3, qhbits0), 1));
                const h3: i8x16 = @bitCast(neon.shrN(neon.bic(m3, qhbits1), 1));

                const w0 = neon.sub(@as(i8x16, @bitCast(neon.@"and"(neon.shrN(q3bits0, 4), m3b))), h0);
                const w1 = neon.sub(@as(i8x16, @bitCast(neon.@"and"(neon.shrN(q3bits1, 4), m3b))), h1);
                const w2 = neon.sub(@as(i8x16, @bitCast(neon.@"and"(neon.shrN(q3bits0, 6), m3b))), h2);
                const w3 = neon.sub(@as(i8x16, @bitCast(neon.@"and"(neon.shrN(q3bits1, 6), m3b))), h3);

                isum += dotv(w0, b2[0]) * scale[sc + 0];
                isum += dotv(w1, b2[1]) * scale[sc + 1];
                isum += dotv(w2, b2[2]) * scale[sc + 2];
                isum += dotv(w3, b2[3]) * scale[sc + 3];
                sc += 4;
            }

            // The 256 weights of a super-block use eight high-bit positions,
            // and only four masks exist -- so after the first 128 the mask
            // vectors are shifted down to reach the other four.
            if (j == 0) {
                qhbits0 = neon.shrN(qhbits0, 4);
                qhbits1 = neon.shrN(qhbits1, 4);
            }
        }

        sum = @mulAdd(f32, d, @as(f32, @floatFromInt(isum)), sum);
    }

    s[0] = sum;
}

/// The 12-byte scale/min unpacking `q4_K` and `q5_K` share, in the form the
/// NEON kernels use (arch/arm/quants.c:2358 and arch/arm/quants.c:2884).
///
/// `q5_K` shuffles all four words and reads the mins out of the second half;
/// `q4_K` keeps the mins in a separate `u32x2` because it only needs eight of
/// them. The two are *not* interchangeable, which is why this returns both.
inline fn unpackQ5K(packed_scales: *const [12]u8) [4]u32 {
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;

    var utmp: [4]u32 = undefined;
    @memcpy(std.mem.asBytes(&utmp)[0..12], packed_scales);

    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    return utmp;
}

/// Ports `ggml_vec_dot_q4_K_q8_K` (arch/arm/quants.c:2334 @c1d0e7a00), the `__ARM_NEON`
/// arm.
///
/// Note `sumf -= dmin * mins` **before** the weight loop and `sumf += d * ...`
/// after: two separate float statements per super-block, in that order.
pub export fn ggml_vec_dot_q4_K_q8_K(
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

    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;

    const x = as(blocks.Q4_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);
    const m4b: u8x16 = @splat(0xf);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = y[i].d * f(x[i].d);
        const dmin = y[i].d * f(x[i].dmin);

        // The sixteen group sums folded to eight, pairwise, to line up with
        // the eight mins.
        const q8sums = neon.paddq_s16(
            neon.loadFrom(i16x8, &y[i].bsums),
            neon.loadFrom(i16x8, y[i].bsums[8..]),
        );

        var utmp: [4]u32 = undefined;
        @memcpy(std.mem.asBytes(&utmp)[0..12], &x[i].scales);

        // Eight mins built in a `u32x2` rather than shuffled through `utmp`,
        // which is what makes this different from `q5_K`'s unpacking.
        var mins8: neon.u32x2 = @splat(0);
        mins8 = neon.setLane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = neon.setLane_u32(
            ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4),
            mins8,
            1,
        );
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const mins: i16x8 = @bitCast(@as(u16x8, neon.movl_u8(@bitCast(mins8))));
        const prod = mullSum(q8sums, mins);
        sumf = @mulAdd(f32, -dmin, @as(f32, @floatFromInt(neon.addvq_s32(prod))), sumf);

        const scales: [16]u8 = @bitCast(utmp);

        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        var q4: usize = 0;
        var q8: usize = 0;

        for (0..c.QK_K / 64) |j| {
            const q4bits0 = neon.loadFrom(u8x16, x[i].qs[q4..].ptr);
            const q4bits1 = neon.loadFrom(u8x16, x[i].qs[q4 + 16 ..].ptr);
            q4 += 32;

            // Low nibbles against the first 32 `q8` bytes.
            {
                const b0 = neon.load(i8x16, q8At(&y[i], q8));
                const b1 = neon.load(i8x16, q8At(&y[i], q8 + 16));
                q8 += 32;
                const w0: i8x16 = @bitCast(neon.@"and"(q4bits0, m4b));
                const w1: i8x16 = @bitCast(neon.@"and"(q4bits1, m4b));
                const zero: i32x4 = @splat(0);
                const p1 = neon.dotq_s32(neon.dotq_s32(zero, w0, b0), w1, b1);
                sumi1 += neon.addvq_s32(p1) * scales[2 * j + 0];
            }

            // High nibbles against the next 32.
            {
                const b0 = neon.load(i8x16, q8At(&y[i], q8));
                const b1 = neon.load(i8x16, q8At(&y[i], q8 + 16));
                q8 += 32;
                const w0: i8x16 = @bitCast(neon.shrN(q4bits0, 4));
                const w1: i8x16 = @bitCast(neon.shrN(q4bits1, 4));
                const zero: i32x4 = @splat(0);
                const p2 = neon.dotq_s32(neon.dotq_s32(zero, w0, b0), w1, b1);
                sumi2 += neon.addvq_s32(p2) * scales[2 * j + 1];
            }
        }

        sumf = @mulAdd(f32, d, @as(f32, @floatFromInt(sumi1 + sumi2)), sumf);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q5_K_q8_K` (arch/arm/quants.c:2864 @c1d0e7a00).
///
/// Five-bit weights: four in `qs` and the fifth taken from `qh` two bits at a
/// time, the mask vectors being shifted down by two after each group.
///
/// The epilogue is one statement -- `sumf += d * sumi - dmin * sumi_mins` --
/// where `q4_K` uses two. Kept as written.
pub export fn ggml_vec_dot_q5_K_q8_K(
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

    const x = as(blocks.Q5_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    const m4b: u8x16 = @splat(0xf);
    const mone: u8x16 = @splat(1);
    const mtwo: u8x16 = @splat(2);

    var sumf: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d = y[i].d * f(x[i].d);
        const dmin = y[i].d * f(x[i].dmin);

        const q8sums = neon.paddq_s16(
            neon.loadFrom(i16x8, &y[i].bsums),
            neon.loadFrom(i16x8, y[i].bsums[8..]),
        );

        const utmp = unpackQ5K(&x[i].scales);

        // The mins are the *second* half of the shuffled words here.
        const mins8 = neon.load(neon.u8x8, @as([*]const u8, @ptrCast(&utmp)) + 8);
        const mins: i16x8 = @bitCast(@as(u16x8, neon.movl_u8(mins8)));
        const prod = mullSum(q8sums, mins);
        const sumi_mins = neon.addvq_s32(prod);

        const scales: [16]u8 = @bitCast(utmp);

        var qhbits0 = neon.loadFrom(u8x16, &x[i].qh);
        var qhbits1 = neon.loadFrom(u8x16, x[i].qh[16..]);

        var sumi: i32 = 0;
        var q5: usize = 0;
        var q8: usize = 0;
        var sc: usize = 0;

        for (0..c.QK_K / 64) |_| {
            const q5bits0 = neon.loadFrom(u8x16, x[i].qs[q5..].ptr);
            const q5bits1 = neon.loadFrom(u8x16, x[i].qs[q5 + 16 ..].ptr);
            q5 += 32;

            var b: [4]i8x16 = undefined;
            inline for (0..4) |k| b[k] = neon.load(i8x16, q8At(&y[i], q8 + k * 16));
            q8 += 64;

            const h0 = neon.shlN(neon.@"and"(mone, qhbits0), 4);
            const h1 = neon.shlN(neon.@"and"(mone, qhbits1), 4);
            const h2 = neon.shlN(neon.@"and"(mtwo, qhbits0), 3);
            const h3 = neon.shlN(neon.@"and"(mtwo, qhbits1), 3);

            qhbits0 = neon.shrN(qhbits0, 2);
            qhbits1 = neon.shrN(qhbits1, 2);

            const w0: i8x16 = @bitCast(neon.orr(neon.@"and"(q5bits0, m4b), h0));
            const w1: i8x16 = @bitCast(neon.orr(neon.@"and"(q5bits1, m4b), h1));
            const w2: i8x16 = @bitCast(neon.orr(neon.shrN(q5bits0, 4), h2));
            const w3: i8x16 = @bitCast(neon.orr(neon.shrN(q5bits1, 4), h3));

            const zero: i32x4 = @splat(0);
            sumi += neon.addvq_s32(neon.dotq_s32(neon.dotq_s32(zero, w0, b[0]), w1, b[1])) * scales[sc];
            sc += 1;
            sumi += neon.addvq_s32(neon.dotq_s32(neon.dotq_s32(zero, w2, b[2]), w3, b[3])) * scales[sc];
            sc += 1;
        }

        sumf = @mulAdd(
            f32,
            -dmin,
            @as(f32, @floatFromInt(sumi_mins)),
            @mulAdd(f32, d, @as(f32, @floatFromInt(sumi)), sumf),
        );
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q6_K_q8_K` (arch/arm/quants.c:2964 @c1d0e7a00), the `__ARM_NEON`
/// arm.
///
/// Six-bit weights, four in `ql` and two in `qh`, with signed scales that need
/// no unpacking. The bias of 32 is applied once per super-block as
/// `isum - 32 * isum_mins` rather than per weight.
pub export fn ggml_vec_dot_q6_K_q8_K(
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

    const x = as(blocks.Q6_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb = @divTrunc(n, c.QK_K);

    const m4b: u8x16 = @splat(0xF);
    // Named `mone` in the C and holding 3, which is the two-bit mask.
    const mone: u8x16 = @splat(3);

    var sum: f32 = 0;

    for (0..@intCast(nb)) |i| {
        const d_all = f(x[i].d);

        const q8sums0 = neon.loadFrom(i16x8, &y[i].bsums);
        const q8sums1 = neon.loadFrom(i16x8, y[i].bsums[8..]);

        const scales_v = neon.loadFrom(i8x16, &x[i].scales);
        const q6scales0 = neon.movl_s8(neon.low(scales_v));
        const q6scales1 = neon.movl_s8(neon.high(scales_v));

        const prod = neon.add(
            mullSum(q8sums0, q6scales0),
            mullSum(q8sums1, q6scales1),
        );
        const isum_mins = neon.addvq_s32(prod);

        var isum: i32 = 0;
        var q6: usize = 0;
        var qh: usize = 0;
        var q8: usize = 0;
        var sc: usize = 0;

        for (0..c.QK_K / 128) |_| {
            const qhbits0 = neon.loadFrom(u8x16, x[i].qh[qh..].ptr);
            const qhbits1 = neon.loadFrom(u8x16, x[i].qh[qh + 16 ..].ptr);
            qh += 32;

            var q6bits: [4]u8x16 = undefined;
            inline for (0..4) |k| q6bits[k] = neon.loadFrom(u8x16, x[i].ql[q6 + k * 16 ..].ptr);
            q6 += 64;

            // First half: low nibbles, high bits at shifts 0 and 2.
            {
                var b: [4]i8x16 = undefined;
                inline for (0..4) |k| b[k] = neon.load(i8x16, q8At(&y[i], q8 + k * 16));
                q8 += 64;

                const h0 = neon.shlN(neon.@"and"(mone, qhbits0), 4);
                const h1 = neon.shlN(neon.@"and"(mone, qhbits1), 4);
                const h2 = neon.shlN(neon.@"and"(mone, neon.shrN(qhbits0, 2)), 4);
                const h3 = neon.shlN(neon.@"and"(mone, neon.shrN(qhbits1, 2)), 4);
                const hs = [4]u8x16{ h0, h1, h2, h3 };

                var acc: i32 = 0;
                inline for (0..4) |k| {
                    const w: i8x16 = @bitCast(neon.orr(neon.@"and"(q6bits[k], m4b), hs[k]));
                    acc += dotv(w, b[k]) * x[i].scales[sc + k];
                }
                isum += acc;
                sc += 4;
            }

            // Second half: high nibbles, high bits at shifts 4 and 6.
            {
                var b: [4]i8x16 = undefined;
                inline for (0..4) |k| b[k] = neon.load(i8x16, q8At(&y[i], q8 + k * 16));
                q8 += 64;

                const h0 = neon.shlN(neon.@"and"(mone, neon.shrN(qhbits0, 4)), 4);
                const h1 = neon.shlN(neon.@"and"(mone, neon.shrN(qhbits1, 4)), 4);
                const h2 = neon.shlN(neon.@"and"(mone, neon.shrN(qhbits0, 6)), 4);
                const h3 = neon.shlN(neon.@"and"(mone, neon.shrN(qhbits1, 6)), 4);
                const hs = [4]u8x16{ h0, h1, h2, h3 };

                var acc: i32 = 0;
                inline for (0..4) |k| {
                    const w: i8x16 = @bitCast(neon.orr(neon.shrN(q6bits[k], 4), hs[k]));
                    acc += dotv(w, b[k]) * x[i].scales[sc + k];
                }
                isum += acc;
                sc += 4;
            }
        }

        sum = @mulAdd(f32, d_all * y[i].d, @as(f32, @floatFromInt(isum - 32 * isum_mins)), sum);
    }

    s[0] = sum;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const testing = @import("../testing.zig");
const golden = @import("golden.zig");

test "q2_K dot matches the C" {
    try testing.check(ggml_vec_dot_q2_K_q8_K, c.GGML_TYPE_Q2_K, c.GGML_TYPE_Q8_K, golden.q2_K);
}

test "q3_K dot matches the C" {
    try testing.check(ggml_vec_dot_q3_K_q8_K, c.GGML_TYPE_Q3_K, c.GGML_TYPE_Q8_K, golden.q3_K);
}

test "q4_K dot matches the C" {
    try testing.check(ggml_vec_dot_q4_K_q8_K, c.GGML_TYPE_Q4_K, c.GGML_TYPE_Q8_K, golden.q4_K);
}

test "q5_K dot matches the C" {
    try testing.check(ggml_vec_dot_q5_K_q8_K, c.GGML_TYPE_Q5_K, c.GGML_TYPE_Q8_K, golden.q5_K);
}

test "q6_K dot matches the C" {
    try testing.check(ggml_vec_dot_q6_K_q8_K, c.GGML_TYPE_Q6_K, c.GGML_TYPE_Q8_K, golden.q6_K);
}

test "the widening product does not truncate" {
    // `mins * bsums` exceeds sixteen bits routinely -- a `q8_K` group sum can
    // reach 32*127 and a min 15 -- so `vmull_s16` has to widen before
    // multiplying. Doing this on `i16x8` would wrap.
    const a: i16x8 = @splat(15);
    const b: i16x8 = @splat(4064); // 32 * 127
    const got: [4]i32 = mullSum(a, b);

    // Each lane holds two widened products.
    try std.testing.expectEqual(@as(i32, 15 * 4064 * 2), got[0]);
}
