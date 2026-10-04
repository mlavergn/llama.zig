//! The `q4_K` × `q8_K` interleaved gemv and gemm, generic forms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # The super-block scales are unpacked first
//!
//! `q4_K` packs eight 6-bit scales and eight 6-bit mins into twelve bytes
//! per sub-block. The `utmp` shuffle at the top of each `l` iteration is
//! the standard `q4_K` unpack, done for all eight sub-blocks at once
//! because the columns are interleaved. `uaux_0` exists because the
//! shuffle overwrites `utmp[1]` before reading it — reordering those four
//! lines silently corrupts every scale.
//!
//! # `8x4` and `8x8` differ by more than `blocklen`
//!
//! The quant and scale indices divide by `32 / blocklen` — 8 at
//! `blocklen = 4`, 4 at `blocklen = 8` — because that is how many `k`
//! steps cover one 32-value sub-block. The C writes the two constants out
//! rather than deriving them; here it is one `comptime` expression, which
//! is the same value and harder to mistype.
//!
//! # The mins are subtracted at the end
//!
//! `s = sumf - sum_minf`. The gemv weights each min by one pair of bsums;
//! the gemm indexes bsums as `(sb * 8) + (m * 4) - ((sb % 2) * 6)`, which
//! is not a typo — it walks the interleaved layout.

const std = @import("std");
const impl = @import("../../impl.zig");
const blocks = @import("blocks.zig");
const epilogue = @import("epilogue.zig");
const convert = @import("../convert.zig");

const c = impl.c;

const QK_K = 256;

const kmask1: u32 = 0x3f3f3f3f;
const kmask2: u32 = 0x0f0f0f0f;
const kmask3: u32 = 0x03030303;

/// The `utmp` shuffle, shared by every kernel here.
inline fn unpackScales(utmp: *[32]u32, bl: *const blocks.block_q4_Kx8) void {
    for (0..8) |sb| {
        const src = bl.scales[sb * 12 ..][0..12];
        @memcpy(@as([*]u8, @ptrCast(utmp))[sb * 16 ..][0..12], src);
        utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
        const uaux_0 = utmp[sb * 4 + 1] & kmask1;
        utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
        utmp[sb * 4 + 2] = uaux_0;
        utmp[sb * 4 + 0] &= kmask1;
    }
}

/// Ports `ggml_gemv_q4_K_8x4_q8_K_generic` and
/// `ggml_gemv_q4_K_8x8_q8_K_generic`
/// (ggml-cpu/repack.cpp:887, 958 @c1d0e7a00).
fn gemv(
    comptime blocklen: i32,
    n: c_int,
    s: [*]f32,
    bs: usize,
    vx: *const anyopaque,
    vy: *const anyopaque,
    nr: c_int,
    nc: c_int,
) void {
    _ = bs;
    _ = nr;
    const qk: i32 = QK_K;
    const nb: i32 = @divTrunc(n, qk);
    const ncols_interleaved: i32 = 8;
    // How many `k` steps cover one 32-value sub-block.
    const kdiv: i32 = @divTrunc(32, blocklen);

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [8]f32 = undefined;
    var sum_minf: [8]f32 = undefined;
    var utmp: [32]u32 = undefined;

    const a_ptr: [*]const c.block_q8_K = @ptrCast(@alignCast(vy));
    var x: i32 = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const b_ptr: [*]const blocks.block_q4_Kx8 =
            @as([*]const blocks.block_q4_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..8) |j| {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        var l: i32 = 0;
        while (l < nb) : (l += 1) {
            const bl = &b_ptr[@intCast(l)];
            const al = &a_ptr[@intCast(l)];
            unpackScales(&utmp, bl);

            const ub: [*]const u8 = @ptrCast(&utmp);
            var k: i32 = 0;
            while (k < @divTrunc(qk, 2 * blocklen)) : (k += 1) {
                const scales_0 = ub + @as(usize, @intCast(@divTrunc(k, kdiv) * 32));
                const scales_1 = ub + @as(usize, @intCast(@divTrunc(k, kdiv) * 32 + 16));
                for (0..8) |j| {
                    var sumi: i32 = 0;
                    var i: i32 = 0;
                    while (i < blocklen) : (i += 1) {
                        const q = bl.qs[@intCast(k * ncols_interleaved * blocklen + @as(i32, @intCast(j)) * blocklen + i)];
                        const v0: i32 = @as(i8, @bitCast(q & 0xF));
                        const v1: i32 = @as(i8, @bitCast(q >> 4));
                        const base = @divTrunc(k, kdiv) * 64 + @rem(k, kdiv) * blocklen + i;
                        var sumi1: i32 = v0 * al.qs[@intCast(base)];
                        var sumi2: i32 = v1 * al.qs[@intCast(base + 32)];
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] = epilogue.accumulate(sumf[j], sumi, convert.cpuFp16ToFp32(bl.d[j]), al.d);
                }
            }
            for (0..8) |sb| {
                const mins = ub + 8 + sb * 16;
                for (0..8) |j| {
                    // `mins[j] * (bsums[0] + bsums[1])` is an *integer*
                    // product in the C, both operands promoted to `int`.
                    // Zig would add the two `i16` at `i16` width, which
                    // wraps where the C widens.
                    const mins_prod: i32 = @as(i32, mins[j]) *
                        (@as(i32, al.bsums[sb * 2]) + @as(i32, al.bsums[sb * 2 + 1]));
                    sum_minf[j] = epilogue.accumulate(sum_minf[j], mins_prod, convert.cpuFp16ToFp32(bl.dmin[j]), al.d);
                }
            }
        }
        for (0..8) |j| s[@intCast(x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[j] - sum_minf[j];
    }
}

/// Ports `ggml_gemm_q4_K_8x4_q8_K_generic` and
/// `ggml_gemm_q4_K_8x8_q8_K_generic`
/// (ggml-cpu/repack.cpp:1822, 1905 @c1d0e7a00).
fn gemm(
    comptime blocklen: i32,
    n: c_int,
    s: [*]f32,
    bs: usize,
    vx: *const anyopaque,
    vy: *const anyopaque,
    nr: c_int,
    nc: c_int,
) void {
    const qk: i32 = QK_K;
    const nb: i32 = @divTrunc(n, qk);
    const ncols_interleaved: i32 = 8;
    const kdiv: i32 = @divTrunc(32, blocklen);

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nr, 4) == 0, "nr % 4 == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [4][8]f32 = undefined;
    var sum_minf: [4][8]f32 = undefined;
    var utmp: [32]u32 = undefined;

    var y: i32 = 0;
    while (y < @divTrunc(nr, 4)) : (y += 1) {
        const a_ptr: [*]const blocks.block_q8_Kx4 =
            @as([*]const blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));
        var x: i32 = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const b_ptr: [*]const blocks.block_q4_Kx8 =
                @as([*]const blocks.block_q4_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));
            for (0..4) |m| {
                for (0..8) |j| {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            var l: i32 = 0;
            while (l < nb) : (l += 1) {
                const bl = &b_ptr[@intCast(l)];
                const al = &a_ptr[@intCast(l)];
                unpackScales(&utmp, bl);

                const ub: [*]const u8 = @ptrCast(&utmp);
                var k: i32 = 0;
                while (k < @divTrunc(qk, 2 * blocklen)) : (k += 1) {
                    const scales_0 = ub + @as(usize, @intCast(@divTrunc(k, kdiv) * 32));
                    const scales_1 = ub + @as(usize, @intCast(@divTrunc(k, kdiv) * 32 + 16));
                    for (0..4) |m| {
                        for (0..8) |j| {
                            var sumi: i32 = 0;
                            var i: i32 = 0;
                            while (i < blocklen) : (i += 1) {
                                const q = bl.qs[@intCast(k * ncols_interleaved * blocklen + @as(i32, @intCast(j)) * blocklen + i)];
                                const v0: i32 = @as(i8, @bitCast(q & 0xF));
                                const v1: i32 = @as(i8, @bitCast(q >> 4));
                                const base = @divTrunc(k, kdiv) * 256 + @rem(k, kdiv) * 4 * blocklen +
                                    @as(i32, @intCast(m)) * blocklen + i;
                                var sumi1: i32 = v0 * al.qs[@intCast(base)];
                                var sumi2: i32 = v1 * al.qs[@intCast(base + 128)];
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] = epilogue.accumulate(sumf[m][j], sumi, convert.cpuFp16ToFp32(bl.d[j]), al.d[m]);
                        }
                    }
                }
                for (0..8) |sb| {
                    const mins = ub + 8 + sb * 16;
                    for (0..4) |m| {
                        // Walks the interleaved bsum layout; the `- (sb % 2) * 6`
                        // is the C's and is not a typo.
                        const off: i32 = @as(i32, @intCast(sb)) * 8 + @as(i32, @intCast(m)) * 4 -
                            (@as(i32, @intCast(sb % 2)) * 6);
                        const bsums = @as([*]const i16, &al.bsums) + @as(usize, @intCast(off));
                        for (0..8) |j| {
                            // An integer product in the C; see the gemv.
                            const mins_prod: i32 = @as(i32, mins[j]) *
                                (@as(i32, bsums[0]) + @as(i32, bsums[1]));
                            sum_minf[m][j] = epilogue.accumulate(sum_minf[m][j], mins_prod, convert.cpuFp16ToFp32(bl.dmin[j]), al.d[m]);
                        }
                    }
                }
            }
            for (0..4) |m| {
                for (0..8) |j| {
                    s[@intCast((y * 4 + @as(i32, @intCast(m))) * @as(i32, @intCast(bs)) + x * ncols_interleaved + @as(i32, @intCast(j)))] =
                        sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

pub export fn ggml_gemv_q4_K_8x4_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemv_q4_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q4_K_8x4_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q4_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(8, n, s, bs, vx, vy, nr, nc);
}
