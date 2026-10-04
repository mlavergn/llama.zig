//! The `q5_K` × `q8_K` interleaved gemv and gemm, generic forms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # `q4_K` plus a fifth bit
//!
//! The scale unpack, the mins subtraction and the `q8_K` indexing are
//! `q4_K`'s — see `q4_k.zig` for the `uaux_0` ordering hazard, which
//! applies here too. What is new is the high bit: `qh` holds one bit per
//! quant, and the two nibbles of a byte take bits `qh_shift` and
//! `qh_shift + 1` of their `qh` byte, where `qh_shift = (k / (32 /
//! blocklen)) * 2`.
//!
//! **The `qh` index is not the `qs` index.** `qs` runs
//! `k * ncols * blocklen + j * blocklen + i`; `qh` re-derives a position
//! within the 32-value sub-block and indexes by *chunk*, because one `qh`
//! byte serves eight quants across the interleave. Reusing the `qs` offset
//! reads the wrong high bits and is the obvious way to get this wrong.

const std = @import("std");
const impl = @import("../../impl.zig");
const blocks = @import("blocks.zig");
const epilogue = @import("epilogue.zig");
const convert = @import("../convert.zig");
const q4_k = @import("q4_k.zig");

const c = impl.c;

const QK_K = 256;
const K_SCALE_SIZE = 12;

const kmask1: u32 = 0x3f3f3f3f;
const kmask2: u32 = 0x0f0f0f0f;
const kmask3: u32 = 0x03030303;

/// The `utmp` shuffle. Identical to `q4_k.zig`'s but over
/// `block_q5_Kx8`, whose `scales` field is the same twelve bytes per
/// sub-block.
inline fn unpackScales(utmp: *[32]u32, bl: *const blocks.block_q5_Kx8) void {
    for (0..8) |sb| {
        @memcpy(@as([*]u8, @ptrCast(utmp))[sb * 16 ..][0..K_SCALE_SIZE], bl.scales[sb * K_SCALE_SIZE ..][0..K_SCALE_SIZE]);
        utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
        const uaux_0 = utmp[sb * 4 + 1] & kmask1;
        utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
        utmp[sb * 4 + 2] = uaux_0;
        utmp[sb * 4 + 0] &= kmask1;
    }
}

/// Both nibbles of one `qs` byte, with their `qh` bits folded in.
inline fn fiveBit(bl: *const blocks.block_q5_Kx8, b_qs_offset: i32, b_qh_offset: i32, qh_shift: u3) struct { i32, i32 } {
    const qh_val = bl.qh[@intCast(b_qh_offset)];
    const h0: u8 = (qh_val >> qh_shift) & 1;
    const h1: u8 = (qh_val >> (qh_shift + 1)) & 1;
    const q = bl.qs[@intCast(b_qs_offset)];
    const v0: i32 = @as(i8, @bitCast((q & 0xF) | (h0 << 4)));
    const v1: i32 = @as(i8, @bitCast((q >> 4) | (h1 << 4)));
    return .{ v0, v1 };
}

/// Ports `ggml_gemv_q5_K_NxM_q8_K_generic_impl`
/// (ggml-cpu/repack.cpp:551 @c1d0e7a00).
fn gemv(
    comptime blocklen: i32,
    comptime ncols_interleaved: i32,
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
    const kdiv: i32 = @divTrunc(32, blocklen);

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [ncols_interleaved]f32 = undefined;
    var sum_minf: [ncols_interleaved]f32 = undefined;
    var utmp: [32]u32 = undefined;

    const a_ptr: [*]const c.block_q8_K = @ptrCast(@alignCast(vy));
    var x: i32 = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const b_ptr: [*]const blocks.block_q5_Kx8 =
            @as([*]const blocks.block_q5_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..ncols_interleaved) |j| {
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
                const scale_stride: i32 = 32;
                const scales_0 = ub + @as(usize, @intCast(@divTrunc(k, kdiv) * scale_stride));
                const scales_1 = ub + @as(usize, @intCast(@divTrunc(k, kdiv) * scale_stride + 16));
                const qh_shift: u3 = @intCast(@divTrunc(k, kdiv) * 2);
                for (0..ncols_interleaved) |j| {
                    var sumi: i32 = 0;
                    var i: i32 = 0;
                    while (i < blocklen) : (i += 1) {
                        const jj: i32 = @intCast(j);
                        const b_qs_offset = k * ncols_interleaved * blocklen + jj * blocklen + i;

                        const qh_idx = @rem(k * blocklen + i, 32);
                        const qh_chunk = @divTrunc(qh_idx, blocklen);
                        const qh_pos = @rem(qh_idx, blocklen);
                        const b_qh_offset = qh_chunk * (blocklen * ncols_interleaved) + jj * blocklen + qh_pos;

                        const v0, const v1 = fiveBit(bl, b_qs_offset, b_qh_offset, qh_shift);
                        const q8_offset = @divTrunc(k, kdiv) * 64 + @rem(k, kdiv) * blocklen + i;

                        var sumi1: i32 = v0 * al.qs[@intCast(q8_offset)];
                        var sumi2: i32 = v1 * al.qs[@intCast(q8_offset + 32)];
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] = epilogue.accumulate(sumf[j], sumi, convert.cpuFp16ToFp32(bl.d[j]), al.d);
                }
            }
            for (0..8) |sb| {
                const mins = ub + 8 + sb * 16;
                for (0..ncols_interleaved) |j| {
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
        for (0..ncols_interleaved) |j| s[@intCast(x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[j] - sum_minf[j];
    }
}

/// Ports `ggml_gemm_q5_K_NxM_q8_K_generic_impl`
/// (ggml-cpu/repack.cpp:647 @c1d0e7a00).
fn gemm(
    comptime blocklen: i32,
    comptime ncols_interleaved: i32,
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
    const kdiv: i32 = @divTrunc(32, blocklen);

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nr, 4) == 0, "nr % 4 == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [4][ncols_interleaved]f32 = undefined;
    var sum_minf: [4][ncols_interleaved]f32 = undefined;
    var utmp: [32]u32 = undefined;

    var y: i32 = 0;
    while (y < @divTrunc(nr, 4)) : (y += 1) {
        const a_ptr: [*]const blocks.block_q8_Kx4 =
            @as([*]const blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));
        var x: i32 = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const b_ptr: [*]const blocks.block_q5_Kx8 =
                @as([*]const blocks.block_q5_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));
            for (0..4) |m| {
                for (0..ncols_interleaved) |j| {
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
                    const qh_shift: u3 = @intCast(@divTrunc(k, kdiv) * 2);
                    for (0..4) |m| {
                        for (0..ncols_interleaved) |j| {
                            var sumi: i32 = 0;
                            var i: i32 = 0;
                            while (i < blocklen) : (i += 1) {
                                const jj: i32 = @intCast(j);
                                const b_qs_offset = k * ncols_interleaved * blocklen + jj * blocklen + i;

                                const qh_idx = @rem(k * blocklen + i, 32);
                                const qh_chunk = @divTrunc(qh_idx, blocklen);
                                const qh_pos = @rem(qh_idx, blocklen);
                                const b_qh_offset = qh_chunk * (blocklen * ncols_interleaved) + jj * blocklen + qh_pos;

                                const v0, const v1 = fiveBit(bl, b_qs_offset, b_qh_offset, qh_shift);
                                const q8_offset = @divTrunc(k, kdiv) * 256 +
                                    @rem(k, kdiv) * 4 * blocklen + @as(i32, @intCast(m)) * blocklen + i;

                                var sumi1: i32 = v0 * al.qs[@intCast(q8_offset)];
                                var sumi2: i32 = v1 * al.qs[@intCast(q8_offset + 128)];
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
                        const off: i32 = @as(i32, @intCast(sb)) * 8 + @as(i32, @intCast(m)) * 4 -
                            (@as(i32, @intCast(sb % 2)) * 6);
                        const bsums = @as([*]const i16, &al.bsums) + @as(usize, @intCast(off));
                        for (0..ncols_interleaved) |j| {
                            // An integer product in the C; see the gemv.
                            const mins_prod: i32 = @as(i32, mins[j]) *
                                (@as(i32, bsums[0]) + @as(i32, bsums[1]));
                            sum_minf[m][j] = epilogue.accumulate(sum_minf[m][j], mins_prod, convert.cpuFp16ToFp32(bl.dmin[j]), al.d[m]);
                        }
                    }
                }
            }
            for (0..4) |m| {
                for (0..ncols_interleaved) |j| {
                    s[@intCast((y * 4 + @as(i32, @intCast(m))) * @as(i32, @intCast(bs)) + x * ncols_interleaved + @as(i32, @intCast(j)))] =
                        sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

pub export fn ggml_gemv_q5_K_8x4_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(4, 8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemv_q5_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(8, 8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q5_K_8x4_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(4, 8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q5_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(8, 8, n, s, bs, vx, vy, nr, nc);
}

comptime {
    _ = q4_k;
}
