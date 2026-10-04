//! The `q6_K` × `q8_K` interleaved gemv and gemm, generic forms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Two halves per step, not two nibbles
//!
//! Unlike `q4_K` and `q5_K`, which pair the low and high nibble of one
//! byte, `q6_K` pairs two positions **128 apart** in the super-block:
//! `base_l` and `base_h = base_l + 64`. Each gets its own scale, its own
//! `qh` shift and its own `qh` half. They are not two halves of one value.
//!
//! # Six bits, assembled then biased
//!
//! `q = ((hi_2 << 4) | lo_4) - 32`. The `- 32` is the format's zero point
//! and applies to the assembled six-bit value, not to either part.
//!
//! # No mins
//!
//! `q6_K` has no `dmin`, so there is no `sum_minf` term and the result is
//! `sumf` directly — the one structural difference from its K-quant
//! siblings.

const std = @import("std");
const impl = @import("../../impl.zig");
const blocks = @import("blocks.zig");
const epilogue = @import("epilogue.zig");
const convert = @import("../convert.zig");

const c = impl.c;

const QK_K = 256;

/// One `(ql, qh)` pair's six-bit value, for either half.
inline fn sixBit(
    comptime blocklen: i32,
    comptime ncols_interleaved: i32,
    bl: *const blocks.block_q6_Kx8,
    ql_pos: i32,
    comptime upper: bool,
    qh_half: i32,
    base: i32,
    i: i32,
    j: i32,
    qh_shift: u3,
) i32 {
    const lo: i32 = if (upper) (bl.ql[@intCast(ql_pos)] >> 4) & 0xF else bl.ql[@intCast(ql_pos)] & 0xF;
    const qh_idx = qh_half + @rem(base + i, 32);
    const qh_chunk = @divTrunc(qh_idx, blocklen);
    const qh_pos = @rem(qh_idx, blocklen);
    const qh_offset = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;
    const hi_2: i32 = (bl.qh[@intCast(qh_offset)] >> qh_shift) & 0x3;
    return ((hi_2 << 4) | lo) - 32;
}

/// Ports `ggml_gemv_q6_K_NxM_q8_K_generic_impl`
/// (ggml-cpu/repack.cpp:357 @c1d0e7a00).
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
    const blocks_per_half: i32 = @divTrunc(64, blocklen);

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [ncols_interleaved]f32 = undefined;

    const a_ptr: [*]const c.block_q8_K = @ptrCast(@alignCast(vy));
    var x: i32 = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const b_ptr: [*]const blocks.block_q6_Kx8 =
            @as([*]const blocks.block_q6_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..ncols_interleaved) |j| sumf[j] = 0.0;

        var l: i32 = 0;
        while (l < nb) : (l += 1) {
            const bl = &b_ptr[@intCast(l)];
            const al = &a_ptr[@intCast(l)];
            var k: i32 = 0;
            while (k < @divTrunc(qk, 2 * blocklen)) : (k += 1) {
                const base_l = @divTrunc(k, blocks_per_half) * 128 + @rem(k, blocks_per_half) * blocklen;
                const base_h = base_l + 64;

                const scale_idx_l = @divTrunc(base_l, 16);
                const scale_idx_h = @divTrunc(base_h, 16);

                const qh_shift_l: u3 = @intCast(@divTrunc(@rem(base_l, 128), 32) * 2);
                const qh_shift_h: u3 = @intCast(@divTrunc(@rem(base_h, 128), 32) * 2);

                const qh_half_l = @divTrunc(base_l, 128) * 32;
                const qh_half_h = @divTrunc(base_h, 128) * 32;

                for (0..ncols_interleaved) |ju| {
                    const j: i32 = @intCast(ju);
                    const scale_l: i32 = bl.scales[@intCast(scale_idx_l * ncols_interleaved + j)];
                    const scale_h: i32 = bl.scales[@intCast(scale_idx_h * ncols_interleaved + j)];

                    var sumi_l: i32 = 0;
                    var sumi_h: i32 = 0;

                    var i: i32 = 0;
                    while (i < blocklen) : (i += 1) {
                        const ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        const q_l = sixBit(blocklen, ncols_interleaved, bl, ql_pos, false, qh_half_l, base_l, i, j, qh_shift_l);
                        const q_h = sixBit(blocklen, ncols_interleaved, bl, ql_pos, true, qh_half_h, base_h, i, j, qh_shift_h);

                        const a_l: i32 = al.qs[@intCast(base_l + i)];
                        const a_h: i32 = al.qs[@intCast(base_h + i)];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[ju] = epilogue.accumulate(sumf[ju], sumi_l * scale_l + sumi_h * scale_h, convert.cpuFp16ToFp32(bl.d[ju]), al.d);
                }
            }
        }

        for (0..ncols_interleaved) |j| s[@intCast(x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[j];
    }
}

/// Ports `ggml_gemm_q6_K_NxM_q8_K_generic_impl`
/// (ggml-cpu/repack.cpp:447 @c1d0e7a00).
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
    const blocks_per_half: i32 = @divTrunc(64, blocklen);
    const q8_half_stride: i32 = 512;
    const q8_low_high_step: i32 = 256;

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nr, 4) == 0, "nr % 4 == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [4][ncols_interleaved]f32 = undefined;

    var y: i32 = 0;
    while (y < @divTrunc(nr, 4)) : (y += 1) {
        const a_ptr: [*]const blocks.block_q8_Kx4 =
            @as([*]const blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));
        var x: i32 = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const b_ptr: [*]const blocks.block_q6_Kx8 =
                @as([*]const blocks.block_q6_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));
            for (0..4) |m| {
                for (0..ncols_interleaved) |j| sumf[m][j] = 0.0;
            }
            var l: i32 = 0;
            while (l < nb) : (l += 1) {
                const bl = &b_ptr[@intCast(l)];
                const al = &a_ptr[@intCast(l)];
                var k: i32 = 0;
                while (k < @divTrunc(qk, 2 * blocklen)) : (k += 1) {
                    const base_l = @divTrunc(k, blocks_per_half) * 128 + @rem(k, blocks_per_half) * blocklen;
                    const base_h = base_l + 64;

                    const scale_idx_l = @divTrunc(base_l, 16);
                    const scale_idx_h = @divTrunc(base_h, 16);

                    const qh_shift_l: u3 = @intCast(@divTrunc(@rem(base_l, 128), 32) * 2);
                    const qh_shift_h: u3 = @intCast(@divTrunc(@rem(base_h, 128), 32) * 2);

                    const qh_half_l = @divTrunc(base_l, 128) * 32;
                    const qh_half_h = @divTrunc(base_h, 128) * 32;

                    const q8_base = @divTrunc(k, blocks_per_half) * q8_half_stride +
                        @rem(k, blocks_per_half) * (blocklen * 4);

                    for (0..4) |mu| {
                        const m: i32 = @intCast(mu);
                        for (0..ncols_interleaved) |ju| {
                            const j: i32 = @intCast(ju);
                            const scale_l: i32 = bl.scales[@intCast(scale_idx_l * ncols_interleaved + j)];
                            const scale_h: i32 = bl.scales[@intCast(scale_idx_h * ncols_interleaved + j)];

                            var sumi_l: i32 = 0;
                            var sumi_h: i32 = 0;

                            var i: i32 = 0;
                            while (i < blocklen) : (i += 1) {
                                const ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                const q_l = sixBit(blocklen, ncols_interleaved, bl, ql_pos, false, qh_half_l, base_l, i, j, qh_shift_l);
                                const q_h = sixBit(blocklen, ncols_interleaved, bl, ql_pos, true, qh_half_h, base_h, i, j, qh_shift_h);

                                const q8_l: i32 = al.qs[@intCast(q8_base + m * blocklen + i)];
                                const q8_h: i32 = al.qs[@intCast(q8_base + m * blocklen + i + q8_low_high_step)];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[mu][ju] = epilogue.accumulate(sumf[mu][ju], sumi_l * scale_l + sumi_h * scale_h, convert.cpuFp16ToFp32(bl.d[ju]), al.d[mu]);
                        }
                    }
                }
            }
            for (0..4) |m| {
                for (0..ncols_interleaved) |j| {
                    s[@intCast((y * 4 + @as(i32, @intCast(m))) * @as(i32, @intCast(bs)) + x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[m][j];
                }
            }
        }
    }
}

pub export fn ggml_gemv_q6_K_8x4_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(4, 8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemv_q6_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(8, 8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q6_K_8x4_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(4, 8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q6_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(8, 8, n, s, bs, vx, vy, nr, nc);
}
