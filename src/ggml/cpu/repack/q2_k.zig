//! The `q2_K` × `q8_K` interleaved gemv and gemm.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Four quants per byte, four scales per step
//!
//! Two bits each, so one `qs` byte carries four values and the `k` loop
//! runs to `qk / (4 * blocklen)` rather than `qk / (2 * blocklen)`. Each
//! of the four takes its own scale from a different 16-byte slice of
//! `scales`, indexed by `offset = ((k / 2) % 2) + j * 2` — the same offset
//! for all four, into four different slices.
//!
//! # `scales` holds both the scales and the mins
//!
//! Low nibble is the scale, high nibble is the min: the kernels mask
//! `& 0xF` for one and shift `>> 4` for the other, over the same bytes.
//! There is no separate mins array as there is in `q4_K`.
//!
//! # This one uses `GGML_FP16_TO_FP32`, not `GGML_CPU_FP16_TO_FP32`
//!
//! Every other kernel in `repack.cpp` reaches for the `_CPU_` form. `q2_K`
//! does not, in all four places. Reproduced as written — on this target
//! both resolve to a conversion instruction, but they are different macros
//! and the port follows the C rather than normalising it.
//!
//! # These carry the live name
//!
//! `arch-fallback.h` renames the generic to the bare name here, so
//! `repack.o` exports `ggml_gemv_q2_K_8x8_q8_K` without a suffix.

const std = @import("std");
const impl = @import("../../impl.zig");
const blocks = @import("blocks.zig");
const epilogue = @import("epilogue.zig");

const c = impl.c;

const QK_K = 256;

/// Ports `ggml_gemv_q2_K_8x8_q8_K_generic`
/// (ggml-cpu/repack.cpp:1029 @c1d0e7a00).
fn gemv(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    _ = bs;
    _ = nr;
    const qk: i32 = QK_K;
    const nb: i32 = @divTrunc(n, qk);
    const ncols_interleaved: i32 = 8;
    const blocklen: i32 = 8;

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [8]f32 = undefined;
    var sum_minf: [8]f32 = undefined;

    const a_ptr: [*]const c.block_q8_K = @ptrCast(@alignCast(vy));
    var x: i32 = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const b_ptr: [*]const blocks.block_q2_Kx8 =
            @as([*]const blocks.block_q2_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));
        for (0..8) |j| {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        var l: i32 = 0;
        while (l < nb) : (l += 1) {
            const bl = &b_ptr[@intCast(l)];
            const al = &a_ptr[@intCast(l)];
            var k: i32 = 0;
            while (k < @divTrunc(qk, 4 * blocklen)) : (k += 1) {
                const sc = @as([*]const u8, &bl.scales) + @as(usize, @intCast(@divTrunc(k, 4) * 64));
                for (0..8) |ju| {
                    const j: i32 = @intCast(ju);
                    var sumi: i32 = 0;
                    const offset: usize = @intCast(@rem(@divTrunc(k, 2), 2) + j * 2);
                    var i: i32 = 0;
                    while (i < blocklen) : (i += 1) {
                        const q = bl.qs[@intCast(k * ncols_interleaved * blocklen + j * blocklen + i)];
                        const v0: i32 = @as(i8, @bitCast(q & 3));
                        const v1: i32 = @as(i8, @bitCast((q >> 2) & 3));
                        const v2: i32 = @as(i8, @bitCast((q >> 4) & 3));
                        const v3: i32 = @as(i8, @bitCast((q >> 6) & 3));
                        const base = (k >> 2) * 128 + @rem(k, 4) * blocklen + i;
                        var sumi1: i32 = v0 * al.qs[@intCast(base)];
                        var sumi2: i32 = v1 * al.qs[@intCast(base + 32)];
                        var sumi3: i32 = v2 * al.qs[@intCast(base + 64)];
                        var sumi4: i32 = v3 * al.qs[@intCast(base + 96)];
                        sumi1 = sumi1 * (sc[offset] & 0xF);
                        sumi2 = sumi2 * (sc[16 + offset] & 0xF);
                        sumi3 = sumi3 * (sc[32 + offset] & 0xF);
                        sumi4 = sumi4 * (sc[48 + offset] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[ju] = epilogue.accumulate(sumf[ju], sumi, impl.fp16ToFp32(bl.d[ju]), al.d);
                }
            }
            for (0..8) |sb| {
                const mins = @as([*]const u8, &bl.scales) + sb * 16;
                for (0..8) |j| {
                    const mins_prod: i32 = @as(i32, mins[j * 2] >> 4) * al.bsums[sb * 2] +
                        @as(i32, mins[j * 2 + 1] >> 4) * al.bsums[sb * 2 + 1];
                    sum_minf[j] = epilogue.accumulate(sum_minf[j], mins_prod, impl.fp16ToFp32(bl.dmin[j]), al.d);
                }
            }
        }
        for (0..8) |j| s[@intCast(x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[j] - sum_minf[j];
    }
}

/// Ports `ggml_gemm_q2_K_8x8_q8_K_generic`
/// (ggml-cpu/repack.cpp:1986 @c1d0e7a00).
fn gemm(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    const qk: i32 = QK_K;
    const nb: i32 = @divTrunc(n, qk);
    const ncols_interleaved: i32 = 8;
    const blocklen: i32 = 8;

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nr, 4) == 0, "nr % 4 == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [4][8]f32 = undefined;
    var sum_minf: [4][8]f32 = undefined;

    var y: i32 = 0;
    while (y < @divTrunc(nr, 4)) : (y += 1) {
        const a_ptr: [*]const blocks.block_q8_Kx4 =
            @as([*]const blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));
        var x: i32 = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const b_ptr: [*]const blocks.block_q2_Kx8 =
                @as([*]const blocks.block_q2_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));
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
                var k: i32 = 0;
                while (k < @divTrunc(qk, 4 * blocklen)) : (k += 1) {
                    const sc = @as([*]const u8, &bl.scales) + @as(usize, @intCast(@divTrunc(k, 4) * 64));
                    for (0..4) |mu| {
                        const m: i32 = @intCast(mu);
                        for (0..8) |ju| {
                            const j: i32 = @intCast(ju);
                            var sumi: i32 = 0;
                            const offset: usize = @intCast(@rem(@divTrunc(k, 2), 2) + j * 2);
                            var i: i32 = 0;
                            while (i < blocklen) : (i += 1) {
                                const q = bl.qs[@intCast(k * ncols_interleaved * blocklen + j * blocklen + i)];
                                const v0: i32 = @as(i8, @bitCast(q & 3));
                                const v1: i32 = @as(i8, @bitCast((q >> 2) & 3));
                                const v2: i32 = @as(i8, @bitCast((q >> 4) & 3));
                                const v3: i32 = @as(i8, @bitCast((q >> 6) & 3));
                                const base = (k >> 2) * 512 + @rem(k, 4) * 4 * blocklen + m * blocklen + i;
                                var sumi1: i32 = v0 * al.qs[@intCast(base)];
                                var sumi2: i32 = v1 * al.qs[@intCast(base + 128)];
                                var sumi3: i32 = v2 * al.qs[@intCast(base + 256)];
                                var sumi4: i32 = v3 * al.qs[@intCast(base + 384)];
                                sumi1 = sumi1 * (sc[offset] & 0xF);
                                sumi2 = sumi2 * (sc[16 + offset] & 0xF);
                                sumi3 = sumi3 * (sc[32 + offset] & 0xF);
                                sumi4 = sumi4 * (sc[48 + offset] & 0xF);
                                sumi += sumi1 + sumi2 + sumi3 + sumi4;
                            }
                            sumf[mu][ju] = epilogue.accumulate(sumf[mu][ju], sumi, impl.fp16ToFp32(bl.d[ju]), al.d[mu]);
                        }
                    }
                }
                for (0..8) |sb| {
                    const mins = @as([*]const u8, &bl.scales) + sb * 16;
                    for (0..4) |mu| {
                        const off: i32 = @as(i32, @intCast(sb)) * 8 + @as(i32, @intCast(mu)) * 4 -
                            (@as(i32, @intCast(sb % 2)) * 6);
                        const bsums = @as([*]const i16, &al.bsums) + @as(usize, @intCast(off));
                        for (0..8) |j| {
                            const mins_prod: i32 = @as(i32, mins[j * 2] >> 4) * bsums[0] +
                                @as(i32, mins[j * 2 + 1] >> 4) * bsums[1];
                            sum_minf[mu][j] = epilogue.accumulate(sum_minf[mu][j], mins_prod, impl.fp16ToFp32(bl.dmin[j]), al.d[mu]);
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

pub export fn ggml_gemv_q2_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q2_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(n, s, bs, vx, vy, nr, nc);
}
