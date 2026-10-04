//! The `q4_0` × `q8_0` interleaved gemv and gemm, generic forms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Six functions, two bodies
//!
//! `4x4`, `4x8` and `8x8` differ only in `ncols_interleaved`, `blocklen`
//! and which interleaved block type they read — `block_q4_0x4` for the two
//! four-column shapes, `block_q4_0x8` for the eight. All three are one
//! function over `comptime` parameters.
//!
//! # These are dead on this target
//!
//! `arch/arm/repack.cpp` supplies all six without the `_generic` suffix and
//! `arch-fallback.h` renames nothing on ARM, so nothing calls these — with
//! two exceptions measured from the object files: `ggml_gemv_q4_0_8x8_q8_0`
//! and `ggml_gemm_q4_0_8x8_q8_0` fall back to the generic when their
//! runtime feature check fails.
//!
//! The rest have the same status as the `_generic` dot products in
//! `cpu/quants/`: exported, unreachable, and therefore gated only by
//! goldens. See `CLAUDE.md`.
//!
//! # The `>> 4` is on the sum, not the product
//!
//! `v0` and `v1` are the nibbles shifted into the high half of a byte, so
//! each product carries four extra bits; the C shifts **the pair's sum**
//! right by four, once, rather than each product. Shifting earlier loses
//! the low bits of the odd term.

const std = @import("std");
const impl = @import("../../impl.zig");
const blocks = @import("blocks.zig");
const epilogue = @import("epilogue.zig");
const convert = @import("../convert.zig");

const c = impl.c;

const QK8_0 = 32;

/// Ports `ggml_gemv_q4_0_4x4_q8_0_generic`,
/// `ggml_gemv_q4_0_4x8_q8_0_generic` and `ggml_gemv_q4_0_8x8_q8_0_generic`
/// (ggml-cpu/repack.cpp:754, 799, 843 @c1d0e7a00).
fn gemv(
    comptime ncols_interleaved: i32,
    comptime blocklen: i32,
    comptime BlockX: type,
    n: c_int,
    s: [*]f32,
    bs: usize,
    vx: *const anyopaque,
    vy: *const anyopaque,
    nr: c_int,
    nc: c_int,
) void {
    _ = bs;
    const qk: i32 = QK8_0;
    const nb: i32 = @divTrunc(n, qk);

    impl.assert(nr == 1, "nr == 1");
    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [ncols_interleaved]f32 = undefined;

    const a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
    var x: i32 = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const b_ptr: [*]const BlockX = @as([*]const BlockX, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..ncols_interleaved) |j| sumf[j] = 0.0;
        var l: i32 = 0;
        while (l < nb) : (l += 1) {
            const bl = &b_ptr[@intCast(l)];
            const al = &a_ptr[@intCast(l)];
            var k: i32 = 0;
            while (k < @divTrunc(qk, 2 * blocklen)) : (k += 1) {
                for (0..ncols_interleaved) |j| {
                    var sumi: i32 = 0;
                    var i: i32 = 0;
                    while (i < blocklen) : (i += 1) {
                        const q = bl.qs[@intCast(k * ncols_interleaved * blocklen + @as(i32, @intCast(j)) * blocklen + i)];
                        const v0: i32 = @as(i8, @bitCast(@as(u8, @bitCast(q)) << 4));
                        const v1: i32 = @as(i8, @bitCast(@as(u8, @bitCast(q)) & 0xF0));
                        sumi += ((v0 * al.qs[@intCast(k * blocklen + i)]) +
                            (v1 * al.qs[@intCast(k * blocklen + i + @divTrunc(qk, 2))])) >> 4;
                    }
                    sumf[j] = epilogue.accumulate(sumf[j], sumi, convert.cpuFp16ToFp32(bl.d[j]), convert.cpuFp16ToFp32(al.d));
                }
            }
        }
        for (0..ncols_interleaved) |j| s[@intCast(x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[j];
    }
}

/// Ports `ggml_gemm_q4_0_4x4_q8_0_generic`,
/// `ggml_gemm_q4_0_4x8_q8_0_generic` and `ggml_gemm_q4_0_8x8_q8_0_generic`
/// (ggml-cpu/repack.cpp:1658, 1714, 1768 @c1d0e7a00).
fn gemm(
    comptime ncols_interleaved: i32,
    comptime blocklen: i32,
    comptime BlockX: type,
    n: c_int,
    s: [*]f32,
    bs: usize,
    vx: *const anyopaque,
    vy: *const anyopaque,
    nr: c_int,
    nc: c_int,
) void {
    const qk: i32 = QK8_0;
    const nb: i32 = @divTrunc(n, qk);

    impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    impl.assert(@rem(nr, 4) == 0, "nr % 4 == 0");
    impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [4][ncols_interleaved]f32 = undefined;

    var y: i32 = 0;
    while (y < @divTrunc(nr, 4)) : (y += 1) {
        const a_ptr: [*]const blocks.block_q8_0x4 =
            @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));
        var x: i32 = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const b_ptr: [*]const BlockX =
                @as([*]const BlockX, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));
            for (0..4) |m| {
                for (0..ncols_interleaved) |j| sumf[m][j] = 0.0;
            }
            var l: i32 = 0;
            while (l < nb) : (l += 1) {
                const bl = &b_ptr[@intCast(l)];
                const al = &a_ptr[@intCast(l)];
                var k: i32 = 0;
                while (k < @divTrunc(qk, 2 * blocklen)) : (k += 1) {
                    for (0..4) |m| {
                        for (0..ncols_interleaved) |j| {
                            var sumi: i32 = 0;
                            var i: i32 = 0;
                            while (i < blocklen) : (i += 1) {
                                const q = bl.qs[@intCast(k * ncols_interleaved * blocklen + @as(i32, @intCast(j)) * blocklen + i)];
                                const v0: i32 = @as(i8, @bitCast(@as(u8, @bitCast(q)) << 4));
                                const v1: i32 = @as(i8, @bitCast(@as(u8, @bitCast(q)) & 0xF0));
                                sumi += ((v0 * al.qs[@intCast(k * 4 * blocklen + @as(i32, @intCast(m)) * blocklen + i)]) +
                                    (v1 * al.qs[@intCast(k * 4 * blocklen + @as(i32, @intCast(m)) * blocklen + i + @divTrunc(qk, 2) * 4)])) >> 4;
                            }
                            sumf[m][j] = epilogue.accumulate(sumf[m][j], sumi, convert.cpuFp16ToFp32(bl.d[j]), convert.cpuFp16ToFp32(al.d[m]));
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

const Gemv = fn (c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) callconv(.c) void;

pub export fn ggml_gemv_q4_0_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(4, 4, blocks.block_q4_0x4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemv_q4_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(4, 8, blocks.block_q4_0x4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemv_q4_0_8x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(8, 8, blocks.block_q4_0x8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q4_0_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(4, 4, blocks.block_q4_0x4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q4_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(4, 8, blocks.block_q4_0x4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_q4_0_8x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(8, 8, blocks.block_q4_0x8, n, s, bs, vx, vy, nr, nc);
}

comptime {
    _ = Gemv;
}
