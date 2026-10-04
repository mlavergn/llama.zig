//! The `iq4_nl` and `mxfp4` interleaved gemv and gemm, generic forms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Why these two share a file
//!
//! Both are four-bit codebook formats: the nibble indexes a 16-entry table
//! of signed values rather than being the value. `iq4_nl` scales by an
//! `f16` delta per column, `mxfp4` by a shared 8-bit exponent. Otherwise
//! the loops are identical, so they are one body with the table and the
//! scale as `comptime` parameters.
//!
//! **Unlike `q4_0` there is no `>> 4` on the sum** — the codebook values
//! are already the real magnitudes, so nothing is shifted into a byte's
//! high half to be scaled back.
//!
//! # `mxfp4`'s exponent is computed, not looked up
//!
//! `GGML_CPU_E8M0_TO_FP32_HALF` has a table arm and a compute arm, and the
//! table is **x86-only** (simd-mappings.h:133 @c1d0e7a00), the `#else` arm. On ARM the compute arm is
//! what expands, so this calls `impl.e8m0ToFp32Half` — the same choice
//! `cpu/quants/legacy.zig` made.
//!
//! # Four of these eight carry the live name
//!
//! `arch-fallback.h` renames `_generic` to the bare name for the shapes ARM
//! does not implement, which here is both `8x8` pairs. Measured from the
//! object file rather than read off the header: `repack.o` exports
//! `ggml_gemv_iq4_nl_8x8_q8_0` and its three siblings without a suffix, and
//! the `4x4` pairs with one.

const std = @import("std");
const impl = @import("../../impl.zig");
const blocks = @import("blocks.zig");
const epilogue = @import("epilogue.zig");
const convert = @import("../convert.zig");

const c = impl.c;

const QK8_0 = 32;

/// Which codebook and scale a format uses.
const Fmt = enum { iq4_nl, mxfp4 };

inline fn codebook(comptime fmt: Fmt, nibble: u8) i32 {
    return switch (fmt) {
        .iq4_nl => c.kvalues_iq4nl[nibble],
        .mxfp4 => c.kvalues_mxfp4[nibble],
    };
}

inline fn colScale(comptime fmt: Fmt, blk: anytype, j: usize) f32 {
    return switch (fmt) {
        .iq4_nl => convert.cpuFp16ToFp32(blk.d[j]),
        .mxfp4 => impl.e8m0ToFp32Half(blk.e[j]),
    };
}

/// Ports `ggml_gemv_iq4_nl_4x4_q8_0_generic`,
/// `ggml_gemv_iq4_nl_8x8_q8_0_generic`,
/// `ggml_gemv_mxfp4_4x4_q8_0_generic` and
/// `ggml_gemv_mxfp4_8x8_q8_0_generic`
/// (ggml-cpu/repack.cpp:1122, 1160, 1198, 1236 @c1d0e7a00).
fn gemv(
    comptime fmt: Fmt,
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
        const b_ptr: [*]const BlockX =
            @as([*]const BlockX, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

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
                        const v0 = codebook(fmt, q & 0x0F);
                        const v1 = codebook(fmt, q >> 4);
                        sumi += (v0 * al.qs[@intCast(k * blocklen + i)]) +
                            (v1 * al.qs[@intCast(k * blocklen + i + @divTrunc(qk, 2))]);
                    }
                    sumf[j] = epilogue.accumulate(sumf[j], sumi, colScale(fmt, bl, j), convert.cpuFp16ToFp32(al.d));
                }
            }
        }
        for (0..ncols_interleaved) |j| s[@intCast(x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[j];
    }
}

/// Ports `ggml_gemm_iq4_nl_4x4_q8_0_generic`,
/// `ggml_gemm_iq4_nl_8x8_q8_0_generic`,
/// `ggml_gemm_mxfp4_4x4_q8_0_generic` and
/// `ggml_gemm_mxfp4_8x8_q8_0_generic`
/// (ggml-cpu/repack.cpp:2092, 2148, 2192, 2236 @c1d0e7a00).
fn gemm(
    comptime fmt: Fmt,
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
                                const v0 = codebook(fmt, q & 0x0F);
                                const v1 = codebook(fmt, q >> 4);
                                sumi += (v0 * al.qs[@intCast(k * 4 * blocklen + @as(i32, @intCast(m)) * blocklen + i)]) +
                                    (v1 * al.qs[@intCast(k * 4 * blocklen + @as(i32, @intCast(m)) * blocklen + i + @divTrunc(qk, 2) * 4)]);
                            }
                            sumf[m][j] = epilogue.accumulate(sumf[m][j], sumi, colScale(fmt, bl, j), convert.cpuFp16ToFp32(al.d[m]));
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

pub export fn ggml_gemv_iq4_nl_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(.iq4_nl, 4, 4, blocks.block_iq4_nlx4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_iq4_nl_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(.iq4_nl, 4, 4, blocks.block_iq4_nlx4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemv_mxfp4_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(.mxfp4, 4, 4, blocks.block_mxfp4x4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_mxfp4_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(.mxfp4, 4, 4, blocks.block_mxfp4x4, n, s, bs, vx, vy, nr, nc);
}

// The four `8x8` shapes: `arch-fallback.h` renames the generic to the bare
// name on this target, so these carry no suffix.
pub export fn ggml_gemv_iq4_nl_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(.iq4_nl, 8, 8, blocks.block_iq4_nlx8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_iq4_nl_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(.iq4_nl, 8, 8, blocks.block_iq4_nlx8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemv_mxfp4_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(.mxfp4, 8, 8, blocks.block_mxfp4x8, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_mxfp4_8x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(.mxfp4, 8, 8, blocks.block_mxfp4x8, n, s, bs, vx, vy, nr, nc);
}
