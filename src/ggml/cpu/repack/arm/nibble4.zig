//! The NEON `iq4_nl` and `mxfp4` interleaved gemv and gemm.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # The codebook is a table-lookup instruction
//!
//! Where the scalar form indexes `kvalues_iq4nl[nibble]`, NEON loads the
//! whole 16-entry table into a vector once and uses `vqtbl1q_s8` to look
//! up sixteen nibbles at a time. Same values, one instruction.
//!
//! # No `<< 4` correction here
//!
//! Unlike `q4_0`, the codebook already yields the real magnitudes, so the
//! nibbles are masked down (`& 0xF`, `>> 4`) rather than shifted up, and
//! the accumulator converts with a plain `vcvtq_f32_s32` — no
//! `vcvtq_n_f32_s32` divide.
//!
//! # The two formats differ only in the table and the column scale
//!
//! `iq4_nl` loads four `f16` deltas; `mxfp4` builds its scale vector from
//! four `e8m0` exponents through `ggml_e8m0_to_fp32_half`, element by
//! element — there is no vector form of that conversion.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../blocks.zig");
const neon = @import("../../quants/arm/neon.zig");

const c = impl.c;
const f32x4 = neon.f32x4;
const i32x4 = neon.i32x4;
const i8x16 = neon.i8x16;
const u8x16 = neon.u8x16;

const Fmt = enum { iq4_nl, mxfp4 };

inline fn table(comptime fmt: Fmt) i8x16 {
    return switch (fmt) {
        .iq4_nl => @bitCast(@as(@Vector(16, i8), c.kvalues_iq4nl[0..16].*)),
        .mxfp4 => @bitCast(@as(@Vector(16, i8), c.kvalues_mxfp4[0..16].*)),
    };
}

inline fn loadq(p: [*]const i8) i8x16 {
    return @bitCast(@as(@Vector(16, i8), p[0..16].*));
}

inline fn loadqu(p: [*]const u8) u8x16 {
    return p[0..16].*;
}

/// The column scale: four `f16` deltas, or four `e8m0` exponents converted
/// one at a time.
inline fn colScales(comptime fmt: Fmt, bl: anytype) f32x4 {
    return switch (fmt) {
        .iq4_nl => neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&bl.d))),
        .mxfp4 => f32x4{
            impl.e8m0ToFp32Half(bl.e[0]),
            impl.e8m0ToFp32Half(bl.e[1]),
            impl.e8m0ToFp32Half(bl.e[2]),
            impl.e8m0ToFp32Half(bl.e[3]),
        },
    };
}

/// Ports `ggml_gemv_iq4_nl_4x4_q8_0` and `ggml_gemv_mxfp4_4x4_q8_0`
/// (arch/arm/repack.cpp:431, 501 @c1d0e7a00).
fn gemv(comptime fmt: Fmt, comptime BlockX: type, n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    _ = bs;
    _ = nr;
    const nb = @divTrunc(n, 32);
    const ncols_interleaved: c_int = 4;
    const kvalues = table(fmt);

    const res_ptr = s;
    var x: c_int = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const b_ptr: [*]const BlockX =
            @as([*]const BlockX, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));
        const a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));

        var sumf: f32x4 = @splat(0);
        var l: usize = 0;
        while (l < @as(usize, @intCast(nb))) : (l += 1) {
            const qs: [*]const u8 = @ptrCast(&b_ptr[l].qs);
            var b_lo: [4]i8x16 = undefined;
            var b_hi: [4]i8x16 = undefined;
            inline for (0..4) |i| {
                const bq = loadqu(qs + i * 16);
                b_hi[i] = neon.qtbl1q_s8(kvalues, bq >> @as(@Vector(16, u3), @splat(4)));
                b_lo[i] = neon.qtbl1q_s8(kvalues, bq & @as(u8x16, @splat(0x0F)));
            }

            const aqs: [*]const i8 = @ptrCast(&a_ptr[l].qs);
            const a_0 = loadq(aqs);
            const a_1 = loadq(aqs + 16);

            var sumi: i32x4 = @splat(0);
            inline for (0..4) |i| {
                sumi = neon.dotq_laneq_s32(sumi, b_lo[i], a_0, i);
                sumi = neon.dotq_laneq_s32(sumi, b_hi[i], a_1, i);
            }

            const a_d = neon.cvt_f32_f16(neon.dup_f16x4(a_ptr[l].d));
            const b_d = colScales(fmt, &b_ptr[l]);
            const d = a_d * b_d;
            sumf = neon.fma_f32(sumf, d, neon.cvt_f32_s32(sumi));
        }
        (res_ptr + @as(usize, @intCast(x * 4)))[0..4].* = sumf;
    }
}

/// Ports `ggml_gemm_iq4_nl_4x4_q8_0` and `ggml_gemm_mxfp4_4x4_q8_0`
/// (arch/arm/repack.cpp:3166, 3242 @c1d0e7a00).
fn gemm(comptime fmt: Fmt, comptime BlockX: type, n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void {
    const nb = @divTrunc(n, 32);
    const ncols_interleaved: c_int = 4;
    const kvalues = table(fmt);

    var y: c_int = 0;
    while (y < @divTrunc(nr, 4)) : (y += 1) {
        const a_ptr: [*]const blocks.block_q8_0x4 =
            @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));
        var x: c_int = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const b_ptr: [*]const BlockX =
                @as([*]const BlockX, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

            var sumf: [4]f32x4 = .{@as(f32x4, @splat(0))} ** 4;

            var l: usize = 0;
            while (l < @as(usize, @intCast(nb))) : (l += 1) {
                const a_d = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&a_ptr[l].d)));
                const b_d = colScales(fmt, &b_ptr[l]);

                var sumi: [4]i32x4 = .{@as(i32x4, @splat(0))} ** 4;

                const qs: [*]const u8 = @ptrCast(&b_ptr[l].qs);
                const aqs: [*]const i8 = @ptrCast(&a_ptr[l].qs);
                for (0..4) |k| {
                    const bq = loadqu(qs + 16 * k);
                    const b_hi = neon.qtbl1q_s8(kvalues, bq >> @as(@Vector(16, u3), @splat(4)));
                    const b_lo = neon.qtbl1q_s8(kvalues, bq & @as(u8x16, @splat(0x0F)));

                    const a_0 = loadq(aqs + 16 * k);
                    const a_1 = loadq(aqs + 16 * k + 64);

                    inline for (0..4) |m| sumi[m] = neon.dotq_laneq_s32(sumi[m], b_lo, a_0, m);
                    inline for (0..4) |m| sumi[m] = neon.dotq_laneq_s32(sumi[m], b_hi, a_1, m);
                }

                inline for (0..4) |m| {
                    const scale = b_d * @as(f32x4, @splat(a_d[m]));
                    sumf[m] = neon.fma_f32(sumf[m], scale, neon.cvt_f32_s32(sumi[m]));
                }
            }

            for (0..4) |m| {
                const off: usize = @intCast((y * 4 + @as(c_int, @intCast(m))) * @as(c_int, @intCast(bs)) + x * 4);
                s[off..][0..4].* = sumf[m];
            }
        }
    }
}

pub export fn ggml_gemv_iq4_nl_4x4_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(.iq4_nl, blocks.block_iq4_nlx4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemv_mxfp4_4x4_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(.mxfp4, blocks.block_mxfp4x4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_iq4_nl_4x4_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(.iq4_nl, blocks.block_iq4_nlx4, n, s, bs, vx, vy, nr, nc);
}
pub export fn ggml_gemm_mxfp4_4x4_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(.mxfp4, blocks.block_mxfp4x4, n, s, bs, vx, vy, nr, nc);
}

comptime {
    _ = std;
}
