//! The NEON `q8_0` × `q8_0` interleaved gemv.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # The two shapes take different routes to the same sum
//!
//! `4x4` broadcasts one 32-bit **lane** of the activation across a vector
//! and issues eight `vdotq_laneq_s32`, one accumulator. `4x8` duplicates
//! each 8-byte activation half with `vcombine_s8`, issues eight plain
//! `vdotq_s32` into **two** accumulators, and merges them with
//! `vpaddq_s32`. They are not the same loop with a constant changed, so
//! they are written out separately as the C does.
//!
//! # `vpaddq_s32` is a pairwise add across two vectors
//!
//! `vpaddq_s32(a, b)` is `[a0+a1, a2+a3, b0+b1, b2+b3]`, not `a + b`.
//! Getting that wrong gives a plausible-looking result with the columns
//! scrambled.
//!
//! # One FMA per block
//!
//! `acc = vfmaq_f32(acc, f32(ret), ad * bd)` — fused, one rounding. The
//! scale product `ad * bd` is a separate multiply the C does not fuse.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../blocks.zig");
const neon = @import("../../quants/arm/neon.zig");

const c = impl.c;
const f32x4 = neon.f32x4;
const i32x4 = neon.i32x4;
const i8x16 = neon.i8x16;
const i8x8 = neon.i8x8;

/// Four consecutive `i8x16` from memory: `vld1q_s8_x4`.
inline fn load4q(p: [*]const i8) [4]i8x16 {
    return .{
        @bitCast(@as(@Vector(16, i8), p[0..16].*)),
        @bitCast(@as(@Vector(16, i8), p[16..32].*)),
        @bitCast(@as(@Vector(16, i8), p[32..48].*)),
        @bitCast(@as(@Vector(16, i8), p[48..64].*)),
    };
}

/// Two consecutive `i8x16` from memory: `vld1q_s8_x2`.
inline fn load2q(p: [*]const i8) [2]i8x16 {
    return .{
        @bitCast(@as(@Vector(16, i8), p[0..16].*)),
        @bitCast(@as(@Vector(16, i8), p[16..32].*)),
    };
}

/// Four consecutive `i8x8` from memory: `vld1_s8_x4`.
inline fn load4d(p: [*]const i8) [4]i8x8 {
    return .{
        @bitCast(@as(@Vector(8, i8), p[0..8].*)),
        @bitCast(@as(@Vector(8, i8), p[8..16].*)),
        @bitCast(@as(@Vector(8, i8), p[16..24].*)),
        @bitCast(@as(@Vector(8, i8), p[24..32].*)),
    };
}

/// Ports `ggml_gemv_q8_0_4x4_q8_0` (arch/arm/repack.cpp:1699 @c1d0e7a00).
pub export fn ggml_gemv_q8_0_4x4_q8_0(n: c_int, s_out: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const nb = @divTrunc(n, 32);
    const ncols_interleaved: c_int = 4;

    var b_ptr: [*]const blocks.block_q8_0x4 = @ptrCast(@alignCast(vx));
    var s = s_out;

    var col: c_int = 0;
    while (col < nc) : (col += ncols_interleaved) {
        var a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
        var acc: f32x4 = @splat(0);
        var b: c_int = 0;
        while (b < nb) : (b += 1) {
            const qs: [*]const i8 = @ptrCast(&b_ptr[0].qs);
            const b_low = load4q(qs);
            const b_high = load4q(qs + 64);
            const bd = neon.load_f16x4(@ptrCast(&b_ptr[0].d));

            const a = load2q(@ptrCast(&a_ptr[0].qs));
            const ad = neon.dup_f16x4(a_ptr[0].d);

            var ret: i32x4 = @splat(0);
            inline for (0..4) |l| ret = neon.dotq_laneq_s32(ret, b_low[l], a[0], l);
            inline for (0..4) |l| ret = neon.dotq_laneq_s32(ret, b_high[l], a[1], l);

            acc = neon.fma_f32(acc, neon.cvt_f32_s32(ret), neon.cvt_f32_f16(ad) * neon.cvt_f32_f16(bd));
            a_ptr += 1;
            b_ptr += 1;
        }
        s[0..4].* = acc;
        s += @intCast(ncols_interleaved);
    }
}

/// Ports `ggml_gemv_q8_0_4x8_q8_0` (arch/arm/repack.cpp:1757 @c1d0e7a00).
pub export fn ggml_gemv_q8_0_4x8_q8_0(n: c_int, s_out: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const nb = @divTrunc(n, 32);
    const ncols_interleaved: c_int = 4;

    var b_ptr: [*]const blocks.block_q8_0x4 = @ptrCast(@alignCast(vx));
    var s = s_out;

    var col: c_int = 0;
    while (col < nc) : (col += ncols_interleaved) {
        var a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
        var acc: f32x4 = @splat(0);
        var b: c_int = 0;
        while (b < nb) : (b += 1) {
            const qs: [*]const i8 = @ptrCast(&b_ptr[0].qs);
            const b_low = load4q(qs);
            const b_high = load4q(qs + 64);
            const bd = neon.load_f16x4(@ptrCast(&b_ptr[0].d));

            const a_chunks = load4d(@ptrCast(&a_ptr[0].qs));
            const ad = neon.dup_f16x4(a_ptr[0].d);

            // Each 8-byte half is used twice, so it is duplicated into a
            // full vector rather than lane-broadcast.
            const a0 = neon.combine(a_chunks[0], a_chunks[0]);
            const a1 = neon.combine(a_chunks[1], a_chunks[1]);
            const a2 = neon.combine(a_chunks[2], a_chunks[2]);
            const a3 = neon.combine(a_chunks[3], a_chunks[3]);

            var ret0: i32x4 = @splat(0);
            var ret1: i32x4 = @splat(0);
            ret0 = neon.dotq_s32(ret0, b_low[0], a0);
            ret1 = neon.dotq_s32(ret1, b_low[1], a0);
            ret0 = neon.dotq_s32(ret0, b_low[2], a1);
            ret1 = neon.dotq_s32(ret1, b_low[3], a1);
            ret0 = neon.dotq_s32(ret0, b_high[0], a2);
            ret1 = neon.dotq_s32(ret1, b_high[1], a2);
            ret0 = neon.dotq_s32(ret0, b_high[2], a3);
            ret1 = neon.dotq_s32(ret1, b_high[3], a3);

            // Pairwise across the two accumulators, not `ret0 + ret1`.
            const ret = neon.paddq_s32(ret0, ret1);

            acc = neon.fma_f32(acc, neon.cvt_f32_s32(ret), neon.cvt_f32_f16(ad) * neon.cvt_f32_f16(bd));
            a_ptr += 1;
            b_ptr += 1;
        }
        s[0..4].* = acc;
        s += @intCast(ncols_interleaved);
    }
}

/// Ports `ggml_gemm_q8_0_4x4_q8_0` (arch/arm/repack.cpp:4938 @c1d0e7a00).
///
/// Four row accumulators, each fed by `vdotq_laneq_s32` on a different
/// 32-bit lane of the activation — so one pass over the weights serves all
/// four rows. The per-row scale is `b_d * a_d[m]` with `a_d[m]` broadcast,
/// and `vmlaq_f32` folds it in **fused**, like its `_n_` sibling.
pub export fn ggml_gemm_q8_0_4x4_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    const nb = @divTrunc(n, 32);
    const ncols_interleaved: c_int = 4;

    var y: c_int = 0;
    while (y < @divTrunc(nr, 4)) : (y += 1) {
        const a_ptr: [*]const blocks.block_q8_0x4 =
            @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));
        var x: c_int = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const b_ptr: [*]const blocks.block_q8_0x4 =
                @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

            var sumf: [4]f32x4 = .{@as(f32x4, @splat(0))} ** 4;

            var l: usize = 0;
            while (l < @as(usize, @intCast(nb))) : (l += 1) {
                const a_d = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&a_ptr[l].d)));
                const b_d = neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(&b_ptr[l].d)));

                var sumi: [4]i32x4 = .{@as(i32x4, @splat(0))} ** 4;

                var k_group: usize = 0;
                while (k_group < 8) : (k_group += 4) {
                    const a = load4q(@as([*]const i8, @ptrCast(&a_ptr[l].qs)) + 16 * k_group);
                    const b = load4q(@as([*]const i8, @ptrCast(&b_ptr[l].qs)) + 16 * k_group);
                    for (0..4) |k| {
                        inline for (0..4) |m| {
                            sumi[m] = neon.dotq_laneq_s32(sumi[m], b[k], a[k], m);
                        }
                    }
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

comptime {
    _ = std;
}
