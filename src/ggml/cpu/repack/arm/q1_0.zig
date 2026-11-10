//! The NEON `q1_0` × `q8_0` interleaved gemvs and gemms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.9.0
//! (`cd959bae9`), where they are new. Each declaration below names the C++
//! it replaces and the line it began at.
//!
//! # Which arm compiles where
//!
//! The two gemvs and the `4x4` gemm have a body behind
//! `__ARM_FEATURE_DOTPROD`, which every Apple core has; the `4x8` gemm's is
//! behind `__ARM_FEATURE_MATMUL_INT8`, so the baseline (M1) takes its
//! generic form and `apple_m3` its body. At `generic`, Linux arm64's
//! baseline, all four fall to `repack/q1_0.zig`. Each picks at comptime, as
//! the C's `#if` does.
//!
//! # A weight bit becomes a byte, then a dot product
//!
//! `ggml_q1_0_unpack_pair` turns two bytes of bits into sixteen `±1` bytes
//! through a 256-entry table, and the kernels then run the same `sdot` and
//! `smmla` they run for `q8_0`. All of that is integer and exact; the float
//! work per block is two fused steps, `accb += f32(sumi) * ad` and then
//! `acc += accb * bd`, each lane its own column (or row).

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../blocks.zig");
const neon = @import("../../quants/arm/neon.zig");
const convert = @import("../../convert.zig");

const c = impl.c;
const f32x4 = neon.f32x4;
const i32x4 = neon.i32x4;
const i8x16 = neon.i8x16;
const i8x8 = neon.i8x8;

const QK1_0 = c.QK1_0;

// The `_generic` bodies in `cpu/repack/q1_0.zig`, the C's trailing call
// after each `#if`.
extern fn ggml_gemv_q1_0_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemv_q1_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q1_0_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q1_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;

/// Ports `table_q1_signs` (arch/arm/repack.cpp:61 @cd959bae9): for each
/// byte, eight bytes of `+1` where a bit is set and `-1` (`0xff`) where it
/// is clear, bit `j` in byte `j`. The C builds it with the `B1`..`B8`
/// token-pasting macros; this computes the same 256 words.
const table_q1_signs: [256]u64 = blk: {
    @setEvalBranchQuota(4096);
    var t: [256]u64 = undefined;
    for (0..256) |i| {
        var v: u64 = 0;
        for (0..8) |j| {
            const byte: u64 = if ((i >> j) & 1 != 0) 0x01 else 0xff;
            v |= byte << (8 * j);
        }
        t[i] = v;
    }
    break :blk t;
};

/// Ports `ggml_q1_0_unpack_pair` (arch/arm/repack.cpp:63 @cd959bae9):
/// `vcombine_u8(vcreate_u8(table[bits0]), vcreate_u8(table[bits1]))`.
inline fn unpackPair(bits0: u8, bits1: u8) i8x16 {
    const lo: i8x8 = @bitCast(table_q1_signs[bits0]);
    const hi: i8x8 = @bitCast(table_q1_signs[bits1]);
    return neon.combine(lo, hi);
}

inline fn loadq(p: [*]const i8) i8x16 {
    return @bitCast(@as(@Vector(16, i8), p[0..16].*));
}

inline fn loadd(p: [*]const i8) i8x8 {
    return @bitCast(@as(@Vector(8, i8), p[0..8].*));
}

inline fn qsOf(b: *const blocks.block_q1_0x4) [*]const u8 {
    return @ptrCast(&b.qs);
}

/// `vcvt_f32_f16(vld1_f16(d))` over a block's four scales.
inline fn scales4(d: *const [4]c.ggml_half) f32x4 {
    return neon.cvt_f32_f16(neon.load_f16x4(@ptrCast(d)));
}

/// Ports `ggml_gemv_q1_0_4x4_q8_0` (arch/arm/repack.cpp:1963 @cd959bae9).
pub export fn ggml_gemv_q1_0_4x4_q8_0(n: c_int, s_out: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !neon.has_dotprod) return ggml_gemv_q1_0_4x4_q8_0_generic(n, s_out, bs, vx, vy, nr, nc);
    const qk: c_int = QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));
    const ncols_interleaved: c_int = 4;

    var s = s_out;
    var col: c_int = 0;
    while (col < nc) : (col += ncols_interleaved) {
        const b_ptr = @as([*]const blocks.block_q1_0x4, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(@divTrunc(col, ncols_interleaved))) * nb;
        const a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
        var acc: f32x4 = @splat(0);

        for (0..nb) |l| {
            const b_d = scales4(&b_ptr[l].d);
            const qs = qsOf(&b_ptr[l]);
            var accb: f32x4 = @splat(0);

            for (0..4) |k| {
                const a_blk = &a_ptr[l * 4 + k];
                const ad = convert.cpuFp16ToFp32(a_blk.d);
                var ret: i32x4 = @splat(0);

                var tile: usize = 0;
                while (tile < 8) : (tile += 4) {
                    const signs0 = unpackPair(qs[k * 16 + 2 * (tile + 0) + 0], qs[k * 16 + 2 * (tile + 0) + 1]);
                    const signs1 = unpackPair(qs[k * 16 + 2 * (tile + 1) + 0], qs[k * 16 + 2 * (tile + 1) + 1]);
                    const signs2 = unpackPair(qs[k * 16 + 2 * (tile + 2) + 0], qs[k * 16 + 2 * (tile + 2) + 1]);
                    const signs3 = unpackPair(qs[k * 16 + 2 * (tile + 3) + 0], qs[k * 16 + 2 * (tile + 3) + 1]);
                    const q_tiles = loadq(@as([*]const i8, &a_blk.qs) + tile * 4);

                    ret = neon.dotq_laneq_s32(ret, signs0, q_tiles, 0);
                    ret = neon.dotq_laneq_s32(ret, signs1, q_tiles, 1);
                    ret = neon.dotq_laneq_s32(ret, signs2, q_tiles, 2);
                    ret = neon.dotq_laneq_s32(ret, signs3, q_tiles, 3);
                }

                accb = neon.fma_f32(accb, neon.cvt_f32_s32(ret), @splat(ad));
            }
            acc = neon.fma_f32(acc, accb, b_d);
        }
        s[0..4].* = acc;
        s += @intCast(ncols_interleaved);
    }
}

/// Ports `ggml_gemv_q1_0_4x8_q8_0` (arch/arm/repack.cpp:2024 @cd959bae9).
pub export fn ggml_gemv_q1_0_4x8_q8_0(n: c_int, s_out: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !neon.has_dotprod) return ggml_gemv_q1_0_4x8_q8_0_generic(n, s_out, bs, vx, vy, nr, nc);
    const qk: c_int = QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));
    const ncols_interleaved: c_int = 4;

    var s = s_out;
    var col: c_int = 0;
    while (col < nc) : (col += ncols_interleaved) {
        const b_ptr = @as([*]const blocks.block_q1_0x4, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(@divTrunc(col, ncols_interleaved))) * nb;
        const a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
        var acc: f32x4 = @splat(0);

        for (0..nb) |l| {
            const b_d = scales4(&b_ptr[l].d);
            var accb: f32x4 = @splat(0);

            for (0..4) |k| {
                const a_blk = &a_ptr[l * 4 + k];
                const b_qs = qsOf(&b_ptr[l]) + k * 16;
                const ad = convert.cpuFp16ToFp32(a_blk.d);
                const aq: [*]const i8 = &a_blk.qs;
                const a0 = neon.combine(loadd(aq), loadd(aq));
                const a1 = neon.combine(loadd(aq + 8), loadd(aq + 8));
                const a2 = neon.combine(loadd(aq + 16), loadd(aq + 16));
                const a3 = neon.combine(loadd(aq + 24), loadd(aq + 24));

                var ret0: i32x4 = @splat(0);
                var ret1: i32x4 = @splat(0);

                ret0 = neon.dotq_s32(ret0, unpackPair(b_qs[0], b_qs[1]), a0);
                ret1 = neon.dotq_s32(ret1, unpackPair(b_qs[2], b_qs[3]), a0);
                ret0 = neon.dotq_s32(ret0, unpackPair(b_qs[4], b_qs[5]), a1);
                ret1 = neon.dotq_s32(ret1, unpackPair(b_qs[6], b_qs[7]), a1);
                ret0 = neon.dotq_s32(ret0, unpackPair(b_qs[8], b_qs[9]), a2);
                ret1 = neon.dotq_s32(ret1, unpackPair(b_qs[10], b_qs[11]), a2);
                ret0 = neon.dotq_s32(ret0, unpackPair(b_qs[12], b_qs[13]), a3);
                ret1 = neon.dotq_s32(ret1, unpackPair(b_qs[14], b_qs[15]), a3);

                accb = neon.fma_f32(accb, neon.cvt_f32_s32(neon.paddq_s32(ret0, ret1)), @splat(ad));
            }

            acc = neon.fma_f32(acc, accb, b_d);
        }

        s[0..4].* = acc;
        s += @intCast(ncols_interleaved);
    }
}

/// Ports `ggml_gemm_q1_0_4x4_q8_0` (arch/arm/repack.cpp:5608 @cd959bae9).
pub export fn ggml_gemm_q1_0_4x4_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !neon.has_dotprod) return ggml_gemm_q1_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
    const qk: c_int = QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));
    const ncols_interleaved = 4;

    const nrb: usize = @intCast(@divTrunc(nr, 4));
    const ncb: usize = @intCast(@divTrunc(nc, ncols_interleaved));
    for (0..nrb) |y| {
        const a_ptr = @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + 4 * y * nb;
        for (0..ncb) |x| {
            const b_ptr = @as([*]const blocks.block_q1_0x4, @ptrCast(@alignCast(vx))) + x * nb;

            var sumf: [4]f32x4 = @splat(@splat(0));

            for (0..nb) |l| {
                const b_d = scales4(&b_ptr[l].d);
                const qs = qsOf(&b_ptr[l]);
                var blockf: [4]f32x4 = @splat(@splat(0));

                for (0..4) |k| {
                    const a_blk = &a_ptr[4 * l + k];
                    const a_d = scales4(&a_blk.d);

                    var sumi: [4]i32x4 = @splat(@splat(0));

                    for (0..8) |tile| {
                        const signs = unpackPair(qs[k * 16 + 2 * tile + 0], qs[k * 16 + 2 * tile + 1]);
                        const a_tile = loadq(@as([*]const i8, &a_blk.qs) + tile * 16);

                        sumi[0] = neon.dotq_laneq_s32(sumi[0], signs, a_tile, 0);
                        sumi[1] = neon.dotq_laneq_s32(sumi[1], signs, a_tile, 1);
                        sumi[2] = neon.dotq_laneq_s32(sumi[2], signs, a_tile, 2);
                        sumi[3] = neon.dotq_laneq_s32(sumi[3], signs, a_tile, 3);
                    }

                    // vfmaq_laneq_f32(blockf, f32(sumi), a_d, m)
                    inline for (0..4) |m| {
                        blockf[m] = neon.fma_f32(blockf[m], neon.cvt_f32_s32(sumi[m]), @splat(a_d[m]));
                    }
                }

                inline for (0..4) |m| sumf[m] = neon.fma_f32(sumf[m], blockf[m], b_d);
            }

            for (0..4) |m| {
                (s + (y * 4 + m) * bs + x * 4)[0..4].* = sumf[m];
            }
        }
    }
}

/// Ports `ggml_gemm_q1_0_4x8_q8_0` (arch/arm/repack.cpp:5686 @cd959bae9),
/// the `__ARM_FEATURE_MATMUL_INT8` arm.
pub export fn ggml_gemm_q1_0_4x8_q8_0(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !neon.has_i8mm) return ggml_gemm_q1_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
    const qk: c_int = QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));
    const ncols_interleaved = 4;

    const nrb: usize = @intCast(@divTrunc(nr, 4));
    const ncb: usize = @intCast(@divTrunc(nc, ncols_interleaved));
    for (0..nrb) |y| {
        const a_ptr = @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + 4 * y * nb;

        for (0..ncb) |x| {
            const b_ptr = @as([*]const blocks.block_q1_0x4, @ptrCast(@alignCast(vx))) + x * nb;

            var sumf: [4]f32x4 = @splat(@splat(0));

            for (0..nb) |l| {
                const b_d = scales4(&b_ptr[l].d);
                var blockf: [4]f32x4 = @splat(@splat(0));

                for (0..4) |k| {
                    const a_blk = &a_ptr[4 * l + k];
                    const b_qs = qsOf(&b_ptr[l]) + k * 16;
                    const aq: [*]const i8 = &a_blk.qs;

                    var acc: [4]i32x4 = @splat(@splat(0));

                    for (0..4) |chunk| {
                        const a01 = loadq(aq + chunk * 32);
                        const a23 = loadq(aq + chunk * 32 + 16);
                        const b01 = unpackPair(b_qs[chunk * 4 + 0], b_qs[chunk * 4 + 1]);
                        const b23 = unpackPair(b_qs[chunk * 4 + 2], b_qs[chunk * 4 + 3]);

                        acc[0] = neon.mmlaq_s32(acc[0], a01, b01);
                        acc[1] = neon.mmlaq_s32(acc[1], a01, b23);
                        acc[2] = neon.mmlaq_s32(acc[2], a23, b01);
                        acc[3] = neon.mmlaq_s32(acc[3], a23, b23);
                    }

                    const rows = [4]i32x4{
                        .{ acc[0][0], acc[0][1], acc[1][0], acc[1][1] },
                        .{ acc[0][2], acc[0][3], acc[1][2], acc[1][3] },
                        .{ acc[2][0], acc[2][1], acc[3][0], acc[3][1] },
                        .{ acc[2][2], acc[2][3], acc[3][2], acc[3][3] },
                    };
                    const a_d = scales4(&a_blk.d);

                    inline for (0..4) |m| {
                        blockf[m] = neon.fma_f32(blockf[m], neon.cvt_f32_s32(rows[m]), @splat(a_d[m]));
                    }
                }

                inline for (0..4) |m| sumf[m] = neon.fma_f32(sumf[m], blockf[m], b_d);
            }

            for (0..4) |m| {
                (s + (y * 4 + m) * bs + x * 4)[0..4].* = sumf[m];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the sign table puts bit j in byte j, set as +1" {
    const v: [8]i8 = @bitCast(table_q1_signs[0b1000_0001]);
    try std.testing.expectEqual(@as(i8, 1), v[0]);
    try std.testing.expectEqual(@as(i8, -1), v[1]);
    try std.testing.expectEqual(@as(i8, -1), v[6]);
    try std.testing.expectEqual(@as(i8, 1), v[7]);
}
