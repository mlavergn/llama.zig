//! The `q1_0` × `q8_0` interleaved gemv and gemm, generic forms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.9.0
//! (`cd959bae9`), where they are new. Each declaration below names the C++
//! it replaces and the line it began at.
//!
//! # One sign bit per weight
//!
//! A `q1_0` weight is `±d`: bit set is `+d`, bit clear is `-d`. A
//! `block_q1_0` holds 128 of them (`QK1_0`), so it spans four `q8_0`
//! activation blocks, and the kernels walk `k` over those four.
//!
//! The two shapes interleave the bits differently (`convert.zig`'s
//! `makeBlockQ1_0x4`): `4x4` packs four weights of two columns into each
//! byte -- `bits_lo` holds columns 0 and 1, `bits_hi` columns 2 and 3 --
//! and `4x8` keeps each source byte whole, eight weights of one column.
//!
//! # Where these are live
//!
//! On ARM `arch/arm/repack.cpp` has a NEON body for all four
//! (`repack/arm/q1_0.zig`), behind `__ARM_FEATURE_DOTPROD` -- the `4x8`
//! gemm behind `__ARM_FEATURE_MATMUL_INT8` -- so these run there only at
//! `generic`, Linux arm64's baseline, and the `4x8` gemm at the baseline
//! too. On x86 `arch-fallback.h` renames all four to the bare names
//! (`x86/fallback.zig`), so they are the real kernels.
//!
//! # The float arithmetic
//!
//! The gemvs accumulate `sumf[j] += (bit ? scale[j] : -scale[j]) * q`, with
//! `scale[j] = d0[j] * d1` rounded on its own: one fused step per weight,
//! on both architectures.
//!
//! The gemms write `sumf[m][j] += sign[j] * q_row[m] * d0[j] * row_scale`,
//! whose `sign * q` is an `int` product -- the expression
//! `epilogue.accumulate` names -- and **whether clang fuses it depends on
//! the architecture**, read off the reference's disassembly at every CPU
//! the gates build for. On aarch64 (baseline, `apple_m3`, `generic`,
//! `apple_a7`, `apple_a12`) the SLP vectorizer packs the four (or eight)
//! `p` iterations of one `sumf[m][j]` into a vector, rounds both multiplies
//! there (`scvtf.4s`, `fmul.4s`, `fmul.4s`) and adds the lanes into
//! `sumf` one by one with plain `fadd`: no fusion at all. On x86-64-v3 and
//! v4 it packs the four columns `j` instead, and `sumf[m]` takes a
//! `vmulps` then a `vfmadd`: the outer multiply fused, as in
//! `epilogue.accumulate`. `gemmStep` keys on the architecture for that.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../../impl.zig");
const blocks = @import("blocks.zig");
const epilogue = @import("epilogue.zig");
const convert = @import("../convert.zig");

const c = impl.c;

const QK1_0 = c.QK1_0;
const QK8_0 = 32;

/// One `sumf[m][j] += sign[j] * q_row[m] * d0[j] * row_scale` of a
/// generic gemm, as clang compiles it on this architecture; see the file
/// header.
inline fn gemmStep(acc: f32, sumi: i32, d_weight: f32, d_act: f32) f32 {
    if (comptime builtin.cpu.arch == .x86_64) return epilogue.accumulate(acc, sumi, d_weight, d_act);
    return acc + @as(f32, @floatFromInt(sumi)) * d_weight * d_act;
}

/// `(bits & (1u << p)) ? 1 : -1`, the sign a weight's bit stands for.
inline fn sign(bits: u8, p: u3) i32 {
    return if (bits & (@as(u8, 1) << p) != 0) 1 else -1;
}

/// `((bits & (1u << p)) ? scale : -scale)`.
inline fn signed(bits: u8, p: u3, scale: f32) f32 {
    return if (bits & (@as(u8, 1) << p) != 0) scale else -scale;
}

/// The four column scales of one interleaved block.
inline fn colScales(b: *const blocks.block_q1_0x4) [4]f32 {
    return .{
        convert.cpuFp16ToFp32(b.d[0]),
        convert.cpuFp16ToFp32(b.d[1]),
        convert.cpuFp16ToFp32(b.d[2]),
        convert.cpuFp16ToFp32(b.d[3]),
    };
}

/// Ports `ggml_gemv_q1_0_4x4_q8_0_generic` (ggml-cpu/repack.cpp:1368 @cd959bae9).
pub export fn ggml_gemv_q1_0_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const qk: c_int = QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));
    const ncols_interleaved = 4;

    var sumf: [4]f32 = undefined;

    const a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
    const ncb: usize = @intCast(@divTrunc(nc, ncols_interleaved));
    for (0..ncb) |x| {
        const b_ptr = @as([*]const blocks.block_q1_0x4, @ptrCast(@alignCast(vx))) + x * nb;

        for (0..ncols_interleaved) |j| sumf[j] = 0.0;

        for (0..nb) |l| {
            const d0 = colScales(&b_ptr[l]);

            for (0..QK1_0 / QK8_0) |k| {
                const a_blk = &a_ptr[l * (QK1_0 / QK8_0) + k];
                const d1 = convert.cpuFp16ToFp32(a_blk.d);
                const scale = [4]f32{ d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };

                for (0..QK8_0 / 4) |tile| {
                    const bits_lo: u8 = @bitCast(b_ptr[l].qs[k * 16 + 2 * tile + 0]);
                    const bits_hi: u8 = @bitCast(b_ptr[l].qs[k * 16 + 2 * tile + 1]);

                    for (0..4) |pi| {
                        const p: u3 = @intCast(pi);
                        const q: f32 = @floatFromInt(a_blk.qs[tile * 4 + pi]);

                        sumf[0] = @mulAdd(f32, signed(bits_lo, p, scale[0]), q, sumf[0]);
                        sumf[1] = @mulAdd(f32, signed(bits_lo, 4 + p, scale[1]), q, sumf[1]);
                        sumf[2] = @mulAdd(f32, signed(bits_hi, p, scale[2]), q, sumf[2]);
                        sumf[3] = @mulAdd(f32, signed(bits_hi, 4 + p, scale[3]), q, sumf[3]);
                    }
                }
            }
        }

        for (0..ncols_interleaved) |j| s[x * ncols_interleaved + j] = sumf[j];
    }
}

/// Ports `ggml_gemv_q1_0_4x8_q8_0_generic` (ggml-cpu/repack.cpp:1431 @cd959bae9).
pub export fn ggml_gemv_q1_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    _ = bs;
    _ = nr;
    const qk: c_int = QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));
    const ncols_interleaved = 4;
    const blocklen = 8;

    var sumf: [4]f32 = undefined;

    const a_ptr: [*]const c.block_q8_0 = @ptrCast(@alignCast(vy));
    const ncb: usize = @intCast(@divTrunc(nc, ncols_interleaved));
    for (0..ncb) |x| {
        const b_ptr = @as([*]const blocks.block_q1_0x4, @ptrCast(@alignCast(vx))) + x * nb;

        for (0..ncols_interleaved) |j| sumf[j] = 0.0;

        for (0..nb) |l| {
            const d0 = colScales(&b_ptr[l]);

            for (0..QK1_0 / blocklen) |k| {
                const a_blk = &a_ptr[l * (QK1_0 / QK8_0) + k / (QK8_0 / blocklen)];
                const d1 = convert.cpuFp16ToFp32(a_blk.d);
                const scale = [4]f32{ d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };
                const bits0: u8 = @bitCast(b_ptr[l].qs[k * ncols_interleaved + 0]);
                const bits1: u8 = @bitCast(b_ptr[l].qs[k * ncols_interleaved + 1]);
                const bits2: u8 = @bitCast(b_ptr[l].qs[k * ncols_interleaved + 2]);
                const bits3: u8 = @bitCast(b_ptr[l].qs[k * ncols_interleaved + 3]);
                const q_offset = (k % (QK8_0 / blocklen)) * blocklen;

                for (0..blocklen) |pi| {
                    const p: u3 = @intCast(pi);
                    const q: f32 = @floatFromInt(a_blk.qs[q_offset + pi]);

                    sumf[0] = @mulAdd(f32, signed(bits0, p, scale[0]), q, sumf[0]);
                    sumf[1] = @mulAdd(f32, signed(bits1, p, scale[1]), q, sumf[1]);
                    sumf[2] = @mulAdd(f32, signed(bits2, p, scale[2]), q, sumf[2]);
                    sumf[3] = @mulAdd(f32, signed(bits3, p, scale[3]), q, sumf[3]);
                }
            }
        }

        for (0..ncols_interleaved) |j| s[x * ncols_interleaved + j] = sumf[j];
    }
}

/// Ports `ggml_gemm_q1_0_4x4_q8_0_generic` (ggml-cpu/repack.cpp:2513 @cd959bae9).
pub export fn ggml_gemm_q1_0_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    const qk: c_int = QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));
    const ncols_interleaved = 4;

    var sumf: [4][4]f32 = undefined;

    const nrb: usize = @intCast(@divTrunc(nr, 4));
    const ncb: usize = @intCast(@divTrunc(nc, ncols_interleaved));
    for (0..nrb) |y| {
        const a_ptr = @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + 4 * y * nb;
        for (0..ncb) |x| {
            const b_ptr = @as([*]const blocks.block_q1_0x4, @ptrCast(@alignCast(vx))) + x * nb;

            for (0..4) |m| {
                for (0..ncols_interleaved) |j| sumf[m][j] = 0.0;
            }

            for (0..nb) |l| {
                const d0 = colScales(&b_ptr[l]);

                for (0..QK1_0 / QK8_0) |k| {
                    const a_blk = &a_ptr[4 * l + k];
                    const a_d = colScalesQ8(a_blk);

                    for (0..QK8_0 / 4) |tile| {
                        const bits_lo: u8 = @bitCast(b_ptr[l].qs[k * 16 + 2 * tile + 0]);
                        const bits_hi: u8 = @bitCast(b_ptr[l].qs[k * 16 + 2 * tile + 1]);
                        const tile_offset = tile * 16;

                        for (0..4) |pi| {
                            const p: u3 = @intCast(pi);
                            const q_row = [4]i32{
                                a_blk.qs[tile_offset + 0 * 4 + pi],
                                a_blk.qs[tile_offset + 1 * 4 + pi],
                                a_blk.qs[tile_offset + 2 * 4 + pi],
                                a_blk.qs[tile_offset + 3 * 4 + pi],
                            };
                            const sg = [4]i32{
                                sign(bits_lo, p),
                                sign(bits_lo, 4 + p),
                                sign(bits_hi, p),
                                sign(bits_hi, 4 + p),
                            };

                            for (0..4) |m| {
                                const row_scale = a_d[m];
                                for (0..4) |j| {
                                    sumf[m][j] = gemmStep(sumf[m][j], sg[j] * q_row[m], d0[j], row_scale);
                                }
                            }
                        }
                    }
                }
            }

            for (0..4) |m| {
                for (0..ncols_interleaved) |j| {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

/// Ports `ggml_gemm_q1_0_4x8_q8_0_generic` (ggml-cpu/repack.cpp:2598 @cd959bae9).
pub export fn ggml_gemm_q1_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    const qk: c_int = QK1_0;
    const nb: usize = @intCast(@divTrunc(n, qk));
    const ncols_interleaved = 4;
    const blocklen = 8;

    var sumf: [4][4]f32 = undefined;

    const nrb: usize = @intCast(@divTrunc(nr, 4));
    const ncb: usize = @intCast(@divTrunc(nc, ncols_interleaved));
    for (0..nrb) |y| {
        const a_ptr = @as([*]const blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + 4 * y * nb;
        for (0..ncb) |x| {
            const b_ptr = @as([*]const blocks.block_q1_0x4, @ptrCast(@alignCast(vx))) + x * nb;

            for (0..4) |m| {
                for (0..ncols_interleaved) |j| sumf[m][j] = 0.0;
            }

            for (0..nb) |l| {
                const d0 = colScales(&b_ptr[l]);

                for (0..QK1_0 / blocklen) |k| {
                    const a_blk = &a_ptr[4 * l + k / (QK8_0 / blocklen)];
                    const a_d = colScalesQ8(a_blk);
                    const bits = [4]u8{
                        @bitCast(b_ptr[l].qs[k * ncols_interleaved + 0]),
                        @bitCast(b_ptr[l].qs[k * ncols_interleaved + 1]),
                        @bitCast(b_ptr[l].qs[k * ncols_interleaved + 2]),
                        @bitCast(b_ptr[l].qs[k * ncols_interleaved + 3]),
                    };
                    const q_offset = (k % (QK8_0 / blocklen)) * 4 * blocklen;

                    for (0..blocklen) |pi| {
                        const p: u3 = @intCast(pi);
                        const q_row = [4]i32{
                            a_blk.qs[q_offset + 0 * blocklen + pi],
                            a_blk.qs[q_offset + 1 * blocklen + pi],
                            a_blk.qs[q_offset + 2 * blocklen + pi],
                            a_blk.qs[q_offset + 3 * blocklen + pi],
                        };
                        const sg = [4]i32{ sign(bits[0], p), sign(bits[1], p), sign(bits[2], p), sign(bits[3], p) };

                        for (0..4) |m| {
                            const row_scale = a_d[m];
                            for (0..4) |j| {
                                sumf[m][j] = gemmStep(sumf[m][j], sg[j] * q_row[m], d0[j], row_scale);
                            }
                        }
                    }
                }
            }

            for (0..4) |m| {
                for (0..ncols_interleaved) |j| {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

/// The four row scales of one interleaved activation block.
inline fn colScalesQ8(a: *const blocks.block_q8_0x4) [4]f32 {
    return .{
        convert.cpuFp16ToFp32(a.d[0]),
        convert.cpuFp16ToFp32(a.d[1]),
        convert.cpuFp16ToFp32(a.d[2]),
        convert.cpuFp16ToFp32(a.d[3]),
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a set bit is the positive sign" {
    try std.testing.expectEqual(@as(i32, 1), sign(0b0000_0100, 2));
    try std.testing.expectEqual(@as(i32, -1), sign(0b0000_0100, 3));
    try std.testing.expectEqual(@as(f32, -2.0), signed(0, 7, 2.0));
}
