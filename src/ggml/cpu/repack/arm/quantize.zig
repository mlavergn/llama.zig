//! The NEON `ggml_quantize_mat_q8_0_*`: quantize four rows at a time into
//! an interleaved `block_q8_0x4`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # The max is a vector tree, and the order is load-bearing
//!
//! The C reduces eight `f32x4` to one by three rounds of `vmaxq_f32` —
//! `(0,1)(2,3)(4,5)(6,7)` then `(0,2)(4,6)` then `(0,4)` — and only then
//! takes `vmaxvq_f32`. For a max that is associative and the order cannot
//! change the result, but it is reproduced as written so the code reads
//! against the C.
//!
//! # `vcvtnq_s32_f32`, not a cast
//!
//! Round-half-to-even to the nearest integer, which is what the scalar
//! `_generic` form spells `roundf`. They differ at exact ties, and the
//! generic is the dead one here.
//!
//! # `4x4` and `4x8` differ in the write interleave only
//!
//! Per `j` iteration each row contributes `blck_size_interleave` bytes at
//! stride `4 * blck`: one vector at 4, two at 8. Written as one loop over
//! `32 / blck` iterations with `blck / 4` vectors per row, which is the
//! C's two bodies expressed once.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../blocks.zig");
const convert = @import("../../convert.zig");
const neon = @import("../../quants/arm/neon.zig");

const c = impl.c;
const f32x4 = neon.f32x4;

/// Ports `ggml_quantize_mat_q8_0_4x4` and `ggml_quantize_mat_q8_0_4x8`
/// (arch/arm/repack.cpp:51, 119 @c1d0e7a00).
fn quantizeMat(comptime blck: i64, x: [*]const f32, vy: *anyopaque, k: i64) void {
    const nb = @divTrunc(k, 32);
    const y: [*]blocks.block_q8_0x4 = @ptrCast(@alignCast(vy));

    var srcv: [4][8]f32x4 = undefined;
    var id: [4]f32 = undefined;

    var i: i64 = 0;
    while (i < nb) : (i += 1) {
        var asrcv: [8]f32x4 = undefined;
        var amaxv: [8]f32x4 = undefined;

        for (0..4) |row_iter| {
            for (0..8) |j| {
                const off: usize = @intCast(@as(i64, @intCast(row_iter)) * k + i * 32 + 4 * @as(i64, @intCast(j)));
                srcv[row_iter][j] = x[off..][0..4].*;
            }
            for (0..8) |j| asrcv[j] = neon.abs_f32(srcv[row_iter][j]);

            for (0..4) |j| amaxv[2 * j] = neon.max_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (0..2) |j| amaxv[4 * j] = neon.max_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (0..1) |j| amaxv[8 * j] = neon.max_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const amax = neon.maxvq_f32(amaxv[0]);

            const d = amax / ((1 << 7) - 1);
            id[row_iter] = if (d != 0.0) 1.0 / d else 0.0;

            y[@intCast(i)].d[row_iter] = convert.cpuFp32ToFp16(d);
        }

        // Per `j`, each of the four rows writes `blck` bytes at stride
        // `4 * blck`; `blck / 4` vectors cover them.
        const vecs_per_row: usize = @intCast(@divTrunc(blck, 4));
        var j: usize = 0;
        while (j < @as(usize, @intCast(@divTrunc(32, blck)))) : (j += 1) {
            for (0..4) |row| {
                for (0..vecs_per_row) |v| {
                    const src = neon.mul_n_f32(srcv[row][j * vecs_per_row + v], id[row]);
                    const vi = neon.cvtnq_s32_f32(src);
                    const base: usize = @intCast(@as(i64, @intCast(j)) * blck * 4 +
                        @as(i64, @intCast(row)) * blck + @as(i64, @intCast(v)) * 4);
                    inline for (0..4) |lane| {
                        y[@intCast(i)].qs[base + lane] = @intCast(vi[lane]);
                    }
                }
            }
        }
    }
}

pub export fn ggml_quantize_mat_q8_0_4x4(x: [*]const f32, vy: *anyopaque, k: i64) callconv(.c) void {
    quantizeMat(4, x, vy, k);
}

pub export fn ggml_quantize_mat_q8_0_4x8(x: [*]const f32, vy: *anyopaque, k: i64) callconv(.c) void {
    quantizeMat(8, x, vy, k);
}

comptime {
    _ = std;
    _ = c;
}
