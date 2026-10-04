//! `ggml_quantize_mat_*`: quantize four rows at a time into an interleaved
//! block.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Four functions, two bodies
//!
//! `_4x4` and `_4x8` differ only in `blck_size_interleave` — 4 against 8 —
//! and, for the `q8_K` pair, in one bsum index expression that follows from
//! it. They are written here as two functions over a `comptime` interleave,
//! with the index expression selected the same way.
//!
//! # These are the `_generic` fallbacks
//!
//! `arch/arm/repack.cpp` supplies `ggml_quantize_mat_q8_0_4x4` and `_4x8`
//! without the suffix, and `arch-fallback.h` renames nothing on ARM, so
//! those two shadow the ones here. The `q8_K` pair has no ARM override and
//! **is** the live implementation — note the asymmetry, which is why the
//! exported names differ between the two halves.

const std = @import("std");
const impl = @import("../../impl.zig");
const blocks = @import("blocks.zig");
const helpers = @import("../../quants/helpers.zig");
const convert = @import("../convert.zig");

const c = impl.c;

const QK8_0 = 32;
const QK_K = 256;

/// Ports `ggml_quantize_mat_q8_0_4x4_generic` and
/// `ggml_quantize_mat_q8_0_4x8_generic`
/// (ggml-cpu/repack.cpp:135, 173 @c1d0e7a00).
fn quantizeMatQ80(comptime blck_size_interleave: i32, x: [*]const f32, vy: *anyopaque, k: i64) void {
    impl.assert(QK8_0 == 32, "QK8_0 == 32");
    impl.assert(@rem(k, QK8_0) == 0, "k % QK8_0 == 0");
    const nb: i64 = @divTrunc(k, QK8_0);

    const y: [*]blocks.block_q8_0x4 = @ptrCast(@alignCast(vy));

    // scalar
    var srcv: [4][QK8_0]f32 = undefined;
    var id: [4]f32 = undefined;

    var i: i64 = 0;
    while (i < nb) : (i += 1) {
        for (0..4) |row_iter| {
            var amax: f32 = 0.0; // absolute max

            for (0..QK8_0) |j| {
                srcv[row_iter][j] = x[@intCast(@as(i64, @intCast(row_iter)) * k + i * QK8_0 + @as(i64, @intCast(j)))];
                amax = @max(amax, @abs(srcv[row_iter][j]));
            }

            const d = amax / ((1 << 7) - 1);
            id[row_iter] = if (d != 0.0) 1.0 / d else 0.0;

            y[@intCast(i)].d[row_iter] = convert.cpuFp32ToFp16(d);
        }

        for (0..QK8_0 * 4) |j| {
            const jj: i32 = @intCast(j);
            var src_offset = @divTrunc(jj, 4 * blck_size_interleave) * blck_size_interleave;
            const src_id = @divTrunc(@rem(jj, 4 * blck_size_interleave), blck_size_interleave);
            src_offset += @rem(jj, blck_size_interleave);

            const x0 = srcv[@intCast(src_id)][@intCast(src_offset)] * id[@intCast(src_id)];
            y[@intCast(i)].qs[j] = @intFromFloat(roundf(x0));
        }
    }
}

extern fn roundf(x: f32) f32;

/// Ports `ggml_quantize_mat_q8_K_4x4_generic` and
/// `ggml_quantize_mat_q8_K_4x8_generic`
/// (ggml-cpu/repack.cpp:211, 262 @c1d0e7a00).
///
/// Note the max search keeps the **signed** value at the greatest magnitude,
/// not the magnitude, and the scale is `-127/max`. A sign slip there flips
/// every quant in the super-block.
fn quantizeMatQ8K(comptime blck_size_interleave: i32, x: [*]const f32, vy: *anyopaque, k: i64) void {
    impl.assert(QK_K == 256, "QK_K == 256");
    impl.assert(@rem(k, QK_K) == 0, "k % QK_K == 0");
    const nb: i64 = @divTrunc(k, QK_K);

    const y: [*]blocks.block_q8_Kx4 = @ptrCast(@alignCast(vy));

    // scalar
    var srcv: [4][QK_K]f32 = undefined;
    var iscale: [4]f32 = undefined;

    var i: i64 = 0;
    while (i < nb) : (i += 1) {
        for (0..4) |row_iter| {
            var amax: f32 = 0.0; // absolute max
            var max: f32 = 0;

            for (0..QK_K) |j| {
                srcv[row_iter][j] = x[@intCast(@as(i64, @intCast(row_iter)) * k + i * QK_K + @as(i64, @intCast(j)))];
                // Update the maximum value of the corresponding super block
                if (amax < @abs(srcv[row_iter][j])) {
                    amax = @abs(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = if (amax != 0.0) -127.0 / max else 0;
            y[@intCast(i)].d[row_iter] = if (amax != 0.0) 1 / iscale[row_iter] else 0;
        }

        for (0..QK_K / 4) |j| {
            y[@intCast(i)].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of `blck_size_interleave`
        // bytes from corresponding super blocks. Bsums values are interleaved
        // in sequence of four bsums from each super block taken for
        // interleaving, i.e. first four bsums from the first super block,
        // followed by first four bsums from second super block and so on.
        for (0..QK_K * 4) |j| {
            const jj: i32 = @intCast(j);
            var src_offset = @divTrunc(jj, 4 * blck_size_interleave) * blck_size_interleave;
            const src_id = @divTrunc(@rem(jj, 4 * blck_size_interleave), blck_size_interleave);
            src_offset += @rem(jj, blck_size_interleave);

            // The mask follows the interleave: 15 and `>> 2` at 4, 31 and
            // `>> 3` at 8.
            const index: usize = @intCast(if (blck_size_interleave == 4)
                (((jj & 15) >> 2) << 2) + ((jj >> 8) << 4) + ((jj >> 6) & 3)
            else
                (((jj & 31) >> 3) << 2) + ((jj >> 8) << 4) + ((jj >> 6) & 3));

            const x0 = srcv[@intCast(src_id)][@intCast(src_offset)] * iscale[@intCast(src_id)];
            y[@intCast(i)].qs[j] = @intCast(helpers.nearestInt(x0));
            y[@intCast(i)].bsums[index] += y[@intCast(i)].qs[j];
        }
    }
}

pub export fn ggml_quantize_mat_q8_0_4x4_generic(x: [*]const f32, vy: *anyopaque, k: i64) callconv(.c) void {
    quantizeMatQ80(4, x, vy, k);
}

pub export fn ggml_quantize_mat_q8_0_4x8_generic(x: [*]const f32, vy: *anyopaque, k: i64) callconv(.c) void {
    quantizeMatQ80(8, x, vy, k);
}

/// No ARM override exists for the `q8_K` pair, so these two carry the live
/// name rather than a `_generic` one.
pub export fn ggml_quantize_mat_q8_K_4x4(x: [*]const f32, vy: *anyopaque, k: i64) callconv(.c) void {
    quantizeMatQ8K(4, x, vy, k);
}

pub export fn ggml_quantize_mat_q8_K_4x8(x: [*]const f32, vy: *anyopaque, k: i64) callconv(.c) void {
    quantizeMatQ8K(8, x, vy, k);
}
