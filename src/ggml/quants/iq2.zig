//! Quantizing to the 2-bit codebook formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 3270-3678. Each function names the C function it replaces and the line it
//! began at.
//!
//! # Same search as `iq3.zig`, wider groups
//!
//! Eight weights per codebook entry rather than four, two bits per digit
//! rather than three. The sign-splitting, the even-flip rule, and the scale
//! search are all the same shape -- see `iq3.zig` for the explanation.
//!
//! # These require an importance matrix
//!
//! `iq2_xxs` and `iq2_xs` assert it: `ggml_quantize_requires_imatrix` names
//! them, and the fit has nothing to work with otherwise. At two bits per
//! weight there is no accuracy to spare on a uniform weighting.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const codebook = @import("codebook.zig");
const h = @import("helpers.zig");
const c = impl.c;

const fp16 = impl.fp32ToFp16;
const QK_K = c.QK_K;

/// Ports `iq2_find_best_neighbour` (ggml-quants.c:3270 @c1d0e7a00).
///
/// `iq3.zig`'s twin, over 8-element entries.
fn findBestNeighbour(
    neighbours: [*]const u16,
    grid: [*]const u64,
    xval: [*]const f32,
    weight: [*]const f32,
    scale: f32,
    l_out: [*]i8,
) i32 {
    const num_neighbors = neighbours[0];
    impl.assert(num_neighbors > 0, "num_neighbors > 0");

    var best_d2: f32 = std.math.floatMax(f32);
    var grid_index: i32 = -1;
    for (1..num_neighbors + 1) |j| {
        const pg: *const [8]i8 = @ptrCast(&grid[neighbours[j]]);
        var d2: f32 = 0;
        for (0..8) |i| {
            const q: f32 = @floatFromInt(pg[i]);
            const diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    impl.assert(grid_index >= 0, "grid_index >= 0");

    const pg: *const [8]i8 = @ptrCast(&grid[@intCast(grid_index)]);
    for (0..8) |i| l_out[i] = @intCast(@divTrunc(pg[i] - 1, 2));
    return grid_index;
}

/// Splits a group of 32 into magnitudes and sign bytes, forcing an even number
/// of negations per group of eight.
///
/// Shared by both formats here and identical to `iq3.zig`'s. See that file for
/// why the count must be even.
fn splitSigns(xb: [*]const f32, weight: *const [32]f32, xval: *[32]f32, block_signs: *[4]u8) void {
    for (0..4) |k| {
        var nflip: usize = 0;
        var s: u8 = 0;
        for (0..8) |i| {
            if (xb[8 * k + i] >= 0) {
                xval[8 * k + i] = xb[8 * k + i];
            } else {
                xval[8 * k + i] = -xb[8 * k + i];
                nflip += 1;
                s |= @as(u8, 1) << @intCast(i);
            }
        }
        if (nflip % 2 != 0) {
            var imin: usize = 0;
            var min = weight[8 * k] * xb[8 * k] * xb[8 * k];
            for (1..8) |i| {
                const ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                if (ax < min) {
                    min = ax;
                    imin = i;
                }
            }
            xval[8 * k + imin] = -xval[8 * k + imin];
            s ^= @as(u8, 1) << @intCast(imin);
        }
        block_signs[k] = s & 127;
    }
}

/// Ports `quantize_row_iq2_xxs_impl` (ggml-quants.c:3294 @c1d0e7a00).
fn iq2XxsImpl(x: [*]const f32, vy: ?*anyopaque, n: i64, quant_weights: [*]const f32) void {
    const t = c.GGML_TYPE_IQ2_XXS;
    const kgrid = codebook.iq2Grid(t);
    const kmap = codebook.iq2Map(t);
    const kneighbors = codebook.iq2Neighbours(t);

    std.debug.assert(@rem(n, QK_K) == 0);
    const k_max_q = 3;
    const nbl: usize = @intCast(@divExact(n, QK_K));
    const y: [*]blocks.IQ2_XXS = @ptrCast(@alignCast(vy.?));

    var scales: [QK_K / 32]f32 = undefined;
    var weight: [32]f32 = undefined;
    var xval: [32]f32 = undefined;
    var l_buf: [32]i8 = undefined;
    var l_aux: [32]i8 = undefined;
    var waux: [32]f32 = undefined;
    var block_signs: [4]u8 = undefined;
    var q2: [2 * (QK_K / 32)]u32 = undefined;

    for (0..nbl) |ibl| {
        y[ibl].d = fp16(0.0);
        @memset(&q2, 0);
        var max_scale: f32 = 0;
        const xbl = x + QK_K * ibl;
        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        // Note: no 2x here, where iq3_xxs has one.
        const sigma2 = sumx2 / QK_K;

        for (0..QK_K / 32) |ib| {
            const xb = xbl + 32 * ib;
            const qw = quant_weights + QK_K * ibl + 32 * ib;
            for (0..32) |i| weight[i] = qw[i] * @sqrt(sigma2 + xb[i] * xb[i]);
            for (0..32) |i| waux[i] = @sqrt(weight[i]);

            splitSigns(xb, &weight, &xval, &block_signs);

            var max = xval[0];
            for (1..32) |i| max = @max(max, xval[i]);
            if (max < h.group_max_eps) {
                scales[ib] = 0;
                @memset(&l_buf, 0);
                continue;
            }

            // Unlike iq3_xxs, the starting scale comes from a proper fit
            // rather than max/(2*kMaxQ-1).
            var scale = h.makeQpQuants(32, k_max_q + 1, &xval, @ptrCast(&l_buf), &weight);
            const eff_max = scale * k_max_q;
            if (eff_max <= 0) {
                scales[ib] = 0;
                @memset(&l_buf, 0);
                continue;
            }

            var best: f32 = 0;
            var is: i32 = -6;
            while (is <= 6) : (is += 1) {
                const id = (2 * k_max_q - 1 + @as(f32, @floatFromInt(is)) * 0.1) / eff_max;
                const this_scale = 1 / id;
                for (0..4) |k| {
                    for (0..8) |i| {
                        const l = h.nearestInt(0.5 * (id * xval[8 * k + i] - 1));
                        l_aux[8 * k + i] = @intCast(@max(0, @min(k_max_q - 1, l)));
                    }
                    var u: u16 = 0;
                    for (0..8) |i| u |= @as(u16, @intCast(l_aux[8 * k + i])) << @intCast(2 * i);
                    if (kmap[u] < 0) {
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        _ = findBestNeighbour(neighbours, kgrid, xval[8 * k ..].ptr, waux[8 * k ..].ptr, this_scale, l_aux[8 * k ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..32) |i| {
                    const w = weight[i];
                    const q: f32 = @floatFromInt(2 * @as(i32, l_aux[i]) + 1);
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 and sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    @memcpy(&l_buf, &l_aux);
                }
            }

            if (scale > 0) {
                const id = 1 / scale;
                for (0..4) |k| {
                    var u: u16 = 0;
                    for (0..8) |i| {
                        var l = h.nearestInt(0.5 * (id * xval[8 * k + i] - 1));
                        l = @max(0, @min(k_max_q - 1, l));
                        u |= @as(u16, @intCast(l)) << @intCast(2 * i);
                    }
                    var grid_index = kmap[u];
                    if (grid_index < 0) {
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        grid_index = findBestNeighbour(neighbours, kgrid, xval[8 * k ..].ptr, waux[8 * k ..].ptr, scale, l_buf[8 * k ..].ptr);
                    }
                    const pg: *const [8]i8 = @ptrCast(&kgrid[@intCast(grid_index)]);
                    for (0..8) |i| l_buf[8 * k + i] = @intCast(@divTrunc(pg[i] - 1, 2));
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..32) |i| {
                    const w = weight[i];
                    const q: f32 = @floatFromInt(2 * @as(i32, l_buf[i]) + 1);
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }

            if (scale < 0) {
                scale = -scale;
                for (0..4) |k| block_signs[k] = (~block_signs[k]) & 127;
            }

            for (0..4) |k| {
                var u: u16 = 0;
                for (0..8) |i| u |= @as(u16, @intCast(l_buf[8 * k + i])) << @intCast(2 * i);
                const grid_index = kmap[u];
                if (grid_index < 0) impl.abort("fatal error: point not on grid");
                q2[2 * ib + 0] |= @as(u32, @intCast(grid_index)) << @intCast(8 * k);
                q2[2 * ib + 1] |= @as(u32, block_signs[k]) << @intCast(7 * k);
            }
            impl.assert(scale >= 0, "scale >= 0");
            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) {
            @memset(std.mem.sliceAsBytes(y[ibl].qs[0..]), 0);
            continue;
        }

        const d = max_scale / 31;
        y[ibl].d = fp16(d);
        const id = 1 / d;
        for (0..QK_K / 32) |ib| {
            var l = h.nearestInt(0.5 * (id * scales[ib] - 1));
            l = @max(0, @min(15, l));
            q2[2 * ib + 1] |= @as(u32, @intCast(l)) << 28;
        }
        @memcpy(std.mem.sliceAsBytes(y[ibl].qs[0..]), std.mem.sliceAsBytes(q2[0..])[0 .. QK_K / 4]);
    }
}

/// Ports `quantize_iq2_xxs` (ggml-quants.c:3652 @c1d0e7a00).
pub export fn quantize_iq2_xxs(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    impl.assert(@rem(n_per_row, QK_K) == 0, "n_per_row%QK_K == 0");
    impl.assert(quant_weights != null, "missing quantization weights");
    const nblock: usize = @intCast(@divExact(n_per_row, QK_K));
    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;
    for (0..@intCast(nrow)) |_| {
        iq2XxsImpl(s, qrow, n_per_row, quant_weights);
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ2_XXS);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ2_XXS);
}

/// Ports `quantize_row_iq2_xs_impl` (ggml-quants.c:3472 @c1d0e7a00).
///
/// Half the group size of `iq2_xxs` -- 16 weights per scale rather than 32 --
/// so the scales are finer and stored in their own array rather than packed
/// into the top of the quant words. It also tracks which groups landed on the
/// grid, and only re-snaps those, where `iq2_xxs` re-snaps all of them.
fn iq2XsImpl(x: [*]const f32, vy: ?*anyopaque, n: i64, quant_weights: [*]const f32) void {
    const t = c.GGML_TYPE_IQ2_XS;
    const kgrid = codebook.iq2Grid(t);
    const kmap = codebook.iq2Map(t);
    const kneighbors = codebook.iq2Neighbours(t);

    std.debug.assert(@rem(n, QK_K) == 0);
    const k_max_q = 3;
    const nbl: usize = @intCast(@divExact(n, QK_K));
    const y: [*]blocks.IQ2_XS = @ptrCast(@alignCast(vy.?));

    var scales: [QK_K / 16]f32 = undefined;
    var weight: [16]f32 = undefined;
    var xval: [16]f32 = undefined;
    var l_buf: [16]i8 = undefined;
    var l_aux: [16]i8 = undefined;
    var waux: [16]f32 = undefined;
    var is_on_grid: [2]bool = undefined;
    var is_on_grid_aux: [2]bool = undefined;
    var block_signs: [2]u8 = undefined;
    var q2: [2 * (QK_K / 16)]u16 = undefined;

    for (0..nbl) |ibl| {
        y[ibl].d = fp16(0.0);
        @memset(&q2, 0);
        @memset(&y[ibl].scales, 0);

        var max_scale: f32 = 0;
        const xbl = x + QK_K * ibl;
        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = sumx2 / QK_K;

        for (0..QK_K / 16) |ib| {
            const xb = xbl + 16 * ib;
            const qw = quant_weights + QK_K * ibl + 16 * ib;
            for (0..16) |i| weight[i] = qw[i] * @sqrt(sigma2 + xb[i] * xb[i]);
            for (0..16) |i| waux[i] = @sqrt(weight[i]);

            // Two groups of eight rather than four.
            for (0..2) |k| {
                var nflip: usize = 0;
                var s: u8 = 0;
                for (0..8) |i| {
                    if (xb[8 * k + i] >= 0) {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        nflip += 1;
                        s |= @as(u8, 1) << @intCast(i);
                    }
                }
                if (nflip % 2 != 0) {
                    var imin: usize = 0;
                    var min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for (1..8) |i| {
                        const ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= @as(u8, 1) << @intCast(imin);
                }
                block_signs[k] = s & 127;
            }

            var max = xval[0];
            for (1..16) |i| max = @max(max, xval[i]);
            @memset(&l_buf, 0);
            if (max < h.group_max_eps) {
                scales[ib] = 0;
                continue;
            }

            var scale = max / (2 * k_max_q - 1);
            var best: f32 = 0;
            is_on_grid[0] = true;
            is_on_grid[1] = true;

            var is: i32 = -9;
            while (is <= 9) : (is += 1) {
                const id = (2 * k_max_q - 1 + @as(f32, @floatFromInt(is)) * 0.1) / max;
                const this_scale = 1 / id;
                for (0..2) |k| {
                    for (0..8) |i| {
                        const l = h.nearestInt(0.5 * (id * xval[8 * k + i] - 1));
                        l_aux[8 * k + i] = @intCast(@max(0, @min(k_max_q - 1, l)));
                    }
                    var u: u16 = 0;
                    for (0..8) |i| u |= @as(u16, @intCast(l_aux[8 * k + i])) << @intCast(2 * i);
                    is_on_grid_aux[k] = true;
                    if (kmap[u] < 0) {
                        is_on_grid_aux[k] = false;
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        _ = findBestNeighbour(neighbours, kgrid, xval[8 * k ..].ptr, waux[8 * k ..].ptr, this_scale, l_aux[8 * k ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..16) |i| {
                    const w = weight[i];
                    const q: f32 = @floatFromInt(2 * @as(i32, l_aux[i]) + 1);
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 and sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for (0..16) |i| l_buf[i] = l_aux[i];
                    for (0..2) |k| is_on_grid[k] = is_on_grid_aux[k];
                }
            }

            var n_not_ongrid: usize = 0;
            for (0..2) |k| {
                if (!is_on_grid[k]) n_not_ongrid += 1;
            }
            if (n_not_ongrid > 0 and scale > 0) {
                const id = 1 / scale;
                for (0..2) |k| {
                    if (is_on_grid[k]) continue;
                    var u: u16 = 0;
                    for (0..8) |i| {
                        var l = h.nearestInt(0.5 * (id * xval[8 * k + i] - 1));
                        l = @max(0, @min(k_max_q - 1, l));
                        // Written back here, unlike iq2_xxs, which relies on
                        // the neighbour search to fill L.
                        l_buf[8 * k + i] = @intCast(l);
                        u |= @as(u16, @intCast(l)) << @intCast(2 * i);
                    }
                    if (kmap[u] < 0) {
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        _ = findBestNeighbour(neighbours, kgrid, xval[8 * k ..].ptr, waux[8 * k ..].ptr, scale, l_buf[8 * k ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..16) |i| {
                    const w = weight[i];
                    const q: f32 = @floatFromInt(2 * @as(i32, l_buf[i]) + 1);
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }

            if (scale < 0) {
                scale = -scale;
                for (0..2) |k| block_signs[k] = (~block_signs[k]) & 127;
            }

            for (0..2) |k| {
                var u: u16 = 0;
                for (0..8) |i| u |= @as(u16, @intCast(l_buf[8 * k + i])) << @intCast(2 * i);
                const grid_index = kmap[u];
                if (grid_index < 0) impl.abort("fatal error: point not on grid");
                // Nine bits of index, seven of sign, in one u16.
                q2[2 * ib + k] = @as(u16, @intCast(grid_index)) | (@as(u16, block_signs[k]) << 9);
            }
            impl.assert(scale >= 0, "scale >= 0");
            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) {
            @memset(std.mem.sliceAsBytes(y[ibl].qs[0..]), 0);
            continue;
        }

        const d = max_scale / 31;
        y[ibl].d = fp16(d);
        const id = 1 / d;
        for (0..QK_K / 16) |ib| {
            var l = h.nearestInt(0.5 * (id * scales[ib] - 1));
            l = @max(0, @min(15, l));
            if (ib % 2 == 0) {
                y[ibl].scales[ib / 2] = @intCast(l);
            } else {
                y[ibl].scales[ib / 2] |= @as(u8, @intCast(l)) << 4;
            }
        }
        @memcpy(std.mem.sliceAsBytes(y[ibl].qs[0..]), std.mem.sliceAsBytes(q2[0..])[0 .. QK_K / 4]);
    }
}

/// Ports `quantize_iq2_xs` (ggml-quants.c:3664 @c1d0e7a00).
pub export fn quantize_iq2_xs(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    impl.assert(@rem(n_per_row, QK_K) == 0, "n_per_row%QK_K == 0");
    impl.assert(quant_weights != null, "missing quantization weights");
    const nblock: usize = @intCast(@divExact(n_per_row, QK_K));
    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;
    for (0..@intCast(nrow)) |_| {
        iq2XsImpl(s, qrow, n_per_row, quant_weights);
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ2_XS);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ2_XS);
}

/// Ports `quantize_row_iq2_s_impl` (ggml-quants.c:5142 @c1d0e7a00).
///
/// The 1024-entry codebook, so ten bits of index: eight in `qs` and two in
/// `qh`. Signs get a whole byte each, stored in the tail of `qs`, so the
/// even-flip rule does not apply here either.
///
/// Note the fallback weighting when there is no imatrix: `0.25*sigma2 + x*x`,
/// where every other format here uses `x*x` alone.
fn iq2SImpl(x: [*]const f32, vy: ?*anyopaque, n: i64, quant_weights: ?[*]const f32) void {
    const t = c.GGML_TYPE_IQ2_S;
    const kgrid = codebook.iq2Grid(t);
    const kmap = codebook.iq2Map(t);
    const kneighbors = codebook.iq2Neighbours(t);

    std.debug.assert(@rem(n, QK_K) == 0);
    const k_max_q = 3;
    const nbl: usize = @intCast(@divExact(n, QK_K));
    const y: [*]blocks.IQ2_S = @ptrCast(@alignCast(vy.?));

    var scales: [QK_K / 16]f32 = undefined;
    var weight: [16]f32 = undefined;
    var xval: [16]f32 = undefined;
    var l_buf: [16]i8 = undefined;
    var l_aux: [16]i8 = undefined;
    var waux: [16]f32 = undefined;
    var is_on_grid: [2]bool = undefined;
    var is_on_grid_aux: [2]bool = undefined;
    var block_signs: [2]u8 = undefined;

    for (0..nbl) |ibl| {
        @memset(std.mem.asBytes(&y[ibl]), 0);
        y[ibl].d = fp16(0.0);

        var max_scale: f32 = 0;
        const xbl = x + QK_K * ibl;
        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = 2 * sumx2 / QK_K;

        for (0..QK_K / 16) |ib| {
            const xb = xbl + 16 * ib;
            if (quant_weights) |qw_all| {
                const qw = qw_all + QK_K * ibl + 16 * ib;
                for (0..16) |i| weight[i] = qw[i] * @sqrt(sigma2 + xb[i] * xb[i]);
            } else {
                for (0..16) |i| weight[i] = 0.25 * sigma2 + xb[i] * xb[i];
            }
            for (0..16) |i| waux[i] = @sqrt(weight[i]);

            for (0..2) |k| {
                var s: u8 = 0;
                for (0..8) |i| {
                    if (xb[8 * k + i] >= 0) {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        s |= @as(u8, 1) << @intCast(i);
                    }
                }
                block_signs[k] = s;
            }

            var max = xval[0];
            for (1..16) |i| max = @max(max, xval[i]);
            @memset(&l_buf, 0);
            if (max < h.group_max_eps_iq2_s) {
                scales[ib] = 0;
                continue;
            }

            var best: f32 = 0;
            var scale = max / (2 * k_max_q - 1);
            is_on_grid[0] = true;
            is_on_grid[1] = true;

            var is: i32 = -9;
            while (is <= 9) : (is += 1) {
                const id = (2 * k_max_q - 1 + @as(f32, @floatFromInt(is)) * 0.1) / max;
                const this_scale = 1 / id;
                for (0..2) |k| {
                    for (0..8) |i| {
                        const l = h.nearestInt(0.5 * (id * xval[8 * k + i] - 1));
                        l_aux[8 * k + i] = @intCast(@max(0, @min(k_max_q - 1, l)));
                    }
                    var u: u16 = 0;
                    for (0..8) |i| u |= @as(u16, @intCast(l_aux[8 * k + i])) << @intCast(2 * i);
                    is_on_grid_aux[k] = true;
                    if (kmap[u] < 0) {
                        is_on_grid_aux[k] = false;
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        _ = findBestNeighbour(neighbours, kgrid, xval[8 * k ..].ptr, waux[8 * k ..].ptr, this_scale, l_aux[8 * k ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..16) |i| {
                    const w = weight[i];
                    const q: f32 = @floatFromInt(2 * @as(i32, l_aux[i]) + 1);
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 and sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for (0..16) |i| l_buf[i] = l_aux[i];
                    for (0..2) |k| is_on_grid[k] = is_on_grid_aux[k];
                }
            }

            var n_not_ongrid: usize = 0;
            for (0..2) |k| {
                if (!is_on_grid[k]) n_not_ongrid += 1;
            }
            if (n_not_ongrid > 0 and scale > 0) {
                const id = 1 / scale;
                for (0..2) |k| {
                    if (is_on_grid[k]) continue;
                    var u: u16 = 0;
                    for (0..8) |i| {
                        var l = h.nearestInt(0.5 * (id * xval[8 * k + i] - 1));
                        l = @max(0, @min(k_max_q - 1, l));
                        u |= @as(u16, @intCast(l)) << @intCast(2 * i);
                        l_buf[8 * k + i] = @intCast(l);
                    }
                    if (kmap[u] < 0) {
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        _ = findBestNeighbour(neighbours, kgrid, xval[8 * k ..].ptr, waux[8 * k ..].ptr, scale, l_buf[8 * k ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..16) |i| {
                    const w = weight[i];
                    const q: f32 = @floatFromInt(2 * @as(i32, l_buf[i]) + 1);
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }

            if (scale < 0) {
                scale = -scale;
                for (0..2) |k| block_signs[k] = ~block_signs[k];
            }

            for (0..2) |k| {
                var u: u16 = 0;
                for (0..8) |i| u |= @as(u16, @intCast(l_buf[8 * k + i])) << @intCast(2 * i);
                const grid_index = kmap[u];
                if (grid_index < 0) impl.abort("fatal error: point not on grid");
                const i8_idx = 2 * ib + k;
                y[ibl].qs[i8_idx] = @intCast(grid_index & 255);
                y[ibl].qh[i8_idx / 4] |= @as(u8, @intCast(grid_index >> 8)) << @intCast(2 * (i8_idx % 4));
                // The sign bytes live past the grid indices in the same array.
                y[ibl].qs[QK_K / 8 + i8_idx] = block_signs[k];
            }
            impl.assert(scale >= 0, "scale >= 0");
            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) continue;

        const d = max_scale / 31;
        // Below one, where iq3's fudge factors are above.
        y[ibl].d = fp16(d * 0.9875);
        const id = 1 / d;
        for (0..QK_K / 16) |ib| {
            var l = h.nearestInt(0.5 * (id * scales[ib] - 1));
            l = @max(0, @min(15, l));
            if (ib % 2 == 0) {
                y[ibl].scales[ib / 2] = @intCast(l);
            } else {
                y[ibl].scales[ib / 2] |= @as(u8, @intCast(l)) << 4;
            }
        }
    }
}

/// Ports `quantize_iq2_s` (ggml-quants.c:5311 @c1d0e7a00).
pub export fn quantize_iq2_s(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    impl.assert(@rem(n_per_row, QK_K) == 0, "n_per_row%QK_K == 0");
    const nblock: usize = @intCast(@divExact(n_per_row, QK_K));
    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;
    for (0..@intCast(nrow)) |_| {
        iq2SImpl(s, qrow, n_per_row, if (quant_weights != null) quant_weights else null);
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ2_S);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ2_S);
}

/// Ports `quantize_row_iq2_s_ref` (ggml-quants.c:5323 @c1d0e7a00).
pub export fn quantize_row_iq2_s_ref(x: [*c]const f32, y: [*c]blocks.IQ2_S, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    _ = quantize_iq2_s(x, @ptrCast(y), 1, k, null);
}
