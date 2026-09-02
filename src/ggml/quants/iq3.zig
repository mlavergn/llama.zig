//! Quantizing to the 3-bit codebook formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 3914-4170 for `iq3_xxs`. Each function names the C function it replaces and
//! the line it began at.
//!
//! # How a codebook quantizer searches
//!
//! For each group of 4 weights:
//!
//! 1. **Take absolute values and record the signs.** The codebook holds only
//!    positive vectors, so the sign pattern is stored separately -- seven bits
//!    for eight elements, because the eighth is implied.
//! 2. **Quantize to a 4-digit base-8 pattern** and look it up. A hit is the
//!    codebook index; a miss lands in the neighbour list `codebook.zig` built.
//! 3. **Search 31 candidate scales** around the naive one, keeping whichever
//!    maximises `sumqx^2 / sumq2` -- the same ratio the K-quants use.
//!
//! # The odd-flip rule
//!
//! Only seven sign bits are stored, so the number of negated elements in each
//! group of eight must be **even**. When it comes out odd the quantizer flips
//! whichever element contributes least to the weighted error -- deliberately
//! making one weight wrong to keep the encoding representable.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const codebook = @import("codebook.zig");
const h = @import("helpers.zig");
const c = impl.c;

const fp16 = impl.fp32ToFp16;
const QK_K = c.QK_K;

/// Ports `iq3_find_best_neighbour` (ggml-quants.c:3914 @c1d0e7a00).
///
/// Picks the weighted-closest entry from a precomputed neighbour list, and
/// writes its digits back into `l_out`.
fn findBestNeighbour(
    neighbours: [*]const u16,
    grid: [*]const u32,
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
        const pg: *const [4]i8 = @ptrCast(&grid[neighbours[j]]);
        var d2: f32 = 0;
        for (0..4) |i| {
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

    const pg: *const [4]i8 = @ptrCast(&grid[@intCast(grid_index)]);
    for (0..4) |i| l_out[i] = @intCast(@divTrunc(pg[i] - 1, 2));
    return grid_index;
}

/// Ports `quantize_row_iq3_xxs_impl` (ggml-quants.c:3938 @c1d0e7a00).
///
/// Handles both `iq3_xxs` (grid 256) and the grid-512 shape, which is why it
/// takes the block stride rather than a block type: the two write different
/// structs but run the identical search.
fn iq3XxsImpl(grid_size: c_int, x: [*]const f32, vy: ?*anyopaque, n: i64, quant_weights: ?[*]const f32) void {
    const kgrid = codebook.iq3Grid(grid_size);
    const kmap = codebook.iq3Map(grid_size);
    const kneighbors = codebook.iq3Neighbours(grid_size);

    std.debug.assert(@rem(n, QK_K) == 0);
    const k_max_q = 8;
    const nbl: usize = @intCast(@divExact(n, QK_K));

    // The two destination layouts share a prefix -- an f16 scale then the
    // quant bytes -- so the walk is expressed as a stride rather than a type.
    const block_size: usize = if (grid_size == 256) @sizeOf(blocks.IQ3_XXS) else @sizeOf(blocks.IQ3_S);
    const quant_size = block_size - @sizeOf(blocks.Half);

    var dh: [*]blocks.Half = @ptrCast(@alignCast(vy.?));
    var qs: [*]u8 = @as([*]u8, @ptrCast(@alignCast(vy.?))) + @sizeOf(blocks.Half);

    var scales: [QK_K / 32]f32 = undefined;
    var weight: [32]f32 = undefined;
    var xval: [32]f32 = undefined;
    var l_buf: [32]i8 = undefined;
    var l_aux: [32]i8 = undefined;
    var waux: [32]f32 = undefined;
    var is_on_grid: [8]bool = undefined;
    var is_on_grid_aux: [8]bool = undefined;
    var block_signs: [8]u8 = undefined;
    var q3: [3 * (QK_K / 8) + QK_K / 32]u8 = undefined;

    for (0..nbl) |ibl| {
        dh[0] = fp16(0.0);
        @memset(&q3, 0);
        // Two views into the same buffer: the scale-and-sign words start where
        // the grid indices end, and `qh` overlaps the same tail for grid-512.
        const scales_and_signs: [*]u32 = @ptrCast(@alignCast(&q3[QK_K / 4]));
        const qh: [*]u8 = @ptrCast(&q3[3 * (QK_K / 8)]);

        var max_scale: f32 = 0;
        const xbl = x + QK_K * ibl;
        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = 2 * sumx2 / QK_K;

        for (0..QK_K / 32) |ib| {
            const xb = xbl + 32 * ib;
            if (quant_weights) |qw_all| {
                const qw = qw_all + QK_K * ibl + 32 * ib;
                for (0..32) |i| weight[i] = qw[i] * @sqrt(sigma2 + xb[i] * xb[i]);
            } else {
                for (0..32) |i| weight[i] = xb[i] * xb[i];
            }
            for (0..32) |i| waux[i] = @sqrt(weight[i]);

            // Split off the signs, forcing an even count per group of eight.
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
                    // Flip the element whose weighted magnitude is smallest:
                    // the cheapest place to be deliberately wrong.
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
            for (1..32) |i| max = @max(max, xval[i]);
            @memset(&l_buf, 0);
            if (max < h.group_max_eps_iq3_xxs) {
                scales[ib] = 0;
                continue;
            }

            var best: f32 = 0;
            var scale = max / (2 * k_max_q - 1);
            for (0..8) |k| is_on_grid[k] = true;

            var is: i32 = -15;
            while (is <= 15) : (is += 1) {
                const id = (2 * k_max_q - 1 + @as(f32, @floatFromInt(is)) * 0.2) / max;
                const this_scale = 1 / id;
                for (0..8) |k| {
                    for (0..4) |i| {
                        // The 0.5*(id*x - 1) maps onto the odd values the
                        // codebook stores: level l represents 2l+1.
                        const l = h.nearestInt(0.5 * (id * xval[4 * k + i] - 1));
                        l_aux[4 * k + i] = @intCast(@max(0, @min(k_max_q - 1, l)));
                    }
                    var u: u16 = 0;
                    for (0..4) |i| u |= @as(u16, @intCast(l_aux[4 * k + i])) << @intCast(3 * i);
                    var grid_index = kmap[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        grid_index = findBestNeighbour(neighbours, kgrid, xval[4 * k ..].ptr, waux[4 * k ..].ptr, this_scale, l_aux[4 * k ..].ptr);
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
                    for (0..32) |i| l_buf[i] = l_aux[i];
                    for (0..8) |k| is_on_grid[k] = is_on_grid_aux[k];
                }
            }

            // Any group still off the grid is snapped to its best neighbour,
            // and the scale refitted against the result.
            var n_not_ongrid: usize = 0;
            for (0..8) |k| {
                if (!is_on_grid[k]) n_not_ongrid += 1;
            }
            if (n_not_ongrid > 0 and scale > 0) {
                const id = 1 / scale;
                for (0..8) |k| {
                    if (is_on_grid[k]) continue;
                    var u: u16 = 0;
                    for (0..4) |i| {
                        var l = h.nearestInt(0.5 * (id * xval[4 * k + i] - 1));
                        l = @max(0, @min(k_max_q - 1, l));
                        u |= @as(u16, @intCast(l)) << @intCast(3 * i);
                    }
                    var grid_index = kmap[u];
                    if (grid_index < 0) {
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        grid_index = findBestNeighbour(neighbours, kgrid, xval[4 * k ..].ptr, waux[4 * k ..].ptr, scale, l_buf[4 * k ..].ptr);
                    }
                    const pg: *const [4]i8 = @ptrCast(&kgrid[@intCast(grid_index)]);
                    for (0..4) |i| l_buf[4 * k + i] = @intCast(@divTrunc(pg[i] - 1, 2));
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
                // The C notes this should never happen. The scale is stored
                // unsigned, so a negative one is absorbed by inverting every
                // sign instead.
                scale = -scale;
                for (0..4) |k| block_signs[k] = (~block_signs[k]) & 127;
            }

            for (0..8) |k| {
                var u: u16 = 0;
                for (0..4) |i| u |= @as(u16, @intCast(l_buf[4 * k + i])) << @intCast(3 * i);
                const grid_index = kmap[u];
                if (grid_index < 0) impl.abort("fatal error: point not on grid");
                if (grid_size == 256) {
                    q3[8 * ib + k] = @intCast(grid_index);
                } else {
                    q3[8 * ib + k] = @intCast(grid_index & 255);
                    qh[ib] |= @as(u8, @intCast(grid_index >> 8)) << @intCast(k);
                }
            }
            scales_and_signs[ib] = @as(u32, block_signs[0]) |
                (@as(u32, block_signs[1]) << 7) |
                (@as(u32, block_signs[2]) << 14) |
                (@as(u32, block_signs[3]) << 21);
            impl.assert(scale >= 0, "scale >= 0");
            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) {
            @memset(qs[0..quant_size], 0);
            dh = @ptrFromInt(@intFromPtr(dh) + block_size);
            qs += block_size;
            continue;
        }

        const d = max_scale / 31;
        // The C's comment: "small improvement via this fudge factor".
        dh[0] = fp16(d * 1.0125);
        const id = 1 / d;
        for (0..QK_K / 32) |ib| {
            var l = h.nearestInt(0.5 * (id * scales[ib] - 1));
            l = @max(0, @min(15, l));
            scales_and_signs[ib] |= @as(u32, @intCast(l)) << 28;
        }
        @memcpy(qs[0..quant_size], q3[0..quant_size]);
        dh = @ptrFromInt(@intFromPtr(dh) + block_size);
        qs += block_size;
    }
}

/// Ports `quantize_iq3_xxs` (ggml-quants.c:4152 @c1d0e7a00).
pub export fn quantize_iq3_xxs(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    impl.assert(@rem(n_per_row, QK_K) == 0, "n_per_row%QK_K == 0");
    const nblock: usize = @intCast(@divExact(n_per_row, QK_K));
    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;
    for (0..@intCast(nrow)) |_| {
        iq3XxsImpl(256, s, qrow, n_per_row, if (quant_weights != null) quant_weights else null);
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ3_XXS);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ3_XXS);
}

/// Ports `quantize_row_iq3_xxs_ref` (ggml-quants.c:4164 @c1d0e7a00).
pub export fn quantize_row_iq3_xxs_ref(x: [*c]const f32, y: [*c]blocks.IQ3_XXS, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    iq3XxsImpl(256, x, @ptrCast(y), k, null);
}

// -----------------------------------------------------------------------------
// iq3_s
//
// The same lattice search over a 512-entry codebook rather than 256, so the
// index needs nine bits: eight in `qs` and the ninth gathered into `qh`. Signs
// get a byte each here rather than being packed seven-to-a-word, and the scale
// is stored per pair of sub-blocks.

/// Ports `IQ3S_BLOCK_SIZE` (ggml-quants.c:4352 @c1d0e7a00).
const iq3s_block_size = 32;

/// Ports `quantize_row_iq3_s_impl` (ggml-quants.c:4169 @c1d0e7a00).
///
/// Two differences from `iq3XxsImpl` worth naming, because both look like
/// mistakes and are not:
///
/// - **Signs are not forced even.** `iq3_s` stores a whole byte per group of
///   eight, so all eight bits are available and the odd-flip rule that
///   `iq3_xxs` needs does not apply.
/// - **The re-snap pass ignores `is_on_grid`.** The C's `continue` on that
///   flag is commented out, so every group is re-snapped once any group is off
///   the grid. Reinstating it would change the output.
fn iq3SImpl(block_size: usize, x: [*]const f32, vy: ?*anyopaque, n: i64, quant_weights: ?[*]const f32) void {
    const kgrid = codebook.iq3Grid(512);
    const kmap = codebook.iq3Map(512);
    const kneighbors = codebook.iq3Neighbours(512);

    std.debug.assert(@rem(n, QK_K) == 0);
    const k_max_q = 8;
    const nbl: usize = @intCast(@divExact(n, QK_K));
    const y: [*]blocks.IQ3_S = @ptrCast(@alignCast(vy.?));

    const bs4 = block_size / 4;
    const bs8 = block_size / 8;

    var scales: [QK_K / iq3s_block_size]f32 = undefined;
    var weight: [iq3s_block_size]f32 = undefined;
    var xval: [iq3s_block_size]f32 = undefined;
    var l_buf: [iq3s_block_size]i8 = undefined;
    var l_aux: [iq3s_block_size]i8 = undefined;
    var waux: [iq3s_block_size]f32 = undefined;
    var is_on_grid: [iq3s_block_size / 4]bool = undefined;
    var is_on_grid_aux: [iq3s_block_size / 4]bool = undefined;
    var block_signs: [iq3s_block_size / 8]u8 = undefined;

    for (0..nbl) |ibl| {
        @memset(std.mem.asBytes(&y[ibl]), 0);
        y[ibl].d = fp16(0.0);
        var qs: [*]u8 = &y[ibl].qs;
        const qh: [*]u8 = &y[ibl].qh;
        var signs: [*]u8 = &y[ibl].signs;

        var max_scale: f32 = 0;
        const xbl = x + QK_K * ibl;
        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = 2 * sumx2 / QK_K;

        for (0..@divExact(@as(usize, QK_K), block_size)) |ib| {
            const xb = xbl + block_size * ib;
            if (quant_weights) |qw_all| {
                const qw = qw_all + QK_K * ibl + block_size * ib;
                for (0..block_size) |i| weight[i] = qw[i] * @sqrt(sigma2 + xb[i] * xb[i]);
            } else {
                for (0..block_size) |i| weight[i] = xb[i] * xb[i];
            }
            for (0..block_size) |i| waux[i] = @sqrt(weight[i]);

            // All eight sign bits are stored, so no parity fix-up.
            for (0..bs8) |k| {
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
            for (1..block_size) |i| max = @max(max, xval[i]);
            @memset(&l_buf, 0);
            // A plain `!max`, not the epsilon the other formats use.
            if (max == 0) {
                scales[ib] = 0;
                continue;
            }

            var best: f32 = 0;
            var scale = max / (2 * k_max_q - 1);
            for (0..bs4) |k| is_on_grid[k] = false;

            var is: i32 = -9;
            while (is <= 9) : (is += 1) {
                const id = (2 * k_max_q - 1 + @as(f32, @floatFromInt(is)) * 0.2) / max;
                const this_scale = 1 / id;
                for (0..bs4) |k| {
                    for (0..4) |i| {
                        const l = h.nearestInt(0.5 * (id * xval[4 * k + i] - 1));
                        l_aux[4 * k + i] = @intCast(@max(0, @min(k_max_q - 1, l)));
                    }
                    var u: u16 = 0;
                    for (0..4) |i| u |= @as(u16, @intCast(l_aux[4 * k + i])) << @intCast(3 * i);
                    is_on_grid_aux[k] = true;
                    if (kmap[u] < 0) {
                        is_on_grid_aux[k] = false;
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        _ = findBestNeighbour(neighbours, kgrid, xval[4 * k ..].ptr, waux[4 * k ..].ptr, this_scale, l_aux[4 * k ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..block_size) |i| {
                    const w = weight[i];
                    const q: f32 = @floatFromInt(2 * @as(i32, l_aux[i]) + 1);
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 and sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for (0..block_size) |i| l_buf[i] = l_aux[i];
                    for (0..bs4) |k| is_on_grid[k] = is_on_grid_aux[k];
                }
            }

            var n_not_ongrid: usize = 0;
            for (0..bs4) |k| {
                if (!is_on_grid[k]) n_not_ongrid += 1;
            }
            if (n_not_ongrid > 0 and scale > 0) {
                const id = 1 / scale;
                for (0..bs4) |k| {
                    // No `if (is_on_grid[k]) continue;` -- commented out in
                    // the C, and reinstating it changes the output.
                    var u: u16 = 0;
                    for (0..4) |i| {
                        var l = h.nearestInt(0.5 * (id * xval[4 * k + i] - 1));
                        l = @max(0, @min(k_max_q - 1, l));
                        u |= @as(u16, @intCast(l)) << @intCast(3 * i);
                    }
                    var grid_index = kmap[u];
                    if (grid_index < 0) {
                        const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                        grid_index = findBestNeighbour(neighbours, kgrid, xval[4 * k ..].ptr, waux[4 * k ..].ptr, scale, l_buf[4 * k ..].ptr);
                    }
                    const pg: *const [4]i8 = @ptrCast(&kgrid[@intCast(grid_index)]);
                    for (0..4) |i| l_buf[4 * k + i] = @intCast(@divTrunc(pg[i] - 1, 2));
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..block_size) |i| {
                    const w = weight[i];
                    const q: f32 = @floatFromInt(2 * @as(i32, l_buf[i]) + 1);
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }

            if (scale < 0) {
                scale = -scale;
                // A full inversion, not `& 127`: all eight bits are stored.
                for (0..bs8) |k| block_signs[k] = ~block_signs[k];
            }

            for (0..bs4) |k| {
                var u: u16 = 0;
                for (0..4) |i| u |= @as(u16, @intCast(l_buf[4 * k + i])) << @intCast(3 * i);
                const grid_index = kmap[u];
                if (grid_index < 0) impl.abort("fatal error: point not on grid");
                qs[k] = @intCast(grid_index & 255);
                // The ninth bit, indexed across the whole super-block.
                const bit = ib * bs4 + k;
                qh[bit / 8] |= @as(u8, @intCast(grid_index >> 8)) << @intCast(bit % 8);
            }
            qs += bs4;
            for (0..bs8) |k| signs[k] = block_signs[k];
            signs += bs8;

            impl.assert(scale >= 0, "scale >= 0");
            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) continue;

        const d = max_scale / 31;
        // A different fudge factor from iq3_xxs's 1.0125.
        y[ibl].d = fp16(d * 1.033);
        const id = 1 / d;
        var ib: usize = 0;
        while (ib < @divExact(@as(usize, QK_K), block_size)) : (ib += 2) {
            var l1 = h.nearestInt(0.5 * (id * scales[ib + 0] - 1));
            l1 = @max(0, @min(15, l1));
            var l2 = h.nearestInt(0.5 * (id * scales[ib + 1] - 1));
            l2 = @max(0, @min(15, l2));
            y[ibl].scales[ib / 2] = @as(u8, @intCast(l1)) | (@as(u8, @intCast(l2)) << 4);
        }
    }
}

/// Ports `quantize_iq3_s` (ggml-quants.c:4353 @c1d0e7a00).
pub export fn quantize_iq3_s(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    impl.assert(@rem(n_per_row, QK_K) == 0, "n_per_row%QK_K == 0");
    const nblock: usize = @intCast(@divExact(n_per_row, QK_K));
    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;
    for (0..@intCast(nrow)) |_| {
        iq3SImpl(iq3s_block_size, s, qrow, n_per_row, if (quant_weights != null) quant_weights else null);
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ3_S);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ3_S);
}

/// Ports `quantize_row_iq3_s_ref` (ggml-quants.c:4375 @c1d0e7a00).
pub export fn quantize_row_iq3_s_ref(x: [*c]const f32, y: [*c]blocks.IQ3_S, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    _ = quantize_iq3_s(x, @ptrCast(y), 1, k, null);
}
