//! Quantizing to the 1-bit formats, `iq1_s` and `iq1_m`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 4383-4964. Each function names the C function it replaces and the line it
//! began at.
//!
//! # A different search entirely
//!
//! The other i-quants scan candidate scales and quantize against each. At
//! roughly one bit per weight there are only **three** levels -- conceptually
//! -1, 0 and +1 -- and that makes the problem exactly solvable rather than
//! searchable:
//!
//! Sort the block's weights. Any assignment of three levels that is optimal
//! must be *contiguous* in sorted order -- some prefix takes the low level, a
//! middle run takes the middle, the rest take the high. So the optimum is
//! found by trying every pair of split points, which is `O(n^2)` over 32
//! weights and exact, where a scale search would be approximate.
//!
//! Prefix sums of `weight[i]*x[i]` and `weight[i]` make each split's score a
//! constant-time subtraction, which is what keeps the double loop cheap.
//!
//! # The delta, and the two shifts
//!
//! The levels are not -1, 0, +1 but those values shifted by `IQ1S_DELTA`
//! either up (`x_p`) or down (`x_m`). Each block picks whichever shift scores
//! better, and records the choice in a bit of its scale field. That is why a
//! "zero" weight does not dequantize to zero.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const codebook = @import("codebook.zig");
const h = @import("helpers.zig");
const c = impl.c;

const fp16 = impl.fp32ToFp16;
const QK_K = c.QK_K;

/// Ports `IQ1S_BLOCK_SIZE` (ggml-quants.c:4506 @c1d0e7a00).
const iq1s_block_size = 32;
/// Ports `IQ1M_BLOCK_SIZE` (ggml-quants.c:4507 @c1d0e7a00).
const iq1m_block_size = 16;
const ngrid_iq1s = 2048;

/// Ports `iq1_find_best_neighbour2` (ggml-quants.c:4443 @c1d0e7a00).
///
/// Weighted-closest entry, with the grid values mapped through `xg` -- the
/// shifted level table -- rather than used directly.
///
/// The C has a third block after the fallback scan that computes sums and
/// discards them; it exists only to print diagnostics that are themselves
/// commented out. It has no effect and is not reproduced.
fn findBestNeighbour2(
    neighbours: [*]const u16,
    grid: [*]const u64,
    xval: [*]const f32,
    weight: [*]const f32,
    scale: f32,
    xg: *const [3]f32,
    l_out: [*]i8,
    ngrid: usize,
) i32 {
    const num_neighbors = neighbours[0];
    impl.assert(num_neighbors > 0, "num_neighbors > 0");

    var best_score: f32 = std.math.floatMax(f32);
    var grid_index: i32 = -1;
    for (1..num_neighbors + 1) |j| {
        const pg: *const [8]i8 = @ptrCast(&grid[neighbours[j]]);
        var d2: f32 = 0;
        for (0..8) |i| {
            const q = xg[@intCast(@divTrunc(pg[i] - 1, 2))];
            const diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_score) {
            best_score = d2;
            grid_index = @intCast(neighbours[j]);
        }
    }
    if (grid_index < 0) {
        // Fallback: scan the whole codebook. Note the C indexes `xval[i]`
        // here with the *outer* loop variable rather than the inner `j`,
        // which reads as a bug but is what the shipped code does -- and this
        // branch is unreachable anyway, since the neighbour list is never
        // empty.
        for (0..ngrid) |i| {
            const grid_i: *const [8]i8 = @ptrCast(&grid[i]);
            var d2: f32 = 0;
            for (0..8) |j| {
                const q = xg[@intCast(@divTrunc(grid_i[j] - 1, 2))];
                const diff = scale * q - xval[i];
                d2 += weight[j] * diff * diff;
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = @intCast(i);
            }
        }
    }
    impl.assert(grid_index >= 0, "grid_index >= 0");

    const pg: *const [8]i8 = @ptrCast(&grid[@intCast(grid_index)]);
    for (0..8) |i| l_out[i] = @intCast(@divTrunc(pg[i] - 1, 2));
    return grid_index;
}

/// One weight and its original position, for the sort.
///
/// The C sorts a `float[2*n]` where every other slot is an `int` index
/// reinterpreted as a float, and reads the indices back through an aliased
/// `int *`. That works because the comparator only ever looks at the even
/// slots, but it is type punning; a pair struct says the same thing.
const Pair = struct {
    x: f32,
    idx: u32,

    /// Ports `iq1_sort_helper` (ggml-quants.c:4500 @c1d0e7a00), **plus a tie-break the
    /// C does not have**.
    ///
    /// The C compares only the value, and hands that to `qsort`, which leaves
    /// equal elements in an *unspecified* order. That order is not cosmetic:
    /// the split search below accumulates `sumx` in sorted order, so it decides
    /// the rounding, and on a block of identical weights the two candidate
    /// splits score identically in exact arithmetic and are separated only by
    /// that rounding.
    ///
    /// Measured: macOS libc's `qsort` on 32 equal elements returns
    /// `31 1 2 ... 30 0` -- it swaps the ends. glibc would differ, and so
    /// would a future Apple release.
    ///
    /// Matching that would mean reproducing one libc's internals. Instead the
    /// order is made total here, so our output depends on nothing but the
    /// input. It differs from the reference only where the C's own result is
    /// unspecified; `iq_test.zig` skips that one golden case and says so.
    fn lessThan(_: void, a: Pair, b: Pair) bool {
        if (a.x != b.x) return a.x < b.x;
        return a.idx < b.idx;
    }
};

/// Ports `quantize_row_iq1_s_impl` (ggml-quants.c:4508 @c1d0e7a00).
fn iq1SImpl(x: [*]const f32, vy: ?*anyopaque, n: i64, quant_weights: [*]const f32) void {
    const t = c.GGML_TYPE_IQ1_S;
    const kgrid = codebook.iq2Grid(t);
    const kmap = codebook.iq2Map(t);
    const kneighbors = codebook.iq2Neighbours(t);

    std.debug.assert(@rem(n, QK_K) == 0);
    const y: [*]blocks.IQ1_S = @ptrCast(@alignCast(vy.?));
    const nbl: usize = @intCast(@divExact(n, QK_K));
    const block_size = iq1s_block_size;

    const d = blocks.iq1s_delta;
    const x_p = [3]f32{ -1 + d, d, 1 + d };
    const x_m = [3]f32{ -1 - d, -d, 1 - d };

    var scales: [QK_K / block_size]f32 = undefined;
    var weight: [block_size]f32 = undefined;
    var sumx: [block_size + 1]f32 = undefined;
    var sumw: [block_size + 1]f32 = undefined;
    var pairs: [block_size]Pair = undefined;
    var l_buf: [block_size]i8 = undefined;
    var index: [block_size / 8]u16 = undefined;
    var shifts: [QK_K / block_size]i8 = undefined;

    for (0..nbl) |ibl| {
        y[ibl].d = fp16(0.0);
        @memset(&y[ibl].qs, 0);
        @memset(&y[ibl].qh, 0);

        var max_scale: f32 = 0;
        const xbl = x + QK_K * ibl;
        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = 2 * sumx2 / QK_K;

        for (0..QK_K / block_size) |ib| {
            const xb = xbl + block_size * ib;
            const qw = quant_weights + QK_K * ibl + block_size * ib;
            for (0..block_size) |i| weight[i] = qw[i] * @sqrt(sigma2 + xb[i] * xb[i]);

            var max = @abs(xb[0]);
            for (1..block_size) |i| max = @max(max, @abs(xb[i]));
            if (max < h.group_max_eps_iq1_s) {
                scales[ib] = 0;
                shifts[ib] = 1;
                @memset(&l_buf, 1);
                continue;
            }

            // Exact weighted-SSD minimisation: sort, prefix-sum, then try
            // every pair of split points. See the note at the top.
            for (0..block_size) |j| pairs[j] = .{ .x = xb[j], .idx = @intCast(j) };
            std.sort.pdq(Pair, pairs[0..block_size], {}, Pair.lessThan);

            sumx[0] = 0;
            sumw[0] = 0;
            for (0..block_size) |j| {
                const i = pairs[j].idx;
                sumx[j + 1] = sumx[j] + weight[i] * xb[i];
                sumw[j + 1] = sumw[j] + weight[i];
            }

            var best_score: f32 = -std.math.floatMax(f32);
            var scale = max;
            var besti1: i32 = -1;
            var besti2: i32 = -1;
            var best_shift: i8 = 0;

            for (0..block_size + 1) |lo| {
                for (lo..block_size + 1) |hi| {
                    // Both shifts are scored at every split, and whichever
                    // wins overall decides the block's shift bit.
                    inline for (.{ .{ &x_p, @as(i8, 1) }, .{ &x_m, @as(i8, -1) } }) |case| {
                        const xx = case[0];
                        const sumqx = (sumx[lo] - sumx[0]) * xx[0] +
                            (sumx[hi] - sumx[lo]) * xx[1] +
                            (sumx[block_size] - sumx[hi]) * xx[2];
                        const sumq2 = (sumw[lo] - sumw[0]) * xx[0] * xx[0] +
                            (sumw[hi] - sumw[lo]) * xx[1] * xx[1] +
                            (sumw[block_size] - sumw[hi]) * xx[2] * xx[2];
                        if (sumq2 > 0 and sumqx * sumqx > best_score * sumq2) {
                            scale = sumqx / sumq2;
                            best_score = scale * sumqx;
                            besti1 = @intCast(lo);
                            besti2 = @intCast(hi);
                            best_shift = case[1];
                        }
                    }
                }
            }

            if (besti1 < 0 or besti2 < 0 or best_shift == 0) {
                scales[ib] = 0;
                shifts[ib] = 1;
                @memset(&l_buf, 1);
                continue;
            }

            for (0..@intCast(besti1)) |j| l_buf[pairs[j].idx] = 0;
            for (@intCast(besti1)..@intCast(besti2)) |j| l_buf[pairs[j].idx] = 1;
            for (@intCast(besti2)..block_size) |j| l_buf[pairs[j].idx] = 2;

            if (scale < 0) {
                // Mirror the levels rather than storing a negative scale.
                for (0..block_size) |j| l_buf[j] = 2 - l_buf[j];
                scale = -scale;
                best_shift = -best_shift;
            }

            var all_on_grid = true;
            const xx: *const [3]f32 = if (best_shift == 1) &x_p else &x_m;
            for (0..block_size / 8) |k| {
                var u: u16 = 0;
                for (0..8) |j| u |= @as(u16, @intCast(l_buf[8 * k + j])) << @intCast(2 * j);
                var grid_index = kmap[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                    grid_index = findBestNeighbour2(neighbours, kgrid, xb + 8 * k, weight[8 * k ..].ptr, scale, xx, l_buf[8 * k ..].ptr, ngrid_iq1s);
                    impl.assert(grid_index >= 0, "grid_index >= 0");
                }
                index[k] = @intCast(grid_index);
            }

            if (!all_on_grid) {
                // Snapping changed the levels, so refit the scale to them.
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..block_size / 8) |k| {
                    const pg: *const [8]i8 = @ptrCast(&kgrid[index[k]]);
                    for (0..8) |j| {
                        const w = weight[8 * k + j];
                        const q = xx[@intCast(@divTrunc(pg[j] - 1, 2))];
                        sumqx += w * q * xb[8 * k + j];
                        sumq2 += w * q * q;
                    }
                }
                if (sumqx > 0 and sumq2 > 0) scale = sumqx / sumq2;
            }

            var hbits: u16 = 0;
            for (0..block_size / 8) |k| {
                y[ibl].qs[(block_size / 8) * ib + k] = @intCast(index[k] & 255);
                hbits |= (index[k] >> 8) << @intCast(3 * k);
            }
            y[ibl].qh[ib] = hbits;

            impl.assert(scale >= 0, "scale >= 0");
            scales[ib] = scale;
            shifts[ib] = best_shift;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) continue;

        const dv = max_scale / 15;
        // The C: "1.125f is another fudge factor. Don't ask me why it is
        // needed."
        y[ibl].d = fp16(dv * 1.125);
        const id = 1 / dv;
        for (0..QK_K / block_size) |ib| {
            var l = h.nearestInt(0.5 * (id * scales[ib] - 1));
            l = @max(0, @min(7, l));
            // Bit 3 of the stored scale carries the shift choice.
            if (shifts[ib] == -1) l |= 8;
            y[ibl].qh[ib] |= @as(u16, @intCast(l)) << 12;
        }
    }
}

/// Ports `quantize_iq1_s` (ggml-quants.c:4672 @c1d0e7a00).
pub export fn quantize_iq1_s(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    impl.assert(@rem(n_per_row, QK_K) == 0, "n_per_row%QK_K == 0");
    impl.assert(quant_weights != null, "missing quantization weights");
    const nblock: usize = @intCast(@divExact(n_per_row, QK_K));
    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;
    for (0..@intCast(nrow)) |_| {
        iq1SImpl(s, qrow, n_per_row, quant_weights);
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ1_S);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ1_S);
}

// -----------------------------------------------------------------------------
// iq1_m
//
// The same exact-split search as `iq1_s`, over 16-weight blocks instead of 32,
// and with a twist: each block's two halves choose their shift *independently*.
// So there are four combinations rather than two, scored together at every
// split point, and `best_k` records which won.

/// Ports `masks` (ggml-quants.c:4720 @c1d0e7a00): the `qh` bits encoding the shift pair.
const iq1m_masks = [4]u8{ 0x00, 0x80, 0x08, 0x88 };

/// Ports `quantize_row_iq1_m_impl` (ggml-quants.c:4692 @c1d0e7a00).
///
/// Two things here have no counterpart in `iq1_s`:
///
/// - **Four shift combinations.** `x_p`/`x_m` are chosen per half-block, so
///   `k` runs 0..3 as `(+,+), (+,-), (-,+), (-,-)`. The inner loops accumulate
///   all four scores at once, splitting on whether the element index falls in
///   the first or second half.
/// - **A second scale fit at the end.** After the per-block scales are
///   quantized, the super-block scale is refitted against the *rounded*
///   values, which is why the weights are recomputed in that final loop.
fn iq1MImpl(x: [*]const f32, vy: ?*anyopaque, n: i64, quant_weights: ?[*]const f32) void {
    const t = c.GGML_TYPE_IQ1_M;
    const kgrid = codebook.iq2Grid(t);
    const kmap = codebook.iq2Map(t);
    const kneighbors = codebook.iq2Neighbours(t);

    std.debug.assert(@rem(n, QK_K) == 0);
    const y: [*]blocks.IQ1_M = @ptrCast(@alignCast(vy.?));
    const nbl: usize = @intCast(@divExact(n, QK_K));
    const block_size = iq1m_block_size;

    const d = blocks.iq1m_delta;
    const x_p = [3]f32{ -1 + d, d, 1 + d };
    const x_m = [3]f32{ -1 - d, -d, 1 - d };

    var scales: [QK_K / block_size]f32 = undefined;
    var weight: [block_size]f32 = undefined;
    var pairs: [block_size]Pair = undefined;
    var l_buf: [block_size]i8 = undefined;
    var index: [block_size / 8]u16 = undefined;
    var shifts: [QK_K / block_size]i8 = undefined;
    var sumqx: [4]f32 = undefined;
    var sumq2: [4]f32 = undefined;

    for (0..nbl) |ibl| {
        @memset(&y[ibl].qs, 0);
        @memset(&y[ibl].qh, 0);
        @memset(&y[ibl].scales, 0);

        var max_scale: f32 = 0;
        const xbl = x + QK_K * ibl;
        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = 2 * sumx2 / QK_K;

        for (0..QK_K / block_size) |ib| {
            const xb = xbl + block_size * ib;
            if (quant_weights) |qw_all| {
                const qw = qw_all + QK_K * ibl + block_size * ib;
                for (0..block_size) |i| weight[i] = qw[i] * @sqrt(sigma2 + xb[i] * xb[i]);
            } else {
                for (0..block_size) |i| weight[i] = xb[i] * xb[i];
            }

            var max = @abs(xb[0]);
            for (1..block_size) |i| max = @max(max, @abs(xb[i]));
            if (max < h.group_max_eps_iq1_m) {
                scales[ib] = 0;
                shifts[ib] = 0;
                @memset(&l_buf, 1);
                continue;
            }

            for (0..block_size) |j| pairs[j] = .{ .x = xb[j], .idx = @intCast(j) };
            std.sort.pdq(Pair, pairs[0..block_size], {}, Pair.lessThan);

            var best_score: f32 = -std.math.floatMax(f32);
            var scale = max;
            var besti1: i32 = -1;
            var besti2: i32 = -1;
            var best_k: i32 = -1;

            for (0..block_size + 1) |lo| {
                for (lo..block_size + 1) |hi| {
                    @memset(&sumqx, 0);
                    @memset(&sumq2, 0);
                    // Three runs, each contributing its level to all four
                    // shift combinations. Which of x_p/x_m a combination uses
                    // for this element depends on which half it is in.
                    accumulate(&sumqx, &sumq2, pairs[0..lo], weight[0..], xb, &x_p, &x_m, 0, block_size);
                    accumulate(&sumqx, &sumq2, pairs[lo..hi], weight[0..], xb, &x_p, &x_m, 1, block_size);
                    accumulate(&sumqx, &sumq2, pairs[hi..block_size], weight[0..], xb, &x_p, &x_m, 2, block_size);

                    for (0..4) |k| {
                        if (sumq2[k] > 0 and sumqx[k] * sumqx[k] > best_score * sumq2[k]) {
                            scale = sumqx[k] / sumq2[k];
                            best_score = scale * sumqx[k];
                            besti1 = @intCast(lo);
                            besti2 = @intCast(hi);
                            best_k = @intCast(k);
                        }
                    }
                }
            }

            if (besti1 < 0 or besti2 < 0 or best_k < 0) {
                scales[ib] = 0;
                shifts[ib] = 0;
                @memset(&l_buf, 1);
                continue;
            }

            for (0..@intCast(besti1)) |j| l_buf[pairs[j].idx] = 0;
            for (@intCast(besti1)..@intCast(besti2)) |j| l_buf[pairs[j].idx] = 1;
            for (@intCast(besti2)..block_size) |j| l_buf[pairs[j].idx] = 2;

            if (scale < 0) {
                for (0..block_size) |j| l_buf[j] = 2 - l_buf[j];
                scale = -scale;
                // Mirroring the levels swaps both halves' shifts: 0<->3, 1<->2.
                best_k = switch (best_k) {
                    0 => 3,
                    1 => 2,
                    2 => 1,
                    else => 0,
                };
            }

            var all_on_grid = true;
            for (0..block_size / 8) |k| {
                const xx = shiftFor(best_k, k, &x_p, &x_m);
                var u: u16 = 0;
                for (0..8) |j| u |= @as(u16, @intCast(l_buf[8 * k + j])) << @intCast(2 * j);
                var grid_index = kmap[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const neighbours = kneighbors + @as(usize, @intCast(-kmap[u] - 1));
                    grid_index = findBestNeighbour2(neighbours, kgrid, xb + 8 * k, weight[8 * k ..].ptr, scale, xx, l_buf[8 * k ..].ptr, ngrid_iq1s);
                    impl.assert(grid_index >= 0, "grid_index >= 0");
                }
                index[k] = @intCast(grid_index);
            }

            if (!all_on_grid) {
                var sx: f32 = 0;
                var s2: f32 = 0;
                for (0..block_size / 8) |k| {
                    const xx = shiftFor(best_k, k, &x_p, &x_m);
                    const pg: *const [8]i8 = @ptrCast(&kgrid[index[k]]);
                    for (0..8) |j| {
                        const w = weight[8 * k + j];
                        const q = xx[@intCast(@divTrunc(pg[j] - 1, 2))];
                        sx += w * q * xb[8 * k + j];
                        s2 += w * q * q;
                    }
                }
                if (sx > 0 and s2 > 0) scale = sx / s2;
            }

            y[ibl].qs[2 * ib + 0] = @intCast(index[0] & 255);
            y[ibl].qs[2 * ib + 1] = @intCast(index[1] & 255);
            y[ibl].qh[ib] = @intCast((index[0] >> 8) | ((index[1] >> 8) << 4));

            impl.assert(scale >= 0, "scale >= 0");
            scales[ib] = scale;
            shifts[ib] = @intCast(best_k);
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) continue;

        const sc: [*]u16 = @ptrCast(@alignCast(&y[ibl].scales));
        var dv = max_scale / 15;
        const id = 1 / dv;
        var sx: f32 = 0;
        var s2: f32 = 0;

        for (0..QK_K / block_size) |ib| {
            var l = h.nearestInt(0.5 * (id * scales[ib] - 1));
            l = @max(0, @min(7, l));
            sc[ib / 4] |= @as(u16, @intCast(l)) << @intCast(3 * (ib % 4));
            y[ibl].qh[ib] |= iq1m_masks[@intCast(shifts[ib])];

            const xb = xbl + block_size * ib;
            if (quant_weights) |qw_all| {
                const qw = qw_all + QK_K * ibl + block_size * ib;
                for (0..block_size) |i| weight[i] = qw[i] * @sqrt(sigma2 + xb[i] * xb[i]);
            } else {
                for (0..block_size) |i| weight[i] = xb[i] * xb[i];
            }
            for (0..block_size / 8) |k| {
                const xx = shiftFor(shifts[ib], k, &x_p, &x_m);
                const gi = @as(usize, y[ibl].qs[2 * ib + k]) |
                    ((@as(usize, y[ibl].qh[ib]) << @intCast(8 - 4 * k)) & 0x700);
                const pg: *const [8]i8 = @ptrCast(&kgrid[gi]);
                for (0..8) |j| {
                    const w = weight[8 * k + j];
                    // Note the `(2l+1)`: the super-block scale is fitted
                    // against the *rounded* sub-block scale, not the fitted
                    // one.
                    const q = xx[@intCast(@divTrunc(pg[j] - 1, 2))] * @as(f32, @floatFromInt(2 * l + 1));
                    sx += w * q * xb[8 * k + j];
                    s2 += w * q * q;
                }
            }
        }
        if (s2 > 0) dv = sx / s2;

        // The C: "1.1125f is another fudge factor."
        const s16: u16 = fp16(dv * 1.1125);
        // Scattered four bits at a time across the four scale shorts; see
        // `blocks.iq1mScale` for the inverse.
        sc[0] |= (s16 & 0x000f) << 12;
        sc[1] |= (s16 & 0x00f0) << 8;
        sc[2] |= (s16 & 0x0f00) << 4;
        sc[3] |= (s16 & 0xf000) << 0;
    }
}

/// Which shift table a half-block uses, given the combination index.
///
/// `k == 0` is the first half and takes `x_p` for combinations 0 and 1;
/// later halves alternate on the low bit. Ports the pair of ternaries the C
/// repeats at three call sites.
inline fn shiftFor(best_k: i32, k: usize, x_p: *const [3]f32, x_m: *const [3]f32) *const [3]f32 {
    if (k == 0) return if (best_k < 2) x_p else x_m;
    return if (@rem(best_k, 2) == 0) x_p else x_m;
}

/// Accumulates one sorted run's contribution to all four shift combinations.
///
/// The C writes this out three times -- once per run -- with the level index
/// as the only difference, and inside each an if/else on which half the
/// element belongs to. Both are parameters here.
fn accumulate(
    sumqx: *[4]f32,
    sumq2: *[4]f32,
    run: []const Pair,
    weight: []const f32,
    xb: [*]const f32,
    x_p: *const [3]f32,
    x_m: *const [3]f32,
    level: usize,
    block_size: usize,
) void {
    for (run) |p| {
        const i = p.idx;
        const w = weight[i];
        const xp = x_p[level];
        const xm = x_m[level];
        if (i < block_size / 2) {
            // First half: combinations 0 and 1 use the positive shift.
            sumqx[0] += w * xp * xb[i];
            sumqx[1] += w * xp * xb[i];
            sumqx[2] += w * xm * xb[i];
            sumqx[3] += w * xm * xb[i];
            sumq2[0] += w * xp * xp;
            sumq2[1] += w * xp * xp;
            sumq2[2] += w * xm * xm;
            sumq2[3] += w * xm * xm;
        } else {
            // Second half: 0 and 2 use the positive shift instead.
            sumqx[0] += w * xp * xb[i];
            sumqx[2] += w * xp * xb[i];
            sumqx[1] += w * xm * xb[i];
            sumqx[3] += w * xm * xb[i];
            sumq2[0] += w * xp * xp;
            sumq2[2] += w * xp * xp;
            sumq2[1] += w * xm * xm;
            sumq2[3] += w * xm * xm;
        }
    }
}

/// Ports `quantize_iq1_m` (ggml-quants.c:4946 @c1d0e7a00).
pub export fn quantize_iq1_m(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    impl.assert(@rem(n_per_row, QK_K) == 0, "n_per_row%QK_K == 0");
    const nblock: usize = @intCast(@divExact(n_per_row, QK_K));
    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;
    for (0..@intCast(nrow)) |_| {
        iq1MImpl(s, qrow, n_per_row, if (quant_weights != null) quant_weights else null);
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ1_M);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ1_M);
}
