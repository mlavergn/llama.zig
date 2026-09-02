//! The non-linear 4-bit formats, `iq4_nl` and `iq4_xs`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 4966-5145. Each function names the C function it replaces and the line it
//! began at.
//!
//! # Non-linear, not a codebook of vectors
//!
//! Unlike the other i-quants, these store a per-weight index into
//! `kvalues_iq4nl` -- sixteen *scalar* levels spaced to match how model weights
//! are actually distributed, dense near zero and sparse in the tails. A uniform
//! 4-bit grid spends half its levels where almost no weights live.
//!
//! `iq4_nl` is 32 weights with one f16 scale. `iq4_xs` is a 256-weight
//! super-block with a 6-bit scale per 32. Both go through the same routine,
//! which is why it takes the block sizes as parameters.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const h = @import("helpers.zig");
const c = impl.c;

const fp16 = impl.fp32ToFp16;
const QK_K = c.QK_K;

/// Ports `best_index_int8` (ggml-quants.c:28 @c1d0e7a00).
///
/// Binary search for the nearest level, then a tie-break between the two
/// straddling entries. The table is sorted ascending, which is what lets the
/// search work at all -- `kvalues_mxfp4` is not, which is why `legacy.zig`
/// searches its table linearly instead.
pub fn bestIndexInt8(n: usize, val: [*]const i8, x: f32) usize {
    if (x <= @as(f32, @floatFromInt(val[0]))) return 0;
    if (x >= @as(f32, @floatFromInt(val[n - 1]))) return n - 1;
    var ml: usize = 0;
    var mu: usize = n - 1;
    while (mu - ml > 1) {
        const mav = (ml + mu) / 2;
        if (x < @as(f32, @floatFromInt(val[mav]))) mu = mav else ml = mav;
    }
    return if (x - @as(f32, @floatFromInt(val[mu - 1])) < @as(f32, @floatFromInt(val[mu])) - x) mu - 1 else mu;
}

/// Ports `quantize_row_iq4_nl_impl` (ggml-quants.c:4966 @c1d0e7a00).
///
/// One routine for both formats: `iq4_nl` calls it with
/// `super_block_size == block_size == 32` and no sub-block scales, `iq4_xs`
/// with 256 and 32.
///
/// Parameters:
/// - `super_block_size`, `block_size`: the two granularities.
/// - `x`: the weights.
/// - `dh`: receives the f16 super-block scale.
/// - `q4`: receives the packed 4-bit indices.
/// - `scales_h`, `scales_l`: receive the 6-bit sub-block scales, split. Unused
///   when there is only one sub-block.
/// - `scales`, `weight`, `l_buf`: caller-provided scratch.
/// - `values`: the level table, `kvalues_iq4nl`.
/// - `quant_weights`: optional importance matrix.
/// - `ntry`: how far either side of the naive scale to search. **Negative
///   flips the sign of the initial scale** as well as skipping the search,
///   which is how `quantize_row_iq4_nl_ref` differs from `quantize_iq4_nl`.
pub fn quantizeRowIq4NlImpl(
    super_block_size: usize,
    block_size: usize,
    x: [*]const f32,
    dh: *blocks.Half,
    q4: [*]u8,
    scales_h: *u16,
    scales_l: ?[*]u8,
    scales: [*]f32,
    weight: [*]f32,
    l_buf: [*]u8,
    values: [*]const i8,
    quant_weights: ?[*]const f32,
    ntry: i32,
) void {
    var sigma2: f32 = 0;
    for (0..super_block_size) |j| sigma2 += x[j] * x[j];
    sigma2 *= 2.0 / @as(f32, @floatFromInt(super_block_size));

    @memset(q4[0 .. super_block_size / 2], 0);
    dh.* = fp16(0.0);

    var max_scale: f32 = 0;
    var amax_scale: f32 = 0;
    const n_blocks = super_block_size / block_size;

    for (0..n_blocks) |ib| {
        const xb = x + ib * block_size;
        const lb = l_buf + ib * block_size;
        if (quant_weights) |qw_all| {
            const qw = qw_all + ib * block_size;
            for (0..block_size) |j| weight[j] = qw[j] * @sqrt(sigma2 + xb[j] * xb[j]);
        } else {
            for (0..block_size) |j| weight[j] = xb[j] * xb[j];
        }
        var amax: f32 = 0;
        var max: f32 = 0;
        for (0..block_size) |j| {
            const ax = @abs(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax < h.group_max_eps) {
            scales[ib] = 0;
            continue;
        }
        // values[0] is the most negative level, so dividing by it maps the
        // block's extreme onto the end of the table.
        var d: f32 = if (ntry > 0) -max / @as(f32, @floatFromInt(values[0])) else max / @as(f32, @floatFromInt(values[0]));
        var id = 1 / d;
        var sumqx: f32 = 0;
        var sumq2: f32 = 0;
        for (0..block_size) |j| {
            const al = id * xb[j];
            const l = bestIndexInt8(16, values, al);
            lb[j] = @intCast(l);
            const q = @as(f32, @floatFromInt(values[l]));
            const w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = if (sumq2 > 0) sumqx / sumq2 else 0.0;
        var best = d * sumqx;
        var itry: i32 = -ntry;
        while (itry <= ntry) : (itry += 1) {
            id = (@as(f32, @floatFromInt(itry)) + @as(f32, @floatFromInt(values[0]))) / max;
            sumqx = 0;
            sumq2 = 0;
            for (0..block_size) |j| {
                const al = id * xb[j];
                const l = bestIndexInt8(16, values, al);
                const q = @as(f32, @floatFromInt(values[l]));
                const w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 and sumqx * sumqx > best * sumq2) {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        const abs_d = @abs(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    if (n_blocks > 1) {
        @memset(@as([*]u8, @ptrCast(scales_h))[0 .. ((n_blocks + 7) / 8) * @sizeOf(u16)], 0);
        const d = -max_scale / 32;
        dh.* = fp16(d);
        const id: f32 = if (d != 0) 1 / d else 0.0;
        for (0..n_blocks) |ib| {
            var l = h.nearestInt(id * scales[ib]);
            l = @max(-32, @min(31, l));
            const dl = d * @as(f32, @floatFromInt(l));
            const idl: f32 = if (dl != 0) 1 / dl else 0.0;
            const lb = l_buf + ib * block_size;
            const xb = x + ib * block_size;
            // Requantize against the rounded sub-block scale, not the fitted
            // one -- the same absorb-the-rounding step the K-quants make.
            for (0..block_size) |j| {
                lb[j] = @intCast(bestIndexInt8(16, values, idl * xb[j]));
            }
            l += 32;
            const l_l: u8 = @intCast(l & 0xf);
            const l_h: u8 = @intCast(l >> 4);
            if (ib % 2 == 0) scales_l.?[ib / 2] = l_l else scales_l.?[ib / 2] |= l_l << 4;
            scales_h.* |= @as(u16, l_h) << @intCast(2 * (ib % 8));
        }
    } else {
        dh.* = fp16(scales[0]);
        if (ntry > 0) {
            const id: f32 = if (scales[0] != 0) 1 / scales[0] else 0;
            for (0..super_block_size) |j| {
                l_buf[j] = @intCast(bestIndexInt8(16, values, id * x[j]));
            }
        }
    }

    for (0..super_block_size / 32) |i| {
        for (0..16) |j| {
            q4[16 * i + j] = l_buf[32 * i + j] | (l_buf[32 * i + 16 + j] << 4);
        }
    }
}

/// Ports `quantize_row_iq4_nl_ref` (ggml-quants.c:5100 @c1d0e7a00).
///
/// Note `ntry = -1`: no search, and the initial scale keeps the sign of `max`
/// rather than flipping it. The chunk entry point uses `7` instead.
pub export fn quantize_row_iq4_nl_ref(x: [*c]const f32, y: [*c]blocks.IQ4_NL, k: i64) void {
    impl.assert(@rem(k, c.QK4_NL) == 0, "k%QK4_NL == 0");
    const nblock: usize = @intCast(@divExact(k, c.QK4_NL));

    // **Zeroed, where the C leaves this uninitialized.** For an all-zero block
    // the impl takes its `amax < eps` early-out, never writes `L`, and then
    // packs `q4` from it anyway -- reading uninitialized stack. That is not
    // theoretical: two consecutive calls to the C in one process return
    // different bytes for the same input.
    //
    // Matching that is impossible and not worth wanting, so this produces the
    // defined answer instead: an all-zero block quantizes to all-zero quants.
    // The `zeros` golden for IQ4_NL is skipped in the tests for the same
    // reason -- it captured one sample of that garbage.
    var l_buf: [c.QK4_NL]u8 = @splat(0);
    var weight: [c.QK4_NL]f32 = undefined;
    var unused_h: u16 = undefined;
    var scale: f32 = undefined;

    for (0..nblock) |ibl| {
        quantizeRowIq4NlImpl(
            c.QK4_NL,
            32,
            x + c.QK4_NL * ibl,
            @ptrCast(&y[ibl].d),
            @ptrCast(&y[ibl].qs),
            &unused_h,
            null,
            @ptrCast(&scale),
            &weight,
            &l_buf,
            @ptrCast(&c.kvalues_iq4nl),
            null,
            -1,
        );
    }
}

/// Ports `quantize_iq4_nl` (ggml-quants.c:5077 @c1d0e7a00).
pub export fn quantize_iq4_nl(
    src: [*c]const f32,
    dst: ?*anyopaque,
    nrow: i64,
    n_per_row: i64,
    quant_weights: [*c]const f32,
) usize {
    impl.assert(@rem(n_per_row, c.QK4_NL) == 0, "n_per_row%QK4_NL == 0");
    const nblock: usize = @intCast(@divExact(n_per_row, c.QK4_NL));

    // Zeroed for the same reason as in `quantize_row_iq4_nl_ref` above.
    var l_buf: [c.QK4_NL]u8 = @splat(0);
    var weight: [c.QK4_NL]f32 = undefined;
    var unused_h: u16 = undefined;
    var scale: f32 = undefined;

    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;

    for (0..@intCast(nrow)) |_| {
        const iq4: [*]blocks.IQ4_NL = @ptrCast(@alignCast(qrow));
        for (0..nblock) |ibl| {
            const qw: ?[*]const f32 = if (quant_weights != null) quant_weights + c.QK4_NL * ibl else null;
            quantizeRowIq4NlImpl(
                c.QK4_NL,
                32,
                s + c.QK4_NL * ibl,
                &iq4[ibl].d,
                &iq4[ibl].qs,
                &unused_h,
                null,
                @ptrCast(&scale),
                &weight,
                &l_buf,
                @ptrCast(&c.kvalues_iq4nl),
                qw,
                7,
            );
        }
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ4_NL);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ4_NL);
}

/// Ports `quantize_iq4_xs` (ggml-quants.c:5115 @c1d0e7a00).
pub export fn quantize_iq4_xs(
    src: [*c]const f32,
    dst: ?*anyopaque,
    nrow: i64,
    n_per_row: i64,
    quant_weights: [*c]const f32,
) usize {
    impl.assert(@rem(n_per_row, QK_K) == 0, "n_per_row%QK_K == 0");
    const nblock: usize = @intCast(@divExact(n_per_row, QK_K));

    var l_buf: [QK_K]u8 = undefined;
    var weight: [32]f32 = undefined;
    var scales: [QK_K / 32]f32 = undefined;

    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;

    for (0..@intCast(nrow)) |_| {
        const iq4: [*]blocks.IQ4_XS = @ptrCast(@alignCast(qrow));
        for (0..nblock) |ibl| {
            const qw: ?[*]const f32 = if (quant_weights != null) quant_weights + QK_K * ibl else null;
            quantizeRowIq4NlImpl(
                QK_K,
                32,
                s + QK_K * ibl,
                &iq4[ibl].d,
                &iq4[ibl].qs,
                &iq4[ibl].scales_h,
                &iq4[ibl].scales_l,
                &scales,
                &weight,
                &l_buf,
                @ptrCast(&c.kvalues_iq4nl),
                qw,
                7,
            );
        }
        s += @intCast(n_per_row);
        qrow += nblock * @sizeOf(blocks.IQ4_XS);
    }
    return @as(usize, @intCast(nrow)) * nblock * @sizeOf(blocks.IQ4_XS);
}

/// Ports `quantize_row_iq4_xs_ref` (ggml-quants.c:5135 @c1d0e7a00).
pub export fn quantize_row_iq4_xs_ref(x: [*c]const f32, y: [*c]blocks.IQ4_XS, k: i64) void {
    std.debug.assert(@rem(k, QK_K) == 0);
    _ = quantize_iq4_xs(x, @ptrCast(y), 1, k, null);
}

// -----------------------------------------------------------------------------
// Unit Tests

const t = @import("testing.zig");
const iq_dequant = @import("iq_dequant.zig");

test {
    std.testing.refAllDecls(@This());
}

/// Checks a format that has both a reference quantizer and a chunk entry
/// point, against every input pattern.
///
/// The chunk path is checked with *and* without an importance matrix, because
/// for these formats the two take different branches -- `quant_weights`
/// changes the error weighting, and `ntry` differs between the ref and chunk
/// entry points besides.
fn checkIq4(
    comptime name: []const u8,
    comptime Block: type,
    ref: *const fn ([*c]const f32, [*c]Block, i64) callconv(.c) void,
    chunk: *const fn ([*c]const f32, ?*anyopaque, i64, i64, [*c]const f32) callconv(.c) usize,
    dequant: *const fn ([*c]const Block, [*c]f32, i64) callconv(.c) void,
) !void {
    for (t.all_patterns) |pattern| {
        // `quantize_row_iq4_nl_ref` reads uninitialized memory for an all-zero
        // block -- see the note in the function. The golden recorded one
        // sample of it, and the C does not even agree with itself across two
        // calls, so there is nothing here to match.
        if (std.mem.eql(u8, name, "IQ4_NL") and pattern == .zeros) continue;

        const g = t.find(name, pattern);

        var src: [t.n_elem]f32 = undefined;
        var imatrix: [t.n_per_row]f32 = undefined;
        t.fillSrc(pattern, &src);
        t.fillImatrix(&imatrix);

        var buf: [t.n_elem * 4]u8 align(16) = undefined;

        @memset(&buf, 0);
        ref(&src, @ptrCast(@alignCast(&buf)), @intCast(t.n_elem));
        const used = g.row_size * t.n_rows;
        std.testing.expectEqual(g.ref.?, t.fnv(buf[0..used])) catch |e| {
            std.debug.print("{s}: quantize_row_*_ref differs on '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        var out: [t.n_elem]f32 = undefined;
        @memset(&out, 0);
        dequant(@ptrCast(@alignCast(&buf)), &out, @intCast(t.n_elem));
        std.testing.expectEqual(g.deq.?, t.fnv(std.mem.sliceAsBytes(out[0..]))) catch |e| {
            std.debug.print("{s}: dequantize differs on '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        @memset(&buf, 0);
        const n0 = chunk(&src, &buf, t.n_rows, t.n_per_row, null);
        std.testing.expectEqual(g.chunk.?, t.fnv(buf[0..n0])) catch |e| {
            std.debug.print("{s}: quantize_* differs on '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        @memset(&buf, 0);
        const n1 = chunk(&src, &buf, t.n_rows, t.n_per_row, &imatrix);
        std.testing.expectEqual(g.chunk_imatrix.?, t.fnv(buf[0..n1])) catch |e| {
            std.debug.print("{s}: quantize_* with imatrix differs on '{s}'\n", .{ name, pattern.name() });
            return e;
        };
    }
}

test "iq4_nl matches the C on every pattern, ref and chunk" {
    try checkIq4("IQ4_NL", blocks.IQ4_NL, quantize_row_iq4_nl_ref, quantize_iq4_nl, iq_dequant.dequantize_row_iq4_nl);
}

test "iq4_xs matches the C on every pattern, ref and chunk" {
    try checkIq4("IQ4_XS", blocks.IQ4_XS, quantize_row_iq4_xs_ref, quantize_iq4_xs, iq_dequant.dequantize_row_iq4_xs);
}

test "the level table is sorted, which is what the binary search needs" {
    // bestIndexInt8 binary-searches kvalues_iq4nl. kvalues_mxfp4 is *not*
    // sorted, which is why legacy.zig searches that one linearly -- using this
    // routine there would silently return the wrong level.
    var prev: i8 = std.math.minInt(i8);
    for (c.kvalues_iq4nl) |v| {
        try std.testing.expect(v > prev);
        prev = v;
    }
}

test "bestIndexInt8 picks the nearer level and breaks ties low" {
    const vals: [*]const i8 = @ptrCast(&c.kvalues_iq4nl);
    // Below the table and above it clamp to the ends.
    try std.testing.expectEqual(@as(usize, 0), bestIndexInt8(16, vals, -1000.0));
    try std.testing.expectEqual(@as(usize, 15), bestIndexInt8(16, vals, 1000.0));
    // An exact hit returns that level.
    for (0..16) |i| {
        const x: f32 = @floatFromInt(c.kvalues_iq4nl[i]);
        try std.testing.expectEqual(i, bestIndexInt8(16, vals, x));
    }
    // Exactly halfway goes to the *higher* index. The C's comparison is
    // `x - val[mu-1] < val[mu] - x`, which is false at a tie, so it falls
    // through to `mu`. Guessing "ties round down" was wrong.
    const mid = (@as(f32, @floatFromInt(c.kvalues_iq4nl[3])) + @as(f32, @floatFromInt(c.kvalues_iq4nl[4]))) / 2.0;
    try std.testing.expectEqual(@as(usize, 4), bestIndexInt8(16, vals, mid));
}
