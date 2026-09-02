//! The `quantize_*` chunk entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), the
//! `quantize_*` functions scattered through the file, plus the
//! `quantize_row_*_impl` helpers they call. Each names the C function it
//! replaces and the line it began at.
//!
//! # What these add over the `_ref` quantizers
//!
//! An **importance matrix**. `quantize_row_*_ref` weights every weight in a
//! block equally; these accept `quant_weights` from a calibration run and
//! weight the fit by how much error in each weight actually costs the model.
//! That is what makes a 4-bit quantization of a real model usable.
//!
//! Without an imatrix almost all of them simply forward to the `_ref` version,
//! which is why `chunk` and `ref` share a checksum in `golden.zig` for most
//! formats.
//!
//! # The row loop
//!
//! Every one has the same shape: iterate rows, quantize each into `row_size`
//! bytes, return the total. The differences are only in what happens per row,
//! so `eachRow` below carries the loop and the callers supply the body.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const h = @import("helpers.zig");
const legacy = @import("legacy.zig");
const k_quants = @import("k.zig");
const ternary = @import("ternary.zig");
const c = impl.c;

const fp16 = impl.fp32ToFp16;
const unfp16 = impl.fp16ToFp32;
const QK_K = c.QK_K;

/// `ggml_row_size` without depending on the traits table.
///
/// The real one lives in `types.zig` and reads `type_traits`, which would drag
/// the whole ported `ggml.c` into this file's test link. The block sizes are
/// fixed by the format, so computing directly costs nothing and keeps
/// `test-quants` standing alone.
fn rowSize(comptime Block: type, comptime block_elems: usize, n_per_row: i64) usize {
    return @as(usize, @intCast(n_per_row)) / block_elems * @sizeOf(Block);
}

/// The row loop every chunk entry point shares.
///
/// Parameters:
/// - `Block`: the block type, for the row stride.
/// - `block_elems`: weights per block.
/// - `nrow`, `n_per_row`: the shape.
/// - `dst`: destination.
/// - `body`: called once per row with the row's source, destination, and the
///   row's slice of the importance matrix.
///
/// Return: total bytes written.
inline fn eachRow(
    comptime Block: type,
    comptime block_elems: usize,
    src: [*]const f32,
    dst: ?*anyopaque,
    nrow: i64,
    n_per_row: i64,
    quant_weights: ?[*]const f32,
    body: anytype,
) usize {
    const row_size = rowSize(Block, block_elems, n_per_row);
    var qrow: [*]u8 = @ptrCast(dst.?);
    var s = src;
    for (0..@intCast(nrow)) |_| {
        body(s, @as([*]Block, @ptrCast(@alignCast(qrow))), n_per_row, quant_weights);
        s += @intCast(n_per_row);
        qrow += row_size;
    }
    return @as(usize, @intCast(nrow)) * row_size;
}

/// The `qw[j] * sqrt(sigma2 + x*x)` weighting the imatrix paths share.
///
/// `sigma2 + xb[j]*xb[j]` is a contraction site: the C fuses it into one FMA
/// and this does not, so the last bit can differ. Deliberate -- the goldens are
/// built with `-ffp-contract=off` rather than the fusion being chased with
/// `@mulAdd`. See the note at the top of `helpers.zig`.
inline fn imatrixWeights(dst: [*]f32, qw: [*]const f32, xb: [*]const f32, n: usize, sigma2: f32) void {
    for (0..n) |j| dst[j] = qw[j] * @sqrt(sigma2 + xb[j] * xb[j]);
}

// -----------------------------------------------------------------------------
// Formats whose chunk entry point ignores the importance matrix
//
// These have no imatrix-aware fit, so the chunk form is the reference form
// with a row loop around it. The C spells each out; they differ only in type.

/// Builds a chunk entry point that forwards to a reference quantizer.
fn forwardingChunk(
    comptime Block: type,
    comptime block_elems: usize,
    comptime ref: anytype,
) fn ([*c]const f32, ?*anyopaque, i64, i64, [*c]const f32) callconv(.c) usize {
    return struct {
        fn f(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) callconv(.c) usize {
            _ = quant_weights; // not used
            ref(src, @ptrCast(@alignCast(dst)), nrow * n_per_row);
            return @as(usize, @intCast(nrow)) * rowSize(Block, block_elems, n_per_row);
        }
    }.f;
}

// Ports `quantize_q8_0` (ggml-quants.c:2295 @c1d0e7a00), `quantize_mxfp4`,
// `quantize_nvfp4`, `quantize_tq1_0` (2420) and `quantize_tq2_0`.
//
// `q1_0` and `q2_0` are *not* here: their C versions take the imatrix
// argument, ignore it, and still run the row loop, so they keep that shape
// below.
comptime {
    @export(&forwardingChunk(blocks.Q8_0, c.QK8_0, legacy.quantize_row_q8_0_ref), .{ .name = "quantize_q8_0" });
    @export(&forwardingChunk(blocks.MXFP4, c.QK_MXFP4, legacy.quantize_row_mxfp4_ref), .{ .name = "quantize_mxfp4" });
    @export(&forwardingChunk(blocks.NVFP4, c.QK_NVFP4, legacy.quantize_row_nvfp4_ref), .{ .name = "quantize_nvfp4" });
    @export(&forwardingChunk(blocks.TQ1_0, QK_K, ternary.quantize_row_tq1_0_ref), .{ .name = "quantize_tq1_0" });
    @export(&forwardingChunk(blocks.TQ2_0, QK_K, ternary.quantize_row_tq2_0_ref), .{ .name = "quantize_tq2_0" });
}

/// Ports `quantize_q1_0` (ggml-quants.c:2098 @c1d0e7a00).
///
/// Takes the imatrix and ignores it, but still loops per row rather than
/// quantizing the whole buffer at once. The two are equivalent here; the shape
/// is kept so the correspondence with the C is checkable.
pub export fn quantize_q1_0(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    if (quant_weights == null) {
        legacy.quantize_row_q1_0_ref(src, @ptrCast(@alignCast(dst)), nrow * n_per_row);
        return @as(usize, @intCast(nrow)) * rowSize(blocks.Q1_0, c.QK1_0, n_per_row);
    }
    return eachRow(blocks.Q1_0, c.QK1_0, src, dst, nrow, n_per_row, null, struct {
        fn f(s: [*]const f32, y: [*]blocks.Q1_0, npr: i64, _: ?[*]const f32) void {
            legacy.quantize_row_q1_0_ref(s, y, npr);
        }
    }.f);
}

/// Ports `quantize_q2_0` (ggml-quants.c:2113 @c1d0e7a00).
pub export fn quantize_q2_0(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) usize {
    if (quant_weights == null) {
        legacy.quantize_row_q2_0_ref(src, @ptrCast(@alignCast(dst)), nrow * n_per_row);
        return @as(usize, @intCast(nrow)) * rowSize(blocks.Q2_0, c.QK2_0, n_per_row);
    }
    return eachRow(blocks.Q2_0, c.QK2_0, src, dst, nrow, n_per_row, null, struct {
        fn f(s: [*]const f32, y: [*]blocks.Q2_0, npr: i64, _: ?[*]const f32) void {
            legacy.quantize_row_q2_0_ref(s, y, npr);
        }
    }.f);
}

// -----------------------------------------------------------------------------
// The legacy formats with an imatrix-aware fit

/// Ports `quantize_row_q4_0_impl` (ggml-quants.c:2070 @c1d0e7a00).
fn q4_0Impl(x: [*]const f32, y: [*]blocks.Q4_0, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const qw_all = quant_weights orelse {
        legacy.quantize_row_q4_0_ref(x, y, n_per_row);
        return;
    };

    var weight: [c.QK4_0]f32 = undefined;
    var l_buf: [c.QK4_0]i8 = undefined;

    var sum_x2: f32 = 0;
    for (0..@intCast(n_per_row)) |j| sum_x2 += x[j] * x[j];
    const sigma2 = sum_x2 / @as(f32, @floatFromInt(n_per_row));

    const nb: usize = @intCast(@divExact(n_per_row, c.QK4_0));
    for (0..nb) |ib| {
        const xb = x + c.QK4_0 * ib;
        imatrixWeights(&weight, qw_all + c.QK4_0 * ib, xb, c.QK4_0, sigma2);
        const d = h.makeQxQuants(c.QK4_0, 8, xb, &l_buf, 1, &weight);
        y[ib].d = fp16(d);
        for (0..16) |j| {
            y[ib].qs[j] = @as(u8, @bitCast(l_buf[j])) | (@as(u8, @bitCast(l_buf[j + 16])) << 4);
        }
    }
}

/// Ports `quantize_row_q4_1_impl` (ggml-quants.c:2143 @c1d0e7a00).
fn q4_1Impl(x: [*]const f32, y: [*]blocks.Q4_1, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const qw_all = quant_weights orelse {
        legacy.quantize_row_q4_1_ref(x, y, n_per_row);
        return;
    };

    var weight: [c.QK4_1]f32 = undefined;
    var l_buf: [c.QK4_1]u8 = undefined;
    var l_aux: [c.QK4_1]u8 = undefined;

    var sum_x2: f32 = 0;
    for (0..@intCast(n_per_row)) |j| sum_x2 += x[j] * x[j];
    const sigma2 = sum_x2 / @as(f32, @floatFromInt(n_per_row));

    const nb: usize = @intCast(@divExact(n_per_row, c.QK4_1));
    for (0..nb) |ib| {
        const xb = x + c.QK4_1 * ib;
        imatrixWeights(&weight, qw_all + c.QK4_1 * ib, xb, c.QK4_1, sigma2);
        var min: f32 = undefined;
        const d = h.makeQkx3Quants(c.QK4_1, 15, xb, &weight, &l_buf, &min, &l_aux, -0.9, 0.05, 36, false);
        y[ib].d = fp16(d);
        y[ib].m = fp16(-min);
        for (0..16) |j| y[ib].qs[j] = l_buf[j] | (l_buf[j + 16] << 4);
    }
}

/// Ports `quantize_row_q5_0_impl` (ggml-quants.c:2188 @c1d0e7a00).
fn q5_0Impl(x: [*]const f32, y: [*]blocks.Q5_0, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const qw_all = quant_weights orelse {
        legacy.quantize_row_q5_0_ref(x, y, n_per_row);
        return;
    };

    var weight: [c.QK5_0]f32 = undefined;
    var l_buf: [c.QK5_0]i8 = undefined;

    var sum_x2: f32 = 0;
    for (0..@intCast(n_per_row)) |j| sum_x2 += x[j] * x[j];
    const sigma2 = sum_x2 / @as(f32, @floatFromInt(n_per_row));

    const nb: usize = @intCast(@divExact(n_per_row, c.QK5_0));
    for (0..nb) |ib| {
        const xb = x + c.QK5_0 * ib;
        imatrixWeights(&weight, qw_all + c.QK5_0 * ib, xb, c.QK5_0, sigma2);
        const d = h.makeQxQuants(c.QK5_0, 16, xb, &l_buf, 1, &weight);
        y[ib].d = fp16(d);

        var qh: u32 = 0;
        for (0..16) |j| {
            const xi0: u8 = @bitCast(l_buf[j]);
            const xi1: u8 = @bitCast(l_buf[j + 16]);
            y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j + 0);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + c.QK5_0 / 2);
        }
        @memcpy(&y[ib].qh, std.mem.asBytes(&qh));
    }
}

/// Ports `quantize_row_q5_1_impl` (ggml-quants.c:2242 @c1d0e7a00).
fn q5_1Impl(x: [*]const f32, y: [*]blocks.Q5_1, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const qw_all = quant_weights orelse {
        legacy.quantize_row_q5_1_ref(x, y, n_per_row);
        return;
    };

    var weight: [c.QK5_1]f32 = undefined;
    var l_buf: [c.QK5_1]u8 = undefined;
    var l_aux: [c.QK5_1]u8 = undefined;

    var sum_x2: f32 = 0;
    for (0..@intCast(n_per_row)) |j| sum_x2 += x[j] * x[j];
    const sigma2 = sum_x2 / @as(f32, @floatFromInt(n_per_row));

    const nb: usize = @intCast(@divExact(n_per_row, c.QK5_1));
    for (0..nb) |ib| {
        const xb = x + c.QK5_1 * ib;
        imatrixWeights(&weight, qw_all + c.QK5_1 * ib, xb, c.QK5_1, sigma2);
        var min: f32 = undefined;
        const d = h.makeQkx3Quants(c.QK5_1, 31, xb, &weight, &l_buf, &min, &l_aux, -0.9, 0.05, 36, false);
        y[ib].d = fp16(d);
        y[ib].m = fp16(-min);

        var qh: u32 = 0;
        for (0..16) |j| {
            const xi0 = l_buf[j];
            const xi1 = l_buf[j + 16];
            y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= @as(u32, (xi0 & 0x10) >> 4) << @intCast(j + 0);
            qh |= @as(u32, (xi1 & 0x10) >> 4) << @intCast(j + c.QK5_1 / 2);
        }
        @memcpy(&y[ib].qh, std.mem.asBytes(&qh));
    }
}

/// Builds a chunk entry point that runs an imatrix-aware impl per row.
///
/// The `!quant_weights` short-circuit quantizes the whole buffer in one call
/// rather than per row, exactly as the C does. Equivalent, and kept.
fn implChunk(
    comptime Block: type,
    comptime block_elems: usize,
    comptime ref: anytype,
    comptime impl_fn: anytype,
) fn ([*c]const f32, ?*anyopaque, i64, i64, [*c]const f32) callconv(.c) usize {
    return struct {
        fn f(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, quant_weights: [*c]const f32) callconv(.c) usize {
            if (quant_weights == null) {
                ref(src, @ptrCast(@alignCast(dst)), nrow * n_per_row);
                return @as(usize, @intCast(nrow)) * rowSize(Block, block_elems, n_per_row);
            }
            return eachRow(Block, block_elems, src, dst, nrow, n_per_row, quant_weights, impl_fn);
        }
    }.f;
}

comptime {
    @export(&implChunk(blocks.Q4_0, c.QK4_0, legacy.quantize_row_q4_0_ref, q4_0Impl), .{ .name = "quantize_q4_0" });
    @export(&implChunk(blocks.Q4_1, c.QK4_1, legacy.quantize_row_q4_1_ref, q4_1Impl), .{ .name = "quantize_q4_1" });
    @export(&implChunk(blocks.Q5_0, c.QK5_0, legacy.quantize_row_q5_0_ref, q5_0Impl), .{ .name = "quantize_q5_0" });
    @export(&implChunk(blocks.Q5_1, c.QK5_1, legacy.quantize_row_q5_1_ref, q5_1Impl), .{ .name = "quantize_q5_1" });
}

// -----------------------------------------------------------------------------
// The K-quants
//
// Their imatrix impls differ from the `_ref` versions in more than weighting:
// they fit the sub-block scales with `makeQpQuants` against a per-sub-block
// weight sum, where the reference divides by the block maximum. So these are
// genuinely different code, not the reference with a weight argument.

/// Ports `quantize_row_q2_K_impl` (ggml-quants.c:1149 @c1d0e7a00).
fn q2_KImpl(x_in: [*]const f32, y: [*]blocks.Q2_K, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const qw_all = quant_weights.?; // the C asserts this
    const nb: usize = @intCast(@divExact(n_per_row, QK_K));
    var x = x_in;

    var l_buf: [QK_K]u8 = undefined;
    var l_aux: [16]u8 = undefined;
    var mins: [QK_K / 16]f32 = undefined;
    var scales: [QK_K / 16]f32 = undefined;
    var sw: [QK_K / 16]f32 = undefined;
    var weight: [16]f32 = undefined;
    var ls: [QK_K / 16]u8 = undefined;
    var lm: [QK_K / 16]u8 = undefined;

    for (0..nb) |i| {
        @memset(&sw, 0);
        var sumx2: f32 = 0;
        for (0..QK_K) |j| sumx2 += x[j] * x[j];
        const sigma2 = sumx2 / QK_K;

        for (0..QK_K / 16) |j| {
            const qw = qw_all + QK_K * i + 16 * j;
            imatrixWeights(&weight, qw, x + 16 * j, 16, sigma2);
            // Note the bound: `QK_K/16`, not 16. They are equal at QK_K = 256,
            // so this reads all sixteen weights either way -- but the C wrote
            // it this way and the two would part company at another QK_K.
            for (0..QK_K / 16) |l| sw[j] += weight[l];
            scales[j] = h.makeQkx3Quants(16, 3, x + 16 * j, &weight, @ptrCast(&l_buf[16 * j]), &mins[j], &l_aux, -0.9, 0.05, 36, false);
        }

        var dm = h.makeQpQuants(QK_K / 16, 15, &scales, &ls, &sw);
        var mm = h.makeQpQuants(QK_K / 16, 15, &mins, &lm, &sw);
        y[i].d = fp16(dm);
        y[i].dmin = fp16(mm);
        // Read back through f16 so the requantize below uses the stored value.
        dm = unfp16(y[i].d);
        mm = unfp16(y[i].dmin);

        for (0..QK_K / 16) |j| y[i].scales[j] = ls[j] | (lm[j] << 4);

        for (0..QK_K / 16) |j| {
            const d = dm * @as(f32, @floatFromInt(y[i].scales[j] & 0xF));
            if (d == 0) continue;
            const m = mm * @as(f32, @floatFromInt(y[i].scales[j] >> 4));
            for (0..16) |ii| {
                var l = h.nearestInt((x[16 * j + ii] + m) / d);
                l = @max(0, @min(3, l));
                l_buf[16 * j + ii] = @intCast(l);
            }
        }

        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                y[i].qs[j / 4 + l] = l_buf[j + l] |
                    (l_buf[j + l + 32] << 2) |
                    (l_buf[j + l + 64] << 4) |
                    (l_buf[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

/// Ports `quantize_row_q3_K_impl` (ggml-quants.c:1355 @c1d0e7a00).
///
/// Unlike `q2_K`'s, this one tolerates a null imatrix -- it falls back to
/// `x*x` weighting rather than forwarding to the reference.
fn q3_KImpl(x_in: [*]const f32, y: [*]blocks.Q3_K, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const nb: usize = @intCast(@divExact(n_per_row, QK_K));
    var x = x_in;

    var l_buf: [QK_K]i8 = undefined;
    var scales: [QK_K / 16]f32 = undefined;
    var weight: [16]f32 = undefined;
    var sw: [QK_K / 16]f32 = undefined;
    var ls: [QK_K / 16]i8 = undefined;

    for (0..nb) |i| {
        var sumx2: f32 = 0;
        for (0..QK_K) |j| sumx2 += x[j] * x[j];
        // Note the 2x, which q2_K's version does not have.
        const sigma2 = 2 * sumx2 / QK_K;

        for (0..QK_K / 16) |j| {
            if (quant_weights) |qw_all| {
                imatrixWeights(&weight, qw_all + QK_K * i + 16 * j, x + 16 * j, 16, sigma2);
            } else {
                for (0..16) |l| weight[l] = x[16 * j + l] * x[16 * j + l];
            }
            var sumw: f32 = 0;
            for (0..16) |l| sumw += weight[l];
            sw[j] = sumw;
            scales[j] = h.makeQxQuants(16, 4, x + 16 * j, @ptrCast(&l_buf[16 * j]), 1, &weight);
        }

        @memset(&y[i].scales, 0);
        const d_block = h.makeQxQuants(QK_K / 16, 32, &scales, &ls, 1, &sw);
        for (0..QK_K / 16) |j| {
            var l: i32 = ls[j];
            if (j < 8) {
                y[i].scales[j] = @as(u8, @intCast(l)) & 0xF;
            } else {
                y[i].scales[j - 8] |= (@as(u8, @intCast(l)) & 0xF) << 4;
            }
            l >>= 4;
            y[i].scales[j % 4 + 8] |= @as(u8, @intCast(l)) << @intCast(2 * (j / 4));
        }
        y[i].d = fp16(d_block);

        for (0..QK_K / 16) |j| {
            var sc: i8 = if (j < 8)
                @bitCast(y[i].scales[j] & 0xF)
            else
                @bitCast(y[i].scales[j - 8] >> 4);
            sc = @bitCast(@as(u8, @bitCast(sc)) | (((y[i].scales[8 + j % 4] >> @intCast(2 * (j / 4))) & 3) << 4));
            sc -%= 32;
            const d = unfp16(y[i].d) * @as(f32, @floatFromInt(sc));
            if (d == 0) continue;
            for (0..16) |ii| {
                var l = h.nearestInt(x[16 * j + ii] / d);
                l = @max(-4, @min(3, l));
                l_buf[16 * j + ii] = @intCast(l + 4);
            }
        }

        @memset(&y[i].hmask, 0);
        var m: usize = 0;
        var hm: u8 = 1;
        for (0..QK_K) |j| {
            if (l_buf[j] > 3) {
                y[i].hmask[m] |= hm;
                l_buf[j] -= 4;
            }
            m += 1;
            if (m == QK_K / 8) {
                m = 0;
                hm <<= 1;
            }
        }

        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                y[i].qs[j / 4 + l] = @as(u8, @bitCast(l_buf[j + l])) |
                    (@as(u8, @bitCast(l_buf[j + l + 32])) << 2) |
                    (@as(u8, @bitCast(l_buf[j + l + 64])) << 4) |
                    (@as(u8, @bitCast(l_buf[j + l + 96])) << 6);
            }
        }

        x += QK_K;
    }
}

/// The scale fitting `quantize_row_q4_K_impl` and `quantize_row_q5_K_impl`
/// share (ggml-quants.c:1553, 1758 @c1d0e7a00).
///
/// Without an imatrix the weighting is `sqrt(sigma2) + |x|`, matching the
/// reference; with one it is the usual `qw * sqrt(sigma2 + x*x)`.
fn fitScalesKImpl(
    comptime nmax: i32,
    x: [*]const f32,
    i: usize,
    quant_weights: ?[*]const f32,
    l_buf: [*]u8,
    scales: *[QK_K / 32]f32,
    mins: *[QK_K / 32]f32,
    sw: *[QK_K / 32]f32,
) void {
    var weights: [32]f32 = undefined;
    var l_aux: [32]u8 = undefined;

    var sum_x2: f32 = 0;
    for (0..QK_K) |l| sum_x2 += x[l] * x[l];
    const sigma2 = 2 * sum_x2 / QK_K;
    const av_x = @sqrt(sigma2);

    for (0..QK_K / 32) |j| {
        if (quant_weights) |qw_all| {
            imatrixWeights(&weights, qw_all + QK_K * i + 32 * j, x + 32 * j, 32, sigma2);
        } else {
            for (0..32) |l| weights[l] = av_x + @abs(x[32 * j + l]);
        }
        var sumw: f32 = 0;
        for (0..32) |l| sumw += weights[l];
        sw[j] = sumw;
        scales[j] = h.makeQkx3Quants(32, nmax, x + 32 * j, &weights, l_buf + 32 * j, &mins[j], &l_aux, -0.9, 0.05, 36, false);
    }
}

/// Ports `quantize_row_q4_K_impl` (ggml-quants.c:1553 @c1d0e7a00).
fn q4_KImpl(x_in: [*]const f32, y: [*]blocks.Q4_K, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const nb: usize = @intCast(@divExact(n_per_row, QK_K));
    var x = x_in;

    var l_buf: [QK_K]u8 = undefined;
    var ls: [QK_K / 32]u8 = undefined;
    var lm: [QK_K / 32]u8 = undefined;
    var sw: [QK_K / 32]f32 = undefined;
    var mins: [QK_K / 32]f32 = undefined;
    var scales: [QK_K / 32]f32 = undefined;

    for (0..nb) |i| {
        fitScalesKImpl(15, x, i, quant_weights, &l_buf, &scales, &mins, &sw);

        const d_block = h.makeQpQuants(QK_K / 32, 63, &scales, &ls, &sw);
        const m_block = h.makeQpQuants(QK_K / 32, 63, &mins, &lm, &sw);

        @memset(&y[i].scales, 0);
        for (0..QK_K / 32) |j| {
            // No MIN(63) here, unlike the reference: makeQpQuants already
            // bounds the level by its nmax.
            packScalePair(&y[i].scales, j, ls[j], lm[j]);
        }
        y[i].d = fp16(d_block);
        y[i].dmin = fp16(m_block);

        requantizeK(15, x, &y[i].scales, unfp16(y[i].d), unfp16(y[i].dmin), &l_buf);

        var q: [*]u8 = &y[i].qs;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            for (0..32) |l| q[l] = l_buf[j + l] | (l_buf[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

/// Ports `quantize_row_q5_K_impl` (ggml-quants.c:1758 @c1d0e7a00).
fn q5_KImpl(x_in: [*]const f32, y: [*]blocks.Q5_K, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const nb: usize = @intCast(@divExact(n_per_row, QK_K));
    var x = x_in;

    var l_buf: [QK_K]u8 = undefined;
    var ls: [QK_K / 32]u8 = undefined;
    var lm: [QK_K / 32]u8 = undefined;
    var sw: [QK_K / 32]f32 = undefined;
    var mins: [QK_K / 32]f32 = undefined;
    var scales: [QK_K / 32]f32 = undefined;

    for (0..nb) |i| {
        fitScalesKImpl(31, x, i, quant_weights, &l_buf, &scales, &mins, &sw);

        const d_block = h.makeQpQuants(QK_K / 32, 63, &scales, &ls, &sw);
        const m_block = h.makeQpQuants(QK_K / 32, 63, &mins, &lm, &sw);

        @memset(&y[i].scales, 0);
        for (0..QK_K / 32) |j| {
            // This one *does* clamp, where q4_K's impl does not. The C is
            // inconsistent between the two; both are reproduced as written.
            packScalePair(&y[i].scales, j, @min(63, ls[j]), @min(63, lm[j]));
        }
        y[i].d = fp16(d_block);
        y[i].dmin = fp16(m_block);

        requantizeK(31, x, &y[i].scales, unfp16(y[i].d), unfp16(y[i].dmin), &l_buf);

        var qh: [*]u8 = &y[i].qh;
        var ql: [*]u8 = &y[i].qs;
        @memset(y[i].qh[0..], 0);

        var m1: u8 = 1;
        var m2: u8 = 2;
        var n: usize = 0;
        while (n < QK_K) : (n += 64) {
            for (0..32) |j| {
                var l1: i32 = l_buf[n + j];
                if (l1 > 15) {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                var l2: i32 = l_buf[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[j] = @as(u8, @intCast(l1)) | (@as(u8, @intCast(l2)) << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

/// One sub-block's scale/min pair into the packed 6-bit layout.
inline fn packScalePair(scales_out: [*]u8, j: usize, ls: u8, lm: u8) void {
    if (j < 4) {
        scales_out[j] = ls;
        scales_out[j + 4] = lm;
    } else {
        scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        scales_out[j - 4] |= (ls >> 4) << 6;
        scales_out[j - 0] |= (lm >> 4) << 6;
    }
}

/// The requantize-against-the-rounded-scales pass `q4_K` and `q5_K` share.
inline fn requantizeK(comptime nmax: i32, x: [*]const f32, scales: [*]const u8, d_all: f32, dmin_all: f32, l_buf: [*]u8) void {
    for (0..QK_K / 32) |j| {
        var sc: u8 = undefined;
        var m: u8 = undefined;
        h.getScaleMinK4(j, scales, &sc, &m);
        const d = d_all * @as(f32, @floatFromInt(sc));
        if (d == 0) continue;
        const dm = dmin_all * @as(f32, @floatFromInt(m));
        for (0..32) |ii| {
            var l = h.nearestInt((x[32 * j + ii] + dm) / d);
            l = @max(0, @min(nmax, l));
            l_buf[32 * j + ii] = @intCast(l);
        }
    }
}

/// Ports `quantize_row_q6_K_impl` (ggml-quants.c:1970 @c1d0e7a00).
///
/// Note it passes the raw importance matrix straight to `makeQxQuants` as the
/// weighting, rather than the `qw * sqrt(sigma2 + x*x)` the others build. The
/// C has the alternative commented out just above.
fn q6_KImpl(x_in: [*]const f32, y: [*]blocks.Q6_K, n_per_row: i64, quant_weights: ?[*]const f32) void {
    const nb: usize = @intCast(@divExact(n_per_row, QK_K));
    var x = x_in;

    var l_buf: [QK_K]i8 = undefined;
    var scales: [QK_K / 16]f32 = undefined;

    for (0..nb) |i| {
        var max_scale: f32 = 0;
        var max_abs_scale: f32 = 0;

        for (0..QK_K / 16) |ib| {
            const scale = if (quant_weights) |qw_all|
                h.makeQxQuants(16, 32, x + 16 * ib, @ptrCast(&l_buf[16 * ib]), 1, qw_all + QK_K * i + 16 * ib)
            else
                h.makeQxQuants(16, 32, x + 16 * ib, @ptrCast(&l_buf[16 * ib]), 1, null);
            scales[ib] = scale;
            const abs_scale = @abs(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < h.group_max_eps) {
            @memset(std.mem.asBytes(&y[i]), 0);
            y[i].d = fp16(0.0);
            x += QK_K;
            continue;
        }

        const iscale = -128.0 / max_scale;
        y[i].d = fp16(1 / iscale);
        for (0..QK_K / 16) |ib| {
            y[i].scales[ib] = @intCast(@min(127, h.nearestInt(iscale * scales[ib])));
        }

        for (0..QK_K / 16) |j| {
            const d = unfp16(y[i].d) * @as(f32, @floatFromInt(y[i].scales[j]));
            if (d == 0) continue;
            for (0..16) |ii| {
                var l = h.nearestInt(x[16 * j + ii] / d);
                l = @max(-32, @min(31, l));
                l_buf[16 * j + ii] = @intCast(l + 32);
            }
        }

        var ql: [*]u8 = &y[i].ql;
        var qh: [*]u8 = &y[i].qh;
        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                const q1: u8 = @as(u8, @bitCast(l_buf[j + l + 0])) & 0xF;
                const q2: u8 = @as(u8, @bitCast(l_buf[j + l + 32])) & 0xF;
                const q3: u8 = @as(u8, @bitCast(l_buf[j + l + 64])) & 0xF;
                const q4: u8 = @as(u8, @bitCast(l_buf[j + l + 96])) & 0xF;
                ql[l + 0] = q1 | (q3 << 4);
                ql[l + 32] = q2 | (q4 << 4);
                qh[l] = (@as(u8, @bitCast(l_buf[j + l])) >> 4) |
                    ((@as(u8, @bitCast(l_buf[j + l + 32])) >> 4) << 2) |
                    ((@as(u8, @bitCast(l_buf[j + l + 64])) >> 4) << 4) |
                    ((@as(u8, @bitCast(l_buf[j + l + 96])) >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

comptime {
    @export(&implChunk(blocks.Q2_K, QK_K, k_quants.quantize_row_q2_K_ref, q2_KImpl), .{ .name = "quantize_q2_K" });
    @export(&implChunk(blocks.Q3_K, QK_K, k_quants.quantize_row_q3_K_ref, q3_KImpl), .{ .name = "quantize_q3_K" });
    @export(&implChunk(blocks.Q4_K, QK_K, k_quants.quantize_row_q4_K_ref, q4_KImpl), .{ .name = "quantize_q4_K" });
    @export(&implChunk(blocks.Q5_K, QK_K, k_quants.quantize_row_q5_K_ref, q5_KImpl), .{ .name = "quantize_q5_K" });
    @export(&implChunk(blocks.Q6_K, QK_K, k_quants.quantize_row_q6_K_ref, q6_KImpl), .{ .name = "quantize_q6_K" });
}

// -----------------------------------------------------------------------------
// Unit Tests

const t = @import("testing.zig");

test {
    std.testing.refAllDecls(@This());
}

/// The exported chunk symbols are generated in `comptime` blocks, so they have
/// no Zig-visible names. Reaching them as the C does is also the stronger
/// test: a mis-wired `@export` name shows up here rather than at the swap.
const ChunkFn = *const fn ([*c]const f32, ?*anyopaque, i64, i64, [*c]const f32) callconv(.c) usize;

extern fn quantize_q4_0(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q4_1(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q5_0(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q5_1(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q8_0(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_mxfp4(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_nvfp4(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_tq1_0(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_tq2_0(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q2_K(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q3_K(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q4_K(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q5_K(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;
extern fn quantize_q6_K(src: [*c]const f32, dst: ?*anyopaque, nrow: i64, n_per_row: i64, qw: [*c]const f32) usize;

/// Checks one format's chunk entry point, with and without an importance
/// matrix, on every input pattern.
fn checkChunk(comptime name: []const u8, chunk: ChunkFn) !void {
    for (t.all_patterns) |pattern| {
        const g = t.find(name, pattern);

        var src: [t.n_elem]f32 = undefined;
        var imatrix: [t.n_per_row]f32 = undefined;
        t.fillSrc(pattern, &src);
        t.fillImatrix(&imatrix);

        var buf: [t.n_elem * 4]u8 align(16) = undefined;

        if (g.chunk) |want| {
            @memset(&buf, 0);
            const n = chunk(&src, &buf, t.n_rows, t.n_per_row, null);
            std.testing.expectEqual(want, t.fnv(buf[0..n])) catch |e| {
                std.debug.print("{s}: quantize_* differs on '{s}'\n", .{ name, pattern.name() });
                return e;
            };
        }

        if (g.chunk_imatrix) |want| {
            @memset(&buf, 0);
            const n = chunk(&src, &buf, t.n_rows, t.n_per_row, &imatrix);
            std.testing.expectEqual(want, t.fnv(buf[0..n])) catch |e| {
                std.debug.print("{s}: quantize_* with imatrix differs on '{s}'\n", .{ name, pattern.name() });
                return e;
            };
        }
    }
}

test "the legacy chunk entry points match the C" {
    try checkChunk("Q4_0", quantize_q4_0);
    try checkChunk("Q4_1", quantize_q4_1);
    try checkChunk("Q5_0", quantize_q5_0);
    try checkChunk("Q5_1", quantize_q5_1);
    try checkChunk("Q8_0", quantize_q8_0);
    try checkChunk("MXFP4", quantize_mxfp4);
    try checkChunk("NVFP4", quantize_nvfp4);
}

test "the ternary chunk entry points match the C" {
    try checkChunk("TQ1_0", quantize_tq1_0);
    try checkChunk("TQ2_0", quantize_tq2_0);
}

test "the K-quant chunk entry points match the C" {
    try checkChunk("Q2_K", quantize_q2_K);
    try checkChunk("Q3_K", quantize_q3_K);
    try checkChunk("Q4_K", quantize_q4_K);
    try checkChunk("Q5_K", quantize_q5_K);
    try checkChunk("Q6_K", quantize_q6_K);
}

test "a chunk entry point reports the byte count its type implies" {
    // Independent of the checksums: a wrong return value would silently
    // truncate or overrun whatever the caller writes next.
    var src: [t.n_elem]f32 = undefined;
    t.fillSrc(.random, &src);
    var buf: [t.n_elem * 4]u8 align(16) = undefined;

    inline for (.{
        .{ "Q4_0", quantize_q4_0 },
        .{ "Q6_K", quantize_q6_K },
        .{ "TQ2_0", quantize_tq2_0 },
    }) |case| {
        const g = t.find(case[0], .random);
        const n = case[1](&src, &buf, t.n_rows, t.n_per_row, null);
        try std.testing.expectEqual(g.row_size * t.n_rows, n);
    }
}
