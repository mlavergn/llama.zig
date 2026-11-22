//! `soft_max`, `soft_max_ext_back` and `clamp`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v1.0.0 (`b1b770d90`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # `MIN` and `MAX` are ternaries, not `@min` and `@max`
//!
//! `ggml-impl.h` defines them as `((a) < (b) ? (a) : (b))` and
//! `((a) > (b) ? (a) : (b))`. On a NaN the comparison is false and the
//! *second* operand wins, where `@min`/`@max` would return the non-NaN one.
//! `clamp` and the sink correction in `soft_max` are written with the
//! ternaries so a NaN input propagates — or does not — exactly as in the C.
//!
//! # `NDEBUG` blocks
//!
//! `soft_max` and `soft_max_ext_back` scan for NaN and infinity under
//! `#ifndef NDEBUG`. The release build defines `NDEBUG`, so those loops are
//! not ported; the plain `assert`s around them become `std.debug.assert`.
//!
//! # Loop index names
//!
//! `i01`, `i02`, `i03`, `i11`, `i12`, `i13` and `i1` are Zig integer type
//! names. Renamed `j01`, `j02`, `j03`, `j11`, `j12`, `j13` and `j1`, digit for
//! digit.

const std = @import("std");
const ggml = @import("../../module.zig");
const stdlib = @import("stdlib");

/// Typed pointer at a byte offset; see `common.ptr`.
/// Stride as `i64`; see `common.sz`.
/// The C's `MIN` macro (ggml-impl.h:36 @b1b770d90), over `f32`.
inline fn minF(a: f32, b: f32) f32 {
    return if (a < b) a else b;
}

/// The C's `MAX` macro (ggml-impl.h:40 @b1b770d90), over `f32`.
inline fn maxF(a: f32, b: f32) f32 {
    return if (a > b) a else b;
}

// -----------------------------------------------------------------------------
// soft_max

/// Ports `ggml_compute_forward_soft_max_f32` (ops.cpp:5709 @b1b770d90).
fn softMaxF32(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1: ?*ggml.impl.ggml_c.ggml_tensor = dst.src[1];
    const src2: ?*ggml.impl.ggml_c.ggml_tensor = dst.src[2];

    std.debug.assert(ggml.impl.ggml_c.ggml_is_contiguous(dst));
    std.debug.assert(ggml.impl.ggml_c.ggml_are_same_shape(src0, dst));

    const scale = ggml.impl.getOpParamsF32(dst, 0);
    const max_bias = ggml.impl.getOpParamsF32(dst, 1);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = ggml.cpu.defs.UnaryLocals.of(src0, dst);

    const nb11: i64 = if (src1) |t| ggml.cpu.ops.common.sz(t.nb[1]) else 1;
    const nb12: i64 = if (src1) |t| ggml.cpu.ops.common.sz(t.nb[2]) else 1;
    const nb13: i64 = if (src1) |t| ggml.cpu.ops.common.sz(t.nb[3]) else 1;

    const ne12: i64 = if (src1) |t| t.ne[2] else 1;
    const ne13: i64 = if (src1) |t| t.ne[3] else 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    const n_head: u32 = @truncate(@as(u64, @bitCast(l.ne02)));
    const n_head_log2: u32 = @as(u32, 1) << @intFromFloat(@floor(stdlib.libm.log2(@floatFromInt(n_head))));

    const m0 = stdlib.libm.powf(2.0, -(max_bias) / @as(f32, @floatFromInt(n_head_log2)));
    const m1 = stdlib.libm.powf(2.0, -(max_bias / 2.0) / @as(f32, @floatFromInt(n_head_log2)));

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    const wp = wbase + (@as(usize, @intCast(l.ne00)) + ggml.cpu.ops.common.cache_line_size_f32) * @as(usize, @intCast(ith));

    const use_f16 = if (src1) |t| t.type == ggml.impl.ggml_c.GGML_TYPE_F16 else false;

    // sinks
    const sk: ?[*]const f32 = if (src2) |t| ggml.cpu.ops.common.ptr(f32, t.data, 0) else null;

    const nc: usize = @intCast(l.ne00);

    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            var j01: i64 = ith;
            while (j01 < l.ne01) : (j01 += nth) {
                const j11 = j01;
                const j12 = @rem(j02, ne12);
                const j13 = @rem(j03, ne13);

                // ALiBi
                const h: u32 = @truncate(@as(u64, @bitCast(j02))); // head
                const slope: f32 = if (max_bias > 0.0)
                    (if (h < n_head_log2)
                        stdlib.libm.powf(m0, @floatFromInt(h + 1))
                    else
                        stdlib.libm.powf(m1, @floatFromInt(2 *% (h -% n_head_log2) +% 1)))
                else
                    1.0;

                const sp = ggml.cpu.ops.common.ptr(f32, src0.data, j01 * ggml.cpu.ops.common.sz(l.nb01) + j02 * ggml.cpu.ops.common.sz(l.nb02) + j03 * ggml.cpu.ops.common.sz(l.nb03));
                const dp = ggml.cpu.ops.common.ptr(f32, dst.data, j01 * ggml.cpu.ops.common.sz(l.nb1) + j02 * ggml.cpu.ops.common.sz(l.nb2) + j03 * ggml.cpu.ops.common.sz(l.nb3));

                ggml.cpu.ops.vecinline.cpy_f32(l.ne00, wp, sp);
                ggml.cpu.ops.vecinline.scale_f32(l.ne00, wp, scale);
                if (src1) |m| {
                    // broadcast the mask across rows
                    const moff = j11 * nb11 + j12 * nb12 + j13 * nb13;
                    // `wp[i] += slope*mask` is one expression, and clang
                    // fuses it: one rounding, not two.
                    if (use_f16) {
                        const mp_f16 = ggml.cpu.ops.common.ptr(ggml.impl.ggml_c.ggml_fp16_t, m.data, moff);
                        for (0..nc) |i| wp[i] = @mulAdd(f32, slope, ggml.impl.fp16ToFp32(mp_f16[i]), wp[i]);
                    } else {
                        const mp_f32 = ggml.cpu.ops.common.ptr(f32, m.data, moff);
                        for (0..nc) |i| wp[i] = @mulAdd(f32, slope, mp_f32[i], wp[i]);
                    }
                }

                var max: f32 = -std.math.inf(f32);
                ggml.cpu.ops.vecinline.max_f32(l.ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                if (sk) |k| {
                    max = maxF(max, k[@intCast(j02)]);
                }

                var sum: f64 = ggml.cpu.vec.ggml_vec_soft_max_f32(@intCast(l.ne00), dp, wp, max);
                // The C's `assert(sum > 0.0)` is compiled out under NDEBUG and
                // is **not** an invariant: a fully masked row has max = -inf
                // and sum = NaN, and the C carries on. `std.debug.assert`
                // would make that an optimizer assumption in ReleaseFast, so
                // it is left out rather than ported.

                if (sk) |k| {
                    sum += @as(f64, stdlib.libm.expf(k[@intCast(j02)] - max));
                }

                sum = 1.0 / sum;
                ggml.cpu.ops.vecinline.scale_f32(l.ne00, dp, @floatCast(sum));
            }
        }
    }
}

/// Ports `ggml_compute_forward_soft_max` (ops.cpp:5821 @b1b770d90).
pub fn ggml_compute_forward_soft_max(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => softMaxF32(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// soft_max_ext_back

/// Ports `ggml_compute_forward_soft_max_ext_back_f32` (ops.cpp:5842 @b1b770d90).
///
/// `dx = scale * (dy - dot(y, dy)) * y`, one element at a time, in the C's
/// order: one loop, not four passes, so `(dy - dot)*y` is not rounded
/// before scaling.
/// No product in it is added to anything, so there is nothing to contract.
fn softMaxExtBackF32(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(src1, dst), "ggml_are_same_shape(src1, dst)");

    const scale = ggml.impl.getOpParamsF32(dst, 0);
    const max_bias = ggml.impl.getOpParamsF32(dst, 1);

    ggml.impl.assert(max_bias == 0.0, "max_bias == 0.0f");

    // TODO: handle transposed/permuted matrices

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nc: i64 = src0.ne[0];
    const nr: i64 = ggml.impl.ggml_c.ggml_nrows(src0);

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    var j1: i64 = ir0;
    while (j1 < ir1) : (j1 += 1) {
        const dy = ggml.cpu.ops.common.ptr(f32, src0.data, j1 * ggml.cpu.ops.common.sz(src0.nb[1]));
        const y = ggml.cpu.ops.common.ptr(f32, src1.data, j1 * ggml.cpu.ops.common.sz(src1.nb[1]));
        const dx = ggml.cpu.ops.common.ptr(f32, dst.data, j1 * ggml.cpu.ops.common.sz(dst.nb[1]));

        // linear runtime, no additional memory
        var dot_y_dy: f32 = 0;
        ggml.cpu.vec.ggml_vec_dot_f32(@intCast(nc), &dot_y_dy, 0, y, 0, dy, 0, 1);
        for (0..@intCast(nc)) |i| {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }
    }
}

/// Ports `ggml_compute_forward_soft_max_ext_back` (ops.cpp:5925 @b1b770d90).
pub fn ggml_compute_forward_soft_max_ext_back(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => softMaxExtBackF32(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// clamp

/// Ports `ggml_compute_forward_clamp_f32` and `ggml_compute_forward_clamp_f16`
/// (ops.cpp:5945, 5981 @b1b770d90).
///
/// The two differ only in the element type, so they are one function over a
/// comptime type. `MAX(MIN(v, max), min)` is the C's macro pair, NaN
/// behaviour included: a NaN `v` makes `MIN` return `max`. Each
/// row is addressed through all three outer strides, so a non-contiguous view
/// clamps the right rows.
fn clamp(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    const min = ggml.impl.getOpParamsF32(dst, 0);
    const max = ggml.impl.getOpParamsF32(dst, 1);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const n: i64 = ggml.impl.ggml_c.ggml_nrows(src0);
    const nc: usize = @intCast(src0.ne[0]);

    // GGML_TENSOR_UNARY_OP_LOCALS
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const nb00 = src0.nb[0];
    const nb01 = src0.nb[1];
    const nb02 = src0.nb[2];
    const nb03 = src0.nb[3];
    const nb0 = dst.nb[0];
    const nb1 = dst.nb[1];
    const nb2 = dst.nb[2];
    const nb3 = dst.nb[3];

    ggml.impl.assert(nb0 == @sizeOf(T), "nb0 == sizeof(T)");
    ggml.impl.assert(nb00 == @sizeOf(T), "nb00 == sizeof(T)");

    var j: i64 = ith;
    while (j < n) : (j += nth) {
        const j1 = @rem(j, ne01);
        const j2 = @rem(@divTrunc(j, ne01), ne02);
        const j3 = @divTrunc(j, ne01 * ne02);

        const dst_ptr = ggml.cpu.ops.common.ptr(T, dst.data, j1 * ggml.cpu.ops.common.sz(nb1) + j2 * ggml.cpu.ops.common.sz(nb2) + j3 * ggml.cpu.ops.common.sz(nb3));
        const src0_ptr = ggml.cpu.ops.common.ptr(T, src0.data, j1 * ggml.cpu.ops.common.sz(nb01) + j2 * ggml.cpu.ops.common.sz(nb02) + j3 * ggml.cpu.ops.common.sz(nb03));

        for (0..nc) |i| {
            const v = ggml.cpu.ops.common.toF32(T, src0_ptr[i]);
            dst_ptr[i] = ggml.cpu.ops.common.fromF32(T, maxF(minF(v, max), min));
        }
    }
}

/// Ports `ggml_compute_forward_clamp` (ops.cpp:6018 @b1b770d90).
pub fn ggml_compute_forward_clamp(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => clamp(f32, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F16 => clamp(ggml.impl.ggml_c.ggml_fp16_t, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "MIN and MAX hand a NaN through the way the C macros do" {
    const nan = std.math.nan(f32);
    // MIN(NaN, 1): NaN < 1 is false, so the second operand.
    try std.testing.expectEqual(@as(f32, 1), minF(nan, 1));
    // MIN(1, NaN): 1 < NaN is false, so NaN.
    try std.testing.expect(std.math.isNan(minF(1, nan)));
    // clamp(NaN) to [-0.5, 0.5] is MAX(MIN(NaN, 0.5), -0.5) = 0.5.
    try std.testing.expectEqual(@as(f32, 0.5), maxF(minF(nan, 0.5), -0.5));
}
