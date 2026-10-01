//! `sum`, `cumsum`, `sum_rows`, `mean`, `argmax` and `count_equal`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Two sums that are not the same sum
//!
//! `sum` accumulates through `ggml_vec_sum_f32_ggf`, which sums in `double`.
//! `sum_rows` and `mean` accumulate through `ggml_vec_sum_f32`, which on this
//! target is `vDSP_sve` and sums in `f32`. They look interchangeable in the C
//! and are not; see `vecinline.zig`.
//!
//! # Loop index names
//!
//! `i1`, `i2`, `i3`, `i01`, `i02`, `i03` are Zig integer type names. Renamed
//! `j1`, `j2`, `j3`, `j01`, `j02`, `j03`, digit for digit.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");
const threading = @import("../threading.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

/// Ports `ggml_compute_forward_sum_f32` (ops.cpp:1284 @c1d0e7a00).
fn sumF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(c.ggml_is_scalar(dst), "ggml_is_scalar(dst)");
    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const nb01 = src0.nb[1];
    const nb02 = src0.nb[2];
    const nb03 = src0.nb[3];

    var sum: f64 = 0;
    var row_sum: f64 = 0;

    const s0: [*]const u8 = @ptrCast(src0.data.?);

    var j03: i64 = 0;
    while (j03 < ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < ne02) : (j02 += 1) {
            var j01: i64 = 0;
            while (j01 < ne01) : (j01 += 1) {
                vec.sum_f32_ggf(ne00, &row_sum, @ptrCast(@alignCast(s0 +
                    @as(usize, @intCast(j01)) * nb01 + @as(usize, @intCast(j02)) * nb02 + @as(usize, @intCast(j03)) * nb03)));
                sum += row_sum;
            }
        }
    }
    @as([*]f32, @ptrCast(@alignCast(dst.data.?)))[0] = @floatCast(sum);
}

/// Ports `ggml_compute_forward_sum_f16` and `ggml_compute_forward_sum_bf16`
/// (ops.cpp:1316, 1349 @c1d0e7a00).
///
/// The two differ only in the element type and in which `_ggf` helper they
/// call, so they are one function over a comptime type. Both accumulate in
/// `f32`, unlike the `f32` variant above, which accumulates in `double`.
fn sumNarrow(comptime T: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(c.ggml_is_scalar(dst), "ggml_is_scalar(dst)");
    impl.assert(src0.nb[0] == @sizeOf(T), "src0->nb[0] == sizeof(T)");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const nb01 = src0.nb[1];
    const nb02 = src0.nb[2];
    const nb03 = src0.nb[3];

    var sum: f32 = 0;
    var row_sum: f32 = 0;

    const s0: [*]const u8 = @ptrCast(src0.data.?);

    var j03: i64 = 0;
    while (j03 < ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < ne02) : (j02 += 1) {
            var j01: i64 = 0;
            while (j01 < ne01) : (j01 += 1) {
                const row: [*]const T = @ptrCast(@alignCast(s0 +
                    @as(usize, @intCast(j01)) * nb01 + @as(usize, @intCast(j02)) * nb02 + @as(usize, @intCast(j03)) * nb03));
                if (T == c.ggml_fp16_t) vec.sum_f16_ggf(ne00, &row_sum, row) else vec.sum_bf16_ggf(ne00, &row_sum, row);
                sum += row_sum;
            }
        }
    }
    @as([*]T, @ptrCast(@alignCast(dst.data.?)))[0] = common.fromF32(T, sum);
}

/// Ports `ggml_compute_forward_sum` (ops.cpp:1382 @c1d0e7a00).
pub export fn ggml_compute_forward_sum(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => sumF32(params, dst),
        c.GGML_TYPE_F16 => sumNarrow(c.ggml_fp16_t, params, dst),
        c.GGML_TYPE_BF16 => sumNarrow(c.ggml_bf16_t, params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_cumsum_f32` (ops.cpp:1410 @c1d0e7a00).
fn cumsumF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(dst.nb[0] == @sizeOf(f32), "dst->nb[0] == sizeof(float)");

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.ne0 == l.ne00, "ne0 == ne00");
    impl.assert(l.ne1 == l.ne01, "ne1 == ne01");
    impl.assert(l.ne2 == l.ne02, "ne2 == ne02");
    impl.assert(l.ne3 == l.ne03, "ne3 == ne03");

    const ir0, const ir1 = common.getThreadRange(params, src0);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne02 * l.ne01, l.ne01);
        const j01 = ir - j03 * l.ne02 * l.ne01 - j02 * l.ne01;

        const src_row: [*]const f32 = @ptrCast(@alignCast(s0 +
            @as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03));
        const dst_row: [*]f32 = @ptrCast(@alignCast(dd +
            @as(usize, @intCast(j01)) * l.nb1 + @as(usize, @intCast(j02)) * l.nb2 + @as(usize, @intCast(j03)) * l.nb3));

        vec.cumsum_f32(l.ne00, dst_row, src_row);
    }
}

/// Ports `ggml_compute_forward_cumsum` (ops.cpp:1440 @c1d0e7a00).
pub export fn ggml_compute_forward_cumsum(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => cumsumF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_sum_rows_f32` (ops.cpp:1460 @c1d0e7a00).
fn sumRowsF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(dst.nb[0] == @sizeOf(f32), "dst->nb[0] == sizeof(float)");

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.ne0 == 1, "ne0 == 1");
    impl.assert(l.ne1 == l.ne01, "ne1 == ne01");
    impl.assert(l.ne2 == l.ne02, "ne2 == ne02");
    impl.assert(l.ne3 == l.ne03, "ne3 == ne03");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var j3: i64 = 0;
    while (j3 < l.ne03) : (j3 += 1) {
        var j2: i64 = 0;
        while (j2 < l.ne02) : (j2 += 1) {
            var j1: i64 = 0;
            while (j1 < l.ne01) : (j1 += 1) {
                const src_row: [*]const f32 = @ptrCast(@alignCast(s0 +
                    @as(usize, @intCast(j1)) * l.nb01 + @as(usize, @intCast(j2)) * l.nb02 + @as(usize, @intCast(j3)) * l.nb03));
                const dst_row: [*]f32 = @ptrCast(@alignCast(dd +
                    @as(usize, @intCast(j1)) * l.nb1 + @as(usize, @intCast(j2)) * l.nb2 + @as(usize, @intCast(j3)) * l.nb3));
                var row_sum: f32 = 0;
                vec.sum_f32(l.ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

/// Ports `ggml_compute_forward_sum_rows` (ops.cpp:1493 @c1d0e7a00).
pub export fn ggml_compute_forward_sum_rows(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => sumRowsF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_mean_f32` (ops.cpp:1513 @c1d0e7a00).
fn meanF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.ne0 == 1, "ne0 == 1");
    impl.assert(l.ne1 == l.ne01, "ne1 == ne01");
    impl.assert(l.ne2 == l.ne02, "ne2 == ne02");
    impl.assert(l.ne3 == l.ne03, "ne3 == ne03");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            var j01: i64 = 0;
            while (j01 < l.ne01) : (j01 += 1) {
                const out: *f32 = @ptrCast(@alignCast(dd +
                    @as(usize, @intCast(j01)) * l.nb1 + @as(usize, @intCast(j02)) * l.nb2 + @as(usize, @intCast(j03)) * l.nb3));
                vec.sum_f32(l.ne00, out, @ptrCast(@alignCast(s0 +
                    @as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03)));
                out.* /= @floatFromInt(l.ne00);
            }
        }
    }
}

/// Ports `ggml_compute_forward_mean` (ops.cpp:1550 @c1d0e7a00).
pub export fn ggml_compute_forward_mean(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => meanF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_argmax_f32` (ops.cpp:1570 @c1d0e7a00).
fn argmaxF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(dst.nb[0] == @sizeOf(f32), "dst->nb[0] == sizeof(float)");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const nb01 = src0.nb[1];
    const nb0 = dst.nb[0];

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var j1: i64 = 0;
    while (j1 < ne01) : (j1 += 1) {
        const src: [*]const f32 = @ptrCast(@alignCast(s0 + @as(usize, @intCast(j1)) * nb01));
        const dst_: [*]i32 = @ptrCast(@alignCast(dd + @as(usize, @intCast(j1)) * nb0));
        var v: i32 = 0;
        vec.argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

/// Ports `ggml_compute_forward_argmax` (ops.cpp:1598 @c1d0e7a00).
pub export fn ggml_compute_forward_argmax(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => argmaxF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_count_equal_i32` (ops.cpp:1618 @c1d0e7a00).
fn countEqualI32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = common.BinaryLocals.of(src0, src1, dst);

    impl.assert(src0.type == c.GGML_TYPE_I32, "src0->type == GGML_TYPE_I32");
    impl.assert(src1.type == c.GGML_TYPE_I32, "src1->type == GGML_TYPE_I32");
    impl.assert(c.ggml_are_same_shape(src0, src1), "ggml_are_same_shape(src0, src1)");
    impl.assert(c.ggml_is_scalar(dst), "ggml_is_scalar(dst)");
    impl.assert(dst.type == c.GGML_TYPE_I64, "dst->type == GGML_TYPE_I64");

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const sums: [*]i64 = @ptrCast(@alignCast(params.wdata.?));
    var sum_thread: i64 = 0;

    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const u8 = @ptrCast(src1.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // The C divides by `ne03` and `ne02` here where the pattern every
        // other kernel follows would use `ne02*ne01` and `ne01` -- see `i02`
        // and `i01` (ops.cpp:1650, 1651 @c1d0e7a00). Reproduced verbatim -- it only
        // coincides with the usual form when the trailing dimensions are 1,
        // which is the shape this op is used at, and the reference's
        // behaviour is what the port is measured against.
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne03, l.ne01);
        const j01 = ir - j03 * l.ne03 - j02 * l.ne02;

        const data0 = s0 + @as(usize, @intCast(j03)) * l.nb03 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j01)) * l.nb01;
        const data1 = s1 + @as(usize, @intCast(j03)) * l.nb13 + @as(usize, @intCast(j02)) * l.nb12 + @as(usize, @intCast(j01)) * l.nb11;

        var j00: i64 = 0;
        while (j00 < l.ne00) : (j00 += 1) {
            const val0 = @as(*const i32, @ptrCast(@alignCast(data0 + @as(usize, @intCast(j00)) * l.nb00))).*;
            const val1 = @as(*const i32, @ptrCast(@alignCast(data1 + @as(usize, @intCast(j00)) * l.nb10))).*;
            sum_thread += @intFromBool(val0 == val1);
        }
    }
    if (ith != 0) {
        sums[@intCast(ith)] = sum_thread;
    }
    threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));

    if (ith != 0) return;

    var ith_other: i64 = 1;
    while (ith_other < nth) : (ith_other += 1) {
        sum_thread += sums[@intCast(ith_other)];
    }
    @as(*i64, @ptrCast(@alignCast(dst.data.?))).* = sum_thread;
}

/// Ports `ggml_compute_forward_count_equal` (ops.cpp:1678 @c1d0e7a00).
pub export fn ggml_compute_forward_count_equal(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_I32 => countEqualI32(params, dst),
        else => impl.abort("fatal error"),
    }
}
