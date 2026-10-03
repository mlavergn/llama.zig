//! `out_prod`, `scale` and `set`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # `out_prod` accumulates through `ggml_vec_mad_f32`
//!
//! Every `dst` element is built up one fused multiply-add at a time, in the
//! order the C's tiling visits `i01`. `GGML_VEC_MAD_UNROLL` is 32, so the
//! `#if GGML_VEC_MAD_UNROLL > 2` arm compiles: whole groups of 32 rows go
//! through `ggml_vec_mad_f32_unroll` and the remainder through
//! `ggml_vec_mad_f32`. Both fold one row at a time in ascending order, so the
//! summation order is the same either way — but the grouping is kept, because
//! it is the C's and costs nothing.
//!
//! # Accelerate is live in `scale`
//!
//! `ggml_vec_scale_f32` and `ggml_vec_mad1_f32` both take their
//! `GGML_USE_ACCELERATE` arms on this target; see `vecinline.zig`.
//!
//! # Loop index names
//!
//! `i1`, `i2`, `i3`, `i01`…`i13` are Zig integer type names. Renamed `j1`,
//! `j2`, `j3`, `j01`…`j13`, digit for digit — the `cpu/mulmat.zig`
//! convention. Do not renumber them.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");
const threading = @import("../threading.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

/// Byte offset `i*nb`; see `common.byteOff`.
const at = common.byteOff;

/// Zeroes `dst` from thread 0 and waits for every thread — the prologue the
/// three `out_prod` kernels share verbatim.
fn zeroDstAndSync(params: *const ComputeParams, dst: *Tensor, l: common.BinaryLocals) void {
    if (params.ith == 0) {
        vec.set_f32(l.ne0 * l.ne1 * l.ne2 * l.ne3, @ptrCast(@alignCast(dst.data.?)), 0);
    }
    threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
}

/// Ports `ggml_compute_forward_out_prod_f32` (ops.cpp:4238 @c1d0e7a00), the
/// `GGML_VEC_MAD_UNROLL > 2` arm.
fn outProdF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = common.BinaryLocals.of(src0, src1, dst);

    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    impl.assert(l.ne0 == l.ne00, "ne0 == ne00");
    impl.assert(l.ne1 == l.ne10, "ne1 == ne10");
    impl.assert(l.ne2 == l.ne12, "ne2 == ne12");
    impl.assert(l.ne3 == l.ne13, "ne3 == ne13");

    impl.assert(@rem(l.ne2, l.ne02) == 0, "ne2 % ne02 == 0");
    impl.assert(@rem(l.ne3, l.ne03) == 0, "ne3 % ne03 == 0");

    // we don't support permuted src0 or src1
    impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");

    // dst cannot be transposed or permuted
    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    zeroDstAndSync(params, dst, l);

    // parallelize by last three dimensions

    // total rows in dst
    const nr = l.ne1 * l.ne2 * l.ne3;

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    // block-tiling attempt
    const blck_0: i64 = @max(@as(i64, vec.mad_unroll), 32);
    const blck_1: i64 = 16;

    // dps == dst per src0, used for group query attention
    const dps2 = @divTrunc(l.ne2, l.ne02);
    const dps3 = @divTrunc(l.ne3, l.ne03);

    const s0b: [*]const u8 = @ptrCast(src0.data.?);
    const s1b: [*]const u8 = @ptrCast(src1.data.?);
    const db: [*]u8 = @ptrCast(dst.data.?);

    const unroll: i64 = vec.mad_unroll;

    var bir = ir0;
    while (bir < ir1) : (bir += blck_1) {
        const bir1 = @min(bir + blck_1, ir1);
        var bi01: i64 = 0;
        while (bi01 < l.ne01) : (bi01 += blck_0) {
            const bne01 = @min(bi01 + blck_0, l.ne01);
            var ir = bir;
            while (ir < bir1) : (ir += 1) {
                // dst indices
                const j3 = @divTrunc(ir, l.ne2 * l.ne1);
                const j2 = @divTrunc(ir - j3 * l.ne2 * l.ne1, l.ne1);
                const j1 = ir - j3 * l.ne2 * l.ne1 - j2 * l.ne1;

                const j02 = @divTrunc(j2, dps2);
                const j03 = @divTrunc(j3, dps3);

                //const int64_t i10 = i1;
                const j12 = j2;
                const j13 = j3;

                const d: [*]f32 = @ptrCast(@alignCast(db + at(j1, l.nb1) + at(j2, l.nb2) + at(j3, l.nb3)));

                const bne01_unroll = bne01 - @rem(bne01, unroll);
                var j01 = bi01;
                while (j01 < bne01_unroll) : (j01 += unroll) {
                    const j11 = j01;

                    const s0: [*]const f32 = @ptrCast(@alignCast(s0b + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03)));
                    const s1: [*]const f32 = @ptrCast(@alignCast(s1b + at(j1, l.nb10) + at(j11, l.nb11) + at(j12, l.nb12) + at(j13, l.nb13)));

                    vec.mad_f32_unroll(l.ne0, @intCast(l.nb01), @intCast(l.nb11), d, s0, s1);
                }
                j01 = bne01_unroll;
                while (j01 < bne01) : (j01 += 1) {
                    const j11 = j01;

                    const s0: [*]const f32 = @ptrCast(@alignCast(s0b + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03)));
                    const s1: [*]const f32 = @ptrCast(@alignCast(s1b + at(j1, l.nb10) + at(j11, l.nb11) + at(j12, l.nb12) + at(j13, l.nb13)));

                    vec.mad_f32(l.ne0, d, s0, s1[0]);
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_out_prod_q_f32` (ops.cpp:4359 @c1d0e7a00) and
/// `ggml_compute_forward_out_prod_f16_f32` (ops.cpp:4448 @c1d0e7a00).
///
/// The two differ in how a `src0` row is widened into the per-thread scratch
/// row — the type's `to_float`, or `ggml_fp16_to_fp32_row` — and in their
/// type asserts. The loop that follows is the same `ggml_vec_mad_f32` per
/// `i01`, so they are one function over a comptime flag.
fn outProdWiden(comptime is_f16: bool, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = common.BinaryLocals.of(src0, src1, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const @"type" = src0.type;
    const dequantize_row_q = if (is_f16) undefined else c.ggml_get_type_traits(@"type").*.to_float.?;

    if (is_f16) {
        impl.assert(src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F16");
        impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
        impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    }

    impl.assert(l.ne02 == l.ne12, "ne02 == ne12");
    impl.assert(l.ne03 == l.ne13, "ne03 == ne13");
    impl.assert(l.ne2 == l.ne12, "ne2  == ne12");
    impl.assert(l.ne3 == l.ne13, "ne3  == ne13");

    if (is_f16) {
        impl.assert(l.nb00 == @sizeOf(c.ggml_fp16_t), "nb00 == sizeof(ggml_fp16_t)");
    } else {
        // we don't support permuted src0 dim0
        impl.assert(l.nb00 == c.ggml_type_size(@"type"), "nb00 == ggml_type_size(type)");
    }

    // dst dim0 cannot be transposed or permuted
    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");

    impl.assert(l.ne0 == l.ne00, "ne0 == ne00");
    impl.assert(l.ne1 == l.ne10, "ne1 == ne10");
    impl.assert(l.ne2 == l.ne02, "ne2 == ne02");
    impl.assert(l.ne3 == l.ne03, "ne3 == ne03");

    zeroDstAndSync(params, dst, l);

    // total rows in dst
    const nr = l.ne1 * l.ne2 * l.ne3;

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    const wdata = wbase + (@as(usize, @intCast(l.ne0)) + common.cache_line_size_f32) * @as(usize, @intCast(ith));

    const s0b: [*]const u8 = @ptrCast(src0.data.?);
    const s1b: [*]const u8 = @ptrCast(src1.data.?);
    const db: [*]u8 = @ptrCast(dst.data.?);

    var ir = ir0;
    while (ir < ir1) : (ir += 1) {
        // dst indices
        const j3 = @divTrunc(ir, l.ne2 * l.ne1);
        const j2 = @divTrunc(ir - j3 * l.ne2 * l.ne1, l.ne1);
        const j1 = ir - j3 * l.ne2 * l.ne1 - j2 * l.ne1;

        const j02 = j2;
        const j03 = j3;

        //const int64_t i10 = i1;
        const j12 = j2;
        const j13 = j3;

        const d: [*]f32 = @ptrCast(@alignCast(db + at(j1, l.nb1) + at(j2, l.nb2) + at(j3, l.nb3)));

        var j01: i64 = 0;
        while (j01 < l.ne01) : (j01 += 1) {
            const j11 = j01;

            const s0 = s0b + at(j01, l.nb01) + at(j02, l.nb02) + at(j03, l.nb03);
            const s1: [*]const f32 = @ptrCast(@alignCast(s1b + at(j1, l.nb10) + at(j11, l.nb11) + at(j12, l.nb12) + at(j13, l.nb13)));

            if (is_f16) {
                c.ggml_fp16_to_fp32_row(@ptrCast(@alignCast(s0)), wdata, l.ne0);
            } else {
                dequantize_row_q(s0, wdata, l.ne0);
            }
            vec.mad_f32(l.ne0, d, wdata, s1[0]);
        }
    }
}

/// Ports `ggml_compute_forward_out_prod` (ops.cpp:4512 @c1d0e7a00).
pub export fn ggml_compute_forward_out_prod(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_Q1_0,
        c.GGML_TYPE_Q2_0,
        c.GGML_TYPE_Q4_0,
        c.GGML_TYPE_Q4_1,
        c.GGML_TYPE_Q5_0,
        c.GGML_TYPE_Q5_1,
        c.GGML_TYPE_Q8_0,
        c.GGML_TYPE_MXFP4,
        c.GGML_TYPE_NVFP4,
        c.GGML_TYPE_Q2_K,
        c.GGML_TYPE_Q3_K,
        c.GGML_TYPE_Q4_K,
        c.GGML_TYPE_Q5_K,
        c.GGML_TYPE_Q6_K,
        c.GGML_TYPE_TQ1_0,
        c.GGML_TYPE_TQ2_0,
        c.GGML_TYPE_IQ2_XXS,
        c.GGML_TYPE_IQ2_XS,
        c.GGML_TYPE_IQ3_XXS,
        c.GGML_TYPE_IQ1_S,
        c.GGML_TYPE_IQ1_M,
        c.GGML_TYPE_IQ4_NL,
        c.GGML_TYPE_IQ4_XS,
        c.GGML_TYPE_IQ3_S,
        c.GGML_TYPE_IQ2_S,
        => outProdWiden(false, params, dst),
        c.GGML_TYPE_F16 => outProdWiden(true, params, dst),
        c.GGML_TYPE_F32 => outProdF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_scale_f32` (ops.cpp:4564 @c1d0e7a00).
///
/// The bias path reads `src0` at `i1*nb1` — `dst`'s row stride, where the
/// scale-only path uses `nb01`. The two agree for the contiguous tensors the
/// asserts require, so it is harmless, but it is the C's and is kept.
fn scaleF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");

    const s = impl.getOpParamsF32(dst, 0); // scale factor
    const b = impl.getOpParamsF32(dst, 1); // bias

    const ith: c_int = params.ith;
    const nth: c_int = params.nth;

    const nc: c_int = @intCast(src0.ne[0]);
    const nr: c_int = @intCast(c.ggml_nrows(src0));

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const nb01 = src0.nb[1];

    const nb1 = dst.nb[1];

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    if (b == 0.0) {
        var j1 = ir0;
        while (j1 < ir1) : (j1 += 1) {
            if (dst.data != src0.data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                const n = @as(usize, @intCast(nc)) * @sizeOf(f32);
                @memcpy((dd + at(j1, nb1))[0..n], (s0 + at(j1, nb01))[0..n]);
            }
            vec.scale_f32(nc, @ptrCast(@alignCast(dd + at(j1, nb1))), s);
        }
    } else {
        var j1 = ir0;
        while (j1 < ir1) : (j1 += 1) {
            vec.mad1_f32(
                nc,
                @ptrCast(@alignCast(dd + at(j1, nb1))),
                @ptrCast(@alignCast(s0 + at(j1, nb1))),
                s,
                b,
            );
        }
    }
}

/// Ports `ggml_compute_forward_scale` (ops.cpp:4616 @c1d0e7a00).
pub export fn ggml_compute_forward_scale(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => scaleF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_set_f32` (ops.cpp:4636 @c1d0e7a00) and
/// `ggml_compute_forward_set_i32` (ops.cpp:4707 @c1d0e7a00).
///
/// Identical but for the element type and the `ggml_vec_cpy_*` they call.
fn setT(comptime T: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");
    impl.assert(c.ggml_is_contiguous(dst) and c.ggml_is_contiguous(src0), "ggml_is_contiguous(dst) && ggml_is_contiguous(src0)");

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    const op: [*]const i32 = @ptrCast(&dst.op_params);
    const nb1: usize = @intCast(op[0]);
    const nb2: usize = @intCast(op[1]);
    const nb3: usize = @intCast(op[2]);
    const offset: usize = @intCast(op[3]);
    const inplace: bool = op[4] != 0;

    if (!inplace) {
        if (params.ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            const n = c.ggml_nbytes(dst);
            const d: [*]u8 = @ptrCast(dst.data.?);
            const s: [*]const u8 = @ptrCast(src0.data.?);
            @memcpy(d[0..n], s[0..n]);
        }
        threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
    }

    const ith: c_int = params.ith;
    const nth: c_int = params.nth;

    const nr: c_int = @intCast(c.ggml_nrows(src1));
    const nc: c_int = @intCast(src1.ne[0]);

    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const nb10 = src1.nb[0];
    const nb11 = src1.nb[1];
    const nb12 = src1.nb[2];
    const nb13 = src1.nb[3];

    // src0 and dst as viewed during set
    const nb0: usize = c.ggml_element_size(src0);

    const im0: usize = @intCast(if (ne10 == 0) 0 else ne10 - 1);
    const im1: usize = @intCast(if (ne11 == 0) 0 else ne11 - 1);
    const im2: usize = @intCast(if (ne12 == 0) 0 else ne12 - 1);
    const im3: usize = @intCast(if (ne13 == 0) 0 else ne13 - 1);

    impl.assert(offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= c.ggml_nbytes(dst), "offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst)");

    impl.assert(nb10 == @sizeOf(T), "nb10 == sizeof(T)");

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir = ir0;
    while (ir < ir1) : (ir += 1) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const j3: i64 = @divTrunc(ir, ne12 * ne11);
        const j2: i64 = @divTrunc(ir - j3 * ne12 * ne11, ne11);
        const j1: i64 = ir - j3 * ne12 * ne11 - j2 * ne11;

        const y: [*]T = @ptrCast(@alignCast(dd + at(j3, nb3) + at(j2, nb2) + at(j1, nb1) + offset));
        const x: [*]const T = @ptrCast(@alignCast(s1 + at(j3, nb13) + at(j2, nb12) + at(j1, nb11)));
        if (T == f32) vec.cpy_f32(nc, y, x) else vec.cpy_i32(nc, y, x);
    }
}

/// Ports `ggml_compute_forward_set` (ops.cpp:4778 @c1d0e7a00).
pub export fn ggml_compute_forward_set(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => setT(f32, params, dst),
        c.GGML_TYPE_I32 => setT(i32, params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
