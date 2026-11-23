//! `out_prod`, `scale` and `set`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v1.0.0 (`b1b770d90`).
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
const ggml = @import("../../module.zig");

/// Byte offset `i*nb`; see `common.byteOff`.
/// Zeroes `dst` from thread 0 and waits for every thread — the prologue the
/// three `out_prod` kernels share verbatim.
fn zeroDstAndSync(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor, l: ggml.cpu.defs.BinaryLocals) void {
    if (params.ith == 0) {
        ggml.cpu.ops.vecinline.set_f32(l.ne0 * l.ne1 * l.ne2 * l.ne3, @ptrCast(@alignCast(dst.data.?)), 0);
    }
    ggml.cpu.threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
}

/// Ports `ggml_compute_forward_out_prod_f32` (ops.cpp:4529 @b1b770d90), the
/// `GGML_VEC_MAD_UNROLL > 2` arm.
fn outProdF32(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    ggml.impl.assert(l.ne0 == l.ne00, "ne0 == ne00");
    ggml.impl.assert(l.ne1 == l.ne10, "ne1 == ne10");
    ggml.impl.assert(l.ne2 == l.ne12, "ne2 == ne12");
    ggml.impl.assert(l.ne3 == l.ne13, "ne3 == ne13");

    ggml.impl.assert(@rem(l.ne2, l.ne02) == 0, "ne2 % ne02 == 0");
    ggml.impl.assert(@rem(l.ne3, l.ne03) == 0, "ne3 % ne03 == 0");

    // we don't support permuted src0 or src1
    ggml.impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");

    // dst cannot be transposed or permuted
    ggml.impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");

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
    const blck_0: i64 = @max(@as(i64, ggml.cpu.ops.vecinline.mad_unroll), 32);
    const blck_1: i64 = 16;

    // dps == dst per src0, used for group query attention
    const dps2 = @divTrunc(l.ne2, l.ne02);
    const dps3 = @divTrunc(l.ne3, l.ne03);

    const s0b: [*]const u8 = @ptrCast(src0.data.?);
    const s1b: [*]const u8 = @ptrCast(src1.data.?);
    const db: [*]u8 = @ptrCast(dst.data.?);

    const unroll: i64 = ggml.cpu.ops.vecinline.mad_unroll;

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

                const d: [*]f32 = @ptrCast(@alignCast(db + ggml.cpu.ops.common.byteOff(j1, l.nb1) + ggml.cpu.ops.common.byteOff(j2, l.nb2) + ggml.cpu.ops.common.byteOff(j3, l.nb3)));

                const bne01_unroll = bne01 - @rem(bne01, unroll);
                var j01 = bi01;
                while (j01 < bne01_unroll) : (j01 += unroll) {
                    const j11 = j01;

                    const s0: [*]const f32 = @ptrCast(@alignCast(s0b + ggml.cpu.ops.common.byteOff(j01, l.nb01) + ggml.cpu.ops.common.byteOff(j02, l.nb02) + ggml.cpu.ops.common.byteOff(j03, l.nb03)));
                    const s1: [*]const f32 = @ptrCast(@alignCast(s1b + ggml.cpu.ops.common.byteOff(j1, l.nb10) + ggml.cpu.ops.common.byteOff(j11, l.nb11) + ggml.cpu.ops.common.byteOff(j12, l.nb12) + ggml.cpu.ops.common.byteOff(j13, l.nb13)));

                    ggml.cpu.ops.vecinline.mad_f32_unroll(l.ne0, @intCast(l.nb01), @intCast(l.nb11), d, s0, s1);
                }
                j01 = bne01_unroll;
                while (j01 < bne01) : (j01 += 1) {
                    const j11 = j01;

                    const s0: [*]const f32 = @ptrCast(@alignCast(s0b + ggml.cpu.ops.common.byteOff(j01, l.nb01) + ggml.cpu.ops.common.byteOff(j02, l.nb02) + ggml.cpu.ops.common.byteOff(j03, l.nb03)));
                    const s1: [*]const f32 = @ptrCast(@alignCast(s1b + ggml.cpu.ops.common.byteOff(j1, l.nb10) + ggml.cpu.ops.common.byteOff(j11, l.nb11) + ggml.cpu.ops.common.byteOff(j12, l.nb12) + ggml.cpu.ops.common.byteOff(j13, l.nb13)));

                    ggml.cpu.ops.vecinline.mad_f32(l.ne0, d, s0, s1[0]);
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_out_prod_q_f32` (ops.cpp:4650 @b1b770d90) and
/// `ggml_compute_forward_out_prod_f16_f32` (ops.cpp:4739 @b1b770d90).
///
/// The two differ in how a `src0` row is widened into the per-thread scratch
/// row — the type's `to_float`, or `ggml_fp16_to_fp32_row` — and in their
/// type asserts. The loop that follows is the same `ggml_vec_mad_f32` per
/// `i01`, so they are one function over a comptime flag.
fn outProdWiden(comptime is_f16: bool, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const @"type" = src0.type;
    const dequantize_row_q = if (is_f16) undefined else ggml.types.ggml_get_type_traits(@"type").*.to_float.?;

    if (is_f16) {
        ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F16");
        ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
        ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    }

    ggml.impl.assert(l.ne02 == l.ne12, "ne02 == ne12");
    ggml.impl.assert(l.ne03 == l.ne13, "ne03 == ne13");
    ggml.impl.assert(l.ne2 == l.ne12, "ne2  == ne12");
    ggml.impl.assert(l.ne3 == l.ne13, "ne3  == ne13");

    if (is_f16) {
        ggml.impl.assert(l.nb00 == @sizeOf(ggml.impl.ggml_c.ggml_fp16_t), "nb00 == sizeof(ggml_fp16_t)");
    } else {
        // we don't support permuted src0 dim0
        ggml.impl.assert(l.nb00 == ggml.types.ggml_type_size(@"type"), "nb00 == ggml_type_size(type)");
    }

    // dst dim0 cannot be transposed or permuted
    ggml.impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");

    ggml.impl.assert(l.ne0 == l.ne00, "ne0 == ne00");
    ggml.impl.assert(l.ne1 == l.ne10, "ne1 == ne10");
    ggml.impl.assert(l.ne2 == l.ne02, "ne2 == ne02");
    ggml.impl.assert(l.ne3 == l.ne03, "ne3 == ne03");

    zeroDstAndSync(params, dst, l);

    // total rows in dst
    const nr = l.ne1 * l.ne2 * l.ne3;

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    const wdata = wbase + (@as(usize, @intCast(l.ne0)) + ggml.cpu.ops.common.cache_line_size_f32) * @as(usize, @intCast(ith));

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

        const d: [*]f32 = @ptrCast(@alignCast(db + ggml.cpu.ops.common.byteOff(j1, l.nb1) + ggml.cpu.ops.common.byteOff(j2, l.nb2) + ggml.cpu.ops.common.byteOff(j3, l.nb3)));

        var j01: i64 = 0;
        while (j01 < l.ne01) : (j01 += 1) {
            const j11 = j01;

            const s0 = s0b + ggml.cpu.ops.common.byteOff(j01, l.nb01) + ggml.cpu.ops.common.byteOff(j02, l.nb02) + ggml.cpu.ops.common.byteOff(j03, l.nb03);
            const s1: [*]const f32 = @ptrCast(@alignCast(s1b + ggml.cpu.ops.common.byteOff(j1, l.nb10) + ggml.cpu.ops.common.byteOff(j11, l.nb11) + ggml.cpu.ops.common.byteOff(j12, l.nb12) + ggml.cpu.ops.common.byteOff(j13, l.nb13)));

            if (is_f16) {
                ggml.runtime.ggml_fp16_to_fp32_row(@ptrCast(@alignCast(s0)), wdata, l.ne0);
            } else {
                dequantize_row_q(s0, wdata, l.ne0);
            }
            ggml.cpu.ops.vecinline.mad_f32(l.ne0, d, wdata, s1[0]);
        }
    }
}

/// Ports `ggml_compute_forward_out_prod` (ops.cpp:4803 @b1b770d90).
pub fn ggml_compute_forward_out_prod(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0,
        ggml.impl.ggml_c.GGML_TYPE_Q2_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4,
        ggml.impl.ggml_c.GGML_TYPE_F8_E4M3,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K,
        ggml.impl.ggml_c.GGML_TYPE_TQ1_0,
        ggml.impl.ggml_c.GGML_TYPE_TQ2_0,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_M,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S,
        => outProdWiden(false, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F16 => outProdWiden(true, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F32 => outProdF32(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_scale_f32` (ops.cpp:4856 @b1b770d90).
///
/// Rows need only be contiguous within themselves, so each flat
/// row index is split into `i1`, `i2`, `i3` and every row addressed through
/// its tensor's own strides.
fn scaleF32(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(src0), "ggml_is_contiguous_rows(src0)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(dst), "ggml_is_contiguous_rows(dst)");
    ggml.impl.assert(ggml.types.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");

    const s = ggml.impl.getOpParamsF32(dst, 0); // scale factor
    const b = ggml.impl.getOpParamsF32(dst, 1); // bias

    const ith: c_int = params.ith;
    const nth: c_int = params.nth;

    const nc: c_int = @intCast(src0.ne[0]);
    const nr: c_int = @intCast(ggml.types.ggml_nrows(src0));

    const l = ggml.cpu.defs.UnaryLocals.of(src0, dst);

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var i = ir0;
    while (i < ir1) : (i += 1) {
        const j1 = @rem(@as(i64, i), l.ne01);
        const j2 = @rem(@divTrunc(@as(i64, i), l.ne01), l.ne02);
        const j3 = @divTrunc(@as(i64, i), l.ne01 * l.ne02);

        const y = dd + ggml.cpu.ops.common.byteOff(j1, l.nb1) + ggml.cpu.ops.common.byteOff(j2, l.nb2) + ggml.cpu.ops.common.byteOff(j3, l.nb3);
        const x = s0 + ggml.cpu.ops.common.byteOff(j1, l.nb01) + ggml.cpu.ops.common.byteOff(j2, l.nb02) + ggml.cpu.ops.common.byteOff(j3, l.nb03);

        if (b == 0.0) {
            if (dst.data != src0.data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                const n = @as(usize, @intCast(nc)) * @sizeOf(f32);
                @memcpy(y[0..n], x[0..n]);
            }
            ggml.cpu.ops.vecinline.scale_f32(nc, @ptrCast(@alignCast(y)), s);
        } else {
            ggml.cpu.ops.vecinline.mad1_f32(nc, @ptrCast(@alignCast(y)), @ptrCast(@alignCast(x)), s, b);
        }
    }
}

/// Ports `ggml_compute_forward_scale_bf16` (ops.cpp:4908 @b1b770d90).
///
/// Unlike `scaleF32` there is no `b == 0` split: every element is
/// `s*x + b`, one expression, which clang contracts to one fused
/// multiply-add before narrowing back to BF16 -- `fmla.4s` in the
/// vectorised body and `fmadd` in the tail of `llama.cpp.zmake`'s `ops.o`.
/// `ops-diff` cannot see it (BF16 rounding hides the last f32 bit; an
/// unfused injection passed 2^20 elements), so the disassembly is the
/// oracle.
fn scaleBf16(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(src0), "ggml_is_contiguous_rows(src0)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(dst), "ggml_is_contiguous_rows(dst)");
    ggml.impl.assert(ggml.types.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");

    const s = ggml.impl.getOpParamsF32(dst, 0); // scale factor
    const b = ggml.impl.getOpParamsF32(dst, 1); // bias

    const ith: c_int = params.ith;
    const nth: c_int = params.nth;

    const nc: c_int = @intCast(src0.ne[0]);
    const nr: c_int = @intCast(ggml.types.ggml_nrows(src0));

    const l = ggml.cpu.defs.UnaryLocals.of(src0, dst);

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var i = ir0;
    while (i < ir1) : (i += 1) {
        const j1 = @rem(@as(i64, i), l.ne01);
        const j2 = @rem(@divTrunc(@as(i64, i), l.ne01), l.ne02);
        const j3 = @divTrunc(@as(i64, i), l.ne01 * l.ne02);

        const x: [*]const ggml.impl.ggml_c.ggml_bf16_t = @ptrCast(@alignCast(s0 + ggml.cpu.ops.common.byteOff(j1, l.nb01) + ggml.cpu.ops.common.byteOff(j2, l.nb02) + ggml.cpu.ops.common.byteOff(j3, l.nb03)));
        const y: [*]ggml.impl.ggml_c.ggml_bf16_t = @ptrCast(@alignCast(dd + ggml.cpu.ops.common.byteOff(j1, l.nb1) + ggml.cpu.ops.common.byteOff(j2, l.nb2) + ggml.cpu.ops.common.byteOff(j3, l.nb3)));
        for (0..@intCast(nc)) |k| {
            y[k] = .{ .bits = ggml.impl.fp32ToBf16(@mulAdd(f32, s, ggml.impl.bf16ToFp32(x[k].bits), b)) };
        }
    }
}

/// Ports `ggml_compute_forward_scale` (ops.cpp:4952 @b1b770d90).
pub fn ggml_compute_forward_scale(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => scaleF32(params, dst),
        ggml.impl.ggml_c.GGML_TYPE_BF16 => scaleBf16(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_set_f32` (ops.cpp:4976 @b1b770d90) and
/// `ggml_compute_forward_set_i32` (ops.cpp:5047 @b1b770d90).
///
/// Identical but for the element type and the `ggml_vec_cpy_*` they call.
fn setT(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert(ggml.types.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(dst) and ggml.types.ggml_is_contiguous(src0), "ggml_is_contiguous(dst) && ggml_is_contiguous(src0)");

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
            const n = ggml.types.ggml_nbytes(dst);
            const d: [*]u8 = @ptrCast(dst.data.?);
            const s: [*]const u8 = @ptrCast(src0.data.?);
            @memcpy(d[0..n], s[0..n]);
        }
        ggml.cpu.threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
    }

    const ith: c_int = params.ith;
    const nth: c_int = params.nth;

    const nr: c_int = @intCast(ggml.types.ggml_nrows(src1));
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
    const nb0: usize = ggml.types.ggml_element_size(src0);

    const im0: usize = @intCast(if (ne10 == 0) 0 else ne10 - 1);
    const im1: usize = @intCast(if (ne11 == 0) 0 else ne11 - 1);
    const im2: usize = @intCast(if (ne12 == 0) 0 else ne12 - 1);
    const im3: usize = @intCast(if (ne13 == 0) 0 else ne13 - 1);

    ggml.impl.assert(offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml.types.ggml_nbytes(dst), "offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst)");

    ggml.impl.assert(nb10 == @sizeOf(T), "nb10 == sizeof(T)");

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

        const y: [*]T = @ptrCast(@alignCast(dd + ggml.cpu.ops.common.byteOff(j3, nb3) + ggml.cpu.ops.common.byteOff(j2, nb2) + ggml.cpu.ops.common.byteOff(j1, nb1) + offset));
        const x: [*]const T = @ptrCast(@alignCast(s1 + ggml.cpu.ops.common.byteOff(j3, nb13) + ggml.cpu.ops.common.byteOff(j2, nb12) + ggml.cpu.ops.common.byteOff(j1, nb11)));
        if (T == f32) ggml.cpu.ops.vecinline.cpy_f32(nc, y, x) else ggml.cpu.ops.vecinline.cpy_i32(nc, y, x);
    }
}

/// Ports `ggml_compute_forward_set` (ops.cpp:5118 @b1b770d90).
pub fn ggml_compute_forward_set(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => setT(f32, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_I32 => setT(i32, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
