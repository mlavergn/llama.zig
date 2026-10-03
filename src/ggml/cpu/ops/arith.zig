//! `add`, `add_id`, `add1` and `acc`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Accelerate is live here
//!
//! `GGML_USE_ACCELERATE` is defined for this target, so `add1_f32` and
//! `acc_f32` compile their `vDSP_vadd` arm and not the `ggml_vec_*` one. The
//! `#else` arms are *not* ported; `cpu/binary_ops.zig` made the same choice
//! for the same reason. Note `ggml_vec_add1_f32` is still referenced by the C
//! under `GGML_UNUSED` there, which is why `vecinline.zig` has it.
//!
//! # Loop index names
//!
//! `i1`, `i2`, `i3`, `i11` and the rest are Zig integer type names. Renamed
//! `j1`, `j2`, `j3`, `j11`, digit for digit — the `cpu/mulmat.zig`
//! convention. Do not renumber them.

const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");
const threading = @import("../threading.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

const Stride = isize;
const Length = c_ulong;
extern fn vDSP_vadd(a: [*]const f32, ia: Stride, b: [*]const f32, ib: Stride, dst: [*]f32, id: Stride, n: Length) void;

/// Ports `ggml_compute_forward_add_non_quantized`
/// (ggml-cpu/binary-ops.cpp:140 @c1d0e7a00), which `cpu/binary_ops.zig`
/// already provides. Declared rather than imported so the dispatcher below
/// reads like the C's, which calls it across a translation-unit boundary too.
extern fn ggml_compute_forward_add_non_quantized(params: *const ComputeParams, dst: *Tensor) void;

/// Ports `ggml_compute_forward_add_q_f32` (ops.cpp:578 @c1d0e7a00).
fn addQF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(c.ggml_are_same_shape(src0, src1) and c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst)");

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const l = common.BinaryLocals.of(src0, src1, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const @"type" = src0.type;
    const dtype = dst.type;
    const dequantize_row_q = c.ggml_get_type_traits(@"type").*.to_float.?;
    const quantize_row_q = c.ggml_get_type_traits_cpu(dtype).*.from_float;

    // we don't support permuted src0 or src1
    impl.assert(l.nb00 == c.ggml_type_size(@"type"), "nb00 == ggml_type_size(type)");
    impl.assert(l.nb10 == @sizeOf(f32), "nb10 == sizeof(float)");

    // dst cannot be transposed or permuted
    impl.assert(l.nb0 <= l.nb1, "nb0 <= nb1");
    impl.assert(l.nb1 <= l.nb2, "nb1 <= nb2");
    impl.assert(l.nb2 <= l.nb3, "nb2 <= nb3");

    impl.assert(c.ggml_is_quantized(src0.type), "ggml_is_quantized(src0->type)");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    const wdata = wbase + (@as(usize, @intCast(l.ne00)) + common.cache_line_size_f32) * @as(usize, @intCast(ith));

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // src0 indices
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne02 * l.ne01, l.ne01);
        const j01 = ir - j03 * l.ne02 * l.ne01 - j02 * l.ne01;

        // src1 and dst are same shape as src0 => same indices
        const j13 = j03;
        const j12 = j02;
        const j11 = j01;

        const j3 = j03;
        const j2 = j02;
        const j1 = j01;

        const src0_row = s0 + (@as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03);
        const src1_row: [*]const f32 = @ptrCast(@alignCast(s1 + (@as(usize, @intCast(j11)) * l.nb11 + @as(usize, @intCast(j12)) * l.nb12 + @as(usize, @intCast(j13)) * l.nb13)));
        const dst_row = dd + (@as(usize, @intCast(j1)) * l.nb1 + @as(usize, @intCast(j2)) * l.nb2 + @as(usize, @intCast(j3)) * l.nb3);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, l.ne00);
        // add src1
        vec.acc_f32(l.ne00, wdata, src1_row);
        // quantize row to dst
        if (quantize_row_q) |q| {
            q(wdata, dst_row, l.ne00);
        } else {
            const n = @as(usize, @intCast(l.ne0)) * l.nb0;
            @memcpy(dst_row[0..n], @as([*]const u8, @ptrCast(wdata))[0..n]);
        }
    }
}

/// Ports `ggml_compute_forward_add` (ops.cpp:654 @c1d0e7a00).
pub export fn ggml_compute_forward_add(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32, c.GGML_TYPE_F16, c.GGML_TYPE_BF16 => ggml_compute_forward_add_non_quantized(params, dst),
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
        => addQF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_add_id_f32` (ops.cpp:704 @c1d0e7a00).
fn addIdF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);
    const src2 = impl.one(Tensor, dst.src[2]);

    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(src2.type == c.GGML_TYPE_I32, "src2->type == GGML_TYPE_I32");

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(src1.nb[0] == @sizeOf(f32), "src1->nb[0] == sizeof(float)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const l = common.TernaryLocals.of(src0, src1, src2, dst);

    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(l.nb10 == @sizeOf(f32), "nb10 == sizeof(float)");

    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const s2: [*]const u8 = @ptrCast(src2.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // src0 indices
        const j3 = @divTrunc(ir, l.ne2 * l.ne1);
        const j2 = @divTrunc(ir - j3 * l.ne2 * l.ne1, l.ne1);
        const j1 = ir - j3 * l.ne2 * l.ne1 - j2 * l.ne1;

        // src1 indices
        const idx_ptr: *const i32 = @ptrCast(@alignCast(s2 + @as(usize, @intCast(j1)) * l.nb20 + @as(usize, @intCast(j2)) * l.nb21));
        const j11: i64 = idx_ptr.*;

        impl.assert(j11 >= 0 and j11 < l.ne11, "i11 >= 0 && i11 < ne11");

        vec.add_f32(
            l.ne0,
            @ptrCast(@alignCast(dd + @as(usize, @intCast(j3)) * l.nb3 + @as(usize, @intCast(j2)) * l.nb2 + @as(usize, @intCast(j1)) * l.nb1)),
            @ptrCast(@alignCast(s0 + @as(usize, @intCast(j3)) * l.nb03 + @as(usize, @intCast(j2)) * l.nb02 + @as(usize, @intCast(j1)) * l.nb01)),
            @ptrCast(@alignCast(s1 + @as(usize, @intCast(j11)) * l.nb11)),
        );
    }
}

/// Ports `ggml_compute_forward_add_id` (ops.cpp:755 @c1d0e7a00).
pub export fn ggml_compute_forward_add_id(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => addIdF32(params, dst),
        else => impl.abort("unsupported type for ggml_compute_forward_add_id"),
    }
}

/// Ports `ggml_compute_forward_add1_f32` (ops.cpp:775 @c1d0e7a00).
///
/// The `GGML_USE_ACCELERATE` arm: `vDSP_vadd` with a **zero stride** on the
/// scalar operand, which is how it broadcasts `*src1->data` across the row.
fn add1F32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");
    impl.assert(c.ggml_is_scalar(src1), "ggml_is_scalar(src1)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");

    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const f32 = @ptrCast(@alignCast(src1.data.?));
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // src0 and dst are same shape => same indices
        const j3 = @divTrunc(ir, l.ne2 * l.ne1);
        const j2 = @divTrunc(ir - j3 * l.ne2 * l.ne1, l.ne1);
        const j1 = ir - j3 * l.ne2 * l.ne1 - j2 * l.ne1;

        vDSP_vadd(
            @ptrCast(@alignCast(s0 + @as(usize, @intCast(j3)) * l.nb03 + @as(usize, @intCast(j2)) * l.nb02 + @as(usize, @intCast(j1)) * l.nb01)),
            1,
            s1,
            0,
            @ptrCast(@alignCast(dd + @as(usize, @intCast(j3)) * l.nb3 + @as(usize, @intCast(j2)) * l.nb2 + @as(usize, @intCast(j1)) * l.nb1)),
            1,
            @intCast(l.ne0),
        );
    }
}

/// Ports `ggml_compute_forward_add1_f16_f32`,
/// `ggml_compute_forward_add1_f16_f16`,
/// `ggml_compute_forward_add1_bf16_f32` and
/// `ggml_compute_forward_add1_bf16_bf16`
/// (ops.cpp:825, 873, 986, 1034 @c1d0e7a00).
///
/// The four differ only in the element type and in whether the scalar arrives
/// as an `f32` or as that same narrow type, so they are one function over two
/// comptime parameters. Their assertions differ only in naming the types this
/// already fixes.
fn add1Narrow(comptime T: type, comptime scalar_is_f32: bool, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");
    impl.assert(c.ggml_is_scalar(src1), "ggml_is_scalar(src1)");

    // scalar to add
    const v: f32 = if (scalar_is_f32)
        @as(*const f32, @ptrCast(@alignCast(src1.data.?))).*
    else
        common.toF32(T, @as(*const T, @ptrCast(@alignCast(src1.data.?))).*);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.nb0 == @sizeOf(T), "nb0 == sizeof(T)");
    impl.assert(l.nb00 == @sizeOf(T), "nb00 == sizeof(T)");

    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // src0 and dst are same shape => same indices
        const j3 = @divTrunc(ir, l.ne2 * l.ne1);
        const j2 = @divTrunc(ir - j3 * l.ne2 * l.ne1, l.ne1);
        const j1 = ir - j3 * l.ne2 * l.ne1 - j2 * l.ne1;

        const dst_ptr: [*]T = @ptrCast(@alignCast(dd + @as(usize, @intCast(j3)) * l.nb3 + @as(usize, @intCast(j2)) * l.nb2 + @as(usize, @intCast(j1)) * l.nb1));
        const src0_ptr: [*]const T = @ptrCast(@alignCast(s0 + @as(usize, @intCast(j3)) * l.nb03 + @as(usize, @intCast(j2)) * l.nb02 + @as(usize, @intCast(j1)) * l.nb01));

        var i: usize = 0;
        while (i < @as(usize, @intCast(l.ne0))) : (i += 1) {
            dst_ptr[i] = common.fromF32(T, common.toF32(T, src0_ptr[i]) + v);
        }
    }
}

/// Ports `ggml_compute_forward_add1_q_f32` (ops.cpp:921 @c1d0e7a00).
fn add1QF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");
    impl.assert(c.ggml_is_scalar(src1), "ggml_is_scalar(src1)");

    // scalar to add
    const v = @as(*const f32, @ptrCast(@alignCast(src1.data.?))).*;

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const l = common.UnaryLocals.of(src0, dst);

    const @"type" = src0.type;
    const dequantize_row_q = c.ggml_get_type_traits(@"type").*.to_float.?;
    const quantize_row_q = c.ggml_get_type_traits_cpu(@"type").*.from_float.?;

    // we don't support permuted src0
    impl.assert(l.nb00 == c.ggml_type_size(@"type"), "nb00 == ggml_type_size(type)");

    // dst cannot be transposed or permuted
    impl.assert(l.nb0 <= l.nb1, "nb0 <= nb1");
    impl.assert(l.nb1 <= l.nb2, "nb1 <= nb2");
    impl.assert(l.nb2 <= l.nb3, "nb2 <= nb3");

    impl.assert(c.ggml_is_quantized(src0.type), "ggml_is_quantized(src0->type)");
    impl.assert(dst.type == src0.type, "dst->type == src0->type");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    const wdata = wbase + (@as(usize, @intCast(l.ne0)) + common.cache_line_size_f32) * @as(usize, @intCast(ith));

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // src0 and dst are same shape => same indices
        const j3 = @divTrunc(ir, l.ne2 * l.ne1);
        const j2 = @divTrunc(ir - j3 * l.ne2 * l.ne1, l.ne1);
        const j1 = ir - j3 * l.ne2 * l.ne1 - j2 * l.ne1;

        const src0_row = s0 + (@as(usize, @intCast(j1)) * l.nb01 + @as(usize, @intCast(j2)) * l.nb02 + @as(usize, @intCast(j3)) * l.nb03);
        // The C writes `i3*nb0` in this one expression where every sibling
        // writes `i3*nb3` -- see `dst_row` (ops.cpp:973 @c1d0e7a00).
        // Reproduced verbatim: this is the reference's behaviour, and
        // correcting it here would make the port disagree with the thing it
        // is measured against.
        const dst_row = dd + (@as(usize, @intCast(j1)) * l.nb1 + @as(usize, @intCast(j2)) * l.nb2 + @as(usize, @intCast(j3)) * l.nb0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, l.ne0);
        // add src1
        vec.acc1_f32(l.ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, l.ne0);
    }
}

/// Ports `ggml_compute_forward_add1` (ops.cpp:1082 @c1d0e7a00).
pub export fn ggml_compute_forward_add1(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => add1F32(params, dst),
        c.GGML_TYPE_F16 => {
            if (src1.type == c.GGML_TYPE_F16) {
                add1Narrow(c.ggml_fp16_t, false, params, dst);
            } else if (src1.type == c.GGML_TYPE_F32) {
                add1Narrow(c.ggml_fp16_t, true, params, dst);
            } else {
                impl.abort("fatal error");
            }
        },
        c.GGML_TYPE_BF16 => {
            if (src1.type == c.GGML_TYPE_BF16) {
                add1Narrow(c.ggml_bf16_t, false, params, dst);
            } else if (src1.type == c.GGML_TYPE_F32) {
                add1Narrow(c.ggml_bf16_t, true, params, dst);
            } else {
                impl.abort("fatal error");
            }
        },
        c.GGML_TYPE_Q1_0,
        c.GGML_TYPE_Q2_0,
        c.GGML_TYPE_Q4_0,
        c.GGML_TYPE_Q4_1,
        c.GGML_TYPE_Q5_0,
        c.GGML_TYPE_Q5_1,
        c.GGML_TYPE_Q8_0,
        c.GGML_TYPE_Q8_1,
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
        => add1QF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_acc_f32` (ops.cpp:1156 @c1d0e7a00).
fn accF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(src0, dst)");
    impl.assert(c.ggml_is_contiguous(dst) and c.ggml_is_contiguous(src0), "ggml_is_contiguous(dst) && ggml_is_contiguous(src0)");

    // view src0 and dst with these strides and data offset in bytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    const op: [*]const i32 = @ptrCast(&dst.op_params);
    const nb1: usize = @intCast(op[0]);
    const nb2: usize = @intCast(op[1]);
    const nb3: usize = @intCast(op[2]);
    const offset: usize = @intCast(op[3]);
    const inplace: bool = op[4] != 0;

    if (!inplace) {
        if (params.ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race
            // conditions. => do it in INIT phase
            const n = c.ggml_nbytes(dst);
            const d: [*]u8 = @ptrCast(dst.data.?);
            const s: [*]const u8 = @ptrCast(src0.data.?);
            @memcpy(d[0..n], s[0..n]);
        }
        threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
    }

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src1));
    const nc = src1.ne[0];

    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const nb10 = src1.nb[0];
    const nb11 = src1.nb[1];
    const nb12 = src1.nb[2];
    const nb13 = src1.nb[3];

    // src0 and dst as viewed during acc
    const nb0: usize = c.ggml_element_size(src0);

    const nb00 = nb0;
    const nb01 = nb1;
    const nb02 = nb2;
    const nb03 = nb3;

    impl.assert(offset + @as(usize, @intCast(if (ne10 == 0) 0 else ne10 - 1)) * nb0 + @as(usize, @intCast(if (ne11 == 0) 0 else ne11 - 1)) * nb1 +
        @as(usize, @intCast(if (ne12 == 0) 0 else ne12 - 1)) * nb2 + @as(usize, @intCast(if (ne13 == 0) 0 else ne13 - 1)) * nb3 < c.ggml_nbytes(dst), "acc fits in dst");
    impl.assert(offset + @as(usize, @intCast(if (ne10 == 0) 0 else ne10 - 1)) * nb00 + @as(usize, @intCast(if (ne11 == 0) 0 else ne11 - 1)) * nb01 +
        @as(usize, @intCast(if (ne12 == 0) 0 else ne12 - 1)) * nb02 + @as(usize, @intCast(if (ne13 == 0) 0 else ne13 - 1)) * nb03 < c.ggml_nbytes(src0), "acc fits in src0");

    impl.assert(nb10 == @sizeOf(f32), "nb10 == sizeof(float)");

    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const j3 = @divTrunc(ir, ne12 * ne11);
        const j2 = @divTrunc(ir - j3 * ne12 * ne11, ne11);
        const j1 = ir - j3 * ne12 * ne11 - j2 * ne11;

        vDSP_vadd(
            @ptrCast(@alignCast(s0 + @as(usize, @intCast(j3)) * nb03 + @as(usize, @intCast(j2)) * nb02 + @as(usize, @intCast(j1)) * nb01 + offset)),
            1,
            @ptrCast(@alignCast(s1 + @as(usize, @intCast(j3)) * nb13 + @as(usize, @intCast(j2)) * nb12 + @as(usize, @intCast(j1)) * nb11)),
            1,
            @ptrCast(@alignCast(dd + @as(usize, @intCast(j3)) * nb3 + @as(usize, @intCast(j2)) * nb2 + @as(usize, @intCast(j1)) * nb1 + offset)),
            1,
            @intCast(nc),
        );
    }
}

/// Ports `ggml_compute_forward_acc` (ops.cpp:1236 @c1d0e7a00).
pub export fn ggml_compute_forward_acc(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => accF32(params, dst),
        else => impl.abort("fatal error"),
    }
}
