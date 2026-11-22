//! The ops that are one `ggml_vk_op_f32` call with their push constants
//! filled in: elementwise arithmetic, unary and GLU, copies, rows, norms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v1.0.0
//! (`b1b770d90`).

const std = @import("std");
const ggml = @import("../module.zig");

fn f32Param(t: *const ggml.impl.ggml_c.ggml_tensor, i: usize) f32 {
    return @bitCast(t.op_params[i]);
}

/// Ports `ggml_vk_get_rows` (ggml-vulkan.cpp:10288 @b1b770d90).
pub fn getRows(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.vulkan.ops.binaryPc(ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_get_rows_back` (ggml-vulkan.cpp:10303 @b1b770d90).
pub fn getRowsBack(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK, ggml.vulkan.ops.binaryPc(ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_acc` (ggml-vulkan.cpp:10318 @b1b770d90): ACC and SET, with
/// the destination view's strides and offset from the op's parameters.
pub fn acc(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const ts0 = ggml.vulkan.ops.typeSize(src0);
    // `int` from an `int32_t` op param divided by an unsigned size: unsigned
    // division of the param's bits.
    const op: [*]const u32 = @ptrCast(&dst.op_params);
    const nb1 = op[0] / ts0;
    const nb2 = op[1] / ts0;
    const nb3 = op[2] / ts0;
    const offset = op[3] / ts0;
    var pc = ggml.vulkan.ops.binaryPc(ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(src0)), src0, src1, dst, 0, 0, @bitCast(offset));
    pc.nb01 = nb1;
    pc.nb02 = nb2;
    pc.nb03 = nb3;
    pc.nb21 = nb1;
    pc.nb22 = nb2;
    pc.nb23 = nb3;
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, dst.op, pc);
}

/// Ports `ggml_vk_multi_add` (ggml-vulkan.cpp:10338 @b1b770d90): a run of
/// ADDs as one dispatch over every operand, each bound separately.
pub fn multiAdd(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, cgraph: *const ggml.impl.CGraph, node_idx: c_int) void {
    const nodeAt = struct {
        fn f(cg: *const ggml.impl.CGraph, i: c_int) *ggml.impl.ggml_c.ggml_tensor {
            return cg.nodes[@intCast(i)].?;
        }
    }.f;
    const first_node = nodeAt(cgraph, node_idx);
    const dst = nodeAt(cgraph, node_idx + ctx.num_additional_fused_ops);
    const max = ggml.vulkan.pipeline.max_parameter_count;

    // Every tensor the op uses; the last is the destination.
    var tensors: [max]*const ggml.impl.ggml_c.ggml_tensor = undefined;
    const num_srcs: u32 = @intCast(ctx.num_additional_fused_ops + 2);
    const num_tensors = num_srcs + 1;
    ggml.impl.assert(num_tensors + @intFromBool(ctx.do_add_rms_partials) <= max, "num_tensors + ctx->do_add_rms_partials <= MAX_PARAMETER_COUNT");

    tensors[0] = first_node.src[0];
    tensors[1] = first_node.src[1];
    var i: c_int = 0;
    while (i < ctx.num_additional_fused_ops) : (i += 1) {
        // check whether the previous result is src[0] or src[1]
        const next = nodeAt(cgraph, node_idx + i + 1);
        tensors[@intCast(i + 2)] = if (nodeAt(cgraph, node_idx + i) == next.src[0]) next.src[1] else next.src[0];
    }
    tensors[num_srcs] = dst;

    // The C++ leaves the strides past `num_tensors` uninitialised; the
    // shader never reads them.
    var pc: ggml.vulkan.push.MultiAdd = .{};
    pc.ne20 = ggml.vulkan.ops.u32Of(dst.ne[0]);
    pc.ne21 = ggml.vulkan.ops.u32Of(dst.ne[1]);
    pc.ne22 = ggml.vulkan.ops.u32Of(dst.ne[2]);
    pc.ne23 = ggml.vulkan.ops.u32Of(dst.ne[3]);
    for (0..num_tensors) |k| {
        const t = tensors[k];
        inline for (0..4) |j| pc.nb[k][j] = @as(u32, @truncate(t.nb[j])) / @sizeOf(f32);
    }
    pc.rms_partials = @intFromBool(ctx.do_add_rms_partials);

    const pipeline = ggml.vulkan.op_pipeline.getPipeline(ctx, tensors[0], tensors[1], null, dst, dst.op) orelse {
        ggml.impl.fprintfStderr("ggml_vulkan: Error: Missing multi_add", .{});
        ggml.impl.abort("fatal error");
    };
    ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);

    var buf: [max]*ggml.vulkan.buffer.Buffer = undefined;
    var offset: [max]usize = undefined;
    for (0..num_tensors) |k| {
        var uma = false;
        offset[k] = 0;
        if (ctx.device.uma) {
            if (ggml.vulkan.buffer.hostGet(ctx.device, tensors[k].data, &offset[k])) |b| {
                buf[k] = b;
                uma = true;
            }
        }
        if (!uma) {
            buf[k] = ggml.vulkan.buffer.tensorBuffer(tensors[k]);
            offset[k] = ggml.vulkan.buffer.tensorOffset(tensors[k]) + tensors[k].view_offs;
        }
    }
    // If any remaining descriptors are unused, just point them at src[0]
    for (num_tensors..max) |k| {
        buf[k] = buf[0];
        offset[k] = 0;
    }
    if (ctx.do_add_rms_partials) {
        buf[num_tensors] = ctx.prealloc_add_rms_partials.?;
        offset[num_tensors] = ctx.prealloc_size_add_rms_partials_offset;
    }

    var subs: [max]ggml.vulkan.buffer.Subbuffer = undefined;
    for (&subs, 0..) |*sb, k| sb.* = ggml.vulkan.context.subbuffer(ctx, buf[k], offset[k]);
    ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &subs, &pc, ggml.vulkan.ops.split512(ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(dst))));
}

/// Ports `ggml_vk_add` (ggml-vulkan.cpp:10445 @b1b770d90).
pub fn add(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_ADD, ggml.vulkan.ops.binaryPc(ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(src0)), src0, src1, dst, 0, 0, @intFromBool(ctx.do_add_rms_partials)));
}

/// Ports `ggml_vk_out_prod` (ggml-vulkan.cpp:10460 @b1b770d90).
pub fn outProd(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_OUT_PROD, ggml.vulkan.ops.binaryPc(ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(dst)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_sub`, `ggml_vk_mul` and `ggml_vk_div`
/// (ggml-vulkan.cpp:10478, 10493, 10553 @b1b770d90).
pub fn binary(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, op, ggml.vulkan.ops.binaryPc(ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_unary_mul_op_index` (ggml-vulkan.cpp:10508 @b1b770d90):
/// the index into `pipeline_unary_mul` for the supported unary ops, or -1.
pub fn unaryMulOpIndex(op: ggml.impl.ggml_c.enum_ggml_unary_op) c_int {
    return switch (op) {
        ggml.impl.ggml_c.GGML_UNARY_OP_GELU => 0,
        ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID => 1,
        ggml.impl.ggml_c.GGML_UNARY_OP_SILU => 2,
        ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS => 3,
        else => -1,
    };
}

/// Ports `ggml_vk_unary_mul` (ggml-vulkan.cpp:10518 @b1b770d90): a UNARY and
/// the MUL that consumes it, as one binary dispatch. With `op_on_b` the
/// unary applies to the B operand, which tiles into A.
pub fn unaryMul(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, cgraph: *const ggml.impl.CGraph, node_idx: c_int) void {
    const unary: *const ggml.impl.ggml_c.ggml_tensor = cgraph.nodes[@intCast(node_idx)].?;
    const mul: *ggml.impl.ggml_c.ggml_tensor = cgraph.nodes[@intCast(node_idx + 1)].?;

    // unary on src1 that tiles into src0
    const op_on_b = mul.src[1] == unary and
        !ggml.impl.ggml_c.ggml_are_same_shape(unary.src[0], mul.src[0]) and
        ggml.impl.ggml_c.ggml_can_repeat(unary, mul.src[0]);

    const src0: *const ggml.impl.ggml_c.ggml_tensor = if (op_on_b) mul.src[0] else unary.src[0];
    const src1: *const ggml.impl.ggml_c.ggml_tensor = if (op_on_b) unary.src[0] else (if (mul.src[0] == unary) mul.src[1] else mul.src[0]);

    const is_f16: usize = @intFromBool(src0.type == ggml.impl.ggml_c.GGML_TYPE_F16);
    const norepeat: usize = @intFromBool(ggml.impl.ggml_c.ggml_are_same_shape(src0, src1));
    const oi = unaryMulOpIndex(ggml.impl.ggml_c.ggml_get_unary_op(unary));
    if (oi < 0) ggml.impl.abort("fatal error");
    const pipeline = ctx.device.pipeline_unary_mul[@intCast(oi)][is_f16][norepeat][@intFromBool(op_on_b)];

    const ne: u32 = ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(if (op_on_b) mul else src0));
    ggml.vulkan.ops.opF32Override(ctx, subctx, src0, src1, null, null, mul, ggml.impl.ggml_c.GGML_OP_UNARY, ggml.vulkan.ops.binaryPc(ne, src0, src1, mul, 0, 0, 0), pipeline);
}

/// Ports `ggml_vk_add_id` (ggml-vulkan.cpp:10568 @b1b770d90).
pub fn addId(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, src2: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, src2, null, dst, ggml.impl.ggml_c.GGML_OP_ADD_ID, ggml.vulkan.push.AddId{
        .ne0 = ggml.vulkan.ops.u32Of(dst.ne[0]),
        .ne1 = ggml.vulkan.ops.u32Of(dst.ne[1]),
        .s01 = ggml.vulkan.ops.strideIn(src0.nb[1], ggml.vulkan.ops.typeSize(src0)),
        .s02 = ggml.vulkan.ops.strideIn(src0.nb[2], ggml.vulkan.ops.typeSize(src0)),
        .s11 = ggml.vulkan.ops.strideIn(src1.nb[1], ggml.vulkan.ops.typeSize(src1)),
        .s21 = ggml.vulkan.ops.strideIn(src2.nb[1], ggml.vulkan.ops.typeSize(src2)),
    });
}

/// Ports `ggml_vk_concat` (ggml-vulkan.cpp:10947 @b1b770d90): dimension 0
/// addressed in copy units, so a quantized row moves whole blocks.
pub fn concat(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const unit_size = ggml.vulkan.op_pipeline.concatUnitSize(dst.type);
    const units_per_block = ggml.vulkan.ops.typeSize(dst) / unit_size;
    const block_size = ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_blck_size(dst.type));
    const quantized = ggml.impl.ggml_c.ggml_is_quantized(dst.type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    const ne00 = ggml.vulkan.ops.u32Of(src0.ne[0]) / block_size *% units_per_block;
    const ne10 = ggml.vulkan.ops.u32Of(src1.ne[0]) / block_size *% units_per_block;
    const ne20 = ggml.vulkan.ops.u32Of(dst.ne[0]) / block_size *% units_per_block;
    const nb00: u32 = if (quantized) 1 else @truncate(src0.nb[0] / unit_size);
    const nb10: u32 = if (quantized) 1 else @truncate(src1.nb[0] / unit_size);
    const nb20: u32 = if (quantized) 1 else @truncate(dst.nb[0] / unit_size);

    const pc: ggml.vulkan.push.Concat = .{
        .ne = ne20 *% ggml.vulkan.ops.u32Of(dst.ne[1]) *% ggml.vulkan.ops.u32Of(dst.ne[2]) *% ggml.vulkan.ops.u32Of(dst.ne[3]),
        .ne00 = ne00,
        .ne01 = ggml.vulkan.ops.u32Of(src0.ne[1]),
        .ne02 = ggml.vulkan.ops.u32Of(src0.ne[2]),
        .ne03 = ggml.vulkan.ops.u32Of(src0.ne[3]),
        .nb00 = nb00,
        .nb01 = ggml.vulkan.ops.strideIn(src0.nb[1], unit_size),
        .nb02 = ggml.vulkan.ops.strideIn(src0.nb[2], unit_size),
        .nb03 = ggml.vulkan.ops.strideIn(src0.nb[3], unit_size),
        .ne10 = ne10,
        .ne11 = ggml.vulkan.ops.u32Of(src1.ne[1]),
        .ne12 = ggml.vulkan.ops.u32Of(src1.ne[2]),
        .ne13 = ggml.vulkan.ops.u32Of(src1.ne[3]),
        .nb10 = nb10,
        .nb11 = ggml.vulkan.ops.strideIn(src1.nb[1], unit_size),
        .nb12 = ggml.vulkan.ops.strideIn(src1.nb[2], unit_size),
        .nb13 = ggml.vulkan.ops.strideIn(src1.nb[3], unit_size),
        .ne20 = ne20,
        .ne21 = ggml.vulkan.ops.u32Of(dst.ne[1]),
        .ne22 = ggml.vulkan.ops.u32Of(dst.ne[2]),
        .ne23 = ggml.vulkan.ops.u32Of(dst.ne[3]),
        .nb20 = nb20,
        .nb21 = ggml.vulkan.ops.strideIn(dst.nb[1], unit_size),
        .nb22 = ggml.vulkan.ops.strideIn(dst.nb[2], unit_size),
        .nb23 = ggml.vulkan.ops.strideIn(dst.nb[3], unit_size),
        .misalign_offsets = 0,
        .param1 = 0,
        .param2 = 0,
        .param3 = dst.op_params[0],
    };
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_CONCAT, pc);
}

/// Ports `ggml_vk_upscale` (ggml-vulkan.cpp:10974 @b1b770d90).
pub fn upscale(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const ts0 = ggml.vulkan.ops.typeSize(src0);
    const mode: u32 = @bitCast(dst.op_params[0]);
    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne0 = dst.ne[0];
    const ne1 = dst.ne[1];

    var sf0: f32 = @as(f32, @floatFromInt(ne0)) / @as(f32, @floatFromInt(ne00));
    var sf1: f32 = @as(f32, @floatFromInt(ne1)) / @as(f32, @floatFromInt(ne01));
    const sf2: f32 = @as(f32, @floatFromInt(dst.ne[2])) / @as(f32, @floatFromInt(src0.ne[2]));
    const sf3: f32 = @as(f32, @floatFromInt(dst.ne[3])) / @as(f32, @floatFromInt(src0.ne[3]));
    var pixel_offset: f32 = 0.5;

    if (mode & ggml.impl.ggml_c.GGML_SCALE_FLAG_ALIGN_CORNERS != 0) {
        if (ne0 > 1 and ne00 > 1) sf0 = @as(f32, @floatFromInt(ne0 - 1)) / @as(f32, @floatFromInt(ne00 - 1));
        if (ne1 > 1 and ne01 > 1) sf1 = @as(f32, @floatFromInt(ne1 - 1)) / @as(f32, @floatFromInt(ne01 - 1));
        pixel_offset = 0.0;
    }

    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_UPSCALE, ggml.vulkan.push.Upscale{
        .ne = ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(dst)),
        .a_offset = 0,
        .d_offset = 0,
        .ne00 = ggml.vulkan.ops.u32Of(ne00),
        .ne01 = ggml.vulkan.ops.u32Of(ne01),
        .nb00 = ggml.vulkan.ops.strideIn(src0.nb[0], ts0),
        .nb01 = ggml.vulkan.ops.strideIn(src0.nb[1], ts0),
        .nb02 = ggml.vulkan.ops.strideIn(src0.nb[2], ts0),
        .nb03 = ggml.vulkan.ops.strideIn(src0.nb[3], ts0),
        .ne10 = ggml.vulkan.ops.u32Of(ne0),
        .ne11 = ggml.vulkan.ops.u32Of(ne1),
        .ne12 = ggml.vulkan.ops.u32Of(dst.ne[2]),
        .ne13 = ggml.vulkan.ops.u32Of(dst.ne[3]),
        .sf0 = sf0,
        .sf1 = sf1,
        .sf2 = sf2,
        .sf3 = sf3,
        .pixel_offset = pixel_offset,
    });
}

/// The unary-push-constant ops with no parameters, SQR, SQRT, SIN, COS,
/// LOG and every plain UNARY: ports `ggml_vk_sqr`, `ggml_vk_sqrt`, `ggml_vk_sin`,
/// `ggml_vk_cos`, `ggml_vk_log` and `ggml_vk_unary`
/// (ggml-vulkan.cpp:11009, 11013, 11068, 11072, 11076, 11469 @b1b770d90).
pub fn unaryPlain(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, op, ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, 0));
}

/// Ports `ggml_vk_scale` (ggml-vulkan.cpp:11001 @b1b770d90) and
/// `ggml_vk_clamp` (ggml-vulkan.cpp:11093 @b1b770d90): two float params.
pub fn unary2(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op) void {
    var p = ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = f32Param(dst, 0);
    p.param2 = f32Param(dst, 1);
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, op, p);
}

/// Ports `ggml_vk_arange` (ggml-vulkan.cpp:11017 @b1b770d90) and
/// `ggml_vk_fill` (ggml-vulkan.cpp:11039 @b1b770d90): a destination-only
/// dispatch. FILL spreads its workgroups over two dimensions so a large
/// tensor stays within `maxComputeWorkGroupCount`.
pub fn fillLike(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op) void {
    const n: u64 = @bitCast(ggml.impl.ggml_c.ggml_nelements(dst));
    if (op == ggml.impl.ggml_c.GGML_OP_FILL) ggml.impl.assert(n > 0, "n > 0");
    const pc: ggml.vulkan.push.Op = .{
        .KX = @truncate(n),
        .KY = 1,
        .param1 = f32Param(dst, 0),
        .param2 = if (op == ggml.impl.ggml_c.GGML_OP_ARANGE) f32Param(dst, 2) else 0.0,
        .param3 = 0,
        .param4 = 0,
    };
    const pipeline = ggml.vulkan.op_pipeline.getPipeline(ctx, null, null, null, dst, op).?;

    var elements: [3]u32 = .{ @truncate(n), 1, 1 };
    if (op == ggml.impl.ggml_c.GGML_OP_FILL) {
        // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
        const limits = &ctx.device.properties.limits;
        const total_wg: u32 = @truncate((n + pipeline.wg_denoms[0] - 1) / pipeline.wg_denoms[0]);
        const wg_x = @min(total_wg, limits.maxComputeWorkGroupCount[0]);
        const wg_y = (total_wg +% wg_x -% 1) / wg_x;
        ggml.impl.assert(wg_y <= limits.maxComputeWorkGroupCount[1], "wg_y <= ctx->device->properties.limits.maxComputeWorkGroupCount[1]");
        elements = .{ wg_x *% pipeline.wg_denoms[0], wg_y *% pipeline.wg_denoms[1], 1 };
    }

    ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);
    const dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, false);
    ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{dst_buf}, &pc, elements);
}

/// Ports `ggml_vk_tri` (ggml-vulkan.cpp:11080 @b1b770d90).
pub fn tri(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    var p = ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = f32Param(dst, 0);
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_TRI, p);
}

/// Ports `ggml_vk_diag`, `ggml_vk_repeat` and `ggml_vk_repeat_back`
/// (ggml-vulkan.cpp:11087, 11132, 11137 @b1b770d90): the destination's
/// element count.
pub fn unaryDstCount(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, op, ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, ggml.impl.ggml_c.ggml_nelements(dst)));
}

/// Ports `ggml_vk_pad` (ggml-vulkan.cpp:11101 @b1b770d90).
pub fn pad(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_PAD, ggml.vulkan.ops.padPushConstantsInit(src0, dst));
}

/// Ports `ggml_vk_pad_reflect_1d` (ggml-vulkan.cpp:11106 @b1b770d90): the
/// pads travel as float bits.
pub fn padReflect1d(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    var p = ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, ggml.impl.ggml_c.ggml_nelements(dst));
    p.param1 = @bitCast(dst.op_params[0]);
    p.param2 = @bitCast(dst.op_params[1]);
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D, p);
}

/// Ports `ggml_vk_roll` (ggml-vulkan.cpp:11117 @b1b770d90): the four shifts
/// biased by 0x8000 and packed in pairs as float bits.
pub fn roll(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const s = dst.op_params;
    const s01: u32 = (@as(u32, @bitCast(s[0] +% 0x8000)) << 16) | @as(u32, @bitCast(s[1] +% 0x8000));
    const s23: u32 = (@as(u32, @bitCast(s[2] +% 0x8000)) << 16) | @as(u32, @bitCast(s[3] +% 0x8000));
    var p = ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = @bitCast(s01);
    p.param2 = @bitCast(s23);
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_ROLL, p);
}

/// Ports `ggml_vk_cpy` (ggml-vulkan.cpp:11142 @b1b770d90): a quantized-to-
/// quantized copy counts in 2- or 4-byte units.
pub fn cpy(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    var ne = ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(src0));
    if (ggml.impl.ggml_c.ggml_is_quantized(src0.type) and ggml.impl.ggml_c.ggml_is_quantized(dst.type)) {
        ne /= ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_blck_size(src0.type));
        const ts = ggml.vulkan.ops.typeSize(src0);
        if (ts % 4 == 0) ne *%= ts / 4 else ne *%= ts / 2;
    }
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_CPY, ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, ne));
}

/// Ports `ggml_vk_set_rows` (ggml-vulkan.cpp:11158 @b1b770d90).
pub fn setRows(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    // set_rows can have a nonempty dst with empty srcs.
    if (ggml.impl.ggml_c.ggml_is_empty(src0) or ggml.impl.ggml_c.ggml_is_empty(src1)) return;
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_SET_ROWS, ggml.vulkan.ops.binaryPc(ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_silu_back` (ggml-vulkan.cpp:11180 @b1b770d90).
pub fn siluBack(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_SILU_BACK, ggml.vulkan.push.Op{ .KX = ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(src0)), .KY = 0, .param1 = 0, .param2 = 0, .param3 = 0, .param4 = 0 });
}

/// Ports `ggml_vk_norm` (ggml-vulkan.cpp:11184 @b1b770d90) and
/// `ggml_vk_l2_norm` (ggml-vulkan.cpp:11462 @b1b770d90): epsilon in param1.
pub fn normEps(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op) void {
    var p = ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = f32Param(dst, 0);
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, op, p);
}

/// Ports `ggml_vk_group_norm` (ggml-vulkan.cpp:11192 @b1b770d90).
pub fn groupNorm(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const num_groups: u32 = @bitCast(dst.op_params[0]);
    const eps = f32Param(dst, 1);
    // int64 arithmetic, then narrowed.
    const ng: i64 = num_groups;
    const group_size = ggml.vulkan.ops.u32Of(src0.ne[0] *% src0.ne[1] *% @divTrunc(src0.ne[2] + ng - 1, ng));
    const num_workgroups = ggml.vulkan.ops.u32Of(ng *% src0.ne[3]);
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_GROUP_NORM, ggml.vulkan.push.Op{ .KX = group_size, .KY = num_workgroups, .param1 = eps, .param2 = 0, .param3 = 0, .param4 = 0 });
}

/// Ports `ggml_vk_rms_norm_back` (ggml-vulkan.cpp:11457 @b1b770d90).
pub fn rmsNormBack(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_RMS_NORM_BACK, ggml.vulkan.push.Op{ .KX = ggml.vulkan.ops.u32Of(src0.ne[0]), .KY = ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nrows(src0)), .param1 = f32Param(dst, 0), .param2 = 0, .param3 = 0, .param4 = 0 });
}

/// Ports `ggml_vk_xielu` (ggml-vulkan.cpp:11473 @b1b770d90).
pub fn xielu(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    var p = ggml.vulkan.ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = f32Param(dst, 1);
    p.param2 = f32Param(dst, 2);
    p.param3 = f32Param(dst, 3);
    p.param4 = f32Param(dst, 4);
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_UNARY, p);
}

/// Ports `ggml_vk_glu` (ggml-vulkan.cpp:11483 @b1b770d90): single (halves of
/// one row, possibly swapped) or split (two tensors).
pub fn glu(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: ?*const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const swapped = dst.op_params[1] != 0;
    const split = src1 != null;
    const alpha = f32Param(dst, 2);
    const limit = f32Param(dst, 3);

    if (!split) {
        ggml.impl.assert(@divTrunc(src0.ne[0], 2) == dst.ne[0], "src0->ne[0] / 2 == dst->ne[0]");
    } else {
        ggml.impl.assert(src0.ne[0] == src1.?.ne[0], "src0->ne[0] == src1->ne[0]");
        ggml.impl.assert(src0.ne[0] == dst.ne[0], "src0->ne[0] == dst->ne[0]");
        ggml.impl.assert(src0.type == src1.?.type, "src0->type == src1->type");
    }

    const mode: u32 = if (split) 2 else if (swapped) 1 else 0;
    const s0 = ggml.impl.ggml_c.ggml_type_size(src0.type);
    const s1 = if (split) ggml.impl.ggml_c.ggml_type_size(src1.?.type) else s0;
    const sd = ggml.impl.ggml_c.ggml_type_size(dst.type);
    const b = if (split) src1.? else src0;

    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_GLU, ggml.vulkan.push.Glu{
        .N = ggml.vulkan.ops.u32Of(ggml.impl.ggml_c.ggml_nelements(dst)),
        .ne00 = ggml.vulkan.ops.u32Of(src0.ne[0]),
        .ne20 = ggml.vulkan.ops.u32Of(dst.ne[0]),
        .mode = mode,
        .alpha = alpha,
        .limit = limit,
        .nb00 = @truncate(src0.nb[0] / s0),
        .nb01 = @truncate(src0.nb[1] / s0),
        .nb02 = @truncate(src0.nb[2] / s0),
        .nb03 = @truncate(src0.nb[3] / s0),
        .nb10 = @truncate(b.nb[0] / s1),
        .nb11 = @truncate(b.nb[1] / s1),
        .nb12 = @truncate(b.nb[2] / s1),
        .nb13 = @truncate(b.nb[3] / s1),
        .nb20 = @truncate(dst.nb[0] / sd),
        .nb21 = @truncate(dst.nb[1] / sd),
        .nb22 = @truncate(dst.nb[2] / sd),
        .nb23 = @truncate(dst.nb[3] / sd),
        .ne21 = ggml.vulkan.ops.u32Of(dst.ne[1]),
        .ne22 = ggml.vulkan.ops.u32Of(dst.ne[2]),
    });
}

/// Ports `ggml_vk_snake_dispatch_fused` (ggml-vulkan.cpp:12357 @b1b770d90):
/// the five-node snake chain, y = x + sin(a*x)^2 * inv_b, as one dispatch.
pub fn snakeFused(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, cgraph: *const ggml.impl.CGraph, node_idx: c_int) void {
    const at = struct {
        fn f(cg: *const ggml.impl.CGraph, i: c_int) *ggml.impl.ggml_c.ggml_tensor {
            return cg.nodes[@intCast(i)].?;
        }
    }.f;
    const mul0 = at(cgraph, node_idx + 0);
    const sqr = at(cgraph, node_idx + 2);
    const mul1 = at(cgraph, node_idx + 3);
    const add_ = at(cgraph, node_idx + 4);

    // x carries the full activation shape, a is the broadcast operand
    const x: *const ggml.impl.ggml_c.ggml_tensor = if (ggml.impl.ggml_c.ggml_are_same_shape(mul0, mul0.src[0])) mul0.src[0] else mul0.src[1];
    const a: *const ggml.impl.ggml_c.ggml_tensor = if (x == mul0.src[0]) mul0.src[1] else mul0.src[0];
    // mul1 reads sqr and inv_b in either operand order
    const inv_b: *const ggml.impl.ggml_c.ggml_tensor = if (mul1.src[0] == sqr) mul1.src[1] else mul1.src[0];

    const dev = ctx.device;
    const pipeline = switch (x.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => dev.pipeline_snake_f32,
        ggml.impl.ggml_c.GGML_TYPE_F16 => dev.pipeline_snake_f16,
        ggml.impl.ggml_c.GGML_TYPE_BF16 => dev.pipeline_snake_bf16,
        else => ggml.impl.abort("unsupported type"),
    }.?;
    ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);

    const pc: ggml.vulkan.push.Snake = .{ .ne0 = ggml.vulkan.ops.u32Of(x.ne[0]), .ne1 = ggml.vulkan.ops.u32Of(x.ne[1]) };
    ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{
        ggml.vulkan.context.tensorSubbuffer(ctx, x, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, a, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, inv_b, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, add_, false),
    }, &pc, .{ pc.ne0, pc.ne1, 1 });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
