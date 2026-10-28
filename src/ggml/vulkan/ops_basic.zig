//! The ops that are one `ggml_vk_op_f32` call with their push constants
//! filled in: elementwise arithmetic, unary and GLU, copies, rows, norms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.0
//! (`5266f24da`).

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;
const Tensor = c.ggml_tensor;
const CGraph = impl.CGraph;
const push = m.push;
const ops = m.ops;
const u32Of = ops.u32Of;
const strideIn = ops.strideIn;
const typeSize = ops.typeSize;

fn f32Param(t: *const Tensor, i: usize) f32 {
    return @bitCast(t.op_params[i]);
}

/// Ports `ggml_vk_get_rows` (ggml-vulkan.cpp:12543 @5266f24da).
pub fn getRows(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_GET_ROWS, ops.binaryPc(u32Of(c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_get_rows_back` (ggml-vulkan.cpp:12558 @5266f24da).
pub fn getRowsBack(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_GET_ROWS_BACK, ops.binaryPc(u32Of(c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_acc` (ggml-vulkan.cpp:12573 @5266f24da): ACC and SET, with
/// the destination view's strides and offset from the op's parameters.
pub fn acc(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    const ts0 = typeSize(src0);
    // `int` from an `int32_t` op param divided by an unsigned size: unsigned
    // division of the param's bits.
    const op: [*]const u32 = @ptrCast(&dst.op_params);
    const nb1 = op[0] / ts0;
    const nb2 = op[1] / ts0;
    const nb3 = op[2] / ts0;
    const offset = op[3] / ts0;
    var pc = ops.binaryPc(u32Of(c.ggml_nelements(src0)), src0, src1, dst, 0, 0, @bitCast(offset));
    pc.nb01 = nb1;
    pc.nb02 = nb2;
    pc.nb03 = nb3;
    pc.nb21 = nb1;
    pc.nb22 = nb2;
    pc.nb23 = nb3;
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, dst.op, pc);
}

/// Ports `ggml_vk_multi_add` (ggml-vulkan.cpp:12593 @5266f24da): a run of
/// ADDs as one dispatch over every operand, each bound separately.
pub fn multiAdd(ctx: *BackendCtx, subctx: *Context, cgraph: *const CGraph, node_idx: c_int) void {
    const nodeAt = struct {
        fn f(cg: *const CGraph, i: c_int) *Tensor {
            return cg.nodes[@intCast(i)].?;
        }
    }.f;
    const first_node = nodeAt(cgraph, node_idx);
    const dst = nodeAt(cgraph, node_idx + ctx.num_additional_fused_ops);
    const max = m.pipeline.max_parameter_count;

    // Every tensor the op uses; the last is the destination.
    var tensors: [max]*const Tensor = undefined;
    const num_srcs: u32 = @intCast(ctx.num_additional_fused_ops + 2);
    const num_tensors = num_srcs + 1;
    impl.assert(num_tensors + @intFromBool(ctx.do_add_rms_partials) <= max, "num_tensors + ctx->do_add_rms_partials <= MAX_PARAMETER_COUNT");

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
    var pc: push.MultiAdd = .{};
    pc.ne20 = u32Of(dst.ne[0]);
    pc.ne21 = u32Of(dst.ne[1]);
    pc.ne22 = u32Of(dst.ne[2]);
    pc.ne23 = u32Of(dst.ne[3]);
    for (0..num_tensors) |k| {
        const t = tensors[k];
        inline for (0..4) |j| pc.nb[k][j] = @as(u32, @truncate(t.nb[j])) / @sizeOf(f32);
    }
    pc.rms_partials = @intFromBool(ctx.do_add_rms_partials);

    const pipeline = m.op_pipeline.getPipeline(ctx, tensors[0], tensors[1], null, dst, dst.op) orelse {
        impl.fprintfStderr("ggml_vulkan: Error: Missing multi_add", .{});
        impl.abort("fatal error");
    };
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    var buf: [max]*m.buffer.Buffer = undefined;
    var offset: [max]usize = undefined;
    for (0..num_tensors) |k| {
        var uma = false;
        offset[k] = 0;
        if (ctx.device.uma) {
            if (m.buffer.hostGet(ctx.device, tensors[k].data, &offset[k])) |b| {
                buf[k] = b;
                uma = true;
            }
        }
        if (!uma) {
            buf[k] = m.buffer.tensorBuffer(tensors[k]);
            offset[k] = m.buffer.tensorOffset(tensors[k]) + tensors[k].view_offs;
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

    var subs: [max]m.buffer.Subbuffer = undefined;
    for (&subs, 0..) |*sb, k| sb.* = m.context.subbuffer(ctx, buf[k], offset[k]);
    m.context.dispatch(ctx, subctx, pipeline, &subs, &pc, ops.split512(u32Of(c.ggml_nelements(dst))));
}

/// Ports `ggml_vk_add` (ggml-vulkan.cpp:12700 @5266f24da).
pub fn add(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_ADD, ops.binaryPc(u32Of(c.ggml_nelements(src0)), src0, src1, dst, 0, 0, @intFromBool(ctx.do_add_rms_partials)));
}

/// Ports `ggml_vk_out_prod` (ggml-vulkan.cpp:12715 @5266f24da).
pub fn outProd(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_OUT_PROD, ops.binaryPc(u32Of(c.ggml_nelements(dst)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_sub`, `ggml_vk_mul`, `ggml_vk_div` and `ggml_vk_add1`
/// (ggml-vulkan.cpp:12733, 12748, 12763, 13227 @5266f24da).
pub fn binary(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor, op: c.enum_ggml_op) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, op, ops.binaryPc(u32Of(c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_add_id` (ggml-vulkan.cpp:12778 @5266f24da).
pub fn addId(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, src2: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, src2, null, dst, c.GGML_OP_ADD_ID, push.AddId{
        .ne0 = u32Of(dst.ne[0]),
        .ne1 = u32Of(dst.ne[1]),
        .s01 = strideIn(src0.nb[1], typeSize(src0)),
        .s02 = strideIn(src0.nb[2], typeSize(src0)),
        .s11 = strideIn(src1.nb[1], typeSize(src1)),
        .s21 = strideIn(src2.nb[1], typeSize(src2)),
    });
}

/// Ports `ggml_vk_concat` (ggml-vulkan.cpp:13157 @5266f24da): dimension 0
/// addressed in copy units, so a quantized row moves whole blocks.
pub fn concat(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    const unit_size = m.op_pipeline.concatUnitSize(dst.type);
    const units_per_block = typeSize(dst) / unit_size;
    const block_size = u32Of(c.ggml_blck_size(dst.type));
    const quantized = c.ggml_is_quantized(dst.type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    const ne00 = u32Of(src0.ne[0]) / block_size *% units_per_block;
    const ne10 = u32Of(src1.ne[0]) / block_size *% units_per_block;
    const ne20 = u32Of(dst.ne[0]) / block_size *% units_per_block;
    const nb00: u32 = if (quantized) 1 else @truncate(src0.nb[0] / unit_size);
    const nb10: u32 = if (quantized) 1 else @truncate(src1.nb[0] / unit_size);
    const nb20: u32 = if (quantized) 1 else @truncate(dst.nb[0] / unit_size);

    const pc: push.Concat = .{
        .ne = ne20 *% u32Of(dst.ne[1]) *% u32Of(dst.ne[2]) *% u32Of(dst.ne[3]),
        .ne00 = ne00,
        .ne01 = u32Of(src0.ne[1]),
        .ne02 = u32Of(src0.ne[2]),
        .ne03 = u32Of(src0.ne[3]),
        .nb00 = nb00,
        .nb01 = strideIn(src0.nb[1], unit_size),
        .nb02 = strideIn(src0.nb[2], unit_size),
        .nb03 = strideIn(src0.nb[3], unit_size),
        .ne10 = ne10,
        .ne11 = u32Of(src1.ne[1]),
        .ne12 = u32Of(src1.ne[2]),
        .ne13 = u32Of(src1.ne[3]),
        .nb10 = nb10,
        .nb11 = strideIn(src1.nb[1], unit_size),
        .nb12 = strideIn(src1.nb[2], unit_size),
        .nb13 = strideIn(src1.nb[3], unit_size),
        .ne20 = ne20,
        .ne21 = u32Of(dst.ne[1]),
        .ne22 = u32Of(dst.ne[2]),
        .ne23 = u32Of(dst.ne[3]),
        .nb20 = nb20,
        .nb21 = strideIn(dst.nb[1], unit_size),
        .nb22 = strideIn(dst.nb[2], unit_size),
        .nb23 = strideIn(dst.nb[3], unit_size),
        .misalign_offsets = 0,
        .param1 = 0,
        .param2 = 0,
        .param3 = dst.op_params[0],
    };
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_CONCAT, pc);
}

/// Ports `ggml_vk_upscale` (ggml-vulkan.cpp:13184 @5266f24da).
pub fn upscale(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const ts0 = typeSize(src0);
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

    if (mode & c.GGML_SCALE_FLAG_ALIGN_CORNERS != 0) {
        if (ne0 > 1 and ne00 > 1) sf0 = @as(f32, @floatFromInt(ne0 - 1)) / @as(f32, @floatFromInt(ne00 - 1));
        if (ne1 > 1 and ne01 > 1) sf1 = @as(f32, @floatFromInt(ne1 - 1)) / @as(f32, @floatFromInt(ne01 - 1));
        pixel_offset = 0.0;
    }

    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_UPSCALE, push.Upscale{
        .ne = u32Of(c.ggml_nelements(dst)),
        .a_offset = 0,
        .d_offset = 0,
        .ne00 = u32Of(ne00),
        .ne01 = u32Of(ne01),
        .nb00 = strideIn(src0.nb[0], ts0),
        .nb01 = strideIn(src0.nb[1], ts0),
        .nb02 = strideIn(src0.nb[2], ts0),
        .nb03 = strideIn(src0.nb[3], ts0),
        .ne10 = u32Of(ne0),
        .ne11 = u32Of(ne1),
        .ne12 = u32Of(dst.ne[2]),
        .ne13 = u32Of(dst.ne[3]),
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
/// (ggml-vulkan.cpp:13219, 13223, 13286, 13290, 13294, 13614 @5266f24da).
pub fn unaryPlain(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor, op: c.enum_ggml_op) void {
    ops.opF32(ctx, subctx, src0, null, null, null, dst, op, ops.unaryPushConstantsInit(src0, dst, 0));
}

/// Ports `ggml_vk_scale` (ggml-vulkan.cpp:13211 @5266f24da) and
/// `ggml_vk_clamp` (ggml-vulkan.cpp:13311 @5266f24da): two float params.
pub fn unary2(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor, op: c.enum_ggml_op) void {
    var p = ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = f32Param(dst, 0);
    p.param2 = f32Param(dst, 1);
    ops.opF32(ctx, subctx, src0, null, null, null, dst, op, p);
}

/// Ports `ggml_vk_arange` (ggml-vulkan.cpp:13242 @5266f24da) and
/// `ggml_vk_fill` (ggml-vulkan.cpp:13264 @5266f24da): a destination-only
/// dispatch.
pub fn fillLike(ctx: *BackendCtx, subctx: *Context, dst: *Tensor, op: c.enum_ggml_op) void {
    const pc: push.Op = .{
        .KX = u32Of(c.ggml_nelements(dst)),
        .KY = 1,
        .param1 = f32Param(dst, 0),
        .param2 = if (op == c.GGML_OP_ARANGE) f32Param(dst, 2) else 0.0,
        .param3 = 0,
        .param4 = 0,
    };
    const pipeline = m.op_pipeline.getPipeline(ctx, null, null, null, dst, op).?;
    m.context.requestDescriptorSets(ctx, pipeline, 1);
    const dst_buf = m.context.tensorSubbuffer(ctx, dst, false);
    m.context.dispatch(ctx, subctx, pipeline, &.{dst_buf}, &pc, .{ u32Of(c.ggml_nelements(dst)), 1, 1 });
}

/// Ports `ggml_vk_tri` (ggml-vulkan.cpp:13298 @5266f24da).
pub fn tri(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    var p = ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = f32Param(dst, 0);
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_TRI, p);
}

/// Ports `ggml_vk_diag`, `ggml_vk_repeat` and `ggml_vk_repeat_back`
/// (ggml-vulkan.cpp:13305, 13350, 13355 @5266f24da): the destination's
/// element count.
pub fn unaryDstCount(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor, op: c.enum_ggml_op) void {
    ops.opF32(ctx, subctx, src0, null, null, null, dst, op, ops.unaryPushConstantsInit(src0, dst, c.ggml_nelements(dst)));
}

/// Ports `ggml_vk_pad` (ggml-vulkan.cpp:13319 @5266f24da).
pub fn pad(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_PAD, ops.padPushConstantsInit(src0, dst));
}

/// Ports `ggml_vk_pad_reflect_1d` (ggml-vulkan.cpp:13324 @5266f24da): the
/// pads travel as float bits.
pub fn padReflect1d(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    var p = ops.unaryPushConstantsInit(src0, dst, c.ggml_nelements(dst));
    p.param1 = @bitCast(dst.op_params[0]);
    p.param2 = @bitCast(dst.op_params[1]);
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_PAD_REFLECT_1D, p);
}

/// Ports `ggml_vk_roll` (ggml-vulkan.cpp:13335 @5266f24da): the four shifts
/// biased by 0x8000 and packed in pairs as float bits.
pub fn roll(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const s = dst.op_params;
    const s01: u32 = (@as(u32, @bitCast(s[0] +% 0x8000)) << 16) | @as(u32, @bitCast(s[1] +% 0x8000));
    const s23: u32 = (@as(u32, @bitCast(s[2] +% 0x8000)) << 16) | @as(u32, @bitCast(s[3] +% 0x8000));
    var p = ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = @bitCast(s01);
    p.param2 = @bitCast(s23);
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_ROLL, p);
}

/// Ports `ggml_vk_cpy` (ggml-vulkan.cpp:13360 @5266f24da): a quantized-to-
/// quantized copy counts in 2- or 4-byte units.
pub fn cpy(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    var ne = u32Of(c.ggml_nelements(src0));
    if (c.ggml_is_quantized(src0.type) and c.ggml_is_quantized(dst.type)) {
        ne /= u32Of(c.ggml_blck_size(src0.type));
        const ts = typeSize(src0);
        if (ts % 4 == 0) ne *%= ts / 4 else ne *%= ts / 2;
    }
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_CPY, ops.unaryPushConstantsInit(src0, dst, ne));
}

/// Ports `ggml_vk_set_rows` (ggml-vulkan.cpp:13376 @5266f24da).
pub fn setRows(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    // set_rows can have a nonempty dst with empty srcs.
    if (c.ggml_is_empty(src0) or c.ggml_is_empty(src1)) return;
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_SET_ROWS, ops.binaryPc(u32Of(c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

/// Ports `ggml_vk_silu_back` (ggml-vulkan.cpp:13398 @5266f24da).
pub fn siluBack(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_SILU_BACK, push.Op{ .KX = u32Of(c.ggml_nelements(src0)), .KY = 0, .param1 = 0, .param2 = 0, .param3 = 0, .param4 = 0 });
}

/// Ports `ggml_vk_norm` (ggml-vulkan.cpp:13402 @5266f24da) and
/// `ggml_vk_l2_norm` (ggml-vulkan.cpp:13607 @5266f24da): epsilon in param1.
pub fn normEps(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor, op: c.enum_ggml_op) void {
    var p = ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = f32Param(dst, 0);
    ops.opF32(ctx, subctx, src0, null, null, null, dst, op, p);
}

/// Ports `ggml_vk_group_norm` (ggml-vulkan.cpp:13410 @5266f24da).
pub fn groupNorm(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const num_groups: u32 = @bitCast(dst.op_params[0]);
    const eps = f32Param(dst, 1);
    // int64 arithmetic, then narrowed.
    const ng: i64 = num_groups;
    const group_size = u32Of(src0.ne[0] *% src0.ne[1] *% @divTrunc(src0.ne[2] + ng - 1, ng));
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_GROUP_NORM, push.Op{ .KX = group_size, .KY = 0, .param1 = eps, .param2 = 0, .param3 = 0, .param4 = 0 });
}

/// Ports `ggml_vk_rms_norm_back` (ggml-vulkan.cpp:13602 @5266f24da).
pub fn rmsNormBack(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_RMS_NORM_BACK, push.Op{ .KX = u32Of(src0.ne[0]), .KY = u32Of(src0.ne[1]), .param1 = f32Param(dst, 0), .param2 = 0, .param3 = 0, .param4 = 0 });
}

/// Ports `ggml_vk_xielu` (ggml-vulkan.cpp:13618 @5266f24da).
pub fn xielu(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    var p = ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = f32Param(dst, 1);
    p.param2 = f32Param(dst, 2);
    p.param3 = f32Param(dst, 3);
    p.param4 = f32Param(dst, 4);
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_UNARY, p);
}

/// Ports `ggml_vk_glu` (ggml-vulkan.cpp:13628 @5266f24da): single (halves of
/// one row, possibly swapped) or split (two tensors).
pub fn glu(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: ?*const Tensor, dst: *Tensor) void {
    const swapped = dst.op_params[1] != 0;
    const split = src1 != null;
    const alpha = f32Param(dst, 2);
    const limit = f32Param(dst, 3);

    if (!split) {
        impl.assert(@divTrunc(src0.ne[0], 2) == dst.ne[0], "src0->ne[0] / 2 == dst->ne[0]");
    } else {
        impl.assert(src0.ne[0] == src1.?.ne[0], "src0->ne[0] == src1->ne[0]");
        impl.assert(src0.ne[0] == dst.ne[0], "src0->ne[0] == dst->ne[0]");
        impl.assert(src0.type == src1.?.type, "src0->type == src1->type");
    }

    const mode: u32 = if (split) 2 else if (swapped) 1 else 0;
    const s0 = c.ggml_type_size(src0.type);
    const s1 = if (split) c.ggml_type_size(src1.?.type) else s0;
    const sd = c.ggml_type_size(dst.type);
    const b = if (split) src1.? else src0;

    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_GLU, push.Glu{
        .N = u32Of(c.ggml_nelements(dst)),
        .ne00 = u32Of(src0.ne[0]),
        .ne20 = u32Of(dst.ne[0]),
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
        .ne21 = u32Of(dst.ne[1]),
        .ne22 = u32Of(dst.ne[2]),
    });
}

/// Ports `ggml_vk_diag_mask_inf` (ggml-vulkan.cpp:13676 @5266f24da).
pub fn diagMaskInf(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_DIAG_MASK_INF, push.DiagMask{ .ncols = u32Of(src0.ne[0]), .rows_per_channel = u32Of(src0.ne[1]), .n_past = dst.op_params[0] });
}

/// Ports `ggml_vk_snake_dispatch_fused` (ggml-vulkan.cpp:14518 @5266f24da):
/// the five-node snake chain, y = x + sin(a*x)^2 * inv_b, as one dispatch.
pub fn snakeFused(ctx: *BackendCtx, subctx: *Context, cgraph: *const CGraph, node_idx: c_int) void {
    const at = struct {
        fn f(cg: *const CGraph, i: c_int) *Tensor {
            return cg.nodes[@intCast(i)].?;
        }
    }.f;
    const mul0 = at(cgraph, node_idx + 0);
    const sqr = at(cgraph, node_idx + 2);
    const mul1 = at(cgraph, node_idx + 3);
    const add_ = at(cgraph, node_idx + 4);

    // x carries the full activation shape, a is the broadcast operand
    const x: *const Tensor = if (c.ggml_are_same_shape(mul0, mul0.src[0])) mul0.src[0] else mul0.src[1];
    const a: *const Tensor = if (x == mul0.src[0]) mul0.src[1] else mul0.src[0];
    // mul1 reads sqr and inv_b in either operand order
    const inv_b: *const Tensor = if (mul1.src[0] == sqr) mul1.src[1] else mul1.src[0];

    const dev = ctx.device;
    const pipeline = switch (x.type) {
        c.GGML_TYPE_F32 => dev.pipeline_snake_f32,
        c.GGML_TYPE_F16 => dev.pipeline_snake_f16,
        c.GGML_TYPE_BF16 => dev.pipeline_snake_bf16,
        else => impl.abort("unsupported type"),
    }.?;
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    const pc: push.Snake = .{ .ne0 = u32Of(x.ne[0]), .ne1 = u32Of(x.ne[1]) };
    m.context.dispatch(ctx, subctx, pipeline, &.{
        m.context.tensorSubbuffer(ctx, x, false),
        m.context.tensorSubbuffer(ctx, a, false),
        m.context.tensorSubbuffer(ctx, inv_b, false),
        m.context.tensorSubbuffer(ctx, add_, false),
    }, &pc, .{ pc.ne0, pc.ne1, 1 });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
