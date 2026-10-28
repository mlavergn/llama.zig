//! MUL_MAT_ID, the mixture-of-experts matmul: a matrix-vector kernel per
//! token row for small batches, otherwise one matrix-matrix dispatch over
//! every expert after counting how many rows each received.
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
const Buffer = m.buffer.Buffer;
const Subbuffer = m.buffer.Subbuffer;
const Pipeline = m.pipeline.Pipeline;
const MatmulPipeline = m.pipeline.MatmulPipeline;
const push = m.push;
const ops = m.ops;
const mm = m.ops_matmul;
const ceilDiv = ops.ceilDiv;
const alignSize = mm.alignSize;
const sub = mm.sub;

const mat_vec_fusion_flags_bias0: u32 = 0x1;
const mat_vec_fusion_flags_scale0: u32 = 0x4;
const mat_vec_fusion_flags_scale1: u32 = 0x8;

fn at(cg: *const impl.CGraph, i: c_int) *Tensor {
    return cg.nodes[@intCast(i)].?;
}

fn typeSizeOf(t: c.ggml_type) u64 {
    return c.ggml_type_size(t);
}

fn blck(t: c.ggml_type) u64 {
    return @intCast(c.ggml_blck_size(t));
}

/// `make_y_staged_dst`: src1's shape with the f16 type and the K-padded
/// row layout B is staged into.
fn yStagedDst(src1: *const Tensor, f16_type: c.ggml_type, row_stride: u32, ne11: u64) Tensor {
    var t = src1.*;
    t.type = f16_type;
    t.nb[0] = c.ggml_type_size(f16_type);
    t.nb[1] = t.nb[0] * row_stride;
    t.nb[2] = t.nb[1] * ne11;
    t.nb[3] = t.nb[2] * @as(usize, @intCast(t.ne[2]));
    return t;
}

/// Ports `ggml_vk_mul_mat_id_q_f16` (ggml-vulkan.cpp:10186 @5266f24da).
pub fn mulMatIdQF16(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, ids: *const Tensor, dst: *Tensor) void {
    const dim01 = m.op_pipeline.dim01Contiguous;
    impl.assert(dim01(src1) or src1.type == c.GGML_TYPE_F32 or src1.type == c.GGML_TYPE_F16, "ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16");
    impl.assert(ids.type == c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    const dev = ctx.device;
    const limits = &dev.properties.limits;
    const ne00: u64 = @intCast(src0.ne[0]);
    const ne01: u64 = @intCast(src0.ne[1]);
    const ne02: u64 = @intCast(src0.ne[2]);
    const ne10: u64 = @intCast(src1.ne[0]);
    const ne11: u64 = @intCast(src1.ne[1]);
    const ne12: u64 = @intCast(src1.ne[2]);
    const ne13: u64 = @intCast(src1.ne[3]);
    const nei0: u64 = @intCast(ids.ne[0]);
    const nei1: u64 = @intCast(ids.ne[1]);
    const nbi0: u32 = @truncate(ids.nb[0]);
    const nbi1: u32 = @truncate(ids.nb[1]);
    const nbi2: u32 = @truncate(ids.nb[2]);
    const ne20: u64 = @intCast(dst.ne[0]);
    const ne21: u64 = @intCast(dst.ne[1]);
    const n_as = ne02;
    // n_as counts, n_as offsets, one total, then one packed row id per (expert, token).
    // Hoisting requires 16-bit indices for the packing and a table that fits one binding.
    const hoisted_row_id_words: u64 = 2 *% n_as +% 1 +% nei0 *% nei1;
    const hoist_row_ids = n_as <= 256 and nei0 <= 0xffff and nei1 <= 0xffff and
        hoisted_row_id_words *% @sizeOf(u32) <= limits.maxStorageBufferRange;
    const prec: c.ggml_prec = @intCast(dst.op_params[0]);

    var qx = mm.Source.pinned(ctx, src0);
    var qy = mm.Source.pinned(ctx, src1);
    var qi = mm.Source.pinned(ctx, ids);

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    const x_non_contig = (dev.coopmat2 and src0.type == c.GGML_TYPE_F32) or !dim01(src0);

    // If src0 is BF16, try to use a BF16 x BF16 multiply
    const f16_type: c.ggml_type = if (src0.type == c.GGML_TYPE_BF16) c.GGML_TYPE_BF16 else c.GGML_TYPE_F16;

    var y_decode_vector_staging = false;
    if (comptime m.instance.glslc.coopmat2_decode_vector) {
        // B must already be, or be convertible to, the matmul B type used by this path.
        const supported = dev.coopmat2_decode_vector and
            (f16_type != c.GGML_TYPE_BF16 or dev.coopmat2_bf16_support) and
            (src1.type == c.GGML_TYPE_F32 or src1.type == f16_type);
        // If B is copied to prealloc_y, a 4-element-aligned row stride can be chosen.
        const uses_prealloc = !dim01(src1) or src1.type != f16_type;
        // Direct B reads are safe only if row starts and the buffer offset are 4-element aligned.
        const is_aligned = (ne10 % 4 == 0) and
            (uses_prealloc or m.context.getMisalignBytes(ctx, src1) % (4 * typeSizeOf(src1.type)) == 0);
        // Stage B only when decode-vector is available and direct B reads would be misaligned.
        y_decode_vector_staging = supported and !is_aligned;
    }

    const y_non_contig = y_decode_vector_staging or
        (dev.coopmat2 and src1.type == c.GGML_TYPE_F32) or
        (src0.type == c.GGML_TYPE_BF16 and src1.type != c.GGML_TYPE_BF16) or
        !dim01(src1);

    const y_f32_kernel = src1.type == c.GGML_TYPE_F32 and !y_non_contig;

    var quantize_y = dev.integer_dot_product and src1.type == c.GGML_TYPE_F32 and c.ggml_is_contiguous(src1) and !y_non_contig and (ne11 *% ne10) % 4 == 0;

    // Check for mmq first
    var mmp: ?*MatmulPipeline = if (quantize_y) mm.getMulMatMatIdPipeline(ctx, src0.type, c.GGML_TYPE_Q8_1, prec) else null;
    if (mmp == null) {
        // Fall back to f16 dequant mul mat
        mmp = mm.getMulMatMatIdPipeline(ctx, src0.type, if (y_non_contig) f16_type else src1.type, prec);
        quantize_y = false;
    }

    const qx_needs_dequant = mmp == null or x_non_contig;
    var qy_needs_dequant = !quantize_y and ((src1.type != f16_type and !y_f32_kernel) or y_non_contig);

    if (qx_needs_dequant) {
        // Fall back to dequant + f16 mulmat
        mmp = mm.getMulMatMatIdPipeline(ctx, f16_type, if (y_f32_kernel) c.GGML_TYPE_F32 else f16_type, prec);
    }
    const mp = mmp.?;

    const effective_src1_type: c.ggml_type = if (quantize_y) c.GGML_TYPE_Q8_1 else if (y_f32_kernel) c.GGML_TYPE_F32 else src1.type;
    const x_type = if (qx_needs_dequant) f16_type else src0.type;

    const kpad: u32 = if (quantize_y) 0 else @truncate(alignSize(ne10, mm.guessMatmulIdPipelineAlign(ctx, mp, @truncate(ne01), @truncate(nei1), x_type, effective_src1_type)));
    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    const y_staged_row_stride: u32 = if (dev.coopmat2 and !quantize_y) @truncate(alignSize(ne10, 64)) else @truncate(ne10);
    const y_needs_k_padding = ne10 != y_staged_row_stride;
    const y_needs_reformat = y_non_contig or y_needs_k_padding;
    qy_needs_dequant = qy_needs_dequant or y_needs_k_padding;

    // Not implemented
    impl.assert(y_needs_reformat or !qy_needs_dequant, "y_needs_reformat || !qy_needs_dequant");

    const aligned = !quantize_y and ne10 == kpad and ne01 > 8 and nei1 > 8;

    var pipeline = mm.guessMatmulIdPipeline(ctx, mp, @truncate(ne01), @truncate(nei1), aligned, x_type, effective_src1_type);
    if (c.ggml_nbytes(src0) > limits.maxStorageBufferRange) pipeline = mm.get64bIndexingPipeline(pipeline);

    const x_ne: u64 = @intCast(c.ggml_nelements(src0));
    const y_ne: u64 = @as(u64, y_staged_row_stride) *% ne11 *% ne12 *% ne13;
    const d_ne: u64 = @intCast(c.ggml_nelements(dst));

    const qx_sz = typeSizeOf(src0.type) * x_ne / blck(src0.type);
    const qy_sz = typeSizeOf(src1.type) * @as(u64, @intCast(c.ggml_nelements(src1))) / blck(src1.type);
    const x_sz = if (!qx_needs_dequant) qx_sz else @sizeOf(c.ggml_fp16_t) * x_ne;
    const y_sz: u64 = if (quantize_y)
        alignSize(y_ne, 128) * typeSizeOf(c.GGML_TYPE_Q8_1) / blck(c.GGML_TYPE_Q8_1)
    else if (y_f32_kernel) @sizeOf(f32) * y_ne else @sizeOf(c.ggml_fp16_t) * y_ne;
    const ids_sz: u64 = nbi2;
    const d_sz = @sizeOf(f32) * d_ne;

    const to_fp16_vk_0 = if (x_non_contig) m.op_pipeline.getCpyPipeline(ctx, src0, null, f16_type) else mm.getToFp16(ctx, src0.type);
    var to_fp16_vk_1: ?*Pipeline = undefined;
    if (y_needs_reformat) {
        var staged: Tensor = undefined;
        var staged_ptr: ?*const Tensor = null;
        if (y_needs_k_padding) {
            staged = yStagedDst(src1, f16_type, y_staged_row_stride, ne11);
            staged_ptr = &staged;
        }
        to_fp16_vk_1 = m.op_pipeline.getCpyPipeline(ctx, src1, staged_ptr, f16_type);
    } else {
        to_fp16_vk_1 = mm.getToFp16(ctx, src1.type);
    }
    impl.assert(!qx_needs_dequant or to_fp16_vk_0 != null, "!qx_needs_dequant || to_fp16_vk_0 != nullptr");
    impl.assert(!qy_needs_dequant or to_fp16_vk_1 != null, "!qy_needs_dequant || to_fp16_vk_1 != nullptr");
    const to_q8_1: ?*Pipeline = if (quantize_y) mm.getQuantizePipeline(ctx, c.GGML_TYPE_Q8_1) else null;

    const count_experts = dev.pipeline_count_experts.?;
    const expert_data_size: u64 = @sizeOf(u32) *% (if (hoist_row_ids) hoisted_row_id_words else n_as);

    {
        if ((qx_needs_dequant and x_sz > limits.maxStorageBufferRange) or
            (qy_needs_dequant and y_sz > limits.maxStorageBufferRange))
        {
            impl.abort("Requested preallocation size is too large");
        }
        if (qx_needs_dequant and ctx.prealloc_size_x < x_sz) {
            ctx.prealloc_size_x = x_sz;
            m.nodes.preallocateBuffers(ctx, subctx);
        }
        if ((qy_needs_dequant or quantize_y) and ctx.prealloc_size_y < y_sz) {
            ctx.prealloc_size_y = y_sz;
            m.nodes.preallocateBuffers(ctx, subctx);
        }
        if (ctx.prealloc_size_split_k < expert_data_size) {
            ctx.prealloc_size_split_k = expert_data_size;
            m.nodes.preallocateBuffers(ctx, subctx);
        }

        // Request descriptor sets
        m.context.requestDescriptorSets(ctx, pipeline, 1);
        if (qx_needs_dequant) m.context.requestDescriptorSets(ctx, to_fp16_vk_0.?, 1);
        if (qy_needs_dequant) m.context.requestDescriptorSets(ctx, to_fp16_vk_1.?, 1);
        if (quantize_y) m.context.requestDescriptorSets(ctx, to_q8_1.?, 1);
        m.context.requestDescriptorSets(ctx, count_experts, 1);
    }

    const d_D = m.buffer.tensorBuffer(dst);
    const d_buf_offset = m.buffer.tensorOffset(dst) + dst.view_offs;
    qx.resolve(src0);
    qy.resolve(src1);
    qi.resolve(ids);

    var d_X: *Buffer = undefined;
    var x_buf_offset: u64 = 0;
    var d_Y: *Buffer = undefined;
    var y_buf_offset: u64 = 0;
    if (qx_needs_dequant) {
        d_X = ctx.prealloc_x.?;
        impl.assert(d_X.size >= x_sz, "d_X->size >= x_sz");
    } else {
        d_X = qx.buf.?;
        x_buf_offset = qx.offset;
        impl.assert(qx_sz == x_sz, "qx_sz == x_sz");
    }
    if (qy_needs_dequant) {
        d_Y = ctx.prealloc_y.?;
        impl.assert(d_Y.size >= y_sz, "d_Y->size >= y_sz");
    } else if (quantize_y) {
        d_Y = ctx.prealloc_y.?;
        impl.assert(d_Y.size >= ceilDiv(y_sz, 144) * 144, "d_Y->size >= CEIL_DIV(y_sz, 144) * 144");
    } else {
        d_Y = qy.buf.?;
        y_buf_offset = qy.offset;
        impl.assert(qy_sz == y_sz, "qy_sz == y_sz");
    }

    if (x_non_contig or qx_needs_dequant) {
        if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);
    }

    // Count how many times each expert is used
    const expert_count_buf = sub(ctx.prealloc_split_k, 0, expert_data_size);
    if (ctx.prealloc_split_k_need_sync) m.context.syncBuffers(ctx, subctx);
    {
        const its = typeSizeOf(ids.type);
        var pc: push.CountExperts = .{
            .ne00 = @truncate(nei0),
            .ne01 = @truncate(nei1),
            .nb00 = @truncate(nbi0 / its),
            .nb01 = @truncate(nbi1 / its),
            .a_offset = @truncate(m.context.getMisalignBytes(ctx, ids) / its),
            .n_experts = @truncate(n_as),
            .hoist_row_ids = @intFromBool(hoist_row_ids),
            .ne00mp = 0,
            .ne00L = 0,
        };
        ops.initPushconstFastdiv(&pc);
        m.context.dispatch(ctx, subctx, count_experts, &.{ sub(qi.buf, qi.offset, ids_sz), expert_count_buf }, &pc, .{ if (hoist_row_ids) 1 else @as(u32, @truncate(n_as)), 1, 1 });
    }

    if (x_non_contig) {
        mm.cpyToContiguous(ctx, subctx, to_fp16_vk_0.?, src0, m.context.subbuffer(ctx, qx.buf.?, qx.offset), m.context.subbuffer(ctx, d_X, 0));
    } else if (qx_needs_dequant) {
        const pc = [5]u32{ @truncate(ne01), @truncate(ne10), @truncate(ne10), @truncate(ne10), ops.u32Of(c.ggml_nelements(src0)) };
        m.context.dispatch(ctx, subctx, to_fp16_vk_0.?, &.{ sub(qx.buf, qx.offset, qx_sz), sub(d_X, 0, x_sz) }, &pc, .{ @truncate(x_ne), 1, 1 });
    }
    if (y_needs_reformat) {
        if (ctx.prealloc_y_last_pipeline_used != to_fp16_vk_1 or
            ctx.prealloc_y_last_tensor_used != src1 or
            ctx.prealloc_y_last_k_padded != y_needs_k_padding)
        {
            if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
            if (y_needs_k_padding) {
                impl.assert(y_sz % 4 == 0, "y_sz % 4 == 0");
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                m.common.dispatch.vkCmdFillBuffer.?(subctx.cmd(), d_Y.buffer, 0, y_sz, 0);
                m.context.syncBuffers(ctx, subctx);
                const staged = yStagedDst(src1, f16_type, y_staged_row_stride, ne11);
                const ts = c.ggml_type_size(staged.type);
                mm.cpyToStrided(ctx, subctx, to_fp16_vk_1.?, src1, m.context.subbuffer(ctx, qy.buf.?, qy.offset), m.context.subbuffer(ctx, d_Y, 0), @truncate(staged.nb[0] / ts), @truncate(staged.nb[1] / ts), @truncate(staged.nb[2] / ts), @truncate(staged.nb[3] / ts));
            } else {
                mm.cpyToContiguous(ctx, subctx, to_fp16_vk_1.?, src1, m.context.subbuffer(ctx, qy.buf.?, qy.offset), m.context.subbuffer(ctx, d_Y, 0));
            }
            ctx.prealloc_y_last_pipeline_used = to_fp16_vk_1;
            ctx.prealloc_y_last_tensor_used = src1;
            ctx.prealloc_y_last_k_padded = y_needs_k_padding;
        }
    }
    if (quantize_y) {
        if (ctx.prealloc_y_last_pipeline_used != to_q8_1 or
            ctx.prealloc_y_last_tensor_used != src1 or
            ctx.prealloc_y_last_k_padded)
        {
            if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
            mm.quantizeQ81(ctx, subctx, m.context.subbuffer(ctx, qy.buf.?, qy.offset), m.context.subbuffer(ctx, d_Y, 0), @truncate(y_ne));
            ctx.prealloc_y_last_pipeline_used = to_q8_1;
            ctx.prealloc_y_last_tensor_used = src1;
            ctx.prealloc_y_last_k_padded = false;
        }
    }
    m.context.syncBuffers(ctx, subctx);

    var stride_batch_x: u32 = @truncate(ne00 *% ne01);
    const stride_b_y: u32 = if (y_needs_k_padding) y_staged_row_stride else @truncate(ne10);
    var stride_batch_y: u32 = if (y_needs_k_padding) y_staged_row_stride *% @as(u32, @truncate(ne11)) else @truncate(ne10 *% ne11);
    if (!dim01(src0) and !qx_needs_dequant) stride_batch_x = @truncate(src0.nb[0] / typeSizeOf(src0.type));
    if (!dim01(src1) and !qy_needs_dequant and !quantize_y) stride_batch_y = @truncate(src1.nb[0] / typeSizeOf(src1.type));

    // compute
    mm.matmulId(
        ctx,
        subctx,
        pipeline,
        sub(d_X, x_buf_offset, x_sz),
        sub(d_Y, y_buf_offset, y_sz),
        sub(d_D, d_buf_offset, d_sz),
        sub(qi.buf, qi.offset, ids_sz),
        expert_count_buf,
        @truncate(ne01),
        @truncate(ne21),
        @truncate(ne10),
        @truncate(ne10),
        stride_b_y,
        @truncate(ne01),
        stride_batch_x,
        stride_batch_y,
        @truncate(ne20 *% ne21),
        @truncate(n_as),
        @truncate(nei0),
        @truncate(nei1),
        @truncate(nbi1 / typeSizeOf(ids.type)),
        @truncate(ne11),
        hoist_row_ids,
    );

    if (x_non_contig or qx_needs_dequant) ctx.prealloc_x_need_sync = true;
    if (y_needs_reformat or quantize_y) ctx.prealloc_y_need_sync = true;
    ctx.prealloc_split_k_need_sync = true;
}

/// Ports `ggml_vk_mul_mat_vec_id_q_f16` (ggml-vulkan.cpp:10550 @5266f24da):
/// one matrix-vector dispatch per token row, the expert read from `ids`,
/// with a fused ADD_ID bias or MUL scale and a second scale.
pub fn mulMatVecIdQF16(ctx: *BackendCtx, subctx: *Context, cgraph: *const impl.CGraph, node_idx: c_int) void {
    const dim01 = m.op_pipeline.dim01Contiguous;
    const dst = at(cgraph, node_idx);
    const src0: *const Tensor = dst.src[0];
    const src1: *const Tensor = dst.src[1];
    const ids: *const Tensor = dst.src[2];
    impl.assert(dim01(src0) or src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_BF16, "ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16");
    impl.assert(dim01(src1) or src1.type == c.GGML_TYPE_F32 or src1.type == c.GGML_TYPE_F16, "ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16");
    impl.assert(ids.type == c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    const dev = ctx.device;
    const limits = &dev.properties.limits;
    const ne00: u64 = @intCast(src0.ne[0]);
    const ne01: u64 = @intCast(src0.ne[1]);
    const ne10: u64 = @intCast(src1.ne[0]);
    const ne11: u64 = @intCast(src1.ne[1]);
    const ne12: u64 = @intCast(src1.ne[2]);
    const nei0: u64 = @intCast(ids.ne[0]);
    const nei1: u64 = @intCast(ids.ne[1]);
    const nbi1: u32 = @truncate(ids.nb[1] / @sizeOf(c_int));
    const ne20: u64 = @intCast(dst.ne[0]);
    const ne21: u64 = @intCast(dst.ne[1]);

    const x_non_contig = !dim01(src0);
    const y_non_contig = !dim01(src1);

    const f16_f32_kernel = src1.type == c.GGML_TYPE_F32;
    var quantize_y = dev.integer_dot_product and src1.type == c.GGML_TYPE_F32 and c.ggml_is_contiguous(src1) and !y_non_contig and (ne11 *% ne10) % 4 == 0 and
        mm.shouldUseMmvq(dev, @truncate(ne01), @truncate(ne12), @truncate(ne10), src0.type);

    const to_fp16_vk_0: ?*Pipeline = if (x_non_contig) m.op_pipeline.getCpyPipeline(ctx, src0, null, src0.type) else null;
    const to_fp16_vk_1 = if (y_non_contig) m.op_pipeline.getCpyPipeline(ctx, src1, null, src1.type) else mm.getToFp16(ctx, src1.type);

    // Check for mmq first
    var dmmv: ?*Pipeline = if (quantize_y) mm.getDequantizeMulMatVecId(ctx, src0.type, c.GGML_TYPE_Q8_1, @truncate(ne20), @truncate(ne00)) else null;
    if (dmmv == null) {
        // Fall back to f16 dequant mul mat
        dmmv = mm.getDequantizeMulMatVecId(ctx, src0.type, src1.type, @truncate(ne20), @truncate(ne00));
        quantize_y = false;
    }
    const to_q8_1: ?*Pipeline = if (quantize_y) mm.getQuantizePipeline(ctx, c.GGML_TYPE_Q8_1) else null;

    const qx_needs_dequant = x_non_contig;
    const qy_needs_dequant = !quantize_y and ((src1.type != c.GGML_TYPE_F16 and !f16_f32_kernel) or y_non_contig);

    if (c.ggml_nbytes(src0) > limits.maxStorageBufferRange) dmmv = mm.get64bIndexingPipeline(dmmv.?);

    // Not implemented
    impl.assert(y_non_contig or !qy_needs_dequant, "y_non_contig || !qy_needs_dequant");
    impl.assert(!qx_needs_dequant or to_fp16_vk_0 != null, "!qx_needs_dequant || to_fp16_vk_0 != nullptr");
    impl.assert(!qy_needs_dequant or to_fp16_vk_1 != null, "!qy_needs_dequant || to_fp16_vk_1 != nullptr");
    const pipeline = dmmv orelse impl.abort("GGML_ASSERT(dmmv != nullptr) failed");

    const x_ne: u64 = @intCast(c.ggml_nelements(src0));
    const y_ne: u64 = @intCast(c.ggml_nelements(src1));
    const min_align: u64 = limits.minStorageBufferOffsetAlignment;

    const qx_sz = alignSize(typeSizeOf(src0.type) * x_ne / blck(src0.type), min_align);
    const x_sz = if (x_non_contig) alignSize(typeSizeOf(src0.type) * x_ne, min_align) else qx_sz;
    const y_sz: u64 = if (quantize_y)
        alignSize(y_ne, 128) * typeSizeOf(c.GGML_TYPE_Q8_1) / blck(c.GGML_TYPE_Q8_1)
    else if (f16_f32_kernel) @sizeOf(f32) * y_ne else @sizeOf(c.ggml_fp16_t) * y_ne;

    {
        if ((qx_needs_dequant and x_sz > limits.maxStorageBufferRange) or
            (qy_needs_dequant and y_sz > limits.maxStorageBufferRange))
        {
            impl.abort("Requested preallocation size is too large");
        }
        if (qx_needs_dequant and ctx.prealloc_size_x < x_sz) {
            ctx.prealloc_size_x = x_sz;
            m.nodes.preallocateBuffers(ctx, subctx);
        }
        if ((qy_needs_dequant or quantize_y) and ctx.prealloc_size_y < y_sz) {
            ctx.prealloc_size_y = y_sz;
            m.nodes.preallocateBuffers(ctx, subctx);
        }

        // Request descriptor sets
        if (qx_needs_dequant) m.context.requestDescriptorSets(ctx, to_fp16_vk_0.?, 1);
        if (qy_needs_dequant) m.context.requestDescriptorSets(ctx, to_fp16_vk_1.?, 1);
        if (quantize_y) m.context.requestDescriptorSets(ctx, to_q8_1.?, 1);
        m.context.requestDescriptorSets(ctx, pipeline, @truncate(nei1));
    }

    const d_D = m.context.tensorSubbuffer(ctx, at(cgraph, node_idx + ctx.num_additional_fused_ops), false);
    const d_Qx = m.context.tensorSubbuffer(ctx, src0, false);
    const d_Qy = m.context.tensorSubbuffer(ctx, src1, false);
    const d_ids = m.context.tensorSubbuffer(ctx, ids, false);
    const d_X = if (qx_needs_dequant) sub(ctx.prealloc_x, 0, ctx.prealloc_x.?.size) else d_Qx;
    const d_Y = if (qy_needs_dequant or quantize_y) sub(ctx.prealloc_y, 0, ctx.prealloc_y.?.size) else d_Qy;

    if (x_non_contig) {
        if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);
        impl.assert(x_sz == alignSize(typeSizeOf(src0.type) * x_ne, min_align), "x_sz == ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment)");
        mm.cpyToContiguous(ctx, subctx, to_fp16_vk_0.?, src0, d_Qx, d_X);
    }
    if (y_non_contig) {
        impl.assert(y_sz == typeSizeOf(src1.type) * y_ne, "y_sz == ggml_type_size(src1->type) * y_ne");
        if (ctx.prealloc_y_last_pipeline_used != to_fp16_vk_1 or
            ctx.prealloc_y_last_tensor_used != src1 or
            ctx.prealloc_y_last_k_padded)
        {
            if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
            mm.cpyToContiguous(ctx, subctx, to_fp16_vk_1.?, src1, d_Qy, d_Y);
            ctx.prealloc_y_last_pipeline_used = to_fp16_vk_1;
            ctx.prealloc_y_last_tensor_used = src1;
            ctx.prealloc_y_last_k_padded = false;
        }
    }
    if (quantize_y) {
        if (ctx.prealloc_y_last_pipeline_used != to_q8_1 or
            ctx.prealloc_y_last_tensor_used != src1 or
            ctx.prealloc_y_last_k_padded)
        {
            if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
            mm.quantizeQ81(ctx, subctx, d_Qy, d_Y, @truncate(y_ne));
            ctx.prealloc_y_last_pipeline_used = to_q8_1;
            ctx.prealloc_y_last_tensor_used = src1;
            ctx.prealloc_y_last_k_padded = false;
        }
    }

    var stride_batch_y: u32 = @truncate(ne10 *% ne11);
    if (!dim01(src1) and !qy_needs_dequant) stride_batch_y = @truncate(src1.nb[2] / typeSizeOf(src1.type));

    const max_groups_x = limits.maxComputeWorkGroupCount[0];
    var groups_x: u32 = @truncate(ne01);
    var groups_z: u32 = 1;
    if (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = ceilDiv(groups_x, groups_z);
    }

    var fusion_flags: u32 = 0;
    var d_F0 = d_D;
    if (ctx.num_additional_fused_ops > 0) {
        const next = at(cgraph, node_idx + 1);
        d_F0 = m.context.tensorSubbuffer(ctx, next.src[1], false);
        if (next.op == c.GGML_OP_MUL) {
            fusion_flags |= mat_vec_fusion_flags_scale0;
        } else {
            impl.assert(next.op == c.GGML_OP_ADD_ID, "cgraph->nodes[node_idx + 1]->op == GGML_OP_ADD_ID");
            fusion_flags |= mat_vec_fusion_flags_bias0;
        }
    }
    var d_F1 = d_D;
    if (ctx.num_additional_fused_ops > 1) {
        d_F1 = m.context.tensorSubbuffer(ctx, at(cgraph, node_idx + 2).src[1], false);
        fusion_flags |= mat_vec_fusion_flags_scale1;
    }

    // Loop over the batch dimension
    var expert_i1: u32 = 0;
    while (expert_i1 < nei1) : (expert_i1 += 1) {
        const pc: push.MatVecId = .{
            .ncols = @truncate(ne00),
            .stride_a = @truncate(ne10),
            .stride_b = @truncate(ne10),
            .stride_d = @truncate(ne01),
            .batch_stride_a = @truncate(ne00 *% ne01),
            .batch_stride_b = stride_batch_y,
            .batch_stride_d = @truncate(ne20 *% ne21),
            .fusion_flags = fusion_flags,
            .nei0 = @truncate(nei0),
            .ne11 = @truncate(ne11),
            .expert_i1 = expert_i1,
            .nbi1 = nbi1,
        };
        m.context.dispatch(ctx, subctx, pipeline, &.{ d_X, d_Y, d_D, d_F0, d_F1, d_ids }, &pc, .{ groups_x, @truncate(nei0), groups_z });
    }

    if (x_non_contig) ctx.prealloc_x_need_sync = true;
    if (y_non_contig or quantize_y) ctx.prealloc_y_need_sync = true;
}

/// Ports `ggml_vk_mul_mat_id` (ggml-vulkan.cpp:10793 @5266f24da).
pub fn mulMatId(ctx: *BackendCtx, subctx: *Context, cgraph: *const impl.CGraph, node_idx: c_int) void {
    const dst = at(cgraph, node_idx);
    if (ops.useMulMatVecId(cgraph, node_idx)) {
        mulMatVecIdQF16(ctx, subctx, cgraph, node_idx);
    } else {
        mulMatIdQF16(ctx, subctx, dst.src[0], dst.src[1], dst.src[2], dst);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
