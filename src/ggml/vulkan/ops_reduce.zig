//! Row-wise ops: softmax and its back pass, the fused topk_moe router,
//! RoPE, the sorts and top-k, sums and scans, argmax and count_equal,
//! the triangular solve.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;
const Tensor = c.ggml_tensor;
const CGraph = impl.CGraph;
const Subbuffer = m.buffer.Subbuffer;
const push = m.push;
const ops = m.ops;
const u32Of = ops.u32Of;

extern fn powf(x: f32, y: f32) f32;
extern fn log2f(x: f32) f32;
extern fn floorf(x: f32) f32;
extern fn ceilf(x: f32) f32;

fn at(cg: *const CGraph, i: c_int) *Tensor {
    return cg.nodes[@intCast(i)].?;
}

fn f32Param(t: *const Tensor, i: usize) f32 {
    return @bitCast(t.op_params[i]);
}

fn opPc(kx: u32, ky: u32, p1: f32, p2: f32) push.Op {
    return .{ .KX = kx, .KY = ky, .param1 = p1, .param2 = p2, .param3 = 0, .param4 = 0 };
}

/// A preallocated buffer's range from `offset`, as the C++ spells
/// `{ ctx->prealloc_x, 0, size }`.
fn prealloc(buf: ?*m.buffer.Buffer, offset: u64, size: u64) Subbuffer {
    return .{ .buffer = buf, .offset = offset, .size = size };
}

/// Ports `ggml_vk_soft_max` (ggml-vulkan.cpp:13479 @c1d0e7a00): one pass up
/// to 16384 columns, three over preallocated scratch beyond.
pub fn softMax(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: ?*const Tensor, src2: ?*const Tensor, dst: *Tensor) void {
    const scale = f32Param(dst, 0);
    const max_bias = f32Param(dst, 1);

    const ncols = u32Of(src0.ne[0]);
    const nrows_x = u32Of(c.ggml_nrows(src0));
    const nrows_y = u32Of(src0.ne[1]);

    const ne12: u32 = if (src1) |s| u32Of(s.ne[2]) else 0;
    const ne13: u32 = if (src1) |s| u32Of(s.ne[3]) else 0;
    const nb11: u32 = if (src1) |s| @truncate(s.nb[1] / s.nb[0]) else 0;
    const nb12: u32 = if (src1) |s| @truncate(s.nb[2] / s.nb[0]) else 0;
    const nb13: u32 = if (src1) |s| @truncate(s.nb[3] / s.nb[0]) else 0;

    const n_head_kv = u32Of(src0.ne[2]);
    const n_head_log2: u32 = @as(u32, 1) << @intFromFloat(floorf(log2f(@floatFromInt(n_head_kv))));
    const nhl: f32 = @floatFromInt(n_head_log2);
    const m0 = powf(2.0, -max_bias / nhl);
    const m1 = powf(2.0, -(max_bias / 2.0) / nhl);

    const pc: push.SoftMax = .{
        .KX = ncols,
        .KY = if (src1 != null) nrows_y else 0,
        .ne00 = u32Of(src0.ne[0]),
        .ne01 = u32Of(src0.ne[1]),
        .ne02 = u32Of(src0.ne[2]),
        .ne12 = ne12,
        .ne13 = ne13,
        .nb11 = nb11,
        .nb12 = nb12,
        .nb13 = nb13,
        .scale = scale,
        .max_bias = max_bias,
        .m0 = m0,
        .m1 = m1,
        .n_head_log2 = n_head_log2,
        .nrows_x = nrows_x,
        .has_sinks = @intFromBool(src2 != null),
    };

    if (ncols <= 16384) {
        ops.opF32(ctx, subctx, src0, src1, src2, null, dst, c.GGML_OP_SOFT_MAX, pc);
        return;
    }

    const buf_a = m.context.tensorSubbuffer(ctx, src0, false);
    const buf_b = if (src1) |s| m.context.tensorSubbuffer(ctx, s, false) else buf_a;
    const buf_c = if (src2) |s| m.context.tensorSubbuffer(ctx, s, false) else buf_a;
    const buf_d = m.context.tensorSubbuffer(ctx, dst, false);

    const elems_per_wg: u32 = 128 * 4;
    const num_wgs = ops.ceilDiv(ncols, elems_per_wg);
    const tmp_size: usize = @as(usize, num_wgs *% nrows_x) * @sizeOf(f32);

    if (ctx.prealloc_size_x < tmp_size) {
        ctx.prealloc_size_x = tmp_size;
        m.nodes.preallocateBuffers(ctx, subctx);
    }
    if (ctx.prealloc_size_y < tmp_size) {
        ctx.prealloc_size_y = tmp_size;
        m.nodes.preallocateBuffers(ctx, subctx);
    }
    if (ctx.prealloc_x_need_sync or ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);

    const buf_x = prealloc(ctx.prealloc_x, 0, tmp_size);
    const buf_y = prealloc(ctx.prealloc_y, 0, tmp_size);
    const elements = [3]u32{ num_wgs, nrows_x, 1 };

    const f16mask = src1 != null and src1.?.type == c.GGML_TYPE_F16;
    const dev = ctx.device;
    const p1 = (if (f16mask) dev.pipeline_soft_max_large1_f32_f16 else dev.pipeline_soft_max_large1_f32).?;
    const p2 = (if (f16mask) dev.pipeline_soft_max_large2_f32_f16 else dev.pipeline_soft_max_large2_f32).?;
    const p3 = (if (f16mask) dev.pipeline_soft_max_large3_f32_f16 else dev.pipeline_soft_max_large3_f32).?;
    m.context.requestDescriptorSets(ctx, p1, 1);
    m.context.requestDescriptorSets(ctx, p2, 1);
    m.context.requestDescriptorSets(ctx, p3, 1);

    const bufs = [_]Subbuffer{ buf_a, buf_b, buf_c, buf_d, buf_x, buf_y };
    m.context.dispatch(ctx, subctx, p1, &bufs, &pc, elements);
    m.context.syncBuffers(ctx, subctx);
    m.context.dispatch(ctx, subctx, p2, &bufs, &pc, elements);
    m.context.syncBuffers(ctx, subctx);
    m.context.dispatch(ctx, subctx, p3, &bufs, &pc, elements);

    ctx.prealloc_x_need_sync = true;
    ctx.prealloc_y_need_sync = true;
}

/// Ports `ggml_vk_soft_max_back` (ggml-vulkan.cpp:13563 @c1d0e7a00).
pub fn softMaxBack(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_SOFT_MAX_BACK, opPc(u32Of(src0.ne[0]), u32Of(c.ggml_nrows(src0)), f32Param(dst, 0), f32Param(dst, 1)));
}

/// Ports `ggml_vk_topk_moe` (ggml-vulkan.cpp:13568 @c1d0e7a00): the expert
/// router -- gating, top-k selection, weights and optional normalisation
/// -- in one dispatch, whichever of the five patterns matched.
pub fn topkMoe(ctx: *BackendCtx, subctx: *Context, cgraph: *const CGraph, node_idx: c_int) void {
    const mode = ctx.fused_topk_moe_mode;
    const has_bias = mode == .sigmoid_norm_bias or mode == .sqrt_softplus_norm_bias;
    const logits: *const Tensor = at(cgraph, node_idx + 0).src[0];
    const bias: *const Tensor = switch (mode) {
        .sigmoid_norm_bias => at(cgraph, node_idx + 2).src[1],
        .sqrt_softplus_norm_bias => at(cgraph, node_idx + 3).src[1],
        else => logits,
    };
    const weights = at(cgraph, node_idx + ctx.num_additional_fused_ops);
    const ids = switch (mode) {
        .sigmoid_norm_bias => at(cgraph, node_idx + 4),
        .sqrt_softplus_norm_bias => at(cgraph, node_idx + 5),
        .late_softmax => at(cgraph, node_idx + 1),
        else => at(cgraph, node_idx + 3),
    };

    impl.assert(logits.type == c.GGML_TYPE_F32, "logits->type == GGML_TYPE_F32");
    impl.assert(bias.type == c.GGML_TYPE_F32, "bias->type == GGML_TYPE_F32");
    impl.assert(weights.type == c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    impl.assert(ids.type == c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    const n_experts: c_int = @truncate(logits.ne[0]);
    const n_rows: c_int = @truncate(logits.ne[1]);
    const n_expert_used: c_int = @truncate(weights.ne[1]);

    impl.assert(ids.nb[1] / c.ggml_type_size(ids.type) == @as(usize, @intCast(n_experts)), "ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts");

    const pipeline = m.op_pipeline.getPipeline(ctx, null, null, null, at(cgraph, node_idx), c.GGML_OP_SOFT_MAX).?;
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    var pc: push.TopkMoe = .{};
    pc.n_rows = @bitCast(n_rows);
    pc.n_experts_push = @bitCast(n_experts);
    pc.n_expert_used = @bitCast(n_expert_used);
    pc.clamp_min = -std.math.inf(f32);
    pc.clamp_max = std.math.inf(f32);
    const clamp_idx: ?c_int = switch (mode) {
        .early_softmax_norm => 7,
        .sigmoid_norm_bias => 8,
        .sqrt_softplus_norm_bias => 9,
        else => null,
    };
    if (clamp_idx) |ci| {
        const clamp = at(cgraph, node_idx + ci);
        impl.assert(clamp.op == c.GGML_OP_CLAMP, "clamp->op == GGML_OP_CLAMP");
        pc.clamp_min = f32Param(clamp, 0);
        pc.clamp_max = f32Param(clamp, 1);
    }

    // GATING_FUNC_SOFTMAX 0, _SIGMOID 1, _SOFTMAX_WEIGHT 2, _SQRT_SOFTPLUS 3
    pc.gating_func = switch (mode) {
        .sigmoid_norm_bias => 1,
        .sqrt_softplus_norm_bias => 3,
        .late_softmax => 2,
        else => 0,
    };
    pc.has_bias = @intFromBool(has_bias);
    pc.with_norm = @intFromBool(mode == .early_softmax_norm or has_bias);
    if (ctx.fused_topk_moe_scale) {
        impl.assert(weights.op == c.GGML_OP_SCALE, "weights->op == GGML_OP_SCALE");
        pc.output_scale = f32Param(weights, 0);
        pc.output_bias = f32Param(weights, 1);
    } else {
        pc.output_scale = 1.0;
        pc.output_bias = 0.0;
    }

    impl.assert(n_expert_used <= n_experts, "n_expert_used <= n_experts");

    const rows_per_block: u32 = 4;
    m.context.dispatch(ctx, subctx, pipeline, &.{
        m.context.tensorSubbuffer(ctx, logits, false),
        m.context.tensorSubbuffer(ctx, bias, false),
        m.context.tensorSubbuffer(ctx, weights, false),
        m.context.tensorSubbuffer(ctx, ids, false),
    }, &pc, .{ ops.ceilDiv(@as(u32, @bitCast(n_rows)), rows_per_block), 1, 1 });
}

/// Ports `ggml_vk_rope` (ggml-vulkan.cpp:13654 @c1d0e7a00), forward and
/// back; fused with VIEW and SET_ROWS it writes the cache rows directly.
pub fn rope(ctx: *BackendCtx, subctx: *Context, cgraph: *const CGraph, node_idx: c_int, backprop: bool) void {
    var dst = at(cgraph, node_idx);
    const src0: *const Tensor = dst.src[0];
    const src1: ?*const Tensor = dst.src[1];
    const src2: ?*const Tensor = dst.src[2];
    var src3: ?*const Tensor = null;

    var set_rows_stride: u32 = 0;
    // Fused rope + view + set_rows: the set_rows destination stride, its row
    // indices as src3, and its tensor as the destination.
    if (ctx.num_additional_fused_ops > 0) {
        const sr = at(cgraph, node_idx + 2);
        set_rows_stride = @truncate(sr.nb[1] / c.ggml_type_size(sr.type));
        src3 = sr.src[1];
        dst = sr;
    }
    ops.opF32(ctx, subctx, src0, src1, src2, src3, dst, c.GGML_OP_ROPE, m.ops_more.makeRopeConstants(at(cgraph, node_idx), src0, src2 != null, backprop, set_rows_stride));
}

/// Ports `ggml_vk_argsort` (ggml-vulkan.cpp:13688 @c1d0e7a00): a bitonic
/// sort, one workgroup when the row fits, otherwise passes over a padded
/// scratch row with a barrier between each.
pub fn argsort(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const op: [*]const u32 = @ptrCast(&dst.op_params);
    const ncols = u32Of(src0.ne[0]);
    const nrows = u32Of(c.ggml_nrows(src0));
    const ncols_pad_log2: u32 = @intFromFloat(ceilf(log2f(@floatFromInt(ncols))));
    const ncolsp2: u32 = @as(u32, 1) << @intCast(ncols_pad_log2);

    const pc: push.Argsort = .{ .ncols = ncols, .ncols_padded = ncolsp2, .ncols_padded_log2 = ncols_pad_log2, .nrows = nrows, .order = op[0], .outer_start = 0, .outer_end = 0, .inner_start = 0, .inner_end = 0 };

    // Pick the largest workgroup size <= ncolsp2
    const pipeline_idx = @min(ncols_pad_log2, m.device.num_argsort_pipelines - 1);
    const dev = ctx.device;
    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    const use_small = ncols_pad_log2 <= dev.max_workgroup_size_log2 and dev.pipeline_argsort_f32[pipeline_idx] != null;
    var pipeline = (if (use_small) dev.pipeline_argsort_f32[pipeline_idx] else dev.pipeline_argsort_large_f32[pipeline_idx]).?;

    const src0_buf = m.context.tensorSubbuffer(ctx, src0, false);
    const dst_buf = m.context.tensorSubbuffer(ctx, dst, false);
    var subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    if (!use_small) {
        const x_sz: usize = @as(usize, ncolsp2) * nrows * 2 * @sizeOf(c_int);
        if (ctx.prealloc_size_x < x_sz) {
            ctx.prealloc_size_x = x_sz;
            m.nodes.preallocateBuffers(ctx, subctx);
        }
        if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);
        subbuf1 = prealloc(ctx.prealloc_x, 0, ctx.prealloc_x.?.size);
    }

    const elements = [3]u32{ ncolsp2, @min(u32Of(c.ggml_nrows(src0)), dev.properties.limits.maxComputeWorkGroupCount[1]), 1 };

    // First dispatch initializes tmp_idx and does the passes that stay in a workgroup.
    {
        var pc2 = pc;
        pc2.outer_start = 0;
        pc2.outer_end = @min(ncols_pad_log2, dev.max_workgroup_size_log2);
        pc2.inner_start = 0;
        pc2.inner_end = 100;
        m.context.requestDescriptorSets(ctx, pipeline, 1);
        m.context.dispatch(ctx, subctx, pipeline, &.{ src0_buf, subbuf1, dst_buf }, &pc2, elements);
    }
    if (!use_small) {
        m.context.syncBuffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        var outer = dev.max_workgroup_size_log2;
        while (outer < ncols_pad_log2) : (outer += 1) {
            var inner: u32 = 0;
            while (inner < outer + 1) : (inner += 1) {
                var pc2 = pc;
                pc2.outer_start = outer;
                pc2.outer_end = outer + 1;
                pc2.inner_start = inner;
                pc2.inner_end = inner + 1;
                // Once the inner index stays within a workgroup, run the rest in one dispatch.
                if (outer - inner < pipeline_idx) {
                    pc2.inner_end = 100;
                    inner = outer;
                    pipeline = dev.pipeline_argsort_large_f32[pipeline_idx].?;
                } else {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = dev.pipeline_argsort_large_f32[pipeline_idx - 2].?;
                }
                m.context.requestDescriptorSets(ctx, pipeline, 1);
                m.context.dispatch(ctx, subctx, pipeline, &.{ src0_buf, subbuf1, dst_buf }, &pc2, elements);
                m.context.syncBuffers(ctx, subctx);
            }
        }
        ctx.prealloc_x_need_sync = true;
    }
}

/// Ports `ggml_vk_topk` (ggml-vulkan.cpp:13774 @c1d0e7a00): passes that each
/// reduce every workgroup's elements to its K largest, double-buffered in
/// scratch, until K remain.
pub fn topk(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const ncols = u32Of(src0.ne[0]);
    const nrows = u32Of(c.ggml_nrows(src0));
    const k = u32Of(dst.ne[0]);
    const dev = ctx.device;

    const pc: push.Topk = .{ .orig_ncols = ncols, .ncols_input = ncols, .ncols_output = ncols, .k = k, .nrows = nrows, .first_pass = 0, .last_pass = 0 };

    if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);

    var elements: [3]u32 = .{ 0, @min(nrows, dev.properties.limits.maxComputeWorkGroupCount[1]), 1 };

    var num_elements = ncols;
    var done_one_iter = false;
    var dbl_buf_index: u32 = 0;
    var dbl_buf_size: usize = 0;
    const ntp: u32 = m.device.num_topk_pipelines;
    while (num_elements > k or !done_one_iter) {
        // Prefer going as small as num_topk_pipelines - 3 for perf reasons,
        // but a larger K needs a larger workgroup.
        var max_pipeline: u32 = ntp - 1;
        const log2k: u32 = @intFromFloat(log2f(@floatFromInt(k)));
        const preferred_pipeline = @max(ntp - 3, log2k + 2);
        max_pipeline = @min(preferred_pipeline, max_pipeline);
        var min_pipeline: u32 = log2k + 1;
        // require full subgroup
        min_pipeline = @max(min_pipeline, dev.subgroup_size_log2);

        var pipeline_idx: u32 = @intFromFloat(ceilf(log2f(@floatFromInt(num_elements))));
        pipeline_idx = @min(pipeline_idx, max_pipeline);
        pipeline_idx = @max(pipeline_idx, min_pipeline);

        if (num_elements > (@as(u32, 1) << @intCast(pipeline_idx))) {
            // If we could finish on this iteration (a single workgroup), do so.
            var i = pipeline_idx;
            while (i < ntp) : (i += 1) {
                if (num_elements <= (@as(u32, 1) << @intCast(i))) {
                    pipeline_idx = i;
                    break;
                }
            }
        }

        var pipeline = dev.pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        while (pipeline == null) {
            pipeline_idx -= 1;
            impl.assert(pipeline_idx >= min_pipeline, "pipeline_idx >= min_pipeline");
            pipeline = dev.pipeline_topk_f32[pipeline_idx];
        }
        const pl = pipeline.?;

        var pc2 = pc;
        pc2.ncols_input = num_elements;
        // Number of elements remaining after this pass
        const num_dst_elements = (num_elements / pl.wg_denoms[0]) *% k +% @min(k, num_elements % pl.wg_denoms[0]);
        pc2.ncols_output = num_dst_elements;

        if (!done_one_iter) {
            // Reserve space for ivec2 per element, double buffered: K per workgroup per row
            dbl_buf_size = @as(usize, num_dst_elements *% nrows *% 2) * @sizeOf(c_int);
            const a: usize = @intCast(dev.properties.limits.minStorageBufferOffsetAlignment);
            dbl_buf_size = (dbl_buf_size + a - 1) & ~(a - 1);
            const x_sz = dbl_buf_size * 2;
            if (ctx.prealloc_size_x < x_sz) {
                ctx.prealloc_size_x = x_sz;
                m.nodes.preallocateBuffers(ctx, subctx);
            }
        }

        var src_buf: Subbuffer = undefined;
        var dst_buf: Subbuffer = undefined;
        if (num_elements == ncols) {
            pc2.first_pass = 1;
            src_buf = m.context.tensorSubbuffer(ctx, src0, false);
        } else {
            src_buf = prealloc(ctx.prealloc_x, dbl_buf_index * dbl_buf_size, dbl_buf_size);
        }
        if (num_dst_elements == k) {
            pc2.last_pass = 1;
            dst_buf = m.context.tensorSubbuffer(ctx, dst, false);
        } else {
            dst_buf = prealloc(ctx.prealloc_x, (dbl_buf_index ^ 1) * dbl_buf_size, dbl_buf_size);
        }

        elements[0] = num_elements;
        m.context.requestDescriptorSets(ctx, pl, 1);
        m.context.dispatch(ctx, subctx, pl, &.{ src_buf, dst_buf }, &pc2, elements);
        num_elements = num_dst_elements;
        dbl_buf_index ^= 1;
        if (num_elements > k) m.context.syncBuffers(ctx, subctx);
        done_one_iter = true;
    }
    ctx.prealloc_x_need_sync = true;
}

/// Ports `ggml_vk_sum`, `ggml_vk_sum_rows` and `ggml_vk_mean`
/// (ggml-vulkan.cpp:13882, 13887, 13892 @c1d0e7a00).
pub fn sumRows(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor, op: c.enum_ggml_op) void {
    var p = ops.sumRowsPushConstantsInit(src0, dst, if (op == c.GGML_OP_SUM) c.ggml_nelements(src0) else src0.ne[0]);
    if (op == c.GGML_OP_MEAN) p.weight = 1.0 / @as(f32, @floatFromInt(src0.ne[0]));
    ops.opF32(ctx, subctx, src0, null, null, null, dst, op, p);
}

/// Ports `ggml_vk_cumsum` (ggml-vulkan.cpp:13898 @c1d0e7a00): one pass for
/// short rows or many rows, otherwise block partials and a second pass.
pub fn cumsum(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const pc = ops.sumRowsPushConstantsInit(src0, dst, src0.ne[0]);
    if (dst.ne[0] <= 4096 or c.ggml_nrows(dst) >= ctx.device.shader_core_count) {
        ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_CUMSUM, pc);
        return;
    }

    const dev = ctx.device;
    const p1 = dev.pipeline_cumsum_multipass1_f32 orelse impl.abort("GGML_ASSERT(pipeline1 != nullptr && pipeline2 != nullptr) failed");
    const p2 = dev.pipeline_cumsum_multipass2_f32 orelse impl.abort("GGML_ASSERT(pipeline1 != nullptr && pipeline2 != nullptr) failed");
    m.context.requestDescriptorSets(ctx, p1, 1);
    m.context.requestDescriptorSets(ctx, p2, 1);

    const elements = [3]u32{ u32Of(dst.ne[0]), u32Of(c.ggml_nrows(dst)), 1 };
    const temp_size: usize = @sizeOf(f32) * @as(usize, elements[0]) * @as(usize, @intCast(c.ggml_nrows(dst)));
    if (ctx.prealloc_size_split_k < temp_size) {
        ctx.prealloc_size_split_k = temp_size;
        m.nodes.preallocateBuffers(ctx, subctx);
    }

    const src_buf = m.context.tensorSubbuffer(ctx, src0, false);
    const dst_buf = m.context.tensorSubbuffer(ctx, dst, false);
    const temp_buf = m.context.subbuffer(ctx, ctx.prealloc_split_k.?, 0);

    if (ctx.prealloc_split_k_need_sync) m.context.syncBuffers(ctx, subctx);

    m.context.dispatch(ctx, subctx, p1, &.{ src_buf, dst_buf, temp_buf }, &pc, elements);
    m.context.syncBuffers(ctx, subctx);
    m.context.dispatch(ctx, subctx, p2, &.{ src_buf, dst_buf, temp_buf }, &pc, elements);

    ctx.prealloc_split_k_need_sync = true;
}

/// Ports `ggml_vk_argmax` (ggml-vulkan.cpp:13945 @c1d0e7a00).
pub fn argmax(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_ARGMAX, opPc(u32Of(src0.ne[0]), u32Of(src0.ne[1]), 0, 0));
}

/// Ports `ggml_vk_count_equal` (ggml-vulkan.cpp:13949 @c1d0e7a00).
pub fn countEqual(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_COUNT_EQUAL, opPc(u32Of(c.ggml_nelements(src0)), 0, 0, 0));
}

/// Ports `ggml_vk_solve_tri` (ggml-vulkan.cpp:13953 @c1d0e7a00).
pub fn solveTri(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_SOLVE_TRI, ops.binaryPc(u32Of(c.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
