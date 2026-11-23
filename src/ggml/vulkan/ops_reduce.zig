//! Row-wise ops: softmax and its back pass, the fused topk_moe router,
//! RoPE, the sorts and top-k (with the fused QSA indexer top-k), sums and
//! scans, cross-entropy loss and its back pass, argmax and count_equal, the
//! triangular solve.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v1.0.0
//! (`b1b770d90`).

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

fn at(cg: *const ggml.impl.CGraph, i: c_int) *ggml.impl.ggml_c.ggml_tensor {
    return cg.nodes[@intCast(i)].?;
}

fn f32Param(t: *const ggml.impl.ggml_c.ggml_tensor, i: usize) f32 {
    return @bitCast(t.op_params[i]);
}

fn opPc(kx: u32, ky: u32, p1: f32, p2: f32) ggml.vulkan.push.Op {
    return .{ .KX = kx, .KY = ky, .param1 = p1, .param2 = p2, .param3 = 0, .param4 = 0 };
}

/// A preallocated buffer's range from `offset`, as the C++ spells
/// `{ ctx->prealloc_x, 0, size }`.
fn prealloc(buf: ?*ggml.vulkan.buffer.Buffer, offset: u64, size: u64) ggml.vulkan.buffer.Subbuffer {
    return .{ .buffer = buf, .offset = offset, .size = size };
}

/// Ports `ggml_vk_soft_max` (ggml-vulkan.cpp:11531 @b1b770d90): one pass up
/// to 16384 columns, three over preallocated scratch beyond.
pub fn softMax(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: ?*const ggml.impl.ggml_c.ggml_tensor, src2: ?*const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const scale = f32Param(dst, 0);
    const max_bias = f32Param(dst, 1);

    const ncols = ggml.vulkan.ops.u32Of(src0.ne[0]);
    const nrows_x = ggml.vulkan.ops.u32Of(ggml.types.ggml_nrows(src0));
    const nrows_y = ggml.vulkan.ops.u32Of(src0.ne[1]);

    const ne12: u32 = if (src1) |s| ggml.vulkan.ops.u32Of(s.ne[2]) else 0;
    const ne13: u32 = if (src1) |s| ggml.vulkan.ops.u32Of(s.ne[3]) else 0;
    const nb11: u32 = if (src1) |s| @truncate(s.nb[1] / s.nb[0]) else 0;
    const nb12: u32 = if (src1) |s| @truncate(s.nb[2] / s.nb[0]) else 0;
    const nb13: u32 = if (src1) |s| @truncate(s.nb[3] / s.nb[0]) else 0;

    const n_head_kv = ggml.vulkan.ops.u32Of(src0.ne[2]);
    const n_head_log2: u32 = @as(u32, 1) << @intFromFloat(@floor(@as(f32, stdlib.libm.log2fInt(n_head_kv))));
    const nhl: f32 = @floatFromInt(n_head_log2);
    const m0 = stdlib.libm.powf(2.0, -max_bias / nhl);
    const m1 = stdlib.libm.powf(2.0, -(max_bias / 2.0) / nhl);

    const pc: ggml.vulkan.push.SoftMax = .{
        .KX = ncols,
        .KY = if (src1 != null) nrows_y else 0,
        .ne00 = ggml.vulkan.ops.u32Of(src0.ne[0]),
        .ne01 = ggml.vulkan.ops.u32Of(src0.ne[1]),
        .ne02 = ggml.vulkan.ops.u32Of(src0.ne[2]),
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
        ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, src2, null, dst, ggml.impl.ggml_c.GGML_OP_SOFT_MAX, pc);
        return;
    }

    const buf_a = ggml.vulkan.context.tensorSubbuffer(ctx, src0, false);
    const buf_b = if (src1) |s| ggml.vulkan.context.tensorSubbuffer(ctx, s, false) else buf_a;
    const buf_c = if (src2) |s| ggml.vulkan.context.tensorSubbuffer(ctx, s, false) else buf_a;
    const buf_d = ggml.vulkan.context.tensorSubbuffer(ctx, dst, false);

    const elems_per_wg: u32 = 128 * 4;
    const num_wgs = ggml.vulkan.ops.ceilDiv(ncols, elems_per_wg);
    const tmp_size: usize = @as(usize, num_wgs *% nrows_x) * @sizeOf(f32);

    if (ctx.prealloc_size_x < tmp_size) {
        ctx.prealloc_size_x = tmp_size;
        ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
    }
    if (ctx.prealloc_size_y < tmp_size) {
        ctx.prealloc_size_y = tmp_size;
        ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
    }
    if (ctx.prealloc_x_need_sync or ctx.prealloc_y_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);

    const buf_x = prealloc(ctx.prealloc_x, 0, tmp_size);
    const buf_y = prealloc(ctx.prealloc_y, 0, tmp_size);

    // the partial results overwrite a matmul input converted into prealloc_y
    ctx.prealloc_y_last_pipeline_used = null;
    ctx.prealloc_y_last_tensor_used = null;

    const elements = [3]u32{ num_wgs, nrows_x, 1 };

    const f16mask = src1 != null and src1.?.type == ggml.impl.ggml_c.GGML_TYPE_F16;
    const dev = ctx.device;
    const p1 = (if (f16mask) dev.pipeline_soft_max_large1_f32_f16 else dev.pipeline_soft_max_large1_f32).?;
    const p2 = (if (f16mask) dev.pipeline_soft_max_large2_f32_f16 else dev.pipeline_soft_max_large2_f32).?;
    const p3 = (if (f16mask) dev.pipeline_soft_max_large3_f32_f16 else dev.pipeline_soft_max_large3_f32).?;
    ggml.vulkan.context.requestDescriptorSets(ctx, p1, 1);
    ggml.vulkan.context.requestDescriptorSets(ctx, p2, 1);
    ggml.vulkan.context.requestDescriptorSets(ctx, p3, 1);

    const bufs = [_]ggml.vulkan.buffer.Subbuffer{ buf_a, buf_b, buf_c, buf_d, buf_x, buf_y };
    ggml.vulkan.context.dispatch(ctx, subctx, p1, &bufs, &pc, elements);
    ggml.vulkan.context.syncBuffers(ctx, subctx);
    ggml.vulkan.context.dispatch(ctx, subctx, p2, &bufs, &pc, elements);
    ggml.vulkan.context.syncBuffers(ctx, subctx);
    ggml.vulkan.context.dispatch(ctx, subctx, p3, &bufs, &pc, elements);

    ctx.prealloc_x_need_sync = true;
    ctx.prealloc_y_need_sync = true;
}

/// Ports `ggml_vk_soft_max_back` (ggml-vulkan.cpp:11619 @b1b770d90).
pub fn softMaxBack(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK, opPc(ggml.vulkan.ops.u32Of(src0.ne[0]), ggml.vulkan.ops.u32Of(ggml.types.ggml_nrows(src0)), f32Param(dst, 0), f32Param(dst, 1)));
}

/// Ports `ggml_vk_topk_moe` (ggml-vulkan.cpp:11624 @b1b770d90): the expert
/// router -- gating, top-k selection, weights and optional normalisation
/// -- in one dispatch, whichever of the five patterns matched.
pub fn topkMoe(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, cgraph: *const ggml.impl.CGraph, node_idx: c_int) void {
    const mode = ctx.fused_topk_moe_mode;
    const has_bias = mode == .sigmoid_norm_bias or mode == .sqrt_softplus_norm_bias;
    const logits: *const ggml.impl.ggml_c.ggml_tensor = at(cgraph, node_idx + 0).src[0];
    const bias: *const ggml.impl.ggml_c.ggml_tensor = switch (mode) {
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

    ggml.impl.assert(logits.type == ggml.impl.ggml_c.GGML_TYPE_F32, "logits->type == GGML_TYPE_F32");
    ggml.impl.assert(bias.type == ggml.impl.ggml_c.GGML_TYPE_F32, "bias->type == GGML_TYPE_F32");
    ggml.impl.assert(weights.type == ggml.impl.ggml_c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    ggml.impl.assert(ids.type == ggml.impl.ggml_c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    const n_experts: c_int = @truncate(logits.ne[0]);
    const n_rows: c_int = @truncate(logits.ne[1]);
    const n_expert_used: c_int = @truncate(weights.ne[1]);

    ggml.impl.assert(ids.nb[1] / ggml.types.ggml_type_size(ids.type) == @as(usize, @intCast(n_experts)), "ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts");

    const pipeline = ggml.vulkan.op_pipeline.getPipeline(ctx, null, null, null, at(cgraph, node_idx), ggml.impl.ggml_c.GGML_OP_SOFT_MAX).?;
    ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);

    var pc: ggml.vulkan.push.TopkMoe = .{};
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
        ggml.impl.assert(clamp.op == ggml.impl.ggml_c.GGML_OP_CLAMP, "clamp->op == GGML_OP_CLAMP");
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
        ggml.impl.assert(weights.op == ggml.impl.ggml_c.GGML_OP_SCALE, "weights->op == GGML_OP_SCALE");
        pc.output_scale = f32Param(weights, 0);
        pc.output_bias = f32Param(weights, 1);
    } else {
        pc.output_scale = 1.0;
        pc.output_bias = 0.0;
    }

    ggml.impl.assert(n_expert_used <= n_experts, "n_expert_used <= n_experts");

    const rows_per_block: u32 = 4;
    ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{
        ggml.vulkan.context.tensorSubbuffer(ctx, logits, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, bias, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, weights, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, ids, false),
    }, &pc, .{ ggml.vulkan.ops.ceilDiv(@as(u32, @bitCast(n_rows)), rows_per_block), 1, 1 });
}

/// Ports `ggml_vk_rope` (ggml-vulkan.cpp:11710 @b1b770d90), forward and
/// back; fused with VIEW and SET_ROWS it writes the cache rows directly.
pub fn rope(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, cgraph: *const ggml.impl.CGraph, node_idx: c_int, backprop: bool) void {
    var dst = at(cgraph, node_idx);
    const src0: *const ggml.impl.ggml_c.ggml_tensor = dst.src[0];
    const src1: ?*const ggml.impl.ggml_c.ggml_tensor = dst.src[1];
    const src2: ?*const ggml.impl.ggml_c.ggml_tensor = dst.src[2];
    var src3: ?*const ggml.impl.ggml_c.ggml_tensor = null;

    var set_rows_stride: u32 = 0;
    // Fused rope + view + set_rows: the set_rows destination stride, its row
    // indices as src3, and its tensor as the destination.
    if (ctx.num_additional_fused_ops > 0) {
        const sr = at(cgraph, node_idx + 2);
        set_rows_stride = @truncate(sr.nb[1] / ggml.types.ggml_type_size(sr.type));
        src3 = sr.src[1];
        dst = sr;
    }
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, src2, src3, dst, ggml.impl.ggml_c.GGML_OP_ROPE, ggml.vulkan.ops_more.makeRopeConstants(at(cgraph, node_idx), src0, src2 != null, backprop, set_rows_stride));
}

/// Ports `ggml_vk_argsort` (ggml-vulkan.cpp:11730 @b1b770d90): a bitonic
/// sort, one workgroup when the row fits, otherwise passes over a padded
/// scratch row with a barrier between each.
pub fn argsort(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const op: [*]const u32 = @ptrCast(&dst.op_params);
    const ncols = ggml.vulkan.ops.u32Of(src0.ne[0]);
    const nrows = ggml.vulkan.ops.u32Of(ggml.types.ggml_nrows(src0));
    const ncols_pad_log2: u32 = @intFromFloat(@ceil(@as(f32, stdlib.libm.log2fInt(ncols))));
    const ncolsp2: u32 = @as(u32, 1) << @intCast(ncols_pad_log2);

    const pc: ggml.vulkan.push.Argsort = .{ .ncols = ncols, .ncols_padded = ncolsp2, .ncols_padded_log2 = ncols_pad_log2, .nrows = nrows, .order = op[0], .outer_start = 0, .outer_end = 0, .inner_start = 0, .inner_end = 0 };

    // Pick the largest workgroup size <= ncolsp2
    const pipeline_idx = @min(ncols_pad_log2, ggml.vulkan.device.num_argsort_pipelines - 1);
    const dev = ctx.device;

    const max_wg_log2 = @min(dev.max_workgroup_size_log2, ggml.vulkan.device.num_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    const use_small = ncols_pad_log2 <= max_wg_log2 and dev.pipeline_argsort_f32[pipeline_idx] != null;
    var pipeline = (if (use_small) dev.pipeline_argsort_f32[pipeline_idx] else dev.pipeline_argsort_large_f32[pipeline_idx]).?;

    const src0_buf = ggml.vulkan.context.tensorSubbuffer(ctx, src0, false);
    const dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, false);
    var subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    if (!use_small) {
        const x_sz: usize = @as(usize, ncolsp2) * nrows * 2 * @sizeOf(c_int);
        if (ctx.prealloc_size_x < x_sz) {
            ctx.prealloc_size_x = x_sz;
            ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
        }
        if (ctx.prealloc_x_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);
        subbuf1 = prealloc(ctx.prealloc_x, 0, ctx.prealloc_x.?.size);
    }

    const elements = [3]u32{ ncolsp2, @min(ggml.vulkan.ops.u32Of(ggml.types.ggml_nrows(src0)), dev.properties.limits.maxComputeWorkGroupCount[1]), 1 };

    // First dispatch initializes tmp_idx and does the passes that stay in a workgroup.
    {
        var pc2 = pc;
        pc2.outer_start = 0;
        pc2.outer_end = @min(ncols_pad_log2, max_wg_log2);
        pc2.inner_start = 0;
        pc2.inner_end = 100;
        ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);
        ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{ src0_buf, subbuf1, dst_buf }, &pc2, elements);
    }
    if (!use_small) {
        ggml.vulkan.context.syncBuffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        var outer = max_wg_log2;
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
                ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);
                ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{ src0_buf, subbuf1, dst_buf }, &pc2, elements);
                ggml.vulkan.context.syncBuffers(ctx, subctx);
            }
        }
        ctx.prealloc_x_need_sync = true;
    }
}

/// Whether the tournament top-k has a pipeline for `k`: the smallest it
/// can use, `max(log2(k) + 1, log2(subgroup))`, exists. The C++ writes
/// this test out in `ggml_vk_topk`, `ggml_vk_can_fuse_topk_qsa` and
/// `supports_op`; past it, top-k falls back to radix-select.
pub fn topkTournamentFits(dev: *const ggml.vulkan.device.Device, k: u32) bool {
    const k_min_pipeline = @max(@as(u32, @intFromFloat(stdlib.libm.log2fInt(k))) + 1, dev.subgroup_size_log2);
    return k_min_pipeline < ggml.vulkan.device.num_topk_pipelines and dev.pipeline_topk_f32[k_min_pipeline] != null;
}

/// Ports `ggml_vk_topk` (ggml-vulkan.cpp:11818 @b1b770d90): passes that each
/// reduce every workgroup's elements to its K largest, double-buffered in
/// scratch, until K remain -- or, past the tournament's K limit, one
/// radix-select per row.
pub fn topk(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const ncols = ggml.vulkan.ops.u32Of(src0.ne[0]);
    const nrows = ggml.vulkan.ops.u32Of(ggml.types.ggml_nrows(src0));
    const k = ggml.vulkan.ops.u32Of(dst.ne[0]);
    const dev = ctx.device;

    // tournament path is faster where it fits; use radix-select only past its k limit
    if (!topkTournamentFits(dev, k)) {
        const pipeline = dev.pipeline_topk_radix_f32 orelse ggml.impl.abort("GGML_ASSERT(pipeline != nullptr) failed");

        if (ctx.prealloc_x_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);

        const rpc: ggml.vulkan.push.TopkRadix = .{ .ncols = ncols, .k = k, .nrows = nrows, .n_tps = 0, .n_blocks = 0, .n_stream = 0 };
        const elements = [3]u32{
            pipeline.wg_denoms[0],
            @min(nrows, dev.properties.limits.maxComputeWorkGroupCount[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        const src0_buf = ggml.vulkan.context.tensorSubbuffer(ctx, src0, false);
        const dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, false);
        ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);
        ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{ src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, &rpc, elements);
        return;
    }

    const pc: ggml.vulkan.push.Topk = .{ .orig_ncols = ncols, .ncols_input = ncols, .ncols_output = ncols, .k = k, .nrows = nrows, .first_pass = 0, .last_pass = 0 };

    if (ctx.prealloc_x_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);

    var elements: [3]u32 = .{ 0, @min(nrows, dev.properties.limits.maxComputeWorkGroupCount[1]), 1 };

    var num_elements = ncols;
    var done_one_iter = false;
    var dbl_buf_index: u32 = 0;
    var dbl_buf_size: usize = 0;
    const ntp: u32 = ggml.vulkan.device.num_topk_pipelines;
    while (num_elements > k or !done_one_iter) {
        // Prefer going as small as num_topk_pipelines - 3 for perf reasons,
        // but a larger K needs a larger workgroup.
        var max_pipeline: u32 = ntp - 1;
        const log2k: u32 = @intFromFloat(stdlib.libm.log2fInt(k));
        const preferred_pipeline = @max(ntp - 3, log2k + 2);
        max_pipeline = @min(preferred_pipeline, max_pipeline);
        var min_pipeline: u32 = log2k + 1;
        // require full subgroup
        min_pipeline = @max(min_pipeline, dev.subgroup_size_log2);

        var pipeline_idx: u32 = @intFromFloat(@ceil(@as(f32, stdlib.libm.log2fInt(num_elements))));
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
            ggml.impl.assert(pipeline_idx >= min_pipeline, "pipeline_idx >= min_pipeline");
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
                ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
            }
        }

        var src_buf: ggml.vulkan.buffer.Subbuffer = undefined;
        var dst_buf: ggml.vulkan.buffer.Subbuffer = undefined;
        if (num_elements == ncols) {
            pc2.first_pass = 1;
            src_buf = ggml.vulkan.context.tensorSubbuffer(ctx, src0, false);
        } else {
            src_buf = prealloc(ctx.prealloc_x, dbl_buf_index * dbl_buf_size, dbl_buf_size);
        }
        if (num_dst_elements == k) {
            pc2.last_pass = 1;
            dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, false);
        } else {
            dst_buf = prealloc(ctx.prealloc_x, (dbl_buf_index ^ 1) * dbl_buf_size, dbl_buf_size);
        }

        elements[0] = num_elements;
        ggml.vulkan.context.requestDescriptorSets(ctx, pl, 1);
        ggml.vulkan.context.dispatch(ctx, subctx, pl, &.{ src_buf, dst_buf }, &pc2, elements);
        num_elements = num_dst_elements;
        dbl_buf_index ^= 1;
        if (num_elements > k) ggml.vulkan.context.syncBuffers(ctx, subctx);
        done_one_iter = true;
    }
    ctx.prealloc_x_need_sync = true;
}

/// Ports `ggml_vk_topk_qsa` (ggml-vulkan.cpp:11951 @b1b770d90): the qwen4
/// QSA indexer's GET_ROWS, PERMUTE, CONT, CPY, RESHAPE, ADD and TOP_K as one
/// radix-select that gathers each block's score to its cells and adds the
/// raw f16 mask on the fly.
pub fn topkQsa(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, cgraph: *const ggml.impl.CGraph, node_idx: c_int) void {
    const get_rows = at(cgraph, node_idx + 0);
    const add = at(cgraph, node_idx + ctx.num_additional_fused_ops - 1);
    const top_k = at(cgraph, node_idx + ctx.num_additional_fused_ops);

    const scores: *const ggml.impl.ggml_c.ggml_tensor = get_rows.src[0]; // [n_tps, n_blocks, n_stream]
    const cell_blk: *const ggml.impl.ggml_c.ggml_tensor = get_rows.src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    var mask: *const ggml.impl.ggml_c.ggml_tensor = add.src[1];
    while (mask.op == ggml.impl.ggml_c.GGML_OP_RESHAPE or mask.op == ggml.impl.ggml_c.GGML_OP_CPY) {
        mask = mask.src[0];
    }

    const n_tps = ggml.vulkan.ops.u32Of(scores.ne[0]);
    const n_blocks = ggml.vulkan.ops.u32Of(scores.ne[1]);
    const n_stream = ggml.vulkan.ops.u32Of(scores.ne[2]);
    const n_kv = ggml.vulkan.ops.u32Of(cell_blk.ne[0]);
    const width = ggml.vulkan.ops.u32Of(top_k.ne[0]);
    const nrows = n_tps *% n_stream;

    const dev = ctx.device;
    const pipeline = dev.pipeline_topk_radix_qsa orelse ggml.impl.abort("GGML_ASSERT(pipeline != nullptr) failed");

    // scratch holds the gathered+masked input, materialized once and reused across passes
    const scratch_size: usize = @as(usize, n_kv) * nrows * @sizeOf(f32);
    if (ctx.prealloc_size_x < scratch_size) {
        ctx.prealloc_size_x = scratch_size;
        ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
    }
    if (ctx.prealloc_x_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);

    const pc: ggml.vulkan.push.TopkRadix = .{ .ncols = n_kv, .k = width, .nrows = nrows, .n_tps = n_tps, .n_blocks = n_blocks, .n_stream = n_stream };
    const elements = [3]u32{
        pipeline.wg_denoms[0],
        @min(nrows, dev.properties.limits.maxComputeWorkGroupCount[1]),
        1,
    };
    const scratch_buf = prealloc(ctx.prealloc_x, 0, ctx.prealloc_x.?.size);
    ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);
    ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{
        ggml.vulkan.context.tensorSubbuffer(ctx, scores, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, top_k, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, cell_blk, false),
        ggml.vulkan.context.tensorSubbuffer(ctx, mask, false),
        scratch_buf,
    }, &pc, elements);
    ctx.prealloc_x_need_sync = true;
}

/// Ports `ggml_vk_sum`, `ggml_vk_sum_rows` and `ggml_vk_mean`
/// (ggml-vulkan.cpp:12000, 12005, 12010 @b1b770d90).
pub fn sumRows(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op) void {
    var p = ggml.vulkan.ops.sumRowsPushConstantsInit(src0, dst, if (op == ggml.impl.ggml_c.GGML_OP_SUM) ggml.types.ggml_nelements(src0) else src0.ne[0]);
    if (op == ggml.impl.ggml_c.GGML_OP_MEAN) p.weight = 1.0 / @as(f32, @floatFromInt(src0.ne[0]));
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, op, p);
}

/// Ports `ggml_vk_cumsum` (ggml-vulkan.cpp:12016 @b1b770d90): one pass for
/// short rows or many rows, otherwise block partials and a second pass.
pub fn cumsum(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const pc = ggml.vulkan.ops.sumRowsPushConstantsInit(src0, dst, src0.ne[0]);
    if (dst.ne[0] <= 4096 or ggml.types.ggml_nrows(dst) >= ctx.device.shader_core_count) {
        ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_CUMSUM, pc);
        return;
    }

    const dev = ctx.device;
    const p1 = dev.pipeline_cumsum_multipass1_f32 orelse ggml.impl.abort("GGML_ASSERT(pipeline1 != nullptr && pipeline2 != nullptr) failed");
    const p2 = dev.pipeline_cumsum_multipass2_f32 orelse ggml.impl.abort("GGML_ASSERT(pipeline1 != nullptr && pipeline2 != nullptr) failed");
    ggml.vulkan.context.requestDescriptorSets(ctx, p1, 1);
    ggml.vulkan.context.requestDescriptorSets(ctx, p2, 1);

    const elements = [3]u32{ ggml.vulkan.ops.u32Of(dst.ne[0]), ggml.vulkan.ops.u32Of(ggml.types.ggml_nrows(dst)), 1 };
    const temp_size: usize = @sizeOf(f32) * @as(usize, elements[0]) * @as(usize, @intCast(ggml.types.ggml_nrows(dst)));
    if (ctx.prealloc_size_split_k < temp_size) {
        ctx.prealloc_size_split_k = temp_size;
        ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
    }

    const src_buf = ggml.vulkan.context.tensorSubbuffer(ctx, src0, false);
    const dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, false);
    const temp_buf = ggml.vulkan.context.subbuffer(ctx, ctx.prealloc_split_k.?, 0);

    if (ctx.prealloc_split_k_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);

    ggml.vulkan.context.dispatch(ctx, subctx, p1, &.{ src_buf, dst_buf, temp_buf }, &pc, elements);
    ggml.vulkan.context.syncBuffers(ctx, subctx);
    ggml.vulkan.context.dispatch(ctx, subctx, p2, &.{ src_buf, dst_buf, temp_buf }, &pc, elements);

    ctx.prealloc_split_k_need_sync = true;
}

/// Ports `ggml_vk_nrows_elements` (ggml-vulkan.cpp:12063 @b1b770d90): the
/// dispatch size for one workgroup per row, folded into y and z past the
/// x limit.
fn nrowsElements(nr: u32) [3]u32 {
    if (nr > 262144) return .{ 512, 512, ggml.vulkan.ops.ceilDiv(nr, 262144) };
    if (nr > 512) return .{ 512, ggml.vulkan.ops.ceilDiv(nr, 512), 1 };
    return .{ nr, 1, 1 };
}

/// Ports `ggml_vk_cross_entropy_loss` (ggml-vulkan.cpp:12073 @b1b770d90):
/// one loss per row into scratch, then `sum_rows` over them into the scalar.
pub fn crossEntropyLoss(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0: *const ggml.impl.ggml_c.ggml_tensor = dst.src[0];
    const src1: *const ggml.impl.ggml_c.ggml_tensor = dst.src[1];

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type  == GGML_TYPE_F32");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    ggml.impl.assert(ggml.types.ggml_are_same_shape(src0, src1), "ggml_are_same_shape(src0, src1)");
    ggml.impl.assert(ggml.types.ggml_is_scalar(dst), "ggml_is_scalar(dst)");

    const nclasses: u32 = @truncate(@as(u64, @bitCast(src0.ne[0])));
    const nrows: u32 = @truncate(@as(u64, @bitCast(ggml.types.ggml_nrows(src0))));

    const dev = ctx.device;
    const pipeline = ggml.vulkan.op_pipeline.getPipeline(ctx, src0, src1, null, dst, ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS) orelse ggml.impl.abort("GGML_ASSERT(pipeline != nullptr) failed");

    ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);
    ggml.vulkan.context.requestDescriptorSets(ctx, dev.pipeline_sum_rows_f32.?, 1);

    const src0_buf = ggml.vulkan.context.tensorSubbuffer(ctx, src0, false);
    const src1_buf = ggml.vulkan.context.tensorSubbuffer(ctx, src1, false);
    const dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, true);

    const pc = opPc(nclasses, nrows, 0.0, 0.0);

    const tmp_size: usize = @as(usize, nrows) * @sizeOf(f32);
    if (ctx.prealloc_size_x < tmp_size) {
        ctx.prealloc_size_x = tmp_size;
        ggml.vulkan.nodes.preallocateBuffers(ctx, subctx);
    }
    if (ctx.prealloc_x_need_sync) ggml.vulkan.context.syncBuffers(ctx, subctx);

    const tmp_buf = prealloc(ctx.prealloc_x, 0, tmp_size);
    ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, tmp_buf }, &pc, nrowsElements(nrows));
    ggml.vulkan.context.syncBuffers(ctx, subctx);

    var sp: ggml.vulkan.push.SumRows = .{};
    sp.n_cols = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.ne03 = 1;
    sp.weight = 1.0;
    ggml.vulkan.ops.initPushconstFastdiv(&sp);
    sp.misalign_offsets = @truncate(ggml.vulkan.context.getMisalignBytes(ctx, dst) / ggml.types.ggml_type_size(dst.type));

    ggml.vulkan.context.dispatch(ctx, subctx, dev.pipeline_sum_rows_f32.?, &.{ tmp_buf, dst_buf }, &sp, .{ 1, 1, 1 });
    ctx.prealloc_x_need_sync = true;
}

/// Ports `ggml_vk_cross_entropy_loss_back` (ggml-vulkan.cpp:12127 @b1b770d90).
pub fn crossEntropyLossBack(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const grad: *const ggml.impl.ggml_c.ggml_tensor = dst.src[0];
    const logits: *const ggml.impl.ggml_c.ggml_tensor = dst.src[1];
    const labels: *const ggml.impl.ggml_c.ggml_tensor = dst.src[2];

    ggml.impl.assert(grad.type == ggml.impl.ggml_c.GGML_TYPE_F32, "grad->type   == GGML_TYPE_F32");
    ggml.impl.assert(logits.type == ggml.impl.ggml_c.GGML_TYPE_F32, "logits->type == GGML_TYPE_F32");
    ggml.impl.assert(labels.type == ggml.impl.ggml_c.GGML_TYPE_F32, "labels->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type    == GGML_TYPE_F32");
    ggml.impl.assert(ggml.types.ggml_is_scalar(grad), "ggml_is_scalar(grad)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(grad), "ggml_is_contiguous(grad)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(logits), "ggml_is_contiguous(logits)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(labels), "ggml_is_contiguous(labels)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    ggml.impl.assert(ggml.types.ggml_are_same_shape(logits, labels), "ggml_are_same_shape(logits, labels)");
    ggml.impl.assert(ggml.types.ggml_are_same_shape(logits, dst), "ggml_are_same_shape(logits, dst)");

    const nclasses: u32 = @truncate(@as(u64, @bitCast(logits.ne[0])));
    const nrows: u32 = @truncate(@as(u64, @bitCast(ggml.types.ggml_nrows(logits))));

    const pipeline = ggml.vulkan.op_pipeline.getPipeline(ctx, grad, logits, labels, dst, ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS_BACK) orelse ggml.impl.abort("GGML_ASSERT(pipeline != nullptr) failed");

    ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);

    const grad_buf = ggml.vulkan.context.tensorSubbuffer(ctx, grad, false);
    const logits_buf = ggml.vulkan.context.tensorSubbuffer(ctx, logits, false);
    const labels_buf = ggml.vulkan.context.tensorSubbuffer(ctx, labels, false);
    const dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, false);

    const pc = opPc(nclasses, nrows, 0.0, 0.0);
    ggml.vulkan.context.dispatch(ctx, subctx, pipeline, &.{ grad_buf, logits_buf, labels_buf, dst_buf }, &pc, nrowsElements(nrows));
}

/// Ports `ggml_vk_argmax` (ggml-vulkan.cpp:12161 @b1b770d90).
pub fn argmax(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, null, null, null, dst, ggml.impl.ggml_c.GGML_OP_ARGMAX, opPc(ggml.vulkan.ops.u32Of(src0.ne[0]), ggml.vulkan.ops.u32Of(src0.ne[1]), 0, 0));
}

/// Ports `ggml_vk_count_equal` (ggml-vulkan.cpp:12165 @b1b770d90).
pub fn countEqual(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL, opPc(ggml.vulkan.ops.u32Of(ggml.types.ggml_nelements(src0)), 0, 0, 0));
}

/// Ports `ggml_vk_solve_tri` (ggml-vulkan.cpp:12169 @b1b770d90).
pub fn solveTri(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.vulkan.ops.opF32(ctx, subctx, src0, src1, null, null, dst, ggml.impl.ggml_c.GGML_OP_SOLVE_TRI, ggml.vulkan.ops.binaryPc(ggml.vulkan.ops.u32Of(ggml.types.ggml_nelements(src0)), src0, src1, dst, 0, 0, 0));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
