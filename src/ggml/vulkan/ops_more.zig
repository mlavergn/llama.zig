//! Ops with more to them than one `op_f32` call: rms_norm and its fusions,
//! RoPE's constants, the recurrent and state-space kernels, the lightning
//! indexer, the optimizer steps.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.1
//! (`b29c606e2`).

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
const typeSize = ops.typeSize;

extern fn powf(x: f32, y: f32) f32;

fn at(cg: *const CGraph, i: c_int) *Tensor {
    return cg.nodes[@intCast(i)].?;
}

fn f32Param(t: *const Tensor, i: usize) f32 {
    return @bitCast(t.op_params[i]);
}

/// Ports `ggml_vk_make_rope_constants` (ggml-vulkan.cpp:13768 @b29c606e2).
pub fn makeRopeConstants(dst: *const Tensor, src0: *const Tensor, has_ff: bool, backprop: bool, set_rows_stride: u32) push.Rope {
    const op = dst.op_params;
    const n_dims = op[1];
    const mode = op[2];
    const n_offs = op[15];
    const n_ctx_orig = op[4];
    const freq_base = f32Param(dst, 5);
    const freq_scale = f32Param(dst, 6);
    const ext_factor = f32Param(dst, 7);
    const attn_factor = f32Param(dst, 8);
    const beta_fast = f32Param(dst, 9);
    const beta_slow = f32Param(dst, 10);
    var sections: [4]i32 = .{ 0, 0, 0, 0 };
    if (mode & c.GGML_ROPE_TYPE_MROPE != 0) sections = op[11..15].*;

    const is_imrope = mode == c.GGML_ROPE_TYPE_IMROPE;

    var corr_dims: [2]f32 = undefined;
    c.ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr_dims);

    const theta_scale = powf(freq_base, -2.0 / @as(f32, @floatFromInt(n_dims)));

    const ts0 = c.ggml_type_size(src0.type);
    const tsd = c.ggml_type_size(dst.type);
    return .{
        .rope_mode = @bitCast(mode),
        .nrows = u32Of(c.ggml_nrows(src0)),
        .n_dims = @bitCast(n_dims),
        .n_offs = @bitCast(n_offs),
        .freq_scale = freq_scale,
        .freq_base = freq_base,
        .ext_factor = ext_factor,
        .attn_factor = attn_factor,
        .corr_dims = corr_dims,
        .theta_scale = theta_scale,
        .has_ff = @intFromBool(has_ff),
        .sections = sections,
        .is_imrope = @intFromBool(is_imrope),
        .is_back = @intFromBool(backprop),
        .set_rows_stride = set_rows_stride,
        .ne00 = u32Of(src0.ne[0]),
        .ne01 = u32Of(src0.ne[1]),
        .ne02 = u32Of(src0.ne[2]),
        .nb01 = @truncate(src0.nb[1] / ts0),
        .nb02 = @truncate(src0.nb[2] / ts0),
        .nb03 = @truncate(src0.nb[3] / ts0),
        .nb11 = @truncate(dst.nb[1] / tsd),
        .nb12 = @truncate(dst.nb[2] / tsd),
        .nb13 = @truncate(dst.nb[3] / tsd),
        .a_offset = 0, // filled in by init_pushconst_tensor_offsets
        .d_offset = 0,
    };
}

/// A tensor's binding the way the fused dispatches find it: through
/// pinned memory on a UMA device, its own buffer otherwise.
fn bindingOf(ctx: *const BackendCtx, t: *const Tensor, buf: **m.buffer.Buffer, offset: *usize) void {
    if (ctx.device.uma) {
        if (m.buffer.hostGet(ctx.device, t.data, offset)) |b| {
            buf.* = b;
            return;
        }
    }
    buf.* = m.buffer.tensorBuffer(t);
    offset.* = m.buffer.tensorOffset(t) + t.view_offs;
}

/// Ports `ggml_vk_rms_norm_push_constants` (ggml-vulkan.cpp:13816 @b29c606e2).
fn rmsNormPushConstants(src0: *const Tensor, src1: *const Tensor, dst: *const Tensor, eps: f32, num_partials: u32) push.Binary {
    return ops.binaryPc(u32Of(c.ggml_nelements(src0)), src0, src1, dst, eps, 0, @bitCast(num_partials));
}

/// Ports `ggml_vk_rms_norm_finish` (ggml-vulkan.cpp:13833 @b29c606e2).
fn rmsNormFinish(ctx: *BackendCtx, src0: *const Tensor) void {
    if (ctx.do_add_rms_partials_offset_calculation) {
        ctx.prealloc_size_add_rms_partials_offset += ops.rmsPartialsSize(ctx, src0);
        ctx.do_add_rms_partials = false;
        ctx.do_add_rms_partials_offset_calculation = false;
    }
}

/// Ports `ggml_vk_rms_norm` (ggml-vulkan.cpp:13841 @b29c606e2): plain, or
/// heading whichever fusion `fused_rms_norm_mode` names -- MUL; MUL and ADD
/// (and a second MUL); MUL and ROPE (and the SET_ROWS a view of the rope
/// feeds); or the SET_ROWS a view of the norm feeds.
pub fn rmsNorm(ctx: *BackendCtx, subctx: *Context, cgraph: *const CGraph, node_idx: c_int) void {
    const rms = at(cgraph, node_idx);
    const src0: *const Tensor = rms.src[0];
    const eps = f32Param(rms, 0);
    const mode = ctx.fused_rms_norm_mode;

    if (mode == .view_set_rows) {
        impl.assert(ctx.num_additional_fused_ops == 2, "ctx->num_additional_fused_ops == 2");
        const set_rows = at(cgraph, node_idx + 2);
        const indices: *const Tensor = set_rows.src[1];
        var pc = rmsNormPushConstants(src0, src0, set_rows, eps, 0);
        ops.initPushconstTensorOffsets(ctx, &pc, src0, src0, null, null, set_rows);

        const pipeline = (if (set_rows.type == c.GGML_TYPE_F16) ctx.device.pipeline_rms_norm_set_rows_f32_f16 else ctx.device.pipeline_rms_norm_set_rows_f32_f32).?;
        m.context.requestDescriptorSets(ctx, pipeline, 1);
        m.context.dispatch(ctx, subctx, pipeline, &.{
            m.context.tensorSubbuffer(ctx, src0, true),
            m.context.tensorSubbuffer(ctx, src0, true),
            m.context.tensorSubbuffer(ctx, set_rows, true),
            m.context.tensorSubbuffer(ctx, indices, false),
        }, &pc, .{ u32Of(src0.ne[1]), u32Of(src0.ne[2]), u32Of(src0.ne[3]) });
        rmsNormFinish(ctx, src0);
        return;
    }

    if (mode == .mul_add or mode == .mul_add_mul) {
        const mul = at(cgraph, node_idx + 1);
        const add = at(cgraph, node_idx + 2);
        const weight: *const Tensor = if (mul.src[0] == rms) mul.src[1] else mul.src[0];
        const residual: *const Tensor = if (add.src[0] == mul) add.src[1] else add.src[0];
        const do_post_multiply = mode == .mul_add_mul;
        impl.assert(ctx.num_additional_fused_ops == @as(c_int, if (do_post_multiply) 3 else 2), "ctx->num_additional_fused_ops == (do_post_multiply ? 3 : 2)");
        const dst = if (do_post_multiply) at(cgraph, node_idx + 3) else add;
        const post_scale: *const Tensor = if (do_post_multiply) (if (dst.src[0] == add) dst.src[1] else dst.src[0]) else src0;

        const num_partials: u32 = if (ctx.do_add_rms_partials) ops.rmsNumPartials(ctx, dst) else 0;
        var pc = rmsNormPushConstants(src0, weight, dst, eps, num_partials);
        ops.initPushconstTensorOffsets(ctx, &pc, src0, weight, residual, post_scale, dst);

        const dev = ctx.device;
        const pipeline = (if (ctx.do_add_rms_partials)
            (if (do_post_multiply) dev.pipeline_rms_norm_mul_add_mul_partials_f32 else dev.pipeline_rms_norm_mul_add_partials_f32)
        else
            (if (do_post_multiply) dev.pipeline_rms_norm_mul_add_mul_f32 else dev.pipeline_rms_norm_mul_add_f32)).?;
        m.context.requestDescriptorSets(ctx, pipeline, 1);
        if (ctx.do_add_rms_partials) {
            m.context.dispatch(ctx, subctx, pipeline, &.{
                m.context.tensorSubbuffer(ctx, src0, true),
                m.context.tensorSubbuffer(ctx, weight, true),
                m.context.tensorSubbuffer(ctx, dst, true),
                m.context.subbuffer(ctx, ctx.prealloc_add_rms_partials.?, ctx.prealloc_size_add_rms_partials_offset),
                m.context.tensorSubbuffer(ctx, residual, false),
                m.context.tensorSubbuffer(ctx, post_scale, false),
            }, &pc, .{ u32Of(@divTrunc(src0.ne[0] + 127, 128)), 1, 1 });
        } else {
            m.context.dispatch(ctx, subctx, pipeline, &.{
                m.context.tensorSubbuffer(ctx, src0, true),
                m.context.tensorSubbuffer(ctx, weight, true),
                m.context.tensorSubbuffer(ctx, dst, true),
                m.context.tensorSubbuffer(ctx, residual, false),
                m.context.tensorSubbuffer(ctx, post_scale, false),
            }, &pc, .{ u32Of(src0.ne[1]), u32Of(src0.ne[2]), u32Of(src0.ne[3]) });
        }
        rmsNormFinish(ctx, src0);
        return;
    }

    var dst: *Tensor = undefined;
    var src1: *const Tensor = undefined;
    if (mode != .count) {
        const mul = at(cgraph, node_idx + 1);
        dst = mul;
        src1 = if (mul.src[0] == rms) mul.src[1] else mul.src[0];
    } else {
        dst = rms;
        src1 = src0;
    }

    const num_partials: u32 = if (ctx.do_add_rms_partials) ops.rmsNumPartials(ctx, dst) else 0;
    const bin = rmsNormPushConstants(src0, src1, dst, eps, num_partials);

    if (mode == .mul_rope or mode == .mul_rope_view_set_rows) {
        const max_tensors = 7;
        var tensors: [max_tensors]?*const Tensor = @splat(null);

        const mul = at(cgraph, node_idx + 1);
        const rope = at(cgraph, node_idx + 2);
        const other_src: *const Tensor = if (mul.src[0] == rms) mul.src[1] else mul.src[0];
        const do_set_rows = mode == .mul_rope_view_set_rows;
        impl.assert(ctx.num_additional_fused_ops == @as(c_int, if (do_set_rows) 4 else 2), "ctx->num_additional_fused_ops == (do_set_rows ? 4 : 2)");

        tensors[0] = rms.src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope.src[1]; // pos
        tensors[4] = rope.src[2]; // ff
        const t5 = at(cgraph, node_idx + ctx.num_additional_fused_ops); // dst
        tensors[5] = t5;
        tensors[6] = if (do_set_rows) t5.src[1] else null;
        const set_rows_stride: u32 = if (do_set_rows) @truncate(t5.nb[1] / c.ggml_type_size(t5.type)) else 0;

        var pc: push.RmsNormMulRope = .{};
        pc.bin = bin;
        pc.rope = makeRopeConstants(rope, rope.src[0], tensors[4] != null, false, set_rows_stride);

        const pipeline = (if (t5.type == c.GGML_TYPE_F16) ctx.device.pipeline_rms_norm_mul_rope_f32_f16 else ctx.device.pipeline_rms_norm_mul_rope_f32_f32).?;
        m.context.requestDescriptorSets(ctx, pipeline, 1);

        var buf: [max_tensors]*m.buffer.Buffer = undefined;
        var offset: [max_tensors]usize = undefined;
        for (0..max_tensors) |i| {
            const t = tensors[i] orelse {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                offset[i] = 0;
                continue;
            };
            offset[i] = 0;
            bindingOf(ctx, t, &buf[i], &offset[i]);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_offset = m.context.getMisalignBytes(ctx, t5) / typeSize(t5);
        offset[5] &= ~(@as(usize, @intCast(ctx.device.properties.limits.minStorageBufferOffsetAlignment)) - 1);

        const s = rms.src[0].*;
        var subs: [max_tensors]Subbuffer = undefined;
        for (&subs, 0..) |*sb, i| sb.* = m.context.subbuffer(ctx, buf[i], offset[i]);
        m.context.dispatch(ctx, subctx, pipeline, &subs, &pc, .{ u32Of(s.ne[1]), u32Of(s.ne[2]), u32Of(s.ne[3]) });
    } else {
        impl.assert(mode == .mul or mode == .count, "ctx->fused_rms_norm_mode == RMS_NORM_MUL || ctx->fused_rms_norm_mode == RMS_NORM_COUNT");
        ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_RMS_NORM, bin);
    }

    rmsNormFinish(ctx, src0);
}

/// Ports `ggml_vk_op_f32_wkv` (ggml-vulkan.cpp:13120 @b29c606e2) with
/// `ggml_vk_rwkv_wkv6` and `ggml_vk_rwkv_wkv7` (ggml-vulkan.cpp:13161, 13179 @b29c606e2).
pub fn rwkvWkv(ctx: *BackendCtx, subctx: *Context, dst: *Tensor, version: u32) void {
    impl.assert(version == 6 or version == 7, "version == 6 || version == 7");
    const num_srcs: usize = if (version == 6) 6 else 7;
    const s0 = dst.src[0].*;
    const pc: push.RwkvWkv6 = .{
        .B = u32Of(dst.src[num_srcs - 1].*.ne[1]), // n_seqs
        .T = u32Of(s0.ne[2]), // seq_length
        .C = u32Of(dst.ne[0]), // n_embed
        .H = u32Of(s0.ne[1]), // n_heads
    };
    for (0..num_srcs) |i| impl.assert(!c.ggml_is_quantized(dst.src[i].*.type), "!ggml_is_quantized(dst->src[i]->type)");
    impl.assert(dst.buffer != null, "dst->buffer != nullptr");

    const pipeline = m.op_pipeline.getPipeline(ctx, dst.src[0], dst.src[1], dst.src[2], dst, dst.op) orelse impl.abort("GGML_ASSERT(pipeline != nullptr) failed");
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    var bufs: [8]Subbuffer = undefined;
    for (0..num_srcs) |i| bufs[i] = m.context.tensorSubbuffer(ctx, dst.src[i], false);
    bufs[num_srcs] = m.context.tensorSubbuffer(ctx, dst, false);
    m.context.dispatch(ctx, subctx, pipeline, bufs[0 .. num_srcs + 1], &pc, .{ pc.B *% pc.H, 1, 1 });
}

/// Ports `ggml_vk_gated_linear_attn` (ggml-vulkan.cpp:13197 @b29c606e2).
pub fn gatedLinearAttn(ctx: *BackendCtx, subctx: *Context, dst: *Tensor) void {
    const s0 = dst.src[0].*;
    const n_seqs = u32Of(dst.src[4].*.ne[1]);
    const n_heads = u32Of(s0.ne[1]);
    impl.assert(dst.buffer != null, "dst->buffer != nullptr");
    const pipeline = m.op_pipeline.getPipeline(ctx, dst.src[0], dst.src[1], dst.src[2], dst, dst.op) orelse impl.abort("GGML_ASSERT(pipeline != nullptr) failed");
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    var bufs: [6]Subbuffer = undefined;
    for (0..5) |i| bufs[i] = m.context.tensorSubbuffer(ctx, dst.src[i], false);
    bufs[5] = m.context.tensorSubbuffer(ctx, dst, false);
    const pc: push.GatedLinearAttn = .{ .B = n_seqs, .T = u32Of(s0.ne[2]), .C = u32Of(dst.ne[0]), .H = n_heads, .scale = f32Param(dst, 0) };
    // (uint32_t)(n_seqs * n_heads) of two size_t
    m.context.dispatch(ctx, subctx, pipeline, &bufs, &pc, .{ u32Of(@as(i64, @intCast(dst.src[4].*.ne[1])) *% s0.ne[1]), 1, 1 });
}

/// Ports `ggml_vk_lightning_indexer` (ggml-vulkan.cpp:13232 @b29c606e2): one
/// score per (kv cell, token, stream) from the indexer's q, k, head weights
/// and mask.
pub fn lightningIndexer(ctx: *BackendCtx, subctx: *Context, dst: *Tensor) void {
    const q: *const Tensor = dst.src[0];
    const k: *const Tensor = dst.src[1];
    const w: *const Tensor = dst.src[2];
    const mk: *const Tensor = dst.src[3];

    const pipeline = m.op_pipeline.getPipeline(ctx, q, k, w, dst, dst.op) orelse impl.abort("GGML_ASSERT(pipeline != nullptr) failed");

    m.context.requestDescriptorSets(ctx, pipeline, 1);

    const n_kv = u32Of(k.ne[2]);
    const n_heads = u32Of(q.ne[1]);
    const n_tokens = u32Of(q.ne[2]);
    const n_streams = u32Of(q.ne[3]);
    const n_masks = u32Of(mk.ne[3]);

    const n_outputs = u32Of(dst.ne[0] *% dst.ne[1] *% dst.ne[3]);
    const dispatch_x = @min(n_outputs, ctx.device.properties.limits.maxComputeWorkGroupCount[0]);
    const dispatch_y = ops.ceilDiv(n_outputs, dispatch_x);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    const f = @sizeOf(f32);
    const h = @sizeOf(c.ggml_fp16_t);
    const pc: push.LightningIndexer = .{
        .n_kv = n_kv,
        .n_heads = n_heads,
        .n_tokens = n_tokens,
        .n_streams = n_streams,
        .n_masks = n_masks,
        .dispatch_x = dispatch_x,
        .q_nb1 = u32Of(q.nb[1] / f),
        .q_nb2 = u32Of(q.nb[2] / f),
        .q_nb3 = u32Of(q.nb[3] / f),
        .k_nb2 = u32Of(k.nb[2]),
        .k_nb3 = u32Of(k.nb[3]),
        .w_nb1 = u32Of(w.nb[1] / f),
        .w_nb3 = u32Of(w.nb[3] / f),
        .m_nb1 = u32Of(mk.nb[1] / h),
        .m_nb3 = u32Of(mk.nb[3] / h),
        .d_nb1 = u32Of(dst.nb[1] / f),
        .d_nb3 = u32Of(dst.nb[3] / f),
    };

    m.context.dispatch(ctx, subctx, pipeline, &.{
        m.context.tensorSubbuffer(ctx, q, false),
        m.context.tensorSubbuffer(ctx, k, false),
        m.context.tensorSubbuffer(ctx, w, false),
        m.context.tensorSubbuffer(ctx, mk, false),
        m.context.tensorSubbuffer(ctx, dst, false),
    }, &pc, .{ dispatch_x, dispatch_y, 1 });
}

/// Ports `ggml_vk_gated_delta_net` (ggml-vulkan.cpp:13281 @b29c606e2).
pub fn gatedDeltaNet(ctx: *BackendCtx, subctx: *Context, dst: *Tensor) void {
    const src_q = dst.src[0].*;
    const src_v = dst.src[2].*;
    const src_beta = dst.src[4].*;
    impl.assert(dst.buffer != null, "dst->buffer != nullptr");

    const S_v = u32Of(src_v.ne[0]);
    const H = u32Of(src_v.ne[1]);
    const n_tokens = u32Of(src_v.ne[2]);
    const n_seqs = u32Of(src_v.ne[3]);
    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const K: u32 = @bitCast(dst.op_params[0]);
    const s_off = S_v *% H *% n_tokens *% n_seqs;

    const pipeline = m.op_pipeline.getPipeline(ctx, dst.src[0], dst.src[1], dst.src[2], dst, dst.op) orelse impl.abort("GGML_ASSERT(pipeline != nullptr) failed");
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    var bufs: [7]Subbuffer = undefined;
    for (0..6) |i| bufs[i] = m.context.tensorSubbuffer(ctx, dst.src[i], false);
    bufs[6] = m.context.tensorSubbuffer(ctx, dst, false);

    const f = @sizeOf(f32);
    const pc: push.GatedDeltaNet = .{
        .H = H,
        .n_tokens = n_tokens,
        .n_seqs = n_seqs,
        .s_off = s_off,
        .sq1 = @truncate(src_q.nb[1] / f),
        .sq2 = @truncate(src_q.nb[2] / f),
        .sq3 = @truncate(src_q.nb[3] / f),
        .sv1 = @truncate(src_v.nb[1] / f),
        .sv2 = @truncate(src_v.nb[2] / f),
        .sv3 = @truncate(src_v.nb[3] / f),
        .sb1 = @truncate(src_beta.nb[1] / f),
        .sb2 = @truncate(src_beta.nb[2] / f),
        .sb3 = @truncate(src_beta.nb[3] / f),
        .neq1 = u32Of(src_q.ne[1]),
        .rq3 = u32Of(@divTrunc(src_v.ne[3], src_q.ne[3])),
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(S_v))),
        .K = K,
    };
    m.context.dispatch(ctx, subctx, pipeline, &bufs, &pc, .{ H, n_seqs, S_v });
}

/// Ports `ggml_vk_ssm_scan` (ggml-vulkan.cpp:13338 @b29c606e2): Mamba-2 only.
pub fn ssmScan(ctx: *BackendCtx, subctx: *Context, dst: *Tensor) void {
    const src0 = dst.src[0].*;
    const src1 = dst.src[1].*;
    const src2 = dst.src[2].*;
    const src3 = dst.src[3].*;
    const src4 = dst.src[4].*;
    const src5 = dst.src[5].*;
    impl.assert(dst.buffer != null, "dst->buffer != nullptr");

    const head_dim = u32Of(src0.ne[1]);
    const n_head = u32Of(src1.ne[1]);
    const n_group = u32Of(src4.ne[1]);
    const n_tok = u32Of(src1.ne[2]);
    const n_seq = u32Of(src1.ne[3]);

    const is_mamba2 = src3.nb[1] == @sizeOf(f32);
    impl.assert(is_mamba2, "is_mamba2");

    const pipeline = m.op_pipeline.getPipeline(ctx, dst.src[0], dst.src[1], dst.src[2], dst, dst.op) orelse impl.abort("GGML_ASSERT(pipeline != nullptr) failed");
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    const s_off: i64 = c.ggml_nelements(dst.src[1]) * @sizeOf(f32);
    const pc: push.SsmScan = .{
        .nb02 = @truncate(src0.nb[2]),
        .nb03 = @truncate(src0.nb[3]),
        .nb12 = @truncate(src1.nb[2]),
        .nb13 = @truncate(src1.nb[3]),
        .nb21 = @truncate(src2.nb[1]),
        .nb22 = @truncate(src2.nb[2]),
        .nb31 = @truncate(src3.nb[1]),
        .nb42 = @truncate(src4.nb[2]),
        .nb43 = @truncate(src4.nb[3]),
        .nb52 = @truncate(src5.nb[2]),
        .nb53 = @truncate(src5.nb[3]),
        .s_off = u32Of(s_off),
        .n_head = n_head,
        .d_head = head_dim,
        .n_group = n_group,
        .n_tok = n_tok,
        .n_seq = n_seq,
        .K = @bitCast(dst.op_params[0]),
    };

    // A missing source binds an empty subbuffer; the C++ stops at the first.
    var bufs: [8]Subbuffer = @splat(.{});
    var i: usize = 0;
    while (i < 7 and dst.src[i] != null) : (i += 1) bufs[i] = m.context.tensorSubbuffer(ctx, dst.src[i], false);
    bufs[7] = m.context.tensorSubbuffer(ctx, dst, false);

    const d_state = u32Of(src0.ne[0]);
    const num_subgroups = d_state / ctx.device.subgroup_size;
    const num_workgroups_x = ops.ceilDiv(n_head *% head_dim, num_subgroups);
    m.context.dispatch(ctx, subctx, pipeline, &bufs, &pc, .{ num_workgroups_x, n_seq, 1 });
}

/// Ports `ggml_vk_ssm_conv` (ggml-vulkan.cpp:13395 @b29c606e2): plain, or
/// fused with SILU, or with a bias ADD and SILU.
pub fn ssmConv(ctx: *BackendCtx, subctx: *Context, cgraph: *const CGraph, node_idx: c_int) void {
    const conv = at(cgraph, node_idx);
    const src0 = conv.src[0].*;
    const src1 = conv.src[1].*;

    var dst: *Tensor = conv;
    var bias: ?*const Tensor = null;
    if (ctx.num_additional_fused_ops == 1) {
        dst = at(cgraph, node_idx + 1); // silu
    } else if (ctx.num_additional_fused_ops == 2) {
        const add = at(cgraph, node_idx + 1);
        bias = if (add.src[0] == conv) add.src[1] else add.src[0];
        dst = at(cgraph, node_idx + 2); // silu
    }
    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    const src2: *const Tensor = bias orelse conv.src[0];

    ops.opF32(ctx, subctx, conv.src[0], conv.src[1], src2, null, dst, c.GGML_OP_SSM_CONV, push.SsmConv{
        .nb01 = @truncate(src0.nb[1]),
        .nb02 = @truncate(src0.nb[2]),
        .nb11 = @truncate(src1.nb[1]),
        .dst_nb0 = @truncate(dst.nb[0]),
        .dst_nb1 = @truncate(dst.nb[1]),
        .dst_nb2 = @truncate(dst.nb[2]),
        .nc = u32Of(src1.ne[0]),
        .ncs = u32Of(src0.ne[0]),
        .nr = u32Of(src0.ne[1]),
        .n_t = u32Of(dst.ne[1]),
        .n_s = u32Of(dst.ne[2]),
    });
}

/// Ports `ggml_vk_op_f32_opt_step_adamw` and `ggml_vk_opt_step_adamw`
/// (ggml-vulkan.cpp:13428, 13469 @b29c606e2).
pub fn optStepAdamw(ctx: *BackendCtx, subctx: *Context, dst: *Tensor) void {
    const x = dst.src[0];
    const g = dst.src[1];
    const gm = dst.src[2];
    const gv = dst.src[3];
    const p = dst.src[4];
    for ([_]*const Tensor{ x, g, gm, gv, p }) |t| impl.assert(t.type == c.GGML_TYPE_F32, "->type == GGML_TYPE_F32");
    impl.assert(dst.buffer != null, "dst->buffer != nullptr");
    for ([_]*const Tensor{ x, g, gm, gv, p }) |t| impl.assert(c.ggml_is_contiguous(t), "ggml_is_contiguous");
    impl.assert(c.ggml_are_same_shape(x, g), "ggml_are_same_shape(x, g)");
    impl.assert(c.ggml_are_same_shape(x, gm), "ggml_are_same_shape(x, gm)");
    impl.assert(c.ggml_are_same_shape(x, gv), "ggml_are_same_shape(x, gv)");
    impl.assert(c.ggml_nelements(p) == 7, "ggml_nelements(p) == 7");

    const pipeline = m.op_pipeline.getPipeline(ctx, g, gm, gv, dst, c.GGML_OP_OPT_STEP_ADAMW) orelse impl.abort("GGML_ASSERT(pipeline != nullptr) failed");
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    const n = u32Of(c.ggml_nelements(x));
    const pc: push.Op = .{ .KX = n, .KY = 0, .param1 = 0, .param2 = 0, .param3 = 0, .param4 = 0 };
    m.context.dispatch(ctx, subctx, pipeline, &.{
        m.context.tensorSubbuffer(ctx, x, false),
        m.context.tensorSubbuffer(ctx, g, false),
        m.context.tensorSubbuffer(ctx, gm, false),
        m.context.tensorSubbuffer(ctx, gv, false),
        m.context.tensorSubbuffer(ctx, p, false),
    }, &pc, .{ n, 1, 1 });
}

/// Ports `ggml_vk_opt_step_sgd` (ggml-vulkan.cpp:13478 @b29c606e2).
pub fn optStepSgd(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, src2: *const Tensor, dst: *Tensor) void {
    const n = u32Of(c.ggml_nelements(dst.src[0]));
    ops.opF32(ctx, subctx, src0, src1, src2, null, dst, c.GGML_OP_OPT_STEP_SGD, push.Op{ .KX = n, .KY = 0, .param1 = 0, .param2 = 0, .param3 = 0, .param4 = 0 });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
