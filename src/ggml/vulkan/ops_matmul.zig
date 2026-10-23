//! Matrix multiplication: the pipeline getters, tile and split-k choice,
//! the matmul dispatches, and the matrix-matrix path that dequantizes or
//! quantizes its operands into scratch first.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const vk = m.common.vk;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;
const Tensor = c.ggml_tensor;
const Buffer = m.buffer.Buffer;
const Subbuffer = m.buffer.Subbuffer;
const Pipeline = m.pipeline.Pipeline;
const MatmulPipeline = m.pipeline.MatmulPipeline;
const push = m.push;
const ops = m.ops;
const ceilDiv = ops.ceilDiv;

/// Mirror `DMMV_WG_SIZE_SUBGROUP` and `DMMV_WG_SIZE_LARGE`
/// (ggml-vulkan.cpp:547, 548 @c1d0e7a00), the indices into the dmmv tables.
pub const dmmv_wg_size_subgroup = 0;
pub const dmmv_wg_size_large = 1;

const vendor_id_nvidia = 0x10de;
const vendor_id_intel = 0x8086;

/// `ROUNDUP_POW2`.
pub fn roundupPow2(x: u64, n: u64) u64 {
    return (x + n - 1) & ~(n - 1);
}

/// Ports `ggml_vk_align_size` (ggml-vulkan.cpp:8264 @c1d0e7a00).
pub fn alignSize(width: u64, a: u64) u64 {
    return ((width + a - 1) / a) * a;
}

/// A subbuffer with an explicit size, as the C++ spells `{ buf, offset, size }`.
pub fn sub(buf: ?*Buffer, offset: u64, size: u64) Subbuffer {
    return .{ .buffer = buf, .offset = offset, .size = size };
}

/// The types with a dequantize-to-fp16 shader: the quantized types and F32.
fn hasDequant(t: c.ggml_type) bool {
    return switch (t) {
        c.GGML_TYPE_F32,
        c.GGML_TYPE_Q1_0,
        c.GGML_TYPE_Q2_0,
        c.GGML_TYPE_Q4_0,
        c.GGML_TYPE_Q4_1,
        c.GGML_TYPE_Q5_0,
        c.GGML_TYPE_Q5_1,
        c.GGML_TYPE_Q8_0,
        c.GGML_TYPE_Q2_K,
        c.GGML_TYPE_Q3_K,
        c.GGML_TYPE_Q4_K,
        c.GGML_TYPE_Q5_K,
        c.GGML_TYPE_Q6_K,
        c.GGML_TYPE_IQ1_S,
        c.GGML_TYPE_IQ1_M,
        c.GGML_TYPE_IQ2_XXS,
        c.GGML_TYPE_IQ2_XS,
        c.GGML_TYPE_IQ2_S,
        c.GGML_TYPE_IQ3_XXS,
        c.GGML_TYPE_IQ3_S,
        c.GGML_TYPE_IQ4_XS,
        c.GGML_TYPE_IQ4_NL,
        c.GGML_TYPE_MXFP4,
        c.GGML_TYPE_NVFP4,
        c.GGML_TYPE_TQ2_0,
        => true,
        else => false,
    };
}

/// The quantized types with a matrix-matrix dequant shader.
fn hasMmQuant(t: c.ggml_type) bool {
    return t != c.GGML_TYPE_F32 and hasDequant(t);
}

/// The types with a matrix-vector shader.
fn hasDmmv(t: c.ggml_type) bool {
    return t == c.GGML_TYPE_F16 or t == c.GGML_TYPE_BF16 or hasDequant(t);
}

/// The types with an integer-dot (q8_1) matrix-vector shader.
fn hasDmmvQ81(t: c.ggml_type) bool {
    return switch (t) {
        c.GGML_TYPE_Q2_0,
        c.GGML_TYPE_Q4_0,
        c.GGML_TYPE_Q4_1,
        c.GGML_TYPE_Q5_0,
        c.GGML_TYPE_Q5_1,
        c.GGML_TYPE_Q8_0,
        c.GGML_TYPE_MXFP4,
        c.GGML_TYPE_Q2_K,
        c.GGML_TYPE_Q3_K,
        c.GGML_TYPE_Q4_K,
        c.GGML_TYPE_Q5_K,
        c.GGML_TYPE_Q6_K,
        c.GGML_TYPE_IQ1_S,
        c.GGML_TYPE_IQ1_M,
        => true,
        else => false,
    };
}

/// Ports `ggml_vk_get_to_fp16` (ggml-vulkan.cpp:7646 @c1d0e7a00).
pub fn getToFp16(ctx: *BackendCtx, t: c.ggml_type) ?*Pipeline {
    if (!hasDequant(t)) return null;
    return ctx.device.pipeline_dequant[@intCast(t)];
}

/// Ports `ggml_vk_get_mul_mat_mat_pipeline` (ggml-vulkan.cpp:7682 @c1d0e7a00).
pub fn getMulMatMatPipeline(ctx: *BackendCtx, src0_type: c.ggml_type, src1_type: c.ggml_type, prec: c.ggml_prec) ?*MatmulPipeline {
    const dev = ctx.device;
    if (src0_type == c.GGML_TYPE_F32 and src1_type == c.GGML_TYPE_F32) return dev.pipeline_matmul_f32;
    if (src0_type == c.GGML_TYPE_F32 and src1_type == c.GGML_TYPE_F16) return dev.pipeline_matmul_f32_f16;
    if (src0_type == c.GGML_TYPE_BF16 and src1_type == c.GGML_TYPE_BF16) return dev.pipeline_matmul_bf16;
    if (prec == c.GGML_PREC_DEFAULT and dev.fp16 and !(dev.coopmat_support and !dev.coopmat_acc_f16_support)) {
        if (src0_type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_F32) return dev.pipeline_matmul_f16_f32.f16acc;
        if (src0_type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_F16) return dev.pipeline_matmul_f16.f16acc;
    } else {
        if (src0_type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_F32) return dev.pipeline_matmul_f16_f32.f32acc;
        if (src0_type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_F16) return dev.pipeline_matmul_f16.f32acc;
    }

    // MMQ
    if (src1_type == c.GGML_TYPE_Q8_1) {
        const pipelines = dev.pipeline_dequant_mul_mat_mat_q8_1[@intCast(src0_type)].f32acc;
        if (pipelines.isEmpty()) return null;
        return pipelines;
    }

    if (src1_type != c.GGML_TYPE_F32 and !dev.coopmat2) return null;
    if (!hasMmQuant(src0_type)) return null;

    const t: usize = @intCast(src0_type);
    if (dev.coopmat2) {
        std.debug.assert(src1_type == c.GGML_TYPE_F16);
        return if (prec == c.GGML_PREC_DEFAULT) dev.pipeline_dequant_mul_mat_mat_f16[t].f16acc else dev.pipeline_dequant_mul_mat_mat_f16[t].f32acc;
    }
    if (dev.coopmat_support) {
        return if (dev.fp16 and dev.coopmat_acc_f16_support and prec == c.GGML_PREC_DEFAULT) dev.pipeline_dequant_mul_mat_mat[t].f16acc else dev.pipeline_dequant_mul_mat_mat[t].f32acc;
    }
    return if (dev.fp16 and prec == c.GGML_PREC_DEFAULT) dev.pipeline_dequant_mul_mat_mat[t].f16acc else dev.pipeline_dequant_mul_mat_mat[t].f32acc;
}

/// The dmmv workgroup-size heuristic both vector getters share: larger
/// workgroups when M is small, on NVIDIA from Turing on and on Intel.
fn dmmvWg(ctx: *const BackendCtx, a_type: c.ggml_type, mm: u32, k: u32) usize {
    const dev = ctx.device;
    var wg: usize = dmmv_wg_size_subgroup;
    if ((dev.vendor_id == vendor_id_nvidia and dev.architecture != .nvidia_pre_turing) or dev.vendor_id == vendor_id_intel) {
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        if (a_type == c.GGML_TYPE_Q6_K) {
            if (mm < 4096 and k >= 1024) wg = dmmv_wg_size_large;
        } else {
            if (mm <= 8192 and k >= 1024) wg = dmmv_wg_size_large;
        }
    }
    return wg;
}

/// Ports `ggml_vk_get_dequantize_mul_mat_vec` (ggml-vulkan.cpp:7764 @c1d0e7a00).
pub fn getDequantizeMulMatVec(ctx: *BackendCtx, a_type: c.ggml_type, b_type: c.ggml_type, num_cols: u32, mm: u32, k: u32) ?*Pipeline {
    impl.assert(b_type == c.GGML_TYPE_F32 or b_type == c.GGML_TYPE_F16 or b_type == c.GGML_TYPE_Q8_1, "b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1");
    impl.assert(num_cols >= 1 and num_cols <= m.device.mul_mat_vec_max_cols, "num_cols >= 1 && num_cols <= mul_mat_vec_max_cols");
    if (b_type == c.GGML_TYPE_Q8_1 and !hasDmmvQ81(a_type)) return null;
    if (!hasDmmv(a_type)) return null;

    var wg = dmmvWg(ctx, a_type, mm, k);
    const dev = ctx.device;
    const t: usize = @intCast(a_type);
    if (b_type == c.GGML_TYPE_Q8_1) {
        if (dev.vendor_id == vendor_id_intel) wg = dmmv_wg_size_subgroup;
        return dev.pipeline_dequant_mul_mat_vec_q8_1_f32[wg][t][num_cols - 1];
    }
    return if (b_type == c.GGML_TYPE_F32) dev.pipeline_dequant_mul_mat_vec_f32_f32[wg][t][num_cols - 1] else dev.pipeline_dequant_mul_mat_vec_f16_f32[wg][t][num_cols - 1];
}

/// Ports `ggml_vk_get_mul_mat_mat_id_pipeline` (ggml-vulkan.cpp:7851 @c1d0e7a00).
pub fn getMulMatMatIdPipeline(ctx: *BackendCtx, src0_type: c.ggml_type, src1_type: c.ggml_type, prec: c.ggml_prec) ?*MatmulPipeline {
    const dev = ctx.device;
    if (src0_type == c.GGML_TYPE_F32 and src1_type == c.GGML_TYPE_F32) return dev.pipeline_matmul_id_f32;
    if (src0_type == c.GGML_TYPE_BF16 and src1_type == c.GGML_TYPE_BF16) return dev.pipeline_matmul_id_bf16;
    if (prec == c.GGML_PREC_DEFAULT and dev.fp16 and !(dev.coopmat_support and !dev.coopmat_acc_f16_support)) {
        if (src0_type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_F32) return dev.pipeline_matmul_id_f16_f32.f16acc;
        if (src0_type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_F16) return dev.pipeline_matmul_id_f16.f16acc;
    } else {
        if (src0_type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_F32) return dev.pipeline_matmul_id_f16_f32.f32acc;
        if (src0_type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_F16) return dev.pipeline_matmul_id_f16.f32acc;
    }

    // MMQ
    if (src1_type == c.GGML_TYPE_Q8_1) {
        const pipelines = dev.pipeline_dequant_mul_mat_mat_id_q8_1[@intCast(src0_type)].f32acc;
        if (pipelines.isEmpty()) return null;
        return pipelines;
    }

    impl.assert(src1_type == c.GGML_TYPE_F32 or (dev.coopmat2 and src1_type == c.GGML_TYPE_F16), "src1_type == GGML_TYPE_F32 || (ctx->device->coopmat2 && src1_type == GGML_TYPE_F16)");
    if (!hasMmQuant(src0_type)) return null;

    const mmp = &dev.pipeline_dequant_mul_mat_mat_id[@intCast(src0_type)];
    // XXX TODO 'prec' is not actually allowed in mul_mat_id.
    const prefer_fp16acc = dev.fp16;
    const support_fp16acc = !mmp.f16acc.isEmpty();
    const support_fp32acc = !mmp.f32acc.isEmpty();
    if (support_fp16acc and (prefer_fp16acc or !support_fp32acc)) return mmp.f16acc;
    impl.assert(support_fp32acc, "support_fp32acc");
    return mmp.f32acc;
}

/// Ports `ggml_vk_get_dequantize_mul_mat_vec_id` (ggml-vulkan.cpp:7932 @c1d0e7a00).
pub fn getDequantizeMulMatVecId(ctx: *BackendCtx, a_type: c.ggml_type, b_type: c.ggml_type, mm: u32, k: u32) ?*Pipeline {
    impl.assert(b_type == c.GGML_TYPE_F32 or b_type == c.GGML_TYPE_Q8_1, "b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1");
    if (b_type == c.GGML_TYPE_Q8_1 and !hasDmmvQ81(a_type)) return null;
    if (!hasDmmv(a_type)) return null;

    var wg = dmmvWg(ctx, a_type, mm, k);
    const dev = ctx.device;
    const t: usize = @intCast(a_type);
    if (b_type == c.GGML_TYPE_Q8_1) {
        if (dev.vendor_id == vendor_id_intel) wg = dmmv_wg_size_subgroup;
        return dev.pipeline_dequant_mul_mat_vec_id_q8_1_f32[wg][t];
    }
    return dev.pipeline_dequant_mul_mat_vec_id_f32[wg][t];
}

/// Ports `ggml_vk_guess_split_k` (ggml-vulkan.cpp:8723 @c1d0e7a00).
pub fn guessSplitK(ctx: *const BackendCtx, mm: u32, n: u32, k: u32, disable_split_k: bool, pipeline: *const Pipeline) u32 {
    if (disable_split_k) return 1;

    const cores = ctx.device.shader_core_count;
    var split_k: u32 = 1;
    if (cores != 0 and mm >= pipeline.wg_denoms[0] and n >= pipeline.wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        const m_tiles = ceilDiv(mm, pipeline.wg_denoms[0]);
        const n_tiles = ceilDiv(n, pipeline.wg_denoms[1]);
        if (k >= 2048) {
            if (m_tiles *% n_tiles <= cores / 2) {
                split_k = cores / (m_tiles *% n_tiles);
            } else if (m_tiles *% n_tiles <= cores * 2 / 3) {
                split_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            split_k = @min(split_k, 8);

            // ggml_vk_matmul aligns the splits to a multiple of 256; reduce the
            // count while the rounded-up size would leave the last split empty.
            while (split_k != 1) {
                var k_split = ceilDiv(k, split_k);
                k_split = @intCast(roundupPow2(k_split, 256));
                if (k_split *% (split_k - 1) < k) break;
                split_k -= 1;
            }
        }
    }
    return split_k;
}

fn pick(aligned: bool, a: ?*Pipeline, u: ?*Pipeline) *Pipeline {
    return (if (aligned) a else u).?;
}

/// Ports `ggml_vk_guess_matmul_pipeline` (ggml-vulkan.cpp:8765 @c1d0e7a00).
pub fn guessMatmulPipeline(ctx: *const BackendCtx, mmp: *const MatmulPipeline, mm: u32, n: u32, aligned: bool, src0_type: c.ggml_type, src1_type: c.ggml_type) *Pipeline {
    const dev = ctx.device;
    const t: usize = @intCast(src0_type);
    // The q8_1 (integer dot) mmq path has its own shared-memory layout.
    const is_q8_1 = src1_type == c.GGML_TYPE_Q8_1;
    const mm_l = if (is_q8_1) dev.mul_mat_l_int[t] else dev.mul_mat_l[t];
    const mm_m = if (is_q8_1) dev.mul_mat_m_int[t] else dev.mul_mat_m[t];
    const mm_s = if (is_q8_1) dev.mul_mat_s_int[t] else dev.mul_mat_s[t];

    if (dev.coopmat2) {
        const cores = dev.shader_core_count;
        const tiles_l = ceilDiv(mm, mmp.a_l.?.wg_denoms[0]) *% ceilDiv(n, mmp.a_l.?.wg_denoms[1]);
        const tiles_m = ceilDiv(mm, mmp.a_m.?.wg_denoms[0]) *% ceilDiv(n, mmp.a_m.?.wg_denoms[1]);

        // Use large shader when the N dimension is greater than the medium shader's tile size
        const crossover_large = mmp.m.?.wg_denoms[1];
        // Prefer large when medium or large tiles would overfill the GPU, or
        // large with split_k 3 fits and medium with split_k 2 does not.
        const prefer_large = tiles_m > cores or tiles_l > cores or
            (tiles_l <= cores / 3 and tiles_m > cores / 2);

        if ((mm_l and (n > crossover_large and prefer_large)) or (!mm_m and !mm_s)) return pick(aligned, mmp.a_l, mmp.l);
        // Use medium shader when the N dimension is greater than the small shader's tile size
        const crossover_medium = mmp.s.?.wg_denoms[1];
        if ((mm_m and (n > crossover_medium)) or !mm_s) return pick(aligned, mmp.a_m, mmp.m);
        return pick(aligned, mmp.a_s, mmp.s);
    }

    if ((mm_s and (mm <= 32 or n <= 32)) or (!mm_m and !mm_l)) return pick(aligned, mmp.a_s, mmp.s);
    if ((mm_m and (mm <= 64 or n <= 64)) or !mm_l) return pick(aligned, mmp.a_m, mmp.m);
    return pick(aligned, mmp.a_l, mmp.l);
}

/// Ports `ggml_vk_guess_matmul_pipeline_align` (ggml-vulkan.cpp:8811 @c1d0e7a00).
pub fn guessMatmulPipelineAlign(ctx: *const BackendCtx, mmp: *const MatmulPipeline, mm: u32, n: u32, src0_type: c.ggml_type, src1_type: c.ggml_type) u32 {
    return guessMatmulPipeline(ctx, mmp, mm, n, true, src0_type, src1_type).@"align";
}

/// Ports `ggml_vk_matmul` (ggml-vulkan.cpp:8816 @c1d0e7a00): batches split
/// over the z workgroup limit, and with split_k > 1 a pass into scratch and
/// a reduction.
pub fn matmul(
    ctx: *BackendCtx,
    subctx: *Context,
    pipeline: *Pipeline,
    a: Subbuffer,
    b: Subbuffer,
    d: Subbuffer,
    split_k_buffer: Subbuffer,
    mm: u32,
    n: u32,
    k: u32,
    stride_a: u32,
    stride_b: u32,
    stride_d: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    split_k: u32,
    batch: u32,
    ne02: u32,
    ne12: u32,
    broadcast2: u32,
    broadcast3: u32,
    padded_n: u32,
) void {
    const max_z = ctx.device.properties.limits.maxComputeWorkGroupCount[2];
    var pc: push.MatMat = .{
        .M = mm,
        .N = n,
        .K = k,
        .stride_a = stride_a,
        .stride_b = stride_b,
        .stride_d = stride_d,
        .batch_stride_a = batch_stride_a,
        .batch_stride_b = batch_stride_b,
        .batch_stride_d = batch_stride_d,
        .base_work_group_z = 0,
        .num_batches = batch,
        .k_split = k,
        .ne02 = ne02,
        .ne12 = ne12,
        .broadcast2 = broadcast2,
        .broadcast3 = broadcast3,
        .padded_N = padded_n,
    };
    if (split_k == 1) {
        m.context.requestDescriptorSets(ctx, pipeline, ceilDiv(batch, max_z));
        var base: u32 = 0;
        while (base < batch) {
            const groups_z = @min(batch - base, max_z);
            pc.base_work_group_z = base;
            m.context.dispatch(ctx, subctx, pipeline, &.{ a, b, d }, &pc, .{ mm, n, groups_z });
            base += groups_z;
        }
        return;
    }

    if (ctx.prealloc_split_k_need_sync) m.context.syncBuffers(ctx, subctx);

    impl.assert(batch_stride_d == mm *% n, "batch_stride_d == m * n");

    // Round the split size up to a multiple of 256 (k-quant alignment)
    const k_split: u32 = @intCast(roundupPow2(ceilDiv(k, split_k), 256));
    m.context.requestDescriptorSets(ctx, pipeline, ceilDiv(batch, max_z));

    var base: u32 = 0;
    while (base < batch) {
        const groups_z = @min(batch - base, max_z);
        pc.base_work_group_z = base;
        pc.k_split = k_split;
        // Make sure enough workgroups get assigned for split k to work
        m.context.dispatch(ctx, subctx, pipeline, &.{ a, b, split_k_buffer }, &pc, .{ (ceilDiv(mm, pipeline.wg_denoms[0]) *% pipeline.wg_denoms[0]) *% split_k, n, groups_z });
        base += groups_z;
    }
    m.context.syncBuffers(ctx, subctx);
    const pc2 = [2]u32{ mm *% n *% batch, split_k };
    m.context.dispatch(ctx, subctx, ctx.device.pipeline_matmul_split_k_reduce.?, &.{ split_k_buffer, d }, &pc2, .{ mm *% n *% batch, 1, 1 });
    ctx.prealloc_split_k_need_sync = true;
}

/// Ports `ggml_vk_guess_matmul_id_pipeline` (ggml-vulkan.cpp:8865 @c1d0e7a00).
pub fn guessMatmulIdPipeline(ctx: *const BackendCtx, mmp: *const MatmulPipeline, mm: u32, n: u32, aligned: bool, src0_type: c.ggml_type, src1_type: c.ggml_type) *Pipeline {
    const dev = ctx.device;
    const t: usize = @intCast(src0_type);
    const is_q8_1 = src1_type == c.GGML_TYPE_Q8_1;
    const mm_l = if (is_q8_1) dev.mul_mat_id_l_int[t] else dev.mul_mat_id_l[t];
    const mm_m = if (is_q8_1) dev.mul_mat_id_m_int[t] else dev.mul_mat_id_m[t];
    const mm_s = if (is_q8_1) dev.mul_mat_id_s_int[t] else dev.mul_mat_id_s[t];

    if (dev.coopmat2) {
        // Use large shader when the N dimension is greater than the medium shader's tile size
        const crossover_large = mmp.m.?.wg_denoms[1];
        if ((mm_l and (n > crossover_large)) or (!mm_m and !mm_s)) return pick(aligned, mmp.a_l, mmp.l);
        // Use medium shader when the N dimension is greater than the small shader's tile size
        const crossover_medium = mmp.s.?.wg_denoms[1];
        if ((mm_m and (n > crossover_medium)) or !mm_s) return pick(aligned, mmp.a_m, mmp.m);
        return pick(aligned, mmp.a_s, mmp.s);
    }

    if ((mm_s and (mm <= 32 or n <= 32)) or (!mm_m and !mm_l)) return pick(aligned, mmp.a_s, mmp.s);
    if ((mm_m and (mm <= 64 or n <= 64)) or !mm_l) return pick(aligned, mmp.a_m, mmp.m);
    return pick(aligned, mmp.a_l, mmp.l);
}

/// Ports `ggml_vk_guess_matmul_id_pipeline_align` (ggml-vulkan.cpp:8898 @c1d0e7a00).
pub fn guessMatmulIdPipelineAlign(ctx: *const BackendCtx, mmp: *const MatmulPipeline, mm: u32, n: u32, src0_type: c.ggml_type, src1_type: c.ggml_type) u32 {
    return guessMatmulIdPipeline(ctx, mmp, mm, n, true, src0_type, src1_type).@"align";
}

/// Ports `ggml_vk_matmul_id` (ggml-vulkan.cpp:8903 @c1d0e7a00).
pub fn matmulId(
    ctx: *BackendCtx,
    subctx: *Context,
    pipeline: *Pipeline,
    a: Subbuffer,
    b: Subbuffer,
    d: Subbuffer,
    ids: Subbuffer,
    expert_count_buf: Subbuffer,
    mm: u32,
    n: u32,
    k: u32,
    stride_a: u32,
    stride_b: u32,
    stride_d: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    n_as: u32,
    nei0: u32,
    nei1: u32,
    nbi1: u32,
    ne11: u32,
    padded_n: u32,
) void {
    const pc: push.MatMatId = .{
        .M = mm,
        .N = n,
        .K = k,
        .stride_a = stride_a,
        .stride_b = stride_b,
        .stride_d = stride_d,
        .batch_stride_a = batch_stride_a,
        .batch_stride_b = batch_stride_b,
        .batch_stride_d = batch_stride_d,
        .nei0 = nei0,
        .nei1 = nei1,
        .nbi1 = nbi1,
        .ne11 = ne11,
        .padded_N = padded_n,
    };
    m.context.dispatch(ctx, subctx, pipeline, &.{ a, b, d, ids, expert_count_buf }, &pc, .{ mm, nei1, n_as });
}

/// The dispatch grid both copy helpers use: up to 512 x 512 x n.
fn cpyElements(ne: u32) [3]u32 {
    if (ne > 262144) return .{ 512, 512, ceilDiv(ne, 262144) };
    if (ne > 512) return .{ 512, ceilDiv(ne, 512), 1 };
    return .{ ne, 1, 1 };
}

/// Ports `ggml_vk_cpy_to_contiguous` (ggml-vulkan.cpp:9067 @c1d0e7a00).
pub fn cpyToContiguous(ctx: *BackendCtx, subctx: *Context, pipeline: *Pipeline, tensor: *const Tensor, in: Subbuffer, out: Subbuffer) void {
    const ne = ops.u32Of(c.ggml_nelements(tensor));
    var pc = ops.unaryPushConstantsInit(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = ops.u32Of(tensor.ne[0]);
    pc.nb12 = ops.u32Of(tensor.ne[0] *% tensor.ne[1]);
    pc.nb13 = ops.u32Of(tensor.ne[0] *% tensor.ne[1] *% tensor.ne[2]);
    ops.initPushconstFastdiv(&pc);
    m.context.dispatch(ctx, subctx, pipeline, &.{ in, out }, &pc, cpyElements(ne));
    m.context.syncBuffers(ctx, subctx);
}

/// Ports `ggml_vk_cpy_to_strided` (ggml-vulkan.cpp:9094 @c1d0e7a00): copy
/// into a caller-defined dense layout, strides in output elements.
pub fn cpyToStrided(ctx: *BackendCtx, subctx: *Context, pipeline: *Pipeline, tensor: *const Tensor, in: Subbuffer, out: Subbuffer, nb10: u32, nb11: u32, nb12: u32, nb13: u32) void {
    const ne = ops.u32Of(c.ggml_nelements(tensor));
    var pc = ops.unaryPushConstantsInit(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    ops.initPushconstFastdiv(&pc);
    m.context.dispatch(ctx, subctx, pipeline, &.{ in, out }, &pc, cpyElements(ne));
    m.context.syncBuffers(ctx, subctx);
}

/// Ports `ggml_vk_get_quantize_pipeline` (ggml-vulkan.cpp:9122 @c1d0e7a00).
pub fn getQuantizePipeline(ctx: *BackendCtx, t: c.ggml_type) *Pipeline {
    if (t == c.GGML_TYPE_Q8_1) return ctx.device.pipeline_quantize_q8_1_x4.?;
    impl.fprintfStderr("Missing quantize pipeline for type: %s\n", .{c.ggml_type_name(t)});
    impl.abort("fatal error");
}

/// Ports `ggml_vk_quantize_q8_1` (ggml-vulkan.cpp:9132 @c1d0e7a00).
pub fn quantizeQ81(ctx: *BackendCtx, subctx: *Context, in: Subbuffer, out: Subbuffer, ne: u32) void {
    const pipeline = getQuantizePipeline(ctx, c.GGML_TYPE_Q8_1);
    const num_blocks = ceilDiv(ne, pipeline.wg_denoms[0]);
    // Clamp to the max workgroup count; the shader iterates over the blocks.
    const max_elements = @min(@as(u64, ctx.device.properties.limits.maxComputeWorkGroupCount[0]) * pipeline.wg_denoms[0], std.math.maxInt(u32));
    const elements = @min(ne, @as(u32, @intCast(max_elements)));
    const pc: push.QuantizeQ81 = .{ .ne = ne, .num_blocks = num_blocks };
    m.context.dispatch(ctx, subctx, pipeline, &.{ in, out }, &pc, .{ elements, 1, 1 });
    m.context.syncBuffers(ctx, subctx);
}

/// Ports `ggml_vk_get_64b_indexing_pipeline` (ggml-vulkan.cpp:9151 @c1d0e7a00):
/// the 64-bit-indexing variant along the pipeline's `next` chain, when the
/// headers have the extension.
pub fn get64bIndexingPipeline(pipeline: *Pipeline) *Pipeline {
    if (comptime @hasDecl(vk, "VK_EXT_shader_64bit_indexing")) {
        var p: ?*Pipeline = pipeline;
        while (p) |q| : (p = q.next) {
            if (q.is_64b_indexing) return q;
        }
    }
    return pipeline;
}

/// A source tensor's binding: through pinned memory on a UMA device when
/// it lives there, else its own buffer at its offset.
pub const Source = struct {
    buf: ?*Buffer = null,
    offset: u64 = 0,
    uma: bool = false,

    pub fn pinned(ctx: *const BackendCtx, t: *const Tensor) Source {
        var s: Source = .{};
        if (ctx.device.uma) {
            var off: usize = 0;
            s.buf = m.buffer.hostGet(ctx.device, t.data, &off);
            s.offset = off;
            s.uma = s.buf != null;
        }
        return s;
    }

    /// Falls back to the tensor's own buffer when it was not pinned.
    pub fn resolve(s: *Source, t: *const Tensor) void {
        if (!s.uma) {
            s.buf = m.buffer.tensorBuffer(t);
            s.offset = m.buffer.tensorOffset(t) + t.view_offs;
        }
    }
};

fn typeSizeOf(t: c.ggml_type) u64 {
    return c.ggml_type_size(t);
}

/// Ports `ggml_vk_mul_mat_q_f16` (ggml-vulkan.cpp:9165 @c1d0e7a00): the
/// matrix-matrix path. Operands the chosen shader cannot read directly are
/// converted into `prealloc_x`/`prealloc_y` first -- dequantized to fp16,
/// made contiguous, or quantized to q8_1 for the integer-dot shaders.
pub fn mulMatQF16(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor, disable_split_k: bool) void {
    const dim01 = m.op_pipeline.dim01Contiguous;
    impl.assert(dim01(src0) or src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_BF16, "ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16");
    impl.assert(dim01(src1) or src1.type == c.GGML_TYPE_F32 or src1.type == c.GGML_TYPE_F16, "ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16");

    const dev = ctx.device;
    const ne00: u64 = @intCast(src0.ne[0]);
    const ne01: u64 = @intCast(src0.ne[1]);
    const ne02: u64 = @intCast(src0.ne[2]);
    const ne03: u64 = @intCast(src0.ne[3]);
    const ne10: u64 = @intCast(src1.ne[0]);
    const ne11: u64 = @intCast(src1.ne[1]);
    const ne12: u64 = @intCast(src1.ne[2]);
    const ne13: u64 = @intCast(src1.ne[3]);
    const ne21: u64 = @intCast(dst.ne[1]);
    const stride_d: u32 = @truncate(dst.nb[1] / typeSizeOf(dst.type));
    const stride_batch_d: u32 = @truncate(stride_d *% ne21);
    const r2 = ne12 / ne02;
    const r3 = ne13 / ne03;
    const prec: c.ggml_prec = @intCast(dst.op_params[0]);

    var qx = Source.pinned(ctx, src0);
    var qy = Source.pinned(ctx, src1);

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    const x_non_contig = (dev.coopmat2 and src0.type == c.GGML_TYPE_F32) or !dim01(src0);
    const y_non_contig = (dev.coopmat2 and src1.type == c.GGML_TYPE_F32) or
        (src0.type == c.GGML_TYPE_BF16 and src1.type != c.GGML_TYPE_BF16) or
        !dim01(src1);

    // If src0 is BF16, try to use a BF16 x BF16 multiply
    const f16_type: c.ggml_type = if (src0.type == c.GGML_TYPE_BF16) c.GGML_TYPE_BF16 else c.GGML_TYPE_F16;
    const y_f32_kernel = src1.type == c.GGML_TYPE_F32 and !y_non_contig;

    var quantize_y = dev.integer_dot_product and src1.type == c.GGML_TYPE_F32 and c.ggml_is_contiguous(src1) and !y_non_contig and (ne11 *% ne10) % 4 == 0;

    // Check for mmq first
    var mmp: ?*MatmulPipeline = if (quantize_y) getMulMatMatPipeline(ctx, src0.type, c.GGML_TYPE_Q8_1, prec) else null;
    if (mmp == null) {
        // Fall back to f16 dequant mul mat
        mmp = getMulMatMatPipeline(ctx, src0.type, if (y_non_contig) f16_type else src1.type, prec);
        quantize_y = false;
    }

    const qx_needs_dequant = mmp == null or x_non_contig;
    const qy_needs_dequant = !quantize_y and ((src1.type != f16_type and !y_f32_kernel) or y_non_contig);

    if (qx_needs_dequant) {
        // Fall back to dequant + f16 mulmat
        mmp = getMulMatMatPipeline(ctx, f16_type, if (y_f32_kernel) c.GGML_TYPE_F32 else f16_type, prec);
    }
    const mp = mmp.?;

    // Not implemented
    impl.assert(y_non_contig or !qy_needs_dequant, "y_non_contig || !qy_needs_dequant");

    const effective_src1_type: c.ggml_type = if (quantize_y) c.GGML_TYPE_Q8_1 else if (y_f32_kernel) c.GGML_TYPE_F32 else src1.type;
    const x_type = if (qx_needs_dequant) f16_type else src0.type;

    const kpad: u32 = if (quantize_y) 0 else @truncate(alignSize(ne10, guessMatmulPipelineAlign(ctx, mp, @truncate(ne01), @truncate(ne11), x_type, effective_src1_type)));
    const aligned = !quantize_y and ne10 == kpad and ne01 > 8 and ne11 > 8;

    var pipeline = guessMatmulPipeline(ctx, mp, @truncate(ne01), @truncate(ne11), aligned, x_type, effective_src1_type);

    if (c.ggml_nbytes(src0) > dev.properties.limits.maxStorageBufferRange) pipeline = get64bIndexingPipeline(pipeline);

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    const padded_n: u32 = if (qy_needs_dequant) @truncate(roundupPow2(@as(u32, @truncate(ne11)), pipeline.wg_denoms[1])) else @truncate(ne11);
    const x_ne: u64 = @intCast(c.ggml_nelements(src0));
    // 128 elements per Q8_1 x4 block
    const y_ne: u64 = @as(u64, padded_n) *% ne10 *% ne12 *% ne13;
    const d_ne: u64 = @intCast(c.ggml_nelements(dst));

    const split_k = guessSplitK(ctx, @truncate(ne01), @truncate(ne11), @truncate(ne10), disable_split_k, pipeline);

    const qx_sz = typeSizeOf(src0.type) * x_ne / @as(u64, @intCast(c.ggml_blck_size(src0.type)));
    const qy_sz = typeSizeOf(src1.type) * y_ne / @as(u64, @intCast(c.ggml_blck_size(src1.type)));
    const x_sz = if (!qx_needs_dequant) qx_sz else @sizeOf(c.ggml_fp16_t) * x_ne;
    const y_sz: u64 = if (quantize_y)
        alignSize(y_ne, 128) * typeSizeOf(c.GGML_TYPE_Q8_1) / @as(u64, @intCast(c.ggml_blck_size(c.GGML_TYPE_Q8_1)))
    else if (y_f32_kernel) @sizeOf(f32) * y_ne else @sizeOf(c.ggml_fp16_t) * y_ne;
    const d_sz = @sizeOf(f32) * d_ne;

    const to_fp16_vk_0 = if (x_non_contig) m.op_pipeline.getCpyPipeline(ctx, src0, null, f16_type) else getToFp16(ctx, src0.type);
    const to_fp16_vk_1 = if (y_non_contig) m.op_pipeline.getCpyPipeline(ctx, src1, null, f16_type) else getToFp16(ctx, src1.type);
    impl.assert(!qx_needs_dequant or to_fp16_vk_0 != null, "!qx_needs_dequant || to_fp16_vk_0 != nullptr");
    impl.assert(!qy_needs_dequant or to_fp16_vk_1 != null, "!qy_needs_dequant || to_fp16_vk_1 != nullptr");
    const to_q8_1: ?*Pipeline = if (quantize_y) getQuantizePipeline(ctx, c.GGML_TYPE_Q8_1) else null;

    {
        const max_range = dev.properties.limits.maxStorageBufferRange;
        const split_k_size: u64 = if (split_k > 1) d_sz * split_k else 0;
        if ((qx_needs_dequant and x_sz > max_range) or
            (qy_needs_dequant and y_sz > max_range) or
            (split_k > 1 and split_k_size > max_range))
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
        if (split_k > 1 and ctx.prealloc_size_split_k < split_k_size) {
            ctx.prealloc_size_split_k = split_k_size;
            m.nodes.preallocateBuffers(ctx, subctx);
        }

        // Request descriptor sets
        if (qx_needs_dequant) m.context.requestDescriptorSets(ctx, to_fp16_vk_0.?, 1);
        if (qy_needs_dequant) m.context.requestDescriptorSets(ctx, to_fp16_vk_1.?, 1);
        if (quantize_y) m.context.requestDescriptorSets(ctx, to_q8_1.?, 1);
        if (split_k > 1) m.context.requestDescriptorSets(ctx, dev.pipeline_matmul_split_k_reduce.?, 1);
    }

    const d_D = m.buffer.tensorBuffer(dst);
    const d_buf_offset = m.buffer.tensorOffset(dst) + dst.view_offs;
    impl.assert(d_D.size >= d_buf_offset + d_sz, "d_D->size >= d_buf_offset + d_sz");
    qx.resolve(src0);
    qy.resolve(src1);

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

    if (x_non_contig) {
        cpyToContiguous(ctx, subctx, to_fp16_vk_0.?, src0, m.context.subbuffer(ctx, qx.buf.?, qx.offset), m.context.subbuffer(ctx, d_X, 0));
    } else if (qx_needs_dequant) {
        const pc = [5]u32{ @truncate(ne01), @truncate(ne10), @truncate(ne10), @truncate(ne10), ops.u32Of(c.ggml_nelements(src0)) };
        m.context.dispatch(ctx, subctx, to_fp16_vk_0.?, &.{ sub(qx.buf, qx.offset, qx_sz), sub(d_X, 0, x_sz) }, &pc, .{ @truncate(x_ne), 1, 1 });
        m.context.syncBuffers(ctx, subctx);
    }
    if (y_non_contig) {
        if (ctx.prealloc_y_last_pipeline_used != to_fp16_vk_1 or
            ctx.prealloc_y_last_tensor_used != src1 or
            ctx.prealloc_y_last_decode_vector_staging)
        {
            if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
            cpyToContiguous(ctx, subctx, to_fp16_vk_1.?, src1, m.context.subbuffer(ctx, qy.buf.?, qy.offset), m.context.subbuffer(ctx, d_Y, 0));
            ctx.prealloc_y_last_pipeline_used = to_fp16_vk_1;
            ctx.prealloc_y_last_tensor_used = src1;
            ctx.prealloc_y_last_decode_vector_staging = false;
        }
    }
    if (quantize_y) {
        if (ctx.prealloc_y_last_pipeline_used != to_q8_1 or
            ctx.prealloc_y_last_tensor_used != src1 or
            ctx.prealloc_y_last_decode_vector_staging)
        {
            if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
            quantizeQ81(ctx, subctx, m.context.subbuffer(ctx, qy.buf.?, qy.offset), m.context.subbuffer(ctx, d_Y, 0), @truncate(y_ne));
            ctx.prealloc_y_last_pipeline_used = to_q8_1;
            ctx.prealloc_y_last_tensor_used = src1;
            ctx.prealloc_y_last_decode_vector_staging = false;
        }
    }

    var stride_batch_x: u32 = @truncate(ne00 *% ne01);
    var stride_batch_y: u32 = @truncate(ne10 *% ne11);
    if (!dim01(src0) and !qx_needs_dequant) stride_batch_x = @truncate(src0.nb[0] / typeSizeOf(src0.type));
    if (!dim01(src1) and !qy_needs_dequant and !quantize_y) stride_batch_y = @truncate(src1.nb[0] / typeSizeOf(src1.type));

    // compute
    matmul(
        ctx,
        subctx,
        pipeline,
        sub(d_X, x_buf_offset, x_sz),
        sub(d_Y, y_buf_offset, y_sz),
        m.context.subbuffer(ctx, d_D, d_buf_offset),
        sub(ctx.prealloc_split_k, 0, d_sz * split_k),
        @truncate(ne01),
        @truncate(ne11),
        @truncate(ne10),
        @truncate(ne10),
        @truncate(ne10),
        stride_d,
        stride_batch_x,
        stride_batch_y,
        stride_batch_d,
        split_k,
        @truncate(ne12 *% ne13),
        @truncate(ne02),
        @truncate(ne12),
        @truncate(r2),
        @truncate(r3),
        padded_n,
    );

    if (x_non_contig or qx_needs_dequant) ctx.prealloc_x_need_sync = true;
    if (y_non_contig or quantize_y) ctx.prealloc_y_need_sync = true;
}

const vendor_id_amd = 0x1002;
const mat_vec_fusion_flags_bias0: u32 = 0x1;
const mat_vec_fusion_flags_bias1: u32 = 0x2;

fn at(cg: *const impl.CGraph, i: c_int) *Tensor {
    return cg.nodes[@intCast(i)].?;
}

/// Ports `ggml_vk_should_use_mmvq` (ggml-vulkan.cpp:9434 @c1d0e7a00): whether
/// quantizing B to q8_1 for the integer-dot matrix-vector shader pays.
pub fn shouldUseMmvq(dev: *const m.device.Device, mm: u32, n: u32, k: u32, src0_type: c.ggml_type) bool {
    _ = mm;
    if (dev.mmvq_mode == 1) return true;
    if (dev.mmvq_mode == -1) return false;

    // q6_k only has 2-byte alignment; MMVQ is only a win on Intel.
    const mmvq_q6 = dev.vendor_id == vendor_id_intel;
    if (src0_type == c.GGML_TYPE_Q6_K and !mmvq_q6) return false;

    // MMVQ is generally good for batches
    if (n > 1) return true;

    // Quantization overhead is not worth it for small k
    switch (dev.vendor_id) {
        vendor_id_nvidia => {
            if (src0_type == c.GGML_TYPE_Q2_0 or src0_type == c.GGML_TYPE_Q2_K or src0_type == c.GGML_TYPE_Q3_K or src0_type == c.GGML_TYPE_IQ1_S or src0_type == c.GGML_TYPE_IQ1_M) return true;
            if (k <= 4096) return false;
            return switch (src0_type) {
                c.GGML_TYPE_MXFP4, c.GGML_TYPE_Q8_0 => dev.architecture == .nvidia_pre_turing,
                else => true,
            };
        },
        vendor_id_amd => {
            if (k < 2048) return false;
            return switch (src0_type) {
                c.GGML_TYPE_Q8_0 => dev.architecture == .amd_gcn,
                else => true,
            };
        },
        vendor_id_intel => {
            if (dev.architecture == .intel_xe2) {
                if (src0_type == c.GGML_TYPE_Q2_0 or src0_type == c.GGML_TYPE_Q2_K or src0_type == c.GGML_TYPE_Q3_K or src0_type == c.GGML_TYPE_Q6_K) return true;
            }
            // Intel's Windows proprietary driver: MMVQ for !Q2/Q3/Q6 is worse than fp16.
            if (dev.driver_id == vk.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS) return false;
            if (k < 2048) return false;
            return switch (src0_type) {
                // From tests on A770 Linux, may need more tuning
                c.GGML_TYPE_Q4_0, c.GGML_TYPE_Q5_1 => false,
                else => true,
            };
        },
        else => return true,
    }
}

/// The fused bias bindings the matrix-vector shaders take: one or two ADDs
/// after the MUL_MAT, each set only when that ADD was fused.
const Biases = struct {
    f0: Subbuffer,
    f1: Subbuffer,
    flags: u32 = 0,
};

/// The `d_F0`/`d_F1` selection shared by the three matrix-vector paths.
/// `second_any_operand` is the q_f16 form, which accepts the bias on
/// either side of the second ADD; p021 and nc take its `src[1]`.
fn fusedBiases(ctx: *const BackendCtx, cgraph: *const impl.CGraph, node_idx: c_int, d_D: Subbuffer, second_any_operand: bool) Biases {
    const dst = at(cgraph, node_idx);
    var b: Biases = .{ .f0 = d_D, .f1 = d_D };
    if (ctx.num_additional_fused_ops > 0) {
        const add = at(cgraph, node_idx + 1);
        const bias: *const Tensor = if (add.src[0] == dst) add.src[1] else add.src[0];
        b.f0 = m.context.tensorSubbuffer(ctx, bias, false);
        b.flags |= mat_vec_fusion_flags_bias0;
    }
    if (if (second_any_operand) ctx.num_additional_fused_ops == 2 else ctx.num_additional_fused_ops > 1) {
        const add = at(cgraph, node_idx + 2);
        const bias: *const Tensor = if (!second_any_operand) add.src[1] else if (add.src[0] == at(cgraph, node_idx + 1)) add.src[1] else add.src[0];
        b.f1 = m.context.tensorSubbuffer(ctx, bias, false);
        b.flags |= mat_vec_fusion_flags_bias1;
    }
    return b;
}

/// Ports `ggml_vk_mul_mat_vec_q_f16` (ggml-vulkan.cpp:9515 @c1d0e7a00): the
/// matrix-vector path, for one column or up to `mul_mat_vec_max_cols`,
/// with B optionally quantized to q8_1 and one or two biases fused.
pub fn mulMatVecQF16(ctx: *BackendCtx, subctx: *Context, cgraph: *const impl.CGraph, node_idx: c_int) void {
    const dim01 = m.op_pipeline.dim01Contiguous;
    const dst = at(cgraph, node_idx);
    const src0: *const Tensor = dst.src[0];
    const src1: *const Tensor = dst.src[1];
    impl.assert(dim01(src0) or src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_BF16, "ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16");
    impl.assert(dim01(src1) or src1.type == c.GGML_TYPE_F32 or src1.type == c.GGML_TYPE_F16, "ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16");

    const dev = ctx.device;
    const limits = &dev.properties.limits;
    const ne00: u64 = @intCast(src0.ne[0]);
    const ne01: u64 = @intCast(src0.ne[1]);
    const ne02: u64 = @intCast(src0.ne[2]);
    const ne03: u64 = @intCast(src0.ne[3]);
    const ne10: u64 = @intCast(src1.ne[0]);
    const ne11: u64 = @intCast(src1.ne[1]);
    const ne12: u64 = @intCast(src1.ne[2]);
    const ne13: u64 = @intCast(src1.ne[3]);
    const ne20: u64 = @intCast(dst.ne[0]);
    const ne21: u64 = @intCast(dst.ne[1]);
    const r2 = ne12 / ne02;
    const r3 = ne13 / ne03;

    // batch_n computes a few vector results and assumes ne12 and ne13 are
    // 1; it overloads the batch strides to hold the row strides.
    impl.assert(ne11 == 1 or ne12 *% ne13 == 1, "ne11 == 1 || ne12 * ne13 == 1");
    const batch_n = ne11 > 1;

    const x_non_contig = !dim01(src0);
    const y_non_contig = !dim01(src1);

    const f16_f32_kernel = src1.type == c.GGML_TYPE_F32;
    var quantize_y = dev.integer_dot_product and src1.type == c.GGML_TYPE_F32 and c.ggml_is_contiguous(src1) and !y_non_contig and (ne11 *% ne10) % 4 == 0 and
        shouldUseMmvq(dev, @truncate(ne01), @truncate(ne11), @truncate(ne10), src0.type);

    const to_fp16_vk_0: ?*Pipeline = if (x_non_contig) m.op_pipeline.getCpyPipeline(ctx, src0, null, src0.type) else null;
    const to_fp16_vk_1 = if (y_non_contig) m.op_pipeline.getCpyPipeline(ctx, src1, null, src1.type) else getToFp16(ctx, src1.type);

    // Check for mmq first
    var dmmv: ?*Pipeline = if (quantize_y) getDequantizeMulMatVec(ctx, src0.type, c.GGML_TYPE_Q8_1, @truncate(ne11), @truncate(ne20), @truncate(ne00)) else null;
    if (dmmv == null) {
        // Fall back to f16 dequant mul mat
        dmmv = getDequantizeMulMatVec(ctx, src0.type, src1.type, @truncate(ne11), @truncate(ne20), @truncate(ne00));
        quantize_y = false;
    }
    const to_q8_1: ?*Pipeline = if (quantize_y) getQuantizePipeline(ctx, c.GGML_TYPE_Q8_1) else null;

    if (c.ggml_nbytes(src0) > limits.maxStorageBufferRange) dmmv = get64bIndexingPipeline(dmmv.?);

    const qx_needs_dequant = x_non_contig;
    const qy_needs_dequant = !quantize_y and ((src1.type != c.GGML_TYPE_F16 and !f16_f32_kernel) or y_non_contig);

    // Not implemented
    impl.assert(y_non_contig or !qy_needs_dequant, "y_non_contig || !qy_needs_dequant");
    impl.assert(!qx_needs_dequant or to_fp16_vk_0 != null, "!qx_needs_dequant || to_fp16_vk_0 != nullptr");
    impl.assert(!qy_needs_dequant or to_fp16_vk_1 != null, "!qy_needs_dequant || to_fp16_vk_1 != nullptr");
    const pipeline = dmmv orelse impl.abort("GGML_ASSERT(dmmv != nullptr) failed");

    const x_ne: u64 = @intCast(c.ggml_nelements(src0));
    const y_ne: u64 = @intCast(c.ggml_nelements(src1));
    const min_align: u64 = limits.minStorageBufferOffsetAlignment;

    const qx_sz = alignSize(typeSizeOf(src0.type) * x_ne / @as(u64, @intCast(c.ggml_blck_size(src0.type))), min_align);
    const x_sz = if (x_non_contig) alignSize(typeSizeOf(src0.type) * x_ne, min_align) else qx_sz;
    const y_sz: u64 = if (quantize_y)
        alignSize(y_ne, 128) * typeSizeOf(c.GGML_TYPE_Q8_1) / @as(u64, @intCast(c.ggml_blck_size(c.GGML_TYPE_Q8_1)))
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
    }

    const d_D = m.context.tensorSubbuffer(ctx, at(cgraph, node_idx + ctx.num_additional_fused_ops), false);
    const d_Qx = m.context.tensorSubbuffer(ctx, src0, false);
    const d_Qy = m.context.tensorSubbuffer(ctx, src1, false);
    var d_X: Subbuffer = undefined;
    var d_Y: Subbuffer = undefined;
    if (qx_needs_dequant) {
        d_X = sub(ctx.prealloc_x, 0, ctx.prealloc_x.?.size);
    } else {
        d_X = d_Qx;
        impl.assert(qx_sz == x_sz, "qx_sz == x_sz");
    }
    d_Y = if (qy_needs_dequant or quantize_y) sub(ctx.prealloc_y, 0, ctx.prealloc_y.?.size) else d_Qy;

    if (x_non_contig) {
        if (ctx.prealloc_x_need_sync) m.context.syncBuffers(ctx, subctx);
        impl.assert(x_sz == alignSize(typeSizeOf(src0.type) * x_ne, min_align), "x_sz == ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment)");
        cpyToContiguous(ctx, subctx, to_fp16_vk_0.?, src0, d_Qx, d_X);
    }
    if (y_non_contig) {
        impl.assert(y_sz == typeSizeOf(src1.type) * y_ne, "y_sz == ggml_type_size(src1->type) * y_ne");
        if (ctx.prealloc_y_last_pipeline_used != to_fp16_vk_1 or
            ctx.prealloc_y_last_tensor_used != src1 or
            ctx.prealloc_y_last_decode_vector_staging)
        {
            if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
            cpyToContiguous(ctx, subctx, to_fp16_vk_1.?, src1, d_Qy, d_Y);
            ctx.prealloc_y_last_pipeline_used = to_fp16_vk_1;
            ctx.prealloc_y_last_tensor_used = src1;
            ctx.prealloc_y_last_decode_vector_staging = false;
        }
    }
    if (quantize_y) {
        if (ctx.prealloc_y_last_pipeline_used != to_q8_1 or
            ctx.prealloc_y_last_tensor_used != src1 or
            ctx.prealloc_y_last_decode_vector_staging)
        {
            if (ctx.prealloc_y_need_sync) m.context.syncBuffers(ctx, subctx);
            quantizeQ81(ctx, subctx, d_Qy, d_Y, @truncate(y_ne));
            ctx.prealloc_y_last_pipeline_used = to_q8_1;
            ctx.prealloc_y_last_tensor_used = src1;
            ctx.prealloc_y_last_decode_vector_staging = false;
        }
    }

    // For batch_n, A is the same for each batch, and B/D use the row stride as the batch stride
    var stride_batch_x: u32 = if (batch_n) 0 else @truncate(ne00 *% ne01);
    var stride_batch_y: u32 = if (batch_n) @truncate(ne10) else @truncate(ne10 *% ne11);
    const stride_batch_d: u32 = if (batch_n) @truncate(ne20) else @truncate(ne20 *% ne21);
    if (!dim01(src0) and !qx_needs_dequant) stride_batch_x = @truncate(src0.nb[0] / typeSizeOf(src0.type));
    if (!dim01(src1) and !qy_needs_dequant) stride_batch_y = @truncate(src1.nb[0] / typeSizeOf(src1.type));

    const max_groups_x = limits.maxComputeWorkGroupCount[0];
    var groups_x: u32 = @truncate(ne01);
    var groups_z: u32 = 1;
    if (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = ceilDiv(groups_x, groups_z);
    }

    const biases = fusedBiases(ctx, cgraph, node_idx, d_D, true);

    const batches: u32 = @truncate(ne12 *% ne13);
    m.context.requestDescriptorSets(ctx, pipeline, ceilDiv(batches, limits.maxComputeWorkGroupCount[1]));

    var base_work_group_y: u32 = 0;
    while (base_work_group_y < batches) {
        const groups_y = @min(batches - base_work_group_y, limits.maxComputeWorkGroupCount[1]);
        const pc: push.MatVec = .{
            .ncols = @truncate(ne00),
            .stride_a = @truncate(ne10),
            .stride_b = @truncate(ne10),
            .stride_d = @truncate(ne01),
            .batch_stride_a = stride_batch_x,
            .batch_stride_b = stride_batch_y,
            .batch_stride_d = stride_batch_d,
            .fusion_flags = biases.flags,
            .base_work_group_y = base_work_group_y,
            .ne02 = @truncate(ne02),
            .ne12 = @truncate(ne12),
            .broadcast2 = @truncate(r2),
            .broadcast3 = @truncate(r3),
        };
        m.context.dispatch(ctx, subctx, pipeline, &.{ d_X, d_Y, d_D, biases.f0, biases.f1 }, &pc, .{ groups_x, groups_y, groups_z });
        base_work_group_y += groups_y;
    }

    if (x_non_contig) ctx.prealloc_x_need_sync = true;
    if (y_non_contig or quantize_y) ctx.prealloc_y_need_sync = true;
}

/// Ports `ggml_vk_mul_mat_vec_p021_f16_f32` (ggml-vulkan.cpp:9758 @c1d0e7a00):
/// f16 x f32 over the 0213-permuted operands attention produces, with
/// several Q heads per K head under grouped-query attention.
pub fn mulMatVecP021F16F32(ctx: *BackendCtx, subctx: *Context, cgraph: *const impl.CGraph, node_idx: c_int) void {
    const dst = at(cgraph, node_idx);
    const src0: *const Tensor = dst.src[0];
    const src1: *const Tensor = dst.src[1];
    impl.assert(c.ggml_is_permuted(src0) and c.ggml_is_permuted(src1), "ggml_is_permuted(src0) && ggml_is_permuted(src1)");
    impl.assert(src0.nb[0] <= src0.nb[1] and src0.nb[2] <= src0.nb[3], "src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]");
    impl.assert(src1.nb[0] <= src1.nb[1] and src1.nb[2] <= src1.nb[3], "src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]");
    impl.assert(src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F16");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

    const ne00: u64 = @intCast(src0.ne[0]);
    const ne01: u64 = @intCast(src0.ne[1]);
    const ne02: u64 = @intCast(src0.ne[2]);
    const ne11: u64 = @intCast(src1.ne[1]);
    const ne12: u64 = @intCast(src1.ne[2]);
    impl.assert(ne11 == 1, "ne11 == 1");

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    var gqa_ratio: u32 = @as(u32, @truncate(ne12)) / @as(u32, @truncate(ne02));
    if (gqa_ratio > 8 or gqa_ratio == 0 or ne12 != ne02 *% gqa_ratio) gqa_ratio = 1;

    var pipeline = ctx.device.pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1].?;
    if (c.ggml_nbytes(src0) > ctx.device.properties.limits.maxStorageBufferRange) pipeline = get64bIndexingPipeline(pipeline);
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    const out = at(cgraph, node_idx + ctx.num_additional_fused_ops);
    const d_D = m.context.tensorSubbuffer(ctx, out, true);
    const d_Qx = m.context.tensorSubbuffer(ctx, src0, false);
    const d_Qy = m.context.tensorSubbuffer(ctx, src1, true);
    const biases = fusedBiases(ctx, cgraph, node_idx, d_D, false);

    var pc: push.MatVecP021 = .{
        .ncols_x = @truncate(ne00),
        .nrows_x = @truncate(ne01),
        .nchannels_x = @truncate(ne02),
        .nchannels_y = @truncate(ne12),
        .b_offset = 0,
        .d_offset = 0,
        .fusion_flags = biases.flags,
    };
    ops.initPushconstTensorOffsets(ctx, &pc, src0, src1, null, null, out);

    var workgroups_z: u32 = @truncate(ne12);
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    if (gqa_ratio > 1) workgroups_z /= gqa_ratio;

    m.context.dispatch(ctx, subctx, pipeline, &.{ d_Qx, d_Qy, d_D, biases.f0, biases.f1 }, &pc, .{ 1, @truncate(ne01), workgroups_z });
}

/// Ports `ggml_vk_mul_mat_vec_nc_f16_f32` (ggml-vulkan.cpp:9850 @c1d0e7a00):
/// f16 x f32 with a non-contiguous, unpermuted A.
pub fn mulMatVecNcF16F32(ctx: *BackendCtx, subctx: *Context, cgraph: *const impl.CGraph, node_idx: c_int) void {
    const dst = at(cgraph, node_idx);
    const src0: *const Tensor = dst.src[0];
    const src1: *const Tensor = dst.src[1];
    impl.assert(!c.ggml_is_transposed(src0), "!ggml_is_transposed(src0)");
    impl.assert(!c.ggml_is_transposed(src1), "!ggml_is_transposed(src1)");
    impl.assert(!c.ggml_is_permuted(src0), "!ggml_is_permuted(src0)");
    impl.assert(src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F16");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

    const ne00: u64 = @intCast(src0.ne[0]);
    const ne01: u64 = @intCast(src0.ne[1]);
    const ne02: u64 = @intCast(src0.ne[2]);
    const ne03: u64 = @intCast(src0.ne[3]);
    const ne11: u64 = @intCast(src1.ne[1]);
    const ne12: u64 = @intCast(src1.ne[2]);
    const f16sz = @sizeOf(c.ggml_fp16_t);

    const nb03: u32 = @truncate(src0.nb[3] / f16sz);
    const nb13: u32 = @truncate(src1.nb[3] / @sizeOf(f32));
    const nb23: u32 = @truncate(dst.nb[3] / @sizeOf(f32));

    impl.assert(ne11 == 1, "ne11 == 1");
    impl.assert(src0.ne[3] == src1.ne[3], "src0->ne[3] == src1->ne[3]"); // checked in supports_op

    const row_stride_x: u32 = @truncate(src0.nb[1] / f16sz);
    const channel_stride_x: u32 = @truncate(src0.nb[2] / f16sz);
    const channel_stride_y: u32 = @truncate(src1.nb[2] / @sizeOf(f32));

    var pipeline = ctx.device.pipeline_mul_mat_vec_nc_f16_f32.?;
    if (c.ggml_nbytes(src0) > ctx.device.properties.limits.maxStorageBufferRange) pipeline = get64bIndexingPipeline(pipeline);
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    const out = at(cgraph, node_idx + ctx.num_additional_fused_ops);
    const d_D = m.context.tensorSubbuffer(ctx, out, true);
    const d_Qx = m.context.tensorSubbuffer(ctx, src0, false);
    const d_Qy = m.context.tensorSubbuffer(ctx, src1, true);
    const biases = fusedBiases(ctx, cgraph, node_idx, d_D, false);

    var pc: push.MatVecNc = .{
        .ncols_x = @truncate(ne00),
        .nrows_x = @truncate(ne01),
        .row_stride_x = row_stride_x,
        .channel_stride_x = channel_stride_x,
        .channel_stride_y = channel_stride_y,
        .channel_x_divisor = @truncate(ne12 / ne02),
        .ne12 = @truncate(ne12),
        .b_offset = 0,
        .d_offset = 0,
        .nb03 = nb03,
        .nb13 = nb13,
        .nb23 = nb23,
        .fusion_flags = biases.flags,
    };
    ops.initPushconstTensorOffsets(ctx, &pc, src0, src1, null, null, out);

    m.context.dispatch(ctx, subctx, pipeline, &.{ d_Qx, d_Qy, d_D, biases.f0, biases.f1 }, &pc, .{ @truncate(ne03), @truncate(ne01), @truncate(ne12) });
}

/// Ports `ggml_vk_fwht_pipeline_idx` (ggml-vulkan.cpp:9944 @c1d0e7a00).
fn fwhtPipelineIdx(n: i64) i32 {
    return switch (n) {
        64 => 0,
        128 => 1,
        256 => 2,
        512 => 3,
        else => -1,
    };
}

/// Ports `ggml_vk_can_use_fwht` (ggml-vulkan.cpp:9954 @c1d0e7a00): a
/// MUL_MAT whose A is hinted to be a Hadamard matrix runs as a fast
/// Walsh-Hadamard transform of B instead.
fn canUseFwht(ctx: *const BackendCtx, src1: *const Tensor, dst: *const Tensor) bool {
    if (ctx.num_additional_fused_ops != 0) return false;
    if (dst.op_params[1] != c.GGML_HINT_SRC0_IS_HADAMARD) return false;
    const idx = fwhtPipelineIdx(src1.ne[0]);
    if (idx < 0 or ctx.device.pipeline_fwht_f32[@intCast(idx)] == null) return false;
    if (src1.type != c.GGML_TYPE_F32 or dst.type != c.GGML_TYPE_F32) return false;
    if (!c.ggml_is_contiguous(src1)) return false;
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    return true;
}

/// Ports `ggml_vk_fwht` (ggml-vulkan.cpp:9980 @c1d0e7a00).
fn fwht(ctx: *BackendCtx, subctx: *Context, src: *const Tensor, dst: *Tensor) void {
    const idx = fwhtPipelineIdx(src.ne[0]);
    const pipeline = ctx.device.pipeline_fwht_f32[@intCast(idx)].?;

    const rows_per_workgroup: u32 = 4;
    const n_rows = ops.u32Of(c.ggml_nrows(src));
    const max_workgroups_x = ctx.device.properties.limits.maxComputeWorkGroupCount[0];
    const workgroups_x = @min(ceilDiv(n_rows, rows_per_workgroup), max_workgroups_x);
    m.context.requestDescriptorSets(ctx, pipeline, 1);

    const src_buf = m.context.tensorSubbuffer(ctx, src, true);
    const dst_buf = m.context.tensorSubbuffer(ctx, dst, true);

    var pc: push.Fwht = .{ .n_rows = n_rows, .src_offset = 0, .dst_offset = 0, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(src.ne[0]))) };
    ops.initPushconstTensorOffsets(ctx, &pc, src, null, null, null, dst);
    m.context.dispatch(ctx, subctx, pipeline, &.{ src_buf, dst_buf }, &pc, .{ workgroups_x, 1, 1 });
}

/// Ports `ggml_vk_mul_mat` (ggml-vulkan.cpp:10006 @c1d0e7a00): chooses the
/// path -- M split for an A past the storage range, FWHT, the two
/// attention-shaped vector kernels, the general vector kernel, or the
/// matrix-matrix one.
pub fn mulMat(ctx: *BackendCtx, subctx: *Context, cgraph: *const impl.CGraph, node_idx: c_int) void {
    const dst = at(cgraph, node_idx);
    const src0: *Tensor = dst.src[0];
    const src1: *Tensor = dst.src[1];
    const limits = &ctx.device.properties.limits;

    // A huge A splits along M -- the convolution case. Split_k does not
    // work with M splitting, and only batch size 1 is supported.
    const nbytes = c.ggml_nbytes(src0);
    const needs_split = dst.ne[2] == 1 and dst.ne[3] == 1 and nbytes > limits.maxStorageBufferRange;
    if (needs_split) {
        // The rows that fit, halved to allow for any additional offsets
        const M_split: u32 = @truncate(limits.maxStorageBufferRange / (2 * src0.nb[1]));
        var m_offset: u32 = 0;
        while (m_offset < dst.ne[0]) {
            const cur_M_size = @min(M_split, @as(u32, @truncate(@as(u64, @bitCast(dst.ne[0] - m_offset)))));
            var dst2 = dst.*;
            var src02 = src0.*;
            dst2.view_src = dst.view_src orelse dst;
            src02.view_src = src0.view_src orelse src0;
            dst2.view_offs += m_offset * dst.nb[0];
            src02.view_offs += m_offset * src0.nb[1];
            dst2.ne[0] = cur_M_size;
            src02.ne[1] = cur_M_size;
            mulMatQF16(ctx, subctx, &src02, src1, &dst2, true);
            m_offset += cur_M_size;
        }
    } else if (canUseFwht(ctx, src1, dst)) {
        fwht(ctx, subctx, src1, dst);
    } else if (src0.type == c.GGML_TYPE_F16 and c.ggml_is_permuted(src0) and c.ggml_is_permuted(src1) and dst.ne[1] == 1 and
        // detect 0213 permutation, and batch size of 1
        src0.nb[0] <= src0.nb[2] and
        src0.nb[2] <= src0.nb[1] and
        src0.nb[1] <= src0.nb[3] and
        src1.nb[0] <= src1.nb[2] and
        src1.nb[2] <= src1.nb[1] and
        src1.nb[1] <= src1.nb[3] and
        src0.ne[3] == 1 and
        src1.ne[3] == 1 and
        src0.ne[1] <= limits.maxComputeWorkGroupCount[1] and
        src1.ne[2] <= limits.maxComputeWorkGroupCount[2])
    {
        mulMatVecP021F16F32(ctx, subctx, cgraph, node_idx);
    } else if (src0.type == c.GGML_TYPE_F16 and !c.ggml_is_contiguous(src0) and !c.ggml_is_transposed(src1) and dst.ne[1] == 1 and
        !c.ggml_is_permuted(src0) and !c.ggml_is_permuted(src1) and
        src0.ne[3] <= limits.maxComputeWorkGroupCount[0] and
        src0.ne[1] <= limits.maxComputeWorkGroupCount[1] and
        src1.ne[2] <= limits.maxComputeWorkGroupCount[2])
    {
        mulMatVecNcF16F32(ctx, subctx, cgraph, node_idx);
        // mul_mat_vec batches ne12*ne13 when ne11 == 1, or takes ne11 as the
        // batch (up to mul_mat_vec_max_cols) when ne12 and ne13 are one.
    } else if ((dst.ne[1] == 1 or (dst.ne[1] <= m.device.mul_mat_vec_max_cols and src1.ne[2] * src1.ne[3] == 1)) and
        (src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_BF16 or c.ggml_is_quantized(src0.type)))
    {
        mulMatVecQF16(ctx, subctx, cgraph, node_idx);
    } else {
        mulMatQF16(ctx, subctx, src0, src1, dst, false);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
