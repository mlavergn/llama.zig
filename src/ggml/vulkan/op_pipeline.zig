//! Which pipeline computes a node: the per-op choice by source and
//! destination types, shapes and the current fusion, and the copy
//! pipelines `CPY`, `CONT` and `DUP` share with the matmul staging.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const allocator = common.allocator;
const BackendCtx = m.context.BackendCtx;
const Pipeline = m.pipeline.Pipeline;
const Tensor = c.ggml_tensor;
const ConvShape = m.backend.ConvShape;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

fn isF16(t: *const Tensor) usize {
    return @intFromBool(t.type == c.GGML_TYPE_F16);
}

/// Ports `ggml_vk_concat_unit_size` (ggml-vulkan.cpp:2421 @c1d0e7a00): the
/// widest copy unit, for a quantized type the widest that divides a block.
pub fn concatUnitSize(ty: c.ggml_type) u32 {
    const type_size: u32 = @intCast(c.ggml_type_size(ty));
    if (!c.ggml_is_quantized(ty)) return type_size;
    if (type_size % 8 == 0) return 8;
    if (type_size % 4 == 0) return 4;
    if (type_size % 2 == 0) return 2;
    return 1;
}

/// Ports `ggml_vk_concat_supported` (ggml-vulkan.cpp:2441 @c1d0e7a00).
pub fn concatSupported(src0: *const Tensor, src1: *const Tensor, dst: *const Tensor) bool {
    if (src0.type != src1.type or src0.type != dst.type) return false;
    if (!c.ggml_is_quantized(src0.type)) {
        const type_size = c.ggml_type_size(src0.type);
        return type_size == 1 or type_size == 2 or type_size == 4 or type_size == 8;
    }
    // Quantized tensor rows are block-aligned when created.
    return c.ggml_is_contiguous_rows(src0) and c.ggml_is_contiguous_rows(src1) and c.ggml_is_contiguous_rows(dst);
}

/// Ports `ggml_vk_dim01_contiguous` (ggml-vulkan.cpp:8919 @c1d0e7a00).
pub fn dim01Contiguous(t: *const Tensor) bool {
    return t.nb[0] == c.ggml_type_size(t.type) and
        t.nb[1] == (t.nb[0] * @as(usize, @intCast(t.ne[0]))) / @as(usize, @intCast(c.ggml_blck_size(t.type))) and
        (t.ne[3] == 1 or t.nb[3] == t.nb[2] * @as(usize, @intCast(t.ne[2])));
}

/// Ports `ggml_vk_get_cpy_pipeline` (ggml-vulkan.cpp:8926 @c1d0e7a00): a
/// copy from `src`'s type to `to`, the contiguous variant when both sides
/// are, the transposing ones when a swap puts the innermost dimension
/// elsewhere.
pub fn getCpyPipeline(ctx: *BackendCtx, src: *const Tensor, dst: ?*const Tensor, to: c.ggml_type) ?*Pipeline {
    const dev = ctx.device;
    // Choose "contiguous copy" shader if src/dst are contiguous
    const contig = c.ggml_is_contiguous(src) and (dst == null or c.ggml_is_contiguous(dst.?));
    const to_size = c.ggml_type_size(to);

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    const transpose = dst != null and src.nb[1] == to_size and c.ggml_are_same_shape(dst.?, src);
    if (transpose and src.type == to) {
        if (to_size == 4) return dev.pipeline_cpy_transpose_32;
        if (to_size == 2) return dev.pipeline_cpy_transpose_16;
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    const transpose02 = dst != null and !contig and src.nb[2] == to_size and c.ggml_is_contiguous(dst.?) and c.ggml_are_same_shape(dst.?, src);
    if (transpose02 and src.type == to) {
        if (to_size == 4) return dev.pipeline_cpy_transpose_02_32;
        if (to_size == 2) return dev.pipeline_cpy_transpose_02_16;
    }

    const pairs = .{
        .{ c.GGML_TYPE_F32, c.GGML_TYPE_F32, "f32_f32" },
        .{ c.GGML_TYPE_F32, c.GGML_TYPE_F16, "f32_f16" },
        .{ c.GGML_TYPE_F16, c.GGML_TYPE_F16, "f16_f16" },
        .{ c.GGML_TYPE_F16, c.GGML_TYPE_F32, "f16_f32" },
        .{ c.GGML_TYPE_F32, c.GGML_TYPE_BF16, "f32_bf16" },
        .{ c.GGML_TYPE_BF16, c.GGML_TYPE_F32, "bf16_f32" },
        .{ c.GGML_TYPE_F32, c.GGML_TYPE_I32, "f32_i32" },
        .{ c.GGML_TYPE_I32, c.GGML_TYPE_F32, "i32_f32" },
    };
    inline for (pairs) |p| {
        if (src.type == p[0] and to == p[1]) {
            return if (contig) @field(dev, "pipeline_contig_cpy_" ++ p[2]) else @field(dev, "pipeline_cpy_" ++ p[2]);
        }
    }

    const cpy_quants = [_]c.ggml_type{ c.GGML_TYPE_Q1_0, c.GGML_TYPE_Q2_0, c.GGML_TYPE_Q4_0, c.GGML_TYPE_Q4_1, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q5_1, c.GGML_TYPE_Q8_0, c.GGML_TYPE_IQ4_NL };
    if (src.type == c.GGML_TYPE_F32 and std.mem.indexOfScalar(c.ggml_type, &cpy_quants, to) != null) return dev.pipeline_cpy_f32_quant[@intCast(to)];
    if (to == c.GGML_TYPE_F32 and std.mem.indexOfScalar(c.ggml_type, &cpy_quants, src.type) != null) return dev.pipeline_cpy_quant_f32[@intCast(src.type)];

    if (src.type == to) {
        // Copy two or four bytes at a time, depending on block size.
        const src_size = c.ggml_type_size(src.type);
        impl.assert(c.ggml_is_quantized(to) or src_size == 2 or src_size == 4, "ggml_is_quantized(to) || ggml_type_size(src->type) == 2 || ggml_type_size(src->type) == 4");
        if (src_size % 4 == 0) return if (contig) dev.pipeline_contig_cpy_f32_f32 else dev.pipeline_cpy_f32_f32;
        return if (contig) dev.pipeline_contig_cpy_f16_f16 else dev.pipeline_cpy_f16_f16;
    }

    impl.fprintfStderr("Missing CPY op for types: %s %s\n", .{ c.ggml_type_name(src.type), c.ggml_type_name(to) });
    impl.abort("fatal error");
}

/// Ports `ggml_vk_conv_select_shape` (ggml-vulkan.cpp:11129 @c1d0e7a00):
/// the largest tile that still gives every shader core two tiles.
pub fn convSelectShape(ctx: *const BackendCtx, K: u32, NPQ: u32) ConvShape {
    const nTiles = struct {
        fn f(k: u32, npq: u32, s: ConvShape) u32 {
            const bs = m.backend.conv_block_sizes[@backingInt(s)];
            return (std.math.divCeil(u32, k, bs.K) catch unreachable) * (std.math.divCeil(u32, npq, bs.NPQ) catch unreachable);
        }
    }.f;
    // We can't query number of shader cores on Intel, use 32 as a placeholder.
    const shader_core_count: u32 = if (ctx.device.shader_core_count > 0) ctx.device.shader_core_count else 32;

    // 128x128 isn't used with cm1 due to shared memory size.
    var allow_128x128 = true;
    if (m.instance.glslc.coopmat and !ctx.device.coopmat2 and ctx.device.coopmat_support and ctx.device.coopmat_support_16x16x16_f16acc) allow_128x128 = false;

    if (allow_128x128 and K > 64 and nTiles(K, NPQ, .@"128x128") >= shader_core_count * 2) return .@"128x128";
    if (K <= 32 and nTiles(K, NPQ, .@"32x256") >= shader_core_count * 2) return .@"32x256";
    if (K <= 64 and nTiles(K, NPQ, .@"64x128") >= shader_core_count * 2) return .@"64x128";
    // cm1 fallback for large K when 128x128 isn't available
    if (!allow_128x128 and K > 64 and nTiles(K, NPQ, .@"64x128") >= shader_core_count * 2) return .@"64x128";
    return .@"64x32";
}

/// A dynamic pipeline: found in `map` by `key`, or made and recorded there,
/// under the compile mutex.
fn dynamicPipeline(ctx: *BackendCtx, map: anytype, key: anytype) *Pipeline {
    const dev = ctx.device;
    _ = std.c.pthread_mutex_lock(&dev.compile_mutex);
    defer _ = std.c.pthread_mutex_unlock(&dev.compile_mutex);
    const gop = map.getOrPut(allocator, key) catch oom();
    if (!gop.found_existing or gop.value_ptr.* == null) gop.value_ptr.* = Pipeline.create();
    return gop.value_ptr.*.?;
}

fn opI32(t: *const Tensor, i: usize) u32 {
    return @bitCast(t.op_params[i]);
}

/// Ports `ggml_vk_op_get_pipeline` (ggml-vulkan.cpp:11161 @c1d0e7a00).
///
/// Return: the pipeline, or null where the op has none for these types.
pub fn getPipeline(ctx: *BackendCtx, src0_: ?*const Tensor, src1_: ?*const Tensor, src2_: ?*const Tensor, dst: *const Tensor, op: c.enum_ggml_op) ?*Pipeline {
    const dev = ctx.device;
    const F32 = c.GGML_TYPE_F32;
    const F16 = c.GGML_TYPE_F16;
    switch (op) {
        c.GGML_OP_GET_ROWS => {
            const src0 = src0_.?;
            impl.assert(src1_.?.type == c.GGML_TYPE_I32, "src1->type == GGML_TYPE_I32");
            if (src0.type == c.GGML_TYPE_I32) {
                // i32 src only supports i32 result
                impl.assert(dst.type == c.GGML_TYPE_I32, "dst->type == GGML_TYPE_I32");
                return dev.pipeline_get_rows[@intCast(src0.type)];
            }
            if (dst.type == F16) return dev.pipeline_get_rows[@intCast(src0.type)];
            if (dst.type == F32) return dev.pipeline_get_rows_f32[@intCast(src0.type)];
            return null;
        },
        c.GGML_OP_GET_ROWS_BACK => {
            if (src0_.?.type == F32 and src1_.?.type == c.GGML_TYPE_I32 and dst.type == F32) return dev.pipeline_get_rows_back_f32;
            return null;
        },
        c.GGML_OP_ACC => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_acc_f32;
            return null;
        },
        c.GGML_OP_SET => {
            const src0 = src0_.?;
            if (src0.type == src1_.?.type and src0.type == dst.type and (src0.type == F32 or src0.type == c.GGML_TYPE_I32)) return dev.pipeline_set_f32;
            return null;
        },
        c.GGML_OP_ADD, c.GGML_OP_SUB, c.GGML_OP_MUL, c.GGML_OP_DIV => {
            const src0 = src0_.?;
            const src1 = src1_.?;
            if ((src0.type != F32 and src0.type != F16) or (src1.type != F32 and src1.type != F16) or (dst.type != F32 and dst.type != F16)) return null;
            const same = c.ggml_are_same_shape(src0, src1);
            const tbl = switch (op) {
                c.GGML_OP_ADD => blk: {
                    if (ctx.num_additional_fused_ops > 0) {
                        const n: usize = @intCast(ctx.num_additional_fused_ops);
                        return if (ctx.do_add_rms_partials) dev.pipeline_multi_add_rms[n] else dev.pipeline_multi_add[n];
                    }
                    if (ctx.do_add_rms_partials) break :blk if (same) &dev.pipeline_add_rms_norepeat else &dev.pipeline_add_rms;
                    break :blk if (same) &dev.pipeline_add_norepeat else &dev.pipeline_add;
                },
                c.GGML_OP_SUB => if (same) &dev.pipeline_sub_norepeat else &dev.pipeline_sub,
                c.GGML_OP_MUL => if (same) &dev.pipeline_mul_norepeat else &dev.pipeline_mul,
                else => if (same) &dev.pipeline_div_norepeat else &dev.pipeline_div,
            };
            return tbl[isF16(src0)][isF16(src1)][isF16(dst)];
        },
        c.GGML_OP_ADD_ID => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and src2_.?.type == c.GGML_TYPE_I32 and dst.type == F32) return dev.pipeline_add_id_f32;
            return null;
        },
        c.GGML_OP_OUT_PROD => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_out_prod_f32;
            return null;
        },
        c.GGML_OP_CONCAT => {
            if (!concatSupported(src0_.?, src1_.?, dst)) return null;
            return switch (concatUnitSize(src0_.?.type)) {
                1 => dev.pipeline_concat_i8,
                2 => dev.pipeline_concat_i16,
                4 => dev.pipeline_concat_i32,
                8 => dev.pipeline_concat_i64,
                else => null,
            };
        },
        c.GGML_OP_UPSCALE => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const mode = opI32(dst, 0) & (0xFF | @as(u32, c.GGML_SCALE_FLAG_ANTIALIAS));
                return switch (mode) {
                    c.GGML_SCALE_MODE_NEAREST => dev.pipeline_upscale_nearest_f32,
                    c.GGML_SCALE_MODE_BILINEAR => dev.pipeline_upscale_bilinear_f32,
                    c.GGML_SCALE_MODE_BICUBIC => dev.pipeline_upscale_bicubic_f32,
                    c.GGML_SCALE_MODE_BILINEAR | c.GGML_SCALE_FLAG_ANTIALIAS => dev.pipeline_upscale_bilinear_antialias_f32,
                    else => null,
                };
            }
            return null;
        },
        c.GGML_OP_SCALE => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_scale_f32;
            return null;
        },
        c.GGML_OP_SQR, c.GGML_OP_SQRT, c.GGML_OP_SIN, c.GGML_OP_COS, c.GGML_OP_LOG, c.GGML_OP_TRI, c.GGML_OP_DIAG, c.GGML_OP_CLAMP, c.GGML_OP_LEAKY_RELU => {
            const src0 = src0_.?;
            if (src0.type == dst.type and (src0.type == F32 or src0.type == F16)) {
                const tbl = switch (op) {
                    c.GGML_OP_SQR => &dev.pipeline_sqr,
                    c.GGML_OP_SQRT => &dev.pipeline_sqrt,
                    c.GGML_OP_SIN => &dev.pipeline_sin,
                    c.GGML_OP_COS => &dev.pipeline_cos,
                    c.GGML_OP_LOG => &dev.pipeline_log,
                    c.GGML_OP_TRI => &dev.pipeline_tri,
                    c.GGML_OP_DIAG => &dev.pipeline_diag,
                    c.GGML_OP_CLAMP => &dev.pipeline_clamp,
                    else => &dev.pipeline_leaky_relu,
                };
                return tbl[isF16(dst)];
            }
            return null;
        },
        c.GGML_OP_PAD => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_pad_f32;
            return null;
        },
        c.GGML_OP_PAD_REFLECT_1D => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_pad_reflect_1d_f32;
            return null;
        },
        c.GGML_OP_ROLL => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_roll_f32;
            return null;
        },
        c.GGML_OP_REPEAT => {
            const s0 = c.ggml_type_size(src0_.?.type);
            const sd = c.ggml_type_size(dst.type);
            if (s0 == @sizeOf(f32) and sd == @sizeOf(f32)) return dev.pipeline_repeat_i32;
            if (s0 == 2 and sd == 2) return dev.pipeline_repeat_i16;
            return null;
        },
        c.GGML_OP_REPEAT_BACK => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_repeat_back_f32;
            return null;
        },
        c.GGML_OP_CPY, c.GGML_OP_CONT, c.GGML_OP_DUP => return getCpyPipeline(ctx, src0_.?, dst, dst.type),
        c.GGML_OP_SET_ROWS => {
            const src0 = src0_.?;
            if (src0.type != F32 and src0.type != F16) return null;
            const src_idx = isF16(src0);
            if (src1_.?.type == c.GGML_TYPE_I64) return dev.pipeline_set_rows_i64[src_idx][@intCast(dst.type)];
            if (src1_.?.type == c.GGML_TYPE_I32) return dev.pipeline_set_rows_i32[src_idx][@intCast(dst.type)];
            return null;
        },
        c.GGML_OP_SILU_BACK => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_silu_back_f32;
            return null;
        },
        c.GGML_OP_NORM => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_norm_f32;
            return null;
        },
        c.GGML_OP_GROUP_NORM => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_group_norm_f32;
            return null;
        },
        c.GGML_OP_RMS_NORM => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const fused = ctx.num_additional_fused_ops > 0;
                if (ctx.do_add_rms_partials) return if (fused) dev.pipeline_rms_norm_mul_partials_f32 else dev.pipeline_rms_norm_partials_f32;
                return if (fused) dev.pipeline_rms_norm_mul_f32 else dev.pipeline_rms_norm_f32;
            }
            return null;
        },
        c.GGML_OP_RMS_NORM_BACK => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_rms_norm_back_f32;
            return null;
        },
        c.GGML_OP_L2_NORM => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_l2_norm_f32;
            return null;
        },
        c.GGML_OP_UNARY => {
            const src0 = src0_.?;
            if ((src0.type != F32 and src0.type != F16) or (dst.type != F32 and dst.type != F16) or src0.type != dst.type) return null;
            const i = isF16(dst);
            return switch (c.ggml_get_unary_op(dst)) {
                c.GGML_UNARY_OP_EXP => dev.pipeline_exp[i],
                c.GGML_UNARY_OP_EXPM1 => dev.pipeline_expm1[i],
                c.GGML_UNARY_OP_ELU => dev.pipeline_elu[i],
                c.GGML_UNARY_OP_SILU => dev.pipeline_silu[i],
                c.GGML_UNARY_OP_GELU => dev.pipeline_gelu[i],
                c.GGML_UNARY_OP_GELU_ERF => dev.pipeline_gelu_erf[i],
                c.GGML_UNARY_OP_GELU_QUICK => dev.pipeline_gelu_quick[i],
                c.GGML_UNARY_OP_RELU => dev.pipeline_relu[i],
                c.GGML_UNARY_OP_XIELU => dev.pipeline_xielu[i],
                c.GGML_UNARY_OP_NEG => dev.pipeline_neg[i],
                c.GGML_UNARY_OP_TANH => dev.pipeline_tanh[i],
                c.GGML_UNARY_OP_SIGMOID => dev.pipeline_sigmoid[i],
                c.GGML_UNARY_OP_HARDSIGMOID => dev.pipeline_hardsigmoid[i],
                c.GGML_UNARY_OP_HARDSWISH => dev.pipeline_hardswish[i],
                c.GGML_UNARY_OP_ABS => dev.pipeline_abs[i],
                c.GGML_UNARY_OP_SOFTPLUS => dev.pipeline_softplus[i],
                c.GGML_UNARY_OP_STEP => dev.pipeline_step[i],
                c.GGML_UNARY_OP_ROUND => dev.pipeline_round[i],
                c.GGML_UNARY_OP_CEIL => dev.pipeline_ceil[i],
                c.GGML_UNARY_OP_FLOOR => dev.pipeline_floor[i],
                c.GGML_UNARY_OP_TRUNC => dev.pipeline_trunc[i],
                c.GGML_UNARY_OP_SGN => dev.pipeline_sgn[i],
                else => null,
            };
        },
        c.GGML_OP_GLU => {
            const src0 = src0_.?;
            if ((src0.type != F32 and src0.type != F16) or (dst.type != F32 and dst.type != F16) or src0.type != dst.type) return null;
            const i = isF16(dst);
            return switch (c.ggml_get_glu_op(dst)) {
                c.GGML_GLU_OP_GEGLU => dev.pipeline_geglu[i],
                c.GGML_GLU_OP_REGLU => dev.pipeline_reglu[i],
                c.GGML_GLU_OP_SWIGLU => dev.pipeline_swiglu[i],
                c.GGML_GLU_OP_SWIGLU_OAI => dev.pipeline_swiglu_oai[i],
                c.GGML_GLU_OP_GEGLU_ERF => dev.pipeline_geglu_erf[i],
                c.GGML_GLU_OP_GEGLU_QUICK => dev.pipeline_geglu_quick[i],
                else => null,
            };
        },
        c.GGML_OP_DIAG_MASK_INF => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_diag_mask_inf_f32;
            return null;
        },
        c.GGML_OP_SOFT_MAX => {
            const src0 = src0_.?;
            impl.assert(src1_ == null or src1_.?.type == F32 or src1_.?.type == F16, "!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16");
            impl.assert(src2_ == null or src2_.?.type == F32, "!src2 || src2->type == GGML_TYPE_F32");
            if (ctx.num_additional_fused_ops != 0) {
                const idx: u32 = @intFromFloat(@ceil(@log2(@as(f32, @floatFromInt(dst.ne[0])))));
                impl.assert(idx < m.device.num_topk_moe_pipelines, "idx < num_topk_moe_pipelines");
                // use n_experts from push constant if it's not equal to the power of two spec constant
                const use_push = dst.ne[0] != (@as(i64, 1) << @intCast(idx));
                return dev.pipeline_topk_moe[idx][@intFromBool(use_push)];
            }
            if (src0.type == F32 and (src1_ == null or src1_.?.type == F32) and dst.type == F32) {
                return if (src0.ne[0] > 1024) dev.pipeline_soft_max_f32_wg512 else dev.pipeline_soft_max_f32;
            }
            if (src0.type == F32 and src1_.?.type == F16 and dst.type == F32) {
                return if (src0.ne[0] > 1024) dev.pipeline_soft_max_f32_f16_wg512 else dev.pipeline_soft_max_f32_f16;
            }
            return null;
        },
        c.GGML_OP_SOFT_MAX_BACK => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_soft_max_back_f32;
            return null;
        },
        c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => {
            const src0 = src0_.?;
            const rope: *const Tensor = if (ctx.num_additional_fused_ops == 2) dst.src[0].*.src[0] else dst;
            const mode = rope.op_params[2];
            const is_neox = mode & c.GGML_ROPE_TYPE_NEOX != 0;
            const is_mrope = mode & c.GGML_ROPE_TYPE_MROPE != 0;
            const is_vision = mode == c.GGML_ROPE_TYPE_VISION;
            const s32 = src0.type == F32;
            const s16 = src0.type == F16;
            const d32 = dst.type == F32;
            const d16 = dst.type == F16;
            if (is_neox) {
                if (s32 and d32) return dev.pipeline_rope_neox_f32;
                if (s32 and d16) return dev.pipeline_rope_neox_f32_f16;
                if (s16 and d16) return dev.pipeline_rope_neox_f16;
            } else if (is_mrope and !is_vision) {
                if (s32 and d32) return dev.pipeline_rope_multi_f32;
                if (s32 and d16) return dev.pipeline_rope_multi_f32_f16;
                if (s16 and d16) return dev.pipeline_rope_multi_f16;
            } else if (is_vision) {
                if (s32 and d32) return dev.pipeline_rope_vision_f32;
                if (s16 and d16) return dev.pipeline_rope_vision_f16;
            } else {
                if (s32 and d32) return dev.pipeline_rope_norm_f32;
                if (s32 and d16) return dev.pipeline_rope_norm_f32_f16;
                if (s16 and d16) return dev.pipeline_rope_norm_f16;
            }
            return null;
        },
        c.GGML_OP_SUM, c.GGML_OP_SUM_ROWS, c.GGML_OP_MEAN => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_sum_rows_f32;
            return null;
        },
        c.GGML_OP_CUMSUM => {
            if (src0_.?.type == F32 and dst.type == F32) return if (src0_.?.ne[0] <= 512) dev.pipeline_cumsum_small_f32 else dev.pipeline_cumsum_f32;
            return null;
        },
        c.GGML_OP_SOLVE_TRI => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const key: m.device.SolveTriPipelineState = .{ .N = @truncate(@as(u64, @bitCast(src0_.?.ne[0]))), .K = @truncate(@as(u64, @bitCast(src1_.?.ne[0]))) };
                return dynamicPipeline(ctx, &dev.pipeline_solve_tri_f32, key);
            }
            return null;
        },
        c.GGML_OP_ARGMAX => {
            if (src0_.?.type == F32 and dst.type == c.GGML_TYPE_I32) return dev.pipeline_argmax_f32;
            return null;
        },
        c.GGML_OP_COUNT_EQUAL => {
            if (src0_.?.type == c.GGML_TYPE_I32 and src1_.?.type == c.GGML_TYPE_I32 and dst.type == c.GGML_TYPE_I64) return dev.pipeline_count_equal_i32;
            return null;
        },
        c.GGML_OP_IM2COL => {
            if (src1_.?.type == F32 and dst.type == F32) return dev.pipeline_im2col_f32;
            if (src1_.?.type == F32 and dst.type == F16) return dev.pipeline_im2col_f32_f16;
            return null;
        },
        c.GGML_OP_IM2COL_3D => {
            if (src1_.?.type == F32 and dst.type == F32) return dev.pipeline_im2col_3d_f32;
            if (src1_.?.type == F32 and dst.type == F16) return dev.pipeline_im2col_3d_f32_f16;
            return null;
        },
        c.GGML_OP_TIMESTEP_EMBEDDING => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_timestep_embedding_f32;
            return null;
        },
        c.GGML_OP_CONV_TRANSPOSE_1D => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_conv_transpose_1d_f32;
            return null;
        },
        c.GGML_OP_COL2IM_1D => return switch (src0_.?.type) {
            F32 => dev.pipeline_col2im_1d_f32,
            F16 => dev.pipeline_col2im_1d_f16,
            c.GGML_TYPE_BF16 => dev.pipeline_col2im_1d_bf16,
            else => null,
        },
        c.GGML_OP_POOL_1D => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_pool1d_f32;
            return null;
        },
        c.GGML_OP_POOL_2D => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_pool2d_f32;
            return null;
        },
        c.GGML_OP_RWKV_WKV6 => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_rwkv_wkv6_f32;
            return null;
        },
        c.GGML_OP_RWKV_WKV7 => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_rwkv_wkv7_f32;
            return null;
        },
        c.GGML_OP_GATED_LINEAR_ATTN => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_gated_linear_attn_f32;
            return null;
        },
        c.GGML_OP_GATED_DELTA_NET => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const S_v: u32 = @truncate(@as(u64, @bitCast(dst.src[2].*.ne[0])));
                const kda: usize = @intFromBool(dst.src[3].*.ne[0] == @as(i64, S_v));
                const si: usize = switch (S_v) {
                    16 => 0,
                    32 => 1,
                    64 => 2,
                    128 => 3,
                    else => return null,
                };
                return dev.pipeline_gated_delta_net[si][kda];
            }
            return null;
        },
        c.GGML_OP_SSM_SCAN => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const d_state: u32 = @truncate(@as(u64, @bitCast(src0_.?.ne[0])));
                if (d_state == 128) return dev.pipeline_ssm_scan_f32_d128;
                if (d_state == 256) return dev.pipeline_ssm_scan_f32_d256;
            }
            return null;
        },
        c.GGML_OP_SSM_CONV => {
            if (src0_.?.type == F32 and dst.type == F32) {
                return switch (ctx.num_additional_fused_ops) {
                    0 => dev.pipeline_ssm_conv_f32,
                    1 => dev.pipeline_ssm_conv_silu_f32,
                    2 => dev.pipeline_ssm_conv_bias_silu_f32,
                    else => null,
                };
            }
            return null;
        },
        c.GGML_OP_OPT_STEP_ADAMW => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_opt_step_adamw_f32;
            return null;
        },
        c.GGML_OP_OPT_STEP_SGD => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_opt_step_sgd_f32;
            return null;
        },
        c.GGML_OP_CONV_2D, c.GGML_OP_CONV_TRANSPOSE_2D => {
            const src0 = src0_.?;
            const src1 = src1_.?;
            if (src1.type == F32 and dst.type == F32) {
                const K: u32 = @truncate(@as(u64, @bitCast(dst.ne[2]))); // Cout
                const NPQ: u32 = @truncate(@as(u64, @bitCast(dst.ne[3] *% dst.ne[1] *% dst.ne[0]))); // N * OH * OW
                const shape = convSelectShape(ctx, K, NPQ);

                const transpose = dst.op == c.GGML_OP_CONV_TRANSPOSE_2D;
                const KW: u32 = @truncate(@as(u64, @bitCast(src0.ne[0])));
                const KH: u32 = @truncate(@as(u64, @bitCast(src0.ne[1])));
                const s0 = opI32(dst, 0);
                const s1 = if (!transpose) opI32(dst, 1) else s0;
                const p0: u32 = if (!transpose) opI32(dst, 2) else 0;
                const p1: u32 = if (!transpose) opI32(dst, 3) else 0;
                const d0: u32 = if (!transpose) opI32(dst, 4) else 1;
                const d1: u32 = if (!transpose) opI32(dst, 5) else 1;

                // tile-aligned shapes let the shader skip bounds checks
                const Cin: u32 = @truncate(@as(u64, @bitCast(src1.ne[2])));
                const CRS = Cin *% KW *% KH;
                const bs = m.backend.conv_block_sizes[@backingInt(shape)];
                const aligned: u32 = @intFromBool(K % bs.K == 0 and CRS % bs.CRS == 0 and NPQ % bs.NPQ == 0);
                const key: m.device.Conv2dPipelineState = .{ .s0 = s0, .s1 = s1, .p0 = p0, .p1 = p1, .d0 = d0, .d1 = d1, .KW = KW, .KH = KH, .aligned = aligned };

                const si = @backingInt(shape);
                const map = if (op == c.GGML_OP_CONV_2D)
                    (if (src0.type == F32) &dev.pipeline_conv2d_f32[si] else if (src0.type == F16) &dev.pipeline_conv2d_f16_f32[si] else null)
                else
                    (if (src0.type == F32) &dev.pipeline_conv_transpose_2d_f32[si] else if (src0.type == F16) &dev.pipeline_conv_transpose_2d_f16_f32[si] else null);
                // The C++ dereferences a null map here; supports_op keeps it unreachable.
                return dynamicPipeline(ctx, map.?, key);
            }
            return null;
        },
        c.GGML_OP_CONV_2D_DW => {
            const src1 = src1_.?;
            if (src0_.?.type == F32 and dst.type == F32) {
                if (c.ggml_is_contiguous(src1)) return dev.pipeline_conv2d_dw_whcn_f32;
                if (c.ggml_is_contiguous_channels(src1)) return dev.pipeline_conv2d_dw_cwhn_f32;
            } else if (src0_.?.type == F16 and dst.type == F32) {
                if (c.ggml_is_contiguous(src1)) return dev.pipeline_conv2d_dw_whcn_f16_f32;
                if (c.ggml_is_contiguous_channels(src1)) return dev.pipeline_conv2d_dw_cwhn_f16_f32;
            }
            return null;
        },
        c.GGML_OP_CONV_3D => {
            const src0 = src0_.?;
            if (src1_.?.type == F32 and dst.type == F32) {
                const OC = opI32(dst, 11);
                const IC = opI32(dst, 9);
                const N = opI32(dst, 10);
                const NPQ: u32 = N *% @as(u32, @truncate(@as(u64, @bitCast(dst.ne[2] *% dst.ne[1] *% dst.ne[0]))));
                const shape = convSelectShape(ctx, OC, NPQ);

                const KW: u32 = @truncate(@as(u64, @bitCast(src0.ne[0])));
                const KH: u32 = @truncate(@as(u64, @bitCast(src0.ne[1])));
                const KD: u32 = @truncate(@as(u64, @bitCast(src0.ne[2])));
                const CRS = IC *% KW *% KH *% KD;
                const bs = m.backend.conv_block_sizes[@backingInt(shape)];
                const aligned: u32 = @intFromBool(OC % bs.K == 0 and CRS % bs.CRS == 0 and NPQ % bs.NPQ == 0);
                const key: m.device.Conv3dPipelineState = .{ .s0 = opI32(dst, 0), .s1 = opI32(dst, 1), .s2 = opI32(dst, 2), .p0 = opI32(dst, 3), .p1 = opI32(dst, 4), .p2 = opI32(dst, 5), .d0 = opI32(dst, 6), .d1 = opI32(dst, 7), .d2 = opI32(dst, 8), .KW = KW, .KH = KH, .KD = KD, .aligned = aligned };

                const si = @backingInt(shape);
                const map = if (src0.type == F32) &dev.pipeline_conv3d_f32[si] else if (src0.type == F16) &dev.pipeline_conv3d_f16_f32[si] else return null;
                return dynamicPipeline(ctx, map, key);
            }
            return null;
        },
        c.GGML_OP_ADD1 => {
            const t0 = src0_.?.type;
            const t1 = src1_.?.type;
            if (t0 == F16 and t1 == F16 and dst.type == F16) return dev.pipeline_add1_f16_f16;
            if (t0 == F16 and t1 == F32 and dst.type == F16) return dev.pipeline_add1_f16_f32;
            if (t0 == F32 and t1 == F32 and dst.type == F32) return dev.pipeline_add1_f32_f32;
            return null;
        },
        c.GGML_OP_ARANGE => {
            if (dst.type == F32) return dev.pipeline_arange_f32;
            return null;
        },
        c.GGML_OP_FILL => {
            if (dst.type == F32) return dev.pipeline_fill_f32;
            if (dst.type == F16) return dev.pipeline_fill_f16;
            return null;
        },
        else => return null,
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
