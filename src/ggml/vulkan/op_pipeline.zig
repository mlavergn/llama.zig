//! Which pipeline computes a node: the per-op choice by source and
//! destination types, shapes and the current fusion, and the copy
//! pipelines `CPY`, `CONT` and `DUP` share with the matmul staging.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v1.0.0
//! (`b1b770d90`).

const std = @import("std");
const ggml = @import("../module.zig");

fn oom() noreturn {
    ggml.impl.abort("ggml_vulkan: out of memory");
}

fn isF16(t: *const ggml.impl.ggml_c.ggml_tensor) usize {
    return @intFromBool(t.type == ggml.impl.ggml_c.GGML_TYPE_F16);
}

/// Ports `ggml_vk_concat_unit_size` (ggml-vulkan.cpp:299 @b1b770d90): the
/// widest copy unit, for a quantized type the widest that divides a block.
pub fn concatUnitSize(ty: ggml.impl.ggml_c.ggml_type) u32 {
    const type_size: u32 = @intCast(ggml.impl.ggml_c.ggml_type_size(ty));
    if (!ggml.impl.ggml_c.ggml_is_quantized(ty)) return type_size;
    if (type_size % 8 == 0) return 8;
    if (type_size % 4 == 0) return 4;
    if (type_size % 2 == 0) return 2;
    return 1;
}

/// Ports `ggml_vk_concat_supported` (ggml-vulkan.cpp:318 @b1b770d90).
pub fn concatSupported(src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *const ggml.impl.ggml_c.ggml_tensor) bool {
    if (src0.type != src1.type or src0.type != dst.type) return false;
    if (!ggml.impl.ggml_c.ggml_is_quantized(src0.type)) {
        const type_size = ggml.impl.ggml_c.ggml_type_size(src0.type);
        return type_size == 1 or type_size == 2 or type_size == 4 or type_size == 8;
    }
    // Quantized tensor rows are block-aligned when created.
    return ggml.impl.ggml_c.ggml_is_contiguous_rows(src0) and ggml.impl.ggml_c.ggml_is_contiguous_rows(src1) and ggml.impl.ggml_c.ggml_is_contiguous_rows(dst);
}

/// Ports `ggml_vk_dim01_contiguous` (ggml-vulkan.cpp:6405 @b1b770d90).
pub fn dim01Contiguous(t: *const ggml.impl.ggml_c.ggml_tensor) bool {
    return t.nb[0] == ggml.impl.ggml_c.ggml_type_size(t.type) and
        t.nb[1] == (t.nb[0] * @as(usize, @intCast(t.ne[0]))) / @as(usize, @intCast(ggml.impl.ggml_c.ggml_blck_size(t.type))) and
        (t.ne[3] == 1 or t.nb[3] == t.nb[2] * @as(usize, @intCast(t.ne[2])));
}

/// Ports `ggml_vk_get_cpy_pipeline` (ggml-vulkan.cpp:6417 @b1b770d90): a
/// copy from `src`'s type to `to`, the contiguous variant when both sides
/// are, the transposing ones when a swap puts the innermost dimension
/// elsewhere.
pub fn getCpyPipeline(ctx: *ggml.vulkan.context.BackendCtx, src: *const ggml.impl.ggml_c.ggml_tensor, dst: ?*const ggml.impl.ggml_c.ggml_tensor, to: ggml.impl.ggml_c.ggml_type) ?*ggml.vulkan.pipeline.Pipeline {
    const dev = ctx.device;
    // Choose "contiguous copy" shader if src/dst are contiguous
    const contig = ggml.impl.ggml_c.ggml_is_contiguous(src) and (dst == null or ggml.impl.ggml_c.ggml_is_contiguous(dst.?));
    const to_size = ggml.impl.ggml_c.ggml_type_size(to);

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    const transpose = dst != null and src.nb[1] == to_size and ggml.impl.ggml_c.ggml_are_same_shape(dst.?, src);
    if (transpose and src.type == to) {
        if (to_size == 4) return dev.pipeline_cpy_transpose_32;
        if (to_size == 2) return dev.pipeline_cpy_transpose_16;
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    const transpose02 = dst != null and !contig and src.nb[2] == to_size and ggml.impl.ggml_c.ggml_is_contiguous(dst.?) and ggml.impl.ggml_c.ggml_are_same_shape(dst.?, src);
    if (transpose02 and src.type == to) {
        if (to_size == 4) return dev.pipeline_cpy_transpose_02_32;
        if (to_size == 2) return dev.pipeline_cpy_transpose_02_16;
    }

    // Same-type F32 and F16 copies take the type-agnostic 32- and 16-bit
    // copies at the end.
    const pairs = .{
        .{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16, "f32_f16" },
        .{ ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_F32, "f16_f32" },
        .{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_BF16, "f32_bf16" },
        .{ ggml.impl.ggml_c.GGML_TYPE_BF16, ggml.impl.ggml_c.GGML_TYPE_F32, "bf16_f32" },
        .{ ggml.impl.ggml_c.GGML_TYPE_BF16, ggml.impl.ggml_c.GGML_TYPE_F16, "bf16_f16" },
        .{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_I32, "f32_i32" },
        .{ ggml.impl.ggml_c.GGML_TYPE_I32, ggml.impl.ggml_c.GGML_TYPE_F32, "i32_f32" },
    };
    inline for (pairs) |p| {
        if (src.type == p[0] and to == p[1]) {
            return if (contig) @field(dev, "pipeline_contig_cpy_" ++ p[2]) else @field(dev, "pipeline_cpy_" ++ p[2]);
        }
    }

    const cpy_quants = [_]ggml.impl.ggml_c.ggml_type{ ggml.impl.ggml_c.GGML_TYPE_Q1_0, ggml.impl.ggml_c.GGML_TYPE_Q2_0, ggml.impl.ggml_c.GGML_TYPE_Q4_0, ggml.impl.ggml_c.GGML_TYPE_Q4_1, ggml.impl.ggml_c.GGML_TYPE_Q5_0, ggml.impl.ggml_c.GGML_TYPE_Q5_1, ggml.impl.ggml_c.GGML_TYPE_Q8_0, ggml.impl.ggml_c.GGML_TYPE_IQ4_NL };
    if (src.type == ggml.impl.ggml_c.GGML_TYPE_F32 and std.mem.indexOfScalar(ggml.impl.ggml_c.ggml_type, &cpy_quants, to) != null) return dev.pipeline_cpy_f32_quant[@intCast(to)];
    if (to == ggml.impl.ggml_c.GGML_TYPE_F32 and std.mem.indexOfScalar(ggml.impl.ggml_c.ggml_type, &cpy_quants, src.type) != null) return dev.pipeline_cpy_quant_f32[@intCast(src.type)];

    if (src.type == to) {
        // Copy two or four bytes at a time, depending on block size.
        const src_size = ggml.impl.ggml_c.ggml_type_size(src.type);
        ggml.impl.assert(ggml.impl.ggml_c.ggml_is_quantized(to) or src_size == 2 or src_size == 4, "ggml_is_quantized(to) || ggml_type_size(src->type) == 2 || ggml_type_size(src->type) == 4");
        if (src_size % 4 == 0) return if (contig) dev.pipeline_contig_cpy_32 else dev.pipeline_cpy_32;
        return if (contig) dev.pipeline_contig_cpy_16 else dev.pipeline_cpy_16;
    }

    ggml.impl.fprintfStderr("Missing CPY op for types: %s %s\n", .{ ggml.impl.ggml_c.ggml_type_name(src.type), ggml.impl.ggml_c.ggml_type_name(to) });
    ggml.impl.abort("fatal error");
}

/// Ports `ggml_vk_conv_select_shape` (ggml-vulkan.cpp:9045 @b1b770d90):
/// the largest tile that still gives every shader core two tiles.
pub fn convSelectShape(ctx: *const ggml.vulkan.context.BackendCtx, K: u32, NPQ: u32) ggml.vulkan.backend.ConvShape {
    const nTiles = struct {
        fn f(k: u32, npq: u32, s: ggml.vulkan.backend.ConvShape) u32 {
            const bs = ggml.vulkan.backend.conv_block_sizes[@backingInt(s)];
            return (std.math.divCeil(u32, k, bs.K) catch unreachable) * (std.math.divCeil(u32, npq, bs.NPQ) catch unreachable);
        }
    }.f;
    // We can't query number of shader cores on Intel, use 32 as a placeholder.
    const shader_core_count: u32 = if (ctx.device.shader_core_count > 0) ctx.device.shader_core_count else 32;

    // 128x128 isn't used with cm1 due to shared memory size.
    var allow_128x128 = true;
    if (ggml.vulkan.instance.glslc.coopmat and !ctx.device.coopmat2 and ctx.device.coopmat_support and ctx.device.coopmat_support_16x16x16_f16acc) allow_128x128 = false;

    if (allow_128x128 and K > 64 and nTiles(K, NPQ, .@"128x128") >= shader_core_count * 2) return .@"128x128";
    if (K <= 32 and nTiles(K, NPQ, .@"32x256") >= shader_core_count * 2) return .@"32x256";
    if (K <= 64 and nTiles(K, NPQ, .@"64x128") >= shader_core_count * 2) return .@"64x128";
    // cm1 fallback for large K when 128x128 isn't available
    if (!allow_128x128 and K > 64 and nTiles(K, NPQ, .@"64x128") >= shader_core_count * 2) return .@"64x128";
    return .@"64x32";
}

/// Ports `ggml_vk_conv3d_select_shape` (ggml-vulkan.cpp:9077 @b1b770d90):
/// the 256x128 coopmat2 tile, which reuses each loaded input across 256
/// output channels instead of 128, when `K` fills whole channel tiles and
/// there are at least as many workgroups as GPU cores; otherwise
/// `convSelectShape`'s answer.
pub fn conv3dSelectShape(ctx: *const ggml.vulkan.context.BackendCtx, K: u32, NPQ: u32) ggml.vulkan.backend.ConvShape {
    const block = ggml.vulkan.backend.conv_block_sizes[@backingInt(ggml.vulkan.backend.ConvShape.@"256x128")];
    const dev = ctx.device;
    const shader_core_count = dev.shader_core_count;
    // Slowdowns were observed on DGX Spark with drivers older than R615.
    const driver_supported = dev.driver_id != ggml.vulkan.common.vk.VK_DRIVER_ID_NVIDIA_PROPRIETARY or
        (dev.properties.driverVersion >> 22) >= 615;
    if (dev.coopmat2 and shader_core_count > 0 and K % block.K == 0 and driver_supported) {
        const n_tiles = @as(u64, K / block.K) * (std.math.divCeil(u64, NPQ, block.NPQ) catch unreachable);
        // Use the largest cm2 padding and include the dummy Csh[8].
        const pad: u32 = 8;
        const csh_elems: u32 = 8;
        const shmem: u32 = (block.K * (block.CRS + pad) + block.CRS * (block.NPQ + pad) + csh_elems) * @sizeOf(u16);
        // Reuse each loaded input across 256 output channels instead of 128.
        // Require full channel tiles and at least as many workgroups as GPU cores.
        if (n_tiles >= shader_core_count and shmem <= dev.properties.limits.maxComputeSharedMemorySize) return .@"256x128";
    }
    return convSelectShape(ctx, K, NPQ);
}

/// Ports `ggml_vk_rms_norm_pipeline` (ggml-vulkan.cpp:9097 @b1b770d90): the
/// 128, 256 or 512 block size, by row length.
///
/// Return: the pipeline, borrowed from the device.
pub fn rmsNormPipeline(pipelines: *const [3]?*ggml.vulkan.pipeline.Pipeline, ncols: i64) ?*ggml.vulkan.pipeline.Pipeline {
    if (pipelines[0] != null and ncols <= 128) return pipelines[0];
    if (pipelines[1] != null and ncols <= 256) return pipelines[1];
    return pipelines[2];
}

/// Ports `ggml_vk_concat_is_transposed` (ggml-vulkan.cpp:9107 @b1b770d90):
/// whether a dim-0 concat of 2-d tensors has a transposed `src1` the
/// tiled-transpose shader can read, one 32x32 tile per workgroup.
pub fn concatIsTransposed(device: *const ggml.vulkan.device.Device, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *const ggml.impl.ggml_c.ggml_tensor) bool {
    if (dst.op_params[0] != 0) return false; // dim 0 only
    if (src0.ne[2] != 1 or src0.ne[3] != 1 or src1.ne[2] != 1 or src1.ne[3] != 1) return false;
    const ts = ggml.impl.ggml_c.ggml_type_size(src0.type);
    if (src0.nb[0] != ts or dst.nb[0] != ts) return false; // src0 and dst rows must be contiguous
    if (src1.nb[0] <= src1.nb[1]) return false; // src1 must actually be transposed
    if (@divTrunc(src1.ne[1] + 31, 32) > device.properties.limits.maxComputeWorkGroupCount[0]) return false;
    return src0.ne[1] == src1.ne[1] and dst.ne[1] == src1.ne[1];
}

/// A dynamic pipeline: found in `map` by `key`, or made and recorded there,
/// under the compile mutex.
fn dynamicPipeline(ctx: *ggml.vulkan.context.BackendCtx, map: anytype, key: anytype) *ggml.vulkan.pipeline.Pipeline {
    const dev = ctx.device;
    dev.compile_mutex.lock();
    defer dev.compile_mutex.unlock();
    const gop = map.getOrPut(std.heap.c_allocator, key) catch oom();
    if (!gop.found_existing or gop.value_ptr.* == null) gop.value_ptr.* = ggml.vulkan.pipeline.Pipeline.create();
    return gop.value_ptr.*.?;
}

fn opI32(t: *const ggml.impl.ggml_c.ggml_tensor, i: usize) u32 {
    return @bitCast(t.op_params[i]);
}

/// Ports `ggml_vk_op_get_pipeline` (ggml-vulkan.cpp:9127 @b1b770d90).
///
/// Return: the pipeline, or null where the op has none for these types.
pub fn getPipeline(ctx: *ggml.vulkan.context.BackendCtx, src0_: ?*const ggml.impl.ggml_c.ggml_tensor, src1_: ?*const ggml.impl.ggml_c.ggml_tensor, src2_: ?*const ggml.impl.ggml_c.ggml_tensor, dst: *const ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op) ?*ggml.vulkan.pipeline.Pipeline {
    const dev = ctx.device;
    const F32 = ggml.impl.ggml_c.GGML_TYPE_F32;
    const F16 = ggml.impl.ggml_c.GGML_TYPE_F16;
    switch (op) {
        ggml.impl.ggml_c.GGML_OP_GET_ROWS => {
            const src0 = src0_.?;
            ggml.impl.assert(src1_.?.type == ggml.impl.ggml_c.GGML_TYPE_I32, "src1->type == GGML_TYPE_I32");
            if (src0.type == ggml.impl.ggml_c.GGML_TYPE_I32) {
                // i32 src only supports i32 result
                ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_I32, "dst->type == GGML_TYPE_I32");
                return dev.pipeline_get_rows[@intCast(src0.type)];
            }
            if (dst.type == F16) return dev.pipeline_get_rows[@intCast(src0.type)];
            if (dst.type == F32) return dev.pipeline_get_rows_f32[@intCast(src0.type)];
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK => {
            if (src0_.?.type == F32 and src1_.?.type == ggml.impl.ggml_c.GGML_TYPE_I32 and dst.type == F32) return dev.pipeline_get_rows_back_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_ACC => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_acc_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_SET => {
            const src0 = src0_.?;
            if (src0.type == src1_.?.type and src0.type == dst.type and (src0.type == F32 or src0.type == ggml.impl.ggml_c.GGML_TYPE_I32)) return dev.pipeline_set_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_SUB, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_DIV => {
            const src0 = src0_.?;
            const src1 = src1_.?;
            if ((src0.type != F32 and src0.type != F16) or (src1.type != F32 and src1.type != F16) or (dst.type != F32 and dst.type != F16)) return null;
            const same = ggml.impl.ggml_c.ggml_are_same_shape(src0, src1);
            const tbl = switch (op) {
                ggml.impl.ggml_c.GGML_OP_ADD => blk: {
                    if (ctx.num_additional_fused_ops > 0) {
                        const n: usize = @intCast(ctx.num_additional_fused_ops);
                        return if (ctx.do_add_rms_partials) dev.pipeline_multi_add_rms[n] else dev.pipeline_multi_add[n];
                    }
                    if (ctx.do_add_rms_partials) break :blk if (same) &dev.pipeline_add_rms_norepeat else &dev.pipeline_add_rms;
                    break :blk if (same) &dev.pipeline_add_norepeat else &dev.pipeline_add;
                },
                ggml.impl.ggml_c.GGML_OP_SUB => if (same) &dev.pipeline_sub_norepeat else &dev.pipeline_sub,
                ggml.impl.ggml_c.GGML_OP_MUL => if (same) &dev.pipeline_mul_norepeat else &dev.pipeline_mul,
                else => if (same) &dev.pipeline_div_norepeat else &dev.pipeline_div,
            };
            return tbl[isF16(src0)][isF16(src1)][isF16(dst)];
        },
        ggml.impl.ggml_c.GGML_OP_ADD_ID => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and src2_.?.type == ggml.impl.ggml_c.GGML_TYPE_I32 and dst.type == F32) return dev.pipeline_add_id_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_OUT_PROD => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_out_prod_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CONCAT => {
            if (!concatSupported(src0_.?, src1_.?, dst)) return null;
            if (!ggml.impl.ggml_c.ggml_is_quantized(src0_.?.type) and concatIsTransposed(dev, src0_.?, src1_.?, dst)) {
                switch (concatUnitSize(src0_.?.type)) {
                    1 => return dev.pipeline_concat_transpose_i8,
                    2 => return dev.pipeline_concat_transpose_i16,
                    4 => return dev.pipeline_concat_transpose_i32,
                    else => {},
                }
            }
            return switch (concatUnitSize(src0_.?.type)) {
                1 => dev.pipeline_concat_i8,
                2 => dev.pipeline_concat_i16,
                4 => dev.pipeline_concat_i32,
                8 => dev.pipeline_concat_i64,
                else => null,
            };
        },
        ggml.impl.ggml_c.GGML_OP_UPSCALE => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const mode = opI32(dst, 0) & (0xFF | @as(u32, ggml.impl.ggml_c.GGML_SCALE_FLAG_ANTIALIAS));
                return switch (mode) {
                    ggml.impl.ggml_c.GGML_SCALE_MODE_NEAREST => dev.pipeline_upscale_nearest_f32,
                    ggml.impl.ggml_c.GGML_SCALE_MODE_BILINEAR => dev.pipeline_upscale_bilinear_f32,
                    ggml.impl.ggml_c.GGML_SCALE_MODE_BICUBIC => dev.pipeline_upscale_bicubic_f32,
                    ggml.impl.ggml_c.GGML_SCALE_MODE_BILINEAR | ggml.impl.ggml_c.GGML_SCALE_FLAG_ANTIALIAS => dev.pipeline_upscale_bilinear_antialias_f32,
                    else => null,
                };
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_SCALE => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_scale_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_SQR, ggml.impl.ggml_c.GGML_OP_SQRT, ggml.impl.ggml_c.GGML_OP_SIN, ggml.impl.ggml_c.GGML_OP_COS, ggml.impl.ggml_c.GGML_OP_LOG, ggml.impl.ggml_c.GGML_OP_TRI, ggml.impl.ggml_c.GGML_OP_DIAG, ggml.impl.ggml_c.GGML_OP_CLAMP, ggml.impl.ggml_c.GGML_OP_LEAKY_RELU => {
            const src0 = src0_.?;
            if (src0.type == dst.type and (src0.type == F32 or src0.type == F16)) {
                const tbl = switch (op) {
                    ggml.impl.ggml_c.GGML_OP_SQR => &dev.pipeline_sqr,
                    ggml.impl.ggml_c.GGML_OP_SQRT => &dev.pipeline_sqrt,
                    ggml.impl.ggml_c.GGML_OP_SIN => &dev.pipeline_sin,
                    ggml.impl.ggml_c.GGML_OP_COS => &dev.pipeline_cos,
                    ggml.impl.ggml_c.GGML_OP_LOG => &dev.pipeline_log,
                    ggml.impl.ggml_c.GGML_OP_TRI => &dev.pipeline_tri,
                    ggml.impl.ggml_c.GGML_OP_DIAG => &dev.pipeline_diag,
                    ggml.impl.ggml_c.GGML_OP_CLAMP => &dev.pipeline_clamp,
                    else => &dev.pipeline_leaky_relu,
                };
                return tbl[isF16(dst)];
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_PAD => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_pad_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_pad_reflect_1d_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_ROLL => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_roll_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_REPEAT => {
            const s0 = ggml.impl.ggml_c.ggml_type_size(src0_.?.type);
            const sd = ggml.impl.ggml_c.ggml_type_size(dst.type);
            if (s0 == @sizeOf(f32) and sd == @sizeOf(f32)) return dev.pipeline_repeat_i32;
            if (s0 == 2 and sd == 2) return dev.pipeline_repeat_i16;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_REPEAT_BACK => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_repeat_back_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CPY, ggml.impl.ggml_c.GGML_OP_CONT, ggml.impl.ggml_c.GGML_OP_DUP => return getCpyPipeline(ctx, src0_.?, dst, dst.type),
        ggml.impl.ggml_c.GGML_OP_SET_ROWS => {
            const src0 = src0_.?;
            if (src0.type != F32 and src0.type != F16) return null;
            const src_idx = isF16(src0);
            if (src1_.?.type == ggml.impl.ggml_c.GGML_TYPE_I64) return dev.pipeline_set_rows_i64[src_idx][@intCast(dst.type)];
            if (src1_.?.type == ggml.impl.ggml_c.GGML_TYPE_I32) return dev.pipeline_set_rows_i32[src_idx][@intCast(dst.type)];
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_SILU_BACK => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_silu_back_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_NORM => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_norm_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_GROUP_NORM => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_group_norm_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_RMS_NORM => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const fused = ctx.fused_rms_norm_mode == .mul;
                if (ctx.do_add_rms_partials) return if (fused) dev.pipeline_rms_norm_mul_partials_f32 else dev.pipeline_rms_norm_partials_f32;
                return rmsNormPipeline(if (fused) &dev.pipeline_rms_norm_mul_f32 else &dev.pipeline_rms_norm_f32, src0_.?.ne[0]);
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_RMS_NORM_BACK => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_rms_norm_back_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_L2_NORM => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_l2_norm_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_UNARY => {
            const src0 = src0_.?;
            if ((src0.type != F32 and src0.type != F16) or (dst.type != F32 and dst.type != F16) or src0.type != dst.type) return null;
            const i = isF16(dst);
            return switch (ggml.impl.ggml_c.ggml_get_unary_op(dst)) {
                ggml.impl.ggml_c.GGML_UNARY_OP_EXP => dev.pipeline_exp[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_EXPM1 => dev.pipeline_expm1[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_ELU => dev.pipeline_elu[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_SILU => dev.pipeline_silu[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_GELU => dev.pipeline_gelu[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_GELU_ERF => dev.pipeline_gelu_erf[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_GELU_QUICK => dev.pipeline_gelu_quick[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_RELU => dev.pipeline_relu[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_XIELU => dev.pipeline_xielu[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_NEG => dev.pipeline_neg[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_TANH => dev.pipeline_tanh[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID => dev.pipeline_sigmoid[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_HARDSIGMOID => dev.pipeline_hardsigmoid[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_HARDSWISH => dev.pipeline_hardswish[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_ABS => dev.pipeline_abs[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS => dev.pipeline_softplus[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_STEP => dev.pipeline_step[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_ROUND => dev.pipeline_round[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_CEIL => dev.pipeline_ceil[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_FLOOR => dev.pipeline_floor[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_TRUNC => dev.pipeline_trunc[i],
                ggml.impl.ggml_c.GGML_UNARY_OP_SGN => dev.pipeline_sgn[i],
                else => null,
            };
        },
        ggml.impl.ggml_c.GGML_OP_GLU => {
            const src0 = src0_.?;
            if ((src0.type != F32 and src0.type != F16) or (dst.type != F32 and dst.type != F16) or src0.type != dst.type) return null;
            const i = isF16(dst);
            return switch (ggml.impl.ggml_c.ggml_get_glu_op(dst)) {
                ggml.impl.ggml_c.GGML_GLU_OP_GEGLU => dev.pipeline_geglu[i],
                ggml.impl.ggml_c.GGML_GLU_OP_REGLU => dev.pipeline_reglu[i],
                ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU => dev.pipeline_swiglu[i],
                ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_OAI => dev.pipeline_swiglu_oai[i],
                ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_CLAMP => dev.pipeline_swiglu_clamp[i],
                ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_ERF => dev.pipeline_geglu_erf[i],
                ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_QUICK => dev.pipeline_geglu_quick[i],
                else => null,
            };
        },
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX => {
            const src0 = src0_.?;
            ggml.impl.assert(src1_ == null or src1_.?.type == F32 or src1_.?.type == F16, "!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16");
            ggml.impl.assert(src2_ == null or src2_.?.type == F32, "!src2 || src2->type == GGML_TYPE_F32");
            if (ctx.num_additional_fused_ops != 0) {
                const idx: u32 = @intFromFloat(@ceil(@log2(@as(f32, @floatFromInt(dst.ne[0])))));
                ggml.impl.assert(idx < ggml.vulkan.device.num_topk_moe_pipelines, "idx < num_topk_moe_pipelines");
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
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) return dev.pipeline_soft_max_back_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_ROPE_BACK => {
            const src0 = src0_.?;
            const rope: *const ggml.impl.ggml_c.ggml_tensor = if (ctx.num_additional_fused_ops == 2) dst.src[0].*.src[0] else dst;
            const mode = rope.op_params[2];
            const is_neox = mode & ggml.impl.ggml_c.GGML_ROPE_TYPE_NEOX != 0;
            const is_mrope = mode & ggml.impl.ggml_c.GGML_ROPE_TYPE_MROPE != 0;
            const is_vision = mode == ggml.impl.ggml_c.GGML_ROPE_TYPE_VISION;
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
        ggml.impl.ggml_c.GGML_OP_SUM, ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_MEAN => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_sum_rows_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS => {
            if (src0_.?.type == F32 and src1_.?.type == F32 and dst.type == F32) {
                return if (src0_.?.ne[0] > 1024) dev.pipeline_cross_entropy_loss_f32_wg512 else dev.pipeline_cross_entropy_loss_f32;
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => {
            // src0 is the scalar grad; src1 is logits
            if (src0_.?.type == F32 and src1_.?.type == F32 and src2_ != null and src2_.?.type == F32 and dst.type == F32) {
                return if (src1_.?.ne[0] > 1024) dev.pipeline_cross_entropy_loss_back_f32_wg512 else dev.pipeline_cross_entropy_loss_back_f32;
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CUMSUM => {
            if (src0_.?.type == F32 and dst.type == F32) return if (src0_.?.ne[0] <= 512) dev.pipeline_cumsum_small_f32 else dev.pipeline_cumsum_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_SOLVE_TRI => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const key: ggml.vulkan.device.SolveTriPipelineState = .{ .N = @truncate(@as(u64, @bitCast(src0_.?.ne[0]))), .K = @truncate(@as(u64, @bitCast(src1_.?.ne[0]))) };
                return dynamicPipeline(ctx, &dev.pipeline_solve_tri_f32, key);
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_ARGMAX => {
            if (src0_.?.type == F32 and dst.type == ggml.impl.ggml_c.GGML_TYPE_I32) return dev.pipeline_argmax_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL => {
            if (src0_.?.type == ggml.impl.ggml_c.GGML_TYPE_I32 and src1_.?.type == ggml.impl.ggml_c.GGML_TYPE_I32 and dst.type == ggml.impl.ggml_c.GGML_TYPE_I64) return dev.pipeline_count_equal_i32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_IM2COL => {
            if (src1_.?.type == F32 and dst.type == F32) return dev.pipeline_im2col_f32;
            if (src1_.?.type == F32 and dst.type == F16) return dev.pipeline_im2col_f32_f16;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_IM2COL_3D => {
            if (src1_.?.type == F32 and dst.type == F32) return dev.pipeline_im2col_3d_f32;
            if (src1_.?.type == F32 and dst.type == F16) return dev.pipeline_im2col_3d_f32_f16;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_TIMESTEP_EMBEDDING => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_timestep_embedding_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_conv_transpose_1d_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_COL2IM_1D => return switch (src0_.?.type) {
            F32 => dev.pipeline_col2im_1d_f32,
            F16 => dev.pipeline_col2im_1d_f16,
            ggml.impl.ggml_c.GGML_TYPE_BF16 => dev.pipeline_col2im_1d_bf16,
            else => null,
        },
        ggml.impl.ggml_c.GGML_OP_POOL_1D => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_pool1d_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_POOL_2D => {
            if (src0_.?.type == F32 and dst.type == F32) {
                // Wide windows use neighboring subgroup lanes to read neighboring inputs.
                if (dev.pipeline_pool2d_subgroup_f32 != null and dst.op_params[1] >= @as(i32, @bitCast(dev.subgroup_size))) {
                    return dev.pipeline_pool2d_subgroup_f32;
                }
                return dev.pipeline_pool2d_f32;
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV6 => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_rwkv_wkv6_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV7 => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_rwkv_wkv7_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_GATED_LINEAR_ATTN => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_gated_linear_attn_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER => {
            // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
            if (ggml.vulkan.device.lightningIndexerKTypeSupported(src1_.?.type)) return dev.pipeline_lightning_indexer_f32[@intCast(src1_.?.type)];
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET => {
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
        ggml.impl.ggml_c.GGML_OP_SSM_SCAN => {
            if (src0_.?.type == F32 and dst.type == F32) {
                const d_state: u32 = @truncate(@as(u64, @bitCast(src0_.?.ne[0])));
                if (d_state == 128) return dev.pipeline_ssm_scan_f32_d128;
                if (d_state == 256) return dev.pipeline_ssm_scan_f32_d256;
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_SSM_CONV => {
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
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_ADAMW => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_opt_step_adamw_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_SGD => {
            if (src0_.?.type == F32 and dst.type == F32) return dev.pipeline_opt_step_sgd_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CONV_2D, ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D => {
            const src0 = src0_.?;
            const src1 = src1_.?;
            if (src1.type == F32 and dst.type == F32) {
                const K: u32 = @truncate(@as(u64, @bitCast(dst.ne[2]))); // Cout
                const NPQ: u32 = @truncate(@as(u64, @bitCast(dst.ne[3] *% dst.ne[1] *% dst.ne[0]))); // N * OH * OW
                const shape = convSelectShape(ctx, K, NPQ);

                const transpose = dst.op == ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D;
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
                const bs = ggml.vulkan.backend.conv_block_sizes[@backingInt(shape)];
                const aligned: u32 = @intFromBool(K % bs.K == 0 and CRS % bs.CRS == 0 and NPQ % bs.NPQ == 0);
                const key: ggml.vulkan.device.Conv2dPipelineState = .{ .s0 = s0, .s1 = s1, .p0 = p0, .p1 = p1, .d0 = d0, .d1 = d1, .KW = KW, .KH = KH, .aligned = aligned };

                const si = @backingInt(shape);
                const map = if (op == ggml.impl.ggml_c.GGML_OP_CONV_2D)
                    (if (src0.type == F32) &dev.pipeline_conv2d_f32[si] else if (src0.type == F16) &dev.pipeline_conv2d_f16_f32[si] else null)
                else
                    (if (src0.type == F32) &dev.pipeline_conv_transpose_2d_f32[si] else if (src0.type == F16) &dev.pipeline_conv_transpose_2d_f16_f32[si] else null);
                // The C++ dereferences a null map here; supports_op keeps it unreachable.
                return dynamicPipeline(ctx, map.?, key);
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CONV_2D_DW => {
            const src1 = src1_.?;
            if (src0_.?.type == F32 and dst.type == F32) {
                if (ggml.impl.ggml_c.ggml_is_contiguous(src1)) return dev.pipeline_conv2d_dw_whcn_f32;
                if (ggml.impl.ggml_c.ggml_is_contiguous_channels(src1)) return dev.pipeline_conv2d_dw_cwhn_f32;
            } else if (src0_.?.type == F16 and dst.type == F32) {
                if (ggml.impl.ggml_c.ggml_is_contiguous(src1)) return dev.pipeline_conv2d_dw_whcn_f16_f32;
                if (ggml.impl.ggml_c.ggml_is_contiguous_channels(src1)) return dev.pipeline_conv2d_dw_cwhn_f16_f32;
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_CONV_3D => {
            const src0 = src0_.?;
            if (src1_.?.type == F32 and dst.type == F32) {
                const OC = opI32(dst, 11);
                const IC = opI32(dst, 9);
                const N = opI32(dst, 10);
                const NPQ: u32 = N *% @as(u32, @truncate(@as(u64, @bitCast(dst.ne[2] *% dst.ne[1] *% dst.ne[0]))));
                const shape = conv3dSelectShape(ctx, OC, NPQ);

                const KW: u32 = @truncate(@as(u64, @bitCast(src0.ne[0])));
                const KH: u32 = @truncate(@as(u64, @bitCast(src0.ne[1])));
                const KD: u32 = @truncate(@as(u64, @bitCast(src0.ne[2])));
                const CRS = IC *% KW *% KH *% KD;
                const bs = ggml.vulkan.backend.conv_block_sizes[@backingInt(shape)];
                const aligned: u32 = @intFromBool(OC % bs.K == 0 and CRS % bs.CRS == 0 and NPQ % bs.NPQ == 0);
                const key: ggml.vulkan.device.Conv3dPipelineState = .{ .s0 = opI32(dst, 0), .s1 = opI32(dst, 1), .s2 = opI32(dst, 2), .p0 = opI32(dst, 3), .p1 = opI32(dst, 4), .p2 = opI32(dst, 5), .d0 = opI32(dst, 6), .d1 = opI32(dst, 7), .d2 = opI32(dst, 8), .KW = KW, .KH = KH, .KD = KD, .aligned = aligned };

                const si = @backingInt(shape);
                const map = if (src0.type == F32) &dev.pipeline_conv3d_f32[si] else if (src0.type == F16) &dev.pipeline_conv3d_f16_f32[si] else return null;
                return dynamicPipeline(ctx, map, key);
            }
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_ARANGE => {
            if (dst.type == F32) return dev.pipeline_arange_f32;
            return null;
        },
        ggml.impl.ggml_c.GGML_OP_FILL => {
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

test "rmsNormPipeline picks the block by row length, and skips a slot with no pipeline" {
    var slots: [3]ggml.vulkan.pipeline.Pipeline = undefined;
    const p: [3]?*ggml.vulkan.pipeline.Pipeline = .{ &slots[0], &slots[1], &slots[2] };
    try std.testing.expectEqual(p[0], rmsNormPipeline(&p, 1));
    try std.testing.expectEqual(p[0], rmsNormPipeline(&p, 128));
    try std.testing.expectEqual(p[1], rmsNormPipeline(&p, 129));
    try std.testing.expectEqual(p[1], rmsNormPipeline(&p, 256));
    try std.testing.expectEqual(p[2], rmsNormPipeline(&p, 257));
    try std.testing.expectEqual(p[2], rmsNormPipeline(&p, 1024));

    const none: [3]?*ggml.vulkan.pipeline.Pipeline = .{ null, null, &slots[2] };
    try std.testing.expectEqual(none[2], rmsNormPipeline(&none, 64));
}

test "conv3dSelectShape takes 256x128 only on coopmat2, past R615 on NVIDIA, with a workgroup per core" {
    // Only these fields are read on either path.
    var device: ggml.vulkan.device.Device = undefined;
    device.coopmat2 = true;
    device.coopmat_support = false;
    device.coopmat_support_16x16x16_f16acc = false;
    device.shader_core_count = 48;
    device.driver_id = ggml.vulkan.common.vk.VK_DRIVER_ID_NVIDIA_PROPRIETARY;
    device.properties.driverVersion = 615 << 22;
    // The shared memory the tile needs: (256 * 24 + 16 * 136 + 8) * 2.
    device.properties.limits.maxComputeSharedMemorySize = 16656;
    var ctx: ggml.vulkan.context.BackendCtx = undefined;
    ctx.device = &device;

    // 48 tiles of 128 is the least that gives each of the 48 cores one.
    try std.testing.expectEqual(ggml.vulkan.backend.ConvShape.@"256x128", conv3dSelectShape(&ctx, 256, 48 * 128));
    try std.testing.expectEqual(ggml.vulkan.backend.ConvShape.@"256x128", conv3dSelectShape(&ctx, 256, 47 * 128 + 1));
    try std.testing.expect(conv3dSelectShape(&ctx, 256, 47 * 128) != .@"256x128");
    // Two channel tiles count double; a partial channel tile never qualifies.
    try std.testing.expectEqual(ggml.vulkan.backend.ConvShape.@"256x128", conv3dSelectShape(&ctx, 512, 24 * 128));
    try std.testing.expect(conv3dSelectShape(&ctx, 384, 96 * 128) != .@"256x128");

    device.properties.limits.maxComputeSharedMemorySize = 16655;
    try std.testing.expect(conv3dSelectShape(&ctx, 256, 48 * 128) != .@"256x128");
    device.properties.limits.maxComputeSharedMemorySize = 16656;

    device.properties.driverVersion = (615 << 22) - 1;
    try std.testing.expect(conv3dSelectShape(&ctx, 256, 48 * 128) != .@"256x128");
    // The driver test is NVIDIA's alone.
    device.driver_id = ggml.vulkan.common.vk.VK_DRIVER_ID_MESA_NVK;
    try std.testing.expectEqual(ggml.vulkan.backend.ConvShape.@"256x128", conv3dSelectShape(&ctx, 256, 48 * 128));

    device.shader_core_count = 0;
    try std.testing.expect(conv3dSelectShape(&ctx, 256, 48 * 128) != .@"256x128");
    device.shader_core_count = 48;
    device.coopmat2 = false;
    try std.testing.expect(conv3dSelectShape(&ctx, 256, 48 * 128) != .@"256x128");
}
