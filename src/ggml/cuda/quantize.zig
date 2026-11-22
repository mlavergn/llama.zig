//! Quantizing `src1` for the integer matrix kernels: `q8_1` rows for MMVQ,
//! the `block_q8_1_mmq` and FP4 layouts for MMQ.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/quantize.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/quantize.cuh` at v1.0.0 (`b1b770d90`). The
//! `quantize_*` kernels stay CUDA. There is no `quantize_cuda_t` typedef, and
//! `quantize_row_q8_1_cuda` takes no `ids` or `type_src0`; few-row MMQ has
//! its own `src1` layout.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `CUDA_QUANTIZE_BLOCK_SIZE` (quantize.cuh:8 @b1b770d90).
pub const block_size = 256;

/// Ports `CUDA_QUANTIZE_BLOCK_SIZE_MMQ` (quantize.cuh:9 @b1b770d90).
pub const block_size_mmq = 128;

/// Ports `MMQ_FEW_NCOLS` (quantize.cuh:28 @b1b770d90): `src1` columns per
/// column group of few-row MMQ.
pub const mmq_few_ncols = 8;

/// Ports `MMQ_FEW_Q8_1_INT4` (quantize.cuh:29 @b1b770d90): `int4`s per 256
/// values and column group.
pub const mmq_few_q8_1_int4 = 4 * ggml.cuda.common.warp_size + 4 * 4;

/// CUDA's `int4`, for sizing allocations.
pub const Int4 = extern struct { x: c_int, y: c_int, z: c_int, w: c_int };

fn u32Of(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `quantize_row_q8_1_cuda` (quantize.cu:559 @b1b770d90).
pub fn quantizeRowQ8_1(x: ?*const f32, vy: ?*anyopaque, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    ggml.impl.assert(@rem(ne0, ggml.impl.ggml_c.QK8_1) == 0, "ne0 % QK8_1 == 0");

    const ne2_fastdiv = ggml.cuda.common.initFastdivValues(@bitCast(ne2));

    const block_num_x = @divTrunc(ne0 + block_size - 1, block_size);
    ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("quantize", "quantize_q8_1"), .{
        .grid = .{ .x = u32Of(block_num_x), .y = u32Of(ne1), .z = u32Of(ne2 * ne3) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, vy, ne00, s01, s02, s03, ne0, u32Of(ne1), ne2_fastdiv });
}

fn q8_1MmqKernel(comptime layout: ggml.cuda.mmq.Q8_1DsLayout, comptime scatter: bool) ggml.cuda.kernels.Kernel {
    return ggml.cuda.kernels.get("quantize", ggml.cuda.kernels.tmpl("quantize_mmq_q8_1", &.{ std.fmt.comptimePrint("(mmq_q8_1_ds_layout){d}", .{@backingInt(layout)}), ggml.cuda.kernels.boolean(scatter) }));
}

/// Ports `quantize_mmq_q8_1_cuda` (quantize.cu:574 @b1b770d90).
pub fn quantizeMmqQ8_1(x: ?*const f32, ids: ?*const i32, vy: ?*anyopaque, type_src0: ggml.impl.ggml_c.enum_ggml_type, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    ggml.impl.assert(@rem(ne00, 4) == 0, "ne00 % 4 == 0");
    ggml.impl.assert(@rem(ne0, ggml.cuda.mmq.qk8_1_mmq) == 0, "ne0 % QK8_1_MMQ == 0");

    // ne1 tends to assume the highest values, therefore use it as the "x"
    // dimension of the CUDA grid:
    const block_num_y = @divTrunc(ne0 + 4 * block_size_mmq - 1, 4 * block_size_mmq);
    const p: ggml.cuda.kernels.LaunchParams = .{
        .grid = .{ .x = u32Of(ne1), .y = u32Of(block_num_y), .z = u32Of(ne2 * ne3) },
        .block = .{ .x = block_size_mmq },
        .stream = stream,
    };
    const args = .{ x, ids, vy, ne00, s01, s02, s03, ne0, @as(c_int, @truncate(ne1)), @as(c_int, @truncate(ne2)), @as(c_int, 0) };
    switch (ggml.cuda.mmq.getQ8_1DsLayout(type_src0)) {
        inline else => |layout| ggml.cuda.kernels.launch(q8_1MmqKernel(layout, false), p, args),
    }
}

fn mmqFewQ8_1Kernel(comptime layout: ggml.cuda.mmq.Q8_1DsLayout) ggml.cuda.kernels.Kernel {
    return ggml.cuda.kernels.get("quantize", ggml.cuda.kernels.tmpl("quantize_mmq_few_q8_1", &.{std.fmt.comptimePrint("(mmq_q8_1_ds_layout){d}", .{@backingInt(layout)})}));
}

/// Ports `quantize_mmq_few_q8_1_cuda` (quantize.cu:655 @b1b770d90):
/// `src1` of few-row MMQ in the layout of the MMA fragments, one
/// thread per 4 values of a column, the columns padded to whole groups.
pub fn quantizeMmqFewQ8_1(x: ?*const f32, vy: ?*anyopaque, ds_layout: ggml.cuda.mmq.Q8_1DsLayout, ne00: i64, s01: i64, s02: i64, s03: i64, ne1: i64, ne2: i64, ne3: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    ggml.impl.assert(@rem(ne00, ggml.impl.ggml_c.QK_K) == 0, "ne00 % QK_K == 0");

    const block_num_y = @divTrunc(ne00 + 4 * block_size_mmq - 1, 4 * block_size_mmq);
    const p: ggml.cuda.kernels.LaunchParams = .{
        .grid = .{ .x = u32Of(@as(i64, @intCast(stdlib.num.pad(usize, @intCast(ne1), mmq_few_ncols)))), .y = u32Of(block_num_y), .z = u32Of(ne2 * ne3) },
        .block = .{ .x = block_size_mmq },
        .stream = stream,
    };
    const args = .{ x, vy, ne00, s01, s02, s03, @as(c_int, @truncate(ne1)), @as(c_int, @truncate(ne2)) };
    switch (ds_layout) {
        .d4 => ggml.cuda.kernels.launchPdl(comptime mmqFewQ8_1Kernel(.d4), p, args),
        .ds4 => ggml.cuda.kernels.launchPdl(comptime mmqFewQ8_1Kernel(.ds4), p, args),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `quantize_scatter_mmq_q8_1_cuda` (quantize.cu:679 @b1b770d90).
pub fn quantizeScatterMmqQ8_1(x: ?*const f32, ids_src1_inv: ?*const i32, vy: ?*anyopaque, type_src0: ggml.impl.ggml_c.enum_ggml_type, ne00: i64, stride_token: i64, ne0: i64, n_tokens: i64, nrows_dst: i64, n_expert_used: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    ggml.impl.assert(@rem(ne00, 4) == 0, "ne00 % 4 == 0");
    ggml.impl.assert(@rem(ne0, ggml.cuda.mmq.qk8_1_mmq) == 0, "ne0 % QK8_1_MMQ == 0");

    const block_num_y = @divTrunc(ne0 + 4 * block_size_mmq - 1, 4 * block_size_mmq);
    const p: ggml.cuda.kernels.LaunchParams = .{
        .grid = .{ .x = u32Of(n_tokens), .y = u32Of(block_num_y), .z = 1 },
        .block = .{ .x = block_size_mmq },
        .stream = stream,
    };
    const args = .{ x, ids_src1_inv, vy, ne00, @as(i64, 0), stride_token, @as(i64, 0), ne0, @as(c_int, @truncate(nrows_dst)), @as(c_int, 1), n_expert_used };
    switch (ggml.cuda.mmq.getQ8_1DsLayout(type_src0)) {
        inline else => |layout| ggml.cuda.kernels.launch(q8_1MmqKernel(layout, true), p, args),
    }
}

/// Ports `quantize_scatter_mmq_fp4_cuda` (quantize.cu:708 @b1b770d90).
pub fn quantizeScatterMmqFp4(x: ?*const f32, ids_src1_inv: ?*const i32, vy: ?*anyopaque, scale: ?*f32, type_src0: ggml.impl.ggml_c.enum_ggml_type, use_aligned_float8: bool, ne00: i64, stride_token: i64, ne0: i64, n_tokens: i64, nrows_dst: i64, n_expert_used: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    ggml.impl.assert(ne0 > 0, "ne0 > 0");
    if (type_src0 == ggml.impl.ggml_c.GGML_TYPE_NVFP4) {
        ggml.impl.assert(scale != null, "scale");
        ggml.impl.assert(@rem(ne00, ggml.impl.ggml_c.QK_NVFP4) == 0, "ne00 % QK_NVFP4 == 0");
        const p: ggml.cuda.kernels.LaunchParams = .{ .grid = .{ .x = u32Of(n_tokens) }, .block = .{ .x = block_size_mmq }, .stream = stream };
        const args = .{ x, ids_src1_inv, vy, scale, ne00, @as(i64, 0), stride_token, @as(i64, 0), ne0, nrows_dst, @as(i64, 1), n_expert_used };
        if (use_aligned_float8) {
            ggml.cuda.kernels.launch(ggml.cuda.kernels.get("quantize", "quantize_mmq_nvfp4<(bool)1, (bool)1>"), p, args);
        } else {
            ggml.cuda.kernels.launch(ggml.cuda.kernels.get("quantize", "quantize_mmq_nvfp4<(bool)1, (bool)0>"), p, args);
        }
    } else {
        ggml.impl.assert(type_src0 == ggml.impl.ggml_c.GGML_TYPE_MXFP4, "type_src0 == GGML_TYPE_MXFP4");
        const nwarps = 8;
        const vals_per_block = nwarps * 2 * ggml.impl.ggml_c.QK_MXFP4;
        const block_num_y = @divTrunc(ne0 + vals_per_block - 1, vals_per_block);
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("quantize", "quantize_mmq_mxfp4<(bool)1>"), .{
            .grid = .{ .x = u32Of(n_tokens), .y = u32Of(block_num_y), .z = 1 },
            .block = .{ .x = ggml.cuda.common.warp_size, .y = nwarps },
            .stream = stream,
        }, .{ x, ids_src1_inv, vy, ne00, @as(i64, 0), stride_token, @as(i64, 0), ne0, @as(c_int, @truncate(nrows_dst)), @as(c_int, 1), n_expert_used });
    }
}

/// Ports `quantize_mmq_fp4_cuda` (quantize.cu:737 @b1b770d90).
pub fn quantizeMmqFp4(x: ?*const f32, ids: ?*const i32, vy: ?*anyopaque, scale: ?*f32, type_src0: ggml.impl.ggml_c.enum_ggml_type, use_aligned_float8: bool, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    ggml.impl.assert(type_src0 == ggml.impl.ggml_c.GGML_TYPE_MXFP4 or type_src0 == ggml.impl.ggml_c.GGML_TYPE_NVFP4, "type_src0 == GGML_TYPE_MXFP4 || type_src0 == GGML_TYPE_NVFP4");
    ggml.impl.assert(ne0 > 0, "ne0 > 0");

    if (type_src0 == ggml.impl.ggml_c.GGML_TYPE_NVFP4) {
        ggml.impl.assert(scale != null, "scale");
        ggml.impl.assert(@rem(ne00, ggml.impl.ggml_c.QK_NVFP4) == 0, "ne00 % QK_NVFP4 == 0");
        const p: ggml.cuda.kernels.LaunchParams = .{ .grid = .{ .x = u32Of(ne1), .y = u32Of(ne2 * ne3) }, .block = .{ .x = block_size_mmq }, .stream = stream };
        const args = .{ x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, @as(c_int, 0) };
        if (use_aligned_float8) {
            ggml.cuda.kernels.launch(ggml.cuda.kernels.get("quantize", "quantize_mmq_nvfp4<(bool)0, (bool)1>"), p, args);
        } else {
            ggml.cuda.kernels.launch(ggml.cuda.kernels.get("quantize", "quantize_mmq_nvfp4<(bool)0, (bool)0>"), p, args);
        }
    } else {
        ggml.impl.assert(@rem(ne0, 2 * ggml.impl.ggml_c.QK_MXFP4) == 0, "ne0 % (2 * QK_MXFP4) == 0");

        const nwarps = 8;
        const vals_per_warp = 2 * ggml.impl.ggml_c.QK_MXFP4;
        const vals_per_block = nwarps * vals_per_warp;

        const block_num_y = @divTrunc(ne0 + vals_per_block - 1, vals_per_block);
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("quantize", "quantize_mmq_mxfp4<(bool)0>"), .{
            .grid = .{ .x = u32Of(ne1), .y = u32Of(block_num_y), .z = u32Of(ne2 * ne3) },
            .block = .{ .x = ggml.cuda.common.warp_size, .y = nwarps },
            .stream = stream,
        }, .{ x, ids, vy, ne00, s01, s02, s03, ne0, @as(c_int, @truncate(ne1)), @as(c_int, @truncate(ne2)), @as(c_int, 0) });
    }
}
