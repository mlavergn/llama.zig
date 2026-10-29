//! Quantizing `src1` for the integer matrix kernels: `q8_1` rows for MMVQ,
//! the `block_q8_1_mmq` and FP4 layouts for MMQ.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/quantize.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/quantize.cuh` at v0.4.1 (`b29c606e2`). The
//! `quantize_*` kernels stay CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const mmq = @import("mmq.zig");

const c = impl.c;
const cu = common.cu;

/// Ports `CUDA_QUANTIZE_BLOCK_SIZE` (quantize.cuh:8 @b29c606e2).
pub const block_size = 256;

/// Ports `CUDA_QUANTIZE_BLOCK_SIZE_MMQ` (quantize.cuh:9 @b29c606e2).
pub const block_size_mmq = 128;

/// Ports `quantize_cuda_t` (quantize.cuh:14 @b29c606e2).
pub const QuantizeFn = *const fn (x: ?*const f32, ids: ?*const i32, vy: ?*anyopaque, type_src0: c.enum_ggml_type, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: cu.cudaStream_t) void;

fn u32Of(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `quantize_row_q8_1_cuda` (quantize.cu:558 @b29c606e2).
pub fn quantizeRowQ8_1(x: ?*const f32, ids: ?*const i32, vy: ?*anyopaque, type_src0: c.enum_ggml_type, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: cu.cudaStream_t) void {
    _ = type_src0;
    impl.assert(ids == null, "!ids");
    impl.assert(@rem(ne0, c.QK8_1) == 0, "ne0 % QK8_1 == 0");

    const ne2_fastdiv = common.initFastdivValues(@bitCast(ne2));

    const block_num_x = @divTrunc(ne0 + block_size - 1, block_size);
    kernels.launchPdl(kernels.get("quantize", "quantize_q8_1"), .{
        .grid = .{ .x = u32Of(block_num_x), .y = u32Of(ne1), .z = u32Of(ne2 * ne3) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, vy, ne00, s01, s02, s03, ne0, u32Of(ne1), ne2_fastdiv });
}

fn q8_1MmqKernel(comptime layout: mmq.Q8_1DsLayout, comptime scatter: bool) kernels.Kernel {
    return kernels.get("quantize", kernels.tmpl("quantize_mmq_q8_1", &.{ std.fmt.comptimePrint("(mmq_q8_1_ds_layout){d}", .{@backingInt(layout)}), kernels.boolean(scatter) }));
}

/// Ports `quantize_mmq_q8_1_cuda` (quantize.cu:575 @b29c606e2).
pub fn quantizeMmqQ8_1(x: ?*const f32, ids: ?*const i32, vy: ?*anyopaque, type_src0: c.enum_ggml_type, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: cu.cudaStream_t) void {
    impl.assert(@rem(ne00, 4) == 0, "ne00 % 4 == 0");
    impl.assert(@rem(ne0, mmq.qk8_1_mmq) == 0, "ne0 % QK8_1_MMQ == 0");

    // ne1 tends to assume the highest values, therefore use it as the "x"
    // dimension of the CUDA grid:
    const block_num_y = @divTrunc(ne0 + 4 * block_size_mmq - 1, 4 * block_size_mmq);
    const p: kernels.LaunchParams = .{
        .grid = .{ .x = u32Of(ne1), .y = u32Of(block_num_y), .z = u32Of(ne2 * ne3) },
        .block = .{ .x = block_size_mmq },
        .stream = stream,
    };
    const args = .{ x, ids, vy, ne00, s01, s02, s03, ne0, @as(c_int, @truncate(ne1)), @as(c_int, @truncate(ne2)), @as(c_int, 0) };
    switch (mmq.getQ8_1DsLayout(type_src0)) {
        inline else => |layout| kernels.launch(q8_1MmqKernel(layout, false), p, args),
    }
}

/// Ports `quantize_scatter_mmq_q8_1_cuda` (quantize.cu:606 @b29c606e2).
pub fn quantizeScatterMmqQ8_1(x: ?*const f32, ids_src1_inv: ?*const i32, vy: ?*anyopaque, type_src0: c.enum_ggml_type, ne00: i64, stride_token: i64, ne0: i64, n_tokens: i64, nrows_dst: i64, n_expert_used: c_int, stream: cu.cudaStream_t) void {
    impl.assert(@rem(ne00, 4) == 0, "ne00 % 4 == 0");
    impl.assert(@rem(ne0, mmq.qk8_1_mmq) == 0, "ne0 % QK8_1_MMQ == 0");

    const block_num_y = @divTrunc(ne0 + 4 * block_size_mmq - 1, 4 * block_size_mmq);
    const p: kernels.LaunchParams = .{
        .grid = .{ .x = u32Of(n_tokens), .y = u32Of(block_num_y), .z = 1 },
        .block = .{ .x = block_size_mmq },
        .stream = stream,
    };
    const args = .{ x, ids_src1_inv, vy, ne00, @as(i64, 0), stride_token, @as(i64, 0), ne0, @as(c_int, @truncate(nrows_dst)), @as(c_int, 1), n_expert_used };
    switch (mmq.getQ8_1DsLayout(type_src0)) {
        inline else => |layout| kernels.launch(q8_1MmqKernel(layout, true), p, args),
    }
}

/// Ports `quantize_scatter_mmq_fp4_cuda` (quantize.cu:636 @b29c606e2).
pub fn quantizeScatterMmqFp4(x: ?*const f32, ids_src1_inv: ?*const i32, vy: ?*anyopaque, scale: ?*f32, type_src0: c.enum_ggml_type, use_aligned_float8: bool, ne00: i64, stride_token: i64, ne0: i64, n_tokens: i64, nrows_dst: i64, n_expert_used: c_int, stream: cu.cudaStream_t) void {
    impl.assert(ne0 > 0, "ne0 > 0");
    if (type_src0 == c.GGML_TYPE_NVFP4) {
        impl.assert(scale != null, "scale");
        impl.assert(@rem(ne00, c.QK_NVFP4) == 0, "ne00 % QK_NVFP4 == 0");
        const p: kernels.LaunchParams = .{ .grid = .{ .x = u32Of(n_tokens) }, .block = .{ .x = block_size_mmq }, .stream = stream };
        const args = .{ x, ids_src1_inv, vy, scale, ne00, @as(i64, 0), stride_token, @as(i64, 0), ne0, nrows_dst, @as(i64, 1), n_expert_used };
        if (use_aligned_float8) {
            kernels.launch(kernels.get("quantize", "quantize_mmq_nvfp4<(bool)1, (bool)1>"), p, args);
        } else {
            kernels.launch(kernels.get("quantize", "quantize_mmq_nvfp4<(bool)1, (bool)0>"), p, args);
        }
    } else {
        impl.assert(type_src0 == c.GGML_TYPE_MXFP4, "type_src0 == GGML_TYPE_MXFP4");
        const nwarps = 8;
        const vals_per_block = nwarps * 2 * c.QK_MXFP4;
        const block_num_y = @divTrunc(ne0 + vals_per_block - 1, vals_per_block);
        kernels.launch(kernels.get("quantize", "quantize_mmq_mxfp4<(bool)1>"), .{
            .grid = .{ .x = u32Of(n_tokens), .y = u32Of(block_num_y), .z = 1 },
            .block = .{ .x = common.warp_size, .y = nwarps },
            .stream = stream,
        }, .{ x, ids_src1_inv, vy, ne00, @as(i64, 0), stride_token, @as(i64, 0), ne0, @as(c_int, @truncate(nrows_dst)), @as(c_int, 1), n_expert_used });
    }
}

/// Ports `quantize_mmq_fp4_cuda` (quantize.cu:665 @b29c606e2).
pub fn quantizeMmqFp4(x: ?*const f32, ids: ?*const i32, vy: ?*anyopaque, scale: ?*f32, type_src0: c.enum_ggml_type, use_aligned_float8: bool, ne00: i64, s01: i64, s02: i64, s03: i64, ne0: i64, ne1: i64, ne2: i64, ne3: i64, stream: cu.cudaStream_t) void {
    impl.assert(type_src0 == c.GGML_TYPE_MXFP4 or type_src0 == c.GGML_TYPE_NVFP4, "type_src0 == GGML_TYPE_MXFP4 || type_src0 == GGML_TYPE_NVFP4");
    impl.assert(ne0 > 0, "ne0 > 0");

    if (type_src0 == c.GGML_TYPE_NVFP4) {
        impl.assert(scale != null, "scale");
        impl.assert(@rem(ne00, c.QK_NVFP4) == 0, "ne00 % QK_NVFP4 == 0");
        const p: kernels.LaunchParams = .{ .grid = .{ .x = u32Of(ne1), .y = u32Of(ne2 * ne3) }, .block = .{ .x = block_size_mmq }, .stream = stream };
        const args = .{ x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, @as(c_int, 0) };
        if (use_aligned_float8) {
            kernels.launch(kernels.get("quantize", "quantize_mmq_nvfp4<(bool)0, (bool)1>"), p, args);
        } else {
            kernels.launch(kernels.get("quantize", "quantize_mmq_nvfp4<(bool)0, (bool)0>"), p, args);
        }
    } else {
        impl.assert(@rem(ne0, 2 * c.QK_MXFP4) == 0, "ne0 % (2 * QK_MXFP4) == 0");

        const nwarps = 8;
        const vals_per_warp = 2 * c.QK_MXFP4;
        const vals_per_block = nwarps * vals_per_warp;

        const block_num_y = @divTrunc(ne0 + vals_per_block - 1, vals_per_block);
        kernels.launch(kernels.get("quantize", "quantize_mmq_mxfp4<(bool)0>"), .{
            .grid = .{ .x = u32Of(ne1), .y = u32Of(block_num_y), .z = u32Of(ne2 * ne3) },
            .block = .{ .x = common.warp_size, .y = nwarps },
            .stream = stream,
        }, .{ x, ids, vy, ne00, s01, s02, s03, ne0, @as(c_int, @truncate(ne1)), @as(c_int, @truncate(ne2)), @as(c_int, 0) });
    }
}
