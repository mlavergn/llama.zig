//! `GGML_OP_CONV_2D_DW`: depthwise 2-D convolution, in WHCN or CWHN layout.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/conv2d-dw.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/conv2d-dw.cuh` at v0.4.1 (`b29c606e2`).
//! The kernel, `conv2d_dw_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_CONV2D_DW_BLOCK_SIZE` (conv2d-dw.cuh:4 @b29c606e2).
const block_size = 256;

/// Ports `ggml_cuda_op_conv2d_dw` (conv2d-dw.cu:119 @b29c606e2).
pub fn opConv2dDw(ctx: *common.Context, dst: *Tensor) void {
    const kernel = impl.one(Tensor, dst.src[0]);
    const input = impl.one(Tensor, dst.src[1]);

    impl.assert(kernel.type == c.GGML_TYPE_F32 and input.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32, "kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32");
    const w_d: ?*const anyopaque = kernel.data;
    const x_d: ?*const anyopaque = input.data;
    const y_d: ?*anyopaque = dst.data;

    const stride_x = impl.getOpParamsI32(dst, 0);
    const stride_y = impl.getOpParamsI32(dst, 1);
    const padding_x = impl.getOpParamsI32(dst, 2);
    const padding_y = impl.getOpParamsI32(dst, 3);
    const dilation_x = impl.getOpParamsI32(dst, 4);
    const dilation_y = impl.getOpParamsI32(dst, 5);

    const in_w: c_int = @truncate(input.ne[0]);
    const in_h: c_int = @truncate(input.ne[1]);
    const kernel_w: c_int = @truncate(kernel.ne[0]);
    const kernel_h: c_int = @truncate(kernel.ne[1]);
    const out_w: c_int = @truncate(dst.ne[0]);
    const out_h: c_int = @truncate(dst.ne[1]);
    const channels: c_int = @truncate(dst.ne[2]);
    const batches: c_int = @truncate(dst.ne[3]);

    const st = ctx.stream();

    const total: c_int = batches *% channels *% out_h *% out_w;
    const blocks: c_int = @divTrunc(total +% (block_size - 1), block_size);

    const p: kernels.LaunchParams = .{ .grid = .{ .x = @bitCast(blocks) }, .block = .{ .x = block_size }, .shmem = 0, .stream = st };
    const args = .{ x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches };
    if (c.ggml_is_contiguous(input)) {
        kernels.launch(kernels.get("conv2d-dw", "conv2d_dw_kernel<float, whcn_layout>"), p, args);
    } else if (c.ggml_is_contiguous_channels(input)) {
        kernels.launch(kernels.get("conv2d-dw", "conv2d_dw_kernel<float, cwhn_layout>"), p, args);
    } else {
        impl.abort("Unsupported memory layout for conv_2d_dw");
    }
}
