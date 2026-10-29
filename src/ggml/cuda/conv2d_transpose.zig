//! `GGML_OP_CONV_TRANSPOSE_2D`: transposed 2-D convolution with zero
//! padding.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/conv2d-transpose.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/conv2d-transpose.cuh` at v0.4.1
//! (`b29c606e2`). The kernel, `conv2d_transpose_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE` (conv2d-transpose.cuh:3 @b29c606e2).
const block_size = 256;

/// Ports `ggml_cuda_conv_2d_transpose_p0` (conv2d-transpose.cu:71
/// @b29c606e2). Input is (W, H, C_in, N), kernel is (W, H, C_out, C_in).
pub fn conv2dTransposeP0(ctx: *common.Context, dst: *Tensor) void {
    const kernel = impl.one(Tensor, dst.src[0]);
    const input = impl.one(Tensor, dst.src[1]);
    impl.assert(kernel.type == c.GGML_TYPE_F16 or kernel.type == c.GGML_TYPE_F32, "kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32");
    impl.assert(input.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32, "input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32");

    const input_data: ?*const anyopaque = input.data;
    const output_data: ?*anyopaque = dst.data;
    const kernel_data: ?*const anyopaque = kernel.data;

    const input_w: c_int = @truncate(input.ne[0]);
    const input_h: c_int = @truncate(input.ne[1]);
    const output_w: c_int = @truncate(dst.ne[0]);
    const output_h: c_int = @truncate(dst.ne[1]);
    const channels_in: c_int = @truncate(input.ne[2]);
    const channels_out: c_int = @truncate(kernel.ne[2]);
    const kernel_w: c_int = @truncate(kernel.ne[0]);
    const kernel_h: c_int = @truncate(kernel.ne[1]);
    const stride: c_int = dst.op_params[0];
    const batches: c_int = @truncate(input.ne[3]);

    impl.assert(channels_in == kernel.ne[3], "channels_in == kernel->ne[3]");
    impl.assert(stride > 0, "stride > 0");

    const st = ctx.stream();

    impl.assert(c.ggml_is_contiguous(input), "ggml_is_contiguous(input)");
    impl.assert(c.ggml_is_contiguous(kernel), "ggml_is_contiguous(kernel)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const total: c_int = output_w *% output_h *% channels_out *% batches;
    const blocks: c_int = @divTrunc(total +% (block_size - 1), block_size);

    const p: kernels.LaunchParams = .{ .grid = .{ .x = @bitCast(blocks) }, .block = .{ .x = block_size }, .shmem = 0, .stream = st };
    const args = .{ input_data, kernel_data, output_data, input_w, input_h, output_w, output_h, kernel_w, kernel_h, stride, channels_in, channels_out, batches };
    if (kernel.type == c.GGML_TYPE_F16) {
        kernels.launch(kernels.get("conv2d-transpose", "conv2d_transpose_kernel<__half>"), p, args);
    } else {
        kernels.launch(kernels.get("conv2d-transpose", "conv2d_transpose_kernel<float>"), p, args);
    }
}
