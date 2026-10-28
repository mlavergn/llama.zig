//! `GGML_OP_CONV_TRANSPOSE_1D`: transposed 1-D convolution.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/conv-transpose-1d.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/conv-transpose-1d.cuh` at v0.4.0
//! (`5266f24da`). The kernel, `conv_transpose_1d_kernel`, stays CUDA.
//!
//! Only the stride is read from `op_params`; padding and dilation are the
//! C's hard-coded 0 and 1.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE` (conv-transpose-1d.cuh:3 @5266f24da).
const block_size = 256;

/// An `int64_t` into an `int` parameter.
fn int(x: i64) c_int {
    return @truncate(x);
}

/// Ports `conv_transpose_1d_f32_f32_cuda` (conv-transpose-1d.cu:42 @5266f24da).
fn convTranspose1dF32F32Cuda(
    s0: c_int,
    p0: c_int,
    d0: c_int,
    output_size: c_int,
    src0_ne0: c_int,
    src0_ne1: c_int,
    src0_ne2: c_int,
    src0_ne3: c_int,
    src1_ne0: c_int,
    src1_ne1: c_int,
    src1_ne2: c_int,
    src1_ne3: c_int,
    dst_ne0: c_int,
    dst_ne1: c_int,
    dst_ne2: c_int,
    dst_ne3: c_int,
    src0: ?*const anyopaque,
    src1: ?*const anyopaque,
    dst: ?*anyopaque,
    stream: cu.cudaStream_t,
) void {
    const num_blocks: c_int = @divTrunc(output_size +% (block_size - 1), block_size);
    kernels.launch(kernels.get("conv-transpose-1d", "conv_transpose_1d_kernel"), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = stream,
    }, .{
        s0,       p0,       d0,       output_size,
        src0_ne0, src0_ne1, src0_ne2, src0_ne3,
        src1_ne0, src1_ne1, src1_ne2, src1_ne3,
        dst_ne0,  dst_ne1,  dst_ne2,  dst_ne3,
        src0,     src1,     dst,
    });
}

/// Ports `ggml_cuda_op_conv_transpose_1d` (conv-transpose-1d.cu:59 @5266f24da).
pub fn opConvTranspose1d(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;

    const src1 = impl.one(Tensor, dst.src[1]);
    const src1_d: ?*const anyopaque = src1.data;

    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");

    const s0 = impl.getOpParamsI32(dst, 0);
    const p0: c_int = 0; //opts[3];
    const d0: c_int = 1; //opts[4];

    const output_size = c.ggml_nelements(dst);

    convTranspose1dF32F32Cuda(s0, p0, d0, int(output_size), int(src0.ne[0]), int(src0.ne[1]), int(src0.ne[2]), int(src0.ne[3]), int(src1.ne[0]), int(src1.ne[1]), int(src1.ne[2]), int(src1.ne[3]), int(dst.ne[0]), int(dst.ne[1]), int(dst.ne[2]), int(dst.ne[3]), src0_d, src1_d, dst_d, stream);
}
