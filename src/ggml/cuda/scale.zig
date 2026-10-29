//! `GGML_OP_SCALE`: `dst = scale * x + bias`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/scale.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/scale.cuh` at v0.4.1 (`b29c606e2`). The
//! kernel, `scale_f32`, stays CUDA and is launched from here.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_SCALE_BLOCK_SIZE` (scale.cuh:3 @b29c606e2).
const block_size = 256;

/// Ports `MAX_GRIDDIM_X` (scale.cu:3 @b29c606e2).
const max_griddim_x = 0x7FFFFFFF;

/// Ports `scale_f32_cuda` (scale.cu:16 @b29c606e2).
fn scaleF32Cuda(x: [*c]const f32, dst: [*c]f32, scale: f32, bias: f32, nelements: i64, stream: common.cu.cudaStream_t) void {
    const num_blocks = @divTrunc(nelements + block_size - 1, block_size);
    kernels.launchPdl(kernels.get("scale", "scale_f32"), .{
        .grid = .{ .x = @intCast(@min(max_griddim_x, num_blocks)) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, dst, scale, bias, nelements });
}

/// Ports `ggml_cuda_op_scale` (scale.cu:22 @b29c606e2).
pub fn opScale(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const scale = impl.getOpParamsF32(dst, 0);
    const bias = impl.getOpParamsF32(dst, 1);

    scaleF32Cuda(@ptrCast(@alignCast(src0.data)), @ptrCast(@alignCast(dst.data)), scale, bias, c.ggml_nelements(src0), ctx.stream());
}
