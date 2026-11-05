//! `GGML_OP_SCALE`: `dst = scale * x + bias`, in F32 or (since v0.6.0) BF16.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/scale.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/scale.cuh` at v0.7.0 (`620d2225c`). The
//! kernel, `scale_f32`, stays CUDA and is launched from here; since v0.6.0
//! it is a template over the element type, and computes in F32 either way.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_SCALE_BLOCK_SIZE` (scale.cuh:3 @620d2225c).
const block_size = 256;

/// Ports `MAX_GRIDDIM_X` (scale.cu:4 @620d2225c).
const max_griddim_x = 0x7FFFFFFF;

/// Ports `scale_f32_cuda` (scale.cu:19 @620d2225c), its element type `T`
/// comptime: `f32` or `kernels.bf16`.
fn scaleF32Cuda(comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, scale: f32, bias: f32, nelements: i64, stream: common.cu.cudaStream_t) void {
    const num_blocks = @divTrunc(nelements + block_size - 1, block_size);
    kernels.launchPdl(kernels.get("scale", kernels.tmpl("scale_f32", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = @intCast(@min(max_griddim_x, num_blocks)) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, dst, scale, bias, nelements });
}

/// Ports `ggml_cuda_op_scale` (scale.cu:25 @620d2225c).
pub fn opScale(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_BF16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16");
    impl.assert(dst.type == src0.type, "dst->type == src0->type");

    const scale = impl.getOpParamsF32(dst, 0);
    const bias = impl.getOpParamsF32(dst, 1);

    if (src0.type == c.GGML_TYPE_BF16) {
        scaleF32Cuda(kernels.bf16, src0.data, dst.data, scale, bias, c.ggml_nelements(src0), stream);
    } else {
        scaleF32Cuda(f32, src0.data, dst.data, scale, bias, c.ggml_nelements(src0), stream);
    }
}
