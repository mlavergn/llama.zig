//! `GGML_OP_TIMESTEP_EMBEDDING`: sinusoidal embeddings of diffusion
//! timesteps.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/tsembd.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/tsembd.cuh` at v0.4.0 (`5266f24da`). The
//! kernel, `timestep_embedding_f32`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE` (tsembd.cuh:3 @5266f24da).
const block_size = 256;

/// Ports `timestep_embedding_f32_cuda` (tsembd.cu:26 @5266f24da), which
/// launches with `<<<>>>`.
fn timestepEmbeddingF32Cuda(x: ?*const anyopaque, dst: ?*anyopaque, ne00: c_int, nb1: c_int, dim: c_int, max_period: c_int, stream: cu.cudaStream_t) void {
    const half_ceil: c_int = @divTrunc(dim + 1, 2);
    const num_blocks: c_int = @divTrunc(half_ceil + block_size - 1, block_size);
    const grid: kernels.Dim3 = .{ .x = @bitCast(num_blocks), .y = @bitCast(ne00), .z = 1 };
    kernels.launch(kernels.get("tsembd", "timestep_embedding_f32"), .{ .grid = grid, .block = .{ .x = block_size }, .stream = stream }, .{ x, dst, nb1, dim, max_period });
}

/// Ports `ggml_cuda_op_timestep_embedding` (tsembd.cu:34 @5266f24da).
pub fn opTimestepEmbedding(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const dim: c_int = dst.op_params[0];
    const max_period: c_int = dst.op_params[1];

    timestepEmbeddingF32Cuda(src0_d, dst_d, @truncate(src0.ne[0]), @truncate(@as(i64, @bitCast(dst.nb[1]))), dim, max_period, stream);
}
