//! `GGML_OP_TIMESTEP_EMBEDDING`: sinusoidal embeddings of diffusion
//! timesteps.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/tsembd.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/tsembd.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `timestep_embedding_f32`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE` (tsembd.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `timestep_embedding_f32_cuda` (tsembd.cu:26 @b1b770d90), which
/// launches with `<<<>>>`.
fn timestepEmbeddingF32Cuda(x: ?*const anyopaque, dst: ?*anyopaque, ne00: c_int, nb1: c_int, dim: c_int, max_period: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const half_ceil: c_int = @divTrunc(dim + 1, 2);
    const num_blocks: c_int = @divTrunc(half_ceil + block_size - 1, block_size);
    const grid: ggml.cuda.kernels.Dim3 = .{ .x = @bitCast(num_blocks), .y = @bitCast(ne00), .z = 1 };
    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("tsembd", "timestep_embedding_f32"), .{ .grid = grid, .block = .{ .x = block_size }, .stream = stream }, .{ x, dst, nb1, dim, max_period });
}

/// Ports `ggml_cuda_op_timestep_embedding` (tsembd.cu:34 @b1b770d90).
pub fn opTimestepEmbedding(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const dim: c_int = dst.op_params[0];
    const max_period: c_int = dst.op_params[1];

    timestepEmbeddingF32Cuda(src0_d, dst_d, @truncate(src0.ne[0]), @truncate(@as(i64, @bitCast(dst.nb[1]))), dim, max_period, stream);
}
