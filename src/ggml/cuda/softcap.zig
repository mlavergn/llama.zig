//! The fused `SCALE` + `TANH` + `SCALE` chain: `dst = tanh(scale * x) *
//! softcap`. Reached only through `ggml_cuda_try_fuse`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/softcap.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/softcap.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `softcap_f32`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_SOFTCAP_BLOCK_SIZE` (softcap.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `softcap_f32_cuda` (softcap.cu:15 @b1b770d90).
fn softcapF32Cuda(x: ?*const anyopaque, dst: ?*anyopaque, scale: f32, softcap: f32, k: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(k + block_size - 1, block_size);
    ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("softcap", "softcap_f32"), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, dst, scale, softcap, k });
}

/// Ports `ggml_cuda_op_softcap` (softcap.cu:22 @b1b770d90): `src` is the
/// first `SCALE`, `dst` the last.
pub fn opSoftcap(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor, src: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, src.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const scale: f32 = @bitCast(src.op_params[0]);
    const softcap: f32 = @bitCast(dst.op_params[0]);

    softcapF32Cuda(src0_d, dst_d, scale, softcap, @truncate(ggml.types.ggml_nelements(src0)), stream);
}
