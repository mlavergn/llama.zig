//! The fused `SCALE` + `TANH` + `SCALE` chain: `dst = tanh(scale * x) *
//! softcap`. Reached only through `ggml_cuda_try_fuse`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/softcap.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/softcap.cuh` at v0.4.1 (`b29c606e2`). The
//! kernel, `softcap_f32`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_SOFTCAP_BLOCK_SIZE` (softcap.cuh:3 @b29c606e2).
const block_size = 256;

/// Ports `softcap_f32_cuda` (softcap.cu:15 @b29c606e2).
fn softcapF32Cuda(x: ?*const anyopaque, dst: ?*anyopaque, scale: f32, softcap: f32, k: c_int, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(k + block_size - 1, block_size);
    kernels.launchPdl(kernels.get("softcap", "softcap_f32"), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, dst, scale, softcap, k });
}

/// Ports `ggml_cuda_op_softcap` (softcap.cu:22 @b29c606e2): `src` is the
/// first `SCALE`, `dst` the last.
pub fn opSoftcap(ctx: *common.Context, dst: *Tensor, src: *Tensor) void {
    const src0 = impl.one(Tensor, src.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const scale: f32 = @bitCast(src.op_params[0]);
    const softcap: f32 = @bitCast(dst.op_params[0]);

    softcapF32Cuda(src0_d, dst_d, scale, softcap, @truncate(c.ggml_nelements(src0)), stream);
}

comptime {
    // No caller until `ggml_cuda_try_fuse` is ported; referencing it keeps it
    // compiled and type-checked.
    _ = &opSoftcap;
}
