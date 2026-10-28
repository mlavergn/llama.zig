//! `GGML_OP_POOL_1D`: 1-D max or average pooling.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/pool1d.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/pool1d.cuh` at v0.4.0 (`5266f24da`). The
//! kernel, `pool1d_nchw_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_POOL1D_BLOCK_SIZE` (pool1d.cuh:3 @5266f24da).
const block_size = 256;

/// An `int64_t` into an `int` parameter.
fn int(x: i64) c_int {
    return @truncate(x);
}

/// Ports `pool1d_nchw_kernel_f32_f32_cuda` (pool1d.cu:52 @5266f24da).
fn pool1dNchwKernelF32F32Cuda(iw: c_int, ow: c_int, kw: c_int, sw: c_int, pw: c_int, parallel_elements: c_int, src: ?*const anyopaque, dst: ?*anyopaque, op: c.enum_ggml_op_pool, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(parallel_elements +% (block_size - 1), block_size);
    kernels.launch(kernels.get("pool1d", "pool1d_nchw_kernel"), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = stream,
    }, .{ iw, ow, kw, sw, pw, parallel_elements, src, dst, op });
}

/// Ports `ggml_cuda_op_pool1d` (pool1d.cu:63 @5266f24da).
pub fn opPool1d(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const op: c.enum_ggml_op_pool = @bitCast(impl.getOpParamsI32(dst, 0));
    const k0 = impl.getOpParamsI32(dst, 1);
    const s0 = impl.getOpParamsI32(dst, 2);
    const p0 = impl.getOpParamsI32(dst, 3);

    const IW = src0.ne[0];
    const OW = dst.ne[0];
    const nr = c.ggml_nrows(src0);

    const parallel_elements = int(nr * OW);

    pool1dNchwKernelF32F32Cuda(int(IW), int(OW), k0, s0, p0, parallel_elements, src0_d, dst_d, op, stream);
}
