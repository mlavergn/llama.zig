//! `GGML_OP_POOL_2D`: 2-D max or average pooling.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/pool2d.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/pool2d.cuh` at v0.4.0 (`5266f24da`). The
//! kernel, `pool2d_nchw_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_POOL2D_BLOCK_SIZE` (pool2d.cuh:3 @5266f24da).
const block_size = 256;

/// An `int64_t` into an `int` parameter.
fn int(x: i64) c_int {
    return @truncate(x);
}

/// Ports `pool2d_nchw_kernel_f32_f32_cuda` (pool2d.cu:53 @5266f24da).
fn pool2dNchwKernelF32F32Cuda(ih: c_int, iw: c_int, oh: c_int, ow: c_int, kh: c_int, kw: c_int, sh: c_int, sw: c_int, ph: c_int, pw: c_int, parallel_elements: c_int, src: ?*const anyopaque, dst: ?*anyopaque, op: c.enum_ggml_op_pool, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(parallel_elements +% (block_size - 1), block_size);
    kernels.launch(kernels.get("pool2d", "pool2d_nchw_kernel<float, float>"), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = stream,
    }, .{ ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op });
}

/// Ports `ggml_cuda_op_pool2d` (pool2d.cu:65 @5266f24da).
pub fn opPool2d(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const op: c.enum_ggml_op_pool = @bitCast(impl.getOpParamsI32(dst, 0));
    const k0 = impl.getOpParamsI32(dst, 1);
    const k1 = impl.getOpParamsI32(dst, 2);
    const s0 = impl.getOpParamsI32(dst, 3);
    const s1 = impl.getOpParamsI32(dst, 4);
    const p0 = impl.getOpParamsI32(dst, 5);
    const p1 = impl.getOpParamsI32(dst, 6);

    const IH = src0.ne[1];
    const IW = src0.ne[0];

    const N = dst.ne[3];
    const OC = dst.ne[2];
    const OH = dst.ne[1];
    const OW = dst.ne[0];

    const parallel_elements = int(N * OC * OH * OW);

    pool2dNchwKernelF32F32Cuda(int(IH), int(IW), int(OH), int(OW), k1, k0, s1, s0, p1, p0, parallel_elements, src0_d, dst_d, op, stream);
}
