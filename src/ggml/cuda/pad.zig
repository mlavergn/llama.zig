//! `GGML_OP_PAD`: zero padding on every side of every dimension, or
//! circular (torus) padding.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/pad.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/pad.cuh` at v0.4.1 (`b29c606e2`). The
//! kernel, `pad_f32`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_PAD_BLOCK_SIZE` (pad.cuh:3 @b29c606e2).
const block_size = 256;

/// Ports `pad_f32_cuda` (pad.cu:64 @b29c606e2), which launches with
/// `<<<>>>`.
fn padF32Cuda(
    src: ?*const anyopaque,
    s00: usize,
    s01: usize,
    s02: usize,
    s03: usize,
    dst: ?*anyopaque,
    lp0: c_int,
    rp0: c_int,
    lp1: c_int,
    rp1: c_int,
    lp2: c_int,
    rp2: c_int,
    lp3: c_int,
    rp3: c_int,
    ne0: c_int,
    ne1: c_int,
    ne2: c_int,
    ne3: c_int,
    circular: bool,
    stream: cu.cudaStream_t,
) void {
    const num_blocks: c_int = @divTrunc(ne0 + block_size - 1, block_size);
    const grid: kernels.Dim3 = .{ .x = @bitCast(num_blocks), .y = @bitCast(ne1), .z = @bitCast(ne2 *% ne3) };
    kernels.launch(kernels.get("pad", "pad_f32"), .{ .grid = grid, .block = .{ .x = block_size }, .stream = stream }, .{
        src,      s00, s01, s02, s03, dst,
        lp0,      rp0, lp1, rp1, lp2, rp2,
        lp3,      rp3, ne0, ne1, ne2, ne3,
        circular,
    });
}

/// Ports `ggml_cuda_op_pad` (pad.cu:76 @b29c606e2).
pub fn opPad(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    // GGML_TENSOR_UNARY_OP_LOCALS
    const nb00 = src0.nb[0];
    const nb01 = src0.nb[1];
    const nb02 = src0.nb[2];
    const nb03 = src0.nb[3];

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const lp0: i32 = dst.op_params[0];
    const rp0: i32 = dst.op_params[1];
    const lp1: i32 = dst.op_params[2];
    const rp1: i32 = dst.op_params[3];
    const lp2: i32 = dst.op_params[4];
    const rp2: i32 = dst.op_params[5];
    const lp3: i32 = dst.op_params[6];
    const rp3: i32 = dst.op_params[7];
    const circular: i32 = dst.op_params[8];

    const ts = c.ggml_type_size(src0.type);
    const s00: usize = nb00 / ts;
    const s01: usize = nb01 / ts;
    const s02: usize = nb02 / ts;
    const s03: usize = nb03 / ts;

    padF32Cuda(src0_d, s00, s01, s02, s03, dst_d, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, @truncate(dst.ne[0]), @truncate(dst.ne[1]), @truncate(dst.ne[2]), @truncate(dst.ne[3]), circular != 0, stream);
}
