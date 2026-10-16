//! `GGML_OP_ROLL`: a cyclic shift along each dimension.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/roll.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/roll.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernel, `roll_f32_cuda` (a `__global__`, despite the name), stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_ROLL_BLOCK_SIZE` (roll.cuh:3 @c1d0e7a00).
const block_size = 256;

/// Ports `ggml_cuda_op_roll` (roll.cu:45 @c1d0e7a00), which launches with
/// `<<<>>>`.
pub fn opRoll(ctx: *common.Context, dst: *Tensor) void {
    const s0: c_int = dst.op_params[0];
    const s1: c_int = dst.op_params[1];
    const s2: c_int = dst.op_params[2];
    const s3: c_int = dst.op_params[3];

    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;

    // GGML_TENSOR_UNARY_OP_LOCALS
    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];

    impl.assert(src0.type == c.GGML_TYPE_F32, "dst->src[0]->type == GGML_TYPE_F32");
    impl.assert(c.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(dst->src[0], dst)");

    const stream = ctx.stream();

    const sz: i64 = ne00 * ne01 * ne02 * ne03;
    const num_blocks: i64 = @divTrunc(sz + block_size - 1, block_size);

    kernels.launch(kernels.get("roll", "roll_f32_cuda"), .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ src0_d, dst_d, ne00, ne01, ne02, ne03, s0, s1, s2, s3 });
}
