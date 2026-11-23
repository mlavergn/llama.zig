//! `GGML_OP_ROLL`: a cyclic shift along each dimension.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/roll.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/roll.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `roll_f32_cuda` (a `__global__`, despite the name), stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_ROLL_BLOCK_SIZE` (roll.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `ggml_cuda_op_roll` (roll.cu:49 @b1b770d90), which launches with
/// `<<<>>>`. The kernel reads the source through its byte
/// strides, so a non-contiguous view rolls as it is.
pub fn opRoll(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const s0: c_int = dst.op_params[0];
    const s1: c_int = dst.op_params[1];
    const s2: c_int = dst.op_params[2];
    const s3: c_int = dst.op_params[3];

    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;

    // GGML_TENSOR_UNARY_OP_LOCALS
    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    // `size_t` into the kernel's `int64_t` parameters.
    const nb00: i64 = @bitCast(src0.nb[0]);
    const nb01: i64 = @bitCast(src0.nb[1]);
    const nb02: i64 = @bitCast(src0.nb[2]);
    const nb03: i64 = @bitCast(src0.nb[3]);

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->src[0]->type == GGML_TYPE_F32");
    ggml.impl.assert(ggml.types.ggml_are_same_shape(src0, dst), "ggml_are_same_shape(dst->src[0], dst)");

    const stream = ctx.stream();

    const sz: i64 = ne00 * ne01 * ne02 * ne03;
    const num_blocks: i64 = @divTrunc(sz + block_size - 1, block_size);

    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("roll", "roll_f32_cuda"), .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ src0_d, dst_d, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, s0, s1, s2, s3 });
}
