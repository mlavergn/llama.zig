//! `GGML_OP_ACC`: `dst = src0`, with `src1` added into a strided view of it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/acc.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/acc.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `acc_f32`, stays CUDA.
//!
//! The view's strides and offset come from `op_params`, which are
//! `int32_t`, divided by `sizeof(float)`: an `int32_t` divided by a
//! `size_t` is an unsigned division, and the quotient is converted back to
//! `int64_t`. Reproduced as such.

const ggml = @import("../module.zig");
/// Ports `CUDA_ACC_BLOCK_SIZE` (acc.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `acc_f32_cuda` (acc.cu:30 @b1b770d90), which launches with
/// `<<<>>>`.
fn accF32Cuda(x: ?*const anyopaque, y: ?*const anyopaque, dst: ?*anyopaque, n_elements: i64, ne10: i64, ne11: i64, ne12: i64, ne13: i64, s1: i64, s2: i64, s3: i64, off: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const num_blocks: c_int = @truncate(@divTrunc(n_elements + block_size - 1, block_size));
    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("acc", "acc_f32"), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, y, dst, n_elements, ne10, ne11, ne12, ne13, s1, s2, s3, off });
}

/// `op_params[i] / sizeof(float)`.
fn floats(p: i32) i64 {
    return @bitCast(@as(u64, @bitCast(@as(i64, p))) / @sizeOf(f32));
}

/// Ports `ggml_cuda_op_acc` (acc.cu:37 @b1b770d90).
pub fn opAcc(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]);

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const anyopaque = src1.data;
    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    ggml.impl.assert(src0.type == ggml.impl.c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(src1.type == ggml.impl.c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    ggml.impl.assert(ggml.impl.c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    ggml.impl.assert(dst.nb[0] == ggml.impl.c.ggml_element_size(dst), "dst->nb[0] == ggml_element_size(dst)");
    ggml.impl.assert(ggml.impl.c.ggml_is_contiguously_allocated(dst), "ggml_is_contiguously_allocated(dst)");

    const s1 = floats(dst.op_params[0]);
    const s2 = floats(dst.op_params[1]);
    const s3 = floats(dst.op_params[2]);
    const off = floats(dst.op_params[3]);

    accF32Cuda(src0_d, src1_d, dst_d, ggml.impl.c.ggml_nelements(dst), src1.ne[0], src1.ne[1], src1.ne[2], src1.ne[3], s1, s2, s3, off, stream);
}
