//! `GGML_OP_ACC`: `dst = src0`, with `src1` added into a strided view of it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/acc.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/acc.cuh` at v0.4.1 (`b29c606e2`). The
//! kernel, `acc_f32`, stays CUDA.
//!
//! The view's strides and offset come from `op_params`, which are
//! `int32_t`, divided by `sizeof(float)`: an `int32_t` divided by a
//! `size_t` is an unsigned division, and the quotient is converted back to
//! `int64_t`. Reproduced as such.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_ACC_BLOCK_SIZE` (acc.cuh:3 @b29c606e2).
const block_size = 256;

/// Ports `acc_f32_cuda` (acc.cu:30 @b29c606e2), which launches with
/// `<<<>>>`.
fn accF32Cuda(x: ?*const anyopaque, y: ?*const anyopaque, dst: ?*anyopaque, n_elements: i64, ne10: i64, ne11: i64, ne12: i64, ne13: i64, s1: i64, s2: i64, s3: i64, off: i64, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @truncate(@divTrunc(n_elements + block_size - 1, block_size));
    kernels.launch(kernels.get("acc", "acc_f32"), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, y, dst, n_elements, ne10, ne11, ne12, ne13, s1, s2, s3, off });
}

/// `op_params[i] / sizeof(float)`.
fn floats(p: i32) i64 {
    return @bitCast(@as(u64, @bitCast(@as(i64, p))) / @sizeOf(f32));
}

/// Ports `ggml_cuda_op_acc` (acc.cu:37 @b29c606e2).
pub fn opAcc(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const anyopaque = src1.data;
    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    impl.assert(dst.nb[0] == c.ggml_element_size(dst), "dst->nb[0] == ggml_element_size(dst)");
    impl.assert(c.ggml_is_contiguously_allocated(dst), "ggml_is_contiguously_allocated(dst)");

    const s1 = floats(dst.op_params[0]);
    const s2 = floats(dst.op_params[1]);
    const s3 = floats(dst.op_params[2]);
    const off = floats(dst.op_params[3]);

    accF32Cuda(src0_d, src1_d, dst_d, c.ggml_nelements(dst), src1.ne[0], src1.ne[1], src1.ne[2], src1.ne[3], s1, s2, s3, off, stream);
}
