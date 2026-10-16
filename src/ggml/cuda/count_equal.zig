//! `GGML_OP_COUNT_EQUAL`: the number of equal elements of two tensors.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/count-equal.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/count-equal.cuh` at v0.3.0 (`c1d0e7a00`).
//! The kernel, `count_equal<int>`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_COUNT_EQUAL_CHUNK_SIZE` (count-equal.cuh:3 @c1d0e7a00).
const chunk_size = 128;

/// `GGML_PAD(x, n)` on `int64_t`.
fn pad(x: i64, n: i64) i64 {
    return (x + n - 1) & ~(n - 1);
}

/// Ports `ggml_cuda_count_equal` (count-equal.cu:28 @c1d0e7a00), which
/// launches with `<<<>>>`.
pub fn opCountEqual(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(src0.type == src1.type, "src0->type == src1->type");
    impl.assert(dst.type == c.GGML_TYPE_I64, " dst->type == GGML_TYPE_I64");

    impl.assert(c.ggml_are_same_shape(src0, src1), "ggml_are_same_shape(src0, src1)");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();
    const nsm: c_int = common.info().devices[@intCast(common.getDevice())].nsm;

    const ne: i64 = c.ggml_nelements(src0);
    impl.assert(ne < (1 << 30), "ne < (1 << 30) && \"atomicAdd implementation only supports int\"");
    const dne: i64 = pad(@divTrunc(ne + 4 * nsm - 1, 4 * nsm), chunk_size);

    common.cudaCheck(cu.cudaMemsetAsync(dst_d, 0, c.ggml_nbytes(dst), stream), "cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream)", @src());

    const blocks_dim: kernels.Dim3 = .{ .x = common.warp_size };
    const blocks_num: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(@min(@as(i64, 4) * nsm, @divTrunc(ne + chunk_size - 1, chunk_size))))) };

    switch (src0.type) {
        c.GGML_TYPE_I32 => {
            const src0_d: ?*const anyopaque = src0.data;
            const src1_d: ?*const anyopaque = src1.data;
            kernels.launch(kernels.get("count-equal", "count_equal<int>"), .{ .grid = blocks_num, .block = blocks_dim, .stream = stream }, .{ src0_d, src1_d, dst_d, dne, ne });
        },
        else => impl.assert(false, "false"),
    }
}
