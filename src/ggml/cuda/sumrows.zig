//! `GGML_OP_SUM_ROWS`, and `sum_rows_f32_cuda`, which other ops call.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/sumrows.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/sumrows.cuh` at v0.4.0 (`5266f24da`). The
//! kernel, `reduce_rows_f32<false>` from `reduce_rows.cuh`, stays CUDA.
//!
//! The block size depends on the device's SM count: 512 threads when there
//! are fewer than two rows per SM, else 32 or 128 by row length.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

const kernel = kernels.get("sumrows", "reduce_rows_f32<(bool)0>");

/// Ports `sum_rows_f32_cuda` (sumrows.cu:4 @5266f24da).
pub fn sumRowsF32Cuda(x: ?*const anyopaque, dst: ?*anyopaque, ncols: c_int, nrows: c_int, stream: cu.cudaStream_t) void {
    const id = common.getDevice();
    const nsm = common.info().devices[@intCast(id)].nsm;
    const block_nums: kernels.Dim3 = .{ .x = @bitCast(nrows) };
    if (@divTrunc(nrows, nsm) < 2) {
        const block_dims: kernels.Dim3 = .{ .x = 512 };
        kernels.launchPdl(kernel, .{ .grid = block_nums, .block = block_dims, .stream = stream }, .{ x, dst, ncols });
    } else {
        const block_dims: kernels.Dim3 = .{ .x = if (ncols < 1024) 32 else 128 };
        kernels.launchPdl(kernel, .{ .grid = block_nums, .block = block_dims, .stream = stream }, .{ x, dst, ncols });
    }
}

/// Ports `ggml_cuda_op_sum_rows` (sumrows.cu:19 @5266f24da).
pub fn opSumRows(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const ncols: i64 = src0.ne[0];
    const nrows: i64 = c.ggml_nrows(src0);

    const block_nums: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(nrows))) };

    const id = common.getDevice();
    const nsm = common.info().devices[@intCast(id)].nsm;
    const n: c_int = @truncate(ncols);
    if (@divTrunc(nrows, nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        const block_dims: kernels.Dim3 = .{ .x = 512 };
        kernels.launchPdl(kernel, .{ .grid = block_nums, .block = block_dims, .stream = stream }, .{ src0_d, dst_d, n });
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        const block_dims: kernels.Dim3 = .{ .x = if (ncols < 1024) 32 else 128 };
        kernels.launchPdl(kernel, .{ .grid = block_nums, .block = block_dims, .stream = stream }, .{ src0_d, dst_d, n });
    }
}

comptime {
    // Its one caller, `sum_f32_cuda`, reaches it only when
    // `GGML_CUDA_USE_CUB` is undefined, which it never is on this target
    // (CUDART >= 11.7). Referencing it keeps it compiled and type-checked.
    _ = &sumRowsF32Cuda;
}
