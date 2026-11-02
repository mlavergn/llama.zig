//! `GGML_OP_SUM_ROWS`, and `sum_rows_f32_cuda`, which other ops call.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/sumrows.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/sumrows.cuh` at v0.5.0 (`7fe450e19`). The
//! kernels, `reduce_rows_f32<false>` and `reduce_rows_f32_strided<false>`
//! from `reduce_rows.cuh`, stay CUDA.
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
const kernel_strided = kernels.get("sumrows", "reduce_rows_f32_strided<(bool)0>");

/// Ports `sum_rows_f32_cuda` (sumrows.cu:4 @7fe450e19).
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

/// Ports `ggml_cuda_op_sum_rows` (sumrows.cu:19 @7fe450e19). A source with
/// contiguous rows but gaps between them takes the strided kernel, which
/// reads each row through `nb[1..3]`.
pub fn opSumRows(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous_rows(src0), "ggml_is_contiguous_rows(src0)");

    const ncols: i64 = src0.ne[0];
    const nrows: i64 = c.ggml_nrows(src0);

    if (c.ggml_is_contiguous(src0)) {
        sumRowsF32Cuda(src0_d, dst_d, @truncate(ncols), @truncate(nrows), stream);
        return;
    }

    const block_nums: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(nrows))) };

    const id = common.getDevice();
    const nsm = common.info().devices[@intCast(id)].nsm;
    var block_dims: kernels.Dim3 = undefined;
    if (@divTrunc(nrows, nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        block_dims = .{ .x = 512 };
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        block_dims = .{ .x = if (ncols < 1024) 32 else 128 };
    }
    const src0_d_bytes: ?[*]const u8 = @ptrCast(src0.data);
    kernels.launchPdl(kernel_strided, .{ .grid = block_nums, .block = block_dims, .stream = stream }, .{
        src0_d_bytes,
        dst_d,
        @as(c_int, @truncate(ncols)),
        src0.ne[1],
        src0.ne[2],
        @as(i64, @bitCast(src0.nb[1])),
        @as(i64, @bitCast(src0.nb[2])),
        @as(i64, @bitCast(src0.nb[3])),
    });
}
