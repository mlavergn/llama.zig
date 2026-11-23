//! `GGML_OP_MEAN`: the mean of each row.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mean.cu` at v1.0.0
//! (`b1b770d90`). The kernels, `reduce_rows_f32<true>` and
//! `reduce_rows_f32_strided<true>` from `reduce_rows.cuh`, stay CUDA.
//!
//! A single long row (`nrows == 1` and more than 65536 columns, or more than
//! 32768 while CUDA graphs are capturing or replaying) goes to
//! `cub::DeviceReduce::Sum`, through `cub_shim.cu`,
//! and then `divide_by_count`.

const ggml = @import("../module.zig");
const kernel = ggml.cuda.kernels.get("mean", "reduce_rows_f32<(bool)1>");
const kernel_strided = ggml.cuda.kernels.get("mean", "reduce_rows_f32_strided<(bool)1>");
const divide_by_count = ggml.cuda.kernels.get("mean", "divide_by_count<float>");

/// Ports `ggml_cuda_op_mean` (mean.cu:13 @b1b770d90), the
/// `GGML_CUDA_USE_CUB` and `USE_CUDA_GRAPH` arms. A source with contiguous
/// rows but gaps between them takes the strided kernel.
pub fn opMean(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(src0), "ggml_is_contiguous_rows(src0)");

    const ncols: i64 = src0.ne[0];
    const nrows: i64 = ggml.types.ggml_nrows(src0);

    // Special case for reducing vectors
    var iscapturing: ggml.cuda.common.cu.cudaStreamCaptureStatus = undefined;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamIsCapturing(stream, &iscapturing), "cudaStreamIsCapturing(stream, &iscapturing)", @src());
    // Determine if CUDA graphs are effectively disabled for this context
    // (no graph instance exists and we're not capturing, OR graphs are
    // explicitly enabled)
    const graphs_off = (!ctx.anyCudaGraphHasInstance() and iscapturing == ggml.cuda.common.cu.cudaStreamCaptureStatusNone) or ctx.anyCudaGraphEnabled();
    if (nrows == 1 and ((ncols > 65536 and graphs_off) or
        // CUDA graphs are enabled - use lower threshold
        (ncols > 32768 and !graphs_off)))
    {
        // Single row - use device-wide reduction
        var tmp_size: usize = 0;
        const pool = ctx.pool();
        const x: ?*const f32 = @ptrCast(@alignCast(src0_d));
        const y: ?*f32 = @ptrCast(@alignCast(dst_d));

        _ = ggml.cuda.sum.ggml_cuda_cub_reduce_sum_f32(null, &tmp_size, x, y, ncols, stream);

        var tmp_alloc = ggml.cuda.common.PoolAlloc(u8).initAlloc(pool, tmp_size);
        defer tmp_alloc.deinit();
        _ = ggml.cuda.sum.ggml_cuda_cub_reduce_sum_f32(tmp_alloc.get(), &tmp_size, x, y, ncols, stream);

        // Divide by ncols
        ggml.cuda.kernels.launch(divide_by_count, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 }, .stream = stream }, .{ y, @as(usize, @bitCast(ncols)) });
        return;
    }

    const block_nums: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(nrows))) };

    const id = ggml.cuda.common.getDevice();
    const nsm = ggml.cuda.common.info().devices[@intCast(id)].nsm;

    const n: c_int = @truncate(ncols);
    // Heuristic for block size selection to optimize occupancy.
    // See discussion in: https://github.com/ggml-org/llama.cpp/pull/15132
    var block_dims: ggml.cuda.kernels.Dim3 = undefined;
    if (@divTrunc(nrows, nsm) < 2) {
        block_dims = .{ .x = 512 };
    } else {
        block_dims = .{ .x = if (ncols < 1024) 32 else 128 };
    }
    const launch_params: ggml.cuda.kernels.LaunchParams = .{ .grid = block_nums, .block = block_dims, .stream = stream };

    if (ggml.types.ggml_is_contiguous(src0)) {
        ggml.cuda.kernels.launchPdl(kernel, launch_params, .{ src0_d, dst_d, n });
        return;
    }

    const src0_d_bytes: ?[*]const u8 = @ptrCast(src0.data);
    ggml.cuda.kernels.launchPdl(kernel_strided, launch_params, .{
        src0_d_bytes,
        dst_d,
        n,
        src0.ne[1],
        src0.ne[2],
        @as(i64, @bitCast(src0.nb[1])),
        @as(i64, @bitCast(src0.nb[2])),
        @as(i64, @bitCast(src0.nb[3])),
    });
}
