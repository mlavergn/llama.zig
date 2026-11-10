//! `GGML_OP_MEAN`: the mean of each row.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mean.cu` at v0.9.0
//! (`cd959bae9`). The kernels, `reduce_rows_f32<true>` and
//! `reduce_rows_f32_strided<true>` from `reduce_rows.cuh`, stay CUDA.
//!
//! A single long row (`nrows == 1` and more than 65536 columns, or more than
//! 32768 while CUDA graphs are capturing or replaying) goes to
//! `cub::DeviceReduce::Sum`, through `cub_shim.cu` (PLAN.md, Decision 47),
//! and then `divide_by_count`.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

const sum = @import("sum.zig");

const kernel = kernels.get("mean", "reduce_rows_f32<(bool)1>");
const kernel_strided = kernels.get("mean", "reduce_rows_f32_strided<(bool)1>");
const divide_by_count = kernels.get("mean", "divide_by_count<float>");

/// Ports `ggml_cuda_op_mean` (mean.cu:13 @cd959bae9), the
/// `GGML_CUDA_USE_CUB` and `USE_CUDA_GRAPH` arms. A source with contiguous
/// rows but gaps between them takes the strided kernel.
pub fn opMean(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous_rows(src0), "ggml_is_contiguous_rows(src0)");

    const ncols: i64 = src0.ne[0];
    const nrows: i64 = c.ggml_nrows(src0);

    // Special case for reducing vectors
    var iscapturing: cu.cudaStreamCaptureStatus = undefined;
    common.cudaCheck(cu.cudaStreamIsCapturing(stream, &iscapturing), "cudaStreamIsCapturing(stream, &iscapturing)", @src());
    // Determine if CUDA graphs are effectively disabled for this context
    // (no graph instance exists and we're not capturing, OR graphs are
    // explicitly enabled)
    const graphs_off = (!ctx.anyCudaGraphHasInstance() and iscapturing == cu.cudaStreamCaptureStatusNone) or ctx.anyCudaGraphEnabled();
    if (nrows == 1 and ((ncols > 65536 and graphs_off) or
        // CUDA graphs are enabled - use lower threshold
        (ncols > 32768 and !graphs_off)))
    {
        // Single row - use device-wide reduction
        var tmp_size: usize = 0;
        const pool = ctx.pool();
        const x: ?*const f32 = @ptrCast(@alignCast(src0_d));
        const y: ?*f32 = @ptrCast(@alignCast(dst_d));

        _ = sum.ggml_cuda_cub_reduce_sum_f32(null, &tmp_size, x, y, ncols, stream);

        var tmp_alloc = common.PoolAlloc(u8).initAlloc(pool, tmp_size);
        defer tmp_alloc.deinit();
        _ = sum.ggml_cuda_cub_reduce_sum_f32(tmp_alloc.get(), &tmp_size, x, y, ncols, stream);

        // Divide by ncols
        kernels.launch(divide_by_count, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 }, .stream = stream }, .{ y, @as(usize, @bitCast(ncols)) });
        return;
    }

    const block_nums: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(nrows))) };

    const id = common.getDevice();
    const nsm = common.info().devices[@intCast(id)].nsm;

    const n: c_int = @truncate(ncols);
    // Heuristic for block size selection to optimize occupancy.
    // See discussion in: https://github.com/ggml-org/llama.cpp/pull/15132
    var block_dims: kernels.Dim3 = undefined;
    if (@divTrunc(nrows, nsm) < 2) {
        block_dims = .{ .x = 512 };
    } else {
        block_dims = .{ .x = if (ncols < 1024) 32 else 128 };
    }
    const launch_params: kernels.LaunchParams = .{ .grid = block_nums, .block = block_dims, .stream = stream };

    if (c.ggml_is_contiguous(src0)) {
        kernels.launchPdl(kernel, launch_params, .{ src0_d, dst_d, n });
        return;
    }

    const src0_d_bytes: ?[*]const u8 = @ptrCast(src0.data);
    kernels.launchPdl(kernel_strided, launch_params, .{
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
