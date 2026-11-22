//! `CROSS_ENTROPY_LOSS` and `CROSS_ENTROPY_LOSS_BACK`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss.cuh` at v1.0.0
//! (`b1b770d90`). The kernels -- `cross_entropy_loss_f32` and
//! `cross_entropy_loss_back_f32` -- stay CUDA.
//!
//! # The forward op is not wired
//!
//! `ggml_cuda_cross_entropy_loss` reduces the per-row losses with
//! `sum_f32_cuda`, which is `sum.zig`'s.

const ggml = @import("../module.zig");
/// Ports `CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE` (cross-entropy-loss.cuh:3 @b1b770d90).
/// Unused by the C's host code too.
const block_size = 256;

/// Ports `ggml_cuda_cross_entropy_loss` (cross-entropy-loss.cu:94 @b1b770d90).
pub fn opCrossEntropyLoss(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const ne00 = src0.ne[0];
    const nrows = ggml.impl.ggml_c.ggml_nrows(src0);

    const pool = ctx.pool();
    const stream = ctx.stream();

    const blocks_dim: ggml.cuda.kernels.Dim3 = .{ .x = ggml.cuda.common.warp_size };
    const blocks_num: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(nrows))) };
    const nbytes_shared: usize = @as(usize, @bitCast(ne00)) *% @sizeOf(f32);

    const id = ggml.cuda.common.getDevice();
    const smpbo = ggml.cuda.common.info().devices[@intCast(id)].smpbo;

    var dst_tmp = ggml.cuda.common.PoolAlloc(f32).initAlloc(pool, blocks_num.x);
    defer dst_tmp.deinit();
    const dst_tmp_ptr: ?*anyopaque = @ptrCast(dst_tmp.ptr);

    const args = .{ src0.data, src1.data, dst_tmp_ptr, @as(c_int, @truncate(ne00)), @as(c_int, @truncate(nrows)) };
    if (nbytes_shared <= smpbo) {
        const k = comptime ggml.cuda.kernels.get("cross-entropy-loss", "cross_entropy_loss_f32<(bool)1>");
        ggml.cuda.kernels.setSharedMemoryLimit(k, smpbo);
        ggml.cuda.kernels.launch(k, .{ .grid = blocks_num, .block = blocks_dim, .shmem = nbytes_shared, .stream = stream }, args);
    } else {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("cross-entropy-loss", "cross_entropy_loss_f32<(bool)0>"), .{ .grid = blocks_num, .block = blocks_dim, .shmem = 0, .stream = stream }, args);
    }
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetLastError(), "cudaGetLastError()", @src());

    // Combine results from individual blocks:
    ggml.cuda.sum.sumF32Cuda(pool, @ptrCast(dst_tmp.ptr), @ptrCast(@alignCast(dst.data)), blocks_num.x, stream);
}

/// Ports `ggml_cuda_cross_entropy_loss_back` (cross-entropy-loss.cu:137 @b1b770d90).
pub fn opCrossEntropyLossBack(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const grad = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src0f = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);
    const src1f = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[2]);

    ggml.impl.assert(src0f.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0f->type == GGML_TYPE_F32");
    ggml.impl.assert(src1f.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1f->type == GGML_TYPE_F32");
    ggml.impl.assert(grad.type == ggml.impl.ggml_c.GGML_TYPE_F32, " grad->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "  dst->type == GGML_TYPE_F32");

    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_scalar(grad), "ggml_is_scalar(grad)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0f), "ggml_is_contiguous(src0f)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src1f), "ggml_is_contiguous(src1f)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(src0f, src1f), "ggml_are_same_shape(src0f, src1f)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(src0f, dst), "ggml_are_same_shape(src0f, dst)");

    const ne00 = src0f.ne[0];
    const nrows = ggml.impl.ggml_c.ggml_nrows(src0f);

    const stream = ctx.stream();

    const blocks_dim: ggml.cuda.kernels.Dim3 = .{ .x = ggml.cuda.common.warp_size };
    const blocks_num: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(nrows))) };
    const nbytes_shared: usize = @as(usize, @bitCast(ne00)) *% @sizeOf(f32);

    const id = ggml.cuda.common.getDevice();
    const smpbo = ggml.cuda.common.info().devices[@intCast(id)].smpbo;

    const args = .{ grad.data, src0f.data, src1f.data, dst.data, @as(c_int, @truncate(ne00)) };
    if (nbytes_shared <= smpbo) {
        const k = comptime ggml.cuda.kernels.get("cross-entropy-loss", "cross_entropy_loss_back_f32<(bool)1>");
        ggml.cuda.kernels.setSharedMemoryLimit(k, smpbo);
        ggml.cuda.kernels.launch(k, .{ .grid = blocks_num, .block = blocks_dim, .shmem = nbytes_shared, .stream = stream }, args);
    } else {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("cross-entropy-loss", "cross_entropy_loss_back_f32<(bool)0>"), .{ .grid = blocks_num, .block = blocks_dim, .shmem = 0, .stream = stream }, args);
    }
}
