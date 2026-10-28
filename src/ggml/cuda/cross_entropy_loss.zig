//! `CROSS_ENTROPY_LOSS` and `CROSS_ENTROPY_LOSS_BACK`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss.cuh` at v0.4.0
//! (`5266f24da`). The kernels -- `cross_entropy_loss_f32` and
//! `cross_entropy_loss_back_f32` -- stay CUDA.
//!
//! # The forward op is not wired
//!
//! `ggml_cuda_cross_entropy_loss` reduces the per-row losses with
//! `sum_f32_cuda`, which is `sum.zig`'s.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const sum = @import("sum.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

const warp_size = common.warp_size;

/// Ports `CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE` (cross-entropy-loss.cuh:3 @5266f24da).
/// Unused by the C's host code too.
const block_size = 256;

/// Ports `ggml_cuda_cross_entropy_loss` (cross-entropy-loss.cu:94 @5266f24da).
pub fn opCrossEntropyLoss(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const ne00 = src0.ne[0];
    const nrows = c.ggml_nrows(src0);

    const pool = ctx.pool();
    const stream = ctx.stream();

    const blocks_dim: kernels.Dim3 = .{ .x = warp_size };
    const blocks_num: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(nrows))) };
    const nbytes_shared: usize = @as(usize, @bitCast(ne00)) *% @sizeOf(f32);

    const id = common.getDevice();
    const smpbo = common.info().devices[@intCast(id)].smpbo;

    var dst_tmp = common.PoolAlloc(f32).initAlloc(pool, blocks_num.x);
    defer dst_tmp.deinit();
    const dst_tmp_ptr: ?*anyopaque = @ptrCast(dst_tmp.ptr);

    const args = .{ src0.data, src1.data, dst_tmp_ptr, @as(c_int, @truncate(ne00)), @as(c_int, @truncate(nrows)) };
    if (nbytes_shared <= smpbo) {
        const k = comptime kernels.get("cross-entropy-loss", "cross_entropy_loss_f32<(bool)1>");
        kernels.setSharedMemoryLimit(k, smpbo);
        kernels.launch(k, .{ .grid = blocks_num, .block = blocks_dim, .shmem = nbytes_shared, .stream = stream }, args);
    } else {
        kernels.launch(kernels.get("cross-entropy-loss", "cross_entropy_loss_f32<(bool)0>"), .{ .grid = blocks_num, .block = blocks_dim, .shmem = 0, .stream = stream }, args);
    }
    common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());

    // Combine results from individual blocks:
    sum.sumF32Cuda(pool, @ptrCast(dst_tmp.ptr), @ptrCast(@alignCast(dst.data)), blocks_num.x, stream);
}

/// Ports `ggml_cuda_cross_entropy_loss_back` (cross-entropy-loss.cu:137 @5266f24da).
pub fn opCrossEntropyLossBack(ctx: *common.Context, dst: *Tensor) void {
    const grad = impl.one(Tensor, dst.src[0]);
    const src0f = impl.one(Tensor, dst.src[1]);
    const src1f = impl.one(Tensor, dst.src[2]);

    impl.assert(src0f.type == c.GGML_TYPE_F32, "src0f->type == GGML_TYPE_F32");
    impl.assert(src1f.type == c.GGML_TYPE_F32, "src1f->type == GGML_TYPE_F32");
    impl.assert(grad.type == c.GGML_TYPE_F32, " grad->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "  dst->type == GGML_TYPE_F32");

    impl.assert(c.ggml_is_scalar(grad), "ggml_is_scalar(grad)");
    impl.assert(c.ggml_is_contiguous(src0f), "ggml_is_contiguous(src0f)");
    impl.assert(c.ggml_is_contiguous(src1f), "ggml_is_contiguous(src1f)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    impl.assert(c.ggml_are_same_shape(src0f, src1f), "ggml_are_same_shape(src0f, src1f)");
    impl.assert(c.ggml_are_same_shape(src0f, dst), "ggml_are_same_shape(src0f, dst)");

    const ne00 = src0f.ne[0];
    const nrows = c.ggml_nrows(src0f);

    const stream = ctx.stream();

    const blocks_dim: kernels.Dim3 = .{ .x = warp_size };
    const blocks_num: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(nrows))) };
    const nbytes_shared: usize = @as(usize, @bitCast(ne00)) *% @sizeOf(f32);

    const id = common.getDevice();
    const smpbo = common.info().devices[@intCast(id)].smpbo;

    const args = .{ grad.data, src0f.data, src1f.data, dst.data, @as(c_int, @truncate(ne00)) };
    if (nbytes_shared <= smpbo) {
        const k = comptime kernels.get("cross-entropy-loss", "cross_entropy_loss_back_f32<(bool)1>");
        kernels.setSharedMemoryLimit(k, smpbo);
        kernels.launch(k, .{ .grid = blocks_num, .block = blocks_dim, .shmem = nbytes_shared, .stream = stream }, args);
    } else {
        kernels.launch(kernels.get("cross-entropy-loss", "cross_entropy_loss_back_f32<(bool)0>"), .{ .grid = blocks_num, .block = blocks_dim, .shmem = 0, .stream = stream }, args);
    }
}
