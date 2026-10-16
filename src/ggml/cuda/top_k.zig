//! `GGML_OP_TOP_K`: the indices of each row's `k` largest values.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/top-k.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/top-k.cuh` at v0.3.0 (`c1d0e7a00`). It has
//! no kernel of its own: it sorts with `argsort.zig`'s row sorters and
//! copies the first `k` columns.
//!
//! # Which arm
//!
//! `CUB_TOP_K_AVAILABLE` needs CCCL >= 3.2 and this toolkit ships 3.0, so
//! upstream's `cub::DeviceTopK` arm is not compiled; `GGML_CUDA_USE_CUB` is
//! defined, so the "argsort + copy" arm is the one ported. `top_k_cub` is
//! therefore not ported -- the reference does not contain it either.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const argsort = @import("argsort.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `ggml_cuda_op_top_k` (top-k.cu:51 @c1d0e7a00), the
/// `GGML_CUDA_USE_CUB` arm without `CUB_TOP_K_AVAILABLE`, with that arm's
/// own `next_power_of_2` (top-k.cu:41 @c1d0e7a00), which is `argsort.cu`'s.
pub fn opTopK(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    var src0_d: [*]const f32 = @ptrCast(@alignCast(src0.data.?));
    var dst_d: [*]c_int = @ptrCast(@alignCast(dst.data.?));
    const stream = ctx.stream();

    // are these asserts truly necessary?
    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_I32, "dst->type == GGML_TYPE_I32");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const ncols: i64 = src0.ne[0];
    const nrows: i64 = c.ggml_nrows(src0);
    const k: i64 = dst.ne[0];
    const pool = ctx.pool();

    // Fall back to argsort + copy
    const ncols_pad = argsort.nextPowerOf2(@truncate(ncols));
    const shared_mem: usize = @as(usize, @bitCast(@as(isize, ncols_pad))) *% @sizeOf(c_int);
    const max_shared_mem = common.info().devices[@intCast(common.getDevice())].smpb;
    const use_bitonic = shared_mem <= max_shared_mem and ncols <= 1024;
    const chunk_nrows = argsort.cubChunkNrows(src0.nb[1], nrows);

    var temp_dst_alloc = common.PoolAlloc(c_int).initAlloc(pool, @bitCast(ncols * chunk_nrows));
    defer temp_dst_alloc.deinit();
    const tmp_dst = temp_dst_alloc.get().?;

    var i: i64 = 0;
    while (i < nrows) : (i += chunk_nrows) {
        const iter_nrows: c_int = @truncate(@min(@as(i64, chunk_nrows), nrows - i));

        if (use_bitonic) {
            argsort.argsortF32I32CudaBitonic(src0_d, tmp_dst, @truncate(ncols), iter_nrows, c.GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort.argsortF32I32CudaCub(pool, src0_d, tmp_dst, @truncate(ncols), iter_nrows, c.GGML_SORT_ORDER_DESC, stream);
        }
        const isz: i64 = @sizeOf(c_int);
        common.cudaCheck(cu.cudaMemcpy2DAsync(
            dst_d,
            @bitCast(k * isz),
            tmp_dst,
            @bitCast(ncols * isz),
            @bitCast(k * isz),
            @intCast(iter_nrows),
            cu.cudaMemcpyDeviceToDevice,
            stream,
        ), "cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows, cudaMemcpyDeviceToDevice, stream)", @src());

        src0_d += @as(usize, @bitCast(ncols * iter_nrows));
        dst_d += @as(usize, @bitCast(k * iter_nrows));
    }
}
