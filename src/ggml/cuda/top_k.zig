//! `GGML_OP_TOP_K`: the indices of each row's `k` largest values.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/top-k.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/top-k.cuh` at v1.0.0 (`b1b770d90`). The
//! radix-select kernels -- `top_k_radix_init`, `top_k_radix_histogram`,
//! `top_k_radix_select`, `top_k_radix_gather` -- stay CUDA; the argsort arms sort with
//! `argsort.zig`'s row sorters and copy the first `k` columns.
//!
//! # Which arms
//!
//! `CUB_TOP_K_AVAILABLE` needs CCCL >= 3.4.3 (`DeviceTopK` races before it)
//! and this toolkit ships 3.0, so upstream's `cub::DeviceTopK` arm is not
//! compiled and `top_k_cub` is not ported -- the reference does not
//! contain it either. `GGML_CUDA_USE_CUB` is defined. (#28713)
//! It chooses among three implementations by shape: bitonic argsort for short
//! rows (and up to 1024 columns while the rows fit in one wave of SMs),
//! radix select for more than one long row, and CUB argsort, or radix
//! select past 4096 columns, for one long row.

const ggml = @import("../module.zig");
/// Ports `GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC` (top-k.cu:10 @b1b770d90),
/// the CUDA arm.
const ncols_threshold_bitonic = 512;
/// Ports `GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT` (top-k.cu:15 @b1b770d90).
const ncols_threshold_argsort = 4096;
/// Ports `GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS` (top-k.cu:23
/// @b1b770d90), the CUDA arm: bitonic up to this width while nrows fits in
/// one wave of SMs.
const ncols_threshold_bitonic_few_rows = 1024;
/// Ports `GGML_CUDA_TOP_K_NROWS_THRESHOLD` (top-k.cu:45 @b1b770d90), the arm
/// without `CUB_TOP_K_AVAILABLE`: max rows for the per-row CUB path before
/// switching to radix or bitonic.
const nrows_threshold = 1;

/// Ports `top_k_radix_state` (top-k.cu:92 @b1b770d90), for its size: the
/// host only allocates the states. It is `alignas(128)`, one
/// state per cache line, and has no `equal_count`.
const RadixState = extern struct {
    prefix: u32 align(128),
    prefix_mask: u32,
    rank: c_int,
    greater_count: c_int,
};

comptime {
    if (@sizeOf(RadixState) != 128) @compileError("top_k_radix_state is alignas(128)");
}

/// Ports `top_k_radix_cuda` (top-k.cu:261 @b1b770d90), which launches with
/// `<<<>>>`. The rows go in chunks that bound the histograms
/// to 64 MB. Each block of a row reads a contiguous range of
/// `block_cols` columns, and the gather breaks ties by column from the last
/// pass's histograms.
fn topKRadixCuda(pool: *ggml.cuda.common.Pool, src_in: [*]const f32, dst_in: [*]c_int, ncols: c_int, nrows: i64, k: c_int, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const BLOCK_SIZE = 256;
    const RADIX_BITS = 8;
    const NBINS = 1 << RADIX_BITS;
    const blocks_per_row: c_int = @intCast(@min(@divTrunc(@as(i64, ncols) + 1023, 1024), 64));
    // each block of a row reads a contiguous range of whole tiles
    const block_cols: c_int = @truncate(@divTrunc(@as(i64, ncols) + blocks_per_row * BLOCK_SIZE - 1, blocks_per_row * BLOCK_SIZE) * BLOCK_SIZE);

    var src = src_in;
    var dst = dst_in;

    // chunk the rows to bound the histogram memory to 64 MB
    const chunk_nrows: i64 = ggml.cuda.argsort.chunkNrows(@as(usize, @intCast(blocks_per_row)) * NBINS * @sizeOf(c_int), nrows);

    var states_alloc = ggml.cuda.common.PoolAlloc(RadixState).initAlloc(pool, @intCast(chunk_nrows));
    defer states_alloc.deinit();
    var histograms_alloc = ggml.cuda.common.PoolAlloc(c_int).initAlloc(pool, @as(usize, @intCast(chunk_nrows)) * @as(usize, @intCast(blocks_per_row)) * NBINS);
    defer histograms_alloc.deinit();
    const states: ?*anyopaque = @ptrCast(states_alloc.get().?);
    const histograms: ?*anyopaque = @ptrCast(histograms_alloc.get().?);

    const init_k = ggml.cuda.kernels.get("top-k", "top_k_radix_init");
    const histogram_k = ggml.cuda.kernels.get("top-k", ggml.cuda.kernels.tmpl("top_k_radix_histogram", &.{ ggml.cuda.kernels.int(BLOCK_SIZE), ggml.cuda.kernels.int(RADIX_BITS) }));
    const select_k = ggml.cuda.kernels.get("top-k", ggml.cuda.kernels.tmpl("top_k_radix_select", &.{ ggml.cuda.kernels.int(BLOCK_SIZE), ggml.cuda.kernels.int(RADIX_BITS) }));
    const gather_k = ggml.cuda.kernels.get("top-k", ggml.cuda.kernels.tmpl("top_k_radix_gather", &.{ ggml.cuda.kernels.int(BLOCK_SIZE), ggml.cuda.kernels.int(RADIX_BITS) }));

    var i: i64 = 0;
    while (i < nrows) : (i += chunk_nrows) {
        const iter_nrows: c_int = @truncate(@min(chunk_nrows, nrows - i));
        const row_blocks: u32 = @bitCast(@divTrunc(iter_nrows + BLOCK_SIZE - 1, BLOCK_SIZE));

        ggml.cuda.kernels.launch(init_k, .{ .grid = .{ .x = row_blocks }, .block = .{ .x = BLOCK_SIZE }, .stream = stream }, .{ states, iter_nrows, k });

        const row_grid: ggml.cuda.kernels.Dim3 = .{ .x = @bitCast(blocks_per_row *% iter_nrows) };
        var shift: c_int = 32 - RADIX_BITS;
        while (shift >= 0) : (shift -= RADIX_BITS) {
            ggml.cuda.kernels.launch(histogram_k, .{ .grid = row_grid, .block = .{ .x = BLOCK_SIZE }, .stream = stream }, .{ @as(?*const anyopaque, src), @as(?*const anyopaque, states), histograms, ncols, block_cols, blocks_per_row, shift });
            ggml.cuda.kernels.launch(select_k, .{ .grid = .{ .x = @bitCast(iter_nrows) }, .block = .{ .x = BLOCK_SIZE }, .stream = stream }, .{ @as(?*const anyopaque, histograms), states, blocks_per_row, shift });
        }

        ggml.cuda.kernels.launch(gather_k, .{ .grid = row_grid, .block = .{ .x = BLOCK_SIZE }, .stream = stream }, .{ @as(?*const anyopaque, src), @as(?*anyopaque, dst), @as(?*const anyopaque, histograms), states, ncols, block_cols, k, blocks_per_row });

        src += @as(usize, @intCast(ncols)) * @as(usize, @intCast(iter_nrows));
        dst += @as(usize, @intCast(k)) * @as(usize, @intCast(iter_nrows));
    }
}

/// Ports `top_k_argsort_cuda` (top-k.cu:302 @b1b770d90): argsort each chunk
/// of rows into a scratch buffer, bitonic or CUB, and copy the first `k`
/// columns out.
fn topKArgsortCuda(pool: *ggml.cuda.common.Pool, src_in: [*]const f32, dst_in: [*]c_int, ncols: c_int, nrows: i64, k: c_int, use_cub: bool, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const chunk_nrows: i64 = ggml.cuda.argsort.chunkNrows(@as(usize, @intCast(ncols)) * @sizeOf(c_int), nrows);

    var src = src_in;
    var dst = dst_in;

    var tmp_alloc = ggml.cuda.common.PoolAlloc(c_int).initAlloc(pool, @as(usize, @intCast(ncols)) * @as(usize, @intCast(chunk_nrows)));
    defer tmp_alloc.deinit();
    const tmp = tmp_alloc.get().?;

    var i: i64 = 0;
    while (i < nrows) : (i += chunk_nrows) {
        const iter_nrows: c_int = @truncate(@min(chunk_nrows, nrows - i));

        if (use_cub) {
            ggml.cuda.argsort.argsortF32I32CudaCub(pool, src, tmp, ncols, iter_nrows, ggml.impl.ggml_c.GGML_SORT_ORDER_DESC, stream);
        } else {
            ggml.cuda.argsort.argsortF32I32CudaBitonic(src, tmp, ncols, iter_nrows, ggml.impl.ggml_c.GGML_SORT_ORDER_DESC, stream);
        }
        const isz: usize = @sizeOf(c_int);
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpy2DAsync(
            dst,
            @as(usize, @intCast(k)) * isz,
            tmp,
            @as(usize, @intCast(ncols)) * isz,
            @as(usize, @intCast(k)) * isz,
            @intCast(iter_nrows),
            ggml.cuda.common.cu.cudaMemcpyDeviceToDevice,
            stream,
        ), "cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows, cudaMemcpyDeviceToDevice, stream)", @src());

        src += @as(usize, @intCast(ncols)) * @as(usize, @intCast(iter_nrows));
        dst += @as(usize, @intCast(k)) * @as(usize, @intCast(iter_nrows));
    }
}

/// Ports `ggml_cuda_op_top_k` (top-k.cu:330 @b1b770d90), the
/// `GGML_CUDA_USE_CUB` arm without `CUB_TOP_K_AVAILABLE`, with
/// `next_power_of_2` (top-k.cu:77 @b1b770d90), which is `argsort.cu`'s.
pub fn opTopK(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src0_d: [*]const f32 = @ptrCast(@alignCast(src0.data.?));
    const dst_d: [*]c_int = @ptrCast(@alignCast(dst.data.?));
    const stream = ctx.stream();

    // are these asserts truly necessary?
    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_I32, "dst->type == GGML_TYPE_I32");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const ncols: i64 = src0.ne[0];
    const nrows: i64 = ggml.impl.ggml_c.ggml_nrows(src0);
    const k: i64 = dst.ne[0];
    const pool = ctx.pool();

    const device = ggml.cuda.common.getDevice();
    const dev = &ggml.cuda.common.info().devices[@intCast(device)];

    const bitonic_short = ncols <= ncols_threshold_bitonic;
    const bitonic_few_rows = nrows > nrows_threshold and
        ncols <= ncols_threshold_bitonic_few_rows and
        nrows <= dev.nsm;

    // `int` parameters below: the C converts the `int64_t`s implicitly.
    const ncols_i: c_int = @truncate(ncols);
    const k_i: c_int = @truncate(k);

    if (bitonic_short or bitonic_few_rows) {
        // the padded row must fit in shared memory
        const ncols_pad = ggml.cuda.argsort.nextPowerOf2(ncols_i);
        if (@as(usize, @bitCast(@as(isize, ncols_pad))) *% @sizeOf(c_int) <= dev.smpb) {
            topKArgsortCuda(pool, src0_d, dst_d, ncols_i, nrows, k_i, false, stream);
            return;
        }
    }

    if (nrows > nrows_threshold) {
        topKRadixCuda(pool, src0_d, dst_d, ncols_i, nrows, k_i, stream);
        return;
    }

    if (ncols <= ncols_threshold_argsort) {
        topKArgsortCuda(pool, src0_d, dst_d, ncols_i, nrows, k_i, true, stream);
    } else {
        topKRadixCuda(pool, src0_d, dst_d, ncols_i, nrows, k_i, stream);
    }
}
