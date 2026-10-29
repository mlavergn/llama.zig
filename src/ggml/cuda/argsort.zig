//! `GGML_OP_ARGSORT`, and the two row sorters `top_k.zig` reuses.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/argsort.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/argsort.cuh` at v0.4.1 (`b29c606e2`). The
//! kernels -- `init_indices`, `init_offsets`, `k_argsort_f32_i32` -- stay
//! CUDA.
//!
//! # CUB
//!
//! `GGML_CUDA_USE_CUB` is defined for this build (common.cuh:115 @b29c606e2), so rows
//! too long for the bitonic kernel go to CUB's device-wide sorts. Their
//! host-side dispatch is CUB's header templates, which Zig cannot
//! instantiate; `cub_shim.cu` wraps exactly upstream's six calls behind a C
//! ABI (PLAN.md, Decision 47) and this file keeps everything around them.
//!
//! The toolkit here ships CCCL 3.0, so `STRIDED_ITERATOR_AVAILABLE` (CCCL
//! >= 3.1) is not defined: the segment offsets are an `int` array filled by
//! `init_offsets`, not a strided counting iterator. `cub_shim.cu` refuses
//! to compile against a CCCL where that would differ.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

// -----------------------------------------------------------------------------
// The CUB shim (`cub_shim.cu`)

const CubSort = fn (d_temp_storage: ?*anyopaque, temp_storage_bytes: *usize, temp_keys: [*]f32, temp_indices: [*]c_int, dst: [*]c_int, ncols: c_int, stream: cu.cudaStream_t) callconv(.c) cu.cudaError_t;
const CubSegmentedSort = fn (d_temp_storage: ?*anyopaque, temp_storage_bytes: *usize, temp_keys: [*]f32, temp_indices: [*]c_int, dst: [*]c_int, ncols: c_int, nrows: c_int, offset_iterator: [*]c_int, stream: cu.cudaStream_t) callconv(.c) cu.cudaError_t;

extern fn ggml_cuda_cub_radix_sort_pairs(d_temp_storage: ?*anyopaque, temp_storage_bytes: *usize, temp_keys: [*]f32, temp_indices: [*]c_int, dst: [*]c_int, ncols: c_int, stream: cu.cudaStream_t) cu.cudaError_t;
extern fn ggml_cuda_cub_radix_sort_pairs_descending(d_temp_storage: ?*anyopaque, temp_storage_bytes: *usize, temp_keys: [*]f32, temp_indices: [*]c_int, dst: [*]c_int, ncols: c_int, stream: cu.cudaStream_t) cu.cudaError_t;
extern fn ggml_cuda_cub_segmented_radix_sort_pairs(d_temp_storage: ?*anyopaque, temp_storage_bytes: *usize, temp_keys: [*]f32, temp_indices: [*]c_int, dst: [*]c_int, ncols: c_int, nrows: c_int, offset_iterator: [*]c_int, stream: cu.cudaStream_t) cu.cudaError_t;
extern fn ggml_cuda_cub_segmented_radix_sort_pairs_descending(d_temp_storage: ?*anyopaque, temp_storage_bytes: *usize, temp_keys: [*]f32, temp_indices: [*]c_int, dst: [*]c_int, ncols: c_int, nrows: c_int, offset_iterator: [*]c_int, stream: cu.cudaStream_t) cu.cudaError_t;
extern fn ggml_cuda_cub_segmented_sort_pairs(d_temp_storage: ?*anyopaque, temp_storage_bytes: *usize, temp_keys: [*]f32, temp_indices: [*]c_int, dst: [*]c_int, ncols: c_int, nrows: c_int, offset_iterator: [*]c_int, stream: cu.cudaStream_t) cu.cudaError_t;
extern fn ggml_cuda_cub_segmented_sort_pairs_descending(d_temp_storage: ?*anyopaque, temp_storage_bytes: *usize, temp_keys: [*]f32, temp_indices: [*]c_int, dst: [*]c_int, ncols: c_int, nrows: c_int, offset_iterator: [*]c_int, stream: cu.cudaStream_t) cu.cudaError_t;

comptime {
    // The externs above must match the shim's two shapes.
    _ = @as(*const CubSort, &ggml_cuda_cub_radix_sort_pairs);
    _ = @as(*const CubSort, &ggml_cuda_cub_radix_sort_pairs_descending);
    _ = @as(*const CubSegmentedSort, &ggml_cuda_cub_segmented_radix_sort_pairs);
    _ = @as(*const CubSegmentedSort, &ggml_cuda_cub_segmented_radix_sort_pairs_descending);
    _ = @as(*const CubSegmentedSort, &ggml_cuda_cub_segmented_sort_pairs);
    _ = @as(*const CubSegmentedSort, &ggml_cuda_cub_segmented_sort_pairs_descending);
}

// -----------------------------------------------------------------------------
// CUB

/// Ports `argsort_f32_i32_cuda_cub_chunk_nrows` (argsort.cu:33 @b29c606e2):
/// the suggested maximum number of rows to process during one
/// `argsortF32I32CudaCub` call.
pub fn cubChunkNrows(nb01: usize, nrows: i64) c_int {
    // perform argsort in chunks up to approximately this size (currently
    // 64MB) to avoid excessive temporary buffers memory usage
    const chunk_bytes: c_int = 1 << 26;

    // calculate how many rows will fit in one chunk (must be at least one)
    const chunk_nrows: c_int = @max(@as(c_int, @truncate(@as(i64, @bitCast(@as(usize, chunk_bytes) / nb01)))), 1);

    // limit the resulting amount to total nrows
    return @truncate(@min(@as(i64, chunk_nrows), nrows));
}

/// Ports `argsort_f32_i32_cuda_cub` (argsort.cu:45 @b29c606e2).
pub fn argsortF32I32CudaCub(pool: *common.Pool, x: ?*const anyopaque, dst: [*]c_int, ncols: c_int, nrows: c_int, order: c.ggml_sort_order, stream: cu.cudaStream_t) void {
    var temp_indices_alloc = common.PoolAlloc(c_int).initAlloc(pool, @as(usize, @bitCast(@as(isize, ncols *% nrows))));
    defer temp_indices_alloc.deinit();
    var temp_keys_alloc = common.PoolAlloc(f32).initAlloc(pool, @as(usize, @bitCast(@as(isize, ncols *% nrows))));
    defer temp_keys_alloc.deinit();

    const temp_indices = temp_indices_alloc.get().?;
    const temp_keys = temp_keys_alloc.get().?;

    const block_size = 256;
    const grid_size: kernels.Dim3 = .{ .x = @bitCast(@divTrunc(ncols + block_size - 1, block_size)), .y = @bitCast(nrows) };
    kernels.launch(kernels.get("argsort", "init_indices"), .{
        .grid = grid_size,
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ @as(?*anyopaque, temp_indices), ncols, nrows });

    // offset_iterator needs to populate nrows + 1 elements, so we also have
    // to ceildiv nrows + 1 by block_size
    const nrows_offset: c_int = nrows + 1;
    var offsets_alloc = common.PoolAlloc(c_int).initAlloc(pool, @as(usize, @bitCast(@as(isize, nrows_offset))));
    defer offsets_alloc.deinit();
    const offset_iterator = offsets_alloc.get().?;
    const offset_grid: kernels.Dim3 = .{ .x = @bitCast(@divTrunc(nrows_offset + block_size - 1, block_size)) };
    kernels.launch(kernels.get("argsort", "init_offsets"), .{
        .grid = offset_grid,
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ @as(?*anyopaque, offset_iterator), ncols, nrows });

    const nbytes: usize = @as(usize, @bitCast(@as(isize, ncols *% nrows))) *% @sizeOf(f32);
    common.cudaCheck(cu.cudaMemcpyAsync(temp_keys, x, nbytes, cu.cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream)", @src());

    var temp_storage_bytes: usize = 0;

    // Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not
    // support stream capture, while DeviceSegmentedRadixSort does.
    var capture_status: cu.cudaStreamCaptureStatus = undefined;
    common.cudaCheck(cu.cudaStreamIsCapturing(stream, &capture_status), "cudaStreamIsCapturing(stream, &capture_status)", @src());
    const is_capturing = capture_status != cu.cudaStreamCaptureStatusNone;

    // The C repeats the same six-way dispatch twice, once with a null
    // temporary buffer to size it and once to sort.
    const Dispatch = struct {
        fn run(d_temp_storage: ?*anyopaque, bytes: *usize, o: c.ggml_sort_order, n: c_int, cap: bool, keys: [*]f32, idx: [*]c_int, out: [*]c_int, nc: c_int, offs: [*]c_int, s: cu.cudaStream_t) void {
            if (o == c.GGML_SORT_ORDER_ASC) {
                if (n == 1) {
                    common.cudaCheck(ggml_cuda_cub_radix_sort_pairs(d_temp_storage, bytes, keys, idx, out, nc, s), "DeviceRadixSort::SortPairs(...)", @src());
                } else if (cap) {
                    common.cudaCheck(ggml_cuda_cub_segmented_radix_sort_pairs(d_temp_storage, bytes, keys, idx, out, nc, n, offs, s), "DeviceSegmentedRadixSort::SortPairs(...)", @src());
                } else {
                    common.cudaCheck(ggml_cuda_cub_segmented_sort_pairs(d_temp_storage, bytes, keys, idx, out, nc, n, offs, s), "DeviceSegmentedSort::SortPairs(...)", @src());
                }
            } else {
                if (n == 1) {
                    common.cudaCheck(ggml_cuda_cub_radix_sort_pairs_descending(d_temp_storage, bytes, keys, idx, out, nc, s), "DeviceRadixSort::SortPairsDescending(...)", @src());
                } else if (cap) {
                    common.cudaCheck(ggml_cuda_cub_segmented_radix_sort_pairs_descending(d_temp_storage, bytes, keys, idx, out, nc, n, offs, s), "DeviceSegmentedRadixSort::SortPairsDescending(...)", @src());
                } else {
                    common.cudaCheck(ggml_cuda_cub_segmented_sort_pairs_descending(d_temp_storage, bytes, keys, idx, out, nc, n, offs, s), "DeviceSegmentedSort::SortPairsDescending(...)", @src());
                }
            }
        }
    };

    Dispatch.run(null, &temp_storage_bytes, order, nrows, is_capturing, temp_keys, temp_indices, dst, ncols, offset_iterator, stream);

    var temp_storage_alloc = common.PoolAlloc(u8).initAlloc(pool, temp_storage_bytes);
    defer temp_storage_alloc.deinit();
    const d_temp_storage: ?*anyopaque = temp_storage_alloc.get();

    Dispatch.run(d_temp_storage, &temp_storage_bytes, order, nrows, is_capturing, temp_keys, temp_indices, dst, ncols, offset_iterator, stream);
}

// -----------------------------------------------------------------------------
// Bitonic

/// Ports `next_power_of_2` (argsort.cu:216 @b29c606e2).
pub fn nextPowerOf2(x: c_int) c_int {
    var n: c_int = 1;
    while (n < x) {
        n *%= 2;
    }
    return n;
}

/// Ports `argsort_f32_i32_cuda_bitonic` (argsort.cu:224 @b29c606e2).
pub fn argsortF32I32CudaBitonic(x: ?*const anyopaque, dst: ?*anyopaque, ncols: c_int, nrows: c_int, order: c.ggml_sort_order, stream: cu.cudaStream_t) void {
    // bitonic sort requires ncols to be power of 2
    const ncols_pad = nextPowerOf2(ncols);

    const block_dims: kernels.Dim3 = .{ .x = @bitCast(ncols_pad) };
    const block_nums: kernels.Dim3 = .{ .x = @bitCast(nrows) };
    const shared_mem: usize = @as(usize, @bitCast(@as(isize, ncols_pad))) *% @sizeOf(c_int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    impl.assert(shared_mem <= common.info().devices[@intCast(common.getDevice())].smpb, "shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb");

    const p: kernels.LaunchParams = .{ .grid = block_nums, .block = block_dims, .shmem = shared_mem, .stream = stream };
    const args = .{ x, dst, ncols, ncols_pad };
    if (order == c.GGML_SORT_ORDER_ASC) {
        kernels.launch(kernels.get("argsort", std.fmt.comptimePrint("k_argsort_f32_i32<(ggml_sort_order){d}>", .{c.GGML_SORT_ORDER_ASC})), p, args);
    } else if (order == c.GGML_SORT_ORDER_DESC) {
        kernels.launch(kernels.get("argsort", std.fmt.comptimePrint("k_argsort_f32_i32<(ggml_sort_order){d}>", .{c.GGML_SORT_ORDER_DESC})), p, args);
    } else {
        impl.abort("fatal error");
    }
}

// -----------------------------------------------------------------------------
// The op

/// Ports `ggml_cuda_op_argsort` (argsort.cu:251 @b29c606e2), the
/// `GGML_CUDA_USE_CUB` arm.
pub fn opArgsort(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    var src0_d: [*]const f32 = @ptrCast(@alignCast(src0.data));
    var dst_d: [*]f32 = @ptrCast(@alignCast(dst.data));
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_I32, " dst->type == GGML_TYPE_I32");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const ncols: i64 = src0.ne[0];
    const nrows: i64 = c.ggml_nrows(src0);

    const order: c.ggml_sort_order = @bitCast(impl.getOpParamsI32(dst, 0));

    const ncols_pad = nextPowerOf2(@truncate(ncols));
    const shared_mem: usize = @as(usize, @bitCast(@as(isize, ncols_pad))) *% @sizeOf(c_int);
    const max_shared_mem = common.info().devices[@intCast(common.getDevice())].smpb;

    // early return if we can use bitonic argsort
    if (shared_mem <= max_shared_mem and ncols <= 1024) {
        argsortF32I32CudaBitonic(src0_d, dst_d, @truncate(ncols), @truncate(nrows), order, stream);
        return;
    }

    const chunk_nrows = cubChunkNrows(src0.nb[1], nrows);

    const pool = ctx.pool();

    var i: i64 = 0;
    while (i < nrows) : (i += chunk_nrows) {
        const iter_nrows: c_int = @truncate(@min(@as(i64, chunk_nrows), nrows - i));

        argsortF32I32CudaCub(pool, src0_d, @ptrCast(dst_d), @truncate(ncols), iter_nrows, order, stream);

        src0_d += @as(usize, @bitCast(ncols * iter_nrows));
        dst_d += @as(usize, @bitCast(ncols * iter_nrows));
    }
}
