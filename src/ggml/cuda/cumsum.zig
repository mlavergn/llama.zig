//! `GGML_OP_CUMSUM`: the inclusive prefix sum of every row.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/cumsum.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/cumsum.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels, `cumsum_cub_kernel` and `cumsum_kernel`, stay CUDA.
//!
//! # CUB
//!
//! `GGML_CUDA_USE_CUB` is defined for this build (common.cuh:111 @c1d0e7a00). Long
//! contiguous rows go to `cub::DeviceScan::InclusiveSum`, whose host-side
//! dispatch is CUB's header templates; `cub_shim.cu` wraps that one call
//! behind a C ABI (PLAN.md, Decision 47). Only `float` is instantiated:
//! upstream comments out the `F16` and `BF16` cases.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_CUMSUM_BLOCK_SIZE` (cumsum.cuh:3 @c1d0e7a00).
const block_size = 256;

/// `cub_shim.cu`'s wrapper of `cumsum_cub<float>`'s `InclusiveSum`.
extern fn ggml_cuda_cub_inclusive_sum_f32(d_temp_storage: ?*anyopaque, tmp_size: *usize, src: [*]const f32, dst: [*]f32, ne: i64, stream: cu.cudaStream_t) cu.cudaError_t;

/// A `dim3` member from an `int64_t`: C++ converts modulo 2^32.
fn dimOf(v: i64) u32 {
    return @truncate(@as(u64, @bitCast(v)));
}

/// `int64_t / size_t`: the C converts the signed side, divides unsigned,
/// and the quotient goes back into an `int64_t` parameter.
fn strideOf(nb: i64, ts: usize) i64 {
    return @bitCast(@as(u64, @bitCast(nb)) / ts);
}

/// Ports `cumsum_cub` (cumsum.cu:190 @c1d0e7a00) for `T = float`. Upstream
/// checks neither call's result, and neither does this.
fn cumsumCub(pool: *common.Pool, src: [*]const f32, dst: [*]f32, ne: i64, stream: cu.cudaStream_t) void {
    var tmp_size: usize = 0;

    // Query how much temp storage CUDA UnBound (CUB) needs
    _ = ggml_cuda_cub_inclusive_sum_f32(null, &tmp_size, src, dst, ne, stream);

    var tmp_alloc = common.PoolAlloc(u8).initAlloc(pool, tmp_size);
    defer tmp_alloc.deinit();

    // Perform the inclusive scan
    _ = ggml_cuda_cub_inclusive_sum_f32(tmp_alloc.get(), &tmp_size, src, dst, ne, stream);
}

/// Ports `cumsum_cuda` (cumsum.cu:214 @c1d0e7a00) for `T = float`.
fn cumsumCuda(
    ctx: *common.Context,
    src: [*]const f32,
    dst: [*]f32,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: i64,
    nb01: i64,
    nb02: i64,
    nb03: i64,
    nb0: i64,
    nb1: i64,
    nb2: i64,
    nb3: i64,
    stream: cu.cudaStream_t,
) void {
    const type_size: usize = @sizeOf(f32);
    var use_cub = false;

    // Check if we can use CUB (data must be contiguous along innermost
    // dimension)
    const is_contiguous = (@as(u64, @bitCast(nb00)) == type_size) and (@as(u64, @bitCast(nb0)) == type_size);

    if (is_contiguous) {
        use_cub = true;
        const nrows: i64 = ne01 * ne02 * ne03;
        // Heuristics were determined as part of
        // https://github.com/ggml-org/llama.cpp/pull/17004
        if (((nrows == 1) and (ne00 > 1024)) or (@divTrunc(ne00, nrows) > 4096)) {
            var i: c_int = 0;
            while (i < nrows) : (i += 1) {
                const off: usize = @bitCast(@as(i64, i) * ne00);
                cumsumCub(ctx.pool(), src + off, dst + off, ne00, stream);
            }
            return;
        }
    }

    const grid_dims: kernels.Dim3 = .{ .x = dimOf(ne01), .y = dimOf(ne02), .z = dimOf(ne03) };
    const info = &common.info().devices[@intCast(common.getDevice())];
    const warp_size: c_int = info.warp_size;
    const num_warps: c_int = @truncate(@divTrunc(ne00 + warp_size - 1, warp_size));
    var bs: c_int = num_warps * warp_size;
    bs = @min(bs, block_size);
    const block_dims: kernels.Dim3 = .{ .x = @bitCast(bs) };
    const warps_per_block: c_int = @divTrunc(bs, warp_size);
    const shmem_size: usize = @as(usize, @bitCast(@as(isize, bs + warps_per_block + 2))) *% @sizeOf(f32);

    if (use_cub and ne00 >= 1024) {
        kernels.launch(kernels.get("cumsum", "cumsum_cub_kernel<float, (int)256>"), .{
            .grid = grid_dims,
            .block = .{ .x = block_size },
            .stream = stream,
        }, .{
            @as(?*const anyopaque, src), @as(?*anyopaque, dst),
            ne00,                        ne01,
            ne02,                        ne03,
            strideOf(nb01, type_size),   strideOf(nb02, type_size),
            strideOf(nb03, type_size),   strideOf(nb1, type_size),
            strideOf(nb2, type_size),    strideOf(nb3, type_size),
        });
    } else {
        kernels.launch(kernels.get("cumsum", "cumsum_kernel<float>"), .{
            .grid = grid_dims,
            .block = block_dims,
            .shmem = shmem_size,
            .stream = stream,
        }, .{
            @as(?*const anyopaque, src), @as(?*anyopaque, dst),
            ne00,                        ne01,
            ne02,                        ne03,
            strideOf(nb00, type_size),   strideOf(nb01, type_size),
            strideOf(nb02, type_size),   strideOf(nb03, type_size),
            strideOf(nb0, type_size),    strideOf(nb1, type_size),
            strideOf(nb2, type_size),    strideOf(nb3, type_size),
        });
    }
}

/// Ports `ggml_cuda_op_cumsum` (cumsum.cu:267 @c1d0e7a00).
pub fn opCumsum(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const stream = ctx.stream();

    impl.assert(src0.type == dst.type, "src0->type == dst->type");
    switch (src0.type) {
        c.GGML_TYPE_F32 => {
            const nb = struct {
                fn f(t: *const Tensor, i: usize) i64 {
                    return @bitCast(@as(u64, t.nb[i]));
                }
            }.f;
            cumsumCuda(
                ctx,
                @ptrCast(@alignCast(src0.data)),
                @ptrCast(@alignCast(dst.data)),
                src0.ne[0],
                src0.ne[1],
                src0.ne[2],
                src0.ne[3],
                nb(src0, 0),
                nb(src0, 1),
                nb(src0, 2),
                nb(src0, 3),
                nb(dst, 0),
                nb(dst, 1),
                nb(dst, 2),
                nb(dst, 3),
                stream,
            );
        },
        // F16 and BF16 are commented out upstream ("We do not support those
        // on CPU for now anyway").
        else => impl.abort("fatal error"),
    }
}
