//! `SOFT_MAX` and `SOFT_MAX_BACK`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/softmax.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/softmax.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels -- `soft_max_f32`, `soft_max_f32_parallelize_cols`,
//! `soft_max_back_f32` -- stay CUDA.
//!
//! # Three routes
//!
//! A row that fits in shared memory takes `soft_max_f32<true, ncols,
//! block>`, specialised on the eight power-of-two widths from 32 to 4096
//! and generic otherwise. A row that does not fit takes either the
//! cooperative `soft_max_f32_parallelize_cols`, one grid across every SM
//! with a grid-wide barrier, or `soft_max_f32<false, 0, 0>`, which reads
//! the row from global memory.
//!
//! The C launches the cooperative kernel with `cudaLaunchCooperativeKernel`;
//! here it is `kernels.launchCooperative`, which calls the driver's
//! `cuLaunchCooperativeKernel` -- the same entry point with the same
//! arguments, one level down (see that function).

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;

const warp_size = common.warp_size;

extern fn powf(x: f32, y: f32) f32;
extern fn log2f(x: f32) f32;
extern fn floorf(x: f32) f32;

/// Ports `CUDA_SOFT_MAX_BLOCK_SIZE` (softmax.cuh:3 @c1d0e7a00).
const block_size = 1024;

/// Ports `soft_max_params` (softmax.cu:25 @c1d0e7a00). Passed to the
/// kernels by value, so the layout is the C's.
const Params = extern struct {
    nheads: i64,
    n_head_log2: u32,
    ncols: i64,
    nrows_x: i64,
    nrows_y: i64,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb11: i64,
    nb12: i64,
    nb13: i64,

    ne12: i64,
    ne13: i64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
};

comptime {
    std.debug.assert(@sizeOf(Params) == 128);
}

/// `soft_max_f32<use_shared, ncols, block, T>`.
fn softMaxKernel(comptime use_shared: bool, comptime ncols: comptime_int, comptime block: comptime_int, comptime T: type) kernels.Kernel {
    return kernels.get("softmax", kernels.tmpl("soft_max_f32", &.{ kernels.boolean(use_shared), kernels.int(ncols), kernels.int(block), kernels.typeName(T) }));
}

/// Ports `launch_soft_max_kernels` (softmax.cu:279 @c1d0e7a00),
/// instantiated with `<32, 64, 128, 256, 512, 1024, 2048, 4096>`.
fn launchSoftMaxKernels(comptime T: type, x: ?*const anyopaque, mask: ?*const anyopaque, sinks: ?*const anyopaque, dst: ?*anyopaque, p: *const Params, stream: cu.cudaStream_t, block_dims: kernels.Dim3, block_nums: kernels.Dim3, nbytes_shared: usize) void {
    const id = common.getDevice();
    const smpbo = common.info().devices[@intCast(id)].smpbo;

    // The C's unary fold over `launch_kernel`.
    inline for (.{ 32, 64, 128, 256, 512, 1024, 2048, 4096 }) |ncols| {
        const block = if (ncols > 1024) 1024 else ncols;
        if (p.ncols == ncols) {
            const k = comptime softMaxKernel(true, ncols, block, T);
            kernels.setSharedMemoryLimit(k, smpbo);
            kernels.launch(k, .{ .grid = block_nums, .block = block_dims, .shmem = nbytes_shared, .stream = stream }, .{ x, mask, sinks, dst, p.* });
            return;
        }
    }

    //default case
    const k = comptime softMaxKernel(true, 0, 0, T);
    kernels.setSharedMemoryLimit(k, smpbo);
    kernels.launch(k, .{ .grid = block_nums, .block = block_dims, .shmem = nbytes_shared, .stream = stream }, .{ x, mask, sinks, dst, p.* });
}

/// Ports `soft_max_f32_cuda` (softmax.cu:328 @c1d0e7a00).
fn softMaxF32Cuda(comptime T: type, x: ?*const anyopaque, mask: ?*const anyopaque, sinks: ?*const anyopaque, dst: ?*anyopaque, params: *const Params, stream: cu.cudaStream_t, ctx: *common.Context) void {
    var nth: c_int = warp_size;
    const ncols_x = params.ncols;

    while (nth < ncols_x and nth < block_size) nth *= 2;
    const block_dims: kernels.Dim3 = .{ .x = @bitCast(nth) };
    const block_nums: kernels.Dim3 = .{
        .x = @truncate(@as(u64, @bitCast(params.ne01))),
        .y = @truncate(@as(u64, @bitCast(params.ne02))),
        .z = @truncate(@as(u64, @bitCast(params.ne03))),
    };
    // GGML_PAD(ncols_x, WARP_SIZE)
    const padded: i64 = (ncols_x + warp_size - 1) & ~@as(i64, warp_size - 1);
    const nbytes_shared: usize = @as(usize, @bitCast(padded + warp_size)) *% @sizeOf(f32);
    comptime std.debug.assert(block_size == 1024); // These values need to be adjusted.

    const id = common.getDevice();
    const dev = &common.info().devices[@intCast(id)];
    const smpbo = dev.smpbo;

    if (nbytes_shared <= smpbo) {
        launchSoftMaxKernels(T, x, mask, sinks, dst, params, stream, block_dims, block_nums, nbytes_shared);
    } else {
        // Parallelize across SMs for top-p/dist-sampling
        if (dev.supports_cooperative_launch and
            @divTrunc(ncols_x, params.ne01 * params.ne02 * params.ne03) > 8192 and mask == null and sinks == null and
            params.scale == 1.0 and params.max_bias == 0.0)
        {
            // The C sizes both in *elements* as `nsm * sizeof(float)`.
            const n: usize = @as(usize, @intCast(dev.nsm)) * @sizeOf(f32);
            var tmp_maxs_alloc = common.PoolAlloc(f32).initAlloc(ctx.pool(), n);
            defer tmp_maxs_alloc.deinit();
            var tmp_sums_alloc = common.PoolAlloc(f32).initAlloc(ctx.pool(), n);
            defer tmp_sums_alloc.deinit();

            kernels.launchCooperative(kernels.get("softmax", "soft_max_f32_parallelize_cols"), .{
                .grid = .{ .x = @bitCast(dev.nsm) },
                .block = .{ .x = warp_size * 8 },
                .shmem = 0,
                .stream = stream,
            }, .{ x, dst, @as(?*anyopaque, @ptrCast(tmp_maxs_alloc.ptr)), @as(?*anyopaque, @ptrCast(tmp_sums_alloc.ptr)), params.* });
        } else {
            const nbytes_shared_low = warp_size * @sizeOf(f32);
            kernels.launch(softMaxKernel(false, 0, 0, T), .{
                .grid = block_nums,
                .block = block_dims,
                .shmem = nbytes_shared_low,
                .stream = stream,
            }, .{ x, mask, sinks, dst, params.* });
        }
    }
}

/// Ports `soft_max_back_f32_cuda` (softmax.cu:374 @c1d0e7a00).
fn softMaxBackF32Cuda(grad: ?*const anyopaque, dstf: ?*const anyopaque, dst: ?*anyopaque, ncols: c_int, nrows: c_int, scale: f32, stream: cu.cudaStream_t) void {
    const block_dims: kernels.Dim3 = .{ .x = warp_size };
    const block_nums: kernels.Dim3 = .{ .x = @bitCast(nrows) };

    kernels.launch(kernels.get("softmax", "soft_max_back_f32"), .{
        .grid = block_nums,
        .block = block_dims,
        .shmem = 0,
        .stream = stream,
    }, .{ grad, dstf, dst, ncols, scale });
}

/// Ports `ggml_cuda_op_soft_max` (softmax.cu:383 @c1d0e7a00).
pub fn opSoftMax(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1: ?*const Tensor = dst.src[1];
    const src2: ?*const Tensor = dst.src[2];

    const src1_d: ?*const anyopaque = if (src1) |t| t.data else null;
    const src2_d: ?*const anyopaque = if (src2) |t| t.data else null;

    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    impl.assert(src1 == null or src1.?.type == c.GGML_TYPE_F16 or src1.?.type == c.GGML_TYPE_F32, "!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32"); // src1 contains mask and it is optional

    const nrows_x = c.ggml_nrows(src0);
    const nrows_y = src0.ne[1];

    const ne00 = src0.ne[0];

    const scale = impl.getOpParamsF32(dst, 0);
    const max_bias = impl.getOpParamsF32(dst, 1);

    const use_f16 = src1 != null and src1.?.type == c.GGML_TYPE_F16;

    const nb11: i64 = if (src1) |t| @intCast(t.nb[1]) else 1;
    const nb12: i64 = if (src1) |t| @intCast(t.nb[2]) else 1;
    const nb13: i64 = if (src1) |t| @intCast(t.nb[3]) else 1;

    const ne12: i64 = if (src1) |t| t.ne[2] else 1;
    const ne13: i64 = if (src1) |t| t.ne[3] else 1;

    const n_head: u32 = @truncate(@as(u64, @bitCast(src0.ne[2])));
    const n_head_log2: u32 = @as(u32, 1) << @intFromFloat(floorf(log2f(@floatFromInt(n_head))));

    const m0 = powf(2.0, -(max_bias) / @as(f32, @floatFromInt(n_head_log2)));
    const m1 = powf(2.0, -(max_bias / 2.0) / @as(f32, @floatFromInt(n_head_log2)));

    var params: Params = std.mem.zeroes(Params);
    params.nheads = src0.ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0.ne[0];
    params.ne01 = src0.ne[1];
    params.ne02 = src0.ne[2];
    params.ne03 = src0.ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    if (use_f16) {
        softMaxF32Cuda(half, src0.data, src1_d, src2_d, dst.data, &params, stream, ctx);
    } else {
        softMaxF32Cuda(f32, src0.data, src1_d, src2_d, dst.data, &params, stream, ctx);
    }
}

/// Ports `ggml_cuda_op_soft_max_back` (softmax.cu:454 @c1d0e7a00).
pub fn opSoftMaxBack(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // grad
    const src1 = impl.one(Tensor, dst.src[1]); // forward pass output

    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const ncols = src0.ne[0];
    const nrows = c.ggml_nrows(src0);

    const scale = impl.getOpParamsF32(dst, 0);
    const max_bias = impl.getOpParamsF32(dst, 1);

    impl.assert(max_bias == 0.0, "max_bias == 0.0f");

    softMaxBackF32Cuda(src0.data, src1.data, dst.data, @truncate(ncols), @truncate(nrows), scale, stream);
}
