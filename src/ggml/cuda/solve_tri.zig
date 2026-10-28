//! `GGML_OP_SOLVE_TRI`: batched triangular solves, by a warp kernel for
//! small systems and cuBLAS's `trsmBatched` otherwise.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/solve_tri.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/solve_tri.cuh` at v0.4.0 (`5266f24da`).
//! The kernels, `get_batch_pointers` and `solve_tri_f32_fast`, stay CUDA.
//!
//! `ggml_cuda_op_solve_tri` calls `ggml_is_contiguous` on both sources and
//! discards the answers; the port keeps the calls and asserts nothing.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `MAX_N_FAST` (solve_tri.cu:5 @5266f24da).
const max_n_fast = 64;
/// Ports `MAX_K_FAST` (solve_tri.cu:6 @5266f24da).
const max_k_fast = 32;

/// An `int64_t` into a `dim3` member, which is `unsigned int`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `solve_tri_f32_cublas` (solve_tri.cu:30 @5266f24da).
fn solveTriF32Cublas(
    ctx: *common.Context,
    A: ?*const anyopaque,
    B: ?*const anyopaque,
    X: ?*anyopaque,
    n: c_int,
    k: c_int,
    ne02: i64,
    ne03: i64,
    s02: usize,
    s03: usize,
    s12: usize,
    s13: usize,
    s2: usize,
    s3: usize,
    stream: cu.cudaStream_t,
) void {
    _ = s12;
    _ = s13;
    const alpha: f32 = 1.0;
    const total_batches = ne02 * ne03;
    if (total_batches == 0) {
        return;
    }

    // Bulk copy B -> X (contiguous tensors)
    if (@as(?*const anyopaque, X) != B) {
        const total_elements_BX: i64 = @as(i64, n *% k) * total_batches;
        common.cudaCheck(cu.cudaMemcpyAsync(X, B, @as(usize, @bitCast(total_elements_BX)) * @sizeOf(f32), cu.cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync(X, B, total_elements_BX * sizeof(float), cudaMemcpyDeviceToDevice, stream)", @src());
    }

    const id = common.getDevice();

    var A_ptrs_alloc = common.PoolAlloc(?*const f32).initAlloc(ctx.poolFor(id), @bitCast(total_batches));
    defer A_ptrs_alloc.deinit();
    var X_ptrs_alloc = common.PoolAlloc(?*f32).initAlloc(ctx.poolFor(id), @bitCast(total_batches));
    defer X_ptrs_alloc.deinit();

    const A_ptrs_dev = A_ptrs_alloc.get().?;
    const X_ptrs_dev = X_ptrs_alloc.get().?;

    kernels.launch(kernels.get("solve_tri", "get_batch_pointers"), .{
        .grid = .{ .x = dim(@divTrunc(total_batches + 255, 256)) },
        .block = .{ .x = 256 },
        .shmem = 0,
        .stream = stream,
    }, .{ A, X, @as(?*anyopaque, @ptrCast(A_ptrs_dev)), @as(?*anyopaque, @ptrCast(X_ptrs_dev)), ne02, total_batches, s02, s03, s2, s3 });

    // Yes, this is necessary, without this we get RMSE errors
    common.cublasCheck(cu.cublasSetMathMode(ctx.cublasHandle(), cu.CUBLAS_DEFAULT_MATH), "cublasSetMathMode(ctx.cublas_handle(), CUBLAS_DEFAULT_MATH)", @src());
    common.cublasCheck(cu.cublasStrsmBatched(ctx.cublasHandle(), cu.CUBLAS_SIDE_RIGHT, cu.CUBLAS_FILL_MODE_UPPER, cu.CUBLAS_OP_N, cu.CUBLAS_DIAG_NON_UNIT, k, n, &alpha, @ptrCast(A_ptrs_dev), n, @ptrCast(X_ptrs_dev), k, @truncate(total_batches)), "cublasStrsmBatched(ctx.cublas_handle(), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, k, n, &alpha, A_ptrs_dev, n, X_ptrs_dev, k, total_batches)", @src());

    // revert to standard mode from common.cuh
    common.cublasCheck(cu.cublasSetMathMode(ctx.cublasHandle(), cu.CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(ctx.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH)", @src());
}

fn fastKernel(comptime n: comptime_int, comptime k: comptime_int) kernels.Kernel {
    return kernels.get("solve_tri", kernels.tmpl("solve_tri_f32_fast", &.{ kernels.int(n), kernels.int(k) }));
}

/// Ports `solve_tri_f32_cuda` (solve_tri.cu:181 @5266f24da).
fn solveTriF32Cuda(
    A: ?*const anyopaque,
    B: ?*const anyopaque,
    X: ?*anyopaque,
    n: c_int,
    k: c_int,
    ne02: i64,
    ne03: i64,
    nb02: usize,
    nb03: usize,
    nb12: usize,
    nb13: usize,
    nb2: usize,
    nb3: usize,
    stream: cu.cudaStream_t,
) void {
    const ne02_fd = common.initFastdivValues(@as(u32, @truncate(@as(u64, @bitCast(ne02)))));
    const threads: kernels.Dim3 = .{ .x = common.warp_size, .y = @bitCast(k) };
    const grid: kernels.Dim3 = .{ .x = dim(ne02 * ne03) };
    const p: kernels.LaunchParams = .{ .grid = grid, .block = threads, .shmem = 0, .stream = stream };
    const general = .{ A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k };
    // Typed, so the two zeros are runtime fields like the rest.
    const fixed: @TypeOf(general) = .{ A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0 };
    if (n == 64) {
        switch (k) {
            inline 32, 16, 14, 12, 10, 8, 6, 4, 2, 1 => |kk| kernels.launch(fastKernel(64, kk), p, fixed),
            else => kernels.launch(fastKernel(0, 0), p, general),
        }
    } else { // run general case
        kernels.launch(fastKernel(0, 0), p, general);
    }
}

/// Ports `ggml_cuda_op_solve_tri` (solve_tri.cu:250 @5266f24da).
pub fn opSolveTri(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // A (n×n, lower triangular)
    const src1 = impl.one(Tensor, dst.src[1]); // B (n×k)

    _ = c.ggml_is_contiguous(src0);
    _ = c.ggml_is_contiguous(src1);

    const n = src0.ne[0];
    const k = src1.ne[0];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];

    const f = @sizeOf(f32);
    if (n <= max_n_fast and k <= max_k_fast) {
        solveTriF32Cuda(src0.data, src1.data, dst.data, @truncate(n), @truncate(k), src0.ne[2], src0.ne[3], src0.nb[2] / f, src0.nb[3] / f, src1.nb[2] / f, src1.nb[3] / f, dst.nb[2] / f, dst.nb[3] / f, ctx.stream());
    } else {
        solveTriF32Cublas(ctx, src0.data, src1.data, dst.data, @truncate(n), @truncate(k), ne02, ne03, src0.nb[2] / f, src0.nb[3] / f, src1.nb[2] / f, src1.nb[3] / f, dst.nb[2] / f, dst.nb[3] / f, ctx.stream());
    }
}
