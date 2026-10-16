//! `GGML_OP_OUT_PROD`: the outer product summed over the shared dimension,
//! by cuBLAS -- one GEMM, a strided batch, or a pointer batch whose
//! pointers a small kernel computes on the device.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/out-prod.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/out-prod.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernel, `k_compute_out_prod_ptrs`, stays CUDA.
//!
//! Like `ssm-scan.cu`'s SSD path, this calls cuBLAS through
//! `ctx.cublas_handle()` without a `cublasSetStream` of its own; the handle
//! was bound to the context's current stream when it was created.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// An `int64_t` into an `int` parameter.
fn int(x: i64) c_int {
    return @truncate(x);
}

/// `p + i3*s` with the C's `int64_t * size_t` arithmetic, which is
/// unsigned.
fn at(p: anytype, off: u64) @TypeOf(p) {
    return @ptrFromInt(@intFromPtr(p) +% off *% @sizeOf(f32));
}

fn u(x: i64) u64 {
    return @bitCast(x);
}

/// Ports `ggml_cuda_out_prod` (out-prod.cu:27 @c1d0e7a00).
pub fn outProd(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const nb01 = src0.nb[1];
    const nb02 = src0.nb[2];
    const nb03 = src0.nb[3];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const nb10 = src1.nb[0];
    const nb11 = src1.nb[1];
    const nb12 = src1.nb[2];
    const nb13 = src1.nb[3];
    const ne0 = dst.ne[0];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];
    const nb1 = dst.nb[1];
    const nb2 = dst.nb[2];
    const nb3 = dst.nb[3];

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type  == GGML_TYPE_F32");

    impl.assert(ne01 == ne11, "ne01 == ne11");
    impl.assert(ne0 == ne00, "ne0 == ne00");
    impl.assert(ne1 == ne10, "ne1 == ne10");

    impl.assert(@rem(ne2, src0.ne[2]) == 0, "ne2 % src0->ne[2] == 0");
    impl.assert(@rem(ne3, src0.ne[3]) == 0, "ne3 % src0->ne[3] == 0");

    impl.assert(ne2 == src1.ne[2], "ne2 == src1->ne[2]");
    impl.assert(ne3 == src1.ne[3], "ne3 == src1->ne[3]");

    const src0_d: [*c]const f32 = @ptrCast(@alignCast(src0.data));
    const src1_d: [*c]const f32 = @ptrCast(@alignCast(src1.data));
    const dst_d: [*c]f32 = @ptrCast(@alignCast(dst.data));

    const stream = ctx.stream();
    const handle = ctx.cublasHandle();

    const alpha: f32 = 1.0;
    const beta: f32 = 0.0;

    const lda: i64 = @bitCast(nb01 / @sizeOf(f32));
    const ldc: i64 = @bitCast(nb1 / @sizeOf(f32));

    const src1_T = c.ggml_is_transposed(src1);
    const src1_cublas_op: cu.cublasOperation_t = if (src1_T) cu.CUBLAS_OP_N else cu.CUBLAS_OP_T;
    const ldb: i64 = @bitCast((if (src1_T) nb10 else nb11) / @sizeOf(f32));
    impl.assert((if (src1_T) nb11 else nb10) == @sizeOf(f32), "(src1_T ?        nb11 :        nb10) == sizeof(float)");

    // data strides in dimensions 2/3
    const s02 = nb02 / @sizeOf(f32);
    const s03 = nb03 / @sizeOf(f32);
    const s12 = nb12 / @sizeOf(f32);
    const s13 = nb13 / @sizeOf(f32);
    const s2 = nb2 / @sizeOf(f32);
    const s3 = nb3 / @sizeOf(f32);

    // dps == dst per src0, used for group query attention
    const dps2 = @divTrunc(ne2, ne02);
    const dps3 = @divTrunc(ne3, ne03);

    const int_max: i64 = std.math.maxInt(c_int);
    if (dps2 == 1 and ne2 > 1) {
        // src0 has uniform stride s02 along dim 2; batch the inner loop with a strided GEMM
        impl.assert(ne2 <= int_max, "ne2 <= std::numeric_limits<int>::max()");
        const batch_count: c_int = @intCast(ne2);
        var j3: i64 = 0; // the C's `i3`; Zig reserves the name
        while (j3 < ne3) : (j3 += 1) {
            common.cublasCheck(cu.cublasSgemmStridedBatched(handle, cu.CUBLAS_OP_N, src1_cublas_op, int(ne0), int(ne1), int(ne01), &alpha, at(src0_d, u(@divTrunc(j3, dps3)) *% s03), int(lda), @bitCast(s02), at(src1_d, u(j3) *% s13), int(ldb), @bitCast(s12), &beta, at(dst_d, u(j3) *% s3), int(ldc), @bitCast(s2), batch_count), "cublasSgemmStridedBatched(handle, CUBLAS_OP_N, src1_cublas_op, ne0, ne1, ne01, &alpha, src0_d + (i3/dps3)*s03, lda, s02, src1_d +  i3     *s13, ldb, s12, &beta,  dst_d  +  i3     *s3,  ldc, s2, batch_count)", @src());
        }
    } else if (ne2 > 1 or ne3 > 1) {
        // dps2 > 1 (src0 broadcast along dim 2 with non-uniform stride) or multiple GEMMs
        // along dim 3: compute per-GEMM pointers on the device and use a single batched GEMM.
        impl.assert(ne3 > 0, "ne3 > 0");
        impl.assert(ne2 <= @divTrunc(int_max, ne3), "ne2 <= (int64_t) std::numeric_limits<int>::max() / ne3");
        const batch_count: c_int = @intCast(ne2 * ne3);

        var ptrs_a = common.PoolAlloc(?*const f32).initAlloc(ctx.pool(), @intCast(batch_count));
        defer ptrs_a.deinit();
        var ptrs_b = common.PoolAlloc(?*const f32).initAlloc(ctx.pool(), @intCast(batch_count));
        defer ptrs_b.deinit();
        var ptrs_c = common.PoolAlloc(?*f32).initAlloc(ctx.pool(), @intCast(batch_count));
        defer ptrs_c.deinit();

        const block_dims: kernels.Dim3 = .{ .x = 16, .y = 16 };
        const grid_dims: kernels.Dim3 = .{
            .x = @truncate(u(@divTrunc(ne2 + block_dims.x - 1, block_dims.x))),
            .y = @truncate(u(@divTrunc(ne3 + block_dims.y - 1, block_dims.y))),
        };
        kernels.launch(kernels.get("out-prod", "k_compute_out_prod_ptrs"), .{
            .grid = grid_dims,
            .block = block_dims,
            .shmem = 0,
            .stream = stream,
        }, .{
            @as(?*const anyopaque, src0_d),           @as(?*const anyopaque, src1_d),           @as(?*anyopaque, dst_d),
            @as(?*anyopaque, @ptrCast(ptrs_a.get())), @as(?*anyopaque, @ptrCast(ptrs_b.get())), @as(?*anyopaque, @ptrCast(ptrs_c.get())),
            ne2,                                      ne3,                                      dps2,
            dps3,                                     s02,                                      s03,
            s12,                                      s13,                                      s2,
            s3,
        });
        common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());

        common.cublasCheck(cu.cublasSgemmBatched(handle, cu.CUBLAS_OP_N, src1_cublas_op, int(ne0), int(ne1), int(ne01), &alpha, @ptrCast(ptrs_a.get()), int(lda), @ptrCast(ptrs_b.get()), int(ldb), &beta, @ptrCast(ptrs_c.get()), int(ldc), batch_count), "cublasSgemmBatched(handle, CUBLAS_OP_N, src1_cublas_op, ne0, ne1, ne01, &alpha, ptrs_a.get(), lda, ptrs_b.get(), ldb, &beta,  ptrs_c.get(), ldc, batch_count)", @src());
    } else {
        // ne2 == 1 && ne3 == 1: single GEMM
        common.cublasCheck(cu.cublasSgemm_v2(handle, cu.CUBLAS_OP_N, src1_cublas_op, int(ne0), int(ne1), int(ne01), &alpha, src0_d, int(lda), src1_d, int(ldb), &beta, dst_d, int(ldc)), "cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op, ne0, ne1, ne01, &alpha, src0_d, lda, src1_d, ldb, &beta,  dst_d,  ldc)", @src());
    }
}
