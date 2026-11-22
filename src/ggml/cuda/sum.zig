//! `GGML_OP_SUM`: the sum of every element, through CUB's device-wide
//! reduction.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/sum.cu` at v1.0.0 (`b1b770d90`),
//! the `GGML_CUDA_USE_CUB` arm. `cub::DeviceReduce::Sum` is reached through
//! `cub_shim.cu`.

const ggml = @import("../module.zig");
/// `DeviceReduce::Sum(d_temp_storage, *tmp_size, x, dst, ne, stream)`; see
/// `cub_shim.cu`.
pub extern fn ggml_cuda_cub_reduce_sum_f32(d_temp_storage: ?*anyopaque, tmp_size: *usize, x: ?*const f32, dst: ?*f32, ne: i64, stream: ggml.cuda.common.cu.cudaStream_t) ggml.cuda.common.cu.cudaError_t;

/// Ports `sum_f32_cuda` (sum.cu:11 @b1b770d90). Upstream ignores both of
/// CUB's return values, and so does this.
///
/// Parameters:
/// - `pool`: the device's pool, for CUB's temporary storage.
/// - `x`, `dst`: `ne` floats in, one float out.
/// - `stream`: the stream the reduction runs on.
pub fn sumF32Cuda(pool: *ggml.cuda.common.Pool, x: ?*const f32, dst: ?*f32, ne: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    var tmp_size: usize = 0;
    _ = ggml_cuda_cub_reduce_sum_f32(null, &tmp_size, x, dst, ne, stream);
    var tmp_alloc = ggml.cuda.common.PoolAlloc(u8).initAlloc(pool, tmp_size);
    defer tmp_alloc.deinit();
    _ = ggml_cuda_cub_reduce_sum_f32(tmp_alloc.get(), &tmp_size, x, dst, ne, stream);
}

/// Ports `ggml_cuda_op_sum` (sum.cu:25 @b1b770d90).
pub fn opSum(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguously_allocated(src0), "ggml_is_contiguously_allocated(src0)");

    const src0_d: ?*const f32 = @ptrCast(@alignCast(src0.data));
    const dst_d: ?*f32 = @ptrCast(@alignCast(dst.data));

    const ne = ggml.impl.ggml_c.ggml_nelements(src0);

    sumF32Cuda(ctx.pool(), src0_d, dst_d, ne, ctx.stream());
}
