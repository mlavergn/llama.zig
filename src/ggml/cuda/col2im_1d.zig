//! `GGML_OP_COL2IM_1D`: fold 1-D columns back into a signal, summing
//! overlaps -- the inverse of `IM2COL`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/col2im-1d.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/col2im-1d.cuh` at v0.4.1 (`b29c606e2`).
//! The kernel, `col2im_1d_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `ggml_cuda_op_col2im_1d` (col2im-1d.cu:41 @b29c606e2).
pub fn opCol2im1d(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const s0 = impl.getOpParamsI32(dst, 0);
    const OC = impl.getOpParamsI32(dst, 1);
    const p0 = impl.getOpParamsI32(dst, 2);

    const K_OC: c_int = @truncate(src0.ne[0]);
    const T_in: c_int = @truncate(src0.ne[1]);
    const K: c_int = @divTrunc(K_OC, OC);
    const T_out: c_int = @truncate(dst.ne[0]);

    const T_out_fd = common.initFastdivValues(@as(u32, @bitCast(T_out)));

    const total: c_int = T_out *% OC;
    const block_size: c_int = 256;
    const num_blocks: c_int = @divTrunc(total +% (block_size - 1), block_size);

    const p: kernels.LaunchParams = .{ .grid = .{ .x = @bitCast(num_blocks) }, .block = .{ .x = @bitCast(block_size) }, .shmem = 0, .stream = stream };
    const args = .{ @as(?*const anyopaque, src0.data), @as(?*anyopaque, dst.data), T_in, T_out_fd, OC, K, K_OC, s0, p0, total };
    switch (src0.type) {
        c.GGML_TYPE_F32 => kernels.launch(kernels.get("col2im-1d", "col2im_1d_kernel<float>"), p, args),
        c.GGML_TYPE_F16 => kernels.launch(kernels.get("col2im-1d", "col2im_1d_kernel<__half>"), p, args),
        c.GGML_TYPE_BF16 => kernels.launch(kernels.get("col2im-1d", "col2im_1d_kernel<__nv_bfloat16>"), p, args),
        else => impl.abort("col2im_1d: unsupported type"),
    }
}
