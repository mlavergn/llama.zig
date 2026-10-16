//! `GGML_OP_DIAG_MASK_INF`: the causal mask, `-FLT_MAX` added past
//! `n_past + row`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/diagmask.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/diagmask.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernel, `diag_mask_inf_f32`, stays CUDA.
//!
//! The block is `(1, 32, 1)` and the grid `(nrows, ceil(ncols / 32), 1)`:
//! rows on x, columns on y.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_DIAG_MASK_INF_BLOCK_SIZE` (diagmask.cuh:3 @c1d0e7a00).
const block_size = 32;

/// Ports `diag_mask_inf_f32_cuda` (diagmask.cu:17 @c1d0e7a00), which
/// launches with `<<<>>>`.
fn diagMaskInfF32Cuda(x: ?*const anyopaque, dst: ?*anyopaque, ncols_x: c_int, nrows_x: c_int, rows_per_channel: c_int, n_past: c_int, stream: cu.cudaStream_t) void {
    const block_dims: kernels.Dim3 = .{ .x = 1, .y = block_size, .z = 1 };
    const block_num_x: c_int = @divTrunc(ncols_x + block_size - 1, block_size);
    const block_nums: kernels.Dim3 = .{ .x = @bitCast(nrows_x), .y = @bitCast(block_num_x), .z = 1 };
    kernels.launch(kernels.get("diagmask", "diag_mask_inf_f32"), .{ .grid = block_nums, .block = block_dims, .stream = stream }, .{ x, dst, ncols_x, rows_per_channel, n_past });
}

/// Ports `ggml_cuda_op_diag_mask_inf` (diagmask.cu:24 @c1d0e7a00).
pub fn opDiagMaskInf(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const ne00: i64 = src0.ne[0];
    const ne01: i64 = src0.ne[1];
    const nrows0: c_int = @truncate(c.ggml_nrows(src0));

    const n_past: c_int = dst.op_params[0];

    diagMaskInfF32Cuda(src0_d, dst_d, @truncate(ne00), nrows0, @truncate(ne01), n_past, stream);
}
