//! `GGML_OP_ARGMAX`: the index of each row's maximum.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/argmax.cu` at v0.4.1
//! (`b29c606e2`). The kernel, `argmax_f32`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `ggml_cuda_argmax` (argmax.cu:69 @b29c606e2), which launches with
/// `<<<>>>`.
pub fn opArgmax(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_I32, " dst->type == GGML_TYPE_I32");

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const ne00: i64 = src0.ne[0];
    const nrows: i64 = c.ggml_nrows(src0);

    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    const num_blocks: i64 = nrows;
    const num_threads: i64 = @min(1024, @divTrunc(ne00 + common.warp_size - 1, common.warp_size) * common.warp_size);
    const blocks_dim: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(num_threads))) };
    const blocks_num: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) };

    kernels.launch(kernels.get("argmax", "argmax_f32"), .{ .grid = blocks_num, .block = blocks_dim, .stream = stream }, .{ src0_d, dst_d, ne00 });
}
