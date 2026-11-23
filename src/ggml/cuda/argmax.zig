//! `GGML_OP_ARGMAX`: the index of each row's maximum.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/argmax.cu` at v1.0.0
//! (`b1b770d90`). The kernel, `argmax_f32`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `ggml_cuda_argmax` (argmax.cu:69 @b1b770d90), which launches with
/// `<<<>>>`.
pub fn opArgmax(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_I32, " dst->type == GGML_TYPE_I32");

    ggml.impl.assert(ggml.types.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const ne00: i64 = src0.ne[0];
    const nrows: i64 = ggml.types.ggml_nrows(src0);

    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    const num_blocks: i64 = nrows;
    const num_threads: i64 = @min(1024, @divTrunc(ne00 + ggml.cuda.common.warp_size - 1, ggml.cuda.common.warp_size) * ggml.cuda.common.warp_size);
    const blocks_dim: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(num_threads))) };
    const blocks_num: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) };

    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("argmax", "argmax_f32"), .{ .grid = blocks_num, .block = blocks_dim, .stream = stream }, .{ src0_d, dst_d, ne00 });
}
