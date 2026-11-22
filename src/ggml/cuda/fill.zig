//! `GGML_OP_FILL`: every element set to one value.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fill.cu` at v1.0.0
//! (`b1b770d90`). The kernel, `fill_kernel`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_FILL_BLOCK_SIZE` (fill.cu:4 @b1b770d90).
const block_size = 256;

/// Ports `ggml_cuda_op_fill` (fill.cu:15 @b1b770d90), which launches with
/// `<<<>>>`.
pub fn opFill(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const value: f32 = @bitCast(dst.op_params[0]);

    const k: i64 = ggml.impl.ggml_c.ggml_nelements(dst);
    const num_blocks: i64 = @divTrunc(k + block_size - 1, block_size);
    const p: ggml.cuda.kernels.LaunchParams = .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
        .block = .{ .x = block_size },
        .stream = stream,
    };

    switch (dst.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => ggml.cuda.kernels.launch(ggml.cuda.kernels.get("fill", "fill_kernel<float>"), p, .{ dst_d, k, value }),
        ggml.impl.ggml_c.GGML_TYPE_F16 => ggml.cuda.kernels.launch(ggml.cuda.kernels.get("fill", "fill_kernel<__half>"), p, .{ dst_d, k, ggml.cuda.kernels.toHalf(value) }),
        else => ggml.impl.abort("unsupported type"),
    }
}
