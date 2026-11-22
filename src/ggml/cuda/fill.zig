//! `GGML_OP_FILL`: every element set to one value.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fill.cu` at v1.0.0
//! (`b1b770d90`). The kernel, `fill_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// Ports `CUDA_FILL_BLOCK_SIZE` (fill.cu:4 @b1b770d90).
const block_size = 256;

/// Ports `ggml_cuda_op_fill` (fill.cu:15 @b1b770d90), which launches with
/// `<<<>>>`.
pub fn opFill(ctx: *common.Context, dst: *Tensor) void {
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const value: f32 = @bitCast(dst.op_params[0]);

    const k: i64 = c.ggml_nelements(dst);
    const num_blocks: i64 = @divTrunc(k + block_size - 1, block_size);
    const p: kernels.LaunchParams = .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
        .block = .{ .x = block_size },
        .stream = stream,
    };

    switch (dst.type) {
        c.GGML_TYPE_F32 => kernels.launch(kernels.get("fill", "fill_kernel<float>"), p, .{ dst_d, k, value }),
        c.GGML_TYPE_F16 => kernels.launch(kernels.get("fill", "fill_kernel<__half>"), p, .{ dst_d, k, kernels.toHalf(value) }),
        else => impl.abort("unsupported type"),
    }
}
