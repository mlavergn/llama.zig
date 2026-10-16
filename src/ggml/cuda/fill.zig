//! `GGML_OP_FILL`: every element set to one value.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fill.cu` at v0.3.0
//! (`c1d0e7a00`). The kernel, `fill_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// Ports `CUDA_FILL_BLOCK_SIZE` (fill.cu:4 @c1d0e7a00).
const block_size = 256;

/// `ggml_cuda_cast<half>(x)`: `__float2half`, round to nearest even.
fn toHalf(x: f32) half {
    return @fromBackingInt(@intCast(@as(u16, @bitCast(@as(f16, @floatCast(x))))));
}

/// Ports `ggml_cuda_op_fill` (fill.cu:15 @c1d0e7a00), which launches with
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
        c.GGML_TYPE_F16 => kernels.launch(kernels.get("fill", "fill_kernel<__half>"), p, .{ dst_d, k, toHalf(value) }),
        else => impl.abort("unsupported type"),
    }
}
