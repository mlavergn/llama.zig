//! The fused Snake activation, `y = x + sin^2(a * x) * inv_b`. There is no
//! `GGML_OP_SNAKE`: the compute loop's fusion recognises the op chain and
//! calls `opSnakeFused` with the operands it matched.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/snake.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/snake.cuh` at v0.4.1 (`b29c606e2`). The
//! kernel, `snake_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `launch_snake` (snake.cu:28 @b29c606e2). `x` is `[T, C]` with `T`
/// contiguous, `a` and `inv_b` are `[1, C]`.
fn launchSnake(ctx: *common.Context, x: *const Tensor, a: *const Tensor, inv_b: *const Tensor, dst: *Tensor) void {
    const a_d: ?*const anyopaque = a.data;
    const inv_b_d: ?*const anyopaque = inv_b.data;

    const T: c_int = @truncate(x.ne[0]);
    const C: c_int = @truncate(x.ne[1]);
    const total: c_int = T *% C;
    const T_len_fastdiv = common.initFastdivValues(@bitCast(@as(i64, T)));

    const block_size: c_int = 256;
    const grid_size: c_int = @divTrunc(total +% (block_size - 1), block_size);

    const stream = ctx.stream();

    const p: kernels.LaunchParams = .{ .grid = .{ .x = @bitCast(grid_size) }, .block = .{ .x = @bitCast(block_size) }, .shmem = 0, .stream = stream };
    const args = .{ @as(?*const anyopaque, x.data), a_d, inv_b_d, @as(?*anyopaque, dst.data), total, T_len_fastdiv };
    switch (x.type) {
        c.GGML_TYPE_F32 => kernels.launch(kernels.get("snake", "snake_kernel<float>"), p, args),
        c.GGML_TYPE_F16 => kernels.launch(kernels.get("snake", "snake_kernel<__half>"), p, args),
        c.GGML_TYPE_BF16 => kernels.launch(kernels.get("snake", "snake_kernel<__nv_bfloat16>"), p, args),
        else => impl.abort("snake: unsupported type"),
    }
}

/// Ports `ggml_cuda_op_snake_fused` (snake.cu:66 @b29c606e2): the fusion
/// entry point. The caller supplies `x`, `a` and `inv_b` explicitly.
pub fn opSnakeFused(ctx: *common.Context, x: *const Tensor, a: *const Tensor, inv_b: *const Tensor, dst: *Tensor) void {
    launchSnake(ctx, x, a, inv_b, dst);
}
