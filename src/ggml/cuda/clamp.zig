//! `GGML_OP_CLAMP`: `dst = min(max(x, min), max)`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/clamp.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/clamp.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernel, `op_clamp_kernel`, stays CUDA.
//!
//! For `F16` the bounds are converted to `half` on the host, `(half)min`,
//! which rounds to nearest even exactly as Zig's `@floatCast` to `f16` does.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// Ports `CUDA_CLAMP_BLOCK_SIZE` (clamp.cuh:3 @c1d0e7a00).
const block_size = 256;

/// `(half) x`.
fn toHalf(x: f32) half {
    return @fromBackingInt(@intCast(@as(u16, @bitCast(@as(f16, @floatCast(x))))));
}

/// Ports `clamp_cuda` (clamp.cu:19 @c1d0e7a00), which launches with `<<<>>>`.
fn clampCuda(comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, min: T, max: T, k: c_int, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(k + block_size - 1, block_size);
    kernels.launch(kernels.get("clamp", kernels.tmpl("op_clamp_kernel", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ x, dst, min, max, k });
}

/// Ports `ggml_cuda_op_clamp` (clamp.cu:25 @c1d0e7a00).
pub fn opClamp(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");

    const min: f32 = @bitCast(dst.op_params[0]);
    const max: f32 = @bitCast(dst.op_params[1]);

    const k: c_int = @truncate(c.ggml_nelements(src0));
    if (src0.type == c.GGML_TYPE_F16) {
        clampCuda(half, src0_d, dst_d, toHalf(min), toHalf(max), k, stream);
    } else {
        clampCuda(f32, src0_d, dst_d, min, max, k, stream);
    }
}
