//! `GGML_OP_CLAMP`: `dst = min(max(x, min), max)`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/clamp.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/clamp.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `op_clamp_kernel`, stays CUDA.
//!
//! For `F16` the bounds are converted to `half` on the host, `(half)min`,
//! which rounds to nearest even exactly as Zig's `@floatCast` to `f16` does.

const std = @import("std");
const ggml = @import("../module.zig");

/// Ports `CUDA_CLAMP_BLOCK_SIZE` (clamp.cuh:3 @b1b770d90).
const block_size = 256;

/// `size_t` into a `uint32_t` parameter.
fn u32Of(x: usize) u32 {
    return @truncate(x);
}

/// Ports `clamp_cuda` (clamp.cu:30 @b1b770d90), which launches with `<<<>>>`.
/// `src0` and `dst` may be views: rows are contiguous, dims
/// 1..3 follow the strides, in elements, and the kernel divides the flat
/// index with `fast_div_modulo`.
fn clampCuda(comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, min: T, max: T, src0: *const ggml.impl.ggml_c.ggml_tensor, t: *const ggml.impl.ggml_c.ggml_tensor, stream: ggml.cuda.common.cuda_c.cudaStream_t) void {
    const k: i64 = ggml.types.ggml_nelements(src0);
    const ts: usize = @sizeOf(T);
    ggml.impl.assert(k <= std.math.maxInt(u32), "k <= std::numeric_limits<uint32_t>::max()");

    const ne0 = ggml.cuda.common.initFastdivValues(@bitCast(src0.ne[0]));
    const ne1 = ggml.cuda.common.initFastdivValues(@bitCast(src0.ne[1]));
    const ne2 = ggml.cuda.common.initFastdivValues(@bitCast(src0.ne[2]));

    const num_blocks: i64 = @divTrunc(k + block_size - 1, block_size);
    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("clamp", ggml.cuda.kernels.tmpl("op_clamp_kernel", &.{ggml.cuda.kernels.typeName(T)})), .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{
        x,                      dst,                    min,                    max,                 @as(u32, @intCast(k)), ne0,                 ne1, ne2,
        u32Of(src0.nb[1] / ts), u32Of(src0.nb[2] / ts), u32Of(src0.nb[3] / ts), u32Of(t.nb[1] / ts), u32Of(t.nb[2] / ts),   u32Of(t.nb[3] / ts),
    });
}

/// Ports `ggml_cuda_op_clamp` (clamp.cu:46 @b1b770d90).
pub fn opClamp(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32 or dst.type == ggml.impl.ggml_c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    ggml.impl.assert(src0.type == dst.type, "src0->type == dst->type");
    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(src0) and ggml.types.ggml_is_contiguous_rows(dst), "ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst)");

    const min: f32 = @bitCast(dst.op_params[0]);
    const max: f32 = @bitCast(dst.op_params[1]);

    if (src0.type == ggml.impl.ggml_c.GGML_TYPE_F16) {
        clampCuda(ggml.cuda.kernels.half, src0_d, dst_d, ggml.cuda.kernels.toHalf(min), ggml.cuda.kernels.toHalf(max), src0, dst, stream);
    } else {
        clampCuda(f32, src0_d, dst_d, min, max, src0, dst, stream);
    }
}
