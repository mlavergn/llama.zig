//! `GGML_OP_CLAMP`: `dst = min(max(x, min), max)`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/clamp.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/clamp.cuh` at v0.8.0 (`8642541b2`). The
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

/// Ports `CUDA_CLAMP_BLOCK_SIZE` (clamp.cuh:3 @8642541b2).
const block_size = 256;

/// `(half) x`.
fn toHalf(x: f32) half {
    return @fromBackingInt(@intCast(@as(u16, @bitCast(@as(f16, @floatCast(x))))));
}

/// `size_t` into a `uint32_t` parameter.
fn u32Of(x: usize) u32 {
    return @truncate(x);
}

/// Ports `clamp_cuda` (clamp.cu:30 @8642541b2), which launches with `<<<>>>`.
/// Since v0.7.0 `src0` and `dst` may be views: rows are contiguous, dims
/// 1..3 follow the strides, in elements, and the kernel divides the flat
/// index with `fast_div_modulo`.
fn clampCuda(comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, min: T, max: T, src0: *const Tensor, t: *const Tensor, stream: cu.cudaStream_t) void {
    const k: i64 = c.ggml_nelements(src0);
    const ts: usize = @sizeOf(T);
    impl.assert(k <= std.math.maxInt(u32), "k <= std::numeric_limits<uint32_t>::max()");

    const ne0 = common.initFastdivValues(@bitCast(src0.ne[0]));
    const ne1 = common.initFastdivValues(@bitCast(src0.ne[1]));
    const ne2 = common.initFastdivValues(@bitCast(src0.ne[2]));

    const num_blocks: i64 = @divTrunc(k + block_size - 1, block_size);
    kernels.launch(kernels.get("clamp", kernels.tmpl("op_clamp_kernel", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{
        x,                      dst,                    min,                    max,                 @as(u32, @intCast(k)), ne0,                 ne1, ne2,
        u32Of(src0.nb[1] / ts), u32Of(src0.nb[2] / ts), u32Of(src0.nb[3] / ts), u32Of(t.nb[1] / ts), u32Of(t.nb[2] / ts),   u32Of(t.nb[3] / ts),
    });
}

/// Ports `ggml_cuda_op_clamp` (clamp.cu:46 @8642541b2).
pub fn opClamp(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");
    impl.assert(c.ggml_is_contiguous_rows(src0) and c.ggml_is_contiguous_rows(dst), "ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst)");

    const min: f32 = @bitCast(dst.op_params[0]);
    const max: f32 = @bitCast(dst.op_params[1]);

    if (src0.type == c.GGML_TYPE_F16) {
        clampCuda(half, src0_d, dst_d, toHalf(min), toHalf(max), src0, dst, stream);
    } else {
        clampCuda(f32, src0_d, dst_d, min, max, src0, dst, stream);
    }
}
