//! `GGML_OP_SCALE`: `dst = scale * x + bias`, in F32 or (since v0.6.0) BF16.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/scale.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/scale.cuh` at v0.8.0 (`8642541b2`). The
//! kernel, `scale_f32`, stays CUDA and is launched from here; since v0.6.0
//! it is a template over the element type, and computes in F32 either way.
//! Since v0.8.0 `src` and `dst` may be views: rows are contiguous, dimensions
//! 1 to 3 follow the strides.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_SCALE_BLOCK_SIZE` (scale.cuh:3 @cd959bae9).
const block_size = 256;

/// Ports `MAX_GRIDDIM_X` (scale.cu:4 @cd959bae9).
const max_griddim_x = 0x7FFFFFFF;

/// A byte stride in elements, into the kernel's `uint32_t` parameter.
fn elems(nb: usize, ts: usize) u32 {
    return @truncate(nb / ts);
}

/// Ports `scale_f32_cuda` (scale.cu:31 @cd959bae9), its element type `T`
/// comptime: `f32` or `kernels.bf16`.
fn scaleF32Cuda(comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, scale: f32, bias: f32, src0: *const Tensor, t: *const Tensor, stream: common.cu.cudaStream_t) void {
    const k = c.ggml_nelements(src0);
    const ts: usize = @sizeOf(T);
    impl.assert(k <= std.math.maxInt(u32), "k <= std::numeric_limits<uint32_t>::max()");

    const ne0 = common.initFastdivValues(@bitCast(src0.ne[0]));
    const ne1 = common.initFastdivValues(@bitCast(src0.ne[1]));
    const ne2 = common.initFastdivValues(@bitCast(src0.ne[2]));

    const num_blocks = @divTrunc(k + block_size - 1, block_size);
    kernels.launchPdl(kernels.get("scale", kernels.tmpl("scale_f32", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = @intCast(@min(max_griddim_x, num_blocks)) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{
        x,                     dst,                   scale,                 bias,
        @as(u32, @intCast(k)), ne0,                   ne1,                   ne2,
        elems(src0.nb[1], ts), elems(src0.nb[2], ts), elems(src0.nb[3], ts), elems(t.nb[1], ts),
        elems(t.nb[2], ts),    elems(t.nb[3], ts),
    });
}

/// Ports `ggml_cuda_op_scale` (scale.cu:47 @cd959bae9).
pub fn opScale(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_BF16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16");
    impl.assert(dst.type == src0.type, "dst->type == src0->type");
    impl.assert(c.ggml_is_contiguous_rows(src0) and c.ggml_is_contiguous_rows(dst), "ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst)");

    const scale = impl.getOpParamsF32(dst, 0);
    const bias = impl.getOpParamsF32(dst, 1);

    if (src0.type == c.GGML_TYPE_BF16) {
        scaleF32Cuda(kernels.bf16, src0.data, dst.data, scale, bias, src0, dst, stream);
    } else {
        scaleF32Cuda(f32, src0.data, dst.data, scale, bias, src0, dst, stream);
    }
}
