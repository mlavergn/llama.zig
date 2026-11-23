//! `GGML_OP_SCALE`: `dst = scale * x + bias`, in F32 or BF16.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/scale.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/scale.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `scale_f32`, stays CUDA and is launched from here;
//! it is a template over the element type, and computes in F32 either way.
//! `src` and `dst` may be views: rows are contiguous, dimensions
//! 1 to 3 follow the strides.

const std = @import("std");
const ggml = @import("../module.zig");

/// Ports `CUDA_SCALE_BLOCK_SIZE` (scale.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `MAX_GRIDDIM_X` (scale.cu:4 @b1b770d90).
const max_griddim_x = 0x7FFFFFFF;

/// A byte stride in elements, into the kernel's `uint32_t` parameter.
fn elems(nb: usize, ts: usize) u32 {
    return @truncate(nb / ts);
}

/// Ports `scale_f32_cuda` (scale.cu:31 @b1b770d90), its element type `T`
/// comptime: `f32` or `kernels.bf16`.
fn scaleF32Cuda(comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, scale: f32, bias: f32, src0: *const ggml.impl.ggml_c.ggml_tensor, t: *const ggml.impl.ggml_c.ggml_tensor, stream: ggml.cuda.common.cuda_c.cudaStream_t) void {
    const k = ggml.types.ggml_nelements(src0);
    const ts: usize = @sizeOf(T);
    ggml.impl.assert(k <= std.math.maxInt(u32), "k <= std::numeric_limits<uint32_t>::max()");

    const ne0 = ggml.cuda.common.initFastdivValues(@bitCast(src0.ne[0]));
    const ne1 = ggml.cuda.common.initFastdivValues(@bitCast(src0.ne[1]));
    const ne2 = ggml.cuda.common.initFastdivValues(@bitCast(src0.ne[2]));

    const num_blocks = @divTrunc(k + block_size - 1, block_size);
    ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("scale", ggml.cuda.kernels.tmpl("scale_f32", &.{ggml.cuda.kernels.typeName(T)})), .{
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

/// Ports `ggml_cuda_op_scale` (scale.cu:47 @b1b770d90).
pub fn opScale(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const stream = ctx.stream();

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src0.type == ggml.impl.ggml_c.GGML_TYPE_BF16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16");
    ggml.impl.assert(dst.type == src0.type, "dst->type == src0->type");
    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(src0) and ggml.types.ggml_is_contiguous_rows(dst), "ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst)");

    const scale = ggml.impl.getOpParamsF32(dst, 0);
    const bias = ggml.impl.getOpParamsF32(dst, 1);

    if (src0.type == ggml.impl.ggml_c.GGML_TYPE_BF16) {
        scaleF32Cuda(ggml.cuda.kernels.bf16, src0.data, dst.data, scale, bias, src0, dst, stream);
    } else {
        scaleF32Cuda(f32, src0.data, dst.data, scale, bias, src0, dst, stream);
    }
}
