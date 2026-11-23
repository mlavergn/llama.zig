//! `GGML_OP_ARANGE`: `dst[i] = start + step * i`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/arange.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/arange.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `arange_f32`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_ARANGE_BLOCK_SIZE` (arange.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `arange_f32_cuda` (arange.cu:12 @b1b770d90), which launches with
/// `<<<>>>`.
fn arangeF32Cuda(dst: ?*anyopaque, ne0: c_int, start: f32, step: f32, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(ne0 + block_size - 1, block_size);
    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("arange", "arange_f32"), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = block_size },
        .stream = stream,
    }, .{ dst, ne0, start, step });
}

/// Ports `ggml_cuda_op_arange` (arange.cu:17 @b1b770d90).
pub fn opArange(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const start: f32 = @bitCast(dst.op_params[0]);
    const stop: f32 = @bitCast(dst.op_params[1]);
    const step: f32 = @bitCast(dst.op_params[2]);

    // `ceil` is the `double` overload: the quotient is a `float`, widened.
    const steps: i64 = @intFromFloat(@ceil(@as(f64, (stop - start) / step)));
    ggml.impl.assert(ggml.types.ggml_nelements(dst) == steps, "ggml_nelements(dst) == steps");

    arangeF32Cuda(dst_d, @truncate(dst.ne[0]), start, step, stream);
}
