//! `OPT_STEP_SGD`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/opt-step-sgd.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/opt-step-sgd.cuh` at v1.0.0
//! (`b1b770d90`). The kernel, `opt_step_sgd_f32`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_OPT_STEP_SGD_BLOCK_SIZE` (opt-step-sgd.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `opt_step_sgd_f32_cuda` (opt-step-sgd.cu:18 @b1b770d90).
fn optStepSgdF32Cuda(x: ?*anyopaque, g: ?*const anyopaque, pars: ?*const anyopaque, k: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const block_dims: ggml.cuda.kernels.Dim3 = .{ .x = block_size };
    const block_nums: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(@divTrunc(k + block_size - 1, block_size)))) };
    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("opt-step-sgd", "opt_step_sgd_f32"), .{
        .grid = block_nums,
        .block = block_dims,
        .shmem = 0,
        .stream = stream,
    }, .{ x, g, pars, k });
}

/// Ports `ggml_cuda_opt_step_sgd` (opt-step-sgd.cu:26 @b1b770d90).
pub fn opOptStepSgd(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src0_grad = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);
    const params = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[2]);

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type      == GGML_TYPE_F32");
    ggml.impl.assert(src0_grad.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0_grad->type == GGML_TYPE_F32");
    ggml.impl.assert(params.type == ggml.impl.ggml_c.GGML_TYPE_F32, "params->type    == GGML_TYPE_F32");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0_grad), "ggml_is_contiguous(src0_grad)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(params), "ggml_is_contiguous(params)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(src0, src0_grad), "ggml_are_same_shape(src0, src0_grad)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_nelements(params) == 2, "ggml_nelements(params) == 2");

    const stream = ctx.stream();

    const ne = ggml.impl.ggml_c.ggml_nelements(src0);

    optStepSgdF32Cuda(src0.data, src0_grad.data, params.data, ne, stream);
}
