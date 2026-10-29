//! `OPT_STEP_SGD`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/opt-step-sgd.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/opt-step-sgd.cuh` at v0.4.1
//! (`b29c606e2`). The kernel, `opt_step_sgd_f32`, stays CUDA.

const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_OPT_STEP_SGD_BLOCK_SIZE` (opt-step-sgd.cuh:3 @b29c606e2).
const block_size = 256;

/// Ports `opt_step_sgd_f32_cuda` (opt-step-sgd.cu:18 @b29c606e2).
fn optStepSgdF32Cuda(x: ?*anyopaque, g: ?*const anyopaque, pars: ?*const anyopaque, k: i64, stream: cu.cudaStream_t) void {
    const block_dims: kernels.Dim3 = .{ .x = block_size };
    const block_nums: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(@divTrunc(k + block_size - 1, block_size)))) };
    kernels.launch(kernels.get("opt-step-sgd", "opt_step_sgd_f32"), .{
        .grid = block_nums,
        .block = block_dims,
        .shmem = 0,
        .stream = stream,
    }, .{ x, g, pars, k });
}

/// Ports `ggml_cuda_opt_step_sgd` (opt-step-sgd.cu:26 @b29c606e2).
pub fn opOptStepSgd(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_grad = impl.one(Tensor, dst.src[1]);
    const params = impl.one(Tensor, dst.src[2]);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type      == GGML_TYPE_F32");
    impl.assert(src0_grad.type == c.GGML_TYPE_F32, "src0_grad->type == GGML_TYPE_F32");
    impl.assert(params.type == c.GGML_TYPE_F32, "params->type    == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    impl.assert(c.ggml_is_contiguous(src0_grad), "ggml_is_contiguous(src0_grad)");
    impl.assert(c.ggml_is_contiguous(params), "ggml_is_contiguous(params)");
    impl.assert(c.ggml_are_same_shape(src0, src0_grad), "ggml_are_same_shape(src0, src0_grad)");
    impl.assert(c.ggml_nelements(params) == 2, "ggml_nelements(params) == 2");

    const stream = ctx.stream();

    const ne = c.ggml_nelements(src0);

    optStepSgdF32Cuda(src0.data, src0_grad.data, params.data, ne, stream);
}
