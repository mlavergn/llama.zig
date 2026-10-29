//! `OPT_STEP_ADAMW`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/opt-step-adamw.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/opt-step-adamw.cuh` at v0.4.1
//! (`b29c606e2`). The kernel, `opt_step_adamw_f32`, stays CUDA.

const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_OPT_STEP_ADAMW_BLOCK_SIZE` (opt-step-adamw.cuh:3 @b29c606e2).
const block_size = 256;

/// Ports `opt_step_adamw_f32_cuda` (opt-step-adamw.cu:37 @b29c606e2).
fn optStepAdamwF32Cuda(x: ?*anyopaque, g: ?*const anyopaque, g_m: ?*anyopaque, g_v: ?*anyopaque, pars: ?*const anyopaque, k: i64, stream: cu.cudaStream_t) void {
    const block_dims: kernels.Dim3 = .{ .x = block_size };
    const block_nums: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(@divTrunc(k + block_size - 1, block_size)))) };
    kernels.launch(kernels.get("opt-step-adamw", "opt_step_adamw_f32"), .{
        .grid = block_nums,
        .block = block_dims,
        .shmem = 0,
        .stream = stream,
    }, .{ x, g, g_m, g_v, pars, k });
}

/// Ports `ggml_cuda_opt_step_adamw` (opt-step-adamw.cu:45 @b29c606e2).
pub fn opOptStepAdamw(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_grad = impl.one(Tensor, dst.src[1]);
    const src0_grad_m = impl.one(Tensor, dst.src[2]);
    const src0_grad_v = impl.one(Tensor, dst.src[3]);
    const adamw_params = impl.one(Tensor, dst.src[4]);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type         == GGML_TYPE_F32");
    impl.assert(src0_grad.type == c.GGML_TYPE_F32, "src0_grad->type    == GGML_TYPE_F32");
    impl.assert(src0_grad_m.type == c.GGML_TYPE_F32, "src0_grad_m->type  == GGML_TYPE_F32");
    impl.assert(src0_grad_v.type == c.GGML_TYPE_F32, "src0_grad_v->type  == GGML_TYPE_F32");
    impl.assert(adamw_params.type == c.GGML_TYPE_F32, "adamw_params->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    impl.assert(c.ggml_is_contiguous(src0_grad), "ggml_is_contiguous(src0_grad)");
    impl.assert(c.ggml_is_contiguous(src0_grad_m), "ggml_is_contiguous(src0_grad_m)");
    impl.assert(c.ggml_is_contiguous(src0_grad_v), "ggml_is_contiguous(src0_grad_v)");
    impl.assert(c.ggml_is_contiguous(adamw_params), "ggml_is_contiguous(adamw_params)");
    impl.assert(c.ggml_are_same_shape(src0, src0_grad), "ggml_are_same_shape(src0, src0_grad)");
    impl.assert(c.ggml_are_same_shape(src0, src0_grad_m), "ggml_are_same_shape(src0, src0_grad_m)");
    impl.assert(c.ggml_are_same_shape(src0, src0_grad_v), "ggml_are_same_shape(src0, src0_grad_v)");
    impl.assert(c.ggml_nelements(adamw_params) == 7, "ggml_nelements(adamw_params) == 7");

    const stream = ctx.stream();

    const ne = c.ggml_nelements(src0);

    optStepAdamwF32Cuda(src0.data, src0_grad.data, src0_grad_m.data, src0_grad_v.data, adamw_params.data, ne, stream);
}
