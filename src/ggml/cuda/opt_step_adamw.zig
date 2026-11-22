//! `OPT_STEP_ADAMW`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/opt-step-adamw.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/opt-step-adamw.cuh` at v1.0.0
//! (`b1b770d90`). The kernel, `opt_step_adamw_f32`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_OPT_STEP_ADAMW_BLOCK_SIZE` (opt-step-adamw.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `opt_step_adamw_f32_cuda` (opt-step-adamw.cu:37 @b1b770d90).
fn optStepAdamwF32Cuda(x: ?*anyopaque, g: ?*const anyopaque, g_m: ?*anyopaque, g_v: ?*anyopaque, pars: ?*const anyopaque, k: i64, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const block_dims: ggml.cuda.kernels.Dim3 = .{ .x = block_size };
    const block_nums: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(@divTrunc(k + block_size - 1, block_size)))) };
    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("opt-step-adamw", "opt_step_adamw_f32"), .{
        .grid = block_nums,
        .block = block_dims,
        .shmem = 0,
        .stream = stream,
    }, .{ x, g, g_m, g_v, pars, k });
}

/// Ports `ggml_cuda_opt_step_adamw` (opt-step-adamw.cu:45 @b1b770d90).
pub fn opOptStepAdamw(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src0_grad = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);
    const src0_grad_m = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[2]);
    const src0_grad_v = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[3]);
    const adamw_params = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[4]);

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type         == GGML_TYPE_F32");
    ggml.impl.assert(src0_grad.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0_grad->type    == GGML_TYPE_F32");
    ggml.impl.assert(src0_grad_m.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0_grad_m->type  == GGML_TYPE_F32");
    ggml.impl.assert(src0_grad_v.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0_grad_v->type  == GGML_TYPE_F32");
    ggml.impl.assert(adamw_params.type == ggml.impl.ggml_c.GGML_TYPE_F32, "adamw_params->type == GGML_TYPE_F32");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0_grad), "ggml_is_contiguous(src0_grad)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0_grad_m), "ggml_is_contiguous(src0_grad_m)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0_grad_v), "ggml_is_contiguous(src0_grad_v)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(adamw_params), "ggml_is_contiguous(adamw_params)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(src0, src0_grad), "ggml_are_same_shape(src0, src0_grad)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(src0, src0_grad_m), "ggml_are_same_shape(src0, src0_grad_m)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_are_same_shape(src0, src0_grad_v), "ggml_are_same_shape(src0, src0_grad_v)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_nelements(adamw_params) == 7, "ggml_nelements(adamw_params) == 7");

    const stream = ctx.stream();

    const ne = ggml.impl.ggml_c.ggml_nelements(src0);

    optStepAdamwF32Cuda(src0.data, src0_grad.data, src0_grad_m.data, src0_grad_v.data, adamw_params.data, ne, stream);
}
