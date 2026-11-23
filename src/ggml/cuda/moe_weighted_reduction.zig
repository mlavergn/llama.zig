//! The fused mixture-of-experts output reduction: each token's expert rows,
//! optionally scaled per expert, weighted by the router and summed in one
//! kernel. Reached only through `fusion.zig`'s fusion pass, which matches the
//! `MUL` (`MUL`) `VIEW`... `ADD`... chain the graph builds for it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/moe-weighted-reduction.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/moe-weighted-reduction.cuh` at v1.0.0
//! (`b1b770d90`), the host half. The kernel, `moe_weighted_reduction_f32`,
//! stays CUDA.

const ggml = @import("../module.zig");
/// Ports `launch_moe_weighted_reduction` (moe-weighted-reduction.cu:27 @b1b770d90),
/// which launches with `<<<>>>`.
fn launch(experts: ?*const anyopaque, expert_scale: ?*const anyopaque, weights: ?*const anyopaque, dst: ?*anyopaque, n_embd: i64, n_tokens: i64, n_expert_used: c_int, stream: ggml.cuda.common.cuda_c.cudaStream_t) void {
    const threads = 256;
    // `dim3`'s members are `unsigned int`.
    const blocks: ggml.cuda.kernels.Dim3 = .{
        .x = @truncate(@as(u64, @bitCast(n_tokens))),
        .y = @truncate(@as(u64, @bitCast(@divTrunc(n_embd + threads - 1, threads)))),
        .z = 1,
    };
    ggml.cuda.kernels.launch(ggml.cuda.kernels.get("moe-weighted-reduction", "moe_weighted_reduction_f32"), .{
        .grid = blocks,
        .block = .{ .x = threads },
        .stream = stream,
    }, .{ experts, expert_scale, weights, dst, n_embd, n_expert_used });
}

/// Ports `ggml_cuda_op_moe_weighted_reduction` (moe-weighted-reduction.cu:41 @b1b770d90).
pub fn opMoeWeightedReduction(ctx: *ggml.cuda.common.Context, experts: *const ggml.impl.ggml_c.ggml_tensor, expert_scale: ?*const ggml.impl.ggml_c.ggml_tensor, weights: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    ggml.impl.assert(experts.type == ggml.impl.ggml_c.GGML_TYPE_F32, "experts->type == GGML_TYPE_F32");
    ggml.impl.assert(weights.type == ggml.impl.ggml_c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    ggml.impl.assert(expert_scale == null or expert_scale.?.type == ggml.impl.ggml_c.GGML_TYPE_F32, "expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(experts), "ggml_is_contiguous(experts)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(weights), "ggml_is_contiguous(weights)");
    ggml.impl.assert(expert_scale == null or ggml.types.ggml_is_contiguous(expert_scale.?), "expert_scale == nullptr || ggml_is_contiguous(expert_scale)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const n_embd: i64 = experts.ne[0];
    const n_expert_used: i64 = experts.ne[1];
    const n_tokens: i64 = experts.ne[2] * experts.ne[3];
    const stream = ctx.stream();

    launch(
        experts.data,
        if (expert_scale) |s| s.data else null,
        weights.data,
        dst.data,
        n_embd,
        n_tokens,
        @truncate(n_expert_used),
        stream,
    );
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cuda_c.cudaGetLastError(), "cudaGetLastError()", @src());
}
