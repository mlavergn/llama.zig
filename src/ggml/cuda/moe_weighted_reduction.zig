//! The fused mixture-of-experts output reduction: each token's expert rows,
//! optionally scaled per expert, weighted by the router and summed in one
//! kernel. Reached only through `fusion.zig`'s fusion pass, which matches the
//! `MUL` (`MUL`) `VIEW`... `ADD`... chain the graph builds for it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/moe-weighted-reduction.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/moe-weighted-reduction.cuh` at v0.4.1
//! (`b29c606e2`), the host half. The kernel, `moe_weighted_reduction_f32`,
//! stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `launch_moe_weighted_reduction` (moe-weighted-reduction.cu:27 @b29c606e2),
/// which launches with `<<<>>>`.
fn launch(experts: ?*const anyopaque, expert_scale: ?*const anyopaque, weights: ?*const anyopaque, dst: ?*anyopaque, n_embd: i64, n_tokens: i64, n_expert_used: c_int, stream: cu.cudaStream_t) void {
    const threads = 256;
    // `dim3`'s members are `unsigned int`.
    const blocks: kernels.Dim3 = .{
        .x = @truncate(@as(u64, @bitCast(n_tokens))),
        .y = @truncate(@as(u64, @bitCast(@divTrunc(n_embd + threads - 1, threads)))),
        .z = 1,
    };
    kernels.launch(kernels.get("moe-weighted-reduction", "moe_weighted_reduction_f32"), .{
        .grid = blocks,
        .block = .{ .x = threads },
        .stream = stream,
    }, .{ experts, expert_scale, weights, dst, n_embd, n_expert_used });
}

/// Ports `ggml_cuda_op_moe_weighted_reduction` (moe-weighted-reduction.cu:41 @b29c606e2).
pub fn opMoeWeightedReduction(ctx: *common.Context, experts: *const Tensor, expert_scale: ?*const Tensor, weights: *const Tensor, dst: *Tensor) void {
    impl.assert(experts.type == c.GGML_TYPE_F32, "experts->type == GGML_TYPE_F32");
    impl.assert(weights.type == c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    impl.assert(expert_scale == null or expert_scale.?.type == c.GGML_TYPE_F32, "expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous(experts), "ggml_is_contiguous(experts)");
    impl.assert(c.ggml_is_contiguous(weights), "ggml_is_contiguous(weights)");
    impl.assert(expert_scale == null or c.ggml_is_contiguous(expert_scale.?), "expert_scale == nullptr || ggml_is_contiguous(expert_scale)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

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
    common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
}
