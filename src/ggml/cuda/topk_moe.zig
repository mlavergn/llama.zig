//! The fused mixture-of-experts router: gating, top-k selection, optional
//! bias, weight normalisation and scale in one kernel. Reached only through
//! `graph.zig`'s fusion pass.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/topk-moe.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/topk-moe.cuh` at v0.3.0 (`c1d0e7a00`), the
//! host half. The kernel, `topk_moe_cuda`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `ggml_cuda_topk_moe_args` (topk-moe.cuh:6 @c1d0e7a00).
pub const Args = struct {
    sigmoid: bool = false,
    sqrt_softplus: bool = false,
    softmax: bool = false,
    delayed_softmax: bool = false,
    prob_bias: bool = false,
    norm: bool = false,
    scale: bool = false,
};

/// Ports `topk_moe_config` (topk-moe.cu:9 @c1d0e7a00); passed to the kernel
/// by value, four `bool`s.
const Config = extern struct {
    use_sigmoid: bool,
    use_sqrt_softplus: bool,
    with_norm: bool,
    delayed_softmax: bool,
};

/// Ports `launch_topk_moe_cuda` (topk-moe.cu:272 @c1d0e7a00).
fn launchTopkMoe(comptime has_bias: bool, ctx: *common.Context, logits: ?*const f32, weights: ?*f32, ids: ?*i32, bias: ?*f32, n_rows: c_int, n_expert: c_int, n_expert_used: c_int, clamp_val: f32, scale_val: f32, config: Config) void {
    impl.assert(!(config.with_norm and config.delayed_softmax), "!(config.with_norm && config.delayed_softmax) && \"delayed softmax is not supported with weight normalization\"");
    const rows_per_block = 4;
    const p: kernels.LaunchParams = .{
        .grid = .{ .x = @bitCast(@divTrunc(n_rows + rows_per_block - 1, rows_per_block)) },
        .block = .{ .x = common.warp_size, .y = rows_per_block },
        .stream = ctx.stream(),
    };
    const args = .{ logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config };
    switch (n_expert) {
        inline 1, 2, 4, 8, 16, 32, 64, 128, 256, 288, 512, 576 => |n| kernels.launchPdl(
            kernels.get("topk-moe", kernels.tmpl("topk_moe_cuda", &.{ kernels.int(n), kernels.boolean(has_bias) })),
            p,
            args,
        ),
        else => impl.assert(false, "false && \"fatal error\""),
    }
}

/// Ports `ggml_cuda_op_topk_moe` (topk-moe.cu:346 @c1d0e7a00).
pub fn opTopkMoe(ctx: *common.Context, logits: *const Tensor, weights: *Tensor, ids: *Tensor, clamp: ?*const Tensor, scale: ?*const Tensor, bias: ?*const Tensor, args: *const Args) void {
    impl.assert(logits.type == c.GGML_TYPE_F32, "logits->type == GGML_TYPE_F32");
    impl.assert(weights.type == c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    impl.assert(ids.type == c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    const n_experts: c_int = @truncate(logits.ne[0]);
    const n_rows: c_int = @truncate(logits.ne[1]);

    const logits_d: ?*const f32 = @ptrCast(@alignCast(logits.data));
    const weights_d: ?*f32 = @ptrCast(@alignCast(weights.data));
    const ids_d: ?*i32 = @ptrCast(@alignCast(ids.data));
    const bias_d: ?*f32 = if (bias) |b| @ptrCast(@alignCast(b.data)) else null;

    const scale_val: f32 = if (scale) |s| impl.getOpParamsF32(s, 0) else 1.0;

    impl.assert(ids.nb[1] / c.ggml_type_size(ids.type) == @as(usize, @intCast(n_experts)), "ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts");

    const n_expert_used: c_int = @truncate(weights.ne[1]);

    const with_norm = clamp != null;

    var clamp_val: f32 = -std.math.inf(f32);
    if (clamp) |cl| clamp_val = impl.getOpParamsF32(cl, 0);

    const config: Config = .{
        .use_sigmoid = args.sigmoid,
        .use_sqrt_softplus = args.sqrt_softplus,
        .with_norm = with_norm,
        .delayed_softmax = args.delayed_softmax,
    };

    if (bias != null) {
        launchTopkMoe(true, ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val, scale_val, config);
    } else {
        launchTopkMoe(false, ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val, scale_val, config);
    }
}

/// Ports `ggml_cuda_should_use_topk_moe` (topk-moe.cu:394 @c1d0e7a00).
pub fn shouldUseTopkMoe(gating_op: *const Tensor, weights: *const Tensor, logits: *const Tensor, ids: *const Tensor) bool {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up
    // to 512, or one of the non-power-of-2 expert counts of supported models
    const n_expert: c_int = @intCast(ids.nb[1] / ids.nb[0]);
    if (((n_expert & (n_expert - 1)) != 0 or n_expert > 512) and n_expert != 288 and n_expert != 576) return false;

    if (!c.ggml_is_contiguous(weights) or !c.ggml_is_contiguous(logits)) return false;

    if (gating_op.op == c.GGML_OP_SOFT_MAX) {
        const softmax = gating_op;
        const scale = impl.getOpParamsF32(softmax, 0);
        const max_bias = impl.getOpParamsF32(softmax, 1);

        if (!c.ggml_is_contiguous(impl.one(Tensor, softmax.src[0]))) return false;
        if (scale != 1.0 or max_bias != 0.0) return false;
        // don't fuse when masks or sinks are present
        if (softmax.src[1] != null or softmax.src[2] != null) return false;
    } else if (gating_op.op == c.GGML_OP_UNARY) {
        const op = c.ggml_get_unary_op(gating_op);
        if (op != c.GGML_UNARY_OP_SIGMOID and op != c.GGML_UNARY_OP_SOFTPLUS) return false;
    }

    return true;
}
