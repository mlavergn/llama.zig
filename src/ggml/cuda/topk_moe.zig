//! The fused mixture-of-experts router: gating, top-k selection, optional
//! bias, weight normalisation and scale in one kernel. Reached only through
//! `graph.zig`'s fusion pass.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/topk-moe.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/topk-moe.cuh` at v1.0.0 (`b1b770d90`), the
//! host half. The kernel, `topk_moe_cuda`, stays CUDA.

const std = @import("std");
const ggml = @import("../module.zig");

/// Ports `TOPK_MOE_ROWS_PER_BLOCK` (topk-moe.cuh:7 @b1b770d90): rows that one
/// CUDA block handles. `fusion.zig`'s memory-range check reads it too: one
/// block reads all its logits before it writes, so up to this many rows the
/// logits may alias the outputs.
pub const rows_per_block = 8;

/// Ports `ggml_cuda_topk_moe_args` (topk-moe.cuh:9 @b1b770d90).
pub const Args = struct {
    sigmoid: bool = false,
    sqrt_softplus: bool = false,
    softmax: bool = false,
    delayed_softmax: bool = false,
    prob_bias: bool = false,
    norm: bool = false,
    scale: bool = false,
    /// > 0: pick experts only from the top `n_group_used` expert groups.
    n_groups: c_int = 0,
    n_group_used: c_int = 0,
};

/// Ports `topk_moe_config` (topk-moe.cu:9 @b1b770d90); passed to the kernel
/// by value: four `bool`s and two `int`s, 12 bytes in all.
const Config = extern struct {
    use_sigmoid: bool,
    use_sqrt_softplus: bool,
    with_norm: bool,
    delayed_softmax: bool,
    n_groups: c_int,
    n_group_used: c_int,
};

/// Ports `launch_topk_moe_cuda` (topk-moe.cu:344 @b1b770d90).
fn launchTopkMoe(comptime has_bias: bool, ctx: *ggml.cuda.common.Context, logits: ?*const f32, weights: ?*f32, ids: ?*i32, bias: ?*f32, n_rows: c_int, n_expert: c_int, n_expert_used: c_int, clamp_val: f32, scale_val: f32, config: Config) void {
    ggml.impl.assert(!(config.with_norm and config.delayed_softmax), "!(config.with_norm && config.delayed_softmax) && \"delayed softmax is not supported with weight normalization\"");
    const p: ggml.cuda.kernels.LaunchParams = .{
        .grid = .{ .x = @bitCast(@divTrunc(n_rows + rows_per_block - 1, rows_per_block)) },
        .block = .{ .x = ggml.cuda.common.warp_size, .y = rows_per_block },
        .stream = ctx.stream(),
    };
    const args = .{ logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config };
    switch (n_expert) {
        inline 1, 2, 4, 8, 16, 32, 64, 128, 256, 288, 384, 512, 576 => |n| ggml.cuda.kernels.launchPdl(
            ggml.cuda.kernels.get("topk-moe", ggml.cuda.kernels.tmpl("topk_moe_cuda", &.{ ggml.cuda.kernels.int(n), ggml.cuda.kernels.boolean(has_bias) })),
            p,
            args,
        ),
        else => ggml.impl.assert(false, "false && \"fatal error\""),
    }
}

/// Ports `ggml_cuda_op_topk_moe` (topk-moe.cu:425 @b1b770d90).
pub fn opTopkMoe(ctx: *ggml.cuda.common.Context, logits: *const ggml.impl.ggml_c.ggml_tensor, weights: *ggml.impl.ggml_c.ggml_tensor, ids: *ggml.impl.ggml_c.ggml_tensor, clamp: ?*const ggml.impl.ggml_c.ggml_tensor, scale: ?*const ggml.impl.ggml_c.ggml_tensor, bias: ?*const ggml.impl.ggml_c.ggml_tensor, args: *const Args) void {
    ggml.impl.assert(logits.type == ggml.impl.ggml_c.GGML_TYPE_F32, "logits->type == GGML_TYPE_F32");
    ggml.impl.assert(weights.type == ggml.impl.ggml_c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    ggml.impl.assert(ids.type == ggml.impl.ggml_c.GGML_TYPE_I32, "ids->type == GGML_TYPE_I32");

    const n_experts: c_int = @truncate(logits.ne[0]);
    const n_rows: c_int = @truncate(logits.ne[1]);

    const logits_d: ?*const f32 = @ptrCast(@alignCast(logits.data));
    const weights_d: ?*f32 = @ptrCast(@alignCast(weights.data));
    const ids_d: ?*i32 = @ptrCast(@alignCast(ids.data));
    const bias_d: ?*f32 = if (bias) |b| @ptrCast(@alignCast(b.data)) else null;

    const scale_val: f32 = if (scale) |s| ggml.impl.getOpParamsF32(s, 0) else 1.0;

    ggml.impl.assert(ids.nb[1] / ggml.types.ggml_type_size(ids.type) == @as(usize, @intCast(n_experts)), "ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts");

    const n_expert_used: c_int = @truncate(weights.ne[1]);

    const with_norm = clamp != null;

    var clamp_val: f32 = -std.math.inf(f32);
    if (clamp) |cl| clamp_val = ggml.impl.getOpParamsF32(cl, 0);

    const config: Config = .{
        .use_sigmoid = args.sigmoid,
        .use_sqrt_softplus = args.sqrt_softplus,
        .with_norm = with_norm,
        .delayed_softmax = args.delayed_softmax,
        .n_groups = args.n_groups,
        .n_group_used = args.n_group_used,
    };

    if (bias != null) {
        launchTopkMoe(true, ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val, scale_val, config);
    } else {
        launchTopkMoe(false, ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val, scale_val, config);
    }
}

/// Ports `ggml_cuda_should_use_topk_moe` (topk-moe.cu:475 @b1b770d90).
pub fn shouldUseTopkMoe(gating_op: *const ggml.impl.ggml_c.ggml_tensor, weights: *const ggml.impl.ggml_c.ggml_tensor, logits: *const ggml.impl.ggml_c.ggml_tensor, ids: *const ggml.impl.ggml_c.ggml_tensor) bool {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up
    // to 512, or one of the non-power-of-2 expert counts of supported models
    const n_expert: c_int = @intCast(ids.nb[1] / ids.nb[0]);
    if (((n_expert & (n_expert - 1)) != 0 or n_expert > 512) and n_expert != 288 and n_expert != 384 and n_expert != 576) return false;

    if (!ggml.types.ggml_is_contiguous(weights) or !ggml.types.ggml_is_contiguous(logits)) return false;

    if (gating_op.op == ggml.impl.ggml_c.GGML_OP_SOFT_MAX) {
        const softmax = gating_op;
        const scale = ggml.impl.getOpParamsF32(softmax, 0);
        const max_bias = ggml.impl.getOpParamsF32(softmax, 1);

        if (!ggml.types.ggml_is_contiguous(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, softmax.src[0]))) return false;
        if (scale != 1.0 or max_bias != 0.0) return false;
        // don't fuse when masks or sinks are present
        if (softmax.src[1] != null or softmax.src[2] != null) return false;
    } else if (gating_op.op == ggml.impl.ggml_c.GGML_OP_UNARY) {
        const op = ggml.types.ggml_get_unary_op(gating_op);
        if (op != ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID and op != ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS) return false;
    }

    return true;
}
