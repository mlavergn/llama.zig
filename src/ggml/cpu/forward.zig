//! The op dispatch: given a graph node, run the CPU kernel for its op.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v1.0.0, `b1b770d90`),
//! `ggml_compute_forward` at line 1711.
//!
//! # The kernels
//!
//! Every arm below calls a kernel ported from `ggml-cpu/ops.cpp` (or
//! `mul_mat` and `mul_mat_id`, in `mulmat.zig` because they coordinate
//! threads rather than just computing), imported by its C name from the file
//! that defines it.
//!
//! # The switch is exhaustive on purpose
//!
//! The C's `switch` covers all 102 members of `enum ggml_op` and ends in a
//! `default` that aborts, so a new op added upstream fails loudly rather than
//! silently computing nothing. Zig would let the `else` stand alone; keeping
//! every op named means the compiler flags the addition instead.

const std = @import("std");
const ggml = @import("../module.zig");

/// `ggml-cpu/traits.cpp`'s entry point, ported.
///
/// Returns true when the tensor carries an extra-buffer accelerator that has
/// already computed the result, in which case the switch is skipped entirely.
/// Ports `ggml_compute_forward` (ggml-cpu.c:1760 @b1b770d90).
///
/// Parameters:
/// - `params`: this thread's index, count, and slice of the work buffer.
/// - `tensor`: the node to compute, in place, into its own `data`.
pub fn computeForward(params: *const ggml.cpu.defs.ComputeParams, tensor: *ggml.impl.ggml_c.ggml_tensor) void {
    if (tensor.op == ggml.impl.ggml_c.GGML_OP_NONE or ggml.types.ggml_is_empty(tensor)) return;

    // An extra-buffer op, handled by an accelerator rather than by us.
    if (ggml.cpu.extra.ggml_cpu_extra_compute_forward(@constCast(params), tensor)) return;

    switch (tensor.op) {
        ggml.impl.ggml_c.GGML_OP_DUP => ggml.cpu.ops.dup.ggml_compute_forward_dup(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ADD => ggml.cpu.ops.arith.ggml_compute_forward_add(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ADD_ID => ggml.cpu.ops.arith.ggml_compute_forward_add_id(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ACC => ggml.cpu.ops.arith.ggml_compute_forward_acc(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SUB => ggml.cpu.binary_ops.ggml_compute_forward_sub(params, tensor),
        ggml.impl.ggml_c.GGML_OP_MUL => ggml.cpu.binary_ops.ggml_compute_forward_mul(params, tensor),
        ggml.impl.ggml_c.GGML_OP_DIV => ggml.cpu.binary_ops.ggml_compute_forward_div(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SQR => ggml.cpu.unary_ops.entry(.sqr)(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SQRT => ggml.cpu.unary_ops.entry(.sqrt)(params, tensor),
        ggml.impl.ggml_c.GGML_OP_LOG => ggml.cpu.unary_ops.entry(.log)(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SIN => ggml.cpu.unary_ops.entry(.sin)(params, tensor),
        ggml.impl.ggml_c.GGML_OP_COS => ggml.cpu.unary_ops.entry(.cos)(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SUM => ggml.cpu.ops.reduce.ggml_compute_forward_sum(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SUM_ROWS => ggml.cpu.ops.reduce.ggml_compute_forward_sum_rows(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CUMSUM => ggml.cpu.ops.reduce.ggml_compute_forward_cumsum(params, tensor),
        ggml.impl.ggml_c.GGML_OP_MEAN => ggml.cpu.ops.reduce.ggml_compute_forward_mean(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ARGMAX => ggml.cpu.ops.reduce.ggml_compute_forward_argmax(params, tensor),
        ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL => ggml.cpu.ops.reduce.ggml_compute_forward_count_equal(params, tensor),
        ggml.impl.ggml_c.GGML_OP_REPEAT => ggml.cpu.ops.repeat.ggml_compute_forward_repeat(params, tensor),
        ggml.impl.ggml_c.GGML_OP_REPEAT_BACK => ggml.cpu.ops.repeat.ggml_compute_forward_repeat_back(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CONCAT => ggml.cpu.ops.repeat.ggml_compute_forward_concat(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SILU_BACK => ggml.cpu.ops.activation.ggml_compute_forward_silu_back(params, tensor),
        ggml.impl.ggml_c.GGML_OP_NORM => ggml.cpu.ops.norm.ggml_compute_forward_norm(params, tensor),
        ggml.impl.ggml_c.GGML_OP_RMS_NORM => ggml.cpu.ops.norm.ggml_compute_forward_rms_norm(params, tensor),
        ggml.impl.ggml_c.GGML_OP_RMS_NORM_BACK => ggml.cpu.ops.norm.ggml_compute_forward_rms_norm_back(params, tensor),
        ggml.impl.ggml_c.GGML_OP_GROUP_NORM => ggml.cpu.ops.norm.ggml_compute_forward_group_norm(params, tensor),
        ggml.impl.ggml_c.GGML_OP_L2_NORM => ggml.cpu.ops.norm.ggml_compute_forward_l2_norm(params, tensor),
        ggml.impl.ggml_c.GGML_OP_MUL_MAT => ggml.cpu.mulmat.ggml_compute_forward_mul_mat(params, tensor),
        ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID => ggml.cpu.mulmat.computeForwardMulMatId(params, tensor),
        ggml.impl.ggml_c.GGML_OP_OUT_PROD => ggml.cpu.ops.linalg.ggml_compute_forward_out_prod(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SCALE => ggml.cpu.ops.linalg.ggml_compute_forward_scale(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SET => ggml.cpu.ops.linalg.ggml_compute_forward_set(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CPY => ggml.cpu.ops.dup.ggml_compute_forward_cpy(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CONT => ggml.cpu.ops.dup.ggml_compute_forward_cont(params, tensor),
        ggml.impl.ggml_c.GGML_OP_GET_ROWS => ggml.cpu.ops.gather.ggml_compute_forward_get_rows(params, tensor),
        ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK => ggml.cpu.ops.gather.ggml_compute_forward_get_rows_back(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SET_ROWS => ggml.cpu.ops.gather.ggml_compute_forward_set_rows(params, tensor),
        ggml.impl.ggml_c.GGML_OP_DIAG => ggml.cpu.ops.gather.ggml_compute_forward_diag(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX => ggml.cpu.ops.softmax.ggml_compute_forward_soft_max(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK => ggml.cpu.ops.softmax.ggml_compute_forward_soft_max_ext_back(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ROPE => ggml.cpu.ops.rope.ggml_compute_forward_rope(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ROPE_BACK => ggml.cpu.ops.rope.ggml_compute_forward_rope_back(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CLAMP => ggml.cpu.ops.softmax.ggml_compute_forward_clamp(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D => ggml.cpu.ops.conv.ggml_compute_forward_conv_transpose_1d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_IM2COL => ggml.cpu.ops.conv.ggml_compute_forward_im2col(params, tensor),
        ggml.impl.ggml_c.GGML_OP_IM2COL_BACK => ggml.cpu.ops.conv.ggml_compute_forward_im2col_back_f32(params, tensor),
        ggml.impl.ggml_c.GGML_OP_IM2COL_3D => ggml.cpu.ops.conv.ggml_compute_forward_im2col_3d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_COL2IM_1D => ggml.cpu.ops.conv.ggml_compute_forward_col2im_1d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CONV_2D => ggml.cpu.ops.conv.ggml_compute_forward_conv_2d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CONV_3D => ggml.cpu.ops.conv.ggml_compute_forward_conv_3d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CONV_2D_DW => ggml.cpu.ops.conv.ggml_compute_forward_conv_2d_dw(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D => ggml.cpu.ops.conv.ggml_compute_forward_conv_transpose_2d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_POOL_1D => ggml.cpu.ops.pool.ggml_compute_forward_pool_1d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_POOL_2D => ggml.cpu.ops.pool.ggml_compute_forward_pool_2d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_POOL_2D_BACK => ggml.cpu.ops.pool.ggml_compute_forward_pool_2d_back(params, tensor),
        ggml.impl.ggml_c.GGML_OP_UPSCALE => ggml.cpu.ops.pad.ggml_compute_forward_upscale(params, tensor),
        ggml.impl.ggml_c.GGML_OP_PAD => ggml.cpu.ops.pad.ggml_compute_forward_pad(params, tensor),
        ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D => ggml.cpu.ops.pad.ggml_compute_forward_pad_reflect_1d(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ROLL => ggml.cpu.ops.pad.ggml_compute_forward_roll(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ARANGE => ggml.cpu.ops.pad.ggml_compute_forward_arange(params, tensor),
        ggml.impl.ggml_c.GGML_OP_TIMESTEP_EMBEDDING => ggml.cpu.ops.pad.ggml_compute_forward_timestep_embedding(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ARGSORT => ggml.cpu.ops.sort.ggml_compute_forward_argsort(params, tensor),
        ggml.impl.ggml_c.GGML_OP_TOP_K => ggml.cpu.ops.sort.ggml_compute_forward_top_k(params, tensor),
        ggml.impl.ggml_c.GGML_OP_LEAKY_RELU => ggml.cpu.ops.activation.ggml_compute_forward_leaky_relu(params, tensor),
        ggml.impl.ggml_c.GGML_OP_TRI => ggml.cpu.ops.sort.ggml_compute_forward_tri(params, tensor),
        ggml.impl.ggml_c.GGML_OP_FILL => ggml.cpu.ops.sort.ggml_compute_forward_fill(params, tensor),
        ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT => ggml.cpu.ops.attention.ggml_compute_forward_flash_attn_ext(params, tensor),
        ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_BACK => {
            const t = ggml.impl.getOpParamsI32(tensor, 0);
            ggml.impl.assert(t == 0 or t == 1, "t == 0 || t == 1");
            ggml.cpu.ops.attention.ggml_compute_forward_flash_attn_back(params, t != 0, tensor);
        },
        ggml.impl.ggml_c.GGML_OP_SSM_CONV => ggml.cpu.ops.ssm.ggml_compute_forward_ssm_conv(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SSM_SCAN => ggml.cpu.ops.ssm.ggml_compute_forward_ssm_scan(params, tensor),
        ggml.impl.ggml_c.GGML_OP_WIN_PART => ggml.cpu.ops.window.ggml_compute_forward_win_part(params, tensor),
        ggml.impl.ggml_c.GGML_OP_WIN_UNPART => ggml.cpu.ops.window.ggml_compute_forward_win_unpart(params, tensor),
        ggml.impl.ggml_c.GGML_OP_UNARY => ggml.cpu.ops.activation.ggml_compute_forward_unary(params, tensor),
        ggml.impl.ggml_c.GGML_OP_GLU => ggml.cpu.ops.activation.ggml_compute_forward_glu(params, tensor),
        ggml.impl.ggml_c.GGML_OP_GET_REL_POS => ggml.cpu.ops.window.ggml_compute_forward_get_rel_pos(params, tensor),
        ggml.impl.ggml_c.GGML_OP_ADD_REL_POS => ggml.cpu.ops.window.ggml_compute_forward_add_rel_pos(params, tensor),
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV6 => ggml.cpu.ops.recurrent.ggml_compute_forward_rwkv_wkv6(params, tensor),
        ggml.impl.ggml_c.GGML_OP_GATED_LINEAR_ATTN => ggml.cpu.ops.recurrent.ggml_compute_forward_gla(params, tensor),
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV7 => ggml.cpu.ops.recurrent.ggml_compute_forward_rwkv_wkv7(params, tensor),
        ggml.impl.ggml_c.GGML_OP_SOLVE_TRI => ggml.cpu.ops.custom.ggml_compute_forward_solve_tri(params, tensor),
        ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET => ggml.cpu.ops.recurrent.ggml_compute_forward_gated_delta_net(params, tensor),
        ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER => ggml.cpu.ops.attention.ggml_compute_forward_lightning_indexer(params, tensor),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_COMB => ggml.cpu.ops.recurrent.ggml_compute_forward_dsv4_hc_comb(params, tensor),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_PRE => ggml.cpu.ops.recurrent.ggml_compute_forward_dsv4_hc_pre(params, tensor),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST => ggml.cpu.ops.recurrent.ggml_compute_forward_dsv4_hc_post(params, tensor),
        ggml.impl.ggml_c.GGML_OP_TOP_K_MASK => ggml.cpu.ops.sort.ggml_compute_forward_top_k_mask(params, tensor),
        ggml.impl.ggml_c.GGML_OP_MAP_CUSTOM1 => ggml.cpu.ops.custom.ggml_compute_forward_map_custom1(params, tensor),
        ggml.impl.ggml_c.GGML_OP_MAP_CUSTOM2 => ggml.cpu.ops.custom.ggml_compute_forward_map_custom2(params, tensor),
        ggml.impl.ggml_c.GGML_OP_MAP_CUSTOM3 => ggml.cpu.ops.custom.ggml_compute_forward_map_custom3(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CUSTOM => ggml.cpu.ops.custom.ggml_compute_forward_custom(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS => ggml.cpu.ops.custom.ggml_compute_forward_cross_entropy_loss(params, tensor),
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => ggml.cpu.ops.custom.ggml_compute_forward_cross_entropy_loss_back(params, tensor),
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_ADAMW => ggml.cpu.ops.custom.ggml_compute_forward_opt_step_adamw(params, tensor),
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_SGD => ggml.cpu.ops.custom.ggml_compute_forward_opt_step_sgd(params, tensor),

        // The C spells these out as five separate no-op cases.
        ggml.impl.ggml_c.GGML_OP_NONE, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_PERMUTE, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_TRANSPOSE => {},

        ggml.impl.ggml_c.GGML_OP_COUNT => ggml.impl.abort("fatal error"),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an empty tensor is skipped before the switch" {
    // `ggml_is_empty` fires on any zero dimension. Reaching the switch with
    // one would call a kernel with nothing to write, which several of them
    // handle by reading `src[0]` unconditionally.
    var tensor: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    tensor.op = ggml.impl.ggml_c.GGML_OP_ADD;
    tensor.ne = .{ 4, 0, 1, 1 };

    try std.testing.expect(ggml.types.ggml_is_empty(&tensor));
}

test "the shape-only ops are no-ops" {
    // These five produce a view rather than data, so the graph walks past
    // them. Any of them reaching a kernel would be a crash.
    const nops = [_]ggml.impl.ggml_c.enum_ggml_op{
        ggml.impl.ggml_c.GGML_OP_NONE,      ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_PERMUTE,
        ggml.impl.ggml_c.GGML_OP_TRANSPOSE, ggml.impl.ggml_c.GGML_OP_VIEW,
    };

    var params: ggml.cpu.defs.ComputeParams = std.mem.zeroes(ggml.cpu.defs.ComputeParams);
    params.nth = 1;

    for (nops) |op| {
        var tensor: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
        tensor.op = op;
        tensor.ne = .{ 1, 1, 1, 1 };
        tensor.nb = .{ 4, 4, 4, 4 };
        tensor.type = ggml.impl.ggml_c.GGML_TYPE_F32;

        // NONE returns at the top; the other four fall through to an empty
        // switch arm. Neither path may touch `data`, which is null here.
        computeForward(&params, &tensor);
    }
}
