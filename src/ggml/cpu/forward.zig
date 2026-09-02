//! The op dispatch: given a graph node, run the CPU kernel for its op.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! `ggml_compute_forward` at line 1711.
//!
//! # The kernels are still C++
//!
//! Every arm below calls into `ggml-cpu/ops.cpp`, which this port does not
//! touch. Only the dispatch is Zig, plus `mul_mat` and `mul_mat_id`, which
//! live in `mulmat.zig` because they coordinate threads rather than just
//! computing.
//!
//! The kernels are declared by hand rather than reached through `@cImport`
//! because `ops.h` cannot see `struct ggml_compute_params` -- it includes only
//! `ggml.h`, where the type is never defined -- so translate-c would hand back
//! an opaque pointer and every one of these calls would need a cast.
//!
//! # The switch is exhaustive on purpose
//!
//! The C's `switch` covers all 102 members of `enum ggml_op` and ends in a
//! `default` that aborts, so a new op added upstream fails loudly rather than
//! silently computing nothing. Zig would let the `else` stand alone; keeping
//! every op named means the compiler flags the addition instead.

const std = @import("std");
const impl = @import("../impl.zig");
const types = @import("../types.zig");
const defs = @import("defs.zig");
const mulmat = @import("mulmat.zig");
const c = impl.c;

const Tensor = defs.Tensor;
const ComputeParams = defs.ComputeParams;

/// Build-time switch that makes this file abort on a hot path.
///
/// Off by default and compiled out entirely. Separate from `alloc.zig`'s
/// `probe_ported` because a single flag would only ever prove whichever site
/// runs first, and these two run in different phases: allocation while the
/// graph is being planned, dispatch while it is being computed.
const probe_cpu = @import("config").probe_cpu;

/// Aborts if the probe is enabled, proving the CPU dispatch is reached.
///
/// It is reached: with Metal carrying the model, most ops never touch this
/// file, but some do fall back and the probe fires on the first of them. That
/// is exactly what makes it worth having -- "the port links" and "the port
/// runs" are different claims.
inline fn probe() void {
    if (probe_cpu) @panic(@import("../alloc.zig").probe_marker);
}

/// `ggml-cpu/traits.cpp`, still C++.
///
/// Returns true when the tensor carries an extra-buffer accelerator that has
/// already computed the result, in which case the switch is skipped entirely.
extern fn ggml_cpu_extra_compute_forward(params: *const ComputeParams, op: *Tensor) bool;

/// The op kernels, all of them still C++ in `ggml-cpu/ops.cpp`.
///
/// Signatures come from `ggml-cpu/ops.h`. `flash_attn_back` is the only one
/// that takes an extra argument.
const kernels = struct {
    extern fn ggml_compute_forward_dup(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_add(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_add_id(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_add1(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_acc(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_sub(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_mul(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_div(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_sqr(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_sqrt(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_log(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_sin(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_cos(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_sum(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_sum_rows(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_cumsum(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_mean(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_argmax(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_count_equal(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_repeat(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_repeat_back(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_concat(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_silu_back(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_norm(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_rms_norm(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_rms_norm_back(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_group_norm(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_l2_norm(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_out_prod(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_scale(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_set(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_cpy(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_cont(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_get_rows(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_get_rows_back(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_set_rows(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_diag(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_diag_mask_inf(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_diag_mask_zero(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_soft_max(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_soft_max_ext_back(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_rope(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_rope_back(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_clamp(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_conv_transpose_1d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_im2col(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_im2col_back_f32(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_im2col_3d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_col2im_1d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_conv_2d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_conv_3d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_conv_2d_dw(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_conv_transpose_2d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_pool_1d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_pool_2d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_pool_2d_back(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_upscale(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_pad(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_pad_reflect_1d(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_roll(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_arange(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_timestep_embedding(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_argsort(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_top_k(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_leaky_relu(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_tri(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_fill(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_flash_attn_ext(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_flash_attn_back(params: *const ComputeParams, masked: bool, dst: *Tensor) void;
    extern fn ggml_compute_forward_ssm_conv(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_ssm_scan(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_win_part(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_win_unpart(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_unary(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_glu(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_get_rel_pos(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_add_rel_pos(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_rwkv_wkv6(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_gla(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_rwkv_wkv7(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_solve_tri(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_gated_delta_net(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_lightning_indexer(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_dsv4_hc_comb(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_dsv4_hc_pre(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_dsv4_hc_post(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_map_custom1(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_map_custom2(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_map_custom3(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_custom(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_cross_entropy_loss(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_cross_entropy_loss_back(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_opt_step_adamw(params: *const ComputeParams, dst: *Tensor) void;
    extern fn ggml_compute_forward_opt_step_sgd(params: *const ComputeParams, dst: *Tensor) void;
};

/// Ports `ggml_compute_forward` (ggml-cpu.c:1711 @c1d0e7a00).
///
/// Parameters:
/// - `params`: this thread's index, count, and slice of the work buffer.
/// - `tensor`: the node to compute, in place, into its own `data`.
pub fn computeForward(params: *const ComputeParams, tensor: *Tensor) void {
    probe();

    if (tensor.op == c.GGML_OP_NONE or types.ggml_is_empty(tensor)) return;

    // An extra-buffer op, handled by an accelerator rather than by us.
    if (ggml_cpu_extra_compute_forward(params, tensor)) return;

    switch (tensor.op) {
        c.GGML_OP_DUP => kernels.ggml_compute_forward_dup(params, tensor),
        c.GGML_OP_ADD => kernels.ggml_compute_forward_add(params, tensor),
        c.GGML_OP_ADD_ID => kernels.ggml_compute_forward_add_id(params, tensor),
        c.GGML_OP_ADD1 => kernels.ggml_compute_forward_add1(params, tensor),
        c.GGML_OP_ACC => kernels.ggml_compute_forward_acc(params, tensor),
        c.GGML_OP_SUB => kernels.ggml_compute_forward_sub(params, tensor),
        c.GGML_OP_MUL => kernels.ggml_compute_forward_mul(params, tensor),
        c.GGML_OP_DIV => kernels.ggml_compute_forward_div(params, tensor),
        c.GGML_OP_SQR => kernels.ggml_compute_forward_sqr(params, tensor),
        c.GGML_OP_SQRT => kernels.ggml_compute_forward_sqrt(params, tensor),
        c.GGML_OP_LOG => kernels.ggml_compute_forward_log(params, tensor),
        c.GGML_OP_SIN => kernels.ggml_compute_forward_sin(params, tensor),
        c.GGML_OP_COS => kernels.ggml_compute_forward_cos(params, tensor),
        c.GGML_OP_SUM => kernels.ggml_compute_forward_sum(params, tensor),
        c.GGML_OP_SUM_ROWS => kernels.ggml_compute_forward_sum_rows(params, tensor),
        c.GGML_OP_CUMSUM => kernels.ggml_compute_forward_cumsum(params, tensor),
        c.GGML_OP_MEAN => kernels.ggml_compute_forward_mean(params, tensor),
        c.GGML_OP_ARGMAX => kernels.ggml_compute_forward_argmax(params, tensor),
        c.GGML_OP_COUNT_EQUAL => kernels.ggml_compute_forward_count_equal(params, tensor),
        c.GGML_OP_REPEAT => kernels.ggml_compute_forward_repeat(params, tensor),
        c.GGML_OP_REPEAT_BACK => kernels.ggml_compute_forward_repeat_back(params, tensor),
        c.GGML_OP_CONCAT => kernels.ggml_compute_forward_concat(params, tensor),
        c.GGML_OP_SILU_BACK => kernels.ggml_compute_forward_silu_back(params, tensor),
        c.GGML_OP_NORM => kernels.ggml_compute_forward_norm(params, tensor),
        c.GGML_OP_RMS_NORM => kernels.ggml_compute_forward_rms_norm(params, tensor),
        c.GGML_OP_RMS_NORM_BACK => kernels.ggml_compute_forward_rms_norm_back(params, tensor),
        c.GGML_OP_GROUP_NORM => kernels.ggml_compute_forward_group_norm(params, tensor),
        c.GGML_OP_L2_NORM => kernels.ggml_compute_forward_l2_norm(params, tensor),
        c.GGML_OP_MUL_MAT => mulmat.ggml_compute_forward_mul_mat(params, tensor),
        c.GGML_OP_MUL_MAT_ID => mulmat.computeForwardMulMatId(params, tensor),
        c.GGML_OP_OUT_PROD => kernels.ggml_compute_forward_out_prod(params, tensor),
        c.GGML_OP_SCALE => kernels.ggml_compute_forward_scale(params, tensor),
        c.GGML_OP_SET => kernels.ggml_compute_forward_set(params, tensor),
        c.GGML_OP_CPY => kernels.ggml_compute_forward_cpy(params, tensor),
        c.GGML_OP_CONT => kernels.ggml_compute_forward_cont(params, tensor),
        c.GGML_OP_GET_ROWS => kernels.ggml_compute_forward_get_rows(params, tensor),
        c.GGML_OP_GET_ROWS_BACK => kernels.ggml_compute_forward_get_rows_back(params, tensor),
        c.GGML_OP_SET_ROWS => kernels.ggml_compute_forward_set_rows(params, tensor),
        c.GGML_OP_DIAG => kernels.ggml_compute_forward_diag(params, tensor),
        c.GGML_OP_DIAG_MASK_INF => kernels.ggml_compute_forward_diag_mask_inf(params, tensor),
        c.GGML_OP_DIAG_MASK_ZERO => kernels.ggml_compute_forward_diag_mask_zero(params, tensor),
        c.GGML_OP_SOFT_MAX => kernels.ggml_compute_forward_soft_max(params, tensor),
        c.GGML_OP_SOFT_MAX_BACK => kernels.ggml_compute_forward_soft_max_ext_back(params, tensor),
        c.GGML_OP_ROPE => kernels.ggml_compute_forward_rope(params, tensor),
        c.GGML_OP_ROPE_BACK => kernels.ggml_compute_forward_rope_back(params, tensor),
        c.GGML_OP_CLAMP => kernels.ggml_compute_forward_clamp(params, tensor),
        c.GGML_OP_CONV_TRANSPOSE_1D => kernels.ggml_compute_forward_conv_transpose_1d(params, tensor),
        c.GGML_OP_IM2COL => kernels.ggml_compute_forward_im2col(params, tensor),
        c.GGML_OP_IM2COL_BACK => kernels.ggml_compute_forward_im2col_back_f32(params, tensor),
        c.GGML_OP_IM2COL_3D => kernels.ggml_compute_forward_im2col_3d(params, tensor),
        c.GGML_OP_COL2IM_1D => kernels.ggml_compute_forward_col2im_1d(params, tensor),
        c.GGML_OP_CONV_2D => kernels.ggml_compute_forward_conv_2d(params, tensor),
        c.GGML_OP_CONV_3D => kernels.ggml_compute_forward_conv_3d(params, tensor),
        c.GGML_OP_CONV_2D_DW => kernels.ggml_compute_forward_conv_2d_dw(params, tensor),
        c.GGML_OP_CONV_TRANSPOSE_2D => kernels.ggml_compute_forward_conv_transpose_2d(params, tensor),
        c.GGML_OP_POOL_1D => kernels.ggml_compute_forward_pool_1d(params, tensor),
        c.GGML_OP_POOL_2D => kernels.ggml_compute_forward_pool_2d(params, tensor),
        c.GGML_OP_POOL_2D_BACK => kernels.ggml_compute_forward_pool_2d_back(params, tensor),
        c.GGML_OP_UPSCALE => kernels.ggml_compute_forward_upscale(params, tensor),
        c.GGML_OP_PAD => kernels.ggml_compute_forward_pad(params, tensor),
        c.GGML_OP_PAD_REFLECT_1D => kernels.ggml_compute_forward_pad_reflect_1d(params, tensor),
        c.GGML_OP_ROLL => kernels.ggml_compute_forward_roll(params, tensor),
        c.GGML_OP_ARANGE => kernels.ggml_compute_forward_arange(params, tensor),
        c.GGML_OP_TIMESTEP_EMBEDDING => kernels.ggml_compute_forward_timestep_embedding(params, tensor),
        c.GGML_OP_ARGSORT => kernels.ggml_compute_forward_argsort(params, tensor),
        c.GGML_OP_TOP_K => kernels.ggml_compute_forward_top_k(params, tensor),
        c.GGML_OP_LEAKY_RELU => kernels.ggml_compute_forward_leaky_relu(params, tensor),
        c.GGML_OP_TRI => kernels.ggml_compute_forward_tri(params, tensor),
        c.GGML_OP_FILL => kernels.ggml_compute_forward_fill(params, tensor),
        c.GGML_OP_FLASH_ATTN_EXT => kernels.ggml_compute_forward_flash_attn_ext(params, tensor),
        c.GGML_OP_FLASH_ATTN_BACK => {
            const t = impl.getOpParamsI32(tensor, 0);
            impl.assert(t == 0 or t == 1, "t == 0 || t == 1");
            kernels.ggml_compute_forward_flash_attn_back(params, t != 0, tensor);
        },
        c.GGML_OP_SSM_CONV => kernels.ggml_compute_forward_ssm_conv(params, tensor),
        c.GGML_OP_SSM_SCAN => kernels.ggml_compute_forward_ssm_scan(params, tensor),
        c.GGML_OP_WIN_PART => kernels.ggml_compute_forward_win_part(params, tensor),
        c.GGML_OP_WIN_UNPART => kernels.ggml_compute_forward_win_unpart(params, tensor),
        c.GGML_OP_UNARY => kernels.ggml_compute_forward_unary(params, tensor),
        c.GGML_OP_GLU => kernels.ggml_compute_forward_glu(params, tensor),
        c.GGML_OP_GET_REL_POS => kernels.ggml_compute_forward_get_rel_pos(params, tensor),
        c.GGML_OP_ADD_REL_POS => kernels.ggml_compute_forward_add_rel_pos(params, tensor),
        c.GGML_OP_RWKV_WKV6 => kernels.ggml_compute_forward_rwkv_wkv6(params, tensor),
        c.GGML_OP_GATED_LINEAR_ATTN => kernels.ggml_compute_forward_gla(params, tensor),
        c.GGML_OP_RWKV_WKV7 => kernels.ggml_compute_forward_rwkv_wkv7(params, tensor),
        c.GGML_OP_SOLVE_TRI => kernels.ggml_compute_forward_solve_tri(params, tensor),
        c.GGML_OP_GATED_DELTA_NET => kernels.ggml_compute_forward_gated_delta_net(params, tensor),
        c.GGML_OP_LIGHTNING_INDEXER => kernels.ggml_compute_forward_lightning_indexer(params, tensor),
        c.GGML_OP_DSV4_HC_COMB => kernels.ggml_compute_forward_dsv4_hc_comb(params, tensor),
        c.GGML_OP_DSV4_HC_PRE => kernels.ggml_compute_forward_dsv4_hc_pre(params, tensor),
        c.GGML_OP_DSV4_HC_POST => kernels.ggml_compute_forward_dsv4_hc_post(params, tensor),
        c.GGML_OP_MAP_CUSTOM1 => kernels.ggml_compute_forward_map_custom1(params, tensor),
        c.GGML_OP_MAP_CUSTOM2 => kernels.ggml_compute_forward_map_custom2(params, tensor),
        c.GGML_OP_MAP_CUSTOM3 => kernels.ggml_compute_forward_map_custom3(params, tensor),
        c.GGML_OP_CUSTOM => kernels.ggml_compute_forward_custom(params, tensor),
        c.GGML_OP_CROSS_ENTROPY_LOSS => kernels.ggml_compute_forward_cross_entropy_loss(params, tensor),
        c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => kernels.ggml_compute_forward_cross_entropy_loss_back(params, tensor),
        c.GGML_OP_OPT_STEP_ADAMW => kernels.ggml_compute_forward_opt_step_adamw(params, tensor),
        c.GGML_OP_OPT_STEP_SGD => kernels.ggml_compute_forward_opt_step_sgd(params, tensor),

        // The C spells these out as five separate no-op cases.
        c.GGML_OP_NONE, c.GGML_OP_RESHAPE, c.GGML_OP_PERMUTE, c.GGML_OP_VIEW, c.GGML_OP_TRANSPOSE => {},

        c.GGML_OP_COUNT => impl.abort("fatal error"),
        else => impl.abort("fatal error"),
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
    var tensor: Tensor = std.mem.zeroes(Tensor);
    tensor.op = c.GGML_OP_ADD;
    tensor.ne = .{ 4, 0, 1, 1 };

    try std.testing.expect(types.ggml_is_empty(&tensor));
}

test "the shape-only ops are no-ops" {
    // These five produce a view rather than data, so the graph walks past
    // them. Any of them reaching a kernel would be a crash.
    const nops = [_]c.enum_ggml_op{
        c.GGML_OP_NONE,      c.GGML_OP_RESHAPE, c.GGML_OP_PERMUTE,
        c.GGML_OP_TRANSPOSE, c.GGML_OP_VIEW,
    };

    var params: ComputeParams = std.mem.zeroes(ComputeParams);
    params.nth = 1;

    for (nops) |op| {
        var tensor: Tensor = std.mem.zeroes(Tensor);
        tensor.op = op;
        tensor.ne = .{ 1, 1, 1, 1 };
        tensor.nb = .{ 4, 4, 4, 4 };
        tensor.type = c.GGML_TYPE_F32;

        // NONE returns at the top; the other four fall through to an empty
        // switch arm. Neither path may touch `data`, which is null here.
        computeForward(&params, &tensor);
    }
}
