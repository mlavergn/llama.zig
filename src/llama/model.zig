//! A model, as far as its metadata: architecture, size class,
//! hyperparameters, vocabulary.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-model.cpp` -- `llama_expert_gating_func_name`,
//!   `LLAMA_ROPE_SCALING_TYPES` and its two lookups, `llama_model_create`,
//!   `llama_model_base`'s `load_stats`, `load_hparams` and `load_vocab`,
//!   `llama_model`'s `arch_name`, `type_name`, `desc` and `print_info`, and
//!   the conditional cases of `llama_model_rope_type`
//! - `llama.cpp/src/models/qwen35.cpp` -- `llama_model_qwen35`'s
//!   `load_arch_hparams`
//! - `llama.cpp/src/llama.cpp` -- the metadata half of `llama_model_load`
//!
//! All at v1.0.0 (`b1b770d90`). The size names and the fixed RoPE rules are
//! generated into `model_table.zig`.
//!
//! # Scope
//!
//! Everything `llama_model_load` does before `load_tensors`. The C++ makes
//! one subclass per architecture; the port has one `Model` and dispatches
//! the architecture-specific step on `arch`, and only Qwen3.5's exists so
//! far. Any other architecture the C++ knows loads a
//! vocabulary-only model completely, and otherwise stops at `error.Unported`
//! once the base hyperparameters are read -- distinct from a C++ exception,
//! so the gate can tell "not yet ported" from "the C++ would have thrown".

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");

fn oom() noreturn {
    mod.impl.abort("llama-model.cpp", 0, "out of memory");
}

/// Ports `llama_expert_gating_func_name` (llama-model.cpp:1160 @b1b770d90).
pub fn expertGatingFuncName(ty: u32) [:0]const u8 {
    return switch (ty) {
        1 => "softmax",
        2 => "sigmoid",
        4 => "sqrtsoftplus",
        else => "unknown",
    };
}

/// Ports `LLAMA_ROPE_SCALING_TYPES` (llama-model.cpp:1169 @b1b770d90), in
/// the map's key order.
const rope_scaling_types = [_]struct { c_int, [:0]const u8 }{
    .{ mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_NONE, "none" },
    .{ mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_LINEAR, "linear" },
    .{ mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_YARN, "yarn" },
    .{ mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_LONGROPE, "longrope" },
};

/// Ports `llama_rope_scaling_type_name` (llama-model.cpp:1176 @b1b770d90).
///
/// Return: the name, or `error.Thrown` with `std`'s `out_of_range` for a type
/// the map lacks, as `map`'s `at` throws.
pub fn ropeScalingTypeName(ty: c_int, exc: *mod.impl.Exception) mod.impl.Thrown![:0]const u8 {
    for (rope_scaling_types) |e| if (e[0] == ty) return e[1];
    return exc.throw(.out_of_range, "map::at:  key not found", .{});
}

/// Ports `llama_rope_scaling_type_from_string` (llama-model.cpp:1180 @b1b770d90).
pub fn ropeScalingTypeFromString(name: []const u8) c_int {
    for (rope_scaling_types) |e| if (std.mem.eql(u8, e[1], name)) return e[0];
    return mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

/// Ports `LLAMA_ROPE_SECTION_ORDERS` (llama-model.cpp:1190 @b1b770d90), in
/// the map's key order.
const rope_section_orders = [_]struct { c_int, [:0]const u8 }{
    .{ mod.hparams.RopeSectionOrder.tyxz, "tyxz" },
    .{ mod.hparams.RopeSectionOrder.zyxt, "zyxt" },
};

/// Ports `llama_rope_section_order_name` (llama-model.cpp:1195 @b1b770d90).
///
/// Return: the name, or `error.Thrown` with `std`'s `out_of_range` for an
/// order the map lacks, as `map`'s `at` throws.
pub fn ropeSectionOrderName(order: c_int, exc: *mod.impl.Exception) mod.impl.Thrown![:0]const u8 {
    for (rope_section_orders) |e| if (e[0] == order) return e[1];
    return exc.throw(.out_of_range, "map::at:  key not found", .{});
}

/// Ports `llama_rope_section_order_from_string` (llama-model.cpp:1199
/// @b1b770d90).
fn ropeSectionOrderFromString(name: []const u8) c_int {
    for (rope_section_orders) |e| if (std.mem.eql(u8, e[1], name)) return e[0];
    return mod.hparams.RopeSectionOrder.unspecified;
}

/// Ports `LLM_CLS_ACT_TYPES_FROM_STRING` (llama-model.cpp:1227 @b1b770d90):
/// the transformers names of a classifier head's activation; `gelu` is the
/// exact (erf) one.
const cls_act_types_from_string = [_]struct { []const u8, c_int }{
    .{ "gelu", mod.impl.llama_c.GGML_UNARY_OP_GELU_ERF },
    .{ "silu", mod.impl.llama_c.GGML_UNARY_OP_SILU },
    .{ "tanh", mod.impl.llama_c.GGML_UNARY_OP_TANH },
};

/// Ports `llama_model_default_params` (llama-model.cpp:3553 @b1b770d90).
pub fn defaultParams() mod.impl.llama_c.llama_model_params {
    return .{
        .devices = null,
        .tensor_buft_overrides = null,
        .n_gpu_layers = -1,
        .split_mode = mod.impl.llama_c.LLAMA_SPLIT_MODE_LAYER,
        .load_mode = mod.impl.llama_c.LLAMA_LOAD_MODE_AUTO,
        .lazy_mode = mod.impl.llama_c.LLAMA_LAZY_MODE_AUTO,
        .main_gpu = 0,
        .tensor_split = null,
        .progress_callback = null,
        .progress_callback_user_data = null,
        .kv_overrides = null,
        .vocab_only = false,
        .check_tensors = false,
        .use_extra_bufts = true,
        .no_host = false,
        .no_alloc = false,
        .load_mtp = false,
    };
}

// BEGIN GENERATED by scripts/gen-layer -- do not edit
/// Ports `llama_layer_posnet` (llama-model.h:172 @b1b770d90).
pub const LayerPosnet = struct {
    // resnet
    norm1: ?*mod.impl.llama_c.ggml_tensor = null,
    norm1_b: ?*mod.impl.llama_c.ggml_tensor = null,

    conv1: ?*mod.impl.llama_c.ggml_tensor = null,
    conv1_b: ?*mod.impl.llama_c.ggml_tensor = null,

    norm2: ?*mod.impl.llama_c.ggml_tensor = null,
    norm2_b: ?*mod.impl.llama_c.ggml_tensor = null,

    conv2: ?*mod.impl.llama_c.ggml_tensor = null,
    conv2_b: ?*mod.impl.llama_c.ggml_tensor = null,

    // attention
    attn_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,

    attn_q: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_q_b: ?*mod.impl.llama_c.ggml_tensor = null,

    attn_k: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_k_b: ?*mod.impl.llama_c.ggml_tensor = null,

    attn_v: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_v_b: ?*mod.impl.llama_c.ggml_tensor = null,

    attn_o: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_o_b: ?*mod.impl.llama_c.ggml_tensor = null,

    // normalize
    norm: ?*mod.impl.llama_c.ggml_tensor = null,
    norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
};

/// Ports `llama_layer_convnext` (llama-model.h:207 @b1b770d90).
pub const LayerConvnext = struct {
    dw: ?*mod.impl.llama_c.ggml_tensor = null,
    dw_b: ?*mod.impl.llama_c.ggml_tensor = null,

    norm: ?*mod.impl.llama_c.ggml_tensor = null,
    norm_b: ?*mod.impl.llama_c.ggml_tensor = null,

    pw1: ?*mod.impl.llama_c.ggml_tensor = null,
    pw1_b: ?*mod.impl.llama_c.ggml_tensor = null,

    pw2: ?*mod.impl.llama_c.ggml_tensor = null,
    pw2_b: ?*mod.impl.llama_c.ggml_tensor = null,

    gamma: ?*mod.impl.llama_c.ggml_tensor = null,
};

/// Ports `llama_layer_shortconv` (llama-model.h:223 @b1b770d90).
pub const LayerShortconv = struct {
    in_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    conv: ?*mod.impl.llama_c.ggml_tensor = null,
    out_proj: ?*mod.impl.llama_c.ggml_tensor = null,
};

/// Ports `llama_layer_nextn` (llama-model.h:229 @b1b770d90).
pub const LayerNextn = struct {
    eh_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    eh_proj_s: ?*mod.impl.llama_c.ggml_tensor = null,
    eh_proj_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    embed_tokens: ?*mod.impl.llama_c.ggml_tensor = null,
    enorm: ?*mod.impl.llama_c.ggml_tensor = null,
    hnorm: ?*mod.impl.llama_c.ggml_tensor = null,
    shared_head_head: ?*mod.impl.llama_c.ggml_tensor = null,
    shared_head_head_s: ?*mod.impl.llama_c.ggml_tensor = null,
    shared_head_head_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    shared_head_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    // qwen4exp: the MTP block collapses its hyper-connection streams with its own mixer
    hc_head_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_head_down: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_head_up: ?*mod.impl.llama_c.ggml_tensor = null,
};

/// Ports `llama_layer_switch_lora` (llama-model.h:247 @b1b770d90).
pub const LayerSwitchLora = struct {
    a_q: ?*mod.impl.llama_c.ggml_tensor = null,
    b_q: ?*mod.impl.llama_c.ggml_tensor = null,
    a_k: ?*mod.impl.llama_c.ggml_tensor = null,
    b_k: ?*mod.impl.llama_c.ggml_tensor = null,
    a_v: ?*mod.impl.llama_c.ggml_tensor = null,
    b_v: ?*mod.impl.llama_c.ggml_tensor = null,
    a_o: ?*mod.impl.llama_c.ggml_tensor = null,
    b_o: ?*mod.impl.llama_c.ggml_tensor = null,

    a_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    b_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    a_up: ?*mod.impl.llama_c.ggml_tensor = null,
    b_up: ?*mod.impl.llama_c.ggml_tensor = null,
    a_down: ?*mod.impl.llama_c.ggml_tensor = null,
    b_down: ?*mod.impl.llama_c.ggml_tensor = null,
};

/// Ports `llama_layer` (llama-model.h:265 @b1b770d90): one member for
/// every tensor any architecture names, in the C++'s order.
pub const Layer = struct {
    // normalization
    attn_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_norm_2: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_norm_2_b: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_q_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_q_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_k_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_k_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_out_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_out_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_q_a_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_kv_a_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_sub_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_post_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_sub_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_norm_cross: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_norm_enc: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_dt_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_b_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_c_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    // attention
    wq: ?*mod.impl.llama_c.ggml_tensor = null,
    wk: ?*mod.impl.llama_c.ggml_tensor = null,
    wv: ?*mod.impl.llama_c.ggml_tensor = null,
    k_cache_s: ?*mod.impl.llama_c.ggml_tensor = null,
    v_cache_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wo: ?*mod.impl.llama_c.ggml_tensor = null,
    wqkv: ?*mod.impl.llama_c.ggml_tensor = null,
    wg: ?*mod.impl.llama_c.ggml_tensor = null,
    wq_a: ?*mod.impl.llama_c.ggml_tensor = null,
    wq_b: ?*mod.impl.llama_c.ggml_tensor = null,
    wkv_a_mqa: ?*mod.impl.llama_c.ggml_tensor = null,
    wkv_b: ?*mod.impl.llama_c.ggml_tensor = null,
    wkv: ?*mod.impl.llama_c.ggml_tensor = null,
    wk_b: ?*mod.impl.llama_c.ggml_tensor = null,
    wv_b: ?*mod.impl.llama_c.ggml_tensor = null,
    wqkv_b: ?*mod.impl.llama_c.ggml_tensor = null,
    wo_a: ?*mod.impl.llama_c.ggml_tensor = null,
    wo_b: ?*mod.impl.llama_c.ggml_tensor = null,
    wq_cross: ?*mod.impl.llama_c.ggml_tensor = null,
    wk_cross: ?*mod.impl.llama_c.ggml_tensor = null,
    wv_cross: ?*mod.impl.llama_c.ggml_tensor = null,
    wo_cross: ?*mod.impl.llama_c.ggml_tensor = null,
    wq_enc: ?*mod.impl.llama_c.ggml_tensor = null,
    wk_enc: ?*mod.impl.llama_c.ggml_tensor = null,
    wv_enc: ?*mod.impl.llama_c.ggml_tensor = null,
    wo_enc: ?*mod.impl.llama_c.ggml_tensor = null,
    wqkv_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    wqkv_gate_b: ?*mod.impl.llama_c.ggml_tensor = null,
    // K2 Horizon MoVA
    attn_v_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_v_gate_b: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_v_exps: ?*mod.impl.llama_c.ggml_tensor = null,

    // relative position bias
    attn_rel_b: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_rel_b_enc: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_rel_b_cross: ?*mod.impl.llama_c.ggml_tensor = null,

    // normalization
    ffn_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_post_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_post_norm_1: ?*mod.impl.llama_c.ggml_tensor = null, // gemma4
    ffn_post_norm_2: ?*mod.impl.llama_c.ggml_tensor = null, // gemma4
    ffn_pre_norm_2: ?*mod.impl.llama_c.ggml_tensor = null, // gemma4
    ffn_norm_ada_down: ?*mod.impl.llama_c.ggml_tensor = null, // voxtral realtime
    ffn_norm_ada_up: ?*mod.impl.llama_c.ggml_tensor = null, // voxtral realtime
    layer_out_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    layer_out_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_norm_exps: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_norm_enc: ?*mod.impl.llama_c.ggml_tensor = null,

    // ff
    ffn_gate: ?*mod.impl.llama_c.ggml_tensor = null, // w1
    ffn_down: ?*mod.impl.llama_c.ggml_tensor = null, // w2
    ffn_up: ?*mod.impl.llama_c.ggml_tensor = null, // w3
    ffn_gate_enc: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_enc: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_enc: ?*mod.impl.llama_c.ggml_tensor = null,

    // ff MoE
    ffn_gate_inp: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_inp_s: ?*mod.impl.llama_c.ggml_tensor = null, // gemma4
    ffn_gate_exps: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_exps: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_exps: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_up_exps: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_inp_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_exps_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_exps_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_exps_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_up_exps_b: ?*mod.impl.llama_c.ggml_tensor = null,

    // ff MoE per-expert scales (NVFP4 per-tensor scale2)
    ffn_gate_exps_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_exps_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_exps_s: ?*mod.impl.llama_c.ggml_tensor = null,

    // ff MoE latent proj
    ffn_latent_down: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_latent_up: ?*mod.impl.llama_c.ggml_tensor = null,

    // ff MoE norms around the experts, in latent space
    ffn_pre_exp_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_post_exp_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    // norm preserving residual coefficients
    attn_resid_alpha: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_resid_beta: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_resid_alpha: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_resid_beta: ?*mod.impl.llama_c.ggml_tensor = null,

    // value residual mix
    attn_vres_lambda: ?*mod.impl.llama_c.ggml_tensor = null,

    // ff shared expert (shexp)
    ffn_gate_inp_shexp: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_shexp: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_shexp: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_shexp: ?*mod.impl.llama_c.ggml_tensor = null,

    // ff adjugate experts (chexps)
    ffn_gate_chexps: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_chexps: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_chexps: ?*mod.impl.llama_c.ggml_tensor = null,

    // ff bias
    ffn_gate_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_b: ?*mod.impl.llama_c.ggml_tensor = null, // b2
    ffn_up_b: ?*mod.impl.llama_c.ggml_tensor = null, // b3
    ffn_act: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_exp_probs_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_exp_probs_b_vl: ?*mod.impl.llama_c.ggml_tensor = null, // deepseek4 vision (bias for image tokens)
    ffn_gate_tid2eid: ?*mod.impl.llama_c.ggml_tensor = null,

    dflash_attn_conv_base: ?*mod.impl.llama_c.ggml_tensor = null,
    dflash_attn_conv_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    dflash_ffn_conv_base: ?*mod.impl.llama_c.ggml_tensor = null,
    dflash_ffn_conv_proj: ?*mod.impl.llama_c.ggml_tensor = null,

    lilicorr_attn_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_mlp_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_attn_in_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_attn_in_proj_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_attn_out_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_attn_out_proj_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_attn_out_proj_s: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_mlp_0: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_mlp_0_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_mlp_0_s: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_mlp_2: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_mlp_2_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_mlp_2_s: ?*mod.impl.llama_c.ggml_tensor = null,

    // mamba proj
    ssm_in: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_x: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_dt: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_out: ?*mod.impl.llama_c.ggml_tensor = null,

    // mamba
    ssm_conv1d: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_a: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_d: ?*mod.impl.llama_c.ggml_tensor = null,

    // mamba bias
    ssm_conv1d_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_dt_b: ?*mod.impl.llama_c.ggml_tensor = null,

    // qwen3next
    ssm_beta_alpha: ?*mod.impl.llama_c.ggml_tensor = null,

    // qwen3.5
    ssm_alpha: ?*mod.impl.llama_c.ggml_tensor = null,

    // rwkv
    time_mix_w1: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_w2: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_lerp_x: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_lerp_w: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_lerp_k: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_lerp_v: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_lerp_r: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_lerp_g: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_lerp_fused: ?*mod.impl.llama_c.ggml_tensor = null,

    time_mix_first: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_decay: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_decay_w1: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_decay_w2: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_key: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_key_b: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_value: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_value_b: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_receptance: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_receptance_b: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_gate: ?*mod.impl.llama_c.ggml_tensor = null,

    // rwkv7
    time_mix_w0: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_a0: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_a1: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_a2: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_v0: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_v1: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_v2: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_g1: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_g2: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_k_k: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_k_a: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_r_k: ?*mod.impl.llama_c.ggml_tensor = null,

    time_mix_ln: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_ln_b: ?*mod.impl.llama_c.ggml_tensor = null,
    time_mix_output: ?*mod.impl.llama_c.ggml_tensor = null,

    channel_mix_lerp_k: ?*mod.impl.llama_c.ggml_tensor = null,
    channel_mix_lerp_r: ?*mod.impl.llama_c.ggml_tensor = null,

    channel_mix_key: ?*mod.impl.llama_c.ggml_tensor = null,
    channel_mix_receptance: ?*mod.impl.llama_c.ggml_tensor = null,
    channel_mix_value: ?*mod.impl.llama_c.ggml_tensor = null,

    // long rope factors
    rope_long: ?*mod.impl.llama_c.ggml_tensor = null,
    rope_short: ?*mod.impl.llama_c.ggml_tensor = null,
    rope_freqs: ?*mod.impl.llama_c.ggml_tensor = null,

    // bitnet scale
    wq_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wk_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wv_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wo_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wqkv_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wqkv_gate_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_shexp_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_shexp_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_shexp_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_out_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_alpha_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_beta_s: ?*mod.impl.llama_c.ggml_tensor = null,

    // input scales
    wq_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wk_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wv_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wo_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wqkv_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    wqkv_gate_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_exps_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_exps_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_exps_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_gate_shexp_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_up_shexp_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_down_shexp_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_in_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_out_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_alpha_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_beta_in_s: ?*mod.impl.llama_c.ggml_tensor = null,

    // altup & laurel
    per_layer_inp_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    per_layer_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    per_layer_post_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    altup_correct_coef: ?*mod.impl.llama_c.ggml_tensor = null,
    altup_correct_scale: ?*mod.impl.llama_c.ggml_tensor = null,
    altup_predict_coef: ?*mod.impl.llama_c.ggml_tensor = null,
    altup_router: ?*mod.impl.llama_c.ggml_tensor = null,
    altup_router_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    laurel_l: ?*mod.impl.llama_c.ggml_tensor = null,
    laurel_r: ?*mod.impl.llama_c.ggml_tensor = null,
    laurel_post_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    // openai-moe
    attn_sinks: ?*mod.impl.llama_c.ggml_tensor = null,

    // DeepSeek-V4
    attn_kv_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_fn: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_base: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_scale: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_fn: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_base: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_scale: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_comp_wkv: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_comp_wgate: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_comp_ape: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_comp_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_comp_wkv: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_comp_wgate: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_comp_ape: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_comp_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    // cogvlm
    visexp_attn_wqkv: ?*mod.impl.llama_c.ggml_tensor = null,
    visexp_attn_wo: ?*mod.impl.llama_c.ggml_tensor = null,
    visexp_ffn_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    visexp_ffn_down: ?*mod.impl.llama_c.ggml_tensor = null,
    visexp_ffn_up: ?*mod.impl.llama_c.ggml_tensor = null,

    // xIELU activation parameters for Apertus
    ffn_act_alpha_n: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_act_alpha_p: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_act_beta: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_act_eps: ?*mod.impl.llama_c.ggml_tensor = null,

    // Kimi Linear KDA (using ssm_ prefix for consistency)
    // Note: ssm_dt_b already exists above (mamba bias), reused for Kimi dt_bias
    ssm_q_conv: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_k_conv: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_v_conv: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_f_a: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_f_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_beta: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_g_a: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_g_b: ?*mod.impl.llama_c.ggml_tensor = null,
    ssm_o_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    // kimi-k3
    ssm_g: ?*mod.impl.llama_c.ggml_tensor = null, // full-rank KDA gate (replaces ssm_g_a/ssm_g_b)
    attn_res_score: ?*mod.impl.llama_c.ggml_tensor = null, // fused res_norm*res_proj, pre-attention
    ffn_res_score: ?*mod.impl.llama_c.ggml_tensor = null, // fused res_norm*res_proj, pre-FFN
    ffn_routed_down: ?*mod.impl.llama_c.ggml_tensor = null, // latent MoE: n_embd -> n_expert_latent
    ffn_routed_up: ?*mod.impl.llama_c.ggml_tensor = null, // latent MoE: n_expert_latent -> n_embd
    ffn_routed_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    // DSA (deepseek sparse attention)
    indexer_k_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_k_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_attn_k: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_attn_q_b: ?*mod.impl.llama_c.ggml_tensor = null, // note: for lora a/b, not bias
    indexer_q_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_q_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,

    // SSMax: per-head query scale s * log(pos + 1) + b
    attn_ssmax_s: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_ssmax_b: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_ssmax_s: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_ssmax_b: ?*mod.impl.llama_c.ggml_tensor = null,

    // glm5-next k-pool indexer
    indexer_kpool_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    indexer_kpool_ape: ?*mod.impl.llama_c.ggml_tensor = null,

    // deepseek41
    engram_kv: ?*mod.impl.llama_c.ggml_tensor = null,
    engram_gate: ?*mod.impl.llama_c.ggml_tensor = null, // q_weight*k_weight folded

    // MSA
    index_q_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    index_k_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    index_q_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    index_k_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    hc_attn_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_down: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_up: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_inject: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_down: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_up: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_inject: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_down_inject: ?*mod.impl.llama_c.ggml_tensor = null, // [hc_attn_down | hc_attn_inject] in one weight
    hc_ffn_down_inject: ?*mod.impl.llama_c.ggml_tensor = null, // [hc_ffn_down | hc_ffn_inject] in one weight
    wqkv_z: ?*mod.impl.llama_c.ggml_tensor = null, // [wqkv | wqkv_gate] in one weight
    ssm_beta_alpha_cat: ?*mod.impl.llama_c.ggml_tensor = null, // [ssm_beta | ssm_alpha] in one weight, unlike the grouped ssm_beta_alpha

    ple_key: ?*mod.impl.llama_c.ggml_tensor = null,
    ple_value: ?*mod.impl.llama_c.ggml_tensor = null,
    ple_norm_key: ?*mod.impl.llama_c.ggml_tensor = null,
    ple_norm_query: ?*mod.impl.llama_c.ggml_tensor = null,
    ple_norm_conv: ?*mod.impl.llama_c.ggml_tensor = null,
    ple_conv1d: ?*mod.impl.llama_c.ggml_tensor = null,

    // gemma4 layer output scale, reused for talkie embedding skip scale
    out_scale: ?*mod.impl.llama_c.ggml_tensor = null,

    // metis16 per-pass hyper-connection mixes, attention pass LoRA, expert route embeddings
    hc_attn_mix: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_read: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_attn_write: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_mix: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_read: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_ffn_write: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_pass_lora_down: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_pass_lora_up: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_pass_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_exp_embd: ?*mod.impl.llama_c.ggml_tensor = null,

    // limite
    attn_ve_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_xsa_alpha: ?*mod.impl.llama_c.ggml_tensor = null,
    attn_resid_scale: ?*mod.impl.llama_c.ggml_tensor = null,
    ffn_resid_scale: ?*mod.impl.llama_c.ggml_tensor = null,
    mudd_down: ?*mod.impl.llama_c.ggml_tensor = null,
    mudd_up: ?*mod.impl.llama_c.ggml_tensor = null,
    mudd_up_b: ?*mod.impl.llama_c.ggml_tensor = null,

    posnet: LayerPosnet = .{},

    convnext: LayerConvnext = .{},

    shortconv: LayerShortconv = .{},

    nextn: LayerNextn = .{},

    switch_lora: LayerSwitchLora = .{},
};
// END GENERATED by scripts/gen-layer

/// Mirrors `llama_device` (llama-model.h:687 @b1b770d90).
pub const Device = struct {
    is_meta: bool,
    dev: mod.impl.llama_c.ggml_backend_dev_t,
};

/// `llama_model::impl`'s `layer_dev` (llama-model.cpp:1377 @b1b770d90): the
/// device a layer is on and the buffer types it may use.
pub const LayerDev = struct {
    dev: mod.impl.llama_c.ggml_backend_dev_t = null,
    buft_list: ?*mod.loader.BuftList = null,
};

/// A tensor context and the backend buffers holding its tensors' data.
pub const CtxBufs = struct {
    ctx: *mod.impl.llama_c.ggml_context,
    bufs: std.ArrayList(mod.impl.llama_c.ggml_backend_buffer_t) = .empty,
};

/// A GPU device's buffer types, `gpu_buft_list`'s entries.
const GpuBufts = struct {
    dev: mod.impl.llama_c.ggml_backend_dev_t,
    list: *mod.loader.BuftList,
};

/// One entry of `memory_breakdown`, in buffer-type pointer order.
pub const BuftSize = struct {
    buft: mod.impl.llama_c.ggml_backend_buffer_type_t,
    size: usize,
};

/// Ports `llama_hadamard_transform` (llama-model.h:714 @b1b770d90): the
/// transform of a folded weight, applied to the matmul input -- an optional
/// sign flip, then the normalized block Hadamard rotation.
pub const HadamardTransform = struct {
    rot: *mod.impl.llama_c.ggml_tensor,
    /// Null for the identity sign mode.
    signs: ?*mod.impl.llama_c.ggml_tensor = null,

    /// When `perm_rep > 1`, the input is permuted from the tiled head order
    /// `[hd, nk, rep]` to the grouped order `[hd, rep, nk]` before the signs
    /// and the rotation.
    perm_hd: i64 = 0,
    perm_nk: i64 = 0,
    perm_rep: i64 = 0,

    /// Ports `llama_hadamard_transform`'s `set_gdn_v_perm` (llama-model.h:725
    /// @b1b770d90): a gated delta net's `ssm_out` gets its value heads in
    /// tiled order where the fold used the grouped order; records the head
    /// geometry for that permutation.
    ///
    /// Return: false when it does not match the input width.
    pub fn setGdnVPerm(self: *HadamardTransform, n_in: i64, n_v: i64, n_k: i64) bool {
        if (n_k <= 0 or n_v <= 0 or @rem(n_v, n_k) != 0 or @rem(n_in, n_v) != 0) {
            return false;
        }
        self.perm_hd = @divTrunc(n_in, n_v);
        self.perm_nk = n_k;
        self.perm_rep = @divTrunc(n_v, n_k);
        return true;
    }
};

/// `llama_hadamard_rotations` (llama-model.h:735 @b1b770d90): a weight to
/// its transform. Only looked up, so any order stands in for the C++'s
/// `std::unordered_map`.
pub const HadamardRotations = std.AutoHashMapUnmanaged(*const mod.impl.llama_c.ggml_tensor, HadamardTransform);

/// Ports `llama_hadamard` (llama-model.h:737 @b1b770d90): the folded weights
/// of a `prism.hadamard` model.
pub const Hadamard = struct {
    /// Names and block sizes, from the metadata in `load_hparams`; the
    /// transforms are made in `load_tensors`, in these maps' libc++ order,
    /// which decides the buffer type the inverse transforms share and
    /// which missing weight is reported.
    weight_blocks: ?stdlib.libcxx.StringMap(u32) = null,
    inverse_blocks: ?stdlib.libcxx.StringMap(u32) = null,

    /// `std::map<uint32_t, std::vector<int32_t>>`, kept sorted by width.
    sign_data: std.ArrayList(SignData) = .empty,

    gdn_v_grouped: bool = false,
    tied_output: bool = false,

    /// A folded weight to its activation transform.
    rot: HadamardRotations = .empty,
    /// A latent lookup table to its inverse transform.
    inv: HadamardRotations = .empty,

    pub const SignData = struct {
        width: u32,
        values: []i32,
    };

    /// Return: nothing.
    pub fn deinit(self: *Hadamard) void {
        if (self.weight_blocks) |*m| m.deinit();
        if (self.inverse_blocks) |*m| m.deinit();
        for (self.sign_data.items) |sd| std.heap.c_allocator.free(sd.values);
        self.sign_data.deinit(std.heap.c_allocator);
        self.rot.deinit(std.heap.c_allocator);
        self.inv.deinit(std.heap.c_allocator);
        self.* = .{};
    }

    fn findSigns(self: *const Hadamard, width: u32) ?*const SignData {
        for (self.sign_data.items) |*sd| if (sd.width == width) return sd;
        return null;
    }

    fn blocks(m: *?stdlib.libcxx.StringMap(u32)) *stdlib.libcxx.StringMap(u32) {
        if (m.* == null) m.* = stdlib.libcxx.StringMap(u32).init(std.heap.c_allocator) catch oom();
        return &m.*.?;
    }
};

/// Ports `llama_rope_params` (llama-model.h:751 @b1b770d90): a layer's RoPE,
/// as the model graph and the K-shift both apply it.
pub const RopeParams = struct {
    /// 0 if the layer has no RoPE.
    n_rot: i32 = 0,
    /// The first rotated dim in a cached K head.
    n_offs: i32 = 0,
    type: i32 = mod.impl.llama_c.LLAMA_ROPE_TYPE_NONE,
    n_ctx_orig: i32 = 0,

    freq_base: f32 = 0.0,
    freq_scale: f32 = 0.0,
    ext_factor: f32 = 0.0,
    attn_factor: f32 = 0.0,
    beta_fast: f32 = 0.0,
    beta_slow: f32 = 0.0,

    factors: ?*mod.impl.llama_c.ggml_tensor = null,
    /// The K norm weight applied after RoPE: a K-shift has to undo it.
    k_norm: ?*mod.impl.llama_c.ggml_tensor = null,
};

/// `llama_model`, as far as `print_info`.
/// Ports `llama_prec_policy` (llama-model.h:700 @b1b770d90): the
/// per-weight activation precision a model's metadata asks for.
///
/// `prec_src1` is a `std::unordered_map` keyed by the weight's address in
/// the C++, which only the saver iterates -- in an order that depends on
/// those addresses, so nothing reproducible. Here it keeps the order
/// `load` inserted, `tensors_by_name`'s.
pub const PrecPolicy = struct {
    /// Keyed by the weight `res->src[0]`: the accumulation type the op
    /// should use. Unused upstream for now, and never filled.
    prec_acc: std.AutoArrayHashMapUnmanaged(*const mod.impl.llama_c.ggml_tensor, mod.impl.llama_c.enum_ggml_prec) = .empty,
    /// Keyed by the weight `res->src[0]`: the activation precision.
    prec_src1: std.AutoArrayHashMapUnmanaged(*const mod.impl.llama_c.ggml_tensor, mod.impl.llama_c.enum_ggml_prec) = .empty,

    pub fn deinit(self: *PrecPolicy) void {
        self.prec_acc.deinit(std.heap.c_allocator);
        self.prec_src1.deinit(std.heap.c_allocator);
    }

    /// Ports `llama_prec_policy`'s `apply` (llama-model.cpp:1391 @b1b770d90):
    /// a matrix product's activation precision, from its weight.
    ///
    /// Return: whether a precision was set.
    pub fn apply(self: *const PrecPolicy, res: ?*mod.impl.llama_c.ggml_tensor) bool {
        const r = res orelse return false;
        const w = r.src[0] orelse return false;
        const prec = self.prec_src1.get(w) orelse return false;
        return mod.impl.llama_c.ggml_prec_set_src(r, prec, 1);
    }

    /// Ports `llama_prec_policy`'s `load` (llama-model.cpp:1404 @b1b770d90):
    /// the weights named in `general.tensor_extra.name` whose
    /// `general.tensor_extra.prec_a4` entry is false cannot take 4-bit
    /// activations, and keep theirs at 8 bits.
    ///
    /// Return: nothing, or `error.Thrown` for a malformed pair of keys.
    pub fn load(self: *PrecPolicy, ml: *const mod.loader.Loader, model: *const Model, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        const names = (try ml.meta.getArrList([]const u8, .general_tensor_extra_name, false, exc)) orelse return;
        defer std.heap.c_allocator.free(names);

        const ctx = ml.meta.ctx;
        const key = ml.meta.name(.general_tensor_extra_prec_a4);
        const kid = ml.meta.findNamed(key.slice());
        if (kid < 0 or mod.impl.llama_c.gguf_get_kv_type(ctx, kid) != mod.impl.llama_c.GGUF_TYPE_ARRAY or mod.impl.llama_c.gguf_get_arr_type(ctx, kid) != mod.impl.llama_c.GGUF_TYPE_BOOL) {
            return exc.throw(.runtime_error, "{s} must be a bool array", .{key.slice()});
        }

        const n_values = mod.impl.llama_c.gguf_get_arr_n(ctx, kid);
        if (n_values != names.len) {
            return exc.throw(.runtime_error, "{s} tensor/value length mismatch ({d} vs {d})", .{ key.slice(), names.len, n_values });
        }

        // the weights that cannot take 4-bit activations keep src1 at a
        // higher precision
        const values: [*]const i8 = @ptrCast(mod.impl.llama_c.gguf_get_arr_data(ctx, kid).?);
        var want: std.StringHashMapUnmanaged(void) = .empty;
        defer want.deinit(std.heap.c_allocator);
        for (0..n_values) |i| {
            if (values[i] == 0) want.put(std.heap.c_allocator, names[i], {}) catch oom();
        }

        // resolve the names to tensors
        for (model.tensors_by_name.items) |w| {
            if (want.contains(mod.gguf_meta.cString(&w.name))) {
                const gop = self.prec_src1.getOrPut(std.heap.c_allocator, w) catch oom();
                if (!gop.found_existing) gop.value_ptr.* = mod.impl.llama_c.GGML_PREC_Q8;
            }
        }
    }
};

pub const Model = struct {
    type: mod.model_table.Type = .unknown,
    arch: mod.arch_table.Arch = .unknown,
    /// `general.name`.
    name: std.ArrayList(u8) = .empty,
    hparams: mod.hparams.Hparams = .{},
    vocab: mod.vocab.Vocab,
    /// The per-weight activation precision policy.
    prec_policy: PrecPolicy = .{},
    /// The `prism.hadamard` folded weights.
    hdmd: Hadamard = .{},
    /// Whether the KV cache rotates its attention, for the contexts created
    /// from this model; see `llama_model_set_attn_rot`.
    attn_rot: bool = true,
    /// `can_prefetch` (llama-model.h:950 @b1b770d90): the weights whose rows
    /// a graph input may prefetch before a gather.
    can_prefetch: std.AutoHashMapUnmanaged(*const mod.impl.llama_c.ggml_tensor, void) = .empty,
    classifier_labels: std.ArrayList([:0]u8) = .empty,
    decision_labels: std.ArrayList([:0]u8) = .empty,

    /// deepseek41 Engram: token id -> id of its normalized form, hashed into
    /// the n-grams.
    engram_token_map: std.ArrayList(i32) = .empty,
    /// `llama_model_sensenova_u1`'s `rope_freq_base_spatial`
    /// (src/models/models.h:559 @b1b770d90).
    rope_freq_base_spatial: f32 = 10000.0,
    /// Every scalar metadata value as text, in `std`'s `unordered_map` order.
    gguf_kv: stdlib.libcxx.StringMap([:0]u8),

    // `llama_model`'s `impl`
    n_elements: u64 = 0,
    n_bytes: usize = 0,
    desc_str: std.ArrayList(u8) = .empty,
    ftype: c_uint = mod.impl.llama_c.LLAMA_FTYPE_ALL_F32,

    /// The parameters it was loaded with; `tensor_split` points into
    /// `tensor_split_owned` when one was given.
    params: mod.impl.llama_c.llama_model_params,
    tensor_split_owned: [16]f32 = @splat(0),
    has_tensor_overrides: bool = false,

    devices: std.ArrayList(Device) = .empty,

    // BEGIN GENERATED by scripts/gen-layer (llama_model) -- do not edit
    // The tensors of `llama_model` (llama-model.h:768 @b1b770d90), in its order.
    tok_embd: ?*mod.impl.llama_c.ggml_tensor = null,
    type_embd: ?*mod.impl.llama_c.ggml_tensor = null,
    pos_embd: ?*mod.impl.llama_c.ggml_tensor = null,
    tok_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    tok_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    output_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    output_res_score: ?*mod.impl.llama_c.ggml_tensor = null, // kimi-k3: final cross-layer residual mix
    output_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    output: ?*mod.impl.llama_c.ggml_tensor = null,
    output_b: ?*mod.impl.llama_c.ggml_tensor = null,
    output_norm_enc: ?*mod.impl.llama_c.ggml_tensor = null,

    // NVFP4 per-tensor scale2, input_scale for LM head
    output_s: ?*mod.impl.llama_c.ggml_tensor = null,
    output_in_s: ?*mod.impl.llama_c.ggml_tensor = null,

    // NextN/MTP model-level projections
    nextn_proj_pre: ?*mod.impl.llama_c.ggml_tensor = null,
    nextn_proj_post: ?*mod.impl.llama_c.ggml_tensor = null,

    // hrm-text initial low-cycle state
    hrm_z_l_init: ?*mod.impl.llama_c.ggml_tensor = null,

    // metis16 hyper-connection stream embeddings, depth memory and continuation head
    stream_embd: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_state_write: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_meta_write: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_entry_bias: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_query: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_query_b: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_key: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_key_b: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_value: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_value_b: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_out: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_out_b: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_route_state: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_route_retr: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_route_diff: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_route_hist: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_route_b: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_stream_gate: ?*mod.impl.llama_c.ggml_tensor = null,
    depth_mem_stream_gate_b: ?*mod.impl.llama_c.ggml_tensor = null,
    cont_hidden_state: ?*mod.impl.llama_c.ggml_tensor = null,
    cont_hidden_retr: ?*mod.impl.llama_c.ggml_tensor = null,
    cont_hidden_diff: ?*mod.impl.llama_c.ggml_tensor = null,
    cont_hidden_route: ?*mod.impl.llama_c.ggml_tensor = null,
    cont_hidden_b: ?*mod.impl.llama_c.ggml_tensor = null,
    cont_output: ?*mod.impl.llama_c.ggml_tensor = null,
    cont_output_b: ?*mod.impl.llama_c.ggml_tensor = null,

    // DeepSeek-V4
    hc_head_fn: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_head_base: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_head_scale: ?*mod.impl.llama_c.ggml_tensor = null,

    // classifier
    cls: ?*mod.impl.llama_c.ggml_tensor = null,
    cls_b: ?*mod.impl.llama_c.ggml_tensor = null,
    cls_out: ?*mod.impl.llama_c.ggml_tensor = null,
    cls_out_b: ?*mod.impl.llama_c.ggml_tensor = null,
    cls_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    cls_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    conv1d: ?*mod.impl.llama_c.ggml_tensor = null,
    conv1d_b: ?*mod.impl.llama_c.ggml_tensor = null,

    // gemma3n altup
    altup_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    altup_unembd_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    per_layer_tok_embd: ?*mod.impl.llama_c.ggml_tensor = null,

    // limite value embeddings
    value_embd: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_head_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_head_down: ?*mod.impl.llama_c.ggml_tensor = null,
    hc_head_up: ?*mod.impl.llama_c.ggml_tensor = null,
    per_layer_model_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    per_layer_proj_norm: ?*mod.impl.llama_c.ggml_tensor = null,

    // eagle3 / dflash feature fusion layer
    fc: ?*mod.impl.llama_c.ggml_tensor = null,
    fc_s: ?*mod.impl.llama_c.ggml_tensor = null,
    fc_in_s: ?*mod.impl.llama_c.ggml_tensor = null,
    d2t: ?*mod.impl.llama_c.ggml_tensor = null, // draft to target vocabulary mapping

    // dspark
    dspark_markov_w1: ?*mod.impl.llama_c.ggml_tensor = null,
    dspark_markov_w2: ?*mod.impl.llama_c.ggml_tensor = null,
    dspark_markov_w2_s: ?*mod.impl.llama_c.ggml_tensor = null,
    dspark_conf_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    dspark_conf_proj_b: ?*mod.impl.llama_c.ggml_tensor = null,
    dspark_joint_proj: ?*mod.impl.llama_c.ggml_tensor = null, // rnn markov head
    dspark_joint_proj_b: ?*mod.impl.llama_c.ggml_tensor = null,
    dspark_mask_embd: ?*mod.impl.llama_c.ggml_tensor = null, // per-slot mask embeddings
    dflash_selector_prev: ?*mod.impl.llama_c.ggml_tensor = null,
    dflash_selector_next: ?*mod.impl.llama_c.ggml_tensor = null,
    dflash_selector_hidden: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_token_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_token_proj_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_token_proj_s: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_pass_hidden_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_pass_hidden_proj_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_pass_hidden_proj_s: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_feature_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_feature_norm_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_feature_mlp_1: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_feature_mlp_1_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_feature_mlp_3: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_feature_mlp_3_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_slot_embedding: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_rank_embedding: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_attention_bias: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_context_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_context_proj_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_context_proj_s: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_output_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_anchor_norm: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_factor_input_proj: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_factor_input_proj_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_out_head: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_out_head_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_in_head: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_in_head_b: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_anchor_out_head: ?*mod.impl.llama_c.ggml_tensor = null,
    lilicorr_anchor_out_head_b: ?*mod.impl.llama_c.ggml_tensor = null,

    //Dense linear projections for SentenceTransformers models like embeddinggemma
    // For Sentence Transformers models structure see
    // https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
    dense_2_out_layers: ?*mod.impl.llama_c.ggml_tensor = null,
    dense_2_out_layers_b: ?*mod.impl.llama_c.ggml_tensor = null,
    dense_3_out_layers: ?*mod.impl.llama_c.ggml_tensor = null,
    // END GENERATED by scripts/gen-layer (llama_model)
    layers: []Layer = &.{},

    /// `tensors_by_name`: every weight created, in context order; the name
    /// is the tensor's own.
    tensors_by_name: std.ArrayList(*mod.impl.llama_c.ggml_tensor) = .empty,

    t_load_us: i64 = 0,
    t_start_us: i64 = 0,

    /// The model's memory-mapped files.
    mappings: std.ArrayList(mod.mmap.Mmap) = .empty,
    mlock_bufs: std.ArrayList(*mod.mmap.Mlock) = .empty,
    mlock_mmaps: std.ArrayList(*mod.mmap.Mlock) = .empty,
    /// The weights' tensor contexts and the buffers holding their data.
    ctxs_bufs: std.ArrayList(CtxBufs) = .empty,

    cpu_buft_list: *mod.loader.BuftList,
    gpu_buft_list: std.ArrayList(GpuBufts) = .empty,
    dev_input: LayerDev = .{},
    dev_output: LayerDev = .{},
    dev_layer: std.ArrayList(LayerDev) = .empty,

    /// The loader, while `load_tensors` runs.
    ml: ?*mod.loader.Loader = null,

    /// What `meta_split.getSplitState` is handed, for
    /// `LLAMA_SPLIT_MODE_TENSOR`; it points back at this model.
    get_split_state_ud: mod.meta_split.Userdata = .{},

    /// `loras`: the LoRA adapters loaded against this model, which the
    /// model frees with itself.
    loras: std.ArrayList(*mod.adapter.Lora) = .empty,

    /// `target_layer_ids` (llama-model.h:926 @b1b770d90): the target model's
    /// layers an eagle3 or DFlash draft reads its features from.
    target_layer_ids: std.ArrayList(i32) = .empty,

    /// The members `llama_model_granite_switch` adds to its base --
    /// `n_adapters`, `max_lora_rank`, `router_gain`, `adapter_token_to_slot`
    /// and `adapter_token_to_substitute` (src/models/models.h:1869, 1870, 1871,
    /// 1873, 1874 @b1b770d90) -- which only `models/granite_switch.zig` reads
    /// or writes.
    granite_switch: GraniteSwitch = .{},

    /// The members `llama_model_nanbeige` adds to its base -- `n_loops`,
    /// `n_layer_phys` and `skip_loop_final_norm` (src/models/models.h:446, 447,
    /// 448 @b1b770d90) -- which only `models/nanbeige.zig` reads or writes.
    nanbeige: Nanbeige = .{},
    /// `llama_model_clef`'s members (src/models/models.h:2597 @b1b770d90): its
    /// decision head.
    clef: mod.models.clef.State = .{},
    /// `llama_model_vllm_decision2`'s members (src/models/models.h:588
    /// @b1b770d90): its decision head's tensors, which the model owns.
    vllm_decision2: mod.models.vllm_decision2.State = .{},
    /// `llama_model_metis16`'s members -- `n_pass`, `n_stream`, `n_mem`,
    /// `n_route`, `n_layer_phys` and `cont_limit` (src/models/models.h:1715,
    /// 1716, 1717, 1718, 1719, 1720 @b1b770d90) -- which only
    /// `models/metis16.zig` reads or writes.
    metis16: mod.models.metis16.State = .{},

    /// Mirrors those members, with their defaults.
    pub const Nanbeige = struct {
        n_loops: c_int = 1,
        n_layer_phys: c_int = 0,
        skip_loop_final_norm: bool = false,
    };

    /// Mirrors `llama_model_granite_switch`'s members, with their defaults.
    /// The maps are libc++'s layout, since `load_arch_tensors` reports the
    /// first bad substitute in their order; they are created when the
    /// hyperparameters load.
    pub const GraniteSwitch = struct {
        n_adapters: u32 = 0,
        max_lora_rank: u32 = 0,
        router_gain: f32 = 15.0,
        adapter_token_to_slot: ?stdlib.libcxx.I32Map(i32) = null,
        adapter_token_to_substitute: ?stdlib.libcxx.I32Map(i32) = null,

        /// Return: nothing.
        pub fn deinit(self: *GraniteSwitch) void {
            if (self.adapter_token_to_slot) |*m| m.deinit();
            if (self.adapter_token_to_substitute) |*m| m.deinit();
            self.* = .{};
        }
    };

    /// Ports `llama_model_create` (llama-model.cpp:406 @b1b770d90), through
    /// its architecture overload (llama-model.cpp:393 @b1b770d90).
    ///
    /// Every architecture the C++ has a class for is accepted: the one
    /// architecture-specific step is `loadArchHparams`, which is where an
    /// unported one stops.
    ///
    /// Return: the model, or `error.Thrown` where the C++ throws.
    pub fn create(ml: *const mod.loader.Loader, params: mod.impl.llama_c.llama_model_params, exc: *mod.impl.Exception) mod.impl.Thrown!Model {
        const a = ml.getArch();
        if (a == .unknown) return exc.throw(.runtime_error, "unknown model architecture: '{s}'", .{ml.getArchName()});
        return createArch(a, params, exc);
    }

    /// Ports `llama_model_create`'s architecture overload (llama-model.cpp:393
    /// @b1b770d90), which `llama_quant_model_from_metadata` calls directly.
    ///
    /// Parameters:
    /// - `a`: the architecture; not `.unknown`.
    /// - `params`: the model parameters, copied.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the model, or `error.Thrown` where the C++ throws.
    pub fn createArch(a: mod.arch_table.Arch, params: mod.impl.llama_c.llama_model_params, exc: *mod.impl.Exception) mod.impl.Thrown!Model {
        if (!mod.model_table.mapped[@intCast(@backingInt(a))]) {
            return exc.throw(.runtime_error, "unsupported model architecture: '{s}'", .{mod.arch.archName(a)});
        }
        if (params.split_mode == mod.impl.llama_c.LLAMA_SPLIT_MODE_TENSOR and !mod.arch.supportsSmTensor(a)) {
            return exc.throw(.runtime_error, "LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '{s}'", .{mod.arch.archName(a)});
        }
        const cpu_list = std.heap.c_allocator.create(mod.loader.BuftList) catch oom();
        cpu_list.* = .empty;
        var m: Model = .{
            .arch = a,
            .vocab = mod.vocab.Vocab.init() catch oom(),
            .gguf_kv = stdlib.libcxx.StringMap([:0]u8).init(std.heap.c_allocator) catch oom(),
            .params = params,
            .cpu_buft_list = cpu_list,
        };
        // Ports the constructor of `llama_model` (llama-model.cpp:1441
        // @b1b770d90): `tensor_split` is borrowed by the parameters but
        // read again later, so it is copied. `params.tensor_split` points
        // into this value, so a model that moves -- as this return does --
        // must call `ownTensorSplit` at its final address.
        if (params.tensor_split != null) {
            @memcpy(&m.tensor_split_owned, params.tensor_split[0..16]);
            m.params.tensor_split = &m.tensor_split_owned;
        }
        m.has_tensor_overrides = params.tensor_buft_overrides != null and params.tensor_buft_overrides[0].pattern != null;
        m.name.appendSlice(std.heap.c_allocator, "n/a") catch oom();
        return m;
    }

    /// Return: nothing.
    /// `can_prefetch.count(t)`: whether a graph input may prefetch `t`'s rows.
    pub fn canPrefetch(self: *const Model, t: ?*const mod.impl.llama_c.ggml_tensor) bool {
        const p = t orelse return false;
        return self.can_prefetch.contains(p);
    }

    pub fn deinit(self: *Model) void {
        // `~llama_model` deletes the LoRA adapters still registered.
        mod.adapter.freeModelLoras(self);
        for (self.ctxs_bufs.items) |*cb| {
            mod.impl.llama_c.ggml_free(cb.ctx);
            for (cb.bufs.items) |buf| mod.impl.llama_c.ggml_backend_buffer_free(buf);
            cb.bufs.deinit(std.heap.c_allocator);
        }
        self.ctxs_bufs.deinit(std.heap.c_allocator);
        for (self.mappings.items) |*m| m.deinit();
        self.mappings.deinit(std.heap.c_allocator);
        for (self.mlock_bufs.items) |l| std.heap.c_allocator.destroy(l);
        self.mlock_bufs.deinit(std.heap.c_allocator);
        for (self.mlock_mmaps.items) |l| std.heap.c_allocator.destroy(l);
        self.mlock_mmaps.deinit(std.heap.c_allocator);
        self.tensors_by_name.deinit(std.heap.c_allocator);
        self.prec_policy.deinit();
        self.hdmd.deinit();
        self.can_prefetch.deinit(std.heap.c_allocator);
        std.heap.c_allocator.free(self.layers);
        self.devices.deinit(std.heap.c_allocator);
        for (self.gpu_buft_list.items) |g| {
            g.list.deinit(std.heap.c_allocator);
            std.heap.c_allocator.destroy(g.list);
        }
        self.gpu_buft_list.deinit(std.heap.c_allocator);
        self.cpu_buft_list.deinit(std.heap.c_allocator);
        std.heap.c_allocator.destroy(self.cpu_buft_list);
        self.dev_layer.deinit(std.heap.c_allocator);
        self.name.deinit(std.heap.c_allocator);
        self.vocab.deinit();
        for (self.classifier_labels.items) |l| std.heap.c_allocator.free(l);
        self.classifier_labels.deinit(std.heap.c_allocator);
        for (self.decision_labels.items) |l| std.heap.c_allocator.free(l);
        self.decision_labels.deinit(std.heap.c_allocator);
        self.engram_token_map.deinit(std.heap.c_allocator);
        var it = self.gguf_kv.iterator();
        while (it.next()) |e| std.heap.c_allocator.free(e.value.*);
        self.gguf_kv.deinit();
        self.desc_str.deinit(std.heap.c_allocator);
        self.loras.deinit(std.heap.c_allocator);
        self.target_layer_ids.deinit(std.heap.c_allocator);
        self.granite_switch.deinit();
        self.clef.deinit();
    }

    /// Ports `llama_model_base`'s `load_stats` (llama-model.cpp:1457 @b1b770d90).
    pub fn loadStats(self: *Model, ml: *const mod.loader.Loader) void {
        self.n_elements = ml.n_elements;
        self.n_bytes = ml.n_bytes;
    }

    /// Ports `llama_model_base`'s `load_vocab` (llama-model.cpp:1683 @b1b770d90).
    pub fn loadVocab(self: *Model, ml: *const mod.loader.Loader, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        try self.vocab.load(&ml.meta, exc);
    }

    /// Ports `llama_model_base`'s `load_hparams` (llama-model.cpp:1462 @b1b770d90).
    pub fn loadHparams(self: *Model, ml: *const mod.loader.Loader, exc: *mod.impl.Exception) (mod.impl.Thrown || error{Unported})!void {
        const ctx = ml.meta.ctx;
        const meta = &ml.meta;
        const hp = &self.hparams;

        // Every scalar value as text, the first of a repeated key kept.
        var i: i64 = 0;
        while (i < mod.impl.llama_c.gguf_get_n_kv(ctx)) : (i += 1) {
            if (mod.impl.llama_c.gguf_get_kv_type(ctx, i) == mod.impl.llama_c.GGUF_TYPE_ARRAY) continue;
            var value: std.ArrayList(u8) = .empty;
            mod.loader.kvToStr(ctx, i, std.math.maxInt(usize), &value);
            const slot = self.gguf_kv.getOrPut(mod.gguf_meta.cString(mod.impl.llama_c.gguf_get_key(ctx, i)), undefined) catch oom();
            if (slot.found) {
                value.deinit(std.heap.c_allocator);
            } else {
                slot.value.* = value.toOwnedSliceSentinel(std.heap.c_allocator, 0) catch oom();
            }
        }

        try self.loadHparamsHadamard(ml, exc);

        if (try meta.getKey([]const u8, .general_name, false, exc)) |n| {
            self.name.clearRetainingCapacity();
            self.name.appendSlice(std.heap.c_allocator, n) catch oom();
        }

        // Everything past this point is not vocabulary-related; a CLIP
        // model loads tensors only.
        if (hp.vocab_only or ml.getArch() == .clip) return;

        hp.n_ctx_train = (try meta.getKey(u32, .context_length, true, exc)).?;
        hp.n_embd = (try meta.getKey(u32, .embedding_length, true, exc)).?;
        if (try meta.getKey(u32, .embedding_length_out, false, exc)) |v| hp.n_embd_out_impl = v;
        if (try meta.getKey(bool, .attention_causal, false, exc)) |v| hp.causal_attn = v;
        hp.pooling_type = try meta.getPoolingType(.pooling_type, false, exc);
        hp.pooling_type_cls = try meta.getPoolingType(.classifier_pooling_type, false, exc);
        if (try meta.getKey(bool, .classifier_free_guidance, false, exc)) |v| hp.classifier_free_guidance = v;
        if (try meta.getKey([]const u8, .classifier_activation, false, exc)) |act_cls| {
            var found: ?c_int = null;
            for (cls_act_types_from_string) |e| {
                if (std.mem.eql(u8, e[0], act_cls)) found = e[1];
            }
            mod.impl.assert(found != null, "llama-model.cpp", 1497, "it != LLM_CLS_ACT_TYPES_FROM_STRING.end() && \"unsupported classifier activation\"");
            hp.act_cls = found.?;
        }
        hp.n_layer_all = (try meta.getKey(u32, .block_count, true, exc)).?;
        mod.impl.assert(hp.n_layer_all > 0 and hp.n_layer_all <= mod.hparams.max_layers, "llama-model.cpp", 1501, "hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS");
        if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
        mod.impl.assert(hp.n_layer_nextn <= hp.n_layer_all, "llama-model.cpp", 1503, "hparams.n_layer_nextn <= hparams.n_layer_all");
        if (try meta.getKey(u32, .expert_count, false, exc)) |v| hp.n_expert = v;
        // the routing count may be given per layer
        hp.n_expert_used_arr = @splat(0);
        _ = try meta.getKeyOrArr(u32, .expert_used_count, &hp.n_expert_used_arr, hp.n_layer_all, false, exc);
        if (try meta.getKey(u32, .expert_group_count, false, exc)) |v| hp.n_expert_groups = v;
        if (try meta.getKey(u32, .expert_group_used_count, false, exc)) |v| hp.n_group_used = v;

        if (self.arch == .hunyuan_vl or self.arch == .hunyuan_dense) {
            if (hp.n_expert <= 1) {
                hp.n_expert = 0;
                hp.n_expert_used_arr = @splat(0);
            }
        }

        if (self.arch == .wavtokenizer_dec) {
            hp.n_embd = (try meta.getKey(u32, .features_length, true, exc)).?;
            hp.n_embd_out_impl = (try meta.getKey(u32, .embedding_length, true, exc)).?;
            hp.posnet.n_embd = (try meta.getKey(u32, .posnet_embedding_length, true, exc)).?;
            hp.posnet.n_layer = (try meta.getKey(u32, .posnet_block_count, true, exc)).?;
            hp.convnext.n_embd = (try meta.getKey(u32, .convnext_embedding_length, true, exc)).?;
            hp.convnext.n_layer = (try meta.getKey(u32, .convnext_block_count, true, exc)).?;
            mod.impl.assert(hp.posnet.n_layer <= hp.n_layer_all, "llama-model.cpp", 1527, "hparams.posnet.n_layer <= hparams.n_layer_all");
            mod.impl.assert(hp.convnext.n_layer <= hp.n_layer_all, "llama-model.cpp", 1528, "hparams.convnext.n_layer <= hparams.n_layer_all");
        }

        mod.impl.assert(hp.n_expert <= mod.hparams.max_experts, "llama-model.cpp", 1534, "hparams.n_expert <= LLAMA_MAX_EXPERTS");
        // models may route a different number of experts per layer, so validate the maximum
        const n_expert_used_max = hp.nExpertUsedMax();

        mod.impl.assert(n_expert_used_max <= hp.n_expert, "llama-model.cpp", 1535, "n_expert_used_max <= hparams.n_expert");
        if (hp.n_expert > 0) {
            mod.impl.assert(n_expert_used_max > 0, "llama-model.cpp", 1537, "n_expert_used_max > 0");
            mod.impl.assert(hp.n_expert_groups < hp.n_expert, "llama-model.cpp", 1538, "hparams.n_expert_groups < hparams.n_expert");
            if (hp.n_expert_groups > 1) {
                mod.impl.assert(hp.n_expert % hp.n_expert_groups == 0, "llama-model.cpp", 1540, "hparams.n_expert % hparams.n_expert_groups == 0");
                mod.impl.assert(hp.n_group_used > 0, "llama-model.cpp", 1541, "hparams.n_group_used > 0");
                mod.impl.assert(hp.n_group_used < hp.n_expert_groups, "llama-model.cpp", 1542, "hparams.n_group_used < hparams.n_expert_groups");
            }
        } else {
            mod.impl.assert(n_expert_used_max == 0, "llama-model.cpp", 1545, "n_expert_used_max == 0");
            mod.impl.assert(hp.n_expert_groups == 0, "llama-model.cpp", 1546, "hparams.n_expert_groups == 0");
        }

        hp.n_head_arr = @splat(0);
        hp.n_head_kv_arr = @splat(0);
        hp.n_ff_arr = @splat(0);
        hp.n_ff_exp_arr = @splat(0);
        hp.rope_sections = @splat(0);
        hp.rope_pattern = @splat(1);
        hp.is_swa_impl = @splat(0);
        hp.is_recr_impl = @splat(@intFromBool(mod.arch.isRecurrent(ml.getArch())));
        hp.is_indexer_full_impl = @splat(0);
        for (&hp.kv_source_impl, 0..) |*v, j| v.* = @intCast(j); // std::iota
        hp.xielu_alpha_n = @splat(0);
        hp.xielu_alpha_p = @splat(0);
        hp.xielu_beta = @splat(0);
        hp.xielu_eps = @splat(0);
        hp.swiglu_clamp_exp = @splat(0);
        hp.swiglu_clamp_shexp = @splat(0);
        hp.swiglu_clamp = @splat(0);

        _ = try meta.getKeyOrArr(u32, .feed_forward_length, &hp.n_ff_arr, hp.n_layer_all, false, exc);
        _ = try meta.getKeyOrArr(u32, .attention_head_count, &hp.n_head_arr, hp.n_layer_all, false, exc);

        hp.deepstack_mapping_arr = @splat(-1);

        // n_head_kv is optional, and defaults to n_head.
        hp.n_head_kv_arr = hp.n_head_arr;
        _ = try meta.getKeyOrArr(u32, .attention_head_count_kv, &hp.n_head_kv_arr, hp.n_layer_all, false, exc);

        hp.rope_finetuned = (try meta.getKey(bool, .rope_scaling_finetuned, false, exc)) orelse false;

        hp.n_ctx_orig_yarn = hp.n_ctx_train;
        if (try meta.getKey(u32, .rope_scaling_orig_ctx_len, false, exc)) |v| hp.n_ctx_orig_yarn = v;

        hp.rope_freq_base_train = 10000.0;
        if (try meta.getKey(f32, .rope_freq_base, false, exc)) |v| hp.rope_freq_base_train = v;

        const rope_scaling = (try meta.getKey([]const u8, .rope_scaling_type, false, exc)) orelse "linear";
        hp.rope_scaling_type_train = ropeScalingTypeFromString(rope_scaling);
        mod.impl.assert(hp.rope_scaling_type_train != mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED, "llama-model.cpp", 1595, "hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED");

        const rope_section_order = (try meta.getKey([]const u8, .rope_section_order, false, exc)) orelse "tyxz";
        hp.rope_section_order = ropeSectionOrderFromString(rope_section_order);
        if (hp.rope_section_order == mod.hparams.RopeSectionOrder.unspecified) {
            return exc.throw(.runtime_error, "unknown rope section order: {s}", .{rope_section_order});
        }

        // rope_freq_scale is the inverse of the stored factor; either key.
        var ropescale: f32 = 0.0;
        if (try meta.getKey(f32, .rope_scaling_factor, false, exc)) |v| {
            ropescale = v;
        } else if (try meta.getKey(f32, .rope_scale_linear, false, exc)) |v| {
            ropescale = v;
        }
        hp.rope_freq_scale_train = if (ropescale == 0.0) 1.0 else 1.0 / ropescale;

        if (try meta.getKey(f32, .rope_scaling_attn_factor, false, exc)) |v| hp.rope_attn_factor = v;
        if (try meta.getKey(f32, .rope_scaling_alpha, false, exc)) |v| hp.rope_scaling_alpha = v;

        // Non-transformer models have no attention heads.
        if (hp.nHead(0) > 0) {
            hp.n_embd_head_k_full = hp.n_embd / hp.nHead(0);
            if (try meta.getKey(u32, .attention_key_length, false, exc)) |v| hp.n_embd_head_k_full = v;
            hp.n_embd_head_v_full = hp.n_embd / hp.nHead(0);
            if (try meta.getKey(u32, .attention_value_length, false, exc)) |v| hp.n_embd_head_v_full = v;

            hp.n_rot_full = hp.n_embd_head_k_full;
            if (try meta.getKey(u32, .rope_dimension_count, false, exc)) |v| hp.n_rot_full = v;

            if (self.arch == .llama or self.arch == .deci or self.arch == .falcon or self.arch == .llama_embed) {
                if (hp.n_rot_full != hp.n_embd_head_k_full) {
                    return exc.throw(.runtime_error, "invalid n_rot: {d}, expected {d}", .{ hp.n_rot_full, hp.n_embd_head_k_full });
                }
            }
        } else {
            hp.n_rot_full = 0;
            hp.n_embd_head_k_full = 0;
            hp.n_embd_head_v_full = 0;
        }

        // Head size and n_rot for SWA layers.
        hp.n_embd_head_k_swa = hp.n_embd_head_k_full;
        hp.n_embd_head_v_swa = hp.n_embd_head_v_full;
        if (try meta.getKey(u32, .attention_key_length_swa, false, exc)) |v| hp.n_embd_head_k_swa = v;
        if (try meta.getKey(u32, .attention_value_length_swa, false, exc)) |v| hp.n_embd_head_v_swa = v;
        hp.n_rot_swa = hp.n_rot_full;
        if (try meta.getKey(u32, .rope_dimension_count_swa, false, exc)) |v| hp.n_rot_swa = v;

        // Classifier models.
        if (try meta.getArrList([]const u8, .classifier_output_labels, false, exc)) |labels| {
            defer std.heap.c_allocator.free(labels);
            for (self.classifier_labels.items) |l| std.heap.c_allocator.free(l);
            self.classifier_labels.clearRetainingCapacity();
            for (labels) |l| self.classifier_labels.append(std.heap.c_allocator, std.heap.c_allocator.dupeSentinel(u8, l, 0) catch oom()) catch oom();
        }
        if (self.classifier_labels.items.len != 0) hp.n_cls_out = @truncate(self.classifier_labels.items.len);

        // for decision models
        if (try meta.getArrList([]const u8, .decision_labels, false, exc)) |labels| {
            defer std.heap.c_allocator.free(labels);
            for (self.decision_labels.items) |l| std.heap.c_allocator.free(l);
            self.decision_labels.clearRetainingCapacity();
            for (labels) |l| self.decision_labels.append(std.heap.c_allocator, std.heap.c_allocator.dupeSentinel(u8, l, 0) catch oom()) catch oom();
        }

        try self.loadArchHparams(ml, exc);

        self.n_bytes = ml.n_bytes;

        self.desc_str.clearRetainingCapacity();
        self.desc_str.print(std.heap.c_allocator, "{s} {s} {s}", .{ self.archName(), self.typeName(), ml.ftypeNameOf() }) catch oom();

        self.ftype = ml.ftype;

        if (hp.f_max_alibi_bias > 0.0) hp.use_alibi = true;

        hp.rope_type = self.ropeType();

        if (hp.rope_section_order != mod.hparams.RopeSectionOrder.tyxz and hp.nPosPerEmbd() != 4) {
            return exc.throw(.runtime_error, "rope section order {s} requires M-RoPE", .{try ropeSectionOrderName(hp.rope_section_order, exc)});
        }
    }

    /// Ports `llama_model_base`'s `load_hparams_hadamard` (llama-model.cpp:2175
    /// @b1b770d90): reads and checks the `prism.hadamard` metadata and records
    /// the folded weights; `loadTensorsHadamard` makes the tensors.
    ///
    /// Return: nothing, or `error.Thrown` for metadata the C++ refuses.
    fn loadHparamsHadamard(self: *Model, ml: *const mod.loader.Loader, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        const meta = &ml.meta;
        const hdmd = &self.hdmd;

        var hadamard_version: u32 = 0;
        if (try meta.getKey(bool, .prism_hadamard_tied_output, false, exc)) |v| hdmd.tied_output = v;
        if (try meta.getKey(u32, .prism_hadamard_version, false, exc)) |version| {
            hadamard_version = version;
            if (hadamard_version != 1 and hadamard_version != 2) {
                return exc.throw(.runtime_error, "unsupported prism.hadamard.version: {d}", .{hadamard_version});
            }

            if ((hadamard_version == 2) != hdmd.tied_output) {
                return exc.throw(.runtime_error, "prism.hadamard version 2 requires tied_output=true; version 1 forbids it", .{});
            }
            if (hdmd.tied_output and ml.getWeight("output.weight") != null) {
                return exc.throw(.runtime_error, "prism.hadamard.tied_output requires output.weight to be absent", .{});
            }

            const block_size = (try meta.getKey(u32, .prism_hadamard_block_size, true, exc)).?;
            const transform = (try meta.getKey([]const u8, .prism_hadamard_transform, true, exc)).?;
            const axis = (try meta.getKey([]const u8, .prism_hadamard_axis, true, exc)).?;
            const sign_mode = (try meta.getKey([]const u8, .prism_hadamard_sign_mode, true, exc)).?;
            const weight_names = (try meta.getArrList([]const u8, .prism_hadamard_weight_names, true, exc)).?;
            defer std.heap.c_allocator.free(weight_names);

            if (block_size == 0 or (block_size & (block_size -% 1)) != 0) {
                return exc.throw(.runtime_error, "invalid prism.hadamard.block_size: {d}", .{block_size});
            }
            if (!std.mem.eql(u8, transform, "normalized-sylvester-walsh-hadamard")) {
                return exc.throw(.runtime_error, "unsupported prism.hadamard.transform: {s}", .{transform});
            }
            if (!std.mem.eql(u8, axis, "input-last-dimension")) {
                return exc.throw(.runtime_error, "unsupported prism.hadamard.axis: {s}", .{axis});
            }
            if (!std.mem.eql(u8, sign_mode, "identity") and !std.mem.eql(u8, sign_mode, "explicit")) {
                return exc.throw(.runtime_error, "unsupported prism.hadamard.sign_mode: {s}", .{sign_mode});
            }
            if (weight_names.len == 0) {
                return exc.throw(.runtime_error, "prism.hadamard.weight_names is empty", .{});
            }

            if (std.mem.eql(u8, sign_mode, "explicit")) {
                const sign_widths = (try meta.getArrList(i32, .prism_hadamard_sign_widths, true, exc)).?;
                defer std.heap.c_allocator.free(sign_widths);
                const sign_values = (try meta.getArrList(i32, .prism_hadamard_sign_values, true, exc)).?;
                defer std.heap.c_allocator.free(sign_values);
                // explicit mode with no widths gives an empty sign table,
                // which acts as identity and changes the model
                if (sign_widths.len == 0) {
                    return exc.throw(.runtime_error, "prism.hadamard.sign_mode is explicit but sign_widths is empty", .{});
                }
                var off: usize = 0;
                for (sign_widths) |width| {
                    if (width <= 0 or @as(u32, @bitCast(width)) % block_size != 0 or off + @as(usize, @intCast(width)) > sign_values.len) {
                        return exc.throw(.runtime_error, "invalid prism.hadamard sign width: {d}", .{width});
                    }
                    const w: u32 = @intCast(width);
                    if (hdmd.findSigns(w) != null) {
                        return exc.throw(.runtime_error, "duplicate prism.hadamard sign width: {d}", .{width});
                    }
                    const vec = std.heap.c_allocator.dupe(i32, sign_values[off..][0..w]) catch oom();
                    var at: usize = 0;
                    while (at < hdmd.sign_data.items.len and hdmd.sign_data.items[at].width < w) at += 1;
                    hdmd.sign_data.insert(std.heap.c_allocator, at, .{ .width = w, .values = vec }) catch oom();
                    for (vec) |v| {
                        if (v != 1 and v != -1) {
                            return exc.throw(.runtime_error, "prism.hadamard sign values must be +/-1", .{});
                        }
                    }
                    off += w;
                }
                if (off != sign_values.len) {
                    return exc.throw(.runtime_error, "prism.hadamard.sign_values length mismatch", .{});
                }
            }

            if (try meta.getKey(bool, .prism_hadamard_gdn_v_grouped, false, exc)) |v| hdmd.gdn_v_grouped = v;

            // only build_lora_mm and build_lora_mm_id apply the transform:
            // refuse the archs and tensor kinds that can skip them
            switch (self.arch) {
                .llama, .qwen3, .qwen3moe, .qwen35, .qwen35moe, .qwen3next => {},
                else => return exc.throw(.runtime_error, "prism.hadamard: arch '{s}' is not verified to apply the activation transform to all folded weights", .{self.archName()}),
            }

            // a folded weight W_f = W*D*H is the weight W with the +1/-1
            // signs D and the normalized block Hadamard H folded in; H*H = I
            // and D*D = I, so W*x = W_f*(H*(D*x)): the graph applies D, then
            // H, to the matmul input
            const weight_blocks = Hadamard.blocks(&hdmd.weight_blocks);
            for (weight_names) |weight_name| {
                if (!isFoldableWeight(weight_name)) {
                    return exc.throw(.runtime_error, "prism.hadamard: weight '{s}' is not on a verified Hadamard-aware matmul path", .{weight_name});
                }
                const slot = weight_blocks.getOrPut(weight_name, block_size) catch oom();
                if (slot.found) {
                    return exc.throw(.runtime_error, "duplicate prism.hadamard weight: {s}", .{weight_name});
                }
            }

            // the tables read by row lookup store latent rows: the inverse
            // transform goes on the lookup result
            const inverse_names = (try meta.getArrList([]const u8, .prism_hadamard_inverse_weight_names, false, exc)) orelse &.{};
            defer std.heap.c_allocator.free(inverse_names);
            const inverse_blocks = Hadamard.blocks(&hdmd.inverse_blocks);
            for (inverse_names) |name| {
                // the graph applies the inverse only after the token-embedding
                // lookup, other latent tables stay rotated
                if (!std.mem.eql(u8, name, "token_embd.weight")) {
                    return exc.throw(.runtime_error, "prism.hadamard: weight '{s}' is not a verified inverse-after-lookup table", .{name});
                }
                if (weight_blocks.get(name) != null or (inverse_blocks.getOrPut(name, block_size) catch oom()).found) {
                    return exc.throw(.runtime_error, "duplicate prism.hadamard inverse weight: {s}", .{name});
                }
            }
        }

        const inverse_embd: ?u32 = if (hdmd.inverse_blocks) |*m| (if (m.get("token_embd.weight")) |v| v.* else null) else null;
        if (hdmd.tied_output) {
            if (hadamard_version != 2) {
                return exc.throw(.runtime_error, "prism.hadamard.tied_output requires version 2", .{});
            }
            const block = inverse_embd orelse return exc.throw(.runtime_error, "prism.hadamard.tied_output requires a latent token embedding", .{});
            _ = Hadamard.blocks(&hdmd.weight_blocks).getOrPut("token_embd.weight", block) catch oom();
        } else if (inverse_embd != null and ml.getWeight("output.weight") == null) {
            return exc.throw(.runtime_error, "a tied Hadamard output requires version 2 and tied_output=true", .{});
        }
    }

    /// The `is_foldable_weight` lambda of `load_hparams_hadamard`
    /// (llama-model.cpp:2175 @b1b770d90): the output head, or a block's
    /// weight of a kind on a verified Hadamard-aware matmul path.
    fn isFoldableWeight(name: []const u8) bool {
        const kinds = [_][]const u8{
            "attn_q",           "attn_k",         "attn_v",       "attn_qkv",       "attn_gate",   "attn_output",
            "ffn_gate",         "ffn_up",         "ffn_down",     "ffn_gate_exps",  "ffn_up_exps", "ffn_down_exps",
            "ffn_gate_up_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ssm_out",
        };
        if (std.mem.eql(u8, name, "output.weight")) {
            return true; // the output head is built through build_lora_mm in every arch
        }
        if (!std.mem.startsWith(u8, name, "blk.")) {
            return false;
        }
        var pos: usize = 4;
        while (pos < name.len and std.ascii.isDigit(name[pos])) {
            pos += 1;
        }
        if (pos == 4 or pos >= name.len or name[pos] != '.') {
            return false;
        }
        pos += 1;
        for (kinds) |kind| {
            const rest = name[pos..];
            if (rest.len == kind.len + ".weight".len and std.mem.startsWith(u8, rest, kind) and std.mem.endsWith(u8, rest, ".weight")) {
                return true;
            }
        }
        return false;
    }

    /// Ports `llama_model_base`'s `load_tensors_hadamard` (llama-model.cpp:2340
    /// @b1b770d90): one Hadamard matrix per block size and one sign vector
    /// per width, which the file does not hold. Each goes on the buffer
    /// type of its weights, never a CPU extra type, and each transform into
    /// `hdmd.rot` or `hdmd.inv`.
    ///
    /// Return: nothing, or `error.Thrown`.
    fn loadTensorsHadamard(self: *Model, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        const hdmd = &self.hdmd;
        const n_weight = if (hdmd.weight_blocks) |*m| m.count() else 0;
        const n_inverse = if (hdmd.inverse_blocks) |*m| m.count() else 0;
        if (n_weight == 0 and n_inverse == 0) {
            return;
        }

        const Rotation = struct {
            block_size: u32,
            buft: mod.impl.llama_c.ggml_backend_buffer_type_t,
            tensor: *mod.impl.llama_c.ggml_tensor,
        };
        const SignTensor = struct {
            width: u32,
            buft: mod.impl.llama_c.ggml_backend_buffer_type_t,
            tensor: *mod.impl.llama_c.ggml_tensor,
        };

        var rotations: std.ArrayList(Rotation) = .empty;
        defer rotations.deinit(std.heap.c_allocator);
        var sign_tensors: std.ArrayList(SignTensor) = .empty;
        defer sign_tensors.deinit(std.heap.c_allocator);

        // the inverse transforms use the buffer type of the forward
        // rotations, not the host type of a CPU-mapped table: no PCIe round
        // trip per token
        var preferred_buft: mod.impl.llama_c.ggml_backend_buffer_type_t = null;

        for ([_]bool{ true, false }) |forward| {
            const blocks_opt = if (forward) &hdmd.weight_blocks else &hdmd.inverse_blocks;
            const blocks = if (blocks_opt.*) |*m| m else continue;
            const target = if (forward) &hdmd.rot else &hdmd.inv;
            var it = blocks.iterator();
            while (it.next()) |entry| {
                const weight_name = entry.key;
                const block_size = entry.value.*;
                var weight: ?*mod.impl.llama_c.ggml_tensor = self.getTensor(weight_name);
                if (hdmd.tied_output and std.mem.eql(u8, weight_name, "token_embd.weight")) {
                    weight = if (forward) self.output else self.tok_embd;
                    if (weight == null or !std.mem.eql(u8, mod.gguf_meta.cString(mod.impl.llama_c.ggml_get_name(weight)), "token_embd.weight")) {
                        return exc.throw(.runtime_error, "prism.hadamard.tied_output is not bound to the token embedding", .{});
                    }
                }
                const w = weight orelse return exc.throw(.runtime_error, "prism.hadamard weight not found: {s}", .{weight_name});
                if (@rem(w.ne[0], @as(i64, block_size)) != 0) {
                    return exc.throw(.runtime_error, "prism.hadamard block size {d} does not divide input dimension {d} for {s}", .{ block_size, w.ne[0], weight_name });
                }
                if (w.buffer == null) {
                    return exc.throw(.runtime_error, "prism.hadamard weight has no buffer: {s}", .{weight_name});
                }

                var buft = mod.impl.llama_c.ggml_backend_buffer_get_type(w.buffer);
                // CPU extra buffer types (CPU_REPACK) only accept the
                // tensors they can repack
                if (mod.impl.llama_c.ggml_backend_buft_get_device(buft)) |dev| {
                    if (mod.impl.llama_c.ggml_backend_dev_type(dev) == mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU) {
                        buft = mod.impl.llama_c.ggml_backend_dev_buffer_type(dev);
                    }
                }
                if (forward) {
                    preferred_buft = buft;
                } else if (preferred_buft != null) {
                    buft = preferred_buft;
                }

                var rot_at: ?usize = null;
                for (rotations.items, 0..) |r, i| {
                    if (r.block_size == block_size and r.buft == buft) {
                        rot_at = i;
                        break;
                    }
                }

                if (rot_at == null) {
                    const ctx = mod.impl.llama_c.ggml_init(.{ .mem_size = mod.impl.llama_c.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = true }) orelse
                        return exc.throw(.runtime_error, "failed to create Hadamard rotation context", .{});

                    const rotation: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_new_tensor_2d(ctx, mod.impl.llama_c.GGML_TYPE_F32, block_size, block_size);
                    _ = mod.impl.llama_c.ggml_format_name(rotation, "prism.hadamard.%u", block_size);

                    const buffer = mod.impl.llama_c.ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
                    if (buffer == null) {
                        mod.impl.llama_c.ggml_free(ctx);
                        return exc.throw(.runtime_error, "unable to allocate {s} Hadamard rotation buffer", .{mod.gguf_meta.cString(mod.impl.llama_c.ggml_backend_buft_name(buft))});
                    }
                    mod.impl.llama_c.ggml_backend_buffer_set_usage(buffer, mod.impl.llama_c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

                    const n: usize = block_size;
                    const data = std.heap.c_allocator.alloc(f32, n * n) catch oom();
                    defer std.heap.c_allocator.free(data);
                    const scale = 1.0 / stdlib.libm.sqrtfInt(block_size);
                    for (0..n) |row| {
                        for (0..n) |col| {
                            var parity: u32 = @intCast(row & col);
                            parity ^= parity >> 16;
                            parity ^= parity >> 8;
                            parity ^= parity >> 4;
                            parity ^= parity >> 2;
                            parity ^= parity >> 1;
                            data[row * n + col] = if ((parity & 1) != 0) -scale else scale;
                        }
                    }
                    mod.impl.llama_c.ggml_backend_tensor_set(rotation, data.ptr, 0, data.len * @sizeOf(f32));

                    var cb: CtxBufs = .{ .ctx = ctx };
                    cb.bufs.append(std.heap.c_allocator, buffer) catch oom();
                    self.ctxs_bufs.append(std.heap.c_allocator, cb) catch oom();
                    rotations.append(std.heap.c_allocator, .{ .block_size = block_size, .buft = buft, .tensor = rotation }) catch oom();
                    rot_at = rotations.items.len - 1;
                }

                var sign_tensor: ?*mod.impl.llama_c.ggml_tensor = null;
                if (hdmd.sign_data.items.len != 0) {
                    const width: u32 = @intCast(w.ne[0]);
                    const sd = hdmd.findSigns(width) orelse
                        return exc.throw(.runtime_error, "prism.hadamard has no sign vector for width {d} ({s})", .{ width, weight_name });
                    for (sign_tensors.items) |st| {
                        if (st.width == width and st.buft == buft) sign_tensor = st.tensor;
                    }
                    if (sign_tensor == null) {
                        const ctx = mod.impl.llama_c.ggml_init(.{ .mem_size = mod.impl.llama_c.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = true }) orelse
                            return exc.throw(.runtime_error, "failed to create Hadamard sign context", .{});

                        const signs: *mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_new_tensor_1d(ctx, mod.impl.llama_c.GGML_TYPE_F32, width);
                        _ = mod.impl.llama_c.ggml_format_name(signs, "prism.hadamard.signs.%u", width);

                        const buffer = mod.impl.llama_c.ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
                        if (buffer == null) {
                            mod.impl.llama_c.ggml_free(ctx);
                            return exc.throw(.runtime_error, "unable to allocate {s} Hadamard sign buffer", .{mod.gguf_meta.cString(mod.impl.llama_c.ggml_backend_buft_name(buft))});
                        }
                        mod.impl.llama_c.ggml_backend_buffer_set_usage(buffer, mod.impl.llama_c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

                        const data = std.heap.c_allocator.alloc(f32, width) catch oom();
                        defer std.heap.c_allocator.free(data);
                        for (data, sd.values) |*d, v| d.* = @floatFromInt(v);
                        mod.impl.llama_c.ggml_backend_tensor_set(signs, data.ptr, 0, data.len * @sizeOf(f32));

                        var cb: CtxBufs = .{ .ctx = ctx };
                        cb.bufs.append(std.heap.c_allocator, buffer) catch oom();
                        self.ctxs_bufs.append(std.heap.c_allocator, cb) catch oom();
                        sign_tensors.append(std.heap.c_allocator, .{ .width = width, .buft = buft, .tensor = signs }) catch oom();
                        sign_tensor = signs;
                    }
                }

                var transform: HadamardTransform = .{ .rot = rotations.items[rot_at.?].tensor, .signs = sign_tensor };
                // the GDN output projection reads its value heads in tiled
                // order, see setGdnVPerm
                if (hdmd.gdn_v_grouped and std.mem.indexOf(u8, weight_name, ".ssm_out.") != null and
                    !transform.setGdnVPerm(w.ne[0], self.hparams.ssm_dt_rank, self.hparams.ssm_n_group))
                {
                    return exc.throw(.runtime_error, "prism.hadamard: bad GDN head geometry for {s}", .{weight_name});
                }
                const slot = target.getOrPut(std.heap.c_allocator, w) catch oom();
                if (!slot.found_existing) slot.value_ptr.* = transform;
            }
        }

        mod.impl.logInfo("%s: loaded %zu Hadamard-folded weight(s) (%zu inverse-lookup) using %zu rotation(s) and %zu sign vector(s)\n", .{
            "load_tensors_hadamard",
            @as(usize, hdmd.rot.count() + hdmd.inv.count()),
            @as(usize, hdmd.inv.count()),
            rotations.items.len,
            sign_tensors.items.len,
        });
    }

    /// The architecture's own `load_arch_hparams`.
    fn loadArchHparams(self: *Model, ml: *const mod.loader.Loader, exc: *mod.impl.Exception) (mod.impl.Thrown || error{Unported})!void {
        const entry = mod.models.dispatch.forArch(self.arch) orelse return error.Unported;
        try entry.loadArchHparams(self, ml, exc);
    }

    /// Ports `llama_model`'s `arch_name` (llama-model.cpp:2516 @b1b770d90).
    pub fn archName(self: *const Model) [:0]const u8 {
        return mod.arch.archName(self.arch);
    }

    /// Ports `llama_model`'s `type_name` (llama-model.cpp:2520 @b1b770d90).
    pub fn typeName(self: *const Model) [:0]const u8 {
        return mod.model_table.type_names[@intCast(@backingInt(self.type))];
    }

    /// Ports `llama_model`'s `desc` (llama-model.cpp:2524 @b1b770d90).
    pub fn desc(self: *const Model) []const u8 {
        return self.desc_str.items;
    }

    /// Ports `llama_model_rope_type` (llama-model.cpp:3651 @b1b770d90): the
    /// generated rules, and the five that read the hyperparameters.
    pub fn ropeType(self: *const Model) mod.impl.llama_c.enum_llama_rope_type {
        return switch (mod.model_table.rope_rules[@intCast(@backingInt(self.arch))]) {
            .fixed => |t| t,
            .abort => mod.impl.abort("llama-model.cpp", 3855, "unknown architecture"),
            .conditional => switch (self.arch) {
                // drafts for M-RoPE targets carry rope sections and follow the target's temporal dim;
                // DSV4 and ML4 DSpark drafters use normal RoPE; legacy DFlash backbones are NeoX
                .dflash => blk: {
                    const sec = self.hparams.rope_sections;
                    if (sec[0] != 0 or sec[1] != 0 or sec[2] != 0 or sec[3] != 0) break :blk mod.impl.llama_c.LLAMA_ROPE_TYPE_MROPE;
                    break :blk if (self.hparams.dsv4_hc_mult > 0 or self.hparams.n_lora_kv > 0) mod.impl.llama_c.LLAMA_ROPE_TYPE_NORM else mod.impl.llama_c.LLAMA_ROPE_TYPE_NEOX;
                },
                .glm4 => if (self.hparams.useMrope()) mod.impl.llama_c.LLAMA_ROPE_TYPE_MROPE else mod.impl.llama_c.LLAMA_ROPE_TYPE_NORM,
                .glm4_moe, .hunyuan_vl => if (self.hparams.useMrope()) mod.impl.llama_c.LLAMA_ROPE_TYPE_MROPE else mod.impl.llama_c.LLAMA_ROPE_TYPE_NEOX,
                // VL files carry mrope sections; text-only files keep NORM rope
                .bailingmoe3 => if (self.hparams.useMrope()) mod.impl.llama_c.LLAMA_ROPE_TYPE_MROPE else mod.impl.llama_c.LLAMA_ROPE_TYPE_NORM,
                else => unreachable,
            },
        };
    }

    /// Ports `llama_model`'s `print_info` (llama-model.cpp:2584 @b1b770d90).
    pub fn printInfo(self: *const Model, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        const hp = &self.hparams;
        const f = "print_info";
        const rope_scaling_type = try ropeScalingTypeName(hp.rope_scaling_type_train, exc);

        mod.impl.logInfo("%s: arch                  = %s\n", .{ f, self.archName().ptr });
        mod.impl.logInfo("%s: vocab_only            = %d\n", .{ f, @as(c_int, @intFromBool(hp.vocab_only)) });
        mod.impl.logInfo("%s: no_alloc              = %d\n", .{ f, @as(c_int, @intFromBool(self.noAlloc())) });

        if (!hp.vocab_only) {
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(std.heap.c_allocator);
            const n = hp.n_layer_all;

            mod.impl.logInfo("%s: n_ctx_train           = %u\n", .{ f, hp.n_ctx_train });
            mod.impl.logInfo("%s: n_embd_inp            = %u\n", .{ f, hp.nEmbdInp() });
            mod.impl.logInfo("%s: n_embd                = %u\n", .{ f, hp.n_embd });
            mod.impl.logInfo("%s: n_embd_out            = %u\n", .{ f, hp.nEmbdOut() });
            mod.impl.logInfo("%s: n_layer               = %u\n", .{ f, hp.nLayer() });
            mod.impl.logInfo("%s: n_layer_all           = %u\n", .{ f, hp.n_layer_all });
            mod.impl.logInfo("%s: n_head                = %s\n", .{ f, printF(&buf, hp, mod.hparams.Hparams.nHead, n) });
            mod.impl.logInfo("%s: n_head_kv             = %s\n", .{ f, printF(&buf, hp, mod.hparams.Hparams.nHeadKv, n) });
            mod.impl.logInfo("%s: n_rot                 = %u\n", .{ f, hp.n_rot_full });
            mod.impl.logInfo("%s: n_swa                 = %u\n", .{ f, hp.n_swa });
            mod.impl.logInfo("%s: is_swa_any            = %u\n", .{ f, @as(c_uint, @intFromBool(hp.isSwaAny())) });
            mod.impl.logInfo("%s: non_causal_type       = %d\n", .{ f, hp.non_causal_type });
            mod.impl.logInfo("%s: n_embd_head_k         = %u\n", .{ f, hp.n_embd_head_k_full });
            mod.impl.logInfo("%s: n_embd_head_v         = %u\n", .{ f, hp.n_embd_head_v_full });
            mod.impl.logInfo("%s: n_gqa                 = %s\n", .{ f, printF(&buf, hp, mod.hparams.Hparams.nGqa, n) });
            mod.impl.logInfo("%s: n_embd_k_gqa          = %s\n", .{ f, printF(&buf, hp, mod.hparams.Hparams.nEmbdKGqa, n) });
            mod.impl.logInfo("%s: n_embd_v_gqa          = %s\n", .{ f, printF(&buf, hp, mod.hparams.Hparams.nEmbdVGqa, n) });
            mod.impl.logInfo("%s: f_norm_eps            = %.1e\n", .{ f, @as(f64, hp.f_norm_eps) });
            mod.impl.logInfo("%s: f_norm_rms_eps        = %.1e\n", .{ f, @as(f64, hp.f_norm_rms_eps) });
            mod.impl.logInfo("%s: f_clamp_kqv           = %.1e\n", .{ f, @as(f64, hp.f_clamp_kqv) });
            mod.impl.logInfo("%s: f_max_alibi_bias      = %.1e\n", .{ f, @as(f64, hp.f_max_alibi_bias) });
            mod.impl.logInfo("%s: f_logit_scale         = %.1e\n", .{ f, @as(f64, hp.f_logit_scale) });
            mod.impl.logInfo("%s: f_attn_scale          = %.1e\n", .{ f, @as(f64, hp.f_attention_scale) });
            mod.impl.logInfo("%s: f_attn_value_scale    = %.4f\n", .{ f, @as(f64, hp.f_attn_value_scale) });
            mod.impl.logInfo("%s: n_ff                  = %s\n", .{ f, printF(&buf, hp, mod.hparams.Hparams.nFf, n) });
            mod.impl.logInfo("%s: n_expert              = %u\n", .{ f, hp.n_expert });
            mod.impl.logInfo("%s: n_expert_used         = %u\n", .{ f, hp.nExpertUsed(0) });
            mod.impl.logInfo("%s: n_expert_groups       = %d\n", .{ f, hp.n_expert_groups });
            mod.impl.logInfo("%s: n_group_used          = %d\n", .{ f, hp.n_group_used });
            mod.impl.logInfo("%s: causal attn           = %d\n", .{ f, @as(c_int, @intFromBool(hp.causal_attn)) });
            mod.impl.logInfo("%s: pooling type          = %d\n", .{ f, hp.pooling_type });
            mod.impl.logInfo("%s: rope type             = %d\n", .{ f, hp.rope_type });
            mod.impl.logInfo("%s: rope scaling          = %s\n", .{ f, rope_scaling_type.ptr });
            mod.impl.logInfo("%s: freq_base_train       = %.1f\n", .{ f, @as(f64, hp.rope_freq_base_train) });
            mod.impl.logInfo("%s: freq_scale_train      = %g\n", .{ f, @as(f64, hp.rope_freq_scale_train) });
            if (hp.swa_type != 0) {
                mod.impl.logInfo("%s: freq_base_swa         = %.1f\n", .{ f, @as(f64, hp.rope_freq_base_train_swa) });
                mod.impl.logInfo("%s: freq_scale_swa        = %g\n", .{ f, @as(f64, hp.rope_freq_scale_train_swa) });
                mod.impl.logInfo("%s: n_embd_head_k_swa     = %u\n", .{ f, hp.n_embd_head_k_swa });
                mod.impl.logInfo("%s: n_embd_head_v_swa     = %u\n", .{ f, hp.n_embd_head_v_swa });
                mod.impl.logInfo("%s: n_rot_swa             = %u\n", .{ f, hp.n_rot_swa });
            }
            mod.impl.logInfo("%s: n_ctx_orig_yarn       = %u\n", .{ f, hp.n_ctx_orig_yarn });
            mod.impl.logInfo("%s: rope_yarn_log_mul     = %.4f\n", .{ f, @as(f64, hp.rope_yarn_log_mul) });
            mod.impl.logInfo("%s: rope_finetuned        = %s\n", .{ f, @as([*:0]const u8, if (hp.rope_finetuned) "yes" else "unknown") });
            if (self.arch == .granite) {
                var any = false;
                for (hp.deepstack_mapping_arr) |e| any = any or e >= 0;
                if (any) mod.impl.logInfo("%s: deepstack_mapping_arr = %s\n", .{ f, printF(&buf, hp, deepstackAt, n) });
            }
            const s = hp.rope_sections;
            if (s[0] != 0 or s[1] != 0 or s[2] != 0 or s[3] != 0) {
                mod.impl.logInfo("%s: mrope sections        = [%d, %d, %d, %d]\n", .{ f, s[0], s[1], s[2], s[3] });
            }
            if (hp.rope_section_order != mod.hparams.RopeSectionOrder.tyxz) {
                mod.impl.logInfo("%s: rope section order    = %s\n", .{ f, (try ropeSectionOrderName(hp.rope_section_order, exc)).ptr });
            }
            if (self.classifier_labels.items.len != 0) {
                mod.impl.logInfo("%s: n_cls_out             = %u\n", .{ f, hp.n_cls_out });
                for (self.classifier_labels.items, 0..) |label, i| {
                    mod.impl.logInfo("%s: cls_label[%2zu]         = %.*s\n", .{ f, i, @as(c_int, @intCast(label.len)), label.ptr });
                }
            }

            switch (self.arch) {
                .mamba, .mamba2, .jamba, .falcon_h1, .plamo2, .granite_hybrid, .qwen3next, .qwen35, .qwen35moe, .clover_moe, .nemotron_h, .nemotron_h_moe => {
                    mod.impl.logInfo("%s: ssm_d_conv            = %u\n", .{ f, hp.ssm_d_conv });
                    mod.impl.logInfo("%s: ssm_d_inner           = %u\n", .{ f, hp.ssm_d_inner });
                    mod.impl.logInfo("%s: ssm_d_state           = %u\n", .{ f, hp.ssm_d_state });
                    mod.impl.logInfo("%s: ssm_dt_rank           = %u\n", .{ f, hp.ssm_dt_rank });
                    mod.impl.logInfo("%s: ssm_n_group           = %u\n", .{ f, hp.ssm_n_group });
                    mod.impl.logInfo("%s: ssm_dt_b_c_rms        = %d\n", .{ f, @as(c_int, @intFromBool(hp.ssm_dt_b_c_rms)) });
                },
                else => {},
            }

            mod.impl.logInfo("%s: model type            = %s\n", .{ f, self.typeName().ptr });
            const ne: f64 = @floatFromInt(self.n_elements);
            if (ne >= 1e12) {
                mod.impl.logInfo("%s: model params          = %.2f T\n", .{ f, ne * 1e-12 });
            } else if (ne >= 1e9) {
                mod.impl.logInfo("%s: model params          = %.2f B\n", .{ f, ne * 1e-9 });
            } else if (ne >= 1e6) {
                mod.impl.logInfo("%s: model params          = %.2f M\n", .{ f, ne * 1e-6 });
            } else {
                mod.impl.logInfo("%s: model params          = %.2f K\n", .{ f, ne * 1e-3 });
            }

            mod.impl.logInfo("%s: general.name          = %.*s\n", .{ f, @as(c_int, @intCast(self.name.items.len)), self.name.items.ptr });

            self.printArchInfo(exc);
        }

        try self.vocab.printInfo(exc);
    }

    /// The architecture-specific tail of `print_info`.
    fn printArchInfo(self: *const Model, _: *mod.impl.Exception) void {
        const hp = &self.hparams;
        const f = "print_info";
        const a = self.arch;
        if (a == .deepseek) {
            mod.impl.logInfo("%s: n_layer_dense_lead    = %d\n", .{ f, hp.n_layer_dense_lead });
            mod.impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.nFfExp(0) });
            mod.impl.logInfo("%s: n_expert_shared       = %d\n", .{ f, hp.n_expert_shared });
            mod.impl.logInfo("%s: expert_weights_scale  = %.1f\n", .{ f, @as(f64, hp.expert_weights_scale) });
        }
        if (a == .deepseek2 or a == .deepseek2ocr or a == .deepseek32 or a == .glm_dsa or a == .dots3note or a == .mistral4 or
            a == .mistral_large4 or a == .hy_v4)
        {
            mod.impl.logInfo("%s: n_layer_dense_lead    = %d\n", .{ f, hp.n_layer_dense_lead });
            mod.impl.logInfo("%s: n_lora_q              = %d\n", .{ f, hp.n_lora_q });
            mod.impl.logInfo("%s: n_lora_kv             = %d\n", .{ f, hp.n_lora_kv });
            mod.impl.logInfo("%s: n_embd_head_k_mla     = %d\n", .{ f, hp.nEmbdHeadKMla() });
            mod.impl.logInfo("%s: n_embd_head_v_mla     = %d\n", .{ f, hp.nEmbdHeadVMla() });
            mod.impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.nFfExp(0) });
            mod.impl.logInfo("%s: n_expert_shared       = %d\n", .{ f, hp.n_expert_shared });
            mod.impl.logInfo("%s: expert_weights_scale  = %.1f\n", .{ f, @as(f64, hp.expert_weights_scale) });
            mod.impl.logInfo("%s: expert_weights_norm   = %d\n", .{ f, @as(c_int, @intFromBool(hp.expert_weights_norm)) });
            mod.impl.logInfo("%s: expert_gating_func    = %s\n", .{ f, expertGatingFuncName(hp.expert_gating_func).ptr });
        }
        if (a == .qwen2moe) {
            mod.impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.nFfExp(0) });
            mod.impl.logInfo("%s: n_ff_shexp            = %d\n", .{ f, hp.n_ff_shexp });
        }
        if (a == .mellum or a == .cohere2moe or a == .qwen3moe or a == .openai_moe or a == .qwen3vlmoe or a == .rnd1) {
            mod.impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.nFfExp(0) });
        }
        if (a == .minicpm or a == .granite or a == .granite_moe or a == .granite_hybrid or a == .granite_switch or a == .nemotron_h_moe) {
            mod.impl.logInfo("%s: f_embedding_scale     = %f\n", .{ f, @as(f64, hp.f_embedding_scale) });
            mod.impl.logInfo("%s: f_residual_scale      = %f\n", .{ f, @as(f64, hp.f_residual_scale) });
            mod.impl.logInfo("%s: f_attention_scale     = %f\n", .{ f, @as(f64, hp.f_attention_scale) });
            mod.impl.logInfo("%s: n_ff_shexp            = %d\n", .{ f, hp.n_ff_shexp });
        }
        if (a == .bailingmoe) {
            mod.impl.logInfo("%s: n_layer_dense_lead    = %d\n", .{ f, hp.n_layer_dense_lead });
            mod.impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.nFfExp(0) });
            mod.impl.logInfo("%s: n_expert_shared       = %d\n", .{ f, hp.n_expert_shared });
            mod.impl.logInfo("%s: expert_weights_scale  = %.1f\n", .{ f, @as(f64, hp.expert_weights_scale) });
            mod.impl.logInfo("%s: expert_weights_norm   = %d\n", .{ f, @as(c_int, @intFromBool(hp.expert_weights_norm)) });
        }
        if (a == .bailingmoe2 or a == .bailingmoe3) {
            mod.impl.logInfo("%s: n_layer_dense_lead    = %d\n", .{ f, hp.n_layer_dense_lead });
            mod.impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.nFfExp(0) });
            mod.impl.logInfo("%s: n_ff_shexp            = %d\n", .{ f, hp.n_ff_shexp });
            mod.impl.logInfo("%s: n_expert_shared       = %d\n", .{ f, hp.n_expert_shared });
            mod.impl.logInfo("%s: expert_weights_scale  = %.1f\n", .{ f, @as(f64, hp.expert_weights_scale) });
            mod.impl.logInfo("%s: expert_weights_norm   = %d\n", .{ f, @as(c_int, @intFromBool(hp.expert_weights_norm)) });
            mod.impl.logInfo("%s: expert_gating_func    = %s\n", .{ f, expertGatingFuncName(hp.expert_gating_func).ptr });
            mod.impl.logInfo("%s: n_layer_nextn         = %d\n", .{ f, hp.n_layer_nextn });
        }
        if (a == .smallthinker or a == .lfm2moe) {
            mod.impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.nFfExp(0) });
            mod.impl.logInfo("%s: expert_gating_func    = %s\n", .{ f, expertGatingFuncName(hp.expert_gating_func).ptr });
        }
        if (a == .grovemoe) {
            mod.impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.nFfExp(0) });
            mod.impl.logInfo("%s: n_ff_chexp            = %d\n", .{ f, hp.n_ff_chexp });
            mod.impl.logInfo("%s: n_group_experts       = %d\n", .{ f, hp.n_group_experts });
            mod.impl.logInfo("%s: expert_group_scale    = %.2f\n", .{ f, @as(f64, hp.expert_group_scale) });
        }
    }

    // -------------------------------------------------------------------------
    // Tensors

    /// Ports `llama_model_base`'s `load_tensors` (llama-model.cpp:1689 @b1b770d90).
    ///
    /// Return: false when the progress callback cancelled; `error.Thrown`
    /// where the C++ throws; `error.Unported` for an architecture whose
    /// `load_arch_tensors` is not ported.
    pub fn loadTensors(self: *Model, ml: *mod.loader.Loader, exc: *mod.impl.Exception) (mod.impl.Thrown || error{Unported})!bool {
        const split_mode = self.params.split_mode;
        const use_mlock = self.params.load_mode == mod.impl.llama_c.LLAMA_LOAD_MODE_MLOCK or self.params.load_mode == mod.impl.llama_c.LLAMA_LOAD_MODE_MMAP_MLOCK;
        const tensor_split = self.params.tensor_split;

        const n_layer_all: c_int = @intCast(self.hparams.n_layer_all);
        const n_gpu_layers: c_int = @bitCast(self.nGpuLayers());

        const use_mmap_buffer = true;

        self.ml = ml;
        defer self.ml = null;

        if (ml.use_mmap and self.params.load_mode == mod.impl.llama_c.LLAMA_LOAD_MODE_AUTO) {
            for (self.devices.items) |dev| {
                var props: mod.impl.llama_c.ggml_backend_dev_props = undefined;
                mod.impl.llama_c.ggml_backend_dev_get_props(dev.dev, &props);
                if (!props.caps.mmap_support) {
                    ml.use_mmap = false;
                    break;
                }
            }
        }

        // Resolve AUTO on systems without mmap support (e.g. iGPUs): fall
        // back to OFF unless mmap is in use (upstream #28160, #31943).
        if (ml.lazy.mode == mod.impl.llama_c.LLAMA_LAZY_MODE_AUTO and !ml.use_mmap) {
            for (self.devices.items) |dev| {
                var props: mod.impl.llama_c.ggml_backend_dev_props = undefined;
                mod.impl.llama_c.ggml_backend_dev_get_props(dev.dev, &props);
                if (!props.caps.mmap_support) {
                    ml.lazy.mode = mod.impl.llama_c.LLAMA_LAZY_MODE_OFF;
                    break;
                }
            }
        }

        const load_mode_name = if (self.params.load_mode == mod.impl.llama_c.LLAMA_LOAD_MODE_AUTO)
            loadModeName(if (ml.use_mmap) mod.impl.llama_c.LLAMA_LOAD_MODE_MMAP else mod.impl.llama_c.LLAMA_LOAD_MODE_NONE)
        else
            loadModeName(self.params.load_mode);

        mod.impl.logInfo("%s: loading model tensors, this can take a while... (load_mode = %s)\n", .{ "load_tensors", load_mode_name.ptr });

        // Build a list of buffer types for the CPU and GPU devices.
        try makeCpuBuftList(self.cpu_buft_list, self.devices.items, self.params.use_extra_bufts, self.params.no_host, exc);
        for (self.devices.items) |dev| {
            const list = std.heap.c_allocator.create(mod.loader.BuftList) catch oom();
            list.* = .empty;
            try makeGpuBuftList(list, dev.dev, split_mode, tensor_split, exc);
            // CPU buffer types as a fallback.
            list.appendSlice(std.heap.c_allocator, self.cpu_buft_list.items) catch oom();
            self.gpu_buft_list.append(std.heap.c_allocator, .{ .dev = dev.dev, .list = list }) catch oom();
        }

        const cpu_dev = mod.impl.llama_c.ggml_backend_dev_by_type(mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU) orelse
            return exc.throw(.runtime_error, "load_tensors: no CPU backend found", .{});

        // Calculate the split points.
        const n_devices = self.devices.items.len;
        var all_zero = true;
        if (tensor_split != null) {
            for (tensor_split[0..n_devices]) |x| if (x != 0.0) {
                all_zero = false;
            };
        }
        const splits = std.heap.c_allocator.alloc(f32, n_devices) catch oom();
        defer std.heap.c_allocator.free(splits);
        if (all_zero) {
            // Default split, by free memory.
            for (self.devices.items, 0..) |d, i| {
                var total: usize = 0;
                var free: usize = 0;
                mod.impl.llama_c.ggml_backend_dev_memory(d.dev, &free, &total);
                // Devices with nothing to report fall back to host memory.
                if (free == 0 and total == 0) mod.impl.llama_c.ggml_backend_dev_memory(cpu_dev, &free, &total);
                splits[i] = @floatFromInt(free);
            }
        } else {
            @memcpy(splits, tensor_split[0..n_devices]);
        }

        // Sum and normalize the splits to get the split points.
        var split_sum: f32 = 0.0;
        for (splits) |*s| {
            split_sum += s.*;
            s.* = split_sum;
        }
        for (splits, 0..) |*s, i| {
            s.* = if (split_sum != 0.0) s.* / split_sum else @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(n_devices));
        }

        const i_gpu_start: c_int = @max(n_layer_all + 1 - n_gpu_layers, 0);
        const act_gpu_layers: c_int = if (n_devices == 0) 0 else @min(n_gpu_layers, n_layer_all + 1);

        // The input layer always stays on the CPU: there is very little
        // benefit to offloading it.
        self.dev_input = .{ .dev = cpu_dev, .buft_list = self.cpu_buft_list };

        // The repeating layers go to the devices according to the splits.
        self.dev_layer.resize(std.heap.c_allocator, @intCast(n_layer_all)) catch oom();
        var il: c_int = 0;
        while (il < n_layer_all) : (il += 1) {
            self.dev_layer.items[@intCast(il)] = self.layerBuftList(il, cpu_dev, splits, i_gpu_start, act_gpu_layers);
        }

        // The output layer.
        self.dev_output = self.layerBuftList(n_layer_all, cpu_dev, splits, i_gpu_start, act_gpu_layers);

        // Create tensors for the weights.
        {
            const n_expert: i64 = self.hparams.n_expert;
            if (n_expert > 0 and self.hparams.nExpertUsedMax() == 0) {
                return exc.throw(.runtime_error, "model has expert layers but no expert layers are used", .{});
            }

            self.layers = std.heap.c_allocator.alloc(Layer, @intCast(n_layer_all)) catch oom();
            @memset(self.layers, .{});

            // The per-model loading function.
            try self.loadArchTensors(ml, exc);

            // Optional per-tensor and per-expert ".scale" tensors, e.g.
            // NVFP4's scale2, loaded for every architecture at once.
            // Only the first n_layer_all: T5 may have grown `layers` to its
            // decoder's layer count, and those are not visited.
            const nr = mod.loader.tensor_not_required;
            for (self.layers[0..@intCast(n_layer_all)], 0..) |*layer, i_| {
                const i: i32 = @intCast(i_);
                const one = [_]i64{1};
                const nexp = [_]i64{n_expert};

                // KV cache scales (FP8 KV cache), F32 when present.
                if (self.hparams.hasKv(@intCast(i))) {
                    layer.k_cache_s = try self.createTensor(.{ .tensor = .attn_k, .suffix = "k_scale", .bid = i }, &one, nr, exc);
                    layer.v_cache_s = try self.createTensor(.{ .tensor = .attn_v, .suffix = "v_scale", .bid = i }, &one, nr, exc);

                    if ((layer.k_cache_s != null and layer.k_cache_s.?.type != mod.impl.llama_c.GGML_TYPE_F32) or
                        (layer.v_cache_s != null and layer.v_cache_s.?.type != mod.impl.llama_c.GGML_TYPE_F32))
                    {
                        return exc.throw(.runtime_error, "KV cache scales for layer {d} must be F32", .{i});
                    }
                }

                // Attention weight scales.
                if (layer.wq_s == null and layer.wq != null) layer.wq_s = try self.createTensor(.{ .tensor = .attn_q, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.wk_s == null and layer.wk != null) layer.wk_s = try self.createTensor(.{ .tensor = .attn_k, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.wv_s == null and layer.wv != null) layer.wv_s = try self.createTensor(.{ .tensor = .attn_v, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.wo_s == null and layer.wo != null) layer.wo_s = try self.createTensor(.{ .tensor = .attn_out, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.wqkv_s == null and layer.wqkv != null) layer.wqkv_s = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.wqkv_gate_s == null and layer.wqkv_gate != null) layer.wqkv_gate_s = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "scale", .bid = i }, &one, nr, exc);

                // Dense FFN weight scales.
                if (layer.ffn_gate_s == null and layer.ffn_gate != null) layer.ffn_gate_s = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_down_s == null and layer.ffn_down != null) layer.ffn_down_s = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_up_s == null and layer.ffn_up != null) layer.ffn_up_s = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_gate_shexp_s == null and layer.ffn_gate_shexp != null) layer.ffn_gate_shexp_s = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_down_shexp_s == null and layer.ffn_down_shexp != null) layer.ffn_down_shexp_s = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_up_shexp_s == null and layer.ffn_up_shexp != null) layer.ffn_up_shexp_s = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "scale", .bid = i }, &one, nr, exc);

                // MoE expert weight scales, one per expert.
                if (layer.ffn_gate_exps_s == null and layer.ffn_gate_exps != null) layer.ffn_gate_exps_s = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "scale", .bid = i }, &nexp, nr, exc);
                if (layer.ffn_down_exps_s == null and layer.ffn_down_exps != null) layer.ffn_down_exps_s = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "scale", .bid = i }, &nexp, nr, exc);
                if (layer.ffn_up_exps_s == null and layer.ffn_up_exps != null) layer.ffn_up_exps_s = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "scale", .bid = i }, &nexp, nr, exc);

                // Recurrent and linear-attention weight scales.
                if (layer.ssm_in_s == null and layer.ssm_in != null) layer.ssm_in_s = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.ssm_out_s == null and layer.ssm_out != null) layer.ssm_out_s = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.ssm_alpha_s == null and layer.ssm_alpha != null) layer.ssm_alpha_s = try self.createTensor(.{ .tensor = .ssm_alpha, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.ssm_beta_s == null and layer.ssm_beta != null) layer.ssm_beta_s = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.nextn.eh_proj_s == null and layer.nextn.eh_proj != null) layer.nextn.eh_proj_s = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "scale", .bid = i }, &one, nr, exc);
                if (layer.nextn.shared_head_head_s == null and layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "scale", .bid = i }, &one, nr, exc);

                // Input scales.
                if (layer.wq_in_s == null and layer.wq != null) layer.wq_in_s = try self.createTensor(.{ .tensor = .attn_q, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.wk_in_s == null and layer.wk != null) layer.wk_in_s = try self.createTensor(.{ .tensor = .attn_k, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.wv_in_s == null and layer.wv != null) layer.wv_in_s = try self.createTensor(.{ .tensor = .attn_v, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.wo_in_s == null and layer.wo != null) layer.wo_in_s = try self.createTensor(.{ .tensor = .attn_out, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.wqkv_in_s == null and layer.wqkv != null) layer.wqkv_in_s = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.wqkv_gate_in_s == null and layer.wqkv_gate != null) layer.wqkv_gate_in_s = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_gate_in_s == null and layer.ffn_gate != null) layer.ffn_gate_in_s = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_down_in_s == null and layer.ffn_down != null) layer.ffn_down_in_s = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_up_in_s == null and layer.ffn_up != null) layer.ffn_up_in_s = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_gate_exps_in_s == null and layer.ffn_gate_exps != null) layer.ffn_gate_exps_in_s = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "input_scale", .bid = i }, &nexp, nr, exc);
                if (layer.ffn_down_exps_in_s == null and layer.ffn_down_exps != null) layer.ffn_down_exps_in_s = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "input_scale", .bid = i }, &nexp, nr, exc);
                if (layer.ffn_up_exps_in_s == null and layer.ffn_up_exps != null) layer.ffn_up_exps_in_s = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "input_scale", .bid = i }, &nexp, nr, exc);
                if (layer.ffn_gate_shexp_in_s == null and layer.ffn_gate_shexp != null) layer.ffn_gate_shexp_in_s = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_down_shexp_in_s == null and layer.ffn_down_shexp != null) layer.ffn_down_shexp_in_s = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ffn_up_shexp_in_s == null and layer.ffn_up_shexp != null) layer.ffn_up_shexp_in_s = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ssm_in_in_s == null and layer.ssm_in != null) layer.ssm_in_in_s = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ssm_out_in_s == null and layer.ssm_out != null) layer.ssm_out_in_s = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ssm_alpha_in_s == null and layer.ssm_alpha != null) layer.ssm_alpha_in_s = try self.createTensor(.{ .tensor = .ssm_alpha, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.ssm_beta_in_s == null and layer.ssm_beta != null) layer.ssm_beta_in_s = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.nextn.eh_proj_in_s == null and layer.nextn.eh_proj != null) layer.nextn.eh_proj_in_s = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "input_scale", .bid = i }, &one, nr, exc);
                if (layer.nextn.shared_head_head_in_s == null and layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_in_s = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "input_scale", .bid = i }, &one, nr, exc);
            }
            // Output scales, for any output type, when the file has them.
            if (self.output != null) {
                const one = [_]i64{1};
                if (self.output_s == null and outputScaleInFile(ml, "scale")) self.output_s = try self.createTensor(.{ .tensor = .output, .suffix = "scale" }, &one, nr, exc);
                if (self.output_in_s == null and outputScaleInFile(ml, "input_scale")) self.output_in_s = try self.createTensor(.{ .tensor = .output, .suffix = "input_scale" }, &one, nr, exc);
            }
        }
        try ml.doneGettingTensors(false, exc);

        if (self.per_layer_tok_embd) |t| {
            if (ml.lazy.has(t)) {
                mod.impl.logInfo("%s: enabling prefetch for '%s'\n", .{ "load_tensors", &t.name });
                self.can_prefetch.put(std.heap.c_allocator, t, {}) catch oom();
            }
        }

        if (self.output != null and self.tok_embd != null and
            std.mem.eql(u8, mod.gguf_meta.cString(&self.output.?.name), mod.gguf_meta.cString(&self.tok_embd.?.name)) and
            (self.output_s != null or self.output_in_s != null))
        {
            return exc.throw(.runtime_error, "output scales require a separate output weight", .{});
        }

        // Populate tensors_by_name.
        for (ml.ctx_map.items) |e| {
            var t: ?*mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_get_first_tensor(e.ctx.?);
            while (t) |cur| : (t = mod.impl.llama_c.ggml_get_next_tensor(e.ctx.?, cur)) {
                self.tensors_by_name.append(std.heap.c_allocator, cur) catch oom();
            }
        }

        // the per-weight activation precision policy
        try self.prec_policy.load(ml, self, exc);

        try ml.initMappings(true, if (use_mlock) &self.mlock_mmaps else null, exc);
        self.mappings.ensureTotalCapacity(std.heap.c_allocator, ml.mappings.items.len) catch oom();

        // Create the backend buffers. The model owns them through
        // `ctxs_bufs`; this is a view of them for loading the data.
        const CtxBufsView = struct { *mod.impl.llama_c.ggml_context, []const mod.impl.llama_c.ggml_backend_buffer_t };
        var ctx_bufs: std.ArrayList(CtxBufsView) = .empty;
        defer ctx_bufs.deinit(std.heap.c_allocator);

        for (ml.ctx_map.items) |*entry| {
            const buft = entry.key.buft;
            const ctx = entry.ctx.?;

            // Skip contexts without tensors.
            if (mod.impl.llama_c.ggml_get_first_tensor(ctx) == null) continue;

            // Check if it is possible to use buffer_from_host_ptr with this buffer type.
            var dev = mod.impl.llama_c.ggml_backend_buft_get_device(buft);
            if (dev == null) {
                // The CPU backend's buffer type has a null device.
                dev = mod.impl.llama_c.ggml_backend_dev_by_type(mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU) orelse
                    return exc.throw(.runtime_error, "load_tensors: no CPU backend found", .{});
            }
            var props: mod.impl.llama_c.ggml_backend_dev_props = undefined;
            mod.impl.llama_c.ggml_backend_dev_get_props(dev, &props);
            const buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
            const is_default_buft = buft == mod.impl.llama_c.ggml_backend_dev_buffer_type(dev);

            var bufs: std.ArrayList(mod.impl.llama_c.ggml_backend_buffer_t) = .empty;

            // a lazy context is mapped whatever the load mode, but the memory-fit pass maps nothing
            const is_lazy_mapped = entry.key.lazy and !ml.no_alloc;

            if ((ml.use_mmap or is_lazy_mapped) and use_mmap_buffer and buffer_from_host_ptr_supported and is_default_buft) {
                mod.impl.assert(!ml.no_alloc, "llama-model.cpp", 2053, "!ml.no_alloc");
                const max_size = mod.impl.llama_c.ggml_get_max_tensor_size(ctx);

                for (0..ml.files.items.len) |idx| {
                    // Map the regions holding this context's tensors. On
                    // Apple silicon a backend buffer pins everything it
                    // wraps, so the regions other contexts own (lazy
                    // tensors, MoE tensors kept on the CPU) are left out and
                    // out of RAM; it also allows partial offloading when the
                    // model is larger than a Metal buffer but not than RAM.
                    const addr: usize = @intFromPtr(ml.mappings.items[idx].addr);

                    const ranges = ml.getMappingRanges(@intCast(idx), ctx);
                    defer std.heap.c_allocator.free(ranges);
                    for (ranges) |range| {
                        const buf = mod.impl.llama_c.ggml_backend_dev_buffer_from_host_ptr(dev, @ptrFromInt(addr + range.first), range.last - range.first, max_size) orelse
                            return exc.throw(.runtime_error, "unable to allocate {s} buffer", .{std.mem.span(mod.impl.llama_c.ggml_backend_buft_name(buft))});
                        bufs.append(std.heap.c_allocator, buf) catch oom();
                    }
                }
            } else {
                var buf: mod.impl.llama_c.ggml_backend_buffer_t = undefined;
                if (ml.no_alloc) {
                    buf = mod.impl.llama_c.ggml_backend_buft_alloc_buffer(buft, 0); // dummy buffer
                    var t: ?*mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_get_first_tensor(ctx);
                    while (t) |cur| : (t = mod.impl.llama_c.ggml_get_next_tensor(ctx, cur)) cur.buffer = buf;
                } else {
                    buf = mod.impl.llama_c.ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
                }
                if (buf == null) return exc.throw(.runtime_error, "unable to allocate {s} buffer", .{std.mem.span(mod.impl.llama_c.ggml_backend_buft_name(buft))});
                // lock the buffers that hold the tensors: a multi_buffer has
                // no base, and its parts are pageable memory when pinning fails
                if (use_mlock and !ml.no_alloc and mod.impl.llama_c.ggml_backend_buffer_is_host(buf)) {
                    var lock_bufs: std.ArrayList(mod.impl.llama_c.ggml_backend_buffer_t) = .empty;
                    defer lock_bufs.deinit(std.heap.c_allocator);
                    var t: ?*mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_get_first_tensor(ctx);
                    while (t) |cur| : (t = mod.impl.llama_c.ggml_get_next_tensor(ctx, cur)) {
                        if (cur.buffer != null and std.mem.indexOfScalar(mod.impl.llama_c.ggml_backend_buffer_t, lock_bufs.items, cur.buffer) == null) {
                            lock_bufs.append(std.heap.c_allocator, cur.buffer) catch oom();
                        }
                    }
                    for (lock_bufs.items) |lock_buf| {
                        const lock = std.heap.c_allocator.create(mod.mmap.Mlock) catch oom();
                        lock.* = .{};
                        self.mlock_bufs.append(std.heap.c_allocator, lock) catch oom();
                        lock.init(mod.impl.llama_c.ggml_backend_buffer_get_base(lock_buf).?);
                        lock.growTo(mod.impl.llama_c.ggml_backend_buffer_get_size(lock_buf));
                    }
                }
                bufs.append(std.heap.c_allocator, buf) catch oom();
            }

            // These buffers hold weights; the scheduler prefers the backend
            // holding an op's weight.
            for (bufs.items) |buf| mod.impl.llama_c.ggml_backend_buffer_set_usage(buf, mod.impl.llama_c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            self.ctxs_bufs.append(std.heap.c_allocator, .{ .ctx = ctx, .bufs = bufs }) catch oom();
            entry.ctx = null;

            ctx_bufs.append(std.heap.c_allocator, .{ ctx, self.ctxs_bufs.items[self.ctxs_bufs.items.len - 1].bufs.items }) catch oom();
        }

        if (supportsGpuOffload()) {
            const n_gpu: c_int = @min(n_gpu_layers, n_layer_all);
            var n_repeating = n_gpu;
            if (n_repeating > 0) {
                mod.impl.logInfo("%s: offloading output layer to GPU\n", .{"load_tensors"});
                n_repeating -= 1;
            }
            mod.impl.logInfo("%s: offloading %d repeating layers to GPU\n", .{ "load_tensors", n_repeating });

            const max_backend_supported_layers = n_layer_all + 1;
            const max_offloadable_layers = n_layer_all + 1;

            mod.impl.logInfo("%s: offloaded %d/%d layers to GPU\n", .{ "load_tensors", @min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers });
        }

        // Memory requirements per buffer type.
        for (self.ctxs_bufs.items) |cb| {
            for (cb.bufs.items) |buf| {
                mod.impl.logInfo("%s: %12s model buffer size = %8.2f MiB\n", .{
                    "load_tensors", mod.impl.llama_c.ggml_backend_buffer_name(buf), @as(f64, @floatFromInt(mod.impl.llama_c.ggml_backend_buffer_get_size(buf))) / 1024.0 / 1024.0,
                });
            }
        }

        if (ml.no_alloc) return true;

        // without mmap, load non-host buffers first: their tensors go through a staging buffer, which is cheapest while the fewest weights are resident
        if (!ml.use_mmap) {
            // `std::stable_partition`: the staged contexts first, each group
            // in its own order, a context's test its first buffer.
            const staged = struct {
                fn f(e: *const CtxBufsView) bool {
                    return e[1].len != 0 and !mod.impl.llama_c.ggml_backend_buffer_is_host(e[1][0]);
                }
            }.f;
            var partitioned: std.ArrayList(CtxBufsView) = .empty;
            defer partitioned.deinit(std.heap.c_allocator);
            partitioned.ensureTotalCapacity(std.heap.c_allocator, ctx_bufs.items.len) catch oom();
            for (ctx_bufs.items) |*e| if (staged(e)) partitioned.appendAssumeCapacity(e.*);
            for (ctx_bufs.items) |*e| if (!staged(e)) partitioned.appendAssumeCapacity(e.*);
            @memcpy(ctx_bufs.items, partitioned.items);
        }

        // Load tensor data.
        for (ctx_bufs.items) |e| {
            if (!try ml.loadAllData(e[0], e[1], if (use_mlock) &self.mlock_mmaps else null, self.params.progress_callback, self.params.progress_callback_user_data, exc)) {
                return false;
            }
        }

        if (use_mmap_buffer) {
            for (ml.mappings.items) |m| self.mappings.append(std.heap.c_allocator, m) catch oom();
            ml.mappings.clearRetainingCapacity();
        }

        try self.loadTensorsHadamard(exc);

        return true;
    }

    /// `get_layer_buft_list`, a lambda of `load_tensors` (llama-model.cpp:1689
    /// @b1b770d90): layer `il`'s device and buffer types.
    fn layerBuftList(self: *Model, il: c_int, cpu_dev: mod.impl.llama_c.ggml_backend_dev_t, splits: []const f32, i_gpu_start: c_int, act_gpu_layers: c_int) LayerDev {
        const n_layer_all: c_int = @intCast(self.hparams.n_layer_all);
        const is_swa: c_int = @intFromBool(il < n_layer_all and self.hparams.isSwa(@intCast(il)));
        if (il < i_gpu_start or (il - i_gpu_start) >= act_gpu_layers) {
            mod.impl.logDebug("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", .{ il, mod.impl.llama_c.ggml_backend_dev_name(cpu_dev), is_swa });
            return .{ .dev = cpu_dev, .buft_list = self.cpu_buft_list };
        }
        // `std`'s `upper_bound`: the first split point greater than the layer's
        // fraction.
        const frac = @as(f32, @floatFromInt(il - i_gpu_start)) / @as(f32, @floatFromInt(act_gpu_layers));
        var layer_gpu: usize = 0;
        while (layer_gpu < splits.len and !(frac < splits[layer_gpu])) layer_gpu += 1;
        const dev = self.devices.items[layer_gpu].dev;
        mod.impl.logDebug("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", .{ il, mod.impl.llama_c.ggml_backend_dev_name(dev), is_swa });
        return .{ .dev = dev, .buft_list = self.gpuBuftList(dev).? };
    }

    fn gpuBuftList(self: *const Model, dev: mod.impl.llama_c.ggml_backend_dev_t) ?*mod.loader.BuftList {
        for (self.gpu_buft_list.items) |g| if (g.dev == dev) return g.list;
        return null;
    }

    /// The test `load_tensors` (llama-model.cpp:1689 @b1b770d90) makes before
    /// creating an output scale: whether the file names `output.<suffix>`,
    /// as a loaded weight or in the metadata.
    fn outputScaleInFile(ml: *const mod.loader.Loader, suffix: []const u8) bool {
        const n = mod.arch.tensorName(.output, suffix, -1, -1);
        if (ml.getWeight(n.slice()) != null) return true;
        var z: [mod.arch.max_name + 1]u8 = undefined;
        @memcpy(z[0..n.len], n.slice());
        z[n.len] = 0;
        return mod.impl.llama_c.gguf_find_tensor(ml.meta.ctx, &z) >= 0;
    }

    /// Ports `llama_model_base`'s `create_tensor` (llama-model.cpp:2505
    /// @b1b770d90), through the loader the model is being loaded from
    /// (llama-model.cpp:4025 @b1b770d90). A block that is no model layer --
    /// a head's, past `dev_layer` -- goes with the output.
    ///
    /// Return: the tensor, null when absent and optional or skipped, or
    /// `error.Thrown`.
    pub fn createTensor(self: *Model, tn: mod.loader.Tn, ne: []const i64, flags: u32, exc: *mod.impl.Exception) mod.impl.Thrown!?*mod.impl.llama_c.ggml_tensor {
        mod.impl.assert(self.ml != null, "llama-model.cpp", 4026, "ml != nullptr");
        var buft_list_layer: ?*const mod.loader.BuftList = null;
        if (tn.bid != -1) {
            buft_list_layer = if (@as(usize, @bitCast(@as(isize, tn.bid))) < self.dev_layer.items.len) self.dev_layer.items[@intCast(tn.bid)].buft_list.? else self.dev_output.buft_list.?;
        }
        return self.ml.?.createTensor(&self.hparams, self.cpu_buft_list, self.dev_input.buft_list.?, self.dev_output.buft_list.?, buft_list_layer, tn, ne, flags, exc);
    }

    /// Ports `llama_model_base`'s `create_tensor_qkv` (llama-model.cpp:4047
    /// @b1b770d90): a fused QKV projection when the file has one, else the
    /// three separate ones.
    pub fn createTensorQkv(self: *Model, layer: *Layer, bid: i32, n_embd_: i64, n_embd_q_: i64, n_embd_k_: i64, n_embd_v_: i64, flags: u32, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        const n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;
        const nr = mod.loader.tensor_not_required;
        const siv = mod.loader.tensor_skip_if_virtual;

        if (flags & mod.loader.tensor_skip != 0) {
            const skip = nr | mod.loader.tensor_skip;
            _ = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_embd_qkv }, skip | siv, exc);
            _ = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "bias", .bid = bid }, &.{n_embd_qkv}, skip | siv, exc);
            _ = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_embd_q_ }, skip, exc);
            _ = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_embd_k_ }, skip, exc);
            _ = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_embd_v_ }, skip, exc);
            _ = try self.createTensor(.{ .tensor = .attn_q, .suffix = "bias", .bid = bid }, &.{n_embd_q_}, skip, exc);
            _ = try self.createTensor(.{ .tensor = .attn_k, .suffix = "bias", .bid = bid }, &.{n_embd_k_}, skip, exc);
            _ = try self.createTensor(.{ .tensor = .attn_v, .suffix = "bias", .bid = bid }, &.{n_embd_v_}, skip, exc);
            return;
        }

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_embd_qkv }, nr | siv, exc);
        if (layer.wqkv != null) {
            layer.wqkv_b = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "bias", .bid = bid }, &.{n_embd_qkv}, nr | siv, exc);
            // Fused weights may coexist with separate Q/K/V biases in
            // legacy or custom GGUFs.
            if (layer.wqkv_b == null) {
                layer.wq_b = try self.createTensor(.{ .tensor = .attn_q, .suffix = "bias", .bid = bid }, &.{n_embd_q_}, nr, exc);
                layer.wk_b = try self.createTensor(.{ .tensor = .attn_k, .suffix = "bias", .bid = bid }, &.{n_embd_k_}, nr, exc);
                layer.wv_b = try self.createTensor(.{ .tensor = .attn_v, .suffix = "bias", .bid = bid }, &.{n_embd_v_}, nr, exc);
            }
        } else {
            layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_embd_q_ }, flags, exc);
            layer.wk = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_embd_k_ }, flags, exc);
            layer.wv = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_embd_v_ }, flags, exc);
            layer.wq_b = try self.createTensor(.{ .tensor = .attn_q, .suffix = "bias", .bid = bid }, &.{n_embd_q_}, nr, exc);
            layer.wk_b = try self.createTensor(.{ .tensor = .attn_k, .suffix = "bias", .bid = bid }, &.{n_embd_k_}, nr, exc);
            layer.wv_b = try self.createTensor(.{ .tensor = .attn_v, .suffix = "bias", .bid = bid }, &.{n_embd_v_}, nr, exc);
        }
    }

    /// Mirrors `nextn_flags_t` (llama-model.h:1062 @b1b770d90), a struct of
    /// `llama_model_base`:
    /// the tensor flags for a file that holds the trunk, the NextN layers,
    /// or both.
    pub const NextnFlags = struct {
        /// `TENSOR_NOT_REQUIRED` when the file holds only the NextN layers.
        trunk: u32,
        /// `TENSOR_NOT_REQUIRED` when the file holds only the trunk,
        /// `TENSOR_SKIP` when MTP is not loaded.
        mtp: u32,
    };

    /// Ports `llama_model_base`'s `nextn_flags` (llama-model.cpp:4085
    /// @b1b770d90): a file without the first trunk layer is MTP-only, a file
    /// without the first NextN layer is trunk-only.
    ///
    /// Parameters:
    /// - `ml`: the loader whose tensors are probed.
    /// - `trunk_probe`: the tensor of layer 0 that marks the trunk, the
    ///   C++'s default `LLM_TENSOR_ATTN_NORM`.
    ///
    /// Return: the flags.
    pub fn nextnFlags(self: *const Model, ml: *const mod.loader.Loader, trunk_probe: mod.arch_table.Tensor) NextnFlags {
        var trunk: u32 = 0;
        var mtp: u32 = 0;

        if (self.hparams.n_layer_nextn > 0) {
            const probe = (mod.loader.Tn{ .tensor = trunk_probe, .suffix = "weight", .bid = 0 }).str();
            if (ml.getWeight(probe.slice()) == null) {
                trunk = mod.loader.tensor_not_required;
            }
            const eh = (mod.loader.Tn{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = @intCast(self.hparams.nLayer()) }).str();
            if (ml.getWeight(eh.slice()) == null) {
                mtp = mod.loader.tensor_not_required;
            }
        }

        if (!ml.load_mtp) {
            mtp |= mod.loader.tensor_skip;
        }

        return .{ .trunk = trunk, .mtp = mtp };
    }

    /// Ports `llama_model_base`'s `load_swa_pattern` (llama-model.cpp:4106
    /// @b1b770d90): the SWA pattern as one flag per layer when the key is an
    /// array, else as a period -- the key's value, or `n_pattern` when it is
    /// absent -- expanded by `set_swa_pattern`.
    ///
    /// Parameters:
    /// - `ml`: the loader whose metadata is read.
    /// - `n_pattern`: the period when the key is absent.
    /// - `dense_first`: whether the full-attention layer opens each period.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: nothing, or `error.Thrown` for a malformed key.
    pub fn loadSwaPattern(self: *Model, ml: *const mod.loader.Loader, n_pattern_: u32, dense_first: bool, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        const hp = &self.hparams;
        if (try ml.meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, false, exc)) {
            return;
        }

        var n_pattern = n_pattern_;
        if (try ml.meta.getKey(u32, .attention_sliding_window_pattern, false, exc)) |v| n_pattern = v;
        hp.setSwaPattern(n_pattern, dense_first);
    }

    /// Ports `llama_model_base`'s `create_tensor_gate_up_exps`
    /// (llama-model.cpp:4030 @b1b770d90): the merged gate and up experts
    /// when the file has them, the two separate tensors otherwise. Under
    /// `TENSOR_SKIP` every form is created skipped and `layer` is untouched.
    ///
    /// Parameters:
    /// - `layer`: the layer whose `ffn_gate_up_exps`, or `ffn_gate_exps`
    ///   and `ffn_up_exps`, are set.
    /// - `flags`: for the separate tensors; the merged one is always
    ///   optional.
    pub fn createTensorGateUpExps(self: *Model, layer: *Layer, bid: i32, n_embd_: i64, n_ff_: i64, n_expert_: i64, flags: u32, exc: *mod.impl.Exception) mod.impl.Thrown!void {
        if (flags & mod.loader.tensor_skip != 0) {
            const skip = mod.loader.tensor_not_required | mod.loader.tensor_skip;

            _ = try self.createTensor(.{ .tensor = .ffn_gate_up_exps, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_ff_ * 2, n_expert_ }, skip | mod.loader.tensor_skip_if_virtual, exc);
            _ = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_ff_, n_expert_ }, skip, exc);
            _ = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_ff_, n_expert_ }, skip, exc);
            return;
        }

        layer.ffn_gate_up_exps = try self.createTensor(.{ .tensor = .ffn_gate_up_exps, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_ff_ * 2, n_expert_ }, mod.loader.tensor_not_required, exc);
        if (layer.ffn_gate_up_exps == null) {
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_ff_, n_expert_ }, flags, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = bid }, &.{ n_embd_, n_ff_, n_expert_ }, flags, exc);
        }
    }

    /// Mirrors `LLAMA_LOAD_LOCALS` (llama-model.h:1092 @b1b770d90): the
    /// dimensions every `load_arch_tensors` reads, taken once.
    pub const Locals = struct {
        n_layer: i32,
        n_layer_all: i32,
        n_layer_nextn: i32,
        n_head: i64,
        n_head_kv: i64,
        n_embd: i64,
        n_embd_k_gqa: i64,
        n_embd_v_gqa: i64,
        n_embd_head_k: i64,
        n_embd_head_v: i64,
        n_ff: i64,
        n_embd_gqa: i64,
        n_vocab: i64,
        n_token_types: i64,
        n_rot: i64,
        n_expert: i64,
        n_expert_used: i64,
        n_ctx_train: i64,
    };

    /// `LLAMA_LOAD_LOCALS` for this model.
    ///
    /// Return: the locals, by value.
    pub fn loadLocals(self: *const Model) Locals {
        const hp = &self.hparams;
        return .{
            .n_layer = @intCast(hp.nLayer()),
            .n_layer_all = @intCast(hp.n_layer_all),
            .n_layer_nextn = @intCast(hp.n_layer_nextn),
            .n_head = hp.nHead(0),
            .n_head_kv = hp.nHeadKv(0),
            .n_embd = hp.n_embd,
            .n_embd_k_gqa = hp.nEmbdKGqa(0),
            .n_embd_v_gqa = hp.nEmbdVGqa(0),
            .n_embd_head_k = hp.nEmbdHeadK(0),
            .n_embd_head_v = hp.nEmbdHeadV(0),
            .n_ff = hp.nFf(0),
            .n_embd_gqa = hp.nEmbdVGqa(0),
            .n_vocab = self.vocab.nTokens(),
            .n_token_types = self.vocab.n_token_types,
            .n_rot = hp.nRot(0),
            .n_expert = hp.n_expert,
            .n_expert_used = hp.nExpertUsed(0),
            .n_ctx_train = hp.n_ctx_train,
        };
    }

    /// The architecture's own `load_arch_tensors`.
    fn loadArchTensors(self: *Model, ml: *mod.loader.Loader, exc: *mod.impl.Exception) (mod.impl.Thrown || error{Unported})!void {
        const entry = mod.models.dispatch.forArch(self.arch) orelse return error.Unported;
        try entry.loadArchTensors(self, ml, exc);
    }

    /// Ports `llama_model`'s `n_gpu_layers` (llama-model.cpp:2548 @b1b770d90):
    /// plus one for the output layer when all are asked for.
    pub fn nGpuLayers(self: *const Model) u32 {
        return if (self.params.n_gpu_layers >= 0) @intCast(self.params.n_gpu_layers) else self.hparams.n_layer_all + 1;
    }

    /// Ports `llama_model`'s `n_devices` (llama-model.cpp:2540 @b1b770d90).
    pub fn nDevices(self: *const Model) usize {
        return self.devices.items.len;
    }

    /// Ports `llama_model`'s `n_tensors` (llama-model.cpp:2536 @b1b770d90).
    pub fn nTensors(self: *const Model) usize {
        return self.tensors_by_name.items.len;
    }

    /// Ports `llama_model`'s `dev_layer` (llama-model.cpp:2809 @b1b770d90).
    pub fn devLayer(self: *const Model, il: usize) mod.impl.llama_c.ggml_backend_dev_t {
        return self.dev_layer.items[il].dev;
    }

    /// Ports `llama_model`'s `dev_output` (llama-model.cpp:2813 @b1b770d90).
    pub fn devOutput(self: *const Model) mod.impl.llama_c.ggml_backend_dev_t {
        return self.dev_output.dev;
    }

    /// Ports `llama_model`'s `select_buft` (llama-model.cpp:2857 @b1b770d90),
    /// through `select_buft` and `buft_supported` (llama-model.cpp:2845,
    /// 2818 @b1b770d90): the first of layer `il`'s buffer types whose
    /// device can add two `n_embd` vectors held in it.
    ///
    /// Return: the buffer type, or `error.Thrown` when none can.
    pub fn selectBuft(self: *const Model, il: usize, exc: *mod.impl.Exception) mod.impl.Thrown!mod.impl.llama_c.ggml_backend_buffer_type_t {
        for (self.dev_layer.items[il].buft_list.?.items) |cur| {
            const ctx = mod.impl.llama_c.ggml_init(.{ .mem_size = mod.impl.llama_c.ggml_tensor_overhead() * 8, .mem_buffer = null, .no_alloc = true }) orelse
                return exc.throw(.runtime_error, "failed to create ggml context", .{});
            defer mod.impl.llama_c.ggml_free(ctx);
            const buf = mod.impl.llama_c.ggml_backend_buft_alloc_buffer(cur.buft, 0);
            defer mod.impl.llama_c.ggml_backend_buffer_free(buf);
            const a = mod.impl.llama_c.ggml_new_tensor_1d(ctx, mod.impl.llama_c.GGML_TYPE_F32, self.hparams.n_embd);
            const layer_dir = mod.impl.llama_c.ggml_new_tensor_1d(ctx, mod.impl.llama_c.GGML_TYPE_F32, self.hparams.n_embd);
            const op_tensor = mod.impl.llama_c.ggml_add(ctx, a, layer_dir);
            for (op_tensor.*.src) |src| {
                if (src) |s| s.*.buffer = buf;
            }
            if (mod.impl.llama_c.ggml_backend_dev_supports_op(cur.dev, op_tensor)) return cur.buft;
        }
        return exc.throw(.runtime_error, "no suitable buffer type found", .{});
    }

    /// Ports `llama_model`'s `has_tensor_overrides` (llama-model.cpp:2867 @b1b770d90).
    /// Points `params.tensor_split` at this model's own copy, after the
    /// model has moved to where it will stay. Measured: without it the
    /// pointer was left in `create`'s stack frame, and under
    /// `LLAMA_SPLIT_MODE_TENSOR` with an uneven split the KV cache was sized
    /// from whatever had since been written there.
    ///
    /// Return: nothing.
    pub fn ownTensorSplit(self: *Model) void {
        if (self.params.tensor_split != null) self.params.tensor_split = &self.tensor_split_owned;
    }

    pub fn hasTensorOverrides(self: *const Model) bool {
        return self.has_tensor_overrides;
    }

    /// Ports `llama_model_has_encoder` (llama-model.cpp:3957 @b1b770d90).
    pub fn hasEncoder(self: *const Model) bool {
        return switch (self.arch) {
            .t5, .t5encoder, .eagle3, .dflash => true,
            else => false,
        };
    }

    /// Ports `llama_model_has_decoder` (llama-model.cpp:3967 @b1b770d90).
    pub fn hasDecoder(self: *const Model) bool {
        return self.arch != .t5encoder;
    }

    /// Ports `llama_model`'s `get_tensor` (llama-model.cpp:2871 @b1b770d90).
    ///
    /// Return: the first tensor with that name, or null.
    pub fn getTensor(self: *const Model, name: []const u8) ?*mod.impl.llama_c.ggml_tensor {
        for (self.tensors_by_name.items) |t| {
            if (std.mem.eql(u8, mod.gguf_meta.cString(mod.impl.llama_c.ggml_get_name(t)), name)) return t;
        }
        return null;
    }

    /// Ports `llama_model`'s `no_alloc` (llama-model.cpp:2557 @b1b770d90).
    pub fn noAlloc(self: *const Model) bool {
        return self.params.no_alloc;
    }

    /// Ports `llama_model`'s `get_rope_params` (llama-model.cpp:2883
    /// @b1b770d90): the RoPE of layer `il`, read by both the model graph and
    /// the K-shift. This is the base class's body; the virtual call is
    /// `getRopeParams`, which an architecture's override may replace.
    pub fn getRopeParamsBase(self: *const Model, cparams: *const mod.cparams.Cparams, il: u32) RopeParams {
        const hp = &self.hparams;
        return .{
            .n_rot = if (hp.hasRope(il)) @bitCast(hp.nRot(il)) else 0,
            .n_offs = if (hp.n_lora_kv > 0) @bitCast(hp.nEmbdHeadK(il) -% hp.nRot(il)) else 0,
            .type = hp.rope_type,
            .n_ctx_orig = @bitCast(cparams.n_ctx_orig_yarn),
            .freq_base = self.getRopeFreqBase(cparams, il),
            .freq_scale = self.getRopeFreqScale(cparams, il),
            .ext_factor = cparams.yarn_ext_factor,
            .attn_factor = cparams.yarn_attn_factor,
            .beta_fast = cparams.yarn_beta_fast,
            .beta_slow = cparams.yarn_beta_slow,
            .factors = self.getRopeFactors(cparams, il),
        };
    }

    /// Ports `llama_model`'s `get_rope_params_indexer` (llama-model.cpp:2901
    /// @b1b770d90): the same for the keys of the lightning indexer -- NeoX,
    /// unscaled, no offset and no factors, starting from the virtual
    /// `get_rope_params`. The base class's body; the virtual call is
    /// `getRopeParamsIndexer`.
    pub fn getRopeParamsIndexerBase(self: *const Model, cparams: *const mod.cparams.Cparams, il: u32) RopeParams {
        var res = self.getRopeParams(cparams, il);

        res.n_offs = 0;
        res.type = mod.impl.llama_c.LLAMA_ROPE_TYPE_NEOX;
        res.freq_base = cparams.rope_freq_base;
        res.freq_scale = cparams.rope_freq_scale;
        res.factors = null;

        return res;
    }

    /// The virtual call of `get_rope_params` (llama-model.h:994 @b1b770d90):
    /// the architecture's override when it has one, else `getRopeParamsBase`.
    pub fn getRopeParams(self: *const Model, cparams: *const mod.cparams.Cparams, il: u32) RopeParams {
        if (mod.models.dispatch.forArch(self.arch)) |e| if (e.getRopeParams) |f| return f(self, cparams, il);
        return self.getRopeParamsBase(cparams, il);
    }

    /// The virtual call of `get_rope_params_indexer` (llama-model.h:997
    /// @b1b770d90): the architecture's override when it has one, else
    /// `getRopeParamsIndexerBase`.
    pub fn getRopeParamsIndexer(self: *const Model, cparams: *const mod.cparams.Cparams, il: u32) RopeParams {
        if (mod.models.dispatch.forArch(self.arch)) |e| if (e.getRopeParamsIndexer) |f| return f(self, cparams, il);
        return self.getRopeParamsIndexerBase(cparams, il);
    }

    /// Ports `llama_model`'s `get_rope_freq_base` (llama-model.cpp:2913 @b1b770d90).
    pub fn getRopeFreqBase(self: *const Model, cparams: *const mod.cparams.Cparams, il: u32) f32 {
        return if (self.hparams.isSwa(il)) self.hparams.rope_freq_base_train_swa else cparams.rope_freq_base;
    }

    /// Ports `llama_model`'s `get_rope_freq_scale` (llama-model.cpp:2917 @b1b770d90).
    pub fn getRopeFreqScale(self: *const Model, cparams: *const mod.cparams.Cparams, il: u32) f32 {
        return if (self.hparams.isSwa(il)) self.hparams.rope_freq_scale_train_swa else cparams.rope_freq_scale;
    }

    /// Ports `llama_model`'s `get_rope_factors` (llama-model.cpp:2921
    /// @b1b770d90): long or short frequency factors by the context size.
    pub fn getRopeFactors(self: *const Model, cparams: *const mod.cparams.Cparams, il: u32) ?*mod.impl.llama_c.ggml_tensor {
        const layer = &self.layers[il];
        if (layer.rope_freqs != null) return layer.rope_freqs;
        if (cparams.n_ctx_seq > self.hparams.n_ctx_orig_yarn) return layer.rope_long;
        return layer.rope_short;
    }

    /// Ports `llama_model`'s `create_memory` (llama-model.cpp:2936 @b1b770d90),
    /// its default arm -- a recurrent memory, a hybrid one, or a KV cache --
    /// the encoders' arm, MiniMax-M3's MSA cache, the DSA and DSA-iSWA arms
    /// of GLM-DSA, DeepSeek V3.2, HY-V4 and dots.llm3, and DeepSeek V4's DSV4
    /// cache.
    ///
    /// Return: the memory, or null for a model without one; `error.Thrown`.
    pub fn createMemory(self: *const Model, params: *const mod.memory.Params, cparams: *const mod.cparams.Cparams, exc: *mod.impl.Exception) mod.impl.Thrown!?mod.memory.Memory {
        const hp = &self.hparams;
        const a = self.arch;
        // Dense MTP heads use a plain attention KV cache.
        const mtp_on_hybrid_qwen = params.ctx_type == mod.impl.llama_c.LLAMA_CONTEXT_TYPE_MTP and (a == .qwen3next or a == .qwen35 or a == .qwen35moe or a == .bailingmoe3);
        const mtp_on_hybrid_nemotron = params.ctx_type == mod.impl.llama_c.LLAMA_CONTEXT_TYPE_MTP and a == .nemotron_h_moe;
        const swa_type: mod.hparams.SwaType = @fromBackingInt(@intCast(hp.swa_type));

        // The non-causal LFM2 and LFM3 decision graphs read the whole prompt
        // in one batch; nothing is kept.
        if (a == .lfm2 and !hp.causal_attn and hp.n_layer_decision > 0) {
            return null;
        }
        if (a == .lfm3 and !hp.causal_attn) {
            return null;
        }

        // The encoders, embedding models and diffusion models attend within
        // the batch and keep nothing between decodes.
        switch (a) {
            .bert, .jina_bert_v2, .jina_bert_v3, .nomic_bert, .nomic_bert_moe, .neo_bert, .eurobert, .wavtokenizer_dec, .modern_bert, .gemma_embedding, .gemma_embedding2, .dream, .llada, .llada_moe, .rnd1, .clef => return null,
            .minimax_m3 => {
                // The sparse (MSA) layers carry an indexer key cache; the
                // leading dense layers do not.
                const filter_idx: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterMsaIdx };

                const m = try mod.kv_cache_msa.KvCacheMsa.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, hp.n_swa, swa_type, .{}, filter_idx, .{}, exc);
                return m.memoryOf();
            },
            .glm_dsa, .deepseek32 => {
                if (params.ctx_type == mod.impl.llama_c.LLAMA_CONTEXT_TYPE_MTP and hp.n_layer_nextn > 0) {
                    // The NextN/MTP draft head runs dense MLA, without the
                    // DSA indexer, so the MTP context has a plain attention
                    // KV cache of the nextn layers alone -- as the hybrid
                    // Qwen3.5's MTP context does.
                    const filter: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterNextn };
                    const m = try mod.kv_cache.KvCache.create(self, params.no_alloc, hp, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, hp.n_swa, swa_type, null, filter, .{}, .{}, exc);
                    return m.memoryOf();
                }
                // The main context: a DSA cache for the trunk layers alone,
                // which never attend the nextn layers.
                var filter_mla: mod.memory.LayerFilter = .{};
                if (hp.n_layer_nextn > 0) filter_mla = .{ .ctx = hp, .func = filterTrunk };
                const filter_lid: mod.memory.LayerFilter = .{ .ctx = self, .func = filterLidDsa };
                const m = try mod.kv_cache_dsa.KvCacheDsa.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, hp.n_swa, swa_type, filter_mla, filter_lid, .{}, exc);
                return m.memoryOf();
            },
            // `LLM_ARCH_GLM5_NEXT` (llama-model.cpp:3070 @b1b770d90): the KDA
            // layers are recurrent, the DSA layers a K-only MLA cache and an
            // indexer cache; the trunk graph never attends the NextN block.
            .glm5_next => {
                var filter_attn: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterTrunkAttn };
                var filter_idx: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterTrunkAttnIndexerFull };
                var filter_recr: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterTrunkRecr };

                // The draft head is a single DSA layer.
                if (params.ctx_type == mod.impl.llama_c.LLAMA_CONTEXT_TYPE_MTP) {
                    if (hp.n_layer_nextn == 0) {
                        return exc.throw(.runtime_error, "GLM5-Next MTP requires the NextN block, convert without --no-mtp", .{});
                    }
                    filter_attn = .{ .ctx = hp, .func = filterNextn };
                    filter_idx = .{ .ctx = hp, .func = filterNextn };
                    filter_recr = .{ .ctx = hp, .func = filterNone };
                }

                const m = try mod.memory_hybrid_idx.MemoryHybridIdx.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.n_ctx_seq, hp.n_swa, swa_type, mod.impl.llama_c.GGML_TYPE_F32, mod.impl.llama_c.GGML_TYPE_F32, @max(1, cparams.n_seq_max), cparams.n_seq_max, cparams.n_rs_seq, cparams.offload_kqv, cparams.kv_unified, filter_attn, filter_recr, filter_idx, exc);
                return m.memoryOf();
            },
            // `LLM_ARCH_HY_V4` (llama-model.cpp:3114 @b1b770d90).
            .hy_v4 => {
                if (hp.indexer_top_k == 0) {
                    // A full-attention checkpoint: no indexer, so no
                    // indexer key cache.
                    const m = try mod.kv_cache.KvCache.create(self, params.no_alloc, hp, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, hp.n_swa, swa_type, null, .{}, .{}, .{}, exc);
                    return m.memoryOf();
                }
                // Only "full" layers own an indexer, so the shared layers
                // need no indexer cache.
                const filter_lid: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterIndexerFull };
                const m = try mod.kv_cache_dsa.KvCacheDsa.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, hp.n_swa, swa_type, .{}, filter_lid, .{}, exc);
                return m.memoryOf();
            },
            // `LLM_ARCH_MISTRAL_LARGE4` (llama-model.cpp:3044 @b1b770d90):
            // only "full" layers own an indexer, so the shared layers need
            // no indexer cache, and layers without their own KV read the
            // MLA cache of their source layer.
            .mistral_large4 => {
                const filter_lid: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterIndexerFull };
                const reuse: mod.memory.LayerMap = .{ .ctx = hp, .func = reuseKvSource };
                const m = try mod.kv_cache_dsa.KvCacheDsa.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, hp.n_swa, swa_type, .{}, filter_lid, reuse, exc);
                return m.memoryOf();
            },
            .dots3note => {
                mod.impl.assert(swa_type != .none, "llama-model.cpp", 3158, "hparams.swa_type != LLAMA_SWA_TYPE_NONE");

                if (params.ctx_type == mod.impl.llama_c.LLAMA_CONTEXT_TYPE_MTP and hp.n_layer_nextn > 0) {
                    // The MTP draft context: a plain attention KV cache of
                    // the nextn layer alone.
                    const filter: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterNextn };
                    const m = try mod.kv_cache.KvCache.create(self, params.no_alloc, hp, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, hp.n_swa, swa_type, null, filter, .{}, .{}, exc);
                    return m.memoryOf();
                }
                // The main context: a DSA cache for the trunk's
                // full-attention layers and a window-sized SWA cache.
                var filter_mla: mod.memory.LayerFilter = .{};
                if (hp.n_layer_nextn > 0) filter_mla = .{ .ctx = hp, .func = filterTrunk };
                const filter_lid: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterLidFull };
                const m = try mod.kv_cache_dsa_iswa.KvCacheDsaIswa.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, params.swa_full, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, cparams.n_ubatch, filter_mla, filter_lid, .{}, exc);
                return m.memoryOf();
            },
            // `LLM_ARCH_DEEPSEEK4` (llama-model.cpp:3208 @b1b770d90): an MTP draft
            // context holds only the nextn layers, in an iSWA cache its graph
            // reads K-only; the main context the DSV4 cache.
            .deepseek41, .deepseek4 => {
                mod.impl.assert(swa_type != .none, "llama-model.cpp", 3210, "hparams.swa_type != LLAMA_SWA_TYPE_NONE");

                if (params.ctx_type == mod.impl.llama_c.LLAMA_CONTEXT_TYPE_MTP) {
                    const filter_mtp: mod.memory.LayerFilter = .{ .ctx = hp, .func = filterNextn };
                    const m = try mod.kv_cache_iswa.KvCacheIswa.create(self, params.no_alloc, null, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, params.swa_full, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, cparams.n_ubatch, null, filter_mtp, .{}, .{}, exc);
                    return m.memoryOf();
                }
                const m = try mod.kv_cache_dsv4.KvCacheDsv4.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, params.swa_full, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, cparams.n_ubatch, cparams.n_rs_seq, .{}, .{}, exc);
                return m.memoryOf();
            },
            // DeepSeek-V4's and Mistral Large 4's DSpark stages keep one
            // MLA-style K per position, their window the draft ring; any
            // other DFlash falls through to the default arm.
            .dflash => if (hp.dsv4_hc_mult > 0 or hp.n_lora_kv > 0) {
                mod.impl.assert(swa_type != .none, "llama-model.cpp", 3255, "hparams.swa_type != LLAMA_SWA_TYPE_NONE");
                const m = try mod.kv_cache_iswa.KvCacheIswa.create(self, params.no_alloc, null, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, params.swa_full, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, cparams.n_ubatch, null, .{}, .{}, .{}, exc);
                return m.memoryOf();
            },
            else => {},
        }

        if (mod.arch.isRecurrent(a)) {
            const m = try mod.memory_recurrent.MemoryRecurrent.create(self, params.no_alloc, mod.impl.llama_c.GGML_TYPE_F32, mod.impl.llama_c.GGML_TYPE_F32, cparams.offload_kqv, @max(1, cparams.n_seq_max), cparams.n_seq_max, cparams.n_rs_seq, .{}, exc);
            return m.memoryOf();
        }
        if (mod.arch.isHybrid(a) and !mtp_on_hybrid_qwen and !mtp_on_hybrid_nemotron) {
            // Hybrid architectures differ mainly in their layer filters.
            var filter_attn: mod.memory.LayerFilter = .{};
            var filter_recr: mod.memory.LayerFilter = .{};
            // only the sparse-attention architectures use llama_memory_hybrid_idx
            // a null filter_idx means the GGUF has no indexer tensors
            var filter_idx: mod.memory.LayerFilter = .{};
            const needs_mem_idx = a == .qwen4exp or a == .kimi_k31;
            if (a == .kimi_k31) {
                // DSA layers keep a K-only MLA cache plus a per-token indexer
                // key cache, the KDA layers stay recurrent.
                filter_attn = .{ .ctx = hp, .func = filterTrunkAttn };
                filter_recr = .{ .ctx = hp, .func = filterTrunkRecr };
                filter_idx = .{ .ctx = hp, .func = filterTrunkAttnIndexerFull };
            } else if (a == .falcon_h1) {
                filter_attn = .{ .ctx = hp, .func = filterAll };
                filter_recr = .{ .ctx = hp, .func = filterAll };
            } else if (a == .nemotron_h or a == .nemotron_h_moe) {
                filter_attn = .{ .ctx = hp, .func = filterNemotronAttn };
                filter_recr = .{ .ctx = hp, .func = filterNemotronRecr };
            } else if (a == .qwen3next or a == .qwen35 or a == .qwen35moe or a == .clover_moe or a == .qwen4exp or a == .minimax_01) {
                filter_attn = .{ .ctx = hp, .func = filterTrunkAttn };
                filter_recr = .{ .ctx = hp, .func = filterTrunkRecr };

                if (a == .qwen4exp and hp.indexer_head_size > 0) {
                    // QSA runs on the dense-attention layers only
                    filter_idx = .{ .ctx = hp, .func = filterTrunkAttn };
                }

                // The MTP draft context holds the MTP block alone: its
                // attention and indexer, no recurrent layer.
                if (a == .qwen4exp and params.ctx_type == mod.impl.llama_c.LLAMA_CONTEXT_TYPE_MTP) {
                    filter_attn = .{ .ctx = hp, .func = filterNextn };
                    filter_recr = .{ .ctx = hp, .func = filterNone };
                    if (filter_idx.func != null) filter_idx = .{ .ctx = hp, .func = filterNextn };
                }
            }
            if (swa_type != .none) {
                // Hybrid models with SWA use the hybrid iSWA memory.
                const m = try mod.memory_hybrid_iswa.MemoryHybridIswa.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, params.swa_full, cparams.n_ctx_seq, cparams.n_ubatch, mod.impl.llama_c.GGML_TYPE_F32, mod.impl.llama_c.GGML_TYPE_F32, @max(1, cparams.n_seq_max), cparams.n_seq_max, cparams.n_rs_seq, cparams.offload_kqv, cparams.kv_unified, filter_attn, filter_recr, exc);
                return m.memoryOf();
            }
            if (needs_mem_idx) {
                // sparse attention over a per-token indexer cache, in its own memory type
                const m = try mod.memory_hybrid_idx.MemoryHybridIdx.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.n_ctx_seq, hp.n_swa, swa_type, mod.impl.llama_c.GGML_TYPE_F32, mod.impl.llama_c.GGML_TYPE_F32, @max(1, cparams.n_seq_max), cparams.n_seq_max, cparams.n_rs_seq, cparams.offload_kqv, cparams.kv_unified, filter_attn, filter_recr, filter_idx, exc);
                return m.memoryOf();
            }
            const m = try mod.memory_hybrid.MemoryHybrid.create(self, params.no_alloc, params.type_k, params.type_v, !cparams.flash_attn, cparams.n_ctx_seq, hp.n_swa, swa_type, mod.impl.llama_c.GGML_TYPE_F32, mod.impl.llama_c.GGML_TYPE_F32, @max(1, cparams.n_seq_max), cparams.n_seq_max, cparams.n_rs_seq, cparams.offload_kqv, cparams.kv_unified, filter_attn, filter_recr, exc);
            return m.memoryOf();
        }

        var filter: mod.memory.LayerFilter = .{};
        var reuse: mod.memory.LayerMap = .{};
        var share: mod.memory.LayerMap = .{};

        if (a == .gemma3n or a == .gemma4) reuse = .{ .ctx = hp, .func = reuseKvSource };

        if (mtp_on_hybrid_qwen or mtp_on_hybrid_nemotron) filter = .{ .ctx = hp, .func = filterNextn };
        // No filter when `n_layer_nextn` is repurposed for a router layer the
        // trunk attends, or when a model is entirely nextn layers and has no
        // trunk.
        if (hp.n_layer_nextn > 0 and hp.nLayer() > 0 and hp.router_layer < 0) {
            filter = if (params.ctx_type == mod.impl.llama_c.LLAMA_CONTEXT_TYPE_MTP) .{ .ctx = hp, .func = filterNextn } else .{ .ctx = hp, .func = filterTrunk };
        }
        if (swa_type != .none) {
            mod.impl.assert(hp.isSwaAny(), "llama-model.cpp", 3444, "hparams.is_swa_any()");

            if (a == .gemma4_assistant) {
                // The assistant reads the target model's last full and last
                // SWA layers.
                const other: *const mod.context.Context = @ptrCast(@alignCast(cparams.ctx_other.?));
                const mem_other = other.getMemory();
                const share_ctx: ShareFromOther = .{ .hparams = hp, .n_layer_other = @intCast(other.getModel().hparams.nLayer()) };
                share = .{ .ctx = &share_ctx, .func = ShareFromOther.call };
                const m = try mod.kv_cache_iswa.KvCacheIswa.create(self, params.no_alloc, null, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, params.swa_full, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, cparams.n_ubatch, mem_other, filter, reuse, share, exc);
                return m.memoryOf();
            }
            const m = try mod.kv_cache_iswa.KvCacheIswa.create(self, params.no_alloc, null, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, params.swa_full, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, cparams.n_ubatch, null, filter, reuse, share, exc);
            return m.memoryOf();
        }
        mod.impl.assert(!hp.isSwaAny(), "llama-model.cpp", 3494, "!hparams.is_swa_any()");
        const mem_other: ?*mod.kv_cache.KvCache = null;
        const m = try mod.kv_cache.KvCache.create(self, params.no_alloc, hp, params.type_k, params.type_v, !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified, cparams.n_ctx_seq, cparams.n_seq_max, hp.n_swa, swa_type, mem_other, filter, reuse, share, exc);
        return m.memoryOf();
    }

    /// Ports `llama_model`'s `build_graph` (llama-model.cpp:3521 @b1b770d90):
    /// the architecture's graph, then pooling, the LM head, backend sampling,
    /// the dense projections, and the device copies of the layer inputs and
    /// of the MTP hidden state.
    ///
    /// Return: the graph, owned by `params.res`.
    pub fn buildGraph(self: *const Model, params: *const mod.graph.Params) *mod.impl.llama_c.ggml_cgraph {
        const b = mod.graph.Builder.init(params);
        const entry = mod.models.dispatch.forArch(self.arch) orelse mod.impl.abort("llama-model.cpp", 3522, "build_arch_graph: architecture not ported");
        entry.buildArchGraph(self, &b, params);
        b.buildPooling(self.cls, self.cls_b, self.cls_out, self.cls_out_b, self.cls_norm);
        // the LM head, if the context outputs logits
        b.buildLogits();
        b.buildSampling();
        // Models converted with --sentence-transformers-dense-modules
        // project the pooled output twice more.
        b.buildDenseOut(self.dense_2_out_layers, self.dense_2_out_layers_b, self.dense_3_out_layers);
        b.buildLayerInpDev();
        b.buildHNextnDev();
        b.res.setOutputs(params);
        return b.res.getGf();
    }

    /// Ports `llama_model`'s `memory_breakdown` (llama-model.cpp:2561
    /// @b1b770d90): bytes per buffer type, in the `std`'s `map`'s pointer
    /// order.
    ///
    /// Return: the entries, the caller's to free with `impl.allocator`.
    pub fn memoryBreakdown(self: *const Model) []BuftSize {
        var ret: std.ArrayList(BuftSize) = .empty;
        for (self.ctxs_bufs.items) |cb| {
            if (self.noAlloc()) {
                mod.impl.assert(cb.bufs.items.len == 1, "llama-model.cpp", 2565, "bufs.size() == 1");
                const buf = cb.bufs.items[0];
                mod.impl.assert(mod.impl.llama_c.ggml_backend_buffer_get_base(buf) == null, "llama-model.cpp", 2567, "ggml_backend_buffer_get_base(buf) == nullptr");
                const buft = mod.impl.llama_c.ggml_backend_buffer_get_type(buf);
                addBuftSize(&ret, buft, mod.impl.llama_c.ggml_backend_alloc_ctx_tensors_from_buft_size(cb.ctx, buft));
            } else {
                for (cb.bufs.items) |buf| addBuftSize(&ret, mod.impl.llama_c.ggml_backend_buffer_get_type(buf), mod.impl.llama_c.ggml_backend_buffer_get_size(buf));
            }
        }
        return ret.toOwnedSlice(std.heap.c_allocator) catch oom();
    }
};

/// `uint32_t % uint32_t` as aarch64 computes it: `udiv` then `msub`, and
/// `udiv` by zero gives zero, so `a % 0 == a`. The C++ is undefined there;
/// this is what the reference binary does.
pub fn remU32(a: u32, b: u32) u32 {
    return if (b == 0) a else a % b;
}

fn deepstackAt(hp: *const mod.hparams.Hparams, il: u32) u32 {
    return @bitCast(hp.deepstack_mapping_arr[il]);
}

/// `print_f` in `print_info`: one value for every layer, or all of them in
/// brackets when they differ. The lambda returns `int32_t`, so a value past
/// `INT32_MAX` prints negative.
fn printF(buf: *std.ArrayList(u8), hp: *const mod.hparams.Hparams, comptime get: fn (*const mod.hparams.Hparams, u32) u32, n: u32) [*:0]const u8 {
    buf.clearRetainingCapacity();
    var is_var = false;
    const v0: i32 = @bitCast(get(hp, 0));
    var il: u32 = 0;
    while (il < n) : (il += 1) if (@as(i32, @bitCast(get(hp, il))) != v0) {
        is_var = true;
    };
    if (is_var) {
        buf.append(std.heap.c_allocator, '[') catch oom();
        il = 0;
        while (il < n) : (il += 1) {
            buf.print(std.heap.c_allocator, "{d}", .{@as(i32, @bitCast(get(hp, il)))}) catch oom();
            if (il < n - 1) buf.appendSlice(std.heap.c_allocator, ", ") catch oom();
        }
        buf.append(std.heap.c_allocator, ']') catch oom();
    } else {
        buf.print(std.heap.c_allocator, "{d}", .{v0}) catch oom();
    }
    buf.append(std.heap.c_allocator, 0) catch oom();
    return @ptrCast(buf.items.ptr);
}

// -----------------------------------------------------------------------------
// Memory layer filters, the lambdas of `create_memory`

fn hpOf(ctx: ?*const anyopaque) *const mod.hparams.Hparams {
    return @ptrCast(@alignCast(ctx.?));
}
fn filterAll(_: ?*const anyopaque, _: i32) bool {
    return true;
}
fn filterNemotronAttn(ctx: ?*const anyopaque, il: i32) bool {
    const hp = hpOf(ctx);
    return !hp.isRecr(@intCast(il)) and hp.nFf(@intCast(il)) == 0;
}
fn filterNemotronRecr(ctx: ?*const anyopaque, il: i32) bool {
    const hp = hpOf(ctx);
    return hp.isRecr(@intCast(il)) and hp.nFf(@intCast(il)) == 0;
}
fn filterTrunkAttn(ctx: ?*const anyopaque, il: i32) bool {
    const hp = hpOf(ctx);
    return @as(u32, @intCast(il)) < hp.nLayer() and !hp.isRecr(@intCast(il));
}
fn filterTrunkRecr(ctx: ?*const anyopaque, il: i32) bool {
    const hp = hpOf(ctx);
    return @as(u32, @intCast(il)) < hp.nLayer() and hp.isRecr(@intCast(il));
}
fn filterNone(_: ?*const anyopaque, _: i32) bool {
    return false;
}
/// `create_memory`'s `filter_idx` for GLM5-Next (llama-model.cpp:3077
/// @b1b770d90) and Kimi-K3.1 (llama-model.cpp:3319 @b1b770d90): the
/// trunk's attention layers with a full indexer.
fn filterTrunkAttnIndexerFull(ctx: ?*const anyopaque, il: i32) bool {
    const hp = hpOf(ctx);
    const l: u32 = @intCast(il);
    return l < hp.nLayer() and !hp.isRecr(l) and hp.isIndexerFull(l);
}
fn filterNextn(ctx: ?*const anyopaque, il: i32) bool {
    return @as(u32, @intCast(il)) >= hpOf(ctx).nLayer();
}
fn filterTrunk(ctx: ?*const anyopaque, il: i32) bool {
    return @as(u32, @intCast(il)) < hpOf(ctx).nLayer();
}
/// `create_memory`'s `filter_idx` for MiniMax-M3 (llama-model.cpp:2972
/// @b1b770d90): the layers at or past the leading dense ones.
fn filterMsaIdx(ctx: ?*const anyopaque, il: i32) bool {
    return @as(u32, @bitCast(il)) >= hpOf(ctx).n_layer_dense_lead;
}

/// `create_memory`'s `filter_lid` for GLM-DSA and DeepSeek V3.2
/// (llama-model.cpp:3025 @b1b770d90): the trunk layers, and for GLM-DSA
/// only those with a full indexer.
fn filterLidDsa(ctx: ?*const anyopaque, il_: i32) bool {
    const model: *const Model = @ptrCast(@alignCast(ctx.?));
    const il: u32 = @bitCast(il_);
    return il < model.hparams.nLayer() and (model.arch != .glm_dsa or model.hparams.isIndexerFull(il));
}

/// `create_memory`'s `filter_lid` for HY-V4 (llama-model.cpp:3137
/// @b1b770d90): the layers with a full indexer. Unguarded, as the C++'s
/// lambda is: a layer past `n_layer` aborts in `is_indexer_full`.
fn filterIndexerFull(ctx: ?*const anyopaque, il_: i32) bool {
    return hpOf(ctx).isIndexerFull(@bitCast(il_));
}

/// `create_memory`'s `filter_lid` for dots.llm3 (llama-model.cpp:3188
/// @b1b770d90): the trunk layers with a full indexer.
fn filterLidFull(ctx: ?*const anyopaque, il_: i32) bool {
    const hp = hpOf(ctx);
    const il: u32 = @bitCast(il_);
    return il < hp.nLayer() and hp.isIndexerFull(il);
}

/// `create_memory`'s `reuse` for Gemma 3n and Gemma 4 (llama-model.cpp:3424
/// @b1b770d90) and Mistral Large 4 (llama-model.cpp:3050 @b1b770d90): a
/// layer without its own KV reads the cells of its source layer.
fn reuseKvSource(ctx: ?*const anyopaque, il_: i32) i32 {
    const hp = hpOf(ctx);
    const il: u32 = @bitCast(il_);
    return if (hp.hasKv(il)) -1 else @bitCast(hp.kvSource(il));
}

/// `create_memory`'s `share` for the Gemma 4 assistant (llama-model.cpp:3449
/// @b1b770d90): its SWA layers read the target's last SWA layer, the others
/// its last layer. The C++ asks `llama_model_n_layer` of the target on every
/// call; the answer does not change, so it is taken once.
const ShareFromOther = struct {
    hparams: *const mod.hparams.Hparams,
    n_layer_other: i32,

    fn call(ctx: ?*const anyopaque, il: i32) i32 {
        const self: *const ShareFromOther = @ptrCast(@alignCast(ctx.?));
        if (self.hparams.isSwa(@bitCast(il))) return self.n_layer_other - 2;
        return self.n_layer_other - 1;
    }
};

// -----------------------------------------------------------------------------
// Devices and buffer types

pub fn addBuftSize(list: *std.ArrayList(BuftSize), buft: mod.impl.llama_c.ggml_backend_buffer_type_t, size: usize) void {
    var pos: usize = 0;
    while (pos < list.items.len) : (pos += 1) {
        if (list.items[pos].buft == buft) {
            list.items[pos].size += size;
            return;
        }
        if (@intFromPtr(list.items[pos].buft) > @intFromPtr(buft)) break;
    }
    list.insert(std.heap.c_allocator, pos, .{ .buft = buft, .size = size }) catch oom();
}

/// Ports `llama_load_mode_name` (src/llama.cpp:50 @b1b770d90).
pub fn loadModeName(load_mode: mod.impl.llama_c.enum_llama_load_mode) [:0]const u8 {
    return switch (load_mode) {
        mod.impl.llama_c.LLAMA_LOAD_MODE_AUTO => "auto",
        mod.impl.llama_c.LLAMA_LOAD_MODE_NONE => "none",
        mod.impl.llama_c.LLAMA_LOAD_MODE_MMAP => "mmap",
        mod.impl.llama_c.LLAMA_LOAD_MODE_MLOCK => "mlock",
        mod.impl.llama_c.LLAMA_LOAD_MODE_MMAP_MLOCK => "mmap+mlock",
        mod.impl.llama_c.LLAMA_LOAD_MODE_DIRECT_IO => "dio",
        else => mod.impl.abort("llama.cpp", 65, "fatal error"),
    };
}

/// Ports `llama_supports_rpc` (src/llama.cpp:111 @b1b770d90).
pub fn supportsRpc() bool {
    if (mod.impl.llama_c.ggml_backend_reg_count() == 0) mod.impl.llama_c.ggml_backend_load_all();
    return mod.impl.llama_c.ggml_backend_reg_by_name("RPC") != null;
}

/// Ports `llama_supports_gpu_offload` (src/llama.cpp:102 @b1b770d90).
pub fn supportsGpuOffload() bool {
    if (mod.impl.llama_c.ggml_backend_reg_count() == 0) mod.impl.llama_c.ggml_backend_load_all();
    return mod.impl.llama_c.ggml_backend_dev_by_type(mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_GPU) != null or
        mod.impl.llama_c.ggml_backend_dev_by_type(mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_IGPU) != null or
        supportsRpc();
}

/// Ports `make_cpu_buft_list` (llama-model.cpp:1242 @b1b770d90): ACCEL,
/// then a GPU host buffer, then the CPU's extra buffer types, then the CPU.
fn makeCpuBuftList(list: *mod.loader.BuftList, devices: []const Device, use_extra_bufts: bool, no_host: bool, exc: *mod.impl.Exception) mod.impl.Thrown!void {
    // ACCEL buffer types.
    var i: usize = 0;
    while (i < mod.impl.llama_c.ggml_backend_dev_count()) : (i += 1) {
        const dev = mod.impl.llama_c.ggml_backend_dev_get(i);
        if (mod.impl.llama_c.ggml_backend_dev_type(dev) == mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            const buft = mod.impl.llama_c.ggml_backend_dev_buffer_type(dev);
            if (buft != mod.impl.llama_c.ggml_backend_cpu_buffer_type()) list.append(std.heap.c_allocator, .{ .dev = dev, .buft = buft }) catch oom();
        }
    }

    // A host buffer type, from the first device that has one: it cuts
    // transfer time when large batches are offloaded to a GPU.
    if (!no_host) {
        for (devices) |dev| {
            if (mod.impl.llama_c.ggml_backend_dev_host_buffer_type(dev.dev)) |buft| {
                list.append(std.heap.c_allocator, .{ .dev = dev.dev, .buft = buft }) catch oom();
                break;
            }
        }
    }

    // Extra buffer types.
    if (use_extra_bufts) {
        const cpu_dev = mod.impl.llama_c.ggml_backend_dev_by_type(mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU) orelse
            return exc.throw(.runtime_error, "make_cpu_buft_list: no CPU backend found", .{});
        const cpu_reg = mod.impl.llama_c.ggml_backend_dev_backend_reg(cpu_dev);
        const get_extra: ?*const fn (mod.impl.llama_c.ggml_backend_dev_t) callconv(.c) [*c]mod.impl.llama_c.ggml_backend_buffer_type_t =
            @ptrCast(@alignCast(mod.impl.llama_c.ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts")));
        if (get_extra) |f| {
            var extra = f(cpu_dev);
            while (extra != null and extra[0] != null) : (extra += 1) list.append(std.heap.c_allocator, .{ .dev = cpu_dev, .buft = extra[0] }) catch oom();
        }
    }

    // The CPU buffer type.
    i = 0;
    while (i < mod.impl.llama_c.ggml_backend_dev_count()) : (i += 1) {
        const dev = mod.impl.llama_c.ggml_backend_dev_get(i);
        if (mod.impl.llama_c.ggml_backend_dev_type(dev) == mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU) {
            list.append(std.heap.c_allocator, .{ .dev = dev, .buft = mod.impl.llama_c.ggml_backend_dev_buffer_type(dev) }) catch oom();
        }
    }
}

/// Ports `make_gpu_buft_list` (llama-model.cpp:1304 @b1b770d90): the
/// device's split buffer type when splitting by rows, its default buffer
/// type, then its extra ones.
fn makeGpuBuftList(list: *mod.loader.BuftList, dev: mod.impl.llama_c.ggml_backend_dev_t, split_mode: mod.impl.llama_c.enum_llama_split_mode, tensor_split: [*c]const f32, exc: *mod.impl.Exception) mod.impl.Thrown!void {
    if (split_mode == mod.impl.llama_c.LLAMA_SPLIT_MODE_ROW) {
        const reg = mod.impl.llama_c.ggml_backend_dev_backend_reg(dev);
        const split_fn: ?*const fn (c_int, [*c]const f32) callconv(.c) mod.impl.llama_c.ggml_backend_buffer_type_t =
            @ptrCast(@alignCast(mod.impl.llama_c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type")));
        if (split_fn) |f| {
            var dev_index: usize = 0;
            while (dev_index < mod.impl.llama_c.ggml_backend_reg_dev_count(reg)) : (dev_index += 1) {
                if (mod.impl.llama_c.ggml_backend_reg_dev_get(reg, dev_index) == dev) break;
            } else return exc.throw(.runtime_error, "device {s} not found in its backend reg", .{std.mem.span(mod.impl.llama_c.ggml_backend_dev_name(dev))});
            if (f(@intCast(dev_index), tensor_split)) |buft| list.append(std.heap.c_allocator, .{ .dev = dev, .buft = buft }) catch oom();
        } else {
            return exc.throw(.runtime_error, "device {s} does not support split buffers", .{std.mem.span(mod.impl.llama_c.ggml_backend_dev_name(dev))});
        }
    }

    // The device's default buffer type.
    list.append(std.heap.c_allocator, .{ .dev = dev, .buft = mod.impl.llama_c.ggml_backend_dev_buffer_type(dev) }) catch oom();

    // The device's extra buffer types, if any.
    const reg = mod.impl.llama_c.ggml_backend_dev_backend_reg(dev);
    if (reg != null) {
        const get_extra: ?*const fn (mod.impl.llama_c.ggml_backend_dev_t) callconv(.c) [*c]mod.impl.llama_c.ggml_backend_buffer_type_t =
            @ptrCast(@alignCast(mod.impl.llama_c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts")));
        if (get_extra) |f| {
            var extra = f(dev);
            while (extra != null and extra[0] != null) : (extra += 1) list.append(std.heap.c_allocator, .{ .dev = dev, .buft = extra[0] }) catch oom();
        }
    }
}

/// Ports `llama_prepare_model_devices` (src/llama.cpp:158 @b1b770d90): the
/// devices the model will use, given or chosen.
///
/// `LLAMA_SPLIT_MODE_TENSOR` builds one meta device over the given
/// devices, or over every device with its own memory, which slices each
/// tensor across them as `meta_split.getSplitState` says. That is the
/// multi-GPU path; a one-GPU machine reaches it only through a harness.
///
/// Return: false on a bad `main_gpu` or no device for tensor parallelism,
/// as the C++ does.
pub fn prepareDevices(params: *const mod.impl.llama_c.llama_model_params, model: *Model) bool {
    const func = "llama_prepare_model_devices";

    if (params.devices) |devs| {
        if (params.split_mode == mod.impl.llama_c.LLAMA_SPLIT_MODE_TENSOR) {
            var n_devs: usize = 0;
            while (devs[n_devs] != null) n_devs += 1;
            if (n_devs == 0) {
                mod.impl.logError("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", .{func});
                return false;
            }
            mod.impl.logInfo("%s: creating a Meta device with %zu devices\n", .{ func, n_devs });
            for (0..n_devs) |i| mod.impl.logInfo("%s: - device %zu: %s\n", .{ func, i, mod.impl.llama_c.ggml_backend_dev_name(devs[i]) });
            model.get_split_state_ud = .{ .n_devices = n_devs, .model = model };
            model.devices.append(std.heap.c_allocator, .{
                .is_meta = true,
                .dev = mod.impl.llama_c.ggml_backend_meta_device(devs, n_devs, mod.meta_split.getSplitState, &model.get_split_state_ud),
            }) catch oom();
        } else {
            var d = devs;
            while (d[0] != null) : (d += 1) model.devices.append(std.heap.c_allocator, .{ .is_meta = false, .dev = d[0] }) catch oom();
        }
    } else {
        // Default device selection.
        var gpus: std.ArrayList(Device) = .empty;
        defer gpus.deinit(std.heap.c_allocator);
        var igpus: std.ArrayList(Device) = .empty;
        defer igpus.deinit(std.heap.c_allocator);
        var rpc_servers: std.ArrayList(Device) = .empty;
        defer rpc_servers.deinit(std.heap.c_allocator);

        if (params.split_mode == mod.impl.llama_c.LLAMA_SPLIT_MODE_TENSOR) {
            var devs: std.ArrayList(mod.impl.llama_c.ggml_backend_dev_t) = .empty;
            defer devs.deinit(std.heap.c_allocator);
            devs.ensureTotalCapacity(std.heap.c_allocator, mod.impl.llama_c.ggml_backend_dev_count()) catch oom();
            var i: usize = 0;
            while (i < mod.impl.llama_c.ggml_backend_dev_count()) : (i += 1) {
                const dev = mod.impl.llama_c.ggml_backend_dev_get(i);
                if (mod.impl.llama_c.ggml_backend_dev_buffer_type(dev) == mod.impl.llama_c.ggml_backend_cpu_buffer_type()) {
                    mod.impl.logInfo("%s: skipping %s (%s) for tensor parallelism\n", .{ func, mod.impl.llama_c.ggml_backend_dev_name(dev), mod.impl.llama_c.ggml_backend_dev_description(dev) });
                    continue;
                }
                devs.append(std.heap.c_allocator, dev) catch oom();
            }
            if (devs.items.len == 0) {
                mod.impl.logError("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", .{func});
                return false;
            }
            mod.impl.logInfo("%s: creating a Meta device for tensor parallelism from %zu devices:\n", .{ func, devs.items.len });
            for (devs.items, 0..) |dev, k| {
                mod.impl.logInfo("%s: - device %zu: %s (%s)\n", .{ func, k, mod.impl.llama_c.ggml_backend_dev_name(dev), mod.impl.llama_c.ggml_backend_dev_description(dev) });
            }
            model.get_split_state_ud = .{ .n_devices = devs.items.len, .model = model };
            gpus.append(std.heap.c_allocator, .{
                .is_meta = true,
                .dev = mod.impl.llama_c.ggml_backend_meta_device(devs.items.ptr, devs.items.len, mod.meta_split.getSplitState, &model.get_split_state_ud),
            }) catch oom();
        } else {
            var i: usize = 0;
            while (i < mod.impl.llama_c.ggml_backend_dev_count()) : (i += 1) {
                const dev = mod.impl.llama_c.ggml_backend_dev_get(i);
                switch (mod.impl.llama_c.ggml_backend_dev_type(dev)) {
                    // CPU backends are handled separately.
                    mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU, mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_ACCEL => {},
                    mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_GPU => {
                        const reg = mod.impl.llama_c.ggml_backend_dev_backend_reg(dev);
                        if (std.mem.eql(u8, std.mem.span(mod.impl.llama_c.ggml_backend_reg_name(reg)), "RPC")) {
                            rpc_servers.append(std.heap.c_allocator, .{ .is_meta = false, .dev = dev }) catch oom();
                        } else {
                            // Skip a GPU already present under the same device id.
                            var props: mod.impl.llama_c.ggml_backend_dev_props = undefined;
                            mod.impl.llama_c.ggml_backend_dev_get_props(dev, &props);
                            var dup: ?Device = null;
                            for (gpus.items) |g| {
                                var d_props: mod.impl.llama_c.ggml_backend_dev_props = undefined;
                                mod.impl.llama_c.ggml_backend_dev_get_props(g.dev, &d_props);
                                if (props.device_id != null and d_props.device_id != null and std.mem.orderZ(u8, props.device_id, d_props.device_id) == .eq) {
                                    dup = g;
                                    break;
                                }
                            }
                            if (dup) |g| {
                                mod.impl.logInfo("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n", .{
                                    func,                                                                              mod.impl.llama_c.ggml_backend_dev_name(dev),   mod.impl.llama_c.ggml_backend_dev_description(dev),
                                    @as([*c]const u8, if (props.device_id != null) props.device_id else "unknown id"), mod.impl.llama_c.ggml_backend_dev_name(g.dev), mod.impl.llama_c.ggml_backend_dev_description(g.dev),
                                });
                            } else {
                                gpus.append(std.heap.c_allocator, .{ .is_meta = false, .dev = dev }) catch oom();
                            }
                        }
                    },
                    mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_IGPU => {
                        // Integrated devices seen by several backends are kept
                        // once, unless they are the same backend's.
                        if (igpus.items.len == 0 or mod.impl.llama_c.ggml_backend_dev_backend_reg(dev) == mod.impl.llama_c.ggml_backend_dev_backend_reg(igpus.items[igpus.items.len - 1].dev)) {
                            igpus.append(std.heap.c_allocator, .{ .is_meta = false, .dev = dev }) catch oom();
                        }
                    },
                    else => mod.impl.abort("llama.cpp", 270, "fatal error"),
                }
            }
        }

        // RPC servers first, to minimize network transfers.
        model.devices.insertSlice(std.heap.c_allocator, 0, rpc_servers.items) catch oom();
        model.devices.appendSlice(std.heap.c_allocator, gpus.items) catch oom();
        // Integrated GPUs only if there is no discrete one.
        if (gpus.items.len == 0) model.devices.appendSlice(std.heap.c_allocator, igpus.items) catch oom();
    }

    // In single-GPU mode, keep only the main GPU.
    if (params.split_mode == mod.impl.llama_c.LLAMA_SPLIT_MODE_NONE and model.devices.items.len != 0) {
        if (params.main_gpu < 0) {
            model.devices.clearRetainingCapacity();
        } else {
            if (params.main_gpu >= @as(c_int, @intCast(model.devices.items.len))) {
                mod.impl.logError("%s: invalid value for main_gpu: %d (available devices: %zu)\n", .{ func, params.main_gpu, model.devices.items.len });
                return false;
            }
            const main_gpu = model.devices.items[@intCast(params.main_gpu)];
            model.devices.clearRetainingCapacity();
            model.devices.append(std.heap.c_allocator, main_gpu) catch oom();
        }
    }

    for (model.devices.items) |dev| {
        var props: mod.impl.llama_c.ggml_backend_dev_props = undefined;
        mod.impl.llama_c.ggml_backend_dev_get_props(dev.dev, &props);
        mod.impl.logInfo("%s: using device %s (%s) (%s) - %zu MiB free\n", .{
            func,                                                                              mod.impl.llama_c.ggml_backend_dev_name(dev.dev), mod.impl.llama_c.ggml_backend_dev_description(dev.dev),
            @as([*c]const u8, if (props.device_id != null) props.device_id else "unknown id"), props.memory_free / 1024 / 1024,
        });
    }
    return true;
}

/// What loading a model's metadata produced.
pub const Loaded = union(enum) {
    /// Loaded; the model is the caller's to `deinit`.
    model: Model,
    /// The C++ would have thrown; `exc` holds the message `llama_model_load` logs.
    thrown,
    /// The architecture is one the port does not have yet.
    unported,
};

/// Ports the metadata half of `llama_model_load` (src/llama.cpp:316 @b1b770d90):
/// the loader and its `print_info`, the model, its hyperparameters,
/// vocabulary and statistics, and its `print_info` -- everything before
/// `load_tensors`, with the same messages for what it catches.
///
/// Parameters:
/// - `source`: where the model comes from.
/// - `splits`: the split paths, as the loader takes them.
/// - `params`: the parameters this half reads.
/// - `overrides`: metadata overrides, or null.
/// - `exc`: receives what is thrown, with the C++'s wrapping.
///
/// Return: the outcome; a loaded model is the caller's.
pub fn loadMetadata(
    source: mod.loader.Loader.Source,
    splits: *std.ArrayList([:0]const u8),
    params: mod.impl.llama_c.llama_model_params,
    overrides: ?[*]const mod.impl.llama_c.llama_model_kv_override,
    exc: *mod.impl.Exception,
) Loaded {
    var ml = mod.loader.Loader.init(source, splits, mod.impl.llama_c.LLAMA_LOAD_MODE_MMAP, false, params.no_alloc, false, overrides, exc) catch return .thrown;
    defer ml.deinit();
    ml.lazy.mode = params.lazy_mode;
    ml.printInfo();

    var model = Model.create(&ml, params, exc) catch return .thrown;
    model.ownTensorSplit();

    model.hparams.vocab_only = params.vocab_only;

    model.loadHparams(&ml, exc) catch |e| {
        model.deinit();
        if (e == error.Unported) return .unported;
        rewrap(exc, "error loading model hyperparameters: ");
        return .thrown;
    };
    if (model.arch == .clip) {
        model.deinit();
        _ = exc.throw(.runtime_error, "CLIP cannot be used as main model, use it with --mmproj instead", .{}) catch {};
        return .thrown;
    }
    model.loadVocab(&ml, exc) catch {
        model.deinit();
        rewrap(exc, "error loading model vocabulary: ");
        return .thrown;
    };

    model.loadStats(&ml);
    model.printInfo(exc) catch {
        model.deinit();
        return .thrown;
    };

    if (params.vocab_only) mod.impl.logInfo("%s: vocab only - skipping tensors\n", .{"llama_model_load"});
    return .{ .model = model };
}

/// What `llama_model_load` returns: 0 and a model, -1 on an error, -2 when
/// the progress callback cancelled.
pub const LoadResult = struct {
    status: c_int,
    model: ?*Model = null,
};

/// Ports `llama_model_load` (src/llama.cpp:316 @b1b770d90): everything, the
/// tensors included, with the C++'s messages for what it catches.
///
/// An architecture whose `load_arch_hparams` or `load_arch_tensors` is not
/// ported fails here as the C++ fails on an unknown architecture, with
/// `unsupported model architecture` in the error: once libllama swaps, a
/// model the port cannot run must not load.
///
/// Parameters:
/// - `source`: where the model comes from.
/// - `splits`: the split paths, as the loader takes them.
/// - `set_tensor_data`, `set_tensor_data_ud`: how a model built from
///   metadata gets its data; null otherwise.
/// - `params`: the model parameters.
///
/// Return: the status, and on 0 the model, the caller's to `destroy`.
pub fn load(
    source: mod.loader.Loader.Source,
    splits: *std.ArrayList([:0]const u8),
    set_tensor_data: mod.impl.llama_c.llama_model_set_tensor_data_t,
    set_tensor_data_ud: ?*anyopaque,
    params: *mod.impl.llama_c.llama_model_params,
) LoadResult {
    var exc: mod.impl.Exception = .{};
    if (loadInner(source, splits, set_tensor_data, set_tensor_data_ud, params, &exc)) |r| {
        return r;
    } else |e| {
        if (e == error.Unported) {
            mod.impl.logError("%s: error loading model: unsupported model architecture: '%s' (not ported)\n", .{ "llama_model_load", arch_name_failed });
        } else {
            exc.buf[exc.len] = 0;
            mod.impl.logError("%s: error loading model: %s\n", .{ "llama_model_load", @as([*:0]const u8, @ptrCast(&exc.buf)) });
        }
        return .{ .status = -1 };
    }
}

/// The architecture an unported load stopped at, for its message.
var arch_name_failed: [*:0]const u8 = "";

fn loadInner(
    source: mod.loader.Loader.Source,
    splits: *std.ArrayList([:0]const u8),
    set_tensor_data: mod.impl.llama_c.llama_model_set_tensor_data_t,
    set_tensor_data_ud: ?*anyopaque,
    params: *mod.impl.llama_c.llama_model_params,
    exc: *mod.impl.Exception,
) (mod.impl.Thrown || error{Unported})!LoadResult {
    var ml = try mod.loader.Loader.init(source, splits, params.load_mode, params.check_tensors, params.no_alloc, params.load_mtp, params.kv_overrides, exc);
    defer ml.deinit();
    ml.set_tensor_data = set_tensor_data;
    ml.set_tensor_data_ud = set_tensor_data_ud;
    ml.tensor_buft_overrides = params.tensor_buft_overrides;
    ml.lazy.mode = params.lazy_mode;

    ml.printInfo();
    const model = std.heap.c_allocator.create(Model) catch oom();
    model.* = Model.create(&ml, params.*, exc) catch |e| {
        std.heap.c_allocator.destroy(model);
        return e;
    };
    model.ownTensorSplit();
    errdefer destroy(model);

    if (!prepareDevices(params, model)) {
        destroy(model);
        return .{ .status = -1 };
    }

    // Loading time is recalculated after the first eval, to take the page
    // faults mmap defers into account.
    model.t_load_us = 0;
    const tm = mod.impl.TimeMeas.init(&model.t_load_us, false);
    defer tm.deinit();
    model.t_start_us = tm.t_start_us;

    model.hparams.vocab_only = params.vocab_only;

    model.loadHparams(&ml, exc) catch |e| {
        if (e == error.Unported) {
            arch_name_failed = model.archName().ptr;
            return e;
        }
        rewrap(exc, "error loading model hyperparameters: ");
        return e;
    };
    if (model.arch == .clip) return exc.throw(.runtime_error, "CLIP cannot be used as main model, use it with --mmproj instead", .{});
    model.loadVocab(&ml, exc) catch |e| {
        rewrap(exc, "error loading model vocabulary: ");
        return e;
    };

    model.loadStats(&ml);
    try model.printInfo(exc);

    if (params.vocab_only) {
        mod.impl.logInfo("%s: vocab only - skipping tensors\n", .{"llama_model_load"});
        return .{ .status = 0, .model = model };
    }

    const ok = model.loadTensors(&ml, exc) catch |e| {
        if (e == error.Unported) arch_name_failed = model.archName().ptr;
        return e;
    };
    if (!ok) {
        destroy(model);
        return .{ .status = -2 };
    }
    return .{ .status = 0, .model = model };
}

/// Frees a model `load` returned.
///
/// Return: nothing.
pub fn destroy(model: *Model) void {
    model.deinit();
    std.heap.c_allocator.destroy(model);
}

/// The default progress callback of `llama_model_load_from_file_impl`
/// (src/llama.cpp:379 @b1b770d90): a dot per percent.
fn defaultProgress(progress: f32, ctx: ?*anyopaque) callconv(.c) bool {
    const cur_percentage_p: *c_uint = @ptrCast(@alignCast(ctx.?));
    const percentage: c_uint = @intFromFloat(100 * progress);
    while (percentage > cur_percentage_p.*) {
        cur_percentage_p.* = percentage;
        mod.impl.logCont(".", .{});
        if (percentage >= 100) mod.impl.logCont("\n", .{});
    }
    return true;
}

/// Ports `llama_model_load_from_file_impl` (src/llama.cpp:379 @b1b770d90).
///
/// Parameters:
/// - `source`: exactly one of metadata, a path, or an open file.
/// - `splits`: the split paths, as the loader takes them.
/// - `set_tensor_data`, `set_tensor_data_ud`: for a model from metadata.
/// - `params_`: the model parameters.
///
/// Return: the model, the caller's to `destroy`, or null.
pub fn loadFromFileImpl(
    source: ?mod.loader.Loader.Source,
    splits: *std.ArrayList([:0]const u8),
    set_tensor_data: mod.impl.llama_c.llama_model_set_tensor_data_t,
    set_tensor_data_ud: ?*anyopaque,
    params_: mod.impl.llama_c.llama_model_params,
) ?*Model {
    var params = params_;
    const src = source orelse {
        mod.impl.logError("%s: exactly one out metadata, path_model, and file must be defined\n", .{"llama_model_load_from_file_impl"});
        return null;
    };
    mod.impl.llama_c.ggml_time_init();

    if (!params.vocab_only and mod.impl.llama_c.ggml_backend_reg_count() == 0) {
        mod.impl.logError("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", .{"llama_model_load_from_file_impl"});
        return null;
    }

    var cur_percentage: c_uint = 0;
    if (params.progress_callback == null) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = defaultProgress;
    }

    const r = load(src, splits, set_tensor_data, set_tensor_data_ud, &params);
    mod.impl.assert(r.status <= 0, "llama.cpp", 429, "status <= 0");
    if (r.status < 0) {
        if (r.status == -1) {
            mod.impl.logError("%s: failed to load model\n", .{"llama_model_load_from_file_impl"});
        } else if (r.status == -2) {
            mod.impl.logInfo("%s: cancelled model load\n", .{"llama_model_load_from_file_impl"});
        }
        if (r.model) |m| destroy(m);
        return null;
    }
    return r.model;
}

/// `throw std::runtime_error(prefix + std::string(e.what()))`.
fn rewrap(exc: *mod.impl.Exception, comptime prefix: []const u8) void {
    var what: [1024]u8 = undefined;
    const w = exc.what();
    @memcpy(what[0..w.len], w);
    exc.set(.runtime_error, prefix ++ "{s}", .{what[0..w.len]});
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(LayerNextn);
    std.testing.refAllDecls(Layer);
    std.testing.refAllDecls(Device);
    std.testing.refAllDecls(LayerDev);
    std.testing.refAllDecls(CtxBufs);
    std.testing.refAllDecls(BuftSize);
    std.testing.refAllDecls(Model);
    std.testing.refAllDecls(LoadResult);
}

test "rope scaling names round-trip and an unknown one is unspecified" {
    var exc: mod.impl.Exception = .{};
    try std.testing.expectEqualStrings("yarn", try ropeScalingTypeName(mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_YARN, &exc));
    try std.testing.expectEqual(@as(c_int, mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_LONGROPE), ropeScalingTypeFromString("longrope"));
    try std.testing.expectEqual(@as(c_int, mod.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED), ropeScalingTypeFromString("Linear"));
}

test "a modulus by zero is the dividend, as udiv makes it" {
    try std.testing.expectEqual(@as(u32, 7), remU32(7, 0));
    try std.testing.expectEqual(@as(u32, 3), remU32(7, 4));
}
