//! BailingMoE3 (Ling 3): a hybrid of Kimi delta attention -- a gated delta
//! net with a bounded per-channel decay and short causal convolutions on
//! Q, K and V -- and gated multi-head latent attention over a K-only
//! cache, with leading dense blocks, then a mixture of experts plus shared
//! experts, and an optional NextN/MTP block past the main stack.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/bailingmoe3.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_bailingmoe3`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`,
//! `bailingmoe3_causal_conv1d`, and its `graph` and `graph_mtp`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_bailingmoe3`'s `load_arch_hparams` (models/bailingmoe3.cpp:6 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_embd_head_k_mla_impl = (try meta.getKey(u32, .attention_key_length_mla, true, exc)).?;
    hp.n_embd_head_v_mla_impl = (try meta.getKey(u32, .attention_value_length_mla, true, exc)).?;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    if (try meta.getKey(u32, .attention_q_lora_rank, false, exc)) |v| hp.n_lora_q = v;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.n_embd_head_kda = (try meta.getKey(u32, .kda_head_dim, true, exc)).?;
    if (try meta.getKey(bool, .kda_safe_gate, false, exc)) |v| {
        hp.kda_safe_gate = v;
    } else {
        hp.kda_safe_gate = true;
    }
    hp.kda_gate_lower_bound = (try meta.getKey(f32, .kda_gate_lower_bound, true, exc)).?;
    _ = try meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, false, exc);
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    hp.n_layer_dense_lead = (try meta.getKey(u32, .leading_dense_block_count, true, exc)).?;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;
    _ = try meta.getKeyOrArr(f32, .swiglu_clamp_exp, &hp.swiglu_clamp_exp, hp.n_layer_all, false, exc);
    _ = try meta.getKeyOrArr(f32, .swiglu_clamp_shexp, &hp.swiglu_clamp_shexp, hp.n_layer_all, false, exc);

    if (hp.n_ff_shexp == 0) {
        hp.n_ff_shexp = hp.nFfExp(0) *% @max(1, hp.n_expert_shared);
    }

    llama.impl.assert(hp.kda_safe_gate, "models/bailingmoe3.cpp", 33, "hparams.kda_safe_gate");
    llama.impl.assert(hp.kda_gate_lower_bound < 0.0, "models/bailingmoe3.cpp", 34, "hparams.kda_gate_lower_bound < 0.0f");

    var il: u32 = 0;
    while (il < hp.nLayer()) : (il += 1) {
        hp.is_recr_impl[il] = @intFromBool(hp.nHeadKv(il) == 0);
    }

    self.type = switch (hp.nLayer()) {
        24 => if (hp.n_embd == 1536 and hp.n_expert == 128) .@"7_9b_a1_3b" else .unknown,
        42 => if (hp.n_embd == 2560 and hp.n_expert == 512) .@"124b_a5_1b" else .unknown,
        else => .unknown,
    };
}

/// Ports `llama_model_bailingmoe3`'s `load_arch_tensors` (models/bailingmoe3.cpp:47 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, ml: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;
    const n_embd = l.n_embd;
    const n_head = l.n_head;
    const n_expert = l.n_expert;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    const head_dim: i64 = hp.n_embd_head_kda;
    const d_inner = head_dim * n_head;
    const d_conv: i64 = hp.ssm_d_conv;
    const kv_lora_rank: i64 = hp.n_lora_kv;
    const q_lora_rank: i64 = hp.n_lora_q;
    const qk_rope_head_dim: i64 = hp.nRot(0);
    const qk_head_dim: i64 = hp.nEmbdHeadKMla();
    const v_head_dim: i64 = hp.nEmbdHeadVMla();
    const n_ff_exp: i64 = hp.nFfExp(0);
    const n_ff_shexp: i64 = hp.n_ff_shexp;

    const nf = self.nextnFlags(ml, .attn_norm);
    const trunk_flags = nf.trunk;
    const mtp_flags = nf.mtp;

    var il: i32 = 0;
    while (il < l.n_layer) : (il += 1) {
        const layer = &self.layers[@intCast(il)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, trunk_flags, exc);

        if (hp.isRecr(@intCast(il))) {
            layer.ssm_q_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_q, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner, 1 }, trunk_flags, exc);
            layer.ssm_k_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_k, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner, 1 }, trunk_flags, exc);
            layer.ssm_v_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_v, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner, 1 }, trunk_flags, exc);

            try self.createTensorQkv(layer, il, n_embd, d_inner, d_inner, d_inner, trunk_flags, exc);
            layer.ssm_f_a = try self.createTensor(.{ .tensor = .ssm_f_a, .suffix = "weight", .bid = il }, &.{ n_embd, d_inner }, trunk_flags, exc);
            layer.ssm_beta = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "weight", .bid = il }, &.{ n_embd, n_head }, trunk_flags, exc);
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = il }, &.{ 1, n_head }, trunk_flags, exc);
            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = il }, &.{d_inner}, trunk_flags, exc);
            layer.ssm_g_a = try self.createTensor(.{ .tensor = .ssm_g_a, .suffix = "weight", .bid = il }, &.{ n_embd, d_inner }, trunk_flags, exc);
            layer.ssm_o_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = il }, &.{head_dim}, trunk_flags, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ d_inner, n_embd }, trunk_flags, exc);
        } else {
            try loadMla(self, layer, il, trunk_flags, n_embd, n_head, q_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_head_dim, v_head_dim, exc);
        }

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, trunk_flags, exc);
        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = il }, &.{ n_embd, l.n_ff }, trunk_flags, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = il }, &.{ n_embd, l.n_ff }, trunk_flags, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = il }, &.{ l.n_ff, n_embd }, trunk_flags, exc);
        } else {
            try loadMoe(self, layer, il, trunk_flags, n_embd, n_expert, n_ff_exp, n_ff_shexp, exc);
        }
    }

    il = l.n_layer;
    while (il < l.n_layer_all) : (il += 1) {
        const layer = &self.layers[@intCast(il)];
        const flags = mtp_flags;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
        try loadMla(self, layer, il, flags, n_embd, n_head, q_lora_rank, kv_lora_rank, qk_rope_head_dim, qk_head_dim, v_head_dim, exc);
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
        try loadMoe(self, layer, il, flags, n_embd, n_expert, n_ff_exp, n_ff_shexp, exc);
        layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = il }, &.{ 2 * n_embd, n_embd }, flags, exc);
        layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
        layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
        layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .layer_out_norm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
    }
}

/// The MLA tensors, which `load_arch_tensors` (models/bailingmoe3.cpp:47
/// @b1b770d90) loads alike for the trunk's attention layers and the MTP
/// block.
fn loadMla(self: *llama.model.Model, layer: *llama.model.Layer, il: i32, flags: u32, n_embd: i64, n_head: i64, q_lora_rank: i64, kv_lora_rank: i64, qk_rope_head_dim: i64, qk_head_dim: i64, v_head_dim: i64, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    if (q_lora_rank > 0) {
        layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = il }, &.{ n_embd, q_lora_rank }, flags, exc);
        layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = il }, &.{q_lora_rank}, flags, exc);
        layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = il }, &.{ q_lora_rank, n_head * qk_head_dim }, flags, exc);
    } else {
        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = il }, &.{ n_embd, n_head * qk_head_dim }, flags, exc);
    }
    layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = il }, &.{ n_embd, kv_lora_rank + qk_rope_head_dim }, flags, exc);
    layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = il }, &.{kv_lora_rank}, flags, exc);
    layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = il }, &.{ qk_head_dim - qk_rope_head_dim, kv_lora_rank, n_head }, flags, exc);
    layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = il }, &.{ kv_lora_rank, v_head_dim, n_head }, flags, exc);
    layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = il }, &.{ n_embd, n_head }, flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ n_head * v_head_dim, n_embd }, flags, exc);
}

/// The MoE tensors, routed and shared, which `load_arch_tensors`
/// (models/bailingmoe3.cpp:47 @b1b770d90) loads alike for the trunk's MoE
/// layers and the MTP block.
fn loadMoe(self: *llama.model.Model, layer: *llama.model.Layer, il: i32, flags: u32, n_embd: i64, n_expert: i64, n_ff_exp: i64, n_ff_shexp: i64, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = il }, &.{ n_embd, n_expert }, flags, exc);
    layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = il }, &.{n_expert}, flags, exc);
    layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_exp, n_expert }, flags, exc);
    layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_exp, n_expert }, flags, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = il }, &.{ n_ff_exp, n_embd, n_expert }, flags, exc);
    layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, flags, exc);
    layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, flags, exc);
    layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = il }, &.{ n_ff_shexp, n_embd }, flags, exc);
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_bailingmoe3`'s `build_arch_graph` (models/bailingmoe3.cpp:156
/// @b1b770d90): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, params: *const llama.graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

/// Ports `llama_model_bailingmoe3::graph`'s `build_causal_conv1d`
/// (models/bailingmoe3.cpp:163 @b1b770d90): the short convolution of one of
/// Q (`qkv` 0), K (1) or V (2), its projection in front, its slice of the
/// conv state in front of the input, and the conv history of the ubatch's
/// cells stored through `build_conv_state_store`.
fn buildCausalConv1d(b: *const llama.graph.Builder, inp_rs: *const llama.graph.InpRs, conv_states_all: *llama.impl.c.ggml_tensor, conv_state_all: *llama.impl.c.ggml_tensor, qkv: i64, x: *llama.impl.c.ggml_tensor, proj_w: ?*llama.impl.c.ggml_tensor, conv_w: ?*llama.impl.c.ggml_tensor, d_conv: i64, head_dim: i64, n_head: i64, n_seq_tokens: i64, n_seqs: i64) *llama.impl.c.ggml_tensor {
    const ctx = b.ctx0;
    const d_inner = head_dim * n_head;
    const conv_state_size = (d_conv - 1) * d_inner;
    const total_state_size = 3 * conv_state_size;

    const es_state: usize = llama.impl.c.ggml_element_size(conv_state_all);
    const conv_state = llama.impl.c.ggml_view_3d(ctx, conv_state_all, d_conv - 1, d_inner, n_seqs, @as(usize, @intCast(d_conv - 1)) * es_state, @as(usize, @intCast(total_state_size)) * es_state, @as(usize, @intCast(qkv * conv_state_size)) * es_state);

    var x_proj = llama.impl.c.ggml_mul_mat(ctx, proj_w, x);
    x_proj = llama.impl.c.ggml_reshape_3d(ctx, x_proj, d_inner, n_seq_tokens, n_seqs);
    const conv_x: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, conv_state, llama.impl.c.ggml_transpose(ctx, x_proj), 0);

    b.buildConvStateStore(inp_rs, conv_states_all, conv_x, d_conv - 1, llama.impl.c.ggml_row_size(conv_states_all.type, qkv * conv_state_size));

    const conv_weight = llama.impl.c.ggml_reshape_2d(ctx, conv_w, d_conv, d_inner);
    var out = llama.impl.c.ggml_ssm_conv(ctx, conv_x, conv_weight);
    out = llama.impl.c.ggml_silu(ctx, llama.impl.c.ggml_reshape_2d(ctx, out, d_inner, b.n_tokens));
    return llama.impl.c.ggml_reshape_4d(ctx, out, head_dim, n_head, n_seq_tokens, n_seqs);
}

/// The dimensions of the MLA attention, shared by `graph` and `graph_mtp`.
const MlaDims = struct {
    n_head: i64,
    qk_head_dim: i64,
    v_head_dim: i64,
    qk_rope_head_dim: i64,
    qk_nope_head_dim: i64,
    kv_lora_rank: i64,
    kq_scale: f32,

    fn of(hp: *const llama.hparams.Hparams) MlaDims {
        const qk_head_dim: i64 = hp.nEmbdHeadKMla();
        const qk_rope_head_dim: i64 = hp.nRot(0);
        return .{
            .n_head = hp.nHead(0),
            .qk_head_dim = qk_head_dim,
            .v_head_dim = hp.nEmbdHeadVMla(),
            .qk_rope_head_dim = qk_rope_head_dim,
            .qk_nope_head_dim = qk_head_dim - qk_rope_head_dim,
            .kv_lora_rank = hp.n_lora_kv,
            .kq_scale = 1.0 / stdlib.libm.sqrtfInt(qk_head_dim),
        };
    }
};

/// The gated MLA attention of a trunk attention layer in `graph`
/// (models/bailingmoe3.cpp:189 @b1b770d90) and of the MTP block in
/// `graph_mtp` (models/bailingmoe3.cpp:379 @b1b770d90), the same lines:
/// Q through the optional low-rank pair, the latent and its RoPE key, Q
/// absorbing `wk_b`, MQA over the K-only cache with `wv_b` decompressing
/// its output, then the per-head sigmoid gate and the output projection.
fn buildMlaAttn(b: *const llama.graph.Builder, layer: *const llama.model.Layer, inp_attn: *llama.graph.InpAttnK, cur: *llama.impl.c.ggml_tensor, inp_pos: *llama.impl.c.ggml_tensor, d: *const MlaDims, il: c_int) *llama.impl.c.ggml_tensor {
    const ctx = b.ctx0;
    const n_tokens = b.n_tokens;
    const n_head = d.n_head;

    const attn_input = cur;
    var q_all: *llama.impl.c.ggml_tensor = undefined;
    if (layer.wq_a != null) {
        q_all = llama.impl.c.ggml_mul_mat(ctx, layer.wq_a, cur);
        b.cb(q_all, "q_a", il);
        q_all = b.buildNorm(q_all, layer.attn_q_a_norm, null, .rms, il);
        b.cb(q_all, "q_a_norm", il);
        q_all = llama.impl.c.ggml_mul_mat(ctx, layer.wq_b, q_all);
        b.cb(q_all, "q_b", il);
    } else {
        q_all = llama.impl.c.ggml_mul_mat(ctx, layer.wq, cur);
    }
    var q_nope: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, q_all, d.qk_nope_head_dim, n_head, n_tokens, llama.impl.c.ggml_row_size(q_all.type, d.qk_head_dim), llama.impl.c.ggml_row_size(q_all.type, d.qk_head_dim) * @as(usize, @intCast(n_head)), 0);
    var q_pe: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, q_all, d.qk_rope_head_dim, n_head, n_tokens, llama.impl.c.ggml_row_size(q_all.type, d.qk_head_dim), llama.impl.c.ggml_row_size(q_all.type, d.qk_head_dim) * @as(usize, @intCast(n_head)), llama.impl.c.ggml_row_size(q_all.type, d.qk_nope_head_dim));

    const kv_all: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
    var kv: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_2d(ctx, kv_all, d.kv_lora_rank, n_tokens, llama.impl.c.ggml_row_size(kv_all.type, d.kv_lora_rank + d.qk_rope_head_dim), 0);
    var k_pe: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, kv_all, d.qk_rope_head_dim, 1, n_tokens, llama.impl.c.ggml_row_size(kv_all.type, d.kv_lora_rank + d.qk_rope_head_dim), llama.impl.c.ggml_row_size(kv_all.type, d.kv_lora_rank + d.qk_rope_head_dim), llama.impl.c.ggml_row_size(kv_all.type, d.kv_lora_rank));

    q_pe = b.buildRope(q_pe, inp_pos, il);
    k_pe = b.buildRope(k_pe, inp_pos, il);
    kv = b.buildNorm(kv, layer.attn_kv_a_norm, null, .rms, il);

    q_nope = llama.impl.c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    q_nope = llama.impl.c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
    q_nope = llama.impl.c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);

    const q: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, q_nope, q_pe, 0);
    kv = llama.impl.c.ggml_reshape_3d(ctx, kv, d.kv_lora_rank, 1, n_tokens);
    const k: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, kv, k_pe, 0);

    var out = b.buildAttn(inp_attn, null, null, null, q, k, kv, null, null, layer.wv_b, d.kq_scale, il);

    var attn_gate: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.wqkv_gate, attn_input);
    attn_gate = llama.impl.c.ggml_sigmoid(ctx, llama.impl.c.ggml_reshape_3d(ctx, attn_gate, 1, n_head, n_tokens));
    out = llama.impl.c.ggml_reshape_3d(ctx, out, d.v_head_dim, n_head, n_tokens);
    out = llama.impl.c.ggml_mul(ctx, out, attn_gate);
    out = llama.impl.c.ggml_mul_mat(ctx, layer.wo, llama.impl.c.ggml_cont_2d(ctx, out, d.v_head_dim * n_head, n_tokens));
    return out;
}

/// The routed experts plus the shared experts, of the trunk's MoE layers in
/// `graph` (models/bailingmoe3.cpp:189 @b1b770d90) and of the MTP block in
/// `graph_mtp` (models/bailingmoe3.cpp:379 @b1b770d90).
fn buildMoe(b: *const llama.graph.Builder, layer: *const llama.model.Layer, cur: *llama.impl.c.ggml_tensor, il: c_int) *llama.impl.c.ggml_tensor {
    const hp = b.hparams;
    const moe = b.buildMoeFfn(cur, .{
        .gate_inp = layer.ffn_gate_inp,
        .up_exps = layer.ffn_up_exps,
        .gate_exps = layer.ffn_gate_exps,
        .down_exps = layer.ffn_down_exps.?,
        .exp_probs_b = layer.ffn_exp_probs_b,
        .n_expert = b.n_expert,
        .n_expert_used = b.n_expert_used,
        .type_op = .silu,
        .norm_w = hp.expert_weights_norm,
        .w_scale = hp.expert_weights_scale,
        .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
    }, il);
    const shared = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
    return llama.impl.c.ggml_add(b.ctx0, moe, shared);
}

/// Ports the constructor of `llama_model_bailingmoe3`'s `graph` (models/bailingmoe3.cpp:189 @b1b770d90).
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    b.cb(inpL, "model.input_embed", -1);

    const inp = b.buildInpMemHybridK();
    const inp_rs = inp.getRecr();
    const inp_attn = inp.getAttn();

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    const d = MlaDims.of(hp);
    const n_head = d.n_head;
    const head_dim: i64 = hp.n_embd_head_kda;
    const d_inner = n_head * head_dim;
    const d_conv: i64 = hp.ssm_d_conv;
    const n_seqs: i64 = b.ubatch.n_seqs;
    const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;
    const n_tokens = b.n_tokens;

    llama.impl.assert(n_seqs > 0, "models/bailingmoe3.cpp", 215, "n_seqs > 0");
    llama.impl.assert(b.ubatch.equalSeqs(), "models/bailingmoe3.cpp", 216, "ubatch.equal_seqs()");
    llama.impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/bailingmoe3.cpp", 217, "ubatch.n_tokens == n_seq_tokens * n_seqs");

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        b.res.t_layer_inp[@intCast(il)] = inpL;

        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        if (hp.isRecr(@intCast(il))) {
            const mctx_cur = inp_rs.mctx;
            const conv_states_all = mctx_cur.getRL(il);
            const conv_state_all = b.buildRs(inp_rs, conv_states_all, @intCast(hp.nEmbdR()), @intCast(n_seqs), null);

            var q = buildCausalConv1d(b, inp_rs, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs);
            var k = buildCausalConv1d(b, inp_rs, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs);
            const v = buildCausalConv1d(b, inp_rs, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs);

            var gate = llama.impl.c.ggml_mul_mat(ctx, layer.ssm_f_a, cur);
            gate = llama.impl.c.ggml_add(ctx, gate, layer.ssm_dt_b);
            gate = llama.impl.c.ggml_reshape_3d(ctx, gate, head_dim, n_head, n_tokens);
            const a = llama.impl.c.ggml_reshape_3d(ctx, layer.ssm_a, 1, n_head, 1);
            gate = llama.impl.c.ggml_scale(ctx, llama.impl.c.ggml_sigmoid(ctx, llama.impl.c.ggml_mul(ctx, gate, a)), hp.kda_gate_lower_bound);
            gate = llama.impl.c.ggml_reshape_4d(ctx, gate, head_dim, n_head, n_seq_tokens, n_seqs);
            b.cb(gate, "kda_gate", il);

            var beta = llama.impl.c.ggml_mul_mat(ctx, layer.ssm_beta, cur);
            beta = llama.impl.c.ggml_sigmoid(ctx, llama.impl.c.ggml_reshape_4d(ctx, beta, 1, n_head, n_seq_tokens, n_seqs));

            q = llama.models.delta_net.buildGdnL2Norm(ctx, q, hp.f_norm_rms_eps);
            k = llama.models.delta_net.buildGdnL2Norm(ctx, k, hp.f_norm_rms_eps);

            const states_all = mctx_cur.getSL(il);
            var state = b.buildRs(inp_rs, states_all, @intCast(hp.nEmbdS()), @intCast(n_seqs), null);
            state = llama.impl.c.ggml_reshape_4d(ctx, state, head_dim, head_dim, n_head, n_seqs);

            var out = llama.impl.c.ggml_cont(ctx, llama.models.delta_net.buildRecurrentAttn(b, inp_rs, states_all, q, k, v, gate, beta, state, il));

            var out_gate = llama.impl.c.ggml_mul_mat(ctx, layer.ssm_g_a, cur);
            out_gate = llama.impl.c.ggml_reshape_3d(ctx, out_gate, head_dim, n_head, n_tokens);
            out = llama.impl.c.ggml_reshape_3d(ctx, out, head_dim, n_head, n_tokens);
            out = b.buildNorm(out, layer.ssm_o_norm, null, .rms, il);
            out = llama.impl.c.ggml_mul(ctx, out, llama.impl.c.ggml_sigmoid(ctx, out_gate));
            cur = llama.impl.c.ggml_mul_mat(ctx, layer.wo, llama.impl.c.ggml_cont_2d(ctx, out, d_inner, n_tokens));
            b.cb(cur, "kda_out", il);
        } else {
            cur = buildMlaAttn(b, layer, inp_attn, cur, inp_pos, &d, il);
            b.cb(cur, "mla_out", il);
        }

        if (il == b.n_layer - 1 and b.cropBeforeNextn(inp_out_ids)) {
            cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpSA);
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);

        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        } else {
            cur = buildMoe(b, layer, cur, il);
        }

        cur = llama.impl.c.ggml_add(ctx, cur, ffn_inp);
        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);
        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (b.cropAfterNextn(inp_out_ids)) cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = llama.impl.c.ggml_mul_mat(ctx, model.output, cur);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

/// Ports the constructor of `llama_model_bailingmoe3`'s `graph_mtp`
/// (models/bailingmoe3.cpp:379 @b1b770d90): the NextN draft head.
fn buildGraphMtp(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    llama.impl.assert(hp.n_layer_nextn == 1, "models/bailingmoe3.cpp", 381, "hparams.n_layer_nextn == 1 && \"BailingMoE3 MTP requires one NextN layer\"");

    const il: c_int = @as(c_int, @intCast(hp.nLayer())) + b.cparams.nextn_layer_offset;
    llama.impl.assert(b.cparams.nextn_layer_offset >= 0 and b.cparams.nextn_layer_offset < @as(c_int, @intCast(hp.n_layer_nextn)), "models/bailingmoe3.cpp", 386, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range\"");
    const layer = &model.layers[@intCast(il)];

    llama.impl.assert(layer.nextn.eh_proj != null, "models/bailingmoe3.cpp", 389, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    llama.impl.assert(layer.nextn.enorm != null, "models/bailingmoe3.cpp", 390, "layer.nextn.enorm && \"MTP block missing nextn.enorm\"");
    llama.impl.assert(layer.nextn.hnorm != null, "models/bailingmoe3.cpp", 391, "layer.nextn.hnorm && \"MTP block missing nextn.hnorm\"");
    llama.impl.assert(layer.nextn.shared_head_norm != null, "models/bailingmoe3.cpp", 392, "layer.nextn.shared_head_norm && \"MTP block missing final norm\"");

    const d = MlaDims.of(hp);

    const inp = std.heap.c_allocator.create(llama.graph.InpEmbdH) catch llama.impl.abort("models/bailingmoe3.cpp", 402, "out of memory");
    inp.* = .{ .n_embd = hp.nEmbdInp(), .n_embd_state = hp.n_embd };
    inp.tokens = llama.impl.c.ggml_new_tensor_1d(ctx, llama.impl.c.GGML_TYPE_I32, b.n_tokens);
    llama.impl.c.ggml_set_input(inp.tokens);
    inp.embd = llama.impl.c.ggml_new_tensor_2d(ctx, llama.impl.c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    llama.impl.c.ggml_set_input(inp.embd);
    inp.h = b.buildInpEmbdState(inp, hp.n_embd);

    const tok_embd: *llama.impl.c.ggml_tensor = if (b.ubatch.token != null) llama.impl.c.ggml_get_rows(ctx, model.tok_embd, inp.tokens) else inp.embd.?;
    const h_norm = b.buildNorm(inp.h.?, layer.nextn.hnorm, null, .rms, il);
    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    var cur: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.nextn.eh_proj, llama.impl.c.ggml_concat(ctx, e_norm, h_norm, 0));
    b.cb(cur, "mtp_eh_proj", il);

    _ = b.res.addInput(llama.graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();
    const inp_attn = b.buildAttnInpK();

    const inpSA = cur;
    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);

    cur = buildMlaAttn(b, layer, inp_attn, cur, inp_pos, &d, il);

    const ffn_inp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpSA);
    cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);

    cur = buildMoe(b, layer, cur, il);
    cur = llama.impl.c.ggml_add(ctx, cur, ffn_inp);
    cur = b.buildNorm(cur, layer.nextn.shared_head_norm, null, .rms, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
    cur = llama.impl.c.ggml_mul_mat(ctx, model.output, cur);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
