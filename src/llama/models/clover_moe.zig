//! Clover MoE: a hybrid of gated-delta-net linear attention and gated full
//! attention, as in Qwen3.5 MoE, with plain RoPE in place of M-RoPE, a
//! per-layer expert width and no multi-token-prediction block.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/clover-moe.cpp` at v0.9.0 (`cd959bae9`):
//! `llama_model_clover_moe`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const delta_net = @import("delta_net.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_clover_moe`'s `load_arch_hparams` (models/clover-moe.cpp:4 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    if (hp.nEmbdHeadK(0) != hp.nEmbdHeadV(0)) {
        return exc.throw(.runtime_error, "Clover requires equal key and value head sizes", .{});
    }

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (hp.n_expert_shared > 0) {
        hp.n_ff_shexp = (try meta.getKey(u32, .expert_shared_feed_forward_length, true, exc)).?;
    }
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

    if (!try meta.getKeyOrArr(u32, .attention_recurrent_layers, &hp.is_recr_impl, hp.n_layer_all, false, exc)) {
        const full_attn_interval = (try meta.getKey(u32, .full_attention_interval, true, exc)).?;
        if (full_attn_interval == 0) {
            return exc.throw(.runtime_error, "Clover requires a positive full attention interval", .{});
        }
        var i: u32 = 0;
        while (i < hp.n_layer_all) : (i += 1) {
            hp.is_recr_impl[i] = @intFromBool(model_mod.remU32(i + 1, full_attn_interval) != 0);
        }
    }

    var has_full = false;
    var has_linear = false;
    var i: u32 = 0;
    while (i < hp.n_layer_all) : (i += 1) {
        has_full = has_full or !hp.isRecr(i);
        has_linear = has_linear or hp.isRecr(i);
    }
    if (!has_full or !has_linear) {
        return exc.throw(.runtime_error, "Clover requires both full and linear attention layers", .{});
    }

    if (hp.ssm_dt_rank == 0 or hp.ssm_n_group == 0 or hp.ssm_d_state == 0 or
        model_mod.remU32(hp.ssm_dt_rank, hp.ssm_n_group) != 0 or
        @as(u64, hp.ssm_d_inner) != @as(u64, hp.ssm_d_state) * hp.ssm_dt_rank or hp.ssm_d_conv == 0)
    {
        return exc.throw(.runtime_error, "Clover has invalid gated delta net dimensions", .{});
    }

    self.type = switch (hp.nLayer()) {
        40 => .@"35b_a3b",
        else => .unknown,
    };
}

/// Ports `llama_model_clover_moe`'s `load_arch_tensors` (models/clover-moe.cpp:55 @cd959bae9).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    _ = ml;
    const hp = &self.hparams;
    const n_layer: i32 = @intCast(hp.nLayer());
    const n_head: i64 = hp.nHead(0);
    const n_embd: i64 = hp.n_embd;
    const n_embd_k_gqa: i64 = hp.nEmbdKGqa(0);
    const n_embd_v_gqa: i64 = hp.nEmbdVGqa(0);
    const n_embd_head_k: i64 = hp.nEmbdHeadK(0);
    const n_vocab: i64 = self.vocab.nTokens();
    const n_expert: i64 = hp.n_expert;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ n_embd, n_vocab }, nr, exc);

    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, n_vocab }, loader.tensor_duplicated, exc);
    }

    var il: i32 = 0;
    while (il < n_layer) : (il += 1) {
        const layer = &self.layers[@intCast(il)];

        const n_ff_exp: i64 = hp.nFfExp(@intCast(il));
        const n_ff_shexp: i64 = hp.n_ff_shexp;

        const head_k_dim: i64 = hp.ssm_d_state;
        const head_v_dim: i64 = hp.ssm_d_state;
        const n_k_heads: i64 = hp.ssm_n_group;
        const n_v_heads: i64 = hp.ssm_dt_rank;
        const key_dim = head_k_dim * n_k_heads;
        const value_dim = head_v_dim * n_v_heads;
        const conv_dim = key_dim * 2 + value_dim;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = il }, &.{n_embd}, 0, exc);

        if (!hp.isRecr(@intCast(il))) {
            try self.createTensorQkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ n_embd_head_k * n_head, n_embd }, 0, exc);

            layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = il }, &.{n_embd_head_k}, 0, exc);
            layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = il }, &.{n_embd_head_k}, 0, exc);
        } else {
            layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = il }, &.{ n_embd, key_dim * 2 + value_dim }, 0, exc);
            layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = il }, &.{ n_embd, value_dim }, 0, exc);
            layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = il }, &.{ hp.ssm_d_conv, conv_dim }, 0, exc);
            layer.ssm_dt = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = il }, &.{hp.ssm_dt_rank}, 0, exc);
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = il }, &.{hp.ssm_dt_rank}, 0, exc);
            layer.ssm_beta = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "weight", .bid = il }, &.{ n_embd, n_v_heads }, 0, exc);
            layer.ssm_alpha = try self.createTensor(.{ .tensor = .ssm_alpha, .suffix = "weight", .bid = il }, &.{ n_embd, n_v_heads }, 0, exc);
            layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = il }, &.{head_v_dim}, 0, exc);
            layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = il }, &.{ value_dim, n_embd }, 0, exc);
        }

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = il }, &.{ n_embd, n_expert }, 0, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = il }, &.{ n_ff_exp, n_embd, n_expert }, 0, exc);
        try self.createTensorGateUpExps(layer, il, n_embd, n_ff_exp, n_expert, 0, exc);

        if (hp.n_expert_shared > 0) {
            layer.ffn_gate_inp_shexp = try self.createTensor(.{ .tensor = .ffn_gate_inp_shexp, .suffix = "weight", .bid = il }, &.{n_embd}, 0, exc);
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, 0, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, 0, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = il }, &.{ n_ff_shexp, n_embd }, 0, exc);
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_clover_moe`'s `build_arch_graph` (models/clover-moe.cpp:115 @cd959bae9).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    _ = params;
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_clover_moe`'s `graph` (models/clover-moe.cpp:119 @cd959bae9).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/clover-moe.cpp", 123, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    b.cb(inpL, "model.input_embed", -1);

    const inp = b.buildInpMemHybrid();
    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = inpL;

        var inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        c.ggml_build_forward_expand(b.gf, cur);

        if (hp.isRecr(@intCast(il))) {
            cur = buildLayerAttnLinear(model, b, inp.getRecr(), cur, il);
        } else {
            cur = buildLayerAttn(model, b, inp.getAttn(), cur, inp_pos, il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        cur = c.ggml_add(ctx, cur, inpSA);
        b.cb(cur, "attn_residual", il);

        const ffn_residual = cur;

        const attn_post_norm = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(attn_post_norm, "attn_post_norm", il);

        cur = buildLayerFfn(model, b, attn_post_norm, il);
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_residual);
        b.cb(cur, "post_moe", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

/// Ports `llama_model_clover_moe::graph`'s `build_qkvz` (models/clover-moe.cpp:190 @cd959bae9).
fn buildQkvz(model: *const Model, b: *const graph.Builder, input: *Tensor, il: c_int) struct { *Tensor, *Tensor } {
    const layer = &model.layers[@intCast(il)];
    const n_seqs: i64 = b.ubatch.n_seqs;
    const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

    var qkv_mixed = b.buildLoraMm(layer.wqkv.?, input, layer.wqkv_s);
    qkv_mixed = c.ggml_reshape_3d(b.ctx0, qkv_mixed, qkv_mixed.*.ne[0], n_seq_tokens, n_seqs);
    b.cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    const z = b.buildLoraMm(layer.wqkv_gate.?, input, layer.wqkv_gate_s);
    b.cb(z, "z", il);

    return .{ qkv_mixed, z };
}

/// Ports `llama_model_clover_moe::graph`'s `build_norm_gated` (models/clover-moe.cpp:206 @cd959bae9).
fn buildNormGated(b: *const graph.Builder, input: *Tensor, weights: ?*Tensor, gate: *Tensor, il: c_int) *Tensor {
    const normalized = b.buildNorm(input, weights, null, .rms, il);
    return c.ggml_mul(b.ctx0, normalized, c.ggml_silu(b.ctx0, gate));
}

/// Ports `llama_model_clover_moe::graph`'s `build_layer_attn` (models/clover-moe.cpp:215
/// @cd959bae9): a joint Q and gate projection, Q and K norms, RoPE, attention
/// and the sigmoid gate.
fn buildLayerAttn(model: *const Model, b: *const graph.Builder, inp: *graph.InpAttnKv, cur_: *Tensor, inp_pos: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/clover-moe.cpp", 221, "n_embd_head == hparams.n_embd_head_k()");
    const n_head = b.n_head;
    const n_head_kv = b.n_head_kv;
    const n_tokens = b.n_tokens;

    const qkv = b.buildQkvHeads(layer, cur_, n_embd_head * 2, n_head, n_embd_head, n_head_kv, n_embd_head, n_head_kv, il, false);
    const Qcur_full = qkv.q;
    var Kcur = qkv.k;
    var Vcur = qkv.v;
    b.cb(Qcur_full, "Qcur_full", il);

    const es = c.ggml_element_size(Qcur_full);
    const head_sz: usize = @intCast(n_embd_head);
    var Qcur = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, es * head_sz * 2, es * head_sz * 2 * @as(usize, @intCast(n_head)), 0);
    b.cb(Qcur, "Qcur_reshaped", il);

    Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
    b.cb(Qcur, "Qcur_normed", il);

    b.cb(Kcur, "Kcur", il);

    b.cb(Vcur, "Vcur", il);

    Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
    b.cb(Kcur, "Kcur_normed", il);

    var gate = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, es * head_sz * 2, es * head_sz * 2 * @as(usize, @intCast(n_head)), es * head_sz);
    gate = c.ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
    b.cb(gate, "gate_reshaped", il);

    Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    b.cb(Qcur, "Qcur", il);
    b.cb(Kcur, "Kcur", il);
    b.cb(Vcur, "Vcur", il);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    var cur = b.buildAttn(inp, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(cur, "attn_pregate", il);

    const gate_sigmoid = c.ggml_sigmoid(ctx, gate);
    b.cb(gate_sigmoid, "gate_sigmoid", il);

    cur = c.ggml_mul(ctx, cur, gate_sigmoid);
    b.cb(cur, "attn_gated", il);

    cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
    b.cb(cur, "attn_output", il);

    return cur;
}

/// Ports `llama_model_clover_moe::graph`'s `build_layer_attn_linear` (models/clover-moe.cpp:286
/// @cd959bae9): the gated delta net layer.
fn buildLayerAttnLinear(model: *const Model, b: *const graph.Builder, inp: *graph.InpRs, cur_: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    const hp = b.hparams;
    const mctx_cur = inp.mctx;

    const d_inner: i64 = hp.ssm_d_inner;
    const n_seqs: i64 = b.ubatch.n_seqs;
    const head_k_dim: i64 = hp.ssm_d_state;
    const num_k_heads: i64 = hp.ssm_n_group;
    const num_v_heads: i64 = hp.ssm_dt_rank;
    const head_v_dim: i64 = @divTrunc(d_inner, num_v_heads);
    const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

    impl.assert(n_seqs != 0, "models/clover-moe.cpp", 300, "n_seqs != 0");
    impl.assert(b.ubatch.equalSeqs(), "models/clover-moe.cpp", 301, "ubatch.equal_seqs()");
    impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/clover-moe.cpp", 302, "ubatch.n_tokens == n_seq_tokens * n_seqs");

    const qkv_mixed, const z = buildQkvz(model, b, cur_, il);

    var beta = b.buildLoraMm(layer.ssm_beta.?, cur_, layer.ssm_beta_s);
    var alpha = b.buildLoraMm(layer.ssm_alpha.?, cur_, layer.ssm_alpha_s);
    beta = c.ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    b.cb(beta, "beta", il);

    beta = c.ggml_sigmoid(ctx, beta);
    b.cb(beta, "beta_sigmoid", il);

    alpha = c.ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    b.cb(alpha, "alpha", il);

    const alpha_biased = c.ggml_add(ctx, alpha, layer.ssm_dt);
    const alpha_softplus = c.ggml_softplus(ctx, alpha_biased);
    b.cb(alpha_softplus, "a_softplus", il);

    var gate = c.ggml_mul(ctx, alpha_softplus, layer.ssm_a);
    b.cb(gate, "gate", il);

    gate = c.ggml_reshape_4d(ctx, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    const conv_states_all = mctx_cur.getRL(il);
    const ssm_states_all = mctx_cur.getSL(il);

    const conv_kernel = layer.ssm_conv1d.?;
    const conv_kernel_size = conv_kernel.ne[0];
    const conv_channels = d_inner + 2 * @as(i64, hp.ssm_n_group) * hp.ssm_d_state;

    const conv_input = delta_net.buildConvState(b, inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    var state = b.buildRs(inp, ssm_states_all, @intCast(hp.nEmbdS()), @intCast(n_seqs), null);
    state = c.ggml_reshape_4d(ctx, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    b.cb(state, "state_predelta", il);

    const conv_output_proper = c.ggml_ssm_conv(ctx, conv_input, conv_kernel);
    b.cb(conv_output_proper, "conv_output_raw", il);

    const conv_output_silu = c.ggml_silu(ctx, conv_output_proper);
    b.cb(conv_output_silu, "conv_output_silu", il);

    const conv_qkv_mix = conv_output_silu;

    const qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const nb1_qkv = c.ggml_row_size(conv_qkv_mix.*.type, qkv_dim);
    const ty = conv_qkv_mix.*.type;

    var q_conv = c.ggml_view_4d(ctx, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_k_dim), nb1_qkv, nb1_qkv * @as(usize, @intCast(n_seq_tokens)), 0);
    var k_conv = c.ggml_view_4d(ctx, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_k_dim), nb1_qkv, nb1_qkv * @as(usize, @intCast(n_seq_tokens)), @as(usize, @intCast(head_k_dim * num_k_heads)) * c.ggml_element_size(conv_qkv_mix));
    const v_conv = c.ggml_view_4d(ctx, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_v_dim), nb1_qkv, nb1_qkv * @as(usize, @intCast(n_seq_tokens)), c.ggml_row_size(ty, 2 * head_k_dim * num_k_heads));

    b.cb(q_conv, "q_conv", il);
    b.cb(k_conv, "k_conv", il);
    b.cb(v_conv, "v_conv", il);

    q_conv = delta_net.buildGdnL2Norm(ctx, q_conv, 1e-6);
    k_conv = delta_net.buildGdnL2Norm(ctx, k_conv, 1e-6);

    if (num_k_heads != num_v_heads and (!b.cparams.fused_gdn_ar or !b.cparams.fused_gdn_ch)) {
        q_conv = c.ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = c.ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    const output = delta_net.buildRecurrentAttn(b, inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    const z_2d = c.ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    const attn_out_norm = buildNormGated(b, output, layer.ssm_norm, z_2d, il);

    const final_output = c.ggml_reshape_2d(ctx, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    b.cb(final_output, "final_output", il);

    const cur = b.buildLoraMm(layer.ssm_out.?, final_output, layer.ssm_out_s);
    b.cb(cur, "linear_attn_out", il);

    return cur;
}

/// Ports `llama_model_clover_moe::graph`'s `build_layer_ffn` (models/clover-moe.cpp:397
/// @cd959bae9): the routed experts, plus the shared expert behind its sigmoid
/// gate when there is one.
fn buildLayerFfn(model: *const Model, b: *const graph.Builder, cur_: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    impl.assert(layer.ffn_gate_inp != null, "models/clover-moe.cpp", 398, "model.layers[il].ffn_gate_inp != nullptr");

    const moe_out = b.buildMoeFfn(cur_, .{
        .gate_inp = layer.ffn_gate_inp,
        .up_exps = layer.ffn_up_exps,
        .gate_exps = layer.ffn_gate_exps,
        .down_exps = layer.ffn_down_exps.?,
        .n_expert = b.n_expert,
        .n_expert_used = b.n_expert_used,
        .type_op = .silu,
        .norm_w = true,
        .w_scale = b.hparams.expert_weights_scale,
        .gating_op = .softmax,
        .gate_up_exps = layer.ffn_gate_up_exps,
        .up_exps_s = layer.ffn_up_exps_s,
        .gate_exps_s = layer.ffn_gate_exps_s,
        .down_exps_s = layer.ffn_down_exps_s,
    }, il);
    b.cb(moe_out, "ffn_moe_out", il);

    if (layer.ffn_up_shexp == null) return moe_out;

    var ffn_shexp = b.buildFfn(cur_, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
    b.cb(ffn_shexp, "ffn_shexp", il);

    var shared_gate = b.buildLoraMm(layer.ffn_gate_inp_shexp.?, cur_, null);
    b.cb(shared_gate, "shared_expert_gate", il);

    shared_gate = c.ggml_sigmoid(ctx, shared_gate);
    b.cb(shared_gate, "shared_expert_gate_sigmoid", il);

    ffn_shexp = c.ggml_mul(ctx, ffn_shexp, shared_gate);
    b.cb(ffn_shexp, "ffn_shexp_gated", il);
    const cur: *Tensor = c.ggml_add(ctx, moe_out, ffn_shexp);
    b.cb(cur, "ffn_out", il);
    return cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
