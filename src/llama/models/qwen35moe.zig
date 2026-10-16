//! Qwen3.5 MoE: Qwen3.5's hybrid of gated-delta-net linear attention and
//! full attention, whose feed-forward is a softmax-routed mixture of experts
//! plus a shared expert behind a sigmoid gate, with an optional
//! multi-token-prediction block.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/qwen35moe.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_qwen35moe`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph` and `graph_mtp`.

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
const Layer = model_mod.Layer;
const Tensor = c.ggml_tensor;

/// `int64_t / int64_t` as aarch64 computes it: `sdiv` by zero gives zero.
/// The C++ is undefined there; this is what the reference binary does.
fn divI64(a: i64, b: i64) i64 {
    return if (b == 0) 0 else @divTrunc(a, b);
}

/// Ports `llama_model_qwen35moe`'s `load_arch_hparams` (models/qwen35moe.cpp:4 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    if (try meta.getKey(u32, .expert_feed_forward_length, false, exc)) |v| hp.n_ff_exp = v;
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, true, exc);

    // Linear attention (gated delta net).
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

    // NextN/MTP: an extra decoder block appended beyond the main stack.
    if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
    impl.assert(hp.n_layer_nextn < hp.n_layer_all, "models/qwen35moe.cpp", 20, "hparams.n_layer_nextn < hparams.n_layer_all && \"n_layer_nextn must be < n_layer_impl\"");

    // Recurrent (linear attention) layers; MTP layers are dense.
    if (!try meta.getKeyOrArr(u32, .attention_recurrent_layers, &hp.is_recr_impl, hp.n_layer_all, false, exc)) {
        var full_attn_interval: u32 = 4;
        if (try meta.getKey(u32, .full_attention_interval, false, exc)) |v| full_attn_interval = v;
        var i: u32 = 0;
        while (i < hp.n_layer_all) : (i += 1) {
            hp.is_recr_impl[i] = @intFromBool(i < hp.nLayer() and model_mod.remU32(i + 1, full_attn_interval) != 0);
        }
    }

    self.type = switch (hp.nLayer()) {
        40 => .@"35b_a3b",
        48 => .@"122b_a10b",
        60 => .@"397b_a17b",
        else => .unknown,
    };
}

/// Ports `llama_model_qwen35moe`'s `load_arch_tensors` (models/qwen35moe.cpp:40 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    const n_layer: i32 = @intCast(hp.nLayer());
    const n_layer_all: i32 = @intCast(hp.n_layer_all);
    const n_head: i64 = hp.nHead(0);
    const n_embd: i64 = hp.n_embd;
    const n_embd_k_gqa: i64 = hp.nEmbdKGqa(0);
    const n_embd_v_gqa: i64 = hp.nEmbdVGqa(0);
    const n_embd_head_k: i64 = hp.nEmbdHeadK(0);
    const n_ff: i64 = hp.nFf(0);
    const n_vocab: i64 = self.vocab.nTokens();
    const n_expert: i64 = hp.n_expert;
    const n_expert_used: i64 = hp.n_expert_used;

    const nr = loader.tensor_not_required;
    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    const mtp_flags: u32 = if (!ml.load_mtp) loader.tensor_skip else 0;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ n_embd, n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < n_layer) : (i += 1) try loadBlockTrunk(self, i, trunk_flags, n_embd, n_head, n_embd_head_k, n_embd_k_gqa, n_embd_v_gqa, n_ff, n_expert, n_expert_used, exc);
    i = n_layer;
    while (i < n_layer_all) : (i += 1) try loadBlockMtp(self, i, mtp_flags, n_embd, n_head, n_embd_head_k, n_embd_k_gqa, n_embd_v_gqa, n_ff, n_expert, n_expert_used, n_vocab, exc);
}

/// `load_block_trunk`, a lambda of `load_arch_tensors` (models/qwen35moe.cpp:40
/// @c1d0e7a00).
fn loadBlockTrunk(self: *Model, il: i32, flags: u32, n_embd: i64, n_head: i64, n_embd_head_k: i64, n_embd_k_gqa: i64, n_embd_v_gqa: i64, n_ff: i64, n_expert: i64, n_expert_used: i64, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    const layer = &self.layers[@intCast(il)];
    const nr = loader.tensor_not_required;

    const n_ff_exp: i64 = if (hp.n_ff_exp != 0) hp.n_ff_exp else divI64(n_ff, n_expert_used);
    const n_ff_shexp: i64 = if (hp.n_ff_shexp != 0) hp.n_ff_shexp else n_ff;

    // Dimensions from the hyperparameters.
    const head_k_dim: i64 = hp.ssm_d_state;
    const head_v_dim: i64 = hp.ssm_d_state;
    const n_k_heads: i64 = hp.ssm_n_group;
    const n_v_heads: i64 = hp.ssm_dt_rank;
    const key_dim = head_k_dim * n_k_heads;
    const value_dim = head_v_dim * n_v_heads;
    const conv_dim = key_dim * 2 + value_dim;

    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
    layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);

    if (!hp.isRecr(@intCast(il))) {
        // Attention layers.
        try self.createTensorQkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ n_embd_head_k * n_head, n_embd }, flags, exc);

        // Q/K normalization.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = il }, &.{n_embd_head_k}, flags, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = il }, &.{n_embd_head_k}, flags, exc);
    } else {
        // Linear attention (gated delta net).
        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = il }, &.{ n_embd, key_dim * 2 + value_dim }, nr, exc);
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = il }, &.{ n_embd, value_dim }, nr, exc);
        layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = il }, &.{ hp.ssm_d_conv, conv_dim }, flags, exc);
        layer.ssm_dt = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = il }, &.{hp.ssm_dt_rank}, flags, exc);
        layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = il }, &.{hp.ssm_dt_rank}, flags, exc);
        layer.ssm_beta = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "weight", .bid = il }, &.{ n_embd, n_v_heads }, flags, exc);
        layer.ssm_alpha = try self.createTensor(.{ .tensor = .ssm_alpha, .suffix = "weight", .bid = il }, &.{ n_embd, n_v_heads }, flags, exc);
        layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = il }, &.{head_v_dim}, flags, exc);
        layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = il }, &.{ value_dim, n_embd }, flags, exc);
    }

    // The routed experts.
    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = il }, &.{ n_embd, n_expert }, flags, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = il }, &.{ n_ff_exp, n_embd, n_expert }, flags, exc);
    try self.createTensorGateUpExps(layer, il, n_embd, n_ff_exp, n_expert, flags, exc);

    // The shared experts.
    layer.ffn_gate_inp_shexp = try self.createTensor(.{ .tensor = .ffn_gate_inp_shexp, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
    layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, flags, exc);
    layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, flags, exc);
    layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = il }, &.{ n_ff_shexp, n_embd }, flags, exc);
}

/// `load_block_mtp`, a lambda of `load_arch_tensors` (models/qwen35moe.cpp:40
/// @c1d0e7a00): the MTP block looks like a full-attention decoder block
/// with an MoE FFN.
fn loadBlockMtp(self: *Model, il: i32, mtp_flags: u32, n_embd: i64, n_head: i64, n_embd_head_k: i64, n_embd_k_gqa: i64, n_embd_v_gqa: i64, n_ff: i64, n_expert: i64, n_expert_used: i64, n_vocab: i64, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    const layer = &self.layers[@intCast(il)];
    const nr = loader.tensor_not_required;

    const n_ff_exp: i64 = if (hp.n_ff_exp != 0) hp.n_ff_exp else divI64(n_ff, n_expert_used);
    const n_ff_shexp: i64 = if (hp.n_ff_shexp != 0) hp.n_ff_shexp else n_ff;

    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags, exc);
    layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags, exc);

    try self.createTensorQkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ n_embd_head_k * n_head, n_embd }, mtp_flags, exc);
    layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = il }, &.{n_embd_head_k}, mtp_flags, exc);
    layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = il }, &.{n_embd_head_k}, mtp_flags, exc);

    // The routed experts.
    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = il }, &.{ n_embd, n_expert }, mtp_flags, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = il }, &.{ n_ff_exp, n_embd, n_expert }, mtp_flags, exc);
    try self.createTensorGateUpExps(layer, il, n_embd, n_ff_exp, n_expert, mtp_flags, exc);

    // The shared experts.
    layer.ffn_gate_inp_shexp = try self.createTensor(.{ .tensor = .ffn_gate_inp_shexp, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags, exc);
    layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, mtp_flags, exc);
    layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, mtp_flags, exc);
    layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = il }, &.{ n_ff_shexp, n_embd }, mtp_flags, exc);

    // The NextN tensors that make the block an MTP block.
    layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = il }, &.{ 2 * n_embd, n_embd }, mtp_flags, exc);
    layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags, exc);
    layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags, exc);
    layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = il }, &.{ n_embd, n_vocab }, mtp_flags | nr, exc);
    layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = il }, &.{ n_embd, n_vocab }, mtp_flags | nr, exc);
    layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags | nr, exc);
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_qwen35moe`'s `build_arch_graph` (models/qwen35moe.cpp:153 @c1d0e7a00):
/// the MTP draft head for that graph type, the main model otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

/// Ports the constructor of `llama_model_qwen35moe`'s `graph` (models/qwen35moe.cpp:160 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/qwen35moe.cpp", 164, "n_embd_head == hparams.n_embd_head_k()");

    var sections: [4]c_int = undefined;
    @memcpy(&sections, hp.rope_sections[0..4]);

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    b.cb(inpL, "model.input_embed", -1);

    const inp = b.buildInpMemHybrid();
    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    // The MTP layers are loaded as extra blocks but not run in the main pass.
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = inpL;

        var inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        c.ggml_build_forward_expand(b.gf, cur);

        if (hp.isRecr(@intCast(il))) {
            // Linear attention (the gated delta net).
            cur = buildLayerAttnLinear(model, b, inp.getRecr(), cur, il);
        } else {
            // Full attention.
            cur = buildLayerAttn(model, b, inp.getAttn(), cur, inp_pos, &sections, il);
        }

        if (il == b.n_layer - 1 and b.cparams.embeddings_nextn_masked) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        // The residual connection.
        cur = c.ggml_add(ctx, cur, inpSA);
        b.cb(cur, "attn_residual", il);

        // Kept for the FFN's residual connection.
        const ffn_residual = cur;

        // The post-attention norm.
        const attn_post_norm = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(attn_post_norm, "attn_post_norm", il);

        // The MoE FFN, without its residual.
        cur = buildLayerFfn(model, b, attn_post_norm, il);
        b.cb(cur, "ffn_out", il);

        // The FFN's residual: what came before the post-attention norm.
        cur = c.ggml_add(ctx, cur, ffn_residual);
        b.cb(cur, "post_moe", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }
    var cur = inpL;

    // The post-norm hidden state feeds both the LM head and the MTP seed.
    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (!b.cparams.embeddings_nextn_masked) cur = c.ggml_get_rows(ctx, cur, inp_out_ids);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_qwen35moe::graph`'s `build_qkvz` (models/qwen35moe.cpp:255 @c1d0e7a00).
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

/// Ports `llama_model_qwen35moe::graph`'s `build_norm_gated` (models/qwen35moe.cpp:271 @c1d0e7a00).
fn buildNormGated(b: *const graph.Builder, input: *Tensor, weights: ?*Tensor, gate: *Tensor, layer: c_int) *Tensor {
    const normalized = b.buildNorm(input, weights, null, .rms, layer);
    const gated_silu = c.ggml_silu(b.ctx0, gate);
    return c.ggml_mul(b.ctx0, normalized, gated_silu);
}

/// Ports `llama_model_qwen35moe::graph`'s `build_layer_attn` (models/qwen35moe.cpp:282
/// @c1d0e7a00): a joint Q and gate projection, Q and K norms, M-RoPE,
/// attention, and the sigmoid gate.
fn buildLayerAttn(model: *const Model, b: *const graph.Builder, inp: *graph.InpAttnKv, cur_: *Tensor, inp_pos: *Tensor, sections: *[4]c_int, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/qwen35moe.cpp", 289, "n_embd_head == hparams.n_embd_head_k()");
    const n_head = b.n_head;
    const n_head_kv = b.n_head_kv;
    const n_tokens = b.n_tokens;

    // One Q projection outputs the query and the gate.
    const Qcur_full = b.buildLoraMm(layer.wq.?, cur_, layer.wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    b.cb(Qcur_full, "Qcur_full", il);

    const es = c.ggml_element_size(Qcur_full);
    const head_sz: usize = @intCast(n_embd_head);
    var Qcur = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, es * head_sz * 2, es * head_sz * 2 * @as(usize, @intCast(n_head)), 0);
    b.cb(Qcur, "Qcur_reshaped", il);

    Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
    b.cb(Qcur, "Qcur_normed", il);

    var Kcur = b.buildLoraMm(layer.wk.?, cur_, layer.wk_s);
    b.cb(Kcur, "Kcur", il);

    var Vcur = b.buildLoraMm(layer.wv.?, cur_, layer.wv_s);
    b.cb(Vcur, "Vcur", il);

    Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
    b.cb(Kcur, "Kcur_normed", il);

    var gate = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, es * head_sz * 2, es * head_sz * 2 * @as(usize, @intCast(n_head)), es * head_sz);
    gate = c.ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
    b.cb(gate, "gate_reshaped", il);

    Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    // M-RoPE.
    Qcur = c.ggml_rope_multi(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_multi(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

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

/// Ports `llama_model_qwen35moe::graph`'s `build_layer_attn_linear` (models/qwen35moe.cpp:363
/// @c1d0e7a00): the gated delta net layer.
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

    impl.assert(n_seqs != 0, "models/qwen35moe.cpp", 377, "n_seqs != 0");
    impl.assert(b.ubatch.equalSeqs(), "models/qwen35moe.cpp", 378, "ubatch.equal_seqs()");
    impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/qwen35moe.cpp", 379, "ubatch.n_tokens == n_seq_tokens * n_seqs");

    // The input projections.
    const qkv_mixed, const z = buildQkvz(model, b, cur_, il);

    var beta = b.buildLoraMm(layer.ssm_beta.?, cur_, layer.ssm_beta_s);
    beta = c.ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    b.cb(beta, "beta", il);

    beta = c.ggml_sigmoid(ctx, beta);
    b.cb(beta, "beta_sigmoid", il);

    var alpha = b.buildLoraMm(layer.ssm_alpha.?, cur_, layer.ssm_alpha_s);
    alpha = c.ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    b.cb(alpha, "alpha", il);

    const alpha_biased = c.ggml_add(ctx, alpha, layer.ssm_dt);
    const alpha_softplus = c.ggml_softplus(ctx, alpha_biased);
    b.cb(alpha_softplus, "a_softplus", il);

    var gate = c.ggml_mul(ctx, alpha_softplus, layer.ssm_a); // -A_log.exp() * softplus
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

    // The whole conv dimension.
    const qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const nb1_qkv = c.ggml_row_size(conv_qkv_mix.*.type, qkv_dim);
    const ty = conv_qkv_mix.*.type;

    // The convolved Q, K and V.
    var q_conv = c.ggml_view_4d(ctx, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_k_dim), nb1_qkv, nb1_qkv * @as(usize, @intCast(n_seq_tokens)), 0);
    var k_conv = c.ggml_view_4d(ctx, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_k_dim), nb1_qkv, nb1_qkv * @as(usize, @intCast(n_seq_tokens)), @as(usize, @intCast(head_k_dim * num_k_heads)) * c.ggml_element_size(conv_qkv_mix));
    const v_conv = c.ggml_view_4d(ctx, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_v_dim), nb1_qkv, nb1_qkv * @as(usize, @intCast(n_seq_tokens)), c.ggml_row_size(ty, 2 * head_k_dim * num_k_heads));

    b.cb(q_conv, "q_conv", il);
    b.cb(k_conv, "k_conv", il);
    b.cb(v_conv, "v_conv", il);

    const eps_norm = hp.f_norm_rms_eps;

    q_conv = c.ggml_l2_norm(ctx, q_conv, eps_norm);
    k_conv = c.ggml_l2_norm(ctx, k_conv, eps_norm);

    // Differing key and value head counts repeat to match, unless the
    // fused delta net handles the broadcast.
    if (num_k_heads != num_v_heads and (!b.cparams.fused_gdn_ar or !b.cparams.fused_gdn_ch)) {
        impl.assert(@rem(num_v_heads, num_k_heads) == 0, "models/qwen35moe.cpp", 466, "num_v_heads % num_k_heads == 0");
        q_conv = c.ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = c.ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    b.cb(q_conv, "q_conv_predelta", il);
    b.cb(k_conv, "k_conv_predelta", il);
    b.cb(v_conv, "v_conv_predelta", il);

    const output = delta_net.buildRecurrentAttn(b, inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    const z_2d = c.ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // The gated norm, self.norm(core_attn_out, z).
    const attn_out_norm = buildNormGated(b, output, layer.ssm_norm, z_2d, il);

    // [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    const final_output = c.ggml_reshape_3d(ctx, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    b.cb(final_output, "final_output", il);

    // The output projection.
    var cur = b.buildLoraMm(layer.ssm_out.?, final_output, layer.ssm_out_s);
    b.cb(cur, "linear_attn_out", il);

    // Back to the original dimensions.
    cur = c.ggml_reshape_2d(ctx, cur, b.n_embd, n_seq_tokens * n_seqs);
    return cur;
}

/// Ports `llama_model_qwen35moe::graph`'s `build_layer_ffn` (models/qwen35moe.cpp:497
/// @c1d0e7a00): the routed experts, plus the shared expert behind its
/// sigmoid gate when there is one.
fn buildLayerFfn(model: *const Model, b: *const graph.Builder, cur_: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    impl.assert(layer.ffn_gate_inp != null, "models/qwen35moe.cpp", 499, "model.layers[il].ffn_gate_inp != nullptr");

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

    // The shared expert, when present, following the Qwen3-Next reference.
    if (layer.ffn_up_shexp == null) return moe_out;

    var ffn_shexp = b.buildFfn(cur_, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
    b.cb(ffn_shexp, "ffn_shexp", il);

    // The shared expert has its own gate, one value per token, sigmoided.
    var shared_gate = b.buildLoraMm(layer.ffn_gate_inp_shexp.?, cur_, null);
    b.cb(shared_gate, "shared_expert_gate", il);

    shared_gate = c.ggml_sigmoid(ctx, shared_gate);
    b.cb(shared_gate, "shared_expert_gate_sigmoid", il);

    // The gate applied to the shared expert's output.
    ffn_shexp = c.ggml_mul(ctx, ffn_shexp, shared_gate);
    b.cb(ffn_shexp, "ffn_shexp_gated", il);

    const cur: *Tensor = c.ggml_add(ctx, moe_out, ffn_shexp);
    b.cb(cur, "ffn_out", il);
    return cur;
}

/// Ports the constructor of `llama_model_qwen35moe`'s `graph_mtp` (models/qwen35moe.cpp:554
/// @c1d0e7a00): the MoE series' MTP draft head.
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, "models/qwen35moe.cpp", 556, "hparams.n_layer_nextn > 0 && \"QWEN35MOE MTP requires n_layer_nextn > 0\"");
    impl.assert(hp.n_layer_nextn == 1, "models/qwen35moe.cpp", 557, "hparams.n_layer_nextn == 1 && \"QWEN35MOE MTP currently only supports a single MTP block\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/qwen35moe.cpp", 560, "n_embd_head == hparams.n_embd_head_k()");

    // The MTP layer is stored right after the main layers.
    const il: c_int = @intCast(hp.nLayer());
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, "models/qwen35moe.cpp", 565, "layer.nextn.eh_proj    && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, "models/qwen35moe.cpp", 566, "layer.nextn.enorm      && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, "models/qwen35moe.cpp", 567, "layer.nextn.hnorm      && \"MTP block missing nextn.hnorm\"");
    impl.assert(layer.ffn_gate_inp != null, "models/qwen35moe.cpp", 568, "layer.ffn_gate_inp     && \"MTP block missing ffn_gate_inp\"");

    var sections: [4]c_int = undefined;
    @memcpy(&sections, hp.rope_sections[0..4]);

    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort("models/qwen35moe.cpp", 0, "out of memory");
    inp.* = .{ .n_embd = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    c.ggml_set_input(inp.embd);

    var tok_embd: *Tensor = undefined;
    if (b.ubatch.token != null) {
        const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;
        tok_embd = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    } else {
        tok_embd = inp.embd.?;
    }
    b.cb(tok_embd, "mtp_tok_embd", il);

    inp.h = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.n_embd, b.n_tokens);
    c.ggml_set_input(inp.h);
    _ = c.ggml_set_name(inp.h, "mtp_h_input");

    const h_embd = inp.h.?;

    _ = b.res.addInput(graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    const inp_attn = b.buildAttnInpKv();

    const h_norm = b.buildNorm(h_embd, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat = c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(cur, "mtp_eh_proj", il);

    const inpSA = cur;

    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    const Qcur_full = b.buildLoraMm(layer.wq.?, cur, layer.wq_s);
    b.cb(Qcur_full, "mtp_Qcur_full", il);

    const es = c.ggml_element_size(Qcur_full);
    const head_sz: usize = @intCast(n_embd_head);
    const n_head: usize = @intCast(b.n_head);

    var Qcur = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, b.n_head, b.n_tokens, es * head_sz * 2, es * head_sz * 2 * n_head, 0);
    Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
    b.cb(Qcur, "mtp_Qcur_normed", il);

    var gate = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, b.n_head, b.n_tokens, es * head_sz * 2, es * head_sz * 2 * n_head, es * head_sz);
    gate = c.ggml_cont_2d(ctx, gate, n_embd_head * b.n_head, b.n_tokens);
    b.cb(gate, "mtp_gate", il);

    var Kcur = b.buildLoraMm(layer.wk.?, cur, layer.wk_s);
    Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
    Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
    b.cb(Kcur, "mtp_Kcur_normed", il);

    var Vcur = b.buildLoraMm(layer.wv.?, cur, layer.wv_s);
    Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);
    b.cb(Vcur, "mtp_Vcur", il);

    Qcur = c.ggml_rope_multi(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), &sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_multi(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), &sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(cur, "mtp_attn_pregate", il);

    cur = c.ggml_mul(ctx, cur, c.ggml_sigmoid(ctx, gate));
    cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
    b.cb(cur, "mtp_attn_out", il);

    cur = c.ggml_add(ctx, cur, inpSA);
    b.cb(cur, "mtp_attn_residual", il);

    const ffn_residual = cur;
    cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_post_norm", il);

    // The MoE FFN: routed experts plus the gated shared expert.
    const moe_out = b.buildMoeFfn(cur, .{
        .gate_inp = layer.ffn_gate_inp,
        .up_exps = layer.ffn_up_exps,
        .gate_exps = layer.ffn_gate_exps,
        .down_exps = layer.ffn_down_exps.?,
        .n_expert = b.n_expert,
        .n_expert_used = b.n_expert_used,
        .type_op = .silu,
        .norm_w = true,
        .w_scale = hp.expert_weights_scale,
        .gating_op = .softmax,
        .gate_up_exps = layer.ffn_gate_up_exps,
        .up_exps_s = layer.ffn_up_exps_s,
        .gate_exps_s = layer.ffn_gate_exps_s,
        .down_exps_s = layer.ffn_down_exps_s,
    }, il);
    b.cb(moe_out, "mtp_ffn_moe_out", il);

    if (layer.ffn_up_shexp != null) {
        var ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
        b.cb(ffn_shexp, "mtp_ffn_shexp", il);

        var shared_gate = b.buildLoraMm(layer.ffn_gate_inp_shexp.?, cur, null);
        shared_gate = c.ggml_sigmoid(ctx, shared_gate);
        b.cb(shared_gate, "mtp_shared_expert_gate_sigmoid", il);

        ffn_shexp = c.ggml_mul(ctx, ffn_shexp, shared_gate);
        b.cb(ffn_shexp, "mtp_ffn_shexp_gated", il);

        cur = c.ggml_add(ctx, moe_out, ffn_shexp);
    } else {
        cur = moe_out;
    }
    b.cb(cur, "mtp_ffn_out", il);

    cur = c.ggml_add(ctx, cur, ffn_residual);
    b.cb(cur, "mtp_post_ffn", il);

    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    impl.assert(head_norm_w != null, "models/qwen35moe.cpp", 725, "head_norm_w && \"QWEN35MOE MTP: missing both nextn.shared_head_norm and output_norm\"");
    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
    impl.assert(head_w != null, "models/qwen35moe.cpp", 736, "head_w && \"QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)\"");
    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
