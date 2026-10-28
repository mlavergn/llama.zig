//! Qwen3-Next: a hybrid of gated-delta-net linear attention and full
//! attention whose feed-forward is a softmax-routed mixture of experts plus
//! a shared expert behind a sigmoid gate, with an optional
//! multi-token-prediction block.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/qwen3next.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_qwen3next`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_qwen3next`'s `load_arch_hparams` (models/qwen3next.cpp:4 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, false, exc);
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    // Linear attention (gated delta net).
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

    // The recurrent (linear attention) layers.
    if (!try meta.getKeyOrArr(u32, .attention_recurrent_layers, &hp.is_recr_impl, hp.n_layer_all, false, exc)) {
        var full_attn_interval: u32 = 4;
        if (try meta.getKey(u32, .full_attention_interval, false, exc)) |v| full_attn_interval = v;
        var i: u32 = 0;
        while (i < hp.n_layer_all) : (i += 1) {
            hp.is_recr_impl[i] = @intFromBool(i < hp.nLayer() and model_mod.remU32(i + 1, full_attn_interval) != 0);
        }
    }

    self.type = switch (hp.nLayer()) {
        48 => .@"80b_a3b",
        else => .unknown,
    };
}

/// The projection sizes `load_arch_tensors` derives once, before its
/// lambdas.
const Dims = struct {
    n_ff_exp: i64,
    head_v_dim: i64,
    key_dim: i64,
    value_dim: i64,
    conv_dim: i64,
    qkvz_dim: i64,
    ba_dim: i64,
};

/// Ports `llama_model_qwen3next`'s `load_arch_tensors` (models/qwen3next.cpp:31 @5266f24da).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    if (l.n_expert == 0) {
        return exc.throw(.runtime_error, "{s} model cannot have zero experts", .{self.archName()});
    }

    const nr = loader.tensor_not_required;
    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    const mtp_flags: u32 = if (!ml.load_mtp) loader.tensor_skip else 0;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    // The dimensions, from the hyperparameters.
    const head_k_dim: i64 = hp.ssm_d_state;
    const head_v_dim: i64 = hp.ssm_d_state;
    const n_k_heads: i64 = hp.ssm_n_group;
    const n_v_heads: i64 = hp.ssm_dt_rank;
    const key_dim = head_k_dim * n_k_heads;
    const value_dim = head_v_dim * n_v_heads;
    const d: Dims = .{
        .n_ff_exp = if (hp.nFfExp(0) != 0) hp.nFfExp(0) else divI64(l.n_ff, l.n_expert_used),
        .head_v_dim = head_v_dim,
        .key_dim = key_dim,
        .value_dim = value_dim,
        .conv_dim = key_dim * 2 + value_dim,
        // The projection sizes.
        .qkvz_dim = key_dim * 2 + value_dim * 2,
        .ba_dim = n_v_heads * 2,
    };

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) try loadBlockTrunk(self, &l, &d, i, trunk_flags, exc);
    i = l.n_layer;
    while (i < l.n_layer_all) : (i += 1) try loadBlockMtp(self, &l, &d, i, mtp_flags, exc);
}

/// `load_block_trunk`, a lambda of `load_arch_tensors` (models/qwen3next.cpp:31
/// @5266f24da).
fn loadBlockTrunk(self: *Model, l: *const Model.Locals, d: *const Dims, il: i32, flags: u32, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    const layer = &self.layers[@intCast(il)];
    const nr = loader.tensor_not_required;
    const n_embd = l.n_embd;
    const n_ff_shexp: i64 = if (hp.n_ff_shexp > 0) hp.n_ff_shexp else hp.nFf(@intCast(il));

    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
    layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);

    if (!hp.isRecr(@intCast(il))) {
        // Attention layers.
        try self.createTensorQkv(layer, il, n_embd, l.n_embd_head_k * l.n_head * 2, l.n_embd_k_gqa, l.n_embd_v_gqa, flags, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ l.n_embd_head_k * l.n_head, n_embd }, flags, exc);
        // Q/K normalization.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = il }, &.{l.n_embd_head_k}, flags, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = il }, &.{l.n_embd_head_k}, flags, exc);
    } else {
        // Linear attention (gated delta net); ssm_in is used by legacy GGUF.
        layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = il }, &.{ n_embd, d.qkvz_dim }, nr | flags, exc);
        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = il }, &.{ n_embd, d.key_dim * 2 + d.value_dim }, nr | flags, exc);
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = il }, &.{ n_embd, d.value_dim }, nr | flags, exc);
        layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = il }, &.{ hp.ssm_d_conv, d.conv_dim }, flags, exc);
        layer.ssm_dt = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = il }, &.{hp.ssm_dt_rank}, flags, exc);
        layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = il }, &.{hp.ssm_dt_rank}, flags, exc);
        layer.ssm_beta_alpha = try self.createTensor(.{ .tensor = .ssm_beta_alpha, .suffix = "weight", .bid = il }, &.{ n_embd, d.ba_dim }, flags, exc);
        layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = il }, &.{d.head_v_dim}, flags, exc);
        layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = il }, &.{ d.value_dim, n_embd }, flags, exc);
    }

    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = il }, &.{ n_embd, l.n_expert }, flags, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = il }, &.{ d.n_ff_exp, n_embd, l.n_expert }, flags, exc);
    try self.createTensorGateUpExps(layer, il, n_embd, d.n_ff_exp, l.n_expert, flags, exc);

    // The shared experts.
    layer.ffn_gate_inp_shexp = try self.createTensor(.{ .tensor = .ffn_gate_inp_shexp, .suffix = "weight", .bid = il }, &.{n_embd}, flags, exc);
    layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, flags, exc);
    layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = il }, &.{ n_embd, n_ff_shexp }, flags, exc);
    layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = il }, &.{ n_ff_shexp, n_embd }, flags, exc);
}

/// `load_block_mtp`, a lambda of `load_arch_tensors` (models/qwen3next.cpp:31
/// @5266f24da): the MTP head is the trunk block (full attention and the
/// FFN) plus the NextN tensors.
fn loadBlockMtp(self: *Model, l: *const Model.Locals, d: *const Dims, il: i32, mtp_flags: u32, exc: *Exception) Thrown!void {
    try loadBlockTrunk(self, l, d, il, mtp_flags, exc);

    const layer = &self.layers[@intCast(il)];
    const nr = loader.tensor_not_required;
    const n_embd = l.n_embd;

    // The NextN tensors that make the block an MTP block.
    layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = il }, &.{ 2 * n_embd, n_embd }, mtp_flags, exc);
    layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags, exc);
    layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags, exc);
    layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = il }, &.{ n_embd, l.n_vocab }, mtp_flags | nr, exc);
    layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = il }, &.{ n_embd, l.n_vocab }, mtp_flags | nr, exc);
    layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = il }, &.{n_embd}, mtp_flags | nr, exc);
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_qwen3next`'s `build_arch_graph` (models/qwen3next.cpp:131
/// @5266f24da): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

/// Ports the constructor of `llama_model_qwen3next`'s `graph` (models/qwen3next.cpp:138 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    b.cb(inpL, "model.embed_tokens", -1);

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
            cur = buildLayerAttn(model, b, inp.getAttn(), cur, inp_pos, il);
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

        // The FFN, MoE or dense, without its residual.
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

    // The post-norm hidden state is the input to both the LM head and the
    // MTP head.
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

/// Ports `llama_model_qwen3next::graph`'s `build_norm_gated` (models/qwen3next.cpp:225 @5266f24da).
fn buildNormGated(b: *const graph.Builder, input: *Tensor, weights: ?*Tensor, gate: *Tensor, layer: c_int) *Tensor {
    const normalized = b.buildNorm(input, weights, null, .rms, layer);
    const gated_silu = c.ggml_silu(b.ctx0, gate);
    return c.ggml_mul(b.ctx0, normalized, gated_silu);
}

/// Ports `llama_model_qwen3next::graph`'s `build_layer_attn` (models/qwen3next.cpp:236
/// @5266f24da): a joint Q and gate projection, the split, Q and K norms,
/// RoPE, attention, and the sigmoid gate.
fn buildLayerAttn(model: *const Model, b: *const graph.Builder, inp: *graph.InpAttnKv, cur_: *Tensor, inp_pos: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/qwen3next.cpp", 242, "n_embd_head == hparams.n_embd_head_k()");
    const n_head = b.n_head;
    const n_head_kv = b.n_head_kv;
    const n_tokens = b.n_tokens;

    // One Q projection outputs the query and the gate.
    var Qcur_full = b.buildLoraMm(layer.wq.?, cur_, layer.wq_s);
    b.cb(Qcur_full, "Qcur_full", il);

    Qcur_full = c.ggml_reshape_4d(ctx, Qcur_full, n_embd_head * 2, n_head, n_tokens, 1);

    // The split into the query and the gate, along the feature dimension.
    var Qcur = c.ggml_view_4d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, 1, Qcur_full.*.nb[1], Qcur_full.*.nb[2], Qcur_full.*.nb[3], 0);
    b.cb(Qcur, "Qcur_view", il);

    var gate = c.ggml_view_4d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, 1, Qcur_full.*.nb[1], Qcur_full.*.nb[2], Qcur_full.*.nb[3], @as(usize, @intCast(n_embd_head)) * c.ggml_element_size(Qcur_full));
    b.cb(gate, "gate", il);

    var Kcur = b.buildLoraMm(layer.wk.?, cur_, layer.wk_s);
    b.cb(Kcur, "Kcur", il);

    var Vcur = b.buildLoraMm(layer.wv.?, cur_, layer.wv_s);
    b.cb(Vcur, "Vcur", il);

    Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
    b.cb(Qcur, "Qcur_normed", il);

    Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
    b.cb(Kcur, "Kcur_normed", il);

    Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    b.cb(Qcur, "Qcur", il);
    b.cb(Kcur, "Kcur", il);
    b.cb(Vcur, "Vcur", il);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    var cur = b.buildAttn(inp, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(cur, "attn_pregate", il);

    // CUDA lacks non-contiguous unary ops, hence the copy.
    gate = c.ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);

    gate = c.ggml_sigmoid(ctx, gate);
    b.cb(gate, "gate_sigmoid", il);

    cur = c.ggml_mul(ctx, cur, gate);
    b.cb(cur, "attn_gated", il);

    cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
    b.cb(cur, "attn_output", il);

    return cur;
}

/// Ports `llama_model_qwen3next::graph`'s `build_qkvz` (models/qwen3next.cpp:314
/// @5266f24da): the separate QKV and gate projections, or the legacy joint
/// one split four ways.
fn buildQkvz(model: *const Model, b: *const graph.Builder, input: *Tensor, il: c_int) struct { *Tensor, *Tensor } {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    const hp = b.hparams;
    const d_inner: i64 = hp.ssm_d_inner;
    const n_seqs: i64 = b.ubatch.n_seqs;
    const head_k_dim: i64 = hp.ssm_d_state;
    const num_k_heads: i64 = hp.ssm_n_group;
    const num_v_heads: i64 = hp.ssm_dt_rank;
    const head_v_dim: i64 = @divTrunc(d_inner, num_v_heads);
    const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

    if (layer.wqkv) |wqkv| {
        // The optimized path.
        var qkv_mixed = b.buildLoraMm(wqkv, input, null);
        qkv_mixed = c.ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed.*.ne[0], n_seq_tokens, n_seqs);
        b.cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        const z = b.buildLoraMm(layer.wqkv_gate.?, input, null);
        b.cb(z, "z", il);

        return .{ qkv_mixed, z };
    }

    // The legacy (slower) path.
    const mixed_qkvz = b.buildLoraMm(layer.ssm_in.?, input, null);
    b.cb(mixed_qkvz, "linear_attn_mixed_qkvz", il);

    const qkvz_new_dim = 2 * head_k_dim + 2 * head_v_dim * @divTrunc(num_v_heads, num_k_heads);
    const mixed_qkvz_reshaped: *Tensor = c.ggml_reshape_4d(ctx, mixed_qkvz, qkvz_new_dim, num_k_heads, n_seq_tokens, n_seqs);

    // The split into query, key, value and z.
    const split_sizes_qkvz = [4]i64{
        head_k_dim, // the query
        head_k_dim, // the key
        @divTrunc(head_v_dim * num_v_heads, num_k_heads), // the value
        @divTrunc(head_v_dim * num_v_heads, num_k_heads), // z
    };
    const nb1 = mixed_qkvz_reshaped.nb[1];
    const nb2 = mixed_qkvz_reshaped.nb[2];
    const nb3 = mixed_qkvz_reshaped.nb[3];
    const es = c.ggml_element_size(mixed_qkvz_reshaped);

    const query = c.ggml_view_4d(ctx, mixed_qkvz_reshaped, split_sizes_qkvz[0], num_k_heads, n_seq_tokens, n_seqs, nb1, nb2, nb3, 0);
    b.cb(query, "q", il);

    const key = c.ggml_view_4d(ctx, mixed_qkvz_reshaped, split_sizes_qkvz[1], num_k_heads, n_seq_tokens, n_seqs, nb1, nb2, nb3, @as(usize, @intCast(split_sizes_qkvz[0])) * es);
    b.cb(key, "k", il);

    const value = c.ggml_view_4d(ctx, mixed_qkvz_reshaped, split_sizes_qkvz[2], num_k_heads, n_seq_tokens, n_seqs, nb1, nb2, nb3, @as(usize, @intCast(split_sizes_qkvz[0] + split_sizes_qkvz[1])) * es);
    b.cb(value, "v", il);

    var z = c.ggml_view_4d(ctx, mixed_qkvz_reshaped, split_sizes_qkvz[3], num_k_heads, n_seq_tokens, n_seqs, nb1, nb2, nb3, @as(usize, @intCast(split_sizes_qkvz[0] + split_sizes_qkvz[1] + split_sizes_qkvz[2])) * es);
    z = c.ggml_cont(ctx, z);
    b.cb(z, "z", il);

    // Each flattened over its heads:
    // [head_dim, num_heads, n_tokens, n_seqs] -> [head_dim * num_heads, n_tokens, n_seqs]
    const query_flat = c.ggml_cont_3d(ctx, query, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
    b.cb(query_flat, "query_flat", il);

    const key_flat = c.ggml_cont_3d(ctx, key, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
    b.cb(key_flat, "key_flat", il);

    const value_flat = c.ggml_cont_3d(ctx, value, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    b.cb(value_flat, "value_flat", il);

    // Concatenated along the feature dimension: [conv_dim, n_tokens, n_seqs].
    var qkv_mixed = c.ggml_concat(ctx, query_flat, key_flat, 0);
    qkv_mixed = c.ggml_concat(ctx, qkv_mixed, value_flat, 0);
    b.cb(qkv_mixed, "qkv_mixed", il);

    return .{ qkv_mixed, z };
}

/// Ports `llama_model_qwen3next::graph`'s `build_layer_attn_linear` (models/qwen3next.cpp:395
/// @5266f24da): the gated delta net layer.
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

    impl.assert(n_seqs != 0, "models/qwen3next.cpp", 409, "n_seqs != 0");
    impl.assert(b.ubatch.equalSeqs(), "models/qwen3next.cpp", 410, "ubatch.equal_seqs()");
    impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/qwen3next.cpp", 411, "ubatch.n_tokens == n_seq_tokens * n_seqs");

    // The input projections.
    const qkv_mixed, const z = buildQkvz(model, b, cur_, il);

    const mixed_ba = b.buildLoraMm(layer.ssm_beta_alpha.?, cur_, null);
    b.cb(mixed_ba, "linear_attn_mixed_ba", il);

    // [batch, seq_len, hidden_size] -> [batch, seq_len, num_k_heads, 2*num_v_heads/num_k_heads]
    const ba_new_dim = @divTrunc(2 * num_v_heads, num_k_heads);
    const mixed_ba_reshaped: *Tensor = c.ggml_reshape_4d(ctx, mixed_ba, ba_new_dim, num_k_heads, n_seq_tokens, n_seqs);

    // The split into beta and alpha.
    const split_sizes_ba = [2]i64{
        @divTrunc(num_v_heads, num_k_heads), // beta
        @divTrunc(num_v_heads, num_k_heads), // alpha
    };

    var bt = c.ggml_view_4d(ctx, mixed_ba_reshaped, split_sizes_ba[0], num_k_heads, n_seq_tokens, n_seqs, mixed_ba_reshaped.nb[1], mixed_ba_reshaped.nb[2], mixed_ba_reshaped.nb[3], 0);
    b.cb(bt, "b", il);

    const a = c.ggml_view_4d(ctx, mixed_ba_reshaped, split_sizes_ba[1], num_k_heads, n_seq_tokens, n_seqs, mixed_ba_reshaped.nb[1], mixed_ba_reshaped.nb[2], mixed_ba_reshaped.nb[3], @as(usize, @intCast(split_sizes_ba[0])) * c.ggml_element_size(mixed_ba_reshaped));
    b.cb(a, "a", il);

    // CUDA lacks non-contiguous unary ops, hence the copy.
    bt = c.ggml_cont(ctx, bt);

    var beta = c.ggml_sigmoid(ctx, bt);

    // a with its head dimensions merged:
    // [batch, seq_len, num_k_heads, num_v_heads/num_k_heads] -> [batch, seq_len, num_v_heads]
    const alpha = c.ggml_cont_3d(ctx, a, num_v_heads, n_seq_tokens, n_seqs);

    const alpha_biased = c.ggml_add(ctx, alpha, layer.ssm_dt);
    const alpha_softplus = c.ggml_softplus(ctx, alpha_biased);
    b.cb(alpha_softplus, "a_softplus", il);

    var gate = c.ggml_mul(ctx, alpha_softplus, layer.ssm_a); // -A_log.exp() * softplus
    b.cb(gate, "gate", il);

    beta = c.ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
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

    // Differing key and value head counts repeat to match.
    if (num_k_heads != num_v_heads) {
        impl.assert(@rem(num_v_heads, num_k_heads) == 0, "models/qwen3next.cpp", 518, "num_v_heads % num_k_heads == 0");
        const repeat_factor = @divTrunc(num_v_heads, num_k_heads);

        // A repeat interleave: reshape to (repeat part, 1, the rest), repeat,
        // reshape back.
        const q_reshaped = c.ggml_reshape_4d(ctx, q_conv, head_k_dim, 1, num_k_heads, n_seq_tokens * n_seqs);
        const k_reshaped = c.ggml_reshape_4d(ctx, k_conv, head_k_dim, 1, num_k_heads, n_seq_tokens * n_seqs);

        // Repeated along the new dimension of size 1.
        const q_repeated = c.ggml_repeat_4d(ctx, q_reshaped, head_k_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs);
        const k_repeated = c.ggml_repeat_4d(ctx, k_reshaped, head_k_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs);

        // [head_dim, repeat_factor, num_k_heads, n_seq_tokens * n_seqs] ->
        // [head_dim, repeat_factor * num_k_heads, n_seq_tokens, n_seqs]
        q_conv = c.ggml_reshape_4d(ctx, q_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
        k_conv = c.ggml_reshape_4d(ctx, k_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
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
    var cur = b.buildLoraMm(layer.ssm_out.?, final_output, null);
    b.cb(cur, "linear_attn_out", il);

    // Back to the original dimensions.
    cur = c.ggml_reshape_2d(ctx, cur, b.n_embd, n_seq_tokens * n_seqs);
    return cur;
}

/// Ports `llama_model_qwen3next::graph`'s `build_layer_ffn` (models/qwen3next.cpp:564
/// @5266f24da): the routed experts plus the gated shared expert, or a
/// dense FFN for a layer without a router.
fn buildLayerFfn(model: *const Model, b: *const graph.Builder, cur_: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];

    if (layer.ffn_gate_inp == null) {
        // The dense branch, not in use.
        const cur = b.buildFfn(cur_, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);
        return cur;
    }

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

    // The shared expert, when present.
    if (layer.ffn_up_shexp == null) return moe_out;

    var ffn_shexp = b.buildFfn(cur_, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
    b.cb(ffn_shexp, "ffn_shexp", il);

    // The shared expert has its own gate, one value per token, sigmoided.
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

/// Ports the constructor of `llama_model_qwen3next`'s `graph_mtp` (models/qwen3next.cpp:627
/// @5266f24da): the MTP draft head.
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, "models/qwen3next.cpp", 629, "hparams.n_layer_nextn > 0 && \"QWEN3NEXT MTP requires n_layer_nextn > 0\"");
    impl.assert(hp.n_layer_nextn == 1, "models/qwen3next.cpp", 630, "hparams.n_layer_nextn == 1 && \"QWEN3NEXT MTP currently only supports a single MTP block\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/qwen3next.cpp", 633, "n_embd_head == hparams.n_embd_head_k()");

    const il: c_int = @intCast(hp.nLayer());
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, "models/qwen3next.cpp", 638, "layer.nextn.eh_proj    && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, "models/qwen3next.cpp", 639, "layer.nextn.enorm      && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, "models/qwen3next.cpp", 640, "layer.nextn.hnorm      && \"MTP block missing nextn.hnorm\"");
    impl.assert(layer.ffn_gate_inp != null, "models/qwen3next.cpp", 641, "layer.ffn_gate_inp     && \"MTP block missing ffn_gate_inp\"");

    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort("models/qwen3next.cpp", 644, "out of memory");
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

    var inpSA = cur;

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

    var Kcur = b.buildLoraMm(layer.wk.?, cur, layer.wk_s);
    Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
    Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
    b.cb(Kcur, "mtp_Kcur_normed", il);

    var Vcur = b.buildLoraMm(layer.wv.?, cur, layer.wv_s);
    Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

    Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    b.cb(Qcur, "mtp_Qcur", il);
    b.cb(Kcur, "mtp_Kcur", il);
    b.cb(Vcur, "mtp_Vcur", il);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(cur, "mtp_attn_pregate", il);

    var gate = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, b.n_head, b.n_tokens, es * head_sz * 2, es * head_sz * 2 * n_head, es * head_sz);

    // CUDA lacks non-contiguous unary ops, hence the copy.
    gate = c.ggml_cont_2d(ctx, gate, n_embd_head * b.n_head, b.n_tokens);
    b.cb(gate, "mtp_gate", il);

    cur = c.ggml_mul(ctx, cur, c.ggml_sigmoid(ctx, gate));
    cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
    b.cb(cur, "mtp_attn_out", il);

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);

    cur = c.ggml_add(ctx, cur, inpSA);
    b.cb(cur, "mtp_attn_residual", il);

    const ffn_residual = cur;
    cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_post_norm", il);

    // The MoE FFN: routed experts plus the gated shared expert, as in the
    // trunk.
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
    impl.assert(head_norm_w != null, "models/qwen3next.cpp", 805, "head_norm_w && \"QWEN3NEXT MTP: missing both nextn.shared_head_norm and output_norm\"");
    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
    impl.assert(head_w != null, "models/qwen3next.cpp", 813, "head_w && \"QWEN3NEXT MTP: missing LM head (nextn.shared_head_head or model.output)\"");
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
