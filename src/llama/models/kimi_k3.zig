//! Kimi-K3: Kimi Linear's hybrid of Kimi delta attention and MLA, with
//! five parts Kimi Linear does not have -- cross-layer residual attention
//! (`attn_res_block_size`), a latent MoE whose routed experts run at
//! `n_expert_latent`, the situ activation in place of SwiGLU everywhere, a
//! sigmoid output gate on MLA, and a single full-rank KDA output gate.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/kimi-k3.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_kimi_k3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, `kimi_k3_situ`, `kimi_k3_conv1d`, and its `graph`
//! with `res_push`, `res_mix`, `build_kda_layer`, `build_mla_layer` and
//! `build_latent_moe`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const delta_net = @import("delta_net.zig");
const arch = @import("../arch.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Layer = model_mod.Layer;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_kimi_k3`'s `load_arch_hparams` (models/kimi-k3.cpp:14 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_embd_head_k_mla_impl = (try meta.getKey(u32, .attention_key_length_mla, true, exc)).?;
    hp.n_embd_head_v_mla_impl = (try meta.getKey(u32, .attention_value_length_mla, true, exc)).?;
    if (try meta.getKey(u32, .attention_q_lora_rank, false, exc)) |v| hp.n_lora_q = v;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.n_embd_head_kda = (try meta.getKey(u32, .kda_head_dim, true, exc)).?;
    if (try meta.getKey(f32, .kda_gate_lower_bound, false, exc)) |v| hp.kda_gate_lower_bound = v;

    // The MLA cache holds the compressed latent; set here too, since older
    // GGUFs have no value_length key.
    hp.n_embd_head_v_full = hp.n_lora_kv;

    // n_head_kv == 0 marks a KDA (recurrent) layer, as in Kimi Linear.
    var i: u32 = 0;
    while (i < hp.nLayer()) : (i += 1) {
        hp.is_recr_impl[i] = @intFromBool(hp.nHeadKv(i) == 0);
    }

    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;
    if (try meta.getKey(u32, .expert_latent_length, false, exc)) |v| hp.n_expert_latent = v;

    hp.attn_res_block_size = (try meta.getKey(u32, .attn_res_block_size, true, exc)).?;
    hp.situ_beta = (try meta.getKey(f32, .activation_situ_beta, true, exc)).?;
    hp.situ_linear_beta = (try meta.getKey(f32, .activation_situ_linear_beta, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        93 => .@"2_8t_a50b", // Kimi-K3
        else => .unknown,
    };
}

/// Ports `llama_model_kimi_k3`'s `load_arch_tensors` (models/kimi-k3.cpp:51 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const n_embd = l.n_embd;
    const n_head = l.n_head;

    const n_embd_latent: i64 = if (hp.n_expert_latent > 0) hp.n_expert_latent else n_embd;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, 0, exc);

    if (hp.attn_res_block_size > 0) {
        self.output_res_score = try self.createTensor(.{ .tensor = .output_res_score, .suffix = "weight" }, &.{n_embd}, 0, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);

        if (hp.attn_res_block_size > 0) {
            layer.attn_res_score = try self.createTensor(.{ .tensor = .attn_res_score, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);
            layer.ffn_res_score = try self.createTensor(.{ .tensor = .ffn_res_score, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);
        }

        const head_dim: i64 = hp.n_embd_head_kda;
        const d_conv: i64 = hp.ssm_d_conv;
        const d_inner = head_dim * n_head;

        if (hp.isRecr(@intCast(i))) {
            // The conv1d weights may be 4D, [d_conv, 1, d_inner, 1], or 3D,
            // when quantization drops the trailing 1.
            layer.ssm_q_conv = try loadConv(self, .ssm_conv1d_q, i, d_conv, d_inner, exc);
            layer.ssm_k_conv = try loadConv(self, .ssm_conv1d_k, i, d_conv, d_inner, exc);
            layer.ssm_v_conv = try loadConv(self, .ssm_conv1d_v, i, d_conv, d_inner, exc);

            try self.createTensorQkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0, exc);

            layer.ssm_f_a = try self.createTensor(.{ .tensor = .ssm_f_a, .suffix = "weight", .bid = i }, &.{ n_embd, head_dim }, 0, exc);
            layer.ssm_f_b = try self.createTensor(.{ .tensor = .ssm_f_b, .suffix = "weight", .bid = i }, &.{ head_dim, d_inner }, 0, exc);
            layer.ssm_beta = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "weight", .bid = i }, &.{ n_embd, n_head }, 0, exc);

            // K3's A_log is a plain 1-D [n_head] tensor; Kimi Linear's is padded.
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = i }, &.{n_head}, 0, exc);
            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{d_inner}, 0, exc);

            // K3 has one full-rank gate in place of Kimi Linear's g_a/g_b pair.
            layer.ssm_g = try self.createTensor(.{ .tensor = .ssm_g, .suffix = "weight", .bid = i }, &.{ n_embd, d_inner }, 0, exc);
            layer.ssm_o_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = i }, &.{head_dim}, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ d_inner, n_embd }, 0, exc);
        } else {
            const q_lora_rank: i64 = hp.n_lora_q;
            const kv_lora_rank: i64 = hp.n_lora_kv;
            const n_embd_head_k: i64 = hp.nEmbdHeadKMla();
            const n_embd_head_v: i64 = hp.nEmbdHeadVMla();
            const qk_rope_head_dim: i64 = hp.nRot(0);
            const qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

            layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, nr, exc);
            layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, 0, exc);

            if (layer.attn_q_a_norm != null) {
                layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ n_embd, q_lora_rank }, 0, exc);
                layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, n_head * n_embd_head_k }, 0, exc);
            } else {
                layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ n_embd, n_head * n_embd_head_k }, 0, exc);
            }

            layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ n_embd, kv_lora_rank + qk_rope_head_dim }, 0, exc);
            layer.wkv_b = try self.createTensor(.{ .tensor = .attn_kv_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_head * (qk_nope_head_dim + n_embd_head_v) }, nr | loader.tensor_skip_if_virtual, exc);
            if (layer.wkv_b == null) {
                layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ qk_nope_head_dim, kv_lora_rank, n_head }, 0, exc);
                layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v, n_head }, 0, exc);
            }

            // K3's sigmoid output gate on the attention output, before o_proj.
            layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ n_embd, n_head * n_embd_head_v }, nr, exc);

            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_head * n_embd_head_v, n_embd }, 0, exc);
        }

        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_ff }, 0, exc);
        } else {
            const n_ff_exp: i64 = hp.n_ff_exp;

            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);

            // The routed experts live in the latent space.
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ n_embd_latent, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, n_embd_latent, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ n_embd_latent, n_ff_exp, l.n_expert }, 0, exc);

            if (hp.n_expert_latent > 0) {
                layer.ffn_routed_down = try self.createTensor(.{ .tensor = .ffn_routed_down, .suffix = "weight", .bid = i }, &.{ n_embd, n_embd_latent }, 0, exc);
                layer.ffn_routed_up = try self.createTensor(.{ .tensor = .ffn_routed_up, .suffix = "weight", .bid = i }, &.{ n_embd_latent, n_embd }, 0, exc);
                layer.ffn_routed_norm = try self.createTensor(.{ .tensor = .ffn_routed_norm, .suffix = "weight", .bid = i }, &.{n_embd_latent}, nr, exc);
            }

            // The shared experts stay at n_embd, moe_intermediate_size *
            // n_expert_shared wide.
            const n_ff_shexp = n_ff_exp * @as(i64, if (hp.n_expert_shared > 0) hp.n_expert_shared else 1);
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_shexp }, nr, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, n_embd }, nr, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_shexp }, nr, exc);
        }
    }
}

/// `conv`, a lambda of `load_arch_tensors` (models/kimi-k3.cpp:51
/// @c1d0e7a00): a conv1d weight, 4D or else 3D.
fn loadConv(self: *Model, tid: arch.Tensor, il: i32, d_conv: i64, d_inner: i64, exc: *Exception) Thrown!?*Tensor {
    const t = try self.createTensor(.{ .tensor = tid, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner, 1 }, loader.tensor_not_required, exc);
    return t orelse try self.createTensor(.{ .tensor = tid, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner }, 0, exc);
}

/// Ports `llama_model_kimi_k3`'s `build_arch_graph` (models/kimi-k3.cpp:167 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    var g: Graph = .{ .model = model, .b = b };
    g.build();
}

/// Ports `kimi_k3_situ` (models/kimi-k3.cpp:173 @c1d0e7a00):
/// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) *
/// linear_beta*tanh(up/linear_beta); a linear_beta <= 0 leaves the up
/// branch as it is.
fn situ(ctx: *c.ggml_context, gate: *Tensor, up_: *Tensor, beta: f32, linear_beta: f32) *Tensor {
    var up = up_;
    var a: *Tensor = c.ggml_scale(ctx, c.ggml_tanh(ctx, c.ggml_scale(ctx, gate, 1.0 / beta)), beta);
    a = c.ggml_mul(ctx, a, c.ggml_sigmoid(ctx, gate));

    if (linear_beta > 0.0) {
        up = c.ggml_scale(ctx, c.ggml_tanh(ctx, c.ggml_scale(ctx, up, 1.0 / linear_beta)), linear_beta);
    }
    return c.ggml_mul(ctx, a, up);
}

/// Ports `kimi_k3_conv1d` (models/kimi-k3.cpp:356 @c1d0e7a00): the causal
/// conv1d over one of Q, K or V; `qkv` picks which third of the conv state.
fn conv1d(gf: *c.ggml_cgraph, ctx: *c.ggml_context, conv_states_all: *Tensor, conv_state_all: *Tensor, qkv: i64, x: *Tensor, proj_w: ?*Tensor, conv_w: ?*Tensor, d_conv: i64, head_dim: i64, n_head: i64, n_seq_tokens: i64, n_seqs: i64, n_tokens: i64, kv_head: i64) *Tensor {
    const d_inner = head_dim * n_head;
    const conv_state_size = (d_conv - 1) * d_inner;
    const n_embd_r_total = 3 * conv_state_size;

    const es_state: usize = c.ggml_element_size(conv_state_all);
    const conv_state_x = c.ggml_view_3d(ctx, conv_state_all, d_conv - 1, d_inner, n_seqs, @as(usize, @intCast(d_conv - 1)) * es_state, @as(usize, @intCast(n_embd_r_total)) * es_state, @as(usize, @intCast(qkv * conv_state_size)) * es_state);

    const x_proj = c.ggml_mul_mat(ctx, proj_w, x);
    const x_3d = c.ggml_reshape_3d(ctx, x_proj, d_inner, n_seq_tokens, n_seqs);
    const conv_x: *Tensor = c.ggml_concat(ctx, conv_state_x, c.ggml_transpose(ctx, x_3d), 0);

    const last_conv_x = c.ggml_view_3d(ctx, conv_x, d_conv - 1, d_inner, n_seqs, conv_x.nb[1], conv_x.nb[2], @as(usize, @intCast(n_seq_tokens)) * conv_x.nb[0]);
    const es_all: usize = c.ggml_element_size(conv_states_all);
    c.ggml_build_forward_expand(gf, c.ggml_cpy(ctx, last_conv_x, c.ggml_view_3d(ctx, conv_states_all, d_conv - 1, d_inner, n_seqs, @as(usize, @intCast(d_conv - 1)) * es_all, @as(usize, @intCast(n_embd_r_total)) * es_all, @as(usize, @intCast(kv_head * n_embd_r_total + qkv * conv_state_size)) * es_all)));

    const conv_weight = c.ggml_reshape_2d(ctx, conv_w, d_conv, d_inner);
    var Xcur = c.ggml_ssm_conv(ctx, conv_x, conv_weight);
    Xcur = c.ggml_reshape_2d(ctx, Xcur, d_inner, n_tokens);
    Xcur = c.ggml_silu(ctx, Xcur);

    return c.ggml_reshape_4d(ctx, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

/// `llama_model_kimi_k3`'s `graph`: the builder, and the stack of banked
/// residual checkpoints the cross-layer residual attention reads.
const Graph = struct {
    model: *const Model,
    b: *const graph.Builder,
    /// The banked checkpoints, `[n_embd, n_ckpt, n_tokens]`.
    resi_stack: ?*Tensor = null,

    /// Ports `llama_model_kimi_k3::graph`'s `res_push` (models/kimi-k3.cpp:190
    /// @c1d0e7a00): append the new checkpoint without refolding the chain.
    /// The layout is `[n_embd, n_ckpt, n_tokens]`: rms_norm reduces over
    /// ne0, dsv4_hc_pre over ne1.
    fn resPush(self: *Graph, cur: *Tensor, n_embd: i64, n_tokens: i64) void {
        const ctx = self.b.ctx0;
        const ckpt: *Tensor = c.ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

        self.resi_stack = if (self.resi_stack) |s| c.ggml_concat(ctx, s, ckpt, 1) else ckpt;
    }

    /// Ports `llama_model_kimi_k3::graph`'s `res_mix` (models/kimi-k3.cpp:196
    /// @c1d0e7a00): a softmax over the banked checkpoints and the current
    /// stream, scored by `score_w`, and their weighted sum.
    fn resMix(self: *const Graph, cur: *Tensor, score_w: ?*Tensor, n_tokens: i64, il: c_int) *Tensor {
        const ctx = self.b.ctx0;
        // Layer 0: nothing banked yet.
        const src = self.resi_stack orelse return cur; // [n_embd, n_ckpt, n_tokens]

        const n_ckpt: c_int = @intCast(src.ne[1]);
        const eps = self.b.hparams.f_norm_rms_eps;

        // One rms_norm scores every checkpoint; the scores use the
        // normalized values, the sum below the raw ones.
        var sc_src: *Tensor = c.ggml_rms_norm(ctx, src, eps);
        sc_src = c.ggml_mul(ctx, sc_src, score_w);
        sc_src = c.ggml_sum_rows(ctx, sc_src); // [1, n_ckpt, n_tokens]
        sc_src = c.ggml_reshape_2d(ctx, sc_src, n_ckpt, n_tokens);

        // The current stream is scored apart, so the stack stays append-only.
        var sc_cur: *Tensor = c.ggml_rms_norm(ctx, cur, eps);
        sc_cur = c.ggml_mul(ctx, sc_cur, score_w);
        sc_cur = c.ggml_sum_rows(ctx, sc_cur); // [1, n_tokens]

        const scores = c.ggml_concat(ctx, sc_src, sc_cur, 0); // [n_ckpt+1, n_tokens]
        const probs: *Tensor = c.ggml_soft_max(ctx, scores); // over ne0 = n_ckpt+1
        self.b.cb(probs, "res_probs", il);

        // The sum split: hc_pre takes the stack, a broadcast multiply the
        // current stream.
        const p_src = c.ggml_cont(ctx, c.ggml_view_2d(ctx, probs, n_ckpt, n_tokens, probs.nb[1], 0));
        const p_cur = c.ggml_cont(ctx, c.ggml_view_2d(ctx, probs, 1, n_tokens, probs.nb[1], probs.nb[0] * @as(usize, @intCast(n_ckpt))));

        var out = c.ggml_dsv4_hc_pre(ctx, src, p_src);
        out = c.ggml_add(ctx, out, c.ggml_mul(ctx, cur, p_cur));

        return out;
    }

    /// Ports the constructor of `llama_model_kimi_k3`'s `graph`
    /// (models/kimi-k3.cpp:234 @c1d0e7a00).
    fn build(self: *Graph) void {
        const model = self.model;
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_embd: i64 = b.n_embd;
        const n_tokens = b.n_tokens;

        var inpL = b.buildInpEmbd(model.tok_embd.?);
        b.cb(inpL, "inp_embd", -1);

        // K3's MLA is nope-only, so there is no position input.

        const is_mla = hp.isMla();
        const inp_kv = if (!is_mla) b.buildInpMemHybrid() else null;
        const inp_k = if (is_mla) b.buildInpMemHybridK() else null;
        const inp_rs = if (is_mla) inp_k.?.getRecr() else inp_kv.?.getRecr();
        const inp_attn_kv = if (!is_mla) inp_kv.?.getAttn() else null;
        const inp_attn_k = if (is_mla) inp_k.?.getAttn() else null;

        const inp_out_ids = b.buildInpOutIds();

        const n_head_kda: i64 = hp.nHead(0);
        const head_dim: i64 = hp.n_embd_head_kda;
        const d_conv: i64 = hp.ssm_d_conv;
        const d_inner = n_head_kda * head_dim;
        const n_seqs: i64 = b.ubatch.n_seqs;
        const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

        impl.assert(n_seqs != 0, "models/kimi-k3.cpp", 260, "n_seqs != 0");
        impl.assert(b.ubatch.equalSeqs(), "models/kimi-k3.cpp", 261, "ubatch.equal_seqs()");
        impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/kimi-k3.cpp", 262, "ubatch.n_tokens == n_seq_tokens * n_seqs");

        const mla: Mla = .{
            .n_embd_head_k_mla = hp.nEmbdHeadKMla(),
            .n_embd_head_v_mla = hp.nEmbdHeadVMla(),
            .kv_lora_rank = hp.n_lora_kv,
            .n_embd_head_qk_rope = hp.nRot(0),
            .n_embd_head_qk_nope = @as(i64, hp.nEmbdHeadKMla()) - hp.nRot(0),
            .kq_scale = 1.0 / impl.sqrtf(@floatFromInt(hp.nEmbdHeadKMla())),
        };

        const res_bs: u32 = hp.attn_res_block_size;
        const use_attn_res = res_bs > 0;
        const n_embd_latent: i64 = if (hp.n_expert_latent > 0) hp.n_expert_latent else n_embd;

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];

            // The residual stream, banked on checkpoint layers and then
            // restarted from the attention output alone.
            var prefix_sum = inpL;

            var cur = if (use_attn_res) self.resMix(prefix_sum, layer.attn_res_score, n_tokens, il) else prefix_sum;

            var banked = false;
            if (use_attn_res and @as(u32, @intCast(il)) % res_bs == 0) {
                self.resPush(prefix_sum, n_embd, n_tokens); // banks the RAW layer input, not `cur`
                banked = true;
            }

            cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);
            c.ggml_build_forward_expand(b.gf, cur);

            if (hp.isRecr(@intCast(il))) {
                cur = self.buildKdaLayer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda, d_inner, n_seq_tokens, n_seqs, il);
            } else {
                cur = self.buildMlaLayer(cur, layer, inp_attn_k, inp_attn_kv, &mla, il);
            }

            prefix_sum = if (banked) cur else c.ggml_add(ctx, prefix_sum, cur);
            b.cb(prefix_sum, "prefix_sum_attn", il);

            cur = if (use_attn_res) self.resMix(prefix_sum, layer.ffn_res_score, n_tokens, il) else prefix_sum;

            cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
                const g = c.ggml_mul_mat(ctx, layer.ffn_gate, cur);
                const u = c.ggml_mul_mat(ctx, layer.ffn_up, cur);
                cur = situ(ctx, g, u, hp.situ_beta, hp.situ_linear_beta);
                cur = c.ggml_mul_mat(ctx, layer.ffn_down, cur);
                b.cb(cur, "ffn_out", il);
            } else {
                cur = self.buildLatentMoe(cur, layer, n_embd_latent, il);
            }

            prefix_sum = c.ggml_add(ctx, prefix_sum, cur);
            prefix_sum = b.buildCvec(prefix_sum, il);
            b.cb(prefix_sum, "l_out", il);

            inpL = prefix_sum;
        }

        var cur = inpL;

        // The final mix, then only the output tokens.
        if (use_attn_res) {
            cur = self.resMix(cur, model.output_res_score, n_tokens, -1);
        }
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);

        cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        cur = c.ggml_mul_mat(ctx, model.output, cur);
        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;

        c.ggml_build_forward_expand(b.gf, cur);
    }

    /// Ports `llama_model_kimi_k3::graph`'s `build_kda_layer`
    /// (models/kimi-k3.cpp:391 @c1d0e7a00).
    fn buildKdaLayer(self: *const Graph, cur_: *Tensor, layer: *const Layer, inp_rs: *graph.InpRs, d_conv: i64, head_dim: i64, n_head_kda: i64, d_inner: i64, n_seq_tokens: i64, n_seqs: i64, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_tokens = b.n_tokens;
        var cur = cur_;

        const mctx_cur = inp_rs.mctx;
        const kv_head: i64 = mctx_cur.getHead();

        const conv_states_all = mctx_cur.getRL(il);
        const conv_state_all = b.buildRs(inp_rs, conv_states_all, @intCast(hp.nEmbdR()), @intCast(n_seqs), null);

        var Qcur = conv1d(b.gf, ctx, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
        var Kcur = conv1d(b.gf, ctx, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
        const Vcur = conv1d(b.gf, ctx, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
        b.cb(Qcur, "kda_q_conv", il);
        b.cb(Kcur, "kda_k_conv", il);
        b.cb(Vcur, "kda_v_conv", il);

        // gate_lower_bound is not a clamp: set, it swaps the decay gate's
        // activation --
        //   unset (Kimi Linear): g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
        //   set   (K3, -5.0):    g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
        // ssm_a holds -exp(A_log), folded at conversion, so exp(A_log) == -ssm_a.
        const f_a = c.ggml_mul_mat(ctx, layer.ssm_f_a, cur);
        var g1 = c.ggml_mul_mat(ctx, layer.ssm_f_b, f_a);
        g1 = c.ggml_add(ctx, g1, layer.ssm_dt_b);

        const A = c.ggml_reshape_3d(ctx, layer.ssm_a, 1, n_head_kda, 1);

        if (hp.kda_gate_lower_bound > -std.math.inf(f32)) {
            g1 = c.ggml_reshape_3d(ctx, g1, head_dim, n_head_kda, n_tokens);
            g1 = c.ggml_mul(ctx, g1, A); // -exp(A_log) * (...)
            g1 = c.ggml_sigmoid(ctx, c.ggml_scale(ctx, g1, -1.0));
            g1 = c.ggml_scale(ctx, g1, hp.kda_gate_lower_bound);
        } else {
            g1 = c.ggml_softplus(ctx, g1);
            g1 = c.ggml_reshape_3d(ctx, g1, head_dim, n_head_kda, n_tokens);
            g1 = c.ggml_mul(ctx, g1, A);
        }
        b.cb(g1, "kda_g1", il);

        g1 = c.ggml_reshape_4d(ctx, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

        var beta = c.ggml_mul_mat(ctx, layer.ssm_beta, cur);
        beta = c.ggml_reshape_4d(ctx, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
        beta = c.ggml_sigmoid(ctx, beta);
        b.cb(beta, "kda_beta", il);

        const cur_3d: *Tensor = c.ggml_reshape_3d(ctx, cur, cur.ne[0], n_seq_tokens, n_seqs);

        const ssm_states_all = mctx_cur.getSL(il);
        var state = b.buildRs(inp_rs, ssm_states_all, @intCast(hp.nEmbdS()), @intCast(n_seqs), null);
        state = c.ggml_reshape_4d(ctx, state, head_dim, head_dim, n_head_kda, n_seqs);

        const eps = hp.f_norm_rms_eps;
        Qcur = c.ggml_l2_norm(ctx, Qcur, eps);
        Kcur = c.ggml_l2_norm(ctx, Kcur, eps);

        const attn_out = delta_net.buildDeltaNet(b, Qcur, Kcur, Vcur, g1, beta, state, il);

        const output = c.ggml_cont(ctx, attn_out.out);
        b.cb(output, "kda_scan_out", il);
        const new_state = attn_out.state;

        const n_embd_s: i64 = hp.nEmbdS();
        c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, new_state, c.ggml_view_1d(ctx, ssm_states_all, n_embd_s * n_seqs, @as(usize, @intCast(kv_head * n_embd_s)) * c.ggml_element_size(ssm_states_all))));

        // K3's single full-rank gate; Kimi Linear factors it as g_b(g_a(x)).
        const cur_2d = c.ggml_reshape_2d(ctx, cur_3d, cur_3d.ne[0], n_seq_tokens * n_seqs);
        var g2 = c.ggml_mul_mat(ctx, layer.ssm_g, cur_2d);
        g2 = c.ggml_reshape_3d(ctx, g2, head_dim, n_head_kda, n_seq_tokens * n_seqs);

        const o = c.ggml_reshape_3d(ctx, output, head_dim, n_head_kda, n_seq_tokens * n_seqs);
        const normed = b.buildNorm(o, layer.ssm_o_norm, null, .rms, il);
        b.cb(g2, "kda_g2", il);
        b.cb(normed, "kda_normed", il);
        var gated = c.ggml_mul(ctx, normed, c.ggml_sigmoid(ctx, g2));

        gated = c.ggml_cont_2d(ctx, gated, d_inner, n_tokens);
        cur = c.ggml_mul_mat(ctx, layer.wo, gated);
        b.cb(cur, "kda_out", il);

        return cur;
    }

    /// The MLA dimensions the constructor hands `build_mla_layer`.
    const Mla = struct {
        n_embd_head_k_mla: i64,
        n_embd_head_v_mla: i64,
        kv_lora_rank: i64,
        n_embd_head_qk_rope: i64,
        n_embd_head_qk_nope: i64,
        kq_scale: f32,
    };

    /// Ports `llama_model_kimi_k3::graph`'s `build_mla_layer`
    /// (models/kimi-k3.cpp:481 @c1d0e7a00): nope-only MLA, with K3's
    /// sigmoid output gate.
    fn buildMlaLayer(self: *const Graph, cur: *Tensor, layer: *const Layer, inp_attn_k: ?*graph.InpAttnK, inp_attn_kv: ?*graph.InpAttnKv, d: *const Mla, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const n_tokens = b.n_tokens;
        const n_head: i64 = b.n_head;

        const inp_gate = cur; // the output gate reads the *normed* layer input

        var Qcur: *Tensor = undefined;
        if (layer.wq_a != null) {
            Qcur = c.ggml_mul_mat(ctx, layer.wq_a, cur);
            Qcur = b.buildNorm(Qcur, layer.attn_q_a_norm, null, .rms, il);
            Qcur = c.ggml_mul_mat(ctx, layer.wq_b, Qcur);
        } else {
            Qcur = c.ggml_mul_mat(ctx, layer.wq, cur);
        }

        const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);

        var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, d.kv_lora_rank, n_tokens, c.ggml_row_size(kv_cmpr_pe.type, d.kv_lora_rank + d.n_embd_head_qk_rope), 0);
        const k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, d.n_embd_head_qk_rope, 1, n_tokens, c.ggml_row_size(kv_cmpr_pe.type, d.kv_lora_rank + d.n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, d.kv_lora_rank + d.n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, d.kv_lora_rank));

        // No RoPE: mla_use_nope is asserted at conversion.
        kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);

        var out: *Tensor = undefined;
        if (layer.wk_b != null and layer.wv_b != null) {
            var q_nope: *Tensor = c.ggml_view_3d(ctx, Qcur, d.n_embd_head_qk_nope, n_head, n_tokens, c.ggml_row_size(Qcur.type, d.n_embd_head_k_mla), c.ggml_row_size(Qcur.type, d.n_embd_head_k_mla) * @as(usize, @intCast(n_head)), 0);
            const q_pe: *Tensor = c.ggml_view_3d(ctx, Qcur, d.n_embd_head_qk_rope, n_head, n_tokens, c.ggml_row_size(Qcur.type, d.n_embd_head_k_mla), c.ggml_row_size(Qcur.type, d.n_embd_head_k_mla) * @as(usize, @intCast(n_head)), c.ggml_row_size(Qcur.type, d.n_embd_head_qk_nope));

            q_nope = c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
            var q_nope_absorbed: *Tensor = c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
            q_nope_absorbed = c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);

            const Q: *Tensor = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
            const kv_cmpr_3d: *Tensor = c.ggml_reshape_3d(ctx, kv_cmpr, d.kv_lora_rank, 1, n_tokens);
            const K: *Tensor = c.ggml_concat(ctx, kv_cmpr_3d, k_pe, 0);
            const V = kv_cmpr_3d;

            // No wo: the output projection comes after the gate below.
            out = b.buildAttn(inp_attn_k.?, null, null, null, Q, K, V, null, null, layer.wv_b, d.kq_scale, il);
        } else {
            const Q: *Tensor = c.ggml_reshape_3d(ctx, Qcur, d.n_embd_head_k_mla, n_head, n_tokens);
            const kv: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_b, kv_cmpr);
            const kv_per_head = d.n_embd_head_qk_nope + d.n_embd_head_v_mla;

            const k_nope: *Tensor = c.ggml_view_3d(ctx, kv, d.n_embd_head_qk_nope, n_head, n_tokens, c.ggml_row_size(kv.type, kv_per_head), c.ggml_row_size(kv.type, kv_per_head * n_head), 0);
            const V: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, kv, d.n_embd_head_v_mla, n_head, n_tokens, c.ggml_row_size(kv.type, kv_per_head), c.ggml_row_size(kv.type, kv_per_head * n_head), c.ggml_row_size(kv.type, d.n_embd_head_qk_nope)));

            const k_pe_t = c.ggml_new_tensor_3d(ctx, k_pe.type, d.n_embd_head_qk_rope, n_head, n_tokens);
            const K: *Tensor = c.ggml_concat(ctx, c.ggml_repeat(ctx, k_pe, k_pe_t), k_nope, 0);

            out = b.buildAttn(inp_attn_kv.?, null, null, null, Q, K, V, null, null, null, d.kq_scale, il);
        }

        // K3: attn_output *= sigmoid(g_proj(x)), then o_proj.
        if (layer.wqkv_gate != null) {
            const g = c.ggml_sigmoid(ctx, c.ggml_mul_mat(ctx, layer.wqkv_gate, inp_gate));
            out = c.ggml_mul(ctx, out, g);
            b.cb(out, "mla_gated", il);
        }

        out = c.ggml_mul_mat(ctx, layer.wo, out);
        b.cb(out, "mla_out", il);

        return out;
    }

    /// Ports `llama_model_kimi_k3::graph`'s `build_latent_moe`
    /// (models/kimi-k3.cpp:566 @c1d0e7a00): down-project, run the routed
    /// experts in the latent space, norm, up-project; the shared experts
    /// stay at n_embd and read the input as it came.
    fn buildLatentMoe(self: *const Graph, cur: *Tensor, layer: *const Layer, n_embd_latent: i64, il: c_int) *Tensor {
        _ = n_embd_latent;
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;

        const identity = cur;

        const routed_in: *Tensor = if (layer.ffn_routed_down) |w| c.ggml_mul_mat(ctx, w, cur) else cur;

        // The router scores the full-width input while the experts take the
        // latent one, so the logits are computed here and handed to
        // build_moe_ffn.
        const logits: *Tensor = c.ggml_mul_mat(ctx, layer.ffn_gate_inp, identity);
        b.cb(logits, "ffn_moe_logits", il);

        var moe_out = b.buildMoeFfn(routed_in, .{
            .gate_inp = null, // unused: the logits above are passed instead
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = layer.ffn_exp_probs_b,
            .n_expert = hp.n_expert,
            .n_expert_used = hp.n_expert_used,
            .type_op = .situ,
            .norm_w = hp.expert_weights_norm,
            .w_scale = hp.expert_weights_scale,
            .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
            .probs_in = logits,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        if (layer.ffn_routed_norm != null) {
            moe_out = b.buildNorm(moe_out, layer.ffn_routed_norm, null, .rms, il);
        }
        if (layer.ffn_routed_up) |w| {
            moe_out = c.ggml_mul_mat(ctx, w, moe_out);
        }

        if (layer.ffn_gate_shexp != null) {
            const g = c.ggml_mul_mat(ctx, layer.ffn_gate_shexp, identity);
            const u = c.ggml_mul_mat(ctx, layer.ffn_up_shexp, identity);
            var sh = situ(ctx, g, u, hp.situ_beta, hp.situ_linear_beta);
            sh = c.ggml_mul_mat(ctx, layer.ffn_down_shexp, sh);
            b.cb(sh, "ffn_shexp", il);
            moe_out = c.ggml_add(ctx, moe_out, sh);
        }

        b.cb(moe_out, "ffn_out", il);
        return moe_out;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
