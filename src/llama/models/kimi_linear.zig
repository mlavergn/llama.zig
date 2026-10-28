//! Kimi Linear: a hybrid of Kimi delta attention (KDA) -- a gated delta
//! net with a per-channel decay and short causal convolutions on Q, K and
//! V -- and multi-head latent attention without RoPE, with leading dense
//! blocks and then a mixture of experts plus a shared expert.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/kimi-linear.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_kimi_linear`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, `causal_conv1d`, and its
//! `graph`. `models.h` also declares `build_kda_autoregressive` and
//! `build_kda_chunking` on that `graph`; neither is defined upstream, and
//! the recurrence runs through the shared delta net.

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

/// Ports `llama_model_kimi_linear`'s `load_arch_hparams` (models/kimi-linear.cpp:4 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_embd_head_k_mla_impl = (try meta.getKey(u32, .attention_key_length_mla, true, exc)).?;
    hp.n_embd_head_v_mla_impl = (try meta.getKey(u32, .attention_value_length_mla, true, exc)).?;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.n_embd_head_kda = (try meta.getKey(u32, .kda_head_dim, true, exc)).?;

    // The MLA qk_rope_head_dim, for reference: qk_rope_head_dim = 64,
    // qk_nope_head_dim = 128, qk_head_dim = 192.

    // The KDA layers are the recurrent ones, marked by n_head_kv = 0 as in
    // Jamba; the MLA layers have n_head_kv = n_head.
    var i: u32 = 0;
    while (i < hp.nLayer()) : (i += 1) {
        hp.is_recr_impl[i] = @intFromBool(hp.nHeadKv(i) == 0);
    }

    // The MoE parameters; Kimi's moe_intermediate_size is 1024.
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        27 => .@"48b_a3b", // Kimi-Linear-48B-A3B
        else => .unknown,
    };
}

/// Ports `llama_model_kimi_linear`'s `load_arch_tensors` (models/kimi-linear.cpp:34 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const n_embd = l.n_embd;
    const n_head = l.n_head;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);

        // KDA's head_dim is linear_attn_config.head_dim, 128.
        const n_embd_head_k_kda: i64 = hp.n_embd_head_kda;
        const n_embd_head_v_kda: i64 = hp.n_embd_head_kda;
        const ssm_d_conv: i64 = hp.ssm_d_conv;

        if (hp.isRecr(@intCast(i))) {
            // The conv1d weights are 4D, [d_conv, 1, d_inner, 1], or 3D,
            // [d_conv, 1, d_inner], when quantization drops the trailing 1.
            layer.ssm_q_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_q, .suffix = "weight", .bid = i }, &.{ ssm_d_conv, 1, n_embd_head_k_kda * n_head, 1 }, nr, exc);
            if (layer.ssm_q_conv == null) {
                layer.ssm_q_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_q, .suffix = "weight", .bid = i }, &.{ ssm_d_conv, 1, n_embd_head_k_kda * n_head }, 0, exc);
            }

            layer.ssm_k_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_k, .suffix = "weight", .bid = i }, &.{ ssm_d_conv, 1, n_embd_head_k_kda * n_head, 1 }, nr, exc);
            if (layer.ssm_k_conv == null) {
                layer.ssm_k_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_k, .suffix = "weight", .bid = i }, &.{ ssm_d_conv, 1, n_embd_head_k_kda * n_head }, 0, exc);
            }
            layer.ssm_v_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_v, .suffix = "weight", .bid = i }, &.{ ssm_d_conv, 1, n_embd_head_v_kda * n_head, 1 }, nr, exc);
            if (layer.ssm_v_conv == null) {
                layer.ssm_v_conv = try self.createTensor(.{ .tensor = .ssm_conv1d_v, .suffix = "weight", .bid = i }, &.{ ssm_d_conv, 1, n_embd_head_v_kda * n_head }, 0, exc);
            }

            // The q, k and v projections.
            try self.createTensorQkv(layer, i, n_embd, n_embd_head_k_kda * n_head, n_embd_head_k_kda * n_head, n_embd_head_v_kda * n_head, 0, exc);

            // KDA's f_a_proj and f_b_proj.
            layer.ssm_f_a = try self.createTensor(.{ .tensor = .ssm_f_a, .suffix = "weight", .bid = i }, &.{ n_embd, n_embd_head_k_kda }, 0, exc); // head_dim
            layer.ssm_f_b = try self.createTensor(.{ .tensor = .ssm_f_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_k_kda, n_embd_head_k_kda * n_head }, 0, exc); // projection_size

            // b_proj, the beta mixing coefficient.
            layer.ssm_beta = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "weight", .bid = i }, &.{ n_embd, n_head }, 0, exc);

            // A_log, [1, num_heads, 1, 1] in the GGUF, or [1, num_heads]
            // after quantization; convert_hf_to_gguf.py applied -exp(A_log).
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = i }, &.{ 1, n_head, 1, 1 }, nr, exc);
            if (layer.ssm_a == null) {
                layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = i }, &.{ 1, n_head }, 0, exc);
            }

            // dt_bias, [n_embd_head_k_kda * n_head].
            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{n_embd_head_k_kda * n_head}, 0, exc);

            // g_a_proj and g_b_proj, the output gate.
            layer.ssm_g_a = try self.createTensor(.{ .tensor = .ssm_g_a, .suffix = "weight", .bid = i }, &.{ n_embd, n_embd_head_k_kda }, 0, exc);
            layer.ssm_g_b = try self.createTensor(.{ .tensor = .ssm_g_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_k_kda, n_embd_head_k_kda * n_head }, 0, exc);

            // o_norm, reusing SSM_NORM: a FusedRMSNormGated.
            layer.ssm_o_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = i }, &.{n_embd_head_k_kda}, 0, exc);

            // o_proj.
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_head_v_kda * n_head, n_embd }, 0, exc);
        } else {
            // An MLA layer, with the MLA head sizes.
            const q_lora_rank: i64 = hp.n_lora_q;
            const kv_lora_rank: i64 = hp.n_lora_kv;
            const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
            const n_embd_head_v_mla: i64 = hp.nEmbdHeadVMla();

            layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, nr, exc);
            layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, 0, exc);

            if (layer.attn_q_a_norm != null) {
                layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ n_embd, q_lora_rank }, 0, exc);
                layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, n_head * n_embd_head_k_mla }, 0, exc);
            } else {
                // Kimi's MLA without Q compression: wq = [n_embd, n_head * n_embd_head_k_mla].
                layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ n_embd, n_head * n_embd_head_k_mla }, 0, exc);
            }

            // Kimi's qk_rope_head_dim is 64, the actual RoPE dimension of
            // MLA; hparams.n_rot may say 72 from the conversion.
            const qk_rope_head_dim: i64 = hp.nRot(0);
            layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ n_embd, kv_lora_rank + qk_rope_head_dim }, 0, exc);
            // Legacy GGUFs that do not split wkv_b, with the MLA KV cache
            // disabled.
            layer.wkv_b = try self.createTensor(.{ .tensor = .attn_kv_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_head * (n_embd_head_k_mla - qk_rope_head_dim + n_embd_head_v_mla) }, nr | loader.tensor_skip_if_virtual, exc);
            if (layer.wkv_b == null) { // The MLA KV cache enabled.
                layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_k_mla - qk_rope_head_dim, kv_lora_rank, n_head }, 0, exc);
                layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v_mla, n_head }, 0, exc);
            }
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_head * n_embd_head_v_mla, n_embd }, 0, exc);
        }

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);

        // The MoE intermediate size, not the dense FFN's.
        const n_ff_exp: i64 = hp.nFfExp(0);

        // n_layer_dense_lead says which layers are dense: with
        // first_k_dense_replace = 1, layer 0 is dense and the rest MoE.
        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            // A dense FFN layer, with the ordinary n_ff.
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_ff }, 0, exc);
        } else {
            // An MoE layer, with n_ff_exp in place of n_ff.
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_expert }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_exp, l.n_expert }, 0, exc);

            // The shared experts are moe_intermediate_size *
            // num_shared_experts wide, 2D: [n_embd, n_ff_shexp] or
            // [n_ff_shexp, n_embd].
            const n_ff_shexp_actual = n_ff_exp * @as(i64, if (hp.n_expert_shared > 0) hp.n_expert_shared else 1);
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_shexp_actual }, nr, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp_actual, n_embd }, nr, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_shexp_actual }, nr, exc);

            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);
        }
    }
}

/// Ports `llama_model_kimi_linear`'s `build_arch_graph` (models/kimi-linear.cpp:169 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports `causal_conv1d` (models/kimi-linear.cpp:175 @5266f24da): the short
/// convolution of one of Q (`qkv` 0), K (1) or V (2), its projection in
/// front, its slice of the conv state in front of the input, and the last
/// `d_conv - 1` columns written back.
fn causalConv1d(gf: *c.ggml_cgraph, ctx: *c.ggml_context, conv_states_all: *Tensor, conv_state_all: *Tensor, qkv: i64, x: *Tensor, proj_w: ?*Tensor, conv_w: ?*Tensor, d_conv: i64, head_dim: i64, n_head: i64, n_seq_tokens: i64, n_seqs: i64, n_tokens: i64, kv_head: i64) *Tensor {
    const d_inner = head_dim * n_head;
    const conv_state_size = (d_conv - 1) * d_inner;
    const n_embd_r_total = 3 * conv_state_size; // Q + K + V

    // conv_state_all is [n_embd_r_total, n_seqs]: for each sequence, the Q
    // state's conv_state_size elements, then K's, then V's. The view is
    // [d_conv-1, d_inner, n_seqs], with nb1 = (d_conv-1) * element_size
    // between channels and nb2 = n_embd_r_total * element_size between
    // sequences.
    const es_state: usize = c.ggml_element_size(conv_state_all);
    const conv_state_x = c.ggml_view_3d(ctx, conv_state_all, d_conv - 1, d_inner, n_seqs, @as(usize, @intCast(d_conv - 1)) * es_state, @as(usize, @intCast(n_embd_r_total)) * es_state, @as(usize, @intCast(qkv * conv_state_size)) * es_state);

    // Step 1: the projection, to [d_inner, n_tokens].
    const x_proj = c.ggml_mul_mat(ctx, proj_w, x);

    // {d_inner, n_tokens} -> {d_inner, n_seq_tokens, n_seqs}
    const x_3d = c.ggml_reshape_3d(ctx, x_proj, d_inner, n_seq_tokens, n_seqs);

    // The conv state and the input concatenated: {d_conv-1 + n_seq_tokens, d_inner, n_seqs}.
    const conv_x: *Tensor = c.ggml_concat(ctx, conv_state_x, c.ggml_transpose(ctx, x_3d), 0);

    // The last (d_conv-1) columns saved back to the conv state.
    const last_conv_x = c.ggml_view_3d(ctx, conv_x, d_conv - 1, d_inner, n_seqs, conv_x.nb[1], conv_x.nb[2], @as(usize, @intCast(n_seq_tokens)) * conv_x.nb[0]);
    const es_all: usize = c.ggml_element_size(conv_states_all);
    c.ggml_build_forward_expand(gf, c.ggml_cpy(ctx, last_conv_x, c.ggml_view_3d(
        ctx,
        conv_states_all,
        d_conv - 1,
        d_inner,
        n_seqs,
        @as(usize, @intCast(d_conv - 1)) * es_all, // nb1: one channel's conv taps, contiguous
        @as(usize, @intCast(n_embd_r_total)) * es_all, // nb2: between sequences, over the K and V states
        @as(usize, @intCast(kv_head * n_embd_r_total + qkv * conv_state_size)) * es_all, // the first sequence's Q, K or V state
    )));

    // The GGUF's conv weight [d_conv, 1, d_inner, 1] (or [d_conv, 1,
    // d_inner]) is laid out w[conv_step + channel * d_conv], which is what
    // ggml_ssm_conv computes with as [d_conv, d_inner].
    const conv_weight = c.ggml_reshape_2d(ctx, conv_w, d_conv, d_inner);

    // The conv1d; ggml_ssm_conv gives {d_inner, n_seq_tokens, n_seqs}.
    var Xcur = c.ggml_ssm_conv(ctx, conv_x, conv_weight);
    // 2D for the bias add: {d_inner, n_tokens}.
    Xcur = c.ggml_reshape_2d(ctx, Xcur, d_inner, n_tokens);
    Xcur = c.ggml_silu(ctx, Xcur);

    return c.ggml_reshape_4d(ctx, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

/// Ports the constructor of `llama_model_kimi_linear`'s `graph`
/// (models/kimi-linear.cpp:234 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    b.cb(inpL, "model.embed_tokens", -1);

    // Kimi's MLA has no RoPE (rotary_emb=None in vLLM), so no positions.

    const is_mla = hp.isMla();
    const inp_kv = if (!is_mla) b.buildInpMemHybrid() else null;
    const inp_k = if (is_mla) b.buildInpMemHybridK() else null;
    const inp_rs = if (is_mla) inp_k.?.getRecr() else inp_kv.?.getRecr();
    const inp_attn_kv = if (!is_mla) inp_kv.?.getAttn() else null;
    const inp_attn_k = if (is_mla) inp_k.?.getAttn() else null;

    // The output ids, which tokens to output.
    const inp_out_ids = b.buildInpOutIds();

    // Kimi's dimensions.
    const n_head: i64 = hp.nHead(0);
    const head_dim: i64 = hp.n_embd_head_kda;
    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner = n_head * head_dim; // 32 * 128 = 4096
    const n_seqs: i64 = b.ubatch.n_seqs;
    const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;
    const n_tokens = b.n_tokens;

    // The batch must suit the recurrent layers.
    impl.assert(n_seqs != 0, "models/kimi-linear.cpp", 263, "n_seqs != 0");
    impl.assert(b.ubatch.equalSeqs(), "models/kimi-linear.cpp", 264, "ubatch.equal_seqs()");
    impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/kimi-linear.cpp", 265, "ubatch.n_tokens == n_seq_tokens * n_seqs");

    // The MLA parameters.
    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_v_mla: i64 = hp.nEmbdHeadVMla();
    const kv_lora_rank: i64 = hp.n_lora_kv;
    // qk_rope_head_dim, 64 in Kimi's config, is hparams.n_rot: wkv_a_mqa
    // is [n_embd, kv_lora_rank + qk_rope_head_dim].
    const n_embd_head_qk_rope: i64 = hp.nRot(0); // config.qk_rope_head_dim
    const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope; // 192 - 64 = 128
    // MLA's attention scale.
    const kq_scale_mla: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head_k_mla));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // The attention norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        c.ggml_build_forward_expand(b.gf, cur);

        if (hp.isRecr(@intCast(il))) {
            // A KDA (Kimi delta attention) layer with recurrent state, after
            // vLLM's kda.py.
            const mctx_cur = inp_rs.mctx;
            const kv_head: i64 = mctx_cur.getHead();

            // The conv states, from r_l: Q, K and V each have their own.
            const conv_states_all = mctx_cur.getRL(il);
            b.cb(conv_states_all, "conv_states_all", il);
            const conv_state_all = b.buildRs(inp_rs, conv_states_all, @intCast(hp.nEmbdR()), @intCast(n_seqs), null);
            var Qcur = causalConv1d(b.gf, ctx, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            var Kcur = causalConv1d(b.gf, ctx, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            const Vcur = causalConv1d(b.gf, ctx, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // g1 = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
            const f_a = c.ggml_mul_mat(ctx, layer.ssm_f_a, cur);
            var g1 = c.ggml_mul_mat(ctx, layer.ssm_f_b, f_a);
            b.cb(g1, "g1 f_b(f_a(cur))", il);
            g1 = c.ggml_add(ctx, g1, layer.ssm_dt_b);
            g1 = c.ggml_softplus(ctx, g1);
            g1 = c.ggml_reshape_3d(ctx, g1, head_dim, n_head, n_tokens);

            // A_log, [1, n_head] or [1, n_head, 1, 1], broadcast as
            // [1, n_head, 1] against g1's [head_dim, n_head, n_tokens];
            // convert_hf_to_gguf.py already applied -exp(A_log).
            const A = c.ggml_reshape_3d(ctx, layer.ssm_a, 1, n_head, 1);
            g1 = c.ggml_mul(ctx, g1, A);
            b.cb(g1, "kda_g1", il);

            g1 = c.ggml_reshape_4d(ctx, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            // Beta, the mixing coefficient.
            var beta = c.ggml_mul_mat(ctx, layer.ssm_beta, cur);
            beta = c.ggml_reshape_4d(ctx, beta, 1, n_head, n_seq_tokens, n_seqs);
            b.cb(beta, "kda_beta", il);

            beta = c.ggml_sigmoid(ctx, beta);

            // For the recurrence:
            // {n_embd, n_tokens} -> {n_embd, n_seq_tokens, n_seqs}
            cur = c.ggml_reshape_3d(ctx, cur, cur.*.ne[0], n_seq_tokens, n_seqs);

            // The SSM state, and the KDA recurrence over it.
            const ssm_states_all = mctx_cur.getSL(il);
            var state = b.buildRs(inp_rs, ssm_states_all, @intCast(hp.nEmbdS()), @intCast(n_seqs), null);
            state = c.ggml_reshape_4d(ctx, state, head_dim, head_dim, n_head, n_seqs);

            const eps_norm = hp.f_norm_rms_eps;

            Qcur = c.ggml_l2_norm(ctx, Qcur, eps_norm);
            Kcur = c.ggml_l2_norm(ctx, Kcur, eps_norm);

            // Chunked or recurrent, by the token count.
            const attn_out = delta_net.buildDeltaNet(b, Qcur, Kcur, Vcur, g1, beta, state, il);

            const output = c.ggml_cont(ctx, attn_out.out);
            const new_state = attn_out.state;
            b.cb(output, "attn_output", il);
            b.cb(new_state, "new_state", il);

            // The recurrent states updated.
            const n_embd_s: i64 = hp.nEmbdS();
            c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, new_state, c.ggml_view_1d(ctx, ssm_states_all, n_embd_s * n_seqs, @as(usize, @intCast(kv_head * n_embd_s)) * c.ggml_element_size(ssm_states_all))));

            // The output gate, g2 = g_b(g_a(x)).
            const cur_2d = c.ggml_reshape_2d(ctx, cur, cur.*.ne[0], n_seq_tokens * n_seqs);
            const g_a = c.ggml_mul_mat(ctx, layer.ssm_g_a, cur_2d);
            var g2 = c.ggml_mul_mat(ctx, layer.ssm_g_b, g_a);
            b.cb(g2, "g2 g_b(g_a(cur_2d))", il);
            g2 = c.ggml_reshape_3d(ctx, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            // o_norm with a sigmoid gate, RMSNorm(x) * sigmoid(g): Kimi
            // gates with a sigmoid, not the SiLU FusedRMSNormGated defaults to.
            const attn_out_final = c.ggml_reshape_3d(ctx, output, head_dim, n_head, n_seq_tokens * n_seqs);
            const normed = b.buildNorm(attn_out_final, layer.ssm_o_norm, null, .rms, il);
            b.cb(normed, "kda_normed", il);
            const gate = c.ggml_sigmoid(ctx, g2);
            var gated = c.ggml_mul(ctx, normed, gate);

            // The output projection.
            gated = c.ggml_cont_2d(ctx, gated, d_inner, n_tokens);
            cur = c.ggml_mul_mat(ctx, layer.wo, gated);
            b.cb(cur, "kda_out", il);
        } else {
            // An MLA (multi-head latent attention) layer, after vLLM's
            // mla.py. Step 1: q = q_proj(hidden_states), viewed as
            // [n_tokens, n_head, qk_head_dim]; Kimi's MLA has no RoPE.
            var Qcur: *Tensor = c.ggml_mul_mat(ctx, layer.wq, cur);

            // Step 2: the KV compression, kv_cmpr_pe =
            // kv_a_proj_with_mqa(hidden_states), [kv_lora_rank +
            // qk_rope_head_dim, n_tokens].
            const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);

            // The split: kv_cmpr = kv_lora[:kv_lora_rank], k_pe =
            // kv_lora[kv_lora_rank:].
            var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), 0);
            const k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));
            // No RoPE: k_pe is used as it is. kv_c is normalized.
            kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);

            if (layer.wk_b != null and layer.wv_b != null) { // The MLA KV cache enabled.
                // q_nope.
                var q_nope: *Tensor = c.ggml_view_3d(ctx, Qcur, n_embd_head_qk_nope, n_head, n_tokens, c.ggml_row_size(Qcur.type, n_embd_head_k_mla), c.ggml_row_size(Qcur.type, n_embd_head_k_mla) * @as(usize, @intCast(n_head)), 0);
                b.cb(q_nope, "q_nope", il);

                // And {n_embd_head_qk_rope, n_head, n_tokens}.
                const q_pe: *Tensor = c.ggml_view_3d(ctx, Qcur, n_embd_head_qk_rope, n_head, n_tokens, c.ggml_row_size(Qcur.type, n_embd_head_k_mla), c.ggml_row_size(Qcur.type, n_embd_head_k_mla) * @as(usize, @intCast(n_head)), c.ggml_row_size(Qcur.type, n_embd_head_qk_nope));
                b.cb(q_pe, "q_pe", il);

                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
                b.cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                var q_nope_absorbed: *Tensor = c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
                b.cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);
                b.cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}; the
                // rope part goes first, for the in-place context shift of
                // build_rope_shift().
                Qcur = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
                b.cb(Qcur, "Qcur", il);

                kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, n_tokens);
                b.cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                const Kcur: *Tensor = c.ggml_concat(ctx, kv_cmpr, k_pe, 0);
                b.cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                const Vcur = kv_cmpr;
                b.cb(Vcur, "Vcur", il);

                cur = b.buildAttn(inp_attn_k.?, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, layer.wv_b, kq_scale_mla, il);
                b.cb(cur, "mla_out", il);
            } else { // The MLA KV cache disabled: an MHA KV cache instead.
                Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head_k_mla, n_head, n_tokens);
                b.cb(Qcur, "mla_Q", il);
                // The KV decompression, kv = kv_b_proj(kv_c_normed).
                const kv: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_b, kv_cmpr);
                const kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;

                // The split into k_nope and v.
                const k_nope: *Tensor = c.ggml_view_3d(ctx, kv, n_embd_head_qk_nope, n_head, n_tokens, c.ggml_row_size(kv.type, kv_per_head), c.ggml_row_size(kv.type, kv_per_head * n_head), 0);
                var Vcur: *Tensor = c.ggml_view_3d(ctx, kv, n_embd_head_v_mla, n_head, n_tokens, c.ggml_row_size(kv.type, kv_per_head), c.ggml_row_size(kv.type, kv_per_head * n_head), c.ggml_row_size(kv.type, n_embd_head_qk_nope));
                Vcur = c.ggml_cont(ctx, Vcur);
                b.cb(Vcur, "mla_V", il);

                // K = [k_nope, k_pe], with k_nope [qk_nope_head_dim, n_head,
                // n_tokens] and k_pe [qk_rope_head_dim, 1, n_tokens]
                // broadcast to every head.
                const k_pe_target = c.ggml_new_tensor_3d(ctx, k_pe.type, n_embd_head_qk_rope, n_head, n_tokens);
                const k_pe_repeated = c.ggml_repeat(ctx, k_pe, k_pe_target);
                const Kcur: *Tensor = c.ggml_concat(ctx, k_pe_repeated, k_nope, 0);
                b.cb(Kcur, "mla_K", il);

                // Softmax attention over the MHA KV cache, with its mask.
                cur = b.buildAttn(inp_attn_kv.?, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale_mla, il);
                b.cb(cur, "mla_out", il);
            }
        }

        // The last layer keeps the output tokens only.
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        // The residual.
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The FFN norm.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            // A dense FFN layer.
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // An MoE layer: Kimi renormalizes (moe_renormalize=True) and
            // scales by routed_scaling_factor, stored as
            // expert_weights_scale (2.446).
            const moe_out = b.buildMoeFfn(cur, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = layer.ffn_gate_exps,
                .down_exps = layer.ffn_down_exps.?,
                .exp_probs_b = layer.ffn_exp_probs_b,
                .n_expert = hp.n_expert,
                .n_expert_used = hp.nExpertUsed(0),
                .type_op = .silu,
                .norm_w = true,
                .w_scale = hp.expert_weights_scale,
                .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
            }, il);
            b.cb(moe_out, "ffn_moe_out", il);

            // The shared expert.
            {
                const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
                b.cb(ffn_shexp, "ffn_shexp", il);

                cur = c.ggml_add(ctx, moe_out, ffn_shexp);
                b.cb(cur, "ffn_out", il);
            }
        }
        // The residual.
        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    // The final norm.
    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The output.
    cur = c.ggml_mul_mat(ctx, model.output, cur);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
