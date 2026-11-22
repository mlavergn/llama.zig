//! Kimi-K3.1: Kimi-K3 with DeepSeek sparse attention (DSA) on the
//! full-attention layers. The KDA layers, the cross-layer residual
//! attention, the latent MoE, situ and the MLA shapes are K3's; each MLA
//! layer the GGUF marks as an indexer layer runs a lightning indexer over
//! its own per-token key cache -- one cell per attention cell -- keeps the
//! top-k cells and attends only to those, and the other MLA layers reuse
//! the previous indexer layer's selection.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/kimi-k31.cpp` at v1.0.0 (`b1b770d90`),
//! with its declaration in `llama.cpp/src/models/models.h`:
//! `llama_model_kimi_k31`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, `kimi_k31_situ`, and its `graph` with `res_push`,
//! `res_mix`, `build_kda_conv1d`, `build_kda_layer`, `build_dsa_layer` and
//! `build_latent_moe`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

const src = "models/kimi-k31.cpp";

/// Ports `llama_model_kimi_k31`'s `load_arch_hparams` (models/kimi-k31.cpp:15 @b1b770d90).
///
/// Parameters:
/// - `self`: the model whose hyperparameters are read.
/// - `ml`: the loader holding the GGUF metadata.
/// - `exc`: receives the exception a missing key throws.
///
/// Return: `error.Thrown` when a required key is missing.
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
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

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;
    if (try meta.getKey(u32, .expert_latent_length, false, exc)) |v| hp.n_expert_latent = v;

    hp.attn_res_block_size = (try meta.getKey(u32, .attn_res_block_size, true, exc)).?;
    hp.situ_beta = (try meta.getKey(f32, .activation_situ_beta, true, exc)).?;
    hp.situ_linear_beta = (try meta.getKey(f32, .activation_situ_linear_beta, true, exc)).?;

    // The DSA indexer, on the dense-attention layers only.
    hp.f_norm_eps = 1e-5; // eps of the indexer key LayerNorm
    if (try meta.getKey(f32, .attention_layernorm_eps, false, exc)) |v| hp.f_norm_eps = v;
    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;
    // Every DSA layer scores its own selection unless the GGUF says
    // otherwise.
    @memset(&hp.is_indexer_full_impl, 1);
    _ = try meta.getKeyOrArr(u32, .attention_indexer_types, &hp.is_indexer_full_impl, hp.nLayer(), false, exc);

    if (hp.indexer_top_k > 0) {
        llama.impl.assert(hp.indexer_n_head > 0 and hp.indexer_head_size > 0, src, 57, "hparams.indexer_n_head > 0 && hparams.indexer_head_size > 0");
        llama.impl.assert(hp.n_lora_q > 0, src, 58, "hparams.n_lora_q > 0 && \"the indexer queries come from the MLA q latent\"");
    }

    self.type = switch (hp.nLayer()) {
        93 => .@"2_8t_a50b", // Kimi-K3.1-Pro
        else => .unknown,
    };
}

/// Ports `llama_model_kimi_k31`'s `load_arch_tensors` (models/kimi-k31.cpp:67 @b1b770d90).
///
/// Parameters:
/// - `self`: the model whose tensors are created.
/// - `exc`: receives the exception a missing tensor throws.
///
/// Return: `error.Thrown` when a required tensor is missing or misshapen.
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;
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
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = i }, &.{n_head}, 0, exc);
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
            layer.wkv_b = try self.createTensor(.{ .tensor = .attn_kv_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_head * (qk_nope_head_dim + n_embd_head_v) }, nr | llama.loader.tensor_skip_if_virtual, exc);
            if (layer.wkv_b == null) {
                layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ qk_nope_head_dim, kv_lora_rank, n_head }, 0, exc);
                layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v, n_head }, 0, exc);
            }

            // K3's sigmoid output gate on the attention output, before o_proj.
            layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ n_embd, n_head * n_embd_head_v }, nr, exc);

            // The DSA lightning indexer, on the layers the GGUF marks as
            // indexer layers.
            if (hp.indexer_top_k > 0 and hp.isIndexerFull(@intCast(i))) {
                const n_idx_head: i64 = hp.indexer_n_head;
                const n_idx_dim: i64 = hp.indexer_head_size;

                layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, n_idx_head * n_idx_dim }, 0, exc);
                layer.indexer_attn_k = try self.createTensor(.{ .tensor = .indexer_attn_k, .suffix = "weight", .bid = i }, &.{ n_embd, n_idx_dim }, 0, exc);
                layer.indexer_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{n_idx_dim}, 0, exc);
                layer.indexer_k_norm_b = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "bias", .bid = i }, &.{n_idx_dim}, 0, exc);
                layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ n_embd, n_idx_head }, 0, exc);
            }

            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_head * n_embd_head_v, n_embd }, 0, exc);
        }

        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_ff }, 0, exc);
        } else {
            const n_ff_exp: i64 = hp.nFfExp(0);

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

/// `conv`, a lambda of `load_arch_tensors` (models/kimi-k31.cpp:67
/// @b1b770d90): a conv1d weight, 4D or else 3D.
fn loadConv(self: *llama.model.Model, tid: llama.arch_table.Tensor, il: i32, d_conv: i64, d_inner: i64, exc: *llama.impl.Exception) llama.impl.Thrown!?*llama.impl.c.ggml_tensor {
    const t = try self.createTensor(.{ .tensor = tid, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner, 1 }, llama.loader.tensor_not_required, exc);
    return t orelse try self.createTensor(.{ .tensor = tid, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner }, 0, exc);
}

/// Ports `llama_model_kimi_k31`'s `build_arch_graph` (models/kimi-k31.cpp:195 @b1b770d90).
///
/// Parameters:
/// - `model`: the model the graph reads its tensors from.
/// - `b`: the graph builder.
///
/// Return: nothing; the graph is built into `b`'s result.
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    var g: Graph = .{ .model = model, .b = b };
    g.build();
}

/// Ports `kimi_k31_situ` (models/kimi-k31.cpp:201 @b1b770d90):
/// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) *
/// linear_beta*tanh(up/linear_beta); a linear_beta <= 0 leaves the up
/// branch as it is.
fn situ(ctx: *llama.impl.c.ggml_context, gate: *llama.impl.c.ggml_tensor, up_: *llama.impl.c.ggml_tensor, beta: f32, linear_beta: f32) *llama.impl.c.ggml_tensor {
    var up = up_;
    var a: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_scale(ctx, llama.impl.c.ggml_tanh(ctx, llama.impl.c.ggml_scale(ctx, gate, 1.0 / beta)), beta);
    a = llama.impl.c.ggml_mul(ctx, a, llama.impl.c.ggml_sigmoid(ctx, gate));

    if (linear_beta > 0.0) {
        up = llama.impl.c.ggml_scale(ctx, llama.impl.c.ggml_tanh(ctx, llama.impl.c.ggml_scale(ctx, up, 1.0 / linear_beta)), linear_beta);
    }
    return llama.impl.c.ggml_mul(ctx, a, up);
}

/// `llama_model_kimi_k31`'s `graph`: the builder, and the stack of banked
/// residual checkpoints the cross-layer residual attention reads.
const Graph = struct {
    model: *const llama.model.Model,
    b: *const llama.graph.Builder,
    /// The banked checkpoints, `[n_embd, n_ckpt, n_tokens]`.
    resi_stack: ?*llama.impl.c.ggml_tensor = null,

    /// Ports `llama_model_kimi_k31::graph`'s `res_push`
    /// (models/kimi-k31.cpp:218 @b1b770d90): append the new checkpoint
    /// without refolding the chain. The layout is `[n_embd, n_ckpt,
    /// n_tokens]`: rms_norm reduces over ne0, dsv4_hc_pre over ne1.
    fn resPush(self: *Graph, cur: *llama.impl.c.ggml_tensor, n_embd: i64, n_tokens: i64) void {
        const ctx = self.b.ctx0;
        const ckpt: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

        self.resi_stack = if (self.resi_stack) |s| llama.impl.c.ggml_concat(ctx, s, ckpt, 1) else ckpt;
    }

    /// Ports `llama_model_kimi_k31::graph`'s `res_mix`
    /// (models/kimi-k31.cpp:224 @b1b770d90): a softmax over the banked
    /// checkpoints and the current stream, scored by `score_w`, and their
    /// weighted sum.
    fn resMix(self: *const Graph, cur: *llama.impl.c.ggml_tensor, score_w: ?*llama.impl.c.ggml_tensor, n_tokens: i64, il: c_int) *llama.impl.c.ggml_tensor {
        const ctx = self.b.ctx0;
        // Layer 0: nothing banked yet.
        const stack = self.resi_stack orelse return cur; // [n_embd, n_ckpt, n_tokens]

        const n_ckpt: c_int = @intCast(stack.ne[1]);
        const eps = self.b.hparams.f_norm_rms_eps;

        // One rms_norm scores every checkpoint; the scores use the
        // normalized values, the sum below the raw ones.
        var sc_src: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_rms_norm(ctx, stack, eps);
        sc_src = llama.impl.c.ggml_mul(ctx, sc_src, score_w);
        sc_src = llama.impl.c.ggml_sum_rows(ctx, sc_src); // [1, n_ckpt, n_tokens]
        sc_src = llama.impl.c.ggml_reshape_2d(ctx, sc_src, n_ckpt, n_tokens);

        // The current stream is scored apart, so the stack stays append-only.
        var sc_cur: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_rms_norm(ctx, cur, eps);
        sc_cur = llama.impl.c.ggml_mul(ctx, sc_cur, score_w);
        sc_cur = llama.impl.c.ggml_sum_rows(ctx, sc_cur); // [1, n_tokens]

        const scores = llama.impl.c.ggml_concat(ctx, sc_src, sc_cur, 0); // [n_ckpt+1, n_tokens]
        const probs: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_soft_max(ctx, scores); // over ne0 = n_ckpt+1
        self.b.cb(probs, "res_probs", il);

        // The sum split: hc_pre takes the stack, a broadcast multiply the
        // current stream.
        const p_src = llama.impl.c.ggml_cont(ctx, llama.impl.c.ggml_view_2d(ctx, probs, n_ckpt, n_tokens, probs.nb[1], 0));
        const p_cur = llama.impl.c.ggml_cont(ctx, llama.impl.c.ggml_view_2d(ctx, probs, 1, n_tokens, probs.nb[1], probs.nb[0] * @as(usize, @intCast(n_ckpt))));

        var out = llama.impl.c.ggml_dsv4_hc_pre(ctx, stack, p_src);
        out = llama.impl.c.ggml_add(ctx, out, llama.impl.c.ggml_mul(ctx, cur, p_cur));

        return out;
    }

    /// The MLA dimensions the constructor hands `build_dsa_layer`.
    const Mla = struct {
        n_embd_head_k_mla: i64,
        kv_lora_rank: i64,
        n_embd_head_qk_rope: i64,
        n_embd_head_qk_nope: i64,
        kq_scale: f32,
    };

    /// Ports the constructor of `llama_model_kimi_k31`'s `graph`
    /// (models/kimi-k31.cpp:262 @b1b770d90).
    fn build(self: *Graph) void {
        const model = self.model;
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_embd: i64 = b.n_embd;
        const n_tokens = b.n_tokens;

        var inpL = b.buildInpEmbd(model.tok_embd.?);
        b.cb(inpL, "inp_embd", -1);

        // K3.1's MLA is nope-only, so there is no position input.

        const inp_k = b.buildInpMemHybridK();
        const inp_rs = inp_k.getRecr();
        const inp_attn = inp_k.getAttn();

        // The indexer cache sits next to the attention cache and shares its
        // cells.
        const mctx_hyb = llama.memory_hybrid_idx.HybridIdxContext.fromBase(inp_k.mctx);
        const mctx_idx = mctx_hyb.getIdx();
        if (mctx_idx) |idx| {
            llama.impl.assert(idx.getNKv() == inp_attn.mctx.getNKv(), src, 282, "mctx_idx->get_n_kv() == inp_attn->mctx->get_n_kv() && \"the indexer cache must track the attention cache cell for cell\"");
        }

        const inp_out_ids = b.buildInpOutIds();

        const n_head_kda: i64 = hp.nHead(0);
        const head_dim: i64 = hp.n_embd_head_kda;
        const d_conv: i64 = hp.ssm_d_conv;
        const d_inner = n_head_kda * head_dim;
        const n_seqs: i64 = b.ubatch.n_seqs;
        const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

        llama.impl.assert(n_seqs != 0, src, 294, "n_seqs != 0");
        llama.impl.assert(b.ubatch.equalSeqs(), src, 295, "ubatch.equal_seqs()");
        llama.impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, src, 296, "ubatch.n_tokens == n_seq_tokens * n_seqs");

        const mla: Mla = .{
            .n_embd_head_k_mla = hp.nEmbdHeadKMla(),
            .kv_lora_rank = hp.n_lora_kv,
            .n_embd_head_qk_rope = hp.nRot(0),
            .n_embd_head_qk_nope = @as(i64, hp.nEmbdHeadKMla()) - hp.nRot(0),
            .kq_scale = 1.0 / stdlib.libm.sqrtfInt(hp.nEmbdHeadKMla()),
        };

        const res_bs: u32 = hp.attn_res_block_size;
        const use_attn_res = res_bs > 0;
        const n_embd_latent: i64 = if (hp.n_expert_latent > 0) hp.n_expert_latent else n_embd;

        var prev_sel: ?*llama.impl.c.ggml_tensor = null;

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
            llama.impl.c.ggml_build_forward_expand(b.gf, cur);

            if (hp.isRecr(@intCast(il))) {
                cur = self.buildKdaLayer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda, d_inner, n_seq_tokens, n_seqs, il);
            } else {
                const dsa = self.buildDsaLayer(cur, layer, mctx_idx, inp_attn, prev_sel, &mla, il);
                cur = dsa.out;
                prev_sel = dsa.sel;
            }

            prefix_sum = if (banked) cur else llama.impl.c.ggml_add(ctx, prefix_sum, cur);
            b.cb(prefix_sum, "prefix_sum_attn", il);

            cur = if (use_attn_res) self.resMix(prefix_sum, layer.ffn_res_score, n_tokens, il) else prefix_sum;

            cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
                const g = llama.impl.c.ggml_mul_mat(ctx, layer.ffn_gate, cur);
                const u = llama.impl.c.ggml_mul_mat(ctx, layer.ffn_up, cur);
                cur = situ(ctx, g, u, hp.situ_beta, hp.situ_linear_beta);
                cur = llama.impl.c.ggml_mul_mat(ctx, layer.ffn_down, cur);
                b.cb(cur, "ffn_out", il);
            } else {
                cur = self.buildLatentMoe(cur, layer, n_embd_latent, il);
            }

            prefix_sum = llama.impl.c.ggml_add(ctx, prefix_sum, cur);
            prefix_sum = b.buildCvec(prefix_sum, il);
            b.cb(prefix_sum, "l_out", il);

            inpL = prefix_sum;
        }

        var cur = inpL;

        // The final mix, then only the output tokens.
        if (use_attn_res) {
            cur = self.resMix(cur, model.output_res_score, n_tokens, -1);
        }
        cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);

        cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        cur = llama.impl.c.ggml_mul_mat(ctx, model.output, cur);
        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;
    }

    /// Ports `llama_model_kimi_k31::graph`'s `build_kda_conv1d`
    /// (models/kimi-k31.cpp:391 @b1b770d90): the causal conv1d over one of
    /// Q, K or V; `qkv` picks which third of the conv state. The conv
    /// history of the ubatch's cells is stored through
    /// `build_conv_state_store`.
    fn buildKdaConv1d(self: *const Graph, inp_rs: *llama.graph.InpRs, conv_states_all: *llama.impl.c.ggml_tensor, conv_state_all: *llama.impl.c.ggml_tensor, qkv: i64, x: *llama.impl.c.ggml_tensor, proj_w: ?*llama.impl.c.ggml_tensor, conv_w: ?*llama.impl.c.ggml_tensor, d_conv: i64, head_dim: i64, n_head_kda: i64, n_seq_tokens: i64, n_seqs: i64) *llama.impl.c.ggml_tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const d_inner = head_dim * n_head_kda;
        const conv_state_size = (d_conv - 1) * d_inner;
        const n_embd_r_total = 3 * conv_state_size;

        const es_state: usize = llama.impl.c.ggml_element_size(conv_state_all);
        const conv_state_x = llama.impl.c.ggml_view_3d(ctx, conv_state_all, d_conv - 1, d_inner, n_seqs, @as(usize, @intCast(d_conv - 1)) * es_state, @as(usize, @intCast(n_embd_r_total)) * es_state, @as(usize, @intCast(qkv * conv_state_size)) * es_state);

        const x_proj = llama.impl.c.ggml_mul_mat(ctx, proj_w, x);
        const x_3d = llama.impl.c.ggml_reshape_3d(ctx, x_proj, d_inner, n_seq_tokens, n_seqs);
        const conv_x: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, conv_state_x, llama.impl.c.ggml_transpose(ctx, x_3d), 0);

        b.buildConvStateStore(inp_rs, conv_states_all, conv_x, d_conv - 1, llama.impl.c.ggml_row_size(conv_states_all.type, qkv * conv_state_size));

        const conv_weight = llama.impl.c.ggml_reshape_2d(ctx, conv_w, d_conv, d_inner);
        var Xcur = llama.impl.c.ggml_ssm_conv(ctx, conv_x, conv_weight);
        Xcur = llama.impl.c.ggml_reshape_2d(ctx, Xcur, d_inner, b.n_tokens);
        Xcur = llama.impl.c.ggml_silu(ctx, Xcur);

        return llama.impl.c.ggml_reshape_4d(ctx, Xcur, head_dim, n_head_kda, n_seq_tokens, n_seqs);
    }

    /// Ports `llama_model_kimi_k31::graph`'s `build_kda_layer`
    /// (models/kimi-k31.cpp:419 @b1b770d90).
    fn buildKdaLayer(self: *const Graph, cur_: *llama.impl.c.ggml_tensor, layer: *const llama.model.Layer, inp_rs: *llama.graph.InpRs, d_conv: i64, head_dim: i64, n_head_kda: i64, d_inner: i64, n_seq_tokens: i64, n_seqs: i64, il: c_int) *llama.impl.c.ggml_tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_tokens = b.n_tokens;
        var cur = cur_;

        const mctx_cur = inp_rs.mctx;

        const conv_states_all = mctx_cur.getRL(il);
        const conv_state_all = b.buildRs(inp_rs, conv_states_all, @intCast(hp.nEmbdR()), @intCast(n_seqs), null);

        var Qcur = self.buildKdaConv1d(inp_rs, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs);
        var Kcur = self.buildKdaConv1d(inp_rs, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs);
        const Vcur = self.buildKdaConv1d(inp_rs, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs);
        b.cb(Qcur, "kda_q_conv", il);
        b.cb(Kcur, "kda_k_conv", il);
        b.cb(Vcur, "kda_v_conv", il);

        // gate_lower_bound is not a clamp: set, it swaps the decay gate's
        // activation --
        //   unset (Kimi Linear): g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
        //   set   (K3, -5.0):    g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
        // ssm_a holds -exp(A_log), folded at conversion, so exp(A_log) == -ssm_a.
        const f_a = llama.impl.c.ggml_mul_mat(ctx, layer.ssm_f_a, cur);
        var g1 = llama.impl.c.ggml_mul_mat(ctx, layer.ssm_f_b, f_a);
        g1 = llama.impl.c.ggml_add(ctx, g1, layer.ssm_dt_b);

        const A = llama.impl.c.ggml_reshape_3d(ctx, layer.ssm_a, 1, n_head_kda, 1);

        if (hp.kda_gate_lower_bound > -std.math.inf(f32)) {
            g1 = llama.impl.c.ggml_reshape_3d(ctx, g1, head_dim, n_head_kda, n_tokens);
            g1 = llama.impl.c.ggml_mul(ctx, g1, A); // -exp(A_log) * (...)
            g1 = llama.impl.c.ggml_sigmoid(ctx, llama.impl.c.ggml_scale(ctx, g1, -1.0));
            g1 = llama.impl.c.ggml_scale(ctx, g1, hp.kda_gate_lower_bound);
        } else {
            g1 = llama.impl.c.ggml_softplus(ctx, g1);
            g1 = llama.impl.c.ggml_reshape_3d(ctx, g1, head_dim, n_head_kda, n_tokens);
            g1 = llama.impl.c.ggml_mul(ctx, g1, A);
        }
        b.cb(g1, "kda_g1", il);

        g1 = llama.impl.c.ggml_reshape_4d(ctx, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

        var beta = llama.impl.c.ggml_mul_mat(ctx, layer.ssm_beta, cur);
        beta = llama.impl.c.ggml_reshape_4d(ctx, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
        beta = llama.impl.c.ggml_sigmoid(ctx, beta);
        b.cb(beta, "kda_beta", il);

        const cur_3d: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_reshape_3d(ctx, cur, cur.ne[0], n_seq_tokens, n_seqs);

        const ssm_states_all = mctx_cur.getSL(il);
        var state = b.buildRs(inp_rs, ssm_states_all, @intCast(hp.nEmbdS()), @intCast(n_seqs), null);
        state = llama.impl.c.ggml_reshape_4d(ctx, state, head_dim, head_dim, n_head_kda, n_seqs);

        const eps_norm = hp.f_norm_rms_eps;
        Qcur = llama.models.delta_net.buildGdnL2Norm(ctx, Qcur, eps_norm);
        Kcur = llama.models.delta_net.buildGdnL2Norm(ctx, Kcur, eps_norm);

        var output = llama.models.delta_net.buildRecurrentAttn(b, inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g1, beta, state, il);
        output = llama.impl.c.ggml_cont(ctx, output);
        b.cb(output, "kda_scan_out", il);

        // K3's single full-rank gate; Kimi Linear factors it as g_b(g_a(x)).
        const cur_2d = llama.impl.c.ggml_reshape_2d(ctx, cur_3d, cur_3d.ne[0], n_seq_tokens * n_seqs);
        var g2 = llama.impl.c.ggml_mul_mat(ctx, layer.ssm_g, cur_2d);
        g2 = llama.impl.c.ggml_reshape_3d(ctx, g2, head_dim, n_head_kda, n_seq_tokens * n_seqs);

        const o = llama.impl.c.ggml_reshape_3d(ctx, output, head_dim, n_head_kda, n_seq_tokens * n_seqs);
        const normed = b.buildNorm(o, layer.ssm_o_norm, null, .rms, il);
        b.cb(g2, "kda_g2", il);
        b.cb(normed, "kda_normed", il);
        var gated = llama.impl.c.ggml_mul(ctx, normed, llama.impl.c.ggml_sigmoid(ctx, g2));

        gated = llama.impl.c.ggml_cont_2d(ctx, gated, d_inner, n_tokens);
        cur = llama.impl.c.ggml_mul_mat(ctx, layer.wo, gated);
        b.cb(cur, "kda_out", il);

        return cur;
    }

    /// Mirrors the graph's `dsa_out` (src/models/models.h:2960 @b1b770d90):
    /// the layer's output, and the selection the next shared indexer layer
    /// reuses.
    const DsaOut = struct {
        out: *llama.impl.c.ggml_tensor,
        sel: *llama.impl.c.ggml_tensor,
    };

    /// Ports `llama_model_kimi_k31::graph`'s `build_dsa_layer`
    /// (models/kimi-k31.cpp:502 @b1b770d90): nope-only MLA over the top-k
    /// cells the lightning indexer picks, with K3's sigmoid output gate.
    fn buildDsaLayer(self: *const Graph, cur: *llama.impl.c.ggml_tensor, layer: *const llama.model.Layer, mctx_idx: ?*const llama.kv_cache.KvCacheContext, inp_attn: *llama.graph.InpAttnK, prev_sel: ?*llama.impl.c.ggml_tensor, d: *const Mla, il: c_int) DsaOut {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_tokens = b.n_tokens;
        const n_head: i64 = b.n_head;

        const mctx_mla = inp_attn.mctx;

        const n_idx_head: i64 = hp.indexer_n_head;
        const n_idx_dim: i64 = hp.indexer_head_size;

        const inp_gate = cur; // the output gate reads the *normed* layer input

        // The MLA query; the indexer reads the same q latent.
        llama.impl.assert(layer.wq_a != null and layer.wq_b != null, src, 517, "layer.wq_a && layer.wq_b && \"K3.1 MLA is served from the q latent\"");

        var qr: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.wq_a, cur);
        qr = b.buildNorm(qr, layer.attn_q_a_norm, null, .rms, il);
        const Qcur: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.wq_b, qr);

        const kv_cmpr_pe: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);

        var kv_cmpr: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_2d(ctx, kv_cmpr_pe, d.kv_lora_rank, n_tokens, llama.impl.c.ggml_row_size(kv_cmpr_pe.type, d.kv_lora_rank + d.n_embd_head_qk_rope), 0);
        const k_pe: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, kv_cmpr_pe, d.n_embd_head_qk_rope, 1, n_tokens, llama.impl.c.ggml_row_size(kv_cmpr_pe.type, d.kv_lora_rank + d.n_embd_head_qk_rope), llama.impl.c.ggml_row_size(kv_cmpr_pe.type, d.kv_lora_rank + d.n_embd_head_qk_rope), llama.impl.c.ggml_row_size(kv_cmpr_pe.type, d.kv_lora_rank));

        // No RoPE: mla_use_nope is asserted at conversion.
        kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);

        // The cache holds the compressed latent, so the MLA is served
        // absorbed.
        llama.impl.assert(layer.wk_b != null and layer.wv_b != null, src, 536, "layer.wk_b && layer.wv_b");

        var q_nope: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, Qcur, d.n_embd_head_qk_nope, n_head, n_tokens, llama.impl.c.ggml_row_size(Qcur.type, d.n_embd_head_k_mla), llama.impl.c.ggml_row_size(Qcur.type, d.n_embd_head_k_mla) * @as(usize, @intCast(n_head)), 0);
        const q_pe: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, Qcur, d.n_embd_head_qk_rope, n_head, n_tokens, llama.impl.c.ggml_row_size(Qcur.type, d.n_embd_head_k_mla), llama.impl.c.ggml_row_size(Qcur.type, d.n_embd_head_k_mla) * @as(usize, @intCast(n_head)), llama.impl.c.ggml_row_size(Qcur.type, d.n_embd_head_qk_nope));

        q_nope = llama.impl.c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
        var q_nope_absorbed: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
        q_nope_absorbed = llama.impl.c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);

        const Q: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
        const kv_cmpr_3d: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_reshape_3d(ctx, kv_cmpr, d.kv_lora_rank, 1, n_tokens);
        const K: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, kv_cmpr_3d, k_pe, 0);

        // DSA: keep only the top-k cells of this layer.
        var sel_opt = prev_sel;
        if (hp.isIndexerFull(@intCast(il)) and layer.indexer_attn_q_b != null) {
            llama.impl.assert(mctx_idx != null, src, 557, "mctx_idx != nullptr");
            const idx = mctx_idx.?;

            var iq: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.indexer_attn_q_b, qr);
            iq = llama.impl.c.ggml_reshape_3d(ctx, iq, n_idx_dim, n_idx_head, n_tokens);

            var ik: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.indexer_attn_k, cur);
            ik = b.buildNorm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, .norm, il);
            ik = llama.impl.c.ggml_reshape_3d(ctx, ik, n_idx_dim, 1, n_tokens);
            b.cb(ik, "indexer_k", il);

            // The indexer keys go to the cells of the matching attention
            // layer.
            llama.impl.c.ggml_build_forward_expand(b.gf, idx.cpyK(ctx, ik, inp_attn.getKIdxs(), il));

            var iw: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.indexer_proj, cur);
            // Pre-scaled, so the huge score tensor is not scaled afterwards.
            iw = llama.impl.c.ggml_scale(ctx, iw, 1.0 / stdlib.libm.sqrtfInt(n_idx_dim * n_idx_head));

            ik = idx.getK(ctx, il);

            // The ubatch split into its streams; k carries one row per
            // stream.
            const n_stream = ik.ne[3];
            const n_stream_u: usize = @intCast(n_stream);
            iq = llama.impl.c.ggml_view_4d(ctx, iq, n_idx_dim, n_idx_head, @divTrunc(n_tokens, n_stream), n_stream, iq.nb[1], iq.nb[2], iq.nb[3] / n_stream_u, 0);
            iw = llama.impl.c.ggml_view_4d(ctx, iw, n_idx_head, @divTrunc(n_tokens, n_stream), 1, n_stream, iw.nb[1], iw.nb[2] / n_stream_u, iw.nb[2] / n_stream_u, 0);

            var kq_mask_lid: *llama.impl.c.ggml_tensor = inp_attn.self_attn_mask.raw.?;
            if (b.cparams.fused_lid and kq_mask_lid.type != llama.impl.c.GGML_TYPE_F16) {
                kq_mask_lid = llama.impl.c.ggml_cast(ctx, kq_mask_lid, llama.impl.c.GGML_TYPE_F16);
            }

            var score: *llama.impl.c.ggml_tensor = undefined;
            if (b.cparams.fused_lid) {
                score = llama.impl.c.ggml_lightning_indexer(ctx, iq, ik, iw, kq_mask_lid);
                b.res.addFusedNode(.{ .op = .lightning_indexer, .tensor = score, .il = il });
            } else {
                const q_p = llama.impl.c.ggml_permute(ctx, iq, 0, 2, 1, 3); // [n_idx_dim, n_tokens, n_idx_head]
                const k_p = llama.impl.c.ggml_permute(ctx, ik, 0, 2, 1, 3); // [n_idx_dim, n_kv, 1]

                var kq: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, k_p, q_p); // [n_kv, n_tokens, n_idx_head]
                kq = llama.impl.c.ggml_cont(ctx, llama.impl.c.ggml_permute(ctx, kq, 2, 1, 0, 3)); // [n_idx_head, n_tokens, n_kv]

                score = llama.impl.c.ggml_relu(ctx, kq);
                score = llama.impl.c.ggml_mul(ctx, score, iw);
                score = llama.impl.c.ggml_sum_rows(ctx, score); // [1, n_tokens, n_kv]
                score = llama.impl.c.ggml_cont(ctx, llama.impl.c.ggml_permute(ctx, score, 2, 1, 0, 3)); // [n_kv, n_tokens, 1]

                score = llama.impl.c.ggml_add(ctx, score, kq_mask_lid);
            }
            b.cb(score, "indexer_score", il);

            const n_top_k: i64 = @min(score.ne[0], @as(i64, hp.indexer_top_k));
            const top_k: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_cont(ctx, llama.impl.c.ggml_top_k(ctx, score, @intCast(n_top_k)));
            b.cb(top_k, "indexer_top_k", il);

            const sel_new = b.buildTopKMask(inp_attn.self_attn_mask.raw.?, top_k, il);
            b.cb(sel_new, "kq_mask_dsa", il);
            sel_opt = sel_new;
        } else {
            llama.impl.assert(sel_opt != null, src, 615, "sel != nullptr && \"shared indexer layer must follow a full indexer layer\"");
        }
        const sel = sel_opt.?;

        llama.impl.c.ggml_build_forward_expand(b.gf, Q);
        llama.impl.c.ggml_build_forward_expand(b.gf, K);
        llama.impl.c.ggml_build_forward_expand(b.gf, mctx_mla.cpyK(ctx, K, inp_attn.getKIdxs(), il));

        const k: *llama.impl.c.ggml_tensor = mctx_mla.getK(ctx, il);
        const v: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_4d(ctx, k, d.kv_lora_rank, k.ne[1], k.ne[2], k.ne[3], k.nb[1], k.nb[2], k.nb[3], 0);

        // The selection is applied as an additive mask, so attention only
        // gets cheaper where the backend can skip masked columns: CUDA flash
        // attention gathers the visible cells when n_kv_max > 0 and the
        // cache holds at least 4096 cells. On the dense fallbacks the mask
        // keeps the result correct, but the kq and kqv products still run
        // over the whole cache and the indexer is pure overhead.
        const n_sel: i64 = @min(@as(i64, mctx_mla.getNKv()), @as(i64, hp.indexer_top_k));
        var out = b.buildAttnMha(Q, k, v, null, sel, null, null, layer.wv_b, n_sel, d.kq_scale, il);
        b.cb(out, "kqv_out", il);

        // K3.1: attn_output *= sigmoid(g_proj(x)), then o_proj.
        if (layer.wqkv_gate != null) {
            const g = llama.impl.c.ggml_sigmoid(ctx, llama.impl.c.ggml_mul_mat(ctx, layer.wqkv_gate, inp_gate));
            out = llama.impl.c.ggml_mul(ctx, out, g);
            b.cb(out, "mla_gated", il);
        }

        out = llama.impl.c.ggml_mul_mat(ctx, layer.wo, out);
        b.cb(out, "mla_out", il);

        return .{ .out = out, .sel = sel };
    }

    /// Ports `llama_model_kimi_k31::graph`'s `build_latent_moe`
    /// (models/kimi-k31.cpp:653 @b1b770d90): down-project, run the routed
    /// experts in the latent space, norm, up-project; the shared experts
    /// stay at n_embd and read the input as it came.
    fn buildLatentMoe(self: *const Graph, cur: *llama.impl.c.ggml_tensor, layer: *const llama.model.Layer, n_embd_latent: i64, il: c_int) *llama.impl.c.ggml_tensor {
        _ = n_embd_latent;
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;

        const identity = cur;

        const routed_in: *llama.impl.c.ggml_tensor = if (layer.ffn_routed_down) |w| llama.impl.c.ggml_mul_mat(ctx, w, cur) else cur;

        // The router scores the full-width input while the experts take the
        // latent one, so the logits are computed here and handed to
        // build_moe_ffn.
        const logits: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.ffn_gate_inp, identity);
        b.cb(logits, "ffn_moe_logits", il);

        var moe_out = b.buildMoeFfn(routed_in, .{
            .gate_inp = null, // unused: the logits above are passed instead
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = layer.ffn_exp_probs_b,
            .n_expert = hp.n_expert,
            .n_expert_used = hp.nExpertUsed(0),
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
            moe_out = llama.impl.c.ggml_mul_mat(ctx, w, moe_out);
        }

        if (layer.ffn_gate_shexp != null) {
            const g = llama.impl.c.ggml_mul_mat(ctx, layer.ffn_gate_shexp, identity);
            const u = llama.impl.c.ggml_mul_mat(ctx, layer.ffn_up_shexp, identity);
            var sh = situ(ctx, g, u, hp.situ_beta, hp.situ_linear_beta);
            sh = llama.impl.c.ggml_mul_mat(ctx, layer.ffn_down_shexp, sh);
            b.cb(sh, "ffn_shexp", il);
            moe_out = llama.impl.c.ggml_add(ctx, moe_out, sh);
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
