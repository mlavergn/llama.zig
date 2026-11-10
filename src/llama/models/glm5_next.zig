//! GLM5-Next (GLM-5.3-Flash): Kimi delta attention (KDA) layers between
//! nope-only MLA layers whose sparse attention a k-pool DSA indexer picks,
//! inside hyper-connections that carry `hc` residual streams, then a
//! DeepSeek-style mixture of experts with shared experts.
//!
//! The k-pool indexer scores pools of `indexer_kpool` consecutive tokens
//! rather than single tokens: each new token caches its indexer key and a
//! pool gate; once a pool is complete its keys are softmax-pooled by the
//! gate (plus a learned position embedding) into one pooled key, cached in
//! the same row. Each query attends to the cells of its top
//! `indexer_top_k / kpool` pools and the incomplete tail. "Shared" indexer
//! layers reuse the previous full layer's selection. Where every pool lives
//! is the memory's: `memory_hybrid_idx.zig`'s k-pool layout.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/glm5-next.cpp` at v0.9.0
//! (`cd959bae9`), with its declaration in `llama.cpp/src/models/models.h`:
//! `llama_model_glm5_next`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, `glm5_conv1d`, the `llm_graph_input_kpool` input,
//! the view and affine helpers, and its `graph` with `build_inp_kpool`, the
//! hyper-connection builders, `build_kda_layer`, `build_kpool_select` and
//! `build_dsa_layer`, and the NextN (MTP) draft head `graph_mtp`: one DSA
//! block after the trunk.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const batch = @import("../batch.zig");
const memory_hybrid = @import("../memory_hybrid.zig");
const memory_hybrid_idx = @import("../memory_hybrid_idx.zig");
const arch = @import("../arch.zig");
const delta_net = @import("delta_net.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Layer = model_mod.Layer;
const Builder = graph.Builder;
const Tensor = c.ggml_tensor;
const Ubatch = batch.Ubatch;
const HybridContext = memory_hybrid.HybridContext;
const HybridIdxContext = memory_hybrid_idx.HybridIdxContext;
const allocator = impl.allocator;

const src = "models/glm5-next.cpp";

fn oom() noreturn {
    impl.abort(src, 0, "out of memory");
}

/// Ports `llama_model_glm5_next`'s `load_arch_hparams` (models/glm5-next.cpp:7 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(f32, .attention_layernorm_eps, false, exc)) |v| hp.f_norm_eps = v;
    hp.n_embd_head_k_mla_impl = (try meta.getKey(u32, .attention_key_length_mla, true, exc)).?;
    hp.n_embd_head_v_mla_impl = (try meta.getKey(u32, .attention_value_length_mla, true, exc)).?;
    hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.n_embd_head_kda = (try meta.getKey(u32, .kda_head_dim, true, exc)).?;
    if (try meta.getKey(f32, .kda_gate_lower_bound, false, exc)) |v| hp.kda_gate_lower_bound = v;

    // The MLA cache holds the compressed latent.
    hp.n_embd_head_v_full = hp.n_lora_kv;

    // n_head_kv == 0 marks a KDA (recurrent) layer, the NextN block too.
    var i: u32 = 0;
    while (i < hp.n_layer_all) : (i += 1) {
        hp.is_recr_impl[i] = @intFromBool(hp.nHeadKv(i) == 0);
    }

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }
    _ = try meta.getKeyOrArr(f32, .swiglu_clamp_exp, &hp.swiglu_clamp_exp, hp.n_layer_all, false, exc);
    if (!try meta.getKeyOrArr(f32, .swiglu_clamp_shexp, &hp.swiglu_clamp_shexp, hp.n_layer_all, false, exc)) {
        hp.swiglu_clamp_shexp = hp.swiglu_clamp_exp;
    }

    // The DSA indexer, with k-pool compression.
    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;
    hp.indexer_kpool = (try meta.getKey(u32, .attention_indexer_kpool, true, exc)).?;
    if (try meta.getKey(bool, .attention_indexer_kpool_select_tail, false, exc)) |v| hp.indexer_kpool_select_tail = v;
    impl.assert(hp.indexer_kpool > 1 and hp.indexer_top_k % hp.indexer_kpool == 0, src, 45, "hparams.indexer_kpool > 1 && hparams.indexer_top_k % hparams.indexer_kpool == 0");
    @memset(&hp.is_indexer_full_impl, 1);
    _ = try meta.getKeyOrArr(u32, .attention_indexer_types, &hp.is_indexer_full_impl, hp.nLayer(), false, exc);

    // mHC.
    hp.dsv4_hc_mult = (try meta.getKey(u32, .hyper_connection_count, true, exc)).?;
    hp.dsv4_hc_sinkhorn_iters = (try meta.getKey(u32, .hyper_connection_sinkhorn_iterations, true, exc)).?;
    hp.dsv4_hc_eps = (try meta.getKey(f32, .hyper_connection_epsilon, true, exc)).?;
    impl.assert(hp.dsv4_hc_mult == 4, src, 53, "hparams.dsv4_hc_mult == 4 && \"mHC with hc_mult != 4 is not supported\"");

    self.type = switch (hp.nLayer()) {
        45 => .@"320b_a18b", // GLM-5.3-Flash
        else => .unknown,
    };
}

/// Ports `llama_model_glm5_next`'s `load_arch_tensors` (models/glm5-next.cpp:61 @cd959bae9).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const n_embd = l.n_embd;
    const n_head = l.n_head;

    const hc: i64 = hp.dsv4_hc_mult;
    const hc_mix_dim = (2 + hc) * hc;

    const nf = self.nextnFlags(ml, .attn_norm);
    const trunk_flags: u32 = nf.trunk;
    const mtp_flags: u32 = nf.mtp;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const ui: u32 = @intCast(i);

        const is_nextn = i >= l.n_layer;
        const flags: u32 = if (is_nextn) mtp_flags else trunk_flags;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{n_embd}, flags, exc);
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{n_embd}, flags, exc);

        if (i < l.n_layer) {
            layer.hc_attn_fn = try self.createTensor(.{ .tensor = .hc_attn_fn, .suffix = "weight", .bid = i }, &.{ hc * n_embd, hc_mix_dim }, flags, exc);
            layer.hc_attn_base = try self.createTensor(.{ .tensor = .hc_attn_base, .suffix = "weight", .bid = i }, &.{hc_mix_dim}, flags, exc);
            layer.hc_attn_scale = try self.createTensor(.{ .tensor = .hc_attn_scale, .suffix = "weight", .bid = i }, &.{3}, flags, exc);
            layer.hc_ffn_fn = try self.createTensor(.{ .tensor = .hc_ffn_fn, .suffix = "weight", .bid = i }, &.{ hc * n_embd, hc_mix_dim }, flags, exc);
            layer.hc_ffn_base = try self.createTensor(.{ .tensor = .hc_ffn_base, .suffix = "weight", .bid = i }, &.{hc_mix_dim}, flags, exc);
            layer.hc_ffn_scale = try self.createTensor(.{ .tensor = .hc_ffn_scale, .suffix = "weight", .bid = i }, &.{3}, flags, exc);
        }

        const head_dim: i64 = hp.n_embd_head_kda;
        const d_conv: i64 = hp.ssm_d_conv;
        const d_inner = head_dim * n_head;

        if (hp.isRecr(ui)) {
            // The conv1d weights are 4D, or 3D when quantization drops the
            // trailing 1.
            layer.ssm_q_conv = try loadConv(self, .ssm_conv1d_q, i, d_conv, d_inner, flags, exc);
            layer.ssm_k_conv = try loadConv(self, .ssm_conv1d_k, i, d_conv, d_inner, flags, exc);
            layer.ssm_v_conv = try loadConv(self, .ssm_conv1d_v, i, d_conv, d_inner, flags, exc);

            try self.createTensorQkv(layer, i, n_embd, d_inner, d_inner, d_inner, flags, exc);

            layer.ssm_f_a = try self.createTensor(.{ .tensor = .ssm_f_a, .suffix = "weight", .bid = i }, &.{ n_embd, head_dim }, flags, exc);
            layer.ssm_f_b = try self.createTensor(.{ .tensor = .ssm_f_b, .suffix = "weight", .bid = i }, &.{ head_dim, d_inner }, flags, exc);
            layer.ssm_beta = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "weight", .bid = i }, &.{ n_embd, n_head }, flags, exc);

            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = i }, &.{n_head}, flags, exc);
            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{d_inner}, flags, exc);

            layer.ssm_g_a = try self.createTensor(.{ .tensor = .ssm_g_a, .suffix = "weight", .bid = i }, &.{ n_embd, head_dim }, flags, exc);
            layer.ssm_g_b = try self.createTensor(.{ .tensor = .ssm_g_b, .suffix = "weight", .bid = i }, &.{ head_dim, d_inner }, flags, exc);
            layer.ssm_o_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = i }, &.{head_dim}, flags, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ d_inner, n_embd }, flags, exc);
        } else {
            const q_lora_rank: i64 = hp.n_lora_q;
            const kv_lora_rank: i64 = hp.n_lora_kv;
            const n_embd_head_k: i64 = hp.nEmbdHeadKMla();
            const n_embd_head_v: i64 = hp.nEmbdHeadVMla();
            const qk_rope_head_dim: i64 = hp.nRot(0);
            const qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

            layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, flags, exc);
            layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, flags, exc);

            layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ n_embd, q_lora_rank }, flags, exc);
            layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, n_head * n_embd_head_k }, flags, exc);

            layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ n_embd, kv_lora_rank + qk_rope_head_dim }, flags, exc);
            layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ qk_nope_head_dim, kv_lora_rank, n_head }, flags, exc);
            layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v, n_head }, flags, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_head * n_embd_head_v, n_embd }, flags, exc);

            const n_indexer_head: i64 = hp.indexer_n_head;
            const n_embd_indexer: i64 = hp.indexer_head_size;
            const kpool: i64 = hp.indexer_kpool;

            // A shared indexer layer has no indexer of its own; the NextN
            // block always has a full one.
            const full = i >= l.n_layer or hp.isIndexerFull(ui);
            const iflags = flags | (if (full) 0 else nr);

            layer.indexer_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{n_embd_indexer}, iflags, exc);
            layer.indexer_k_norm_b = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "bias", .bid = i }, &.{n_embd_indexer}, iflags, exc);
            layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ n_embd, n_indexer_head }, iflags, exc);
            layer.indexer_attn_k = try self.createTensor(.{ .tensor = .indexer_attn_k, .suffix = "weight", .bid = i }, &.{ n_embd, n_embd_indexer }, iflags, exc);
            layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, n_indexer_head * n_embd_indexer }, iflags, exc);
            layer.indexer_kpool_gate = try self.createTensor(.{ .tensor = .indexer_kpool_gate, .suffix = "weight", .bid = i }, &.{ n_embd, n_embd_indexer }, iflags, exc);
            layer.indexer_kpool_ape = try self.createTensor(.{ .tensor = .indexer_kpool_ape, .suffix = "weight", .bid = i }, &.{ n_embd_indexer, kpool }, iflags, exc);
        }

        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_ff }, flags, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, n_embd }, flags, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_ff }, flags, exc);
        } else {
            const n_ff_exp: i64 = hp.nFfExp(ui);
            const n_expert_shared: i64 = hp.n_expert_shared;

            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_expert }, flags, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, flags, exc);

            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_exp, l.n_expert }, flags, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, n_embd, l.n_expert }, flags, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_exp, l.n_expert }, flags, exc);

            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_exp * n_expert_shared }, flags, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, n_embd }, flags, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff_exp * n_expert_shared }, flags, exc);
        }

        if (i >= l.n_layer) {
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * n_embd, n_embd }, flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{n_embd}, flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{n_embd}, flags, exc);
            layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_vocab }, flags | nr, exc);
            layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ n_embd, l.n_vocab }, flags | nr, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{n_embd}, flags | nr, exc);
        }
    }
}

/// `conv`, a lambda of `load_arch_tensors` (models/glm5-next.cpp:61
/// @cd959bae9): a conv1d weight, 4D or else 3D.
fn loadConv(self: *Model, tid: arch.Tensor, il: i32, d_conv: i64, d_inner: i64, flags: u32, exc: *Exception) Thrown!?*Tensor {
    const t = try self.createTensor(.{ .tensor = tid, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner, 1 }, loader.tensor_not_required, exc);
    return t orelse try self.createTensor(.{ .tensor = tid, .suffix = "weight", .bid = il }, &.{ d_conv, 1, d_inner }, flags, exc);
}

/// Ports `llama_model_glm5_next`'s `build_arch_graph` (models/glm5-next.cpp:186
/// @cd959bae9): the NextN draft head for an MTP context, else the trunk.
pub fn buildArchGraph(model: *const Model, b: *const Builder, params: *const graph.Params) void {
    var g: Graph = .{ .model = model, .b = b };
    if (params.gtype == .decoder_mtp) {
        g.buildMtp();
        return;
    }
    g.build();
}

/// Ports `glm5_conv1d` (models/glm5-next.cpp:194 @cd959bae9): the causal
/// conv1d over one of Q (`qkv` 0), K (1) or V (2), writing `K_rs` conv
/// windows back, group `s` the window `s` tokens back.
fn conv1d(gf: *c.ggml_cgraph, ctx: *c.ggml_context, conv_states_all: *Tensor, conv_state_all: *Tensor, qkv: i64, x: *Tensor, proj_w: ?*Tensor, conv_w: ?*Tensor, d_conv: i64, head_dim: i64, n_head: i64, n_seq_tokens: i64, n_seqs: i64, n_tokens: i64, kv_head: i64, mem_size: i64, K_rs: i64) *Tensor {
    const d_inner = head_dim * n_head;
    const conv_state_size = (d_conv - 1) * d_inner;
    const n_embd_r_total = 3 * conv_state_size;

    const es_state: usize = c.ggml_element_size(conv_state_all);
    const conv_state_x = c.ggml_view_3d(ctx, conv_state_all, d_conv - 1, d_inner, n_seqs, @as(usize, @intCast(d_conv - 1)) * es_state, @as(usize, @intCast(n_embd_r_total)) * es_state, @as(usize, @intCast(qkv * conv_state_size)) * es_state);

    const x_proj = c.ggml_mul_mat(ctx, proj_w, x);
    const x_3d = c.ggml_reshape_3d(ctx, x_proj, d_inner, n_seq_tokens, n_seqs);
    const conv_x: *Tensor = c.ggml_concat(ctx, conv_state_x, c.ggml_transpose(ctx, x_3d), 0);

    // Group s holds the conv window s tokens back.
    // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K_rs tokens must share one
    // ubatch.
    const es_all: usize = c.ggml_element_size(conv_states_all);
    var s: i64 = 0;
    while (s < K_rs) : (s += 1) {
        const s_idx: i64 = @max(0, n_seq_tokens - s);
        const conv_x_s = c.ggml_view_3d(ctx, conv_x, d_conv - 1, d_inner, n_seqs, conv_x.nb[1], conv_x.nb[2], @as(usize, @intCast(s_idx)) * conv_x.nb[0]);
        c.ggml_build_forward_expand(gf, c.ggml_cpy(ctx, conv_x_s, c.ggml_view_3d(ctx, conv_states_all, d_conv - 1, d_inner, n_seqs, @as(usize, @intCast(d_conv - 1)) * es_all, @as(usize, @intCast(n_embd_r_total)) * es_all, @as(usize, @intCast((s * mem_size + kv_head) * n_embd_r_total + qkv * conv_state_size)) * es_all)));
    }

    const conv_weight = c.ggml_reshape_2d(ctx, conv_w, d_conv, d_inner);
    var Xcur = c.ggml_ssm_conv(ctx, conv_x, conv_weight);
    Xcur = c.ggml_reshape_2d(ctx, Xcur, d_inner, n_tokens);
    Xcur = c.ggml_silu(ctx, Xcur);

    return c.ggml_reshape_4d(ctx, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

/// Ports `llama_model_glm5_next`'s `llm_graph_input_kpool` (models/glm5-next.cpp:237
/// @cd959bae9): the k-pool indexer's inputs, which the memory's k-pool
/// layout fills.
pub const InpKpool = struct {
    /// I64 `[n_tokens]`.
    k_idxs: ?*Tensor = null,
    /// I32 `[n_pool]`: the cell caching each pool's pooled key.
    pool_cells: ?*Tensor = null,
    /// I32 `[kpool, n_pool]`: member cells per pool, `n_kv` for the padded
    /// pools.
    pool_idxs: ?*Tensor = null,
    /// F32/F16 `[n_pool, n_tokens]`.
    pool_mask: ?*Tensor = null,
    /// I32 `[kpool - 1, n_tokens]`.
    tail_idxs: ?*Tensor = null,
    /// F32 `[n_sel, 1, 1, n_tokens]`: 0 for live selection slots, -inf for
    /// dead ones. Null under the fused top-k mask.
    sel_mask: ?*Tensor = null,
    /// I32 `[kpool*n_new]`: the members of the pools completed this ubatch.
    /// `n_new` is never below 1, see `buildInpKpool`.
    new_pool_idxs: ?*Tensor = null,
    /// I64 `[n_new]`: the cell to write each new pooled key into.
    new_pool_rep: ?*Tensor = null,

    mctx: *const HybridIdxContext,
    kpool: u32,
    n_new: u32 = 0,
    n_sel: u32 = 0,
    n_kv: u32 = 0,

    /// Ports `llm_graph_input_kpool`'s `set_input` (models/glm5-next.cpp:242 @cd959bae9).
    pub fn setInput(self: *InpKpool, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        self.mctx.getIdx().?.setInputKIdxs(self.k_idxs.?, ubatch);
        self.mctx.setInputKpool(self.pool_cells.?, self.pool_idxs.?, self.pool_mask.?, self.tail_idxs.?, self.sel_mask, self.new_pool_idxs.?, self.new_pool_rep.?, ubatch, null);
    }

    /// Ports `llm_graph_input_kpool`'s `can_reuse` (models/glm5-next.cpp:247 @cd959bae9).
    pub fn canReuse(self: *InpKpool, params: *const graph.Params) bool {
        self.mctx = HybridIdxContext.fromBase(params.mctx.?.cast(HybridContext));

        const idx = self.mctx.getIdx() orelse return false;

        const n_tokens: i64 = params.ubatch.n_tokens;

        var res = true;

        res = res and self.k_idxs.?.ne[0] == n_tokens;
        res = res and self.pool_cells.?.ne[0] == self.mctx.getNKpool();
        res = res and self.pool_mask.?.ne[1] == n_tokens;
        res = res and self.tail_idxs.?.ne[1] == n_tokens;
        // The scatter mask's shape follows n_kv.
        res = res and self.n_kv == idx.getNKv();
        res = res and self.n_new == @max(self.mctx.getNKpoolNew(), 1);

        return res;
    }
};

/// Ports `dsv4_elem_offset` (models/glm5-next.cpp:344 @cd959bae9).
fn elemOffset(t: *const Tensor, i: i64) usize {
    return c.ggml_row_size(t.type, i);
}

/// Ports `dsv4_view_1d` (models/glm5-next.cpp:348 @cd959bae9).
fn view1d(ctx: *c.ggml_context, t: *Tensor, ne0: i64, j0: i64) *Tensor {
    return c.ggml_view_1d(ctx, t, ne0, elemOffset(t, j0));
}

/// Ports `dsv4_view_2d` (models/glm5-next.cpp:352 @cd959bae9).
fn view2d(ctx: *c.ggml_context, t: *Tensor, ne0: i64, ne1: i64, j0: i64) *Tensor {
    return c.ggml_view_2d(ctx, t, ne0, ne1, t.nb[1], elemOffset(t, j0));
}

/// Ports `dsv4_hc_affine` (models/glm5-next.cpp:371 @cd959bae9).
fn hcAffine(ctx: *c.ggml_context, x_: *Tensor, scale: *Tensor, base: *Tensor) *Tensor {
    var x: *Tensor = c.ggml_mul(ctx, x_, scale);
    x = c.ggml_add(ctx, x, base);
    return x;
}

/// Mirrors `llama_model_glm5_next`'s `graph` (src/models/models.h:3082 @cd959bae9): the
/// builder and the model. Its `no_build` constructor
/// (src/models/models.h:3087 @cd959bae9), which builds nothing so that `graph_mtp` can reuse the
/// helpers, is this struct before `build` or `buildMtp` runs.
const Graph = struct {
    model: *const Model,
    b: *const Builder,

    /// Ports `llama_model_glm5_next::graph`'s `build_inp_kpool`
    /// (models/glm5-next.cpp:285 @cd959bae9).
    fn buildInpKpool(self: *const Graph, mctx_hyb: *const HybridIdxContext) *InpKpool {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_tokens = b.n_tokens;

        const mctx_idx = mctx_hyb.getIdx();
        impl.assert(mctx_idx != null, src, 287, "mctx_idx != nullptr");

        const kpool: u32 = hp.indexer_kpool;
        const n_pool: u32 = mctx_hyb.getNKpool();
        const n_kv: u32 = mctx_idx.?.getNKv();
        // A ubatch that completes no pool still builds one dummy entry, so
        // the graph does not change shape every kpool tokens.
        const n_new: u32 = @max(mctx_hyb.getNKpoolNew(), 1);

        // The fused lightning indexer wants an f16 mask.
        const type_mask: c.ggml_type = if (b.cparams.fused_lid) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;

        const inp = allocator.create(InpKpool) catch oom();
        inp.* = .{ .mctx = mctx_hyb, .kpool = kpool };

        inp.k_idxs = mctx_idx.?.buildInputKIdxs(ctx, b.ubatch);
        inp.pool_cells = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n_pool);
        inp.pool_idxs = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, kpool, n_pool);
        inp.pool_mask = c.ggml_new_tensor_2d(ctx, type_mask, n_pool, n_tokens);
        inp.tail_idxs = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, kpool - 1, n_tokens);
        c.ggml_set_input(inp.pool_cells);
        c.ggml_set_input(inp.pool_idxs);
        c.ggml_set_input(inp.pool_mask);
        c.ggml_set_input(inp.tail_idxs);

        c.ggml_build_forward_expand(b.gf, inp.pool_cells);
        c.ggml_build_forward_expand(b.gf, inp.pool_idxs);
        c.ggml_build_forward_expand(b.gf, inp.pool_mask);
        c.ggml_build_forward_expand(b.gf, inp.tail_idxs);

        inp.n_kv = n_kv;

        // The selection width: the top pools plus the optional tail, also
        // the sparse attention's bound.
        {
            const n_top_pool: i64 = @min(@as(i64, n_pool), @as(i64, hp.indexer_top_k / kpool));
            const n_sel: i64 = @as(i64, kpool) * n_top_pool + (if (hp.indexer_kpool_select_tail) @as(i64, kpool) - 1 else 0);
            inp.n_sel = @intCast(n_sel);

            // The scatter maps the dead slots of this mask to dump rows; the
            // fused mask op reads liveness from the scores instead.
            if (!b.cparams.fused_top_k_mask) {
                inp.sel_mask = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, n_sel, 1, 1, n_tokens);
                c.ggml_set_input(inp.sel_mask);
                // set_input_kpool always fills the mask, so it stays
                // allocated in every graph.
                c.ggml_build_forward_expand(b.gf, inp.sel_mask);
            }
        }

        inp.n_new = n_new;

        inp.new_pool_idxs = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, @as(i64, kpool) * n_new);
        c.ggml_set_input(inp.new_pool_idxs);
        // One scatter row per new pool, each a distinct rep row (see
        // kpool_build_state).
        inp.new_pool_rep = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I64, n_new);
        c.ggml_set_input(inp.new_pool_rep);

        return b.res.addInput(InpKpool, inp);
    }

    /// Ports `llama_model_glm5_next::graph`'s `build_hc_mean`
    /// (models/glm5-next.cpp:361 @cd959bae9): the mean over the
    /// hyper-connection streams, `[n_embd, hc, n_tokens]` to `[n_embd,
    /// n_tokens]`.
    fn buildHcMean(self: *const Graph, x: *Tensor) *Tensor {
        const ctx = self.b.ctx0;
        const hc = x.ne[1];

        var acc: *Tensor = c.ggml_view_2d(ctx, x, x.ne[0], x.ne[2], x.nb[2], 0);
        var s: i64 = 1;
        while (s < hc) : (s += 1) {
            acc = c.ggml_add(ctx, acc, c.ggml_view_2d(ctx, x, x.ne[0], x.ne[2], x.nb[2], @as(usize, @intCast(s)) * x.nb[1]));
        }
        return c.ggml_scale(ctx, acc, 1.0 / @as(f32, @floatFromInt(hc)));
    }

    /// Ports the overload of `llama_model_glm5_next::graph`'s `build_hc_pre`
    /// that takes the weights (models/glm5-next.cpp:381 @cd959bae9): the
    /// streams' weighted sum.
    fn buildHcPreWeighted(self: *const Graph, x: *Tensor, weights: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        impl.assert(x.ne[0] == b.n_embd, src, 385, "x->ne[0] == n_embd");
        impl.assert(x.ne[1] == b.hparams.dsv4_hc_mult, src, 386, "x->ne[1] == hparams.dsv4_hc_mult");

        const hc: i64 = b.hparams.dsv4_hc_mult;
        const nt = x.ne[2];

        if (b.cparams.fused_dsv4_hc_pre and il >= 0) {
            const result: *Tensor = c.ggml_dsv4_hc_pre(ctx, x, weights);
            b.res.addFusedNode(.{ .op = .dsv4_hc_pre, .tensor = result, .il = il });
            return result;
        }

        var result: ?*Tensor = null;
        var ih: i64 = 0;
        while (ih < hc) : (ih += 1) {
            const uih: usize = @intCast(ih);
            const xh: *Tensor = c.ggml_view_2d(ctx, x, b.n_embd, nt, x.nb[2], uih * x.nb[1]);
            const wh: *Tensor = c.ggml_view_2d(ctx, weights, 1, nt, weights.nb[1], uih * weights.nb[0]);
            const cur: *Tensor = c.ggml_mul(ctx, xh, wh);
            result = if (result) |r| c.ggml_add(ctx, r, cur) else cur;
        }

        return result.?;
    }

    /// Ports `llama_model_glm5_next::graph`'s `build_hc_sinkhorn`
    /// (models/glm5-next.cpp:408 @cd959bae9). `comb` is `[dst_hc, src_hc,
    /// n_tokens]`: a row softmax over dst, one column normalization, then
    /// repeated row and column normalization.
    fn buildHcSinkhorn(self: *const Graph, comb_: *Tensor, il: c_int) *Tensor {
        _ = il;
        const b = self.b;
        const ctx = b.ctx0;

        var comb: *Tensor = c.ggml_soft_max(ctx, comb_);

        var eps: *Tensor = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 1);
        eps = c.ggml_fill(ctx, eps, b.hparams.dsv4_hc_eps);

        comb = c.ggml_add(ctx, comb, eps);

        const Norm = struct {
            fn cols(ctx_: *c.ggml_context, comb_p: **Tensor, eps_: *Tensor) void {
                const comb_src_dst: *Tensor = c.ggml_cont(ctx_, c.ggml_permute(ctx_, comb_p.*, 1, 0, 2, 3));
                var col_sum: *Tensor = c.ggml_sum_rows(ctx_, comb_src_dst);
                col_sum = c.ggml_add(ctx_, col_sum, eps_);
                col_sum = c.ggml_permute(ctx_, col_sum, 1, 0, 2, 3);
                comb_p.* = c.ggml_div(ctx_, comb_p.*, col_sum);
            }

            fn rows(ctx_: *c.ggml_context, comb_p: **Tensor, eps_: *Tensor) void {
                var row_sum: *Tensor = c.ggml_sum_rows(ctx_, comb_p.*);
                row_sum = c.ggml_add(ctx_, row_sum, eps_);
                comb_p.* = c.ggml_div(ctx_, comb_p.*, row_sum);
            }
        };

        Norm.cols(ctx, &comb, eps);
        var i: u32 = 1;
        while (i < b.hparams.dsv4_hc_sinkhorn_iters) : (i += 1) {
            Norm.rows(ctx, &comb, eps);
            Norm.cols(ctx, &comb, eps);
        }

        return comb;
    }

    /// Mirrors the graph's `hc_pre` (src/models/models.h:3099 @cd959bae9):
    /// `cur` is the collapsed input,
    /// `post` and `comb` feed `build_hc_post`.
    const HcPre = struct {
        cur: *Tensor,
        post: *Tensor,
        comb: *Tensor,
    };

    /// Ports `llama_model_glm5_next::graph`'s `build_hc_pre`
    /// (models/glm5-next.cpp:446 @cd959bae9): a hyper-connection's mixes --
    /// the pre weights it collapses the streams with, here; the post weights
    /// and the stream combination for `build_hc_post`.
    fn buildHcPre(self: *const Graph, x: *Tensor, hc_fn: *Tensor, hc_scale: *Tensor, hc_base: *Tensor, il: c_int) HcPre {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const hc: i64 = hp.dsv4_hc_mult;
        const hc_dim = hc * b.n_embd;
        const hc_mix_dim = (2 + hc) * hc;
        const nt = x.ne[2];

        impl.assert(hc == 4, src, 457, "hc == 4");
        impl.assert(hc_fn.ne[1] == hc_mix_dim, src, 458, "hc_fn->ne[1] == hc_mix_dim");

        const flat: *Tensor = c.ggml_reshape_2d(ctx, x, hc_dim, nt);
        const flat_norm: *Tensor = c.ggml_rms_norm(ctx, flat, b.norm_rms_eps);
        const mixes: *Tensor = c.ggml_mul_mat(ctx, hc_fn, flat_norm);
        b.cb(mixes, "hc_mixes", il);

        const scale_pre = view1d(ctx, hc_scale, 1, 0);
        const scale_post = view1d(ctx, hc_scale, 1, 1);

        const base_pre = view1d(ctx, hc_base, hc, 0);
        const base_post = view1d(ctx, hc_base, hc, hc);

        var pre = view2d(ctx, mixes, hc, nt, 0);
        pre = hcAffine(ctx, pre, scale_pre, base_pre);
        pre = c.ggml_sigmoid(ctx, pre);
        pre = c.ggml_scale_bias(ctx, pre, 1.0, hp.dsv4_hc_eps);
        b.cb(pre, "hc_pre", il);

        var post = view2d(ctx, mixes, hc, nt, hc);
        post = hcAffine(ctx, post, scale_post, base_post);
        post = c.ggml_sigmoid(ctx, post);
        post = c.ggml_scale(ctx, post, 2.0);
        b.cb(post, "hc_post", il);

        var comb: *Tensor = undefined;
        if (b.cparams.fused_dsv4_hc_comb) {
            comb = c.ggml_dsv4_hc_comb(ctx, mixes, hc_scale, hc_base, hp.dsv4_hc_eps, @intCast(hp.dsv4_hc_sinkhorn_iters));
            b.res.addFusedNode(.{ .op = .dsv4_hc_comb, .tensor = comb, .il = il });
        } else {
            const scale_comb = view1d(ctx, hc_scale, 1, 2);
            const base_comb = view1d(ctx, hc_base, hc * hc, 2 * hc);

            comb = view2d(ctx, mixes, hc * hc, nt, 2 * hc);
            comb = hcAffine(ctx, comb, scale_comb, base_comb);
            comb = c.ggml_reshape_3d(ctx, comb, hc, hc, nt);
            comb = self.buildHcSinkhorn(comb, il);
        }
        b.cb(comb, "hc_comb", il);

        return .{ .cur = self.buildHcPreWeighted(x, pre, il), .post = post, .comb = comb };
    }

    /// Ports `llama_model_glm5_next::graph`'s `build_hc_post`
    /// (models/glm5-next.cpp:502 @cd959bae9): the block's output spread over
    /// the streams by `post`, plus the residual streams mixed by `comb`.
    fn buildHcPost(self: *const Graph, x: *Tensor, residual: *Tensor, post: *Tensor, comb: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        impl.assert(x.ne[0] == b.n_embd, src, 508, "x->ne[0] == n_embd");
        impl.assert(residual.ne[1] == b.hparams.dsv4_hc_mult, src, 509, "residual->ne[1] == hparams.dsv4_hc_mult");

        if (b.cparams.fused_dsv4_hc_post) {
            const result: *Tensor = c.ggml_dsv4_hc_post(ctx, x, residual, post, comb);
            b.res.addFusedNode(.{ .op = .dsv4_hc_post, .tensor = result, .il = il });
            return result;
        }

        const hc: i64 = b.hparams.dsv4_hc_mult;
        const nt = x.ne[1];

        var out: ?*Tensor = null;
        var dst: i64 = 0;
        while (dst < hc) : (dst += 1) {
            const udst: usize = @intCast(dst);
            const post_dst: *Tensor = c.ggml_view_2d(ctx, post, 1, nt, post.nb[1], udst * post.nb[0]);
            var cur: *Tensor = c.ggml_mul(ctx, x, post_dst);

            var s: i64 = 0;
            while (s < hc) : (s += 1) {
                const us: usize = @intCast(s);
                const res_src: *Tensor = c.ggml_view_2d(ctx, residual, b.n_embd, nt, residual.nb[2], us * residual.nb[1]);
                const comb_src_dst: *Tensor = c.ggml_view_2d(ctx, comb, 1, nt, comb.nb[2], udst * comb.nb[0] + us * comb.nb[1]);
                cur = c.ggml_add(ctx, cur, c.ggml_mul(ctx, res_src, comb_src_dst));
            }

            cur = c.ggml_reshape_3d(ctx, cur, b.n_embd, 1, nt);
            out = if (out) |o| c.ggml_concat(ctx, o, cur, 1) else cur;
        }

        return out.?;
    }

    /// Ports the constructor of `llama_model_glm5_next`'s `graph`
    /// (models/glm5-next.cpp:674 @cd959bae9).
    fn build(self: *Graph) void {
        const model = self.model;
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_embd: i64 = b.n_embd;
        const n_tokens = b.n_tokens;

        const inp = b.buildInpEmbd(model.tok_embd.?);
        b.cb(inp, "inp_embd", -1);

        // The recurrent state and the K-only MLA cache through the generic
        // hybrid input, plus the indexer cache.
        const inp_hyb = b.buildInpMemHybridK();
        const mctx_hyb = HybridIdxContext.fromBase(inp_hyb.mctx);
        const inp_rs = inp_hyb.getRecr();
        const inp_attn = inp_hyb.getAttn();
        const inp_kpool = self.buildInpKpool(mctx_hyb);

        const inp_out_ids = b.buildInpOutIds();

        const n_head_kda: i64 = hp.nHead(0);
        const head_dim: i64 = hp.n_embd_head_kda;
        const d_conv: i64 = hp.ssm_d_conv;
        const d_inner = n_head_kda * head_dim;
        const n_seqs: i64 = b.ubatch.n_seqs;
        const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

        impl.assert(n_seqs != 0, src, 699, "n_seqs != 0");
        impl.assert(b.ubatch.equalSeqs(), src, 700, "ubatch.equal_seqs()");
        impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, src, 701, "ubatch.n_tokens == n_seq_tokens * n_seqs");

        const hc: i64 = hp.dsv4_hc_mult;
        var inpL: *Tensor = c.ggml_reshape_3d(ctx, inp, n_embd, 1, n_tokens);
        inpL = c.ggml_repeat_4d(ctx, inpL, n_embd, hc, n_tokens, 1);
        b.cb(inpL, "hc_init", -1);

        var prev_sel: ?*Tensor = null;

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];

            var residual = inpL;
            const attn_hc = self.buildHcPre(inpL, layer.hc_attn_fn.?, layer.hc_attn_scale.?, layer.hc_attn_base.?, il);
            var cur = attn_hc.cur;
            b.cb(cur, "hc_attn_pre", il);

            cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);
            c.ggml_build_forward_expand(b.gf, cur);

            if (hp.isRecr(@intCast(il))) {
                cur = self.buildKdaLayer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda, d_inner, n_seq_tokens, n_seqs, il);
            } else {
                const dsa = self.buildDsaLayer(cur, layer, mctx_hyb, inp_attn, inp_kpool, prev_sel, il);
                cur = dsa.cur;
                prev_sel = dsa.sel;
            }

            inpL = self.buildHcPost(cur, residual, attn_hc.post, attn_hc.comb, il);
            b.cb(inpL, "hc_attn_post", il);

            residual = inpL;
            const ffn_hc = self.buildHcPre(inpL, layer.hc_ffn_fn.?, layer.hc_ffn_scale.?, layer.hc_ffn_base.?, il);
            cur = ffn_hc.cur;
            b.cb(cur, "hc_ffn_pre", il);

            cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
                cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
                b.cb(cur, "ffn_out", il);
            } else {
                const moe_out = b.buildMoeFfn(cur, .{
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
                b.cb(moe_out, "ffn_moe_out", il);

                const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
                b.cb(ffn_shexp, "ffn_shexp", il);

                cur = c.ggml_add(ctx, moe_out, ffn_shexp);
                b.cb(cur, "ffn_out", il);
            }

            inpL = self.buildHcPost(cur, residual, ffn_hc.post, ffn_hc.comb, il);
            inpL = b.buildCvec(inpL, il);
            b.cb(inpL, "l_out", il);
        }

        // Narrow to the output tokens, then collapse the streams. Unmasked
        // nextn embeddings need every row.
        if (b.cropBeforeNextn(inp_out_ids)) {
            var flat: *Tensor = c.ggml_reshape_2d(ctx, inpL, n_embd * hc, n_tokens);
            flat = c.ggml_get_rows(ctx, flat, inp_out_ids);
            inpL = c.ggml_reshape_3d(ctx, flat, n_embd, hc, b.n_outputs);
        }

        var cur = self.buildHcMean(inpL);
        b.cb(cur, "hc_head", -1);

        cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

        // The post-norm hidden state feeds the draft head.
        b.cb(cur, "h_nextn", -1);
        b.res.t_h_nextn = cur;

        if (b.cropAfterNextn(inp_out_ids)) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
        }
        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        cur = c.ggml_mul_mat(ctx, model.output, cur);
        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;
    }

    /// Ports the constructor of `llama_model_glm5_next`'s `graph_mtp`
    /// (models/glm5-next.cpp:544 @cd959bae9): the NextN draft head, one DSA
    /// block after the trunk, fed by the trunk's post-norm hidden state.
    fn buildMtp(self: *Graph) void {
        const model = self.model;
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_embd: i64 = b.n_embd;
        const n_tokens = b.n_tokens;

        impl.assert(hp.n_layer_nextn > 0, src, 547, "hparams.n_layer_nextn > 0 && \"GLM5-Next MTP requires n_layer_nextn > 0\"");
        impl.assert(hp.n_layer_nextn == 1, src, 548, "hparams.n_layer_nextn == 1 && \"GLM5-Next MTP currently supports a single NextN block\"");

        const il: c_int = @as(c_int, @intCast(hp.nLayer())) + b.cparams.nextn_layer_offset;
        impl.assert(b.cparams.nextn_layer_offset >= 0 and b.cparams.nextn_layer_offset < @as(c_int, @intCast(hp.n_layer_nextn)), src, 553, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range [0, n_layer_nextn)\"");

        const layer = &model.layers[@intCast(il)];

        impl.assert(layer.nextn.eh_proj != null and layer.nextn.enorm != null and layer.nextn.hnorm != null, src, 558, "layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && \"GLM5-Next MTP block is missing - load the model with MTP enabled\"");

        impl.assert(hp.nEmbdOut() == @as(u32, @truncate(@as(u64, @bitCast(n_embd)))), src, 560, "hparams.n_embd_out() == (uint32_t) n_embd && \"GLM5-Next MTP hidden width mismatch\"");

        const inp_hyb = b.buildInpMemHybridK();
        const mctx_hyb = HybridIdxContext.fromBase(inp_hyb.mctx);
        const inp_attn = inp_hyb.getAttn();
        const inp_kpool = self.buildInpKpool(mctx_hyb);

        c.ggml_build_forward_expand(b.gf, inp_hyb.getRecr().s_copy);

        const inp_out_ids = b.buildInpOutIds();

        const n_embd_inp: i64 = hp.nEmbdInp();
        const inp = allocator.create(graph.InpEmbdH) catch oom();
        inp.* = .{ .n_embd = n_embd_inp, .n_embd_state = n_embd };

        inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n_tokens);
        c.ggml_set_input(inp.tokens);

        inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd_inp, n_tokens);
        c.ggml_set_input(inp.embd);

        inp.h = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd, n_tokens);
        c.ggml_set_input(inp.h);
        _ = c.ggml_set_name(inp.h, "mtp_h_input");

        var tok_embd: *Tensor = undefined;
        if (b.ubatch.token != null) {
            tok_embd = c.ggml_get_rows(ctx, layer.nextn.embed_tokens orelse model.tok_embd.?, inp.tokens);
        } else {
            tok_embd = inp.embd.?;
        }
        b.cb(tok_embd, "mtp_tok_embd", il);

        const h = inp.h.?;

        _ = b.res.addInput(graph.InpEmbdH, inp);

        const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
        b.cb(e_norm, "mtp_enorm", il);

        const h_norm = b.buildNorm(h, layer.nextn.hnorm, null, .rms, il);
        b.cb(h_norm, "mtp_hnorm", il);

        var cur = b.buildLoraMm(layer.nextn.eh_proj.?, c.ggml_concat(ctx, e_norm, h_norm, 0), layer.nextn.eh_proj_s);
        b.cb(cur, "mtp_eh_proj", il);

        var inpSA = cur;

        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "mtp_attn_norm", il);

        // The NextN block has a full indexer, nothing to carry in.
        const dsa = self.buildDsaLayer(cur, layer, mctx_hyb, inp_attn, inp_kpool, null, il);
        cur = dsa.cur;
        b.cb(cur, "mtp_attn_out", il);

        // Narrow to the output tokens before the position-wise FFN; unmasked
        // nextn embeddings need every row.
        if (b.cropBeforeNextn(inp_out_ids)) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "mtp_ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "mtp_ffn_norm", il);

        const moe_out = b.buildMoeFfn(cur, .{
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
        b.cb(moe_out, "mtp_ffn_moe_out", il);

        const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
        b.cb(ffn_shexp, "mtp_ffn_shexp", il);

        cur = c.ggml_add(ctx, moe_out, ffn_shexp);
        b.cb(cur, "mtp_ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "mtp_post_ffn", il);

        const head_norm = layer.nextn.shared_head_norm orelse model.output_norm;
        impl.assert(head_norm != null, src, 655, "head_norm && \"GLM5-Next MTP: missing both nextn.shared_head_norm and output_norm\"");
        cur = b.buildNorm(cur, head_norm, null, .rms, -1);
        b.cb(cur, "h_nextn", -1);
        b.res.t_h_nextn = cur;

        if (b.cropAfterNextn(inp_out_ids)) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
        }
        b.cb(cur, "mtp_shared_head_norm", -1);

        const head_w = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head else model.output;
        const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
        impl.assert(head_w != null, src, 667, "head_w && \"GLM5-Next MTP: missing both nextn.shared_head_head and output\"");
        cur = b.buildLoraMm(head_w.?, cur, head_s);
        b.cb(cur, "result_output", -1);

        b.res.t_logits = cur;
    }

    /// Ports `llama_model_glm5_next::graph`'s `build_kda_layer`
    /// (models/glm5-next.cpp:809 @cd959bae9): KDA with the g_a/g_b output
    /// gate.
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

        const mem_size: i64 = mctx_cur.getSize();
        const K_rs: i64 = @as(i64, b.cparams.n_rs_seq) + 1;

        var Qcur = conv1d(b.gf, ctx, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
        var Kcur = conv1d(b.gf, ctx, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
        const Vcur = conv1d(b.gf, ctx, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
        b.cb(Qcur, "kda_q_conv", il);
        b.cb(Kcur, "kda_k_conv", il);
        b.cb(Vcur, "kda_v_conv", il);

        // The decay gate; ssm_a holds -exp(A_log).
        const f_a = c.ggml_mul_mat(ctx, layer.ssm_f_a, cur);
        var g1: *Tensor = c.ggml_mul_mat(ctx, layer.ssm_f_b, f_a);
        g1 = c.ggml_add(ctx, g1, layer.ssm_dt_b);

        const A = c.ggml_reshape_3d(ctx, layer.ssm_a, 1, n_head_kda, 1);

        if (hp.kda_gate_lower_bound > -std.math.inf(f32)) {
            g1 = c.ggml_reshape_3d(ctx, g1, head_dim, n_head_kda, n_tokens);
            g1 = c.ggml_mul(ctx, g1, A);
            g1 = c.ggml_sigmoid(ctx, c.ggml_scale(ctx, g1, -1.0));
            g1 = c.ggml_scale(ctx, g1, hp.kda_gate_lower_bound);
        } else {
            g1 = c.ggml_softplus(ctx, g1);
            g1 = c.ggml_reshape_3d(ctx, g1, head_dim, n_head_kda, n_tokens);
            g1 = c.ggml_mul(ctx, g1, A);
        }
        b.cb(g1, "kda_g1", il);

        g1 = c.ggml_reshape_4d(ctx, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

        var beta: *Tensor = c.ggml_mul_mat(ctx, layer.ssm_beta, cur);
        beta = c.ggml_reshape_4d(ctx, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
        beta = c.ggml_sigmoid(ctx, beta);
        b.cb(beta, "kda_beta", il);

        const ssm_states_all = mctx_cur.getSL(il);
        var state = b.buildRs(inp_rs, ssm_states_all, @intCast(hp.nEmbdS()), @intCast(n_seqs), null);
        state = c.ggml_reshape_4d(ctx, state, head_dim, head_dim, n_head_kda, n_seqs);

        // FLA's l2 norm.
        const l2_eps: f32 = 1e-6;
        Qcur = delta_net.buildGdnL2Norm(ctx, Qcur, l2_eps);
        Kcur = delta_net.buildGdnL2Norm(ctx, Kcur, l2_eps);

        var output = delta_net.buildRecurrentAttn(b, inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g1, beta, state, il);
        output = c.ggml_cont(ctx, output);
        b.cb(output, "kda_scan_out", il);

        // The output gate, then RMSNorm(o) * sigmoid(g2).
        const g_a = c.ggml_mul_mat(ctx, layer.ssm_g_a, cur);
        var g2: *Tensor = c.ggml_mul_mat(ctx, layer.ssm_g_b, g_a);
        g2 = c.ggml_reshape_3d(ctx, g2, head_dim, n_head_kda, n_tokens);

        const o = c.ggml_reshape_3d(ctx, output, head_dim, n_head_kda, n_tokens);
        const normed = b.buildNorm(o, layer.ssm_o_norm, null, .rms, il);
        b.cb(g2, "kda_g2", il);
        b.cb(normed, "kda_normed", il);
        var gated: *Tensor = c.ggml_mul(ctx, normed, c.ggml_sigmoid(ctx, g2));

        gated = c.ggml_cont_2d(ctx, gated, d_inner, n_tokens);
        cur = c.ggml_mul_mat(ctx, layer.wo, gated);
        b.cb(cur, "kda_out", il);

        return cur;
    }

    /// Ports `llama_model_glm5_next::graph`'s `build_kpool_select`
    /// (models/glm5-next.cpp:889 @cd959bae9): score the pools of kpool
    /// consecutive tokens, and expand the selected pools and the
    /// incomplete tail into an additive `[n_kv, n_tokens]` mask.
    fn buildKpoolSelect(self: *const Graph, cur: *Tensor, qr: *Tensor, layer: *const Layer, mctx_hyb: *const HybridIdxContext, inp_kpool: *InpKpool, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_tokens = b.n_tokens;

        const mctx_lid = mctx_hyb.getIdx().?;

        const n_indexer_head: i64 = hp.indexer_n_head;
        const n_embd_indexer: i64 = hp.indexer_head_size;
        const kpool: i64 = hp.indexer_kpool;
        const n_pool: i64 = inp_kpool.pool_cells.?.ne[0];
        const n_new: i64 = inp_kpool.n_new;

        var iq: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_attn_q_b, qr);
        iq = c.ggml_reshape_3d(ctx, iq, n_embd_indexer, n_indexer_head, n_tokens);
        b.cb(iq, "indexer_q", il);

        // The per-token key and pool gate scores, cached together.
        var ik: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_attn_k, cur);
        ik = b.buildNorm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, .norm, il);
        b.cb(ik, "indexer_k", il);

        const ig: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_kpool_gate, cur);
        b.cb(ig, "indexer_gate", il);

        // A cache row is key | gate | pooled.
        const pzero: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd_indexer, n_tokens), 0.0);
        var packed_: *Tensor = c.ggml_concat(ctx, c.ggml_concat(ctx, ik, ig, 0), pzero, 0);
        packed_ = c.ggml_reshape_3d(ctx, packed_, 3 * n_embd_indexer, 1, n_tokens);
        c.ggml_build_forward_expand(b.gf, mctx_lid.cpyK(ctx, packed_, inp_kpool.k_idxs.?, il));

        const kpool_cache = mctx_hyb.getKpoolAccess(ctx, il, n_embd_indexer);
        const n_kv: i64 = mctx_lid.getNKv();

        // Flash attention requires an f16 mask.
        const type_mask: c.ggml_type = if (b.cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;

        // Pool the entries this ubatch completed. The last one is a dummy
        // when the ubatch completes none.
        var rows: *Tensor = kpool_cache.gatherKeyGate(inp_kpool.new_pool_idxs.?);
        rows = c.ggml_reshape_3d(ctx, rows, 2 * n_embd_indexer, kpool, n_new);

        var pk: *Tensor = c.ggml_view_3d(ctx, rows, n_embd_indexer, kpool, n_new, rows.nb[1], rows.nb[2], 0);
        const pg: *Tensor = c.ggml_view_3d(ctx, rows, n_embd_indexer, kpool, n_new, rows.nb[1], rows.nb[2], c.ggml_row_size(rows.type, n_embd_indexer));

        var logits: *Tensor = c.ggml_add(ctx, pg, layer.indexer_kpool_ape);
        logits = c.ggml_cont(ctx, c.ggml_permute(ctx, logits, 1, 0, 2, 3)); // [kpool, head_dim, n_new]
        // The pool axis is folded into rows: soft_max maps ne[2] to CUDA's
        // gridDim.y, which caps at 65535.
        var probs: *Tensor = c.ggml_soft_max(ctx, c.ggml_reshape_2d(ctx, logits, kpool, n_embd_indexer * n_new));
        probs = c.ggml_reshape_3d(ctx, probs, kpool, n_embd_indexer, n_new);

        pk = c.ggml_cont(ctx, c.ggml_permute(ctx, pk, 1, 0, 2, 3));
        var pooled_new: *Tensor = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, probs, pk)); // [1, head_dim, n_new]
        pooled_new = c.ggml_reshape_2d(ctx, pooled_new, n_embd_indexer, n_new);
        b.cb(pooled_new, "indexer_pool_k_new", il);

        // Scatter the fresh pooled keys into their rep rows, then gather all
        // n_pool of them by cell: the older pools come from the rows earlier
        // ubatches wrote.
        c.ggml_build_forward_expand(b.gf, kpool_cache.scatterPooled(pooled_new, inp_kpool.new_pool_rep.?));
        var pooled: *Tensor = kpool_cache.gatherPooled(inp_kpool.pool_cells.?);
        pooled = c.ggml_reshape_3d(ctx, pooled, n_embd_indexer, 1, n_pool);
        b.cb(pooled, "indexer_pool_k", il);

        var sel_idx: *Tensor = undefined;
        {
            var weights: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_proj, cur);
            weights = c.ggml_scale(ctx, weights, 1.0 / impl.sqrtf(@floatFromInt(n_embd_indexer * n_indexer_head)));
            b.cb(weights, "indexer_weights", il);

            var score: *Tensor = undefined;
            if (b.cparams.fused_lid) {
                score = c.ggml_lightning_indexer(ctx, iq, pooled, weights, inp_kpool.pool_mask);
                b.res.addFusedNode(.{ .op = .lightning_indexer, .tensor = score, .il = il });
            } else {
                const q_p = c.ggml_permute(ctx, iq, 0, 2, 1, 3); // [head_dim, n_tokens, n_head]
                const k_p = c.ggml_permute(ctx, pooled, 0, 2, 1, 3); // [head_dim, n_pool, 1]

                var kq: *Tensor = c.ggml_mul_mat(ctx, k_p, q_p); // [n_pool, n_tokens, n_head]
                kq = c.ggml_cont(ctx, c.ggml_permute(ctx, kq, 2, 1, 0, 3)); // [n_head, n_tokens, n_pool]
                score = c.ggml_relu(ctx, kq);
                score = c.ggml_mul(ctx, score, weights);
                score = c.ggml_sum_rows(ctx, score); // [1, n_tokens, n_pool]
                score = c.ggml_cont(ctx, c.ggml_permute(ctx, score, 2, 1, 0, 3)); // [n_pool, n_tokens, 1]
                score = c.ggml_add(ctx, score, inp_kpool.pool_mask);
            }
            b.cb(score, "indexer_score", il);

            const n_top_pool: i64 = @min(n_pool, @as(i64, hp.indexer_top_k / hp.indexer_kpool));
            var top_k: *Tensor = c.ggml_top_k(ctx, score, @intCast(n_top_pool)); // [n_top_pool, n_tokens], unordered

            // Keep the cells of the live picked pools and the tail, -inf
            // elsewhere. Every retained cell is in the token's sequence, so
            // the kq_mask is not needed.
            if (b.cparams.fused_top_k_mask) {
                const tail: ?*Tensor = if (hp.indexer_kpool_select_tail) inp_kpool.tail_idxs else null;
                impl.assert(kpool * n_top_pool + (if (tail) |t| t.ne[0] else 0) == @as(i64, inp_kpool.n_sel), src, 981, "kpool*n_top_pool + (tail ? tail->ne[0] : 0) == (int64_t) inp_kpool->n_sel");

                const keep: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_1d(ctx, type_mask, n_kv), 0.0);
                const sel: *Tensor = c.ggml_top_k_mask(ctx, keep, top_k, score, inp_kpool.pool_idxs, tail); // [n_kv, n_tokens]
                b.res.addFusedNode(.{ .op = .top_k_mask, .tensor = sel, .il = il });
                b.cb(sel, "indexer_sel", il);

                return sel;
            }

            // The selection mask marks the first min(nv, n_top_pool) slots
            // as the visible pools, so the set is ordered by descending score.
            const sel_score = c.ggml_get_rows(ctx, c.ggml_reshape_3d(ctx, score, 1, n_pool, n_tokens), top_k); // [1, n_top_pool, n_tokens]
            const sel_order = c.ggml_argsort(ctx, c.ggml_reshape_2d(ctx, sel_score, n_top_pool, n_tokens), c.GGML_SORT_ORDER_DESC);
            top_k = c.ggml_get_rows(ctx, c.ggml_reshape_3d(ctx, c.ggml_cast(ctx, top_k, c.GGML_TYPE_F32), 1, n_top_pool, n_tokens), sel_order);
            top_k = c.ggml_cast(ctx, c.ggml_cont(ctx, c.ggml_reshape_2d(ctx, top_k, n_top_pool, n_tokens)), c.GGML_TYPE_I32);
            b.cb(top_k, "indexer_top_k", il);

            sel_idx = c.ggml_get_rows(ctx, inp_kpool.pool_idxs, c.ggml_reshape_1d(ctx, top_k, n_top_pool * n_tokens)); // [kpool, n_top_pool*n_tokens]
            sel_idx = c.ggml_reshape_2d(ctx, sel_idx, kpool * n_top_pool, n_tokens);

            if (hp.indexer_kpool_select_tail) {
                // Append the incomplete tail, n_kv for missing cells.
                sel_idx = c.ggml_concat(ctx, sel_idx, inp_kpool.tail_idxs, 0);
            }
        }
        const n_sel = sel_idx.ne[0];
        impl.assert(n_sel == @as(i64, inp_kpool.n_sel), src, 1011, "n_sel == (int64_t) inp_kpool->n_sel");

        c.ggml_build_forward_expand(b.gf, sel_idx);

        var mask_all: *Tensor = c.ggml_new_tensor_4d(ctx, type_mask, n_kv + n_sel, 1, 1, 1);
        mask_all = c.ggml_fill(ctx, mask_all, -std.math.inf(f32));
        mask_all = c.ggml_repeat_4d(ctx, mask_all, n_kv + n_sel, n_tokens, 1, 1);
        mask_all = c.ggml_reshape_3d(ctx, mask_all, 1, n_kv + n_sel, n_tokens);

        var zeros: *Tensor = c.ggml_new_tensor_4d(ctx, type_mask, n_sel, 1, 1, 1);
        zeros = c.ggml_fill(ctx, zeros, 0.0);
        zeros = c.ggml_repeat_4d(ctx, zeros, n_sel, n_tokens, 1, 1);
        zeros = c.ggml_reshape_3d(ctx, zeros, 1, n_sel, n_tokens);

        // Live slots (visible pools, real tail cells) address disjoint
        // cells. Each dead slot writes its own dump row n_kv + slot, so a
        // token's scatter indices are unique: idx = dump + live*(idx - dump),
        // live = exp(mask).
        const sel_mask = inp_kpool.sel_mask.?;
        impl.assert(sel_mask.ne[0] == n_sel and sel_mask.ne[3] == n_tokens, src, 1027, "inp_kpool->sel_mask->ne[0] == n_sel && inp_kpool->sel_mask->ne[3] == n_tokens");
        const live = c.ggml_reshape_2d(ctx, c.ggml_exp(ctx, sel_mask), n_sel, n_tokens);
        const dump = c.ggml_arange(ctx, @floatFromInt(n_kv), @floatFromInt(n_kv + n_sel), 1.0);
        var idx_f: *Tensor = c.ggml_cast(ctx, sel_idx, c.GGML_TYPE_F32);
        idx_f = c.ggml_add(ctx, c.ggml_mul(ctx, c.ggml_sub(ctx, idx_f, dump), live), dump);
        sel_idx = c.ggml_cast(ctx, idx_f, c.GGML_TYPE_I32);

        var sel: *Tensor = c.ggml_set_rows(ctx, mask_all, zeros, c.ggml_reshape_3d(ctx, sel_idx, n_sel, n_tokens, 1));

        sel = c.ggml_view_2d(ctx, sel, n_kv, n_tokens, sel.nb[2], 0);
        sel = c.ggml_cont(ctx, sel);
        b.cb(sel, "indexer_sel", il);

        return sel;
    }

    /// Mirrors the graph's `dsa_out` (src/models/models.h:3136 @cd959bae9): the
    /// attention output and the
    /// selection to carry to the next shared layer.
    const DsaOut = struct {
        cur: *Tensor,
        sel: *Tensor,
    };

    /// Ports `llama_model_glm5_next::graph`'s `build_dsa_layer`
    /// (models/glm5-next.cpp:1046 @cd959bae9): nope-only MLA, attending
    /// sparsely over the indexer's selection.
    fn buildDsaLayer(self: *const Graph, cur: *Tensor, layer: *const Layer, mctx_hyb: *const HybridIdxContext, inp_attn: *graph.InpAttnK, inp_kpool: *InpKpool, prev_sel: ?*Tensor, il: c_int) DsaOut {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_tokens = b.n_tokens;
        const n_head: i64 = b.n_head;

        const mctx_mla = mctx_hyb.getAttn();

        const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
        const kv_lora_rank: i64 = hp.n_lora_kv;
        const n_embd_head_qk_nope = n_embd_head_k_mla - @as(i64, hp.nRot(0));
        const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head_k_mla));

        impl.assert(hp.nRot(0) == 0, src, 1058, "hparams.n_rot() == 0 && \"GLM5-Next MLA is nope-only\"");

        var qr: *Tensor = c.ggml_mul_mat(ctx, layer.wq_a, cur);
        qr = b.buildNorm(qr, layer.attn_q_a_norm, null, .rms, il);
        b.cb(qr, "q_resid", il);

        var q: *Tensor = c.ggml_mul_mat(ctx, layer.wq_b, qr);
        q = c.ggml_reshape_3d(ctx, q, n_embd_head_qk_nope, n_head, n_tokens);

        var kv_cmpr: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
        kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
        kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, n_tokens);
        b.cb(kv_cmpr, "kv_cmpr", il);

        // wk_b is absorbed, so the cache holds only the latent.
        var q_absorbed: *Tensor = c.ggml_permute(ctx, q, 0, 2, 1, 3);
        q_absorbed = c.ggml_mul_mat(ctx, layer.wk_b, q_absorbed);
        q_absorbed = c.ggml_permute(ctx, q_absorbed, 0, 2, 1, 3);
        b.cb(q_absorbed, "q_absorbed", il);

        const k: *Tensor = mctx_mla.getK(ctx, il);
        const v: *Tensor = c.ggml_view_4d(ctx, k, kv_lora_rank, k.ne[1], k.ne[2], k.ne[3], k.nb[1], k.nb[2], k.nb[3], 0);

        var sel_opt = prev_sel;
        if (il >= @as(c_int, @intCast(hp.nLayer())) or hp.isIndexerFull(@intCast(il))) { // the NextN block always has a full indexer
            var sel_full = self.buildKpoolSelect(cur, qr, layer, mctx_hyb, inp_kpool, il);

            // The selection replaces the kq_mask: split it into the streams
            // of the K view.
            const n_stream = k.ne[3];
            impl.assert(sel_full.ne[0] == k.ne[2] and @rem(sel_full.ne[1], n_stream) == 0, src, 1087, "sel->ne[0] == k->ne[2] && sel->ne[1] % n_stream == 0");
            sel_full = c.ggml_reshape_4d(ctx, sel_full, sel_full.ne[0], @divTrunc(sel_full.ne[1], n_stream), 1, n_stream);
            sel_opt = sel_full;
        } else {
            impl.assert(sel_opt != null, src, 1090, "sel != nullptr && \"shared indexer layer must follow a full indexer layer\"");
        }
        const sel = sel_opt.?;

        c.ggml_build_forward_expand(b.gf, q_absorbed);
        c.ggml_build_forward_expand(b.gf, kv_cmpr);
        c.ggml_build_forward_expand(b.gf, mctx_mla.cpyK(ctx, kv_cmpr, inp_attn.getKIdxs(), il));

        b.cb(sel, "kq_mask_dsa", il);

        var out = b.buildAttnMha(q_absorbed, k, v, null, sel, null, null, layer.wv_b, inp_kpool.n_sel, kq_scale, il);
        b.cb(out, "kqv_out", il);

        out = c.ggml_mul_mat(ctx, layer.wo, out);
        b.cb(out, "attn_out", il);

        return .{ .cur = out, .sel = sel };
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
