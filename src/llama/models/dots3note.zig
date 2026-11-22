//! dots.llm3 (dots3note): absorbed multi-head latent attention over K-only
//! caches, interleaving full-attention layers with DeepSeek sparse
//! attention and sliding-window layers of their own MLA geometry; a norm on
//! the shared RoPE key; a head-wise sigmoid gate on the attention output
//! before `wo`; leading dense blocks, then a mixture of experts with shared
//! experts.
//!
//! The lightning indexer is DeepSeek V3.2's (`deepseek32.zig`), run on the
//! full-attention layers alone. The NextN/MTP tensors load but nothing
//! builds a graph from them yet.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/dots3note.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_dots3note`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`. The C++
//! notes it was adapted from `deepseek32.cpp` (the DSA indexer and absorbed
//! MLA) and `step35.cpp` (the head-wise output gate).

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_dots3note`'s `load_arch_hparams` (src/models/dots3note.cpp:8 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    // The eps of the indexer's k_norm layer norm.
    hp.f_norm_eps = 1e-6;

    // MoE.
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_layer_dense_lead = (try meta.getKey(u32, .leading_dense_block_count, true, exc)).?;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;

    // The MLA of the full-attention layers.
    hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    hp.n_embd_head_k_mla_impl = (try meta.getKey(u32, .attention_key_length_mla, true, exc)).?;
    hp.n_embd_head_v_mla_impl = (try meta.getKey(u32, .attention_value_length_mla, true, exc)).?;

    // The MLA of the sliding-window layers.
    hp.n_lora_kv_swa = (try meta.getKey(u32, .attention_kv_lora_rank_swa, true, exc)).?;
    hp.n_embd_head_k_mla_swa = (try meta.getKey(u32, .attention_key_length_mla_swa, true, exc)).?;
    hp.n_embd_head_v_mla_swa = (try meta.getKey(u32, .attention_value_length_mla_swa, true, exc)).?;

    hp.swa_type = @backingInt(llama.hparams.SwaType.standard);
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.rope_freq_base_train_swa = (try meta.getKey(f32, .rope_freq_base_swa, true, exc)).?;
    _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);

    // DSA.
    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;
    _ = try meta.getArr(u32, .attention_indexer_types, &hp.is_indexer_full_impl, true, exc);

    self.type = switch (hp.nLayer()) {
        46 => .@"288b_a19b",
        else => .unknown,
    };
}

/// Ports `llama_model_dots3note`'s `load_arch_tensors` (src/models/dots3note.cpp:48 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, ml: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    _ = ml;
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;

    if (!hp.isMla()) {
        return exc.throw(.runtime_error, "DOTS3NOTE architecture requires MLA", .{});
    }

    const n_embd_head_qk_rope: i64 = hp.nRot(0);

    const q_lora_rank: i64 = hp.n_lora_q;
    const n_ff_exp: i64 = hp.nFfExp(0);
    const n_expert_shared: i64 = hp.n_expert_shared;

    const indexer_head_size: i64 = hp.indexer_head_size;
    const indexer_n_head: i64 = hp.indexer_n_head;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const iu: u32 = @intCast(i);

        const is_mtp = i >= l.n_layer;
        // The NextN/MTP block has the sliding-window geometry.
        const is_swa = is_mtp or hp.isSwa(iu);

        // The MTP tensors stay in the GGUF, but no MTP graph uses them yet.
        const flags: u32 = if (is_mtp) llama.loader.tensor_skip | nr else 0;

        const n_head_l: i64 = hp.nHead(iu);

        const kv_lora_rank: i64 = if (is_swa) hp.n_lora_kv_swa else hp.n_lora_kv;
        const n_embd_head_k_mla: i64 = if (is_swa) hp.n_embd_head_k_mla_swa else hp.nEmbdHeadKMla();
        const n_embd_head_v_mla: i64 = if (is_swa) hp.n_embd_head_v_mla_swa else hp.nEmbdHeadVMla();
        const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, flags, exc);
        layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, flags, exc);
        // The norm on the shared RoPE key, before RoPE.
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{n_embd_head_qk_rope}, flags, exc);

        layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_lora_rank }, flags, exc);
        layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, n_head_l * n_embd_head_k_mla }, flags, exc);

        layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ l.n_embd, kv_lora_rank + n_embd_head_qk_rope }, flags, exc);

        layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_qk_nope, kv_lora_rank, n_head_l }, flags, exc);
        layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v_mla, n_head_l }, flags, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_head_l * n_embd_head_v_mla, l.n_embd }, flags, exc);

        // The head-wise sigmoid output gate.
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_head_l }, flags, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        // The DSA indexer.
        if (!is_mtp and hp.isIndexerFull(iu)) {
            layer.indexer_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{indexer_head_size}, flags, exc);
            layer.indexer_k_norm_b = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "bias", .bid = i }, &.{indexer_head_size}, flags, exc);
            layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, indexer_n_head }, flags, exc);
            layer.indexer_attn_k = try self.createTensor(.{ .tensor = .indexer_attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, indexer_head_size }, flags, exc);
            layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, indexer_n_head * indexer_head_size }, flags, exc);
        }

        if (is_mtp or i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, flags, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
        } else {
            if (l.n_expert == 0 or l.n_expert_used == 0) {
                return exc.throw(.runtime_error, "n_expert and n_expert_used must be > 0", .{});
            }

            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, flags, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, flags, exc);

            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);

            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, flags, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, flags, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, flags, exc);
        }

        if (is_mtp) {
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, flags, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_dots3note`'s `get_rope_params` (src/models/dots3note.cpp:148 @b1b770d90).
///
/// Parameters:
/// - `model`: the model.
/// - `cparams`: the context's parameters.
/// - `il`: the layer.
///
/// Return: the RoPE of layer `il`, for the graph and the K-shift alike.
pub fn getRopeParams(model: *const llama.model.Model, cparams: *const llama.cparams.Cparams, il: u32) llama.model.RopeParams {
    var res = model.getRopeParamsBase(cparams, il);

    res.freq_scale = cparams.rope_freq_scale;

    return res;
}

/// Ports `llama_model_dots3note`'s `build_arch_graph` (src/models/dots3note.cpp:156 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, params: *const llama.graph.Params) void {
    _ = params;
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_dots3note`'s `graph`
/// (src/models/dots3note.cpp:160 @b1b770d90).
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    llama.impl.assert(hp.isMla(), "models/dots3note.cpp", 162, "hparams.is_mla()");

    const n_embd_head_qk_rope: i64 = hp.nRot(0);

    const n_indexer_head: i64 = hp.indexer_n_head;
    const n_embd_indexer_head: i64 = hp.indexer_head_size;
    const n_indexer_top_k: u32 = hp.indexer_top_k;

    // The indexer head's layout is `[rope | nope]`.
    llama.impl.assert(hp.nRot(0) <= n_embd_indexer_head, "models/dots3note.cpp", 171, "hparams.n_rot() <= n_embd_indexer_head");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKDsaIswa();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const ilu: u32 = @intCast(il);
        var inpSA = inpL;

        const is_swa = hp.isSwa(ilu);

        const n_head_l: i64 = hp.nHead(ilu);

        const kv_lora_rank: i64 = if (is_swa) hp.n_lora_kv_swa else hp.n_lora_kv;
        const n_embd_head_k_mla: i64 = if (is_swa) hp.n_embd_head_k_mla_swa else hp.nEmbdHeadKMla();
        const n_embd_head_v_mla: i64 = if (is_swa) hp.n_embd_head_v_mla_swa else hp.nEmbdHeadVMla();
        const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

        const kq_scale: f32 = 1.0 / stdlib.libm.sqrtfInt(n_embd_head_k_mla);

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const attn_inp = cur;

            var qr: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_mul_mat(ctx, layer.wq_a, cur);
            b.cb(qr, "qr", il);

            qr = b.buildNorm(qr, layer.attn_q_a_norm, null, .rms, il);
            b.cb(qr, "qr", il);

            var top_k: ?*llama.impl.llama_c.ggml_tensor = null;

            // The lightning indexer, on the full-attention layers alone.
            if (!is_swa) {
                const t = llama.models.deepseek32.lightningIndexer(b, layer, inp_attn.getDsa(), cur, qr, inp_pos, n_indexer_head, n_embd_indexer_head, n_indexer_top_k, il);
                b.cb(t, "top_k", il);
                top_k = t;
            }

            const q: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_mul_mat(ctx, layer.wq_b, qr);
            b.cb(q, "q", il);

            // Split into `{n_embd_head_qk_nope, n_head_l, n_tokens}`.
            var q_nope: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_view_3d(ctx, q, n_embd_head_qk_nope, n_head_l, b.n_tokens, llama.impl.llama_c.ggml_row_size(q.type, n_embd_head_k_mla), llama.impl.llama_c.ggml_row_size(q.type, n_embd_head_k_mla) * @as(usize, @intCast(n_head_l)), 0);
            b.cb(q_nope, "q_nope", il);

            // And `{n_embd_head_qk_rope, n_head_l, n_tokens}`.
            var q_pe: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_view_3d(ctx, q, n_embd_head_qk_rope, n_head_l, b.n_tokens, llama.impl.llama_c.ggml_row_size(q.type, n_embd_head_k_mla), llama.impl.llama_c.ggml_row_size(q.type, n_embd_head_k_mla) * @as(usize, @intCast(n_head_l)), llama.impl.llama_c.ggml_row_size(q.type, n_embd_head_qk_nope));
            b.cb(q_pe, "q_pe", il);

            const kv_cmpr_pe: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
            b.cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // Split into `{kv_lora_rank, n_tokens}`.
            var kv_cmpr: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, b.n_tokens, llama.impl.llama_c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), 0);
            b.cb(kv_cmpr, "kv_cmpr", il);

            // And `{n_embd_head_qk_rope, 1, n_tokens}`.
            var k_pe: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_view_3d(ctx, kv_cmpr_pe, n_embd_head_qk_rope, 1, b.n_tokens, llama.impl.llama_c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), llama.impl.llama_c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), llama.impl.llama_c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));
            b.cb(k_pe, "k_pe", il);

            // The norm on the shared RoPE key, before RoPE.
            k_pe = b.buildNorm(k_pe, layer.attn_k_norm, null, .rms, il);
            b.cb(k_pe, "k_pe", il);

            q_pe = b.buildRope(q_pe, inp_pos, il);
            b.cb(q_pe, "q_pe", il);

            k_pe = b.buildRope(k_pe, inp_pos, il);
            b.cb(k_pe, "k_pe", il);

            kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
            b.cb(kv_cmpr, "kv_cmpr", il);

            // MLA attention with the absorption optimization.
            {
                // `{n_embd_head_qk_nope, n_tokens, n_head_l}`.
                q_nope = llama.impl.llama_c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
                b.cb(q_nope, "q_nope_perm", il);

                // `{n_embd_head_qk_nope, kv_lora_rank, n_head_l}` x
                // `{n_embd_head_qk_nope, n_tokens, n_head_l}`.
                var q_nope_absorbed: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
                b.cb(q_nope_absorbed, "q_nope_absorbed", il);

                // `{kv_lora_rank, n_head_l, n_tokens}`.
                q_nope_absorbed = llama.impl.llama_c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);
                b.cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // `{n_embd_head_qk_rope + kv_lora_rank, n_head_l, n_tokens}`.
                const Qcur: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
                b.cb(Qcur, "Qcur", il);

                kv_cmpr = llama.impl.llama_c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, b.n_tokens);
                b.cb(kv_cmpr, "kv_cmpr_reshape", il);

                // `{n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}`.
                const Kcur: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_concat(ctx, kv_cmpr, k_pe, 0);
                b.cb(Kcur, "Kcur", il);

                // `{kv_lora_rank, 1, n_tokens}`.
                const Vcur = kv_cmpr;
                b.cb(Vcur, "Vcur", il);

                // The head-wise output gate comes before `wo`, so `wo` stays
                // out of `build_attn`.
                if (is_swa) {
                    cur = b.buildAttn(inp_attn.getSwa(), null, null, null, Qcur, Kcur, Vcur, null, null, layer.wv_b, kq_scale, il);
                } else {
                    cur = b.buildAttnDsa(inp_attn.getDsa(), null, null, null, Qcur, Kcur, Vcur, null, null, layer.wv_b, top_k.?, kq_scale, il);
                }
                b.cb(cur, "attn_out", il);

                var gate = b.buildLoraMm(layer.wqkv_gate.?, attn_inp, null);
                b.cb(gate, "attn_gate", il);

                gate = llama.impl.llama_c.ggml_sigmoid(ctx, gate);
                b.cb(gate, "attn_gate_sigmoid", il);

                // The per-head gate, broadcast over the head dimension.
                var attn_3d: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_reshape_3d(ctx, cur, n_embd_head_v_mla, n_head_l, b.n_tokens);
                const gate_3d: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_reshape_3d(ctx, gate, 1, n_head_l, b.n_tokens);
                attn_3d = llama.impl.llama_c.ggml_mul(ctx, attn_3d, gate_3d);
                b.cb(attn_3d, "attn_gated", il);

                cur = llama.impl.llama_c.ggml_reshape_2d(ctx, attn_3d, n_embd_head_v_mla * n_head_l, b.n_tokens);

                cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
                b.cb(cur, "attn_output", il);
            }
        }

        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.llama_c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (ilu < hp.n_layer_dense_lead) {
            cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            const moe_out = llama.models.deepseek32.buildMoe(b, layer, cur, il);
            b.cb(moe_out, "ffn_moe_out", il);

            const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
            b.cb(ffn_shexp, "ffn_shexp", il);

            cur = llama.impl.llama_c.ggml_add(ctx, moe_out, ffn_shexp);
            b.cb(cur, "ffn_out", il);
        }

        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = llama.impl.llama_c.ggml_mul_mat(ctx, model.output, cur);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
