//! Mistral Large 4: absorbed MLA attention with a DSA lightning indexer
//! (full and shared indexer layers), NoPE layers whose cache rows keep the
//! RoPE layers' layout, layers that read another layer's KV cache, SSMax
//! query scaling, and a mixture of experts with a shared expert.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mistral-large4.cpp` at v0.9.0
//! (`cd959bae9`): `llama_model_mistral_large4`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_mistral_large4`'s `load_arch_hparams` (models/mistral-large4.cpp:6 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?; // eps for the indexer layer norms

    // MoE parameters.
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;

    // MLA parameters.
    if (try meta.getKey(u32, .attention_q_lora_rank, false, exc)) |v| hp.n_lora_q = v;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    hp.n_embd_head_k_mla_impl = (try meta.getKey(u32, .attention_key_length_mla, true, exc)).?;
    hp.n_embd_head_v_mla_impl = (try meta.getKey(u32, .attention_value_length_mla, true, exc)).?;

    // DSA parameters.
    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;

    @memset(&hp.is_indexer_full_impl, 1);
    _ = try meta.getKeyOrArr(u32, .attention_indexer_types, &hp.is_indexer_full_impl, hp.nLayer(), false, exc);
    if (!hp.isIndexerFull(0)) {
        return exc.throw(.runtime_error, "the first layer must have a full indexer", .{});
    }

    _ = try meta.getKeyOrArr(u32, .attention_rope_pattern, &hp.rope_pattern, hp.nLayer(), false, exc);
    _ = try meta.getKeyOrArr(u32, .attention_kv_source_map, &hp.kv_source_impl, hp.nLayer(), false, exc);
    var il: u32 = 0;
    while (il < hp.nLayer()) : (il += 1) {
        const il_src = hp.kvSource(il);
        if (il_src > il or !hp.hasKv(il_src) or hp.hasRope(il_src) != hp.hasRope(il)) {
            return exc.throw(.runtime_error, "invalid KV source layer {d} for layer {d}", .{ il_src, il });
        }
    }

    hp.f_final_logit_softcapping = 0.0;
    if (try meta.getKey(f32, .final_logit_softcapping, false, exc)) |v| hp.f_final_logit_softcapping = v;
    if (try meta.getKey(f32, .rope_scaling_yarn_log_mul, false, exc)) |v| hp.rope_yarn_log_mul = v;

    // SSMax uses log(pos + 1), computed through the attention temperature
    // input as 1 + log(pos + 1).
    hp.n_attn_temp_floor_scale = 1;
    hp.f_attn_temp_scale = 1.0;
    hp.f_attn_temp_offset = 0.0;

    self.type = .unknown;
}

/// Ports `llama_model_mistral_large4`'s `load_arch_tensors` (models/mistral-large4.cpp:58 @cd959bae9).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    _ = ml;
    const l = self.loadLocals();
    const hp = &self.hparams;

    if (!hp.isMla()) {
        return exc.throw(.runtime_error, "MISTRAL_LARGE4 architecture requires MLA", .{});
    }

    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_v_mla: i64 = hp.nEmbdHeadVMla();

    const kv_lora_rank: i64 = hp.n_lora_kv;

    const n_ff_exp: i64 = hp.nFfExp(0);
    const n_expert_shared: i64 = hp.n_expert_shared;

    const n_indexer_head: i64 = hp.indexer_n_head;
    const n_embd_indexer_head: i64 = hp.indexer_head_size;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // The output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const iu: u32 = @intCast(i);

        // NoPE layers use the full head for the non-rotary part.
        const n_rot_l: i64 = if (hp.hasRope(iu)) l.n_rot else 0;
        const n_embd_head_qk_nope = n_embd_head_k_mla - n_rot_l;

        // Learned RoPE frequencies (LeRoPE), shared by all layers.
        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, if (i != 0) loader.tensor_duplicated else 0, exc);

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_head * n_embd_head_k_mla }, 0, exc);

        if (hp.hasKv(iu)) {
            layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ l.n_embd, kv_lora_rank + n_rot_l }, 0, exc);
            layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, 0, exc);
        }

        layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_qk_nope, kv_lora_rank, l.n_head }, 0, exc);
        layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v_mla, l.n_head }, 0, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_head * n_embd_head_v_mla, l.n_embd }, 0, exc);

        layer.attn_ssmax_s = try self.createTensor(.{ .tensor = .attn_ssmax, .suffix = "weight", .bid = i }, &.{l.n_head}, 0, exc);
        layer.attn_ssmax_b = try self.createTensor(.{ .tensor = .attn_ssmax, .suffix = "bias", .bid = i }, &.{l.n_head}, 0, exc);

        // The DSA indexer.
        if (hp.isIndexerFull(iu)) {
            layer.indexer_q_norm = try self.createTensor(.{ .tensor = .indexer_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.indexer_q_norm_b = try self.createTensor(.{ .tensor = .indexer_q_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_indexer_head * n_embd_indexer_head }, 0, exc);
            layer.indexer_attn_k = try self.createTensor(.{ .tensor = .indexer_attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_indexer_head }, 0, exc);
            layer.indexer_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{n_embd_indexer_head}, 0, exc);
            layer.indexer_k_norm_b = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "bias", .bid = i }, &.{n_embd_indexer_head}, 0, exc);
            layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_indexer_head }, 0, exc);
            layer.indexer_ssmax_s = try self.createTensor(.{ .tensor = .indexer_ssmax, .suffix = "weight", .bid = i }, &.{n_indexer_head}, 0, exc);
            layer.indexer_ssmax_b = try self.createTensor(.{ .tensor = .indexer_ssmax, .suffix = "bias", .bid = i }, &.{n_indexer_head}, 0, exc);
        }

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        } else {
            if (l.n_expert == 0) {
                return exc.throw(.runtime_error, "n_expert must be > 0", .{});
            }
            if (l.n_expert_used == 0) {
                return exc.throw(.runtime_error, "n_expert_used must be > 0", .{});
            }

            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);

            // The MoE branch.
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

            // The shared expert branch.
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, 0, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_mistral_large4`'s `build_arch_graph` (models/mistral-large4.cpp:151 @cd959bae9).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    _ = params;
    buildGraph(model, b);
}

/// `build_ssmax`, a lambda of the `graph` constructor
/// (models/mistral-large4.cpp:155 @cd959bae9): the per-head SSMax scale
/// `s * log(pos + 1) + b`, `{1, n_head, n_tokens}`.
fn buildSsmax(b: *const graph.Builder, ssmax_pos: *Tensor, s: ?*Tensor, bias: ?*Tensor, n_head_l: i64) *Tensor {
    const ctx = b.ctx0;
    const pos = c.ggml_repeat_4d(ctx, ssmax_pos, 1, n_head_l, b.n_tokens, 1);
    const res = c.ggml_mul(ctx, pos, c.ggml_reshape_3d(ctx, s, 1, n_head_l, 1));
    return c.ggml_add(ctx, res, c.ggml_reshape_3d(ctx, bias, 1, n_head_l, 1));
}

/// `pad_rope`, a lambda of the `graph` constructor
/// (models/mistral-large4.cpp:155 @cd959bae9): NoPE layers keep the cache
/// layout of the RoPE layers by zero-filling the RoPE part.
fn padRope(b: *const graph.Builder, x: *Tensor) *Tensor {
    const ctx = b.ctx0;
    var zeros = c.ggml_new_tensor_3d(ctx, x.type, b.n_rot, x.ne[1], x.ne[2]);
    zeros = c.ggml_fill(ctx, zeros, 0.0);
    return c.ggml_concat(ctx, x, zeros, 0);
}

/// Ports the constructor of `llama_model_mistral_large4`'s `graph` (models/mistral-large4.cpp:155 @cd959bae9).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const cp = b.cparams;
    impl.assert(hp.isMla(), "models/mistral-large4.cpp", 157, "hparams.is_mla()");

    const n_embd_head_k: i64 = hp.nEmbdHeadKMla();

    const n_indexer_head: i64 = hp.indexer_n_head;
    const n_embd_indexer_head: i64 = hp.indexer_head_size;
    const n_indexer_top_k: u32 = hp.indexer_top_k;

    // The indexer head layout is [rope | nope].
    impl.assert(b.n_rot <= n_embd_indexer_head, "models/mistral-large4.cpp", 166, "n_rot <= n_embd_indexer_head");

    const kv_lora_rank: i64 = hp.n_lora_kv;
    const n_head = b.n_head;
    const n_tokens = b.n_tokens;
    const n_rot = b.n_rot;

    // YaRN is used without magnitude scaling (apply_scale = false), so
    // there is no mscale in kq_scale.
    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head_k));

    // {n_embd, n_tokens}
    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    // The SSMax position basis: log(pos + 1), {1, 1, n_tokens}.
    const ssmax_pos = c.ggml_scale_bias(ctx, b.buildInpAttnScale(), 1.0, -1.0);
    b.cb(ssmax_pos, "ssmax_pos", -1);

    const inp_attn_dsa = b.buildAttnInpKDsa();

    const inp_out_ids = b.buildInpOutIds();

    // The shared indexer layers reuse the top_k of the previous full indexer layer.
    var prev_top_k: ?*Tensor = null;

    // DSpark reads the last hidden state of every token: the output rows
    // are selected after the last layer.
    const n_layer_u: usize = @intCast(b.n_layer);
    const extract_last = n_layer_u < cp.embeddings_layer_inp.len and cp.embeddings_layer_inp[n_layer_u];

    var cur: *Tensor = undefined;
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const ilu: u32 = @intCast(il);

        b.res.t_layer_inp[@intCast(il)] = inpL;

        const use_rope = hp.hasRope(ilu);

        const n_rot_l: i64 = if (use_rope) n_rot else 0;
        const n_embd_head_qk_nope = n_embd_head_k - n_rot_l;

        const rope_factors = model.getRopeFactors(cp, ilu);

        var inpSA = inpL;

        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            var top_k: *Tensor = undefined;

            // The lightning indexer.
            if (hp.isIndexerFull(ilu)) {
                var indexer_q = b.buildNorm(cur, layer.indexer_q_norm, layer.indexer_q_norm_b, .norm, il);
                b.cb(indexer_q, "indexer_q_norm", il);

                indexer_q = c.ggml_mul_mat(ctx, layer.indexer_attn_q_b, indexer_q);
                b.cb(indexer_q, "indexer_q", il);

                // {n_embd_indexer_head, n_indexer_head, n_tokens}
                indexer_q = c.ggml_reshape_3d(ctx, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);
                indexer_q = c.ggml_mul(ctx, indexer_q, buildSsmax(b, ssmax_pos, layer.indexer_ssmax_s, layer.indexer_ssmax_b, n_indexer_head));
                b.cb(indexer_q, "indexer_q_ssmax", il);

                var indexer_k: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_attn_k, cur);
                b.cb(indexer_k, "indexer_k", il);

                indexer_k = b.buildNorm(indexer_k, layer.indexer_k_norm, layer.indexer_k_norm_b, .norm, il);
                b.cb(indexer_k, "indexer_k", il);

                // {n_embd_indexer_head, 1, n_tokens}
                indexer_k = c.ggml_reshape_3d(ctx, indexer_k, n_embd_indexer_head, 1, n_tokens);

                if (use_rope) {
                    indexer_q = c.ggml_rope_ext(ctx, indexer_q, inp_pos, rope_factors, @intCast(n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                    b.cb(indexer_q, "indexer_q", il);

                    indexer_k = c.ggml_rope_ext(ctx, indexer_k, inp_pos, rope_factors, @intCast(n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                    b.cb(indexer_k, "indexer_k", il);
                }

                // The Hadamard transform on the indexer q and k.
                indexer_q = c.ggml_mul_mat(ctx, inp_attn_dsa.self_k_rot_lid[ilu], indexer_q);
                b.cb(indexer_q, "indexer_q", il);
                indexer_k = c.ggml_mul_mat(ctx, inp_attn_dsa.self_k_rot_lid[ilu], indexer_k);
                b.cb(indexer_k, "indexer_k", il);

                // Store the indexer keys to the KV cache.
                const mctx_lid = inp_attn_dsa.mctx.getLid();
                const k_idxs_lid = inp_attn_dsa.getKIdxsLid();
                c.ggml_build_forward_expand(b.gf, mctx_lid.cpyK(ctx, indexer_k, k_idxs_lid, il));

                var indexer_weights: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_proj, cur);
                b.cb(indexer_weights, "indexer_weights", il);

                // The cached indexer keys.
                indexer_k = mctx_lid.getK(ctx, il);

                // Split the batch into streams if needed.
                const n_stream = indexer_k.ne[3];
                const ns: usize = @intCast(n_stream);
                indexer_q = c.ggml_view_4d(ctx, indexer_q, indexer_q.ne[0], indexer_q.ne[1], @divTrunc(indexer_q.ne[2], n_stream), n_stream, indexer_q.nb[1], indexer_q.nb[2], indexer_q.nb[3] / ns, 0);
                indexer_weights = c.ggml_view_4d(ctx, indexer_weights, indexer_weights.ne[0], @divTrunc(indexer_weights.ne[1], n_stream), indexer_weights.ne[2], n_stream, indexer_weights.nb[1], indexer_weights.nb[2] / ns, indexer_weights.nb[3] / ns, 0);

                // The weights pre-scaled, to avoid scaling the huge indexer_score tensor.
                indexer_weights = c.ggml_scale(ctx, indexer_weights, 1.0 / impl.sqrtf(@floatFromInt(n_embd_indexer_head * n_indexer_head)));
                b.cb(indexer_weights, "indexer_weights", il);

                var indexer_score: *Tensor = undefined;
                if (cp.fused_lid) {
                    indexer_score = c.ggml_lightning_indexer(ctx, indexer_q, indexer_k, indexer_weights, inp_attn_dsa.self_attn_mask_lid.raw);
                    b.cb(indexer_score, "indexer_score", il);
                    b.res.addFusedNode(.{ .op = .lightning_indexer, .tensor = indexer_score, .il = il });
                } else {
                    // The indexer kq.
                    indexer_q = c.ggml_permute(ctx, indexer_q, 0, 2, 1, 3);
                    b.cb(indexer_q, "indexer_q", il);
                    indexer_k = c.ggml_permute(ctx, indexer_k, 0, 2, 1, 3);
                    b.cb(indexer_k, "indexer_k", il);

                    var indexer_kq: *Tensor = c.ggml_mul_mat(ctx, indexer_k, indexer_q);
                    b.cb(indexer_kq, "indexer_kq", il);

                    // ReLU requires contiguous tensors.
                    indexer_kq = c.ggml_cont(ctx, c.ggml_permute(ctx, indexer_kq, 2, 1, 0, 3));
                    b.cb(indexer_kq, "indexer_kq", il);

                    indexer_score = c.ggml_relu(ctx, indexer_kq);
                    b.cb(indexer_score, "indexer_score", il);

                    // The scores times the indexer weights.
                    indexer_score = c.ggml_mul(ctx, indexer_score, indexer_weights);
                    b.cb(indexer_score, "indexer_score", il);

                    // Summed over the q's n_indexer_head dimension.
                    indexer_score = c.ggml_sum_rows(ctx, indexer_score);
                    b.cb(indexer_score, "indexer_score", il);

                    // Permuted to match the KQ mask.
                    indexer_score = c.ggml_cont(ctx, c.ggml_permute(ctx, indexer_score, 2, 1, 0, 3));
                    b.cb(indexer_score, "indexer_score", il);

                    // The masked indexer scores.
                    const indexer_kq_mask = inp_attn_dsa.self_attn_mask_lid.raw;
                    indexer_score = c.ggml_add(ctx, indexer_score, indexer_kq_mask);
                    b.cb(indexer_score, "indexer_score", il);
                }

                // The indices of the top k indexer scores.
                const n_top_k: u32 = if (indexer_score.ne[0] < n_indexer_top_k) @intCast(indexer_score.ne[0]) else n_indexer_top_k;
                top_k = c.ggml_cont(ctx, c.ggml_top_k(ctx, indexer_score, @intCast(n_top_k)));
                prev_top_k = top_k;
                b.cb(top_k, "top_k", il);
            } else {
                impl.assert(prev_top_k != null, "models/mistral-large4.cpp", 337, "prev_top_k != nullptr && \"shared indexer layer must follow a previous full indexer layer\"");
                top_k = prev_top_k.?;
                b.cb(top_k, "top_k", il);
            }

            const q: *Tensor = c.ggml_mul_mat(ctx, layer.wq, cur);
            b.cb(q, "q", il);

            // {n_embd_head_qk_nope, n_head, n_tokens}
            var q_nope: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_nope, n_head, n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * @as(usize, @intCast(n_head)), 0);
            b.cb(q_nope, "q_nope", il);

            // {n_embd_head_qk_nope, n_tokens, n_head}
            q_nope = c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
            b.cb(q_nope, "q_nope_perm", il);

            // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
            var q_nope_absorbed: *Tensor = c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
            b.cb(q_nope_absorbed, "q_nope_absorbed", il);

            // {kv_lora_rank, n_head, n_tokens}
            q_nope_absorbed = c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);
            b.cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

            // {kv_lora_rank + n_rot, n_head, n_tokens}. RoPE must go first,
            // for in-place context shifting in build_rope_shift().
            var Qcur: *Tensor = undefined;
            if (use_rope) {
                var q_pe: *Tensor = c.ggml_view_3d(ctx, q, n_rot, n_head, n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * @as(usize, @intCast(n_head)), c.ggml_row_size(q.type, n_embd_head_qk_nope));
                q_pe = c.ggml_rope_ext(ctx, q_pe, inp_pos, rope_factors, @intCast(n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                b.cb(q_pe, "q_pe", il);

                Qcur = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
            } else {
                Qcur = padRope(b, q_nope_absorbed);
            }
            b.cb(Qcur, "Qcur", il);

            Qcur = c.ggml_mul(ctx, Qcur, buildSsmax(b, ssmax_pos, layer.attn_ssmax_s, layer.attn_ssmax_b, n_head));
            b.cb(Qcur, "Qcur_ssmax", il);

            // Layers without their own KV read the cache their source layer wrote.
            var Kcur: ?*Tensor = null;
            var Vcur: ?*Tensor = null;

            if (hp.hasKv(ilu)) {
                const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
                b.cb(kv_cmpr_pe, "kv_cmpr_pe", il);

                // {kv_lora_rank, n_tokens}
                var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_rot_l), 0);
                b.cb(kv_cmpr, "kv_cmpr", il);

                kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
                b.cb(kv_cmpr, "kv_cmpr", il);

                kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, n_tokens);
                b.cb(kv_cmpr, "kv_cmpr_reshape", il);

                if (use_rope) {
                    // {n_rot, 1, n_tokens}
                    var k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, n_rot, 1, n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_rot), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_rot), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));
                    k_pe = c.ggml_rope_ext(ctx, k_pe, inp_pos, rope_factors, @intCast(n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                    b.cb(k_pe, "k_pe", il);

                    Kcur = c.ggml_concat(ctx, kv_cmpr, k_pe, 0);
                } else {
                    Kcur = padRope(b, kv_cmpr);
                }
                b.cb(Kcur.?, "Kcur", il);

                Vcur = kv_cmpr;
                b.cb(Vcur.?, "Vcur", il);
            }

            // MLA with the absorption optimization becomes MQA (GQA with one group).
            cur = b.buildAttnDsa(inp_attn_dsa, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, layer.wv_b, top_k, kq_scale, il);
        }

        if (il == b.n_layer - 1 and !extract_last) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (ilu < hp.n_layer_dense_lead) {
            cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            const moe_out = b.buildMoeFfn(cur, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = layer.ffn_gate_exps,
                .down_exps = layer.ffn_down_exps.?,
                .n_expert = b.n_expert,
                .n_expert_used = b.n_expert_used,
                .type_op = .silu,
                .norm_w = hp.expert_weights_norm,
                .w_scale = hp.expert_weights_scale,
                .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
                .gate_up_exps = layer.ffn_gate_up_exps,
                .up_exps_s = layer.ffn_up_exps_s,
                .gate_exps_s = layer.ffn_gate_exps_s,
                .down_exps_s = layer.ffn_down_exps_s,
            }, il);
            b.cb(moe_out, "ffn_moe_out", il);

            const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
            b.cb(ffn_shexp, "ffn_shexp", il);

            cur = c.ggml_add(ctx, moe_out, ffn_shexp);
            b.cb(cur, "ffn_out", il);
        }

        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    if (extract_last) {
        b.res.t_layer_inp[n_layer_u] = inpL;
        inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
    }

    cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = c.ggml_mul_mat(ctx, model.output, cur);

    if (hp.f_final_logit_softcapping != 0.0) {
        cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
        cur = c.ggml_tanh(ctx, cur);
        cur = c.ggml_scale(ctx, cur, hp.f_final_logit_softcapping);
    }

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
