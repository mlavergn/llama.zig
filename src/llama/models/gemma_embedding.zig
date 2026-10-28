//! EmbeddingGemma: Gemma 3's blocks as a bidirectional encoder -- symmetric
//! sliding-window attention on most layers, RMS norms before and after the
//! attention and the feed-forward, and RMS-normed queries and keys --
//! producing embeddings, optionally through two dense projections.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/gemma-embedding.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_gemma_embedding`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_gemma_embedding`'s `load_arch_hparams` (models/gemma-embedding.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.swa_type = @backingInt(hparams_mod.SwaType.symmetric);
    var swa_period: u32 = 6;
    if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |v| swa_period = v;
    hp.setSwaPattern(swa_period, false);

    hp.causal_attn = false; // Embeddings do not use causal attention.

    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    // Applied only if the model was converted with
    // --sentence-transformers-dense-modules.
    if (try meta.getKey(u32, .dense_2_feat_in, false, exc)) |v| hp.dense_2_feat_in = v;
    if (try meta.getKey(u32, .dense_2_feat_out, false, exc)) |v| hp.dense_2_feat_out = v;
    if (try meta.getKey(u32, .dense_3_feat_in, false, exc)) |v| hp.dense_3_feat_in = v;
    if (try meta.getKey(u32, .dense_3_feat_out, false, exc)) |v| hp.dense_3_feat_out = v;

    impl.assert(hp.dense_2_feat_in == 0 or hp.dense_2_feat_in == hp.n_embd, "models/gemma-embedding.cpp", 21, "(hparams.dense_2_feat_in == 0 || hparams.dense_2_feat_in == hparams.n_embd) && \"dense_2_feat_in must be equal to n_embd\"");
    impl.assert(hp.dense_3_feat_out == 0 or hp.dense_3_feat_out == hp.n_embd, "models/gemma-embedding.cpp", 22, "(hparams.dense_3_feat_out == 0 || hparams.dense_3_feat_out == hparams.n_embd) && \"dense_3_feat_out must be equal to n_embd\"");

    self.type = switch (hp.nLayer()) {
        24 => .@"0_3b",
        else => .unknown,
    };
    hp.f_attention_scale = 1.0 / impl.sqrtf(@floatFromInt(hp.nEmbdHeadK(0)));
}

/// Ports `llama_model_gemma_embedding`'s `load_arch_tensors` (models/gemma-embedding.cpp:32 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    // The dense linear weights.
    self.dense_2_out_layers = try self.createTensor(.{ .tensor = .dense_2_out, .suffix = "weight" }, &.{ l.n_embd, hp.dense_2_feat_out }, nr, exc);
    self.dense_3_out_layers = try self.createTensor(.{ .tensor = .dense_3_out, .suffix = "weight" }, &.{ hp.dense_3_feat_in, l.n_embd }, nr, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
    }
}

/// Ports `llama_model_gemma_embedding`'s `build_arch_graph` (models/gemma-embedding.cpp:71 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_gemma_embedding`'s `graph`
/// (models/gemma-embedding.cpp:75 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadK(0);

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

    // Raw embeddings given as input (e.g. encoded images) are not scaled.
    inpL = c.ggml_scale(ctx, inpL, if (b.ubatch.token != null) impl.sqrtf(@floatFromInt(b.n_embd)) else 1.0);
    b.cb(inpL, "inp_scaled", -1);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpNoCache();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
        const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Kcur, "Kcur_normed", il);

            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
            Qcur = c.ggml_scale(ctx, Qcur, hp.f_attention_scale);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0, il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }

        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        const sa_out: *Tensor = c.ggml_add(ctx, cur, inpL);
        b.cb(sa_out, "sa_out", il);

        cur = b.buildNorm(sa_out, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // The feed-forward network.
        {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .gelu, .par, il);
            b.cb(cur, "ffn_out", il);
        }

        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, -1);
        b.cb(cur, "ffn_post_norm", -1);

        cur = c.ggml_add(ctx, cur, sa_out);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
