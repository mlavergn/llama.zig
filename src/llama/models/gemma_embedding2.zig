//! EmbeddingGemma 2: a bidirectional Gemma 4-style encoder producing
//! embeddings -- symmetric sliding-window attention on most layers, per-layer
//! head sizes, RMS-normed Q, K and V, RMS norms around the attention and the
//! feed-forward, per-layer inputs projected from the hidden state alone
//! (Gemma 3n's per-layer embeddings without the per-layer token table), a
//! per-layer output scale, and a final projection to the embedding size.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/gemma-embedding2.cpp` at v0.7.0
//! (`620d2225c`): `llama_model_gemma_embedding2`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph` with
//! `build_inp_per_layer`.

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

/// Ports `llama_model_gemma_embedding2`'s `load_arch_hparams`
/// (models/gemma-embedding2.cpp:3 @620d2225c).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.swa_type = @backingInt(hparams_mod.SwaType.symmetric);
    try self.loadSwaPattern(ml, 6, false, exc);

    hp.causal_attn = false; // Embeddings do not use causal attention.
    hp.f_attention_scale = 1.0; // As in Gemma 4: q_norm makes the scaling unnecessary.

    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_embd_per_layer = (try meta.getKey(u32, .embedding_length_per_layer, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        24 => .@"0_3b",
        else => .unknown,
    };
}

/// Ports `llama_model_gemma_embedding2`'s `load_arch_tensors`
/// (models/gemma-embedding2.cpp:21 @620d2225c).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const n_embd_per_layer: i64 = hp.n_embd_per_layer;
    const n_embd_out: i64 = hp.nEmbdOut();

    if (l.n_embd_head_k != l.n_embd_head_v) {
        return exc.throw(.runtime_error, "EmbeddingGemma2 requires n_embd_head_k == n_embd_head_v", .{});
    }
    if (hp.n_embd_head_k_swa != hp.n_embd_head_v_swa) {
        return exc.throw(.runtime_error, "EmbeddingGemma2 requires n_embd_head_k_swa == n_embd_head_v_swa", .{});
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.per_layer_model_proj = try self.createTensor(.{ .tensor = .per_layer_model_proj, .suffix = "weight", .bid = 0 }, &.{ l.n_embd, n_embd_per_layer * l.n_layer }, 0, exc);
    self.per_layer_proj_norm = try self.createTensor(.{ .tensor = .per_layer_proj_norm, .suffix = "weight", .bid = 0 }, &.{n_embd_per_layer}, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    // Projects the final hidden state to the embedding size.
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, n_embd_out }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const iu: u32 = @intCast(i);
        const n_head: i64 = hp.nHead(iu);
        const n_embd_head: i64 = hp.nEmbdHeadK(iu);
        const n_embd_k: i64 = hp.nEmbdKGqa(iu);
        const n_embd_v: i64 = hp.nEmbdVGqa(iu);

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_head * n_head }, 0, exc);
        layer.wk = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_k }, 0, exc);
        layer.wv = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_v }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_head * n_head, l.n_embd }, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.per_layer_inp_gate = try self.createTensor(.{ .tensor = .per_layer_inp_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_per_layer }, 0, exc);
        layer.per_layer_proj = try self.createTensor(.{ .tensor = .per_layer_proj, .suffix = "weight", .bid = i }, &.{ n_embd_per_layer, l.n_embd }, 0, exc);
        layer.per_layer_post_norm = try self.createTensor(.{ .tensor = .per_layer_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.out_scale = try self.createTensor(.{ .tensor = .layer_out_scale, .suffix = "weight", .bid = i }, &.{1}, 0, exc);
    }
}

/// Ports `llama_model_gemma_embedding2`'s `build_arch_graph`
/// (models/gemma-embedding2.cpp:75 @620d2225c).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_gemma_embedding2`'s `graph`
/// (models/gemma-embedding2.cpp:79 @620d2225c). `t_embd` is not expanded
/// here: `build_pooling` expands what it reads.
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_tokens = b.n_tokens;

    // Important: do not normalize weights for raw embeddings input (i.e.
    // encoded image embeddings): `build_inp_embd` scales the token rows only.
    var inpL: *Tensor = b.buildInpEmbdScaled(model.tok_embd.?, impl.sqrtf(@floatFromInt(b.n_embd)));
    b.cb(inpL, "inp_scaled", -1);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpNoCache();

    const inp_out_ids = b.buildInpOutIds();

    // inp_per_layer has shape [n_embd_per_layer, n_tokens, n_layer].
    const inp_per_layer = buildInpPerLayer(model, b, inpL);

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const ilu: u32 = @intCast(il);
        const n_embd_head: i64 = hp.nEmbdHeadK(ilu);
        const n_head: i64 = hp.nHead(ilu);
        const n_head_kv: i64 = hp.nHeadKv(ilu);

        const freq_base_l = model.getRopeFreqBase(b.cparams, ilu);
        const freq_scale_l = model.getRopeFreqScale(b.cparams, ilu);
        const n_rot_l: c_int = @intCast(hp.nRot(ilu));

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            var Qcur = b.buildLoraMm(layer.wq.?, cur, layer.wq_s);
            var Kcur = b.buildLoraMm(layer.wk.?, cur, layer.wk_s);
            var Vcur = b.buildLoraMm(layer.wv.?, cur, layer.wv_s);
            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            Vcur = c.ggml_rms_norm(ctx, Vcur, hp.f_norm_rms_eps);
            b.cb(Qcur, "Qcur_normed", il);
            b.cb(Kcur, "Kcur_normed", il);
            b.cb(Vcur, "Vcur_normed", il);

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, n_rot_l, b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, n_rot_l, b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            b.cb(Qcur, "Qcur_pos", il);
            b.cb(Kcur, "Kcur_pos", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, hp.f_attention_scale, il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }

        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        const attn_out: *Tensor = c.ggml_add(ctx, cur, inpL);
        b.cb(attn_out, "attn_out", il);

        // The feed-forward network.
        cur = b.buildNorm(attn_out, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .gelu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, -1);
        b.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx, cur, attn_out);

        // The per-layer embedding.
        {
            const pe_in = cur;
            b.cb(cur, "pe_in", il);

            cur = b.buildLoraMm(layer.per_layer_inp_gate.?, cur, null); // [n_embd_per_layer, n_tokens]
            cur = c.ggml_gelu(ctx, cur);

            var inp_this_layer: *Tensor = c.ggml_view_2d(ctx, inp_per_layer, inp_per_layer.ne[0], inp_per_layer.ne[1], inp_per_layer.nb[1], @as(usize, @intCast(il)) * inp_per_layer.nb[2]);

            if (il == b.n_layer - 1) {
                inp_this_layer = c.ggml_get_rows(ctx, inp_this_layer, inp_out_ids);
            }

            cur = c.ggml_mul(ctx, cur, inp_this_layer);
            cur = b.buildLoraMm(layer.per_layer_proj.?, cur, null); // [n_embd, n_tokens]
            cur = b.buildNorm(cur, layer.per_layer_post_norm, null, .rms, il);
            b.cb(cur, "per_layer_embd_out", il);

            cur = c.ggml_add(ctx, pe_in, cur);
        }

        // layer_scalar
        cur = c.ggml_mul(ctx, cur, layer.out_scale);
        b.cb(cur, "out_scaled", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    // Projecting per token is the same as projecting after mean pooling.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    b.cb(cur, "result_embd", -1);
    b.res.t_embd = cur;
}

/// Ports `llama_model_gemma_embedding2::graph`'s `build_inp_per_layer`
/// (models/gemma-embedding2.cpp:219 @620d2225c): the Python code's
/// `EmbeddingGemma2TextPLE`. The model has no per-layer token embeddings;
/// the per-layer inputs come from the projection alone.
///
/// Parameters:
/// - `inpL`: `[n_embd, n_tokens]`.
///
/// Return: `[n_embd_per_layer, n_tokens, n_layer]`.
fn buildInpPerLayer(model: *const Model, b: *const graph.Builder, inpL: *Tensor) *Tensor {
    const ctx = b.ctx0;
    const n_embd_per_layer: i64 = b.hparams.n_embd_per_layer;

    var cur: *Tensor = c.ggml_mul_mat(ctx, model.per_layer_model_proj, inpL); // [n_embd_per_layer * n_layer, n_tokens]
    cur = c.ggml_scale(ctx, cur, 1.0 / impl.sqrtf(@floatFromInt(b.n_embd)));
    cur = c.ggml_reshape_3d(ctx, cur, n_embd_per_layer, b.n_layer, b.n_tokens);

    cur = b.buildNorm(cur, model.per_layer_proj_norm, null, .rms, -1);
    b.cb(cur, "inp_per_layer", -1);

    // Permute to [n_embd_per_layer, n_tokens, n_layer].
    cur = c.ggml_cont(ctx, c.ggml_permute(ctx, cur, 0, 2, 1, 3));
    return cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
