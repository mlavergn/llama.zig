//! Gemma 3: Gemma 2's sandwich of RMS norms around attention and the
//! feed-forward, with RMS-normed queries and keys, a five-to-one pattern of
//! sliding-window and global layers, and optional soft-capped logits.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/gemma3.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_gemma3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph<iswa>`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_gemma3`'s `load_arch_hparams` (models/gemma3.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    var found_swa = false;
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| {
        hp.n_swa = v;
        found_swa = true;
    }
    if (found_swa and hp.n_swa > 0) {
        hp.swa_type = @backingInt(llama.hparams.SwaType.standard);
        try self.loadSwaPattern(ml, 6, false, exc);

        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    } else {
        hp.swa_type = @backingInt(llama.hparams.SwaType.none);
    }

    hp.f_final_logit_softcapping = 0.0;
    if (try meta.getKey(f32, .final_logit_softcapping, false, exc)) |v| hp.f_final_logit_softcapping = v;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        18 => .@"270m",
        26 => .@"1b",
        32 => .@"8b", // Rnj-1
        34 => .@"4b",
        48 => .@"12b",
        62 => .@"27b",
        else => .unknown,
    };

    // https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L289
    hp.f_attention_scale = if (self.type == .@"27b")
        1.0 / stdlib.libm.sqrtfInt(hp.n_embd / hp.nHead(0))
    else
        1.0 / stdlib.libm.sqrtfInt(hp.nEmbdHeadK(0));
}

/// Ports `llama_model_gemma3`'s `load_arch_tensors` (models/gemma3.cpp:34 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    // Dense linear weights.
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

/// Ports `llama_model_gemma3`'s `build_arch_graph` (models/gemma3.cpp:73 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    if (b.hparams.swa_type == @backingInt(llama.hparams.SwaType.standard)) {
        buildGraph(true, model, b);
    } else {
        buildGraph(false, model, b);
    }
}

/// Ports the constructor of `llama_model_gemma3`'s `graph<iswa>`
/// (models/gemma3.cpp:82 @b1b770d90).
fn buildGraph(comptime iswa: bool, model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadK(0);

    // Important: do not scale raw embeddings input (encoded image embeddings):
    // `build_inp_embd` scales the token rows only.
    var inpL: *llama.impl.c.ggml_tensor = b.buildInpEmbdScaled(model.tok_embd.?, stdlib.libm.sqrtfInt(b.n_embd));
    b.cb(inpL, "inp_scaled", -1);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

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

            Qcur = b.buildRope(Qcur, inp_pos, il);

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Kcur, "Kcur_normed", il);

            Kcur = b.buildRope(Kcur, inp_pos, il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            // https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
            Qcur = llama.impl.c.ggml_scale(ctx, Qcur, hp.f_attention_scale);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0, il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = llama.impl.c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }
        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        const sa_out: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpL);
        b.cb(sa_out, "sa_out", il);

        cur = b.buildNorm(sa_out, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // The feed-forward network.
        {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .gelu, .par, il);
            b.cb(cur, "ffn_out", il);
        }
        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, -1);
        b.cb(cur, "ffn_post_norm", il);

        cur = llama.impl.c.ggml_add(ctx, cur, sa_out);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    if (hp.f_final_logit_softcapping != 0.0) {
        cur = llama.impl.c.ggml_scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
        cur = llama.impl.c.ggml_tanh(ctx, cur);
        cur = llama.impl.c.ggml_scale(ctx, cur, hp.f_final_logit_softcapping);
    }

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
