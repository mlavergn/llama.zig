//! ModernBERT: a bidirectional encoder with LayerNorms without biases,
//! RoPE, alternating local (symmetric sliding-window) and global attention
//! and a gated feed-forward whose up projection carries the gate,
//! producing embeddings rather than logits.
//!
//! # Provenance
//!
//! A decision model (since v0.6.0) puts `n_layer_decision` blocks of a
//! head after the encoder: plain pre-norm blocks with biases and no
//! positional encoding, evaluated once per question type with that type's
//! embedding added, each ending in the classifier, so the output is one
//! score per question type.
//!
//! Ported from `llama.cpp/src/models/modern-bert.cpp` at v0.7.0
//! (`620d2225c`): `N_DECISION_TYPES`, `llama_model_modern_bert`'s
//! `load_arch_hparams`, `load_arch_tensors` and `build_arch_graph`, and
//! its `graph` with `build_decision_head`; and from
//! `llama.cpp/src/llama-model.cpp`, `llm_ffn_op_type_from_string`, which
//! only this architecture calls.

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

/// Ports `N_DECISION_TYPES` (models/modern-bert.cpp:4 @620d2225c): the
/// question types of a decision model -- choice, score and noul.
const n_decision_types: u32 = 3;

/// Ports `LLM_FFN_OP_TYPES_FROM_STRING` (llama-model.cpp:1091 @620d2225c):
/// the GGUF activation names, mapped to the gated feed-forward op the graph
/// builders use. `"gelu"` is the exact (erf) variant.
const ffn_op_types_from_string = [_]struct { []const u8, graph.FfnOp }{
    .{ "gelu", .geglu_erf },
    .{ "gelu_python", .geglu_erf },
    .{ "gelu_pytorch_tanh", .geglu },
    .{ "gelu_new", .geglu },
    .{ "gelu_fast", .geglu },
    .{ "gelu_accurate", .geglu },
    .{ "gelu_python_tanh", .geglu },
    .{ "geglu", .geglu },
    .{ "silu", .swiglu },
    .{ "swish", .swiglu },
    .{ "swiglu", .swiglu },
    .{ "relu", .relu },
    .{ "reglu", .reglu },
};

/// Ports `llm_ffn_op_type_from_string` (llama-model.cpp:1114 @620d2225c).
///
/// Parameters:
/// - `name`: the activation's name.
/// - `fallback`: the op for a name the map lacks.
///
/// Return: the op.
fn ffnOpTypeFromString(name: []const u8, fallback: graph.FfnOp) graph.FfnOp {
    for (ffn_op_types_from_string) |e| if (std.mem.eql(u8, e[0], name)) return e[1];
    return fallback;
}

/// Ports `llama_model_modern_bert`'s `load_arch_hparams` (models/modern-bert.cpp:6 @620d2225c).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    const found_swa = try meta.getKey(u32, .attention_sliding_window, false, exc);
    if (found_swa) |v| hp.n_swa = v;
    if (found_swa != null and hp.n_swa > 0) {
        hp.swa_type = @backingInt(hparams_mod.SwaType.symmetric);
        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
        try self.loadSwaPattern(ml, 3, true, exc);
    } else {
        hp.swa_type = @backingInt(hparams_mod.SwaType.none);
    }

    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    // Some ModernBert derivatives (e.g. IBM Granite Embedding 97m R2) use
    // SiLU/SwiGLU in the FFN instead of the default GELU/GeGLU.
    hp.llm_ffn_op = @backingInt(graph.FfnOp.geglu_erf);
    if (try meta.getKey([]const u8, .hidden_act, false, exc)) |hidden_act| {
        hp.llm_ffn_op = @backingInt(ffnOpTypeFromString(hidden_act, .geglu_erf));
    }

    // GGUFs without a classifier pooling type use mean (gte-reranker-modernbert-base).
    if (hp.pooling_type_cls == c.LLAMA_POOLING_TYPE_UNSPECIFIED) {
        hp.pooling_type_cls = c.LLAMA_POOLING_TYPE_MEAN;
    }

    // GGUFs without a classifier activation use gelu, the transformers
    // default.
    if (try meta.getKey([]const u8, .classifier_activation, false, exc) == null) {
        hp.act_cls = c.GGML_UNARY_OP_GELU_ERF;
    }

    if (try meta.getKey(u32, .decision_block_count, false, exc)) |v| hp.n_layer_decision = v;
    if (hp.n_layer_decision > 0) {
        if (hp.n_layer_decision >= hp.nLayer()) {
            return exc.throw(.runtime_error, "invalid number of decision blocks", .{});
        }
        // The head's blocks always use full attention.
        var il: u32 = hp.nLayer() - hp.n_layer_decision;
        while (il < hp.nLayer()) : (il += 1) hp.is_swa_impl[il] = 0;
        // The output is one score per question type.
        hp.n_embd_out_impl = n_decision_types;
    }

    self.type = switch (hp.nLayer()) {
        12 => .@"47m", // granite-embedding-small
        22 => .@"149m", // modern-bert-base
        28 => .@"395m", // modern-bert-large
        else => .unknown,
    };
}

/// Ports `llama_model_modern_bert`'s `load_arch_tensors` (models/modern-bert.cpp:61 @620d2225c).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.tok_norm = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "weight", .bid = 0 }, &.{l.n_embd}, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    const n_layer_enc: i32 = l.n_layer - @as(i32, @intCast(hp.n_layer_decision));

    var i: i32 = 0;
    while (i < n_layer_enc) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        if (i != 0) {
            layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        } else {
            // Layer 0 uses the identity.
            layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        }

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, 3 * l.n_embd }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);

        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, 2 * l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
    }

    self.cls_out = try self.createTensor(.{ .tensor = .cls_out, .suffix = "weight" }, &.{ l.n_embd, hp.n_cls_out }, nr, exc);
    self.cls_out_b = try self.createTensor(.{ .tensor = .cls_out, .suffix = "bias" }, &.{hp.n_cls_out}, nr, exc);
    self.cls = try self.createTensor(.{ .tensor = .cls, .suffix = "weight" }, &.{ l.n_embd, l.n_embd }, nr, exc);
    self.cls_norm = try self.createTensor(.{ .tensor = .cls_norm, .suffix = "weight" }, &.{l.n_embd}, nr, exc);

    if (hp.n_layer_decision == 0) return;

    // The decision head: plain pre-norm blocks with biases.
    i = n_layer_enc;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const n_ff_head: i64 = hp.nFf(@intCast(i));

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_b = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, 3 * l.n_embd }, 0, exc);
        layer.wqkv_b = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "bias", .bid = i }, &.{3 * l.n_embd}, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_norm_b = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_head }, 0, exc);
        layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{n_ff_head}, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ n_ff_head, l.n_embd }, 0, exc);
        layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
    }

    if (l.n_token_types != n_decision_types) {
        return exc.throw(.runtime_error, "decision model must have one token type per question type", .{});
    }
    self.type_embd = try self.createTensor(.{ .tensor = .token_types, .suffix = "weight" }, &.{ l.n_embd, l.n_token_types }, 0, exc);

    self.cls_b = try self.createTensor(.{ .tensor = .cls, .suffix = "bias" }, &.{l.n_embd}, 0, exc);
    self.cls_norm_b = try self.createTensor(.{ .tensor = .cls_norm, .suffix = "bias" }, &.{l.n_embd}, 0, exc);

    if (self.cls == null or self.cls_norm == null or self.cls_out == null or self.cls_out_b == null) {
        return exc.throw(.runtime_error, "decision model is missing the scorer tensors", .{});
    }
}

/// Ports `llama_model_modern_bert`'s `build_arch_graph` (models/modern-bert.cpp:134 @620d2225c).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_modern_bert`'s `graph`
/// (models/modern-bert.cpp:138 @620d2225c).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/modern-bert.cpp", 141, "n_embd_head == hparams.n_embd_head_k()");

    const inp_pos = b.buildInpPos();

    // The input embeddings.
    var inpL = b.buildInpEmbd(model.tok_embd.?);
    b.cb(inpL, "inp_embd", -1);

    // The embeddings' layer norm.
    inpL = b.buildNorm(inpL, model.tok_norm, null, .norm, 0);
    b.cb(inpL, "inp_norm", 0);

    const inp_out_ids = b.buildInpOutIds();

    const inp_attn = b.buildAttnInpNoCache();

    const n_layer_dec: c_int = @intCast(hp.n_layer_decision);
    const n_layer_enc: c_int = @as(c_int, @intCast(b.n_layer)) - n_layer_dec;

    var il: c_int = 0;
    while (il < n_layer_enc) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
        const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

        var cur = inpL;

        // The attention's layer norm.
        if (layer.attn_norm != null) {
            cur = b.buildNorm(inpL, layer.attn_norm, null, .norm, il);
            b.cb(cur, "attn_norm", il);
        }

        // Self-attention.
        const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
        var Qcur = qkv.q;
        var Kcur = qkv.k;
        const Vcur = qkv.v;

        // RoPE.
        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        b.cb(Qcur, "Qcur", il);
        b.cb(Kcur, "Kcur", il);
        b.cb(Vcur, "Vcur", il);

        cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        b.cb(cur, "kqv_out", il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }

        // Re-add the layer input.
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward's layer norm.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .norm, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, null, null, null, layer.ffn_down, null, null, null, @fromBackingInt(hp.llm_ffn_op), .seq, il);

        // The attention output bypasses the feed-forward.
        cur = c.ggml_add(ctx, cur, ffn_inp);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .norm, -1);
    b.cb(cur, "final_norm_out", -1);

    if (n_layer_dec > 0) {
        cur = buildDecisionHead(model, b, cur, inp_attn, inp_out_ids);
    }

    b.res.t_embd = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_modern_bert::graph`'s `build_decision_head`
/// (models/modern-bert.cpp:248 @620d2225c): one score per question type for
/// each output token, read at the option markers by the caller. The
/// question type is not a graph input, so the head is evaluated once per
/// type.
fn buildDecisionHead(model: *const Model, b: *const graph.Builder, inp: *Tensor, inp_attn: *graph.InpAttnNoCache, inp_out_ids: *Tensor) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const n_layer: c_int = @intCast(b.n_layer);
    const n_layer_enc: c_int = n_layer - @as(c_int, @intCast(hp.n_layer_decision));
    const type_embd = model.type_embd.?;

    var scores: ?*Tensor = null;

    var it: u32 = 0;
    while (it < n_decision_types) : (it += 1) {
        const type_row = c.ggml_view_1d(ctx, type_embd, b.n_embd, it * type_embd.nb[1]);
        var inpL: *Tensor = c.ggml_add(ctx, inp, type_row);

        var il: c_int = n_layer_enc;
        while (il < n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];

            var cur = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .norm, il);
            b.cb(cur, "attn_norm", il);

            // No positional encoding in the head.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, qkv.q, qkv.k, qkv.v, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
            b.cb(cur, "kqv_out", il);

            if (il == n_layer - 1) {
                cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
                inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
            }

            const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
            b.cb(ffn_inp, "ffn_inp", il);

            cur = b.buildNorm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, .norm, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, null, null, null, layer.ffn_down, layer.ffn_down_b, null, null, .relu, .seq, il);

            inpL = c.ggml_add(ctx, cur, ffn_inp);
        }

        // The scorer.
        var cur = b.buildNorm(inpL, model.cls_norm, model.cls_norm_b, .norm, -1);
        cur = c.ggml_add(ctx, b.buildLoraMm(model.cls.?, cur, null), model.cls_b);
        cur = c.ggml_gelu_erf(ctx, cur);
        cur = c.ggml_add(ctx, b.buildLoraMm(model.cls_out.?, cur, null), model.cls_out_b);

        scores = if (scores) |s| c.ggml_concat(ctx, s, cur, 0) else cur;
    }
    b.cb(scores.?, "decision_scores", -1);

    return scores.?;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an unknown activation falls back" {
    try std.testing.expectEqual(graph.FfnOp.swiglu, ffnOpTypeFromString("silu", .geglu));
    try std.testing.expectEqual(graph.FfnOp.geglu, ffnOpTypeFromString("tanh", .geglu));
}
