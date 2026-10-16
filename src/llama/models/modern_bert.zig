//! ModernBERT: a bidirectional encoder with LayerNorms without biases,
//! RoPE, alternating local (symmetric sliding-window) and global attention
//! and a gated feed-forward whose up projection carries the gate,
//! producing embeddings rather than logits.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/modern-bert.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_modern_bert`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`; and from
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

/// Ports `LLM_FFN_OP_TYPES_FROM_STRING` (llama-model.cpp:989 @c1d0e7a00):
/// the GGUF `<arch>.hidden_activation` strings that map cleanly to a gated
/// feed-forward op.
const ffn_op_types_from_string = [_]struct { []const u8, graph.FfnOp }{
    .{ "gelu", .geglu },
    .{ "geglu", .geglu },
    .{ "silu", .swiglu },
    .{ "swish", .swiglu },
    .{ "swiglu", .swiglu },
    .{ "relu", .relu },
    .{ "reglu", .reglu },
};

/// Ports `llm_ffn_op_type_from_string` (llama-model.cpp:999 @c1d0e7a00).
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

/// Ports `llama_model_modern_bert`'s `load_arch_hparams` (models/modern-bert.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    const found_swa = try meta.getKey(u32, .attention_sliding_window, false, exc);
    if (found_swa) |v| hp.n_swa = v;
    if (found_swa != null and hp.n_swa > 0) {
        hp.swa_type = @backingInt(hparams_mod.SwaType.symmetric);
        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
        var swa_period: u32 = 3;
        if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |v| swa_period = v;
        hp.setSwaPattern(swa_period, true);
    } else {
        hp.swa_type = @backingInt(hparams_mod.SwaType.none);
    }

    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    // Some ModernBert derivatives (e.g. IBM Granite Embedding 97m R2) use
    // SiLU/SwiGLU in the FFN instead of the default GELU/GeGLU.
    hp.llm_ffn_op = @backingInt(graph.FfnOp.geglu);
    if (try meta.getKey([]const u8, .hidden_act, false, exc)) |hidden_act| {
        hp.llm_ffn_op = @backingInt(ffnOpTypeFromString(hidden_act, .geglu));
    }

    self.type = switch (hp.nLayer()) {
        12 => .@"47m", // granite-embedding-small
        22 => .@"149m", // modern-bert-base
        28 => .@"395m", // modern-bert-large
        else => .unknown,
    };
}

/// Ports `llama_model_modern_bert`'s `load_arch_tensors` (models/modern-bert.cpp:36 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.tok_norm = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "weight", .bid = 0 }, &.{l.n_embd}, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
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
}

/// Ports `llama_model_modern_bert`'s `build_arch_graph` (models/modern-bert.cpp:70 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_modern_bert`'s `graph`
/// (models/modern-bert.cpp:74 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/modern-bert.cpp", 77, "n_embd_head == hparams.n_embd_head_k()");

    const inp_pos = b.buildInpPos();

    // The input embeddings.
    var inpL = b.buildInpEmbd(model.tok_embd.?);
    b.cb(inpL, "inp_embd", -1);

    // The embeddings' layer norm.
    inpL = b.buildNorm(inpL, model.tok_norm, null, .norm, 0);
    b.cb(inpL, "inp_norm", 0);

    const inp_out_ids = b.buildInpOutIds();

    const inp_attn = b.buildAttnInpNoCache();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
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

    b.res.t_embd = cur;
    c.ggml_build_forward_expand(b.gf, cur);
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
