//! Jina-BERT v2: BERT with ALiBi in place of position embeddings, optional
//! LayerNorms on the queries and keys and after the attention, and a GeGLU
//! feed-forward whose up projection may carry the gate.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/jina-bert-v2.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_jina_bert_v2`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`. Its graph is
//! `llama_model_bert`'s, in `bert.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const bert = @import("bert.zig");

const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;

/// Ports `llama_model_jina_bert_v2`'s `load_arch_hparams` (models/jina-bert-v2.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_eps = (try ml.meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;
    hp.f_max_alibi_bias = 8.0;

    self.type = switch (hp.nLayer()) {
        4 => .@"33m", // jina-embeddings-small
        12 => .@"137m", // jina-embeddings-base
        else => .unknown,
    };
}

/// Ports `llama_model_jina_bert_v2`'s `load_arch_tensors` (models/jina-bert-v2.cpp:14 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc); // word_embeddings
    self.type_embd = try self.createTensor(.{ .tensor = .token_types, .suffix = "weight" }, &.{ l.n_embd, l.n_token_types }, 0, exc); // token_type_embeddings

    self.tok_norm = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "weight", .bid = 0 }, &.{l.n_embd}, 0, exc); // LayerNorm
    self.tok_norm_b = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "bias", .bid = 0 }, &.{l.n_embd}, 0, exc); // LayerNorm bias

    self.cls = try self.createTensor(.{ .tensor = .cls, .suffix = "weight" }, &.{ l.n_embd, 1 }, nr, exc);
    self.cls_b = try self.createTensor(.{ .tensor = .cls, .suffix = "bias" }, &.{1}, nr, exc);
    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)]; // JinaBertLayer

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, l.n_embd_gqa, l.n_embd_gqa, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.attn_q_norm_b = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.attn_k_norm_b = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc); // output_dens
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc); // output_dens

        layer.attn_out_norm = try self.createTensor(.{ .tensor = .attn_out_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc); // output_norm
        layer.attn_out_norm_b = try self.createTensor(.{ .tensor = .attn_out_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.attn_norm_2 = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.attn_norm_2_b = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);

        const tn_ffn_up_weight: loader.Tn = .{ .tensor = .ffn_up, .suffix = "weight", .bid = i };
        const tn_name = tn_ffn_up_weight.str();
        const t_ffn_up = ml.getTensorMeta(tn_name.slice());
        const n_ffn_up: i64 = if (t_ffn_up) |t| t.ne[1] else l.n_ff;

        impl.assert(n_ffn_up == l.n_ff or n_ffn_up == l.n_ff * 2, "models/jina-bert-v2.cpp", 51, "n_ffn_up == n_ff || n_ffn_up == n_ff * 2");
        layer.ffn_up = try self.createTensor(tn_ffn_up_weight, &.{ l.n_embd, n_ffn_up }, 0, exc);
        layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{n_ffn_up}, nr, exc);

        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.layer_out_norm = try self.createTensor(.{ .tensor = .layer_out_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.layer_out_norm_b = try self.createTensor(.{ .tensor = .layer_out_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
    }
}

/// Ports `llama_model_jina_bert_v2`'s `build_arch_graph` (models/jina-bert-v2.cpp:63 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    bert.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
