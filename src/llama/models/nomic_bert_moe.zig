//! Nomic-BERT MoE: Nomic-BERT with a mixture of GELU experts in place of
//! the feed-forward on every `moe_every_n_layers`-th layer.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/nomic-bert-moe.cpp` at v0.4.1
//! (`b29c606e2`): `llama_model_nomic_bert_moe`'s `load_arch_hparams`,
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

/// Ports `llama_model_nomic_bert_moe`'s `load_arch_hparams` (models/nomic-bert-moe.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;
    if (try meta.getKey(u32, .moe_every_n_layers, false, exc)) |v| hp.moe_every_n_layers = v;

    self.type = switch (hp.nLayer()) {
        12 => .@"475m",
        else => .unknown,
    };
}

/// Ports `llama_model_nomic_bert_moe`'s `load_arch_tensors` (models/nomic-bert-moe.cpp:14 @b29c606e2):
/// BERT's body without positions or classifier, with an expert layer every
/// `moe_every_n_layers`.
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    return bert.loadBertTensors(self, .{ .pos_cls = false, .moe = true, .gate = false }, exc);
}

/// Ports `llama_model_nomic_bert_moe`'s `build_arch_graph` (models/nomic-bert-moe.cpp:53 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    bert.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
