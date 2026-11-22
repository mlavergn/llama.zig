//! Nomic-BERT: BERT with RoPE in place of learned positions and a SwiGLU
//! feed-forward, producing embeddings.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/nomic-bert.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_nomic_bert`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`. Its graph is
//! `llama_model_bert`'s, in `bert.zig`.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_nomic_bert`'s `load_arch_hparams` (models/nomic-bert.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_eps = (try ml.meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        12 => .@"137m",
        else => .unknown,
    };
}

/// Ports `llama_model_nomic_bert`'s `load_arch_tensors` (models/nomic-bert.cpp:13 @b1b770d90):
/// BERT's body without positions or classifier, with a SwiGLU gate.
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    return llama.models.bert.loadBertTensors(self, .{ .pos_cls = false, .moe = false, .gate = true }, exc);
}

/// Ports `llama_model_nomic_bert`'s `build_arch_graph` (models/nomic-bert.cpp:48 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    llama.models.bert.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
