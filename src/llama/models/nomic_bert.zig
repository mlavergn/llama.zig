//! Nomic-BERT: BERT with RoPE in place of learned positions and a SwiGLU
//! feed-forward, producing embeddings.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/nomic-bert.cpp` at v0.4.1
//! (`b29c606e2`): `llama_model_nomic_bert`'s `load_arch_hparams`,
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

/// Ports `llama_model_nomic_bert`'s `load_arch_hparams` (models/nomic-bert.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_eps = (try ml.meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        12 => .@"137m",
        else => .unknown,
    };
}

/// Ports `llama_model_nomic_bert`'s `load_arch_tensors` (models/nomic-bert.cpp:13 @b29c606e2):
/// BERT's body without positions or classifier, with a SwiGLU gate.
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    return bert.loadBertTensors(self, .{ .pos_cls = false, .moe = false, .gate = true }, exc);
}

/// Ports `llama_model_nomic_bert`'s `build_arch_graph` (models/nomic-bert.cpp:48 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    bert.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
