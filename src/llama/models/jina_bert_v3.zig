//! Jina-BERT v3: BERT with RoPE in place of learned positions, producing
//! embeddings.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/jina-bert-v3.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_jina_bert_v3`'s `load_arch_hparams`,
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

/// Ports `llama_model_jina_bert_v3`'s `load_arch_hparams` (models/jina-bert-v3.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_eps = (try ml.meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        24 => .@"558m",
        else => .unknown,
    };
}

/// Ports `llama_model_jina_bert_v3`'s `load_arch_tensors` (models/jina-bert-v3.cpp:13 @5266f24da),
/// whose body is `llama_model_bert`'s `load_arch_tensors` (models/bert.cpp:23 @5266f24da)
/// line for line, its checks on the architecture included.
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    return bert.loadArchTensors(self, ml, exc);
}

/// Ports `llama_model_jina_bert_v3`'s `build_arch_graph` (models/jina-bert-v3.cpp:66 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    bert.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
