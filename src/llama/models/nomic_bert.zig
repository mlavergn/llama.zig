//! Nomic-BERT: BERT with RoPE in place of learned positions and a SwiGLU
//! feed-forward, producing embeddings.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/nomic-bert.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_nomic_bert`'s `load_arch_hparams`,
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

/// Ports `llama_model_nomic_bert`'s `load_arch_hparams` (models/nomic-bert.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;
    if (try meta.getKey(u32, .moe_every_n_layers, false, exc)) |v| hp.moe_every_n_layers = v;

    if (hp.nLayer() == 12 and hp.n_embd == 768) {
        if (self.arch == .nomic_bert) {
            self.type = .@"137m";
        } else if (self.arch == .nomic_bert_moe and hp.moe_every_n_layers == 2) {
            self.type = .@"475m";
        }
    }
}

/// Ports `llama_model_nomic_bert`'s `load_arch_tensors` (models/nomic-bert.cpp:16 @c1d0e7a00),
/// whose body is `llama_model_bert`'s `load_arch_tensors` (models/bert.cpp:23 @c1d0e7a00)
/// line for line, its checks on the architecture included.
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    return bert.loadArchTensors(self, ml, exc);
}

/// Ports `llama_model_nomic_bert`'s `build_arch_graph` (models/nomic-bert.cpp:69 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    bert.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
