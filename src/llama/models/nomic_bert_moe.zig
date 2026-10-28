//! Nomic-BERT MoE: Nomic-BERT with a mixture of GELU experts in place of
//! the feed-forward on every `moe_every_n_layers`-th layer.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/nomic-bert-moe.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_nomic_bert_moe`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`. Its graph is
//! `llama_model_bert`'s, in `bert.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const bert = @import("bert.zig");
const nomic_bert = @import("nomic_bert.zig");

const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;

/// Ports `llama_model_nomic_bert_moe`'s `load_arch_hparams` (models/nomic-bert-moe.cpp:3 @5266f24da),
/// whose body is `llama_model_nomic_bert`'s `load_arch_hparams` (models/nomic-bert.cpp:3 @5266f24da)
/// line for line.
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    return nomic_bert.loadArchHparams(self, ml, exc);
}

/// Ports `llama_model_nomic_bert_moe`'s `load_arch_tensors` (models/nomic-bert-moe.cpp:16 @5266f24da),
/// whose body is `llama_model_bert`'s `load_arch_tensors` (models/bert.cpp:23 @5266f24da)
/// line for line, its checks on the architecture included.
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    return bert.loadArchTensors(self, ml, exc);
}

/// Ports `llama_model_nomic_bert_moe`'s `build_arch_graph` (models/nomic-bert-moe.cpp:69 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    bert.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
