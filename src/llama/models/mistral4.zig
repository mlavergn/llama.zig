//! Mistral 4: DeepSeek-V2's architecture -- multi-head latent attention,
//! leading dense blocks, then a mixture of experts with shared experts --
//! without its MTP draft head.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mistral4.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_mistral4`'s `build_arch_graph`. The class derives from
//! `llama_model_deepseek2` and reuses its `load_arch_hparams`,
//! `load_arch_tensors` and `graph`, which `deepseek2.zig` holds.

const std = @import("std");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const deepseek2 = @import("deepseek2.zig");

const Model = model_mod.Model;

/// Ports `llama_model_mistral4`'s `build_arch_graph` (models/mistral4.cpp:3
/// @5266f24da): DeepSeek-V2's main graph, whatever the graph type.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    deepseek2.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
