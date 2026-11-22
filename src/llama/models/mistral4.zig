//! Mistral 4: DeepSeek-V2's architecture -- multi-head latent attention,
//! leading dense blocks, then a mixture of experts with shared experts --
//! without its MTP draft head.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mistral4.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_mistral4`'s `build_arch_graph`. The class derives from
//! `llama_model_deepseek2` and reuses its `load_arch_hparams`,
//! `load_arch_tensors` and `graph`, which `deepseek2.zig` holds.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_mistral4`'s `build_arch_graph` (models/mistral4.cpp:3
/// @b1b770d90): DeepSeek-V2's main graph, whatever the graph type.
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    llama.models.deepseek2.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
