//! Hunyuan's dense models: Hunyuan-VL's language model, loaded the same way
//! and built with the same graph.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/hunyuan-dense.cpp` at v0.4.1
//! (`b29c606e2`): `llama_model_hunyuan_dense`'s `build_arch_graph`. The
//! class derives from `llama_model_hunyuan_vl` and reuses its
//! `load_arch_hparams`, `load_arch_tensors` and `graph`, which
//! `hunyuan_vl.zig` ports; the registry names them.

const std = @import("std");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hunyuan_vl = @import("hunyuan_vl.zig");

const Model = model_mod.Model;

/// Ports `llama_model_hunyuan_dense`'s `build_arch_graph`
/// (models/hunyuan-dense.cpp:3 @b29c606e2): Hunyuan-VL's graph.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    hunyuan_vl.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
