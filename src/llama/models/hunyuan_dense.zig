//! Hunyuan's dense models: Hunyuan-VL's language model, loaded the same way
//! and built with the same graph.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/hunyuan-dense.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_hunyuan_dense`'s `get_rope_params` and
//! `build_arch_graph`. The
//! class derives from `llama_model_hunyuan_vl` and reuses its
//! `load_arch_hparams`, `load_arch_tensors` and `graph`, which
//! `hunyuan_vl.zig` ports; the registry names them.

const std = @import("std");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hunyuan_vl = @import("hunyuan_vl.zig");

const Model = model_mod.Model;
const Cparams = @import("../cparams.zig").Cparams;
const RopeParams = model_mod.RopeParams;

/// Ports `llama_model_hunyuan_dense`'s `get_rope_params` (models/hunyuan-dense.cpp:3 @b1b770d90):
/// the RoPE of a layer, for the graph and the K-shift alike.
///
/// Parameters:
/// - `model`: the model.
/// - `cparams`: the context's parameters.
/// - `il`: the layer.
///
/// Return: the layer's RoPE parameters.
pub fn getRopeParams(model: *const Model, cparams: *const Cparams, il: u32) RopeParams {
    var res = model.getRopeParamsBase(cparams, il);

    // The K norm applied after RoPE, which a K-shift has to undo.
    res.k_norm = model.layers[il].attn_k_norm;

    return res;
}

/// Ports `llama_model_hunyuan_dense`'s `build_arch_graph`
/// (models/hunyuan-dense.cpp:11 @b1b770d90): Hunyuan-VL's graph.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    hunyuan_vl.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
