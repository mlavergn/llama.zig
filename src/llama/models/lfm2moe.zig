//! LFM2-MoE: LFM2 with a mixture of experts after the leading dense
//! blocks. Its tensors and graph are LFM2's; only the hyperparameters
//! differ.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/lfm2moe.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_lfm2moe`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`. Its `graph` is `llama_model_lfm2`'s, in `lfm2.zig`.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_lfm2moe`'s `load_arch_hparams` (models/lfm2moe.cpp:5 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.n_shortconv_l_cache = (try meta.getKey(u32, .shortconv_l_cache, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;

    var il: u32 = 0;
    while (il < hp.nLayer()) : (il += 1) {
        hp.is_recr_impl[il] = @intFromBool(hp.nHeadKv(il) == 0);
    }

    self.type = switch (hp.nLayer()) {
        24 => .@"8b_a1b",
        40 => .@"24b_a2b",
        else => .unknown,
    };
}

/// Ports `llama_model_lfm2moe`'s `load_arch_tensors` (models/lfm2moe.cpp:23
/// @b1b770d90): LFM2's body.
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    try llama.models.lfm2.loadTensors(self, "models/lfm2moe.cpp", 43, 61, exc);
}

/// Ports `llama_model_lfm2moe`'s `build_arch_graph` (models/lfm2moe.cpp:78
/// @b1b770d90): LFM2's graph, with or without the sliding window.
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, params: *const llama.graph.Params) void {
    llama.models.lfm2.buildArchGraph(model, b, params);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
