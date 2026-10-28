//! LFM2-MoE: LFM2 with a mixture of experts after the leading dense
//! blocks. Its tensors and graph are LFM2's; only the hyperparameters
//! differ.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/lfm2moe.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_lfm2moe`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`. Its `graph` is `llama_model_lfm2`'s, in `lfm2.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const lfm2 = @import("lfm2.zig");

const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;

/// Ports `llama_model_lfm2moe`'s `load_arch_hparams` (models/lfm2moe.cpp:5 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
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
/// @5266f24da): LFM2's body.
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    try lfm2.loadTensors(self, "models/lfm2moe.cpp", 43, 61, exc);
}

/// Ports `llama_model_lfm2moe`'s `build_arch_graph` (models/lfm2moe.cpp:78
/// @5266f24da): LFM2's graph, with or without the sliding window.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    lfm2.buildArchGraph(model, b, params);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
