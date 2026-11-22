//! T5's encoder alone: the encoder stack of an encoder-decoder T5, with no
//! decoder layers and no cross-attention, for embeddings.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/t5encoder.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_t5encoder`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`. Its graph is `llama_model_t5`'s encoder graph,
//! which `t5.zig` ports.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_t5encoder`'s `load_arch_hparams` (models/t5encoder.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_rel_attn_bkts = (try meta.getKey(u32, .attention_relative_buckets_count, true, exc)).?;
    self.type = .unknown;
}

/// Ports `llama_model_t5encoder`'s `load_arch_tensors` (models/t5encoder.cpp:9 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const nr = llama.loader.tensor_not_required;
    const dup = llama.loader.tensor_duplicated;

    const n_rel_attn_bkts: i64 = self.hparams.n_rel_attn_bkts;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm_enc = try self.createTensor(.{ .tensor = .enc_output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    try llama.models.t5.loadEncoderLayers(self, l, n_rel_attn_bkts, exc);
}

/// Ports `llama_model_t5encoder`'s `build_arch_graph` (models/t5encoder.cpp:42
/// @b1b770d90): T5's encoder graph, whatever the graph type.
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    llama.models.t5.buildEncoder(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
