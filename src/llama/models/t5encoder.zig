//! T5's encoder alone: the encoder stack of an encoder-decoder T5, with no
//! decoder layers and no cross-attention, for embeddings.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/t5encoder.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_model_t5encoder`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`. Its graph is `llama_model_t5`'s encoder graph,
//! which `t5.zig` ports.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const t5 = @import("t5.zig");

const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;

/// Ports `llama_model_t5encoder`'s `load_arch_hparams` (models/t5encoder.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_rel_attn_bkts = (try meta.getKey(u32, .attention_relative_buckets_count, true, exc)).?;
    self.type = .unknown;
}

/// Ports `llama_model_t5encoder`'s `load_arch_tensors` (models/t5encoder.cpp:9 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    const n_rel_attn_bkts: i64 = self.hparams.n_rel_attn_bkts;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm_enc = try self.createTensor(.{ .tensor = .enc_output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    try t5.loadEncoderLayers(self, l, n_rel_attn_bkts, exc);
}

/// Ports `llama_model_t5encoder`'s `build_arch_graph` (models/t5encoder.cpp:42
/// @b29c606e2): T5's encoder graph, whatever the graph type.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    t5.buildEncoder(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
