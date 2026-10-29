//! A context's resolved parameters.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-cparams.h` -- `LLAMA_MAX_SEQ` and `llama_cparams`
//! - `llama.cpp/src/llama-cparams.cpp` -- `llama_max_parallel_sequences`
//!
//! All at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("impl.zig");

const c = impl.c;

/// `LLAMA_MAX_SEQ` (llama-cparams.h:8 @b29c606e2).
pub const max_seq: u32 = 256;

/// Ports `llama_max_parallel_sequences` (llama-cparams.cpp:3 @b29c606e2).
pub fn maxParallelSequences() usize {
    return max_seq;
}

/// Mirrors `llama_cparams` (llama-cparams.h:10 @b29c606e2).
pub const Cparams = struct {
    /// Context size used during inference.
    n_ctx: u32 = 0,
    /// Context for a single sequence.
    n_ctx_seq: u32 = 0,
    n_batch: u32 = 0,
    n_ubatch: u32 = 0,
    n_seq_max: u32 = 0,
    /// Recurrent-state snapshots per sequence, for rollback.
    n_rs_seq: u32 = 0,
    /// The most outputs the context supports.
    n_outputs_max: u32 = 0,
    n_outputs_max_per_seq: u32 = 0,
    n_threads: i32 = 0,
    n_threads_batch: i32 = 0,

    nextn_layer_offset: i32 = 0,

    rope_freq_base: f32 = 0,
    rope_freq_scale: f32 = 0,

    n_ctx_orig_yarn: u32 = 0,
    yarn_ext_factor: f32 = 0,
    yarn_attn_factor: f32 = 0,
    yarn_beta_fast: f32 = 0,
    yarn_beta_slow: f32 = 0,

    embeddings: bool = false,
    /// Also extract the hidden state before the final output norm.
    embeddings_nextn: bool = false,
    /// Extract it only for rows where `batch.logits != 0`.
    embeddings_nextn_masked: bool = false,
    causal_attn: bool = false,
    offload_kqv: bool = false,
    flash_attn: bool = false,
    auto_fa: bool = false,
    /// Fused gated delta net, autoregressive.
    fused_gdn_ar: bool = false,
    /// Fused gated delta net, chunked.
    fused_gdn_ch: bool = false,
    auto_fgdn: bool = false,
    /// Fused lightning indexer.
    fused_lid: bool = false,
    auto_flid: bool = false,
    fused_dsv4_hc_pre: bool = false,
    fused_dsv4_hc_comb: bool = false,
    fused_dsv4_hc_post: bool = false,
    auto_fhc: bool = false,
    no_perf: bool = false,
    warmup: bool = false,
    op_offload: bool = false,
    kv_unified: bool = false,
    pipeline_parallel: bool = false,

    /// `std::vector<bool>`, `[n_layer()]`: extract each layer's input
    /// embeddings. Shares its storage across copies: the C++ copies the
    /// vector into every graph's parameters, and nothing writes it after
    /// the context is built.
    embeddings_layer_inp: []const bool = &.{},

    ctx_type: c.enum_llama_context_type = 0,
    /// The RoPE scaling the context resolved: the parameter, or the
    /// model's trained one when unspecified.
    rope_scaling_type: c.enum_llama_rope_scaling_type = 0,
    pooling_type: c.enum_llama_pooling_type = 0,

    cb_eval: c.ggml_backend_sched_eval_callback = null,
    cb_eval_user_data: ?*anyopaque = null,

    /// The other context, `llama_context *`, opaque here.
    ctx_other: ?*anyopaque = null,
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
