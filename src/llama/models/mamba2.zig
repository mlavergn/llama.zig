//! Mamba-2: Mamba's stack with the Mamba-2 block, whose state-space
//! parameters are per head and grouped, and a grouped RMS norm before the
//! output projection.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mamba2.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_model_mamba2`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`. Its graph is `llama_model_mamba`'s, in `mamba.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const mamba = @import("mamba.zig");

const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;

/// Ports `llama_model_mamba2`'s `load_arch_hparams` (models/mamba2.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        24 => switch (hp.n_embd) {
            768 => .small,
            else => .unknown,
        },
        48 => switch (hp.n_embd) {
            1024 => .medium,
            1536 => .large,
            2048 => .xl,
            else => .unknown,
        },
        64 => switch (hp.n_embd) {
            2560 => .@"3b",
            4096 => .@"7b",
            else => .unknown,
        },
        else => .unknown,
    };
}

/// Ports `llama_model_mamba2`'s `load_arch_tensors` (models/mamba2.cpp:35 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const n_group: i64 = hp.ssm_n_group;
    const dt_rank: i64 = hp.ssm_dt_rank;

    const conv_dim = d_inner + 2 * n_group * d_state;
    const d_in_proj = d_inner + conv_dim + dt_rank;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    {
        self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

        self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_not_required, exc);
        // Without an output, the input token embedding is used, duplicated
        // to allow offloading.
        if (self.output == null) {
            self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
        }
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        // Norm.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = i }, &.{ l.n_embd, d_in_proj }, 0, exc);

        layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = i }, &.{ d_conv, d_inner + 2 * n_group * d_state }, 0, exc);
        layer.ssm_conv1d_b = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "bias", .bid = i }, &.{d_inner + 2 * n_group * d_state}, 0, exc);

        layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{dt_rank}, 0, exc);

        // No "weight" suffix for these.
        layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = i }, &.{ 1, dt_rank }, 0, exc);
        layer.ssm_d = try self.createTensor(.{ .tensor = .ssm_d, .bid = i }, &.{ 1, dt_rank }, 0, exc);

        layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = i }, &.{ @divTrunc(d_inner, n_group), n_group }, 0, exc);

        // out_proj
        layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = i }, &.{ d_inner, l.n_embd }, 0, exc);
    }
}

/// Ports `llama_model_mamba2`'s `build_arch_graph` (models/mamba2.cpp:85
/// @b29c606e2): `llama_model_mamba`'s graph, which picks the Mamba-2 block
/// by the architecture.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    mamba.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
