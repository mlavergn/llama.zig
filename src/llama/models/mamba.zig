//! Mamba: a stack of selective state-space blocks with no attention, each
//! behind an RMS norm and a residual.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mamba.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_mamba`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`, which Mamba-2 shares.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_mamba`'s `load_arch_hparams` (models/mamba.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    if (try meta.getKey(bool, .ssm_dt_b_c_rms, false, exc)) |v| hp.ssm_dt_b_c_rms = v;

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
            else => .unknown,
        },
        else => .unknown,
    };
}

/// Ports `llama_model_mamba`'s `load_arch_tensors` (models/mamba.cpp:34 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const dt_rank: i64 = hp.ssm_dt_rank;

    // Only an expansion factor of 2 is supported for now.
    if (2 * l.n_embd != d_inner) {
        return exc.throw(.runtime_error, "only an expansion factor of 2 is supported for now", .{});
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_not_required, exc);
    // Without an output, the input token embedding is used, duplicated to
    // allow offloading.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        // Norm.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = i }, &.{ l.n_embd, 2 * d_inner }, 0, exc);

        layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = i }, &.{ d_conv, d_inner }, 0, exc);
        layer.ssm_conv1d_b = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "bias", .bid = i }, &.{d_inner}, 0, exc);

        layer.ssm_x = try self.createTensor(.{ .tensor = .ssm_x, .suffix = "weight", .bid = i }, &.{ d_inner, dt_rank + 2 * d_state }, 0, exc);

        layer.ssm_dt = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "weight", .bid = i }, &.{ dt_rank, d_inner }, 0, exc);
        layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{d_inner}, 0, exc);

        // No "weight" suffix for these.
        layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = i }, &.{ d_state, d_inner }, 0, exc);
        layer.ssm_d = try self.createTensor(.{ .tensor = .ssm_d, .bid = i }, &.{d_inner}, 0, exc);

        // out_proj
        layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = i }, &.{ d_inner, l.n_embd }, 0, exc);
    }
}

/// Ports `llama_model_mamba`'s `build_arch_graph` (models/mamba.cpp:83 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_mamba`'s `graph` (models/mamba.cpp:87
/// @b1b770d90): Mamba-1 or Mamba-2 blocks by the model's architecture.
pub fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;

    // {n_embd, n_tokens}
    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const rs_inp = b.buildRsInp();

    const inp_out_ids = b.buildInpOutIds();

    var cur: *llama.impl.llama_c.ggml_tensor = undefined;
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        // Norm.
        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        if (model.arch == .mamba2) {
            cur = llama.models.mamba_base.buildMamba2Layer(b, rs_inp, cur, model, il);
        } else {
            cur = llama.models.mamba_base.buildMambaLayer(b, rs_inp, cur, model, il);
        }

        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = llama.impl.llama_c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }

        // The residual.
        cur = llama.impl.llama_c.ggml_add(ctx, cur, inpL);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    // The final RMS norm.
    cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
