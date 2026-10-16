//! Falcon-H1: every layer runs attention and a Mamba-2 block in parallel
//! on the same normed input and sums them, then a SwiGLU feed-forward.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/falcon-h1.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_falcon_h1`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const mamba_base = @import("mamba_base.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_falcon_h1`'s `load_arch_hparams` (models/falcon-h1.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    // Common parameters.
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    // SSM parameters.
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

    // Every layer is recurrent, and every layer attends too.
    hp.is_recr_impl = @splat(1);

    self.type = switch (hp.nLayer()) {
        36 => .@"0_5b",
        24 => .@"1_5b",
        66 => .@"1b",
        32 => .@"3b",
        44 => .@"7b",
        72 => .@"34b",
        else => .unknown,
    };
}

/// Ports `llama_model_falcon_h1`'s `load_arch_tensors` (models/falcon-h1.cpp:34 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    // Common.
    const hidden_size: i64 = hp.n_embd;

    // The Mamba-2 mixer's SSM parameters.
    const ssm_conv_kernel_size: i64 = hp.ssm_d_conv;
    const ssm_n_groups: i64 = hp.ssm_n_group;
    const ssm_state_size: i64 = hp.ssm_d_state;
    const ssm_intermediate_size: i64 = hp.ssm_d_inner; // TODO expand
    const ssm_num_heads: i64 = hp.ssm_dt_rank;
    const ssm_conv_dim = ssm_intermediate_size + 2 * ssm_n_groups * ssm_state_size;
    const ssm_projection_size = ssm_intermediate_size + ssm_conv_dim + ssm_num_heads;

    // Attention parameters.
    const attn_num_attention_head: i64 = hp.nHead(0);
    const attn_num_key_value_head: i64 = hp.nHeadKv(0);

    // FFN parameters.
    const ffn_intermediate_size: i64 = hp.nFf(0);

    // Embeddings.
    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ hidden_size, l.n_vocab }, 0, exc);

    // Output.
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ hidden_size, l.n_vocab }, nr, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{hidden_size}, 0, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ hidden_size, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        // The SSM layers.
        // ssm in
        layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = i }, &.{ hidden_size, ssm_projection_size }, 0, exc);
        // ssm 1d conv
        layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = i }, &.{ ssm_conv_kernel_size, ssm_conv_dim }, 0, exc);
        layer.ssm_conv1d_b = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "bias", .bid = i }, &.{ssm_conv_dim}, nr, exc);
        // ssm_dt
        layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{ssm_num_heads}, 0, exc);
        // No "weight" suffix for these.
        layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = i }, &.{ 1, ssm_num_heads }, 0, exc);
        layer.ssm_d = try self.createTensor(.{ .tensor = .ssm_d, .bid = i }, &.{ 1, ssm_num_heads }, 0, exc);
        // ssm_norm
        layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = i }, &.{ @divTrunc(ssm_intermediate_size, ssm_n_groups), ssm_n_groups }, nr, exc);
        // out_proj
        layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = i }, &.{ ssm_intermediate_size, hidden_size }, 0, exc);

        // The attention layers, with optional bias.
        try self.createTensorQkv(layer, i, hidden_size, l.n_embd_head_k * attn_num_attention_head, attn_num_key_value_head * l.n_embd_head_k, attn_num_key_value_head * l.n_embd_head_v, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * attn_num_attention_head, hidden_size }, 0, exc);
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{hidden_size}, nr, exc);
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{hidden_size}, 0, exc);

        // The feed-forward, with optional biases. Its norm has no suffix.
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .bid = i }, &.{hidden_size}, 0, exc);
        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | (if (i != 0) loader.tensor_duplicated else 0), exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ hidden_size, ffn_intermediate_size }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ ffn_intermediate_size, hidden_size }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ hidden_size, ffn_intermediate_size }, 0, exc);

        layer.ffn_gate_b = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "bias", .bid = i }, &.{ffn_intermediate_size}, nr, exc);
        layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{hidden_size}, nr, exc);
        layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{ffn_intermediate_size}, nr, exc);
    }
}

/// Ports `llama_model_falcon_h1`'s `build_arch_graph` (models/falcon-h1.cpp:108 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_falcon_h1`'s `graph`
/// (models/falcon-h1.cpp:112 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    // The inputs of the recurrent memory and the KV cache.
    const inp = b.buildInpMemHybrid();

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    const inp_out_ids = b.buildInpOutIds();

    var cur: *Tensor = undefined;
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
        var Qcur = qkv.q;
        var Kcur = qkv.k;
        const Vcur = qkv.v;

        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), hp.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), hp.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        b.cb(Qcur, "Qcur-post-rope", il);
        b.cb(Kcur, "Kcur-post-rope", il);
        b.cb(Vcur, "Vcur-post-rope", il);

        const attn_out = b.buildAttn(inp.getAttn(), layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
        b.cb(attn_out, "attn_out", il);

        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        // The Mamba-2 layer.
        b.cb(cur, "ssm_in", il);

        const ssm_out = mamba_base.buildMamba2Layer(b, inp.getRecr(), cur, model, il);
        b.cb(ssm_out, "ssm_out", il);

        // Aggregation.
        cur = c.ggml_add(ctx, attn_out, ssm_out);
        inpSA = c.ggml_add(ctx, cur, inpSA);
        b.cb(cur, "layer_out", il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp = inpSA;
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx, cur, inpSA);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
