//! Jamba: Mamba-1 blocks interleaved with attention layers that have no
//! positional encoding, each followed by a SwiGLU feed-forward or a mixture
//! of experts. A layer is a Mamba layer when it has no KV heads.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/jamba.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_model_jamba`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

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

/// Ports `llama_model_jamba`'s `load_arch_hparams` (models/jamba.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    var i: u32 = 0;
    while (i < hp.nLayer()) : (i += 1) {
        hp.is_recr_impl[i] = @intFromBool(hp.nHeadKv(i) == 0);
    }

    // TODO: Jamba layers are a bit heterogeneous, so naming this is hard.
    // 12 layers is 900M 8x???M, 32 is 51B 16x?B; both are unknown.
    self.type = .unknown;
}

/// Ports `llama_model_jamba`'s `load_arch_tensors` (models/jamba.cpp:23 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const dt_rank: i64 = hp.ssm_dt_rank;

    // Only an expansion factor of 2 is supported for now.
    impl.assert(2 * l.n_embd == d_inner, "models/jamba.cpp", 32, "2 * n_embd == d_inner");

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    {
        self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

        self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
        // Without an output, the input token embedding is used, duplicated
        // to allow offloading.
        if (self.output == null) {
            self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
        }
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const n_head_kv: i64 = hp.nHeadKv(@intCast(i));
        const n_embd_gqa: i64 = hp.nEmbdVGqa(@intCast(i));

        const layer = &self.layers[@intCast(i)];

        // Norm.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (n_head_kv == 0) {
            // A Mamba layer.
            layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = i }, &.{ l.n_embd, 2 * d_inner }, 0, exc);

            layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = i }, &.{ d_conv, d_inner }, 0, exc);
            layer.ssm_conv1d_b = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "bias", .bid = i }, &.{d_inner}, 0, exc);

            layer.ssm_x = try self.createTensor(.{ .tensor = .ssm_x, .suffix = "weight", .bid = i }, &.{ d_inner, dt_rank + 2 * d_state }, 0, exc);

            layer.ssm_dt_norm = try self.createTensor(.{ .tensor = .ssm_dt_norm, .suffix = "weight", .bid = i }, &.{dt_rank}, 0, exc);

            layer.ssm_dt = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "weight", .bid = i }, &.{ dt_rank, d_inner }, 0, exc);
            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{d_inner}, 0, exc);

            layer.ssm_b_norm = try self.createTensor(.{ .tensor = .ssm_b_norm, .suffix = "weight", .bid = i }, &.{d_state}, 0, exc);
            layer.ssm_c_norm = try self.createTensor(.{ .tensor = .ssm_c_norm, .suffix = "weight", .bid = i }, &.{d_state}, 0, exc);

            // No "weight" suffix for these.
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = i }, &.{ d_state, d_inner }, 0, exc);
            layer.ssm_d = try self.createTensor(.{ .tensor = .ssm_d, .bid = i }, &.{d_inner}, 0, exc);

            // out_proj
            layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = i }, &.{ d_inner, l.n_embd }, 0, exc);
        } else {
            // An attention layer.
            try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, n_embd_gqa, n_embd_gqa, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        }

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, nr, exc);

        if (layer.ffn_gate_inp != null) {
            // MoE.
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, 0, exc);
        } else {
            // A feed-forward network, not MoE.
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        }
    }
}

/// Ports `llama_model_jamba`'s `build_arch_graph` (models/jamba.cpp:104 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_jamba`'s `graph` (models/jamba.cpp:108 @b29c606e2).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    // {n_embd, n_tokens}
    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_hybrid = b.buildInpMemHybrid();

    const inp_out_ids = b.buildInpOutIds();

    var cur: *Tensor = undefined;
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const n_head_kv: i64 = hp.nHeadKv(@intCast(il));

        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        if (n_head_kv == 0) {
            cur = mamba_base.buildMambaLayer(b, inp_hybrid.getRecr(), cur, model, il);
        } else {
            // Attention.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, n_head_kv, il);

            // No RoPE :)
            cur = b.buildAttn(inp_hybrid.getAttn(), layer.wo, null, layer.wo_s, qkv.q, qkv.k, qkv.v, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }
        // The residual. The C++ names `cur` here, not `ffn_inp`.
        const ffn_inp: *Tensor = c.ggml_add(ctx, inpL, cur);
        b.cb(cur, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // The feed-forward network.
        if (layer.ffn_gate_inp == null) {
            // FFN.
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The MoE branch.
            cur = b.buildMoeFfn(cur, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = layer.ffn_gate_exps,
                .down_exps = layer.ffn_down_exps.?,
                .n_expert = b.n_expert,
                .n_expert_used = b.n_expert_used,
                .type_op = .silu,
                .norm_w = false,
                .w_scale = hp.expert_weights_scale,
                .gating_op = .softmax,
            }, il);
            b.cb(cur, "ffn_moe_out", il);
        }
        // The residual.
        cur = c.ggml_add(ctx, ffn_inp, cur);

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

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
