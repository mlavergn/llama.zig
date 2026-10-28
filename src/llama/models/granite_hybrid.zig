//! Granite hybrid (Granite 4): Mamba-2 blocks interleaved with attention
//! layers, each followed by a SwiGLU feed-forward or a mixture of experts
//! with an optional shared expert, with Granite's residual and logit
//! scales. A layer is recurrent when it has no KV heads.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/granite-hybrid.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_granite_hybrid`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph` with
//! `build_attention_layer` and `build_layer_ffn`.

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

/// Ports `llama_model_granite_hybrid`'s `load_arch_hparams`
/// (models/granite-hybrid.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(f32, .logit_scale, false, exc)) |v| hp.f_logit_scale = v;
    if (try meta.getKey(f32, .residual_scale, false, exc)) |v| hp.f_residual_scale = v;
    if (try meta.getKey(f32, .embedding_scale, false, exc)) |v| hp.f_embedding_scale = v;
    if (try meta.getKey(f32, .attention_scale, false, exc)) |v| hp.f_attention_scale = v;

    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

    // Granite uses rope_finetuned as a switch for RoPE, so it defaults to
    // true.
    var rope_finetuned = true;
    if (try meta.getKey(bool, .rope_scaling_finetuned, false, exc)) |v| rope_finetuned = v;
    hp.rope_finetuned = rope_finetuned; // needed for round trip save
    hp.rope_pattern = @splat(@intFromBool(rope_finetuned));

    // A layer is recurrent if and only if its n_head_kv is 0.
    var i: u32 = 0;
    while (i < hp.nLayer()) : (i += 1) {
        hp.is_recr_impl[i] = @intFromBool(hp.nHeadKv(i) == 0);
    }

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.n_embd) {
        768 => .@"350m",
        1536 => if (hp.nFf(0) == 512) .@"7b_a1b" else .@"1b",
        2048, 2560 => .@"3b",
        4096 => .@"32b",
        else => .unknown,
    };

    // For Granite MoE Shared.
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
}

/// Ports `llama_model_granite_hybrid`'s `load_arch_tensors`
/// (models/granite-hybrid.cpp:41 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    // The Mamba-2 mixer's SSM parameters.
    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const n_ssm_head: i64 = hp.ssm_dt_rank;
    const n_group: i64 = hp.ssm_n_group;
    const d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head;

    // Only an expansion factor of 2 is supported for now.
    impl.assert(2 * l.n_embd == d_inner, "models/granite-hybrid.cpp", 54, "2 * n_embd == d_inner");

    // Embeddings.
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
        const layer = &self.layers[@intCast(i)];
        const rope_dup: u32 = if (i != 0) loader.tensor_duplicated else 0;

        // Norm.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (hp.isRecr(@intCast(i))) {
            // The SSM layers.
            layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = i }, &.{ l.n_embd, d_in_proj }, 0, exc);

            layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = i }, &.{ d_conv, d_inner + 2 * n_group * d_state }, 0, exc);
            layer.ssm_conv1d_b = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "bias", .bid = i }, &.{d_inner + 2 * n_group * d_state}, nr, exc);

            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{n_ssm_head}, 0, exc);

            // No "weight" suffix for these.
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = i }, &.{ 1, n_ssm_head }, 0, exc);
            layer.ssm_d = try self.createTensor(.{ .tensor = .ssm_d, .bid = i }, &.{ 1, n_ssm_head }, 0, exc);

            layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = i }, &.{ @divTrunc(d_inner, n_group), n_group }, 0, exc);

            // out_proj
            layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = i }, &.{ d_inner, l.n_embd }, 0, exc);
        } else {
            // The attention layers, with optional bias.
            const n_head_i: i64 = hp.nHead(@intCast(i));
            const n_embd_k_gqa_i: i64 = hp.nEmbdKGqa(@intCast(i));
            const n_embd_v_gqa_i: i64 = hp.nEmbdVGqa(@intCast(i));
            try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * n_head_i, n_embd_k_gqa_i, n_embd_v_gqa_i, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * n_head_i, l.n_embd }, 0, exc);
            layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
        }

        // The feed-forward, with optional biases.
        if (l.n_expert > 0) {
            // The MoE FFN.
            layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, nr, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, 0, exc);

            // For Granite MoE Shared.
            if (hp.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, 0, exc);
                layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, 0, exc);
                layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ hp.n_ff_shexp, l.n_embd }, 0, exc);
            }
        } else {
            layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_gate_b = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "bias", .bid = i }, &.{l.n_ff}, nr, exc);
            layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
            layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{l.n_ff}, nr, exc);
        }
    }
}

/// Ports `llama_model_granite_hybrid`'s `build_arch_graph`
/// (models/granite-hybrid.cpp:131 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_granite_hybrid`'s `graph`
/// (models/granite-hybrid.cpp:135 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/granite-hybrid.cpp", 138, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp = b.buildInpMemHybrid();

    const inp_out_ids = b.buildInpOutIds();

    // The positions, when RoPE is on.
    var inp_pos: ?*Tensor = null;
    if (hp.hasRope(0)) {
        inp_pos = b.buildInpPos();
    }

    var cur: *Tensor = undefined;
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // Norm.
        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        if (hp.isRecr(@intCast(il))) {
            // An SSM layer.
            cur = mamba_base.buildMamba2Layer(b, inp.getRecr(), cur, model, il);
        } else {
            // An attention layer.
            cur = buildAttentionLayer(model, b, cur, inp_pos, inp.getAttn(), n_embd_head, il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        // FFN.
        cur = buildLayerFfn(model, b, cur, inpSA, il);

        // The next layer's input.
        inpL = cur;
    }

    cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    // Granite scales the logits.
    if (hp.f_logit_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_logit_scale);
    }
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_granite_hybrid::graph`'s `build_attention_layer`
/// (models/granite-hybrid.cpp:202 @5266f24da).
fn buildAttentionLayer(model: *const Model, b: *const graph.Builder, cur_: *Tensor, inp_pos: ?*Tensor, inp_attn: *graph.InpAttnKv, n_embd_head: i64, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];

    const qkv = b.buildQkv(layer, cur_, n_embd_head, hp.nHead(@intCast(il)), hp.nHeadKv(@intCast(il)), il);
    var Qcur = qkv.q;
    var Kcur = qkv.k;
    const Vcur = qkv.v;

    if (hp.hasRope(@intCast(il))) {
        const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));
        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    }

    b.cb(Qcur, "Qcur", il);
    b.cb(Kcur, "Kcur", il);
    b.cb(Vcur, "Vcur", il);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;
    const cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(cur, "attn_out", il);
    return cur;
}

/// Ports `llama_model_granite_hybrid::graph`'s `build_layer_ffn`
/// (models/granite-hybrid.cpp:232 @5266f24da).
fn buildLayerFfn(model: *const Model, b: *const graph.Builder, cur_: *Tensor, inpSA: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    var cur = cur_;

    // Granite scales the residual.
    if (hp.f_residual_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, hp.f_residual_scale);
    }
    const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "ffn_inp", il);

    // The feed-forward network, not MoE.
    if (layer.ffn_gate_inp == null) {
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);
    } else {
        // The MoE branch.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        const moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .silu,
            .norm_w = true,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .softmax,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        // For Granite MoE Shared.
        if (hp.n_ff_shexp > 0) {
            const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
            b.cb(ffn_shexp, "ffn_shexp", il);

            cur = c.ggml_add(ctx, moe_out, ffn_shexp);
            b.cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    }

    // Granite scales the residual.
    if (hp.f_residual_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, hp.f_residual_scale);
    }
    cur = c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "ffn_out", il);

    cur = b.buildCvec(cur, il);
    b.cb(cur, "l_out", il);

    return cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
