//! gpt-oss: alternating sliding-window and global attention with learned
//! attention sinks, and a mixture of experts with biased projections and
//! the clamped `swiglu_oai` activation in every layer.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/openai-moe.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_openai_moe`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_openai_moe`'s `load_arch_hparams` (models/openai-moe.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    var swa_period: u32 = 2;
    if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |v| swa_period = v;
    hp.setSwaPattern(swa_period, false);

    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    self.type = switch (hp.nLayer()) {
        24 => .@"20b",
        36 => .@"120b",
        else => .unknown,
    };
}

/// Ports `llama_model_openai_moe`'s `load_arch_tensors` (models/openai-moe.cpp:24 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const n_ff_exp: i64 = self.hparams.nFfExp(0);

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_head * l.n_rot, l.n_head_kv * l.n_rot, l.n_head_kv * l.n_rot, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_head * l.n_rot, l.n_embd }, 0, exc);

        layer.attn_sinks = try self.createTensor(.{ .tensor = .attn_sinks, .suffix = "weight", .bid = i }, &.{l.n_head}, 0, exc);

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate_inp_b = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);
        layer.ffn_gate_exps_b = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "bias", .bid = i }, &.{ n_ff_exp, l.n_expert }, 0, exc);
        layer.ffn_down_exps_b = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "bias", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_up_exps_b = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "bias", .bid = i }, &.{ n_ff_exp, l.n_expert }, 0, exc);
    }
}

/// Ports `llama_model_openai_moe`'s `build_arch_graph` (models/openai-moe.cpp:60 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_openai_moe`'s `graph`
/// (models/openai-moe.cpp:64 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKvIswa();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = inpL;

        const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
        const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

        var inpSA = inpL;

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, b.n_rot, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, layer.attn_sinks, null, 1.0 / impl.sqrtf(@floatFromInt(b.n_rot)), il);

            b.cb(cur, "attn_out", il);
        }
        if (il == b.n_layer - 1 and b.cparams.embeddings_nextn_masked) {
            // Skip computing the output of unused tokens.
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = ffn_inp;
        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        // The mixture of experts.
        cur = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .gate_inp_b = layer.ffn_gate_inp_b,
            .up_exps = layer.ffn_up_exps,
            .up_exps_b = layer.ffn_up_exps_b,
            .gate_exps = layer.ffn_gate_exps,
            .gate_exps_b = layer.ffn_gate_exps_b,
            .down_exps = layer.ffn_down_exps.?,
            .down_exps_b = layer.ffn_down_exps_b,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .swiglu_oai_moe,
            .norm_w = false,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .softmax_weight,
        }, il);
        b.cb(cur, "ffn_moe_out", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    b.res.t_h_nextn = cur;

    if (!b.cparams.embeddings_nextn_masked) {
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

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
