//! Grok: pre-norm decoder blocks with RoPE, soft-capped attention logits,
//! RMS norms after the attention and after the feed-forward, and a
//! GELU mixture of experts, averaged with a dense GELU feed-forward where
//! the model has one; the logits are scaled and optionally soft-capped.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/grok.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_grok`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_grok`'s `load_arch_hparams` (models/grok.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    // Defaults for old GGUFs.
    hp.yarn_beta_fast = 8.0;
    hp.f_logit_scale = 0.5773502691896257;
    hp.f_embedding_scale = 78.38367176906169;
    hp.f_attn_out_scale = 0.08838834764831845;
    hp.f_attn_logit_softcapping = 30.0;
    hp.f_router_logit_softcapping = 30.0;
    // Grok-1 has no final logit soft-capping.
    hp.f_final_logit_softcapping = 0.0;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .expert_feed_forward_length, false, exc)) |v| hp.n_ff_exp = v;
    if (try meta.getKey(f32, .logit_scale, false, exc)) |v| hp.f_logit_scale = v;
    if (try meta.getKey(f32, .embedding_scale, false, exc)) |v| hp.f_embedding_scale = v;
    if (try meta.getKey(f32, .attention_output_scale, false, exc)) |v| hp.f_attn_out_scale = v;
    if (try meta.getKey(f32, .attn_logit_softcapping, false, exc)) |v| hp.f_attn_logit_softcapping = v;
    if (try meta.getKey(f32, .router_logit_softcapping, false, exc)) |v| hp.f_router_logit_softcapping = v;
    if (try meta.getKey(f32, .final_logit_softcapping, false, exc)) |v| hp.f_final_logit_softcapping = v;

    if (try meta.getKey(u32, .attention_temperature_length, false, exc)) |v| hp.attn_temp_length = v;
    if (try meta.getKey(f32, .rope_scaling_yarn_ext_factor, false, exc)) |v| hp.yarn_ext_factor = v;
    if (try meta.getKey(f32, .rope_scaling_yarn_attn_factor, false, exc)) |v| hp.yarn_attn_factor = v;
    if (try meta.getKey(f32, .rope_scaling_yarn_beta_fast, false, exc)) |v| hp.yarn_beta_fast = v;
    if (try meta.getKey(f32, .rope_scaling_yarn_beta_slow, false, exc)) |v| hp.yarn_beta_slow = v;

    self.type = switch (hp.nLayer()) {
        64 => .@"314b",
        else => .unknown,
    };
}

/// Ports `llama_model_grok`'s `load_arch_tensors` (models/grok.cpp:35 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    if (l.n_expert == 0) {
        return exc.throw(.runtime_error, "{s} model cannot have zero experts", .{self.archName()});
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    // Grok-1's n_ff_exp is n_ff (the C++ leaves `/ n_expert_used` commented out).
    const n_ff_exp: i64 = if (hp.n_ff_exp != 0) hp.n_ff_exp else l.n_ff;
    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, l.n_embd_gqa, l.n_embd_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);

        layer.attn_out_norm = try self.createTensor(.{ .tensor = .attn_out_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, nr, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, nr, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .layer_out_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        if (layer.ffn_post_norm == null) {
            layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        }
    }
}

/// Ports `llama_model_grok`'s `build_arch_graph` (models/grok.cpp:82 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_grok`'s `graph`
/// (models/grok.cpp:86 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/grok.cpp", 89, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/grok.cpp", 90, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0, il);
        }
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        cur = b.buildNorm(cur, layer.attn_out_norm, null, .rms, il);
        b.cb(cur, "attn_out_norm", il);

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // The mixture of experts.
        const moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .gelu,
            .norm_w = true,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .softmax,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        if (layer.ffn_up != null) {
            const ffn_out = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .gelu, .par, il);
            b.cb(ffn_out, "ffn_out", il);

            // `std::sqrt(2) / 2`, a double, narrowed to the float argument.
            const inv_sqrt2: f32 = @floatCast(@sqrt(@as(f64, 2.0)) / 2.0);
            cur = c.ggml_scale(ctx, c.ggml_add(ctx, ffn_out, moe_out), inv_sqrt2);
            b.cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, il);
        b.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    cur = c.ggml_scale(ctx, cur, hp.f_logit_scale);

    // Final logit soft-capping.
    if (hp.f_final_logit_softcapping != 0.0) {
        cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
        cur = c.ggml_tanh(ctx, cur);
        cur = c.ggml_scale(ctx, cur, hp.f_final_logit_softcapping);
    }
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
