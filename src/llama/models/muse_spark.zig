//! Muse Spark: interleaved sliding-window RoPE and NoPE full-attention
//! layers, Q/K norms, a weightless per-head norm and a sigmoid gate on the
//! attention output, a norm-preserving residual with per-channel alpha and
//! beta, and a mixture of experts in every layer that the full stream
//! routes and that runs in a latent space.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/muse-spark.cpp` at v0.9.0
//! (`cd959bae9`): `llama_model_muse_spark`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`.

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

/// Ports `llama_model_muse_spark`'s `load_arch_hparams` (models/muse-spark.cpp:3 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_post_norm_eps = (try meta.getKey(f32, .post_norm_eps, true, exc)).?;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.rope_freq_base_train_swa = (try meta.getKey(f32, .rope_freq_base_swa, true, exc)).?;
    hp.f_final_logit_softcapping = (try meta.getKey(f32, .final_logit_softcapping, true, exc)).?;
    hp.f_logit_scale = (try meta.getKey(f32, .logit_scale, true, exc)).?;
    hp.moe_latent_size = (try meta.getKey(u32, .moe_latent_size, true, exc)).?;
    hp.expert_weights_scale = (try meta.getKey(f32, .expert_weights_scale, true, exc)).?;
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;
    hp.expert_weights_norm = (try meta.getKey(bool, .expert_weights_norm, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, false, exc);

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);

    self.type = switch (hp.nLayer()) {
        62 => .@"816b_a42b",
        else => .unknown,
    };
}

/// Ports `llama_model_muse_spark`'s `load_arch_tensors` (models/muse-spark.cpp:25 @cd959bae9).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const n_embd_moe: i64 = hp.moe_latent_size;
    const n_ff_exp: i64 = hp.nFfExp(0);

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        // The pre-attention norm. All block norms hold a delta folded at
        // conversion time.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // The Q, K, V and O projections.
        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        // The QK norm. Its weights are synthesized at conversion to absorb
        // the metap scale.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        // The attention output gate: sigmoid(gate) * attn_out before o_proj.
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head }, 0, exc);

        // The norm-preserving residual: alpha scales the stream, beta the
        // branch.
        layer.attn_resid_alpha = try self.createTensor(.{ .tensor = .attn_resid_alpha, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_resid_beta = try self.createTensor(.{ .tensor = .attn_resid_beta, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_resid_alpha = try self.createTensor(.{ .tensor = .ffn_resid_alpha, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_resid_beta = try self.createTensor(.{ .tensor = .ffn_resid_beta, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // Every layer routes. The router reads the full stream, the experts
        // run in a latent space entered by `ffn_latent_down` and left by
        // `ffn_latent_up`.
        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);

        layer.ffn_latent_down = try self.createTensor(.{ .tensor = .ffn_latent_down, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_moe }, 0, exc);
        layer.ffn_latent_up = try self.createTensor(.{ .tensor = .ffn_latent_up, .suffix = "weight", .bid = i }, &.{ n_embd_moe, l.n_embd }, 0, exc);

        layer.ffn_pre_exp_norm = try self.createTensor(.{ .tensor = .ffn_pre_exp_norm, .suffix = "weight", .bid = i }, &.{n_embd_moe}, 0, exc);
        layer.ffn_post_exp_norm = try self.createTensor(.{ .tensor = .ffn_post_exp_norm, .suffix = "weight", .bid = i }, &.{n_embd_moe}, 0, exc);

        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ n_embd_moe, n_ff_exp, l.n_expert }, 0, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ n_embd_moe, n_ff_exp, l.n_expert }, 0, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, n_embd_moe, l.n_expert }, 0, exc);
    }
}

/// Ports `llama_model_muse_spark`'s `build_arch_graph` (models/muse-spark.cpp:236 @cd959bae9).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_muse_spark`'s `graph`
/// (models/muse-spark.cpp:78 @cd959bae9).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/muse-spark.cpp", 81, "n_embd_head == hparams.n_embd_head_k()");

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);
    inpL = b.buildNorm(inpL, null, null, .rms, -1);
    b.cb(inpL, "embd_norm", -1);

    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = inpL;

        const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
        const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

        var inpSA = inpL;

        // RoPE runs on the SWA layers, NoPE on the global ones.
        const use_rope = hp.isSwa(@intCast(il));

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            var gate = b.buildLoraMm(layer.wqkv_gate.?, cur, null);
            b.cb(gate, "attn_gate_proj", il);

            // `attn_q_norm` carries the metap scale, `attn_k_norm` is the
            // identity.
            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);
            b.cb(Kcur, "Kcur_normed", il);

            if (use_rope) {
                Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                b.cb(Qcur, "Qcur_rope", il);

                Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                b.cb(Kcur, "Kcur_rope", il);
            }

            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);

            // A weightless norm on the attention output, per head.
            cur = c.ggml_reshape_3d(ctx, cur, n_embd_head, b.n_head, b.n_tokens);
            cur = c.ggml_rms_norm(ctx, cur, hp.f_norm_rms_eps);
            cur = c.ggml_reshape_2d(ctx, cur, n_embd_head * b.n_head, b.n_tokens);
            b.cb(cur, "attn_o_norm", il);

            gate = c.ggml_sigmoid(ctx, gate);
            b.cb(gate, "attn_gate_sig", il);
            cur = c.ggml_mul(ctx, cur, gate);
            b.cb(cur, "attn_gated", il);

            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
            b.cb(cur, "attn_o_proj", il);
        }

        // The post-attention norm holds no weight.
        cur = c.ggml_rms_norm(ctx, cur, hp.f_post_norm_eps);
        b.cb(cur, "attn_post_norm", il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, c.ggml_mul(ctx, inpSA, layer.attn_resid_alpha), c.ggml_mul(ctx, cur, layer.attn_resid_beta));
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // Enter the latent space, run the experts, leave it.
        var moe_inp = b.buildLoraMm(layer.ffn_latent_down.?, cur, null);
        b.cb(moe_inp, "ffn_latent_down", il);

        moe_inp = b.buildNorm(moe_inp, layer.ffn_pre_exp_norm, null, .rms, il);
        b.cb(moe_inp, "ffn_pre_exp_norm", il);

        // The router reads the full stream, not the latent one.
        const router_logits = b.buildLoraMm(layer.ffn_gate_inp.?, cur, null);
        b.cb(router_logits, "ffn_moe_logits", il);

        const moe_out = b.buildMoeFfn(moe_inp, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = layer.ffn_exp_probs_b,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .silu,
            .norm_w = hp.expert_weights_norm,
            .w_scale = hp.expert_weights_scale,
            .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
            .probs_in = router_logits,
            .post_exp_norm = layer.ffn_post_exp_norm,
            .post_exp_norm_eps = hp.f_post_norm_eps,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        cur = b.buildLoraMm(layer.ffn_latent_up.?, moe_out, null);
        b.cb(cur, "ffn_latent_up", il);

        // The post-FFN norm scales the input, then normalizes.
        cur = c.ggml_mul(ctx, cur, layer.ffn_post_norm);
        cur = c.ggml_rms_norm(ctx, cur, hp.f_post_norm_eps);
        b.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx, c.ggml_mul(ctx, ffn_inp, layer.ffn_resid_alpha), c.ggml_mul(ctx, cur, layer.ffn_resid_beta));
        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    cur = c.ggml_scale(ctx, cur, hp.f_logit_scale);

    if (hp.f_final_logit_softcapping != 0.0) {
        cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
        cur = c.ggml_tanh(ctx, cur);
        cur = c.ggml_scale(ctx, cur, hp.f_final_logit_softcapping);
    }

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
