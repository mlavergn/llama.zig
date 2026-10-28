//! Muse Glimmer: sandwich-normed decoder blocks with an RMS-normed input
//! embedding, RMS-normed queries and keys, a sigmoid gate on the attention
//! output before its projection, RoPE on the sliding-window layers and none
//! on the global ones, and scaled, optionally soft-capped logits.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/muse-glimmer.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_muse_glimmer`'s `load_arch_hparams`,
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

/// Ports `llama_model_muse_glimmer`'s `load_arch_hparams` (src/models/muse-glimmer.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    if (try meta.getKey(f32, .final_logit_softcapping, false, exc)) |v| hp.f_final_logit_softcapping = v;
    hp.f_logit_scale = (try meta.getKey(f32, .logit_scale, true, exc)).?;

    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |swa_period| {
        hp.setSwaPattern(swa_period, false);
    } else {
        _ = try meta.getKeyOrArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, hp.nLayer(), true, exc);
    }

    self.type = switch (hp.nLayer()) {
        52 => .@"30b",
        else => .unknown,
    };
}

/// Ports `llama_model_muse_glimmer`'s `load_arch_tensors` (src/models/muse-glimmer.cpp:26 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        // The pre- and post-attention norms; Muse Glimmer's `weight + 1` is
        // applied at conversion time.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // The Q, K, V and O projections.
        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        // The QK norm, whose weights are synthesized at conversion time to
        // absorb `qk_scale_factor`.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        // The attention output gate, sigmoid(gate) * attn_out before o_proj,
        // as in AFMoE.
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head }, 0, exc);

        // The pre- and post-FFN norms; FFN_PRE_NORM is an alias of
        // LLM_TENSOR_FFN_NORM.
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // A dense feed-forward; unlike AFMoE, no mixture of experts.
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_muse_glimmer`'s `build_arch_graph` (src/models/muse-glimmer.cpp:206 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_muse_glimmer`'s `graph`
/// (src/models/muse-glimmer.cpp:62 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/muse-glimmer.cpp", 65, "n_embd_head == hparams.n_embd_head_k()");

    // The post-attention and post-FFN norms' epsilon, not f_norm_rms_eps.
    const post_norm_eps: f32 = 1e-8;

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
        // The per-layer residual, exposed for speculative drafts (see
        // LLM_KV_TARGET_LAYERS).
        b.res.t_layer_inp[@intCast(il)] = inpL;

        const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
        const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

        var inpSA = inpL;

        // RoPE runs on the sliding-window layers, NoPE on the full ones.
        const use_rope = hp.isSwa(@intCast(il));

        // The pre-attention norm, its weight + 1 folded in at conversion.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention, with the output gate around the attention, as in
        // AFMoE.
        {
            // The input, saved for the gate.
            const attn_inp = cur;

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            // gate = wqkv_gate @ attn_inp, from the pre-attention hidden
            // state.
            var gate = b.buildLoraMm(layer.wqkv_gate.?, attn_inp, null);
            b.cb(gate, "attn_gate_proj", il);

            // The QK norm. attn_q_norm's weight was synthesized at
            // conversion to broadcast qk_scale_factor across head_dim;
            // attn_k_norm's is identity (ones).
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

            // The attention. wo is deferred: the gate goes between attn_out
            // and o_proj.
            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);

            gate = c.ggml_sigmoid(ctx, gate);
            b.cb(gate, "attn_gate_sig", il);
            cur = c.ggml_mul(ctx, cur, gate);
            b.cb(cur, "attn_gated", il);

            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
            b.cb(cur, "attn_o_proj", il);
        }

        cur = c.ggml_rms_norm(ctx, cur, post_norm_eps);
        cur = c.ggml_mul(ctx, cur, layer.attn_post_norm);
        b.cb(cur, "attn_post_norm", il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The pre-FFN norm.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // A dense SwiGLU feed-forward.
        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_rms_norm(ctx, cur, post_norm_eps);
        cur = c.ggml_mul(ctx, cur, layer.ffn_post_norm);
        b.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = inpL;

    // The final norm.
    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head, then the output multiplier.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    cur = c.ggml_scale(ctx, cur, hp.f_logit_scale);

    // The final logits' tanh soft-cap, from Gemma 3.
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
