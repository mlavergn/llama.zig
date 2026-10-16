//! AFMoE: MuP-scaled embeddings, interleaved sliding-window attention with
//! RMS-normed Q and K and a sigmoid gate on the attention output before
//! its projection, norms before and after both the attention and the
//! feed-forward, and a sigmoid-routed mixture of experts with a shared
//! expert past the dense leading layers.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/afmoe.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_afmoe`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_afmoe`'s `load_arch_hparams` (models/afmoe.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| hp.n_swa = v;

    // Interleaved sliding-window attention; the pattern is three sliding,
    // one full (global_attn_every_n_layers = 4).
    if (hp.n_swa > 0) {
        hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
        var swa_period: u32 = 4;
        if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |v| swa_period = v;
        hp.setSwaPattern(swa_period, false);

        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    } else {
        hp.swa_type = @backingInt(hparams_mod.SwaType.none);
    }

    // A sigmoid when unset.
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }

    self.type = switch (hp.nLayer()) {
        56 => .@"6b",
        32 => .@"26b",
        else => .unknown,
    };
}

/// Ports `llama_model_afmoe`'s `load_arch_tensors` (models/afmoe.cpp:40 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const n_expert_shared: i64 = hp.n_expert_shared;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    const n_ff_exp: i64 = hp.n_ff_exp;

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        // Attention norms, before and after.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // The attention projections.
        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        // The Q and K norms.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        // The attention gate.
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head }, 0, exc);

        // Feed-forward norms, before and after.
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (@as(u32, @intCast(i)) >= hp.n_layer_dense_lead) {
            // The mixture-of-experts layers.
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);

            // The grouped expert weights.
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

            // The shared expert.
            if (n_expert_shared > 0) {
                const n_ff_shexp = n_ff_exp * n_expert_shared;
                layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
                layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, 0, exc);
                layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
            }
        } else {
            // The dense layers.
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        }
    }
}

/// Ports `llama_model_afmoe`'s `build_arch_graph` (models/afmoe.cpp:105 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_afmoe`'s `graph`
/// (models/afmoe.cpp:109 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/afmoe.cpp", 111, "n_embd_head == hparams.n_embd_head_k()");

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

    // MuP scaling, the embeddings by sqrt(hidden_size): with mup_enabled
    // and hidden_size 1024, a scale of 32.
    inpL = c.ggml_scale(ctx, inpL, impl.sqrtf(@floatFromInt(b.n_embd)));
    b.cb(inpL, "inp_embd_scaled", -1);

    // The positions.
    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
        const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

        var inpSA = inpL;

        // This overlaps with the SWA layers in current models, so the
        // per-layer RoPE base and scale may be superfluous.
        const use_rope = hp.n_no_rope_layer_step > 0 and
            @rem(@as(u32, @intCast(il + 1)), hp.n_no_rope_layer_step) != 0;

        // The attention norm before.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const attn_inp = cur; // Kept for the gate.

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            // The gate, from the input.
            var gate: *Tensor = b.buildLoraMm(layer.wqkv_gate.?, attn_inp, null);
            b.cb(gate, "attn_gate_proj", il);

            // The Q and K norms.
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

            // The output projection comes after the gate.
            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);

            // The attention gate, attn_out * sigmoid(gate), before o_proj.
            gate = c.ggml_sigmoid(ctx, gate);
            b.cb(gate, "attn_gate_sig", il);
            cur = c.ggml_mul(ctx, cur, gate);
            b.cb(cur, "attn_gated", il);

            // Then the output projection.
            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
            b.cb(cur, "attn_o_proj", il);
        }

        // The attention norm after.
        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward norm before.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // A mixture of experts or a dense feed-forward.
        if (@as(u32, @intCast(il)) >= hp.n_layer_dense_lead) {
            // Sigmoid routing, normalized (route_norm = True) and scaled
            // (route_scale = 2.826).
            const moe_out = b.buildMoeFfn(cur, .{
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
            }, il);
            b.cb(moe_out, "ffn_moe_out", il);

            // The shared expert.
            if (hp.n_expert_shared > 0) {
                const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
                b.cb(ffn_shexp, "ffn_shexp", il);

                cur = c.ggml_add(ctx, moe_out, ffn_shexp);
                b.cb(cur, "ffn_out", il);
            } else {
                cur = moe_out;
            }
        } else {
            // A dense layer.
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        }

        // The feed-forward norm after.
        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, il);
        b.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
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
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
