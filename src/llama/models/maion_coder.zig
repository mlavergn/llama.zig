//! Maion Coder: interleaved sliding-window and full attention, RoPE on the
//! sliding-window layers only, RMS-normed Q and K, norms before and after
//! both the attention and the feed-forward, and every `n_moe_layer_step`th
//! layer a softmax-routed mixture of experts that runs in a latent space
//! (a down projection before the experts, an up projection after) with the
//! router reading the full hidden state.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/maion-coder.cpp` at v0.8.0
//! (`8642541b2`): `llama_model_maion_coder`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`. The class is
//! declared in `llama.cpp/src/models/models.h`.

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

// Mirrors the declaration of `llama_model_maion_coder` (src/models/models.h:224 @8642541b2).

/// Ports `llama_model_maion_coder`'s `load_arch_hparams` (models/maion-coder.cpp:3 @8642541b2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_moe_layer_step = (try meta.getKey(u32, .interleave_moe_layer_step, true, exc)).?;
    hp.moe_latent_size = (try meta.getKey(u32, .moe_latent_size, true, exc)).?;

    if (hp.n_moe_layer_step == 0 or hp.moe_latent_size == 0 or hp.n_expert == 0) {
        return exc.throw(.runtime_error, "maion-coder requires a positive MoE layer step, latent size and expert count", .{});
    }

    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    const n_swa_layers = (try meta.getArrN(.attention_sliding_window_pattern, true, exc)).?;
    if (hp.n_swa == 0 or n_swa_layers != hp.n_layer_all) {
        return exc.throw(.runtime_error, "maion-coder requires a positive sliding window and a sliding window pattern for each layer", .{});
    }
    _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);
    hp.rope_pattern = hp.is_swa_impl;
    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;

    self.type = .unknown;
}

/// Ports `llama_model_maion_coder`'s `load_arch_tensors` (models/maion-coder.cpp:28 @8642541b2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const moe_n_embd: i64 = hp.moe_latent_size;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const is_moe_layer = @rem(@as(u32, @bitCast(i + 1)), hp.n_moe_layer_step) == 0;
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (is_moe_layer) {
            const n_ff_exp: i64 = hp.nFfExp(@intCast(i));
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_latent_down = try self.createTensor(.{ .tensor = .ffn_latent_down, .suffix = "weight", .bid = i }, &.{ l.n_embd, moe_n_embd }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ moe_n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ moe_n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, moe_n_embd, l.n_expert }, 0, exc);
            layer.ffn_latent_up = try self.createTensor(.{ .tensor = .ffn_latent_up, .suffix = "weight", .bid = i }, &.{ moe_n_embd, l.n_embd }, 0, exc);
        } else {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        }
    }
}

/// Ports `llama_model_maion_coder`'s `build_arch_graph` (models/maion-coder.cpp:67 @8642541b2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_maion_coder`'s `graph`
/// (models/maion-coder.cpp:71 @8642541b2).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadK(0);

    impl.assert(n_embd_head == hp.nEmbdHeadV(0), "models/maion-coder.cpp", 74, "n_embd_head == hparams.n_embd_head_v()");
    impl.assert(n_embd_head == b.n_rot, "models/maion-coder.cpp", 75, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        // The per-layer input hidden state, for speculative drafts (DFlash,
        // eagle3) to read.
        b.res.t_layer_inp[@intCast(il)] = inpL;

        var inpSA = inpL;
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
        var Qcur = qkv.q;
        var Kcur = qkv.k;
        const Vcur = qkv.v;
        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);

        // The full-attention layers have no RoPE.
        if (hp.isSwa(@intCast(il))) {
            const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
            const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));
            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        }

        cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);

        if (layer.ffn_gate_inp == null) {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        } else {
            const router_logits = b.buildLoraMm(layer.ffn_gate_inp.?, cur, null);
            const inp_latent = b.buildLoraMm(layer.ffn_latent_down.?, cur, null);
            cur = b.buildMoeFfn(inp_latent, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = layer.ffn_gate_exps,
                .down_exps = layer.ffn_down_exps.?,
                .exp_probs_b = null,
                .n_expert = b.n_expert,
                .n_expert_used = hp.nExpertUsed(@intCast(il)),
                .type_op = .silu,
                .norm_w = true,
                .w_scale = 0.0,
                .gating_op = .softmax,
                .probs_in = router_logits,
            }, il);
            cur = b.buildLoraMm(layer.ffn_latent_up.?, cur, null);
        }

        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, il);
        cur = c.ggml_add(ctx, cur, ffn_inp);
        cur = b.buildCvec(cur, il);
        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
