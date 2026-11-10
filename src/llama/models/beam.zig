//! Beam: interleaved sliding-window and NoPE full-attention layers, Q/K
//! norms, a per-element sigmoid gate on the attention output with a
//! per-head bias, norms before and after both blocks, a residual stream
//! that averages over the sublayers, a sigmoid-routed mixture of experts
//! past the dense leading layers, and a position-dependent query scale on
//! the NoPE layers.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/beam.cpp` at v0.9.0 (`cd959bae9`):
//! `llama_model_beam`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, `beam_residual_add`, and its `graph`.

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

/// Ports `llama_model_beam`'s `load_arch_hparams` (models/beam.cpp:3 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_layer_dense_lead = (try meta.getKey(u32, .leading_dense_block_count, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    hp.expert_weights_scale = (try meta.getKey(f32, .expert_weights_scale, true, exc)).?;
    hp.expert_weights_norm = (try meta.getKey(bool, .expert_weights_norm, true, exc)).?;
    hp.f_final_logit_softcapping = 0.0; // An absent key is no softcap (the default is 30).
    if (try meta.getKey(f32, .final_logit_softcapping, false, exc)) |v| hp.f_final_logit_softcapping = v;

    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }
    if (hp.n_layer_dense_lead > hp.nLayer()) {
        return exc.throw(.runtime_error, "beam: leading_dense_block_count exceeds the layer count", .{});
    }

    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.swa_type = @backingInt(hparams_mod.SwaType.standard); // Masked iff p1 - p0 >= n_swa.
    const n_pattern = (try meta.getArrN(.attention_sliding_window_pattern, true, exc)).?;
    if (hp.n_swa == 0 or n_pattern != hp.n_layer_all) {
        return exc.throw(.runtime_error, "beam: requires a positive sliding window and a sliding window pattern for each layer", .{});
    }
    _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);
    if (!hp.isSwaAny()) {
        return exc.throw(.runtime_error, "beam: sliding window pattern has no sliding window layer", .{});
    }

    // The full layers are NoPE.
    hp.rope_pattern = hp.is_swa_impl;
    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    // The NoPE query scale uses pos + 1.
    hp.f_attn_temp_scale = 0.1;
    hp.attn_temp_length = 2048;
    if (try meta.getKey(f32, .attention_temperature_scale, false, exc)) |v| hp.f_attn_temp_scale = v;
    if (try meta.getKey(u32, .attention_temperature_length, false, exc)) |v| hp.attn_temp_length = v;
    if (hp.f_attn_temp_scale == 0.0) {
        return exc.throw(.runtime_error, "beam: attention.temperature_scale must be nonzero", .{});
    }
    if (hp.attn_temp_length == 0) {
        return exc.throw(.runtime_error, "beam: attention.temperature_length must be positive", .{});
    }
    hp.n_attn_temp_floor_scale = hp.attn_temp_length;
    hp.f_attn_temp_offset = 1.0;
    hp.attn_temp_pow = true;

    self.type = .unknown;
}

/// Ports `llama_model_beam`'s `load_arch_tensors` (models/beam.cpp:59 @cd959bae9).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head }, 0, exc);
        layer.wqkv_gate_b = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "bias", .bid = i }, &.{l.n_head}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (@as(u32, @intCast(i)) < hp.n_layer_dense_lead) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        } else {
            const n_ff_exp: i64 = hp.nFfExp(@intCast(i));

            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);

            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
        }
    }
}

/// Ports `llama_model_beam`'s `build_arch_graph` (models/beam.cpp:105 @cd959bae9).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports `beam_residual_add` (models/beam.cpp:109 @cd959bae9): the running
/// average of the sublayers' outputs, `i` the sublayer's ordinal.
fn beamResidualAdd(ctx: *c.ggml_context, h: *Tensor, r: *Tensor, i: c_int) *Tensor {
    const di: f64 = @floatFromInt(i);
    const h_scale: f32 = @floatCast(1.0 / @sqrt(di));
    const r_scale: f32 = @floatCast(@sqrt((di - 1.0) / di));
    return c.ggml_add(ctx, c.ggml_scale(ctx, h, h_scale), c.ggml_scale(ctx, r, r_scale));
}

/// Ports the constructor of `llama_model_beam`'s `graph`
/// (models/beam.cpp:115 @cd959bae9).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/beam.cpp", 117, "n_embd_head == hparams.n_embd_head_k()");

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    var has_nope = false;
    {
        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            has_nope = has_nope or !hp.isSwa(@intCast(il));
        }
    }
    const inp_attn_scale: ?*Tensor = if (has_nope) b.buildInpAttnScale() else null;

    const inp_attn = b.buildAttnInpKvIswa();

    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        const is_swa = hp.isSwa(@intCast(il));

        var inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        {
            const attn_inp = cur;

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);
            b.cb(Kcur, "Kcur_normed", il);

            if (is_swa) {
                const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
                const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

                Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                b.cb(Qcur, "Qcur_rope", il);
                b.cb(Kcur, "Kcur_rope", il);
            } else {
                Qcur = c.ggml_mul(ctx, Qcur, inp_attn_scale);
                b.cb(Qcur, "Qcur_scaled", il);
            }

            // The output projection comes after the gate.
            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);

            var gate = b.buildLoraMm(layer.wqkv_gate.?, attn_inp, null);
            b.cb(gate, "attn_gate_proj", il);
            gate = c.ggml_reshape_3d(ctx, gate, n_embd_head, b.n_head, b.n_tokens);
            gate = c.ggml_add(ctx, gate, c.ggml_reshape_3d(ctx, layer.wqkv_gate_b, 1, b.n_head, 1));
            gate = c.ggml_sigmoid(ctx, gate);
            gate = c.ggml_reshape_2d(ctx, gate, n_embd_head * b.n_head, b.n_tokens);
            b.cb(gate, "attn_gate_sig", il);

            cur = c.ggml_mul(ctx, cur, gate);
            b.cb(cur, "attn_gated", il);

            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
            b.cb(cur, "attn_o_proj", il);
        }

        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const i_attn: c_int = 2 * (il + 1);
        const ffn_inp = beamResidualAdd(ctx, cur, inpSA, i_attn);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            cur = b.buildMoeFfn(cur, .{
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
            b.cb(cur, "ffn_moe_out", il);
        }

        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, il);
        b.cb(cur, "ffn_post_norm", il);

        cur = beamResidualAdd(ctx, cur, ffn_inp, i_attn + 1);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

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
