//! K2 Horizon: a decoder whose RMS norms run over equal groups of the hidden
//! state, with a per-head RMS norm on Q and K, an optional softplus output
//! gate before the attention projection, a sigmoid-routed mixture of
//! experts with shared experts past the dense leading layers, and, on the
//! MoE layers, MoVA -- a mixture of value experts that routes each token's
//! V projection through its top value experts.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/k2-horizon.cpp` at v0.9.0 (`cd959bae9`):
//! `llama_model_k2_horizon`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph` and `graph::build_routed_value`, and the
//! file's `k2_horizon_group_rms_norm`. The class is declared in
//! `llama.cpp/src/models/models.h`.

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
const Layer = model_mod.Layer;
const Tensor = c.ggml_tensor;

// Mirrors the declaration of `llama_model_k2_horizon` (src/models/models.h:3220 @cd959bae9).

/// Ports `llama_model_k2_horizon`'s `load_arch_hparams` (models/k2-horizon.cpp:3 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    if (try meta.getKey(f32, .rope_scaling_yarn_beta_fast, false, exc)) |v| hp.yarn_beta_fast = v;
    if (try meta.getKey(f32, .rope_scaling_yarn_beta_slow, false, exc)) |v| hp.yarn_beta_slow = v;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .attention_groupnorm_groups, false, exc)) |v| hp.n_norm_groups = v;
    if (hp.n_norm_groups == 0) {
        hp.n_norm_groups = 1;
    }

    if (hp.n_expert > 0) {
        _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
        if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
        if (try meta.getKey(u32, .moe_every_n_layers, false, exc)) |v| hp.moe_every_n_layers = v;
        if (try meta.getKey(u32, .expert_shared_count, false, exc)) |v| hp.n_expert_shared = v;
        if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
        if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
        if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
        if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
        if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
            hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
        }
    }

    // MoVA.
    if (try meta.getKey(u32, .attention_value_expert_count, false, exc)) |v| hp.n_value_expert = v;
    if (try meta.getKey(u32, .attention_value_expert_used_count, false, exc)) |v| hp.n_value_expert_used = v;
    if (hp.n_value_expert > 0) {
        impl.assert(hp.n_value_expert <= hparams_mod.max_experts, "models/k2-horizon.cpp", 30, "hparams.n_value_expert <= LLAMA_MAX_EXPERTS");
        impl.assert(hp.n_value_expert_used > 0, "models/k2-horizon.cpp", 31, "hparams.n_value_expert_used > 0");
        impl.assert(hp.n_value_expert_used <= hp.n_value_expert, "models/k2-horizon.cpp", 32, "hparams.n_value_expert_used <= hparams.n_value_expert");
    } else {
        impl.assert(hp.n_value_expert_used == 0, "models/k2-horizon.cpp", 34, "hparams.n_value_expert_used == 0");
    }

    self.type = switch (hp.nLayer()) {
        28 => .@"1b",
        36 => switch (hp.n_embd) {
            2560 => .@"4b",
            4096 => .@"7b",
            else => .unknown,
        },
        48 => .@"36b",
        64 => .@"32b",
        else => .unknown,
    };
}

/// Ports `llama_model_k2_horizon`'s `load_arch_tensors` (models/k2-horizon.cpp:51 @cd959bae9).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        const is_moe_layer = l.n_expert > 0 and @as(u32, @bitCast(i)) >= hp.n_layer_dense_lead;
        const is_mova_layer = is_moe_layer and hp.n_value_expert > 0;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head }, 0, exc);
        layer.wk = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        // One norm weight per head, stored flat; viewed as {head_dim, n_head}
        // so it splits by head like Q and K.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k, l.n_head }, nr | loader.tensor_allow_reshape, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k, l.n_head_kv }, nr | loader.tensor_allow_reshape, exc);

        if (is_mova_layer) {
            const n_value_expert: i64 = hp.n_value_expert;
            layer.attn_v_gate = try self.createTensor(.{ .tensor = .attn_v_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_value_expert }, 0, exc);
            layer.attn_v_gate_b = try self.createTensor(.{ .tensor = .attn_v_gate, .suffix = "bias", .bid = i }, &.{n_value_expert}, nr, exc);
            layer.attn_v_exps = try self.createTensor(.{ .tensor = .attn_v_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_v_gqa, n_value_expert }, 0, exc);
        } else {
            layer.wv = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_v_gqa }, 0, exc);
        }

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_v * l.n_head, l.n_embd }, 0, exc);
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_v * l.n_head }, nr, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (is_moe_layer) {
            const n_ff_exp: i64 = hp.nFfExp(@intCast(i));
            if (n_ff_exp == 0) {
                return exc.throw(.runtime_error, "K2 Horizon MoE layer requires expert_feed_forward_length", .{});
            }

            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr, exc);

            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);

            if (hp.n_expert_shared > 0) {
                const n_ff_shexp: i64 = if (hp.n_ff_shexp > 0) hp.n_ff_shexp else n_ff_exp * hp.n_expert_shared;

                layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
                layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
                layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, 0, exc);
            }
        } else {
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        }
    }
}

/// Ports `llama_model_k2_horizon`'s `build_arch_graph` (models/k2-horizon.cpp:119 @cd959bae9).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports `k2_horizon_group_rms_norm` (models/k2-horizon.cpp:124 @cd959bae9): an RMS
/// norm over `n_groups` equal slices of `ne[0]`, then one full-width weight.
fn groupRmsNorm(ctx: *c.ggml_context, cur_: *Tensor, weight: ?*Tensor, n_groups: i64, eps: f32) *Tensor {
    impl.assert(n_groups > 0 and @rem(cur_.ne[0], n_groups) == 0, "models/k2-horizon.cpp", 125, "n_groups > 0 && cur->ne[0] % n_groups == 0");

    const n_embd = cur_.ne[0];
    const n_tokens = cur_.ne[1];

    var cur: *Tensor = c.ggml_reshape_3d(ctx, cur_, @divTrunc(n_embd, n_groups), n_groups, n_tokens);
    cur = c.ggml_rms_norm(ctx, cur, eps);
    cur = c.ggml_reshape_2d(ctx, cur, n_embd, n_tokens);

    return if (weight) |w| c.ggml_mul(ctx, cur, w) else cur;
}

/// Ports `llama_model_k2_horizon::graph`'s `build_routed_value`
/// (models/k2-horizon.cpp:138 @cd959bae9): MoVA, which routes each token to
/// `n_value_expert_used` value experts, V = sum_k w_k * silu(W_k x).
fn buildRoutedValue(b: *const graph.Builder, layer: *const Layer, cur: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd = cur.ne[0];
    const n_tokens = cur.ne[1];
    const n_embd_gqa: i64 = hp.nEmbdVGqa(@intCast(il));
    const n_values: i64 = hp.n_value_expert;
    const n_used: i64 = hp.n_value_expert_used;

    const logits = b.buildLoraMm(layer.attn_v_gate.?, cur, null);
    var probs: *Tensor = undefined;

    const gating: hparams_mod.ExpertGatingFunc = @fromBackingInt(hp.expert_gating_func);
    switch (gating) {
        .softmax => probs = c.ggml_soft_max(ctx, logits),
        .sigmoid => probs = c.ggml_sigmoid(ctx, logits),
        else => impl.abort("models/k2-horizon.cpp", 151, "unsupported K2 Horizon value-router gating function"),
    }

    // The bias only affects which experts are selected, not their weights.
    var selection_probs = probs;
    if (layer.attn_v_gate_b) |gate_b| {
        selection_probs = c.ggml_add(ctx, probs, gate_b);
        b.cb(selection_probs, "v_moe_probs_biased", il);
    }

    const selected_experts: *Tensor = c.ggml_argsort_top_k(ctx, selection_probs, @intCast(n_used));

    probs = c.ggml_reshape_3d(ctx, probs, 1, n_values, n_tokens);
    var weights: *Tensor = c.ggml_get_rows(ctx, probs, selected_experts);

    if (hp.expert_weights_norm) {
        weights = c.ggml_reshape_2d(ctx, weights, n_used, n_tokens);
        var weights_sum: *Tensor = c.ggml_sum_rows(ctx, weights);
        weights_sum = c.ggml_clamp(ctx, weights_sum, 6.103515625e-5, std.math.inf(f32));
        weights = c.ggml_div(ctx, weights, weights_sum);
        weights = c.ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
        b.cb(weights, "v_moe_weights_norm", il);
    }

    if (hp.expert_weights_scale != 0.0 and hp.expert_weights_scale != 1.0) {
        weights = c.ggml_scale(ctx, weights, hp.expert_weights_scale);
        b.cb(weights, "v_moe_weights_scaled", il);
    }

    b.cb(logits, "v_moe_logits", il);
    b.cb(probs, "v_moe_probs", il);
    b.cb(selected_experts.src[0], "v_moe_argsort", il);
    b.cb(selected_experts, "v_moe_topk", il);
    b.cb(weights, "v_moe_weights", il);

    var values = b.buildLoraMmId(layer.attn_v_exps.?, c.ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens), selected_experts, null);
    values = c.ggml_silu(ctx, values);
    values = c.ggml_mul(ctx, values, weights);
    b.cb(values, "v_moe_weighted", il);

    // Sum the selected experts. 3-d views of {n_embd_gqa, 1, n_tokens} keep
    // the strides of `values`, which lets the tensor-parallel backend follow
    // its split through the views; ordering the views before the adds lets
    // backends fuse the sum.
    var value_views: [hparams_mod.max_experts]?*Tensor = @splat(null);
    var i: i64 = 0;
    while (i < n_used) : (i += 1) {
        const ui: usize = @intCast(i);
        value_views[ui] = c.ggml_view_3d(ctx, values, n_embd_gqa, 1, n_tokens, values.nb[1], values.nb[2], ui * values.nb[1]);
        c.ggml_build_forward_expand(b.gf, value_views[ui]);
    }

    var value_out: *Tensor = value_views[0].?;
    i = 1;
    while (i < n_used) : (i += 1) {
        value_out = c.ggml_add(ctx, value_out, value_views[@intCast(i)]);
        c.ggml_build_forward_expand(b.gf, value_out);
    }
    if (n_used == 1) {
        value_out = c.ggml_cont(ctx, value_out);
    }
    b.cb(value_out, "Vcur_routed", il);

    return value_out;
}

/// Ports the constructor of `llama_model_k2_horizon`'s `graph`
/// (models/k2-horizon.cpp:213 @cd959bae9).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/k2-horizon.cpp", 215, "n_embd_head == hparams.n_embd_head_k()");

    const n_norm_groups: i64 = hp.n_norm_groups;

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        b.res.t_layer_inp[@intCast(il)] = inpL;

        var inpSA = inpL;

        const is_moe_layer = b.n_expert > 0 and @as(u32, @bitCast(il)) >= hp.n_layer_dense_lead;
        const is_mova_layer = is_moe_layer and hp.n_value_expert > 0;

        var cur = groupRmsNorm(ctx, inpL, layer.attn_norm, n_norm_groups, hp.f_norm_rms_eps);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const attn_inp = cur; // Kept for the output gate.

            var Qcur = b.buildLoraMm(layer.wq.?, cur, layer.wq_s);
            var Kcur = b.buildLoraMm(layer.wk.?, cur, layer.wk_s);
            var Vcur = if (is_mova_layer) buildRoutedValue(b, layer, cur, il) else b.buildLoraMm(layer.wv.?, cur, layer.wv_s);

            Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
            Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
            Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

            // A per-head RMS norm, with a separate weight for every head.
            if (layer.attn_q_norm) |w| {
                Qcur = b.buildNorm(Qcur, w, null, .rms, il);
            }
            if (layer.attn_k_norm) |w| {
                Kcur = b.buildNorm(Kcur, w, null, .rms, il);
            }

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            // With an output gate, o_proj comes after the gating.
            const gated = layer.wqkv_gate != null;

            cur = b.buildAttn(inp_attn, if (gated) null else layer.wo, if (gated) null else layer.wo_b, if (gated) null else layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);

            if (gated) {
                // A softplus with beta = ln(2): log2(1 + 2^x).
                const ln2: f32 = 0.6931471805599453;
                var gate = b.buildLoraMm(layer.wqkv_gate.?, attn_inp, layer.wqkv_gate_s);
                gate = c.ggml_scale(ctx, gate, ln2);
                gate = c.ggml_softplus(ctx, gate);
                gate = c.ggml_scale(ctx, gate, 1.4426950408889634); // 1 / ln(2)

                cur = c.ggml_mul(ctx, cur, gate);
                cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
                if (layer.wo_b) |wo_b| {
                    cur = c.ggml_add(ctx, cur, wo_b);
                }
            }
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = groupRmsNorm(ctx, ffn_inp, layer.ffn_norm, n_norm_groups, hp.f_norm_rms_eps);
        b.cb(cur, "ffn_norm", il);

        if (is_moe_layer) {
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
                .gating_op = @fromBackingInt(hp.expert_gating_func),
            }, il);

            if (layer.ffn_gate_shexp != null) {
                const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
                cur = c.ggml_add(ctx, moe_out, ffn_shexp);
            } else {
                cur = moe_out;
            }
        } else {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        }
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = groupRmsNorm(ctx, inpL, model.output_norm, n_norm_groups, hp.f_norm_rms_eps);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
