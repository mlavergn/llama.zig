//! Laguna (poolside): a sigmoid-routed mixture of experts with a
//! score-correction bias and one shared expert, a softplus gate on the
//! attention output, Q and K norms, and RoPE per layer type -- YaRN on the
//! full-attention layers, plain RoPE on the sliding-window ones. XS.2 is
//! hybrid full and sliding-window with a gate per head; M.1 is all full
//! attention with a gate per element. The mixture and gate follow AFMoE's.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/laguna.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_laguna`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_laguna`'s `load_arch_hparams` (models/laguna.cpp:9 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_layer_dense_lead = (try meta.getKey(u32, .leading_dense_block_count, true, exc)).?;
    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;

    // Laguna ships one shared expert and stores its size directly, since
    // the routed and shared experts may differ. The count is not in the
    // config: 1 unless the key says otherwise.
    hp.n_expert_shared = 1;
    if (try meta.getKey(u32, .expert_shared_count, false, exc)) |v| hp.n_expert_shared = v;
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    if (hp.n_ff_shexp == 0) {
        // Weightless fixtures (test-llama-archs) omit the key; a nonzero
        // size keeps the shared expert built. Real GGUFs carry the exact
        // value.
        hp.n_ff_shexp = hp.n_ff_exp * hp.n_expert_shared;
    }

    // Sliding-window attention is optional. XS.2 is hybrid (full, then
    // three sliding, repeating with period 4); M.1 has no sliding window.
    // Without a window, swa_type stays NONE and the per-layer-type RoPE is
    // skipped.
    hp.n_swa = 0;
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| hp.n_swa = v;
    if (hp.n_swa > 0) {
        hp.swa_type = @backingInt(hparams_mod.SwaType.standard);

        var swa_period: u32 = 4;
        if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |v| swa_period = v;
        hp.setSwaPattern(swa_period, true); // XS.2: full at il % 4 == 0.

        // RoPE per layer type: the full layers take YaRN with theta 500000
        // over 64 dimensions, the SWA layers default RoPE with theta 10000
        // over 128. The base load_hparams has read the non-SWA fields; the
        // SWA mirrors are read here.
        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = 1.0; // Plain RoPE, not the full layers' 1/factor.
        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
        if (try meta.getKey(u32, .rope_dimension_count_swa, false, exc)) |v| hp.n_rot_swa = v;
    }

    // A sigmoid when the key is absent, as in the HF reference.
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }

    self.type = switch (hp.nLayer()) {
        40 => .@"30b_a3b", // Laguna-XS.2
        48 => .@"118b_a8b", // Laguna-S.2
        70 => .@"230b_a10b", // Laguna-M.1
        else => .unknown,
    };
}

/// Ports `llama_model_laguna`'s `load_arch_tensors` (models/laguna.cpp:67 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        // Tied embeddings.
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    const n_ff_exp: i64 = hp.n_ff_exp;
    const n_ff_shexp: i64 = hp.n_ff_shexp;

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        // The head count per layer: Laguna's full and SWA layers differ
        // (48 against 64 in XS.2). The KV head count is uniform.
        const n_head_il: i64 = hp.nHead(@intCast(i));
        const n_head_kv_il: i64 = hp.nHeadKv(@intCast(i));
        const n_embd_q_il = l.n_embd_head_k * n_head_il;
        const n_embd_k_il = l.n_embd_head_k * n_head_kv_il;
        const n_embd_v_il = l.n_embd_head_v * n_head_kv_il;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, n_embd_q_il, n_embd_k_il, n_embd_v_il, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_q_il, l.n_embd }, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        // The attention output gate: per head in XS.2 (g_proj to n_head,
        // one scalar per head broadcast over head_dim), per element in M.1
        // (g_proj to n_head*head_dim, as in afmoe). The stored tensor's
        // width says which, and must be exactly one of the two; the graph
        // makes the same check. Weightless fixtures (test-llama-archs) have
        // no gate tensor and take the per-head layout, so its reshape path
        // still runs.
        const n_gate_per_head = n_head_il;
        const n_gate_per_elem = l.n_embd_head_k * n_head_il;
        const gate_name = (loader.Tn{ .tensor = .attn_gate, .suffix = "weight", .bid = i }).str();
        const gate_meta = ml.getTensorMeta(gate_name.slice());
        var n_gate_out: i64 = undefined;
        if (gate_meta) |gm| {
            n_gate_out = gm.ne[1];
            if (n_gate_out != n_gate_per_head and n_gate_out != n_gate_per_elem) {
                c.ggml_abort("models/laguna.cpp", 119, "Laguna: unexpected attention gate width %lld at layer %d (expected %lld per-head or %lld per-element)", @as(c_longlong, n_gate_out), @as(c_int, i), @as(c_longlong, n_gate_per_head), @as(c_longlong, n_gate_per_elem));
            }
        } else {
            n_gate_out = n_gate_per_head;
        }
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_gate_out }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (@as(u32, @intCast(i)) >= hp.n_layer_dense_lead) {
            // A mixture-of-experts layer.
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);

            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);

            // The always-on shared expert.
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, 0, exc);
        } else {
            // A dense layer, the leading n_layer_dense_lead ones.
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        }
    }
}

/// Ports `llama_model_laguna`'s `build_arch_graph` (models/laguna.cpp:150 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    // XS.2 is hybrid SWA and takes the interleaved-SWA KV input; M.1 is
    // all full attention and takes the plain one. The C++ picks the input,
    // and with it the `build_attn` overload, at run time.
    const has_swa = @as(hparams_mod.SwaType, @fromBackingInt(@intCast(model.hparams.swa_type))) != .none;
    if (has_swa) buildGraph(true, model, b) else buildGraph(false, model, b);
}

/// Ports the constructor of `llama_model_laguna`'s `graph`
/// (models/laguna.cpp:154 @c1d0e7a00).
fn buildGraph(comptime has_swa: bool, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/laguna.cpp", 156, "n_embd_head == hparams.n_embd_head_k()");

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);
    // No MuP embedding scale: laguna omits it, where afmoe scales by
    // sqrt(hidden).

    const inp_pos = b.buildInpPos();
    const inp_attn = if (has_swa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();
    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const is_swa_il = hp.isSwa(@intCast(il));
        const n_head_il: i64 = hp.nHead(@intCast(il));
        const n_head_kv_il: i64 = hp.nHeadKv(@intCast(il));

        // RoPE per layer type. The SWA layers run plain RoPE, without YaRN:
        // their YaRN ext and beta parameters are zero.
        const n_rot_l: c_int = if (is_swa_il) @intCast(hp.n_rot_swa) else @intCast(b.n_rot);
        const freq_base_l = if (is_swa_il) hp.rope_freq_base_train_swa else b.freq_base;
        const freq_scale_l = if (is_swa_il) hp.rope_freq_scale_train_swa else b.freq_scale;
        const ext_factor_l: f32 = if (is_swa_il) 0.0 else b.ext_factor;
        // The framework handles YaRN's magnitude scaling (mscale):
        // llama_context divides cparams.yarn_attn_factor by
        // 1 + 0.1*ln(factor) beforehand, cancelling ggml rope_yarn's own
        // mscale *= 1 + 0.1*ln(1/freq_scale). attn_factor passes straight
        // through, as in every other architecture; the SWA layers run plain
        // RoPE (ext_factor 0, no mscale) and force 1.0.
        const attn_factor_l: f32 = if (is_swa_il) 1.0 else b.attn_factor;
        const beta_fast_l: f32 = if (is_swa_il) 0.0 else b.beta_fast;
        const beta_slow_l: f32 = if (is_swa_il) 0.0 else b.beta_slow;
        const n_ctx_orig_l: c_int = if (is_swa_il) @intCast(hp.n_ctx_train) else b.n_ctx_orig;

        var inpSA = inpL;

        // The pre-norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const attn_inp = cur; // Kept for the gate projection.

            const qkv = b.buildQkv(layer, cur, n_embd_head, n_head_il, n_head_kv_il, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            // g_proj on the hidden state before attention, as in the HF
            // reference: the gate reads the same input as Q, K and V, not
            // the attention output.
            var gate: *Tensor = b.buildLoraMm(layer.wqkv_gate.?, attn_inp, null);
            b.cb(gate, "attn_gate_proj", il);

            // The Q and K RMS norms at head_dim (Qwen3 style).
            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);
            b.cb(Kcur, "Kcur_normed", il);

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, n_rot_l, b.rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, n_rot_l, b.rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            b.cb(Qcur, "Qcur_rope", il);
            b.cb(Kcur, "Kcur_rope", il);

            // The output projection waits until after the gate.
            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);

            // The softplus output gate; the unary kernel computes softplus
            // in fp32 and casts back. Its two shapes follow g_proj's output
            // width, as at load time: per head in XS.2, gate
            // [n_head_il, n_tokens] reshaped to [1, n_head_il, n_tokens]
            // and broadcast over head_dim against cur
            // [head_dim, n_head, T]; per element in M.1, gate
            // [n_head_il*head_dim, n_tokens] spans the whole attention
            // output, a direct ggml_mul.
            gate = c.ggml_softplus(ctx, gate);
            b.cb(gate, "attn_gate_softplus", il);

            const n_tokens = cur.ne[1];
            if (layer.wqkv_gate.?.ne[1] == n_head_il) {
                cur = c.ggml_reshape_3d(ctx, cur, n_embd_head, n_head_il, n_tokens);
                gate = c.ggml_reshape_3d(ctx, gate, 1, n_head_il, n_tokens);
                cur = c.ggml_mul(ctx, cur, gate);
                cur = c.ggml_reshape_2d(ctx, cur, n_embd_head * n_head_il, n_tokens);
            } else {
                cur = c.ggml_mul(ctx, cur, gate);
            }
            b.cb(cur, "attn_gated", il);

            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
            b.cb(cur, "attn_o_proj", il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The pre-norm only, no norm after the attention.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (@as(u32, @intCast(il)) >= hp.n_layer_dense_lead) {
            // The mixture of experts: sigmoid routing, the score-correction
            // bias, the sum norm and routed_scaling_factor, all in
            // build_moe_ffn.
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

            // The always-on shared expert, summed in parallel.
            const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
            b.cb(ffn_shexp, "ffn_shexp", il);

            cur = c.ggml_add(ctx, moe_out, ffn_shexp);
            b.cb(cur, "ffn_out", il);
        } else {
            // The dense feed-forward of the leading n_layer_dense_lead
            // layers (XS.2: 1, M.1: 3).
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        }

        // No norm after the feed-forward.
        cur = c.ggml_add(ctx, cur, ffn_inp);
        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = inpL;
    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, null);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
