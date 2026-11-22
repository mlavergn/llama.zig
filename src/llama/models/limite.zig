//! Limite: a pre-norm transformer with value embeddings, exclusive self
//! attention (XSA), per-head output gates, learned residual scales and
//! dense (MUDD) cross-layer connections, on sliding-window layers with RoPE
//! and full-attention layers without positional encoding.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/limite.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_limite`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_limite`'s `load_arch_hparams` (models/limite.cpp:7 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_attention_scale = (try meta.getKey(f32, .attention_scale, true, exc)).?;
    if (try meta.getKey(f32, .attention_xsa_eps, false, exc)) |v| hp.f_xsa_eps = v;
    _ = try meta.getArr(f32, .final_logit_sigmoid_capping, &hp.final_logit_sigmoid_capping, false, exc);

    hp.swa_type = @backingInt(llama.hparams.SwaType.standard);
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    try self.loadSwaPattern(ml, 4, false, exc);
    hp.rope_pattern = hp.is_swa_impl;

    @memset(&hp.mudd_tap_idx, -1);
    if (try meta.getKey(u32, .mudd_tap_count, false, exc)) |v| hp.n_mudd_taps = v;
    if (hp.n_mudd_taps > 0) {
        hp.n_mudd_ff = (try meta.getKey(u32, .mudd_feed_forward_length, true, exc)).?;
        _ = try meta.getArr(i32, .mudd_tap_indices, &hp.mudd_tap_idx, true, exc);
    }

    self.type = switch (hp.nLayer()) {
        48 => .@"1b",
        else => .unknown,
    };
}

/// Ports `llama_model_limite`'s `load_arch_tensors` (models/limite.cpp:31 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, ml: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    _ = ml;
    const hp = &self.hparams;
    const n_layer: i32 = @intCast(hp.nLayer());
    const n_head: i64 = hp.nHead(0);
    const n_head_kv: i64 = hp.nHeadKv(0);
    const n_embd: i64 = hp.n_embd;
    const n_embd_k_gqa: i64 = hp.nEmbdKGqa(0);
    const n_embd_v_gqa: i64 = hp.nEmbdVGqa(0);
    const n_embd_head_k: i64 = hp.nEmbdHeadK(0);
    const n_ff: i64 = hp.nFf(0);
    const n_vocab: i64 = self.vocab.nTokens();
    const nr = llama.loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, n_vocab }, 0, exc);
    self.value_embd = try self.createTensor(.{ .tensor = .value_embd, .suffix = "weight" }, &.{ n_embd_v_gqa, n_vocab }, 0, exc);

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ n_embd, n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    const n_taps: i64 = hp.n_mudd_taps;
    const n_mudd: i64 = hp.n_mudd_ff;

    var i: i32 = 0;
    while (i < n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        try self.createTensorQkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_head_k * n_head, n_embd }, 0, exc);

        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ n_embd, n_head }, nr, exc);
        layer.attn_ve_gate = try self.createTensor(.{ .tensor = .attn_ve_gate, .suffix = "weight", .bid = i }, &.{ n_embd, n_head_kv }, nr, exc);
        layer.attn_xsa_alpha = try self.createTensor(.{ .tensor = .attn_xsa_alpha, .suffix = "weight", .bid = i }, &.{n_head}, nr, exc);

        layer.attn_resid_scale = try self.createTensor(.{ .tensor = .attn_resid_scale, .suffix = "weight", .bid = i }, &.{1}, 0, exc);
        layer.ffn_resid_scale = try self.createTensor(.{ .tensor = .ffn_resid_scale, .suffix = "weight", .bid = i }, &.{1}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ n_ff, n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ n_embd, n_ff }, 0, exc);

        if (n_taps > 0 and hp.mudd_tap_idx[@intCast(i * n_taps)] >= 0) {
            // Rows [0, n_taps) mix the attention input, rows [n_taps, 2*n_taps) mix the residual.
            layer.mudd_down = try self.createTensor(.{ .tensor = .mudd_down, .suffix = "weight", .bid = i }, &.{ n_embd, n_mudd }, 0, exc);
            layer.mudd_up = try self.createTensor(.{ .tensor = .mudd_up, .suffix = "weight", .bid = i }, &.{ n_mudd, 2 * n_taps }, 0, exc);
            layer.mudd_up_b = try self.createTensor(.{ .tensor = .mudd_up, .suffix = "bias", .bid = i }, &.{2 * n_taps}, 0, exc);
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_limite`'s `get_rope_params` (models/limite.cpp:72
/// @b1b770d90): the SWA layers use the global base and scale.
///
/// Parameters:
/// - `model`: the model.
/// - `cparams`: the context's parameters.
/// - `il`: the layer.
///
/// Return: the layer's RoPE.
pub fn getRopeParams(model: *const llama.model.Model, cparams: *const llama.cparams.Cparams, il: u32) llama.model.RopeParams {
    var res = model.getRopeParamsBase(cparams, il);

    res.freq_base = cparams.rope_freq_base;
    res.freq_scale = cparams.rope_freq_scale;

    return res;
}

/// Ports `llama_model_limite`'s `build_arch_graph` (models/limite.cpp:82 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, params: *const llama.graph.Params) void {
    _ = params;
    buildGraph(model, b);
}

/// `mix`, a lambda of the `graph` constructor (models/limite.cpp:86
/// @b1b770d90): the taps weighted by one half of the MUDD weights, the
/// fused hyper-connection kernel when there are exactly four.
fn mix(b: *const llama.graph.Builder, stack: *llama.impl.llama_c.ggml_tensor, w: *llama.impl.llama_c.ggml_tensor, n_taps: i64, offset: usize) *llama.impl.llama_c.ggml_tensor {
    const ctx = b.ctx0;
    if (n_taps == 4) {
        return llama.impl.llama_c.ggml_dsv4_hc_pre(ctx, stack, llama.impl.llama_c.ggml_cont(ctx, llama.impl.llama_c.ggml_view_2d(ctx, w, n_taps, b.n_tokens, w.nb[1], offset)));
    }
    const wj = llama.impl.llama_c.ggml_view_3d(ctx, w, 1, n_taps, b.n_tokens, w.nb[0], w.nb[1], offset);
    const m = llama.impl.llama_c.ggml_cont(ctx, llama.impl.llama_c.ggml_permute(ctx, llama.impl.llama_c.ggml_mul(ctx, stack, wj), 1, 0, 2, 3));
    return llama.impl.llama_c.ggml_reshape_2d(ctx, llama.impl.llama_c.ggml_sum_rows(ctx, m), b.n_embd, b.n_tokens);
}

/// Ports the constructor of `llama_model_limite`'s `graph` (models/limite.cpp:86 @b1b770d90).
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadK(0);
    const n_taps: i64 = hp.n_mudd_taps;
    const n_head = b.n_head;
    const n_head_kv = b.n_head_kv;
    const n_tokens = b.n_tokens;
    const n_embd = b.n_embd;

    llama.impl.assert(n_embd_head == hp.nEmbdHeadV(0), "models/limite.cpp", 90, "n_embd_head == hparams.n_embd_head_v()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // Value embeddings are looked up by token id; embeddings input is not supported.
    var inp_ve = llama.impl.llama_c.ggml_get_rows(ctx, model.value_embd, b.res.t_inp_tokens);
    inp_ve = llama.impl.llama_c.ggml_reshape_3d(ctx, inp_ve, n_embd_head, n_head_kv, n_tokens);
    b.cb(inp_ve, "inp_ve", -1);

    inpL = b.buildNorm(inpL, null, null, .rms, -1);
    b.cb(inpL, "inp_norm", -1);

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKvIswa();

    const inp_out_ids = b.buildInpOutIds();

    // hist[il] is the input of layer il, read by the dense connections.
    var hist: [llama.hparams.max_layers]?*llama.impl.llama_c.ggml_tensor = @splat(null);

    var cur: *llama.impl.llama_c.ggml_tensor = undefined;
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        hist[@intCast(il)] = inpL;

        var resid = inpL;

        if (layer.mudd_up != null) {
            const taps = hp.mudd_tap_idx[@as(usize, @intCast(il)) * @as(usize, @intCast(n_taps)) ..];

            var w = b.buildNorm(inpL, null, null, .rms, il);
            w = llama.impl.llama_c.ggml_gelu_erf(ctx, b.buildLoraMm(layer.mudd_down.?, w, null));
            w = llama.impl.llama_c.ggml_add(ctx, b.buildLoraMm(layer.mudd_up.?, w, null), layer.mudd_up_b);
            b.cb(w, "mudd_w", il);

            var stack: ?*llama.impl.llama_c.ggml_tensor = null;
            var j: usize = 0;
            while (j < n_taps) : (j += 1) {
                llama.impl.assert(taps[j] >= 0 and taps[j] <= il, "models/limite.cpp", 131, "taps[j] >= 0 && taps[j] <= il");
                const h = llama.impl.llama_c.ggml_reshape_3d(ctx, hist[@intCast(taps[j])], n_embd, 1, n_tokens);
                stack = if (stack) |s| llama.impl.llama_c.ggml_concat(ctx, s, h, 1) else h;
            }

            cur = mix(b, stack.?, w, n_taps, 0);
            resid = mix(b, stack.?, w, n_taps, @as(usize, @intCast(n_taps)) * w.nb[0]);
            b.cb(cur, "mudd_attn", il);
            b.cb(resid, "mudd_resid", il);
        } else {
            cur = inpL;
        }

        const attn_in = b.buildNorm(cur, null, null, .rms, il);
        b.cb(attn_in, "attn_norm", il);

        // Self-attention.
        {
            const qkv = b.buildQkv(layer, attn_in, n_embd_head, n_head, n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            var Vcur = qkv.v;

            if (layer.attn_ve_gate) |ve_gate| {
                var gate = llama.impl.llama_c.ggml_sigmoid(ctx, b.buildLoraMm(ve_gate, attn_in, null));
                gate = llama.impl.llama_c.ggml_reshape_3d(ctx, gate, 1, n_head_kv, n_tokens);

                Vcur = llama.impl.llama_c.ggml_add(ctx, Vcur, llama.impl.llama_c.ggml_mul(ctx, inp_ve, gate));
                b.cb(Vcur, "Vcur_ve", il);
            }

            Qcur = llama.impl.llama_c.ggml_rms_norm(ctx, Qcur, hp.f_norm_rms_eps);
            Kcur = llama.impl.llama_c.ggml_rms_norm(ctx, Kcur, hp.f_norm_rms_eps);
            b.cb(Qcur, "Qcur_normed", il);
            b.cb(Kcur, "Kcur_normed", il);

            // The full attention layers use no positional encoding.
            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);
            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);

            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, hp.f_attention_scale, il);
            b.cb(cur, "attn_raw", il);

            if (layer.attn_xsa_alpha) |xsa_alpha| {
                // Remove from each head output its projection on the token's own value.
                const n_group = @divTrunc(n_head, n_head_kv);

                const y = llama.impl.llama_c.ggml_reshape_4d(ctx, cur, n_embd_head, n_group, n_head_kv, n_tokens);
                var vn = llama.impl.llama_c.ggml_l2_norm(ctx, Vcur, hp.f_xsa_eps);
                vn = llama.impl.llama_c.ggml_reshape_4d(ctx, vn, n_embd_head, 1, n_head_kv, n_tokens);
                vn = llama.impl.llama_c.ggml_repeat(ctx, vn, y);

                var proj = llama.impl.llama_c.ggml_sum_rows(ctx, llama.impl.llama_c.ggml_mul(ctx, y, vn));
                proj = llama.impl.llama_c.ggml_mul(ctx, proj, llama.impl.llama_c.ggml_reshape_4d(ctx, xsa_alpha, 1, n_group, n_head_kv, 1));

                cur = llama.impl.llama_c.ggml_sub(ctx, y, llama.impl.llama_c.ggml_mul(ctx, vn, proj));
                b.cb(cur, "attn_xsa", il);
            }

            cur = llama.impl.llama_c.ggml_reshape_3d(ctx, cur, n_embd_head, n_head, n_tokens);

            if (layer.wqkv_gate) |wqkv_gate| {
                const gate = llama.impl.llama_c.ggml_sigmoid(ctx, b.buildLoraMm(wqkv_gate, attn_in, null));
                cur = llama.impl.llama_c.ggml_mul(ctx, cur, llama.impl.llama_c.ggml_reshape_3d(ctx, gate, 1, n_head, n_tokens));
                b.cb(cur, "attn_gated", il);
            }

            cur = llama.impl.llama_c.ggml_reshape_2d(ctx, cur, n_embd_head * n_head, n_tokens);
            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
            b.cb(cur, "attn_out", il);
        }

        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            resid = llama.impl.llama_c.ggml_get_rows(ctx, resid, inp_out_ids);
        }

        cur = llama.impl.llama_c.ggml_add(ctx, llama.impl.llama_c.ggml_mul(ctx, resid, layer.attn_resid_scale), cur);
        b.cb(cur, "attn_resid", il);

        const ffn_inp = cur;

        cur = b.buildNorm(cur, null, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = llama.impl.llama_c.ggml_add(ctx, llama.impl.llama_c.ggml_mul(ctx, ffn_inp, layer.ffn_resid_scale), cur);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = b.buildNorm(inpL, null, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    // scale * sigmoid((x + shift) / temperature)
    const cap = &hp.final_logit_sigmoid_capping;
    if (cap[0] != 0.0 and cap[2] != 0.0) {
        cur = llama.impl.llama_c.ggml_scale_bias(ctx, cur, 1.0 / cap[2], cap[1] / cap[2]);
        cur = llama.impl.llama_c.ggml_sigmoid(ctx, cur);
        cur = llama.impl.llama_c.ggml_scale(ctx, cur, cap[0]);
    }

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
