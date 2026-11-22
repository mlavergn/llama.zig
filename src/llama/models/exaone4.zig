//! EXAONE 4.0: post-norm decoder blocks -- RMS norms after attention and
//! after the feed-forward -- with RMS-normed queries and keys, and in the
//! 32B model three sliding-window layers with RoPE for every global layer
//! without it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/exaone4.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_exaone4`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph<iswa>`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_exaone4`'s `load_arch_hparams` (models/exaone4.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    if (hp.nLayer() == 64) { // 32B
        hp.swa_type = @backingInt(llama.hparams.SwaType.standard);
        hp.n_swa = 4096;
        try self.loadSwaPattern(ml, 4, false, exc);
        hp.rope_pattern = hp.is_swa_impl;

        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    }

    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| hp.n_swa = v;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        30 => .@"1_2b",
        64 => .@"32b",
        else => .unknown,
    };
}

/// Ports `llama_model_exaone4`'s `load_arch_tensors` (models/exaone4.cpp:25 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const nr = llama.loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer_all) : (i += 1) {
        const is_nextn = i >= l.n_layer;
        var flags: u32 = 0;
        if (is_nextn) {
            // NextN/MTP layers are preserved in GGUF but are not executed yet.
            flags |= llama.loader.tensor_skip;
        }

        const layer = &self.layers[@intCast(i)];

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, flags, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, flags, exc);

        if (!is_nextn) {
            const rope_dup: u32 = if (i != 0) llama.loader.tensor_duplicated else 0;
            layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
        }

        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, flags, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, flags, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, flags, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        if (is_nextn) {
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags | nr, exc);
        }
    }
}

/// Ports `llama_model_exaone4`'s `get_rope_params` (models/exaone4.cpp:74 @b1b770d90):
/// the RoPE of a layer, for the graph and the K-shift alike.
///
/// Parameters:
/// - `model`: the model.
/// - `cparams`: the context's parameters.
/// - `il`: the layer.
///
/// Return: the layer's RoPE parameters.
pub fn getRopeParams(model: *const llama.model.Model, cparams: *const llama.cparams.Cparams, il: u32) llama.model.RopeParams {
    var res = model.getRopeParamsBase(cparams, il);

    // The SWA layers use the global base and scale.
    res.freq_base = cparams.rope_freq_base;
    res.freq_scale = cparams.rope_freq_scale;

    return res;
}

/// Ports `llama_model_exaone4`'s `build_arch_graph` (models/exaone4.cpp:84 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    if (b.hparams.swa_type == @backingInt(llama.hparams.SwaType.standard)) {
        buildGraph(true, model, b);
    } else {
        buildGraph(false, model, b);
    }
}

/// Ports the constructor of `llama_model_exaone4`'s `graph<iswa>`
/// (models/exaone4.cpp:93 @b1b770d90).
fn buildGraph(comptime iswa: bool, model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadK(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadV(0), "models/exaone4.cpp", 97, "n_embd_head == hparams.n_embd_head_v()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/exaone4.cpp", 98, "n_embd_head == n_rot");

    var inpL: *llama.impl.llama_c.ggml_tensor = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        var cur = inpL;

        // Self-attention.
        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);
            b.cb(Kcur, "Kcur_normed", il);

            // RoPE only on the layers `rope_pattern` names: the sliding-window
            // ones, or every layer of a model without them.
            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);
            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
            b.cb(cur, "attn_out", il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.llama_c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        const ffn_inp: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        cur = b.buildFfn(ffn_inp, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, -1);
        b.cb(cur, "ffn_post_norm", -1);

        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);

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
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
