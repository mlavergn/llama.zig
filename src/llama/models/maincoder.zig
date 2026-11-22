//! Maincoder: pre-norm decoder blocks with RoPE, Q and K RMS-normed per
//! head after rotation, and a SwiGLU feed-forward.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/maincoder.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_maincoder`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_maincoder`'s `load_arch_hparams` (models/maincoder.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try ml.meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        32 => .@"1b",
        else => .unknown,
    };
}

/// Ports `llama_model_maincoder`'s `load_arch_tensors` (models/maincoder.cpp:12 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const nr = llama.loader.tensor_not_required;
    const dup = llama.loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_maincoder`'s `get_rope_params` (models/maincoder.cpp:43
/// @b1b770d90): a K-shift has to undo the K norm applied after RoPE.
///
/// Parameters:
/// - `model`: the model.
/// - `cparams`: the context's parameters.
/// - `il`: the layer.
///
/// Return: the layer's RoPE.
pub fn getRopeParams(model: *const llama.model.Model, cparams: *const llama.cparams.Cparams, il: u32) llama.model.RopeParams {
    var res = model.getRopeParamsBase(cparams, il);

    res.k_norm = model.layers[il].attn_k_norm;

    return res;
}

/// Ports `llama_model_maincoder`'s `build_arch_graph` (models/maincoder.cpp:51
/// @b1b770d90) and the constructor of its `graph` (models/maincoder.cpp:55
/// @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/maincoder.cpp", 58, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/maincoder.cpp", 59, "n_embd_head == n_rot");

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

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Kcur, "Kcur_normed", il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.llama_c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

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
