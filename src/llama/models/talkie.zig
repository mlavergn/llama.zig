//! Talkie: decoder blocks whose norms carry no weights, Q normed per head
//! with a one-element gain and K without one, both after RoPE, and a
//! learned-scale skip from the normed embedding into every layer's output.
//! The logits are scaled.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/talkie.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_talkie`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_talkie`'s `load_arch_hparams` (models/talkie.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try ml.meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_logit_scale = (try ml.meta.getKey(f32, .logit_scale, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        40 => .@"13b",
        else => .unknown,
    };
}

/// Ports `llama_model_talkie`'s `load_arch_tensors` (models/talkie.cpp:13 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        // No K gain.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{ 1, l.n_head }, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);

        layer.out_scale = try self.createTensor(.{ .tensor = .layer_out_scale, .suffix = "weight", .bid = i }, &.{1}, 0, exc);
    }
}

/// Ports `llama_model_talkie`'s `build_arch_graph` (models/talkie.cpp:36
/// @b1b770d90) and the constructor of its `graph` (models/talkie.cpp:40
/// @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadK(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadV(0), "models/talkie.cpp", 43, "n_embd_head == hparams.n_embd_head_v()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/talkie.cpp", 44, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    inpL = b.buildNorm(inpL, null, null, .rms, -1);
    b.cb(inpL, "inp_norm", -1);

    const embd_skip = inpL;

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / stdlib.libm.sqrtfInt(n_embd_head);

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;
        var inp_skip = embd_skip;

        var cur = b.buildNorm(inpL, null, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            // The reference applies the QK norm after RoPE.
            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_norm", il);

            Kcur = b.buildNorm(Kcur, null, null, .rms, il);
            b.cb(Kcur, "Kcur_norm", il);

            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);
        }

        if (il == b.n_layer - 1) {
            cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.c.ggml_get_rows(ctx, inpSA, inp_out_ids);
            inp_skip = llama.impl.c.ggml_get_rows(ctx, inp_skip, inp_out_ids);
        }

        const ffn_inp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, null, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = llama.impl.c.ggml_add(ctx, cur, ffn_inp);

        const skip: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul(ctx, inp_skip, layer.out_scale);
        b.cb(skip, "embd_skip", il);

        cur = llama.impl.c.ggml_add(ctx, cur, skip);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, null, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, null);
    cur = llama.impl.c.ggml_scale(ctx, cur, hp.f_logit_scale);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
