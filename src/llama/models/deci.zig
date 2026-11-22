//! Deci (Llama-3.1-Nemotron): Llama's decoder blocks with per-layer head
//! and feed-forward widths, where a layer may have no attention, a
//! "linear attention" that is only the output projection, or no
//! feed-forward at all.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/deci.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_deci`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_deci`'s `load_arch_hparams` (models/deci.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try ml.meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        32 => .@"7b",
        80 => .@"70b",
        162 => .@"405b",
        else => .unknown,
    };
}

/// Ports `llama_model_deci`'s `load_arch_tensors` (models/deci.cpp:14 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
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
        const iu: u32 = @intCast(i);
        const n_embd_k_gqa: i64 = hp.nEmbdKGqa(iu);
        const n_embd_v_gqa: i64 = hp.nEmbdVGqa(iu);
        const n_ff: i64 = hp.nFf(iu);
        const n_head: i64 = hp.nHead(iu);
        const n_head_kv: i64 = hp.nHeadKv(iu);

        if (n_head_kv == 0 and n_head > 0) {
            // The linear attention of DeciLMCausalModel.
            layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        } else if (n_head_kv > 0) {
            layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

            try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * n_head, l.n_embd }, 0, exc);
        }

        // Optional bias tensors.
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        if (n_ff > 0) {
            layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        }

        const rope_dup: u32 = if (i != 0) dup else 0;
        if (hp.rope_scaling_type_train == llama.impl.llama_c.LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long = try self.createTensor(.{ .tensor = .rope_factors_long, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
            layer.rope_short = try self.createTensor(.{ .tensor = .rope_factors_short, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
        } else {
            layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
        }

        if (n_ff > 0) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff }, 0, exc);
        }

        // Optional MLP bias.
        layer.ffn_gate_b = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "bias", .bid = i }, &.{n_ff}, nr, exc);
        layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{n_ff}, nr, exc);
    }
}

/// Ports `llama_model_deci`'s `build_arch_graph` (models/deci.cpp:76
/// @b1b770d90) and the constructor of its `graph` (models/deci.cpp:80
/// @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/deci.cpp", 83, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/deci.cpp", 84, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / stdlib.libm.sqrtfInt(n_embd_head) else hp.f_attention_scale;

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const ilu: u32 = @intCast(il);
        var inpSA = inpL;
        const n_head_kv: i64 = hp.nHeadKv(ilu);
        const n_head: i64 = hp.nHead(ilu);
        const n_ff: i64 = hp.nFf(ilu);

        var cur: *llama.impl.llama_c.ggml_tensor = undefined;
        if (n_head == 0) {
            // The attention-free layer of Llama-3_1-Nemotron-51B.
            cur = inpL;
        } else {
            // Norm.
            cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);
        }
        if (n_head > 0 and n_head_kv == 0) {
            // The "linear attention" of Llama-3_1-Nemotron-51B.
            cur = b.buildLoraMm(layer.wo.?, cur, null);
            b.cb(cur, "wo", il);
        } else if (n_head > 0) {
            // Self-attention.
            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, n_embd_head, n_head, n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.llama_c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        // The FFN-free layer of Llama-3_1-Nemotron-Ultra-253B, which leaves
        // the next layer's input where it was.
        if (n_ff == 0) {
            continue;
        }
        // Modified to support the attention-free layer of
        // Llama-3_1-Nemotron-51B.
        var ffn_inp = cur;
        if (n_head > 0) {
            ffn_inp = llama.impl.llama_c.ggml_add(ctx, cur, inpSA);
            b.cb(ffn_inp, "ffn_inp", il);
        }
        // The feed-forward network.
        if (layer.ffn_gate_inp == null) {
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        }
        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

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
