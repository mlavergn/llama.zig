//! GroveMoE: pre-norm blocks with Q and K norms whose feed-forward is a
//! mixture of experts plus "adjugate" chunk experts, one per group of
//! experts, routed by the same router and added back at a fixed scale.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/grovemoe.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_grovemoe`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_grovemoe`'s `load_arch_hparams` (models/grovemoe.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    if (try meta.getKey(u32, .expert_chunk_feed_forward_length, false, exc)) |v| hp.n_ff_chexp = v;
    hp.expert_group_scale = (try meta.getKey(f32, .expert_group_scale, true, exc)).?;
    hp.n_group_experts = (try meta.getKey(u32, .experts_per_group, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        48 => .@"30b_a3b",
        else => .unknown,
    };
}

/// Ports `llama_model_grovemoe`'s `load_arch_tensors` (models/grovemoe.cpp:16 @c1d0e7a00).
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

    impl.assert(l.n_expert > 0, "models/grovemoe.cpp", 29, "n_expert > 0 && \"n_expert must be > 0 for GROVEMOE\"");
    impl.assert(l.n_expert_used > 0, "models/grovemoe.cpp", 30, "n_expert_used > 0 && \"n_expert_used must be > 0 for GROVEMOE\"");
    impl.assert(hp.n_group_experts > 0, "models/grovemoe.cpp", 31, "hparams.n_group_experts > 0 && \"n_group_experts must be > 0 for GROVEMOE\"");

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);

        // The mixture of experts.
        const n_ff_exp: i64 = if (hp.n_ff_exp != 0) hp.n_ff_exp else @divTrunc(l.n_ff, l.n_expert_used);
        const n_ff_chexp: i64 = if (hp.n_ff_chexp != 0) hp.n_ff_chexp else l.n_embd_head_k;
        const n_chunk_expert: i64 = @divTrunc(l.n_expert, hp.n_group_experts);

        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

        layer.ffn_gate_chexps = try self.createTensor(.{ .tensor = .ffn_gate_chexps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_chexp, n_chunk_expert }, 0, exc);
        layer.ffn_down_chexps = try self.createTensor(.{ .tensor = .ffn_down_chexps, .suffix = "weight", .bid = i }, &.{ n_ff_chexp, l.n_embd, n_chunk_expert }, 0, exc);
        layer.ffn_up_chexps = try self.createTensor(.{ .tensor = .ffn_up_chexps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_chexp, n_chunk_expert }, 0, exc);
    }
}

/// Ports `llama_model_grovemoe`'s `build_arch_graph` (models/grovemoe.cpp:63 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_grovemoe`'s `graph`
/// (models/grovemoe.cpp:67 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const n_chunk_expert: i64 = @divTrunc(b.n_expert, @as(i64, hp.n_group_experts));

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/grovemoe.cpp", 72, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/grovemoe.cpp", 73, "n_embd_head == n_rot");

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

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Kcur, "Kcur_normed", il);

            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The mixture of experts.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        const probs = b.buildLoraMm(layer.ffn_gate_inp.?, cur, null); // [n_expert, n_tokens]
        b.cb(probs, "ffn_moe_logits", il);

        var moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = null,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .silu,
            .norm_w = true,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .softmax,
            .probs_in = probs,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);
        cur = moe_out;

        // The C++'s TODO: make the expert selection and weights only once.
        moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = null,
            .up_exps = layer.ffn_up_chexps,
            .gate_exps = layer.ffn_gate_chexps,
            .down_exps = layer.ffn_down_chexps.?,
            .n_expert = n_chunk_expert,
            .n_expert_used = if (b.n_expert_used > n_chunk_expert) n_chunk_expert else b.n_expert_used,
            .type_op = .silu,
            .norm_w = true,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .softmax,
            .probs_in = probs,
        }, il);
        b.cb(moe_out, "ffn_adj_moe_out", il);

        cur = c.ggml_add(ctx, cur, c.ggml_scale(ctx, moe_out, hp.expert_group_scale));
        b.cb(cur, "ffn_final_moe_out", il);

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
