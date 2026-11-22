//! SmallThinker: a mixture of experts with a ReLU-gated feed-forward whose
//! router reads the layer's input before the attention norm, and optional
//! interleaved sliding-window attention with RoPE on the sliding layers
//! only.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/smallthinker.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_smallthinker`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph<iswa>`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_smallthinker`'s `load_arch_hparams` (models/smallthinker.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    var found_swa = false;
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| {
        hp.n_swa = v;
        found_swa = true;
    }

    if (found_swa and hp.n_swa > 0) {
        hp.swa_type = @backingInt(llama.hparams.SwaType.standard);
        hp.n_swa = 4096;
        try self.loadSwaPattern(ml, 4, true, exc);

        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    } else {
        hp.swa_type = @backingInt(llama.hparams.SwaType.none);
        hp.n_no_rope_layer_step = hp.nLayer();
    }

    var il: u32 = 0;
    while (il < hp.n_layer_all) : (il += 1) {
        hp.rope_pattern[il] = @intFromBool(hp.n_no_rope_layer_step == hp.nLayer() or il % hp.n_no_rope_layer_step != 0);
    }

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, false, exc);
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;

    self.type = switch (hp.nLayer()) {
        32 => .@"4b",
        52 => .@"20b",
        else => .unknown,
    };
}

/// Ports `llama_model_smallthinker`'s `load_arch_tensors` (models/smallthinker.cpp:34 @b1b770d90).
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

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        llama.impl.assert(l.n_expert > 0, "models/smallthinker.cpp", 58, "n_expert > 0 && \"n_expert must be > 0 for SMALLTHINKER\"");
        llama.impl.assert(l.n_expert_used > 0, "models/smallthinker.cpp", 59, "n_expert_used > 0 && \"n_expert_used must be > 0 for SMALLTHINKER\"");

        // The mixture of experts.
        const n_ff_exp: i64 = self.hparams.nFfExp(0);
        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
    }
}

/// Ports `llama_model_smallthinker`'s `build_arch_graph` (models/smallthinker.cpp:70 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    if (@as(llama.hparams.SwaType, @fromBackingInt(@intCast(model.hparams.swa_type))) == .standard) {
        buildGraph(true, model, b);
    } else {
        buildGraph(false, model, b);
    }
}

/// Ports the constructor of `llama_model_smallthinker`'s `graph<iswa>`
/// (models/smallthinker.cpp:79 @b1b770d90).
fn buildGraph(comptime iswa: bool, model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/smallthinker.cpp", 82, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/smallthinker.cpp", 83, "n_embd_head == n_rot");

    var inpL: *llama.impl.c.ggml_tensor = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        var probs: *llama.impl.c.ggml_tensor = b.buildLoraMm(layer.ffn_gate_inp.?, inpL, null); // [n_expert, n_tokens]
        b.cb(probs, "ffn_moe_logits", il);

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
            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.c.ggml_get_rows(ctx, inpSA, inp_out_ids);
            probs = llama.impl.c.ggml_get_rows(ctx, probs, inp_out_ids);
        }
        const ffn_inp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The mixture of experts.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        const ffn_out = b.buildMoeFfn(cur, .{
            .gate_inp = null,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .relu,
            .norm_w = true,
            .w_scale = hp.expert_weights_scale,
            .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
            .probs_in = probs,
        }, il);

        b.cb(ffn_out, "ffn_out", il);
        cur = ffn_out;

        cur = llama.impl.c.ggml_add(ctx, cur, ffn_inp);

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
