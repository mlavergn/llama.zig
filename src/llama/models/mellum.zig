//! Mellum: a softmax-routed mixture of experts in every layer, RMS-normed
//! Q and K, and optional interleaved sliding-window attention whose
//! sliding layers take plain RoPE without YaRN scaling, with an optional
//! NextN draft head.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mellum.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_mellum`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph<iswa>` and `graph_mtp<iswa>`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_mellum`'s `load_arch_hparams` (models/mellum.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| hp.n_swa = v;

    hp.expert_weights_norm = true;
    hp.expert_gating_func = @backingInt(llama.hparams.ExpertGatingFunc.softmax);
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;

    if (hp.n_swa > 0) {
        hp.swa_type = @backingInt(llama.hparams.SwaType.standard);

        try self.loadSwaPattern(ml, 4, false, exc);

        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        // The sliding-window layers take plain RoPE, without YaRN.
        hp.rope_freq_scale_train_swa = 1.0;

        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    } else {
        hp.swa_type = @backingInt(llama.hparams.SwaType.none);
    }

    self.type = switch (hp.nLayer()) {
        28 => .@"12b_a2_5b",
        else => .unknown,
    };
}

/// One decoder block's tensors, the constructor's `load_block` lambda
/// (models/mellum.cpp:52 @b1b770d90).
fn loadBlock(self: *llama.model.Model, l: *const llama.model.Model.Locals, i: i32, flags: u32, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    const layer = &self.layers[@intCast(i)];

    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

    try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, flags, exc);

    layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, flags, exc);
    layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, flags, exc);

    layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, flags, exc);

    const n_ff_exp: i64 = if (hp.nFfExp(0) != 0) hp.nFfExp(0) else @divTrunc(l.n_ff, l.n_expert_used);

    layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
    layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);
}

/// Ports `llama_model_mellum`'s `load_arch_tensors` (models/mellum.cpp:32 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, ml: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();

    const nf = self.nextnFlags(ml, .attn_norm);
    const trunk_flags = nf.trunk;
    const mtp_flags = nf.mtp;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    if (l.n_expert == 0) {
        return exc.throw(.runtime_error, "n_expert must be > 0 for Mellum", .{});
    }
    if (l.n_expert_used == 0) {
        return exc.throw(.runtime_error, "n_expert_used must be > 0 for Mellum", .{});
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        try loadBlock(self, &l, i, trunk_flags, exc);
    }

    // The MTP block is a regular Mellum decoder block plus the NextN
    // projection and norms.
    i = l.n_layer;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        try loadBlock(self, &l, i, mtp_flags, exc);

        layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, mtp_flags, exc);
        layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
        layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
        layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
    }
}

/// Ports `llama_model_mellum`'s `get_rope_params` (models/mellum.cpp:91
/// @b1b770d90): the sliding-window layers take plain RoPE, without YaRN.
///
/// Parameters:
/// - `model`: the model.
/// - `cparams`: the context's parameters.
/// - `il`: the layer.
///
/// Return: the layer's RoPE parameters.
pub fn getRopeParams(model: *const llama.model.Model, cparams: *const llama.cparams.Cparams, il: u32) llama.model.RopeParams {
    var res = model.getRopeParamsBase(cparams, il);

    if (model.hparams.isSwa(il)) {
        res.ext_factor = 0.0;
        res.attn_factor = 1.0;
        res.beta_fast = 32.0;
        res.beta_slow = 1.0;
    }

    return res;
}

/// Ports `llama_model_mellum`'s `build_arch_graph` (models/mellum.cpp:104 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, params: *const llama.graph.Params) void {
    const iswa = @as(llama.hparams.SwaType, @fromBackingInt(@intCast(model.hparams.swa_type))) == .standard;

    if (params.gtype == .decoder_mtp) {
        if (iswa) {
            return buildGraphMtp(true, model, b);
        }
        return buildGraphMtp(false, model, b);
    }

    if (iswa) {
        return buildGraph(true, model, b);
    }
    buildGraph(false, model, b);
}

/// Ports the constructor of `llama_model_mellum`'s `graph<iswa>`
/// (models/mellum.cpp:121 @b1b770d90).
fn buildGraph(comptime iswa: bool, model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/mellum.cpp", 124, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/mellum.cpp", 125, "n_embd_head == n_rot");

    var inpL: *llama.impl.c.ggml_tensor = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

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

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Kcur, "Kcur_normed", il);

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
        }
        if (il == b.n_layer - 1 and b.cropBeforeNextn(inp_out_ids)) {
            cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The mixture of experts.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        const moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .silu,
            .norm_w = hp.expert_weights_norm,
            .w_scale = hp.expert_weights_scale,
            .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
            .up_exps_s = layer.ffn_up_exps_s,
            .gate_exps_s = layer.ffn_gate_exps_s,
            .down_exps_s = layer.ffn_down_exps_s,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);
        cur = moe_out;

        cur = llama.impl.c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    // The post-norm hidden state feeds both the LM head and the MTP head.
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (b.cropAfterNextn(inp_out_ids)) {
        cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

/// Ports the constructor of `llama_model_mellum`'s `graph_mtp<iswa>`
/// (models/mellum.cpp:244 @b1b770d90): the NextN draft head.
fn buildGraphMtp(comptime iswa: bool, model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    llama.impl.assert(hp.n_layer_nextn == 1, "models/mellum.cpp", 245, "hparams.n_layer_nextn == 1 && \"MELLUM MTP currently only supports a single MTP block\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/mellum.cpp", 249, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/mellum.cpp", 250, "n_embd_head == n_rot");

    const il: c_int = @intCast(hp.nLayer());
    const layer = &model.layers[@intCast(il)];

    llama.impl.assert(layer.nextn.eh_proj != null, "models/mellum.cpp", 255, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    llama.impl.assert(layer.nextn.enorm != null, "models/mellum.cpp", 256, "layer.nextn.enorm && \"MTP block missing nextn.enorm\"");
    llama.impl.assert(layer.nextn.hnorm != null, "models/mellum.cpp", 257, "layer.nextn.hnorm && \"MTP block missing nextn.hnorm\"");
    llama.impl.assert(layer.nextn.shared_head_norm != null, "models/mellum.cpp", 258, "layer.nextn.shared_head_norm && \"MTP block missing nextn.shared_head_norm\"");

    const inp = std.heap.c_allocator.create(llama.graph.InpEmbdH) catch llama.impl.abort("models/mellum.cpp", 261, "out of memory");
    inp.* = .{ .n_embd = hp.nEmbdInp(), .n_embd_state = hp.n_embd };
    inp.tokens = llama.impl.c.ggml_new_tensor_1d(ctx, llama.impl.c.GGML_TYPE_I32, b.n_tokens);
    llama.impl.c.ggml_set_input(inp.tokens);
    inp.embd = llama.impl.c.ggml_new_tensor_2d(ctx, llama.impl.c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    llama.impl.c.ggml_set_input(inp.embd);
    inp.h = b.buildInpEmbdState(inp, hp.n_embd);

    b.ubatch.assertEmbdOrToken("models/mellum.cpp", 274);
    const tok_embd: *llama.impl.c.ggml_tensor = if (b.ubatch.token != null) llama.impl.c.ggml_get_rows(ctx, model.tok_embd, inp.tokens) else inp.embd.?;
    b.cb(tok_embd, "mtp_tok_embd", il);

    const h_embd: *llama.impl.c.ggml_tensor = inp.h.?;

    _ = b.res.addInput(llama.graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const h_norm = b.buildNorm(h_embd, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const concat: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(cur, "mtp_eh_proj", il);

    const inpSA = cur;

    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    // Self-attention.
    {
        const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
        var Qcur = qkv.q;
        var Kcur = qkv.k;
        const Vcur = qkv.v;

        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        b.cb(Qcur, "mtp_Qcur_normed", il);

        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
        b.cb(Kcur, "mtp_Kcur_normed", il);

        Qcur = b.buildRope(Qcur, inp_pos, il);
        Kcur = b.buildRope(Kcur, inp_pos, il);

        b.cb(Qcur, "mtp_Qcur", il);
        b.cb(Kcur, "mtp_Kcur", il);
        b.cb(Vcur, "mtp_Vcur", il);

        cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
        b.cb(cur, "mtp_attn_out", il);
    }

    const ffn_inp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "mtp_ffn_inp", il);

    cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "mtp_ffn_norm", il);

    cur = b.buildMoeFfn(cur, .{
        .gate_inp = layer.ffn_gate_inp,
        .up_exps = layer.ffn_up_exps,
        .gate_exps = layer.ffn_gate_exps,
        .down_exps = layer.ffn_down_exps.?,
        .n_expert = b.n_expert,
        .n_expert_used = b.n_expert_used,
        .type_op = .silu,
        .norm_w = hp.expert_weights_norm,
        .w_scale = hp.expert_weights_scale,
        .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
        .up_exps_s = layer.ffn_up_exps_s,
        .gate_exps_s = layer.ffn_gate_exps_s,
        .down_exps_s = layer.ffn_down_exps_s,
    }, il);
    b.cb(cur, "mtp_ffn_moe_out", il);

    cur = llama.impl.c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "mtp_ffn_out", il);

    cur = b.buildNorm(cur, layer.nextn.shared_head_norm, null, .rms, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
