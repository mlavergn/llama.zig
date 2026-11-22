//! Cohere2 MoE: Cohere's parallel block, where attention and the
//! feed-forward both read the one attention norm and are added to the
//! residual together, over interleaved sliding-window attention with RoPE
//! on the sliding and dense leading layers; a mixture of experts with an
//! averaged shared expert, scaled logits, and a NextN block for multi-token
//! prediction.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/cohere2moe.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_cohere2moe`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, its `graph` and its
//! `graph_mtp`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_cohere2moe`'s `load_arch_hparams` (models/cohere2moe.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    var found_norm = false;
    if (try meta.getKey(f32, .attention_layernorm_eps, false, exc)) |v| {
        hp.f_norm_eps = v;
        found_norm = true;
    }
    var found_norm_rms = false;
    if (try meta.getKey(f32, .attention_layernorm_rms_eps, false, exc)) |v| {
        hp.f_norm_rms_eps = v;
        found_norm_rms = true;
    }
    if (!found_norm and !found_norm_rms) {
        return exc.throw(.runtime_error, "missing Cohere2 MoE norm epsilon", .{});
    }
    if (!found_norm_rms) {
        hp.f_norm_rms_eps = 0.0;
    }

    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.f_logit_scale = (try meta.getKey(f32, .logit_scale, true, exc)).?;
    hp.n_layer_dense_lead = (try meta.getKey(u32, .leading_dense_block_count, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    if (try meta.getKey(u32, .expert_shared_count, false, exc)) |v| hp.n_expert_shared = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;

    if (hp.expert_gating_func == @backingInt(llama.hparams.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(llama.hparams.ExpertGatingFunc.sigmoid);
    }

    hp.swa_type = @backingInt(llama.hparams.SwaType.standard);
    try self.loadSwaPattern(ml, 4, true, exc);

    // The dense prefix and the MTP layers take RoPE even when not SWA.
    var il: u32 = 0;
    while (il < hp.n_layer_all) : (il += 1) {
        hp.rope_pattern[il] = @intFromBool(hp.isSwa(il) or il < hp.n_layer_dense_lead or il >= hp.nLayer());
    }

    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    self.type = switch (hp.nLayer()) {
        49 => .@"30b_a3b",
        else => .unknown,
    };
}

/// Ports `llama_model_cohere2moe`'s `load_arch_tensors` (models/cohere2moe.cpp:45 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, ml: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const nr = llama.loader.tensor_not_required;

    const nf = self.nextnFlags(ml, .attn_norm);
    const trunk_flags = nf.trunk;
    const mtp_flags = nf.mtp;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    if (l.n_expert == 0) {
        return exc.throw(.runtime_error, "n_expert must be > 0 for Cohere2Moe", .{});
    }
    if (l.n_expert_used == 0) {
        return exc.throw(.runtime_error, "n_expert_used must be > 0 for Cohere2Moe", .{});
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        try loadBlockTrunk(self, &l, i, trunk_flags, exc);
    }
    // The MTP/NextN layers load as extra decoder blocks.
    i = l.n_layer;
    while (i < l.n_layer_all) : (i += 1) {
        try loadBlockMtp(self, &l, i, mtp_flags, exc);
    }
}

/// Ports `load_arch_tensors`'s `load_block_trunk` lambda
/// (models/cohere2moe.cpp:70 @b1b770d90).
fn loadBlockTrunk(self: *llama.model.Model, l: *const llama.model.Model.Locals, i: i32, flags: u32, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    const layer = &self.layers[@intCast(i)];

    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

    try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, flags, exc);

    if (@as(u32, @intCast(i)) < hp.n_layer_dense_lead) {
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, flags, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
    } else {
        const n_ff_exp: i64 = if (hp.nFfExp(0) != 0) hp.nFfExp(0) else l.n_ff;

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, flags, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
        try self.createTensorGateUpExps(layer, i, l.n_embd, n_ff_exp, l.n_expert, flags, exc);

        if (hp.n_expert_shared > 0) {
            const n_ff_shexp: i64 = if (hp.n_ff_shexp != 0) hp.n_ff_shexp else n_ff_exp * hp.n_expert_shared;
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, flags, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, flags, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, flags, exc);
        }
    }
}

/// Ports `load_arch_tensors`'s `load_block_mtp` lambda
/// (models/cohere2moe.cpp:98 @b1b770d90).
fn loadBlockMtp(self: *llama.model.Model, l: *const llama.model.Model.Locals, i: i32, flags: u32, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;
    const layer = &self.layers[@intCast(i)];

    // The MTP block looks like a full-attention Cohere2 MoE decoder block.
    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

    try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, flags, exc);

    const n_ff_exp: i64 = if (hp.nFfExp(0) != 0) hp.nFfExp(0) else l.n_ff;

    // The routed experts.
    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, flags, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
    try self.createTensorGateUpExps(layer, i, l.n_embd, n_ff_exp, l.n_expert, flags, exc);

    if (hp.n_expert_shared > 0) {
        const n_ff_shexp: i64 = if (hp.n_ff_shexp != 0) hp.n_ff_shexp else n_ff_exp * hp.n_expert_shared;

        // The shared experts.
        layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, flags, exc);
        layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, flags, exc);
        layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, flags, exc);
    }

    // The NextN tensors that define the MTP block. The optional three
    // take no `flags`, as in the C++.
    layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, flags, exc);
    layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
    layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
    layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
}

/// Ports `llama_model_cohere2moe`'s `get_rope_params`
/// (models/cohere2moe.cpp:141 @b1b770d90): the SWA layers take the global
/// base and scale.
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

/// Ports `llama_model_cohere2moe`'s `build_arch_graph` (models/cohere2moe.cpp:151
/// @b1b770d90): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, params: *const llama.graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_cohere2moe`'s `graph`
/// (models/cohere2moe.cpp:158 @b1b770d90).
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/cohere2moe.cpp", 161, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/cohere2moe.cpp", 162, "n_embd_head == n_rot");

    const cohere2moe_norm_type: llama.graph.NormType = if (hp.f_norm_rms_eps == 0.0) .norm else .rms;
    const f_logit_scale = hp.f_logit_scale;
    var inpL: *llama.impl.c.ggml_tensor = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    // The MTP/NextN layers load as extra decoder blocks, but the main pass
    // does not run them.
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        var cur = b.buildNorm(inpL, model.layers[@intCast(il)].attn_norm, null, cohere2moe_norm_type, il);
        b.cb(cur, "attn_norm", il);

        var ffn_inp = cur;

        {
            const layer = &model.layers[@intCast(il)];

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
        }

        if (il == b.n_layer - 1 and b.cropBeforeNextn(inp_out_ids)) {
            cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = llama.impl.c.ggml_get_rows(ctx, inpL, inp_out_ids);
            ffn_inp = llama.impl.c.ggml_get_rows(ctx, ffn_inp, inp_out_ids);
        }

        const attn_out = cur;

        const layer = &model.layers[@intCast(il)];

        if (layer.ffn_gate_inp == null) {
            cur = b.buildFfn(ffn_inp, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            cur = b.buildMoeFfn(ffn_inp, .{
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
                .gate_up_exps = layer.ffn_gate_up_exps,
                .up_exps_s = layer.ffn_up_exps_s,
                .gate_exps_s = layer.ffn_gate_exps_s,
                .down_exps_s = layer.ffn_down_exps_s,
            }, il);
            b.cb(cur, "ffn_moe_out", il);

            if (layer.ffn_up_shexp != null) {
                const ffn_shexp = b.buildFfn(ffn_inp, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
                b.cb(ffn_shexp, "ffn_shexp", il);

                cur = llama.impl.c.ggml_add(ctx, cur, ffn_shexp);
                cur = llama.impl.c.ggml_scale(ctx, cur, 0.5);
                b.cb(cur, "ffn_out", il);
            }
        }

        cur = llama.impl.c.ggml_add(ctx, cur, inpL);
        cur = llama.impl.c.ggml_add(ctx, cur, attn_out);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = inpL;
    cur = b.buildNorm(cur, model.output_norm, null, cohere2moe_norm_type, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (b.cropAfterNextn(inp_out_ids)) {
        cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, null);

    if (f_logit_scale != 0.0) {
        cur = llama.impl.c.ggml_scale(ctx, cur, f_logit_scale);
    }

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

/// Ports the constructor of `llama_model_cohere2moe`'s `graph_mtp`
/// (models/cohere2moe.cpp:281 @b1b770d90).
fn buildGraphMtp(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    llama.impl.assert(hp.n_layer_nextn > 0, "models/cohere2moe.cpp", 282, "hparams.n_layer_nextn > 0 && \"COHERE2MOE MTP requires n_layer_nextn > 0\"");
    llama.impl.assert(hp.n_layer_nextn == 1, "models/cohere2moe.cpp", 283, "hparams.n_layer_nextn == 1 && \"COHERE2MOE MTP currently only supports a single MTP block\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/cohere2moe.cpp", 286, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/cohere2moe.cpp", 287, "n_embd_head == n_rot");

    const il: c_int = @intCast(hp.nLayer());
    const layer = &model.layers[@intCast(il)];
    llama.impl.assert(layer.nextn.eh_proj != null, "models/cohere2moe.cpp", 291, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    llama.impl.assert(layer.nextn.enorm != null, "models/cohere2moe.cpp", 292, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    llama.impl.assert(layer.nextn.hnorm != null, "models/cohere2moe.cpp", 293, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");
    llama.impl.assert(layer.ffn_gate_inp != null, "models/cohere2moe.cpp", 294, "layer.ffn_gate_inp  && \"MTP block missing ffn_gate_inp\"");

    const cohere2moe_norm_type: llama.graph.NormType = if (hp.f_norm_rms_eps == 0.0) .norm else .rms;

    // The C++'s TODO: extract into a common `build_inp_embd_h`.
    const inp = std.heap.c_allocator.create(llama.graph.InpEmbdH) catch llama.impl.abort("models/cohere2moe.cpp", 299, "out of memory");
    inp.* = .{ .n_embd = hp.nEmbdInp(), .n_embd_state = hp.n_embd };

    inp.tokens = llama.impl.c.ggml_new_tensor_1d(ctx, llama.impl.c.GGML_TYPE_I32, b.n_tokens);
    llama.impl.c.ggml_set_input(inp.tokens);

    inp.embd = llama.impl.c.ggml_new_tensor_2d(ctx, llama.impl.c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    llama.impl.c.ggml_set_input(inp.embd);

    // The C++'s TODO: make this static with `ggml_build_forward_select`,
    // as `build_inp_embd` does.
    var tok_embd: *llama.impl.c.ggml_tensor = undefined;
    b.ubatch.assertEmbdOrToken("models/cohere2moe.cpp", 310);
    if (b.ubatch.token != null) {
        const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;
        tok_embd = llama.impl.c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    } else {
        tok_embd = inp.embd.?;
    }
    b.cb(tok_embd, "mtp_tok_embd", il);

    inp.h = b.buildInpEmbdState(inp, hp.n_embd);

    const h_embd = inp.h.?;

    _ = b.res.addInput(llama.graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();
    const inp_attn = b.buildAttnInpKvIswa();

    const h_norm = b.buildNorm(h_embd, layer.nextn.hnorm, null, cohere2moe_norm_type, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, cohere2moe_norm_type, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(cur, "mtp_eh_proj", il);

    const inpL = cur;

    cur = b.buildNorm(cur, layer.attn_norm, null, cohere2moe_norm_type, il);
    b.cb(cur, "mtp_attn_norm", il);
    const ffn_inp = cur;

    const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
    var Qcur = qkv.q;
    var Kcur = qkv.k;
    const Vcur = qkv.v;
    Qcur = b.buildRope(Qcur, inp_pos, il);
    Kcur = b.buildRope(Kcur, inp_pos, il);

    b.cb(Qcur, "mtp_Qcur", il);
    b.cb(Kcur, "mtp_Kcur", il);
    b.cb(Vcur, "mtp_Vcur", il);

    cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
    b.cb(cur, "mtp_attn_out", il);

    const attn_out = cur;

    cur = b.buildMoeFfn(ffn_inp, .{
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
        .gate_up_exps = layer.ffn_gate_up_exps,
        .up_exps_s = layer.ffn_up_exps_s,
        .gate_exps_s = layer.ffn_gate_exps_s,
        .down_exps_s = layer.ffn_down_exps_s,
    }, il);
    b.cb(cur, "mtp_ffn_moe_out", il);

    if (layer.ffn_up_shexp != null) {
        const ffn_shexp = b.buildFfn(ffn_inp, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
        b.cb(ffn_shexp, "mtp_ffn_shexp", il);

        cur = llama.impl.c.ggml_add(ctx, cur, ffn_shexp);
        cur = llama.impl.c.ggml_scale(ctx, cur, 0.5);
        b.cb(cur, "mtp_ffn_out", il);
    }

    cur = llama.impl.c.ggml_add(ctx, cur, inpL);
    cur = llama.impl.c.ggml_add(ctx, cur, attn_out);
    b.cb(cur, "mtp_post_ffn", il);

    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    llama.impl.assert(head_norm_w != null, "models/cohere2moe.cpp", 400, "head_norm_w && \"COHERE2MOE MTP: missing both nextn.shared_head_norm and output_norm\"");
    cur = b.buildNorm(cur, head_norm_w, null, cohere2moe_norm_type, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    llama.impl.assert(head_w != null, "models/cohere2moe.cpp", 410, "head_w && \"COHERE2MOE MTP: missing LM head (nextn.shared_head_head or model.output)\"");
    cur = b.buildLoraMm(head_w.?, cur, if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else null);

    if (hp.f_logit_scale != 0.0) {
        cur = llama.impl.c.ggml_scale(ctx, cur, hp.f_logit_scale);
    }

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
