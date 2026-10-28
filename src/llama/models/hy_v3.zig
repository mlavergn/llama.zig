//! Hunyuan V3 (HY V3): pre-norm blocks with Q and K norms, dense blocks
//! where the file has no router, a sigmoid-routed mixture of experts with
//! a selection bias and an always-on shared expert elsewhere; and optional
//! NextN/MTP blocks past the main stack.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/hy-v3.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_hy_v3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph` and `graph_mtp`.

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
const Layer = model_mod.Layer;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_hy_v3`'s `load_arch_hparams` (models/hy-v3.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;

    // HY V3 routes by a sigmoid with an expert-selection bias by default.
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }

    self.type = switch (hp.nLayer()) {
        48 => .@"30b_a3b",
        else => .unknown,
    };
}

/// Ports `llama_model_hy_v3`'s `load_arch_tensors` (models/hy-v3.cpp:22 @5266f24da).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    // Trunk-only: the metadata declares MTP layers but their tensors live in
    // a separate file (a user split of target and draft), so they are
    // optional and the trunk loads cleanly.
    var probe_buf: [64]u8 = undefined;
    const mtp_probe = std.fmt.bufPrint(&probe_buf, "blk.{d}.nextn.eh_proj.weight", .{l.n_layer}) catch unreachable;
    const trunk_only = hp.n_layer_nextn > 0 and ml.getWeight(mtp_probe) == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    var mtp_flags: u32 = if (trunk_only) nr else 0;

    if (!ml.load_mtp) mtp_flags |= loader.tensor_skip;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) try loadBlock(self, &l, i, trunk_flags, exc);

    // The NextN/MTP blocks: a whole hy_v3 decoder block plus the NextN
    // projections.
    i = l.n_layer;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        try loadBlock(self, &l, i, mtp_flags, exc);

        layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, mtp_flags, exc);
        layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
        layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
        layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr, exc);
        layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr, exc);
        // hy_v3 keeps the MTP block's trailing final_layernorm here,
        // applied after the decoder block and before the shared LM head.
        layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
    }
}

/// `load_block`, a lambda of `load_arch_tensors` (models/hy-v3.cpp:22
/// @5266f24da).
fn loadBlock(self: *Model, l: *const Model.Locals, i: i32, flags: u32, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    const layer = &self.layers[@intCast(i)];
    const nr = loader.tensor_not_required;
    const n_ff_exp: i64 = if (hp.nFfExp(0) != 0) hp.nFfExp(0) else @divTrunc(l.n_ff, if (l.n_expert_used > 0) l.n_expert_used else 1);
    const n_ff_shexp: i64 = if (hp.n_ff_shexp != 0) hp.n_ff_shexp else n_ff_exp;

    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

    try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, flags, exc);

    layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, flags, exc);
    layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, flags, exc);

    layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

    // The dense FFN of the leading dense blocks (first_k_dense_replace).
    layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);
    layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, nr, exc);
    layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);

    // The routed experts: a sigmoid router with an expert-selection bias.
    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, nr, exc);
    layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .bid = i }, &.{l.n_expert}, nr, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, nr, exc);
    try self.createTensorGateUpExps(layer, i, l.n_embd, n_ff_exp, l.n_expert, nr, exc);

    // The shared expert: always active, without a gate.
    layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, nr, exc);
    layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, nr, exc);
    layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, nr, exc);
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_hy_v3`'s `build_arch_graph` (models/hy-v3.cpp:99
/// @5266f24da): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

/// The mixture of experts and the shared expert of `layer`, or its dense
/// FFN when it has no router; common to `graph` and `graph_mtp`, which
/// differ only in the names they give the results.
fn buildFfnBlock(b: *const graph.Builder, layer: *const Layer, cur_: *Tensor, comptime mtp: bool, il: c_int) *Tensor {
    const hp = b.hparams;
    const p = if (mtp) "mtp_" else "";
    var cur = cur_;
    if (layer.ffn_gate_inp == null) {
        // The dense FFN of the leading dense blocks.
        cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, layer.ffn_up_s, layer.ffn_gate, layer.ffn_gate_b, layer.ffn_gate_s, layer.ffn_down, layer.ffn_down_b, layer.ffn_down_s, null, .silu, .par, il);
        b.cb(cur, p ++ "ffn_dense_out", il);
    } else {
        // The routed experts: sigmoid gating with an expert-selection bias.
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
            .gate_up_exps = layer.ffn_gate_up_exps,
            .up_exps_s = layer.ffn_up_exps_s,
            .gate_exps_s = layer.ffn_gate_exps_s,
            .down_exps_s = layer.ffn_down_exps_s,
        }, il);
        b.cb(moe_out, p ++ "ffn_moe_out", il);

        // The shared expert: always active, without a gate.
        const sh_out = b.buildFfn(cur, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
        b.cb(sh_out, p ++ "ffn_shared_out", il);

        cur = c.ggml_add(b.ctx0, moe_out, sh_out);
        b.cb(cur, p ++ "ffn_out", il);
    }
    return cur;
}

/// Ports the constructor of `llama_model_hy_v3`'s `graph`
/// (models/hy-v3.cpp:106 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/hy-v3.cpp", 109, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/hy-v3.cpp", 110, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKv();
    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    // The MTP/NextN layers are loaded as extra decoder blocks but not run
    // in the main pass.
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);
        }

        if (il == b.n_layer - 1 and b.cparams.embeddings_nextn_masked) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = buildFfnBlock(b, layer, cur, false, il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);

    // The hidden state after the final norm, which the MTP head's hnorm
    // consumes: vLLM feeds the target model's normed output states, and the
    // MTP layer itself returns final_layernorm(h), so the chained state is
    // post-norm.
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (!b.cparams.embeddings_nextn_masked) cur = c.ggml_get_rows(ctx, cur, inp_out_ids);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_hy_v3`'s `graph_mtp`
/// (models/hy-v3.cpp:239 @5266f24da): the MoE draft head. As vLLM's
/// HYV3MultiTokenPredictorLayer: enorm(embed) and hnorm(prev_hidden),
/// concatenated, through eh_proj, a hy_v3 decoder block, the final
/// layernorm (stored as `nextn.shared_head_norm`), and the main model's LM
/// head -- the checkpoint has no MTP head or embeddings of its own.
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, "models/hy-v3.cpp", 241, "hparams.n_layer_nextn > 0 && \"HY_V3 MTP requires n_layer_nextn > 0\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/hy-v3.cpp", 244, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/hy-v3.cpp", 245, "n_embd_head == n_rot");

    const il: c_int = @as(c_int, @intCast(hp.nLayer())) + b.cparams.nextn_layer_offset;
    impl.assert(b.cparams.nextn_layer_offset >= 0 and b.cparams.nextn_layer_offset < @as(c_int, @intCast(hp.n_layer_nextn)), "models/hy-v3.cpp", 250, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range [0, n_layer_nextn)\"");
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, "models/hy-v3.cpp", 253, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, "models/hy-v3.cpp", 254, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, "models/hy-v3.cpp", 255, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");

    const inp = impl.allocator.create(graph.InpEmbd) catch impl.abort("models/hy-v3.cpp", 257, "out of memory");
    inp.* = .{ .n_embd = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.n_embd, b.n_tokens);
    c.ggml_set_input(inp.embd);
    _ = c.ggml_set_name(inp.embd, "mtp_h_input");

    const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;

    const h_input = inp.embd.?;
    const tok_embd = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    b.cb(tok_embd, "mtp_tok_embd", il);

    _ = b.res.addInput(graph.InpEmbd, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();
    const inp_attn = b.buildAttnInpKv();

    const h_norm = b.buildNorm(h_input, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat = c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, null);
    b.cb(cur, "mtp_eh_proj", il);

    const inpSA = cur;

    // The MTP block: a whole hy_v3 decoder layer, as in the trunk's graph.
    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    {
        const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));

        const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
        var Qcur = qkv.q;
        var Kcur = qkv.k;
        const Vcur = qkv.v;

        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);

        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

        cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
        b.cb(cur, "mtp_attn_out", il);
    }

    const ffn_inp = c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "mtp_ffn_inp", il);

    cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "mtp_ffn_norm", il);

    cur = buildFfnBlock(b, layer, cur, true, il);

    cur = c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "mtp_post_ffn", il);

    // The final layernorm, after the decoder block and before the shared
    // head. The normed hidden state seeds the next MTP step, as in vLLM,
    // where HYV3MultiTokenPredictorLayer returns final_layernorm(h).
    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    impl.assert(head_norm_w != null, "models/hy-v3.cpp", 373, "head_norm_w && \"HY_V3 MTP: missing both nextn.shared_head_norm and output_norm\"");
    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
    impl.assert(head_w != null, "models/hy-v3.cpp", 384, "head_w && \"HY_V3 MTP: missing LM head (nextn.shared_head_head or model.output)\"");
    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
