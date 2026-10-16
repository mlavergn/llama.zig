//! Step 3.5: interleaved sliding-window attention with partial RoPE per
//! layer -- the full-attention layers rotate half the dimensions -- RMS
//! Q and K norms and a head-wise sigmoid gate on the attention output; a
//! sigmoid-routed mixture of experts with a shared expert, clamped SwiGLU,
//! and NextN blocks for multi-token prediction.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/step35.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_step35`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph` and its `graph_mtp`.

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
const Tensor = c.ggml_tensor;

/// Ports `llama_model_step35`'s `load_arch_hparams` (models/step35.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);

    // The full-attention layers rotate only half the RoPE dimensions.
    hp.n_rot_full = hp.n_rot_full / 2;

    // The MoE and SWA parameters.
    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;

    // Step35 gates with a sigmoid unless the GGUF says otherwise.
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }

    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    _ = try meta.getKeyOrArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, hp.nLayer(), true, exc);

    _ = try meta.getKeyOrArr(f32, .swiglu_clamp_exp, &hp.swiglu_clamp_exp, hp.nLayer(), false, exc);
    _ = try meta.getKeyOrArr(f32, .swiglu_clamp_shexp, &hp.swiglu_clamp_shexp, hp.nLayer(), false, exc);

    // NextN/MTP (Step3p5): an extra decoder block past the main stack.
    if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
    impl.assert(hp.n_layer_nextn < hp.n_layer_all, "models/step35.cpp", 33, "hparams.n_layer_nextn < hparams.n_layer_all && \"n_layer_nextn must be < n_layer_impl\"");

    self.type = switch (hp.nLayer()) {
        45 => .@"196b_a11b",
        else => .unknown,
    };
}

/// Ports `llama_model_step35`'s `load_arch_tensors` (models/step35.cpp:41 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    // Trunk-only: the GGUF declares MTP layers in its metadata but the MTP
    // tensors live in a separate file (a split target and draft, say), so
    // they are not required and the trunk loads cleanly.
    var probe_buf: [64]u8 = undefined;
    const mtp_probe = std.fmt.bufPrint(&probe_buf, "blk.{d}.nextn.eh_proj.weight", .{l.n_layer}) catch unreachable;
    const trunk_only = hp.n_layer_nextn > 0 and ml.getWeight(mtp_probe) == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    var mtp_flags: u32 = if (trunk_only) nr else 0;

    if (!ml.load_mtp) {
        mtp_flags |= loader.tensor_skip;
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, trunk_flags, exc);

    // STEP35 has partial RoPE dimensions per layer; the RoPE factors are
    // one shared tensor ("rope_freqs.weight"), of which ggml reads the
    // first n_rot_l/2 entries per layer.
    var n_rot_max: u32 = 0;
    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        n_rot_max = @max(n_rot_max, hp.nRot(@intCast(i)));
    }
    if (n_rot_max == 0) {
        n_rot_max = @intCast(l.n_rot);
    }

    i = 0;
    while (i < l.n_layer) : (i += 1) {
        try loadBlockTrunk(self, &l, n_rot_max, i, trunk_flags, exc);
    }
    // Every one of the n_layer_nextn MTP blocks is required: the
    // multi-block draft chain runs every head (head k at offset k). The
    // GGUF declares the count in step35.nextn_predict_layers.
    i = l.n_layer;
    while (i < l.n_layer_all) : (i += 1) {
        try loadBlockMtp(self, &l, n_rot_max, mtp_flags, i, exc);
    }
}

/// Ports `load_arch_tensors`'s `load_block_trunk` lambda
/// (models/step35.cpp:73 @c1d0e7a00).
fn loadBlockTrunk(self: *Model, l: *const Model.Locals, n_rot_max: u32, i: i32, flags: u32, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;
    const layer = &self.layers[@intCast(i)];

    const n_head_l: i64 = hp.nHead(@intCast(i));
    const n_embd_k_gqa: i64 = hp.nEmbdKGqa(@intCast(i));
    const n_embd_v_gqa: i64 = hp.nEmbdVGqa(@intCast(i));

    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
    layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, nr, exc);
    layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, nr, exc);

    // Optional RoPE factors (Llama 3) or LongRoPE tensors.
    const rope_dup: u32 = if (i != 0) dup else 0;
    if (hp.rope_scaling_type_train == c.LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
        layer.rope_long = try self.createTensor(.{ .tensor = .rope_factors_long, .suffix = "weight", .bid = i }, &.{n_rot_max / 2}, nr | rope_dup, exc);
        layer.rope_short = try self.createTensor(.{ .tensor = .rope_factors_short, .suffix = "weight", .bid = i }, &.{n_rot_max / 2}, nr | rope_dup, exc);
    } else {
        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{n_rot_max / 2}, nr | rope_dup, exc);
    }

    try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * n_head_l, n_embd_k_gqa, n_embd_v_gqa, flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_v * n_head_l, l.n_embd }, flags, exc);

    // The head-wise attention gate (Step35's self_attn.g_proj).
    layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_head_l }, nr, exc);

    layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

    // The dense MLP of the leading dense blocks.
    layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);
    layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, nr, exc);
    layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);

    // The routed experts, and the selection bias (router_bias).
    const n_ff_exp: i64 = hp.n_ff_exp;
    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, nr, exc);
    layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, nr, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, nr, exc);
    layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, nr, exc);
    layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr, exc);

    // The shared-expert MLP.
    layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, nr, exc);
    layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, nr, exc);
    layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ hp.n_ff_shexp, l.n_embd }, nr, exc);
}

/// Ports `load_arch_tensors`'s `load_block_mtp` lambda
/// (models/step35.cpp:119 @c1d0e7a00).
fn loadBlockMtp(self: *Model, l: *const Model.Locals, n_rot_max: u32, mtp_flags: u32, i: i32, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;
    const layer = &self.layers[@intCast(i)];

    const n_head_l: i64 = hp.nHead(@intCast(i));
    const n_embd_k_gqa: i64 = hp.nEmbdKGqa(@intCast(i));
    const n_embd_v_gqa: i64 = hp.nEmbdVGqa(@intCast(i));

    // The MTP block is a full Step3p5 decoder layer (mtp_block) plus the
    // NextN wiring (enorm, hnorm, eh_proj and an optional shared head).
    // Every declared block is required -- the draft chain runs all
    // n_layer_nextn heads -- so each takes `mtp_flags` as it is, already
    // not required for a trunk-only GGUF.

    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
    layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, nr, exc);
    layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, nr, exc);

    if (hp.rope_scaling_type_train == c.LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
        layer.rope_long = try self.createTensor(.{ .tensor = .rope_factors_long, .suffix = "weight", .bid = i }, &.{n_rot_max / 2}, nr | dup, exc);
        layer.rope_short = try self.createTensor(.{ .tensor = .rope_factors_short, .suffix = "weight", .bid = i }, &.{n_rot_max / 2}, nr | dup, exc);
    } else {
        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{n_rot_max / 2}, nr | dup, exc);
    }

    try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * n_head_l, n_embd_k_gqa, n_embd_v_gqa, mtp_flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_v * n_head_l, l.n_embd }, mtp_flags, exc);

    layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_head_l }, nr, exc);

    layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);

    // The dense MLP, present when the MTP block is not a mixture of experts.
    layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);
    layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, nr, exc);
    layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);

    // The routed experts, and the selection bias (router_bias).
    const n_ff_exp: i64 = hp.n_ff_exp;
    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, nr, exc);
    layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, nr, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, nr, exc);
    layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, nr, exc);
    layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr, exc);

    layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, nr, exc);
    layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, nr, exc);
    layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ hp.n_ff_shexp, l.n_embd }, nr, exc);

    // The NextN tensors that define the MTP block.
    layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, mtp_flags, exc);
    layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
    layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
    layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
}

/// Ports `llama_model_step35`'s `build_arch_graph` (models/step35.cpp:188
/// @c1d0e7a00): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    buildGraph(model, b);
}

/// The head-wise attention gate, sigmoid(g_proj(x)), shared by both graphs.
fn headGate(b: *const graph.Builder, wqkv_gate: *Tensor, cur: *Tensor, attn_out: *Tensor, n_head_l: i64, comptime names: [5][*:0]const u8, il: c_int) *Tensor {
    const ctx = b.ctx0;
    var gate = b.buildLoraMm(wqkv_gate, cur, null); // [n_head_l, n_tokens]
    b.cb(gate, names[0], il);

    gate = c.ggml_sigmoid(ctx, gate);
    b.cb(gate, names[1], il);

    // Reshape and broadcast to [n_embd_head_v, n_head_l, n_tokens].
    var attn_3d: *Tensor = c.ggml_reshape_3d(ctx, attn_out, b.n_embd_head_v, n_head_l, b.n_tokens);
    const gate_3d: *Tensor = c.ggml_reshape_3d(ctx, gate, 1, n_head_l, b.n_tokens);
    b.cb(gate_3d, names[2], il);

    attn_3d = c.ggml_mul(ctx, attn_3d, gate_3d);
    b.cb(attn_3d, names[3], il);

    const gated: *Tensor = c.ggml_reshape_2d(ctx, attn_3d, b.n_embd_head_v * n_head_l, b.n_tokens);
    b.cb(gated, names[4], il);
    return gated;
}

/// Ports the constructor of `llama_model_step35`'s `graph`
/// (models/step35.cpp:195 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    // The MTP/NextN layers load as extra decoder blocks, but the main pass
    // does not run them.
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        const n_head_l: i64 = hp.nHead(@intCast(il));
        const n_head_kv_l: i64 = hp.nHeadKv(@intCast(il));

        const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
        const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

        var cur = inpL;

        // The input of the attention norm, to pinpoint layer-boundary issues.
        b.cb(cur, "attn_norm_in", il);

        // Self-attention.
        {
            cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);
            var Qcur = b.buildLoraMm(layer.wq.?, cur, null);
            var Kcur = b.buildLoraMm(layer.wk.?, cur, null);
            var Vcur = b.buildLoraMm(layer.wv.?, cur, null);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            Qcur = c.ggml_reshape_3d(ctx, Qcur, b.n_embd_head_k, n_head_l, b.n_tokens);
            Kcur = c.ggml_reshape_3d(ctx, Kcur, b.n_embd_head_k, n_head_kv_l, b.n_tokens);
            Vcur = c.ggml_reshape_3d(ctx, Vcur, b.n_embd_head_v, n_head_kv_l, b.n_tokens);

            // The per-head Q and K RMS norms (Step35's q_norm and k_norm).
            if (layer.attn_q_norm != null) {
                Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
                b.cb(Qcur, "Qcur_normed", il);
            }
            if (layer.attn_k_norm != null) {
                Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
                b.cb(Kcur, "Kcur_normed", il);
            }

            // RoPE, with partial rotary factors per layer.
            const is_swa = hp.isSwa(@intCast(il));
            const rope_factors = if (is_swa) null else model.getRopeFactors(b.cparams, @intCast(il));
            const n_rot_l: i64 = hp.nRot(@intCast(il));
            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(n_rot_l), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(n_rot_l), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            b.cb(Qcur, "Qcur_pos", il);
            b.cb(Kcur, "Kcur_pos", il);

            const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(b.n_embd_head_k));
            var attn_out = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(attn_out, "attn_out", il);
            // The head-wise attention gate, sigmoid(g_proj(x)) in torch.
            if (layer.wqkv_gate) |g| {
                attn_out = headGate(b, g, cur, attn_out, n_head_l, .{ "attn_gate", "attn_gate_sigmoid", "attn_gate_3d", "attn_gated_3d", "attn_gated" }, il);
            }

            // The output projection.
            cur = b.buildLoraMm(layer.wo.?, attn_out, layer.wo_s);
            b.cb(cur, "attn_proj", il);
        }

        if (il == b.n_layer - 1 and b.cparams.embeddings_nextn_masked) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // The feed-forward.
        if (layer.ffn_gate_inp == null) {
            // The dense MLP.
            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The routed experts.
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
            }, il);
            b.cb(moe_out, "ffn_moe_out", il);

            // The shared-expert MLP, always added on Step35's MoE layers.
            const sh_out = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
            b.cb(sh_out, "ffn_shared_out", il);

            cur = c.ggml_add(ctx, moe_out, sh_out);
            b.cb(cur, "ffn_out", il);
        }
        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (!b.cparams.embeddings_nextn_masked) {
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_step35`'s `graph_mtp`
/// (models/step35.cpp:370 @c1d0e7a00): the `LLM_GRAPH_TYPE_DECODER_MTP`
/// draft head for Step3p5.
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, "models/step35.cpp", 372, "hparams.n_layer_nextn > 0 && \"STEP35 MTP requires n_layer_nextn > 0\"");

    // Multi-block MTP: the graph runs the MTP head that
    // cparams.nextn_layer_offset selects (0 is the first trained head). The
    // speculative driver bumps the offset per draft step to chain heads
    // 45, 46 and 47; offset 0 behaves as a single block.
    const il: c_int = @as(c_int, @intCast(hp.nLayer())) + b.cparams.nextn_layer_offset;
    impl.assert(b.cparams.nextn_layer_offset >= 0 and b.cparams.nextn_layer_offset < @as(c_int, @intCast(hp.n_layer_nextn)), "models/step35.cpp", 381, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range [0, n_layer_nextn)\"");
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, "models/step35.cpp", 384, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, "models/step35.cpp", 385, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, "models/step35.cpp", 386, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");

    const n_head_l: i64 = hp.nHead(@intCast(il));
    const n_head_kv_l: i64 = hp.nHeadKv(@intCast(il));

    const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
    const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

    const inp = impl.allocator.create(graph.InpEmbd) catch impl.abort("models/step35.cpp", 394, "out of memory");
    inp.* = .{ .n_embd = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.n_embd, b.n_tokens);
    c.ggml_set_input(inp.embd);
    _ = c.ggml_set_name(inp.embd, "mtp_h_input");

    const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;

    const h_input = inp.embd.?;
    const tok_embd: *Tensor = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    b.cb(tok_embd, "mtp_tok_embd", il);

    _ = b.res.addInput(graph.InpEmbd, inp);

    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKvIswa();

    const h_norm = b.buildNorm(h_input, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat: *Tensor = c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, null);
    b.cb(cur, "mtp_eh_proj", il);

    const inpSA = cur;

    // mtp_block: a full Step3p5 decoder layer, attention with the optional
    // head-wise gate, then the MoE or dense feed-forward.
    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    var Qcur = b.buildLoraMm(layer.wq.?, cur, layer.wq_s);
    var Kcur = b.buildLoraMm(layer.wk.?, cur, layer.wk_s);
    var Vcur = b.buildLoraMm(layer.wv.?, cur, layer.wv_s);
    b.cb(Qcur, "mtp_Qcur", il);
    b.cb(Kcur, "mtp_Kcur", il);
    b.cb(Vcur, "mtp_Vcur", il);

    Qcur = c.ggml_reshape_3d(ctx, Qcur, b.n_embd_head_k, n_head_l, b.n_tokens);
    Kcur = c.ggml_reshape_3d(ctx, Kcur, b.n_embd_head_k, n_head_kv_l, b.n_tokens);
    Vcur = c.ggml_reshape_3d(ctx, Vcur, b.n_embd_head_v, n_head_kv_l, b.n_tokens);

    if (layer.attn_q_norm != null) {
        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        b.cb(Qcur, "mtp_Qcur_normed", il);
    }
    if (layer.attn_k_norm != null) {
        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
        b.cb(Kcur, "mtp_Kcur_normed", il);
    }

    const is_swa = hp.isSwa(@intCast(il));
    const rope_factors = if (is_swa) null else model.getRopeFactors(b.cparams, @intCast(il));
    const n_rot_l: i64 = hp.nRot(@intCast(il));

    Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(n_rot_l), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(n_rot_l), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    b.cb(Qcur, "mtp_Qcur_pos", il);
    b.cb(Kcur, "mtp_Kcur_pos", il);

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(b.n_embd_head_k));
    var attn_out = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(attn_out, "mtp_attn_out", il);

    // The head-wise attention gate, sigmoid(g_proj(x)).
    if (layer.wqkv_gate) |g| {
        attn_out = headGate(b, g, cur, attn_out, n_head_l, .{ "mtp_attn_gate", "mtp_attn_gate_sigmoid", "mtp_attn_gate_3d", "mtp_attn_gated_3d", "mtp_attn_gated" }, il);
    }

    cur = b.buildLoraMm(layer.wo.?, attn_out, layer.wo_s);
    b.cb(cur, "mtp_attn_proj", il);

    cur = c.ggml_add(ctx, cur, inpSA);
    b.cb(cur, "mtp_attn_residual", il);

    const ffn_inp = cur;
    cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "mtp_ffn_norm", il);

    // The feed-forward, a dense MLP or a mixture of experts as in the trunk.
    if (layer.ffn_gate_inp == null) {
        cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
        b.cb(cur, "mtp_ffn_out", il);
    } else {
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
        }, il);
        b.cb(moe_out, "mtp_ffn_moe_out", il);

        const sh_out = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
        b.cb(sh_out, "mtp_ffn_shared_out", il);

        cur = c.ggml_add(ctx, moe_out, sh_out);
        b.cb(cur, "mtp_ffn_out", il);
    }
    cur = c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "mtp_post_ffn", il);

    const inp_out_ids = b.buildInpOutIds();
    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);

    // The hidden state before the norm, which seeds the next MTP step of the
    // autoregressive draft loop.
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    impl.assert(head_norm_w != null, "models/step35.cpp", 549, "head_norm_w && \"STEP35 MTP: missing both nextn.shared_head_norm and output_norm\"");
    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    impl.assert(head_w != null, "models/step35.cpp", 554, "head_w && \"STEP35 MTP: missing LM head (nextn.shared_head_head or model.output)\"");
    cur = b.buildLoraMm(head_w.?, cur, null);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
