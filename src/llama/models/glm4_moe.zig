//! GLM-4.5 (and Solar Open): pre-norm blocks with optional Q and K norms
//! and a post-attention norm, leading dense blocks, then a sigmoid-routed
//! mixture of experts with a selection bias and shared experts; and an
//! optional NextN/MTP block past the main stack.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/glm4-moe.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_glm4_moe`'s `load_arch_hparams`, `load_arch_tensors` and
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
const Tensor = c.ggml_tensor;

/// Ports `llama_model_glm4_moe`'s `load_arch_hparams` (models/glm4-moe.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, false, exc);

    // The MoE parameters.
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;

    // The expert gating function; GLM-4.5 uses a sigmoid.
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }

    self.type = switch (hp.nLayer()) {
        46 => .@"106b_a12b", // GLM-4.5-Air
        48 => .@"102b_a12b", // Solar Open
        92 => .@"355b_a32b", // GLM-4.5
        else => .unknown,
    };
}

/// Ports `llama_model_glm4_moe`'s `load_arch_tensors` (models/glm4-moe.cpp:28 @5266f24da).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const n_expert_shared: i64 = hp.n_expert_shared;
    const nr = loader.tensor_not_required;

    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    var probe_buf: [64]u8 = undefined;
    const mtp_probe = std.fmt.bufPrint(&probe_buf, "blk.{d}.nextn.eh_proj.weight", .{l.n_layer}) catch unreachable;
    const trunk_only = hp.n_layer_nextn > 0 and ml.getWeight(mtp_probe) == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    var mtp_flags: u32 = if (trunk_only) nr else 0;

    if (!ml.load_mtp) mtp_flags |= loader.tensor_skip;

    impl.assert(hp.n_expert > 0, "models/glm4-moe.cpp", 42, "hparams.n_expert > 0 && \"n_expert must be > 0 for GLM4_MOE MoE layers\"");
    impl.assert(hp.nExpertUsed(0) > 0, "models/glm4-moe.cpp", 43, "hparams.n_expert_used() > 0 && \"n_expert_used must be > 0 for GLM4_MOE MoE layers\"");

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const flags = if (i < l.n_layer) trunk_flags else mtp_flags;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        // GLM-style attention, with biases.
        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, flags, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, flags, exc);

        // The Q and K norms, optional for the GLM-4.5 355B variant.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, nr | flags, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, nr | flags, exc);

        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        // GLM 4.5 is a hybrid: the first `n_layer_dense_lead` layers are
        // dense, the rest a mixture of experts.
        const use_moe = @as(u32, @intCast(i)) >= hp.n_layer_dense_lead;

        if (use_moe) {
            // The mixture-of-experts layers.
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, flags, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, flags, exc);

            // The routed experts.
            const n_ff_exp: i64 = if (hp.nFfExp(0) != 0) hp.nFfExp(0) else @divTrunc(l.n_ff, l.n_expert_used);

            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);

            // The shared expert.
            if (n_expert_shared > 0) {
                const n_ff_shexp = n_ff_exp * n_expert_shared;
                layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, flags, exc);
                layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, flags, exc);
                layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, flags, exc);
            }
        } else {
            // The dense layers, with separate gate and up projections.
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, flags, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
        }

        // The NextN/MTP tensors.
        if (i >= l.n_layer) {
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

            // The optional ones.
            layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr | flags, exc);
            layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr | flags, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr | flags, exc);
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_glm4_moe`'s `build_arch_graph` (models/glm4-moe.cpp:125
/// @5266f24da): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

/// Ports the constructor of `llama_model_glm4_moe`'s `graph_mtp`
/// (models/glm4-moe.cpp:132 @5266f24da).
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, "models/glm4-moe.cpp", 134, "hparams.n_layer_nextn > 0 && \"GLM4_MOE MTP requires n_layer_nextn > 0\"");
    impl.assert(hp.n_layer_nextn == 1, "models/glm4-moe.cpp", 135, "hparams.n_layer_nextn == 1 && \"GLM4_MOE MTP currently only supports a single MTP block\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/glm4-moe.cpp", 138, "n_embd_head == hparams.n_embd_head_k()");

    const il: c_int = @as(c_int, @intCast(hp.nLayer())) + b.cparams.nextn_layer_offset;
    impl.assert(b.cparams.nextn_layer_offset >= 0 and b.cparams.nextn_layer_offset < @as(c_int, @intCast(hp.n_layer_nextn)), "models/glm4-moe.cpp", 143, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range [0, n_layer_nextn)\"");

    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, "models/glm4-moe.cpp", 147, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, "models/glm4-moe.cpp", 148, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, "models/glm4-moe.cpp", 149, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");
    impl.assert(layer.ffn_gate_inp != null, "models/glm4-moe.cpp", 150, "layer.ffn_gate_inp  && \"MTP block missing ffn_gate_inp\"");

    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort("models/glm4-moe.cpp", 152, "out of memory");
    inp.* = .{ .n_embd = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    c.ggml_set_input(inp.embd);

    var tok_embd: *Tensor = undefined;
    if (b.ubatch.token != null) {
        const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;
        tok_embd = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    } else {
        tok_embd = inp.embd.?;
    }
    b.cb(tok_embd, "mtp_tok_embd", il);

    inp.h = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.n_embd, b.n_tokens);
    c.ggml_set_input(inp.h);
    _ = c.ggml_set_name(inp.h, "mtp_h_input");

    const h_embd = inp.h.?;

    _ = b.res.addInput(graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    const inp_attn = b.buildAttnInpKv();

    const h_norm = b.buildNorm(h_embd, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat = c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(cur, "mtp_eh_proj", il);

    const inpSA = cur;

    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
    var Qcur = qkv.q;
    var Kcur = qkv.k;
    const Vcur = qkv.v;

    if (layer.attn_q_norm != null) {
        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        b.cb(Qcur, "mtp_Qcur_normed", il);
    }
    if (layer.attn_k_norm != null) {
        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
        b.cb(Kcur, "mtp_Kcur_normed", il);
    }

    Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    b.cb(Qcur, "mtp_Qcur", il);
    b.cb(Kcur, "mtp_Kcur", il);
    b.cb(Vcur, "mtp_Vcur", il);

    cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
    b.cb(cur, "mtp_attn_out", il);

    const ffn_inp = c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "mtp_ffn_inp", il);

    cur = b.buildNorm(ffn_inp, layer.attn_post_norm, null, .rms, il);
    b.cb(cur, "mtp_post_attn_norm", il);

    const routed_out = b.buildMoeFfn(cur, .{
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
    b.cb(routed_out, "mtp_ffn_moe_out", il);

    const shared_out = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
    b.cb(shared_out, "mtp_ffn_shexp_out", il);

    cur = c.ggml_add(ctx, routed_out, shared_out);
    b.cb(cur, "mtp_ffn_out", il);

    cur = c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "mtp_post_ffn", il);

    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    impl.assert(head_norm_w != null, "models/glm4-moe.cpp", 265, "head_norm_w && \"GLM4_MOE MTP: missing both nextn.shared_head_norm and output_norm\"");

    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
    impl.assert(head_w != null, "models/glm4-moe.cpp", 282, "head_w && \"GLM4_MOE MTP: missing LM head (nextn.shared_head_head or model.output)\"");

    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_glm4_moe`'s `graph`
/// (models/glm4-moe.cpp:291 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/glm4-moe.cpp", 294, "n_embd_head == hparams.n_embd_head_k()");

    var sections: [4]c_int = undefined;
    @memcpy(&sections, hp.rope_sections[0..4]);

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const use_mrope = hp.useMrope();
    if (b.ubatch.embd != null and !use_mrope) {
        // Stopped here rather than give wrong results.
        impl.abort("models/glm4-moe.cpp", 307, "This GGUF does not support multimodal. Please reconvert it.");
    }

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    // The NextN layers are run by `graph_mtp`.
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // The pre-attention norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            // The Q and K norms, when there are any (GLM-4.5 355B).
            if (layer.attn_q_norm != null) {
                Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
                b.cb(Qcur, "Qcur_normed", il);
            }
            if (layer.attn_k_norm != null) {
                Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
                b.cb(Kcur, "Kcur_normed", il);
            }

            if (use_mrope) {
                Qcur = c.ggml_rope_multi(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), &sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                Kcur = c.ggml_rope_multi(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), &sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            } else {
                // Normal RoPE.
                Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            }

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }
        if (il == b.n_layer - 1 and (!b.cparams.embeddings_nextn or b.cparams.embeddings_nextn_masked)) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The post-attention norm.
        cur = b.buildNorm(ffn_inp, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "post_attn_norm", il);

        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            // A dense layer.
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The routed experts.
            const routed_out = b.buildMoeFfn(cur, .{
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
            b.cb(routed_out, "ffn_moe_out", il);

            // The shared expert, on the same input.
            const shared_out = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
            b.cb(shared_out, "ffn_shexp_out", il);

            // The routed output plus the shared one.
            cur = c.ggml_add(ctx, routed_out, shared_out);
            b.cb(cur, "ffn_out", il);
        }
        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;
    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (b.cparams.embeddings_nextn and !b.cparams.embeddings_nextn_masked) {
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

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
