//! Cohere2 MoE: Cohere's parallel block, where attention and the
//! feed-forward both read the one attention norm and are added to the
//! residual together, over interleaved sliding-window attention with RoPE
//! on the sliding and dense leading layers; a mixture of experts with an
//! averaged shared expert, scaled logits, and a NextN block for multi-token
//! prediction.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/cohere2moe.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_cohere2moe`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, its `graph` and its
//! `graph_mtp`.

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

/// Ports `llama_model_cohere2moe`'s `load_arch_hparams` (models/cohere2moe.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
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
    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    if (try meta.getKey(u32, .expert_shared_count, false, exc)) |v| hp.n_expert_shared = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;

    if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
    impl.assert(hp.n_layer_nextn < hp.n_layer_all, "models/cohere2moe.cpp", 24, "hparams.n_layer_nextn < hparams.n_layer_all && \"n_layer_nextn must be < n_layer\"");

    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |swa_period| {
        hp.setSwaPattern(swa_period, true);
    } else {
        _ = try meta.getKeyOrArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, hp.nLayer(), true, exc);
    }

    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    self.type = switch (hp.nLayer()) {
        49 => .@"30b_a3b",
        else => .unknown,
    };
}

/// Ports `llama_model_cohere2moe`'s `load_arch_tensors` (models/cohere2moe.cpp:48 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    // Trunk-only: the GGUF declares MTP layers in its metadata but the MTP
    // tensors live in a separate file, so they are not required and the
    // trunk loads cleanly.
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
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
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
/// (models/cohere2moe.cpp:82 @c1d0e7a00).
fn loadBlockTrunk(self: *Model, l: *const Model.Locals, i: i32, flags: u32, exc: *Exception) Thrown!void {
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
        const n_ff_exp: i64 = if (hp.n_ff_exp != 0) hp.n_ff_exp else l.n_ff;

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
/// (models/cohere2moe.cpp:110 @c1d0e7a00).
fn loadBlockMtp(self: *Model, l: *const Model.Locals, i: i32, flags: u32, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const layer = &self.layers[@intCast(i)];

    // The MTP block looks like a full-attention Cohere2 MoE decoder block.
    layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

    try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, flags, exc);
    layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, flags, exc);

    const n_ff_exp: i64 = if (hp.n_ff_exp != 0) hp.n_ff_exp else l.n_ff;

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

/// Ports `llama_model_cohere2moe`'s `build_arch_graph` (models/cohere2moe.cpp:153
/// @c1d0e7a00): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_cohere2moe`'s `graph`
/// (models/cohere2moe.cpp:160 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/cohere2moe.cpp", 163, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/cohere2moe.cpp", 164, "n_embd_head == n_rot");

    const cohere2moe_norm_type: graph.NormType = if (hp.f_norm_rms_eps == 0.0) .norm else .rms;
    const f_logit_scale = hp.f_logit_scale;
    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    // The MTP/NextN layers load as extra decoder blocks, but the main pass
    // does not run them.
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const is_swa = hp.isSwa(@intCast(il));
        // The dense-prefix full-attention layers take RoPE; the later ones
        // follow the SWA pattern.
        const force_rope = @as(u32, @intCast(il)) < hp.n_layer_dense_lead;

        var cur = b.buildNorm(inpL, model.layers[@intCast(il)].attn_norm, null, cohere2moe_norm_type, il);
        b.cb(cur, "attn_norm", il);

        var ffn_inp = cur;

        {
            const layer = &model.layers[@intCast(il)];

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            if (is_swa or force_rope) {
                const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));

                Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            }

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }

        if (il == b.n_layer - 1 and b.cparams.embeddings_nextn_masked) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
            ffn_inp = c.ggml_get_rows(ctx, ffn_inp, inp_out_ids);
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

                cur = c.ggml_add(ctx, cur, ffn_shexp);
                cur = c.ggml_scale(ctx, cur, 0.5);
                b.cb(cur, "ffn_out", il);
            }
        }

        cur = c.ggml_add(ctx, cur, inpL);
        cur = c.ggml_add(ctx, cur, attn_out);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = inpL;
    cur = b.buildNorm(cur, model.output_norm, null, cohere2moe_norm_type, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (!b.cparams.embeddings_nextn_masked) {
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, null);

    if (f_logit_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, f_logit_scale);
    }

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_cohere2moe`'s `graph_mtp`
/// (models/cohere2moe.cpp:299 @c1d0e7a00).
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, "models/cohere2moe.cpp", 300, "hparams.n_layer_nextn > 0 && \"COHERE2MOE MTP requires n_layer_nextn > 0\"");
    impl.assert(hp.n_layer_nextn == 1, "models/cohere2moe.cpp", 301, "hparams.n_layer_nextn == 1 && \"COHERE2MOE MTP currently only supports a single MTP block\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/cohere2moe.cpp", 304, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/cohere2moe.cpp", 305, "n_embd_head == n_rot");

    const il: c_int = @intCast(hp.nLayer());
    const layer = &model.layers[@intCast(il)];
    impl.assert(layer.nextn.eh_proj != null, "models/cohere2moe.cpp", 309, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, "models/cohere2moe.cpp", 310, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, "models/cohere2moe.cpp", 311, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");
    impl.assert(layer.ffn_gate_inp != null, "models/cohere2moe.cpp", 312, "layer.ffn_gate_inp  && \"MTP block missing ffn_gate_inp\"");

    const cohere2moe_norm_type: graph.NormType = if (hp.f_norm_rms_eps == 0.0) .norm else .rms;

    // The C++'s TODO: extract into a common `build_inp_embd_h`.
    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort("models/cohere2moe.cpp", 317, "out of memory");
    inp.* = .{ .n_embd = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    c.ggml_set_input(inp.embd);

    // The C++'s TODO: make this static with `ggml_build_forward_select`,
    // as `build_inp_embd` does.
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
    const inp_attn = b.buildAttnInpKvIswa();

    const h_norm = b.buildNorm(h_embd, layer.nextn.hnorm, null, cohere2moe_norm_type, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, cohere2moe_norm_type, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat: *Tensor = c.ggml_concat(ctx, e_norm, h_norm, 0);
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
    const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));
    Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    b.cb(Qcur, "mtp_Qcur", il);
    b.cb(Kcur, "mtp_Kcur", il);
    b.cb(Vcur, "mtp_Vcur", il);

    cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
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

        cur = c.ggml_add(ctx, cur, ffn_shexp);
        cur = c.ggml_scale(ctx, cur, 0.5);
        b.cb(cur, "mtp_ffn_out", il);
    }

    cur = c.ggml_add(ctx, cur, inpL);
    cur = c.ggml_add(ctx, cur, attn_out);
    b.cb(cur, "mtp_post_ffn", il);

    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    impl.assert(head_norm_w != null, "models/cohere2moe.cpp", 426, "head_norm_w && \"COHERE2MOE MTP: missing both nextn.shared_head_norm and output_norm\"");
    cur = b.buildNorm(cur, head_norm_w, null, cohere2moe_norm_type, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    impl.assert(head_w != null, "models/cohere2moe.cpp", 436, "head_w && \"COHERE2MOE MTP: missing LM head (nextn.shared_head_head or model.output)\"");
    cur = b.buildLoraMm(head_w.?, cur, if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else null);

    if (hp.f_logit_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, hp.f_logit_scale);
    }

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
