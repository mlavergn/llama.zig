//! EAGLE-3: a one-layer speculative-decoding draft that reads its target
//! model's hidden features.
//!
//! The encoder fuses three of the target's layers -- low, middle and high,
//! concatenated -- through one projection, `fc`, into the draft's width;
//! the decoder takes those fused features with each draft token's
//! embedding, normalizes both, attends over their concatenation, and
//! projects to a draft vocabulary that `d2t` may map back onto the target's.
//! The token embeddings and the LM head may be the draft's own or the
//! target's, reached through `ctx_other`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/eagle3.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_eagle3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph<is_enc>` with `build_inp_embd_enc`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const context_mod = @import("../context.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("models/eagle3.cpp", 0, "out of memory");
}

/// Ports `llama_model_eagle3`'s `load_arch_hparams` (models/eagle3.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    const func = "load_arch_hparams";

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    const ids = (try meta.getArrList(i32, .target_layers, false, exc)) orelse {
        return exc.throw(.runtime_error, "EAGLE3 model requires 'extract_layers' in GGUF metadata", .{});
    };
    defer allocator.free(ids);
    self.target_layer_ids.clearRetainingCapacity();
    self.target_layer_ids.appendSlice(allocator, ids) catch oom();
    if (self.target_layer_ids.items.len != 3) {
        return exc.throw(.runtime_error, "EAGLE3 requires exactly 3 entries in 'extract_layers'", .{});
    }
    const t = self.target_layer_ids.items;
    impl.logInfo("%s: EAGLE3 extract_layers = [%d, %d, %d]\n", .{ func, t[0], t[1], t[2] });

    const n_embd_tgt = (try meta.getKey(u32, .target_hidden_size, true, exc)).?;
    impl.logInfo("%s: EAGLE3 n_embd_tgt = %u (draft n_embd = %u)\n", .{ func, n_embd_tgt, hp.n_embd });

    hp.n_embd_inp_enc_impl = @as(u32, @truncate(self.target_layer_ids.items.len)) *% n_embd_tgt;

    // norm_before_residual is optional and false by default; Red Hat's
    // EAGLE-3 speculators set it.
    if (try meta.getKey(bool, .norm_before_residual, false, exc)) |v| hp.norm_before_residual = v;
    if (hp.norm_before_residual) {
        // Upstream's message, typo included.
        impl.logInfo("%s: EAGLE3gnorm_before_residual = true\n", .{func});
    }

    // norm_before_fc is optional and false by default; EAGLE-3.1 sets it
    // (e.g. nvidia/gpt-oss-120b-Eagle3-v3).
    if (try meta.getKey(bool, .norm_before_fc, false, exc)) |v| hp.norm_before_fc = v;

    self.type = .unknown;
}

/// Ports `llama_model_eagle3`'s `load_arch_tensors` (models/eagle3.cpp:38 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const func = "load_arch_tensors";

    const n_embd_inp: i64 = hp.nEmbdInpEnc();
    const n_embd_attn_input = 2 * l.n_embd;

    // The draft vocabulary's size comes from d2t, the draft-to-target
    // mapping, when the file has one -- only when the draft's vocabulary
    // differs from the target's.
    var n_draft_vocab = l.n_vocab;
    if (ml.getTensorMeta("d2t")) |d2t_meta| {
        n_draft_vocab = d2t_meta.ne[0];
        self.d2t = try self.createTensor(.{ .tensor = .d2t }, &.{n_draft_vocab}, 0, exc);
        impl.logInfo("%s: EAGLE3 using d2t mapping (draft_vocab_size = %lld)\n", .{ func, @as(c_longlong, n_draft_vocab) });
    } else {
        self.d2t = null;
        impl.logInfo("%s: EAGLE3 without d2t - sharing same vocab_size with target (vocab_size = %lld)\n", .{ func, @as(c_longlong, n_draft_vocab) });
    }

    // The feature fusion layer: the three target layers projected to the
    // draft's width.
    self.fc = try self.createTensor(.{ .tensor = .fc, .suffix = "weight" }, &.{ n_embd_inp, l.n_embd }, 0, exc);

    // An RMS norm on the fused target features, fc's input, only with
    // norm_before_fc.
    if (hp.norm_before_fc) {
        self.output_norm_enc = try self.createTensor(.{ .tensor = .enc_output_norm, .suffix = "weight" }, &.{n_embd_inp}, 0, exc);
    }

    // The output, over the draft vocabulary.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, n_draft_vocab }, nr, exc);

    // The token embeddings are optional: Llama 3.3 70B's EAGLE-3 has its own.
    const tn_embd: loader.Tn = .{ .tensor = .token_embd, .suffix = "weight" };
    const tn_embd_name = tn_embd.str();
    if (ml.getTensorMeta(tn_embd_name.slice())) |tok_embd_meta| {
        const n_target_vocab = tok_embd_meta.ne[1];
        self.tok_embd = try self.createTensor(tn_embd, &.{ l.n_embd, n_target_vocab }, 0, exc);
        impl.logInfo("%s: EAGLE3 using its own token_embd (vocab = %lld)\n", .{ func, @as(c_longlong, n_target_vocab) });
    }

    // The single decoder layer.
    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        // input_layernorm, applied to the token embeddings.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // EAGLE-3's hidden_norm, applied to the fused target features.
        layer.attn_norm_2 = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // Attention takes the normed input embeddings and the normed fused
        // target features, concatenated.
        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ n_embd_attn_input, l.n_embd_head_k * l.n_head }, 0, exc);
        layer.wk = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = i }, &.{ n_embd_attn_input, l.n_embd_k_gqa }, 0, exc);
        layer.wv = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = i }, &.{ n_embd_attn_input, l.n_embd_v_gqa }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

        // Llama 3's RoPE frequency factors, only when the EAGLE-3 config
        // has rope_scaling.
        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr, exc);
    }
}

/// Ports `llama_model_eagle3`'s `build_arch_graph` (models/eagle3.cpp:103 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    switch (params.gtype) {
        .encoder => buildGraphEnc(model, b),
        .default, .decoder => buildGraphDec(model, b),
        else => impl.abort("models/eagle3.cpp", 111, "invalid graph type"),
    }
}

/// `llama_get_model(cparams.ctx_other)`: the target model a draft shares
/// weights with.
fn otherModel(b: *const graph.Builder) *const Model {
    const other: *const context_mod.Context = @ptrCast(@alignCast(b.cparams.ctx_other.?));
    return other.getModel();
}

/// Ports `llama_model_eagle3`'s `graph<true>`'s `build_inp_embd_enc`
/// (models/eagle3.cpp:116 @c1d0e7a00): the target model's features, three
/// layers concatenated, given as the batch's embeddings.
fn buildInpEmbdEnc(b: *const graph.Builder) *Tensor {
    const hp = b.hparams;

    const inp_target = allocator.create(graph.InpEmbd) catch oom();
    inp_target.* = .{ .n_embd = hp.nEmbdInpEnc() };
    inp_target.embd = c.ggml_new_tensor_2d(b.ctx0, c.GGML_TYPE_F32, hp.nEmbdInpEnc(), b.n_tokens);
    c.ggml_set_input(inp_target.embd);

    const cur = inp_target.embd.?;
    b.cb(cur, "inp_embd", -1);

    _ = b.res.addInput(graph.InpEmbd, inp_target);

    return cur;
}

/// Ports the constructor of `llama_model_eagle3`'s `graph<true>`
/// (models/eagle3.cpp:137 @c1d0e7a00): the encoder, which fuses the target
/// model's features -- e.g. `[12288, n_tokens]` from its low, middle and
/// high layers -- into `g_embeddings`, e.g. `[4096, n_tokens]`.
fn buildGraphEnc(model: *const Model, b: *const graph.Builder) void {
    var cur = buildInpEmbdEnc(b);

    // An RMS norm on the fused target features before fc.
    if (b.hparams.norm_before_fc) {
        cur = b.buildNorm(cur, model.output_norm_enc, null, .rms, -1);
        b.cb(cur, "enc_input_norm", -1);
    }

    // The feature fusion layer.
    cur = b.buildLoraMm(model.fc.?, cur, null);
    b.cb(cur, "fc_out", -1);

    // The g_embeddings go to t_h_nextn, as MTP's do, so they can be read
    // through llama_get_embeddings_nextn(ctx_dft).
    c.ggml_set_output(cur);
    b.res.t_h_nextn = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_eagle3`'s `graph<false>`
/// (models/eagle3.cpp:164 @c1d0e7a00): the decoder, over draft tokens and
/// the encoder's g_embeddings, to draft logits.
fn buildGraphDec(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/eagle3.cpp", 167, "n_embd_head == hparams.n_embd_head_k()");
    // EAGLE-3 has a single decoder layer.
    impl.assert(b.n_layer == 1, "models/eagle3.cpp", 168, "n_layer == 1");

    // The decoder's two inputs: the token embeddings -- EAGLE-3's own, as
    // Llama 3.3 70B's has, or the target model's, as Llama 3.1 8B's uses --
    // and the encoder's g_embeddings.
    var tok_embd = model.tok_embd;
    if (model.tok_embd == null) {
        impl.assert(b.cparams.ctx_other != null, "models/eagle3.cpp", 178, "cparams.ctx_other != nullptr");
        const model_other = otherModel(b);

        impl.assert(model_other.tok_embd != null, "models/eagle3.cpp", 181, "model_other->tok_embd != nullptr && \"EAGLE3 decoder requires token embeddings (own or from target model)\"");
        tok_embd = model_other.tok_embd;
    }

    const inp = allocator.create(graph.InpEmbd) catch oom();
    inp.* = .{ .n_embd = b.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, b.n_embd, b.n_tokens);
    c.ggml_set_input(inp.embd);

    const inp_embd: *Tensor = c.ggml_get_rows(ctx, tok_embd, inp.tokens);
    b.cb(inp_embd, "inp_embd", -1);

    const inp_g = inp.embd.?;
    b.cb(inp_g, "inp_g_embeddings", -1);

    _ = b.res.addInput(graph.InpEmbd, inp);

    var inpL = inp_g;

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    // The single decoder layer.
    const il: c_int = 0;
    {
        const layer = &model.layers[@intCast(il)];

        // input_layernorm on the token embeddings.
        const embd_norm = b.buildNorm(inp_embd, layer.attn_norm, null, .rms, il);
        b.cb(embd_norm, "embd_norm", il);

        // hidden_norm on inp_g.
        const g_norm = b.buildNorm(inp_g, layer.attn_norm_2, null, .rms, -1);
        b.cb(g_norm, "g_norm", il);

        // norm_before_residual decides what goes into the residual
        // connection, for Red Hat's EAGLE-3 speculators: false, the
        // default, takes the raw inp_g; true takes g_norm. inpL is the
        // concatenated input.
        const inpSA = if (hp.norm_before_residual) g_norm else inpL;

        // The normed inp_embd and inp_g, concatenated. Upstream passes the
        // layer index as the dimension, which is 0 here.
        var cur: *Tensor = c.ggml_concat(ctx, embd_norm, g_norm, il);
        b.cb(cur, "concat_embd", il);

        // Self-attention over the concatenated input.
        var Qcur = b.buildLoraMm(layer.wq.?, cur, null);
        b.cb(Qcur, "Qcur", il);

        var Kcur = b.buildLoraMm(layer.wk.?, cur, null);
        b.cb(Kcur, "Kcur", il);

        var Vcur = b.buildLoraMm(layer.wv.?, cur, null);
        b.cb(Vcur, "Vcur", il);

        Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
        Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
        Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

        // RoPE frequency factors, null when there are none.
        const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));

        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        b.cb(Qcur, "Qcur_rope", il);
        b.cb(Kcur, "Kcur_rope", il);

        cur = b.buildAttn(inp_attn, layer.wo, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);

        // The residual, updated.
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The FFN norm on the sum.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "post_attn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        // The output, with the residual.
        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "eagle3_prenorm", il);

        inpL = cur;
    }

    var cur = inpL;

    // The pre-norm state, the next token's g_embeddings in autoregressive
    // generation.
    c.ggml_set_output(cur);
    b.res.t_h_nextn = cur;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    // The LM head, to the draft vocabulary: the draft's own, or the target
    // model's when it has none.
    var output = model.output;
    if (output == null) {
        impl.assert(b.cparams.ctx_other != null, "models/eagle3.cpp", 311, "cparams.ctx_other != nullptr");
        const model_other = otherModel(b);

        impl.assert(model_other.output != null, "models/eagle3.cpp", 314, "model_other->output != nullptr && \"EAGLE3 decoder requires an output projection (own or from target model)\"");
        output = model_other.output;
    }
    cur = b.buildLoraMm(output.?, cur, null);

    if (model.d2t) |d2t| {
        const n_draft_vocab = cur.ne[0];
        const n_outputs = cur.ne[1];
        const n_vocab: i64 = model.vocab.nTokens();

        impl.assert(d2t.type == c.GGML_TYPE_I64, "models/eagle3.cpp", 324, "model.d2t->type == GGML_TYPE_I64");
        impl.assert(d2t.ne[0] == n_draft_vocab, "models/eagle3.cpp", 325, "model.d2t->ne[0] == n_draft_vocab");

        const logits: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, 1, n_vocab, n_outputs), -std.math.inf(f32));
        cur = c.ggml_set_rows(ctx, logits, c.ggml_reshape_3d(ctx, cur, 1, n_draft_vocab, n_outputs), c.ggml_reshape_3d(ctx, d2t, n_draft_vocab, 1, 1));
        cur = c.ggml_reshape_2d(ctx, cur, n_vocab, n_outputs);
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
