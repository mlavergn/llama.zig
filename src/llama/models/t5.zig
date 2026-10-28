//! T5: an encoder-decoder transformer with RMS norm, learned relative
//! position buckets in place of positional embeddings, and a ReLU
//! feed-forward (flan-T5: a gated GELU one). The encoder attends over the
//! whole input without a cache; the decoder attends to itself through a
//! KV cache and to the encoder's output through cross-attention.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/t5.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_t5`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and both specialisations of its `graph<is_enc>`.
//! `llama_model_t5encoder` (`t5encoder.zig`) reuses the encoder graph.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Layer = model_mod.Layer;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_t5`'s `load_arch_hparams` (models/t5.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_rel_attn_bkts = (try meta.getKey(u32, .attention_relative_buckets_count, true, exc)).?;

    if (try meta.getKey(u32, .decoder_start_token_id, false, exc)) |dec_start_token_id| {
        hp.dec_start_token_id = @bitCast(dec_start_token_id);
    }

    hp.dec_n_layer = hp.nLayer();
    if (try meta.getKey(u32, .decoder_block_count, false, exc)) |v| hp.dec_n_layer = v;

    self.type = switch (hp.nLayer()) {
        6 => .@"60m", // t5-small
        8 => .@"80m", // flan-t5-small
        12 => switch (hp.nFf(0)) {
            3072 => .@"220m", // t5-base
            2048 => .@"250m", // flan-t5-base
            else => .unknown,
        },
        24 => switch (hp.nFf(0)) {
            4096 => .@"770m", // t5-large
            2816 => .@"780m", // flan-t5-large
            16384 => .@"3b", // t5-3b
            5120 => .@"3b", // flan-t5-xl
            65536 => .@"11b", // t5-11b
            10240 => .@"11b", // flan-t5-xxl
            else => .unknown,
        },
        else => .unknown,
    };
}

/// Ports `llama_model_t5`'s `load_arch_tensors` (models/t5.cpp:38 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    const n_rel_attn_bkts: i64 = hp.n_rel_attn_bkts;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm_enc = try self.createTensor(.{ .tensor = .enc_output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .dec_output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    // n_layer is the number of encoder layers, dec_n_layer the number of
    // decoder layers.
    const dec_n_layer: i32 = @intCast(hp.dec_n_layer);
    if (dec_n_layer > l.n_layer) {
        resizeLayers(self, @intCast(dec_n_layer));
    }

    // The encoder layers.
    try loadEncoderLayers(self, l, n_rel_attn_bkts, exc);

    // The decoder layers.
    var i: i32 = 0;
    while (i < dec_n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .dec_attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_rel_b = try self.createTensor(.{ .tensor = .dec_attn_rel_b, .suffix = "weight", .bid = i }, &.{ l.n_head, n_rel_attn_bkts }, nr, exc);

        layer.wq = try self.createTensor(.{ .tensor = .dec_attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        layer.wk = try self.createTensor(.{ .tensor = .dec_attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        layer.wv = try self.createTensor(.{ .tensor = .dec_attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_v_gqa }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .dec_attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_v_gqa, l.n_embd }, 0, exc);

        layer.attn_norm_cross = try self.createTensor(.{ .tensor = .dec_cross_attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        // This tensor seems to be unused in the HF transformers implementation.
        layer.attn_rel_b_cross = try self.createTensor(.{ .tensor = .dec_cross_attn_rel_b, .suffix = "weight", .bid = i }, &.{ l.n_head, n_rel_attn_bkts }, nr | loader.tensor_skip_if_virtual, exc);

        layer.wq_cross = try self.createTensor(.{ .tensor = .dec_cross_attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        layer.wk_cross = try self.createTensor(.{ .tensor = .dec_cross_attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        layer.wv_cross = try self.createTensor(.{ .tensor = .dec_cross_attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_v_gqa }, 0, exc);
        layer.wo_cross = try self.createTensor(.{ .tensor = .dec_cross_attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_v_gqa, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .dec_ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .dec_ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .dec_ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .dec_ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// The resize of `layers` to the decoder's layer count in `llama_model_t5`'s
/// `load_arch_tensors`: the new layers start empty, the existing ones keep
/// their tensors.
fn resizeLayers(self: *Model, n: usize) void {
    const old = self.layers.len;
    self.layers = impl.allocator.realloc(self.layers, n) catch impl.abort("models/t5.cpp", 59, "out of memory");
    if (n > old) @memset(self.layers[old..], .{});
}

/// The encoder layers of `llama_model_t5`'s `load_arch_tensors`
/// (models/t5.cpp:38 @5266f24da), which `llama_model_t5encoder`'s repeats
/// unchanged.
pub fn loadEncoderLayers(self: *Model, l: Model.Locals, n_rel_attn_bkts: i64, exc: *Exception) Thrown!void {
    const nr = loader.tensor_not_required;

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm_enc = try self.createTensor(.{ .tensor = .enc_attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_rel_b_enc = try self.createTensor(.{ .tensor = .enc_attn_rel_b, .suffix = "weight", .bid = i }, &.{ l.n_head, n_rel_attn_bkts }, nr, exc);

        layer.wq_enc = try self.createTensor(.{ .tensor = .enc_attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        layer.wk_enc = try self.createTensor(.{ .tensor = .enc_attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        layer.wv_enc = try self.createTensor(.{ .tensor = .enc_attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_v_gqa }, 0, exc);
        layer.wo_enc = try self.createTensor(.{ .tensor = .enc_attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_v_gqa, l.n_embd }, 0, exc);

        layer.ffn_norm_enc = try self.createTensor(.{ .tensor = .enc_ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate_enc = try self.createTensor(.{ .tensor = .enc_ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr, exc);
        layer.ffn_down_enc = try self.createTensor(.{ .tensor = .enc_ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up_enc = try self.createTensor(.{ .tensor = .enc_ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_t5`'s `build_arch_graph` (models/t5.cpp:109
/// @5266f24da): the encoder graph for an encode, the decoder graph
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    switch (params.gtype) {
        .encoder => buildEncoder(model, b),
        .default, .decoder => buildDecoder(model, b),
        else => impl.abort("models/t5.cpp", 117, "invalid graph type"),
    }
}

/// Ports the constructor of `llama_model_t5`'s `graph` specialised for the
/// decoder (models/t5.cpp:122 @5266f24da).
fn buildDecoder(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/t5.cpp", 126, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const embd_enc = b.buildInpCrossEmbd();
    const pos_bucket_dec = b.buildInpPosBucketDec();

    const n_outputs_enc: i64 = embd_enc.ne[1];

    const inp_attn_self = b.buildAttnInpKv();
    const inp_attn_cross = b.buildAttnInpCross();

    const inp_out_ids = b.buildInpOutIds();

    const dec_n_layer: c_int = @intCast(hp.dec_n_layer);

    var il: c_int = 0;
    while (il < dec_n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const inpSA = inpL;

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);

            const attn_rel_b = layer.attn_rel_b orelse model.layers[0].attn_rel_b;
            const kq_b = b.buildPosBias(pos_bucket_dec, attn_rel_b.?);

            cur = b.buildAttn(inp_attn_self, layer.wo, layer.wo_b, layer.wo_s, qkv.q, qkv.k, qkv.v, kq_b, null, null, 1.0, il);
            b.cb(cur, "kqv_out", il);
        }
        cur = c.ggml_add(ctx, cur, inpSA);
        b.cb(cur, "cross_inp", il);

        var inpCA = cur;

        // Norm.
        cur = b.buildNorm(cur, layer.attn_norm_cross, null, .rms, il);
        b.cb(cur, "attn_norm_cross", il);

        // Cross-attention.
        {
            var Qcur = b.buildLoraMm(layer.wq_cross.?, cur, null);
            b.cb(Qcur, "Qcur", il);

            var Kcur = b.buildLoraMm(layer.wk_cross.?, embd_enc, null);
            b.cb(Kcur, "Kcur", il);

            var Vcur = b.buildLoraMm(layer.wv_cross.?, embd_enc, null);
            b.cb(Vcur, "Vcur", il);

            Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
            Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, n_outputs_enc);
            Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, n_outputs_enc);

            cur = b.buildAttn(inp_attn_cross, layer.wo_cross, null, null, Qcur, Kcur, Vcur, null, null, null, 1.0, il);
            b.cb(cur, "kqv_out", il);
        }
        if (il == dec_n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpCA = c.ggml_get_rows(ctx, inpCA, inp_out_ids);
        }
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpCA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        {
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            // T5 uses ReLU, flan-T5 a gated GELU.
            const gated = layer.ffn_gate != null;
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, if (gated) .gelu else .relu, if (gated) .par else .seq, il);
            b.cb(cur, "ffn_out", il);
        }
        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;
    b.cb(cur, "result_embd", -1);

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_t5`'s `graph` specialised for the
/// encoder (models/t5.cpp:277 @5266f24da), which `llama_model_t5encoder`
/// also builds.
pub fn buildEncoder(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/t5.cpp", 280, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const pos_bucket_enc = b.buildInpPosBucketEnc();

    const inp_attn = b.buildAttnInpNoCache();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm_enc, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            var Qcur = b.buildLoraMm(layer.wq_enc.?, cur, null);
            b.cb(Qcur, "Qcur", il);

            var Kcur = b.buildLoraMm(layer.wk_enc.?, cur, null);
            b.cb(Kcur, "Kcur", il);

            var Vcur = b.buildLoraMm(layer.wv_enc.?, cur, null);
            b.cb(Vcur, "Vcur", il);

            Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
            Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
            Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

            const attn_rel_b = layer.attn_rel_b_enc orelse model.layers[0].attn_rel_b_enc;
            const kq_b = b.buildPosBias(pos_bucket_enc, attn_rel_b.?);

            cur = b.buildAttn(inp_attn, layer.wo_enc, null, null, Qcur, Kcur, Vcur, kq_b, null, null, 1.0, il);
            b.cb(cur, "kqv_out", il);
        }
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        {
            cur = b.buildNorm(ffn_inp, layer.ffn_norm_enc, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            // T5 uses ReLU, flan-T5 a gated GELU.
            const gated = layer.ffn_gate_enc != null;
            cur = b.buildFfn(cur, layer.ffn_up_enc, null, null, layer.ffn_gate_enc, null, null, layer.ffn_down_enc, null, null, null, if (gated) .gelu else .relu, if (gated) .par else .seq, il);
            b.cb(cur, "ffn_out", il);
        }
        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;
    b.cb(cur, "result_embd", -1);

    cur = b.buildNorm(cur, model.output_norm_enc, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
