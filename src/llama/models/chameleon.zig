//! Chameleon: Llama's block with layer-normed query and key heads, and
//! optionally Swin-style norms after attention and the feed-forward in
//! place of the pre-norms; the image tokens' logits are suppressed so that
//! it generates text only.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/chameleon.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_chameleon`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_chameleon`'s `load_arch_hparams` (models/chameleon.cpp:4 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_norm_eps = 1e-5; // The QK norm's epsilon, torch's default.
    if (try meta.getKey(bool, .swin_norm, false, exc)) |v| hp.swin_norm = v;

    self.type = switch (hp.nLayer()) {
        32 => .@"7b",
        48 => .@"34b",
        else => .unknown,
    };
}

/// Ports `llama_model_chameleon`'s `load_arch_tensors` (models/chameleon.cpp:16 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k, l.n_head }, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k, l.n_head_kv }, 0, exc);
        layer.attn_q_norm_b = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "bias", .bid = i }, &.{ l.n_embd_head_k, l.n_head }, nr, exc);
        layer.attn_k_norm_b = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "bias", .bid = i }, &.{ l.n_embd_head_k, l.n_head_kv }, nr, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, l.n_embd_gqa, l.n_embd_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_chameleon`'s `build_arch_graph` (models/chameleon.cpp:49 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_chameleon`'s `graph`
/// (models/chameleon.cpp:53 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/chameleon.cpp", 56, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/chameleon.cpp", 57, "n_embd_head == n_rot");

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // Norm.
        var cur: *Tensor = undefined;
        if (hp.swin_norm) {
            cur = inpL;
        } else {
            cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);
        }

        // Self-attention.
        {
            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            if (layer.attn_q_norm != null) {
                Qcur = b.buildNorm(Qcur, layer.attn_q_norm, layer.attn_q_norm_b, .norm, il);
                b.cb(Qcur, "Qcur", il);
            }

            if (layer.attn_k_norm != null) {
                Kcur = b.buildNorm(Kcur, layer.attn_k_norm, layer.attn_k_norm_b, .norm, il);
                b.cb(Kcur, "Kcur", il);
            }

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        if (hp.swin_norm) {
            cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        if (!hp.swin_norm) {
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);
        }

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        if (hp.swin_norm) {
            cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);
        }

        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

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
    b.cb(cur, "result_output_with_img_logits", -1);

    // TODO: this suppresses the output of image tokens, which is required to
    // enable text-only outputs. Needs to be removed once image outputs are
    // supported.
    const img_token_end_idx: c_int = 8196;
    const img_token_start_idx: c_int = 4;
    const num_img_tokens: c_int = img_token_end_idx - img_token_start_idx;
    // A 1d tensor of num_img_tokens values of -FLT_MAX, which keeps every
    // text token's logit above every image token's.
    var img_logits: *Tensor = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, num_img_tokens);
    img_logits = c.ggml_clamp(ctx, img_logits, -std.math.floatMax(f32), -std.math.floatMax(f32));
    b.cb(img_logits, "img_logits", -1);

    cur = c.ggml_set_1d(ctx, cur, img_logits, c.ggml_element_size(cur) * @as(usize, @intCast(img_token_start_idx)));

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
