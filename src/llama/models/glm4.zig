//! GLM-4: pre-norm decoder blocks with a second RMS norm after attention
//! and after the feed-forward, before each residual add; RoPE, or
//! multimodal RoPE for GLM-OCR's sections; a SwiGLU feed-forward whose gate
//! and up projections are one tensor. GLM-OCR's NextN layers are loaded and
//! not run.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/glm4.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_glm4`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_glm4`'s `load_arch_hparams` (models/glm4.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, false, exc);

    // The NextN/MTP parameters (GLM-OCR).
    if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
    impl.assert(hp.n_layer_nextn < hp.n_layer_all, "models/glm4.cpp", 9, "hparams.n_layer_nextn < hparams.n_layer_all && \"n_layer_nextn must be < n_layer_impl\"");

    self.type = switch (hp.nLayer()) {
        17 => .@"1b", // GLM-OCR
        40 => .@"9b",
        61 => .@"32b",
        else => .unknown,
    };
}

/// Ports `llama_model_glm4`'s `load_arch_tensors` (models/glm4.cpp:19 @c1d0e7a00).
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
    while (i < l.n_layer_all) : (i += 1) {
        var flags: u32 = 0;
        if (i >= l.n_layer) {
            // Every tensor of the NextN layers is skipped.
            flags |= loader.tensor_skip;
        }

        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, flags, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, flags, exc);

        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, flags, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff * 2 }, flags, exc);

        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        // The NextN/MTP tensors, kept but unused, for the last
        // `nextn_predict_layers` layers.
        if (i >= l.n_layer) {
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

            // Optional.
            layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, flags | nr, exc);
            layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, flags | nr, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags | nr, exc);
        }
    }
}

/// Ports `llama_model_glm4`'s `build_arch_graph` (models/glm4.cpp:68 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_glm4`'s `graph` (models/glm4.cpp:72
/// @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/glm4.cpp", 75, "n_embd_head == hparams.n_embd_head_k()");

    var sections: [4]c_int = undefined;
    @memcpy(&sections, hp.rope_sections[0..4]);

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const use_mrope = hp.useMrope();
    if (b.ubatch.embd != null and !use_mrope) {
        // Stopped here, rather than give wrong results.
        impl.abort("models/glm4.cpp", 88, "This GGUF does not support multimodal. Please reconvert it.");
    }

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    // Up to the last layer only: the NextN layers' tensors are loaded but
    // not run.
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
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        // The post-attention norm.
        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "post_attn_norm", il);

        // Add the input: the residual comes after the post-attention norm.
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        {
            // The pre-MLP norm.
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            // The MLP.
            cur = b.buildFfn(cur, layer.ffn_up, null, null, null, null, null, layer.ffn_down, null, null, null, .swiglu, .seq, il);
            b.cb(cur, "ffn_out", il);

            // The post-MLP norm.
            cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, il);
            b.cb(cur, "post_mlp_norm", il);
        }
        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    // The final norm.
    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The output projection.
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
