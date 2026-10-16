//! Qwen3-VL's language model: Qwen3's decoder blocks with multimodal RoPE,
//! and "deepstack" visual features added to the first layers' outputs.
//! Qwen3-TTS's talker reuses it whole, with a codec head narrower than the
//! vocabulary as its output.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/qwen3vl.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_qwen3vl`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`. `llama_model_qwen3tts`
//! (`llama.cpp/src/models/qwen3tts.cpp`) adds nothing of its own, so the
//! registry gives it these three.

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

/// Ports `llama_model_qwen3vl`'s `load_arch_hparams` (src/models/qwen3vl.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    if (try ml.meta.getKey(u32, .num_deepstack_layers, false, exc)) |v| hp.n_deepstack_layers = v;
    _ = try ml.meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, true, exc);
    hp.f_norm_rms_eps = (try ml.meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        28 => .@"1_7b",
        36 => if (hp.n_embd == 2560) .@"4b" else .@"8b",
        64 => .@"32b",
        else => .unknown,
    };
}

/// Ports `llama_model_qwen3vl`'s `load_arch_tensors` (src/models/qwen3vl.cpp:16 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nr = loader.tensor_not_required;

    var n_vocab_out = l.n_vocab;
    if (self.arch == .qwen3tts) {
        n_vocab_out = 3072;
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, n_vocab_out }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    // The reranker's output head.
    self.cls_out = try self.createTensor(.{ .tensor = .cls_out, .suffix = "weight" }, &.{ l.n_embd, self.hparams.n_cls_out }, nr, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_qwen3vl`'s `build_arch_graph` (src/models/qwen3vl.cpp:55 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_qwen3vl`'s `graph` (src/models/qwen3vl.cpp:59
/// @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_deepstack_layers: usize = hp.n_deepstack_layers;

    const n_embd: i64 = hp.n_embd;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/qwen3vl.cpp", 65, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/qwen3vl.cpp", 66, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    var sections: [4]c_int = undefined;
    @memcpy(&sections, hp.rope_sections[0..4]);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

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

            Qcur = c.ggml_rope_multi(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), &sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Kcur, "Kcur_normed", il);

            Kcur = c.ggml_rope_multi(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), &sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        if (il < @as(c_int, @intCast(n_deepstack_layers))) {
            const t_inp_embd = b.res.t_inp_embd.?;
            const ds: *Tensor = c.ggml_view_2d(ctx, t_inp_embd, n_embd, b.n_tokens, t_inp_embd.nb[1], @as(usize, @intCast((il + 1) * n_embd)) * @sizeOf(f32));
            cur = c.ggml_add(ctx, cur, ds);
            b.cb(cur, "deepstack_out", il);
        }

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    const n_vocab_in = model.tok_embd.?.ne[1];
    const n_vocab_out = model.output.?.ne[1];
    if (n_vocab_in > n_vocab_out) {
        // Qwen3-TTS, with the codec head as its output.
        impl.assert(model.output_norm != null, "models/qwen3vl.cpp", 178, "model.output_norm");
        const pad = n_vocab_in - n_vocab_out;

        // A scalar -inf tensor, by this trick, to pad the output with.
        var neg_inf: *Tensor = c.ggml_scale_bias(ctx, c.ggml_view_1d(ctx, model.output_norm.?, 1, 0), 0.0, -std.math.inf(f32));
        neg_inf = c.ggml_repeat_4d(ctx, neg_inf, pad, cur.ne[1], 1, 1);
        cur = c.ggml_concat(ctx, neg_inf, cur, 0); // [padded .. n_vocab_out, n_stream]
    } else if (n_vocab_in < n_vocab_out) {
        impl.abort("models/qwen3vl.cpp", 189, "invalid case");
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
