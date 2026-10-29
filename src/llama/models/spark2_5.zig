//! Spark 2.5: interleaved sliding-window attention with per-layer head
//! counts and head sizes, a head-wise sigmoid gate on the attention output
//! computed from the attention input, and a GELU-gated feed-forward.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/spark2-5.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_model_spark2_5`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

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

/// Ports `llama_model_spark2_5`'s `load_arch_hparams` (models/spark2-5.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);

    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    self.type = switch (hp.nLayer()) {
        28 => .@"1_7b",
        else => .unknown,
    };
}

/// Ports `llama_model_spark2_5`'s `load_arch_tensors` (models/spark2-5.cpp:20 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    // The output, or failing that the token embeddings (tied embeddings).
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_not_required, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const iu: u32 = @intCast(i);

        const n_head_i: i64 = hp.nHead(iu);
        const n_head_kv_i: i64 = hp.nHeadKv(iu);
        const n_embd_q: i64 = @as(i64, hp.nEmbdHeadK(iu)) * n_head_i;
        const n_embd_k: i64 = @as(i64, hp.nEmbdHeadK(iu)) * n_head_kv_i;
        const n_embd_v: i64 = @as(i64, hp.nEmbdHeadV(iu)) * n_head_kv_i;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        try self.createTensorQkv(layer, i, l.n_embd, n_embd_q, n_embd_k, n_embd_v, 0, exc);
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_head_i }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_q, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
    }
}

/// Ports `llama_model_spark2_5`'s `build_arch_graph` (models/spark2-5.cpp:52 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_spark2_5`'s `graph`
/// (models/spark2-5.cpp:56 @b29c606e2).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/spark2-5.cpp", 59, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(hp.swa_type == @backingInt(hparams_mod.SwaType.standard), "models/spark2-5.cpp", 60, "hparams.swa_type == LLAMA_SWA_TYPE_STANDARD");

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    const kq_scale = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const ilu: u32 = @intCast(il);
        var inpSA = inpL;
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        const n_head_i: i64 = hp.nHead(ilu);
        const n_head_kv_i: i64 = hp.nHeadKv(ilu);
        const n_rot_i: i64 = hp.nRot(ilu);
        const freq_base_i = model.getRopeFreqBase(b.cparams, ilu);
        const freq_scale_i = model.getRopeFreqScale(b.cparams, ilu);

        const attn_inp = cur;
        const qkv = b.buildQkv(layer, cur, n_embd_head, n_head_i, n_head_kv_i, il);
        var Qcur = qkv.q;
        var Kcur = qkv.k;
        const Vcur = qkv.v;

        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(n_rot_i), b.rope_type, b.n_ctx_orig, freq_base_i, freq_scale_i, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(n_rot_i), b.rope_type, b.n_ctx_orig, freq_base_i, freq_scale_i, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        b.cb(Qcur, "Qcur_rope", il);
        b.cb(Kcur, "Kcur_rope", il);

        cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
        b.cb(cur, "attn_out", il);

        // The head-wise gate, from the attention input.
        var gate: *Tensor = b.buildLoraMm(layer.wqkv_gate.?, attn_inp, null);
        gate = c.ggml_sigmoid(ctx, gate);
        b.cb(gate, "attn_gate", il);

        const n_tokens_i = cur.ne[1];
        cur = c.ggml_reshape_3d(ctx, cur, n_embd_head, n_head_i, n_tokens_i);
        gate = c.ggml_reshape_3d(ctx, gate, 1, n_head_i, n_tokens_i);
        cur = c.ggml_mul(ctx, cur, gate);
        cur = c.ggml_reshape_2d(ctx, cur, n_embd_head * n_head_i, n_tokens_i);
        b.cb(cur, "attn_gated", il);

        cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
        b.cb(cur, "attn_out_proj", il);

        if (il == b.n_layer - 1) {
            // Skip computing the output of unused tokens.
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .gelu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, null);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
