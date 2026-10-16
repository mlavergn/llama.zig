//! MPT: pre-norm decoder blocks with LayerNorm and optional biases, ALiBi
//! or learned position embeddings, a fused and optionally clamped QKV
//! projection, optional Q/K LayerNorm and a GELU feed-forward.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mpt.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_mpt`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_mpt`'s `load_arch_hparams` (models/mpt.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;
    if (try meta.getKey(f32, .attention_clamp_kqv, false, exc)) |v| hp.f_clamp_kqv = v;
    if (try meta.getKey(f32, .attention_max_alibi_bias, false, exc)) |v| hp.f_max_alibi_bias = v;

    self.type = switch (hp.nLayer()) {
        32 => .@"7b",
        48 => .@"30b",
        else => .unknown,
    };
}

/// Ports `llama_model_mpt`'s `load_arch_tensors` (models/mpt.cpp:15 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.pos_embd = try self.createTensor(.{ .tensor = .pos_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_ctx_train }, nr, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output_norm_b = try self.createTensor(.{ .tensor = .output_norm, .suffix = "bias" }, &.{l.n_embd}, nr, exc);

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        // It needs to be on the GPU.
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_b = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd + 2 * l.n_embd_gqa }, 0, exc);
        layer.wqkv_b = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "bias", .bid = i }, &.{l.n_embd + 2 * l.n_embd_gqa}, nr, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_norm_b = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{l.n_ff}, nr, exc);

        // FIXME: test-llama-archs crashes if q_norm is created.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr | loader.tensor_skip_if_virtual, exc);
        layer.attn_q_norm_b = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr | loader.tensor_skip_if_virtual, exc);

        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.attn_k_norm_b = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        // The AWQ ScaleActivation layer.
        layer.ffn_act = try self.createTensor(.{ .tensor = .ffn_act, .suffix = "scales", .bid = i }, &.{l.n_ff}, nr, exc);
    }
}

/// Ports `llama_model_mpt`'s `build_arch_graph` (models/mpt.cpp:63 @c1d0e7a00)
/// and the constructor of its `graph` (models/mpt.cpp:67 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/mpt.cpp", 70, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_attn = b.buildAttnInpKv();

    if (model.pos_embd) |pos_embd| {
        // The positions.
        const inp_pos = b.buildInpPos();
        const pos: *Tensor = c.ggml_get_rows(ctx, pos_embd, inp_pos);
        b.cb(pos, "pos_embd", -1);

        inpL = c.ggml_add(ctx, inpL, pos);
        b.cb(inpL, "inpL", -1);
    }

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        const attn_norm = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .norm, il);
        b.cb(attn_norm, "attn_norm", il);

        // Self-attention.
        var cur = attn_norm;
        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            // Q/K LayerNorm.
            if (layer.attn_q_norm != null) {
                Qcur = c.ggml_reshape_2d(ctx, Qcur, n_embd_head * b.n_head, b.n_tokens);
                Kcur = c.ggml_reshape_2d(ctx, Kcur, n_embd_head * b.n_head_kv, b.n_tokens);

                Qcur = b.buildNorm(Qcur, layer.attn_q_norm, layer.attn_q_norm_b, .norm, il);

                Kcur = b.buildNorm(Kcur, layer.attn_k_norm, layer.attn_k_norm_b, .norm, il);

                Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
                Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
            }

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }

        // Add the input.
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        {
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, .norm, il);
            b.cb(cur, "ffn_norm", il);
            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, null, null, null, layer.ffn_down, layer.ffn_down_b, null, layer.ffn_act, .gelu, .seq, il);
            b.cb(cur, "ffn_out", il);
        }

        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, model.output_norm_b, .norm, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

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
