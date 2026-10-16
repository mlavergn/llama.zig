//! GPT-2: pre-norm decoder blocks with LayerNorm and biases, learned
//! position embeddings, a fused QKV projection and a GELU feed-forward.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/gpt2.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_gpt2`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_gpt2`'s `load_arch_hparams` (models/gpt2.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_eps = (try ml.meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        12 => .small,
        24 => .medium,
        36 => .large,
        48 => .xl,
        else => .unknown,
    };
}

/// Ports `llama_model_gpt2`'s `load_arch_tensors` (models/gpt2.cpp:15 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.pos_embd = try self.createTensor(.{ .tensor = .pos_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_ctx_train }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output_norm_b = try self.createTensor(.{ .tensor = .output_norm, .suffix = "bias" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_b = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd + 2 * l.n_embd_gqa }, 0, exc);
        layer.wqkv_b = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "bias", .bid = i }, &.{l.n_embd + 2 * l.n_embd_gqa}, 0, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_norm_b = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{l.n_ff}, 0, exc);
    }
}

/// Ports `llama_model_gpt2`'s `build_arch_graph` (models/gpt2.cpp:54 @c1d0e7a00)
/// and the constructor of its `graph` (models/gpt2.cpp:58 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/gpt2.cpp", 61, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const pos: *Tensor = c.ggml_get_rows(ctx, model.pos_embd, inp_pos);
    b.cb(pos, "pos_embd", -1);

    inpL = c.ggml_add(ctx, inpL, pos);
    b.cb(inpL, "inpL", -1);

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        var cur = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .norm, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, qkv.q, qkv.k, qkv.v, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
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

            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, null, null, null, layer.ffn_down, layer.ffn_down_b, null, null, .gelu, .seq, il);
            b.cb(cur, "ffn_out", il);
        }

        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, model.output_norm_b, .norm, -1);

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
