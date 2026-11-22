//! Falcon: parallel attention and feed-forward over one shared LayerNorm
//! (two for Falcon-40B), RoPE, multi-query attention through a fused QKV
//! projection, and a GELU feed-forward without biases.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/falcon.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_falcon`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_falcon`'s `load_arch_hparams` (models/falcon.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_eps = (try ml.meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        32 => .@"7b",
        60 => .@"40b",
        else => .unknown,
    };
}

/// Ports `llama_model_falcon`'s `load_arch_tensors` (models/falcon.cpp:13 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const nr = llama.loader.tensor_not_required;
    const dup = llama.loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output_norm_b = try self.createTensor(.{ .tensor = .output_norm, .suffix = "bias" }, &.{l.n_embd}, 0, exc);

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        // It needs to be on the GPU.
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_b = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.attn_norm_2 = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.attn_norm_2_b = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd + 2 * l.n_embd_gqa }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);

        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_falcon`'s `build_arch_graph` (models/falcon.cpp:46 @b1b770d90)
/// and the constructor of its `graph` (models/falcon.cpp:50 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/falcon.cpp", 53, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/falcon.cpp", 54, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        var attn_norm = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .norm, il);
        b.cb(attn_norm, "attn_norm", il);

        // Self-attention.
        var cur: *llama.impl.llama_c.ggml_tensor = undefined;
        {
            if (layer.attn_norm_2 != null) {
                // Falcon-40B.
                cur = b.buildNorm(inpL, layer.attn_norm_2, layer.attn_norm_2_b, .norm, il);
                b.cb(cur, "attn_norm_2", il);
            } else {
                cur = attn_norm;
            }

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            // NeoX mode, mode 2.
            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
        }

        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = llama.impl.llama_c.ggml_get_rows(ctx, inpL, inp_out_ids);
            attn_norm = llama.impl.llama_c.ggml_get_rows(ctx, attn_norm, inp_out_ids);
        }

        const ffn_inp = cur;

        // The feed-forward network, on the attention norm, not the result.
        {
            cur = b.buildFfn(attn_norm, layer.ffn_up, null, null, null, null, null, layer.ffn_down, null, null, null, .gelu, .seq, il);
            b.cb(cur, "ffn_out", il);
        }

        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);
        cur = llama.impl.llama_c.ggml_add(ctx, cur, inpL);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    // Norm.
    cur = b.buildNorm(cur, model.output_norm, model.output_norm_b, .norm, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
