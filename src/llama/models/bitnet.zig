//! BitNet b1.58: Llama's block with ternary weights, each projection
//! carrying its own scale, and an extra RMS norm before the attention
//! output and before the feed-forward's down projection.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/bitnet.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_bitnet`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_bitnet`'s `load_arch_hparams` (models/bitnet.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        26 => .@"3b",
        else => .unknown,
    };
}

/// Ports `llama_model_bitnet`'s `load_arch_tensors` (models/bitnet.cpp:12 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_sub_norm = try self.createTensor(.{ .tensor = .attn_sub_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        layer.wq_s = try self.createTensor(.{ .tensor = .attn_q, .suffix = "scale", .bid = i }, &.{1}, nr, exc);
        layer.wk = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_gqa }, 0, exc);
        layer.wk_s = try self.createTensor(.{ .tensor = .attn_k, .suffix = "scale", .bid = i }, &.{1}, nr, exc);
        layer.wv = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_gqa }, 0, exc);
        layer.wv_s = try self.createTensor(.{ .tensor = .attn_v, .suffix = "scale", .bid = i }, &.{1}, nr, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        layer.wo_s = try self.createTensor(.{ .tensor = .attn_out, .suffix = "scale", .bid = i }, &.{1}, nr, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_sub_norm = try self.createTensor(.{ .tensor = .ffn_sub_norm, .suffix = "weight", .bid = i }, &.{l.n_ff}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_gate_s = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "scale", .bid = i }, &.{1}, nr, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_down_s = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "scale", .bid = i }, &.{1}, nr, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up_s = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "scale", .bid = i }, &.{1}, nr, exc);
    }
}

/// Ports `llama_model_bitnet`'s `build_arch_graph` (models/bitnet.cpp:47 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_bitnet`'s `graph`
/// (models/bitnet.cpp:51 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/bitnet.cpp", 54, "n_embd_head == hparams.n_embd_head_k()");

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);

            cur = b.buildNorm(cur, layer.attn_sub_norm, null, .rms, il);
            b.cb(cur, "attn_sub_norm", il);

            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
            if (layer.wo_b) |wo_b| {
                cur = c.ggml_add(ctx, cur, wo_b);
            }
            b.cb(cur, "attn_out", il);
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

        cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, null, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_sub_out", il);

        cur = b.buildNorm(cur, layer.ffn_sub_norm, null, .rms, il);
        b.cb(cur, "ffn_sub_norm", il);

        cur = b.buildLoraMm(layer.ffn_down.?, cur, layer.ffn_down_s);
        b.cb(cur, "ffn_down", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "l_out", il);

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
    // FIXME: do not use model.tok_embd directly, duplicate as model.output
    cur = b.buildLoraMm(model.tok_embd.?, cur, null);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
