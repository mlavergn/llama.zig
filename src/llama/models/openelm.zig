//! OpenELM: pre-norm decoder blocks with RMSNorm whose head and
//! feed-forward counts vary layer by layer, a fused QKV projection with
//! per-head Q/K RMSNorm, RoPE, a SwiGLU feed-forward and a tied output.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/openelm.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_model_openelm`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_openelm`'s `load_arch_hparams` (models/openelm.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try ml.meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        16 => .@"270m",
        20 => .@"450m",
        28 => .@"1b",
        36 => .@"3b",
        else => .unknown,
    };
}

/// Ports `llama_model_openelm`'s `load_arch_tensors` (models/openelm.cpp:15 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const dup = loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    // The output is the input token embedding.
    self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const n_head: i64 = hp.nHead(@intCast(i));
        const n_head_qkv: i64 = 2 * @as(i64, hp.nHeadKv(@intCast(i))) + n_head;
        const n_ff: i64 = hp.nFf(@intCast(i));

        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_head_qkv * l.n_embd_head_k }, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_head * l.n_embd_head_k, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff }, 0, exc);
    }
}

/// Ports `llama_model_openelm`'s `build_arch_graph` (models/openelm.cpp:46 @b29c606e2)
/// and the constructor of its `graph` (models/openelm.cpp:50 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/openelm.cpp", 53, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const n_head: i64 = hp.nHead(@intCast(il));
        const n_head_kv: i64 = hp.nHeadKv(@intCast(il));
        const n_head_qkv = 2 * n_head_kv + n_head;

        var cur = inpL;
        var residual = cur;

        // Norm.
        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            cur = b.buildLoraMm(layer.wqkv.?, cur, null);
            b.cb(cur, "wqkv", il);

            cur = c.ggml_reshape_3d(ctx, cur, b.n_embd_head_k, n_head_qkv, b.n_tokens);

            var Qcur: *Tensor = c.ggml_view_3d(ctx, cur, n_embd_head, n_head, b.n_tokens, cur.nb[1], cur.nb[2], 0);
            b.cb(Qcur, "Qcur", il);

            var Kcur: *Tensor = c.ggml_view_3d(ctx, cur, n_embd_head, n_head_kv, b.n_tokens, cur.nb[1], cur.nb[2], cur.nb[1] * @as(usize, @intCast(n_head)));
            b.cb(Kcur, "Kcur", il);

            const Vcur: *Tensor = c.ggml_view_3d(ctx, cur, n_embd_head, n_head_kv, b.n_tokens, cur.nb[1], cur.nb[2], cur.nb[1] * @as(usize, @intCast(n_head + n_head_kv)));
            b.cb(Vcur, "Vcur", il);

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            b.cb(Qcur, "Qcur", il);

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Kcur, "Kcur", il);

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            // The C names Q's callback "Vcur" here; it is kept.
            b.cb(Qcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }
        if (il == b.n_layer - 1) {
            residual = c.ggml_get_rows(ctx, residual, inp_out_ids);
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
        }
        const ffn_inp: *Tensor = c.ggml_add(ctx, residual, cur);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        {
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        }
        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }
    var cur = inpL;

    // Norm.
    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

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
