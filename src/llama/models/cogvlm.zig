//! CogVLM's language model: Llama-style decoder blocks with a second set of
//! QKV, output and feed-forward weights -- the visual expert -- used in
//! place of the text weights when the batch is image embeddings.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/cogvlm.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_cogvlm`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_cogvlm`'s `load_arch_hparams` (src/models/cogvlm.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try ml.meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        32 => .@"13b",
        else => .unknown,
    };
}

/// Ports `llama_model_cogvlm`'s `load_arch_tensors` (src/models/cogvlm.cpp:12 @c1d0e7a00).
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
        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head * 3 }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.visexp_attn_wqkv = try self.createTensor(.{ .tensor = .visexp_attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head * 3 }, 0, exc);
        layer.visexp_attn_wo = try self.createTensor(.{ .tensor = .visexp_attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | (if (i != 0) loader.tensor_duplicated else 0), exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

        layer.visexp_ffn_gate = try self.createTensor(.{ .tensor = .visexp_ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.visexp_ffn_down = try self.createTensor(.{ .tensor = .visexp_ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.visexp_ffn_up = try self.createTensor(.{ .tensor = .visexp_ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_cogvlm`'s `build_arch_graph` (src/models/cogvlm.cpp:49 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_cogvlm`'s `graph` (src/models/cogvlm.cpp:53
/// @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/cogvlm.cpp", 58, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/cogvlm.cpp", 59, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    // Input tokens are text; an input embedding vector is an image.
    const is_text = b.ubatch.token != null;

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        // The text weights, or the image's.
        var wqkv: *Tensor = undefined;
        var wo: *Tensor = undefined;
        var wo_s: ?*Tensor = undefined;
        var ffn_gate: *Tensor = undefined;
        var ffn_down: *Tensor = undefined;
        var ffn_up: *Tensor = undefined;

        if (is_text) {
            wqkv = layer.wqkv.?;
            wo = layer.wo.?;
            wo_s = layer.wo_s;
            ffn_gate = layer.ffn_gate.?;
            ffn_down = layer.ffn_down.?;
            ffn_up = layer.ffn_up.?;
        } else {
            wqkv = layer.visexp_attn_wqkv.?;
            wo = layer.visexp_attn_wo.?;
            wo_s = null;
            ffn_gate = layer.visexp_ffn_gate.?;
            ffn_down = layer.visexp_ffn_down.?;
            ffn_up = layer.visexp_ffn_up.?;
        }

        const inpSA = inpL;
        var cur = b.buildNorm(inpSA, layer.attn_norm, null, .rms, il);

        // Self-attention.
        {
            const qkv = b.buildLoraMm(wqkv, cur, null);

            // Q, K and V, split from the first dimension.
            var Qcur: *Tensor = c.ggml_view_3d(ctx, qkv, n_embd_head, b.n_head, b.n_tokens, @as(usize, @intCast(n_embd_head)) * @sizeOf(f32), qkv.nb[1], 0);
            var Kcur: *Tensor = c.ggml_view_3d(ctx, qkv, n_embd_head, b.n_head_kv, b.n_tokens, @as(usize, @intCast(n_embd_head)) * @sizeOf(f32), qkv.nb[1], @as(usize, @intCast(b.n_embd)) * c.ggml_element_size(qkv));
            const Vcur: *Tensor = c.ggml_view_3d(ctx, qkv, n_embd_head, b.n_head_kv, b.n_tokens, @as(usize, @intCast(n_embd_head)) * @sizeOf(f32), qkv.nb[1], 2 * @as(usize, @intCast(b.n_embd)) * c.ggml_element_size(qkv));

            Qcur = c.ggml_rope(ctx, Qcur, inp_pos, @intCast(n_embd_head), b.rope_type);
            Kcur = c.ggml_rope(ctx, Kcur, inp_pos, @intCast(n_embd_head), b.rope_type);

            cur = b.buildAttn(inp_attn, wo, null, wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, ffn_up, null, null, ffn_gate, null, null, ffn_down, null, null, null, .silu, .par, il);

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
