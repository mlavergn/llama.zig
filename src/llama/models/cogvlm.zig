//! CogVLM's language model: Llama-style decoder blocks with a second set of
//! QKV, output and feed-forward weights -- the visual expert -- used in
//! place of the text weights when the batch is image embeddings.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/cogvlm.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_cogvlm`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_cogvlm`'s `load_arch_hparams` (src/models/cogvlm.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try ml.meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        32 => .@"13b",
        else => .unknown,
    };
}

/// Ports `llama_model_cogvlm`'s `load_arch_tensors` (src/models/cogvlm.cpp:12 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const nr = llama.loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head * 3 }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.visexp_attn_wqkv = try self.createTensor(.{ .tensor = .visexp_attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head * 3 }, 0, exc);
        layer.visexp_attn_wo = try self.createTensor(.{ .tensor = .visexp_attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | (if (i != 0) llama.loader.tensor_duplicated else 0), exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

        layer.visexp_ffn_gate = try self.createTensor(.{ .tensor = .visexp_ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.visexp_ffn_down = try self.createTensor(.{ .tensor = .visexp_ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.visexp_ffn_up = try self.createTensor(.{ .tensor = .visexp_ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_cogvlm`'s `get_rope_params` (src/models/cogvlm.cpp:49 @b1b770d90).
///
/// Parameters:
/// - `model`: the model.
/// - `cparams`: the context's parameters.
/// - `il`: the layer.
///
/// Return: the RoPE of layer `il`, for the graph and the K-shift alike.
pub fn getRopeParams(model: *const llama.model.Model, cparams: *const llama.cparams.Cparams, il: u32) llama.model.RopeParams {
    var res = model.getRopeParamsBase(cparams, il);

    // A plain ggml_rope.
    res.n_ctx_orig = 0;
    res.freq_base = 10000.0;
    res.freq_scale = 1.0;
    res.ext_factor = 0.0;
    res.attn_factor = 1.0;
    res.beta_fast = 0.0;
    res.beta_slow = 0.0;
    res.factors = null;

    return res;
}

/// Ports `llama_model_cogvlm`'s `build_arch_graph` (src/models/cogvlm.cpp:65 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_cogvlm`'s `graph` (src/models/cogvlm.cpp:69
/// @b1b770d90).
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const kq_scale: f32 = 1.0 / stdlib.libm.sqrtfInt(n_embd_head);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/cogvlm.cpp", 74, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/cogvlm.cpp", 75, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    // Input tokens are text; an input embedding vector is an image.
    b.ubatch.assertEmbdOrToken("models/cogvlm.cpp", 89);
    const is_text = b.ubatch.token != null;

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        // The text weights, or the image's.
        var wqkv: *llama.impl.llama_c.ggml_tensor = undefined;
        var wo: *llama.impl.llama_c.ggml_tensor = undefined;
        var wo_s: ?*llama.impl.llama_c.ggml_tensor = undefined;
        var ffn_gate: *llama.impl.llama_c.ggml_tensor = undefined;
        var ffn_down: *llama.impl.llama_c.ggml_tensor = undefined;
        var ffn_up: *llama.impl.llama_c.ggml_tensor = undefined;

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
            var Qcur: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_view_3d(ctx, qkv, n_embd_head, b.n_head, b.n_tokens, @as(usize, @intCast(n_embd_head)) * @sizeOf(f32), qkv.nb[1], 0);
            var Kcur: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_view_3d(ctx, qkv, n_embd_head, b.n_head_kv, b.n_tokens, @as(usize, @intCast(n_embd_head)) * @sizeOf(f32), qkv.nb[1], @as(usize, @intCast(b.n_embd)) * llama.impl.llama_c.ggml_element_size(qkv));
            const Vcur: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_view_3d(ctx, qkv, n_embd_head, b.n_head_kv, b.n_tokens, @as(usize, @intCast(n_embd_head)) * @sizeOf(f32), qkv.nb[1], 2 * @as(usize, @intCast(b.n_embd)) * llama.impl.llama_c.ggml_element_size(qkv));

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            cur = b.buildAttn(inp_attn, wo, null, wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);
        }

        const ffn_inp: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, ffn_up, null, null, ffn_gate, null, null, ffn_down, null, null, null, .silu, .par, il);

        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);
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
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
