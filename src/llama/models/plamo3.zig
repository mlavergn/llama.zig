//! PLaMo 3: sandwich-normed decoder blocks -- RMS norms before and after
//! attention and the feed-forward -- with a fused QKV projection, RMS-normed
//! query and key heads, a SwiGLU feed-forward from one fused up projection,
//! and seven sliding-window layers for every global one.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/plamo3.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_model_plamo3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph<iswa>`.

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

/// Ports `llama_model_plamo3`'s `load_arch_hparams` (models/plamo3.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    var found_swa = false;
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| {
        hp.n_swa = v;
        found_swa = true;
    }
    if (found_swa and hp.n_swa > 0) {
        hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
        var swa_period: u32 = 8;
        if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |v| swa_period = v;
        hp.setSwaPattern(swa_period, false);
    } else {
        hp.swa_type = @backingInt(hparams_mod.SwaType.none);
    }

    self.type = switch (hp.nLayer()) {
        24 => .@"2b",
        else => .unknown,
    };
}

/// Ports `llama_model_plamo3`'s `load_arch_tensors` (models/plamo3.cpp:22 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const head_dim_q: i64 = hp.nEmbdHeadK(0);
    const head_dim_v: i64 = hp.nEmbdHeadV(0);

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_not_required, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        const num_attention_heads: i64 = hp.nHead(@intCast(i));
        const num_key_value_heads: i64 = hp.nHeadKv(@intCast(i));
        const q_proj_dim = num_attention_heads * head_dim_q;
        const k_proj_dim = num_key_value_heads * head_dim_q;
        const v_proj_dim = num_key_value_heads * head_dim_v;
        const n_ff_cur: i64 = hp.nFf(@intCast(i));

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_proj_dim + k_proj_dim + v_proj_dim }, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{head_dim_q}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{head_dim_q}, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ num_attention_heads * head_dim_v, l.n_embd }, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_cur * 2 }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ n_ff_cur, l.n_embd }, 0, exc);
    }
}

/// Ports `llama_model_plamo3`'s `build_arch_graph` (models/plamo3.cpp:62 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    if (b.hparams.swa_type != @backingInt(hparams_mod.SwaType.none)) {
        buildGraph(true, model, b);
    } else {
        buildGraph(false, model, b);
    }
}

/// Ports the constructor of `llama_model_plamo3`'s `graph<iswa>`
/// (models/plamo3.cpp:71 @b29c606e2).
fn buildGraph(comptime iswa: bool, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const head_dim_q: i64 = hp.nEmbdHeadK(0);
    const head_dim_v: i64 = hp.nEmbdHeadV(0);

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();

    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var residual = inpL;

        const freq_base_l: f32 = if (iswa) model.getRopeFreqBase(b.cparams, @intCast(il)) else b.freq_base;
        const freq_scale_l: f32 = if (iswa) model.getRopeFreqScale(b.cparams, @intCast(il)) else b.freq_scale;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        const qkv = b.buildLoraMm(layer.wqkv.?, cur, null);
        // The C++ names `cur` here, not `qkv`.
        b.cb(cur, "wqkv", il);

        const n_head: i32 = @intCast(hp.nHead(@intCast(il)));
        const n_head_kv: i32 = @intCast(hp.nHeadKv(@intCast(il)));

        const q_offset: i64 = 0;
        const k_offset: i64 = head_dim_q * n_head;
        const v_offset: i64 = k_offset + head_dim_q * n_head_kv;

        const elt: usize = c.ggml_element_size(qkv);
        var Qcur: *Tensor = c.ggml_view_3d(ctx, qkv, head_dim_q, n_head, b.n_tokens, @as(usize, @intCast(head_dim_q)) * @sizeOf(f32), qkv.nb[1], @as(usize, @intCast(q_offset)) * elt);
        var Kcur: *Tensor = c.ggml_view_3d(ctx, qkv, head_dim_q, n_head_kv, b.n_tokens, @as(usize, @intCast(head_dim_q)) * @sizeOf(f32), qkv.nb[1], @as(usize, @intCast(k_offset)) * elt);
        const Vcur: *Tensor = c.ggml_view_3d(ctx, qkv, head_dim_v, n_head_kv, b.n_tokens, @as(usize, @intCast(head_dim_v)) * @sizeOf(f32), qkv.nb[1], @as(usize, @intCast(v_offset)) * elt);

        b.cb(Qcur, "Qcur", il);
        b.cb(Kcur, "Kcur", il);
        b.cb(Vcur, "Vcur", il);

        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        b.cb(Qcur, "attn_q_norm", il);
        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
        b.cb(Kcur, "attn_k_norm", il);

        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        const attn_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(head_dim_q));

        cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, attn_scale, il);
        b.cb(cur, "attn_out", il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            residual = c.ggml_get_rows(ctx, residual, inp_out_ids);
        }

        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        cur = c.ggml_add(ctx, cur, residual);
        b.cb(cur, "attn_residual", il);

        residual = cur;

        cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, null, null, null, layer.ffn_down, null, null, null, .swiglu, .seq, il);
        b.cb(cur, "ffn_out", il);

        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, il);
        b.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx, cur, residual);
        b.cb(cur, "ffn_residual", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
