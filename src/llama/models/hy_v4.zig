//! Hunyuan V4 (HY-V4): absorbed multi-head latent attention with learned
//! sinks and a sigmoid gate on the decompressed output, inside independent
//! hyper-connections (iHC) that carry `hc` residual streams -- DeepSeek-V4's
//! layout without the stream-combining Sinkhorn term -- then leading dense
//! blocks and a mixture of experts with shared experts. Checkpoints with an
//! indexer run DeepSeek sparse attention, its lightning indexer owned by the
//! "full" layers and its top-k reused by the "shared" ones after them;
//! the rest attend to every cached key.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/hy-v4.cpp` at v0.4.1 (`b29c606e2`):
//! the view helpers, `llama_model_hy_v4`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, `hy_v4_hc_reduce`, and its
//! `graph` with the hyper-connection, attention and indexer builders. The
//! memory these graphs read is the K-only cache, or with an indexer
//! `kv_cache_dsa.zig`'s.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Builder = graph.Builder;
const Tensor = c.ggml_tensor;

const src = "models/hy-v4.cpp";

// -----------------------------------------------------------------------------
// Views

/// Ports `hy_v4_elem_offset` (models/hy-v4.cpp:12 @b29c606e2).
fn elemOffset(t: *const Tensor, i: i64) usize {
    return c.ggml_row_size(t.type, i);
}

/// Ports `hy_v4_view_1d` (models/hy-v4.cpp:16 @b29c606e2): `ne0`
/// elements from element `j0`.
fn view1d(ctx: *c.ggml_context, t: *Tensor, ne0: i64, j0: i64) *Tensor {
    return c.ggml_view_1d(ctx, t, ne0, elemOffset(t, j0));
}

/// Ports `hy_v4_view_2d` (models/hy-v4.cpp:20 @b29c606e2): `ne0` by
/// `ne1` from element `j0` of each row.
fn view2d(ctx: *c.ggml_context, t: *Tensor, ne0: i64, ne1: i64, j0: i64) *Tensor {
    return c.ggml_view_2d(ctx, t, ne0, ne1, t.nb[1], elemOffset(t, j0));
}

// -----------------------------------------------------------------------------
// Loading

/// Ports `llama_model_hy_v4`'s `load_arch_hparams` (models/hy-v4.cpp:24 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    hp.n_embd_head_k_mla_impl = (try meta.getKey(u32, .attention_key_length_mla, true, exc)).?;
    hp.n_embd_head_v_mla_impl = (try meta.getKey(u32, .attention_value_length_mla, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;

    // The routed experts' SwiGLU logits clamp; the shared and dense experts
    // are not clamped, so `swiglu_clamp_shexp` stays at its zero default.
    _ = try meta.getKeyOrArr(f32, .swiglu_clamp_exp, &hp.swiglu_clamp_exp, hp.n_layer_all, false, exc);

    hp.dsv4_hc_mult = (try meta.getKey(u32, .hyper_connection_count, true, exc)).?;
    hp.dsv4_hc_eps = (try meta.getKey(f32, .hyper_connection_epsilon, true, exc)).?;
    hp.hc_magnitude = (try meta.getKey(f32, .hyper_connection_magnitude, true, exc)).?;

    // DSA is absent on the all-full-attention checkpoints, whose
    // `indexer_top_k` stays 0.
    if (try meta.getKey(u32, .attention_indexer_head_count, false, exc)) |v| hp.indexer_n_head = v;
    if (try meta.getKey(u32, .attention_indexer_key_length, false, exc)) |v| hp.indexer_head_size = v;
    if (try meta.getKey(u32, .attention_indexer_top_k, false, exc)) |v| hp.indexer_top_k = v;

    if (hp.indexer_top_k > 0) {
        // The reference plumbs `rms_norm_eps` into the indexer's k_norm
        // LayerNorm, and `build_norm` reads `f_norm_eps` for `LLM_NORM`.
        hp.f_norm_eps = hp.f_norm_rms_eps;

        if (hp.indexer_n_head == 0 or hp.indexer_head_size <= hp.nRot(0)) {
            return exc.throw(.runtime_error, "hy_v4: bad indexer head count / key length", .{});
        }

        _ = try meta.getKeyOrArr(u32, .attention_indexer_types, &hp.is_indexer_full_impl, hp.nLayer(), false, exc);
        if (!hp.isIndexerFull(0)) {
            return exc.throw(.runtime_error, "hy_v4: layer 0 must own an indexer, nothing precedes it to share", .{});
        }
    }

    impl.assert(hp.isMla(), src, 65, "hparams.is_mla()");

    self.type = .unknown;
}

/// Ports `llama_model_hy_v4`'s `load_arch_tensors` (models/hy-v4.cpp:70 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_v_mla: i64 = hp.nEmbdHeadVMla();
    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    impl.assert(n_embd_head_qk_nope >= 1, src, 77, "n_embd_head_qk_nope >= 1");

    const q_lora_rank: i64 = hp.n_lora_q;
    const kv_lora_rank: i64 = hp.n_lora_kv;
    const n_ff_exp: i64 = hp.nFfExp(0);
    const n_expert_shared: i64 = hp.n_expert_shared;
    const hc: i64 = hp.dsv4_hc_mult;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // The global iHC head, which collapses the streams before the final
    // norm.
    self.hc_head_fn = try self.createTensor(.{ .tensor = .hc_head_fn, .suffix = "weight" }, &.{ hc * l.n_embd, hc }, 0, exc);
    self.hc_head_base = try self.createTensor(.{ .tensor = .hc_head_base, .suffix = "weight" }, &.{hc}, 0, exc);
    self.hc_head_scale = try self.createTensor(.{ .tensor = .hc_head_scale, .suffix = "weight" }, &.{1}, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_sinks = try self.createTensor(.{ .tensor = .attn_sinks, .suffix = "weight", .bid = i }, &.{l.n_head}, 0, exc);

        layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_lora_rank }, 0, exc);
        layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, 0, exc);
        layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, l.n_head * n_embd_head_k_mla }, 0, exc);
        layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ l.n_embd, kv_lora_rank + n_embd_head_qk_rope }, 0, exc);
        layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, 0, exc);
        layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_qk_nope, kv_lora_rank, l.n_head }, 0, exc);
        layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v_mla, l.n_head }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_head * n_embd_head_v_mla, l.n_embd }, 0, exc);
        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_head * n_embd_head_v_mla }, 0, exc);

        // Only the "full" indexer layers ship weights; the "shared" ones
        // reuse their top-k.
        if (hp.indexer_top_k > 0 and hp.isIndexerFull(@intCast(i))) {
            const n_indexer_head: i64 = hp.indexer_n_head;
            const n_embd_indexer: i64 = hp.indexer_head_size;

            layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, n_indexer_head * n_embd_indexer }, 0, exc);
            layer.indexer_attn_k = try self.createTensor(.{ .tensor = .indexer_attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_indexer }, 0, exc);
            layer.indexer_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{n_embd_indexer}, 0, exc);
            layer.indexer_k_norm_b = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "bias", .bid = i }, &.{n_embd_indexer}, 0, exc);
            layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_indexer_head }, 0, exc);
        }

        layer.hc_attn_fn = try self.createTensor(.{ .tensor = .hc_attn_fn, .suffix = "weight", .bid = i }, &.{ hc * l.n_embd, 2 * hc }, 0, exc);
        layer.hc_attn_base = try self.createTensor(.{ .tensor = .hc_attn_base, .suffix = "weight", .bid = i }, &.{2 * hc}, 0, exc);
        layer.hc_attn_scale = try self.createTensor(.{ .tensor = .hc_attn_scale, .suffix = "weight", .bid = i }, &.{2}, 0, exc);
        layer.hc_ffn_fn = try self.createTensor(.{ .tensor = .hc_ffn_fn, .suffix = "weight", .bid = i }, &.{ hc * l.n_embd, 2 * hc }, 0, exc);
        layer.hc_ffn_base = try self.createTensor(.{ .tensor = .hc_ffn_base, .suffix = "weight", .bid = i }, &.{2 * hc}, 0, exc);
        layer.hc_ffn_scale = try self.createTensor(.{ .tensor = .hc_ffn_scale, .suffix = "weight", .bid = i }, &.{2}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        } else {
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr, exc);

            if (l.n_expert == 0) {
                return exc.throw(.runtime_error, "n_expert must be > 0", .{});
            }
            if (l.n_expert_used == 0) {
                return exc.throw(.runtime_error, "n_expert_used must be > 0", .{});
            }

            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);

            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, 0, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
        }
    }
}

/// Ports `llama_model_hy_v4`'s `build_arch_graph` (models/hy-v4.cpp:157 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Hyper-connections

/// Ports `hy_v4_hc_reduce` (models/hy-v4.cpp:163 @b29c606e2): the streams
/// `x[:, i, :]` weighted by `w[i, :]` and summed, `[n_embd, n_tokens]`, in
/// f32 as the reference runs it.
fn hcReduce(ctx: *c.ggml_context, x: *Tensor, w: *Tensor, hc: i64, n_embd: i64, nt: i64, out_type: c.ggml_type) *Tensor {
    const x_f32: *Tensor = c.ggml_cast(ctx, x, c.GGML_TYPE_F32);
    var result: ?*Tensor = null;
    var ih: i64 = 0;
    while (ih < hc) : (ih += 1) {
        const uih: usize = @intCast(ih);
        const xh: *Tensor = c.ggml_view_2d(ctx, x_f32, n_embd, nt, x_f32.nb[2], uih * x_f32.nb[1]);
        const wh: *Tensor = c.ggml_view_2d(ctx, w, 1, nt, w.nb[1], uih * w.nb[0]);
        const cur: *Tensor = c.ggml_mul(ctx, xh, wh);
        result = if (result) |r| c.ggml_add(ctx, r, cur) else cur;
    }
    return c.ggml_cast(ctx, result.?, out_type);
}

/// Ports `llama_model_hy_v4`'s graph's `build_hc_pre`
/// (models/hy-v4.cpp:175 @b29c606e2): an iHC's mixes -- the pre weights it
/// reduces the streams with, here, and the post weights, scaled by
/// `hc_magnitude`, into `post` for `build_hc_post`.
fn buildHcPre(b: *const Builder, x: *Tensor, hc_fn: *Tensor, hc_scale: *Tensor, hc_base: *Tensor, post: *?*Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const hc: i64 = hp.dsv4_hc_mult;
    const nt = x.ne[2];
    impl.assert(x.ne[0] == b.n_embd and x.ne[1] == hc, src, 184, "x->ne[0] == n_embd && x->ne[1] == hc");

    const flat: *Tensor = c.ggml_reshape_2d(ctx, x, hc * b.n_embd, nt);
    const flat_norm: *Tensor = c.ggml_rms_norm(ctx, flat, hp.f_norm_rms_eps);
    // `[2*hc, nt]`.
    const mixes: *Tensor = c.ggml_mul_mat(ctx, hc_fn, flat_norm);
    b.cb(mixes, "hc_mixes", il);

    const scale_pre = view1d(ctx, hc_scale, 1, 0);
    const scale_post = view1d(ctx, hc_scale, 1, 1);
    const base_pre = view1d(ctx, hc_base, hc, 0);
    const base_post = view1d(ctx, hc_base, hc, hc);

    // pre = sigmoid(mixes[:hc]*scale_pre + base_pre) + eps
    var pre = view2d(ctx, mixes, hc, nt, 0);
    pre = c.ggml_mul(ctx, pre, scale_pre);
    pre = c.ggml_add(ctx, pre, base_pre);
    pre = c.ggml_sigmoid(ctx, pre);
    pre = c.ggml_scale_bias(ctx, pre, 1.0, hp.dsv4_hc_eps);
    b.cb(pre, "hc_pre", il);

    // post = magnitude*sigmoid(mixes[hc:2hc]*scale_post + base_post) + eps
    var po = view2d(ctx, mixes, hc, nt, hc);
    po = c.ggml_mul(ctx, po, scale_post);
    po = c.ggml_add(ctx, po, base_post);
    po = c.ggml_sigmoid(ctx, po);
    po = c.ggml_scale(ctx, po, hp.hc_magnitude);
    po = c.ggml_scale_bias(ctx, po, 1.0, hp.dsv4_hc_eps);
    post.* = po;
    b.cb(po, "hc_post_gate", il);

    return hcReduce(ctx, x, pre, hc, b.n_embd, nt, x.type);
}

/// Ports `llama_model_hy_v4`'s graph's `build_hc_post`
/// (models/hy-v4.cpp:217 @b29c606e2): the block's output spread over the
/// streams by `post` and added to the residual streams, in f32 so the
/// rounding does not accumulate across the layers.
///
/// Return: `[n_embd, hc, n_tokens]`, in the residual's type.
fn buildHcPost(b: *const Builder, x: *Tensor, residual: *Tensor, post: *Tensor, il: c_int) *Tensor {
    _ = il;
    const ctx = b.ctx0;
    const hc: i64 = b.hparams.dsv4_hc_mult;
    const nt = x.ne[1];
    impl.assert(x.ne[0] == b.n_embd, src, 225, "x->ne[0] == n_embd");
    impl.assert(residual.ne[1] == hc, src, 226, "residual->ne[1] == hc");

    // post.float() * x.float() + residual.float(), then back to the
    // residual's type.
    const x_f32: *Tensor = c.ggml_cast(ctx, x, c.GGML_TYPE_F32);
    const post_f32: *Tensor = c.ggml_cast(ctx, post, c.GGML_TYPE_F32);
    const res_f32: *Tensor = c.ggml_cast(ctx, residual, c.GGML_TYPE_F32);

    var out: ?*Tensor = null;
    var i: i64 = 0;
    while (i < hc) : (i += 1) {
        const ui: usize = @intCast(i);
        const res_i: *Tensor = c.ggml_view_2d(ctx, res_f32, b.n_embd, nt, res_f32.nb[2], ui * res_f32.nb[1]);
        const post_i: *Tensor = c.ggml_view_2d(ctx, post_f32, 1, nt, post_f32.nb[1], ui * post_f32.nb[0]);
        var cur: *Tensor = c.ggml_add(ctx, res_i, c.ggml_mul(ctx, x_f32, post_i));
        cur = c.ggml_reshape_3d(ctx, cur, b.n_embd, 1, nt);
        out = if (out) |o| c.ggml_concat(ctx, o, cur, 1) else cur;
    }

    return c.ggml_cast(ctx, out.?, residual.type);
}

/// Ports `llama_model_hy_v4`'s graph's `build_hc_head`
/// (models/hy-v4.cpp:248 @b29c606e2): the streams reduced to one for the
/// output head.
fn buildHcHead(b: *const Builder, x: *Tensor, hc_fn: *Tensor, hc_scale: *Tensor, hc_base: *Tensor) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const hc: i64 = hp.dsv4_hc_mult;
    const nt = x.ne[2];

    const flat: *Tensor = c.ggml_reshape_2d(ctx, x, hc * b.n_embd, nt);
    const flat_norm: *Tensor = c.ggml_rms_norm(ctx, flat, hp.f_norm_rms_eps);
    // `[hc, nt]`.
    const mixes: *Tensor = c.ggml_mul_mat(ctx, hc_fn, flat_norm);
    b.cb(mixes, "hc_head_mixes", -1);

    var pre: *Tensor = c.ggml_mul(ctx, mixes, hc_scale);
    pre = c.ggml_add(ctx, pre, hc_base);
    pre = c.ggml_sigmoid(ctx, pre);
    pre = c.ggml_scale_bias(ctx, pre, 1.0, hp.dsv4_hc_eps);
    b.cb(pre, "hc_head_pre", -1);

    return hcReduce(ctx, x, pre, hc, b.n_embd, nt, x.type);
}

// -----------------------------------------------------------------------------
// Attention

/// The MLA projections both attention builders share, from `q` (after
/// `wq_b`) and the block input: the absorbed Q, the compressed K and V.
const MlaQkv = struct { q: *Tensor, k: *Tensor, v: *Tensor };

fn buildMlaQkv(b: *const Builder, layer: *const model_mod.Layer, q: *Tensor, cur: *Tensor, inp_pos: *Tensor, il: c_int) MlaQkv {
    const ctx = b.ctx0;
    const hp = b.hparams;

    const n_embd_head_k: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const kv_lora_rank: u32 = hp.n_lora_kv;
    const n_head_u: usize = @intCast(b.n_head);

    var q_nope: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_nope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * n_head_u, 0);
    var q_pe: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_rope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * n_head_u, c.ggml_row_size(q.type, n_embd_head_qk_nope));

    const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
    var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), 0);
    var k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, n_embd_head_qk_rope, 1, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));

    q_pe = c.ggml_rope_ext(ctx, q_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    b.cb(q_pe, "q_pe", il);
    k_pe = c.ggml_rope_ext(ctx, k_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    b.cb(k_pe, "k_pe", il);

    kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
    b.cb(kv_cmpr, "kv_cmpr", il);

    // MLA absorption: q_nope @ wk_b, into the compressed space.
    q_nope = c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    var q_nope_absorbed: *Tensor = c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
    q_nope_absorbed = c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);

    // RoPE must go first, for the in-place context shift of
    // `build_rope_shift`.
    const Qcur: *Tensor = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);

    kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, b.n_tokens);
    const Kcur: *Tensor = c.ggml_concat(ctx, kv_cmpr, k_pe, 0);

    return .{ .q = Qcur, .k = Kcur, .v = kv_cmpr };
}

/// The gated-MLA tail both attention builders share: the sigmoid gate on
/// the decompressed attention output, then `wo`.
fn buildGatedOut(b: *const Builder, layer: *const model_mod.Layer, attn_: *Tensor, cur: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    var attn = attn_;

    var gate: *Tensor = c.ggml_mul_mat(ctx, layer.wqkv_gate, cur);
    gate = c.ggml_sigmoid(ctx, gate);
    attn = c.ggml_mul(ctx, attn, gate);
    b.cb(attn, "attn_gated", il);

    const out = b.buildLoraMm(layer.wo.?, attn, null);
    b.cb(out, "attn_out", il);

    return out;
}

/// Ports `llama_model_hy_v4`'s graph's `build_attention`
/// (models/hy-v4.cpp:270 @b29c606e2): MLA as MQA over a K-only cache, the
/// gate before `wo`.
fn buildAttention(b: *const Builder, model: *const Model, inp_attn: *graph.InpAttnK, cur: *Tensor, inp_pos: *Tensor, kq_scale: f32, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];

    var q: *Tensor = c.ggml_mul_mat(ctx, layer.wq_a, cur);
    q = b.buildNorm(q, layer.attn_q_a_norm, null, .rms, il);
    q = c.ggml_mul_mat(ctx, layer.wq_b, q);

    const qkv = buildMlaQkv(b, layer, q, cur, inp_pos, il);

    // MLA as MQA; `wo` is applied below, so the gate can sit before it.
    const attn = b.buildAttn(inp_attn, null, null, null, qkv.q, qkv.k, qkv.v, null, layer.attn_sinks, layer.wv_b, kq_scale, il);
    // `[n_head * n_embd_head_v, n_tokens]`.
    b.cb(attn, "attn_kqv", il);

    return buildGatedOut(b, layer, attn, cur, il);
}

/// Ports `llama_model_hy_v4`'s graph's `build_indexer_top_k`
/// (models/hy-v4.cpp:342 @b29c606e2): the lightning indexer -- scores
/// every cached indexer key for each query, stores this ubatch's keys,
/// and picks the top-k positions attention may see.
///
/// Return: the top-k indices, `[n_top_k, n_tokens]`.
fn buildIndexerTopK(b: *const Builder, model: *const Model, inp_attn_dsa: *graph.InpAttnKDsa, cur: *Tensor, qr: *Tensor, inp_pos: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];

    const n_indexer_head: i64 = hp.indexer_n_head;
    const n_embd_indexer: i64 = hp.indexer_head_size;
    const n_embd_indexer_rope: i64 = hp.nRot(0);
    const n_embd_indexer_nope = n_embd_indexer - n_embd_indexer_rope;

    // The nope rows come first, so only the last `n_embd_indexer_rope`
    // rows rotate, as on the MLA path.
    var iq: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_attn_q_b, qr);

    iq = c.ggml_reshape_3d(ctx, iq, n_embd_indexer, n_indexer_head, b.n_tokens);

    iq = c.ggml_rope_ext(ctx, iq, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    iq = c.ggml_rope_set_offset(iq, @intCast(n_embd_indexer_nope));
    b.cb(iq, "indexer_q", il);

    var ik: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_attn_k, cur);

    ik = b.buildNorm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, .norm, il);

    ik = c.ggml_reshape_3d(ctx, ik, n_embd_indexer, 1, b.n_tokens);

    ik = c.ggml_rope_ext(ctx, ik, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    ik = c.ggml_rope_set_offset(ik, @intCast(n_embd_indexer_nope));
    b.cb(ik, "indexer_k", il);

    // The reference applies a Hadamard rotation here, which only helps its
    // FP8 kernels; it is orthogonal, so it does not change q.k and is
    // skipped.

    const mctx_lid = inp_attn_dsa.mctx.getLid();
    const k_idxs_lid = inp_attn_dsa.getKIdxsLid();
    c.ggml_build_forward_expand(b.gf, mctx_lid.cpyK(ctx, ik, k_idxs_lid, il));

    var iw: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_proj, cur);

    ik = mctx_lid.getK(ctx, il);

    const n_stream = ik.ne[3];
    const n_stream_u: usize = @intCast(n_stream);
    iq = c.ggml_view_4d(ctx, iq, iq.ne[0], iq.ne[1], @divTrunc(iq.ne[2], n_stream), n_stream, iq.nb[1], iq.nb[2], iq.nb[3] / n_stream_u, 0);
    iw = c.ggml_view_4d(ctx, iw, iw.ne[0], @divTrunc(iw.ne[1], n_stream), iw.ne[2], n_stream, iw.nb[1], iw.nb[2] / n_stream_u, iw.nb[3] / n_stream_u, 0);

    // Both of the reference's scale factors, folded into the weights
    // before the big score tensor.
    iw = c.ggml_scale(ctx, iw, 1.0 / impl.sqrtf(@floatFromInt(n_embd_indexer * n_indexer_head)));

    var score: *Tensor = undefined;
    if (b.cparams.fused_lid) {
        score = c.ggml_lightning_indexer(ctx, iq, ik, iw, inp_attn_dsa.getKqMaskLid());
        b.cb(score, "indexer_score", il);
        b.res.addFusedNode(.{ .op = .lightning_indexer, .tensor = score, .il = il });
    } else {
        iq = c.ggml_permute(ctx, iq, 0, 2, 1, 3);
        ik = c.ggml_permute(ctx, ik, 0, 2, 1, 3);

        score = c.ggml_mul_mat(ctx, ik, iq);
        score = c.ggml_cont(ctx, c.ggml_permute(ctx, score, 2, 1, 0, 3));
        score = c.ggml_relu(ctx, score);
        score = c.ggml_mul(ctx, score, iw);
        score = c.ggml_sum_rows(ctx, score);
        score = c.ggml_cont(ctx, c.ggml_permute(ctx, score, 2, 1, 0, 3));
        score = c.ggml_add(ctx, score, inp_attn_dsa.getKqMaskLid());
        b.cb(score, "indexer_score", il);
    }

    const n_top_k: u32 = if (score.ne[0] < @as(i64, hp.indexer_top_k)) @intCast(score.ne[0]) else hp.indexer_top_k;

    return c.ggml_cont(ctx, c.ggml_top_k(ctx, score, @intCast(n_top_k)));
}

/// Ports `llama_model_hy_v4`'s graph's `build_attention_dsa`
/// (models/hy-v4.cpp:421 @b29c606e2): `build_attention` over the DSA
/// cache, attending only to the top-k keys -- this layer's own when it
/// owns an indexer, else the last full layer's, kept in `last_top_k`.
fn buildAttentionDsa(b: *const Builder, model: *const Model, inp_attn_dsa: *graph.InpAttnKDsa, cur: *Tensor, inp_pos: *Tensor, last_top_k: *?*Tensor, kq_scale: f32, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];

    var qr: *Tensor = c.ggml_mul_mat(ctx, layer.wq_a, cur);
    qr = b.buildNorm(qr, layer.attn_q_a_norm, null, .rms, il);

    if (hp.isIndexerFull(@intCast(il))) {
        last_top_k.* = buildIndexerTopK(b, model, inp_attn_dsa, cur, qr, inp_pos, il);
        b.cb(last_top_k.*.?, "top_k", il);
    }
    impl.assert(last_top_k.* != null, src, 443, "*last_top_k != nullptr");

    const q: *Tensor = c.ggml_mul_mat(ctx, layer.wq_b, qr);

    const qkv = buildMlaQkv(b, layer, q, cur, inp_pos, il);

    const attn = b.buildAttnDsa(inp_attn_dsa, null, null, null, qkv.q, qkv.k, qkv.v, null, layer.attn_sinks, layer.wv_b, last_top_k.*.?, kq_scale, il);
    b.cb(attn, "attn_kqv", il);

    return buildGatedOut(b, layer, attn, cur, il);
}

// -----------------------------------------------------------------------------
// The graph

/// Ports the constructor of `llama_model_hy_v4`'s `graph`
/// (models/hy-v4.cpp:497 @b29c606e2).
fn buildGraph(model: *const Model, b: *const Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const hc: i64 = hp.dsv4_hc_mult;
    const n_embd_head_k: i64 = hp.nEmbdHeadKMla();
    const kq_scale = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head_k));

    var cur: *Tensor = undefined;

    const is_dsa = hp.indexer_top_k > 0;

    const inp = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();
    const inp_attn: ?*graph.InpAttnK = if (is_dsa) null else b.buildAttnInpK();
    const inp_attn_dsa: ?*graph.InpAttnKDsa = if (is_dsa) b.buildAttnInpKDsa() else null;
    const inp_out_ids = b.buildInpOutIds();

    // The top-k of the last "full" indexer layer, which the "shared" ones
    // after it reuse.
    var last_top_k: ?*Tensor = null;

    // The one embedding, expanded into `hc` parallel residual streams.
    var inpL: *Tensor = c.ggml_reshape_3d(ctx, inp, b.n_embd, 1, b.n_tokens);
    inpL = c.ggml_repeat_4d(ctx, inpL, b.n_embd, hc, b.n_tokens, 1);
    b.cb(inpL, "hc_init", -1);

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var residual = inpL;
        var post: ?*Tensor = null;

        cur = buildHcPre(b, inpL, layer.hc_attn_fn.?, layer.hc_attn_scale.?, layer.hc_attn_base.?, &post, il);
        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        cur = if (is_dsa)
            buildAttentionDsa(b, model, inp_attn_dsa.?, cur, inp_pos, &last_top_k, kq_scale, il)
        else
            buildAttention(b, model, inp_attn.?, cur, inp_pos, kq_scale, il);

        inpL = buildHcPost(b, cur, residual, post.?, il);
        b.cb(inpL, "hc_attn_out", il);

        residual = inpL;
        cur = buildHcPre(b, inpL, layer.hc_ffn_fn.?, layer.hc_ffn_scale.?, layer.hc_ffn_base.?, &post, il);
        cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            const moe_out = b.buildMoeFfn(cur, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = layer.ffn_gate_exps,
                .down_exps = layer.ffn_down_exps.?,
                .exp_probs_b = layer.ffn_exp_probs_b,
                .n_expert = b.n_expert,
                .n_expert_used = b.n_expert_used,
                .type_op = .silu,
                .norm_w = hp.expert_weights_norm,
                .w_scale = hp.expert_weights_scale,
                .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
                .probs_in = null,
                .gate_up_exps = null,
            }, il);
            b.cb(moe_out, "ffn_moe_out", il);

            const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
            b.cb(ffn_shexp, "ffn_shexp", il);

            cur = c.ggml_add(ctx, moe_out, ffn_shexp);
            b.cb(cur, "ffn_out", il);
        }

        inpL = buildHcPost(b, cur, residual, post.?, il);
        b.cb(inpL, "l_out", il);
    }

    // Prune to the requested output rows once, after every stream is done.
    {
        var flat: *Tensor = c.ggml_reshape_2d(ctx, inpL, b.n_embd * hc, b.n_tokens);
        flat = c.ggml_get_rows(ctx, flat, inp_out_ids);
        inpL = c.ggml_reshape_3d(ctx, flat, b.n_embd, hc, b.n_outputs);
    }

    cur = buildHcHead(b, inpL, model.hc_head_fn.?, model.hc_head_scale.?, model.hc_head_base.?);
    b.cb(cur, "hc_head", -1);

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = c.ggml_mul_mat(ctx, model.output, cur);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
