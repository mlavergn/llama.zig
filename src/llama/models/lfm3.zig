//! LFM3: Liquid's hybrid of gated short convolutions and grouped-query
//! attention layers, the attention interleaving full and sliding-window
//! layers with Q/K norms and a per-head sigmoid gate on its output, each
//! followed by a SwiGLU feed-forward, with optional norms after both.
//!
//! A decision model (d1) reads the prompt in one batch without a memory:
//! the full-attention layers read it both ways, the sliding-window layers
//! stay causal, and the short convolution runs from a zero state.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/lfm3.cpp` at v0.8.0 (`8642541b2`):
//! `llama_model_lfm3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph<decision>` with the lambda inside it.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const memory_hybrid_iswa = @import("../memory_hybrid_iswa.zig");
const lfm_base = @import("lfm_base.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_lfm3`'s `load_arch_hparams` (models/lfm3.cpp:4 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.n_shortconv_l_cache = (try meta.getKey(u32, .shortconv_l_cache, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    var il: u32 = 0;
    while (il < hp.nLayer()) : (il += 1) {
        hp.is_recr_impl[il] = @intFromBool(hp.nHeadKv(il) == 0);
    }

    // LFM3 always mixes full and sliding-window attention layers.
    _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);
    if (!hp.isSwaAny()) {
        return exc.throw(.runtime_error, "LFM3 requires sliding window attention layers", .{});
    }
    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;

    hp.rope_freq_base_train_swa = (try meta.getKey(f32, .rope_freq_base_swa, true, exc)).?;

    // d1: the full-attention layers read the prompt both ways, the
    // sliding-window layers stay causal.
    if (!hp.causal_attn) {
        hp.non_causal_type = hparams_mod.non_causal_type_dense_only;
    }

    self.type = switch (hp.nFf(0)) {
        7680 => .@"1_7b",
        else => .unknown,
    };
}

/// Ports `llama_model_lfm3`'s `load_arch_tensors` (models/lfm3.cpp:33 @cd959bae9).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var rope_freqs_flag: u32 = 0;

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const ui: u32 = @intCast(i);

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        // Absent when `use_sandwich_norm` is false.
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        if ((layer.attn_post_norm == null) != (layer.ffn_post_norm == null)) {
            return exc.throw(.runtime_error, "LFM3 requires both post norms or neither", .{});
        }

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

        if (!hp.isRecr(ui)) {
            // `n_embd_head_k * n_head` may differ from `n_embd`.
            const n_embd_q = l.n_embd_head_k * l.n_head;
            const n_embd_k: i64 = hp.nEmbdKGqa(ui);
            const n_embd_v: i64 = hp.nEmbdVGqa(ui);
            if (n_embd_k != n_embd_v) {
                return exc.throw(.runtime_error, "LFM3 requires equal key and value sizes", .{});
            }

            layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
            layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

            try self.createTensorQkv(layer, i, l.n_embd, n_embd_q, n_embd_k, n_embd_v, 0, exc);

            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_q, l.n_embd }, 0, exc);

            layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_head }, 0, exc);

            if (!hp.isSwa(ui)) {
                layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_embd_head_k, 2)}, rope_freqs_flag, exc);
                rope_freqs_flag = dup;
            }
        } else {
            layer.shortconv.conv = try self.createTensor(.{ .tensor = .shortconv_conv, .suffix = "weight", .bid = i }, &.{ hp.n_shortconv_l_cache, l.n_embd }, 0, exc);
            layer.shortconv.in_proj = try self.createTensor(.{ .tensor = .shortconv_inproj, .suffix = "weight", .bid = i }, &.{ l.n_embd, 3 * l.n_embd }, 0, exc);
            layer.shortconv.out_proj = try self.createTensor(.{ .tensor = .shortconv_outproj, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        }
    }
}

/// Ports `llama_model_lfm3`'s `build_arch_graph` (models/lfm3.cpp:91 @cd959bae9).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    if (!model.hparams.causal_attn) {
        return buildGraph(true, model, b);
    }
    buildGraph(false, model, b);
}

/// The constructor's `build_attn_block` lambda (models/lfm3.cpp:101
/// @8642541b2): Q/K norms, RoPE on the full layers through `rope_freqs`,
/// attention, and a per-head sigmoid gate before the output projection.
fn buildAttnBlock(model: *const Model, b: *const graph.Builder, cur_in: *Tensor, inp_pos: *Tensor, inp_attn: anytype, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const uil: u32 = @intCast(il);
    const layer = &model.layers[uil];

    const n_embd_head: i64 = hp.nEmbdHeadK(uil);
    const n_head_il: i64 = hp.nHead(uil);
    const n_head_kv: i64 = hp.nHeadKv(uil);

    const n_rot_l: c_int = @intCast(hp.nRot(uil));
    const freq_base_l = model.getRopeFreqBase(b.cparams, uil);
    const freq_scale_l = model.getRopeFreqScale(b.cparams, uil);

    const rope_factors: ?*Tensor = if (hp.isSwa(uil)) null else layer.rope_freqs;

    const qkv = b.buildQkv(layer, cur_in, n_embd_head, n_head_il, n_head_kv, il);
    var q = qkv.q;
    var k = qkv.k;
    const v = qkv.v;

    q = b.buildNorm(q, layer.attn_q_norm, null, .rms, il);
    b.cb(q, "model.layers.{}.self_attn.q_layernorm", il);
    k = b.buildNorm(k, layer.attn_k_norm, null, .rms, il);
    b.cb(k, "model.layers.{}.self_attn.k_layernorm", il);

    q = c.ggml_rope_ext(ctx, q, inp_pos, rope_factors, n_rot_l, b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    k = c.ggml_rope_ext(ctx, k, inp_pos, rope_factors, n_rot_l, b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    var attn_out = b.buildAttn(inp_attn, null, null, null, q, k, v, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
    b.cb(attn_out, "attn_out", il);

    var gate = b.buildLoraMm(layer.wqkv_gate.?, cur_in, null);
    gate = c.ggml_sigmoid(ctx, gate);
    b.cb(gate, "model.layers.{}.self_attn.gate", il);

    var attn_3d: *Tensor = c.ggml_reshape_3d(ctx, attn_out, n_embd_head, n_head_il, b.n_tokens);
    const gate_3d: *Tensor = c.ggml_reshape_3d(ctx, gate, 1, n_head_il, b.n_tokens);
    attn_3d = c.ggml_mul(ctx, attn_3d, gate_3d);
    attn_out = c.ggml_reshape_2d(ctx, attn_3d, n_embd_head * n_head_il, b.n_tokens);

    const out = b.buildLoraMm(layer.wo.?, attn_out, layer.wo_s);
    b.cb(out, "model.layers.{}.self_attn.out_proj", il);

    return out;
}

/// Ports the constructor of `llama_model_lfm3`'s `graph<decision>`
/// (models/lfm3.cpp:99 @8642541b2).
fn buildGraph(comptime decision: bool, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var cur = b.buildInpEmbd(model.tok_embd.?);
    b.cb(cur, "model.embed_tokens", -1);

    c.ggml_build_forward_expand(b.gf, cur);

    // decision: no memory, the whole prompt is read in one batch.
    const inp_hybrid = if (decision) {} else b.buildInpMemHybridIswa();
    const inp_attn = if (decision) b.buildAttnInpNoCache() else {};

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var prev_cur = cur;

        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "model.layers.{}.operator_norm", il);

        if (hp.isRecr(@intCast(il))) {
            if (decision) {
                cur = lfm_base.buildLfmShortconvNoState(model, b, cur, il);
            } else {
                const mctx_recr = b.mctx.?.cast(memory_hybrid_iswa.HybridIswaContext).getRecr();
                cur = lfm_base.buildLfmShortconv(model, b, mctx_recr, inp_hybrid.getRecr(), cur, il);
            }
        } else if (decision) {
            cur = buildAttnBlock(model, b, cur, inp_pos, inp_attn, il);
        } else {
            cur = buildAttnBlock(model, b, cur, inp_pos, inp_hybrid.getAttn(), il);
        }

        if (layer.attn_post_norm != null) {
            cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
            b.cb(cur, "model.layers.{}.post_operator_norm", il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            prev_cur = c.ggml_get_rows(ctx, prev_cur, inp_out_ids);
        }

        cur = c.ggml_add(ctx, prev_cur, cur);

        const ffn_norm_out = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        var ffn_out = lfm_base.buildLfmDenseFfn(model, b, ffn_norm_out, il);

        if (layer.ffn_post_norm != null) {
            ffn_out = b.buildNorm(ffn_out, layer.ffn_post_norm, null, .rms, il);
            b.cb(ffn_out, "model.layers.{}.post_ffn_norm", il);
        }

        cur = c.ggml_add(ctx, cur, ffn_out);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);
    }

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
