//! The Qwen3-TTS generator: the code predictor (backbone hidden state to
//! the 16 RVQ codes of a frame) and code2wav (codes to PCM).
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/qwen3tts-gen.cpp` at v0.4.0
//! (`5266f24da`), and the `PROJECTOR_TYPE_QWEN3TTS_GEN` arm of
//! `clip_get_graph_builder` (`llama.cpp/tools/mtmd/clip.cpp`). Each
//! declaration names what it ports and the line it began at.
//!
//! # The nested builders
//!
//! Upstream's `code_gen` and `code2wav` are `clip_graph`s copied from the
//! parent, sharing its context and graph; here each holds a
//! `graph.Graph.initChild` of the parent. `code2wav::state_in` is a
//! `std::map` read only through `at()`, so a hash map stands in;
//! `state_out` is a vector, iterated in insertion order, and stays a list.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const module = @import("module.zig");
const c = impl.c;
const gpa = impl.gpa;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

extern fn sqrtf(x: f32) f32;
extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;

/// `string_format` into an arena, for the slot and node names built with
/// `std::to_string` upstream.
fn fmtName(a: std.mem.Allocator, comptime fmt: [*:0]const u8, args: anytype) [:0]const u8 {
    return impl.stringFormat(a, fmt, args) catch impl.abort("qwen3tts-gen.cpp", 0, "out of memory");
}

/// `std::map::at` on a missing key: an uncaught `std::out_of_range`.
fn mapAt(m: *const std.StringHashMapUnmanaged(T), key: []const u8) T {
    return m.get(key) orelse impl.abortUncaught("std::out_of_range", "map::at:  key not found");
}

/// Ports `code_gen` (mtmd/models/models.h:261 @5266f24da), nested in
/// `clip_graph_qwen3tts_gen`: the
/// MTP-style code predictor, one token per codebook.
const CodeGen = struct {
    g: graph.Graph,
    top_k: c_int,
    top_p: f32,

    /// Ports `do_sampling` (qwen3tts-gen.cpp:6 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code_gen`: top-k, top-p, then a random draw,
    /// all on the device.
    fn doSampling(self: *const CodeGen, logits_in: T, inp_rand: T) T {
        const ctx0 = self.g.ctx0;
        const logits = c.ggml_reshape_1d(ctx0, logits_in, c.ggml_nelements(logits_in));
        const n_vocab = ne(logits, 0);

        var cur = logits;
        var candidates: T = null; // maps row index back to vocab id

        if (self.top_k > 0 and self.top_k < n_vocab) {
            const idx = c.ggml_top_k(ctx0, cur, self.top_k);
            candidates = idx;
            cur = self.sortBy(cur, idx);
            self.g.cb(cur, "sample_top_k_logits", -1);
        }

        if (self.top_p < 1.0) {
            const sorted_idx = c.ggml_argsort(ctx0, cur, c.GGML_SORT_ORDER_DESC);
            const sorted_logits = self.sortBy(cur, sorted_idx);
            candidates = if (candidates != null) self.sortBy(candidates, sorted_idx) else sorted_idx;

            const probs = c.ggml_soft_max(ctx0, sorted_logits);
            const cdf = c.ggml_cumsum(ctx0, probs);

            // keep_mask[i] = 1 once cdf[i] crosses top_p
            const cdf_scaled = c.ggml_scale_bias(ctx0, cdf, -1.0, self.top_p);
            var keep_mask = c.ggml_step(ctx0, cdf_scaled);
            var idxf = c.ggml_sum(ctx0, keep_mask);
            idxf = c.ggml_clamp(ctx0, idxf, 0.0, @as(f32, @floatFromInt(ne(keep_mask, 0))) - 1);
            const ones = c.ggml_scale_bias(ctx0, idxf, 0.0, 1.0);

            // top-p must include the crossing element, so force it to 1
            var keep_mask_2d = c.ggml_reshape_2d(ctx0, keep_mask, 1, ne(keep_mask, 0));
            keep_mask_2d = c.ggml_set_rows(ctx0, keep_mask_2d, ones, c.ggml_cast(ctx0, idxf, c.GGML_TYPE_I32));
            keep_mask = c.ggml_reshape_1d(ctx0, keep_mask_2d, ne(keep_mask, 0));

            // log(1) = 0 (keep), log(0) = -inf (drop)
            const bias = c.ggml_log(ctx0, keep_mask);
            cur = c.ggml_add(ctx0, sorted_logits, bias);
            self.g.cb(cur, "sample_top_p_logits", -1);
        }

        // draw one token: find where the cdf crosses inp_rand
        const probs = c.ggml_soft_max(ctx0, cur);
        const cumsum = c.ggml_cumsum(ctx0, probs);

        const diff = c.ggml_sub(ctx0, cumsum, inp_rand);
        const cross_mask = c.ggml_step(ctx0, diff);
        const idxf = c.ggml_sum(ctx0, cross_mask);
        var idx = c.ggml_cast(ctx0, c.ggml_scale_bias(ctx0, idxf, -1.0, @floatFromInt(ne(cross_mask, 0))), c.GGML_TYPE_I32);

        if (candidates != null) {
            const cand_2d = c.ggml_reshape_2d(ctx0, candidates, 1, ne(candidates, 0));
            idx = c.ggml_get_rows(ctx0, cand_2d, idx);
        }
        self.g.cb(idx, "sample_token_id", -1);

        return idx;
    }

    /// The `sort_by` lambda of `do_sampling`: `a`'s rows in `idx`'s order.
    fn sortBy(self: *const CodeGen, a: T, idx: T) T {
        const ctx0 = self.g.ctx0;
        const a2d = c.ggml_reshape_2d(ctx0, a, 1, ne(a, 0));
        return c.ggml_reshape_1d(ctx0, c.ggml_get_rows(ctx0, a2d, idx), ne(idx, 0));
    }

    /// Ports `cache_set` (qwen3tts-gen.cpp:71 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code_gen`: a new cache with row `row_idx`
    /// set to `value`.
    fn cacheSet(self: *const CodeGen, cache: T, row_idx: c_int, value: T) T {
        const ctx0 = self.g.ctx0;
        const n_embd = ne(cache, 0);
        const n_cache = ne(cache, 1);
        impl.ggmlAssert(row_idx >= 0 and row_idx < n_cache, "qwen3tts-gen.cpp", 74, "row_idx >= 0 && row_idx < n_cache");

        // append value as the last row, then gather it back into place
        const value_2d = c.ggml_reshape_2d(ctx0, value, n_embd, 1);
        const cache_ext = c.ggml_concat(ctx0, cache, value_2d, 1); // [n_embd, n_cache + 1]

        // gather indices [0..row_idx-1, n_cache, row_idx+1..n_cache-1]
        var idx = self.constI32(cache, @floatFromInt(n_cache));
        if (row_idx > 0) {
            const prefix = c.ggml_cast(ctx0, c.ggml_arange(ctx0, 0.0, @floatFromInt(row_idx), 1.0), c.GGML_TYPE_I32);
            idx = c.ggml_concat(ctx0, prefix, idx, 0);
        }
        if (row_idx < n_cache - 1) {
            const suffix = c.ggml_cast(ctx0, c.ggml_arange(ctx0, @floatFromInt(row_idx + 1), @floatFromInt(n_cache), 1.0), c.GGML_TYPE_I32);
            idx = c.ggml_concat(ctx0, idx, suffix, 0);
        }

        const result = c.ggml_get_rows(ctx0, cache_ext, idx);
        self.g.cb(result, "cache_set_out", -1);
        return result;
    }

    /// Ports `const_i32` (qwen3tts-gen.cpp:98 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code_gen`: a constant with no host upload.
    fn constI32(self: *const CodeGen, anchor: T, value: f32) T {
        const ctx0 = self.g.ctx0;
        var v = c.ggml_view_1d(ctx0, anchor, 1, 0);
        if (v.*.type != c.GGML_TYPE_F32) v = c.ggml_cast(ctx0, v, c.GGML_TYPE_F32);
        return c.ggml_cast(ctx0, c.ggml_scale_bias(ctx0, v, 0.0, value), c.GGML_TYPE_I32);
    }

    /// Ports `causal_mask_row` (qwen3tts-gen.cpp:107 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code_gen`.
    fn causalMaskRow(self: *const CodeGen, n_kv_pad: i64, pos: c_int) T {
        const ctx0 = self.g.ctx0;
        const ones = c.ggml_fill(ctx0, c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_kv_pad, n_kv_pad), 1.0);
        const keep = c.ggml_tri(ctx0, ones, c.GGML_TRI_TYPE_LOWER_DIAG);
        const row = c.ggml_view_1d(ctx0, keep, n_kv_pad, @as(usize, @intCast(pos)) * nb(keep, 1));
        const mask = c.ggml_log(ctx0, row); // 0 = keep, -inf = masked
        return c.ggml_reshape_4d(ctx0, mask, n_kv_pad, 1, 1, 1);
    }

    /// Ports `project_in` (qwen3tts-gen.cpp:116 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code_gen`: talker hidden size to predictor
    /// hidden size.
    fn projectIn(self: *const CodeGen, cur_in: T) T {
        const model = self.g.model;
        if (model.gen_code_proj_in_w == null) return cur_in;
        var cur = c.ggml_mul_mat(self.g.ctx0, model.gen_code_proj_in_w, cur_in);
        if (model.gen_code_proj_in_b != null) cur = c.ggml_add(self.g.ctx0, cur, model.gen_code_proj_in_b);
        return cur;
    }

    /// Ports `layer_forward` (qwen3tts-gen.cpp:128 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code_gen`: one layer at position `pos`,
    /// writing k/v into the caches at row `pos`.
    fn layerForward(self: *const CodeGen, cur_in: T, layer: *const model_mod.Layer, inp_pos: T, kq_mask: T, k_cache_layer: *T, v_cache_layer: *T, n_kv_pad: i64, pos: c_int, il: c_int) T {
        const ctx0 = self.g.ctx0;
        const hparams = self.g.hparams;
        const n_head = hparams.n_head;
        const n_head_kv = hparams.n_head_kv;
        const d_head = @divTrunc(ne(layer.q_w, 1), n_head); // real head_dim, not n_embd / n_head
        const kq_scale = 1.0 / sqrtf(@floatFromInt(d_head));

        const residual = cur_in;

        var h = c.ggml_rms_norm(ctx0, cur_in, hparams.eps);
        h = c.ggml_mul(ctx0, h, layer.ln_1_w);

        var q = c.ggml_mul_mat(ctx0, layer.q_w, h);
        var k = c.ggml_mul_mat(ctx0, layer.k_w, h);
        const v = c.ggml_mul_mat(ctx0, layer.v_w, h);

        q = c.ggml_reshape_3d(ctx0, q, d_head, n_head, 1);
        k = c.ggml_reshape_3d(ctx0, k, d_head, n_head_kv, 1);

        q = c.ggml_rms_norm(ctx0, q, hparams.eps);
        q = c.ggml_mul(ctx0, q, layer.q_norm);
        k = c.ggml_rms_norm(ctx0, k, hparams.eps);
        k = c.ggml_mul(ctx0, k, layer.k_norm);

        q = c.ggml_rope_ext(ctx0, q, inp_pos, null, @intCast(d_head), c.GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        k = c.ggml_rope_ext(ctx0, k, inp_pos, null, @intCast(d_head), c.GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

        // write k/v into the cache at row pos, flat layout
        const k_flat = c.ggml_reshape_1d(ctx0, k, d_head * n_head_kv);
        k_cache_layer.* = self.cacheSet(k_cache_layer.*, pos, k_flat);
        v_cache_layer.* = self.cacheSet(v_cache_layer.*, pos, v);

        const q_cur = c.ggml_reshape_4d(ctx0, q, d_head, n_head, 1, 1);
        const k_cur = c.ggml_reshape_4d(ctx0, k_cache_layer.*, d_head, n_head_kv, n_kv_pad, 1);
        const v_cur = c.ggml_reshape_4d(ctx0, v_cache_layer.*, d_head, n_head_kv, n_kv_pad, 1);

        const attn_out = self.g.buildAttn(layer.o_w, layer.o_b, q_cur, k_cur, v_cur, kq_mask, kq_scale, il, null);

        var cur = c.ggml_add(ctx0, residual, attn_out);

        var h2 = c.ggml_rms_norm(ctx0, cur, hparams.eps);
        h2 = c.ggml_mul(ctx0, h2, layer.ln_2_w);

        const gate = c.ggml_mul_mat(ctx0, layer.ff_gate_w, h2);
        const up = c.ggml_mul_mat(ctx0, layer.ff_up_w, h2);
        const gu = c.ggml_swiglu_split(ctx0, gate, up);
        const down = c.ggml_mul_mat(ctx0, layer.ff_down_w, gu);

        cur = c.ggml_add(ctx0, cur, down);
        return cur;
    }

    /// Ports `prefill` (qwen3tts-gen.cpp:191 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code_gen`: position 0 seeds the cache from
    /// the hidden state; position 1 embeds code0 and samples codebook 1.
    fn prefill(self: *const CodeGen, k_cache: []T, v_cache: []T, out_code_cache: *T, h_state: T, code0_embd: T, inp_rand: T) void {
        const ctx0 = self.g.ctx0;
        const model = self.g.model;
        const hparams = self.g.hparams;
        const n_kv_pad = ne(k_cache[0], 1);

        {
            var cur = self.projectIn(h_state);
            const kq_mask = self.causalMaskRow(n_kv_pad, 0);
            const inp_pos = self.constI32(k_cache[0], 0.0);
            for (model.layers.items, 0..) |*layer, il| {
                cur = self.layerForward(cur, layer, inp_pos, kq_mask, &k_cache[il], &v_cache[il], n_kv_pad, 0, @intCast(il));
            }
            // position 0's output is unused, it only seeded the cache
        }

        {
            var cur = self.projectIn(code0_embd);
            const kq_mask = self.causalMaskRow(n_kv_pad, 1);
            const inp_pos = self.constI32(k_cache[0], 1.0);
            for (model.layers.items, 0..) |*layer, il| {
                cur = self.layerForward(cur, layer, inp_pos, kq_mask, &k_cache[il], &v_cache[il], n_kv_pad, 1, @intCast(il));
            }

            cur = c.ggml_rms_norm(ctx0, cur, hparams.eps);
            cur = c.ggml_mul(ctx0, cur, model.gen_code_norm_w);

            const head_w = model.gen_code_head_w;
            const head_g = c.ggml_view_2d(ctx0, head_w, ne(head_w, 0), ne(head_w, 1), nb(head_w, 1), 0); // lm_head[0]
            const logits = c.ggml_mul_mat(ctx0, head_g, cur);

            const sampled = self.doSampling(logits, inp_rand);
            out_code_cache.* = self.cacheSet(out_code_cache.*, 1, sampled);
        }
    }

    /// Ports `step` (qwen3tts-gen.cpp:236 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code_gen`: one decode step of the code
    /// predictor, `step_idx` in [1, n_acoustic - 1].
    fn step(self: *const CodeGen, k_cache: []T, v_cache: []T, out_code_cache: T, inp_rand: T, step_idx: c_int) T {
        const ctx0 = self.g.ctx0;
        const model = self.g.model;
        const hparams = self.g.hparams;
        const n_acoustic = ne(model.gen_code_head_w, 2);
        impl.ggmlAssert(step_idx >= 1 and step_idx < n_acoustic, "qwen3tts-gen.cpp", 243, "step_idx >= 1 && step_idx < n_acoustic");
        impl.ggmlAssert(k_cache.len == model.layers.items.len, "qwen3tts-gen.cpp", 244, "k_cache.size() == model.layers.size()");
        impl.ggmlAssert(v_cache.len == model.layers.items.len, "qwen3tts-gen.cpp", 245, "v_cache.size() == model.layers.size()");

        const n_kv_pad = ne(k_cache[0], 1);
        const pos = step_idx + 1; // new cache row and RoPE position

        // embed the previous code via this step's codebook table (rows are already scalars)
        const code_in = c.ggml_view_1d(ctx0, out_code_cache, 1, @as(usize, @intCast(step_idx)) * nb(out_code_cache, 1));

        const embd_w = model.gen_code_embd_w; // [n_embd_talker, vocab, n_acoustic]
        const embd_g = c.ggml_view_2d(ctx0, embd_w, ne(embd_w, 0), ne(embd_w, 1), nb(embd_w, 1), @as(usize, @intCast(step_idx - 1)) * nb(embd_w, 2));
        var cur = c.ggml_get_rows(ctx0, embd_g, code_in);
        cur = c.ggml_reshape_1d(ctx0, cur, ne(cur, 0));
        self.g.cb(cur, "step_embd_in", step_idx);

        cur = self.projectIn(cur);
        self.g.cb(cur, "step_proj_in", step_idx);

        const kq_mask = self.causalMaskRow(n_kv_pad, pos);
        const inp_pos = self.constI32(k_cache[0], @floatFromInt(pos));

        for (model.layers.items, 0..) |*layer, il| {
            cur = self.layerForward(cur, layer, inp_pos, kq_mask, &k_cache[il], &v_cache[il], n_kv_pad, pos, @intCast(il));
            self.g.cb(cur, "step_layer_out", @intCast(il));
        }

        // final norm, this step's lm_head, sample, write the result
        cur = c.ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = c.ggml_mul(ctx0, cur, model.gen_code_norm_w);

        const head_w = model.gen_code_head_w; // [n_embd_pred, vocab, n_acoustic]
        const head_g = c.ggml_view_2d(ctx0, head_w, ne(head_w, 0), ne(head_w, 1), nb(head_w, 1), @as(usize, @intCast(step_idx)) * nb(head_w, 2));
        const logits = c.ggml_mul_mat(ctx0, head_g, cur);
        self.g.cb(logits, "step_logits", step_idx);

        const sampled = self.doSampling(logits, inp_rand);
        self.g.cb(sampled, "step_sampled", step_idx);

        return self.cacheSet(out_code_cache, pos, sampled);
    }
};

/// One `state_out` entry: a slot name and the tensor holding its new state.
const StateOut = struct { name: [:0]const u8, t: T };

/// `DAC_DILATIONS` (qwen3tts-gen.cpp:581 @5266f24da).
const dac_dilations = [3]c_int{ 1, 3, 9 };

/// Ports `code2wav` (mtmd/models/models.h:306 @5266f24da), nested in
/// `clip_graph_qwen3tts_gen`: RVQ
/// codes to raw PCM (quantizer, pre_conv, pre_transformer, upsample, DAC).
const Code2Wav = struct {
    g: graph.Graph,
    /// for the names built per call; freed with the builder
    arena: std.mem.Allocator,
    state_in: std.StringHashMapUnmanaged(T) = .empty,
    state_out: std.ArrayList(StateOut) = .empty,

    fn pushState(self: *Code2Wav, name: [:0]const u8, t: T) void {
        self.state_out.append(self.arena, .{ .name = name, .t = t }) catch impl.abort("qwen3tts-gen.cpp", 0, "out of memory");
    }

    /// Ports `causal_conv1d` (qwen3tts-gen.cpp:289 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code2wav`: stride 1, prepending the
    /// persisted left context instead of zero padding.
    fn causalConv1d(self: *Code2Wav, x: T, w: T, b: T, dilation: c_int, state_name: [:0]const u8) T {
        const ctx0 = self.g.ctx0;
        const K: c_int = @intCast(ne(w, 0));
        const pad = (K - 1) * dilation;

        var x_full = x;
        if (pad > 0) {
            const left = mapAt(&self.state_in, state_name); // [pad, IC]
            x_full = c.ggml_concat(ctx0, left, x, 0);
        }
        var y = c.ggml_conv_1d(ctx0, w, x_full, 1, 0, dilation); // [T, OC, 1]
        y = c.ggml_reshape_2d(ctx0, y, ne(y, 0), ne(y, 1));
        if (b != null) y = c.ggml_add(ctx0, y, c.ggml_reshape_2d(ctx0, b, 1, ne(b, 0)));
        if (pad > 0) {
            const new_left = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, x_full, pad, ne(x_full, 1), nb(x_full, 1), @as(usize, @intCast(ne(x_full, 0) - pad)) * nb(x_full, 0)));
            self.pushState(state_name, new_left);
        }
        return y;
    }

    /// Ports `causal_conv1d_dw` (qwen3tts-gen.cpp:313 @5266f24da), a member
    /// of `clip_graph_qwen3tts_gen::code2wav`.
    fn causalConv1dDw(self: *Code2Wav, x: T, w: T, b: T, state_name: [:0]const u8) T {
        const ctx0 = self.g.ctx0;
        const K: c_int = @intCast(ne(w, 0));
        const pad = K - 1;

        var x_full = x;
        if (pad > 0) {
            const left = mapAt(&self.state_in, state_name); // [pad, C]
            x_full = c.ggml_concat(ctx0, left, x, 0);
        }
        var y = c.ggml_conv_1d_dw(ctx0, w, x_full, 1, 0, 1); // [T, C, 1]
        y = c.ggml_reshape_2d(ctx0, y, ne(y, 0), ne(y, 1));
        if (b != null) y = c.ggml_add(ctx0, y, c.ggml_reshape_2d(ctx0, b, 1, ne(b, 0)));
        if (pad > 0) {
            const new_left = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, x_full, pad, ne(x_full, 1), nb(x_full, 1), @as(usize, @intCast(ne(x_full, 0) - pad)) * nb(x_full, 0)));
            self.pushState(state_name, new_left);
        }
        return y;
    }

    /// Ports `causal_conv_transpose1d` (qwen3tts-gen.cpp:337 @5266f24da), a
    /// member of `clip_graph_qwen3tts_gen::code2wav`: the (kernel - stride)
    /// overlap tail is kept as state for the next call.
    fn causalConvTranspose1d(self: *Code2Wav, x: T, w: T, b: T, stride: c_int, state_name: [:0]const u8) T {
        const ctx0 = self.g.ctx0;
        const K: c_int = @intCast(ne(w, 0));
        const OC: c_int = @intCast(ne(w, 1));
        const trim = K - stride;
        const emit_len = ne(x, 0) * stride;

        // transposed conv as GEMM + col2im scatter-add, y: [emit_len + trim, OC]
        var w2 = c.ggml_reshape_2d(ctx0, w, @as(i64, K) * OC, ne(w, 2));
        w2 = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, w2));
        const xt = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x));
        const col = c.ggml_mul_mat(ctx0, w2, xt);
        const y = c.ggml_col2im_1d(ctx0, col, stride, OC, 0);

        var out = y;
        if (trim > 0) {
            const tail = mapAt(&self.state_in, state_name); // [trim, OC]
            const head = c.ggml_add(ctx0, c.ggml_view_2d(ctx0, y, trim, ne(y, 1), nb(y, 1), 0), tail);
            if (emit_len > trim) {
                const middle = c.ggml_view_2d(ctx0, y, emit_len - trim, ne(y, 1), nb(y, 1), @as(usize, @intCast(trim)) * nb(y, 0));
                out = c.ggml_concat(ctx0, head, middle, 0);
            } else {
                out = head;
            }
            const new_tail = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, y, trim, ne(y, 1), nb(y, 1), @as(usize, @intCast(emit_len)) * nb(y, 0)));
            self.pushState(state_name, new_tail);
        }
        if (b != null) out = c.ggml_add(ctx0, out, c.ggml_reshape_2d(ctx0, b, 1, ne(b, 0)));
        return out;
    }

    /// Ports `snake` (qwen3tts-gen.cpp:371 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code2wav`: SnakeBeta, `x + sin(alpha*x)^2 *
    /// inv_beta`.
    fn snake(self: *const Code2Wav, x: T, alpha: T, beta: T) T {
        const ctx0 = self.g.ctx0;
        const a = c.ggml_reshape_2d(ctx0, alpha, 1, ne(alpha, 0));
        const b = c.ggml_reshape_2d(ctx0, beta, 1, ne(beta, 0));

        // expand reshapes first so mul/sin/sqr/mul/add lands as consecutive nodes, letting backends fuse them
        c.ggml_build_forward_expand(self.g.gf, a);
        c.ggml_build_forward_expand(self.g.gf, b);

        var s = c.ggml_sin(ctx0, c.ggml_mul(ctx0, x, a));
        s = c.ggml_sqr(ctx0, s);
        s = c.ggml_mul(ctx0, s, b);
        return c.ggml_add(ctx0, x, s);
    }

    /// Ports `quant_decode` (qwen3tts-gen.cpp:387 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code2wav`: T frames of 16 codes to
    /// `[512, T]`.
    fn quantDecode(self: *const Code2Wav, inp_codes: T) T {
        const ctx0 = self.g.ctx0;
        const c2w = &self.g.model.c2w;
        const Tn = ne(inp_codes, 0);

        const sem = c.ggml_get_rows(ctx0, c2w.quant_first_cb_w, groupIds(ctx0, inp_codes, Tn, 0)); // [256, T]
        const sem_out = c.ggml_mul_mat(ctx0, c2w.quant_first_out_w, sem); // [512, T]

        var acc: T = null;
        const n_acoustic = ne(c2w.quant_rest_cb_w, 2);
        var g: i64 = 1;
        while (g <= n_acoustic) : (g += 1) {
            const cbw = c2w.quant_rest_cb_w;
            const cb_g = c.ggml_view_2d(ctx0, cbw, ne(cbw, 0), ne(cbw, 1), nb(cbw, 1), @as(usize, @intCast(g - 1)) * nb(cbw, 2));
            const embd = c.ggml_get_rows(ctx0, cb_g, groupIds(ctx0, inp_codes, Tn, g)); // [256, T]
            acc = if (acc != null) c.ggml_add(ctx0, acc, embd) else embd;
        }
        const ac_out = c.ggml_mul_mat(ctx0, c2w.quant_rest_out_w, acc); // [512, T]

        const hidden = c.ggml_add(ctx0, sem_out, ac_out);
        self.g.cb(hidden, "wav_quant_hidden", -1);
        return hidden;
    }

    /// The `group_ids` lambda of `quant_decode`: codebook `g`'s ids over all
    /// T frames.
    fn groupIds(ctx0: ?*c.ggml_context, inp_codes: T, Tn: i64, g: i64) T {
        return c.ggml_view_1d(ctx0, inp_codes, Tn, @as(usize, @intCast(g)) * nb(inp_codes, 1));
    }

    /// Ports `tfm_layer_forward` (qwen3tts-gen.cpp:417 @5266f24da), a member
    /// of `clip_graph_qwen3tts_gen::code2wav`: one pre_transformer layer over
    /// a batch of N = sliding_window new frames, attending over the (W-1)
    /// frames kept from the last batch.
    fn tfmLayerForward(self: *Code2Wav, cur_in: T, layer: *const model_mod.Layer, il: c_int) T {
        const ctx0 = self.g.ctx0;
        const hparams = self.g.hparams;
        const n_head = hparams.wav_tfm_n_head;
        const n_head_kv = hparams.wav_tfm_n_head_kv;
        const d_head = @divTrunc(ne(layer.q_w, 1), n_head);
        const kq_scale = 1.0 / sqrtf(@floatFromInt(d_head));
        const W: i64 = hparams.wav_tfm_swa; // == N, frames per batch
        const N = ne(cur_in, 1);
        const prefix = W - 1;
        const total_kv = prefix + N;

        const residual = cur_in;
        var h = c.ggml_rms_norm(ctx0, cur_in, hparams.wav_tfm_eps);
        h = c.ggml_mul(ctx0, h, layer.ln_1_w);

        var q = c.ggml_mul_mat(ctx0, layer.q_w, h); // [n_head*d_head, N]
        var k = c.ggml_mul_mat(ctx0, layer.k_w, h); // [n_head_kv*d_head, N]
        const v = c.ggml_mul_mat(ctx0, layer.v_w, h); // [n_head_kv*d_head, N]

        q = c.ggml_reshape_3d(ctx0, q, d_head, n_head, N);
        k = c.ggml_reshape_3d(ctx0, k, d_head, n_head_kv, N);

        // real, ever-increasing positions: base (persisted) .. base+N-1
        const base = c.ggml_reshape_1d(ctx0, mapAt(&self.state_in, "tfm_pos"), 1);
        const offset = c.ggml_arange(ctx0, 0.0, @floatFromInt(N), 1.0);
        const pos = c.ggml_cast(ctx0, c.ggml_add(ctx0, offset, base), c.GGML_TYPE_I32);

        q = c.ggml_rope_ext(ctx0, q, pos, null, @intCast(d_head), c.GGML_ROPE_TYPE_NEOX, 0, hparams.wav_tfm_rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        k = c.ggml_rope_ext(ctx0, k, pos, null, @intCast(d_head), c.GGML_ROPE_TYPE_NEOX, 0, hparams.wav_tfm_rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

        // the position counter is the same for all layers, push it once from layer 0
        if (il == 0) self.pushState("tfm_pos", c.ggml_scale_bias(ctx0, mapAt(&self.state_in, "tfm_pos"), 1.0, @floatFromInt(N)));

        const k_new = c.ggml_reshape_2d(ctx0, k, d_head * n_head_kv, N);
        const v_new = c.ggml_reshape_2d(ctx0, v, d_head * n_head_kv, N);

        const k_name = fmtName(self.arena, "tfm_k_%d", .{il});
        const v_name = fmtName(self.arena, "tfm_v_%d", .{il});
        const old_k = mapAt(&self.state_in, k_name); // [d_head*n_head_kv, W-1]
        const old_v = mapAt(&self.state_in, v_name);

        const k_full = c.ggml_concat(ctx0, old_k, k_new, 1); // [.., prefix+N]
        const v_full = c.ggml_concat(ctx0, old_v, v_new, 1);

        // next batch's prefix: the last (W-1) frames of this batch
        self.pushState(k_name, c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, k_full, ne(k_full, 0), prefix, nb(k_full, 1), @as(usize, @intCast(N)) * nb(k_full, 1))));
        self.pushState(v_name, c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, v_full, ne(v_full, 0), prefix, nb(v_full, 1), @as(usize, @intCast(N)) * nb(v_full, 1))));

        // banded causal mask: key j is visible to query i iff 0 <= (prefix+i) - j < W
        const pos_k = c.ggml_reshape_2d(ctx0, c.ggml_arange(ctx0, 0.0, @floatFromInt(total_kv), 1.0), total_kv, 1);
        const pos_q = c.ggml_reshape_2d(ctx0, c.ggml_arange(ctx0, @floatFromInt(prefix), @floatFromInt(prefix + N), 1.0), 1, N);
        const pos_q_grid = c.ggml_repeat_4d(ctx0, pos_q, total_kv, N, 1, 1);
        const diff = c.ggml_sub(ctx0, pos_q_grid, pos_k); // [total_kv, N]

        const causal_keep = c.ggml_step(ctx0, c.ggml_scale_bias(ctx0, diff, 1.0, 0.5)); // diff >= 0
        const in_window = c.ggml_step(ctx0, c.ggml_scale_bias(ctx0, diff, -1.0, @as(f32, @floatFromInt(W)) - 0.5)); // diff < W
        var keep = c.ggml_mul(ctx0, causal_keep, in_window);

        // on a cold start, key j is real state only when j >= prefix - tfm_pos, mask out the rest
        const warm = c.ggml_step(ctx0, c.ggml_scale_bias(ctx0, c.ggml_add(ctx0, pos_k, base), 1.0, 0.5 - @as(f32, @floatFromInt(prefix)))); // j + pos > prefix - 0.5
        keep = c.ggml_mul(ctx0, keep, warm);

        const mask = c.ggml_reshape_4d(ctx0, c.ggml_log(ctx0, keep), total_kv, N, 1, 1); // 0 = keep, -inf = masked

        const q_cur = c.ggml_reshape_4d(ctx0, q, d_head, n_head, N, 1);
        const k_cur = c.ggml_reshape_4d(ctx0, k_full, d_head, n_head_kv, total_kv, 1);
        const v_cur = c.ggml_reshape_4d(ctx0, v_full, d_head, n_head_kv, total_kv, 1);

        var attn_out = self.g.buildAttn(layer.o_w, layer.o_b, q_cur, k_cur, v_cur, mask, kq_scale, il, null);
        if (layer.ls_1_w != null) attn_out = c.ggml_mul(ctx0, attn_out, layer.ls_1_w);
        var cur = c.ggml_add(ctx0, residual, attn_out);

        const residual2 = cur;
        var h2 = c.ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
        h2 = c.ggml_mul(ctx0, h2, layer.ln_2_w);

        const gate = c.ggml_mul_mat(ctx0, layer.ff_gate_w, h2);
        const up = c.ggml_mul_mat(ctx0, layer.ff_up_w, h2);
        const gu = c.ggml_swiglu_split(ctx0, gate, up);
        var down = c.ggml_mul_mat(ctx0, layer.ff_down_w, gu);
        if (layer.ls_2_w != null) down = c.ggml_mul(ctx0, down, layer.ls_2_w);
        cur = c.ggml_add(ctx0, residual2, down);
        return cur;
    }

    /// Ports `convnext_block` (qwen3tts-gen.cpp:511 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code2wav`: dwconv, LayerNorm, pwconv1, GELU,
    /// pwconv2, layer scale, residual.
    fn convnextBlock(self: *Code2Wav, x: T, blk: *const model_mod.Code2Wav.UpsampleBlock, state_prefix: [:0]const u8) T {
        const ctx0 = self.g.ctx0;
        const residual = x;

        const h = self.causalConv1dDw(x, blk.dwconv_w, blk.dwconv_b, fmtName(self.arena, "%s_dwconv", .{state_prefix.ptr})); // [T, C]
        var hc = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, h)); // [C, T]

        hc = c.ggml_norm(ctx0, hc, 1e-6);
        hc = c.ggml_mul(ctx0, hc, blk.norm_w);
        hc = c.ggml_add(ctx0, hc, blk.norm_b);

        var g = c.ggml_mul_mat(ctx0, blk.pw1_w, hc);
        g = c.ggml_add(ctx0, g, blk.pw1_b);
        g = c.ggml_gelu(ctx0, g);
        g = c.ggml_mul_mat(ctx0, blk.pw2_w, g);
        g = c.ggml_add(ctx0, g, blk.pw2_b);
        g = c.ggml_mul(ctx0, g, blk.gamma);

        const g_t = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, g)); // back to [T, C]
        return c.ggml_add(ctx0, residual, g_t);
    }

    /// Ports `dac_res_unit` (qwen3tts-gen.cpp:534 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code2wav`.
    fn dacResUnit(self: *Code2Wav, x: T, res: *const model_mod.Code2Wav.DacRes, dilation: c_int, state_name: [:0]const u8) T {
        const residual = x;
        var h = self.snake(x, res.act1_alpha, res.act1_beta);
        h = self.causalConv1d(h, res.conv1_w, res.conv1_b, dilation, state_name);
        h = self.snake(h, res.act2_alpha, res.act2_beta);
        h = self.causalConv1d(h, res.conv2_w, res.conv2_b, 1, ""); // k=1, no left-context needed
        return c.ggml_add(self.g.ctx0, residual, h);
    }

    /// Ports `decode` (qwen3tts-gen.cpp:544 @5266f24da), a member of
    /// `clip_graph_qwen3tts_gen::code2wav`: a batch of N = sliding_window
    /// frames of codes to PCM.
    fn decode(self: *Code2Wav, inp_codes: T) T {
        const ctx0 = self.g.ctx0;
        const hparams = self.g.hparams;
        const c2w = &self.g.model.c2w;

        // 1. quantizer decode: N frames of 16 codes -> [512, N] (C-first)
        const hidden = self.quantDecode(inp_codes);

        // 2. pre_conv: [512, N] -> T-first [N, 512] -> causal conv k=3 -> [N, 1024]
        var x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, hidden)); // [N, 512]
        x = self.causalConv1d(x, c2w.pre_conv_w, c2w.pre_conv_b, 1, "pre_conv"); // [N, 1024]
        self.g.cb(x, "wav_pre_conv_out", -1);

        // 3. pre_transformer: back to C-first [1024, N], project down, run the layers, project back up
        var cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x)); // [1024, N]
        cur = c.ggml_mul_mat(ctx0, c2w.tfm_in_proj_w, cur);
        cur = c.ggml_add(ctx0, cur, c2w.tfm_in_proj_b); // [512 (tfm hidden), N]

        var il: c_int = 0;
        while (il < hparams.wav_tfm_n_layer) : (il += 1) {
            cur = self.tfmLayerForward(cur, &c2w.tfm_layers.items[@intCast(il)], il);
        }

        cur = c.ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
        cur = c.ggml_mul(ctx0, cur, c2w.tfm_output_norm_w);
        cur = c.ggml_mul_mat(ctx0, c2w.tfm_out_proj_w, cur);
        cur = c.ggml_add(ctx0, cur, c2w.tfm_out_proj_b); // [1024, N]
        self.g.cb(cur, "wav_tfm_out", -1);

        // 4. upsample: 2x (causal ConvTranspose1d, stride 2 + ConvNeXt block), back to T-first
        // kernel == stride here, so there is no overlap tail to persist
        x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur)); // [N, 1024]
        for (c2w.upsample.items, 0..) |*up, iu| {
            x = self.causalConvTranspose1d(x, up.conv_w, up.conv_b, 2, "");
            x = self.convnextBlock(x, up, fmtName(self.arena, "up%zu", .{iu}));
            self.g.cb(x, "wav_upsample_out", @intCast(iu));
        }

        // 5. DAC decoder: conv_pre -> n blocks (SnakeBeta -> ConvTranspose1d -> 3 res units) -> conv_post
        x = self.causalConv1d(x, c2w.dac_entry_w, c2w.dac_entry_b, 1, "dac_entry");
        self.g.cb(x, "wav_dac_entry_out", -1);

        for (c2w.dac.items, 0..) |*blk, ib| {
            const stride: c_int = @intCast(@divTrunc(ne(blk.conv_w, 0), 2)); // kernel == 2*stride for all 4 blocks
            x = self.snake(x, blk.snake_alpha, blk.snake_beta);
            x = self.causalConvTranspose1d(x, blk.conv_w, blk.conv_b, stride, fmtName(self.arena, "dac%zu_tail", .{ib}));
            for (blk.res.items, 0..) |*res, ir| {
                x = self.dacResUnit(x, res, dac_dilations[ir], fmtName(self.arena, "dac%zu_res%zu", .{ ib, ir }));
            }
            self.g.cb(x, "wav_dac_block_out", @intCast(ib));
        }

        x = self.snake(x, c2w.dac_post_snake_alpha, c2w.dac_post_snake_beta);
        x = self.causalConv1d(x, c2w.dac_post_conv_w, c2w.dac_post_conv_b, 1, "dac_post_conv"); // [n_samples, 1]

        x = c.ggml_clamp(ctx0, x, -1.0, 1.0);
        x = c.ggml_reshape_1d(ctx0, x, ne(x, 0));
        self.g.cb(x, "wav_audio_out", -1);
        return x;
    }
};

/// Ports `list_c2w_state_slots` (qwen3tts-gen.cpp:609 @5266f24da): the
/// position counter, K/V per pre_transformer layer, and the left context or
/// tail of each stateful conv. The caller frees the slice with
/// `module.freeSlots`.
pub fn listStateSlots(hparams: *const model_mod.Hparams, model: *const model_mod.Model) clip.Error![]module.StateSlot {
    const c2w = &model.c2w;
    var slots: std.ArrayList(module.StateSlot) = .empty;
    errdefer {
        for (slots.items) |s| gpa.free(s.name);
        slots.deinit(gpa);
    }

    if (c2w.pre_conv_w == null) return slots.toOwnedSlice(gpa); // not a code2wav model, it keeps no state between calls

    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "tfm_pos", 0), .ne0 = 1, .ne1 = 1 });

    // prefix is (W-1) frames, the batch itself gives the other N=W frames (see tfm_layer_forward)
    const d_head = @divTrunc(ne(c2w.tfm_layers.items[0].q_w, 1), hparams.wav_tfm_n_head);
    const kv_ch = d_head * hparams.wav_tfm_n_head_kv;
    const prefix: i64 = hparams.wav_tfm_swa - 1;
    var il: c_int = 0;
    while (il < hparams.wav_tfm_n_layer) : (il += 1) {
        try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "tfm_k_%d", .{il}), .ne0 = kv_ch, .ne1 = prefix });
        try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "tfm_v_%d", .{il}), .ne0 = kv_ch, .ne1 = prefix });
    }

    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "pre_conv", 0), .ne0 = ne(c2w.pre_conv_w, 0) - 1, .ne1 = ne(c2w.pre_conv_w, 1) });

    for (c2w.upsample.items, 0..) |*up, iu| {
        try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "up%zu_dwconv", .{iu}), .ne0 = ne(up.dwconv_w, 0) - 1, .ne1 = ne(up.dwconv_w, 2) });
    }

    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "dac_entry", 0), .ne0 = ne(c2w.dac_entry_w, 0) - 1, .ne1 = ne(c2w.dac_entry_w, 1) });

    for (c2w.dac.items, 0..) |*blk, ib| {
        const stride = @divTrunc(ne(blk.conv_w, 0), 2); // kernel == 2*stride for all 4 blocks
        try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "dac%zu_tail", .{ib}), .ne0 = stride, .ne1 = ne(blk.conv_w, 1) });
        for (blk.res.items, 0..) |*res, ir| {
            try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "dac%zu_res%zu", .{ ib, ir }), .ne0 = (ne(res.conv1_w, 0) - 1) * dac_dilations[ir], .ne1 = ne(res.conv1_w, 1) });
        }
    }

    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "dac_post_conv", 0), .ne0 = ne(c2w.dac_post_conv_w, 0) - 1, .ne1 = ne(c2w.dac_post_conv_w, 1) });

    return slots.toOwnedSlice(gpa);
}

/// Ports `build` (qwen3tts-gen.cpp:657 @5266f24da), a member of
/// `clip_graph_qwen3tts_gen`, with the builder's parameters derived as
/// clip_get_graph_builder's `PROJECTOR_TYPE_QWEN3TTS_GEN` arm does
/// (mtmd/clip.cpp:1119 @5266f24da). Both sub-graphs are always built, so the
/// topology stays constant; `ggml_build_forward_select` picks the one that
/// runs.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    const gen_process: impl.GenProcessType = if (params) |p| p.gen_process else .gen_code;
    const top_k: c_int = if (params) |p| p.top_k else 50;
    const top_p: f32 = if (params) |p| p.top_p else 1.0;

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    impl.ggmlAssert(g.n_batch == 1, "qwen3tts-gen.cpp", 658, "n_batch == 1"); // this module only ever processes one frame at a time

    const idx: c_int = switch (gen_process) {
        .gen_code => 0,
        .gen_wav => 1,
        else => impl.abort("qwen3tts-gen.cpp", 664, "unknown gen_process"),
    };

    // ---- CLIP_GEN_PROCESS_GEN_CODE: backbone hidden state -> 16 RVQ codes + next-step embd ----
    // not build_inp_raw(), a GEN_WAV call's `img` has no hidden-state data
    const h_state = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, g.n_mmproj_embd);
    _ = c.ggml_set_name(h_state, "inp_raw"); // must keep this exact name, clip_encode() sets it by name
    c.ggml_set_input(h_state);

    const code0 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, 1);
    _ = c.ggml_set_name(code0, "inp_code0");
    c.ggml_set_input(code0);

    var code0_embd = c.ggml_get_rows(ctx0, model.gen_code_out_embd_w, code0);
    code0_embd = c.ggml_reshape_1d(ctx0, code0_embd, ne(code0_embd, 0));
    g.cb(code0_embd, "code0_embd", -1);

    const n_acoustic = ne(model.gen_code_head_w, 2); // 15
    const n_codes: c_int = @intCast(n_acoustic + 1); // 16
    const n_kv_pad: i64 = n_codes;
    const n_layer = model.layers.items.len;
    const n_head = hparams.n_head;
    const n_head_kv = hparams.n_head_kv;
    const d_head = @divTrunc(ne(model.layers.items[0].q_w, 1), n_head);

    // zero-filled per layer k/v caches, so masked-out rows can't hold garbage
    const k_cache = try arena.alloc(T, n_layer);
    const v_cache = try arena.alloc(T, n_layer);
    for (0..n_layer) |il| {
        k_cache[il] = c.ggml_fill(ctx0, c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0);
        v_cache[il] = c.ggml_fill(ctx0, c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0);
    }

    const cg: CodeGen = .{ .g = graph.Graph.initChild(g), .top_k = top_k, .top_p = top_p };

    var out_code_cache = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_I32, 1, n_codes);
    out_code_cache = cg.cacheSet(out_code_cache, 0, code0);

    const inp_rand0 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
    _ = c.ggml_set_name(inp_rand0, "inp_rand_0");
    c.ggml_set_input(inp_rand0);

    cg.prefill(k_cache, v_cache, &out_code_cache, h_state, code0_embd, inp_rand0);

    var gi: c_int = 1;
    while (gi < n_acoustic) : (gi += 1) {
        const inp_rand = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
        _ = c.ggml_set_name(inp_rand, fmtName(arena, "inp_rand_%d", .{gi}).ptr);
        c.ggml_set_input(inp_rand);
        out_code_cache = cg.step(k_cache, v_cache, out_code_cache, inp_rand, gi);
    }

    // output 1: this frame's 16 sampled codes, for the caller's code2wav window
    const out_codes = c.ggml_cont(ctx0, out_code_cache);
    _ = c.ggml_set_name(out_codes, "out_codes");
    c.ggml_set_output(out_codes);

    // output 2: sum of all 16 codebook embeddings, fed back to the talker for the next frame
    var out_embd = code0_embd;
    var g2: i64 = 1;
    while (g2 <= n_acoustic) : (g2 += 1) {
        const code_g = c.ggml_view_1d(ctx0, out_code_cache, 1, @as(usize, @intCast(g2)) * nb(out_code_cache, 1));

        const ew = model.gen_code_embd_w;
        const embd_g = c.ggml_view_2d(ctx0, ew, ne(ew, 0), ne(ew, 1), nb(ew, 1), @as(usize, @intCast(g2 - 1)) * nb(ew, 2));
        var e = c.ggml_get_rows(ctx0, embd_g, code_g);
        e = c.ggml_reshape_1d(ctx0, e, ne(e, 0));

        out_embd = c.ggml_add(ctx0, out_embd, e);
    }
    out_embd = c.ggml_reshape_2d(ctx0, out_embd, ne(out_embd, 0), 1);
    g.cb(out_embd, "gen_audio_out", -1);

    // ---- CLIP_GEN_PROCESS_GEN_WAV: 16 RVQ codes -> raw PCM ----
    const n_frames: i64 = hparams.wav_tfm_swa; // frames per batch, == the attention window

    const inp_codes = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_I32, n_frames, n_codes);
    _ = c.ggml_set_name(inp_codes, "inp_codes");
    c.ggml_set_input(inp_codes);

    var c2w: Code2Wav = .{ .g = graph.Graph.initChild(g), .arena = arena };
    const slots = try listStateSlots(hparams, model);
    defer module.freeSlots(slots);
    for (slots) |slot| {
        const t = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, slot.ne0, slot.ne1);
        _ = c.ggml_set_name(t, fmtName(arena, "state_in_%s", .{slot.name.ptr}).ptr);
        c.ggml_set_input(t);
        try c2w.state_in.put(arena, try arena.dupe(u8, slot.name), t);
    }

    const out_audio = c2w.decode(inp_codes);
    _ = c.ggml_set_name(out_audio, "out_audio");
    c.ggml_set_output(out_audio);

    for (c2w.state_out.items) |slot| {
        _ = c.ggml_set_name(slot.t, fmtName(arena, "state_out_%s", .{slot.name.ptr}).ptr);
        c.ggml_set_output(slot.t);
    }

    // out_embd goes last, clip_encode() reads it back via ggml_graph_node(gf, -1)
    var outs: [2]?*c.ggml_tensor = .{ out_codes, out_audio };
    _ = c.ggml_build_forward_select(g.gf, &outs, 2, idx);
    for (c2w.state_out.items) |slot| {
        outs = .{ out_codes, slot.t };
        _ = c.ggml_build_forward_select(g.gf, &outs, 2, idx);
    }
    outs = .{ out_embd, out_audio };
    _ = c.ggml_build_forward_select(g.gf, &outs, 2, idx);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
