//! Ripple (Ricursive AI): a dense decoder whose sliding-window layers alone
//! are rotated, the global ones carrying no positional encoding, with QK
//! norms, post-attention and post-FFN norms, an optional sigmoid output
//! gate, an optional value residual mixing the token embeddings into each
//! layer's values, and a per-layer SwiGLU clamp.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/ricursive.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_ricursive`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`.

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

/// A `runtime_error` whose `what()` is upstream's `format(...)`, which is
/// `snprintf`: `%g` has no `std.fmt` equivalent.
fn throwC(exc: *Exception, comptime fmt: [:0]const u8, args: anytype) Thrown {
    var buf: [512]u8 = undefined;
    const n = @call(.auto, c.snprintf, .{ &buf, buf.len, fmt.ptr } ++ args);
    const len: usize = @min(@as(usize, @intCast(n)), buf.len - 1);
    return exc.throw(.runtime_error, "{s}", .{buf[0..len]});
}

/// Ports `llama_model_ricursive`'s `load_arch_hparams` (models/ricursive.cpp:3 @b1b770d90).
///
/// Parameters:
/// - `self`: the model being loaded.
/// - `ml`: the loader, for the metadata.
/// - `exc`: where a throw is recorded.
///
/// Return: nothing; `error.Thrown` for a malformed file.
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(f32, .embedding_scale, false, exc)) |v| hp.f_embedding_scale = v;
    // Optional: absent or 0 means no clamp; a scalar applies to every layer.
    _ = try meta.getKeyOrArr(f32, .swiglu_clamp, &hp.swiglu_clamp, hp.n_layer_all, false, exc);
    var il: u32 = 0;
    while (il < hp.n_layer_all) : (il += 1) {
        if (!std.math.isFinite(hp.swiglu_clamp[il]) or hp.swiglu_clamp[il] < 0.0) {
            return throwC(exc, "ricursive: swiglu_clamp for layer %u must be finite and non-negative, got %g", .{ @as(c_uint, il), @as(f64, hp.swiglu_clamp[il]) });
        }
    }

    if (hp.nEmbdHeadK(0) != hp.nEmbdHeadV(0)) {
        return exc.throw(.runtime_error, "ricursive: key_length and value_length must be equal", .{});
    }

    // RoPE is only applied on the SWA layers, so a model without SWA would
    // have no positional encoding.
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    if (hp.n_swa == 0) {
        return exc.throw(.runtime_error, "ricursive: sliding_window must be > 0", .{});
    }
    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    if (try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, false, exc)) {
        // load_swa_pattern bounds the array only by LLAMA_MAX_LAYERS: a short
        // one would leave the last layers dense and without RoPE.
        const n_entries: u32 = (try meta.getArrN(.attention_sliding_window_pattern, false, exc)) orelse 0;
        if (n_entries != hp.nLayer()) {
            return throwC(exc, "ricursive: sliding_window_pattern has %u entries, expected %u", .{ @as(c_uint, n_entries), @as(c_uint, hp.nLayer()) });
        }
    } else {
        // Default: every 5th layer is global (full attention iff i % 5 == 4).
        try self.loadSwaPattern(ml, 5, false, exc);
    }
    if (!hp.isSwaAny()) {
        return exc.throw(.runtime_error, "ricursive: no sliding window layer, the model would have no positional encoding", .{});
    }

    // RoPE exactly on the SWA layers; rope_pattern is optional because the
    // first published GGUFs do not carry it.
    hp.rope_pattern = hp.is_swa_impl;
    _ = try meta.getKeyOrArr(u32, .attention_rope_pattern, &hp.rope_pattern, hp.nLayer(), false, exc);
    il = 0;
    while (il < hp.nLayer()) : (il += 1) {
        if (hp.hasRope(il) != hp.isSwa(il)) {
            const rope: [:0]const u8 = if (hp.hasRope(il)) "RoPE" else "no RoPE";
            const swa: [:0]const u8 = if (hp.isSwa(il)) "a sliding window" else "no sliding window";
            return throwC(exc, "ricursive: layer %u has %s but %s", .{ @as(c_uint, il), rope.ptr, swa.ptr });
        }
    }

    // The SWA layers are the only RoPE layers; they use the model's RoPE
    // parameters.
    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    self.type = switch (hp.nLayer()) {
        80 => .@"65b",
        else => .unknown,
    };
}

/// Ports `llama_model_ricursive`'s `load_arch_tensors` (models/ricursive.cpp:60 @b1b770d90).
///
/// Parameters:
/// - `self`: the model being loaded.
/// - `exc`: where a throw is recorded.
///
/// Return: nothing; `error.Thrown` when a tensor is missing or the gate or
/// value residual is present on some layers only.
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    const n_embd_q: i64 = l.n_embd_head_k * l.n_head;

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_q }, 0, exc);
        layer.wk = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        layer.wv = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_v_gqa }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_q, l.n_embd }, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_q }, nr, exc);
        layer.attn_vres_lambda = try self.createTensor(.{ .tensor = .attn_vres_lambda, .suffix = "weight", .bid = i }, &.{1}, nr, exc);

        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
    }

    // The gate and the value residual are per-model switches.
    i = 1;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        if ((layer.wqkv_gate == null) != (self.layers[0].wqkv_gate == null)) {
            return throwC(exc, "ricursive: attn_gate must be present in all layers or in none (layers 0 and %d differ)", .{@as(c_int, i)});
        }
        if ((layer.attn_vres_lambda == null) != (self.layers[0].attn_vres_lambda == null)) {
            return throwC(exc, "ricursive: attn_vres_lambda must be present in all layers or in none (layers 0 and %d differ)", .{@as(c_int, i)});
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_ricursive`'s `build_arch_graph`
/// (models/ricursive.cpp:109 @b1b770d90) and the constructor of its `graph`
/// (models/ricursive.cpp:113 @b1b770d90).
///
/// Parameters:
/// - `model`: the model.
/// - `b`: the graph builder.
///
/// Return: nothing; the graph is built into `b`.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The value residual: x0, the token embeddings scaled, the same tensor
    // for all layers.
    const inp_vres = inpL;

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKvIswa();

    const inp_out_ids = b.buildInpOutIds();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const ilu: u32 = @intCast(il);
        const layer = &model.layers[ilu];

        var inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        const attn_in = cur;

        {
            var Qcur = b.buildLoraMm(layer.wq.?, attn_in, layer.wq_s);
            var Kcur = b.buildLoraMm(layer.wk.?, attn_in, layer.wk_s);

            // v_in = lerp(x0, attn_norm(x), sigmoid(lambda)), lambda learned
            // per layer.
            var v_in = attn_in;
            if (layer.attn_vres_lambda) |lambda| {
                const mix = c.ggml_sigmoid(ctx, lambda);
                v_in = c.ggml_add(ctx, inp_vres, c.ggml_mul(ctx, c.ggml_sub(ctx, attn_in, inp_vres), mix));
                b.cb(v_in, "vres_in", il);
            }
            var Vcur = b.buildLoraMm(layer.wv.?, v_in, layer.wv_s);

            Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
            Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
            Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);
            b.cb(Kcur, "Kcur_normed", il);

            // The global layers use no RoPE (NoPE): only the sliding-window
            // layers are rotated.
            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);
            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);

            if (layer.wqkv_gate) |wg| {
                var gate = b.buildLoraMm(wg, attn_in, layer.wqkv_gate_s);
                gate = c.ggml_sigmoid(ctx, gate);
                b.cb(gate, "attn_gate", il);
                cur = c.ggml_mul(ctx, cur, gate);
                b.cb(cur, "attn_gated", il);
            }

            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
        }

        // `inp_out_ids` is never null here.
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        {
            const gate = b.buildLoraMm(layer.ffn_gate.?, cur, layer.ffn_gate_s);
            const up = b.buildLoraMm(layer.ffn_up.?, cur, layer.ffn_up_s);
            cur = if (hp.swiglu_clamp[ilu] > 0.0)
                c.ggml_swiglu_clamp(ctx, gate, up, hp.swiglu_clamp[ilu])
            else
                c.ggml_swiglu_split(ctx, gate, up);
            b.cb(cur, "ffn_swiglu", il);
            cur = b.buildLoraMm(layer.ffn_down.?, cur, layer.ffn_down_s);
            b.cb(cur, "ffn_out", il);
        }

        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, il);
        b.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);
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
