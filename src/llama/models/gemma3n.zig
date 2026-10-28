//! Gemma 3n: Gemma 3's normed attention over a stack of `n_altup` parallel
//! residual streams (AltUp), of which one is active per layer and the rest
//! are predicted and corrected around it; a low-rank residual (LAuReL) beside
//! attention; per-layer token embeddings gated into every layer; activation
//! sparsity in the first ten feed-forwards; and the last layers reading the
//! KV cache of earlier ones.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/gemma3n.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_gemma3n`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph` and the graph's helpers.

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

/// Ports `llama_model_gemma3n`'s `load_arch_hparams` (models/gemma3n.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    var swa_period: u32 = 5;
    if (try meta.getKeyOrArrScalar(.attention_sliding_window_pattern, false, exc)) |v| swa_period = v;
    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    hp.setSwaPattern(swa_period, false);

    hp.n_layer_kv_from_start = 20;
    hp.f_attention_scale = 1.0;

    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        30 => .e2b,
        35 => .e4b,
        else => .unknown,
    };
}

/// Ports `llama_model_gemma3n`'s `load_arch_tensors` (models/gemma3n.cpp:23 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const n_altup: i64 = hp.n_altup;
    const laurel_rank: i64 = hp.laurel_rank;
    const n_embd_altup: i64 = hp.n_embd_altup;

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_not_required, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.altup_proj = try self.createTensor(.{ .tensor = .altup_proj, .suffix = "weight" }, &.{ l.n_embd, l.n_embd, n_altup - 1 }, 0, exc);
    self.altup_unembd_proj = try self.createTensor(.{ .tensor = .altup_unembd_proj, .suffix = "weight" }, &.{ l.n_embd, l.n_embd, n_altup - 1 }, 0, exc);

    self.per_layer_tok_embd = try self.createTensor(.{ .tensor = .per_layer_token_embd, .suffix = "weight" }, &.{ n_embd_altup * l.n_layer, l.n_vocab }, 0, exc);
    self.per_layer_model_proj = try self.createTensor(.{ .tensor = .per_layer_model_proj, .suffix = "weight", .bid = 0 }, &.{ l.n_embd, n_embd_altup * l.n_layer }, 0, exc);
    self.per_layer_proj_norm = try self.createTensor(.{ .tensor = .per_layer_proj_norm, .suffix = "weight", .bid = 0 }, &.{n_embd_altup}, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // AltUp and LAuReL.
        layer.per_layer_inp_gate = try self.createTensor(.{ .tensor = .per_layer_inp_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_altup }, 0, exc);
        layer.per_layer_proj = try self.createTensor(.{ .tensor = .per_layer_proj, .suffix = "weight", .bid = i }, &.{ n_embd_altup, l.n_embd }, 0, exc);
        layer.per_layer_post_norm = try self.createTensor(.{ .tensor = .per_layer_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.altup_correct_coef = try self.createTensor(.{ .tensor = .altup_correct_coef, .suffix = "weight", .bid = i }, &.{ n_altup, n_altup }, 0, exc);
        layer.altup_correct_scale = try self.createTensor(.{ .tensor = .altup_correct_scale, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.altup_predict_coef = try self.createTensor(.{ .tensor = .altup_predict_coef, .suffix = "weight", .bid = i }, &.{ n_altup, n_altup * n_altup }, 0, exc);
        layer.altup_router = try self.createTensor(.{ .tensor = .altup_router, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_altup }, 0, exc);
        layer.altup_router_norm = try self.createTensor(.{ .tensor = .altup_router_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.laurel_l = try self.createTensor(.{ .tensor = .laurel_l, .suffix = "weight", .bid = i }, &.{ l.n_embd, laurel_rank }, 0, exc);
        layer.laurel_r = try self.createTensor(.{ .tensor = .laurel_r, .suffix = "weight", .bid = i }, &.{ laurel_rank, l.n_embd }, 0, exc);
        layer.laurel_post_norm = try self.createTensor(.{ .tensor = .laurel_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
    }
}

/// Ports `llama_model_gemma3n`'s `build_arch_graph` (models/gemma3n.cpp:80 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const g: Graph = .{
        .model = model,
        .b = b,
        .n_embd_head = b.hparams.nEmbdHeadK(0),
        .n_embd_altup = b.hparams.n_embd_altup,
        .n_altup = b.hparams.n_altup,
        .i_altup_act = @intCast(b.hparams.i_altup_act),
    };
    g.build();
}

/// Ports `ggml_view_2d_slice` (models/gemma3n.cpp:85 @5266f24da): the 2D
/// slice of a 3D tensor at `idx` along its third dimension.
fn view2dSlice(ctx: *c.ggml_context, x: *Tensor, idx: c_int) *Tensor {
    impl.assert(idx < @as(c_int, @intCast(x.ne[2])), "models/gemma3n.cpp", 86, "idx < (int) x->ne[2]");
    return c.ggml_view_2d(ctx, x, x.ne[0], x.ne[1], c.ggml_row_size(x.type, x.ne[0]), @intCast(idx * x.ne[0] * x.ne[1] * @as(i64, @intCast(c.ggml_element_size(x)))));
}

/// Ports `llama_model_gemma3n`'s `graph` (src/models/models.h:829 @5266f24da): the
/// builder and the constants its helpers share.
const Graph = struct {
    model: *const Model,
    b: *const graph.Builder,

    n_embd_head: i64,
    n_embd_altup: i64,
    n_altup: i64,
    i_altup_act: c_int,
    /// The number of layers using activation sparsity.
    n_layer_sparsity: c_int = 10,
    /// `std_multiplier = normal_dist.icdf(0.95)`.
    f_sparsity_std_mul: f32 = 1.6448533535003662,

    /// Ports the constructor of `llama_model_gemma3n`'s `graph`
    /// (models/gemma3n.cpp:91 @5266f24da).
    fn build(self: *const Graph) void {
        const model = self.model;
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_embd = b.n_embd;
        const n_tokens = b.n_tokens;
        const n_altup = self.n_altup;

        var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

        // Important: do not normalize weights for raw embeddings input (i.e.
        // encoded image embeddings).
        inpL = c.ggml_scale(ctx, inpL, if (b.ubatch.token != null) impl.sqrtf(@floatFromInt(n_embd)) else 1.0);
        b.cb(inpL, "inp_scaled", -1);

        // The positions.
        const inp_pos = b.buildInpPos();

        const inp_attn = b.buildAttnInpKvIswa();

        var inp_per_layer: *Tensor = self.buildInpPerLayer();
        c.ggml_build_forward_expand(b.gf, inp_per_layer);

        // inp_per_layer now has shape [n_embd_altup, n_tokens, n_layer].
        inp_per_layer = self.projectPerLayerInputs(inpL, inp_per_layer);

        // inpL has only one altup; project it to the rest, which are
        // concatenated along its last dimension.
        {
            const target_magnitude = self.calcMagnitude(inpL);
            const inp_repeated: *Tensor = c.ggml_repeat_4d(ctx, inpL, n_embd, n_tokens, n_altup - 1, 1);
            var altup_added: *Tensor = c.ggml_mul_mat(ctx, model.altup_proj, inp_repeated); // [n_embd, n_tokens, n_altup - 1]
            const new_magnitude = self.calcMagnitude(altup_added);
            altup_added = c.ggml_div(ctx, c.ggml_mul(ctx, altup_added, target_magnitude), new_magnitude);
            inpL = c.ggml_concat(ctx, inpL, altup_added, 2); // [n_embd, n_tokens, n_altup]
            b.cb(inpL, "inp_stacked", -1);
        }
        // inpL now has shape [n_embd, n_tokens, n_altup].

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];
            // This block closely resembles Gemma3p5DecoderLayer in the
            // Python code.
            const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
            const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

            var cur: *Tensor = inpL; // [n_embd, n_tokens, n_altup]
            const predictions = self.altupPredict(cur, il); // [n_embd, n_tokens, n_altup]

            // The predicted value goes through self-attention and LAuReL.
            const active_prediction = view2dSlice(ctx, predictions, self.i_altup_act); // [n_embd, n_tokens]
            cur = active_prediction;
            b.cb(cur, "active_prediction", il);

            // Norm.
            cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);

            // LAuReL.
            const laurel_out = self.laurel(cur, il); // [n_embd, n_tokens]

            // Self-attention.
            if (hp.hasKv(@intCast(il))) {
                const qkv = b.buildQkv(layer, cur, self.n_embd_head, b.n_head, b.n_head_kv, il);
                var Qcur = qkv.q;
                var Kcur = qkv.k;
                var Vcur = qkv.v;

                Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
                Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
                Vcur = c.ggml_rms_norm(ctx, Vcur, hp.f_norm_rms_eps);

                b.cb(Qcur, "Qcur_normed", il);
                b.cb(Kcur, "Kcur_normed", il);
                b.cb(Vcur, "Vcur_normed", il);

                Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

                Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

                b.cb(Qcur, "Qcur_pos", il);
                b.cb(Kcur, "Kcur_pos", il);

                cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, hp.f_attention_scale, il);
            } else {
                // Reuse the KV cache of earlier layers.
                var Qcur: *Tensor = b.buildLoraMm(layer.wq.?, cur, null);
                b.cb(Qcur, "Qcur", il);
                Qcur = c.ggml_reshape_3d(ctx, Qcur, self.n_embd_head, b.n_head, n_tokens);

                Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
                b.cb(Qcur, "Qcur_normed", il);

                Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                b.cb(Qcur, "Qcur_pos", il);

                cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, null, null, null, null, null, hp.f_attention_scale, il);
            }
            cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
            b.cb(cur, "attn_post_norm", il);

            cur = c.ggml_add(ctx, cur, active_prediction); // [n_embd, n_tokens]
            b.cb(cur, "attn_gated", il);

            const attn_laurel: *Tensor = c.ggml_scale(ctx, c.ggml_add(ctx, cur, laurel_out), 1.0 / impl.sqrtf(2.0)); // [n_embd, n_tokens]
            b.cb(attn_laurel, "attn_laurel", il);

            cur = b.buildNorm(attn_laurel, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            // The feed-forward network.
            {
                const up_proj = b.buildLoraMm(layer.ffn_up.?, cur, null);
                var gate_proj = b.buildLoraMm(layer.ffn_gate.?, cur, null);

                if (il < self.n_layer_sparsity) {
                    // Apply activation sparsity.
                    gate_proj = self.gaussianTopk(gate_proj);
                }
                gate_proj = c.ggml_gelu(ctx, gate_proj);

                cur = c.ggml_mul(ctx, up_proj, gate_proj);
                cur = b.buildLoraMm(layer.ffn_down.?, cur, null);
                b.cb(cur, "ffn_out", il);
            }
            cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, -1);
            b.cb(cur, "ffn_post_norm", il);

            const attn_ffw_laurel_gated: *Tensor = c.ggml_add(ctx, cur, attn_laurel); // [n_embd, n_tokens]
            b.cb(attn_ffw_laurel_gated, "attn_ffw_laurel_gated", il);

            var corrected = self.altupCorrect(predictions, attn_ffw_laurel_gated, il); // [n_embd, n_tokens, n_altup]

            var first_prediction: *Tensor = undefined; // [n_embd, n_tokens]
            {
                first_prediction = view2dSlice(ctx, corrected, self.i_altup_act); // [n_embd, n_tokens]
                first_prediction = c.ggml_mul(ctx, first_prediction, layer.altup_correct_scale);
                first_prediction = b.buildLoraMm(layer.per_layer_inp_gate.?, first_prediction, null);
                first_prediction = c.ggml_gelu(ctx, first_prediction); // [n_embd_altup, n_tokens]
                b.cb(first_prediction, "first_prediction_gated", il);

                const inp_this_layer = view2dSlice(ctx, inp_per_layer, il); // [n_embd_altup, n_tokens]
                first_prediction = c.ggml_mul(ctx, first_prediction, inp_this_layer); // [n_embd_altup, n_tokens]
                b.cb(first_prediction, "first_prediction_scaled", il);

                first_prediction = b.buildLoraMm(layer.per_layer_proj.?, first_prediction, null); // [n_embd, n_tokens]
                first_prediction = b.buildNorm(first_prediction, layer.per_layer_post_norm, null, .rms, il);
                b.cb(first_prediction, "first_prediction_out", il);
            }
            // Equivalent to the Python code's
            // `corrected_predictions[1:] += first_prediction`.
            {
                const slice_first = view2dSlice(ctx, corrected, 0);
                const slice_rest: *Tensor = c.ggml_view_3d(ctx, corrected, n_embd, n_tokens, n_altup - 1, c.ggml_row_size(corrected.type, n_embd), c.ggml_row_size(corrected.type, n_embd * n_tokens), @intCast(n_embd * n_tokens * @as(i64, @intCast(c.ggml_element_size(corrected)))));
                const tmp: *Tensor = c.ggml_add(ctx, slice_rest, first_prediction); // [n_embd, n_tokens, n_altup - 1]
                corrected = c.ggml_concat(ctx, slice_first, tmp, 2); // [n_embd, n_tokens, n_altup]
            }
            cur = corrected; // [n_embd, n_tokens, n_altup]
            cur = b.buildCvec(cur, il);
            b.cb(cur, "l_out", il);

            // The next layer's input.
            inpL = cur;
        }
        var cur = inpL; // [n_embd, n_tokens, n_altup]

        // cur has several altups; merge them back into one.
        {
            const target_magnitude = self.calcMagnitude(view2dSlice(ctx, cur, self.i_altup_act)); // [n_embd, n_tokens]
            // A view that skips the first slice (the active altup).
            const alt_slice: *Tensor = c.ggml_view_3d(ctx, cur, n_embd, n_tokens, n_altup - 1, c.ggml_row_size(cur.type, n_embd), c.ggml_row_size(cur.type, n_embd * n_tokens), @intCast(n_embd * n_tokens * @as(i64, @intCast(c.ggml_element_size(cur)))));
            var altup_unembd: *Tensor = c.ggml_mul_mat(ctx, model.altup_unembd_proj, alt_slice); // [n_embd, n_tokens, n_altup - 1]
            const new_magnitude = self.calcMagnitude(altup_unembd);
            altup_unembd = c.ggml_div(ctx, c.ggml_mul(ctx, altup_unembd, target_magnitude), new_magnitude);
            b.cb(altup_unembd, "altup_unembd", -1);

            // Equivalent to `torch.mean(hidden_states, dim=0)`.
            cur = view2dSlice(ctx, cur, 0); // [n_embd, n_tokens]
            var i: c_int = 0;
            while (i < n_altup - 1) : (i += 1) {
                cur = c.ggml_add(ctx, cur, view2dSlice(ctx, altup_unembd, i));
            }
            cur = c.ggml_scale(ctx, cur, 1.0 / @as(f32, @floatFromInt(n_altup))); // [n_embd, n_tokens]
            b.cb(cur, "unembd_merged", -1);
        }
        // cur now has shape [n_embd, n_tokens].

        // TODO: move this to right after the last KV layer.
        {
            // Skip computing output for unused tokens.
            const inp_out_ids = b.buildInpOutIds();
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
        }
        cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        cur = b.buildLoraMm(model.output.?, cur, model.output_s);

        {
            // Final logit soft-capping.
            cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
            cur = c.ggml_tanh(ctx, cur);
            cur = c.ggml_scale(ctx, cur, hp.f_final_logit_softcapping);
        }
        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;

        c.ggml_build_forward_expand(b.gf, cur);
    }

    /// Ports `llama_model_gemma3n`'s `graph`'s `calc_magnitude`
    /// (models/gemma3n.cpp:313 @5266f24da).
    fn calcMagnitude(self: *const Graph, x: *Tensor) *Tensor {
        const ctx = self.b.ctx0;
        return c.ggml_sqrt(ctx, c.ggml_sum_rows(ctx, c.ggml_sqr(ctx, x)));
    }

    /// Ports `llama_model_gemma3n`'s `graph`'s `build_inp_per_layer`
    /// (models/gemma3n.cpp:319 @5266f24da): equivalent to the Python code's
    /// `get_per_layer_inputs()`.
    ///
    /// Return: `[n_embd_altup, n_layer, n_tokens]`.
    fn buildInpPerLayer(self: *const Graph) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const model = self.model;
        var inp_per_layer: *Tensor = undefined;
        const tok_embd_scale = impl.sqrtf(@floatFromInt(self.n_embd_altup));
        if (b.ubatch.token != null) {
            const inp = impl.allocator.create(graph.InpEmbd) catch impl.abort("models/gemma3n.cpp", 320, "out of memory");
            inp.* = .{ .n_embd = b.n_embd };
            inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.ubatch.n_tokens);
            c.ggml_set_input(inp.tokens);
            b.res.t_inp_tokens = inp.tokens;
            inp_per_layer = c.ggml_get_rows(ctx, model.per_layer_tok_embd, inp.tokens);
            inp_per_layer = c.ggml_reshape_3d(ctx, inp_per_layer, self.n_embd_altup, b.n_layer, b.n_tokens);
            inp_per_layer = c.ggml_scale(ctx, inp_per_layer, tok_embd_scale);
            b.cb(inp_per_layer, "inp_per_layer_selected", -1);
            _ = b.res.addInput(graph.InpEmbd, inp);
        } else {
            // The multimodal embedding path uses the padding token's (ID 0)
            // embedding.
            // TODO: verify if this is the correct behavior in transformers
            // implementation.
            const embd_size = model.per_layer_tok_embd.?.ne[0]; // n_embd_altup * n_layer

            // Extract and dequantize the padding token's embedding (row 0).
            const padding: *Tensor = c.ggml_view_1d(ctx, model.per_layer_tok_embd, embd_size, 0);
            inp_per_layer = c.ggml_cast(ctx, padding, c.GGML_TYPE_F32);
            inp_per_layer = c.ggml_scale(ctx, inp_per_layer, tok_embd_scale);

            // Reshape to [n_embd_altup, n_layer, 1].
            inp_per_layer = c.ggml_reshape_3d(ctx, inp_per_layer, self.n_embd_altup, b.n_layer, 1);
            b.cb(inp_per_layer, "inp_per_layer_multimodal", -1);
        }
        return inp_per_layer;
    }

    /// Ports `llama_model_gemma3n`'s `graph`'s `project_per_layer_inputs`
    /// (models/gemma3n.cpp:352 @5266f24da): equivalent to the Python code's
    /// `project_per_layer_inputs()`, the per-layer inputs with `n_layer` as
    /// the last dimension.
    ///
    /// Return: `[n_embd_altup, n_tokens, n_layer]`.
    fn projectPerLayerInputs(self: *const Graph, inp_batch: *Tensor, inp_per_layer_: *Tensor) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const per_layer_projection_scale = 1.0 / impl.sqrtf(@floatFromInt(b.n_embd));
        const per_layer_input_scale = 1.0 / impl.sqrtf(2.0);

        var per_layer_proj: *Tensor = c.ggml_mul_mat(ctx, self.model.per_layer_model_proj, inp_batch);
        per_layer_proj = c.ggml_scale(ctx, per_layer_proj, per_layer_projection_scale);
        per_layer_proj = c.ggml_reshape_3d(ctx, per_layer_proj, self.n_embd_altup, b.n_layer, b.n_tokens);

        per_layer_proj = b.buildNorm(per_layer_proj, self.model.per_layer_proj_norm, null, .rms, -1);
        b.cb(per_layer_proj, "per_layer_proj", -1);

        var inp_per_layer: *Tensor = c.ggml_add(ctx, per_layer_proj, inp_per_layer_);
        inp_per_layer = c.ggml_scale(ctx, inp_per_layer, per_layer_input_scale);
        b.cb(inp_per_layer, "inp_per_layer", -1);

        // Permute to [n_embd_altup, n_tokens, n_layer].
        inp_per_layer = c.ggml_cont(ctx, c.ggml_permute(ctx, inp_per_layer, 0, 2, 1, 3));
        return inp_per_layer;
    }

    /// Ports `llama_model_gemma3n`'s `graph`'s `laurel` (models/gemma3n.cpp:375
    /// @5266f24da).
    fn laurel(self: *const Graph, cur: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const layer = &self.model.layers[@intCast(il)];
        var tmp = cur;
        tmp = b.buildLoraMm(layer.laurel_l.?, tmp, null);
        tmp = b.buildLoraMm(layer.laurel_r.?, tmp, null);
        tmp = b.buildNorm(tmp, layer.laurel_post_norm, null, .rms, il);
        tmp = c.ggml_add(b.ctx0, tmp, cur);
        b.cb(tmp, "laurel_out", il);
        return tmp;
    }

    /// Ports `llama_model_gemma3n`'s `graph`'s `gaussian_topk`
    /// (models/gemma3n.cpp:387 @5266f24da).
    ///
    /// Return: `[n_embd, n_tokens]`, as `x`.
    fn gaussianTopk(self: *const Graph, x: *Tensor) *Tensor {
        const ctx = self.b.ctx0;
        const mean: *Tensor = c.ggml_mean(ctx, x);
        const std_: *Tensor = c.ggml_sqrt(ctx, c.ggml_scale(ctx, c.ggml_sum_rows(ctx, c.ggml_sqr(ctx, c.ggml_sub(ctx, x, mean))), 1.0 / @as(f32, @floatFromInt(x.ne[0] - 1))));
        const cutoff_x: *Tensor = c.ggml_add(ctx, mean, c.ggml_scale(ctx, std_, self.f_sparsity_std_mul));
        return c.ggml_relu(ctx, c.ggml_sub(ctx, x, cutoff_x));
    }

    // AltUp.

    /// Ports `llama_model_gemma3n`'s `graph`'s `altup_compute_router_modalities`
    /// (models/gemma3n.cpp:402 @5266f24da): equivalent to the Python code's
    /// `compute_router_modalities()`.
    ///
    /// Return: `[n_altup, n_tokens]`, from `x` `[n_embd, n_tokens]`.
    fn altupComputeRouterModalities(self: *const Graph, x: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const layer = &self.model.layers[@intCast(il)];
        var router_inputs = b.buildNorm(x, layer.altup_router_norm, null, .rms, il);

        // router_input_scale
        router_inputs = c.ggml_scale(ctx, router_inputs, 1.0 / @as(f32, @floatFromInt(b.n_embd)));

        const output: *Tensor = c.ggml_mul_mat(ctx, layer.altup_router, router_inputs);
        return c.ggml_tanh(ctx, output); // [n_altup, n_tokens]
    }

    /// Ports `llama_model_gemma3n`'s `graph`'s `altup_predict`
    /// (models/gemma3n.cpp:414 @5266f24da).
    ///
    /// Return: `[n_embd, n_tokens, n_altup]`, as `cur`.
    fn altupPredict(self: *const Graph, cur: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const layer = &self.model.layers[@intCast(il)];
        const n_altup = self.n_altup;
        const activated = view2dSlice(ctx, cur, self.i_altup_act); // [n_embd, n_tokens]
        const modalities = self.altupComputeRouterModalities(activated, il); // [n_altup, n_tokens]
        b.cb(modalities, "modalities", il);

        var all_coefs = b.buildLoraMm(layer.altup_predict_coef.?, modalities, null);
        b.cb(all_coefs, "all_coefs", il);
        // The first dimension has n_altup^2 elements; reshape it to 2D, for
        // a 3D tensor.
        all_coefs = c.ggml_reshape_3d(ctx, all_coefs, n_altup, n_altup, b.n_tokens);

        // Permute to [n_altup, n_embd, n_tokens].
        const cur_permuted: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, cur, 1, 2, 0, 3));
        var predictions: *Tensor = c.ggml_mul_mat(ctx, cur_permuted, all_coefs); // [n_altup, n_embd, n_tokens]

        // The final shape must be cur's: [n_embd, n_tokens, n_altup].
        predictions = c.ggml_cont(ctx, c.ggml_permute(ctx, predictions, 0, 2, 1, 3));
        predictions = c.ggml_add(ctx, predictions, cur);
        b.cb(predictions, "predictions", il);

        return predictions;
    }

    /// Ports `llama_model_gemma3n`'s `graph`'s `altup_correct`
    /// (models/gemma3n.cpp:439 @5266f24da).
    ///
    /// Return: `[n_embd, n_tokens, n_altup]`, from `predictions`
    /// `[n_embd, n_tokens, n_altup]` and `activated` `[n_embd, n_tokens]`.
    fn altupCorrect(self: *const Graph, predictions: *Tensor, activated: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const layer = &self.model.layers[@intCast(il)];
        const modalities = self.altupComputeRouterModalities(activated, il); // [n_altup, n_tokens]
        b.cb(modalities, "modalities", il);

        const active_prediction = view2dSlice(ctx, predictions, self.i_altup_act);
        var innovation: *Tensor = c.ggml_sub(ctx, activated, active_prediction); // [n_embd, n_tokens]
        b.cb(innovation, "innovation", il);

        var all_coefs = b.buildLoraMm(layer.altup_correct_coef.?, modalities, null); // [n_altup, n_tokens]
        all_coefs = c.ggml_scale_bias(ctx, all_coefs, 1.0, 1.0); // + 1.0
        b.cb(all_coefs, "all_coefs", il);
        all_coefs = c.ggml_transpose(ctx, all_coefs); // [n_tokens, n_altup]
        all_coefs = c.ggml_cont_3d(ctx, all_coefs, 1, b.n_tokens, self.n_altup); // [1, n_tokens, n_altup]

        innovation = c.ggml_repeat_4d(ctx, innovation, b.n_embd, b.n_tokens, self.n_altup, 1);
        var corrected: *Tensor = c.ggml_mul(ctx, innovation, all_coefs); // [n_embd, n_tokens, n_altup]
        corrected = c.ggml_add(ctx, corrected, predictions); // [n_embd, n_tokens, n_altup]
        b.cb(corrected, "corrected", il);

        return corrected;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
