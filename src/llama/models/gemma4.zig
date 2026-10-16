//! Gemma 4: Gemma 3's sandwich of RMS norms with per-layer head sizes for
//! the sliding-window and global layers, proportional RoPE on the global
//! ones, an RMS-normed V that falls back to K when the file has no V
//! projection, Gemma 3n's per-layer token embeddings, an optional mixture of
//! experts beside the dense feed-forward, a per-layer output scale, and the
//! last layers reading the KV cache of earlier ones.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/gemma4.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_gemma4`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_gemma4`'s `load_arch_hparams` (models/gemma4.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    _ = try meta.getKeyOrArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, hp.nLayer(), true, exc);

    var n_kv_shared_layers: u32 = 0;
    if (try meta.getKey(u32, .attention_shared_kv_layers, false, exc)) |v| n_kv_shared_layers = v;

    // `uint32_t - int32_t` is unsigned in the C++, then narrowed.
    hp.n_layer_kv_from_start = @bitCast(hp.n_layer_all -% n_kv_shared_layers);
    hp.f_attention_scale = 1.0; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)

    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    if (try meta.getKey(u32, .expert_feed_forward_length, false, exc)) |v| hp.n_ff_exp = v;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_embd_per_layer = (try meta.getKey(u32, .embedding_length_per_layer, true, exc)).?;
    hp.n_embd_head_k_swa = (try meta.getKey(u32, .attention_key_length_swa, true, exc)).?;
    hp.n_embd_head_v_swa = (try meta.getKey(u32, .attention_value_length_swa, true, exc)).?;
    if (try meta.getKey(f32, .final_logit_softcapping, false, exc)) |v| hp.f_final_logit_softcapping = v;

    self.type = switch (hp.nLayer()) {
        30 => .@"26b_a4b",
        35 => .e2b,
        42 => .e4b,
        60 => .@"31b",
        else => .unknown,
    };
}

/// Ports `llama_model_gemma4`'s `load_arch_tensors` (models/gemma4.cpp:31 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const n_embd_per_layer: i64 = hp.n_embd_per_layer;
    const n_ff_exp: i64 = hp.n_ff_exp;

    if (l.n_embd_head_k != l.n_embd_head_v) {
        return exc.throw(.runtime_error, "Gemma 4 requires n_embd_head_k == n_embd_head_v", .{});
    }
    if (hp.n_embd_head_k_swa != hp.n_embd_head_v_swa) {
        return exc.throw(.runtime_error, "Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa", .{});
    }

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    if (n_embd_per_layer > 0) {
        self.per_layer_tok_embd = try self.createTensor(.{ .tensor = .per_layer_token_embd, .suffix = "weight" }, &.{ n_embd_per_layer * l.n_layer, l.n_vocab }, 0, exc);
        self.per_layer_model_proj = try self.createTensor(.{ .tensor = .per_layer_model_proj, .suffix = "weight", .bid = 0 }, &.{ l.n_embd, n_embd_per_layer * l.n_layer }, 0, exc);
        self.per_layer_proj_norm = try self.createTensor(.{ .tensor = .per_layer_proj_norm, .suffix = "weight", .bid = 0 }, &.{n_embd_per_layer}, 0, exc);
    }

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    var rope_freqs_flag: u32 = 0;

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const iu: u32 = @intCast(i);
        const n_head: i64 = hp.nHead(iu);
        const n_embd_head: i64 = hp.nEmbdHeadK(iu);
        const n_embd_k: i64 = hp.nEmbdKGqa(iu);
        const n_embd_v: i64 = hp.nEmbdVGqa(iu);
        const kv_flags: u32 = if (hp.hasKv(iu)) 0 else nr;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // use_alternative_attention: v_proj is optional, and k_proj is used
        // in its absence.
        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_head * n_head }, 0, exc);
        layer.wk = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_k }, kv_flags, exc);
        layer.wv = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_v }, nr, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_head * n_head, l.n_embd }, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, kv_flags, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.out_scale = try self.createTensor(.{ .tensor = .layer_out_scale, .suffix = "weight", .bid = i }, &.{1}, nr, exc);

        if (!hp.isSwa(iu)) {
            // Full-attention layers use rope_freqs for proportional RoPE.
            layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(n_embd_head, 2)}, rope_freqs_flag, exc);
            rope_freqs_flag = loader.tensor_duplicated;
        }

        // use_double_wide_mlp
        const n_ff_cur: i64 = hp.nFf(iu);

        // Expert layers use the normal FFN as the shared expert, as the
        // Python code does.
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_cur }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_cur }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ n_ff_cur, l.n_embd }, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // The MoE router.
        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, nr, exc);
        const has_expert = layer.ffn_gate_inp != null;

        // Norms.
        if (has_expert) {
            layer.ffn_gate_inp_s = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "scale", .bid = i }, &.{l.n_embd}, 0, exc);

            layer.ffn_pre_norm_2 = try self.createTensor(.{ .tensor = .ffn_pre_norm_2, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.ffn_post_norm_1 = try self.createTensor(.{ .tensor = .ffn_post_norm_1, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.ffn_post_norm_2 = try self.createTensor(.{ .tensor = .ffn_post_norm_2, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

            // The MoE FFN.
            layer.ffn_gate_up_exps = try self.createTensor(.{ .tensor = .ffn_gate_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * 2, l.n_expert }, nr, exc);

            if (layer.ffn_gate_up_exps == null) {
                layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
                layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            }

            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);

            // The per-expert scale is loaded as down_exps_s by the shared
            // pass after this one.
        }

        // Per-layer embeddings.
        if (n_embd_per_layer > 0) {
            layer.per_layer_inp_gate = try self.createTensor(.{ .tensor = .per_layer_inp_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_per_layer }, 0, exc);
            layer.per_layer_proj = try self.createTensor(.{ .tensor = .per_layer_proj, .suffix = "weight", .bid = i }, &.{ n_embd_per_layer, l.n_embd }, 0, exc);
            layer.per_layer_post_norm = try self.createTensor(.{ .tensor = .per_layer_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        }
    }
}

/// Ports `llama_model_gemma4`'s `build_arch_graph` (models/gemma4.cpp:134 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const g: Graph = .{ .model = model, .b = b, .n_embd_per_layer = b.hparams.n_embd_per_layer };
    g.build();
}

/// Ports `ggml_view_2d_slice` (models/gemma4.cpp:139 @c1d0e7a00): the 2D
/// slice of a 3D tensor at `idx` along its third dimension.
fn view2dSlice(ctx: *c.ggml_context, x: *Tensor, idx: c_int) *Tensor {
    impl.assert(idx < @as(c_int, @intCast(x.ne[2])), "models/gemma4.cpp", 140, "idx < (int) x->ne[2]");
    return c.ggml_view_2d(ctx, x, x.ne[0], x.ne[1], c.ggml_row_size(x.type, x.ne[0]), @intCast(idx * x.ne[0] * x.ne[1] * @as(i64, @intCast(c.ggml_element_size(x)))));
}

/// Ports `llama_model_gemma4`'s `graph` (src/models/models.h:859 @c1d0e7a00).
const Graph = struct {
    model: *const Model,
    b: *const graph.Builder,
    n_embd_per_layer: i64,

    /// Ports the constructor of `llama_model_gemma4`'s `graph`
    /// (models/gemma4.cpp:145 @c1d0e7a00).
    fn build(self: *const Graph) void {
        const model = self.model;
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const cp = b.cparams;
        const n_tokens = b.n_tokens;

        var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

        // Important: do not normalize weights for raw embeddings input (i.e.
        // encoded image embeddings).
        inpL = c.ggml_scale(ctx, inpL, if (b.ubatch.token != null) impl.sqrtf(@floatFromInt(b.n_embd)) else 1.0);
        b.cb(inpL, "inp_scaled", -1);

        // The positions.
        const inp_pos = b.buildInpPos();

        const inp_attn = b.buildAttnInpKvIswa();

        const inp_out_ids = b.buildInpOutIds();

        var inp_per_layer: ?*Tensor = null;
        if (model.per_layer_tok_embd != null) {
            const ipl = self.buildInpPerLayer();
            c.ggml_build_forward_expand(b.gf, ipl);

            // inp_per_layer has shape [n_embd_per_layer, n_tokens, n_layer].
            inp_per_layer = self.projectPerLayerInputs(inpL, ipl);
        }

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];
            const ilu: u32 = @intCast(il);
            const n_embd_head: i64 = hp.nEmbdHeadK(ilu);
            impl.assert(n_embd_head == hp.nEmbdHeadV(ilu), "models/gemma4.cpp", 177, "n_embd_head == hparams.n_embd_head_v(il)");

            const n_head: i64 = hp.nHead(ilu);
            const n_head_kv: i64 = hp.nHeadKv(ilu);

            const freq_base_l = model.getRopeFreqBase(cp, ilu);
            const freq_scale_l = model.getRopeFreqScale(cp, ilu);
            const n_rot_l: c_int = @intCast(hp.nRot(ilu));

            b.res.t_layer_inp[@intCast(il)] = inpL;

            // Norm.
            var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);

            var freq_factors: ?*Tensor = null;
            if (!hp.isSwa(ilu)) {
                // Full-attention layers use rope_freqs for proportional RoPE.
                freq_factors = layer.rope_freqs;
            }

            // The Q projection, shared by the KV and the non-KV layers, to
            // mirror Gemma4Attention in the PyTorch code.
            var Qcur: *Tensor = undefined;
            {
                Qcur = b.buildLoraMm(layer.wq.?, cur, layer.wq_s);
                b.cb(Qcur, "Qcur", il);

                Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens);

                Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
                b.cb(Qcur, "Qcur_normed", il);

                Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, freq_factors, n_rot_l, b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                b.cb(Qcur, "Qcur_pos", il);
            }

            // Self-attention.
            if (hp.hasKv(ilu)) {
                var Kcur: *Tensor = b.buildLoraMm(layer.wk.?, cur, layer.wk_s);
                b.cb(Kcur, "Kcur", il);

                // Without a v_proj, Kcur is Vcur.
                var Vcur: *Tensor = if (layer.wv) |wv| b.buildLoraMm(wv, cur, layer.wv_s) else Kcur;
                b.cb(Vcur, "Vcur", il);

                Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

                Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
                Vcur = c.ggml_rms_norm(ctx, Vcur, hp.f_norm_rms_eps);

                b.cb(Kcur, "Kcur_normed", il);
                b.cb(Vcur, "Vcur_normed", il);

                Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, freq_factors, n_rot_l, b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

                b.cb(Kcur, "Kcur_pos", il);

                cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, hp.f_attention_scale, il);
            } else {
                // Reuse the KV cache of earlier layers.
                cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, null, null, null, null, null, hp.f_attention_scale, il);
            }

            // TODO @ngxson : strip unused token right after the last KV layer
            // to speed up prompt processing. Keep all rows when extracting
            // unmasked nextn embeddings: the MTP target needs the hidden
            // state for every token.
            if (il == b.n_layer - 1 and cp.embeddings_nextn_masked) {
                cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
                inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
            }
            cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
            b.cb(cur, "attn_post_norm", il);

            const attn_out: *Tensor = c.ggml_add(ctx, cur, inpL);
            b.cb(attn_out, "attn_out", il);

            // The feed-forward network.
            const is_moe_layer = layer.ffn_gate_inp != null;
            if (is_moe_layer) {
                // The MLP, as the shared expert.
                var cur_mlp = b.buildNorm(attn_out, layer.ffn_norm, null, .rms, il);
                b.cb(cur_mlp, "ffn_norm_1", il);

                cur_mlp = b.buildFfn(cur_mlp, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .gelu, .par, il);
                cur_mlp = b.buildNorm(cur_mlp, layer.ffn_post_norm_1, null, .rms, il);
                b.cb(cur_mlp, "ffn_mlp", il);

                // The expert FFN.
                var cur_moe = b.buildNorm(attn_out, layer.ffn_pre_norm_2, null, .rms, il);
                b.cb(cur_moe, "ffn_norm_2", il);

                // Custom MoE logits: the router works on attn_out, not cur.
                var tmp: *Tensor = c.ggml_rms_norm(ctx, attn_out, hp.f_norm_rms_eps);
                tmp = c.ggml_scale(ctx, tmp, 1.0 / impl.sqrtf(@floatFromInt(b.n_embd)));
                tmp = c.ggml_mul(ctx, tmp, layer.ffn_gate_inp_s);
                const logits = b.buildLoraMm(layer.ffn_gate_inp.?, tmp, null); // [n_expert, n_tokens]
                b.cb(logits, "ffn_moe_logits", il);

                cur_moe = b.buildMoeFfn(cur_moe, .{
                    .gate_inp = null,
                    .up_exps = layer.ffn_up_exps,
                    .gate_exps = layer.ffn_gate_exps,
                    .down_exps = layer.ffn_down_exps.?,
                    .exp_probs_b = null, // not used by Gemma 4
                    .n_expert = b.n_expert,
                    .n_expert_used = b.n_expert_used,
                    .type_op = .gelu,
                    .norm_w = true,
                    .w_scale = 1.0,
                    .gating_op = .softmax,
                    .probs_in = logits,
                    .gate_up_exps = layer.ffn_gate_up_exps,
                    .up_exps_s = layer.ffn_up_exps_s,
                    .gate_exps_s = layer.ffn_gate_exps_s,
                    .down_exps_s = layer.ffn_down_exps_s,
                }, il);
                cur_moe = b.buildNorm(cur_moe, layer.ffn_post_norm_2, null, .rms, il);
                b.cb(cur_moe, "ffn_moe", il);

                cur = c.ggml_add(ctx, cur_mlp, cur_moe);
                b.cb(cur, "ffn_moe_combined", il);
            } else {
                cur = b.buildNorm(attn_out, layer.ffn_norm, null, .rms, il);
                b.cb(cur, "ffn_norm", il);

                cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .gelu, .par, il);
                b.cb(cur, "ffn_out", il);
            }
            cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, -1);
            b.cb(cur, "ffn_post_norm", il);

            // The residual connection.
            cur = c.ggml_add(ctx, cur, attn_out);

            // The per-layer embedding.
            if (inp_per_layer) |ipl| {
                const pe_in = cur;
                b.cb(cur, "pe_in", il);

                cur = b.buildLoraMm(layer.per_layer_inp_gate.?, cur, null); // [n_embd_per_layer, n_tokens]
                cur = c.ggml_gelu(ctx, cur);

                var inp_this_layer = view2dSlice(ctx, ipl, il); // [n_embd_per_layer, n_tokens]

                // TODO @ngxson : improve this
                if (il == b.n_layer - 1 and cp.embeddings_nextn_masked) {
                    inp_this_layer = c.ggml_get_rows(ctx, inp_this_layer, inp_out_ids);
                }

                cur = c.ggml_mul(ctx, cur, inp_this_layer);
                cur = b.buildLoraMm(layer.per_layer_proj.?, cur, null); // [n_embd, n_tokens]
                cur = b.buildNorm(cur, layer.per_layer_post_norm, null, .rms, il);
                b.cb(cur, "per_layer_embd_out", il);

                // The residual connection.
                cur = c.ggml_add(ctx, pe_in, cur);
            }

            // layer_scalar
            if (layer.out_scale) |s| {
                cur = c.ggml_mul(ctx, cur, s);
                b.cb(cur, "out_scaled", il);
            }

            cur = b.buildCvec(cur, il);
            b.cb(cur, "l_out", il);

            // The next layer's input.
            inpL = cur;
        }
        var cur = inpL;

        cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

        // Expose the post-output-norm hidden state (the LM head's input) so
        // MTP draft contexts can read it through
        // llama_get_embeddings_nextn_ith() as the recurrent h input. This
        // matches the reference (transformers/vLLM/SGLang), which feeds the
        // drafter the target's post-final-norm hidden state.
        b.cb(cur, "h_nextn", -1);
        b.res.t_h_nextn = cur;

        if (!cp.embeddings_nextn_masked) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
        }

        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        // The LM head.
        cur = b.buildLoraMm(model.output.?, cur, model.output_s);

        if (hp.f_final_logit_softcapping != 0.0) {
            cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
            cur = c.ggml_tanh(ctx, cur);
            cur = c.ggml_scale(ctx, cur, hp.f_final_logit_softcapping);
        }

        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;

        c.ggml_build_forward_expand(b.gf, cur);
    }

    /// Ports `llama_model_gemma4`'s `graph`'s `build_inp_per_layer`
    /// (models/gemma4.cpp:413 @c1d0e7a00): equivalent to the Python code's
    /// `get_per_layer_inputs()`.
    ///
    /// Return: `[n_embd_per_layer, n_layer, n_tokens]`.
    fn buildInpPerLayer(self: *const Graph) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const model = self.model;

        var inp_per_layer: *Tensor = undefined;
        const tok_embd_scale = impl.sqrtf(@floatFromInt(self.n_embd_per_layer));
        if (b.ubatch.token != null) {
            const inp = impl.allocator.create(graph.InpEmbd) catch impl.abort("models/gemma4.cpp", 414, "out of memory");
            inp.* = .{ .n_embd = b.n_embd };
            inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.ubatch.n_tokens);
            c.ggml_set_input(inp.tokens);
            b.res.t_inp_tokens = inp.tokens;

            inp_per_layer = c.ggml_get_rows(ctx, model.per_layer_tok_embd, inp.tokens);
            inp_per_layer = c.ggml_reshape_3d(ctx, inp_per_layer, self.n_embd_per_layer, b.n_layer, b.n_tokens);
            inp_per_layer = c.ggml_scale(ctx, inp_per_layer, tok_embd_scale);
            b.cb(inp_per_layer, "inp_per_layer_selected", -1);

            _ = b.res.addInput(graph.InpEmbd, inp);
        } else {
            // The multimodal embedding path uses the padding token's (ID 0)
            // embedding.
            // TODO: verify if this is the correct behavior in transformers
            // implementation.
            const embd_size = model.per_layer_tok_embd.?.ne[0]; // n_embd_per_layer * n_layer

            // Extract and dequantize the padding token's embedding (row 0).
            const padding: *Tensor = c.ggml_view_1d(ctx, model.per_layer_tok_embd, embd_size, 0);
            inp_per_layer = c.ggml_cast(ctx, padding, c.GGML_TYPE_F32);
            inp_per_layer = c.ggml_scale(ctx, inp_per_layer, tok_embd_scale);

            // Reshape to [n_embd_per_layer, n_layer, 1].
            inp_per_layer = c.ggml_reshape_3d(ctx, inp_per_layer, self.n_embd_per_layer, b.n_layer, 1);
            b.cb(inp_per_layer, "inp_per_layer_multimodal", -1);
        }
        return inp_per_layer;
    }

    /// Ports `llama_model_gemma4`'s `graph`'s `project_per_layer_inputs`
    /// (models/gemma4.cpp:451 @c1d0e7a00): equivalent to the Python code's
    /// `project_per_layer_inputs()`, the per-layer inputs with `n_layer` as
    /// the last dimension, from `inp_batch` `[n_embd, n_tokens]` and
    /// `build_inp_per_layer`'s `[n_embd_per_layer, n_layer, n_tokens]`.
    ///
    /// Return: `[n_embd_per_layer, n_tokens, n_layer]`.
    fn projectPerLayerInputs(self: *const Graph, inp_batch: *Tensor, inp_per_layer_: *Tensor) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const per_layer_projection_scale = 1.0 / impl.sqrtf(@floatFromInt(b.n_embd));
        const per_layer_input_scale = 1.0 / impl.sqrtf(2.0);

        // This product is computed in the input layer, i.e. on the CPU.
        var per_layer_proj: *Tensor = c.ggml_mul_mat(ctx, self.model.per_layer_model_proj, inp_batch);
        per_layer_proj = c.ggml_scale(ctx, per_layer_proj, per_layer_projection_scale);
        per_layer_proj = c.ggml_reshape_3d(ctx, per_layer_proj, self.n_embd_per_layer, b.n_layer, b.n_tokens);

        per_layer_proj = b.buildNorm(per_layer_proj, self.model.per_layer_proj_norm, null, .rms, -1);
        b.cb(per_layer_proj, "per_layer_proj", -1);

        var inp_per_layer: *Tensor = c.ggml_add(ctx, per_layer_proj, inp_per_layer_);
        inp_per_layer = c.ggml_scale(ctx, inp_per_layer, per_layer_input_scale);
        b.cb(inp_per_layer, "inp_per_layer", -1);

        // Permute to [n_embd_per_layer, n_tokens, n_layer].
        inp_per_layer = c.ggml_cont(ctx, c.ggml_permute(ctx, inp_per_layer, 0, 2, 1, 3));
        return inp_per_layer;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
