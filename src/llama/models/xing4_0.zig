//! Xing4.0: DeepSeek-V2's multi-head latent attention and mixture of
//! experts with shared experts, carried over four residual streams by
//! manifold-constrained hyper-connections (DeepSeek-V4's, with the
//! combination matrix read transposed and its logits clamped before
//! Sinkhorn), and an optional NextN/MTP block past the main stack.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/xing4-0.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_xing4_0`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph` with `build_hc_pre`, `build_hc_sinkhorn`
//! and `build_hc_post`, its `graph_mtp`, and the file's static helpers.

const std = @import("std");
const libm = @import("stdlib").libm;
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
const Builder = graph.Builder;

const src = "models/xing4-0.cpp";

// -----------------------------------------------------------------------------
// Helpers

/// Ports `xing4_0_elem_offset` (models/xing4-0.cpp:7 @b1b770d90): the byte
/// offset of element `i` along a row.
fn elemOffset(t: *const Tensor, i: i64) usize {
    return c.ggml_row_size(t.type, i);
}

/// Ports `xing4_0_view_1d` (models/xing4-0.cpp:11 @b1b770d90).
fn view1d(ctx: *c.ggml_context, t: *Tensor, ne0: i64, j0: i64) *Tensor {
    return c.ggml_view_1d(ctx, t, ne0, elemOffset(t, j0));
}

/// Ports `xing4_0_view_2d` (models/xing4-0.cpp:15 @b1b770d90).
fn view2d(ctx: *c.ggml_context, t: *Tensor, ne0: i64, ne1: i64, j0: i64) *Tensor {
    return c.ggml_view_2d(ctx, t, ne0, ne1, t.nb[1], elemOffset(t, j0));
}

/// Ports `xing4_0_hc_mean` (models/xing4-0.cpp:25 @b1b770d90): the mean
/// over the hyper-connection streams, `[n_embd, hc, n_tokens]` to
/// `[n_embd, n_tokens]`.
fn hcMean(ctx: *c.ggml_context, x: *Tensor) *Tensor {
    const hc = x.ne[1];

    var acc: *Tensor = c.ggml_view_2d(ctx, x, x.ne[0], x.ne[2], x.nb[2], 0);
    var s: i64 = 1;
    while (s < hc) : (s += 1) {
        acc = c.ggml_add(ctx, acc, c.ggml_view_2d(ctx, x, x.ne[0], x.ne[2], x.nb[2], @as(usize, @intCast(s)) * x.nb[1]));
    }
    return c.ggml_scale(ctx, acc, 1.0 / @as(f32, @floatFromInt(hc)));
}

/// Ports `xing4_0_hc_affine` (models/xing4-0.cpp:35 @b1b770d90).
fn hcAffine(ctx: *c.ggml_context, x_: *Tensor, scale: *Tensor, base: *Tensor) *Tensor {
    var x: *Tensor = c.ggml_mul(ctx, x_, scale);
    x = c.ggml_add(ctx, x, base);
    return x;
}

// -----------------------------------------------------------------------------
// Loading

/// Ports `llama_model_xing4_0`'s `load_arch_hparams` (models/xing4-0.cpp:45 @b1b770d90).
///
/// Parameters:
/// - `self`: the model whose hyperparameters are filled.
/// - `ml`: the loader whose metadata is read.
/// - `exc`: set when a key is missing or mistyped.
///
/// Return: nothing; `error.Thrown` with `exc` set on failure.
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    // Read, as the C++ reads it, and left unused.
    var n_vocab: u32 = 0;
    if (try meta.getKey(u32, .vocab_size, false, exc)) |v| {
        n_vocab = v;
    } else if (try meta.getArrN(.tokenizer_list, false, exc)) |v| {
        n_vocab = v;
    }
    _ = &n_vocab;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    if (try meta.getKey(u32, .attention_key_length_mla, false, exc)) |v| hp.n_embd_head_k_mla_impl = v;
    if (try meta.getKey(u32, .attention_value_length_mla, false, exc)) |v| hp.n_embd_head_v_mla_impl = v;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        // The fallback for GGUFs with no `expert_gating_func`.
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.softmax);
    }

    // The manifold-constrained hyper-connections, in DeepSeek-V4's
    // hparams members: the saver writes them unconditionally.
    hp.dsv4_hc_mult = (try meta.getKey(u32, .hyper_connection_count, true, exc)).?;
    hp.dsv4_hc_sinkhorn_iters = (try meta.getKey(u32, .hyper_connection_sinkhorn_iterations, true, exc)).?;
    hp.dsv4_hc_eps = (try meta.getKey(f32, .hyper_connection_epsilon, true, exc)).?;
    impl.assert(hp.dsv4_hc_mult == 4, src, 70, "hparams.dsv4_hc_mult == 4 && \"xing4_0 requires 4 residual streams\"");

    if (try meta.getKey(f32, .rope_scaling_yarn_log_mul, false, exc)) |v| {
        hp.rope_yarn_log_mul = v;
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] Cancel the factor the convert
        // script applied.
        hp.rope_yarn_log_mul /= 0.1;
    }
}

/// Ports `llama_model_xing4_0`'s `load_arch_tensors` (models/xing4-0.cpp:79 @b1b770d90).
///
/// Parameters:
/// - `self`: the model whose tensors are created.
/// - `ml`: the loader the tensors come from.
/// - `exc`: set when a required tensor is missing or the experts are absent.
///
/// Return: nothing; `error.Thrown` with `exc` set on failure.
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const n_expert_shared: i64 = hp.n_expert_shared;
    const hc_dim: i64 = @as(i64, hp.dsv4_hc_mult) * l.n_embd;
    const hc_mix_dim: i64 = (2 + @as(i64, hp.dsv4_hc_mult)) * @as(i64, hp.dsv4_hc_mult);

    // `blk.0.attn_norm.weight` absent: an MTP-only file; the first NextN
    // block's `eh_proj` absent: a trunk-only one; and TENSOR_SKIP on the
    // MTP block unless it is loaded.
    const nf = self.nextnFlags(ml, .attn_norm);
    const trunk_flags = nf.trunk;
    const mtp_flags = nf.mtp;

    impl.assert(hp.isMla(), src, 95, "hparams.is_mla() && \"xing4_0 requires MLA\"");
    impl.assert(hp.n_lora_q > 0, src, 96, "hparams.n_lora_q > 0 && \"xing4_0 requires q_lora_rank\"");

    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_v_mla: i64 = hp.nEmbdHeadVMla();

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    impl.assert(n_embd_head_qk_nope >= 1, src, 103, "n_embd_head_qk_nope >= 1");

    const q_lora_rank: i64 = hp.n_lora_q;
    const kv_lora_rank: i64 = hp.n_lora_kv;

    const n_ff_exp: i64 = hp.nFfExp(0);

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    // The output, or failing that the token embeddings (tied embeddings).
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const flags = if (i < l.n_layer) trunk_flags else mtp_flags;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, flags, exc);
        layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, flags, exc);

        layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_lora_rank }, flags, exc);
        layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, l.n_head * n_embd_head_k_mla }, flags, exc);

        layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ l.n_embd, kv_lora_rank + n_embd_head_qk_rope }, flags, exc);

        layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_qk_nope, kv_lora_rank, l.n_head }, flags, exc);
        layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v_mla, l.n_head }, flags, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_head * n_embd_head_v_mla, l.n_embd }, flags, exc);

        // The hyper-connection tensors, trunk layers only.
        if (i < l.n_layer) {
            layer.hc_attn_fn = try self.createTensor(.{ .tensor = .hc_attn_fn, .suffix = "weight", .bid = i }, &.{ hc_dim, hc_mix_dim }, flags, exc);
            layer.hc_attn_base = try self.createTensor(.{ .tensor = .hc_attn_base, .suffix = "weight", .bid = i }, &.{hc_mix_dim}, flags, exc);
            layer.hc_attn_scale = try self.createTensor(.{ .tensor = .hc_attn_scale, .suffix = "weight", .bid = i }, &.{3}, flags, exc);
            layer.hc_ffn_fn = try self.createTensor(.{ .tensor = .hc_ffn_fn, .suffix = "weight", .bid = i }, &.{ hc_dim, hc_mix_dim }, flags, exc);
            layer.hc_ffn_base = try self.createTensor(.{ .tensor = .hc_ffn_base, .suffix = "weight", .bid = i }, &.{hc_mix_dim}, flags, exc);
            layer.hc_ffn_scale = try self.createTensor(.{ .tensor = .hc_ffn_scale, .suffix = "weight", .bid = i }, &.{3}, flags, exc);
        }

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, flags, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
        } else {
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, flags, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr | flags, exc);

            if (l.n_expert == 0) {
                return exc.throw(.runtime_error, "n_expert must be > 0", .{});
            }
            if (l.n_expert_used == 0) {
                return exc.throw(.runtime_error, "n_expert_used must be > 0", .{});
            }

            // The routed experts.
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
            try self.createTensorGateUpExps(layer, i, l.n_embd, n_ff_exp, l.n_expert, flags, exc);

            // The shared experts.
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, flags, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, flags, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, flags, exc);
        }

        // The NextN/MTP tensors.
        if (i >= l.n_layer) {
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr | flags, exc);
            layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr | flags, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr | flags, exc);
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_xing4_0`'s `build_arch_graph` (models/xing4-0.cpp:188
/// @b1b770d90): the MTP draft head for that graph type, the main model
/// otherwise.
///
/// Parameters:
/// - `model`: the model.
/// - `b`: the graph builder.
/// - `params`: the graph's parameters; `gtype` picks the graph.
pub fn buildArchGraph(model: *const Model, b: *const Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

/// The YaRN pre-scaling both graphs share: the KQ scale, from the
/// `attn_factor` `llama_hparams`'s `yarn_attn_factor_adjust` adjusted.
fn mlaKqScale(b: *const Builder, n_embd_head_k: i64) f32 {
    const hp = b.hparams;
    // First cancel the adjustment, to get the original attn_factor.
    const attn_factor_org = b.attn_factor * (1.0 + 0.1 * libm.logf(1.0 / b.freq_scale));
    // Then pre-scale the KQ scale with it.
    const mscale = attn_factor_org * (1.0 + 0.1 * hp.rope_yarn_log_mul * libm.logf(1.0 / b.freq_scale));
    return 1.0 * mscale * mscale / libm.sqrtfInt(n_embd_head_k);
}

/// Ports the constructor of `llama_model_xing4_0`'s `graph_mtp`
/// (models/xing4-0.cpp:195 @b1b770d90).
fn buildGraphMtp(model: *const Model, b: *const Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, src, 197, "hparams.n_layer_nextn > 0 && \"xing4_0 MTP requires n_layer_nextn > 0\"");
    impl.assert(hp.n_layer_nextn == 1, src, 198, "hparams.n_layer_nextn == 1 && \"xing4_0 MTP currently only supports a single MTP block\"");
    impl.assert(hp.isMla(), src, 199, "hparams.is_mla() && \"xing4_0 MTP requires MLA\"");

    // The appended MTP block is stored straight after the decoder layers.
    const il: c_int = @intCast(hp.nLayer());
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, src, 205, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, src, 206, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, src, 207, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");

    impl.assert(@as(u32, @intCast(il)) >= hp.n_layer_dense_lead, src, 209, "(uint32_t) il >= hparams.n_layer_dense_lead && \"xing4_0 MTP block expected to use MoE FFN\"");

    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const kv_lora_rank: i64 = hp.n_lora_kv;

    impl.assert(n_embd_head_qk_nope >= 1, src, 216, "n_embd_head_qk_nope >= 1");
    impl.assert(hp.n_lora_q > 0, src, 217, "hparams.n_lora_q > 0");

    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort(src, 219, "out of memory");
    inp.* = .{ .n_embd = hp.nEmbdInp(), .n_embd_state = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    c.ggml_set_input(inp.embd);

    var tok_embd: *Tensor = undefined;
    b.ubatch.assertEmbdOrToken(src, 228);
    if (b.ubatch.token != null) {
        const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;
        tok_embd = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    } else {
        tok_embd = inp.embd.?;
    }
    b.cb(tok_embd, "mtp_tok_embd", il);

    inp.h = b.buildInpEmbdState(inp, hp.n_embd);

    const h_embd = inp.h.?;

    _ = b.res.addInput(graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    const inp_attn_k = b.buildAttnInpK();

    const h_norm = b.buildNorm(h_embd, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat: *Tensor = c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(cur, "mtp_eh_proj", il);

    const inpSA = cur;

    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    var q: *Tensor = c.ggml_mul_mat(ctx, layer.wq_a, cur);
    b.cb(q, "mtp_q_a", il);

    q = b.buildNorm(q, layer.attn_q_a_norm, null, .rms, il);
    b.cb(q, "mtp_q_a_norm", il);

    q = c.ggml_mul_mat(ctx, layer.wq_b, q);
    b.cb(q, "mtp_q_b", il);

    var q_nope: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_nope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k_mla), c.ggml_row_size(q.type, n_embd_head_k_mla) * @as(usize, @intCast(b.n_head)), 0);
    b.cb(q_nope, "mtp_q_nope", il);

    var q_pe: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_rope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k_mla), c.ggml_row_size(q.type, n_embd_head_k_mla) * @as(usize, @intCast(b.n_head)), c.ggml_row_size(q.type, n_embd_head_qk_nope));
    b.cb(q_pe, "mtp_q_pe", il);

    const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
    b.cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

    var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), 0);
    b.cb(kv_cmpr, "mtp_kv_cmpr", il);

    var k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, n_embd_head_qk_rope, 1, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));
    b.cb(k_pe, "mtp_k_pe", il);

    kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
    b.cb(kv_cmpr, "mtp_kv_cmpr_norm", il);

    impl.assert(b.ext_factor >= 0.0, src, 308, "ext_factor >= 0.0f");

    const kq_scale = mlaKqScale(b, n_embd_head_k_mla);

    q_pe = b.buildRope(q_pe, inp_pos, il);
    b.cb(q_pe, "mtp_q_pe_rope", il);

    k_pe = b.buildRope(k_pe, inp_pos, il);
    b.cb(k_pe, "mtp_k_pe_rope", il);

    q_nope = c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    b.cb(q_nope, "mtp_q_nope_perm", il);

    var q_nope_absorbed: *Tensor = c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
    b.cb(q_nope_absorbed, "mtp_q_nope_absorbed", il);

    q_nope_absorbed = c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);
    b.cb(q_nope_absorbed, "mtp_q_nope_absorbed_perm", il);

    const Qcur: *Tensor = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
    b.cb(Qcur, "mtp_Qcur", il);

    kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, hp.n_lora_kv, 1, b.n_tokens);
    b.cb(kv_cmpr, "mtp_kv_cmpr_reshape", il);

    const Kcur: *Tensor = c.ggml_concat(ctx, kv_cmpr, k_pe, 0);
    b.cb(Kcur, "mtp_Kcur", il);

    const Vcur = kv_cmpr;
    b.cb(Vcur, "mtp_Vcur", il);

    cur = b.buildAttn(inp_attn_k, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, layer.wv_b, kq_scale, il);
    b.cb(cur, "mtp_attn_out", il);

    const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "mtp_ffn_inp", il);

    cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "mtp_ffn_norm", il);

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
        .gate_up_exps = layer.ffn_gate_up_exps,
    }, il);
    b.cb(moe_out, "mtp_ffn_moe_out", il);

    const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
    b.cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = c.ggml_add(ctx, moe_out, ffn_shexp);
    b.cb(cur, "mtp_ffn_out", il);

    cur = c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "mtp_post_ffn", il);

    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    impl.assert(head_norm_w != null, src, 388, "head_norm_w && \"xing4_0 MTP: missing both nextn.shared_head_norm and output_norm\"");

    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    // `inp_out_ids` is always built here, so the C++'s null check holds.
    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;

    impl.assert(head_w != null, src, 407, "head_w && \"xing4_0 MTP: missing LM head (nextn.shared_head_head or model.output)\"");

    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
}

/// Ports the overload of `llama_model_xing4_0`'s graph's `build_hc_pre`
/// that takes the weights (models/xing4-0.cpp:415 @b1b770d90): the
/// streams' weighted sum, `[n_embd, hc, n_tokens]` by `[hc, n_tokens]`.
fn buildHcPreWeighted(b: *const Builder, x: *Tensor, weights: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    impl.assert(x.ne[0] == b.n_embd, src, 421, "x->ne[0] == n_embd");
    impl.assert(x.ne[1] == b.hparams.dsv4_hc_mult, src, 422, "x->ne[1] == hparams.dsv4_hc_mult");

    const hc: i64 = b.hparams.dsv4_hc_mult;
    const nt = x.ne[2];

    if (b.cparams.fused_dsv4_hc_pre and il >= 0) {
        const result: *Tensor = c.ggml_dsv4_hc_pre(ctx, x, weights);
        b.res.addFusedNode(.{ .op = .dsv4_hc_pre, .tensor = result, .il = il });
        return result;
    }

    var result: ?*Tensor = null;
    var ih: i64 = 0;
    while (ih < hc) : (ih += 1) {
        const xh: *Tensor = c.ggml_view_2d(ctx, x, b.n_embd, nt, x.nb[2], @as(usize, @intCast(ih)) * x.nb[1]);
        const wh: *Tensor = c.ggml_view_2d(ctx, weights, 1, nt, weights.nb[1], @as(usize, @intCast(ih)) * weights.nb[0]);
        const cur: *Tensor = c.ggml_mul(ctx, xh, wh);
        result = if (result) |r| c.ggml_add(ctx, r, cur) else cur;
    }

    return result.?;
}

/// Ports `llama_model_xing4_0`'s graph's `build_hc_sinkhorn`
/// (models/xing4-0.cpp:444 @b1b770d90). `comb` is `[src_hc, dst_hc,
/// n_tokens]`; unlike DeepSeek-V4's, no epsilon is added after the
/// softmax, which keeps the reference kernel's semantics.
fn buildHcSinkhorn(b: *const Builder, comb_: *Tensor, il: c_int) *Tensor {
    _ = il;
    const ctx = b.ctx0;

    var comb: *Tensor = c.ggml_soft_max(ctx, comb_);

    var eps: *Tensor = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 1);
    eps = c.ggml_fill(ctx, eps, b.hparams.dsv4_hc_eps);

    const Norm = struct {
        /// Over ne1 (dst): each src row sums to 1.
        fn dst(ctx_: *c.ggml_context, comb_p: **Tensor, eps_: *Tensor) void {
            var sum: *Tensor = c.ggml_cont(ctx_, c.ggml_permute(ctx_, comb_p.*, 1, 0, 2, 3));
            sum = c.ggml_sum_rows(ctx_, sum);
            sum = c.ggml_add(ctx_, sum, eps_);
            sum = c.ggml_permute(ctx_, sum, 1, 0, 2, 3);
            comb_p.* = c.ggml_div(ctx_, comb_p.*, sum);
        }

        /// Over ne0 (src): each dst column sums to 1.
        fn srcs(ctx_: *c.ggml_context, comb_p: **Tensor, eps_: *Tensor) void {
            var sum: *Tensor = c.ggml_sum_rows(ctx_, comb_p.*);
            sum = c.ggml_add(ctx_, sum, eps_);
            comb_p.* = c.ggml_div(ctx_, comb_p.*, sum);
        }
    };

    Norm.dst(ctx, &comb, eps);
    var i: u32 = 1;
    while (i < b.hparams.dsv4_hc_sinkhorn_iters) : (i += 1) {
        Norm.srcs(ctx, &comb, eps);
        Norm.dst(ctx, &comb, eps);
    }

    return comb;
}

/// What `build_hc_pre` returns, and writes through its `post` and `comb`
/// out-parameters: `cur` is the collapsed input, `post` and `comb` feed
/// `buildHcPost`.
const HcPre = struct {
    cur: *Tensor,
    post: *Tensor,
    comb: *Tensor,
};

/// Ports `llama_model_xing4_0`'s graph's `build_hc_pre`
/// (models/xing4-0.cpp:481 @b1b770d90): a hyper-connection's mixes, and the
/// streams collapsed with their pre weights.
fn buildHcPre(b: *const Builder, x: *Tensor, hc_fn: *Tensor, hc_scale: *Tensor, hc_base: *Tensor, il: c_int) HcPre {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const hc: i64 = hp.dsv4_hc_mult;
    const hc_dim = hc * b.n_embd;
    const hc_mix_dim = (2 + hc) * hc;
    const nt = x.ne[2];

    impl.assert(hc == 4, src, 494, "hc == 4");
    impl.assert(hc_fn.ne[1] == hc_mix_dim, src, 495, "hc_fn->ne[1] == hc_mix_dim");

    const flat: *Tensor = c.ggml_reshape_2d(ctx, x, hc_dim, nt);
    const flat_norm: *Tensor = c.ggml_rms_norm(ctx, flat, b.norm_rms_eps);
    const mixes: *Tensor = c.ggml_mul_mat(ctx, hc_fn, flat_norm);
    b.cb(mixes, "hc_mixes", il);

    const scale_pre = view1d(ctx, hc_scale, 1, 0);
    const scale_post = view1d(ctx, hc_scale, 1, 1);

    const base_pre = view1d(ctx, hc_base, hc, 0);
    const base_post = view1d(ctx, hc_base, hc, hc);

    var pre = view2d(ctx, mixes, hc, nt, 0);
    pre = hcAffine(ctx, pre, scale_pre, base_pre);
    // The reference with_clamp kernel has no hc_pre_eps on the pre gate.
    pre = c.ggml_sigmoid(ctx, pre);
    b.cb(pre, "hc_pre", il);

    var post = view2d(ctx, mixes, hc, nt, hc);
    post = hcAffine(ctx, post, scale_post, base_post);
    post = c.ggml_sigmoid(ctx, post);
    post = c.ggml_scale(ctx, post, 2.0);
    b.cb(post, "hc_post", il);

    var comb: *Tensor = undefined;
    if (b.cparams.fused_dsv4_hc_comb) {
        // The mixes buffer goes in unchanged: its flat [src + 4*dst] layout
        // makes DeepSeek-V4's dst-indexed softmax run over xing4_0's src
        // axis, so the Sinkhorn phases align and the output is [src, dst]
        // as the decomposed path expects.
        comb = c.ggml_dsv4_hc_comb(ctx, mixes, hc_scale, hc_base, hp.dsv4_hc_eps, @intCast(hp.dsv4_hc_sinkhorn_iters));
        b.res.addFusedNode(.{ .op = .dsv4_hc_comb, .tensor = comb, .il = il });
    } else {
        const scale_comb = view1d(ctx, hc_scale, 1, 2);
        const base_comb = view1d(ctx, hc_base, hc * hc, 2 * hc);

        comb = view2d(ctx, mixes, hc * hc, nt, 2 * hc);
        comb = hcAffine(ctx, comb, scale_comb, base_comb);
        // The reference MHC kernel clamps the raw comb logits before
        // Sinkhorn.
        comb = c.ggml_clamp(ctx, comb, -30.0, 30.0);
        comb = c.ggml_reshape_3d(ctx, comb, hc, hc, nt);
        comb = buildHcSinkhorn(b, comb, il);
    }
    b.cb(comb, "hc_comb", il);

    const result = buildHcPreWeighted(b, x, pre, il);
    return .{ .cur = result, .post = post, .comb = comb };
}

/// Ports `llama_model_xing4_0`'s graph's `build_hc_post`
/// (models/xing4-0.cpp:545 @b1b770d90): the block's output spread over the
/// streams by `post`, plus the residual streams mixed by `comb`, read as
/// `comb[src, dst]` (DeepSeek-V4 reads `comb[dst, src]`).
fn buildHcPost(b: *const Builder, x: *Tensor, residual: *Tensor, post: *Tensor, comb: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    impl.assert(x.ne[0] == b.n_embd, src, 553, "x->ne[0] == n_embd");
    impl.assert(residual.ne[1] == b.hparams.dsv4_hc_mult, src, 554, "residual->ne[1] == hparams.dsv4_hc_mult");

    if (b.cparams.fused_dsv4_hc_post) {
        // DeepSeek-V4's op reads comb[dst, src]; transpose.
        const comb_t: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, comb, 1, 0, 2, 3));
        const result: *Tensor = c.ggml_dsv4_hc_post(ctx, x, residual, post, comb_t);
        b.res.addFusedNode(.{ .op = .dsv4_hc_post, .tensor = result, .il = il });
        return result;
    }

    const hc: i64 = b.hparams.dsv4_hc_mult;
    const nt = x.ne[1];

    var out: ?*Tensor = null;
    var dst: i64 = 0;
    while (dst < hc) : (dst += 1) {
        const udst: usize = @intCast(dst);
        const post_dst: *Tensor = c.ggml_view_2d(ctx, post, 1, nt, post.nb[1], udst * post.nb[0]);
        var cur: *Tensor = c.ggml_mul(ctx, x, post_dst);

        var s: i64 = 0;
        while (s < hc) : (s += 1) {
            const us: usize = @intCast(s);
            const res_src: *Tensor = c.ggml_view_2d(ctx, residual, b.n_embd, nt, residual.nb[2], us * residual.nb[1]);
            // comb's element [src, dst] mixes residual stream src into
            // output stream dst.
            const comb_src_dst: *Tensor = c.ggml_view_2d(ctx, comb, 1, nt, comb.nb[2], us * comb.nb[0] + udst * comb.nb[1]);
            cur = c.ggml_add(ctx, cur, c.ggml_mul(ctx, res_src, comb_src_dst));
        }

        cur = c.ggml_reshape_3d(ctx, cur, b.n_embd, 1, nt);
        out = if (out) |o| c.ggml_concat(ctx, o, cur, 1) else cur;
    }

    return out.?;
}

/// Ports the constructor of `llama_model_xing4_0`'s `graph`
/// (models/xing4-0.cpp:591 @b1b770d90).
fn buildGraph(model: *const Model, b: *const Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const hc: i64 = hp.dsv4_hc_mult;

    const n_embd_head_k: i64 = hp.nEmbdHeadKMla();

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const kv_lora_rank: i64 = hp.n_lora_kv;

    // The KQ scale and attn_factor are pre-scaled for YaRN
    // ([TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]).
    impl.assert(b.ext_factor >= 0.0, src, 607, "ext_factor >= 0.0f");
    const kq_scale = mlaKqScale(b, n_embd_head_k);

    var cur: *Tensor = undefined;

    const inp = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();

    const inp_attn_k = b.buildAttnInpK();

    const inp_out_ids = b.buildInpOutIds();

    var inpL: *Tensor = c.ggml_reshape_3d(ctx, inp, b.n_embd, 1, b.n_tokens);
    inpL = c.ggml_repeat_4d(ctx, inpL, b.n_embd, hc, b.n_tokens, 1);
    b.cb(inpL, "hc_init", -1);

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        var residual = inpL;

        const attn_hc = buildHcPre(b, inpL, layer.hc_attn_fn.?, layer.hc_attn_scale.?, layer.hc_attn_base.?, il);
        cur = attn_hc.cur;
        var post = attn_hc.post;
        var comb = attn_hc.comb;
        b.cb(cur, "hc_attn_pre", il);

        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // MLA self-attention, as DeepSeek-V2's.
        var q: *Tensor = c.ggml_mul_mat(ctx, layer.wq_a, cur);
        b.cb(q, "q_a", il);

        q = b.buildNorm(q, layer.attn_q_a_norm, null, .rms, il);
        b.cb(q, "q_anorm", il);

        q = c.ggml_mul_mat(ctx, layer.wq_b, q);
        b.cb(q, "q_b", il);

        // {n_embd_head_qk_nope, n_head, n_tokens}
        var q_nope: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_nope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * @as(usize, @intCast(b.n_head)), 0);
        b.cb(q_nope, "q_nope", il);

        // {n_embd_head_qk_rope, n_head, n_tokens}
        var q_pe: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_rope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * @as(usize, @intCast(b.n_head)), c.ggml_row_size(q.type, n_embd_head_qk_nope));
        b.cb(q_pe, "q_pe_in", il);

        const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
        b.cb(kv_cmpr_pe, "kv_cmpr_pe", il);

        // {kv_lora_rank, n_tokens}
        var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), 0);
        b.cb(kv_cmpr, "kv_cmpr_in", il);

        // {n_embd_head_qk_rope, 1, n_tokens}
        var k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, n_embd_head_qk_rope, 1, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));
        b.cb(k_pe, "k_pe_in", il);

        q_pe = b.buildRope(q_pe, inp_pos, il);
        b.cb(q_pe, "q_pe", il);

        k_pe = b.buildRope(k_pe, inp_pos, il);
        b.cb(k_pe, "k_pe", il);

        kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
        b.cb(kv_cmpr, "kv_cmpr", il);

        // {n_embd_head_qk_nope, n_tokens, n_head}
        q_nope = c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
        b.cb(q_nope, "q_nope_perm", il);

        // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
        var q_nope_absorbed: *Tensor = c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
        b.cb(q_nope_absorbed, "q_nope_absorbed", il);

        // {kv_lora_rank, n_head, n_tokens}
        q_nope_absorbed = c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);
        b.cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

        // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}; RoPE goes
        // first for the in-place shift in build_rope_shift.
        const Qcur: *Tensor = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
        b.cb(Qcur, "Qcur", il);

        kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, b.n_tokens);
        b.cb(kv_cmpr, "kv_cmpr_reshape", il);

        // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
        const Kcur: *Tensor = c.ggml_concat(ctx, kv_cmpr, k_pe, 0);
        b.cb(Kcur, "Kcur", il);

        // {kv_lora_rank, 1, n_tokens}
        const Vcur = kv_cmpr;
        b.cb(Vcur, "Vcur", il);

        // MLA with the absorption optimization becomes MQA (GQA with one
        // group).
        cur = b.buildAttn(inp_attn_k, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, layer.wv_b, kq_scale, il);
        b.cb(cur, "attn_out", il);

        // `inp_out_ids` is always built here, so the C++'s null check holds.
        if (il == b.n_layer - 1 and (!b.cparams.embeddings_nextn or b.cparams.embeddings_nextn_masked)) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            residual = c.ggml_reshape_3d(ctx, c.ggml_get_rows(ctx, c.ggml_reshape_2d(ctx, residual, b.n_embd * hc, b.n_tokens), inp_out_ids), b.n_embd, hc, b.n_outputs);
            post = c.ggml_get_rows(ctx, post, inp_out_ids);
            comb = c.ggml_reshape_3d(ctx, c.ggml_get_rows(ctx, c.ggml_reshape_2d(ctx, comb, hc * hc, b.n_tokens), inp_out_ids), hc, hc, b.n_outputs);
        }

        inpL = buildHcPost(b, cur, residual, post, comb, il);
        b.cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const ffn_hc = buildHcPre(b, inpL, layer.hc_ffn_fn.?, layer.hc_ffn_scale.?, layer.hc_ffn_base.?, il);
        cur = ffn_hc.cur;
        post = ffn_hc.post;
        comb = ffn_hc.comb;
        b.cb(cur, "hc_ffn_pre", il);

        c.ggml_build_forward_expand(b.gf, residual);
        c.ggml_build_forward_expand(b.gf, post);
        c.ggml_build_forward_expand(b.gf, comb);

        cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The mixture of experts.
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
                .gate_up_exps = layer.ffn_gate_up_exps,
            }, il);
            b.cb(moe_out, "ffn_moe_out", il);

            // The shared experts.
            {
                const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
                b.cb(ffn_shexp, "ffn_shexp", il);

                cur = c.ggml_add(ctx, moe_out, ffn_shexp);
                b.cb(cur, "ffn_out", il);
            }
        }

        inpL = buildHcPost(b, cur, residual, post, comb, il);
        inpL = b.buildCvec(inpL, il);
        b.cb(inpL, "l_out", il);
    }
    cur = inpL;

    // The residual streams averaged to one for the head.
    cur = hcMean(ctx, cur);
    b.cb(cur, "hc_mean", -1);

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (b.cparams.embeddings_nextn and !b.cparams.embeddings_nextn_masked) {
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = c.ggml_mul_mat(ctx, model.output, cur);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
