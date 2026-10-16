//! Nemotron-H: a stack of single-mixer blocks, each a Mamba-2 block, an
//! attention layer with no positional encoding, or a squared-ReLU
//! feed-forward -- dense, or a mixture of experts with a shared expert and
//! an optional latent projection -- each behind an RMS norm and a residual.
//! A layer with neither KV heads nor a feed-forward is recurrent.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/nemotron-h.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_nemotron_h`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph` with
//! `build_attention_layer` and `build_ffn_layer`, which Nemotron-H MoE
//! shares.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const mamba_base = @import("mamba_base.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// `int64_t / int64_t` as aarch64 computes it: `sdiv` by zero gives zero.
/// The C++ is undefined there; this is what the reference binary does.
fn divI64(a: i64, b: i64) i64 {
    return if (b == 0) 0 else @divTrunc(a, b);
}

/// Ports `llama_model_nemotron_h`'s `load_arch_hparams`
/// (models/nemotron-h.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

    // NextN/MTP: an optional draft head appended as extra trailing blocks.
    if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
    impl.assert(hp.n_layer_nextn < hp.n_layer_all, "models/nemotron-h.cpp", 12, "hparams.n_layer_nextn < hparams.n_layer_all && \"n_layer_nextn must be < n_layer_all\"");

    // A layer is recurrent if and only if its n_head_kv and its n_ff are
    // both 0. The appended MTP blocks are dense, not recurrent.
    var i: u32 = 0;
    while (i < hp.n_layer_all) : (i += 1) {
        hp.is_recr_impl[i] = @intFromBool(i < hp.nLayer() and hp.nHeadKv(i) == 0 and hp.nFf(i) == 0);
    }

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?; // the MTP head's final_layernorm

    if (try meta.getKey(u32, .expert_feed_forward_length, false, exc)) |v| hp.n_ff_exp = v;
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    if (try meta.getKey(u32, .expert_shared_count, false, exc)) |v| hp.n_expert_shared = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(u32, .moe_latent_size, false, exc)) |v| hp.moe_latent_size = v;

    self.type = switch (hp.nLayer()) {
        52 => .@"31b_a3_5b", // Nemotron-H MoE 31B
        56 => .@"9b",
        88 => .@"120b_a12b",
        else => .unknown,
    };
}

/// Ports `llama_model_nemotron_h`'s `load_arch_tensors`
/// (models/nemotron-h.cpp:38 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    const mtp_flags: u32 = if (!ml.load_mtp) loader.tensor_skip else 0;

    // The Mamba-2 mixer's SSM parameters.
    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const n_ssm_head: i64 = hp.ssm_dt_rank;
    const n_group: i64 = hp.ssm_n_group;
    const d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head;
    const moe_n_embd: i64 = if (hp.moe_latent_size > 0) hp.moe_latent_size else l.n_embd;

    // Embeddings.
    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    {
        self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
        self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
        // Without an output, the input token embedding is used, duplicated
        // to allow offloading.
        if (self.output == null) {
            self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
        }
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const iu: u32 = @intCast(i);

        // Every block uses the attention norm.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, trunk_flags, exc);

        if (hp.isRecr(iu)) {
            // The SSM layers.
            layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = i }, &.{ l.n_embd, d_in_proj }, trunk_flags, exc);

            layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = i }, &.{ d_conv, d_inner + 2 * n_group * d_state }, trunk_flags, exc);
            layer.ssm_conv1d_b = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "bias", .bid = i }, &.{d_inner + 2 * n_group * d_state}, nr, exc);

            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{n_ssm_head}, trunk_flags, exc);

            // No "weight" suffix for these.
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = i }, &.{ 1, n_ssm_head }, trunk_flags, exc);
            layer.ssm_d = try self.createTensor(.{ .tensor = .ssm_d, .bid = i }, &.{ 1, n_ssm_head }, trunk_flags, exc);

            layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = i }, &.{ @divTrunc(d_inner, n_group), n_group }, trunk_flags, exc);

            // out_proj
            layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = i }, &.{ d_inner, l.n_embd }, trunk_flags, exc);
        } else if (hp.nFf(iu) == 0) {
            // The attention layers, with optional bias.
            const n_head_i: i64 = hp.nHead(iu);
            const n_embd_k_gqa_i: i64 = hp.nEmbdKGqa(iu);
            const n_embd_v_gqa_i: i64 = hp.nEmbdVGqa(iu);
            try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * n_head_i, n_embd_k_gqa_i, n_embd_v_gqa_i, trunk_flags, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * n_head_i, l.n_embd }, trunk_flags, exc);
            layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
        } else {
            if (l.n_expert != 0) {
                const n_ff_exp: i64 = if (hp.n_ff_exp != 0) hp.n_ff_exp else divI64(l.n_ff, l.n_expert_used);
                const n_ff_shexp: i64 = hp.n_ff_shexp;

                layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, trunk_flags, exc);
                layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, trunk_flags, exc);

                // The MoE branch.
                layer.ffn_latent_down = try self.createTensor(.{ .tensor = .ffn_latent_down, .suffix = "weight", .bid = i }, &.{ l.n_embd, moe_n_embd }, nr, exc);
                layer.ffn_latent_up = try self.createTensor(.{ .tensor = .ffn_latent_up, .suffix = "weight", .bid = i }, &.{ moe_n_embd, l.n_embd }, nr, exc);

                layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, moe_n_embd, l.n_expert }, trunk_flags, exc);
                layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ moe_n_embd, n_ff_exp, l.n_expert }, trunk_flags, exc);

                // The shared expert branch.
                layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, trunk_flags, exc);
                layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, trunk_flags, exc);
            } else {
                // The MLP layers.
                layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ hp.nFf(iu), l.n_embd }, trunk_flags, exc);
                layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.nFf(iu) }, trunk_flags, exc);
                layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
                layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{hp.nFf(iu)}, nr, exc);
            }
        }
    }

    // The NextN/MTP draft head: each predict layer folds an attention
    // sub-layer and a MoE sub-layer into one trailing block.
    i = l.n_layer;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const iu: u32 = @intCast(i);

        const n_head_i: i64 = hp.nHead(iu);
        const n_embd_k_gqa_i: i64 = hp.nEmbdKGqa(iu);
        const n_embd_v_gqa_i: i64 = hp.nEmbdVGqa(iu);
        const n_ff_exp: i64 = if (hp.n_ff_exp != 0) hp.n_ff_exp else divI64(l.n_ff, l.n_expert_used);
        const n_ff_shexp: i64 = hp.n_ff_shexp;

        // The NextN input-fusion tensors.
        layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
        layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
        layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, mtp_flags, exc);
        layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);

        // The attention sub-layer.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * n_head_i, n_embd_k_gqa_i, n_embd_v_gqa_i, mtp_flags, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * n_head_i, l.n_embd }, mtp_flags, exc);
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, mtp_flags | nr, exc);

        // The MoE sub-layer.
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, mtp_flags, exc);
        layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, mtp_flags, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, moe_n_embd, l.n_expert }, mtp_flags, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ moe_n_embd, n_ff_exp, l.n_expert }, mtp_flags, exc);
        layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, mtp_flags, exc);
        layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, mtp_flags, exc);
    }
}

/// Ports `llama_model_nemotron_h`'s `build_arch_graph`
/// (models/nemotron-h.cpp:162 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_nemotron_h`'s `graph`
/// (models/nemotron-h.cpp:166 @c1d0e7a00).
pub fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const cp = b.cparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/nemotron-h.cpp", 169, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    c.ggml_build_forward_expand(b.gf, inpL);

    const inp = b.buildInpMemHybrid();

    const inp_out_ids = b.buildInpOutIds();
    const n_layer: usize = @intCast(b.n_layer);
    const extract_final_inp = n_layer < cp.embeddings_layer_inp.len and cp.embeddings_layer_inp[n_layer];

    var cur: *Tensor = undefined;
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        b.res.t_layer_inp[@intCast(il)] = inpL;

        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // Norm.
        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        if (hp.isRecr(@intCast(il))) {
            // An SSM layer.
            cur = mamba_base.buildMamba2Layer(b, inp.getRecr(), cur, model, il);
        } else if (hp.nFf(@intCast(il)) == 0) {
            // An attention layer.
            cur = buildAttentionLayer(model, b, cur, inp.getAttn(), n_embd_head, il);
        } else {
            cur = buildFfnLayer(model, b, cur, il);
        }

        if (il == b.n_layer - 1 and cp.embeddings_nextn_masked and !extract_final_inp) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        // The residual.
        cur = c.ggml_add(ctx, cur, inpSA);
        b.cb(cur, "nemotron_h_block_out", il);

        // The next layer's input.
        inpL = cur;
    }

    cur = inpL;
    if (extract_final_inp) {
        b.res.t_layer_inp[n_layer] = cur;

        if (cp.embeddings_nextn_masked) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
        }
    }

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    // The seed of the MTP/NextN draft head.
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (!cp.embeddings_nextn_masked) {
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_nemotron_h::graph`'s `build_attention_layer`
/// (models/nemotron-h.cpp:244 @c1d0e7a00): attention without RoPE.
fn buildAttentionLayer(model: *const Model, b: *const graph.Builder, cur_: *Tensor, inp_attn: *graph.InpAttnKv, n_embd_head: i64, il: c_int) *Tensor {
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    const qkv = b.buildQkv(layer, cur_, n_embd_head, hp.nHead(@intCast(il)), hp.nHeadKv(@intCast(il)), il);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;
    const cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, qkv.q, qkv.k, qkv.v, null, null, null, kq_scale, il);
    b.cb(cur, "attn_out", il);
    return cur;
}

/// Ports `llama_model_nemotron_h::graph`'s `build_ffn_layer`
/// (models/nemotron-h.cpp:260 @c1d0e7a00): a squared-ReLU FFN, or the
/// mixture of experts beside a shared expert.
fn buildFfnLayer(model: *const Model, b: *const graph.Builder, cur_: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    var cur = cur_;

    if (layer.ffn_gate_inp == null) {
        cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, layer.ffn_up_s, null, null, null, layer.ffn_down, layer.ffn_down_b, layer.ffn_down_s, null, .relu_sqr, .par, il);
        b.cb(cur, "ffn_out", il);
    } else {
        const inp_emb = cur;
        var inp_latent = cur;

        if (layer.ffn_latent_down) |w| {
            inp_latent = c.ggml_mul_mat(ctx, w, cur);
        }

        const router_logits = b.buildLoraMm(layer.ffn_gate_inp.?, cur, null);
        b.cb(router_logits, "ffn_moe_logits", il);

        var moe_out = b.buildMoeFfn(inp_latent, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = null, // no gate
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = layer.ffn_exp_probs_b,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .relu_sqr,
            .norm_w = hp.expert_weights_norm,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .sigmoid,
            .probs_in = router_logits,
            .gate_up_exps = null,
            .up_exps_s = layer.ffn_up_exps_s,
            .gate_exps_s = null, // no gate
            .down_exps_s = layer.ffn_down_exps_s,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        if (layer.ffn_latent_up) |w| {
            moe_out = c.ggml_mul_mat(ctx, w, moe_out);
        }

        const ffn_shexp = b.buildFfn(inp_emb, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, null, null, null, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .relu_sqr, .par, il);
        b.cb(ffn_shexp, "ffn_shexp", il);

        cur = c.ggml_add(ctx, moe_out, ffn_shexp);
        b.cb(cur, "ffn_out", il);
    }

    cur = b.buildCvec(cur, il);
    b.cb(cur, "l_out", il);

    return cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
