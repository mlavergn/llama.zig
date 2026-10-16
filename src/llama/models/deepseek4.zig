//! DeepSeek-V4: multi-head attention with one shared K/V head over a raw
//! sliding window, joined per layer by compressed long-range rows -- CSA
//! layers attend to a lightning indexer's top-k of one row per 4 tokens,
//! HCA layers to every row of one per 128 -- inside hyper-connections that
//! carry `hc` residual streams instead of one; then a mixture of experts
//! with a shared expert, hash-routed in the leading layers; and an
//! optional MTP block past the main stack.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/deepseek4.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_deepseek4`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, its `graph` with the
//! hyper-connection, compressor and attention builders, and `graph_mtp`.
//! The memory these graphs read is `kv_cache_dsv4.zig`'s, and the MTP
//! head's the iSWA cache's.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const kv_cache_dsv4 = @import("../kv_cache_dsv4.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Builder = graph.Builder;
const Tensor = c.ggml_tensor;

const src = "models/deepseek4.cpp";

/// Ports `dsv4_rope_attn_factor` (models/deepseek4.cpp:11 @c1d0e7a00): the
/// YaRN magnitude correction, or none without extrapolation.
fn ropeAttnFactor(freq_scale: f32, ext_factor: f32) f32 {
    if (ext_factor == 0.0) return 1.0;
    return 1.0 / (1.0 + 0.1 * impl.logf(1.0 / freq_scale));
}

/// Ports `llama_model_deepseek4`'s `load_arch_hparams` (models/deepseek4.cpp:19 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
    if (hp.n_layer_nextn > 0 and hp.n_layer_nextn < hp.n_layer_all) {
        // A file whose MTP block was stripped declares it all the same.
        const n_layer_main = hp.n_layer_all - hp.n_layer_nextn;
        var probe_buf: [64]u8 = undefined;
        const mtp_probe = std.fmt.bufPrint(&probe_buf, "blk.{d}.nextn.eh_proj.weight", .{n_layer_main}) catch unreachable;
        if (ml.getWeight(mtp_probe) == null) hp.n_layer_nextn = 0;
    }
    impl.assert(hp.n_layer_nextn < hp.n_layer_all, src, 28, "hparams.n_layer_nextn < hparams.n_layer_all && \"n_layer_nextn must be < block_count\"");

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;

    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    hp.expert_weights_scale = (try meta.getKey(f32, .expert_weights_scale, true, exc)).?;
    hp.expert_weights_norm = (try meta.getKey(bool, .expert_weights_norm, true, exc)).?;
    _ = try meta.getKeyOrArr(f32, .swiglu_clamp_exp, &hp.swiglu_clamp_exp, hp.n_layer_all, true, exc);
    if (!try meta.getKeyOrArr(f32, .swiglu_clamp_shexp, &hp.swiglu_clamp_shexp, hp.n_layer_all, false, exc)) {
        hp.swiglu_clamp_shexp = hp.swiglu_clamp_exp;
    }

    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;

    hp.dsv4_o_group_count = (try meta.getKey(u32, .attention_output_group_count, true, exc)).?;
    hp.dsv4_o_lora_rank = (try meta.getKey(u32, .attention_output_lora_rank, true, exc)).?;
    hp.dsv4_compress_rope_base = (try meta.getKey(f32, .attention_compress_rope_freq_base, true, exc)).?;
    hp.dsv4_hc_mult = (try meta.getKey(u32, .hyper_connection_count, true, exc)).?;
    hp.dsv4_hc_sinkhorn_iters = (try meta.getKey(u32, .hyper_connection_sinkhorn_iterations, true, exc)).?;
    hp.dsv4_hc_eps = (try meta.getKey(f32, .hyper_connection_epsilon, true, exc)).?;
    hp.dsv4_hash_layer_count = (try meta.getKey(u32, .hash_layer_count, true, exc)).?;

    // The hidden state the MTP head and the embeddings see is every
    // hyper-connection stream.
    hp.n_embd_out_impl = hp.dsv4_hc_mult *% hp.n_embd;

    const n_compress_ratios = (try meta.getArrN(.attention_compress_ratios, true, exc)).?;
    if (n_compress_ratios < hp.n_layer_all) {
        return exc.throw(.runtime_error, "DeepSeek-V4 compress_ratios is shorter than block_count", .{});
    }
    impl.assert(n_compress_ratios <= hparams_mod.max_layers, src, 62, "n_compress_ratios <= LLAMA_MAX_LAYERS");
    _ = try meta.getArr(u32, .attention_compress_ratios, &hp.dsv4_compress_ratios, true, exc);

    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;
    if (hp.expert_gating_func != @backingInt(hparams_mod.ExpertGatingFunc.sqrt_softplus)) {
        return exc.throw(.runtime_error, "DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring", .{});
    }
    // Every trunk layer attends within the window, and so does the MTP
    // block.
    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    hp.setSwaPattern(0, false);
    var il: u32 = hp.nLayer();
    while (il < hp.n_layer_all) : (il += 1) hp.is_swa_impl[il] = 1;

    self.type = switch (hp.nLayer()) {
        43 => .unknown,
        else => .unknown,
    };
}

/// Ports `llama_model_deepseek4`'s `load_arch_tensors` (models/deepseek4.cpp:81 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const q_lora_rank: i64 = hp.n_lora_q;
    const n_ff_exp: i64 = hp.n_ff_exp;
    const n_expert_shared: i64 = hp.n_expert_shared;

    const n_embd_head: i64 = hp.nEmbdHeadK(0);
    const o_groups: i64 = hp.dsv4_o_group_count;
    const o_lora_rank: i64 = hp.dsv4_o_lora_rank;
    const hc_mult: i64 = hp.dsv4_hc_mult;
    const hc_dim = hc_mult * l.n_embd;
    const hc_mix_dim = (2 + hc_mult) * hc_mult;

    const mtp_only = (l.n_layer_nextn > 0) and (ml.getWeight("blk.0.attn_norm.weight") == null);
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    const mtp_flags: u32 = if (ml.load_mtp) 0 else loader.tensor_skip;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.hc_head_fn = try self.createTensor(.{ .tensor = .hc_head_fn, .suffix = "weight" }, &.{ hc_dim, hc_mult }, 0, exc);
    self.hc_head_base = try self.createTensor(.{ .tensor = .hc_head_base, .suffix = "weight" }, &.{hc_mult}, 0, exc);
    self.hc_head_scale = try self.createTensor(.{ .tensor = .hc_head_scale, .suffix = "weight" }, &.{1}, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const flags = if (i < l.n_layer) trunk_flags else mtp_flags;

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        layer.attn_sinks = try self.createTensor(.{ .tensor = .attn_sinks, .suffix = "weight", .bid = i }, &.{l.n_head}, flags, exc);
        layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_lora_rank }, flags, exc);
        layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, flags, exc);
        layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, l.n_head * n_embd_head }, flags, exc);
        layer.wkv = try self.createTensor(.{ .tensor = .attn_kv, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_head }, flags, exc);
        layer.attn_kv_norm = try self.createTensor(.{ .tensor = .attn_kv_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, flags, exc);
        // The file holds `wo_a` as (n_head * n_embd_head / o_groups,
        // o_lora_rank*o_groups); it is reshaped here rather than in the
        // graph.
        layer.wo_a = try self.createTensor(.{ .tensor = .attn_out_a, .suffix = "weight", .bid = i }, &.{ @divTrunc(l.n_head * n_embd_head, o_groups), o_lora_rank, o_groups }, flags | loader.tensor_allow_reshape, exc);
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out_b, .suffix = "weight", .bid = i }, &.{ o_groups * o_lora_rank, l.n_embd }, flags, exc);

        layer.hc_attn_fn = try self.createTensor(.{ .tensor = .hc_attn_fn, .suffix = "weight", .bid = i }, &.{ hc_dim, hc_mix_dim }, flags, exc);
        layer.hc_attn_base = try self.createTensor(.{ .tensor = .hc_attn_base, .suffix = "weight", .bid = i }, &.{hc_mix_dim}, flags, exc);
        layer.hc_attn_scale = try self.createTensor(.{ .tensor = .hc_attn_scale, .suffix = "weight", .bid = i }, &.{3}, flags, exc);
        layer.hc_ffn_fn = try self.createTensor(.{ .tensor = .hc_ffn_fn, .suffix = "weight", .bid = i }, &.{ hc_dim, hc_mix_dim }, flags, exc);
        layer.hc_ffn_base = try self.createTensor(.{ .tensor = .hc_ffn_base, .suffix = "weight", .bid = i }, &.{hc_mix_dim}, flags, exc);
        layer.hc_ffn_scale = try self.createTensor(.{ .tensor = .hc_ffn_scale, .suffix = "weight", .bid = i }, &.{3}, flags, exc);

        const ratio: i64 = hp.dsv4_compress_ratios[@intCast(i)];
        if (ratio != 0) {
            const coff: i64 = if (ratio == 4) 2 else 1;

            layer.attn_comp_wkv = try self.createTensor(.{ .tensor = .attn_compressor_wkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, coff * n_embd_head }, flags, exc);
            layer.attn_comp_wgate = try self.createTensor(.{ .tensor = .attn_compressor_wgate, .suffix = "weight", .bid = i }, &.{ l.n_embd, coff * n_embd_head }, flags, exc);
            layer.attn_comp_ape = try self.createTensor(.{ .tensor = .attn_compressor_ape, .suffix = "weight", .bid = i }, &.{ coff * n_embd_head, ratio }, flags, exc);
            layer.attn_comp_norm = try self.createTensor(.{ .tensor = .attn_compressor_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, flags, exc);

            if (ratio == 4) {
                const n_embd_indexer: i64 = hp.indexer_head_size;

                layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.indexer_n_head }, flags, exc);
                layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, hp.indexer_n_head * n_embd_indexer }, flags, exc);

                layer.indexer_comp_wkv = try self.createTensor(.{ .tensor = .indexer_compressor_wkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, 2 * n_embd_indexer }, flags, exc);
                layer.indexer_comp_wgate = try self.createTensor(.{ .tensor = .indexer_compressor_wgate, .suffix = "weight", .bid = i }, &.{ l.n_embd, 2 * n_embd_indexer }, flags, exc);
                layer.indexer_comp_ape = try self.createTensor(.{ .tensor = .indexer_compressor_ape, .suffix = "weight", .bid = i }, &.{ 2 * n_embd_indexer, ratio }, flags, exc);
                layer.indexer_comp_norm = try self.createTensor(.{ .tensor = .indexer_compressor_norm, .suffix = "weight", .bid = i }, &.{n_embd_indexer}, flags, exc);
            } else if (ratio != 128) {
                return exc.throw(.runtime_error, "DeepSeek-V4 loader only supports compression ratios 0, 4, and 128", .{});
            }
        }

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, flags, exc);
        if (@as(u32, @intCast(i)) < hp.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = try self.createTensor(.{ .tensor = .ffn_gate_tid2eid, .suffix = "weight", .bid = i }, &.{ l.n_expert_used, l.n_vocab }, flags, exc);
        } else {
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, flags, exc);
        }
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);

        layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, flags, exc);
        layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, flags, exc);
        layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, flags, exc);

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

/// Ports `llama_model_deepseek4`'s `build_arch_graph` (models/deepseek4.cpp:182
/// @c1d0e7a00): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Views and the compressor state

/// Ports `dsv4_elem_offset` (models/deepseek4.cpp:189 @c1d0e7a00).
fn elemOffset(t: *const Tensor, i: i64) usize {
    return c.ggml_row_size(t.type, i);
}

/// Ports `dsv4_view_1d` (models/deepseek4.cpp:193 @c1d0e7a00): `ne0`
/// elements from element `j0`.
fn view1d(ctx: *c.ggml_context, t: *Tensor, ne0: i64, j0: i64) *Tensor {
    return c.ggml_view_1d(ctx, t, ne0, elemOffset(t, j0));
}

/// Ports `dsv4_view_2d` (models/deepseek4.cpp:197 @c1d0e7a00): `ne0` by
/// `ne1` from element `j0` of each row.
fn view2d(ctx: *c.ggml_context, t: *Tensor, ne0: i64, ne1: i64, j0: i64) *Tensor {
    return c.ggml_view_2d(ctx, t, ne0, ne1, t.nb[1], elemOffset(t, j0));
}

/// Ports `dsv4_append_zero_row` (models/deepseek4.cpp:206 @c1d0e7a00): one
/// more row, zero or -inf, built from the first row so it stays in the
/// graph.
fn appendZeroRow(ctx: *c.ggml_context, t: *Tensor, neg_inf: bool) *Tensor {
    var row: *Tensor = c.ggml_view_1d(ctx, t, t.ne[0], 0);
    row = if (neg_inf) c.ggml_scale_bias(ctx, row, 0.0, -std.math.inf(f32)) else c.ggml_scale(ctx, row, 0.0);
    row = c.ggml_reshape_2d(ctx, row, t.ne[0], 1);

    return c.ggml_concat(ctx, t, row, 1);
}

/// Mirrors `dsv4_state_tensors` (models/deepseek4.cpp:214 @c1d0e7a00).
const StateTensors = struct {
    kv: ?*Tensor = null,
    score: ?*Tensor = null,
};

/// Ports `dsv4_build_state_restore` (models/deepseek4.cpp:219 @c1d0e7a00):
/// a compressor's state, with a pending rollback's snapshot rows copied
/// back over the current plane first.
fn buildStateRestore(ctx: *c.ggml_context, inp: *const graph.InpDsv4Comp, state: *const kv_cache_dsv4.CompState, il: c_int) StateTensors {
    var restored: StateTensors = .{
        .kv = state.getKvAll(ctx, il),
        .score = state.getScoreAll(ctx, il),
    };

    const restore_src = inp.state_restore_src_idxs orelse return restored;
    const restore_dst = inp.state_restore_dst_idxs orelse return restored;

    const kv_rows: *Tensor = c.ggml_get_rows(ctx, restored.kv, restore_src);
    restored.kv = state.cpyKv(ctx, kv_rows, restore_dst, il);

    const score_rows: *Tensor = c.ggml_get_rows(ctx, restored.score, restore_src);
    restored.score = state.cpyScore(ctx, score_rows, restore_dst, il);

    return restored;
}

/// Ports `dsv4_build_state_snapshot` (models/deepseek4.cpp:242 @c1d0e7a00):
/// the rollback planes, from the graph-local `[state | current]` rows.
fn buildStateSnapshot(ctx: *c.ggml_context, inp: *const graph.InpDsv4Comp, state: *const kv_cache_dsv4.CompState, source_kv: ?*Tensor, source_score: ?*Tensor, il: c_int) StateTensors {
    const snap_src = inp.state_snapshot_src_idxs orelse return .{};
    const snap_dst = inp.state_snapshot_dst_idxs orelse return .{};
    if (source_kv == null or source_score == null) return .{};

    const kv_rows: *Tensor = c.ggml_get_rows(ctx, source_kv, snap_src);
    const kv = state.cpyKv(ctx, kv_rows, snap_dst, il);

    const score_rows: *Tensor = c.ggml_get_rows(ctx, source_score, snap_src);
    const score = state.cpyScore(ctx, score_rows, snap_dst, il);

    return .{ .kv = kv, .score = score };
}

/// `DSV4_CSA_RATIO` (models/deepseek4.cpp:263 @c1d0e7a00).
const csa_ratio: i64 = 4;
/// `DSV4_HCA_RATIO` (models/deepseek4.cpp:264 @c1d0e7a00).
const hca_ratio: i64 = 128;

/// Ports `dsv4_hc_mean` (models/deepseek4.cpp:267 @c1d0e7a00): the mean over
/// the hyper-connection streams, `[n_embd, hc, n_tokens]` to `[n_embd,
/// n_tokens]`.
fn hcMean(ctx: *c.ggml_context, x: *Tensor) *Tensor {
    const hc = x.ne[1];

    var acc: *Tensor = c.ggml_view_2d(ctx, x, x.ne[0], x.ne[2], x.nb[2], 0);
    var s: i64 = 1;
    while (s < hc) : (s += 1) {
        acc = c.ggml_add(ctx, acc, c.ggml_view_2d(ctx, x, x.ne[0], x.ne[2], x.nb[2], @as(usize, @intCast(s)) * x.nb[1]));
    }
    return c.ggml_scale(ctx, acc, 1.0 / @as(f32, @floatFromInt(hc)));
}

/// Ports `dsv4_hc_affine` (models/deepseek4.cpp:277 @c1d0e7a00).
fn hcAffine(ctx: *c.ggml_context, x_: *Tensor, scale: *Tensor, base: *Tensor) *Tensor {
    var x: *Tensor = c.ggml_mul(ctx, x_, scale);
    x = c.ggml_add(ctx, x, base);
    return x;
}

// -----------------------------------------------------------------------------
// Hyper-connections

/// Ports the overload of `llama_model_deepseek4`'s graph's `build_hc_pre`
/// that takes the weights (models/deepseek4.cpp:287 @c1d0e7a00): the
/// streams' weighted sum, `[n_embd, hc, n_tokens]` by `[hc, n_tokens]`.
pub fn buildHcPreWeighted(b: *const Builder, x: *Tensor, weights: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    impl.assert(x.ne[0] == b.n_embd, src, 291, "x->ne[0] == n_embd");
    impl.assert(x.ne[1] == b.hparams.dsv4_hc_mult, src, 292, "x->ne[1] == hparams.dsv4_hc_mult");

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

/// Ports `llama_model_deepseek4`'s graph's `build_hc_sinkhorn`
/// (models/deepseek4.cpp:314 @c1d0e7a00). `comb` is `[dst_hc, src_hc,
/// n_tokens]`, and Sinkhorn follows the reference: a row softmax over dst,
/// one column normalization, then repeated row and column normalization.
pub fn buildHcSinkhorn(b: *const Builder, comb_: *Tensor, il: c_int) *Tensor {
    _ = il;
    const ctx = b.ctx0;

    var comb: *Tensor = c.ggml_soft_max(ctx, comb_);

    var eps: *Tensor = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 1);
    eps = c.ggml_fill(ctx, eps, b.hparams.dsv4_hc_eps);

    comb = c.ggml_add(ctx, comb, eps);

    const Norm = struct {
        fn cols(ctx_: *c.ggml_context, comb_p: **Tensor, eps_: *Tensor) void {
            const comb_src_dst: *Tensor = c.ggml_cont(ctx_, c.ggml_permute(ctx_, comb_p.*, 1, 0, 2, 3));
            var col_sum: *Tensor = c.ggml_sum_rows(ctx_, comb_src_dst);
            col_sum = c.ggml_add(ctx_, col_sum, eps_);
            col_sum = c.ggml_permute(ctx_, col_sum, 1, 0, 2, 3);
            comb_p.* = c.ggml_div(ctx_, comb_p.*, col_sum);
        }

        fn rows(ctx_: *c.ggml_context, comb_p: **Tensor, eps_: *Tensor) void {
            var row_sum: *Tensor = c.ggml_sum_rows(ctx_, comb_p.*);
            row_sum = c.ggml_add(ctx_, row_sum, eps_);
            comb_p.* = c.ggml_div(ctx_, comb_p.*, row_sum);
        }
    };

    Norm.cols(ctx, &comb, eps);
    var i: u32 = 1;
    while (i < b.hparams.dsv4_hc_sinkhorn_iters) : (i += 1) {
        Norm.rows(ctx, &comb, eps);
        Norm.cols(ctx, &comb, eps);
    }

    return comb;
}

/// Ports `llama_model_deepseek4`'s graph's `build_hc_pre`
/// (models/deepseek4.cpp:351 @c1d0e7a00): a hyper-connection's mixes --
/// the pre weights it reduces the streams with, here; the post weights and
/// the stream combination, into `post` and `comb`, for `build_hc_post`.
pub fn buildHcPre(b: *const Builder, x: *Tensor, hc_fn: *Tensor, hc_scale: *Tensor, hc_base: *Tensor, post: *?*Tensor, comb: *?*Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const hc: i64 = hp.dsv4_hc_mult;
    const hc_dim = hc * b.n_embd;
    const hc_mix_dim = (2 + hc) * hc;
    const nt = x.ne[2];

    impl.assert(hc == 4, src, 364, "hc == 4");
    impl.assert(hc_fn.ne[1] == hc_mix_dim, src, 365, "hc_fn->ne[1] == hc_mix_dim");

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
    pre = c.ggml_sigmoid(ctx, pre);
    pre = c.ggml_scale_bias(ctx, pre, 1.0, hp.dsv4_hc_eps);
    b.cb(pre, "hc_pre", il);

    var p = view2d(ctx, mixes, hc, nt, hc);
    p = hcAffine(ctx, p, scale_post, base_post);
    p = c.ggml_sigmoid(ctx, p);
    p = c.ggml_scale(ctx, p, 2.0);
    post.* = p;
    b.cb(p, "hc_post", il);

    var cm: *Tensor = undefined;
    if (b.cparams.fused_dsv4_hc_comb) {
        cm = c.ggml_dsv4_hc_comb(ctx, mixes, hc_scale, hc_base, hp.dsv4_hc_eps, @intCast(hp.dsv4_hc_sinkhorn_iters));
        b.res.addFusedNode(.{ .op = .dsv4_hc_comb, .tensor = cm, .il = il });
    } else {
        const scale_comb = view1d(ctx, hc_scale, 1, 2);
        const base_comb = view1d(ctx, hc_base, hc * hc, 2 * hc);

        cm = view2d(ctx, mixes, hc * hc, nt, 2 * hc);
        cm = hcAffine(ctx, cm, scale_comb, base_comb);
        cm = c.ggml_reshape_3d(ctx, cm, hc, hc, nt);
        cm = buildHcSinkhorn(b, cm, il);
    }
    comb.* = cm;
    b.cb(cm, "hc_comb", il);

    const result = buildHcPreWeighted(b, x, pre, il);
    return result;
}

/// Ports `llama_model_deepseek4`'s graph's `build_hc_post`
/// (models/deepseek4.cpp:409 @c1d0e7a00): the block's output spread over
/// the streams by `post`, plus the residual streams mixed by `comb`.
pub fn buildHcPost(b: *const Builder, x: *Tensor, residual: *Tensor, post: *Tensor, comb: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    impl.assert(x.ne[0] == b.n_embd, src, 415, "x->ne[0] == n_embd");
    impl.assert(residual.ne[1] == b.hparams.dsv4_hc_mult, src, 416, "residual->ne[1] == hparams.dsv4_hc_mult");

    if (b.cparams.fused_dsv4_hc_post) {
        const result: *Tensor = c.ggml_dsv4_hc_post(ctx, x, residual, post, comb);
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
            const comb_src_dst: *Tensor = c.ggml_view_2d(ctx, comb, 1, nt, comb.nb[2], udst * comb.nb[0] + us * comb.nb[1]);
            cur = c.ggml_add(ctx, cur, c.ggml_mul(ctx, res_src, comb_src_dst));
        }

        cur = c.ggml_reshape_3d(ctx, cur, b.n_embd, 1, nt);
        out = if (out) |o| c.ggml_concat(ctx, o, cur, 1) else cur;
    }

    return out.?;
}

/// Ports `llama_model_deepseek4`'s graph's `build_hc_head`
/// (models/deepseek4.cpp:446 @c1d0e7a00): the streams reduced to one for
/// the output head.
pub fn buildHcHead(b: *const Builder, x: *Tensor, hc_fn: *Tensor, hc_scale: *Tensor, hc_base: *Tensor) *Tensor {
    const ctx = b.ctx0;
    const hc: i64 = b.hparams.dsv4_hc_mult;
    const hc_dim = hc * b.n_embd;
    const nt = x.ne[2];

    const flat: *Tensor = c.ggml_reshape_2d(ctx, x, hc_dim, nt);
    const flat_norm: *Tensor = c.ggml_rms_norm(ctx, flat, b.norm_rms_eps);
    const mixes: *Tensor = c.ggml_mul_mat(ctx, hc_fn, flat_norm);
    b.cb(mixes, "hc_head_mixes", -1);

    var pre = hcAffine(ctx, mixes, hc_scale, hc_base);
    pre = c.ggml_sigmoid(ctx, pre);
    pre = c.ggml_scale_bias(ctx, pre, 1.0, b.hparams.dsv4_hc_eps);
    b.cb(pre, "hc_head_pre", -1);

    return buildHcPreWeighted(b, x, pre, -1);
}

// -----------------------------------------------------------------------------
// Compressors

/// Ports `llama_model_deepseek4`'s graph's `build_hca_compressed_kv_from_state`
/// (models/deepseek4.cpp:468 @c1d0e7a00): each completed block's 128 state
/// rows, softmax-weighted by their scores into one row, normalized and
/// rotated at the block's first position.
pub fn buildHcaCompressedKvFromState(b: *const Builder, kv_state: *Tensor, score_state: *Tensor, state_read_idxs: ?*Tensor, comp_pos: ?*Tensor, norm: ?*Tensor, n_embd_head: i64, name: [*:0]const u8, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head_rope: i64 = hp.nRot(0);
    const n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const n_blocks: i64 = if (comp_pos) |p| p.ne[0] else 0;

    impl.assert(n_blocks > 0, src, 481, "n_blocks > 0");
    impl.assert(state_read_idxs != null, src, 482, "state_read_idxs");
    impl.assert(state_read_idxs.?.ne[0] == hca_ratio * n_blocks, src, 483, "state_read_idxs->ne[0] == DSV4_HCA_RATIO*n_blocks");
    impl.assert(n_embd_head >= n_embd_head_rope, src, 484, "n_embd_head >= n_embd_head_rope");

    var kv: *Tensor = c.ggml_get_rows(ctx, kv_state, state_read_idxs);
    kv = c.ggml_reshape_3d(ctx, kv, n_embd_head, hca_ratio, n_blocks);
    b.cb(kv, name, il);

    var score: *Tensor = c.ggml_get_rows(ctx, score_state, state_read_idxs);
    score = c.ggml_reshape_3d(ctx, score, n_embd_head, hca_ratio, n_blocks);
    b.cb(score, name, il);

    const values: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, kv, 1, 0, 2, 3));
    const scores: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, score, 1, 0, 2, 3));

    const weights: *Tensor = c.ggml_soft_max(ctx, scores);
    var comp: *Tensor = c.ggml_mul(ctx, values, weights);
    comp = c.ggml_sum_rows(ctx, comp);
    comp = c.ggml_cont(ctx, c.ggml_permute(ctx, comp, 1, 0, 2, 3));
    b.cb(comp, name, il);

    comp = b.buildNorm(comp, norm, null, .rms, il);
    b.cb(comp, name, il);

    comp = c.ggml_rope_ext(ctx, comp, comp_pos, null, @intCast(n_embd_head_rope), b.rope_type, b.n_ctx_orig, hp.dsv4_compress_rope_base, b.freq_scale, b.ext_factor, ropeAttnFactor(b.freq_scale, b.ext_factor), b.beta_fast, b.beta_slow);
    comp = c.ggml_rope_set_offset(comp, @intCast(n_embd_head_nope));
    b.cb(comp, name, il);

    return comp;
}

/// Ports `llama_model_deepseek4`'s graph's `build_overlap_compressed_kv_from_state`
/// (models/deepseek4.cpp:515 @c1d0e7a00): CSA's and the indexer's
/// overlapping compression -- each block weighs its previous window's
/// first half of the state with its own window's second half.
pub fn buildOverlapCompressedKvFromState(b: *const Builder, kv_state_: *Tensor, score_state_: *Tensor, state_read_idxs: ?*Tensor, comp_pos: ?*Tensor, norm: ?*Tensor, ratio: i64, n_embd_head: i64, name: [*:0]const u8, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head_rope: i64 = hp.nRot(0);
    const n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const n_blocks: i64 = if (comp_pos) |p| p.ne[0] else 0;

    impl.assert(n_blocks > 0, src, 529, "n_blocks > 0");
    impl.assert(state_read_idxs != null, src, 530, "state_read_idxs");
    impl.assert(state_read_idxs.?.ne[0] == 2 * ratio * n_blocks, src, 531, "state_read_idxs->ne[0] == 2*ratio*n_blocks");
    impl.assert(kv_state_.ne[0] == 2 * n_embd_head, src, 532, "kv_state->ne[0] == 2*n_embd_head");
    impl.assert(score_state_.ne[0] == 2 * n_embd_head, src, 533, "score_state->ne[0] == 2*n_embd_head");
    impl.assert(n_embd_head >= n_embd_head_rope, src, 534, "n_embd_head >= n_embd_head_rope");

    const kv_state = appendZeroRow(ctx, kv_state_, false);
    const score_state = appendZeroRow(ctx, score_state_, true);

    const n_read = ratio * n_blocks;
    const un_read: usize = @intCast(n_read);

    const kv_rows: *Tensor = c.ggml_get_rows(ctx, kv_state, state_read_idxs);
    const score_rows: *Tensor = c.ggml_get_rows(ctx, score_state, state_read_idxs);

    var kv_prev: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, kv_rows, n_embd_head, n_read, kv_rows.nb[1], 0));
    kv_prev = c.ggml_reshape_3d(ctx, kv_prev, n_embd_head, ratio, n_blocks);
    b.cb(kv_prev, name, il);

    var score_prev: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, score_rows, n_embd_head, n_read, score_rows.nb[1], 0));
    score_prev = c.ggml_reshape_3d(ctx, score_prev, n_embd_head, ratio, n_blocks);
    b.cb(score_prev, name, il);

    var kv_cur: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, kv_rows, n_embd_head, n_read, kv_rows.nb[1], un_read * kv_rows.nb[1] + c.ggml_row_size(kv_rows.type, n_embd_head)));
    kv_cur = c.ggml_reshape_3d(ctx, kv_cur, n_embd_head, ratio, n_blocks);

    var score_cur: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, score_rows, n_embd_head, n_read, score_rows.nb[1], un_read * score_rows.nb[1] + c.ggml_row_size(score_rows.type, n_embd_head)));
    score_cur = c.ggml_reshape_3d(ctx, score_cur, n_embd_head, ratio, n_blocks);

    var values: *Tensor = c.ggml_concat(ctx, kv_prev, kv_cur, 1);
    var scores: *Tensor = c.ggml_concat(ctx, score_prev, score_cur, 1);

    values = c.ggml_cont(ctx, c.ggml_permute(ctx, values, 1, 0, 2, 3));
    scores = c.ggml_cont(ctx, c.ggml_permute(ctx, scores, 1, 0, 2, 3));

    const weights: *Tensor = c.ggml_soft_max(ctx, scores);
    var comp: *Tensor = c.ggml_mul(ctx, values, weights);
    comp = c.ggml_sum_rows(ctx, comp);
    comp = c.ggml_cont(ctx, c.ggml_permute(ctx, comp, 1, 0, 2, 3));
    b.cb(comp, name, il);

    comp = b.buildNorm(comp, norm, null, .rms, il);
    b.cb(comp, name, il);

    comp = c.ggml_rope_ext(ctx, comp, comp_pos, null, @intCast(n_embd_head_rope), b.rope_type, b.n_ctx_orig, hp.dsv4_compress_rope_base, b.freq_scale, b.ext_factor, ropeAttnFactor(b.freq_scale, b.ext_factor), b.beta_fast, b.beta_slow);
    comp = c.ggml_rope_set_offset(comp, @intCast(n_embd_head_nope));
    b.cb(comp, name, il);

    return comp;
}

// -----------------------------------------------------------------------------
// Attention

/// Ports `llama_model_deepseek4`'s graph's `build_lid_top_k`
/// (models/deepseek4.cpp:588 @c1d0e7a00): the lightning indexer's score of
/// every compressed row for each query, and the top-k rows.
pub fn buildLidTopK(b: *const Builder, model: *const Model, inp_dsv4: *const graph.InpDsv4, qr: *Tensor, cur: *Tensor, inp_pos: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    const inp_lid = inp_dsv4.getLid();
    const n_embd_indexer_head: i64 = hp.indexer_head_size;
    const n_embd_indexer_head_rope: i64 = hp.nRot(0);
    const n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const n_indexer_head: i64 = hp.indexer_n_head;
    const nt = cur.ne[1];

    impl.assert(inp_lid.kq_mask != null, src, 603, "inp_lid.kq_mask");
    impl.assert(inp_lid.k_rot != null, src, 604, "inp_lid.k_rot");
    impl.assert(n_embd_indexer_head >= n_embd_indexer_head_rope, src, 605, "n_embd_indexer_head >= n_embd_indexer_head_rope");

    var indexer_q = b.buildLoraMm(layer.indexer_attn_q_b.?, qr, null);
    indexer_q = c.ggml_reshape_3d(ctx, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    b.cb(indexer_q, "lid_q", il);

    indexer_q = c.ggml_rope_ext(ctx, indexer_q, inp_pos, null, @intCast(n_embd_indexer_head_rope), b.rope_type, b.n_ctx_orig, hp.dsv4_compress_rope_base, b.freq_scale, b.ext_factor, ropeAttnFactor(b.freq_scale, b.ext_factor), b.beta_fast, b.beta_slow);
    indexer_q = c.ggml_rope_set_offset(indexer_q, @intCast(n_embd_indexer_head_nope));
    b.cb(indexer_q, "lid_q_rope", il);

    indexer_q = graph.mulMatHadamard(ctx, indexer_q, inp_lid.k_rot.?);
    b.cb(indexer_q, "lid_q_rot", il);

    var indexer_weights = b.buildLoraMm(layer.indexer_proj.?, cur, null);
    indexer_weights = c.ggml_scale(ctx, indexer_weights, 1.0 / impl.sqrtf(@floatFromInt(n_embd_indexer_head * n_indexer_head)));
    b.cb(indexer_weights, "lid_weights", il);

    var indexer_k = inp_dsv4.mctx.getLid().getK(ctx, il);
    const kq_mask = inp_lid.kq_mask.?;
    const n_lid = kq_mask.ne[0];
    impl.assert(n_lid > 0, src, 626, "n_lid > 0");
    impl.assert(n_lid <= indexer_k.ne[2], src, 627, "n_lid <= indexer_k->ne[2]");

    indexer_k = c.ggml_view_4d(ctx, indexer_k, indexer_k.ne[0], indexer_k.ne[1], n_lid, indexer_k.ne[3], indexer_k.nb[1], indexer_k.nb[2], indexer_k.nb[3], 0);
    b.cb(indexer_k, "lid_k", il);

    const n_stream = indexer_k.ne[3];
    const un_stream: usize = @intCast(n_stream);
    indexer_q = c.ggml_view_4d(ctx, indexer_q, indexer_q.ne[0], indexer_q.ne[1], @divTrunc(indexer_q.ne[2], n_stream), n_stream, indexer_q.nb[1], indexer_q.nb[2], indexer_q.nb[3] / un_stream, 0);
    indexer_weights = c.ggml_view_4d(ctx, indexer_weights, indexer_weights.ne[0], @divTrunc(indexer_weights.ne[1], n_stream), indexer_weights.ne[2], n_stream, indexer_weights.nb[1], indexer_weights.nb[2] / un_stream, indexer_weights.nb[3] / un_stream, 0);

    var indexer_score: *Tensor = undefined;
    if (b.cparams.fused_lid) {
        indexer_score = c.ggml_lightning_indexer(ctx, indexer_q, indexer_k, indexer_weights, kq_mask);
        b.cb(indexer_score, "lid_score_masked", il);
        b.res.addFusedNode(.{ .op = .lightning_indexer, .tensor = indexer_score, .il = il });
    } else {
        indexer_q = c.ggml_permute(ctx, indexer_q, 0, 2, 1, 3);
        b.cb(indexer_q, "lid_q", il);
        indexer_k = c.ggml_permute(ctx, indexer_k, 0, 2, 1, 3);
        b.cb(indexer_k, "lid_k", il);

        var indexer_kq: *Tensor = c.ggml_mul_mat(ctx, indexer_k, indexer_q);
        b.cb(indexer_kq, "lid_kq", il);

        indexer_kq = c.ggml_cont(ctx, c.ggml_permute(ctx, indexer_kq, 2, 1, 0, 3));
        b.cb(indexer_kq, "lid_kq", il);

        indexer_score = c.ggml_relu(ctx, indexer_kq);
        indexer_score = c.ggml_mul(ctx, indexer_score, indexer_weights);
        indexer_score = c.ggml_sum_rows(ctx, indexer_score);
        indexer_score = c.ggml_cont(ctx, c.ggml_permute(ctx, indexer_score, 2, 1, 0, 3));
        b.cb(indexer_score, "lid_score", il);

        indexer_score = c.ggml_add(ctx, indexer_score, kq_mask);
        b.cb(indexer_score, "lid_score_masked", il);
    }

    const n_top_k: u32 = if (indexer_score.ne[0] < hp.indexer_top_k) @intCast(indexer_score.ne[0]) else hp.indexer_top_k;
    const top_k: *Tensor = c.ggml_cont(ctx, c.ggml_top_k(ctx, indexer_score, @intCast(n_top_k)));
    b.cb(top_k, "lid_top_k", il);

    return top_k;
}

/// Ports `llama_model_deepseek4`'s graph's `build_top_k_mask`
/// (models/deepseek4.cpp:676 @c1d0e7a00): the compressed mask with
/// everything outside each query's top-k rows masked too.
pub fn buildTopKMask(b: *const Builder, kq_mask: *Tensor, top_k: *Tensor, name: [*:0]const u8, il: c_int) *Tensor {
    const ctx = b.ctx0;

    var kq_mask_all: *Tensor = c.ggml_fill(ctx, kq_mask, -std.math.inf(f32));
    kq_mask_all = c.ggml_view_4d(ctx, kq_mask_all, 1, kq_mask_all.ne[0], kq_mask_all.ne[1], kq_mask_all.ne[3], kq_mask_all.nb[0], kq_mask_all.nb[1], kq_mask_all.nb[2], 0);

    const top_k_3d: *Tensor = c.ggml_view_4d(ctx, top_k, top_k.ne[0], top_k.ne[1], top_k.ne[3], 1, top_k.nb[1], top_k.nb[2], @as(usize, @intCast(top_k.ne[3])) * top_k.nb[3], 0);

    var zeros: *Tensor = c.ggml_new_tensor_4d(ctx, if (b.cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32, 1, top_k_3d.ne[0], top_k_3d.ne[1], top_k_3d.ne[2]);
    zeros = c.ggml_fill(ctx, zeros, 0.0);

    var kq_mask_top_k: *Tensor = c.ggml_set_rows(ctx, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = c.ggml_view_4d(ctx, kq_mask_top_k, kq_mask_top_k.ne[1], kq_mask_top_k.ne[2], 1, kq_mask_top_k.ne[3], kq_mask_top_k.nb[2], kq_mask_top_k.nb[3], kq_mask_top_k.nb[3], 0);

    kq_mask_top_k = c.ggml_add(ctx, kq_mask_top_k, kq_mask);
    b.cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

/// Ports `llama_model_deepseek4`'s graph's `build_csa_lid_attention`
/// (models/deepseek4.cpp:705 @c1d0e7a00): attention over the raw window
/// and the indexer's top-k CSA rows, as one K.
pub fn buildCsaLidAttention(b: *const Builder, model: *const Model, inp_dsv4: *const graph.InpDsv4, inp_attn: *const graph.InpDsv4Raw, q_: *Tensor, kv_: *Tensor, qr: *Tensor, cur: *Tensor, inp_pos: *Tensor, sinks: ?*Tensor, kq_scale: f32, il: c_int) *Tensor {
    const ctx = b.ctx0;
    var q = q_;
    var kv = kv_;
    const inp_csa = inp_dsv4.getCsa();
    impl.assert(inp_csa.kq_mask != null, src, 718, "inp_csa.kq_mask");

    const top_k = buildLidTopK(b, model, inp_dsv4, qr, cur, inp_pos, il);

    const k_rot = inp_attn.self_k_rot;
    if (k_rot) |r| {
        q = graph.mulMatHadamard(ctx, q, r);
        kv = graph.mulMatHadamard(ctx, kv, r);
    }

    c.ggml_build_forward_expand(b.gf, q);
    c.ggml_build_forward_expand(b.gf, kv);

    const mctx_raw = inp_attn.mctx;

    c.ggml_build_forward_expand(b.gf, mctx_raw.cpyK(ctx, kv, inp_attn.getKIdxs(), il));

    const raw_k = mctx_raw.getK(ctx, il);
    b.cb(raw_k, "csa_raw_k", il);

    var csa_k = inp_dsv4.mctx.getCsa().getK(ctx, il);
    const n_csa = inp_csa.kq_mask.?.ne[0];
    impl.assert(n_csa > 0, src, 740, "n_csa > 0");
    impl.assert(n_csa <= csa_k.ne[2], src, 741, "n_csa <= csa_k->ne[2]");

    csa_k = c.ggml_view_4d(ctx, csa_k, csa_k.ne[0], csa_k.ne[1], n_csa, csa_k.ne[3], csa_k.nb[1], csa_k.nb[2], csa_k.nb[3], 0);
    b.cb(csa_k, "csa_comp_k", il);

    const k_all: *Tensor = c.ggml_concat(ctx, raw_k, csa_k, 2);
    b.cb(k_all, "csa_k_all", il);

    const raw_mask = inp_attn.getKqMask();
    const csa_mask = buildTopKMask(b, inp_csa.kq_mask.?, top_k, "csa_top_k_mask", il);

    const kq_mask: *Tensor = c.ggml_concat(ctx, raw_mask, csa_mask, 0);
    b.cb(kq_mask, "csa_lid_kq_mask", il);

    var out = b.buildAttnMha(q, k_all, k_all, null, kq_mask, sinks, null, kq_scale, il);
    if (k_rot) |r| out = graph.mulMatHadamard(ctx, out, r);
    b.cb(out, "attn_csa_lid", il);

    return out;
}

/// Ports `llama_model_deepseek4`'s graph's `build_hca_attention`
/// (models/deepseek4.cpp:766 @c1d0e7a00): attention over the raw window
/// and every visible HCA row, as one K.
pub fn buildHcaAttention(b: *const Builder, inp_dsv4: *const graph.InpDsv4, inp_attn: *const graph.InpDsv4Raw, q_: *Tensor, kv_: *Tensor, sinks: ?*Tensor, kq_scale: f32, il: c_int) *Tensor {
    const ctx = b.ctx0;
    var q = q_;
    var kv = kv_;
    const inp_hca = inp_dsv4.getHca();
    impl.assert(inp_hca.kq_mask != null, src, 775, "inp_hca.kq_mask");

    const k_rot = inp_attn.self_k_rot;
    if (k_rot) |r| {
        q = graph.mulMatHadamard(ctx, q, r);
        kv = graph.mulMatHadamard(ctx, kv, r);
    }

    c.ggml_build_forward_expand(b.gf, q);
    c.ggml_build_forward_expand(b.gf, kv);

    const mctx_raw = inp_attn.mctx;

    c.ggml_build_forward_expand(b.gf, mctx_raw.cpyK(ctx, kv, inp_attn.getKIdxs(), il));

    const raw_k = mctx_raw.getK(ctx, il);
    b.cb(raw_k, "hca_raw_k", il);

    var hca_k = inp_dsv4.mctx.getHca().getK(ctx, il);
    const n_hca = inp_hca.kq_mask.?.ne[0];
    impl.assert(n_hca > 0, src, 795, "n_hca > 0");
    impl.assert(n_hca <= hca_k.ne[2], src, 796, "n_hca <= hca_k->ne[2]");

    hca_k = c.ggml_view_4d(ctx, hca_k, hca_k.ne[0], hca_k.ne[1], n_hca, hca_k.ne[3], hca_k.nb[1], hca_k.nb[2], hca_k.nb[3], 0);
    b.cb(hca_k, "hca_comp_k", il);

    const k_all: *Tensor = c.ggml_concat(ctx, raw_k, hca_k, 2);
    b.cb(k_all, "hca_k_all", il);

    const raw_mask = inp_attn.getKqMask();
    const hca_mask = inp_hca.kq_mask;

    const kq_mask: *Tensor = c.ggml_concat(ctx, raw_mask, hca_mask, 0);
    b.cb(kq_mask, "hca_kq_mask", il);

    var out = b.buildAttnMha(q, k_all, k_all, null, kq_mask, sinks, null, kq_scale, il);
    if (k_rot) |r| out = graph.mulMatHadamard(ctx, out, r);
    b.cb(out, "attn_hca", il);

    return out;
}

/// Ports `llama_model_deepseek4`'s graph's `build_raw_attention`
/// (models/deepseek4.cpp:821 @c1d0e7a00): attention over the raw window
/// alone.
pub fn buildRawAttention(b: *const Builder, inp_attn: *const graph.InpDsv4Raw, q_: *Tensor, kv_: *Tensor, sinks: ?*Tensor, kq_scale: f32, il: c_int) *Tensor {
    const ctx = b.ctx0;
    var q = q_;
    var kv = kv_;
    impl.assert(b.hparams.isSwa(@intCast(il)), src, 828, "hparams.is_swa(il)");

    const k_rot = inp_attn.self_k_rot;

    if (k_rot) |r| {
        q = graph.mulMatHadamard(ctx, q, r);
        kv = graph.mulMatHadamard(ctx, kv, r);
    }

    c.ggml_build_forward_expand(b.gf, q);
    c.ggml_build_forward_expand(b.gf, kv);

    const mctx_cur = inp_attn.mctx;

    c.ggml_build_forward_expand(b.gf, mctx_cur.cpyK(ctx, kv, inp_attn.getKIdxs(), il));

    const kq_mask = inp_attn.getKqMask();

    const k = mctx_cur.getK(ctx, il);

    var out = b.buildAttnMha(q, k, k, null, kq_mask, sinks, null, kq_scale, il);
    if (k_rot) |r| out = graph.mulMatHadamard(ctx, out, r);
    b.cb(out, "attn_raw", il);

    return out;
}

/// The two inputs `build_attention` is overloaded on: the DSV4 memory's,
/// or the MTP head's K-only iSWA cache's.
pub const AttnInput = union(enum) {
    dsv4: *const graph.InpDsv4,
    mtp: *graph.InpAttnKIswa,
};

/// Ports `llama_model_deepseek4`'s graph's `build_attention`, both overloads
/// (models/deepseek4.cpp:857, 866 @c1d0e7a00), and the
/// `build_attention_impl` they forward to (models/deepseek4.cpp:875
/// @c1d0e7a00): one shared K/V head, RoPE'd on its tail, with the layer's
/// compressor states updated and its compressed rows committed before
/// attention; then the output de-rotated and projected through the grouped
/// low-rank `wo_a` and `wo_b`.
pub fn buildAttention(b: *const Builder, model: *const Model, inp: AttnInput, cur: *Tensor, inp_pos: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const inp_dsv4: ?*const graph.InpDsv4 = switch (inp) {
        .dsv4 => |p| p,
        .mtp => null,
    };
    const inp_mtp: ?*graph.InpAttnKIswa = switch (inp) {
        .dsv4 => null,
        .mtp => |p| p,
    };
    impl.assert((inp_dsv4 == null) != (inp_mtp == null), src, 882, "(inp_dsv4 == nullptr) != (inp_mtp == nullptr)");

    const layer = &model.layers[@intCast(il)];
    const inp_attn: ?*const graph.InpDsv4Raw = if (inp_dsv4) |d| d.getRaw() else null;

    const n_embd_head: i64 = hp.nEmbdHeadK(0);
    const n_embd_head_rope: i64 = hp.nRot(0);
    const n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const n_groups: i64 = hp.dsv4_o_group_count;
    const n_heads_group = @divTrunc(b.n_head, n_groups);
    const o_lora_rank: i64 = hp.dsv4_o_lora_rank;
    const o_group_dim = n_heads_group * n_embd_head;
    const nt = cur.ne[1];

    impl.assert(n_embd_head == b.n_embd_head_v, src, 896, "n_embd_head == n_embd_head_v");
    impl.assert(@rem(b.n_head, n_groups) == 0, src, 897, "n_head % n_groups == 0");

    // A compressed layer rotates as the compressor's rows do.
    const use_compress_rope = hp.dsv4_compress_ratios[@intCast(il)] != 0;
    const freq_base_l = if (use_compress_rope) hp.dsv4_compress_rope_base else b.freq_base;
    const freq_scale_l = if (use_compress_rope) b.freq_scale else 1.0;
    const ext_factor_l = if (use_compress_rope) b.ext_factor else 0.0;
    const attn_factor_l = ropeAttnFactor(freq_scale_l, ext_factor_l);
    const beta_fast_l = if (use_compress_rope) b.beta_fast else 0.0;
    const beta_slow_l = if (use_compress_rope) b.beta_slow else 0.0;
    const n_ctx_orig_l: i32 = if (use_compress_rope) b.n_ctx_orig else 0;

    var qr = b.buildLoraMm(layer.wq_a.?, cur, null);
    b.cb(qr, "qr", il);

    qr = b.buildNorm(qr, layer.attn_q_a_norm, null, .rms, il);
    b.cb(qr, "qr_norm", il);

    var q = b.buildLoraMm(layer.wq_b.?, qr, null);
    q = c.ggml_reshape_3d(ctx, q, n_embd_head, b.n_head, nt);
    q = c.ggml_rms_norm(ctx, q, b.norm_rms_eps);
    b.cb(q, "q_norm", il);

    q = c.ggml_rope_ext(ctx, q, inp_pos, null, @intCast(n_embd_head_rope), b.rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    q = c.ggml_rope_set_offset(q, @intCast(n_embd_head_nope));
    b.cb(q, "q", il);

    var kv = b.buildLoraMm(layer.wkv.?, cur, null);
    kv = b.buildNorm(kv, layer.attn_kv_norm, null, .rms, il);
    kv = c.ggml_reshape_3d(ctx, kv, n_embd_head, 1, nt);
    b.cb(kv, "kv_norm", il);

    kv = c.ggml_rope_ext(ctx, kv, inp_pos, null, @intCast(n_embd_head_rope), b.rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    kv = c.ggml_rope_set_offset(kv, @intCast(n_embd_head_nope));
    b.cb(kv, "kv", il);

    const ratio: i64 = hp.dsv4_compress_ratios[@intCast(il)];
    impl.assert(inp_dsv4 != null or ratio == 0, src, 935, "inp_dsv4 || ratio == 0");

    var hca_state_kv: ?*Tensor = null;
    var hca_state_score: ?*Tensor = null;
    var hca_source_kv: ?*Tensor = null;
    var hca_source_score: ?*Tensor = null;
    if (ratio == hca_ratio and inp_dsv4.?.getHca().state_pos != null) {
        hca_state_kv = b.buildLoraMm(layer.attn_comp_wkv.?, cur, null);
        b.cb(hca_state_kv.?, "hca_state_kv", il);

        var score = b.buildLoraMm(layer.attn_comp_wgate.?, cur, null);
        b.cb(score, "hca_state_score", il);

        const ape = layer.attn_comp_ape;

        const ape_rows: *Tensor = c.ggml_get_rows(ctx, ape, inp_dsv4.?.getHca().state_pos);
        score = c.ggml_add(ctx, score, ape_rows);
        hca_state_score = score;
        b.cb(score, "hca_state_score_ape", il);
    }

    if (ratio == csa_ratio and inp_dsv4.?.getCsa().state_pos != null) {
        const d = inp_dsv4.?;
        const inp_csa = d.getCsa();
        const inp_lid = d.getLid();

        var csa_state_kv = b.buildLoraMm(layer.attn_comp_wkv.?, cur, null);
        b.cb(csa_state_kv, "csa_state_kv", il);

        var csa_state_score = b.buildLoraMm(layer.attn_comp_wgate.?, cur, null);
        b.cb(csa_state_score, "csa_state_score", il);

        const csa_ape = layer.attn_comp_ape;

        const csa_ape_rows: *Tensor = c.ggml_get_rows(ctx, csa_ape, inp_csa.state_pos);
        csa_state_score = c.ggml_add(ctx, csa_state_score, csa_ape_rows);
        b.cb(csa_state_score, "csa_state_score_ape", il);

        impl.assert(inp_csa.state_write_idxs != null, src, 969, "inp_dsv4->get_csa().state_write_idxs");

        const csa_state = d.mctx.getCsaState();
        const csa_restored = buildStateRestore(ctx, inp_csa, csa_state, il);
        const csa_base_kv = view2d(ctx, csa_restored.kv.?, csa_restored.kv.?.ne[0], csa_state.getNRows(), 0);
        const csa_base_score = view2d(ctx, csa_restored.score.?, csa_restored.score.?.ne[0], csa_state.getNRows(), 0);

        const csa_source_kv: *Tensor = c.ggml_concat(ctx, csa_base_kv, csa_state_kv, 1);
        const csa_source_score: *Tensor = c.ggml_concat(ctx, csa_base_score, csa_state_score, 1);

        var kv_comp_csa_state = buildOverlapCompressedKvFromState(b, csa_source_kv, csa_source_score, inp_csa.state_read_idxs, inp_csa.state_write_pos, layer.attn_comp_norm, csa_ratio, n_embd_head, "csa_state_compress", il);

        if (inp_csa.k_rot) |r| {
            kv_comp_csa_state = graph.mulMatHadamard(ctx, kv_comp_csa_state, r);
            b.cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        c.ggml_build_forward_expand(b.gf, d.mctx.getCsa().cpyK(ctx, kv_comp_csa_state, inp_csa.state_write_idxs.?, il));

        const csa_snapshot_source_kv: *Tensor = c.ggml_concat(ctx, csa_restored.kv, csa_state_kv, 1);
        const csa_snapshot_source_score: *Tensor = c.ggml_concat(ctx, csa_restored.score, csa_state_score, 1);

        const csa_snapshot = buildStateSnapshot(ctx, inp_csa, csa_state, csa_snapshot_source_kv, csa_snapshot_source_score, il);
        if (csa_snapshot.kv) |t| c.ggml_build_forward_expand(b.gf, t);
        if (csa_snapshot.score) |t| c.ggml_build_forward_expand(b.gf, t);

        const csa_persist_kv: *Tensor = c.ggml_get_rows(ctx, csa_state_kv, inp_csa.state_persist_src_idxs);
        const csa_persist_score: *Tensor = c.ggml_get_rows(ctx, csa_state_score, inp_csa.state_persist_src_idxs);

        csa_state_kv = d.mctx.getCsaState().cpyKv(ctx, csa_persist_kv, inp_csa.state_persist_dst_idxs.?, il);
        csa_state_score = d.mctx.getCsaState().cpyScore(ctx, csa_persist_score, inp_csa.state_persist_dst_idxs.?, il);

        c.ggml_build_forward_expand(b.gf, csa_state_kv);
        c.ggml_build_forward_expand(b.gf, csa_state_score);

        var lid_state_kv = b.buildLoraMm(layer.indexer_comp_wkv.?, cur, null);
        b.cb(lid_state_kv, "lid_state_kv", il);

        var lid_state_score = b.buildLoraMm(layer.indexer_comp_wgate.?, cur, null);
        b.cb(lid_state_score, "lid_state_score", il);

        const lid_ape = layer.indexer_comp_ape;

        const lid_ape_rows: *Tensor = c.ggml_get_rows(ctx, lid_ape, inp_lid.state_pos);
        lid_state_score = c.ggml_add(ctx, lid_state_score, lid_ape_rows);
        b.cb(lid_state_score, "lid_state_score_ape", il);

        impl.assert(inp_lid.state_write_idxs != null, src, 1038, "inp_dsv4->get_lid().state_write_idxs");

        const lid_state = d.mctx.getLidState();
        const lid_restored = buildStateRestore(ctx, inp_lid, lid_state, il);
        const lid_base_kv = view2d(ctx, lid_restored.kv.?, lid_restored.kv.?.ne[0], lid_state.getNRows(), 0);
        const lid_base_score = view2d(ctx, lid_restored.score.?, lid_restored.score.?.ne[0], lid_state.getNRows(), 0);

        const lid_source_kv: *Tensor = c.ggml_concat(ctx, lid_base_kv, lid_state_kv, 1);
        const lid_source_score: *Tensor = c.ggml_concat(ctx, lid_base_score, lid_state_score, 1);

        var kv_comp_lid_state = buildOverlapCompressedKvFromState(b, lid_source_kv, lid_source_score, inp_lid.state_read_idxs, inp_lid.state_write_pos, layer.indexer_comp_norm, csa_ratio, hp.indexer_head_size, "lid_state_compress", il);

        if (inp_lid.k_rot) |r| {
            kv_comp_lid_state = graph.mulMatHadamard(ctx, kv_comp_lid_state, r);
            b.cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        c.ggml_build_forward_expand(b.gf, d.mctx.getLid().cpyK(ctx, kv_comp_lid_state, inp_lid.state_write_idxs.?, il));

        const lid_snapshot_source_kv: *Tensor = c.ggml_concat(ctx, lid_restored.kv, lid_state_kv, 1);
        const lid_snapshot_source_score: *Tensor = c.ggml_concat(ctx, lid_restored.score, lid_state_score, 1);

        const lid_snapshot = buildStateSnapshot(ctx, inp_lid, lid_state, lid_snapshot_source_kv, lid_snapshot_source_score, il);
        if (lid_snapshot.kv) |t| c.ggml_build_forward_expand(b.gf, t);
        if (lid_snapshot.score) |t| c.ggml_build_forward_expand(b.gf, t);

        const lid_persist_kv: *Tensor = c.ggml_get_rows(ctx, lid_state_kv, inp_lid.state_persist_src_idxs);
        const lid_persist_score: *Tensor = c.ggml_get_rows(ctx, lid_state_score, inp_lid.state_persist_src_idxs);

        lid_state_kv = d.mctx.getLidState().cpyKv(ctx, lid_persist_kv, inp_lid.state_persist_dst_idxs.?, il);
        lid_state_score = d.mctx.getLidState().cpyScore(ctx, lid_persist_score, inp_lid.state_persist_dst_idxs.?, il);

        c.ggml_build_forward_expand(b.gf, lid_state_kv);
        c.ggml_build_forward_expand(b.gf, lid_state_score);
    }

    var hca_state: ?*const kv_cache_dsv4.CompState = null;
    var hca_restored: StateTensors = .{};
    if (ratio == hca_ratio and inp_dsv4.?.getHca().state_write_idxs != null) {
        const d = inp_dsv4.?;
        const inp_hca = d.getHca();
        impl.assert(hca_state_kv != null, src, 1099, "hca_state_kv");
        impl.assert(hca_state_score != null, src, 1100, "hca_state_score");

        const st = d.mctx.getHcaState();
        hca_state = st;
        hca_restored = buildStateRestore(ctx, inp_hca, st, il);
        const hca_base_kv = view2d(ctx, hca_restored.kv.?, hca_restored.kv.?.ne[0], st.getNRows(), 0);
        const hca_base_score = view2d(ctx, hca_restored.score.?, hca_restored.score.?.ne[0], st.getNRows(), 0);

        hca_source_kv = c.ggml_concat(ctx, hca_base_kv, hca_state_kv, 1);
        hca_source_score = c.ggml_concat(ctx, hca_base_score, hca_state_score, 1);

        var kv_comp_hca = buildHcaCompressedKvFromState(b, hca_source_kv.?, hca_source_score.?, inp_hca.state_read_idxs, inp_hca.state_write_pos, layer.attn_comp_norm, n_embd_head, "hca_state_compress", il);

        if (inp_hca.k_rot) |r| {
            kv_comp_hca = graph.mulMatHadamard(ctx, kv_comp_hca, r);
            b.cb(kv_comp_hca, "hca_state_compress_rot", il);
        }

        c.ggml_build_forward_expand(b.gf, d.mctx.getHca().cpyK(ctx, kv_comp_hca, inp_hca.state_write_idxs.?, il));
    }

    if (ratio == hca_ratio and inp_dsv4.?.getHca().state_pos != null) {
        const d = inp_dsv4.?;
        const inp_hca = d.getHca();
        impl.assert(hca_state_kv != null, src, 1132, "hca_state_kv");
        impl.assert(hca_state_score != null, src, 1133, "hca_state_score");

        if (hca_state == null) hca_state = d.mctx.getHcaState();
        const st = hca_state.?;
        if (hca_restored.kv == null) hca_restored = buildStateRestore(ctx, inp_hca, st, il);
        if (hca_source_kv == null or hca_source_score == null) {
            const hca_base_kv = view2d(ctx, hca_restored.kv.?, hca_restored.kv.?.ne[0], st.getNRows(), 0);
            const hca_base_score = view2d(ctx, hca_restored.score.?, hca_restored.score.?.ne[0], st.getNRows(), 0);

            hca_source_kv = c.ggml_concat(ctx, hca_base_kv, hca_state_kv, 1);
            hca_source_score = c.ggml_concat(ctx, hca_base_score, hca_state_score, 1);
        }

        const hca_snapshot_source_kv: *Tensor = c.ggml_concat(ctx, hca_restored.kv, hca_state_kv, 1);
        const hca_snapshot_source_score: *Tensor = c.ggml_concat(ctx, hca_restored.score, hca_state_score, 1);

        const hca_snapshot = buildStateSnapshot(ctx, inp_hca, st, hca_snapshot_source_kv, hca_snapshot_source_score, il);
        if (hca_snapshot.kv) |t| c.ggml_build_forward_expand(b.gf, t);
        if (hca_snapshot.score) |t| c.ggml_build_forward_expand(b.gf, t);

        const hca_persist_kv: *Tensor = c.ggml_get_rows(ctx, hca_state_kv, inp_hca.state_persist_src_idxs);
        const hca_persist_score: *Tensor = c.ggml_get_rows(ctx, hca_state_score, inp_hca.state_persist_src_idxs);

        hca_state_kv = d.mctx.getHcaState().cpyKv(ctx, hca_persist_kv, inp_hca.state_persist_dst_idxs.?, il);
        hca_state_score = d.mctx.getHcaState().cpyScore(ctx, hca_persist_score, inp_hca.state_persist_dst_idxs.?, il);

        c.ggml_build_forward_expand(b.gf, hca_state_kv);
        c.ggml_build_forward_expand(b.gf, hca_state_score);
    }

    const kq_scale = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    var out: *Tensor = undefined;
    if (inp_mtp) |m| {
        out = b.buildAttn(m, null, null, null, q, kv, kv, null, layer.attn_sinks, null, kq_scale, il);
        b.cb(out, "attn_raw", il);
    } else if (ratio == csa_ratio and
        inp_dsv4.?.getCsa().kq_mask != null and
        inp_dsv4.?.getLid().kq_mask != null and
        inp_dsv4.?.getLid().k_rot != null)
    {
        out = buildCsaLidAttention(b, model, inp_dsv4.?, inp_attn.?, q, kv, qr, cur, inp_pos, layer.attn_sinks, kq_scale, il);
    } else if (ratio == hca_ratio and inp_dsv4.?.getHca().kq_mask != null) {
        out = buildHcaAttention(b, inp_dsv4.?, inp_attn.?, q, kv, layer.attn_sinks, kq_scale, il);
    } else {
        out = buildRawAttention(b, inp_attn.?, q, kv, layer.attn_sinks, kq_scale, il);
    }

    out = c.ggml_reshape_3d(ctx, out, n_embd_head, b.n_head, nt);
    out = c.ggml_rope_ext_back(ctx, out, inp_pos, null, @intCast(n_embd_head_rope), b.rope_type, n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = c.ggml_rope_set_offset(out, @intCast(n_embd_head_nope));
    b.cb(out, "attn_derope", il);

    out = c.ggml_reshape_3d(ctx, out, o_group_dim, n_groups, nt);
    out = c.ggml_permute(ctx, out, 0, 2, 1, 3);
    var oa: *Tensor = c.ggml_mul_mat(ctx, layer.wo_a, out);
    b.cb(oa, "attn_wo_a", il);
    oa = c.ggml_permute(ctx, oa, 0, 2, 1, 3);
    oa = c.ggml_cont_2d(ctx, oa, o_lora_rank * n_groups, nt);

    out = b.buildLoraMm(layer.wo_b.?, oa, null);
    b.cb(out, "attn_out", il);

    return out;
}

// -----------------------------------------------------------------------------
// The graphs

/// Ports the constructor of `llama_model_deepseek4`'s `graph`
/// (models/deepseek4.cpp:1219 @c1d0e7a00).
pub fn buildGraph(model: *const Model, b: *const Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    var cur: *Tensor = undefined;

    const inp = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();
    const inp_dsv4 = b.buildInpDsv4();
    const inp_attn = inp_dsv4.getRaw();
    c.ggml_build_forward_expand(b.gf, inp_attn.self_kq_mask);

    const hc: i64 = hp.dsv4_hc_mult;
    var inpL: *Tensor = c.ggml_reshape_3d(ctx, inp, b.n_embd, 1, b.n_tokens);
    inpL = c.ggml_repeat_4d(ctx, inpL, b.n_embd, hc, b.n_tokens, 1);
    b.cb(inpL, "hc_init", -1);

    const layer_inp = b.cparams.embeddings_layer_inp;

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const uil: usize = @intCast(il);
        if (uil < layer_inp.len and layer_inp[uil]) {
            const t = hcMean(ctx, inpL);
            b.res.t_layer_inp[uil] = t;
            b.cb(t, "layer_inp", il);
            c.ggml_build_forward_expand(b.gf, t);
        }

        const layer = &model.layers[uil];
        var residual = inpL;
        var post: ?*Tensor = null;
        var comb: ?*Tensor = null;

        cur = buildHcPre(b, inpL, layer.hc_attn_fn.?, layer.hc_attn_scale.?, layer.hc_attn_base.?, &post, &comb, il);
        b.cb(cur, "hc_attn_pre", il);

        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        cur = buildAttention(b, model, .{ .dsv4 = inp_dsv4 }, cur, inp_pos, il);

        inpL = buildHcPost(b, cur, residual, post.?, comb.?, il);
        b.cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = buildHcPre(b, inpL, layer.hc_ffn_fn.?, layer.hc_ffn_scale.?, layer.hc_ffn_base.?, &post, &comb, il);
        b.cb(cur, "hc_ffn_pre", il);

        c.ggml_build_forward_expand(b.gf, residual);
        c.ggml_build_forward_expand(b.gf, post);
        c.ggml_build_forward_expand(b.gf, comb);

        cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        var selected_experts: ?*Tensor = null;
        var exp_probs_b = layer.ffn_exp_probs_b;
        if (@as(u32, @intCast(il)) < hp.dsv4_hash_layer_count) {
            // The leading layers route each token by a fixed table.
            selected_experts = c.ggml_get_rows(ctx, layer.ffn_gate_tid2eid, b.res.t_inp_tokens);
            exp_probs_b = null;
        }

        const moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = exp_probs_b,
            .n_expert = b.n_expert,
            .n_expert_used = hp.n_expert_used,
            .type_op = .silu,
            .norm_w = hp.expert_weights_norm,
            .w_scale = hp.expert_weights_scale,
            .gating_op = @fromBackingInt(hp.expert_gating_func),
            .selected_experts_in = selected_experts,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
        b.cb(ffn_shexp, "ffn_shexp", il);

        cur = c.ggml_add(ctx, moe_out, ffn_shexp);
        b.cb(cur, "ffn_out", il);

        inpL = buildHcPost(b, cur, residual, post.?, comb.?, il);
        inpL = b.buildCvec(inpL, il);
        b.cb(inpL, "l_last", il);
    }

    const n_layer: usize = @intCast(b.n_layer);
    if (n_layer < layer_inp.len and layer_inp[n_layer]) {
        const t = hcMean(ctx, inpL);
        b.res.t_layer_inp[n_layer] = t;
        b.cb(t, "layer_inp", @intCast(n_layer));
        c.ggml_build_forward_expand(b.gf, t);
    }

    const flat: *Tensor = c.ggml_reshape_2d(ctx, inpL, b.n_embd * hc, b.n_tokens);
    const flat_out: *Tensor = c.ggml_get_rows(ctx, flat, inp_out_ids);

    if (b.cparams.embeddings_nextn) {
        const h_nextn = if (b.cparams.embeddings_nextn_masked) flat_out else inpL;
        b.cb(h_nextn, "h_nextn", -1);
        b.res.t_h_nextn = h_nextn;
    }

    inpL = c.ggml_reshape_3d(ctx, flat_out, b.n_embd, hc, b.n_outputs);

    cur = buildHcHead(b, inpL, model.hc_head_fn.?, model.hc_head_scale.?, model.hc_head_base.?);
    b.cb(cur, "hc_head", -1);

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = c.ggml_mul_mat(ctx, model.output, cur);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_deepseek4`'s `graph_mtp`
/// (models/deepseek4.cpp:1352 @c1d0e7a00): the MTP block over the trunk's
/// hidden streams and the next token's embedding, through the same
/// hyper-connections, then the shared head.
pub fn buildGraphMtp(model: *const Model, b: *const Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const cp = b.cparams;
    impl.assert(hp.n_layer_nextn > 0, src, 1354, "hparams.n_layer_nextn > 0 && \"DEEPSEEK4 MTP requires n_layer_nextn > 0\"");
    impl.assert(hp.n_layer_nextn == 1, src, 1355, "hparams.n_layer_nextn == 1 && \"DEEPSEEK4 MTP currently only supports a single MTP block\"");
    impl.assert(cp.nextn_layer_offset >= 0 and cp.nextn_layer_offset < @as(i32, @intCast(hp.n_layer_nextn)), src, 1356, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range [0, n_layer_nextn)\"");
    impl.assert(b.ubatch.token != null, src, 1359, "ubatch.token && \"DEEPSEEK4 MTP requires token input\"");

    const hc: i64 = hp.dsv4_hc_mult;
    impl.assert(hp.nEmbdOut() == @as(u32, @truncate(@as(u64, @bitCast(b.n_embd * hc)))), src, 1362, "hparams.n_embd_out() == (uint32_t) (n_embd*hc) && \"DEEPSEEK4 MTP hidden width mismatch\"");

    const il: c_int = @intCast(@as(i64, hp.nLayer()) + cp.nextn_layer_offset);
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, src, 1367, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, src, 1368, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, src, 1369, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");

    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort(src, 1371, "out of memory");
    inp.* = .{ .n_embd = hp.nEmbdOut() };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.nEmbdOut(), b.n_tokens);
    c.ggml_set_input(inp.embd);

    inp.h = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.nEmbdOut(), b.n_tokens);
    c.ggml_set_input(inp.h);
    _ = c.ggml_set_name(inp.h, "mtp_h_input");

    const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;
    const tok_embd: *Tensor = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    b.cb(tok_embd, "mtp_tok_embd", il);

    const h_state: *Tensor = c.ggml_reshape_3d(ctx, inp.h, b.n_embd, hc, b.n_tokens);
    b.cb(h_state, "mtp_h_state", il);

    _ = b.res.addInput(graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();
    const inp_attn = b.buildAttnInpKIswa();

    const h_norm = b.buildNorm(h_state, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    var e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    e_norm = c.ggml_reshape_3d(ctx, e_norm, b.n_embd, 1, b.n_tokens);
    e_norm = c.ggml_repeat_4d(ctx, e_norm, b.n_embd, hc, b.n_tokens, 1);
    b.cb(e_norm, "mtp_enorm", il);

    const concat: *Tensor = c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var inpL = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(inpL, "mtp_eh_proj", il);

    var residual = inpL;
    var post: ?*Tensor = null;
    var comb: ?*Tensor = null;

    var cur = buildHcPre(b, inpL, layer.hc_attn_fn.?, layer.hc_attn_scale.?, layer.hc_attn_base.?, &post, &comb, il);
    b.cb(cur, "mtp_hc_attn_pre", il);

    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    cur = buildAttention(b, model, .{ .mtp = inp_attn }, cur, inp_pos, il);

    inpL = buildHcPost(b, cur, residual, post.?, comb.?, il);
    b.cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    cur = buildHcPre(b, inpL, layer.hc_ffn_fn.?, layer.hc_ffn_scale.?, layer.hc_ffn_base.?, &post, &comb, il);
    b.cb(cur, "mtp_hc_ffn_pre", il);

    cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "mtp_ffn_norm", il);

    impl.assert(@as(u32, @intCast(il)) >= hp.dsv4_hash_layer_count, src, 1440, "(uint32_t) il >= hparams.dsv4_hash_layer_count && \"DEEPSEEK4 MTP does not support hash-routed MTP blocks\"");
    const moe_out = b.buildMoeFfn(cur, .{
        .gate_inp = layer.ffn_gate_inp,
        .up_exps = layer.ffn_up_exps,
        .gate_exps = layer.ffn_gate_exps,
        .down_exps = layer.ffn_down_exps.?,
        .exp_probs_b = layer.ffn_exp_probs_b,
        .n_expert = b.n_expert,
        .n_expert_used = hp.n_expert_used,
        .type_op = .silu,
        .norm_w = hp.expert_weights_norm,
        .w_scale = hp.expert_weights_scale,
        .gating_op = @fromBackingInt(hp.expert_gating_func),
    }, il);
    b.cb(moe_out, "mtp_ffn_moe_out", il);

    const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
    b.cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = c.ggml_add(ctx, moe_out, ffn_shexp);
    b.cb(cur, "mtp_ffn_out", il);

    inpL = buildHcPost(b, cur, residual, post.?, comb.?, il);
    inpL = b.buildCvec(inpL, il);
    b.cb(inpL, "mtp_l_out", il);

    const flat: *Tensor = c.ggml_reshape_2d(ctx, inpL, b.n_embd * hc, b.n_tokens);
    const h_nextn: *Tensor = c.ggml_get_rows(ctx, flat, inp_out_ids);
    b.cb(h_nextn, "h_nextn", -1);
    b.res.t_h_nextn = h_nextn;

    inpL = c.ggml_reshape_3d(ctx, h_nextn, b.n_embd, hc, b.n_outputs);

    cur = buildHcHead(b, inpL, model.hc_head_fn.?, model.hc_head_scale.?, model.hc_head_base.?);
    b.cb(cur, "mtp_hc_head", -1);

    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    impl.assert(head_norm_w != null, src, 1479, "head_norm_w && \"DEEPSEEK4 MTP missing shared head norm\"");
    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);
    b.cb(cur, "mtp_shared_head_norm", -1);
    b.res.t_embd = cur;

    const head_w = layer.nextn.shared_head_head orelse model.output;
    impl.assert(head_w != null, src, 1485, "head_w && \"DEEPSEEK4 MTP missing LM head\"");
    cur = c.ggml_mul_mat(ctx, head_w, cur);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the RoPE magnitude correction is one without extrapolation" {
    try std.testing.expectEqual(@as(f32, 1.0), ropeAttnFactor(0.25, 0.0));
    try std.testing.expect(ropeAttnFactor(0.25, 1.0) < 1.0);
}
