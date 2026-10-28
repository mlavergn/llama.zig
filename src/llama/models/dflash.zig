//! DFlash: a block-diffusion speculative-decoding draft over its target
//! model's hidden features, and DSpark, DFlash with a Markov head and a
//! confidence head.
//!
//! The encoder fuses the target's chosen layers through `fc` and a norm.
//! The decoder runs in one of two modes, chosen by the batch: given those
//! fused features as embeddings, it projects them through each layer's K
//! and V and writes them into the cache; given tokens -- the committed one
//! followed by mask tokens -- it attends over that cache and the block to
//! draft the whole block at once. DSpark then chains a low-rank Markov bias
//! through each block's positions and scores each position's acceptance.
//! Its DeepSeek-V4 variant runs full DeepSeek-V4 blocks as the stages.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/dflash.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_dflash`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph<is_enc>` with `build_inp_embd_enc`,
//! `build_dspark_markov_head`, and `graph_dsv4`, which builds on
//! DeepSeek-V4's graph through `models/deepseek4.zig`'s hyper-connection
//! and attention builders.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const context_mod = @import("../context.zig");
const hparams_mod = @import("../hparams.zig");
const deepseek4 = @import("deepseek4.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("models/dflash.cpp", 0, "out of memory");
}

/// Ports `llama_model_dflash`'s `load_arch_hparams` (models/dflash.cpp:7 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(f32, .logit_scale, false, exc)) |v| hp.f_logit_scale = v;
    hp.f_final_logit_softcapping = 0.0;
    if (try meta.getKey(f32, .final_logit_softcapping, false, exc)) |v| hp.f_final_logit_softcapping = v;

    // Drafts for M-RoPE targets carry the degenerate sections
    // [n_rot/2, 0, 0, 0].
    _ = try meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, false, exc);

    if (try meta.getKey(u32, .dflash_block_size, false, exc)) |v| hp.dflash_block_size = v;
    if (try meta.getKey(u32, .dflash_conv_kernel_size, false, exc)) |v| hp.dflash_conv_kernel_size = v;
    if (try meta.getKey(u32, .dflash_conv_group_size, false, exc)) |v| hp.dflash_conv_group_size = v;
    if (try meta.getKey(u32, .dflash_selector_rank, false, exc)) |v| hp.dflash_selector_rank = v;
    if (try meta.getKey(u32, .dflash_selector_top_k, false, exc)) |v| hp.dflash_selector_top_k = v;

    const ids = (try meta.getArrList(i32, .target_layers, false, exc)) orelse {
        return exc.throw(.runtime_error, "DFlash model requires 'target_layers' in GGUF metadata", .{});
    };
    defer allocator.free(ids);
    self.target_layer_ids.clearRetainingCapacity();
    self.target_layer_ids.appendSlice(allocator, ids) catch oom();

    hp.n_embd_inp_enc_impl = @as(u32, @truncate(self.target_layer_ids.items.len)) *% hp.n_embd;

    var layers: std.ArrayList(u8) = .empty;
    defer layers.deinit(allocator);
    var sep: []const u8 = "";
    for (self.target_layer_ids.items) |id| {
        layers.appendSlice(allocator, sep) catch oom();
        layers.print(allocator, "{d}", .{id}) catch oom();
        sep = ", ";
    }
    layers.append(allocator, 0) catch oom();
    impl.logInfo("%s: DFlash extract_layers = [%s]\n", .{ "load_arch_hparams", @as([*:0]const u8, @ptrCast(layers.items.ptr)) });

    // The DeepSeek-V4 DSpark backbone: the stages are full DeepSeek-V4
    // blocks, with a uniform sliding window (the draft's KV ring).
    if (try meta.getKey(u32, .hyper_connection_count, false, exc)) |v| hp.dsv4_hc_mult = v;
    if (hp.dsv4_hc_mult > 0) {
        hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
        hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
        _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
        hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
        hp.expert_weights_scale = (try meta.getKey(f32, .expert_weights_scale, true, exc)).?;
        hp.expert_weights_norm = (try meta.getKey(bool, .expert_weights_norm, true, exc)).?;
        hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;
        _ = try meta.getKeyOrArr(f32, .swiglu_clamp_exp, &hp.swiglu_clamp_exp, hp.n_layer_all, true, exc);
        if (!try meta.getKeyOrArr(f32, .swiglu_clamp_shexp, &hp.swiglu_clamp_shexp, hp.n_layer_all, false, exc)) {
            hp.swiglu_clamp_shexp = hp.swiglu_clamp_exp;
        }
        hp.dsv4_o_group_count = (try meta.getKey(u32, .attention_output_group_count, true, exc)).?;
        hp.dsv4_o_lora_rank = (try meta.getKey(u32, .attention_output_lora_rank, true, exc)).?;
        hp.dsv4_hc_sinkhorn_iters = (try meta.getKey(u32, .hyper_connection_sinkhorn_iterations, true, exc)).?;
        hp.dsv4_hc_eps = (try meta.getKey(f32, .hyper_connection_epsilon, true, exc)).?;
        _ = try meta.getArr(u32, .attention_compress_ratios, &hp.dsv4_compress_ratios, false, exc);

        // Against a division by zero.
        impl.assert(hp.dsv4_o_group_count > 0, "models/dflash.cpp", 58, "hparams.dsv4_o_group_count > 0");

        if (hp.expert_gating_func != @backingInt(hparams_mod.ExpertGatingFunc.sqrt_softplus)) {
            return exc.throw(.runtime_error, "DSpark DSV4 draft expects sqrtsoftplus MoE scoring", .{});
        }
        var il: u32 = 0;
        while (il < hp.n_layer_all) : (il += 1) {
            if (hp.dsv4_compress_ratios[il] != 0) {
                return exc.throw(.runtime_error, "DSpark DSV4 draft expects uncompressed attention on all stages", .{});
            }
        }

        impl.assert(hp.n_swa > 0, "models/dflash.cpp", 69, "hparams.n_swa > 0");
        hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
        hp.setSwaPattern(0, false);
        il = 0;
        while (il < hp.n_layer_all) : (il += 1) {
            hp.is_swa_impl[il] = 1;
        }
        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;

        self.type = .unknown;
        return;
    }

    // Optional interleaved sliding-window attention with a per-layer
    // pattern array. DFlash has a single RoPE, so the SWA RoPE is the main
    // one.
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| {
        hp.n_swa = v;
        if (hp.n_swa > 0) {
            hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
            _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);
            hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
            hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
        }
    }

    self.type = .unknown;
}

/// Ports `llama_model_dflash`'s `load_arch_tensors` (models/dflash.cpp:94 @5266f24da).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const func = "load_arch_tensors";

    const n_embd_inp: i64 = hp.nEmbdInpEnc();

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // A reduced draft vocabulary, optional: d2t maps the draft's rows to
    // target token ids.
    var n_vocab_draft = l.n_vocab;
    if (ml.getTensorMeta("d2t")) |d2t_meta| {
        n_vocab_draft = d2t_meta.ne[0];
        self.d2t = try self.createTensor(.{ .tensor = .d2t }, &.{n_vocab_draft}, 0, exc);
        impl.logInfo("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", .{ func, @as(c_longlong, n_vocab_draft) });
    }

    // DSpark is DFlash plus a semi-autoregressive Markov head and a
    // confidence head. Upstream's TODO: only Qwen3-style backbones are
    // supported for now; others (e.g. Gemma 4) need their own conversion
    // path and graph changes.
    if (ml.getTensorMeta("markov_w1.weight")) |markov_meta| {
        const dspark_markov_rank = markov_meta.ne[0];

        self.dspark_markov_w1 = try self.createTensor(.{ .tensor = .dspark_markov_w1, .suffix = "weight" }, &.{ dspark_markov_rank, l.n_vocab }, 0, exc);
        self.dspark_markov_w2 = try self.createTensor(.{ .tensor = .dspark_markov_w2, .suffix = "weight" }, &.{ dspark_markov_rank, n_vocab_draft }, 0, exc);
        self.dspark_markov_w2_s = try self.createTensor(.{ .tensor = .dspark_markov_w2, .suffix = "scale" }, &.{1}, nr, exc);

        self.dspark_conf_proj = try self.createTensor(.{ .tensor = .dspark_conf_proj, .suffix = "weight" }, &.{ l.n_embd + dspark_markov_rank, 1 }, nr, exc);
        self.dspark_conf_proj_b = try self.createTensor(.{ .tensor = .dspark_conf_proj, .suffix = "bias" }, &.{1}, nr, exc);

        impl.logInfo("%s: DFlash with DSpark markov head (rank = %lld)\n", .{ func, @as(c_longlong, dspark_markov_rank) });
    }

    // DFlash2: a conv around each attention and FFN block, and a selector
    // that scores candidate transitions between block positions.
    const selector_meta = ml.getTensorMeta("selector_hidden.weight");
    if (selector_meta != null) {
        const rank: i64 = hp.dflash_selector_rank;
        if (rank <= 0 or hp.dflash_block_size <= 0 or hp.dflash_selector_top_k <= 0 or
            hp.dflash_conv_kernel_size <= 0 or hp.dflash_conv_group_size <= 0)
        {
            return exc.throw(.runtime_error, "DFlash2 model is missing conv/selector metadata", .{});
        }
        if (@rem(l.n_embd, @as(i64, hp.dflash_conv_group_size)) != 0) {
            return exc.throw(.runtime_error, "DFlash2 hidden size must be divisible by conv_group_size", .{});
        }
        // `uint32_t` arithmetic in the C++, compared as `int64_t`.
        if (l.n_embd < @as(i64, hp.dflash_selector_top_k *% (hp.dflash_selector_top_k +% 1))) {
            return exc.throw(.runtime_error, "DFlash2 hidden size is too small for the selector lattice", .{});
        }

        self.dflash_selector_prev = try self.createTensor(.{ .tensor = .dflash_selector_prev, .suffix = "weight" }, &.{ rank, l.n_vocab }, 0, exc);
        self.dflash_selector_next = try self.createTensor(.{ .tensor = .dflash_selector_next, .suffix = "weight" }, &.{ rank, l.n_vocab }, 0, exc);
        self.dflash_selector_hidden = try self.createTensor(.{ .tensor = .dflash_selector_hidden, .suffix = "weight" }, &.{ l.n_embd, rank }, 0, exc);

        impl.logInfo("%s: DFlash2 conv kernel = %u, group = %u, selector rank = %u, top-k = %u\n", .{ func, hp.dflash_conv_kernel_size, hp.dflash_conv_group_size, hp.dflash_selector_rank, hp.dflash_selector_top_k });
    }

    self.fc = try self.createTensor(.{ .tensor = .fc, .suffix = "weight" }, &.{ n_embd_inp, l.n_embd }, 0, exc);
    self.fc_s = try self.createTensor(.{ .tensor = .fc, .suffix = "scale" }, &.{1}, nr, exc);
    // The encoder's hidden_norm, after fc.
    self.output_norm_enc = try self.createTensor(.{ .tensor = .enc_output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    // The decoder's final norm.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    // Optional: reduced-vocabulary drafts ship their own LM head, and
    // full-vocabulary drafts can share the target's through ctx_other. A
    // draft with its own embeddings and head references no target tensors,
    // so it can run on devices the target does not use (e.g. -devd with a
    // tensor-split target).
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, n_vocab_draft }, nr, exc);

    if (hp.dsv4_hc_mult > 0) {
        const q_lora_rank: i64 = hp.n_lora_q;
        const n_ff_exp: i64 = hp.nFfExp(0);
        const n_expert_shared: i64 = hp.n_expert_shared;
        const n_embd_head: i64 = hp.nEmbdHeadK(0);
        const o_groups: i64 = hp.dsv4_o_group_count;
        const o_lora_rank: i64 = hp.dsv4_o_lora_rank;
        const hc_mult: i64 = hp.dsv4_hc_mult;
        const hc_dim = hc_mult * l.n_embd;
        const hc_mix_dim = (2 + hc_mult) * hc_mult;

        self.hc_head_fn = try self.createTensor(.{ .tensor = .hc_head_fn, .suffix = "weight" }, &.{ hc_dim, hc_mult }, 0, exc);
        self.hc_head_base = try self.createTensor(.{ .tensor = .hc_head_base, .suffix = "weight" }, &.{hc_mult}, 0, exc);
        self.hc_head_scale = try self.createTensor(.{ .tensor = .hc_head_scale, .suffix = "weight" }, &.{1}, 0, exc);

        var i: i32 = 0;
        while (i < l.n_layer) : (i += 1) {
            const layer = &self.layers[@intCast(i)];

            layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.attn_sinks = try self.createTensor(.{ .tensor = .attn_sinks, .suffix = "weight", .bid = i }, &.{l.n_head}, 0, exc);
            layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_lora_rank }, 0, exc);
            layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, 0, exc);
            layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, l.n_head * n_embd_head }, 0, exc);
            layer.wkv = try self.createTensor(.{ .tensor = .attn_kv, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_head }, 0, exc);
            layer.attn_kv_norm = try self.createTensor(.{ .tensor = .attn_kv_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, 0, exc);
            layer.wo_a = try self.createTensor(.{ .tensor = .attn_out_a, .suffix = "weight", .bid = i }, &.{ @divTrunc(l.n_head * n_embd_head, o_groups), o_lora_rank, o_groups }, loader.tensor_allow_reshape, exc);
            layer.wo_b = try self.createTensor(.{ .tensor = .attn_out_b, .suffix = "weight", .bid = i }, &.{ o_groups * o_lora_rank, l.n_embd }, 0, exc);

            layer.hc_attn_fn = try self.createTensor(.{ .tensor = .hc_attn_fn, .suffix = "weight", .bid = i }, &.{ hc_dim, hc_mix_dim }, 0, exc);
            layer.hc_attn_base = try self.createTensor(.{ .tensor = .hc_attn_base, .suffix = "weight", .bid = i }, &.{hc_mix_dim}, 0, exc);
            layer.hc_attn_scale = try self.createTensor(.{ .tensor = .hc_attn_scale, .suffix = "weight", .bid = i }, &.{3}, 0, exc);
            layer.hc_ffn_fn = try self.createTensor(.{ .tensor = .hc_ffn_fn, .suffix = "weight", .bid = i }, &.{ hc_dim, hc_mix_dim }, 0, exc);
            layer.hc_ffn_base = try self.createTensor(.{ .tensor = .hc_ffn_base, .suffix = "weight", .bid = i }, &.{hc_mix_dim}, 0, exc);
            layer.hc_ffn_scale = try self.createTensor(.{ .tensor = .hc_ffn_scale, .suffix = "weight", .bid = i }, &.{3}, 0, exc);

            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);
            layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, 0, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
        }
        return;
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head }, 0, exc);
        layer.wk = try self.createTensor(.{ .tensor = .attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_k_gqa }, 0, exc);
        layer.wv = try self.createTensor(.{ .tensor = .attn_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_v_gqa }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        // Optional per-head attention sinks (Nemotron DSpark, say).
        layer.attn_sinks = try self.createTensor(.{ .tensor = .attn_sinks, .suffix = "weight", .bid = i }, &.{l.n_head}, nr, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

        if (selector_meta != null) {
            const kernel: i64 = hp.dflash_conv_kernel_size;
            const groups: i64 = @divTrunc(l.n_embd, @as(i64, hp.dflash_conv_group_size));
            const projected = 2 * kernel * groups;
            layer.dflash_attn_conv_base = try self.createTensor(.{ .tensor = .dflash_attn_conv_base, .bid = i }, &.{ l.n_embd, kernel, 2 }, 0, exc);
            layer.dflash_attn_conv_proj = try self.createTensor(.{ .tensor = .dflash_attn_conv_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, projected }, 0, exc);
            layer.dflash_ffn_conv_base = try self.createTensor(.{ .tensor = .dflash_ffn_conv_base, .bid = i }, &.{ l.n_embd, kernel, 2 }, 0, exc);
            layer.dflash_ffn_conv_proj = try self.createTensor(.{ .tensor = .dflash_ffn_conv_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, projected }, 0, exc);
        }
    }
}

/// Ports `llama_model_dflash`'s `build_arch_graph` (models/dflash.cpp:243 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    switch (params.gtype) {
        .encoder => buildGraphEnc(model, b),
        .default, .decoder => {
            if (b.hparams.dsv4_hc_mult > 0) {
                return buildGraphDsv4(model, b);
            }
            // Optional iSWA: the matching attention input.
            if (@as(hparams_mod.SwaType, @fromBackingInt(@intCast(b.hparams.swa_type))) != .none) {
                buildGraphDec(true, model, b);
            } else {
                buildGraphDec(false, model, b);
            }
        },
        else => impl.abort("models/dflash.cpp", 254, "invalid graph type"),
    }
}

/// `llama_get_model(cparams.ctx_other)`: the target model a draft shares
/// weights with.
fn otherModel(b: *const graph.Builder) *const Model {
    const other: *const context_mod.Context = @ptrCast(@alignCast(b.cparams.ctx_other.?));
    return other.getModel();
}

/// Ports `llama_model_dflash`'s `graph<true>`'s `build_inp_embd_enc`
/// (models/dflash.cpp:259 @5266f24da): the target model's features, its
/// chosen layers concatenated, given as the batch's embeddings.
fn buildInpEmbdEnc(b: *const graph.Builder) *Tensor {
    const hp = b.hparams;

    const n_embd_inp: i64 = hp.nEmbdInpEnc();
    const inp_target = allocator.create(graph.InpEmbd) catch oom();
    inp_target.* = .{ .n_embd = n_embd_inp };

    inp_target.embd = c.ggml_new_tensor_2d(b.ctx0, c.GGML_TYPE_F32, n_embd_inp, b.n_tokens);
    c.ggml_set_input(inp_target.embd);

    const cur = inp_target.embd.?;
    b.cb(cur, "inp_embd", -1);

    _ = b.res.addInput(graph.InpEmbd, inp_target);

    return cur;
}

/// Ports the constructor of `llama_model_dflash`'s `graph<true>`
/// (models/dflash.cpp:276 @5266f24da): the encoder, the target model's
/// features through the fusion layer and its norm.
fn buildGraphEnc(model: *const Model, b: *const graph.Builder) void {
    var cur = buildInpEmbdEnc(b);

    cur = b.buildLoraMm(model.fc.?, cur, model.fc_s);
    b.cb(cur, "fc_out", -1);

    cur = b.buildNorm(cur, model.output_norm_enc, null, .rms, -1);
    b.cb(cur, "enc_norm_out", -1);

    c.ggml_set_output(cur);
    b.res.t_h_nextn = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// `std::stoi` on a metadata value. A value with no leading integer, or one
/// past `int`'s range, throws from inside the graph build in the C++, where
/// nothing catches it.
fn stoi(s: [:0]const u8) i64 {
    var end: [*c]u8 = undefined;
    // strtol saturates at LONG_MIN and LONG_MAX, both past int's range.
    const v = strtol(s.ptr, &end, 10);
    if (@intFromPtr(end) == @intFromPtr(s.ptr)) {
        c.ggml_abort("models/dflash.cpp", 309, "terminating due to uncaught exception of type std::invalid_argument: stoi: no conversion");
    }
    if (v < std.math.minInt(c_int) or v > std.math.maxInt(c_int)) {
        c.ggml_abort("models/dflash.cpp", 309, "terminating due to uncaught exception of type std::out_of_range: stoi: out of range");
    }
    return v;
}

extern "c" fn strtol(s: [*:0]const u8, end: *[*c]u8, base: c_int) c_long;

/// Ports `build_dspark_markov_head` (models/dflash.cpp:292 @5266f24da):
/// DSpark's Markov bias on the draft logits, chained through each block's
/// positions, and the confidence head beside it, which is optional.
fn buildDsparkMarkovHead(b: *const graph.Builder, model: *const Model, tokens: *Tensor) void {
    const ctx = b.ctx0;
    const res = b.res;

    const w1 = model.dspark_markov_w1;
    const w2 = model.dspark_markov_w2;
    impl.assert(w1 != null and w2 != null, "models/dflash.cpp", 298, "w1 && w2 && \"DSpark markov weights not loaded\"");

    // The confidence head is optional.
    const has_conf = model.dspark_conf_proj != null;

    // [n_vocab, n_tokens]
    const base = res.t_logits.?;
    const n_vocab = base.ne[0];
    const n_tok = base.ne[1];

    const it = model.gguf_kv.get("dflash.block_size");
    impl.assert(it != null, "models/dflash.cpp", 308, "it != model.gguf_kv.end() && \"DSpark draft requires 'dflash.block_size' in GGUF metadata\"");
    const block_size = stoi(it.?.*);
    impl.assert(block_size > 0, "models/dflash.cpp", 310, "block_size > 0");

    // The bonus anchor of SpecForge's exports: slot 0 is a bonus token, not
    // a prediction slot.
    const it_anchor = model.gguf_kv.get("dflash.sample_from_anchor");
    const sample_from_anchor = it_anchor == null or std.mem.eql(u8, it_anchor.?.*, "true");
    const i_draft_beg: i64 = if (sample_from_anchor) 0 else 1;

    const n_blocks: i64 = b.ubatch.n_seqs_unq;
    impl.assert(n_blocks > 0 and @rem(n_tok, n_blocks) == 0, "models/dflash.cpp", 318, "n_blocks > 0 && n_tok % n_blocks == 0 && \"DSpark markov head requires equal-size blocks\"");
    // The tokens per block in this ubatch at run time (the anchor and the
    // drafted positions), bounded by the training block_size.
    const block_drafts = @divTrunc(n_tok, n_blocks);
    if (block_drafts > block_size) {
        return;
    }

    // The anchor (the last committed) token of every block: token 0 of
    // each, a strided view.
    const token_stride: usize = @as(usize, @intCast(block_drafts)) * tokens.nb[0];
    const base_stride: usize = @as(usize, @intCast(block_drafts)) * base.nb[1];

    var prev: *Tensor = c.ggml_view_2d(ctx, tokens, 1, n_blocks, token_stride, 0);
    prev = c.ggml_cont_1d(ctx, prev, n_blocks);

    var cat: ?*Tensor = null;
    var cat_conf: ?*Tensor = null;

    if (!sample_from_anchor) {
        // The bonus anchor slot: its logits pass through unbiased, and the
        // (unread) confidence column is padding.
        cat = c.ggml_cont(ctx, c.ggml_view_2d(ctx, base, n_vocab, n_blocks, base_stride, 0));
        if (has_conf) {
            cat_conf = c.ggml_sigmoid(ctx, c.ggml_cont(ctx, c.ggml_view_2d(ctx, base, 1, n_blocks, base_stride, 0)));
        }
    }

    // Upstream's TODO: the in-graph chain is greedy (argmax); the sampling
    // parameters affect only the final token pick, not the Markov
    // conditioning path.
    var i = i_draft_beg;
    while (i < block_drafts) : (i += 1) {
        // [R, n_blocks]
        const w1_prev: *Tensor = c.ggml_get_rows(ctx, w1, prev);
        // [n_vocab_draft, n_blocks]
        var bias: *Tensor = b.buildLoraMm(w2.?, w1_prev, model.dspark_markov_w2_s);
        if (model.d2t) |d2t| {
            // A reduced draft vocabulary: the bias scattered to the target
            // rows (base is -inf on the others).
            const n_draft_vocab = bias.ne[0];
            const full: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, 1, n_vocab, n_blocks), 0.0);
            bias = c.ggml_set_rows(ctx, full, c.ggml_reshape_3d(ctx, bias, 1, n_draft_vocab, n_blocks), c.ggml_reshape_3d(ctx, d2t, n_draft_vocab, 1, 1));
            bias = c.ggml_reshape_2d(ctx, bias, n_vocab, n_blocks);
        }

        // Position i of every block, a strided view: [n_vocab, n_blocks].
        const base_i: *Tensor = c.ggml_view_2d(ctx, base, n_vocab, n_blocks, base_stride, @as(usize, @intCast(i)) * base.nb[1]);
        const col: *Tensor = c.ggml_add(ctx, base_i, bias);

        cat = if (cat) |x| c.ggml_concat(ctx, x, col, 1) else col;

        if (has_conf) {
            // The confidence head's input, which predicts each position's
            // acceptance: [n_embd, n_tok].
            const conf_inp = res.t_embd.?;
            // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b),
            // [1, n_blocks].
            const conf_inp_i: *Tensor = c.ggml_view_2d(ctx, conf_inp, conf_inp.ne[0], n_blocks, @as(usize, @intCast(block_drafts)) * conf_inp.nb[1], @as(usize, @intCast(i)) * conf_inp.nb[1]);
            const feat: *Tensor = c.ggml_concat(ctx, c.ggml_cont(ctx, conf_inp_i), w1_prev, 0);
            var conf: *Tensor = c.ggml_mul_mat(ctx, model.dspark_conf_proj, feat);
            if (model.dspark_conf_proj_b) |pb| {
                conf = c.ggml_add(ctx, conf, pb);
            }
            conf = c.ggml_sigmoid(ctx, conf);

            cat_conf = if (cat_conf) |x| c.ggml_concat(ctx, x, conf, 1) else conf;
        }

        if (i + 1 < block_drafts) {
            prev = c.ggml_argmax(ctx, col);
        }
    }

    // cat is position-major; the ubatch's block-major order is restored.
    var out: *Tensor = c.ggml_reshape_3d(ctx, cat.?, n_vocab, n_blocks, block_drafts);
    // [n_vocab, block_drafts, n_blocks]
    out = c.ggml_cont(ctx, c.ggml_permute(ctx, out, 0, 2, 1, 3));
    out = c.ggml_reshape_2d(ctx, out, n_vocab, n_tok);

    if (has_conf) {
        var conf: *Tensor = c.ggml_reshape_3d(ctx, cat_conf.?, 1, n_blocks, block_drafts);
        conf = c.ggml_cont(ctx, c.ggml_permute(ctx, conf, 0, 2, 1, 3));
        conf = c.ggml_reshape_2d(ctx, conf, 1, n_tok);

        // The [1, n_tok] confidences broadcast to n_embd-wide rows, so
        // llama_get_embeddings_nextn can carry them.
        conf = c.ggml_repeat(ctx, conf, res.t_embd);
        res.t_h_nextn = conf;
        c.ggml_build_forward_expand(b.gf, conf);
    }

    res.t_logits = out;
    c.ggml_build_forward_expand(b.gf, out);
}

/// Ports `build_dflash2_conv` (models/dflash.cpp:405 @5266f24da): DFlash2's
/// causal convolution within each block, its `kernel_size` taps weighted by
/// a static base plus the token's own dynamic coefficients, one weight per
/// group of `group_size` channels. `side` picks the input (0) or output (1)
/// half of `dynamic` and `base`.
fn buildDflash2Conv(b: *const graph.Builder, hidden_: *Tensor, dynamic_: ?*Tensor, base_: ?*Tensor, side: c_int) *Tensor {
    const hp = b.hparams;
    var hidden = hidden_;
    const hidden_size: i64 = hidden.ne[0];
    const n_tokens: i64 = hidden.ne[1];
    const n_blocks: i64 = b.ubatch.n_seqs_unq;
    const kernel_size: i64 = hp.dflash_conv_kernel_size;
    const group_size: i64 = hp.dflash_conv_group_size;
    const n_groups = @divTrunc(hidden_size, group_size);

    impl.assert(n_blocks > 0 and @rem(n_tokens, n_blocks) == 0, "models/dflash.cpp", 419, "n_blocks > 0 && n_tokens % n_blocks == 0");
    impl.assert(dynamic_ != null and base_ != null and side >= 0 and side < 2, "models/dflash.cpp", 420, "dynamic && base && side >= 0 && side < 2");
    var dynamic = dynamic_.?;
    const base = base_.?;

    const block_size = @divTrunc(n_tokens, n_blocks);
    const ctx = b.ctx0;
    // ggml_cont copies even a tensor that is already contiguous.
    if (!c.ggml_is_contiguous(hidden) or hidden.ne[1] != n_tokens) {
        hidden = c.ggml_cont_2d(ctx, hidden, hidden_size, n_tokens);
    }
    if (!c.ggml_is_contiguous(dynamic) or dynamic.ne[1] != n_tokens) {
        dynamic = c.ggml_cont_2d(ctx, dynamic, dynamic.ne[0], n_tokens);
    }
    const blocks: *Tensor = c.ggml_reshape_3d(ctx, hidden, hidden_size, block_size, n_blocks);
    const coeffs: *Tensor = c.ggml_reshape_4d(ctx, dynamic, n_groups, kernel_size, 2, n_tokens);
    const side_u: usize = @intCast(side);
    const coeffs_side: *Tensor = c.ggml_view_3d(ctx, coeffs, n_groups, kernel_size, n_tokens, coeffs.nb[1], coeffs.nb[3], side_u * coeffs.nb[2]);

    var coeff_all: *Tensor = c.ggml_cont(ctx, coeffs_side);
    coeff_all = c.ggml_reshape_4d(ctx, coeff_all, 1, n_groups, kernel_size, n_tokens);
    coeff_all = c.ggml_repeat_4d(ctx, coeff_all, group_size, n_groups, kernel_size, n_tokens);

    const base_side: *Tensor = c.ggml_reshape_4d(ctx, c.ggml_view_1d(ctx, base, hidden_size * kernel_size, side_u * base.nb[2]), group_size, n_groups, kernel_size, 1);

    const weight_all: *Tensor = c.ggml_add(ctx, coeff_all, base_side);

    var result: ?*Tensor = null;
    var tap: i64 = 0;
    while (tap < kernel_size) : (tap += 1) {
        var values: *Tensor = blocks;
        if (tap > 0) {
            const zeros: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_3d(ctx, hidden.type, hidden_size, @min(tap, block_size), n_blocks), 0.0);
            if (tap < block_size) {
                const previous: *Tensor = c.ggml_view_3d(ctx, blocks, hidden_size, block_size - tap, n_blocks, blocks.nb[1], blocks.nb[2], 0);
                values = c.ggml_concat(ctx, zeros, previous, 1);
            } else {
                values = zeros;
            }
        }
        values = c.ggml_reshape_2d(ctx, values, hidden_size, n_tokens);

        const weight: *Tensor = c.ggml_reshape_2d(ctx, c.ggml_cont(ctx, c.ggml_view_4d(ctx, weight_all, group_size, n_groups, 1, n_tokens, weight_all.nb[1], weight_all.nb[2], weight_all.nb[3], @as(usize, @intCast(tap)) * weight_all.nb[2])), hidden_size, n_tokens);

        const term: *Tensor = c.ggml_mul(ctx, weight, values);
        result = if (result) |r| c.ggml_add(ctx, r, term) else term;
    }
    // A zero-tap kernel returns null in the C++; the loader refuses one.
    return result.?;
}

/// Ports `build_dflash2_selector` (models/dflash.cpp:475 @5266f24da):
/// DFlash2's selector, the top-k candidates at every block position and
/// the pairwise transition scores between neighbouring positions' sets,
/// packed into the nextn output slot for the walk on the CPU.
fn buildDflash2Selector(b: *const graph.Builder, model: *const Model, tokens_: ?*Tensor) void {
    const ctx = b.ctx0;
    const res = b.res;

    const hp = b.hparams;
    const n_tokens: i64 = b.n_tokens;
    const n_embd: i64 = b.n_embd;

    const top_k: i64 = hp.dflash_selector_top_k;
    const rank: i64 = hp.dflash_selector_rank;
    const n_blocks: i64 = b.ubatch.n_seqs_unq;
    impl.assert(n_blocks > 0 and @rem(n_tokens, n_blocks) == 0, "models/dflash.cpp", 486, "n_blocks > 0 && n_tokens % n_blocks == 0");
    impl.assert(res.t_logits.?.ne[1] == n_tokens, "models/dflash.cpp", 487, "res->t_logits->ne[1] == n_tokens");
    const tokens = tokens_ orelse return;

    const t_logits = res.t_logits.?;
    const tokens_per_block = @divTrunc(n_tokens, n_blocks);
    const block_size: i64 = @min(tokens_per_block, @as(i64, hp.dflash_block_size));
    const row_used = top_k + top_k * top_k;

    const candidates: *Tensor = c.ggml_top_k(ctx, t_logits, @intCast(top_k));
    const logits_rows: *Tensor = c.ggml_reshape_3d(ctx, t_logits, 1, t_logits.ne[0], n_tokens);
    const unary: *Tensor = c.ggml_reshape_2d(ctx, c.ggml_get_rows(ctx, logits_rows, candidates), top_k, n_tokens);
    const gate = b.buildLoraMm(model.dflash_selector_hidden.?, res.t_embd.?, null);

    // Everything below indexes [.., tokens_per_block, n_blocks]: the block
    // position varies fastest, the sequences are the outer dimension.
    const sel: Selector = .{
        .b = b,
        .model = model,
        .top_k = top_k,
        .rank = rank,
        .n_blocks = n_blocks,
        .n_embd = n_embd,
        .row_used = row_used,
        .cand_blk = c.ggml_reshape_3d(ctx, candidates, top_k, tokens_per_block, n_blocks),
        .unary_blk = c.ggml_reshape_3d(ctx, unary, top_k, tokens_per_block, n_blocks),
        .gate_blk = c.ggml_reshape_3d(ctx, gate, rank, tokens_per_block, n_blocks),
    };

    var packed_: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, n_embd, 1, n_blocks), 0.0);

    if (block_size > 1) {
        // Position 1 alone: its predecessor is the anchor token, one id per
        // sequence rather than a candidate set.
        const anchor_ids: *Tensor = c.ggml_cont_1d(ctx, c.ggml_view_2d(ctx, tokens, 1, n_blocks, @as(usize, @intCast(tokens_per_block)) * tokens.nb[0], 0), n_blocks);
        packed_ = c.ggml_concat(ctx, packed_, sel.scoreRun(1, 1, anchor_ids), 1);
    }
    if (block_size > 2) {
        const cand_blk = sel.cand_blk;
        const prev_ids: *Tensor = c.ggml_reshape_1d(ctx, c.ggml_cont(ctx, c.ggml_view_3d(ctx, cand_blk, top_k, block_size - 2, n_blocks, cand_blk.nb[1], cand_blk.nb[2], cand_blk.nb[1])), top_k * (block_size - 2) * n_blocks);
        packed_ = c.ggml_concat(ctx, packed_, sel.scoreRun(2, block_size - 2, prev_ids), 1);
    }

    packed_ = c.ggml_reshape_2d(ctx, packed_, n_embd, block_size * n_blocks);
    b.cb(packed_, "dflash2_lattice", -1);
    res.t_h_nextn = packed_;
    c.ggml_build_forward_expand(b.gf, packed_);
}

/// What `build_dflash2_selector`'s `score_run` lambda captures.
const Selector = struct {
    b: *const graph.Builder,
    model: *const Model,
    top_k: i64,
    rank: i64,
    n_blocks: i64,
    n_embd: i64,
    row_used: i64,
    cand_blk: *Tensor,
    unary_blk: *Tensor,
    gate_blk: *Tensor,

    /// Ports `build_dflash2_selector`'s `score_run` (models/dflash.cpp:510
    /// @5266f24da): a position's score reads only the candidate sets at
    /// pos - 1 and pos, so a run of positions has no internal dependency
    /// and scores in one batched matmul.
    fn scoreRun(self: *const Selector, beg_pos: i64, n_pos: i64, pred_ids: *Tensor) *Tensor {
        const ctx = self.b.ctx0;
        const top_k = self.top_k;
        const rank = self.rank;
        const n_blocks = self.n_blocks;
        const cand_blk = self.cand_blk;
        const unary_blk = self.unary_blk;
        const gate_blk = self.gate_blk;
        const beg: usize = @intCast(beg_pos);

        const cand_run: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, cand_blk, top_k, n_pos, n_blocks, cand_blk.nb[1], cand_blk.nb[2], beg * cand_blk.nb[1]));
        const unary_run: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, unary_blk, top_k, n_pos, n_blocks, unary_blk.nb[1], unary_blk.nb[2], beg * unary_blk.nb[1]));
        const gate_run: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, gate_blk, rank, n_pos, n_blocks, gate_blk.nb[1], gate_blk.nb[2], beg * gate_blk.nb[1]));

        const n_pred = @divTrunc(pred_ids.ne[0], n_pos * n_blocks);

        const successor: *Tensor = c.ggml_reshape_4d(ctx, c.ggml_get_rows(ctx, self.model.dflash_selector_next, c.ggml_reshape_1d(ctx, cand_run, top_k * n_pos * n_blocks)), rank, top_k, n_pos, n_blocks);
        const predecessor: *Tensor = c.ggml_reshape_4d(ctx, c.ggml_get_rows(ctx, self.model.dflash_selector_prev, pred_ids), rank, n_pred, n_pos, n_blocks);

        const gate_bcast: *Tensor = c.ggml_reshape_4d(ctx, gate_run, rank, 1, n_pos, n_blocks);
        const cond: *Tensor = c.ggml_mul(ctx, predecessor, c.ggml_repeat(ctx, gate_bcast, predecessor));
        var score: *Tensor = c.ggml_mul_mat(ctx, successor, cond);
        if (n_pred == 1) {
            score = c.ggml_repeat_4d(ctx, score, top_k, top_k, n_pos, n_blocks);
        }
        const unary_bcast: *Tensor = c.ggml_reshape_4d(ctx, unary_run, top_k, 1, n_pos, n_blocks);
        score = c.ggml_add(ctx, score, c.ggml_repeat(ctx, unary_bcast, score));

        const row: *Tensor = c.ggml_concat(ctx, c.ggml_cast(ctx, cand_run, c.GGML_TYPE_F32), c.ggml_reshape_3d(ctx, score, top_k * top_k, n_pos, n_blocks), 0);
        return c.ggml_pad(ctx, row, @intCast(self.n_embd - self.row_used), 0, 0, 0);
    }
};

/// `graph<false>`'s `build_rope` lambda (models/dflash.cpp:595 @5266f24da):
/// drafts for M-RoPE targets use the degenerate sections, the temporal
/// dimension only.
fn buildRope(b: *const graph.Builder, sections: *[4]c_int, cur: *Tensor, pos: *Tensor) *Tensor {
    const ctx = b.ctx0;
    return if (b.rope_type == c.GGML_ROPE_TYPE_MROPE)
        c.ggml_rope_multi(ctx, cur, pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow)
    else
        c.ggml_rope_ext(ctx, cur, pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
}

/// Ports the constructor of `llama_model_dflash`'s `graph<false>`
/// (models/dflash.cpp:570 @5266f24da): the decoder, in one of two modes by
/// the batch's type. An embedding batch is fused target features, projected
/// and written into the K/V cache; a token batch is a noise block, diffused
/// by attending over the committed tokens and the block's masks to draft
/// tokens. `iswa` is the C++'s run-time `use_iswa`.
fn buildGraphDec(comptime iswa: bool, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_inp: i64 = hp.nEmbdInpEnc();
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/dflash.cpp", 574, "n_embd_head == hparams.n_embd_head_k()");

    const inp_pos = b.buildInpPos();

    // Optional iSWA: the matching attention input.
    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    // Drafts for M-RoPE targets use degenerate sections (the temporal
    // dimension only).
    var sections: [4]c_int = undefined;
    @memcpy(&sections, hp.rope_sections[0..4]);

    // KV cache injection.
    if (b.ubatch.embd != null) {
        const inp = allocator.create(graph.InpEmbd) catch oom();
        inp.* = .{ .n_embd = n_embd_inp };

        inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd_inp, b.n_tokens);
        c.ggml_set_input(inp.embd);

        const inp_target = inp.embd.?;
        b.cb(inp_target, "inp_target_features", -1);

        _ = b.res.addInput(graph.InpEmbd, inp);

        // The target features fused through the encoder.
        var inp_g = b.buildLoraMm(model.fc.?, inp_target, model.fc_s);
        inp_g = b.buildNorm(inp_g, model.output_norm_enc, null, .rms, -1);
        b.cb(inp_g, "inp_g_embeddings", -1);

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];

            var Kcur = b.buildLoraMm(layer.wk.?, inp_g, layer.wk_s);
            var Vcur = b.buildLoraMm(layer.wv.?, inp_g, layer.wv_s);

            Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
            Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            Kcur = buildRope(b, &sections, Kcur, inp_pos);
            b.cb(Kcur, "Kcur_injected", il);
            b.cb(Vcur, "Vcur_injected", il);

            if (iswa) {
                // Each layer's K and V go to its sub-cache: SWA layers to
                // the sliding one, full layers to the dense one.
                const is_swa = hp.isSwa(@intCast(il));
                const kv = if (is_swa) inp_attn.mctx.getSwa() else inp_attn.mctx.getBase();
                const k_idxs = if (is_swa) inp_attn.getKIdxsSwa() else inp_attn.getKIdxs();
                const v_idxs = if (is_swa) inp_attn.getVIdxsSwa() else inp_attn.getVIdxs();
                // K and V rotated into the cache's rotated space.
                const k_rot = if (is_swa) inp_attn.self_k_rot_swa else inp_attn.self_k_rot;
                const v_rot = if (is_swa) inp_attn.self_v_rot_swa else inp_attn.self_v_rot;
                if (k_rot) |r| {
                    Kcur = graph.mulMatHadamard(ctx, Kcur, r);
                }
                if (v_rot) |r| {
                    Vcur = graph.mulMatHadamard(ctx, Vcur, r);
                }
                c.ggml_build_forward_expand(b.gf, kv.cpyK(ctx, Kcur, k_idxs, il));
                c.ggml_build_forward_expand(b.gf, kv.cpyV(ctx, Vcur, v_idxs, il));
            } else {
                // K and V rotated into the cache's rotated space.
                if (inp_attn.self_k_rot) |r| {
                    Kcur = graph.mulMatHadamard(ctx, Kcur, r);
                }
                if (inp_attn.self_v_rot) |r| {
                    Vcur = graph.mulMatHadamard(ctx, Vcur, r);
                }
                c.ggml_build_forward_expand(b.gf, inp_attn.mctx.cpyK(ctx, Kcur, inp_attn.getKIdxs(), il));
                c.ggml_build_forward_expand(b.gf, inp_attn.mctx.cpyV(ctx, Vcur, inp_attn.getVIdxs(), il));
            }
        }

        b.res.t_embd = inp_g;

        c.ggml_build_forward_expand(b.gf, inp_g);
        return;
    }

    // The token embeddings, the target model's when the draft has none
    // (shared through ctx_other).
    var tok_embd = model.tok_embd;
    if (tok_embd == null) {
        impl.assert(b.cparams.ctx_other != null, "models/dflash.cpp", 675, "cparams.ctx_other != nullptr");
        const model_other = otherModel(b);

        impl.assert(model_other.tok_embd != null, "models/dflash.cpp", 678, "model_other->tok_embd != nullptr && \"DFlash decoder requires the target model's token embeddings\"");
        tok_embd = model_other.tok_embd;
    }

    const inp = allocator.create(graph.InpEmbd) catch oom();
    inp.* = .{ .n_embd = b.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);
    b.res.t_inp_tokens = inp.tokens;

    const inp_tokens = inp.tokens.?;

    var inpL: *Tensor = c.ggml_get_rows(ctx, tok_embd, inp.tokens);
    b.cb(inpL, "inp_noise_embd", -1);

    _ = b.res.addInput(graph.InpEmbd, inp);

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        var noise_norm = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(noise_norm, "noise_norm", il);

        var attn_dynamic: ?*Tensor = null;
        if (layer.dflash_attn_conv_proj) |proj| {
            attn_dynamic = b.buildLoraMm(proj, noise_norm, null);
            noise_norm = buildDflash2Conv(b, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            b.cb(noise_norm, "attn_conv_in", il);
        }

        var Qcur = b.buildLoraMm(layer.wq.?, noise_norm, layer.wq_s);
        var Kcur = b.buildLoraMm(layer.wk.?, noise_norm, layer.wk_s);
        var Vcur = b.buildLoraMm(layer.wv.?, noise_norm, layer.wv_s);

        Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
        Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
        Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);

        Qcur = buildRope(b, &sections, Qcur, inp_pos);
        Kcur = buildRope(b, &sections, Kcur, inp_pos);
        b.cb(Qcur, "Qcur", il);
        b.cb(Kcur, "Kcur", il);
        b.cb(Vcur, "Vcur", il);

        // Cache-aware, non-causal attention.
        var cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, layer.attn_sinks, null, kq_scale, il);

        if (attn_dynamic != null) {
            cur = buildDflash2Conv(b, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            b.cb(cur, "attn_conv_out", il);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        var ffn_dynamic: ?*Tensor = null;
        if (layer.dflash_ffn_conv_proj) |proj| {
            ffn_dynamic = b.buildLoraMm(proj, cur, null);
            cur = buildDflash2Conv(b, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            b.cb(cur, "ffn_conv_in", il);
        }

        cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        if (ffn_dynamic != null) {
            cur = buildDflash2Conv(b, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            b.cb(cur, "ffn_conv_out", il);
        }

        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    b.res.t_embd = cur;

    // The LM head, the target model's when the draft has none (shared
    // through ctx_other).
    var output = model.output;
    var output_s = model.output_s;
    if (output == null) {
        impl.assert(b.cparams.ctx_other != null, "models/dflash.cpp", 776, "cparams.ctx_other != nullptr");
        const model_other = otherModel(b);
        impl.assert(model_other.output != null, "models/dflash.cpp", 778, "model_other->output != nullptr && \"DFlash decoder requires the target model's output projection\"");
        output = model_other.output;
        output_s = model_other.output_s;
    }

    cur = b.buildLoraMm(output.?, cur, output_s);

    // DFlash2 feeds these logits to the selector, so they need the target's
    // output transforms; DFlash1 and DSpark read them through the sampler
    // instead.
    if (model.dflash_selector_hidden != null) {
        if (hp.f_logit_scale != 0.0) {
            cur = c.ggml_scale(ctx, cur, hp.f_logit_scale);
        }
        if (hp.f_final_logit_softcapping > 0.0) {
            cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
            cur = c.ggml_tanh(ctx, cur);
            cur = c.ggml_scale(ctx, cur, hp.f_final_logit_softcapping);
        }
    }

    // Reduced-draft-vocabulary exports: the draft logits scattered to the
    // target vocabulary through d2t.
    if (model.d2t) |d2t| {
        const n_draft_vocab = cur.ne[0];
        const n_outputs = cur.ne[1];
        const n_vocab: i64 = model.vocab.nTokens();

        impl.assert(d2t.type == c.GGML_TYPE_I64, "models/dflash.cpp", 804, "model.d2t->type == GGML_TYPE_I64");
        impl.assert(d2t.ne[0] == n_draft_vocab, "models/dflash.cpp", 805, "model.d2t->ne[0] == n_draft_vocab");

        const logits: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, 1, n_vocab, n_outputs), -std.math.inf(f32));
        cur = c.ggml_set_rows(ctx, logits, c.ggml_reshape_3d(ctx, cur, 1, n_draft_vocab, n_outputs), c.ggml_reshape_3d(ctx, d2t, n_draft_vocab, 1, 1));
        cur = c.ggml_reshape_2d(ctx, cur, n_vocab, n_outputs);
    }
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);

    // DSpark: the draft logits biased by the Markov head.
    if (model.dspark_markov_w1 != null) {
        buildDsparkMarkovHead(b, model, inp_tokens);
    }

    if (model.dflash_selector_hidden != null) {
        buildDflash2Selector(b, model, inp_tokens);
    }
}

/// `cparams.ctx_other`'s model, or the draft's own tensor when it has one:
/// the token embeddings for `graph_dsv4`.
fn dsv4TokEmbd(model: *const Model, b: *const graph.Builder) *Tensor {
    if (model.tok_embd) |t| return t;
    impl.assert(b.cparams.ctx_other != null, "models/dflash.cpp", 888, "cparams.ctx_other != nullptr");
    const model_other = otherModel(b);

    impl.assert(model_other.tok_embd != null, "models/dflash.cpp", 891, "model_other->tok_embd != nullptr && \"DSpark decoder requires the target model's token embeddings\"");
    return model_other.tok_embd.?;
}

/// Ports the constructor of `llama_model_dflash`'s `graph_dsv4`
/// (models/dflash.cpp:831 @5266f24da): the DeepSeek-V4 DSpark decoder, in
/// the same two modes as the DFlash one. An embedding batch is the
/// encoder's fused target features: each stage projects them through its
/// `wkv` and writes K into the draft's ring cache. A token batch is a noise
/// block through the full DeepSeek-V4 stages -- hyper-connections, MLA
/// attention, a mixture of experts -- then the Markov and confidence heads.
/// It derives from DeepSeek-V4's graph without building that graph, using
/// only its builders.
fn buildGraphDsv4(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_inp: i64 = hp.nEmbdInpEnc();
    const n_embd_head: i64 = hp.nEmbdHeadK(0);
    const n_embd_head_rope: i64 = hp.nRot(0);
    const n_embd_head_nope = n_embd_head - n_embd_head_rope;

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKIswa();

    // KV cache injection: the fused target features from the encoder.
    if (b.ubatch.embd != null) {
        const inp = allocator.create(graph.InpEmbd) catch oom();
        inp.* = .{ .n_embd = n_embd_inp };

        inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd_inp, b.n_tokens);
        c.ggml_set_input(inp.embd);

        const inp_target = inp.embd.?;
        b.cb(inp_target, "inp_target_features", -1);

        _ = b.res.addInput(graph.InpEmbd, inp);

        // The target features fused through the encoder.
        var inp_g = b.buildLoraMm(model.fc.?, inp_target, model.fc_s);
        inp_g = b.buildNorm(inp_g, model.output_norm_enc, null, .rms, -1);
        b.cb(inp_g, "inp_g_embeddings", -1);

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];

            // The main track's KV, kv_norm(wkv(main_x)), RoPE'd on its
            // trailing dimensions with the same parameters as the
            // uncompressed layers in build_attention_impl.
            var kv = b.buildLoraMm(layer.wkv.?, inp_g, null);
            kv = b.buildNorm(kv, layer.attn_kv_norm, null, .rms, il);
            kv = c.ggml_reshape_3d(ctx, kv, n_embd_head, 1, b.n_tokens);

            kv = c.ggml_rope_ext(ctx, kv, inp_pos, null, @intCast(n_embd_head_rope), b.rope_type, 0, b.freq_base, 1.0, 0.0, 1.0, 0.0, 0.0);
            kv = c.ggml_rope_set_offset(kv, @intCast(n_embd_head_nope));
            b.cb(kv, "kv_injected", il);

            if (inp_attn.self_k_rot_swa) |r| {
                kv = graph.mulMatHadamard(ctx, kv, r);
            }
            c.ggml_build_forward_expand(b.gf, inp_attn.mctx.getSwa().cpyK(ctx, kv, inp_attn.getKIdxsSwa(), il));
        }

        b.res.t_embd = inp_g;

        c.ggml_build_forward_expand(b.gf, inp_g);
        return;
    }

    // The token embeddings, the target model's when the draft has none
    // (shared through ctx_other).
    const tok_embd = dsv4TokEmbd(model, b);

    const inp = allocator.create(graph.InpEmbd) catch oom();
    inp.* = .{ .n_embd = b.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    const inp_tokens = inp.tokens.?;

    var inpL: *Tensor = c.ggml_get_rows(ctx, tok_embd, inp.tokens);
    b.cb(inpL, "inp_noise_embd", -1);

    _ = b.res.addInput(graph.InpEmbd, inp);

    const hc: i64 = hp.dsv4_hc_mult;
    inpL = c.ggml_reshape_3d(ctx, inpL, b.n_embd, 1, b.n_tokens);
    inpL = c.ggml_repeat_4d(ctx, inpL, b.n_embd, hc, b.n_tokens, 1);
    b.cb(inpL, "hc_init", -1);

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        var residual = inpL;
        var post: ?*Tensor = null;
        var comb: ?*Tensor = null;

        var cur = deepseek4.buildHcPre(b, inpL, layer.hc_attn_fn.?, layer.hc_attn_scale.?, layer.hc_attn_base.?, &post, &comb, il);
        b.cb(cur, "hc_attn_pre", il);

        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        cur = deepseek4.buildAttention(b, model, .{ .mtp = inp_attn }, cur, inp_pos, il);

        inpL = deepseek4.buildHcPost(b, cur, residual, post.?, comb.?, il);
        b.cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = deepseek4.buildHcPre(b, inpL, layer.hc_ffn_fn.?, layer.hc_ffn_scale.?, layer.hc_ffn_base.?, &post, &comb, il);
        b.cb(cur, "hc_ffn_pre", il);

        cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        const moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = layer.ffn_exp_probs_b,
            .n_expert = b.n_expert,
            .n_expert_used = hp.nExpertUsed(0),
            .type_op = .silu,
            .norm_w = hp.expert_weights_norm,
            .w_scale = hp.expert_weights_scale,
            .gating_op = @fromBackingInt(hp.expert_gating_func),
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
        b.cb(ffn_shexp, "ffn_shexp", il);

        cur = c.ggml_add(ctx, moe_out, ffn_shexp);
        b.cb(cur, "ffn_out", il);

        inpL = deepseek4.buildHcPost(b, cur, residual, post.?, comb.?, il);
        b.cb(inpL, "l_out", il);
    }

    var cur = deepseek4.buildHcHead(b, inpL, model.hc_head_fn.?, model.hc_head_scale.?, model.hc_head_base.?);
    b.cb(cur, "hc_head", -1);

    // The confidence head's input: the reference scores the collapsed
    // hidden state before the norm.
    b.res.t_embd = cur;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    // The LM head, the target model's when the draft has none (shared
    // through ctx_other).
    var output = model.output;
    var output_s = model.output_s;
    if (output == null) {
        impl.assert(b.cparams.ctx_other != null, "models/dflash.cpp", 985, "cparams.ctx_other != nullptr");
        const model_other = otherModel(b);
        impl.assert(model_other.output != null, "models/dflash.cpp", 987, "model_other->output != nullptr && \"DSpark decoder requires the target model's output projection\"");
        output = model_other.output;
        output_s = model_other.output_s;
    }

    cur = b.buildLoraMm(output.?, cur, output_s);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);

    if (model.dspark_markov_w1 != null) {
        buildDsparkMarkovHead(b, model, inp_tokens);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
