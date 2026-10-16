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
//! Ported from `llama.cpp/src/models/dflash.cpp` at v0.3.0 (`c1d0e7a00`):
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

/// Ports `llama_model_dflash`'s `load_arch_hparams` (models/dflash.cpp:7 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

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
        hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
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
        impl.assert(hp.dsv4_o_group_count > 0, "models/dflash.cpp", 46, "hparams.dsv4_o_group_count > 0");

        if (hp.expert_gating_func != @backingInt(hparams_mod.ExpertGatingFunc.sqrt_softplus)) {
            return exc.throw(.runtime_error, "DSpark DSV4 draft expects sqrtsoftplus MoE scoring", .{});
        }
        var il: u32 = 0;
        while (il < hp.n_layer_all) : (il += 1) {
            if (hp.dsv4_compress_ratios[il] != 0) {
                return exc.throw(.runtime_error, "DSpark DSV4 draft expects uncompressed attention on all stages", .{});
            }
        }

        impl.assert(hp.n_swa > 0, "models/dflash.cpp", 57, "hparams.n_swa > 0");
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

/// Ports `llama_model_dflash`'s `load_arch_tensors` (models/dflash.cpp:82 @c1d0e7a00).
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

        self.dspark_conf_proj = try self.createTensor(.{ .tensor = .dspark_conf_proj, .suffix = "weight" }, &.{ l.n_embd + dspark_markov_rank, 1 }, 0, exc);
        self.dspark_conf_proj_b = try self.createTensor(.{ .tensor = .dspark_conf_proj, .suffix = "bias" }, &.{1}, nr, exc);

        impl.logInfo("%s: DFlash with DSpark markov head (rank = %lld)\n", .{ func, @as(c_longlong, dspark_markov_rank) });
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
        const n_ff_exp: i64 = hp.n_ff_exp;
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

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_dflash`'s `build_arch_graph` (models/dflash.cpp:194 @c1d0e7a00).
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
        else => impl.abort("models/dflash.cpp", 205, "invalid graph type"),
    }
}

/// `llama_get_model(cparams.ctx_other)`: the target model a draft shares
/// weights with.
fn otherModel(b: *const graph.Builder) *const Model {
    const other: *const context_mod.Context = @ptrCast(@alignCast(b.cparams.ctx_other.?));
    return other.getModel();
}

/// Ports `llama_model_dflash`'s `graph<true>`'s `build_inp_embd_enc`
/// (models/dflash.cpp:210 @c1d0e7a00): the target model's features, its
/// chosen layers concatenated, given as the batch's embeddings.
fn buildInpEmbdEnc(b: *const graph.Builder) *Tensor {
    const hp = b.hparams;

    const inp_target = allocator.create(graph.InpEmbd) catch oom();
    inp_target.* = .{ .n_embd = hp.nEmbdInpEnc() };

    inp_target.embd = c.ggml_new_tensor_2d(b.ctx0, c.GGML_TYPE_F32, hp.nEmbdInpEnc(), b.n_tokens);
    c.ggml_set_input(inp_target.embd);

    const cur = inp_target.embd.?;
    b.cb(cur, "inp_embd", -1);

    _ = b.res.addInput(graph.InpEmbd, inp_target);

    return cur;
}

/// Ports the constructor of `llama_model_dflash`'s `graph<true>`
/// (models/dflash.cpp:226 @c1d0e7a00): the encoder, the target model's
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
        c.ggml_abort("models/dflash.cpp", 256, "terminating due to uncaught exception of type std::invalid_argument: stoi: no conversion");
    }
    if (v < std.math.minInt(c_int) or v > std.math.maxInt(c_int)) {
        c.ggml_abort("models/dflash.cpp", 256, "terminating due to uncaught exception of type std::out_of_range: stoi: out of range");
    }
    return v;
}

extern "c" fn strtol(s: [*:0]const u8, end: *[*c]u8, base: c_int) c_long;

/// Ports `build_dspark_markov_head` (models/dflash.cpp:242 @c1d0e7a00):
/// DSpark's Markov bias on the draft logits, chained through each block's
/// positions, and the confidence head beside it.
fn buildDsparkMarkovHead(b: *const graph.Builder, model: *const Model, tokens: *Tensor) void {
    const ctx = b.ctx0;
    const res = b.res;

    const w1 = model.dspark_markov_w1;
    const w2 = model.dspark_markov_w2;
    impl.assert(w1 != null and w2 != null and model.dspark_conf_proj != null, "models/dflash.cpp", 248, "w1 && w2 && model.dspark_conf_proj && \"DSpark markov/confidence weights not loaded\"");

    // [n_vocab, n_tokens]
    const base = res.t_logits.?;
    const n_vocab = base.ne[0];
    const n_tok = base.ne[1];

    const it = model.gguf_kv.get("dflash.block_size");
    impl.assert(it != null, "models/dflash.cpp", 255, "it != model.gguf_kv.end() && \"DSpark draft requires 'dflash.block_size' in GGUF metadata\"");
    const block_size = stoi(it.?.*);
    impl.assert(block_size > 0, "models/dflash.cpp", 257, "block_size > 0");

    // The bonus anchor of SpecForge's exports: slot 0 is a bonus token, not
    // a prediction slot.
    const it_anchor = model.gguf_kv.get("dflash.sample_from_anchor");
    const sample_from_anchor = it_anchor == null or std.mem.eql(u8, it_anchor.?.*, "true");
    const i_draft_beg: i64 = if (sample_from_anchor) 0 else 1;

    const n_blocks: i64 = b.ubatch.n_seqs_unq;
    impl.assert(n_blocks > 0 and @rem(n_tok, n_blocks) == 0, "models/dflash.cpp", 265, "n_blocks > 0 && n_tok % n_blocks == 0 && \"DSpark markov head requires equal-size blocks\"");
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

    // The confidence head's input, which predicts each position's
    // acceptance: [n_embd, n_tok].
    const conf_inp = res.t_embd.?;

    var cat: ?*Tensor = null;
    var cat_conf: ?*Tensor = null;

    if (!sample_from_anchor) {
        // The bonus anchor slot: its logits pass through unbiased, and the
        // (unread) confidence column is padding.
        cat = c.ggml_cont(ctx, c.ggml_view_2d(ctx, base, n_vocab, n_blocks, base_stride, 0));
        cat_conf = c.ggml_sigmoid(ctx, c.ggml_cont(ctx, c.ggml_view_2d(ctx, base, 1, n_blocks, base_stride, 0)));
    }

    // Upstream's TODO: the in-graph chain is greedy (argmax); the sampling
    // parameters affect only the final token pick, not the Markov
    // conditioning path.
    var i = i_draft_beg;
    while (i < block_drafts) : (i += 1) {
        // [R, n_blocks]
        const w1_prev: *Tensor = c.ggml_get_rows(ctx, w1, prev);
        // [n_vocab_draft, n_blocks]
        var bias: *Tensor = c.ggml_mul_mat(ctx, w2, w1_prev);
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

        if (i + 1 < block_drafts) {
            prev = c.ggml_argmax(ctx, col);
        }
    }

    // cat is position-major; the ubatch's block-major order is restored.
    var out: *Tensor = c.ggml_reshape_3d(ctx, cat.?, n_vocab, n_blocks, block_drafts);
    // [n_vocab, block_drafts, n_blocks]
    out = c.ggml_cont(ctx, c.ggml_permute(ctx, out, 0, 2, 1, 3));
    out = c.ggml_reshape_2d(ctx, out, n_vocab, n_tok);

    {
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

/// Ports the constructor of `llama_model_dflash`'s `graph<false>`
/// (models/dflash.cpp:353 @c1d0e7a00): the decoder, in one of two modes by
/// the batch's type. An embedding batch is fused target features, projected
/// and written into the K/V cache; a token batch is a noise block, diffused
/// by attending over the committed tokens and the block's masks to draft
/// tokens. `iswa` is the C++'s run-time `use_iswa`.
fn buildGraphDec(comptime iswa: bool, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/dflash.cpp", 356, "n_embd_head == hparams.n_embd_head_k()");

    const inp_pos = b.buildInpPos();

    // Optional iSWA: the matching attention input.
    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

    // KV cache injection.
    if (b.ubatch.embd != null) {
        const inp = allocator.create(graph.InpEmbd) catch oom();
        inp.* = .{ .n_embd = b.n_embd };

        inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, b.n_embd, b.n_tokens);
        c.ggml_set_input(inp.embd);

        const inp_g = inp.embd.?;
        b.cb(inp_g, "inp_g_embeddings", -1);

        _ = b.res.addInput(graph.InpEmbd, inp);

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];

            var Kcur = b.buildLoraMm(layer.wk.?, inp_g, null);
            var Vcur = b.buildLoraMm(layer.wv.?, inp_g, null);

            Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
            Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
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
        impl.assert(b.cparams.ctx_other != null, "models/dflash.cpp", 442, "cparams.ctx_other != nullptr");
        const model_other = otherModel(b);

        impl.assert(model_other.tok_embd != null, "models/dflash.cpp", 445, "model_other->tok_embd != nullptr && \"DFlash decoder requires the target model's token embeddings\"");
        tok_embd = model_other.tok_embd;
    }

    const inp = allocator.create(graph.InpEmbd) catch oom();
    inp.* = .{ .n_embd = b.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    const inp_tokens = inp.tokens.?;

    var inpL: *Tensor = c.ggml_get_rows(ctx, tok_embd, inp.tokens);
    b.cb(inpL, "inp_noise_embd", -1);

    _ = b.res.addInput(graph.InpEmbd, inp);

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        const noise_norm = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(noise_norm, "noise_norm", il);

        var Qcur = b.buildLoraMm(layer.wq.?, noise_norm, null);
        var Kcur = b.buildLoraMm(layer.wk.?, noise_norm, null);
        var Vcur = b.buildLoraMm(layer.wv.?, noise_norm, null);

        Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
        Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
        Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, b.n_head_kv, b.n_tokens);

        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);

        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        b.cb(Qcur, "Qcur", il);
        b.cb(Kcur, "Kcur", il);
        b.cb(Vcur, "Vcur", il);

        // Cache-aware, non-causal attention.
        var cur = b.buildAttn(inp_attn, layer.wo, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

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
        impl.assert(b.cparams.ctx_other != null, "models/dflash.cpp", 526, "cparams.ctx_other != nullptr");
        const model_other = otherModel(b);
        impl.assert(model_other.output != null, "models/dflash.cpp", 528, "model_other->output != nullptr && \"DFlash decoder requires the target model's output projection\"");
        output = model_other.output;
        output_s = model_other.output_s;
    }

    cur = b.buildLoraMm(output.?, cur, output_s);

    // Reduced-draft-vocabulary exports: the draft logits scattered to the
    // target vocabulary through d2t.
    if (model.d2t) |d2t| {
        const n_draft_vocab = cur.ne[0];
        const n_outputs = cur.ne[1];
        const n_vocab: i64 = model.vocab.nTokens();

        impl.assert(d2t.type == c.GGML_TYPE_I64, "models/dflash.cpp", 541, "model.d2t->type == GGML_TYPE_I64");
        impl.assert(d2t.ne[0] == n_draft_vocab, "models/dflash.cpp", 542, "model.d2t->ne[0] == n_draft_vocab");

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
}

/// `cparams.ctx_other`'s model, or the draft's own tensor when it has one:
/// the token embeddings for `graph_dsv4`.
fn dsv4TokEmbd(model: *const Model, b: *const graph.Builder) *Tensor {
    if (model.tok_embd) |t| return t;
    impl.assert(b.cparams.ctx_other != null, "models/dflash.cpp", 615, "cparams.ctx_other != nullptr");
    const model_other = otherModel(b);

    impl.assert(model_other.tok_embd != null, "models/dflash.cpp", 618, "model_other->tok_embd != nullptr && \"DSpark decoder requires the target model's token embeddings\"");
    return model_other.tok_embd.?;
}

/// Ports the constructor of `llama_model_dflash`'s `graph_dsv4`
/// (models/dflash.cpp:564 @c1d0e7a00): the DeepSeek-V4 DSpark decoder, in
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
    const n_embd_head: i64 = hp.nEmbdHeadK(0);
    const n_embd_head_rope: i64 = hp.nRot(0);
    const n_embd_head_nope = n_embd_head - n_embd_head_rope;

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKIswa();

    // KV cache injection: the fused target features from the encoder.
    if (b.ubatch.embd != null) {
        const inp = allocator.create(graph.InpEmbd) catch oom();
        inp.* = .{ .n_embd = b.n_embd };

        inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, b.n_embd, b.n_tokens);
        c.ggml_set_input(inp.embd);

        const inp_g = inp.embd.?;
        b.cb(inp_g, "inp_g_embeddings", -1);

        _ = b.res.addInput(graph.InpEmbd, inp);

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
            .n_expert_used = hp.n_expert_used,
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
        impl.assert(b.cparams.ctx_other != null, "models/dflash.cpp", 712, "cparams.ctx_other != nullptr");
        const model_other = otherModel(b);
        impl.assert(model_other.output != null, "models/dflash.cpp", 714, "model_other->output != nullptr && \"DSpark decoder requires the target model's output projection\"");
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
