//! Qwen4-Exp: the Qwen3.5 hybrid -- gated delta-net linear attention with a
//! full gated attention layer every few -- with hyper-connections in place
//! of every layer norm, QSA sparse attention on the full layers, a PLE
//! n-gram hash embedding, and a softmax-routed mixture of experts with a
//! gated shared expert.
//!
//! Hyper-connections keep `hc` parallel residual streams, `[n_embd, hc, T]`:
//! before each block a low-rank gated mixer collapses them into one input
//! and computes how the block's output is scattered back. QSA scores whole
//! blocks of `compress_ratio` cached tokens with mean-pooled indexer keys and
//! lets each query attend only to the best of them, through a mask. PLE
//! hashes each token's preceding n-grams on the host into rows of a shared
//! table, gates the gathered embedding against the residual, and runs a
//! dilated causal convolution whose history lives in the recurrent cache.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/qwen4exp.cpp` at v0.4.1 (`b29c606e2`),
//! with its declaration in `llama.cpp/src/models/models.h`:
//! `qwen4exp_require_nonzero`, `qwen4exp_require_arr_len`,
//! `llama_model_qwen4exp`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph` with every member, and the two inputs,
//! `llm_graph_input_qsa` and `llm_graph_input_ple`.
//!
//! The C++ graph is a class whose two maps -- one QSA input set per
//! compress ratio, one recurrent-row gather per cache tensor -- live as long
//! as one build; here they are the fields of `Graph`, which lives as long
//! as `buildArchGraph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const batch = @import("../batch.zig");
const kv_cache = @import("../kv_cache.zig");
const memory_hybrid = @import("../memory_hybrid.zig");
const memory_hybrid_idx = @import("../memory_hybrid_idx.zig");
const arch_mod = @import("../arch.zig");
const delta_net = @import("delta_net.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const Ubatch = batch.Ubatch;
const HybridContext = memory_hybrid.HybridContext;
const HybridIdxContext = memory_hybrid_idx.HybridIdxContext;
const allocator = impl.allocator;

const src = "models/qwen4exp.cpp";

fn oom() noreturn {
    impl.abort(src, 0, "out of memory");
}

/// `LLAMA_MAX_PLE_NGRAM` (llama-hparams.h:13 @b29c606e2).
const max_ple_ngram: u32 = 8;
/// `LLAMA_MAX_PLE_HEADS` (llama-hparams.h:14 @b29c606e2).
const max_ple_heads: u32 = 64;

/// `LLAMA_TOKEN_NULL`.
const token_null: i32 = -1;

/// Ports `qwen4exp_require_nonzero` (models/qwen4exp.cpp:10 @b29c606e2):
/// bad metadata must be catchable, where a `GGML_ASSERT` would abort the
/// whole process.
fn requireNonzero(ml: *const loader.Loader, kid: arch_mod.Kv, value: u32, exc: *Exception) Thrown!void {
    if (value == 0) {
        const nm = ml.meta.name(kid);
        return exc.throw(.runtime_error, "{s} must be greater than zero, got {d}", .{ nm.slice(), value });
    }
}

/// Ports `qwen4exp_require_arr_len` (models/qwen4exp.cpp:17 @b29c606e2):
/// `get_arr` copies a short array as it is, leaving a zero tail the n-gram
/// hash would silently drop.
fn requireArrLen(ml: *const loader.Loader, kid: arch_mod.Kv, n_min: u32, exc: *Exception) Thrown!void {
    const n_arr = (try ml.meta.getArrN(kid, true, exc)) orelse 0;
    if (n_arr < n_min) {
        const nm = ml.meta.name(kid);
        return exc.throw(.runtime_error, "{s} has {d} entries, but at least {d} are required", .{ nm.slice(), n_arr, n_min });
    }
}

/// Ports `llama_model_qwen4exp`'s `load_arch_hparams` (models/qwen4exp.cpp:26 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, false, exc);
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    _ = try meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, true, exc);

    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;
    try requireNonzero(ml, .ssm_conv_kernel, hp.ssm_d_conv, exc);
    try requireNonzero(ml, .ssm_inner_size, hp.ssm_d_inner, exc);
    try requireNonzero(ml, .ssm_state_size, hp.ssm_d_state, exc);
    try requireNonzero(ml, .ssm_time_step_rank, hp.ssm_dt_rank, exc);
    try requireNonzero(ml, .ssm_group_count, hp.ssm_n_group, exc);

    // Hyper-connections. The low rank is qwen4exp's own: DeepSeek-V4 leaves
    // it absent (full rank).
    hp.dsv4_hc_mult = (try meta.getKey(u32, .hyper_connection_count, true, exc)).?;
    hp.hc_low_rank = (try meta.getKey(u32, .hyper_connection_low_rank, true, exc)).?;
    // A count of 1 has nothing to mix: transformers' configuration, vLLM
    // and SGLang all raise on hc_count <= 1.
    if (hp.dsv4_hc_mult <= 1) {
        const nm = meta.name(.hyper_connection_count);
        return exc.throw(.runtime_error, "{s} must be greater than one, got {d}", .{ nm.slice(), hp.dsv4_hc_mult });
    }
    try requireNonzero(ml, .hyper_connection_low_rank, hp.hc_low_rank, exc);
    hp.n_embd_out_impl = hp.dsv4_hc_mult *% hp.n_embd;

    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;
    try requireNonzero(ml, .attention_indexer_head_count, hp.indexer_n_head, exc);
    try requireNonzero(ml, .attention_indexer_key_length, hp.indexer_head_size, exc);
    try requireNonzero(ml, .attention_indexer_top_k, hp.indexer_top_k, exc);
    _ = try meta.getKeyOrArr(u32, .attention_compress_ratios, &hp.dsv4_compress_ratios, hp.n_layer_all, false, exc);

    // The PLE n-gram hash embeddings; without the key group every field
    // stays zero.
    hp.is_ple_impl = @splat(0);
    hp.ple_n_heads = 0;

    const n_ple = (try meta.getArrN(.ple_layers, false, exc)) orelse 0;
    if (n_ple > 0) {
        const ple_layers = (try meta.getArrList(u32, .ple_layers, true, exc)).?;
        defer allocator.free(ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE
            // modules cannot be represented.
            const nm = meta.name(.ple_layers);
            return exc.throw(.runtime_error, "{s} lists {d} layers, but only one PLE layer is supported", .{ nm.slice(), n_ple });
        }
        for (ple_layers) |il| {
            if (il >= hp.n_layer_all) {
                return exc.throw(.runtime_error, "PLE layer {d} is out of range", .{il});
            }
            hp.is_ple_impl[il / 64] |= @as(u64, 1) << @intCast(il % 64);
        }

        hp.ple_ngram_size = (try meta.getKey(u32, .ple_ngram_size, true, exc)).?;
        hp.ple_heads_per_ngram = (try meta.getKey(u32, .ple_heads_per_ngram, true, exc)).?;
        hp.ple_conv_kernel = (try meta.getKey(u32, .ple_conv_kernel, true, exc)).?;
        hp.ple_eos_token_id = (try meta.getKey(u32, .ple_eos_token_id, true, exc)).?;
        // Optional: files written before this key fall back to the EOS
        // token.
        if (try meta.getKey(u32, .ple_image_token_id, false, exc)) |v| hp.ple_image_token_id = v;
        hp.n_embd_per_layer = (try meta.getKey(u32, .embedding_length_per_layer, true, exc)).?;
        try requireNonzero(ml, .ple_conv_kernel, hp.ple_conv_kernel, exc);
        try requireNonzero(ml, .embedding_length_per_layer, hp.n_embd_per_layer, exc);

        hp.ple_n_heads = (hp.ple_ngram_size -% 1) *% hp.ple_heads_per_ngram;
        hp.ple_head_dim = hp.n_embd_per_layer;
        if (hp.ple_ngram_size < 2 or hp.ple_ngram_size > max_ple_ngram) {
            return exc.throw(.runtime_error, "PLE n-gram size {d} is out of range", .{hp.ple_ngram_size});
        }
        if (hp.ple_n_heads == 0 or hp.ple_n_heads > max_ple_heads) {
            return exc.throw(.runtime_error, "PLE head count {d} is out of range", .{hp.ple_n_heads});
        }

        try requireArrLen(ml, .ple_layer_multipliers, hp.ple_ngram_size, exc);
        try requireArrLen(ml, .ple_head_offsets, hp.ple_n_heads, exc);
        try requireArrLen(ml, .ple_head_vocab_sizes, hp.ple_n_heads, exc);

        _ = try meta.getArr(u64, .ple_layer_multipliers, &hp.ple_layer_multipliers, true, exc);

        // The file stores the head ranges as uint64, so they are read at
        // that width and narrowed to the int32 the gather uses.
        var head_offsets: [max_ple_heads]u64 = @splat(0);
        var head_vocab_sizes: [max_ple_heads]u64 = @splat(0);
        _ = try meta.getArr(u64, .ple_head_offsets, &head_offsets, true, exc);
        _ = try meta.getArr(u64, .ple_head_vocab_sizes, &head_vocab_sizes, true, exc);
        const int32_max: u64 = std.math.maxInt(i32);
        var h: u32 = 0;
        while (h < hp.ple_n_heads) : (h += 1) {
            if (head_vocab_sizes[h] == 0 or
                head_offsets[h] > int32_max or
                head_vocab_sizes[h] > int32_max or
                head_offsets[h] + head_vocab_sizes[h] > int32_max)
            {
                return exc.throw(.runtime_error, "PLE head {d} range does not fit the int32 row index", .{h});
            }
            hp.ple_head_offsets[h] = @intCast(head_offsets[h]);
            hp.ple_head_vocab_sizes[h] = @intCast(head_vocab_sizes[h]);
        }
    }

    // Linear attention everywhere but every full_attention_interval-th
    // layer.
    if (!try meta.getKeyOrArr(u32, .attention_recurrent_layers, &hp.is_recr_impl, hp.n_layer_all, false, exc)) {
        var full_attn_interval: u32 = 4;
        if (try meta.getKey(u32, .full_attention_interval, false, exc)) |v| full_attn_interval = v;
        try requireNonzero(ml, .full_attention_interval, full_attn_interval, exc);
        var i: u32 = 0;
        while (i < hp.n_layer_all) : (i += 1) {
            hp.is_recr_impl[i] = @intFromBool(i < hp.nLayer() and (i + 1) % full_attn_interval != 0);
        }
    }

    // The PLE conv history is a row of the recurrent cache, which only the
    // linear layers have.
    var i: u32 = 0;
    while (i < hp.n_layer_all) : (i += 1) {
        if (hp.isPle(i) and !hp.isRecr(i)) {
            return exc.throw(.runtime_error, "PLE layer {d} is not a linear attention layer", .{i});
        }
    }

    self.type = switch (hp.nLayer()) {
        48 => .a3b,
        else => .unknown,
    };
}

/// Ports `llama_model_qwen4exp`'s `load_arch_tensors` (models/qwen4exp.cpp:150 @b29c606e2).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const hc: i64 = hp.dsv4_hc_mult;
    const hc_dim = hc * l.n_embd;
    const hc_lr: i64 = hp.hc_low_rank;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // There is no output_norm: the final hyper-connection mixer carries it.
    self.hc_head_norm = try self.createTensor(.{ .tensor = .hc_head_norm, .suffix = "weight" }, &.{hc_dim}, 0, exc);
    self.hc_head_down = try self.createTensor(.{ .tensor = .hc_head_down, .suffix = "weight" }, &.{ hc_dim, hc_lr }, 0, exc);
    self.hc_head_up = try self.createTensor(.{ .tensor = .hc_head_up, .suffix = "weight" }, &.{ hc_lr, hc_dim }, 0, exc);

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_not_required, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    // The flat [ple_head_dim, n_rows] gather target.
    if (hp.ple_n_heads > 0) {
        // The head ranges are what the gather indexes, so they set the
        // minimum row count.
        var ple_rows: i64 = 0;
        var h: u32 = 0;
        while (h < hp.ple_n_heads) : (h += 1) {
            ple_rows = @max(ple_rows, @as(i64, hp.ple_head_offsets[h]) + hp.ple_head_vocab_sizes[h]);
        }

        // The converter pads the table; a model synthesized from metadata
        // has no tensor to ask.
        const ple_name = arch_mod.tensorName(.per_layer_token_embd, "weight", -1, -1);
        if (ml.getWeight(ple_name.slice())) |ple_w| {
            if (ple_w.tensor.*.ne[1] < ple_rows) {
                return exc.throw(.runtime_error, "{s} has {d} rows, too few for the PLE head ranges ({d})", .{ ple_name.slice(), ple_w.tensor.*.ne[1], ple_rows });
            }
            ple_rows = ple_w.tensor.*.ne[1];
        }

        self.per_layer_tok_embd = try self.createTensor(.{ .tensor = .per_layer_token_embd, .suffix = "weight" }, &.{ hp.ple_head_dim, ple_rows }, loader.tensor_read_lazy, exc);
    }

    var il: i32 = 0;
    while (il < l.n_layer) : (il += 1) {
        const layer = &self.layers[@intCast(il)];
        const ilu: u32 = @intCast(il);

        const n_ff_exp: i64 = if (hp.nFfExp(0) != 0) hp.nFfExp(0) else @divTrunc(l.n_ff, l.n_expert_used);
        const n_ff_shexp: i64 = if (hp.n_ff_shexp != 0) hp.n_ff_shexp else l.n_ff;

        const head_k_dim: i64 = hp.ssm_d_state;
        const head_v_dim: i64 = hp.ssm_d_state;
        const n_k_heads: i64 = hp.ssm_n_group;
        const n_v_heads: i64 = hp.ssm_dt_rank;
        const key_dim = head_k_dim * n_k_heads;
        const value_dim = head_v_dim * n_v_heads;
        const conv_dim = key_dim * 2 + value_dim;

        // Two HC modules per layer: before the token mixer and before the
        // MoE.
        layer.hc_attn_norm = try self.createTensor(.{ .tensor = .hc_attn_norm, .suffix = "weight", .bid = il }, &.{hc_dim}, 0, exc);
        layer.hc_attn_down = try self.createTensor(.{ .tensor = .hc_attn_down, .suffix = "weight", .bid = il }, &.{ hc_dim, hc_lr }, 0, exc);
        layer.hc_attn_up = try self.createTensor(.{ .tensor = .hc_attn_up, .suffix = "weight", .bid = il }, &.{ hc_lr, hc_dim }, 0, exc);
        layer.hc_attn_inject = try self.createTensor(.{ .tensor = .hc_attn_inject, .suffix = "weight", .bid = il }, &.{ hc_dim, hc }, 0, exc);
        layer.hc_ffn_norm = try self.createTensor(.{ .tensor = .hc_ffn_norm, .suffix = "weight", .bid = il }, &.{hc_dim}, 0, exc);
        layer.hc_ffn_down = try self.createTensor(.{ .tensor = .hc_ffn_down, .suffix = "weight", .bid = il }, &.{ hc_dim, hc_lr }, 0, exc);
        layer.hc_ffn_up = try self.createTensor(.{ .tensor = .hc_ffn_up, .suffix = "weight", .bid = il }, &.{ hc_lr, hc_dim }, 0, exc);
        layer.hc_ffn_inject = try self.createTensor(.{ .tensor = .hc_ffn_inject, .suffix = "weight", .bid = il }, &.{ hc_dim, hc }, 0, exc);

        if (!hp.isRecr(ilu)) {
            // Full attention: wq holds [q|gate] interleaved per head.
            try self.createTensorQkv(layer, il, l.n_embd, l.n_embd_head_k * l.n_head * 2, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

            layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = il }, &.{l.n_embd_head_k}, 0, exc);
            layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = il }, &.{l.n_embd_head_k}, 0, exc);

            const idx_dim: i64 = hp.indexer_head_size;
            layer.index_q_proj = try self.createTensor(.{ .tensor = .indexer_q_proj, .suffix = "weight", .bid = il }, &.{ l.n_embd, @as(i64, hp.indexer_n_head) * idx_dim }, 0, exc);
            layer.index_k_proj = try self.createTensor(.{ .tensor = .indexer_k_proj, .suffix = "weight", .bid = il }, &.{ l.n_embd, idx_dim }, 0, exc);
            layer.index_q_norm = try self.createTensor(.{ .tensor = .indexer_q_norm, .suffix = "weight", .bid = il }, &.{idx_dim}, 0, exc);
            layer.index_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = il }, &.{idx_dim}, 0, exc);
        } else {
            layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = il }, &.{ l.n_embd, key_dim * 2 + value_dim }, 0, exc);
            layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = il }, &.{ l.n_embd, value_dim }, 0, exc);
            layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = il }, &.{ hp.ssm_d_conv, conv_dim }, 0, exc);
            layer.ssm_dt = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = il }, &.{hp.ssm_dt_rank}, 0, exc);
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = il }, &.{hp.ssm_dt_rank}, 0, exc);
            layer.ssm_beta = try self.createTensor(.{ .tensor = .ssm_beta, .suffix = "weight", .bid = il }, &.{ l.n_embd, n_v_heads }, 0, exc);
            layer.ssm_alpha = try self.createTensor(.{ .tensor = .ssm_alpha, .suffix = "weight", .bid = il }, &.{ l.n_embd, n_v_heads }, 0, exc);
            layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = il }, &.{head_v_dim}, 0, exc);
            layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = il }, &.{ value_dim, l.n_embd }, 0, exc);
        }

        if (hp.isPle(ilu)) {
            layer.ple_key = try self.createTensor(.{ .tensor = .ple_key, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc_dim }, 0, exc);
            layer.ple_value = try self.createTensor(.{ .tensor = .ple_value, .suffix = "weight", .bid = il }, &.{ l.n_embd, l.n_embd }, 0, exc);
            layer.ple_norm_key = try self.createTensor(.{ .tensor = .ple_norm_key, .suffix = "weight", .bid = il }, &.{hc_dim}, 0, exc);
            layer.ple_norm_query = try self.createTensor(.{ .tensor = .ple_norm_query, .suffix = "weight", .bid = il }, &.{hc_dim}, 0, exc);
            layer.ple_norm_conv = try self.createTensor(.{ .tensor = .ple_norm_conv, .suffix = "weight", .bid = il }, &.{hc_dim}, 0, exc);
            layer.ple_conv1d = try self.createTensor(.{ .tensor = .ple_conv1d, .suffix = "weight", .bid = il }, &.{ hp.ple_conv_kernel, hc_dim }, 0, exc);
        }

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = il }, &.{ l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = il }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
        try self.createTensorGateUpExps(layer, il, l.n_embd, n_ff_exp, l.n_expert, 0, exc);

        layer.ffn_gate_inp_shexp = try self.createTensor(.{ .tensor = .ffn_gate_inp_shexp, .suffix = "weight", .bid = il }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = il }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
        layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = il }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
        layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = il }, &.{ n_ff_shexp, l.n_embd }, 0, exc);
    }
}

/// Ports `llama_model_qwen4exp`'s `build_arch_graph` (models/qwen4exp.cpp:260 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    var g: Graph = .{ .model = model, .b = b };
    defer g.deinit();
    g.build();
}

/// Ports `llama_model_qwen4exp`'s `llm_graph_input_qsa` (models/qwen4exp.cpp:473
/// @b29c606e2): QSA attends to a budget of whole blocks of `compress_ratio`
/// tokens, plus the incomplete tail; one mean-pooled indexer key scores
/// each block, and `set_input` resolves the cache layout. Per stream: a
/// cell index names a different token in each stream.
pub const InpQsa = struct {
    /// I32 `[n_tokens]`.
    k_idxs: ?*Tensor = null,
    /// I32 `[n_kv, n_stream]`.
    cell_blk: ?*Tensor = null,
    /// I32 `[ratio*n_blocks, n_stream]`.
    blk_cells: ?*Tensor = null,
    /// I32 `[4*n_blocks*n_stream]`.
    blk_pos: ?*Tensor = null,
    /// F32 `[n_blocks or n_kv, n_tokens/n_stream, n_stream]`.
    bias: ?*Tensor = null,

    mctx: *const HybridIdxContext,
    ratio: u32,

    /// The per-cell half of the bias is the attention mask, so only the
    /// per-block half is uploaded.
    blk_bias: bool,

    /// Ports `llm_graph_input_qsa`'s `set_input` (models/qwen4exp.cpp:479 @b29c606e2).
    pub fn setInput(self: *InpQsa, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        self.mctx.getIdx().?.setInputKIdxs(self.k_idxs.?, ubatch);
        self.mctx.setInputQsa(self.cell_blk.?, self.blk_cells.?, self.blk_pos.?, self.bias.?, ubatch, self.ratio, self.blk_bias);
    }

    /// Ports `llm_graph_input_qsa`'s `can_reuse` (models/qwen4exp.cpp:484 @b29c606e2).
    pub fn canReuse(self: *InpQsa, params: *const graph.Params) bool {
        self.mctx = HybridIdxContext.fromBase(params.mctx.?.cast(HybridContext));

        const idx = self.mctx.getIdx() orelse return false;

        const n_kv: i64 = idx.getNKv();
        const n_stream: i64 = self.mctx.getNStream();
        const ratio: i64 = self.ratio;
        const n_blocks = @divTrunc(n_kv + ratio - 1, ratio);
        const n_tokens: i64 = params.ubatch.n_tokens;

        var res = true;

        res = res and @rem(n_tokens, n_stream) == 0;

        res = res and self.k_idxs.?.ne[0] == n_tokens;
        res = res and self.cell_blk.?.ne[0] == n_kv;
        res = res and self.cell_blk.?.ne[1] == n_stream;
        res = res and self.blk_cells.?.ne[0] == ratio * n_blocks;
        res = res and self.blk_pos.?.ne[0] == 4 * n_blocks * n_stream;
        res = res and self.bias.?.ne[0] == (if (self.blk_bias) n_blocks else n_kv);
        res = res and self.bias.?.ne[1] == @divTrunc(n_tokens, n_stream);

        return res;
    }
};

/// Ports `llm_graph_input_ple` (models/qwen4exp.cpp:1028 @b29c606e2): the
/// PLE n-gram hash embedding's rows. Each token gathers `ple_n_heads` rows
/// of a shared table:
///
///     mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1])
///     row     = mixed_n % vocab[h] + offset[h]
///
/// The hash runs on the host, because ggml has no int64 and no xor. An EOS
/// resets the window.
pub const InpPle = struct {
    /// I32 `[ple_n_heads * n_tokens]`.
    rows: ?*Tensor = null,

    pmodel: *const Model,

    /// The predecessor tokens live in the attention KV cells (`ext.tok`).
    mctx: ?*const kv_cache.KvCacheContext,

    /// Scratch, reused across `setInput` calls.
    prev: std.ArrayList(i32) = .empty,

    pub fn deinit(self: *InpPle) void {
        self.prev.deinit(allocator);
    }

    /// Ports `llm_graph_input_ple`'s `can_reuse` (models/qwen4exp.cpp:1036 @b29c606e2).
    pub fn canReuse(self: *InpPle, params: *const graph.Params) bool {
        self.mctx = HybridIdxContext.fromBase(params.mctx.?.cast(HybridContext)).getAttn();
        return self.rows.?.ne[0] == @as(i64, self.pmodel.hparams.ple_n_heads) * @as(i64, params.ubatch.n_tokens);
    }

    /// Ports `llm_graph_input_ple`'s `set_input` (models/qwen4exp.cpp:1052 @b29c606e2).
    pub fn setInput(self: *InpPle, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const hp = &self.pmodel.hparams;

        // An image arrives as an embedding batch, so ubatch.token is null,
        // but every position still needs a row for ggml_get_rows: it
        // stands in the image token id the reference hashes, or EOS if the
        // file has no such key. Gemma3n and Gemma 4 do the same with a
        // hard-coded row 0 of per_layer_token_embd.
        const img_tok: i32 = if (hp.ple_image_token_id != 0) @bitCast(hp.ple_image_token_id) else @bitCast(hp.ple_eos_token_id);

        const n_tokens: i64 = ubatch.n_tokens;
        const n_gram: i64 = hp.ple_ngram_size;
        const n_heads: i64 = hp.ple_n_heads;
        const per_gram: i64 = hp.ple_heads_per_ngram;
        const eos: i64 = hp.ple_eos_token_id;
        const n_prev = n_gram - 1;

        const idx = allocator.alloc(i32, @intCast(n_heads * n_tokens)) catch oom();
        defer allocator.free(idx);
        @memset(idx, 0);

        impl.assert(self.mctx != null, src, 1073, "mctx != nullptr");

        var i: i64 = 0;
        while (i < n_tokens) : (i += 1) {
            // The preceding tokens would be ambiguous; see get_prev_tokens.
            impl.assert(ubatch.n_seq_id[@intCast(i)] == 1, src, 1077, "ubatch->n_seq_id[i] == 1 && \"PLE n-gram embeddings do not support tokens shared by multiple sequences\"");
        }

        // The predecessors come from the KV cells (ext.tok); apply_ubatch
        // already stored this ubatch, so its own tokens count too.
        self.mctx.?.getPrevTokens(ubatch, @intCast(n_prev), &self.prev);

        i = 0;
        while (i < n_tokens) : (i += 1) {
            // An EOS in the window resets everything at or before it, and a
            // missing predecessor (before the sequence's start, or with no
            // cached cell) reads as EOS. The token's own EOS does not cut
            // its own context, as in the reference.
            var ctx: [max_ple_ngram]i64 = undefined;
            ctx[0] = if (ubatch.token != null) ubatch.token[@intCast(i)] else img_tok;
            var cut = false;
            var s: i64 = 1;
            while (s < n_gram) : (s += 1) {
                // The predecessor s positions back; prev is oldest-first,
                // its missing entries LLAMA_TOKEN_NULL.
                const t: i32 = if (cut) token_null else self.prev.items[@intCast(i * n_prev + (n_prev - s))];
                cut = cut or t < 0 or t == eos;
                ctx[@intCast(s)] = if (cut) eos else t;
            }

            var n: i64 = 2;
            while (n <= n_gram) : (n += 1) {
                var mixed: u64 = @as(u64, @bitCast(ctx[0])) *% hp.ple_layer_multipliers[0];
                var j: i64 = 1;
                while (j < n) : (j += 1) {
                    mixed ^= @as(u64, @bitCast(ctx[@intCast(j)])) *% hp.ple_layer_multipliers[@intCast(j)];
                }
                const base = (n - 2) * per_gram;
                var g: i64 = 0;
                while (g < per_gram) : (g += 1) {
                    const h_i: usize = @intCast(base + g);
                    const row: u64 = mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i];
                    idx[@intCast(i * n_heads + @as(i64, @intCast(h_i)))] = @bitCast(@as(u32, @truncate(row)));
                }
            }
        }

        c.ggml_backend_tensor_set(self.rows, idx.ptr, 0, idx.len * c.ggml_element_size(self.rows));
    }
};

/// Ports `llama_model_qwen4exp`'s `graph` (src/models/models.h:2369
/// @b29c606e2): the builder and the two maps it keeps for one build.
const Graph = struct {
    model: *const Model,
    b: *const graph.Builder,

    /// The QSA cache layout inputs depend not on the layer but only on its
    /// compress ratio, so the layers sharing a ratio share one input set.
    qsa_inps: std.ArrayList(struct { ratio: u32, inp: *InpQsa }) = .empty,

    /// build_rs writes the state tensor in place, so one gather per cache
    /// tensor is reused.
    rs_rows: std.ArrayList(struct { key: *Tensor, rows: *Tensor }) = .empty,

    fn deinit(self: *Graph) void {
        self.qsa_inps.deinit(allocator);
        self.rs_rows.deinit(allocator);
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_hc_mix` (models/qwen4exp.cpp:266
    /// @b29c606e2): hyper-connections keep `hc` parallel residual streams,
    /// `[n_embd, hc, T]`, in place of the layer norms. Returns the mixed
    /// `[n_embd, T]` stream; `inject` gets the `[hc, T]` scatter weights.
    fn buildHcMix(self: *Graph, x: *Tensor, w_norm: ?*Tensor, w_down: ?*Tensor, w_up: ?*Tensor, w_inject: ?*Tensor, inject: ?*?*Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const hc: i64 = hp.dsv4_hc_mult;
        const hc_dim = hc * b.n_embd;
        const nt = x.ne[2];

        // A grouped RMSNorm: reduced over one stream, then all streams
        // scaled by the [hc_dim] gamma, which the converter folded to
        // (1 + w).
        var xn: *Tensor = c.ggml_rms_norm(ctx, x, hp.f_norm_rms_eps);
        xn = c.ggml_reshape_2d(ctx, xn, hc_dim, nt);
        xn = c.ggml_mul(ctx, xn, w_norm);
        b.cb(xn, "hc_norm", il);

        var lo = b.buildLoraMm(w_down.?, xn, null);
        lo = c.ggml_silu(ctx, c.ggml_scale(ctx, lo, 1.0 / @as(f32, @floatFromInt(hc))));
        const gate: *Tensor = c.ggml_sigmoid(ctx, b.buildLoraMm(w_up.?, lo, null));
        b.cb(gate, "hc_gate", il);

        var gated: *Tensor = c.ggml_mul(ctx, xn, gate);
        gated = c.ggml_reshape_3d(ctx, gated, b.n_embd, hc, nt);

        // The streams collapse to their mean.
        const row = c.ggml_row_size(gated.type, b.n_embd);
        const hcu: usize = @intCast(hc);
        var mixed: *Tensor = c.ggml_view_2d(ctx, gated, b.n_embd, nt, row * hcu, 0);
        mixed = c.ggml_cont(ctx, mixed);
        var ci: usize = 1;
        while (ci < hcu) : (ci += 1) {
            const s: *Tensor = c.ggml_view_2d(ctx, gated, b.n_embd, nt, row * hcu, row * ci);
            mixed = c.ggml_add(ctx, mixed, s);
        }
        mixed = c.ggml_scale(ctx, mixed, 1.0 / @as(f32, @floatFromInt(hc)));
        b.cb(mixed, "hc_mixed", il);

        if (inject) |out| {
            out.* = b.buildLoraMm(w_inject.?, xn, null);
            b.cb(out.*.?, "hc_inject", il);
        }

        return mixed;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_hc_combine`
    /// (models/qwen4exp.cpp:314 @b29c606e2): the block's output scattered
    /// back into the streams by the injection weights.
    fn buildHcCombine(self: *Graph, residual: *Tensor, block_out: *Tensor, inject: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hc: i64 = b.hparams.dsv4_hc_mult;
        const nt = residual.ne[2];

        // 2*sigmoid centres the scatter weights on 1, so a zero injection
        // is a plain residual add.
        var w: *Tensor = c.ggml_sigmoid(ctx, c.ggml_scale(ctx, inject, 1.0 / @as(f32, @floatFromInt(hc))));
        w = c.ggml_scale(ctx, w, 2.0);
        w = c.ggml_reshape_3d(ctx, w, 1, hc, nt);

        var bo: *Tensor = c.ggml_reshape_3d(ctx, block_out, b.n_embd, 1, nt);
        bo = c.ggml_repeat_4d(ctx, bo, b.n_embd, hc, nt, 1);

        const cur: *Tensor = c.ggml_add(ctx, residual, c.ggml_mul(ctx, bo, w));
        b.cb(cur, "hc_combine", il);

        return cur;
    }

    /// Ports the constructor of `llama_model_qwen4exp`'s `graph`
    /// (models/qwen4exp.cpp:336 @b29c606e2).
    fn build(self: *Graph) void {
        const b = self.b;
        const model = self.model;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const hc: i64 = hp.dsv4_hc_mult;

        impl.assert(hp.nEmbdHeadV(0) == hp.nEmbdHeadK(0), src, 340, "hparams.n_embd_head_v() == hparams.n_embd_head_k()");

        var sections: [4]c_int = undefined;
        @memcpy(&sections, hp.rope_sections[0..4]);

        const inpL = b.buildInpEmbd(model.tok_embd.?);
        b.cb(inpL, "model.input_embed", -1);
        c.ggml_build_forward_expand(b.gf, inpL);

        const inp = b.buildInpMemHybrid();

        // qwen4exp always builds llama_memory_hybrid_idx, so this downcast
        // is safe; the indexer cache inside it is absent when the GGUF has
        // no indexer tensors.
        const mctx_hyb = HybridIdxContext.fromBase(inp.mctx);

        if (mctx_hyb.getIdx()) |mctx_idx| {
            impl.assert(mctx_idx.getNKv() == inp.mctx.getAttn().getNKv(), src, 357, "mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() && \"the indexer cache must track the attention cache cell for cell\"");
        }

        const inp_pos = b.buildInpPos();
        const inp_out_ids = b.buildInpOutIds();

        var ple_emb: ?*Tensor = null;
        if (hp.ple_n_heads > 0) {
            ple_emb = self.buildInpPle(mctx_hyb);
            // ple_emb and build_inp_embd in the same graph split.
            c.ggml_build_forward_expand(b.gf, ple_emb);
        }

        // The wide residual starts as hc identical copies of the embedding.
        var res_hc: *Tensor = c.ggml_repeat_4d(ctx, c.ggml_reshape_3d(ctx, inpL, b.n_embd, 1, b.n_tokens), b.n_embd, hc, b.n_tokens, 1);
        b.cb(res_hc, "hc_init", -1);

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];
            b.res.t_layer_inp[@intCast(il)] = res_hc;

            if (hp.isPle(@intCast(il))) {
                res_hc = self.buildPle(inp.getRecr(), ple_emb.?, res_hc, il);
            }

            var inject: ?*Tensor = null;
            var cur = self.buildHcMix(res_hc, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, &inject, il);

            c.ggml_build_forward_expand(b.gf, cur);

            if (hp.isRecr(@intCast(il))) {
                cur = self.buildLayerAttnLinear(inp.getRecr(), cur, il);
            } else {
                cur = self.buildLayerAttn(inp.getAttn(), mctx_hyb, cur, inp_pos, &sections, il);
            }

            if (il == b.n_layer - 1) {
                // Everything below is per token, so the rows that produce
                // no output go.
                cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
                inject = c.ggml_get_rows(ctx, inject, inp_out_ids);

                res_hc = c.ggml_reshape_2d(ctx, res_hc, b.n_embd * hc, res_hc.ne[2]);
                res_hc = c.ggml_get_rows(ctx, res_hc, inp_out_ids);
                res_hc = c.ggml_reshape_3d(ctx, res_hc, b.n_embd, hc, res_hc.ne[1]);
            }

            res_hc = self.buildHcCombine(res_hc, cur, inject.?, il);

            cur = self.buildHcMix(res_hc, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, &inject, il);

            cur = self.buildLayerFfn(cur, il);
            b.cb(cur, "ffn_out", il);

            res_hc = self.buildHcCombine(res_hc, cur, inject.?, il);

            // "l_last" is the layer output's name that build_cvec and
            // imatrix look for.
            b.cb(res_hc, "l_last", il);
        }

        // The final mixer is the output norm: there is no separate one.
        var cur = self.buildHcMix(res_hc, model.hc_head_norm, model.hc_head_down, model.hc_head_up, null, null, -1);

        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        cur = b.buildLoraMm(model.output.?, cur, model.output_s);
        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;

        c.ggml_build_forward_expand(b.gf, cur);
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_qkvz` (models/qwen4exp.cpp:443
    /// @b29c606e2): the mixed QKV projection and the gate, z.
    fn buildQkvz(self: *Graph, input: *Tensor, il: c_int) struct { *Tensor, *Tensor } {
        const b = self.b;
        const layer = &self.model.layers[@intCast(il)];
        const n_seqs: i64 = b.ubatch.n_seqs;
        const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

        var qkv_mixed = b.buildLoraMm(layer.wqkv.?, input, layer.wqkv_s);
        qkv_mixed = c.ggml_reshape_3d(b.ctx0, qkv_mixed, qkv_mixed.ne[0], n_seq_tokens, n_seqs);
        b.cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        const z = b.buildLoraMm(layer.wqkv_gate.?, input, layer.wqkv_gate_s);
        b.cb(z, "z", il);

        return .{ qkv_mixed, z };
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_norm_gated`
    /// (models/qwen4exp.cpp:459 @b29c606e2): the one numerical difference
    /// from Qwen3.5's delta net, a sigmoid output gate, not a SiLU.
    fn buildNormGated(self: *Graph, input: *Tensor, weights: ?*Tensor, gate: *Tensor, layer: c_int) *Tensor {
        const b = self.b;
        const normalized = b.buildNorm(input, weights, null, .rms, layer);
        const gated: *Tensor = c.ggml_sigmoid(b.ctx0, gate);

        return c.ggml_mul(b.ctx0, normalized, gated);
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_qsa_top_k`
    /// (models/qwen4exp.cpp:525 @b29c606e2): the token indices this layer's
    /// queries may attend to.
    fn buildQsaTopK(self: *Graph, mctx_hyb: *const HybridIdxContext, cur: *Tensor, inp_pos: *Tensor, kq_mask: ?*Tensor, sections: *[4]c_int, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const layer = &self.model.layers[@intCast(il)];
        const mctx_idx = mctx_hyb.getIdx().?;

        const idx_dim: i64 = hp.indexer_head_size;
        const n_idx_h: i64 = hp.indexer_n_head;
        const r: i64 = hp.dsv4_compress_ratios[@intCast(il)];
        const n_kv: i64 = mctx_idx.getNKv();

        impl.assert(r > 0, src, 539, "r > 0");

        const n_blocks = @divTrunc(n_kv + r - 1, r);

        // build_attn_qsa and the KQ mask need the tokens to divide evenly
        // across the streams.
        const n_tokens: i64 = b.n_tokens;
        const n_stream: i64 = mctx_hyb.getNStream();
        impl.assert(@rem(n_tokens, n_stream) == 0, src, 545, "n_tokens % n_stream == 0");
        const n_tps = @divTrunc(n_tokens, n_stream);

        // Only the "which block is visible" half of the bias varies per
        // block; the rest is the visible-or-not test the attention mask
        // already carries, so only the per-block half is uploaded: 1/ratio
        // of the cells. ALiBi writes distances instead of a mask and
        // non-causal attention keeps future cells, so both opt out. The
        // mask also holds an M-RoPE rule for the query's own position, but
        // only 2D image positions can differ there.
        const blk_bias = if (kq_mask) |m|
            m.ne[0] == n_kv and m.ne[1] == n_tps and m.ne[3] == n_stream and b.cparams.causal_attn and !hp.use_alibi
        else
            false;

        // Nothing above depends on the layer, so the layers sharing a ratio
        // share one input set.
        var inp: *InpQsa = undefined;
        const ratio: u32 = @intCast(r);
        for (self.qsa_inps.items) |e| {
            if (e.ratio == ratio) {
                inp = e.inp;
                break;
            }
        } else {
            const qsa = allocator.create(InpQsa) catch oom();
            qsa.* = .{ .mctx = mctx_hyb, .ratio = ratio, .blk_bias = blk_bias };

            qsa.k_idxs = mctx_idx.buildInputKIdxs(ctx, b.ubatch);
            qsa.cell_blk = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, n_kv, n_stream);
            qsa.blk_cells = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, r * n_blocks, n_stream);
            qsa.blk_pos = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, 4 * n_blocks * n_stream);
            qsa.bias = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, if (blk_bias) n_blocks else n_kv, n_tps, n_stream);

            c.ggml_set_input(qsa.cell_blk);
            c.ggml_set_input(qsa.blk_cells);
            c.ggml_set_input(qsa.blk_pos);
            c.ggml_set_input(qsa.bias);

            inp = b.res.addInput(InpQsa, qsa);
            self.qsa_inps.append(allocator, .{ .ratio = ratio, .inp = inp }) catch oom();
        }

        // The cached indexer keys are raw: pooling precedes the norm and
        // the rotation, so neither is applied.
        var k_raw = b.buildLoraMm(layer.index_k_proj.?, cur, null);
        k_raw = c.ggml_reshape_3d(ctx, k_raw, idx_dim, 1, n_tokens);
        b.cb(k_raw, "indexer_k_raw", il);

        c.ggml_build_forward_expand(b.gf, mctx_idx.cpyK(ctx, k_raw, inp.k_idxs.?, il));

        // One key head, so the rows are contiguous. get_k gives
        // [idx_dim, n_head_kv, n_kv, n_stream].
        var k_all = mctx_idx.getK(ctx, il);
        k_all = c.ggml_view_3d(ctx, k_all, idx_dim, n_kv, n_stream, k_all.nb[2], k_all.nb[3], 0);

        // Gathered per stream: blk_cells row s indexes stream s's own
        // cells.
        var members: *Tensor = c.ggml_get_rows(ctx, k_all, inp.blk_cells);
        members = c.ggml_reshape_4d(ctx, members, idx_dim, r, n_blocks, n_stream);

        // The mean over the block's members; r is small, so summing slices
        // beats a transpose and sum_rows.
        var pooled: ?*Tensor = null;
        var i: i64 = 0;
        while (i < r) : (i += 1) {
            const slice: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, members, idx_dim, n_blocks, n_stream, members.nb[2], members.nb[3], @as(usize, @intCast(i)) * members.nb[1]));
            pooled = if (pooled) |p| c.ggml_add(ctx, p, slice) else slice;
        }
        var pk: *Tensor = c.ggml_scale(ctx, pooled, 1.0 / @as(f32, @floatFromInt(r)));
        b.cb(pk, "indexer_k_pooled", il);

        // Blocks are counted along ne1: rms_norm launches gridDim.y = ne2,
        // capped at 65535, and 262144/4 = 65536.
        pk = c.ggml_reshape_3d(ctx, pk, idx_dim, n_blocks * n_stream, 1);
        pk = b.buildNorm(pk, layer.index_k_norm, null, .rms, il);

        // RoPE wants [n_dims, n_head, n_tokens]: every stream's blocks laid
        // flat, split after.
        pk = c.ggml_reshape_3d(ctx, pk, idx_dim, 1, n_blocks * n_stream);
        pk = c.ggml_rope_multi(ctx, pk, inp.blk_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        pk = c.ggml_reshape_3d(ctx, pk, idx_dim, n_blocks, n_stream);
        b.cb(pk, "indexer_k", il);

        var q = b.buildLoraMm(layer.index_q_proj.?, cur, null);
        q = c.ggml_reshape_3d(ctx, q, idx_dim, n_idx_h, n_tokens);
        q = b.buildNorm(q, layer.index_q_norm, null, .rms, il);
        q = c.ggml_rope_multi(ctx, q, inp_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        b.cb(q, "indexer_q", il);

        // Each head's dot product is rectified before the sum, as in the
        // DeepSeek lightning indexer. mul_mat matches ne[2], so the queries
        // of stream s meet only the blocks of stream s.
        var score: *Tensor = c.ggml_mul_mat(ctx, pk, c.ggml_reshape_3d(ctx, q, idx_dim, n_idx_h * n_tps, n_stream));
        score = c.ggml_reshape_4d(ctx, score, n_blocks, n_idx_h, n_tps, n_stream);
        score = c.ggml_relu(ctx, score);

        // The heads sit side by side on ne[1], and there are few of them.
        var summed: ?*Tensor = null;
        var h: i64 = 0;
        while (h < n_idx_h) : (h += 1) {
            const slice: *Tensor = c.ggml_view_3d(ctx, score, n_blocks, n_tps, n_stream, score.nb[2], score.nb[3], @as(usize, @intCast(h)) * score.nb[1]);
            summed = if (summed) |s| c.ggml_add(ctx, s, slice) else c.ggml_cont(ctx, slice);
        }

        score = summed.?;
        b.cb(score, "indexer_score", il);

        // One value per block, so biasing here is cheaper than after the
        // cells are expanded.
        if (blk_bias) {
            score = c.ggml_add(ctx, score, inp.bias);
        }

        // Every token of a block gets the block's score; the budget is
        // whole blocks, so top-k cuts on a block boundary.
        var expanded: *Tensor = c.ggml_get_rows(ctx, c.ggml_cont(ctx, c.ggml_permute(ctx, score, 1, 0, 2, 3)), inp.cell_blk);
        expanded = c.ggml_cont(ctx, c.ggml_permute(ctx, expanded, 1, 0, 2, 3));

        if (blk_bias) {
            // Flash attention keeps the mask in f16; the scores are f32.
            const m = kq_mask.?;
            const mask: *Tensor = if (m.type == c.GGML_TYPE_F32) m else c.ggml_cast(ctx, m, c.GGML_TYPE_F32);
            expanded = c.ggml_add(ctx, expanded, c.ggml_reshape_3d(ctx, mask, n_kv, n_tps, n_stream));
        } else {
            expanded = c.ggml_add(ctx, expanded, inp.bias);
        }
        b.cb(expanded, "indexer_score_tokens", il);

        // The reference returns indexer_top_k + compress_ratio - 1: whole
        // blocks plus the tail.
        const width: i64 = @min(n_kv, @as(i64, hp.indexer_top_k) + r - 1);

        var top_k: *Tensor = c.ggml_cont(ctx, c.ggml_top_k(ctx, expanded, @intCast(width)));

        // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching
        // the KQ mask.
        top_k = c.ggml_reshape_4d(ctx, top_k, width, n_tps, 1, n_stream);
        b.cb(top_k, "indexer_top_k", il);

        return top_k;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_attn_qsa`
    /// (models/qwen4exp.cpp:678 @b29c606e2): dense GQA self-attention
    /// restricted to the cells top_k names. The mask is built as the MLA
    /// sparse path in llm_graph_context::build_attn builds it.
    fn buildAttnQsa(self: *Graph, inp: *graph.InpAttnKv, q_cur_: *Tensor, k_cur_: *Tensor, v_cur_: *Tensor, top_k: *Tensor, kq_scale: f32, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        var q_cur = q_cur_;
        var k_cur = k_cur_;
        var v_cur = v_cur_;

        // Q, K and V rotated before they reach a quantized cache, as the
        // dense path does. The indexer has already scored with its own
        // query in build_qsa_top_k, so top_k is unaffected.
        if (inp.self_k_rot) |r| {
            q_cur = graph.mulMatHadamard(ctx, q_cur, r);
            k_cur = graph.mulMatHadamard(ctx, k_cur, r);
        }

        if (inp.self_v_rot) |r| {
            v_cur = graph.mulMatHadamard(ctx, v_cur, r);
        }

        // Added to the graph together so that they are not reordered,
        // which reduces the graph's splits; K is expanded last to enable the
        // RoPE fusion that writes straight into the KV cache.
        c.ggml_build_forward_expand(b.gf, q_cur);
        c.ggml_build_forward_expand(b.gf, v_cur);
        c.ggml_build_forward_expand(b.gf, k_cur);

        const mctx_cur = inp.mctx;

        // Store to the KV cache.
        {
            const k_idxs = inp.getKIdxs();
            const v_idxs = inp.getVIdxs();

            c.ggml_build_forward_expand(b.gf, mctx_cur.cpyK(ctx, k_cur, k_idxs, il));
            c.ggml_build_forward_expand(b.gf, mctx_cur.cpyV(ctx, v_cur, v_idxs, il));
        }

        const kq_mask = inp.getKqMask().?;

        // The new KQ mask, filled with -INFINITY to start.
        var kq_mask_all: *Tensor = c.ggml_fill(ctx, kq_mask, -std.math.inf(f32));

        // The KQ mask with rows of size 1:
        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream].
        kq_mask_all = c.ggml_view_4d(ctx, kq_mask_all, 1, kq_mask_all.ne[0], kq_mask_all.ne[1], kq_mask_all.ne[3], kq_mask_all.nb[0], kq_mask_all.nb[1], kq_mask_all.nb[2], 0);

        // The top-k indices: [n_top_k, n_batch, 1, n_stream] ->
        // [n_top_k, n_batch, n_stream, 1].
        const top_k_3d: *Tensor = c.ggml_view_4d(ctx, top_k, top_k.ne[0], top_k.ne[1], top_k.ne[3], 1, top_k.nb[1], top_k.nb[2], @as(usize, @intCast(top_k.ne[3])) * top_k.nb[3], 0);

        // Zeros with rows of size 1, [1, n_top_k, n_batch, n_stream]: the
        // values that unmask the top-k elements.
        var zeros: *Tensor = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, 1, top_k_3d.ne[0], top_k_3d.ne[1], top_k_3d.ne[2]);
        zeros = c.ggml_fill(ctx, zeros, 0.0);

        // The KQ mask with the elements top_k names unmasked:
        // ggml_set_rows([1, n_kv, n_batch, n_stream],
        // [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1]).
        var kq_mask_top_k: *Tensor = c.ggml_set_rows(ctx, kq_mask_all, zeros, top_k_3d);

        // The KQ mask's original shape restored:
        // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream].
        kq_mask_top_k = c.ggml_view_4d(ctx, kq_mask_top_k, kq_mask_top_k.ne[1], kq_mask_top_k.ne[2], 1, kq_mask_top_k.ne[3], kq_mask_top_k.nb[2], kq_mask_top_k.nb[3], kq_mask_top_k.nb[3], 0);

        // Combined with the original KQ mask.
        kq_mask_top_k = c.ggml_add(ctx, kq_mask_top_k, kq_mask);

        const q = q_cur;
        const k = mctx_cur.getK(ctx, il);
        const v = mctx_cur.getV(ctx, il);

        // Upstream's TODO: sparse attention, when it is ready (its call
        // would pass top_k.ne[0] as n_kv_max).
        var cur = b.buildAttnMha(q, k, v, null, kq_mask_top_k, null, null, 0, kq_scale, il);
        b.cb(cur, "kqv_out", il);

        // The rotation is its own inverse, so it is undone on the value
        // side of the output.
        if (inp.self_v_rot) |r| {
            cur = graph.mulMatHadamard(ctx, cur, r);
        }

        return cur;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_layer_attn`
    /// (models/qwen4exp.cpp:761 @b29c606e2): a joint Q and gate
    /// projection, Q and K norms, M-RoPE, dense or QSA attention, and the
    /// sigmoid gate.
    fn buildLayerAttn(self: *Graph, inp: *graph.InpAttnKv, mctx_hyb: *const HybridIdxContext, cur_: *Tensor, inp_pos: *Tensor, sections: *[4]c_int, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const layer = &self.model.layers[@intCast(il)];
        const n_embd_head: i64 = hp.nEmbdHeadV(0);
        impl.assert(n_embd_head == hp.nEmbdHeadK(0), src, 769, "n_embd_head == hparams.n_embd_head_k()");
        const n_head = b.n_head;
        const n_head_kv = b.n_head_kv;
        const n_tokens = b.n_tokens;

        // The indexer reads the same block input as Q, K and V; no indexer
        // cache or no ratio means dense.
        const qsa = mctx_hyb.getIdx() != null and hp.dsv4_compress_ratios[@intCast(il)] > 0;

        const top_k: ?*Tensor = if (qsa) self.buildQsaTopK(mctx_hyb, cur_, inp_pos, inp.getKqMask(), sections, il) else null;

        // As Qwen3Next, one Q projection outputs the query and the gate.
        const Qcur_full = b.buildLoraMm(layer.wq.?, cur_, layer.wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
        b.cb(Qcur_full, "Qcur_full", il);

        const es = c.ggml_element_size(Qcur_full);
        const head_sz: usize = @intCast(n_embd_head);
        var Qcur: *Tensor = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, es * head_sz * 2, es * head_sz * 2 * @as(usize, @intCast(n_head)), 0);
        b.cb(Qcur, "Qcur_reshaped", il);

        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        b.cb(Qcur, "Qcur_normed", il);

        var Kcur = b.buildLoraMm(layer.wk.?, cur_, layer.wk_s);
        b.cb(Kcur, "Kcur", il);

        var Vcur = b.buildLoraMm(layer.wv.?, cur_, layer.wv_s);
        b.cb(Vcur, "Vcur", il);

        Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
        b.cb(Kcur, "Kcur_normed", il);

        var gate: *Tensor = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, es * head_sz * 2, es * head_sz * 2 * @as(usize, @intCast(n_head)), es * head_sz);
        gate = c.ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
        b.cb(gate, "gate_reshaped", il);

        Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

        // Interleaved M-RoPE.
        Qcur = c.ggml_rope_multi(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        Kcur = c.ggml_rope_multi(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        b.cb(Qcur, "Qcur", il);
        b.cb(Kcur, "Kcur", il);
        b.cb(Vcur, "Vcur", il);

        const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

        var cur: *Tensor = undefined;
        if (top_k) |tk| {
            cur = self.buildAttnQsa(inp, Qcur, Kcur, Vcur, tk, kq_scale, il);
        } else {
            cur = b.buildAttn(inp, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
        }
        b.cb(cur, "attn_pregate", il);

        const gate_sigmoid: *Tensor = c.ggml_sigmoid(ctx, gate);
        b.cb(gate_sigmoid, "gate_sigmoid", il);

        cur = c.ggml_mul(ctx, cur, gate_sigmoid);
        b.cb(cur, "attn_gated", il);

        cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
        b.cb(cur, "attn_output", il);

        return cur;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_layer_attn_linear`
    /// (models/qwen4exp.cpp:847 @b29c606e2): the gated delta-net layer.
    fn buildLayerAttnLinear(self: *Graph, inp: *graph.InpRs, cur_: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const layer = &self.model.layers[@intCast(il)];
        const mctx_cur = inp.mctx;

        const d_inner: i64 = hp.ssm_d_inner;
        const n_seqs: i64 = b.ubatch.n_seqs;
        const head_k_dim: i64 = hp.ssm_d_state;
        const num_k_heads: i64 = hp.ssm_n_group;
        const num_v_heads: i64 = hp.ssm_dt_rank;
        const head_v_dim: i64 = hp.ssm_d_state;
        const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

        impl.assert(n_seqs != 0, src, 861, "n_seqs != 0");
        impl.assert(b.ubatch.equalSeqs(), src, 862, "ubatch.equal_seqs()");
        impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, src, 863, "ubatch.n_tokens == n_seq_tokens * n_seqs");
        impl.assert(head_v_dim * num_v_heads == d_inner, src, 864, "head_v_dim * num_v_heads == d_inner");

        const qkv_mixed, const z = self.buildQkvz(cur_, il);

        var beta = b.buildLoraMm(layer.ssm_beta.?, cur_, layer.ssm_beta_s);
        beta = c.ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
        b.cb(beta, "beta", il);

        beta = c.ggml_sigmoid(ctx, beta);
        b.cb(beta, "beta_sigmoid", il);

        var alpha = b.buildLoraMm(layer.ssm_alpha.?, cur_, layer.ssm_alpha_s);
        alpha = c.ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
        b.cb(alpha, "alpha", il);

        const alpha_biased: *Tensor = c.ggml_add(ctx, alpha, layer.ssm_dt);
        const alpha_softplus: *Tensor = c.ggml_softplus(ctx, alpha_biased);
        b.cb(alpha_softplus, "a_softplus", il);

        var gate: *Tensor = c.ggml_mul(ctx, alpha_softplus, layer.ssm_a); // -A_log.exp() * softplus
        b.cb(gate, "gate", il);

        gate = c.ggml_reshape_4d(ctx, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

        const conv_states_all = mctx_cur.getRL(il);
        const ssm_states_all = mctx_cur.getSL(il);

        const conv_kernel = layer.ssm_conv1d.?;
        const conv_kernel_size = conv_kernel.ne[0];

        // The channels must match how load_arch_tensors sizes wqkv, not
        // ssm_d_inner.
        const conv_channels = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

        const conv_input = self.buildConvStateAt(inp, conv_states_all, qkv_mixed, conv_kernel_size - 1, conv_channels, il);

        var state = b.buildRs(inp, ssm_states_all, @intCast(hp.nEmbdS()), @intCast(n_seqs), null);
        state = c.ggml_reshape_4d(ctx, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        b.cb(state, "state_predelta", il);

        const conv_output_proper: *Tensor = c.ggml_ssm_conv(ctx, conv_input, conv_kernel);
        b.cb(conv_output_proper, "conv_output_raw", il);

        const conv_output_silu: *Tensor = c.ggml_silu(ctx, conv_output_proper);
        b.cb(conv_output_silu, "conv_output_silu", il);

        const conv_qkv_mix = conv_output_silu;

        const ty = conv_qkv_mix.type;
        const nb1_qkv = c.ggml_row_size(ty, conv_channels);
        const nst: usize = @intCast(n_seq_tokens);

        // The convolved Q, K and V.
        var q_conv: *Tensor = c.ggml_view_4d(ctx, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_k_dim), nb1_qkv, nb1_qkv * nst, 0);
        var k_conv: *Tensor = c.ggml_view_4d(ctx, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_k_dim), nb1_qkv, nb1_qkv * nst, @as(usize, @intCast(head_k_dim * num_k_heads)) * c.ggml_element_size(conv_qkv_mix));
        const v_conv: *Tensor = c.ggml_view_4d(ctx, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs, c.ggml_row_size(ty, head_v_dim), nb1_qkv, nb1_qkv * nst, c.ggml_row_size(ty, 2 * head_k_dim * num_k_heads));

        b.cb(q_conv, "q_conv", il);
        b.cb(k_conv, "k_conv", il);
        b.cb(v_conv, "v_conv", il);

        const eps_norm = hp.f_norm_rms_eps;

        q_conv = delta_net.buildGdnL2Norm(ctx, q_conv, eps_norm);
        k_conv = delta_net.buildGdnL2Norm(ctx, k_conv, eps_norm);

        // Repeated to match shapes when the key and value head counts
        // differ; the fused delta net does not need it.
        if (num_k_heads != num_v_heads and (!b.cparams.fused_gdn_ar or !b.cparams.fused_gdn_ch)) {
            impl.assert(@rem(num_v_heads, num_k_heads) == 0, src, 946, "num_v_heads % num_k_heads == 0");
            q_conv = c.ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
            k_conv = c.ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        }

        b.cb(q_conv, "q_conv_predelta", il);
        b.cb(k_conv, "k_conv_predelta", il);
        b.cb(v_conv, "v_conv_predelta", il);

        const output = delta_net.buildRecurrentAttn(b, inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

        const z_2d: *Tensor = c.ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

        // The gated norm, self.norm(core_attn_out, z) in the reference.
        const attn_out_norm = self.buildNormGated(output, layer.ssm_norm, z_2d, il);

        const final_output: *Tensor = c.ggml_reshape_3d(ctx, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
        b.cb(final_output, "final_output", il);

        var cur = b.buildLoraMm(layer.ssm_out.?, final_output, layer.ssm_out_s);
        b.cb(cur, "linear_attn_out", il);

        cur = c.ggml_reshape_2d(ctx, cur, b.n_embd, n_seq_tokens * n_seqs);

        return cur;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_layer_ffn`
    /// (models/qwen4exp.cpp:974 @b29c606e2): the routed experts and the
    /// gated shared expert.
    fn buildLayerFfn(self: *Graph, cur_: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const layer = &self.model.layers[@intCast(il)];

        impl.assert(layer.ffn_gate_inp != null, src, 974, "model.layers[il].ffn_gate_inp != nullptr");

        const moe_out = b.buildMoeFfn(cur_, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = null,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .silu,
            .norm_w = true,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .softmax,
            .probs_in = null,
            .gate_up_exps = layer.ffn_gate_up_exps,
            .up_exps_s = layer.ffn_up_exps_s,
            .gate_exps_s = layer.ffn_gate_exps_s,
            .down_exps_s = layer.ffn_down_exps_s,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        var cur = cur_;
        // The shared experts, as in the Qwen3Next reference.
        if (layer.ffn_up_shexp != null) {
            var ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
            b.cb(ffn_shexp, "ffn_shexp", il);

            // The shared expert has its own sigmoid gate
            // (ffn_gate_inp_shexp, one value per token).
            var shared_gate = b.buildLoraMm(layer.ffn_gate_inp_shexp.?, cur, null);
            b.cb(shared_gate, "shared_expert_gate", il);

            shared_gate = c.ggml_sigmoid(ctx, shared_gate);
            b.cb(shared_gate, "shared_expert_gate_sigmoid", il);

            ffn_shexp = c.ggml_mul(ctx, ffn_shexp, shared_gate);
            b.cb(ffn_shexp, "ffn_shexp_gated", il);

            cur = c.ggml_add(ctx, moe_out, ffn_shexp);
            b.cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }

        return cur;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_conv_state_at`
    /// (models/qwen4exp.cpp:1117 @b29c606e2): a conv history read out of
    /// its own recurrent row, and the new tail written back. The shared
    /// build_conv_state cannot do this: qwen4exp has two such rows per
    /// layer.
    fn buildConvStateAt(self: *Graph, inp: *graph.InpRs, conv_states_all: *Tensor, x: *Tensor, state_cols: i64, channels: i64, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const mctx_cur = inp.mctx;

        const kv_head: u32 = mctx_cur.getHead();

        const n_seqs: i64 = b.ubatch.n_seqs;
        const row_total: i64 = conv_states_all.ne[0];

        // The row is exactly this convolution's state, so the gather is
        // reused whole.
        impl.assert(state_cols * channels == row_total, src, 1131, "state_cols * channels == row_total");

        const rows: *Tensor = for (self.rs_rows.items) |e| {
            if (e.key == conv_states_all) break e.rows;
        } else blk: {
            const r = b.buildRs(inp, conv_states_all, @intCast(row_total), @intCast(n_seqs), null);
            self.rs_rows.append(allocator, .{ .key = conv_states_all, .rows = r }) catch oom();
            break :blk r;
        };

        const state: *Tensor = c.ggml_reshape_3d(ctx, rows, state_cols, channels, n_seqs);
        b.cb(state, "conv_state_at", il);

        const conv_input: *Tensor = c.ggml_concat(ctx, state, c.ggml_transpose(ctx, x), 0);

        // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last state_cols columns kept
        // once per rollback slot, slot s ending s tokens earlier, so a
        // rollback of s tokens reads a history that never saw them.
        const row_size = c.ggml_row_size(conv_states_all.type, row_total);
        const mem_size: u32 = mctx_cur.getSize();

        const n_slots: i64 = @as(i64, b.cparams.n_rs_seq) + 1;

        var slot: i64 = 0;
        while (slot < n_slots) : (slot += 1) {
            const s_idx: i64 = @max(0, conv_input.ne[0] - state_cols - slot);

            const tail: *Tensor = c.ggml_view_3d(ctx, conv_input, state_cols, channels, n_seqs, conv_input.nb[1], conv_input.nb[2], c.ggml_row_size(conv_input.type, s_idx));

            const dst: *Tensor = c.ggml_view_2d(ctx, conv_states_all, state_cols * channels, n_seqs, conv_states_all.nb[1], @as(usize, @intCast(slot * mem_size + kv_head)) * row_size);

            c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, c.ggml_cont(ctx, tail), dst));
        }

        return conv_input;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_inp_ple`
    /// (models/qwen4exp.cpp:1171 @b29c606e2): the PLE rows' input and the
    /// gathered embedding, `[ple_head_dim * ple_n_heads, n_tokens]`.
    fn buildInpPle(self: *Graph, mctx_hyb: *const HybridIdxContext) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const n_heads: i64 = hp.ple_n_heads;

        // The attention cells see every ubatch, whatever the layer types.
        const ple_inp = allocator.create(InpPle) catch oom();
        ple_inp.* = .{ .pmodel = self.model, .mctx = mctx_hyb.getAttn() };

        ple_inp.rows = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n_heads * b.n_tokens);
        c.ggml_set_input(ple_inp.rows);
        const rows = ple_inp.rows.?;
        _ = b.res.addInput(InpPle, ple_inp);

        // Gathered, then the heads flattened: get_rows lays the head
        // dimension out slowest, as the reference does.
        var emb: *Tensor = c.ggml_get_rows(ctx, self.model.per_layer_tok_embd, rows);
        emb = c.ggml_reshape_2d(ctx, emb, @as(i64, hp.ple_head_dim) * n_heads, b.n_tokens);
        b.cb(emb, "ple_embd", -1);

        return emb;
    }

    /// The `grouped_norm` lambda of `build_ple`: both norms group over one
    /// hc stream, with a weight over the whole hc*n_embd layout.
    fn groupedNorm(self: *Graph, x: *Tensor, w: ?*Tensor) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hc: i64 = b.hparams.dsv4_hc_mult;
        const hc_dim = hc * b.n_embd;
        var t: *Tensor = c.ggml_reshape_3d(ctx, x, b.n_embd, hc, b.n_tokens);
        t = c.ggml_rms_norm(ctx, t, b.hparams.f_norm_rms_eps);
        t = c.ggml_reshape_2d(ctx, t, hc_dim, b.n_tokens);
        t = c.ggml_mul(ctx, t, w);
        return c.ggml_reshape_3d(ctx, t, b.n_embd, hc, b.n_tokens);
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_ple` (models/qwen4exp.cpp:1192
    /// @b29c606e2): the PLE embedding gated against the residual, through a
    /// dilated causal convolution, added to every stream.
    fn buildPle(self: *Graph, inp: *graph.InpRs, emb: *Tensor, hidden: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const layer = &self.model.layers[@intCast(il)];
        const hc: i64 = hp.dsv4_hc_mult;
        const hc_dim = hc * b.n_embd;
        const n_tokens = b.n_tokens;

        var key = b.buildLoraMm(layer.ple_key.?, emb, null);
        const value = b.buildLoraMm(layer.ple_value.?, emb, null);

        key = self.groupedNorm(key, layer.ple_norm_key);
        const query = self.groupedNorm(hidden, layer.ple_norm_query);

        // A per-stream dot product, then a signed square root before the
        // sigmoid.
        var s: *Tensor = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, key, query));
        s = c.ggml_scale(ctx, s, 1.0 / impl.sqrtf(@floatFromInt(b.n_embd)));

        const mag: *Tensor = c.ggml_sqrt(ctx, c.ggml_clamp(ctx, c.ggml_abs(ctx, s), 1e-6, std.math.inf(f32)));
        const gate: *Tensor = c.ggml_sigmoid(ctx, c.ggml_mul(ctx, c.ggml_sgn(ctx, s), mag));
        b.cb(gate, "ple_gate", il);

        // The [n_embd, 1, T] value broadcast across the hc streams, scaled
        // by the gate.
        var v3: *Tensor = c.ggml_reshape_3d(ctx, value, b.n_embd, 1, n_tokens);
        v3 = c.ggml_repeat_4d(ctx, v3, b.n_embd, hc, n_tokens, 1);

        const gated: *Tensor = c.ggml_mul(ctx, v3, gate);
        b.cb(gated, "ple_gated_value", il);

        var normalized = self.groupedNorm(c.ggml_reshape_2d(ctx, gated, hc_dim, n_tokens), layer.ple_norm_conv);
        normalized = c.ggml_reshape_2d(ctx, normalized, hc_dim, n_tokens);

        // A depthwise causal conv, dilated by the n-gram size, as a sum of
        // shifted copies (ggml_conv_1d_dw is documented as unreliable):
        //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
        // The earlier ubatches' history is prepended, so a chunked prefill
        // matches a single-shot one.
        const kern: i64 = hp.ple_conv_kernel;
        const dil: i64 = hp.ple_ngram_size;
        const hist = (kern - 1) * dil;

        // The conv history is per sequence, so the input carries the
        // sequence axis too.
        const n_seqs: i64 = b.ubatch.n_seqs;
        const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

        // [hist + n_seq_tokens, hc_dim, n_seqs], the tokens on ne[0].
        const padded = self.buildConvStateAt(inp, inp.mctx.getPL(il).?, c.ggml_reshape_3d(ctx, normalized, hc_dim, n_seq_tokens, n_seqs), hist, hc_dim, il);

        const conv1d = layer.ple_conv1d.?;
        var conv_out: ?*Tensor = null;
        var k: i64 = 0;
        while (k < kern) : (k += 1) {
            // Tap k reads (kern-1-k)*dilation positions back.
            const start = hist - (kern - 1 - k) * dil;

            const shifted: *Tensor = c.ggml_cont(ctx, c.ggml_transpose(ctx, c.ggml_view_3d(ctx, padded, n_seq_tokens, hc_dim, n_seqs, padded.nb[1], padded.nb[2], c.ggml_row_size(padded.type, start))));

            // Column k of the [kern, hc_dim] kernel is one weight per
            // channel.
            var wk: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, conv1d, 1, hc_dim, conv1d.nb[1], @as(usize, @intCast(k)) * conv1d.nb[0]));
            // This kernel keeps the file's type, so it is cast before it
            // multiplies an f32 activation.
            wk = c.ggml_reshape_1d(ctx, wk, hc_dim);
            if (wk.type != c.GGML_TYPE_F32) {
                wk = c.ggml_cast(ctx, wk, c.GGML_TYPE_F32);
            }

            const term: *Tensor = c.ggml_mul(ctx, shifted, wk);
            conv_out = if (conv_out) |co| c.ggml_add(ctx, co, term) else term;
        }

        var co: *Tensor = c.ggml_silu(ctx, conv_out);
        co = c.ggml_reshape_3d(ctx, c.ggml_cont(ctx, co), b.n_embd, hc, n_tokens);
        b.cb(co, "ple_conv_out", il);

        return c.ggml_add(ctx, hidden, c.ggml_add(ctx, gated, co));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
