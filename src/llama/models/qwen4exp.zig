//! Qwen4-Exp: the Qwen3.5 hybrid -- gated delta-net linear attention with a
//! full gated attention layer every few -- with hyper-connections in place
//! of every layer norm, QSA sparse attention on the full layers, a PLE
//! n-gram hash embedding, a softmax-routed mixture of experts with a gated
//! shared expert, and an MTP draft head.
//!
//! Hyper-connections keep `hc` parallel residual streams, `[n_embd, hc, T]`:
//! before each block a low-rank gated mixer collapses them into one input
//! and computes how the block's output is scattered back. QSA scores whole
//! pools of `compress_ratio` cells in sequence order with mean-pooled
//! indexer keys, which persist in the indexer cache beside the raw keys,
//! and lets each query attend only to the best pools and the incomplete
//! tail, through a mask. PLE hashes each token's preceding n-grams on the
//! host into rows of a shared table, gates the gathered embedding against
//! the residual, and runs a dilated causal convolution whose history lives
//! in the recurrent cache.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/qwen4exp.cpp` at v0.8.0 (`8642541b2`),
//! with its declaration in `llama.cpp/src/models/models.h`:
//! `qwen4exp_require_nonzero`, `qwen4exp_require_arr_len`,
//! `llama_model_qwen4exp`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph` with every member, its `graph_mtp`, and
//! the two inputs, `llm_graph_input_kpool` and
//! `llm_graph_input_qwen4exp_ple`.
//!
//! The C++ graph is a class whose map -- one recurrent-row gather per cache
//! tensor -- lives as long as one build; here it is a field of `Graph`,
//! which lives as long as `buildArchGraph`. `graph_mtp` derives from
//! `graph` through its `no_build` constructor, to reuse the helpers alone:
//! here it is `Graph.buildMtp` beside `Graph.build`.

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

/// `LLAMA_MAX_PLE_NGRAM` (llama-hparams.h:13 @8642541b2).
const max_ple_ngram: u32 = 8;
/// `LLAMA_MAX_PLE_HEADS` (llama-hparams.h:14 @8642541b2).
const max_ple_heads: u32 = 64;

/// `LLAMA_TOKEN_NULL`.
const token_null: i32 = -1;

/// Ports `qwen4exp_require_nonzero` (models/qwen4exp.cpp:10 @8642541b2):
/// bad metadata must be catchable, where a `GGML_ASSERT` would abort the
/// whole process.
fn requireNonzero(ml: *const loader.Loader, kid: arch_mod.Kv, value: u32, exc: *Exception) Thrown!void {
    if (value == 0) {
        const nm = ml.meta.name(kid);
        return exc.throw(.runtime_error, "{s} must be greater than zero, got {d}", .{ nm.slice(), value });
    }
}

/// Ports `qwen4exp_require_arr_len` (models/qwen4exp.cpp:17 @8642541b2):
/// `get_arr` copies a short array as it is, leaving a zero tail the n-gram
/// hash would silently drop.
fn requireArrLen(ml: *const loader.Loader, kid: arch_mod.Kv, n_min: u32, exc: *Exception) Thrown!void {
    const n_arr = (try ml.meta.getArrN(kid, true, exc)) orelse 0;
    if (n_arr < n_min) {
        const nm = ml.meta.name(kid);
        return exc.throw(.runtime_error, "{s} has {d} entries, but at least {d} are required", .{ nm.slice(), n_arr, n_min });
    }
}

/// Ports `llama_model_qwen4exp`'s `load_arch_hparams` (models/qwen4exp.cpp:26 @8642541b2).
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

    // QSA pools the indexer keys of blocks of compress_ratio cells, one
    // block size for the whole model.
    hp.indexer_kpool = 0;
    var ir: u32 = 0;
    while (ir < hp.n_layer_all) : (ir += 1) {
        const r = hp.dsv4_compress_ratios[ir];
        if (r == 0) continue;
        if (hp.indexer_kpool != 0 and r != hp.indexer_kpool) {
            return exc.throw(.runtime_error, "QSA layers must share one compress ratio, got {d} and {d}", .{ hp.indexer_kpool, r });
        }
        hp.indexer_kpool = r;
    }
    if (hp.indexer_kpool == 1 or (hp.indexer_kpool > 0 and hp.indexer_top_k % hp.indexer_kpool != 0)) {
        return exc.throw(.runtime_error, "QSA needs a compress ratio above 1 that divides the budget, got {d} and {d}", .{ hp.indexer_kpool, hp.indexer_top_k });
    }
    // The reference groups the visible tokens in cache order and always
    // keeps the tail.
    hp.indexer_kpool_row = 2; // raw key | pooled key
    hp.indexer_kpool_by_order = true;
    hp.indexer_kpool_select_tail = true;

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
        hp.ple_embd_token = @bitCast(if (hp.ple_image_token_id != 0) hp.ple_image_token_id else hp.ple_eos_token_id);
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

/// Ports `llama_model_qwen4exp`'s `load_arch_tensors` (models/qwen4exp.cpp:174 @8642541b2).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    const hc: i64 = hp.dsv4_hc_mult;
    const hc_dim = hc * l.n_embd;
    const hc_lr: i64 = hp.hc_low_rank;

    const nf = self.nextnFlags(ml, .hc_attn_norm);
    const trunk_flags: u32 = nf.trunk;
    const mtp_flags: u32 = nf.mtp;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // There is no output_norm: the final hyper-connection mixer carries it.
    // The gammas load as [n_embd, hc], so the grouped norm multiplies them
    // without a graph reshape.
    self.hc_head_norm = try self.createTensor(.{ .tensor = .hc_head_norm, .suffix = "weight" }, &.{ l.n_embd, hc }, trunk_flags | loader.tensor_allow_reshape, exc);
    self.hc_head_down = try self.createTensor(.{ .tensor = .hc_head_down, .suffix = "weight" }, &.{ hc_dim, hc_lr }, trunk_flags, exc);
    self.hc_head_up = try self.createTensor(.{ .tensor = .hc_head_up, .suffix = "weight" }, &.{ hc_lr, hc_dim }, trunk_flags, exc);

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
        try loadBlock(self, il, trunk_flags, exc);
    }

    // The MTP block: one full-attention QSA layer fed by
    // [enorm(e) ; hnorm(h)_s] -> eh_proj per hc stream.
    il = l.n_layer;
    while (il < l.n_layer_all) : (il += 1) {
        try loadBlock(self, il, mtp_flags, exc);

        const nextn = &self.layers[@intCast(il)].nextn;
        nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = il }, &.{ 2 * l.n_embd, l.n_embd }, mtp_flags, exc);
        nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = il }, &.{l.n_embd}, mtp_flags, exc);
        // RMS per hc stream of the trunk residual, so the gammas load as
        // [n_embd, hc] like the mixer norms.
        nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc }, mtp_flags | loader.tensor_allow_reshape, exc);
        nextn.hc_head_norm = try self.createTensor(.{ .tensor = .nextn_hc_head_norm, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc }, mtp_flags | loader.tensor_allow_reshape, exc);
        nextn.hc_head_down = try self.createTensor(.{ .tensor = .nextn_hc_head_down, .suffix = "weight", .bid = il }, &.{ hc_dim, hc_lr }, mtp_flags, exc);
        nextn.hc_head_up = try self.createTensor(.{ .tensor = .nextn_hc_head_up, .suffix = "weight", .bid = il }, &.{ hc_lr, hc_dim }, mtp_flags, exc);
    }
}

/// Ports the `create_merged` lambda (models/qwen4exp.cpp:235 @8642541b2) of
/// `load_block`: for matmuls that read the same input, the merged `[a | b]`
/// weight the converter writes, else the separate ones.
fn createMerged(self: *Model, il: i32, flags: u32, t_merged: arch_mod.Tensor, t_a: arch_mod.Tensor, t_b: arch_mod.Tensor, n_in: i64, n_a: i64, n_b: i64, merged: *?*Tensor, a: *?*Tensor, b: *?*Tensor, exc: *Exception) Thrown!void {
    if (flags & loader.tensor_skip != 0) {
        const skip = loader.tensor_not_required | loader.tensor_skip;
        _ = try self.createTensor(.{ .tensor = t_merged, .suffix = "weight", .bid = il }, &.{ n_in, n_a + n_b }, skip | loader.tensor_skip_if_virtual, exc);
        _ = try self.createTensor(.{ .tensor = t_a, .suffix = "weight", .bid = il }, &.{ n_in, n_a }, skip, exc);
        _ = try self.createTensor(.{ .tensor = t_b, .suffix = "weight", .bid = il }, &.{ n_in, n_b }, skip, exc);
        return;
    }
    merged.* = try self.createTensor(.{ .tensor = t_merged, .suffix = "weight", .bid = il }, &.{ n_in, n_a + n_b }, flags | loader.tensor_not_required | loader.tensor_skip_if_virtual, exc);
    if (merged.* == null) {
        a.* = try self.createTensor(.{ .tensor = t_a, .suffix = "weight", .bid = il }, &.{ n_in, n_a }, flags, exc);
        b.* = try self.createTensor(.{ .tensor = t_b, .suffix = "weight", .bid = il }, &.{ n_in, n_b }, flags, exc);
    }
}

/// Ports the `load_block` lambda (models/qwen4exp.cpp:220 @8642541b2) of
/// `load_arch_tensors`: one trunk or MTP block, every tensor created with
/// `flags`.
fn loadBlock(self: *Model, il: i32, flags: u32, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const layer = &self.layers[@intCast(il)];
    const ilu: u32 = @intCast(il);

    const hc: i64 = hp.dsv4_hc_mult;
    const hc_dim = hc * l.n_embd;
    const hc_lr: i64 = hp.hc_low_rank;

    const n_ff_exp: i64 = if (hp.nFfExp(0) != 0) hp.nFfExp(0) else @divTrunc(l.n_ff, l.n_expert_used);
    const n_ff_shexp: i64 = if (hp.n_ff_shexp != 0) hp.n_ff_shexp else l.n_ff;

    const head_k_dim: i64 = hp.ssm_d_state;
    const head_v_dim: i64 = hp.ssm_d_state;
    const n_k_heads: i64 = hp.ssm_n_group;
    const n_v_heads: i64 = hp.ssm_dt_rank;
    const key_dim = head_k_dim * n_k_heads;
    const value_dim = head_v_dim * n_v_heads;
    const conv_dim = key_dim * 2 + value_dim;

    // Two HC modules per layer: before the token mixer and before the MoE.
    layer.hc_attn_norm = try self.createTensor(.{ .tensor = .hc_attn_norm, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc }, flags | loader.tensor_allow_reshape, exc);
    layer.hc_attn_up = try self.createTensor(.{ .tensor = .hc_attn_up, .suffix = "weight", .bid = il }, &.{ hc_lr, hc_dim }, flags, exc);
    try createMerged(self, il, flags, .hc_attn_down_inject, .hc_attn_down, .hc_attn_inject, hc_dim, hc_lr, hc, &layer.hc_attn_down_inject, &layer.hc_attn_down, &layer.hc_attn_inject, exc);
    layer.hc_ffn_norm = try self.createTensor(.{ .tensor = .hc_ffn_norm, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc }, flags | loader.tensor_allow_reshape, exc);
    layer.hc_ffn_up = try self.createTensor(.{ .tensor = .hc_ffn_up, .suffix = "weight", .bid = il }, &.{ hc_lr, hc_dim }, flags, exc);
    try createMerged(self, il, flags, .hc_ffn_down_inject, .hc_ffn_down, .hc_ffn_inject, hc_dim, hc_lr, hc, &layer.hc_ffn_down_inject, &layer.hc_ffn_down, &layer.hc_ffn_inject, exc);

    if (!hp.isRecr(ilu)) {
        // Full attention: wq holds [q|gate] interleaved per head.
        try self.createTensorQkv(layer, il, l.n_embd, l.n_embd_head_k * l.n_head * 2, l.n_embd_k_gqa, l.n_embd_v_gqa, flags, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, flags, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = il }, &.{l.n_embd_head_k}, flags, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = il }, &.{l.n_embd_head_k}, flags, exc);

        const idx_dim: i64 = hp.indexer_head_size;
        layer.index_q_proj = try self.createTensor(.{ .tensor = .indexer_q_proj, .suffix = "weight", .bid = il }, &.{ l.n_embd, @as(i64, hp.indexer_n_head) * idx_dim }, flags, exc);
        layer.index_k_proj = try self.createTensor(.{ .tensor = .indexer_k_proj, .suffix = "weight", .bid = il }, &.{ l.n_embd, idx_dim }, flags, exc);
        layer.index_q_norm = try self.createTensor(.{ .tensor = .indexer_q_norm, .suffix = "weight", .bid = il }, &.{idx_dim}, flags, exc);
        layer.index_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = il }, &.{idx_dim}, flags, exc);
    } else {
        try createMerged(self, il, flags, .attn_qkv_gate, .attn_qkv, .attn_gate, l.n_embd, key_dim * 2 + value_dim, value_dim, &layer.wqkv_z, &layer.wqkv, &layer.wqkv_gate, exc);
        layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = il }, &.{ hp.ssm_d_conv, conv_dim }, flags, exc);
        layer.ssm_dt = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = il }, &.{hp.ssm_dt_rank}, flags, exc);
        layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a_noscan, .bid = il }, &.{hp.ssm_dt_rank}, flags, exc);
        try createMerged(self, il, flags, .ssm_beta_alpha_cat, .ssm_beta, .ssm_alpha, l.n_embd, n_v_heads, n_v_heads, &layer.ssm_beta_alpha_cat, &layer.ssm_beta, &layer.ssm_alpha, exc);
        layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = il }, &.{head_v_dim}, flags, exc);
        layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = il }, &.{ value_dim, l.n_embd }, flags, exc);
    }

    if (hp.isPle(ilu)) {
        layer.ple_key = try self.createTensor(.{ .tensor = .ple_key, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc_dim }, flags, exc);
        layer.ple_value = try self.createTensor(.{ .tensor = .ple_value, .suffix = "weight", .bid = il }, &.{ l.n_embd, l.n_embd }, flags, exc);
        layer.ple_norm_key = try self.createTensor(.{ .tensor = .ple_norm_key, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc }, flags | loader.tensor_allow_reshape, exc);
        layer.ple_norm_query = try self.createTensor(.{ .tensor = .ple_norm_query, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc }, flags | loader.tensor_allow_reshape, exc);
        layer.ple_norm_conv = try self.createTensor(.{ .tensor = .ple_norm_conv, .suffix = "weight", .bid = il }, &.{ l.n_embd, hc }, flags | loader.tensor_allow_reshape, exc);
        layer.ple_conv1d = try self.createTensor(.{ .tensor = .ple_conv1d, .suffix = "weight", .bid = il }, &.{ hp.ple_conv_kernel, hc_dim }, flags, exc);
    }

    layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = il }, &.{ l.n_embd, l.n_expert }, flags, exc);
    layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = il }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
    try self.createTensorGateUpExps(layer, il, l.n_embd, n_ff_exp, l.n_expert, flags, exc);

    layer.ffn_gate_inp_shexp = try self.createTensor(.{ .tensor = .ffn_gate_inp_shexp, .suffix = "weight", .bid = il }, &.{l.n_embd}, flags, exc);
    layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = il }, &.{ l.n_embd, n_ff_shexp }, flags, exc);
    layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = il }, &.{ l.n_embd, n_ff_shexp }, flags, exc);
    layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = il }, &.{ n_ff_shexp, l.n_embd }, flags, exc);
}

/// Ports `llama_model_qwen4exp`'s `build_arch_graph` (models/qwen4exp.cpp:324 @8642541b2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    var g: Graph = .{ .model = model, .b = b };
    defer g.deinit();
    if (params.gtype == .decoder_mtp) {
        g.buildMtp();
    } else {
        g.build();
    }
}

/// Ports `llama_model_qwen4exp`'s `llm_graph_input_kpool` (models/qwen4exp.cpp:722
/// @8642541b2): the k-pool inputs the QSA layers share -- blocks of
/// `compress_ratio` cells in sequence order, see `llama_memory_hybrid_idx`.
pub const InpKpool = struct {
    /// I64 `[n_tokens]`.
    k_idxs: ?*Tensor = null,
    /// I32 `[n_pool]`: the cell caching each block's pooled key.
    pool_cells: ?*Tensor = null,
    /// I32 `[kpool, n_pool]`: member cells per block, the `n_kv` sentinel
    /// for the padded blocks.
    pool_idxs: ?*Tensor = null,
    /// F16 `[n_pool, n_tokens]`.
    pool_mask: ?*Tensor = null,
    /// I32 `[kpool - 1, n_tokens]`.
    tail_idxs: ?*Tensor = null,
    /// I32 `[kpool, n_new]`: members of the blocks to re-pool this ubatch.
    new_pool_idxs: ?*Tensor = null,
    /// I32 `[kpool*n_new]`: a view of `new_pool_idxs`, shared by the QSA
    /// layers.
    new_pool_flat: ?*Tensor = null,
    /// I64 `[n_new]`: the cell to write each new pooled key into.
    new_pool_rep: ?*Tensor = null,
    /// I32 `[4*n_new]`: the M-RoPE position of each new block's first
    /// member.
    new_pool_pos: ?*Tensor = null,

    mctx: *const HybridIdxContext,
    kpool: u32,
    /// Padded to a stable bound, never below 1.
    n_new: u32 = 0,
    n_sel: u32 = 0,
    n_kv: u32 = 0,

    /// Ports `llm_graph_input_kpool`'s `set_input` (models/qwen4exp.cpp:727 @8642541b2).
    pub fn setInput(self: *InpKpool, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        self.mctx.getIdx().?.setInputKIdxs(self.k_idxs.?, ubatch);
        self.mctx.setInputKpool(self.pool_cells.?, self.pool_idxs.?, self.pool_mask.?, self.tail_idxs.?, null, self.new_pool_idxs.?, self.new_pool_rep.?, ubatch, self.new_pool_pos);
    }

    /// Ports `llm_graph_input_kpool`'s `can_reuse` (models/qwen4exp.cpp:733 @8642541b2).
    pub fn canReuse(self: *InpKpool, params: *const graph.Params) bool {
        self.mctx = HybridIdxContext.fromBase(params.mctx.?.cast(HybridContext));

        const idx = self.mctx.getIdx() orelse return false;

        var res = true;

        res = res and self.k_idxs.?.ne[0] == params.ubatch.n_tokens;
        res = res and self.pool_cells.?.ne[0] == self.mctx.getNKpool();
        res = res and self.pool_mask.?.ne[1] == params.ubatch.n_tokens;
        res = res and self.tail_idxs.?.ne[1] == params.ubatch.n_tokens;
        // The scatter mask's shape follows n_kv.
        res = res and self.n_kv == idx.getNKv();
        res = res and self.n_new == self.mctx.getNKpoolNew();

        return res;
    }
};

/// Ports `llm_graph_input_qwen4exp_ple` (models/qwen4exp.cpp:1299 @8642541b2): the
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

    /// Ports `llm_graph_input_qwen4exp_ple`'s `can_reuse` (models/qwen4exp.cpp:1307 @8642541b2).
    pub fn canReuse(self: *InpPle, params: *const graph.Params) bool {
        self.mctx = HybridIdxContext.fromBase(params.mctx.?.cast(HybridContext)).getAttn();
        return self.rows.?.ne[0] == @as(i64, self.pmodel.hparams.ple_n_heads) * @as(i64, params.ubatch.n_tokens);
    }

    /// Ports `llm_graph_input_qwen4exp_ple`'s `set_input` (models/qwen4exp.cpp:1323 @8642541b2).
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

        impl.assert(self.mctx != null, src, 1346, "mctx != nullptr");

        var i: i64 = 0;
        while (i < n_tokens) : (i += 1) {
            // The preceding tokens would be ambiguous; see get_prev_tokens.
            impl.assert(ubatch.n_seq_id[@intCast(i)] == 1, src, 1350, "ubatch->n_seq_id[i] == 1 && \"PLE n-gram embeddings do not support tokens shared by multiple sequences\"");
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
            // The `tok_of` lambda: an embedding row, of an embedding or a
            // mixed ubatch, stands in the image token.
            const is_embd = ubatch.token == null or (ubatch.isMixed() and ubatch.type[@intCast(i)] != 0);
            ctx[0] = if (is_embd) img_tok else ubatch.token[@intCast(i)];
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

        {
            const ple = self.pmodel.per_layer_tok_embd;

            const prefetch = self.pmodel.canPrefetch(ple);
            if (prefetch) {
                impl.prefetchRows(ple.?, idx.ptr, idx.len);
            }
        }

        c.ggml_backend_tensor_set(self.rows, idx.ptr, 0, idx.len * c.ggml_element_size(self.rows));
    }
};

/// Ports `hc_stream_mean` (models/qwen4exp.cpp:332 @8642541b2): DFlash
/// drafters read the layer inputs as `[n_embd, T]` rows, so the streams
/// collapse.
fn hcStreamMean(ctx: *c.ggml_context, x: *Tensor) *Tensor {
    const n_embd = x.ne[0];
    const hc = x.ne[1];
    const nt = x.ne[2];

    var acc: *Tensor = c.ggml_view_2d(ctx, x, n_embd, nt, x.nb[2], 0);
    var s: i64 = 1;
    while (s < hc) : (s += 1) {
        acc = c.ggml_add(ctx, acc, c.ggml_view_2d(ctx, x, n_embd, nt, x.nb[2], @as(usize, @intCast(s)) * x.nb[1]));
    }

    return c.ggml_scale(ctx, acc, 1.0 / @as(f32, @floatFromInt(hc)));
}

/// Ports `llama_model_qwen4exp`'s `graph` (src/models/models.h:2763
/// @8642541b2) and `graph_mtp` (src/models/models.h:2869 @8642541b2): the builder
/// and the map it keeps for one build.
const Graph = struct {
    model: *const Model,
    b: *const graph.Builder,

    /// build_rs writes the state tensor in place, so one gather per cache
    /// tensor is reused.
    rs_rows: std.ArrayList(struct { key: *Tensor, rows: *Tensor }) = .empty,

    fn deinit(self: *Graph) void {
        self.rs_rows.deinit(allocator);
    }

    /// Mirrors the graph's `hc_mix` (src/models/models.h:2772 @8642541b2).
    const HcMix = struct {
        /// The mixed `[n_embd, T]` stream.
        cur: *Tensor,
        /// The `[hc, T]` scatter weights, null when the inject weights are.
        inject: ?*Tensor,
    };

    /// Ports `llama_model_qwen4exp::graph`'s `build_hc_mix` (models/qwen4exp.cpp:348
    /// @8642541b2): hyper-connections keep `hc` parallel residual streams,
    /// `[n_embd, hc, T]`, in place of the layer norms. Returns the mixed
    /// `[n_embd, T]` stream and the `[hc, T]` scatter weights.
    fn buildHcMix(self: *Graph, x: *Tensor, w_norm: ?*Tensor, w_down: ?*Tensor, w_up: ?*Tensor, w_inject: ?*Tensor, w_down_inject: ?*Tensor, il: c_int) HcMix {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const hc: i64 = hp.dsv4_hc_mult;
        const hc_dim = hc * b.n_embd;
        const nt = x.ne[2];

        // A grouped RMSNorm: reduced over one stream, then all streams
        // scaled by the [n_embd, hc] gamma, which the converter folded to
        // (1 + w).
        var xn: *Tensor = c.ggml_mul(ctx, c.ggml_rms_norm(ctx, x, hp.f_norm_rms_eps), w_norm);
        xn = c.ggml_reshape_2d(ctx, xn, hc_dim, nt);
        b.cb(xn, "hc_norm", il);

        var lo: *Tensor = undefined;
        var inject: ?*Tensor = null;
        if (w_down_inject) |wdi| {
            // One matmul for both: the hc_down rows, then the hc_inject rows.
            const n_lo = wdi.ne[1] - hc;
            var lo_inject = b.buildLoraMm(wdi, xn, null);
            // The combined rows are scaled before the split; both lo and
            // inject use 1/hc.
            lo_inject = c.ggml_scale(ctx, lo_inject, 1.0 / @as(f32, @floatFromInt(hc)));
            lo = c.ggml_view_2d(ctx, lo_inject, n_lo, nt, lo_inject.nb[1], 0);
            inject = c.ggml_view_2d(ctx, lo_inject, hc, nt, lo_inject.nb[1], c.ggml_row_size(lo_inject.type, n_lo));
            b.cb(inject.?, "hc_inject", il);
        } else {
            lo = c.ggml_scale(ctx, b.buildLoraMm(w_down.?, xn, null), 1.0 / @as(f32, @floatFromInt(hc)));
        }
        lo = c.ggml_silu(ctx, lo);
        const gate: *Tensor = b.buildLoraMm(w_up.?, lo, null);
        b.cb(gate, "hc_gate", il);

        var mixed: *Tensor = undefined;
        if (b.cparams.fused_dsv4_hc_pre and il >= 0) {
            // The sigmoid gate and the mean over the streams in one op.
            mixed = c.ggml_dsv4_hc_pre_gated(ctx, c.ggml_reshape_3d(ctx, xn, b.n_embd, hc, nt), c.ggml_reshape_3d(ctx, gate, b.n_embd, hc, nt), 1.0 / @as(f32, @floatFromInt(hc)));
            b.res.addFusedNode(.{ .op = .dsv4_hc_pre, .tensor = mixed, .il = il });
        } else {
            var gated: *Tensor = c.ggml_mul(ctx, xn, c.ggml_sigmoid(ctx, gate));
            gated = c.ggml_reshape_3d(ctx, gated, b.n_embd, hc, nt);

            // The streams collapse to their mean.
            const row = c.ggml_row_size(gated.type, b.n_embd);
            const hcu: usize = @intCast(hc);
            mixed = c.ggml_view_2d(ctx, gated, b.n_embd, nt, row * hcu, 0);
            mixed = c.ggml_cont(ctx, mixed);
            var ci: usize = 1;
            while (ci < hcu) : (ci += 1) {
                const s: *Tensor = c.ggml_view_2d(ctx, gated, b.n_embd, nt, row * hcu, row * ci);
                mixed = c.ggml_add(ctx, mixed, s);
            }
            mixed = c.ggml_scale(ctx, mixed, 1.0 / @as(f32, @floatFromInt(hc)));
        }
        b.cb(mixed, "hc_mixed", il);

        if (w_down_inject == null and w_inject != null) {
            inject = c.ggml_scale(ctx, b.buildLoraMm(w_inject.?, xn, null), 1.0 / @as(f32, @floatFromInt(hc)));
            b.cb(inject.?, "hc_inject", il);
        }

        return .{ .cur = mixed, .inject = inject };
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_hc_combine`
    /// (models/qwen4exp.cpp:417 @8642541b2): the block's output scattered
    /// back into the streams by the injection weights.
    fn buildHcCombine(self: *Graph, residual: *Tensor, block_out: *Tensor, inject: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hc: i64 = b.hparams.dsv4_hc_mult;
        const nt = residual.ne[2];

        // 2*sigmoid centres the scatter weights on 1, so a zero injection
        // is a plain residual add.
        var w: *Tensor = c.ggml_sigmoid(ctx, inject);
        w = c.ggml_scale(ctx, w, 2.0);

        var cur: *Tensor = undefined;
        if (b.cparams.fused_dsv4_hc_post and il >= 0) {
            // An identity comb: every stream adds the same block output,
            // scaled by its own weight.
            cur = c.ggml_dsv4_hc_post(ctx, block_out, residual, w, null);
            b.res.addFusedNode(.{ .op = .dsv4_hc_post, .tensor = cur, .il = il });
        } else {
            w = c.ggml_reshape_3d(ctx, w, 1, hc, nt);

            var bo: *Tensor = c.ggml_reshape_3d(ctx, block_out, b.n_embd, 1, nt);
            bo = c.ggml_repeat_4d(ctx, bo, b.n_embd, hc, nt, 1);

            cur = c.ggml_add(ctx, residual, c.ggml_mul(ctx, bo, w));
        }
        b.cb(cur, "hc_combine", il);

        return cur;
    }

    /// Ports the constructor of `llama_model_qwen4exp`'s `graph`
    /// (models/qwen4exp.cpp:447 @8642541b2).
    fn build(self: *Graph) void {
        const b = self.b;
        const model = self.model;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const hc: i64 = hp.dsv4_hc_mult;

        impl.assert(hp.nEmbdHeadV(0) == hp.nEmbdHeadK(0), src, 451, "hparams.n_embd_head_v() == hparams.n_embd_head_k()");

        var sections: [4]c_int = undefined;
        @memcpy(&sections, hp.rope_sections[0..4]);

        const inp = b.buildInpMemHybrid();

        // qwen4exp always builds llama_memory_hybrid_idx, so this downcast
        // is safe; the indexer cache inside it is absent when the GGUF has
        // no indexer tensors.
        const mctx_hyb = HybridIdxContext.fromBase(inp.mctx);

        const mctx_idx = mctx_hyb.getIdx();
        if (mctx_idx) |idx| {
            impl.assert(idx.getNKv() == inp.mctx.getAttn().getNKv(), src, 465, "mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() && \"the indexer cache must track the attention cache cell for cell\"");
        }

        var ple_emb: ?*Tensor = null;
        if (hp.ple_n_heads > 0) {
            ple_emb = self.buildInpPle(mctx_hyb);
            // ple_emb and build_inp_embd in the same graph split.
            c.ggml_build_forward_expand(b.gf, ple_emb);
        }

        const inpL = b.buildInpEmbd(model.tok_embd.?);
        b.cb(inpL, "model.input_embed", -1);
        c.ggml_build_forward_expand(b.gf, inpL);

        // The QSA layers share one set of k-pool inputs. The CUDA lightning
        // indexer takes 32 or 64 heads and QSA has a few, so it scores with
        // plain ops.
        var inp_kpool: ?*InpKpool = null;
        if (mctx_idx != null and hp.indexer_kpool > 0) {
            inp_kpool = self.buildInpKpool(mctx_hyb);
        }

        const inp_pos = b.buildInpPos();
        const inp_out_ids = b.buildInpOutIds();

        // The wide residual starts as hc identical copies of the embedding.
        var res_hc: *Tensor = c.ggml_repeat_4d(ctx, c.ggml_reshape_3d(ctx, inpL, b.n_embd, 1, b.n_tokens), b.n_embd, hc, b.n_tokens, 1);
        b.cb(res_hc, "hc_init", -1);
        // hc_init in the same graph split as the first layer (-sm tensor).
        c.ggml_build_forward_expand(b.gf, res_hc);

        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];
            const uil: usize = @intCast(il);
            if (uil < b.cparams.embeddings_layer_inp.len and b.cparams.embeddings_layer_inp[uil]) {
                const t = hcStreamMean(ctx, res_hc);
                b.res.t_layer_inp[uil] = t;
                b.cb(t, "layer_inp", il);
                c.ggml_build_forward_expand(b.gf, t);
            }

            if (hp.isPle(@intCast(il))) {
                res_hc = self.buildPle(inp.getRecr(), ple_emb.?, res_hc, il);
            }

            var attn_mix = self.buildHcMix(res_hc, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, layer.hc_attn_down_inject, il);
            var cur = attn_mix.cur;

            c.ggml_build_forward_expand(b.gf, cur);

            if (hp.isRecr(@intCast(il))) {
                cur = self.buildLayerAttnLinear(inp.getRecr(), cur, il);
            } else {
                cur = self.buildLayerAttn(inp.getAttn(), mctx_hyb, inp_kpool, cur, inp_pos, &sections, il);
            }

            if (il == b.n_layer - 1 and b.cropBeforeNextn(inp_out_ids)) {
                // Everything below is per token, so the rows that produce
                // no output go.
                cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
                attn_mix.inject = c.ggml_get_rows(ctx, attn_mix.inject, inp_out_ids);

                res_hc = c.ggml_reshape_2d(ctx, res_hc, b.n_embd * hc, res_hc.ne[2]);
                res_hc = c.ggml_get_rows(ctx, res_hc, inp_out_ids);
                res_hc = c.ggml_reshape_3d(ctx, res_hc, b.n_embd, hc, res_hc.ne[1]);
            }

            res_hc = self.buildHcCombine(res_hc, cur, attn_mix.inject.?, il);

            const ffn_mix = self.buildHcMix(res_hc, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, layer.hc_ffn_down_inject, il);
            cur = ffn_mix.cur;

            cur = self.buildLayerFfn(cur, il);
            b.cb(cur, "ffn_out", il);

            res_hc = self.buildHcCombine(res_hc, cur, ffn_mix.inject.?, il);

            // "l_last" is the layer output's name that build_cvec and
            // imatrix look for.
            b.cb(res_hc, "l_last", il);
        }

        // The MTP head reads the hc-wide residual, before the final mixer.
        const h_nextn: *Tensor = c.ggml_reshape_2d(ctx, res_hc, b.n_embd * hc, res_hc.ne[2]);
        b.res.t_h_nextn = h_nextn;
        b.cb(h_nextn, "h_nextn", -1);
        c.ggml_build_forward_expand(b.gf, h_nextn);

        if (b.cropAfterNextn(inp_out_ids)) {
            res_hc = c.ggml_reshape_2d(ctx, res_hc, b.n_embd * hc, res_hc.ne[2]);
            res_hc = c.ggml_get_rows(ctx, res_hc, inp_out_ids);
            res_hc = c.ggml_reshape_3d(ctx, res_hc, b.n_embd, hc, res_hc.ne[1]);
        }

        // The final mixer is the output norm: there is no separate one.
        var cur = self.buildHcMix(res_hc, model.hc_head_norm, model.hc_head_down, model.hc_head_up, null, null, -1).cur;

        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        cur = b.buildLoraMm(model.output.?, cur, model.output_s);
        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;
    }

    /// Ports the constructor of `llama_model_qwen4exp`'s `graph_mtp`
    /// (models/qwen4exp.cpp:580 @8642541b2): the MTP draft head, one QSA
    /// block after the trunk, fed by the trunk's hc-wide residual.
    fn buildMtp(self: *Graph) void {
        const b = self.b;
        const model = self.model;
        const ctx = b.ctx0;
        const hp = b.hparams;

        impl.assert(hp.n_layer_nextn == 1, src, 582, "hparams.n_layer_nextn == 1 && \"qwen4exp MTP has a single block\"");
        b.ubatch.assertEmbdOrToken(src, 583);

        const hc: i64 = hp.dsv4_hc_mult;
        impl.assert(hp.nEmbdOut() == @as(u32, @truncate(@as(u64, @bitCast(b.n_embd * hc)))), src, 586, "hparams.n_embd_out() == (uint32_t) (n_embd*hc) && \"qwen4exp MTP hidden width mismatch\"");

        const il: c_int = @intCast(hp.nLayer());
        const layer = &model.layers[@intCast(il)];

        impl.assert(layer.nextn.eh_proj != null and layer.nextn.enorm != null and layer.nextn.hnorm != null and layer.nextn.hc_head_norm != null, src, 592, "layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && layer.nextn.hc_head_norm && \"MTP block missing, load the model with MTP enabled\"");

        var sections: [4]c_int = undefined;
        @memcpy(&sections, hp.rope_sections[0..4]);

        const n_embd_inp: i64 = hp.nEmbdInp();
        const n_embd_out: i64 = hp.nEmbdOut();
        const inp = allocator.create(graph.InpEmbdH) catch oom();
        inp.* = .{ .n_embd = n_embd_inp, .n_embd_state = n_embd_out };

        inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
        c.ggml_set_input(inp.tokens);

        inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd_inp, b.n_tokens);
        c.ggml_set_input(inp.embd);

        inp.h = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd_out, b.n_tokens);
        c.ggml_set_input(inp.h);
        _ = c.ggml_set_name(inp.h, "mtp_h_input");

        var tok_embd: *Tensor = undefined;
        if (b.ubatch.token != null) {
            tok_embd = c.ggml_get_rows(ctx, model.tok_embd, inp.tokens);
        } else {
            tok_embd = inp.embd.?;
        }
        b.cb(tok_embd, "mtp_tok_embd", il);

        const h = inp.h.?;

        _ = b.res.addInput(graph.InpEmbdH, inp);

        const inp_hyb = b.buildInpMemHybrid();
        const mctx_hyb = HybridIdxContext.fromBase(inp_hyb.mctx);

        // The draft memory has no recurrent layer, but its input still has
        // to be allocated.
        c.ggml_build_forward_expand(b.gf, inp_hyb.getRecr().s_copy);

        var inp_kpool: ?*InpKpool = null;
        if (mctx_hyb.getIdx() != null and hp.indexer_kpool > 0) {
            impl.assert(mctx_hyb.getIdx().?.getNKv() == mctx_hyb.getAttn().getNKv(), src, 630, "mctx_hyb->get_idx()->get_n_kv() == mctx_hyb->get_attn()->get_n_kv() && \"the indexer cache must track the attention cache cell for cell\"");
            inp_kpool = self.buildInpKpool(mctx_hyb);
        }

        const inp_pos = b.buildInpPos();
        const inp_out_ids = b.buildInpOutIds();

        const h_norm = b.buildNorm(c.ggml_reshape_3d(ctx, h, b.n_embd, hc, b.n_tokens), layer.nextn.hnorm, null, .rms, il);
        b.cb(h_norm, "mtp_hnorm", il);

        var e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
        e_norm = c.ggml_repeat_4d(ctx, c.ggml_reshape_3d(ctx, e_norm, b.n_embd, 1, b.n_tokens), b.n_embd, hc, b.n_tokens, 1);
        b.cb(e_norm, "mtp_enorm", il);

        var res_hc = b.buildLoraMm(layer.nextn.eh_proj.?, c.ggml_concat(ctx, e_norm, h_norm, 0), null); // [n_embd, hc, n_tokens]
        b.cb(res_hc, "mtp_eh_proj", il);

        const attn_mix = self.buildHcMix(res_hc, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, layer.hc_attn_down_inject, il);
        var cur = attn_mix.cur;
        cur = self.buildLayerAttn(inp_hyb.getAttn(), mctx_hyb, inp_kpool, cur, inp_pos, &sections, il);
        res_hc = self.buildHcCombine(res_hc, cur, attn_mix.inject.?, il);

        const ffn_mix = self.buildHcMix(res_hc, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, layer.hc_ffn_down_inject, il);
        cur = ffn_mix.cur;
        cur = self.buildLayerFfn(cur, il);
        res_hc = self.buildHcCombine(res_hc, cur, ffn_mix.inject.?, il);

        // The next draft step reads this residual as its h.
        const flat: *Tensor = c.ggml_reshape_2d(ctx, res_hc, b.n_embd * hc, b.n_tokens);
        const flat_out: *Tensor = c.ggml_get_rows(ctx, flat, inp_out_ids);
        const h_nextn = if (b.cparams.embeddings_nextn_masked) flat_out else flat;
        b.res.t_h_nextn = h_nextn;
        b.cb(h_nextn, "h_nextn", il);
        c.ggml_build_forward_expand(b.gf, h_nextn);

        cur = self.buildHcMix(c.ggml_reshape_3d(ctx, flat_out, b.n_embd, hc, flat_out.ne[1]), layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up, null, null, il).cur;
        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        cur = b.buildLoraMm(model.output.?, cur, model.output_s);
        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_qkvz` (models/qwen4exp.cpp:676
    /// @8642541b2): the mixed QKV projection and the gate, z.
    fn buildQkvz(self: *Graph, input: *Tensor, il: c_int) struct { *Tensor, *Tensor } {
        const b = self.b;
        const layer = &self.model.layers[@intCast(il)];
        const n_seqs: i64 = b.ubatch.n_seqs;
        const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

        if (layer.wqkv_z) |wqkv_z| {
            // qkv and z read the same input: one matmul, then the qkv rows
            // and the z rows.
            const ctx = b.ctx0;
            const hp = b.hparams;
            const qkvz = b.buildLoraMm(wqkv_z, input, null);
            b.cb(qkvz, "linear_attn_qkvz", il);

            const n_z: i64 = @as(i64, hp.ssm_d_state) * hp.ssm_dt_rank;
            const n_qkv = qkvz.ne[0] - n_z;

            const qkv_mixed: *Tensor = c.ggml_view_3d(ctx, qkvz, n_qkv, n_seq_tokens, n_seqs, qkvz.nb[1], qkvz.nb[1] * @as(usize, @intCast(n_seq_tokens)), 0);
            b.cb(qkv_mixed, "linear_attn_qkv_mixed", il);

            const z: *Tensor = c.ggml_view_2d(ctx, qkvz, n_z, n_seq_tokens * n_seqs, qkvz.nb[1], c.ggml_row_size(qkvz.type, n_qkv));
            b.cb(z, "z", il);

            return .{ qkv_mixed, z };
        }

        var qkv_mixed = b.buildLoraMm(layer.wqkv.?, input, layer.wqkv_s);
        qkv_mixed = c.ggml_reshape_3d(b.ctx0, qkv_mixed, qkv_mixed.ne[0], n_seq_tokens, n_seqs);
        b.cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        const z = b.buildLoraMm(layer.wqkv_gate.?, input, layer.wqkv_gate_s);
        b.cb(z, "z", il);

        return .{ qkv_mixed, z };
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_norm_gated`
    /// (models/qwen4exp.cpp:709 @8642541b2): the one numerical difference
    /// from Qwen3.5's delta net, a sigmoid output gate, not a SiLU.
    fn buildNormGated(self: *Graph, input: *Tensor, weights: ?*Tensor, gate: *Tensor, layer: c_int) *Tensor {
        const b = self.b;
        const normalized = b.buildNorm(input, weights, null, .rms, layer);
        const gated: *Tensor = c.ggml_sigmoid(b.ctx0, gate);

        return c.ggml_mul(b.ctx0, normalized, gated);
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_inp_kpool`
    /// (models/qwen4exp.cpp:771 @8642541b2): the one k-pool input set the
    /// QSA layers share.
    fn buildInpKpool(self: *Graph, mctx_hyb: *const HybridIdxContext) *InpKpool {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const mctx_idx = mctx_hyb.getIdx();
        impl.assert(mctx_idx != null, src, 773, "mctx_idx != nullptr");

        const kpool: u32 = hp.indexer_kpool;
        const n_pool: u32 = mctx_hyb.getNKpool();

        const inp = allocator.create(InpKpool) catch oom();
        inp.* = .{ .mctx = mctx_hyb, .kpool = kpool };

        inp.k_idxs = mctx_idx.?.buildInputKIdxs(ctx, b.ubatch);
        inp.pool_cells = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n_pool);
        inp.pool_idxs = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, kpool, n_pool);
        inp.pool_mask = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F16, n_pool, b.n_tokens);
        inp.tail_idxs = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, kpool - 1, b.n_tokens);
        c.ggml_set_input(inp.pool_cells);
        c.ggml_set_input(inp.pool_idxs);
        c.ggml_set_input(inp.pool_mask);
        c.ggml_set_input(inp.tail_idxs);

        // set_input fills them all, so they stay allocated even when no op
        // reads them.
        c.ggml_build_forward_expand(b.gf, inp.pool_cells);
        c.ggml_build_forward_expand(b.gf, inp.pool_idxs);
        c.ggml_build_forward_expand(b.gf, inp.pool_mask);
        c.ggml_build_forward_expand(b.gf, inp.tail_idxs);

        inp.n_kv = mctx_idx.?.getNKv();
        inp.n_new = mctx_hyb.getNKpoolNew();
        // The top blocks plus the tail.
        inp.n_sel = kpool *% @min(n_pool, hp.indexer_top_k / kpool) +% kpool -% 1;

        inp.new_pool_idxs = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, kpool, inp.n_new);
        c.ggml_set_input(inp.new_pool_idxs);
        // One view for all layers: each view of a host input is a separate
        // upload.
        inp.new_pool_flat = c.ggml_reshape_1d(ctx, inp.new_pool_idxs, @as(i64, kpool) * inp.n_new);
        // One scatter row per new pool, each a distinct rep row (see
        // kpool_build_state).
        inp.new_pool_rep = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I64, inp.n_new);
        c.ggml_set_input(inp.new_pool_rep);
        inp.new_pool_pos = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, 4 * @as(i64, inp.n_new));
        c.ggml_set_input(inp.new_pool_pos);

        return b.res.addInput(InpKpool, inp);
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_qsa_sel`
    /// (models/qwen4exp.cpp:816 @8642541b2): QSA attends to the top blocks of
    /// `compress_ratio` cells plus the incomplete tail, like the glm5-next
    /// k-pool indexer. A block is scored by one pooled key: the mean of its
    /// raw indexer keys, normed and rotated to its first member. Returns the
    /// additive mask `[n_kv, n_tokens]` of the kept cells.
    fn buildQsaSel(self: *Graph, mctx_hyb: *const HybridIdxContext, inp_kpool: *InpKpool, cur: *Tensor, inp_pos: *Tensor, sections: *[4]c_int, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const layer = &self.model.layers[@intCast(il)];
        const mctx_idx = mctx_hyb.getIdx().?;
        const n_tokens: i64 = b.n_tokens;

        const idx_dim: i64 = hp.indexer_head_size;
        const n_idx_h: i64 = hp.indexer_n_head;
        const kpool: i64 = inp_kpool.kpool;
        const n_pool: i64 = inp_kpool.pool_cells.?.ne[0];
        const n_new: i64 = inp_kpool.n_new;

        impl.assert(hp.dsv4_compress_ratios[@intCast(il)] == kpool, src, 831, "hparams.dsv4_compress_ratios[il] == kpool");

        // The cache rows store raw key | pooled key: pooling precedes the
        // norm and the rotation, so the raw key gets neither.
        const k_raw = b.buildLoraMm(layer.index_k_proj.?, cur, null);
        b.cb(k_raw, "indexer_k_raw", il);

        const pzero: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, idx_dim, n_tokens), 0.0);
        const packed_k: *Tensor = c.ggml_reshape_3d(ctx, c.ggml_concat(ctx, k_raw, pzero, 0), 2 * idx_dim, 1, n_tokens);
        c.ggml_build_forward_expand(b.gf, mctx_idx.cpyK(ctx, packed_k, inp_kpool.k_idxs.?, il));

        // The raw keys and the persistent pooled slots, see
        // llama_memory_hybrid_idx::mem_idx_stale.
        const kpool_cache = mctx_hyb.getKpoolAccess(ctx, il, idx_dim);

        // Only the blocks this ubatch completes or regroups are pooled.
        var rows: *Tensor = kpool_cache.gatherKeyGate(inp_kpool.new_pool_flat.?);
        rows = c.ggml_reshape_3d(ctx, rows, idx_dim, kpool, n_new);

        // The mean over the members; kpool is small, so summing slices
        // beats a transpose and sum_rows.
        var pooled_new_: ?*Tensor = null;
        var i: i64 = 0;
        while (i < kpool) : (i += 1) {
            const slice: *Tensor = c.ggml_view_2d(ctx, rows, idx_dim, n_new, rows.nb[2], @as(usize, @intCast(i)) * rows.nb[1]);
            pooled_new_ = if (pooled_new_) |p| c.ggml_add(ctx, p, slice) else c.ggml_cont(ctx, slice);
        }
        var pooled_new: *Tensor = c.ggml_scale(ctx, pooled_new_, 1.0 / @as(f32, @floatFromInt(kpool)));
        pooled_new = b.buildNorm(pooled_new, layer.index_k_norm, null, .rms, il);

        pooled_new = c.ggml_reshape_3d(ctx, pooled_new, idx_dim, 1, n_new);
        pooled_new = c.ggml_rope_multi(ctx, pooled_new, inp_kpool.new_pool_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        pooled_new = c.ggml_reshape_2d(ctx, pooled_new, idx_dim, n_new);
        b.cb(pooled_new, "indexer_pool_k_new", il);

        // The fresh pooled keys scattered into their rep rows, then all
        // n_pool of them gathered by cell: the older pools come from the
        // rows earlier ubatches wrote.
        c.ggml_build_forward_expand(b.gf, kpool_cache.scatterPooled(pooled_new, inp_kpool.new_pool_rep.?));
        var pooled: *Tensor = kpool_cache.gatherPooled(inp_kpool.pool_cells.?);
        pooled = c.ggml_reshape_3d(ctx, pooled, idx_dim, 1, n_pool);
        b.cb(pooled, "indexer_k", il);

        var q = b.buildLoraMm(layer.index_q_proj.?, cur, null);
        q = c.ggml_reshape_3d(ctx, q, idx_dim, n_idx_h, n_tokens);
        q = b.buildNorm(q, layer.index_q_norm, null, .rms, il);
        q = c.ggml_rope_multi(ctx, q, inp_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        b.cb(q, "indexer_q", il);

        // The reference sums the rectified head scores unweighted, scaled by
        // 1/sqrt(head_dim): the lightning indexer with every head weight set
        // to that scale.
        const weights: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_idx_h, n_tokens), 1.0 / impl.sqrtf(@floatFromInt(idx_dim)));
        const score: *Tensor = c.ggml_lightning_indexer(ctx, q, pooled, weights, inp_kpool.pool_mask); // [n_pool, n_tokens]
        b.res.addFusedNode(.{ .op = .lightning_indexer, .tensor = score, .il = il });
        b.cb(score, "indexer_score", il);

        const n_top_pool: i64 = @min(n_pool, @as(i64, hp.indexer_top_k / @as(u32, @intCast(kpool))));
        const top_k: *Tensor = c.ggml_top_k(ctx, score, @intCast(n_top_pool)); // [n_top_pool, n_tokens], unordered
        b.cb(top_k, "indexer_top_k", il);

        // Flash attention requires an f16 mask.
        const type_mask: c.ggml_type = if (b.cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;

        // Keep the cells of the live picked pools and the tail, -inf
        // elsewhere. Every retained cell is in the token's sequence, so the
        // kq_mask is not needed.
        if (b.cparams.fused_top_k_mask) {
            impl.assert(kpool * n_top_pool + inp_kpool.tail_idxs.?.ne[0] == inp_kpool.n_sel, src, 896, "kpool*n_top_pool + inp_kpool->tail_idxs->ne[0] == inp_kpool->n_sel");

            const keep: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_1d(ctx, type_mask, inp_kpool.n_kv), 0.0);
            const sel: *Tensor = c.ggml_top_k_mask(ctx, keep, top_k, score, inp_kpool.pool_idxs, inp_kpool.tail_idxs); // [n_kv, n_tokens]
            b.res.addFusedNode(.{ .op = .top_k_mask, .tensor = sel, .il = il });
            b.cb(sel, "indexer_sel", il);

            return sel;
        }

        // The top blocks, then the incomplete tail with n_kv for missing
        // cells.
        var sel_idx: *Tensor = c.ggml_get_rows(ctx, inp_kpool.pool_idxs, c.ggml_reshape_1d(ctx, top_k, n_top_pool * n_tokens)); // [kpool, n_top_pool*n_tokens]
        sel_idx = c.ggml_reshape_2d(ctx, sel_idx, kpool * n_top_pool, n_tokens);
        sel_idx = c.ggml_concat(ctx, sel_idx, inp_kpool.tail_idxs, 0);
        const n_sel: i64 = sel_idx.ne[0];
        impl.assert(n_sel == inp_kpool.n_sel, src, 912, "n_sel == inp_kpool->n_sel");

        c.ggml_build_forward_expand(b.gf, sel_idx);

        // Zeros scattered for the selected cells into an all -inf row, each
        // dead slot into its own dump row n_kv + slot. Seeding from sel_idx
        // ties the scatter storage's lifetime to this layer.
        const n_kv: i64 = inp_kpool.n_kv;

        var mask_all: *Tensor = c.ggml_new_tensor_4d(ctx, type_mask, n_kv + n_sel, 1, 1, 1);
        mask_all = c.ggml_fill(ctx, mask_all, -std.math.inf(f32));
        mask_all = c.ggml_repeat_4d(ctx, mask_all, n_kv + n_sel, n_tokens, 1, 1);
        mask_all = c.ggml_reshape_3d(ctx, mask_all, 1, n_kv + n_sel, n_tokens);

        var zeros: *Tensor = c.ggml_new_tensor_4d(ctx, type_mask, n_sel, 1, 1, 1);
        zeros = c.ggml_fill(ctx, zeros, 0.0);
        zeros = c.ggml_repeat_4d(ctx, zeros, n_sel, n_tokens, 1, 1);
        zeros = c.ggml_reshape_3d(ctx, zeros, 1, n_sel, n_tokens);

        // Live slots address disjoint cells, but padded pools and missing
        // tail cells share the n_kv sentinel, and top_k fills a short
        // selection with invisible pools that can overlap the tail, so the
        // scatter would write some cells from several threads: every dead
        // slot maps to its own dump row, idx = dump + live*(idx - dump). A
        // picked pool is live when visible: a visible score is a rectified
        // sum >= 0, an invisible one -inf.
        const top_score: *Tensor = c.ggml_get_rows(ctx, c.ggml_reshape_3d(ctx, score, 1, n_pool, n_tokens), top_k); // [1, n_top_pool, n_tokens]
        var live_pool: *Tensor = c.ggml_clamp(ctx, c.ggml_scale_bias(ctx, top_score, 1.0, 1.0), 0.0, 1.0);
        live_pool = c.ggml_reshape_2d(ctx, c.ggml_repeat_4d(ctx, live_pool, kpool, n_top_pool, n_tokens, 1), kpool * n_top_pool, n_tokens);
        // A tail cell is live unless it is the n_kv sentinel.
        var live_tail: *Tensor = c.ggml_cast(ctx, inp_kpool.tail_idxs, c.GGML_TYPE_F32);
        live_tail = c.ggml_clamp(ctx, c.ggml_scale_bias(ctx, live_tail, -1.0, @floatFromInt(n_kv)), 0.0, 1.0);
        const live: *Tensor = c.ggml_concat(ctx, live_pool, live_tail, 0); // [n_sel, n_tokens]

        // The dump rows n_kv + slot as a cumulative sum: the meta backend
        // cannot split an arange, which has no source.
        const dump: *Tensor = c.ggml_scale_bias(ctx, c.ggml_cumsum(ctx, c.ggml_fill(ctx, live, 1.0)), 1.0, @floatFromInt(n_kv - 1));
        var idx_f: *Tensor = c.ggml_cast(ctx, sel_idx, c.GGML_TYPE_F32);
        idx_f = c.ggml_add(ctx, c.ggml_mul(ctx, c.ggml_sub(ctx, idx_f, dump), live), dump);
        sel_idx = c.ggml_cast(ctx, idx_f, c.GGML_TYPE_I32);

        var sel: *Tensor = c.ggml_set_rows(ctx, mask_all, zeros, c.ggml_reshape_3d(ctx, sel_idx, n_sel, n_tokens, 1));

        sel = c.ggml_view_2d(ctx, sel, n_kv, n_tokens, sel.nb[2], 0);
        sel = c.ggml_cont(ctx, sel);
        b.cb(sel, "indexer_sel", il);

        return sel;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_attn_qsa`
    /// (models/qwen4exp.cpp:960 @8642541b2): dense GQA self-attention over
    /// the cells the QSA mask keeps.
    fn buildAttnQsa(self: *Graph, inp: *graph.InpAttnKv, q_cur_: *Tensor, k_cur_: *Tensor, v_cur_: *Tensor, sel_: *Tensor, n_sel: i64, kq_scale: f32, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        var q_cur = q_cur_;
        var k_cur = k_cur_;
        var v_cur = v_cur_;

        // Q, K and V rotated before they reach a quantized cache, as the
        // dense path does. The indexer has already scored with its own
        // query in build_qsa_sel, so the selection is unaffected.
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

        const q = q_cur;
        const k = mctx_cur.getK(ctx, il);
        const v = mctx_cur.getV(ctx, il);

        // The selection replaces the kq_mask: split it into the streams of
        // the K view.
        const n_stream = k.ne[3];
        impl.assert(sel_.ne[0] == k.ne[2] and @rem(sel_.ne[1], n_stream) == 0, src, 1004, "sel->ne[0] == k->ne[2] && sel->ne[1] % n_stream == 0");
        const sel: *Tensor = c.ggml_reshape_4d(ctx, sel_, sel_.ne[0], @divTrunc(sel_.ne[1], n_stream), 1, n_stream);
        b.cb(sel, "kq_mask_qsa", il);

        var cur = b.buildAttnMha(q, k, v, null, sel, null, null, n_sel, kq_scale, il);
        b.cb(cur, "kqv_out", il);

        // The rotation is its own inverse, so it is undone on the value
        // side of the output.
        if (inp.self_v_rot) |r| {
            cur = graph.mulMatHadamard(ctx, cur, r);
        }

        return cur;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_layer_attn`
    /// (models/qwen4exp.cpp:1019 @8642541b2): a joint Q and gate
    /// projection, Q and K norms, M-RoPE, dense or QSA attention, and the
    /// sigmoid gate.
    fn buildLayerAttn(self: *Graph, inp: *graph.InpAttnKv, mctx_hyb: *const HybridIdxContext, inp_kpool: ?*InpKpool, cur_: *Tensor, inp_pos: *Tensor, sections: *[4]c_int, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const layer = &self.model.layers[@intCast(il)];
        const n_embd_head: i64 = hp.nEmbdHeadV(0);
        impl.assert(n_embd_head == hp.nEmbdHeadK(0), src, 1028, "n_embd_head == hparams.n_embd_head_k()");
        const n_head = b.n_head;
        const n_head_kv = b.n_head_kv;
        const n_tokens = b.n_tokens;

        // The indexer reads the same block input as Q, K and V; no k-pool
        // input or no ratio means dense.
        const qsa = inp_kpool != null and hp.dsv4_compress_ratios[@intCast(il)] > 0;

        const sel: ?*Tensor = if (qsa) self.buildQsaSel(mctx_hyb, inp_kpool.?, cur_, inp_pos, sections, il) else null;

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
        if (sel) |sl| {
            cur = self.buildAttnQsa(inp, Qcur, Kcur, Vcur, sl, inp_kpool.?.n_sel, kq_scale, il);
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
    /// (models/qwen4exp.cpp:1106 @8642541b2): the gated delta-net layer.
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

        impl.assert(n_seqs != 0, src, 1120, "n_seqs != 0");
        impl.assert(b.ubatch.equalSeqs(), src, 1121, "ubatch.equal_seqs()");
        impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, src, 1122, "ubatch.n_tokens == n_seq_tokens * n_seqs");
        impl.assert(head_v_dim * num_v_heads == d_inner, src, 1123, "head_v_dim * num_v_heads == d_inner");

        const qkv_mixed, const z = self.buildQkvz(cur_, il);

        var beta: *Tensor = undefined;
        var alpha: *Tensor = undefined;
        if (layer.ssm_beta_alpha_cat) |cat| {
            // beta and alpha read the same input: one matmul, then the beta
            // rows and the alpha rows.
            const ba = b.buildLoraMm(cat, cur_, null);
            b.cb(ba, "beta_alpha", il);

            const nb1_tok: usize = ba.nb[1] * @as(usize, @intCast(n_seq_tokens));
            beta = c.ggml_view_4d(ctx, ba, 1, num_v_heads, n_seq_tokens, n_seqs, c.ggml_element_size(ba), ba.nb[1], nb1_tok, 0);
            alpha = c.ggml_view_3d(ctx, ba, num_v_heads, n_seq_tokens, n_seqs, ba.nb[1], nb1_tok, c.ggml_row_size(ba.type, num_v_heads));
        } else {
            beta = b.buildLoraMm(layer.ssm_beta.?, cur_, layer.ssm_beta_s);
            beta = c.ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);

            alpha = b.buildLoraMm(layer.ssm_alpha.?, cur_, layer.ssm_alpha_s);
            alpha = c.ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
        }
        b.cb(beta, "beta", il);
        b.cb(alpha, "alpha", il);

        beta = c.ggml_sigmoid(ctx, beta);
        b.cb(beta, "beta_sigmoid", il);

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
            impl.assert(@rem(num_v_heads, num_k_heads) == 0, src, 1219, "num_v_heads % num_k_heads == 0");
            q_conv = c.ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
            k_conv = c.ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        }

        b.cb(q_conv, "q_conv_predelta", il);
        b.cb(k_conv, "k_conv_predelta", il);
        b.cb(v_conv, "v_conv_predelta", il);

        const output = delta_net.buildRecurrentAttn(b, inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

        const z_2d: *Tensor = c.ggml_view_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs, c.ggml_row_size(z.type, head_v_dim), z.nb[1], z.nb[1] * nst, 0);

        // The gated norm, self.norm(core_attn_out, z) in the reference.
        const attn_out_norm = self.buildNormGated(output, layer.ssm_norm, z_2d, il);

        const final_output: *Tensor = c.ggml_reshape_2d(ctx, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
        b.cb(final_output, "final_output", il);

        const cur = b.buildLoraMm(layer.ssm_out.?, final_output, layer.ssm_out_s);
        b.cb(cur, "linear_attn_out", il);

        return cur;
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_layer_ffn`
    /// (models/qwen4exp.cpp:1245 @8642541b2): the routed experts and the
    /// gated shared expert.
    fn buildLayerFfn(self: *Graph, cur_: *Tensor, il: c_int) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hp = b.hparams;
        const layer = &self.model.layers[@intCast(il)];

        impl.assert(layer.ffn_gate_inp != null, src, 1246, "model.layers[il].ffn_gate_inp != nullptr");

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
    /// (models/qwen4exp.cpp:1398 @8642541b2): a conv history read out of
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
        impl.assert(state_cols * channels == row_total, src, 1413, "state_cols * channels == row_total");

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
    /// (models/qwen4exp.cpp:1452 @8642541b2): the PLE rows' input and the
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
    /// hc stream, with an [n_embd, hc] weight.
    fn groupedNorm(self: *Graph, x: *Tensor, w: ?*Tensor) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const hc: i64 = b.hparams.dsv4_hc_mult;
        const t: *Tensor = c.ggml_reshape_3d(ctx, x, b.n_embd, hc, b.n_tokens);
        return c.ggml_mul(ctx, c.ggml_rms_norm(ctx, t, b.hparams.f_norm_rms_eps), w);
    }

    /// Ports `llama_model_qwen4exp::graph`'s `build_ple` (models/qwen4exp.cpp:1472
    /// @8642541b2): the PLE embedding gated against the residual, through a
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
