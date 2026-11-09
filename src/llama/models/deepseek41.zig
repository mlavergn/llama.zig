//! DeepSeek-V4.1: DeepSeek-V4 with single-pass hyper-connections and no
//! `hc_head`, a compressed KV and a top-k that source layers compute and
//! later layers reuse, a lazily read Engram n-gram table beside the stream,
//! an optional two-level candidate-block selection, and a sparse attention
//! path that gathers the selected rows for few queries.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/deepseek41.cpp` at v0.8.0 (`8642541b2`):
//! `dsv41_env_on`, `llama_model_deepseek41`'s `load_arch_hparams`,
//! `load_arch_tensors`, `build_arch_graph`, `engram_index` and `engram_hash`,
//! `llm_graph_input_dsv41`, and its `graph` with `build_engram` and
//! `build_attention41`. The base class's builders are `deepseek4.zig`'s.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const batch = @import("../batch.zig");
const kv_cache_dsv4 = @import("../kv_cache_dsv4.zig");
const deepseek4 = @import("deepseek4.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Builder = graph.Builder;
const Tensor = c.ggml_tensor;
const Ubatch = batch.Ubatch;
const allocator = impl.allocator;

const src = "models/deepseek41.cpp";

const inf = std.math.inf(f32);

fn oom() noreturn {
    impl.abort(src, 0, "out of memory");
}

/// `LLAMA_MAX_PLE_NGRAM` (llama-hparams.h:13 @8642541b2).
const max_ple_ngram: usize = 8;
/// `LLAMA_MAX_PLE_HEADS` (llama-hparams.h:14 @8642541b2).
const max_ple_heads: usize = 64;

/// Ports `DSV41_ENGRAM_DEAD` (models/deepseek41.cpp:16 @8642541b2).
const engram_dead: i64 = -1;

/// Ports `dsv41_env_on` (models/deepseek41.cpp:18 @8642541b2).
fn envOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return true;
    return impl.atoi(v) != 0;
}

/// The `out_of_range` lambda of `load_arch_hparams`.
fn outOfRange(v: []const u32, n_layer: u32) bool {
    for (v) |il| {
        if (il >= n_layer) return true;
    }
    return false;
}

/// `std::find` on a list of layers.
fn contains(v: []const u32, il: u32) bool {
    return std.mem.indexOfScalar(u32, v, il) != null;
}

/// `ml.get_arr` into a `std::vector`, whose required form throws and whose
/// optional form leaves it empty.
fn arrList(comptime T: type, meta: anytype, kv: anytype, required: bool, exc: *Exception) Thrown![]T {
    return (try meta.getArrList(T, kv, required, exc)) orelse &.{};
}

/// Ports `llama_model_deepseek41`'s `load_arch_hparams` (models/deepseek41.cpp:23 @8642541b2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    try deepseek4.loadArchHparams(self, ml, exc);

    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.n_embd_out_impl = hp.n_embd; // No hc_head.
    if (hp.indexer_top_k == 0) {
        return exc.throw(.runtime_error, "DeepSeek-V4.1 indexer top_k must be > 0", .{});
    }

    // Each compressed layer reads the KV and top-k of the latest source
    // layer at or before it.
    const kv_src = try arrList(u32, meta, .attention_kv_source_layers, true, exc);
    defer allocator.free(kv_src);
    const idx_src = try arrList(u32, meta, .attention_indexer_source_layers, true, exc);
    defer allocator.free(idx_src);

    const n_layer: u32 = hp.nLayer();
    if (outOfRange(kv_src, n_layer) or outOfRange(idx_src, n_layer)) {
        return exc.throw(.runtime_error, "DeepSeek-V4.1 kv/index source layer out of range", .{});
    }
    hp.dsv41_kv_src = @splat(-1);
    hp.dsv41_topk_src = @splat(-1);
    var last_kv: i32 = -1;
    var last_idx: i32 = -1;
    var il: u32 = 0;
    while (il < n_layer) : (il += 1) {
        const ratio: u32 = hp.dsv4_compress_ratios[il];
        const is_kv = contains(kv_src, il);
        const is_idx = contains(idx_src, il);
        if ((is_kv or is_idx) and ratio == 0) {
            return exc.throw(.runtime_error, "DeepSeek-V4.1 source layer {d} has compress ratio 0", .{il});
        }
        last_kv = if (is_kv) @intCast(il) else last_kv;
        last_idx = if (is_idx) @intCast(il) else last_idx;
        if (ratio == 0) {
            continue;
        }
        if (ratio > 2 or last_kv < 0 or last_idx < last_kv or
            hp.dsv4_compress_ratios[@intCast(last_kv)] != ratio or hp.dsv4_compress_ratios[@intCast(last_idx)] != ratio)
        {
            return exc.throw(.runtime_error, "DeepSeek-V4.1 layer {d}: ratio {d} without a matching kv/index source", .{ il, ratio });
        }
        hp.dsv41_kv_src[il] = last_kv;
        hp.dsv41_topk_src[il] = last_idx;
    }

    if (try meta.getKey(i32, .attention_indexer_candidate_source_layer, false, exc)) |v| hp.dsv41_cand_src = v;
    if (hp.dsv41_cand_src >= 0) {
        hp.dsv41_cand_topk_blocks = (try meta.getKey(u32, .attention_indexer_candidate_top_k_blocks, true, exc)).?;
        hp.dsv41_cand_block_size = (try meta.getKey(u32, .attention_indexer_candidate_block_size, true, exc)).?;
        const cs: i32 = hp.dsv41_cand_src;
        const ucs: u32 = @bitCast(cs);
        if (ucs >= n_layer or hp.dsv41_topk_src[ucs] != cs or hp.dsv41_cand_block_size == 0 or
            hp.dsv41_cand_topk_blocks == 0)
        {
            return exc.throw(.runtime_error, "DeepSeek-V4.1 invalid candidate selection (source must be an index source, block size and top-k blocks > 0)", .{});
        }
        il = ucs + 1;
        while (il < n_layer) : (il += 1) {
            if (hp.dsv41_topk_src[il] == @as(i32, @intCast(il)) and hp.dsv4_compress_ratios[il] != hp.dsv4_compress_ratios[ucs]) {
                return exc.throw(.runtime_error, "DeepSeek-V4.1 candidate consumers must share the source's compress ratio", .{});
            }
        }
    }

    // Engram: the PLE n-gram hash of qwen4exp, with one set of hash constants
    // per Engram layer, stored back to back.
    const e_layers = try arrList(u32, meta, .ple_layers, false, exc);
    defer if (e_layers.len > 0) allocator.free(e_layers);
    if (e_layers.len > 0) {
        hp.ple_ngram_size = (try meta.getKey(u32, .ple_ngram_size, true, exc)).?;
        hp.ple_heads_per_ngram = (try meta.getKey(u32, .ple_heads_per_ngram, true, exc)).?;
        hp.ple_head_dim = (try meta.getKey(u32, .embedding_length_per_layer, true, exc)).?;
        const mult = try arrList(u64, meta, .ple_layer_multipliers, true, exc);
        defer allocator.free(mult);
        const offsets = try arrList(u64, meta, .ple_head_offsets, true, exc);
        defer allocator.free(offsets);
        const primes = try arrList(u64, meta, .ple_head_vocab_sizes, true, exc);
        defer allocator.free(primes);
        hp.engram_pad_id = (try meta.getKey(u32, .engram_pad_token_id, true, exc)).?;
        const token_map = try arrList(i32, meta, .engram_token_map, true, exc);
        self.engram_token_map.deinit(allocator);
        self.engram_token_map = .fromOwnedSlice(token_map);

        const n_e: usize = e_layers.len;
        const n_gram: usize = hp.ple_ngram_size;
        const n_cols: usize = if (n_gram >= 2) (n_gram - 1) *% hp.ple_heads_per_ngram else 0;
        if (n_cols == 0 or hp.ple_head_dim == 0 or
            mult.len != n_e *% n_gram or primes.len != n_e *% n_cols or offsets.len != n_e *% n_cols or
            mult.len > max_ple_ngram or primes.len > max_ple_heads or
            hp.engram_pad_id >= token_map.len)
        {
            return exc.throw(.runtime_error, "DeepSeek-V4.1 engram metadata is inconsistent", .{});
        }
        for (e_layers, 0..) |layer, e| {
            // Module e belongs to the e-th Engram layer.
            if (layer >= n_layer or (e > 0 and layer <= e_layers[e - 1])) {
                return exc.throw(.runtime_error, "DeepSeek-V4.1 engram layer {d} is out of range or out of order", .{layer});
            }
            hp.is_ple_impl[layer / 64] |= @as(u64, 1) << @intCast(layer % 64);
        }
        for (0..n_e * n_cols) |col| {
            // A bucket count of 0 traps in the hash; row ids are int32.
            if (primes[col] < 2 or offsets[col] +% primes[col] > std.math.maxInt(i32)) {
                return exc.throw(.runtime_error, "DeepSeek-V4.1 engram head {d}: bucket range out of range", .{col});
            }
            hp.ple_head_offsets[col] = @truncate(offsets[col]);
            hp.ple_head_vocab_sizes[col] = @truncate(primes[col]);
        }
        @memcpy(hp.ple_layer_multipliers[0..mult.len], mult);
        hp.ple_n_heads = @intCast(n_cols);
        hp.ple_n_modules = @intCast(n_e);
    }
}

/// Ports `llama_model_deepseek41`'s `load_arch_tensors` (models/deepseek41.cpp:130 @8642541b2).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const q_lora_rank: i64 = hp.n_lora_q;
    const n_ff_exp: i64 = hp.nFfExp(0);
    const n_expert_shared: i64 = hp.n_expert_shared;
    const n_embd_head: i64 = hp.nEmbdHeadK(0);
    const o_groups: i64 = hp.dsv4_o_group_count;
    const o_lora_rank: i64 = hp.dsv4_o_lora_rank;
    const hc_mult: i64 = hp.dsv4_hc_mult;
    const hc_dim = hc_mult * l.n_embd;
    const hc_mix_dim = (2 + hc_mult) * hc_mult;
    const n_idx_embd: i64 = hp.indexer_head_size;
    const n_idx_head: i64 = hp.indexer_n_head;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // All the Engram tables in one lazily read tensor, like the qwen4exp
    // n-gram table.
    if (hp.ple_n_heads > 0) {
        var n_rows: i64 = 0;
        for (0..@as(usize, hp.ple_n_modules) * hp.ple_n_heads) |col| {
            n_rows = @max(n_rows, @as(i64, hp.ple_head_offsets[col]) + hp.ple_head_vocab_sizes[col]);
        }
        const name = (loader.Tn{ .tensor = .per_layer_token_embd, .suffix = "weight" }).str();
        if (ml.getWeight(name.slice())) |w| {
            if (w.tensor.*.ne[1] < n_rows) {
                return exc.throw(.runtime_error, "{s} has {d} rows, too few for the Engram heads ({d})", .{ name.slice(), w.tensor.*.ne[1], n_rows });
            }
            n_rows = w.tensor.*.ne[1];
        }
        self.per_layer_tok_embd = try self.createTensor(.{ .tensor = .per_layer_token_embd, .suffix = "weight" }, &.{ hp.ple_head_dim, n_rows }, loader.tensor_read_lazy, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const ui: usize = @intCast(i);

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

        if (hp.dsv41_kv_src[ui] == i) {
            layer.attn_comp_wkv = try self.createTensor(.{ .tensor = .attn_compressor_wkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_head }, 0, exc);
            layer.attn_comp_norm = try self.createTensor(.{ .tensor = .attn_compressor_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, 0, exc);
            if (hp.dsv4_compress_ratios[ui] > 1) {
                layer.attn_comp_wgate = try self.createTensor(.{ .tensor = .attn_compressor_wgate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_head }, 0, exc);
            }
            layer.indexer_attn_k = try self.createTensor(.{ .tensor = .indexer_attn_k, .suffix = "weight", .bid = i }, &.{ n_embd_head, n_idx_embd }, 0, exc);
            layer.indexer_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{n_idx_embd}, 0, exc);
        }
        if (hp.dsv41_topk_src[ui] == i) {
            layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, n_idx_head * n_idx_embd }, 0, exc);
            layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_idx_head }, 0, exc);
        }

        if (engramIndex(self, i) >= 0) {
            layer.engram_kv = try self.createTensor(.{ .tensor = .engram_kv, .suffix = "weight", .bid = i }, &.{ @as(i64, hp.ple_n_heads) * hp.ple_head_dim, l.n_embd * (hc_mult + 1) }, 0, exc);
            layer.engram_gate = try self.createTensor(.{ .tensor = .engram_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, hc_mult }, 0, exc);
        }

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);
        layer.ffn_exp_probs_b_vl = try self.createTensor(.{ .tensor = .ffn_exp_probs_b_vl, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr, exc);
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

        layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
        layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, 0, exc);
        layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
    }
}

/// Ports `llama_model_deepseek41`'s `build_arch_graph` (models/deepseek41.cpp:220 @8642541b2).
pub fn buildArchGraph(model: *const Model, b: *const Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) {
        // The C++ throws here. The port's builders have no error path, so
        // `Context.graphReserve` throws it before any build gets this far.
        impl.abort(src, 222, "DeepSeek-V4.1 has no MTP graph: its DSpark draft is a separate DFLASH GGUF (convert_hf_to_gguf.py --dspark), used with --spec-type draft-dspark");
    }
    buildGraph(model, b);
}

/// Ports `llama_model_deepseek41`'s `engram_index` (models/deepseek41.cpp:229
/// @8642541b2): the Engram n-gram hash module of a layer, or -1.
fn engramIndex(model: *const Model, il: i32) i32 {
    const hp = &model.hparams;
    if (hp.ple_n_heads == 0 or !hp.isPle(@intCast(il))) {
        return -1;
    }
    var e: i32 = 0;
    var j: i32 = 0;
    while (j < il) : (j += 1) {
        e += @intFromBool(hp.isPle(@intCast(j)));
    }
    return e;
}

/// Ports `llama_model_deepseek41`'s `engram_hash` (models/deepseek41.cpp:240
/// @8642541b2): engram.py's `NgramHashState.forward`, the table row of each
/// head for every token of the ubatch.
fn engramHash(model: *const Model, e: usize, ubatch: *const Ubatch, prev: []const i32, out: []i32) void {
    const hp = &model.hparams;
    const n_gram: usize = hp.ple_ngram_size;
    const n_prev: usize = n_gram - 1;
    const n_head: usize = hp.ple_heads_per_ngram;
    const n_cols: usize = hp.ple_n_heads;

    const mult = hp.ple_layer_multipliers[e * n_gram ..];
    const primes = hp.ple_head_vocab_sizes[e * n_cols ..];
    const token_map = model.engram_token_map.items;
    const pad: i64 = token_map[hp.engram_pad_id];
    const offs = hp.ple_head_offsets[e * n_cols ..];

    // The `cid` lambda.
    const Cid = struct {
        fn of(map: []const i32, t: i32) i64 {
            return if (t < 0 or t >= @as(i32, @intCast(map.len))) engram_dead else map[@intCast(t)];
        }
    };

    var toks: [max_ple_ngram]u64 = undefined;
    for (0..ubatch.n_tokens) |i| {
        // Shift 0 is the token itself; once a shift is blocked (the
        // sequence's start or a dead token) every longer one is too.
        const is_embd = ubatch.token == null or (ubatch.isMixed() and ubatch.type[i] != 0);
        var cc: i64 = if (is_embd) engram_dead else Cid.of(token_map, ubatch.token[i]);
        var blocked = cc == engram_dead;
        toks[0] = @bitCast(if (blocked) pad else cc);
        for (1..n_gram) |s| {
            cc = if (@as(i64, ubatch.pos[i]) - @as(i64, @intCast(s)) < 0) engram_dead else Cid.of(token_map, prev[i * n_prev + (n_prev - s)]);
            blocked = blocked or cc == engram_dead;
            toks[s] = @bitCast(if (blocked) pad else cc);
        }

        var rolling: u64 = toks[0] *% mult[0];
        for (1..n_gram) |o| {
            rolling ^= toks[o] *% mult[o];
            for (0..n_head) |h| {
                const col = (o - 1) * n_head + h;
                out[i * n_cols + col] = @truncate(@as(i64, @bitCast(rolling % primes[col] +% offs[col])));
            }
        }
    }
}

/// Ports `llm_graph_input_dsv41` (models/deepseek41.cpp:281 @8642541b2).
/// `cand_pin` is +1e30 on the block with the newest compressed row of each
/// query, so that a partly filled block is always kept; the reference uses
/// +inf, but a finite pin keeps fully masked blocks at -inf rather than NaN.
const InpDsv41 = struct {
    engram_ids: std.ArrayList(*Tensor) = .empty,
    /// 0 for embedding (image) tokens; only in mixed ubatches.
    engram_keep: ?*Tensor = null,
    cand_pin: ?*Tensor = null,
    cand_csa: bool = false,

    model: *const Model,
    mctx: *const kv_cache_dsv4.Dsv4Context,

    pub fn deinit(self: *InpDsv41) void {
        self.engram_ids.deinit(allocator);
    }

    /// The private `cand_plan`.
    fn candPlan(self: *const InpDsv41, ubatch: *const Ubatch) *const kv_cache_dsv4.CompPlan {
        return if (self.cand_csa) self.mctx.getCsaPlanFor(ubatch) else self.mctx.getHcaPlanFor(ubatch);
    }

    /// Ports `llm_graph_input_dsv41`'s `set_input` (models/deepseek41.cpp:303 @8642541b2).
    pub fn setInput(self: *InpDsv41, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const hp = &self.model.hparams;
        const n_tokens: usize = ubatch.n_tokens;

        if (self.engram_ids.items.len > 0) {
            // A token shared by several sequences has the same history in
            // each, so its first sequence resolves it.
            var prev: std.ArrayList(i32) = .empty;
            defer prev.deinit(allocator);
            self.mctx.getRaw().getPrevTokens(ubatch, hp.ple_ngram_size - 1, &prev);

            // All the Engram layers are hashed first, so a lazily read table
            // gets one batched readahead before the first gather.
            const n_ids = n_tokens * hp.ple_n_heads;
            const n_e = self.engram_ids.items.len;
            const all = allocator.alloc(i32, n_e * n_ids) catch oom();
            defer allocator.free(all);
            for (0..n_e) |e| {
                engramHash(self.model, e, ubatch, prev.items, all[e * n_ids ..][0..n_ids]);
            }
            const table = self.model.per_layer_tok_embd;
            if (self.model.canPrefetch(table)) {
                impl.prefetchRows(table.?, all.ptr, all.len);
            }
            for (self.engram_ids.items, 0..) |ids, e| {
                if (ids.buffer != null) {
                    c.ggml_backend_tensor_set(ids, all[e * n_ids ..].ptr, 0, n_ids * @sizeOf(i32));
                }
            }
        }

        // Embedding (image) tokens are hashed dead and get no Engram
        // contribution (gate 0).
        if (self.engram_keep) |keep_t| {
            if (keep_t.buffer != null) {
                const keep = allocator.alloc(f32, n_tokens) catch oom();
                defer allocator.free(keep);
                for (keep, 0..) |*k, i| {
                    const is_embd = ubatch.token == null or (ubatch.isMixed() and ubatch.type[i] != 0);
                    k.* = if (is_embd) 0.0 else 1.0;
                }
                c.ggml_backend_tensor_set(keep_t, keep.ptr, 0, keep.len * @sizeOf(f32));
            }
        }

        if (self.cand_pin) |pin_t| {
            if (pin_t.buffer != null) {
                const plan = self.candPlan(ubatch);
                const n_blk = pin_t.ne[0];
                const bs: i64 = hp.dsv41_cand_block_size;
                impl.assert(@as(i64, @intCast(plan.n_visible.items.len)) == n_tokens, src, 339, "(int64_t) plan.n_visible.size() == (int64_t) ubatch->n_tokens");
                const pin = allocator.alloc(f32, @as(usize, @intCast(n_blk)) * n_tokens) catch oom();
                defer allocator.free(pin);
                @memset(pin, 0.0);
                for (0..n_tokens) |i| {
                    const nv: i64 = plan.n_visible.items[i];
                    if (nv > 0) {
                        impl.assert(@divTrunc(nv - 1, bs) < n_blk, src, 344, "(nv - 1)/bs < n_blk");
                        pin[@intCast(@as(i64, @intCast(i)) * n_blk + @divTrunc(nv - 1, bs))] = 1e30;
                    }
                }
                c.ggml_backend_tensor_set(pin_t, pin.ptr, 0, pin.len * @sizeOf(f32));
            }
        }
    }

    /// Ports `llm_graph_input_dsv41`'s `can_reuse` (models/deepseek41.cpp:352 @8642541b2).
    pub fn canReuse(self: *InpDsv41, params: *const graph.Params) bool {
        self.mctx = params.mctx.?.cast(kv_cache_dsv4.Dsv4Context);
        const hp = &self.model.hparams;
        const n_tokens: i64 = params.ubatch.n_tokens;

        var res = true;
        for (self.engram_ids.items) |t| {
            res = res and t.ne[0] == @as(i64, hp.ple_n_heads) * n_tokens;
        }
        res = res and (self.engram_keep != null) == (self.engram_ids.items.len > 0 and params.ubatch.embd != null);
        if (self.engram_keep) |k| {
            res = res and k.ne[2] == n_tokens;
        }
        if (self.cand_pin) |p| {
            const plan = self.candPlan(&params.ubatch);
            const bs: i64 = hp.dsv41_cand_block_size;
            res = res and plan.n_kv > 0 and p.ne[0] == @divTrunc(plan.n_kv + bs - 1, bs);
            res = res and p.ne[1] * p.ne[3] == n_tokens;
        }
        return res;
    }
};

/// The members of `llama_model_deepseek41`'s `graph` (src/models/models.h:1519
/// @8642541b2) that one build carries from layer to layer.
const Graph = struct {
    /// The top-k of each index source layer.
    topk_of: []?*Tensor,
    /// `[n_layer]` the new compressed rows of each KV source, for mirrors.
    comp_row_of: []?*Tensor,
    /// The candidate blocks of the candidate source layer.
    cand_sel: ?*Tensor = null,
    /// The compressed rows the candidate source scored.
    cand_n_comp: i64 = 0,

    /// The sparse path (`LLAMA_DSV41_SPARSE=0` disables): a fused indexer,
    /// FA bounded to the window and the top-k, and the decode gather.
    sparse: bool = true,
    /// The gather path, shared by layers with the same top-k: the selected
    /// K rows per (storage layer, top-k source), and the sorted top-k and
    /// the mask per top-k source (one input copy per split).
    gather_k_of: std.AutoHashMapUnmanaged(struct { i32, i32 }, ?*Tensor) = .empty,
    gather_sel_of: std.AutoHashMapUnmanaged(i32, ?*Tensor) = .empty,
    gather_mask_of: std.AutoHashMapUnmanaged(i32, ?*Tensor) = .empty,

    fn deinit(self: *Graph) void {
        allocator.free(self.topk_of);
        allocator.free(self.comp_row_of);
        self.gather_k_of.deinit(allocator);
        self.gather_sel_of.deinit(allocator);
        self.gather_mask_of.deinit(allocator);
    }
};

/// Ports `llama_model_deepseek41::graph`'s `build_engram`
/// (models/deepseek41.cpp:372 @8642541b2).
fn buildEngram(b: *const Builder, model: *const Model, x: *Tensor, rows: *Tensor, keep: ?*Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    const hc: i64 = hp.dsv4_hc_mult;
    const nt = x.ne[2];
    const n_embd = b.n_embd;

    impl.assert(rows.ne[0] == @as(i64, hp.ple_head_dim *% hp.ple_n_heads) and rows.ne[1] == nt, src, 382, "rows->ne[0] == (int64_t) (hparams.ple_head_dim*hparams.ple_n_heads) && rows->ne[1] == nt");

    const kv = b.buildLoraMm(layer.engram_kv.?, rows, null);
    b.cb(kv, "engram_kv", il);

    const key: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, kv, n_embd, hc, nt, c.ggml_row_size(kv.type, n_embd), kv.nb[1], 0));
    const value: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, kv, n_embd, 1, nt, c.ggml_row_size(kv.type, n_embd), kv.nb[1], c.ggml_row_size(kv.type, hc * n_embd)));

    // A quantized copy has no binary-op kernel.
    const engram_gate = layer.engram_gate.?;
    const egate: *Tensor = if (c.ggml_is_quantized(engram_gate.type)) c.ggml_cast(ctx, engram_gate, c.GGML_TYPE_F32) else engram_gate;
    var dot: *Tensor = c.ggml_mul(ctx, c.ggml_mul(ctx, c.ggml_rms_norm(ctx, x, b.norm_rms_eps), c.ggml_rms_norm(ctx, key, b.norm_rms_eps)), egate);
    dot = c.ggml_scale(ctx, c.ggml_sum_rows(ctx, dot), 1.0 / impl.sqrtf(@floatFromInt(n_embd)));
    b.cb(dot, "engram_dot", il);

    // sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); sign(0) = +1 as in
    // copysign.
    const mag: *Tensor = c.ggml_sqrt(ctx, c.ggml_clamp(ctx, c.ggml_abs(ctx, dot), 1e-6, inf));
    const sign: *Tensor = c.ggml_scale_bias(ctx, c.ggml_step(ctx, c.ggml_neg(ctx, dot)), -2.0, 1.0);
    var gate: *Tensor = c.ggml_sigmoid(ctx, c.ggml_mul(ctx, mag, sign));
    if (keep) |k| {
        gate = c.ggml_mul(ctx, gate, k);
    }
    b.cb(gate, "engram_gate", il);

    return c.ggml_add(ctx, x, c.ggml_mul(ctx, c.ggml_repeat_4d(ctx, value, n_embd, hc, nt, 1), gate));
}

/// The `rope` lambda of `build_attention41`.
const Rope = struct {
    b: *const Builder,
    n_embd_head_rope: c_int,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,

    fn apply(self: *const Rope, t_: *Tensor, pos: ?*Tensor, n_nope: i64) *Tensor {
        const t: *Tensor = c.ggml_rope_ext(self.b.ctx0, t_, pos, null, self.n_embd_head_rope, self.b.rope_type, self.n_ctx_orig, self.freq_base, self.freq_scale, self.ext_factor, self.attn_factor, self.beta_fast, self.beta_slow);
        return c.ggml_rope_set_offset(t, @intCast(n_nope));
    }
};

/// The `rows` lambda of `build_attention41`: `nc` rows of a per-query tensor
/// from row `t0`.
fn rowsOf(ctx: *c.ggml_context, t: *Tensor, nc: i64, n_stream: i64, t0: i64) *Tensor {
    return c.ggml_view_4d(ctx, t, t.ne[0], nc, 1, n_stream, t.nb[1], t.nb[2], t.nb[3], @as(usize, @intCast(t0)) * t.nb[1]);
}

/// The `join` lambda of `build_attention41`: the parts concatenated pairwise
/// along the rows. `v` is consumed.
fn join(ctx: *c.ggml_context, v: *std.ArrayList(*Tensor)) *Tensor {
    while (v.items.len > 1) {
        var w: usize = 0;
        var i: usize = 0;
        while (i < v.items.len) : (i += 2) {
            v.items[w] = if (i + 1 < v.items.len) c.ggml_concat(ctx, v.items[i], v.items[i + 1], 1) else v.items[i];
            w += 1;
        }
        v.items.len = w;
    }
    return v.items[0];
}

/// Ports `llama_model_deepseek41::graph`'s `build_attention41`
/// (models/deepseek41.cpp:409 @8642541b2).
fn buildAttention41(self: *Graph, b: *const Builder, model: *const Model, inp_dsv4: *const graph.InpDsv4, cur: *Tensor, inp_pos: *Tensor, cand_pin: ?*Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const uil: usize = @intCast(il);
    const layer = &model.layers[uil];
    const inp_attn = inp_dsv4.getRaw();

    const n_embd_head: i64 = hp.nEmbdHeadK(0);
    const n_embd_head_rope: i64 = hp.nRot(0);
    const n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const n_groups: i64 = hp.dsv4_o_group_count;
    const o_lora_rank: i64 = hp.dsv4_o_lora_rank;
    const n_idx_embd: i64 = hp.indexer_head_size;
    const n_idx_head: i64 = hp.indexer_n_head;
    const nt = cur.ne[1];
    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));
    const ratio: i64 = hp.dsv4_compress_ratios[uil];

    const comp_rope = ratio != 0;
    const freq_scale_l: f32 = if (comp_rope) b.freq_scale else 1.0;
    const ext_factor_l: f32 = if (comp_rope) b.ext_factor else 0.0;
    const rope: Rope = .{
        .b = b,
        .n_embd_head_rope = @intCast(n_embd_head_rope),
        .n_ctx_orig = if (comp_rope) b.n_ctx_orig else 0,
        .freq_base = if (comp_rope) hp.dsv4_compress_rope_base else b.freq_base,
        .freq_scale = freq_scale_l,
        .ext_factor = ext_factor_l,
        .attn_factor = deepseek4.ropeAttnFactor(freq_scale_l, ext_factor_l),
        .beta_fast = if (comp_rope) b.beta_fast else 0.0,
        .beta_slow = if (comp_rope) b.beta_slow else 0.0,
    };

    const qr = b.buildNorm(b.buildLoraMm(layer.wq_a.?, cur, null), layer.attn_q_a_norm, null, .rms, il);
    b.cb(qr, "qr_norm", il);

    var q: *Tensor = c.ggml_reshape_3d(ctx, b.buildLoraMm(layer.wq_b.?, qr, null), n_embd_head, b.n_head, nt);
    q = rope.apply(q, inp_pos, n_embd_head_nope);
    b.cb(q, "q", il);

    var kv = b.buildNorm(b.buildLoraMm(layer.wkv.?, cur, null), layer.attn_kv_norm, null, .rms, il);
    kv = rope.apply(c.ggml_reshape_3d(ctx, kv, n_embd_head, 1, nt), inp_pos, n_embd_head_nope);
    b.cb(kv, "kv", il);

    const ci = if (ratio == 2) inp_dsv4.getCsa() else inp_dsv4.getHca();
    const comp_ctx = if (ratio == 2) inp_dsv4.mctx.getCsa() else inp_dsv4.mctx.getHca();
    const comp_state = if (ratio == 2) inp_dsv4.mctx.getCsaState() else inp_dsv4.mctx.getHcaState();

    const rd: i32 = inp_dsv4.mctx.getCompReadLayer(il);

    if (ratio != 0) {
        impl.assert(hp.dsv41_kv_src[uil] >= 0 and @as(i64, comp_state.getRatio()) == ratio, src, 463, "hparams.dsv41_kv_src[il] >= 0 && (int64_t) comp_state->get_ratio() == ratio");

        if (hp.dsv41_kv_src[uil] == il and ci.state_pos != null) {
            const state_kv = b.buildLoraMm(layer.attn_comp_wkv.?, cur, null);
            const state_score: *Tensor = if (ratio > 1) b.buildLoraMm(layer.attn_comp_wgate.?, cur, null) else c.ggml_fill(ctx, state_kv, 0.0);

            var restored_kv = comp_state.getKvAll(ctx, il);
            var restored_score = comp_state.getScoreAll(ctx, il);
            if (ci.state_restore_src_idxs != null and ci.state_restore_dst_idxs != null) {
                restored_kv = comp_state.cpyKv(ctx, c.ggml_get_rows(ctx, restored_kv, ci.state_restore_src_idxs), ci.state_restore_dst_idxs.?, il);
                restored_score = comp_state.cpyScore(ctx, c.ggml_get_rows(ctx, restored_score, ci.state_restore_src_idxs), ci.state_restore_dst_idxs.?, il);
            }

            if (ci.state_write_idxs != null) {
                const n_blocks = ci.state_write_pos.?.ne[0];
                const n_state: i64 = comp_state.getNRows();
                impl.assert(ci.state_read_idxs.?.ne[0] == ratio * n_blocks, src, 479, "ci.state_read_idxs->ne[0] == ratio*n_blocks");

                const src_kv: *Tensor = c.ggml_concat(ctx, c.ggml_view_2d(ctx, restored_kv, restored_kv.ne[0], n_state, restored_kv.nb[1], 0), state_kv, 1);
                const src_score: *Tensor = c.ggml_concat(ctx, c.ggml_view_2d(ctx, restored_score, restored_score.ne[0], n_state, restored_score.nb[1], 0), state_score, 1);

                var v: *Tensor = c.ggml_get_rows(ctx, src_kv, ci.state_read_idxs);
                var s: *Tensor = c.ggml_get_rows(ctx, src_score, ci.state_read_idxs);
                v = c.ggml_cont(ctx, c.ggml_permute(ctx, c.ggml_reshape_3d(ctx, v, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));
                s = c.ggml_cont(ctx, c.ggml_permute(ctx, c.ggml_reshape_3d(ctx, s, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));

                var comp: *Tensor = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, v, c.ggml_soft_max(ctx, s)));
                comp = c.ggml_cont(ctx, c.ggml_permute(ctx, comp, 1, 0, 2, 3));
                comp = b.buildNorm(comp, layer.attn_comp_norm, null, .rms, il);
                b.cb(comp, "comp_latent", il);

                // The index K comes from the RoPE-free latent; a latent
                // stands for the first token of its group.
                var ik = b.buildNorm(b.buildLoraMm(layer.indexer_attn_k.?, comp, null), layer.indexer_k_norm, null, .rms, il);
                ik = rope.apply(ik, ci.state_write_pos, n_idx_embd - n_embd_head_rope);
                comp = rope.apply(comp, ci.state_write_pos, n_embd_head_nope);

                const row: *Tensor = c.ggml_concat(ctx, comp, ik, 0);
                c.ggml_build_forward_expand(b.gf, comp_ctx.cpyK(ctx, row, ci.state_write_idxs.?, il));
                self.comp_row_of[uil] = row;
            }

            if (ci.state_snapshot_src_idxs != null and ci.state_snapshot_dst_idxs != null) {
                const snap_kv: *Tensor = c.ggml_concat(ctx, restored_kv, state_kv, 1);
                const snap_score: *Tensor = c.ggml_concat(ctx, restored_score, state_score, 1);
                c.ggml_build_forward_expand(b.gf, comp_state.cpyKv(ctx, c.ggml_get_rows(ctx, snap_kv, ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs.?, il));
                c.ggml_build_forward_expand(b.gf, comp_state.cpyScore(ctx, c.ggml_get_rows(ctx, snap_score, ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs.?, il));
            }
            c.ggml_build_forward_expand(b.gf, comp_state.cpyKv(ctx, c.ggml_get_rows(ctx, state_kv, ci.state_persist_src_idxs), ci.state_persist_dst_idxs.?, il));
            c.ggml_build_forward_expand(b.gf, comp_state.cpyScore(ctx, c.ggml_get_rows(ctx, state_score, ci.state_persist_src_idxs), ci.state_persist_dst_idxs.?, il));
        }

        if (rd == il and hp.dsv41_kv_src[uil] != il) {
            if (self.comp_row_of[@intCast(hp.dsv41_kv_src[uil])]) |row| {
                // The first consumer on this device: the source's new rows
                // appended to the local mirror.
                c.ggml_build_forward_expand(b.gf, comp_ctx.cpyK(ctx, row, ci.state_write_idxs.?, il));
            }
        }
    }

    var out: *Tensor = undefined;
    if (ratio == 0 or ci.kq_mask == null) {
        // No compressed row visible yet: the window only.
        out = deepseek4.buildRawAttention(b, inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
    } else {
        const ci_mask = ci.kq_mask.?;
        const n_comp = ci_mask.ne[0];

        const comp_all: *Tensor = comp_ctx.getK(ctx, rd);
        impl.assert(comp_all.ne[0] == n_embd_head + n_idx_embd and n_comp <= comp_all.ne[2], src, 528, "comp_all->ne[0] == n_embd_head + n_idx_embd && n_comp <= comp_all->ne[2]");
        const n_stream = comp_all.ne[3];

        const topk_src: i32 = hp.dsv41_topk_src[uil];

        var top_k_opt: ?*Tensor = self.topk_of[@intCast(topk_src)];
        if (topk_src == il) {
            var ik: *Tensor = c.ggml_view_4d(ctx, comp_all, n_idx_embd, 1, n_comp, n_stream, comp_all.nb[1], comp_all.nb[2], comp_all.nb[3], c.ggml_row_size(comp_all.type, n_embd_head));
            // The fused indexer reads the strided view in place, the
            // unfused path needs it contiguous; CUDA has no same-type copy
            // of quantized views, so they are dequantized.
            const fused_idx = self.sparse and b.cparams.fused_lid and ci_mask.type == c.GGML_TYPE_F16;
            if (c.ggml_is_quantized(ik.type)) {
                ik = c.ggml_cast(ctx, ik, c.GGML_TYPE_F32);
            } else if (!fused_idx) {
                ik = c.ggml_cont(ctx, ik);
            }

            const n_tok = @divTrunc(nt, n_stream);
            const n_tok_u: usize = @intCast(n_tok);

            var iq: *Tensor = c.ggml_reshape_3d(ctx, b.buildLoraMm(layer.indexer_attn_q_b.?, qr, null), n_idx_embd, n_idx_head, nt);
            iq = rope.apply(iq, inp_pos, n_idx_embd - n_embd_head_rope);
            iq = c.ggml_view_4d(ctx, iq, iq.ne[0], iq.ne[1], n_tok, n_stream, iq.nb[1], iq.nb[2], iq.nb[2] * n_tok_u, 0);

            var iw: *Tensor = c.ggml_scale(ctx, b.buildLoraMm(layer.indexer_proj.?, cur, null), 1.0 / impl.sqrtf(@floatFromInt(n_idx_embd * n_idx_head)));
            iw = c.ggml_view_4d(ctx, iw, iw.ne[0], n_tok, 1, n_stream, iw.nb[1], iw.nb[1] * n_tok_u, iw.nb[1] * n_tok_u, 0);

            // Queries are chunked so the [n_ih, tokens, n_comp] scores stay
            // near 2^26 elements (at most 64 chunks); the fused indexer runs
            // as one chunk.
            impl.assert(ci_mask.ne[1] == n_tok, src, 552, "ci.kq_mask->ne[1] == n_tok");
            const chunk: i64 = if (fused_idx) n_tok else @max(@max(@divTrunc(n_tok + 63, 64), @divTrunc(@as(i64, 1) << 26, n_idx_head * n_comp * n_stream)), 1);

            const cand_src: i32 = hp.dsv41_cand_src;
            const pick_blk = cand_src == il and cand_pin != null; // Level one: pick the candidate blocks.
            const use_blk = cand_src >= 0 and cand_src < il and self.cand_sel != null; // Level two: score inside them.
            const bs: i64 = hp.dsv41_cand_block_size;
            const n_blk: i64 = if (pick_blk or use_blk) @divTrunc(n_comp + bs - 1, bs) else 0;
            impl.assert(!pick_blk or cand_pin.?.ne[0] == n_blk, src, 561, "!pick_blk || cand_pin->ne[0] == n_blk");
            impl.assert(!use_blk or self.cand_n_comp == n_comp, src, 562, "!use_blk  || cand_n_comp == n_comp");

            const ik_p: ?*Tensor = if (fused_idx) null else c.ggml_permute(ctx, ik, 0, 2, 1, 3);
            var parts_top_k: std.ArrayList(*Tensor) = .empty;
            defer parts_top_k.deinit(allocator);
            var parts_sel: std.ArrayList(*Tensor) = .empty;
            defer parts_sel.deinit(allocator);

            var t0: i64 = 0;
            while (t0 < n_tok) : (t0 += chunk) {
                const nc = @min(chunk, n_tok - t0);
                var score: *Tensor = undefined;
                if (fused_idx) {
                    score = c.ggml_lightning_indexer(ctx, iq, ik, iw, ci_mask);
                    b.res.addFusedNode(.{ .op = .lightning_indexer, .tensor = score, .il = il });
                } else {
                    const iqc: *Tensor = c.ggml_view_4d(ctx, iq, iq.ne[0], iq.ne[1], nc, n_stream, iq.nb[1], iq.nb[2], iq.nb[3], @as(usize, @intCast(t0)) * iq.nb[2]);

                    var kq: *Tensor = c.ggml_mul_mat(ctx, ik_p, c.ggml_permute(ctx, iqc, 0, 2, 1, 3));
                    kq = c.ggml_cont(ctx, c.ggml_permute(ctx, kq, 2, 1, 0, 3));
                    score = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, c.ggml_relu(ctx, kq), rowsOf(ctx, iw, nc, n_stream, t0)));
                    const mask = rowsOf(ctx, ci_mask, nc, n_stream, t0);
                    score = c.ggml_add(ctx, c.ggml_cont(ctx, c.ggml_permute(ctx, score, 2, 1, 0, 3)), if (mask.type == c.GGML_TYPE_F32) mask else c.ggml_cast(ctx, mask, c.GGML_TYPE_F32));
                }
                b.cb(score, "idx_score", il);

                if (pick_blk) {
                    // The best position per block, the newest block pinned,
                    // the top blocks kept.
                    var s2: *Tensor = c.ggml_reshape_2d(ctx, score, n_comp, nc * n_stream);
                    if (n_blk * bs != n_comp) {
                        s2 = c.ggml_concat(ctx, s2, c.ggml_fill(ctx, c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_blk * bs - n_comp, nc * n_stream), -inf), 0);
                    }
                    var bmax: *Tensor = c.ggml_pool_1d(ctx, s2, c.GGML_OP_POOL_MAX, @intCast(bs), @intCast(bs), 0);
                    bmax = c.ggml_add(ctx, c.ggml_reshape_4d(ctx, bmax, n_blk, nc, 1, n_stream), rowsOf(ctx, cand_pin.?, nc, n_stream, t0));
                    b.cb(bmax, "cand_block_score", il);
                    parts_sel.append(allocator, c.ggml_top_k(ctx, bmax, @intCast(@min(@as(i64, hp.dsv41_cand_topk_blocks), n_blk)))) catch oom();
                } else if (use_blk) {
                    // Its own scores, restricted to the source's candidate
                    // blocks: 0 inside them, -inf elsewhere.
                    const bsel: *Tensor = c.ggml_cont(ctx, rowsOf(ctx, self.cand_sel.?, nc, n_stream, t0));
                    const k_b = bsel.ne[0];
                    var bm: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, 1, n_blk, nc * n_stream, 1), -inf);
                    const zeros: *Tensor = c.ggml_fill(ctx, c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, 1, k_b, nc * n_stream, 1), 0.0);
                    bm = c.ggml_set_rows(ctx, bm, zeros, c.ggml_reshape_3d(ctx, bsel, k_b, nc * n_stream, 1));
                    bm = c.ggml_reshape_2d(ctx, c.ggml_repeat_4d(ctx, bm, bs, n_blk, nc * n_stream, 1), bs * n_blk, nc * n_stream);
                    bm = c.ggml_reshape_4d(ctx, c.ggml_cont(ctx, c.ggml_view_2d(ctx, bm, n_comp, nc * n_stream, bm.nb[1], 0)), n_comp, nc, 1, n_stream);
                    b.cb(bm, "cand_mask", il);
                    score = c.ggml_add(ctx, score, bm);
                }

                parts_top_k.append(allocator, c.ggml_top_k(ctx, score, @intCast(@min(n_comp, @as(i64, hp.indexer_top_k))))) catch oom();
            }

            if (pick_blk) {
                self.cand_sel = join(ctx, &parts_sel);
                self.cand_n_comp = n_comp;
            }
            const joined: *Tensor = c.ggml_cont(ctx, join(ctx, &parts_top_k));
            b.cb(joined, "idx_top_k", il);
            self.topk_of[uil] = joined;
            top_k_opt = joined;
        }
        impl.assert(top_k_opt != null, src, 632, "top_k && \"DeepSeek-V4.1: top-k source layer produced no top-k\"");
        const top_k = top_k_opt.?;

        // A quantized K cache stores the window Hadamard-rotated, as
        // build_raw_attention does.
        const k_rot = inp_attn.self_k_rot;
        if (k_rot) |r| {
            q = graph.mulMatHadamard(ctx, q, r);
            kv = graph.mulMatHadamard(ctx, kv, r);
        }
        c.ggml_build_forward_expand(b.gf, q);
        c.ggml_build_forward_expand(b.gf, kv);
        c.ggml_build_forward_expand(b.gf, inp_attn.mctx.cpyK(ctx, kv, inp_attn.getKIdxs(), il));

        var raw_k: *Tensor = inp_attn.mctx.getK(ctx, il);
        const raw_mask: *Tensor = inp_attn.getKqMask().?;

        const k_sel = top_k.ne[0];
        const n_raw = raw_k.ne[2];

        // Few queries per stream: attend over [window | top-k rows of the
        // query], one FA stream per query; the same keys and mask as dense,
        // in a different key order.
        const gather = self.sparse and b.cparams.flash_attn and raw_mask.type == c.GGML_TYPE_F16 and ci_mask.type == c.GGML_TYPE_F16 and
            ((n_stream == 1 and nt <= 32) or nt == n_stream);

        if (gather) {
            // ggml_top_k returns the rows in no fixed order and FA sums the
            // keys in that order: they are sorted, so the result is
            // deterministic.
            const sel_entry = self.gather_sel_of.getOrPutValue(allocator, topk_src, null) catch oom();
            if (sel_entry.value_ptr.* == null) {
                impl.assert(n_comp <= (@as(i64, 1) << 24), src, 659, "n_comp <= (INT64_C(1) << 24)"); // Row ids are exact in F32.
                const ids: *Tensor = c.ggml_reshape_2d(ctx, top_k, k_sel, nt);
                const ord: *Tensor = c.ggml_argsort(ctx, c.ggml_cast(ctx, ids, c.GGML_TYPE_F32), c.GGML_SORT_ORDER_ASC);
                const t: *Tensor = c.ggml_reshape_2d(ctx, c.ggml_get_rows(ctx, c.ggml_reshape_3d(ctx, ids, 1, k_sel, nt), ord), k_sel, nt);
                b.cb(t, "comp_top_k_sorted", il);
                sel_entry.value_ptr.* = t;
            }
            const sel = sel_entry.value_ptr.*.?;

            // Built once per (storage, top-k) pair.
            const k_entry = self.gather_k_of.getOrPutValue(allocator, .{ rd, topk_src }, null) catch oom();
            if (k_entry.value_ptr.* == null) {
                var comp_k: *Tensor = undefined;
                if (n_stream == 1) {
                    const src_rows: *Tensor = c.ggml_view_3d(ctx, comp_all, n_embd_head, n_comp, 1, comp_all.nb[2], comp_all.nb[3], 0);
                    comp_k = c.ggml_get_rows(ctx, src_rows, c.ggml_reshape_1d(ctx, sel, k_sel * nt));
                } else {
                    const src_rows: *Tensor = c.ggml_view_3d(ctx, comp_all, n_embd_head, n_comp, n_stream, comp_all.nb[2], comp_all.nb[3], 0);
                    comp_k = c.ggml_get_rows(ctx, src_rows, sel);
                }
                comp_k = c.ggml_reshape_4d(ctx, comp_k, n_embd_head, 1, k_sel, nt); // F32: get_rows dequantizes.
                if (k_rot) |r| {
                    // A quantized K cache: the window is stored rotated, the
                    // compressed rows are not.
                    comp_k = graph.mulMatHadamard(ctx, comp_k, r);
                }
                if (comp_k.type != raw_k.type) {
                    comp_k = c.ggml_cast(ctx, comp_k, raw_k.type); // Exact: the rows are raw_k's type's values.
                }
                b.cb(comp_k, "comp_k_sel", il);
                k_entry.value_ptr.* = comp_k;
            }
            const comp_k = k_entry.value_ptr.*.?;

            if (n_stream == 1 and nt > 1) {
                if (c.ggml_is_quantized(raw_k.type)) {
                    // No REPEAT kernel for quantized types: byte-exact copies
                    // by doubling concats.
                    var rep: ?*Tensor = null;
                    var pw: *Tensor = raw_k;
                    var n: i64 = nt;
                    while (n > 0) : (n >>= 1) {
                        if (n & 1 != 0) {
                            rep = if (rep) |r| c.ggml_concat(ctx, r, pw, 3) else pw;
                        }
                        if (n > 1) {
                            pw = c.ggml_concat(ctx, pw, pw, 3);
                        }
                    }
                    raw_k = rep.?;
                } else {
                    raw_k = c.ggml_repeat_4d(ctx, raw_k, n_embd_head, 1, n_raw, nt);
                }
            }
            const k_all: *Tensor = c.ggml_concat(ctx, raw_k, comp_k, 2);

            // The window mask, then the comp mask at the selected rows (-inf
            // where top-k picked an invisible row); one per top-k source.
            const mask_entry = self.gather_mask_of.getOrPutValue(allocator, topk_src, null) catch oom();
            if (mask_entry.value_ptr.* == null) {
                var sel_mask: *Tensor = c.ggml_get_rows(ctx, c.ggml_reshape_3d(ctx, ci_mask, 1, n_comp, nt), sel);
                sel_mask = c.ggml_cast(ctx, c.ggml_reshape_4d(ctx, sel_mask, k_sel, 1, 1, nt), c.GGML_TYPE_F16);
                const t: *Tensor = c.ggml_concat(ctx, c.ggml_reshape_4d(ctx, raw_mask, raw_mask.ne[0], 1, 1, nt), sel_mask, 0);
                b.cb(t, "comp_kq_mask_sel", il);
                mask_entry.value_ptr.* = t;
            }

            out = b.buildAttnMha(q, k_all, k_all, null, mask_entry.value_ptr.*.?, layer.attn_sinks, null, 0, kq_scale, il);
        } else {
            // Dense attention over [window | all compressed rows], the
            // selection applied as a mask.
            var comp_k: *Tensor = c.ggml_view_4d(ctx, comp_all, n_embd_head, 1, n_comp, n_stream, comp_all.nb[1], comp_all.nb[2], comp_all.nb[3], 0);
            if (k_rot) |r| {
                // Compressed rows are stored unrotated: rotated in F32, then
                // cast to the cache's type.
                comp_k = if (comp_k.type != c.GGML_TYPE_F32) c.ggml_cast(ctx, comp_k, c.GGML_TYPE_F32) else c.ggml_cont(ctx, comp_k);
                comp_k = graph.mulMatHadamard(ctx, comp_k, r);
                if (comp_k.type != raw_k.type) {
                    comp_k = c.ggml_cast(ctx, comp_k, raw_k.type);
                }
            } else if (c.ggml_is_quantized(comp_k.type)) {
                // A quantized cache without rotation
                // (LLAMA_ATTN_ROT_DISABLE=1): through F32 for the same
                // reason.
                comp_k = c.ggml_cast(ctx, c.ggml_cast(ctx, comp_k, c.GGML_TYPE_F32), raw_k.type);
            } else {
                comp_k = if (comp_k.type != raw_k.type) c.ggml_cast(ctx, comp_k, raw_k.type) else c.ggml_cont(ctx, comp_k);
            }
            b.cb(comp_k, "comp_k", il);
            const k_all: *Tensor = c.ggml_concat(ctx, raw_k, comp_k, 2);

            const sel_mask = deepseek4.buildTopKMask(b, ci_mask, top_k, "comp_top_k_mask", il);
            const kq_mask: *Tensor = c.ggml_concat(ctx, raw_mask, sel_mask, 0);
            b.cb(kq_mask, "comp_kq_mask", il);

            // At most window + top-k visible columns per query: lets CUDA FA
            // gather only those.
            const n_kv_max: i64 = if (self.sparse) @min(raw_mask.ne[0], @as(i64, hp.n_swa)) + k_sel else 0;
            out = b.buildAttnMha(q, k_all, k_all, null, kq_mask, layer.attn_sinks, null, @intCast(n_kv_max), kq_scale, il);
        }
        if (k_rot) |r| {
            out = graph.mulMatHadamard(ctx, out, r);
        }
        b.cb(out, "attn_comp", il);
    }

    out = c.ggml_reshape_3d(ctx, out, n_embd_head, b.n_head, nt);
    out = c.ggml_rope_ext_back(ctx, out, inp_pos, null, @intCast(n_embd_head_rope), b.rope_type, rope.n_ctx_orig, rope.freq_base, rope.freq_scale, rope.ext_factor, rope.attn_factor, rope.beta_fast, rope.beta_slow);
    out = c.ggml_rope_set_offset(out, @intCast(n_embd_head_nope));
    b.cb(out, "attn_derope", il);

    out = c.ggml_permute(ctx, c.ggml_reshape_3d(ctx, out, @divTrunc(b.n_head, n_groups) * n_embd_head, n_groups, nt), 0, 2, 1, 3);
    var oa: *Tensor = c.ggml_mul_mat(ctx, layer.wo_a, out);
    b.cb(oa, "attn_wo_a", il);
    oa = c.ggml_cont_2d(ctx, c.ggml_permute(ctx, oa, 0, 2, 1, 3), o_lora_rank * n_groups, nt);

    out = b.buildLoraMm(layer.wo_b.?, oa, null);
    b.cb(out, "attn_out", il);

    return out;
}

/// Ports the constructor of `llama_model_deepseek41`'s `graph`
/// (models/deepseek41.cpp:768 @8642541b2).
fn buildGraph(model: *const Model, b: *const Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_layer: usize = @intCast(b.n_layer);

    var g: Graph = .{
        .topk_of = allocator.alloc(?*Tensor, n_layer) catch oom(),
        .comp_row_of = allocator.alloc(?*Tensor, n_layer) catch oom(),
        .sparse = envOn("LLAMA_DSV41_SPARSE"),
    };
    defer g.deinit();
    @memset(g.topk_of, null);
    @memset(g.comp_row_of, null);

    var cur: *Tensor = undefined;

    const inp = b.buildInpEmbd(model.tok_embd.?);
    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();
    const inp_dsv4 = b.buildInpDsv4();
    c.ggml_build_forward_expand(b.gf, inp_dsv4.getRaw().self_kq_mask);

    const inp41 = allocator.create(InpDsv41) catch oom();
    inp41.* = .{ .model = model, .mctx = inp_dsv4.mctx };
    // An embedding input has no token ids: it skips Engram, like the
    // reference's masked (image) positions. The gathers go right after the
    // token embeddings, so they share the first graph split (the table stays
    // on the host).
    var engram_rows: std.ArrayList(*Tensor) = .empty;
    defer engram_rows.deinit(allocator);
    if (b.ubatch.token != null and hp.ple_n_heads > 0) {
        var il: c_int = 0;
        while (il < b.n_layer) : (il += 1) {
            if (!hp.isPle(@intCast(il))) {
                continue;
            }
            const ids: *Tensor = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, @as(i64, hp.ple_n_heads) * b.n_tokens);
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrintSentinel(&name_buf, "engram_ids-{d}", .{il}, 0) catch unreachable;
            _ = c.ggml_set_name(ids, name.ptr);
            c.ggml_set_input(ids);
            inp41.engram_ids.append(allocator, ids) catch oom();

            var rows: *Tensor = c.ggml_get_rows(ctx, model.per_layer_tok_embd, ids);
            rows = c.ggml_reshape_2d(ctx, rows, hp.ple_head_dim *% hp.ple_n_heads, b.n_tokens);
            b.cb(rows, "engram_rows", il);
            c.ggml_build_forward_expand(b.gf, rows);
            engram_rows.append(allocator, rows) catch oom();
        }
    }
    var engram_keep: ?*Tensor = null;
    if (hp.ple_n_heads > 0 and b.ubatch.token != null and b.ubatch.embd != null) {
        engram_keep = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, 1, 1, b.n_tokens);
        c.ggml_set_input(engram_keep);
        _ = c.ggml_set_name(engram_keep, "engram_keep");
        inp41.engram_keep = engram_keep;
    }
    var cand_pin: ?*Tensor = null;
    if (hp.dsv41_cand_src >= 0) {
        inp41.cand_csa = hp.dsv4_compress_ratios[@intCast(hp.dsv41_cand_src)] == 2;
        const ci = if (inp41.cand_csa) inp_dsv4.getCsa() else inp_dsv4.getHca();
        if (ci.kq_mask) |m| {
            const bs: i64 = hp.dsv41_cand_block_size;
            cand_pin = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, @divTrunc(m.ne[0] + bs - 1, bs), m.ne[1], 1, m.ne[3]);
            c.ggml_set_input(cand_pin);
            _ = c.ggml_set_name(cand_pin, "cand_pin");
            inp41.cand_pin = cand_pin;
        }
    }
    _ = b.res.addInput(InpDsv41, inp41);

    const hc: i64 = hp.dsv4_hc_mult;
    var inpL: *Tensor = c.ggml_repeat_4d(ctx, c.ggml_reshape_3d(ctx, inp, b.n_embd, 1, b.n_tokens), b.n_embd, hc, b.n_tokens, 1);
    b.cb(inpL, "hc_init", -1);

    // Single-pass hyper-connections: the pre-mix that collapses the stream
    // lags one sublayer.
    var pre_prev: ?*Tensor = null;

    const layer_inp = b.cparams.embeddings_layer_inp;

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const uil: usize = @intCast(il);
        const layer = &model.layers[uil];

        const e = engramIndex(model, il);
        if (e >= 0 and engram_rows.items.len > 0) {
            inpL = buildEngram(b, model, inpL, engram_rows.items[@intCast(e)], engram_keep, il);
            b.cb(inpL, "engram_out", il);
        }

        // DSpark's target features: the mean over the hc copies of the
        // layer's input.
        if (uil < layer_inp.len and layer_inp[uil]) {
            const t = deepseek4.hcMean(ctx, inpL);
            b.res.t_layer_inp[uil] = t;
            b.cb(t, "layer_inp", il);
            c.ggml_build_forward_expand(b.gf, t);
        }

        var residual = inpL;
        const attn_mix = deepseek4.buildHcMixes(b, inpL, layer.hc_attn_fn.?, layer.hc_attn_scale.?, layer.hc_attn_base.?, il);

        // The first layer's attention collapses with the identity pre-mix:
        // copy 0.
        cur = if (pre_prev) |pre| deepseek4.buildHcPreWeighted(b, inpL, pre, il) else c.ggml_cont(ctx, c.ggml_view_2d(ctx, inpL, b.n_embd, b.n_tokens, inpL.nb[2], 0));
        b.cb(cur, "hc_attn_pre", il);

        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        cur = buildAttention41(&g, b, model, inp_dsv4, cur, inp_pos, cand_pin, il);

        inpL = deepseek4.buildHcPost(b, cur, residual, attn_mix.post, attn_mix.comb, il);
        b.cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const ffn_mix = deepseek4.buildHcMixes(b, inpL, layer.hc_ffn_fn.?, layer.hc_ffn_scale.?, layer.hc_ffn_base.?, il);
        pre_prev = ffn_mix.pre;

        cur = deepseek4.buildHcPreWeighted(b, inpL, attn_mix.pre, il);
        b.cb(cur, "hc_ffn_pre", il);

        c.ggml_build_forward_expand(b.gf, residual);
        c.ggml_build_forward_expand(b.gf, ffn_mix.post);
        c.ggml_build_forward_expand(b.gf, ffn_mix.comb);

        cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // F32 router accumulation: the expert selection is near-tie
        // sensitive.
        const logits = b.buildLoraMm(layer.ffn_gate_inp.?, cur, null);
        _ = c.ggml_prec_set_acc(logits, c.GGML_PREC_F32);
        b.cb(logits, "ffn_moe_logits", il);

        // Image (embedding) tokens use the vision router bias, as
        // DeepSeek-V4.
        var exp_probs_b = layer.ffn_exp_probs_b;
        if (b.ubatch.embd != null and layer.ffn_exp_probs_b_vl != null) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        // The reference normalizes only when it routes to more than one
        // expert.
        const moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = exp_probs_b,
            .n_expert = b.n_expert,
            .n_expert_used = hp.nExpertUsed(0),
            .type_op = .silu,
            .norm_w = hp.expert_weights_norm and hp.nExpertUsed(0) > 1,
            .w_scale = hp.expert_weights_scale,
            .gating_op = @fromBackingInt(hp.expert_gating_func),
            .probs_in = logits,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
        b.cb(ffn_shexp, "ffn_shexp", il);

        cur = c.ggml_add(ctx, moe_out, ffn_shexp);
        b.cb(cur, "ffn_out", il);

        inpL = deepseek4.buildHcPost(b, cur, residual, ffn_mix.post, ffn_mix.comb, il);
        inpL = b.buildCvec(inpL, il);
        b.cb(inpL, "l_out", il);
    }

    // No hc_head: collapse with the last FFN's pre-mix.
    inpL = c.ggml_reshape_3d(ctx, c.ggml_get_rows(ctx, c.ggml_reshape_2d(ctx, inpL, b.n_embd * hc, b.n_tokens), inp_out_ids), b.n_embd, hc, inp_out_ids.ne[0]);
    pre_prev = c.ggml_get_rows(ctx, pre_prev, inp_out_ids);
    cur = deepseek4.buildHcPreWeighted(b, inpL, pre_prev.?, -1);
    b.cb(cur, "hc_out", -1);

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = c.ggml_mul_mat(ctx, model.output, cur);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
