//! GLM-5 (GLM-DSA): DeepSeek V3.2's blocks -- absorbed MLA over a K-only
//! cache with DeepSeek sparse attention, leading dense blocks, then a
//! mixture of experts with shared experts -- with the lightning indexer
//! roped in the normal layout, and, from GLM 5.2 on, "shared" indexer
//! layers that reuse the top-k of the previous "full" one. An optional
//! NextN/MTP block past the main stack runs dense MLA.
//!
//! The graphs are DeepSeek V3.2's (`deepseek32.zig`), built with GLM-DSA's
//! `Variant`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/glm-dsa.cpp` at v0.4.1
//! (`b29c606e2`): `GLM_5_2_DEFAULT_INDEXER_TYPES`, `llama_model_glm_dsa`'s
//! `load_arch_hparams`, `load_arch_tensors` and `build_arch_graph`, and its
//! `graph` and `graph_mtp`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const deepseek32 = @import("deepseek32.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;

/// Mirrors `GLM_5_2_DEFAULT_INDEXER_TYPES` (models/glm-dsa.cpp:6 @b29c606e2):
/// GLM-5.2's indexer types, a full indexer every fourth layer after the
/// first two
/// (https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json#L26).
const glm_5_2_default_indexer_types: [hparams_mod.max_layers]u32 = blk: {
    var t: [hparams_mod.max_layers]u32 = @splat(0);
    t[0] = 1;
    t[1] = 1;
    var i: usize = 2;
    while (i < 2 + 19 * 4) : (i += 4) t[i] = 1;
    break :blk t;
};

/// Ports `llama_model_glm_dsa`'s `load_arch_hparams` (models/glm-dsa.cpp:29 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, false, exc);

    // MoE.
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;

    // DeepSeek's MLA.
    hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    if (try meta.getKey(u32, .attention_key_length_mla, false, exc)) |v| hp.n_embd_head_k_mla_impl = v;
    if (try meta.getKey(u32, .attention_value_length_mla, false, exc)) |v| hp.n_embd_head_v_mla_impl = v;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;

    // DSA.
    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;

    // The expert gating function; GLM-4.5 uses sigmoid.
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
    }

    // Backwards compatibility for GLM 5 and 5.1, whose indexers are all
    // full and which have no `indexer_types`.
    const is_pre_5_2 = hp.n_ctx_train < 1048576;
    if (is_pre_5_2) {
        @memset(&hp.is_indexer_full_impl, 1);
    } else {
        hp.is_indexer_full_impl = glm_5_2_default_indexer_types;
    }
    _ = try meta.getKeyOrArr(u32, .attention_indexer_types, &hp.is_indexer_full_impl, hp.nLayer(), false, exc);

    self.type = switch (hp.nLayer()) {
        78 => .@"744b_a40b",
        else => .unknown,
    };
}

/// Ports `llama_model_glm_dsa`'s `load_arch_tensors` (models/glm-dsa.cpp:74 @b29c606e2).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const n_expert_shared: i64 = hp.n_expert_shared;
    const nr = loader.tensor_not_required;

    // MTP-only: the GGUF carries the NextN/MTP blocks alone (a user split
    // of target and draft).
    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    // Trunk-only: the GGUF declares MTP layers in its metadata, but their
    // tensors live in a separate file or were stripped at conversion, so
    // they are not required and the trunk loads cleanly.
    var probe_buf: [64]u8 = undefined;
    const mtp_probe = std.fmt.bufPrint(&probe_buf, "blk.{d}.nextn.eh_proj.weight", .{l.n_layer}) catch unreachable;
    const trunk_only = hp.n_layer_nextn > 0 and ml.getWeight(mtp_probe) == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    var mtp_flags: u32 = if (trunk_only) nr else 0;

    if (!ml.load_mtp) mtp_flags |= loader.tensor_skip;

    const is_mla = hp.isMla();
    if (!is_mla) {
        return exc.throw(.runtime_error, "GLM_DSA architecture requires MLA", .{});
    }

    // The head sizes as MHA, or after "decompression" through `wv_b` with
    // MLA.
    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_v_mla: i64 = hp.nEmbdHeadVMla();

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const q_lora_rank: i64 = hp.n_lora_q;
    const kv_lora_rank: i64 = hp.n_lora_kv;

    const n_ff_exp: i64 = hp.nFfExp(0);

    const indexer_head_size: i64 = hp.indexer_head_size;
    const indexer_n_head: i64 = hp.indexer_n_head;

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
        // The NextN/MTP layers (`i >= n_layer`) are whole decoder blocks
        // the `LLM_GRAPH_TYPE_DECODER_MTP` draft head uses, loaded as
        // qwen35moe, step35 and hy_v3 load theirs.
        const flags = if (i >= l.n_layer) mtp_flags else trunk_flags;

        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, flags, exc);
        layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, flags, exc);

        layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_lora_rank }, flags, exc);
        layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, l.n_head * n_embd_head_k_mla }, flags, exc);

        layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ l.n_embd, kv_lora_rank + n_embd_head_qk_rope }, flags, exc);

        // Only old legacy GGUF files have the unsplit `wkv_b`.
        layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_qk_nope, kv_lora_rank, l.n_head }, flags, exc);
        layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v_mla, l.n_head }, flags, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_head * n_embd_head_v_mla, l.n_embd }, flags, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        // The DSA indexer, which shared indexer layers have none of.
        layer.indexer_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{indexer_head_size}, flags | nr, exc);
        layer.indexer_k_norm_b = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "bias", .bid = i }, &.{indexer_head_size}, flags | nr, exc);
        layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, indexer_n_head }, flags | nr, exc);
        layer.indexer_attn_k = try self.createTensor(.{ .tensor = .indexer_attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, indexer_head_size }, flags | nr, exc);
        layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, indexer_n_head * indexer_head_size }, flags | nr, exc);
        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, flags, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, flags, exc);
        } else {
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, flags, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr, exc);

            if (l.n_expert == 0) {
                return exc.throw(.runtime_error, "n_expert must be > 0", .{});
            }
            if (l.n_expert_used == 0) {
                return exc.throw(.runtime_error, "n_expert_used must be > 0", .{});
            }

            // The routed experts.
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, flags, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, flags, exc);

            // The shared experts.
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, flags, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, flags, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, flags, exc);
        }

        // The NextN/MTP tensors: the NextN wiring around the extra decoder
        // block.
        if (i >= l.n_layer) {
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

            // Optional.
            layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, flags | nr, exc);
            layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, flags | nr, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags | nr, exc);
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// What GLM-DSA's graphs change in DeepSeek V3.2's.
const variant: deepseek32.Variant = .{
    .file = "models/glm-dsa.cpp",
    .name = "GLM_DSA",
    .indexer_rope_type = c.LLAMA_ROPE_TYPE_NORM,
    .shared_indexers = true,
    .line_is_mla = 199,
    .line_n_rot = 214,
    .line_ext_factor = 223,
    .line_prev_top_k = 364,
    .line_mtp = 541,
    .line_mtp_inp = 573,
    .line_mtp_head_norm = 752,
    .line_mtp_head = 763,
};

/// Ports `llama_model_glm_dsa`'s `build_arch_graph` (models/glm-dsa.cpp:189
/// @b29c606e2): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

/// Ports the constructor of `llama_model_glm_dsa`'s `graph`
/// (models/glm-dsa.cpp:196 @b29c606e2): DeepSeek V3.2's, with the indexer
/// roped in the normal layout and the shared indexer layers reusing the
/// previous full layer's top-k.
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    deepseek32.buildGraph(variant, model, b);
}

/// Ports the constructor of `llama_model_glm_dsa`'s `graph_mtp`
/// (models/glm-dsa.cpp:539 @b29c606e2): DeepSeek V3.2's, whose DSA indexer
/// does not run either.
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    deepseek32.buildGraphMtp(variant, model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "GLM-5.2's default indexer types match the C++'s table" {
    // 1, 1, then 1, 0, 0, 0 nineteen times.
    try std.testing.expectEqual(@as(u32, 1), glm_5_2_default_indexer_types[0]);
    try std.testing.expectEqual(@as(u32, 1), glm_5_2_default_indexer_types[1]);
    try std.testing.expectEqual(@as(u32, 1), glm_5_2_default_indexer_types[2]);
    try std.testing.expectEqual(@as(u32, 0), glm_5_2_default_indexer_types[3]);
    try std.testing.expectEqual(@as(u32, 1), glm_5_2_default_indexer_types[74]);
    try std.testing.expectEqual(@as(u32, 0), glm_5_2_default_indexer_types[77]);
    try std.testing.expectEqual(@as(u32, 0), glm_5_2_default_indexer_types[78]);
    var n: u32 = 0;
    for (glm_5_2_default_indexer_types) |t| n += t;
    try std.testing.expectEqual(@as(u32, 21), n);
}
