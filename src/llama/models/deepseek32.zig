//! DeepSeek V3.2: DeepSeek V3's blocks -- absorbed multi-head latent
//! attention over a K-only cache, leading dense blocks, then a mixture of
//! experts with shared experts -- with DeepSeek sparse attention. A
//! lightning indexer scores every cached key for each query through its own
//! small MQA head and cache, and attention sees only the top-k keys it
//! chose. An optional NextN/MTP block past the main stack runs dense MLA,
//! without the indexer.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/deepseek32.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_deepseek32`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph` and
//! `graph_mtp`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_deepseek32`'s `load_arch_hparams` (models/deepseek32.cpp:6 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    // The eps of the layer norm.
    hp.f_norm_eps = 1e-6;
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
    hp.n_ff_exp = (try meta.getKey(u32, .expert_feed_forward_length, true, exc)).?;
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;

    // DSA.
    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;

    // The expert gating function.
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;

    if (try meta.getKey(f32, .rope_scaling_yarn_log_mul, false, exc)) |v| {
        hp.rope_yarn_log_mul = v;
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] Cancel the factor the convert
        // script applied.
        hp.rope_yarn_log_mul /= 0.1;
    }

    // NextN/MTP.
    if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
    impl.assert(hp.n_layer_nextn < hp.n_layer_all, "models/deepseek32.cpp", 42, "hparams.n_layer_nextn < hparams.n_layer_all && \"n_layer_nextn must be < n_layer\"");

    self.type = switch (hp.nLayer()) {
        61 => .@"685b_a37b",
        else => .unknown,
    };
}

/// Ports `llama_model_deepseek32`'s `load_arch_tensors` (models/deepseek32.cpp:50 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    var probe_buf: [64]u8 = undefined;
    const mtp_probe = std.fmt.bufPrint(&probe_buf, "blk.{d}.nextn.eh_proj.weight", .{l.n_layer}) catch unreachable;
    const trunk_only = hp.n_layer_nextn > 0 and ml.getWeight(mtp_probe) == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    var mtp_flags: u32 = if (trunk_only) nr else 0;

    if (!ml.load_mtp) mtp_flags |= loader.tensor_skip;

    const is_mla = hp.isMla();
    if (!is_mla) {
        return exc.throw(.runtime_error, "DEEPSEEK32 architecture requires MLA", .{});
    }

    // The head sizes as MHA, or after "decompression" through `wv_b` with
    // MLA.
    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_v_mla: i64 = hp.nEmbdHeadVMla();

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const q_lora_rank: i64 = hp.n_lora_q;
    const kv_lora_rank: i64 = hp.n_lora_kv;

    const n_ff_exp: i64 = hp.n_ff_exp;
    const n_expert_shared: i64 = hp.n_expert_shared;

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

        // The DSA indexer.
        layer.indexer_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{indexer_head_size}, flags, exc);
        layer.indexer_k_norm_b = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "bias", .bid = i }, &.{indexer_head_size}, flags, exc);
        layer.indexer_proj = try self.createTensor(.{ .tensor = .indexer_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, indexer_n_head }, flags, exc);
        layer.indexer_attn_k = try self.createTensor(.{ .tensor = .indexer_attn_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, indexer_head_size }, flags, exc);
        layer.indexer_attn_q_b = try self.createTensor(.{ .tensor = .indexer_attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, indexer_n_head * indexer_head_size }, flags, exc);
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

        // The NextN/MTP tensors, of the last `nextn_predict_layers` layers.
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

/// Ports `llama_model_deepseek32`'s `build_arch_graph` (models/deepseek32.cpp:159
/// @c1d0e7a00): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(variant, model, b);
    return buildGraph(variant, model, b);
}

/// What GLM-DSA's graphs (`glm_dsa.zig`) change in these: the indexer's
/// RoPE, the layers that reuse the previous full indexer's top-k, and the
/// source the assertions name.
pub const Variant = struct {
    file: [:0]const u8,
    /// The architecture's name in the MTP head's assertions.
    name: [:0]const u8,
    /// The RoPE mode of the indexer's q and k.
    indexer_rope_type: c_int,
    /// Whether layers without a full indexer reuse the previous full
    /// layer's top-k.
    shared_indexers: bool,
    /// The lines of `graph`'s assertions: `is_mla`, the indexer's RoPE
    /// width, `ext_factor`, and the shared indexer's predecessor.
    line_is_mla: c_int,
    line_n_rot: c_int,
    line_ext_factor: c_int,
    line_prev_top_k: c_int = 0,
    /// The line of `graph_mtp`'s first assertion; the rest follow it as in
    /// both files.
    line_mtp: c_int,
    line_mtp_inp: c_int,
    line_mtp_head_norm: c_int,
    line_mtp_head: c_int,
};

const variant: Variant = .{
    .file = "models/deepseek32.cpp",
    .name = "DEEPSEEK32",
    .indexer_rope_type = c.LLAMA_ROPE_TYPE_NEOX,
    .shared_indexers = false,
    .line_is_mla = 169,
    .line_n_rot = 184,
    .line_ext_factor = 193,
    .line_mtp = 501,
    .line_mtp_inp = 533,
    .line_mtp_head_norm = 712,
    .line_mtp_head = 723,
};

/// The YaRN pre-scaling both graphs share: the KQ scale, from the
/// `attn_factor` `llama_hparams`'s `yarn_attn_factor_adjust` adjusted.
fn mlaKqScale(b: *const graph.Builder, n_embd_head_k: i64) f32 {
    const hp = b.hparams;
    const attn_factor_org = b.attn_factor * (1.0 + 0.1 * impl.logf(1.0 / b.freq_scale));
    const mscale = attn_factor_org * (1.0 + 0.1 * hp.rope_yarn_log_mul * impl.logf(1.0 / b.freq_scale));
    return 1.0 * mscale * mscale / impl.sqrtf(@floatFromInt(n_embd_head_k));
}

/// Ports the constructor of `llama_model_deepseek32`'s `graph`
/// (models/deepseek32.cpp:166 @c1d0e7a00), and with `v` GLM-DSA's.
pub fn buildGraph(comptime v: Variant, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const is_mla = hp.isMla();
    impl.assert(is_mla, v.file, v.line_is_mla, "is_mla");

    // The head sizes as MHA, or after "decompression" through `wv_b` with
    // MLA.
    const n_embd_head_k: i64 = hp.nEmbdHeadKMla();

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const n_indexer_head: i64 = hp.indexer_n_head;
    const n_embd_indexer_head: i64 = hp.indexer_head_size;
    const n_indexer_top_k: u32 = hp.indexer_top_k;

    // The indexer head's layout is `[rope | nope]`.
    impl.assert(hp.nRot(0) <= n_embd_indexer_head, v.file, v.line_n_rot, "hparams.n_rot() <= n_embd_indexer_head");

    const kv_lora_rank: u32 = hp.n_lora_kv;

    // `kq_scale` and `attn_factor` are pre-scaled for YaRN RoPE to work
    // (https://github.com/ggml-org/llama.cpp/discussions/7416, and
    // https://github.com/ggml-org/llama.cpp/pull/17945
    // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]): first the adjustment from
    // `llama_hparams`'s `yarn_attn_factor_adjust` is cancelled to recover
    // the original `attn_factor`, which then pre-scales `kq_scale`.
    impl.assert(b.ext_factor >= 0.0, v.file, v.line_ext_factor, "ext_factor >= 0.0f");
    const kq_scale = mlaKqScale(b, n_embd_head_k);

    // `{n_embd, n_tokens}`.
    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn_dsa = b.buildAttnInpKDsa();

    const inp_out_ids = b.buildInpOutIds();

    // GLM-DSA's shared indexer layers reuse the top-k of the previous full
    // indexer layer
    // (https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json#L30).
    var prev_top_k: ?*Tensor = null;

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            var qr: *Tensor = c.ggml_mul_mat(ctx, layer.wq_a, cur);
            b.cb(qr, "qr", il);

            qr = b.buildNorm(qr, layer.attn_q_a_norm, null, .rms, il);
            b.cb(qr, "qr", il);

            var top_k: *Tensor = undefined;
            if (!v.shared_indexers or hp.isIndexerFull(@intCast(il))) {
                // A "full" indexer layer.
                top_k = lightningIndexer(b, v.indexer_rope_type, layer, inp_attn_dsa, cur, qr, inp_pos, n_indexer_head, n_embd_indexer_head, n_indexer_top_k, il);
                prev_top_k = top_k;
                b.cb(top_k, "top_k", il);
            } else {
                // A "shared" indexer layer, which reuses the top-k of a
                // previous full layer.
                impl.assert(prev_top_k != null, v.file, v.line_prev_top_k, "prev_top_k != nullptr && \"shared indexer layer must follow a previous full indexer layer\"");
                top_k = prev_top_k.?;
                b.cb(top_k, "top_k", il);
            }

            const q: *Tensor = c.ggml_mul_mat(ctx, layer.wq_b, qr);
            b.cb(q, "q", il);

            // Split into `{n_embd_head_qk_nope, n_head, n_tokens}`.
            var q_nope: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_nope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * @as(usize, @intCast(b.n_head)), 0);
            b.cb(q_nope, "q_nope", il);

            // And `{n_embd_head_qk_rope, n_head, n_tokens}`.
            var q_pe: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_rope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * @as(usize, @intCast(b.n_head)), c.ggml_row_size(q.type, n_embd_head_qk_nope));
            b.cb(q_pe, "q_pe", il);

            const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
            b.cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // Split into `{kv_lora_rank, n_tokens}`.
            var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), 0);
            b.cb(kv_cmpr, "kv_cmpr", il);

            // And `{n_embd_head_qk_rope, 1, n_tokens}`.
            var k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, n_embd_head_qk_rope, 1, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));
            b.cb(k_pe, "k_pe", il);

            q_pe = c.ggml_rope_ext(ctx, q_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            b.cb(q_pe, "q_pe", il);

            k_pe = c.ggml_rope_ext(ctx, k_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            b.cb(k_pe, "k_pe", il);

            kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
            b.cb(kv_cmpr, "kv_cmpr", il);

            // MLA attention.
            {
                // `{n_embd_head_qk_nope, n_tokens, n_head}`.
                q_nope = c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
                b.cb(q_nope, "q_nope_perm", il);

                // `{n_embd_head_qk_nope, kv_lora_rank, n_head}` x
                // `{n_embd_head_qk_nope, n_tokens, n_head}`.
                var q_nope_absorbed: *Tensor = c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
                b.cb(q_nope_absorbed, "q_nope_absorbed", il);

                // `{kv_lora_rank, n_head, n_tokens}`.
                q_nope_absorbed = c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);
                b.cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // `{n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}`;
                // RoPE goes first, for the in-place context shift of
                // `build_rope_shift`.
                const Qcur: *Tensor = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
                b.cb(Qcur, "Qcur", il);

                kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, b.n_tokens);
                b.cb(kv_cmpr, "kv_cmpr_reshape", il);

                // `{n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}`.
                const Kcur: *Tensor = c.ggml_concat(ctx, kv_cmpr, k_pe, 0);
                b.cb(Kcur, "Kcur", il);

                // `{kv_lora_rank, 1, n_tokens}`.
                const Vcur = kv_cmpr;
                b.cb(Vcur, "Vcur", il);

                // MLA with the absorption optimization turns into MQA (GQA
                // with one group).
                cur = b.buildAttnDsa(inp_attn_dsa, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, layer.wv_b, top_k, kq_scale, il);
            }
        }
        // When unmasked nextn embeddings are requested, `t_h_nextn` must
        // keep every row, so the early output masking is skipped here and
        // applied after the final norm instead.
        if (il == b.n_layer - 1 and (!b.cparams.embeddings_nextn or b.cparams.embeddings_nextn_masked)) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The routed experts.
            const moe_out = buildMoe(b, layer, cur, il);
            b.cb(moe_out, "ffn_moe_out", il);

            // The shared experts.
            {
                const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
                b.cb(ffn_shexp, "ffn_shexp", il);

                cur = c.ggml_add(ctx, moe_out, ffn_shexp);
                b.cb(cur, "ffn_out", il);
            }
        }
        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    // The post-norm hidden state feeds the NextN/MTP draft head.
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

    c.ggml_build_forward_expand(b.gf, cur);
}

/// The lightning indexer block of `graph`'s constructor, which GLM-DSA's
/// and dots.llm3's build too: scores every cached indexer key for each
/// query, stores this ubatch's keys, and picks the top-k positions
/// attention may see.
///
/// Parameters:
/// - `rope_type`: the RoPE mode of the indexer's q and k.
///
/// Return: the top-k indices, `[n_top_k, n_tokens]`.
pub fn lightningIndexer(
    b: *const graph.Builder,
    rope_type: c_int,
    layer: *const model_mod.Layer,
    inp_attn_dsa: *graph.InpAttnKDsa,
    cur: *Tensor,
    qr: *Tensor,
    inp_pos: *Tensor,
    n_indexer_head: i64,
    n_embd_indexer_head: i64,
    n_indexer_top_k: u32,
    il: c_int,
) *Tensor {
    const ctx = b.ctx0;

    var indexer_q: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_attn_q_b, qr);
    b.cb(indexer_q, "indexer_q", il);

    // `{n_embd_indexer_head, n_indexer_head, n_tokens}`.
    indexer_q = c.ggml_reshape_3d(ctx, indexer_q, n_embd_indexer_head, n_indexer_head, b.n_tokens);
    indexer_q = c.ggml_rope_ext(ctx, indexer_q, inp_pos, null, @intCast(b.n_rot), rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    b.cb(indexer_q, "indexer_q", il);

    var indexer_k: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_attn_k, cur);
    b.cb(indexer_k, "indexer_k", il);

    indexer_k = b.buildNorm(indexer_k, layer.indexer_k_norm, layer.indexer_k_norm_b, .norm, il);
    b.cb(indexer_k, "indexer_k", il);

    // `{n_embd_indexer_head, 1, n_tokens}`.
    indexer_k = c.ggml_reshape_3d(ctx, indexer_k, n_embd_indexer_head, 1, b.n_tokens);
    indexer_k = c.ggml_rope_ext(ctx, indexer_k, inp_pos, null, @intCast(b.n_rot), rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    b.cb(indexer_k, "indexer_k", il);

    // The Hadamard transform of the indexer's q and k.
    indexer_q = c.ggml_mul_mat(ctx, inp_attn_dsa.self_k_rot_lid, indexer_q);
    b.cb(indexer_q, "indexer_q", il);
    indexer_k = c.ggml_mul_mat(ctx, inp_attn_dsa.self_k_rot_lid, indexer_k);
    b.cb(indexer_k, "indexer_k", il);

    // Store the indexer keys to their cache.
    const mctx_lid = inp_attn_dsa.mctx.getLid();
    const k_idxs_lid = inp_attn_dsa.getKIdxsLid();
    c.ggml_build_forward_expand(b.gf, mctx_lid.cpyK(ctx, indexer_k, k_idxs_lid, il));

    // The indexer weights.
    var indexer_weights: *Tensor = c.ggml_mul_mat(ctx, layer.indexer_proj, cur);
    b.cb(indexer_weights, "indexer_weights", il);

    // The cached indexer keys.
    indexer_k = mctx_lid.getK(ctx, il);

    // Split the batch into streams if needed.
    const n_stream = indexer_k.ne[3];
    const n_stream_u: usize = @intCast(n_stream);
    indexer_q = c.ggml_view_4d(ctx, indexer_q, indexer_q.ne[0], indexer_q.ne[1], @divTrunc(indexer_q.ne[2], n_stream), n_stream, indexer_q.nb[1], indexer_q.nb[2], indexer_q.nb[3] / n_stream_u, 0);
    indexer_weights = c.ggml_view_4d(ctx, indexer_weights, indexer_weights.ne[0], @divTrunc(indexer_weights.ne[1], n_stream), indexer_weights.ne[2], n_stream, indexer_weights.nb[1], indexer_weights.nb[2] / n_stream_u, indexer_weights.nb[3] / n_stream_u, 0);

    // Pre-scale the weights, to spare scaling the huge `indexer_score`.
    indexer_weights = c.ggml_scale(ctx, indexer_weights, 1.0 / impl.sqrtf(@floatFromInt(n_embd_indexer_head * n_indexer_head)));
    b.cb(indexer_weights, "indexer_weights", il);

    var indexer_score: *Tensor = undefined;
    if (b.cparams.fused_lid) {
        indexer_score = c.ggml_lightning_indexer(ctx, indexer_q, indexer_k, indexer_weights, inp_attn_dsa.getKqMaskLid());
        b.cb(indexer_score, "indexer_score", il);
        b.res.addFusedNode(.{ .op = .lightning_indexer, .tensor = indexer_score, .il = il });
    } else {
        // The indexer's kq.
        indexer_q = c.ggml_permute(ctx, indexer_q, 0, 2, 1, 3);
        b.cb(indexer_q, "indexer_q", il);
        indexer_k = c.ggml_permute(ctx, indexer_k, 0, 2, 1, 3);
        b.cb(indexer_k, "indexer_k", il);

        var indexer_kq: *Tensor = c.ggml_mul_mat(ctx, indexer_k, indexer_q);
        b.cb(indexer_kq, "indexer_kq", il);

        // ReLU requires contiguous tensors.
        indexer_kq = c.ggml_cont(ctx, c.ggml_permute(ctx, indexer_kq, 2, 1, 0, 3));
        b.cb(indexer_kq, "indexer_kq", il);

        // ReLU.
        indexer_score = c.ggml_relu(ctx, indexer_kq);
        b.cb(indexer_score, "indexer_score", il);

        // Scores times the indexer weights.
        indexer_score = c.ggml_mul(ctx, indexer_score, indexer_weights);
        b.cb(indexer_score, "indexer_score", il);

        // Summed over the q heads.
        indexer_score = c.ggml_sum_rows(ctx, indexer_score);
        b.cb(indexer_score, "indexer_score", il);

        // Permuted to match the KQ mask.
        indexer_score = c.ggml_cont(ctx, c.ggml_permute(ctx, indexer_score, 2, 1, 0, 3));
        b.cb(indexer_score, "indexer_score", il);

        // The indexer scores, masked.
        const indexer_kq_mask = inp_attn_dsa.getKqMaskLid();
        indexer_score = c.ggml_add(ctx, indexer_score, indexer_kq_mask);
        b.cb(indexer_score, "indexer_score", il);
    }

    // The indices of the top k indexer scores.
    const n_top_k: u32 = if (indexer_score.ne[0] < n_indexer_top_k) @intCast(indexer_score.ne[0]) else n_indexer_top_k;
    return c.ggml_cont(ctx, c.ggml_top_k(ctx, indexer_score, @intCast(n_top_k)));
}

/// The routed experts of a MoE block, as both graphs build them.
pub fn buildMoe(b: *const graph.Builder, layer: *const model_mod.Layer, cur: *Tensor, il: c_int) *Tensor {
    const hp = b.hparams;
    return b.buildMoeFfn(cur, .{
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
        .up_exps_s = layer.ffn_up_exps_s,
        .gate_exps_s = layer.ffn_gate_exps_s,
        .down_exps_s = layer.ffn_down_exps_s,
    }, il);
}

/// Ports the constructor of `llama_model_deepseek32`'s `graph_mtp`
/// (models/deepseek32.cpp:499 @c1d0e7a00): the `LLM_GRAPH_TYPE_DECODER_MTP`
/// draft head, the DeepSeek family's NextN/MTP layer -- `enorm(embed) +
/// hnorm(prev_hidden)`, concatenated, through `eh_proj`, then a whole
/// decoder block of dense MLA and the sigmoid-gated MoE with its shared
/// expert, exactly as the trunk builds them, then `shared_head_norm`
/// (falling back to `output_norm`) and the shared LM head. The DSA indexer
/// does not run. With `v`, GLM-DSA's, which differs only in its
/// assertions.
pub fn buildGraphMtp(comptime v: Variant, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ln = v.line_mtp;
    impl.assert(hp.n_layer_nextn > 0, v.file, ln, "hparams.n_layer_nextn > 0 && \"" ++ v.name ++ " MTP requires n_layer_nextn > 0\"");
    impl.assert(hp.n_layer_nextn == 1, v.file, ln + 1, "hparams.n_layer_nextn == 1 && \"" ++ v.name ++ " MTP currently only supports a single MTP block\"");
    impl.assert(hp.isMla(), v.file, ln + 2, "hparams.is_mla() && \"" ++ v.name ++ " MTP requires MLA\"");

    const il: c_int = @as(c_int, @intCast(hp.nLayer())) + b.cparams.nextn_layer_offset;
    impl.assert(b.cparams.nextn_layer_offset >= 0 and b.cparams.nextn_layer_offset < @as(c_int, @intCast(hp.n_layer_nextn)), v.file, ln + 5, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range [0, n_layer_nextn)\"");
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, v.file, ln + 10, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, v.file, ln + 11, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, v.file, ln + 12, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");
    impl.assert(layer.ffn_gate_inp != null, v.file, ln + 13, "layer.ffn_gate_inp  && \"MTP block missing ffn_gate_inp\"");

    // The head sizes as MHA, or after "decompression" through `wv_b` with
    // MLA.
    const n_embd_head_k: i64 = hp.nEmbdHeadKMla();

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const kv_lora_rank: u32 = hp.n_lora_kv;

    // The YaRN pre-scaling of the trunk graph, which this must match
    // exactly.
    impl.assert(b.ext_factor >= 0.0, v.file, ln + 25, "ext_factor >= 0.0f");
    const kq_scale = mlaKqScale(b, n_embd_head_k);

    // The C++'s TODO: extract into a common `build_inp_embd_h`.
    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort(v.file, v.line_mtp_inp, "out of memory");
    inp.* = .{ .n_embd = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    c.ggml_set_input(inp.embd);

    var tok_embd: *Tensor = undefined;
    if (b.ubatch.token != null) {
        const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;
        tok_embd = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    } else {
        tok_embd = inp.embd.?;
    }
    b.cb(tok_embd, "mtp_tok_embd", il);

    inp.h = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.n_embd, b.n_tokens);
    c.ggml_set_input(inp.h);
    _ = c.ggml_set_name(inp.h, "mtp_h_input");

    const h_embd = inp.h.?;

    _ = b.res.addInput(graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    // MLA with the absorption optimization uses a K-only cache, V a view
    // of K.
    const inp_attn = b.buildAttnInpK();

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

    // Self-attention: dense MLA, built as the trunk builds it.
    {
        var q: *Tensor = c.ggml_mul_mat(ctx, layer.wq_a, cur);
        b.cb(q, "mtp_q", il);

        q = b.buildNorm(q, layer.attn_q_a_norm, null, .rms, il);
        b.cb(q, "mtp_q", il);

        q = c.ggml_mul_mat(ctx, layer.wq_b, q);
        b.cb(q, "mtp_q", il);

        // Split into `{n_embd_head_qk_nope, n_head, n_tokens}`.
        var q_nope: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_nope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * @as(usize, @intCast(b.n_head)), 0);
        b.cb(q_nope, "mtp_q_nope", il);

        // And `{n_embd_head_qk_rope, n_head, n_tokens}`.
        var q_pe: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_rope, b.n_head, b.n_tokens, c.ggml_row_size(q.type, n_embd_head_k), c.ggml_row_size(q.type, n_embd_head_k) * @as(usize, @intCast(b.n_head)), c.ggml_row_size(q.type, n_embd_head_qk_nope));
        b.cb(q_pe, "mtp_q_pe", il);

        const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
        b.cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

        // Split into `{kv_lora_rank, n_tokens}`.
        var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), 0);
        b.cb(kv_cmpr, "mtp_kv_cmpr", il);

        // And `{n_embd_head_qk_rope, 1, n_tokens}`.
        var k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, n_embd_head_qk_rope, 1, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));
        b.cb(k_pe, "mtp_k_pe", il);

        q_pe = c.ggml_rope_ext(ctx, q_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        b.cb(q_pe, "mtp_q_pe", il);

        k_pe = c.ggml_rope_ext(ctx, k_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        b.cb(k_pe, "mtp_k_pe", il);

        kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
        b.cb(kv_cmpr, "mtp_kv_cmpr", il);

        // `{n_embd_head_qk_nope, n_tokens, n_head}`.
        q_nope = c.ggml_permute(ctx, q_nope, 0, 2, 1, 3);
        b.cb(q_nope, "mtp_q_nope_perm", il);

        // `{n_embd_head_qk_nope, kv_lora_rank, n_head}` x
        // `{n_embd_head_qk_nope, n_tokens, n_head}`.
        var q_nope_absorbed: *Tensor = c.ggml_mul_mat(ctx, layer.wk_b, q_nope);
        b.cb(q_nope_absorbed, "mtp_q_nope_absorbed", il);

        // `{kv_lora_rank, n_head, n_tokens}`.
        q_nope_absorbed = c.ggml_permute(ctx, q_nope_absorbed, 0, 2, 1, 3);
        b.cb(q_nope_absorbed, "mtp_q_nope_absorbed_perm", il);

        // `{n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}`; RoPE
        // goes first, for the in-place context shift of `build_rope_shift`.
        const Qcur: *Tensor = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
        b.cb(Qcur, "mtp_Qcur", il);

        kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, b.n_tokens);
        b.cb(kv_cmpr, "mtp_kv_cmpr_reshape", il);

        // `{n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}`.
        const Kcur: *Tensor = c.ggml_concat(ctx, kv_cmpr, k_pe, 0);
        b.cb(Kcur, "mtp_Kcur", il);

        // `{kv_lora_rank, 1, n_tokens}`.
        const Vcur = kv_cmpr;
        b.cb(Vcur, "mtp_Vcur", il);

        // MLA with the absorption optimization turns into MQA (GQA with
        // one group).
        cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, layer.wv_b, kq_scale, il);
        b.cb(cur, "mtp_attn_out", il);
    }

    const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "mtp_ffn_inp", il);

    cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "mtp_ffn_norm", il);

    // The MoE with its shared expert, built as the trunk builds it.
    const moe_out = buildMoe(b, layer, cur, il);
    b.cb(moe_out, "mtp_ffn_moe_out", il);

    // The shared expert.
    const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, layer.ffn_gate_shexp, null, layer.ffn_gate_shexp_s, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .silu, .par, il);
    b.cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = c.ggml_add(ctx, moe_out, ffn_shexp);
    b.cb(cur, "mtp_ffn_out", il);

    cur = c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "mtp_post_ffn", il);

    // `shared_head_norm` after the decoder block, before the shared LM
    // head; the post-norm hidden state seeds the next MTP step.
    const head_norm_w = layer.nextn.shared_head_norm orelse model.output_norm;
    impl.assert(head_norm_w != null, v.file, v.line_mtp_head_norm, "head_norm_w && \"" ++ v.name ++ " MTP: missing both nextn.shared_head_norm and output_norm\"");
    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
    impl.assert(head_w != null, v.file, v.line_mtp_head, "head_w && \"" ++ v.name ++ " MTP: missing LM head (nextn.shared_head_head or model.output)\"");
    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
