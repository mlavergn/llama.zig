//! DeepSeek-V2 and V3 (and GLM 4.7 Lite, GigaChat 3, Kanana 2, Mistral
//! Large 3): pre-norm blocks with multi-head latent attention -- Q through
//! an optional low-rank pair, K and V compressed into one latent with a
//! shared RoPE key -- leading dense blocks, then a mixture of experts with
//! shared experts; and an optional NextN/MTP block past the main stack.
//!
//! With `key_length_mla` and `value_length_mla` set, the latent is cached
//! as it is and Q absorbs `wk_b`, so attention is MQA over a K-only cache
//! and `wv_b` decompresses its output. Without them, `wkv_b` decompresses
//! K and V before an ordinary MHA over a KV cache.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/deepseek2.cpp` at v0.4.1
//! (`b29c606e2`): `llama_model_deepseek2`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph` and
//! `graph_mtp`. `graph` also carries DeepSeek-OCR's attention, which
//! `llama_model_deepseek2ocr` (`deepseek2ocr.zig`) and
//! `llama_model_mistral4` (`mistral4.zig`) build with.

const std = @import("std");
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

/// Ports `llama_model_deepseek2`'s `load_arch_hparams` (models/deepseek2.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    var n_vocab: u32 = 0;
    if (try meta.getKey(u32, .vocab_size, false, exc)) |v| {
        n_vocab = v;
    } else if (try meta.getArrN(.tokenizer_list, false, exc)) |v| {
        n_vocab = v;
    }

    // The lite variants include DeepSeek-V2-Lite, GigaChat3-10B-A1.8B and
    // Kanana-2-30B-A3B.
    const is_lite = (hp.nLayer() == 27 or hp.nLayer() == 26 or (hp.nLayer() == 48 and n_vocab == 128256));

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    if (!is_lite) {
        hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
    }
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;
    if (try meta.getKey(u32, .attention_key_length_mla, false, exc)) |v| hp.n_embd_head_k_mla_impl = v;
    if (try meta.getKey(u32, .attention_value_length_mla, false, exc)) |v| hp.n_embd_head_v_mla_impl = v;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;
    if (hp.expert_gating_func == @backingInt(hparams_mod.ExpertGatingFunc.none)) {
        // For the DeepSeek V2 and V2.5 GGUFs that have no
        // `expert_gating_func`.
        if ((hp.nLayer() == 47 or hp.nLayer() == 48) and n_vocab == 154880) {
            // GLM 4.7 Lite.
            hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.sigmoid);
        } else {
            hp.expert_gating_func = @backingInt(hparams_mod.ExpertGatingFunc.softmax);
        }
    }

    if (try meta.getKey(f32, .rope_scaling_yarn_log_mul, false, exc)) |v| {
        hp.rope_yarn_log_mul = v;
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] Cancel the factor the convert
        // script applied.
        hp.rope_yarn_log_mul /= 0.1;
    }

    // The optional temperature tuning, used by Mistral Large.
    if (try meta.getKey(f32, .attention_temperature_scale, false, exc)) |v| hp.f_attn_temp_scale = v;
    // The C++'s FIXME: why not `temperature_length`?
    if (try meta.getKey(u32, .attention_temperature_length, false, exc)) |v| hp.n_attn_temp_floor_scale = v;

    hp.f_attn_temp_offset = 0.0;

    self.type = switch (hp.nLayer()) {
        27 => .@"16b",
        47 => .@"30b_a3b",
        60 => .@"236b",
        61 => .@"671b",
        else => .unknown,
    };
}

/// Ports `llama_model_deepseek2`'s `load_arch_tensors` (models/deepseek2.cpp:55 @b29c606e2).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const n_expert_shared: i64 = hp.n_expert_shared;
    const nr = loader.tensor_not_required;

    const mtp_only = hp.n_layer_nextn > 0 and ml.getWeight("blk.0.attn_norm.weight") == null;
    var probe_buf: [64]u8 = undefined;
    const mtp_probe = std.fmt.bufPrint(&probe_buf, "blk.{d}.nextn.eh_proj.weight", .{l.n_layer}) catch unreachable;
    const trunk_only = hp.n_layer_nextn > 0 and ml.getWeight(mtp_probe) == null;
    const trunk_flags: u32 = if (mtp_only) nr else 0;
    var mtp_flags: u32 = if (trunk_only) nr else 0;

    if (!ml.load_mtp) mtp_flags |= loader.tensor_skip;

    const is_mla = hp.isMla();

    // The head sizes as MHA, or after "decompression" through `wv_b` with
    // MLA.
    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_v_mla: i64 = hp.nEmbdHeadVMla();

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    impl.assert(n_embd_head_qk_nope >= 1, "models/deepseek2.cpp", 77, "n_embd_head_qk_nope >= 1");

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
        if (q_lora_rank > 0) {
            layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, flags, exc);
        }

        layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, flags, exc);

        if (q_lora_rank > 0) {
            layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_lora_rank }, flags, exc);
            layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, l.n_head * n_embd_head_k_mla }, flags, exc);
        } else {
            layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_head * n_embd_head_k_mla }, flags, exc);
        }

        layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ l.n_embd, kv_lora_rank + n_embd_head_qk_rope }, flags, exc);

        // Only old legacy GGUF files have the unsplit `wkv_b`.
        if (is_mla) {
            layer.wk_b = try self.createTensor(.{ .tensor = .attn_k_b, .suffix = "weight", .bid = i }, &.{ n_embd_head_qk_nope, kv_lora_rank, l.n_head }, flags, exc);
            layer.wv_b = try self.createTensor(.{ .tensor = .attn_v_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, n_embd_head_v_mla, l.n_head }, flags, exc);
        } else {
            layer.wkv_b = try self.createTensor(.{ .tensor = .attn_kv_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, l.n_head * (n_embd_head_qk_nope + n_embd_head_v_mla) }, flags, exc);
        }

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_head * n_embd_head_v_mla, l.n_embd }, flags, exc);

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
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, mtp_flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, mtp_flags, exc);
            layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr | flags, exc);
            layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr | flags, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr | flags, exc);
        }
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_deepseek2`'s `build_arch_graph` (models/deepseek2.cpp:163
/// @b29c606e2): the MTP draft head for that graph type, the main model
/// otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return buildGraph(model, b);
}

/// The YaRN pre-scaling both graphs share: the KQ scale, from the
/// `attn_factor` `llama_hparams`'s `yarn_attn_factor_adjust` adjusted.
fn mlaKqScale(b: *const graph.Builder, n_embd_head_k: i64) f32 {
    const hp = b.hparams;
    const attn_factor_org = b.attn_factor * (1.0 + 0.1 * impl.logf(1.0 / b.freq_scale));
    const mscale = attn_factor_org * (1.0 + 0.1 * hp.rope_yarn_log_mul * impl.logf(1.0 / b.freq_scale));
    return 1.0 * mscale * mscale / impl.sqrtf(@floatFromInt(n_embd_head_k));
}

/// Ports the constructor of `llama_model_deepseek2`'s `graph_mtp`
/// (models/deepseek2.cpp:170 @b29c606e2).
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, "models/deepseek2.cpp", 172, "hparams.n_layer_nextn > 0 && \"GLM4 MTP requires n_layer_nextn > 0\"");
    impl.assert(hp.n_layer_nextn == 1, "models/deepseek2.cpp", 173, "hparams.n_layer_nextn == 1 && \"GLM4 MTP currently only supports a single MTP block\"");
    impl.assert(hp.isMla(), "models/deepseek2.cpp", 174, "hparams.is_mla() && \"GLM4 MTP requires MLA\"");
    impl.assert(hp.f_attn_temp_scale == 0.0, "models/deepseek2.cpp", 175, "hparams.f_attn_temp_scale == 0.0f && \"GLM4 MTP does not support attention temperature scaling\"");

    // The appended MTP block is stored straight after the decoder layers.
    const il: c_int = @intCast(hp.nLayer());
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null, "models/deepseek2.cpp", 181, "layer.nextn.eh_proj && \"MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, "models/deepseek2.cpp", 182, "layer.nextn.enorm   && \"MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, "models/deepseek2.cpp", 183, "layer.nextn.hnorm   && \"MTP block missing nextn.hnorm\"");

    impl.assert(@as(u32, @intCast(il)) >= hp.n_layer_dense_lead, "models/deepseek2.cpp", 185, "(uint32_t) il >= hparams.n_layer_dense_lead && \"GLM4 MTP block expected to use MoE FFN\"");

    const n_embd_head_k_mla: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const kv_lora_rank: i64 = hp.n_lora_kv;

    impl.assert(n_embd_head_qk_nope >= 1, "models/deepseek2.cpp", 192, "n_embd_head_qk_nope >= 1");
    impl.assert(hp.n_lora_q > 0, "models/deepseek2.cpp", 193, "hparams.n_lora_q > 0");
    impl.assert(layer.wq_a != null, "models/deepseek2.cpp", 194, "layer.wq_a");
    impl.assert(layer.attn_q_a_norm != null, "models/deepseek2.cpp", 195, "layer.attn_q_a_norm");
    impl.assert(layer.wq_b != null, "models/deepseek2.cpp", 196, "layer.wq_b");
    impl.assert(layer.wkv_a_mqa != null, "models/deepseek2.cpp", 197, "layer.wkv_a_mqa");
    impl.assert(layer.attn_kv_a_norm != null, "models/deepseek2.cpp", 198, "layer.attn_kv_a_norm");
    impl.assert(layer.wk_b != null, "models/deepseek2.cpp", 199, "layer.wk_b");

    const has_split_exps = layer.ffn_up_exps != null and layer.ffn_gate_exps != null;

    const has_fused_exps = layer.ffn_gate_up_exps != null;

    impl.assert(has_split_exps or has_fused_exps, "models/deepseek2.cpp", 207, "has_split_exps || has_fused_exps");
    impl.assert(layer.ffn_norm != null, "models/deepseek2.cpp", 208, "layer.ffn_norm");
    impl.assert(layer.ffn_gate_inp != null, "models/deepseek2.cpp", 209, "layer.ffn_gate_inp");
    impl.assert(layer.ffn_down_exps != null, "models/deepseek2.cpp", 210, "layer.ffn_down_exps");
    impl.assert(layer.ffn_gate_shexp != null, "models/deepseek2.cpp", 211, "layer.ffn_gate_shexp");
    impl.assert(layer.ffn_down_shexp != null, "models/deepseek2.cpp", 212, "layer.ffn_down_shexp");
    impl.assert(layer.ffn_up_shexp != null, "models/deepseek2.cpp", 213, "layer.ffn_up_shexp");

    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort("models/deepseek2.cpp", 215, "out of memory");
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

    impl.assert(b.ext_factor >= 0.0, "models/deepseek2.cpp", 305, "ext_factor >= 0.0f");

    const kq_scale = mlaKqScale(b, n_embd_head_k_mla);

    q_pe = c.ggml_rope_ext(ctx, q_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    b.cb(q_pe, "mtp_q_pe_rope", il);

    k_pe = c.ggml_rope_ext(ctx, k_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
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
    impl.assert(head_norm_w != null, "models/deepseek2.cpp", 389, "head_norm_w && \"GLM4 MTP: missing both nextn.shared_head_norm and output_norm\"");

    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;

    impl.assert(head_w != null, "models/deepseek2.cpp", 408, "head_w && \"GLM4 MTP: missing LM head (nextn.shared_head_head or model.output)\"");

    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_deepseek2`'s `graph`
/// (models/deepseek2.cpp:417 @b29c606e2).
pub fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    // The lite variants include DeepSeek-V2-Lite and GigaChat3-10B-A1.8B.
    const is_ocr = model.arch == .deepseek2ocr;

    const is_mla = hp.isMla();

    // The head sizes as MHA, or after "decompression" through `wv_b` with
    // MLA.
    const n_embd_head_k: i64 = hp.nEmbdHeadKMla();
    const n_embd_head_v: i64 = hp.nEmbdHeadVMla();

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    const n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const kv_lora_rank: u32 = hp.n_lora_kv;

    // `kq_scale` and `attn_factor` are pre-scaled for YaRN RoPE to work
    // (https://github.com/ggml-org/llama.cpp/discussions/7416, and
    // https://github.com/ggml-org/llama.cpp/pull/17945
    // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]): first the adjustment from
    // `llama_hparams`'s `yarn_attn_factor_adjust` is cancelled to recover
    // the original `attn_factor`, which then pre-scales `kq_scale`.
    impl.assert(b.ext_factor >= 0.0, "models/deepseek2.cpp", 438, "ext_factor >= 0.0f");
    const kq_scale = mlaKqScale(b, n_embd_head_k);

    // `{n_embd, n_tokens}`.
    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The optional temperature tuning, used by Mistral Large.
    var inp_attn_scale: ?*Tensor = null;
    if (hp.f_attn_temp_scale != 0.0) {
        inp_attn_scale = b.buildInpAttnScale();
    }

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn_kv = if (!is_mla) b.buildAttnInpKv() else null;
    const inp_attn_k = if (is_mla) b.buildAttnInpK() else null;

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        if (is_ocr) {
            const n_embed_head: c_int = @intCast(@divTrunc(hp.n_embd, hp.nHead(0)));
            const ocr_rope_type: c_int = c.GGML_ROPE_TYPE_NEOX;
            impl.assert(n_embed_head == n_embd_head_k and n_embed_head == n_embd_head_v, "models/deepseek2.cpp", 476, "n_embed_head == n_embd_head_k && n_embed_head == n_embd_head_v");

            const qkv = b.buildQkv(layer, cur, n_embed_head, b.n_head, b.n_head, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;
            b.cb(Qcur, "q", il);
            b.cb(Kcur, "k", il);
            b.cb(Vcur, "v", il);

            impl.assert(@abs(@as(f64, b.freq_base) - 10000.0) < 1e-4, "models/deepseek2.cpp", 484, "fabs(freq_base - 10000.0) < 1e-4");
            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, n_embed_head, ocr_rope_type, 0, b.freq_base, 1, 0, 1, 0, 0);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, n_embed_head, ocr_rope_type, 0, b.freq_base, 1, 0, 1, 0, 0);
            b.cb(Qcur, "q_pe", il);
            b.cb(Kcur, "k_pe", il);

            cur = b.buildAttn(inp_attn_kv.?, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);
        } else {
            var q: *Tensor = undefined;

            const is_lite = layer.wq != null;

            if (!is_lite) {
                q = c.ggml_mul_mat(ctx, layer.wq_a, cur);
                b.cb(q, "q", il);

                q = b.buildNorm(q, layer.attn_q_a_norm, null, .rms, il);
                b.cb(q, "q", il);

                q = c.ggml_mul_mat(ctx, layer.wq_b, q);
                b.cb(q, "q", il);
            } else {
                q = c.ggml_mul_mat(ctx, layer.wq, cur);
                b.cb(q, "q", il);
            }
            // `{n_embd_head_k, n_head, n_tokens}`.
            q = c.ggml_reshape_3d(ctx, q, n_embd_head_k, b.n_head, b.n_tokens);
            b.cb(q, "q", il);

            const kv_cmpr_pe: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
            b.cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // Split into `{kv_lora_rank, n_tokens}`.
            var kv_cmpr: *Tensor = c.ggml_view_2d(ctx, kv_cmpr_pe, kv_lora_rank, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), 0);
            b.cb(kv_cmpr, "kv_cmpr", il);

            // And `{n_embd_head_qk_rope, 1, n_tokens}`.
            var k_pe: *Tensor = c.ggml_view_3d(ctx, kv_cmpr_pe, n_embd_head_qk_rope, 1, b.n_tokens, c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank + n_embd_head_qk_rope), c.ggml_row_size(kv_cmpr_pe.type, kv_lora_rank));
            b.cb(k_pe, "k_pe", il);

            k_pe = c.ggml_rope_ext(ctx, k_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            b.cb(k_pe, "k_pe", il);

            kv_cmpr = b.buildNorm(kv_cmpr, layer.attn_kv_a_norm, null, .rms, il);
            b.cb(kv_cmpr, "kv_cmpr", il);

            if (is_mla) {
                // Split into `{n_embd_head_qk_nope, n_head, n_tokens}`.
                var q_nope: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_nope, b.n_head, b.n_tokens, q.nb[1], q.nb[2], 0);
                b.cb(q_nope, "q_nope", il);

                // And `{n_embd_head_qk_rope, n_head, n_tokens}`.
                var q_pe: *Tensor = c.ggml_view_3d(ctx, q, n_embd_head_qk_rope, b.n_head, b.n_tokens, q.nb[1], q.nb[2], c.ggml_row_size(q.type, n_embd_head_qk_nope));
                b.cb(q_pe, "q_pe", il);

                q_pe = c.ggml_rope_ext(ctx, q_pe, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                b.cb(q_pe, "q_pe", il);

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
                var Qcur: *Tensor = c.ggml_concat(ctx, q_nope_absorbed, q_pe, 0);
                b.cb(Qcur, "Qcur", il);

                kv_cmpr = c.ggml_reshape_3d(ctx, kv_cmpr, kv_lora_rank, 1, b.n_tokens);
                b.cb(kv_cmpr, "kv_cmpr_reshape", il);

                // `{n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}`.
                const Kcur: *Tensor = c.ggml_concat(ctx, kv_cmpr, k_pe, 0);
                b.cb(Kcur, "Kcur", il);

                // `{kv_lora_rank, 1, n_tokens}`.
                const Vcur = kv_cmpr;
                b.cb(Vcur, "Vcur", il);

                if (inp_attn_scale) |s| {
                    // Llama 4's temperature scaling.
                    Qcur = c.ggml_mul(ctx, Qcur, s);
                    b.cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                // MLA with the absorption optimization turns into MQA (GQA
                // with one group).
                cur = b.buildAttn(inp_attn_k.?, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, layer.wv_b, kq_scale, il);
            } else {
                const kv: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_b, kv_cmpr);
                b.cb(kv, "kv", il);

                // Split into `{n_embd_head_qk_nope, n_head, n_tokens}`.
                const k_nope: *Tensor = c.ggml_view_3d(ctx, kv, n_embd_head_qk_nope, b.n_head, b.n_tokens, c.ggml_row_size(kv.type, n_embd_head_qk_nope + n_embd_head_v), c.ggml_row_size(kv.type, n_embd_head_qk_nope + n_embd_head_v) * @as(usize, @intCast(b.n_head)), 0);
                b.cb(k_nope, "k_nope_view", il);

                // And `{n_embd_head_v, n_head, n_tokens}`.
                var Vcur: *Tensor = c.ggml_view_3d(ctx, kv, n_embd_head_v, b.n_head, b.n_tokens, c.ggml_row_size(kv.type, n_embd_head_qk_nope + n_embd_head_v), c.ggml_row_size(kv.type, n_embd_head_qk_nope + n_embd_head_v) * @as(usize, @intCast(b.n_head)), c.ggml_row_size(kv.type, n_embd_head_qk_nope));
                b.cb(Vcur, "Vcur_view", il);

                Vcur = c.ggml_cont(ctx, Vcur);
                b.cb(Vcur, "Vcur_cont", il);

                // RoPE is applied to the trailing dimensions only.
                var Qcur: *Tensor = c.ggml_rope_ext(ctx, q, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                Qcur = c.ggml_rope_set_offset(Qcur, @intCast(n_embd_head_qk_nope));
                b.cb(Qcur, "Qcur", il);

                const Kcur: *Tensor = c.ggml_concat(ctx, k_nope, c.ggml_repeat_4d(ctx, k_pe, n_embd_head_qk_rope, b.n_head, b.n_tokens, 1), 0);
                b.cb(Kcur, "Kcur", il);

                if (inp_attn_scale) |s| {
                    // Llama 4's temperature scaling.
                    Qcur = c.ggml_mul(ctx, Qcur, s);
                    b.cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                // MLA without the absorption optimization turns into MHA
                // (GQA with `n_head` groups).
                cur = b.buildAttn(inp_attn_kv.?, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            }
        }
        if (il == b.n_layer - 1 and (!b.cparams.embeddings_nextn or b.cparams.embeddings_nextn_masked)) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (@as(u32, @intCast(il)) < hp.n_layer_dense_lead) {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The routed experts.
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
        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

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

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
