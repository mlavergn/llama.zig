//! MiniCPM3: DeepSeek-V2's multi-head latent attention, decompressed
//! through `wkv_b` into ordinary attention over a KV cache, with a SwiGLU
//! feed-forward, LongRoPE factors, and MiniCPM's scaling of the input
//! embeddings, the residual branches and the LM head.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/minicpm3.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_minicpm3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

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

/// Ports `llama_model_minicpm3`'s `load_arch_hparams` (models/minicpm3.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_lora_q = (try meta.getKey(u32, .attention_q_lora_rank, true, exc)).?;
    hp.n_lora_kv = (try meta.getKey(u32, .attention_kv_lora_rank, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        62 => .@"4b",
        else => .unknown,
    };
}

/// Ports `llama_model_minicpm3`'s `load_arch_tensors` (models/minicpm3.cpp:14 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    const n_embd_head_qk_rope: i64 = hp.nRot(0);
    // Unsigned, as the C++'s subtraction is, before the widening.
    const n_embd_head_qk_nope: i64 = hp.nEmbdHeadK(0) -% hp.nRot(0);

    const q_lora_rank: i64 = hp.n_lora_q;
    const kv_lora_rank: i64 = hp.n_lora_kv;
    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_q_a_norm = try self.createTensor(.{ .tensor = .attn_q_a_norm, .suffix = "weight", .bid = i }, &.{q_lora_rank}, 0, exc);

        layer.attn_kv_a_norm = try self.createTensor(.{ .tensor = .attn_kv_a_norm, .suffix = "weight", .bid = i }, &.{kv_lora_rank}, 0, exc);

        layer.wq_a = try self.createTensor(.{ .tensor = .attn_q_a, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_lora_rank }, 0, exc);
        layer.wq_b = try self.createTensor(.{ .tensor = .attn_q_b, .suffix = "weight", .bid = i }, &.{ q_lora_rank, l.n_head * l.n_embd_head_k }, 0, exc);

        layer.wkv_a_mqa = try self.createTensor(.{ .tensor = .attn_kv_a_mqa, .suffix = "weight", .bid = i }, &.{ l.n_embd, kv_lora_rank + n_embd_head_qk_rope }, 0, exc);
        layer.wkv_b = try self.createTensor(.{ .tensor = .attn_kv_b, .suffix = "weight", .bid = i }, &.{ kv_lora_rank, l.n_head * (n_embd_head_qk_nope + l.n_embd_head_v) }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_head * l.n_embd_head_v, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

        const rope_dup: u32 = if (i != 0) dup else 0;
        layer.rope_long = try self.createTensor(.{ .tensor = .rope_factors_long, .suffix = "weight", .bid = i }, &.{@divTrunc(n_embd_head_qk_rope, 2)}, nr | rope_dup, exc);
        layer.rope_short = try self.createTensor(.{ .tensor = .rope_factors_short, .suffix = "weight", .bid = i }, &.{@divTrunc(n_embd_head_qk_rope, 2)}, nr | rope_dup, exc);
    }
}

/// Ports `llama_model_minicpm3`'s `build_arch_graph` (models/minicpm3.cpp:59 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_minicpm3`'s `graph`
/// (models/minicpm3.cpp:63 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    // The C++'s TODO: if the model varies, read these from it.
    const n_embd_base: i64 = 256;
    const scale_embd: f32 = 12.0;
    const scale_depth: f32 = 1.4;
    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(hp.nEmbdHeadK(0)));

    const n_embd_head_qk_rope: u32 = hp.nRot(0);
    const n_embd_head_qk_nope: u32 = hp.nEmbdHeadK(0) -% hp.nRot(0);

    const kv_lora_rank: u32 = hp.n_lora_kv;

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // Scale the input embeddings.
    inpL = c.ggml_scale(ctx, inpL, scale_embd);
    b.cb(inpL, "inp_scaled", -1);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            // `{n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}`.
            var q: *Tensor = c.ggml_mul_mat(ctx, layer.wq_a, cur);
            b.cb(q, "q", il);

            q = b.buildNorm(q, layer.attn_q_a_norm, null, .rms, il);
            b.cb(q, "q", il);

            // `{q_lora_rank, n_head * n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * n_embd_head_k, n_tokens}`.
            q = c.ggml_mul_mat(ctx, layer.wq_b, q);
            b.cb(q, "q", il);

            // `{n_embd_head_k, n_head, n_tokens}`; RoPE is applied to the
            // trailing dimensions only.
            q = c.ggml_reshape_3d(ctx, q, hp.nEmbdHeadK(0), b.n_head, b.n_tokens);
            b.cb(q, "q", il);

            // `{n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}`.
            const kv_pe_compresseed: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
            b.cb(kv_pe_compresseed, "kv_pe_compresseed", il);

            // Split into `{kv_lora_rank, n_tokens}`.
            var kv_compressed: *Tensor = c.ggml_view_2d(ctx, kv_pe_compresseed, kv_lora_rank, b.n_tokens, kv_pe_compresseed.nb[1], 0);
            b.cb(kv_compressed, "kv_compressed", il);

            // And `{n_embd_head_qk_rope, n_tokens}`.
            var k_pe: *Tensor = c.ggml_view_3d(ctx, kv_pe_compresseed, n_embd_head_qk_rope, 1, b.n_tokens, kv_pe_compresseed.nb[1], kv_pe_compresseed.nb[1], c.ggml_row_size(kv_pe_compresseed.type, kv_lora_rank));
            b.cb(k_pe, "k_pe", il);

            kv_compressed = b.buildNorm(kv_compressed, layer.attn_kv_a_norm, null, .rms, il);
            b.cb(kv_compressed, "kv_compressed", il);

            // `{kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}`.
            const kv: *Tensor = c.ggml_mul_mat(ctx, layer.wkv_b, kv_compressed);
            b.cb(kv, "kv", il);

            // Split into `{n_head * n_embd_head_qk_nope, n_tokens}`.
            const k_nope: *Tensor = c.ggml_view_3d(ctx, kv, n_embd_head_qk_nope, b.n_head, b.n_tokens, c.ggml_row_size(kv.type, n_embd_head_qk_nope + hp.nEmbdHeadV(0)), c.ggml_row_size(kv.type, b.n_head * (n_embd_head_qk_nope + hp.nEmbdHeadV(0))), 0);
            b.cb(k_nope, "k_nope", il);

            // And `{n_head * n_embd_head_v, n_tokens}`.
            var v_states: *Tensor = c.ggml_view_3d(ctx, kv, hp.nEmbdHeadV(0), b.n_head, b.n_tokens, c.ggml_row_size(kv.type, (n_embd_head_qk_nope + hp.nEmbdHeadV(0))), c.ggml_row_size(kv.type, (n_embd_head_qk_nope + hp.nEmbdHeadV(0)) * b.n_head), c.ggml_row_size(kv.type, (n_embd_head_qk_nope)));
            b.cb(v_states, "v_states", il);

            v_states = c.ggml_cont(ctx, v_states);
            b.cb(v_states, "v_states", il);

            q = c.ggml_rope_ext(ctx, q, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            q = c.ggml_rope_set_offset(q, @bitCast(n_embd_head_qk_nope));
            b.cb(q, "q_rope", il);

            // The shared RoPE key.
            k_pe = c.ggml_rope_ext(ctx, k_pe, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            b.cb(k_pe, "k_pe", il);

            const q_states = q;
            b.cb(q_states, "q_states", il);

            const k_states: *Tensor = c.ggml_concat(ctx, k_nope, c.ggml_repeat_4d(ctx, k_pe, n_embd_head_qk_rope, b.n_head, b.n_tokens, 1), 0);
            b.cb(k_states, "k_states", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, q_states, k_states, v_states, null, null, null, kq_scale, il);
        }
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        // Scale the hidden states for the residual connection (the C++
        // asks whether this is correct).
        const scale_res: f32 = scale_depth / impl.sqrtf(@floatFromInt(b.n_layer));
        cur = c.ggml_scale(ctx, cur, scale_res);
        b.cb(cur, "hidden_scaled", il);

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        {
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        }
        // Scale the hidden states for the residual connection.
        cur = c.ggml_scale(ctx, cur, scale_res);
        b.cb(cur, "hidden_scaled_ffn", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head's scaling.
    const scale_lmhead: f32 = @as(f32, @floatFromInt(n_embd_base)) / @as(f32, @floatFromInt(b.n_embd));
    cur = c.ggml_scale(ctx, cur, scale_lmhead);
    b.cb(cur, "lmhead_scaling", -1);

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
