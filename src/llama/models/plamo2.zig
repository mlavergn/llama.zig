//! PLaMo-2: Mamba layers with normed B, C and dt interleaved with attention
//! layers that have Q and K norms, each mixer and each feed-forward wrapped
//! in pre- and post-norms. A layer is a Mamba layer when it has no KV heads.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/plamo2.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_plamo2`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph` with `build_plamo2_attn_layer` and
//! `build_plamo2_mamba_layer`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const mamba_base = @import("mamba_base.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// `std::max(64, int(hparams.n_embd / 16))`, the dt projection's width.
fn dtDim(n_embd: u32) i64 {
    return @max(64, @as(c_int, @bitCast(n_embd / 16)));
}

/// Ports `llama_model_plamo2`'s `load_arch_hparams` (models/plamo2.cpp:4 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    // The Mamba SSM parameters.
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

    // The attention parameters.
    if (try meta.getKey(u32, .attention_key_length, false, exc)) |v| hp.n_embd_head_k_full = v;
    if (try meta.getKey(u32, .attention_value_length, false, exc)) |v| hp.n_embd_head_v_full = v;

    var i: u32 = 0;
    while (i < hp.nLayer()) : (i += 1) {
        hp.is_recr_impl[i] = @intFromBool(hp.nHeadKv(i) == 0);
    }

    switch (hp.nLayer()) {
        16 => self.type = .@"1b",
        32 => {
            if (hp.n_embd == 2048) {
                self.type = .@"2b";
            } else if (hp.n_embd == 4096) {
                self.type = .@"8b";
            }
        },
        else => self.type = .unknown,
    }
}

/// Ports `llama_model_plamo2`'s `load_arch_tensors` (models/plamo2.cpp:35 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    // The Mamba parameters.
    const d_conv: u32 = hp.ssm_d_conv;
    const d_state: u32 = hp.ssm_d_state;
    const num_heads: u32 = hp.ssm_dt_rank;
    const intermediate_size: u32 = hp.ssm_d_inner;
    const dt_dim: i64 = dtDim(hp.n_embd);

    // The attention parameters.
    const qk_dim: u32 = hp.nEmbdHeadK(0);
    const v_dim: u32 = hp.nEmbdHeadV(0);

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_not_required, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const is_mamba_layer = hp.isRecr(@intCast(i));

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (is_mamba_layer) {
            layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = i }, &.{ l.n_embd, 2 * @as(i64, intermediate_size) }, 0, exc);
            layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = i }, &.{ d_conv, intermediate_size }, 0, exc);

            layer.ssm_x = try self.createTensor(.{ .tensor = .ssm_x, .suffix = "weight", .bid = i }, &.{ intermediate_size, dt_dim + 2 * @as(i64, d_state) }, 0, exc);
            layer.ssm_dt = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "weight", .bid = i }, &.{ dt_dim, num_heads }, 0, exc);
            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = i }, &.{num_heads}, 0, exc);

            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = i }, &.{num_heads}, 0, exc);
            layer.ssm_d = try self.createTensor(.{ .tensor = .ssm_d, .bid = i }, &.{num_heads}, 0, exc);

            layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = i }, &.{ intermediate_size, l.n_embd }, 0, exc);

            layer.ssm_dt_norm = try self.createTensor(.{ .tensor = .ssm_dt_norm, .bid = i }, &.{dt_dim}, 0, exc);
            layer.ssm_b_norm = try self.createTensor(.{ .tensor = .ssm_b_norm, .bid = i }, &.{d_state}, 0, exc);
            layer.ssm_c_norm = try self.createTensor(.{ .tensor = .ssm_c_norm, .bid = i }, &.{d_state}, 0, exc);
        } else {
            const num_attention_heads: i64 = hp.nHead(@intCast(i));
            const q_num_heads = num_attention_heads;
            const num_key_value_heads: i64 = hp.nHeadKv(@intCast(i));
            const k_num_heads = num_key_value_heads;
            const v_num_heads = num_key_value_heads;
            const q_proj_dim = q_num_heads * qk_dim;
            const k_proj_dim = k_num_heads * qk_dim;
            const v_proj_dim = v_num_heads * v_dim;

            layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, q_proj_dim + k_proj_dim + v_proj_dim }, 0, exc);
            layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{ qk_dim, num_attention_heads }, 0, exc);
            layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{ qk_dim, k_num_heads }, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ q_num_heads * v_dim, l.n_embd }, 0, exc);
        }

        // Every layer has a post-attention norm, an FFN norm and FFN tensors.
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff * 2 }, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .bid = i }, &.{l.n_embd}, 0, exc);
    }
}

/// Ports `llama_model_plamo2`'s `build_arch_graph` (models/plamo2.cpp:106 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_plamo2`'s `graph` (models/plamo2.cpp:110 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    // {n_embd, n_tokens}
    var inpL = b.buildInpEmbd(model.tok_embd.?);
    b.cb(inpL, "embedding_output", -1);

    const inp_pos = b.buildInpPos();

    const inp_hybrid = b.buildInpMemHybrid();

    const inp_out_ids = b.buildInpOutIds();

    var cur: *Tensor = undefined;
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var residual = inpL;

        // The pre-mixer norm.
        cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);

        // A Mamba layer or an attention layer.
        const is_mamba_layer = hp.isRecr(@intCast(il));

        if (is_mamba_layer) {
            cur = buildPlamo2MambaLayer(b, inp_hybrid.getRecr(), cur, model, il);
        } else {
            cur = buildPlamo2AttnLayer(b, inp_hybrid.getAttn(), inp_pos, cur, model, il);
        }

        // The post-mixer norm.
        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        // The residual connection.
        cur = c.ggml_add(ctx, cur, residual);
        b.cb(cur, "attn_residual", il);
        residual = cur;

        // The pre-FFN norm.
        cur = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_pre_norm", il);

        // The feed-forward network.
        cur = b.buildFfn(cur, layer.ffn_up, null, null, null, null, null, layer.ffn_down, null, null, null, .swiglu, .seq, il);
        b.cb(cur, "ffn_out", il);

        // The post-FFN norm.
        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, il);
        b.cb(cur, "ffn_post_norm", il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            residual = c.ggml_get_rows(ctx, residual, inp_out_ids);
        }

        // The residual connection.
        cur = c.ggml_add(ctx, cur, residual);
        b.cb(cur, "ffn_residual", il);

        // The next layer's input.
        inpL = cur;
    }

    cur = inpL;

    // The final norm.
    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);
    b.cb(cur, "result_output", -1);

    // Marked as an output explicitly, for the backend assignment.
    c.ggml_set_output(cur);

    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_plamo2::graph`'s `build_plamo2_attn_layer`
/// (models/plamo2.cpp:203 @c1d0e7a00).
fn buildPlamo2AttnLayer(b: *const graph.Builder, inp: *graph.InpAttnKv, inp_pos: *Tensor, cur_: *Tensor, model: *const Model, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    var cur: *Tensor = undefined;

    // Self-attention.
    {
        // PLaMo-2 uses a combined QKV tensor.
        const qkv = b.buildLoraMm(layer.wqkv.?, cur_, null);
        b.cb(qkv, "wqkv", il);

        // Split it into Q, K and V. The strides assume F32 whatever the type.
        const n_embd_head_q: i64 = hp.nEmbdHeadK(0);
        const n_embd_head_k: i64 = hp.nEmbdHeadK(0);
        const n_embd_head_v: i64 = hp.nEmbdHeadV(0);
        const n_head: i32 = @bitCast(hp.nHead(@intCast(il)));
        const n_head_kv: i32 = @bitCast(hp.nHeadKv(@intCast(il)));

        const q_offset: i64 = 0;
        const k_offset: i64 = n_embd_head_q * n_head;
        const v_offset: i64 = k_offset + n_embd_head_k * n_head_kv;

        const es = c.ggml_element_size(qkv);
        var Qcur: *Tensor = c.ggml_view_3d(ctx, qkv, n_embd_head_q, n_head, b.n_tokens, @as(usize, @intCast(n_embd_head_q)) * @sizeOf(f32), qkv.nb[1], @as(usize, @intCast(q_offset)) * es);
        var Kcur: *Tensor = c.ggml_view_3d(ctx, qkv, n_embd_head_k, n_head_kv, b.n_tokens, @as(usize, @intCast(n_embd_head_k)) * @sizeOf(f32), qkv.nb[1], @as(usize, @intCast(k_offset)) * es);
        const Vcur: *Tensor = c.ggml_view_3d(ctx, qkv, n_embd_head_v, n_head_kv, b.n_tokens, @as(usize, @intCast(n_embd_head_v)) * @sizeOf(f32), qkv.nb[1], @as(usize, @intCast(v_offset)) * es);

        b.cb(Qcur, "Qcur", il);
        b.cb(Kcur, "Kcur", il);
        b.cb(Vcur, "Vcur", il);

        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        b.cb(Qcur, "Qcur_normed", il);

        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
        b.cb(Kcur, "Kcur_normed", il);

        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

        cur = b.buildAttn(inp, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head_v)), il);
    }

    b.cb(cur, "attn_out", il);

    return cur;
}

/// Ports `llama_model_plamo2::graph`'s `build_plamo2_mamba_layer`
/// (models/plamo2.cpp:258 @c1d0e7a00).
fn buildPlamo2MambaLayer(b: *const graph.Builder, inp: *const graph.InpRs, cur_: *Tensor, model: *const Model, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ubatch = b.ubatch;
    const layer = &model.layers[@intCast(il)];
    const mctx_cur = inp.mctx;

    const kv_head: usize = mctx_cur.getHead();

    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const n_heads: i64 = hp.ssm_dt_rank;
    const head_dim: i64 = @divTrunc(d_inner, n_heads);
    const n_group: i64 = hp.ssm_n_group;
    const n_seqs: i64 = ubatch.n_seqs;

    const n_seq_tokens: i64 = ubatch.n_seq_tokens;

    impl.assert(n_seqs != 0, "models/plamo2.cpp", 277, "n_seqs != 0");
    impl.assert(ubatch.equalSeqs(), "models/plamo2.cpp", 278, "ubatch.equal_seqs()");
    impl.assert(ubatch.n_tokens == n_seq_tokens * n_seqs, "models/plamo2.cpp", 279, "ubatch.n_tokens == n_seq_tokens * n_seqs");
    impl.assert(@rem(d_inner, n_heads) == 0, "models/plamo2.cpp", 280, "d_inner % n_heads == 0");
    impl.assert(n_group == 0, "models/plamo2.cpp", 281, "n_group == 0");

    const conv_states_all = mctx_cur.getRL(il);
    const ssm_states_all = mctx_cur.getSL(il);

    var conv = b.buildRs(inp, conv_states_all, @bitCast(hp.nEmbdR()), @intCast(n_seqs), null);
    conv = c.ggml_reshape_3d(ctx, conv, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    var cur: *Tensor = c.ggml_reshape_3d(ctx, cur_, cur_.ne[0], n_seq_tokens, n_seqs);

    // in_proj: {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    var zx = b.buildLoraMm(layer.ssm_in.?, cur, null);
    b.cb(zx, "mamba_in_proj", il);
    // {8192, 5, 1, 1} -> {8192, 1, 5, 1}
    zx = c.ggml_permute(ctx, zx, 0, 2, 1, 3);
    zx = c.ggml_cont_4d(ctx, zx, head_dim * 2, n_heads, n_seq_tokens, n_seqs);
    b.cb(zx, "mamba_in_proj_out", il);

    // Split into z and x.
    // => {head_dim * n_heads, n_seq_tokens, n_seqs}
    var x: *Tensor = c.ggml_view_4d(ctx, zx, head_dim, n_heads, n_seq_tokens, n_seqs, zx.nb[1], zx.nb[2], zx.nb[3], @as(usize, @intCast(head_dim)) * c.ggml_element_size(zx));
    x = c.ggml_cont_3d(ctx, x, head_dim * n_heads, n_seq_tokens, n_seqs);
    b.cb(x, "mamba_x_split", il);

    const z: *Tensor = c.ggml_view_4d(ctx, zx, head_dim, n_heads, n_seq_tokens, n_seqs, zx.nb[1], zx.nb[2], zx.nb[3], 0);
    b.cb(z, "mamba_z_split", il);

    // conv1d
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        const conv_x: *Tensor = c.ggml_concat(ctx, conv, c.ggml_transpose(ctx, x), 0);
        b.cb(conv_x, "mamba_conv1d_input", il);

        // Copy the last (d_conv - 1) columns back into the state cache.
        const last_conv = c.ggml_view_3d(ctx, conv_x, d_conv - 1, d_inner, n_seqs, conv_x.nb[1], conv_x.nb[2], @as(usize, @intCast(n_seq_tokens)) * conv_x.nb[0]);

        c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, last_conv, c.ggml_view_1d(ctx, conv_states_all, (d_conv - 1) * (d_inner + 2 * n_group * d_state) * n_seqs, kv_head * @as(usize, @intCast((d_conv - 1) * (d_inner + 2 * n_group * d_state))) * c.ggml_element_size(conv_states_all))));
        b.cb(conv_states_all, "mamba_conv1d_state", il);

        // The 1D convolution.
        x = c.ggml_ssm_conv(ctx, conv_x, layer.ssm_conv1d);
        b.cb(x, "mamba_conv1d", il);

        x = c.ggml_silu(ctx, x);
        b.cb(x, "mamba_conv1d_silu", il);
    }

    // The SSM.
    {
        // bcdt_proj: {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        const x_bcdt = b.buildLoraMm(layer.ssm_x.?, x, null);
        b.cb(x_bcdt, "mamba_bcdt_proj", il);

        // Split into dt, B and C.
        const dt_dim: i64 = dtDim(hp.n_embd);
        var B: *Tensor = c.ggml_view_3d(ctx, x_bcdt, d_state, n_seq_tokens, n_seqs, x_bcdt.nb[1], x_bcdt.nb[2], 0);
        var C: *Tensor = c.ggml_view_3d(ctx, x_bcdt, d_state, n_seq_tokens, n_seqs, x_bcdt.nb[1], x_bcdt.nb[2], c.ggml_element_size(x_bcdt) * @as(usize, @intCast(d_state)));
        var dt: *Tensor = c.ggml_view_3d(ctx, x_bcdt, dt_dim, n_seq_tokens, n_seqs, x_bcdt.nb[1], x_bcdt.nb[2], c.ggml_element_size(x_bcdt) * @as(usize, @intCast(2 * d_state)));
        b.cb(B, "mamba_B_raw", il);
        b.cb(C, "mamba_C_raw", il);
        b.cb(dt, "mamba_dt_raw", il);

        // The RMS norm on dt, B and C, PLaMo-2's own.
        B = b.buildNorm(B, layer.ssm_b_norm, null, .rms, il);
        C = b.buildNorm(C, layer.ssm_c_norm, null, .rms, il);
        dt = b.buildNorm(dt, layer.ssm_dt_norm, null, .rms, il);
        b.cb(B, "mamba_B_normed", il);
        b.cb(C, "mamba_C_normed", il);
        b.cb(dt, "mamba_dt_normed", il);

        // dt_proj: {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = b.buildLoraMm(layer.ssm_dt.?, dt, null);
        dt = c.ggml_add(ctx, dt, layer.ssm_dt_b);
        b.cb(dt, "mamba_dt_proj", il);

        const A: *Tensor = c.ggml_reshape_2d(ctx, layer.ssm_a, 1, n_heads);
        b.cb(A, "mamba_A", il);

        const xes = c.ggml_element_size(x);
        const hd: usize = @intCast(head_dim);
        const nh: usize = @intCast(n_heads);
        const nst: usize = @intCast(n_seq_tokens);
        x = c.ggml_view_4d(ctx, x, head_dim, n_heads, n_seq_tokens, n_seqs, hd * xes, hd * nh * xes, hd * nh * nst * xes, 0);
        B = c.ggml_view_4d(ctx, B, d_state, 1, n_seq_tokens, n_seqs, @as(usize, @intCast(d_state)) * B.nb[0], B.nb[1], B.nb[2], 0);
        C = c.ggml_view_4d(ctx, C, d_state, 1, n_seq_tokens, n_seqs, @as(usize, @intCast(d_state)) * C.nb[0], C.nb[1], C.nb[2], 0);

        const rows: mamba_base.SsmRows = .{ .d_state = d_state, .head_dim = head_dim, .n_head = n_heads, .n_slots = mctx_cur.getSize(), .x = x, .dt = dt, .A = A, .B = B, .C = C, .K = 1 };
        const y_ssm = b.buildRs(inp, ssm_states_all, @bitCast(hp.nEmbdS()), @intCast(ubatch.n_seqs), rows.rowsFn());
        b.cb(y_ssm, "mamba_ssm_scan", il);

        // Store the last states.
        const ns: usize = @intCast(n_seqs);
        const ds: usize = @intCast(d_state);
        c.ggml_build_forward_expand(b.gf, c.ggml_cpy(
            ctx,
            c.ggml_view_1d(ctx, y_ssm, n_heads * head_dim * d_state * n_seqs, nh * hd * nst * ns * c.ggml_element_size(y_ssm)),
            c.ggml_view_1d(ctx, ssm_states_all, n_heads * head_dim * d_state * n_seqs, kv_head * ns * nh * hd * ds * c.ggml_element_size(ssm_states_all)),
        ));
        b.cb(ssm_states_all, "mamba_ssm_states", il);

        var y: *Tensor = c.ggml_view_4d(ctx, y_ssm, head_dim, n_heads, n_seq_tokens, n_seqs, hd * c.ggml_element_size(x), hd * nh * c.ggml_element_size(x), hd * nh * nst * c.ggml_element_size(x), 0);
        b.cb(y, "mamba_y_view", il);

        // Add the D parameter and gate with z.
        // {d_inner, n_seq_tokens, n_seqs} * {d_inner} => {d_inner, n_seq_tokens, n_seqs}
        const D: *Tensor = c.ggml_reshape_2d(ctx, layer.ssm_d, 1, n_heads);
        y = c.ggml_add(ctx, y, c.ggml_mul(ctx, x, D));
        b.cb(y, "mamba_y_add_d", il);

        y = c.ggml_swiglu_split(ctx, c.ggml_cont(ctx, z), y);
        b.cb(y, "mamba_y_swiglu_z", il);

        // out_proj: {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        y = c.ggml_view_3d(ctx, y, head_dim * n_heads, n_seq_tokens, n_seqs, y.nb[2], y.nb[3], 0);
        cur = b.buildLoraMm(layer.ssm_out.?, y, null);
        b.cb(cur, "mamba_out_proj", il);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = c.ggml_reshape_2d(ctx, cur, cur.ne[0], n_seq_tokens * n_seqs);
    b.cb(cur, "mamba_out", il);

    return cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
