//! Metis16: a hybrid of attention and Mamba-2 layers run in several passes
//! over shared weights, on hyper-connection streams, with a depth memory
//! written by every attention layer and read before every sublayer, a
//! continuation head at the end of each pass, and a latent mixture of
//! experts routed on the latent and on routing features.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/metis16.cpp` at v0.9.0 (`cd959bae9`):
//! `llama_model_metis16`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph` with the helpers it declares.

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

/// Mirrors `llama_model_metis16`'s members, with their defaults: `n_pass`,
/// `n_stream`, `n_mem`, `n_route`, `n_layer_phys` and `cont_limit`
/// (src/models/models.h:1807, 1808, 1809, 1810, 1811, 1812 @cd959bae9).
pub const State = struct {
    n_pass: u32 = 1,
    n_stream: u32 = 1,
    n_mem: u32 = 0,
    n_route: u32 = 0,
    n_layer_phys: u32 = 0,
    cont_limit: f32 = 0.0,
};

/// Ports `llama_model_metis16`'s `load_arch_hparams` (models/metis16.cpp:5 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
    hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
    hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
    hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
    hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;
    hp.ssm_norm_eps = (try meta.getKey(f32, .ssm_norm_eps, true, exc)).?;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_ff_shexp = (try meta.getKey(u32, .expert_shared_feed_forward_length, true, exc)).?;
    hp.moe_latent_size = (try meta.getKey(u32, .moe_latent_size, true, exc)).?;

    hp.dsv4_hc_mult = (try meta.getKey(u32, .hyper_connection_count, true, exc)).?;
    hp.metis_n_pass = (try meta.getKey(u32, .metis_pass_count, true, exc)).?;
    hp.metis_n_mem = (try meta.getKey(u32, .metis_memory_length, true, exc)).?;
    hp.metis_n_route = (try meta.getKey(u32, .metis_route_length, true, exc)).?;
    hp.metis_pass_lora_rank = (try meta.getKey(u32, .attention_pass_lora_rank, true, exc)).?;
    hp.metis_cont_limit = (try meta.getKey(f32, .metis_continuation_logit_limit, true, exc)).?;

    const st = &self.metis16;
    st.n_pass = hp.metis_n_pass;
    st.n_stream = hp.dsv4_hc_mult;
    st.n_mem = hp.metis_n_mem;
    st.n_route = hp.metis_n_route;
    st.cont_limit = hp.metis_cont_limit;

    if (st.n_pass < 1 or hp.nLayer() % st.n_pass != 0) {
        return exc.throw(.runtime_error, "metis16: {d} layers do not split into {d} passes", .{ hp.nLayer(), st.n_pass });
    }
    st.n_layer_phys = hp.nLayer() / st.n_pass;

    var has_attn = false;
    var il: u32 = 0;
    while (il < hp.nLayer()) : (il += 1) {
        if (hp.nHeadKv(il) != hp.nHeadKv(il % st.n_layer_phys)) {
            return exc.throw(.runtime_error, "metis16: layer {d} does not repeat the layout of layer {d}", .{ il, il % st.n_layer_phys });
        }
        hp.is_recr_impl[il] = @intFromBool(hp.nHeadKv(il) == 0);
        has_attn = has_attn or hp.is_recr_impl[il] == 0;
    }
    if (!has_attn) {
        return exc.throw(.runtime_error, "metis16: at least one attention layer is required", .{});
    }

    self.type = switch (st.n_layer_phys) {
        12 => .@"3b", // Praxis
        20 => .@"10b", // Logos
        else => .unknown,
    };
}

/// Ports `llama_model_metis16`'s `load_arch_tensors` (models/metis16.cpp:56 @cd959bae9).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    _ = ml;
    const hp = &self.hparams;
    const st = &self.metis16;
    const n_layer: u32 = hp.nLayer();
    const n_embd: i64 = hp.n_embd;
    const n_embd_head_k: i64 = hp.nEmbdHeadK(0);
    const n_vocab: i64 = self.vocab.nTokens();
    const n_expert: i64 = hp.n_expert;
    const nr = loader.tensor_not_required;

    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const n_ssm_head: i64 = hp.ssm_dt_rank;
    const n_group: i64 = hp.ssm_n_group;
    const d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head;
    const n_latent: i64 = hp.moe_latent_size;
    const n_ff_shexp: i64 = hp.n_ff_shexp;
    const n_s: i64 = st.n_stream;
    const n_p: i64 = st.n_pass;
    const n_m: i64 = st.n_mem;
    const n_r: i64 = st.n_route;

    var n_attn: i64 = 0;
    var l: u32 = 0;
    while (l < st.n_layer_phys) : (l += 1) {
        n_attn += @intFromBool(!hp.isRecr(l));
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, n_vocab }, 0, exc);
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ n_embd, n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ n_embd, n_vocab }, loader.tensor_duplicated, exc);
    }

    self.stream_embd = try self.createTensor(.{ .tensor = .stream_embd, .suffix = "weight" }, &.{ n_s, n_embd }, 0, exc);

    self.depth_mem_state_write = try self.createTensor(.{ .tensor = .depth_mem_state_write, .suffix = "weight" }, &.{ n_embd, n_m }, 0, exc);
    self.depth_mem_meta_write = try self.createTensor(.{ .tensor = .depth_mem_meta_write, .suffix = "weight" }, &.{ n_r + 3, n_m }, 0, exc);
    self.depth_mem_entry_bias = try self.createTensor(.{ .tensor = .depth_mem_entry_bias, .suffix = "weight" }, &.{ n_m, n_p * (n_attn + 1) }, 0, exc);
    self.depth_mem_query = try self.createTensor(.{ .tensor = .depth_mem_query, .suffix = "weight" }, &.{ n_embd, n_m }, 0, exc);
    self.depth_mem_query_b = try self.createTensor(.{ .tensor = .depth_mem_query, .suffix = "bias" }, &.{n_m}, 0, exc);
    self.depth_mem_key = try self.createTensor(.{ .tensor = .depth_mem_key, .suffix = "weight" }, &.{ n_m, n_m }, 0, exc);
    self.depth_mem_key_b = try self.createTensor(.{ .tensor = .depth_mem_key, .suffix = "bias" }, &.{n_m}, 0, exc);
    self.depth_mem_value = try self.createTensor(.{ .tensor = .depth_mem_value, .suffix = "weight" }, &.{ n_m, n_m }, 0, exc);
    self.depth_mem_value_b = try self.createTensor(.{ .tensor = .depth_mem_value, .suffix = "bias" }, &.{n_m}, 0, exc);
    self.depth_mem_out = try self.createTensor(.{ .tensor = .depth_mem_out, .suffix = "weight" }, &.{ n_m, n_embd }, 0, exc);
    self.depth_mem_out_b = try self.createTensor(.{ .tensor = .depth_mem_out, .suffix = "bias" }, &.{n_embd}, 0, exc);
    self.depth_mem_route_state = try self.createTensor(.{ .tensor = .depth_mem_route_state, .suffix = "weight" }, &.{ n_embd, n_r }, 0, exc);
    self.depth_mem_route_retr = try self.createTensor(.{ .tensor = .depth_mem_route_retr, .suffix = "weight" }, &.{ n_embd, n_r }, 0, exc);
    self.depth_mem_route_diff = try self.createTensor(.{ .tensor = .depth_mem_route_diff, .suffix = "weight" }, &.{ n_embd, n_r }, 0, exc);
    self.depth_mem_route_hist = try self.createTensor(.{ .tensor = .depth_mem_route_hist, .suffix = "weight" }, &.{ n_r, n_r }, 0, exc);
    self.depth_mem_route_b = try self.createTensor(.{ .tensor = .depth_mem_route, .suffix = "bias" }, &.{n_r}, 0, exc);
    self.depth_mem_stream_gate = try self.createTensor(.{ .tensor = .depth_mem_stream_gate, .suffix = "weight" }, &.{ n_embd, n_s }, 0, exc);
    self.depth_mem_stream_gate_b = try self.createTensor(.{ .tensor = .depth_mem_stream_gate, .suffix = "bias" }, &.{n_s}, 0, exc);

    self.cont_hidden_state = try self.createTensor(.{ .tensor = .cont_hidden_state, .suffix = "weight" }, &.{ n_embd, n_r }, 0, exc);
    self.cont_hidden_retr = try self.createTensor(.{ .tensor = .cont_hidden_retr, .suffix = "weight" }, &.{ n_embd, n_r }, 0, exc);
    self.cont_hidden_diff = try self.createTensor(.{ .tensor = .cont_hidden_diff, .suffix = "weight" }, &.{ n_embd, n_r }, 0, exc);
    self.cont_hidden_route = try self.createTensor(.{ .tensor = .cont_hidden_route, .suffix = "weight" }, &.{ n_r, n_r }, 0, exc);
    self.cont_hidden_b = try self.createTensor(.{ .tensor = .cont_hidden, .suffix = "bias" }, &.{n_r}, 0, exc);
    self.cont_output = try self.createTensor(.{ .tensor = .cont_output, .suffix = "weight" }, &.{ n_r, 1 }, 0, exc);
    self.cont_output_b = try self.createTensor(.{ .tensor = .cont_output, .suffix = "bias" }, &.{1}, 0, exc);

    var il: i32 = 0;
    while (il < @as(i32, @intCast(st.n_layer_phys))) : (il += 1) {
        const layer = &self.layers[@intCast(il)];
        const ilu: u32 = @intCast(il);

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, 0, exc);
        layer.hc_attn_mix = try self.createTensor(.{ .tensor = .hc_attn_mix, .suffix = "weight", .bid = il }, &.{ n_s, n_s, n_p }, 0, exc);
        layer.hc_attn_read = try self.createTensor(.{ .tensor = .hc_attn_read, .suffix = "weight", .bid = il }, &.{ n_s, n_p }, 0, exc);
        layer.hc_attn_write = try self.createTensor(.{ .tensor = .hc_attn_write, .suffix = "weight", .bid = il }, &.{ n_s, n_p }, 0, exc);
        layer.hc_ffn_mix = try self.createTensor(.{ .tensor = .hc_ffn_mix, .suffix = "weight", .bid = il }, &.{ n_s, n_s, n_p }, 0, exc);
        layer.hc_ffn_read = try self.createTensor(.{ .tensor = .hc_ffn_read, .suffix = "weight", .bid = il }, &.{ n_s, n_p }, 0, exc);
        layer.hc_ffn_write = try self.createTensor(.{ .tensor = .hc_ffn_write, .suffix = "weight", .bid = il }, &.{ n_s, n_p }, 0, exc);
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = il }, &.{n_embd}, 0, exc);

        if (hp.isRecr(ilu)) {
            layer.ssm_in = try self.createTensor(.{ .tensor = .ssm_in, .suffix = "weight", .bid = il }, &.{ n_embd, d_in_proj }, 0, exc);
            layer.ssm_conv1d = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "weight", .bid = il }, &.{ d_conv, d_inner + 2 * n_group * d_state }, 0, exc);
            layer.ssm_conv1d_b = try self.createTensor(.{ .tensor = .ssm_conv1d, .suffix = "bias", .bid = il }, &.{d_inner + 2 * n_group * d_state}, 0, exc);
            layer.ssm_dt_b = try self.createTensor(.{ .tensor = .ssm_dt, .suffix = "bias", .bid = il }, &.{n_ssm_head}, 0, exc);
            layer.ssm_norm = try self.createTensor(.{ .tensor = .ssm_norm, .suffix = "weight", .bid = il }, &.{ @divTrunc(d_inner, n_group), n_group }, 0, exc);
            layer.ssm_out = try self.createTensor(.{ .tensor = .ssm_out, .suffix = "weight", .bid = il }, &.{ d_inner, n_embd }, 0, exc);

            // No "weight" suffix for these.
            layer.ssm_a = try self.createTensor(.{ .tensor = .ssm_a, .bid = il }, &.{ 1, n_ssm_head }, 0, exc);
            layer.ssm_d = try self.createTensor(.{ .tensor = .ssm_d, .bid = il }, &.{ 1, n_ssm_head }, 0, exc);
        } else {
            const n_qkv = n_embd_head_k * (@as(i64, hp.nHead(ilu)) + 2 * @as(i64, hp.nHeadKv(ilu)));
            const rank: i64 = hp.metis_pass_lora_rank;

            layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = il }, &.{ n_embd, n_qkv }, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = il }, &.{ n_embd_head_k * hp.nHead(ilu), n_embd }, 0, exc);
            layer.attn_pass_lora_down = try self.createTensor(.{ .tensor = .attn_pass_lora_down, .suffix = "weight", .bid = il }, &.{ n_embd, rank }, 0, exc);
            layer.attn_pass_lora_up = try self.createTensor(.{ .tensor = .attn_pass_lora_up, .suffix = "weight", .bid = il }, &.{ rank, n_qkv }, 0, exc);
            layer.attn_pass_gate = try self.createTensor(.{ .tensor = .attn_pass_gate, .suffix = "weight", .bid = il }, &.{n_p}, 0, exc);
        }

        const n_ff_exp: i64 = hp.nFfExp(ilu);

        layer.ffn_latent_down = try self.createTensor(.{ .tensor = .ffn_latent_down, .suffix = "weight", .bid = il }, &.{ n_embd, n_latent }, 0, exc);
        layer.ffn_latent_up = try self.createTensor(.{ .tensor = .ffn_latent_up, .suffix = "weight", .bid = il }, &.{ n_latent, n_embd }, 0, exc);
        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = il }, &.{ n_latent + n_r, n_expert }, 0, exc);
        layer.ffn_gate_inp_b = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "bias", .bid = il }, &.{n_expert}, 0, exc);
        layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = il }, &.{n_expert}, 0, exc);
        layer.ffn_exp_embd = try self.createTensor(.{ .tensor = .ffn_exp_embd, .suffix = "weight", .bid = il }, &.{ n_r, n_expert }, 0, exc);
        layer.ffn_gate_up_exps = try self.createTensor(.{ .tensor = .ffn_gate_up_exps, .suffix = "weight", .bid = il }, &.{ n_latent, 2 * n_ff_exp, n_expert }, 0, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = il }, &.{ n_ff_exp, n_latent, n_expert }, 0, exc);
        layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = il }, &.{ n_latent, n_ff_shexp }, 0, exc);
        layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = il }, &.{ n_latent, n_ff_shexp }, 0, exc);
        layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = il }, &.{ n_ff_shexp, n_latent }, 0, exc);
    }

    // The virtual layers of later passes share the physical layer's weights.
    var v: u32 = st.n_layer_phys;
    while (v < n_layer) : (v += 1) {
        self.layers[v] = self.layers[v % st.n_layer_phys];
    }
}

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_metis16`'s `build_arch_graph` (models/metis16.cpp:168 @cd959bae9).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    _ = params;
    var g: Graph = .{ .b = b, .model = model };
    g.build();
}

/// `llama_model_metis16::graph`: the builder, and the memory bank of the
/// current forward that `mem_write` grows and `mem_read` reads.
const Graph = struct {
    b: *const graph.Builder,
    model: *const Model,
    /// Keys `[n_mem, n_entry, n_tokens]`.
    bank_k: ?*Tensor = null,
    /// Values `[n_entry, n_mem, n_tokens]`.
    bank_vt: ?*Tensor = null,

    /// Ports `llama_model_metis16::graph`'s `stream_mean` (models/metis16.cpp:173
    /// @cd959bae9). Streams are kept stream-innermost: `[n_stream, n_embd, n_tokens]`.
    fn streamMean(self: *const Graph, x: *Tensor) *Tensor {
        const ctx = self.b.ctx0;
        const mean = c.ggml_mean(ctx, c.ggml_reshape_2d(ctx, x, x.ne[0], x.ne[1] * x.ne[2]));
        return c.ggml_reshape_2d(ctx, mean, x.ne[1], x.ne[2]);
    }

    /// Ports `llama_model_metis16::graph`'s `hc_read` (models/metis16.cpp:179
    /// @cd959bae9): the normalized sublayer input.
    fn hcRead(self: *const Graph, x: *Tensor, mix: *Tensor, read: *Tensor, norm: ?*Tensor, p: c_int, mixed: **Tensor, il: c_int) *Tensor {
        const ctx = self.b.ctx0;
        const pu: usize = @intCast(p);
        const n_s = x.ne[0];
        const x2 = c.ggml_reshape_2d(ctx, x, n_s, x.ne[1] * x.ne[2]);

        const m = c.ggml_view_2d(ctx, mix, n_s, n_s, mix.nb[1], pu * mix.nb[2]);
        mixed.* = c.ggml_mul_mat(ctx, m, x2);

        const r = c.ggml_reshape_2d(ctx, c.ggml_view_1d(ctx, read, n_s, pu * read.nb[1]), n_s, 1);
        const src = c.ggml_reshape_2d(ctx, c.ggml_mul_mat(ctx, r, x2), x.ne[1], x.ne[2]);
        return self.b.buildNorm(src, norm, null, .rms, il);
    }

    /// Ports `llama_model_metis16::graph`'s `hc_write` (models/metis16.cpp:192 @cd959bae9).
    fn hcWrite(self: *const Graph, mixed: *Tensor, write: *Tensor, update: *Tensor, p: c_int) *Tensor {
        const ctx = self.b.ctx0;
        const pu: usize = @intCast(p);
        const n_s = mixed.ne[0];
        const w = c.ggml_reshape_2d(ctx, c.ggml_view_1d(ctx, write, n_s, pu * write.nb[1]), 1, n_s);
        const u = c.ggml_reshape_2d(ctx, update, 1, update.ne[0] * update.ne[1]);
        const out = c.ggml_add(ctx, mixed, c.ggml_mul_mat(ctx, w, u));
        return c.ggml_reshape_3d(ctx, out, n_s, update.ne[0], update.ne[1]);
    }

    /// Ports `llama_model_metis16::graph`'s `mem_read` (models/metis16.cpp:201
    /// @cd959bae9): blends the retrieved entries into the streams and returns
    /// their summary.
    fn memRead(self: *const Graph, x: **Tensor) *Tensor {
        const b = self.b;
        const ctx = b.ctx0;
        const model = self.model;
        const n_s = x.*.ne[0];
        const n_m: i64 = model.metis16.n_mem;
        const n_embd = b.n_embd;
        const n_tokens = b.n_tokens;

        const x_e = c.ggml_cont(ctx, c.ggml_permute(ctx, x.*, 1, 0, 2, 3));

        var q = c.ggml_mul_mat(ctx, model.depth_mem_query, c.ggml_reshape_2d(ctx, x_e, n_embd, n_s * n_tokens));
        q = c.ggml_reshape_3d(ctx, c.ggml_add(ctx, q, model.depth_mem_query_b), n_m, n_s, n_tokens);

        var kq = c.ggml_mul_mat(ctx, self.bank_k, q);
        kq = c.ggml_soft_max_ext(ctx, kq, null, 1.0 / impl.sqrtf(@floatFromInt(n_m)), 0.0);

        const ret = c.ggml_mul_mat(ctx, self.bank_vt, kq);
        var proj = c.ggml_mul_mat(ctx, model.depth_mem_out, c.ggml_reshape_2d(ctx, ret, n_m, n_s * n_tokens));
        proj = c.ggml_reshape_3d(ctx, c.ggml_add(ctx, proj, model.depth_mem_out_b), n_embd, n_s, n_tokens);

        var gate = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, x_e, model.depth_mem_stream_gate));
        gate = c.ggml_sigmoid(ctx, c.ggml_add(ctx, gate, c.ggml_reshape_2d(ctx, model.depth_mem_stream_gate_b, 1, n_s)));

        const proj_s = c.ggml_cont(ctx, c.ggml_permute(ctx, proj, 1, 0, 2, 3));
        x.* = c.ggml_add(ctx, x.*, c.ggml_mul(ctx, proj_s, c.ggml_reshape_3d(ctx, gate, n_s, 1, n_tokens)));

        return self.streamMean(proj_s);
    }

    /// Ports `llama_model_metis16::graph`'s `mem_write` (models/metis16.cpp:226 @cd959bae9).
    fn memWrite(self: *Graph, x: *Tensor, summary: ?*Tensor, entropy: ?*Tensor, confidence: ?*Tensor, cont_: ?*Tensor, entry: c_int) void {
        const b = self.b;
        const ctx = b.ctx0;
        const model = self.model;
        const n_m: i64 = model.metis16.n_mem;

        var cont = cont_;
        if (cont == null) {
            cont = c.ggml_scale(ctx, confidence, 0.0);
        }
        const meta = c.ggml_concat(ctx, c.ggml_concat(ctx, summary, entropy, 0), c.ggml_concat(ctx, confidence, cont, 0), 0);

        var e = c.ggml_add(ctx, c.ggml_mul_mat(ctx, model.depth_mem_state_write, self.streamMean(x)), c.ggml_mul_mat(ctx, model.depth_mem_meta_write, meta));
        e = c.ggml_add(ctx, e, c.ggml_view_1d(ctx, model.depth_mem_entry_bias, n_m, @as(usize, @intCast(entry)) * model.depth_mem_entry_bias.?.nb[1]));

        var k = c.ggml_add(ctx, c.ggml_mul_mat(ctx, model.depth_mem_key, e), model.depth_mem_key_b);
        var v = c.ggml_add(ctx, c.ggml_mul_mat(ctx, model.depth_mem_value, e), model.depth_mem_value_b);
        k = c.ggml_reshape_3d(ctx, k, n_m, 1, b.n_tokens);
        v = c.ggml_reshape_3d(ctx, v, 1, n_m, b.n_tokens);
        self.bank_k = if (self.bank_k) |bk| c.ggml_concat(ctx, bk, k, 1) else k;
        self.bank_vt = if (self.bank_vt) |bv| c.ggml_concat(ctx, bv, v, 0) else v;
    }

    /// Ports `llama_model_metis16::graph`'s `route_features` (models/metis16.cpp:246 @cd959bae9).
    fn routeFeatures(self: *const Graph, state: *Tensor, summary: ?*Tensor, diff: ?*Tensor, history: ?*Tensor) *Tensor {
        const ctx = self.b.ctx0;
        const model = self.model;
        var f = c.ggml_mul_mat(ctx, model.depth_mem_route_state, state);
        if (summary) |s| {
            f = c.ggml_add(ctx, f, c.ggml_mul_mat(ctx, model.depth_mem_route_retr, s));
        }
        if (diff) |d| {
            f = c.ggml_add(ctx, f, c.ggml_mul_mat(ctx, model.depth_mem_route_diff, d));
        }
        if (history) |h| {
            f = c.ggml_add(ctx, f, c.ggml_mul_mat(ctx, model.depth_mem_route_hist, h));
        }
        return c.ggml_tanh(ctx, c.ggml_add(ctx, f, model.depth_mem_route_b));
    }

    /// Ports the constructor of `llama_model_metis16`'s `graph` (models/metis16.cpp:261 @cd959bae9).
    fn build(self: *Graph) void {
        const b = self.b;
        const ctx = b.ctx0;
        const model = self.model;
        const hp = b.hparams;
        const st = &model.metis16;
        const n_s: i64 = st.n_stream;
        const n_p: c_int = @intCast(st.n_pass);
        const n_phys: c_int = @intCast(st.n_layer_phys);
        const n_r: i64 = st.n_route;
        const n_hd: i64 = hp.nEmbdHeadK(0);
        const n_exp: i64 = hp.n_expert;
        const n_used: i64 = hp.nExpertUsed(0);
        const n_embd = b.n_embd;
        const n_tokens = b.n_tokens;

        const inpL = b.buildInpEmbd(model.tok_embd.?);
        const inp = b.buildInpMemHybrid();
        const inp_out_ids = b.buildInpOutIds();

        var x: *Tensor = c.ggml_repeat_4d(ctx, c.ggml_reshape_3d(ctx, inpL, 1, n_embd, n_tokens), n_s, n_embd, n_tokens, 1);
        x = c.ggml_add(ctx, x, model.stream_embd);
        b.cb(x, "streams_in", -1);

        var prev_state = self.streamMean(x);
        var history: ?*Tensor = null; // routing history, [n_route, n_tokens]
        var cont: ?*Tensor = null; // depth head reading of the last pass, [1, n_tokens]
        var n_entry: c_int = 0;

        var p: c_int = 0;
        while (p < n_p) : (p += 1) {
            var summ: ?*Tensor = null;
            var ent: ?*Tensor = null;
            var conf: ?*Tensor = null;

            var il_p: c_int = 0;
            while (il_p < n_phys) : (il_p += 1) {
                const il: c_int = p * n_phys + il_p;
                const ilu: u32 = @intCast(il);
                const layer = &model.layers[ilu];
                const is_attn = !hp.isRecr(ilu);
                const writes = is_attn or il_p == n_phys - 1;

                const retrieved: ?*Tensor = if (self.bank_k != null) self.memRead(&x) else null;
                const feats = self.routeFeatures(self.streamMean(x), retrieved, null, history);

                var mixed: *Tensor = undefined;
                var cur = self.hcRead(x, layer.hc_attn_mix.?, layer.hc_attn_read.?, layer.attn_norm, p, &mixed, il);
                if (is_attn) {
                    var qkv = b.buildLoraMm(layer.wqkv.?, cur, null);
                    const lora = c.ggml_mul_mat(ctx, layer.attn_pass_lora_up, c.ggml_mul_mat(ctx, layer.attn_pass_lora_down, cur));
                    qkv = c.ggml_add(ctx, qkv, c.ggml_mul(ctx, lora, c.ggml_view_1d(ctx, layer.attn_pass_gate, 1, @as(usize, @intCast(p)) * layer.attn_pass_gate.?.nb[0])));

                    const n_head: i64 = hp.nHead(ilu);
                    const n_head_kv: i64 = hp.nHeadKv(ilu);
                    const es: usize = c.ggml_element_size(qkv);
                    const hd: usize = @intCast(n_hd);
                    const Q = c.ggml_cont(ctx, c.ggml_view_3d(ctx, qkv, n_hd, n_head, n_tokens, hd * es, qkv.nb[1], 0));
                    const K = c.ggml_cont(ctx, c.ggml_view_3d(ctx, qkv, n_hd, n_head_kv, n_tokens, hd * es, qkv.nb[1], hd * @as(usize, @intCast(n_head)) * es));
                    const V = c.ggml_cont(ctx, c.ggml_view_3d(ctx, qkv, n_hd, n_head_kv, n_tokens, hd * es, qkv.nb[1], hd * @as(usize, @intCast(n_head + n_head_kv)) * es));

                    cur = b.buildAttn(inp.getAttn(), layer.wo, null, null, Q, K, V, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_hd)), il);
                } else {
                    cur = mamba_base.buildMamba2Layer(b, inp.getRecr(), cur, model, il);
                }
                b.cb(cur, "mixer_out", il);
                x = self.hcWrite(mixed, layer.hc_attn_write.?, cur, p);

                cur = self.hcRead(x, layer.hc_ffn_mix.?, layer.hc_ffn_read.?, layer.ffn_norm, p, &mixed, il);
                {
                    const latent = c.ggml_mul_mat(ctx, layer.ffn_latent_down, cur);
                    var logits = c.ggml_mul_mat(ctx, layer.ffn_gate_inp, c.ggml_concat(ctx, latent, feats, 0));
                    logits = c.ggml_add(ctx, logits, layer.ffn_gate_inp_b);

                    // The balance bias picks the experts; the weights come from the unbiased logits.
                    const selected = c.ggml_argsort_top_k(ctx, c.ggml_add(ctx, logits, layer.ffn_exp_probs_b), @intCast(n_used));

                    const moe_out = b.buildMoeFfn(latent, .{
                        .gate_inp = null,
                        .up_exps = null,
                        .gate_exps = null,
                        .down_exps = layer.ffn_down_exps.?,
                        .n_expert = n_exp,
                        .n_expert_used = n_used,
                        .type_op = .silu,
                        .norm_w = false,
                        .w_scale = 0.0,
                        .gating_op = .softmax_weight,
                        .probs_in = logits,
                        .gate_up_exps = layer.ffn_gate_up_exps,
                        .selected_experts_in = selected,
                    }, il_p);
                    const shexp = b.buildFfn(latent, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il_p);
                    cur = c.ggml_mul_mat(ctx, layer.ffn_latent_up, c.ggml_add(ctx, moe_out, shexp));
                    b.cb(cur, "ffn_out", il);

                    // Routing summary: the chosen experts' identity vectors blended by their weights.
                    const logits3 = c.ggml_reshape_3d(ctx, logits, 1, n_exp, n_tokens);
                    const weights = c.ggml_soft_max(ctx, c.ggml_reshape_2d(ctx, c.ggml_get_rows(ctx, logits3, selected), n_used, n_tokens));
                    var rows = c.ggml_get_rows(ctx, layer.ffn_exp_embd, c.ggml_reshape_1d(ctx, c.ggml_cont(ctx, selected), n_used * n_tokens));
                    rows = c.ggml_cont(ctx, c.ggml_permute(ctx, c.ggml_reshape_3d(ctx, rows, n_r, n_used, n_tokens), 1, 0, 2, 3));
                    summ = c.ggml_reshape_2d(ctx, c.ggml_mul_mat(ctx, rows, c.ggml_reshape_3d(ctx, weights, n_used, 1, n_tokens)), n_r, n_tokens);

                    if (writes) {
                        const probs = c.ggml_soft_max(ctx, logits);
                        ent = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, probs, c.ggml_log(ctx, c.ggml_clamp(ctx, probs, 1.0e-9, 1.0))));
                        ent = c.ggml_scale(ctx, ent, -1.0);
                        conf = c.ggml_get_rows(ctx, c.ggml_reshape_3d(ctx, probs, 1, n_exp, n_tokens), c.ggml_reshape_2d(ctx, c.ggml_argmax(ctx, probs), 1, n_tokens));
                        conf = c.ggml_reshape_2d(ctx, conf, 1, n_tokens);
                    }
                }
                x = self.hcWrite(mixed, layer.hc_ffn_write.?, cur, p);
                b.cb(x, "streams", il);

                history = if (history) |h| c.ggml_add(ctx, c.ggml_scale(ctx, h, 0.5), c.ggml_scale(ctx, summ, 0.5)) else c.ggml_scale(ctx, summ, 0.5);

                if (is_attn) {
                    self.memWrite(x, summ, ent, conf, cont, n_entry);
                    n_entry += 1;
                }
            }

            // The end of the pass: one more memory read, the depth head, and an
            // entry recording where the pass left the token.
            const retrieved = self.memRead(&x);
            const state = self.streamMean(x);
            const diff = c.ggml_sub(ctx, state, prev_state);
            const feats = self.routeFeatures(state, retrieved, diff, history);

            var h = c.ggml_mul_mat(ctx, model.cont_hidden_state, state);
            h = c.ggml_add(ctx, h, c.ggml_mul_mat(ctx, model.cont_hidden_retr, retrieved));
            h = c.ggml_add(ctx, h, c.ggml_mul_mat(ctx, model.cont_hidden_diff, diff));
            h = c.ggml_add(ctx, h, c.ggml_mul_mat(ctx, model.cont_hidden_route, feats));
            h = c.ggml_silu(ctx, c.ggml_add(ctx, h, model.cont_hidden_b));

            var logit = c.ggml_add(ctx, c.ggml_mul_mat(ctx, model.cont_output, h), model.cont_output_b);
            const scale = c.ggml_clamp(ctx, c.ggml_sqrt(ctx, c.ggml_mean(ctx, c.ggml_sqr(ctx, state))), 1.0e-6, std.math.inf(f32));
            const innov = c.ggml_sqrt(ctx, c.ggml_mean(ctx, c.ggml_sqr(ctx, diff)));
            logit = c.ggml_add(ctx, logit, c.ggml_scale_bias(ctx, c.ggml_tanh(ctx, c.ggml_div(ctx, innov, scale)), 1.0, -0.5));
            cont = c.ggml_sigmoid(ctx, c.ggml_clamp(ctx, logit, -st.cont_limit, st.cont_limit));
            b.cb(cont.?, "continuation", p);

            self.memWrite(x, summ, ent, conf, cont, n_entry);
            n_entry += 1;
            prev_state = state;
        }

        var cur = self.streamMean(x);
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);

        cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
        b.cb(cur, "result_norm", -1);
        b.res.t_embd = cur;

        cur = b.buildLoraMm(model.output.?, cur, null);
        b.cb(cur, "result_output", -1);
        b.res.t_logits = cur;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
