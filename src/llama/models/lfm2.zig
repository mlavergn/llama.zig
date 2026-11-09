//! LFM2: Liquid's hybrid of gated short convolutions and grouped-query
//! attention with Q/K norms, each followed by a SwiGLU feed-forward or,
//! past the leading dense blocks, a mixture of experts; the attention
//! layers may use a sliding window.
//!
//! A decision model (LiquidAI's d1, since v0.7.0) puts `n_layer_decision`
//! plain pre-norm blocks with biases after a non-causal trunk without a
//! memory, and scores every output token once per question type instead
//! of computing logits. Media entries (an image or audio prefix, given as
//! embeddings) never read the text in the trunk, and the head keeps the
//! two kinds apart.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/lfm2.cpp` at v0.8.0 (`8642541b2`):
//! `llama_model_lfm2`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph<iswa>` with the lambdas inside it, and
//! the decision model's `N_DECISION_TYPES`, `lfm2_is_media`, the graph
//! inputs `llm_graph_input_attn_media` and `llm_graph_input_conv_mask`, and
//! `graph_decision` with `build_decision_head`; and from
//! `llama.cpp/src/models/models.h` at v0.8.0, `llama_model_lfm2`'s
//! `graph_decision`. The short convolution and the feed-forwards are
//! `lfm_base.zig`'s. LFM2-MoE (`lfm2moe.zig`) builds the same graph.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const memory_hybrid = @import("../memory_hybrid.zig");
const memory_hybrid_iswa = @import("../memory_hybrid_iswa.zig");
const lfm_base = @import("lfm_base.zig");

const batch = @import("../batch.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const Ubatch = batch.Ubatch;
const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("models/lfm2.cpp", 0, "out of memory");
}

/// Ports `N_DECISION_TYPES` (models/lfm2.cpp:8 @8642541b2): the question
/// types of a decision model -- choice, score, noul.
const n_decision_types: u32 = 3;

/// Ports `llama_model_lfm2`'s `load_arch_hparams` (models/lfm2.cpp:10 @8642541b2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.n_shortconv_l_cache = (try meta.getKey(u32, .shortconv_l_cache, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    var il: u32 = 0;
    while (il < hp.nLayer()) : (il += 1) {
        hp.is_recr_impl[il] = @intFromBool(hp.nHeadKv(il) == 0);
    }

    hp.n_layer_dense_lead = hp.nLayer();

    self.type = switch (hp.nFf(0)) {
        2560 => .@"230m",
        4608 => .@"350m",
        6912 => .@"700m",
        8192 => .@"1_2b",
        10752 => .@"2_6b",
        else => .unknown,
    };

    if (try meta.getKey(u32, .decision_block_count, false, exc)) |v| hp.n_layer_decision = v;
    if (hp.n_layer_decision > 0) {
        if (hp.n_layer_decision >= hp.nLayer() or hp.causal_attn) {
            return exc.throw(.runtime_error, "invalid decision head", .{});
        }
        hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;
        hp.n_embd_out_impl = n_decision_types;
    }

    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| {
        hp.n_swa = v;
        if (hp.n_swa > 0) {
            hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
            il = 0;
            while (il < hp.nLayer()) : (il += 1) {
                hp.is_swa_impl[il] = @intFromBool(hp.is_recr_impl[il] == 0);
            }
        }
    }
}

/// Ports `llama_model_lfm2`'s `load_arch_tensors` (models/lfm2.cpp:46
/// @8642541b2); LFM2-MoE's (models/lfm2moe.cpp:23 @8642541b2) is the same
/// body without the decision head, which it never reaches: LFM2-MoE reads
/// no decision block count, so its `n_layer_decision` is 0.
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    try loadTensors(self, "models/lfm2.cpp", 102, 120, exc);
}

/// The body both `load_arch_tensors` share, with the lines its two
/// assertions are at in each file.
pub fn loadTensors(self: *Model, comptime file: [:0]const u8, comptime line_moe: c_int, comptime line_gqa: c_int, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm_lfm2, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    const n_layer_trunk: i32 = l.n_layer - @as(i32, @intCast(hp.n_layer_decision));

    if (hp.n_layer_decision > 0) {
        // The decision head: plain pre-norm blocks with biases.
        var i: i32 = n_layer_trunk;
        while (i < l.n_layer) : (i += 1) {
            const layer = &self.layers[@intCast(i)];
            const n_ff_head: i64 = hp.nFf(@intCast(i));

            layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.attn_norm_b = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

            layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, 3 * l.n_embd }, 0, exc);
            layer.wqkv_b = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "bias", .bid = i }, &.{3 * l.n_embd}, 0, exc);
            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
            layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

            layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.ffn_norm_b = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_head }, 0, exc);
            layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{n_ff_head}, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ n_ff_head, l.n_embd }, 0, exc);
            layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
        }

        if (l.n_token_types != n_decision_types) {
            return exc.throw(.runtime_error, "decision model must have one token type per question type", .{});
        }
        self.type_embd = try self.createTensor(.{ .tensor = .token_types, .suffix = "weight" }, &.{ l.n_embd, l.n_token_types }, 0, exc);

        self.cls_norm = try self.createTensor(.{ .tensor = .cls_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
        self.cls_norm_b = try self.createTensor(.{ .tensor = .cls_norm, .suffix = "bias" }, &.{l.n_embd}, 0, exc);
        self.cls = try self.createTensor(.{ .tensor = .cls, .suffix = "weight" }, &.{ l.n_embd, l.n_embd }, 0, exc);
        self.cls_b = try self.createTensor(.{ .tensor = .cls, .suffix = "bias" }, &.{l.n_embd}, 0, exc);
        self.cls_out = try self.createTensor(.{ .tensor = .cls_out, .suffix = "weight" }, &.{ l.n_embd, 1 }, 0, exc);
        self.cls_out_b = try self.createTensor(.{ .tensor = .cls_out, .suffix = "bias" }, &.{1}, 0, exc);
    } else {
        self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

        if (self.output == null) {
            self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
        }
    }

    var i: i32 = 0;
    while (i < n_layer_trunk) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        const is_moe_layer = i >= @as(c_int, @bitCast(hp.n_layer_dense_lead));

        // The FFN or MoE is the same for the attention and conv layers.
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        if (is_moe_layer) {
            impl.assert(l.n_expert != 0 and l.n_expert_used != 0, file, line_moe, "n_expert && n_expert_used");
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.nFfExp(0), l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ hp.nFfExp(0), l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.nFfExp(0), l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);
        } else {
            // Dense.
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        }

        // For the operator norm.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (!hp.isRecr(@intCast(i))) {
            layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
            layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
            impl.assert(l.n_embd_v_gqa == l.n_embd_k_gqa, file, line_gqa, "n_embd_v_gqa == n_embd_k_gqa");

            try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, hp.nEmbdKGqa(@intCast(i)), hp.nEmbdVGqa(@intCast(i)), 0, exc);

            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        } else {
            layer.shortconv.conv = try self.createTensor(.{ .tensor = .shortconv_conv, .suffix = "weight", .bid = i }, &.{ hp.n_shortconv_l_cache, l.n_embd }, 0, exc);
            layer.shortconv.in_proj = try self.createTensor(.{ .tensor = .shortconv_inproj, .suffix = "weight", .bid = i }, &.{ l.n_embd, 3 * l.n_embd }, 0, exc);
            layer.shortconv.out_proj = try self.createTensor(.{ .tensor = .shortconv_outproj, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        }
    }

    // For LFM2-ColBert-350M.
    self.dense_2_out_layers = try self.createTensor(.{ .tensor = .dense_2_out, .suffix = "weight" }, &.{ l.n_embd, hp.nEmbdOut() }, nr, exc);
    self.dense_2_out_layers_b = try self.createTensor(.{ .tensor = .dense_2_out, .suffix = "bias" }, &.{hp.nEmbdOut()}, nr, exc);
}

/// Ports `llama_model_lfm2`'s `build_arch_graph` (models/lfm2.cpp:137
/// @8642541b2); LFM2-MoE's (models/lfm2moe.cpp:78 @8642541b2) is the same
/// without the decision graph, which its `n_layer_decision` of 0 never
/// takes.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    if (model.hparams.n_layer_decision > 0) {
        buildGraphDecision(model, b);
        return;
    }
    const swa_type: hparams_mod.SwaType = @fromBackingInt(@intCast(model.hparams.swa_type));
    if (swa_type == .standard) {
        buildGraph(true, model, b);
    } else {
        buildGraph(false, model, b);
    }
}

/// The constructor's `build_attn_block` lambda (models/lfm2.cpp:155
/// @8642541b2): Q/K norms, RoPE and attention.
fn buildAttnBlock(model: *const Model, b: *const graph.Builder, cur_: *Tensor, inp_pos: *Tensor, inp_attn: anytype, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    impl.assert(hp.nEmbdVGqa(@intCast(il)) == hp.nEmbdKGqa(@intCast(il)), "models/lfm2.cpp", 159, "hparams.n_embd_v_gqa(il) == hparams.n_embd_k_gqa(il)");
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const n_head_kv: i64 = hp.nHeadKv(@intCast(il));

    const qkv = b.buildQkv(layer, cur_, n_embd_head, b.n_head, n_head_kv, il);
    var q = qkv.q;
    var k = qkv.k;
    const v = qkv.v;

    // The QK norms.
    q = b.buildNorm(q, layer.attn_q_norm, null, .rms, il);
    b.cb(q, "model.layers.{}.self_attn.q_layernorm", il);
    k = b.buildNorm(k, layer.attn_k_norm, null, .rms, il);
    b.cb(k, "model.layers.{}.self_attn.k_layernorm", il);

    // RoPE.
    q = c.ggml_rope_ext(ctx, q, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    k = c.ggml_rope_ext(ctx, k, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    const cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, q, k, v, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);

    b.cb(cur, "model.layers.{}.self_attn.out_proj", il);

    return cur;
}

/// Ports the constructor of `llama_model_lfm2`'s `graph<iswa>`
/// (models/lfm2.cpp:149 @8642541b2).
fn buildGraph(comptime iswa: bool, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const MemHybridCtx = if (iswa) memory_hybrid_iswa.HybridIswaContext else memory_hybrid.HybridContext;

    var cur = b.buildInpEmbd(model.tok_embd.?);
    b.cb(cur, "model.embed_tokens", -1);

    c.ggml_build_forward_expand(b.gf, cur);

    const inp_hybrid = if (iswa) b.buildInpMemHybridIswa() else b.buildInpMemHybrid();

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = cur;

        const is_moe_layer = il >= @as(c_int, @bitCast(hp.n_layer_dense_lead));

        var prev_cur = cur;
        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "model.layers.{}.operator_norm", il);

        if (hp.isRecr(@intCast(il))) {
            const mctx_recr = b.mctx.?.cast(MemHybridCtx).getRecr();
            cur = lfm_base.buildLfmShortconv(model, b, mctx_recr, inp_hybrid.getRecr(), cur, il);
        } else {
            cur = buildAttnBlock(model, b, cur, inp_pos, inp_hybrid.getAttn(), il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            prev_cur = c.ggml_get_rows(ctx, prev_cur, inp_out_ids);
        }

        cur = c.ggml_add(ctx, prev_cur, cur);

        const ffn_norm_out = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        const ffn_out = if (is_moe_layer) lfm_base.buildLfmMoeFfn(model, b, ffn_norm_out, il) else lfm_base.buildLfmDenseFfn(model, b, ffn_norm_out, il);
        // The C++ names the norm's output here, not the FFN's.
        b.cb(ffn_norm_out, "model.layers.{}.ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_out);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);
    }

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    if (!b.cparams.embeddings) {
        cur = b.buildLoraMm(model.output.?, cur, model.output_s);
        b.cb(cur, "result_output", -1);

        b.res.t_logits = cur;
    }
}

// -----------------------------------------------------------------------------
// The decision model

/// Ports `lfm2_is_media` (models/lfm2.cpp:251 @8642541b2): a media entry
/// (an image or audio prefix) is an embedding, a text entry a token.
fn isMedia(ubatch: *const Ubatch, i: usize) bool {
    return if (ubatch.isMixed()) ubatch.type[i] != 0 else ubatch.token == null;
}

/// Ports `llm_graph_input_attn_media` (models/lfm2.cpp:257 @8642541b2):
/// non-causal within a sequence, the media never reading the text, so that
/// it is a function of the media alone; in the head, the text and the
/// media each read only their own kind. The C++ derives from
/// `llm_graph_input_attn_no_cache` and overrides `set_input`; here the
/// base is a member the attention reads.
const InpAttnMedia = struct {
    base: graph.InpAttnNoCache,
    is_head: bool,

    /// The `fill_mask` lambda of `set_input`.
    fn fillMask(self: *const InpAttnMedia, comptime T: type, data: [*]T, ubatch: *const Ubatch, is_media: []const bool, zero: T, ninf: T) void {
        const n_tokens: usize = ubatch.n_tokens;
        for (0..n_tokens) |j1| {
            for (0..n_tokens) |j0| {
                var visible = ubatch.seq_id[j0][0] == ubatch.seq_id[j1][0];
                if (self.is_head) {
                    visible = visible and is_media[j0] == is_media[j1];
                } else {
                    visible = visible and !(is_media[j1] and !is_media[j0]);
                }
                data[j1 * n_tokens + j0] = if (visible) zero else ninf;
            }
        }
    }

    /// Ports `llm_graph_input_attn_media`'s `set_input` (models/lfm2.cpp:262
    /// @8642541b2).
    pub fn setInput(self: *InpAttnMedia, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const n_tokens: usize = ubatch.n_tokens;

        const is_media = allocator.alloc(bool, n_tokens) catch oom();
        defer allocator.free(is_media);
        for (is_media, 0..) |*m, i| m.* = isMedia(ubatch, i);

        const m = self.base.self_kq_mask.?;
        impl.assert(c.ggml_backend_buffer_is_host(m.buffer), "models/lfm2.cpp", 284, "ggml_backend_buffer_is_host(self_kq_mask->buffer)");
        if (m.type == c.GGML_TYPE_F16) {
            self.fillMask(u16, @ptrCast(@alignCast(m.data)), ubatch, is_media, c.ggml_fp32_to_fp16(0.0), c.ggml_fp32_to_fp16(-std.math.inf(f32)));
        } else {
            self.fillMask(f32, @ptrCast(@alignCast(m.data)), ubatch, is_media, 0.0, -std.math.inf(f32));
        }
    }
};

/// Ports `llm_graph_input_conv_mask` (models/lfm2.cpp:297 @8642541b2): 1
/// where the previous (next) token is the left (right) neighbour in the
/// same sequence; the last media entry does not read the text on its
/// right.
const InpConvMask = struct {
    /// F32 `[1, n_tokens]`.
    left: ?*Tensor = null,
    /// F32 `[1, n_tokens]`.
    right: ?*Tensor = null,

    /// Ports `llm_graph_input_conv_mask`'s `set_input` (models/lfm2.cpp:299
    /// @8642541b2).
    pub fn setInput(self: *InpConvMask, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const n_tokens: usize = ubatch.n_tokens;

        const data_left = allocator.alloc(f32, n_tokens) catch oom();
        defer allocator.free(data_left);
        const data_right = allocator.alloc(f32, n_tokens) catch oom();
        defer allocator.free(data_right);
        @memset(data_left, 0.0);
        @memset(data_right, 0.0);
        var i: usize = 0;
        while (i + 1 < n_tokens) : (i += 1) {
            const is_next = ubatch.seq_id[i][0] == ubatch.seq_id[i + 1][0] and ubatch.pos[i] +% 1 == ubatch.pos[i + 1];
            data_right[i] = @floatFromInt(@intFromBool(is_next and !(isMedia(ubatch, i) and !isMedia(ubatch, i + 1))));
            data_left[i + 1] = @floatFromInt(@intFromBool(is_next));
        }
        c.ggml_backend_tensor_set(self.left, data_left.ptr, 0, c.ggml_nbytes(self.left));
        c.ggml_backend_tensor_set(self.right, data_right.ptr, 0, c.ggml_nbytes(self.right));
    }
};

/// Ports the constructor of `llama_model_lfm2`'s `graph_decision`
/// (models/lfm2.cpp:317 @8642541b2): the non-causal trunk without a
/// memory, then the decision head. The trunk's short convolution is a
/// centred three-tap one over the batch, a tap outside the sequence
/// reading 0. `llama_model_lfm2` declares it as `graph_decision`
/// (src/models/models.h:2415 @8642541b2).
fn buildGraphDecision(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const n_layer_enc: c_int = @as(c_int, @intCast(b.n_layer)) - @as(c_int, @intCast(hp.n_layer_decision));
    const n_embd = b.n_embd;
    const n_tokens = b.n_tokens;

    var cur = b.buildInpEmbd(model.tok_embd.?);
    b.cb(cur, "model.embed_tokens", -1);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    const type_mask: c.enum_ggml_type = if (b.cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;

    var inp_attn: [2]*graph.InpAttnNoCache = undefined;
    for ([_]bool{ false, true }) |is_head| {
        const inp = allocator.create(InpAttnMedia) catch oom();
        inp.* = .{ .base = .{ .hparams = hp, .cparams = b.cparams.*, .debug = 0 }, .is_head = is_head };
        inp.base.self_kq_mask = c.ggml_new_tensor_4d(ctx, type_mask, n_tokens, n_tokens, 1, 1);
        c.ggml_set_input(inp.base.self_kq_mask);
        inp.base.self_kq_mask_cnv = inp.base.self_kq_mask;
        inp_attn[@intFromBool(is_head)] = &b.res.addInput(InpAttnMedia, inp).base;
    }

    const inp_conv = allocator.create(InpConvMask) catch oom();
    inp_conv.* = .{};
    inp_conv.left = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 1, n_tokens);
    inp_conv.right = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 1, n_tokens);
    c.ggml_set_input(inp_conv.left);
    c.ggml_set_input(inp_conv.right);
    const conv_left = inp_conv.left.?;
    const conv_right = inp_conv.right.?;
    _ = b.res.addInput(InpConvMask, inp_conv);

    var il: c_int = 0;
    while (il < n_layer_enc) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        const inpL = cur;
        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "model.layers.{}.operator_norm", il);

        if (hp.isRecr(@intCast(il))) {
            const bcx = b.buildLoraMm(layer.shortconv.in_proj.?, cur, null);
            b.cb(bcx, "model.layers.{}.conv.in_proj", il);

            const es: usize = c.ggml_element_size(bcx);
            const row: usize = @intCast(n_embd);
            const bv: *Tensor = c.ggml_view_2d(ctx, bcx, n_embd, n_tokens, bcx.nb[1], 0 * row * es);
            const cv: *Tensor = c.ggml_view_2d(ctx, bcx, n_embd, n_tokens, bcx.nb[1], 1 * row * es);
            const xv: *Tensor = c.ggml_view_2d(ctx, bcx, n_embd, n_tokens, bcx.nb[1], 2 * row * es);

            // A centred 3-tap conv; a tap outside the sequence reads 0.
            const bx: *Tensor = c.ggml_mul(ctx, bv, xv);
            const bxp: *Tensor = c.ggml_pad_ext(ctx, bx, 0, 0, 1, 1, 0, 0, 0, 0);
            const prv: *Tensor = c.ggml_view_2d(ctx, bxp, n_embd, n_tokens, bxp.nb[1], 0);
            const nxt: *Tensor = c.ggml_view_2d(ctx, bxp, n_embd, n_tokens, bxp.nb[1], 2 * bxp.nb[1]);

            impl.assert(hp.n_shortconv_l_cache == 3, "models/lfm2.cpp", 369, "hparams.n_shortconv_l_cache == 3");
            const taps: *Tensor = c.ggml_cont(ctx, c.ggml_transpose(ctx, layer.shortconv.conv));
            const tap0: *Tensor = c.ggml_view_1d(ctx, taps, n_embd, 0 * taps.nb[1]);
            const tap1: *Tensor = c.ggml_view_1d(ctx, taps, n_embd, 1 * taps.nb[1]);
            const tap2: *Tensor = c.ggml_view_1d(ctx, taps, n_embd, 2 * taps.nb[1]);

            var y: *Tensor = c.ggml_mul(ctx, bx, tap1);
            y = c.ggml_add(ctx, y, c.ggml_mul(ctx, c.ggml_mul(ctx, prv, tap0), conv_left));
            y = c.ggml_add(ctx, y, c.ggml_mul(ctx, c.ggml_mul(ctx, nxt, tap2), conv_right));
            b.cb(y, "model.layers.{}.conv.conv", il);

            cur = b.buildLoraMm(layer.shortconv.out_proj.?, c.ggml_mul(ctx, cv, y), null);
            b.cb(cur, "model.layers.{}.conv.out_proj", il);
        } else {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, hp.nHeadKv(@intCast(il)), il);
            var q = qkv.q;
            var k = qkv.k;

            q = b.buildNorm(q, layer.attn_q_norm, null, .rms, il);
            k = b.buildNorm(k, layer.attn_k_norm, null, .rms, il);

            q = c.ggml_rope_ext(ctx, q, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            k = c.ggml_rope_ext(ctx, k, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            cur = b.buildAttn(inp_attn[0], layer.wo, null, layer.wo_s, q, k, qkv.v, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
            b.cb(cur, "model.layers.{}.self_attn.out_proj", il);
        }

        cur = c.ggml_add(ctx, cur, inpL);

        var ffn_out = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        ffn_out = b.buildFfn(ffn_out, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);

        cur = c.ggml_add(ctx, cur, ffn_out);
        b.cb(cur, "l_out", il);
    }

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    cur = buildDecisionHead(model, b, cur, inp_attn[1], inp_out_ids);

    b.res.t_embd = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_lfm2::graph_decision`'s `build_decision_head`
/// (models/lfm2.cpp:422 @8642541b2): ModernBERT's
/// (`modern_bert.zig`'s `buildDecisionHead`) with the head counts of the
/// head layers. The question type is not a graph input, so the head is
/// evaluated once per type: `[n_decision_types, n_outputs]`.
fn buildDecisionHead(model: *const Model, b: *const graph.Builder, inp: *Tensor, inp_attn: *graph.InpAttnNoCache, inp_out_ids: *Tensor) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const n_layer: c_int = @intCast(b.n_layer);
    const n_layer_enc: c_int = n_layer - @as(c_int, @intCast(hp.n_layer_decision));
    const type_embd = model.type_embd.?;

    var scores: ?*Tensor = null;

    var it: u32 = 0;
    while (it < n_decision_types) : (it += 1) {
        const type_row = c.ggml_view_1d(ctx, type_embd, b.n_embd, it * type_embd.nb[1]);
        var inpL: *Tensor = c.ggml_add(ctx, inp, type_row);

        var il: c_int = n_layer_enc;
        while (il < n_layer) : (il += 1) {
            const layer = &model.layers[@intCast(il)];

            var cur = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .norm, il);
            b.cb(cur, "attn_norm", il);

            // No positional encoding in the head.
            const qkv = b.buildQkv(layer, cur, n_embd_head, hp.nHead(@intCast(il)), hp.nHeadKv(@intCast(il)), il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, qkv.q, qkv.k, qkv.v, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
            b.cb(cur, "kqv_out", il);

            if (il == n_layer - 1) {
                cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
                inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
            }

            const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
            b.cb(ffn_inp, "ffn_inp", il);

            cur = b.buildNorm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, .norm, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, null, null, null, layer.ffn_down, layer.ffn_down_b, null, null, .relu, .seq, il);

            inpL = c.ggml_add(ctx, cur, ffn_inp);
        }

        // The scorer.
        var cur = b.buildNorm(inpL, model.cls_norm, model.cls_norm_b, .norm, -1);
        cur = c.ggml_add(ctx, b.buildLoraMm(model.cls.?, cur, null), model.cls_b);
        cur = c.ggml_gelu_erf(ctx, cur);
        cur = c.ggml_add(ctx, b.buildLoraMm(model.cls_out.?, cur, null), model.cls_out_b);

        scores = if (scores) |s| c.ggml_concat(ctx, s, cur, 0) else cur;
    }
    b.cb(scores.?, "decision_scores", -1);

    return scores.?;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
