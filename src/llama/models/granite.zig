//! Granite: Llama's decoder blocks with scaled embeddings, residuals,
//! attention and logits, RoPE switched by `rope_finetuned`, an optional
//! mixture of experts with a shared expert, and Granite Vision 4.1's
//! deepstack inputs added into the residual stream.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/granite.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_model_granite`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph` with `build_attention_layer` and
//! `build_layer_ffn`. `llama_model_granite_moe` and `llama_model_minicpm`
//! build with this `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const libcxx = @import("../libcxx.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

fn oom() noreturn {
    impl.abort("models/granite.cpp", 0, "out of memory");
}

/// Ports `llama_model_granite`'s `load_arch_hparams` (models/granite.cpp:5 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_logit_scale = (try meta.getKey(f32, .logit_scale, true, exc)).?;
    if (try meta.getKey(f32, .residual_scale, false, exc)) |v| hp.f_residual_scale = v;
    if (try meta.getKey(f32, .embedding_scale, false, exc)) |v| hp.f_embedding_scale = v;
    if (try meta.getKey(f32, .attention_scale, false, exc)) |v| hp.f_attention_scale = v;

    // Granite4 Vision uses an array deepstack_mapping.
    _ = try meta.getArr(i32, .deepstack_mapping, &hp.deepstack_mapping_arr, false, exc);

    // Count the unique deepstack input indices.
    var unique_deepstack_idxs = libcxx.U32Set.init(impl.allocator) catch oom();
    defer unique_deepstack_idxs.deinit();
    for (hp.deepstack_mapping_arr) |val| {
        if (val >= 0) {
            _ = unique_deepstack_idxs.insert(@bitCast(val)) catch oom();
        }
    }
    hp.n_deepstack_layers = @intCast(unique_deepstack_idxs.count());

    // Ensure all values are valid (avoid overflow attacks), in the set's
    // own order, which decides the one reported.
    var it = unique_deepstack_idxs.iterator();
    while (it.next()) |val| {
        if (val > hp.n_deepstack_layers) {
            return exc.throw(.runtime_error, "Invalid deepstack index: {d} > {d}", .{ val, hp.n_deepstack_layers });
        }
    }

    // Granite uses rope_finetuned as a switch for RoPE, so it defaults to
    // true.
    const rope_finetuned = (try meta.getKey(bool, .rope_scaling_finetuned, false, exc)) orelse true;
    hp.rope_finetuned = rope_finetuned; // Needed for a round-trip save.
    hp.rope_pattern = @splat(@intFromBool(rope_finetuned));

    self.type = switch (hp.nLayer()) {
        32 => .@"3b",
        40 => switch (hp.n_embd) {
            2048 => .@"2b",
            2560 => .@"3b",
            4096 => .@"8b",
            else => .unknown,
        },
        64 => .@"30b",
        // Add additional layer/vocab/etc checks here for other model sizes.
        else => .unknown,
    };

    // Granite MoE Shared.
    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;
}

/// Ports `llama_model_granite`'s `load_arch_tensors` (models/granite.cpp:59 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

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

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        // Optional bias tensors.
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        const rope_dup: u32 = if (i != 0) dup else 0;
        if (hp.rope_scaling_type_train == c.LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long = try self.createTensor(.{ .tensor = .rope_factors_long, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
            layer.rope_short = try self.createTensor(.{ .tensor = .rope_factors_short, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
        } else {
            layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
        }

        if (l.n_expert == 0) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

            // Optional MLP bias.
            layer.ffn_gate_b = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "bias", .bid = i }, &.{l.n_ff}, nr, exc);
            layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
            layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{l.n_ff}, nr, exc);
        } else {
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, nr, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, 0, exc);

            // Granite MoE Shared.
            if (hp.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, 0, exc);
                layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, 0, exc);
                layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ hp.n_ff_shexp, l.n_embd }, 0, exc);
            }
        }
    }
}

/// Ports `llama_model_granite`'s `build_arch_graph` (models/granite.cpp:119 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_granite`'s `graph`
/// (models/granite.cpp:123 @b29c606e2), which Granite MoE and MiniCPM
/// build with too.
///
/// Parameters:
/// - `model`: the model.
/// - `b`: the graph being built.
///
/// Return: nothing; the graph is `b.gf`.
pub fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/granite.cpp", 130, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/granite.cpp", 131, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions, built only when RoPE is enabled.
    var inp_pos: ?*Tensor = null;
    if (hp.hasRope(0)) {
        inp_pos = b.buildInpPos();
    }
    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        // Granite Vision 4.1 deepstack: inject the projector stream that
        // targets decoder layer `il` before the decoder runs. The first
        // deepstack layer is skipped, since that is `inpL`.
        const deepstack_emb_idx = hp.deepstack_mapping_arr[@intCast(il)];
        if (il > 0 and deepstack_emb_idx >= 0) {
            const inp_embd: *Tensor = b.res.t_inp_embd.?;
            const ds: *Tensor = c.ggml_view_2d(ctx, inp_embd, b.n_embd, b.n_tokens, inp_embd.nb[1], @as(usize, @intCast(deepstack_emb_idx)) * @as(usize, @intCast(b.n_embd)) * @sizeOf(f32));
            inpL = c.ggml_add(ctx, inpL, ds);
            b.cb(inpL, "deepstack_in", il);
        }

        var inpSA = inpL;

        // Norm.
        var cur = b.buildNorm(inpL, model.layers[@intCast(il)].attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        cur = buildAttentionLayer(b, cur, inp_pos, inp_attn, model, n_embd_head, il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        // The feed-forward network.
        cur = buildLayerFfn(b, cur, inpSA, model, il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    // Granite scales the logits.
    cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_logit_scale);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_granite`'s `graph`'s `build_attention_layer`
/// (models/granite.cpp:205 @b29c606e2).
fn buildAttentionLayer(
    b: *const graph.Builder,
    cur_: *Tensor,
    inp_pos: ?*Tensor,
    inp_attn: *graph.InpAttnKv,
    model: *const Model,
    n_embd_head: i64,
    il: c_int,
) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ilu: u32 = @intCast(il);
    const layer = &model.layers[ilu];

    const qkv = b.buildQkv(layer, cur_, n_embd_head, hp.nHead(ilu), hp.nHeadKv(ilu), il);
    var Qcur = qkv.q;
    var Kcur = qkv.k;
    const Vcur = qkv.v;

    if (hp.hasRope(ilu)) {
        const rope_factors = model.getRopeFactors(b.cparams, ilu);
        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    }

    b.cb(Qcur, "Qcur", il);
    b.cb(Kcur, "Kcur", il);
    b.cb(Vcur, "Vcur", il);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;
    const cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(cur, "attn_out", il);
    return cur;
}

/// Ports `llama_model_granite`'s `graph`'s `build_layer_ffn`
/// (models/granite.cpp:243 @b29c606e2).
fn buildLayerFfn(b: *const graph.Builder, cur_: *Tensor, inpSA: *Tensor, model: *const Model, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    var cur = cur_;

    // Granite scales the residual.
    if (hp.f_residual_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, hp.f_residual_scale);
    }
    const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "ffn_inp", il);

    if (layer.ffn_gate_inp == null) {
        // The feed-forward network, not a mixture of experts.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);
    } else {
        // The mixture of experts.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        const moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .silu,
            .norm_w = true,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .softmax,
        }, il);
        b.cb(moe_out, "ffn_moe_out", il);

        // Granite MoE Shared.
        if (hp.n_ff_shexp > 0) {
            const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
            b.cb(ffn_shexp, "ffn_shexp", il);

            cur = c.ggml_add(ctx, moe_out, ffn_shexp);
            b.cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    }

    // Granite scales the residual.
    if (hp.f_residual_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, hp.f_residual_scale);
    }
    cur = c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "ffn_out", il);

    cur = b.buildCvec(cur, il);
    b.cb(cur, "l_out", il);

    return cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
