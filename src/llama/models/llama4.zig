//! Llama 4: chunked attention with RoPE on three layers of four and a
//! temperature-tuned global layer without it, L2-normed Q and K, and a
//! sigmoid-routed mixture of experts with a shared expert on every
//! `n_moe_layer_step`-th layer.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/llama4.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_llama4`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph<iswa>`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_llama4`'s `load_arch_hparams` (src/models/llama4.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_moe_layer_step = (try meta.getKey(u32, .interleave_moe_layer_step, true, exc)).?;

    var found_swa = false;
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| {
        hp.n_swa = v;
        found_swa = true;
    }
    if (found_swa and hp.n_swa == 0) {
        hp.swa_type = @backingInt(llama.hparams.SwaType.none);
        hp.n_no_rope_layer_step = hp.nLayer(); // Always use RoPE.
    } else {
        hp.swa_type = @backingInt(llama.hparams.SwaType.chunked);
        hp.n_swa = 8192;
        hp.n_attn_temp_floor_scale = 8192;
        hp.f_attn_temp_scale = 0.1;
        hp.f_attn_temp_offset = 1.0;

        try self.loadSwaPattern(ml, 4, false, exc); // The pattern: three chunked, one full.

        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
        if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    }

    // A step of 0 needs a model without layers; the C divides by zero
    // there, which aarch64 answers with a quotient of 0.
    for (0..hp.n_layer_all) |il| {
        const step = hp.n_no_rope_layer_step;
        hp.rope_pattern[il] = @intFromBool(step == 0 or @rem(@as(u32, @intCast(il + 1)), step) != 0);
    }

    self.type = switch (hp.n_expert) {
        // MobileLLM, without experts.
        0 => switch (hp.n_embd) {
            2048 => .@"140m",
            4096 => .@"360m",
            6144 => .@"950m",
            else => .unknown,
        },
        16 => .@"17b_16e",
        128 => .@"17b_128e",
        else => .unknown,
    };

    hp.use_kq_norm = self.type != .@"17b_128e";
}

/// Ports `llama_model_llama4`'s `load_arch_tensors` (src/models/llama4.cpp:48 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;
    const dup = llama.loader.tensor_duplicated;

    if (l.n_expert == 0) {
        return exc.throw(.runtime_error, "{s} model cannot have zero experts", .{self.archName()});
    }
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
        const is_moe_layer = hp.n_moe_layer_step > 0 and @rem(@as(u32, @intCast(i + 1)), hp.n_moe_layer_step) == 0;

        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        const rope_dup: u32 = if (i != 0) dup else 0;
        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);

        if (is_moe_layer) {
            const n_ff_exp: i64 = hp.nFfExp(0);

            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

            // The shared expert.
            const n_ff_shexp = n_ff_exp;
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_shexp, l.n_embd }, 0, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_shexp }, 0, exc);
        } else {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        }
    }
}

/// Ports `llama_model_llama4`'s `build_arch_graph` (src/models/llama4.cpp:100 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    if (@as(llama.hparams.SwaType, @fromBackingInt(@intCast(model.hparams.swa_type))) == .none) {
        buildGraph(false, model, b);
    } else {
        buildGraph(true, model, b);
    }
}

/// Ports the constructor of `llama_model_llama4`'s `graph<iswa>`
/// (src/models/llama4.cpp:109 @b1b770d90).
fn buildGraph(comptime iswa: bool, model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/llama4.cpp", 112, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/llama4.cpp", 113, "n_embd_head == n_rot");

    var inpL: *llama.impl.llama_c.ggml_tensor = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    // Temperature tuning.
    const inp_attn_scale: ?*llama.impl.llama_c.ggml_tensor = b.buildInpAttnScale();

    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / stdlib.libm.sqrtfInt(n_embd_head) else hp.f_attention_scale;

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        var inpSA = inpL;

        // This overlaps with the SWA layers in current models, so the
        // per-layer RoPE base and scale may be superfluous.
        const use_rope = hp.n_no_rope_layer_step > 0 and
            @rem(@as(u32, @intCast(il + 1)), hp.n_no_rope_layer_step) != 0;

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            if (use_rope) {
                Qcur = b.buildRope(Qcur, inp_pos, il);
                Kcur = b.buildRope(Kcur, inp_pos, il);
            } else if (inp_attn_scale) |s| {
                Qcur = llama.impl.llama_c.ggml_mul(ctx, Qcur, s);
            }
            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            if (use_rope and hp.use_kq_norm) {
                // Llama4TextL2Norm.
                Qcur = llama.impl.llama_c.ggml_rms_norm(ctx, Qcur, hp.f_norm_rms_eps);
                Kcur = llama.impl.llama_c.ggml_rms_norm(ctx, Kcur, hp.f_norm_rms_eps);
                b.cb(Qcur, "Qcur_normed", il);
                b.cb(Kcur, "Kcur_normed", il);
            }
            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.llama_c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        if (layer.ffn_gate_inp == null) {
            // The feed-forward network, not a mixture of experts.
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            const ffn_inp_normed = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            // The C++ names `cur` here, not the normed input.
            b.cb(cur, "ffn_norm", il);

            const moe_out = b.buildMoeFfn(ffn_inp_normed, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = layer.ffn_gate_exps,
                .down_exps = layer.ffn_down_exps.?,
                .n_expert = b.n_expert,
                .n_expert_used = b.n_expert_used,
                .type_op = .silu,
                .norm_w = false,
                .w_scale = hp.expert_weights_scale,
                .gating_op = .sigmoid,
            }, il);

            // The shared experts.
            const shexp_out = b.buildFfn(ffn_inp_normed, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
            b.cb(shexp_out, "ffn_moe_shexp", il);

            cur = llama.impl.llama_c.ggml_add(ctx, moe_out, shexp_out);
            b.cb(cur, "ffn_moe_out_merged", il);
        }
        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
