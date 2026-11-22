//! Phi-3: pre-norm decoder blocks with RoPE (and LongRoPE factors for long
//! contexts), queries pre-scaled before attention, and a fused gate-and-up
//! SwiGLU feed-forward or a softmax-routed mixture of experts.
//!
//! `llama_model_phimoe` builds with `llama_model_phi3`'s `graph` and loads
//! with its own overrides.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/phi3.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_phi3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph<iswa>`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_phi3`'s `load_arch_hparams` (models/phi3.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        24 => .@"1b",
        32 => .@"3b",
        40 => .@"14b",
        else => .unknown,
    };

    var found_swa = false;
    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| {
        hp.n_swa = v;
        found_swa = true;
    }

    if (found_swa and hp.n_swa > 0) {
        llama.impl.logWarn("%s: Phi SWA is currently disabled - results might be suboptimal for some models (see %s)\n", .{ "load_arch_hparams", "https://github.com/ggml-org/llama.cpp/pull/13676" });

        // TODO: fix conversion scripts to correctly populate `n_swa` and
        // `n_swa_pattern`.
        hp.swa_type = @backingInt(llama.hparams.SwaType.none);

        hp.n_swa = 0;
        hp.setSwaPattern(1, false);
    }
}

/// Ports `llama_model_phi3`'s `load_arch_tensors` (models/phi3.cpp:27 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const nr = llama.loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, l.n_embd_gqa, l.n_embd_gqa, nr, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, 2 * l.n_ff }, 0, exc);

        const rope_dup: u32 = if (i != 0) llama.loader.tensor_duplicated else 0;
        layer.rope_long = try self.createTensor(.{ .tensor = .rope_factors_long, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
        layer.rope_short = try self.createTensor(.{ .tensor = .rope_factors_short, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
    }
}

/// Ports `llama_model_phi3`'s `build_arch_graph` (models/phi3.cpp:59 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    if (b.hparams.swa_type != @backingInt(llama.hparams.SwaType.none)) {
        buildGraph(true, model, b);
    } else {
        buildGraph(false, model, b);
    }
}

/// Ports the constructor of `llama_model_phi3`'s `graph<iswa>`
/// (models/phi3.cpp:68 @b1b770d90).
fn buildGraph(comptime iswa: bool, model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/phi3.cpp", 71, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = if (iswa) b.buildAttnInpKvIswa() else b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var residual = inpL;
        var cur: *llama.impl.llama_c.ggml_tensor = undefined;

        // Self-attention.
        {
            const attn_norm_output = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .rms, il);
            b.cb(attn_norm_output, "attn_norm", il);

            const qkv = b.buildQkv(layer, attn_norm_output, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            Qcur = llama.impl.llama_c.ggml_scale(ctx, Qcur, 1.0 / stdlib.libm.sqrtfInt(n_embd_head));
            b.cb(Qcur, "Qcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0, il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            residual = llama.impl.llama_c.ggml_get_rows(ctx, residual, inp_out_ids);
        }
        cur = llama.impl.llama_c.ggml_add(ctx, cur, residual);
        residual = cur;

        cur = b.buildNorm(cur, layer.ffn_norm, layer.ffn_norm_b, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // The feed-forward network.
        if (layer.ffn_gate_inp == null) {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, null, null, null, layer.ffn_down, null, null, null, .swiglu, .seq, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The mixture of experts.
            cur = b.buildMoeFfn(cur, .{
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
            b.cb(cur, "ffn_moe_out", il);
        }
        cur = llama.impl.llama_c.ggml_add(ctx, residual, cur);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = b.buildNorm(inpL, model.output_norm, model.output_norm_b, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    if (model.output_b) |ob| {
        b.cb(cur, "result_output_no_bias", -1);
        cur = llama.impl.llama_c.ggml_add(ctx, cur, ob);
    }
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
