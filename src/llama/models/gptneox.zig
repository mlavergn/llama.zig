//! GPT-NeoX: decoder blocks with LayerNorm and biases, partial RoPE, a fused
//! QKV projection and a GELU feed-forward, run either in parallel with the
//! attention or after it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/gptneox.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_gptneox`'s `load_arch_hparams`, `load_arch_tensors` and
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

/// Ports `llama_model_gptneox`'s `load_arch_hparams` (models/gptneox.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;
    hp.use_par_res = (try meta.getKey(bool, .use_parallel_residual, true, exc)).?;

    const n_ff = hp.nFf(0);
    self.type = switch (hp.nLayer()) {
        6 => switch (n_ff) {
            512 => .@"14m",
            2048 => .@"70m",
            else => .unknown,
        },
        12 => switch (n_ff) {
            3072 => .@"160m",
            else => .unknown,
        },
        16 => switch (n_ff) {
            8192 => .@"1b",
            else => .unknown,
        },
        24 => switch (n_ff) {
            4096 => .@"410m",
            8192 => .@"1_4b",
            else => .unknown,
        },
        32 => switch (n_ff) {
            10240 => .@"2_8b",
            16384 => .@"6_9b",
            else => .unknown,
        },
        36 => switch (n_ff) {
            20480 => .@"12b",
            else => .unknown,
        },
        44 => switch (n_ff) {
            24576 => .@"20b",
            else => .unknown,
        },
        else => .unknown,
    };
}

/// Ports `llama_model_gptneox`'s `load_arch_tensors` (models/gptneox.cpp:50 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output_norm_b = try self.createTensor(.{ .tensor = .output_norm, .suffix = "bias" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_b = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd + 2 * l.n_embd_gqa }, 0, exc);
        layer.wqkv_b = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "bias", .bid = i }, &.{l.n_embd + 2 * l.n_embd_gqa}, 0, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_norm_b = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{l.n_ff}, 0, exc);
    }
}

/// Ports `llama_model_gptneox`'s `build_arch_graph` (models/gptneox.cpp:83 @5266f24da)
/// and the constructor of its `graph` (models/gptneox.cpp:87 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/gptneox.cpp", 90, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];

        var cur = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .norm, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }

        // The feed-forward network.
        if (hp.use_par_res) {
            // Attention and feed-forward computed in parallel:
            // x = x + attn(ln1(x)) + ffn(ln2(x)).
            const attn_out = cur;

            cur = b.buildNorm(inpL, layer.ffn_norm, layer.ffn_norm_b, .norm, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, null, null, null, layer.ffn_down, layer.ffn_down_b, null, null, .gelu, .seq, il);
            b.cb(cur, "ffn_out", il);

            cur = c.ggml_add(ctx, cur, inpL);
            b.cb(cur, "ffn_out", il);

            cur = c.ggml_add(ctx, cur, attn_out);

            cur = b.buildCvec(cur, il);
            b.cb(cur, "l_out", il);

            // The next layer's input.
            inpL = cur;
        } else {
            // Attention and feed-forward computed in sequence:
            // x = x + attn(ln1(x)), then x = x + ffn(ln2(x)).
            const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
            b.cb(ffn_inp, "ffn_inp", il);

            cur = b.buildNorm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, .norm, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, null, null, null, layer.ffn_down, layer.ffn_down_b, null, null, .gelu, .seq, il);
            b.cb(cur, "ffn_out", il);

            cur = c.ggml_add(ctx, cur, ffn_inp);

            cur = b.buildCvec(cur, il);
            b.cb(cur, "l_out", il);

            // The next layer's input.
            inpL = cur;
        }
    }

    var cur = b.buildNorm(inpL, model.output_norm, model.output_norm_b, .norm, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

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
