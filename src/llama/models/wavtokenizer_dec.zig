//! WavTokenizer's decoder: audio-codec tokens to spectrogram features. A
//! 1-D convolution, a "posnet" of group-normed residual convolutions with
//! one self-attention block, then ConvNeXt blocks (a depthwise convolution
//! and a pointwise GELU feed-forward) and a linear head. No attention
//! cache; the output is an embedding, not logits.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/wavtokenizer-dec.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_wavtokenizer_dec`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`.

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

/// Ports `llama_model_wavtokenizer_dec`'s `load_arch_hparams` (models/wavtokenizer-dec.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;
    hp.f_norm_group_eps = (try meta.getKey(f32, .attention_groupnorm_eps, true, exc)).?;
    hp.n_norm_groups = (try meta.getKey(u32, .attention_groupnorm_groups, true, exc)).?;
}

/// Ports `llama_model_wavtokenizer_dec`'s `load_arch_tensors` (models/wavtokenizer-dec.cpp:9 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ hp.n_embd, l.n_vocab }, 0, exc);

    self.conv1d = try self.createTensor(.{ .tensor = .conv1d, .suffix = "weight", .bid = 0 }, &.{ 7, hp.n_embd, hp.posnet.n_embd }, 0, exc);
    self.conv1d_b = try self.createTensor(.{ .tensor = .conv1d, .suffix = "bias", .bid = 0 }, &.{ 1, hp.posnet.n_embd }, 0, exc);

    // The posnet.
    {
        const n_embd: i64 = hp.posnet.n_embd;

        var i: u32 = 0;
        while (i < hp.posnet.n_layer) : (i += 1) {
            const layer = &self.layers[i].posnet;
            const bid: i32 = @intCast(i);

            // The posnet is:
            //
            //  - resnet
            //  - resnet
            //  - attn
            //  - resnet
            //  - resnet
            //  - norm
            //
            switch (i) {
                0, 1, 3, 4 => {
                    layer.norm1 = try self.createTensor(.{ .tensor = .pos_net_norm1, .suffix = "weight", .bid = bid }, &.{ 1, n_embd }, 0, exc);
                    layer.norm1_b = try self.createTensor(.{ .tensor = .pos_net_norm1, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);

                    layer.conv1 = try self.createTensor(.{ .tensor = .pos_net_conv1, .suffix = "weight", .bid = bid }, &.{ 3, n_embd, n_embd }, 0, exc);
                    layer.conv1_b = try self.createTensor(.{ .tensor = .pos_net_conv1, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);

                    layer.norm2 = try self.createTensor(.{ .tensor = .pos_net_norm2, .suffix = "weight", .bid = bid }, &.{ 1, n_embd }, 0, exc);
                    layer.norm2_b = try self.createTensor(.{ .tensor = .pos_net_norm2, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);

                    layer.conv2 = try self.createTensor(.{ .tensor = .pos_net_conv2, .suffix = "weight", .bid = bid }, &.{ 3, n_embd, n_embd }, 0, exc);
                    layer.conv2_b = try self.createTensor(.{ .tensor = .pos_net_conv2, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);
                },
                2 => {
                    layer.attn_norm = try self.createTensor(.{ .tensor = .pos_net_attn_norm, .suffix = "weight", .bid = bid }, &.{ 1, n_embd }, 0, exc);
                    layer.attn_norm_b = try self.createTensor(.{ .tensor = .pos_net_attn_norm, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);

                    layer.attn_q = try self.createTensor(.{ .tensor = .pos_net_attn_q, .suffix = "weight", .bid = bid }, &.{ 1, n_embd, n_embd }, 0, exc);
                    layer.attn_q_b = try self.createTensor(.{ .tensor = .pos_net_attn_q, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);

                    layer.attn_k = try self.createTensor(.{ .tensor = .pos_net_attn_k, .suffix = "weight", .bid = bid }, &.{ 1, n_embd, n_embd }, 0, exc);
                    layer.attn_k_b = try self.createTensor(.{ .tensor = .pos_net_attn_k, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);

                    layer.attn_v = try self.createTensor(.{ .tensor = .pos_net_attn_v, .suffix = "weight", .bid = bid }, &.{ 1, n_embd, n_embd }, 0, exc);
                    layer.attn_v_b = try self.createTensor(.{ .tensor = .pos_net_attn_v, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);

                    layer.attn_o = try self.createTensor(.{ .tensor = .pos_net_attn_out, .suffix = "weight", .bid = bid }, &.{ 1, n_embd, n_embd }, 0, exc);
                    layer.attn_o_b = try self.createTensor(.{ .tensor = .pos_net_attn_out, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);
                },
                5 => {
                    layer.norm = try self.createTensor(.{ .tensor = .pos_net_attn_norm, .suffix = "weight", .bid = bid }, &.{ 1, n_embd }, 0, exc);
                    layer.norm_b = try self.createTensor(.{ .tensor = .pos_net_attn_norm, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);
                },
                else => impl.abort("models/wavtokenizer-dec.cpp", 73, "unknown posnet layer"),
            }
        }
    }

    impl.assert(hp.posnet.n_embd == hp.convnext.n_embd, "models/wavtokenizer-dec.cpp", 78, "hparams.posnet.n_embd == hparams.convnext.n_embd");

    self.tok_norm = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "weight", .bid = 0 }, &.{hp.posnet.n_embd}, 0, exc);
    self.tok_norm_b = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "bias", .bid = 0 }, &.{hp.posnet.n_embd}, 0, exc);

    // The ConvNeXt blocks.
    {
        const n_embd: i64 = hp.convnext.n_embd;

        var i: u32 = 0;
        while (i < hp.convnext.n_layer) : (i += 1) {
            const layer = &self.layers[i].convnext;
            const bid: i32 = @intCast(i);

            layer.dw = try self.createTensor(.{ .tensor = .convnext_dw, .suffix = "weight", .bid = bid }, &.{ 7, 1, n_embd }, 0, exc);
            layer.dw_b = try self.createTensor(.{ .tensor = .convnext_dw, .suffix = "bias", .bid = bid }, &.{ 1, n_embd }, 0, exc);

            layer.norm = try self.createTensor(.{ .tensor = .convnext_norm, .suffix = "weight", .bid = bid }, &.{n_embd}, 0, exc);
            layer.norm_b = try self.createTensor(.{ .tensor = .convnext_norm, .suffix = "bias", .bid = bid }, &.{n_embd}, 0, exc);

            layer.pw1 = try self.createTensor(.{ .tensor = .convnext_pw1, .suffix = "weight", .bid = bid }, &.{ n_embd, l.n_ff }, 0, exc);
            layer.pw1_b = try self.createTensor(.{ .tensor = .convnext_pw1, .suffix = "bias", .bid = bid }, &.{l.n_ff}, 0, exc);

            layer.pw2 = try self.createTensor(.{ .tensor = .convnext_pw2, .suffix = "weight", .bid = bid }, &.{ l.n_ff, n_embd }, 0, exc);
            layer.pw2_b = try self.createTensor(.{ .tensor = .convnext_pw2, .suffix = "bias", .bid = bid }, &.{n_embd}, 0, exc);

            layer.gamma = try self.createTensor(.{ .tensor = .convnext_gamma, .suffix = "weight", .bid = bid }, &.{n_embd}, 0, exc);
        }

        // Output.
        self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
        self.output_norm_b = try self.createTensor(.{ .tensor = .output_norm, .suffix = "bias" }, &.{n_embd}, 0, exc);
    }

    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ hp.convnext.n_embd, hp.nEmbdOut() }, 0, exc);
    self.output_b = try self.createTensor(.{ .tensor = .output, .suffix = "bias" }, &.{hp.nEmbdOut()}, 0, exc);
}

/// Ports `llama_model_wavtokenizer_dec`'s `build_arch_graph` (models/wavtokenizer-dec.cpp:114 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_wavtokenizer_dec`'s `graph`
/// (models/wavtokenizer-dec.cpp:118 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    var cur: *Tensor = c.ggml_cont(ctx, c.ggml_transpose(ctx, inpL));

    cur = c.ggml_conv_1d_ph(ctx, model.conv1d, cur, 1, 1);
    cur = c.ggml_add(ctx, cur, model.conv1d_b);

    // The posnet.
    var il: u32 = 0;
    while (il < hp.posnet.n_layer) : (il += 1) {
        const layer = &model.layers[il].posnet;

        inpL = cur;

        switch (il) {
            0, 1, 3, 4 => {
                cur = b.buildNorm(cur, layer.norm1, layer.norm1_b, .group, 0);

                cur = c.ggml_mul(ctx, c.ggml_sigmoid(ctx, cur), cur);

                cur = c.ggml_conv_1d_ph(ctx, layer.conv1, cur, 1, 1);
                cur = c.ggml_add(ctx, cur, layer.conv1_b);

                cur = b.buildNorm(cur, layer.norm2, layer.norm2_b, .group, 0);

                cur = c.ggml_mul(ctx, c.ggml_sigmoid(ctx, cur), cur);

                cur = c.ggml_conv_1d_ph(ctx, layer.conv2, cur, 1, 1);
                cur = c.ggml_add(ctx, cur, layer.conv2_b);

                cur = c.ggml_add(ctx, cur, inpL);
            },
            2 => {
                cur = b.buildNorm(cur, layer.attn_norm, layer.attn_norm_b, .group, 0);

                var q: *Tensor = c.ggml_conv_1d_ph(ctx, layer.attn_q, cur, 1, 1);
                var k: *Tensor = c.ggml_conv_1d_ph(ctx, layer.attn_k, cur, 1, 1);
                var v: *Tensor = c.ggml_conv_1d_ph(ctx, layer.attn_v, cur, 1, 1);

                q = c.ggml_add(ctx, q, layer.attn_q_b);
                k = c.ggml_add(ctx, k, layer.attn_k_b);
                v = c.ggml_add(ctx, v, layer.attn_v_b);

                q = c.ggml_cont(ctx, c.ggml_transpose(ctx, q));
                k = c.ggml_cont(ctx, c.ggml_transpose(ctx, k));

                var kq: *Tensor = c.ggml_mul_mat(ctx, k, q);

                kq = c.ggml_soft_max_ext(ctx, kq, null, 1.0 / impl.sqrtf(@floatFromInt(hp.posnet.n_embd)), 0.0);

                cur = c.ggml_mul_mat(ctx, kq, v);

                cur = c.ggml_conv_1d_ph(ctx, layer.attn_o, cur, 1, 1);
                cur = c.ggml_add(ctx, cur, layer.attn_o_b);

                cur = c.ggml_add(ctx, cur, inpL);
            },
            5 => {
                cur = b.buildNorm(cur, layer.norm, layer.norm_b, .group, 0);
            },
            else => impl.abort("models/wavtokenizer-dec.cpp", 203, "unknown posnet layer"),
        }
    }
    cur = c.ggml_cont(ctx, c.ggml_transpose(ctx, cur));

    cur = b.buildNorm(cur, model.tok_norm, model.tok_norm_b, .norm, 0);

    cur = c.ggml_cont(ctx, c.ggml_transpose(ctx, cur));

    inpL = cur;

    // The ConvNeXt blocks.
    il = 0;
    while (il < hp.convnext.n_layer) : (il += 1) {
        const layer = &model.layers[il].convnext;

        cur = inpL;

        cur = c.ggml_conv_1d_dw_ph(ctx, layer.dw, cur, 1, 1);
        cur = c.ggml_add(ctx, cur, layer.dw_b);

        cur = c.ggml_cont(ctx, c.ggml_transpose(ctx, cur));

        cur = b.buildNorm(cur, layer.norm, layer.norm_b, .norm, -1);

        cur = b.buildFfn(cur, layer.pw1, layer.pw1_b, null, null, null, null, layer.pw2, layer.pw2_b, null, null, .gelu, .seq, @intCast(il));

        cur = c.ggml_mul(ctx, cur, layer.gamma);

        cur = c.ggml_cont(ctx, c.ggml_transpose(ctx, cur));

        inpL = c.ggml_add(ctx, cur, inpL);
    }
    cur = inpL;

    cur = c.ggml_cont(ctx, c.ggml_transpose(ctx, cur));

    cur = b.buildNorm(cur, model.output_norm, model.output_norm_b, .norm, -1);

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    cur = c.ggml_add(ctx, cur, model.output_b);

    b.cb(cur, "result_embd", -1);
    b.res.t_embd = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
