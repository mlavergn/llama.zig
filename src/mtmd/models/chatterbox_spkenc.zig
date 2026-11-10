//! The Chatterbox speaker encoder: a CAMPPlus x-vector over kaldi fbank
//! features (s3gen/xvector.py), through a 2-d FCM front, a TDNN layer,
//! three CAM dense blocks and statistics pooling. This file's convention is
//! `[C, T]`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/chatterbox-spkenc.cpp` at v0.9.0
//! (`cd959bae9`), and the `clip_graph_chatterbox_spkenc` declaration in
//! `llama.cpp/tools/mtmd/models/models.h`. Each declaration names what it
//! ports and the line it began at. The base class's builders are
//! `chatterbox_base.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const base = @import("chatterbox_base.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;
const Cbx = model_mod.Chatterbox;

/// Ports `bn_eps` (chatterbox-spkenc.cpp:5 @cd959bae9).
const bn_eps: f32 = 1e-5;

/// Ports `bn1d` (chatterbox-spkenc.cpp:8 @cd959bae9), a member of
/// `clip_graph_chatterbox_spkenc`: per-channel batchnorm on x `[C, T]`;
/// `w`/`b` are null on the affine=False variant.
fn bn1d(g: *const graph.Graph, n: *const Cbx.Bn, x: T) T {
    const ctx0 = g.ctx0;
    const sd = c.ggml_sqrt(ctx0, c.ggml_scale_bias(ctx0, n.@"var", 1.0, bn_eps));
    if (n.w == null) {
        return c.ggml_div(ctx0, c.ggml_sub(ctx0, x, n.mean), sd);
    }
    const a = c.ggml_div(ctx0, n.w, sd);
    const shift = c.ggml_sub(ctx0, n.b, c.ggml_mul(ctx0, n.mean, a));
    return c.ggml_add(ctx0, c.ggml_mul(ctx0, x, a), shift);
}

/// Ports `cbx_bn2d` (chatterbox-spkenc.cpp:19 @cd959bae9): batchnorm on a
/// conv2d activation `[W=T, H=F, C, 1]`, stats on ne2.
fn cbxBn2d(ctx0: ?*c.ggml_context, n: *const Cbx.Bn, x: T) T {
    const C = ne(x, 2);
    const mean = c.ggml_reshape_4d(ctx0, n.mean, 1, 1, C, 1);
    const variance = c.ggml_reshape_4d(ctx0, n.@"var", 1, 1, C, 1);
    const w = c.ggml_reshape_4d(ctx0, n.w, 1, 1, C, 1);
    const b = c.ggml_reshape_4d(ctx0, n.b, 1, 1, C, 1);
    const a = c.ggml_div(ctx0, w, c.ggml_sqrt(ctx0, c.ggml_scale_bias(ctx0, variance, 1.0, bn_eps)));
    const shift = c.ggml_sub(ctx0, b, c.ggml_mul(ctx0, mean, a));
    return c.ggml_add(ctx0, c.ggml_mul(ctx0, x, a), shift);
}

/// Ports `bn2d_relu` (chatterbox-spkenc.cpp:30 @cd959bae9), a member of
/// `clip_graph_chatterbox_spkenc`.
fn bn2dRelu(g: *const graph.Graph, n: *const Cbx.Bn, x: T) T {
    return c.ggml_relu(g.ctx0, cbxBn2d(g.ctx0, n, x));
}

/// Ports `res2d` (chatterbox-spkenc.cpp:35 @cd959bae9), a member of
/// `clip_graph_chatterbox_spkenc`: the FCM residual 2-d block, stride on
/// the frequency axis only.
fn res2d(g: *const graph.Graph, r: *const Cbx.SpkRes2d, x: T, stride: c_int) T {
    const ctx0 = g.ctx0;
    var cur = c.ggml_conv_2d(ctx0, r.conv1_w, x, 1, stride, 1, 1, 1, 1);
    cur = bn2dRelu(g, &r.bn1, cur);
    cur = c.ggml_conv_2d(ctx0, r.conv2_w, cur, 1, 1, 1, 1, 1, 1);
    // bn2 without the relu, applied before the residual add
    cur = cbxBn2d(ctx0, &r.bn2, cur);
    var res = x;
    if (r.shortcut_w != null) {
        res = c.ggml_conv_2d(ctx0, r.shortcut_w, x, 1, stride, 0, 0, 1, 1);
        res = cbxBn2d(ctx0, &r.shortcut_bn, res);
    }
    return c.ggml_relu(ctx0, c.ggml_add(ctx0, cur, res));
}

/// Ports `cam_layer` (chatterbox-spkenc.cpp:51 @cd959bae9), a member of
/// `clip_graph_chatterbox_spkenc`: a CAM dense TDNN layer, a bottleneck
/// then a context-gated conv; x `[C_in, T]` to `[growth, T]`.
fn camLayer(g: *const graph.Graph, l: *const Cbx.SpkCamLayer, x: T, dil: c_int, segfix: T) T {
    const ctx0 = g.ctx0;
    var h = c.ggml_relu(ctx0, bn1d(g, &l.nl1_bn, x));
    h = base.cbxConv1d(g, l.linear1_w, null, h, 1, 0, 0);
    h = c.ggml_relu(ctx0, bn1d(g, &l.nl2_bn, h));

    const k = l.local_w;
    const pad: c_int = @divTrunc(@as(c_int, @intCast(ne(k, 0))) - 1, 2) * dil;
    const y = base.cbxConv1dDil(g, k, null, h, pad, dil);

    // context: global mean plus ceil-mode segment means of length 100
    const Tn: c_int = @intCast(ne(h, 1));
    const C: c_int = @intCast(ne(h, 0));
    const S: c_int = @divTrunc(Tn + 99, 100);
    const ht = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, h)); // [T, C]
    const gmean = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, c.ggml_mean(ctx0, ht))); // [C, 1]
    var seg: T = undefined;
    {
        var padded = ht;
        if (S * 100 != Tn) {
            var z = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, S * 100 - Tn, C);
            z = c.ggml_scale(ctx0, z, 0.0);
            padded = c.ggml_concat(ctx0, ht, z, 0);
        }
        var pooled = c.ggml_pool_1d(ctx0, padded, c.GGML_OP_POOL_AVG, 100, 100, 0); // [S, C]
        pooled = c.ggml_mul(ctx0, pooled, segfix);
        var exp = c.ggml_interpolate(ctx0, pooled, S * 100, C, 1, 1, c.GGML_SCALE_MODE_NEAREST);
        exp = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, exp, Tn, C, nb(exp, 1), 0));
        seg = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, exp)); // [C, T]
    }
    var context = c.ggml_add(ctx0, seg, gmean);
    context = base.cbxConv1d(g, l.ctx1_w, l.ctx1_b, context, 1, 0, 0);
    context = c.ggml_relu(ctx0, context);
    context = base.cbxConv1d(g, l.ctx2_w, l.ctx2_b, context, 1, 0, 0);
    const m = c.ggml_sigmoid(ctx0, context);
    return c.ggml_mul(ctx0, y, m);
}

/// Ports `block_dil` (chatterbox-spkenc.cpp:121 @cd959bae9).
const block_dil = [3]c_int{ 1, 2, 2 };

/// Ports `build` (chatterbox-spkenc.cpp:89 @cd959bae9), a member of
/// `clip_graph_chatterbox_spkenc`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const cbx = &g.model.cbx;

    const Tn: c_int = g.img.nx();
    const T1: c_int = @divTrunc(Tn - 1, 2) + 1;
    const S: c_int = @divTrunc(T1 + 99, 100);
    const segfix = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, S);
    _ = c.ggml_set_name(segfix, "inp_segfix");
    c.ggml_set_input(segfix);

    // fbank features [T, 80] from the preprocessor
    const inp = g.buildInpRaw(1);

    // fcm 2d front: [W=T, H=F=80, C=1] -> [T, 10, 32] -> [320, T]
    var x = c.ggml_reshape_4d(ctx0, inp, Tn, 80, 1, 1);
    x = c.ggml_conv_2d(ctx0, cbx.spk_conv1_w, x, 1, 1, 1, 1, 1, 1);
    x = bn2dRelu(g, &cbx.spk_bn1, x);
    x = res2d(g, &cbx.spk_layer1_0, x, 2);
    x = res2d(g, &cbx.spk_layer1_1, x, 1);
    x = res2d(g, &cbx.spk_layer2_0, x, 2);
    x = res2d(g, &cbx.spk_layer2_1, x, 1);
    x = c.ggml_conv_2d(ctx0, cbx.spk_conv2_w, x, 1, 2, 1, 1, 1, 1);
    x = bn2dRelu(g, &cbx.spk_bn2, x);
    x = c.ggml_reshape_2d(ctx0, x, Tn, 320);
    x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x)); // [320, T]
    g.cb(x, "spk_fcm", -1);

    // tdnn k5 stride 2 over time, then the three cam dense blocks
    x = base.cbxConv1d(g, cbx.spk_tdnn_w, null, x, 2, 2, 2); // [128, T1]
    x = c.ggml_relu(ctx0, bn1d(g, &cbx.spk_tdnn_bn, x));
    g.cb(x, "spk_tdnn", -1);

    const blocks = [3]*const Cbx.SpkCamBlock{ &cbx.spk_block1, &cbx.spk_block2, &cbx.spk_block3 };
    var bi: c_int = 1;
    while (bi <= 3) : (bi += 1) {
        const blk = blocks[@intCast(bi - 1)];
        for (blk.layers.items) |*l| {
            const out = camLayer(g, l, x, block_dil[@intCast(bi - 1)], segfix);
            x = c.ggml_concat(ctx0, x, out, 0);
        }
        x = c.ggml_relu(ctx0, bn1d(g, &blk.transit_bn, x));
        x = base.cbxConv1d(g, blk.transit_w, null, x, 1, 0, 0);
        g.cb(x, "spk_block", bi);
    }
    x = c.ggml_relu(ctx0, bn1d(g, &cbx.spk_out_bn, x)); // [512, T1]

    // statistics pooling: mean and unbiased std over time -> [1024, 1]
    const xt = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x)); // [T1, 512]
    const mean = c.ggml_mean(ctx0, xt); // [1, 512]
    const m2 = c.ggml_mean(ctx0, c.ggml_mul(ctx0, xt, xt));
    var variance = c.ggml_sub(ctx0, m2, c.ggml_mul(ctx0, mean, mean));
    variance = c.ggml_scale(ctx0, variance, @as(f32, @floatFromInt(T1)) / @as(f32, @floatFromInt(T1 - 1)));
    const sd = c.ggml_sqrt(ctx0, c.ggml_relu(ctx0, variance));
    const stats = c.ggml_concat(
        ctx0,
        c.ggml_cont(ctx0, c.ggml_transpose(ctx0, mean)),
        c.ggml_cont(ctx0, c.ggml_transpose(ctx0, sd)),
        0,
    ); // [1024, 1]
    g.cb(stats, "spk_stats_pool", -1);

    // the x-vector stays raw, the gen graph normalizes it
    const dw = c.ggml_reshape_2d(ctx0, cbx.spk_dense_w, 1024, 192);
    var emb = c.ggml_mul_mat(ctx0, dw, stats); // [192, 1]
    emb = bn1d(g, &cbx.spk_dense_bn, emb);
    emb = c.ggml_cont(ctx0, c.ggml_reshape_1d(ctx0, emb, 192));
    g.cb(emb, "spk_embd", -1);
    _ = c.ggml_set_name(emb, "out_xvec");
    c.ggml_set_output(emb);
    c.ggml_build_forward_expand(g.gf, emb);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
