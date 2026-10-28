//! The Qwen3-TTS speaker encoder (ECAPA-TDNN): a mel spectrogram to one
//! speaker embedding, through a TDNN frontend, three SE-Res2Net blocks,
//! multi-layer feature aggregation and attentive statistics pooling. This
//! file's convention is `[C, T]`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/qwen3tts-spkenc.cpp` at v0.4.0
//! (`5266f24da`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;
const gpa = impl.gpa;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

/// `SPK_RES2NET_SCALE` (qwen3tts-spkenc.cpp:3 @5266f24da): enc_res2net_scale.
const spk_res2net_scale: c_int = 8;
/// `SPK_DILATIONS` (qwen3tts-spkenc.cpp:4 @5266f24da): enc_dilations[1..3].
const spk_dilations = [3]c_int{ 2, 3, 4 };

/// Ports `conv1d_same` (qwen3tts-spkenc.cpp:8 @5266f24da), a member of
/// `clip_graph_qwen3tts_spkenc`: kernel K, "same" padding (reflect),
/// dilation d; x `[C, T]` to `[out_c, T]`.
fn conv1dSame(g: *const graph.Graph, x: T, w: T, b: T, dilation: c_int) T {
    const ctx0 = g.ctx0;
    const K: c_int = @intCast(ne(w, 0));
    const IC: c_int = @intCast(ne(w, 1));
    const OC: c_int = @intCast(ne(w, 2));
    const pad = @divTrunc((K - 1) * dilation, 2);

    // ggml_pad_reflect_1d pads ne[0], so bring T onto ne[0] first, same layout as im2col wants
    var x_t = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x)); // [T, IC]
    if (pad > 0) {
        x_t = c.ggml_pad_reflect_1d(ctx0, x_t, pad, pad); // [T + 2*pad, IC]
    }
    const x4d = c.ggml_reshape_4d(ctx0, x_t, ne(x_t, 0), IC, 1, 1);

    // dummy F32 kernel, im2col only reads its shape, so a quantized w does not assert
    const dummy = c.ggml_new_tensor_4d(ctx0, c.GGML_TYPE_F32, K, IC, 1, 1);

    var col = c.ggml_im2col(ctx0, dummy, x4d, 1, 1, 0, 0, dilation, 1, false, c.GGML_TYPE_F32);
    const T_out = ne(col, 1);
    col = c.ggml_reshape_2d(ctx0, col, @as(i64, K) * IC, T_out);

    const w2d = c.ggml_reshape_2d(ctx0, w, @as(i64, K) * IC, OC);
    var y = c.ggml_mul_mat(ctx0, w2d, col); // [OC, T_out]
    c.ggml_mul_mat_set_prec(y, c.GGML_PREC_F32);

    const b2d = c.ggml_reshape_2d(ctx0, b, OC, 1);
    y = c.ggml_add(ctx0, y, b2d);
    return y;
}

/// Ports `res2net` (qwen3tts-spkenc.cpp:39 @5266f24da), a member of
/// `clip_graph_qwen3tts_spkenc`: split the channel axis into `scale` chunks
/// and chain dilated conv1d branches; x `[C, T]` to `[C, T]`.
fn res2net(g: *const graph.Graph, x: T, layer: *const model_mod.Layer, dilation: c_int, scale: c_int) T {
    const ctx0 = g.ctx0;
    const C = ne(x, 0);
    const Tn = ne(x, 1);
    const Cs = @divTrunc(C, scale);

    var outs: std.ArrayList(T) = .empty;
    defer outs.deinit(gpa);
    outs.ensureTotalCapacity(gpa, @intCast(scale)) catch oom();

    var prev: T = null;
    var i: c_int = 0;
    while (i < scale) : (i += 1) {
        // `c` upstream, which names the C namespace here
        const ch = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, x, Cs, Tn, nb(x, 1), @as(usize, @intCast(i)) * @as(usize, @intCast(Cs)) * nb(x, 0)));
        if (i == 0) {
            outs.appendAssumeCapacity(ch);
            continue;
        }
        const inp = if (i >= 2) c.ggml_add(ctx0, ch, prev) else ch;
        const k: usize = @intCast(i - 1);
        var y = conv1dSame(g, inp, layer.res2_conv_w.items[k], layer.res2_conv_b.items[k], dilation);
        y = c.ggml_relu(ctx0, y);
        outs.appendAssumeCapacity(y);
        prev = y;
    }

    var acc = outs.items[0];
    for (outs.items[1..]) |o| {
        acc = c.ggml_concat(ctx0, acc, o, 0);
    }
    return acc;
}

/// Ports `se_block` (qwen3tts-spkenc.cpp:73 @5266f24da), a member of
/// `clip_graph_qwen3tts_spkenc`: squeeze-and-excitation gate; x `[C, T]` to
/// `[C, T]`.
fn seBlock(g: *const graph.Graph, x: T, layer: *const model_mod.Layer) T {
    const ctx0 = g.ctx0;
    // temporal mean, keepdim: transpose so T is on ne[0], reduce, transpose back
    const x_t = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x)); // [T, C]
    var mean = c.ggml_mean(ctx0, x_t); // [1, C]
    mean = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, mean)); // [C, 1]

    var h = conv1dSame(g, mean, layer.se_conv1_w, layer.se_conv1_b, 1);
    h = c.ggml_relu(ctx0, h);
    h = conv1dSame(g, h, layer.se_conv2_w, layer.se_conv2_b, 1);
    h = c.ggml_sigmoid(ctx0, h); // [C, 1]

    return c.ggml_mul(ctx0, x, h); // broadcast gate over T
}

/// Ports `se_res2net_block` (qwen3tts-spkenc.cpp:88 @5266f24da), a member
/// of `clip_graph_qwen3tts_spkenc`: tdnn1, res2net, tdnn2, se, plus the
/// residual; x `[C, T]` to `[C, T]`.
fn seRes2netBlock(g: *const graph.Graph, x: T, layer: *const model_mod.Layer, dilation: c_int, scale: c_int) T {
    const ctx0 = g.ctx0;
    const residual = x;
    var h = conv1dSame(g, x, layer.conv_pw1_w, layer.conv_pw1_b, 1); // tdnn1
    h = c.ggml_relu(ctx0, h);
    h = res2net(g, h, layer, dilation, scale);
    h = conv1dSame(g, h, layer.conv_pw2_w, layer.conv_pw2_b, 1); // tdnn2
    h = c.ggml_relu(ctx0, h);
    h = seBlock(g, h, layer);
    return c.ggml_add(ctx0, h, residual);
}

/// Ports `attentive_stats_pool` (qwen3tts-spkenc.cpp:100 @5266f24da), a
/// member of `clip_graph_qwen3tts_spkenc`: x `[C, T]` to `[2*C, 1]`.
fn attentiveStatsPool(g: *const graph.Graph, x: T) T {
    const ctx0 = g.ctx0;
    const model = g.model;
    const Tn = ne(x, 1);

    // mean over T: [C, 1]
    const x_t = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x));
    var mean = c.ggml_mean(ctx0, x_t);
    mean = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, mean));

    // std over T: sqrt(clamp(mean((x - mean)^2), eps))
    const mean_rep = c.ggml_repeat(ctx0, mean, x);
    const centered = c.ggml_sub(ctx0, x, mean_rep);
    const var_t = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, c.ggml_sqr(ctx0, centered)));
    var variance = c.ggml_mean(ctx0, var_t);
    variance = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, variance));
    variance = c.ggml_scale_bias(ctx0, variance, 1.0, 1e-12);
    const std_ = c.ggml_sqrt(ctx0, variance);

    // attention input: cat([x, mean, std]) along channel axis -> [3C, T]
    const std_rep = c.ggml_repeat(ctx0, std_, x);
    var cat = c.ggml_concat(ctx0, x, mean_rep, 0);
    cat = c.ggml_concat(ctx0, cat, std_rep, 0);

    // attention TDNN (3C -> attn_c) + ReLU, tanh, then 1x1 conv (attn_c -> C)
    var a = conv1dSame(g, cat, model.spk_asp_tdnn_w, model.spk_asp_tdnn_b, 1);
    a = c.ggml_relu(ctx0, a);
    a = c.ggml_tanh(ctx0, a);
    a = conv1dSame(g, a, model.spk_asp_attn_w, model.spk_asp_attn_b, 1);

    // softmax over T
    const a_t = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, a)); // [T, C]
    const w_t = c.ggml_soft_max(ctx0, a_t);
    const w = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, w_t)); // [C, T]

    // weighted mean: sum(w * x) over T, multiply by T to undo ggml_mean's 1/T scaling
    const wx = c.ggml_mul(ctx0, w, x);
    const wx_t = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, wx));
    var w_mean = c.ggml_mean(ctx0, wx_t);
    w_mean = c.ggml_scale(ctx0, w_mean, @floatFromInt(Tn));
    w_mean = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, w_mean)); // [C, 1]

    // weighted std: sum(w * (x - w_mean)^2) over T
    const w_mean_rep = c.ggml_repeat(ctx0, w_mean, x);
    const dev = c.ggml_sub(ctx0, x, w_mean_rep);
    const w_var_in = c.ggml_mul(ctx0, w, c.ggml_sqr(ctx0, dev));
    const w_var_t = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, w_var_in));
    var w_var = c.ggml_mean(ctx0, w_var_t);
    w_var = c.ggml_scale(ctx0, w_var, @floatFromInt(Tn));
    w_var = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, w_var));
    w_var = c.ggml_scale_bias(ctx0, w_var, 1.0, 1e-12);
    const w_std = c.ggml_sqrt(ctx0, w_var);

    return c.ggml_concat(ctx0, w_mean, w_std, 0); // [2C, 1]
}

/// Ports `build` (qwen3tts-spkenc.cpp:154 @5266f24da), a member of
/// `clip_graph_qwen3tts_spkenc`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;

    // inp_raw: [T, n_mel, 1, 1], from mtmd_audio_preprocessor_qwen3tts_spk
    var inp = g.buildInpRaw(1);
    inp = c.ggml_reshape_2d(ctx0, inp, ne(inp, 0), ne(inp, 1));

    // this file's convention is [C, T]; the preprocessor delivers [T, C]
    const mel = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp)); // [n_mel, T]
    g.cb(mel, "mel", -1);

    // frontend conv0 TDNN k=5, dilation=1: 128 -> 512
    var cur = conv1dSame(g, mel, model.conv1d_1_w, model.conv1d_1_b, 1);
    cur = c.ggml_relu(ctx0, cur);
    g.cb(cur, "frontend", -1);

    // 3 SE-Res2Net blocks at dilations 2, 3, 4
    impl.ggmlAssert(model.layers.items.len == 3, "qwen3tts-spkenc.cpp", 169, "(int) model.layers.size() == 3");
    var blk_out: [3]T = .{ null, null, null };
    for (0..3) |il| {
        cur = seRes2netBlock(g, cur, &model.layers.items[il], spk_dilations[il], spk_res2net_scale);
        blk_out[il] = cur;
        g.cb(cur, "block_out", @intCast(il));
    }

    // multi-layer feature aggregation: cat blk[0..2] then TDNN k=1 + ReLU
    var cat = c.ggml_concat(ctx0, blk_out[0], blk_out[1], 0);
    cat = c.ggml_concat(ctx0, cat, blk_out[2], 0); // [1536, T]
    var mfa = conv1dSame(g, cat, model.conv_out_w, model.conv_out_b, 1);
    mfa = c.ggml_relu(ctx0, mfa);
    g.cb(mfa, "mfa", -1);

    // attentive statistics pooling: [1536, T] -> [3072, 1]
    const stats = attentiveStatsPool(g, mfa);
    g.cb(stats, "asp", -1);

    // final FC k=1: [3072, 1] -> [enc_dim, 1]
    var emb = conv1dSame(g, stats, model.mm_fc_w, model.mm_fc_b, 1);

    emb = c.ggml_reshape_1d(ctx0, emb, ne(emb, 0));
    emb = c.ggml_cont(ctx0, emb);
    g.cb(emb, "spk_embedding", -1);

    c.ggml_build_forward_expand(g.gf, emb);
    return g.gf;
}

fn oom() noreturn {
    impl.abort("qwen3tts-spkenc.cpp", 0, "out of memory");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
