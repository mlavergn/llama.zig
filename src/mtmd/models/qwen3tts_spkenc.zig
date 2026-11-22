//! The Qwen3-TTS speaker encoder (ECAPA-TDNN): a mel spectrogram to one
//! speaker embedding, through a TDNN frontend, three SE-Res2Net blocks,
//! multi-layer feature aggregation and attentive statistics pooling. This
//! file's convention is `[C, T]`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/qwen3tts-spkenc.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// `SPK_RES2NET_SCALE` (qwen3tts-spkenc.cpp:3 @b1b770d90): enc_res2net_scale.
const spk_res2net_scale: c_int = 8;
/// `SPK_DILATIONS` (qwen3tts-spkenc.cpp:4 @b1b770d90): enc_dilations[1..3].
const spk_dilations = [3]c_int{ 2, 3, 4 };

/// Ports `conv1d_same` (qwen3tts-spkenc.cpp:8 @b1b770d90), a member of
/// `mtmd_graph_qwen3tts_spkenc`: kernel K, "same" padding (reflect),
/// dilation d; x `[C, T]` to `[out_c, T]`.
fn conv1dSame(g: *const mtmd.graph.Graph, x: mtmd.graph.T, w: mtmd.graph.T, b: mtmd.graph.T, dilation: c_int) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const K: c_int = @intCast(mtmd.graph.ne(w, 0));
    const IC: c_int = @intCast(mtmd.graph.ne(w, 1));
    const OC: c_int = @intCast(mtmd.graph.ne(w, 2));
    const pad = @divTrunc((K - 1) * dilation, 2);

    // ggml_pad_reflect_1d pads ne[0], so bring T onto ne[0] first, same layout as im2col wants
    var x_t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, x)); // [T, IC]
    if (pad > 0) {
        x_t = llama.impl.llama_c.ggml_pad_reflect_1d(ctx0, x_t, pad, pad); // [T + 2*pad, IC]
    }
    const x4d = llama.impl.llama_c.ggml_reshape_4d(ctx0, x_t, mtmd.graph.ne(x_t, 0), IC, 1, 1);

    // dummy F32 kernel, im2col only reads its shape, so a quantized w does not assert
    const dummy = llama.impl.llama_c.ggml_new_tensor_4d(ctx0, llama.impl.llama_c.GGML_TYPE_F32, K, IC, 1, 1);

    var col = llama.impl.llama_c.ggml_im2col(ctx0, dummy, x4d, 1, 1, 0, 0, dilation, 1, false, llama.impl.llama_c.GGML_TYPE_F32);
    const T_out = mtmd.graph.ne(col, 1);
    col = llama.impl.llama_c.ggml_reshape_2d(ctx0, col, @as(i64, K) * IC, T_out);

    const w2d = llama.impl.llama_c.ggml_reshape_2d(ctx0, w, @as(i64, K) * IC, OC);
    var y = llama.impl.llama_c.ggml_mul_mat(ctx0, w2d, col); // [OC, T_out]
    _ = llama.impl.llama_c.ggml_prec_set_acc(y, llama.impl.llama_c.GGML_PREC_BF16);

    const b2d = llama.impl.llama_c.ggml_reshape_2d(ctx0, b, OC, 1);
    y = llama.impl.llama_c.ggml_add(ctx0, y, b2d);
    return y;
}

/// Ports `res2net` (qwen3tts-spkenc.cpp:39 @b1b770d90), a member of
/// `mtmd_graph_qwen3tts_spkenc`: split the channel axis into `scale` chunks
/// and chain dilated conv1d branches; x `[C, T]` to `[C, T]`.
fn res2net(g: *const mtmd.graph.Graph, x: mtmd.graph.T, layer: *const mtmd.model.Layer, dilation: c_int, scale: c_int) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const C = mtmd.graph.ne(x, 0);
    const Tn = mtmd.graph.ne(x, 1);
    const Cs = @divTrunc(C, scale);

    var outs: std.ArrayList(mtmd.graph.T) = .empty;
    defer outs.deinit(std.heap.c_allocator);
    outs.ensureTotalCapacity(std.heap.c_allocator, @intCast(scale)) catch oom();

    var prev: mtmd.graph.T = null;
    var i: c_int = 0;
    while (i < scale) : (i += 1) {
        // `c` upstream, which names the C namespace here
        const ch = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, x, Cs, Tn, mtmd.graph.nb(x, 1), @as(usize, @intCast(i)) * @as(usize, @intCast(Cs)) * mtmd.graph.nb(x, 0)));
        if (i == 0) {
            outs.appendAssumeCapacity(ch);
            continue;
        }
        const inp = if (i >= 2) llama.impl.llama_c.ggml_add(ctx0, ch, prev) else ch;
        const k: usize = @intCast(i - 1);
        var y = conv1dSame(g, inp, layer.res2_conv_w.items[k], layer.res2_conv_b.items[k], dilation);
        y = llama.impl.llama_c.ggml_relu(ctx0, y);
        outs.appendAssumeCapacity(y);
        prev = y;
    }

    var acc = outs.items[0];
    for (outs.items[1..]) |o| {
        acc = llama.impl.llama_c.ggml_concat(ctx0, acc, o, 0);
    }
    return acc;
}

/// Ports `se_block` (qwen3tts-spkenc.cpp:73 @b1b770d90), a member of
/// `mtmd_graph_qwen3tts_spkenc`: squeeze-and-excitation gate; x `[C, T]` to
/// `[C, T]`.
fn seBlock(g: *const mtmd.graph.Graph, x: mtmd.graph.T, layer: *const mtmd.model.Layer) mtmd.graph.T {
    const ctx0 = g.ctx0;
    // temporal mean, keepdim: transpose so T is on ne[0], reduce, transpose back
    const x_t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, x)); // [T, C]
    var mean = llama.impl.llama_c.ggml_mean(ctx0, x_t); // [1, C]
    mean = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, mean)); // [C, 1]

    var h = conv1dSame(g, mean, layer.se_conv1_w, layer.se_conv1_b, 1);
    h = llama.impl.llama_c.ggml_relu(ctx0, h);
    h = conv1dSame(g, h, layer.se_conv2_w, layer.se_conv2_b, 1);
    h = llama.impl.llama_c.ggml_sigmoid(ctx0, h); // [C, 1]

    return llama.impl.llama_c.ggml_mul(ctx0, x, h); // broadcast gate over T
}

/// Ports `se_res2net_block` (qwen3tts-spkenc.cpp:88 @b1b770d90), a member
/// of `mtmd_graph_qwen3tts_spkenc`: tdnn1, res2net, tdnn2, se, plus the
/// residual; x `[C, T]` to `[C, T]`.
fn seRes2netBlock(g: *const mtmd.graph.Graph, x: mtmd.graph.T, layer: *const mtmd.model.Layer, dilation: c_int, scale: c_int) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const residual = x;
    var h = conv1dSame(g, x, layer.conv_pw1_w, layer.conv_pw1_b, 1); // tdnn1
    h = llama.impl.llama_c.ggml_relu(ctx0, h);
    h = res2net(g, h, layer, dilation, scale);
    h = conv1dSame(g, h, layer.conv_pw2_w, layer.conv_pw2_b, 1); // tdnn2
    h = llama.impl.llama_c.ggml_relu(ctx0, h);
    h = seBlock(g, h, layer);
    return llama.impl.llama_c.ggml_add(ctx0, h, residual);
}

/// Ports `attentive_stats_pool` (qwen3tts-spkenc.cpp:100 @b1b770d90), a
/// member of `mtmd_graph_qwen3tts_spkenc`: x `[C, T]` to `[2*C, 1]`.
fn attentiveStatsPool(g: *const mtmd.graph.Graph, x: mtmd.graph.T) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const model = g.model;
    const Tn = mtmd.graph.ne(x, 1);

    // mean over T: [C, 1]
    const x_t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, x));
    var mean = llama.impl.llama_c.ggml_mean(ctx0, x_t);
    mean = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, mean));

    // std over T: sqrt(clamp(mean((x - mean)^2), eps))
    const mean_rep = llama.impl.llama_c.ggml_repeat(ctx0, mean, x);
    const centered = llama.impl.llama_c.ggml_sub(ctx0, x, mean_rep);
    const var_t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, llama.impl.llama_c.ggml_sqr(ctx0, centered)));
    var variance = llama.impl.llama_c.ggml_mean(ctx0, var_t);
    variance = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, variance));
    variance = llama.impl.llama_c.ggml_scale_bias(ctx0, variance, 1.0, 1e-12);
    const std_ = llama.impl.llama_c.ggml_sqrt(ctx0, variance);

    // attention input: cat([x, mean, std]) along channel axis -> [3C, T]
    const std_rep = llama.impl.llama_c.ggml_repeat(ctx0, std_, x);
    var cat = llama.impl.llama_c.ggml_concat(ctx0, x, mean_rep, 0);
    cat = llama.impl.llama_c.ggml_concat(ctx0, cat, std_rep, 0);

    // attention TDNN (3C -> attn_c) + ReLU, tanh, then 1x1 conv (attn_c -> C)
    var a = conv1dSame(g, cat, model.spk_asp_tdnn_w, model.spk_asp_tdnn_b, 1);
    a = llama.impl.llama_c.ggml_relu(ctx0, a);
    a = llama.impl.llama_c.ggml_tanh(ctx0, a);
    a = conv1dSame(g, a, model.spk_asp_attn_w, model.spk_asp_attn_b, 1);

    // softmax over T
    const a_t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, a)); // [T, C]
    const w_t = llama.impl.llama_c.ggml_soft_max(ctx0, a_t);
    const w = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, w_t)); // [C, T]

    // weighted mean: sum(w * x) over T, multiply by T to undo ggml_mean's 1/T scaling
    const wx = llama.impl.llama_c.ggml_mul(ctx0, w, x);
    const wx_t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, wx));
    var w_mean = llama.impl.llama_c.ggml_mean(ctx0, wx_t);
    w_mean = llama.impl.llama_c.ggml_scale(ctx0, w_mean, @floatFromInt(Tn));
    w_mean = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, w_mean)); // [C, 1]

    // weighted std: sum(w * (x - w_mean)^2) over T
    const w_mean_rep = llama.impl.llama_c.ggml_repeat(ctx0, w_mean, x);
    const dev = llama.impl.llama_c.ggml_sub(ctx0, x, w_mean_rep);
    const w_var_in = llama.impl.llama_c.ggml_mul(ctx0, w, llama.impl.llama_c.ggml_sqr(ctx0, dev));
    const w_var_t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, w_var_in));
    var w_var = llama.impl.llama_c.ggml_mean(ctx0, w_var_t);
    w_var = llama.impl.llama_c.ggml_scale(ctx0, w_var, @floatFromInt(Tn));
    w_var = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, w_var));
    w_var = llama.impl.llama_c.ggml_scale_bias(ctx0, w_var, 1.0, 1e-12);
    const w_std = llama.impl.llama_c.ggml_sqrt(ctx0, w_var);

    return llama.impl.llama_c.ggml_concat(ctx0, w_mean, w_std, 0); // [2C, 1]
}

/// Ports `build` (qwen3tts-spkenc.cpp:154 @b1b770d90), a member of
/// `mtmd_graph_qwen3tts_spkenc`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*llama.impl.llama_c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;

    // inp_raw: [T, n_mel, 1, 1], from mtmd_audio_preprocessor_qwen3tts_spk
    var inp = g.buildInpRaw(1);
    inp = llama.impl.llama_c.ggml_reshape_2d(ctx0, inp, mtmd.graph.ne(inp, 0), mtmd.graph.ne(inp, 1));

    // this file's convention is [C, T]; the preprocessor delivers [T, C]
    const mel = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, inp)); // [n_mel, T]
    g.cb(mel, "mel", -1);

    // frontend conv0 TDNN k=5, dilation=1: 128 -> 512
    var cur = conv1dSame(g, mel, model.conv1d_1_w, model.conv1d_1_b, 1);
    cur = llama.impl.llama_c.ggml_relu(ctx0, cur);
    g.cb(cur, "frontend", -1);

    // 3 SE-Res2Net blocks at dilations 2, 3, 4
    mtmd.impl.ggmlAssert(model.layers.items.len == 3, "qwen3tts-spkenc.cpp", 169, "(int) model.layers.size() == 3");
    var blk_out: [3]mtmd.graph.T = .{ null, null, null };
    for (0..3) |il| {
        cur = seRes2netBlock(g, cur, &model.layers.items[il], spk_dilations[il], spk_res2net_scale);
        blk_out[il] = cur;
        g.cb(cur, "block_out", @intCast(il));
    }

    // multi-layer feature aggregation: cat blk[0..2] then TDNN k=1 + ReLU
    var cat = llama.impl.llama_c.ggml_concat(ctx0, blk_out[0], blk_out[1], 0);
    cat = llama.impl.llama_c.ggml_concat(ctx0, cat, blk_out[2], 0); // [1536, T]
    var mfa = conv1dSame(g, cat, model.conv_out_w, model.conv_out_b, 1);
    mfa = llama.impl.llama_c.ggml_relu(ctx0, mfa);
    g.cb(mfa, "mfa", -1);

    // attentive statistics pooling: [1536, T] -> [3072, 1]
    const stats = attentiveStatsPool(g, mfa);
    g.cb(stats, "asp", -1);

    // final FC k=1: [3072, 1] -> [enc_dim, 1]
    var emb = conv1dSame(g, stats, model.mm_fc_w, model.mm_fc_b, 1);

    emb = llama.impl.llama_c.ggml_reshape_1d(ctx0, emb, mtmd.graph.ne(emb, 0));
    emb = llama.impl.llama_c.ggml_cont(ctx0, emb);
    g.cb(emb, "spk_embedding", -1);

    llama.impl.llama_c.ggml_build_forward_expand(g.gf, emb);
    return g.gf;
}

fn oom() noreturn {
    mtmd.impl.abort("qwen3tts-spkenc.cpp", 0, "out of memory");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
