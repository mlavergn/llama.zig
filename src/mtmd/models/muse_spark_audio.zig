//! The Muse Spark audio tower: a causal conv pyramid over the raw waveform,
//! no attention and no spectrogram. Each stage folds `patch` samples into the
//! channel dim, projects, then runs `depth` residual blocks (depthwise causal
//! conv, RMS norm, GELU MLP).
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/muse-spark-audio.cpp` at v0.8.0
//! (`8642541b2`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const ne = graph.ne;

/// Ports `build` (muse-spark-audio.cpp:5 @cd959bae9), a member of
/// `clip_graph_muse_spark_audio`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    const kernel = hparams.audio_conv_kernel_size;
    const n_stage: c_int = @intCast(hparams.audio_patch_sizes.items.len);

    // shape [1, n_samples], so the first stage folds the same way as the others
    var cur = g.buildInpRaw(1);
    cur = c.ggml_reshape_2d(ctx0, cur, 1, g.img.nx());
    g.cb(cur, "inp_raw", -1);

    var bid: c_int = 0;
    var stage: c_int = 0;
    while (stage < n_stage) : (stage += 1) {
        const si: usize = @intCast(stage);
        const patch = hparams.audio_patch_sizes.items[si];
        const pe = &model.a_patches.items[si];

        // fold `patch` positions into the channel dim
        cur = c.ggml_reshape_2d(ctx0, cur, ne(cur, 0) * patch, @divTrunc(ne(cur, 1), patch));
        g.cb(cur, "patch_folded", stage);

        impl.ggmlAssert(ne(pe.proj_w, 1) == hparams.audio_hidden_sizes.items[si], "muse-spark-audio.cpp", 23, "pe.proj_w->ne[1] == hparams.audio_hidden_sizes[stage]");
        cur = c.ggml_mul_mat(ctx0, pe.proj_w, cur);
        cur = c.ggml_add(ctx0, cur, pe.proj_b);
        cur = g.buildNorm(cur, pe.norm_w, null, .rms, hparams.eps, stage);
        g.cb(cur, "patch_embd", stage);

        var i: c_int = 0;
        while (i < hparams.audio_depths.items[si]) : ({
            i += 1;
            bid += 1;
        }) {
            const layer = &model.layers.items[@intCast(bid)];
            const residual = cur;

            // depthwise conv over time, pad on the left so it is causal
            var conv = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
            conv = c.ggml_pad_ext(ctx0, conv, kernel - 1, 0, 0, 0, 0, 0, 0, 0);
            conv = c.ggml_conv_1d_dw(ctx0, layer.conv_w, conv, 1, 0, 1);
            conv = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, conv));
            conv = c.ggml_add(ctx0, conv, layer.conv_b);
            g.cb(conv, "conv", bid);

            cur = g.buildNorm(conv, layer.conv_norm_w, null, .rms, hparams.eps, bid);

            cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, null, null, layer.ff_down_w, layer.ff_down_b, .gelu_erf, bid);
            g.cb(cur, "ffn_out", bid);

            cur = c.ggml_add(ctx0, cur, residual);
            g.cb(cur, "block_out", bid);
        }
    }

    cur = g.buildNorm(cur, model.pre_ln_w, null, .rms, hparams.eps, -1);
    g.cb(cur, "enc_norm", -1);

    // straight into the text hidden size, no adapter
    cur = c.ggml_mul_mat(ctx0, model.mm_fc_w, cur);
    cur = c.ggml_add(ctx0, cur, model.mm_fc_b);
    cur = g.buildNorm(cur, model.post_ln_w, null, .rms, hparams.eps, -1);
    g.cb(cur, "projected", -1);

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
