//! The Yasa2 encoder graph: a ConvNeXt-V2 backbone (patch conv, stages of
//! depthwise conv, channel layer norm, GRN), learned position embeddings,
//! adaptive average pooling to 8x8 and an MLP projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/yasa2.cpp` at v1.0.0 (`b1b770d90`).
//! Each declaration names the C++ it replaces and the line it began at.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// Ports `add_channel_bias` (yasa2.cpp:6 @b1b770d90).
fn addChannelBias(ctx0: ?*llama.impl.llama_c.ggml_context, x_whcb: mtmd.graph.T, b_c: mtmd.graph.T) mtmd.graph.T {
    if (b_c == null) return x_whcb;
    const b4 = llama.impl.llama_c.ggml_reshape_4d(ctx0, b_c, 1, 1, mtmd.graph.ne(b_c, 0), 1);
    return llama.impl.llama_c.ggml_add(ctx0, x_whcb, b4);
}

/// Ports `mul_channel_weight` (yasa2.cpp:17 @b1b770d90).
fn mulChannelWeight(ctx0: ?*llama.impl.llama_c.ggml_context, x_whcb: mtmd.graph.T, w_c: mtmd.graph.T) mtmd.graph.T {
    if (w_c == null) return x_whcb;
    const w4 = llama.impl.llama_c.ggml_reshape_4d(ctx0, w_c, 1, 1, mtmd.graph.ne(w_c, 0), 1);
    return llama.impl.llama_c.ggml_mul(ctx0, x_whcb, w4);
}

/// Ports `layer_norm_channels` (yasa2.cpp:28 @b1b770d90), a member of
/// `mtmd_graph_yasa2`: HF's `ConvNextLayerNorm(channels_first)`.
fn layerNormChannels(g: *const mtmd.graph.Graph, inp: mtmd.graph.T, w: mtmd.graph.T, b: mtmd.graph.T, eps: f32) mtmd.graph.T {
    const ctx0 = g.ctx0;
    // u = mean_c(x), s = mean_c((x-u)^2), x = (x-u)/sqrt(s+eps)
    // cast back to input dtype before affine.
    var cur = llama.impl.llama_c.ggml_permute(ctx0, inp, 2, 1, 0, 3); // [W,H,C,B] -> [C,H,W,B]
    cur = llama.impl.llama_c.ggml_cont(ctx0, cur);

    const u = llama.impl.llama_c.ggml_mean(ctx0, cur); // [1,H,W,B]
    const xm = llama.impl.llama_c.ggml_sub(ctx0, cur, u); // [C,H,W,B]

    var s = llama.impl.llama_c.ggml_mul(ctx0, xm, xm); // [C,H,W,B]
    s = llama.impl.llama_c.ggml_mean(ctx0, s); // [1,H,W,B]
    s = llama.impl.llama_c.ggml_clamp(ctx0, s, eps, 1e30); // avoid div-by-zero in no-alloc warmup
    s = llama.impl.llama_c.ggml_sqrt(ctx0, s); // [1,H,W,B]

    var xhat = llama.impl.llama_c.ggml_div(ctx0, xm, s); // [C,H,W,B]
    xhat = llama.impl.llama_c.ggml_permute(ctx0, xhat, 2, 1, 0, 3); // [W,H,C,B]
    xhat = llama.impl.llama_c.ggml_cont(ctx0, xhat);
    xhat = mulChannelWeight(ctx0, xhat, w);
    xhat = addChannelBias(ctx0, xhat, b);
    return xhat;
}

/// Ports `convnext_grn` (yasa2.cpp:51 @b1b770d90), a member of
/// `mtmd_graph_yasa2`: ConvNeXt-V2's global response normalization.
fn convnextGrn(g: *const mtmd.graph.Graph, inp: mtmd.graph.T, w: mtmd.graph.T, b: mtmd.graph.T) mtmd.graph.T {
    const ctx0 = g.ctx0;
    // Gx = ||x||_2 over spatial dims (W,H), Nx = Gx / (mean_c(Gx) + eps)
    // y  = w * (x * Nx) + b + x
    const wdim = mtmd.graph.ne(inp, 0);
    const hdim = mtmd.graph.ne(inp, 1);
    const cdim = mtmd.graph.ne(inp, 2);
    const bdim = mtmd.graph.ne(inp, 3);

    // Keep GRN math in fp32 for stability; fp16/bf16 accumulation can drift.
    const sq = llama.impl.llama_c.ggml_mul(ctx0, inp, inp);
    const sq_flat = llama.impl.llama_c.ggml_reshape_4d(ctx0, sq, wdim * hdim, cdim, 1, bdim); // [WH,C,1,B]
    var gx = llama.impl.llama_c.ggml_sum_rows(ctx0, sq_flat); // [1,C,1,B]
    gx = llama.impl.llama_c.ggml_sqrt(ctx0, gx); // [1,C,1,B]

    var gx_ch_first = llama.impl.llama_c.ggml_permute(ctx0, gx, 1, 0, 2, 3); // [C,1,1,B]
    gx_ch_first = llama.impl.llama_c.ggml_cont(ctx0, gx_ch_first);
    var gx_mean = llama.impl.llama_c.ggml_mean(ctx0, gx_ch_first); // [1,1,1,B]

    gx_mean = llama.impl.llama_c.ggml_clamp(ctx0, gx_mean, 1e-6, 1e30); // approx +eps, warmup-safe
    var nx = llama.impl.llama_c.ggml_div(ctx0, gx, gx_mean); // [1,C,1,B]
    nx = llama.impl.llama_c.ggml_permute(ctx0, nx, 0, 2, 1, 3); // [1,1,C,B]
    nx = llama.impl.llama_c.ggml_cont(ctx0, nx);

    var xnx = llama.impl.llama_c.ggml_mul(ctx0, inp, nx);
    xnx = mulChannelWeight(ctx0, xnx, w);
    xnx = addChannelBias(ctx0, xnx, b);
    return llama.impl.llama_c.ggml_add(ctx0, inp, xnx);
}

/// Ports `build` (yasa2.cpp:81 @b1b770d90), a member of `mtmd_graph_yasa2`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*llama.impl.llama_c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const eps = g.eps;

    var cur = g.buildInpRaw(3);

    // Patch embedding Conv2d(kernel=4, stride=4)
    cur = llama.impl.llama_c.ggml_conv_2d(ctx0, model.yasa_patch_w, cur, g.patch_size, g.patch_size, 0, 0, 1, 1);
    cur = addChannelBias(ctx0, cur, model.yasa_patch_b);
    _ = llama.impl.llama_c.ggml_set_name(cur, "yasa2_patch_conv_out");
    g.cb(cur, "yasa2_patch_conv_out", -1);
    cur = layerNormChannels(g, cur, model.yasa_patch_ln_w, model.yasa_patch_ln_b, eps);
    _ = llama.impl.llama_c.ggml_set_name(cur, "yasa2_patch_ln_out");
    g.cb(cur, "yasa2_patch_ln_out", -1);

    // ConvNeXt stages
    for (model.yasa_stages.items, 0..) |*stage, s| {
        if (stage.down_conv_w != null) {
            cur = layerNormChannels(g, cur, stage.down_ln_w, stage.down_ln_b, eps);
            cur = llama.impl.llama_c.ggml_conv_2d(ctx0, stage.down_conv_w, cur, 2, 2, 0, 0, 1, 1);
            cur = addChannelBias(ctx0, cur, stage.down_conv_b);
            _ = llama.impl.llama_c.ggml_format_name(cur, "yasa2_stage%zu_down_out", s);
        }

        for (stage.blocks.items, 0..) |*blk, bi| {
            const res = cur;

            var x = llama.impl.llama_c.ggml_conv_2d_dw(ctx0, blk.dw_w, cur, 1, 1, 3, 3, 1, 1);
            x = addChannelBias(ctx0, x, blk.dw_b);
            x = layerNormChannels(g, x, blk.ln_w, blk.ln_b, eps);

            // pwconv1/pwconv2 are HF Linear layers over channels; implement via matmul on tokens.
            const w = mtmd.graph.ne(x, 0);
            const h = mtmd.graph.ne(x, 1);
            const b = mtmd.graph.ne(x, 3);

            var tok = llama.impl.llama_c.ggml_reshape_3d(ctx0, x, w * h, mtmd.graph.ne(x, 2), b); // [T,C,B]
            tok = llama.impl.llama_c.ggml_permute(ctx0, tok, 1, 0, 2, 3); // [C,T,B]
            tok = llama.impl.llama_c.ggml_cont(ctx0, tok);

            tok = llama.impl.llama_c.ggml_mul_mat(ctx0, blk.pw1_w, tok); // [4C,T,B]
            if (blk.pw1_b != null) {
                const b1 = llama.impl.llama_c.ggml_reshape_3d(ctx0, blk.pw1_b, mtmd.graph.ne(blk.pw1_b, 0), 1, 1); // [4C,1,1]
                tok = llama.impl.llama_c.ggml_add(ctx0, tok, b1);
            }
            x = llama.impl.llama_c.ggml_permute(ctx0, tok, 1, 0, 2, 3); // [T,4C,B]
            x = llama.impl.llama_c.ggml_cont(ctx0, x);
            x = llama.impl.llama_c.ggml_reshape_4d(ctx0, x, w, h, mtmd.graph.ne(tok, 0), b); // [W,H,4C,B]
            x = llama.impl.llama_c.ggml_gelu_erf(ctx0, x);
            x = convnextGrn(g, x, blk.grn_w, blk.grn_b);

            tok = llama.impl.llama_c.ggml_reshape_3d(ctx0, x, w * h, mtmd.graph.ne(x, 2), b); // [T,4C,B]
            tok = llama.impl.llama_c.ggml_permute(ctx0, tok, 1, 0, 2, 3); // [4C,T,B]
            tok = llama.impl.llama_c.ggml_cont(ctx0, tok);

            tok = llama.impl.llama_c.ggml_mul_mat(ctx0, blk.pw2_w, tok); // [C,T,B]
            if (blk.pw2_b != null) {
                const b2 = llama.impl.llama_c.ggml_reshape_3d(ctx0, blk.pw2_b, mtmd.graph.ne(blk.pw2_b, 0), 1, 1); // [C,1,1]
                tok = llama.impl.llama_c.ggml_add(ctx0, tok, b2);
            }
            x = llama.impl.llama_c.ggml_permute(ctx0, tok, 1, 0, 2, 3); // [T,C,B]
            x = llama.impl.llama_c.ggml_cont(ctx0, x);
            x = llama.impl.llama_c.ggml_reshape_4d(ctx0, x, w, h, mtmd.graph.ne(tok, 0), b); // [W,H,C,B]

            cur = llama.impl.llama_c.ggml_add(ctx0, res, x);
            _ = llama.impl.llama_c.ggml_format_name(cur, "yasa2_stage%zu_blk%zu_out", s, bi);
        }
    }

    // HF path adds vision position embeddings BEFORE adaptive pooling.
    const pre_w = mtmd.graph.ne(cur, 0);
    const pre_h = mtmd.graph.ne(cur, 1);
    var tokens_pre = llama.impl.llama_c.ggml_reshape_3d(ctx0, cur, pre_w * pre_h, mtmd.graph.ne(cur, 2), mtmd.graph.ne(cur, 3)); // [T,C,B]
    tokens_pre = llama.impl.llama_c.ggml_permute(ctx0, tokens_pre, 1, 0, 2, 3); // [C,T,B]
    tokens_pre = llama.impl.llama_c.ggml_cont(ctx0, tokens_pre);
    if (model.yasa_vision_pos_embed != null and mtmd.graph.ne(tokens_pre, 1) == mtmd.graph.ne(model.yasa_vision_pos_embed, 1)) {
        const n_ch = mtmd.graph.ne(model.yasa_vision_pos_embed, 0);
        const n_tokens = mtmd.graph.ne(model.yasa_vision_pos_embed, 1);
        const pos = llama.impl.llama_c.ggml_reshape_3d(ctx0, model.yasa_vision_pos_embed, @as(c_int, @truncate(n_ch)), @as(c_int, @truncate(n_tokens)), 1);
        tokens_pre = llama.impl.llama_c.ggml_add(ctx0, tokens_pre, pos);
    }
    cur = llama.impl.llama_c.ggml_permute(ctx0, tokens_pre, 1, 0, 2, 3); // [T,C,B]
    cur = llama.impl.llama_c.ggml_cont(ctx0, cur);
    cur = llama.impl.llama_c.ggml_reshape_4d(ctx0, cur, pre_w, pre_h, mtmd.graph.ne(cur, 1), mtmd.graph.ne(cur, 2)); // [W,H,C,B]

    // AdaptiveAvgPool2d target is 8x8 for real inputs, but warmup can use tiny images.
    const pooled_w: c_int = @min(8, @as(c_int, @truncate(mtmd.graph.ne(cur, 0))));
    const pooled_h: c_int = @min(8, @as(c_int, @truncate(mtmd.graph.ne(cur, 1))));
    const kw: c_int = @max(1, @divTrunc(@as(c_int, @truncate(mtmd.graph.ne(cur, 0))), pooled_w));
    const kh: c_int = @max(1, @divTrunc(@as(c_int, @truncate(mtmd.graph.ne(cur, 1))), pooled_h));
    cur = llama.impl.llama_c.ggml_pool_2d(ctx0, cur, llama.impl.llama_c.GGML_OP_POOL_AVG, kw, kh, kw, kh, 0, 0);

    // [W,H,C,B] -> [C,T,B]
    var tokens = llama.impl.llama_c.ggml_reshape_3d(ctx0, cur, mtmd.graph.ne(cur, 0) * mtmd.graph.ne(cur, 1), mtmd.graph.ne(cur, 2), mtmd.graph.ne(cur, 3));
    tokens = llama.impl.llama_c.ggml_permute(ctx0, tokens, 1, 0, 2, 3);
    tokens = llama.impl.llama_c.ggml_cont(ctx0, tokens);
    g.cb(tokens, "yasa2_tokens", -1);

    mtmd.impl.ggmlAssert(model.mm_0_w != null and model.mm_2_w != null, "yasa2.cpp", 179, "model.mm_0_w && model.mm_2_w");
    const embeddings = g.buildFfn(tokens, model.mm_0_w, model.mm_0_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);
    g.cb(embeddings, "yasa2_emb", -1);

    llama.impl.llama_c.ggml_build_forward_expand(g.gf, embeddings);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
