//! The Yasa2 encoder graph: a ConvNeXt-V2 backbone (patch conv, stages of
//! depthwise conv, channel layer norm, GRN), learned position embeddings,
//! adaptive average pooling to 8x8 and an MLP projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/yasa2.cpp` at v0.4.0 (`5266f24da`).
//! Each declaration names the C++ it replaces and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;

/// Ports `add_channel_bias` (yasa2.cpp:6 @5266f24da).
fn addChannelBias(ctx0: ?*c.ggml_context, x_whcb: T, b_c: T) T {
    if (b_c == null) return x_whcb;
    const b4 = c.ggml_reshape_4d(ctx0, b_c, 1, 1, ne(b_c, 0), 1);
    return c.ggml_add(ctx0, x_whcb, b4);
}

/// Ports `mul_channel_weight` (yasa2.cpp:17 @5266f24da).
fn mulChannelWeight(ctx0: ?*c.ggml_context, x_whcb: T, w_c: T) T {
    if (w_c == null) return x_whcb;
    const w4 = c.ggml_reshape_4d(ctx0, w_c, 1, 1, ne(w_c, 0), 1);
    return c.ggml_mul(ctx0, x_whcb, w4);
}

/// Ports `layer_norm_channels` (yasa2.cpp:28 @5266f24da), a member of
/// `clip_graph_yasa2`: HF's `ConvNextLayerNorm(channels_first)`.
fn layerNormChannels(g: *const graph.Graph, inp: T, w: T, b: T, eps: f32) T {
    const ctx0 = g.ctx0;
    // u = mean_c(x), s = mean_c((x-u)^2), x = (x-u)/sqrt(s+eps)
    // cast back to input dtype before affine.
    var cur = c.ggml_permute(ctx0, inp, 2, 1, 0, 3); // [W,H,C,B] -> [C,H,W,B]
    cur = c.ggml_cont(ctx0, cur);

    const u = c.ggml_mean(ctx0, cur); // [1,H,W,B]
    const xm = c.ggml_sub(ctx0, cur, u); // [C,H,W,B]

    var s = c.ggml_mul(ctx0, xm, xm); // [C,H,W,B]
    s = c.ggml_mean(ctx0, s); // [1,H,W,B]
    s = c.ggml_clamp(ctx0, s, eps, 1e30); // avoid div-by-zero in no-alloc warmup
    s = c.ggml_sqrt(ctx0, s); // [1,H,W,B]

    var xhat = c.ggml_div(ctx0, xm, s); // [C,H,W,B]
    xhat = c.ggml_permute(ctx0, xhat, 2, 1, 0, 3); // [W,H,C,B]
    xhat = c.ggml_cont(ctx0, xhat);
    xhat = mulChannelWeight(ctx0, xhat, w);
    xhat = addChannelBias(ctx0, xhat, b);
    return xhat;
}

/// Ports `convnext_grn` (yasa2.cpp:51 @5266f24da), a member of
/// `clip_graph_yasa2`: ConvNeXt-V2's global response normalization.
fn convnextGrn(g: *const graph.Graph, inp: T, w: T, b: T) T {
    const ctx0 = g.ctx0;
    // Gx = ||x||_2 over spatial dims (W,H), Nx = Gx / (mean_c(Gx) + eps)
    // y  = w * (x * Nx) + b + x
    const wdim = ne(inp, 0);
    const hdim = ne(inp, 1);
    const cdim = ne(inp, 2);
    const bdim = ne(inp, 3);

    // Keep GRN math in fp32 for stability; fp16/bf16 accumulation can drift.
    const sq = c.ggml_mul(ctx0, inp, inp);
    const sq_flat = c.ggml_reshape_4d(ctx0, sq, wdim * hdim, cdim, 1, bdim); // [WH,C,1,B]
    var gx = c.ggml_sum_rows(ctx0, sq_flat); // [1,C,1,B]
    gx = c.ggml_sqrt(ctx0, gx); // [1,C,1,B]

    var gx_ch_first = c.ggml_permute(ctx0, gx, 1, 0, 2, 3); // [C,1,1,B]
    gx_ch_first = c.ggml_cont(ctx0, gx_ch_first);
    var gx_mean = c.ggml_mean(ctx0, gx_ch_first); // [1,1,1,B]

    gx_mean = c.ggml_clamp(ctx0, gx_mean, 1e-6, 1e30); // approx +eps, warmup-safe
    var nx = c.ggml_div(ctx0, gx, gx_mean); // [1,C,1,B]
    nx = c.ggml_permute(ctx0, nx, 0, 2, 1, 3); // [1,1,C,B]
    nx = c.ggml_cont(ctx0, nx);

    var xnx = c.ggml_mul(ctx0, inp, nx);
    xnx = mulChannelWeight(ctx0, xnx, w);
    xnx = addChannelBias(ctx0, xnx, b);
    return c.ggml_add(ctx0, inp, xnx);
}

/// Ports `build` (yasa2.cpp:81 @5266f24da), a member of `clip_graph_yasa2`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const eps = g.eps;

    var cur = g.buildInpRaw(3);

    // Patch embedding Conv2d(kernel=4, stride=4)
    cur = c.ggml_conv_2d(ctx0, model.yasa_patch_w, cur, g.patch_size, g.patch_size, 0, 0, 1, 1);
    cur = addChannelBias(ctx0, cur, model.yasa_patch_b);
    _ = c.ggml_set_name(cur, "yasa2_patch_conv_out");
    g.cb(cur, "yasa2_patch_conv_out", -1);
    cur = layerNormChannels(g, cur, model.yasa_patch_ln_w, model.yasa_patch_ln_b, eps);
    _ = c.ggml_set_name(cur, "yasa2_patch_ln_out");
    g.cb(cur, "yasa2_patch_ln_out", -1);

    // ConvNeXt stages
    for (model.yasa_stages.items, 0..) |*stage, s| {
        if (stage.down_conv_w != null) {
            cur = layerNormChannels(g, cur, stage.down_ln_w, stage.down_ln_b, eps);
            cur = c.ggml_conv_2d(ctx0, stage.down_conv_w, cur, 2, 2, 0, 0, 1, 1);
            cur = addChannelBias(ctx0, cur, stage.down_conv_b);
            _ = c.ggml_format_name(cur, "yasa2_stage%zu_down_out", s);
        }

        for (stage.blocks.items, 0..) |*blk, bi| {
            const res = cur;

            var x = c.ggml_conv_2d_dw(ctx0, blk.dw_w, cur, 1, 1, 3, 3, 1, 1);
            x = addChannelBias(ctx0, x, blk.dw_b);
            x = layerNormChannels(g, x, blk.ln_w, blk.ln_b, eps);

            // pwconv1/pwconv2 are HF Linear layers over channels; implement via matmul on tokens.
            const w = ne(x, 0);
            const h = ne(x, 1);
            const b = ne(x, 3);

            var tok = c.ggml_reshape_3d(ctx0, x, w * h, ne(x, 2), b); // [T,C,B]
            tok = c.ggml_permute(ctx0, tok, 1, 0, 2, 3); // [C,T,B]
            tok = c.ggml_cont(ctx0, tok);

            tok = c.ggml_mul_mat(ctx0, blk.pw1_w, tok); // [4C,T,B]
            if (blk.pw1_b != null) {
                const b1 = c.ggml_reshape_3d(ctx0, blk.pw1_b, ne(blk.pw1_b, 0), 1, 1); // [4C,1,1]
                tok = c.ggml_add(ctx0, tok, b1);
            }
            x = c.ggml_permute(ctx0, tok, 1, 0, 2, 3); // [T,4C,B]
            x = c.ggml_cont(ctx0, x);
            x = c.ggml_reshape_4d(ctx0, x, w, h, ne(tok, 0), b); // [W,H,4C,B]
            x = c.ggml_gelu_erf(ctx0, x);
            x = convnextGrn(g, x, blk.grn_w, blk.grn_b);

            tok = c.ggml_reshape_3d(ctx0, x, w * h, ne(x, 2), b); // [T,4C,B]
            tok = c.ggml_permute(ctx0, tok, 1, 0, 2, 3); // [4C,T,B]
            tok = c.ggml_cont(ctx0, tok);

            tok = c.ggml_mul_mat(ctx0, blk.pw2_w, tok); // [C,T,B]
            if (blk.pw2_b != null) {
                const b2 = c.ggml_reshape_3d(ctx0, blk.pw2_b, ne(blk.pw2_b, 0), 1, 1); // [C,1,1]
                tok = c.ggml_add(ctx0, tok, b2);
            }
            x = c.ggml_permute(ctx0, tok, 1, 0, 2, 3); // [T,C,B]
            x = c.ggml_cont(ctx0, x);
            x = c.ggml_reshape_4d(ctx0, x, w, h, ne(tok, 0), b); // [W,H,C,B]

            cur = c.ggml_add(ctx0, res, x);
            _ = c.ggml_format_name(cur, "yasa2_stage%zu_blk%zu_out", s, bi);
        }
    }

    // HF path adds vision position embeddings BEFORE adaptive pooling.
    const pre_w = ne(cur, 0);
    const pre_h = ne(cur, 1);
    var tokens_pre = c.ggml_reshape_3d(ctx0, cur, pre_w * pre_h, ne(cur, 2), ne(cur, 3)); // [T,C,B]
    tokens_pre = c.ggml_permute(ctx0, tokens_pre, 1, 0, 2, 3); // [C,T,B]
    tokens_pre = c.ggml_cont(ctx0, tokens_pre);
    if (model.yasa_vision_pos_embed != null and ne(tokens_pre, 1) == ne(model.yasa_vision_pos_embed, 1)) {
        const n_ch = ne(model.yasa_vision_pos_embed, 0);
        const n_tokens = ne(model.yasa_vision_pos_embed, 1);
        const pos = c.ggml_reshape_3d(ctx0, model.yasa_vision_pos_embed, @as(c_int, @truncate(n_ch)), @as(c_int, @truncate(n_tokens)), 1);
        tokens_pre = c.ggml_add(ctx0, tokens_pre, pos);
    }
    cur = c.ggml_permute(ctx0, tokens_pre, 1, 0, 2, 3); // [T,C,B]
    cur = c.ggml_cont(ctx0, cur);
    cur = c.ggml_reshape_4d(ctx0, cur, pre_w, pre_h, ne(cur, 1), ne(cur, 2)); // [W,H,C,B]

    // AdaptiveAvgPool2d target is 8x8 for real inputs, but warmup can use tiny images.
    const pooled_w: c_int = @min(8, @as(c_int, @truncate(ne(cur, 0))));
    const pooled_h: c_int = @min(8, @as(c_int, @truncate(ne(cur, 1))));
    const kw: c_int = @max(1, @divTrunc(@as(c_int, @truncate(ne(cur, 0))), pooled_w));
    const kh: c_int = @max(1, @divTrunc(@as(c_int, @truncate(ne(cur, 1))), pooled_h));
    cur = c.ggml_pool_2d(ctx0, cur, c.GGML_OP_POOL_AVG, kw, kh, kw, kh, 0, 0);

    // [W,H,C,B] -> [C,T,B]
    var tokens = c.ggml_reshape_3d(ctx0, cur, ne(cur, 0) * ne(cur, 1), ne(cur, 2), ne(cur, 3));
    tokens = c.ggml_permute(ctx0, tokens, 1, 0, 2, 3);
    tokens = c.ggml_cont(ctx0, tokens);
    g.cb(tokens, "yasa2_tokens", -1);

    impl.ggmlAssert(model.mm_0_w != null and model.mm_2_w != null, "yasa2.cpp", 179, "model.mm_0_w && model.mm_2_w");
    const embeddings = g.buildFfn(tokens, model.mm_0_w, model.mm_0_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);
    g.cb(embeddings, "yasa2_emb", -1);

    c.ggml_build_forward_expand(g.gf, embeddings);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
