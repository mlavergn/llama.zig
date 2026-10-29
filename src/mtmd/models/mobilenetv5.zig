//! The MobileNetV5 vision encoder of Gemma 3n: edge-residual, universal
//! inverted-residual and multi-query attention blocks, the multi-scale
//! fusion adapter, and the Gemma 3n embedder.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/mobilenetv5.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;

const T = graph.T;
const Block = model_mod.MobilenetV5Block;

extern fn sqrtf(x: f32) f32;

fn neInt(t: T, i: usize) c_int {
    return @intCast(graph.ne(t, i));
}

/// Ports `rms_norm_2d` (mobilenetv5.cpp:5 @b29c606e2), a member of
/// `clip_graph_mobilenetv5`: normalizes over channels for each spatial
/// position of a [W, H, C, B] tensor. `eps` defaults to 1e-6 in the C++.
pub fn rmsNorm2d(g: *graph.Graph, inp: T, weight: T, eps: f32) T {
    const ctx0 = g.ctx0;
    var cur = c.ggml_permute(ctx0, inp, 2, 1, 0, 3);
    cur = c.ggml_cont(ctx0, cur);
    cur = c.ggml_rms_norm(ctx0, cur, eps);

    if (weight != null) cur = c.ggml_mul(ctx0, cur, weight);

    cur = c.ggml_permute(ctx0, cur, 2, 1, 0, 3);
    cur = c.ggml_cont(ctx0, cur);

    return cur;
}

/// Ports `pad_same_2d` (mobilenetv5.cpp:23 @b29c606e2), a member of
/// `clip_graph_mobilenetv5`: Conv2dSame's asymmetric SAME padding.
/// Dilations default to 1 in the C++.
pub fn padSame2d(g: *graph.Graph, inp_in: T, kernel_h: c_int, kernel_w: c_int, stride_h: c_int, stride_w: c_int, dilation_h: c_int, dilation_w: c_int) T {
    var inp = inp_in;
    const ih = graph.ne(inp, 1); // height
    const iw = graph.ne(inp, 0); // width

    // Calculate output size (ceil division)
    const oh = @divTrunc(ih + stride_h - 1, stride_h);
    const ow = @divTrunc(iw + stride_w - 1, stride_w);

    // Calculate padding needed
    const pad_h: i64 = @max(0, (oh - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - ih);
    const pad_w: i64 = @max(0, (ow - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - iw);

    // Split padding asymmetrically
    const pad_h_top: c_int = @intCast(@divTrunc(pad_h, 2));
    const pad_h_bottom: c_int = @intCast(pad_h - pad_h_top);
    const pad_w_left: c_int = @intCast(@divTrunc(pad_w, 2));
    const pad_w_right: c_int = @intCast(pad_w - pad_w_left);

    // For [W, H, C, B]: p0=width, p1=height, p2=channels, p3=batch
    if (pad_h > 0 or pad_w > 0) {
        inp = c.ggml_pad_ext(g.ctx0, inp, pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, 0, 0, 0, 0);
    }

    return inp;
}

/// Ports `build_edge_residual` (mobilenetv5.cpp:57 @b29c606e2), a member
/// of `clip_graph_mobilenetv5`: the stage-0 block.
pub fn buildEdgeResidual(g: *graph.Graph, inp: T, block: *const Block, stride: c_int) T {
    const ctx0 = g.ctx0;
    var cur = inp;

    // 1. Expansion Conv (3x3)
    if (stride == 2) {
        // Downsampling (Block 0): Conv2dSame(kernel=3, stride=2)
        cur = padSame2d(g, cur, 3, 3, stride, stride, 1, 1);
        cur = c.ggml_conv_2d_direct(ctx0, block.s0_conv_exp_w, cur, stride, stride, 0, 0, 1, 1);
    } else {
        // Normal 3x3 Block (Block 1, 2): Conv2d(kernel=3, stride=1, padding=1)
        cur = c.ggml_conv_2d_direct(ctx0, block.s0_conv_exp_w, cur, stride, stride, 1, 1, 1, 1);
    }

    // BN + Activation
    if (block.s0_bn1_w != null) cur = rmsNorm2d(g, cur, block.s0_bn1_w, 1e-6);
    cur = c.ggml_gelu(ctx0, cur);

    // 2. Pointwise Linear Conv (1x1)
    cur = c.ggml_conv_2d_direct(ctx0, block.s0_conv_pwl_w, cur, 1, 1, 0, 0, 1, 1);
    if (block.s0_bn2_w != null) cur = rmsNorm2d(g, cur, block.s0_bn2_w, 1e-6);

    // 3. Residual Connection, only where spatial dimensions and channels match (stride 1)
    if (stride == 1 and graph.ne(inp, 2) == graph.ne(cur, 2) and graph.ne(inp, 0) == graph.ne(cur, 0)) {
        cur = c.ggml_add(ctx0, cur, inp);
    }

    return cur;
}

/// Ports `build_inverted_residual` (mobilenetv5.cpp:91 @b29c606e2), a
/// member of `clip_graph_mobilenetv5`: the universal inverted-residual
/// block of stage 1 onward.
pub fn buildInvertedResidual(g: *graph.Graph, inp: T, block: *const Block, stride: c_int) T {
    const ctx0 = g.ctx0;
    var cur = inp;

    // 1. Depthwise Start (Optional), always stride 1
    if (block.dw_start_w != null) {
        const k = neInt(block.dw_start_w, 0); // 3 or 5
        const p = @divTrunc(k, 2);
        cur = c.ggml_conv_2d_dw(ctx0, block.dw_start_w, cur, 1, 1, p, p, 1, 1);
        if (block.dw_start_bn_w != null) cur = rmsNorm2d(g, cur, block.dw_start_bn_w, 1e-6);
    }

    // 2. Pointwise Expansion (1x1)
    if (block.pw_exp_w != null) {
        cur = c.ggml_conv_2d_direct(ctx0, block.pw_exp_w, cur, 1, 1, 0, 0, 1, 1);
        if (block.pw_exp_bn_w != null) cur = rmsNorm2d(g, cur, block.pw_exp_bn_w, 1e-6);
        cur = c.ggml_gelu(ctx0, cur);
    }

    // 3. Depthwise Mid (Optional), where downsampling happens
    if (block.dw_mid_w != null) {
        const k = neInt(block.dw_mid_w, 0); // 3 or 5

        if (stride > 1) {
            // Stride 2 (Downsample) -> Asymmetric "Same" Padding
            cur = padSame2d(g, cur, k, k, stride, stride, 1, 1);
            cur = c.ggml_conv_2d_dw(ctx0, block.dw_mid_w, cur, stride, stride, 0, 0, 1, 1); // pad=0
        } else {
            // Stride 1 -> Standard Symmetric Padding
            const p = @divTrunc(k, 2);
            cur = c.ggml_conv_2d_dw(ctx0, block.dw_mid_w, cur, stride, stride, p, p, 1, 1);
        }

        if (block.dw_mid_bn_w != null) cur = rmsNorm2d(g, cur, block.dw_mid_bn_w, 1e-6);
        cur = c.ggml_gelu(ctx0, cur);
    }

    // 4. Pointwise Projection (1x1)
    if (block.pw_proj_w != null) {
        cur = c.ggml_conv_2d_direct(ctx0, block.pw_proj_w, cur, 1, 1, 0, 0, 1, 1);
        if (block.pw_proj_bn_w != null) cur = rmsNorm2d(g, cur, block.pw_proj_bn_w, 1e-6);
    }

    // Apply Layer Scaling if present
    if (block.layer_scale_w != null) cur = c.ggml_mul(ctx0, cur, block.layer_scale_w);

    // 5. Residual Connection
    const same_spatial = (graph.ne(inp, 0) == graph.ne(cur, 0)) and (graph.ne(inp, 1) == graph.ne(cur, 1));
    const same_channel = (graph.ne(inp, 2) == graph.ne(cur, 2));
    if (same_spatial and same_channel) cur = c.ggml_add(ctx0, cur, inp);

    return cur;
}

/// Ports `build_mobilenet_attn` (mobilenetv5.cpp:152 @b29c606e2), a member
/// of `clip_graph_mobilenetv5`: multi-query attention with downsampled K
/// and V.
pub fn buildMobilenetAttn(g: *graph.Graph, inp: T, block: *const Block) T {
    const ctx0 = g.ctx0;
    var cur = inp;

    // Norm
    if (block.attn_norm_w != null) cur = rmsNorm2d(g, cur, block.attn_norm_w, 1e-6);

    // 1. Q Calculation
    var q = c.ggml_conv_2d_direct(ctx0, block.attn_q_w, cur, 1, 1, 0, 0, 1, 1);

    // 2. K Calculation (Downsampled), Conv2dSame(kernel 3, stride 2, depthwise)
    var k_inp = cur;
    if (block.attn_k_dw_w != null) {
        const k_size = neInt(block.attn_k_dw_w, 0); // Usually 3
        k_inp = padSame2d(g, cur, k_size, k_size, 2, 2, 1, 1); // Apply SAME padding
        k_inp = c.ggml_conv_2d_dw(ctx0, block.attn_k_dw_w, k_inp, 2, 2, 0, 0, 1, 1); // padding=0
        if (block.attn_k_norm_w != null) k_inp = rmsNorm2d(g, k_inp, block.attn_k_norm_w, 1e-6);
    }
    var k = c.ggml_conv_2d_direct(ctx0, block.attn_k_w, k_inp, 1, 1, 0, 0, 1, 1);

    // 3. V Calculation (Downsampled), Conv2dSame(kernel 3, stride 2, depthwise)
    var v_inp = cur;
    if (block.attn_v_dw_w != null) {
        const v_size = neInt(block.attn_v_dw_w, 0); // Usually 3
        v_inp = padSame2d(g, cur, v_size, v_size, 2, 2, 1, 1); // Apply SAME padding
        v_inp = c.ggml_conv_2d_dw(ctx0, block.attn_v_dw_w, v_inp, 2, 2, 0, 0, 1, 1); // padding=0
        if (block.attn_v_norm_w != null) v_inp = rmsNorm2d(g, v_inp, block.attn_v_norm_w, 1e-6);
    }
    var v = c.ggml_conv_2d_direct(ctx0, block.attn_v_w, v_inp, 1, 1, 0, 0, 1, 1);

    const W = neInt(cur, 0);
    const H = neInt(cur, 1);
    const B = neInt(cur, 3);
    const D = neInt(k, 2); // Head dimension
    const n_head = @divTrunc(neInt(q, 2), D);
    const N = W * H;

    // Process Q: [W, H, D*n_head, B] -> [D, N, n_head, B]
    q = c.ggml_reshape_3d(ctx0, q, N, D * n_head, B);
    q = c.ggml_reshape_4d(ctx0, q, N, D, n_head, B);
    q = c.ggml_permute(ctx0, q, 1, 0, 2, 3); // [D, N, n_head, B]
    q = c.ggml_cont(ctx0, q);

    const Wk = neInt(k, 0);
    const Hk = neInt(k, 1);
    const M = Wk * Hk;

    // Process K: [Wk, Hk, D, B] -> [D, M, 1, B]
    k = c.ggml_reshape_3d(ctx0, k, M, D, B);
    k = c.ggml_reshape_4d(ctx0, k, M, D, 1, B);
    k = c.ggml_permute(ctx0, k, 1, 0, 2, 3); // [D, M, 1, B]
    k = c.ggml_cont(ctx0, k);

    // Process V: [Wk, Hk, D, B] -> [M, D, 1, B]
    v = c.ggml_reshape_3d(ctx0, v, M, D, B);
    v = c.ggml_reshape_4d(ctx0, v, M, D, 1, B);
    v = c.ggml_cont(ctx0, v); // [M, D, 1, B]

    // Multi-Query Attention
    const scale: f32 = 1.0 / sqrtf(@floatFromInt(D));

    // Step 1: Compute Q @ K.T
    var scores = c.ggml_mul_mat(ctx0, k, q);

    scores = c.ggml_scale(ctx0, scores, scale);

    scores = c.ggml_soft_max(ctx0, scores);

    var kqv = c.ggml_mul_mat(ctx0, v, scores);

    kqv = c.ggml_permute(ctx0, kqv, 1, 0, 2, 3);
    kqv = c.ggml_cont(ctx0, kqv);

    kqv = c.ggml_reshape_3d(ctx0, kqv, N, D * n_head, B);
    kqv = c.ggml_reshape_4d(ctx0, kqv, W, H, D * n_head, B);
    kqv = c.ggml_cont(ctx0, kqv);

    // Output projection
    cur = c.ggml_conv_2d_direct(ctx0, block.attn_o_w, kqv, 1, 1, 0, 0, 1, 1);

    // Residual & Layer Scale
    if (graph.ne(inp, 0) == graph.ne(cur, 0) and graph.ne(inp, 2) == graph.ne(cur, 2)) {
        if (block.layer_scale_w != null) cur = c.ggml_mul(ctx0, cur, block.layer_scale_w);
        cur = c.ggml_add(ctx0, cur, inp);
    }

    return cur;
}

/// The `is_stage_start` lambda (mobilenetv5.cpp:266 @b29c606e2).
fn isStageStart(model: *const model_mod.Model, i: c_int) bool {
    if (i == 0) return true;
    for (model.mobilenet_stage_ends.items) |end_idx| {
        if (i == end_idx + 1) return true;
    }
    return false;
}

/// The `is_fusion_point` lambda (mobilenetv5.cpp:274 @b29c606e2).
fn isFusionPoint(model: *const model_mod.Model, i: c_int, total_blocks: c_int) bool {
    const ends = model.mobilenet_stage_ends.items;
    if (ends.len >= 4) {
        if (i == ends[2]) return true; // End of Stage 2
        if (i == ends[3]) return true; // End of Stage 3
    } else {
        if (i == total_blocks - 1) return true;
    }
    return false;
}

/// Ports `build` (mobilenetv5.cpp:248 @b29c606e2), a member of `clip_graph_mobilenetv5`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    const inp = g.buildInpRaw(3);

    // 1. Stem - Conv2dSame(3, 64, kernel_size=(3, 3), stride=(2, 2))
    var cur = padSame2d(g, inp, 3, 3, 2, 2, 1, 1); // Apply SAME padding

    cur = c.ggml_conv_2d_direct(ctx0, model.mobilenet_stem_conv_w, cur, 2, 2, 0, 0, 1, 1); // padding=0
    if (model.mobilenet_stem_conv_b != null) cur = c.ggml_add(ctx0, cur, model.mobilenet_stem_conv_b);
    if (model.mobilenet_stem_norm_w != null) cur = rmsNorm2d(g, cur, model.mobilenet_stem_norm_w, 1e-6);
    cur = c.ggml_gelu(ctx0, cur);

    // 2. Blocks
    var intermediate_features: std.ArrayList(T) = .empty;
    defer intermediate_features.deinit(impl.gpa);
    const total_blocks: c_int = @intCast(model.mobilenet_blocks.items.len);

    var i: c_int = 0;
    while (i < total_blocks) : (i += 1) {
        const block = &model.mobilenet_blocks.items[@intCast(i)];
        const stride: c_int = if (isStageStart(model, i)) 2 else 1;

        if (block.s0_conv_exp_w != null) {
            cur = buildEdgeResidual(g, cur, block, stride);
        } else if (block.attn_q_w != null) {
            cur = buildMobilenetAttn(g, cur, block);
        } else {
            cur = buildInvertedResidual(g, cur, block, stride);
        }

        if (isFusionPoint(model, i, total_blocks)) try intermediate_features.append(impl.gpa, cur);
    }

    // 3. Multi-Scale Fusion Adapter (MSFA)
    if (intermediate_features.items.len != 0) {
        // A. Reference Resolution: intermediate_features[0] is the "High Resolution" target.
        const target_feat = intermediate_features.items[0];
        const high_res_w = neInt(target_feat, 0);
        const high_res_h = neInt(target_feat, 1);

        var resized_feats: std.ArrayList(T) = .empty;
        defer resized_feats.deinit(impl.gpa);

        // B. Resize inputs to match inputs[0] (High Resolution)
        for (intermediate_features.items) |feat_in| {
            var feat = feat_in;
            const feat_w = neInt(feat, 0);
            const feat_h = neInt(feat, 1);

            // PyTorch: if feat_size < high_resolution: interpolate
            if (feat_w < high_res_w or feat_h < high_res_h) {
                // assumes power-of-2 scaling (e.g. 16 -> 32 means scale=2)
                const scale_w = @divTrunc(high_res_w, feat_w);

                // Safety check for non-integer scaling if strictly replicating
                impl.ggmlAssert(@rem(high_res_w, feat_w) == 0, "mobilenetv5.cpp", 325, "high_res_w % feat_w == 0");

                // Upsample (Nearest Neighbor)
                feat = c.ggml_upscale(ctx0, feat, scale_w, c.GGML_SCALE_MODE_NEAREST);
            }
            try resized_feats.append(impl.gpa, feat);
        }

        // C. Concatenate at High Resolution (Channel Dim = 2 in ggml)
        cur = resized_feats.items[0];
        for (resized_feats.items[1..]) |feat| cur = c.ggml_concat(ctx0, cur, feat, 2);

        // D. FFN (UniversalInvertedResidual):
        // Expand Conv -> Norm -> GELU -> Project Conv -> Norm

        // 1. Expansion
        if (model.msfa_ffn_expand_w != null) {
            // 1x1 Conv
            cur = c.ggml_conv_2d_direct(ctx0, model.msfa_ffn_expand_w, cur, 1, 1, 0, 0, 1, 1);

            if (model.msfa_ffn_expand_bn != null) cur = rmsNorm2d(g, cur, model.msfa_ffn_expand_bn, 1e-6);

            cur = c.ggml_gelu(ctx0, cur);
        }

        // 2. Projection (No DW because kernel_size=0)
        if (model.msfa_ffn_project_w != null) {
            // 1x1 Conv
            cur = c.ggml_conv_2d_direct(ctx0, model.msfa_ffn_project_w, cur, 1, 1, 0, 0, 1, 1);

            // UniversalInvertedResidual typically has a norm after projection
            if (model.msfa_ffn_project_bn != null) cur = rmsNorm2d(g, cur, model.msfa_ffn_project_bn, 1e-6);
        }

        // E. Final Downsample to Target Resolution (Output Resolution, e.g. 16x16)
        const target_out_res = 16;
        const current_w = neInt(cur, 0);

        if (current_w > target_out_res) {
            const s = @divTrunc(current_w, target_out_res);

            impl.ggmlAssert(@rem(current_w, target_out_res) == 0, "mobilenetv5.cpp", 376, "current_w % target_out_res == 0");

            // Avg Pool: Kernel=s, Stride=s
            cur = c.ggml_pool_2d(ctx0, cur, c.GGML_OP_POOL_AVG, s, s, s, s, 0, 0);
        }

        // F. Final Norm
        if (model.msfa_concat_norm_w != null) cur = rmsNorm2d(g, cur, model.msfa_concat_norm_w, 1e-6);
    }

    // 4. Gemma 3n Multimodal Projection (Embedder)
    // Input: 'cur' is [Width, Height, Channels, Batch]
    const W = neInt(cur, 0);
    const H = neInt(cur, 1);
    const C = neInt(cur, 2);
    const B = neInt(cur, 3);

    impl.ggmlAssert(C == hparams.n_embd, "mobilenetv5.cpp", 397, "C == hparams.n_embd");

    // 1. Permute and Flatten to [Channels, Tokens, Batch]
    cur = c.ggml_permute(ctx0, cur, 2, 1, 0, 3); // -> [C, H, W, B]
    cur = c.ggml_permute(ctx0, cur, 0, 2, 1, 3); // -> [C, W, H, B]
    cur = c.ggml_cont(ctx0, cur);
    cur = c.ggml_reshape_3d(ctx0, cur, C, W * H, B);
    cur = c.ggml_cont(ctx0, cur);

    // 2. FEATURE SCALING
    // PyTorch: vision_outputs *= self.config.vision_config.hidden_size**0.5
    const scale_factor = sqrtf(@floatFromInt(C));
    cur = c.ggml_scale(ctx0, cur, scale_factor);

    // 3. SOFT EMBEDDING NORM
    // PyTorch: self._norm(x) * self.weight
    {
        const eps: f32 = 1e-6; // Gemma3n uses 1e-6
        cur = c.ggml_rms_norm(ctx0, cur, eps);

        // Weight shape is (2048,) -> Element-wise broadcast multiply
        if (model.mm_soft_emb_norm_w != null) cur = c.ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
    }

    // 4. PROJECTION
    // Weight stored as [out_features, in_features] = [text_hidden_size, vision_hidden_size]
    if (model.mm_input_proj_w != null) cur = g.buildMm(model.mm_input_proj_w, cur);

    // 5. POST PROJECTION NORM
    // Gemma3nRMSNorm(..., with_scale=False): output = rms_norm(x) * 1.0
    {
        const eps: f32 = 1e-6;
        cur = c.ggml_rms_norm(ctx0, cur, eps);

        // If weight is loaded, multiply (should be ~1.0 anyway)
        if (model.mm_post_proj_norm_w != null) cur = c.ggml_mul(ctx0, cur, model.mm_post_proj_norm_w);
    }

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
