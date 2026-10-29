//! The Granite Speech encoder: a block-local Conformer with Shaw relative
//! positions and a mid-stack CTC branch, then a windowed QFormer projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/granite-speech.cpp` at v0.4.1
//! (`b29c606e2`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const T = graph.T;
const gpa = impl.gpa;
const ne = graph.ne;
const nb = graph.nb;

extern fn sqrtf(x: f32) f32;

/// Ports `build` (granite-speech.cpp:5 @b29c606e2), a member of
/// `clip_graph_granite_speech`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_layer = g.n_layer;
    const n_embd = g.n_embd;
    const n_head = g.n_head;
    const d_head = g.d_head;

    const n_frames = g.img.nx();
    const context_size = hparams.audio_chunk_size;
    const ctc_layer = @divTrunc(n_layer, 2);
    const conv_kernel = hparams.audio_conv_kernel_size;
    const conv_pad = @divTrunc(conv_kernel, 2);

    const num_blocks = @divTrunc(n_frames + context_size - 1, context_size);
    const padded_len = num_blocks * context_size;
    const remainder = @rem(n_frames, context_size);

    // Calculate projector input dimension based on feature layers
    const proj_input_dim: c_int = @truncate(@as(i64, n_embd) *% @as(i64, @intCast(hparams.feature_layers.items.len + 1)));
    const use_feature_concat = hparams.feature_layers.items.len != 0;

    const attn_dists = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, @as(i64, context_size) * context_size);
    _ = c.ggml_set_name(attn_dists, "attn_dists");
    c.ggml_set_input(attn_dists);

    var attn_mask: T = null;
    if (remainder > 0) {
        attn_mask = c.ggml_new_tensor_4d(ctx0, c.GGML_TYPE_F32, context_size, context_size, 1, num_blocks);
        _ = c.ggml_set_name(attn_mask, "attn_mask");
        c.ggml_set_input(attn_mask);
    }

    const inp = g.buildInpRaw(1);
    var cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp));
    g.cb(cur, "inp_transposed", -1);

    cur = g.buildMm(model.inp_proj_w, cur);
    cur = c.ggml_add(ctx0, cur, model.inp_proj_b);
    g.cb(cur, "inp_linear", -1);

    // Capture layer 0 if requested (after input_linear)
    var concat_result: T = null;
    if (use_feature_concat) {
        if (std.mem.indexOfScalar(i32, hparams.feature_layers.items, 0) != null) {
            concat_result = cur;
            g.cb(concat_result, "feature_layer_0", -1);
        }
    }

    var il: c_int = 0;
    while (il < n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        var residual = cur;

        // ffn1 (half-step)
        {
            var ffn1 = g.buildNorm(cur, layer.ff_norm_w, layer.ff_norm_b, .normal, g.eps, il);
            g.cb(ffn1, "ffn1_norm", il);

            ffn1 = g.buildFfn(ffn1, layer.ff_up_w, layer.ff_up_b, null, null, layer.ff_down_w, layer.ff_down_b, .silu, il);
            g.cb(ffn1, "ffn1_out", il);

            residual = c.ggml_add(ctx0, residual, c.ggml_scale(ctx0, ffn1, 0.5));
            g.cb(residual, "ffn1_residual", il);
        }

        // build_attn not used here: Shaw RPE needs pos_attn = mul_mat(pos_emb, Q)
        // injected between KQ product and softmax, which build_attn doesn't support
        {
            var normed = g.buildNorm(residual, layer.ln_1_w, layer.ln_1_b, .normal, g.eps, il);
            g.cb(normed, "attn_norm", il);

            if (n_frames < padded_len) {
                normed = c.ggml_pad(ctx0, normed, 0, padded_len - n_frames, 0, 0);
            }

            var Q = g.buildMm(layer.q_w, normed);
            var K = g.buildMm(layer.k_w, normed);
            var V = g.buildMm(layer.v_w, normed);

            Q = c.ggml_reshape_4d(ctx0, Q, d_head, n_head, context_size, num_blocks);
            K = c.ggml_reshape_4d(ctx0, K, d_head, n_head, context_size, num_blocks);
            V = c.ggml_reshape_4d(ctx0, V, d_head, n_head, context_size, num_blocks);

            const Q_perm = c.ggml_permute(ctx0, Q, 0, 2, 1, 3);
            const K_perm = c.ggml_cont(ctx0, c.ggml_permute(ctx0, K, 0, 2, 1, 3));

            const kq = c.ggml_mul_mat(ctx0, K_perm, Q_perm);

            // Shaw RPE: pos_emb ne[2]=1 broadcasts against Q ne[2]=num_blocks in mul_mat
            var pos_emb = c.ggml_get_rows(ctx0, layer.attn_rel_pos_emb, attn_dists);
            pos_emb = c.ggml_reshape_3d(ctx0, pos_emb, d_head, context_size, context_size);
            pos_emb = c.ggml_reshape_4d(ctx0, pos_emb, d_head, context_size, 1, context_size);

            const Q_shaw = c.ggml_permute(ctx0, Q, 0, 1, 3, 2);
            var pos_attn = c.ggml_mul_mat(ctx0, pos_emb, Q_shaw);
            pos_attn = c.ggml_cont(ctx0, c.ggml_permute(ctx0, pos_attn, 0, 2, 3, 1));

            const scores = c.ggml_add(ctx0, kq, pos_attn);
            const attn_weights = c.ggml_soft_max_ext(ctx0, scores, attn_mask, g.kq_scale, 0.0);

            const V_perm = c.ggml_cont(ctx0, c.ggml_permute(ctx0, V, 1, 2, 0, 3));
            var attn_out = c.ggml_mul_mat(ctx0, V_perm, attn_weights);

            attn_out = c.ggml_permute(ctx0, attn_out, 0, 2, 1, 3);
            attn_out = c.ggml_cont_2d(ctx0, attn_out, n_embd, padded_len);

            if (n_frames < padded_len) {
                attn_out = c.ggml_view_2d(ctx0, attn_out, n_embd, n_frames, nb(attn_out, 1), 0);
            }

            cur = g.buildMm(layer.o_w, attn_out);
            cur = c.ggml_add(ctx0, cur, layer.o_b);
            g.cb(cur, "attn_out", il);
        }

        residual = c.ggml_add(ctx0, residual, cur);

        // conv module
        {
            cur = g.buildNorm(residual, layer.norm_conv_w, layer.norm_conv_b, .normal, g.eps, il);
            g.cb(cur, "conv_norm", il);

            var x = g.buildMm(layer.conv_pw1_w, cur);
            x = c.ggml_add(ctx0, x, layer.conv_pw1_b);
            g.cb(x, "conv_pw1", il);

            // GLU: ggml has no fused op, manual split + sigmoid gate
            {
                const d = @divTrunc(ne(x, 0), 2);
                const gate = c.ggml_sigmoid(ctx0, c.ggml_view_2d(ctx0, x, d, ne(x, 1), nb(x, 1), @as(usize, @intCast(d)) * nb(x, 0)));
                x = c.ggml_mul(ctx0, c.ggml_view_2d(ctx0, x, d, ne(x, 1), nb(x, 1), 0), gate);
                x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x));
            }
            g.cb(x, "conv_glu", il);

            x = c.ggml_pad(ctx0, x, conv_pad, 0, 0, 0);
            x = c.ggml_roll(ctx0, x, conv_pad, 0, 0, 0);
            x = c.ggml_pad(ctx0, x, conv_pad, 0, 0, 0);
            x = c.ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            g.cb(x, "conv_dw", il);

            // folded batch norm
            x = c.ggml_add(ctx0, c.ggml_mul(ctx0, x, layer.conv_norm_w), layer.conv_norm_b);
            x = c.ggml_silu(ctx0, x);
            g.cb(x, "conv_bn_silu", il);

            x = g.buildMm(layer.conv_pw2_w, x);
            x = c.ggml_add(ctx0, x, layer.conv_pw2_b);
            g.cb(x, "conv_pw2", il);

            cur = x;
        }

        residual = c.ggml_add(ctx0, residual, cur);

        // ffn2 (half-step)
        {
            var ffn2 = g.buildNorm(residual, layer.ff_norm_1_w, layer.ff_norm_1_b, .normal, g.eps, il);
            g.cb(ffn2, "ffn2_norm", il);

            ffn2 = g.buildFfn(ffn2, layer.ff_up_1_w, layer.ff_up_1_b, null, null, layer.ff_down_1_w, layer.ff_down_1_b, .silu, il);
            g.cb(ffn2, "ffn2_out", il);

            residual = c.ggml_add(ctx0, residual, c.ggml_scale(ctx0, ffn2, 0.5));
        }

        cur = g.buildNorm(residual, layer.ln_2_w, layer.ln_2_b, .normal, g.eps, il);
        g.cb(cur, "layer_out", il);

        // Capture intermediate layer (il + 1) if requested
        if (use_feature_concat) {
            if (hparams.isFeatureLayer(il + 1)) {
                if (concat_result == null) {
                    concat_result = cur;
                } else {
                    concat_result = c.ggml_concat(ctx0, concat_result, cur, 0);
                }
                const name = impl.stringFormat(gpa, "feature_layer_%d", .{il + 1}) catch impl.abort("granite-speech.cpp", 194, "out of memory");
                defer gpa.free(name);
                g.cb(concat_result, name.ptr, il);
            }
        }

        // CTC branch
        if (il + 1 == ctc_layer) {
            var mid = g.buildMm(model.ctc_out_w, cur);
            mid = c.ggml_add(ctx0, mid, model.ctc_out_b);
            mid = c.ggml_soft_max(ctx0, mid);
            mid = g.buildMm(model.ctc_out_mid_w, mid);
            mid = c.ggml_add(ctx0, mid, model.ctc_out_mid_b);
            cur = c.ggml_add(ctx0, cur, mid);
            g.cb(cur, "ctc_branch", il);
        }
    }

    // Append final output to concatenated features if using feature concatenation
    if (use_feature_concat and concat_result != null) {
        concat_result = c.ggml_concat(ctx0, concat_result, cur, 0);
        g.cb(concat_result, "concat_final", -1);
        cur = concat_result;
    }

    g.cb(cur, "encoder_out", -1);

    // QFormer projector
    {
        const window_size = hparams.audio_proj_window_size;
        const num_queries = @divTrunc(window_size, hparams.audio_proj_downsample_rate);
        const proj_n_head = hparams.audio_proj_head_count;
        const proj_d_head = @divTrunc(n_embd, proj_n_head);
        const proj_kq_scale = 1.0 / sqrtf(@floatFromInt(proj_d_head));
        const proj_eps: f32 = 1e-12;
        const nblocks_proj = @divTrunc(n_frames + window_size - 1, window_size);
        const padded_proj = nblocks_proj * window_size;

        if (n_frames < padded_proj) {
            cur = c.ggml_pad(ctx0, cur, 0, padded_proj - n_frames, 0, 0);
        }

        const enc_windows = c.ggml_reshape_3d(ctx0, cur, proj_input_dim, window_size, nblocks_proj);

        const qf = &model.qf_proj_blocks.items[0];
        var queries = g.buildNorm(qf.qf_proj_query, qf.qf_proj_norm_w, qf.qf_proj_norm_b, .normal, proj_eps, -1);
        {
            const q_3d = c.ggml_reshape_3d(ctx0, queries, n_embd, num_queries, 1);
            const q_shape = c.ggml_new_tensor_3d(ctx0, c.GGML_TYPE_F32, n_embd, num_queries, nblocks_proj);
            queries = c.ggml_repeat(ctx0, q_3d, q_shape);
        }

        for (qf.qf_proj_layers.items, 0..) |*pl, pil| {
            const jl: c_int = @intCast(pil);

            // self-attention
            {
                var Q = c.ggml_add(ctx0, g.buildMm(pl.q_w, queries), pl.q_b);
                var K = c.ggml_add(ctx0, g.buildMm(pl.k_w, queries), pl.k_b);
                var V = c.ggml_add(ctx0, g.buildMm(pl.v_w, queries), pl.v_b);

                Q = c.ggml_reshape_4d(ctx0, Q, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                K = c.ggml_reshape_4d(ctx0, K, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                V = c.ggml_reshape_4d(ctx0, V, proj_d_head, proj_n_head, num_queries, nblocks_proj);

                var sa_out = g.buildAttn(pl.o_w, pl.o_b, Q, K, V, null, proj_kq_scale, jl, null);
                sa_out = c.ggml_reshape_3d(ctx0, sa_out, n_embd, num_queries, nblocks_proj);

                queries = g.buildNorm(c.ggml_add(ctx0, sa_out, queries), pl.ln_1_w, pl.ln_1_b, .normal, proj_eps, jl);
            }

            // cross-attention
            {
                var Q = c.ggml_add(ctx0, g.buildMm(pl.cross_attn_q_w, queries), pl.cross_attn_q_b);
                var K = c.ggml_add(ctx0, g.buildMm(pl.cross_attn_k_w, enc_windows), pl.cross_attn_k_b);
                var V = c.ggml_add(ctx0, g.buildMm(pl.cross_attn_v_w, enc_windows), pl.cross_attn_v_b);

                Q = c.ggml_reshape_4d(ctx0, Q, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                K = c.ggml_reshape_4d(ctx0, K, proj_d_head, proj_n_head, window_size, nblocks_proj);
                V = c.ggml_reshape_4d(ctx0, V, proj_d_head, proj_n_head, window_size, nblocks_proj);

                var ca_out = g.buildAttn(pl.cross_attn_o_w, pl.cross_attn_o_b, Q, K, V, null, proj_kq_scale, jl, null);
                ca_out = c.ggml_reshape_3d(ctx0, ca_out, n_embd, num_queries, nblocks_proj);

                queries = g.buildNorm(c.ggml_add(ctx0, ca_out, queries), pl.cross_attn_norm_w, pl.cross_attn_norm_b, .normal, proj_eps, jl);
            }

            // ffn
            {
                const ffn_out = g.buildFfn(queries, pl.ff_up_w, pl.ff_up_b, null, null, pl.ff_down_w, pl.ff_down_b, .gelu, jl);

                queries = g.buildNorm(c.ggml_add(ctx0, ffn_out, queries), pl.ln_2_w, pl.ln_2_b, .normal, proj_eps, jl);
            }
        }

        cur = c.ggml_reshape_2d(ctx0, queries, n_embd, @as(i64, num_queries) * nblocks_proj);
        cur = c.ggml_add(ctx0, g.buildMm(qf.qf_proj_linear_w, cur), qf.qf_proj_linear_b);
        g.cb(cur, "projector_out", -1);
    }

    c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
