//! The FastConformer audio encoder (LFM2-Audio): conv subsampling, then
//! macaron conformer layers with relative-position attention, then the
//! audio adapter.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/conformer.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const ne = graph.ne;
const nb = graph.nb;

/// Ports `build` (conformer.cpp:3 @c1d0e7a00), a member of
/// `clip_graph_conformer`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const d_head = g.d_head;
    const n_head = g.n_head;

    const n_frames = g.img.nx();
    const n_pos = @divTrunc(n_frames, 2);
    const n_pos_embd = @divTrunc(@divTrunc(@divTrunc(n_frames + 1, 2) + 1, 2) + 1, 2) * 2 - 1;
    impl.ggmlAssert(ne(model.position_embeddings, 1) >= n_pos, "conformer.cpp", 7, "model.position_embeddings->ne[1] >= n_pos");

    const pos_emb = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 512, n_pos_embd);
    _ = c.ggml_set_name(pos_emb, "pos_emb");
    c.ggml_set_input(pos_emb);
    c.ggml_build_forward_expand(g.gf, pos_emb);

    const inp = g.buildInpRaw(1);

    var cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp));

    // pre encode, conv subsampling
    {
        // layer.0 - conv2d
        cur = c.ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], cur, 2, 2, 1, 1, 1, 1);
        cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[0]);
        g.cb(cur, "conformer.pre_encode.conv.{}", 0);

        // layer.1 - relu
        cur = c.ggml_relu_inplace(ctx0, cur);

        // layer.2 conv2d dw
        cur = c.ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[2], cur, 2, 2, 1, 1, 1, 1);
        cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[2]);
        g.cb(cur, "conformer.pre_encode.conv.{}", 2);

        // layer.3 conv2d
        cur = c.ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
        cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
        g.cb(cur, "conformer.pre_encode.conv.{}", 3);

        // layer.4 - relu
        cur = c.ggml_relu_inplace(ctx0, cur);

        // layer.5 conv2d dw
        cur = c.ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[5], cur, 2, 2, 1, 1, 1, 1);
        cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[5]);
        g.cb(cur, "conformer.pre_encode.conv.{}", 5);

        // layer.6 conv2d
        cur = c.ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
        cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
        g.cb(cur, "conformer.pre_encode.conv.{}", 6);

        // layer.7 - relu
        cur = c.ggml_relu_inplace(ctx0, cur);

        // flatten channel and frequency axis
        cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 0, 2, 1, 3));
        cur = c.ggml_reshape_2d(ctx0, cur, ne(cur, 0) * ne(cur, 1), ne(cur, 2));

        // calculate out
        cur = g.buildMm(model.pre_encode_out_w, cur);
        cur = c.ggml_add(ctx0, cur, model.pre_encode_out_b);
        g.cb(cur, "conformer.pre_encode.out", -1);
    }

    // pos_emb
    g.cb(pos_emb, "pos_emb", -1);

    const fc_factor: f32 = 0.5;

    var il: c_int = 0;
    while (il < hparams.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];

        var residual = cur;

        g.cb(cur, "layer.in", il);

        // feed_forward1
        cur = g.buildNorm(cur, layer.ff_norm_w, layer.ff_norm_b, .normal, 1e-5, il);
        g.cb(cur, "conformer.layers.{}.norm_feed_forward1", il);

        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, null, null, layer.ff_down_w, layer.ff_down_b, .silu, il);
        g.cb(cur, "conformer.layers.{}.feed_forward1.linear2", il);

        residual = c.ggml_add(ctx0, residual, c.ggml_scale(ctx0, cur, fc_factor));

        // self-attention
        {
            cur = g.buildNorm(residual, layer.ln_1_w, layer.ln_1_b, .normal, 1e-5, il);
            g.cb(cur, "conformer.layers.{}.norm_self_att", il);

            var Qcur = g.buildMm(layer.q_w, cur);
            Qcur = c.ggml_add(ctx0, Qcur, layer.q_b);
            Qcur = c.ggml_reshape_3d(ctx0, Qcur, d_head, n_head, ne(Qcur, 1));
            var Q_bias_u = c.ggml_add(ctx0, Qcur, layer.pos_bias_u);
            Q_bias_u = c.ggml_permute(ctx0, Q_bias_u, 0, 2, 1, 3);
            var Q_bias_v = c.ggml_add(ctx0, Qcur, layer.pos_bias_v);
            Q_bias_v = c.ggml_permute(ctx0, Q_bias_v, 0, 2, 1, 3);

            // TODO @ngxson : some cont can/should be removed when ggml_mul_mat support these cases
            var Kcur = g.buildMm(layer.k_w, cur);
            Kcur = c.ggml_add(ctx0, Kcur, layer.k_b);
            Kcur = c.ggml_reshape_3d(ctx0, Kcur, d_head, n_head, ne(Kcur, 1));
            Kcur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            var Vcur = g.buildMm(layer.v_w, cur);
            Vcur = c.ggml_add(ctx0, Vcur, layer.v_b);
            Vcur = c.ggml_reshape_3d(ctx0, Vcur, d_head, n_head, ne(Vcur, 1));
            Vcur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, Vcur, 1, 2, 0, 3));

            // build_attn won't fit due to matrix_ac and matrix_bd separation
            var matrix_ac = c.ggml_mul_mat(ctx0, Q_bias_u, Kcur);
            matrix_ac = c.ggml_cont(ctx0, c.ggml_permute(ctx0, matrix_ac, 1, 0, 2, 3));
            g.cb(matrix_ac, "conformer.layers.{}.self_attn.id3", il);

            var p = g.buildMm(layer.linear_pos_w, pos_emb);
            g.cb(p, "conformer.layers.{}.self_attn.linear_pos", il);
            p = c.ggml_reshape_3d(ctx0, p, d_head, n_head, ne(p, 1));
            p = c.ggml_permute(ctx0, p, 0, 2, 1, 3);

            var matrix_bd = c.ggml_mul_mat(ctx0, Q_bias_v, p);
            matrix_bd = c.ggml_cont(ctx0, c.ggml_permute(ctx0, matrix_bd, 1, 0, 2, 3));

            // rel shift
            {
                const pos_len = ne(matrix_bd, 0);
                const q_len = ne(matrix_bd, 1);
                const h = ne(matrix_bd, 2);
                matrix_bd = c.ggml_pad(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd = c.ggml_roll(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd = c.ggml_reshape_3d(ctx0, matrix_bd, q_len, pos_len + 1, h);
                matrix_bd = c.ggml_view_3d(ctx0, matrix_bd, q_len, pos_len, h, nb(matrix_bd, 1), nb(matrix_bd, 2), nb(matrix_bd, 0) * @as(usize, @intCast(q_len)));
                matrix_bd = c.ggml_cont_3d(ctx0, matrix_bd, pos_len, q_len, h);
            }

            matrix_bd = c.ggml_view_3d(ctx0, matrix_bd, ne(matrix_ac, 0), ne(matrix_bd, 1), ne(matrix_bd, 2), nb(matrix_bd, 1), nb(matrix_bd, 2), 0);
            var scores = c.ggml_add(ctx0, matrix_ac, matrix_bd);
            // std::sqrt of an int is the double overload: the quotient is a double, rounded once to float
            scores = c.ggml_scale(ctx0, scores, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(d_head)))));
            g.cb(scores, "conformer.layers.{}.self_attn.id0", il);

            const attn = c.ggml_soft_max(ctx0, scores);
            var x = c.ggml_mul_mat(ctx0, attn, Vcur);
            x = c.ggml_permute(ctx0, x, 2, 0, 1, 3);
            x = c.ggml_cont_2d(ctx0, x, ne(x, 0) * ne(x, 1), ne(x, 2));

            var out = g.buildMm(layer.o_w, x);
            out = c.ggml_add(ctx0, out, layer.o_b);
            g.cb(out, "conformer.layers.{}.self_attn.linear_out", il);

            cur = out;
        }

        residual = c.ggml_add(ctx0, residual, cur);
        cur = g.buildNorm(residual, layer.norm_conv_w, layer.norm_conv_b, .normal, 1e-5, il);
        g.cb(cur, "conformer.layers.{}.norm_conv", il);

        // conv
        {
            var x = cur;
            x = g.buildMm(layer.conv_pw1_w, x);
            x = c.ggml_add(ctx0, x, layer.conv_pw1_b);
            g.cb(x, "conformer.layers.{}.conv.pointwise_conv1", il);

            // ggml_glu doesn't support sigmoid
            // TODO @ngxson : support this ops in ggml
            {
                const d = @divTrunc(ne(x, 0), 2);
                const gate = c.ggml_sigmoid(ctx0, c.ggml_view_2d(ctx0, x, d, ne(x, 1), nb(x, 1), @as(usize, @intCast(d)) * nb(x, 0)));
                x = c.ggml_mul(ctx0, c.ggml_view_2d(ctx0, x, d, ne(x, 1), nb(x, 1), 0), gate);
                x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x));
            }

            // use ggml_ssm_conv for f32 precision
            x = c.ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = c.ggml_roll(ctx0, x, 4, 0, 0, 0);
            x = c.ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = c.ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            x = c.ggml_add(ctx0, x, layer.conv_dw_b);

            x = c.ggml_add(ctx0, c.ggml_mul(ctx0, x, layer.conv_norm_w), layer.conv_norm_b);
            x = c.ggml_silu(ctx0, x);

            // pointwise_conv2
            x = g.buildMm(layer.conv_pw2_w, x);
            x = c.ggml_add(ctx0, x, layer.conv_pw2_b);

            cur = x;
        }

        residual = c.ggml_add(ctx0, residual, cur);

        cur = g.buildNorm(residual, layer.ff_norm_1_w, layer.ff_norm_1_b, .normal, 1e-5, il);
        g.cb(cur, "conformer.layers.{}.norm_feed_forward2", il);

        cur = g.buildFfn(cur, layer.ff_up_1_w, layer.ff_up_1_b, null, null, layer.ff_down_1_w, layer.ff_down_1_b, .silu, il); // TODO(tarek): read activation for ffn from hparams
        g.cb(cur, "conformer.layers.{}.feed_forward2.linear2", il);

        residual = c.ggml_add(ctx0, residual, c.ggml_scale(ctx0, cur, fc_factor));
        g.cb(residual, "conformer.layers.{}.conv.id", il);

        cur = g.buildNorm(residual, layer.ln_2_w, layer.ln_2_b, .normal, 1e-5, il);
        g.cb(cur, "conformer.layers.{}.norm_out", il);
    }

    // audio adapter
    cur = g.buildNorm(cur, model.mm_0_w, model.mm_0_b, .normal, 1e-5, -1);
    g.cb(cur, "audio_adapter.model.{}", 0);
    cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_3_w, model.mm_3_b, .gelu_erf, -1);

    g.cb(cur, "projected", -1);

    c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
