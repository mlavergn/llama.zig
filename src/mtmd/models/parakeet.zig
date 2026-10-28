//! The Parakeet (NeMo FastConformer) audio encoder: conv subsampling, then
//! conformer layers with relative-position attention -- full attention, or
//! chunked local attention past a length threshold -- then the sound
//! projection.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/parakeet.cpp` at v0.4.0
//! (`5266f24da`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

/// `PARAKEET_LOCAL_ATTN_THRESHOLD` (parakeet.cpp:3 @5266f24da).
const parakeet_local_attn_threshold: c_int = 8192;
/// `PARAKEET_LOCAL_ATTN_WINDOW` (parakeet.cpp:4 @5266f24da).
const parakeet_local_attn_window: c_int = 128;

/// `1.0f / std::sqrt(d_head)` with an `int` `d_head`: the double overload,
/// so the quotient is a double rounded once to float.
fn invSqrt(d_head: c_int) f32 {
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(d_head))));
}

/// Ports `build` (parakeet.cpp:7 @5266f24da), a member of
/// `clip_graph_parakeet`: conv
/// subsampling and the conformer encoder.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;

    // Conv subsampling
    var inp = g.buildInpRaw(1);
    inp = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp));

    // [freq, time, channels, batch]
    var cur = c.ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], inp, 2, 2, 1, 1, 1, 1);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[0]);
    g.cb(cur, "pre_conv_0", -1);

    cur = c.ggml_relu(ctx0, cur);
    g.cb(cur, "pre_conv_0_relu", -1);

    // [freq, time, channels, batch]
    cur = c.ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[2], cur, 2, 2, 1, 1, 1, 1);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[2]);
    g.cb(cur, "pre_conv_2", -1);

    // [freq, time, channels, batch]
    cur = c.ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
    g.cb(cur, "pre_conv_3", -1);

    cur = c.ggml_relu(ctx0, cur);
    g.cb(cur, "pre_conv_3_relu", -1);

    // [freq, time, channels, batch]
    cur = c.ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[5], cur, 2, 2, 1, 1, 1, 1);
    g.cb(cur, "pre_conv_5_direct", -1);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[5]);
    g.cb(cur, "pre_conv_5", -1);

    // [freq, time, channels, batch]
    cur = c.ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
    g.cb(cur, "pre_conv_6", -1);

    cur = c.ggml_relu(ctx0, cur);
    g.cb(cur, "pre_conv_6_relu", -1);

    // [freq, time, chan]
    cur = c.ggml_permute(ctx0, cur, 0, 2, 1, 3);
    // [freq, chan, time]
    cur = c.ggml_cont(ctx0, cur);

    const n_freq: c_int = @truncate(ne(cur, 0));
    const n_chan: c_int = @truncate(ne(cur, 1));
    const n_frames: c_int = @truncate(ne(cur, 2));

    // [freq, time, chan, batch] -> [(freq * chan), time]
    cur = c.ggml_reshape_2d(ctx0, cur, n_freq * n_chan, n_frames);

    cur = g.buildMm(model.pre_encode_out_w, cur);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_out_b);

    _ = c.ggml_set_name(cur, "pre_enc_out");

    // Encoder

    const hparams = &model.hparams;
    const n_layer = hparams.n_layer;
    const n_state = hparams.n_embd;
    const fc_factor: f32 = 0.5;

    const n_time: c_int = @truncate(ne(cur, 1));
    const local_attn = n_time > parakeet_local_attn_threshold;
    const att_left = if (local_attn) parakeet_local_attn_window else n_time - 1;
    const att_right = if (local_attn) parakeet_local_attn_window else n_time - 1;
    const window_size = if (local_attn) att_left + att_right + 1 else 2 * n_time - 1;
    const d_half = @divTrunc(n_state, 2);
    const mask_dim = if (local_attn) window_size else n_time;

    // mask [key, n_time]
    const attn_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, mask_dim, n_time);
    _ = c.ggml_set_name(attn_mask, "attn_mask");
    c.ggml_set_input(attn_mask);

    var local_mask: T = null;
    if (local_attn) {
        const chunk = att_left + att_right;
        local_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, chunk + window_size - 1, chunk);
        _ = c.ggml_set_name(local_mask, "local_mask");
        c.ggml_set_input(local_mask);
    }

    const pos_freqs = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, d_half);
    _ = c.ggml_set_name(pos_freqs, "pos_freqs");
    c.ggml_set_input(pos_freqs);

    const rel_positions = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 1, window_size);
    _ = c.ggml_set_name(rel_positions, "rel_positions");
    c.ggml_set_input(rel_positions);

    const freqs = c.ggml_repeat_4d(ctx0, pos_freqs, d_half, window_size, 1, 1);
    const theta = c.ggml_mul(ctx0, freqs, rel_positions);

    const sin = c.ggml_reshape_3d(ctx0, c.ggml_sin(ctx0, theta), 1, d_half, window_size);
    const cos = c.ggml_reshape_3d(ctx0, c.ggml_cos(ctx0, theta), 1, d_half, window_size);
    const pos_emb = c.ggml_reshape_2d(ctx0, c.ggml_cont(ctx0, c.ggml_concat(ctx0, sin, cos, 0)), n_state, window_size);
    _ = c.ggml_set_name(pos_emb, "pos_emb");

    var il: c_int = 0;
    while (il < n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        // FFN1
        {
            const residual = cur;
            _ = c.ggml_format_name(cur, "enc_%d_res", il);

            // norm
            cur = c.ggml_norm(ctx0, cur, hparams.eps);
            cur = c.ggml_add(ctx0, c.ggml_mul(ctx0, cur, layer.ff_norm_w), layer.ff_norm_b);
            _ = c.ggml_format_name(cur, "enc_%d_ffn_norm_1", il);

            cur = g.buildFfn(cur, layer.ff_up_w, null, null, null, layer.ff_down_w, null, .silu, il);
            _ = c.ggml_format_name(cur, "enc_%d_ffn_1", il);

            cur = c.ggml_add(ctx0, residual, c.ggml_scale(ctx0, cur, fc_factor));
            _ = c.ggml_format_name(cur, "enc_%d_res_ffn", il);
        }

        // self attention block using relative positional encoding from model.position_embedding.
        {
            // [feat, time_frames, 1, 1]
            const residual = cur;

            cur = c.ggml_norm(ctx0, cur, hparams.eps);
            cur = c.ggml_add(ctx0, c.ggml_mul(ctx0, cur, layer.ln_1_w), layer.ln_1_b);
            _ = c.ggml_format_name(cur, "enc_%d_attn_norm", il);

            const n_head = hparams.n_head;
            const d_head = @divTrunc(n_state, n_head);

            // [feat, time_frames, 1, 1]
            var Q_cur = g.buildMm(layer.q_w, cur);
            var K_cur = g.buildMm(layer.k_w, cur);
            var V_cur = g.buildMm(layer.v_w, cur);

            // [d_head, n_heads, n_time, 1]
            Q_cur = c.ggml_reshape_3d(ctx0, Q_cur, d_head, n_head, n_time);
            K_cur = c.ggml_reshape_3d(ctx0, K_cur, d_head, n_head, n_time);
            V_cur = c.ggml_reshape_3d(ctx0, V_cur, d_head, n_head, n_time);

            // [n_state, window_size]
            var pos = g.buildMm(layer.linear_pos_w, pos_emb);
            // [feat, head, window_size, 1]
            pos = c.ggml_reshape_3d(ctx0, pos, d_head, n_head, ne(pos_emb, 1));
            // [feat, window_size, head, 1]
            pos = c.ggml_cont(ctx0, c.ggml_permute(ctx0, pos, 0, 2, 1, 3));
            _ = c.ggml_format_name(pos, "enc_%d_attn_pos", il);

            if (local_attn) {
                const chunk = att_left + att_right;
                const n_group = @divTrunc(n_time + chunk - 1, chunk);
                const n_time_padded = n_group * chunk;
                const n_kv_chunk = chunk + window_size - 1;
                const n_kv_dense = n_kv_chunk * n_group;
                const need_padding = n_time_padded > n_time;

                Q_cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, Q_cur, 0, 2, 1, 3));
                K_cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, K_cur, 0, 2, 1, 3));
                V_cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, V_cur, 0, 2, 1, 3));

                // content bias
                const bias_u = c.ggml_reshape_3d(ctx0, layer.pos_bias_u, d_head, 1, n_head);
                const Q_u = c.ggml_add(ctx0, Q_cur, bias_u);

                // position bias
                const bias_v = c.ggml_reshape_3d(ctx0, layer.pos_bias_v, d_head, 1, n_head);
                var Q_v = c.ggml_add(ctx0, Q_cur, bias_v);

                // right pad the time dimension
                var Q_u_padded = if (need_padding) c.ggml_pad_ext(ctx0, Q_u, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) else Q_u;
                Q_u_padded = c.ggml_reshape_4d(ctx0, Q_u_padded, d_head, chunk, n_group, n_head);

                // pad front and back for the first and last time frames
                var K_padded = c.ggml_pad_ext(ctx0, K_cur, 0, 0, att_left, att_right, 0, 0, 0, 0);
                if (n_kv_dense > ne(K_padded, 1)) {
                    K_padded = c.ggml_pad_ext(ctx0, K_padded, 0, 0, 0, @truncate(n_kv_dense - ne(K_padded, 1)), 0, 0, 0, 0);
                }

                // sliding window view: each group spans n_kv_chunk keys but steps by chunk
                var K_chunk = c.ggml_view_4d(ctx0, K_padded, d_head, n_kv_chunk, n_group, n_head, nb(K_padded, 1), @as(usize, @intCast(chunk)) * nb(K_padded, 1), nb(K_padded, 2), 0);
                K_chunk = c.ggml_cont(ctx0, K_chunk);

                var content_scores = c.ggml_mul_mat(ctx0, K_chunk, Q_u_padded);

                // trim the dense output down to window_size scores per query
                content_scores = c.ggml_view_4d(ctx0, content_scores, window_size, chunk, n_group, n_head, @as(usize, @intCast(chunk + window_size)) * nb(content_scores, 0), nb(content_scores, 2), nb(content_scores, 3), 0);
                content_scores = c.ggml_cont(ctx0, content_scores);

                // ungroup: [window_size, n_time_padded, n_head]
                content_scores = c.ggml_reshape_3d(ctx0, content_scores, window_size, n_time_padded, n_head);
                if (need_padding) {
                    content_scores = c.ggml_view_3d(ctx0, content_scores, window_size, n_time, n_head, nb(content_scores, 1), nb(content_scores, 2), 0);
                }

                // Q_v: [d_head, time, head]
                Q_v = c.ggml_cont(ctx0, c.ggml_permute(ctx0, Q_v, 0, 2, 1, 3));
                const rel_pos_scores = c.ggml_mul_mat(ctx0, pos, Q_v);

                var attn_scores = c.ggml_add(ctx0, content_scores, rel_pos_scores);
                attn_scores = c.ggml_soft_max_ext(ctx0, attn_scores, attn_mask, invSqrt(d_head), 0.0);
                _ = c.ggml_format_name(attn_scores, "enc_%d_attn_probs", il);

                // expand probs back to n_kv_chunk width for the V matmul
                var probs_padded = if (need_padding) c.ggml_pad_ext(ctx0, attn_scores, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) else attn_scores;

                probs_padded = c.ggml_reshape_4d(ctx0, probs_padded, window_size, chunk, n_group, n_head);
                probs_padded = c.ggml_pad_ext(ctx0, probs_padded, 0, chunk, 0, 0, 0, 0, 0, 0);
                probs_padded = c.ggml_view_4d(ctx0, probs_padded, n_kv_chunk, chunk, n_group, n_head, @as(usize, @intCast(n_kv_chunk)) * nb(probs_padded, 0), nb(probs_padded, 2), nb(probs_padded, 3), 0);
                probs_padded = c.ggml_cont(ctx0, probs_padded);
                probs_padded = c.ggml_mul(ctx0, probs_padded, local_mask);

                var V_padded = c.ggml_pad_ext(ctx0, V_cur, 0, 0, att_left, att_right, 0, 0, 0, 0);
                if (n_kv_dense > ne(V_padded, 1)) {
                    V_padded = c.ggml_pad_ext(ctx0, V_padded, 0, 0, 0, @truncate(n_kv_dense - ne(V_padded, 1)), 0, 0, 0, 0);
                }
                V_padded = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, V_padded));

                var V_chunk = c.ggml_view_4d(ctx0, V_padded, n_kv_chunk, d_head, n_group, n_head, nb(V_padded, 1), @as(usize, @intCast(chunk)) * nb(V_padded, 0), nb(V_padded, 2), 0);
                V_chunk = c.ggml_cont(ctx0, V_chunk);

                cur = c.ggml_mul_mat(ctx0, V_chunk, probs_padded);
                cur = c.ggml_reshape_3d(ctx0, cur, d_head, n_time_padded, n_head);
                if (need_padding) {
                    cur = c.ggml_view_3d(ctx0, cur, d_head, n_time, n_head, nb(cur, 1), nb(cur, 2), 0);
                }
                cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 0, 2, 1, 3));
                cur = c.ggml_reshape_2d(ctx0, cur, n_state, n_time);
                cur = g.buildMm(layer.o_w, cur);
            } else {
                // full attention
                const Q_u = c.ggml_add(ctx0, Q_cur, layer.pos_bias_u);
                _ = c.ggml_format_name(Q_u, "enc_%d_attn_q_u", il);

                const K_prep = c.ggml_permute(ctx0, K_cur, 0, 2, 1, 3);
                const Q_prep = c.ggml_permute(ctx0, Q_u, 0, 2, 1, 3);
                const content_scores = c.ggml_mul_mat(ctx0, K_prep, Q_prep);
                _ = c.ggml_format_name(content_scores, "enc_%d_attn_content_scores", il);

                var Q_v = c.ggml_add(ctx0, Q_cur, layer.pos_bias_v);
                _ = c.ggml_format_name(Q_v, "enc_%d_attn_q_v", il);

                Q_v = c.ggml_permute(ctx0, Q_v, 0, 2, 1, 3);
                Q_v = c.ggml_cont(ctx0, Q_v);
                _ = c.ggml_format_name(Q_v, "enc_%d_attn_q_v_perm", il);

                var rel_pos_scores = c.ggml_mul_mat(ctx0, pos, Q_v);
                _ = c.ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos", il);

                // Relative positional shift
                {
                    const pos_window = ne(rel_pos_scores, 0);
                    const n_frame = ne(rel_pos_scores, 1);
                    const n_head_r = ne(rel_pos_scores, 2); // `n_head` upstream, shadowing the outer one

                    rel_pos_scores = c.ggml_pad(ctx0, rel_pos_scores, 1, 0, 0, 0);
                    rel_pos_scores = c.ggml_roll(ctx0, rel_pos_scores, 1, 0, 0, 0);

                    rel_pos_scores = c.ggml_reshape_3d(ctx0, rel_pos_scores, n_frame, pos_window + 1, n_head_r);
                    rel_pos_scores = c.ggml_cont(ctx0, rel_pos_scores);
                    _ = c.ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_reshaped", il);

                    const center: c_int = @truncate(@divTrunc(pos_window, 2));
                    const offset: usize = nb(rel_pos_scores, 0) *% @as(usize, @bitCast(@as(isize, center + 1)));

                    rel_pos_scores = c.ggml_view_3d(ctx0, rel_pos_scores, n_frame, pos_window, n_head_r, @intCast(pos_window * 4), nb(rel_pos_scores, 2), offset);
                    rel_pos_scores = c.ggml_cont(ctx0, rel_pos_scores);
                    _ = c.ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_shifted", il);

                    rel_pos_scores = c.ggml_view_3d(ctx0, rel_pos_scores, ne(content_scores, 0), ne(content_scores, 1), ne(rel_pos_scores, 2), nb(rel_pos_scores, 1), nb(rel_pos_scores, 2), 0);
                    rel_pos_scores = c.ggml_cont(ctx0, rel_pos_scores);
                    _ = c.ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_shifted_view", il);
                }

                var attn_scores = c.ggml_add(ctx0, content_scores, rel_pos_scores);
                _ = c.ggml_format_name(attn_scores, "enc_%d_attn_scores", il);
                attn_scores = c.ggml_scale(ctx0, attn_scores, invSqrt(d_head));
                attn_scores = c.ggml_add(ctx0, attn_scores, attn_mask);
                _ = c.ggml_format_name(attn_scores, "enc_%d_attn_scores_scaled", il);

                const probs = c.ggml_soft_max(ctx0, attn_scores);
                _ = c.ggml_format_name(probs, "enc_%d_attn_probs", il);

                V_cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, V_cur, 1, 2, 0, 3));
                _ = c.ggml_format_name(V_cur, "enc_%d_attn_v_cur", il);
                cur = c.ggml_mul_mat(ctx0, probs, V_cur);
                _ = c.ggml_format_name(cur, "enc_%d_attn_inp", il);

                cur = c.ggml_permute(ctx0, cur, 2, 0, 1, 3);
                cur = c.ggml_cont_2d(ctx0, cur, n_state, n_time);
                cur = g.buildMm(layer.o_w, cur);
            }
            _ = c.ggml_format_name(cur, "enc_%d_attn_out", il);

            cur = c.ggml_add(ctx0, residual, cur);
            _ = c.ggml_format_name(cur, "enc_%d_attn_res", il);
        }

        // Convolution
        {
            const residual = cur;
            _ = c.ggml_format_name(cur, "enc_%d_residual_conv", il);

            cur = c.ggml_norm(ctx0, cur, hparams.eps);
            cur = c.ggml_add(ctx0, c.ggml_mul(ctx0, cur, layer.norm_conv_w), layer.norm_conv_b);
            _ = c.ggml_format_name(cur, "enc_%d_norm_conv", il);

            // pointwise 1d convolution:
            cur = g.buildMm(layer.conv_pw1_w, cur);
            _ = c.ggml_format_name(cur, "enc_%d_conv_pw1", il);

            {
                const d = @divTrunc(ne(cur, 0), 2);
                const signal = c.ggml_view_2d(ctx0, cur, d, ne(cur, 1), nb(cur, 1), 0);
                const gate = c.ggml_view_2d(ctx0, cur, d, ne(cur, 1), nb(cur, 1), @as(usize, @intCast(d)) * nb(cur, 0));

                cur = c.ggml_mul(ctx0, signal, c.ggml_sigmoid(ctx0, gate));
                _ = c.ggml_format_name(cur, "enc_%d_conv_glu", il);
            }

            cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));

            // use ggml_ssm_conv for f32 precision
            const dw_pad = @divTrunc(hparams.audio_conv_kernel_size - 1, 2);
            cur = c.ggml_pad(ctx0, cur, dw_pad, 0, 0, 0);
            cur = c.ggml_roll(ctx0, cur, dw_pad, 0, 0, 0);
            cur = c.ggml_pad(ctx0, cur, dw_pad, 0, 0, 0);
            _ = c.ggml_format_name(cur, "enc_%d_conv_dw_pad", il);

            cur = c.ggml_ssm_conv(ctx0, cur, layer.conv_dw_w);
            _ = c.ggml_format_name(cur, "enc_%d_conv_1d_dw", il);

            cur = c.ggml_sub(ctx0, cur, layer.conv_norm_mean);
            const std_ = c.ggml_sqrt(ctx0, layer.conv_norm_var);
            cur = c.ggml_div(ctx0, cur, std_);
            cur = c.ggml_add(ctx0, c.ggml_mul(ctx0, cur, layer.conv_norm_w), layer.conv_norm_b);
            _ = c.ggml_format_name(cur, "enc_%d_conv_bn", il);

            cur = c.ggml_silu(ctx0, cur);
            _ = c.ggml_format_name(cur, "enc_%d_conv_silu", il);

            cur = g.buildMm(layer.conv_pw2_w, cur);
            _ = c.ggml_format_name(cur, "enc_%d_conv_pw2", il);

            cur = c.ggml_add(ctx0, residual, cur);
            _ = c.ggml_format_name(cur, "enc_%d_conv_res", il);
        }

        // FFN2
        {
            const residual = cur;
            cur = c.ggml_norm(ctx0, cur, hparams.eps);
            cur = c.ggml_add(ctx0, c.ggml_mul(ctx0, cur, layer.ff_norm_1_w), layer.ff_norm_1_b);
            _ = c.ggml_format_name(cur, "enc_%d_ffn_norm_2", il);

            cur = g.buildFfn(cur, layer.ff_up_1_w, null, null, null, layer.ff_down_1_w, null, .silu, il);
            cur = c.ggml_add(ctx0, residual, c.ggml_scale(ctx0, cur, 0.5));
            _ = c.ggml_format_name(cur, "enc_%d_ffn_res", il);
        }

        cur = c.ggml_norm(ctx0, cur, hparams.eps);
        cur = c.ggml_add(ctx0, c.ggml_mul(ctx0, cur, layer.ln_2_w), layer.ln_2_b);
    }

    g.cb(cur, "encoder_out", -1);

    cur = c.ggml_rms_norm(ctx0, cur, 1e-6);
    cur = c.ggml_mul(ctx0, cur, model.mm_norm_pre_w);
    g.cb(cur, "sound_projection.norm", -1);

    cur = g.buildFfn(cur, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, .relu_sqr, -1);
    g.cb(cur, "projected", -1);

    c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
