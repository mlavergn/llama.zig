//! The Gemma 4 audio Conformer encoder: dual half-step FFNs, chunked local
//! self-attention with sinusoidal relative positions, a depthwise light
//! conv, and an output projection.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/gemma4a.cpp` at v0.4.0 (`5266f24da`).
//! Each declaration names what it ports and the line it began at.
//!
//! Upstream overrides the virtual `build_mm`; here `build` installs
//! `buildMm` as the graph's `mm_hook`, so `graph.Graph.buildFfn` and the
//! rest reach it the way the C++ base class reaches the override.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

extern fn sqrtf(x: f32) f32;
extern fn logf(x: f32) f32;
extern fn expf(x: f32) f32;

/// The `extract_blocks` lambda (gemma4a.cpp:122 @5266f24da) of `build`:
/// K/V block context through an overlapping view. Pad to S*B elements,
/// roll right by P to create left-padding, then view with stride C in the
/// block dimension.
fn extractBlocks(g: *const graph.Graph, t_in: T, S: i64, B: i64, P: i64, C: i64, n_pos: i64) T {
    const ctx0 = g.ctx0;
    // [D, H, N] -> pad to S*B -> roll right by P -> cont (materialize)
    const pad_kv = S * B - n_pos;
    var t = c.ggml_pad(ctx0, t_in, 0, 0, @intCast(pad_kv), 0); // [D, H, S*B]
    t = c.ggml_roll(ctx0, t, 0, 0, @intCast(P), 0); // left-pad by P
    t = c.ggml_cont(ctx0, t); // materialize roll (removes view offset)
    // Overlapping view: stride for B dim is C positions, not S
    // ne = [D, H, S, B], data_size = D*H*S*B*sizeof = source_nbytes (exact fit)
    // nb1=D*sizeof, nb2=D*H*sizeof, nb3=C*D*H*sizeof (overlap: C < S)
    t = c.ggml_view_4d(ctx0, t, g.d_head, g.n_head, S, B, nb(t, 1), nb(t, 2), @as(usize, @intCast(C)) * nb(t, 2), 0);
    t = c.ggml_cont(ctx0, t); // materialize overlapping windows
    return t;
}

/// Ports `build` (gemma4a.cpp:11 @5266f24da), a member of
/// `clip_graph_gemma4a`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    g.mm_hook = buildMm;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const d_head = g.d_head;
    const n_head = g.n_head;

    const res_weight: f32 = 0.5;
    const norm_eps: f32 = 1e-6;

    // 1. Input
    const inp = g.buildInpRaw(1);
    var cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp));

    // 2. Subsampling Conv2D (symmetric padding=1, matching PyTorch)
    {
        for (0..2) |i| {
            cur = c.ggml_conv_2d(ctx0, model.sscp_conv_w[i], cur, 2, 2, 1, 1, 1, 1);
            if (model.sscp_conv_b[i] != null) {
                cur = c.ggml_add(ctx0, cur, model.sscp_conv_b[i]);
            }
            // nn.LayerNorm(channels): permute ch to ne[0], normalize, permute back
            if (model.sscp_norm_w[i] != null) {
                cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 1, 2, 0, 3));
                cur = c.ggml_norm(ctx0, cur, norm_eps);
                cur = c.ggml_mul(ctx0, cur, model.sscp_norm_w[i]);
                cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 2, 0, 1, 3));
            }
            cur = c.ggml_relu(ctx0, cur);
        }
        // Flatten [freq, time, ch, 1] -> [ch*freq, time]
        cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = c.ggml_reshape_2d(ctx0, cur, ne(cur, 0) * ne(cur, 1), ne(cur, 2));
        if (model.sscp_inp_proj_w != null) {
            cur = g.buildMm(model.sscp_inp_proj_w, cur);
            if (model.sscp_inp_proj_b != null) {
                cur = c.ggml_add(ctx0, cur, model.sscp_inp_proj_b);
            }
        }
    }

    const n_pos = ne(cur, 1);

    // Chunked local attention parameters
    const C: i64 = 12; // chunk_size
    const P: i64 = 12; // max_past_horizon (context_left - 1)
    const S: i64 = C + P; // context_size = 24
    const R: i64 = P + 1; // RPE positions = 13
    const B: i64 = @divTrunc(n_pos + C - 1, C); // num_blocks
    const Np: i64 = B * C; // padded sequence length
    const pad_seq = Np - n_pos;

    // Input tensors: blocked RPE and blocked attention mask
    const pos_emb = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, @as(i64, n_head) * d_head, R);
    _ = c.ggml_set_name(pos_emb, "pos_emb");
    c.ggml_set_input(pos_emb);

    const kq_mask = c.ggml_new_tensor_3d(ctx0, c.GGML_TYPE_F32, S, C, B);
    _ = c.ggml_set_name(kq_mask, "kq_mask");
    c.ggml_set_input(kq_mask);

    // 3. Conformer Blocks
    var il: c_int = 0;
    while (il < hparams.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        var residual = cur;

        // FFN 1 (half-step)
        if (layer.ff_norm_w != null and layer.ff_up_w != null and layer.ff_down_w != null) {
            cur = g.buildNorm(cur, layer.ff_norm_w, null, .rms, norm_eps, il);
            cur = g.buildFfn(cur, layer.ff_up_w, null, null, null, layer.ff_down_w, null, .silu, il);
            if (layer.ff_post_norm_w != null) {
                cur = g.buildNorm(cur, layer.ff_post_norm_w, null, .rms, norm_eps, il);
            }
            residual = c.ggml_add(ctx0, residual, c.ggml_scale(ctx0, cur, res_weight));
        }

        // Chunked local self-attention with RPE
        if (layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null) {
            const q_scale = (1.0 / sqrtf(@floatFromInt(d_head))) / logf(2.0);
            const k_scale = logf(1.0 + expf(1.0)) / logf(2.0);
            const softcap: f32 = 50.0;

            const attn_norm_w = if (layer.attn_pre_norm_w != null) layer.attn_pre_norm_w else layer.ln_1_w;
            cur = if (attn_norm_w != null) g.buildNorm(residual, attn_norm_w, null, .rms, norm_eps, il) else residual;

            var Qcur = g.buildMm(layer.q_w, cur);
            var Kcur = g.buildMm(layer.k_w, cur);
            var Vcur = g.buildMm(layer.v_w, cur);

            // [n_embd, n_pos] -> [D, H, N]
            Qcur = c.ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = c.ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = c.ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            // Q/K scaling
            Qcur = c.ggml_scale(ctx0, Qcur, q_scale);
            if (layer.per_dim_scale_w != null) {
                Qcur = c.ggml_mul(ctx0, Qcur, c.ggml_reshape_3d(ctx0, layer.per_dim_scale_w, d_head, 1, 1));
            }
            Kcur = c.ggml_scale(ctx0, Kcur, k_scale);
            if (layer.per_dim_k_scale_w != null) {
                Kcur = c.ggml_mul(ctx0, Kcur, c.ggml_reshape_3d(ctx0, layer.per_dim_k_scale_w, d_head, 1, 1));
            }

            // Q blocking: [D, H, N] -> pad to Np -> reshape [D, H, C, B]
            // ggml permute: ne[ax_i] = src->ne[i], so (0,3,1,2) sends H->3, C->1, B->2
            Qcur = c.ggml_pad(ctx0, Qcur, 0, 0, @intCast(pad_seq), 0); // [D, H, Np]
            Qcur = c.ggml_reshape_4d(ctx0, Qcur, d_head, n_head, C, B); // [D, H, C, B]
            Qcur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, Qcur, 0, 3, 1, 2)); // [D, C, B, H]

            var Kblk = extractBlocks(g, Kcur, S, B, P, C, n_pos);
            // [D, H, S, B] -> [D, S, B, H] via permute(0,3,1,2)
            Kblk = c.ggml_cont(ctx0, c.ggml_permute(ctx0, Kblk, 0, 3, 1, 2));

            var Vblk = extractBlocks(g, Vcur, S, B, P, C, n_pos);
            // [D, H, S, B] -> [S, D, B, H] via permute(1,3,0,2)
            Vblk = c.ggml_cont(ctx0, c.ggml_permute(ctx0, Vblk, 1, 3, 0, 2));

            // Content attention: Q @ K^T
            // Kblk=[D,S,B,H], Qcur=[D,C,B,H] -> mul_mat contracts on D -> [S,C,B,H]
            var matrix_ac = c.ggml_mul_mat(ctx0, Kblk, Qcur);

            // Relative position attention
            if (layer.attn_k_rel_w != null) {
                // RPE: [n_embd, R] -> project -> [D, H, R] -> [D, R, H]
                var p = c.ggml_mul_mat(ctx0, layer.attn_k_rel_w, pos_emb);
                p = c.ggml_reshape_3d(ctx0, p, d_head, n_head, R);
                p = c.ggml_cont(ctx0, c.ggml_permute(ctx0, p, 0, 2, 1, 3)); // [D, R, H]

                // Q_flat @ RPE^T: [D, C*B, H] @ [D, R, H] -> [R, C*B, H]
                const Q_flat = c.ggml_reshape_3d(ctx0, Qcur, d_head, C * B, n_head);
                var matrix_bd = c.ggml_mul_mat(ctx0, p, Q_flat); // [R, C*B, H]
                matrix_bd = c.ggml_reshape_4d(ctx0, matrix_bd, R, C, B, n_head); // [R, C, B, H]

                // Blocked relative shift (appendix B of Transformer-XL)
                {
                    matrix_bd = c.ggml_pad(ctx0, matrix_bd, @intCast(S + 1 - R), 0, 0, 0); // [S+1, C, B, H]
                    matrix_bd = c.ggml_reshape_3d(ctx0, matrix_bd, (S + 1) * C, B, n_head);
                    matrix_bd = c.ggml_view_3d(ctx0, matrix_bd, C * S, B, n_head, nb(matrix_bd, 1), nb(matrix_bd, 2), 0);
                    matrix_bd = c.ggml_cont(ctx0, matrix_bd); // [C*S, B, H]
                    matrix_bd = c.ggml_reshape_4d(ctx0, matrix_bd, S, C, B, n_head); // [S, C, B, H]
                }

                matrix_ac = c.ggml_add(ctx0, matrix_ac, matrix_bd);
            }

            var scores = matrix_ac; // [S, C, B, H]

            // Softcap
            scores = c.ggml_scale(ctx0, scores, 1.0 / softcap);
            scores = c.ggml_tanh(ctx0, scores);
            scores = c.ggml_scale(ctx0, scores, softcap);

            // Blocked attention mask: [S, C, B] broadcasts over H
            scores = c.ggml_add(ctx0, scores, kq_mask);

            const attn = c.ggml_soft_max(ctx0, scores);

            // attn @ V: [S,C,B,H] @ [S,D,B,H] -> [D,C,B,H]
            var x = c.ggml_mul_mat(ctx0, Vblk, attn);

            // [D,C,B,H] -> [D,H,C,B] via permute(0,2,3,1) -> flatten -> trim
            x = c.ggml_cont(ctx0, c.ggml_permute(ctx0, x, 0, 2, 3, 1));
            x = c.ggml_cont_2d(ctx0, x, @as(i64, d_head) * n_head, C * B);
            if (pad_seq > 0) {
                x = c.ggml_view_2d(ctx0, x, @as(i64, d_head) * n_head, n_pos, nb(x, 1), 0);
                x = c.ggml_cont(ctx0, x);
            }

            x = g.buildMm(layer.o_w, x);
            if (layer.o_b != null) {
                x = c.ggml_add(ctx0, x, layer.o_b);
            }

            if (layer.attn_post_norm_w != null) {
                x = g.buildNorm(x, layer.attn_post_norm_w, null, .rms, norm_eps, il);
            }
            residual = c.ggml_add(ctx0, residual, x);
        }

        // Convolution Module
        if (layer.norm_conv_w != null and layer.conv_pw1_w != null and layer.conv_dw_w != null and layer.conv_pw2_w != null) {
            cur = g.buildNorm(residual, layer.norm_conv_w, null, .rms, norm_eps, il);
            var x = g.buildMm(layer.conv_pw1_w, cur);

            // GLU
            {
                const d = @divTrunc(ne(x, 0), 2);
                const gate = c.ggml_sigmoid(ctx0, c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, x, d, ne(x, 1), nb(x, 1), @as(usize, @intCast(d)) * nb(x, 0))));
                x = c.ggml_mul(ctx0, c.ggml_view_2d(ctx0, x, d, ne(x, 1), nb(x, 1), 0), gate);
                x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x));
            }

            // Causal depthwise Conv1D via ggml_ssm_conv (pad+roll for left-only padding).
            x = c.ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = c.ggml_roll(ctx0, x, 4, 0, 0, 0);
            x = c.ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            if (layer.conv_dw_b != null) {
                x = c.ggml_add(ctx0, x, layer.conv_dw_b);
            }

            if (layer.conv_norm_w != null) {
                x = c.ggml_rms_norm(ctx0, x, norm_eps);
                x = c.ggml_mul(ctx0, x, layer.conv_norm_w);
            }
            x = c.ggml_silu(ctx0, x);
            x = g.buildMm(layer.conv_pw2_w, x);
            residual = c.ggml_add(ctx0, residual, x);
        }

        // FFN 2 (half-step)
        if (layer.ff_norm_1_w != null and layer.ff_up_1_w != null and layer.ff_down_1_w != null) {
            cur = g.buildNorm(residual, layer.ff_norm_1_w, null, .rms, norm_eps, il);
            cur = g.buildFfn(cur, layer.ff_up_1_w, null, null, null, layer.ff_down_1_w, null, .silu, il);
            if (layer.ff_post_norm_1_w != null) {
                cur = g.buildNorm(cur, layer.ff_post_norm_1_w, null, .rms, norm_eps, il);
            }
            residual = c.ggml_add(ctx0, residual, c.ggml_scale(ctx0, cur, res_weight));
        }

        // Layer output norm
        cur = if (layer.ln_2_w != null) g.buildNorm(residual, layer.ln_2_w, null, .rms, norm_eps, il) else residual;
    }

    // 4. Output Projection
    if (model.audio_out_proj_w != null) {
        cur = g.buildMm(model.audio_out_proj_w, cur);
        if (model.audio_out_proj_b != null) {
            cur = c.ggml_add(ctx0, cur, model.audio_out_proj_b);
        }
    }

    // 5. Audio Multimodal Embedder
    cur = c.ggml_rms_norm(ctx0, cur, norm_eps);
    if (model.mm_soft_emb_norm_w != null) {
        cur = c.ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
    }
    if (model.mm_input_proj_w != null) {
        cur = g.buildMm(model.mm_input_proj_w, cur);
    }

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

/// Ports `build_mm` (gemma4a.cpp:279 @5266f24da), a member of
/// `clip_graph_gemma4a`: a Gemma4ClippableLinear clamps its input and
/// output to the bounds the model file records for that weight.
///
/// Parameters:
///   - `g`: the graph builder.
///   - `w`: the weight; its name keys `clamp_info_map`.
///   - `x`: the input.
///
/// Return: `w @ x`, clamped where the model says so.
pub fn buildMm(g: *const graph.Graph, w: T, x: T) T {
    const ctx0 = g.ctx0;
    const ci = g.model.clamp_info_map.get(std.mem.sliceTo(&w.?.name, 0)) orelse return c.ggml_mul_mat(ctx0, w, x);
    const clamped = c.ggml_clamp(ctx0, x, ci.inp_min, ci.inp_max);
    const out = c.ggml_mul_mat(ctx0, w, clamped);
    return c.ggml_clamp(ctx0, out, ci.out_min, ci.out_max);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
