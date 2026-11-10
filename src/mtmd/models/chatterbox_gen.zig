//! Chatterbox's generation graphs: the flow encoder and the CFM estimator
//! (speech tokens to mel, TTS), the HiFT vocoder (mel and source STFT to a
//! spectrum, TTS_VOCODE) and the S3 tokenizer (log-mel to FSQ codes,
//! TOKENIZE). This file's convention is `[C, T]`: channels on `ne0`, time
//! on `ne1`. The host side of each stage is in `clip.zig`'s `encode`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/chatterbox-gen.cpp` at v0.9.0
//! (`cd959bae9`) but its `clip_graph_chatterbox_base` members, which are
//! `chatterbox_base.zig`; and the `PROJECTOR_TYPE_CHATTERBOX` arm of
//! `clip_get_graph_builder` (`llama.cpp/tools/mtmd/clip.cpp`). Each
//! declaration names what it ports and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const base = @import("chatterbox_base.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;
const Cbx = model_mod.Chatterbox;

extern fn cosf(f32) f32;

/// `M_PI`.
const m_pi: f64 = 3.14159265358979323846;

/// Ports `cbx_rel_shift` (chatterbox-gen.cpp:51 @cd959bae9): the
/// Transformer-XL relative shift, bd `[2T-1, T, H]` to `[T, T, H]` with
/// `out[j, i, h] = bd[(T-1) - i + j, i, h]`; the same buffer walk as the
/// espnet rel_shift: pad one column, drop the first row, keep the first T
/// columns.
fn cbxRelShift(ctx0: ?*c.ggml_context, bd: T, t_len: c_int) T {
    const h: c_int = @intCast(ne(bd, 2));
    var z = c.ggml_new_tensor_3d(ctx0, c.GGML_TYPE_F32, 1, t_len, h);
    z = c.ggml_scale(ctx0, z, 0.0);
    var p = c.ggml_cont(ctx0, c.ggml_concat(ctx0, z, bd, 0)); // [2T, T, H]
    p = c.ggml_reshape_3d(ctx0, p, t_len, 2 * t_len, h); // [T, 2T, H]
    p = c.ggml_view_3d(ctx0, p, t_len, 2 * t_len - 1, h, nb(p, 1), nb(p, 2), nb(p, 1)); // drop first row
    p = c.ggml_cont(ctx0, p);
    p = c.ggml_reshape_3d(ctx0, p, 2 * t_len - 1, t_len, h); // [2T-1, T, H]
    p = c.ggml_view_3d(ctx0, p, t_len, t_len, h, nb(p, 1), nb(p, 2), 0); // first T columns
    return c.ggml_cont(ctx0, p);
}

/// Ports `cbx_mish` (chatterbox-gen.cpp:120 @cd959bae9): mish = x *
/// tanh(softplus(x)).
fn cbxMish(ctx0: ?*c.ggml_context, x: T) T {
    return c.ggml_mul(ctx0, x, c.ggml_tanh(ctx0, c.ggml_softplus(ctx0, x)));
}

/// Ports `cbx_snake` (chatterbox-gen.cpp:450 @cd959bae9): snake activation
/// with per-channel alpha, x + sin^2(alpha x) / alpha.
fn cbxSnake(ctx0: ?*c.ggml_context, x: T, alpha: T) T {
    var sx = c.ggml_sin(ctx0, c.ggml_mul(ctx0, x, alpha));
    sx = c.ggml_mul(ctx0, sx, sx);
    sx = c.ggml_div(ctx0, sx, alpha);
    return c.ggml_add(ctx0, x, sx);
}

/// Ports `clip_graph_chatterbox` (mtmd/models/models.h:232 @cd959bae9):
/// the builder's parameters beside the parent graph.
const Gen = struct {
    g: *graph.Graph,
    gen_process: impl.GenProcessType,
    n_tokens: c_int,
    n_prompt_mel: c_int,
    vocode_n_mel: c_int,
    vocode_n_stft: c_int,

    fn linear(self: *const Gen, w: T, b: T, x: T) T {
        return base.cbxLinear(self.g, w, b, x);
    }

    fn conv1d(self: *const Gen, k: T, b: T, x: T, stride: c_int, pad_l: c_int, pad_r: c_int) T {
        return base.cbxConv1d(self.g, k, b, x, stride, pad_l, pad_r);
    }

    fn conv1dDil(self: *const Gen, k: T, b: T, x: T, pad: c_int, dil: c_int) T {
        return base.cbxConv1dDil(self.g, k, b, x, pad, dil);
    }

    fn norm(self: *const Gen, x: T, w: T, b: T) T {
        return self.g.buildNorm(x, w, b, .normal, 1e-5, -1);
    }

    /// Ports `enc_layer` (chatterbox-gen.cpp:65 @cd959bae9), a member of
    /// `clip_graph_chatterbox`: espnet rel-pos self attention block,
    /// pre-norm, x `[512, T]`, pos `[512, 2T-1]`.
    fn encLayer(self: *const Gen, l: *const Cbx.EncLayer, x_in: T, pos: T, t_len: c_int) T {
        const ctx0 = self.g.ctx0;
        const n_head = 8;
        const d_head = 64;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

        var x = x_in;
        var res = x;
        var cur = self.norm(x, l.norm_mha_w, l.norm_mha_b);

        var q = self.linear(l.attn_q_w, l.attn_q_b, cur);
        var k = self.linear(l.attn_k_w, l.attn_k_b, cur);
        var v = self.linear(l.attn_v_w, l.attn_v_b, cur);
        var pe = self.linear(l.attn_pos_w, null, pos); // [512, 2T-1]

        q = c.ggml_reshape_3d(ctx0, q, d_head, n_head, t_len);
        k = c.ggml_reshape_3d(ctx0, k, d_head, n_head, t_len);
        v = c.ggml_reshape_3d(ctx0, v, d_head, n_head, t_len);
        pe = c.ggml_reshape_3d(ctx0, pe, d_head, n_head, 2 * t_len - 1);

        const u = l.attn_pos_bias_u; // [64, 8]
        const w = l.attn_pos_bias_v;

        var qu = c.ggml_add(ctx0, q, c.ggml_reshape_3d(ctx0, u, d_head, n_head, 1));
        var qv = c.ggml_add(ctx0, q, c.ggml_reshape_3d(ctx0, w, d_head, n_head, 1));

        // per head: [64, T] tensors, scores [T(k), T(q)]
        qu = c.ggml_cont(ctx0, c.ggml_permute(ctx0, qu, 0, 2, 1, 3)); // [64, T, 8]
        qv = c.ggml_cont(ctx0, c.ggml_permute(ctx0, qv, 0, 2, 1, 3));
        k = c.ggml_cont(ctx0, c.ggml_permute(ctx0, k, 0, 2, 1, 3));
        v = c.ggml_cont(ctx0, c.ggml_permute(ctx0, v, 0, 2, 1, 3));
        pe = c.ggml_cont(ctx0, c.ggml_permute(ctx0, pe, 0, 2, 1, 3)); // [64, 2T-1, 8]

        const ac = c.ggml_mul_mat(ctx0, k, qu); // [T(k), T(q), 8]
        var bd = c.ggml_mul_mat(ctx0, pe, qv); // [2T-1, T(q), 8]
        bd = cbxRelShift(ctx0, bd, t_len); // [T(k), T(q), 8]

        const scores = c.ggml_scale(ctx0, c.ggml_add(ctx0, ac, bd), scale);
        const probs = c.ggml_soft_max(ctx0, scores);

        var o = c.ggml_mul_mat(ctx0, c.ggml_cont(ctx0, c.ggml_transpose(ctx0, v)), probs); // [64, T, 8]
        o = c.ggml_cont(ctx0, c.ggml_permute(ctx0, o, 0, 2, 1, 3)); // [64, 8, T]
        o = c.ggml_reshape_2d(ctx0, o, n_head * d_head, t_len);
        o = self.linear(l.attn_out_w, l.attn_out_b, o);
        x = c.ggml_add(ctx0, res, o);

        res = x;
        cur = self.norm(x, l.norm_ff_w, l.norm_ff_b);
        cur = self.linear(l.ffn_1_w, l.ffn_1_b, cur);
        cur = c.ggml_silu(ctx0, cur); // swish
        cur = self.linear(l.ffn_2_w, l.ffn_2_b, cur);
        x = c.ggml_add(ctx0, res, cur);
        return x;
    }

    /// Ports `causal_block` (chatterbox-gen.cpp:125 @cd959bae9), a member of
    /// `clip_graph_chatterbox`: conv k3 left-padded, layer norm over
    /// channels, mish; x `[C, T]`.
    fn causalBlock(self: *const Gen, b: *const Cbx.CausalBlock, x_in: T) T {
        var x = self.conv1d(b.conv_w, b.conv_b, x_in, 1, @as(c_int, @intCast(ne(b.conv_w, 0))) - 1, 0);
        x = self.norm(x, b.norm_w, b.norm_b);
        return cbxMish(self.g.ctx0, x);
    }

    /// Ports `resnet` (chatterbox-gen.cpp:132 @cd959bae9), a member of
    /// `clip_graph_chatterbox`: resnet block with time conditioning; x
    /// `[C, T]`, temb `[1024]`.
    fn resnet(self: *const Gen, r: *const Cbx.Resnet, x: T, temb: T) T {
        const ctx0 = self.g.ctx0;
        var h = self.causalBlock(&r.block1, x);
        const tproj = self.linear(r.mlp_w, r.mlp_b, cbxMish(ctx0, temb));
        h = c.ggml_add(ctx0, h, tproj); // broadcast [256, 1] over T
        h = self.causalBlock(&r.block2, h);
        const res = self.conv1d(r.res_conv_w, r.res_conv_b, x, 1, 0, 0);
        return c.ggml_add(ctx0, h, res);
    }

    /// Ports `tfm_block` (chatterbox-gen.cpp:142 @cd959bae9), a member of
    /// `clip_graph_chatterbox`: diffusers-style transformer block, full
    /// attention; x `[256, T]`.
    fn tfmBlock(self: *const Gen, b: *const Cbx.TfmBlock, x_in: T) T {
        const ctx0 = self.g.ctx0;
        const n_head = 8;
        const d_head = 64;
        const t_len = ne(x_in, 1);

        var cur = self.norm(x_in, b.norm1_w, b.norm1_b);
        const q = c.ggml_reshape_3d(ctx0, c.ggml_mul_mat(ctx0, b.attn_q_w, cur), d_head, n_head, t_len);
        const k = c.ggml_reshape_3d(ctx0, c.ggml_mul_mat(ctx0, b.attn_k_w, cur), d_head, n_head, t_len);
        const v = c.ggml_reshape_3d(ctx0, c.ggml_mul_mat(ctx0, b.attn_v_w, cur), d_head, n_head, t_len);
        const x = c.ggml_add(ctx0, x_in, self.g.buildAttn(b.attn_out_w, b.attn_out_b, q, k, v, null, 1.0 / @sqrt(@as(f32, @floatFromInt(d_head))), -1, null));

        cur = self.norm(x, b.norm3_w, b.norm3_b);
        cur = self.linear(b.ff_in_w, b.ff_in_b, cur);
        cur = c.ggml_gelu_erf(ctx0, cur);
        cur = self.linear(b.ff_out_w, b.ff_out_b, cur);
        return c.ggml_add(ctx0, x, cur);
    }

    /// Ports `estimator` (chatterbox-gen.cpp:161 @cd959bae9), a member of
    /// `clip_graph_chatterbox`: one estimator evaluation; x_noise, mu, cond
    /// `[80, T]`, spks `[80]`, temb `[1024]`.
    fn estimator(self: *const Gen, x_noise: T, mu: T, spks: T, cond: T, temb: T, t_len: c_int) T {
        const ctx0 = self.g.ctx0;
        const cbx = &self.g.model.cbx;

        // channels live on ne0, time on ne1: pack along ne0
        var x = c.ggml_concat(ctx0, x_noise, mu, 0); // [160, T]
        const spks_b = c.ggml_repeat(ctx0, c.ggml_reshape_2d(ctx0, spks, 80, 1), c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 80, t_len));
        x = c.ggml_concat(ctx0, x, spks_b, 0); // [240, T]
        x = c.ggml_concat(ctx0, x, cond, 0); // [320, T]
        x = c.ggml_cont(ctx0, x);

        // down
        x = self.resnet(&cbx.est_down.res, x, temb);
        for (cbx.est_down.tfm.items) |*b| x = self.tfmBlock(b, x);
        const skip = x;
        x = self.conv1d(cbx.est_down.conv_w, cbx.est_down.conv_b, x, 1, @as(c_int, @intCast(ne(cbx.est_down.conv_w, 0))) - 1, 0);

        // mid
        for (cbx.est_mid.items) |*m| {
            x = self.resnet(&m.res, x, temb);
            for (m.tfm.items) |*b| x = self.tfmBlock(b, x);
        }

        // up with skip
        x = c.ggml_concat(ctx0, x, skip, 0); // [512, T]
        x = self.resnet(&cbx.est_up.res, x, temb);
        for (cbx.est_up.tfm.items) |*b| x = self.tfmBlock(b, x);
        x = self.conv1d(cbx.est_up.conv_w, cbx.est_up.conv_b, x, 1, @as(c_int, @intCast(ne(cbx.est_up.conv_w, 0))) - 1, 0);

        x = self.causalBlock(&cbx.est_final_block, x);
        x = self.conv1d(cbx.est_final_proj_w, cbx.est_final_proj_b, x, 1, 0, 0); // [80, T]
        return x;
    }

    /// Ports `build_s3tok` (chatterbox-gen.cpp:203 @cd959bae9), a member of
    /// `clip_graph_chatterbox`: s3 tokenizer encoder, log-mel `[T, 128]` to
    /// fsq code `[8, T / 4]`, rounded to base 3 tokens on the host.
    fn buildS3tok(self: *const Gen, t_len: c_int) T {
        const g = self.g;
        const ctx0 = g.ctx0;
        const cbx = &g.model.cbx;
        const n_head = 20;
        const d_head = 64;
        const t1 = @divTrunc(t_len - 1, 2) + 1;
        const t2 = @divTrunc(t1 - 1, 2) + 1;

        const inp = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, t_len, 128);
        _ = c.ggml_set_name(inp, "inp_raw");
        c.ggml_set_input(inp);

        const pos = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, t2);
        _ = c.ggml_set_name(pos, "inp_pos");
        c.ggml_set_input(pos);

        var x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp)); // [128, T]
        x = self.conv1d(cbx.s3tok_conv1_w, cbx.s3tok_conv1_b, x, 2, 1, 1);
        x = c.ggml_gelu_erf(ctx0, x);
        x = self.conv1d(cbx.s3tok_conv2_w, cbx.s3tok_conv2_b, x, 2, 1, 1);
        x = c.ggml_gelu_erf(ctx0, x); // [1280, T2]

        for (cbx.s3tok_blocks.items) |*blk| {
            var res = x;
            var cur = self.norm(x, blk.attn_ln_w, blk.attn_ln_b);
            var q = self.linear(blk.attn_q_w, blk.attn_q_b, cur);
            var k = c.ggml_mul_mat(ctx0, blk.attn_k_w, cur);
            var v = self.linear(blk.attn_v_w, blk.attn_v_b, cur);

            // fsmn memory: depthwise conv over time on v plus residual, added to the attention output
            var fsm = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, v)); // [T2, 1280]
            {
                const w = blk.fsmn_w;
                const m = c.ggml_conv_1d_dw(ctx0, w, fsm, 1, @divTrunc(@as(c_int, @intCast(ne(w, 0))) - 1, 2), 1);
                fsm = c.ggml_add(ctx0, c.ggml_reshape_2d(ctx0, m, ne(fsm, 0), ne(fsm, 1)), fsm);
            }
            fsm = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, fsm)); // [1280, T2]

            q = c.ggml_reshape_3d(ctx0, q, d_head, n_head, t2);
            k = c.ggml_reshape_3d(ctx0, k, d_head, n_head, t2);
            v = c.ggml_reshape_3d(ctx0, v, d_head, n_head, t2);
            q = c.ggml_rope_ext(ctx0, q, pos, null, d_head, c.GGML_ROPE_TYPE_NEOX, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
            k = c.ggml_rope_ext(ctx0, k, pos, null, d_head, c.GGML_ROPE_TYPE_NEOX, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
            const o = g.buildAttn(blk.attn_out_w, blk.attn_out_b, q, k, v, null, 1.0 / @sqrt(@as(f32, @floatFromInt(d_head))), -1, null);
            x = c.ggml_add(ctx0, res, c.ggml_add(ctx0, o, fsm));

            res = x;
            cur = self.norm(x, blk.mlp_ln_w, blk.mlp_ln_b);
            cur = self.linear(blk.mlp_in_w, blk.mlp_in_b, cur);
            cur = c.ggml_gelu_erf(ctx0, cur);
            cur = self.linear(blk.mlp_out_w, blk.mlp_out_b, cur);
            x = c.ggml_add(ctx0, res, cur);
        }

        x = self.linear(cbx.s3tok_down_w, cbx.s3tok_down_b, x); // [8, T2]
        x = c.ggml_tanh(ctx0, x);

        _ = c.ggml_set_name(x, "out_fsq");
        c.ggml_set_output(x);
        c.ggml_build_forward_expand(g.gf, x);
        return x;
    }

    /// The `time_mlp` lambda (chatterbox-gen.cpp:374 @cd959bae9) of build.
    fn timeMlp(self: *const Gen, e_in: T) T {
        const cbx = &self.g.model.cbx;
        var e = self.linear(cbx.time_mlp_1_w, cbx.time_mlp_1_b, e_in);
        e = c.ggml_silu(self.g.ctx0, e);
        e = self.linear(cbx.time_mlp_2_w, cbx.time_mlp_2_b, e);
        return e;
    }

    /// Ports `hift_resblock` (chatterbox-gen.cpp:458 @cd959bae9), a member
    /// of `clip_graph_chatterbox`: hifigan-snake resblock; kernels with
    /// dilations 1/3/5 on convs1, 1 on convs2.
    fn hiftResblock(self: *const Gen, r: *const Cbx.HiftRes, x_in: T) T {
        const ctx0 = self.g.ctx0;
        const dil = [3]c_int{ 1, 3, 5 };
        var x = x_in;
        for (r.units.items, 0..) |*u, j| {
            const d = dil[j % 3];
            const p1 = @divTrunc(@as(c_int, @intCast(ne(u.conv1_w, 0) - 1)), 2) * d;
            const p2 = @divTrunc(@as(c_int, @intCast(ne(u.conv2_w, 0) - 1)), 2);
            var xt = cbxSnake(ctx0, x, u.act1_alpha);
            xt = self.conv1dDil(u.conv1_w, u.conv1_b, xt, p1, d);
            xt = cbxSnake(ctx0, xt, u.act2_alpha);
            xt = self.conv1dDil(u.conv2_w, u.conv2_b, xt, p2, 1);
            x = c.ggml_add(ctx0, x, xt);
        }
        return x;
    }

    /// Ports `build_vocoder` (chatterbox-gen.cpp:475 @cd959bae9), a member
    /// of `clip_graph_chatterbox`: mel `[80, T]` + source stft `[18,
    /// T_stft]` to the conv_post output `[18, T_stft2]`.
    fn buildVocoder(self: *const Gen, n_mel: c_int, n_stft: c_int) void {
        const g = self.g;
        const ctx0 = g.ctx0;
        const cbx = &g.model.cbx;

        const mel = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 80, n_mel);
        _ = c.ggml_set_name(mel, "inp_mel");
        c.ggml_set_input(mel);
        const sstft = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 18, n_stft);
        _ = c.ggml_set_name(sstft, "inp_sstft");
        c.ggml_set_input(sstft);

        var x = self.conv1dDil(cbx.hift_pre_w, cbx.hift_pre_b, mel, 3, 1);

        for (cbx.hift_ups.items, 0..) |*up, i| {
            const uk = up.up_w;
            const kk: c_int = @intCast(ne(uk, 0));
            const s = @divTrunc(kk, 2);
            const p = @divTrunc(kk - s, 2);

            x = c.ggml_leaky_relu(ctx0, x, 0.1, false);
            // conv transpose then trim the torch padding P on both sides
            var xt = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x));
            xt = c.ggml_conv_transpose_1d(ctx0, uk, xt, s, 0, 1);
            xt = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, xt, ne(xt, 0) - 2 * p, ne(xt, 1), nb(xt, 1), @as(usize, @intCast(p)) * c.ggml_element_size(xt)));
            x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, xt));
            x = c.ggml_add(ctx0, x, up.up_b);

            const is_last = i + 1 == cbx.hift_ups.items.len;
            if (is_last) {
                var xr = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x));
                xr = c.ggml_pad_reflect_1d(ctx0, xr, 1, 0);
                x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, xr));
            }

            // source injection: strided conv on the source stft, one resblock
            const sk = up.source_down_w;
            const sk_n: c_int = @intCast(ne(sk, 0));
            const ss: c_int = if (sk_n > 1) @divTrunc(sk_n, 2) else 1;
            const sp: c_int = if (sk_n > 1) @divTrunc(ss, 2) else 0;
            var si: T = undefined;
            {
                var st = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, sstft));
                st = c.ggml_conv_1d(ctx0, sk, st, ss, sp, 1);
                si = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, st));
                si = c.ggml_add(ctx0, si, up.source_down_b);
            }
            si = self.hiftResblock(&up.source_res, si);
            // align lengths: the reflection pad on the last stage adds one step
            if (@as(c_int, @intCast(ne(si, 1))) != @as(c_int, @intCast(ne(x, 1)))) {
                const n = @min(ne(si, 1), ne(x, 1));
                si = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, si, ne(si, 0), n, nb(si, 1), 0));
                x = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, x, ne(x, 0), n, nb(x, 1), 0));
            }
            x = c.ggml_add(ctx0, x, si);

            var acc = self.hiftResblock(&up.res_0, x);
            acc = c.ggml_add(ctx0, acc, self.hiftResblock(&up.res_1, x));
            acc = c.ggml_add(ctx0, acc, self.hiftResblock(&up.res_2, x));
            x = c.ggml_scale(ctx0, acc, 1.0 / 3.0);
        }

        x = c.ggml_leaky_relu(ctx0, x, 0.01, false);
        x = self.conv1dDil(cbx.hift_post_w, cbx.hift_post_b, x, 3, 1);
        _ = c.ggml_set_name(x, "out_spec");
        c.ggml_set_output(x);
        c.ggml_build_forward_expand(g.gf, x);
    }

    /// Ports `build` (chatterbox-gen.cpp:265 @cd959bae9), a member of
    /// `clip_graph_chatterbox`.
    fn build(self: *const Gen) void {
        const g = self.g;
        const ctx0 = g.ctx0;

        if (self.gen_process == .tts_vocode) {
            self.buildVocoder(self.vocode_n_mel, self.vocode_n_stft);
            return;
        }
        if (self.gen_process == .tokenize) {
            _ = self.buildS3tok(g.img.nx());
            return;
        }
        if (self.gen_process != .tts) {
            // load-time buffer sizing path
            const inp = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 1);
            _ = c.ggml_set_name(inp, "inp_stub");
            c.ggml_set_input(inp);
            const cur = c.ggml_dup(ctx0, inp);
            _ = c.ggml_set_name(cur, "out_stub");
            c.ggml_set_output(cur);
            c.ggml_build_forward_expand(g.gf, cur);
            return;
        }

        const cbx = &g.model.cbx;

        const t1 = self.n_tokens; // token-rate length (prompt + generated)
        const t2 = 2 * self.n_tokens; // mel-rate length after the x2 upsample

        const inp_tokens = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, t1);
        _ = c.ggml_set_name(inp_tokens, "inp_tokens");
        c.ggml_set_input(inp_tokens);

        const pos1 = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 512, 2 * t1 - 1);
        _ = c.ggml_set_name(pos1, "inp_pos1");
        c.ggml_set_input(pos1);

        const pos2 = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 512, 2 * t2 - 1);
        _ = c.ggml_set_name(pos2, "inp_pos2");
        c.ggml_set_input(pos2);

        // token embedding
        var x = c.ggml_get_rows(ctx0, cbx.input_embedding_w, inp_tokens); // [512, T1]

        // embed: linear + layer norm, then the espnet xscale
        x = self.linear(cbx.embed_linear_w, cbx.embed_linear_b, x);
        x = self.norm(x, cbx.embed_norm_w, cbx.embed_norm_b);
        x = c.ggml_scale(ctx0, x, @sqrt(@as(f32, 512.0)));
        g.cb(x, "fenc_embd", -1);

        // pre-lookahead: conv k=4 right-padded 3, leaky 0.01, conv k=3 left-padded 2, residual
        {
            const res = x;
            var cur = self.conv1d(cbx.pre_conv1_w, cbx.pre_conv1_b, x, 1, 0, 3);
            cur = c.ggml_leaky_relu(ctx0, cur, 0.01, false);
            cur = self.conv1d(cbx.pre_conv2_w, cbx.pre_conv2_b, cur, 1, 2, 0);
            x = c.ggml_add(ctx0, res, cur);
            g.cb(x, "fenc_pre_lookahead", -1);
        }

        for (cbx.enc.items, 0..) |*l, i| {
            x = self.encLayer(l, x, pos1, t1);
            g.cb(x, "fenc_enc", @intCast(i));
        }

        // upsample x2: nearest repeat, left pad 4, conv k=5
        {
            var xt = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x)); // [T1, 512]
            xt = c.ggml_interpolate(ctx0, xt, 2 * t1, 512, 1, 1, c.GGML_SCALE_MODE_NEAREST);
            var z = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 4, 512);
            z = c.ggml_scale(ctx0, z, 0.0);
            xt = c.ggml_concat(ctx0, z, xt, 0);
            const y = c.ggml_conv_1d(ctx0, cbx.up_conv_w, xt, 1, 0, 1);
            x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, y)); // [512, T2]
            x = c.ggml_add(ctx0, x, cbx.up_conv_b);
            g.cb(x, "fenc_upsample", -1);
        }

        // up embed: linear + layer norm + xscale
        x = self.linear(cbx.up_embed_linear_w, cbx.up_embed_linear_b, x);
        x = self.norm(x, cbx.up_embed_norm_w, cbx.up_embed_norm_b);
        x = c.ggml_scale(ctx0, x, @sqrt(@as(f32, 512.0)));

        for (cbx.up_enc.items, 0..) |*l, i| {
            x = self.encLayer(l, x, pos2, t2);
            g.cb(x, "fenc_up_enc", @intCast(i));
        }

        x = self.norm(x, cbx.after_norm_w, cbx.after_norm_b);

        // encoder projection to the mel channel count
        const mu = self.linear(cbx.encoder_proj_w, cbx.encoder_proj_b, x); // [80, T2]
        g.cb(mu, "flow_mu", -1);

        // cfm solver unrolled in the graph: meanflow takes 2 euler steps without cfg, its time embeds mix t and r
        // the classic solver takes 10 euler steps on the cosine schedule with cfg 0.7
        const meanflow = cbx.time_embed_mixer_w != null;
        const n_steps: c_int = if (meanflow) 2 else 10;

        // span points, same schedule as the host side sinusoid fill in clip.cpp
        var span: [11]f32 = undefined;
        {
            var i: c_int = 0;
            while (i <= n_steps) : (i += 1) {
                const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_steps));
                span[@intCast(i)] = if (meanflow) u else 1.0 - cosf(u * @as(f32, @floatCast(m_pi)) / 2.0);
            }
        }

        const noise = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 80, t2);
        _ = c.ggml_set_name(noise, "inp_noise");
        c.ggml_set_input(noise);
        // sinusoidal time embeddings, one row per span point
        const temb_sin = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 320, n_steps + 1);
        _ = c.ggml_set_name(temb_sin, "inp_temb");
        c.ggml_set_input(temb_sin);

        // mel-rate conditions of the voice: prompt features then zeros, and the speaker vector
        const pf = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 80, self.n_prompt_mel);
        _ = c.ggml_set_name(pf, "inp_prompt_feat");
        c.ggml_set_input(pf);
        var zc = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 80, t2 - self.n_prompt_mel);
        zc = c.ggml_scale(ctx0, zc, 0.0);
        const cond = c.ggml_concat(ctx0, pf, zc, 1); // [80, T2]
        // raw 192-dim campplus x-vector, normalized then through the flow speaker affine
        const xvec = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 192);
        _ = c.ggml_set_name(xvec, "inp_xvec");
        c.ggml_set_input(xvec);
        const n2 = c.ggml_sqrt(ctx0, c.ggml_sum(ctx0, c.ggml_mul(ctx0, xvec, xvec)));
        const unit = c.ggml_div(ctx0, xvec, n2);
        var spks = self.linear(cbx.spk_affine_w, cbx.spk_affine_b, c.ggml_reshape_2d(ctx0, unit, 192, 1));
        spks = c.ggml_cont(ctx0, c.ggml_reshape_1d(ctx0, spks, 80));

        const cfg: f32 = 0.7;
        const mu_zero: T = if (meanflow) null else c.ggml_scale(ctx0, mu, 0.0);
        const cond_zero: T = if (meanflow) null else c.ggml_scale(ctx0, cond, 0.0);
        const spks_zero: T = if (meanflow) null else c.ggml_scale(ctx0, spks, 0.0);

        var mx = noise;
        {
            var i: c_int = 0;
            while (i < n_steps) : (i += 1) {
                const temb = self.stepTemb(temb_sin, meanflow, i);
                var d = self.estimator(mx, mu, spks, cond, temb, t2);
                if (!meanflow) {
                    const du = self.estimator(mx, mu_zero, spks_zero, cond_zero, temb, t2);
                    d = c.ggml_add(ctx0, c.ggml_scale(ctx0, d, 1.0 + cfg), c.ggml_scale(ctx0, du, -cfg));
                }
                const k: usize = @intCast(i);
                mx = c.ggml_add(ctx0, mx, c.ggml_scale(ctx0, d, span[k + 1] - span[k]));
                g.cb(mx, "cfm_step", i);
            }
        }
        var mel = mx;

        // trim the prompt frames at mel rate
        mel = c.ggml_view_2d(ctx0, mel, 80, t2 - self.n_prompt_mel, nb(mel, 1), @as(usize, @intCast(self.n_prompt_mel)) * nb(mel, 1));
        mel = c.ggml_cont(ctx0, mel);
        _ = c.ggml_set_name(mel, "out_mel");
        c.ggml_set_output(mel);
        c.ggml_build_forward_expand(g.gf, mel);

        // f0 predictor on the trimmed mel: 5x (conv k3 same-pad + elu), abs(linear)
        {
            var fx = mel;
            for (cbx.f0_condnet.items) |*fc| {
                fx = self.conv1d(fc.w, fc.b, fx, 1, 1, 1);
                fx = c.ggml_elu(ctx0, fx);
            }
            fx = self.linear(cbx.f0_classifier_w, cbx.f0_classifier_b, fx);
            fx = c.ggml_abs(ctx0, fx); // [1, T]
            _ = c.ggml_set_name(fx, "out_f0");
            c.ggml_set_output(fx);
            c.ggml_build_forward_expand(g.gf, fx);
        }
    }

    /// The `span_emb` and `step_temb` lambdas (chatterbox-gen.cpp:380, 383
    /// @cd959bae9) of build: the time embedding of solver step `i`.
    fn stepTemb(self: *const Gen, temb_sin: T, meanflow: bool, i: c_int) T {
        const ctx0 = self.g.ctx0;
        const span_emb = struct {
            fn f(ctx: ?*c.ggml_context, ts: T, k: c_int) T {
                return c.ggml_view_2d(ctx, ts, 320, 1, nb(ts, 1), @as(usize, @intCast(k)) * nb(ts, 1));
            }
        }.f;
        if (!meanflow) return self.timeMlp(span_emb(ctx0, temb_sin, i));
        // arguments evaluated left to right, as clang does
        const e0 = self.timeMlp(span_emb(ctx0, temb_sin, i));
        const e1 = self.timeMlp(span_emb(ctx0, temb_sin, i + 1));
        const e = c.ggml_concat(ctx0, e0, e1, 0); // [2048, 1]
        return c.ggml_mul_mat(ctx0, self.g.model.cbx.time_embed_mixer_w, e); // [1024, 1]
    }
};

/// Ports the `PROJECTOR_TYPE_CHATTERBOX` arm (mtmd/clip.cpp:1208 @cd959bae9)
/// of clip_get_graph_builder, which derives the builder's parameters,
/// and the builder's `build`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    const gen_process: impl.GenProcessType = if (params) |p| p.gen_process else .gen_code;
    var n_tokens: c_int = if (params) |p| (if (p.codes) |cd| @intCast(cd.items.len) else 0) else 0;
    var n_prompt_mel: c_int = 0;
    if (n_tokens > 0) {
        impl.ggmlAssert(params.?.ref != null, "clip.cpp", 1214, "params->ref");
        n_tokens += @intCast(params.?.ref.?.tokens.items.len);
        n_prompt_mel = @intCast(params.?.ref.?.feat.items.len / 80);
    }
    const vnm: c_int = if (params) |p| p.vocode_n_mel else 0;
    const vns: c_int = if (params) |p| p.vocode_n_stft else 0;
    const self: Gen = .{
        .g = g,
        .gen_process = gen_process,
        .n_tokens = n_tokens,
        .n_prompt_mel = n_prompt_mel,
        .vocode_n_mel = vnm,
        .vocode_n_stft = vns,
    };
    self.build();
    return g.gf;
}

// -----------------------------------------------------------------------------
// clip_encode's host side: the input arm and the three stages after compute.
// clip.cpp is compiled with `-ffp-contract=on`: a same-type `a*b + c` is one
// `llvm.fmuladd` (`@mulAdd` here, the left product fused), `x += a*b` too,
// and in `a*b - c*d` only the left product fuses. On Darwin clang folds a
// `sin`/`cos` pair on one argument into `__sincos_stret` (`sinCosF64`).

extern fn sin(f64) f64;
extern fn cos(f64) f64;
extern fn exp(f64) f64;
extern fn log(f64) f64;
extern fn fmin(f64, f64) f64;
extern fn fmax(f64, f64) f64;
extern fn tanhf(f32) f32;

const Double2 = extern struct { sin: f64, cos: f64 };
extern fn __sincos_stret(x: f64) Double2;

/// `sin(x)` and `cos(x)` as clang emits the pair on one argument: one
/// `__sincos_stret` on Darwin, two calls elsewhere (`-fmath-errno`).
fn sinCosF64(x: f64) Double2 {
    if (comptime @import("builtin").os.tag.isDarwin()) return __sincos_stret(x);
    return .{ .sin = sin(x), .cos = cos(x) };
}

/// libc++'s `uniform_real_distribution<double>(a, b)(g)`
/// (`<__random/uniform_real_distribution.h>`): `(b - a) * canonical + a`,
/// contracted.
fn uniformRealF64(g: *libcxx.Mt19937, a: f64, b: f64) f64 {
    return @mulAdd(f64, b - a, libcxx.canonicalF64(g), a);
}

const libcxx = @import("llama").libcxx;
const arms_inputs = @import("../arms_inputs.zig");
const gpa = impl.gpa;

fn usz(x: anytype) usize {
    return @intCast(x);
}

/// The `fill_pos` lambda of the CHATTERBOX input arm (mtmd/clip.cpp:6784
/// @cd959bae9): espnet relative positional encoding, entry k holds the
/// sinusoid of relative position (T-1) - k.
fn fillPos(st: *const clip.EncodeState, name: [*:0]const u8, t_len: c_int) clip.Error!void {
    const d = 512;
    const pos = try gpa.alloc(f32, usz(2 * t_len - 1) * d);
    defer gpa.free(pos);
    var k: c_int = 0;
    while (k < 2 * t_len - 1) : (k += 1) {
        const rel: f64 = @floatFromInt(t_len - 1 - k);
        var i: usize = 0;
        while (i < d / 2) : (i += 1) {
            const div = exp(-@as(f64, @floatFromInt(2 * i)) * log(10000.0) / @as(f64, d));
            const sc = sinCosF64(rel * div);
            pos[usz(k) * d + 2 * i] = @floatCast(sc.sin);
            pos[usz(k) * d + 2 * i + 1] = @floatCast(sc.cos);
        }
    }
    st.setInputF32(name, pos);
}

/// Ports the `PROJECTOR_TYPE_CHATTERBOX` arm (mtmd/clip.cpp:6742 @cd959bae9)
/// of clip_encode's input switch.
///
/// Return: false where the C++ `return false`s out of `clip_encode` (a
/// speech token out of range), having logged why.
pub fn setInputs(st: *clip.EncodeState) clip.Error!bool {
    const func = "clip_encode";
    const params = st.params;
    const model = &st.ctx.model;
    if (params.gen_process == .tts_vocode) {
        st.setInputF32("inp_mel", params.mel_in.?.items);
        st.setInputF32("inp_sstft", params.sstft_in.?.items);
        return true;
    }
    if (params.gen_process == .tokenize) {
        const t1 = @divTrunc(st.imgs.entries.items[0].nx() - 1, 2) + 1;
        const t2 = @divTrunc(t1 - 1, 2) + 1;
        const pos = try gpa.alloc(i32, usz(t2));
        defer gpa.free(pos);
        for (pos, 0..) |*p, i| p.* = @intCast(i);
        st.setInputI32("inp_pos", pos);
        return true;
    }
    if (params.gen_process != .tts) return true;

    const n_gen: c_int = @intCast(params.codes.?.items.len);
    impl.ggmlAssert(params.ref != null, "clip.cpp", 6763, "params->ref");
    const ref = params.ref.?;
    const n_prompt: c_int = @intCast(ref.tokens.items.len);
    const tokens = try gpa.alloc(i32, usz(n_prompt + n_gen));
    defer gpa.free(tokens);
    @memcpy(tokens[0..usz(n_prompt)], ref.tokens.items);
    @memcpy(tokens[usz(n_prompt)..], params.codes.?.items);
    const n_codes = ne(model.cbx.input_embedding_w, 1);
    for (tokens) |t| {
        if (t < 0 or t >= n_codes) {
            impl.logErr("%s: speech token out of range (%d, codebook size %lld)\n", .{ func, t, @as(c_longlong, n_codes) });
            return false;
        }
    }
    const t1 = n_prompt + n_gen;
    const t2 = 2 * t1;
    st.setInputI32("inp_tokens", tokens);

    // mel-rate reference conditioning of the voice
    st.setInputF32("inp_prompt_feat", ref.feat.items);
    st.setInputF32("inp_xvec", ref.spk.items);

    try fillPos(st, "inp_pos1", t1);
    try fillPos(st, "inp_pos2", t2);

    // noise and the time embeddings of the solver span points, matcha layout (sines then cosines, scale 1000)
    {
        const noise = try gpa.alloc(f32, 80 * usz(t2));
        defer gpa.free(noise);
        var nd: arms_inputs.NormalF32 = .{ .mean = 0.0, .stddev = 1.0 };
        for (noise) |*f| f.* = nd.next(&params.gen_state.?.rng);
        st.setInputF32("inp_noise", noise);
    }

    const meanflow = model.cbx.time_embed_mixer_w != null;
    const n_steps: c_int = if (meanflow) 2 else 10;
    const temb = try gpa.alloc(f32, 320 * usz(n_steps + 1));
    defer gpa.free(temb);
    var s_: c_int = 0;
    while (s_ <= n_steps) : (s_ += 1) {
        const u = @as(f64, @floatFromInt(s_)) / @as(f64, @floatFromInt(n_steps));
        const t = if (meanflow) u else 1.0 - cos(u * m_pi / 2.0);
        var i: usize = 0;
        while (i < 160) : (i += 1) {
            const div = exp(-@as(f64, @floatFromInt(i)) * log(10000.0) / 159.0);
            const sc = sinCosF64(1000.0 * t * div);
            temb[usz(s_) * 320 + i] = @floatCast(sc.sin);
            temb[usz(s_) * 320 + i + 160] = @floatCast(sc.cos);
        }
    }
    st.setInputF32("inp_temb", temb);
    return true;
}

/// Ports clip_encode's `CLIP_GEN_PROCESS_TOKENIZE` branch after compute
/// (mtmd/clip.cpp:7459 @cd959bae9): the FSQ codes, and the speech table rows
/// of the codes.
pub fn encodeTokenizeOut(ctx: *clip.Ctx, params: *clip.EncodeParams, gf: ?*c.ggml_cgraph) clip.Error!bool {
    const func = "clip_encode";
    const fsq = c.ggml_graph_get_tensor(gf, "out_fsq");
    impl.ggmlAssert(fsq != null and params.out_codes != null, "clip.cpp", 7461, "fsq != nullptr && params->out_codes");
    const n_tok: usize = @intCast(ne(fsq, 1));
    const h = try gpa.alloc(f32, 8 * n_tok);
    defer gpa.free(h);
    c.ggml_backend_tensor_get(fsq, h.ptr, 0, c.ggml_nbytes(fsq));

    // fsq round to base 3: h in (-1, 1) maps to digits {0, 1, 2}
    const out_codes = params.out_codes.?;
    try out_codes.resize(gpa, n_tok);
    for (0..n_tok) |t| {
        var code: i32 = 0;
        var i: usize = 8;
        while (i > 0) {
            i -= 1;
            const d: i32 = @as(i32, @intFromFloat(@round(h[t * 8 + i] * 0.9990000128746033))) + 1;
            code = code *% 3 +% d;
        }
        out_codes.items[t] = code;
    }

    // speech table rows of the codes, the multilingual variant adds its learned positions as its perceiver expects
    if (params.out_code_embd) |out_code_embd| {
        const tensors = &ctx.model.cbx_tensors;
        const tab_it = tensors.get("a.gen.code.out_embd.weight");
        impl.ggmlAssert(tab_it != null, "clip.cpp", 7481, "tab_it != tensors.end()");
        const tab: *c.ggml_tensor = tab_it.?.?;
        const n_e: usize = @intCast(tab.ne[0]);
        const pos: ?*c.ggml_tensor = if (tensors.get("a.gen.t3.speech_pos_emb")) |p| p else null;
        if ((tab.type != c.GGML_TYPE_F16 and tab.type != c.GGML_TYPE_F32) or
            (pos != null and (pos.?.type != c.GGML_TYPE_F32 or pos.?.ne[0] != @as(i64, @intCast(n_e)) or pos.?.ne[1] < @as(i64, @intCast(n_tok)))))
        {
            impl.logErr("%s: unexpected speech table or positions\n", .{func});
            return false;
        }

        try out_code_embd.resize(gpa, n_tok * n_e);
        const h16 = try gpa.alloc(c.ggml_fp16_t, n_e);
        defer gpa.free(h16);
        const pr = try gpa.alloc(f32, n_e);
        defer gpa.free(pr);
        for (0..n_tok) |t| {
            const dst = out_code_embd.items[t * n_e ..][0..n_e];
            if (out_codes.items[t] >= tab.ne[1]) {
                impl.logErr("%s: speech code beyond the speech table\n", .{func});
                return false;
            }
            const r = @as(usize, @intCast(out_codes.items[t])) * n_e;
            if (tab.type == c.GGML_TYPE_F16) {
                c.ggml_backend_tensor_get(tab, h16.ptr, r * @sizeOf(c.ggml_fp16_t), n_e * @sizeOf(c.ggml_fp16_t));
                for (dst, h16) |*o, v| o.* = c.ggml_fp16_to_fp32(v);
            } else {
                c.ggml_backend_tensor_get(tab, dst.ptr, r * @sizeOf(f32), n_e * @sizeOf(f32));
            }
            if (pos) |p| {
                c.ggml_backend_tensor_get(p, pr.ptr, t * n_e * @sizeOf(f32), n_e * @sizeOf(f32));
                for (dst, pr) |*o, v| o.* += v;
            }
        }
    }
    return true;
}

/// Ports clip_encode's `CLIP_GEN_PROCESS_TTS_VOCODE` branch after compute
/// (mtmd/clip.cpp:7521 @cd959bae9).
pub fn encodeVocodeOut(params: *clip.EncodeParams, gf: ?*c.ggml_cgraph) clip.Error!bool {
    const sp = c.ggml_graph_get_tensor(gf, "out_spec");
    impl.ggmlAssert(sp != null and params.out_spec != null, "clip.cpp", 7523, "sp != nullptr && params->out_spec");
    const out = params.out_spec.?;
    try out.resize(gpa, @intCast(c.ggml_nelements(sp)));
    c.ggml_backend_tensor_get(sp, out.items.ptr, 0, c.ggml_nbytes(sp));
    return true;
}

/// Ports clip_encode's chatterbox `CLIP_GEN_PROCESS_TTS` branch after compute
/// (mtmd/clip.cpp:7529 @cd959bae9): f0 to the harmonic source and its STFT on
/// the host, the vocoder graph through a nested `clip_encode`, then the
/// iSTFT overlap-add and the reference's trim_fade.
pub fn encodeTtsOut(ctx: *clip.Ctx, params: *clip.EncodeParams, gf: ?*c.ggml_cgraph) clip.Error!bool {
    const func = "clip_encode";
    const mel = c.ggml_graph_get_tensor(gf, "out_mel");
    impl.ggmlAssert(mel != null, "clip.cpp", 7531, "mel != nullptr");
    var mel_data: std.ArrayList(f32) = .empty;
    defer mel_data.deinit(gpa);
    try mel_data.resize(gpa, @intCast(c.ggml_nelements(mel)));
    c.ggml_backend_tensor_get(mel, mel_data.items.ptr, 0, c.ggml_nbytes(mel));

    // hift: f0 -> harmonic source -> source stft on the host, then the vocoder graph and the istft
    const f0_t = c.ggml_graph_get_tensor(gf, "out_f0");
    impl.ggmlAssert(f0_t != null, "clip.cpp", 7537, "f0_t != nullptr");
    const n_mel_out: c_int = @intCast(c.ggml_nelements(f0_t));
    const f0 = try gpa.alloc(f32, usz(n_mel_out));
    defer gpa.free(f0);
    c.ggml_backend_tensor_get(f0_t, f0.ptr, 0, c.ggml_nbytes(f0_t));

    // SineGen: f0 upsampled x480, 9 harmonics with cumulative phase, uv gating and noise, merged by l_linear + tanh
    const ups_total = 480;
    const sr: f64 = 24000.0;
    const n_wav: i64 = @as(i64, n_mel_out) * ups_total;
    var lw: [9]f32 = @splat(0.0);
    var lb: f32 = 0.0;
    if (clip.cbxReadTensor(ctx, "a.gen.hift.m_source.l_linear.weight", &lw, lw.len) != 9 or
        clip.cbxReadTensor(ctx, "a.gen.hift.m_source.l_linear.bias", @ptrCast(&lb), 1) != 1)
    {
        impl.logErr("%s: unexpected hift source module\n", .{func});
        return false;
    }

    const srng = &params.gen_state.?.rng;
    var snd: arms_inputs.NormalF32 = .{ .mean = 0.0, .stddev = 1.0 };
    var phase: [9]f64 = undefined;
    for (&phase, 0..) |*p, hh| p.* = if (hh == 0) 0.0 else uniformRealF64(srng, -m_pi, m_pi);
    const src = try gpa.alloc(f32, usz(n_wav));
    defer gpa.free(src);
    var cum: [9]f64 = @splat(0.0);
    var t: i64 = 0;
    while (t < n_wav) : (t += 1) {
        const f = f0[usz(@divTrunc(t, ups_total))];
        const uv: f32 = if (f > 10.0) 1.0 else 0.0; // nsf_voiced_threshold
        const namp = @mulAdd(f32, uv, 0.003, (1.0 - uv) * 0.1 / 3.0);
        var merged = lb;
        for (0..9) |hh| {
            cum[hh] += @as(f64, f) * @as(f64, @floatFromInt(hh + 1)) / sr;
            cum[hh] -= @floor(cum[hh]);
            var sine: f32 = 0.1 * @as(f32, @floatCast(sin(@mulAdd(f64, 2.0 * m_pi, cum[hh], phase[hh]))));
            sine = @mulAdd(f32, sine, uv, namp * snd.next(srng));
            merged = @mulAdd(f32, lw[hh], sine, merged);
        }
        src[usz(t)] = tanhf(merged);
    }

    // stft of the source: n_fft 16, hop 4, hann window, centered
    const n_fft = 16;
    const hop = 4;
    const n_bins = 9;
    var win: [n_fft]f32 = undefined;
    for (&win, 0..) |*w, i| w.* = @mulAdd(f32, -0.5, cosf(2.0 * @as(f32, @floatCast(m_pi)) * @as(f32, @floatFromInt(i)) / @as(f32, n_fft)), 0.5);
    const n_stft: c_int = @intCast(@divTrunc(n_wav, hop) + 1);
    var sstft: std.ArrayList(f32) = .empty;
    defer sstft.deinit(gpa);
    try sstft.resize(gpa, usz(n_stft) * 18);
    var fr: c_int = 0;
    while (fr < n_stft) : (fr += 1) {
        const c0: i64 = @as(i64, fr) * hop - n_fft / 2; // centered, reflect padded
        for (0..n_bins) |k| {
            var re: f64 = 0.0;
            var im: f64 = 0.0;
            for (0..n_fft) |i| {
                var idx: i64 = c0 + @as(i64, @intCast(i));
                if (idx < 0) idx = -idx;
                if (idx >= n_wav) idx = 2 * (n_wav - 1) - idx;
                const v = @as(f64, src[usz(idx)]) * win[i];
                const a = 2.0 * m_pi * @as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(i)) / @as(f64, n_fft);
                const sc = sinCosF64(a);
                re = @mulAdd(f64, v, sc.cos, re);
                im = @mulAdd(f64, -v, sc.sin, im);
            }
            sstft.items[usz(fr) * 18 + k] = @floatCast(re);
            sstft.items[usz(fr) * 18 + 9 + k] = @floatCast(im);
        }
    }

    // vocoder graph on mel + source stft
    var spec: std.ArrayList(f32) = .empty;
    defer spec.deinit(gpa);
    {
        var vp = params.*;
        vp.gen_process = .tts_vocode;
        vp.mel_in = &mel_data;
        vp.sstft_in = &sstft;
        vp.vocode_n_mel = n_mel_out;
        vp.vocode_n_stft = n_stft;
        vp.out_audio = null;
        vp.out_spec = &spec;
        if (!try clip.encode(ctx, &vp)) {
            impl.logErr("%s: vocoder stage failed\n", .{func});
            return false;
        }
    }

    // istft: mag = clipped exp, phase = sin, hann overlap-add
    const n_frames_out: c_int = @intCast(spec.items.len / 18);
    const n_out: i64 = @as(i64, n_frames_out - 1) * hop;
    const acc = try gpa.alloc(f64, usz(n_out + n_fft));
    defer gpa.free(acc);
    @memset(acc, 0.0);
    const wsum = try gpa.alloc(f64, usz(n_out + n_fft));
    defer gpa.free(wsum);
    @memset(wsum, 0.0);
    fr = 0;
    while (fr < n_frames_out) : (fr += 1) {
        var frame: [16]f64 = undefined;
        for (0..n_fft) |i| {
            var v: f64 = 0.0;
            for (0..n_bins) |k| {
                const mag = fmin(exp(@as(f64, spec.items[usz(fr) * 18 + k])), 1e2);
                const ph = sin(@as(f64, spec.items[usz(fr) * 18 + 9 + k]));
                const pc = sinCosF64(ph);
                const re = mag * pc.cos;
                const im = mag * pc.sin;
                const a = 2.0 * m_pi * @as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(i)) / @as(f64, n_fft);
                const w: f64 = if (k == 0 or k == n_fft / 2) 1.0 else 2.0;
                const sc = sinCosF64(a);
                v = @mulAdd(f64, w, @mulAdd(f64, re, sc.cos, -(im * sc.sin)), v);
            }
            frame[i] = v / n_fft;
        }
        const o: usize = usz(@as(i64, fr) * hop);
        for (0..n_fft) |i| {
            acc[o + i] = @mulAdd(f64, frame[i], win[i], acc[o + i]);
            wsum[o + i] = @mulAdd(f64, win[i], win[i], wsum[o + i]);
        }
    }
    impl.ggmlAssert(params.out_audio != null, "clip.cpp", 7644, "params->out_audio");
    const out_audio = params.out_audio.?;
    try out_audio.resize(gpa, @intCast(@min(n_out - n_fft / 2, n_wav)));
    for (out_audio.items, 0..) |*o, i| {
        const j = i + n_fft / 2; // drop the centering pad
        const v = if (wsum[j] > 1e-11) acc[j] / wsum[j] else 0.0;
        o.* = @floatCast(fmax(-0.99, fmin(0.99, v)));
    }

    // trim_fade of the reference: silence 20 ms then fade in 20 ms, hides the artifact of the prompt boundary
    const n_trim: usize = 24000 / 50;
    var i: usize = 0;
    while (i < 2 * n_trim and i < out_audio.items.len) : (i += 1) {
        const g: f64 = if (i < n_trim) 0.0 else (cos(m_pi * (1.0 - @as(f64, @floatFromInt(i - n_trim)) / @as(f64, @floatFromInt(n_trim)))) + 1.0) / 2.0;
        out_audio.items[i] *= @as(f32, @floatCast(g));
    }
    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
