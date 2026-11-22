//! The MiMo-Audio encoder: a Whisper-style conv stem and windowed ViT with a
//! skip connection, a strided downsample, residual vector quantization, the
//! code embeddings, and a group-local transformer before the projection.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/mimo-audio.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const mtmd = @import("../module.zig");

/// The `add_pos` lambda (mimo-audio.cpp:46 @b1b770d90) of `build`. `ctx` is
/// the positions tensor.
fn addPos(ctx: ?*anyopaque, g: *mtmd.graph.Graph, x: mtmd.graph.T, layer: *const mtmd.model.Layer) mtmd.graph.T {
    _ = layer;
    const inp_pos: mtmd.graph.T = @ptrCast(@alignCast(ctx));
    return mtmd.impl.c.ggml_rope_ext(g.ctx0, x, inp_pos, null, g.d_head, mtmd.impl.c.GGML_ROPE_TYPE_NEOX, 0, g.hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
}

/// `skip_capture_il` (mimo-audio.cpp:37 @b1b770d90): encoder_skip_layer_id=3
/// (1-indexed) captures the output of layer index 2.
const skip_capture_il: c_int = 2;

/// The `callback_layer_out` lambda (mimo-audio.cpp:40 @b1b770d90) of `build`. `ctx` is the `skip_hidden` it writes.
fn captureSkip(ctx: ?*anyopaque, g: *mtmd.graph.Graph, layer_cur: mtmd.graph.T, il: c_int) void {
    _ = g;
    const skip_hidden: *mtmd.graph.T = @ptrCast(@alignCast(ctx.?));
    if (il == skip_capture_il) {
        skip_hidden.* = layer_cur;
    }
}

/// The `apply_local_rope` lambda (mimo-audio.cpp:151 @b1b770d90) of `build`.
fn applyLocalRope(g: *const mtmd.graph.Graph, x: mtmd.graph.T, local_pos: mtmd.graph.T, local_rope_theta: f32) mtmd.graph.T {
    return mtmd.impl.c.ggml_rope_ext(g.ctx0, x, local_pos, null, g.d_head, mtmd.impl.c.GGML_ROPE_TYPE_NEOX, 0, local_rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
}

/// Ports `build` (mimo-audio.cpp:3 @b1b770d90), a member of
/// `mtmd_graph_mimo_audio`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_layer = g.n_layer;
    const n_embd = g.n_embd;
    const n_head = g.n_head;
    const d_head = g.d_head;

    const inp = g.buildInpRaw(1); // [n_frames, n_mel, 1]

    var cur = mtmd.impl.c.ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
    cur = mtmd.impl.c.ggml_add(ctx0, cur, model.conv1d_1_b);
    cur = mtmd.impl.c.ggml_gelu_erf(ctx0, cur);

    cur = mtmd.impl.c.ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
    cur = mtmd.impl.c.ggml_add(ctx0, cur, model.conv1d_2_b);
    cur = mtmd.impl.c.ggml_gelu_erf(ctx0, cur);

    var inpL = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, cur)); // [n_embd, n_pos]
    const n_pos = mtmd.graph.ne(inpL, 1);
    g.cb(inpL, "after_conv1d", -1);

    mtmd.impl.ggmlAssert(@as(c_int, @intCast(hparams.wa_pattern_mode.items.len)) == n_layer, "mimo-audio.cpp", 18, "(int) hparams.wa_pattern_mode.size() == n_layer");

    const inp_pos = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, n_pos);
    _ = mtmd.impl.c.ggml_set_name(inp_pos, "mimo_audio_positions");
    mtmd.impl.c.ggml_set_input(inp_pos);

    const full_mask = g.buildInpAttnMask(n_pos, n_pos, 0);
    const window_mask = g.buildInpAttnMask(n_pos, n_pos, 1);

    const attn_mask_layers = std.heap.c_allocator.alloc(mtmd.graph.T, @intCast(n_layer)) catch oom();
    defer std.heap.c_allocator.free(attn_mask_layers);
    for (attn_mask_layers, 0..) |*m, il| {
        m.* = if (hparams.wa_pattern_mode.items[il] == -1) full_mask else window_mask;
    }

    mtmd.impl.ggmlAssert(n_layer > skip_capture_il, "mimo-audio.cpp", 38, "n_layer > skip_capture_il");
    var skip_hidden: mtmd.graph.T = null;

    // the skip connection below must be added before the post-transformer  norm,
    // so build_vit must not apply that norm itself
    const opts: mtmd.graph.VitOpts = .{
        .attn_mask_layers = attn_mask_layers,
        .skip_post_ln = true,
        .callback_layer_out = .{ .ctx = @ptrCast(&skip_hidden), .f = captureSkip },
    };

    inpL = g.buildVit(inpL, n_pos, .normal, hparams.ffn_op, null, .{ .ctx = inp_pos, .f = addPos }, opts);
    inpL = mtmd.impl.c.ggml_reshape_2d(ctx0, inpL, n_embd, n_pos); // build_vit restores a (size-1) batch dim

    mtmd.impl.ggmlAssert(skip_hidden != null, "mimo-audio.cpp", 54, "skip_hidden != nullptr");
    inpL = mtmd.impl.c.ggml_add(ctx0, inpL, skip_hidden);

    inpL = g.buildNorm(inpL, model.post_ln_w, model.post_ln_b, .normal, g.eps, -1);
    g.cb(inpL, "after_transformer", -1);

    // downsample: strided conv (no bias) + gelu + layernorm
    {
        var ds = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, inpL)); // [n_pos, n_embd]
        ds = mtmd.impl.c.ggml_conv_1d(ctx0, model.downsample_conv_w, ds, 2, 0, 1);
        ds = mtmd.impl.c.ggml_gelu_erf(ctx0, ds);
        ds = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, ds)); // [n_embd, n_pos/2]
        ds = g.buildNorm(ds, model.downsample_norm_w, model.downsample_norm_b, .normal, g.eps, -1);
        inpL = ds;
    }
    g.cb(inpL, "after_downsample", -1);

    // RVQ quantize: codebook ne=[dim, max_bins, n_q]
    // quantize input vector to codes (type=I32)
    var codes: std.ArrayList(mtmd.graph.T) = .empty;
    defer codes.deinit(std.heap.c_allocator);
    {
        mtmd.impl.ggmlAssert(model.rvq_codebook != null, "mimo-audio.cpp", 75, "model.rvq_codebook != nullptr");
        const dim = mtmd.graph.ne(model.rvq_codebook, 0);
        mtmd.impl.ggmlAssert(dim == mtmd.graph.ne(inpL, 0), "mimo-audio.cpp", 77, "dim == inpL->ne[0]");
        mtmd.impl.ggmlAssert(@as(i64, @intCast(hparams.rvq_codebook_size.items.len)) == mtmd.graph.ne(model.rvq_codebook, 2), "mimo-audio.cpp", 78, "(int64_t) hparams.rvq_codebook_size.size() == model.rvq_codebook->ne[2]");

        var residual = inpL; // [dim, n_pos_ds]

        for (hparams.rvq_codebook_size.items, 0..) |bins_q, q| {
            const bins: i64 = bins_q;
            var codebook_q = mtmd.impl.c.ggml_view_2d(ctx0, model.rvq_codebook, dim, bins, mtmd.graph.nb(model.rvq_codebook, 1), q * mtmd.graph.nb(model.rvq_codebook, 2));
            codebook_q = mtmd.impl.c.ggml_cont(ctx0, codebook_q);

            var codebook_norm = mtmd.impl.c.ggml_sum_rows(ctx0, mtmd.impl.c.ggml_sqr(ctx0, codebook_q)); // [1, bins]
            codebook_norm = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, codebook_norm)); // [bins, 1]

            const dot = mtmd.impl.c.ggml_mul_mat(ctx0, codebook_q, residual); // [bins, n_pos_ds]
            const scores = mtmd.impl.c.ggml_sub(ctx0, mtmd.impl.c.ggml_scale(ctx0, dot, 2.0), codebook_norm);

            const idx = mtmd.impl.c.ggml_argmax(ctx0, scores); // [n_pos_ds]
            codes.append(std.heap.c_allocator, idx) catch oom();

            const quant = mtmd.impl.c.ggml_get_rows(ctx0, codebook_q, idx); // [dim, n_pos_ds]
            residual = mtmd.impl.c.ggml_sub(ctx0, residual, quant);
            g.cb(idx, "rvq_code", @intCast(q));
        }
    }

    // convert codes to LLM embeddings
    var code_embd_sum: mtmd.graph.T = null;
    {
        mtmd.impl.ggmlAssert(model.mm_a_code_embd != null, "mimo-audio.cpp", 106, "model.mm_a_code_embd != nullptr");
        const dim = mtmd.graph.ne(model.mm_a_code_embd, 0);
        const vocab = mtmd.graph.ne(model.mm_a_code_embd, 1);
        mtmd.impl.ggmlAssert(@as(i64, @intCast(codes.items.len)) == mtmd.graph.ne(model.mm_a_code_embd, 2), "mimo-audio.cpp", 109, "(int64_t) codes.size() == model.mm_a_code_embd->ne[2]");
        mtmd.impl.ggmlAssert(dim == mtmd.graph.ne(inpL, 0), "mimo-audio.cpp", 110, "dim == inpL->ne[0]");

        for (codes.items, 0..) |code_i, i| {
            var table_i = mtmd.impl.c.ggml_view_2d(ctx0, model.mm_a_code_embd, dim, vocab, mtmd.graph.nb(model.mm_a_code_embd, 1), i * mtmd.graph.nb(model.mm_a_code_embd, 2));
            table_i = mtmd.impl.c.ggml_cont(ctx0, table_i);

            const embd_i = mtmd.impl.c.ggml_get_rows(ctx0, table_i, code_i); // [dim, n_pos_ds]
            code_embd_sum = if (code_embd_sum != null) mtmd.impl.c.ggml_add(ctx0, code_embd_sum, embd_i) else embd_i;
        }
        g.cb(code_embd_sum, "code_embd_sum", -1);
    }

    // input_local_transformer
    // groups of `group_size` consecutive downsampled frames are processed together, attending only within their own group.
    // Implemented as a block-diagonal mask + in-group-repeating positions
    // (rather than a real batch dim) - same technique as the encoder's masks above, and as gemma4a's / deepseekocr2's chunked attention.

    // note: hand-rolled here instead of build_vit() because this is a second, independent layer stack
    // (own layer array/count, RMSNorm instead of LN, SiLU FFN, own RoPE theta)

    var projected: mtmd.graph.T = undefined;
    {
        const group_size = hparams.audio_local_group_size;
        mtmd.impl.ggmlAssert(group_size > 0, "mimo-audio.cpp", 134, "group_size > 0");
        const n_pos_ds = mtmd.graph.ne(code_embd_sum, 1);
        const n_groups = @divTrunc(n_pos_ds + group_size - 1, group_size);
        const n_padded = n_groups * group_size;

        var cur_local = code_embd_sum;
        if (n_padded != n_pos_ds) {
            cur_local = mtmd.impl.c.ggml_pad(ctx0, cur_local, 0, @truncate(n_padded - n_pos_ds), 0, 0);
        }

        const local_pos = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, n_padded);
        _ = mtmd.impl.c.ggml_set_name(local_pos, "mimo_audio_local_positions");
        mtmd.impl.c.ggml_set_input(local_pos);

        const local_mask = g.buildInpAttnMask(n_padded, n_padded, 2);

        const local_rope_theta: f32 = 640000.0; // audio_config.rope_theta (differs from the encoder's)

        var il: c_int = 0;
        while (il < hparams.audio_local_n_layer) : (il += 1) {
            const layer = &model.mm_a_local_layers.items[@intCast(il)];

            const attn_in = g.buildNorm(cur_local, layer.ln_1_w, null, .rms, g.eps, il);

            var Qcur = g.buildMm(layer.q_w, attn_in);
            if (layer.q_b != null) {
                Qcur = mtmd.impl.c.ggml_add(ctx0, Qcur, layer.q_b);
            }
            var Kcur = g.buildMm(layer.k_w, attn_in);
            if (layer.k_b != null) {
                Kcur = mtmd.impl.c.ggml_add(ctx0, Kcur, layer.k_b);
            }
            var Vcur = g.buildMm(layer.v_w, attn_in);
            if (layer.v_b != null) {
                Vcur = mtmd.impl.c.ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_padded);
            Kcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_padded);
            Vcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_padded);

            Qcur = applyLocalRope(g, Qcur, local_pos, local_rope_theta);
            Kcur = applyLocalRope(g, Kcur, local_pos, local_rope_theta);

            const attn_out = g.buildAttn(layer.o_w, null, Qcur, Kcur, Vcur, local_mask, g.kq_scale, il, null);
            cur_local = mtmd.impl.c.ggml_add(ctx0, cur_local, attn_out);

            const ffn_in = g.buildNorm(cur_local, layer.ln_2_w, null, .rms, g.eps, il);
            const ffn_out = g.buildFfn(ffn_in, layer.ff_up_w, null, layer.ff_gate_w, null, layer.ff_down_w, null, .silu, il);
            cur_local = mtmd.impl.c.ggml_add(ctx0, cur_local, ffn_out);
        }

        cur_local = g.buildNorm(cur_local, model.mm_a_local_norm_w, null, .rms, g.eps, -1);
        g.cb(cur_local, "after_local_transformer", -1);

        // flatten each group of `group_size` frames into one (group_size*n_embd)-dim vector
        // (matching AudioProjection's flattened input)
        const grouped = mtmd.impl.c.ggml_reshape_2d(ctx0, cur_local, @as(i64, n_embd) * group_size, n_groups);

        // AudioProjection: Linear (no bias) -> GELU -> Linear (no bias)
        projected = g.buildFfn(grouped, model.mm_1_w, null, null, null, model.mm_2_w, null, .gelu_erf, -1);
        g.cb(projected, "after_projection", -1);
    }

    mtmd.impl.c.ggml_build_forward_expand(g.gf, projected);
    return g.gf;
}

fn oom() noreturn {
    mtmd.impl.abort("mimo-audio.cpp", 0, "out of memory");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
