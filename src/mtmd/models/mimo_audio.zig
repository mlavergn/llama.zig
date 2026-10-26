//! The MiMo-Audio encoder: a Whisper-style conv stem and windowed ViT with a
//! skip connection, a strided downsample, residual vector quantization, the
//! code embeddings, and a group-local transformer before the projection.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/mimo-audio.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;
const T = graph.T;
const gpa = impl.gpa;
const ne = graph.ne;
const nb = graph.nb;

/// The `add_pos` lambda (mimo-audio.cpp:51 @c1d0e7a00) of `build`. `ctx` is
/// the positions tensor.
fn addPos(ctx: ?*anyopaque, g: *graph.Graph, x: T, layer: *const model_mod.Layer) T {
    _ = layer;
    const inp_pos: T = @ptrCast(@alignCast(ctx));
    return c.ggml_rope_ext(g.ctx0, x, inp_pos, null, g.d_head, c.GGML_ROPE_TYPE_NEOX, 0, g.hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
}

/// `skip_capture_il` (mimo-audio.cpp:42 @c1d0e7a00): encoder_skip_layer_id=3
/// (1-indexed) captures the output of layer index 2.
const skip_capture_il: c_int = 2;

/// The `callback_layer_out` lambda (mimo-audio.cpp:45 @c1d0e7a00) of `build`. `ctx` is the `skip_hidden` it writes.
fn captureSkip(ctx: ?*anyopaque, g: *graph.Graph, layer_cur: T, il: c_int) void {
    _ = g;
    const skip_hidden: *T = @ptrCast(@alignCast(ctx.?));
    if (il == skip_capture_il) {
        skip_hidden.* = layer_cur;
    }
}

/// The `apply_local_rope` lambda (mimo-audio.cpp:158 @c1d0e7a00) of `build`.
fn applyLocalRope(g: *const graph.Graph, x: T, local_pos: T, local_rope_theta: f32) T {
    return c.ggml_rope_ext(g.ctx0, x, local_pos, null, g.d_head, c.GGML_ROPE_TYPE_NEOX, 0, local_rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
}

/// Ports `build` (mimo-audio.cpp:3 @c1d0e7a00), a member of
/// `clip_graph_mimo_audio`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_layer = g.n_layer;
    const n_embd = g.n_embd;
    const n_head = g.n_head;
    const d_head = g.d_head;

    const inp = g.buildInpRaw(1); // [n_frames, n_mel, 1]

    var cur = c.ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
    cur = c.ggml_add(ctx0, cur, model.conv1d_1_b);
    cur = c.ggml_gelu_erf(ctx0, cur);

    cur = c.ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
    cur = c.ggml_add(ctx0, cur, model.conv1d_2_b);
    cur = c.ggml_gelu_erf(ctx0, cur);

    var inpL = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur)); // [n_embd, n_pos]
    const n_pos = ne(inpL, 1);
    g.cb(inpL, "after_conv1d", -1);

    impl.ggmlAssert(@as(c_int, @intCast(hparams.wa_pattern_mode.items.len)) == n_layer, "mimo-audio.cpp", 18, "(int) hparams.wa_pattern_mode.size() == n_layer");

    const inp_pos = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos);
    _ = c.ggml_set_name(inp_pos, "mimo_audio_positions");
    c.ggml_set_input(inp_pos);

    const full_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_pos, n_pos);
    _ = c.ggml_set_name(full_mask, "mimo_audio_full_mask");
    c.ggml_set_input(full_mask);

    const window_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_pos, n_pos);
    _ = c.ggml_set_name(window_mask, "mimo_audio_window_mask");
    c.ggml_set_input(window_mask);

    const attn_mask_layers = gpa.alloc(T, @intCast(n_layer)) catch oom();
    defer gpa.free(attn_mask_layers);
    for (attn_mask_layers, 0..) |*m, il| {
        m.* = if (hparams.wa_pattern_mode.items[il] == -1) full_mask else window_mask;
    }

    impl.ggmlAssert(n_layer > skip_capture_il, "mimo-audio.cpp", 43, "n_layer > skip_capture_il");
    var skip_hidden: T = null;

    // the skip connection below must be added before the post-transformer  norm,
    // so build_vit must not apply that norm itself
    const opts: graph.VitOpts = .{
        .attn_mask_layers = attn_mask_layers,
        .skip_post_ln = true,
        .callback_layer_out = .{ .ctx = @ptrCast(&skip_hidden), .f = captureSkip },
    };

    inpL = g.buildVit(inpL, n_pos, .normal, hparams.ffn_op, null, .{ .ctx = inp_pos, .f = addPos }, opts);
    inpL = c.ggml_reshape_2d(ctx0, inpL, n_embd, n_pos); // build_vit restores a (size-1) batch dim

    impl.ggmlAssert(skip_hidden != null, "mimo-audio.cpp", 59, "skip_hidden != nullptr");
    inpL = c.ggml_add(ctx0, inpL, skip_hidden);

    inpL = g.buildNorm(inpL, model.post_ln_w, model.post_ln_b, .normal, g.eps, -1);
    g.cb(inpL, "after_transformer", -1);

    // downsample: strided conv (no bias) + gelu + layernorm
    {
        var ds = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inpL)); // [n_pos, n_embd]
        ds = c.ggml_conv_1d(ctx0, model.downsample_conv_w, ds, 2, 0, 1);
        ds = c.ggml_gelu_erf(ctx0, ds);
        ds = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, ds)); // [n_embd, n_pos/2]
        ds = g.buildNorm(ds, model.downsample_norm_w, model.downsample_norm_b, .normal, g.eps, -1);
        inpL = ds;
    }
    g.cb(inpL, "after_downsample", -1);

    // RVQ quantize: codebook ne=[dim, max_bins, n_q]
    // quantize input vector to codes (type=I32)
    var codes: std.ArrayList(T) = .empty;
    defer codes.deinit(gpa);
    {
        impl.ggmlAssert(model.rvq_codebook != null, "mimo-audio.cpp", 80, "model.rvq_codebook != nullptr");
        const dim = ne(model.rvq_codebook, 0);
        impl.ggmlAssert(dim == ne(inpL, 0), "mimo-audio.cpp", 82, "dim == inpL->ne[0]");
        impl.ggmlAssert(@as(i64, @intCast(hparams.rvq_codebook_size.items.len)) == ne(model.rvq_codebook, 2), "mimo-audio.cpp", 83, "(int64_t) hparams.rvq_codebook_size.size() == model.rvq_codebook->ne[2]");

        var residual = inpL; // [dim, n_pos_ds]

        for (hparams.rvq_codebook_size.items, 0..) |bins_q, q| {
            const bins: i64 = bins_q;
            var codebook_q = c.ggml_view_2d(ctx0, model.rvq_codebook, dim, bins, nb(model.rvq_codebook, 1), q * nb(model.rvq_codebook, 2));
            codebook_q = c.ggml_cont(ctx0, codebook_q);

            var codebook_norm = c.ggml_sum_rows(ctx0, c.ggml_sqr(ctx0, codebook_q)); // [1, bins]
            codebook_norm = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, codebook_norm)); // [bins, 1]

            const dot = c.ggml_mul_mat(ctx0, codebook_q, residual); // [bins, n_pos_ds]
            const scores = c.ggml_sub(ctx0, c.ggml_scale(ctx0, dot, 2.0), codebook_norm);

            const idx = c.ggml_argmax(ctx0, scores); // [n_pos_ds]
            codes.append(gpa, idx) catch oom();

            const quant = c.ggml_get_rows(ctx0, codebook_q, idx); // [dim, n_pos_ds]
            residual = c.ggml_sub(ctx0, residual, quant);
            g.cb(idx, "rvq_code", @intCast(q));
        }
    }

    // convert codes to LLM embeddings
    var code_embd_sum: T = null;
    {
        impl.ggmlAssert(model.mm_a_code_embd != null, "mimo-audio.cpp", 111, "model.mm_a_code_embd != nullptr");
        const dim = ne(model.mm_a_code_embd, 0);
        const vocab = ne(model.mm_a_code_embd, 1);
        impl.ggmlAssert(@as(i64, @intCast(codes.items.len)) == ne(model.mm_a_code_embd, 2), "mimo-audio.cpp", 114, "(int64_t) codes.size() == model.mm_a_code_embd->ne[2]");
        impl.ggmlAssert(dim == ne(inpL, 0), "mimo-audio.cpp", 115, "dim == inpL->ne[0]");

        for (codes.items, 0..) |code_i, i| {
            var table_i = c.ggml_view_2d(ctx0, model.mm_a_code_embd, dim, vocab, nb(model.mm_a_code_embd, 1), i * nb(model.mm_a_code_embd, 2));
            table_i = c.ggml_cont(ctx0, table_i);

            const embd_i = c.ggml_get_rows(ctx0, table_i, code_i); // [dim, n_pos_ds]
            code_embd_sum = if (code_embd_sum != null) c.ggml_add(ctx0, code_embd_sum, embd_i) else embd_i;
        }
        g.cb(code_embd_sum, "code_embd_sum", -1);
    }

    // input_local_transformer
    // groups of `group_size` consecutive downsampled frames are processed together, attending only within their own group.
    // Implemented as a block-diagonal mask + in-group-repeating positions
    // (rather than a real batch dim) - same technique as the encoder's masks above, and as gemma4a's / deepseekocr2's chunked attention.

    // note: hand-rolled here instead of build_vit() because this is a second, independent layer stack
    // (own layer array/count, RMSNorm instead of LN, SiLU FFN, own RoPE theta)

    var projected: T = undefined;
    {
        const group_size = hparams.audio_local_group_size;
        impl.ggmlAssert(group_size > 0, "mimo-audio.cpp", 139, "group_size > 0");
        const n_pos_ds = ne(code_embd_sum, 1);
        const n_groups = @divTrunc(n_pos_ds + group_size - 1, group_size);
        const n_padded = n_groups * group_size;

        var cur_local = code_embd_sum;
        if (n_padded != n_pos_ds) {
            cur_local = c.ggml_pad(ctx0, cur_local, 0, @truncate(n_padded - n_pos_ds), 0, 0);
        }

        const local_pos = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_padded);
        _ = c.ggml_set_name(local_pos, "mimo_audio_local_positions");
        c.ggml_set_input(local_pos);

        const local_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_padded, n_padded);
        _ = c.ggml_set_name(local_mask, "mimo_audio_local_mask");
        c.ggml_set_input(local_mask);

        const local_rope_theta: f32 = 640000.0; // audio_config.rope_theta (differs from the encoder's)

        var il: c_int = 0;
        while (il < hparams.audio_local_n_layer) : (il += 1) {
            const layer = &model.mm_a_local_layers.items[@intCast(il)];

            const attn_in = g.buildNorm(cur_local, layer.ln_1_w, null, .rms, g.eps, il);

            var Qcur = g.buildMm(layer.q_w, attn_in);
            if (layer.q_b != null) {
                Qcur = c.ggml_add(ctx0, Qcur, layer.q_b);
            }
            var Kcur = g.buildMm(layer.k_w, attn_in);
            if (layer.k_b != null) {
                Kcur = c.ggml_add(ctx0, Kcur, layer.k_b);
            }
            var Vcur = g.buildMm(layer.v_w, attn_in);
            if (layer.v_b != null) {
                Vcur = c.ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = c.ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_padded);
            Kcur = c.ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_padded);
            Vcur = c.ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_padded);

            Qcur = applyLocalRope(g, Qcur, local_pos, local_rope_theta);
            Kcur = applyLocalRope(g, Kcur, local_pos, local_rope_theta);

            const attn_out = g.buildAttn(layer.o_w, null, Qcur, Kcur, Vcur, local_mask, g.kq_scale, il, null);
            cur_local = c.ggml_add(ctx0, cur_local, attn_out);

            const ffn_in = g.buildNorm(cur_local, layer.ln_2_w, null, .rms, g.eps, il);
            const ffn_out = g.buildFfn(ffn_in, layer.ff_up_w, null, layer.ff_gate_w, null, layer.ff_down_w, null, .silu, il);
            cur_local = c.ggml_add(ctx0, cur_local, ffn_out);
        }

        cur_local = g.buildNorm(cur_local, model.mm_a_local_norm_w, null, .rms, g.eps, -1);
        g.cb(cur_local, "after_local_transformer", -1);

        // flatten each group of `group_size` frames into one (group_size*n_embd)-dim vector
        // (matching AudioProjection's flattened input)
        const grouped = c.ggml_reshape_2d(ctx0, cur_local, @as(i64, n_embd) * group_size, n_groups);

        // AudioProjection: Linear (no bias) -> GELU -> Linear (no bias)
        projected = g.buildFfn(grouped, model.mm_1_w, null, null, null, model.mm_2_w, null, .gelu_erf, -1);
        g.cb(projected, "after_projection", -1);
    }

    c.ggml_build_forward_expand(g.gf, projected);
    return g.gf;
}

fn oom() noreturn {
    impl.abort("mimo-audio.cpp", 0, "out of memory");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
