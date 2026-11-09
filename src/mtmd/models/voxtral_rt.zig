//! The Voxtral Realtime encoder: a causal whisper-like encoder with a sliding
//! window KV cache. The input holds 2 context mel frames followed by the new
//! frames, so the causal conv stem sees the same left context as on the full
//! spectrogram.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/voxtral-rt.cpp` at v0.8.0
//! (`8642541b2`), and the `PROJECTOR_TYPE_VOXTRAL_RT` arm of
//! `clip_get_graph_builder` (`llama.cpp/tools/mtmd/clip.cpp`). Each
//! declaration names what it ports and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const ne = graph.ne;
const nb = graph.nb;

/// Ports `build` (voxtral-rt.cpp:5 @8642541b2), a member of
/// `clip_graph_voxtral_rt`, with the state `clip_get_graph_builder`
/// (mtmd/clip.cpp:979 @8642541b2) passes it: null during warmup, the cache is
/// then a graph input.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    const state: ?*clip.EncState = if (params) |p| p.enc_state else null;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const d_head = g.d_head;
    const n_head = g.n_head;
    const n_head_kv = g.n_head_kv;
    const eps = g.eps;

    const n_frames_in = g.img.nx();
    const n_new = @divTrunc(n_frames_in - 2, 2); // encoder frames, the conv stem has stride 2
    const n_window = hparams.attn_window_size;
    const stack = hparams.proj_stack_factor;
    const n_embd_kv = n_head_kv * d_head;
    const n_ctx: c_int = if (state) |s| s.n_ctx else n_window + n_new;

    impl.ggmlAssert(n_window > 0 and stack > 0, "voxtral-rt.cpp", 13, "n_window > 0 && stack > 0");
    impl.ggmlAssert(n_new > 0 and @rem(n_new, stack) == 0, "voxtral-rt.cpp", 14, "n_new > 0 && n_new % stack == 0");
    impl.ggmlAssert(n_new <= n_ctx - n_window, "voxtral-rt.cpp", 15, "n_new <= n_ctx - n_window");

    const inp = g.buildInpRaw(1); // [n_frames_in, n_mel]

    const stem_tail = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, 1, n_embd);
    _ = c.ggml_set_name(stem_tail, "inp_stem_tail");
    c.ggml_set_input(stem_tail);

    const pos = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_new);
    _ = c.ggml_set_name(pos, "inp_pos");
    c.ggml_set_input(pos);

    const kv_idx = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I64, n_new);
    _ = c.ggml_set_name(kv_idx, "inp_kv_idx");
    c.ggml_set_input(kv_idx);

    const kq_mask = g.buildInpAttnMask(n_ctx, n_new, 0);

    // causal conv stem
    var cur = c.ggml_conv_1d(ctx0, model.conv1d_1_w, inp, 1, 0, 1);
    cur = c.ggml_add(ctx0, cur, model.conv1d_1_b);
    cur = c.ggml_gelu_erf(ctx0, cur); // [2*n_new, n_embd]

    // the last frame is the left context of the second conv on the next call
    var tail = c.ggml_view_2d(ctx0, cur, 1, n_embd, nb(cur, 1), @as(usize, @intCast(2 * n_new - 1)) * c.ggml_element_size(cur));
    tail = c.ggml_cont(ctx0, tail);
    _ = c.ggml_set_name(tail, "out_stem_tail");
    c.ggml_set_output(tail);
    c.ggml_build_forward_expand(g.gf, tail);

    cur = c.ggml_concat(ctx0, stem_tail, cur, 0); // [2*n_new + 1, n_embd]
    cur = c.ggml_conv_1d(ctx0, model.conv1d_2_w, cur, 2, 0, 1);
    cur = c.ggml_add(ctx0, cur, model.conv1d_2_b);
    cur = c.ggml_gelu_erf(ctx0, cur); // [n_new, n_embd]
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur)); // [n_embd, n_new]
    g.cb(cur, "after_conv1d", -1);

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const ilu: usize = @intCast(il);
        const layer = &model.layers.items[ilu];

        var x = g.buildNorm(cur, layer.ln_1_w, null, .rms, eps, il);

        var q = c.ggml_add(ctx0, g.buildMm(layer.q_w, x), layer.q_b);
        var k = g.buildMm(layer.k_w, x);
        const v = c.ggml_add(ctx0, g.buildMm(layer.v_w, x), layer.v_b);

        q = c.ggml_reshape_3d(ctx0, q, d_head, n_head, n_new);
        k = c.ggml_reshape_3d(ctx0, k, d_head, n_head_kv, n_new);

        // interleaved rope on absolute frame positions
        q = c.ggml_rope_ext(ctx0, q, pos, null, d_head, c.GGML_ROPE_TYPE_NORMAL, 0, hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        k = c.ggml_rope_ext(ctx0, k, pos, null, d_head, c.GGML_ROPE_TYPE_NORMAL, 0, hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        g.cb(q, "q", il);
        g.cb(k, "k", il);
        g.cb(v, "v", il);

        var k_cache: graph.T = undefined;
        var v_cache: graph.T = undefined;
        if (state) |s| {
            k_cache = s.k.items[ilu];
            v_cache = s.v.items[ilu];
        } else {
            k_cache = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F16, n_embd_kv, n_ctx);
            v_cache = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F16, n_embd_kv, n_ctx);
            c.ggml_set_input(k_cache);
            c.ggml_set_input(v_cache);
        }

        // write the new frames into the ring, the mask selects the window out of the whole ring
        c.ggml_build_forward_expand(g.gf, c.ggml_set_rows(ctx0, k_cache, c.ggml_reshape_2d(ctx0, k, n_embd_kv, n_new), kv_idx));
        c.ggml_build_forward_expand(g.gf, c.ggml_set_rows(ctx0, v_cache, v, kv_idx));

        const k_all = c.ggml_reshape_3d(ctx0, k_cache, d_head, n_head_kv, n_ctx);
        const v_all = c.ggml_reshape_3d(ctx0, v_cache, d_head, n_head_kv, n_ctx);

        x = g.buildAttn(layer.o_w, layer.o_b, q, k_all, v_all, kq_mask, g.kq_scale, il, null);
        cur = c.ggml_add(ctx0, cur, x);

        x = g.buildNorm(cur, layer.ln_2_w, null, .rms, eps, il);
        x = g.buildFfn(x, layer.ff_up_w, null, layer.ff_gate_w, null, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
        cur = c.ggml_add(ctx0, cur, x);
        g.cb(cur, "layer_out", il);
    }

    cur = g.buildNorm(cur, model.post_ln_w, null, .rms, eps, -1);
    g.cb(cur, "after_transformer", -1);

    cur = g.buildStack(cur, stack, n_embd);
    cur = g.buildMm(model.mm_1_w, cur);
    cur = c.ggml_gelu_erf(ctx0, cur);
    cur = g.buildMm(model.mm_2_w, cur);
    g.cb(cur, "projected", -1);

    c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
