//! The Whisper audio encoder and the projectors that sit on it: Ultravox,
//! Qwen2-Audio, Voxtral, Music Flamingo, MERaLiON and GLM-A.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/whisper-enc.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const mtmd = @import("../module.zig");

/// Ports `build` (whisper-enc.cpp:3 @b1b770d90), a member of
/// `mtmd_graph_whisper_enc`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    const n_frames = g.img.nx();
    const n_pos = @divTrunc(n_frames, 2);
    mtmd.impl.ggmlAssert(mtmd.graph.ne(model.position_embeddings, 1) >= n_pos, "whisper-enc.cpp", 6, "model.position_embeddings->ne[1] >= n_pos");

    var inp = g.buildInpRaw(1);

    // conv1d block
    {
        // convolution + gelu
        var cur = mtmd.impl.c.ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
        cur = mtmd.impl.c.ggml_add(ctx0, cur, model.conv1d_1_b);

        cur = mtmd.impl.c.ggml_gelu_erf(ctx0, cur);

        cur = mtmd.impl.c.ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
        cur = mtmd.impl.c.ggml_add(ctx0, cur, model.conv1d_2_b);

        cur = mtmd.impl.c.ggml_gelu_erf(ctx0, cur);
        // transpose
        inp = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, cur));
        g.cb(inp, "after_conv1d", -1);
    }

    // sanity check (only check one layer, but it should be the same for all)
    const l0 = &model.layers.items[0];
    mtmd.impl.ggmlAssert(l0.ln_1_w != null and l0.ln_1_b != null, "whisper-enc.cpp", 28, "model.layers[0].ln_1_w && model.layers[0].ln_1_b");
    mtmd.impl.ggmlAssert(l0.ln_2_w != null and l0.ln_2_b != null, "whisper-enc.cpp", 29, "model.layers[0].ln_2_w && model.layers[0].ln_2_b");
    mtmd.impl.ggmlAssert(l0.q_b != null, "whisper-enc.cpp", 30, "model.layers[0].q_b");
    mtmd.impl.ggmlAssert(l0.v_b != null, "whisper-enc.cpp", 31, "model.layers[0].v_b");
    mtmd.impl.ggmlAssert(l0.k_b == null, "whisper-enc.cpp", 32, "!model.layers[0].k_b"); // no bias for k

    const pos_embd_selected = mtmd.impl.c.ggml_view_2d(ctx0, model.position_embeddings, mtmd.graph.ne(model.position_embeddings, 0), n_pos, mtmd.graph.nb(model.position_embeddings, 1), 0);
    var cur = g.buildVit(inp, n_pos, .normal, hparams.ffn_op, pos_embd_selected, null, .{});

    g.cb(cur, "after_transformer", -1);

    if (model.audioHasStackFrames()) {
        // StackAudioFrames
        // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
        cur = g.buildStack(cur, hparams.proj_stack_factor, g.n_embd);
        g.cb(cur, "after_stacked", -1);
    }

    switch (g.proj_type) {
        .ultravox => {
            // UltravoxProjector
            // pre-norm
            cur = mtmd.impl.c.ggml_rms_norm(ctx0, cur, 1e-6);
            cur = mtmd.impl.c.ggml_mul(ctx0, cur, model.mm_norm_pre_w);

            // ffn in
            cur = g.buildMm(model.mm_1_w, cur);

            // swiglu
            // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
            cur = mtmd.impl.c.ggml_swiglu_swapped(ctx0, cur);

            // mid-norm
            cur = mtmd.impl.c.ggml_rms_norm(ctx0, cur, 1e-6);
            cur = mtmd.impl.c.ggml_mul(ctx0, cur, model.mm_norm_mid_w);

            // ffn out
            cur = g.buildMm(model.mm_2_w, cur);
        },
        .qwen2a => {
            // projector
            cur = g.buildMm(model.mm_fc_w, cur);
            cur = mtmd.impl.c.ggml_add(ctx0, cur, model.mm_fc_b);
        },
        .voxtral => {
            // projector
            cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);
        },
        .music_flamingo => {
            // projector
            cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);
        },
        .meralion => {
            // stack (above) -> ln -> linear0+silu -> GLU -> out
            cur = mtmd.impl.c.ggml_norm(ctx0, cur, hparams.eps);
            cur = mtmd.impl.c.ggml_mul(ctx0, cur, model.mm_norm_pre_w);
            cur = mtmd.impl.c.ggml_add(ctx0, cur, model.mm_norm_pre_b);

            cur = mtmd.impl.c.ggml_mul_mat(ctx0, model.mm_0_w, cur);
            cur = mtmd.impl.c.ggml_add(ctx0, cur, model.mm_0_b);
            cur = mtmd.impl.c.ggml_silu(ctx0, cur);

            var gate = mtmd.impl.c.ggml_mul_mat(ctx0, model.mm_1_w, cur);
            gate = mtmd.impl.c.ggml_add(ctx0, gate, model.mm_1_b);
            gate = mtmd.impl.c.ggml_silu(ctx0, gate);

            var pool = mtmd.impl.c.ggml_mul_mat(ctx0, model.mm_2_w, cur);
            pool = mtmd.impl.c.ggml_add(ctx0, pool, model.mm_2_b);

            cur = mtmd.impl.c.ggml_mul(ctx0, gate, pool);

            cur = mtmd.impl.c.ggml_mul_mat(ctx0, model.mm_3_w, cur);
            cur = mtmd.impl.c.ggml_add(ctx0, cur, model.mm_3_b);
        },
        .glma => {
            cur = mtmd.impl.c.ggml_norm(ctx0, cur, hparams.eps);
            cur = mtmd.impl.c.ggml_mul(ctx0, cur, model.mm_norm_pre_w);
            cur = mtmd.impl.c.ggml_add(ctx0, cur, model.mm_norm_pre_b);
            cur = g.buildStack(cur, hparams.proj_stack_factor, g.n_embd);
            cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, hparams.ffn_op, 0);
            cur = mtmd.impl.c.ggml_concat(ctx0, model.mm_boi, cur, 1);
            cur = mtmd.impl.c.ggml_concat(ctx0, cur, model.mm_eoi, 1);
        },
        else => mtmd.impl.abort("whisper-enc.cpp", 129, "build: unknown projector type"),
    }

    g.cb(cur, "projected", -1);

    mtmd.impl.c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
