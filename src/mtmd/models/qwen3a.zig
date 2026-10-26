//! The Qwen3-ASR audio encoder: the mel spectrogram in 100-frame chunks
//! through three strided conv2d blocks, a ViT with per-chunk positions, and
//! an MLP projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/qwen3a.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration names what it ports and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

/// The `conv_block` lambda (qwen3a.cpp:30 @c1d0e7a00) of `build`: conv
/// output `[OW, OH, C_out, n_chunks]`.
fn convBlock(ctx0: ?*c.ggml_context, x_in: T, w: T, b: T) T {
    var x = c.ggml_conv_2d(ctx0, w, x_in, 2, 2, 1, 1, 1, 1);
    if (b != null) {
        x = c.ggml_add(ctx0, x, c.ggml_reshape_4d(ctx0, b, 1, 1, ne(x, 2), 1));
    }
    return c.ggml_gelu_erf(ctx0, x);
}

/// Ports `build` (qwen3a.cpp:3 @c1d0e7a00), a member of
/// `clip_graph_qwen3a`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    // Ref implementation: https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py

    // inp_raw: [n_frames, n_mel, 1]  (nx=n_frames, ny=n_mel)
    var inp = g.buildInpRaw(1);

    const n_frames = ne(inp, 0); // total frames, padded to multiple of chunk_size
    const n_mel = ne(inp, 1); // 128
    const chunk_size: i64 = 100; // n_window * 2 (n_window=50 from model config)
    const n_chunks = @divTrunc(n_frames, chunk_size);

    impl.ggmlAssert(@rem(n_frames, chunk_size) == 0, "qwen3a.cpp", 14, "n_frames % chunk_size == 0"); // preprocessor should already pad the input
    impl.ggmlAssert(inp.?.type == c.GGML_TYPE_F32, "qwen3a.cpp", 15, "inp->type == GGML_TYPE_F32");

    // View mel spectrogram as batched 100-frame chunks: [chunk_size, n_mel, 1, n_chunks]
    inp = c.ggml_view_4d(
        ctx0,
        inp,
        chunk_size,
        n_mel,
        1,
        n_chunks,
        @intCast(n_frames * @sizeOf(f32)), // nb[1]: stride over mel bins
        @intCast(chunk_size * @sizeOf(f32)), // nb[2]: stride for C=1 (unused)
        @intCast(chunk_size * @sizeOf(f32)), // nb[3]: stride over chunks
        0,
    );
    inp = c.ggml_cont(ctx0, inp);
    g.cb(inp, "inp_chunks", -1);

    // 3 x conv2d + gelu
    {
        inp = convBlock(ctx0, inp, model.conv2d_1_w, model.conv2d_1_b);
        inp = convBlock(ctx0, inp, model.conv2d_2_w, model.conv2d_2_b);
        inp = convBlock(ctx0, inp, model.conv2d_3_w, model.conv2d_3_b);
        // inp: [OW=13, OH=16, OC=480, n_chunks]
        g.cb(inp, "after_conv_blocks", -1);
    }

    // permute [OW=25, OH=16, OC=480, n_chunks] -> [OH=16, OC=480, OW=25, n_chunks]
    // reshape to [OH*OC=7680, OW*n_chunks]
    // feature index h+16*c = c*16+f (matches python code)
    inp = c.ggml_cont(ctx0, c.ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = c.ggml_reshape_2d(ctx0, inp, ne(inp, 0) * ne(inp, 1), ne(inp, 2) * ne(inp, 3));

    // Project to d_model: [d_model, 25*n_chunks]
    inp = c.ggml_mul_mat(ctx0, model.conv_out_w, inp);
    if (model.conv_out_b != null) {
        inp = c.ggml_add(ctx0, inp, model.conv_out_b);
    }
    g.cb(inp, "after_conv_out", -1);

    const n_pos = ne(inp, 1); // 25 * n_chunks

    // Per-chunk positional embeddings: repeat pos[0:13] for each chunk
    // (position indices reset 0..12 per chunk, not sequential across chunks)
    {
        const tokens_per_chunk = @divTrunc(n_pos, n_chunks); // 13
        const pos_tmp = c.ggml_view_2d(ctx0, model.position_embeddings, ne(model.position_embeddings, 0), tokens_per_chunk, nb(model.position_embeddings, 1), 0);
        const tgt = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, ne(model.position_embeddings, 0), n_pos);
        inp = c.ggml_add(ctx0, inp, c.ggml_repeat(ctx0, pos_tmp, tgt));
    }

    var cur = g.buildVit(inp, n_pos, .normal, hparams.ffn_op, null, null, .{}); // pos embd already added above
    g.cb(cur, "after_transformer", -1);

    // MLP projector
    cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);
    g.cb(cur, "projected", -1);

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
