//! The pocket-tts voice-prompt encoder: a raw 24 kHz waveform to one
//! conditioning row per 12.5 Hz frame, through the mimi encoder (SEANet,
//! transformer, downsample) and `flow_lm.speaker_proj_weight`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/pockettts-spkenc.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// Ports `tfm_layer_forward` (pockettts-spkenc.cpp:7 @b1b770d90), a member
/// of `mtmd_graph_pockettts_spkenc`: a pre-norm block with layer scale on
/// both residual paths, see mimi_transformer.py.
fn tfmLayerForward(g: *const mtmd.graph.Graph, cur_in: mtmd.graph.T, layer: *const mtmd.model.Layer, inp_pos: mtmd.graph.T, kq_mask: mtmd.graph.T, il: c_int) mtmd.graph.T {
    const ctx0 = g.ctx0;
    var inp = cur_in;

    var cur = g.buildNorm(cur_in, layer.ln_1_w, layer.ln_1_b, .normal, g.eps, il);

    var Qcur = g.buildMm(layer.q_w, cur);
    var Kcur = g.buildMm(layer.k_w, cur);
    var Vcur = g.buildMm(layer.v_w, cur);

    const n_pos = mtmd.graph.ne(cur, 1);
    Qcur = llama.impl.llama_c.ggml_reshape_3d(ctx0, Qcur, g.d_head, g.n_head, n_pos);
    Kcur = llama.impl.llama_c.ggml_reshape_3d(ctx0, Kcur, g.d_head, g.n_head, n_pos);
    Vcur = llama.impl.llama_c.ggml_reshape_3d(ctx0, Vcur, g.d_head, g.n_head, n_pos);

    Qcur = llama.impl.llama_c.ggml_rope_ext(ctx0, Qcur, inp_pos, null, g.d_head, llama.impl.llama_c.GGML_ROPE_TYPE_NORMAL, 0, g.hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
    Kcur = llama.impl.llama_c.ggml_rope_ext(ctx0, Kcur, inp_pos, null, g.d_head, llama.impl.llama_c.GGML_ROPE_TYPE_NORMAL, 0, g.hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

    cur = g.buildAttn(layer.o_w, null, Qcur, Kcur, Vcur, kq_mask, g.kq_scale, il, null);
    cur = llama.impl.llama_c.ggml_mul(ctx0, cur, layer.ls_1_w);
    cur = llama.impl.llama_c.ggml_add(ctx0, cur, inp);

    inp = cur;
    cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .normal, g.eps, il);
    cur = g.buildFfn(cur, layer.ff_up_w, null, null, null, layer.ff_down_w, null, .gelu, il);
    cur = llama.impl.llama_c.ggml_mul(ctx0, cur, layer.ls_2_w);
    cur = llama.impl.llama_c.ggml_add(ctx0, cur, inp);

    return cur;
}

/// Ports `build` (pockettts-spkenc.cpp:39 @b1b770d90), a member of
/// `mtmd_graph_pockettts_spkenc`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*llama.impl.llama_c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    var arena_state: std.heap.ArenaAllocator = .init(std.heap.c_allocator);
    defer arena_state.deinit();

    // the preprocessor hands over the waveform as a single-row "mel", already [n_samples, 1]
    const inp_raw = g.buildInpRaw(1);
    var cur = llama.impl.llama_c.ggml_reshape_2d(ctx0, inp_raw, mtmd.graph.ne(inp_raw, 0), mtmd.graph.ne(inp_raw, 1));

    var seanet: mtmd.models.pockettts_seanet.Seanet = .init(g, arena_state.allocator());
    cur = seanet.encode(cur);
    g.cb(cur, "mimi_enc", -1);

    // [T, 512] -> transformer works on [512, T]
    cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));

    const inp_pos = llama.impl.llama_c.ggml_new_tensor_1d(ctx0, llama.impl.llama_c.GGML_TYPE_I32, mtmd.graph.ne(cur, 1));
    _ = llama.impl.llama_c.ggml_set_name(inp_pos, "inp_pos");
    llama.impl.llama_c.ggml_set_input(inp_pos);

    // the mimi transformer is causal with a sliding window, see _build_attention_mask()
    const kq_mask = g.buildInpAttnMask(mtmd.graph.ne(cur, 1), mtmd.graph.ne(cur, 1), 0);

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        cur = tfmLayerForward(g, cur, &model.layers.items[@intCast(il)], inp_pos, kq_mask, il);
    }
    g.cb(cur, "mimi_enc_tfm", -1);

    // downsample to the model frame rate, [512, T] -> [T, 512] -> [T / 16, 32]
    cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));
    cur = seanet.conv1d(cur, model.downsample_w, null, hparams.mimi_downsample, 1, true, "");
    g.cb(cur, "mimi_downsample", -1);

    // voice latent -> backbone embd
    cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));
    cur = g.buildMm(model.spk_proj_w, cur);
    g.cb(cur, "spk_proj", -1);

    llama.impl.llama_c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
