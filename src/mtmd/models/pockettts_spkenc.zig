//! The pocket-tts voice-prompt encoder: a raw 24 kHz waveform to one
//! conditioning row per 12.5 Hz frame, through the mimi encoder (SEANet,
//! transformer, downsample) and `flow_lm.speaker_proj_weight`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/pockettts-spkenc.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const seanet_mod = @import("pockettts_seanet.zig");
const c = impl.c;
const gpa = impl.gpa;
const T = graph.T;
const ne = graph.ne;

/// Ports `tfm_layer_forward` (pockettts-spkenc.cpp:7 @c1d0e7a00), a member
/// of `clip_graph_pockettts_spkenc`: a pre-norm block with layer scale on
/// both residual paths, see mimi_transformer.py.
fn tfmLayerForward(g: *const graph.Graph, cur_in: T, layer: *const model_mod.Layer, inp_pos: T, kq_mask: T, il: c_int) T {
    const ctx0 = g.ctx0;
    var inp = cur_in;

    var cur = g.buildNorm(cur_in, layer.ln_1_w, layer.ln_1_b, .normal, g.eps, il);

    var Qcur = g.buildMm(layer.q_w, cur);
    var Kcur = g.buildMm(layer.k_w, cur);
    var Vcur = g.buildMm(layer.v_w, cur);

    const n_pos = ne(cur, 1);
    Qcur = c.ggml_reshape_3d(ctx0, Qcur, g.d_head, g.n_head, n_pos);
    Kcur = c.ggml_reshape_3d(ctx0, Kcur, g.d_head, g.n_head, n_pos);
    Vcur = c.ggml_reshape_3d(ctx0, Vcur, g.d_head, g.n_head, n_pos);

    Qcur = c.ggml_rope_ext(ctx0, Qcur, inp_pos, null, g.d_head, c.GGML_ROPE_TYPE_NORMAL, 0, g.hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
    Kcur = c.ggml_rope_ext(ctx0, Kcur, inp_pos, null, g.d_head, c.GGML_ROPE_TYPE_NORMAL, 0, g.hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

    cur = g.buildAttn(layer.o_w, null, Qcur, Kcur, Vcur, kq_mask, g.kq_scale, il, null);
    cur = c.ggml_mul(ctx0, cur, layer.ls_1_w);
    cur = c.ggml_add(ctx0, cur, inp);

    inp = cur;
    cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .normal, g.eps, il);
    cur = g.buildFfn(cur, layer.ff_up_w, null, null, null, layer.ff_down_w, null, .gelu, il);
    cur = c.ggml_mul(ctx0, cur, layer.ls_2_w);
    cur = c.ggml_add(ctx0, cur, inp);

    return cur;
}

/// Ports `build` (pockettts-spkenc.cpp:39 @c1d0e7a00), a member of
/// `clip_graph_pockettts_spkenc`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();

    // the preprocessor hands over the waveform as a single-row "mel", already [n_samples, 1]
    const inp_raw = g.buildInpRaw(1);
    var cur = c.ggml_reshape_2d(ctx0, inp_raw, ne(inp_raw, 0), ne(inp_raw, 1));

    var seanet: seanet_mod.Seanet = .init(g, arena_state.allocator());
    cur = seanet.encode(cur);
    g.cb(cur, "mimi_enc", -1);

    // [T, 512] -> transformer works on [512, T]
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));

    const inp_pos = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, ne(cur, 1));
    _ = c.ggml_set_name(inp_pos, "inp_pos");
    c.ggml_set_input(inp_pos);

    // the mimi transformer is causal with a sliding window, see _build_attention_mask()
    const kq_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, ne(cur, 1), ne(cur, 1));
    _ = c.ggml_set_name(kq_mask, "kq_mask");
    c.ggml_set_input(kq_mask);

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        cur = tfmLayerForward(g, cur, &model.layers.items[@intCast(il)], inp_pos, kq_mask, il);
    }
    g.cb(cur, "mimi_enc_tfm", -1);

    // downsample to the model frame rate, [512, T] -> [T, 512] -> [T / 16, 32]
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = seanet.conv1d(cur, model.downsample_w, null, hparams.mimi_downsample, 1, true, "");
    g.cb(cur, "mimi_downsample", -1);

    // voice latent -> backbone embd
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = g.buildMm(model.spk_proj_w, cur);
    g.cb(cur, "spk_proj", -1);

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
