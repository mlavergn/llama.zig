//! Soprano TTS: the Vocos decoder, normalized Qwen3 hidden states to STFT
//! coefficients, and the host-side ISTFT that turns them into PCM.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/soprano.cpp` at v1.0.0 (`b1b770d90`),
//! and the `PROJECTOR_TYPE_SOPRANO` arm of `mtmd_get_graph_builder`
//! (`llama.cpp/tools/mtmd/mtmd-graph.cpp`). Each declaration names what it ports
//! and the line it began at.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// `upscale` (mtmd/models/models.h:544 @b1b770d90), a member of `mtmd_graph_soprano`:
/// the SopranoDecoder default.
const upscale: c_int = 4;

/// Ports `build` (soprano.cpp:5 @b1b770d90), a member of `mtmd_graph_soprano`,
/// with `n_frames` derived as mtmd_get_graph_builder's `PROJECTOR_TYPE_SOPRANO`
/// arm does (mtmd-graph.cpp:1030 @b1b770d90).
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*llama.impl.llama_c.ggml_cgraph {
    const ctx0 = g.ctx0;
    const hparams = g.hparams;
    const v = &g.model.vocos;
    const n_frames: c_int = if (params != null and params.?.feats != null)
        @intCast(@divTrunc(@as(i64, @intCast(params.?.feats.?.items.len)), hparams.projection_dim))
    else
        2;

    const n_inp = hparams.projection_dim;
    const n_up = upscale * (n_frames - 1) + 1;
    var cur = llama.impl.llama_c.ggml_new_tensor_2d(ctx0, llama.impl.llama_c.GGML_TYPE_F32, n_inp, n_frames);
    _ = llama.impl.llama_c.ggml_set_name(cur, "inp_feats");
    llama.impl.llama_c.ggml_set_input(cur);

    cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));
    cur = llama.impl.llama_c.ggml_interpolate(ctx0, cur, n_up, n_inp, 1, 1, llama.impl.llama_c.GGML_SCALE_MODE_BILINEAR | llama.impl.llama_c.GGML_SCALE_FLAG_ALIGN_CORNERS);
    if (llama.impl.llama_c.ggml_n_dims(v.input_w) == 3) {
        const k: c_int = @intCast(mtmd.graph.ne(v.input_w, 0));
        cur = llama.impl.llama_c.ggml_conv_1d(ctx0, v.input_w, cur, 1, @divTrunc(k, 2), 1);
        cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));
    } else {
        cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));
        cur = g.buildMm(v.input_w, cur);
    }
    cur = llama.impl.llama_c.ggml_add(ctx0, cur, v.input_b);
    cur = g.buildNorm(cur, v.norm_w, v.norm_b, .normal, g.eps, -1);

    for (v.blocks.items, 0..) |*b, il_u| {
        const il: c_int = @intCast(il_u);
        const k: c_int = @intCast(mtmd.graph.ne(b.dwconv_w, 0));
        const residual = cur;
        cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));
        cur = llama.impl.llama_c.ggml_conv_1d_dw(ctx0, b.dwconv_w, cur, 1, @divTrunc(k, 2), 1);
        cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));
        cur = llama.impl.llama_c.ggml_add(ctx0, cur, b.dwconv_b);
        cur = g.buildNorm(cur, b.norm_w, b.norm_b, .normal, g.eps, il);
        cur = llama.impl.llama_c.ggml_add(ctx0, g.buildMm(b.pw1_w, cur), b.pw1_b);
        cur = llama.impl.llama_c.ggml_gelu_erf(ctx0, cur);
        cur = llama.impl.llama_c.ggml_add(ctx0, g.buildMm(b.pw2_w, cur), b.pw2_b);
        cur = llama.impl.llama_c.ggml_add(ctx0, residual, llama.impl.llama_c.ggml_mul(ctx0, cur, b.gamma));
        g.cb(cur, "vocos_block", il);
    }
    cur = g.buildNorm(cur, v.output_norm_w, v.output_norm_b, .normal, g.eps, -1);
    cur = llama.impl.llama_c.ggml_add(ctx0, g.buildMm(v.output_w, cur), v.output_b);
    g.cb(cur, "vocos_head", -1);

    // the head holds n_fft/2+1 log-magnitudes followed by n_fft/2+1 phases
    const n_bins = @divTrunc(mtmd.graph.ne(v.output_w, 1), 2);
    var mag = llama.impl.llama_c.ggml_view_2d(ctx0, cur, n_bins, n_up, mtmd.graph.nb(cur, 1), 0);
    var phase = llama.impl.llama_c.ggml_view_2d(ctx0, cur, n_bins, n_up, mtmd.graph.nb(cur, 1), @as(usize, @intCast(n_bins)) * @sizeOf(f32));
    mag = llama.impl.llama_c.ggml_clamp(ctx0, llama.impl.llama_c.ggml_exp(ctx0, llama.impl.llama_c.ggml_cont(ctx0, mag)), 0.0, 100.0);
    phase = llama.impl.llama_c.ggml_cont(ctx0, phase);
    var re = llama.impl.llama_c.ggml_mul(ctx0, mag, llama.impl.llama_c.ggml_cos(ctx0, phase));
    var im = llama.impl.llama_c.ggml_mul(ctx0, mag, llama.impl.llama_c.ggml_sin(ctx0, phase));
    re = llama.impl.llama_c.ggml_reshape_3d(ctx0, re, 1, n_bins, n_up);
    im = llama.impl.llama_c.ggml_reshape_3d(ctx0, im, 1, n_bins, n_up);
    cur = llama.impl.llama_c.ggml_concat(ctx0, re, im, 0);
    _ = llama.impl.llama_c.ggml_set_name(cur, "out_audio");
    llama.impl.llama_c.ggml_set_output(cur);
    llama.impl.llama_c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

/// Ports `decode_spectrum` (soprano.cpp:62 @b1b770d90), a member of
/// `mtmd_graph_soprano`: the graph's STFT frames to PCM, on the host.
pub fn decodeSpectrum(model: *const mtmd.model.Model, spectrum: []const f32, pcm: *std.ArrayList(f32)) mtmd.clip.Error!void {
    const n_frame_vals: usize = @intCast(mtmd.graph.ne(model.vocos.output_w, 1));
    const n_fft: c_int = @intCast(n_frame_vals - 2);
    const hop_length = model.hparams.audio_hop_len;
    const n_frames = spectrum.len / n_frame_vals;
    var istft = try mtmd.audio.StreamingIstft.init(n_fft, hop_length);
    defer istft.deinit();
    pcm.clearRetainingCapacity();
    for (0..n_frames) |i| {
        var frame = try istft.processFrame(spectrum[i * n_frame_vals ..].ptr);
        defer frame.deinit(std.heap.c_allocator);
        try pcm.appendSlice(std.heap.c_allocator, frame.items);
    }
    var tail = try istft.flush();
    defer tail.deinit(std.heap.c_allocator);
    try pcm.appendSlice(std.heap.c_allocator, tail.items);
    // the shared ISTFT removes (n_fft - hop) / 2 samples, torch.istft(center=True) removes n_fft / 2
    const n_drop: usize = @intCast(@divTrunc(hop_length, 2));
    pcm.replaceRangeAssumeCapacity(0, n_drop, &.{});
    try pcm.resize(std.heap.c_allocator, (n_frames -% 1) *% @as(usize, @intCast(hop_length)));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
