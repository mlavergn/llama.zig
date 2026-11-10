//! Soprano TTS: the Vocos decoder, normalized Qwen3 hidden states to STFT
//! coefficients, and the host-side ISTFT that turns them into PCM.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/soprano.cpp` at v0.9.0 (`cd959bae9`),
//! and the `PROJECTOR_TYPE_SOPRANO` arm of `clip_get_graph_builder`
//! (`llama.cpp/tools/mtmd/clip.cpp`). Each declaration names what it ports
//! and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const audio = @import("../audio.zig");
const c = impl.c;
const gpa = impl.gpa;
const ne = graph.ne;
const nb = graph.nb;

/// `upscale` (mtmd/models/models.h:501 @cd959bae9), a member of `clip_graph_soprano`:
/// the SopranoDecoder default.
const upscale: c_int = 4;

/// Ports `build` (soprano.cpp:5 @cd959bae9), a member of `clip_graph_soprano`,
/// with `n_frames` derived as clip_get_graph_builder's `PROJECTOR_TYPE_SOPRANO`
/// arm does (mtmd/clip.cpp:1247 @cd959bae9).
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    const ctx0 = g.ctx0;
    const hparams = g.hparams;
    const v = &g.model.vocos;
    const n_frames: c_int = if (params != null and params.?.feats != null)
        @intCast(@divTrunc(@as(i64, @intCast(params.?.feats.?.items.len)), hparams.projection_dim))
    else
        2;

    const n_inp = hparams.projection_dim;
    const n_up = upscale * (n_frames - 1) + 1;
    var cur = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_inp, n_frames);
    _ = c.ggml_set_name(cur, "inp_feats");
    c.ggml_set_input(cur);

    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = c.ggml_interpolate(ctx0, cur, n_up, n_inp, 1, 1, c.GGML_SCALE_MODE_BILINEAR | c.GGML_SCALE_FLAG_ALIGN_CORNERS);
    if (c.ggml_n_dims(v.input_w) == 3) {
        const k: c_int = @intCast(ne(v.input_w, 0));
        cur = c.ggml_conv_1d(ctx0, v.input_w, cur, 1, @divTrunc(k, 2), 1);
        cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    } else {
        cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
        cur = g.buildMm(v.input_w, cur);
    }
    cur = c.ggml_add(ctx0, cur, v.input_b);
    cur = g.buildNorm(cur, v.norm_w, v.norm_b, .normal, g.eps, -1);

    for (v.blocks.items, 0..) |*b, il_u| {
        const il: c_int = @intCast(il_u);
        const k: c_int = @intCast(ne(b.dwconv_w, 0));
        const residual = cur;
        cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
        cur = c.ggml_conv_1d_dw(ctx0, b.dwconv_w, cur, 1, @divTrunc(k, 2), 1);
        cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
        cur = c.ggml_add(ctx0, cur, b.dwconv_b);
        cur = g.buildNorm(cur, b.norm_w, b.norm_b, .normal, g.eps, il);
        cur = c.ggml_add(ctx0, g.buildMm(b.pw1_w, cur), b.pw1_b);
        cur = c.ggml_gelu_erf(ctx0, cur);
        cur = c.ggml_add(ctx0, g.buildMm(b.pw2_w, cur), b.pw2_b);
        cur = c.ggml_add(ctx0, residual, c.ggml_mul(ctx0, cur, b.gamma));
        g.cb(cur, "vocos_block", il);
    }
    cur = g.buildNorm(cur, v.output_norm_w, v.output_norm_b, .normal, g.eps, -1);
    cur = c.ggml_add(ctx0, g.buildMm(v.output_w, cur), v.output_b);
    g.cb(cur, "vocos_head", -1);

    // the head holds n_fft/2+1 log-magnitudes followed by n_fft/2+1 phases
    const n_bins = @divTrunc(ne(v.output_w, 1), 2);
    var mag = c.ggml_view_2d(ctx0, cur, n_bins, n_up, nb(cur, 1), 0);
    var phase = c.ggml_view_2d(ctx0, cur, n_bins, n_up, nb(cur, 1), @as(usize, @intCast(n_bins)) * @sizeOf(f32));
    mag = c.ggml_clamp(ctx0, c.ggml_exp(ctx0, c.ggml_cont(ctx0, mag)), 0.0, 100.0);
    phase = c.ggml_cont(ctx0, phase);
    var re = c.ggml_mul(ctx0, mag, c.ggml_cos(ctx0, phase));
    var im = c.ggml_mul(ctx0, mag, c.ggml_sin(ctx0, phase));
    re = c.ggml_reshape_3d(ctx0, re, 1, n_bins, n_up);
    im = c.ggml_reshape_3d(ctx0, im, 1, n_bins, n_up);
    cur = c.ggml_concat(ctx0, re, im, 0);
    _ = c.ggml_set_name(cur, "out_audio");
    c.ggml_set_output(cur);
    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

/// Ports `decode_spectrum` (soprano.cpp:62 @cd959bae9), a member of
/// `clip_graph_soprano`: the graph's STFT frames to PCM, on the host.
pub fn decodeSpectrum(model: *const model_mod.Model, spectrum: []const f32, pcm: *std.ArrayList(f32)) clip.Error!void {
    const n_frame_vals: usize = @intCast(ne(model.vocos.output_w, 1));
    const n_fft: c_int = @intCast(n_frame_vals - 2);
    const hop_length = model.hparams.audio_hop_len;
    const n_frames = spectrum.len / n_frame_vals;
    var istft = try audio.StreamingIstft.init(n_fft, hop_length);
    defer istft.deinit();
    pcm.clearRetainingCapacity();
    for (0..n_frames) |i| {
        var frame = try istft.processFrame(spectrum[i * n_frame_vals ..].ptr);
        defer frame.deinit(gpa);
        try pcm.appendSlice(gpa, frame.items);
    }
    var tail = try istft.flush();
    defer tail.deinit(gpa);
    try pcm.appendSlice(gpa, tail.items);
    // the shared ISTFT removes (n_fft - hop) / 2 samples, torch.istft(center=True) removes n_fft / 2
    const n_drop: usize = @intCast(@divTrunc(hop_length, 2));
    pcm.replaceRangeAssumeCapacity(0, n_drop, &.{});
    try pcm.resize(gpa, (n_frames -% 1) *% @as(usize, @intCast(hop_length)));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
