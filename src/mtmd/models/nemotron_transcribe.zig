//! The Nemotron Transcribe audio encoder, NeMo's ParallelExpertEncoder: an
//! ASR transformer and a speaker (Sortformer) transformer over the same
//! log-mel, the speaker activity fused into the ASR states. A recording is
//! encoded in windows with left and right context, one graph each, the
//! speaker cache and FIFO carried from one window to the next.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/nemotron-transcribe.cpp` at v0.9.0
//! (`cd959bae9`), the `clip_graph_nemotron_transcribe` declaration of
//! `llama.cpp/tools/mtmd/models/models.h`, and the
//! `PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE` arm of `clip_get_graph_builder`
//! (`llama.cpp/tools/mtmd/clip.cpp`). Each declaration names what it ports
//! and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const audio = @import("../audio.zig");
const libcxx = @import("llama").libcxx;
const c = impl.c;
const gpa = impl.gpa;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

extern fn sqrtf(x: f32) f32;
extern fn expf(x: f32) f32;

/// `SPK_ACTIVITY_THR` (nemotron-transcribe.cpp:11 @cd959bae9): a speaker is
/// active in an 80 ms frame where its prediction exceeds this.
const spk_activity_thr: f32 = 0.5;

// NeMo streaming Sortformer: PEE online defaults and the checkpoint's sortformer_modules config

/// `SPK_FIFO_LEN` (nemotron-transcribe.cpp:14 @cd959bae9).
const spk_fifo_len: c_int = 40;
/// `SPK_CACHE_LEN` (nemotron-transcribe.cpp:15 @cd959bae9).
const spk_cache_len: c_int = 188;
/// `SPK_SIL_FRAMES` (nemotron-transcribe.cpp:16 @cd959bae9).
const spk_sil_frames: c_int = 1;
/// `SPK_PRED_SCORE_THR` (nemotron-transcribe.cpp:17 @cd959bae9).
const spk_pred_score_thr: f32 = 0.25;
/// `SPK_BOOST_LATEST` (nemotron-transcribe.cpp:18 @cd959bae9).
const spk_boost_latest: f32 = 0.05;
/// `SPK_STRONG_BOOST_RATE` (nemotron-transcribe.cpp:19 @cd959bae9).
const spk_strong_boost_rate: f32 = 0.75;
/// `SPK_WEAK_BOOST_RATE` (nemotron-transcribe.cpp:20 @cd959bae9).
const spk_weak_boost_rate: f32 = 1.5;
/// `SPK_MIN_POS_RATE` (nemotron-transcribe.cpp:21 @cd959bae9).
const spk_min_pos_rate: f32 = 0.5;

// _compress_spkcache runs here on exp(score): its additive log terms become factors and -inf becomes 0

/// `SPK_SCORE_OFFSET` (nemotron-transcribe.cpp:24 @cd959bae9): -log(0.5) in _get_log_pred_scores.
const spk_score_offset: f32 = 2.0;
/// `SPK_STRONG_BOOST` (nemotron-transcribe.cpp:25 @cd959bae9): _boost_topk_scores, scale_factor 2.
const spk_strong_boost: f32 = 4.0;
/// `SPK_WEAK_BOOST` (nemotron-transcribe.cpp:26 @cd959bae9): _boost_topk_scores, scale_factor 1.
const spk_weak_boost: f32 = 2.0;
/// `SPK_SPEECH_THR` (nemotron-transcribe.cpp:27 @cd959bae9): a speaker's frames at or below it are not cached.
const spk_speech_thr: f32 = 0.5;

// NeMo ParallelExpertEncoder online defaults, in output frames: 40 s windows
// with 4 s of context on each side

/// `window_core` (mtmd/models/models.h:44 @cd959bae9), a member of `clip_graph_nemotron_transcribe`.
pub const window_core: c_int = 500;
/// `window_left` (mtmd/models/models.h:45 @cd959bae9), a member of `clip_graph_nemotron_transcribe`.
pub const window_left: c_int = 50;
/// `window_right` (mtmd/models/models.h:46 @cd959bae9), a member of `clip_graph_nemotron_transcribe`.
pub const window_right: c_int = 50;

// every update has a full window, which overflows the FIFO and the cache (NeMo's update period of 300 never applies)
comptime {
    std.debug.assert(window_core - spk_fifo_len > spk_cache_len);
}

/// `std::max(a, b)` for a `float`: `b` when `a < b`, else `a`.
inline fn maxF(a: f32, b: f32) f32 {
    return if (a < b) b else a;
}

/// Ports `build_spk_encoder` (nemotron-transcribe.cpp:30 @cd959bae9), a
/// member of `clip_graph_nemotron_transcribe`: the speaker expert has its own
/// layers, width and head count, which build_vit does not take; full rotary.
fn buildSpkEncoder(g: *graph.Graph, cur_in: T, pos: T) T {
    const ctx0 = g.ctx0;
    const model = g.model;
    const eps = g.eps;
    const n_head: i64 = g.hparams.spk_n_head;
    const n_embd = ne(cur_in, 0);
    const n_pos = ne(cur_in, 1);
    const d_head = @divTrunc(n_embd, n_head);

    var cur = g.buildNorm(cur_in, model.spk_pre_ln_w, model.spk_pre_ln_b, .normal, eps, -1);
    for (model.spk_layers.items, 0..) |*layer, ilu| {
        const il: c_int = @intCast(ilu);
        var x = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, .normal, eps, il);

        const qkv = g.buildMm(layer.qkv_w, x);
        const es = c.ggml_element_size(qkv);
        const de: usize = @intCast(d_head);
        const ee: usize = @intCast(n_embd);
        var q = c.ggml_view_3d(ctx0, qkv, d_head, n_head, n_pos, de * es, nb(qkv, 1), 0);
        var k = c.ggml_view_3d(ctx0, qkv, d_head, n_head, n_pos, de * es, nb(qkv, 1), ee * es);
        const v = c.ggml_view_3d(ctx0, qkv, d_head, n_head, n_pos, de * es, nb(qkv, 1), 2 * ee * es);
        q = c.ggml_rope_ext(ctx0, q, pos, null, @intCast(d_head), c.GGML_ROPE_TYPE_NEOX, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        k = c.ggml_rope_ext(ctx0, k, pos, null, @intCast(d_head), c.GGML_ROPE_TYPE_NEOX, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);

        x = g.buildAttn(layer.o_w, layer.o_b, q, k, v, null, 1.0 / sqrtf(@floatFromInt(d_head)), il, null);
        cur = c.ggml_add(ctx0, cur, x);

        x = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .normal, eps, il);
        x = g.buildFfn(x, layer.ff_up_w, layer.ff_up_b, null, null, layer.ff_down_w, layer.ff_down_b, .gelu_erf, il);
        cur = c.ggml_add(ctx0, cur, x);
    }

    return g.buildNorm(cur, model.spk_post_ln_w, model.spk_post_ln_b, .normal, eps, -1);
}

/// The `stack_mel` lambda (nemotron-transcribe.cpp:76 @cd959bae9) of `build`.
fn stackMel(g: *graph.Graph, mel: T, sub: i32, n_mel: i32) T {
    return g.buildStack(c.ggml_cont(g.ctx0, c.ggml_transpose(g.ctx0, mel)), sub, n_mel);
}

/// The `add_pos` lambda (nemotron-transcribe.cpp:93 @cd959bae9) of `build`:
/// partial rotary over the ASR expert's positions.
const AsrPos = struct {
    asr_pos: T,
    rope_dims: c_int,

    fn addPos(ctx: ?*anyopaque, g: *graph.Graph, cur: T, layer: *const model_mod.Layer) T {
        _ = layer;
        const self: *AsrPos = @ptrCast(@alignCast(ctx.?));
        return c.ggml_rope_ext(g.ctx0, cur, self.asr_pos, null, self.rope_dims, c.GGML_ROPE_TYPE_NEOX, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    }
};

/// Ports `build` (nemotron-transcribe.cpp:60 @cd959bae9), a member of
/// `clip_graph_nemotron_transcribe`, with the state `clip_get_graph_builder`
/// (mtmd/clip.cpp:980 @cd959bae9) passes it: null during warmup.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    const state: ?*clip.EncState = if (params) |p| p.enc_state else null;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;

    const sub: i64 = hparams.subsampling_factor;
    const n_mel: i64 = hparams.n_mel_bins;
    const n_mel_frames: i64 = g.img.nx();
    const n_pos = @divTrunc(n_mel_frames + sub - 1, sub);
    impl.ggmlAssert(g.img.ny() == n_mel, "nemotron-transcribe.cpp", 65, "img.ny() == n_mel");

    // warmup: the largest graph, with a full speaker cache and FIFO
    const n_prev: i64 = if (state) |s|
        @divTrunc(@as(i64, @intCast(s.spk_prev.items.len)), hparams.spk_n_embd)
    else
        spk_cache_len + spk_fifo_len;

    // [n_frames, n_mel] log-mel: raw for the speaker expert, normalized over the recording (encode_windows) for the ASR one
    const inp = g.buildInpRaw(1);
    const inp_norm = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_mel_frames, n_mel);
    _ = c.ggml_set_name(inp_norm, "inp_norm");
    c.ggml_set_input(inp_norm);

    // positions over [cache | FIFO | window]; the ASR expert takes the first n_pos
    const pos = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_prev + n_pos);
    _ = c.ggml_set_name(pos, "positions");
    c.ggml_set_input(pos);

    var spk_prev: T = null;
    if (n_prev > 0) {
        spk_prev = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, hparams.spk_n_embd, n_prev);
        _ = c.ggml_set_name(spk_prev, "spk_prev");
        c.ggml_set_input(spk_prev);
    }

    const asr_pos = c.ggml_view_1d(ctx0, pos, n_pos, 0);
    var add_pos: AsrPos = .{ .asr_pos = asr_pos, .rope_dims = hparams.audio_rope_dims };
    const asr = g.buildVit(g.buildMm(model.inp_proj_w, stackMel(g, inp_norm, @intCast(sub), @intCast(n_mel))), n_pos, .normal, .gelu_erf, null, .{ .ctx = &add_pos, .f = AsrPos.addPos }, .{});
    g.cb(asr, "asr_out", -1);

    // speaker expert over [cache | FIFO | window]; the window's rows and predictions feed the next window's state
    const chunk = g.buildMm(model.spk_inp_proj_w, stackMel(g, inp, @intCast(sub), @intCast(n_mel)));
    _ = c.ggml_set_name(chunk, "spk_chunk");
    c.ggml_set_output(chunk);
    var spk = if (spk_prev != null) c.ggml_concat(ctx0, spk_prev, chunk, 1) else chunk;
    const n_spk_pos = n_prev + n_pos;
    spk = buildSpkEncoder(g, spk, pos);
    g.cb(spk, "spk_out", -1);

    // Sortformer head, 10 ms frames
    var cur = c.ggml_add(ctx0, g.buildMm(model.spk_head_proj_w, spk), model.spk_head_proj_b);
    const d_tf = ne(cur, 0);
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = c.ggml_conv_1d(ctx0, model.spk_head_up_w, cur, 1, 1, 1); // [n_spk_pos, sub*d_tf]
    cur = c.ggml_add(ctx0, cur, c.ggml_reshape_2d(ctx0, model.spk_head_up_b, 1, sub * d_tf));
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = c.ggml_reshape_2d(ctx0, cur, d_tf, sub * n_spk_pos);
    cur = c.ggml_add(ctx0, g.buildMm(model.spk_head_fc1_w, c.ggml_relu(ctx0, cur)), model.spk_head_fc1_b);
    cur = c.ggml_add(ctx0, g.buildMm(model.spk_head_fc2_w, c.ggml_relu(ctx0, cur)), model.spk_head_fc2_b);
    const probs = c.ggml_sigmoid(ctx0, cur); // [n_spk, sub*n_spk_pos]
    g.cb(probs, "spk_probs", -1);

    // 80 ms predictions
    var preds = c.ggml_pool_2d(ctx0, probs, c.GGML_OP_POOL_AVG, 1, @intCast(sub), 1, @intCast(sub), 0, 0); // [n_spk, n_spk_pos]
    _ = c.ggml_set_name(preds, "spk_preds");
    c.ggml_set_output(preds);

    // fusion, on the window's rows
    preds = c.ggml_view_2d(ctx0, preds, ne(preds, 0), n_pos, nb(preds, 1), @as(usize, @intCast(n_prev)) * nb(preds, 1));
    var act = c.ggml_step(ctx0, c.ggml_scale_bias(ctx0, preds, 1.0, -spk_activity_thr));
    act = g.buildNorm(act, model.fusion_spk_norm_w, model.fusion_spk_norm_b, .normal, eps, -1);
    cur = g.buildNorm(asr, model.fusion_asr_norm_w, model.fusion_asr_norm_b, .normal, eps, -1);
    cur = c.ggml_add(ctx0, cur, g.buildMm(model.fusion_spk_kernel, act));
    g.cb(cur, "fused", -1);

    cur = c.ggml_add(ctx0, g.buildMm(model.mm_fc_w, cur), model.mm_fc_b);
    g.cb(cur, "projected", -1);

    c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

/// The comparator of `spk_topk`: larger first, ties to the lower index.
fn topkLess(v: []const f32, a: i32, b: i32) bool {
    const va = v[@intCast(a)];
    const vb = v[@intCast(b)];
    return va > vb or (va == vb and a < b);
}

/// Ports `spk_topk` (nemotron-transcribe.cpp:145 @cd959bae9): the indices
/// of the k largest of v[0, n), ties to the lower index. The caller frees
/// the slice.
fn spkTopk(v: []const f32, n: c_int, k_in: c_int) clip.Error![]i32 {
    const idx = try gpa.alloc(i32, @intCast(n));
    for (idx, 0..) |*x, i| x.* = @intCast(i);
    const k = std.math.clamp(k_in, 0, n);
    defer gpa.free(idx);
    libcxx.partialSort(i32, idx, @intCast(k), v[0..@intCast(n)], topkLess);
    return gpa.dupe(i32, idx[0..@intCast(k)]);
}

fn lessI32(_: void, a: i32, b: i32) bool {
    return a < b;
}

/// Ports `spk_compress_cache` (nemotron-transcribe.cpp:157 @cd959bae9):
/// NeMo SortformerModules._compress_spkcache, keeping the SPK_CACHE_LEN
/// highest-scoring rows, speaker by speaker.
fn spkCompressCache(emb: *std.ArrayList(f32), preds: *std.ArrayList(f32), n: c_int, d: c_int, n_spk: c_int, sil_emb: []const f32) clip.Error!void {
    const per_spk = @max(0, @divTrunc(spk_cache_len, n_spk) - spk_sil_frames);
    const strong: c_int = @intFromFloat(@floor(@as(f32, @floatFromInt(per_spk)) * spk_strong_boost_rate));
    const weak: c_int = @intFromFloat(@floor(@as(f32, @floatFromInt(per_spk)) * spk_weak_boost_rate));
    const min_pos: c_int = @intFromFloat(@floor(@as(f32, @floatFromInt(per_spk)) * spk_min_pos_rate));
    const disabled: f32 = 0.0;
    const latest = expf(spk_boost_latest);

    // speaker-major: each speaker's n rows, then its silence rows (+inf)
    const nt: usize = @intCast(n + spk_sil_frames);
    const nu: usize = @intCast(n);
    const ns: usize = @intCast(n_spk);
    const scores = try gpa.alloc(f32, ns * nt);
    defer gpa.free(scores);
    @memset(scores, std.math.inf(f32));
    for (0..ns) |s| {
        const sc = scores[s * nt ..][0..nt];
        var n_pos: c_int = 0;
        for (0..nu) |t| {
            const pt = preds.items[t * ns ..][0..ns];
            var prod: f32 = 1.0;
            for (0..ns) |j| {
                if (j != s) {
                    prod *= maxF(1.0 - pt[j], spk_pred_score_thr);
                }
            }
            prod *= maxF(pt[s], spk_pred_score_thr);
            sc[t] = if (pt[s] > spk_speech_thr) prod * spk_score_offset else disabled;
            n_pos += @intFromBool(sc[t] > 1.0);
        }
        if (n_pos >= min_pos) {
            for (0..nu) |t| {
                if (!(sc[t] > 1.0)) {
                    sc[t] = disabled;
                }
            }
        }
        var t: usize = @intCast(spk_cache_len);
        while (t < nu) : (t += 1) sc[t] *= latest;
        const top_strong = try spkTopk(sc, n, strong);
        defer gpa.free(top_strong);
        for (top_strong) |i| sc[@intCast(i)] *= spk_strong_boost;
        const top_weak = try spkTopk(sc, n, weak);
        defer gpa.free(top_weak);
        for (top_weak) |i| sc[@intCast(i)] *= spk_weak_boost;
    }

    const pick = try spkTopk(scores, @intCast(scores.len), spk_cache_len);
    defer gpa.free(pick);
    const placeholder: i32 = @intCast(scores.len);
    for (pick) |*i| {
        if (scores[@intCast(i.*)] == disabled) {
            i.* = placeholder;
        }
    }
    libcxx.sort(i32, pick, {}, lessI32);

    const du: usize = @intCast(d);
    const cl: usize = @intCast(spk_cache_len);
    var out_emb: std.ArrayList(f32) = .empty;
    errdefer out_emb.deinit(gpa);
    try out_emb.appendNTimes(gpa, 0, cl * du);
    var out_preds: std.ArrayList(f32) = .empty;
    errdefer out_preds.deinit(gpa);
    try out_preds.appendNTimes(gpa, 0.0, cl * ns);
    for (0..cl) |i| {
        const t: i64 = if (pick[i] == placeholder) -1 else @rem(pick[i], @as(i32, @intCast(nt)));
        const e = out_emb.items[i * du ..][0..du];
        if (t < 0 or t >= n) {
            @memcpy(e[0..sil_emb.len], sil_emb);
        } else {
            const tu: usize = @intCast(t);
            @memcpy(e, emb.items[tu * du ..][0..du]);
            @memcpy(out_preds.items[i * ns ..][0..ns], preds.items[tu * ns ..][0..ns]);
        }
    }
    std.mem.swap(std.ArrayList(f32), emb, &out_emb);
    std.mem.swap(std.ArrayList(f32), preds, &out_preds);
    out_emb.deinit(gpa);
    out_preds.deinit(gpa);
}

/// Ports `spk_stream` (nemotron-transcribe.cpp:228 @cd959bae9): the
/// speaker cache, with its predictions, and FIFO of one recording.
const SpkStream = struct {
    cache: std.ArrayList(f32) = .empty,
    cache_preds: std.ArrayList(f32) = .empty,
    fifo: std.ArrayList(f32) = .empty,

    fn deinit(self: *SpkStream) void {
        self.fifo.deinit(gpa);
        self.cache_preds.deinit(gpa);
        self.cache.deinit(gpa);
    }
};

/// Ports `update_spk_state` (nemotron-transcribe.cpp:237 @cd959bae9): NeMo
/// SortformerModules.streaming_update, core rows join the FIFO, its overflow
/// moves to the cache, which is compressed.
fn updateSpkState(ss: *SpkStream, st: *const clip.EncState, model: *const model_mod.Model, core_begin: i64, n_core: i64) clip.Error!void {
    const d: i64 = model.hparams.spk_n_embd;
    const n_spk: i64 = ne(model.spk_head_fc2_w, 1);
    const n_cache: i64 = @divTrunc(@as(i64, @intCast(ss.cache.items.len)), d);
    const n_fifo: i64 = @divTrunc(@as(i64, @intCast(ss.fifo.items.len)), d);
    const n_rows: i64 = @divTrunc(@as(i64, @intCast(st.spk_chunk.items.len)), d);
    impl.ggmlAssert((n_cache + n_fifo + n_rows) * n_spk == @as(i64, @intCast(st.spk_preds.items.len)) and core_begin + n_core <= n_rows, "nemotron-transcribe.cpp", 244, "(n_cache + n_fifo + n_rows) * n_spk == (int64_t) st.spk_preds.size() && core_begin + n_core <= n_rows");
    const preds = st.spk_preds.items;
    const core_preds = preds[@intCast((n_cache + n_fifo + core_begin) * n_spk)..];

    var fifo_preds: std.ArrayList(f32) = .empty;
    defer fifo_preds.deinit(gpa);
    try fifo_preds.appendSlice(gpa, preds[@intCast(n_cache * n_spk)..@intCast((n_cache + n_fifo) * n_spk)]);
    try ss.fifo.appendSlice(gpa, st.spk_chunk.items[@intCast(core_begin * d)..@intCast((core_begin + n_core) * d)]);
    try fifo_preds.appendSlice(gpa, core_preds[0..@intCast(n_core * n_spk)]);

    const n_pop = n_fifo + n_core - spk_fifo_len;
    try ss.cache.appendSlice(gpa, ss.fifo.items[0..@intCast(n_pop * d)]);
    try ss.cache_preds.appendSlice(gpa, fifo_preds.items[0..@intCast(n_pop * n_spk)]);
    ss.fifo.replaceRangeAssumeCapacity(0, @intCast(n_pop * d), &.{});
    try spkCompressCache(&ss.cache, &ss.cache_preds, @intCast(@divTrunc(@as(i64, @intCast(ss.cache.items.len)), d)), @intCast(d), @intCast(n_spk), model.spk_sil_emb.items);
}

/// Ports `encode_windows` (nemotron-transcribe.cpp:260 @cd959bae9), a static
/// member of `clip_graph_nemotron_transcribe`: NeMo
/// ParallelExpertEncoder._forward_online, one recording encoded in windows
/// with left/right context, one graph each, the speaker state carried.
pub fn encodeWindows(ctx: *clip.Ctx, model: *const model_mod.Model, params: *clip.EncodeParams) clip.Error!bool {
    impl.ggmlAssert(params.imgs.?.entries.items.len == 1 and params.out_embd != null, "nemotron-transcribe.cpp", 261, "params->imgs->entries.size() == 1 && params->out_embd != nullptr");
    const rec = &params.imgs.?.entries.items[0];
    const buf = rec.roBuf();
    const n_frames: i64 = rec.nx();
    const n_rows: i64 = rec.ny();
    const n_embd: i64 = clip.nMmprojEmbd(ctx);
    const sub: i64 = model.hparams.subsampling_factor;
    const core: i64 = @as(i64, window_core) * sub;
    const left: i64 = @as(i64, window_left) * sub;
    const right: i64 = @as(i64, window_right) * sub;
    const n_win: i64 = @max(1, @divTrunc(n_frames + core - 1, core));
    impl.ggmlAssert(@as(i64, @intCast(buf.len)) == n_frames * n_rows, "nemotron-transcribe.cpp", 272, "(int64_t) buf.size() == n_frames * n_rows");

    const out = params.out_embd.?;
    impl.ggmlAssert(out.items.len == @as(usize, @intCast(n_embd)) *% @as(usize, @bitCast(@as(isize, clip.nOutputTokens(ctx, rec)))), "nemotron-transcribe.cpp", 275, "out.size() == (size_t) n_embd * clip_n_output_tokens(ctx, &rec)");

    // the ASR expert's input: per-feature normalization over the whole recording (NeMo normalize_batch, per_feature)
    const nr: usize = @intCast(n_rows);
    const nfu: usize = @intCast(n_frames);
    const mean = try gpa.alloc(f64, nr);
    defer gpa.free(mean);
    const denom = try gpa.alloc(f64, nr);
    defer gpa.free(denom);
    for (0..nr) |r| {
        const row = buf[r * nfu ..][0..nfu];
        var sum: f64 = 0.0;
        for (row) |v| sum += @as(f64, v);
        mean[r] = sum / @as(f64, @floatFromInt(n_frames));
        const sq_diff_sum = audio.strictSquareSum(row, mean[r]);
        denom[r] = @sqrt(sq_diff_sum / (@as(f64, @floatFromInt(n_frames)) - 1.0)) + 1e-5;
    }

    var ss: SpkStream = .{};
    defer ss.deinit();
    var st: clip.EncState = .{};
    defer {
        st.spk_preds.deinit(gpa);
        st.spk_chunk.deinit(gpa);
        st.spk_prev.deinit(gpa);
        st.mel_norm.deinit(gpa);
    }
    var n_out: usize = 0;
    var w: i64 = 0;
    while (w < n_win) : (w += 1) {
        const core_begin = w * core;
        const core_end = @min(core_begin + core, n_frames);
        const win_begin = @max(core_begin - left, 0);
        const win_end = @min(core_end + right, n_frames);
        const win_len = win_end - win_begin;
        const wl: usize = @intCast(win_len);
        const wb: usize = @intCast(win_begin);

        const data = try gpa.alloc(f32, wl * nr);
        defer gpa.free(data);
        try st.mel_norm.resize(gpa, data.len);
        for (0..nr) |r| {
            const src = buf[r * nfu + wb ..][0..wl];
            @memcpy(data[r * wl ..][0..wl], src);
            for (0..wl) |t| {
                st.mel_norm.items[r * wl + t] = @floatCast((@as(f64, src[t]) - mean[r]) / denom[r]);
            }
        }
        var batch: impl.ImageF32Batch = .{};
        defer batch.deinit();
        batch.is_audio = true;
        try batch.entries.append(gpa, .{});
        try batch.entries.items[0].setSize(.{ .width = @intCast(win_len), .height = @intCast(n_rows) }, false, true);
        try batch.entries.items[0].cpyBuf(data);

        var embd: std.ArrayList(f32) = .empty;
        defer embd.deinit(gpa);
        try embd.appendNTimes(gpa, 0.0, @as(usize, @intCast(n_embd)) * @as(usize, @intCast(clip.nOutputTokens(ctx, &batch.entries.items[0]))));
        var wp = params.*;
        wp.imgs = &batch;
        wp.out_embd = &embd;
        wp.enc_state = &st;
        if (!try clip.encode(ctx, &wp)) {
            return false;
        }

        // output frames of the core, rounded up as NeMo's _asr_output_frame_boundary
        const n_left: usize = @intCast(@divTrunc(core_begin - win_begin, sub));
        const n_core: usize = @intCast(@divTrunc(core_end - core_begin + sub - 1, sub));
        const ne_u: usize = @intCast(n_embd);
        impl.ggmlAssert((n_out + n_core) * ne_u <= out.items.len, "nemotron-transcribe.cpp", 331, "(n_out + n_core) * n_embd <= out.size()");
        @memcpy(out.items[n_out * ne_u ..][0 .. n_core * ne_u], embd.items[n_left * ne_u ..][0 .. n_core * ne_u]);
        n_out += n_core;

        if (w + 1 < n_win) {
            try updateSpkState(&ss, &st, model, @intCast(n_left), @intCast(n_core));
            st.spk_prev.clearRetainingCapacity();
            try st.spk_prev.appendSlice(gpa, ss.cache.items);
            try st.spk_prev.appendSlice(gpa, ss.fifo.items);
        }
    }
    impl.ggmlAssert(n_out * @as(usize, @intCast(n_embd)) == out.items.len, "nemotron-transcribe.cpp", 341, "n_out * n_embd == out.size()");
    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
