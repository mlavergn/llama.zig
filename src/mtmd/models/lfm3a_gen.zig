//! LFM3-Audio generation.
//!
//! - GEN_CODE: one MaskGIT pass of the depthformer, hidden state + known
//!   codes (-1 = masked) to logits per codebook; once all codes are known,
//!   only the feedback embedding is built.
//! - GEN_WAV: a window of frames to polar STFT frames of the liquid-codec
//!   decoder, the ISTFT runs on the host.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/lfm3a-gen.cpp` at v1.0.0 (`b1b770d90`),
//! and the `PROJECTOR_TYPE_LFM3A_GEN` arm of `mtmd_get_graph_builder`
//! (`llama.cpp/tools/mtmd/mtmd-graph.cpp`). Each declaration names what it ports
//! and the line it began at.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `list_lfm3a_gen_state_slots` (lfm3a-gen.cpp:12 @b1b770d90): the
/// state carried between GEN_WAV calls: rope offset, short-conv taps and
/// attention window of the codec decoder. The caller frees the slice with
/// `module.freeSlots`.
pub fn listStateSlots(hparams: *const mtmd.model.Hparams, model: *const mtmd.model.Model) mtmd.clip.Error![]mtmd.models.dispatch.StateSlot {
    var slots: std.ArrayList(mtmd.models.dispatch.StateSlot) = .empty;
    errdefer {
        for (slots.items) |s| std.heap.c_allocator.free(s.name);
        slots.deinit(std.heap.c_allocator);
    }
    if (model.c2w.tfm_layers.items.len == 0) return slots.toOwnedSlice(std.heap.c_allocator);
    const prefix: i64 = hparams.wav_tfm_swa - 1;
    try slots.append(std.heap.c_allocator, .{ .name = try std.heap.c_allocator.dupeSentinel(u8, "pos", 0), .ne0 = 1, .ne1 = 1 });
    for (model.c2w.tfm_layers.items, 0..) |*layer, il| {
        if (layer.conv_dw_w != null) {
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "conv_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.conv_dw_w, 0) - 1, .ne1 = hparams.wav_tfm_n_embd });
        } else {
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "k_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.k_w, 1), .ne1 = prefix });
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "v_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.v_w, 1), .ne1 = prefix });
        }
    }
    return slots.toOwnedSlice(std.heap.c_allocator);
}

/// Ports `build_lfm_attn` (lfm3a-gen.cpp:50 @b1b770d90), a member of
/// `mtmd_graph_lfm3a_gen`: GQA with q/k RMSNorm, NEOX rope, an optional cached
/// K/V prefix and the LFM3 per-head sigmoid gate.
fn buildLfmAttn(g: *const mtmd.graph.Graph, cur: mtmd.graph.T, layer: *const mtmd.model.Layer, n_head: c_int, n_head_kv: c_int, eps: f32, rope_theta: f32, il: c_int, inp_pos: mtmd.graph.T, kq_mask: mtmd.graph.T, k_cache: mtmd.graph.T, v_cache: mtmd.graph.T, k_all: ?*mtmd.graph.T, v_all: ?*mtmd.graph.T) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const n_pos = mtmd.graph.ne(cur, 1);
    const head_dim = mtmd.graph.ne(layer.q_norm, 0);

    var q_cur = llama.impl.llama_c.ggml_reshape_3d(ctx0, g.buildMm(layer.q_w, cur), head_dim, n_head, n_pos);
    var k_cur = llama.impl.llama_c.ggml_reshape_3d(ctx0, g.buildMm(layer.k_w, cur), head_dim, n_head_kv, n_pos);
    const v_cur = g.buildMm(layer.v_w, cur);

    q_cur = g.buildNorm(q_cur, layer.q_norm, null, .rms, eps, il);
    k_cur = g.buildNorm(k_cur, layer.k_norm, null, .rms, eps, il);
    q_cur = llama.impl.llama_c.ggml_rope_ext(ctx0, q_cur, inp_pos, null, @intCast(head_dim), llama.impl.llama_c.GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
    k_cur = llama.impl.llama_c.ggml_rope_ext(ctx0, k_cur, inp_pos, null, @intCast(head_dim), llama.impl.llama_c.GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

    var k = llama.impl.llama_c.ggml_reshape_2d(ctx0, k_cur, head_dim * n_head_kv, n_pos);
    var v = v_cur;
    if (k_cache != null) {
        k = llama.impl.llama_c.ggml_concat(ctx0, k_cache, k, 1);
        v = llama.impl.llama_c.ggml_concat(ctx0, v_cache, v, 1);
    }
    if (k_all) |p| {
        p.* = k;
        v_all.?.* = v;
    }
    const n_kv = mtmd.graph.ne(k, 1);

    var attn = g.buildAttn(
        if (layer.attn_gate_w != null) null else layer.o_w,
        null,
        q_cur,
        llama.impl.llama_c.ggml_reshape_3d(ctx0, k, head_dim, n_head_kv, n_kv),
        llama.impl.llama_c.ggml_reshape_3d(ctx0, v, head_dim, n_head_kv, n_kv),
        kq_mask,
        1.0 / stdlib.libm.sqrtfInt(head_dim),
        il,
        null,
    );
    if (layer.attn_gate_w != null) {
        const gate = llama.impl.llama_c.ggml_sigmoid(ctx0, g.buildMm(layer.attn_gate_w, cur));
        attn = llama.impl.llama_c.ggml_mul(ctx0, llama.impl.llama_c.ggml_reshape_3d(ctx0, attn, head_dim, n_head, n_pos), llama.impl.llama_c.ggml_reshape_3d(ctx0, gate, 1, n_head, n_pos));
        attn = g.buildMm(layer.o_w, llama.impl.llama_c.ggml_reshape_2d(ctx0, attn, head_dim * n_head, n_pos));
    }
    return attn;
}

/// Ports `build_lfm_conv` (lfm3a-gen.cpp:92 @b1b770d90), a member of
/// `mtmd_graph_lfm3a_gen`: short conv, in_proj -> b * x -> causal depthwise
/// conv over [conv_state | x] -> c * conv -> out_proj.
fn buildLfmConv(g: *const mtmd.graph.Graph, cur_in: mtmd.graph.T, layer: *const mtmd.model.Layer, il: c_int, conv_state: mtmd.graph.T, conv_state_out: *mtmd.graph.T) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const n_embd_c = mtmd.graph.ne(cur_in, 0);
    const n_pos = mtmd.graph.ne(cur_in, 1);

    const bcx = g.buildMm(layer.conv_pw1_w, cur_in);
    const b = llama.impl.llama_c.ggml_view_2d(ctx0, bcx, n_embd_c, n_pos, mtmd.graph.nb(bcx, 1), 0);
    const cc = llama.impl.llama_c.ggml_view_2d(ctx0, bcx, n_embd_c, n_pos, mtmd.graph.nb(bcx, 1), llama.impl.llama_c.ggml_row_size(bcx.?.type, n_embd_c));
    const x = llama.impl.llama_c.ggml_view_2d(ctx0, bcx, n_embd_c, n_pos, mtmd.graph.nb(bcx, 1), llama.impl.llama_c.ggml_row_size(bcx.?.type, 2 * n_embd_c));

    var bx = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, llama.impl.llama_c.ggml_mul(ctx0, b, x)));
    bx = llama.impl.llama_c.ggml_concat(ctx0, conv_state, bx, 0);
    const d_conv = mtmd.graph.ne(conv_state, 0);
    conv_state_out.* = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, bx, d_conv, n_embd_c, mtmd.graph.nb(bx, 1), @as(usize, @intCast(n_pos)) * mtmd.graph.nb(bx, 0)));

    var cur = llama.impl.llama_c.ggml_ssm_conv(ctx0, llama.impl.llama_c.ggml_reshape_3d(ctx0, bx, mtmd.graph.ne(bx, 0), n_embd_c, 1), layer.conv_dw_w);
    cur = llama.impl.llama_c.ggml_mul(ctx0, cc, llama.impl.llama_c.ggml_reshape_2d(ctx0, cur, n_embd_c, n_pos));
    g.cb(cur, "conv", il);
    return g.buildMm(layer.conv_pw2_w, cur);
}

/// The first half of `build_lfm_layer` (lfm3a-gen.cpp:32 @b1b770d90), the
/// pre-norm before its `op`. A `std::function` here splits the method around
/// the call instead.
fn lfmLayerPre(g: *const mtmd.graph.Graph, inp_l: mtmd.graph.T, layer: *const mtmd.model.Layer, eps: f32, il: c_int) mtmd.graph.T {
    return g.buildNorm(inp_l, layer.ln_1_w, null, .rms, eps, il);
}

/// The second half of `build_lfm_layer` (lfm3a-gen.cpp:32 @b1b770d90): the LFM3
/// sandwich norms when the layer has them, and the FFN.
fn lfmLayerPost(g: *const mtmd.graph.Graph, inp_l_in: mtmd.graph.T, cur_in: mtmd.graph.T, layer: *const mtmd.model.Layer, eps: f32, il: c_int) mtmd.graph.T {
    const ctx0 = g.ctx0;
    var cur = cur_in;
    if (layer.attn_post_norm_w != null) cur = g.buildNorm(cur, layer.attn_post_norm_w, null, .rms, eps, il);
    const inp_l = llama.impl.llama_c.ggml_add(ctx0, cur, inp_l_in);

    cur = g.buildNorm(inp_l, layer.ln_2_w, null, .rms, eps, il);
    cur = g.buildFfn(cur, layer.ff_up_w, null, layer.ff_gate_w, null, layer.ff_down_w, null, .silu, il);
    if (layer.ff_post_norm_w != null) cur = g.buildNorm(cur, layer.ff_post_norm_w, null, .rms, eps, il);
    return llama.impl.llama_c.ggml_add(ctx0, inp_l, cur);
}

/// Ports `build_gen_code` (lfm3a-gen.cpp:113 @b1b770d90), a member of
/// `mtmd_graph_lfm3a_gen`.
fn buildGenCode(g: *mtmd.graph.Graph, all_known: bool) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const n_embd = g.n_embd;
    const n_mmproj_embd = g.n_mmproj_embd;

    const embd_w = model.gen_code_embd_w; // [n_embd, n_vocab, n_codebooks], tied to the output head
    const n_cb = mtmd.graph.ne(embd_w, 2);
    const n_vocab = mtmd.graph.ne(embd_w, 1);

    // rows of the merged per-codebook tables: code + codebook * n_vocab
    const inp_codes = llama.impl.llama_c.ggml_new_tensor_1d(ctx0, llama.impl.llama_c.GGML_TYPE_I32, n_cb);
    _ = llama.impl.llama_c.ggml_set_name(inp_codes, "inp_codes");
    llama.impl.llama_c.ggml_set_input(inp_codes);

    // feedback embedding for the backbone
    var fb = llama.impl.llama_c.ggml_get_rows(ctx0, model.gen_code_out_embd_w, inp_codes); // [n_mmproj_embd, n_cb]
    fb = llama.impl.llama_c.ggml_sum_rows(ctx0, llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, fb)));
    fb = llama.impl.llama_c.ggml_reshape_2d(ctx0, fb, n_mmproj_embd, 1);
    g.cb(fb, "gen_embd", -1);
    if (all_known) return fb;

    const h_state = llama.impl.llama_c.ggml_new_tensor_1d(ctx0, llama.impl.llama_c.GGML_TYPE_F32, n_mmproj_embd);
    _ = llama.impl.llama_c.ggml_set_name(h_state, "inp_raw");
    llama.impl.llama_c.ggml_set_input(h_state);

    // 1 for a revealed codebook, 0 for a masked one
    const inp_mask = llama.impl.llama_c.ggml_new_tensor_2d(ctx0, llama.impl.llama_c.GGML_TYPE_F32, 1, n_cb);
    _ = llama.impl.llama_c.ggml_set_name(inp_mask, "inp_mask");
    llama.impl.llama_c.ggml_set_input(inp_mask);

    // one position per codebook: projected hidden state + token embedding or the mask embedding
    var cur = llama.impl.llama_c.ggml_add(ctx0, g.buildMm(model.gen_code_proj_in_w, h_state), model.gen_code_proj_in_b);
    cur = llama.impl.llama_c.ggml_reshape_2d(ctx0, cur, n_embd, n_cb);

    var tok = llama.impl.llama_c.ggml_get_rows(ctx0, llama.impl.llama_c.ggml_reshape_2d(ctx0, embd_w, n_embd, n_vocab * n_cb), inp_codes);
    tok = llama.impl.llama_c.ggml_mul(ctx0, tok, inp_mask);
    const msk = llama.impl.llama_c.ggml_mul(ctx0, llama.impl.llama_c.ggml_repeat(ctx0, model.gen_code_mask_embd, cur), llama.impl.llama_c.ggml_scale_bias(ctx0, inp_mask, -1.0, 1.0));
    cur = llama.impl.llama_c.ggml_add(ctx0, cur, llama.impl.llama_c.ggml_add(ctx0, tok, msk));
    g.cb(cur, "depth_inp", -1);

    // bidirectional: no mask, no state
    const inp_pos = llama.impl.llama_c.ggml_cast(ctx0, llama.impl.llama_c.ggml_arange(ctx0, 0.0, @floatFromInt(n_cb), 1.0), llama.impl.llama_c.GGML_TYPE_I32);
    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        const x = lfmLayerPre(g, cur, layer, eps, il);
        const op = buildLfmAttn(g, x, layer, hparams.n_head, hparams.n_head_kv, eps, hparams.rope_theta, il, inp_pos, null, null, null, null, null);
        cur = lfmLayerPost(g, cur, op, layer, eps, il);
    }

    cur = g.buildNorm(cur, model.gen_code_norm_w, null, .rms, eps, -1);
    cur = g.buildNorm(cur, model.gen_code_head_norm_w, null, .rms, eps, -1); // per codebook
    var logits = llama.impl.llama_c.ggml_mul_mat(ctx0, embd_w, llama.impl.llama_c.ggml_reshape_3d(ctx0, cur, n_embd, 1, n_cb)); // [n_vocab, 1, n_cb]
    logits = llama.impl.llama_c.ggml_reshape_2d(ctx0, logits, n_vocab, n_cb);
    _ = llama.impl.llama_c.ggml_set_name(logits, "out_feats");
    llama.impl.llama_c.ggml_set_output(logits);
    llama.impl.llama_c.ggml_build_forward_expand(g.gf, logits);

    return fb;
}

/// Ports `build_gen_wav` (lfm3a-gen.cpp:172 @b1b770d90), a member of
/// `mtmd_graph_lfm3a_gen`.
fn buildGenWav(g: *mtmd.graph.Graph, n_frames: c_int) mtmd.clip.Error!mtmd.graph.T {
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const c2w = &model.c2w;
    const n_cb = mtmd.graph.ne(model.gen_code_embd_w, 2);
    const n_fs: i64 = hparams.wav_frame_stack;
    const n_pos = n_fs * n_frames;
    const prefix: i64 = hparams.wav_tfm_swa - 1;

    var arena_state: std.heap.ArenaAllocator = .init(std.heap.c_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // frame-major codes
    const inp_codes = llama.impl.llama_c.ggml_new_tensor_1d(ctx0, llama.impl.llama_c.GGML_TYPE_I32, n_cb * n_frames);
    _ = llama.impl.llama_c.ggml_set_name(inp_codes, "inp_codes");
    llama.impl.llama_c.ggml_set_input(inp_codes);

    var state_in: std.StringHashMapUnmanaged(mtmd.graph.T) = .empty;
    const StateOut = struct { name: [:0]const u8, t: mtmd.graph.T };
    var state_out: std.ArrayList(StateOut) = .empty;
    {
        const slots = try listStateSlots(hparams, model);
        defer mtmd.models.dispatch.freeSlots(slots);
        for (slots) |slot| {
            const t = llama.impl.llama_c.ggml_new_tensor_2d(ctx0, llama.impl.llama_c.GGML_TYPE_F32, slot.ne0, slot.ne1);
            const name = try mtmd.impl.stringFormat(arena, "state_in_%s", .{slot.name.ptr});
            _ = llama.impl.llama_c.ggml_set_name(t, name.ptr);
            llama.impl.llama_c.ggml_set_input(t);
            try state_in.put(arena, try arena.dupe(u8, slot.name), t);
        }
    }

    // FSQ decode: every codebook shares the level table, one projection over the n_cb code vectors of a frame
    var q = llama.impl.llama_c.ggml_get_rows(ctx0, c2w.quant_first_cb_w, inp_codes); // [n_lvl, n_cb * n_frames]
    q = llama.impl.llama_c.ggml_reshape_2d(ctx0, q, mtmd.graph.ne(q, 0) * n_cb, n_frames);
    var cur = llama.impl.llama_c.ggml_add(ctx0, g.buildMm(c2w.quant_first_out_w, q), c2w.quant_first_out_b);
    cur = llama.impl.llama_c.ggml_add(ctx0, g.buildMm(c2w.tfm_in_proj_w, cur), c2w.tfm_in_proj_b); // [n_embd_dec, n_frames]
    const n_embd_dec = mtmd.graph.ne(cur, 0);

    // nearest-exact upsampling: every frame becomes n_fs decoder positions
    cur = llama.impl.llama_c.ggml_repeat_4d(ctx0, llama.impl.llama_c.ggml_reshape_3d(ctx0, cur, n_embd_dec, 1, n_frames), n_embd_dec, n_fs, n_frames, 1);
    cur = llama.impl.llama_c.ggml_reshape_2d(ctx0, cur, n_embd_dec, n_pos);

    // positions continue across calls from the "pos" slot, see mtmd_subcontext::encode()
    const inp_pos = llama.impl.llama_c.ggml_new_tensor_1d(ctx0, llama.impl.llama_c.GGML_TYPE_I32, n_pos);
    _ = llama.impl.llama_c.ggml_set_name(inp_pos, "positions");
    llama.impl.llama_c.ggml_set_input(inp_pos);
    try state_out.append(arena, .{ .name = "pos", .t = llama.impl.llama_c.ggml_scale_bias(ctx0, state_in.get("pos").?, 1.0, @floatFromInt(n_pos)) });

    // sliding-window causal mask over [cached window | this chunk]
    const kq_mask = g.buildInpAttnMask(prefix + n_pos, n_pos, 0);

    const eps_dec = hparams.wav_tfm_eps;
    for (c2w.tfm_layers.items, 0..) |*layer, il_u| {
        const il: c_int = @intCast(il_u);
        const x = lfmLayerPre(g, cur, layer, eps_dec, il);
        var op: mtmd.graph.T = undefined;
        if (layer.conv_dw_w != null) {
            const name = try mtmd.impl.stringFormat(arena, "conv_%zu", .{il_u});
            var st: mtmd.graph.T = null;
            op = buildLfmConv(g, x, layer, il, state_in.get(name).?, &st);
            try state_out.append(arena, .{ .name = name, .t = st });
        } else {
            // this chunk's tail of the window is kept for the next call
            const k_name = try mtmd.impl.stringFormat(arena, "k_%zu", .{il_u});
            const v_name = try mtmd.impl.stringFormat(arena, "v_%zu", .{il_u});
            var k: mtmd.graph.T = null;
            var v: mtmd.graph.T = null;
            op = buildLfmAttn(g, x, layer, hparams.wav_tfm_n_head, hparams.wav_tfm_n_head_kv, eps_dec, hparams.wav_tfm_rope_theta, il, inp_pos, kq_mask, state_in.get(k_name).?, state_in.get(v_name).?, &k, &v);
            try state_out.append(arena, .{ .name = k_name, .t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, k, mtmd.graph.ne(k, 0), prefix, mtmd.graph.nb(k, 1), @as(usize, @intCast(n_pos)) * mtmd.graph.nb(k, 1))) });
            try state_out.append(arena, .{ .name = v_name, .t = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, v, mtmd.graph.ne(v, 0), prefix, mtmd.graph.nb(v, 1), @as(usize, @intCast(n_pos)) * mtmd.graph.nb(v, 1))) });
        }
        cur = lfmLayerPost(g, cur, op, layer, eps_dec, il);
    }

    cur = g.buildNorm(cur, c2w.tfm_output_norm_w, null, .rms, eps_dec, -1);
    cur = llama.impl.llama_c.ggml_add(ctx0, g.buildMm(c2w.tfm_out_proj_w, cur), c2w.tfm_out_proj_b); // [2 * n_bins, n_pos]
    g.cb(cur, "polar_out", -1);

    for (state_out.items) |s| {
        const name = try mtmd.impl.stringFormat(arena, "state_out_%s", .{s.name.ptr});
        _ = llama.impl.llama_c.ggml_set_name(s.t, name.ptr);
        llama.impl.llama_c.ggml_set_output(s.t);
        llama.impl.llama_c.ggml_build_forward_expand(g.gf, s.t);
    }

    _ = llama.impl.llama_c.ggml_set_name(cur, "out_feats");
    llama.impl.llama_c.ggml_set_output(cur);
    return cur;
}

/// Ports `build` (lfm3a-gen.cpp:253 @b1b770d90), a member of
/// `mtmd_graph_lfm3a_gen`, with the builder's parameters derived as
/// mtmd_get_graph_builder's `PROJECTOR_TYPE_LFM3A_GEN` arm does
/// (mtmd-graph.cpp:996 @b1b770d90).
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*llama.impl.llama_c.ggml_cgraph {
    const gen_process: mtmd.impl.GenProcessType = if (params) |p| p.gen_process else .gen_code;
    const n_cb = mtmd.graph.ne(g.model.gen_code_embd_w, 2);
    var all_known = false;
    var n_frames: c_int = 1;
    if (params != null and params.?.codes != null) {
        const codes = params.?.codes.?.items;
        mtmd.impl.ggmlAssert(@rem(@as(i64, @intCast(codes.len)), n_cb) == 0, "mtmd-graph.cpp", 1003, "params->codes->size() % (size_t) n_cb == 0");
        if (gen_process == .gen_wav) {
            n_frames = @intCast(@divTrunc(@as(i64, @intCast(codes.len)), n_cb));
            mtmd.impl.ggmlAssert(n_frames > 0, "mtmd-graph.cpp", 1006, "n_frames > 0");
        } else {
            // -1 marks a masked codebook, none masked means only the feedback embedding is needed
            all_known = true;
            for (codes) |cc| {
                if (cc < 0) {
                    all_known = false;
                    break;
                }
            }
        }
    }
    const out = if (gen_process == .gen_wav) try buildGenWav(g, n_frames) else buildGenCode(g, all_known);
    llama.impl.llama_c.ggml_build_forward_expand(g.gf, out);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
