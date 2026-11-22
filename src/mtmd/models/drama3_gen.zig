//! The drama-3 TTS generator: the codebook tower (one row: the backbone
//! hidden state and the next main code to the other codes of that frame and
//! the next backbone input, each layer attending over the whole sequence
//! through K/V kept in the gen state) and the Fish DAC codec (frames to PCM,
//! Qwen3-TTS's code2wav blocks without pre_conv and the transformer
//! projections).
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/drama3-gen.cpp` at v1.0.0
//! (`b1b770d90`), new in 5851a8e50, the `mtmd_graph_drama3_gen` declaration
//! in `llama.cpp/tools/mtmd/models/models.h`, and the
//! `PROJECTOR_TYPE_DRAMA3_GEN` arm of `mtmd_get_graph_builder`
//! (`llama.cpp/tools/mtmd/mtmd-graph.cpp`). Each declaration names what it
//! ports and the line it began at.
//!
//! The tower reuses `qwen3tts_gen.CodeGen` for sampling and the codec
//! `qwen3tts_gen.Code2Wav`, each over a `graph.Graph.initChild` of this
//! graph, as upstream constructs `code_gen` and `code2wav` from `*this`.

const std = @import("std");
const libm = @import("stdlib").libm;
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const module = @import("module.zig");
const qwen3tts_gen = @import("qwen3tts_gen.zig");
const c = impl.c;
const gpa = impl.gpa;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

/// `DRAMA3_N_ROT` (drama3-gen.cpp:7 @b1b770d90): tower RoPE, interleaved,
/// first 64 dims of each head.
const drama3_n_rot: c_int = 64;
/// `DRAMA3_ROPE_THETA` (drama3-gen.cpp:8 @b1b770d90).
const drama3_rope_theta: f32 = 1e7;

/// `n_pos_max` (mtmd/models/models.h:524 @b1b770d90), a static member of
/// `mtmd_graph_drama3_gen`: tower KV rows, shared by the prompt and the
/// generated frames.
pub const n_pos_max: c_int = 4096;

/// Ports `list_drama3_state_slots` (drama3-gen.cpp:10 @b1b770d90): the tower
/// KV slots, written in place by the graph, then `list_c2w_state_slots`. The
/// caller frees the slice with `module.freeSlots`.
pub fn listStateSlots(hparams: *const model_mod.Hparams, model: *const model_mod.Model) clip.Error![]module.StateSlot {
    var slots: std.ArrayList(module.StateSlot) = .empty;
    errdefer {
        for (slots.items) |s| gpa.free(s.name);
        slots.deinit(gpa);
    }
    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "tower_pos", 0), .ne0 = 1, .ne1 = 1, .in_place = true });
    for (model.layers.items, 0..) |*layer, il| {
        const n_kv = ne(layer.k_w, 1);
        try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "tower_k_%zu", .{il}), .ne0 = n_kv, .ne1 = n_pos_max, .in_place = true });
        try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "tower_v_%zu", .{il}), .ne0 = n_kv, .ne1 = n_pos_max, .in_place = true });
    }
    const c2w = try qwen3tts_gen.listStateSlots(hparams, model);
    defer gpa.free(c2w);
    // the names move into `slots`
    try slots.ensureUnusedCapacity(gpa, c2w.len);
    slots.appendSliceAssumeCapacity(c2w);
    return slots.toOwnedSlice(gpa);
}

/// The builder's parameters, as `mtmd_get_graph_builder` derives them.
const Builder = struct {
    g: *graph.Graph,
    gen_process: impl.GenProcessType,
    gen_state: ?*clip.GenState, // null on buffer reserve
    n_frames: c_int, // GEN_WAV only
    top_k: c_int,
    top_p: f32,
    temp: f32,
    arena: std.mem.Allocator,

    /// Ports `tower_layer` (drama3-gen.cpp:25 @b1b770d90), a member of
    /// `mtmd_graph_drama3_gen`:
    /// h_out = out_norm(Block(eh_proj([enorm(mask * E[code] + extra); hnorm(h_in)]))).
    fn towerLayer(self: *const Builder, x: T, il: c_int, code: T, mask: T, extra: T, inp_pos: T, k_slot: T, v_slot: T, pos: c_int) T {
        const g = self.g;
        const ctx0 = g.ctx0;
        const hparams = g.hparams;
        const layer = &g.model.layers.items[@intCast(il)];
        const tl = &g.model.drama3_layers.items[@intCast(il)];
        const n_head: c_int = hparams.n_head;
        const n_head_kv: c_int = hparams.n_head_kv;
        const d_head: i64 = @divTrunc(ne(layer.q_w, 1), n_head);
        const n_kv: i64 = d_head * n_head_kv;
        const eps: f32 = hparams.eps;

        var e = c.ggml_mul(ctx0, c.ggml_get_rows(ctx0, tl.embd_w, code), mask);
        if (extra != null) {
            e = c.ggml_add(ctx0, e, extra);
        }
        e = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, e, eps), tl.enorm_w);
        const h = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, x, eps), tl.hnorm_w);
        const mixed = c.ggml_mul_mat(ctx0, tl.eh_proj_w, c.ggml_concat(ctx0, e, h, 0));

        var cur = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, mixed, eps), layer.ln_1_w);
        var q = c.ggml_reshape_3d(ctx0, c.ggml_mul_mat(ctx0, layer.q_w, cur), d_head, n_head, 1);
        var k = c.ggml_reshape_3d(ctx0, c.ggml_mul_mat(ctx0, layer.k_w, cur), d_head, n_head_kv, 1);
        const v = c.ggml_mul_mat(ctx0, layer.v_w, cur);
        const gate = c.ggml_mul_mat(ctx0, layer.attn_gate_w, cur);

        q = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, q, eps), layer.q_norm);
        k = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, k, eps), layer.k_norm);
        q = c.ggml_rope_ext(ctx0, q, inp_pos, null, drama3_n_rot, c.GGML_ROPE_TYPE_NORMAL, 0, drama3_rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        k = c.ggml_rope_ext(ctx0, k, inp_pos, null, drama3_n_rot, c.GGML_ROPE_TYPE_NORMAL, 0, drama3_rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

        // write K/V of this row to the slot, attention uses the cached rows + this row
        const pos_u: usize = @intCast(pos);
        const k_new = c.ggml_reshape_2d(ctx0, k, n_kv, 1);
        c.ggml_build_forward_expand(g.gf, c.ggml_cpy(ctx0, k_new, c.ggml_view_2d(ctx0, k_slot, n_kv, 1, nb(k_slot, 1), pos_u * nb(k_slot, 1))));
        c.ggml_build_forward_expand(g.gf, c.ggml_cpy(ctx0, v, c.ggml_view_2d(ctx0, v_slot, n_kv, 1, nb(v_slot, 1), pos_u * nb(v_slot, 1))));
        var k_all = k_new;
        var v_all = v;
        if (pos > 0) {
            k_all = c.ggml_concat(ctx0, c.ggml_view_2d(ctx0, k_slot, n_kv, pos, nb(k_slot, 1), 0), k_new, 1);
            v_all = c.ggml_concat(ctx0, c.ggml_view_2d(ctx0, v_slot, n_kv, pos, nb(v_slot, 1), 0), v, 1);
        }

        var attn = g.buildAttn(
            null,
            null,
            c.ggml_reshape_4d(ctx0, q, d_head, n_head, 1, 1),
            c.ggml_reshape_4d(ctx0, k_all, d_head, n_head_kv, pos + 1, 1),
            c.ggml_reshape_4d(ctx0, v_all, d_head, n_head_kv, pos + 1, 1),
            null,
            1.0 / libm.sqrtfInt(d_head),
            il,
            null,
        );
        attn = c.ggml_mul(ctx0, attn, c.ggml_sigmoid(ctx0, gate));
        cur = c.ggml_add(ctx0, mixed, c.ggml_mul_mat(ctx0, layer.o_w, attn));

        var ff = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, cur, eps), layer.ln_2_w);
        ff = c.ggml_swiglu_split(ctx0, c.ggml_mul_mat(ctx0, layer.ff_gate_w, ff), c.ggml_mul_mat(ctx0, layer.ff_up_w, ff));
        cur = c.ggml_add(ctx0, cur, c.ggml_mul_mat(ctx0, layer.ff_down_w, ff));

        cur = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, cur, eps), tl.out_norm_w);
        g.cb(cur, "tower_out", il);
        return cur;
    }

    /// Ports `build_tower` (drama3-gen.cpp:83 @b1b770d90), a member of
    /// `mtmd_graph_drama3_gen`: one row, backbone hidden + next main code to
    /// the other codes of that frame + the next backbone input.
    fn buildTower(self: *const Builder) clip.Error!?*c.ggml_cgraph {
        const g = self.g;
        const ctx0 = g.ctx0;
        const model = g.model;
        const n_layer: c_int = @intCast(model.layers.items.len);
        const n_heads: c_int = @intCast(ne(model.gen_code_head_w, 2)); // one per residual codebook
        const n_shared = n_layer - n_heads;
        const n_cb = ne(model.gen_code_embd_w, 2);
        const n_embd_cb = ne(model.gen_code_embd_w, 0);

        var pos: c_int = 0;
        var slots: std.ArrayList(T) = .empty;
        const slot_list = try listStateSlots(g.hparams, model);
        defer module.freeSlots(slot_list);
        if (self.gen_state) |gs| {
            var pos_f: f32 = 0.0;
            c.ggml_backend_tensor_get(gs.slots.items[0], &pos_f, 0, @sizeOf(f32));
            pos = @intFromFloat(pos_f);
            for (gs.slots.items) |t| try slots.append(self.arena, t);
        } else {
            // buffer reserve: no gen state, use stand-in slots
            for (slot_list) |slot| {
                var t: T = null;
                if (slot.in_place) {
                    t = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, slot.ne0, slot.ne1);
                    c.ggml_set_input(t);
                }
                try slots.append(self.arena, t);
            }
        }
        impl.ggmlAssert(pos < n_pos_max, "drama3-gen.cpp", 109, "pos < n_pos_max");

        const h = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, g.n_mmproj_embd);
        _ = c.ggml_set_name(h, "inp_raw");
        c.ggml_set_input(h);

        const inp_code0 = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, 1);
        _ = c.ggml_set_name(inp_code0, "inp_code0");
        c.ggml_set_input(inp_code0);

        // codes 2.. of the input frame, as row ids of the merged codebook table
        const inp_extra = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_cb - 2);
        _ = c.ggml_set_name(inp_extra, "inp_extra");
        c.ggml_set_input(inp_extra);

        // [next frame is speech, input frame is speech]
        const inp_scale = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, 2);
        _ = c.ggml_set_name(inp_scale, "inp_scale");
        c.ggml_set_input(inp_scale);

        const inp_pos = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, 1);
        _ = c.ggml_set_name(inp_pos, "inp_pos");
        c.ggml_set_input(inp_pos);

        const inp_rand = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, n_heads);
        _ = c.ggml_set_name(inp_rand, "inp_rand");
        c.ggml_set_input(inp_rand);

        const mask = c.ggml_view_1d(ctx0, inp_scale, 1, 0);
        const extra_on = c.ggml_view_1d(ctx0, inp_scale, 1, @sizeOf(f32));

        const cb_all = c.ggml_reshape_2d(ctx0, model.gen_code_embd_w, n_embd_cb, ne(model.gen_code_embd_w, 1) * n_cb);
        var extra = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, c.ggml_get_rows(ctx0, cb_all, inp_extra)));
        extra = c.ggml_mul(ctx0, c.ggml_reshape_2d(ctx0, c.ggml_sum_rows(ctx0, extra), n_embd_cb, 1), extra_on);

        const cg: qwen3tts_gen.CodeGen = .{ .g = graph.Graph.initChild(g), .top_k = self.top_k, .top_p = self.top_p };
        var out_codes = cg.cacheSet(c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_I32, 1, n_cb), 0, inp_code0);

        var code = inp_code0;
        var x = c.ggml_reshape_2d(ctx0, h, g.n_mmproj_embd, 1);
        var il: c_int = 0;
        while (il < n_layer) : (il += 1) {
            const ilu: usize = @intCast(il);
            x = self.towerLayer(x, il, code, mask, if (il == 0) extra else null, inp_pos, slots.items[1 + 2 * ilu], slots.items[2 + 2 * ilu], pos);
            if (il < n_shared) {
                continue;
            }
            const ih = il - n_shared;
            const hw = model.gen_code_head_w;
            const head = c.ggml_view_2d(ctx0, hw, ne(hw, 0), ne(hw, 1), nb(hw, 1), @as(usize, @intCast(ih)) * nb(hw, 2));
            var logits = c.ggml_mul_mat(ctx0, head, x);
            if (self.temp > 0.0 and self.temp != 1.0) {
                logits = c.ggml_scale(ctx0, logits, 1.0 / self.temp);
            }
            code = cg.doSampling(logits, c.ggml_view_1d(ctx0, inp_rand, 1, @as(usize, @intCast(ih)) * @sizeOf(f32)));
            out_codes = cg.cacheSet(out_codes, ih + 1, code);
        }

        const pos_slot = slots.items[0];
        c.ggml_build_forward_expand(g.gf, c.ggml_cpy(ctx0, c.ggml_scale_bias(ctx0, pos_slot, 1.0, 1.0), pos_slot));

        out_codes = c.ggml_cont(ctx0, out_codes);
        _ = c.ggml_set_name(out_codes, "out_codes");
        c.ggml_set_output(out_codes);
        c.ggml_build_forward_expand(g.gf, out_codes);

        // backbone input of the next row: (E_lm[main] + E_cb1[code 1]) / sqrt(2)
        const code1 = c.ggml_view_1d(ctx0, out_codes, 1, nb(out_codes, 1));
        const ew = model.gen_code_embd_w;
        const cb1 = c.ggml_view_2d(ctx0, ew, n_embd_cb, ne(ew, 1), nb(ew, 1), nb(ew, 2));
        var out_embd = c.ggml_add(ctx0, c.ggml_get_rows(ctx0, model.gen_code_out_embd_w, inp_code0), c.ggml_get_rows(ctx0, cb1, code1));
        out_embd = c.ggml_scale(ctx0, out_embd, 1.0 / libm.sqrtf(2.0));
        g.cb(out_embd, "gen_audio_out", -1);
        c.ggml_build_forward_expand(g.gf, out_embd);

        return g.gf;
    }

    /// Ports `build_codec` (drama3-gen.cpp:186 @b1b770d90), a member of
    /// `mtmd_graph_drama3_gen`: frames to PCM, qwen3tts code2wav blocks
    /// without pre_conv and tfm projections.
    fn buildCodec(self: *const Builder) clip.Error!?*c.ggml_cgraph {
        const g = self.g;
        const ctx0 = g.ctx0;
        const model = g.model;
        const hparams = g.hparams;
        const arena = self.arena;
        const c2w = &model.c2w;
        const n_cb = ne(model.gen_code_embd_w, 2);
        const n_frames: i64 = self.n_frames;

        const inp_codes = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_I32, n_frames, n_cb);
        _ = c.ggml_set_name(inp_codes, "inp_codes");
        c.ggml_set_input(inp_codes);

        var dec: qwen3tts_gen.Code2Wav = .{ .g = graph.Graph.initChild(g), .arena = arena };
        dec.gelu_erf = true;
        {
            const slots = try qwen3tts_gen.listStateSlots(hparams, model);
            defer module.freeSlots(slots);
            for (slots) |slot| {
                const t = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, slot.ne0, slot.ne1);
                _ = c.ggml_set_name(t, (try impl.stringFormat(arena, "state_in_%s", .{slot.name.ptr})).ptr);
                c.ggml_set_input(t);
                try dec.state_in.put(arena, try arena.dupe(u8, slot.name), t);
            }
        }

        // out_proj is folded into the codebook tables at conversion
        var cur = c.ggml_get_rows(ctx0, c2w.quant_first_cb_w, groupIds(ctx0, inp_codes, n_frames, 0)); // [dim, T]
        var gi: i64 = 1;
        while (gi < n_cb) : (gi += 1) {
            const rc = c2w.quant_rest_cb_w;
            const table = c.ggml_view_2d(ctx0, rc, ne(rc, 0), ne(rc, 1), nb(rc, 1), @as(usize, @intCast(gi - 1)) * nb(rc, 2));
            cur = c.ggml_add(ctx0, cur, c.ggml_get_rows(ctx0, table, groupIds(ctx0, inp_codes, n_frames, gi)));
        }

        var il: c_int = 0;
        while (il < hparams.wav_tfm_n_layer) : (il += 1) {
            cur = dec.tfmLayerForward(cur, &c2w.tfm_layers.items[@intCast(il)], il);
        }
        cur = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps), c2w.tfm_output_norm_w);
        g.cb(cur, "wav_tfm_out", -1);

        var x = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur)); // [T, dim]
        for (c2w.upsample.items, 0..) |*up, iu| {
            x = dec.causalConvTranspose1d(x, up.conv_w, up.conv_b, 2, "");
            x = dec.convnextBlock(x, up, try impl.stringFormat(arena, "up%zu", .{iu}));
        }

        x = dec.causalConv1d(x, c2w.dac_entry_w, c2w.dac_entry_b, 1, "dac_entry");
        for (c2w.dac.items, 0..) |*blk, ib| {
            const stride: c_int = @intCast(@divTrunc(ne(blk.conv_w, 0), 2));
            x = dec.snake(x, blk.snake_alpha, blk.snake_beta);
            x = dec.causalConvTranspose1d(x, blk.conv_w, blk.conv_b, stride, try impl.stringFormat(arena, "dac%zu_tail", .{ib}));
            for (blk.res.items, 0..) |*res, ir| {
                x = dec.dacResUnit(x, res, qwen3tts_gen.dac_dilations[ir], try impl.stringFormat(arena, "dac%zu_res%zu", .{ ib, ir }));
            }
        }
        x = dec.snake(x, c2w.dac_post_snake_alpha, c2w.dac_post_snake_beta);
        x = dec.causalConv1d(x, c2w.dac_post_conv_w, c2w.dac_post_conv_b, 1, "dac_post_conv");
        x = c.ggml_tanh(ctx0, c.ggml_reshape_1d(ctx0, x, ne(x, 0)));
        _ = c.ggml_set_name(x, "out_audio");
        c.ggml_set_output(x);
        c.ggml_build_forward_expand(g.gf, x);

        for (dec.state_out.items) |slot| {
            _ = c.ggml_set_name(slot.t, (try impl.stringFormat(arena, "state_out_%s", .{slot.name.ptr})).ptr);
            c.ggml_set_output(slot.t);
            c.ggml_build_forward_expand(g.gf, slot.t);
        }
        return g.gf;
    }

    /// The `group_ids` lambda of `build_codec`: codebook `gi`'s ids over all
    /// frames.
    fn groupIds(ctx0: ?*c.ggml_context, inp_codes: T, n_frames: i64, gi: i64) T {
        return c.ggml_view_1d(ctx0, inp_codes, n_frames, @as(usize, @intCast(gi)) * nb(inp_codes, 1));
    }
};

/// Ports `build` (drama3-gen.cpp:254 @b1b770d90), a member of
/// `mtmd_graph_drama3_gen`, with the builder's parameters derived as
/// `mtmd_get_graph_builder`'s `PROJECTOR_TYPE_DRAMA3_GEN` arm does
/// (mtmd-graph.cpp:1014 @b1b770d90).
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    const gen_process: impl.GenProcessType = if (params) |p| p.gen_process else .gen_code;
    const n_cb = ne(g.model.gen_code_embd_w, 2);
    const n_frames: c_int = if (params != null and params.?.codes != null) @intCast(@divTrunc(@as(i64, @intCast(params.?.codes.?.items.len)), n_cb)) else 1;

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();

    const b: Builder = .{
        .g = g,
        .gen_process = gen_process,
        .gen_state = if (params) |p| p.gen_state else null,
        .n_frames = n_frames,
        .top_k = if (params) |p| p.top_k else 30,
        .top_p = if (params) |p| p.top_p else 0.9,
        .temp = if (params) |p| p.temp else 1.0,
        .arena = arena_state.allocator(),
    };
    return if (b.gen_process == .gen_wav) b.buildCodec() else b.buildTower();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
