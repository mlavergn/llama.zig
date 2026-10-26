//! The pocket-tts generation stages. GEN_CODE takes a backbone hidden state
//! to the next 32-d latent (flow matching) and an end-of-speech score;
//! GEN_WAV takes a window of latents to PCM through the mimi decoder. There
//! is no codebook anywhere: "codes" in the mtmd API are continuous features
//! here.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/pockettts-gen.cpp` at v0.3.0
//! (`c1d0e7a00`), and the `PROJECTOR_TYPE_POCKETTTS_GEN` arm of
//! `clip_get_graph_builder` (`llama.cpp/tools/mtmd/clip.cpp`). Each
//! declaration names what it ports and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const module = @import("module.zig");
const seanet_mod = @import("pockettts_seanet.zig");
const c = impl.c;
const gpa = impl.gpa;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

/// Ports `clip_graph_pockettts_gen` (mtmd/models/models.h:364
/// @c1d0e7a00): the builder's
/// parameters beside the parent graph.
const Gen = struct {
    g: *graph.Graph,
    gen_process: impl.GenProcessType,
    n_step: c_int, // lsd_decode steps, fixed at graph-build time
    n_frames: c_int, // GEN_WAV only: number of latents to decode

    /// Ports `modulate` (pockettts-gen.cpp:12 @c1d0e7a00), a member of
    /// `clip_graph_pockettts_gen`: AdaLN modulation, `x * (1 + scale) +
    /// shift`.
    fn modulate(self: *const Gen, x: T, shift: T, scale: T) T {
        const ctx0 = self.g.ctx0;
        const cur = c.ggml_mul(ctx0, x, c.ggml_scale_bias(ctx0, scale, 1.0, 1.0));
        return c.ggml_add(ctx0, cur, shift);
    }

    /// Ports `time_embed` (pockettts-gen.cpp:18 @c1d0e7a00), a member of
    /// `clip_graph_pockettts_gen`: TimestepEmbedder in the reference.
    fn timeEmbed(self: *const Gen, te: *const model_mod.FlowNet.TimeEmbd, t: f32) T {
        const ctx0 = self.g.ctx0;
        // t is a graph-build constant, so the cos/sin table can be folded into a scaled copy
        const args = c.ggml_scale(ctx0, te.freqs, t);
        const emb = c.ggml_concat(ctx0, c.ggml_cos(ctx0, args), c.ggml_sin(ctx0, args), 0);

        var cur = self.g.buildMm(te.up_w, emb);
        cur = c.ggml_add(ctx0, cur, te.up_b);
        cur = c.ggml_silu(ctx0, cur);
        cur = self.g.buildMm(te.down_w, cur);
        cur = c.ggml_add(ctx0, cur, te.down_b);

        // this "RMSNorm" divides by the unbiased variance, not the mean square
        // it also rescales the input, not the centered value, see _rms_norm() in mlp.py
        {
            const n = ne(cur, 0);
            const mean = c.ggml_mean(ctx0, cur);
            const dev = c.ggml_sub(ctx0, cur, mean);
            var variance = c.ggml_mean(ctx0, c.ggml_sqr(ctx0, dev));
            variance = c.ggml_scale_bias(ctx0, variance, @as(f32, @floatFromInt(n)) / @as(f32, @floatFromInt(n - 1)), 1e-5);
            cur = c.ggml_div(ctx0, cur, c.ggml_sqrt(ctx0, variance));
            cur = c.ggml_mul(ctx0, cur, te.norm);
        }

        return cur;
    }

    /// Ports `flow_forward` (pockettts-gen.cpp:45 @c1d0e7a00), a member of
    /// `clip_graph_pockettts_gen`: one velocity evaluation, `v(cond, s, t,
    /// x)`.
    fn flowForward(self: *const Gen, cond: T, x: T, s: f32, t: f32) T {
        const ctx0 = self.g.ctx0;
        const flow = &self.g.model.flow;

        var cur = self.g.buildMm(flow.input_proj_w, x);
        cur = c.ggml_add(ctx0, cur, flow.input_proj_b);

        // the two time conditions are averaged, then added to the projected backbone state
        var ts = c.ggml_add(ctx0, self.timeEmbed(&flow.time.items[0], s), self.timeEmbed(&flow.time.items[1], t));
        ts = c.ggml_scale(ctx0, ts, 1.0 / @as(f32, @floatFromInt(flow.time.items.len)));

        // `c` upstream, which names the C namespace here
        var c_emb = self.g.buildMm(flow.cond_embd_w, cond);
        c_emb = c.ggml_add(ctx0, c_emb, flow.cond_embd_b);

        const y = c.ggml_add(ctx0, ts, c_emb);
        self.g.cb(y, "flow_cond", -1);

        const n_ch: i64 = if (flow.blocks.items.len == 0) 0 else ne(flow.blocks.items[0].norm_w, 0);

        for (flow.blocks.items, 0..) |*blk, il| {
            var mod = self.g.buildMm(blk.ada_w, c.ggml_silu(ctx0, y));
            mod = c.ggml_add(ctx0, mod, blk.ada_b);

            const shift = c.ggml_view_1d(ctx0, mod, n_ch, 0);
            const scale = c.ggml_view_1d(ctx0, mod, n_ch, @as(usize, @intCast(n_ch)) * nb(mod, 0));
            const gate = c.ggml_view_1d(ctx0, mod, n_ch, 2 * @as(usize, @intCast(n_ch)) * nb(mod, 0));

            var h = self.g.buildNorm(cur, blk.norm_w, blk.norm_b, .normal, 1e-6, @intCast(il));
            h = self.modulate(h, shift, scale);
            h = self.g.buildMm(blk.up_w, h);
            h = c.ggml_add(ctx0, h, blk.up_b);
            h = c.ggml_silu(ctx0, h);
            h = self.g.buildMm(blk.down_w, h);
            h = c.ggml_add(ctx0, h, blk.down_b);

            cur = c.ggml_add(ctx0, cur, c.ggml_mul(ctx0, gate, h));
            self.g.cb(cur, "flow_blk", @intCast(il));
        }

        // final layer: the norm has no weights, only the AdaLN modulation
        var mod = self.g.buildMm(flow.final_ada_w, c.ggml_silu(ctx0, y));
        mod = c.ggml_add(ctx0, mod, flow.final_ada_b);

        const shift = c.ggml_view_1d(ctx0, mod, n_ch, 0);
        const scale = c.ggml_view_1d(ctx0, mod, n_ch, @as(usize, @intCast(n_ch)) * nb(mod, 0));

        cur = self.g.buildNorm(cur, null, null, .normal, 1e-6, -1);
        cur = self.modulate(cur, shift, scale);
        cur = self.g.buildMm(flow.final_proj_w, cur);
        cur = c.ggml_add(ctx0, cur, flow.final_proj_b);

        return cur;
    }
};

/// Ports `list_pockettts_state_slots` (pockettts-gen.cpp:102 @c1d0e7a00):
/// the state carried between GEN_WAV calls -- rope offset, per-layer KV
/// window, conv left context and the transposed-conv overlap tails. The
/// caller frees the slice with `module.freeSlots`.
pub fn listStateSlots(hparams: *const model_mod.Hparams, model: *const model_mod.Model) clip.Error![]module.StateSlot {
    var slots: std.ArrayList(module.StateSlot) = .empty;
    errdefer {
        for (slots.items) |s| gpa.free(s.name);
        slots.deinit(gpa);
    }
    if (model.gen_upsample_w == null) {
        return slots.toOwnedSlice(gpa); // not a pocket-tts decoder
    }
    const seanet = &model.seanet;

    // the slots below are sized from these
    impl.ggmlAssert(model.gen_tfm_layers.items.len != 0, "pockettts-gen.cpp", 110, "!model.gen_tfm_layers.empty()");
    impl.ggmlAssert(@as(c_int, @intCast(seanet.stages.items.len)) >= hparams.seanet_n_stage, "pockettts-gen.cpp", 111, "(int) seanet.stages.size() >= hparams.seanet_n_stage");
    impl.ggmlAssert(@as(c_int, @intCast(hparams.seanet_ratios.items.len)) >= hparams.seanet_n_stage, "pockettts-gen.cpp", 112, "(int) hparams.seanet_ratios.size() >= hparams.seanet_n_stage");
    impl.ggmlAssert(hparams.mimi_tfm_context > 1 and hparams.mimi_downsample > 0, "pockettts-gen.cpp", 113, "hparams.mimi_tfm_context > 1 && hparams.mimi_downsample > 0");

    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "tfm_pos", 0), .ne0 = 1, .ne1 = 1 });

    const n_embd_a = ne(model.gen_tfm_layers.items[0].q_w, 1);
    const prefix: i64 = hparams.mimi_tfm_context - 1;
    for (0..model.gen_tfm_layers.items.len) |il| {
        try slots.append(gpa, .{ .name = try std.fmt.allocPrintSentinel(gpa, "tfm_k_{d}", .{il}, 0), .ne0 = n_embd_a, .ne1 = prefix });
        try slots.append(gpa, .{ .name = try std.fmt.allocPrintSentinel(gpa, "tfm_v_{d}", .{il}, 0), .ne0 = n_embd_a, .ne1 = prefix });
    }

    // upsample is depthwise, its output channel count is the input one
    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "up", 0), .ne0 = ne(model.gen_upsample_w, 0) - hparams.mimi_downsample, .ne1 = ne(model.gen_upsample_w, 2) });

    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "dec_in", 0), .ne0 = ne(seanet.conv_in_w, 0) - 1, .ne1 = ne(seanet.conv_in_w, 1) });
    var i: c_int = 0;
    while (i < hparams.seanet_n_stage) : (i += 1) {
        const stage = &seanet.stages.items[@intCast(i)];
        const stride = hparams.seanet_ratios.items[@intCast(hparams.seanet_n_stage - 1 - i)];
        try slots.append(gpa, .{ .name = try std.fmt.allocPrintSentinel(gpa, "dec_up_{d}", .{i}, 0), .ne0 = ne(stage.scale_conv_w, 0) - stride, .ne1 = ne(stage.scale_conv_w, 1) });
        try slots.append(gpa, .{ .name = try std.fmt.allocPrintSentinel(gpa, "dec_res_{d}", .{i}, 0), .ne0 = ne(stage.res_conv1_w, 0) - 1, .ne1 = ne(stage.res_conv1_w, 1) });
    }
    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "dec_out", 0), .ne0 = ne(seanet.conv_out_w, 0) - 1, .ne1 = ne(seanet.conv_out_w, 1) });

    return slots.toOwnedSlice(gpa);
}

/// Ports `build` (pockettts-gen.cpp:139 @c1d0e7a00), a member of
/// `clip_graph_pockettts_gen`, with the builder's parameters derived as
/// clip_get_graph_builder's `PROJECTOR_TYPE_POCKETTTS_GEN` arm does
/// (mtmd/clip.cpp:1100 @c1d0e7a00).
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    const model = g.model;
    const hparams = g.hparams;

    // clip_get_graph_builder's POCKETTTS_GEN arm
    const gen_process: impl.GenProcessType = if (params) |p| p.gen_process else .gen_code;
    const n_step: c_int = hparams.flow_n_step;
    const n_latent = ne(model.gen_input_lin_w, 0);
    impl.ggmlAssert(n_step > 0, "clip.cpp", 1105, "n_step > 0");
    impl.ggmlAssert(n_latent > 0, "clip.cpp", 1106, "n_latent > 0");
    // "inp_feats" takes the caller's buffer as-is, the graph must consume all of it
    const feats: ?*const std.ArrayList(f32) = if (params) |p| p.feats else null;
    if (feats) |f| {
        impl.ggmlAssert(f.items.len % @as(usize, @intCast(n_latent)) == 0, "clip.cpp", 1109, "params->feats->size() % (size_t) n_latent == 0");
        impl.ggmlAssert(f.items.len >= @as(usize, @intCast(n_latent)), "clip.cpp", 1110, "params->feats->size() >= (size_t) n_latent");
    }
    const n_frames: c_int = if (feats) |f| @intCast(f.items.len / @as(usize, @intCast(n_latent))) else 1;

    const self: Gen = .{ .g = g, .gen_process = gen_process, .n_step = n_step, .n_frames = n_frames };
    const ctx0 = g.ctx0;

    if (self.gen_process == .gen_code) {
        // the backbone hidden state arrives as the single batch entry
        var h_state = g.buildInpRaw(1);
        h_state = c.ggml_reshape_2d(ctx0, h_state, g.n_mmproj_embd, 1);

        // end-of-speech probe, thresholded on the host side
        var eos = g.buildMm(model.gen_out_eos_w, h_state);
        eos = c.ggml_add(ctx0, eos, model.gen_out_eos_b);
        _ = c.ggml_set_name(eos, "out_eos_score");
        c.ggml_set_output(eos);
        c.ggml_build_forward_expand(g.gf, eos);

        const noise = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_latent, 1);
        _ = c.ggml_set_name(noise, "inp_noise");
        c.ggml_set_input(noise);

        // lsd_decode: integrate the velocity field from the noise sample
        var cur: T = noise;
        var i: c_int = 0;
        while (i < self.n_step) : (i += 1) {
            const s = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(self.n_step));
            const t = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(self.n_step));
            const v = self.flowForward(h_state, cur, s, t);
            cur = c.ggml_add(ctx0, cur, c.ggml_scale(ctx0, v, 1.0 / @as(f32, @floatFromInt(self.n_step))));
        }
        g.cb(cur, "flow_latent", -1);

        _ = c.ggml_set_name(cur, "out_feats");
        c.ggml_set_output(cur);
        c.ggml_build_forward_expand(g.gf, cur);

        // the same latent, projected into the backbone's input space for the next step
        const embd = g.buildMm(model.gen_input_lin_w, cur);
        g.cb(embd, "gen_embd", -1);
        c.ggml_build_forward_expand(g.gf, embd);

        return g.gf;
    }

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // GEN_WAV: [32, n_frames] latents -> PCM
    const feats_t = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, ne(model.gen_input_lin_w, 0), self.n_frames);
    _ = c.ggml_set_name(feats_t, "inp_feats");
    c.ggml_set_input(feats_t);

    // denormalize, then the DummyQuantizer up-projection
    var cur = c.ggml_add(ctx0, c.ggml_mul(ctx0, feats_t, model.gen_emb_std), model.gen_emb_mean);
    cur = g.buildMm(model.gen_quant_out_w, cur);
    g.cb(cur, "quant_out", -1);

    var seanet: seanet_mod.Seanet = .init(g, arena);
    {
        const slots = try listStateSlots(hparams, model);
        defer module.freeSlots(slots);
        for (slots) |slot| {
            const t = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, slot.ne0, slot.ne1);
            const name = try std.fmt.allocPrintSentinel(arena, "state_in_{s}", .{slot.name}, 0);
            _ = c.ggml_set_name(t, name.ptr);
            c.ggml_set_input(t);
            seanet.putState(slot.name, t);
        }
    }

    // model frame rate -> encoder frame rate, depthwise transposed conv
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = seanet.convTranspose1d(cur, model.gen_upsample_w, null, hparams.mimi_downsample, "up");
    g.cb(cur, "mimi_upsample", -1);

    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));

    // positions continue across calls, the counter lives in the state
    const n_pos = ne(cur, 1);
    const prefix: i64 = hparams.mimi_tfm_context - 1;
    const n_kv = prefix + n_pos;

    const base = c.ggml_reshape_1d(ctx0, seanet.stateAt("tfm_pos"), 1);
    const inp_pos = c.ggml_cast(ctx0, c.ggml_add(ctx0, c.ggml_arange(ctx0, 0.0, @floatFromInt(n_pos), 1.0), base), c.GGML_TYPE_I32);
    seanet.pushState("tfm_pos", c.ggml_scale_bias(ctx0, seanet.stateAt("tfm_pos"), 1.0, @floatFromInt(n_pos)));

    // banded causal mask over [cached prefix | this chunk]
    // the last factor masks out cache rows that hold no real frame yet
    const pos_k = c.ggml_reshape_2d(ctx0, c.ggml_arange(ctx0, 0.0, @floatFromInt(n_kv), 1.0), n_kv, 1);
    const pos_q = c.ggml_reshape_2d(ctx0, c.ggml_arange(ctx0, @floatFromInt(prefix), @floatFromInt(prefix + n_pos), 1.0), 1, n_pos);
    const diff = c.ggml_sub(ctx0, c.ggml_repeat_4d(ctx0, pos_q, n_kv, n_pos, 1, 1), pos_k);

    var keep = c.ggml_mul(
        ctx0,
        c.ggml_step(ctx0, c.ggml_scale_bias(ctx0, diff, 1.0, 0.5)), // delta >= 0
        c.ggml_step(ctx0, c.ggml_scale_bias(ctx0, diff, -1.0, @as(f32, @floatFromInt(hparams.mimi_tfm_context)) - 0.5)), // delta < context
    );
    keep = c.ggml_mul(ctx0, keep, c.ggml_step(ctx0, c.ggml_scale_bias(ctx0, c.ggml_add(ctx0, pos_k, base), 1.0, 0.5 - @as(f32, @floatFromInt(prefix)))));
    const kq_mask = c.ggml_reshape_4d(ctx0, c.ggml_log(ctx0, keep), n_kv, n_pos, 1, 1);

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.gen_tfm_layers.items[@intCast(il)];
        var inp = cur;

        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, .normal, g.eps, il);

        var Qcur = g.buildMm(layer.q_w, cur);
        var Kcur = g.buildMm(layer.k_w, cur);
        const Vcur = g.buildMm(layer.v_w, cur);

        Qcur = c.ggml_reshape_3d(ctx0, Qcur, g.d_head, g.n_head, n_pos);
        Kcur = c.ggml_reshape_3d(ctx0, Kcur, g.d_head, g.n_head, n_pos);

        Qcur = c.ggml_rope_ext(ctx0, Qcur, inp_pos, null, g.d_head, c.GGML_ROPE_TYPE_NORMAL, 0, hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        Kcur = c.ggml_rope_ext(ctx0, Kcur, inp_pos, null, g.d_head, c.GGML_ROPE_TYPE_NORMAL, 0, hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

        // prepend the cached window, then keep this chunk's tail for the next call
        const k_name = try std.fmt.allocPrint(arena, "tfm_k_{d}", .{il});
        const v_name = try std.fmt.allocPrint(arena, "tfm_v_{d}", .{il});
        const k_full = c.ggml_concat(ctx0, seanet.stateAt(k_name), c.ggml_reshape_2d(ctx0, Kcur, @as(i64, g.d_head) * g.n_head, n_pos), 1);
        const v_full = c.ggml_concat(ctx0, seanet.stateAt(v_name), Vcur, 1);
        seanet.pushState(k_name, c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, k_full, ne(k_full, 0), prefix, nb(k_full, 1), @as(usize, @intCast(n_pos)) * nb(k_full, 1))));
        seanet.pushState(v_name, c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, v_full, ne(v_full, 0), prefix, nb(v_full, 1), @as(usize, @intCast(n_pos)) * nb(v_full, 1))));

        const q_cur = c.ggml_reshape_4d(ctx0, Qcur, g.d_head, g.n_head, n_pos, 1);
        const k_cur = c.ggml_reshape_4d(ctx0, k_full, g.d_head, g.n_head, n_kv, 1);
        const v_cur = c.ggml_reshape_4d(ctx0, v_full, g.d_head, g.n_head, n_kv, 1);

        cur = g.buildAttn(layer.o_w, null, q_cur, k_cur, v_cur, kq_mask, g.kq_scale, il, null);
        cur = c.ggml_mul(ctx0, cur, layer.ls_1_w);
        cur = c.ggml_add(ctx0, cur, inp);

        inp = cur;
        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .normal, g.eps, il);
        cur = g.buildFfn(cur, layer.ff_up_w, null, null, null, layer.ff_down_w, null, .gelu, il);
        cur = c.ggml_mul(ctx0, cur, layer.ls_2_w);
        cur = c.ggml_add(ctx0, cur, inp);
    }
    g.cb(cur, "mimi_dec_tfm", -1);

    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = seanet.decode(cur);

    for (seanet.state_out.items) |s| {
        const name = try std.fmt.allocPrintSentinel(arena, "state_out_{s}", .{s.name}, 0);
        _ = c.ggml_set_name(s.t, name.ptr);
        c.ggml_set_output(s.t);
        c.ggml_build_forward_expand(g.gf, s.t);
    }

    // [n_samples, 1] -> [n_samples], clamped like the reference output
    cur = c.ggml_reshape_1d(ctx0, cur, ne(cur, 0));
    cur = c.ggml_clamp(ctx0, cur, -1.0, 1.0);
    _ = c.ggml_set_name(cur, "out_audio");
    c.ggml_set_output(cur);
    c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
