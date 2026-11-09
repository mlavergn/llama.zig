//! The LFM3-Audio encoder: a conv downsampler (8x in time), a causal LFM2
//! stack, then an MLP projector. The graph always streams: a zero state
//! encodes a clip from its start.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/lfm3a.cpp` at v0.8.0 (`8642541b2`).
//! Each declaration names what it ports and the line it began at.
//!
//! `state_in` is a `std::map` read only through `at()`, so a hash map stands
//! in; `state_out` is a vector, iterated in insertion order, and stays a
//! list.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const module = @import("module.zig");
const c = impl.c;
const gpa = impl.gpa;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

/// A named `state_out` entry.
const StateOut = struct { name: [:0]const u8, t: T };

/// Ports `list_lfm3a_state_slots` (lfm3a.cpp:5 @8642541b2): the streaming
/// encoder state: base position, conv left context, short-conv taps and
/// attention window. A zero state gives the same result as encoding the whole
/// clip. The caller frees the slice with `module.freeSlots`.
pub fn listStateSlots(hparams: *const model_mod.Hparams, model: *const model_mod.Model) clip.Error![]module.StateSlot {
    var slots: std.ArrayList(module.StateSlot) = .empty;
    errdefer {
        for (slots.items) |s| gpa.free(s.name);
        slots.deinit(gpa);
    }

    const n_mel: i64 = hparams.n_mel_bins;
    const n_ch = ne(model.pre_encode_conv_X_w[0], 3);
    const prefix: i64 = hparams.attn_window_size - 1;

    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "pos", 0), .ne0 = 1, .ne1 = 1 });
    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "ds0", 0), .ne0 = 1, .ne1 = n_mel });
    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "ds1", 0), .ne0 = 1, .ne1 = @divTrunc(n_mel, 2) * n_ch });
    try slots.append(gpa, .{ .name = try gpa.dupeSentinel(u8, "ds2", 0), .ne0 = 1, .ne1 = @divTrunc(n_mel, 4) * n_ch });
    for (model.layers.items, 0..) |*layer, il| {
        if (layer.conv_dw_w != null) {
            try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "conv_%zu", .{il}), .ne0 = ne(layer.conv_dw_w, 0) - 1, .ne1 = hparams.n_embd });
        } else {
            try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "k_%zu", .{il}), .ne0 = ne(layer.k_w, 1), .ne1 = prefix });
            try slots.append(gpa, .{ .name = try impl.stringFormat(gpa, "v_%zu", .{il}), .ne0 = ne(layer.v_w, 1), .ne1 = prefix });
        }
    }

    return slots.toOwnedSlice(gpa);
}

/// The graph's state tensors, as the C++'s two lambdas capture them.
const Builder = struct {
    ctx0: ?*c.ggml_context,
    arena: std.mem.Allocator,
    state_in: std.StringHashMapUnmanaged(T) = .empty,
    state_out: std.ArrayList(StateOut) = .empty,

    /// `std::map::at` on a missing key: an uncaught `std::out_of_range`.
    fn at(self: *const Builder, name: []const u8) T {
        return self.state_in.get(name) orelse impl.abort("lfm3a.cpp", 0, "map::at: key not found");
    }

    fn out(self: *Builder, name: [:0]const u8, t: T) clip.Error!void {
        try self.state_out.append(self.arena, .{ .name = name, .t = t });
    }

    /// The `conv_down` lambda (lfm3a.cpp:46 @8642541b2): a stride-2 conv over
    /// [time, freq, ch]; the left column comes from the state and the right
    /// edge is zero-padded.
    fn convDown(self: *Builder, x_in: T, name: [:0]const u8, w: T, b: T, depthwise: bool) clip.Error!T {
        const ctx0 = self.ctx0;
        const left = c.ggml_reshape_4d(ctx0, self.at(name), 1, ne(x_in, 1), ne(x_in, 2), 1);
        var x = c.ggml_concat(ctx0, left, x_in, 0);
        const last = c.ggml_view_4d(ctx0, x, 1, ne(x, 1), ne(x, 2), 1, nb(x, 1), nb(x, 2), nb(x, 3), @as(usize, @intCast(ne(x, 0) - 1)) * nb(x, 0));
        try self.out(name, c.ggml_reshape_2d(ctx0, c.ggml_cont(ctx0, last), 1, ne(x, 1) * ne(x, 2)));
        x = c.ggml_pad_ext(ctx0, x, 0, 1, 0, 0, 0, 0, 0, 0);
        x = if (depthwise) c.ggml_conv_2d_dw_direct(ctx0, w, x, 2, 2, 0, 1, 1, 1) else c.ggml_conv_2d(ctx0, w, x, 2, 2, 0, 1, 1, 1);
        return c.ggml_add(ctx0, x, b);
    }
};

/// Ports `build` (lfm3a.cpp:30 @8642541b2), a member of `clip_graph_lfm3a`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const d_head = g.d_head;
    const n_head = g.n_head;
    const n_head_kv = g.n_head_kv;
    const eps = g.eps;

    const norm_t: model_mod.NormType = .rms;
    const rope_theta = hparams.rope_theta;
    const prefix: i64 = hparams.attn_window_size - 1;

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // the graph always streams, a zero state encodes a clip from its start
    var b: Builder = .{ .ctx0 = ctx0, .arena = arena };
    {
        const slots = try listStateSlots(hparams, model);
        defer module.freeSlots(slots);
        for (slots) |slot| {
            const t = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, slot.ne0, slot.ne1);
            const name = try impl.stringFormat(arena, "state_in_%s", .{slot.name.ptr});
            _ = c.ggml_set_name(t, name.ptr);
            c.ggml_set_input(t);
            try b.state_in.put(arena, try arena.dupe(u8, slot.name), t);
        }
    }

    // inp: [n_frames, n_mel, 1]
    var cur = g.buildInpRaw(1);

    // downsampler, same layout as the lfm2a pre-encoder but with SiLU
    cur = try b.convDown(cur, "ds0", model.pre_encode_conv_X_w[0], model.pre_encode_conv_X_b[0], false);
    cur = c.ggml_silu_inplace(ctx0, cur);

    cur = try b.convDown(cur, "ds1", model.pre_encode_conv_X_w[2], model.pre_encode_conv_X_b[2], true);
    cur = c.ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
    cur = c.ggml_silu_inplace(ctx0, cur);

    cur = try b.convDown(cur, "ds2", model.pre_encode_conv_X_w[5], model.pre_encode_conv_X_b[5], true);
    cur = c.ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
    cur = c.ggml_silu_inplace(ctx0, cur);
    g.cb(cur, "downsample.conv", -1);

    // [n_pos, n_mel/8, n_ch] -> [n_ch * n_mel/8, n_pos]
    const n_pos: c_int = @intCast(ne(cur, 0));
    cur = c.ggml_reshape_2d(ctx0, cur, n_pos, ne(cur, 1) * ne(cur, 2));
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = g.buildMm(model.pre_encode_out_w, cur);
    cur = c.ggml_add(ctx0, cur, model.pre_encode_out_b);
    g.cb(cur, "downsample.out", -1);

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);
    try b.out("pos", c.ggml_scale_bias(ctx0, b.at("pos"), 1.0, @floatFromInt(n_pos)));

    // causal mask with a sliding window, shared by all attention layers
    const kq_mask = g.buildInpAttnMask(prefix + n_pos, n_pos, 0);

    var inp_l = cur;

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];

        cur = g.buildNorm(inp_l, layer.ln_1_w, null, norm_t, eps, il);
        g.cb(cur, "operator_norm", il);

        if (layer.conv_dw_w != null) {
            // short conv: in_proj -> b * x -> causal depthwise conv -> c * conv -> out_proj
            const bcx = g.buildMm(layer.conv_pw1_w, cur);
            g.cb(bcx, "conv_in_proj", il);

            const bb = c.ggml_view_2d(ctx0, bcx, n_embd, n_pos, nb(bcx, 1), 0);
            const cc = c.ggml_view_2d(ctx0, bcx, n_embd, n_pos, nb(bcx, 1), c.ggml_row_size(bcx.?.type, n_embd));
            const x = c.ggml_view_2d(ctx0, bcx, n_embd, n_pos, nb(bcx, 1), c.ggml_row_size(bcx.?.type, 2 * @as(i64, n_embd)));

            var bx = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, c.ggml_mul(ctx0, bb, x)));
            const d_conv = ne(layer.conv_dw_w, 0) - 1;
            const name = try impl.stringFormat(arena, "conv_%d", .{il});
            bx = c.ggml_concat(ctx0, b.at(name), bx, 0);
            try b.out(name, c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, bx, d_conv, n_embd, nb(bx, 1), @as(usize, @intCast(n_pos)) * nb(bx, 0))));
            bx = c.ggml_reshape_3d(ctx0, bx, ne(bx, 0), n_embd, 1);

            cur = c.ggml_ssm_conv(ctx0, bx, layer.conv_dw_w);
            cur = c.ggml_reshape_2d(ctx0, cur, n_embd, n_pos);
            g.cb(cur, "conv_conv", il);

            cur = c.ggml_mul(ctx0, cc, cur);
            cur = g.buildMm(layer.conv_pw2_w, cur);
            g.cb(cur, "conv_out_proj", il);
        } else {
            var q_cur = g.buildMm(layer.q_w, cur);
            var k_cur = g.buildMm(layer.k_w, cur);
            var v_cur = g.buildMm(layer.v_w, cur);

            q_cur = c.ggml_reshape_3d(ctx0, q_cur, d_head, n_head, n_pos);
            k_cur = c.ggml_reshape_3d(ctx0, k_cur, d_head, n_head_kv, n_pos);
            v_cur = c.ggml_reshape_3d(ctx0, v_cur, d_head, n_head_kv, n_pos);

            q_cur = g.buildNorm(q_cur, layer.q_norm, null, norm_t, eps, il);
            k_cur = g.buildNorm(k_cur, layer.k_norm, null, norm_t, eps, il);

            q_cur = c.ggml_rope_ext(ctx0, q_cur, positions, null, d_head, c.GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            k_cur = c.ggml_rope_ext(ctx0, k_cur, positions, null, d_head, c.GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

            // the cached window goes in front, this chunk's tail is kept for the next call
            const k_name = try impl.stringFormat(arena, "k_%d", .{il});
            const v_name = try impl.stringFormat(arena, "v_%d", .{il});
            const n_embd_kv: i64 = @as(i64, d_head) * n_head_kv;
            const k_all = c.ggml_concat(ctx0, b.at(k_name), c.ggml_reshape_2d(ctx0, k_cur, n_embd_kv, n_pos), 1);
            const v_all = c.ggml_concat(ctx0, b.at(v_name), c.ggml_reshape_2d(ctx0, v_cur, n_embd_kv, n_pos), 1);
            try b.out(k_name, c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, k_all, n_embd_kv, prefix, nb(k_all, 1), @as(usize, @intCast(n_pos)) * nb(k_all, 1))));
            try b.out(v_name, c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, v_all, n_embd_kv, prefix, nb(v_all, 1), @as(usize, @intCast(n_pos)) * nb(v_all, 1))));
            k_cur = c.ggml_reshape_3d(ctx0, k_all, d_head, n_head_kv, prefix + n_pos);
            v_cur = c.ggml_reshape_3d(ctx0, v_all, d_head, n_head_kv, prefix + n_pos);

            cur = g.buildAttn(layer.o_w, null, q_cur, k_cur, v_cur, kq_mask, g.kq_scale, il, null);
            g.cb(cur, "attn_out", il);
        }

        cur = c.ggml_add(ctx0, cur, inp_l);
        inp_l = cur;

        cur = g.buildNorm(cur, layer.ln_2_w, null, norm_t, eps, il);
        g.cb(cur, "ffn_norm", il);

        cur = g.buildFfn(cur, layer.ff_up_w, null, layer.ff_gate_w, null, layer.ff_down_w, null, .silu, il);
        g.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx0, inp_l, cur);
        inp_l = cur;
    }

    cur = g.buildNorm(inp_l, model.post_ln_w, null, norm_t, eps, -1);
    g.cb(cur, "embedding_norm", -1);

    // projector: LayerNorm -> Linear -> GELU -> Linear
    cur = g.buildNorm(cur, model.mm_0_w, model.mm_0_b, .normal, 1e-5, -1);
    cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_3_w, model.mm_3_b, .gelu_erf, -1);
    g.cb(cur, "projected", -1);

    // the state outputs go first so that the embeddings stay the last node of the graph
    for (b.state_out.items) |s| {
        const name = try impl.stringFormat(arena, "state_out_%s", .{s.name.ptr});
        _ = c.ggml_set_name(s.t, name.ptr);
        c.ggml_set_output(s.t);
        c.ggml_build_forward_expand(g.gf, s.t);
    }

    c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
