//! The SEANet convolution stack of the mimi codec, shared by the pocket-tts
//! voice encoder and the mimi decoder. Tensors are T-first, `[T, C]`; the
//! convs are causal, taking their left context from a state slot, or from
//! padding on a cold start.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/pockettts-seanet.cpp` at v1.0.0
//! (`b1b770d90`), and the `mtmd_graph_pockettts_seanet` declaration in
//! `llama.cpp/tools/mtmd/models/models.h`. Each declaration names what it
//! ports and the line it began at.
//!
//! Upstream's class is a `mtmd_graph` copied from its parent whose `build`
//! aborts; here it is a struct over a `graph.Graph.initChild` of the parent.
//! `state_in` is a `std::map` read only through `at()` and `empty()`, so a
//! hash map stands in; `state_out` is a vector the caller iterates in
//! insertion order, and stays a list. The C++ methods are `const` over a
//! `mutable state_out`; here they take the builder by pointer.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// One `state_out` entry: a slot name and the tensor holding its new state.
pub const StateOut = struct { name: [:0]const u8, t: mtmd.graph.T };

/// Ports `div_ceil` (pockettts-seanet.cpp:8 @b1b770d90).
fn divCeil(a: i64, b: i64) i64 {
    return @divTrunc(a, b) + @as(i64, if (@rem(a, b) != 0) 1 else 0);
}

/// Ports `mtmd_graph_pockettts_seanet` (mtmd/models/models.h:448
/// @b1b770d90): stateless
/// unless `state_in` is populated, when the convs carry left context instead
/// of padding.
pub const Seanet = struct {
    g: mtmd.graph.Graph,
    /// owns the slot names this builder makes; must outlive `state_out`
    arena: std.mem.Allocator,
    /// per-call streaming state, keyed by slot name (see
    /// `pockettts_gen.listStateSlots`)
    state_in: std.StringHashMapUnmanaged(mtmd.graph.T) = .empty,
    state_out: std.ArrayList(StateOut) = .empty,

    /// Ports the constructor `mtmd_graph_pockettts_seanet`
    /// (mtmd/models/models.h:449 @b1b770d90): shares the parent's context and graph.
    ///
    /// Parameters:
    ///   - `parent`: the graph builder whose context this one shares.
    ///   - `arena`: allocates the state maps and the slot names; freed by the
    ///     caller after the graph is built.
    ///
    /// Return: the builder; it owns nothing the arena does not.
    pub fn init(parent: *const mtmd.graph.Graph, arena: std.mem.Allocator) Seanet {
        return .{ .g = mtmd.graph.Graph.initChild(parent), .arena = arena };
    }

    /// Ports `state_in.at(name)`; a missing key is an uncaught
    /// `std::out_of_range` upstream.
    ///
    /// Parameters:
    ///   - `name`: the slot name.
    ///
    /// Return: the slot's input tensor.
    pub fn stateAt(self: *const Seanet, name: []const u8) mtmd.graph.T {
        return self.state_in.get(name) orelse mtmd.impl.abortUncaught("std::out_of_range", "map::at:  key not found");
    }

    /// Ports `state_in[name] = t`.
    ///
    /// Parameters:
    ///   - `name`: the slot name, copied into the arena.
    ///   - `t`: the slot's input tensor.
    pub fn putState(self: *Seanet, name: []const u8, t: mtmd.graph.T) void {
        const key = self.arena.dupe(u8, name) catch oom();
        self.state_in.put(self.arena, key, t) catch oom();
    }

    /// Ports `state_out.push_back({name, t})`.
    ///
    /// Parameters:
    ///   - `name`: the slot name, copied into the arena.
    ///   - `t`: the tensor holding the slot's next state.
    pub fn pushState(self: *Seanet, name: []const u8, t: mtmd.graph.T) void {
        const owned = self.arena.dupeSentinel(u8, name, 0) catch oom();
        self.state_out.append(self.arena, .{ .name = owned, .t = t }) catch oom();
    }

    /// Ports `conv1d` (pockettts-seanet.cpp:14 @b1b770d90), a member of
    /// `mtmd_graph_pockettts_seanet`: x `[T, IC]`, w `[K, IC, OC]` to
    /// `[T / stride, OC]`. The convs are causal, so the whole `K - stride`
    /// padding goes on the left.
    ///
    /// Parameters:
    ///   - `x_in`, `w`, `b`: input, kernel, optional bias.
    ///   - `stride`, `dilation`: the conv's.
    ///   - `pad_replicate`: repeat the first frame instead of zero-padding.
    ///   - `state_name`: the slot carrying the left context, or `""`.
    ///
    /// Return: the conv's output, in the graph's context.
    pub fn conv1d(self: *Seanet, x_in: mtmd.graph.T, w: mtmd.graph.T, b: mtmd.graph.T, stride: c_int, dilation: c_int, pad_replicate: bool, state_name: []const u8) mtmd.graph.T {
        const ctx0 = self.g.ctx0;
        var x = x_in;
        const k_size = (mtmd.graph.ne(w, 0) - 1) * dilation + 1;
        const p_total = k_size - stride;

        // trailing padding so the last frame is not dropped, see pad_for_conv1d() in conv.py
        const n_frames = divCeil(mtmd.graph.ne(x, 0) - k_size + p_total, stride);
        const ideal_len = n_frames * stride + k_size - p_total;
        const p_extra = ideal_len - mtmd.graph.ne(x, 0);

        if (state_name.len != 0 and p_total > 0) {
            // streaming: the left context is the tail of the previous call
            const left = self.stateAt(state_name); // [p_total, IC]
            x = llama.impl.llama_c.ggml_concat(ctx0, left, x, 0);
            self.pushState(state_name, llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, x, p_total, mtmd.graph.ne(x, 1), mtmd.graph.nb(x, 1), @as(usize, @intCast(mtmd.graph.ne(x, 0) - p_total)) * mtmd.graph.nb(x, 0))));
        } else if (pad_replicate and p_total > 0) {
            // the resamplers repeat the first frame instead of zero-padding
            const first = llama.impl.llama_c.ggml_view_2d(ctx0, x, 1, mtmd.graph.ne(x, 1), mtmd.graph.nb(x, 1), 0);
            const left = llama.impl.llama_c.ggml_repeat_4d(ctx0, first, p_total, mtmd.graph.ne(x, 1), 1, 1);
            x = llama.impl.llama_c.ggml_concat(ctx0, left, x, 0);
            x = llama.impl.llama_c.ggml_pad_ext(ctx0, x, 0, @intCast(p_extra), 0, 0, 0, 0, 0, 0);
        } else {
            x = llama.impl.llama_c.ggml_pad_ext(ctx0, x, @intCast(p_total), @intCast(p_extra), 0, 0, 0, 0, 0, 0);
        }

        var y = llama.impl.llama_c.ggml_conv_1d(ctx0, w, x, stride, 0, dilation);
        y = llama.impl.llama_c.ggml_reshape_2d(ctx0, y, mtmd.graph.ne(y, 0), mtmd.graph.ne(y, 1));
        if (b != null) {
            y = llama.impl.llama_c.ggml_add(ctx0, y, llama.impl.llama_c.ggml_reshape_2d(ctx0, b, 1, mtmd.graph.ne(b, 0)));
        }
        return y;
    }

    /// Ports `conv_transpose1d` (pockettts-seanet.cpp:51 @b1b770d90), a
    /// member of `mtmd_graph_pockettts_seanet`: x `[T, IC]`, w `[K, OC/groups,
    /// IC]` to `[T * stride, OC]`. The `K - stride` overlap tail belongs to
    /// the next call: added to its head when streaming, else dropped.
    ///
    /// Parameters:
    ///   - `x`, `w`, `b`: input, kernel, optional bias.
    ///   - `stride`: the conv's.
    ///   - `state_name`: the slot carrying the overlap tail, or `""`.
    ///
    /// Return: the conv's output, in the graph's context.
    pub fn convTranspose1d(self: *Seanet, x: mtmd.graph.T, w: mtmd.graph.T, b: mtmd.graph.T, stride: c_int, state_name: []const u8) mtmd.graph.T {
        const ctx0 = self.g.ctx0;
        const K = mtmd.graph.ne(w, 0);
        const Tn = mtmd.graph.ne(x, 0);
        const p_total = K - stride;
        const depthwise = mtmd.graph.ne(w, 1) == 1 and mtmd.graph.ne(w, 2) > 1;
        const OC = if (depthwise) mtmd.graph.ne(w, 2) else mtmd.graph.ne(w, 1);
        const emit_len = Tn * stride;

        // one column per input step, holding the [K, OC] window that col2im scatter-adds at t * stride
        var col: mtmd.graph.T = undefined;
        if (depthwise) {
            // one group per channel: a batched matmul over the channels scales the kernel by each step
            const krn = llama.impl.llama_c.ggml_reshape_3d(ctx0, w, 1, K, OC); // [1, K, OC]
            const xs = llama.impl.llama_c.ggml_reshape_3d(ctx0, x, 1, Tn, OC); // [1, T, OC]
            col = llama.impl.llama_c.ggml_mul_mat(ctx0, krn, xs); // [K, T, OC]
            col = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_permute(ctx0, col, 0, 2, 1, 3)); // [K, OC, T]
            col = llama.impl.llama_c.ggml_reshape_2d(ctx0, col, K * OC, Tn);
        } else {
            var w2 = llama.impl.llama_c.ggml_reshape_2d(ctx0, w, K * OC, mtmd.graph.ne(w, 2));
            w2 = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, w2)); // [IC, K * OC]
            const xt = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, x)); // [IC, T]
            col = llama.impl.llama_c.ggml_mul_mat(ctx0, w2, xt);
        }
        const full = llama.impl.llama_c.ggml_col2im_1d(ctx0, col, stride, @intCast(OC), 0); // [emit_len + p_total, OC]

        var out: mtmd.graph.T = undefined;
        if (state_name.len == 0 or p_total == 0) {
            out = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, full, emit_len, mtmd.graph.ne(full, 1), mtmd.graph.nb(full, 1), 0));
        } else {
            // overlap-add the tail the previous call held back
            const prev = self.stateAt(state_name); // [p_total, OC]
            const head = llama.impl.llama_c.ggml_add(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, full, p_total, mtmd.graph.ne(full, 1), mtmd.graph.nb(full, 1), 0), prev);
            if (emit_len > p_total) {
                const rest = llama.impl.llama_c.ggml_view_2d(ctx0, full, emit_len - p_total, mtmd.graph.ne(full, 1), mtmd.graph.nb(full, 1), @as(usize, @intCast(p_total)) * mtmd.graph.nb(full, 0));
                out = llama.impl.llama_c.ggml_concat(ctx0, head, rest, 0);
            } else {
                out = head;
            }
            self.pushState(state_name, llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, full, p_total, mtmd.graph.ne(full, 1), mtmd.graph.nb(full, 1), @as(usize, @intCast(emit_len)) * mtmd.graph.nb(full, 0))));
        }

        if (b != null) {
            out = llama.impl.llama_c.ggml_add(ctx0, out, llama.impl.llama_c.ggml_reshape_2d(ctx0, b, 1, mtmd.graph.ne(b, 0)));
        }
        return out;
    }

    /// Ports `res_unit` (pockettts-seanet.cpp:102 @b1b770d90), a member of
    /// `mtmd_graph_pockettts_seanet`.
    ///
    /// Parameters:
    ///   - `x`: input, `[T, C]`.
    ///   - `stage`: the SEANet stage whose residual convs to apply.
    ///   - `dilation`: the first conv's.
    ///   - `state_prefix`: the first conv's state slot, or `""`.
    ///
    /// Return: `x` plus the residual branch.
    pub fn resUnit(self: *Seanet, x: mtmd.graph.T, stage: *const mtmd.model.Seanet.Stage, dilation: c_int, state_prefix: []const u8) mtmd.graph.T {
        const ctx0 = self.g.ctx0;
        var h = llama.impl.llama_c.ggml_elu(ctx0, x);
        h = self.conv1d(h, stage.res_conv1_w, stage.res_conv1_b, 1, dilation, false, state_prefix);
        h = llama.impl.llama_c.ggml_elu(ctx0, h);
        // the second conv is pointwise, it needs no left context
        h = self.conv1d(h, stage.res_conv2_w, stage.res_conv2_b, 1, 1, false, "");
        return llama.impl.llama_c.ggml_add(ctx0, x, h);
    }

    /// Ports `encode` (pockettts-seanet.cpp:112 @b1b770d90), a member of
    /// `mtmd_graph_pockettts_seanet`: x `[T, C]` to `[T / hop, dim]`.
    ///
    /// Parameters:
    ///   - `x`: the waveform, `[T, C]`.
    ///
    /// Return: the encoded frames.
    pub fn encode(self: *Seanet, x: mtmd.graph.T) mtmd.graph.T {
        const ctx0 = self.g.ctx0;
        const hparams = self.g.hparams;
        const seanet = &self.g.model.seanet;

        var cur = self.conv1d(x, seanet.conv_in_w, seanet.conv_in_b, 1, 1, false, "");
        self.g.cb(cur, "seanet_enc_in", -1);

        var i: c_int = 0;
        while (i < hparams.seanet_n_stage) : (i += 1) {
            const stage = &seanet.stages.items[@intCast(i)];
            const stride = hparams.seanet_ratios.items[@intCast(i)];

            cur = self.resUnit(cur, stage, 1, "");
            cur = llama.impl.llama_c.ggml_elu(ctx0, cur);
            cur = self.conv1d(cur, stage.scale_conv_w, stage.scale_conv_b, stride, 1, false, "");
            self.g.cb(cur, "seanet_enc_stage", i);
        }

        cur = llama.impl.llama_c.ggml_elu(ctx0, cur);
        cur = self.conv1d(cur, seanet.conv_out_w, seanet.conv_out_b, 1, 1, false, "");
        self.g.cb(cur, "seanet_enc_out", -1);

        return cur;
    }

    /// Ports `decode` (pockettts-seanet.cpp:135 @b1b770d90), a member of
    /// `mtmd_graph_pockettts_seanet`: x `[T, dim]` to `[T * hop, 1]`,
    /// streaming when `state_in` is populated.
    ///
    /// Parameters:
    ///   - `x`: the frames, `[T, dim]`.
    ///
    /// Return: the waveform.
    pub fn decode(self: *Seanet, x: mtmd.graph.T) mtmd.graph.T {
        const ctx0 = self.g.ctx0;
        const hparams = self.g.hparams;
        const seanet = &self.g.model.seanet;
        const stream = self.state_in.count() != 0;

        var cur = self.conv1d(x, seanet.conv_in_w, seanet.conv_in_b, 1, 1, false, if (stream) "dec_in" else "");
        self.g.cb(cur, "seanet_dec_in", -1);

        var i: c_int = 0;
        while (i < hparams.seanet_n_stage) : (i += 1) {
            const stage = &seanet.stages.items[@intCast(i)];
            // the decoder mirrors the encoder, so the ratios are walked backwards
            const stride = hparams.seanet_ratios.items[@intCast(hparams.seanet_n_stage - 1 - i)];

            cur = llama.impl.llama_c.ggml_elu(ctx0, cur);
            const up_name = if (stream) self.idName("dec_up_", i) else "";
            cur = self.convTranspose1d(cur, stage.scale_conv_w, stage.scale_conv_b, stride, up_name);
            const res_name = if (stream) self.idName("dec_res_", i) else "";
            cur = self.resUnit(cur, stage, 1, res_name);
            self.g.cb(cur, "seanet_dec_stage", i);
        }

        cur = llama.impl.llama_c.ggml_elu(ctx0, cur);
        cur = self.conv1d(cur, seanet.conv_out_w, seanet.conv_out_b, 1, 1, false, if (stream) "dec_out" else "");
        self.g.cb(cur, "seanet_dec_out", -1);

        return cur;
    }

    /// `prefix + std::to_string(i)`, in the arena.
    fn idName(self: *const Seanet, comptime prefix: []const u8, i: c_int) []const u8 {
        return std.fmt.allocPrint(self.arena, prefix ++ "{d}", .{i}) catch oom();
    }
};

fn oom() noreturn {
    mtmd.impl.abort("pockettts-seanet.cpp", 0, "out of memory");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "divCeil truncates like C" {
    try std.testing.expectEqual(@as(i64, 3), divCeil(7, 3));
    try std.testing.expectEqual(@as(i64, 2), divCeil(6, 3));
    try std.testing.expectEqual(@as(i64, 1), divCeil(-2, 3)); // C: -2/3 = 0, -2%3 = -2 -> 0 + 1
}
