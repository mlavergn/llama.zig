//! The gated delta net: linear attention with a per-head state, shared by
//! the architectures built on it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/delta-net-base.cpp` at v1.0.0
//! (`b1b770d90`): every member of `llm_build_delta_net_base`. Also from `llama.cpp/src/models/models.h`:
//! `build_gdn_l2_norm`.
//!
//! The C++ makes these members of a subclass of `llm_graph_context`; here
//! they are functions over the builder.

const std = @import("std");
const impl = @import("../impl.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const Builder = graph.Builder;

/// The output and the new state.
pub const OutState = struct { out: *Tensor, state: *Tensor };

/// Ports `build_gdn_l2_norm` (src/models/models.h:14 @b1b770d90): the L2 norm the
/// gated delta nets apply to `q` and `k`, as an RMS norm over `eps / n`
/// scaled by `1 / sqrt(n)` (upstream PR 28068), `n` the row length.
pub fn buildGdnL2Norm(ctx: *c.ggml_context, x: *Tensor, eps: f32) *Tensor {
    const n: f32 = @floatFromInt(x.ne[0]);

    return c.ggml_scale(ctx, c.ggml_rms_norm(ctx, x, eps / n), 1.0 / @sqrt(n));
}

/// The shape checks of `build_delta_net`, whose first `GGML_ASSERT` is on
/// `line` of delta-net-base.cpp.
fn checkShapes(q: *const Tensor, k: *const Tensor, v: *const Tensor, g: *const Tensor, b: *const Tensor, s: *const Tensor, comptime line: c_int) void {
    const S_k = q.ne[0];
    const H_k = q.ne[1];
    const n_tokens = q.ne[2];
    const n_seqs = q.ne[3];
    const S_v = v.ne[0];
    const H_v = v.ne[1];
    impl.assert(S_k == S_v, "delta-net-base.cpp", line, "S_k == S_v");
    impl.assert(@rem(H_v, H_k) == 0, "delta-net-base.cpp", line + 1, "H_v % H_k == 0");
    impl.assert(q.ne[0] == S_k and q.ne[1] == H_k and q.ne[2] == n_tokens and q.ne[3] == n_seqs, "delta-net-base.cpp", line + 3, "q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs");
    impl.assert(k.ne[0] == S_k and k.ne[1] == H_k and k.ne[2] == n_tokens and k.ne[3] == n_seqs, "delta-net-base.cpp", line + 4, "k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs");
    impl.assert(v.ne[0] == S_v and v.ne[1] == H_v and v.ne[2] == n_tokens and v.ne[3] == n_seqs, "delta-net-base.cpp", line + 5, "v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs");
    impl.assert(g.ne[0] == 1 or g.ne[0] == S_v, "delta-net-base.cpp", line + 7, "g->ne[0] == 1   || g->ne[0] == S_v");
    impl.assert(g.ne[1] == H_v and g.ne[2] == n_tokens and g.ne[3] == n_seqs, "delta-net-base.cpp", line + 8, "g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs");
    impl.assert(b.ne[0] == 1 and b.ne[1] == H_v and b.ne[2] == n_tokens and b.ne[3] == n_seqs, "delta-net-base.cpp", line + 9, "b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs");
    impl.assert(s.ne[0] == S_v and s.ne[1] == S_v and s.ne[2] == H_v and s.ne[3] == n_seqs, "delta-net-base.cpp", line + 10, "s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs");
}

/// Ports `llm_build_delta_net_base`'s `build_delta_net` (delta-net-base.cpp:8
/// @b1b770d90): the fused operator with K = 1, whose output carries the
/// final state only. Since upstream ccd360f06 there is no other path.
///
/// Parameters:
/// - `self`: the graph builder.
/// - `q`, `k`, `v`, `g`, `b`: the delta net's inputs, `[S, H, n_tokens, n_seqs]`.
/// - `s`: the state, `[S_v, S_v, H_v, n_seqs]`.
///
/// Return: the output and the new state, both views of one result.
pub fn buildDeltaNet(self: *const Builder, q: *Tensor, k: *Tensor, v: *Tensor, g: *Tensor, b: *Tensor, s: *Tensor) OutState {
    const ctx = self.ctx0;
    const n_tokens = q.ne[2];
    const n_seqs = q.ne[3];
    const S_v = v.ne[0];
    const H_v = v.ne[1];

    checkShapes(q, k, v, g, b, s, 21);

    // K = 1: the output carries the final state only.
    const result = c.ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1);
    const output = c.ggml_view_4d(ctx, result, S_v, H_v, n_tokens, n_seqs, c.ggml_row_size(result.*.type, S_v), c.ggml_row_size(result.*.type, S_v * H_v), c.ggml_row_size(result.*.type, S_v * H_v * n_tokens), 0);
    const new_state = c.ggml_view_4d(ctx, result, S_v, S_v, H_v, n_seqs, c.ggml_row_size(result.*.type, S_v), c.ggml_row_size(result.*.type, S_v * S_v), c.ggml_row_size(result.*.type, S_v * S_v * H_v), c.ggml_row_size(result.*.type, S_v * H_v * n_tokens * n_seqs));

    return .{ .out = output, .state = new_state };
}

/// Ports `llm_build_delta_net_base`'s `build_conv_state` (delta-net-base.cpp:53
/// @b1b770d90): the convolution's input with the saved state in front, and
/// the state written back through `build_conv_state_store`.
pub fn buildConvState(self: *const Builder, inp: *const graph.InpRs, conv_states_all: *Tensor, qkv_mixed_: *Tensor, conv_kernel_size: i64, conv_channels: i64, il: c_int) *Tensor {
    const ctx = self.ctx0;
    const n_seqs: i64 = self.ubatch.n_seqs;

    var conv_states = self.buildRs(inp, conv_states_all, @intCast(self.hparams.nEmbdR()), @intCast(n_seqs), null);
    self.cb(conv_states, "conv_states", il);

    conv_states = c.ggml_reshape_3d(ctx, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    self.cb(conv_states, "conv_states_reshaped", il);

    const qkv_mixed = c.ggml_transpose(ctx, qkv_mixed_);
    self.cb(qkv_mixed, "qkv_mixed_transposed", il);

    const conv_input = c.ggml_concat(ctx, conv_states, qkv_mixed, 0);
    self.cb(conv_input, "conv_input", il);

    self.buildConvStateStore(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0);

    return conv_input;
}

/// Ports `llm_build_delta_net_base`'s `build_recurrent_attn` (delta-net-base.cpp:79
/// @b1b770d90): the delta net over the saved state, with the new state, or
/// its rollback snapshots, written back.
pub fn buildRecurrentAttn(self: *const Builder, inp: *const graph.InpRs, ssm_states_all: *Tensor, q: *Tensor, k: *Tensor, v: *Tensor, g: *Tensor, b: *Tensor, s: *Tensor, il: c_int) *Tensor {
    const ctx = self.ctx0;
    const mctx_cur = inp.mctx;
    const kv_head: usize = mctx_cur.getHead();
    const mem_size: usize = mctx_cur.getSize();

    const S_v = s.ne[0];
    const H_v = s.ne[2];
    const n_seqs = s.ne[3];
    const n_seq_tokens = q.ne[2];

    const keep = self.cparams.n_rs_seq > 0;
    const n_embd_s: i64 = self.hparams.nEmbdS();

    if (!keep) {
        const attn_out = buildDeltaNet(self, q, k, v, g, b, s);
        self.cb(attn_out.out, "attn_output", il);
        self.cb(attn_out.state, "new_state", il);
        c.ggml_build_forward_expand(self.gf, c.ggml_cpy(ctx, attn_out.state, c.ggml_view_2d(ctx, ssm_states_all, n_embd_s, n_seqs, ssm_states_all.nb[1], kv_head * @as(usize, @intCast(n_embd_s)) * c.ggml_element_size(ssm_states_all))));
        return attn_out.out;
    }

    const D = S_v * S_v * H_v;
    const K: i64 = @as(i64, self.cparams.n_rs_seq) + 1;

    // K snapshot slots are written into the output.
    const gdn_out = c.ggml_gated_delta_net(ctx, q, k, v, g, b, s, K);

    const attn_score_elems = S_v * H_v * n_seq_tokens * n_seqs;
    const state_size_per_snap = S_v * S_v * H_v * n_seqs;

    const output = c.ggml_view_4d(ctx, gdn_out, S_v, H_v, n_seq_tokens, n_seqs, c.ggml_row_size(gdn_out.*.type, S_v), c.ggml_row_size(gdn_out.*.type, S_v * H_v), c.ggml_row_size(gdn_out.*.type, S_v * H_v * n_seq_tokens), 0);
    self.cb(output, "attn_output", il);

    const row_size: usize = @as(usize, @intCast(n_embd_s)) * c.ggml_element_size(ssm_states_all);

    // The op writes the last min(n_seq_tokens, K) snapshots.
    const n_written = @min(n_seq_tokens, K);

    // Snapshot slot i goes to rollback group i.
    const src = c.ggml_view_3d(ctx, gdn_out, D, n_seqs, n_written, c.ggml_row_size(gdn_out.*.type, D), c.ggml_row_size(gdn_out.*.type, state_size_per_snap), c.ggml_row_size(gdn_out.*.type, attn_score_elems));
    const dst = c.ggml_view_3d(ctx, ssm_states_all, D, n_seqs, n_written, ssm_states_all.nb[1], mem_size * row_size, kv_head * row_size);
    c.ggml_build_forward_expand(self.gf, c.ggml_cpy(ctx, src, dst));

    return output;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
