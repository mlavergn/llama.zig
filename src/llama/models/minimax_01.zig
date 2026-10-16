//! MiniMax-Text-01: a mixture-of-experts model whose layers are mostly
//! lightning attention -- linear attention with a per-head exponential
//! decay, kept as a recurrent state -- with a softmax attention layer every
//! few layers.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/minimax-01.cpp` at v0.3.0
//! (`c1d0e7a00`): `llama_model_minimax_01`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, its `graph`, and the
//! `llm_graph_input_la` input that file defines for the decays.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const batch = @import("../batch.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Hparams = hparams_mod.Hparams;
const Ubatch = batch.Ubatch;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_minimax_01`'s `load_arch_hparams` (models/minimax-01.cpp:4 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_residual_scale = (try meta.getKey(f32, .residual_scale, true, exc)).?;

    // n_embd_head_la sets the recurrent memory's n_embd_s.
    hp.n_embd_head_la = hp.n_embd_head_k_full;

    // The recurrent (lightning attention) layers.
    if (!try meta.getKeyOrArr(u32, .attention_recurrent_layers, &hp.is_recr_impl, hp.n_layer_all, false, exc)) {
        var full_attn_interval: u32 = 8;
        if (try meta.getKey(u32, .full_attention_interval, false, exc)) |v| full_attn_interval = v;
        var i: u32 = 0;
        while (i < hp.n_layer_all) : (i += 1) {
            hp.is_recr_impl[i] = @intFromBool(i < hp.nLayer() and model_mod.remU32(i + 1, full_attn_interval) != 0);
        }
    }

    self.type = switch (hp.nLayer()) {
        80 => .@"456b",
        else => .unknown,
    };
}

/// Ports `llama_model_minimax_01`'s `load_arch_tensors` (models/minimax-01.cpp:26 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (!hp.isRecr(@intCast(i))) {
            try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        } else {
            layer.attn_norm_2 = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k * l.n_head}, 0, exc);
            layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, 3 * l.n_embd_head_k * l.n_head }, 0, exc);
            layer.wg = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd_head_k * l.n_head }, 0, exc);
        }
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, nr, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd, l.n_expert }, 0, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, 0, exc);
    }
}

/// Ports `llama_model_minimax_01`'s `build_arch_graph` (models/minimax-01.cpp:63 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports `llm_graph_input_la` (models/minimax-01.cpp:67 @c1d0e7a00): the
/// lightning attention's per-head slopes and, for a batch of more than one
/// token per sequence, its decays relative to each sequence's first
/// position.
pub const InpLa = struct {
    hparams: *const Hparams,

    /// F32 `[n_head]`.
    inp_slopes: ?*Tensor = null,
    /// F32 `[1, n_head, n_batch]`.
    inp_q_decay: ?*Tensor = null,
    /// F32 `[1, n_head, n_batch]`.
    inp_k_decay: ?*Tensor = null,
    /// F32 `[n_batch, n_batch, n_head]`.
    inp_diag_decay: ?*Tensor = null,

    /// Ports `llm_graph_input_la`'s `set_input` (models/minimax-01.cpp:71
    /// @c1d0e7a00). It assumes an equal ubatch split.
    pub fn setInput(self: *InpLa, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const n_head: i64 = self.hparams.nHead(0);
        const n_seqs: i32 = @intCast(ubatch.n_seqs);
        const n_seqs_unq: i32 = @intCast(ubatch.n_seqs_unq);
        const n_tokens: i32 = @intCast(ubatch.n_tokens);
        const n_seq_tokens: i32 = @intCast(ubatch.n_seq_tokens);

        const p0 = allocator.alloc(i32, @intCast(n_seqs_unq)) catch impl.abort("models/minimax-01.cpp", 80, "out of memory");
        defer allocator.free(p0);
        @memset(p0, std.math.maxInt(i32));

        // The lowest token position in the ubatch for each stream.
        var i: i32 = 0;
        while (i < n_tokens) : (i += 1) {
            const seq_id = ubatch.seq_id[@intCast(i)][0];
            const seq_idx: usize = @intCast(ubatch.seq_idx[@intCast(seq_id)]);
            const pos = ubatch.pos[@intCast(i)];
            if (p0[seq_idx] > pos) p0[seq_idx] = pos;
        }

        if (self.inp_slopes) |inp_slopes| {
            impl.assert(c.ggml_backend_buffer_is_host(inp_slopes.buffer), "models/minimax-01.cpp", 94, "ggml_backend_buffer_is_host(inp_slopes->buffer)");

            const data: [*]f32 = @ptrCast(@alignCast(inp_slopes.data));

            const start: f32 = impl.powf(2, -impl.powf(2, -(impl.log2f(@floatFromInt(n_head)) - 3)));
            const ratio = start;

            var h: i64 = 0;
            while (h < n_head) : (h += 1) {
                data[@intCast(h)] = start * impl.powf(ratio, @floatFromInt(h));
            }
        }

        if (self.inp_q_decay) |inp_q_decay| {
            impl.assert(c.ggml_backend_buffer_is_host(inp_q_decay.buffer), "models/minimax-01.cpp", 107, "ggml_backend_buffer_is_host(inp_q_decay->buffer)");

            const slopes: [*]const f32 = @ptrCast(@alignCast(self.inp_slopes.?.data));
            const data: [*]f32 = @ptrCast(@alignCast(inp_q_decay.data));

            var s: i32 = 0;
            while (s < n_seqs) : (s += 1) {
                i = 0;
                while (i < n_seq_tokens) : (i += 1) {
                    const t: usize = @intCast(s * n_seq_tokens + i);
                    const seq_id = ubatch.seq_id[t][0];
                    const seq_idx: i64 = ubatch.seq_idx[@intCast(seq_id)];
                    const pos = ubatch.pos[t];
                    const pos_rel: c_int = pos - p0[@intCast(seq_idx)];

                    var h: i64 = 0;
                    while (h < n_head) : (h += 1) {
                        data[@intCast(seq_idx * n_head * n_seq_tokens + i * n_head + h)] = -slopes[@intCast(h)] * @as(f32, @floatFromInt(pos_rel + 1));
                    }
                }
            }
        }

        if (self.inp_k_decay) |inp_k_decay| {
            impl.assert(c.ggml_backend_buffer_is_host(inp_k_decay.buffer), "models/minimax-01.cpp", 127, "ggml_backend_buffer_is_host(inp_k_decay->buffer)");

            const slopes: [*]const f32 = @ptrCast(@alignCast(self.inp_slopes.?.data));
            const data: [*]f32 = @ptrCast(@alignCast(inp_k_decay.data));

            var s: i32 = 0;
            while (s < n_seqs) : (s += 1) {
                i = 0;
                while (i < n_seq_tokens) : (i += 1) {
                    const t: usize = @intCast(s * n_seq_tokens + i);
                    const seq_id = ubatch.seq_id[t][0];
                    const seq_idx: i64 = ubatch.seq_idx[@intCast(seq_id)];
                    const pos = ubatch.pos[t];
                    const pos_rel: c_int = pos - p0[@intCast(seq_idx)];

                    var h: i64 = 0;
                    while (h < n_head) : (h += 1) {
                        data[@intCast(seq_idx * n_head * n_seq_tokens + i * n_head + h)] = -slopes[@intCast(h)] * @as(f32, @floatFromInt(n_seq_tokens - pos_rel - 1));
                    }
                }
            }
        }

        if (self.inp_diag_decay) |inp_diag_decay| {
            impl.assert(c.ggml_backend_buffer_is_host(inp_diag_decay.buffer), "models/minimax-01.cpp", 147, "ggml_backend_buffer_is_host(inp_diag_decay->buffer)");

            const slopes: [*]const f32 = @ptrCast(@alignCast(self.inp_slopes.?.data));
            const data: [*]f32 = @ptrCast(@alignCast(inp_diag_decay.data));
            const T: i64 = n_seq_tokens;

            var s: i32 = 0;
            while (s < n_seqs) : (s += 1) {
                var h: i64 = 0;
                while (h < n_head) : (h += 1) {
                    var j: i32 = 0;
                    while (j < n_seq_tokens) : (j += 1) {
                        const tj: usize = @intCast(s * n_seq_tokens + j);
                        const seq_id = ubatch.seq_id[tj][0];
                        const seq_idx: i64 = ubatch.seq_idx[@intCast(seq_id)];
                        const pos_j = ubatch.pos[tj];
                        const pos_rel_j: c_int = pos_j - p0[@intCast(seq_idx)];

                        i = 0;
                        while (i < n_seq_tokens) : (i += 1) {
                            const pos_i = ubatch.pos[@intCast(s * n_seq_tokens + i)];
                            const pos_rel_i: c_int = pos_i - p0[@intCast(seq_idx)];

                            const index: c_int = pos_rel_j - pos_rel_i;
                            const s_index: f32 = if (index >= 0) -slopes[@intCast(h)] * @as(f32, @floatFromInt(index)) else -std.math.inf(f32);
                            data[@intCast(seq_idx * n_head * T * T + h * T * T + j * T + i)] = s_index;
                        }
                    }
                }
            }
        }
    }

    /// Ports `llm_graph_input_la`'s `can_reuse` (models/minimax-01.cpp:174
    /// @c1d0e7a00): only the decays' token counts are checked, and only
    /// when the new ubatch needs them.
    pub fn canReuse(self: *InpLa, params: *const graph.Params) bool {
        var res = true;

        const n_seq_tokens: i64 = params.ubatch.n_seq_tokens;
        if (n_seq_tokens > 1) {
            res = res and (self.inp_q_decay != null and self.inp_q_decay.?.ne[2] == n_seq_tokens);
            res = res and (self.inp_k_decay != null and self.inp_k_decay.?.ne[2] == n_seq_tokens);
            res = res and (self.inp_diag_decay != null and self.inp_diag_decay.?.ne[1] == n_seq_tokens);
        }

        return res;
    }
};

/// Ports the constructor of `llama_model_minimax_01`'s `graph`
/// (models/minimax-01.cpp:194 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/minimax-01.cpp", 197, "n_embd_head == hparams.n_embd_head_k()");
    // The C++ does not assert n_embd_head == n_rot: MiniMax's head_dim is
    // 128 and n_rot 64.

    const n_seqs: i64 = b.ubatch.n_seqs;
    const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;

    impl.assert(n_seqs != 0, "models/minimax-01.cpp", 203, "n_seqs != 0");
    impl.assert(b.ubatch.equalSeqs(), "models/minimax-01.cpp", 204, "ubatch.equal_seqs()");
    impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/minimax-01.cpp", 205, "ubatch.n_tokens == n_seq_tokens * n_seqs");

    const n_head = b.n_head;

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_hybrid = b.buildInpMemHybrid();
    const inp_rs = inp_hybrid.getRecr();

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    const inp = allocator.create(InpLa) catch impl.abort("models/minimax-01.cpp", 220, "out of memory");
    inp.* = .{ .hparams = hp };

    inp.inp_slopes = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, n_head);
    c.ggml_set_input(inp.inp_slopes);
    b.cb(inp.inp_slopes.?, "slopes", -1);

    if (n_seq_tokens != 1) {
        inp.inp_q_decay = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, 1, n_head, n_seq_tokens, n_seqs);
        c.ggml_set_input(inp.inp_q_decay);
        b.cb(inp.inp_q_decay.?, "q_decay_exp", -1);

        inp.inp_k_decay = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, 1, n_head, n_seq_tokens, n_seqs);
        c.ggml_set_input(inp.inp_k_decay);
        b.cb(inp.inp_k_decay.?, "k_decay_exp", -1);

        inp.inp_diag_decay = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, n_seq_tokens, n_seq_tokens, n_head, n_seqs);
        c.ggml_set_input(inp.inp_diag_decay);
        b.cb(inp.inp_diag_decay.?, "diag_decay_exp", -1);
    }

    const la = b.res.addInput(InpLa, inp);

    const slopes = la.inp_slopes.?;

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = inpL;

        var inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        var residual = cur;

        // Self-attention.
        if (!hp.isRecr(@intCast(il))) {
            // A softmax attention layer.
            const qkv = b.buildQkv(layer, cur, n_embd_head, n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_hybrid.getAttn(), layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
        } else {
            // A lightning attention layer.
            const mctx_cur = inp_rs.mctx;
            const kv_head: usize = mctx_cur.getHead();

            // The C++'s TODO: the conv states are unneeded, but the
            // recurrent memory has no way to make them optional.
            const conv_states_all = mctx_cur.getRL(il);
            const conv_state_all = b.buildRs(inp_rs, conv_states_all, @intCast(hp.nEmbdR()), @intCast(n_seqs), null);
            c.ggml_build_forward_expand(b.gf, conv_state_all);

            const n_layer_f: f64 = @floatFromInt(b.n_layer - 1);
            const slope_scale: f32 = @floatCast(1.0 - 1.0 * @as(f64, @floatFromInt(il)) / n_layer_f + 1e-5);
            const slope_rate: *Tensor = c.ggml_scale(ctx, slopes, slope_scale);
            b.cb(slope_rate, "slope_rate", il);

            cur = c.ggml_reshape_4d(ctx, cur, cur.ne[0], n_seq_tokens, 1, n_seqs);

            var QKVcur = b.buildLoraMm(layer.wqkv.?, cur, null);
            b.cb(QKVcur, "QKVcur", il);

            QKVcur = c.ggml_silu(ctx, QKVcur);
            b.cb(QKVcur, "QKVcur_silu", il);

            QKVcur = c.ggml_reshape_4d(ctx, QKVcur, n_embd_head * 3, n_head, n_seq_tokens, n_seqs);

            const es = c.ggml_element_size(QKVcur);
            const head_bytes: usize = @intCast(n_embd_head);
            const Qcur: *Tensor = c.ggml_view_4d(ctx, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur.nb[1], QKVcur.nb[2], QKVcur.nb[3], 0 * es * head_bytes);
            const Kcur: *Tensor = c.ggml_view_4d(ctx, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur.nb[1], QKVcur.nb[2], QKVcur.nb[3], 1 * es * head_bytes);
            const Vcur: *Tensor = c.ggml_view_4d(ctx, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur.nb[1], QKVcur.nb[2], QKVcur.nb[3], 2 * es * head_bytes);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            // The previous KV.
            const la_states_all = mctx_cur.getSL(il);
            const n_embd_s: usize = hp.nEmbdS();
            const state = b.buildRs(inp_rs, la_states_all, @intCast(n_embd_s), @intCast(n_seqs), null);

            const kv_old: *Tensor = c.ggml_reshape_4d(ctx, state, n_embd_head, n_embd_head, n_head, n_seqs);
            b.cb(kv_old, "kv_old", il);

            var qkv: ?*Tensor = null;
            var kv_new: ?*Tensor = null;

            if (n_seq_tokens == 1) {
                // Lightning attention, the single-token case for generation.
                const slopes_neg: *Tensor = c.ggml_scale(ctx, slope_rate, -1.0);
                b.cb(slopes_neg, "slopes_neg", il);

                const ratio: *Tensor = c.ggml_exp(ctx, slopes_neg);
                b.cb(ratio, "ratio", il);

                const ratio_3d: *Tensor = c.ggml_reshape_3d(ctx, ratio, 1, 1, n_head);
                b.cb(ratio_3d, "ratio3d", il);

                const v_trans: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, Vcur, 1, 2, 0, 3));
                b.cb(v_trans, "v_trans", il);

                const k_trans: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, Kcur, 1, 2, 0, 3));
                b.cb(k_trans, "k_trans", il);

                const kv_cur: *Tensor = c.ggml_mul_mat(ctx, k_trans, v_trans);
                b.cb(kv_cur, "kv_cur", il);

                const kv_old_s: *Tensor = c.ggml_mul(ctx, kv_old, ratio_3d);
                b.cb(kv_old_s, "kv_old_s", il);

                kv_new = c.ggml_add(ctx, kv_old_s, kv_cur);
                b.cb(kv_new.?, "kv_new", il);

                const q_trans: *Tensor = c.ggml_permute(ctx, Qcur, 0, 2, 1, 3);
                b.cb(q_trans, "q_trans", il);

                qkv = c.ggml_mul_mat(ctx, kv_new, q_trans);
                b.cb(qkv.?, "qkv", il);
            } else if (n_seq_tokens > 1) {
                // Lightning attention, the general multi-token case for
                // prompt processing.
                const q_decay_exp = la.inp_q_decay;
                const k_decay_exp = la.inp_k_decay;
                const diag_decay_exp = la.inp_diag_decay;

                const q_decay: *Tensor = c.ggml_exp(ctx, c.ggml_scale(ctx, q_decay_exp, slope_scale));
                b.cb(q_decay, "q_decay", il);
                const k_decay: *Tensor = c.ggml_exp(ctx, c.ggml_scale(ctx, k_decay_exp, slope_scale));
                b.cb(k_decay, "k_decay", il);
                const diag_decay: *Tensor = c.ggml_exp(ctx, c.ggml_scale(ctx, diag_decay_exp, slope_scale));
                b.cb(diag_decay, "diag_decay", il);

                const q_s: *Tensor = c.ggml_mul(ctx, Qcur, q_decay);
                b.cb(q_s, "q_s", il);

                const q_s_trans: *Tensor = c.ggml_permute(ctx, q_s, 0, 2, 1, 3);
                b.cb(q_s_trans, "q_s_trans", il);

                const qkv_none_diag: *Tensor = c.ggml_mul_mat(ctx, kv_old, q_s_trans);
                b.cb(qkv_none_diag, "qkv_none_diag", il);

                const q_trans: *Tensor = c.ggml_permute(ctx, Qcur, 0, 2, 1, 3);
                b.cb(q_trans, "q_trans", il);

                const k_trans: *Tensor = c.ggml_permute(ctx, Kcur, 0, 2, 1, 3);
                b.cb(k_trans, "k_trans", il);

                var qk: *Tensor = c.ggml_mul_mat(ctx, k_trans, q_trans);
                b.cb(qk, "qk", il);

                qk = c.ggml_mul(ctx, qk, diag_decay);
                b.cb(qk, "qk_s", il);

                const v_trans: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, Vcur, 1, 2, 0, 3));
                b.cb(v_trans, "v_trans", il);

                const qkv_diag: *Tensor = c.ggml_mul_mat(ctx, v_trans, qk);
                b.cb(qkv_diag, "qkv_diag", il);

                qkv = c.ggml_add(ctx, qkv_none_diag, qkv_diag);
                b.cb(qkv.?, "qkv", il);

                c.ggml_build_forward_expand(b.gf, qkv);

                const slopes_neg: *Tensor = c.ggml_scale(ctx, slope_rate, @floatCast(-1.0 * @as(f64, @floatFromInt(n_seq_tokens))));
                b.cb(slopes_neg, "slopes_neg", il);

                const block_decay: *Tensor = c.ggml_exp(ctx, slopes_neg);
                b.cb(block_decay, "block_decay", il);

                const block_decay_3d: *Tensor = c.ggml_reshape_3d(ctx, block_decay, 1, 1, n_head);
                b.cb(block_decay_3d, "block_decay_3d", il);

                const kv_old_s: *Tensor = c.ggml_mul(ctx, kv_old, block_decay_3d);
                b.cb(kv_old_s, "kv_old_s", il);

                const k_after_decay: *Tensor = c.ggml_mul(ctx, Kcur, k_decay);
                b.cb(k_after_decay, "k_after_decay", il);

                const k_after_decay_trans: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, k_after_decay, 1, 2, 0, 3));
                b.cb(k_after_decay_trans, "k_after_decay_trans", il);

                const kv_cur: *Tensor = c.ggml_mul_mat(ctx, k_after_decay_trans, v_trans);
                b.cb(kv_cur, "kv_cur", il);

                kv_new = c.ggml_add(ctx, kv_old_s, kv_cur);
                b.cb(kv_new.?, "kv_new", il);
            }

            // Store the new KV.
            c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, kv_new, c.ggml_view_1d(ctx, la_states_all, @as(i64, @intCast(n_embd_s)) * n_seqs, kv_head * n_embd_s * c.ggml_element_size(la_states_all))));

            var qkv_p: *Tensor = c.ggml_cont(ctx, c.ggml_permute(ctx, qkv, 0, 2, 1, 3));
            b.cb(qkv_p, "qkv_permuted", il);

            qkv_p = c.ggml_reshape_4d(ctx, qkv_p, qkv_p.ne[0] * qkv_p.ne[1], qkv_p.ne[2], 1, qkv_p.ne[3]);

            // Norm.
            const qkv_norm = b.buildNorm(qkv_p, layer.attn_norm_2, null, .rms, il);
            b.cb(qkv_norm, "qkv_norm", il);

            var g = b.buildLoraMm(layer.wg.?, cur, null);
            b.cb(g, "g", il);

            g = c.ggml_sigmoid(ctx, g);
            b.cb(g, "g_sigm", il);

            cur = c.ggml_mul(ctx, g, qkv_norm);

            cur = b.buildLoraMm(layer.wo.?, cur, null);
            b.cb(cur, "attn_out", il);

            cur = c.ggml_reshape_2d(ctx, cur, cur.ne[0], n_seq_tokens * n_seqs);
            b.cb(cur, "attn_out", il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
            residual = c.ggml_get_rows(ctx, residual, inp_out_ids);
        }

        residual = c.ggml_scale(ctx, residual, hp.f_residual_scale);
        b.cb(residual, "residual_scaled_attn", il);

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, residual);
        b.cb(ffn_inp, "ffn_inp", il);

        // The MoE branch.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        residual = cur;

        cur = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = layer.ffn_gate_exps,
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = layer.ffn_exp_probs_b,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .silu,
            .norm_w = true,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .softmax,
        }, il);
        b.cb(cur, "ffn_moe_out", il);

        residual = c.ggml_scale(ctx, residual, hp.f_residual_scale);
        b.cb(residual, "residual_scaled_ffn", il);

        cur = c.ggml_add(ctx, cur, residual);
        b.cb(cur, "ffn_out", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
