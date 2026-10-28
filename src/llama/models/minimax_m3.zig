//! MiniMax-M3: MiniMax-M2 style GQA (per-head QK norm, partial rotary)
//! with DeepSeek-V3's leading dense layers and routed plus shared experts
//! (sigmoid gating, routed scaling), the swigluoai activation, and
//! MiniMax Sparse Attention (MSA). MTP is not in the released weights.
//!
//! MSA blocks are defined over token positions. The graph translates
//! between position space (block selection) and cell space (K/V and
//! indexer storage) through per-ubatch position-to-cell maps populated from
//! the KV cells.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/minimax-m3.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_minimax_m3`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, its `graph` with
//! `build_attn_msa_fa`, and `llm_graph_input_msa`, which upstream defines
//! in the model's own file. `msa_params` (src/models/models.h:2137
//! @5266f24da) is not kept on the model: its three fields are the
//! hyperparameters they are loaded from, read where the graph needs them.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const kv_cache_msa = @import("../kv_cache_msa.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const Ubatch = @import("../batch.zig").Ubatch;
const MsaContext = kv_cache_msa.MsaContext;

/// Ports `llama_model_minimax_m3`'s `load_arch_hparams` (src/models/minimax-m3.cpp:13 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    hp.expert_gating_func = (try meta.getKey(u32, .expert_gating_func, true, exc)).?;
    hp.indexer_n_head = (try meta.getKey(u32, .attention_indexer_head_count, true, exc)).?;
    hp.indexer_head_size = (try meta.getKey(u32, .attention_indexer_key_length, true, exc)).?;
    hp.indexer_top_k = (try meta.getKey(u32, .attention_indexer_top_k, true, exc)).?;
    hp.indexer_block_size = (try meta.getKey(u32, .attention_indexer_block_size, true, exc)).?;
    hp.indexer_local_blocks = (try meta.getKey(u32, .attention_indexer_local_blocks, true, exc)).?;

    // Avoid a division by zero.
    impl.assert(hp.indexer_block_size > 0, "models/minimax-m3.cpp", 28, "hparams.indexer_block_size > 0");

    self.type = switch (hp.nLayer()) {
        60 => .@"428b_a23b",
        else => .unknown,
    };
}

/// Ports `llama_model_minimax_m3`'s `load_arch_tensors` (src/models/minimax-m3.cpp:36 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const n_expert_shared: i64 = hp.n_expert_shared;
    const n_ff_exp: i64 = hp.nFfExp(0);

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        // Per-head QK norm: a single head_dim vector applied to every head.
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            // The leading dense layers.
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        } else {
            // The routed experts.
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, 0, exc);

            // The shared expert.
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, 0, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);

            // The indexer.
            layer.index_q_proj = try self.createTensor(.{ .tensor = .indexer_q_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.indexer_n_head *% hp.indexer_head_size }, 0, exc);
            layer.index_k_proj = try self.createTensor(.{ .tensor = .indexer_k_proj, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.indexer_head_size }, 0, exc);
            layer.index_q_norm = try self.createTensor(.{ .tensor = .indexer_q_norm, .suffix = "weight", .bid = i }, &.{hp.indexer_head_size}, 0, exc);
            layer.index_k_norm = try self.createTensor(.{ .tensor = .indexer_k_norm, .suffix = "weight", .bid = i }, &.{hp.indexer_head_size}, 0, exc);
        }
    }
}

/// Ports `llama_model_minimax_m3`'s `build_arch_graph` (src/models/minimax-m3.cpp:87 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// `GGML_PAD` (ggml.h:267 @5266f24da) over `int64_t`: a mask, so for a
/// block size that is not a power of two it is not a round-up, as in the C.
fn padI64(x: i64, n: i64) i64 {
    return (x +% n -% 1) & ~(n -% 1);
}

/// Ports `llm_graph_input_msa` (src/models/minimax-m3.cpp:91 @5266f24da): the
/// per-ubatch maps between positions and cells that MSA's block selection
/// reads, and the bias that forces the local blocks.
const InpMsa = struct {
    /// F32 `[nblk, n_tokens]`: the local-force bias, over position blocks.
    bias: ?*Tensor = null,
    /// F32 `[n_ps, n_tokens]`: 0/-inf visibility, by position.
    pos_mask: ?*Tensor = null,
    /// I32 `[n_ps, ns]`: position to cell, a `get_rows` index.
    pos_slot_i: ?*Tensor = null,
    /// F32 `[n_ps, ns]`: position to cell as gatherable values, at decode.
    pos_slot_f: ?*Tensor = null,
    /// I32 `[n_kv, ns]`: cell to position block, for a batch.
    cell_blk: ?*Tensor = null,

    mctx: *const MsaContext,

    blk: c_int,
    local: c_int,

    /// Ports `llm_graph_input_msa`'s `set_input` (src/models/minimax-m3.cpp:96 @5266f24da).
    pub fn setInput(self: *InpMsa, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        if (self.pos_slot_i) |t| self.mctx.setInputPosSlot(t, ubatch);
        if (self.pos_slot_f) |t| self.mctx.setInputPosSlot(t, ubatch);
        if (self.cell_blk) |t| self.mctx.setInputCellPos(t, ubatch, self.blk);
        if (self.pos_mask) |t| self.mctx.setInputPosMask(t, ubatch);

        // The local-force bias, over position blocks.
        if (self.bias) |bias| {
            if (ubatch.pos == null) return;
            const n_tokens: i64 = ubatch.n_tokens;
            const nblk: i64 = bias.ne[0];
            const data = impl.allocator.alloc(f32, @intCast(nblk * n_tokens)) catch impl.abort("models/minimax-m3.cpp", 106, "out of memory");
            defer impl.allocator.free(data);
            @memset(data, 0.0);
            var i: i64 = 0;
            while (i < n_tokens) : (i += 1) {
                const L: i64 = @divTrunc(ubatch.pos[@intCast(i)], self.blk);
                var l: c_int = 0;
                while (l < self.local and L - l >= 0) : (l += 1) {
                    if (L - l < nblk) {
                        data[@intCast(i * nblk + (L - l))] = 1e30;
                    }
                }
            }
            c.ggml_backend_tensor_set(bias, data.ptr, 0, data.len * @sizeOf(f32));
        }
    }

    /// Ports `llm_graph_input_msa`'s `can_reuse` (src/models/minimax-m3.cpp:121
    /// @5266f24da): valid while the tensors' dimensions still match the new
    /// ubatch and cache window, and the ubatch is in the same regime --
    /// decode graphs have `pos_slot_f`, batch graphs `cell_blk`.
    pub fn canReuse(self: *InpMsa, params: *const graph.Params) bool {
        const mctx_new = params.mctx.?.cast(MsaContext);

        self.mctx = mctx_new;

        const n_ps = padI64(mctx_new.getNPos(), self.blk);
        const ns: i64 = if (params.cparams.kv_unified) 1 else params.ubatch.n_seqs_unq;

        // One token per stream.
        const decode = params.ubatch.n_tokens == ns;

        const n_tokens: i64 = params.ubatch.n_tokens;

        var res = true;

        res = res and self.bias.?.ne[0] * self.blk == n_ps;
        res = res and self.bias.?.ne[1] == n_tokens;

        res = res and self.pos_mask.?.ne[0] == n_ps;
        res = res and self.pos_mask.?.ne[1] == n_tokens;

        res = res and self.pos_slot_i.?.ne[0] == n_ps;
        res = res and self.pos_slot_i.?.ne[1] == ns;

        res = res and decode == (self.pos_slot_f != null);
        res = res and decode == (self.cell_blk == null);

        if (self.pos_slot_f) |t| {
            res = res and t.ne[0] == n_ps;
            res = res and t.ne[1] == ns;
        }

        if (self.cell_blk) |t| {
            res = res and t.ne[0] == @as(i64, mctx_new.getBase().getNKv());
            res = res and t.ne[1] == ns;
        }

        return res;
    }
};

/// Ports `llama_model_minimax_m3`'s `graph`'s `build_attn_msa_fa`
/// (src/models/minimax-m3.cpp:171 @5266f24da): one flash-attention call for
/// every GQA group -- and at a multi-stream decode, every stream -- by
/// mapping them onto its sequence dimension, `ne[3]`.
///
/// Parameters:
/// - `q_cur`: `[D, HQ, T]`.
/// - `k`, `v`: `[D, n_keys, 1, C]`.
/// - `mask`: `[n_keys, R, 1, C]`, F16 and contiguous.
fn buildAttnMsaFa(b: *const graph.Builder, q_cur: *Tensor, k: *Tensor, v: *Tensor, mask: *Tensor, Gp: i64, kq_scale: f32, il: c_int) *Tensor {
    const ctx = b.ctx0;

    const D = q_cur.ne[0];
    const HQ = q_cur.ne[1];
    const T = q_cur.ne[2];
    const C = k.ne[3];
    const R = @divTrunc(HQ * T, Gp * C);
    impl.assert(Gp * C * R == HQ * T, "models/minimax-m3.cpp", 183, "Gp*C*R == HQ*T");
    impl.assert(mask.type == c.GGML_TYPE_F16, "models/minimax-m3.cpp", 184, "mask->type == GGML_TYPE_F16");

    // [D, HQ, T] -> [D, Gp, C, R] -> [D, R, Gp, C]
    // batch  (C = HKV,    R = T): channel = group
    // decode (C = HKV*ns, R = 1): channel = (group, stream), group innermost
    var q: *Tensor = c.ggml_reshape_4d(ctx, q_cur, D, Gp, C, R);
    q = c.ggml_permute(ctx, q, 0, 2, 3, 1);

    var o: *Tensor = c.ggml_flash_attn_ext(ctx, q, k, v, mask, kq_scale, b.hparams.f_max_alibi_bias, 0.0);
    c.ggml_flash_attn_ext_set_prec(o, c.GGML_PREC_F32);
    b.cb(o, "msa_fattn", il);

    // [D, Gp, R, C] -> [D, Gp, C, R] -> [n_embd, T]
    o = c.ggml_permute(ctx, o, 0, 1, 3, 2);
    if (!c.ggml_is_contiguous(o)) {
        // A no-op layout at decode (R == 1), a copy for a batch.
        o = c.ggml_cont(ctx, o);
    }
    return c.ggml_reshape_2d(ctx, o, D * HQ, T);
}

/// Whether the warning that MSA is off without flash attention has been
/// given: once per process, as the C++'s function-local `static`.
var warned_no_fa = false;
/// Whether the warning that MSA is off for a unified multi-sequence cache
/// has been given.
var warned_unified = false;

/// Ports the constructor of `llama_model_minimax_m3`'s `graph`
/// (src/models/minimax-m3.cpp:205 @5266f24da). Partial rotary: the head size
/// is not `n_rot`, so that is not asserted.
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const cparams = b.cparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const n_tokens = b.n_tokens;

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/minimax-m3.cpp", 209, "n_embd_head == hparams.n_embd_head_k()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    // ==========================================
    // The C++'s TODO: avoid such kind of complexity in the model graphs.

    // MSA calls `ggml_flash_attn_ext` directly and assumes the
    // non-transposed V layout llama.cpp only provides when flash attention
    // is on. Block selection is anchored to absolute KV cache slots, which
    // equal positions only for append-only per-stream caches: either a
    // single sequence, or several with `kv_unified == false`, where each
    // stream has its own slot space. A unified cache with several
    // sequences interleaves slots and would silently break the anchoring,
    // so it falls back to dense attention.
    const fa_on = cparams.flash_attn;
    const streams_ok = cparams.n_seq_max == 1 or !cparams.kv_unified;
    const msa_enabled = fa_on and streams_ok;

    const inp_attn = b.buildAttnInpKvMsa(msa_enabled);

    if (!fa_on and !warned_no_fa) {
        impl.logWarn("%s: flash attention disabled; MSA requires it -> running DENSE attention " ++
            "(output may be degraded). Enable flash attention for MSA.\n", .{"graph"});
        warned_no_fa = true;
    }
    if (fa_on and !streams_ok and !warned_unified) {
        impl.logWarn("%s: unified KV cache with n_seq_max > 1; MSA needs per-sequence streams " ++
            "-> running DENSE attention. Output may be degraded. Drop --kv-unified to enable MSA.\n", .{"graph"});
        warned_unified = true;
    }
    // ==========================================

    // The per-graph MSA state, shared by every sparse layer.
    var msa: ?*InpMsa = null;
    var msa_mf: ?*Tensor = null; // an F32 copy of the FA mask, for the final mask add
    var n_kv: i64 = 0;
    var n_ps: i64 = 0;
    var nblk: i64 = 0;
    var ns: i64 = 1;
    var n_tps: i64 = 0;
    var msa_decode = false; // gather (one token per stream) or mask
    const blk: c_int = @bitCast(hp.indexer_block_size);
    const topk_blocks: c_int = @bitCast(hp.indexer_top_k);
    const local: c_int = @bitCast(hp.indexer_local_blocks);
    const Hd: i64 = hp.indexer_n_head; // one indexer head per GQA group

    if (msa_enabled) {
        const mctx_msa = b.mctx.?.cast(MsaContext);

        const msa_kqm = inp_attn.getKqMask().?;
        n_kv = msa_kqm.ne[0];
        n_tps = msa_kqm.ne[1]; // tokens per stream
        ns = msa_kqm.ne[3]; // streams in this ubatch
        impl.assert(msa_kqm.type == c.GGML_TYPE_F16, "models/minimax-m3.cpp", 264, "msa_kqm->type == GGML_TYPE_F16 && \"MSA requires the FA (f16) mask\"");
        impl.assert(n_tps * ns == n_tokens, "models/minimax-m3.cpp", 265, "n_tps*ns == n_tokens");

        // The position axis covers every position in the cache, padded to
        // whole blocks.
        n_ps = padI64(mctx_msa.getNPos(), blk);
        nblk = @divTrunc(n_ps, blk);
        msa_decode = n_tps == 1;

        const inp = impl.allocator.create(InpMsa) catch impl.abort("models/minimax-m3.cpp", 272, "out of memory");
        inp.* = .{ .mctx = mctx_msa, .blk = blk, .local = local };

        // Stream-grouped tokens.
        inp.bias = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, nblk, n_tokens);
        c.ggml_set_input(inp.bias);

        inp.pos_mask = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_ps, n_tokens);
        c.ggml_set_input(inp.pos_mask);

        inp.pos_slot_i = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, n_ps, ns);
        c.ggml_set_input(inp.pos_slot_i);

        if (msa_decode) {
            inp.pos_slot_f = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_ps, ns);
            c.ggml_set_input(inp.pos_slot_f);
        } else {
            inp.cell_blk = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, n_kv, ns);
            c.ggml_set_input(inp.cell_blk);

            msa_mf = c.ggml_cast(ctx, msa_kqm, c.GGML_TYPE_F32);
        }

        msa = b.res.addInput(InpMsa, inp);
    }

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;
        var cur: *Tensor = undefined;

        // Self-attention.
        {
            cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            var Vcur = qkv.v;

            // Per-head QK RMS norm; the weights already include Gemma's +1.
            Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
            b.cb(Qcur, "Qcur_normed", il);
            Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
            b.cb(Kcur, "Kcur_normed", il);

            // Partial rotary: only the first n_rot dimensions are rotated.
            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            const is_sparse = msa_enabled and il >= @as(c_int, @bitCast(hp.n_layer_dense_lead));

            if (!is_sparse) {
                cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);
            } else {
                const m = msa.?;
                const n_idx_dim: i64 = hp.indexer_head_size;

                // The index branch: project, norm, partial RoPE, cache.
                var iq = b.buildLoraMm(layer.index_q_proj.?, cur, null);
                var ik = b.buildLoraMm(layer.index_k_proj.?, cur, null);
                iq = c.ggml_reshape_3d(ctx, iq, n_idx_dim, Hd, n_tokens);
                ik = c.ggml_reshape_3d(ctx, ik, n_idx_dim, 1, n_tokens);
                iq = b.buildNorm(iq, layer.index_q_norm, null, .rms, il); // the +1 is baked in
                ik = b.buildNorm(ik, layer.index_k_norm, null, .rms, il);
                iq = c.ggml_rope_ext(ctx, iq, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
                ik = c.ggml_rope_ext(ctx, ik, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

                const mctx_msa_l = b.mctx.?.cast(MsaContext);
                const mctx_cur = mctx_msa_l.getBase();
                const mctx_idx = mctx_msa_l.getIdx();
                c.ggml_build_forward_expand(b.gf, mctx_idx.cpyK(ctx, ik, inp_attn.getKIdxsIdx(), il));
                const ik_kv = mctx_idx.getK(ctx, il);

                if (inp_attn.base.self_k_rot) |r| {
                    Qcur = graph.mulMatHadamard(ctx, Qcur, r);
                    Kcur = graph.mulMatHadamard(ctx, Kcur, r);
                }
                if (inp_attn.base.self_v_rot) |r| {
                    Vcur = graph.mulMatHadamard(ctx, Vcur, r);
                }

                // The main branch: store K and V, take the cache views.
                c.ggml_build_forward_expand(b.gf, Qcur);
                c.ggml_build_forward_expand(b.gf, Kcur);
                c.ggml_build_forward_expand(b.gf, Vcur);
                c.ggml_build_forward_expand(b.gf, mctx_cur.cpyK(ctx, Kcur, inp_attn.base.getKIdxs(), il));
                c.ggml_build_forward_expand(b.gf, mctx_cur.cpyV(ctx, Vcur, inp_attn.base.getVIdxs(), il));
                const k = mctx_cur.getK(ctx, il);
                const v = mctx_cur.getV(ctx, il);
                impl.assert(!(v.nb[1] > v.nb[2]), "models/minimax-m3.cpp", 372, "!(v->nb[1] > v->nb[2]) && \"MSA assumes v_trans=false (FA on)\"");

                const D = k.ne[0];
                const HKV = k.ne[1];
                const Gp = @divTrunc(b.n_head, HKV);
                impl.assert(HKV == Hd, "models/minimax-m3.cpp", 377, "HKV == Hd && \"MSA: one indexer head per GQA group\"");
                impl.assert(k.ne[3] == ns, "models/minimax-m3.cpp", 378, "k->ne[3] == ns");
                const K: c_int = if (topk_blocks < @as(c_int, @truncate(nblk))) topk_blocks else @truncate(nblk);
                const K64: i64 = K;
                const blk64: i64 = blk;

                const kq_scale = 1.0 / impl.sqrtf(@floatFromInt(n_embd_head));

                if (msa_decode) {
                    // Decode: top-k and gather batched over the streams,
                    // then one grouped flash attention.
                    // Gather the indexer keys through the position-to-cell map.
                    const ik3: *Tensor = c.ggml_view_3d(ctx, ik_kv, n_idx_dim, n_kv, ns, ik_kv.nb[2], ik_kv.nb[3], 0);
                    const ikp: *Tensor = c.ggml_get_rows(ctx, ik3, m.pos_slot_i); // [n_idx_dim, n_ps, ns]
                    const iq4: *Tensor = c.ggml_reshape_4d(ctx, iq, n_idx_dim, Hd, 1, ns);
                    var sc: *Tensor = c.ggml_mul_mat(ctx, c.ggml_reshape_4d(ctx, ikp, n_idx_dim, n_ps, 1, ns), iq4);
                    c.ggml_mul_mat_set_prec(sc, c.GGML_PREC_F32);
                    // Unmapped positions come out -inf, so they can never
                    // rank into the top-k.
                    sc = c.ggml_add_inplace(ctx, sc, c.ggml_reshape_4d(ctx, m.pos_mask, n_ps, 1, 1, ns));
                    const bs: *Tensor = c.ggml_pool_2d(ctx, sc, c.GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);
                    b.cb(bs, "msa_bs", il);

                    const bsf: *Tensor = c.ggml_add(ctx, bs, c.ggml_reshape_4d(ctx, m.bias, nblk, 1, 1, ns));
                    const idx: *Tensor = c.ggml_top_k(ctx, bsf, K); // position blocks

                    // position index: tj[t,k,h,s] = blk*idx[k,h,s] + t   (positions, the mask gather)
                    // cell index:     cs[t,k,h,s] = pos_slot[tj]         (position to cell)
                    // row index:      tr[t,k,h,s] = cs*HKV + h           (per-stream K/V gather)
                    var a: *Tensor = c.ggml_scale(ctx, c.ggml_cast(ctx, idx, c.GGML_TYPE_F32), @floatFromInt(blk));
                    a = c.ggml_reshape_4d(ctx, a, 1, K64, Hd, ns);
                    const tj_rep: *Tensor = c.ggml_repeat_4d(ctx, a, blk64, K64, Hd, ns);
                    const tj: *Tensor = c.ggml_add(ctx, tj_rep, c.ggml_reshape_3d(ctx, c.ggml_arange(ctx, 0.0, @floatFromInt(blk), 1.0), blk64, 1, 1));

                    const tokj: *Tensor = c.ggml_cast(ctx, c.ggml_reshape_2d(ctx, tj, blk64 * K64 * Hd, ns), c.GGML_TYPE_I32);

                    var cs: *Tensor = c.ggml_get_rows(ctx, c.ggml_reshape_3d(ctx, m.pos_slot_f, 1, n_ps, ns), tokj); // [1, blk*K*Hd, ns]
                    cs = c.ggml_reshape_4d(ctx, cs, blk64, K64, Hd, ns);

                    const cs_scaled: *Tensor = c.ggml_scale(ctx, cs, @floatFromInt(HKV));
                    const tr: *Tensor = c.ggml_add(ctx, cs_scaled, c.ggml_reshape_3d(ctx, c.ggml_arange(ctx, 0.0, @floatFromInt(HKV), 1.0), 1, 1, Hd));

                    const tokr: *Tensor = c.ggml_cast(ctx, c.ggml_reshape_2d(ctx, tr, blk64 * K64 * Hd, ns), c.GGML_TYPE_I32);

                    const k3: *Tensor = c.ggml_view_3d(ctx, k, D, HKV * n_kv, ns, k.nb[1], k.nb[3], 0);
                    const v3: *Tensor = c.ggml_view_3d(ctx, v, D, HKV * n_kv, ns, v.nb[1], v.nb[3], 0);
                    const mp: *Tensor = c.ggml_reshape_3d(ctx, m.pos_mask, 1, n_ps, ns);

                    const kg: *Tensor = c.ggml_get_rows(ctx, k3, tokr);
                    const vg: *Tensor = c.ggml_get_rows(ctx, v3, tokr);
                    const mg: *Tensor = c.ggml_get_rows(ctx, mp, tokj);

                    // Fold (group, stream) onto the FA channel dimension.
                    const kt: c.enum_ggml_type = if (c.ggml_is_quantized(k.type)) c.GGML_TYPE_F16 else k.type;
                    const vt: c.enum_ggml_type = if (c.ggml_is_quantized(v.type)) c.GGML_TYPE_F16 else v.type;
                    var kfa: *Tensor = c.ggml_reshape_4d(ctx, kg, D, blk64 * K64, 1, Hd * ns);
                    var vfa: *Tensor = c.ggml_reshape_4d(ctx, vg, D, blk64 * K64, 1, Hd * ns);
                    if (kfa.type != kt) kfa = c.ggml_cast(ctx, kfa, kt);
                    if (vfa.type != vt) vfa = c.ggml_cast(ctx, vfa, vt);
                    // The FA mask must be F16.
                    const mfa: *Tensor = c.ggml_cast(ctx, c.ggml_reshape_4d(ctx, mg, blk64 * K64, 1, 1, Hd * ns), c.GGML_TYPE_F16);

                    cur = buildAttnMsaFa(b, Qcur, kfa, vfa, mfa, Gp, kq_scale, il);
                } else {
                    // A batch: a loop over the streams.
                    const outs = impl.allocator.alloc(*Tensor, @intCast(ns)) catch impl.abort("models/minimax-m3.cpp", 445, "out of memory");
                    defer impl.allocator.free(outs);
                    const mf = msa_mf.?;
                    var st: i64 = 0;
                    while (st < ns) : (st += 1) {
                        const stu: u64 = @intCast(st);
                        const tps: u64 = @intCast(n_tps);
                        const iq_s: *Tensor = c.ggml_view_3d(ctx, iq, n_idx_dim, Hd, n_tps, iq.nb[1], iq.nb[2], stu * tps * iq.nb[2]);
                        const ik_s: *Tensor = c.ggml_view_2d(ctx, ik_kv, n_idx_dim, n_kv, ik_kv.nb[2], stu * ik_kv.nb[3]);
                        const psl_s: *Tensor = c.ggml_view_1d(ctx, m.pos_slot_i, n_ps, stu * m.pos_slot_i.?.nb[1]);
                        const pm_s: *Tensor = c.ggml_view_3d(ctx, m.pos_mask, n_ps, 1, n_tps, m.pos_mask.?.nb[1], m.pos_mask.?.nb[1], stu * tps * m.pos_mask.?.nb[1]);
                        const cb_s: *Tensor = c.ggml_view_1d(ctx, m.cell_blk, n_kv, stu * m.cell_blk.?.nb[1]);
                        const mf_s: *Tensor = c.ggml_view_3d(ctx, mf, n_kv, n_tps, 1, mf.nb[1], mf.nb[3], stu * mf.nb[3]);
                        const bias_s: *Tensor = c.ggml_view_3d(ctx, m.bias, nblk, 1, n_tps, m.bias.?.nb[1], m.bias.?.nb[1], stu * tps * m.bias.?.nb[1]);
                        const q_s: *Tensor = c.ggml_view_3d(ctx, Qcur, D, b.n_head, n_tps, Qcur.nb[1], Qcur.nb[2], stu * tps * Qcur.nb[2]);
                        const k_s: *Tensor = c.ggml_view_4d(ctx, k, D, HKV, n_kv, 1, k.nb[1], k.nb[2], k.nb[3], stu * k.nb[3]);
                        const v_s: *Tensor = c.ggml_view_4d(ctx, v, D, HKV, n_kv, 1, v.nb[1], v.nb[2], v.nb[3], stu * v.nb[3]);

                        // Block scores: the indexer keys are gathered
                        // through the position-to-cell map first. They are
                        // unscaled; only the top-k order matters.
                        const ikp: *Tensor = c.ggml_get_rows(ctx, ik_s, psl_s); // [n_idx_dim, n_ps]
                        var sc: *Tensor = c.ggml_mul_mat(ctx, ikp, c.ggml_reshape_2d(ctx, iq_s, n_idx_dim, Hd * n_tps));
                        // The indexer scores run in F32.
                        c.ggml_mul_mat_set_prec(sc, c.GGML_PREC_F32);
                        sc = c.ggml_reshape_3d(ctx, sc, n_ps, Hd, n_tps);
                        // Unmapped positions (holes, padding, empty cells) come out -inf.
                        sc = c.ggml_add_inplace(ctx, sc, pm_s);
                        const bs: *Tensor = c.ggml_pool_2d(ctx, sc, c.GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);
                        b.cb(bs, "msa_bs", il);

                        // Bias the scores so the locally forced blocks always rank first.
                        const bsf: *Tensor = c.ggml_add(ctx, bs, bias_s); // [nblk, Hd, n_tps]
                        b.cb(bsf, "msa_bsf", il);

                        const idx: *Tensor = c.ggml_top_k(ctx, bsf, K); // [K, Hd, n_tps] i32

                        var ninf: *Tensor = c.ggml_cast(ctx, c.ggml_scale_bias(ctx, bias_s, 0.0, -1e30), c.GGML_TYPE_F16); // [nblk, 1, n_tps]
                        ninf = c.ggml_repeat_4d(ctx, ninf, nblk, Hd, n_tps, 1);
                        const zero: *Tensor = c.ggml_scale(ctx, c.ggml_cast(ctx, idx, c.GGML_TYPE_F32), 0.0);
                        const bm_dst: *Tensor = c.ggml_reshape_3d(ctx, ninf, 1, nblk, Hd * n_tps);
                        const bm_src: *Tensor = c.ggml_reshape_3d(ctx, zero, 1, K64, Hd * n_tps);
                        var bm: *Tensor = c.ggml_set_rows(ctx, bm_dst, bm_src, c.ggml_reshape_2d(ctx, idx, K64, Hd * n_tps));
                        bm = c.ggml_reshape_3d(ctx, bm, nblk, Hd, n_tps);
                        bm = c.ggml_cont(ctx, c.ggml_permute(ctx, bm, 0, 2, 1, 3)); // [nblk, n_tps, Hd]
                        b.cb(bm, "msa_block_mask", il);

                        // Expand blocks to cells through the cell-to-block
                        // map, then combine with the causal mask; empty
                        // cells are masked by the causal mask.
                        const bm2: *Tensor = c.ggml_cont(ctx, c.ggml_transpose(ctx, c.ggml_reshape_2d(ctx, bm, nblk, n_tps * Hd))); // [n_tps*Hd, nblk]
                        const bmc: *Tensor = c.ggml_get_rows(ctx, bm2, cb_s); // [n_tps*Hd, n_kv] F32
                        var bmx: *Tensor = c.ggml_cont(ctx, c.ggml_transpose(ctx, bmc));
                        bmx = c.ggml_reshape_3d(ctx, bmx, n_kv, n_tps, Hd);
                        var mask4: *Tensor = c.ggml_add_inplace(ctx, bmx, mf_s);
                        mask4 = c.ggml_cast(ctx, c.ggml_reshape_4d(ctx, mask4, n_kv, n_tps, 1, Hd), c.GGML_TYPE_F16);
                        b.cb(mask4, "msa_mask4", il);

                        // The cache views, with the groups on ne[3].
                        const kfa: *Tensor = c.ggml_permute(ctx, k_s, 0, 3, 1, 2);
                        const vfa: *Tensor = c.ggml_permute(ctx, v_s, 0, 3, 1, 2);

                        outs[@intCast(st)] = buildAttnMsaFa(b, q_s, kfa, vfa, mask4, Gp, kq_scale, il);
                    }
                    cur = outs[0];
                    st = 1;
                    while (st < ns) : (st += 1) {
                        cur = c.ggml_concat(ctx, cur, outs[@intCast(st)], 1);
                    }
                }
                if (inp_attn.base.self_v_rot) |r| {
                    cur = graph.mulMatHadamard(ctx, cur, r);
                }
                b.cb(cur, "kqv_out", il);
                if (layer.wo) |wo| {
                    cur = b.buildLoraMm(wo, cur, layer.wo_s);
                }
            }
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        if (@as(u32, @bitCast(il)) < hp.n_layer_dense_lead) {
            // The leading dense FFN (swigluoai).
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .swiglu_oai_moe, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The routed experts (swigluoai MoE).
            const moe_out = b.buildMoeFfn(cur, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = layer.ffn_gate_exps,
                .down_exps = layer.ffn_down_exps.?,
                .exp_probs_b = layer.ffn_exp_probs_b,
                .n_expert = b.n_expert,
                .n_expert_used = b.n_expert_used,
                .type_op = .swiglu_oai_moe,
                .norm_w = hp.expert_weights_norm,
                .w_scale = hp.expert_weights_scale,
                .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
            }, il);
            b.cb(moe_out, "ffn_moe_out", il);

            // The shared expert (swigluoai).
            const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .swiglu_oai_moe, .par, il);
            b.cb(ffn_shexp, "ffn_shexp", il);

            cur = c.ggml_add(ctx, moe_out, ffn_shexp);
            b.cb(cur, "ffn_out", il);
        }

        cur = c.ggml_add(ctx, cur, ffn_inp);

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

test "GGML_PAD over int64 is a mask" {
    try std.testing.expectEqual(@as(i64, 256), padI64(256, 64));
    try std.testing.expectEqual(@as(i64, 320), padI64(257, 64));
}
