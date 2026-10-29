//! The Gemma 4 assistant: a multi-token-prediction drafter for a Gemma 4
//! target. Every layer is a next-token layer that computes only a query and
//! attends over the target context's KV cache; its input is the target's
//! token embedding beside the target's hidden state, projected down, and it
//! hands back logits and a hidden state projected up to the target's width.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/gemma4-assistant.cpp` at v0.4.1
//! (`b29c606e2`): `llama_model_gemma4_assistant`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const context_mod = @import("../context.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_gemma4_assistant`'s `load_arch_hparams`
/// (models/gemma4-assistant.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.n_embd_inp_impl = hp.nEmbdOut();

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
    _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);

    // Read and, as in the C++, not used.
    _ = try meta.getKey(u32, .attention_shared_kv_layers, false, exc);

    hp.f_attention_scale = 1.0;

    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.n_embd_head_k_swa = (try meta.getKey(u32, .attention_key_length_swa, true, exc)).?;
    hp.n_embd_head_v_swa = (try meta.getKey(u32, .attention_value_length_swa, true, exc)).?;
}

/// Ports `llama_model_gemma4_assistant`'s `load_arch_tensors`
/// (models/gemma4-assistant.cpp:21 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    if (l.n_embd_head_k != l.n_embd_head_v) {
        return exc.throw(.runtime_error, "Gemma 4 assistant requires n_embd_head_k == n_embd_head_v", .{});
    }
    if (hp.n_embd_head_k_swa != hp.n_embd_head_v_swa) {
        return exc.throw(.runtime_error, "Gemma 4 assistant requires n_embd_head_k_swa == n_embd_head_v_swa", .{});
    }
    if (hp.nEmbdOut() == l.n_embd) {
        return exc.throw(.runtime_error, "Gemma 4 assistant requires embedding_length_out to carry the target hidden size", .{});
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);

    _ = try self.createTensor(.{ .tensor = .masked_embd_centroids, .suffix = "weight" }, &.{}, nr, exc);
    _ = try self.createTensor(.{ .tensor = .masked_embd_ordering }, &.{}, nr, exc);

    const n_embd_backbone: i64 = hp.nEmbdInp();
    self.nextn_proj_post = try self.createTensor(.{ .tensor = .nextn_proj_post, .suffix = "weight" }, &.{ l.n_embd, n_embd_backbone }, 0, exc);

    var rope_freqs_flag: u32 = 0;

    var i: i32 = 0;
    while (i < l.n_layer_nextn) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const iu: u32 = @intCast(i);

        const n_head: i64 = hp.nHead(iu);
        const n_embd_head: i64 = hp.nEmbdHeadK(iu);
        const n_ff: i64 = hp.nFf(iu);

        if (i == 0) {
            self.nextn_proj_pre = try self.createTensor(.{ .tensor = .nextn_proj_pre, .suffix = "weight", .bid = i }, &.{ 2 * n_embd_backbone, l.n_embd }, 0, exc);
        }

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.wq = try self.createTensor(.{ .tensor = .attn_q, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_head * n_head }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_head * n_head, l.n_embd }, 0, exc);

        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{n_embd_head}, 0, exc);
        layer.attn_post_norm = try self.createTensor(.{ .tensor = .attn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.out_scale = try self.createTensor(.{ .tensor = .layer_out_scale, .suffix = "weight", .bid = i }, &.{1}, 0, exc);

        if (!hp.isSwa(iu)) {
            layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(n_embd_head, 2)}, rope_freqs_flag, exc);
            rope_freqs_flag = loader.tensor_duplicated;
        }

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ n_ff, l.n_embd }, 0, exc);
        layer.ffn_post_norm = try self.createTensor(.{ .tensor = .ffn_post_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
    }
}

/// Ports `llama_model_gemma4_assistant`'s `build_arch_graph`
/// (models/gemma4-assistant.cpp:80 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_gemma4_assistant`'s `graph`
/// (models/gemma4-assistant.cpp:84 @b29c606e2).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const cp = b.cparams;
    const n_embd_backbone: i64 = hp.nEmbdInp();

    var inp_tokens: *Tensor = undefined;
    var inp_h: *Tensor = undefined;
    {
        const inp = impl.allocator.create(graph.InpEmbd) catch impl.abort("models/gemma4-assistant.cpp", 91, "out of memory");
        inp.* = .{ .n_embd = n_embd_backbone };

        inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.ubatch.n_tokens);
        b.cb(inp.tokens.?, "inp_tokens", -1);
        c.ggml_set_input(inp.tokens);
        inp_tokens = inp.tokens.?;
        b.res.t_inp_tokens = inp.tokens;

        inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd_backbone, b.ubatch.n_tokens);
        b.cb(inp.embd.?, "inp_h", -1);
        c.ggml_set_input(inp.embd);
        inp_h = inp.embd.?;
        b.res.t_inp_embd = inp.embd;

        _ = b.res.addInput(graph.InpEmbd, inp);
    }

    impl.assert(cp.ctx_other != null, "models/gemma4-assistant.cpp", 108, "cparams.ctx_other != nullptr");
    const ctx_other: *const context_mod.Context = @ptrCast(@alignCast(cp.ctx_other.?));
    const model_other = ctx_other.getModel();

    var x: *Tensor = c.ggml_get_rows(ctx, model_other.tok_embd, inp_tokens);
    x = c.ggml_scale(ctx, x, impl.sqrtf(@floatFromInt(n_embd_backbone)));
    b.cb(x, "inp_embd_target", -1);

    const xh: *Tensor = c.ggml_concat(ctx, x, inp_h, 0);
    b.cb(xh, "inp_xh", -1);

    var cur: *Tensor = c.ggml_mul_mat(ctx, model.nextn_proj_pre, xh);
    b.cb(cur, "pre_proj", -1);

    const inp_attn = b.buildAttnInpKvIswa();
    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    var inpL = cur;

    var il: c_int = 0;
    while (il < b.n_layer_nextn) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const ilu: u32 = @intCast(il);
        const is_swa = hp.isSwa(ilu);

        const n_embd_head: i64 = hp.nEmbdHeadK(ilu);
        const n_head: i64 = hp.nHead(ilu);

        const freq_base_l = model.getRopeFreqBase(cp, ilu);
        const freq_scale_l = model.getRopeFreqScale(cp, ilu);
        const n_rot_l: c_int = @intCast(hp.nRot(ilu));

        const cur_norm = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur_norm, "attn_norm", il);

        var Qcur = b.buildLoraMm(layer.wq.?, cur_norm, null);
        Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, b.n_tokens);
        Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
        b.cb(Qcur, "Qcur_normed", il);

        const freq_factors: ?*Tensor = if (is_swa) null else layer.rope_freqs;
        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, freq_factors, n_rot_l, b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        b.cb(Qcur, "Qcur_pos", il);

        cur = b.buildAttn(inp_attn, layer.wo, null, null, Qcur, null, null, null, null, null, hp.f_attention_scale, il);

        if (il == b.n_layer_nextn - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }

        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        const attn_out: *Tensor = c.ggml_add(ctx, cur, inpL);
        b.cb(attn_out, "attn_out", il);

        cur = b.buildNorm(attn_out, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .gelu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = b.buildNorm(cur, layer.ffn_post_norm, null, .rms, -1);
        b.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx, cur, attn_out);

        cur = c.ggml_mul(ctx, cur, layer.out_scale);
        b.cb(cur, "out_scaled", il);

        inpL = cur;
    }
    cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    const logits = b.buildLoraMm(model.output.?, cur, null);
    b.cb(logits, "result_output", -1);
    b.res.t_logits = logits;

    const h_next: *Tensor = c.ggml_mul_mat(ctx, model.nextn_proj_post, cur);
    b.cb(h_next, "h_nextn", -1);
    b.res.t_h_nextn = h_next;

    c.ggml_build_forward_expand(b.gf, logits);
    c.ggml_build_forward_expand(b.gf, h_next);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
