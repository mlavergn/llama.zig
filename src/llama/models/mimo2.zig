//! MiMo-V2: interleaved sliding-window attention with attention sinks and
//! an optional value scale, Q and K sharing one head size while V has its
//! own, a sigmoid-routed mixture of experts or a dense feed-forward per
//! layer, and an appended NextN block for multi-token prediction.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mimo2.cpp` at v0.4.0 (`5266f24da`):
//! `llama_model_mimo2`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph` and its `graph_mtp`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_mimo2`'s `load_arch_hparams` (models/mimo2.cpp:3 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    hp.swa_type = @backingInt(hparams_mod.SwaType.standard);

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    _ = try meta.getKeyOrArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, hp.nLayer(), true, exc);

    if (try meta.getKey(f32, .attention_value_scale, false, exc)) |value_scale| {
        if (value_scale != 1.0) hp.f_attn_value_scale = value_scale;
    }

    self.type = switch (hp.nLayer()) {
        48 => .@"310b_a15b",
        else => .unknown,
    };
}

/// Ports `llama_model_mimo2`'s `load_arch_tensors` (models/mimo2.cpp:25 @5266f24da).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    var probe_buf: [64]u8 = undefined;
    const mtp_probe = std.fmt.bufPrint(&probe_buf, "blk.{d}.nextn.eh_proj.weight", .{l.n_layer}) catch unreachable;
    const trunk_only = hp.n_layer_nextn > 0 and ml.getWeight(mtp_probe) == null;
    var mtp_flags: u32 = if (trunk_only) nr else 0;

    if (!ml.load_mtp) {
        mtp_flags |= loader.tensor_skip;
    }

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer_all) : (i += 1) {
        const layer = &self.layers[@intCast(i)];
        const n_embd_k_gqa: i64 = hp.nEmbdKGqa(@intCast(i));
        const n_embd_v_gqa: i64 = hp.nEmbdVGqa(@intCast(i));
        const n_head: i64 = hp.nHead(@intCast(i));

        const is_nextn = i >= l.n_layer;
        const flags: u32 = if (is_nextn) mtp_flags else 0;

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, flags, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_v * n_head, l.n_embd }, flags, exc);

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
        layer.attn_sinks = try self.createTensor(.{ .tensor = .attn_sinks, .suffix = "weight", .bid = i }, &.{n_head}, nr | flags, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);

        // The dense branch.
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr | flags, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, nr | flags, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, nr | flags, exc);

        // The mixture-of-experts branch.
        const n_ff_exp: i64 = hp.nFfExp(0);
        layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, nr | flags, exc);
        layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, nr | flags, exc);
        layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, nr | flags, exc);
        layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp, l.n_expert }, nr | flags, exc);
        layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr | flags, exc);

        if (is_nextn) {
            layer.nextn.eh_proj = try self.createTensor(.{ .tensor = .nextn_eh_proj, .suffix = "weight", .bid = i }, &.{ 2 * l.n_embd, l.n_embd }, flags, exc);
            layer.nextn.enorm = try self.createTensor(.{ .tensor = .nextn_enorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.hnorm = try self.createTensor(.{ .tensor = .nextn_hnorm, .suffix = "weight", .bid = i }, &.{l.n_embd}, flags, exc);
            layer.nextn.embed_tokens = try self.createTensor(.{ .tensor = .nextn_embed_tokens, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr | flags, exc);
            layer.nextn.shared_head_head = try self.createTensor(.{ .tensor = .nextn_shared_head_head, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_vocab }, nr | flags, exc);
            layer.nextn.shared_head_norm = try self.createTensor(.{ .tensor = .nextn_shared_head_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr | flags, exc);
            layer.layer_out_norm = try self.createTensor(.{ .tensor = .layer_out_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr | flags, exc);
        }
    }
}

/// Ports `llama_model_mimo2`'s `build_arch_graph` (models/mimo2.cpp:84 @5266f24da):
/// the MTP draft head for that graph type, the main model otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_mimo2`'s `graph`
/// (models/mimo2.cpp:91 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var inpL: *Tensor = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    const v_scale = hp.f_attn_value_scale;
    const emit_h_nextn = b.cparams.embeddings_nextn;
    const crop_last_layer = !emit_h_nextn or b.cparams.embeddings_nextn_masked;

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        const n_head_l: i64 = hp.nHead(@intCast(il));
        const n_head_kv_l: i64 = hp.nHeadKv(@intCast(il));
        const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
        const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));

        var cur = inpL;

        // Self-attention.
        {
            cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);

            var Qcur: *Tensor = undefined;
            var Kcur: *Tensor = undefined;
            var Vcur: *Tensor = undefined;

            if (layer.wqkv) |wqkv| {
                // The fused qkv_proj: Q and K share head_dim_k, V has
                // head_dim_v.
                const qkv = b.buildLoraMm(wqkv, cur, null);
                b.cb(qkv, "wqkv", il);

                const row_k = c.ggml_row_size(qkv.type, b.n_embd_head_k);
                const row_v = c.ggml_row_size(qkv.type, b.n_embd_head_v);
                const row_full = qkv.nb[1];
                const k_off = row_k * @as(usize, @intCast(n_head_l));
                const v_off = k_off + row_k * @as(usize, @intCast(n_head_kv_l));

                Qcur = c.ggml_view_3d(ctx, qkv, b.n_embd_head_k, n_head_l, b.n_tokens, row_k, row_full, 0);
                Kcur = c.ggml_view_3d(ctx, qkv, b.n_embd_head_k, n_head_kv_l, b.n_tokens, row_k, row_full, k_off);
                Vcur = c.ggml_view_3d(ctx, qkv, b.n_embd_head_v, n_head_kv_l, b.n_tokens, row_v, row_full, v_off);
            } else {
                // The split path.
                Qcur = b.buildLoraMm(layer.wq.?, cur, null);
                b.cb(Qcur, "Qcur", il);

                Kcur = b.buildLoraMm(layer.wk.?, cur, null);
                b.cb(Kcur, "Kcur", il);

                Vcur = b.buildLoraMm(layer.wv.?, cur, null);
                b.cb(Vcur, "Vcur", il);

                Qcur = c.ggml_reshape_3d(ctx, Qcur, b.n_embd_head_k, n_head_l, b.n_tokens);
                Kcur = c.ggml_reshape_3d(ctx, Kcur, b.n_embd_head_k, n_head_kv_l, b.n_tokens);
                Vcur = c.ggml_reshape_3d(ctx, Vcur, b.n_embd_head_v, n_head_kv_l, b.n_tokens);
            }

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            const sinks = layer.attn_sinks;

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, sinks, null, 1.0 / impl.sqrtf(@floatFromInt(b.n_embd_head_k)), il);
            b.cb(cur, "attn_out", il);

            if (v_scale != 0.0) {
                cur = c.ggml_scale(ctx, cur, v_scale);
                b.cb(cur, "attn_out_scaled", il);
            }
        }

        if (il == b.n_layer - 1 and crop_last_layer) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        // The feed-forward network.
        if (layer.ffn_gate_inp == null) {
            // The dense branch.
            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The mixture-of-experts branch.
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
                .gating_op = .sigmoid,
            }, il);
            b.cb(cur, "ffn_moe_out", il);
        }

        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;

    if (emit_h_nextn) {
        b.cb(cur, "h_nextn", -1);
        b.res.t_h_nextn = cur;

        if (!b.cparams.embeddings_nextn_masked) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
        }
    }

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports the constructor of `llama_model_mimo2`'s `graph_mtp`
/// (models/mimo2.cpp:261 @5266f24da).
///
/// Mirrors MiMo's appended NextN block: normalize and fuse the token and
/// hidden inputs, run the decoder block, expose its state before the head
/// norm to the next draft step, then apply the shared output norm and LM
/// head. Converted checkpoints may store that shared norm as
/// `layer_out_norm`, so it stays in the fallback chain.
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    impl.assert(hp.n_layer_nextn > 0, "models/mimo2.cpp", 263, "hparams.n_layer_nextn > 0 && \"MIMO2 MTP requires n_layer_nextn > 0\"");

    const il: c_int = @as(c_int, @intCast(hp.nLayer())) + b.cparams.nextn_layer_offset;
    impl.assert(b.cparams.nextn_layer_offset >= 0 and b.cparams.nextn_layer_offset < @as(c_int, @intCast(hp.n_layer_nextn)), "models/mimo2.cpp", 268, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range [0, n_layer_nextn)\"");

    const layer = &model.layers[@intCast(il)];
    impl.assert(layer.nextn.eh_proj != null, "models/mimo2.cpp", 271, "layer.nextn.eh_proj && \"MIMO2 MTP block missing nextn.eh_proj\"");
    impl.assert(layer.nextn.enorm != null, "models/mimo2.cpp", 272, "layer.nextn.enorm   && \"MIMO2 MTP block missing nextn.enorm\"");
    impl.assert(layer.nextn.hnorm != null, "models/mimo2.cpp", 273, "layer.nextn.hnorm   && \"MIMO2 MTP block missing nextn.hnorm\"");
    impl.assert(layer.wqkv != null, "models/mimo2.cpp", 274, "layer.wqkv          && \"MIMO2 MTP requires fused attn_qkv\"");

    const n_head_l: i64 = hp.nHead(@intCast(il));
    const n_head_kv_l: i64 = hp.nHeadKv(@intCast(il));

    const freq_base_l = model.getRopeFreqBase(b.cparams, @intCast(il));
    const freq_scale_l = model.getRopeFreqScale(b.cparams, @intCast(il));
    const v_scale = hp.f_attn_value_scale;

    const inp = impl.allocator.create(graph.InpEmbd) catch impl.abort("models/mimo2.cpp", 283, "out of memory");
    inp.* = .{ .n_embd = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.n_embd, b.n_tokens);
    c.ggml_set_input(inp.embd);
    _ = c.ggml_set_name(inp.embd, "mtp_h_input");

    const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;
    const h_input = inp.embd.?;
    const tok_embd: *Tensor = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    b.cb(tok_embd, "mtp_tok_embd", il);

    _ = b.res.addInput(graph.InpEmbd, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();
    const inp_attn = b.buildAttnInpKvIswa();

    const h_norm = b.buildNorm(h_input, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat: *Tensor = c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(cur, "mtp_eh_proj", il);

    const inpSA = cur;

    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    const qkv = b.buildLoraMm(layer.wqkv.?, cur, layer.wqkv_s);
    b.cb(qkv, "mtp_wqkv", il);

    const row_k = c.ggml_row_size(qkv.type, b.n_embd_head_k);
    const row_v = c.ggml_row_size(qkv.type, b.n_embd_head_v);
    const row_full = qkv.nb[1];
    const k_off = row_k * @as(usize, @intCast(n_head_l));
    const v_off = k_off + row_k * @as(usize, @intCast(n_head_kv_l));

    var Qcur: *Tensor = c.ggml_view_3d(ctx, qkv, b.n_embd_head_k, n_head_l, b.n_tokens, row_k, row_full, 0);
    var Kcur: *Tensor = c.ggml_view_3d(ctx, qkv, b.n_embd_head_k, n_head_kv_l, b.n_tokens, row_k, row_full, k_off);
    const Vcur: *Tensor = c.ggml_view_3d(ctx, qkv, b.n_embd_head_v, n_head_kv_l, b.n_tokens, row_v, row_full, v_off);

    Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, freq_base_l, freq_scale_l, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    b.cb(Qcur, "mtp_Qcur", il);
    b.cb(Kcur, "mtp_Kcur", il);
    b.cb(Vcur, "mtp_Vcur", il);

    cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, layer.attn_sinks, null, 1.0 / impl.sqrtf(@floatFromInt(b.n_embd_head_k)), il);
    b.cb(cur, "mtp_attn_out", il);

    if (v_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, v_scale);
        b.cb(cur, "mtp_attn_out_scaled", il);
    }

    const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "mtp_ffn_inp", il);

    cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "mtp_ffn_norm", il);

    impl.assert(layer.ffn_gate != null and layer.ffn_down != null and layer.ffn_up != null, "models/mimo2.cpp", 364, "layer.ffn_gate && layer.ffn_down && layer.ffn_up && \"MIMO2 MTP requires dense FFN tensors\"");
    cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
    b.cb(cur, "mtp_ffn_out", il);

    cur = c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "mtp_post_ffn", il);

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    const head_norm_w = layer.nextn.shared_head_norm orelse (layer.layer_out_norm orelse model.output_norm);
    impl.assert(head_norm_w != null, "models/mimo2.cpp", 384, "head_norm_w && \"MIMO2 MTP missing head norm fallback\"");
    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
    impl.assert(head_w != null, "models/mimo2.cpp", 390, "head_w && \"MIMO2 MTP missing LM head fallback\"");
    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
