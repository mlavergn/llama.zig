//! MiMo-V2: interleaved sliding-window attention with attention sinks and
//! an optional value scale, Q and K sharing one head size while V has its
//! own, a sigmoid-routed mixture of experts or a dense feed-forward per
//! layer, and an appended NextN block for multi-token prediction.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mimo2.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_mimo2`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `graph` and its `graph_mtp`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_mimo2`'s `load_arch_hparams` (models/mimo2.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    hp.swa_type = @backingInt(llama.hparams.SwaType.standard);

    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_swa = (try meta.getKey(u32, .attention_sliding_window, true, exc)).?;
    if (try meta.getKey(f32, .rope_freq_base_swa, false, exc)) |v| hp.rope_freq_base_train_swa = v;

    _ = try meta.getArr(u32, .attention_sliding_window_pattern, &hp.is_swa_impl, true, exc);

    if (try meta.getKey(f32, .attention_value_scale, false, exc)) |value_scale| {
        if (value_scale != 1.0) hp.f_attn_value_scale = value_scale;
    }

    self.type = switch (hp.nLayer()) {
        48 => .@"310b_a15b",
        else => .unknown,
    };
}

/// Ports `llama_model_mimo2`'s `load_arch_tensors` (models/mimo2.cpp:25 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, ml: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;

    const nf = self.nextnFlags(ml, .attn_norm);
    const trunk_flags = nf.trunk;
    const mtp_flags = nf.mtp;

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
        const flags: u32 = if (is_nextn) mtp_flags else trunk_flags;

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

/// Ports `llama_model_mimo2`'s `build_arch_graph` (models/mimo2.cpp:80 @b1b770d90):
/// the MTP draft head for that graph type, the main model otherwise.
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, params: *const llama.graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_mimo2`'s `graph`
/// (models/mimo2.cpp:87 @b1b770d90).
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var inpL: *llama.impl.c.ggml_tensor = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();
    const inp_attn = b.buildAttnInpKvIswa();
    const inp_out_ids = b.buildInpOutIds();

    const v_scale = hp.f_attn_value_scale;
    const layer_inp = b.cparams.embeddings_layer_inp;
    const n_layer_u: usize = @intCast(b.n_layer);
    const extract_final_inp = n_layer_u < layer_inp.len and layer_inp[n_layer_u];

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = inpL;

        var inpSA = inpL;

        const n_head_l: i64 = hp.nHead(@intCast(il));
        const n_head_kv_l: i64 = hp.nHeadKv(@intCast(il));

        var cur = inpL;

        // Self-attention.
        {
            cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);

            var Qcur: *llama.impl.c.ggml_tensor = undefined;
            var Kcur: *llama.impl.c.ggml_tensor = undefined;
            var Vcur: *llama.impl.c.ggml_tensor = undefined;

            if (layer.wqkv) |wqkv| {
                // The fused qkv_proj: Q and K share head_dim_k, V has
                // head_dim_v.
                const qkv = b.buildLoraMm(wqkv, cur, null);
                b.cb(qkv, "wqkv", il);

                const row_k = llama.impl.c.ggml_row_size(qkv.type, b.n_embd_head_k);
                const row_v = llama.impl.c.ggml_row_size(qkv.type, b.n_embd_head_v);
                const row_full = qkv.nb[1];
                const k_off = row_k * @as(usize, @intCast(n_head_l));
                const v_off = k_off + row_k * @as(usize, @intCast(n_head_kv_l));

                Qcur = llama.impl.c.ggml_view_3d(ctx, qkv, b.n_embd_head_k, n_head_l, b.n_tokens, row_k, row_full, 0);
                Kcur = llama.impl.c.ggml_view_3d(ctx, qkv, b.n_embd_head_k, n_head_kv_l, b.n_tokens, row_k, row_full, k_off);
                Vcur = llama.impl.c.ggml_view_3d(ctx, qkv, b.n_embd_head_v, n_head_kv_l, b.n_tokens, row_v, row_full, v_off);
            } else {
                // The split path.
                Qcur = b.buildLoraMm(layer.wq.?, cur, null);
                b.cb(Qcur, "Qcur", il);

                Kcur = b.buildLoraMm(layer.wk.?, cur, null);
                b.cb(Kcur, "Kcur", il);

                Vcur = b.buildLoraMm(layer.wv.?, cur, null);
                b.cb(Vcur, "Vcur", il);

                Qcur = llama.impl.c.ggml_reshape_3d(ctx, Qcur, b.n_embd_head_k, n_head_l, b.n_tokens);
                Kcur = llama.impl.c.ggml_reshape_3d(ctx, Kcur, b.n_embd_head_k, n_head_kv_l, b.n_tokens);
                Vcur = llama.impl.c.ggml_reshape_3d(ctx, Vcur, b.n_embd_head_v, n_head_kv_l, b.n_tokens);
            }

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            const sinks = layer.attn_sinks;

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, sinks, null, 1.0 / stdlib.libm.sqrtfInt(b.n_embd_head_k), il);
            b.cb(cur, "attn_out", il);

            if (v_scale != 0.0) {
                cur = llama.impl.c.ggml_scale(ctx, cur, v_scale);
                b.cb(cur, "attn_out_scaled", il);
            }
        }

        if (il == b.n_layer - 1 and b.cropBeforeNextn(inp_out_ids) and !extract_final_inp) {
            cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpSA);
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

        cur = llama.impl.c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;
    if (extract_final_inp) {
        b.res.t_layer_inp[n_layer_u] = cur;
        if (b.cropBeforeNextn(inp_out_ids)) {
            cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
        }
    }

    // The pre-norm state for MTP, set even when extraction is off.
    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (b.cropAfterNextn(inp_out_ids)) {
        cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

/// Ports the constructor of `llama_model_mimo2`'s `graph_mtp`
/// (models/mimo2.cpp:250 @b1b770d90).
///
/// Mirrors MiMo's appended NextN block: normalize and fuse the token and
/// hidden inputs, run the decoder block, expose its state before the head
/// norm to the next draft step, then apply the shared output norm and LM
/// head. Converted checkpoints may store that shared norm as
/// `layer_out_norm`, so it stays in the fallback chain.
fn buildGraphMtp(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    llama.impl.assert(hp.n_layer_nextn > 0, "models/mimo2.cpp", 252, "hparams.n_layer_nextn > 0 && \"MIMO2 MTP requires n_layer_nextn > 0\"");

    const il: c_int = @as(c_int, @intCast(hp.nLayer())) + b.cparams.nextn_layer_offset;
    llama.impl.assert(b.cparams.nextn_layer_offset >= 0 and b.cparams.nextn_layer_offset < @as(c_int, @intCast(hp.n_layer_nextn)), "models/mimo2.cpp", 257, "cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn && \"nextn_layer_offset out of range [0, n_layer_nextn)\"");

    const layer = &model.layers[@intCast(il)];
    llama.impl.assert(layer.nextn.eh_proj != null, "models/mimo2.cpp", 260, "layer.nextn.eh_proj && \"MIMO2 MTP block missing nextn.eh_proj\"");
    llama.impl.assert(layer.nextn.enorm != null, "models/mimo2.cpp", 261, "layer.nextn.enorm   && \"MIMO2 MTP block missing nextn.enorm\"");
    llama.impl.assert(layer.nextn.hnorm != null, "models/mimo2.cpp", 262, "layer.nextn.hnorm   && \"MIMO2 MTP block missing nextn.hnorm\"");
    llama.impl.assert(layer.wqkv != null, "models/mimo2.cpp", 263, "layer.wqkv          && \"MIMO2 MTP requires fused attn_qkv\"");

    const n_head_l: i64 = hp.nHead(@intCast(il));
    const n_head_kv_l: i64 = hp.nHeadKv(@intCast(il));

    const v_scale = hp.f_attn_value_scale;

    const inp = std.heap.c_allocator.create(llama.graph.InpEmbdH) catch llama.impl.abort("models/mimo2.cpp", 270, "out of memory");
    inp.* = .{ .n_embd = hp.nEmbdInp(), .n_embd_state = hp.n_embd };

    inp.tokens = llama.impl.c.ggml_new_tensor_1d(ctx, llama.impl.c.GGML_TYPE_I32, b.n_tokens);
    llama.impl.c.ggml_set_input(inp.tokens);

    inp.embd = llama.impl.c.ggml_new_tensor_2d(ctx, llama.impl.c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    llama.impl.c.ggml_set_input(inp.embd);

    inp.h = b.buildInpEmbdState(inp, hp.n_embd);

    const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd.?;
    const h_input = inp.h.?;
    const tok_embd: *llama.impl.c.ggml_tensor = if (b.ubatch.token != null) llama.impl.c.ggml_get_rows(ctx, tok_embd_w, inp.tokens) else inp.embd.?;
    b.cb(tok_embd, "mtp_tok_embd", il);

    _ = b.res.addInput(llama.graph.InpEmbdH, inp);

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();
    const inp_attn = b.buildAttnInpKvIswa();

    const h_norm = b.buildNorm(h_input, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);

    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(cur, "mtp_eh_proj", il);

    const inpSA = cur;

    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    const qkv = b.buildLoraMm(layer.wqkv.?, cur, layer.wqkv_s);
    b.cb(qkv, "mtp_wqkv", il);

    const row_k = llama.impl.c.ggml_row_size(qkv.type, b.n_embd_head_k);
    const row_v = llama.impl.c.ggml_row_size(qkv.type, b.n_embd_head_v);
    const row_full = qkv.nb[1];
    const k_off = row_k * @as(usize, @intCast(n_head_l));
    const v_off = k_off + row_k * @as(usize, @intCast(n_head_kv_l));

    var Qcur: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, qkv, b.n_embd_head_k, n_head_l, b.n_tokens, row_k, row_full, 0);
    var Kcur: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, qkv, b.n_embd_head_k, n_head_kv_l, b.n_tokens, row_k, row_full, k_off);
    const Vcur: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_view_3d(ctx, qkv, b.n_embd_head_v, n_head_kv_l, b.n_tokens, row_v, row_full, v_off);

    Qcur = b.buildRope(Qcur, inp_pos, il);
    Kcur = b.buildRope(Kcur, inp_pos, il);

    b.cb(Qcur, "mtp_Qcur", il);
    b.cb(Kcur, "mtp_Kcur", il);
    b.cb(Vcur, "mtp_Vcur", il);

    cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, layer.attn_sinks, null, 1.0 / stdlib.libm.sqrtfInt(b.n_embd_head_k), il);
    b.cb(cur, "mtp_attn_out", il);

    if (v_scale != 0.0) {
        cur = llama.impl.c.ggml_scale(ctx, cur, v_scale);
        b.cb(cur, "mtp_attn_out_scaled", il);
    }

    const ffn_inp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "mtp_ffn_inp", il);

    cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "mtp_ffn_norm", il);

    llama.impl.assert(layer.ffn_gate != null and layer.ffn_down != null and layer.ffn_up != null, "models/mimo2.cpp", 345, "layer.ffn_gate && layer.ffn_down && layer.ffn_up && \"MIMO2 MTP requires dense FFN tensors\"");
    cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, layer.ffn_gate_b, null, layer.ffn_down, layer.ffn_down_b, null, null, .silu, .par, il);
    b.cb(cur, "mtp_ffn_out", il);

    cur = llama.impl.c.ggml_add(ctx, cur, ffn_inp);
    b.cb(cur, "mtp_post_ffn", il);

    cur = llama.impl.c.ggml_get_rows(ctx, cur, inp_out_ids);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    const head_norm_w = layer.nextn.shared_head_norm orelse (layer.layer_out_norm orelse model.output_norm);
    llama.impl.assert(head_norm_w != null, "models/mimo2.cpp", 365, "head_norm_w && \"MIMO2 MTP missing head norm fallback\"");
    cur = b.buildNorm(cur, head_norm_w, null, .rms, -1);
    b.cb(cur, "mtp_shared_head_norm", -1);

    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
    llama.impl.assert(head_w != null, "models/mimo2.cpp", 371, "head_w && \"MIMO2 MTP missing LM head fallback\"");
    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
