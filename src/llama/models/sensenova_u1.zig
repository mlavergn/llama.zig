//! SenseNova U1: a dense transformer whose Q and K are RMS-normed in two
//! halves, temporal and spatial, and rotated in three axes -- time over the
//! first half of the head at the model's base, height and width over a
//! quarter each at a spatial base.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/sensenova-u1.cpp` at v1.0.0
//! (`b1b770d90`): `llm_graph_input_pos_sensenova_u1`,
//! `llama_model_sensenova_u1`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`. `rope_freq_base_spatial`, a member
//! of the C++'s class (src/models/models.h:559 @b1b770d90), is a field of
//! `Model`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llm_graph_input_pos_sensenova_u1` (src/models/sensenova-u1.cpp:4
/// @b1b770d90). The C++ derives from `llm_graph_input_pos`; here the base
/// is a member.
const InpPosSensenovaU1 = struct {
    base: llama.graph.InpPos = .{ .n_pos_per_embd = 4 },

    /// Ports `llm_graph_input_pos_sensenova_u1`'s `set_input`
    /// (src/models/sensenova-u1.cpp:8 @b1b770d90).
    pub fn setInput(self: *InpPosSensenovaU1, ubatch_: ?*const llama.batch.Ubatch) void {
        const ubatch = ubatch_.?;
        const pos = self.base.pos;
        if (ubatch.pos == null or pos == null) return;

        const n_tokens: usize = ubatch.n_tokens;
        const n_pos_per_embd: usize = self.base.n_pos_per_embd;

        // U1 expects text-token positions in the temporal plane only:
        // `[t, 0, 0, 0]`. The generic M-RoPE input uses `[t, t, t, 0]`, so
        // U1 needs an input of its own.
        if (ubatch.token != null) {
            const pos_data = std.heap.c_allocator.alloc(llama.impl.llama_c.llama_pos, n_tokens * n_pos_per_embd) catch llama.impl.abort("models/sensenova-u1.cpp", 15, "out of memory");
            defer std.heap.c_allocator.free(pos_data);
            @memset(pos_data, 0);
            for (0..n_tokens) |i| {
                pos_data[i] = ubatch.pos[i];
            }
            llama.impl.llama_c.ggml_backend_tensor_set(pos, pos_data.ptr, 0, pos_data.len * llama.impl.llama_c.ggml_element_size(pos));
        } else {
            llama.impl.llama_c.ggml_backend_tensor_set(pos, ubatch.pos, 0, n_tokens * n_pos_per_embd * llama.impl.llama_c.ggml_element_size(pos));
        }
    }

    /// The base's `can_reuse`, which the C++ inherits.
    pub fn canReuse(self: *InpPosSensenovaU1, params: *const llama.graph.Params) bool {
        return self.base.canReuse(params);
    }
};

/// Ports `llama_model_sensenova_u1`'s `load_arch_hparams` (src/models/sensenova-u1.cpp:28 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    self.rope_freq_base_spatial = (try meta.getKey(f32, .rope_freq_base_spatial, true, exc)).?;
    if (hp.nEmbdHeadK(0) % 4 != 0 or hp.nEmbdHeadV(0) != hp.nEmbdHeadK(0)) {
        return exc.throw(.runtime_error, "SenseNova U1 requires equal Q/K/V head sizes divisible by four", .{});
    }
    self.type = switch (hp.nLayer()) {
        42 => .@"8b",
        else => .unknown,
    };
}

/// Ports `llama_model_sensenova_u1`'s `load_arch_tensors` (src/models/sensenova-u1.cpp:41 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const nr = llama.loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_gqa, l.n_embd_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
        layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_sensenova_u1`'s `build_arch_graph` (src/models/sensenova-u1.cpp:72 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    buildGraph(model, b);
}

/// The constructor's `normalize` lambda.
fn normalize(b: *const llama.graph.Builder, x_: *llama.impl.llama_c.ggml_tensor, weight: ?*llama.impl.llama_c.ggml_tensor) *llama.impl.llama_c.ggml_tensor {
    const x: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_rms_norm(b.ctx0, x_, b.hparams.f_norm_rms_eps);

    return llama.impl.llama_c.ggml_mul(b.ctx0, x, weight);
}

/// The constructor's `normalize_and_rotate` lambda: normalize the temporal
/// and spatial halves separately, then rotate time, height and width.
fn normalizeAndRotate(model: *const llama.model.Model, b: *const llama.graph.Builder, pos: *const [3]*llama.impl.llama_c.ggml_tensor, n_embd_head: i64, x_: *llama.impl.llama_c.ggml_tensor, weight: *llama.impl.llama_c.ggml_tensor) *llama.impl.llama_c.ggml_tensor {
    const ctx = b.ctx0;
    const heads = x_.ne[1];
    var x: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_reshape_4d(ctx, x_, @divTrunc(n_embd_head, 2), 2, heads, b.n_tokens);
    x = normalize(b, x, llama.impl.llama_c.ggml_reshape_2d(ctx, weight, @divTrunc(n_embd_head, 2), 2));
    x = llama.impl.llama_c.ggml_reshape_3d(ctx, x, n_embd_head, heads, b.n_tokens);

    const head: c_int = @intCast(n_embd_head);
    var axis: c_int = 0;
    while (axis < 3) : (axis += 1) {
        const dims: c_int = if (axis == 0) @divTrunc(head, 2) else @divTrunc(head, 4);
        const offset: c_int = if (axis == 0) 0 else @divTrunc(head, 2) + @divTrunc((axis - 1) * head, 4);
        const base: f32 = if (axis == 0) b.freq_base else model.rope_freq_base_spatial;
        x = llama.impl.llama_c.ggml_rope_ext(ctx, x, pos[@intCast(axis)], null, dims, llama.impl.llama_c.GGML_ROPE_TYPE_NEOX, b.n_ctx_orig, base, 1.0, 0.0, 1.0, 32.0, 1.0);
        _ = llama.impl.llama_c.ggml_rope_set_offset(x, offset);
    }
    return x;
}

/// Ports the constructor of `llama_model_sensenova_u1`'s `graph`
/// (src/models/sensenova-u1.cpp:76 @b1b770d90).
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/sensenova-u1.cpp", 81, "n_embd_head == hparams.n_embd_head_k()");

    var inpL: *llama.impl.llama_c.ggml_tensor = b.buildInpEmbd(model.tok_embd.?);

    const inp = std.heap.c_allocator.create(InpPosSensenovaU1) catch llama.impl.abort("models/sensenova-u1.cpp", 95, "out of memory");
    inp.* = .{};
    const inp_pos: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_new_tensor_1d(ctx, llama.impl.llama_c.GGML_TYPE_I32, b.n_tokens * hp.nPosPerEmbd());
    inp.base.pos = inp_pos;
    llama.impl.llama_c.ggml_set_input(inp_pos);
    _ = b.res.addInput(InpPosSensenovaU1, inp);

    var pos: [3]*llama.impl.llama_c.ggml_tensor = undefined;
    for (&pos, 0..) |*p, axis| {
        p.* = llama.impl.llama_c.ggml_view_1d(ctx, inp_pos, b.n_tokens, axis * @as(usize, @intCast(b.n_tokens)) * @sizeOf(i32));
    }

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = inpL;

        var inpSA = inpL;

        // Norm.
        var cur = normalize(b, inpL, layer.attn_norm);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = normalizeAndRotate(model, b, &pos, n_embd_head, Qcur, layer.attn_q_norm.?);
            Kcur = normalizeAndRotate(model, b, &pos, n_embd_head, Kcur, layer.attn_k_norm.?);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.llama_c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        cur = normalize(b, ffn_inp, layer.ffn_norm);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = normalize(b, cur, model.output_norm);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
