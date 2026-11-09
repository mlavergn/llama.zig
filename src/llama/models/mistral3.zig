//! Mistral 3: Llama's decoder blocks, with Llama 4's optional attention
//! temperature tuning and YaRN parameters read from the file, and Voxtral
//! Realtime's optional adaptive RMS norm before the feed-forward, conditioned
//! on the output delay.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mistral3.cpp` at v0.8.0 (`8642541b2`):
//! `llama_model_mistral3`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, its `input_t_cond`, and its `graph`.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const batch = @import("../batch.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const Ubatch = batch.Ubatch;
const allocator = impl.allocator;

/// Ports `llama_model_mistral3`'s `load_arch_hparams` (models/mistral3.cpp:3 @cd959bae9).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(f32, .attention_temperature_scale, false, exc)) |v| hp.f_attn_temp_scale = v;

    if (try meta.getKey(f32, .rope_scaling_yarn_beta_fast, false, exc)) |v| hp.yarn_beta_fast = v;
    if (try meta.getKey(f32, .rope_scaling_yarn_beta_slow, false, exc)) |v| hp.yarn_beta_slow = v;
    if (try meta.getKey(f32, .rope_scaling_yarn_log_mul, false, exc)) |v| hp.rope_yarn_log_mul = v;

    if (try meta.getKey(u32, .realtime_delay_tokens, false, exc)) |v| hp.n_tokens_delay = v;

    hp.f_attn_temp_offset = 0.0;

    // TODO: maybe add n_attn_temp_floor_scale as a separate key?
    if (hp.f_attn_temp_scale != 0.0) {
        hp.n_attn_temp_floor_scale = hp.n_ctx_orig_yarn;
        if (hp.n_attn_temp_floor_scale == 0) {
            return exc.throw(.runtime_error, "invalid n_ctx_orig_yarn for attention temperature scaling", .{});
        }
    }

    self.type = switch (hp.nLayer()) {
        26 => .@"3b",
        34 => .@"8b",
        40 => .@"14b",
        else => .unknown,
    };
}

/// Ports `llama_model_mistral3`'s `load_arch_tensors` (models/mistral3.cpp:31 @cd959bae9).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    // Without an output, the input token embedding is used.
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        // Optional bias tensors.
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        // The optional adaptive RMS norm (Voxtral Realtime); the hidden size
        // is taken from the tensor itself.
        const ada_name = (loader.Tn{ .tensor = .ffn_norm_ada_down, .suffix = "weight", .bid = i }).str();
        if (ml.getTensorMeta(ada_name.slice())) |meta| {
            const n_ada: i64 = meta.ne[1];
            layer.ffn_norm_ada_down = try self.createTensor(.{ .tensor = .ffn_norm_ada_down, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ada }, 0, exc);
            layer.ffn_norm_ada_up = try self.createTensor(.{ .tensor = .ffn_norm_ada_up, .suffix = "weight", .bid = i }, &.{ n_ada, l.n_embd }, 0, exc);
        }

        const rope_dup: u32 = if (i != 0) dup else 0;
        if (hp.rope_scaling_type_train == c.LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long = try self.createTensor(.{ .tensor = .rope_factors_long, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
            layer.rope_short = try self.createTensor(.{ .tensor = .rope_factors_short, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
        } else {
            layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);
        }

        if (l.n_expert == 0) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

            // Optional MLP bias.
            layer.ffn_gate_b = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "bias", .bid = i }, &.{l.n_ff}, nr, exc);
            layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
            layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{l.n_ff}, nr, exc);
        } else {
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, nr, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, 0, exc);

            // Granite MoE Shared.
            if (hp.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, 0, exc);
                layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_shexp }, 0, exc);
                layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ hp.n_ff_shexp, l.n_embd }, 0, exc);
            }
        }
    }
}

/// Ports `llama_model_mistral3`'s `input_t_cond` (models/mistral3.cpp:99
/// @cd959bae9): the sinusoidal embedding of the output delay, constant for
/// the whole session.
const InputTCond = struct {
    /// F32 `[n_embd]`.
    t_cond: ?*Tensor = null,
    data: []f32,

    /// `input_t_cond`'s constructor (models/mistral3.cpp:101 @cd959bae9).
    fn create(n_delay: u32, n_embd: i64) *InputTCond {
        const self = allocator.create(InputTCond) catch oom();
        const data = allocator.alloc(f32, @intCast(n_embd)) catch oom();
        @memset(data, 0.0);
        const half: i64 = @divTrunc(n_embd, 2);
        const log_10000 = log(10000.0);
        var i: i64 = 0;
        while (i < half) : (i += 1) {
            const freq: f64 = impl.exp(-log_10000 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half))) * @as(f64, @floatFromInt(n_delay));
            const sc = sinCos(freq);
            data[@intCast(i)] = @floatCast(sc.cos);
            data[@intCast(half + i)] = @floatCast(sc.sin);
        }
        self.* = .{ .data = data };
        return self;
    }

    pub fn deinit(self: *InputTCond) void {
        allocator.free(self.data);
    }

    /// `input_t_cond`'s `set_input` (models/mistral3.cpp:110 @cd959bae9).
    pub fn setInput(self: *InputTCond, _: ?*const Ubatch) void {
        c.ggml_backend_tensor_set(self.t_cond, self.data.ptr, 0, c.ggml_nbytes(self.t_cond));
    }

    /// `input_t_cond`'s `can_reuse` (models/mistral3.cpp:116 @cd959bae9):
    /// the input does not depend on the batch.
    pub fn canReuse(_: *InputTCond, _: *const graph.Params) bool {
        return true;
    }
};

fn oom() noreturn {
    impl.abort("models/mistral3.cpp", 0, "out of memory");
}

extern fn log(x: f64) f64;
extern fn sin(x: f64) f64;
extern fn cos(x: f64) f64;

/// Mirrors the `__double2` that `__sincos_stret` returns (math.h).
const Double2 = extern struct { sin: f64, cos: f64 };

extern fn __sincos_stret(x: f64) Double2;

/// `sin(x)` and `cos(x)` as clang emits them when both appear, as
/// `cpu/ops/common.zig`'s `sinCos` does for `f32`: the reference
/// `mistral3.o` imports `___sincos_stret` on Darwin and neither `_sin` nor
/// `_cos`; elsewhere `-fmath-errno` keeps the pair two calls.
fn sinCos(x: f64) Double2 {
    if (comptime builtin.os.tag.isDarwin()) {
        return __sincos_stret(x);
    }
    return .{ .sin = sin(x), .cos = cos(x) };
}

/// Ports `llama_model_mistral3`'s `build_arch_graph` (models/mistral3.cpp:126
/// @cd959bae9) and the constructor of its `graph` (models/mistral3.cpp:130
/// @cd959bae9).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/mistral3.cpp", 133, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/mistral3.cpp", 134, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    // Optional temperature tuning.
    var inp_attn_scale: ?*Tensor = null;
    if (hp.f_attn_temp_scale != 0.0) {
        inp_attn_scale = b.buildInpAttnScale();
    }

    const inp_attn = b.buildAttnInpKv();

    // (Optional) the time conditioning of the adaptive RMS norm.
    var t_cond: ?*Tensor = null;
    if (model.layers[0].ffn_norm_ada_down != null) {
        const inp = InputTCond.create(hp.n_tokens_delay, b.n_embd);
        t_cond = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, b.n_embd);
        c.ggml_set_input(t_cond);
        b.cb(t_cond.?, "inp_t_cond", -1);
        inp.t_cond = t_cond;
        _ = b.res.addInput(InputTCond, inp);
    }

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        // Norm.
        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        // Self-attention.
        {
            // Llama 3's RoPE frequency factors; null for Llama 2 and others.
            const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));

            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            if (inp_attn_scale) |scale| {
                // Llama 4's temperature scaling.
                Qcur = c.ggml_mul(ctx, Qcur, scale);
                b.cb(Qcur, "Qcur_attn_temp_scaled", il);
            }

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);
        }
        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        if (layer.ffn_gate_inp == null) {
            // The feed-forward network, not a mixture of experts.
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            if (t_cond) |tc| {
                // The adaptive RMS norm: scale the normalized input by
                // (1 + mlp(t_cond)).
                const ada = b.buildFfn(tc, layer.ffn_norm_ada_down, null, null, null, null, null, layer.ffn_norm_ada_up, null, null, null, .gelu_erf, .seq, il);
                cur = c.ggml_add(ctx, cur, c.ggml_mul(ctx, cur, ada));
                b.cb(cur, "ffn_norm_ada", il);
            }

            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, layer.ffn_up_s, layer.ffn_gate, layer.ffn_gate_b, layer.ffn_gate_s, layer.ffn_down, layer.ffn_down_b, layer.ffn_down_s, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The mixture of experts.
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildMoeFfn(cur, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = layer.ffn_gate_exps,
                .down_exps = layer.ffn_down_exps.?,
                .n_expert = b.n_expert,
                .n_expert_used = b.n_expert_used,
                .type_op = .silu,
                .norm_w = true,
                .w_scale = hp.expert_weights_scale,
                .gating_op = .softmax,
                .up_exps_s = layer.ffn_up_exps_s,
                .gate_exps_s = layer.ffn_gate_exps_s,
                .down_exps_s = layer.ffn_down_exps_s,
            }, il);
            b.cb(cur, "ffn_moe_out", il);
        }
        cur = c.ggml_add(ctx, cur, ffn_inp);
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
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
