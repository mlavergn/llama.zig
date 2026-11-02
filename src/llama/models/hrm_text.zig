//! HRM-Text: alternating low and high transformer stacks over the same
//! token stream. The low stack runs `l_cycles` times per high pass, each
//! time on the high state plus its own; the high stack then runs once on
//! the sum, and the whole repeats `h_cycles` times. Every stack pass has
//! its own KV-cache slots, so the GGUF's block count is the expanded slot
//! count, and the slots after the first low and first high pass alias the
//! same physical layers.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/hrm-text.cpp` at v0.5.0
//! (`7fe450e19`): `llama_model_hrm_text`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph` with
//! `build_stack`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

const src = "models/hrm-text.cpp";

/// Ports `llama_model_hrm_text`'s `load_arch_hparams` (models/hrm-text.cpp:6 @7fe450e19).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(f32, .embedding_scale, false, exc)) |v| hp.f_embedding_scale = v;

    hp.n_hrm_layers_per_stack = (try meta.getKey(u32, .hrm_layers_per_stack, true, exc)).?;
    hp.n_hrm_h_cycles = (try meta.getKey(u32, .hrm_h_cycles, true, exc)).?;
    hp.n_hrm_l_cycles = (try meta.getKey(u32, .hrm_l_cycles, true, exc)).?;

    // Prefix-LM prefill is not implemented (causal attention only); the
    // flag is kept for the round trip.
    if (try meta.getKey(bool, .hrm_prefix_lm, false, exc)) |v| hp.hrm_prefix_lm = v;

    impl.assert(hp.n_hrm_layers_per_stack > 0, src, 17, "hparams.n_hrm_layers_per_stack > 0");
    impl.assert(hp.n_hrm_h_cycles > 0, src, 18, "hparams.n_hrm_h_cycles > 0");
    impl.assert(hp.n_hrm_l_cycles > 0, src, 19, "hparams.n_hrm_l_cycles > 0");

    // The GGUF block count is the expanded cache-slot count.
    const n_slot: u32 = hp.n_hrm_layers_per_stack *% hp.n_hrm_h_cycles *% (hp.n_hrm_l_cycles +% 1);
    impl.assert(hp.nLayer() == n_slot, src, 23, "hparams.n_layer() == n_slot");

    self.type = switch (hp.n_embd) {
        1536 => .@"1b",
        else => .unknown,
    };
}

/// Ports `llama_model_hrm_text`'s `load_arch_tensors` (models/hrm-text.cpp:34 @7fe450e19).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // The output, or the input token embedding when there is none.
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_not_required, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    self.hrm_z_l_init = try self.createTensor(.{ .tensor = .hrm_z_l_init }, &.{l.n_embd}, 0, exc);

    const lps: i32 = @intCast(hp.n_hrm_layers_per_stack);
    const n_l: i32 = @intCast(hp.n_hrm_l_cycles);
    const n_h: i32 = @intCast(hp.n_hrm_h_cycles);

    // Blocks [0, lps) hold the low stack, blocks [lps, 2*lps) the high
    // stack. The first low and high passes create the layers; the later
    // passes alias them.
    const l_first: i32 = 0;
    const h_first: i32 = n_l * lps;

    var h: i32 = 0;
    while (h < n_h) : (h += 1) {
        var lc: i32 = 0;
        while (lc < n_l + 1) : (lc += 1) {
            const slot_base = (h * (n_l + 1) + lc) * lps;
            const blk_base: i32 = if (lc == n_l) lps else 0;

            if (h > 0 or (lc > 0 and lc < n_l)) {
                // An alias pass: these cache slots hold the same layers as
                // the first passes.
                const src_base = if (lc == n_l) h_first else l_first;
                var il: i32 = 0;
                while (il < lps) : (il += 1) {
                    self.layers[@intCast(slot_base + il)] = self.layers[@intCast(src_base + il)];
                }
                continue;
            }

            var il: i32 = 0;
            while (il < lps) : (il += 1) {
                const layer = &self.layers[@intCast(slot_base + il)];
                const bid = blk_base + il;

                try self.createTensorQkv(layer, bid, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);

                // The sigmoid attention gate, applied to the attention
                // output before o_proj.
                layer.wqkv_gate = try self.createTensor(.{ .tensor = .attn_gate, .suffix = "weight", .bid = bid }, &.{ l.n_embd, l.n_embd_head_k * l.n_head }, 0, exc);
                layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = bid }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

                layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = bid }, &.{ l.n_embd, l.n_ff }, 0, exc);
                layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = bid }, &.{ l.n_ff, l.n_embd }, 0, exc);
                layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = bid }, &.{ l.n_embd, l.n_ff }, 0, exc);
            }
        }
    }
}

/// Ports `llama_model_hrm_text`'s `build_arch_graph` (models/hrm-text.cpp:88 @7fe450e19).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports `llama_model_hrm_text::graph`'s `build_stack` (models/hrm-text.cpp:93
/// @7fe450e19): one stack invocation, `lps` pre-norm decoder layers in the
/// cache slots from `slot_base`, then the parameterless final norm.
fn buildStack(model: *const Model, b: *const graph.Builder, inp_attn: *graph.InpAttnKv, inp_pos: *Tensor, cur_: *Tensor, slot_base: c_int) *Tensor {
    const ctx = b.ctx0;
    var cur = cur_;
    const kq_scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(b.n_embd_head_k));

    const lps: c_int = @intCast(model.hparams.n_hrm_layers_per_stack);

    var il: c_int = 0;
    while (il < lps) : (il += 1) {
        const s = slot_base + il;
        const layer = &model.layers[@intCast(s)];

        var inpSA = cur;

        cur = b.buildNorm(cur, null, null, .rms, s);
        b.cb(cur, "attn_norm", s);

        // Sigmoid-gated self-attention, shaped as Qwen3-Next's attention
        // layers.
        {
            var gate = b.buildLoraMm(layer.wqkv_gate.?, cur, null);
            b.cb(gate, "attn_gate_proj", s);

            const qkv = b.buildQkv(layer, cur, b.n_embd_head_k, b.n_head, b.n_head_kv, s);
            var Qcur = qkv.q;
            var Kcur = qkv.k;

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            b.cb(Qcur, "Qcur", s);

            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            b.cb(Kcur, "Kcur", s);

            cur = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, qkv.v, null, null, null, kq_scale, s);
            b.cb(cur, "attn_pregate", s);

            gate = c.ggml_sigmoid(ctx, gate);
            b.cb(gate, "attn_gate_sigmoid", s);

            cur = c.ggml_mul(ctx, cur, gate);
            b.cb(cur, "attn_gated", s);

            cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
            b.cb(cur, "attn_out", s);
        }

        cur = c.ggml_add(ctx, cur, inpSA);
        b.cb(cur, "attn_add", s);

        inpSA = cur;
        cur = b.buildNorm(cur, null, null, .rms, s);
        b.cb(cur, "ffn_norm", s);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, s);
        b.cb(cur, "ffn_out", s);

        cur = c.ggml_add(ctx, cur, inpSA);
        b.cb(cur, "ffn_add", s);

        cur = b.buildCvec(cur, s);
        b.cb(cur, "l_out", s);
    }

    cur = b.buildNorm(cur, null, null, .rms, slot_base);
    b.cb(cur, "stack_norm", slot_base);

    return cur;
}

/// Ports the constructor of `llama_model_hrm_text`'s `graph`
/// (models/hrm-text.cpp:168 @7fe450e19).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = &model.hparams;

    // {n_embd, n_tokens}, scaled by f_embedding_scale inside buildInpEmbd.
    var zH: *Tensor = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    // The learned low-cycle state is [n_embd]; the binary ops broadcast it
    // over [n_embd, n_tokens].
    var zL: *Tensor = model.hrm_z_l_init.?;

    const n_l: u32 = hp.n_hrm_l_cycles;
    const lps: u32 = hp.n_hrm_layers_per_stack;

    var h: u32 = 0;
    while (h < hp.n_hrm_h_cycles) : (h += 1) {
        var lc: u32 = 0;
        while (lc < n_l) : (lc += 1) {
            const slot_base: c_int = @bitCast((h *% (n_l +% 1) +% lc) *% lps);

            zL = buildStack(model, b, inp_attn, inp_pos, c.ggml_add(ctx, zH, zL), slot_base);
        }

        const slot_base: c_int = @bitCast((h *% (n_l +% 1) +% n_l) *% lps);

        zH = buildStack(model, b, inp_attn, inp_pos, c.ggml_add(ctx, zH, zL), slot_base);
    }

    var cur = zH;

    cur = c.ggml_get_rows(ctx, cur, inp_out_ids);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

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
