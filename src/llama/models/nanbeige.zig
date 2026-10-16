//! Nanbeige: Llama's decoder blocks, optionally run as a loop -- the
//! physical layers repeated `n_loops` times over shared weights, each pass
//! with its own KV cache slots, and the output norm applied between passes.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/nanbeige.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_nanbeige`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`. The class's own members are
//! `Model.nanbeige`.

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

/// Ports `llama_model_nanbeige`'s `load_arch_hparams` (models/nanbeige.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    const nb = &self.nanbeige;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    const n_loops_u: u32 = (try meta.getKey(u32, .num_loops, false, exc)) orelse 1;
    impl.assert(n_loops_u >= 1, "models/nanbeige.cpp", 8, "n_loops_u >= 1");

    nb.skip_loop_final_norm = false;
    if (try meta.getKey(bool, .skip_loop_final_norm, false, exc)) |v| nb.skip_loop_final_norm = v;

    nb.n_layer_phys = @bitCast(hp.nLayer());

    // Bound-check before casting: a signed int multiply can overflow and
    // bypass the guard.
    const phys_sz: usize = @bitCast(@as(isize, nb.n_layer_phys));
    impl.assert(phys_sz *% @as(usize, n_loops_u) <= hparams_mod.max_layers, "models/nanbeige.cpp", 16, "(size_t) n_layer_phys * (size_t) n_loops_u <= (size_t) LLAMA_MAX_LAYERS");
    nb.n_loops = @bitCast(n_loops_u);

    // Expand the logical layer count before load_tensors allocates the
    // layers and the KV cache.
    if (nb.n_loops > 1) {
        var j: c_int = 1;
        while (j < nb.n_loops) : (j += 1) {
            var i: c_int = 0;
            while (i < nb.n_layer_phys) : (i += 1) {
                const dst: usize = @intCast(i + j * nb.n_layer_phys);
                const src: usize = @intCast(i);
                hp.n_head_arr[dst] = hp.n_head_arr[src];
                hp.n_head_kv_arr[dst] = hp.n_head_kv_arr[src];
                hp.n_ff_arr[dst] = hp.n_ff_arr[src];
                hp.is_swa_impl[dst] = hp.is_swa_impl[src];
                hp.is_recr_impl[dst] = hp.is_recr_impl[src];
            }
        }
        hp.n_layer_all = @truncate(phys_sz *% @as(usize, @bitCast(@as(isize, nb.n_loops))));
    }

    self.type = .unknown;
}

/// Ports `llama_model_nanbeige`'s `load_arch_tensors` (models/nanbeige.cpp:37 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const nb = &self.nanbeige;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    const n_phys: c_int = if (nb.n_layer_phys > 0) nb.n_layer_phys else l.n_layer;
    var i: i32 = 0;
    while (i < n_phys) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd_head_k * l.n_head, l.n_embd_k_gqa, l.n_embd_v_gqa, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd_head_k * l.n_head, l.n_embd }, 0, exc);

        const rope_dup: u32 = if (i != 0) dup else 0;
        layer.rope_freqs = try self.createTensor(.{ .tensor = .rope_freqs, .suffix = "weight", .bid = i }, &.{@divTrunc(l.n_rot, 2)}, nr | rope_dup, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }

    // Share the physical weights across loops; each slot still has its own
    // KV index.
    if (nb.n_loops > 1) {
        var j: c_int = 1;
        while (j < nb.n_loops) : (j += 1) {
            var k: c_int = 0;
            while (k < n_phys) : (k += 1) {
                self.layers[@intCast(k + j * n_phys)] = self.layers[@intCast(k)];
            }
        }
    }
}

/// Ports `llama_model_nanbeige`'s `build_arch_graph` (models/nanbeige.cpp:76
/// @c1d0e7a00) and the constructor of its `graph` (models/nanbeige.cpp:80
/// @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const nb = &model.nanbeige;

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/nanbeige.cpp", 85, "n_embd_head == hparams.n_embd_head_k()");

    const n_phys: c_int = if (nb.n_layer_phys > 0) nb.n_layer_phys else @intCast(b.n_layer);
    const n_loops: c_int = if (nb.n_loops > 0) nb.n_loops else 1;

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        {
            const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));

            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
            Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
            b.cb(cur, "attn_out", il);
        }

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }

        const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, layer.ffn_up_s, layer.ffn_gate, layer.ffn_gate_b, layer.ffn_gate_s, layer.ffn_down, layer.ffn_down_b, layer.ffn_down_s, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        inpL = cur;

        if (n_loops > 1 and @rem(il + 1, n_phys) == 0 and (il + 1) < b.n_layer and !nb.skip_loop_final_norm) {
            cur = b.buildNorm(inpL, model.output_norm, null, .rms, il);
            b.cb(cur, "loop_norm", il);
            inpL = cur;
        }
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
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
