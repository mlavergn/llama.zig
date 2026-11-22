//! DeepSeek-OCR's language model: DeepSeek-V2's blocks with plain
//! multi-head attention in place of the latent one -- separate Q, K and V
//! projections and NeoX RoPE over the whole head -- leading dense blocks,
//! then a mixture of experts with shared experts.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/deepseek2ocr.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_deepseek2ocr`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`. Its graph is
//! `llama_model_deepseek2`'s, which takes the OCR attention when the
//! architecture is this one; `deepseek2.zig` holds it.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_deepseek2ocr`'s `load_arch_hparams` (models/deepseek2ocr.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;

    // As DeepSeek-V2, without MLA.
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    if (try meta.getKey(u32, .leading_dense_block_count, false, exc)) |v| hp.n_layer_dense_lead = v;
    _ = try meta.getKeyOrArr(u32, .expert_feed_forward_length, &hp.n_ff_exp_arr, hp.n_layer_all, true, exc);
    hp.n_expert_shared = (try meta.getKey(u32, .expert_shared_count, true, exc)).?;
    if (try meta.getKey(f32, .expert_weights_scale, false, exc)) |v| hp.expert_weights_scale = v;
    if (try meta.getKey(bool, .expert_weights_norm, false, exc)) |v| hp.expert_weights_norm = v;
    if (try meta.getKey(u32, .expert_gating_func, false, exc)) |v| hp.expert_gating_func = v;

    if (hp.expert_gating_func == @backingInt(llama.hparams.ExpertGatingFunc.none)) {
        hp.expert_gating_func = @backingInt(llama.hparams.ExpertGatingFunc.softmax);
    }

    self.type = switch (hp.nLayer()) {
        12 => .@"3b",
        else => .unknown,
    };
}

/// Ports `llama_model_deepseek2ocr`'s `load_arch_tensors` (models/deepseek2ocr.cpp:23 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const n_expert_shared: i64 = hp.n_expert_shared;
    const nr = llama.loader.tensor_not_required;

    // As DeepSeek-V2, without MLA.
    const n_ff_exp: i64 = hp.nFfExp(0);

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    // The output, or failing that the token embeddings (tied embeddings).
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, llama.loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, l.n_embd, l.n_embd, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);

        // Norms.
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (i < @as(i32, @bitCast(hp.n_layer_dense_lead))) {
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        } else {
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, nr, exc);

            if (l.n_expert == 0) {
                return exc.throw(.runtime_error, "n_expert must be > 0", .{});
            }
            if (l.n_expert_used == 0) {
                return exc.throw(.runtime_error, "n_expert_used must be > 0", .{});
            }

            // The routed experts.
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
            try self.createTensorGateUpExps(layer, i, l.n_embd, n_ff_exp, l.n_expert, 0, exc);

            // The shared experts.
            layer.ffn_gate_shexp = try self.createTensor(.{ .tensor = .ffn_gate_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
            layer.ffn_down_shexp = try self.createTensor(.{ .tensor = .ffn_down_shexp, .suffix = "weight", .bid = i }, &.{ n_ff_exp * n_expert_shared, l.n_embd }, 0, exc);
            layer.ffn_up_shexp = try self.createTensor(.{ .tensor = .ffn_up_shexp, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_ff_exp * n_expert_shared }, 0, exc);
        }
    }
}

/// Ports `llama_model_deepseek2ocr`'s `get_rope_params` (models/deepseek2ocr.cpp:77 @b1b770d90).
///
/// Parameters:
/// - `model`: the model.
/// - `cparams`: the context's parameters.
/// - `il`: the layer.
///
/// Return: the RoPE of layer `il`, for the graph and the K-shift alike.
pub fn getRopeParams(model: *const llama.model.Model, cparams: *const llama.cparams.Cparams, il: u32) llama.model.RopeParams {
    var res = model.getRopeParamsBase(cparams, il);

    res.n_rot = @intCast(model.hparams.nEmbdHeadK(il));
    res.n_offs = 0;
    res.type = llama.impl.c.LLAMA_ROPE_TYPE_NEOX;
    res.n_ctx_orig = 0;
    res.freq_scale = 1.0;
    res.ext_factor = 0.0;
    res.attn_factor = 1.0;
    res.beta_fast = 0.0;
    res.beta_slow = 0.0;
    res.factors = null;

    return res;
}

/// Ports `llama_model_deepseek2ocr`'s `build_arch_graph` (models/deepseek2ocr.cpp:94
/// @b1b770d90): DeepSeek-V2's main graph, whatever the graph type.
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    llama.models.deepseek2.buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
