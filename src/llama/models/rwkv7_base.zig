//! The RWKV-7 blocks RWKV-7 and ARWKV-7 share: the time mix, with its
//! low-rank decay, in-context learning rate and value residual, and
//! RWKV-7's channel mix.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/rwkv7-base.cpp` at v0.4.1
//! (`b29c606e2`): `llm_build_rwkv7_base`'s `build_rwkv7_channel_mix` and
//! `build_rwkv7_time_mix`. Its constructor only stores the model, which the
//! port passes to each function instead.

const std = @import("std");
const impl = @import("../impl.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const arch_mod = @import("../arch.zig");
const batch = @import("../batch.zig");
const memory_recurrent = @import("../memory_recurrent.zig");

const c = impl.c;
const Model = model_mod.Model;
const Layer = model_mod.Layer;
const Tensor = c.ggml_tensor;

/// A C++ `double` literal passed for a `float` parameter: rounded to
/// double first, then to float, as the C++ converts it.
fn doubleToFloat(comptime x: f64) f32 {
    return @floatCast(x);
}

/// Ports `llm_build_rwkv7_base`'s `build_rwkv7_channel_mix`
/// (models/rwkv7-base.cpp:9 @b29c606e2): the squared-ReLU feed-forward.
pub fn buildRwkv7ChannelMix(b: *const graph.Builder, layer: *const Layer, cur_: *Tensor, x_prev: *Tensor, arch: arch_mod.Arch) *Tensor {
    const ctx = b.ctx0;
    var cur = cur_;
    const sx: *Tensor = c.ggml_sub(ctx, x_prev, cur);
    switch (arch) {
        .rwkv7 => {
            const xk: *Tensor = c.ggml_add(ctx, c.ggml_mul(ctx, sx, layer.channel_mix_lerp_k), cur);

            const k: *Tensor = c.ggml_sqr(ctx, c.ggml_relu(ctx, b.buildLoraMm(layer.channel_mix_key.?, xk, null)));

            cur = b.buildLoraMm(layer.channel_mix_value.?, k, null);
        },
        else => impl.abort("models/rwkv7-base.cpp", 25, "fatal error"),
    }
    return cur;
}

/// Ports `llm_build_rwkv7_base`'s `build_rwkv7_time_mix`
/// (models/rwkv7-base.cpp:30 @b29c606e2): the WKV-7 recurrence.
/// `first_layer_value` is the first layer's value, set by the first call
/// and mixed back into every later layer's.
pub fn buildRwkv7TimeMix(model: *const Model, b: *const graph.Builder, inp: *const graph.InpRs, cur_: *Tensor, x_prev: *Tensor, first_layer_value: *?*Tensor, ubatch: *const batch.Ubatch, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const mctx_cur = b.mctx.?.cast(memory_recurrent.RecurrentContext);

    const n_tokens: i64 = ubatch.n_tokens;
    const n_seqs: i64 = ubatch.n_seqs;
    const n_embd: i64 = hp.n_embd;
    const head_size: i64 = hp.wkv_head_size;
    const head_count: i64 = @divTrunc(n_embd, head_size);
    const n_seq_tokens: i64 = ubatch.n_seq_tokens;

    const kv_head: usize = mctx_cur.getHead();

    const layer = &model.layers[@intCast(il)];

    const has_gating = layer.time_mix_g1 != null and layer.time_mix_g2 != null;

    const f32_size: usize = @sizeOf(f32);
    const tok: usize = @intCast(n_embd * n_tokens);

    var sx: *Tensor = c.ggml_sub(ctx, x_prev, cur_);
    const dummy: *Tensor = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, n_embd, n_seq_tokens, n_seqs, if (has_gating) 6 else 5);
    sx = c.ggml_repeat(ctx, sx, dummy);

    const xxx: *Tensor = c.ggml_add(ctx, c.ggml_mul(ctx, sx, layer.time_mix_lerp_fused), cur_);

    const xr: *Tensor = c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], 0);
    const xw: *Tensor = c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * f32_size);
    const xk: *Tensor = c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 2 * f32_size);
    const xv: *Tensor = c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 3 * f32_size);
    const xa: *Tensor = c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 4 * f32_size);
    const xg: ?*Tensor = if (has_gating) c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 5 * f32_size) else null;

    var r = b.buildLoraMm(layer.time_mix_receptance.?, xr, null);
    var w: *Tensor = c.ggml_add(ctx, c.ggml_mul_mat(ctx, layer.time_mix_w2, c.ggml_tanh(ctx, c.ggml_mul_mat(ctx, layer.time_mix_w1, xw))), layer.time_mix_w0);
    w = c.ggml_exp(ctx, c.ggml_scale(ctx, c.ggml_sigmoid(ctx, w), doubleToFloat(-0.606531)));

    var k = b.buildLoraMm(layer.time_mix_key.?, xk, null);
    var v = b.buildLoraMm(layer.time_mix_value.?, xv, null);
    if (first_layer_value.* == null) {
        first_layer_value.* = v;
    } else {
        // Add the first layer's value as a residual connection.
        v = c.ggml_add(ctx, v, c.ggml_mul(ctx, c.ggml_sub(ctx, first_layer_value.*, v), c.ggml_sigmoid(ctx, c.ggml_add(ctx, c.ggml_mul_mat(ctx, layer.time_mix_v2, c.ggml_mul_mat(ctx, layer.time_mix_v1, xv)), layer.time_mix_v0))));
    }
    var g: ?*Tensor = null;
    if (layer.time_mix_g1 != null and layer.time_mix_g2 != null) {
        g = c.ggml_mul_mat(ctx, layer.time_mix_g2, c.ggml_sigmoid(ctx, c.ggml_mul_mat(ctx, layer.time_mix_g1, xg)));
    }
    var a: *Tensor = c.ggml_sigmoid(ctx, c.ggml_add(ctx, c.ggml_mul_mat(ctx, layer.time_mix_a2, c.ggml_mul_mat(ctx, layer.time_mix_a1, xa)), layer.time_mix_a0));

    var kk: *Tensor = c.ggml_reshape_3d(ctx, c.ggml_mul(ctx, k, layer.time_mix_k_k), head_size, head_count, n_tokens);
    kk = c.ggml_l2_norm(ctx, kk, doubleToFloat(1e-12));

    const ka: *Tensor = c.ggml_mul(ctx, k, layer.time_mix_k_a);
    k = c.ggml_add(ctx, k, c.ggml_sub(ctx, c.ggml_mul(ctx, a, ka), ka));

    r = c.ggml_reshape_3d(ctx, r, head_size, head_count, n_tokens);
    w = c.ggml_reshape_3d(ctx, w, head_size, head_count, n_tokens);
    k = c.ggml_reshape_3d(ctx, k, head_size, head_count, n_tokens);
    v = c.ggml_reshape_3d(ctx, v, head_size, head_count, n_tokens);
    a = c.ggml_reshape_3d(ctx, a, head_size, head_count, n_tokens);

    const s_l = mctx_cur.getSL(il);
    const n_embd_s: usize = hp.nEmbdS();
    var wkv_state = b.buildRs(inp, s_l, @intCast(n_embd_s), @intCast(n_seqs), null);

    const wkv_output: *Tensor = c.ggml_rwkv_wkv7(ctx, r, w, k, v, c.ggml_neg(ctx, kk), c.ggml_mul(ctx, kk, a), wkv_state);
    var cur: *Tensor = c.ggml_view_1d(ctx, wkv_output, n_embd * n_tokens, 0);
    wkv_state = c.ggml_view_1d(ctx, wkv_output, n_embd * head_size * n_seqs, tok * f32_size);

    c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, wkv_state, c.ggml_view_1d(ctx, s_l, @as(i64, @intCast(n_embd_s)) * n_seqs, n_embd_s * kv_head * c.ggml_element_size(s_l))));

    if (layer.time_mix_ln != null and layer.time_mix_ln_b != null) {
        // A group norm with head_count groups.
        cur = c.ggml_reshape_3d(ctx, cur, @divTrunc(n_embd, head_count), head_count, n_tokens);
        cur = c.ggml_norm(ctx, cur, 64e-5);

        // Back to regular vectors.
        cur = c.ggml_reshape_2d(ctx, cur, n_embd, n_tokens);
        cur = c.ggml_add(ctx, c.ggml_mul(ctx, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    } else {
        cur = c.ggml_reshape_2d(ctx, cur, n_embd, n_tokens);
    }
    const rk: *Tensor = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, c.ggml_mul(ctx, k, r), c.ggml_reshape_2d(ctx, layer.time_mix_r_k, head_size, head_count)));
    cur = c.ggml_add(ctx, cur, c.ggml_reshape_2d(ctx, c.ggml_mul(ctx, v, rk), n_embd, n_tokens));

    if (has_gating) {
        cur = c.ggml_mul(ctx, cur, g);
    }
    cur = b.buildLoraMm(layer.time_mix_output.?, cur, null);

    return c.ggml_reshape_3d(ctx, cur, n_embd, n_seq_tokens, n_seqs);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
