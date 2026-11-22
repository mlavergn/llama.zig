//! The RWKV-6 blocks RWKV-6 and RWKV6-Qwen2 share: the time mix, with its
//! data-dependent token-shift interpolation and decay, and RWKV-6's
//! channel mix.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/rwkv6-base.cpp` at v1.0.0
//! (`b1b770d90`): `llm_build_rwkv6_base`'s `build_rwkv6_channel_mix` and
//! `build_rwkv6_time_mix`. Its constructor only stores the model, which the
//! port passes to each function instead.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llm_build_rwkv6_base`'s `build_rwkv6_channel_mix`
/// (models/rwkv6-base.cpp:8 @b1b770d90): the squared-ReLU feed-forward,
/// gated by the receptance.
pub fn buildRwkv6ChannelMix(b: *const llama.graph.Builder, layer: *const llama.model.Layer, cur_: *llama.impl.c.ggml_tensor, x_prev: *llama.impl.c.ggml_tensor, arch: llama.arch_table.Arch) *llama.impl.c.ggml_tensor {
    const ctx = b.ctx0;
    var cur = cur_;
    const sx: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_sub(ctx, x_prev, cur);
    switch (arch) {
        .rwkv6 => {
            const xk: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, sx, layer.channel_mix_lerp_k), cur);
            const xr: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, sx, layer.channel_mix_lerp_r), cur);

            const r: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_sigmoid(ctx, b.buildLoraMm(layer.channel_mix_receptance.?, xr, null));
            const k: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_sqr(ctx, llama.impl.c.ggml_relu(ctx, b.buildLoraMm(layer.channel_mix_key.?, xk, null)));
            cur = llama.impl.c.ggml_mul(ctx, r, b.buildLoraMm(layer.channel_mix_value.?, k, null));
        },
        else => llama.impl.abort("models/rwkv6-base.cpp", 25, "fatal error"),
    }
    return cur;
}

/// Ports `llm_build_rwkv6_base`'s `build_rwkv6_time_mix`
/// (models/rwkv6-base.cpp:30 @b1b770d90): the WKV-6 recurrence, or for a
/// QRWKV model (one without `time_mix_first`) gated linear attention.
pub fn buildRwkv6TimeMix(model: *const llama.model.Model, b: *const llama.graph.Builder, inp: *const llama.graph.InpRs, cur_: *llama.impl.c.ggml_tensor, x_prev: *llama.impl.c.ggml_tensor, ubatch: *const llama.batch.Ubatch, il: c_int) *llama.impl.c.ggml_tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const mctx_cur = b.mctx.?.cast(llama.memory_recurrent.RecurrentContext);

    const n_tokens: i64 = ubatch.n_tokens;
    const n_seqs: i64 = ubatch.n_seqs;
    const n_seq_tokens: i64 = ubatch.n_seq_tokens;
    const n_embd: i64 = hp.n_embd;
    const head_size: i64 = hp.wkv_head_size;
    const n_head: i64 = @divTrunc(n_embd, head_size);
    const n_head_kv: i64 = hp.nHeadKv(@intCast(il));

    const kv_head: usize = mctx_cur.getHead();

    const layer = &model.layers[@intCast(il)];

    const is_qrwkv = layer.time_mix_first == null;

    const f32_size: usize = @sizeOf(f32);
    const tok: usize = @intCast(n_embd * n_tokens);

    var sx: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_sub(ctx, x_prev, cur_);

    sx = llama.impl.c.ggml_reshape_2d(ctx, sx, n_embd, n_tokens);
    var cur: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_reshape_2d(ctx, cur_, n_embd, n_tokens);

    var xxx: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, sx, layer.time_mix_lerp_x), cur);

    const w1 = layer.time_mix_w1.?;
    xxx = llama.impl.c.ggml_reshape_4d(ctx, llama.impl.c.ggml_tanh(ctx, llama.impl.c.ggml_mul_mat(ctx, w1, xxx)), @divTrunc(w1.ne[1], 5), 1, 5, n_tokens);

    xxx = llama.impl.c.ggml_cont(ctx, llama.impl.c.ggml_permute(ctx, xxx, 0, 1, 3, 2));

    const w2 = layer.time_mix_w2.?;
    xxx = llama.impl.c.ggml_mul_mat(ctx, llama.impl.c.ggml_reshape_4d(ctx, w2, w2.ne[0], w2.ne[1], 1, 5), xxx);

    var xw: *llama.impl.c.ggml_tensor = undefined;
    var xk: *llama.impl.c.ggml_tensor = undefined;
    var xv: *llama.impl.c.ggml_tensor = undefined;
    var xr: *llama.impl.c.ggml_tensor = undefined;
    var xg: *llama.impl.c.ggml_tensor = undefined;
    if (layer.time_mix_lerp_fused) |lerp_fused| {
        // Fusing these weights is a performance improvement.
        sx = llama.impl.c.ggml_reshape_3d(ctx, sx, n_embd, 1, n_tokens);
        cur = llama.impl.c.ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
        xxx = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, llama.impl.c.ggml_add(ctx, xxx, lerp_fused), sx), cur);
        xw = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], 0);
        xk = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * f32_size);
        xv = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 2 * f32_size);
        xr = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 3 * f32_size);
        xg = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 4 * f32_size);
    } else {
        // For backward compatibility.
        xw = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], 0);
        xk = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * f32_size);
        xv = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 2 * f32_size);
        xr = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 3 * f32_size);
        xg = llama.impl.c.ggml_view_2d(ctx, xxx, n_embd, n_tokens, xxx.nb[1], tok * 4 * f32_size);

        xw = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, llama.impl.c.ggml_add(ctx, xw, layer.time_mix_lerp_w), sx), cur);
        xk = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, llama.impl.c.ggml_add(ctx, xk, layer.time_mix_lerp_k), sx), cur);
        xv = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, llama.impl.c.ggml_add(ctx, xv, layer.time_mix_lerp_v), sx), cur);
        xr = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, llama.impl.c.ggml_add(ctx, xr, layer.time_mix_lerp_r), sx), cur);
        xg = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, llama.impl.c.ggml_add(ctx, xg, layer.time_mix_lerp_g), sx), cur);
    }
    var r = b.buildLoraMm(layer.time_mix_receptance.?, xr, null);
    var k = b.buildLoraMm(layer.time_mix_key.?, xk, null);
    var v = b.buildLoraMm(layer.time_mix_value.?, xv, null);
    if (layer.time_mix_receptance_b) |bias| r = llama.impl.c.ggml_add(ctx, r, bias);
    if (layer.time_mix_key_b) |bias| k = llama.impl.c.ggml_add(ctx, k, bias);
    if (layer.time_mix_value_b) |bias| v = llama.impl.c.ggml_add(ctx, v, bias);
    var g = b.buildLoraMm(layer.time_mix_gate.?, xg, null);
    if (is_qrwkv) {
        g = llama.impl.c.ggml_sigmoid(ctx, g);
    } else {
        g = llama.impl.c.ggml_silu(ctx, g);
    }
    if (n_head_kv != 0 and n_head_kv != n_head) {
        llama.impl.assert(@rem(n_head, n_head_kv) == 0, "models/rwkv6-base.cpp", 110, "n_head % n_head_kv == 0");
        k = llama.impl.c.ggml_reshape_4d(ctx, k, head_size, 1, n_head_kv, n_tokens);
        v = llama.impl.c.ggml_reshape_4d(ctx, v, head_size, 1, n_head_kv, n_tokens);
        const tmp: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_new_tensor_4d(ctx, llama.impl.c.GGML_TYPE_F32, head_size, @divTrunc(n_head, n_head_kv), n_head_kv, n_tokens);
        k = llama.impl.c.ggml_repeat(ctx, k, tmp);
        v = llama.impl.c.ggml_repeat(ctx, v, tmp);
    }
    k = llama.impl.c.ggml_reshape_3d(ctx, k, head_size, n_head, n_tokens);
    v = llama.impl.c.ggml_reshape_3d(ctx, v, head_size, n_head, n_tokens);
    r = llama.impl.c.ggml_reshape_3d(ctx, r, head_size, n_head, n_tokens);

    var w: *llama.impl.c.ggml_tensor = llama.impl.c.ggml_mul_mat(ctx, layer.time_mix_decay_w2, llama.impl.c.ggml_tanh(ctx, llama.impl.c.ggml_mul_mat(ctx, layer.time_mix_decay_w1, xw)));

    w = llama.impl.c.ggml_add(ctx, w, layer.time_mix_decay);
    w = llama.impl.c.ggml_exp(ctx, llama.impl.c.ggml_neg(ctx, llama.impl.c.ggml_exp(ctx, w)));
    w = llama.impl.c.ggml_reshape_3d(ctx, w, head_size, n_head, n_tokens);

    if (is_qrwkv) {
        // k = k * (1 - w)
        k = llama.impl.c.ggml_sub(ctx, k, llama.impl.c.ggml_mul(ctx, k, w));
    }
    const s_l = mctx_cur.getSL(il);
    const n_embd_s: usize = hp.nEmbdS();
    var wkv_state = b.buildRs(inp, s_l, @intCast(n_embd_s), @intCast(n_seqs), null);

    var wkv_output: *llama.impl.c.ggml_tensor = undefined;
    if (is_qrwkv) {
        // `pow(head_size, -0.5f)`: the double overload, narrowed to the
        // float parameter.
        const scale: f32 = @floatCast(stdlib.libm.pow(@floatFromInt(head_size), -0.5));
        wkv_output = llama.impl.c.ggml_gated_linear_attn(ctx, k, v, r, w, wkv_state, scale);
    } else {
        wkv_output = llama.impl.c.ggml_rwkv_wkv6(ctx, k, v, r, layer.time_mix_first, w, wkv_state);
    }
    cur = llama.impl.c.ggml_view_1d(ctx, wkv_output, n_embd * n_tokens, 0);
    wkv_state = llama.impl.c.ggml_view_1d(ctx, wkv_output, n_embd * head_size * n_seqs, tok * f32_size);

    llama.impl.c.ggml_build_forward_expand(b.gf, llama.impl.c.ggml_cpy(ctx, wkv_state, llama.impl.c.ggml_view_1d(ctx, s_l, @as(i64, @intCast(n_embd_s)) * n_seqs, n_embd_s * kv_head * llama.impl.c.ggml_element_size(s_l))));

    if (!is_qrwkv) {
        // A group norm with head_count groups.
        cur = llama.impl.c.ggml_reshape_3d(ctx, cur, @divTrunc(n_embd, n_head), n_head, n_tokens);
        cur = llama.impl.c.ggml_norm(ctx, cur, 64e-5);

        // Back to regular vectors.
        cur = llama.impl.c.ggml_reshape_2d(ctx, cur, n_embd, n_tokens);
        cur = llama.impl.c.ggml_add(ctx, llama.impl.c.ggml_mul(ctx, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    } else {
        cur = llama.impl.c.ggml_reshape_2d(ctx, cur, n_embd, n_tokens);
    }
    cur = llama.impl.c.ggml_mul(ctx, cur, g);
    cur = b.buildLoraMm(layer.time_mix_output.?, cur, null);

    return llama.impl.c.ggml_reshape_3d(ctx, cur, n_embd, n_seq_tokens, n_seqs);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
