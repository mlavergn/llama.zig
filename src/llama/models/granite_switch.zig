//! Granite Switch: Granite's decoder blocks with a bank of LoRA adapters
//! built into the weights, one selected per token by an attention head
//! that runs inside the graph.
//!
//! Each adapter is switched on by its own activation token, which is
//! embedded as a substitute token instead. An extra single-head attention
//! layer after the real ones -- the router -- attends causally over a
//! one-dimensional signal, `+gain` at an activation token and `-gain`
//! elsewhere, so the most recent visible activation token dominates and its
//! adapter slot reads back from V. Slot 0 is the base model's zero delta.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/granite-switch.cpp` at v0.4.0
//! (`5266f24da`): `llama_model_granite_switch`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, its `graph` with
//! `build_switched_lora_delta`, `build_switched_lora_mm`,
//! `build_attention_layer` and `build_layer_ffn`, and the graph input
//! `llm_graph_input_switch`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const batch = @import("../batch.zig");
const hparams_mod = @import("../hparams.zig");
const libcxx = @import("../libcxx.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("models/granite-switch.cpp", 0, "out of memory");
}

/// Ports `llama_model_granite_switch`'s `load_arch_hparams`
/// (models/granite-switch.cpp:5 @5266f24da).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    const gs = &self.granite_switch;

    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
    hp.f_logit_scale = (try meta.getKey(f32, .logit_scale, true, exc)).?;
    if (try meta.getKey(f32, .residual_scale, false, exc)) |v| hp.f_residual_scale = v;
    if (try meta.getKey(f32, .embedding_scale, false, exc)) |v| hp.f_embedding_scale = v;
    if (try meta.getKey(f32, .attention_scale, false, exc)) |v| hp.f_attention_scale = v;

    var rope_finetuned = true;
    if (try meta.getKey(bool, .rope_scaling_finetuned, false, exc)) |v| rope_finetuned = v;
    // Needed for a round-trip save.
    hp.rope_finetuned = rope_finetuned;
    @memset(&hp.rope_pattern, @intFromBool(rope_finetuned));

    self.type = switch (hp.nLayer()) {
        40 => if (hp.n_embd == 4096) .@"8b" else .@"3b",
        64 => .@"30b",
        else => .unknown,
    };

    if (try meta.getKey(u32, .expert_shared_feed_forward_length, false, exc)) |v| hp.n_ff_shexp = v;

    gs.n_adapters = (try meta.getKey(u32, .adapter_count, true, exc)).?;
    gs.max_lora_rank = (try meta.getKey(u32, .adapter_lora_rank, true, exc)).?;
    if (try meta.getKey(f32, .adapter_router_gain, false, exc)) |v| gs.router_gain = v;

    // Bound the counts that size tensors.
    if (gs.n_adapters > 4096) {
        return exc.throw(.runtime_error, "graniteswitch: invalid adapter count {d}", .{gs.n_adapters});
    }
    if (gs.max_lora_rank > 4096) {
        return exc.throw(.runtime_error, "graniteswitch: invalid lora rank {d}", .{gs.max_lora_rank});
    }

    const token_ids = (try meta.getArrList(i32, .adapter_token_ids_activate, true, exc)).?;
    defer allocator.free(token_ids);
    const substitute_ids = (try meta.getArrList(i32, .adapter_token_ids_substitute, true, exc)).?;
    defer allocator.free(substitute_ids);

    if (token_ids.len != gs.n_adapters or substitute_ids.len != gs.n_adapters) {
        return exc.throw(.runtime_error, "graniteswitch: adapter token id arrays ({d} activate, {d} substitute) do not match adapter count {d}", .{ token_ids.len, substitute_ids.len, gs.n_adapters });
    }

    if (gs.adapter_token_to_slot) |*m| m.deinit();
    if (gs.adapter_token_to_substitute) |*m| m.deinit();
    gs.adapter_token_to_slot = libcxx.I32Map(i32).init(allocator) catch oom();
    gs.adapter_token_to_substitute = libcxx.I32Map(i32).init(allocator) catch oom();
    var i: u32 = 0;
    while (i < gs.n_adapters) : (i += 1) {
        // Adapter i is stacked slot i+1; slot 0 is the base, a zero delta.
        gs.adapter_token_to_slot.?.put(token_ids[i], @intCast(i + 1)) catch oom();
        gs.adapter_token_to_substitute.?.put(token_ids[i], substitute_ids[i]) catch oom();
    }

    // An extra single-head attention layer at the end, index n_real, holds
    // the router's K and V. Reusing n_layer_nextn keeps n_layer() == n_real,
    // so the regular layers keep their indices and the KV cache's shift and
    // defrag skip the router layer. n_layer_nextn is repurposed here (there
    // is no MTP): it leaks as 1 into llama_model_n_layer_nextn() and into a
    // re-saved nextn_predict_layers.
    const n_real = hp.nLayer();
    if (n_real >= hparams_mod.max_layers) {
        return exc.throw(.runtime_error, "graniteswitch: block count {d} exceeds LLAMA_MAX_LAYERS", .{n_real});
    }
    hp.router_layer = @intCast(n_real);
    hp.n_layer_all = n_real + 1;
    hp.n_layer_nextn = 1;

    hp.n_head_arr[n_real] = 1;
    hp.n_head_kv_arr[n_real] = 1;
    hp.n_ff_arr[n_real] = 0;
}

/// Ports `llama_model_granite_switch`'s `load_arch_tensors`
/// (models/granite-switch.cpp:74 @5266f24da).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const gs = &self.granite_switch;
    const nr = loader.tensor_not_required;
    const dup = loader.tensor_duplicated;

    // Slot 0 is the base, a zero delta.
    const n_slots: i64 = @as(i64, gs.n_adapters) + 1;
    const n_rank: i64 = gs.max_lora_rank;
    const n_embd_q = l.n_embd_head_k * l.n_head;
    const n_embd_kv = l.n_embd_k_gqa;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // The substitute ids index tok_embd's rows directly, so they are
    // range-checked against n_vocab.
    var it = gs.adapter_token_to_substitute.?.iterator();
    while (it.next()) |e| {
        const sub = e.value;
        if (sub < 0 or @as(i64, sub) >= l.n_vocab) {
            return exc.throw(.runtime_error, "graniteswitch: substitute token id {d} out of range [0, {d})", .{ sub, @as(c_int, @truncate(l.n_vocab)) });
        }
    }

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);
    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, dup, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.wqkv = try self.createTensor(.{ .tensor = .attn_qkv, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_embd_q + 2 * n_embd_kv }, 0, exc);
        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ n_embd_q, l.n_embd }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);

        const sl = &layer.switch_lora;

        sl.a_q = try self.createTensor(.{ .tensor = .attn_q, .suffix = "lora_a", .bid = i }, &.{ l.n_embd, n_rank, n_slots }, 0, exc);
        sl.b_q = try self.createTensor(.{ .tensor = .attn_q, .suffix = "lora_b", .bid = i }, &.{ n_rank, n_embd_q, n_slots }, 0, exc);
        sl.a_k = try self.createTensor(.{ .tensor = .attn_k, .suffix = "lora_a", .bid = i }, &.{ l.n_embd, n_rank, n_slots }, 0, exc);
        sl.b_k = try self.createTensor(.{ .tensor = .attn_k, .suffix = "lora_b", .bid = i }, &.{ n_rank, n_embd_kv, n_slots }, 0, exc);
        sl.a_v = try self.createTensor(.{ .tensor = .attn_v, .suffix = "lora_a", .bid = i }, &.{ l.n_embd, n_rank, n_slots }, 0, exc);
        sl.b_v = try self.createTensor(.{ .tensor = .attn_v, .suffix = "lora_b", .bid = i }, &.{ n_rank, n_embd_kv, n_slots }, 0, exc);

        sl.a_o = try self.createTensor(.{ .tensor = .attn_out, .suffix = "lora_a", .bid = i }, &.{ n_embd_q, n_rank, n_slots }, 0, exc);
        sl.b_o = try self.createTensor(.{ .tensor = .attn_out, .suffix = "lora_b", .bid = i }, &.{ n_rank, l.n_embd, n_slots }, 0, exc);

        sl.a_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "lora_a", .bid = i }, &.{ l.n_embd, n_rank, n_slots }, 0, exc);
        sl.b_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "lora_b", .bid = i }, &.{ n_rank, l.n_ff, n_slots }, 0, exc);
        sl.a_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "lora_a", .bid = i }, &.{ l.n_embd, n_rank, n_slots }, 0, exc);
        sl.b_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "lora_b", .bid = i }, &.{ n_rank, l.n_ff, n_slots }, 0, exc);
        sl.a_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "lora_a", .bid = i }, &.{ l.n_ff, n_rank, n_slots }, 0, exc);
        sl.b_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "lora_b", .bid = i }, &.{ n_rank, l.n_embd, n_slots }, 0, exc);
    }
}

/// Ports `llm_graph_input_switch` (models/granite-switch.cpp:134
/// @5266f24da): the substituted token ids and the router's three signals,
/// set from the batch's tokens.
const InpSwitch = struct {
    /// I32 `[n_tokens]`: the adapter-substituted token ids.
    sub_tokens: ?*Tensor = null,
    /// F32 `[n_tokens]`: the router's K signal, plus or minus the gain.
    router_ksig: ?*Tensor = null,
    /// F32 `[n_tokens]`: the router's V value, the adapter slot or 0.
    router_vval: ?*Tensor = null,
    /// F32 `[n_tokens]`: the router's Q value, constantly 1.0.
    router_q: ?*Tensor = null,

    smodel: *const Model,

    /// Ports `llm_graph_input_switch`'s `set_input`
    /// (models/granite-switch.cpp:151 @5266f24da). K's dimension 0 is
    /// `+gain` for an adapter token and `-gain` otherwise; the causal
    /// softmax then lets a single visible adapter token dominate, so the
    /// readback recovers its slot.
    pub fn setInput(self: *InpSwitch, ubatch_: ?*const batch.Ubatch) void {
        const ubatch = ubatch_.?;
        if (ubatch.token == null) return;

        const n_tokens: usize = ubatch.n_tokens;
        const gs = &self.smodel.granite_switch;

        const sub = allocator.alloc(i32, n_tokens) catch oom();
        defer allocator.free(sub);
        const ksig = allocator.alloc(f32, n_tokens) catch oom();
        defer allocator.free(ksig);
        const vval = allocator.alloc(f32, n_tokens) catch oom();
        defer allocator.free(vval);
        const q = allocator.alloc(f32, n_tokens) catch oom();
        defer allocator.free(q);
        @memset(q, 1.0);

        for (0..n_tokens) |i| {
            const tok = ubatch.token[i];

            if (gs.adapter_token_to_slot.?.get(tok)) |slot| {
                ksig[i] = gs.router_gain;
                vval[i] = @floatFromInt(slot);
            } else {
                ksig[i] = -gs.router_gain;
                vval[i] = 0.0;
            }

            sub[i] = gs.adapter_token_to_substitute.?.get(tok) orelse tok;
        }

        c.ggml_backend_tensor_set(self.sub_tokens, sub.ptr, 0, n_tokens * c.ggml_element_size(self.sub_tokens));
        c.ggml_backend_tensor_set(self.router_ksig, ksig.ptr, 0, n_tokens * c.ggml_element_size(self.router_ksig));
        c.ggml_backend_tensor_set(self.router_vval, vval.ptr, 0, n_tokens * c.ggml_element_size(self.router_vval));
        c.ggml_backend_tensor_set(self.router_q, q.ptr, 0, n_tokens * c.ggml_element_size(self.router_q));
    }
};

/// Ports `llama_model_granite_switch`'s `build_arch_graph`
/// (models/granite-switch.cpp:187 @5266f24da).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports `llama_model_granite_switch`'s `graph`'s
/// `build_switched_lora_delta` (models/granite-switch.cpp:193 @5266f24da):
/// the per-token switched LoRA delta `B_a*(A_a*x)`, the adapter chosen per
/// token by `ids`. `cur` is `{n_in, n_tokens}` and `ids` `{n_tokens}`; the
/// result is `{n_out, n_tokens}`.
fn buildSwitchedLoraDelta(b: *const graph.Builder, lora_a: *Tensor, lora_b: *Tensor, cur: *Tensor, ids: *Tensor) *Tensor {
    const ctx = b.ctx0;
    const n_in = cur.ne[0];
    const n_tokens = cur.ne[1];

    const x: *Tensor = c.ggml_reshape_3d(ctx, cur, n_in, 1, n_tokens);
    const ids2: *Tensor = c.ggml_reshape_2d(ctx, ids, 1, n_tokens);

    // {max_rank, 1, n_tokens}
    const a: *Tensor = c.ggml_mul_mat_id(ctx, lora_a, x, ids2);
    // {n_out, 1, n_tokens}
    const d: *Tensor = c.ggml_mul_mat_id(ctx, lora_b, a, ids2);

    return c.ggml_reshape_2d(ctx, d, d.ne[0], n_tokens);
}

/// Ports `llama_model_granite_switch`'s `graph`'s `build_switched_lora_mm`
/// (models/granite-switch.cpp:210 @5266f24da): the base projection plus the
/// switched delta.
fn buildSwitchedLoraMm(b: *const graph.Builder, w: *Tensor, lora_a: *Tensor, lora_b: *Tensor, cur: *Tensor, ids: *Tensor) *Tensor {
    const base: *Tensor = c.ggml_mul_mat(b.ctx0, w, cur);
    const delta = buildSwitchedLoraDelta(b, lora_a, lora_b, cur, ids);
    return c.ggml_add(b.ctx0, base, delta);
}

/// The router's lanes: a 1-D signal as `{1, 1, n_tokens}`, zero-padded to
/// a head's width, so only dimension 0 carries it.
fn routerLane(b: *const graph.Builder, sig1d: *Tensor, n_embd_head: i64) *Tensor {
    const t: *Tensor = c.ggml_reshape_3d(b.ctx0, sig1d, 1, 1, b.n_tokens);
    return c.ggml_pad(b.ctx0, t, @intCast(n_embd_head - 1), 0, 0, 0);
}

/// Ports the constructor of `llama_model_granite_switch`'s `graph`
/// (models/granite-switch.cpp:221 @5266f24da).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const gs = &model.granite_switch;

    // TODO upstream: support raw embedding input (multimodal or pre-embedded
    // tokens) when needed.
    impl.assert(b.ubatch.token != null, "models/granite-switch.cpp", 229, "ubatch.token && \"granite-switch requires token input\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/granite-switch.cpp", 232, "n_embd_head == hparams.n_embd_head_k()");
    impl.assert(n_embd_head == b.n_rot, "models/granite-switch.cpp", 233, "n_embd_head == n_rot");

    const inp_switch = allocator.create(InpSwitch) catch oom();
    inp_switch.* = .{ .smodel = model };
    inp_switch.sub_tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    inp_switch.router_ksig = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, b.n_tokens);
    inp_switch.router_vval = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, b.n_tokens);
    inp_switch.router_q = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, b.n_tokens);
    c.ggml_set_input(inp_switch.sub_tokens);
    c.ggml_set_input(inp_switch.router_ksig);
    c.ggml_set_input(inp_switch.router_vval);
    c.ggml_set_input(inp_switch.router_q);
    const sub_tokens = inp_switch.sub_tokens.?;
    const router_ksig = inp_switch.router_ksig.?;
    const router_vval = inp_switch.router_vval.?;
    const router_q = inp_switch.router_q.?;
    _ = b.res.addInput(InpSwitch, inp_switch);

    // The substituted ids are embedded directly; build_inp_embd would embed
    // the raw tokens.
    var inpL: *Tensor = c.ggml_get_rows(ctx, model.tok_embd.?, sub_tokens);
    if (hp.f_embedding_scale != 0.0) {
        inpL = c.ggml_scale(ctx, inpL, hp.f_embedding_scale);
    }
    b.cb(inpL, "inp_embd", -1);

    var inp_pos: ?*Tensor = null;
    if (hp.hasRope(0)) {
        inp_pos = b.buildInpPos();
    }
    const inp_attn = b.buildAttnInpKv();

    // A single causal head at layer R recovers the adapter index in the
    // graph: only dimension 0 carries signal (Q[0] = 1, K[0] = +/-gain,
    // V[0] = slot or 0), the rest is zero padding.
    const R = hp.router_layer;
    impl.assert(R >= 0, "models/granite-switch.cpp", 266, "R >= 0");
    const Qr = routerLane(b, router_q, n_embd_head);
    const Kr = routerLane(b, router_ksig, n_embd_head);
    const Vr = routerLane(b, router_vval, n_embd_head);

    const router_out = b.buildAttn(inp_attn, null, null, null, Qr, Kr, Vr, null, null, null, 1.0, R);
    b.cb(router_out, "router_out", R);

    // Row 0 of router_out is the attended slot; clamped and rounded to an
    // I32 index.
    var slot_f: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, router_out, 1, b.n_tokens, router_out.nb[1], 0));
    slot_f = c.ggml_reshape_1d(ctx, slot_f, b.n_tokens);
    slot_f = c.ggml_clamp(ctx, slot_f, 0.0, @floatFromInt(gs.n_adapters));
    slot_f = c.ggml_round(ctx, slot_f);
    var adapter_ids: *Tensor = c.ggml_cast(ctx, slot_f, c.GGML_TYPE_I32);
    b.cb(adapter_ids, "adapter_ids", -1);

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        var inpSA = inpL;

        var cur = b.buildNorm(inpL, model.layers[@intCast(il)].attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        cur = buildAttentionLayer(b, cur, inp_pos, adapter_ids, inp_attn, model, n_embd_head, il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = c.ggml_get_rows(ctx, inpSA, inp_out_ids);
            // adapter_ids stays aligned to the rows kept, through a 2-D
            // round trip for get_rows.
            const n_out = inp_out_ids.ne[0];
            adapter_ids = c.ggml_get_rows(ctx, c.ggml_reshape_2d(ctx, adapter_ids, 1, adapter_ids.ne[0]), inp_out_ids);
            adapter_ids = c.ggml_reshape_1d(ctx, adapter_ids, n_out);
        }

        cur = buildLayerFfn(b, cur, inpSA, adapter_ids, model, il);

        inpL = cur;
    }

    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    cur = c.ggml_scale(ctx, cur, 1.0 / hp.f_logit_scale);
    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_granite_switch`'s `graph`'s `build_attention_layer`
/// (models/granite-switch.cpp:331 @5266f24da): the fused QKV sliced, each
/// part plus its switched delta, and the output projection switched too.
fn buildAttentionLayer(
    b: *const graph.Builder,
    cur_: *Tensor,
    inp_pos: ?*Tensor,
    adapter_ids: *Tensor,
    inp_attn: *graph.InpAttnKv,
    model: *const Model,
    n_embd_head: i64,
    il: c_int,
) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    const sl = &layer.switch_lora;
    const cur = cur_;

    const n_head: i64 = hp.nHead(@intCast(il));
    const n_head_kv: i64 = hp.nHeadKv(@intCast(il));

    const qkv: *Tensor = c.ggml_mul_mat(ctx, layer.wqkv, cur);
    b.cb(qkv, "wqkv", il);

    const n_embd_q = n_embd_head * n_head;
    const n_embd_kv = n_embd_head * n_head_kv;
    const esz: i64 = @intCast(c.ggml_element_size(qkv));

    // The fused QKV sliced into Q, K and V, made contiguous so the LoRA
    // deltas can be added.
    var Qcur: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, qkv, n_embd_q, qkv.ne[1], qkv.nb[1], 0));
    var Kcur: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, qkv, n_embd_kv, qkv.ne[1], qkv.nb[1], @intCast(n_embd_q * esz)));
    var Vcur: *Tensor = c.ggml_cont(ctx, c.ggml_view_2d(ctx, qkv, n_embd_kv, qkv.ne[1], qkv.nb[1], @intCast((n_embd_q + n_embd_kv) * esz)));

    Qcur = c.ggml_add(ctx, Qcur, buildSwitchedLoraDelta(b, sl.a_q.?, sl.b_q.?, cur, adapter_ids));
    Kcur = c.ggml_add(ctx, Kcur, buildSwitchedLoraDelta(b, sl.a_k.?, sl.b_k.?, cur, adapter_ids));
    Vcur = c.ggml_add(ctx, Vcur, buildSwitchedLoraDelta(b, sl.a_v.?, sl.b_v.?, cur, adapter_ids));

    Qcur = c.ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, b.n_tokens);
    Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, b.n_tokens);
    Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, b.n_tokens);

    if (hp.hasRope(@intCast(il))) {
        const rope_factors = model.getRopeFactors(b.cparams, @intCast(il));
        Qcur = c.ggml_rope_ext(ctx, Qcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
        Kcur = c.ggml_rope_ext(ctx, Kcur, inp_pos, rope_factors, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    }
    b.cb(Qcur, "Qcur", il);
    b.cb(Kcur, "Kcur", il);
    b.cb(Vcur, "Vcur", il);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    // wo is null, so build_attn returns the heads concatenated; the output
    // projection is switched below.
    const attn = b.buildAttn(inp_attn, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(attn, "attn_pre_o", il);

    const out = buildSwitchedLoraMm(b, layer.wo.?, sl.a_o.?, sl.b_o.?, attn, adapter_ids);
    b.cb(out, "attn_out", il);
    return out;
}

/// Ports `llama_model_granite_switch`'s `graph`'s `build_layer_ffn`
/// (models/granite-switch.cpp:392 @5266f24da): Granite's scaled residuals
/// around a SwiGLU whose three projections are all switched.
fn buildLayerFfn(b: *const graph.Builder, cur_: *Tensor, inpSA: *Tensor, adapter_ids: *Tensor, model: *const Model, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    const sl = &layer.switch_lora;
    var cur = cur_;

    if (hp.f_residual_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, hp.f_residual_scale);
    }
    const ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpSA);
    b.cb(ffn_inp, "ffn_inp", il);

    cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
    b.cb(cur, "ffn_norm", il);

    var g = buildSwitchedLoraMm(b, layer.ffn_gate.?, sl.a_gate.?, sl.b_gate.?, cur, adapter_ids);
    const u = buildSwitchedLoraMm(b, layer.ffn_up.?, sl.a_up.?, sl.b_up.?, cur, adapter_ids);
    g = c.ggml_silu(ctx, g);
    const gu: *Tensor = c.ggml_mul(ctx, g, u);
    cur = buildSwitchedLoraMm(b, layer.ffn_down.?, sl.a_down.?, sl.b_down.?, gu, adapter_ids);
    b.cb(cur, "ffn_out", il);

    if (hp.f_residual_scale != 0.0) {
        cur = c.ggml_scale(ctx, cur, hp.f_residual_scale);
    }
    cur = c.ggml_add(ctx, cur, ffn_inp);

    cur = b.buildCvec(cur, il);
    b.cb(cur, "l_out", il);

    return cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
