//! How `LLAMA_SPLIT_MODE_TENSOR` slices each tensor across devices: the
//! callback libllama hands ggml's meta device.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-model.h` -- `llama_meta_device_get_split_state_userdata`
//! - `llama.cpp/src/llama-model.cpp` -- `llama_meta_device_get_split_state`
//!
//! All at v0.4.1 (`b29c606e2`).
//!
//! # Gating
//!
//! Tensor parallelism needs several GPUs: this is the path CUDA multi-GPU
//! machines take, and on a one-GPU Mac it runs only where a harness builds a
//! meta device over the devices it has. The tensor-name tests are libc++
//! `std`'s `regex_match`, reproduced by `regex.zig`, so a pattern matches here
//! exactly where it matches there.

const std = @import("std");
const impl = @import("impl.zig");
const regex = @import("regex.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const SplitState = c.struct_ggml_backend_meta_split_state;

/// Ports `llama_meta_device_get_split_state_userdata` (llama-model.h:605
/// @b29c606e2).
pub const Userdata = extern struct {
    n_devices: usize = 0,
    model: ?*const Model = null,
};

const axis_0: c_uint = c.GGML_BACKEND_SPLIT_AXIS_0;
const axis_1: c_uint = c.GGML_BACKEND_SPLIT_AXIS_1;
const axis_2: c_uint = c.GGML_BACKEND_SPLIT_AXIS_2;
const axis_mirrored: c_uint = c.GGML_BACKEND_SPLIT_AXIS_MIRRORED;
const axis_partial: c_uint = c.GGML_BACKEND_SPLIT_AXIS_PARTIAL;

/// The function-local `static const std::regex`es, in declaration order.
const pattern_sources = [_][]const u8{
    "blk\\.\\d*\\.attn_q.weight", // q_weight
    "blk\\.\\d*\\.attn_(k|v).weight", // kv_weight
    "blk\\.\\d*\\.attn_qkv.weight", // qkv_weight
    "blk\\.\\d*\\.attn_q\\.bias", // q_bias
    "blk\\.\\d*\\.attn_(k|v)\\.bias", // kv_bias
    "blk\\.\\d*\\.attn_qkv.bias", // qkv_bias
    "blk\\.\\d*\\.attn_(q|k)_norm\\.weight", // qk_norm
    "cache_(k|v)_l\\d*", // kv_cache
    "cache_idx_(k|v)_l\\d*", // idx_cache
    "dsv4_(csa|hca|lid)_state_(kv|score)_l\\d*", // dsv4_state
    "blk\\.\\d*\\.attn_sinks.weight", // attn_sinks
    "blk\\.\\d*\\.attn_output.weight", // attn_out_weight
    "blk\\.\\d*\\.attn_output.bias", // attn_out_bias
    "blk\\.\\d*\\.attn_output_a\\.weight", // attn_out_a_weight
    "blk\\.\\d*\\.attn_output_b\\.weight", // attn_out_b_weight
    "blk\\.\\d*\\.attn_q_b\\.weight", // attn_q_b_weight
    "blk\\.\\d*\\.attn_gate.weight", // attn_gate_weight
    "blk\\.\\d*\\.ssm_dt.bias", // ssm_dt
    "blk\\.\\d*\\.ssm_a", // ssm_a
    "blk\\.\\d*\\.ssm_alpha.weight", // ssm_alpha
    "blk\\.\\d*\\.ssm_beta.weight", // ssm_beta
    "blk\\.\\d*\\.ssm_ba.weight", // ssm_beta_alpha
    "cache_r_l\\d*", // r_cache
    "cache_ple_r_l\\d*", // ple_r_cache
    "cache_s_l\\d*", // s_cache
    "blk\\.\\d*\\.ssm_conv1d.weight", // ssm_conv1d
    "blk\\.\\d*\\.ssm_out.weight", // ssm_out_weight
    "blk\\.\\d*\\.ffn_up(_exps)?.weight", // ffn_up_weight
    "blk\\.\\d*\\.ffn_up(_exps)?.bias", // ffn_up_bias
    "blk\\.\\d*\\.ffn_gate(_exps)?.weight", // ffn_gate_weight
    "blk\\.\\d*\\.ffn_gate(_exps)?.bias", // ffn_gate_bias
    "blk\\.\\d*\\.ffn_gate_up(_exps)?.weight", // ffn_gate_up_weight
    "blk\\.\\d*\\.ffn_down(_exps)?.weight", // ffn_down_weight
    "blk\\.\\d*\\.ffn_down.bias", // ffn_down_bias
    "blk\\.\\d*\\.ffn_down_exps.bias", // ffn_down_exps_bias
    "blk\\.\\d*\\.ffn_up_shexp.weight", // ffn_up_shexp_weight
    "blk\\.\\d*\\.ffn_gate_shexp.weight", // ffn_gate_shexp_weight
    "blk\\.\\d*\\.ffn_down_shexp.weight", // ffn_down_shexp_weight
    "output\\.weight", // output_weight
    "output\\.bias", // output_bias
};

const P = enum(u8) {
    q_weight,
    kv_weight,
    qkv_weight,
    q_bias,
    kv_bias,
    qkv_bias,
    qk_norm,
    kv_cache,
    idx_cache,
    dsv4_state,
    attn_sinks,
    attn_out_weight,
    attn_out_bias,
    attn_out_a_weight,
    attn_out_b_weight,
    attn_q_b_weight,
    attn_gate_weight,
    ssm_dt,
    ssm_a,
    ssm_alpha,
    ssm_beta,
    ssm_beta_alpha,
    r_cache,
    ple_r_cache,
    s_cache,
    ssm_conv1d,
    ssm_out_weight,
    ffn_up_weight,
    ffn_up_bias,
    ffn_gate_weight,
    ffn_gate_bias,
    ffn_gate_up_weight,
    ffn_down_weight,
    ffn_down_bias,
    ffn_down_exps_bias,
    ffn_up_shexp_weight,
    ffn_gate_shexp_weight,
    ffn_down_shexp_weight,
    output_weight,
    output_bias,
};

comptime {
    std.debug.assert(pattern_sources.len == @typeInfo(P).@"enum".field_names.len);
}

/// The compiled patterns, built once as the C++'s function-local statics
/// are, under the same once-only guarantee.
var patterns: [pattern_sources.len]regex.Regex = undefined;
var patterns_state = std.atomic.Value(u8).init(0);

fn getPatterns() *const [pattern_sources.len]regex.Regex {
    if (patterns_state.load(.acquire) == 2) return &patterns;
    if (patterns_state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) {
        for (pattern_sources, 0..) |src, i| {
            patterns[i] = regex.Regex.compile(src) catch |e| regex.throwRegexError(e);
        }
        patterns_state.store(2, .release);
    } else {
        while (patterns_state.load(.acquire) != 2) std.atomic.spinLoopHint();
    }
    return &patterns;
}

const Ctx = struct {
    tensor: *const Tensor,
    name: []const u8,
    ud: *const Userdata,
    model: *const Model,
    is_dsv4: bool,

    fn m(self: *const Ctx, p: P) bool {
        var match = regex.Match.init();
        defer match.deinit();
        return getPatterns()[@backingInt(p)].match(self.name, &match);
    }
};

/// `tensor_config` (llama-model.cpp:424 @b29c606e2).
const TensorConfig = struct {
    axis: c_uint,
    tensor_axis_0: *const Tensor,
    il: u32,
    rotation: usize,
};

/// One segment along the split axis and how often it repeats.
const Segment = struct { ne: i64, nr: u32 };

const max_segments = 16;

/// `std`'s `stoull` on the digits at the start of `s`.
fn stoull(s: []const u8) u64 {
    var i: usize = 0;
    while (i < s.len and std.ascii.isWhitespace(s[i])) i += 1;
    var v: u64 = 0;
    const start = i;
    while (i < s.len and std.ascii.isDigit(s[i])) : (i += 1) v = v *% 10 +% (s[i] - '0');
    if (i == start) impl.abort("llama-model.cpp", 454, "std::stoull: no conversion");
    return v;
}

fn lcm(a: i64, b: i64) i64 {
    if (a == 0 or b == 0) return 0;
    const aa: u64 = @abs(a);
    const bb: u64 = @abs(b);
    return @intCast(aa / std.math.gcd(aa, bb) * bb);
}

/// The `get_tensor_config_impl` lambda (llama-model.cpp:433 @b29c606e2).
fn configImpl(cx: *const Ctx, axis: c_uint, suffix: []const u8, suffix_fallback: []const u8) TensorConfig {
    const hparams = &cx.model.hparams;
    const n_devices = cx.ud.n_devices;

    // The layers in a tensor can be inhomogeneous; count only the previous
    // layers of the same kind, against aliasing.
    const getIlEff = struct {
        fn f(hp: anytype, il: u32) usize {
            var ret: usize = 0;
            const il_is_recr = hp.isRecr(il);
            const il_is_swa = hp.isSwa(il);
            var il_prev: u32 = 0;
            while (il_prev < il) : (il_prev += 1) {
                ret += @intFromBool(hp.isRecr(il_prev) == il_is_recr and hp.isSwa(il_prev) == il_is_swa);
            }
            return ret;
        }
    }.f;

    var il: u32 = 0;
    var prefix_buf: [96]u8 = undefined;
    var prefix: []const u8 = "";
    var rotation: usize = 0;
    const name = cx.name;
    if (name.len >= 4 and std.mem.eql(u8, name[0..4], "blk.")) {
        const length_prefix = std.mem.indexOfScalarPos(u8, name, 4, '.') orelse impl.abort("llama-model.cpp", 452, "GGML_ASSERT(length_prefix != std::string::npos) failed");
        prefix = name[0 .. length_prefix + 1];
        il = @truncate(stoull(name[4..@min(name.len, 4 + length_prefix)]));
        rotation = getIlEff(hparams, il) % n_devices;
    } else if (name.len >= 6 and std.mem.eql(u8, name[0..6], "cache_")) {
        const layer_index_start = std.mem.indexOfPos(u8, name, 6, "_l") orelse impl.abort("llama-model.cpp", 458, "GGML_ASSERT(layer_index_start != std::string::npos) failed");
        il = @truncate(stoull(name[layer_index_start + 2 ..]));
        prefix = std.fmt.bufPrint(&prefix_buf, "blk.{d}.", .{il}) catch unreachable;
        rotation = getIlEff(hparams, il) % n_devices;
    } else {
        il = 0;
        rotation = hparams.nLayer() % n_devices;
    }

    var tensor_axis_0: ?*const Tensor = null;
    var nb: [160]u8 = undefined;
    if (suffix.len == 0) {
        tensor_axis_0 = cx.tensor;
    } else {
        const full = std.fmt.bufPrint(&nb, "{s}{s}", .{ prefix, suffix }) catch unreachable;
        tensor_axis_0 = cx.model.getTensor(full);
    }
    if (tensor_axis_0 == null) {
        impl.assert(suffix_fallback.len != 0, "llama-model.cpp", 468, "!suffix_fallback.empty()");
        const full = std.fmt.bufPrint(&nb, "{s}{s}", .{ prefix, suffix_fallback }) catch unreachable;
        tensor_axis_0 = cx.model.getTensor(full);
    }
    impl.assert(tensor_axis_0 != null, "llama-model.cpp", 471, "tensor_axis_0 != nullptr");
    return .{ .axis = axis, .tensor_axis_0 = tensor_axis_0.?, .il = il, .rotation = rotation };
}

/// The `get_tensor_config` lambda (llama-model.cpp:475 @b29c606e2).
fn tensorConfig(cx: *const Ctx) TensorConfig {
    if (cx.is_dsv4) {
        if (cx.m(.kv_cache) or cx.m(.dsv4_state)) return configImpl(cx, axis_mirrored, "", "");
        if (cx.m(.attn_sinks)) return configImpl(cx, axis_0, "attn_output_a.weight", "");
        if (cx.m(.attn_q_b_weight)) return configImpl(cx, axis_1, "attn_output_a.weight", "");
        if (cx.m(.attn_out_a_weight)) return configImpl(cx, axis_2, "", "");
        if (cx.m(.attn_out_b_weight)) return configImpl(cx, axis_0, "", "");
        if (cx.m(.ffn_up_shexp_weight) or cx.m(.ffn_gate_shexp_weight)) return configImpl(cx, axis_1, "ffn_down_shexp.weight", "");
        if (cx.m(.ffn_down_shexp_weight)) return configImpl(cx, axis_0, "ffn_down_shexp.weight", "");
    }

    // the qsa indexer has one key head and its projections are mirrored, so its cache cannot be split
    if (cx.m(.idx_cache)) return configImpl(cx, axis_mirrored, "", "");

    // the PLE table is model-level and its conv is mirrored, so every device runs the whole conv and needs the whole history
    if (cx.m(.ple_r_cache)) return configImpl(cx, axis_mirrored, "", "");

    // Standard attention.
    if (cx.m(.q_weight) or cx.m(.kv_weight)) return configImpl(cx, axis_1, "attn_output.weight", "ssm_out.weight");
    if (cx.m(.q_bias) or cx.m(.kv_bias)) return configImpl(cx, axis_0, "attn_output.weight", "ssm_out.weight");
    if (cx.m(.qkv_weight)) return configImpl(cx, axis_1, "attn_output.weight", "ssm_out.weight");
    if (cx.m(.qkv_bias)) return configImpl(cx, axis_0, "attn_output.weight", "ssm_out.weight");
    if (cx.m(.qk_norm)) return configImpl(cx, if (cx.tensor.ne[1] == 1) axis_mirrored else axis_1, "attn_output.weight", "");
    if (cx.m(.kv_cache) or cx.m(.attn_sinks)) return configImpl(cx, axis_0, "attn_output.weight", "");
    if (cx.m(.attn_out_weight)) return configImpl(cx, axis_0, "", "");
    if (cx.m(.attn_out_bias)) return configImpl(cx, axis_mirrored, "", "");

    if (cx.m(.attn_gate_weight)) return configImpl(cx, axis_1, "attn_output.weight", "ssm_out.weight");
    if (cx.m(.ssm_dt) or cx.m(.ssm_a)) return configImpl(cx, axis_0, "ssm_out.weight", "");
    if (cx.m(.ssm_alpha) or cx.m(.ssm_beta) or cx.m(.ssm_beta_alpha)) return configImpl(cx, axis_1, "ssm_out.weight", "");
    if (cx.m(.r_cache) or cx.m(.s_cache)) {
        if (cx.model.arch == .lfm2 or cx.model.arch == .lfm2moe) {
            // The LFM2 shortconv block runs fully mirrored, its conv state too.
            return configImpl(cx, axis_mirrored, "", "");
        }
        return configImpl(cx, axis_0, "ssm_out.weight", "");
    }
    if (cx.m(.ssm_conv1d)) return configImpl(cx, axis_1, "ssm_out.weight", "");
    if (cx.m(.ssm_out_weight)) return configImpl(cx, axis_0, "", "");

    // FFN.
    if (cx.m(.ffn_up_weight) or cx.m(.ffn_gate_weight)) return configImpl(cx, axis_1, "ffn_down.weight", "ffn_down_exps.weight");
    if (cx.m(.ffn_up_bias) or cx.m(.ffn_gate_bias)) return configImpl(cx, axis_0, "ffn_down.weight", "ffn_down_exps.weight");
    if (cx.m(.ffn_gate_up_weight)) return configImpl(cx, axis_1, "ffn_down.weight", "ffn_down_exps.weight");
    if (cx.m(.ffn_down_weight)) return configImpl(cx, axis_0, "ffn_down.weight", "ffn_down_exps.weight");
    if (cx.m(.ffn_down_bias)) return configImpl(cx, axis_mirrored, "", "");
    if (cx.m(.ffn_down_exps_bias)) return configImpl(cx, axis_partial, "ffn_down_exps.weight", "");

    // Output.
    if (cx.m(.output_weight)) {
        if (cx.is_dsv4) return configImpl(cx, axis_mirrored, "", "");
        return configImpl(cx, axis_1, "", "");
    }
    if (cx.m(.output_bias)) {
        impl.assert(cx.model.getTensor("output.weight") != null, "llama-model.cpp", 591, "output_weight != nullptr");
        return configImpl(cx, axis_0, "", "");
    }

    // Everything else.
    return configImpl(cx, axis_mirrored, "", "");
}

fn segs(out: *[max_segments]Segment, list: []const Segment) usize {
    @memcpy(out[0..list.len], list);
    return list.len;
}

/// The `get_split_segments` lambda (llama-model.cpp:599 @b29c606e2).
fn splitSegments(cx: *const Ctx, axis: usize, il: u32, out: *[max_segments]Segment) usize {
    const hparams = &cx.model.hparams;
    const t = cx.tensor;
    const a = cx.model.arch;
    if (a == .qwen3next or a == .qwen35 or a == .qwen35moe or a == .qwen4exp) {
        const head_k_dim: i64 = hparams.ssm_d_state;
        const head_v_dim: i64 = hparams.ssm_d_state;
        const n_k_heads: i64 = hparams.ssm_n_group;
        const n_v_heads: i64 = hparams.ssm_dt_rank;
        const key_dim = head_k_dim * n_k_heads;
        const value_dim = head_v_dim * n_v_heads;
        const d_conv_1: i64 = @as(i64, hparams.ssm_d_conv) - 1;

        // Qwen 3 Next broadcasts V heads as [k0_v0, k0_v1, k1_v2, k1_v3],
        // Qwen 3.5 as [k0_v0, k1_v1, k0_v2, k1_v3], which needs V segmented
        // on the scale of K.
        if (a == .qwen3next) {
            if (cx.m(.qkv_weight) or cx.m(.ssm_conv1d)) {
                impl.assert(t.ne[axis] == 2 * key_dim + value_dim, "llama-model.cpp", 614, "tensor->ne[axis] == 2*key_dim + value_dim");
                return segs(out, &.{ .{ .ne = key_dim, .nr = 2 }, .{ .ne = value_dim, .nr = 1 } });
            }
            if (cx.m(.r_cache)) return segs(out, &.{ .{ .ne = key_dim * d_conv_1, .nr = 2 }, .{ .ne = value_dim * d_conv_1, .nr = 1 } });
        } else {
            const head_ratio = @divTrunc(n_v_heads, n_k_heads);
            const hr: u32 = @truncate(@as(u64, @bitCast(head_ratio)));
            if (cx.m(.qkv_weight) or cx.m(.ssm_conv1d)) {
                impl.assert(t.ne[axis] == 2 * key_dim + value_dim, "llama-model.cpp", 623, "tensor->ne[axis] == 2*key_dim + value_dim");
                return segs(out, &.{.{ .ne = key_dim, .nr = 2 +% hr }});
            }
            if (cx.m(.attn_gate_weight) or cx.m(.ssm_out_weight)) return segs(out, &.{.{ .ne = key_dim, .nr = hr }});
            if (cx.m(.ssm_dt) or cx.m(.ssm_a) or cx.m(.ssm_alpha) or cx.m(.ssm_beta)) return segs(out, &.{.{ .ne = n_k_heads, .nr = hr }});
            if (cx.m(.r_cache)) return segs(out, &.{.{ .ne = key_dim * d_conv_1, .nr = 2 +% hr }});
            if (cx.m(.s_cache)) return segs(out, &.{.{ .ne = n_k_heads * head_v_dim * head_v_dim, .nr = hr }});
        }

        // The FFN is the same for both.
        if (cx.m(.ffn_gate_up_weight)) {
            const n_ff_exp: i64 = hparams.nFfExp(il);
            impl.assert(t.ne[axis] == 2 * n_ff_exp, "llama-model.cpp", 644, "tensor->ne[axis] == 2*n_ff_exp");
            return segs(out, &.{.{ .ne = n_ff_exp, .nr = 2 }});
        }
        return segs(out, &.{.{ .ne = t.ne[axis], .nr = 1 }});
    }

    if (cx.m(.qkv_weight) or cx.m(.qkv_bias)) {
        const n_embd: i64 = hparams.n_embd;
        const n_embd_gqa: i64 = hparams.nEmbdVGqa(il);
        impl.assert(hparams.nEmbdKGqa(0) == n_embd_gqa, "llama-model.cpp", 653, "hparams.n_embd_k_gqa() == n_embd_gqa");
        impl.assert(t.ne[axis] == n_embd + 2 * n_embd_gqa, "llama-model.cpp", 654, "tensor->ne[axis] == n_embd + 2*n_embd_gqa");
        return segs(out, &.{ .{ .ne = n_embd, .nr = 1 }, .{ .ne = n_embd_gqa, .nr = 2 } });
    }
    if (cx.m(.ffn_up_weight) or cx.m(.ffn_up_bias)) {
        const n_ff: i64 = hparams.nFf(il);
        // Fused up + gate tensors named "up", as Phi 3 has, need segmenting.
        if (t.ne[axis] == 2 * n_ff) return segs(out, &.{.{ .ne = n_ff, .nr = 2 }});
        return segs(out, &.{.{ .ne = t.ne[axis], .nr = 1 }});
    }
    if (cx.m(.ffn_gate_up_weight)) {
        const n_ff_exp: i64 = hparams.nFfExp(il);
        impl.assert(t.ne[axis] == 2 * n_ff_exp, "llama-model.cpp", 667, "tensor->ne[axis] == 2*n_ff_exp");
        return segs(out, &.{.{ .ne = n_ff_exp, .nr = 2 }});
    }
    return segs(out, &.{.{ .ne = t.ne[axis], .nr = 1 }});
}

fn fill(out: *[max_segments]i64, n: usize, v: i64) void {
    for (out[0..n]) |*g| g.* = v;
}

/// The `get_split_granularity` lambda (llama-model.cpp:673 @b29c606e2).
fn splitGranularity(cx: *const Ctx, blck_size: i64, il: u32, n_segments: usize, out: *[max_segments]i64) void {
    const hparams = &cx.model.hparams;
    const t = cx.tensor;
    if (hparams.isRecr(il)) {
        // Linear attention.
        const head_dim: i64 = hparams.ssm_d_state;
        const blck_size_perf = lcm(blck_size, 128);
        const granularity_qkv = lcm(blck_size_perf, head_dim);
        if (cx.m(.qkv_weight) or cx.m(.attn_gate_weight) or cx.m(.ssm_conv1d) or cx.m(.ssm_out_weight)) {
            return fill(out, n_segments, granularity_qkv);
        }
        if (cx.m(.ssm_dt) or cx.m(.ssm_a) or cx.m(.ssm_alpha) or cx.m(.ssm_beta)) {
            return fill(out, n_segments, @divTrunc(granularity_qkv, head_dim));
        }
        if (cx.m(.ssm_beta_alpha)) return fill(out, n_segments, 2 * @divTrunc(granularity_qkv, head_dim));
        if (cx.m(.r_cache)) return fill(out, n_segments, granularity_qkv * (@as(i64, hparams.ssm_d_conv) - 1));
        if (cx.m(.s_cache)) return fill(out, n_segments, granularity_qkv * head_dim);
    } else {
        // Regular attention.
        const n_gqa: u32 = hparams.nGqa(il);
        const n_embd_q: u32 = n_gqa *% hparams.nEmbdHeadK(il);

        // Head sizes like 80: raise the granularity only while it does not
        // underuse a device.
        var blck_size_perf = blck_size;
        while (blck_size_perf < 128 and blck_size_perf * @as(i64, @intCast(cx.ud.n_devices)) < n_embd_q) blck_size_perf *= 2;

        const granularity_q = lcm(n_embd_q, blck_size_perf);
        const granularity_head = @divTrunc(granularity_q, @as(i64, hparams.nEmbdHeadK(il)));
        if (cx.m(.attn_sinks)) {
            impl.assert(n_segments == 1, "llama-model.cpp", 711, "segments.size() == 1");
            if (cx.is_dsv4) {
                out[0] = hparams.nHead(il) / hparams.dsv4_o_group_count;
                return;
            }
            out[0] = granularity_head;
            return;
        }

        if (cx.is_dsv4) {
            if (cx.m(.attn_q_b_weight)) {
                impl.assert(n_segments == 1, "llama-model.cpp", 720, "segments.size() == 1");
                // The grouped output projection needs whole groups of heads.
                const n_head_group: i64 = hparams.nHead(il) / hparams.dsv4_o_group_count;
                out[0] = n_head_group * hparams.nEmbdHeadK(il);
                return;
            }
            if (cx.m(.attn_out_a_weight)) {
                impl.assert(n_segments == 1, "llama-model.cpp", 726, "segments.size() == 1");
                out[0] = 1;
                return;
            }
            if (cx.m(.attn_out_b_weight)) {
                impl.assert(n_segments == 1, "llama-model.cpp", 730, "segments.size() == 1");
                // The boundaries align with wo_a's per-group split.
                impl.assert(@rem(@as(i64, hparams.dsv4_o_lora_rank), blck_size) == 0, "llama-model.cpp", 732, "hparams.dsv4_o_lora_rank % blck_size == 0");
                out[0] = hparams.dsv4_o_lora_rank;
                return;
            }
        }
        if (cx.m(.q_weight) or cx.m(.q_bias)) {
            impl.assert(n_segments == 1, "llama-model.cpp", 737, "segments.size() == 1");
            // Q gate tensors double the granularity.
            const a = cx.model.arch;
            if (a == .qwen3next or a == .qwen35 or a == .qwen35moe or a == .qwen4exp) {
                out[0] = lcm(2 * @as(i64, n_embd_q), blck_size_perf);
                return;
            }
            out[0] = granularity_q;
            return;
        }
        if (cx.m(.attn_out_weight)) {
            impl.assert(n_segments == 1, "llama-model.cpp", 746, "segments.size() == 1");
            out[0] = granularity_q;
            return;
        }
        if (cx.m(.attn_gate_weight)) {
            impl.assert(n_segments == 1, "llama-model.cpp", 750, "segments.size() == 1");
            out[0] = if (t.ne[1] == hparams.nHead(il)) granularity_head else granularity_q;
            return;
        }

        const granularity_kv = @divTrunc(granularity_q, @as(i64, n_gqa));
        if (cx.m(.kv_weight) or cx.m(.kv_bias) or cx.m(.kv_cache)) {
            impl.assert(n_segments == 1, "llama-model.cpp", 761, "segments.size() == 1");
            out[0] = granularity_kv;
            return;
        }
        if (cx.m(.qkv_weight) or cx.m(.qkv_bias)) {
            impl.assert(n_segments == 2, "llama-model.cpp", 765, "segments.size() == 2");
            out[0] = granularity_q;
            out[1] = granularity_kv;
            return;
        }
    }

    // FFN.
    if (cx.m(.ffn_up_weight) or cx.m(.ffn_up_bias) or cx.m(.ffn_gate_weight) or cx.m(.ffn_gate_bias) or
        cx.m(.ffn_gate_up_weight) or cx.m(.ffn_down_weight) or cx.m(.ffn_up_shexp_weight) or
        cx.m(.ffn_gate_shexp_weight) or cx.m(.ffn_down_shexp_weight))
    {
        const blck_size_perf = lcm(blck_size, 128);
        impl.assert(n_segments == 1, "llama-model.cpp", 779, "segments.size() == 1");
        out[0] = blck_size_perf;
        return;
    }

    // Everything else.
    impl.assert(n_segments == 1, "llama-model.cpp", 784, "segments.size() == 1");
    out[0] = 1;
}

/// Ports `llama_meta_device_get_split_state` (llama-model.cpp:373
/// @b29c606e2): which slice of `tensor` each device holds.
///
/// Parameters:
/// - `tensor`: a weight or a memory tensor, by its name.
/// - `userdata`: a `*Userdata`.
///
/// Return: the split state, by value as the C callback returns it.
pub fn getSplitState(tensor: ?*const Tensor, userdata: ?*anyopaque) callconv(.c) SplitState {
    const ud: *const Userdata = @ptrCast(@alignCast(userdata.?));
    const model = ud.model.?;
    const hparams = &model.hparams;
    const t = tensor.?;
    const cx: Ctx = .{
        .tensor = t,
        .name = std.mem.sliceTo(&t.name, 0),
        .ud = ud,
        .model = model,
        .is_dsv4 = model.arch == .deepseek4 or (model.arch == .dflash and hparams.dsv4_hc_mult > 0),
    };
    const n_devices = ud.n_devices;

    var split_state = std.mem.zeroes(SplitState);
    const tc = tensorConfig(&cx);
    split_state.axis = tc.axis;
    if (split_state.axis < c.GGML_MAX_DIMS) {
        const blck_size: i64 = c.ggml_blck_size(tc.tensor_axis_0.type);
        const tensor_split = model.params.tensor_split;
        var tensor_split_scan: [c.GGML_BACKEND_META_MAX_DEVICES]f32 = undefined;
        for (0..n_devices) |j| {
            tensor_split_scan[j] = if (tensor_split == null) 0.0 else tensor_split[(j + tc.rotation) % n_devices];
            if (j > 0) tensor_split_scan[j] += tensor_split_scan[j - 1];
        }
        const scan_back = tensor_split_scan[n_devices - 1];
        var segments: [max_segments]Segment = undefined;
        const n_segments = splitSegments(&cx, split_state.axis, tc.il, &segments);
        var granularity: [max_segments]i64 = undefined;
        splitGranularity(&cx, blck_size, tc.il, n_segments, &granularity);
        for (0..n_segments) |is| {
            const ne_s = segments[is].ne;
            const nr_s = segments[is].nr;
            const g_s = granularity[is];
            var low: i64 = 0;
            var j: usize = 0;
            while (j < n_devices - 1) : (j += 1) {
                // Both arms of the C++'s conditional convert to float.
                const high_f: f32 = if (scan_back == 0.0)
                    @floatFromInt(@as(u64, @bitCast(ne_s)) *% (j + 1) / n_devices)
                else
                    @as(f32, @floatFromInt(ne_s)) * tensor_split_scan[j] / scan_back;
                var high: i64 = @intFromFloat(high_f);
                if (@rem(high, g_s) != 0) high -= @rem(high, g_s);
                split_state.ne[is * n_devices + (j + tc.rotation) % n_devices] = high - low;
                low = high;
            }
            split_state.ne[is * n_devices + (j + tc.rotation) % n_devices] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = @intCast(n_segments);
    } else {
        @memset(&split_state.ne, 0);
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
        if (split_state.axis == axis_partial) {
            impl.assert(tc.tensor_axis_0 != t, "llama-model.cpp", 829, "tc.tensor_axis_0 != tensor");
            const source = getSplitState(tc.tensor_axis_0, userdata);
            impl.assert(source.axis < c.GGML_MAX_DIMS, "llama-model.cpp", 831, "source_split_state.axis >= 0 && source_split_state.axis < GGML_MAX_DIMS");
            for (0..n_devices) |j| {
                for (0..source.n_segments) |is| {
                    split_state.ne[j] += source.ne[is * n_devices + j] * source.nr[is];
                }
            }
        }
    }
    return split_state;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "every pattern compiles and matches its own kind of name" {
    const p = getPatterns();
    var match = regex.Match.init();
    defer match.deinit();
    try std.testing.expect(p[@backingInt(P.q_weight)].match("blk.3.attn_q.weight", &match));
    try std.testing.expect(!p[@backingInt(P.q_weight)].match("blk.3.attn_qkv.weight", &match));
    try std.testing.expect(p[@backingInt(P.kv_cache)].match("cache_k_l12", &match));
    try std.testing.expect(p[@backingInt(P.s_cache)].match("cache_s_l0", &match));
    try std.testing.expect(p[@backingInt(P.ffn_up_weight)].match("blk.0.ffn_up_exps.weight", &match));
}

test "stoull reads the leading digits" {
    try std.testing.expectEqual(@as(u64, 12), stoull("12.attn"));
    try std.testing.expectEqual(@as(u64, 7), stoull("7"));
}
