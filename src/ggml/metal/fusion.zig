//! The Metal backend's fusion table: the one place a fusable subgraph is
//! declared.
//!
//! # Provenance
//!
//! Ports `llama.cpp/ggml/src/ggml-metal/ggml-metal-fusion.cpp`, with the
//! types and constants of `llama.cpp/ggml/src/ggml-metal/ggml-metal-fusion.h`,
//! at v0.9.0 (`cd959bae9`). Each declaration below names the C++ it
//! replaces and the line it began at.
//!
//! # Why one table
//!
//! The graph optimizer (`common.zig`'s `ggml_graph_optimize`, through
//! `ggml_metal_fusion_max`) packs fusable runs before it reorders, and the
//! op encoders (`ops.zig`, through `ggml_metal_fusion_next`) fuse them when
//! they encode. Before v0.4.1 each phase had its own idea of what fused;
//! now both consult `fusions`, so they cannot disagree.
//!
//! # Its contract is C, and two callers stay Objective-C
//!
//! All 18 of the C++'s exports are `extern "C"`. `ggml-metal-device.m`
//! owns a `ggml_metal_fusion_info` per device and `ggml-metal-context.m`
//! reads its flags, labels and counts; both stay Objective-C (Decision
//! 13) and call the `ggml_metal_fusion_info_*` functions below by name.
//! The struct itself is opaque to them, so `Info` is free to be a Zig
//! struct. Since v0.5.0 so is `Fusion`: the header declares it opaque,
//! drops `ggml_metal_fusion_all`, and hands out the id through
//! `ggml_metal_fusion_get_id`. Its address still identifies an entry:
//! `count_fusion` counts by offset into the table.
//!
//! # Raw and non-empty op sequences
//!
//! v0.5.0's MoE patterns span RESHAPE and VIEW nodes the encoders never
//! see. Each entry keeps both sequences: `ops_all`, the raw graph's, which
//! `add_alloc_deps` and the MoE checks walk, and `ops`, its non-empty
//! subsequence, which `ggml_metal_fusion_next` matches and the labels
//! name.

const std = @import("std");

const impl = @import("../impl.zig");
const c = impl.c;

const mc = @import("device_c.zig");
const types = @import("../types.zig");
const graph = @import("../graph.zig");
const backend = @import("../backend.zig");
const common = @import("common.zig");

const Tensor = c.ggml_tensor;
const CGraph = impl.CGraph;

/// The allocator ported ggml uses — libc's, as in the rest of `metal/`.
const allocator = std.heap.c_allocator;

/// Ports `GGML_METAL_FUSION_MAX` (ggml-metal-fusion.h:23 @cd959bae9): the
/// most nodes one kernel fuses, and the longest run the optimizer packs.
pub const max = 16;

/// Ports `ggml_metal_fusion_mode` (ggml-metal-fusion.h:25 @cd959bae9).
pub const Mode = enum(c_uint) {
    /// structural checks for the graph optimizer, which runs before the
    /// graph tensors are allocated (weights already are); a check may skip
    /// conditions that differ between batch sizes, and so accept more than
    /// `full`
    structural = 0,
    /// full checks, including buffer placement; used by the op encoders
    full = 1,
};

/// Ports `ggml_metal_fusion_id` (ggml-metal-fusion.h:34 @cd959bae9): which
/// kernel an encoder uses for a matched pattern.
pub const Id = enum(c_uint) {
    none = 0,
    /// NORM/RMS_NORM + MUL
    norm_mul = 1,
    /// NORM/RMS_NORM + MUL + ADD
    norm_mul_add = 2,
    /// NORM/RMS_NORM + SCALE
    norm_scale = 3,
    /// ADD x N (N in [2, 7])
    add_chain = 4,
    /// MUL + SIN + SQR + MUL + ADD
    snake = 5,
    /// GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    gdn_cache = 6,
    /// SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    topk_moe = 7,
    /// MUL + expert VIEWs + ADD chain (MoE output reduction)
    moe_reduce = 8,
    /// SSM_CONV + UNARY (silu)
    ssm_conv_silu = 9,
    /// MUL_MAT + ADD (residual added in the few-row MMA store)
    mul_mat_add = 10,
};

/// The `check` callback's type: `nodes[j]` is the pattern's `j`-th
/// non-empty node; `node_idxs[idx + j]` is its raw graph index.
pub const CheckFn = *const fn (
    fusion: *const Fusion,
    nodes: [*]const *const Tensor,
    gf: *const CGraph,
    node_idxs: [*]const c_int,
    idx: c_int,
    mode: Mode,
) bool;

/// Ports `struct ggml_metal_fusion` (ggml-metal-fusion.cpp:28 @cd959bae9).
///
/// Since v0.5.0 the header declares it opaque, so the layout is ours. Its
/// address is still what identifies an entry: `count_fusion` takes the
/// offset from the start of `fusions`.
pub const Fusion = struct {
    id: Id,

    /// non-empty op sequence, derived from `ops_all`
    ops: []const c.enum_ggml_op,
    /// full raw op sequence (may include empty RESHAPE/VIEW nodes)
    ops_all: []const c.enum_ggml_op,
    /// additional fused output nodes, relative to `ops`
    outs: []const c_int,

    /// if unsafe: the generic chain/shape + `ggml_can_fuse_subgraph`
    /// checks are skipped and `check` is the sole validator (used for
    /// patterns that are not elision chains, e.g. the gdn + cache-cpy
    /// write-through fusion)
    unsafe: bool,

    /// extra backend constraints on top of `ggml_can_fuse_subgraph`
    check: ?CheckFn,
};

/// Ports `ggml_metal_fusion_filter_ops` (ggml-metal-fusion.cpp:16
/// @cd959bae9): the non-empty op sequence of a raw one. The table is a
/// constant, so this runs at compile time.
fn filterOps(comptime ops_all: []const c.enum_ggml_op) []const c.enum_ggml_op {
    comptime {
        var ops: [ops_all.len]c.enum_ggml_op = undefined;
        var n: usize = 0;
        for (ops_all) |op| {
            if (!impl.opIsEmpty(op)) {
                ops[n] = op;
                n += 1;
            }
        }
        const res = ops[0..n].*;
        return &res;
    }
}

/// Ports `ggml_metal_fusion_get_id` (ggml-metal-fusion.cpp:69 @cd959bae9):
/// the pattern's identifier, for callers that see the struct as opaque.
pub export fn ggml_metal_fusion_get_id(fusion: *const Fusion) callconv(.c) Id {
    return fusion.id;
}

// -----------------------------------------------------------------------------
// Helpers

/// `src[i]`, narrowed: a `[*c]` must go through `impl.one` before an array
/// field is indexed through it (`CLAUDE.md`).
inline fn src(t: *const Tensor, i: usize) *const Tensor {
    return impl.one(Tensor, t.src[i]);
}

/// `gf->nodes[i]`.
inline fn gnode(gf: *const CGraph, i: c_int) *const Tensor {
    return gf.nodes[@intCast(i)].?;
}

/// Ports `ggml_metal_fusion_same_buffer` (ggml-metal-fusion.cpp:76
/// @cd959bae9): true if two tensors live in the same Metal buffer.
fn sameBuffer(a_opt: ?*const Tensor, b_opt: ?*const Tensor) bool {
    const a = a_opt orelse return false;
    const b = b_opt orelse return false;

    const ba = impl.one(c.ggml_backend_buffer, if (a.view_src != null) impl.one(Tensor, a.view_src).buffer else a.buffer);
    const bb = impl.one(c.ggml_backend_buffer, if (b.view_src != null) impl.one(Tensor, b.view_src).buffer else b.buffer);

    const ca: *mc.Buffer = @ptrCast(@alignCast(ba.context.?));
    const cb: *mc.Buffer = @ptrCast(@alignCast(bb.context.?));

    return mc.ggml_metal_buffer_get_id(ca, a).metal == mc.ggml_metal_buffer_get_id(cb, b).metal;
}

/// Ports `ggml_metal_fusion_overlap` (ggml-metal-fusion.cpp:91
/// @cd959bae9): true if the memory of two tensors overlaps in the same
/// Metal buffer.
fn overlap(a: *const Tensor, b: *const Tensor) bool {
    const ba = impl.one(c.ggml_backend_buffer, if (a.view_src != null) impl.one(Tensor, a.view_src).buffer else a.buffer);
    const bb = impl.one(c.ggml_backend_buffer, if (b.view_src != null) impl.one(Tensor, b.view_src).buffer else b.buffer);

    const bid_a = mc.ggml_metal_buffer_get_id(@ptrCast(@alignCast(ba.context.?)), a);
    const bid_b = mc.ggml_metal_buffer_get_id(@ptrCast(@alignCast(bb.context.?)), b);

    if (bid_a.metal == null or bid_a.metal != bid_b.metal) {
        return false;
    }

    return if (bid_a.offs <= bid_b.offs)
        bid_b.offs - bid_a.offs < c.ggml_nbytes(a)
    else
        bid_a.offs - bid_b.offs < c.ggml_nbytes(b);
}

// -----------------------------------------------------------------------------
// Pattern checks

/// Ports `ggml_metal_fusion_check_norm` (ggml-metal-fusion.cpp:111
/// @cd959bae9).
///
/// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match
/// the norm input width, be contiguous rows, and the fused outputs must
/// stay F32. NORM/RMS_NORM + SCALE: a plain F32 scale of the norm.
fn checkNorm(fusion: *const Fusion, nodes: [*]const *const Tensor, gf: *const CGraph, node_idxs: [*]const c_int, idx: c_int, mode: Mode) bool {
    _ = mode;
    _ = gf;
    _ = node_idxs;
    _ = idx;

    impl.assert(fusion.ops.len >= 2, "fusion->ops.size() >= 2");

    if (fusion.id == .norm_scale) {
        impl.assert(fusion.ops.len == 2, "fusion->ops.size() == 2");

        const scale = nodes[1];
        if (scale.op != c.GGML_OP_SCALE or scale.src[0] != nodes[0] or scale.src[1] != null or
            scale.type != c.GGML_TYPE_F32)
        {
            return false;
        }

        return true;
    }

    var j: usize = 1;
    while (j < fusion.ops.len) : (j += 1) {
        // the fused MUL/ADD must read the previous node as src0
        if (nodes[j].src[0] != nodes[j - 1]) {
            return false;
        }

        // the weight/bias must have the same row width as the norm input
        if (src(nodes[j], 1).ne[0] != nodes[0].ne[0]) {
            return false;
        }

        if (!types.ggml_is_contiguous_rows(src(nodes[j], 1))) {
            return false;
        }

        if (nodes[j].type != c.GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

/// Ports `ggml_metal_fusion_check_ssm_conv_silu` (ggml-metal-fusion.cpp:161
/// @cd959bae9): SSM_CONV + UNARY (silu).
fn checkSsmConvSilu(fusion: *const Fusion, nodes: [*]const *const Tensor, gf: *const CGraph, node_idxs: [*]const c_int, idx: c_int, mode: Mode) bool {
    _ = fusion;
    _ = gf;
    _ = node_idxs;
    _ = idx;
    _ = mode;

    const conv = nodes[0];
    const un = nodes[1];

    if (conv.op != c.GGML_OP_SSM_CONV or un.op != c.GGML_OP_UNARY or un.src[0] != conv or un.src[1] != null) {
        return false;
    }

    if (types.ggml_get_unary_op(un) != c.GGML_UNARY_OP_SILU) {
        return false;
    }

    if (conv.type != c.GGML_TYPE_F32 or un.type != c.GGML_TYPE_F32 or !types.ggml_is_contiguous_rows(un)) {
        return false;
    }

    return true;
}

/// Ports `ggml_metal_fusion_check_add_chain` (ggml-metal-fusion.cpp:194
/// @cd959bae9).
///
/// ADD x N: each ADD reads the previous ADD as src0, and all addends must
/// share layout (and, in FULL mode, live in the same Metal buffer).
fn checkAddChain(fusion: *const Fusion, nodes: [*]const *const Tensor, gf: *const CGraph, node_idxs: [*]const c_int, idx: c_int, mode: Mode) bool {
    _ = gf;
    _ = node_idxs;
    _ = idx;
    impl.assert(fusion.ops.len >= 2, "fusion->ops.size() >= 2");

    var j: usize = 1;
    while (j < fusion.ops.len) : (j += 1) {
        if (nodes[j].src[0] != nodes[j - 1]) {
            return false;
        }

        if (!impl.areSameLayout(src(nodes[j], 1), src(nodes[j - 1], 1))) {
            return false;
        }

        if (mode == .full) {
            if (!sameBuffer(src(nodes[j], 1), src(nodes[0], 1))) {
                return false;
            }
        }
    }

    return true;
}

/// Ports `ggml_metal_fusion_check_gdn_cache` (ggml-metal-fusion.cpp:230
/// @cd959bae9).
///
/// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state
/// snapshots into the recurrent cache, so the gdn kernel writes them
/// straight to the cache and the cpy is elided. The gdn output has other
/// consumers (the attn scores view), so unlike the other patterns this is
/// not an elision chain: the structural checks live entirely here
/// (`unsafe = true`).
fn checkGdnCache(fusion: *const Fusion, nodes: [*]const *const Tensor, gf: *const CGraph, node_idxs: [*]const c_int, idx: c_int, mode: Mode) bool {
    _ = fusion;
    _ = gf;
    _ = node_idxs;
    _ = idx;

    const gdn = nodes[0];
    const cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn.type != c.GGML_TYPE_F32 or (gdn.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    if (cpy.op != c.GGML_OP_CPY or (cpy.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    const g2 = src(gdn, 2);
    const S_v: i64 = g2.ne[0];
    const H: i64 = g2.ne[1];
    const n_tokens: i64 = g2.ne[2];
    const n_seqs: i64 = g2.ne[3];
    const K: i64 = impl.getOpParamsI32(gdn, 0);
    const tail_off: usize = types.ggml_row_size(c.GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const D: i64 = S_v * S_v * H;
    const n_written: i64 = @min(n_tokens, K);

    const s = src(cpy, 0); // gdn snapshot tail view
    const dst = src(cpy, 1); // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (s.op != c.GGML_OP_VIEW or s.view_src != gdn or
        s.view_offs != tail_off or !c.ggml_is_contiguous(s))
    {
        return false;
    }

    const expected_ne = [c.GGML_MAX_DIMS]i64{ D, n_seqs, n_written, 1 };
    if (dst.type != c.GGML_TYPE_F32 or
        !std.mem.eql(i64, &expected_ne, &dst.ne) or
        dst.nb[0] != types.ggml_type_size(c.GGML_TYPE_F32) or
        dst.nb[1] != types.ggml_row_size(c.GGML_TYPE_F32, D))
    {
        return false;
    }

    if (mode == .full) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst.data == null) {
            return false;
        }
    }

    return true;
}

/// Ports `ggml_metal_fusion_check_snake` (ggml-metal-fusion.cpp:292
/// @cd959bae9): MUL + SIN + SQR + MUL + ADD (snake activation).
fn checkSnake(fusion: *const Fusion, nodes: [*]const *const Tensor, gf: *const CGraph, node_idxs: [*]const c_int, idx: c_int, mode: Mode) bool {
    _ = fusion;
    _ = mode;
    _ = gf;
    _ = node_idxs;
    _ = idx;

    const mul0 = nodes[0];
    const sin_node = nodes[1];
    const sqr = nodes[2];
    const mul1 = nodes[3];
    const add = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const x = if (c.ggml_are_same_shape(mul0, src(mul0, 0))) src(mul0, 0) else src(mul0, 1);
    const a = if (x == src(mul0, 0)) src(mul0, 1) else src(mul0, 0);

    // mul1 reads sqr and inv_b in either operand order
    const inv_b = if (mul1.src[0] == sqr) src(mul1, 1) else src(mul1, 0);

    // closure check: the trailing add reads the same x as the leading mul
    const x_in_add = if (add.src[0] == mul1) src(add, 1) else src(add, 0);

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const types_ok =
        (x.type == c.GGML_TYPE_F32 or x.type == c.GGML_TYPE_F16 or x.type == c.GGML_TYPE_BF16) and
        (a.type == c.GGML_TYPE_F32) and (inv_b.type == c.GGML_TYPE_F32) and
        (mul0.type == x.type) and (sin_node.type == x.type) and
        (sqr.type == x.type) and (mul1.type == x.type) and
        (add.type == x.type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const shape_ok = c.ggml_are_same_shape(a, inv_b) and a.ne[0] == 1 and a.ne[1] == x.ne[1];
    const dim_ok =
        (x.ne[2] == 1) and (x.ne[3] == 1) and
        (add.ne[2] == 1) and (add.ne[3] == 1) and
        (a.ne[2] == 1) and (a.ne[3] == 1) and
        (inv_b.ne[2] == 1) and (inv_b.ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const contig_ok =
        c.ggml_is_contiguous(x) and c.ggml_is_contiguous(add) and
        c.ggml_is_contiguous(a) and c.ggml_is_contiguous(inv_b);

    return types_ok and shape_ok and dim_ok and contig_ok and x_in_add == x;
}

/// Ports `GGML_METAL_TOPK_MOE_MAX_EXPERTS` (ggml-metal-fusion.cpp:346
/// @cd959bae9).
const topk_moe_max_experts = 1024;

// Ports `ops_topk_moe`, `ops_topk_moe_scale`, `ops_topk_moe_norm` and
// `ops_topk_moe_norm_scale` (ggml-metal-fusion.cpp:351, 354, 357, 361
// @cd959bae9): SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale)
// for MoE routing. A multi-output elision chain: the fused kernel writes
// both the selected expert ids and the gathered/normalized routing
// weights.
const ops_topk_moe = [_]c.enum_ggml_op{
    c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE, c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS,
};
const ops_topk_moe_scale = [_]c.enum_ggml_op{
    c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE, c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS, c.GGML_OP_SCALE,
};
const ops_topk_moe_norm = [_]c.enum_ggml_op{
    c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE,  c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS,
    c.GGML_OP_RESHAPE,  c.GGML_OP_SUM_ROWS, c.GGML_OP_CLAMP,   c.GGML_OP_DIV,  c.GGML_OP_RESHAPE,
};
const ops_topk_moe_norm_scale = [_]c.enum_ggml_op{
    c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE,  c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS,
    c.GGML_OP_RESHAPE,  c.GGML_OP_SUM_ROWS, c.GGML_OP_CLAMP,   c.GGML_OP_DIV,  c.GGML_OP_RESHAPE,
    c.GGML_OP_SCALE,
};

/// Ports `ggml_metal_fusion_check_topk_moe` (ggml-metal-fusion.cpp:366
/// @cd959bae9).
///
/// The table matches on the non-empty node sequence; this check walks the
/// raw graph, RESHAPE and VIEW nodes included, and holds it to `ops_all`.
fn checkTopkMoe(fusion: *const Fusion, nodes: [*]const *const Tensor, gf: *const CGraph, node_idxs: [*]const c_int, idx: c_int, mode: Mode) bool {
    impl.assert(fusion.ops.len >= 3, "fusion->ops.size() >= 3");
    _ = nodes;

    const n_ops: c_int = @intCast(fusion.ops.len);

    const with_norm = n_ops >= 6;
    const with_scale = n_ops == 4 or n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    const ops_all = fusion.ops_all;

    const raw_start: c_int = node_idxs[@intCast(idx)];
    var raw_end: c_int = node_idxs[@intCast(idx + n_ops - 1)];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    if (with_norm and !with_scale) {
        if (raw_end + 1 >= gf.n_nodes) {
            return false;
        }
        const trailing_reshape = gnode(gf, raw_end + 1);
        if (trailing_reshape.op != c.GGML_OP_RESHAPE or trailing_reshape.src[0] != gnode(gf, raw_end)) {
            return false;
        }
        raw_end += 1;
    }

    const raw_count: c_int = raw_end - raw_start + 1;
    if (raw_count != @as(c_int, @intCast(ops_all.len))) {
        return false;
    }

    var raw_idxs: [max]c_int = undefined;
    var i: c_int = 0;
    while (i < raw_count) : (i += 1) {
        raw_idxs[@intCast(i)] = raw_start + i;
        if (gnode(gf, raw_start + i).op != ops_all[@intCast(i)]) {
            return false;
        }
    }

    const softmax = gnode(gf, raw_start);
    const probs_reshaped = gnode(gf, raw_start + 1);
    const argsort = gnode(gf, raw_start + 2);
    const ids = gnode(gf, raw_start + 3);
    const get_rows = gnode(gf, raw_start + 4);
    const out = gnode(gf, raw_end);
    const logits = src(softmax, 0);

    // the fused kernel implements plain softmax only
    const scale = impl.getOpParamsF32(softmax, 0);
    const max_bias = impl.getOpParamsF32(softmax, 1);
    if (scale != 1.0 or max_bias != 0.0 or softmax.src[1] != null or softmax.src[2] != null) {
        return false;
    }

    if (logits.type != c.GGML_TYPE_F32 or softmax.type != c.GGML_TYPE_F32 or
        out.type != c.GGML_TYPE_F32 or ids.type != c.GGML_TYPE_I32)
    {
        return false;
    }

    const n_expert: i64 = logits.ne[0];
    const n_tokens: i64 = logits.ne[1];
    const n_expert_used: i64 = ids.ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_expert <= 0 or n_expert_used <= 0 or n_expert_used > n_expert or
        n_expert > topk_moe_max_experts or n_expert_used > topk_moe_max_experts)
    {
        return false;
    }

    if (logits.ne[2] != 1 or logits.ne[3] != 1 or
        ids.ne[1] != n_tokens or ids.ne[2] != 1 or ids.ne[3] != 1 or
        out.ne[0] != 1 or out.ne[1] != n_expert_used or out.ne[2] != n_tokens or out.ne[3] != 1)
    {
        return false;
    }

    if (!c.ggml_is_contiguous(logits) or !c.ggml_is_contiguous(out) or
        ids.nb[0] != types.ggml_type_size(c.GGML_TYPE_I32) or
        ids.nb[1] != types.ggml_type_size(c.GGML_TYPE_I32) *% @as(usize, @bitCast(n_expert)))
    {
        return false;
    }

    if (probs_reshaped.src[0] != softmax or argsort.src[0] != softmax or
        ids.src[0] != argsort or get_rows.src[0] != probs_reshaped or get_rows.src[1] != ids)
    {
        return false;
    }

    if (with_norm) {
        const weights_reshaped = gnode(gf, raw_start + 5);
        const sum_rows = gnode(gf, raw_start + 6);
        const clamp = gnode(gf, raw_start + 7);
        const div = gnode(gf, raw_start + 8);
        const out_reshaped = gnode(gf, raw_start + 9);

        if (weights_reshaped.src[0] != get_rows or sum_rows.src[0] != weights_reshaped or
            clamp.src[0] != sum_rows or div.src[0] != weights_reshaped or div.src[1] != clamp or
            out_reshaped.src[0] != div)
        {
            return false;
        }

        if (with_scale) {
            const scale_node = gnode(gf, raw_start + 10);
            if (scale_node.src[0] != out_reshaped) {
                return false;
            }
        }
    } else if (with_scale) {
        const scale_node = gnode(gf, raw_start + 5);
        if (scale_node.src[0] != get_rows) {
            return false;
        }
    }

    const outputs = [2]c_int{ raw_start + 3, raw_end };
    if (!graph.ggml_can_fuse_subgraph_ext(gf, &raw_idxs, raw_count, ops_all.ptr, &outputs, 2)) {
        return false;
    }

    if (mode == .full) {
        if (logits.data == null or out.data == null or ids.data == null) {
            return false;
        }
    }

    return true;
}

/// Ports `GGML_METAL_MOE_REDUCE_MAX_EXPERTS` (ggml-metal-fusion.cpp:503
/// @cd959bae9).
const moe_reduce_max_experts = 8;

/// Ports `struct ggml_metal_moe_reduce_match` (ggml-metal-fusion.cpp:505
/// @cd959bae9).
const MoeReduceMatch = struct {
    experts: *const Tensor,
    weights: *const Tensor,
    dst: *const Tensor,
    node_count: c_int,
};

/// Ports `ggml_metal_fusion_match_moe_reduce` (ggml-metal-fusion.cpp:512
/// @cd959bae9): a MUL by the routing weights, one VIEW per expert, then
/// one ADD per additional expert summing the views in order.
///
/// Return: the match, or null.
fn matchMoeReduce(gf: *const CGraph, node_idx: c_int, ops_all: []const c.enum_ggml_op) ?MoeReduceMatch {
    if (node_idx < 0 or node_idx + @as(c_int, @intCast(ops_all.len)) > gf.n_nodes) {
        return null;
    }

    const mul = gnode(gf, node_idx);
    if (mul.op != c.GGML_OP_MUL or mul.type != c.GGML_TYPE_F32) {
        return null;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    const raw_count: c_int = @intCast(ops_all.len);
    const n_expert_used: c_int = @divTrunc(raw_count, 2);

    if (n_expert_used < 2 or n_expert_used > moe_reduce_max_experts or
        raw_count != 2 * n_expert_used)
    {
        return null;
    }

    var n_views: c_int = 0;
    while (node_idx + 1 + n_views < gf.n_nodes and
        gnode(gf, node_idx + 1 + n_views).op == c.GGML_OP_VIEW)
    {
        n_views += 1;
    }

    if (n_views != n_expert_used) {
        return null;
    }

    var i: c_int = n_expert_used + 1;
    while (i < raw_count) : (i += 1) {
        if (gnode(gf, node_idx + i).op != c.GGML_OP_ADD) {
            return null;
        }
    }

    var raw_idxs: [max]c_int = undefined;
    i = 0;
    while (i < raw_count) : (i += 1) {
        raw_idxs[@intCast(i)] = node_idx + i;
        if (gnode(gf, node_idx + i).op != ops_all[@intCast(i)]) {
            return null;
        }
    }

    const experts = src(mul, 0);
    const weights = src(mul, 1);
    const dst = gnode(gf, node_idx + raw_count - 1);

    if (experts.type != c.GGML_TYPE_F32 or weights.type != c.GGML_TYPE_F32 or dst.type != c.GGML_TYPE_F32) {
        return null;
    }

    const n_embd: i64 = experts.ne[0];
    const n_tokens: i64 = experts.ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_embd <= 0 or experts.ne[1] != n_expert_used or experts.ne[3] != 1 or
        weights.ne[0] != 1 or weights.ne[1] != n_expert_used or weights.ne[2] != n_tokens or weights.ne[3] != 1 or
        dst.ne[0] != n_embd or dst.ne[1] != n_tokens or dst.ne[2] != 1 or dst.ne[3] != 1)
    {
        return null;
    }

    if (!c.ggml_is_contiguous(experts) or !c.ggml_is_contiguous(weights) or !c.ggml_is_contiguous(dst)) {
        return null;
    }

    i = 1;
    while (i <= n_expert_used) : (i += 1) {
        const view = gnode(gf, node_idx + i);
        if (view.view_src != mul or view.src[0] != mul or
            view.view_offs != @as(usize, @intCast(i - 1)) *% mul.nb[1] or
            view.ne[0] != n_embd or view.ne[1] != n_tokens or
            view.nb[1] != mul.nb[2])
        {
            return null;
        }
    }

    var prev_add: ?*const Tensor = null;
    var j: c_int = 1;
    while (j < n_expert_used) : (j += 1) {
        const add = gnode(gf, node_idx + n_expert_used + j);
        const rhs = gnode(gf, node_idx + j + 1);
        const lhs: ?*const Tensor = if (j == 1) gnode(gf, node_idx + 1) else prev_add;
        if (add.src[0] != lhs or add.src[1] != rhs) {
            return null;
        }
        prev_add = add;
    }

    const outputs = [1]c_int{node_idx + raw_count - 1};
    if (!graph.ggml_can_fuse_subgraph_ext(gf, &raw_idxs, raw_count, ops_all.ptr, &outputs, 1)) {
        return null;
    }

    return .{
        .experts = experts,
        .weights = weights,
        .dst = dst,
        .node_count = raw_count,
    };
}

/// Ports `ggml_metal_fusion_check_moe_reduce` (ggml-metal-fusion.cpp:612
/// @cd959bae9).
fn checkMoeReduce(fusion: *const Fusion, nodes: [*]const *const Tensor, gf: *const CGraph, node_idxs: [*]const c_int, idx: c_int, mode: Mode) bool {
    _ = nodes;

    const match = matchMoeReduce(gf, node_idxs[@intCast(idx)], fusion.ops_all) orelse return false;

    if (@as(i64, @intCast(fusion.ops.len)) != match.experts.ne[1]) {
        return false;
    }

    const raw_end = node_idxs[@intCast(idx)] + match.node_count - 1;
    if (node_idxs[@as(usize, @intCast(idx)) + fusion.ops.len - 1] != raw_end) {
        return false;
    }

    if (mode == .full) {
        if (match.experts.data == null or match.weights.data == null or match.dst.data == null) {
            return false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        if (overlap(match.dst, match.experts) or overlap(match.dst, match.weights)) {
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// Patterns

/// Ports `ggml_metal_tensor_is_weight` (ggml-metal-fusion.cpp:651
/// @cd959bae9): true if `t` is or views a tensor in a buffer marked as
/// weights, such as a bias. The model loader marks its buffers before any
/// graph is optimized, and tensors in unmarked or not yet allocated
/// buffers count as non-weights in both phases.
fn tensorIsWeight(t: *const Tensor) bool {
    const base: *const Tensor = if (t.view_src != null) impl.one(Tensor, t.view_src) else t;

    return base.buffer != null and c.ggml_backend_buffer_get_usage(base.buffer) == c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

/// Ports `ggml_metal_mul_mat_add_operand` (ggml-metal-fusion.cpp:657
/// @cd959bae9): the `ADD`'s other operand when exactly one of its sources
/// is `mm`, and that operand is an F32 non-weight into an F32 sum.
fn mulMatAddOperand(mm: *const Tensor, add: *const Tensor) ?*const Tensor {
    if (add.op != c.GGML_OP_ADD or ((add.src[0] == mm) == (add.src[1] == mm))) {
        return null;
    }

    const other = if (add.src[0] == mm) src(add, 1) else src(add, 0);

    const ok = other.type == c.GGML_TYPE_F32 and add.type == c.GGML_TYPE_F32 and !tensorIsWeight(other);

    return if (ok) other else null;
}

/// Ports `ggml_metal_mul_mat_add_residual` (ggml-metal-fusion.cpp:669
/// @cd959bae9): the operand, when it is a same-shape contiguous residual
/// the MMA store can add.
fn mulMatAddResidual(mm: *const Tensor, add: *const Tensor) ?*const Tensor {
    const res = mulMatAddOperand(mm, add);

    const ok = res != null and c.ggml_are_same_shape(res.?, mm) and
        c.ggml_is_contiguous(res.?) and c.ggml_is_contiguous(mm) and c.ggml_is_contiguous(add);

    return if (ok) res else null;
}

/// Ports `ggml_metal_fusion_check_mul_mat_add` (ggml-metal-fusion.cpp:680
/// @cd959bae9).
///
/// `MUL_MAT + ADD` of an F32 non-weight. The reorder packs it without
/// reading row counts (`STRUCTURAL` stops after the operand and type
/// checks), so ubatch sizes share one node order; the encoder fuses only
/// a same-shape residual on the few-row MMA path, which the sum may
/// overlap only in place.
fn checkMulMatAdd(fusion: *const Fusion, nodes: [*]const *const Tensor, gf: *const CGraph, node_idxs: [*]const c_int, idx: c_int, mode: Mode) bool {
    _ = gf;
    _ = node_idxs;
    _ = idx;
    _ = fusion;

    const mm = nodes[0];
    const add = nodes[1];

    if (mulMatAddOperand(mm, add) == null or
        !common.ggml_metal_op_mul_mat_may_use_mma(mm))
    {
        return false;
    }

    if (mode == .structural) {
        return true;
    }

    const res = mulMatAddResidual(mm, add) orelse return false;

    if (!common.ggml_metal_op_mul_mat_use_mma(mm)) {
        return false;
    }

    return !overlap(add, src(mm, 0)) and !overlap(add, src(mm, 1)) and
        (add.data == res.data or !overlap(add, res));
}

// Ports `ops_norm_mul`, `ops_norm_mul_add`, `ops_norm_scale`,
// `ops_rms_norm_mul`, `ops_rms_norm_mul_add`, `ops_rms_norm_scale`
// (ggml-metal-fusion.cpp:716, 717, 718, 719, 720, 721 @cd959bae9),
// `ops_add_2`, `ops_add_3`, `ops_add_4`, `ops_add_5`, `ops_add_6`,
// `ops_add_7` (ggml-metal-fusion.cpp:723, 724, 725, 726, 727, 728
// @cd959bae9), `ops_snake` (ggml-metal-fusion.cpp:729 @cd959bae9),
// `ops_gdn_cache` (ggml-metal-fusion.cpp:731 @cd959bae9) and
// `ops_ssm_conv_silu` (ggml-metal-fusion.cpp:733 @cd959bae9).
const ops_norm_mul = [_]c.enum_ggml_op{ c.GGML_OP_NORM, c.GGML_OP_MUL };
const ops_norm_mul_add = [_]c.enum_ggml_op{ c.GGML_OP_NORM, c.GGML_OP_MUL, c.GGML_OP_ADD };
const ops_norm_scale = [_]c.enum_ggml_op{ c.GGML_OP_NORM, c.GGML_OP_SCALE };
const ops_rms_norm_mul = [_]c.enum_ggml_op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL };
const ops_rms_norm_mul_add = [_]c.enum_ggml_op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ADD };
const ops_rms_norm_scale = [_]c.enum_ggml_op{ c.GGML_OP_RMS_NORM, c.GGML_OP_SCALE };

const ops_add_2: [2]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_3: [3]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_4: [4]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_5: [5]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_6: [6]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_7: [7]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_snake = [_]c.enum_ggml_op{ c.GGML_OP_MUL, c.GGML_OP_SIN, c.GGML_OP_SQR, c.GGML_OP_MUL, c.GGML_OP_ADD };

const ops_gdn_cache = [_]c.enum_ggml_op{ c.GGML_OP_GATED_DELTA_NET, c.GGML_OP_CPY };

const ops_ssm_conv_silu = [_]c.enum_ggml_op{ c.GGML_OP_SSM_CONV, c.GGML_OP_UNARY };

/// Ports `ops_moe_reduce_2`, `ops_moe_reduce_3`, `ops_moe_reduce_4`,
/// `ops_moe_reduce_5`, `ops_moe_reduce_6`, `ops_moe_reduce_7` and
/// `ops_moe_reduce_8` (ggml-metal-fusion.cpp:735, 738, 741, 745, 749, 753,
/// 757 @cd959bae9): MUL, `n` VIEWs, `n - 1` ADDs. The C spells each out; they are one
/// shape, so one comptime function writes them.
fn opsMoeReduce(comptime n: usize) [2 * n]c.enum_ggml_op {
    var ops: [2 * n]c.enum_ggml_op = undefined;
    ops[0] = c.GGML_OP_MUL;
    for (1..n + 1) |i| ops[i] = c.GGML_OP_VIEW;
    for (n + 1..2 * n) |i| ops[i] = c.GGML_OP_ADD;
    return ops;
}
const ops_moe_reduce_2 = opsMoeReduce(2);
const ops_moe_reduce_3 = opsMoeReduce(3);
const ops_moe_reduce_4 = opsMoeReduce(4);
const ops_moe_reduce_5 = opsMoeReduce(5);
const ops_moe_reduce_6 = opsMoeReduce(6);
const ops_moe_reduce_7 = opsMoeReduce(7);
const ops_moe_reduce_8 = opsMoeReduce(8);

/// Ports `ops_mul_mat_add` (ggml-metal-fusion.cpp:763 @cd959bae9).
const ops_mul_mat_add = [_]c.enum_ggml_op{ c.GGML_OP_MUL_MAT, c.GGML_OP_ADD };

const out_1 = [_]c_int{1};

fn entry(id: Id, comptime ops_all: []const c.enum_ggml_op, outs: []const c_int, unsafe: bool, check: CheckFn) Fusion {
    return .{
        .id = id,
        .ops = filterOps(ops_all),
        .ops_all = ops_all,
        .outs = outs,
        .unsafe = unsafe,
        .check = check,
    };
}

/// Ports `ggml_metal_fusions` (ggml-metal-fusion.cpp:765 @cd959bae9): the
/// single table of all fusions the Metal backend supports, in the C's
/// order — `ggml_metal_fusion_next` keeps the *first* longest match, and
/// the stats labels are indexed by position.
const fusions = [_]Fusion{
    entry(.norm_mul, &ops_norm_mul, &.{}, false, &checkNorm),
    entry(.norm_mul_add, &ops_norm_mul_add, &.{}, false, &checkNorm),
    entry(.norm_scale, &ops_norm_scale, &.{}, false, &checkNorm),
    entry(.norm_mul, &ops_rms_norm_mul, &.{}, false, &checkNorm),
    entry(.norm_mul_add, &ops_rms_norm_mul_add, &.{}, false, &checkNorm),
    entry(.norm_scale, &ops_rms_norm_scale, &.{}, false, &checkNorm),
    entry(.add_chain, &ops_add_2, &.{}, false, &checkAddChain),
    entry(.add_chain, &ops_add_3, &.{}, false, &checkAddChain),
    entry(.add_chain, &ops_add_4, &.{}, false, &checkAddChain),
    entry(.add_chain, &ops_add_5, &.{}, false, &checkAddChain),
    entry(.add_chain, &ops_add_6, &.{}, false, &checkAddChain),
    entry(.add_chain, &ops_add_7, &.{}, false, &checkAddChain),
    entry(.snake, &ops_snake, &.{}, false, &checkSnake),
    entry(.gdn_cache, &ops_gdn_cache, &.{}, true, &checkGdnCache),
    entry(.topk_moe, &ops_topk_moe, &out_1, true, &checkTopkMoe),
    entry(.topk_moe, &ops_topk_moe_scale, &out_1, true, &checkTopkMoe),
    entry(.topk_moe, &ops_topk_moe_norm, &out_1, true, &checkTopkMoe),
    entry(.topk_moe, &ops_topk_moe_norm_scale, &out_1, true, &checkTopkMoe),
    entry(.moe_reduce, &ops_moe_reduce_2, &.{}, true, &checkMoeReduce),
    entry(.moe_reduce, &ops_moe_reduce_3, &.{}, true, &checkMoeReduce),
    entry(.moe_reduce, &ops_moe_reduce_4, &.{}, true, &checkMoeReduce),
    entry(.moe_reduce, &ops_moe_reduce_5, &.{}, true, &checkMoeReduce),
    entry(.moe_reduce, &ops_moe_reduce_6, &.{}, true, &checkMoeReduce),
    entry(.moe_reduce, &ops_moe_reduce_7, &.{}, true, &checkMoeReduce),
    entry(.moe_reduce, &ops_moe_reduce_8, &.{}, true, &checkMoeReduce),
    entry(.ssm_conv_silu, &ops_ssm_conv_silu, &.{}, false, &checkSsmConvSilu),
    entry(.mul_mat_add, &ops_mul_mat_add, &.{}, false, &checkMulMatAdd),
};

comptime {
    for (fusions) |f| {
        std.debug.assert(f.ops.len <= max);
        std.debug.assert(f.ops_all.len <= max);
    }
}

// -----------------------------------------------------------------------------
// Alloc deps

/// Ports `ggml_metal_fusion_match_raw_pattern` (ggml-metal-fusion.cpp:797
/// @cd959bae9): whether the raw nodes from `node_idx` have exactly `ops`.
fn matchRawPattern(gf: *const CGraph, node_idx: c_int, ops: []const c.enum_ggml_op) bool {
    if (node_idx < 0 or node_idx + @as(c_int, @intCast(ops.len)) > gf.n_nodes) {
        return false;
    }

    for (ops, 0..) |op, i| {
        if (gnode(gf, node_idx + @as(c_int, @intCast(i))).op != op) {
            return false;
        }
    }

    return true;
}

/// The C's `void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *)`.
pub const AddAllocDepFn = *const fn (?*anyopaque, ?*Tensor, ?*Tensor) callconv(.c) void;

/// Ports `ggml_metal_fusion_add_pattern_alloc_deps`
/// (ggml-metal-fusion.cpp:812 @cd959bae9): keep every external input of
/// the run alive until its last node.
///
/// The C's `std::set` is a membership test and nothing reads its order,
/// so a linear scan over a bounded array is the same set: a run is at most
/// `max` nodes of `GGML_MAX_SRC` sources each. A source is reported the
/// first time it is seen, and a node of the run joins the set after its
/// own sources -- the C's insertion order, which decides the calls.
fn addPatternAllocDeps(user_data: ?*anyopaque, add_alloc_dep: AddAllocDepFn, gf: *CGraph, fusion: *const Fusion, node_idx: c_int) void {
    const last_node = node_idx + @as(c_int, @intCast(fusion.ops_all.len)) - 1;

    // keep all external inputs alive until the fused output
    var seen: [max * (c.GGML_MAX_SRC + 1)]*const Tensor = undefined;
    var n_seen: usize = 0;
    for (0..fusion.ops_all.len) |j| {
        const node = gf.nodes[@intCast(node_idx + @as(c_int, @intCast(j)))].?;
        for (0..c.GGML_MAX_SRC) |s| {
            const sp = node.src[s];
            if (sp == null) continue;
            const t = impl.one(Tensor, sp);
            if (std.mem.indexOfScalar(*const Tensor, seen[0..n_seen], t) == null) {
                seen[n_seen] = t;
                n_seen += 1;
                add_alloc_dep(user_data, t, gf.nodes[@intCast(last_node)]);
            }
        }
        if (std.mem.indexOfScalar(*const Tensor, seen[0..n_seen], node) == null) {
            seen[n_seen] = node;
            n_seen += 1;
        }
    }
}

/// Ports `ggml_metal_fusion_add_alloc_deps` (ggml-metal-fusion.cpp:834
/// @cd959bae9).
///
/// Called from `graph_optimize`: at each raw position, the table entry
/// whose raw op sequence (`ops_all`) is the longest exact match there --
/// the first of equal length -- has its external inputs kept alive until
/// its last node, and the scan resumes past it. Only the op sequence is
/// matched, not the checks, so a pattern that would be refused still pins
/// its inputs: harmless, and the C's choice.
pub export fn ggml_metal_fusion_add_alloc_deps(
    user_data: ?*anyopaque,
    add_alloc_dep: AddAllocDepFn,
    gf: *CGraph,
) callconv(.c) void {
    var i: c_int = 0;
    while (i < gf.n_nodes) : (i += 1) {
        var best: ?*const Fusion = null;
        var best_raw: c_int = 0;

        for (&fusions) |*fusion| {
            if (@as(c_int, @intCast(fusion.ops_all.len)) <= best_raw) {
                continue;
            }
            if (matchRawPattern(gf, i, fusion.ops_all)) {
                best = fusion;
                best_raw = @intCast(fusion.ops_all.len);
            }
        }

        if (best) |b| {
            addPatternAllocDeps(user_data, add_alloc_dep, gf, b, i);
            i += best_raw - 1;
        }
    }
}

// -----------------------------------------------------------------------------
// Shared fusion info

/// Ports `struct ggml_metal_fusion_info` (ggml-metal-fusion.cpp:874
/// @cd959bae9).
///
/// The device owns one and every backend context on that device shares it,
/// so the counters accumulate across contexts. Opaque to the C callers.
/// The C's two `std::vector`s are slices allocated once by `labelsInit`,
/// which is also the only place the C fills them.
pub const Info = struct {
    /// NUL-terminated, one per table entry once `labels_set`
    labels: [][:0]u8 = &.{},
    counts: []u64 = &.{},
    enabled: bool,
    stats: bool,
    labels_set: bool,
    debug: c_int,
};

/// Ports `ggml_metal_fusion_label` (ggml-metal-fusion.cpp:861 @cd959bae9):
/// the pattern's non-empty op names joined with `+`.
fn label(fusion: *const Fusion) [:0]u8 {
    var buf: std.ArrayList(u8) = .empty;
    var j: usize = 0;
    while (j < fusion.ops.len) : (j += 1) {
        if (j > 0) {
            buf.append(allocator, '+') catch impl.abort("metal: out of memory building a fusion label");
        }
        buf.appendSlice(allocator, std.mem.span(types.ggml_op_name(fusion.ops[j]))) catch
            impl.abort("metal: out of memory building a fusion label");
    }
    return buf.toOwnedSliceSentinel(allocator, 0) catch impl.abort("metal: out of memory building a fusion label");
}

/// Ports `ggml_metal_fusion_info_init` (ggml-metal-fusion.cpp:883
/// @cd959bae9).
///
/// Parameters:
/// - `enabled`: whether the encoders try to fuse at all.
/// - `debug`: `GGML_METAL_FUSION_DEBUG`; above 0 the counters are kept.
///
/// Return: a new info, owned by the caller and freed with
/// `ggml_metal_fusion_info_free`.
pub export fn ggml_metal_fusion_info_init(enabled: bool, debug: c_int) callconv(.c) *Info {
    const finfo = allocator.create(Info) catch impl.abort("metal: out of memory allocating the fusion info");
    finfo.* = .{
        .enabled = enabled,
        .stats = debug > 0,
        .labels_set = false,
        .debug = debug,
    };

    if (finfo.stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

/// Ports `ggml_metal_fusion_info_free` (ggml-metal-fusion.cpp:897
/// @cd959bae9).
pub export fn ggml_metal_fusion_info_free(finfo: ?*Info) callconv(.c) void {
    const self = finfo orelse return; // `delete nullptr` is a no-op
    freeLabels(self);
    allocator.destroy(self);
}

fn freeLabels(finfo: *Info) void {
    for (finfo.labels) |l| allocator.free(l);
    if (finfo.labels.len > 0) allocator.free(finfo.labels);
    if (finfo.counts.len > 0) allocator.free(finfo.counts);
    finfo.labels = &.{};
    finfo.counts = &.{};
}

/// Ports `ggml_metal_fusion_info_enabled` (ggml-metal-fusion.cpp:901
/// @cd959bae9).
pub export fn ggml_metal_fusion_info_enabled(finfo: *const Info) callconv(.c) bool {
    return finfo.enabled;
}

/// Ports `ggml_metal_fusion_info_stats` (ggml-metal-fusion.cpp:905
/// @cd959bae9).
pub export fn ggml_metal_fusion_info_stats(finfo: *const Info) callconv(.c) bool {
    return finfo.stats;
}

/// Ports `ggml_metal_fusion_info_debug` (ggml-metal-fusion.cpp:909
/// @cd959bae9).
pub export fn ggml_metal_fusion_info_debug(finfo: *const Info) callconv(.c) c_int {
    return finfo.debug;
}

/// Ports `ggml_metal_fusion_info_n_fusions` (ggml-metal-fusion.cpp:913
/// @cd959bae9). Zero until the labels are built.
pub export fn ggml_metal_fusion_info_n_fusions(finfo: *const Info) callconv(.c) c_int {
    return @intCast(finfo.labels.len);
}

/// Ports `ggml_metal_fusion_info_label` (ggml-metal-fusion.cpp:917
/// @cd959bae9).
///
/// Return: the label of table entry `idx`, borrowed from `finfo`.
pub export fn ggml_metal_fusion_info_label(finfo: *const Info, idx: c_int) callconv(.c) [*:0]const u8 {
    impl.assert(idx >= 0 and idx < @as(c_int, @intCast(finfo.labels.len)), "idx >= 0 && idx < (int) finfo->labels.size()");
    return finfo.labels[@intCast(idx)].ptr;
}

/// Ports `ggml_metal_fusion_info_count` (ggml-metal-fusion.cpp:922
/// @cd959bae9).
pub export fn ggml_metal_fusion_info_count(finfo: *const Info, idx: c_int) callconv(.c) u64 {
    impl.assert(idx >= 0 and idx < @as(c_int, @intCast(finfo.counts.len)), "idx >= 0 && idx < (int) finfo->counts.size()");
    return finfo.counts[@intCast(idx)];
}

/// Ports `ggml_metal_fusion_info_count_fusion` (ggml-metal-fusion.cpp:927
/// @cd959bae9): one more firing of `fusion`, its table index being its
/// offset from the table's start. Single-threaded by construction: the
/// context encodes on one thread whenever stats are on.
///
/// The C subtracts the pointers whatever `fusion` is; a pointer from
/// outside the table lands out of range and is ignored, and so here.
pub export fn ggml_metal_fusion_info_count_fusion(finfo: *Info, fusion: ?*const Fusion) callconv(.c) void {
    if (!finfo.stats or fusion == null) {
        return;
    }

    const base = @intFromPtr(&fusions[0]);
    const addr = @intFromPtr(fusion.?);
    if (addr < base or (addr - base) % @sizeOf(Fusion) != 0) {
        return;
    }
    const idx = (addr - base) / @sizeOf(Fusion);
    if (idx < finfo.counts.len) {
        finfo.counts[idx] += 1;
    }
}

/// Ports `ggml_metal_fusion_info_set_enabled` (ggml-metal-fusion.cpp:938
/// @cd959bae9).
pub export fn ggml_metal_fusion_info_set_enabled(finfo: *Info, enabled: bool) callconv(.c) void {
    finfo.enabled = enabled;
}

/// Ports `ggml_metal_fusion_info_labels_init` (ggml-metal-fusion.cpp:942
/// @cd959bae9): one label and one zeroed counter per table entry, once.
pub export fn ggml_metal_fusion_info_labels_init(finfo: *Info) callconv(.c) void {
    if (finfo.labels_set) {
        return;
    }

    const all = &fusions;
    const len: usize = all.len;

    freeLabels(finfo);
    finfo.counts = allocator.alloc(u64, len) catch impl.abort("metal: out of memory allocating the fusion counters");
    @memset(finfo.counts, 0);
    finfo.labels = allocator.alloc([:0]u8, len) catch impl.abort("metal: out of memory allocating the fusion labels");

    for (0..len) |i| {
        finfo.labels[i] = label(&all[i]);
    }

    finfo.labels_set = true;
}

/// Ports `ggml_metal_fusion_info_stats_init` (ggml-metal-fusion.cpp:958
/// @cd959bae9).
pub export fn ggml_metal_fusion_info_stats_init(finfo: *Info) callconv(.c) void {
    finfo.stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

/// Ports `ggml_metal_fusion_info_stats_reset` (ggml-metal-fusion.cpp:963
/// @cd959bae9).
pub export fn ggml_metal_fusion_info_stats_reset(finfo: *Info) callconv(.c) void {
    @memset(finfo.counts, 0);
}

/// Ports `ggml_metal_fusion_info_stats_get` (ggml-metal-fusion.cpp:967
/// @cd959bae9).
///
/// Parameters:
/// - `labels`: receives up to `n` labels, borrowed from `finfo`; null asks
///   only for the count.
/// - `counts`: receives the matching counters, or null.
/// - `n`: capacity of both arrays.
///
/// Return: the number of fusions when `labels` is null, else how many
/// entries were filled.
pub export fn ggml_metal_fusion_info_stats_get(finfo: *const Info, labels: ?[*][*:0]const u8, counts: ?[*]u64, n: c_int) callconv(.c) c_int {
    const n_fusions: c_int = @intCast(finfo.labels.len);

    const out = labels orelse return n_fusions;

    const n_fill = @min(n, n_fusions);
    var i: c_int = 0;
    while (i < n_fill) : (i += 1) {
        const k: usize = @intCast(i);
        out[k] = finfo.labels[k].ptr;
        if (counts) |cs| {
            cs[k] = finfo.counts[k];
        }
    }

    return n_fill;
}

// -----------------------------------------------------------------------------
// Memory-range checks

/// The `same_memory` lambda of `ggml_metal_fusion_check_memory_ranges`
/// (ggml-metal-fusion.cpp:990 @cd959bae9): one tensor's data is the
/// other's, or either is a view (at any depth) of the other.
fn sameMemory(a: *const Tensor, b: *const Tensor) bool {
    if (a.data != null and b.data != null and a.data == b.data) {
        return true;
    }
    var v: ?*const Tensor = a;
    while (v) |t| : (v = if (t.view_src == null) null else impl.one(Tensor, t.view_src)) {
        if (t == b) return true;
    }
    v = b;
    while (v) |t| : (v = if (t.view_src == null) null else impl.one(Tensor, t.view_src)) {
        if (t == a) return true;
    }
    return false;
}

/// The `nodes_overlap` lambda of the same function: both allocated, in the
/// same buffer, with intersecting `[data, data + alloc_size)` intervals.
fn nodesOverlap(a: *const Tensor, b: *const Tensor) bool {
    if (a.data == null or b.data == null or a.buffer == null or b.buffer == null) {
        return false;
    }

    if (a.buffer != b.buffer) {
        return false;
    }

    const a_start: i64 = @bitCast(@as(u64, @intFromPtr(a.data)));
    const a_end: i64 = a_start +% @as(i64, @bitCast(@as(u64, backend.ggml_backend_buft_get_alloc_size(impl.one(c.ggml_backend_buffer, a.buffer).buft, a))));
    const b_start: i64 = @bitCast(@as(u64, @intFromPtr(b.data)));
    const b_end: i64 = b_start +% @as(i64, @bitCast(@as(u64, backend.ggml_backend_buft_get_alloc_size(impl.one(c.ggml_backend_buffer, b.buffer).buft, b))));

    return (b_start <= a_start and a_start < b_end) or
        (a_start <= b_start and b_start < a_end);
}

/// The `is_intermediate` lambda: `s` is, or is a view of, one of the
/// pattern's nodes before `j`.
fn isIntermediate(s: *const Tensor, nodes: [*]const *const Tensor, j: usize) bool {
    for (0..j) |k| {
        if (s == nodes[k]) {
            return true;
        }
        var view_src: ?*const Tensor = if (s.view_src == null) null else impl.one(Tensor, s.view_src);
        while (view_src) |v| : (view_src = if (v.view_src == null) null else impl.one(Tensor, v.view_src)) {
            if (v == nodes[k]) {
                return true;
            }
        }
    }
    return false;
}

/// The `check_dst` lambda: no external source of any node overlaps `dst`.
fn checkDst(nodes: [*]const *const Tensor, node_count: usize, dst: *const Tensor) bool {
    for (0..node_count) |j| {
        for (0..c.GGML_MAX_SRC) |s| {
            const sp = nodes[j].src[s];
            if (sp == null) continue;
            const t = impl.one(Tensor, sp);
            if (t.op == c.GGML_OP_NONE or sameMemory(t, dst)) {
                continue;
            }

            if (nodesOverlap(dst, t) and !isIntermediate(t, nodes, j)) {
                return false;
            }
        }
    }
    return true;
}

/// Ports `ggml_metal_fusion_check_memory_ranges` (ggml-metal-fusion.cpp:990
/// @cd959bae9).
///
/// Reject fusions where an external source overlaps any fused output. The
/// fused kernels elide intermediate nodes, so only sources that are not
/// part of the fused subgraph can race with the output. Some fused kernels
/// write through a tensor that also appears as a source (the gdn cache
/// cpy), so a source that *is* the output's memory is not an external
/// read. Only the compute phase has allocated tensors, so only FULL mode
/// asks.
fn checkMemoryRanges(fusion: *const Fusion, nodes: [*]const *const Tensor, node_count: usize) bool {
    if (!checkDst(nodes, node_count, nodes[node_count - 1])) {
        return false;
    }

    for (fusion.outs) |offset| {
        impl.assert(offset >= 0 and offset < @as(c_int, @intCast(node_count)), "offset >= 0 && offset < node_count");
        if (!checkDst(nodes, node_count, nodes[@intCast(offset)])) {
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// Queries

/// Ports `ggml_metal_fusion_next` (ggml-metal-fusion.cpp:1080 @cd959bae9).
///
/// Parameters:
/// - `gf`: the graph.
/// - `node_idxs`: `n_idxs` graph node indices — the non-empty nodes.
/// - `idx`: a position in `node_idxs`, where the pattern must start.
/// - `mode`: structural (optimizer) or full (encoder) checks.
/// - `n_out`: receives the number of nodes consumed, 1 when nothing fuses.
///
/// Return: the longest matching table entry, the first of equal length
/// wins; null when nothing fuses.
pub export fn ggml_metal_fusion_next(
    gf: *const CGraph,
    node_idxs: [*]const c_int,
    n_idxs: c_int,
    idx: c_int,
    mode: Mode,
    n_out: *c_int,
) callconv(.c) ?*const Fusion {
    var res: ?*const Fusion = null;
    var best: c_int = 1;

    for (&fusions) |*fusion| {
        const n_ops: c_int = @intCast(fusion.ops.len);

        // only look for a longer match than the current best
        if (n_ops <= best) {
            continue;
        }
        if (idx + n_ops > n_idxs) {
            continue;
        }

        var nodes: [max]*const Tensor = undefined;
        const nu: usize = fusion.ops.len;
        const base: usize = @intCast(idx);

        // the op sequence must match exactly
        var ok = true;
        for (0..nu) |j| {
            nodes[j] = gf.nodes[@intCast(node_idxs[base + j])].?;
            if (nodes[j].op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            var j: usize = 1;
            while (j < nu and ok) : (j += 1) {
                if (nodes[j].src[0] != nodes[j - 1] and nodes[j].src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!c.ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            var outputs_buf: [max]c_int = undefined;
            outputs_buf[0] = node_idxs[base + nu - 1];
            for (fusion.outs, 0..) |out_offset, i| {
                impl.assert(out_offset >= 0 and out_offset < n_ops, "out_offset >= 0 && out_offset < n_ops");
                outputs_buf[i + 1] = node_idxs[base + @as(usize, @intCast(out_offset))];
            }

            const n_outputs: c_int = 1 + @as(c_int, @intCast(fusion.outs.len));

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!graph.ggml_can_fuse_subgraph_ext(gf, node_idxs + base, n_ops, fusion.ops.ptr, &outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check) |check| {
            if (!check(fusion, &nodes, gf, node_idxs, idx, mode)) continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == .full and !checkMemoryRanges(fusion, &nodes, nu)) {
            continue;
        }

        best = n_ops;
        res = fusion;
    }

    n_out.* = best;

    return res;
}

/// Ports `ggml_metal_fusion_max` (ggml-metal-fusion.cpp:1175 @cd959bae9).
///
/// The optimize phase's question: how many nodes from `idx` (a raw graph
/// index) could be fused, chaining patterns back to back. Matching runs on
/// the same view-transparent sequence the encoders see, so the answer is
/// the raw span to the last matched node, views packed along.
///
/// Since v0.6.0 a zero-element node is no longer skipped here (the
/// encoders still skip it): an empty graph packs like its non-empty
/// counterpart, so a no-output ubatch gets the same node order.
///
/// Return: at least 1.
pub export fn ggml_metal_fusion_max(gf: *const CGraph, idx: c_int) callconv(.c) c_int {
    // a view node cannot start a pattern - pack it alone
    const first = gf.nodes[@intCast(idx)].?;
    if (impl.opIsEmpty(first.op)) {
        return 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    var idxs: [max]c_int = undefined;
    var n_idxs: c_int = 0;
    var i = idx;
    while (i < gf.n_nodes and n_idxs < max) : (i += 1) {
        const t = gf.nodes[@intCast(i)].?;
        if (!impl.opIsEmpty(t.op)) {
            idxs[@intCast(n_idxs)] = i;
            n_idxs += 1;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    var total: c_int = 0;
    var i_f: c_int = 0;

    while (i_f < n_idxs and total < max) {
        var len: c_int = 1;
        const fusion = ggml_metal_fusion_next(gf, &idxs, n_idxs, i_f, .structural, &len);
        if (fusion == null or total + len > max) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return @min(max, idxs[@intCast(i_f - 1)] - idx + 1);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the table's labels are its op names joined by +" {
    const finfo = ggml_metal_fusion_info_init(true, 1);
    defer ggml_metal_fusion_info_free(finfo);

    try std.testing.expectEqual(@as(c_int, fusions.len), ggml_metal_fusion_info_n_fusions(finfo));
    try std.testing.expectEqualStrings("NORM+MUL", std.mem.span(ggml_metal_fusion_info_label(finfo, 0)));
    try std.testing.expectEqualStrings("SSM_CONV+UNARY", std.mem.span(ggml_metal_fusion_info_label(finfo, fusions.len - 2)));
    try std.testing.expectEqualStrings("MUL_MAT+ADD", std.mem.span(ggml_metal_fusion_info_label(finfo, fusions.len - 1)));
    // the labels name the non-empty ops: RESHAPE and VIEW are filtered out
    try std.testing.expectEqualStrings("SOFT_MAX+ARGSORT+GET_ROWS", std.mem.span(ggml_metal_fusion_info_label(finfo, 14)));
    try std.testing.expectEqualStrings("MUL+ADD", std.mem.span(ggml_metal_fusion_info_label(finfo, 18)));

    ggml_metal_fusion_info_count_fusion(finfo, &fusions[4]);
    ggml_metal_fusion_info_count_fusion(finfo, &fusions[4]);
    try std.testing.expectEqual(@as(u64, 2), ggml_metal_fusion_info_count(finfo, 4));

    ggml_metal_fusion_info_stats_reset(finfo);
    try std.testing.expectEqual(@as(u64, 0), ggml_metal_fusion_info_count(finfo, 4));
}

test "a pointer outside the table counts nothing" {
    const finfo = ggml_metal_fusion_info_init(true, 1);
    defer ggml_metal_fusion_info_free(finfo);

    const stray: Fusion = fusions[0];
    ggml_metal_fusion_info_count_fusion(finfo, &stray);
    for (0..fusions.len) |i| {
        try std.testing.expectEqual(@as(u64, 0), ggml_metal_fusion_info_count(finfo, @intCast(i)));
    }
}

test "every entry's ops are its ops_all without the empty ones" {
    for (fusions) |f| {
        var k: usize = 0;
        for (f.ops_all) |op| {
            if (impl.opIsEmpty(op)) continue;
            try std.testing.expectEqual(op, f.ops[k]);
            k += 1;
        }
        try std.testing.expectEqual(f.ops.len, k);
    }
}

test "without debug the info keeps no labels until stats are asked for" {
    const finfo = ggml_metal_fusion_info_init(false, 0);
    defer ggml_metal_fusion_info_free(finfo);

    try std.testing.expect(!ggml_metal_fusion_info_enabled(finfo));
    try std.testing.expect(!ggml_metal_fusion_info_stats(finfo));
    try std.testing.expectEqual(@as(c_int, 0), ggml_metal_fusion_info_stats_get(finfo, null, null, 0));

    ggml_metal_fusion_info_stats_init(finfo);
    try std.testing.expect(ggml_metal_fusion_info_stats(finfo));
    try std.testing.expectEqual(@as(c_int, fusions.len), ggml_metal_fusion_info_stats_get(finfo, null, null, 0));
}
