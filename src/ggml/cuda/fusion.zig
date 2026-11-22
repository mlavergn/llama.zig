//! The CUDA backend's fusion pass: recognising runs of graph nodes that one
//! fused kernel can compute, and launching it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v1.0.0
//! (`b1b770d90`): `ggml_cuda_should_fuse_rope_set_rows`,
//! `ggml_cuda_should_fuse_rms_norm_mul_rope`, `ggml_cuda_try_gdn_cache_fusion`,
//! `ggml_cuda_try_ssm_scan_cache_fusion`, `ggml_cuda_topk_moe_group_fusion`,
//! `ggml_cuda_topk_moe_fusion`, `ggml_cuda_check_fusion_memory_ranges`,
//! `ggml_cuda_match_moe_weighted_reduction`, `ggml_cuda_can_fuse` and
//! `ggml_cuda_try_fuse`; and the C++ overloads of
//! `ggml_can_fuse`, `ggml_can_fuse_subgraph` and `ggml_check_edges` from
//! `llama.cpp/ggml/src/ggml-impl.h`.
//!
//! The C's `std::initializer_list<ggml_op>` arguments are slices here.

const std = @import("std");
const ggml = @import("../module.zig");

fn node(cgraph: *const ggml.impl.CGraph, i: c_int) *ggml.impl.ggml_c.ggml_tensor {
    return cgraph.nodes[@intCast(i)].?;
}

fn src(t: *const ggml.impl.ggml_c.ggml_tensor, i: usize) *ggml.impl.ggml_c.ggml_tensor {
    return ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, t.src[i]);
}

/// `ggml_can_fuse_subgraph` and `ggml_check_edges` live in `graph.zig`,
/// which the Vulkan host layer shares.
/// Ports `ggml_cuda_should_fuse_rope_set_rows` (ggml-cuda.cu:3390 @b1b770d90).
fn shouldFuseRopeSetRows(rope_t: *const ggml.impl.ggml_c.ggml_tensor, view: *const ggml.impl.ggml_c.ggml_tensor, set_rows: *const ggml.impl.ggml_c.ggml_tensor) bool {
    if (rope_t.op != ggml.impl.ggml_c.GGML_OP_ROPE or view.op != ggml.impl.ggml_c.GGML_OP_VIEW or set_rows.op != ggml.impl.ggml_c.GGML_OP_SET_ROWS) return false;
    // ne3 not tested
    if (src(rope_t, 0).ne[3] != 1) return false;
    if (set_rows.type != ggml.impl.ggml_c.GGML_TYPE_F32 and set_rows.type != ggml.impl.ggml_c.GGML_TYPE_F16) return false;
    if (src(set_rows, 1).type != ggml.impl.ggml_c.GGML_TYPE_I64) return false;
    // The view should flatten two dims of rope into one dim
    if (!ggml.impl.ggml_c.ggml_is_contiguous(view) or view.ne[0] != rope_t.ne[0] * rope_t.ne[1]) return false;
    // Only norm/neox/multi shaders have the fusion code
    const mode = rope_t.op_params[2];
    if (mode != ggml.impl.ggml_c.GGML_ROPE_TYPE_NORMAL and mode != ggml.impl.ggml_c.GGML_ROPE_TYPE_NEOX and
        mode != ggml.impl.ggml_c.GGML_ROPE_TYPE_MROPE and mode != ggml.impl.ggml_c.GGML_ROPE_TYPE_IMROPE) return false;
    return true;
}

/// Ports `ggml_cuda_should_fuse_rms_norm_mul_rope` (ggml-cuda.cu:3425 @b1b770d90).
fn shouldFuseRmsNormMulRope(rms_norm: *const ggml.impl.ggml_c.ggml_tensor, mul: *const ggml.impl.ggml_c.ggml_tensor, rope_t: *const ggml.impl.ggml_c.ggml_tensor) bool {
    if (rms_norm.op != ggml.impl.ggml_c.GGML_OP_RMS_NORM or mul.op != ggml.impl.ggml_c.GGML_OP_MUL or rope_t.op != ggml.impl.ggml_c.GGML_OP_ROPE) return false;

    const F32 = ggml.impl.ggml_c.GGML_TYPE_F32;
    if (src(rms_norm, 0).type != F32 or rms_norm.type != F32 or
        src(mul, 0).type != F32 or src(mul, 1).type != F32 or
        mul.type != F32 or rope_t.type != F32) return false;

    if (rope_t.src[0] != mul) return false;

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul.src[1] and !ggml.impl.ggml_c.ggml_are_same_shape(src(mul, 0), rms_norm)) return false;

    if (!ggml.impl.ggml_c.ggml_are_same_shape(rms_norm, mul)) return false;

    //rms_norm kernel assumes contiguous rows
    if (!ggml.impl.ggml_c.ggml_is_contiguous_rows(src(rms_norm, 0)) or
        !ggml.impl.ggml_c.ggml_is_contiguous_rows(src(mul, 0)) or !ggml.impl.ggml_c.ggml_is_contiguous_rows(src(mul, 1))) return false;

    // the fused kernel handles the norm/neox rope modes only
    const mode = rope_t.op_params[2];
    if (mode != ggml.impl.ggml_c.GGML_ROPE_TYPE_NORMAL and mode != ggml.impl.ggml_c.GGML_ROPE_TYPE_NEOX) return false;

    const n_dims = rope_t.op_params[1];
    if (@rem(n_dims, 2) != 0 or @rem(src(rope_t, 0).ne[0], 2) != 0) return false;

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const n_offs = rope_t.op_params[15];
    if (n_offs != 0) return false;

    return true;
}

/// Ports `ggml_cuda_is_view_or_noop` (ggml-cuda.cu:3195 @b1b770d90), for the
/// gated-delta-net match; `graph.zig` has its own copy for the compute loop.
fn isViewOrNoop(t: *const ggml.impl.ggml_c.ggml_tensor) bool {
    return ggml.impl.ggml_c.ggml_is_empty(t) or t.op == ggml.impl.ggml_c.GGML_OP_RESHAPE or t.op == ggml.impl.ggml_c.GGML_OP_TRANSPOSE or
        t.op == ggml.impl.ggml_c.GGML_OP_VIEW or t.op == ggml.impl.ggml_c.GGML_OP_PERMUTE or t.op == ggml.impl.ggml_c.GGML_OP_NONE;
}

/// Ports `ggml_cuda_try_gdn_cache_fusion` (ggml-cuda.cu:3479 @b1b770d90):
/// match gated_delta_net + the strided cpy that scatters its state
/// snapshots into the cache, so the kernel can write them and skip the cpy.
fn tryGdnCacheFusion(cgraph: *const ggml.impl.CGraph, node_idx: c_int, fused_state_cpy: *ggml.cuda.gated_delta_net.FusedCache) c_int {
    const gdn = node(cgraph, node_idx);
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn.op != ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET or gdn.type != ggml.impl.ggml_c.GGML_TYPE_F32 or (gdn.flags & ggml.impl.ggml_c.GGML_TENSOR_FLAG_OUTPUT) != 0) return 0;

    const src_v = src(gdn, 2);
    const S_v = src_v.ne[0];
    const H = src_v.ne[1];
    const n_tokens = src_v.ne[2];
    const n_seqs = src_v.ne[3];
    const D = S_v * S_v * H;
    const K: i64 = ggml.impl.getOpParamsI32(gdn, 0); // snapshot slot count
    const n_written = @min(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const tail_off = ggml.impl.ggml_c.ggml_row_size(ggml.impl.ggml_c.GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    var cpy: ?*const ggml.impl.ggml_c.ggml_tensor = null;
    var skip: c_int = 0;
    var j = node_idx + 1;
    while (j < cgraph.n_nodes and cpy == null) : (j += 1) {
        const n = node(cgraph, j);
        if (isViewOrNoop(n)) continue;
        if (n.op != ggml.impl.ggml_c.GGML_OP_CPY or (n.flags & ggml.impl.ggml_c.GGML_TENSOR_FLAG_OUTPUT) != 0) return 0;
        cpy = n;
        skip = j - node_idx;
    }
    const cp = cpy orelse return 0;

    const s = src(cp, 0); // view of the gdn snapshot tail
    const d = src(cp, 1); // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (s.op != ggml.impl.ggml_c.GGML_OP_VIEW or s.view_src != gdn or s.view_offs != tail_off or !ggml.impl.ggml_c.ggml_is_contiguous(s)) return 0;

    // dst is the [D, n_seqs, n_written] cache view
    const expected_ne = [4]i64{ D, n_seqs, n_written, 1 };
    if (d.op != ggml.impl.ggml_c.GGML_OP_VIEW or d.type != ggml.impl.ggml_c.GGML_TYPE_F32 or d.data == null or
        !std.mem.eql(i64, &expected_ne, &d.ne) or
        d.nb[0] != ggml.impl.ggml_c.ggml_type_size(ggml.impl.ggml_c.GGML_TYPE_F32) or d.nb[1] != ggml.impl.ggml_c.ggml_row_size(ggml.impl.ggml_c.GGML_TYPE_F32, D)) return 0;

    fused_state_cpy.data = @ptrCast(@alignCast(d.data)); // rollback group 0 (newest)
    fused_state_cpy.slot_stride = if (K > 1) @intCast(d.nb[2] / @sizeOf(f32)) else 0;
    return skip;
}

/// Ports `ggml_cuda_try_ssm_scan_cache_fusion` (ggml-cuda.cu:3542
/// @b1b770d90): match ssm_scan + the strided cpy that
/// scatters its state snapshots into the cache, so the kernel writes them
/// and skips the cpy. Only the Mamba-2 kernels (the group scan and SSD)
/// write to the cache, and only for one sequence.
fn trySsmScanCacheFusion(cgraph: *const ggml.impl.CGraph, node_idx: c_int, fused_state_cpy: *ggml.cuda.ssm_scan.FusedCache) c_int {
    const ssm = node(cgraph, node_idx);
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm.op != ggml.impl.ggml_c.GGML_OP_SSM_SCAN or ssm.type != ggml.impl.ggml_c.GGML_TYPE_F32 or (ssm.flags & ggml.impl.ggml_c.GGML_TENSOR_FLAG_OUTPUT) != 0) return 0;

    const K: i64 = ggml.impl.getOpParamsI32(ssm, 0); // snapshot slot count

    const s_t = src(ssm, 0);
    const x = src(ssm, 1);
    const A = src(ssm, 3);

    const d_state = s_t.ne[0];
    const D = d_state * s_t.ne[1] * x.ne[1]; // d_state * head_dim * n_head
    const n_tok = x.ne[2];
    const n_seqs = x.ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A.nb[1] != @sizeOf(f32) or (d_state != 96 and d_state != 128 and d_state != 256)) return 0;

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) return 0;

    const n_written = @min(n_tok, K);
    const tail_off = ggml.impl.ggml_c.ggml_row_size(ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    var cpy: ?*const ggml.impl.ggml_c.ggml_tensor = null;
    var skip: c_int = 0;
    var j = node_idx + 1;
    while (j < cgraph.n_nodes and cpy == null) : (j += 1) {
        const n = node(cgraph, j);
        if (isViewOrNoop(n)) continue;
        if (n.op != ggml.impl.ggml_c.GGML_OP_CPY or (n.flags & ggml.impl.ggml_c.GGML_TENSOR_FLAG_OUTPUT) != 0) return 0;
        cpy = n;
        skip = j - node_idx;
    }
    const cp = cpy orelse return 0;

    const s = src(cp, 0); // view of the scan snapshot tail
    const d = src(cp, 1); // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (s.op != ggml.impl.ggml_c.GGML_OP_VIEW or s.view_src != ssm or s.view_offs != tail_off or !ggml.impl.ggml_c.ggml_is_contiguous(s)) return 0;

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const expected_ne = [4]i64{ D, n_seqs, n_written, 1 };
    if (d.op != ggml.impl.ggml_c.GGML_OP_VIEW or d.type != ggml.impl.ggml_c.GGML_TYPE_F32 or d.data == null or
        !std.mem.eql(i64, &expected_ne, &d.ne) or
        d.nb[0] != ggml.impl.ggml_c.ggml_type_size(ggml.impl.ggml_c.GGML_TYPE_F32) or d.nb[1] != ggml.impl.ggml_c.ggml_row_size(ggml.impl.ggml_c.GGML_TYPE_F32, D)) return 0;

    fused_state_cpy.data = @ptrCast(@alignCast(d.data)); // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = if (K > 1) @intCast(d.nb[2] / @sizeOf(f32)) else 0;
    return skip;
}

/// One row of `topk_moe_group_pattern`: `src0` and `src1` index into the
/// table, -1 is the (biased) probs, -2 is not checked.
const GroupStep = struct { op: ggml.impl.ggml_c.enum_ggml_op, src0: c_int, src1: c_int };

/// Ports `topk_moe_group_pattern` (ggml-cuda.cu:3620 @b1b770d90): expert
/// group selection from `build_moe_ffn` (`n_expert_groups > 1`).
const topk_moe_group_pattern = [_]GroupStep{
    .{ .op = ggml.impl.ggml_c.GGML_OP_RESHAPE, .src0 = -1, .src1 = -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    .{ .op = ggml.impl.ggml_c.GGML_OP_RESHAPE, .src0 = 0, .src1 = -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    .{ .op = ggml.impl.ggml_c.GGML_OP_ARGSORT, .src0 = 0, .src1 = -2 }, //  2
    .{ .op = ggml.impl.ggml_c.GGML_OP_VIEW, .src0 = 2, .src1 = -2 }, //  3: top-2 per group
    .{ .op = ggml.impl.ggml_c.GGML_OP_GET_ROWS, .src0 = 1, .src1 = 3 }, //  4
    .{ .op = ggml.impl.ggml_c.GGML_OP_RESHAPE, .src0 = 4, .src1 = -2 }, //  5
    .{ .op = ggml.impl.ggml_c.GGML_OP_SUM_ROWS, .src0 = 5, .src1 = -2 }, //  6: group scores
    .{ .op = ggml.impl.ggml_c.GGML_OP_RESHAPE, .src0 = 6, .src1 = -2 }, //  7
    .{ .op = ggml.impl.ggml_c.GGML_OP_ARGSORT, .src0 = 7, .src1 = -2 }, //  8
    .{ .op = ggml.impl.ggml_c.GGML_OP_VIEW, .src0 = 8, .src1 = -2 }, //  9: top n_group_used groups
    .{ .op = ggml.impl.ggml_c.GGML_OP_GET_ROWS, .src0 = 0, .src1 = 9 }, // 10
    .{ .op = ggml.impl.ggml_c.GGML_OP_FILL, .src0 = 0, .src1 = -2 }, // 11
    .{ .op = ggml.impl.ggml_c.GGML_OP_SET_ROWS, .src0 = 10, .src1 = 9 }, // 12: unused groups -> -inf
    .{ .op = ggml.impl.ggml_c.GGML_OP_RESHAPE, .src0 = 12, .src1 = -2 }, // 13: [n_expert, n_tokens]
};

/// Ports `ggml_cuda_topk_moe_group_fusion` (ggml-cuda.cu:3637 @b1b770d90):
/// whether `topk_moe_group_pattern` starts at `node_idx` over
/// `selection`. The `HIP_VERSION < 60300000` arm never compiles here.
fn topkMoeGroupFusion(cgraph: *const ggml.impl.CGraph, node_idx: c_int, selection: *const ggml.impl.ggml_c.ggml_tensor, args: *ggml.cuda.topk_moe.Args) bool {
    const n_ops: c_int = topk_moe_group_pattern.len;
    if (node_idx + n_ops > cgraph.n_nodes) return false;

    const S = struct {
        fn srcOk(g: *const ggml.impl.CGraph, base: c_int, sel: *const ggml.impl.ggml_c.ggml_tensor, s: ?*const ggml.impl.ggml_c.ggml_tensor, idx: c_int) bool {
            if (idx == -2) return true;
            const want: *const ggml.impl.ggml_c.ggml_tensor = if (idx == -1) sel else node(g, base + idx);
            return s == want;
        }
    };

    for (topk_moe_group_pattern, 0..) |p, i| {
        const n = node(cgraph, node_idx + @as(c_int, @intCast(i)));
        if (n.op != p.op or !S.srcOk(cgraph, node_idx, selection, n.src[0], p.src0) or !S.srcOk(cgraph, node_idx, selection, n.src[1], p.src1)) {
            return false;
        }
    }

    const n_groups: c_int = @truncate(node(cgraph, node_idx).ne[1]);
    const n_group_used: c_int = @truncate(node(cgraph, node_idx + 9).ne[0]);

    if (node(cgraph, node_idx + 3).ne[0] != 2 or node(cgraph, node_idx + 12).src[2] != node(cgraph, node_idx + 11) or
        ggml.impl.getOpParamsF32(node(cgraph, node_idx + 11), 0) != -std.math.inf(f32) or n_groups > ggml.cuda.common.warp_size)
    {
        return false;
    }

    args.n_groups = n_groups;
    args.n_group_used = n_group_used;
    return true;
}

/// Ports `ggml_cuda_topk_moe_fusion` (ggml-cuda.cu:3673 @b1b770d90).
pub fn topkMoeFusion(cgraph: *const ggml.impl.CGraph, node_idx_in: c_int, args: *ggml.cuda.topk_moe.Args) bool {
    args.sigmoid = false;
    args.sqrt_softplus = false;
    args.softmax = false;
    args.delayed_softmax = false;
    args.prob_bias = false;
    args.norm = false;
    args.n_groups = 0;
    args.n_group_used = 0;

    const n_nodes = cgraph.n_nodes;
    var node_idx = node_idx_in;

    if (node(cgraph, node_idx).op == ggml.impl.ggml_c.GGML_OP_SOFT_MAX) args.softmax = true;

    if (node(cgraph, node_idx).op == ggml.impl.ggml_c.GGML_OP_UNARY) {
        const unary_op = ggml.impl.ggml_c.ggml_get_unary_op(node(cgraph, node_idx));
        if (unary_op == ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS and node_idx + 1 < n_nodes and
            node(cgraph, node_idx + 1).op == ggml.impl.ggml_c.GGML_OP_SQRT and node(cgraph, node_idx + 1).src[0] == node(cgraph, node_idx))
        {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx += 1;
        } else {
            return false;
        }
    }

    if (node(cgraph, node_idx).op == ggml.impl.ggml_c.GGML_OP_ARGSORT) args.delayed_softmax = true;

    node_idx += 1;

    if (args.sigmoid or args.sqrt_softplus or args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes or node(cgraph, node_idx).op != ggml.impl.ggml_c.GGML_OP_RESHAPE or
            node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) return false;
        const probs_reshaped = node(cgraph, node_idx);
        node_idx += 1;

        if (node_idx >= n_nodes) return false;

        // src of bias add is the unreshaped probs (-2 instead of -1)
        var selection: *const ggml.impl.ggml_c.ggml_tensor = node(cgraph, node_idx - 2);
        if (node(cgraph, node_idx).op == ggml.impl.ggml_c.GGML_OP_ADD and node(cgraph, node_idx).src[0] == selection) {
            args.prob_bias = true;
            selection = node(cgraph, node_idx);
            node_idx += 1;
        }

        if (topkMoeGroupFusion(cgraph, node_idx, selection, args)) {
            node_idx += topk_moe_group_pattern.len;
            selection = node(cgraph, node_idx - 1);
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes or node(cgraph, node_idx).op != ggml.impl.ggml_c.GGML_OP_ARGSORT or node(cgraph, node_idx).src[0] != selection) return false;

        node_idx += 1;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes or node(cgraph, node_idx).op != ggml.impl.ggml_c.GGML_OP_VIEW or
            node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) return false;

        // kept groups must hold all used experts
        if (args.n_groups > 0 and node(cgraph, node_idx).ne[0] > args.n_group_used * @divTrunc(probs_reshaped.ne[1], args.n_groups)) return false;
        node_idx += 1;

        if (node_idx >= n_nodes or node(cgraph, node_idx).op != ggml.impl.ggml_c.GGML_OP_GET_ROWS) return false;

        // GET_ROWS
        if (node(cgraph, node_idx).src[0] != probs_reshaped or node(cgraph, node_idx).src[1] != node(cgraph, node_idx - 1)) return false;
        node_idx += 1;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) return false;
        const probs_reshaped = node(cgraph, node_idx - 2);

        // VIEW->ARGSORT
        if (node_idx >= n_nodes or node(cgraph, node_idx).op != ggml.impl.ggml_c.GGML_OP_VIEW or
            node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) return false;
        node_idx += 1;

        // GET_ROWS
        if (node_idx >= n_nodes or node(cgraph, node_idx).src[1] != node(cgraph, node_idx - 1) or
            node(cgraph, node_idx).src[0] != probs_reshaped) return false;
        node_idx += 1;

        const remaining_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_SOFT_MAX, ggml.impl.ggml_c.GGML_OP_RESHAPE };
        for (remaining_ops) |op| {
            if (node_idx >= n_nodes or node(cgraph, node_idx).op != op or node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) return false;
            node_idx += 1;
        }
    }

    // At this point we can check for norm + scale. Everything is now at
    // least valid till the norm
    if (node_idx >= n_nodes) return true;

    if (node(cgraph, node_idx).op == ggml.impl.ggml_c.GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        const norm_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_CLAMP };

        args.norm = true;
        for (norm_ops) |op| {
            if (node(cgraph, node_idx).op == op and node(cgraph, node_idx).src[0] == node(cgraph, node_idx - 1)) {
                node_idx += 1;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (node(cgraph, node_idx).op != ggml.impl.ggml_c.GGML_OP_DIV or node(cgraph, node_idx).src[1] != node(cgraph, node_idx - 1) or
            node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 3))
        {
            args.norm = false;
            return true;
        }
        node_idx += 1;

        if (node(cgraph, node_idx).op != ggml.impl.ggml_c.GGML_OP_RESHAPE or node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) {
            args.norm = false;
            return true;
        }

        node_idx += 1;
    }

    if (node(cgraph, node_idx).op == ggml.impl.ggml_c.GGML_OP_SCALE and node(cgraph, node_idx).src[0] == node(cgraph, node_idx - 1)) {
        args.scale = true;
    }

    return true;
}

/// Ports `ggml_cuda_check_fusion_memory_ranges` (ggml-cuda.cu:3837 @b1b770d90):
/// whether the write (out) nodes overwrite the read nodes in operation.
fn checkFusionMemoryRanges(cgraph: *const ggml.impl.CGraph, node_idx: c_int, node_count: c_int, out_nodes: []const c_int, is_topk_moe: bool) bool {
    const S = struct {
        fn overlap(a: *const ggml.impl.ggml_c.ggml_tensor, b: *const ggml.impl.ggml_c.ggml_tensor) bool {
            const a_start: i64 = @intCast(@intFromPtr(a.data));
            const a_end = a_start + @as(i64, @intCast(ggml.impl.ggml_c.ggml_backend_buft_get_alloc_size(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, a.buffer).buft, a)));
            const b_start: i64 = @intCast(@intFromPtr(b.data));
            const b_end = b_start + @as(i64, @intCast(ggml.impl.ggml_c.ggml_backend_buft_get_alloc_size(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, b.buffer).buft, b)));
            return (b_start <= a_start and a_start < b_end) or (a_start <= b_start and b_start < a_end);
        }
    };

    var is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    var logits_may_alias: ?*const ggml.impl.ggml_c.ggml_tensor = null;
    if (is_topk_moe and ggml.impl.ggml_c.ggml_nrows(node(cgraph, node_idx)) <= ggml.cuda.topk_moe.rows_per_block) {
        logits_may_alias = node(cgraph, node_idx).src[0];
    }

    for (out_nodes) |out| {
        const dst = node(cgraph, out);

        var j = node_idx;
        while (j < node_idx + node_count) : (j += 1) {
            // Loop over all srcs of all nodes in the fusion. If the src
            // overlaps the destination and the src is not an intermediate
            // node that's being elided, then disable fusion.
            for (node(cgraph, j).src) |s_opt| {
                const s = s_opt orelse continue;
                if (s.*.op == ggml.impl.ggml_c.GGML_OP_NONE or s == logits_may_alias) continue;

                if (S.overlap(dst, s)) {
                    var found = false;
                    var k = node_idx;
                    while (k < j) : (k += 1) {
                        if (node(cgraph, k) == s) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

/// Ports `MOE_WEIGHTED_REDUCTION_MAX_EXPERTS` (ggml-cuda.cu:3903 @b1b770d90).
/// The long form spans 2*k + 1 nodes. `ggml_can_fuse_subgraph` accepts at
/// most 31 nodes, so k <= 15; larger values use the per-operation path.
const moe_weighted_reduction_max_experts = 15;

/// Ports `ggml_cuda_moe_weighted_reduction_match` (ggml-cuda.cu:3905 @b1b770d90).
pub const MoeWeightedReductionMatch = struct {
    experts: ?*const ggml.impl.ggml_c.ggml_tensor = null,
    expert_scale: ?*const ggml.impl.ggml_c.ggml_tensor = null,
    weights: ?*const ggml.impl.ggml_c.ggml_tensor = null,
    dst: ?*ggml.impl.ggml_c.ggml_tensor = null,
    node_count: c_int = 0,
};

/// Ports `ggml_cuda_match_moe_weighted_reduction` (ggml-cuda.cu:3913 @b1b770d90),
/// with its `split_mul`, `is_weights` and `is_experts` lambdas. The C's
/// `std::vector`s are bounded by `moe_weighted_reduction_max_experts`, so
/// they are fixed buffers here.
pub fn matchMoeWeightedReduction(cgraph: *const ggml.impl.CGraph, node_idx: c_int, match: *MoeWeightedReductionMatch) bool {
    const first = node(cgraph, node_idx);
    if (first.op != ggml.impl.ggml_c.GGML_OP_MUL or first.type != ggml.impl.ggml_c.GGML_TYPE_F32 or !ggml.impl.ggml_c.ggml_is_contiguous(first)) return false;

    const S = struct {
        fn isWeights(mul: *const ggml.impl.ggml_c.ggml_tensor, tensor_opt: ?*const ggml.impl.ggml_c.ggml_tensor) bool {
            const tensor = tensor_opt orelse return false;
            return tensor.type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.impl.ggml_c.ggml_is_contiguous(tensor) and tensor.ne[0] == 1 and
                tensor.ne[1] == mul.ne[1] and tensor.ne[2] == mul.ne[2] and tensor.ne[3] == mul.ne[3];
        }

        fn isExperts(mul: *const ggml.impl.ggml_c.ggml_tensor, tensor_opt: ?*const ggml.impl.ggml_c.ggml_tensor) bool {
            const tensor = tensor_opt orelse return false;
            return tensor.type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.impl.ggml_c.ggml_is_contiguous(tensor) and
                ggml.impl.ggml_c.ggml_are_same_shape(tensor, mul);
        }

        fn splitMul(mul: *const ggml.impl.ggml_c.ggml_tensor, full: *?*const ggml.impl.ggml_c.ggml_tensor, broadcast: *?*const ggml.impl.ggml_c.ggml_tensor) bool {
            if (isExperts(mul, mul.src[0]) and isWeights(mul, mul.src[1])) {
                full.* = mul.src[0];
                broadcast.* = mul.src[1];
                return true;
            }
            if (isExperts(mul, mul.src[1]) and isWeights(mul, mul.src[0])) {
                full.* = mul.src[1];
                broadcast.* = mul.src[0];
                return true;
            }
            return false;
        }
    };

    var weighted: *const ggml.impl.ggml_c.ggml_tensor = first;
    var experts: ?*const ggml.impl.ggml_c.ggml_tensor = null;
    var expert_scale: ?*const ggml.impl.ggml_c.ggml_tensor = null;
    var weights: ?*const ggml.impl.ggml_c.ggml_tensor = null;
    var mul_count: c_int = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph.n_nodes) {
        const second = node(cgraph, node_idx + 1);
        var scaled: ?*const ggml.impl.ggml_c.ggml_tensor = null;
        var route: ?*const ggml.impl.ggml_c.ggml_tensor = null;
        var raw: ?*const ggml.impl.ggml_c.ggml_tensor = null;
        var scale: ?*const ggml.impl.ggml_c.ggml_tensor = null;
        if (second.op == ggml.impl.ggml_c.GGML_OP_MUL and second.type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.impl.ggml_c.ggml_is_contiguous(second) and
            S.splitMul(second, &scaled, &route) and scaled == @as(?*const ggml.impl.ggml_c.ggml_tensor, first) and S.splitMul(first, &raw, &scale))
        {
            weighted = second;
            experts = raw;
            expert_scale = scale;
            weights = route;
            mul_count = 2;
        }
    }

    if (experts == null and !S.splitMul(first, &experts, &weights)) return false;

    const n_expert_used: c_int = @truncate(weighted.ne[1]);
    const n_tokens: i64 = weighted.ne[2] * weighted.ne[3];
    if (n_expert_used < 2 or n_expert_used > moe_weighted_reduction_max_experts) return false;

    const node_count: c_int = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph.n_nodes) return false;

    var ops_buf: [2 * moe_weighted_reduction_max_experts + 1]ggml.impl.ggml_c.enum_ggml_op = @splat(ggml.impl.ggml_c.GGML_OP_VIEW);
    const ops = ops_buf[0..@intCast(node_count)];
    ops[0] = ggml.impl.ggml_c.GGML_OP_MUL;
    if (mul_count == 2) ops[1] = ggml.impl.ggml_c.GGML_OP_MUL;
    var views_buf: [moe_weighted_reduction_max_experts]*const ggml.impl.ggml_c.ggml_tensor = undefined;
    var n_views: usize = 0;
    var previous: ?*const ggml.impl.ggml_c.ggml_tensor = null;
    var n_adds: c_int = 0;
    var offset: c_int = mul_count;
    while (offset < node_count) : (offset += 1) {
        const candidate = node(cgraph, node_idx + offset);
        ops[@intCast(offset)] = candidate.op;

        if (candidate.op == ggml.impl.ggml_c.GGML_OP_VIEW) {
            const expert: c_int = @intCast(n_views);
            if (expert >= n_expert_used or candidate.src[0] != weighted or candidate.view_src != weighted or
                candidate.type != ggml.impl.ggml_c.GGML_TYPE_F32 or candidate.ne[0] != weighted.ne[0] or
                candidate.ne[1] != n_tokens or candidate.ne[2] != 1 or candidate.ne[3] != 1 or
                candidate.nb[0] != weighted.nb[0] or candidate.nb[1] != weighted.nb[2] or
                candidate.view_offs != @as(usize, @bitCast(@as(isize, expert))) *% weighted.nb[1])
            {
                return false;
            }
            views_buf[n_views] = candidate;
            n_views += 1;
            continue;
        }

        if (candidate.op != ggml.impl.ggml_c.GGML_OP_ADD or n_views < 2 or n_adds + 1 >= @as(c_int, @intCast(n_views))) return false;
        const lhs: ?*const ggml.impl.ggml_c.ggml_tensor = if (n_adds == 0) views_buf[0] else previous;
        const rhs: *const ggml.impl.ggml_c.ggml_tensor = views_buf[@intCast(n_adds + 1)];
        if (candidate.src[0] != lhs or candidate.src[1] != rhs or candidate.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
        previous = candidate;
        n_adds += 1;
    }

    if (@as(c_int, @intCast(n_views)) != n_expert_used or n_adds != n_expert_used - 1 or previous == null) return false;
    const prev = previous.?;
    if (!ggml.impl.ggml_c.ggml_is_contiguous(prev) or prev.ne[0] != weighted.ne[0] or
        prev.ne[1] != n_tokens or prev.ne[2] != 1 or prev.ne[3] != 1)
    {
        return false;
    }

    const output_idx = node_idx + node_count - 1;
    if (!ggml.graph.canFuseSubgraph(cgraph, node_idx, ops, &.{output_idx})) return false;

    match.experts = experts;
    match.expert_scale = expert_scale;
    match.weights = weights;
    match.dst = node(cgraph, output_idx);
    match.node_count = node_count;
    return true;
}

fn opsEqual(a: []const ggml.impl.ggml_c.enum_ggml_op, b: []const ggml.impl.ggml_c.enum_ggml_op) bool {
    return std.mem.eql(ggml.impl.ggml_c.enum_ggml_op, a, b);
}

/// Ports `ggml_cuda_can_fuse` (ggml-cuda.cu:4046 @b1b770d90). The C's
/// `NDEBUG`-only assertion on `unary_ops.size()` compiles out.
fn canFuse(cgraph: *const ggml.impl.CGraph, node_idx: c_int, ops: []const ggml.impl.ggml_c.enum_ggml_op, unary_ops: []const ggml.impl.ggml_c.enum_ggml_unary_op) bool {
    const mul_mat_bias_glu_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_GLU };
    const mul_mat_id_bias_glu_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID, ggml.impl.ggml_c.GGML_OP_ADD_ID, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID, ggml.impl.ggml_c.GGML_OP_ADD_ID, ggml.impl.ggml_c.GGML_OP_GLU };

    const mul_mat_id_glu_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID, ggml.impl.ggml_c.GGML_OP_GLU };
    const mul_mat_glu_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_GLU };

    const n: c_int = @intCast(ops.len);

    if ((opsEqual(&mul_mat_bias_glu_ops, ops) or opsEqual(&mul_mat_id_bias_glu_ops, ops)) and
        ggml.graph.canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 4}))
    {
        const ffn_gate = node(cgraph, node_idx);
        const ffn_gate_bias = node(cgraph, node_idx + 1);
        const ffn_up = node(cgraph, node_idx + 2);
        const ffn_up_bias = node(cgraph, node_idx + 3);
        const glu = node(cgraph, node_idx + 4);

        if (ggml.cuda.mulmat.shouldFuseMulMat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias, null, null)) {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 4}, false);
        }
    }

    if ((opsEqual(&mul_mat_id_glu_ops, ops) or opsEqual(&mul_mat_glu_ops, ops)) and
        ggml.graph.canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 2}))
    {
        const ffn_gate = node(cgraph, node_idx);
        const ffn_up = node(cgraph, node_idx + 1);
        const glu = node(cgraph, node_idx + 2);

        if (ggml.cuda.mulmat.shouldFuseMulMat(ffn_up, ffn_gate, glu, null, null, null, null)) {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 2}, false);
        }
    }

    const rms_norm_mul_rope_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_RMS_NORM, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_ROPE };
    const rms_norm_mul_rope_set_rows_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_RMS_NORM, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_SET_ROWS };

    if (opsEqual(&rms_norm_mul_rope_set_rows_ops, ops) and ggml.graph.canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 4})) {
        const rms_norm = node(cgraph, node_idx);
        const mul = node(cgraph, node_idx + 1);
        const rope_t = node(cgraph, node_idx + 2);
        const view = node(cgraph, node_idx + 3);
        const set_rows = node(cgraph, node_idx + 4);

        if (ggml.graph.checkEdges(cgraph, node_idx, &.{ .{ 1, 0, 0 }, .{ 2, 0, 1 }, .{ 3, 0, 2 }, .{ 4, 0, 3 } }) and
            shouldFuseRmsNormMulRope(rms_norm, mul, rope_t) and
            shouldFuseRopeSetRows(rope_t, view, set_rows))
        {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 4}, false);
        }
    }

    if (opsEqual(&rms_norm_mul_rope_ops, ops) and ggml.graph.canFuse(cgraph, node_idx, ops)) {
        const rms_norm = node(cgraph, node_idx);
        const mul = node(cgraph, node_idx + 1);
        const rope_t = node(cgraph, node_idx + 2);

        if (shouldFuseRmsNormMulRope(rms_norm, mul, rope_t)) {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 2}, false);
        }
        return false;
    }

    const rope_set_rows_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_SET_ROWS };

    if (opsEqual(&rope_set_rows_ops, ops) and ggml.graph.canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 2})) {
        const rope_t = node(cgraph, node_idx);
        const view = node(cgraph, node_idx + 1);
        const set_rows = node(cgraph, node_idx + 2);

        if (shouldFuseRopeSetRows(rope_t, view, set_rows)) {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 2}, false);
        }
    }

    // scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine)
    const hc_post_gate_ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_SCALE, ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_SCALE, ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST };

    if (opsEqual(&hc_post_gate_ops, ops) and ggml.graph.canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 3})) {
        const scale_in = node(cgraph, node_idx);
        const sigmoid = node(cgraph, node_idx + 1);
        const scale_out = node(cgraph, node_idx + 2);

        // the chain must feed hc_post->src[2]; a bias on either scale is not handled
        if (ggml.graph.checkEdges(cgraph, node_idx, &.{ .{ 1, 0, 0 }, .{ 2, 0, 1 }, .{ 3, 2, 2 } }) and
            ggml.impl.ggml_c.ggml_get_unary_op(sigmoid) == ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID and
            ggml.impl.getOpParamsF32(scale_in, 1) == 0.0 and ggml.impl.getOpParamsF32(scale_out, 1) == 0.0 and
            src(scale_in, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and scale_out.type == ggml.impl.ggml_c.GGML_TYPE_F32 and
            ggml.impl.ggml_c.ggml_are_same_shape(src(scale_in, 0), scale_out))
        {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 3}, false);
        }
        return false;
    }

    if (!ggml.graph.canFuse(cgraph, node_idx, ops)) return false;

    if ((ops.len == 2 or ops.len == 3) and ops[0] == ggml.impl.ggml_c.GGML_OP_RMS_NORM and ops[1] == ggml.impl.ggml_c.GGML_OP_MUL) {
        const rms_norm = node(cgraph, node_idx);
        const mul = node(cgraph, node_idx + 1);
        var add: ?*const ggml.impl.ggml_c.ggml_tensor = null;

        if (ops.len == 3 and ops[2] == ggml.impl.ggml_c.GGML_OP_ADD) add = node(cgraph, node_idx + 2);

        ggml.impl.assert(src(rms_norm, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32, "rms_norm->src[0]->type == GGML_TYPE_F32");
        ggml.impl.assert(rms_norm.type == ggml.impl.ggml_c.GGML_TYPE_F32, "rms_norm->type == GGML_TYPE_F32");

        //rms norm only supports F32
        if (src(mul, 0).type != ggml.impl.ggml_c.GGML_TYPE_F32 or src(mul, 1).type != ggml.impl.ggml_c.GGML_TYPE_F32 or mul.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;

        if (add) |a| {
            if (src(a, 0).type != ggml.impl.ggml_c.GGML_TYPE_F32 or src(a, 1).type != ggml.impl.ggml_c.GGML_TYPE_F32 or a.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul.src[1] and !ggml.impl.ggml_c.ggml_are_same_shape(src(mul, 0), rms_norm)) return false;

        //rms_norm kernel assumes contiguous rows
        if (!ggml.impl.ggml_c.ggml_is_contiguous_rows(src(mul, 0)) or !ggml.impl.ggml_c.ggml_is_contiguous_rows(src(mul, 1))) return false;

        if (add) |a| {
            if (!ggml.impl.ggml_c.ggml_is_contiguous(src(a, 0)) or !ggml.impl.ggml_c.ggml_is_contiguous_rows(src(a, 1))) return false;
        }

        return true;
    }

    if (ops.len == 2 and ops[0] == ggml.impl.ggml_c.GGML_OP_RMS_NORM and ops[1] == ggml.impl.ggml_c.GGML_OP_SCALE) {
        const rms_norm = node(cgraph, node_idx);
        const scale = node(cgraph, node_idx + 1);

        ggml.impl.assert(src(rms_norm, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32, "rms_norm->src[0]->type == GGML_TYPE_F32");
        ggml.impl.assert(rms_norm.type == ggml.impl.ggml_c.GGML_TYPE_F32, "rms_norm->type == GGML_TYPE_F32");

        const bias = ggml.impl.getOpParamsF32(scale, 1);

        return bias == 0.0 and scale.type == ggml.impl.ggml_c.GGML_TYPE_F32;
    }

    if (ops.len == 2 and ops[0] == ggml.impl.ggml_c.GGML_OP_SSM_CONV and ops[1] == ggml.impl.ggml_c.GGML_OP_UNARY and
        unary_ops.len == 1 and unary_ops[0] == ggml.impl.ggml_c.GGML_UNARY_OP_SILU)
    {
        const ssm_conv_t = node(cgraph, node_idx);
        const silu = node(cgraph, node_idx + 1);
        if (ggml.impl.ggml_c.ggml_get_unary_op(silu) != unary_ops[0]) return false;
        if (ssm_conv_t.type != ggml.impl.ggml_c.GGML_TYPE_F32 or silu.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
        return true;
    }

    if (ops.len == 3 and ops[0] == ggml.impl.ggml_c.GGML_OP_SSM_CONV and ops[1] == ggml.impl.ggml_c.GGML_OP_ADD and
        ops[2] == ggml.impl.ggml_c.GGML_OP_UNARY and unary_ops.len == 1 and unary_ops[0] == ggml.impl.ggml_c.GGML_UNARY_OP_SILU)
    {
        const ssm_conv_t = node(cgraph, node_idx);
        const add = node(cgraph, node_idx + 1);
        const silu = node(cgraph, node_idx + 2);
        if (ggml.impl.ggml_c.ggml_get_unary_op(silu) != unary_ops[0]) return false;

        if (ssm_conv_t.type != ggml.impl.ggml_c.GGML_TYPE_F32 or add.type != ggml.impl.ggml_c.GGML_TYPE_F32 or silu.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const bias = if (add.src[0] == ssm_conv_t) src(add, 1) else src(add, 0);
        if (bias.type != ggml.impl.ggml_c.GGML_TYPE_F32 or !ggml.impl.ggml_c.ggml_is_contiguous(bias)) return false;
        if (ggml.impl.ggml_c.ggml_nelements(bias) != ssm_conv_t.ne[0] or bias.ne[0] != ssm_conv_t.ne[0]) return false;

        return true;
    }

    if (ops.len == 2 and ops[0] == ggml.impl.ggml_c.GGML_OP_UNARY and ops[1] == ggml.impl.ggml_c.GGML_OP_MUL and unary_ops.len == 1 and
        (unary_ops[0] == ggml.impl.ggml_c.GGML_UNARY_OP_SILU or unary_ops[0] == ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID or unary_ops[0] == ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS))
    {
        const un = node(cgraph, node_idx);
        const mul = node(cgraph, node_idx + 1);

        if (ggml.impl.ggml_c.ggml_get_unary_op(un) != unary_ops[0]) return false;
        if (un.type != ggml.impl.ggml_c.GGML_TYPE_F32 and un.type != ggml.impl.ggml_c.GGML_TYPE_F16 and un.type != ggml.impl.ggml_c.GGML_TYPE_BF16) return false;
        if (un.type != mul.type) return false;

        const other = if (mul.src[0] == un) src(mul, 1) else src(mul, 0);
        if (other.type != un.type) return false;
        if (!ggml.impl.ggml_c.ggml_is_contiguous_1(other) or !ggml.impl.ggml_c.ggml_is_contiguous_1(src(un, 0)) or !ggml.impl.ggml_c.ggml_are_same_shape(other, un)) return false;

        return true;
    }

    if (ops.len == 2 and ops[0] == ggml.impl.ggml_c.GGML_OP_UNARY and ops[1] == ggml.impl.ggml_c.GGML_OP_SQR and
        unary_ops.len == 1 and unary_ops[0] == ggml.impl.ggml_c.GGML_UNARY_OP_RELU)
    {
        const un = node(cgraph, node_idx);
        const sqr = node(cgraph, node_idx + 1);

        if (ggml.impl.ggml_c.ggml_get_unary_op(un) != ggml.impl.ggml_c.GGML_UNARY_OP_RELU) return false;
        if (un.type != ggml.impl.ggml_c.GGML_TYPE_F32 and un.type != ggml.impl.ggml_c.GGML_TYPE_F16) return false;
        if (un.type != sqr.type) return false;
        if (!ggml.impl.ggml_c.ggml_is_contiguous(src(un, 0))) return false;

        return true;
    }

    if (ops.len == 3 and ops[0] == ggml.impl.ggml_c.GGML_OP_SCALE and ops[1] == ggml.impl.ggml_c.GGML_OP_UNARY and ops[2] == ggml.impl.ggml_c.GGML_OP_SCALE and
        unary_ops.len == 1 and unary_ops[0] == ggml.impl.ggml_c.GGML_UNARY_OP_TANH)
    {
        const scale = node(cgraph, node_idx);
        const tanh = node(cgraph, node_idx + 1);
        const scale2 = node(cgraph, node_idx + 2);

        // the fused softcap kernel is F32 only
        if (src(scale, 0).type != ggml.impl.ggml_c.GGML_TYPE_F32 or scale.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;

        // TODO: make the fused softcap kernel stride aware, it reads src and writes dst as dense
        if (!ggml.impl.ggml_c.ggml_is_contiguous(src(scale, 0)) or !ggml.impl.ggml_c.ggml_is_contiguous(scale2)) return false;

        if (ggml.impl.ggml_c.ggml_get_unary_op(tanh) != ggml.impl.ggml_c.GGML_UNARY_OP_TANH) return false;

        // Check for bias
        if (ggml.impl.getOpParamsF32(scale, 1) != 0.0 or ggml.impl.getOpParamsF32(scale2, 1) != 0.0) return false;

        return true;
    }

    return false;
}

/// Ports `ggml_cuda_match_shared_expert` (ggml-cuda.cu:2281 @b1b770d90):
/// whether the routed experts' gate/up/GLU at `routed_idx`
/// and a shared expert's at `shared_idx` can run as one MMVQ launch --
/// both pairs fusable, the same quantized weight shape, the shared input
/// the routed one before its reshape, the same GLU and limit. Only checks
/// that do not depend on the batch size, so that `graph_optimize` gives
/// every ubatch the same topology; `tryFuse` checks the batch.
pub fn matchSharedExpert(graph: *const ggml.impl.CGraph, routed_idx: c_int, shared_idx: c_int) bool {
    if (routed_idx + 2 >= graph.n_nodes or shared_idx + 2 >= graph.n_nodes or shared_idx < routed_idx + 3) {
        return false;
    }
    const nodes = [_]c_int{ routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ops = [_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID, ggml.impl.ggml_c.GGML_OP_GLU, ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_GLU };
    const outputs = [_]c_int{ routed_idx + 2, shared_idx + 2 };
    if (!ggml.graph.ggml_can_fuse_subgraph_ext(graph, &nodes, 6, &ops, &outputs, 2)) {
        return false;
    }

    const routed = node(graph, routed_idx + 2);
    const shared = node(graph, shared_idx + 2);
    const gate = src(routed, 0);
    const up = src(routed, 1);
    const shared_gate = src(shared, 0);
    const shared_up = src(shared, 1);
    const isPair = struct {
        fn f(g: *const ggml.impl.CGraph, a: *const ggml.impl.ggml_c.ggml_tensor, b: *const ggml.impl.ggml_c.ggml_tensor, idx: c_int) bool {
            return (a == node(g, idx) and b == node(g, idx + 1)) or
                (b == node(g, idx) and a == node(g, idx + 1));
        }
    }.f;
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!isPair(graph, gate, up, routed_idx) or !isPair(graph, shared_gate, shared_up, shared_idx) or
        !ggml.cuda.mulmat.shouldFuseMulMat(up, gate, routed, null, null, null, null) or
        !ggml.cuda.mulmat.shouldFuseMulMat(shared_up, shared_gate, shared, null, null, null, null) or
        src(up, 0).buffer == null or
        !ggml.impl.ggml_c.ggml_is_quantized(src(up, 0).type))
    {
        return false;
    }
    const input = src(up, 1);
    const weight = src(up, 0);
    const shared_weight = src(shared_up, 0);
    if (input.op != ggml.impl.ggml_c.GGML_OP_RESHAPE or input.src[0] != shared_up.src[1] or
        input.ne[1] != 1 or input.ne[3] != 1 or !ggml.impl.ggml_c.ggml_is_contiguous(input) or
        !ggml.impl.ggml_c.ggml_is_contiguous(src(shared_up, 1)) or !ggml.impl.ggml_c.ggml_is_matrix(src(shared_up, 1)) or
        weight.type != shared_weight.type or weight.ne[0] != shared_weight.ne[0] or
        weight.ne[1] != shared_weight.ne[1] or weight.nb[1] != shared_weight.nb[1] or weight.ne[3] != 1 or
        !ggml.impl.ggml_c.ggml_is_matrix(shared_weight) or !ggml.impl.ggml_c.ggml_is_contiguous(shared_weight) or
        !ggml.impl.ggml_c.ggml_is_contiguous(src(shared_gate, 0)) or !ggml.impl.ggml_c.ggml_is_contiguous(routed) or !ggml.impl.ggml_c.ggml_is_contiguous(shared))
    {
        return false;
    }
    if (shared_weight.op != ggml.impl.ggml_c.GGML_OP_NONE or src(shared_gate, 0).op != ggml.impl.ggml_c.GGML_OP_NONE or
        ggml.impl.ggml_c.ggml_get_glu_op(routed) != ggml.impl.ggml_c.ggml_get_glu_op(shared) or
        ggml.impl.getOpParamsF32(routed, 3) != ggml.impl.getOpParamsF32(shared, 3))
    {
        return false;
    }
    return true;
}

/// The C's `get_bias_tensor` lambda in `ggml_cuda_try_fuse`.
fn getBiasTensor(bias_node: *const ggml.impl.ggml_c.ggml_tensor, mul_node: *const ggml.impl.ggml_c.ggml_tensor, op_bias: ggml.impl.ggml_c.enum_ggml_op) ?*const ggml.impl.ggml_c.ggml_tensor {
    if (op_bias == ggml.impl.ggml_c.GGML_OP_ADD) {
        if (bias_node.src[0] == mul_node) return src(bias_node, 1);
        if (bias_node.src[1] == mul_node) return src(bias_node, 0);
        return null;
    }
    ggml.impl.assert(op_bias == ggml.impl.ggml_c.GGML_OP_ADD_ID, "op_bias == GGML_OP_ADD_ID");
    ggml.impl.assert(bias_node.src[0] == mul_node, "bias_node->src[0] == mul_node");
    return src(bias_node, 1);
}

fn srcOpt(t: *const ggml.impl.ggml_c.ggml_tensor, i: usize) ?*const ggml.impl.ggml_c.ggml_tensor {
    return if (t.src[i]) |s| s else null;
}

/// Ports `ggml_cuda_try_fuse` (ggml-cuda.cu:4338 @b1b770d90): try and fuse
/// nodes, and return the number of nodes to skip. It includes the
/// `SSM_SCAN -> CPY` cache fusion, beside the gated-delta-net one. The
/// scale lambdas (#31296, #26048) are in `mulmat.zig`, where
/// `graph_optimize` shares them, and lets MMVF (FP8 weights) and MMQ (NVFP4)
/// take the scaled gate/up/GLU and the scaled matmul, cuBLASLt's FP8 GEMM
/// the scaled FP8 `MUL_MAT`.
pub fn tryFuse(cuda_ctx: *ggml.cuda.common.Context, cgraph: *ggml.impl.CGraph, i: c_int) c_int {
    if (ggml.cuda.common.fusionDisabled()) return 0;

    const nd = node(cgraph, i);

    if (nd.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID and cuda_ctx.concurrent_stream_context.concurrent_events.count() == 0 and
        matchSharedExpert(cgraph, i, i + 3) and
        ggml.cuda.mulmat.shouldFuseMulMatVecQ(src(node(cgraph, i + 2), 1)))
    {
        const outputs = [_]c_int{ i + 2, i + 5 };
        if (checkFusionMemoryRanges(cgraph, i, 6, &outputs, false)) {
            const routed = node(cgraph, i + 2);
            const shared = node(cgraph, i + 5);
            const up = src(routed, 1);
            var fusion: ggml.cuda.common.MmFusionArgsHost = .{};
            fusion.gate = src(src(routed, 0), 0);
            fusion.glu_op = ggml.impl.ggml_c.ggml_get_glu_op(routed);
            fusion.glu_limit = ggml.impl.getOpParamsF32(routed, 3);
            fusion.shared_up = src(src(shared, 1), 0);
            fusion.shared_gate = src(src(shared, 0), 0);
            fusion.shared_dst = shared;
            ggml.cuda.mmvq.mulMatVecQ(cuda_ctx, src(up, 0), src(up, 1), src(up, 2), routed, &fusion);
            return 5;
        }
    }

    if (nd.op == ggml.impl.ggml_c.GGML_OP_MUL) {
        var match: MoeWeightedReductionMatch = .{};
        if (matchMoeWeightedReduction(cgraph, i, &match)) {
            const output_idx = i + match.node_count - 1;
            if (checkFusionMemoryRanges(cgraph, i, match.node_count, &.{output_idx}, false)) {
                ggml.cuda.moe_weighted_reduction.opMoeWeightedReduction(cuda_ctx, match.experts.?, match.expert_scale, match.weights.?, match.dst.?);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (nd.op == ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET) {
        var fused_state_cpy: ggml.cuda.gated_delta_net.FusedCache = undefined;
        const nodes_to_skip = tryGdnCacheFusion(cgraph, i, &fused_state_cpy);
        if (nodes_to_skip > 0) {
            ggml.cuda.gated_delta_net.opGatedDeltaNetFusedCache(cuda_ctx, nd, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (nd.op == ggml.impl.ggml_c.GGML_OP_SSM_SCAN) {
        var fused_state_cpy: ggml.cuda.ssm_scan.FusedCache = undefined;
        const nodes_to_skip = trySsmScanCacheFusion(cgraph, i, &fused_state_cpy);
        if (nodes_to_skip > 0) {
            ggml.cuda.ssm_scan.opSsmScanFusedCache(cuda_ctx, nd, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (nd.op == ggml.impl.ggml_c.GGML_OP_UNARY or nd.op == ggml.impl.ggml_c.GGML_OP_SOFT_MAX or nd.op == ggml.impl.ggml_c.GGML_OP_ARGSORT) {
        var args: ggml.cuda.topk_moe.Args = .{};
        const can_fuse = topkMoeFusion(cgraph, i, &args);
        // two for the gating, ADD and RESHAPE, the group pattern, three for
        // the selection, five for the norm and one for the scale
        var ops_buf: [32]ggml.impl.ggml_c.enum_ggml_op = undefined;
        var n_ops: usize = 0;
        const Append = struct {
            fn f(buf: []ggml.impl.ggml_c.enum_ggml_op, n: *usize, items: []const ggml.impl.ggml_c.enum_ggml_op) void {
                for (items) |x| {
                    buf[n.*] = x;
                    n.* += 1;
                }
            }
        };

        if (can_fuse) {
            const logits = src(nd, 0);
            var weights: *ggml.impl.ggml_c.ggml_tensor = undefined;
            var ids: *ggml.impl.ggml_c.ggml_tensor = undefined;
            var bias: ?*const ggml.impl.ggml_c.ggml_tensor = null;
            var clamp: ?*const ggml.impl.ggml_c.ggml_tensor = null;
            var scale: ?*const ggml.impl.ggml_c.ggml_tensor = null;

            if (!args.delayed_softmax) {
                var out_nodes: [2]c_int = undefined; // nodes which can't be elided

                if (args.sigmoid) {
                    Append.f(&ops_buf, &n_ops, &.{ggml.impl.ggml_c.GGML_OP_UNARY});
                } else if (args.sqrt_softplus) {
                    Append.f(&ops_buf, &n_ops, &.{ ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_SQRT });
                } else {
                    Append.f(&ops_buf, &n_ops, &.{ggml.impl.ggml_c.GGML_OP_SOFT_MAX});
                }
                Append.f(&ops_buf, &n_ops, &.{ggml.impl.ggml_c.GGML_OP_RESHAPE});
                if (args.prob_bias) {
                    Append.f(&ops_buf, &n_ops, &.{ggml.impl.ggml_c.GGML_OP_ADD});
                    bias = srcOpt(node(cgraph, i + @as(c_int, @intCast(n_ops)) - 1), 1);
                }
                if (args.n_groups > 0) {
                    for (topk_moe_group_pattern) |p| Append.f(&ops_buf, &n_ops, &.{p.op});
                }
                Append.f(&ops_buf, &n_ops, &.{ ggml.impl.ggml_c.GGML_OP_ARGSORT, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_GET_ROWS });
                out_nodes[0] = i + @as(c_int, @intCast(n_ops)) - 2;
                ids = node(cgraph, out_nodes[0]);

                if (args.norm) {
                    Append.f(&ops_buf, &n_ops, &.{ ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_CLAMP, ggml.impl.ggml_c.GGML_OP_DIV, ggml.impl.ggml_c.GGML_OP_RESHAPE });
                    clamp = node(cgraph, i + @as(c_int, @intCast(n_ops)) - 3);
                }
                if (args.scale) {
                    Append.f(&ops_buf, &n_ops, &.{ggml.impl.ggml_c.GGML_OP_SCALE});
                    scale = node(cgraph, i + @as(c_int, @intCast(n_ops)) - 1);
                }

                weights = node(cgraph, i + @as(c_int, @intCast(n_ops)) - 1);
                out_nodes[1] = i + @as(c_int, @intCast(n_ops)) - 1;

                if (ggml.graph.canFuseSubgraph(cgraph, i, ops_buf[0..n_ops], &out_nodes) and
                    ggml.cuda.topk_moe.shouldUseTopkMoe(nd, logits, weights, ids) and
                    checkFusionMemoryRanges(cgraph, i, @intCast(n_ops), &out_nodes, true))
                {
                    ggml.cuda.topk_moe.opTopkMoe(cuda_ctx, logits, weights, ids, clamp, scale, bias, &args);
                    return @as(c_int, @intCast(n_ops)) - 1;
                }
            } else if (!args.norm and !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                Append.f(&ops_buf, &n_ops, &.{ ggml.impl.ggml_c.GGML_OP_ARGSORT, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_SOFT_MAX, ggml.impl.ggml_c.GGML_OP_RESHAPE });
                weights = node(cgraph, i + 5);
                ids = node(cgraph, i + 1);
                const softmax = node(cgraph, i + 4);

                const out_nodes = [2]c_int{ i + 1, i + 5 };
                if (ggml.graph.canFuseSubgraph(cgraph, i, ops_buf[0..n_ops], &out_nodes) and
                    ggml.cuda.topk_moe.shouldUseTopkMoe(softmax, logits, weights, ids) and
                    checkFusionMemoryRanges(cgraph, i, @intCast(n_ops), &out_nodes, true))
                {
                    ggml.cuda.topk_moe.opTopkMoe(cuda_ctx, logits, weights, ids, clamp, scale, bias, &args);
                    return @as(c_int, @intCast(n_ops)) - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_SET_ROWS }, &.{})) {
        ggml.cuda.rope.opRopeFused(cuda_ctx, node(cgraph, i), node(cgraph, i + 2));
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml.graph.canFuseSubgraph(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_SIN, ggml.impl.ggml_c.GGML_OP_SQR, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_ADD }, &.{i + 4})) {
        const mul0 = node(cgraph, i);
        const sqr = node(cgraph, i + 2);
        const mul1 = node(cgraph, i + 3);
        const add = node(cgraph, i + 4);

        // x carries the full activation shape, a is the broadcast operand
        const x = if (ggml.impl.ggml_c.ggml_are_same_shape(mul0, src(mul0, 0))) src(mul0, 0) else src(mul0, 1);
        const a = if (x == mul0.src[0]) src(mul0, 1) else src(mul0, 0);

        // mul1 reads sqr and inv_b in either operand order
        const inv_b = if (mul1.src[0] == sqr) src(mul1, 1) else src(mul1, 0);

        // closure check: the trailing add must read the same x as the leading mul
        const x_in_add = if (add.src[0] == mul1) src(add, 1) else src(add, 0);

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1].
        const dim_ok = (x.ne[2] == 1 and x.ne[3] == 1) and
            (add.ne[2] == 1 and add.ne[3] == 1) and
            (a.ne[2] == 1 and a.ne[3] == 1);
        const shape_ok = ggml.impl.ggml_c.ggml_are_same_shape(a, inv_b) and a.ne[0] == 1 and a.ne[1] == x.ne[1];

        const sin1 = node(cgraph, i + 1);
        const types_ok = (x.type == ggml.impl.ggml_c.GGML_TYPE_F32 or x.type == ggml.impl.ggml_c.GGML_TYPE_F16 or x.type == ggml.impl.ggml_c.GGML_TYPE_BF16) and
            (a.type == ggml.impl.ggml_c.GGML_TYPE_F32) and (inv_b.type == ggml.impl.ggml_c.GGML_TYPE_F32) and
            (mul0.type == x.type) and (sin1.type == x.type) and
            (sqr.type == x.type) and (mul1.type == x.type) and
            (add.type == x.type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every
        // operand is contiguous
        const contig_ok = ggml.impl.ggml_c.ggml_is_contiguous(x) and ggml.impl.ggml_c.ggml_is_contiguous(add) and
            ggml.impl.ggml_c.ggml_is_contiguous(a) and ggml.impl.ggml_c.ggml_is_contiguous(inv_b);

        if (types_ok and shape_ok and dim_ok and contig_ok and x_in_add == x) {
            ggml.cuda.snake.opSnakeFused(cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (nd.op == ggml.impl.ggml_c.GGML_OP_ADD or nd.op == ggml.impl.ggml_c.GGML_OP_MUL) {
        var n_fuse: c_int = 0;
        var ops: [8]ggml.impl.ggml_c.enum_ggml_op = @splat(nd.op);

        while (n_fuse <= 6) : (n_fuse += 1) {
            if (!ggml.graph.canFuse(cgraph, i + n_fuse, ops[@intCast(n_fuse)..][0..2])) break;
            if (node(cgraph, i + n_fuse) != node(cgraph, i + n_fuse + 1).src[0]) break;
            if (!ggml.impl.areSameLayout(src(node(cgraph, i + n_fuse), 1), src(node(cgraph, i + n_fuse + 1), 1))) break;
        }
        _ = &ops;

        n_fuse += 1;

        if (n_fuse > 1) {
            var fused_node: ggml.impl.ggml_c.ggml_tensor = nd.*;
            var j: c_int = 0;
            while (j < n_fuse - 1) : (j += 1) {
                fused_node.src[@intCast(j + 2)] = node(cgraph, i + j + 1).src[1];
            }
            fused_node.data = node(cgraph, i + n_fuse - 1).data;
            if (nd.op == ggml.impl.ggml_c.GGML_OP_ADD) {
                ggml.cuda.binbcast.opFused(.add, cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml.cuda.binbcast.opFused(.mul, cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    var fused_mul_mat_vec = false;
    var fused_node_count: c_int = 0;

    // gate + glu + up, with optional scale/bias on both lanes.
    for ([_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID }) |op| {
        const bias_op: ggml.impl.ggml_c.enum_ggml_op = if (op == ggml.impl.ggml_c.GGML_OP_MUL_MAT) ggml.impl.ggml_c.GGML_OP_ADD else ggml.impl.ggml_c.GGML_OP_ADD_ID;

        if (op == ggml.impl.ggml_c.GGML_OP_MUL_MAT) {
            for ([_]bool{ false, true }) |with_bias| {
                const gate_idx = i;
                const gate_scale_idx = i + 1;
                const gate_bias_idx: c_int = if (with_bias) i + 2 else -1;
                const up_idx: c_int = if (with_bias) i + 3 else i + 2;
                const up_scale_idx = up_idx + 1;
                const up_bias_idx: c_int = if (with_bias) up_idx + 2 else -1;
                const glu_idx: c_int = if (with_bias) up_idx + 3 else up_idx + 2;

                const out_nodes = [_]c_int{glu_idx};
                const ops: []const ggml.impl.ggml_c.enum_ggml_op = if (with_bias)
                    &.{ op, ggml.impl.ggml_c.GGML_OP_MUL, bias_op, op, ggml.impl.ggml_c.GGML_OP_MUL, bias_op, ggml.impl.ggml_c.GGML_OP_GLU }
                else
                    &.{ op, ggml.impl.ggml_c.GGML_OP_MUL, op, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_GLU };
                const n_ops: c_int = @intCast(ops.len);

                if (!ggml.graph.canFuseSubgraph(cgraph, i, ops, &out_nodes) or
                    !checkFusionMemoryRanges(cgraph, i, n_ops, &out_nodes, false)) continue;

                const gate_n = node(cgraph, gate_idx);
                const gate_scale_n = node(cgraph, gate_scale_idx);
                const gate_out_n = if (with_bias) node(cgraph, gate_bias_idx) else gate_scale_n;
                const up_n = node(cgraph, up_idx);
                const up_scale_n = node(cgraph, up_scale_idx);
                const up_out_n = if (with_bias) node(cgraph, up_bias_idx) else up_scale_n;
                const glu = node(cgraph, glu_idx);

                if (!ggml.cuda.mulmat.shouldFuseMulMat(up_n, gate_n, glu, if (with_bias) up_out_n else null, if (with_bias) gate_out_n else null, up_scale_n, gate_scale_n)) continue;

                const gate_scale = ggml.cuda.mulmat.getMulMatScale(gate_scale_n, gate_n);
                const up_scale = ggml.cuda.mulmat.getMulMatScale(up_scale_n, up_n);
                if (gate_scale == null or up_scale == null) continue;

                const up_bias = if (with_bias) getBiasTensor(up_out_n, up_scale_n, bias_op) else null;
                const gate_bias = if (with_bias) getBiasTensor(gate_out_n, gate_scale_n, bias_op) else null;
                if (with_bias and (!ggml.impl.ggml_c.ggml_are_same_shape(src(gate_out_n, 0), src(gate_out_n, 1)) or
                    !ggml.impl.ggml_c.ggml_are_same_shape(src(up_out_n, 0), src(up_out_n, 1)))) continue;

                const fusion_data: ggml.cuda.common.MmFusionArgsHost = .{
                    .gate = src(gate_n, 0),
                    .x_bias = up_bias,
                    .gate_bias = gate_bias,
                    .x_scale = up_scale,
                    .gate_scale = gate_scale,
                    .glu_op = ggml.impl.ggml_c.ggml_get_glu_op(glu),
                    .glu_limit = ggml.impl.getOpParamsF32(glu, 3),
                };

                if (ggml.cuda.mulmat.shouldFuseMulMatVecF(up_n, gate_n)) {
                    ggml.cuda.mmvf.mulMatVecF(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), node(cgraph, glu_idx), &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }

                if (ggml.cuda.mulmat.shouldFuseMulMatVecQ(up_n)) {
                    ggml.cuda.mmvq.mulMatVecQ(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), node(cgraph, glu_idx), &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }

                if (!with_bias and ggml.cuda.mulmat.shouldFuseMulMatQ(up_n, gate_n, glu, up_scale_n, gate_scale_n)) {
                    var fusion_q = fusion_data;
                    fusion_q.prec_src1 = ggml.impl.ggml_c.ggml_prec_get_src(up_n, 1);
                    ggml.cuda.mmq.mulMatQ(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), node(cgraph, glu_idx), &fusion_q);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) break;
        } else {
            for ([_]bool{ false, true }) |with_bias| {
                const gate_idx = i;
                const gate_scale_idx = i + 4;
                const gate_bias_idx: c_int = if (with_bias) i + 5 else -1;
                const up_idx: c_int = if (with_bias) i + 6 else i + 5;
                const up_scale_idx = up_idx + 4;
                const up_bias_idx: c_int = if (with_bias) up_idx + 5 else -1;
                const glu_idx: c_int = if (with_bias) up_idx + 6 else up_idx + 5;

                const out_nodes = [_]c_int{glu_idx};
                const ops: []const ggml.impl.ggml_c.enum_ggml_op = if (with_bias)
                    &.{ op, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.impl.ggml_c.GGML_OP_MUL, bias_op, op, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.impl.ggml_c.GGML_OP_MUL, bias_op, ggml.impl.ggml_c.GGML_OP_GLU }
                else
                    &.{ op, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.impl.ggml_c.GGML_OP_MUL, op, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_GLU };
                const n_ops: c_int = @intCast(ops.len);

                if (!ggml.graph.canFuseSubgraph(cgraph, i, ops, &out_nodes) or
                    !checkFusionMemoryRanges(cgraph, i, n_ops, &out_nodes, false)) continue;

                const gate_n = node(cgraph, gate_idx);
                const gate_scale_n = node(cgraph, gate_scale_idx);
                const gate_out_n = if (with_bias) node(cgraph, gate_bias_idx) else gate_scale_n;
                const up_n = node(cgraph, up_idx);
                const up_scale_n = node(cgraph, up_scale_idx);
                const up_out_n = if (with_bias) node(cgraph, up_bias_idx) else up_scale_n;
                const glu = node(cgraph, glu_idx);

                if (!ggml.cuda.mulmat.shouldFuseMulMat(up_n, gate_n, glu, if (with_bias) up_out_n else null, if (with_bias) gate_out_n else null, up_scale_n, gate_scale_n)) continue;

                const gate_scale = ggml.cuda.mulmat.getMulMatIdScale(node(cgraph, gate_idx + 1), node(cgraph, gate_idx + 2), node(cgraph, gate_idx + 3), gate_scale_n, gate_n);
                const up_scale = ggml.cuda.mulmat.getMulMatIdScale(node(cgraph, up_idx + 1), node(cgraph, up_idx + 2), node(cgraph, up_idx + 3), up_scale_n, up_n);
                if (gate_scale == null or up_scale == null) continue;

                const up_bias = if (with_bias) getBiasTensor(up_out_n, up_scale_n, bias_op) else null;
                const gate_bias = if (with_bias) getBiasTensor(gate_out_n, gate_scale_n, bias_op) else null;

                const fusion_data: ggml.cuda.common.MmFusionArgsHost = .{
                    .gate = src(gate_n, 0),
                    .x_bias = up_bias,
                    .gate_bias = gate_bias,
                    .x_scale = up_scale,
                    .gate_scale = gate_scale,
                    .glu_op = ggml.impl.ggml_c.ggml_get_glu_op(glu),
                    .glu_limit = ggml.impl.getOpParamsF32(glu, 3),
                };

                if (ggml.cuda.mulmat.shouldFuseMulMatVecF(up_n, gate_n)) {
                    ggml.cuda.mmvf.mulMatVecF(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), node(cgraph, glu_idx), &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }

                if (ggml.cuda.mulmat.shouldFuseMulMatVecQ(up_n)) {
                    ggml.cuda.mmvq.mulMatVecQ(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), node(cgraph, glu_idx), &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }

                if (!with_bias and ggml.cuda.mulmat.shouldFuseMulMatQ(up_n, gate_n, glu, up_scale_n, gate_scale_n)) {
                    var fusion_q = fusion_data;
                    fusion_q.prec_src1 = ggml.impl.ggml_c.ggml_prec_get_src(up_n, 1);
                    ggml.cuda.mmq.mulMatQ(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), node(cgraph, glu_idx), &fusion_q);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) break;
        }

        if (canFuse(cgraph, i, &.{ op, bias_op, op, bias_op, ggml.impl.ggml_c.GGML_OP_GLU }, &.{})) {
            const glu = node(cgraph, i + 4);
            const gate_bias_n = src(glu, 0);
            const up_bias_n = src(glu, 1);

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            var gate_n: *ggml.impl.ggml_c.ggml_tensor = undefined;
            var up_n: *ggml.impl.ggml_c.ggml_tensor = undefined;

            if (gate_bias_n.src[0] == node(cgraph, i) or gate_bias_n.src[1] == node(cgraph, i)) {
                gate_n = node(cgraph, i);
                up_n = node(cgraph, i + 2);
            } else if (gate_bias_n.src[0] == node(cgraph, i + 2) or gate_bias_n.src[1] == node(cgraph, i + 2)) {
                gate_n = node(cgraph, i + 2);
                up_n = node(cgraph, i);
            } else {
                continue;
            }

            const up_bias_tensor = getBiasTensor(up_bias_n, up_n, bias_op);
            const gate_bias_tensor = getBiasTensor(gate_bias_n, gate_n, bias_op);

            if (up_bias_tensor == null or gate_bias_tensor == null) continue;

            // we don't support repeating adds
            if (bias_op == ggml.impl.ggml_c.GGML_OP_ADD and (!ggml.impl.ggml_c.ggml_are_same_shape(src(gate_bias_n, 0), src(gate_bias_n, 1)) or
                !ggml.impl.ggml_c.ggml_are_same_shape(src(up_bias_n, 0), src(up_bias_n, 1)))) continue;

            if (ggml.cuda.mulmat.shouldFuseMulMatVecF(up_n, gate_n)) {
                const fusion_data: ggml.cuda.common.MmFusionArgsHost = .{
                    .gate = src(gate_n, 0),
                    .x_bias = up_bias_tensor,
                    .gate_bias = gate_bias_tensor,
                    .glu_op = ggml.impl.ggml_c.ggml_get_glu_op(glu),
                    .glu_limit = ggml.impl.getOpParamsF32(glu, 3),
                };
                ggml.cuda.mmvf.mulMatVecF(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 5;
                break;
            }

            if (ggml.cuda.mulmat.shouldFuseMulMatVecQ(up_n)) {
                const fusion_data: ggml.cuda.common.MmFusionArgsHost = .{
                    .gate = src(gate_n, 0),
                    .x_bias = up_bias_tensor,
                    .gate_bias = gate_bias_tensor,
                    .glu_op = ggml.impl.ggml_c.ggml_get_glu_op(glu),
                    .glu_limit = ggml.impl.getOpParamsF32(glu, 3),
                };
                ggml.cuda.mmvq.mulMatVecQ(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 5;
                break;
            }
        } else if (canFuse(cgraph, i, &.{ op, op, ggml.impl.ggml_c.GGML_OP_GLU }, &.{})) {
            const glu = node(cgraph, i + 2);
            const gate = src(glu, 0);
            const up = src(glu, 1);

            const ok = (gate == node(cgraph, i) and up == node(cgraph, i + 1)) or
                (gate == node(cgraph, i + 1) and up == node(cgraph, i));

            if (!ok) continue;

            if (ggml.cuda.mulmat.shouldFuseMulMatVecF(up, gate)) {
                const fusion_data: ggml.cuda.common.MmFusionArgsHost = .{ .gate = src(gate, 0), .glu_op = ggml.impl.ggml_c.ggml_get_glu_op(glu), .glu_limit = ggml.impl.getOpParamsF32(glu, 3) };
                ggml.cuda.mmvf.mulMatVecF(cuda_ctx, src(up, 0), src(up, 1), srcOpt(up, 2), glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 3;
                break;
            }

            if (ggml.cuda.mulmat.shouldFuseMulMatVecQ(up)) {
                const fusion_data: ggml.cuda.common.MmFusionArgsHost = .{ .gate = src(gate, 0), .glu_op = ggml.impl.ggml_c.ggml_get_glu_op(glu), .glu_limit = ggml.impl.getOpParamsF32(glu, 3) };
                ggml.cuda.mmvq.mulMatVecQ(cuda_ctx, src(up, 0), src(up, 1), srcOpt(up, 2), glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 3;
                break;
            }

            if (ggml.cuda.mulmat.shouldFuseMulMatQ(up, gate, glu, null, null)) {
                const fusion_data: ggml.cuda.common.MmFusionArgsHost = .{ .gate = src(gate, 0), .glu_op = ggml.impl.ggml_c.ggml_get_glu_op(glu), .glu_limit = ggml.impl.getOpParamsF32(glu, 3), .prec_src1 = ggml.impl.ggml_c.ggml_prec_get_src(up, 1) };
                ggml.cuda.mmq.mulMatQ(cuda_ctx, src(up, 0), src(up, 1), srcOpt(up, 2), glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) return fused_node_count - 1;

    fused_mul_mat_vec = false;
    fused_node_count = 0;

    // mul_mat + scale + optional bias
    outer: for ([_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID }) |op| {
        const bias_op: ggml.impl.ggml_c.enum_ggml_op = if (op == ggml.impl.ggml_c.GGML_OP_MUL_MAT) ggml.impl.ggml_c.GGML_OP_ADD else ggml.impl.ggml_c.GGML_OP_ADD_ID;

        for ([_]bool{ false, true }) |with_bias| {
            const ops: []const ggml.impl.ggml_c.enum_ggml_op = if (op == ggml.impl.ggml_c.GGML_OP_MUL_MAT)
                (if (with_bias) &.{ op, ggml.impl.ggml_c.GGML_OP_MUL, bias_op } else &.{ op, ggml.impl.ggml_c.GGML_OP_MUL })
            else if (with_bias)
                &.{ op, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.impl.ggml_c.GGML_OP_MUL, bias_op }
            else
                &.{ op, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.impl.ggml_c.GGML_OP_MUL };
            const n_ops: c_int = @intCast(ops.len);
            const out_nodes = [_]c_int{i + n_ops - 1};

            if (!ggml.graph.canFuseSubgraph(cgraph, i, ops, &out_nodes) or
                !checkFusionMemoryRanges(cgraph, i, n_ops, &out_nodes, false)) continue;

            const mm_node = node(cgraph, i);
            const scale_node = if (op == ggml.impl.ggml_c.GGML_OP_MUL_MAT) node(cgraph, i + 1) else node(cgraph, i + 4);
            const out_node = if (with_bias) node(cgraph, i + n_ops - 1) else scale_node;

            const scale = if (op == ggml.impl.ggml_c.GGML_OP_MUL_MAT)
                ggml.cuda.mulmat.getMulMatScale(scale_node, mm_node)
            else
                ggml.cuda.mulmat.getMulMatIdScale(node(cgraph, i + 1), node(cgraph, i + 2), node(cgraph, i + 3), scale_node, mm_node);
            if (scale == null) continue;

            const bias = if (with_bias) getBiasTensor(out_node, scale_node, bias_op) else null;
            if (with_bias and bias == null) continue;
            if (with_bias and bias_op == ggml.impl.ggml_c.GGML_OP_ADD and !ggml.impl.ggml_c.ggml_are_same_shape(src(out_node, 0), src(out_node, 1))) continue;
            if (with_bias and bias_op == ggml.impl.ggml_c.GGML_OP_ADD_ID and out_node.src[2] != mm_node.src[2]) continue;

            const src0 = src(mm_node, 0);
            const src1 = src(mm_node, 1);
            const ids = srcOpt(mm_node, 2);

            var fusion_data: ggml.cuda.common.MmFusionArgsHost = .{ .x_bias = bias, .x_scale = scale };

            if (ggml.cuda.mulmat.shouldFuseMulMatVecF(mm_node, null)) {
                ggml.cuda.mmvf.mulMatVecF(cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = n_ops;
                break :outer;
            }

            if (op == ggml.impl.ggml_c.GGML_OP_MUL_MAT and !with_bias and src0.type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 and
                ggml.impl.getOpParamsI32(mm_node, 3) != ggml.impl.ggml_c.GGML_PREC_F32 and
                ggml.cuda.fp8.mulMatFp8(cuda_ctx, src0, src1, scale_node, scale))
            {
                return 1;
            }

            if (!with_bias and src0.type == ggml.impl.ggml_c.GGML_TYPE_NVFP4 and src1.type == ggml.impl.ggml_c.GGML_TYPE_F32 and mm_node.type == ggml.impl.ggml_c.GGML_TYPE_F32) {
                const x = ggml.cuda.common.info().devices[@intCast(cuda_ctx.device)].cc;
                const is_id = op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID;
                const ncols: i64 = if (is_id) src1.ne[2] else src1.ne[1];
                const use_mmvq = if (is_id) ncols <= ggml.cuda.mmvq.getMmidMaxBatch(src0.type, x) else ggml.cuda.mmvq.shouldUseMmvq(src0.type, x, ncols);
                const bad_padding_clear = ggml.impl.ggml_c.ggml_backend_buffer_get_usage(src0.buffer) == ggml.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_COMPUTE and
                    ggml.impl.ggml_c.ggml_nbytes(src0) != ggml.impl.ggml_c.ggml_backend_buffer_get_alloc_size(src0.buffer, src0) and src0.view_src != null;
                if (!use_mmvq and !bad_padding_clear and
                    ggml.cuda.mmq.shouldUseMmq(src0.type, x, ncols, if (is_id) src0.ne[2] else 0))
                {
                    fusion_data.prec_src1 = ggml.impl.ggml_c.ggml_prec_get_src(mm_node, 1);
                    ggml.cuda.mmq.mulMatQ(cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                    return n_ops - 1;
                }
            }

            if (ggml.cuda.mulmat.shouldFuseMulMatVecQ(mm_node)) {
                ggml.cuda.mmvq.mulMatVecQ(cuda_ctx, src(mm_node, 0), src(mm_node, 1), srcOpt(mm_node, 2), out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = n_ops;
                break :outer;
            }
        }
    }

    if (fused_mul_mat_vec) return fused_node_count - 1;

    // mul_mat + add
    for ([_]ggml.impl.ggml_c.enum_ggml_op{ ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID }) |op| {
        const bias_op: ggml.impl.ggml_c.enum_ggml_op = if (op == ggml.impl.ggml_c.GGML_OP_MUL_MAT) ggml.impl.ggml_c.GGML_OP_ADD else ggml.impl.ggml_c.GGML_OP_ADD_ID;

        if (!ggml.graph.canFuse(cgraph, i, &.{ op, bias_op })) continue;

        const mm_node = node(cgraph, i);
        const bias_node = node(cgraph, i + 1);

        var bias_tensor: *const ggml.impl.ggml_c.ggml_tensor = undefined;
        if (bias_op == ggml.impl.ggml_c.GGML_OP_ADD) {
            if (bias_node.src[0] == mm_node) {
                bias_tensor = src(bias_node, 1);
            } else if (bias_node.src[1] == mm_node) {
                bias_tensor = src(bias_node, 0);
            } else {
                continue;
            }
        } else {
            if (bias_node.src[0] != mm_node) continue;
            bias_tensor = src(bias_node, 1);
        }

        const ids = srcOpt(mm_node, 2);

        if (bias_op == ggml.impl.ggml_c.GGML_OP_ADD_ID and bias_node.src[2] != mm_node.src[2]) continue;
        if (bias_op == ggml.impl.ggml_c.GGML_OP_ADD and !ggml.impl.ggml_c.ggml_are_same_shape(src(bias_node, 0), src(bias_node, 1))) continue;

        const fusion_data: ggml.cuda.common.MmFusionArgsHost = .{ .x_bias = bias_tensor };

        if (ggml.cuda.mulmat.shouldFuseMulMatVecF(mm_node, null)) {
            ggml.cuda.mmvf.mulMatVecF(cuda_ctx, src(mm_node, 0), src(mm_node, 1), ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count = 2;
            break;
        }

        if (ggml.cuda.mulmat.shouldFuseMulMatVecQ(mm_node)) {
            ggml.cuda.mmvq.mulMatVecQ(cuda_ctx, src(mm_node, 0), src(mm_node, 1), ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) return fused_node_count - 1;

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_RMS_NORM, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_SET_ROWS }, &.{})) {
        ggml.cuda.rope.opRmsNormMulRopeFused(cuda_ctx, nd, node(cgraph, i + 1), node(cgraph, i + 2), node(cgraph, i + 4));
        return 4;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_RMS_NORM, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_ROPE }, &.{})) {
        ggml.cuda.rope.opRmsNormMulRopeFused(cuda_ctx, nd, node(cgraph, i + 1), node(cgraph, i + 2), null);
        return 2;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_RMS_NORM, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_ADD }, &.{})) {
        ggml.cuda.norm.opRmsNormFusedAdd(cuda_ctx, nd, node(cgraph, i + 1), node(cgraph, i + 2));
        return 2;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_RMS_NORM, ggml.impl.ggml_c.GGML_OP_MUL }, &.{})) {
        ggml.cuda.norm.opRmsNormFused(cuda_ctx, nd, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_RMS_NORM, ggml.impl.ggml_c.GGML_OP_SCALE }, &.{})) {
        ggml.cuda.norm.opRmsNormScaleFused(cuda_ctx, nd, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_SSM_CONV, ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_UNARY }, &.{ggml.impl.ggml_c.GGML_UNARY_OP_SILU})) {
        ggml.cuda.ssm_conv.opSsmConv(cuda_ctx, nd, node(cgraph, i + 1), node(cgraph, i + 2));
        return 2;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_SSM_CONV, ggml.impl.ggml_c.GGML_OP_UNARY }, &.{ggml.impl.ggml_c.GGML_UNARY_OP_SILU})) {
        ggml.cuda.ssm_conv.opSsmConv(cuda_ctx, nd, null, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_MUL }, &.{ggml.impl.ggml_c.GGML_UNARY_OP_SILU}) or
        canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_MUL }, &.{ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID}) or
        canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_MUL }, &.{ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS}))
    {
        ggml.cuda.unary.opUnaryMul(cuda_ctx, nd, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_SQR }, &.{ggml.impl.ggml_c.GGML_UNARY_OP_RELU})) {
        ggml.cuda.unary.opReluSqr(cuda_ctx, nd, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_SCALE, ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_SCALE, ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST }, &.{ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID})) {
        ggml.cuda.dsv4_hc.opDsv4HcPost(cuda_ctx, node(cgraph, i + 3), nd, node(cgraph, i + 2));
        return 3;
    }

    if (canFuse(cgraph, i, &.{ ggml.impl.ggml_c.GGML_OP_SCALE, ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_SCALE }, &.{ggml.impl.ggml_c.GGML_UNARY_OP_TANH})) {
        ggml.cuda.softcap.opSoftcap(cuda_ctx, node(cgraph, i + 2), nd);
        return 2;
    }

    return 0;
}
