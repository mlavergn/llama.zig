//! The CUDA backend's fusion pass: recognising runs of graph nodes that one
//! fused kernel can compute, and launching it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v0.3.0
//! (`c1d0e7a00`): `ggml_cuda_should_fuse_rope_set_rows`,
//! `ggml_cuda_should_fuse_rms_norm_mul_rope`, `ggml_cuda_try_gdn_cache_fusion`,
//! `ggml_cuda_topk_moe_fusion`, `ggml_cuda_check_fusion_memory_ranges`,
//! `ggml_cuda_can_fuse` and `ggml_cuda_try_fuse`; and the C++ overloads of
//! `ggml_can_fuse`, `ggml_can_fuse_subgraph` and `ggml_check_edges` from
//! `llama.cpp/ggml/src/ggml-impl.h`.
//!
//! The C's `std::initializer_list<ggml_op>` arguments are slices here.

const std = @import("std");
const impl = @import("../impl.zig");
const ggml_graph = @import("../graph.zig");
const common = @import("common.zig");
const mulmat = @import("mulmat.zig");
const mmvq = @import("mmvq.zig");
const mmvf = @import("mmvf.zig");
const binbcast = @import("binbcast.zig");
const norm = @import("norm.zig");
const rope = @import("rope.zig");
const ssm_conv = @import("ssm_conv.zig");
const unary = @import("unary.zig");
const softcap = @import("softcap.zig");
const snake = @import("snake.zig");
const gated_delta_net = @import("gated_delta_net.zig");
const topk_moe = @import("topk_moe.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const CGraph = impl.CGraph;
const Op = c.enum_ggml_op;
const UnaryOp = c.enum_ggml_unary_op;

fn node(cgraph: *const CGraph, i: c_int) *Tensor {
    return cgraph.nodes[@intCast(i)].?;
}

fn src(t: *const Tensor, i: usize) *Tensor {
    return impl.one(Tensor, t.src[i]);
}

/// Ports the C++ `ggml_can_fuse_subgraph` overload and the static inline it
/// calls (ggml-impl.h:724, 755 @c1d0e7a00).
fn canFuseSubgraph(cgraph: *const CGraph, node_idx: c_int, ops: []const Op, outputs: []const c_int) bool {
    impl.assert(ops.len < 32, "count < 32");
    if (node_idx + @as(c_int, @intCast(ops.len)) > cgraph.n_nodes) return false;

    var idxs: [32]c_int = undefined;
    for (0..ops.len) |i| idxs[i] = node_idx + @as(c_int, @intCast(i));

    return ggml_graph.ggml_can_fuse_subgraph_ext(cgraph, &idxs, @intCast(ops.len), ops.ptr, outputs.ptr, @intCast(outputs.len));
}

/// Ports `ggml_check_edges` (ggml-impl.h:766 @c1d0e7a00): each edge is
/// `{dst_node, src_idx, src_node}`, relative to `start_idx`.
fn checkEdges(cgraph: *const CGraph, start_idx: c_int, edges: []const [3]c_int) bool {
    for (edges) |e| {
        if (node(cgraph, start_idx + e[0]).src[@intCast(e[1])] != node(cgraph, start_idx + e[2])) return false;
    }
    return true;
}

/// Ports `ggml_cuda_should_fuse_rope_set_rows` (ggml-cuda.cu:2656 @c1d0e7a00).
fn shouldFuseRopeSetRows(rope_t: *const Tensor, view: *const Tensor, set_rows: *const Tensor) bool {
    if (rope_t.op != c.GGML_OP_ROPE or view.op != c.GGML_OP_VIEW or set_rows.op != c.GGML_OP_SET_ROWS) return false;
    // ne3 not tested
    if (src(rope_t, 0).ne[3] != 1) return false;
    if (set_rows.type != c.GGML_TYPE_F32 and set_rows.type != c.GGML_TYPE_F16) return false;
    if (src(set_rows, 1).type != c.GGML_TYPE_I64) return false;
    // The view should flatten two dims of rope into one dim
    if (!c.ggml_is_contiguous(view) or view.ne[0] != rope_t.ne[0] * rope_t.ne[1]) return false;
    // Only norm/neox shaders have the fusion code
    const mode = rope_t.op_params[2];
    if (mode != c.GGML_ROPE_TYPE_NORMAL and mode != c.GGML_ROPE_TYPE_NEOX) return false;
    return true;
}

/// Ports `ggml_cuda_should_fuse_rms_norm_mul_rope` (ggml-cuda.cu:2690 @c1d0e7a00).
fn shouldFuseRmsNormMulRope(rms_norm: *const Tensor, mul: *const Tensor, rope_t: *const Tensor) bool {
    if (rms_norm.op != c.GGML_OP_RMS_NORM or mul.op != c.GGML_OP_MUL or rope_t.op != c.GGML_OP_ROPE) return false;

    const F32 = c.GGML_TYPE_F32;
    if (src(rms_norm, 0).type != F32 or rms_norm.type != F32 or
        src(mul, 0).type != F32 or src(mul, 1).type != F32 or
        mul.type != F32 or rope_t.type != F32) return false;

    if (rope_t.src[0] != mul) return false;

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul.src[1] and !c.ggml_are_same_shape(src(mul, 0), rms_norm)) return false;

    if (!c.ggml_are_same_shape(rms_norm, mul)) return false;

    //rms_norm kernel assumes contiguous rows
    if (!c.ggml_is_contiguous_rows(src(rms_norm, 0)) or
        !c.ggml_is_contiguous_rows(src(mul, 0)) or !c.ggml_is_contiguous_rows(src(mul, 1))) return false;

    // the fused kernel handles the norm/neox rope modes only
    const mode = rope_t.op_params[2];
    if (mode != c.GGML_ROPE_TYPE_NORMAL and mode != c.GGML_ROPE_TYPE_NEOX) return false;

    const n_dims = rope_t.op_params[1];
    if (@rem(n_dims, 2) != 0 or @rem(src(rope_t, 0).ne[0], 2) != 0) return false;

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const n_offs = rope_t.op_params[15];
    if (n_offs != 0) return false;

    return true;
}

/// Ports `ggml_cuda_is_view_or_noop` (ggml-cuda.cu:2542 @c1d0e7a00), for the
/// gated-delta-net match; `graph.zig` has its own copy for the compute loop.
fn isViewOrNoop(t: *const Tensor) bool {
    return c.ggml_is_empty(t) or t.op == c.GGML_OP_RESHAPE or t.op == c.GGML_OP_TRANSPOSE or
        t.op == c.GGML_OP_VIEW or t.op == c.GGML_OP_PERMUTE or t.op == c.GGML_OP_NONE;
}

/// Ports `ggml_cuda_try_gdn_cache_fusion` (ggml-cuda.cu:2744 @c1d0e7a00):
/// match gated_delta_net + the strided cpy that scatters its state
/// snapshots into the cache, so the kernel can write them and skip the cpy.
fn tryGdnCacheFusion(cgraph: *const CGraph, node_idx: c_int, fused_state_cpy: *gated_delta_net.FusedCache) c_int {
    const gdn = node(cgraph, node_idx);
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn.op != c.GGML_OP_GATED_DELTA_NET or gdn.type != c.GGML_TYPE_F32 or (gdn.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0) return 0;

    const src_v = src(gdn, 2);
    const S_v = src_v.ne[0];
    const H = src_v.ne[1];
    const n_tokens = src_v.ne[2];
    const n_seqs = src_v.ne[3];
    const D = S_v * S_v * H;
    const K: i64 = impl.getOpParamsI32(gdn, 0); // snapshot slot count
    const n_written = @min(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const tail_off = c.ggml_row_size(c.GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    var cpy: ?*const Tensor = null;
    var skip: c_int = 0;
    var j = node_idx + 1;
    while (j < cgraph.n_nodes and cpy == null) : (j += 1) {
        const n = node(cgraph, j);
        if (isViewOrNoop(n)) continue;
        if (n.op != c.GGML_OP_CPY or (n.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0) return 0;
        cpy = n;
        skip = j - node_idx;
    }
    const cp = cpy orelse return 0;

    const s = src(cp, 0); // view of the gdn snapshot tail
    const d = src(cp, 1); // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (s.op != c.GGML_OP_VIEW or s.view_src != gdn or s.view_offs != tail_off or !c.ggml_is_contiguous(s)) return 0;

    // dst is the [D, n_seqs, n_written] cache view
    const expected_ne = [4]i64{ D, n_seqs, n_written, 1 };
    if (d.op != c.GGML_OP_VIEW or d.type != c.GGML_TYPE_F32 or d.data == null or
        !std.mem.eql(i64, &expected_ne, &d.ne) or
        d.nb[0] != c.ggml_type_size(c.GGML_TYPE_F32) or d.nb[1] != c.ggml_row_size(c.GGML_TYPE_F32, D)) return 0;

    fused_state_cpy.data = @ptrCast(@alignCast(d.data)); // rollback group 0 (newest)
    fused_state_cpy.slot_stride = if (K > 1) @intCast(d.nb[2] / @sizeOf(f32)) else 0;
    return skip;
}

/// Ports `ggml_cuda_topk_moe_fusion` (ggml-cuda.cu:2806 @c1d0e7a00).
fn topkMoeFusion(cgraph: *const CGraph, node_idx_in: c_int, args: *topk_moe.Args) bool {
    args.sigmoid = false;
    args.sqrt_softplus = false;
    args.softmax = false;
    args.delayed_softmax = false;
    args.prob_bias = false;
    args.norm = false;

    const n_nodes = cgraph.n_nodes;
    var node_idx = node_idx_in;

    if (node(cgraph, node_idx).op == c.GGML_OP_SOFT_MAX) args.softmax = true;

    if (node(cgraph, node_idx).op == c.GGML_OP_UNARY) {
        const unary_op = c.ggml_get_unary_op(node(cgraph, node_idx));
        if (unary_op == c.GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == c.GGML_UNARY_OP_SOFTPLUS and node_idx + 1 < n_nodes and
            node(cgraph, node_idx + 1).op == c.GGML_OP_SQRT and node(cgraph, node_idx + 1).src[0] == node(cgraph, node_idx))
        {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx += 1;
        } else {
            return false;
        }
    }

    if (node(cgraph, node_idx).op == c.GGML_OP_ARGSORT) args.delayed_softmax = true;

    node_idx += 1;

    if (args.sigmoid or args.sqrt_softplus or args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes or node(cgraph, node_idx).op != c.GGML_OP_RESHAPE or
            node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) return false;
        const probs_reshaped = node(cgraph, node_idx);
        node_idx += 1;

        if (node_idx >= n_nodes) return false;

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (node(cgraph, node_idx).op == c.GGML_OP_ADD and node(cgraph, node_idx).src[0] == node(cgraph, node_idx - 2)) {
            args.prob_bias = true;
            node_idx += 1;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes or node(cgraph, node_idx).op != c.GGML_OP_ARGSORT) return false;

        if (args.prob_bias and node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) {
            return false;
        } else if (!args.prob_bias and node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 2)) {
            return false;
        }

        node_idx += 1;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes or node(cgraph, node_idx).op != c.GGML_OP_VIEW or
            node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) return false;
        node_idx += 1;

        if (node_idx >= n_nodes or node(cgraph, node_idx).op != c.GGML_OP_GET_ROWS) return false;

        // GET_ROWS
        if (node(cgraph, node_idx).src[0] != probs_reshaped or node(cgraph, node_idx).src[1] != node(cgraph, node_idx - 1)) return false;
        node_idx += 1;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) return false;
        const probs_reshaped = node(cgraph, node_idx - 2);

        // VIEW->ARGSORT
        if (node_idx >= n_nodes or node(cgraph, node_idx).op != c.GGML_OP_VIEW or
            node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) return false;
        node_idx += 1;

        // GET_ROWS
        if (node_idx >= n_nodes or node(cgraph, node_idx).src[1] != node(cgraph, node_idx - 1) or
            node(cgraph, node_idx).src[0] != probs_reshaped) return false;
        node_idx += 1;

        const remaining_ops = [_]Op{ c.GGML_OP_RESHAPE, c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE };
        for (remaining_ops) |op| {
            if (node_idx >= n_nodes or node(cgraph, node_idx).op != op or node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) return false;
            node_idx += 1;
        }
    }

    // At this point we can check for norm + scale. Everything is now at
    // least valid till the norm
    if (node_idx >= n_nodes) return true;

    if (node(cgraph, node_idx).op == c.GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        const norm_ops = [_]Op{ c.GGML_OP_RESHAPE, c.GGML_OP_SUM_ROWS, c.GGML_OP_CLAMP };

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
        if (node(cgraph, node_idx).op != c.GGML_OP_DIV or node(cgraph, node_idx).src[1] != node(cgraph, node_idx - 1) or
            node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 3))
        {
            args.norm = false;
            return true;
        }
        node_idx += 1;

        if (node(cgraph, node_idx).op != c.GGML_OP_RESHAPE or node(cgraph, node_idx).src[0] != node(cgraph, node_idx - 1)) {
            args.norm = false;
            return true;
        }

        node_idx += 1;
    }

    if (node(cgraph, node_idx).op == c.GGML_OP_SCALE and node(cgraph, node_idx).src[0] == node(cgraph, node_idx - 1)) {
        args.scale = true;
    }

    return true;
}

/// Ports `ggml_cuda_check_fusion_memory_ranges` (ggml-cuda.cu:2961 @c1d0e7a00):
/// whether the write (out) nodes overwrite the read nodes in operation.
fn checkFusionMemoryRanges(cgraph: *const CGraph, node_idx: c_int, node_count: c_int, out_nodes: []const c_int, is_topk_moe: bool) bool {
    const S = struct {
        fn overlap(a: *const Tensor, b: *const Tensor) bool {
            const a_start: i64 = @intCast(@intFromPtr(a.data));
            const a_end = a_start + @as(i64, @intCast(c.ggml_backend_buft_get_alloc_size(impl.one(c.struct_ggml_backend_buffer, a.buffer).buft, a)));
            const b_start: i64 = @intCast(@intFromPtr(b.data));
            const b_end = b_start + @as(i64, @intCast(c.ggml_backend_buft_get_alloc_size(impl.one(c.struct_ggml_backend_buffer, b.buffer).buft, b)));
            return (b_start <= a_start and a_start < b_end) or (a_start <= b_start and b_start < a_end);
        }
    };

    var is_ok = true;
    // exception for topk-moe, as each row is read entirely before writing
    if (c.ggml_nrows(node(cgraph, node_idx)) == 1 and is_topk_moe) return true;

    for (out_nodes) |out| {
        const dst = node(cgraph, out);

        var j = node_idx;
        while (j < node_idx + node_count) : (j += 1) {
            // Loop over all srcs of all nodes in the fusion. If the src
            // overlaps the destination and the src is not an intermediate
            // node that's being elided, then disable fusion.
            for (node(cgraph, j).src) |s_opt| {
                const s = s_opt orelse continue;
                if (s.*.op == c.GGML_OP_NONE) continue;

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

fn opsEqual(a: []const Op, b: []const Op) bool {
    return std.mem.eql(Op, a, b);
}

/// Ports `ggml_cuda_can_fuse` (ggml-cuda.cu:3025 @c1d0e7a00). The C's
/// `NDEBUG`-only assertion on `unary_ops.size()` compiles out.
fn canFuse(cgraph: *const CGraph, node_idx: c_int, ops: []const Op, unary_ops: []const UnaryOp) bool {
    const mul_mat_bias_glu_ops = [_]Op{ c.GGML_OP_MUL_MAT, c.GGML_OP_ADD, c.GGML_OP_MUL_MAT, c.GGML_OP_ADD, c.GGML_OP_GLU };
    const mul_mat_id_bias_glu_ops = [_]Op{ c.GGML_OP_MUL_MAT_ID, c.GGML_OP_ADD_ID, c.GGML_OP_MUL_MAT_ID, c.GGML_OP_ADD_ID, c.GGML_OP_GLU };

    const mul_mat_id_glu_ops = [_]Op{ c.GGML_OP_MUL_MAT_ID, c.GGML_OP_MUL_MAT_ID, c.GGML_OP_GLU };
    const mul_mat_glu_ops = [_]Op{ c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT, c.GGML_OP_GLU };

    const n: c_int = @intCast(ops.len);

    if ((opsEqual(&mul_mat_bias_glu_ops, ops) or opsEqual(&mul_mat_id_bias_glu_ops, ops)) and
        canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 4}))
    {
        const ffn_gate = node(cgraph, node_idx);
        const ffn_gate_bias = node(cgraph, node_idx + 1);
        const ffn_up = node(cgraph, node_idx + 2);
        const ffn_up_bias = node(cgraph, node_idx + 3);
        const glu = node(cgraph, node_idx + 4);

        if (mulmat.shouldFuseMulMat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias, null, null)) {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 4}, false);
        }
    }

    if ((opsEqual(&mul_mat_id_glu_ops, ops) or opsEqual(&mul_mat_glu_ops, ops)) and
        canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 2}))
    {
        const ffn_gate = node(cgraph, node_idx);
        const ffn_up = node(cgraph, node_idx + 1);
        const glu = node(cgraph, node_idx + 2);

        if (mulmat.shouldFuseMulMat(ffn_up, ffn_gate, glu, null, null, null, null)) {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 2}, false);
        }
    }

    const rms_norm_mul_rope_ops = [_]Op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ROPE };
    const rms_norm_mul_rope_set_rows_ops = [_]Op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ROPE, c.GGML_OP_VIEW, c.GGML_OP_SET_ROWS };

    if (opsEqual(&rms_norm_mul_rope_set_rows_ops, ops) and canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 4})) {
        const rms_norm = node(cgraph, node_idx);
        const mul = node(cgraph, node_idx + 1);
        const rope_t = node(cgraph, node_idx + 2);
        const view = node(cgraph, node_idx + 3);
        const set_rows = node(cgraph, node_idx + 4);

        if (checkEdges(cgraph, node_idx, &.{ .{ 1, 0, 0 }, .{ 2, 0, 1 }, .{ 3, 0, 2 }, .{ 4, 0, 3 } }) and
            shouldFuseRmsNormMulRope(rms_norm, mul, rope_t) and
            shouldFuseRopeSetRows(rope_t, view, set_rows))
        {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 4}, false);
        }
    }

    if (opsEqual(&rms_norm_mul_rope_ops, ops) and ggml_graph.canFuse(cgraph, node_idx, ops)) {
        const rms_norm = node(cgraph, node_idx);
        const mul = node(cgraph, node_idx + 1);
        const rope_t = node(cgraph, node_idx + 2);

        if (shouldFuseRmsNormMulRope(rms_norm, mul, rope_t)) {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 2}, false);
        }
        return false;
    }

    const rope_set_rows_ops = [_]Op{ c.GGML_OP_ROPE, c.GGML_OP_VIEW, c.GGML_OP_SET_ROWS };

    if (opsEqual(&rope_set_rows_ops, ops) and canFuseSubgraph(cgraph, node_idx, ops, &.{node_idx + 2})) {
        const rope_t = node(cgraph, node_idx);
        const view = node(cgraph, node_idx + 1);
        const set_rows = node(cgraph, node_idx + 2);

        if (shouldFuseRopeSetRows(rope_t, view, set_rows)) {
            return checkFusionMemoryRanges(cgraph, node_idx, n, &.{node_idx + 2}, false);
        }
    }

    if (!ggml_graph.canFuse(cgraph, node_idx, ops)) return false;

    if ((ops.len == 2 or ops.len == 3) and ops[0] == c.GGML_OP_RMS_NORM and ops[1] == c.GGML_OP_MUL) {
        const rms_norm = node(cgraph, node_idx);
        const mul = node(cgraph, node_idx + 1);
        var add: ?*const Tensor = null;

        if (ops.len == 3 and ops[2] == c.GGML_OP_ADD) add = node(cgraph, node_idx + 2);

        impl.assert(src(rms_norm, 0).type == c.GGML_TYPE_F32, "rms_norm->src[0]->type == GGML_TYPE_F32");
        impl.assert(rms_norm.type == c.GGML_TYPE_F32, "rms_norm->type == GGML_TYPE_F32");

        //rms norm only supports F32
        if (src(mul, 0).type != c.GGML_TYPE_F32 or src(mul, 1).type != c.GGML_TYPE_F32 or mul.type != c.GGML_TYPE_F32) return false;

        if (add) |a| {
            if (src(a, 0).type != c.GGML_TYPE_F32 or src(a, 1).type != c.GGML_TYPE_F32 or a.type != c.GGML_TYPE_F32) return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul.src[1] and !c.ggml_are_same_shape(src(mul, 0), rms_norm)) return false;

        //rms_norm kernel assumes contiguous rows
        if (!c.ggml_is_contiguous_rows(src(mul, 0)) or !c.ggml_is_contiguous_rows(src(mul, 1))) return false;

        if (add) |a| {
            if (!c.ggml_is_contiguous(src(a, 0)) or !c.ggml_is_contiguous_rows(src(a, 1))) return false;
        }

        return true;
    }

    if (ops.len == 2 and ops[0] == c.GGML_OP_SSM_CONV and ops[1] == c.GGML_OP_UNARY and
        unary_ops.len == 1 and unary_ops[0] == c.GGML_UNARY_OP_SILU)
    {
        const ssm_conv_t = node(cgraph, node_idx);
        const silu = node(cgraph, node_idx + 1);
        if (c.ggml_get_unary_op(silu) != unary_ops[0]) return false;
        if (ssm_conv_t.type != c.GGML_TYPE_F32 or silu.type != c.GGML_TYPE_F32) return false;
        return true;
    }

    if (ops.len == 3 and ops[0] == c.GGML_OP_SSM_CONV and ops[1] == c.GGML_OP_ADD and
        ops[2] == c.GGML_OP_UNARY and unary_ops.len == 1 and unary_ops[0] == c.GGML_UNARY_OP_SILU)
    {
        const ssm_conv_t = node(cgraph, node_idx);
        const add = node(cgraph, node_idx + 1);
        const silu = node(cgraph, node_idx + 2);
        if (c.ggml_get_unary_op(silu) != unary_ops[0]) return false;

        if (ssm_conv_t.type != c.GGML_TYPE_F32 or add.type != c.GGML_TYPE_F32 or silu.type != c.GGML_TYPE_F32) return false;

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const bias = if (add.src[0] == ssm_conv_t) src(add, 1) else src(add, 0);
        if (bias.type != c.GGML_TYPE_F32 or !c.ggml_is_contiguous(bias)) return false;
        if (c.ggml_nelements(bias) != ssm_conv_t.ne[0] or bias.ne[0] != ssm_conv_t.ne[0]) return false;

        return true;
    }

    if (ops.len == 2 and ops[0] == c.GGML_OP_UNARY and ops[1] == c.GGML_OP_MUL and unary_ops.len == 1 and
        (unary_ops[0] == c.GGML_UNARY_OP_SILU or unary_ops[0] == c.GGML_UNARY_OP_SIGMOID or unary_ops[0] == c.GGML_UNARY_OP_SOFTPLUS))
    {
        const un = node(cgraph, node_idx);
        const mul = node(cgraph, node_idx + 1);

        if (c.ggml_get_unary_op(un) != unary_ops[0]) return false;
        if (un.type != c.GGML_TYPE_F32 and un.type != c.GGML_TYPE_F16) return false;
        if (un.type != mul.type) return false;

        const other = if (mul.src[0] == un) src(mul, 1) else src(mul, 0);
        if (other.type != un.type) return false;
        if (!c.ggml_is_contiguous_1(other) or !c.ggml_is_contiguous_1(src(un, 0)) or !c.ggml_are_same_shape(other, un)) return false;

        return true;
    }

    if (ops.len == 2 and ops[0] == c.GGML_OP_UNARY and ops[1] == c.GGML_OP_SQR and
        unary_ops.len == 1 and unary_ops[0] == c.GGML_UNARY_OP_RELU)
    {
        const un = node(cgraph, node_idx);
        const sqr = node(cgraph, node_idx + 1);

        if (c.ggml_get_unary_op(un) != c.GGML_UNARY_OP_RELU) return false;
        if (un.type != c.GGML_TYPE_F32 and un.type != c.GGML_TYPE_F16) return false;
        if (un.type != sqr.type) return false;
        if (!c.ggml_is_contiguous(src(un, 0))) return false;

        return true;
    }

    if (ops.len == 3 and ops[0] == c.GGML_OP_SCALE and ops[1] == c.GGML_OP_UNARY and ops[2] == c.GGML_OP_SCALE and
        unary_ops.len == 1 and unary_ops[0] == c.GGML_UNARY_OP_TANH)
    {
        const scale = node(cgraph, node_idx);
        const tanh = node(cgraph, node_idx + 1);
        const scale2 = node(cgraph, node_idx + 2);

        impl.assert(src(scale, 0).type == c.GGML_TYPE_F32, "scale->src[0]->type == GGML_TYPE_F32");
        impl.assert(scale.type == c.GGML_TYPE_F32, "scale->type == GGML_TYPE_F32");

        if (c.ggml_get_unary_op(tanh) != c.GGML_UNARY_OP_TANH) return false;

        // Check for bias
        if (impl.getOpParamsF32(scale, 1) != 0.0 or impl.getOpParamsF32(scale2, 1) != 0.0) return false;

        return true;
    }

    return false;
}

/// The C's `get_mul_mat_scale` lambda in `ggml_cuda_try_fuse`.
fn getMulMatScale(scale_node: *const Tensor, mm_node: *const Tensor) ?*const Tensor {
    const scale_lhs_mm = scale_node.src[0] == mm_node;
    const scale_rhs_mm = scale_node.src[1] == mm_node;
    if (!scale_lhs_mm and !scale_rhs_mm) return null;

    const scale = if (scale_lhs_mm) src(scale_node, 1) else src(scale_node, 0);
    if (src(mm_node, 0).type != c.GGML_TYPE_NVFP4 or scale_node.type != c.GGML_TYPE_F32 or
        scale.type != c.GGML_TYPE_F32 or !c.ggml_is_contiguous(scale) or c.ggml_nelements(scale) != 1 or
        !c.ggml_are_same_shape(scale_node, mm_node)) return null;

    return scale;
}

/// The C's `get_mul_mat_id_scale` lambda in `ggml_cuda_try_fuse`.
fn getMulMatIdScale(reshape: *const Tensor, repeat: *const Tensor, getrows: *const Tensor, scale_node: *const Tensor, mm_node: *const Tensor) ?*const Tensor {
    if (repeat.src[0] != reshape or getrows.src[0] != repeat or getrows.src[1] != mm_node.src[2]) return null;
    if (!((scale_node.src[0] == mm_node and scale_node.src[1] == getrows) or
        (scale_node.src[0] == getrows and scale_node.src[1] == mm_node))) return null;

    const scale = src(reshape, 0);
    if (src(mm_node, 0).type != c.GGML_TYPE_NVFP4 or scale_node.type != c.GGML_TYPE_F32 or
        scale.type != c.GGML_TYPE_F32 or !c.ggml_is_contiguous(scale) or c.ggml_nelements(scale) != src(mm_node, 0).ne[2] or
        !c.ggml_are_same_shape(scale_node, mm_node)) return null;

    return scale;
}

/// The C's `get_bias_tensor` lambda in `ggml_cuda_try_fuse`.
fn getBiasTensor(bias_node: *const Tensor, mul_node: *const Tensor, op_bias: Op) ?*const Tensor {
    if (op_bias == c.GGML_OP_ADD) {
        if (bias_node.src[0] == mul_node) return src(bias_node, 1);
        if (bias_node.src[1] == mul_node) return src(bias_node, 0);
        return null;
    }
    impl.assert(op_bias == c.GGML_OP_ADD_ID, "op_bias == GGML_OP_ADD_ID");
    impl.assert(bias_node.src[0] == mul_node, "bias_node->src[0] == mul_node");
    return src(bias_node, 1);
}

const DisableFusion = struct {
    var value: bool = false;
    fn run() void {
        if (std.c.getenv("GGML_CUDA_DISABLE_FUSION")) |e| value = common.atoi(e) != 0;
    }
    const once = common.Once(run);
};

fn srcOpt(t: *const Tensor, i: usize) ?*const Tensor {
    return if (t.src[i]) |s| s else null;
}

/// Ports `ggml_cuda_try_fuse` (ggml-cuda.cu:3277 @c1d0e7a00): try and fuse
/// nodes, and return the number of nodes to skip.
pub fn tryFuse(cuda_ctx: *common.Context, cgraph: *CGraph, i: c_int) c_int {
    DisableFusion.once.call();
    if (DisableFusion.value) return 0;

    const nd = node(cgraph, i);

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (nd.op == c.GGML_OP_GATED_DELTA_NET) {
        var fused_state_cpy: gated_delta_net.FusedCache = undefined;
        const nodes_to_skip = tryGdnCacheFusion(cgraph, i, &fused_state_cpy);
        if (nodes_to_skip > 0) {
            gated_delta_net.opGatedDeltaNetFusedCache(cuda_ctx, nd, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (nd.op == c.GGML_OP_UNARY or nd.op == c.GGML_OP_SOFT_MAX or nd.op == c.GGML_OP_ARGSORT) {
        var args: topk_moe.Args = .{};
        const can_fuse = topkMoeFusion(cgraph, i, &args);
        var ops_buf: [16]Op = undefined;
        var n_ops: usize = 0;
        const Append = struct {
            fn f(buf: *[16]Op, n: *usize, items: []const Op) void {
                for (items) |x| {
                    buf[n.*] = x;
                    n.* += 1;
                }
            }
        };

        if (can_fuse) {
            const logits = src(nd, 0);
            var weights: *Tensor = undefined;
            var ids: *Tensor = undefined;
            var bias: ?*const Tensor = null;
            var clamp: ?*const Tensor = null;
            var scale: ?*const Tensor = null;

            if (!args.delayed_softmax) {
                var out_nodes: [2]c_int = undefined; // nodes which can't be elided

                if (args.sigmoid) {
                    Append.f(&ops_buf, &n_ops, &.{c.GGML_OP_UNARY});
                } else if (args.sqrt_softplus) {
                    Append.f(&ops_buf, &n_ops, &.{ c.GGML_OP_UNARY, c.GGML_OP_SQRT });
                } else {
                    Append.f(&ops_buf, &n_ops, &.{c.GGML_OP_SOFT_MAX});
                }
                const i_probs = i + @as(c_int, @intCast(n_ops)) - 1; // last node of the gating activation

                if (args.prob_bias) {
                    bias = srcOpt(node(cgraph, i_probs + 2), 1);
                    Append.f(&ops_buf, &n_ops, &.{ c.GGML_OP_RESHAPE, c.GGML_OP_ADD, c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    Append.f(&ops_buf, &n_ops, &.{ c.GGML_OP_RESHAPE, c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = node(cgraph, out_nodes[0]);

                if (args.norm) {
                    Append.f(&ops_buf, &n_ops, &.{ c.GGML_OP_RESHAPE, c.GGML_OP_SUM_ROWS, c.GGML_OP_CLAMP, c.GGML_OP_DIV, c.GGML_OP_RESHAPE });
                    clamp = node(cgraph, i + @as(c_int, @intCast(n_ops)) - 3);
                }
                if (args.scale) {
                    Append.f(&ops_buf, &n_ops, &.{c.GGML_OP_SCALE});
                    scale = node(cgraph, i + @as(c_int, @intCast(n_ops)) - 1);
                }

                weights = node(cgraph, i + @as(c_int, @intCast(n_ops)) - 1);
                out_nodes[1] = i + @as(c_int, @intCast(n_ops)) - 1;

                if (canFuseSubgraph(cgraph, i, ops_buf[0..n_ops], &out_nodes) and
                    topk_moe.shouldUseTopkMoe(nd, logits, weights, ids) and
                    checkFusionMemoryRanges(cgraph, i, @intCast(n_ops), &out_nodes, true))
                {
                    topk_moe.opTopkMoe(cuda_ctx, logits, weights, ids, clamp, scale, bias, &args);
                    return @as(c_int, @intCast(n_ops)) - 1;
                }
            } else if (!args.norm and !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                Append.f(&ops_buf, &n_ops, &.{ c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS, c.GGML_OP_RESHAPE, c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE });
                weights = node(cgraph, i + 5);
                ids = node(cgraph, i + 1);
                const softmax = node(cgraph, i + 4);

                const out_nodes = [2]c_int{ i + 1, i + 5 };
                if (canFuseSubgraph(cgraph, i, ops_buf[0..n_ops], &out_nodes) and
                    topk_moe.shouldUseTopkMoe(softmax, logits, weights, ids) and
                    checkFusionMemoryRanges(cgraph, i, @intCast(n_ops), &out_nodes, true))
                {
                    topk_moe.opTopkMoe(cuda_ctx, logits, weights, ids, clamp, scale, bias, &args);
                    return @as(c_int, @intCast(n_ops)) - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (canFuse(cgraph, i, &.{ c.GGML_OP_ROPE, c.GGML_OP_VIEW, c.GGML_OP_SET_ROWS }, &.{})) {
        rope.opRopeFused(cuda_ctx, node(cgraph, i), node(cgraph, i + 2));
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (canFuseSubgraph(cgraph, i, &.{ c.GGML_OP_MUL, c.GGML_OP_SIN, c.GGML_OP_SQR, c.GGML_OP_MUL, c.GGML_OP_ADD }, &.{i + 4})) {
        const mul0 = node(cgraph, i);
        const sqr = node(cgraph, i + 2);
        const mul1 = node(cgraph, i + 3);
        const add = node(cgraph, i + 4);

        // x carries the full activation shape, a is the broadcast operand
        const x = if (c.ggml_are_same_shape(mul0, src(mul0, 0))) src(mul0, 0) else src(mul0, 1);
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
        const shape_ok = c.ggml_are_same_shape(a, inv_b) and a.ne[0] == 1 and a.ne[1] == x.ne[1];

        const sin1 = node(cgraph, i + 1);
        const types_ok = (x.type == c.GGML_TYPE_F32 or x.type == c.GGML_TYPE_F16 or x.type == c.GGML_TYPE_BF16) and
            (a.type == c.GGML_TYPE_F32) and (inv_b.type == c.GGML_TYPE_F32) and
            (mul0.type == x.type) and (sin1.type == x.type) and
            (sqr.type == x.type) and (mul1.type == x.type) and
            (add.type == x.type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every
        // operand is contiguous
        const contig_ok = c.ggml_is_contiguous(x) and c.ggml_is_contiguous(add) and
            c.ggml_is_contiguous(a) and c.ggml_is_contiguous(inv_b);

        if (types_ok and shape_ok and dim_ok and contig_ok and x_in_add == x) {
            snake.opSnakeFused(cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (nd.op == c.GGML_OP_ADD or nd.op == c.GGML_OP_MUL) {
        var n_fuse: c_int = 0;
        var ops: [8]Op = @splat(nd.op);

        while (n_fuse <= 6) : (n_fuse += 1) {
            if (!ggml_graph.canFuse(cgraph, i + n_fuse, ops[@intCast(n_fuse)..][0..2])) break;
            if (node(cgraph, i + n_fuse) != node(cgraph, i + n_fuse + 1).src[0]) break;
            if (!impl.areSameLayout(src(node(cgraph, i + n_fuse), 1), src(node(cgraph, i + n_fuse + 1), 1))) break;
        }
        _ = &ops;

        n_fuse += 1;

        if (n_fuse > 1) {
            var fused_node: Tensor = nd.*;
            var j: c_int = 0;
            while (j < n_fuse - 1) : (j += 1) {
                fused_node.src[@intCast(j + 2)] = node(cgraph, i + j + 1).src[1];
            }
            fused_node.data = node(cgraph, i + n_fuse - 1).data;
            if (nd.op == c.GGML_OP_ADD) {
                binbcast.opFused(.add, cuda_ctx, &fused_node, n_fuse);
            } else {
                binbcast.opFused(.mul, cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    var fused_mul_mat_vec = false;
    var fused_node_count: c_int = 0;

    // gate + glu + up, with optional scale/bias on both lanes.
    for ([_]Op{ c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT_ID }) |op| {
        const bias_op: Op = if (op == c.GGML_OP_MUL_MAT) c.GGML_OP_ADD else c.GGML_OP_ADD_ID;

        if (op == c.GGML_OP_MUL_MAT) {
            for ([_]bool{ false, true }) |with_bias| {
                const gate_idx = i;
                const gate_scale_idx = i + 1;
                const gate_bias_idx: c_int = if (with_bias) i + 2 else -1;
                const up_idx: c_int = if (with_bias) i + 3 else i + 2;
                const up_scale_idx = up_idx + 1;
                const up_bias_idx: c_int = if (with_bias) up_idx + 2 else -1;
                const glu_idx: c_int = if (with_bias) up_idx + 3 else up_idx + 2;

                const out_nodes = [_]c_int{glu_idx};
                const ops: []const Op = if (with_bias)
                    &.{ op, c.GGML_OP_MUL, bias_op, op, c.GGML_OP_MUL, bias_op, c.GGML_OP_GLU }
                else
                    &.{ op, c.GGML_OP_MUL, op, c.GGML_OP_MUL, c.GGML_OP_GLU };
                const n_ops: c_int = @intCast(ops.len);

                if (!canFuseSubgraph(cgraph, i, ops, &out_nodes) or
                    !checkFusionMemoryRanges(cgraph, i, n_ops, &out_nodes, false)) continue;

                const gate_n = node(cgraph, gate_idx);
                const gate_scale_n = node(cgraph, gate_scale_idx);
                const gate_out_n = if (with_bias) node(cgraph, gate_bias_idx) else gate_scale_n;
                const up_n = node(cgraph, up_idx);
                const up_scale_n = node(cgraph, up_scale_idx);
                const up_out_n = if (with_bias) node(cgraph, up_bias_idx) else up_scale_n;
                const glu = node(cgraph, glu_idx);

                if (!mulmat.shouldFuseMulMat(up_n, gate_n, glu, if (with_bias) up_out_n else null, if (with_bias) gate_out_n else null, up_scale_n, gate_scale_n)) continue;

                const gate_scale = getMulMatScale(gate_scale_n, gate_n);
                const up_scale = getMulMatScale(up_scale_n, up_n);
                if (gate_scale == null or up_scale == null) continue;

                const up_bias = if (with_bias) getBiasTensor(up_out_n, up_scale_n, bias_op) else null;
                const gate_bias = if (with_bias) getBiasTensor(gate_out_n, gate_scale_n, bias_op) else null;
                if (with_bias and (!c.ggml_are_same_shape(src(gate_out_n, 0), src(gate_out_n, 1)) or
                    !c.ggml_are_same_shape(src(up_out_n, 0), src(up_out_n, 1)))) continue;

                const fusion_data: common.MmFusionArgsHost = .{
                    .gate = src(gate_n, 0),
                    .x_bias = up_bias,
                    .gate_bias = gate_bias,
                    .x_scale = up_scale,
                    .gate_scale = gate_scale,
                    .glu_op = c.ggml_get_glu_op(glu),
                };

                if (mulmat.shouldFuseMulMatVecQ(up_n)) {
                    mmvq.mulMatVecQ(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), node(cgraph, glu_idx), &fusion_data);
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
                const ops: []const Op = if (with_bias)
                    &.{ op, c.GGML_OP_RESHAPE, c.GGML_OP_REPEAT, c.GGML_OP_GET_ROWS, c.GGML_OP_MUL, bias_op, op, c.GGML_OP_RESHAPE, c.GGML_OP_REPEAT, c.GGML_OP_GET_ROWS, c.GGML_OP_MUL, bias_op, c.GGML_OP_GLU }
                else
                    &.{ op, c.GGML_OP_RESHAPE, c.GGML_OP_REPEAT, c.GGML_OP_GET_ROWS, c.GGML_OP_MUL, op, c.GGML_OP_RESHAPE, c.GGML_OP_REPEAT, c.GGML_OP_GET_ROWS, c.GGML_OP_MUL, c.GGML_OP_GLU };
                const n_ops: c_int = @intCast(ops.len);

                if (!canFuseSubgraph(cgraph, i, ops, &out_nodes) or
                    !checkFusionMemoryRanges(cgraph, i, n_ops, &out_nodes, false)) continue;

                const gate_n = node(cgraph, gate_idx);
                const gate_scale_n = node(cgraph, gate_scale_idx);
                const gate_out_n = if (with_bias) node(cgraph, gate_bias_idx) else gate_scale_n;
                const up_n = node(cgraph, up_idx);
                const up_scale_n = node(cgraph, up_scale_idx);
                const up_out_n = if (with_bias) node(cgraph, up_bias_idx) else up_scale_n;
                const glu = node(cgraph, glu_idx);

                if (!mulmat.shouldFuseMulMat(up_n, gate_n, glu, if (with_bias) up_out_n else null, if (with_bias) gate_out_n else null, up_scale_n, gate_scale_n)) continue;

                const gate_scale = getMulMatIdScale(node(cgraph, gate_idx + 1), node(cgraph, gate_idx + 2), node(cgraph, gate_idx + 3), gate_scale_n, gate_n);
                const up_scale = getMulMatIdScale(node(cgraph, up_idx + 1), node(cgraph, up_idx + 2), node(cgraph, up_idx + 3), up_scale_n, up_n);
                if (gate_scale == null or up_scale == null) continue;

                const up_bias = if (with_bias) getBiasTensor(up_out_n, up_scale_n, bias_op) else null;
                const gate_bias = if (with_bias) getBiasTensor(gate_out_n, gate_scale_n, bias_op) else null;

                const fusion_data: common.MmFusionArgsHost = .{
                    .gate = src(gate_n, 0),
                    .x_bias = up_bias,
                    .gate_bias = gate_bias,
                    .x_scale = up_scale,
                    .gate_scale = gate_scale,
                    .glu_op = c.ggml_get_glu_op(glu),
                };

                if (mulmat.shouldFuseMulMatVecQ(up_n)) {
                    mmvq.mulMatVecQ(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), node(cgraph, glu_idx), &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) break;
        }

        if (canFuse(cgraph, i, &.{ op, bias_op, op, bias_op, c.GGML_OP_GLU }, &.{})) {
            const glu = node(cgraph, i + 4);
            const gate_bias_n = src(glu, 0);
            const up_bias_n = src(glu, 1);

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            var gate_n: *Tensor = undefined;
            var up_n: *Tensor = undefined;

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
            if (bias_op == c.GGML_OP_ADD and (!c.ggml_are_same_shape(src(gate_bias_n, 0), src(gate_bias_n, 1)) or
                !c.ggml_are_same_shape(src(up_bias_n, 0), src(up_bias_n, 1)))) continue;

            if (mulmat.shouldFuseMulMatVecF(up_n)) {
                const fusion_data: common.MmFusionArgsHost = .{
                    .gate = src(gate_n, 0),
                    .x_bias = up_bias_tensor,
                    .gate_bias = gate_bias_tensor,
                    .glu_op = c.ggml_get_glu_op(glu),
                };
                mmvf.mulMatVecF(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 5;
                break;
            }

            if (mulmat.shouldFuseMulMatVecQ(up_n)) {
                const fusion_data: common.MmFusionArgsHost = .{
                    .gate = src(gate_n, 0),
                    .x_bias = up_bias_tensor,
                    .gate_bias = gate_bias_tensor,
                    .glu_op = c.ggml_get_glu_op(glu),
                };
                mmvq.mulMatVecQ(cuda_ctx, src(up_n, 0), src(up_n, 1), srcOpt(up_n, 2), glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 5;
                break;
            }
        } else if (canFuse(cgraph, i, &.{ op, op, c.GGML_OP_GLU }, &.{})) {
            const glu = node(cgraph, i + 2);
            const gate = src(glu, 0);
            const up = src(glu, 1);

            const ok = (gate == node(cgraph, i) and up == node(cgraph, i + 1)) or
                (gate == node(cgraph, i + 1) and up == node(cgraph, i));

            if (!ok) continue;

            if (mulmat.shouldFuseMulMatVecF(up)) {
                const fusion_data: common.MmFusionArgsHost = .{ .gate = src(gate, 0), .glu_op = c.ggml_get_glu_op(glu) };
                mmvf.mulMatVecF(cuda_ctx, src(up, 0), src(up, 1), srcOpt(up, 2), glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 3;
                break;
            }

            if (mulmat.shouldFuseMulMatVecQ(up)) {
                const fusion_data: common.MmFusionArgsHost = .{ .gate = src(gate, 0), .glu_op = c.ggml_get_glu_op(glu) };
                mmvq.mulMatVecQ(cuda_ctx, src(up, 0), src(up, 1), srcOpt(up, 2), glu, &fusion_data);
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
    outer: for ([_]Op{ c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT_ID }) |op| {
        const bias_op: Op = if (op == c.GGML_OP_MUL_MAT) c.GGML_OP_ADD else c.GGML_OP_ADD_ID;

        for ([_]bool{ false, true }) |with_bias| {
            const ops: []const Op = if (op == c.GGML_OP_MUL_MAT)
                (if (with_bias) &.{ op, c.GGML_OP_MUL, bias_op } else &.{ op, c.GGML_OP_MUL })
            else if (with_bias)
                &.{ op, c.GGML_OP_RESHAPE, c.GGML_OP_REPEAT, c.GGML_OP_GET_ROWS, c.GGML_OP_MUL, bias_op }
            else
                &.{ op, c.GGML_OP_RESHAPE, c.GGML_OP_REPEAT, c.GGML_OP_GET_ROWS, c.GGML_OP_MUL };
            const n_ops: c_int = @intCast(ops.len);
            const out_nodes = [_]c_int{i + n_ops - 1};

            if (!canFuseSubgraph(cgraph, i, ops, &out_nodes) or
                !checkFusionMemoryRanges(cgraph, i, n_ops, &out_nodes, false)) continue;

            const mm_node = node(cgraph, i);
            const scale_node = if (op == c.GGML_OP_MUL_MAT) node(cgraph, i + 1) else node(cgraph, i + 4);
            const out_node = if (with_bias) node(cgraph, i + n_ops - 1) else scale_node;

            const scale = if (op == c.GGML_OP_MUL_MAT)
                getMulMatScale(scale_node, mm_node)
            else
                getMulMatIdScale(node(cgraph, i + 1), node(cgraph, i + 2), node(cgraph, i + 3), scale_node, mm_node);
            if (scale == null) continue;

            const bias = if (with_bias) getBiasTensor(out_node, scale_node, bias_op) else null;
            if (with_bias and bias == null) continue;
            if (with_bias and bias_op == c.GGML_OP_ADD and !c.ggml_are_same_shape(src(out_node, 0), src(out_node, 1))) continue;
            if (with_bias and bias_op == c.GGML_OP_ADD_ID and out_node.src[2] != mm_node.src[2]) continue;

            const fusion_data: common.MmFusionArgsHost = .{ .x_bias = bias, .x_scale = scale };

            if (mulmat.shouldFuseMulMatVecQ(mm_node)) {
                mmvq.mulMatVecQ(cuda_ctx, src(mm_node, 0), src(mm_node, 1), srcOpt(mm_node, 2), out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = n_ops;
                break :outer;
            }
        }
    }

    if (fused_mul_mat_vec) return fused_node_count - 1;

    // mul_mat + add
    for ([_]Op{ c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT_ID }) |op| {
        const bias_op: Op = if (op == c.GGML_OP_MUL_MAT) c.GGML_OP_ADD else c.GGML_OP_ADD_ID;

        if (!ggml_graph.canFuse(cgraph, i, &.{ op, bias_op })) continue;

        const mm_node = node(cgraph, i);
        const bias_node = node(cgraph, i + 1);

        var bias_tensor: *const Tensor = undefined;
        if (bias_op == c.GGML_OP_ADD) {
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

        if (bias_op == c.GGML_OP_ADD_ID and bias_node.src[2] != mm_node.src[2]) continue;
        if (bias_op == c.GGML_OP_ADD and !c.ggml_are_same_shape(src(bias_node, 0), src(bias_node, 1))) continue;

        const fusion_data: common.MmFusionArgsHost = .{ .x_bias = bias_tensor };

        if (mulmat.shouldFuseMulMatVecF(mm_node)) {
            mmvf.mulMatVecF(cuda_ctx, src(mm_node, 0), src(mm_node, 1), ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count = 2;
            break;
        }

        if (mulmat.shouldFuseMulMatVecQ(mm_node)) {
            mmvq.mulMatVecQ(cuda_ctx, src(mm_node, 0), src(mm_node, 1), ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) return fused_node_count - 1;

    if (canFuse(cgraph, i, &.{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ROPE, c.GGML_OP_VIEW, c.GGML_OP_SET_ROWS }, &.{})) {
        rope.opRmsNormMulRopeFused(cuda_ctx, nd, node(cgraph, i + 1), node(cgraph, i + 2), node(cgraph, i + 4));
        return 4;
    }

    if (canFuse(cgraph, i, &.{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ROPE }, &.{})) {
        rope.opRmsNormMulRopeFused(cuda_ctx, nd, node(cgraph, i + 1), node(cgraph, i + 2), null);
        return 2;
    }

    if (canFuse(cgraph, i, &.{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ADD }, &.{})) {
        norm.opRmsNormFusedAdd(cuda_ctx, nd, node(cgraph, i + 1), node(cgraph, i + 2));
        return 2;
    }

    if (canFuse(cgraph, i, &.{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL }, &.{})) {
        norm.opRmsNormFused(cuda_ctx, nd, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ c.GGML_OP_SSM_CONV, c.GGML_OP_ADD, c.GGML_OP_UNARY }, &.{c.GGML_UNARY_OP_SILU})) {
        ssm_conv.opSsmConv(cuda_ctx, nd, node(cgraph, i + 1), node(cgraph, i + 2));
        return 2;
    }

    if (canFuse(cgraph, i, &.{ c.GGML_OP_SSM_CONV, c.GGML_OP_UNARY }, &.{c.GGML_UNARY_OP_SILU})) {
        ssm_conv.opSsmConv(cuda_ctx, nd, null, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ c.GGML_OP_UNARY, c.GGML_OP_MUL }, &.{c.GGML_UNARY_OP_SILU}) or
        canFuse(cgraph, i, &.{ c.GGML_OP_UNARY, c.GGML_OP_MUL }, &.{c.GGML_UNARY_OP_SIGMOID}) or
        canFuse(cgraph, i, &.{ c.GGML_OP_UNARY, c.GGML_OP_MUL }, &.{c.GGML_UNARY_OP_SOFTPLUS}))
    {
        unary.opUnaryMul(cuda_ctx, nd, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ c.GGML_OP_UNARY, c.GGML_OP_SQR }, &.{c.GGML_UNARY_OP_RELU})) {
        unary.opReluSqr(cuda_ctx, nd, node(cgraph, i + 1));
        return 1;
    }

    if (canFuse(cgraph, i, &.{ c.GGML_OP_SCALE, c.GGML_OP_UNARY, c.GGML_OP_SCALE }, &.{c.GGML_UNARY_OP_TANH})) {
        softcap.opSoftcap(cuda_ctx, node(cgraph, i + 2), nd);
        return 2;
    }

    return 0;
}
