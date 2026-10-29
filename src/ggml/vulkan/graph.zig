//! Graph-level helpers of the Vulkan backend: the cost model the perf
//! logger and submission batching use, and the diagnostic node list.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.1
//! (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");

const c = impl.c;

/// `(uint64_t) x` of an `int64_t`.
fn u(x: i64) u64 {
    return @bitCast(x);
}

/// Ports `ggml_vk_get_node_flops` (ggml-vulkan.cpp:2292 @b29c606e2): the
/// floating-point operations of a matmul, convolution or attention node,
/// zero for anything else. Unsigned and wrapping, as the C++'s `uint64_t`.
pub fn getNodeFlops(node: *const c.ggml_tensor) u64 {
    if (node.op == c.GGML_OP_MUL_MAT or node.op == c.GGML_OP_MUL_MAT_ID) {
        const m_ = u(node.ne[0]);
        const n = u(node.ne[1]);
        const k = u(node.src[1].*.ne[0]);
        const batch = u(node.ne[2] *% node.ne[3]);
        return m_ *% n *% (k +% (k -% 1)) *% batch;
    }
    if (node.op == c.GGML_OP_CONV_2D or node.op == c.GGML_OP_CONV_TRANSPOSE_2D) {
        const knl = node.src[0].*;
        const cout = u(node.ne[2]);
        const size_K = u(node.src[1].*.ne[2] *% knl.ne[0] *% knl.ne[1]);
        const size_N = u(node.ne[3] *% node.ne[0] *% node.ne[1]);
        return cout *% size_N *% (size_K +% (size_K -% 1));
    }
    if (node.op == c.GGML_OP_CONV_3D) {
        const knl = node.src[0].*;
        const OC: u64 = @bitCast(@as(i64, node.op_params[11]));
        const IC: u64 = @bitCast(@as(i64, node.op_params[9]));
        const size_K = IC *% u(knl.ne[0] *% knl.ne[1] *% knl.ne[2]);
        const size_N = u(node.ne[3]) / OC *% u(node.ne[0]) *% u(node.ne[1]) *% u(node.ne[2]);
        return OC *% size_N *% (size_K +% (size_K -% 1));
    }
    if (node.op == c.GGML_OP_FLASH_ATTN_EXT) {
        const q = node.src[0].*;
        const k = node.src[1].*;
        const v = node.src[2].*;
        return 2 *% u(q.ne[1]) *% u(q.ne[2]) *% u(k.ne[0] +% v.ne[0]) *% u(k.ne[1]) *% u(q.ne[3]);
    }
    return 0;
}

/// Ports `ggml_vk_print_node_list` (ggml-vulkan.cpp:2324 @b29c606e2).
pub fn printNodeList(cgraph: *const impl.CGraph, start: c_int, end: c_int) void {
    var total_flops: u64 = 0;
    var n_ops: c_int = 0;
    var j = start;
    while (j <= end and j < cgraph.n_nodes) : (j += 1) {
        const node: *const c.ggml_tensor = cgraph.nodes[@intCast(j)].?;
        const flops = getNodeFlops(node);
        total_flops +%= flops;
        n_ops += 1;
        if (flops > 0) {
            impl.logCont("  node %d: %s (%s) [%.2f GFLOP]\n", .{ j, @as([*:0]const u8, @ptrCast(&node.name)), c.ggml_op_name(node.op), @as(f64, @floatFromInt(flops)) / 1e9 });
        } else {
            impl.logCont("  node %d: %s (%s)\n", .{ j, @as([*:0]const u8, @ptrCast(&node.name)), c.ggml_op_name(node.op) });
        }
    }
    impl.logCont("  total: %d ops, %.2f GFLOP\n", .{ n_ops, @as(f64, @floatFromInt(total_flops)) / 1e9 });
}

const m = @import("module.zig");
const ggml_graph = @import("../graph.zig");
const common = m.common;
const vk = common.vk;
const d = &common.dispatch;
const allocator = common.allocator;
const BackendCtx = m.context.BackendCtx;
const CGraph = impl.CGraph;
const Op = c.enum_ggml_op;
const TopkMoeMode = m.context.TopkMoeMode;

fn oom() noreturn {
    impl.abort("ggml_vulkan: out of memory");
}

fn nodeAt(cgraph: *const CGraph, i: c_int) *c.ggml_tensor {
    return cgraph.nodes[@intCast(i)].?;
}

fn srcOf(t: *const c.ggml_tensor, i: usize) *c.ggml_tensor {
    return t.src[i].?;
}

// The fused patterns the graph walk recognises, with the edges that make
// them one subgraph. Each edge is {node, src index, node it must be}.

/// Ports `topk_moe_early_softmax_norm` (ggml-vulkan.cpp:656 @b29c606e2).
pub const topk_moe_early_softmax_norm = [_]Op{ c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE, c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS, c.GGML_OP_RESHAPE, c.GGML_OP_SUM_ROWS, c.GGML_OP_CLAMP, c.GGML_OP_DIV, c.GGML_OP_RESHAPE };
/// Ports `topk_moe_sigmoid_norm_bias` (ggml-vulkan.cpp:661 @b29c606e2).
pub const topk_moe_sigmoid_norm_bias = [_]Op{ c.GGML_OP_UNARY, c.GGML_OP_RESHAPE, c.GGML_OP_ADD, c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS, c.GGML_OP_RESHAPE, c.GGML_OP_SUM_ROWS, c.GGML_OP_CLAMP, c.GGML_OP_DIV, c.GGML_OP_RESHAPE };
/// Ports `topk_moe_sqrt_softplus_norm_bias` (ggml-vulkan.cpp:666 @b29c606e2).
pub const topk_moe_sqrt_softplus_norm_bias = [_]Op{ c.GGML_OP_UNARY, c.GGML_OP_SQRT, c.GGML_OP_RESHAPE, c.GGML_OP_ADD, c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS, c.GGML_OP_RESHAPE, c.GGML_OP_SUM_ROWS, c.GGML_OP_CLAMP, c.GGML_OP_DIV, c.GGML_OP_RESHAPE };
/// Ports `topk_moe_early_softmax` (ggml-vulkan.cpp:673 @b29c606e2).
pub const topk_moe_early_softmax = [_]Op{ c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE, c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS };
/// Ports `topk_moe_late_softmax` (ggml-vulkan.cpp:676 @b29c606e2).
pub const topk_moe_late_softmax = [_]Op{ c.GGML_OP_ARGSORT, c.GGML_OP_VIEW, c.GGML_OP_GET_ROWS, c.GGML_OP_RESHAPE, c.GGML_OP_SOFT_MAX, c.GGML_OP_RESHAPE };
/// Ports `snake_pattern` (ggml-vulkan.cpp:682 @b29c606e2).
pub const snake_pattern = [_]Op{ c.GGML_OP_MUL, c.GGML_OP_SIN, c.GGML_OP_SQR, c.GGML_OP_MUL, c.GGML_OP_ADD };
/// Ports `rms_norm_mul_add_mul_pattern` (ggml-vulkan.cpp:700 @b29c606e2).
pub const rms_norm_mul_add_mul_pattern = [_]Op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ADD, c.GGML_OP_MUL };
/// Ports `rms_norm_mul_add_pattern` (ggml-vulkan.cpp:701 @b29c606e2).
pub const rms_norm_mul_add_pattern = [_]Op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ADD };
/// Ports `rms_norm_mul_rope_view_set_rows_pattern` (ggml-vulkan.cpp:702 @b29c606e2).
pub const rms_norm_mul_rope_view_set_rows_pattern = [_]Op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ROPE, c.GGML_OP_VIEW, c.GGML_OP_SET_ROWS };
/// Ports `rms_norm_view_set_rows_pattern` (ggml-vulkan.cpp:703 @b29c606e2).
pub const rms_norm_view_set_rows_pattern = [_]Op{ c.GGML_OP_RMS_NORM, c.GGML_OP_VIEW, c.GGML_OP_SET_ROWS };
/// Ports `rope_view_set_rows_pattern` (ggml-vulkan.cpp:704 @b29c606e2).
pub const rope_view_set_rows_pattern = [_]Op{ c.GGML_OP_ROPE, c.GGML_OP_VIEW, c.GGML_OP_SET_ROWS };

// qwen4 QSA indexer: gather per-block scores to cells + add f16 mask (cast+reshape) + top-k,
// fused into one radix-select. The cast/reshape are elided; the raw f16 mask is read in-shader.

/// Ports `topk_qsa_pattern` (ggml-vulkan.cpp:688 @b29c606e2).
pub const topk_qsa_pattern = [_]Op{ c.GGML_OP_GET_ROWS, c.GGML_OP_PERMUTE, c.GGML_OP_CONT, c.GGML_OP_CPY, c.GGML_OP_RESHAPE, c.GGML_OP_ADD, c.GGML_OP_TOP_K };
/// Ports `topk_qsa_edges` (ggml-vulkan.cpp:692 @b29c606e2).
const topk_qsa_edges = [_][3]c_int{
    .{ 1, 0, 0 }, // permute->src[0] == get_rows
    .{ 2, 0, 1 }, // cont->src[0]    == permute
    .{ 4, 0, 3 }, // reshape->src[0] == cpy (mask cast)
    .{ 5, 0, 2 }, // add->src[0]     == cont
    .{ 5, 1, 4 }, // add->src[1]     == reshape
    .{ 6, 0, 5 }, // top_k->src[0]   == add
};

/// Ports `topk_moe_early_softmax_norm_edges` (ggml-vulkan.cpp:716 @b29c606e2).
const topk_moe_early_softmax_norm_edges = [_][3]c_int{ .{ 1, 0, 0 }, .{ 2, 0, 0 }, .{ 3, 0, 2 }, .{ 4, 0, 1 }, .{ 4, 1, 3 }, .{ 5, 0, 4 }, .{ 6, 0, 5 }, .{ 7, 0, 6 }, .{ 8, 0, 5 }, .{ 8, 1, 7 }, .{ 9, 0, 8 } };
/// Ports `topk_moe_sigmoid_norm_bias_edges` (ggml-vulkan.cpp:741 @b29c606e2).
const topk_moe_sigmoid_norm_bias_edges = [_][3]c_int{ .{ 1, 0, 0 }, .{ 2, 0, 0 }, .{ 3, 0, 2 }, .{ 4, 0, 3 }, .{ 5, 0, 1 }, .{ 5, 1, 4 }, .{ 6, 0, 5 }, .{ 7, 0, 6 }, .{ 8, 0, 7 }, .{ 9, 0, 6 }, .{ 9, 1, 8 }, .{ 10, 0, 9 } };
/// Ports `topk_moe_sqrt_softplus_norm_bias_edges` (ggml-vulkan.cpp:756 @b29c606e2).
const topk_moe_sqrt_softplus_norm_bias_edges = [_][3]c_int{ .{ 1, 0, 0 }, .{ 2, 0, 1 }, .{ 3, 0, 1 }, .{ 4, 0, 3 }, .{ 5, 0, 4 }, .{ 6, 0, 2 }, .{ 6, 1, 5 }, .{ 7, 0, 6 }, .{ 8, 0, 7 }, .{ 9, 0, 8 }, .{ 10, 0, 7 }, .{ 10, 1, 9 }, .{ 11, 0, 10 } };
/// Ports `topk_moe_early_softmax_edges` (ggml-vulkan.cpp:773 @b29c606e2).
const topk_moe_early_softmax_edges = [_][3]c_int{ .{ 1, 0, 0 }, .{ 2, 0, 0 }, .{ 3, 0, 2 }, .{ 4, 0, 1 }, .{ 4, 1, 3 } };
/// Ports `topk_moe_late_softmax_edges` (ggml-vulkan.cpp:787 @b29c606e2).
const topk_moe_late_softmax_edges = [_][3]c_int{ .{ 1, 0, 0 }, .{ 2, 1, 1 }, .{ 3, 0, 2 }, .{ 4, 0, 3 }, .{ 5, 0, 4 } };
/// Ports `rope_view_set_rows_edges` (ggml-vulkan.cpp:814 @b29c606e2).
const rope_view_set_rows_edges = [_][3]c_int{ .{ 1, 0, 0 }, .{ 2, 0, 1 } };
/// Ports `rms_norm_mul_rope_view_set_rows_edges` (ggml-vulkan.cpp:819 @b29c606e2).
const rms_norm_mul_rope_view_set_rows_edges = [_][3]c_int{ .{ 1, 0, 0 }, .{ 2, 0, 1 }, .{ 3, 0, 2 }, .{ 4, 0, 3 } };
/// Ports `rms_norm_view_set_rows_edges` (ggml-vulkan.cpp:826 @b29c606e2).
const rms_norm_view_set_rows_edges = [_][3]c_int{
    .{ 1, 0, 0 }, // view->src[0]     == rms_norm
    .{ 2, 0, 1 }, // set_rows->src[0] == view
};

/// Ports `ggml_vk_is_empty` (ggml-vulkan.cpp:17265 @b29c606e2).
pub fn isEmpty(node: *const c.ggml_tensor) bool {
    return c.ggml_is_empty(node) or node.op == c.GGML_OP_NONE or node.op == c.GGML_OP_RESHAPE or node.op == c.GGML_OP_TRANSPOSE or node.op == c.GGML_OP_VIEW or node.op == c.GGML_OP_PERMUTE;
}

fn opsAre(ops: []const Op, want: []const Op) bool {
    return std.mem.eql(Op, ops, want);
}

/// Ports `ggml_vk_can_fuse_unary_mul` (ggml-vulkan.cpp:17269 @b29c606e2):
/// whether the MUL at `mul_idx` can take the UNARY at `unary_idx` as one
/// dispatch.
pub fn canFuseUnaryMul(cgraph: *const CGraph, unary_idx: c_int, mul_idx: c_int) bool {
    const unary = nodeAt(cgraph, unary_idx);
    const mul = nodeAt(cgraph, mul_idx);

    if (m.ops_basic.unaryMulOpIndex(c.ggml_get_unary_op(unary)) < 0) return false;
    if (unary.type != c.GGML_TYPE_F32 and unary.type != c.GGML_TYPE_F16) return false;
    if (unary.type != mul.type) return false;
    if (mul.src[0] != unary and mul.src[1] != unary) return false;
    const other_opt = if (mul.src[0] == unary) mul.src[1] else mul.src[0];
    const other = other_opt orelse return false;
    if (other.*.type != unary.type) return false;
    if (!c.ggml_is_contiguous_1(other) or !c.ggml_is_contiguous_1(unary.src[0])) return false;
    // fastmod needs src to tile into dst
    if (mul.src[0] == unary) return c.ggml_can_repeat(other, unary);
    return c.ggml_can_repeat(unary, mul.src[0]);
}

/// Ports `ggml_vk_can_fuse_unary_mul_pair` (ggml-vulkan.cpp:17299 @b29c606e2).
fn canFuseUnaryMulPair(cgraph: *const CGraph, node_idx: c_int) bool {
    return ggml_graph.canFuseSubgraph(cgraph, node_idx, &.{ c.GGML_OP_UNARY, c.GGML_OP_MUL }, &.{node_idx + 1}) and
        canFuseUnaryMul(cgraph, node_idx, node_idx + 1);
}

/// Ports `ggml_vk_can_fuse` (ggml-vulkan.cpp:17306 @b29c606e2): ggml's
/// structural test, then each fusion's own conditions.
pub fn canFuse(ctx: *const BackendCtx, cgraph: *const CGraph, node_idx: c_int, ops: []const Op) bool {
    if (ops.len == 2 and ops[0] == c.GGML_OP_UNARY and ops[1] == c.GGML_OP_MUL) {
        return canFuseUnaryMulPair(cgraph, node_idx);
    }

    if (!ggml_graph.canFuse(cgraph, node_idx, ops)) return false;

    if ((ops.len == 2 or ops.len == 3 or ops.len == 4) and ops[0] == c.GGML_OP_RMS_NORM and ops[1] == c.GGML_OP_MUL) {
        const rms_norm = nodeAt(cgraph, node_idx);
        const mul = nodeAt(cgraph, node_idx + 1);
        impl.assert(srcOf(rms_norm, 0).type == c.GGML_TYPE_F32, "rms_norm->src[0]->type == GGML_TYPE_F32");
        impl.assert(rms_norm.type == c.GGML_TYPE_F32, "rms_norm->type == GGML_TYPE_F32");
        // rms_norm only supports f32
        if (srcOf(mul, 0).type != c.GGML_TYPE_F32 or srcOf(mul, 1).type != c.GGML_TYPE_F32 or mul.type != c.GGML_TYPE_F32) return false;
        // if rms_norm is the B operand, then we don't handle broadcast
        if (rms_norm == srcOf(mul, 1) and !c.ggml_are_same_shape(srcOf(mul, 0), rms_norm)) return false;
        // rms_norm shader assumes contiguous rows
        if (!c.ggml_is_contiguous_rows(srcOf(mul, 0)) or !c.ggml_is_contiguous_rows(srcOf(mul, 1))) return false;

        if (ops.len >= 3 and ops[2] == c.GGML_OP_ADD) {
            const add = nodeAt(cgraph, node_idx + 2);
            const residual = if (srcOf(add, 0) == mul) srcOf(add, 1) else srcOf(add, 0);
            if (srcOf(add, 0) != mul and srcOf(add, 1) != mul) return false;
            if (residual.type != c.GGML_TYPE_F32 or add.type != c.GGML_TYPE_F32 or
                !c.ggml_are_same_shape(add, residual) or !c.ggml_is_contiguous(residual) or
                !c.ggml_is_contiguous(add) or m.context.getMisalignBytes(ctx, residual) != 0) return false;

            var dst: *const c.ggml_tensor = add;
            if (ops.len == 4) {
                if (ops[3] != c.GGML_OP_MUL) return false;
                const post_mul = nodeAt(cgraph, node_idx + 3);
                const scale = if (srcOf(post_mul, 0) == add) srcOf(post_mul, 1) else srcOf(post_mul, 0);
                if (srcOf(post_mul, 0) != add and srcOf(post_mul, 1) != add) return false;
                // The shader reads data_e[0], so the final multiply must use a scalar.
                if (scale.type != c.GGML_TYPE_F32 or post_mul.type != c.GGML_TYPE_F32 or
                    c.ggml_nelements(scale) != 1 or !c.ggml_is_contiguous(post_mul) or
                    m.context.getMisalignBytes(ctx, scale) != 0) return false;
                dst = post_mul;
            }

            if (m.context.getMisalignBytes(ctx, dst) != 0) return false;
        }
    }

    const mmAddOk = struct {
        fn f(cx: *const BackendCtx, mul: *const c.ggml_tensor, add: *const c.ggml_tensor) bool {
            const bias = if (srcOf(add, 0) == mul) srcOf(add, 1) else srcOf(add, 0);
            // mat-vec only
            if (c.ggml_nrows(mul) != 1) return false;
            // shaders assume the types match
            if (mul.type != bias.type) return false;
            // shaders reuse the D shape for bias
            if (!c.ggml_are_same_shape(mul, bias) or !c.ggml_are_same_stride(mul, bias)) return false;
            // unaligned bias isn't handled
            if (m.context.getMisalignBytes(cx, bias) != 0) return false;
            return true;
        }
    }.f;

    if ((ops.len == 2 or ops.len == 3) and ops[0] == c.GGML_OP_MUL_MAT and ops[1] == c.GGML_OP_ADD) {
        const mul = nodeAt(cgraph, node_idx);
        const add = nodeAt(cgraph, node_idx + 1);
        if (!mmAddOk(ctx, mul, add)) return false;
        if (ops.len == 3) {
            if (ops[2] != c.GGML_OP_ADD) return false;
            if (!mmAddOk(ctx, add, nodeAt(cgraph, node_idx + 2))) return false;
        }
    }

    const mmidMulOk = struct {
        fn f(cx: *const BackendCtx, cg: *const CGraph, ni: c_int, mmid: *const c.ggml_tensor, mul: *const c.ggml_tensor) bool {
            const scale = srcOf(mul, 1);
            if (mmid != srcOf(mul, 0)) return false;
            // mat-vec only
            if (!m.ops.useMulMatVecId(cg, ni)) return false;
            // shaders assume the types match
            if (mmid.type != scale.type) return false;
            // shaders assume the bias is contiguous
            if (!c.ggml_is_contiguous(scale)) return false;
            // unaligned bias isn't handled
            if (m.context.getMisalignBytes(cx, scale) != 0) return false;
            // shader only indexes by expert index
            if (scale.ne[0] != 1 or scale.ne[1] != mul.ne[1] or scale.ne[2] != 1 or scale.ne[3] != 1) return false;
            return true;
        }
    }.f;

    if ((ops.len == 2 or ops.len == 3) and ops[0] == c.GGML_OP_MUL_MAT_ID and ops[1] == c.GGML_OP_ADD_ID) {
        const mul = nodeAt(cgraph, node_idx);
        const add = nodeAt(cgraph, node_idx + 1);
        const bias = srcOf(add, 1);
        if (mul != srcOf(add, 0)) return false;
        // mat-vec only
        if (!m.ops.useMulMatVecId(cgraph, node_idx)) return false;
        // shaders assume the types match
        if (mul.type != bias.type) return false;
        // shaders assume the bias is contiguous
        if (!c.ggml_is_contiguous(bias)) return false;
        // the ID tensor must be the same for mul_mat_id and add_id
        if (mul.src[2] != add.src[2]) return false;
        // unaligned bias isn't handled
        if (m.context.getMisalignBytes(ctx, bias) != 0) return false;
        if (ops.len == 3) {
            if (ops[2] != c.GGML_OP_MUL) return false;
            return mmidMulOk(ctx, cgraph, node_idx, add, nodeAt(cgraph, node_idx + 2));
        }
    }

    if (ops.len == 2 and ops[0] == c.GGML_OP_MUL_MAT_ID and ops[1] == c.GGML_OP_MUL) {
        if (!mmidMulOk(ctx, cgraph, node_idx, nodeAt(cgraph, node_idx), nodeAt(cgraph, node_idx + 1))) return false;
    }
    return true;
}

/// Ports `ggml_vk_can_fuse_ssm_conv` (ggml-vulkan.cpp:17503 @b29c606e2):
/// SSM_CONV + UNARY(SILU), or with an ADD between; `num_extra` is 1 or 2.
fn canFuseSsmConv(ctx: *const BackendCtx, cgraph: *const CGraph, node_idx: c_int, num_extra: c_int) bool {
    const conv = nodeAt(cgraph, node_idx);
    if (conv.op != c.GGML_OP_SSM_CONV) return false;

    var silu: *const c.ggml_tensor = undefined;
    if (num_extra == 1) {
        if (!ggml_graph.canFuse(cgraph, node_idx, &.{ c.GGML_OP_SSM_CONV, c.GGML_OP_UNARY })) return false;
        silu = nodeAt(cgraph, node_idx + 1);
    } else if (num_extra == 2) {
        if (!ggml_graph.canFuse(cgraph, node_idx, &.{ c.GGML_OP_SSM_CONV, c.GGML_OP_ADD, c.GGML_OP_UNARY })) return false;
        const add = nodeAt(cgraph, node_idx + 1);
        silu = nodeAt(cgraph, node_idx + 2);
        const bias = if (srcOf(add, 0) == conv) srcOf(add, 1) else srcOf(add, 0);
        if (bias.type != c.GGML_TYPE_F32 or !c.ggml_is_contiguous(bias)) return false;
        // bias must be channel-wise (one element per channel of the conv output)
        if (c.ggml_nelements(bias) != conv.ne[0] or bias.ne[0] != conv.ne[0]) return false;
        if (add.type != c.GGML_TYPE_F32) return false;
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        if (m.context.getMisalignBytes(ctx, bias) != 0) return false;
    } else {
        return false;
    }

    if (c.ggml_get_unary_op(silu) != c.GGML_UNARY_OP_SILU) return false;
    if (conv.type != c.GGML_TYPE_F32 or silu.type != c.GGML_TYPE_F32) return false;
    // The binding must be naturally aligned.
    if (m.context.getMisalignBytes(ctx, silu) != 0) return false;
    return true;
}

/// Ports `ggml_vk_can_fuse_topk_moe` (ggml-vulkan.cpp:17558 @b29c606e2).
fn canFuseTopkMoe(ctx: *const BackendCtx, cgraph: *const CGraph, node_idx: c_int, mode: TopkMoeMode) bool {
    var softmax: *const c.ggml_tensor = undefined;
    var weights: *const c.ggml_tensor = undefined;
    var get_rows: *const c.ggml_tensor = undefined;
    var argsort: *const c.ggml_tensor = undefined;

    switch (mode) {
        .early_softmax_norm => {
            softmax = nodeAt(cgraph, node_idx + 0);
            weights = nodeAt(cgraph, node_idx + 9);
            get_rows = nodeAt(cgraph, node_idx + 4);
            argsort = nodeAt(cgraph, node_idx + 2);
        },
        .sigmoid_norm_bias => {
            softmax = nodeAt(cgraph, node_idx + 0); // really sigmoid
            weights = nodeAt(cgraph, node_idx + 10);
            get_rows = nodeAt(cgraph, node_idx + 5);
            argsort = nodeAt(cgraph, node_idx + 3);
            if (c.ggml_get_unary_op(softmax) != c.GGML_UNARY_OP_SIGMOID) return false;
            // bias is expected to be 1D
            const bias = srcOf(nodeAt(cgraph, node_idx + 2), 1);
            if (c.ggml_nrows(bias) != 1 or !c.ggml_is_contiguous(bias)) return false;
            // sigmoid fusion seems to generate infinities on moltenvk
            if (ctx.device.driver_id == vk.VK_DRIVER_ID_MOLTENVK) return false;
        },
        .sqrt_softplus_norm_bias => {
            softmax = nodeAt(cgraph, node_idx + 0); // really softplus
            weights = nodeAt(cgraph, node_idx + 11);
            get_rows = nodeAt(cgraph, node_idx + 6);
            argsort = nodeAt(cgraph, node_idx + 4);
            if (c.ggml_get_unary_op(softmax) != c.GGML_UNARY_OP_SOFTPLUS) return false;
            // bias is expected to be 1D
            const bias = srcOf(nodeAt(cgraph, node_idx + 3), 1);
            if (c.ggml_nrows(bias) != 1 or !c.ggml_is_contiguous(bias)) return false;
        },
        .early_softmax => {
            softmax = nodeAt(cgraph, node_idx + 0);
            weights = nodeAt(cgraph, node_idx + 4);
            get_rows = nodeAt(cgraph, node_idx + 4);
            argsort = nodeAt(cgraph, node_idx + 2);
        },
        .late_softmax => {
            softmax = nodeAt(cgraph, node_idx + 4);
            weights = nodeAt(cgraph, node_idx + 5);
            get_rows = nodeAt(cgraph, node_idx + 2);
            argsort = nodeAt(cgraph, node_idx + 0);
        },
        .count => return false,
    }

    var probs = srcOf(get_rows, 0);
    if (probs.op != c.GGML_OP_RESHAPE) return false;
    probs = srcOf(probs, 0);
    const selection_probs = srcOf(argsort, 0);

    if (probs != selection_probs and mode != .sigmoid_norm_bias and mode != .sqrt_softplus_norm_bias) return false;
    if (!c.ggml_is_contiguous(srcOf(softmax, 0)) or !c.ggml_is_contiguous(weights)) return false;

    if (softmax.op == c.GGML_OP_SOFT_MAX) {
        const op_params: [*]const f32 = @ptrCast(&softmax.op_params);
        const scale = op_params[0];
        const max_bias = op_params[1];
        if (scale != 1.0 or max_bias != 0.0) return false;
        // don't fuse when masks or sinks are present
        if (softmax.src[1] != null or softmax.src[2] != null) return false;
    }

    const n_expert: c_int = @truncate(softmax.ne[0]);
    if (n_expert > (1 << (m.device.num_topk_moe_pipelines - 1))) return false;

    const dev = ctx.device;
    if (!dev.subgroup_arithmetic or !dev.subgroup_shuffle or !dev.subgroup_require_full_support or dev.disable_fusion) return false;
    return true;
}

/// Ports `ggml_vk_match_ops` (ggml-vulkan.cpp:17670 @b29c606e2): a manual
/// op-sequence match, since `ggml_can_fuse_subgraph` rejects the mask's
/// external reshape/cpy.
fn matchOps(cgraph: *const CGraph, node_idx: c_int, ops: []const Op) bool {
    if (node_idx + @as(c_int, @intCast(ops.len)) > cgraph.n_nodes) return false;
    for (ops, 0..) |op, j| {
        const node = nodeAt(cgraph, node_idx + @as(c_int, @intCast(j)));
        if (node.op != op or
            (node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0 or
            (node.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0)
        {
            return false;
        }
    }
    return true;
}

/// Ports `ggml_vk_can_fuse_topk_qsa` (ggml-vulkan.cpp:17687 @b29c606e2):
/// whether the qwen4 QSA indexer top-k can be fused at `node_idx`, the
/// GET_ROWS.
fn canFuseTopkQsa(ctx: *const BackendCtx, cgraph: *const CGraph, node_idx: c_int) bool {
    const dev = ctx.device;
    if (dev.disable_fusion or dev.pipeline_topk_radix_qsa == null) return false;

    const n_ops: c_int = topk_qsa_pattern.len;
    if (!matchOps(cgraph, node_idx, &topk_qsa_pattern) or
        !ggml_graph.checkEdges(cgraph, node_idx, &topk_qsa_edges))
    {
        return false;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    var j: c_int = 0;
    while (j < n_ops - 1) : (j += 1) {
        const node = nodeAt(cgraph, node_idx + j);
        const want: i32 = if (node.op == c.GGML_OP_CPY) 2 else 1;
        if (ggml_graph.nodeGetUseCount(cgraph, node_idx + j) != want) return false;
    }

    const get_rows = nodeAt(cgraph, node_idx + 0);
    const add = nodeAt(cgraph, node_idx + n_ops - 2);
    const top_k = nodeAt(cgraph, node_idx + n_ops - 1);

    const scores = srcOf(get_rows, 0); // [n_tps, n_blocks, n_stream]
    const cell_blk = srcOf(get_rows, 1); // [n_kv, n_stream]
    const expanded = srcOf(add, 0); // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    var mask_opt: ?*c.ggml_tensor = add.src[1];
    while (mask_opt) |mk| {
        if (mk.op != c.GGML_OP_RESHAPE and mk.op != c.GGML_OP_CPY) break;
        mask_opt = mk.src[0];
    }
    const mask = mask_opt orelse return false;
    if (mask.type != c.GGML_TYPE_F16) return false;

    if (scores.type != c.GGML_TYPE_F32 or cell_blk.type != c.GGML_TYPE_I32 or top_k.type != c.GGML_TYPE_I32) return false;
    if (!c.ggml_is_contiguous(scores) or !c.ggml_is_contiguous(cell_blk) or !c.ggml_is_contiguous(mask) or
        !c.ggml_is_contiguous(expanded) or !c.ggml_is_contiguous(top_k))
    {
        return false;
    }

    const n_tps = scores.ne[0];
    const n_blocks = scores.ne[1];
    const n_stream = scores.ne[2];
    const n_kv = cell_blk.ne[0];
    const width = top_k.ne[0];

    // pin the indexer layout the shader's addressing assumes
    if (scores.ne[3] != 1 or cell_blk.ne[1] != n_stream or c.ggml_nrows(cell_blk) != n_stream or
        c.ggml_nelements(mask) != n_kv *% n_tps *% n_stream or
        expanded.ne[0] != n_kv or expanded.ne[1] != n_tps or expanded.ne[2] != n_stream or
        top_k.ne[1] != n_tps or top_k.ne[2] != n_stream or top_k.ne[3] != 1 or
        n_blocks <= 0 or n_kv <= 0 or width <= 0 or width > n_kv)
    {
        return false;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    if (m.ops_reduce.topkTournamentFits(dev, @truncate(@as(u64, @bitCast(width))))) return false;
    return true;
}

/// Ports `ggml_vk_can_fuse_rope_set_rows` (ggml-vulkan.cpp:17755 @b29c606e2).
fn canFuseRopeSetRows(ctx: *const BackendCtx, cgraph: *const CGraph, node_idx: c_int) bool {
    const rope = nodeAt(cgraph, node_idx + 0);
    const view = nodeAt(cgraph, node_idx + 1);
    const set_rows = nodeAt(cgraph, node_idx + 2);
    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    if (srcOf(rope, 0).ne[3] != 1) return false;
    if (set_rows.type != c.GGML_TYPE_F32 and set_rows.type != c.GGML_TYPE_F16) return false;
    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    if (srcOf(set_rows, 1).type != c.GGML_TYPE_I64 or !c.ggml_is_contiguous(srcOf(set_rows, 1)) or
        set_rows.nb[0] != c.ggml_type_size(set_rows.type) or m.context.getMisalignBytes(ctx, srcOf(set_rows, 1)) != 0) return false;
    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    if (!c.ggml_is_contiguous(view) or
        view.ne[0] != rope.ne[0] *% rope.ne[1] or view.ne[1] != rope.ne[2] or
        view.ne[2] != 1 or view.ne[3] != 1 or
        c.ggml_nelements(srcOf(set_rows, 1)) != rope.ne[2]) return false;
    // Only norm/neox/mrope/imrope shaders have the fusion code
    const mode = rope.op_params[2];
    if (mode != c.GGML_ROPE_TYPE_NORMAL and mode != c.GGML_ROPE_TYPE_NEOX and
        mode != c.GGML_ROPE_TYPE_MROPE and mode != c.GGML_ROPE_TYPE_IMROPE) return false;
    return true;
}

/// Ports `ggml_vk_can_fuse_rms_norm_set_rows` (ggml-vulkan.cpp:17794 @b29c606e2).
fn canFuseRmsNormSetRows(ctx: *const BackendCtx, cgraph: *const CGraph, node_idx: c_int) bool {
    const rms = nodeAt(cgraph, node_idx);
    const view = nodeAt(cgraph, node_idx + 1);
    const set_rows = nodeAt(cgraph, node_idx + 2);
    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    if (srcOf(rms, 0).type != c.GGML_TYPE_F32 or rms.type != c.GGML_TYPE_F32 or
        (set_rows.type != c.GGML_TYPE_F32 and set_rows.type != c.GGML_TYPE_F16) or
        srcOf(set_rows, 1).type != c.GGML_TYPE_I64 or !c.ggml_is_contiguous(srcOf(set_rows, 1)) or
        set_rows.nb[0] != c.ggml_type_size(set_rows.type) or m.context.getMisalignBytes(ctx, srcOf(set_rows, 1)) != 0) return false;
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    if (rms.ne[3] != 1 or !c.ggml_is_contiguous(srcOf(rms, 0)) or !c.ggml_is_contiguous(view)) return false;
    if (view.ne[0] != rms.ne[0] *% rms.ne[1] or view.ne[1] != rms.ne[2] or
        view.ne[2] != 1 or view.ne[3] != 1 or
        c.ggml_nelements(srcOf(set_rows, 1)) != rms.ne[2]) return false;
    return true;
}

/// Ports `ggml_vk_can_fuse_snake` (ggml-vulkan.cpp:17823 @b29c606e2): the
/// five-op chain y = x + sin(a*x)^2 * inv_b with [1, C] operands.
fn canFuseSnake(cgraph: *const CGraph, node_idx: c_int) bool {
    if (!ggml_graph.canFuse(cgraph, node_idx, &snake_pattern)) return false;

    const mul0 = nodeAt(cgraph, node_idx + 0);
    const sin_node = nodeAt(cgraph, node_idx + 1);
    const sqr = nodeAt(cgraph, node_idx + 2);
    const mul1 = nodeAt(cgraph, node_idx + 3);
    const add = nodeAt(cgraph, node_idx + 4);

    const x = if (c.ggml_are_same_shape(mul0, srcOf(mul0, 0))) srcOf(mul0, 0) else srcOf(mul0, 1);
    const a = if (x == srcOf(mul0, 0)) srcOf(mul0, 1) else srcOf(mul0, 0);
    const inv_b = if (srcOf(mul1, 0) == sqr) srcOf(mul1, 1) else srcOf(mul1, 0);
    const x_in_add = if (srcOf(add, 0) == mul1) srcOf(add, 1) else srcOf(add, 0);

    if (x_in_add != x) return false;
    if (x.type != c.GGML_TYPE_F32 and x.type != c.GGML_TYPE_F16 and x.type != c.GGML_TYPE_BF16) return false;
    // data_b and data_c are hardcoded float in the shader.
    if (a.type != c.GGML_TYPE_F32) return false;
    if (inv_b.type != c.GGML_TYPE_F32) return false;
    // Chain intermediates and output share x's precision.
    if (mul0.type != x.type or sin_node.type != x.type or sqr.type != x.type or mul1.type != x.type or add.type != x.type) return false;
    if (!c.ggml_are_same_shape(a, inv_b)) return false;
    if (a.ne[0] != 1 or a.ne[1] != x.ne[1]) return false;
    // Dispatch is 2D over (ne0, ne1).
    if (x.ne[2] != 1 or x.ne[3] != 1) return false;
    if (add.ne[2] != 1 or add.ne[3] != 1) return false;
    if (a.ne[2] != 1 or a.ne[3] != 1) return false;
    if (inv_b.ne[2] != 1 or inv_b.ne[3] != 1) return false;
    // Every operand must be contiguous.
    if (!c.ggml_is_contiguous(x) or !c.ggml_is_contiguous(add) or !c.ggml_is_contiguous(a) or !c.ggml_is_contiguous(inv_b)) return false;
    return true;
}

/// Ports `ggml_vk_tensors_overlap` (ggml-vulkan.cpp:17883 @b29c606e2):
/// whether two tensors share bytes of one buffer, not counting an exact
/// alias when the fusion is elementwise.
fn tensorsOverlap(a: *const c.ggml_tensor, b: *const c.ggml_tensor, elementwise: bool) bool {
    if (m.buffer.tensorBuffer(a) != m.buffer.tensorBuffer(b)) return false;
    const a_base = m.buffer.tensorOffset(a) + a.view_offs;
    const a_size = c.ggml_nbytes(a);
    const b_base = m.buffer.tensorOffset(b) + b.view_offs;
    const b_size = c.ggml_nbytes(b);
    if (elementwise and a_base == b_base and a_size == b_size) return false;
    return (b_base <= a_base and a_base < b_base + b_size) or (a_base <= b_base and b_base < a_base + a_size);
}

/// Ports `ggml_vk_can_fuse_rms_norm_mul_rope` (ggml-vulkan.cpp:17906 @b29c606e2).
fn canFuseRmsNormMulRope(ctx: *const BackendCtx, cgraph: *const CGraph, node_idx: c_int) bool {
    const rms = nodeAt(cgraph, node_idx + 0);
    const mul = nodeAt(cgraph, node_idx + 1);
    const rope = nodeAt(cgraph, node_idx + 2);
    const mode = rope.op_params[2];
    // noncontig tensors aren't tested, and don't seem common in practice
    if (!c.ggml_is_contiguous(rms) or !c.ggml_is_contiguous(mul) or !c.ggml_is_contiguous(rope)) return false;
    // only norm/neox are handled in the shader
    if (mode != c.GGML_ROPE_TYPE_NEOX and mode != c.GGML_ROPE_TYPE_NORMAL) return false;
    // shared memory size for passing data from mul->rope
    if (mul.ne[0] > 1024) return false;
    // conditions for pipeline creation
    if (@sizeOf(m.push.RmsNormMulRope) > ctx.device.properties.limits.maxPushConstantsSize) return false;
    return true;
}

/// Ports `ggml_vk_fuse_multi_add` (ggml-vulkan.cpp:17939 @b29c606e2): how
/// many consecutive ADDs one multi_add dispatch can do, or 0.
fn fuseMultiAdd(ctx: *const BackendCtx, cgraph: *const CGraph, node_idx: c_int) u32 {
    const first_node = nodeAt(cgraph, node_idx);
    if (first_node.op != c.GGML_OP_ADD) return 0;
    if (!ctx.device.multi_add) return 0;

    var num_adds: c_int = 1;
    while (node_idx + num_adds < cgraph.n_nodes and nodeAt(cgraph, node_idx + num_adds).op == c.GGML_OP_ADD and num_adds < m.pipeline.max_fused_adds) {
        num_adds += 1;
    }

    // The shader requires same shapes (strides may differ), all f32, no misalignment.
    var i: c_int = 0;
    while (i < num_adds) : (i += 1) {
        const next_node = nodeAt(cgraph, node_idx + i);
        if (!c.ggml_are_same_shape(first_node, srcOf(next_node, 0)) or
            !c.ggml_are_same_shape(first_node, srcOf(next_node, 1)) or
            next_node.type != c.GGML_TYPE_F32 or
            srcOf(next_node, 0).type != c.GGML_TYPE_F32 or
            srcOf(next_node, 1).type != c.GGML_TYPE_F32 or
            m.context.getMisalignBytes(ctx, next_node) != 0 or
            m.context.getMisalignBytes(ctx, srcOf(next_node, 0)) != 0 or
            m.context.getMisalignBytes(ctx, srcOf(next_node, 1)) != 0)
        {
            num_adds = i;
        }
    }

    // decrease num_adds if they can't all be fused
    const adds: [m.pipeline.max_fused_adds]Op = @splat(c.GGML_OP_ADD);
    while (num_adds > 1 and !ggml_graph.canFuse(cgraph, node_idx, adds[0..@intCast(num_adds)])) num_adds -= 1;

    // a single add is not "fused", so just return zero
    if (num_adds == 1) return 0;
    return @intCast(num_adds);
}

/// Ports `find_first_set` (ggml-vulkan.cpp:17991 @b29c606e2).
fn findFirstSet(x_in: u32) i32 {
    if (x_in == 0) return -1;
    return @intCast(@ctz(x_in));
}

/// The fusion the walk chose for one node: its name, how many nodes after
/// it join, and whether each fused op's sources are read elementwise.
const Fusion = struct {
    name: ?[*:0]const u8 = null,
    srcs_elementwise: [13]bool = undefined,
};

/// The chain of fusion tests at the top of each graph-walk step, in the
/// C++'s order: the first that matches wins.
fn chooseFusion(ctx: *BackendCtx, cgraph: *const CGraph, i: c_int, f: *Fusion) void {
    const ew = &f.srcs_elementwise;
    const set = struct {
        fn run(a: *[13]bool, vals: []const bool) void {
            for (vals, 0..) |v, k| a[k] = v;
        }
    }.run;

    const num_adds = fuseMultiAdd(ctx, cgraph, i);
    if (num_adds != 0) {
        ctx.num_additional_fused_ops = @intCast(num_adds - 1);
        f.name = "MULTI_ADD";
        @memset(ew[0..num_adds], true);
    } else if (canFuse(ctx, cgraph, i, &.{ c.GGML_OP_MUL_MAT, c.GGML_OP_ADD, c.GGML_OP_ADD })) {
        ctx.num_additional_fused_ops = 2;
        f.name = "MUL_MAT_ADD_ADD";
        set(ew, &.{ false, true, true });
    } else if (canFuse(ctx, cgraph, i, &.{ c.GGML_OP_MUL_MAT, c.GGML_OP_ADD })) {
        ctx.num_additional_fused_ops = 1;
        f.name = "MUL_MAT_ADD";
        set(ew, &.{ false, true });
    } else if (canFuse(ctx, cgraph, i, &.{ c.GGML_OP_MUL_MAT_ID, c.GGML_OP_ADD_ID, c.GGML_OP_MUL })) {
        ctx.num_additional_fused_ops = 2;
        f.name = "MUL_MAT_ID_ADD_ID_MUL";
        set(ew, &.{ false, true, true });
    } else if (canFuse(ctx, cgraph, i, &.{ c.GGML_OP_MUL_MAT_ID, c.GGML_OP_ADD_ID })) {
        ctx.num_additional_fused_ops = 1;
        f.name = "MUL_MAT_ID_ADD_ID";
        set(ew, &.{ false, true });
    } else if (canFuse(ctx, cgraph, i, &.{ c.GGML_OP_MUL_MAT_ID, c.GGML_OP_MUL })) {
        ctx.num_additional_fused_ops = 1;
        f.name = "MUL_MAT_ID_MUL";
        set(ew, &.{ false, true });
    } else if (ggml_graph.canFuseSubgraph(cgraph, i, &rms_norm_mul_rope_view_set_rows_pattern, &.{i + 4}) and
        ggml_graph.checkEdges(cgraph, i, &rms_norm_mul_rope_view_set_rows_edges) and
        canFuseRmsNormMulRope(ctx, cgraph, i) and
        canFuseRopeSetRows(ctx, cgraph, i + 2))
    {
        ctx.num_additional_fused_ops = 4;
        ctx.fused_rms_norm_mode = .mul_rope_view_set_rows;
        f.name = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
        set(ew, &.{ false, false, false, false, false });
    } else if (canFuse(ctx, cgraph, i, &.{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ROPE }) and canFuseRmsNormMulRope(ctx, cgraph, i)) {
        ctx.num_additional_fused_ops = 2;
        ctx.fused_rms_norm_mode = .mul_rope;
        f.name = "RMS_NORM_MUL_ROPE";
        // rope is approximately elementwise - whole rows are done by a single workgroup
        set(ew, &.{ false, true, true });
    } else if (canFuse(ctx, cgraph, i, &rms_norm_mul_add_mul_pattern)) {
        ctx.num_additional_fused_ops = 3;
        ctx.fused_rms_norm_mode = .mul_add_mul;
        f.name = "RMS_NORM_MUL_ADD_MUL";
        @memset(ew[0..4], true);
    } else if (canFuse(ctx, cgraph, i, &rms_norm_mul_add_pattern)) {
        ctx.num_additional_fused_ops = 2;
        ctx.fused_rms_norm_mode = .mul_add;
        f.name = "RMS_NORM_MUL_ADD";
        @memset(ew[0..3], true);
    } else if (ggml_graph.canFuseSubgraph(cgraph, i, &rms_norm_view_set_rows_pattern, &.{i + 2}) and
        ggml_graph.checkEdges(cgraph, i, &rms_norm_view_set_rows_edges) and
        canFuseRmsNormSetRows(ctx, cgraph, i))
    {
        ctx.num_additional_fused_ops = 2;
        ctx.fused_rms_norm_mode = .view_set_rows;
        f.name = "RMS_NORM_VIEW_SET_ROWS";
        @memset(ew[0..3], false);
    } else if (canFuse(ctx, cgraph, i, &.{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL })) {
        ctx.num_additional_fused_ops = 1;
        ctx.fused_rms_norm_mode = .mul;
        f.name = "RMS_NORM_MUL";
        // whole rows are consumed before they are overwritten, one workgroup per row
        set(ew, &.{ true, true });
    } else if (canFuse(ctx, cgraph, i, &.{ c.GGML_OP_UNARY, c.GGML_OP_MUL })) {
        ctx.num_additional_fused_ops = 1;
        f.name = switch (c.ggml_get_unary_op(nodeAt(cgraph, i))) {
            c.GGML_UNARY_OP_GELU => "GELU_MUL",
            c.GGML_UNARY_OP_SIGMOID => "SIGMOID_MUL",
            c.GGML_UNARY_OP_SILU => "SILU_MUL",
            else => "SOFTPLUS_MUL",
        };
        set(ew, &.{ true, true });
    } else if (canFuseSsmConv(ctx, cgraph, i, 2)) {
        ctx.num_additional_fused_ops = 2;
        f.name = "SSM_CONV_BIAS_SILU";
        set(ew, &.{ false, true, true });
    } else if (canFuseSsmConv(ctx, cgraph, i, 1)) {
        ctx.num_additional_fused_ops = 1;
        f.name = "SSM_CONV_SILU";
        set(ew, &.{ false, true });
    } else if (ggml_graph.canFuseSubgraph(cgraph, i, &rope_view_set_rows_pattern, &.{i + 2}) and
        ggml_graph.checkEdges(cgraph, i, &rope_view_set_rows_edges) and
        canFuseRopeSetRows(ctx, cgraph, i))
    {
        ctx.num_additional_fused_ops = 2;
        f.name = "ROPE_VIEW_SET_ROWS";
        set(ew, &.{ false, false, false });
    } else if (canFuseSnake(cgraph, i)) {
        ctx.num_additional_fused_ops = 4;
        f.name = "SNAKE";
        // safe under exact aliasing; partial overlaps are still rejected
        @memset(ew[0..5], true);
    } else if (canFuseTopkQsa(ctx, cgraph, i)) {
        ctx.num_additional_fused_ops = topk_qsa_pattern.len - 1;
        ctx.fused_topk_qsa = true;
        f.name = "TOPK_QSA";
        @memset(ew[0..@intCast(ctx.num_additional_fused_ops + 1)], false);
    } else if (topkMoe(ctx, cgraph, i, f, &topk_moe_early_softmax_norm, &.{ i + 3, i + 9 }, &topk_moe_early_softmax_norm_edges, .early_softmax_norm, 3, "TOPK_MOE_EARLY_SOFTMAX_NORM")) {
        //
    } else if (topkMoe(ctx, cgraph, i, f, &topk_moe_sigmoid_norm_bias, &.{ i + 4, i + 10 }, &topk_moe_sigmoid_norm_bias_edges, .sigmoid_norm_bias, 4, "TOPK_MOE_SIGMOID_NORM_BIAS")) {
        //
    } else if (topkMoe(ctx, cgraph, i, f, &topk_moe_sqrt_softplus_norm_bias, &.{ i + 5, i + 11 }, &topk_moe_sqrt_softplus_norm_bias_edges, .sqrt_softplus_norm_bias, 5, "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS")) {
        //
    } else if (topkMoe(ctx, cgraph, i, f, &topk_moe_early_softmax, &.{ i + 3, i + 4 }, &topk_moe_early_softmax_edges, .early_softmax, 3, "TOPK_MOE_EARLY_SOFTMAX")) {
        //
    } else if (topkMoe(ctx, cgraph, i, f, &topk_moe_late_softmax, &.{ i + 1, i + 5 }, &topk_moe_late_softmax_edges, .late_softmax, 1, "TOPK_MOE_LATE_SOFTMAX")) {
        //
    }

    if (ctx.fused_topk_moe_mode != .count) {
        // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
        const n = ctx.num_additional_fused_ops;
        if (ggml_graph.canFuseSubgraph(cgraph, i + n - 1, &.{ c.GGML_OP_DIV, c.GGML_OP_RESHAPE, c.GGML_OP_SCALE }, &.{i + n + 1}) or
            ggml_graph.canFuseSubgraph(cgraph, i + n, &.{ c.GGML_OP_GET_ROWS, c.GGML_OP_SCALE }, &.{i + n + 1}))
        {
            ctx.fused_topk_moe_scale = true;
            ctx.num_additional_fused_ops += 1;
            ew[@intCast(ctx.num_additional_fused_ops)] = false;
        }
    }
}

/// One topk_moe arm of the fusion chain: the subgraph and its edges, then
/// the mode's own test.
fn topkMoe(ctx: *BackendCtx, cgraph: *const CGraph, i: c_int, f: *Fusion, ops: []const Op, outputs: []const c_int, edges: []const [3]c_int, mode: TopkMoeMode, view_bit: u5, comptime name: [*:0]const u8) bool {
    if (!(ggml_graph.canFuseSubgraph(cgraph, i, ops, outputs) and ggml_graph.checkEdges(cgraph, i, edges) and canFuseTopkMoe(ctx, cgraph, i, mode))) return false;
    ctx.num_additional_fused_ops = @intCast(ops.len - 1);
    // view of argsort writes to memory
    ctx.fused_ops_write_mask |= @as(c_int, 1) << view_bit;
    ctx.fused_topk_moe_mode = mode;
    f.name = name;
    @memset(f.srcs_elementwise[0..@intCast(ctx.num_additional_fused_ops + 1)], false);
    return true;
}

/// Ports `ggml_backend_vk_graph_compute` (ggml-vulkan.cpp:18003 @b29c606e2):
/// records the graph node by node, fusing where it can, and submits in
/// batches sized by node count and estimated flops so the GPU starts while
/// the rest is recorded.
pub fn graphCompute(backend: c.ggml_backend_t, cgraph_c: ?*c.ggml_cgraph) callconv(.c) c.enum_ggml_status {
    const ctx: *BackendCtx = @ptrCast(@alignCast(backend.*.context));
    const cgraph: *CGraph = @ptrCast(@alignCast(cgraph_c.?));
    const dev = ctx.device;

    dev.diag_cgraph = null;
    dev.diag_prev_start = -1;
    dev.diag_prev_end = -1;

    if (m.instance.instance.debug_utils_support) {
        dev.debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    var queue_dbg = m.submit.DebugLabel.queue(dev.compute_queue.?.handle, "ggml_backend_vk_graph_compute");
    defer queue_dbg.close();

    ctx.prealloc_size_add_rms_partials_offset = 0;
    ctx.do_add_rms_partials = false;
    ctx.do_add_rms_partials_offset_calculation = false;

    var last_node: c_int = cgraph.n_nodes - 1;
    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    while (last_node > 0 and (isEmpty(nodeAt(cgraph, last_node)) or (nodeAt(cgraph, last_node).flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0)) last_node -= 1;

    // Reserve tensor context space for all nodes
    // `std::vector::resize` value-initialises the slots it adds.
    const old_len = ctx.tensor_ctxs.items.len;
    ctx.tensor_ctxs.resize(allocator, @intCast(cgraph.n_nodes)) catch oom();
    if (ctx.tensor_ctxs.items.len > old_len) @memset(ctx.tensor_ctxs.items[old_len..], null);

    var first_node_in_batch = true; // true if next node will be first node in a batch
    var submit_node_idx: c_int = 0; // index to first node in a batch

    _ = m.context.submitTransferCtx(ctx);

    const perf = m.instance.loggers.perf_logger_enabled;
    if (perf) perfBegin(ctx, cgraph);

    ctx.prealloc_y_last_pipeline_used = null;
    ctx.prealloc_y_last_tensor_used = null;
    ctx.prealloc_y_last_k_padded = false;

    if (ctx.prealloc_size_add_rms_partials != 0) {
        m.nodes.preallocateBuffers(ctx, null);
        const compute_ctx = m.context.getComputeCtx(ctx);
        // initialize partial sums to zero.
        m.submit.bufferMemsetAsync(compute_ctx, ctx.prealloc_add_rms_partials.?, 0, 0, ctx.prealloc_size_add_rms_partials);
        m.context.syncBuffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated: every 200 GFLOP, scaled down
    // by the graph's total so smaller models submit earlier, and at least
    // every max_nodes_per_submit nodes.
    var submitted_nodes: u32 = 0;
    var submit_count: u32 = 0;
    var batch_flops: u64 = 0;
    var total_flops: u64 = 0;
    var flops_cap: u64 = 200_000_000_000;

    // On weaker AMD GPUs larger submissions can hit a driver timeout
    if (dev.vendor_id == common.vendor_id.amd and dev.shader_core_count > 0) {
        if (dev.architecture == .amd_gcn and dev.shader_core_count < 32) {
            flops_cap = 500_000_000 * @as(u64, dev.shader_core_count);
        } else if (dev.architecture != .amd_gcn and dev.shader_core_count < 24) {
            flops_cap = 2_000_000_000 * @as(u64, dev.shader_core_count);
        }
    }
    var flops_per_submit: u64 = @min(flops_cap, ctx.last_total_flops / 40);

    const SubmitAfter = struct {
        ctx: *BackendCtx,
        cgraph: *CGraph,
        first_node_in_batch: *bool,
        submitted_nodes: *u32,
        batch_flops: *u64,
        flops_per_submit: *u64,
        submit_count: *u32,

        fn run(s: @This(), start: c_int, end: c_int) void {
            const cx = s.ctx;
            const dv = cx.device;
            if (dv.serialize_submissions) {
                const res = d.vkWaitForFences.?(dv.device, 1, &cx.fence, vk.VK_TRUE, std.math.maxInt(u64));
                if (res == vk.VK_ERROR_DEVICE_LOST) {
                    m.submit.printDeviceFaultInfo(dv);
                    impl.logError("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n", .{ dv.name.ptr, start, end });
                    printNodeList(s.cgraph, start, end);
                    impl.abort("ggml_vulkan: device lost");
                }
                if (res != vk.VK_SUCCESS) {
                    impl.logError("ggml_vulkan: waitForFences error during serialized submission\n", .{});
                    impl.abort("ggml_vulkan: waitForFences during serialized submission");
                }
                common.checkOrAbort(d.vkResetFences.?(dv.device, 1, &cx.fence), "vk::Device::resetFences");
                cx.submit_pending = false;
                dv.diag_cgraph = s.cgraph;
                dv.diag_prev_start = start;
                dv.diag_prev_end = end;
            }
            s.first_node_in_batch.* = true;
            s.submitted_nodes.* = 0;
            s.batch_flops.* = 0;
            if (s.submit_count.* < 3) s.flops_per_submit.* *%= 2;
            s.submit_count.* += 1;
        }
    };
    const submit_after: SubmitAfter = .{ .ctx = ctx, .cgraph = cgraph, .first_node_in_batch = &first_node_in_batch, .submitted_nodes = &submitted_nodes, .batch_flops = &batch_flops, .flops_per_submit = &flops_per_submit, .submit_count = &submit_count };

    var i: c_int = 0;
    while (i < cgraph.n_nodes) : (i += 1) {
        if (first_node_in_batch) submit_node_idx = i;

        {
            const node_flops = getNodeFlops(nodeAt(cgraph, i));
            total_flops +%= node_flops;
            // Flush the current batch before recording a node that would push it over the flop threshold
            if (flops_per_submit != 0 and submitted_nodes > 0 and batch_flops +% node_flops >= flops_per_submit) {
                const flush_ctx = m.context.getComputeCtx(ctx);
                m.submit.ctxEnd(flush_ctx);
                flush_ctx.exit_tensor_idx = -1;
                ctx.compute_ctx = null;
                m.nodes.computeForward(ctx, cgraph, submit_node_idx, false);
                submit_after.run(submit_node_idx, i - 1);
                submit_node_idx = i;
            }
            batch_flops +%= node_flops;
        }

        // Whether each fused op's srcs contribute elementwise, which decides
        // whether they may alias the destination.
        var fusion: Fusion = .{};

        ctx.fused_topk_moe_mode = .count;
        ctx.fused_topk_moe_scale = false;
        ctx.fused_topk_qsa = false;
        ctx.fused_rms_norm_mode = .count;
        if (!dev.disable_fusion) chooseFusion(ctx, cgraph, i, &fusion);
        impl.assert(ctx.num_additional_fused_ops < @as(c_int, fusion.srcs_elementwise.len), "ctx->num_additional_fused_ops < (int)(sizeof(op_srcs_fused_elementwise) / sizeof(op_srcs_fused_elementwise[0]))");
        ctx.fused_ops_write_mask |= @as(c_int, 1) << @intCast(ctx.num_additional_fused_ops);

        // Disable fusion if it would overwrite src operands still in use.
        if (ctx.num_additional_fused_ops != 0) disableUnsafeFusion(ctx, cgraph, i, &fusion);

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        const almost_ready = (cgraph.n_nodes - i) < @divTrunc(cgraph.n_nodes, 5);
        const submit = (submitted_nodes >= dev.max_nodes_per_submit) or
            (flops_per_submit != 0 and batch_flops >= flops_per_submit) or
            (i + ctx.num_additional_fused_ops >= last_node) or
            (almost_ready and !ctx.almost_ready_fence_pending);

        const enqueued = m.nodes.buildGraph(ctx, cgraph, i, submit_node_idx, i + ctx.num_additional_fused_ops >= last_node, almost_ready, submit);

        if (perf and enqueued) {
            const compute_ctx = m.context.getComputeCtx(ctx);
            if (!m.instance.loggers.perf_logger_concurrent) {
                // track a single node/fusion for the current query
                const qi: usize = @intCast(ctx.query_idx);
                ctx.query_nodes.items[qi] = nodeAt(cgraph, i);
                ctx.query_fusion_names.items[qi] = fusion.name;
                writeTimestamp(ctx, compute_ctx);
                m.context.syncBuffers(ctx, compute_ctx);
            } else {
                // track a fusion string and number of fused ops for the current node_idx
                ctx.query_fusion_names.items[@intCast(i)] = fusion.name;
                ctx.query_fusion_node_count.items[@intCast(i)] = ctx.num_additional_fused_ops;
            }
        }

        if (enqueued) {
            submitted_nodes += 1;
            if (first_node_in_batch) first_node_in_batch = false;
        }

        if (submit and enqueued) submit_after.run(submit_node_idx, i + ctx.num_additional_fused_ops);
        i += ctx.num_additional_fused_ops;
        ctx.num_additional_fused_ops = 0;
        ctx.fused_ops_write_mask = 0;
    }

    ctx.last_total_flops = total_flops;

    if (perf) perfEnd(ctx, cgraph);

    if (!dev.support_async) m.context.synchronize(ctx);

    return c.GGML_STATUS_SUCCESS;
}

/// The overlap test that turns a fusion off: an output of the fused run
/// overlapping a source the run still reads, other than an intermediate it
/// elides.
fn disableUnsafeFusion(ctx: *BackendCtx, cgraph: *const CGraph, i: c_int, fusion: *Fusion) void {
    // There are up to two output nodes - topk_moe has two.
    const bits: u32 = @bitCast(ctx.fused_ops_write_mask & ~(@as(c_int, 1) << @intCast(ctx.num_additional_fused_ops)));
    var output_nodes: [2]?*const c.ggml_tensor = .{ nodeAt(cgraph, i + ctx.num_additional_fused_ops), null };
    if (bits != 0) {
        const output_idx = findFirstSet(bits);
        impl.assert(bits == (@as(u32, 1) << @intCast(output_idx)), "bits == (1u << output_idx)");
        output_nodes[1] = nodeAt(cgraph, i + output_idx);
    }

    var need_disable = false;
    for (output_nodes) |dst_opt| {
        const dst = dst_opt orelse continue;
        // Loop over all srcs of all nodes in the fusion. If the src overlaps
        // the destination and the src is not an intermediate node that's being
        // elided, then disable fusion.
        var k: c_int = 0;
        while (k <= ctx.num_additional_fused_ops) : (k += 1) {
            for (0..c.GGML_MAX_SRC) |s| {
                const src = nodeAt(cgraph, i + k).src[s] orelse continue;
                if (src.*.op == c.GGML_OP_NONE) continue;
                if (tensorsOverlap(src, dst, fusion.srcs_elementwise[@intCast(k)])) {
                    var found = false;
                    var n: c_int = 0;
                    while (n < k) : (n += 1) {
                        if (nodeAt(cgraph, i + n) == src) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) need_disable = true;
                }
            }
        }
    }
    if (need_disable) {
        ctx.num_additional_fused_ops = 0;
        ctx.fused_ops_write_mask = 1;
        ctx.fused_topk_moe_mode = .count;
        ctx.fused_topk_moe_scale = false;
        ctx.fused_topk_qsa = false;
        ctx.fused_rms_norm_mode = .count;
        fusion.name = null;
    }
}

/// `writeTimestamp` into the perf query pool, advancing the query index.
pub fn writeTimestamp(ctx: *BackendCtx, compute_ctx: *m.submit.Context) void {
    d.vkCmdWriteTimestamp.?(compute_ctx.cmd(), vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, ctx.query_pool, @intCast(ctx.query_idx));
    ctx.query_idx += 1;
}

/// The perf logger's setup at the top of `ggml_backend_vk_graph_compute`:
/// size the query pool and write the first timestamp.
fn perfBegin(ctx: *BackendCtx, cgraph: *const CGraph) void {
    const dev = ctx.device;
    if (ctx.num_queries < cgraph.n_nodes + 1) {
        if (ctx.query_pool != null) d.vkDestroyQueryPool.?(dev.device, ctx.query_pool, null);
        var qci = common.structure(vk.VkQueryPoolCreateInfo, vk.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
        qci.queryType = vk.VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = @intCast(cgraph.n_nodes + 100);
        common.checkOrAbort(d.vkCreateQueryPool.?(dev.device, &qci, null, &ctx.query_pool), "vk::Device::createQueryPool");
        ctx.num_queries = @intCast(qci.queryCount);
        const nq: usize = @intCast(ctx.num_queries);
        ctx.query_fusion_names.resize(allocator, nq) catch oom();
        ctx.query_fusion_node_count.resize(allocator, nq) catch oom();
        ctx.query_nodes.resize(allocator, nq) catch oom();
        ctx.query_node_idx.resize(allocator, nq) catch oom();
    }
    d.vkResetQueryPool.?(dev.device, ctx.query_pool, 0, @intCast(cgraph.n_nodes + 1));
    @memset(ctx.query_fusion_names.items, null);
    @memset(ctx.query_fusion_node_count.items, 0);
    @memset(ctx.query_nodes.items, null);
    @memset(ctx.query_node_idx.items, 0);

    impl.assert(ctx.compute_ctx == null, "ctx->compute_ctx.expired()");
    const compute_ctx = m.context.getComputeCtx(ctx);
    ctx.query_idx = 0;
    writeTimestamp(ctx, compute_ctx);
    m.context.syncBuffers(ctx, compute_ctx);
}

/// The perf logger's tail: submit, wait, read the timestamps back, and
/// hand each op's or each group's time to the logger.
fn perfEnd(ctx: *BackendCtx, cgraph: *const CGraph) void {
    const dev = ctx.device;
    impl.assert(ctx.compute_ctx != null, "!ctx->compute_ctx.expired()");
    const compute_ctx = ctx.compute_ctx.?;
    m.submit.ctxEnd(compute_ctx);
    m.submit.submit(compute_ctx, dev.fence);
    m.submit.waitAndResetFence(dev, dev.fence, "GGML_VULKAN_PERF waitForFences");
    ctx.compute_ctx = null;

    // Get the results and pass them to the logger
    const timestamps = allocator.alloc(u64, @intCast(cgraph.n_nodes + 1)) catch oom();
    defer allocator.free(timestamps);
    m.submit.vkCheck(d.vkGetQueryPoolResults.?(dev.device, ctx.query_pool, 0, @intCast(ctx.query_idx), timestamps.len * @sizeOf(u64), timestamps.ptr, @sizeOf(u64), vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT), "get timestamp results", dev);
    const period = dev.properties.limits.timestampPeriod;
    const logger = ctx.perf_logger.?;
    if (!m.instance.loggers.perf_logger_concurrent) {
        // Log each op separately
        var q: usize = 1;
        while (q < @as(usize, @intCast(ctx.query_idx))) : (q += 1) {
            const t: u64 = @intFromFloat(@as(f32, @floatFromInt(timestamps[q] -% timestamps[q - 1])) * period);
            logger.logTiming(&.{ctx.query_nodes.items[q].?}, &.{ctx.query_fusion_names.items[q]}, t);
        }
    } else {
        // Log each group of nodes
        var prev_node_idx: c_int = 0;
        var q: usize = 1;
        while (q < @as(usize, @intCast(ctx.query_idx))) : (q += 1) {
            const cur_node_idx = ctx.query_node_idx.items[q];
            var nodes: std.ArrayList(*c.ggml_tensor) = .empty;
            defer nodes.deinit(allocator);
            var names: std.ArrayList(?[*:0]const u8) = .empty;
            defer names.deinit(allocator);
            var node_idx = prev_node_idx;
            while (node_idx < cur_node_idx) : (node_idx += 1) {
                if (impl.opIsEmpty(nodeAt(cgraph, node_idx).op)) continue;
                nodes.append(allocator, nodeAt(cgraph, node_idx)) catch oom();
                names.append(allocator, ctx.query_fusion_names.items[@intCast(node_idx)]) catch oom();
                node_idx += ctx.query_fusion_node_count.items[@intCast(node_idx)];
            }
            prev_node_idx = cur_node_idx;
            const t: u64 = @intFromFloat(@as(f32, @floatFromInt(timestamps[q] -% timestamps[q - 1])) * period);
            logger.logTiming(nodes.items, names.items, t);
        }
    }
    logger.printTimings(false);
}

/// Ports `ggml_vk_graph_optimize` (ggml-vulkan.cpp:18499 @b29c606e2):
/// reorders the graph so independent nodes run together -- fewer barriers
/// -- without splitting any fusion pattern. A kept topk_moe pattern tells
/// the scheduler, through `params`' `add_alloc_dep`, to keep its external
/// inputs alive until the fused output. Each pass takes the first unused
/// node, then up to 19 following "real" nodes that depend on nothing still
/// pending (or only on a node they fuse with), pulling forward the rope /
/// view+set_rows / bias / MUL tails of those fusions; then the view nodes
/// whose sources are done.
pub fn graphOptimize(backend: c.ggml_backend_t, cgraph: ?*c.ggml_cgraph, params: ?*c.struct_ggml_backend_graph_optimize_params) callconv(.c) void {
    const ctx: *BackendCtx = @ptrCast(@alignCast(backend.*.context));
    if (ctx.device.disable_graph_optimize) return;

    const graph: *CGraph = @ptrCast(@alignCast(cgraph.?));
    const n: usize = @intCast(graph.n_nodes);
    if (n == 0) return;
    const nodes = graph.nodes[0..n];
    const node = struct {
        fn f(g: []?*c.ggml_tensor, i: usize) *c.ggml_tensor {
            return g[i].?;
        }
    }.f;

    // The lambda's own `is_empty`: by op alone. `isEmpty` above ports
    // `ggml_vk_is_empty`, which also counts a zero-element tensor, and a
    // single-token decode has a zero-row GET_ROWS that it would move with
    // the views.
    const isViewOp = struct {
        fn f(t: *const c.ggml_tensor) bool {
            return t.op == c.GGML_OP_NONE or t.op == c.GGML_OP_RESHAPE or t.op == c.GGML_OP_TRANSPOSE or t.op == c.GGML_OP_VIEW or t.op == c.GGML_OP_PERMUTE;
        }
    }.f;

    const isSrcOf = struct {
        fn base(tensor: *const c.ggml_tensor) *const c.ggml_tensor {
            return tensor.view_src orelse tensor;
        }

        fn f(dst: *const c.ggml_tensor, src: *const c.ggml_tensor) bool {
            for (dst.src, src.src) |ds, ss| {
                if (ds == src) return true;
                if (isViewOp(dst) or isViewOp(src)) continue;
                // A source view of dst may read storage written through a different view by src.
                if (ds) |d_src| {
                    if (base(d_src) == base(src)) return true;
                }
                // Moving dst forward may overwrite storage still read through a view by src.
                if (ss) |s_src| {
                    if (base(dst) == base(s_src)) return true;
                }
            }
            // implicit dependency if they view the same tensor
            return base(dst) == base(src);
        }
    }.f;

    const gpa = m.common.allocator;
    var new_order: std.ArrayList(*c.ggml_tensor) = .empty;
    defer new_order.deinit(gpa);
    new_order.ensureTotalCapacity(gpa, n) catch oom();
    const used = gpa.alloc(bool, n) catch oom();
    defer gpa.free(used);
    @memset(used, false);
    var used_node_set: std.AutoHashMapUnmanaged(*const c.ggml_tensor, void) = .empty;
    defer used_node_set.deinit(gpa);
    var current_set: std.ArrayList(usize) = .empty;
    defer current_set.deinit(gpa);

    const Ctx = struct {
        nodes: []?*c.ggml_tensor,
        used: []bool,

        // Check for fusion patterns and avoid reordering them
        fn matchPattern(self: @This(), pattern: []const Op, start: usize) bool {
            if (start + pattern.len > self.nodes.len) return false;
            var is_pattern = true;
            for (pattern, 0..) |op, j| {
                if (self.nodes[start + j].?.op != op or self.used[start + j]) is_pattern = false;
            }
            return is_pattern;
        }

        fn anyPattern(self: @This(), start: usize) bool {
            return self.matchPattern(&topk_moe_early_softmax_norm, start) or
                self.matchPattern(&topk_moe_sigmoid_norm_bias, start) or
                self.matchPattern(&topk_moe_sqrt_softplus_norm_bias, start) or
                self.matchPattern(&topk_moe_early_softmax, start) or
                self.matchPattern(&topk_moe_late_softmax, start) or
                self.matchPattern(&snake_pattern, start) or
                self.inQsaPattern(start) or
                self.matchPattern(&rms_norm_mul_add_mul_pattern, start) or
                self.matchPattern(&rms_norm_mul_add_pattern, start) or
                self.matchPattern(&rms_norm_mul_rope_view_set_rows_pattern, start) or
                self.matchPattern(&rms_norm_view_set_rows_pattern, start) or
                self.matchPattern(&rope_view_set_rows_pattern, start);
        }

        // scheduled or zero-compute nodes in [lo, hi)
        fn emptyOrScheduledBetween(self: @This(), lo: usize, hi: usize) bool {
            for (lo..hi) |v| {
                if (!self.used[v] and !isViewOp(self.nodes[v].?)) return false;
            }
            return true;
        }

        // Protect every interior QSA node (not just the start): the mask branch is
        // independent, so it gets pulled out and breaks keep_pattern otherwise.
        fn inQsaPattern(self: @This(), idx: usize) bool {
            for (0..topk_qsa_pattern.len) |o| {
                if (idx >= o and self.matchPattern(&topk_qsa_pattern, idx - o)) return true;
            }
            return false;
        }
    };
    const pc: Ctx = .{ .nodes = nodes, .used = used };

    var first_unused: usize = 0;
    outer: while (first_unused < n) {
        current_set.clearRetainingCapacity();

        // Kept patterns go out whole, in the C++'s order. The topk_moe ones
        // (`keep_topk_moe_pattern`) first keep their external inputs alive
        // through the fused output (`add_pattern_alloc_deps`), which some
        // variants extend by a trailing SCALE.
        const kept = [_]struct { pattern: []const Op, topk_moe: bool }{
            .{ .pattern = &topk_moe_early_softmax_norm, .topk_moe = true },
            .{ .pattern = &topk_moe_sigmoid_norm_bias, .topk_moe = true },
            .{ .pattern = &topk_moe_sqrt_softplus_norm_bias, .topk_moe = true },
            .{ .pattern = &topk_moe_early_softmax, .topk_moe = true },
            .{ .pattern = &topk_moe_late_softmax, .topk_moe = true },
            .{ .pattern = &snake_pattern, .topk_moe = false },
            .{ .pattern = &topk_qsa_pattern, .topk_moe = false },
            .{ .pattern = &rms_norm_mul_add_mul_pattern, .topk_moe = false },
            .{ .pattern = &rms_norm_mul_add_pattern, .topk_moe = false },
            .{ .pattern = &rms_norm_mul_rope_view_set_rows_pattern, .topk_moe = false },
            .{ .pattern = &rms_norm_view_set_rows_pattern, .topk_moe = false },
            .{ .pattern = &rope_view_set_rows_pattern, .topk_moe = false },
        };
        for (kept) |kp| {
            const pattern = kp.pattern;
            if (pc.matchPattern(pattern, first_unused)) {
                if (kp.topk_moe) {
                    var last_node = first_unused + pattern.len - 1;
                    // Some TOPK_MOE variants fuse a trailing scale.
                    if (last_node + 1 < n and node(nodes, last_node + 1).op == c.GGML_OP_SCALE) last_node += 1;
                    addPatternAllocDeps(params.?, nodes, first_unused, pattern.len, last_node);
                }
                for (0..pattern.len) |j| {
                    new_order.appendAssumeCapacity(node(nodes, first_unused + j));
                    used_node_set.put(gpa, node(nodes, first_unused + j), {}) catch oom();
                    used[first_unused + j] = true;
                }
                while (first_unused < n and used[first_unused]) first_unused += 1;
                continue :outer;
            }
        }

        // First, grab the next unused node.
        current_set.append(gpa, first_unused) catch oom();

        // The next N nodes: take any that depend on nothing not yet run,
        // or only on the node just before them where the pair fuses. This
        // first pass takes real nodes only, the second view nodes, so the
        // two do not interleave in a way that breaks fusion.
        const num_to_check = 20;
        const end = @min(first_unused + num_to_check, n);
        var j = first_unused + 1;
        while (j < end) : (j += 1) {
            if (used[j]) continue;
            if (isViewOp(node(nodes, j))) continue;
            // Don't pull forward nodes from fusion patterns
            if (pc.anyPattern(j)) continue;

            var ok = true;
            for (first_unused..j) |cc| {
                const nc = node(nodes, cc);
                const nj = node(nodes, j);
                const is_back = cc == current_set.items[current_set.items.len - 1];
                const adjacent = j == cc + 1 and is_back;
                const fuses = (is_back and nc.op == c.GGML_OP_RMS_NORM and nj.op == c.GGML_OP_MUL and pc.emptyOrScheduledBetween(cc + 1, j)) or
                    (is_back and nc.op == c.GGML_OP_UNARY and nj.op == c.GGML_OP_MUL and pc.emptyOrScheduledBetween(cc + 1, j)) or
                    adjacent and
                        ((nc.op == c.GGML_OP_MUL_MAT and nj.op == c.GGML_OP_ADD) or
                            (nc.op == c.GGML_OP_MUL_MAT_ID and nj.op == c.GGML_OP_ADD_ID) or
                            (nc.op == c.GGML_OP_MUL_MAT_ID and nj.op == c.GGML_OP_MUL) or
                            (nc.op == c.GGML_OP_ADD and nj.op == c.GGML_OP_ADD) or
                            (nc.op == c.GGML_OP_SSM_CONV and nj.op == c.GGML_OP_ADD) or
                            (nc.op == c.GGML_OP_SSM_CONV and nj.op == c.GGML_OP_UNARY));
                if (!used[cc] and isSrcOf(nj, nc) and !fuses) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;

            current_set.append(gpa, j) catch oom();
            var rope_idx = j;

            // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
            if (j > 0 and node(nodes, j).op == c.GGML_OP_MUL and node(nodes, j - 1).op == c.GGML_OP_RMS_NORM) {
                var k = j + 1;
                while (k < @min(j + 15, n)) : (k += 1) {
                    const nk = node(nodes, k);
                    if (nk.op == c.GGML_OP_ROPE and
                        nk.src[0] == node(nodes, j) and
                        // Check that other srcs are already valid
                        nk.src[1].*.op == c.GGML_OP_NONE and
                        (nk.src[2] == null or nk.src[2].*.op == c.GGML_OP_NONE))
                    {
                        rope_idx = k;
                        current_set.append(gpa, rope_idx) catch oom();
                        used[rope_idx] = true;
                        break;
                    }
                }
            }
            // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
            if (node(nodes, rope_idx).op == c.GGML_OP_ROPE or node(nodes, rope_idx).op == c.GGML_OP_RMS_NORM) {
                var view_idx: ?usize = null;
                var set_rows_idx: ?usize = null;
                var k = rope_idx + 1;
                while (k < @min(rope_idx + 15, n)) : (k += 1) {
                    if (used[k]) continue;
                    const nk = node(nodes, k);
                    if (view_idx == null and nk.op == c.GGML_OP_VIEW and nk.src[0] == node(nodes, rope_idx)) {
                        view_idx = k;
                        continue;
                    }
                    if (view_idx != null and nk.op == c.GGML_OP_SET_ROWS and nk.src[0] == node(nodes, view_idx.?)) {
                        set_rows_idx = k;
                        break;
                    }
                }
                if (set_rows_idx) |sr| {
                    const vi = view_idx.?;
                    const node_idxs = [_]c_int{ @intCast(rope_idx), @intCast(vi), @intCast(sr) };
                    const pull_ops = [_]Op{ node(nodes, rope_idx).op, c.GGML_OP_VIEW, c.GGML_OP_SET_ROWS };
                    const sr_c: c_int = @intCast(sr);
                    var can_pull = ggml_graph.ggml_can_fuse_subgraph_ext(graph, &node_idxs, 3, &pull_ops, &sr_c, 1);

                    var cc = rope_idx + 1;
                    while (can_pull and cc < sr) : (cc += 1) {
                        if (!used[cc] and cc != vi and !isViewOp(node(nodes, cc)) and
                            isSrcOf(node(nodes, sr), node(nodes, cc))) can_pull = false;
                    }

                    if (can_pull) {
                        current_set.append(gpa, vi) catch oom();
                        current_set.append(gpa, sr) catch oom();
                        used[vi] = true;
                        used[sr] = true;
                    }
                }
            }
            // Look for MUL_MAT_ID + ADD_ID + MUL, and MUL_MAT + ADD + ADD:
            // the tail's src1 must be weights or already processed.
            const tails = [_]struct { prev: Op, cur: Op, tail: Op }{
                .{ .prev = c.GGML_OP_MUL_MAT_ID, .cur = c.GGML_OP_ADD_ID, .tail = c.GGML_OP_MUL },
                .{ .prev = c.GGML_OP_MUL_MAT, .cur = c.GGML_OP_ADD, .tail = c.GGML_OP_ADD },
            };
            for (tails) |t| {
                if (j > 0 and node(nodes, j).op == t.cur and node(nodes, j - 1).op == t.prev) {
                    var k = j + 1;
                    while (k < @min(j + 15, n)) : (k += 1) {
                        const nk = node(nodes, k);
                        if (nk.op == t.tail and nk.src[0] == node(nodes, j) and
                            (nk.src[1].*.op == c.GGML_OP_NONE or used_node_set.contains(nk.src[1])))
                        {
                            current_set.append(gpa, k) catch oom();
                            used[k] = true;
                            break;
                        }
                    }
                }
            }
            // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
            if (j > 0 and node(nodes, j).op == c.GGML_OP_ADD and node(nodes, j - 1).op == c.GGML_OP_SSM_CONV) {
                var k = j + 1;
                while (k < @min(j + 15, n)) : (k += 1) {
                    const nk = node(nodes, k);
                    if (nk.op == c.GGML_OP_UNARY and nk.src[0] == node(nodes, j)) {
                        current_set.append(gpa, k) catch oom();
                        used[k] = true;
                        break;
                    }
                }
            }
            // UNARY + MUL: pull the consuming MUL forward
            if (j > 0 and node(nodes, j).op == c.GGML_OP_UNARY) {
                var k = j + 1;
                while (k < @min(j + 15, n)) : (k += 1) {
                    const mul = node(nodes, k);
                    if (mul.op != c.GGML_OP_MUL or (mul.src[0] != node(nodes, j) and mul.src[1] != node(nodes, j))) continue;
                    const other = (if (mul.src[0] == node(nodes, j)) mul.src[1] else mul.src[0]).?;
                    // the other src must either be weights or already processed
                    if (!(other.*.op == c.GGML_OP_NONE or used_node_set.contains(other))) continue;
                    if (!canFuseUnaryMul(graph, @intCast(j), @intCast(k))) continue;
                    current_set.append(gpa, k) catch oom();
                    used[k] = true;
                    break;
                }
            }
        }

        // Second pass grabs view nodes -- unless that would split add->rms_norm or add->add.
        if (node(nodes, current_set.items[current_set.items.len - 1]).op != c.GGML_OP_ADD) {
            j = first_unused + 1;
            while (j < end) : (j += 1) {
                if (used[j]) continue;
                if (!isViewOp(node(nodes, j))) continue;
                var ok = true;
                for (first_unused..j) |cc| {
                    const in_current = std.mem.indexOfScalar(usize, current_set.items, cc) != null;
                    // skip views whose srcs haven't been processed.
                    if (!used[cc] and isSrcOf(node(nodes, j), node(nodes, cc)) and !in_current) {
                        ok = false;
                        break;
                    }
                }
                if (ok) current_set.append(gpa, j) catch oom();
            }
        }

        // Push the current set into new_order
        for (current_set.items) |cc| {
            new_order.appendAssumeCapacity(node(nodes, cc));
            used_node_set.put(gpa, node(nodes, cc), {}) catch oom();
            used[cc] = true;
        }
        while (first_unused < n and used[first_unused]) first_unused += 1;
    }

    // Replace the graph with the new order.
    for (nodes, new_order.items) |*slot, t| slot.* = t;
}

/// The `add_pattern_alloc_deps` lambda of `ggml_vk_graph_optimize`: keeps
/// the external inputs of the pattern starting at `first` alive through
/// node `last_node`, each source reported once, in source order.
fn addPatternAllocDeps(params: *c.struct_ggml_backend_graph_optimize_params, nodes: []?*c.ggml_tensor, first: usize, len: usize, last_node: usize) void {
    var seen: std.AutoHashMapUnmanaged(*c.ggml_tensor, void) = .empty;
    defer seen.deinit(allocator);
    for (0..len) |j| {
        const nd = nodes[first + j].?;
        for (0..c.GGML_MAX_SRC) |s| {
            const src = nd.src[s] orelse continue;
            const gop = seen.getOrPut(allocator, src) catch oom();
            if (!gop.found_existing) params.add_alloc_dep.?(params.user_data, src, nodes[last_node]);
        }
        seen.put(allocator, nd, {}) catch oom();
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
