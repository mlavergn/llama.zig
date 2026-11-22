//! The CUDA backend's compute loop: op dispatch, CUDA graph capture and
//! replay, concurrent streams, and `graph_optimize`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v1.0.0
//! (`b1b770d90`): `ggml_cuda_compute_forward`, the `USE_CUDA_GRAPH` helpers,
//! `ggml_cuda_graph_evaluate_and_capture`, `ggml_backend_cuda_graph_compute`
//! and `ggml_backend_cuda_graph_optimize`.
//!
//! The op switch is the C's, case for case. While the port was in progress
//! an `isPorted` predicate mirrored it and was AND-ed into `supports_op`;
//! it went at the swap, once `cuda-ops-diff` showed
//! `supports_op` equal to the reference's on every case.
//!
//! Fusion, `ggml_cuda_try_fuse`, is in `fusion.zig`.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");
// Elementwise and small reductions.
// Recurrent-state ops and convolutions.

// Normalisation, softmax, rope and the training ops.
// Flash attention.

// -----------------------------------------------------------------------------
// Dispatch

/// Ports `ggml_cuda_compute_forward` (ggml-cuda.cu:2708 @b1b770d90). ggml has no
/// `GGML_OP_ADD1` or `GGML_OP_DIAG_MASK_INF`.
fn computeForward(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) bool {
    switch (dst.op) {
        ggml.impl.ggml_c.GGML_OP_REPEAT => ggml.cuda.binbcast.opRepeat(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_REPEAT_BACK => ggml.cuda.binbcast.opRepeatBack(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_ADD => ggml.cuda.binbcast.opBinary(.add, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SUB => ggml.cuda.binbcast.opBinary(.sub, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_MUL => ggml.cuda.binbcast.opBinary(.mul, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_DIV => ggml.cuda.binbcast.opBinary(.div, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_MUL_MAT => ggml.cuda.mulmat.mulMat(ctx, ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]), ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]), dst),
        ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID => ggml.cuda.mulmat.mulMatId(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER => ggml.cuda.lightning_indexer.opLightningIndexer(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_TOP_K_MASK => ggml.cuda.top_k_mask.opTopKMask(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT => ggml.cuda.fattn.flashAttnExt(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SCALE => ggml.cuda.scale.opScale(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_NONE, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_PERMUTE, ggml.impl.ggml_c.GGML_OP_TRANSPOSE => {},
        // Elementwise and small reductions, in the C's switch order.
        ggml.impl.ggml_c.GGML_OP_ARGMAX => ggml.cuda.argmax.opArgmax(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL => ggml.cuda.count_equal.opCountEqual(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_ADD_ID => ggml.cuda.add_id.opAddId(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_ACC => ggml.cuda.acc.opAcc(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_UNARY => switch (ggml.impl.ggml_c.ggml_get_unary_op(dst)) {
            ggml.impl.ggml_c.GGML_UNARY_OP_ABS => ggml.cuda.unary.opUnary(.abs, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_SGN => ggml.cuda.unary.opUnary(.sgn, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_NEG => ggml.cuda.unary.opUnary(.neg, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_STEP => ggml.cuda.unary.opUnary(.step, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU => ggml.cuda.unary.opUnary(.gelu, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_SILU => ggml.cuda.unary.opUnary(.silu, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU_ERF => ggml.cuda.unary.opUnary(.gelu_erf, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU_QUICK => ggml.cuda.unary.opUnary(.gelu_quick, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_TANH => ggml.cuda.unary.opUnary(.tanh, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_RELU => ggml.cuda.unary.opUnary(.relu, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID => ggml.cuda.unary.opUnary(.sigmoid, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_HARDSIGMOID => ggml.cuda.unary.opUnary(.hardsigmoid, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_HARDSWISH => ggml.cuda.unary.opUnary(.hardswish, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_EXP => ggml.cuda.unary.opUnary(.exp, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_ELU => ggml.cuda.unary.opUnary(.elu, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_XIELU => ggml.cuda.unary.opXielu(ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_FLOOR => ggml.cuda.unary.opUnary(.floor, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_CEIL => ggml.cuda.unary.opUnary(.ceil, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_ROUND => ggml.cuda.unary.opUnary(.round, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_TRUNC => ggml.cuda.unary.opUnary(.trunc, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_EXPM1 => ggml.cuda.unary.opUnary(.expm1, ctx, dst),
            ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS => ggml.cuda.unary.opUnary(.softplus, ctx, dst),
            else => return false,
        },
        ggml.impl.ggml_c.GGML_OP_GLU => switch (ggml.impl.ggml_c.ggml_get_glu_op(dst)) {
            ggml.impl.ggml_c.GGML_GLU_OP_REGLU => ggml.cuda.unary.opGated(.relu, ctx, dst),
            ggml.impl.ggml_c.GGML_GLU_OP_GEGLU => ggml.cuda.unary.opGated(.gelu, ctx, dst),
            ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU => ggml.cuda.unary.opGated(.silu, ctx, dst),
            ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_OAI => ggml.cuda.unary.opSwigluOai(ctx, dst),
            ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_ERF => ggml.cuda.unary.opGated(.gelu_erf, ctx, dst),
            ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_QUICK => ggml.cuda.unary.opGated(.gelu_quick, ctx, dst),
            ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_CLAMP => ggml.cuda.unary.opSwigluClamp(ctx, dst),
            else => return false,
        },
        ggml.impl.ggml_c.GGML_OP_UPSCALE => ggml.cuda.upscale.opUpscale(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_PAD => ggml.cuda.pad.opPad(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D => ggml.cuda.pad_reflect_1d.opPadReflect1d(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_ARANGE => ggml.cuda.arange.opArange(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_TIMESTEP_EMBEDDING => ggml.cuda.tsembd.opTimestepEmbedding(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_LEAKY_RELU => ggml.cuda.unary.opLeakyRelu(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SILU_BACK => ggml.cuda.unary.opSiluBack(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SQR => ggml.cuda.unary.opUnary(.sqr, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SQRT => ggml.cuda.unary.opUnary(.sqrt, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SIN => ggml.cuda.unary.opUnary(.sin, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_COS => ggml.cuda.unary.opUnary(.cos, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CLAMP => ggml.cuda.clamp.opClamp(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_LOG => ggml.cuda.unary.opUnary(.log, ctx, dst),
        ggml.impl.ggml_c.GGML_OP_DIAG => ggml.cuda.diag.opDiag(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_ROLL => ggml.cuda.roll.opRoll(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SUM_ROWS => ggml.cuda.sumrows.opSumRows(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_MEAN => ggml.cuda.mean.opMean(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SUM => ggml.cuda.sum.opSum(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_TRI => ggml.cuda.tri.opTri(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_FILL => ggml.cuda.fill.opFill(ctx, dst),
        // Normalisation, softmax, rope and the training ops.
        ggml.impl.ggml_c.GGML_OP_NORM => ggml.cuda.norm.opNorm(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_GROUP_NORM => ggml.cuda.norm.opGroupNorm(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_L2_NORM => ggml.cuda.norm.opL2Norm(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_RMS_NORM => ggml.cuda.norm.opRmsNorm(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_RMS_NORM_BACK => ggml.cuda.norm.opRmsNormBack(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX => ggml.cuda.softmax.opSoftMax(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK => ggml.cuda.softmax.opSoftMaxBack(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_ROPE => ggml.cuda.rope.opRope(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_ROPE_BACK => ggml.cuda.rope.opRopeBack(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS => ggml.cuda.cross_entropy_loss.opCrossEntropyLoss(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => ggml.cuda.cross_entropy_loss.opCrossEntropyLossBack(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_ADAMW => ggml.cuda.opt_step_adamw.opOptStepAdamw(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_SGD => ggml.cuda.opt_step_sgd.opOptStepSgd(ctx, dst),
        // Recurrent-state ops and convolutions.
        ggml.impl.ggml_c.GGML_OP_OUT_PROD => ggml.cuda.out_prod.outProd(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_IM2COL => ggml.cuda.im2col.opIm2col(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_IM2COL_3D => ggml.cuda.im2col.opIm2col3d(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CONV_2D => ggml.cuda.conv2d.opConv2d(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CONV_3D => ggml.cuda.conv3d.opConv3d(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CONV_2D_DW => ggml.cuda.conv2d_dw.opConv2dDw(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D => ggml.cuda.conv2d_transpose.conv2dTransposeP0(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D => ggml.cuda.conv_transpose_1d.opConvTranspose1d(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_COL2IM_1D => ggml.cuda.col2im_1d.opCol2im1d(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_POOL_2D => ggml.cuda.pool2d.opPool2d(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_POOL_1D => ggml.cuda.pool1d.opPool1d(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SSM_CONV => ggml.cuda.ssm_conv.opSsmConv(ctx, dst, null, null),
        ggml.impl.ggml_c.GGML_OP_SSM_SCAN => ggml.cuda.ssm_scan.opSsmScan(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV6 => ggml.cuda.wkv.opRwkvWkv6(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_GATED_LINEAR_ATTN => ggml.cuda.gla.opGatedLinearAttn(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET => ggml.cuda.gated_delta_net.opGatedDeltaNet(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_COMB => ggml.cuda.dsv4_hc.opDsv4HcComb(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_PRE => ggml.cuda.dsv4_hc.opDsv4HcPre(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST => ggml.cuda.dsv4_hc.opDsv4HcPost(ctx, dst, null, null),
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV7 => ggml.cuda.wkv.opRwkvWkv7(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SOLVE_TRI => ggml.cuda.solve_tri.opSolveTri(ctx, dst),
        // Data movement, indexing and sorting.
        ggml.impl.ggml_c.GGML_OP_GET_ROWS => ggml.cuda.getrows.opGetRows(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK => ggml.cuda.getrows.opGetRowsBack(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SET_ROWS => ggml.cuda.set_rows.opSetRows(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_SET => ggml.cuda.set.opSet(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_DUP => ggml.cuda.cpy.dup(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CPY => ggml.cuda.cpy.cpy(ctx, ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]), ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1])),
        ggml.impl.ggml_c.GGML_OP_CONT => ggml.cuda.cpy.dup(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CONCAT => ggml.cuda.concat.opConcat(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_CUMSUM => ggml.cuda.cumsum.opCumsum(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_TOP_K => ggml.cuda.top_k.opTopK(ctx, dst),
        ggml.impl.ggml_c.GGML_OP_ARGSORT => ggml.cuda.argsort.opArgsort(ctx, dst),
        else => return false,
    }

    const err = ggml.cuda.common.cu.cudaGetLastError();
    if (err != ggml.cuda.common.cu.cudaSuccess) {
        ggml.impl.logError("%s: %s failed\n", .{ "ggml_cuda_compute_forward", ggml.impl.ggml_c.ggml_op_desc(dst) });
        ggml.cuda.common.cudaCheck(err, "err", @src());
    }
    return true;
}

/// Ports `ggml_cuda_try_fuse`; see `fusion.zig`.
/// Ports `ggml_cuda_mul_mat_id_needs_sync`; see `mulmat.zig`.

// -----------------------------------------------------------------------------
// CUDA graphs

/// Ports `ggml_cuda_is_view_or_noop` (ggml-cuda.cu:3195 @b1b770d90).
fn isViewOrNoop(t: *const ggml.impl.ggml_c.ggml_tensor) bool {
    return ggml.impl.ggml_c.ggml_is_empty(t) or t.op == ggml.impl.ggml_c.GGML_OP_RESHAPE or t.op == ggml.impl.ggml_c.GGML_OP_TRANSPOSE or
        t.op == ggml.impl.ggml_c.GGML_OP_VIEW or t.op == ggml.impl.ggml_c.GGML_OP_PERMUTE or t.op == ggml.impl.ggml_c.GGML_OP_NONE;
}

fn nodes(cgraph: *ggml.impl.CGraph) []*ggml.impl.ggml_c.ggml_tensor {
    return @ptrCast(cgraph.nodes[0..@intCast(cgraph.n_nodes)]);
}

/// Ports `ggml_cuda_graph_check_compability` (ggml-cuda.cu:3206 @b1b770d90).
/// Its debug line is gated on `GGML_CUDA_GRAPH_DEBUG`.
fn graphCheckCompatibility(cgraph: *ggml.impl.CGraph) bool {
    var use_cuda_graph = true;
    for (nodes(cgraph)) |node| {
        if (isViewOrNoop(node)) continue;

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID) {
            const x = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].cc;
            if (ggml.cuda.mulmat.mulMatIdNeedsSync(node, x)) {
                // the mul_mat_id fallback path synchronizes the stream, so
                // CUDA graphs cannot be used
                use_cuda_graph = false;
                if (ggml.cuda.common.graphDebugEnabled()) {
                    ggml.impl.logDebug("%s: disabling CUDA graphs due to unsupported node type\n", .{"ggml_cuda_graph_check_compability"});
                }
            }
        }

        if (!use_cuda_graph) break;
    }
    return use_cuda_graph;
}

/// Ports `ggml_cuda_graph_get_key` (ggml-cuda.cu:3239 @b1b770d90).
fn graphGetKey(cgraph: *ggml.impl.CGraph) ?*const anyopaque {
    return cgraph.nodes[0];
}

/// Ports `ggml_cuda_graph_update_required` (ggml-cuda.cu:3243 @b1b770d90).
fn graphUpdateRequired(cuda_ctx: *ggml.cuda.common.Context, cgraph: *ggml.impl.CGraph) bool {
    var res = false;

    const graph = cuda_ctx.cudaGraph(graphGetKey(cgraph));

    if (cgraph.uid != 0 and cgraph.uid == graph.uid) {
        if (ggml.cuda.common.graphDebugEnabled()) {
            ggml.impl.logDebug("CUDA Graph id %zu reused\n", .{cgraph.uid});
        }
        ggml.impl.assert(graph.node_props.items.len == @as(usize, @intCast(cgraph.n_nodes)), "(int)graph->node_props.size() == cgraph->n_nodes");
        return false;
    }

    graph.uid = cgraph.uid;

    const n: usize = @intCast(cgraph.n_nodes);
    // Check if the graph size has changed
    if (graph.node_props.items.len != n) {
        res = true;
        // `resize` value-initialises the new elements.
        const old = graph.node_props.items.len;
        graph.node_props.resize(std.heap.c_allocator, n) catch ggml.impl.abort("ggml-cuda: out of memory");
        if (n > old) @memset(std.mem.sliceAsBytes(graph.node_props.items[old..]), 0);
    }

    for (nodes(cgraph), 0..) |node, i| {
        var prop: ggml.cuda.common.NodeProperties = std.mem.zeroes(ggml.cuda.common.NodeProperties);
        prop.node = node.*;

        for (0..ggml.impl.ggml_c.GGML_MAX_SRC) |j| {
            if (node.src[j]) |s| {
                prop.node_src_data_ptrs[j] = s.*.data;
                prop.node_src_ne[j] = s.*.ne;
                prop.node_src_nb[j] = s.*.nb;
            }
        }

        if (res or !std.mem.eql(u8, std.mem.asBytes(&graph.node_props.items[i]), std.mem.asBytes(&prop))) {
            graph.node_props.items[i] = prop;
            res = true;
        }
    }

    return res;
}

/// `ggml_cuda_graph_get_clusters`'s function-local `runtime_affected`: the
/// CUDA runtimes whose `cudaGraphExecUpdate` keeps stale cluster dims.
const ClusterRuntime = struct {
    var affected: bool = false;
    fn init() void {
        var version: c_int = 0;
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion(&version)", @src());
        // TODO: upper bound once fixed in the CUDA runtime
        affected = version >= 11080;
    }
    const once = ggml.cuda.common.Once(init);
};

/// Ports `ggml_cuda_graph_get_clusters` (ggml-cuda.cu:3288 @b1b770d90)
/// (#31284), under `USE_CUDA_GRAPH_CLUSTER_CHECK`: the cluster dims
/// of every clustered kernel node, on Hopper and later.
fn graphGetClusters(cuda_graph: ggml.cuda.common.cu.cudaGraph_t, device: c_int) std.ArrayList(ggml.cuda.common.GraphCluster) {
    var clusters: std.ArrayList(ggml.cuda.common.GraphCluster) = .empty;
    ClusterRuntime.once.call();
    if (!ClusterRuntime.affected or ggml.cuda.common.info().devices[@intCast(device)].cc < ggml.cuda.common.cc.hopper) {
        return clusters;
    }
    var n_nodes: usize = 0;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphGetNodes(cuda_graph, null, &n_nodes), "cudaGraphGetNodes(cuda_graph, nullptr, &n_nodes)", @src());
    const graph_nodes = std.heap.c_allocator.alloc(ggml.cuda.common.cu.cudaGraphNode_t, n_nodes) catch ggml.impl.abort("ggml-cuda: out of memory");
    defer std.heap.c_allocator.free(graph_nodes);
    // An empty `std::vector`'s `data()` is null; a zero-length Zig slice's
    // pointer is not, and the runtime refuses it.
    const nodes_ptr: ?[*]ggml.cuda.common.cu.cudaGraphNode_t = if (n_nodes == 0) null else graph_nodes.ptr;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphGetNodes(cuda_graph, nodes_ptr, &n_nodes), "cudaGraphGetNodes(cuda_graph, nodes.data(), &n_nodes)", @src());
    for (0..n_nodes) |i| {
        var ty: ggml.cuda.common.cu.enum_cudaGraphNodeType = undefined;
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphNodeGetType(graph_nodes[i], &ty), "cudaGraphNodeGetType(nodes[i], &type)", @src());
        if (ty != ggml.cuda.common.cu.cudaGraphNodeTypeKernel) continue;
        var val: ggml.cuda.common.cu.cudaLaunchAttributeValue = std.mem.zeroes(ggml.cuda.common.cu.cudaLaunchAttributeValue);
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphKernelNodeGetAttribute(graph_nodes[i], ggml.cuda.common.cu.cudaLaunchAttributeClusterDimension, &val), "cudaGraphKernelNodeGetAttribute(nodes[i], cudaLaunchAttributeClusterDimension, &val)", @src());
        const d = val.clusterDim;
        if (d.x *% d.y *% d.z > 1) {
            clusters.append(std.heap.c_allocator, .{ .node = i, .x = d.x, .y = d.y, .z = d.z }) catch ggml.impl.abort("ggml-cuda: out of memory");
        }
    }
    return clusters;
}

/// Ports `ggml_cuda_graph_update_executable` (ggml-cuda.cu:3319 @b1b770d90),
/// the CUDA 12 arm. It takes one chunk's executable and
/// captured graph. Its debug line is `GGML_CUDA_GRAPH_DEBUG`'s. A chunk whose
/// kernel cluster dims changed is re-instantiated rather than updated.
fn graphUpdateExecutable(instance: *ggml.cuda.common.GraphInstance, cuda_graph: ggml.cuda.common.cu.cudaGraph_t, device: c_int) void {
    {
        var clusters = graphGetClusters(cuda_graph, device);
        if (!instance.clustersEql(clusters.items)) {
            if (ggml.cuda.common.graphDebugEnabled()) {
                ggml.impl.logDebug("%s: cluster dims changed, re-instantiating CUDA graph\n", .{"ggml_cuda_graph_update_executable"});
            }
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphExecDestroy(instance.handle), "cudaGraphExecDestroy(instance)", @src());
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphInstantiate(&instance.handle, cuda_graph, 0), "cudaGraphInstantiate(&instance.handle, cuda_graph, nullptr, nullptr, 0)", @src());
            instance.clusters.deinit(std.heap.c_allocator);
            instance.clusters = clusters;
            return;
        }
        clusters.deinit(std.heap.c_allocator);
    }

    var result_info: ggml.cuda.common.cu.cudaGraphExecUpdateResultInfo = undefined;
    const stat = ggml.cuda.common.cu.cudaGraphExecUpdate(instance.handle, cuda_graph, &result_info);

    if (stat == ggml.cuda.common.cu.cudaErrorGraphExecUpdateFailure) {
        if (ggml.cuda.common.graphDebugEnabled()) {
            ggml.impl.logDebug("%s: CUDA graph update failed\n", .{"ggml_cuda_graph_update_executable"});
        }

        // The pre-existing graph exec cannot be updated due to violated
        // constraints, so clear the error and re-instantiate.
        _ = ggml.cuda.common.cu.cudaGetLastError();
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphExecDestroy(instance.handle), "cudaGraphExecDestroy(instance)", @src());
        instance.handle = null;
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphInstantiate(&instance.handle, cuda_graph, 0), "cudaGraphInstantiate(&instance.handle, cuda_graph, nullptr, nullptr, 0)", @src());
    } else {
        ggml.impl.assert(stat == ggml.cuda.common.cu.cudaSuccess, "stat == cudaSuccess");
    }
}

const GraphChunk = struct {
    var size: c_int = 512;
    fn init() void {
        if (stdlib.env.getenv("GGML_CUDA_GRAPH_CHUNK_SIZE")) |env| size = stdlib.num.atoiC(env);
    }
    const once = ggml.cuda.common.Once(init);
};

/// Ports `ggml_cuda_graph_chunk_size` (ggml-cuda.cu:3361 @b1b770d90): the
/// number of ggml nodes per CUDA graph chunk; 0 captures the whole ggml
/// graph as one CUDA graph.
fn graphChunkSize() c_int {
    GraphChunk.once.call();
    return GraphChunk.size;
}

/// Ports `ggml_cuda_graph_launch_chunk` (ggml-cuda.cu:3370 @b1b770d90): end
/// the capture of chunk `idx`, update or create its executable and launch it.
/// A new executable records its cluster dims.
fn graphLaunchChunk(cuda_ctx: *ggml.cuda.common.Context, graph: *ggml.cuda.common.Graph, idx: usize) void {
    var cuda_graph: ggml.cuda.common.cu.cudaGraph_t = null;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamEndCapture(cuda_ctx.stream(), &cuda_graph), "cudaStreamEndCapture(cuda_ctx->stream(), &cuda_graph)", @src());

    if (idx < graph.instances.items.len) {
        graphUpdateExecutable(&graph.instances.items[idx], cuda_graph, cuda_ctx.device);
    } else {
        ggml.impl.assert(idx == graph.instances.items.len, "idx == graph->instances.size()");
        graph.instances.append(std.heap.c_allocator, .{}) catch ggml.impl.abort("ggml-cuda: out of memory");
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphInstantiate(&graph.instances.items[idx].handle, cuda_graph, 0), "cudaGraphInstantiate(&graph->instances[idx].handle, cuda_graph, nullptr, nullptr, 0)", @src());
        graph.instances.items[idx].clusters = graphGetClusters(cuda_graph, cuda_ctx.device);
    }
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphDestroy(cuda_graph), "cudaGraphDestroy(cuda_graph)", @src());

    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphLaunch(graph.instances.items[idx].handle, cuda_ctx.stream()), "cudaGraphLaunch(graph->instances[idx], cuda_ctx->stream())", @src());
}

/// Ports `ggml_cuda_graph_set_enabled` (ggml-cuda.cu:5402 @b1b770d90).
fn graphSetEnabled(cuda_ctx: *ggml.cuda.common.Context, graph_key: ?*const anyopaque) bool {
    const graph = cuda_ctx.cudaGraph(graph_key);

    if (graph.instances.items.len == 0) {
        if (ggml.cuda.common.info().devices[@intCast(cuda_ctx.device)].cc < ggml.cuda.common.cc.volta) {
            if (!graph.disable_due_to_gpu_arch and ggml.cuda.common.graphDebugEnabled()) {
                ggml.impl.logDebug("%s: disabling CUDA graphs due to GPU architecture\n", .{"ggml_cuda_graph_set_enabled"});
            }
            graph.disable_due_to_gpu_arch = true;
        }
    }

    return graph.isEnabled();
}

// -----------------------------------------------------------------------------
// Evaluation

/// The C's `try_launch_concurrent_event` lambda and the state it captures.
const Concurrency = struct {
    cuda_ctx: *ggml.cuda.common.Context,
    is_active: bool = false,
    event: ?*ggml.cuda.common.ConcurrentEvent = null,

    fn tryLaunch(self: *Concurrency, node: *const ggml.impl.ggml_c.ggml_tensor) void {
        const stream_ctx = &self.cuda_ctx.concurrent_stream_context;
        const ev = stream_ctx.concurrent_events.getPtr(node) orelse return;
        self.event = ev;
        self.is_active = true;

        ggml.impl.logDebug("Launching %d streams at %s\n", .{ ev.n_streams, @as([*:0]const u8, @ptrCast(&node.name)) });

        const main_stream = self.cuda_ctx.stream(); // this should be stream 0
        ggml.impl.assert(self.cuda_ctx.curr_stream_no == 0, "cuda_ctx->curr_stream_no == 0");
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaEventRecord(ev.fork_event, main_stream), "cudaEventRecord(concurrent_event->fork_event, main_stream)", @src());

        var i: c_int = 1;
        while (i <= ev.n_streams) : (i += 1) {
            const s = self.cuda_ctx.streamFor(self.cuda_ctx.device, i);
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamWaitEvent(s, ev.fork_event, 0), "cudaStreamWaitEvent(stream, concurrent_event->fork_event)", @src());
        }
    }
};

/// Ports `ggml_cuda_graph_evaluate_and_capture` (ggml-cuda.cu:5172 @b1b770d90).
/// The capture ends and restarts every `graphChunkSize` nodes,
/// each chunk launched as it ends, so the GPU runs one while the next is
/// enqueued.
fn graphEvaluateAndCapture(cuda_ctx: *ggml.cuda.common.Context, cgraph: *ggml.impl.CGraph, use_cuda_graph: bool, cuda_graph_update_required: bool, graph_key: ?*const anyopaque) void {
    var graph_evaluated_or_captured = false;

    const stream_ctx = &cuda_ctx.concurrent_stream_context;
    var conc: Concurrency = .{ .cuda_ctx = cuda_ctx };
    var should_launch_concurrent_events = false;

    const graph: ?*ggml.cuda.common.Graph = if (use_cuda_graph) cuda_ctx.cudaGraph(graph_key) else null;
    const chunk_size = graphChunkSize();
    var chunk_end: c_int = chunk_size;
    var n_chunks: usize = 0;

    while (!graph_evaluated_or_captured) {
        // Only evaluate directly if CUDA graphs are off, or while capturing.
        if (!use_cuda_graph or cuda_graph_update_required) {
            var prev_i: usize = 0;

            if (stream_ctx.concurrent_events.count() > 0) {
                should_launch_concurrent_events = true;
                for (stream_ctx.concurrent_events.values()) |*ev| {
                    should_launch_concurrent_events = should_launch_concurrent_events and ev.isValid();
                }
            }

            if (should_launch_concurrent_events) {
                restoreOriginalOrder(cuda_ctx, cgraph);
            } else {
                stream_ctx.reset();
            }

            const ns = nodes(cgraph);
            var i: usize = 0;
            while (i < ns.len) : (i += 1) {
                const node = ns[i];
                // fused nodes are skipped together, so a cut never splits a fusion
                // a concurrent region must join before its capture can end
                if (use_cuda_graph and chunk_size > 0 and @as(c_int, @intCast(i)) >= chunk_end and !conc.is_active) {
                    graphLaunchChunk(cuda_ctx, graph.?, n_chunks);
                    n_chunks += 1;
                    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamBeginCapture(cuda_ctx.stream(), ggml.cuda.common.cu.cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed)", @src());
                    chunk_end = @as(c_int, @intCast(i)) + chunk_size;
                }
                if (conc.is_active) {
                    const ev = conc.event.?;
                    if (node == ev.join_node) {
                        cuda_ctx.curr_stream_no = 0;
                        var s: c_int = 1;
                        while (s <= ev.n_streams) : (s += 1) {
                            // Wait on join events of forked streams in the main stream
                            const je = ev.join_events.items[@intCast(s - 1)];
                            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaEventRecord(je, cuda_ctx.streamFor(cuda_ctx.device, s)), "cudaEventRecord(concurrent_event->join_events[i - 1], cuda_ctx->stream(cuda_ctx->device, i))", @src());
                            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamWaitEvent(cuda_ctx.stream(), je, 0), "cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1])", @src());
                        }
                        conc.is_active = false;
                        conc.event = null;
                    } else {
                        const mapped = ev.stream_mapping.get(node);
                        ggml.impl.assert(mapped != null, "concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end()");
                        cuda_ctx.curr_stream_no = mapped.?;
                        ggml.impl.logDebug("Setting stream no to %d for node %s\n", .{ cuda_ctx.curr_stream_no, @as([*:0]const u8, @ptrCast(&node.name)) });
                    }
                } else if (i -% prev_i > 1 and i > prev_i) {
                    // the previous node was fused
                    conc.tryLaunch(ns[i - 1]);
                    if (conc.is_active) {
                        // `stream_mapping[node]` inserts 0 when absent.
                        const gop = conc.event.?.stream_mapping.getOrPut(std.heap.c_allocator, node) catch ggml.impl.abort("ggml-cuda: out of memory");
                        if (!gop.found_existing) gop.value_ptr.* = 0;
                        cuda_ctx.curr_stream_no = gop.value_ptr.*;
                        ggml.impl.logDebug("Setting stream no to %d for node %s\n", .{ cuda_ctx.curr_stream_no, @as([*:0]const u8, @ptrCast(&node.name)) });
                    }
                }

                prev_i = i;

                if (isViewOrNoop(node)) continue;
                if ((node.flags & ggml.impl.ggml_c.GGML_TENSOR_FLAG_COMPUTE) == 0) continue;

                const nodes_to_skip = ggml.cuda.fusion.tryFuse(cuda_ctx, cgraph, @intCast(i));
                if (nodes_to_skip != 0) {
                    i += @intCast(nodes_to_skip);
                    continue;
                }

                const ok = computeForward(cuda_ctx, node);
                if (!ok) {
                    ggml.impl.logError("%s: op not supported %s (%s)\n", .{ "ggml_cuda_graph_evaluate_and_capture", @as([*:0]const u8, @ptrCast(&node.name)), ggml.impl.ggml_c.ggml_op_name(node.op) });
                }
                ggml.impl.assert(ok, "ok");

                if (!conc.is_active) conc.tryLaunch(node);
            }
        }

        if (use_cuda_graph and cuda_graph_update_required) { // End CUDA graph capture
            const g = graph.?;
            graphLaunchChunk(cuda_ctx, g, n_chunks);
            n_chunks += 1;

            // drop the chunks left over from a longer graph
            var i: usize = n_chunks;
            while (i < g.instances.items.len) : (i += 1) {
                ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphExecDestroy(g.instances.items[i].handle), "cudaGraphExecDestroy(graph->instances[i])", @src());
                g.instances.items[i].deinit();
            }
            g.instances.resize(std.heap.c_allocator, n_chunks) catch ggml.impl.abort("ggml-cuda: out of memory");
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            ggml.cuda.common.lock.release();
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph and !cuda_graph_update_required) {
        // the chunks were launched during capture, otherwise launch them here
        for (graph.?.instances.items) |instance| {
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGraphLaunch(instance.handle, cuda_ctx.stream()), "cudaGraphLaunch(instance, cuda_ctx->stream())", @src());
        }
    }
}

/// The C's "restore original node order within each concurrent region"
/// block in `ggml_cuda_graph_evaluate_and_capture` (ggml-cuda.cu:5172 @b1b770d90).
fn restoreOriginalOrder(cuda_ctx: *ggml.cuda.common.Context, cgraph: *ggml.impl.CGraph) void {
    var node_to_idx: std.AutoHashMapUnmanaged(*const ggml.impl.ggml_c.ggml_tensor, usize) = .empty;
    defer node_to_idx.deinit(std.heap.c_allocator);
    const ns = nodes(cgraph);
    for (ns, 0..) |node, i| node_to_idx.put(std.heap.c_allocator, node, i) catch ggml.impl.abort("ggml-cuda: out of memory");

    for (cuda_ctx.concurrent_stream_context.concurrent_events.values()) |*ev| {
        var positions: std.ArrayList(usize) = .empty;
        defer positions.deinit(std.heap.c_allocator);

        var all_found = true;
        for (ev.original_order.items) |orig| {
            if (node_to_idx.get(orig)) |idx| {
                positions.append(std.heap.c_allocator, idx) catch ggml.impl.abort("ggml-cuda: out of memory");
            } else {
                all_found = false;
                break;
            }
        }
        if (!all_found or positions.items.len != ev.original_order.items.len) continue;

        std.mem.sort(usize, positions.items, {}, std.sort.asc(usize));

        var is_contiguous = true;
        for (1..positions.items.len) |i| {
            if (positions.items[i] != positions.items[i - 1] + 1) {
                is_contiguous = false;
                break;
            }
        }
        if (!is_contiguous) continue;

        const start_pos = positions.items[0];
        for (ev.original_order.items, 0..) |orig, i| ns[start_pos + i] = @constCast(orig);
    }
}

/// Ports `ggml_backend_cuda_graph_compute` (ggml-cuda.cu:5418 @b1b770d90).
/// A graph reused under the same `uid` skips the compatibility
/// check: `graph->uid` is set only for a compatible graph.
pub fn graphCompute(backend_c: ggml.impl.ggml_c.ggml_backend_t, cgraph_c: ?*ggml.impl.ggml_c.struct_ggml_cgraph) callconv(.c) ggml.impl.ggml_c.enum_ggml_status {
    const cuda_ctx = ggml.cuda.backend.backendCtx(backend_c);
    const cgraph: *ggml.impl.CGraph = @ptrCast(@alignCast(cgraph_c));

    ggml.cuda.common.setDevice(cuda_ctx.device);

    var use_cuda_graph = false;
    var cuda_graph_update_required = false;

    const graph_key = graphGetKey(cgraph);

    _ = graphSetEnabled(cuda_ctx, graph_key);

    const graph = cuda_ctx.cudaGraph(graph_key);
    if (graph.isEnabled()) {
        // graph->uid is set only for a compatible graph (see ggml_cuda_graph_update_required), so a reused graph needs no new check
        const graph_reused = cgraph.uid != 0 and cgraph.uid == graph.uid;
        const graph_compatible = graph_reused or graphCheckCompatibility(cgraph);
        if (graph_compatible) {
            const properties_changed = graphUpdateRequired(cuda_ctx, cgraph);

            if (!graph.warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph.warmup_complete = true;
                    if (ggml.cuda.common.graphDebugEnabled()) {
                        ggml.impl.logDebug("%s: CUDA graph warmup complete\n", .{"ggml_backend_cuda_graph_compute"});
                    }
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph.warmup_complete = false;
                    if (ggml.cuda.common.graphDebugEnabled()) {
                        ggml.impl.logDebug("%s: CUDA graph warmup reset\n", .{"ggml_backend_cuda_graph_compute"});
                    }
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph.instances.items.len == 0;
                }
            }
        }
    }

    if (use_cuda_graph and cuda_graph_update_required) {
        // Start CUDA graph capture
        ggml.cuda.common.lock.acquire();
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamBeginCapture(cuda_ctx.stream(), ggml.cuda.common.cu.cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed)", @src());
    }

    graphEvaluateAndCapture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return ggml.impl.ggml_c.GGML_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------------
// Graph optimisation

fn isNoop(node: *const ggml.impl.ggml_c.ggml_tensor) bool {
    return ggml.impl.ggml_c.ggml_is_empty(node) or node.op == ggml.impl.ggml_c.GGML_OP_NONE or node.op == ggml.impl.ggml_c.GGML_OP_RESHAPE or
        node.op == ggml.impl.ggml_c.GGML_OP_TRANSPOSE or node.op == ggml.impl.ggml_c.GGML_OP_VIEW or node.op == ggml.impl.ggml_c.GGML_OP_PERMUTE;
}

fn dependsOn(dst: *const ggml.impl.ggml_c.ggml_tensor, src: *const ggml.impl.ggml_c.ggml_tensor) bool {
    for (dst.src) |s| {
        if (s == src) return true;
    }
    // implicit dependency if they view the same tensor
    const dst2: *const ggml.impl.ggml_c.ggml_tensor = if (dst.view_src) |v| v else dst;
    const src2: *const ggml.impl.ggml_c.ggml_tensor = if (src.view_src) |v| v else src;
    return dst2 == src2;
}

fn belongsToBranch(node: *const ggml.impl.ggml_c.ggml_tensor, branch: []const *const ggml.impl.ggml_c.ggml_tensor) bool {
    for (branch) |n| {
        if (dependsOn(node, n)) return true;
    }
    return false;
}

fn appendOps(buf: *[16]ggml.impl.ggml_c.enum_ggml_op, n: *usize, items: []const ggml.impl.ggml_c.enum_ggml_op) void {
    for (items) |x| {
        buf[n.*] = x;
        n.* += 1;
    }
}

/// `graph_optimize`'s `add_alloc_deps` lambda (ggml-cuda.cu:5513 @b1b770d90):
/// keeps every node in `[start, last_node)` and each of its sources alive
/// until `nodes[last_node]`.
fn addAllocDeps(
    cgraph: *const ggml.impl.CGraph,
    params: *const ggml.impl.ggml_c.struct_ggml_backend_graph_optimize_params,
    start: usize,
    last_node: usize,
) void {
    const add_alloc_dep = params.add_alloc_dep.?;
    const user_data = params.user_data;
    const last = cgraph.nodes[last_node];
    var i = start;
    while (i < last_node) : (i += 1) {
        const n = cgraph.nodes[i].?;
        add_alloc_dep(user_data, n, last);
        for (n.src) |s| {
            if (s) |sp| add_alloc_dep(user_data, sp, last);
        }
    }
}

const GraphOpt = struct {
    var enabled: bool = false;
    fn init() void {
        const env = stdlib.env.getenv("GGML_CUDA_GRAPH_OPT");
        enabled = env != null and stdlib.num.atoiC(env.?) == 1;
    }
    const once = ggml.cuda.common.Once(init);
};

/// Ports `ggml_backend_cuda_graph_optimize` (ggml-cuda.cu:5508 @b1b770d90),
/// with its `add_alloc_deps` lambda.
///
/// Its first pass moves a shared expert's gate/up/GLU, wherever
/// it follows a routed expert's `MUL_MAT_ID`/`MUL_MAT_ID`/`GLU` before the
/// next routed pair, to right after it -- `std::rotate` of the nodes in
/// between -- so `tryFuse` sees the six nodes together, and keeps the
/// routed input and ids alive until the shared GLU.
///
/// Its second pass runs whether or not CUDA graphs are on. The
/// inputs of every gate/up/GLU `fusion.zig` could send to the fused MMQ
/// kernel are kept alive until the GLU, so they cannot alias its output
/// (under op offload the weights are buffer copies). Every
/// MoE weighted reduction `fusion.zig` would fuse reads its experts, router
/// weights and expert scale at the chain's last node, after the scheduler
/// thinks they are dead, so each is kept allocated until then through
/// `params.add_alloc_dep`. Every top-k MoE chain `tryFuse` would fuse
/// without a delayed softmax gets the same treatment: each of its nodes and
/// their sources are kept alive until the node *after* the chain. That is
/// `nodes[i + ops.size()]`, one past the chain's last node, as the C
/// indexes it -- `last_node` names the end of the range, not its last
/// element -- and it reads `nodes[n_nodes]` for a chain that ends the
/// graph, as the C does. Unlike `tryFuse`, the C checks no memory ranges
/// here.
///
/// The C walks `fan_out`, a `std::unordered_map`, to pick fork nodes, and
/// handles the first qualifying one in each range; which is "first" is
/// libstdc++'s iteration order. With `min_fan_out == max_fan_out == 3` and
/// only `attn_norm` nodes eligible, the forks found are disjoint, so the
/// order affects only which region a skipped overlap would have kept --
/// and overlapping ranges cannot occur, since each fork's range ends at its
/// own join.
pub fn graphOptimize(backend_c: ggml.impl.ggml_c.ggml_backend_t, cgraph_c: ?*ggml.impl.ggml_c.struct_ggml_cgraph, params_c: ?*ggml.impl.ggml_c.struct_ggml_backend_graph_optimize_params) callconv(.c) void {
    const cuda_ctx = ggml.cuda.backend.backendCtx(backend_c);
    const cgraph: *ggml.impl.CGraph = @ptrCast(@alignCast(cgraph_c));

    const disable_fusion = ggml.cuda.common.fusionDisabled();
    if (!disable_fusion) {
        const params = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_graph_optimize_params, params_c);
        const add_alloc_dep = params.add_alloc_dep.?;
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml.cuda.common.setDevice(cuda_ctx.device);
        {
            var r: c_int = 0;
            while (r + 5 < cgraph.n_nodes) : (r += 1) {
                if (cgraph.nodes[@intCast(r)].?.op != ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID) {
                    continue;
                }
                var j: c_int = r + 3;
                while (j + 2 < cgraph.n_nodes) : (j += 1) {
                    if (cgraph.nodes[@intCast(j)].?.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID and cgraph.nodes[@intCast(j + 1)].?.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID) {
                        break;
                    }
                    if (cgraph.nodes[@intCast(j)].?.op != ggml.impl.ggml_c.GGML_OP_MUL_MAT or !ggml.cuda.fusion.matchSharedExpert(cgraph, r, j)) {
                        continue;
                    }
                    // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                    std.mem.rotate(?*ggml.impl.ggml_c.ggml_tensor, cgraph.nodes[@intCast(r + 3)..@intCast(j + 3)], @intCast(j - (r + 3)));
                    const up = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, cgraph.nodes[@intCast(r + 2)].?.src[1]);
                    add_alloc_dep(params.user_data, up.src[1], cgraph.nodes[@intCast(r + 5)]);
                    add_alloc_dep(params.user_data, up.src[2], cgraph.nodes[@intCast(r + 5)]);
                    r += 5;
                    break;
                }
            }
        }
        var i: c_int = 0;
        while (i < cgraph.n_nodes) : (i += 1) {
            // Keep the inputs of the fused gate/up/GLU MMQ kernel alive until the GLU so that they cannot alias its output (under op offload the weights are buffer copies).
            if (i + 2 < cgraph.n_nodes and cgraph.nodes[@intCast(i + 2)].?.op == ggml.impl.ggml_c.GGML_OP_GLU) {
                const glu = cgraph.nodes[@intCast(i + 2)].?;
                const gate: ?*const ggml.impl.ggml_c.ggml_tensor = glu.src[0];
                const up: ?*const ggml.impl.ggml_c.ggml_tensor = glu.src[1];
                const n0: ?*const ggml.impl.ggml_c.ggml_tensor = cgraph.nodes[@intCast(i)];
                const n1: ?*const ggml.impl.ggml_c.ggml_tensor = cgraph.nodes[@intCast(i + 1)];
                const ok = (gate == n0 and up == n1) or (gate == n1 and up == n0);
                if (ok and ggml.cuda.mulmat.mulMatQFusionMatches(up.?, gate.?, glu, cuda_ctx.device, null, null)) {
                    addAllocDeps(cgraph, params, @intCast(i), @intCast(i + 2));
                    i += 2;
                    continue;
                }
            }
            // Same with the NVFP4 scales: MUL_MAT(gate), MUL(gate scale), MUL_MAT(up), MUL(up scale), GLU.
            if (i + 4 < cgraph.n_nodes and cgraph.nodes[@intCast(i + 4)].?.op == ggml.impl.ggml_c.GGML_OP_GLU) {
                const glu = cgraph.nodes[@intCast(i + 4)].?;
                const gate_scale = cgraph.nodes[@intCast(i + 1)].?;
                const up_scale = cgraph.nodes[@intCast(i + 3)].?;
                const mm_gate = cgraph.nodes[@intCast(i)].?;
                const mm_up = cgraph.nodes[@intCast(i + 2)].?;
                if (glu.src[0] == gate_scale and glu.src[1] == up_scale and
                    ggml.cuda.mulmat.mulMatQFusionMatches(mm_up, mm_gate, glu, cuda_ctx.device, up_scale, gate_scale) and
                    ggml.cuda.mulmat.getMulMatScale(gate_scale, mm_gate) != null and ggml.cuda.mulmat.getMulMatScale(up_scale, mm_up) != null)
                {
                    addAllocDeps(cgraph, params, @intCast(i), @intCast(i + 4));
                    i += 4;
                    continue;
                }
            }
            // Same with per-expert NVFP4 scales: MUL_MAT_ID, RESHAPE, REPEAT, GET_ROWS, MUL for gate and for up, GLU.
            if (i + 10 < cgraph.n_nodes and cgraph.nodes[@intCast(i + 10)].?.op == ggml.impl.ggml_c.GGML_OP_GLU) {
                const glu = cgraph.nodes[@intCast(i + 10)].?;
                const gate_scale = cgraph.nodes[@intCast(i + 4)].?;
                const up_scale = cgraph.nodes[@intCast(i + 9)].?;
                const at = struct {
                    fn f(g: *ggml.impl.CGraph, k: c_int) *ggml.impl.ggml_c.ggml_tensor {
                        return g.nodes[@intCast(k)].?;
                    }
                }.f;
                if (glu.src[0] == gate_scale and glu.src[1] == up_scale and
                    ggml.cuda.mulmat.mulMatQFusionMatches(at(cgraph, i + 5), at(cgraph, i), glu, cuda_ctx.device, up_scale, gate_scale) and
                    ggml.cuda.mulmat.getMulMatIdScale(at(cgraph, i + 1), at(cgraph, i + 2), at(cgraph, i + 3), gate_scale, at(cgraph, i)) != null and
                    ggml.cuda.mulmat.getMulMatIdScale(at(cgraph, i + 6), at(cgraph, i + 7), at(cgraph, i + 8), up_scale, at(cgraph, i + 5)) != null)
                {
                    addAllocDeps(cgraph, params, @intCast(i), @intCast(i + 10));
                    i += 10;
                    continue;
                }
            }

            var match: ggml.cuda.fusion.MoeWeightedReductionMatch = .{};
            if (ggml.cuda.fusion.matchMoeWeightedReduction(cgraph, i, &match)) {
                add_alloc_dep(params.user_data, @constCast(match.experts.?), match.dst.?);
                add_alloc_dep(params.user_data, @constCast(match.weights.?), match.dst.?);
                if (match.expert_scale) |expert_scale| {
                    add_alloc_dep(params.user_data, @constCast(expert_scale), match.dst.?);
                }
                i += match.node_count - 1;
            }

            const nd = cgraph.nodes[@intCast(i)].?;
            if (nd.op == ggml.impl.ggml_c.GGML_OP_UNARY or nd.op == ggml.impl.ggml_c.GGML_OP_SOFT_MAX or nd.op == ggml.impl.ggml_c.GGML_OP_ARGSORT) {
                var args: ggml.cuda.topk_moe.Args = .{};
                const can_fuse = ggml.cuda.fusion.topkMoeFusion(cgraph, i, &args);
                var ops_buf: [16]ggml.impl.ggml_c.enum_ggml_op = undefined;
                var n_ops: usize = 0;

                if (can_fuse and !args.delayed_softmax) {
                    const logits = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, nd.src[0]);
                    var out_nodes: [2]c_int = undefined; // nodes which can't be elided

                    if (args.sigmoid) {
                        appendOps(&ops_buf, &n_ops, &.{ggml.impl.ggml_c.GGML_OP_UNARY});
                    } else if (args.sqrt_softplus) {
                        appendOps(&ops_buf, &n_ops, &.{ ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_SQRT });
                    } else {
                        appendOps(&ops_buf, &n_ops, &.{ggml.impl.ggml_c.GGML_OP_SOFT_MAX});
                    }
                    const i_probs = i + @as(c_int, @intCast(n_ops)) - 1; // last node of the gating activation

                    if (args.prob_bias) {
                        appendOps(&ops_buf, &n_ops, &.{ ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_ARGSORT, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_GET_ROWS });
                        out_nodes[0] = i_probs + 4;
                    } else {
                        appendOps(&ops_buf, &n_ops, &.{ ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_ARGSORT, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_GET_ROWS });
                        out_nodes[0] = i_probs + 3;
                    }
                    const ids = cgraph.nodes[@intCast(out_nodes[0])].?;

                    if (args.norm) {
                        appendOps(&ops_buf, &n_ops, &.{ ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_CLAMP, ggml.impl.ggml_c.GGML_OP_DIV, ggml.impl.ggml_c.GGML_OP_RESHAPE });
                    }
                    if (args.scale) {
                        appendOps(&ops_buf, &n_ops, &.{ggml.impl.ggml_c.GGML_OP_SCALE});
                    }

                    const n: c_int = @intCast(n_ops);
                    const weights = cgraph.nodes[@intCast(i + n - 1)].?;
                    out_nodes[1] = i + n - 1;

                    if (ggml.graph.canFuseSubgraph(cgraph, i, ops_buf[0..n_ops], &out_nodes) and
                        ggml.cuda.topk_moe.shouldUseTopkMoe(nd, logits, weights, ids))
                    {
                        addAllocDeps(cgraph, params, @intCast(i), @intCast(i + n));
                        i += n - 1;
                    }
                }
            }
        }
    }

    const graph_key = graphGetKey(cgraph);
    const use_cuda_graph = graphSetEnabled(cuda_ctx, graph_key);

    GraphOpt.once.call();
    if (!GraphOpt.enabled) return;

    const stream_context = &cuda_ctx.concurrent_stream_context;
    stream_context.reset();

    if (!use_cuda_graph) return;

    ggml.cuda.common.setDevice(cuda_ctx.device);

    // number of out-degrees for a particular node
    var fan_out: std.AutoArrayHashMapUnmanaged(?*const ggml.impl.ggml_c.ggml_tensor, c_int) = .empty;
    defer fan_out.deinit(std.heap.c_allocator);
    // reverse mapping of node to index in the cgraph
    var node_indices: std.AutoHashMapUnmanaged(*const ggml.impl.ggml_c.ggml_tensor, usize) = .empty;
    defer node_indices.deinit(std.heap.c_allocator);

    const ns = nodes(cgraph);
    for (ns, 0..) |node, node_idx| {
        node_indices.put(std.heap.c_allocator, node, node_idx) catch ggml.impl.abort("ggml-cuda: out of memory");
        if (isNoop(node)) continue;
        for (node.src) |src| {
            //TODO: check why nrows > 1 fails
            if (!isNoop(node) and ggml.impl.ggml_c.ggml_nrows(node) <= 1) {
                const gop = fan_out.getOrPut(std.heap.c_allocator, src) catch ggml.impl.abort("ggml-cuda: out of memory");
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* += 1;
            }
        }
    }

    const min_fan_out = 3;
    const max_fan_out = 3;

    // store {fork_idx, join_idx}
    var concurrent_node_ranges: std.ArrayList([2]usize) = .empty;
    defer concurrent_node_ranges.deinit(std.heap.c_allocator);

    var it = fan_out.iterator();
    while (it.next()) |kv| {
        const count = kv.value_ptr.*;
        if (count < min_fan_out or count > max_fan_out) continue;
        const root_node = kv.key_ptr.*.?;
        const root_node_idx = node_indices.get(root_node) orelse 0;

        // only optimize for attn_norm
        if (std.mem.indexOf(u8, std.mem.sliceTo(&root_node.name, 0), "attn_norm") == null) continue;

        var is_part_of_event = false;
        for (concurrent_node_ranges.items) |r| {
            if (root_node_idx >= r[0] and root_node_idx <= r[1]) is_part_of_event = true;
        }
        if (is_part_of_event) continue;

        var nodes_per_branch: std.ArrayList(std.ArrayList(*const ggml.impl.ggml_c.ggml_tensor)) = .empty;
        defer {
            for (nodes_per_branch.items) |*b| b.deinit(std.heap.c_allocator);
            nodes_per_branch.deinit(std.heap.c_allocator);
        }
        for (ns[root_node_idx + 1 ..]) |node| {
            if (!isNoop(node) and dependsOn(node, root_node)) {
                var branch: std.ArrayList(*const ggml.impl.ggml_c.ggml_tensor) = .empty;
                branch.append(std.heap.c_allocator, node) catch ggml.impl.abort("ggml-cuda: out of memory");
                nodes_per_branch.append(std.heap.c_allocator, branch) catch ggml.impl.abort("ggml-cuda: out of memory");
            }
        }

        ggml.impl.assert(nodes_per_branch.items.len == @as(usize, @intCast(count)), "nodes_per_branch.size() == (size_t) count");

        //find the join point
        var join_node: ?*const ggml.impl.ggml_c.ggml_tensor = null;
        for (ns[root_node_idx + 1 ..]) |curr_node| {
            var num_joins: usize = 0;
            for (nodes_per_branch.items) |b| {
                if (belongsToBranch(curr_node, b.items)) num_joins += 1;
            }
            if (num_joins >= 2) {
                join_node = curr_node;
                break;
            }

            var found_branch = false;
            for (nodes_per_branch.items) |*b| {
                if (belongsToBranch(curr_node, b.items)) {
                    //continue accumulating
                    if (std.mem.indexOfScalar(*const ggml.impl.ggml_c.ggml_tensor, b.items, curr_node) == null) {
                        b.append(std.heap.c_allocator, curr_node) catch ggml.impl.abort("ggml-cuda: out of memory");
                    }
                    found_branch = true;
                }
            }

            if (!found_branch and isNoop(curr_node)) {
                // we can put it in any branch because it will be ignored
                nodes_per_branch.items[0].append(std.heap.c_allocator, curr_node) catch ggml.impl.abort("ggml-cuda: out of memory");
            }
        }

        const join = join_node orelse continue;

        var event = ggml.cuda.common.ConcurrentEvent.init(@intCast(nodes_per_branch.items.len));
        var keep = false;
        defer if (!keep) event.deinit();
        event.join_node = join;

        for (nodes_per_branch.items, 0..) |b, branch_idx| {
            for (b.items) |n| event.stream_mapping.put(std.heap.c_allocator, n, @intCast(branch_idx + 1)) catch ggml.impl.abort("ggml-cuda: out of memory");
        }

        const fork_node_idx = node_indices.get(root_node).?;
        const join_node_idx = node_indices.get(join).?;

        var current_branch_idx: usize = 0;
        var current_node_idx = fork_node_idx + 1;
        const n_branches = nodes_per_branch.items.len;

        var total_branch_nodes: usize = 0;
        for (nodes_per_branch.items) |b| total_branch_nodes += b.items.len;

        // there are other nodes in the middle which are unaccounted for
        // usually (cpy) nodes, then ignore this fork
        if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
            ggml.impl.logDebug("Skipping %s because the number of nodes in the middle is not equal to the total number of branch nodes %d != %d\n", .{ @as([*:0]const u8, @ptrCast(&root_node.name)), @as(c_int, @intCast(join_node_idx - fork_node_idx - 1)), @as(c_int, @intCast(total_branch_nodes)) });
            continue;
        }

        // Save the original order of nodes in this region before interleaving
        for (ns[fork_node_idx + 1 .. join_node_idx]) |n| event.original_order.append(std.heap.c_allocator, n) catch ggml.impl.abort("ggml-cuda: out of memory");

        const concurrent_events = &stream_context.concurrent_events;
        ggml.impl.assert(!concurrent_events.contains(root_node), "concurrent_events.find(root_node) == concurrent_events.end()");
        concurrent_events.put(std.heap.c_allocator, root_node, event) catch ggml.impl.abort("ggml-cuda: out of memory");
        keep = true;
        ggml.impl.logDebug("Adding stream at node %s %p\n", .{ @as([*:0]const u8, @ptrCast(&root_node.name)), root_node });
        concurrent_node_ranges.append(std.heap.c_allocator, .{ fork_node_idx, join_node_idx }) catch ggml.impl.abort("ggml-cuda: out of memory");

        // interleave tensors to extend lifetimes so that ggml graph doesn't
        // recycle them
        var fronts: std.ArrayList(usize) = .empty;
        defer fronts.deinit(std.heap.c_allocator);
        fronts.appendNTimes(std.heap.c_allocator, 0, n_branches) catch ggml.impl.abort("ggml-cuda: out of memory");
        while (current_node_idx < join_node_idx) {
            const branch = nodes_per_branch.items[current_branch_idx].items;
            const front = &fronts.items[current_branch_idx];

            var has_node = false;
            for (nodes_per_branch.items, fronts.items) |b, f| has_node = has_node or b.items.len > f;
            ggml.impl.assert(has_node, "has_node");

            if (front.* == branch.len) {
                current_branch_idx = (current_branch_idx + 1) % n_branches;
                continue;
            }

            ns[current_node_idx] = @constCast(branch[front.*]);
            current_node_idx += 1;
            front.* += 1;

            // append all empty nodes
            while (front.* < branch.len and isNoop(branch[front.*])) {
                ns[current_node_idx] = @constCast(branch[front.*]);
                current_node_idx += 1;
                front.* += 1;
            }

            current_branch_idx = (current_branch_idx + 1) % n_branches;
        }
    }
}
