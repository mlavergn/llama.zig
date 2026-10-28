//! The CUDA backend's compute loop: op dispatch, CUDA graph capture and
//! replay, concurrent streams, and `graph_optimize`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v0.4.0
//! (`5266f24da`): `ggml_cuda_compute_forward`, the `USE_CUDA_GRAPH` helpers,
//! `ggml_cuda_graph_evaluate_and_capture`, `ggml_backend_cuda_graph_compute`
//! and `ggml_backend_cuda_graph_optimize`.
//!
//! The op switch is the C's, case for case. While the port was in progress
//! an `isPorted` predicate mirrored it and was AND-ed into `supports_op`
//! (PLAN.md, Decision 48); it went at the swap, once `cuda-ops-diff` showed
//! `supports_op` equal to the reference's on every case.
//!
//! Fusion, `ggml_cuda_try_fuse`, is in `fusion.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const backend = @import("backend.zig");
const scale = @import("scale.zig");
const binbcast = @import("binbcast.zig");
const mulmat = @import("mulmat.zig");
const fusion = @import("fusion.zig");
const lightning_indexer = @import("lightning_indexer.zig");
// Elementwise and small reductions.
const unary = @import("unary.zig");
const clamp = @import("clamp.zig");
const sumrows = @import("sumrows.zig");
const mean = @import("mean.zig");
const sum = @import("sum.zig");
const argmax = @import("argmax.zig");
const count_equal = @import("count_equal.zig");
const arange = @import("arange.zig");
const fill = @import("fill.zig");
const tri = @import("tri.zig");
const diag = @import("diag.zig");
const diagmask = @import("diagmask.zig");
const roll = @import("roll.zig");
const pad = @import("pad.zig");
const pad_reflect_1d = @import("pad_reflect_1d.zig");
const upscale = @import("upscale.zig");
const tsembd = @import("tsembd.zig");
const acc = @import("acc.zig");
const add_id = @import("add_id.zig");
// Recurrent-state ops and convolutions.
const ssm_conv = @import("ssm_conv.zig");
const ssm_scan = @import("ssm_scan.zig");
const gated_delta_net = @import("gated_delta_net.zig");
const wkv = @import("wkv.zig");
const gla = @import("gla.zig");
const solve_tri = @import("solve_tri.zig");
const im2col = @import("im2col.zig");
const conv2d = @import("conv2d.zig");
const conv2d_dw = @import("conv2d_dw.zig");
const conv2d_transpose = @import("conv2d_transpose.zig");
const conv_transpose_1d = @import("conv_transpose_1d.zig");
const col2im_1d = @import("col2im_1d.zig");
const pool1d = @import("pool1d.zig");
const pool2d = @import("pool2d.zig");
const out_prod = @import("out_prod.zig");
const dsv4_hc = @import("dsv4_hc.zig");

// Normalisation, softmax, rope and the training ops.
const norm = @import("norm.zig");
const softmax = @import("softmax.zig");
const rope = @import("rope.zig");
const cross_entropy_loss = @import("cross_entropy_loss.zig");
const opt_step_adamw = @import("opt_step_adamw.zig");
const opt_step_sgd = @import("opt_step_sgd.zig");
// Flash attention.
const fattn = @import("fattn.zig");
const getrows = @import("getrows.zig");
const set_rows = @import("set_rows.zig");
const cpy = @import("cpy.zig");
const set = @import("set.zig");
const concat = @import("concat.zig");
const argsort = @import("argsort.zig");
const top_k = @import("top_k.zig");
const cumsum = @import("cumsum.zig");

const cu = common.cu;
const c = impl.c;
const Tensor = c.ggml_tensor;
const Context = common.Context;
const allocator = common.allocator;
const cudaCheck = common.cudaCheck;

// -----------------------------------------------------------------------------
// Dispatch

/// Ports `ggml_cuda_compute_forward` (ggml-cuda.cu:2062 @5266f24da).
fn computeForward(ctx: *Context, dst: *Tensor) bool {
    switch (dst.op) {
        c.GGML_OP_REPEAT => binbcast.opRepeat(ctx, dst),
        c.GGML_OP_REPEAT_BACK => binbcast.opRepeatBack(ctx, dst),
        // TODO in the C: a more efficient ADD1.
        c.GGML_OP_ADD, c.GGML_OP_ADD1 => binbcast.opBinary(.add, ctx, dst),
        c.GGML_OP_SUB => binbcast.opBinary(.sub, ctx, dst),
        c.GGML_OP_MUL => binbcast.opBinary(.mul, ctx, dst),
        c.GGML_OP_DIV => binbcast.opBinary(.div, ctx, dst),
        c.GGML_OP_MUL_MAT => mulmat.mulMat(ctx, impl.one(Tensor, dst.src[0]), impl.one(Tensor, dst.src[1]), dst),
        c.GGML_OP_MUL_MAT_ID => mulmat.mulMatId(ctx, dst),
        c.GGML_OP_LIGHTNING_INDEXER => lightning_indexer.opLightningIndexer(ctx, dst),
        c.GGML_OP_FLASH_ATTN_EXT => fattn.flashAttnExt(ctx, dst),
        c.GGML_OP_SCALE => scale.opScale(ctx, dst),
        c.GGML_OP_NONE, c.GGML_OP_RESHAPE, c.GGML_OP_VIEW, c.GGML_OP_PERMUTE, c.GGML_OP_TRANSPOSE => {},
        // Elementwise and small reductions, in the C's switch order.
        c.GGML_OP_ARGMAX => argmax.opArgmax(ctx, dst),
        c.GGML_OP_COUNT_EQUAL => count_equal.opCountEqual(ctx, dst),
        c.GGML_OP_ADD_ID => add_id.opAddId(ctx, dst),
        c.GGML_OP_ACC => acc.opAcc(ctx, dst),
        c.GGML_OP_UNARY => switch (c.ggml_get_unary_op(dst)) {
            c.GGML_UNARY_OP_ABS => unary.opUnary(.abs, ctx, dst),
            c.GGML_UNARY_OP_SGN => unary.opUnary(.sgn, ctx, dst),
            c.GGML_UNARY_OP_NEG => unary.opUnary(.neg, ctx, dst),
            c.GGML_UNARY_OP_STEP => unary.opUnary(.step, ctx, dst),
            c.GGML_UNARY_OP_GELU => unary.opUnary(.gelu, ctx, dst),
            c.GGML_UNARY_OP_SILU => unary.opUnary(.silu, ctx, dst),
            c.GGML_UNARY_OP_GELU_ERF => unary.opUnary(.gelu_erf, ctx, dst),
            c.GGML_UNARY_OP_GELU_QUICK => unary.opUnary(.gelu_quick, ctx, dst),
            c.GGML_UNARY_OP_TANH => unary.opUnary(.tanh, ctx, dst),
            c.GGML_UNARY_OP_RELU => unary.opUnary(.relu, ctx, dst),
            c.GGML_UNARY_OP_SIGMOID => unary.opUnary(.sigmoid, ctx, dst),
            c.GGML_UNARY_OP_HARDSIGMOID => unary.opUnary(.hardsigmoid, ctx, dst),
            c.GGML_UNARY_OP_HARDSWISH => unary.opUnary(.hardswish, ctx, dst),
            c.GGML_UNARY_OP_EXP => unary.opUnary(.exp, ctx, dst),
            c.GGML_UNARY_OP_ELU => unary.opUnary(.elu, ctx, dst),
            c.GGML_UNARY_OP_XIELU => unary.opXielu(ctx, dst),
            c.GGML_UNARY_OP_FLOOR => unary.opUnary(.floor, ctx, dst),
            c.GGML_UNARY_OP_CEIL => unary.opUnary(.ceil, ctx, dst),
            c.GGML_UNARY_OP_ROUND => unary.opUnary(.round, ctx, dst),
            c.GGML_UNARY_OP_TRUNC => unary.opUnary(.trunc, ctx, dst),
            c.GGML_UNARY_OP_EXPM1 => unary.opUnary(.expm1, ctx, dst),
            c.GGML_UNARY_OP_SOFTPLUS => unary.opUnary(.softplus, ctx, dst),
            else => return false,
        },
        c.GGML_OP_GLU => switch (c.ggml_get_glu_op(dst)) {
            c.GGML_GLU_OP_REGLU => unary.opGated(.relu, ctx, dst),
            c.GGML_GLU_OP_GEGLU => unary.opGated(.gelu, ctx, dst),
            c.GGML_GLU_OP_SWIGLU => unary.opGated(.silu, ctx, dst),
            c.GGML_GLU_OP_SWIGLU_OAI => unary.opSwigluOai(ctx, dst),
            c.GGML_GLU_OP_GEGLU_ERF => unary.opGated(.gelu_erf, ctx, dst),
            c.GGML_GLU_OP_GEGLU_QUICK => unary.opGated(.gelu_quick, ctx, dst),
            c.GGML_GLU_OP_SWIGLU_CLAMP => unary.opSwigluClamp(ctx, dst),
            else => return false,
        },
        c.GGML_OP_UPSCALE => upscale.opUpscale(ctx, dst),
        c.GGML_OP_PAD => pad.opPad(ctx, dst),
        c.GGML_OP_PAD_REFLECT_1D => pad_reflect_1d.opPadReflect1d(ctx, dst),
        c.GGML_OP_ARANGE => arange.opArange(ctx, dst),
        c.GGML_OP_TIMESTEP_EMBEDDING => tsembd.opTimestepEmbedding(ctx, dst),
        c.GGML_OP_LEAKY_RELU => unary.opLeakyRelu(ctx, dst),
        c.GGML_OP_SILU_BACK => unary.opSiluBack(ctx, dst),
        c.GGML_OP_SQR => unary.opUnary(.sqr, ctx, dst),
        c.GGML_OP_SQRT => unary.opUnary(.sqrt, ctx, dst),
        c.GGML_OP_SIN => unary.opUnary(.sin, ctx, dst),
        c.GGML_OP_COS => unary.opUnary(.cos, ctx, dst),
        c.GGML_OP_CLAMP => clamp.opClamp(ctx, dst),
        c.GGML_OP_LOG => unary.opUnary(.log, ctx, dst),
        c.GGML_OP_DIAG => diag.opDiag(ctx, dst),
        c.GGML_OP_DIAG_MASK_INF => diagmask.opDiagMaskInf(ctx, dst),
        c.GGML_OP_ROLL => roll.opRoll(ctx, dst),
        c.GGML_OP_SUM_ROWS => sumrows.opSumRows(ctx, dst),
        c.GGML_OP_MEAN => mean.opMean(ctx, dst),
        c.GGML_OP_SUM => sum.opSum(ctx, dst),
        c.GGML_OP_TRI => tri.opTri(ctx, dst),
        c.GGML_OP_FILL => fill.opFill(ctx, dst),
        // Normalisation, softmax, rope and the training ops.
        c.GGML_OP_NORM => norm.opNorm(ctx, dst),
        c.GGML_OP_GROUP_NORM => norm.opGroupNorm(ctx, dst),
        c.GGML_OP_L2_NORM => norm.opL2Norm(ctx, dst),
        c.GGML_OP_RMS_NORM => norm.opRmsNorm(ctx, dst),
        c.GGML_OP_RMS_NORM_BACK => norm.opRmsNormBack(ctx, dst),
        c.GGML_OP_SOFT_MAX => softmax.opSoftMax(ctx, dst),
        c.GGML_OP_SOFT_MAX_BACK => softmax.opSoftMaxBack(ctx, dst),
        c.GGML_OP_ROPE => rope.opRope(ctx, dst),
        c.GGML_OP_ROPE_BACK => rope.opRopeBack(ctx, dst),
        c.GGML_OP_CROSS_ENTROPY_LOSS => cross_entropy_loss.opCrossEntropyLoss(ctx, dst),
        c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => cross_entropy_loss.opCrossEntropyLossBack(ctx, dst),
        c.GGML_OP_OPT_STEP_ADAMW => opt_step_adamw.opOptStepAdamw(ctx, dst),
        c.GGML_OP_OPT_STEP_SGD => opt_step_sgd.opOptStepSgd(ctx, dst),
        // Recurrent-state ops and convolutions.
        c.GGML_OP_OUT_PROD => out_prod.outProd(ctx, dst),
        c.GGML_OP_IM2COL => im2col.opIm2col(ctx, dst),
        c.GGML_OP_IM2COL_3D => im2col.opIm2col3d(ctx, dst),
        c.GGML_OP_CONV_2D => conv2d.opConv2d(ctx, dst),
        c.GGML_OP_CONV_2D_DW => conv2d_dw.opConv2dDw(ctx, dst),
        c.GGML_OP_CONV_TRANSPOSE_2D => conv2d_transpose.conv2dTransposeP0(ctx, dst),
        c.GGML_OP_CONV_TRANSPOSE_1D => conv_transpose_1d.opConvTranspose1d(ctx, dst),
        c.GGML_OP_COL2IM_1D => col2im_1d.opCol2im1d(ctx, dst),
        c.GGML_OP_POOL_2D => pool2d.opPool2d(ctx, dst),
        c.GGML_OP_POOL_1D => pool1d.opPool1d(ctx, dst),
        c.GGML_OP_SSM_CONV => ssm_conv.opSsmConv(ctx, dst, null, null),
        c.GGML_OP_SSM_SCAN => ssm_scan.opSsmScan(ctx, dst),
        c.GGML_OP_RWKV_WKV6 => wkv.opRwkvWkv6(ctx, dst),
        c.GGML_OP_GATED_LINEAR_ATTN => gla.opGatedLinearAttn(ctx, dst),
        c.GGML_OP_GATED_DELTA_NET => gated_delta_net.opGatedDeltaNet(ctx, dst),
        c.GGML_OP_DSV4_HC_COMB => dsv4_hc.opDsv4HcComb(ctx, dst),
        c.GGML_OP_DSV4_HC_PRE => dsv4_hc.opDsv4HcPre(ctx, dst),
        c.GGML_OP_DSV4_HC_POST => dsv4_hc.opDsv4HcPost(ctx, dst),
        c.GGML_OP_RWKV_WKV7 => wkv.opRwkvWkv7(ctx, dst),
        c.GGML_OP_SOLVE_TRI => solve_tri.opSolveTri(ctx, dst),
        // Data movement, indexing and sorting.
        c.GGML_OP_GET_ROWS => getrows.opGetRows(ctx, dst),
        c.GGML_OP_GET_ROWS_BACK => getrows.opGetRowsBack(ctx, dst),
        c.GGML_OP_SET_ROWS => set_rows.opSetRows(ctx, dst),
        c.GGML_OP_SET => set.opSet(ctx, dst),
        c.GGML_OP_DUP => cpy.dup(ctx, dst),
        c.GGML_OP_CPY => cpy.cpy(ctx, impl.one(Tensor, dst.src[0]), impl.one(Tensor, dst.src[1])),
        c.GGML_OP_CONT => cpy.dup(ctx, dst),
        c.GGML_OP_CONCAT => concat.opConcat(ctx, dst),
        c.GGML_OP_CUMSUM => cumsum.opCumsum(ctx, dst),
        c.GGML_OP_TOP_K => top_k.opTopK(ctx, dst),
        c.GGML_OP_ARGSORT => argsort.opArgsort(ctx, dst),
        else => return false,
    }

    const err = cu.cudaGetLastError();
    if (err != cu.cudaSuccess) {
        impl.logError("%s: %s failed\n", .{ "ggml_cuda_compute_forward", c.ggml_op_desc(dst) });
        cudaCheck(err, "err", @src());
    }
    return true;
}

/// Ports `ggml_cuda_try_fuse`; see `fusion.zig`.
const tryFuse = fusion.tryFuse;

/// Ports `ggml_cuda_mul_mat_id_needs_sync`; see `mulmat.zig`.
const mulMatIdNeedsSync = mulmat.mulMatIdNeedsSync;

// -----------------------------------------------------------------------------
// CUDA graphs

/// Ports `ggml_cuda_is_view_or_noop` (ggml-cuda.cu:2547 @5266f24da).
fn isViewOrNoop(t: *const Tensor) bool {
    return c.ggml_is_empty(t) or t.op == c.GGML_OP_RESHAPE or t.op == c.GGML_OP_TRANSPOSE or
        t.op == c.GGML_OP_VIEW or t.op == c.GGML_OP_PERMUTE or t.op == c.GGML_OP_NONE;
}

fn nodes(cgraph: *impl.CGraph) []*Tensor {
    return @ptrCast(cgraph.nodes[0..@intCast(cgraph.n_nodes)]);
}

/// Ports `ggml_cuda_graph_check_compability` (ggml-cuda.cu:2553 @5266f24da).
fn graphCheckCompatibility(cgraph: *impl.CGraph) bool {
    var use_cuda_graph = true;
    for (nodes(cgraph)) |node| {
        if (isViewOrNoop(node)) continue;

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node.op == c.GGML_OP_MUL_MAT_ID) {
            const x = common.info().devices[@intCast(common.getDevice())].cc;
            if (mulMatIdNeedsSync(node, x)) {
                // the mul_mat_id fallback path synchronizes the stream, so
                // CUDA graphs cannot be used
                use_cuda_graph = false;
            }
        }

        if (!use_cuda_graph) break;
    }
    return use_cuda_graph;
}

/// Ports `ggml_cuda_graph_get_key` (ggml-cuda.cu:2586 @5266f24da).
fn graphGetKey(cgraph: *impl.CGraph) ?*const anyopaque {
    return cgraph.nodes[0];
}

/// Ports `ggml_cuda_graph_update_required` (ggml-cuda.cu:2590 @5266f24da).
fn graphUpdateRequired(cuda_ctx: *Context, cgraph: *impl.CGraph) bool {
    var res = false;

    const graph = cuda_ctx.cudaGraph(graphGetKey(cgraph));

    if (cgraph.uid != 0 and cgraph.uid == graph.uid) {
        impl.logDebug("CUDA Graph id %zu reused\n", .{cgraph.uid});
        impl.assert(graph.node_props.items.len == @as(usize, @intCast(cgraph.n_nodes)), "(int)graph->node_props.size() == cgraph->n_nodes");
        return false;
    }

    graph.uid = cgraph.uid;

    const n: usize = @intCast(cgraph.n_nodes);
    // Check if the graph size has changed
    if (graph.node_props.items.len != n) {
        res = true;
        // `resize` value-initialises the new elements.
        const old = graph.node_props.items.len;
        graph.node_props.resize(allocator, n) catch impl.abort("ggml-cuda: out of memory");
        if (n > old) @memset(std.mem.sliceAsBytes(graph.node_props.items[old..]), 0);
    }

    for (nodes(cgraph), 0..) |node, i| {
        var prop: common.NodeProperties = std.mem.zeroes(common.NodeProperties);
        prop.node = node.*;

        for (0..c.GGML_MAX_SRC) |j| {
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

/// Ports `ggml_cuda_graph_update_executable` (ggml-cuda.cu:2632 @5266f24da),
/// the CUDA 12 arm.
fn graphUpdateExecutable(cuda_ctx: *Context, graph_key: ?*const anyopaque) void {
    const graph = cuda_ctx.cudaGraph(graph_key);

    var result_info: cu.cudaGraphExecUpdateResultInfo = undefined;
    const stat = cu.cudaGraphExecUpdate(graph.instance, graph.graph, &result_info);

    if (stat == cu.cudaErrorGraphExecUpdateFailure) {
        // The pre-existing graph exec cannot be updated due to violated
        // constraints, so clear the error and re-instantiate.
        _ = cu.cudaGetLastError();
        cudaCheck(cu.cudaGraphExecDestroy(graph.instance), "cudaGraphExecDestroy(graph->instance)", @src());
        graph.instance = null;
        cudaCheck(cu.cudaGraphInstantiate(&graph.instance, graph.graph, 0), "cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0)", @src());
    } else {
        impl.assert(stat == cu.cudaSuccess, "stat == cudaSuccess");
    }
}

/// Ports `ggml_cuda_graph_set_enabled` (ggml-cuda.cu:4399 @5266f24da).
fn graphSetEnabled(cuda_ctx: *Context, graph_key: ?*const anyopaque) bool {
    const graph = cuda_ctx.cudaGraph(graph_key);

    if (graph.graph == null) {
        if (common.info().devices[@intCast(cuda_ctx.device)].cc < common.cc.volta) {
            if (!graph.disable_due_to_gpu_arch) {
                impl.logDebug("%s: disabling CUDA graphs due to GPU architecture\n", .{"ggml_cuda_graph_set_enabled"});
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
    cuda_ctx: *Context,
    is_active: bool = false,
    event: ?*common.ConcurrentEvent = null,

    fn tryLaunch(self: *Concurrency, node: *const Tensor) void {
        const stream_ctx = &self.cuda_ctx.concurrent_stream_context;
        const ev = stream_ctx.concurrent_events.getPtr(node) orelse return;
        self.event = ev;
        self.is_active = true;

        impl.logDebug("Launching %d streams at %s\n", .{ ev.n_streams, @as([*:0]const u8, @ptrCast(&node.name)) });

        const main_stream = self.cuda_ctx.stream(); // this should be stream 0
        impl.assert(self.cuda_ctx.curr_stream_no == 0, "cuda_ctx->curr_stream_no == 0");
        cudaCheck(cu.cudaEventRecord(ev.fork_event, main_stream), "cudaEventRecord(concurrent_event->fork_event, main_stream)", @src());

        var i: c_int = 1;
        while (i <= ev.n_streams) : (i += 1) {
            const s = self.cuda_ctx.streamFor(self.cuda_ctx.device, i);
            cudaCheck(cu.cudaStreamWaitEvent(s, ev.fork_event, 0), "cudaStreamWaitEvent(stream, concurrent_event->fork_event)", @src());
        }
    }
};

/// Ports `ggml_cuda_graph_evaluate_and_capture` (ggml-cuda.cu:4180 @5266f24da).
fn graphEvaluateAndCapture(cuda_ctx: *Context, cgraph: *impl.CGraph, use_cuda_graph: bool, cuda_graph_update_required: bool, graph_key: ?*const anyopaque) void {
    var graph_evaluated_or_captured = false;

    const stream_ctx = &cuda_ctx.concurrent_stream_context;
    var conc: Concurrency = .{ .cuda_ctx = cuda_ctx };
    var should_launch_concurrent_events = false;

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
                if (conc.is_active) {
                    const ev = conc.event.?;
                    if (node == ev.join_node) {
                        cuda_ctx.curr_stream_no = 0;
                        var s: c_int = 1;
                        while (s <= ev.n_streams) : (s += 1) {
                            // Wait on join events of forked streams in the main stream
                            const je = ev.join_events.items[@intCast(s - 1)];
                            cudaCheck(cu.cudaEventRecord(je, cuda_ctx.streamFor(cuda_ctx.device, s)), "cudaEventRecord(concurrent_event->join_events[i - 1], cuda_ctx->stream(cuda_ctx->device, i))", @src());
                            cudaCheck(cu.cudaStreamWaitEvent(cuda_ctx.stream(), je, 0), "cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1])", @src());
                        }
                        conc.is_active = false;
                        conc.event = null;
                    } else {
                        const mapped = ev.stream_mapping.get(node);
                        impl.assert(mapped != null, "concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end()");
                        cuda_ctx.curr_stream_no = mapped.?;
                        impl.logDebug("Setting stream no to %d for node %s\n", .{ cuda_ctx.curr_stream_no, @as([*:0]const u8, @ptrCast(&node.name)) });
                    }
                } else if (i -% prev_i > 1 and i > prev_i) {
                    // the previous node was fused
                    conc.tryLaunch(ns[i - 1]);
                    if (conc.is_active) {
                        // `stream_mapping[node]` inserts 0 when absent.
                        const gop = conc.event.?.stream_mapping.getOrPut(allocator, node) catch impl.abort("ggml-cuda: out of memory");
                        if (!gop.found_existing) gop.value_ptr.* = 0;
                        cuda_ctx.curr_stream_no = gop.value_ptr.*;
                        impl.logDebug("Setting stream no to %d for node %s\n", .{ cuda_ctx.curr_stream_no, @as([*:0]const u8, @ptrCast(&node.name)) });
                    }
                }

                prev_i = i;

                if (isViewOrNoop(node)) continue;
                if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) continue;

                const nodes_to_skip = tryFuse(cuda_ctx, cgraph, @intCast(i));
                if (nodes_to_skip != 0) {
                    i += @intCast(nodes_to_skip);
                    continue;
                }

                const ok = computeForward(cuda_ctx, node);
                if (!ok) {
                    impl.logError("%s: op not supported %s (%s)\n", .{ "ggml_cuda_graph_evaluate_and_capture", @as([*:0]const u8, @ptrCast(&node.name)), c.ggml_op_name(node.op) });
                }
                impl.assert(ok, "ok");

                if (!conc.is_active) conc.tryLaunch(node);
            }
        }

        const graph = cuda_ctx.cudaGraph(graph_key);
        if (use_cuda_graph and cuda_graph_update_required) { // End CUDA graph capture
            if (graph.graph != null) {
                cudaCheck(cu.cudaGraphDestroy(graph.graph), "cudaGraphDestroy(graph->graph)", @src());
                graph.graph = null;
            }

            cudaCheck(cu.cudaStreamEndCapture(cuda_ctx.stream(), &graph.graph), "cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph)", @src());
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            common.lock.release();
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        const graph = cuda_ctx.cudaGraph(graph_key);
        if (graph.instance == null) { // Create executable graph from captured graph.
            cudaCheck(cu.cudaGraphInstantiate(&graph.instance, graph.graph, 0), "cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0)", @src());
        }
        if (cuda_graph_update_required) { // Update graph executable
            graphUpdateExecutable(cuda_ctx, graph_key);
        }
        // Launch graph
        cudaCheck(cu.cudaGraphLaunch(graph.instance, cuda_ctx.stream()), "cudaGraphLaunch(graph->instance, cuda_ctx->stream())", @src());
    }
}

/// The C's "restore original node order within each concurrent region"
/// block in `ggml_cuda_graph_evaluate_and_capture` (ggml-cuda.cu:4180 @5266f24da).
fn restoreOriginalOrder(cuda_ctx: *Context, cgraph: *impl.CGraph) void {
    var node_to_idx: std.AutoHashMapUnmanaged(*const Tensor, usize) = .empty;
    defer node_to_idx.deinit(allocator);
    const ns = nodes(cgraph);
    for (ns, 0..) |node, i| node_to_idx.put(allocator, node, i) catch impl.abort("ggml-cuda: out of memory");

    for (cuda_ctx.concurrent_stream_context.concurrent_events.values()) |*ev| {
        var positions: std.ArrayList(usize) = .empty;
        defer positions.deinit(allocator);

        var all_found = true;
        for (ev.original_order.items) |orig| {
            if (node_to_idx.get(orig)) |idx| {
                positions.append(allocator, idx) catch impl.abort("ggml-cuda: out of memory");
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

/// Ports `ggml_backend_cuda_graph_compute` (ggml-cuda.cu:4415 @5266f24da).
pub fn graphCompute(backend_c: c.ggml_backend_t, cgraph_c: ?*c.struct_ggml_cgraph) callconv(.c) c.enum_ggml_status {
    const cuda_ctx = backend.backendCtx(backend_c);
    const cgraph: *impl.CGraph = @ptrCast(@alignCast(cgraph_c));

    common.setDevice(cuda_ctx.device);

    var use_cuda_graph = false;
    var cuda_graph_update_required = false;

    const graph_key = graphGetKey(cgraph);

    _ = graphSetEnabled(cuda_ctx, graph_key);

    const graph = cuda_ctx.cudaGraph(graph_key);
    if (graph.isEnabled()) {
        const graph_compatible = graphCheckCompatibility(cgraph);
        if (graph_compatible) {
            const properties_changed = graphUpdateRequired(cuda_ctx, cgraph);

            if (!graph.warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph.warmup_complete = true;
                    impl.logDebug("%s: CUDA graph warmup complete\n", .{"ggml_backend_cuda_graph_compute"});
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph.warmup_complete = false;
                    impl.logDebug("%s: CUDA graph warmup reset\n", .{"ggml_backend_cuda_graph_compute"});
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph.instance == null;
                }
            }
        }
    }

    if (use_cuda_graph and cuda_graph_update_required) {
        // Start CUDA graph capture
        common.lock.acquire();
        cudaCheck(cu.cudaStreamBeginCapture(cuda_ctx.stream(), cu.cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed)", @src());
    }

    graphEvaluateAndCapture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return c.GGML_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------------
// Graph optimisation

fn isNoop(node: *const Tensor) bool {
    return c.ggml_is_empty(node) or node.op == c.GGML_OP_NONE or node.op == c.GGML_OP_RESHAPE or
        node.op == c.GGML_OP_TRANSPOSE or node.op == c.GGML_OP_VIEW or node.op == c.GGML_OP_PERMUTE;
}

fn dependsOn(dst: *const Tensor, src: *const Tensor) bool {
    for (dst.src) |s| {
        if (s == src) return true;
    }
    // implicit dependency if they view the same tensor
    const dst2: *const Tensor = if (dst.view_src) |v| v else dst;
    const src2: *const Tensor = if (src.view_src) |v| v else src;
    return dst2 == src2;
}

fn belongsToBranch(node: *const Tensor, branch: []const *const Tensor) bool {
    for (branch) |n| {
        if (dependsOn(node, n)) return true;
    }
    return false;
}

const GraphOpt = struct {
    var enabled: bool = false;
    fn init() void {
        const env = std.c.getenv("GGML_CUDA_GRAPH_OPT");
        enabled = env != null and common.atoi(env.?) == 1;
    }
    const once = common.Once(init);
};

/// Ports `ggml_backend_cuda_graph_optimize` (ggml-cuda.cu:4499 @5266f24da).
///
/// Its first pass runs whether or not CUDA graphs are on: every
/// MoE weighted reduction `fusion.zig` would fuse reads its experts, router
/// weights and expert scale at the chain's last node, after the scheduler
/// thinks they are dead, so each is kept allocated until then through
/// `params.add_alloc_dep`.
///
/// The C walks `fan_out`, a `std::unordered_map`, to pick fork nodes, and
/// handles the first qualifying one in each range; which is "first" is
/// libstdc++'s iteration order. With `min_fan_out == max_fan_out == 3` and
/// only `attn_norm` nodes eligible, the forks found are disjoint, so the
/// order affects only which region a skipped overlap would have kept --
/// and overlapping ranges cannot occur, since each fork's range ends at its
/// own join.
pub fn graphOptimize(backend_c: c.ggml_backend_t, cgraph_c: ?*c.struct_ggml_cgraph, params_c: ?*c.struct_ggml_backend_graph_optimize_params) callconv(.c) void {
    const cuda_ctx = backend.backendCtx(backend_c);
    const cgraph: *impl.CGraph = @ptrCast(@alignCast(cgraph_c));

    fusion.DisableFusion.once.call();
    if (!fusion.DisableFusion.value) {
        var i: c_int = 0;
        while (i < cgraph.n_nodes) : (i += 1) {
            if (cgraph.nodes[@intCast(i)].?.op != c.GGML_OP_MUL) continue;

            var match: fusion.MoeWeightedReductionMatch = .{};
            if (!fusion.matchMoeWeightedReduction(cgraph, i, &match)) continue;

            const params = impl.one(c.struct_ggml_backend_graph_optimize_params, params_c);
            const add_alloc_dep = params.add_alloc_dep.?;
            add_alloc_dep(params.user_data, @constCast(match.experts.?), match.dst.?);
            add_alloc_dep(params.user_data, @constCast(match.weights.?), match.dst.?);
            if (match.expert_scale) |expert_scale| {
                add_alloc_dep(params.user_data, @constCast(expert_scale), match.dst.?);
            }
            i += match.node_count - 1;
        }
    }

    const graph_key = graphGetKey(cgraph);
    const use_cuda_graph = graphSetEnabled(cuda_ctx, graph_key);

    GraphOpt.once.call();
    if (!GraphOpt.enabled) return;

    const stream_context = &cuda_ctx.concurrent_stream_context;
    stream_context.reset();

    if (!use_cuda_graph) return;

    common.setDevice(cuda_ctx.device);

    // number of out-degrees for a particular node
    var fan_out: std.AutoArrayHashMapUnmanaged(?*const Tensor, c_int) = .empty;
    defer fan_out.deinit(allocator);
    // reverse mapping of node to index in the cgraph
    var node_indices: std.AutoHashMapUnmanaged(*const Tensor, usize) = .empty;
    defer node_indices.deinit(allocator);

    const ns = nodes(cgraph);
    for (ns, 0..) |node, node_idx| {
        node_indices.put(allocator, node, node_idx) catch impl.abort("ggml-cuda: out of memory");
        if (isNoop(node)) continue;
        for (node.src) |src| {
            //TODO: check why nrows > 1 fails
            if (!isNoop(node) and c.ggml_nrows(node) <= 1) {
                const gop = fan_out.getOrPut(allocator, src) catch impl.abort("ggml-cuda: out of memory");
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* += 1;
            }
        }
    }

    const min_fan_out = 3;
    const max_fan_out = 3;

    // store {fork_idx, join_idx}
    var concurrent_node_ranges: std.ArrayList([2]usize) = .empty;
    defer concurrent_node_ranges.deinit(allocator);

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

        var nodes_per_branch: std.ArrayList(std.ArrayList(*const Tensor)) = .empty;
        defer {
            for (nodes_per_branch.items) |*b| b.deinit(allocator);
            nodes_per_branch.deinit(allocator);
        }
        for (ns[root_node_idx + 1 ..]) |node| {
            if (!isNoop(node) and dependsOn(node, root_node)) {
                var branch: std.ArrayList(*const Tensor) = .empty;
                branch.append(allocator, node) catch impl.abort("ggml-cuda: out of memory");
                nodes_per_branch.append(allocator, branch) catch impl.abort("ggml-cuda: out of memory");
            }
        }

        impl.assert(nodes_per_branch.items.len == @as(usize, @intCast(count)), "nodes_per_branch.size() == (size_t) count");

        //find the join point
        var join_node: ?*const Tensor = null;
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
                    if (std.mem.indexOfScalar(*const Tensor, b.items, curr_node) == null) {
                        b.append(allocator, curr_node) catch impl.abort("ggml-cuda: out of memory");
                    }
                    found_branch = true;
                }
            }

            if (!found_branch and isNoop(curr_node)) {
                // we can put it in any branch because it will be ignored
                nodes_per_branch.items[0].append(allocator, curr_node) catch impl.abort("ggml-cuda: out of memory");
            }
        }

        const join = join_node orelse continue;

        var event = common.ConcurrentEvent.init(@intCast(nodes_per_branch.items.len));
        var keep = false;
        defer if (!keep) event.deinit();
        event.join_node = join;

        for (nodes_per_branch.items, 0..) |b, branch_idx| {
            for (b.items) |n| event.stream_mapping.put(allocator, n, @intCast(branch_idx + 1)) catch impl.abort("ggml-cuda: out of memory");
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
            impl.logDebug("Skipping %s because the number of nodes in the middle is not equal to the total number of branch nodes %d != %d\n", .{ @as([*:0]const u8, @ptrCast(&root_node.name)), @as(c_int, @intCast(join_node_idx - fork_node_idx - 1)), @as(c_int, @intCast(total_branch_nodes)) });
            continue;
        }

        // Save the original order of nodes in this region before interleaving
        for (ns[fork_node_idx + 1 .. join_node_idx]) |n| event.original_order.append(allocator, n) catch impl.abort("ggml-cuda: out of memory");

        const concurrent_events = &stream_context.concurrent_events;
        impl.assert(!concurrent_events.contains(root_node), "concurrent_events.find(root_node) == concurrent_events.end()");
        concurrent_events.put(allocator, root_node, event) catch impl.abort("ggml-cuda: out of memory");
        keep = true;
        impl.logDebug("Adding stream at node %s %p\n", .{ @as([*:0]const u8, @ptrCast(&root_node.name)), root_node });
        concurrent_node_ranges.append(allocator, .{ fork_node_idx, join_node_idx }) catch impl.abort("ggml-cuda: out of memory");

        // interleave tensors to extend lifetimes so that ggml graph doesn't
        // recycle them
        var fronts: std.ArrayList(usize) = .empty;
        defer fronts.deinit(allocator);
        fronts.appendNTimes(allocator, 0, n_branches) catch impl.abort("ggml-cuda: out of memory");
        while (current_node_idx < join_node_idx) {
            const branch = nodes_per_branch.items[current_branch_idx].items;
            const front = &fronts.items[current_branch_idx];

            var has_node = false;
            for (nodes_per_branch.items, fronts.items) |b, f| has_node = has_node or b.items.len > f;
            impl.assert(has_node, "has_node");

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
