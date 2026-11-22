//! Planning a graph: how many threads each node wants, and how much scratch
//! the whole graph needs.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v1.0.0, `b1b770d90`),
//! `ggml_get_n_tasks` at line 2267 and `ggml_graph_plan` at line 2830. Each
//! function names the C it replaces and the line it began at.
//!
//! # The work buffer is one buffer, sized for the worst node
//!
//! `ggml_graph_plan` walks every node, asks how much scratch that node would
//! need, and keeps the maximum. Every node then computes into the same
//! allocation. So an underestimate anywhere is a buffer overrun somewhere
//! else entirely, which is why the arithmetic here has to agree exactly with
//! what the kernels do -- `mul_mat_id` in particular carves four separate
//! sub-allocations out of it, and `mulmat.zig` has to lay them out the same
//! way this file counts them.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

/// `ggml-cpu/traits.cpp`'s entry point, ported.
///
/// Fills `size` and returns true when an extra-buffer accelerator owns this
/// op, in which case its answer replaces the whole switch below.
/// Ports `GGML_IM2COL_WORK_SIZE` (ops.h:23 @b1b770d90).
const im2col_work_size: usize = 16 * 1024 * 1024;

/// Ports `GGML_FA_TILE_Q` and `GGML_FA_TILE_KV` (ggml-cpu/common.h:10 @b1b770d90).
const fa_tile_q: i64 = 64;
const fa_tile_kv: i64 = 64;

/// Ports `GGML_SOFT_MAX_UNROLL` (vec.h:48 @b1b770d90).
const soft_max_unroll: i64 = 4;

/// Ports `GGML_N_TASKS_MAX` (ggml.h:2527 @b1b770d90).
const n_tasks_max: c_int = -1;

/// Ports `ggml_up` (ggml-impl.h:68 @b1b770d90).
inline fn up(n: i64, m: i64) i64 {
    ggml.impl.assert(m & (m - 1) == 0, "(m & (m - 1)) == 0");
    return (n + m - 1) & ~(m - 1);
}

/// Ports `ggml_get_n_tasks` (ggml-cpu.c:2261 @b1b770d90).
///
/// Parameters:
/// - `node`: the graph node to plan for.
/// - `n_threads`: the thread budget for the graph.
///
/// Return: how many threads this node's kernel can use, at least 1.
pub fn getNTasks(node: *ggml.impl.ggml_c.ggml_tensor, n_threads: c_int) c_int {
    // No point multi-threading a no-op.
    if (ggml.types.ggml_is_empty(node)) return 1;

    const n_tasks: c_int = switch (node.op) {
        // Parallel over rows: every one of these splits its work by thread.
        ggml.impl.ggml_c.GGML_OP_CPY, ggml.impl.ggml_c.GGML_OP_DUP, ggml.impl.ggml_c.GGML_OP_CONT, ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_ADD_ID, ggml.impl.ggml_c.GGML_OP_ACC, ggml.impl.ggml_c.GGML_OP_CUMSUM, ggml.impl.ggml_c.GGML_OP_TRI, ggml.impl.ggml_c.GGML_OP_FILL, ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL, ggml.impl.ggml_c.GGML_OP_SOLVE_TRI, ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET, ggml.impl.ggml_c.GGML_OP_DSV4_HC_COMB, ggml.impl.ggml_c.GGML_OP_DSV4_HC_PRE, ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST, ggml.impl.ggml_c.GGML_OP_TOP_K_MASK, ggml.impl.ggml_c.GGML_OP_SILU_BACK, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_DIV, ggml.impl.ggml_c.GGML_OP_NORM, ggml.impl.ggml_c.GGML_OP_RMS_NORM, ggml.impl.ggml_c.GGML_OP_RMS_NORM_BACK, ggml.impl.ggml_c.GGML_OP_L2_NORM, ggml.impl.ggml_c.GGML_OP_GROUP_NORM, ggml.impl.ggml_c.GGML_OP_CONCAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID, ggml.impl.ggml_c.GGML_OP_OUT_PROD, ggml.impl.ggml_c.GGML_OP_SET_ROWS, ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK, ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_ROPE_BACK, ggml.impl.ggml_c.GGML_OP_ADD_REL_POS, ggml.impl.ggml_c.GGML_OP_IM2COL, ggml.impl.ggml_c.GGML_OP_IM2COL_BACK, ggml.impl.ggml_c.GGML_OP_IM2COL_3D, ggml.impl.ggml_c.GGML_OP_CONV_2D, ggml.impl.ggml_c.GGML_OP_CONV_3D, ggml.impl.ggml_c.GGML_OP_CONV_2D_DW, ggml.impl.ggml_c.GGML_OP_COL2IM_1D, ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D, ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D, ggml.impl.ggml_c.GGML_OP_UPSCALE, ggml.impl.ggml_c.GGML_OP_PAD, ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D, ggml.impl.ggml_c.GGML_OP_ROLL, ggml.impl.ggml_c.GGML_OP_ARANGE, ggml.impl.ggml_c.GGML_OP_TIMESTEP_EMBEDDING, ggml.impl.ggml_c.GGML_OP_ARGSORT, ggml.impl.ggml_c.GGML_OP_TOP_K, ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_BACK, ggml.impl.ggml_c.GGML_OP_SSM_CONV, ggml.impl.ggml_c.GGML_OP_SSM_SCAN, ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER, ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS, ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS_BACK, ggml.impl.ggml_c.GGML_OP_OPT_STEP_ADAMW, ggml.impl.ggml_c.GGML_OP_OPT_STEP_SGD => n_threads,

        // Single-threaded, either because the op is cheap or because no
        // one has written the parallel form. `GET_ROWS` could use more
        // threads; the C notes that launching them costs more than it saves
        // once a GPU is carrying the model. `SET_ROWS` is not here
        // (#30454): it plans its per-thread F16 conversion rows for every
        // thread that runs it.
        ggml.impl.ggml_c.GGML_OP_GET_ROWS, ggml.impl.ggml_c.GGML_OP_SUB, ggml.impl.ggml_c.GGML_OP_SQR, ggml.impl.ggml_c.GGML_OP_SQRT, ggml.impl.ggml_c.GGML_OP_LOG, ggml.impl.ggml_c.GGML_OP_SIN, ggml.impl.ggml_c.GGML_OP_COS, ggml.impl.ggml_c.GGML_OP_SUM, ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_MEAN, ggml.impl.ggml_c.GGML_OP_ARGMAX, ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_REPEAT_BACK, ggml.impl.ggml_c.GGML_OP_LEAKY_RELU, ggml.impl.ggml_c.GGML_OP_SCALE, ggml.impl.ggml_c.GGML_OP_SET, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_PERMUTE, ggml.impl.ggml_c.GGML_OP_TRANSPOSE, ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK, ggml.impl.ggml_c.GGML_OP_DIAG, ggml.impl.ggml_c.GGML_OP_CLAMP, ggml.impl.ggml_c.GGML_OP_POOL_1D, ggml.impl.ggml_c.GGML_OP_POOL_2D, ggml.impl.ggml_c.GGML_OP_POOL_2D_BACK, ggml.impl.ggml_c.GGML_OP_WIN_PART, ggml.impl.ggml_c.GGML_OP_WIN_UNPART, ggml.impl.ggml_c.GGML_OP_GET_REL_POS, ggml.impl.ggml_c.GGML_OP_NONE => 1,

        ggml.impl.ggml_c.GGML_OP_UNARY => switch (ggml.types.ggml_get_unary_op(node)) {
            ggml.impl.ggml_c.GGML_UNARY_OP_ABS,
            ggml.impl.ggml_c.GGML_UNARY_OP_SGN,
            ggml.impl.ggml_c.GGML_UNARY_OP_NEG,
            ggml.impl.ggml_c.GGML_UNARY_OP_STEP,
            ggml.impl.ggml_c.GGML_UNARY_OP_TANH,
            ggml.impl.ggml_c.GGML_UNARY_OP_ELU,
            ggml.impl.ggml_c.GGML_UNARY_OP_RELU,
            ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID,
            ggml.impl.ggml_c.GGML_UNARY_OP_HARDSWISH,
            ggml.impl.ggml_c.GGML_UNARY_OP_HARDSIGMOID,
            ggml.impl.ggml_c.GGML_UNARY_OP_EXP,
            ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS,
            ggml.impl.ggml_c.GGML_UNARY_OP_EXPM1,
            ggml.impl.ggml_c.GGML_UNARY_OP_FLOOR,
            ggml.impl.ggml_c.GGML_UNARY_OP_CEIL,
            ggml.impl.ggml_c.GGML_UNARY_OP_ROUND,
            ggml.impl.ggml_c.GGML_UNARY_OP_TRUNC,
            => 1,

            // The four that cost enough per element to be worth splitting.
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU,
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU_ERF,
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU_QUICK,
            ggml.impl.ggml_c.GGML_UNARY_OP_SILU,
            ggml.impl.ggml_c.GGML_UNARY_OP_XIELU,
            => n_threads,

            else => ggml.impl.abort("fatal error"),
        },

        ggml.impl.ggml_c.GGML_OP_GLU => switch (ggml.types.ggml_get_glu_op(node)) {
            ggml.impl.ggml_c.GGML_GLU_OP_REGLU,
            ggml.impl.ggml_c.GGML_GLU_OP_GEGLU,
            ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU,
            ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_OAI,
            ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_ERF,
            ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_QUICK,
            ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_CLAMP,
            => n_threads,

            else => ggml.impl.abort("fatal error"),
        },

        // Bounded by the rows there are to split.
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX => @intCast(@min(n_threads, ggml.types.ggml_nrows(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0])))),

        // Bounded by the head count, which is where these parallelise.
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV6,
        ggml.impl.ggml_c.GGML_OP_GATED_LINEAR_ATTN,
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV7,
        => @intCast(@min(n_threads, ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]).ne[1])),

        // The caller chose a task count when it built the node. All four
        // op-params layouts put `n_tasks` in the same place.
        ggml.impl.ggml_c.GGML_OP_MAP_CUSTOM1,
        ggml.impl.ggml_c.GGML_OP_MAP_CUSTOM2,
        ggml.impl.ggml_c.GGML_OP_MAP_CUSTOM3,
        ggml.impl.ggml_c.GGML_OP_CUSTOM,
        => blk: {
            var p: ggml.impl.CustomOpParams = undefined;
            @memcpy(std.mem.asBytes(&p), std.mem.asBytes(&node.op_params)[0..@sizeOf(ggml.impl.CustomOpParams)]);
            break :blk if (p.n_tasks == n_tasks_max) n_threads else @intCast(@min(p.n_tasks, n_threads));
        },

        ggml.impl.ggml_c.GGML_OP_COUNT => ggml.impl.abort("fatal error"),

        else => {
            // The C prints the op's name before aborting, so an op added
            // upstream and forgotten here says which one it was.
            if (node.op < ggml.impl.ggml_c.GGML_OP_COUNT) {
                ggml.impl.fprintfStderr("ggml_get_n_tasks: op not implemented: %s\n", .{ggml.types.ggml_op_name(node.op)});
            } else {
                ggml.impl.fprintfStderr("ggml_get_n_tasks: op not implemented: %d\n", .{node.op});
            }
            ggml.impl.abort("fatal error");
        },
    };

    std.debug.assert(n_tasks > 0);
    return n_tasks;
}

/// Ports `ggml_graph_plan` (ggml-cpu.c:2821 @b1b770d90).
///
/// Parameters:
/// - `cgraph`: the graph to plan.
/// - `n_threads_in`: requested thread count; 0 or less takes the threadpool's,
///   or `GGML_DEFAULT_N_THREADS` when there is no pool.
/// - `threadpool`: the pool the plan will run on, or null for a disposable one.
///
/// Return: the plan, by value. `work_data` is left null for the caller to
/// allocate `work_size` bytes into.
pub fn ggml_graph_plan(
    cgraph: *const ggml.impl.CGraph,
    n_threads_in: c_int,
    threadpool: ?*ggml.cpu.defs.Threadpool,
) callconv(.c) ggml.impl.ggml_c.struct_ggml_cplan {
    var n_threads = n_threads_in;
    if (n_threads <= 0) {
        n_threads = if (threadpool) |tp| tp.n_threads else ggml.impl.ggml_c.GGML_DEFAULT_N_THREADS;
    }

    var work_size: usize = 0;

    var cplan: ggml.impl.ggml_c.struct_ggml_cplan = std.mem.zeroes(ggml.impl.ggml_c.struct_ggml_cplan);

    var max_tasks: c_int = 1;

    // Thread scheduling per op, and the work-buffer estimate, in one pass.
    for (0..@intCast(cgraph.n_nodes)) |i| {
        const node = cgraph.nodes[i].?;

        const n_tasks = getNTasks(node, n_threads);
        const tasks: i64 = n_tasks;

        max_tasks = @max(max_tasks, n_tasks);

        var cur: usize = 0;

        if (!ggml.cpu.extra.ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            cur = nodeWorkSize(node, tasks);
        }

        work_size = @max(work_size, cur);
    }

    // One cache line per thread of slack, so threads writing into their own
    // slice of the buffer do not share a line at the boundary.
    if (work_size > 0) {
        work_size += ggml.cpu.defs.cache_line_size * @as(usize, @intCast(n_threads));
    }

    cplan.threadpool = @ptrCast(threadpool);
    cplan.n_threads = @min(max_tasks, n_threads);
    cplan.work_size = work_size;
    cplan.work_data = null;

    return cplan;
}

/// The scratch one node needs, in bytes.
///
/// Split out of `ggml_graph_plan`'s loop: the C writes this as a 200-line
/// `switch` nested inside a `for` inside an `if`, which in Zig would run four
/// levels deep before the first case.
///
/// Parameters:
/// - `node`: the node to size.
/// - `n_tasks`: how many threads will run it, from `getNTasks`.
///
/// Return: bytes of work buffer the node's kernel will use.
fn nodeWorkSize(node: *const ggml.impl.ggml_c.ggml_tensor, n_tasks: i64) usize {
    const f32_size = ggml.types.ggml_type_size(ggml.impl.ggml_c.GGML_TYPE_F32);

    switch (node.op) {
        ggml.impl.ggml_c.GGML_OP_CPY, ggml.impl.ggml_c.GGML_OP_DUP => {
            const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]);

            // A `CONT` has no second source, so every pair below is guarded.
            const src1_type: ?ggml.impl.ggml_c.enum_ggml_type =
                if (node.src[1] == null) null else ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]).type;

            const via_f32 = ggml.types.ggml_is_quantized(node.type) or
                // F16 <-> BF16 copies go through an intermediate F32.
                (src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 and src1_type == ggml.impl.ggml_c.GGML_TYPE_BF16) or
                (src0.type == ggml.impl.ggml_c.GGML_TYPE_BF16 and src1_type == ggml.impl.ggml_c.GGML_TYPE_F16) or
                // So does conversion between F32 and I32.
                (src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 and src1_type == ggml.impl.ggml_c.GGML_TYPE_I32) or
                (src0.type == ggml.impl.ggml_c.GGML_TYPE_I32 and src1_type == ggml.impl.ggml_c.GGML_TYPE_F32);

            if (via_f32) return f32_size * @as(usize, @intCast(node.ne[0] * n_tasks));
            return 0;
        },

        ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_ADD_ID => {
            const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]);
            if (ggml.types.ggml_is_quantized(src0.type) or src0.type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3) {
                return f32_size * @as(usize, @intCast(src0.ne[0] * n_tasks));
            }
            return 0;
        },

        ggml.impl.ggml_c.GGML_OP_ACC => {
            // Sized from `src[1]`, the block being written, not `src[0]`.
            if (ggml.types.ggml_is_quantized(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]).type)) {
                return f32_size * @as(usize, @intCast(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]).ne[0] * n_tasks));
            }
            return 0;
        },

        ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL => return ggml.types.ggml_type_size(node.type) * @as(usize, @intCast(n_tasks)),

        ggml.impl.ggml_c.GGML_OP_MUL_MAT => {
            const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]);
            const vec_dot_type = ggml.cpu.traits.table[@intCast(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]).type)].vec_dot_type;

            var cur: usize = 0;
            if (src1.type != vec_dot_type) {
                cur = ggml.types.ggml_row_size(vec_dot_type, ggml.types.ggml_nelements(src1));
            }

            // Workspace for tiled (see tiled.h)
            cur = stdlib.num.pad(usize, cur, 64);
            cur += ggml.cpu.tiled.ggml_tiled_wdata_size(@intCast(n_tasks), node);
            return cur;
        },

        ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID => {
            // Four sub-allocations, each rounded up by `incr_ptr_aligned` in
            // `mulmat.zig`. The trailing `sizeof(int64_t)` on the first three
            // and the trailing cache line on the fourth are that rounding's
            // worst case.
            const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]);
            const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]);
            const ids = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[2]);
            const vec_dot_type = ggml.cpu.traits.table[@intCast(src0.type)].vec_dot_type;
            const n_as: usize = @intCast(src0.ne[2]);

            var cur: usize = 0;

            // src1, staged into the kernel's type.
            if (src1.type != vec_dot_type) {
                cur += ggml.types.ggml_row_size(vec_dot_type, ggml.types.ggml_nelements(src1)) + @sizeOf(i64);
            }
            // matrix_row_counts
            cur += n_as * @sizeOf(i64) + @sizeOf(i64);
            // matrix_rows
            cur += @as(usize, @intCast(@as(i64, @intCast(n_as)) * ids.ne[0] * ids.ne[1])) *
                @sizeOf(ggml.cpu.mulmat.RowMapping) + @sizeOf(i64);
            // atomic_current_chunk
            cur += ggml.cpu.defs.cache_line_size * n_as + ggml.cpu.defs.cache_line_size;
            // Workspace for tiled (see tiled.h)
            cur = stdlib.num.pad(usize, cur, 64);
            cur += ggml.cpu.tiled.ggml_tiled_wdata_size(@intCast(n_tasks), node);

            return cur;
        },

        ggml.impl.ggml_c.GGML_OP_OUT_PROD => {
            const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]);
            if (ggml.types.ggml_is_quantized(src0.type) or src0.type == ggml.impl.ggml_c.GGML_TYPE_F8_E4M3 or
                src0.type == ggml.impl.ggml_c.GGML_TYPE_F16)
            {
                return f32_size * @as(usize, @intCast(src0.ne[0] * n_tasks));
            }
            return 0;
        },

        ggml.impl.ggml_c.GGML_OP_SET_ROWS => {
            const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]);
            if (src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 and node.type != ggml.impl.ggml_c.GGML_TYPE_F16) {
                return f32_size * @as(usize, @intCast(src0.ne[0] * n_tasks));
            }
            return 0;
        },

        ggml.impl.ggml_c.GGML_OP_SOFT_MAX, ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_ROPE_BACK => {
            return f32_size * @as(usize, @intCast(node.ne[0] * n_tasks));
        },

        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D => {
            const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]);
            const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]);

            ggml.impl.assert(src0.ne[3] == 1, "node->src[0]->ne[3] == 1");
            ggml.impl.assert(src1.ne[2] == 1, "node->src[1]->ne[2] == 1");
            ggml.impl.assert(src1.ne[3] == 1, "node->src[1]->ne[3] == 1");

            const ne00 = src0.ne[0]; // K
            const ne01 = src0.ne[1]; // Cout
            const ne02 = src0.ne[2]; // Cin
            const ne10 = src1.ne[0]; // L
            const ne11 = src1.ne[1]; // Cin

            if ((src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src0.type == ggml.impl.ggml_c.GGML_TYPE_BF16) and
                src1.type == ggml.impl.ggml_c.GGML_TYPE_F32)
            {
                // The source is permuted into F32 (#31854); only
                // the kernel is stored as F16.
                return @sizeOf(f32) * @as(usize, @intCast(ne10 * ne11)) +
                    @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) * @as(usize, @intCast(ne00 * ne01 * ne02));
            } else if (src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 and src1.type == ggml.impl.ggml_c.GGML_TYPE_F32) {
                return @sizeOf(f32) * @as(usize, @intCast(ne00 * ne01 * ne02 + ne10 * ne11));
            }
            ggml.impl.abort("fatal error");
        },

        ggml.impl.ggml_c.GGML_OP_CONV_2D, ggml.impl.ggml_c.GGML_OP_CONV_3D => return im2col_work_size,

        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D => {
            const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]);
            const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]);

            const ne00 = src0.ne[0]; // W
            const ne01 = src0.ne[1]; // H
            const ne02 = src0.ne[2]; // Channels out
            const ne03 = src0.ne[3]; // Channels in

            const ne10 = src1.ne[0]; // W
            const ne11 = src1.ne[1]; // H
            const ne12 = src1.ne[2]; // Channels in
            const ne13 = src1.ne[3]; // Batch

            ggml.impl.assert(
                src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F32,
                "node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32",
            );
            ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "node->src[1]->type == GGML_TYPE_F32");

            // The source is permuted into F32 (#31854), ahead of
            // the kernel in its own type.
            return @sizeOf(f32) * @as(usize, @intCast(ne10 * ne11 * ne12 * ne13)) +
                ggml.types.ggml_type_size(src0.type) * @as(usize, @intCast(ne00 * ne01 * ne02 * ne03));
        },

        ggml.impl.ggml_c.GGML_OP_TOP_K => {
            return @sizeOf(i32) * @as(usize, @intCast(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]).ne[0] * n_tasks));
        },

        ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT => {
            const neq2 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]).ne[2]; // query heads
            const dk = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]).ne[0]; // key width
            const dv = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[2]).ne[0]; // value width

            // Prefill: per thread, Q_q + KQ + mask + VKQ32 + V32 + K_f32.
            const prefill = @sizeOf(f32) * @as(usize, @intCast((fa_tile_q * dk +
                2 * fa_tile_q * fa_tile_kv + fa_tile_q * dv +
                fa_tile_kv * dv + fa_tile_kv * dk) * n_tasks));

            // Decode: one KV chunk per thread. Per thread, the VKQ
            // accumulator plus partial M and S, plus scratch for V, Q, VKQ.
            const n_chunks = n_tasks;
            const decode = @sizeOf(f32) * @as(usize, @intCast(neq2 * n_chunks * (2 + dv) +
                n_tasks * (dk + 2 * dv)));

            return @max(prefill, decode);
        },

        ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_BACK => {
            const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]);
            const d = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]).ne[0];
            const ne11 = up(src1.ne[1], soft_max_unroll);
            // Doubled for S and SM in the kernel.
            const mx_dn = @max(d, ne11) * 2;

            const t = src1.type;
            if (t == ggml.impl.ggml_c.GGML_TYPE_F32 or t == ggml.impl.ggml_c.GGML_TYPE_F16 or t == ggml.impl.ggml_c.GGML_TYPE_BF16) {
                // The C writes this as two identical additions with a note
                // that the second overestimates by 2x. Kept as written.
                return 2 * @sizeOf(f32) * @as(usize, @intCast(mx_dn * n_tasks));
            }
            return 0;
        },

        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS => {
            return ggml.types.ggml_type_size(node.type) *
                @as(usize, @intCast(n_tasks + ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[0]).ne[0] * n_tasks));
        },

        ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET => {
            const s_v = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[2]).ne[0];
            const k: i64 = ggml.impl.getOpParamsI32(node, 0);
            const per_thread = s_v + (if (k > 1) s_v * s_v else 0);
            return @as(usize, @intCast(per_thread * n_tasks)) * @sizeOf(f32);
        },

        ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER => {
            // Scratch for dequantising the indexer keys.
            return @sizeOf(f32) * @as(usize, @intCast(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]).ne[0] * n_tasks));
        },

        ggml.impl.ggml_c.GGML_OP_COUNT => ggml.impl.abort("fatal error"),

        else => return 0,
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "ggml_up rounds to a power of two" {
    try std.testing.expectEqual(@as(i64, 4), up(1, 4));
    try std.testing.expectEqual(@as(i64, 4), up(4, 4));
    try std.testing.expectEqual(@as(i64, 8), up(5, 4));
    try std.testing.expectEqual(@as(i64, 0), up(0, 4));
}

test "n_tasks is 1 for an empty node whatever its op" {
    // Checked before the switch, so even an op that would otherwise take
    // every thread gets one.
    var node: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    node.op = ggml.impl.ggml_c.GGML_OP_MUL_MAT;
    node.ne = .{ 0, 1, 1, 1 };

    try std.testing.expectEqual(@as(c_int, 1), getNTasks(&node, 8));
}

test "n_tasks splits the parallel ops and pins the serial ones" {
    var node: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    node.ne = .{ 4, 4, 1, 1 };
    node.nb = .{ 4, 16, 64, 64 };
    node.type = ggml.impl.ggml_c.GGML_TYPE_F32;

    node.op = ggml.impl.ggml_c.GGML_OP_MUL_MAT;
    try std.testing.expectEqual(@as(c_int, 8), getNTasks(&node, 8));

    node.op = ggml.impl.ggml_c.GGML_OP_SUM;
    try std.testing.expectEqual(@as(c_int, 1), getNTasks(&node, 8));

    // `CLAMP` is 1 with a TODO in the C, and reads like an oversight. It is
    // not one to fix here.
    node.op = ggml.impl.ggml_c.GGML_OP_CLAMP;
    try std.testing.expectEqual(@as(c_int, 1), getNTasks(&node, 8));
}

test "soft_max asks for no more threads than there are rows" {
    var src: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    src.ne = .{ 16, 3, 1, 1 };

    var node: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    node.op = ggml.impl.ggml_c.GGML_OP_SOFT_MAX;
    node.ne = .{ 16, 3, 1, 1 };
    node.src[0] = &src;

    try std.testing.expectEqual(@as(c_int, 3), getNTasks(&node, 8));
    try std.testing.expectEqual(@as(c_int, 2), getNTasks(&node, 2));
}

test "a custom op's task count comes out of op_params" {
    var node: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    node.op = ggml.impl.ggml_c.GGML_OP_CUSTOM;
    node.ne = .{ 4, 1, 1, 1 };

    const p = ggml.impl.CustomOpParams{ .fun = null, .n_tasks = 3, .userdata = null };
    @memcpy(std.mem.asBytes(&node.op_params)[0..@sizeOf(ggml.impl.CustomOpParams)], std.mem.asBytes(&p));
    try std.testing.expectEqual(@as(c_int, 3), getNTasks(&node, 8));
    try std.testing.expectEqual(@as(c_int, 2), getNTasks(&node, 2));

    const maxed = ggml.impl.CustomOpParams{ .fun = null, .n_tasks = n_tasks_max, .userdata = null };
    @memcpy(std.mem.asBytes(&node.op_params)[0..@sizeOf(ggml.impl.CustomOpParams)], std.mem.asBytes(&maxed));
    try std.testing.expectEqual(@as(c_int, 8), getNTasks(&node, 8));
}

test "mul_mat_id reserves room for all four sub-allocations" {
    // The kernel carves these out with `incr_ptr_aligned`, so the count here
    // has to cover the alignment slack as well as the blocks themselves. An
    // underestimate is an overrun of a shared buffer.
    var src0: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    src0.type = ggml.impl.ggml_c.GGML_TYPE_F32;
    src0.ne = .{ 8, 4, 3, 1 }; // 3 experts

    var src1: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    src1.type = ggml.impl.ggml_c.GGML_TYPE_F32;
    src1.ne = .{ 8, 2, 1, 1 };
    src1.nb = .{ 4, 32, 64, 64 };

    var ids: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    ids.type = ggml.impl.ggml_c.GGML_TYPE_I32;
    ids.ne = .{ 2, 2, 1, 1 };

    var node: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    node.op = ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID;
    node.ne = .{ 4, 2, 2, 1 };
    node.src[0] = &src0;
    node.src[1] = &src1;
    node.src[2] = &ids;

    const size = nodeWorkSize(&node, 4);

    // src1 is already F32, which is `vec_dot_type` for an F32 src0, so the
    // staging block is skipped and only the last three are counted.
    const counts = 3 * @sizeOf(i64) + @sizeOf(i64);
    const rows = 3 * 2 * 2 * @sizeOf(ggml.cpu.mulmat.RowMapping) + @sizeOf(i64);
    const chunks = ggml.cpu.defs.cache_line_size * 3 + ggml.cpu.defs.cache_line_size;
    // Padded to 64 for the tiled workspace, which an F32 src0
    // never gets.
    try std.testing.expectEqual(stdlib.num.pad(usize, counts + rows + chunks, 64) + ggml.cpu.tiled.ggml_tiled_wdata_size(4, &node), size);
    try std.testing.expectEqual(@as(usize, 0), ggml.cpu.tiled.ggml_tiled_wdata_size(4, &node));
}

test "a node with no scratch needs none" {
    var src0: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    src0.type = ggml.impl.ggml_c.GGML_TYPE_F32;
    src0.ne = .{ 4, 1, 1, 1 };

    var node: ggml.impl.ggml_c.ggml_tensor = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    node.op = ggml.impl.ggml_c.GGML_OP_ADD;
    node.ne = .{ 4, 1, 1, 1 };
    node.src[0] = &src0;

    try std.testing.expectEqual(@as(usize, 0), nodeWorkSize(&node, 4));
}
