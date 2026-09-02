//! Planning a graph: how many threads each node wants, and how much scratch
//! the whole graph needs.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! `ggml_get_n_tasks` at line 2220 and `ggml_graph_plan` at line 2781. Each
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
const impl = @import("../impl.zig");
const types = @import("../types.zig");
const defs = @import("defs.zig");
const traits = @import("traits.zig");
const mulmat = @import("mulmat.zig");
const c = impl.c;

const Tensor = defs.Tensor;
const CGraph = defs.CGraph;

/// `ggml-cpu/traits.cpp`, still C++.
///
/// Fills `size` and returns true when an extra-buffer accelerator owns this
/// op, in which case its answer replaces the whole switch below.
extern fn ggml_cpu_extra_work_size(n_threads: c_int, op: *const Tensor, size: *usize) bool;

/// Ports `GGML_IM2COL_WORK_SIZE` (ops.h:24 @c1d0e7a00).
const im2col_work_size: usize = 16 * 1024 * 1024;

/// Ports `GGML_FA_TILE_Q` and `GGML_FA_TILE_KV` (ggml-cpu/common.h:10 @c1d0e7a00).
const fa_tile_q: i64 = 64;
const fa_tile_kv: i64 = 64;

/// Ports `GGML_SOFT_MAX_UNROLL` (vec.h:49 @c1d0e7a00).
const soft_max_unroll: i64 = 4;

/// Ports `GGML_N_TASKS_MAX` (ggml.h:2663 @c1d0e7a00).
const n_tasks_max: c_int = -1;

/// Ports `ggml_up` (ggml-impl.h:68 @c1d0e7a00).
inline fn up(n: i64, m: i64) i64 {
    impl.assert(m & (m - 1) == 0, "(m & (m - 1)) == 0");
    return (n + m - 1) & ~(m - 1);
}

inline fn min(a: i64, b: i64) i64 {
    return if (a < b) a else b;
}

/// Ports `ggml_get_n_tasks` (ggml-cpu.c:2220 @c1d0e7a00).
///
/// Parameters:
/// - `node`: the graph node to plan for.
/// - `n_threads`: the thread budget for the graph.
///
/// Return: how many threads this node's kernel can use, at least 1.
pub fn getNTasks(node: *Tensor, n_threads: c_int) c_int {
    // No point multi-threading a no-op.
    if (types.ggml_is_empty(node)) return 1;

    const n_tasks: c_int = switch (node.op) {
        // Parallel over rows: every one of these splits its work by thread.
        c.GGML_OP_CPY, c.GGML_OP_DUP, c.GGML_OP_CONT, c.GGML_OP_ADD, c.GGML_OP_ADD_ID, c.GGML_OP_ADD1, c.GGML_OP_ACC, c.GGML_OP_CUMSUM, c.GGML_OP_TRI, c.GGML_OP_FILL, c.GGML_OP_COUNT_EQUAL, c.GGML_OP_SOLVE_TRI, c.GGML_OP_GATED_DELTA_NET, c.GGML_OP_DSV4_HC_COMB, c.GGML_OP_DSV4_HC_PRE, c.GGML_OP_DSV4_HC_POST, c.GGML_OP_SILU_BACK, c.GGML_OP_MUL, c.GGML_OP_DIV, c.GGML_OP_NORM, c.GGML_OP_RMS_NORM, c.GGML_OP_RMS_NORM_BACK, c.GGML_OP_L2_NORM, c.GGML_OP_GROUP_NORM, c.GGML_OP_CONCAT, c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT_ID, c.GGML_OP_OUT_PROD, c.GGML_OP_GET_ROWS, c.GGML_OP_SET_ROWS, c.GGML_OP_DIAG_MASK_ZERO, c.GGML_OP_DIAG_MASK_INF, c.GGML_OP_SOFT_MAX_BACK, c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK, c.GGML_OP_ADD_REL_POS, c.GGML_OP_IM2COL, c.GGML_OP_IM2COL_BACK, c.GGML_OP_IM2COL_3D, c.GGML_OP_CONV_2D, c.GGML_OP_CONV_3D, c.GGML_OP_CONV_2D_DW, c.GGML_OP_COL2IM_1D, c.GGML_OP_CONV_TRANSPOSE_1D, c.GGML_OP_CONV_TRANSPOSE_2D, c.GGML_OP_UPSCALE, c.GGML_OP_PAD, c.GGML_OP_PAD_REFLECT_1D, c.GGML_OP_ROLL, c.GGML_OP_ARANGE, c.GGML_OP_TIMESTEP_EMBEDDING, c.GGML_OP_ARGSORT, c.GGML_OP_TOP_K, c.GGML_OP_FLASH_ATTN_EXT, c.GGML_OP_FLASH_ATTN_BACK, c.GGML_OP_SSM_CONV, c.GGML_OP_SSM_SCAN, c.GGML_OP_LIGHTNING_INDEXER, c.GGML_OP_CROSS_ENTROPY_LOSS, c.GGML_OP_CROSS_ENTROPY_LOSS_BACK, c.GGML_OP_OPT_STEP_ADAMW, c.GGML_OP_OPT_STEP_SGD => n_threads,

        // Single-threaded, either because the op is cheap or because no
        // one has written the parallel form. `GET_ROWS` and `SET_ROWS`
        // could use more threads; the C notes that launching them costs
        // more than it saves once a GPU is carrying the model.
        c.GGML_OP_SUB, c.GGML_OP_SQR, c.GGML_OP_SQRT, c.GGML_OP_LOG, c.GGML_OP_SIN, c.GGML_OP_COS, c.GGML_OP_SUM, c.GGML_OP_SUM_ROWS, c.GGML_OP_MEAN, c.GGML_OP_ARGMAX, c.GGML_OP_REPEAT, c.GGML_OP_REPEAT_BACK, c.GGML_OP_LEAKY_RELU, c.GGML_OP_SCALE, c.GGML_OP_SET, c.GGML_OP_RESHAPE, c.GGML_OP_VIEW, c.GGML_OP_PERMUTE, c.GGML_OP_TRANSPOSE, c.GGML_OP_GET_ROWS_BACK, c.GGML_OP_DIAG, c.GGML_OP_CLAMP, c.GGML_OP_POOL_1D, c.GGML_OP_POOL_2D, c.GGML_OP_POOL_2D_BACK, c.GGML_OP_WIN_PART, c.GGML_OP_WIN_UNPART, c.GGML_OP_GET_REL_POS, c.GGML_OP_NONE => 1,

        c.GGML_OP_UNARY => switch (types.ggml_get_unary_op(node)) {
            c.GGML_UNARY_OP_ABS,
            c.GGML_UNARY_OP_SGN,
            c.GGML_UNARY_OP_NEG,
            c.GGML_UNARY_OP_STEP,
            c.GGML_UNARY_OP_TANH,
            c.GGML_UNARY_OP_ELU,
            c.GGML_UNARY_OP_RELU,
            c.GGML_UNARY_OP_SIGMOID,
            c.GGML_UNARY_OP_HARDSWISH,
            c.GGML_UNARY_OP_HARDSIGMOID,
            c.GGML_UNARY_OP_EXP,
            c.GGML_UNARY_OP_SOFTPLUS,
            c.GGML_UNARY_OP_EXPM1,
            c.GGML_UNARY_OP_FLOOR,
            c.GGML_UNARY_OP_CEIL,
            c.GGML_UNARY_OP_ROUND,
            c.GGML_UNARY_OP_TRUNC,
            => 1,

            // The four that cost enough per element to be worth splitting.
            c.GGML_UNARY_OP_GELU,
            c.GGML_UNARY_OP_GELU_ERF,
            c.GGML_UNARY_OP_GELU_QUICK,
            c.GGML_UNARY_OP_SILU,
            c.GGML_UNARY_OP_XIELU,
            => n_threads,

            else => impl.abort("fatal error"),
        },

        c.GGML_OP_GLU => switch (types.ggml_get_glu_op(node)) {
            c.GGML_GLU_OP_REGLU,
            c.GGML_GLU_OP_GEGLU,
            c.GGML_GLU_OP_SWIGLU,
            c.GGML_GLU_OP_SWIGLU_OAI,
            c.GGML_GLU_OP_GEGLU_ERF,
            c.GGML_GLU_OP_GEGLU_QUICK,
            => n_threads,

            else => impl.abort("fatal error"),
        },

        // Bounded by the rows there are to split.
        c.GGML_OP_SOFT_MAX => @intCast(min(n_threads, types.ggml_nrows(impl.one(Tensor, node.src[0])))),

        // Bounded by the head count, which is where these parallelise.
        c.GGML_OP_RWKV_WKV6,
        c.GGML_OP_GATED_LINEAR_ATTN,
        c.GGML_OP_RWKV_WKV7,
        => @intCast(min(n_threads, impl.one(Tensor, node.src[1]).ne[1])),

        // The caller chose a task count when it built the node. All four
        // op-params layouts put `n_tasks` in the same place.
        c.GGML_OP_MAP_CUSTOM1,
        c.GGML_OP_MAP_CUSTOM2,
        c.GGML_OP_MAP_CUSTOM3,
        c.GGML_OP_CUSTOM,
        => blk: {
            var p: impl.CustomOpParams = undefined;
            @memcpy(std.mem.asBytes(&p), std.mem.asBytes(&node.op_params)[0..@sizeOf(impl.CustomOpParams)]);
            break :blk if (p.n_tasks == n_tasks_max) n_threads else @intCast(min(p.n_tasks, n_threads));
        },

        c.GGML_OP_COUNT => impl.abort("fatal error"),

        else => {
            // The C prints the op's name before aborting, so an op added
            // upstream and forgotten here says which one it was.
            if (node.op < c.GGML_OP_COUNT) {
                impl.logError("ggml_get_n_tasks: op not implemented: %s\n", .{types.ggml_op_name(node.op)});
            } else {
                impl.logError("ggml_get_n_tasks: op not implemented: %d\n", .{node.op});
            }
            impl.abort("fatal error");
        },
    };

    std.debug.assert(n_tasks > 0);
    return n_tasks;
}

/// Ports `ggml_graph_plan` (ggml-cpu.c:2781 @c1d0e7a00).
///
/// Parameters:
/// - `cgraph`: the graph to plan.
/// - `n_threads_in`: requested thread count; 0 or less takes the threadpool's,
///   or `GGML_DEFAULT_N_THREADS` when there is no pool.
/// - `threadpool`: the pool the plan will run on, or null for a disposable one.
///
/// Return: the plan, by value. `work_data` is left null for the caller to
/// allocate `work_size` bytes into.
pub export fn ggml_graph_plan(
    cgraph: *const CGraph,
    n_threads_in: c_int,
    threadpool: ?*defs.Threadpool,
) c.struct_ggml_cplan {
    var n_threads = n_threads_in;
    if (n_threads <= 0) {
        n_threads = if (threadpool) |tp| tp.n_threads else c.GGML_DEFAULT_N_THREADS;
    }

    var work_size: usize = 0;

    var cplan: c.struct_ggml_cplan = std.mem.zeroes(c.struct_ggml_cplan);

    var max_tasks: c_int = 1;

    // Thread scheduling per op, and the work-buffer estimate, in one pass.
    for (0..@intCast(cgraph.n_nodes)) |i| {
        const node = cgraph.nodes[i].?;

        const n_tasks = getNTasks(node, n_threads);
        const tasks: i64 = n_tasks;

        max_tasks = @max(max_tasks, n_tasks);

        var cur: usize = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            cur = nodeWorkSize(node, tasks);
        }

        work_size = @max(work_size, cur);
    }

    // One cache line per thread of slack, so threads writing into their own
    // slice of the buffer do not share a line at the boundary.
    if (work_size > 0) {
        work_size += defs.cache_line_size * @as(usize, @intCast(n_threads));
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
fn nodeWorkSize(node: *const Tensor, n_tasks: i64) usize {
    const f32_size = types.ggml_type_size(c.GGML_TYPE_F32);

    switch (node.op) {
        c.GGML_OP_CPY, c.GGML_OP_DUP => {
            const src0 = impl.one(Tensor, node.src[0]);

            // A `CONT` has no second source, so every pair below is guarded.
            const src1_type: ?c.enum_ggml_type =
                if (node.src[1] == null) null else impl.one(Tensor, node.src[1]).type;

            const via_f32 = types.ggml_is_quantized(node.type) or
                // F16 <-> BF16 copies go through an intermediate F32.
                (src0.type == c.GGML_TYPE_F16 and src1_type == c.GGML_TYPE_BF16) or
                (src0.type == c.GGML_TYPE_BF16 and src1_type == c.GGML_TYPE_F16) or
                // So does conversion between F32 and I32.
                (src0.type == c.GGML_TYPE_F32 and src1_type == c.GGML_TYPE_I32) or
                (src0.type == c.GGML_TYPE_I32 and src1_type == c.GGML_TYPE_F32);

            if (via_f32) return f32_size * @as(usize, @intCast(node.ne[0] * n_tasks));
            return 0;
        },

        c.GGML_OP_ADD, c.GGML_OP_ADD_ID, c.GGML_OP_ADD1 => {
            const src0 = impl.one(Tensor, node.src[0]);
            if (types.ggml_is_quantized(src0.type)) {
                return f32_size * @as(usize, @intCast(src0.ne[0] * n_tasks));
            }
            return 0;
        },

        c.GGML_OP_ACC => {
            // Sized from `src[1]`, the block being written, not `src[0]`.
            if (types.ggml_is_quantized(impl.one(Tensor, node.src[0]).type)) {
                return f32_size * @as(usize, @intCast(impl.one(Tensor, node.src[1]).ne[0] * n_tasks));
            }
            return 0;
        },

        c.GGML_OP_COUNT_EQUAL => return types.ggml_type_size(node.type) * @as(usize, @intCast(n_tasks)),

        c.GGML_OP_MUL_MAT => {
            const src1 = impl.one(Tensor, node.src[1]);
            const vec_dot_type = traits.table[@intCast(impl.one(Tensor, node.src[0]).type)].vec_dot_type;
            if (src1.type != vec_dot_type) {
                return types.ggml_row_size(vec_dot_type, types.ggml_nelements(src1));
            }
            return 0;
        },

        c.GGML_OP_MUL_MAT_ID => {
            // Four sub-allocations, each rounded up by `incr_ptr_aligned` in
            // `mulmat.zig`. The trailing `sizeof(int64_t)` on the first three
            // and the trailing cache line on the fourth are that rounding's
            // worst case.
            const src0 = impl.one(Tensor, node.src[0]);
            const src1 = impl.one(Tensor, node.src[1]);
            const ids = impl.one(Tensor, node.src[2]);
            const vec_dot_type = traits.table[@intCast(src0.type)].vec_dot_type;
            const n_as: usize = @intCast(src0.ne[2]);

            var cur: usize = 0;

            // src1, staged into the kernel's type.
            if (src1.type != vec_dot_type) {
                cur += types.ggml_row_size(vec_dot_type, types.ggml_nelements(src1)) + @sizeOf(i64);
            }
            // matrix_row_counts
            cur += n_as * @sizeOf(i64) + @sizeOf(i64);
            // matrix_rows
            cur += @as(usize, @intCast(@as(i64, @intCast(n_as)) * ids.ne[0] * ids.ne[1])) *
                @sizeOf(mulmat.RowMapping) + @sizeOf(i64);
            // atomic_current_chunk
            cur += defs.cache_line_size * n_as + defs.cache_line_size;

            return cur;
        },

        c.GGML_OP_OUT_PROD => {
            const src0 = impl.one(Tensor, node.src[0]);
            if (types.ggml_is_quantized(src0.type) or src0.type == c.GGML_TYPE_F16) {
                return f32_size * @as(usize, @intCast(src0.ne[0] * n_tasks));
            }
            return 0;
        },

        c.GGML_OP_SET_ROWS => {
            const src0 = impl.one(Tensor, node.src[0]);
            if (src0.type == c.GGML_TYPE_F16 and node.type != c.GGML_TYPE_F16) {
                return f32_size * @as(usize, @intCast(src0.ne[0] * n_tasks));
            }
            return 0;
        },

        c.GGML_OP_SOFT_MAX, c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => {
            return f32_size * @as(usize, @intCast(node.ne[0] * n_tasks));
        },

        c.GGML_OP_CONV_TRANSPOSE_1D => {
            const src0 = impl.one(Tensor, node.src[0]);
            const src1 = impl.one(Tensor, node.src[1]);

            impl.assert(src0.ne[3] == 1, "node->src[0]->ne[3] == 1");
            impl.assert(src1.ne[2] == 1, "node->src[1]->ne[2] == 1");
            impl.assert(src1.ne[3] == 1, "node->src[1]->ne[3] == 1");

            const ne00 = src0.ne[0]; // K
            const ne01 = src0.ne[1]; // Cout
            const ne02 = src0.ne[2]; // Cin
            const ne10 = src1.ne[0]; // L
            const ne11 = src1.ne[1]; // Cin

            if ((src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_BF16) and
                src1.type == c.GGML_TYPE_F32)
            {
                return @sizeOf(c.ggml_fp16_t) * @as(usize, @intCast(ne00 * ne01 * ne02 + ne10 * ne11));
            } else if (src0.type == c.GGML_TYPE_F32 and src1.type == c.GGML_TYPE_F32) {
                return @sizeOf(f32) * @as(usize, @intCast(ne00 * ne01 * ne02 + ne10 * ne11));
            }
            impl.abort("fatal error");
        },

        c.GGML_OP_CONV_2D, c.GGML_OP_CONV_3D => return im2col_work_size,

        c.GGML_OP_CONV_TRANSPOSE_2D => {
            const src0 = impl.one(Tensor, node.src[0]);
            const src1 = impl.one(Tensor, node.src[1]);

            const ne00 = src0.ne[0]; // W
            const ne01 = src0.ne[1]; // H
            const ne02 = src0.ne[2]; // Channels out
            const ne03 = src0.ne[3]; // Channels in

            const ne10 = src1.ne[0]; // W
            const ne11 = src1.ne[1]; // H
            const ne12 = src1.ne[2]; // Channels in

            impl.assert(
                src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_F32,
                "node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32",
            );
            impl.assert(src1.type == c.GGML_TYPE_F32, "node->src[1]->type == GGML_TYPE_F32");

            return types.ggml_type_size(src0.type) *
                @as(usize, @intCast(ne00 * ne01 * ne02 * ne03 + ne10 * ne11 * ne12));
        },

        c.GGML_OP_TOP_K => {
            return @sizeOf(i32) * @as(usize, @intCast(impl.one(Tensor, node.src[0]).ne[0] * n_tasks));
        },

        c.GGML_OP_FLASH_ATTN_EXT => {
            const neq2 = impl.one(Tensor, node.src[0]).ne[2]; // query heads
            const dk = impl.one(Tensor, node.src[1]).ne[0]; // key width
            const dv = impl.one(Tensor, node.src[2]).ne[0]; // value width

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

        c.GGML_OP_FLASH_ATTN_BACK => {
            const src1 = impl.one(Tensor, node.src[1]);
            const d = impl.one(Tensor, node.src[0]).ne[0];
            const ne11 = up(src1.ne[1], soft_max_unroll);
            // Doubled for S and SM in the kernel.
            const mx_dn = @max(d, ne11) * 2;

            const t = src1.type;
            if (t == c.GGML_TYPE_F32 or t == c.GGML_TYPE_F16 or t == c.GGML_TYPE_BF16) {
                // The C writes this as two identical additions with a note
                // that the second overestimates by 2x. Kept as written.
                return 2 * @sizeOf(f32) * @as(usize, @intCast(mx_dn * n_tasks));
            }
            return 0;
        },

        c.GGML_OP_CROSS_ENTROPY_LOSS => {
            return types.ggml_type_size(node.type) *
                @as(usize, @intCast(n_tasks + impl.one(Tensor, node.src[0]).ne[0] * n_tasks));
        },

        c.GGML_OP_GATED_DELTA_NET => {
            const s_v = impl.one(Tensor, node.src[2]).ne[0];
            const k: i64 = impl.getOpParamsI32(node, 0);
            const per_thread = s_v + (if (k > 1) s_v * s_v else 0);
            return @as(usize, @intCast(per_thread * n_tasks)) * @sizeOf(f32);
        },

        c.GGML_OP_LIGHTNING_INDEXER => {
            // Scratch for dequantising the indexer keys.
            return @sizeOf(f32) * @as(usize, @intCast(impl.one(Tensor, node.src[1]).ne[0] * n_tasks));
        },

        c.GGML_OP_COUNT => impl.abort("fatal error"),

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
    var node: Tensor = std.mem.zeroes(Tensor);
    node.op = c.GGML_OP_MUL_MAT;
    node.ne = .{ 0, 1, 1, 1 };

    try std.testing.expectEqual(@as(c_int, 1), getNTasks(&node, 8));
}

test "n_tasks splits the parallel ops and pins the serial ones" {
    var node: Tensor = std.mem.zeroes(Tensor);
    node.ne = .{ 4, 4, 1, 1 };
    node.nb = .{ 4, 16, 64, 64 };
    node.type = c.GGML_TYPE_F32;

    node.op = c.GGML_OP_MUL_MAT;
    try std.testing.expectEqual(@as(c_int, 8), getNTasks(&node, 8));

    node.op = c.GGML_OP_SUM;
    try std.testing.expectEqual(@as(c_int, 1), getNTasks(&node, 8));

    // `CLAMP` is 1 with a TODO in the C, and reads like an oversight. It is
    // not one to fix here.
    node.op = c.GGML_OP_CLAMP;
    try std.testing.expectEqual(@as(c_int, 1), getNTasks(&node, 8));
}

test "soft_max asks for no more threads than there are rows" {
    var src: Tensor = std.mem.zeroes(Tensor);
    src.ne = .{ 16, 3, 1, 1 };

    var node: Tensor = std.mem.zeroes(Tensor);
    node.op = c.GGML_OP_SOFT_MAX;
    node.ne = .{ 16, 3, 1, 1 };
    node.src[0] = &src;

    try std.testing.expectEqual(@as(c_int, 3), getNTasks(&node, 8));
    try std.testing.expectEqual(@as(c_int, 2), getNTasks(&node, 2));
}

test "a custom op's task count comes out of op_params" {
    var node: Tensor = std.mem.zeroes(Tensor);
    node.op = c.GGML_OP_CUSTOM;
    node.ne = .{ 4, 1, 1, 1 };

    const p = impl.CustomOpParams{ .fun = null, .n_tasks = 3, .userdata = null };
    @memcpy(std.mem.asBytes(&node.op_params)[0..@sizeOf(impl.CustomOpParams)], std.mem.asBytes(&p));
    try std.testing.expectEqual(@as(c_int, 3), getNTasks(&node, 8));
    try std.testing.expectEqual(@as(c_int, 2), getNTasks(&node, 2));

    const maxed = impl.CustomOpParams{ .fun = null, .n_tasks = n_tasks_max, .userdata = null };
    @memcpy(std.mem.asBytes(&node.op_params)[0..@sizeOf(impl.CustomOpParams)], std.mem.asBytes(&maxed));
    try std.testing.expectEqual(@as(c_int, 8), getNTasks(&node, 8));
}

test "mul_mat_id reserves room for all four sub-allocations" {
    // The kernel carves these out with `incr_ptr_aligned`, so the count here
    // has to cover the alignment slack as well as the blocks themselves. An
    // underestimate is an overrun of a shared buffer.
    var src0: Tensor = std.mem.zeroes(Tensor);
    src0.type = c.GGML_TYPE_F32;
    src0.ne = .{ 8, 4, 3, 1 }; // 3 experts

    var src1: Tensor = std.mem.zeroes(Tensor);
    src1.type = c.GGML_TYPE_F32;
    src1.ne = .{ 8, 2, 1, 1 };
    src1.nb = .{ 4, 32, 64, 64 };

    var ids: Tensor = std.mem.zeroes(Tensor);
    ids.type = c.GGML_TYPE_I32;
    ids.ne = .{ 2, 2, 1, 1 };

    var node: Tensor = std.mem.zeroes(Tensor);
    node.op = c.GGML_OP_MUL_MAT_ID;
    node.ne = .{ 4, 2, 2, 1 };
    node.src[0] = &src0;
    node.src[1] = &src1;
    node.src[2] = &ids;

    const size = nodeWorkSize(&node, 4);

    // src1 is already F32, which is `vec_dot_type` for an F32 src0, so the
    // staging block is skipped and only the last three are counted.
    const counts = 3 * @sizeOf(i64) + @sizeOf(i64);
    const rows = 3 * 2 * 2 * @sizeOf(mulmat.RowMapping) + @sizeOf(i64);
    const chunks = defs.cache_line_size * 3 + defs.cache_line_size;
    try std.testing.expectEqual(counts + rows + chunks, size);
}

test "a node with no scratch needs none" {
    var src0: Tensor = std.mem.zeroes(Tensor);
    src0.type = c.GGML_TYPE_F32;
    src0.ne = .{ 4, 1, 1, 1 };

    var node: Tensor = std.mem.zeroes(Tensor);
    node.op = c.GGML_OP_ADD;
    node.ne = .{ 4, 1, 1, 1 };
    node.src[0] = &src0;

    try std.testing.expectEqual(@as(usize, 0), nodeWorkSize(&node, 4));
}
