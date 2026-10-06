//! The Metal op encoder: one graph node to one kernel dispatch.
//!
//! # Provenance
//!
//! Ports `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` at v0.3.0
//! (`c1d0e7a00`).
//!
//! # What this file does
//!
//! For each node of a graph it picks a pipeline (`library.zig`), fills the
//! kernel's argument block (`kargs.zig`), binds the buffers, and
//! dispatches. The encoder itself lives in `ggml-metal-device.m`, which
//! stays Objective-C — Decision 13 — and is reached through
//! `device_c.zig`.
//!
//! # The encoders are exported under the C's names
//!
//! Each `ggml_metal_op_*` is a `pub export fn`, so the linker's
//! missing-symbol errors are what proved the port complete: all **66**
//! of the translation unit's unmangled exports resolve. They were
//! deliberately *private* while bodies were still stubs, because
//! `scripts/port-coverage` would otherwise have read `66 / 66 (100%)`
//! for a file whose every dispatch aborted — which is exactly what
//! `repack.cpp` did before `scripts/cluster-check` existed.
//!
//! `encoders_implemented` is the comptime guard for that, in the same
//! shape as `repack.dispatch_implemented`.

const std = @import("std");

const impl = @import("../impl.zig");
const c = impl.c;

const mc = @import("device_c.zig");
const fc = @import("impl_c.zig");
const kargs = @import("kargs.zig");
const library = @import("library.zig");
const tuning = @import("tuning.zig");
const common = @import("common.zig");
const graph = @import("../graph.zig");

const Tensor = c.ggml_tensor;
const Pwp = mc.PipelineWithParams;

/// The allocator ported ggml uses — libc's, which is what a C entry point
/// can reach, so memory crossing the C ABI can be freed by either side.
const allocator = std.heap.c_allocator;

/// `op->src[i]`, **narrowed**.
///
/// `op.src[i]` is a `[*c]ggml_tensor`, and Zig 0.16 types `p.*.ne[0]` on a
/// C pointer as the whole `[4]i64` — the miscompile `CLAUDE.md` records.
/// `.?` does **not** narrow a `[*c]`; `impl.one` does.
inline fn src(op: *const Tensor, i: usize) *const Tensor {
    return impl.one(Tensor, op.src[i]);
}

/// `op->src[i]` when it may be absent.
inline fn srcOpt(op: *const Tensor, i: usize) ?*const Tensor {
    return if (op.src[i] == null) null else impl.one(Tensor, op.src[i]);
}

// Where the C reads a possibly-absent source through
// `GGML_TENSOR_LOCALS(int32_t, ne1, op->src[1], ne)`, the macro is
// **null-guarded** -- `GGML_TENSOR_LOCALS_1` (ggml.h:298 @c1d0e7a00)
// expands to `(pointer) ? (pointer)->array[0] : 0`. So an absent source
// contributes **zeros** to the `kargs`, not garbage and not a
// dereference, which is why `encodeSoftMax` writes
// `if (s1) |t| t.nb[1] else 0` rather than asserting it is present.

/// Ports `ggml_metal_get_buffer_id` (ggml-metal-ops.cpp:17 @c1d0e7a00).
///
/// Return: the Metal buffer and offset backing `t`, or a null handle when
/// `t` is absent. A view reads its source's buffer.
fn getBufferId(t: ?*const Tensor) mc.BufferId {
    const tensor = t orelse return .{ .metal = null, .offs = 0 };

    const buffer: *c.ggml_backend_buffer = if (tensor.view_src != null)
        impl.one(c.ggml_backend_buffer, impl.one(Tensor, tensor.view_src).buffer)
    else
        impl.one(c.ggml_backend_buffer, tensor.buffer);

    const ctx: *mc.Buffer = @ptrCast(@alignCast(buffer.context.?));

    return mc.ggml_metal_buffer_get_id(ctx, tensor);
}

/// The C's `ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), idx)`,
/// with the size taken from the type rather than restated.
///
/// The encoder copies the bytes, so the pointer does not outlive the call
/// and a stack `args` is correct — as it is in the C.
inline fn setBytes(enc: *mc.Encoder, args: anytype, idx: c_int) void {
    const T = @TypeOf(args.*);
    mc.ggml_metal_encoder_set_bytes(enc, @ptrCast(@constCast(args)), @sizeOf(T), idx);
}

/// `ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(t), idx)`.
inline fn setBuffer(enc: *mc.Encoder, t: ?*const Tensor, idx: c_int) void {
    mc.ggml_metal_encoder_set_buffer(enc, getBufferId(t), idx);
}

/// `ggml_metal_pipeline_max_theads_per_threadgroup`, which the encoders
/// clamp thread counts against. The misspelling is upstream's.
inline fn maxThreads(pipeline: Pwp) c_int {
    return mc.ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
}

/// Ports `struct ggml_metal_op` (ggml-metal-ops.cpp:29 @c1d0e7a00).
///
/// The C's one STL member is `std::vector<int> idxs`, the non-empty node
/// indices; here it is a heap slice sized once from `gf->n_nodes`, which
/// is the bound the C reserves to.
pub const Op = struct {
    dev: *mc.Device,
    lib: *mc.Library,
    enc: *mc.Encoder,
    mem_ranges: ?*common.MemRanges,

    use_fusion: bool,
    use_concurrency: bool,
    use_capture: bool,

    debug_graph: c_int,
    debug_fusion: c_int,

    gf: *impl.CGraph,

    idx_start: c_int,
    idx_end: c_int,

    /// non-empty node indices
    idxs: []c_int,
    n_idxs: usize,

    /// Ports `ggml_metal_op`'s constructor (ggml-metal-ops.cpp:30
    /// @c1d0e7a00).
    fn init(
        dev: *mc.Device,
        cmd_buf: mc.CmdBuf,
        gf: *impl.CGraph,
        idx_start: c_int,
        idx_end: c_int,
        use_fusion: bool,
        use_concurrency: bool,
        use_capture: bool,
        debug_graph: c_int,
        debug_fusion: c_int,
    ) *Op {
        const self = allocator.create(Op) catch impl.abort("metal: out of memory allocating the op context");

        self.* = .{
            .dev = dev,
            .lib = mc.ggml_metal_device_get_library(dev) orelse impl.abort("metal: device has no library"),
            .enc = mc.ggml_metal_encoder_init(cmd_buf, use_concurrency) orelse
                impl.abort("metal: could not create a command encoder"),
            .mem_ranges = common.ggml_mem_ranges_init(debug_graph),
            .use_fusion = use_fusion,
            .use_concurrency = use_concurrency,
            .use_capture = use_capture,
            .debug_graph = debug_graph,
            .debug_fusion = debug_fusion,
            .gf = gf,
            .idx_start = idx_start,
            .idx_end = idx_end,
            .idxs = allocator.alloc(c_int, @intCast(@max(gf.n_nodes, 0))) catch
                impl.abort("metal: out of memory allocating the node index list"),
            .n_idxs = 0,
        };

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        var i = idx_start;
        while (i < idx_end) : (i += 1) {
            const n = impl.one(Tensor, c.ggml_graph_node(@ptrCast(gf), i));
            if (!impl.opIsEmpty(n.op) and !c.ggml_is_empty(n)) {
                self.idxs[self.n_idxs] = i;
                self.n_idxs += 1;
            }
        }

        return self;
    }

    /// Ports `ggml_metal_op`'s destructor (ggml-metal-ops.cpp:66
    /// @c1d0e7a00).
    fn deinit(self: *Op) void {
        mc.ggml_metal_encoder_end_encoding(self.enc);
        mc.ggml_metal_encoder_free(self.enc);
        if (self.mem_ranges) |mr| common.ggml_mem_ranges_free(mr);
        allocator.free(self.idxs);
        allocator.destroy(self);
    }

    /// Ports `n_nodes` (ggml-metal-ops.cpp:72 @c1d0e7a00).
    fn nNodes(self: *const Op) c_int {
        return @intCast(self.n_idxs);
    }

    /// Ports `node` (ggml-metal-ops.cpp:76 @c1d0e7a00).
    fn node(self: *const Op, i: c_int) *Tensor {
        impl.assert(i >= 0 and i < self.nNodes(), "i >= 0 && i < (int) idxs.size()");
        return impl.one(Tensor, c.ggml_graph_node(@ptrCast(self.gf), self.idxs[@intCast(i)]));
    }

    /// Ports `can_fuse` (ggml-metal-ops.cpp:81 @c1d0e7a00).
    /// `i0` is renamed `j0`: Zig reserves every `iN` as an integer type
    /// name. Digit for digit, per the convention in `CLAUDE.md`.
    fn canFuse(self: *const Op, j0: c_int, ops: []const c.enum_ggml_op, n_ops: c_int) bool {
        impl.assert(self.use_fusion, "use_fusion");
        impl.assert(j0 >= 0 and j0 < self.nNodes(), "i0 >= 0 && i0 < n_nodes()");

        if (j0 + n_ops > self.nNodes()) {
            return false;
        }

        const n: usize = @intCast(n_ops);
        return graph.canFuseExt(self.gf, self.idxs[@intCast(j0)..], ops[0..n], n);
    }
};

/// Ports `ggml_metal_op_init` (ggml-metal-ops.cpp:114 @c1d0e7a00).
pub export fn ggml_metal_op_init(
    dev: *mc.Device,
    cmd_buf: mc.CmdBuf,
    gf: *impl.CGraph,
    idx_start: c_int,
    idx_end: c_int,
    use_fusion: bool,
    use_concurrency: bool,
    use_capture: bool,
    debug_graph: c_int,
    debug_fusion: c_int,
) callconv(.c) *Op {
    return Op.init(
        dev,
        cmd_buf,
        gf,
        idx_start,
        idx_end,
        use_fusion,
        use_concurrency,
        use_capture,
        debug_graph,
        debug_fusion,
    );
}

/// Ports `ggml_metal_op_free` (ggml-metal-ops.cpp:140 @c1d0e7a00).
pub export fn ggml_metal_op_free(ctx: *Op) callconv(.c) void {
    ctx.deinit();
}

/// Ports `ggml_metal_op_n_nodes` (ggml-metal-ops.cpp:144 @c1d0e7a00).
pub export fn ggml_metal_op_n_nodes(ctx: *Op) callconv(.c) c_int {
    return ctx.nNodes();
}

/// Ports `ggml_metal_op_concurrency_reset` (ggml-metal-ops.cpp:148
/// @c1d0e7a00).
fn concurrencyReset(ctx: *Op) bool {
    const mr = ctx.mem_ranges orelse return true;

    mc.ggml_metal_encoder_memory_barrier(ctx.enc);

    common.ggml_mem_ranges_reset(mr);

    return true;
}

/// Ports `ggml_metal_op_concurrency_check` (ggml-metal-ops.cpp:160
/// @c1d0e7a00).
fn concurrencyCheck(ctx: *Op, node: *const Tensor) bool {
    const mr = ctx.mem_ranges orelse return false;

    return common.ggml_mem_ranges_check(mr, node);
}

/// Ports `ggml_metal_op_encode` (ggml-metal-ops.cpp:522 @c1d0e7a00).
///
/// Return: how many nodes were consumed — more than one when the encoder
/// fused a run of them.
pub export fn ggml_metal_op_encode(ctx: *Op, idx: c_int) callconv(.c) c_int {
    if (ctx.use_capture) {
        mc.ggml_metal_encoder_debug_group_push(ctx.enc, c.ggml_op_desc(ctx.node(idx)));
    }

    const res = encodeImpl(ctx, idx);
    if (idx + res > ctx.nNodes()) {
        impl.abort("fusion error: nodes spanning multiple encoders have been fused. " ++
            "this indicates a bug in the fusion logic " ++
            "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx.use_capture) {
        mc.ggml_metal_encoder_debug_group_pop(ctx.enc);
    }

    return res;
}

/// Ports `ggml_metal_op_concurrency_add` (ggml-metal-ops.cpp:168
/// @c1d0e7a00).
fn concurrencyAdd(ctx: *Op, node: *const Tensor) bool {
    const mr = ctx.mem_ranges orelse return true;

    return common.ggml_mem_ranges_add(mr, node);
}

/// Ports the dispatch switch of `ggml_metal_op_encode_impl`
/// (ggml-metal-ops.cpp:176 @c1d0e7a00), which begins at its line 265.
///
/// Each arm is written as `=> encodeX(ctx, idx)` rather than the C's
/// `n_fuse = …; break;`, so a missing arm is a compile error here where
/// the C would fall to its `default` and abort at run time.
fn dispatch(ctx: *Op, node: *const Tensor, idx: c_int) c_int {
    return switch (node.op) {
        c.GGML_OP_CONCAT => ggml_metal_op_concat(ctx, idx),
        c.GGML_OP_ADD, c.GGML_OP_SUB, c.GGML_OP_MUL, c.GGML_OP_DIV => ggml_metal_op_bin(ctx, idx),
        c.GGML_OP_ADD_ID => ggml_metal_op_add_id(ctx, idx),
        c.GGML_OP_REPEAT => ggml_metal_op_repeat(ctx, idx),
        c.GGML_OP_ACC => ggml_metal_op_acc(ctx, idx),
        c.GGML_OP_SCALE,
        c.GGML_OP_FILL,
        c.GGML_OP_CLAMP,
        c.GGML_OP_LEAKY_RELU,
        c.GGML_OP_SQR,
        c.GGML_OP_SQRT,
        c.GGML_OP_SIN,
        c.GGML_OP_COS,
        c.GGML_OP_LOG,
        c.GGML_OP_UNARY,
        => ggml_metal_op_unary(ctx, idx),
        c.GGML_OP_SILU_BACK => ggml_metal_op_silu_back(ctx, idx),
        c.GGML_OP_GLU => ggml_metal_op_glu(ctx, idx),
        c.GGML_OP_SUM => ggml_metal_op_sum(ctx, idx),
        c.GGML_OP_SUM_ROWS, c.GGML_OP_MEAN => ggml_metal_op_sum_rows(ctx, idx),
        c.GGML_OP_CUMSUM => ggml_metal_op_cumsum(ctx, idx),
        c.GGML_OP_LIGHTNING_INDEXER => ggml_metal_op_lightning_indexer(ctx, idx),
        c.GGML_OP_DSV4_HC_COMB,
        c.GGML_OP_DSV4_HC_PRE,
        c.GGML_OP_DSV4_HC_POST,
        => ggml_metal_op_dsv4_hc(ctx, idx),
        c.GGML_OP_SOFT_MAX => ggml_metal_op_soft_max(ctx, idx),
        c.GGML_OP_SSM_CONV => ggml_metal_op_ssm_conv(ctx, idx),
        c.GGML_OP_SSM_SCAN => ggml_metal_op_ssm_scan(ctx, idx),
        c.GGML_OP_RWKV_WKV6, c.GGML_OP_RWKV_WKV7 => ggml_metal_op_rwkv(ctx, idx),
        c.GGML_OP_GATED_DELTA_NET => ggml_metal_op_gated_delta_net(ctx, idx),
        c.GGML_OP_SOLVE_TRI => ggml_metal_op_solve_tri(ctx, idx),
        c.GGML_OP_MUL_MAT => ggml_metal_op_mul_mat(ctx, idx),
        c.GGML_OP_MUL_MAT_ID => ggml_metal_op_mul_mat_id(ctx, idx),
        c.GGML_OP_GET_ROWS => ggml_metal_op_get_rows(ctx, idx),
        c.GGML_OP_SET_ROWS => ggml_metal_op_set_rows(ctx, idx),
        c.GGML_OP_DIAG => ggml_metal_op_diag(ctx, idx),
        c.GGML_OP_L2_NORM => ggml_metal_op_l2_norm(ctx, idx),
        c.GGML_OP_GROUP_NORM => ggml_metal_op_group_norm(ctx, idx),
        c.GGML_OP_NORM, c.GGML_OP_RMS_NORM => ggml_metal_op_norm(ctx, idx),
        c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => ggml_metal_op_rope(ctx, idx),
        c.GGML_OP_IM2COL => ggml_metal_op_im2col(ctx, idx),
        c.GGML_OP_CONV_2D => ggml_metal_op_conv_2d(ctx, idx),
        c.GGML_OP_CONV_2D_DW => ggml_metal_op_conv_2d_dw(ctx, idx),
        c.GGML_OP_CONV_TRANSPOSE_1D => ggml_metal_op_conv_transpose_1d(ctx, idx),
        c.GGML_OP_CONV_TRANSPOSE_2D => ggml_metal_op_conv_transpose_2d(ctx, idx),
        c.GGML_OP_COL2IM_1D => ggml_metal_op_col2im_1d(ctx, idx),
        c.GGML_OP_CONV_3D => ggml_metal_op_conv_3d(ctx, idx),
        c.GGML_OP_UPSCALE => ggml_metal_op_upscale(ctx, idx),
        c.GGML_OP_PAD => ggml_metal_op_pad(ctx, idx),
        c.GGML_OP_PAD_REFLECT_1D => ggml_metal_op_pad_reflect_1d(ctx, idx),
        c.GGML_OP_ROLL => ggml_metal_op_roll(ctx, idx),
        c.GGML_OP_ARANGE => ggml_metal_op_arange(ctx, idx),
        c.GGML_OP_TIMESTEP_EMBEDDING => ggml_metal_op_timestep_embedding(ctx, idx),
        c.GGML_OP_ARGSORT => ggml_metal_op_argsort(ctx, idx),
        c.GGML_OP_TOP_K => ggml_metal_op_top_k(ctx, idx),
        c.GGML_OP_TRI => ggml_metal_op_tri(ctx, idx),
        c.GGML_OP_FLASH_ATTN_EXT => ggml_metal_op_flash_attn_ext(ctx, idx),
        c.GGML_OP_SET => ggml_metal_op_set(ctx, idx),
        c.GGML_OP_DUP, c.GGML_OP_CPY, c.GGML_OP_CONT => ggml_metal_op_cpy(ctx, idx),
        c.GGML_OP_POOL_1D => ggml_metal_op_pool_1d(ctx, idx),
        c.GGML_OP_POOL_2D => ggml_metal_op_pool_2d(ctx, idx),
        c.GGML_OP_ARGMAX => ggml_metal_op_argmax(ctx, idx),
        c.GGML_OP_OPT_STEP_ADAMW => ggml_metal_op_opt_step_adamw(ctx, idx),
        c.GGML_OP_OPT_STEP_SGD => ggml_metal_op_opt_step_sgd(ctx, idx),
        c.GGML_OP_COUNT_EQUAL => ggml_metal_op_count_equal(ctx, idx),
        else => {
            impl.logError("%s: error: node %3d, op = %8s not implemented\n", .{
                "ggml_metal_op_encode_impl", idx, c.ggml_op_name(node.op),
            });
            impl.abort("metal: op not implemented");
        },
    };
}

/// Ports `ggml_metal_op_encode_impl` (ggml-metal-ops.cpp:176 @c1d0e7a00).
fn encodeImpl(ctx: *Op, idx: c_int) c_int {
    const node = ctx.node(idx);

    if (c.ggml_is_empty(node)) {
        return 1;
    }

    switch (node.op) {
        c.GGML_OP_NONE,
        c.GGML_OP_RESHAPE,
        c.GGML_OP_VIEW,
        c.GGML_OP_TRANSPOSE,
        c.GGML_OP_PERMUTE,
        => {
            // noop -> next node
            if (ctx.debug_graph > 0) {
                impl.logDebug("%s: node[%5d] - %-12s %s\n", .{
                    "ggml_metal_op_encode_impl", idx, c.ggml_op_name(node.op), "(noop)",
                });
            }
            return 1;
        },
        else => {},
    }

    if (!mc.ggml_metal_device_supports_op(ctx.dev, node)) {
        impl.logError("%s: error: unsupported op '%s'\n", .{
            "ggml_metal_op_encode_impl", c.ggml_op_desc(node),
        });
        impl.abort("unsupported op");
    }

    if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 1;
    }

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        const is_concurrent = concurrencyCheck(ctx, node);

        if (!is_concurrent) {
            _ = concurrencyReset(ctx);
        }

        if (ctx.debug_graph > 0) {
            impl.logDebug("%s: node[%5d] - %-12s %-12s %s\n", .{
                "ggml_metal_op_encode_impl",
                idx,
                c.ggml_op_name(node.op),
                c.ggml_get_name(node),
                @as([*:0]const u8, if (is_concurrent) "(concurrent)" else ""),
            });
        }
        if (ctx.debug_graph > 1) {
            debugLogShapes(node);
        }
    }

    const n_fuse = dispatch(ctx, node, idx);

    if (ctx.debug_graph > 0) {
        if (n_fuse > 1) {
            impl.logDebug("%s:               fuse %d ops\n", .{ "ggml_metal_op_encode_impl", n_fuse });
        }
    }

    // update the mem ranges in the encoding context
    var i: c_int = 0;
    while (i < n_fuse) : (i += 1) {
        if (!concurrencyAdd(ctx, ctx.node(idx + i))) {
            _ = concurrencyReset(ctx);
        }
    }

    return n_fuse;
}

/// The `debug_graph > 1` block of `ggml_metal_op_encode_impl`
/// (ggml-metal-ops.cpp:176 @c1d0e7a00), at its line 231 — lifted out so
/// the control flow above reads.
///
/// The C expands ten `GGML_TENSOR_LOCALS` here and prints four sources
/// and the node; the loop is equivalent and spells the same format.
fn debugLogShapes(node: *const Tensor) void {
    for (0..4) |j| {
        const s = srcOpt(node, j) orelse continue;
        impl.logDebug(
            "%s: src%d - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n",
            .{
                "ggml_metal_op_encode_impl", @as(c_int, @intCast(j)), c.ggml_type_name(s.type),
                s.ne[0],                     s.ne[1],                 s.ne[2],
                s.ne[3],                     s.nb[0],                 s.nb[1],
                s.nb[2],                     s.nb[3],                 @as(c_int, if (c.ggml_is_contiguous(s)) 1 else 0),
                &s.name,
            },
        );
    }

    impl.logDebug(
        "%s: node  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n",
        .{
            "ggml_metal_op_encode_impl", c.ggml_type_name(node.type), node.ne[0],
            node.ne[1],                  node.ne[2],                  node.ne[3],
            node.nb[0],                  node.nb[1],                  node.nb[2],
            node.nb[3],                  &node.name,
        },
    );
}

/// Whether the per-op encoder bodies are written.
///
/// **This must be `true` before this file joins `module.zig`.** A symbol
/// count cannot see the difference between a kernel dispatch and an
/// `abort`, and `repack.cpp` proved that the hard way: `port-coverage`
/// read `36 / 36 (100%)` with its dispatch stubbed. `cluster-check`
/// asserts the equivalent flag there; it should assert this one too when
/// the swap happens.
pub const encoders_implemented = true;

/// The 53 per-op encoders are the bulk of what remains.
///
/// They are stubs that abort, rather than left out, so the dispatch
/// switch above compiles and can be diffed against the C's — which it
/// has been, label for label. This is safe only because `module.zig`
/// does not import this file: nothing can reach them. Wired in as they
/// stand, the first node of any graph aborts loudly, which is a
/// deliberately loud failure and not a silent fallback.
const todo = struct {
    fn notPorted(comptime what: []const u8) noreturn {
        impl.abort("metal: ggml_metal_op_" ++ what ++ " is not ported yet");
    }
};

/// Ports `ggml_metal_op_concat` (ggml-metal-ops.cpp:540 @c1d0e7a00).
pub export fn ggml_metal_op_concat(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const dim = impl.getOpParamsI32(op, 0);

    var args: kargs.concat = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne10 = @intCast(s1.ne[0]),
        .ne11 = @intCast(s1.ne[1]),
        .ne12 = @intCast(s1.ne[2]),
        .ne13 = @intCast(s1.ne[3]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .nb13 = s1.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .dim = dim,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_concat(lib, op.type);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    const ne0: c_int = @intCast(op.ne[0]);
    const ne1: c_int = @intCast(op.ne[1]);

    const nth = @min(@as(c_int, 256), ne0);

    // when rows are small, we can batch them together in a single threadgroup
    var nrptg: c_int = 1;
    if (nth < 256) {
        nrptg = @min(@divTrunc(256 + nth - 1, nth), ne1);
        if (nrptg * nth > 256) {
            nrptg = @divTrunc(@as(c_int, 256), nth);
        }
    }

    const nw0 = @divTrunc(ne1 + nrptg - 1, nrptg);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, nw0, @intCast(op.ne[2]), @intCast(op.ne[3]), nth, nrptg, 1);

    return 1;
}

/// Ports `ggml_metal_op_bin` (ggml-metal-ops.cpp:3599 @c1d0e7a00).
///
/// Fuses a run of up to **eight** `ADD`s into one kernel by passing each
/// operand's offset in `o1[]`; all of them must live in the same Metal
/// buffer, which is why `bid_src1.offs` is zeroed and the offsets become
/// relative to it.
pub export fn ggml_metal_op_bin(ctx: *Op, idx: c_int) callconv(.c) c_int {
    if (ctx.use_fusion and canFuseSnake(ctx, idx)) {
        return ggml_metal_op_snake_fused(ctx, idx);
    }

    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const use_fusion = ctx.use_fusion;

    const debug_fusion = ctx.debug_fusion;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");
    impl.assert(c.ggml_is_contiguous_rows(s1), "ggml_is_contiguous_rows(op->src[1])");

    const bid_src0 = getBufferId(s0);
    var bid_src1 = getBufferId(s1);
    var bid_dst = getBufferId(op);

    const ne00: i32 = @intCast(s0.ne[0]);
    const ne10: i32 = @intCast(s1.ne[0]);
    const ne0: i32 = @intCast(op.ne[0]);

    var args: kargs.bin = .{
        .ne00 = ne00,
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne10 = ne10,
        .ne11 = @intCast(s1.ne[1]),
        .ne12 = @intCast(s1.ne[2]),
        .ne13 = @intCast(s1.ne[3]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .nb13 = s1.nb[3],
        .ne0 = ne0,
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .offs = 0,
        // the C's `{ bid_src1.offs }` -- element 0 set, the rest zeroed
        .o1 = .{ bid_src1.offs, 0, 0, 0, 0, 0, 0, 0 },
    };

    var fops: [8]c.enum_ggml_op = undefined;

    var n_fuse: c_int = 1;

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion) {
        for (&fops) |*f| {
            f.* = c.GGML_OP_ADD;
        }

        // note: in metal, we sometimes encode the graph in parallel so we have to avoid fusing ops
        //       across splits. idx_end indicates the last node in the current split
        n_fuse = 0;
        while (n_fuse <= 6) : (n_fuse += 1) {
            if (!ctx.canFuse(idx + n_fuse, fops[@intCast(n_fuse)..], 2)) {
                break;
            }

            const f0 = ctx.node(idx + n_fuse);
            const f1 = ctx.node(idx + n_fuse + 1);

            if (f0 != impl.one(Tensor, f1.src[0])) {
                break;
            }

            // b[0] === b[1] === ...
            if (!impl.areSameLayout(src(f0, 1), src(f1, 1))) {
                break;
            }

            // only fuse ops if src1 is in the same Metal buffer
            const bid_fuse = getBufferId(src(f1, 1));
            if (bid_fuse.metal != bid_src1.metal) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            args.o1[@intCast(n_fuse + 1)] = bid_fuse.offs;
        }

        n_fuse += 1;

        if (debug_fusion > 1 and n_fuse > 1) {
            impl.logDebug("%s: fuse: ADD x %d\n", .{ "ggml_metal_op_bin", n_fuse });
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    const pipeline = library.ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    if (n_fuse > 1) {
        bid_dst = getBufferId(ctx.node(idx + n_fuse - 1));

        var i: c_int = 1;
        while (i < n_fuse) : (i += 1) {
            if (!concurrencyCheck(ctx, ctx.node(idx + i))) {
                _ = concurrencyReset(ctx);

                break;
            }
        }
    }

    if (pipeline.c4) {
        args.ne00 = @divTrunc(ne00, 4);
        args.ne10 = @divTrunc(ne10, 4);
        args.ne0 = @divTrunc(ne0, 4);
    }

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
    mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

    if (pipeline.cnt) {
        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            args.ne0,
            @intCast(c.ggml_nrows(op)),
            1,
            1,
            1,
            1,
        );
    } else {
        const nth_max = @min(@as(c_int, 256), maxThreads(pipeline));

        var nth: c_int = 1;

        while (2 * nth < args.ne0 and nth < nth_max) {
            nth *= 2;
        }

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @intCast(s0.ne[1]),
            @intCast(s0.ne[2]),
            @intCast(s0.ne[3]),
            nth,
            1,
            1,
        );
    }

    return n_fuse;
}

/// Ports `ggml_metal_op_add_id` (ggml-metal-ops.cpp:2749 @c1d0e7a00).
///
/// Short, despite the 250 lines the file gives it -- the rest of that
/// span is the `flash_attn_ext` helper group that follows it.
pub export fn ggml_metal_op_add_id(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = src(op, 2);

    impl.assert(s0.type == c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    impl.assert(s1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(s2.type == c.GGML_TYPE_I32, "op->src[2]->type == GGML_TYPE_I32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    var args: kargs.add_id = .{
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb11 = s1.nb[1],
        .nb21 = s2.nb[1],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_base(lib, c.GGML_OP_ADD_ID);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, s2, 3);
    setBuffer(enc, op, 4);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(s0.ne[0])));

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(s0.ne[1]),
        @intCast(s0.ne[2]),
        1,
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_repeat` (ggml-metal-ops.cpp:609 @c1d0e7a00).
pub export fn ggml_metal_op_repeat(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const pipeline = library.ggml_metal_library_get_pipeline_repeat(lib, op.type);

    var args: kargs.repeat = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(op.ne[0])));

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(op.ne[1]),
        @intCast(op.ne[2]),
        @intCast(op.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_acc` (ggml-metal-ops.cpp:653 @c1d0e7a00).
///
/// Two dispatches when not in-place: a `cpy` of src0 into dst, then an
/// `ADD` of src1 into a *window* of it — which is what the `pnb*` strides
/// and `offs` describe. Note the `bin` args deliberately feed `src[1]`'s
/// shape into the `ne0*` fields and the window strides into `nb0*`.
pub export fn ggml_metal_op_acc(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    impl.assert(s0.type == c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    impl.assert(s1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");
    impl.assert(c.ggml_is_contiguous_rows(s1), "ggml_is_contiguous_rows(op->src[1])");

    const pnb1: usize = @intCast(impl.getOpParamsI32(op, 0));
    const pnb2: usize = @intCast(impl.getOpParamsI32(op, 1));
    const pnb3: usize = @intCast(impl.getOpParamsI32(op, 2));
    const offs: usize = @intCast(impl.getOpParamsI32(op, 3));

    const inplace = impl.getOpParamsI32(op, 4) != 0;

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        const pipeline = library.ggml_metal_library_get_pipeline_cpy(lib, s0.type, op.type);

        var args: kargs.cpy = .{
            .nk0 = s0.ne[0],
            .ne00 = s0.ne[0],
            .ne01 = s0.ne[1],
            .ne02 = s0.ne[2],
            .ne03 = s0.ne[3],
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne0 = op.ne[0],
            .ne1 = op.ne[1],
            .ne2 = op.ne[2],
            .ne3 = op.ne[3],
            .nb0 = op.nb[0],
            .nb1 = op.nb[1],
            .nb2 = op.nb[2],
            .nb3 = op.nb[3],
        };

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, op, 2);

        const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(s0.ne[0])));

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @intCast(s0.ne[1]),
            @intCast(s0.ne[2]),
            @intCast(s0.ne[3]),
            nth,
            1,
            1,
        );

        _ = concurrencyReset(ctx);
    }

    var args: kargs.bin = .{
        .ne00 = @intCast(s1.ne[0]),
        .ne01 = @intCast(s1.ne[1]),
        .ne02 = @intCast(s1.ne[2]),
        .ne03 = @intCast(s1.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = pnb1,
        .nb02 = pnb2,
        .nb03 = pnb3,
        .ne10 = @intCast(s1.ne[0]),
        .ne11 = @intCast(s1.ne[1]),
        .ne12 = @intCast(s1.ne[2]),
        .ne13 = @intCast(s1.ne[3]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .nb13 = s1.nb[3],
        .ne0 = @intCast(s1.ne[0]),
        .ne1 = @intCast(s1.ne[1]),
        .ne2 = @intCast(s1.ne[2]),
        .ne3 = @intCast(s1.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = pnb1,
        .nb2 = pnb2,
        .nb3 = pnb3,
        .offs = offs,
        .o1 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    const pipeline = library.ggml_metal_library_get_pipeline_bin_one(lib, c.GGML_OP_ADD);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    const nth_max = @min(@as(c_int, 256), maxThreads(pipeline));

    var nth: c_int = 1;

    while (2 * nth < args.ne0 and nth < nth_max) {
        nth *= 2;
    }

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(s1.ne[1]),
        @intCast(s1.ne[2]),
        @intCast(s1.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_unary` (ggml-metal-ops.cpp:770 @c1d0e7a00).
///
/// One kernel for nine ops plus twenty-two `GGML_UNARY_OP_*`; the op's
/// own parameters are packed into the six trailing float fields, which
/// start at zero and are filled only by the arms that use them.
pub export fn ggml_metal_op_unary(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const bid_src0 = getBufferId(s0);
    const bid_dst = getBufferId(op);

    const ne00: i32 = @intCast(s0.ne[0]);
    const ne0: i32 = @intCast(op.ne[0]);

    var args: kargs.unary = .{
        .ne00 = ne00,
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = ne0,
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .slope = 0.0,
        .scale = 0.0,
        .bias = 0.0,
        .val = 0.0,
        .min = 0.0,
        .max = 0.0,
    };

    if (op.op == c.GGML_OP_LEAKY_RELU) {
        args.slope = impl.getOpParamsF32(op, 0);
    }

    if (op.op == c.GGML_OP_SCALE) {
        args.scale = impl.getOpParamsF32(op, 0);
        args.bias = impl.getOpParamsF32(op, 1);
    }

    if (op.op == c.GGML_OP_FILL) {
        args.val = impl.getOpParamsF32(op, 0);
    }

    if (op.op == c.GGML_OP_CLAMP) {
        args.min = impl.getOpParamsF32(op, 0);
        args.max = impl.getOpParamsF32(op, 1);
    }

    if (op.op == c.GGML_OP_UNARY and c.ggml_get_unary_op(op) == c.GGML_UNARY_OP_XIELU) {
        args.slope = impl.getOpParamsF32(op, 1); // alpha_n
        args.scale = impl.getOpParamsF32(op, 2); // alpha_p
        args.bias = impl.getOpParamsF32(op, 3); // beta
        args.val = impl.getOpParamsF32(op, 4); // eps
    }

    const pipeline = library.ggml_metal_library_get_pipeline_unary(lib, op);

    if (pipeline.c4) {
        args.ne00 = @divTrunc(ne00, 4);
        args.ne0 = @divTrunc(ne0, 4);
    }

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    if (pipeline.cnt) {
        const n: c_int = @intCast(if (pipeline.c4)
            @divTrunc(c.ggml_nelements(op), 4)
        else
            c.ggml_nelements(op));

        mc.ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } else {
        const nth_max = @min(@as(c_int, 256), maxThreads(pipeline));
        const nth = @min(args.ne00, nth_max);
        const nk0 = @divTrunc(args.ne00 + nth - 1, nth);

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            nk0 * @as(c_int, @intCast(s0.ne[1])),
            @intCast(s0.ne[2]),
            @intCast(s0.ne[3]),
            nth,
            1,
            1,
        );
    }

    return 1;
}

/// Ports `ggml_metal_op_silu_back` (ggml-metal-ops.cpp:3759 @c1d0e7a00).
pub export fn ggml_metal_op_silu_back(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const pipeline = library.ggml_metal_library_get_pipeline_silu_back(lib, op);

    const ne: i64 = c.ggml_nelements(op);

    var args: kargs.silu_back = .{
        .ne = ne,
    };

    var arg_idx: c_int = 0;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, arg_idx);
    arg_idx += 1;
    setBuffer(enc, src(op, 0), arg_idx);
    arg_idx += 1;
    setBuffer(enc, src(op, 1), arg_idx);
    arg_idx += 1;
    setBuffer(enc, op, arg_idx);
    arg_idx += 1;

    const nth: i64 = @min(@as(i64, maxThreads(pipeline)), ne);
    const n = @divTrunc(ne + nth - 1, nth);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n), 1, 1, @intCast(nth), 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_glu` (ggml-metal-ops.cpp:863 @c1d0e7a00).
///
/// `src[1]` is optional: without it the kernel reads both halves out of
/// `src[0]`, which is what `i00`/`i10` select — and note the C passes
/// **0** for both when `src[1]` is present, having just computed them.
pub export fn ggml_metal_op_glu(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = srcOpt(op, 1);

    if (src1) |s1| {
        impl.assert(c.ggml_are_same_shape(src0, s1), "ggml_are_same_shape(op->src[0], op->src[1])");
    }

    const pipeline = library.ggml_metal_library_get_pipeline_glu(lib, op);

    const swp = impl.getOpParamsI32(op, 1);
    const alpha = impl.getOpParamsF32(op, 2);
    const limit = impl.getOpParamsF32(op, 3);

    const ne00: i32 = @intCast(src0.ne[0]);
    const ne0: i32 = @intCast(op.ne[0]);

    // `i00`/`i10` are reserved integer type names in Zig, renamed digit
    // for digit per the convention in `CLAUDE.md`. The `kargs` fields
    // keep the C's spelling.
    const j00: i32 = if (swp != 0) ne0 else 0;
    const j10: i32 = if (swp != 0) 0 else ne0;

    var args: kargs.glu = .{
        .ne00 = ne00,
        .nb01 = src0.nb[1],
        .ne10 = if (src1) |s1| @intCast(s1.ne[0]) else ne00,
        .nb11 = if (src1) |s1| s1.nb[1] else src0.nb[1],
        .ne0 = ne0,
        .nb1 = op.nb[1],
        .i00 = if (src1 != null) 0 else j00,
        .i10 = if (src1 != null) 0 else j10,
        .alpha = alpha,
        .limit = limit,
    };

    const nrows: i64 = c.ggml_nrows(src0);

    const nth: i32 = @min(maxThreads(pipeline), @divTrunc(ne00, 2));

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    if (src1) |s1| {
        setBuffer(enc, s1, 2);
    } else {
        setBuffer(enc, src0, 2);
    }
    setBuffer(enc, op, 3);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(nrows), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_sum` (ggml-metal-ops.cpp:921 @c1d0e7a00).
pub export fn ggml_metal_op_sum(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const n: u64 = @intCast(c.ggml_nelements(s0));

    var args: kargs.sum = .{
        .np = n,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_sum(lib, op);

    var nth: c_int = 32; // SIMD width

    while (nth < @as(c_int, @intCast(n)) and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    nth = @min(nth, maxThreads(pipeline));
    nth = @min(nth, @as(c_int, @intCast(n)));

    const nsg = @divTrunc(nth + 31, 32);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, @as(usize, @intCast(nsg)) * @sizeOf(f32), 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_sum_rows` (ggml-metal-ops.cpp:958 @c1d0e7a00).
pub export fn ggml_metal_op_sum_rows(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const bid_src0 = getBufferId(s0);
    const bid_dst = getBufferId(op);

    const ne00: i64 = s0.ne[0];
    const ne0: i64 = op.ne[0];

    var args: kargs.sum_rows = .{
        .ne00 = ne00,
        .ne01 = s0.ne[1],
        .ne02 = s0.ne[2],
        .ne03 = s0.ne[3],
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = ne0,
        .ne1 = op.ne[1],
        .ne2 = op.ne[2],
        .ne3 = op.ne[3],
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_sum_rows(lib, op);

    if (pipeline.c4) {
        args.ne00 = @divTrunc(ne00, 4);
        args.ne0 = @divTrunc(ne0, 4);
    }

    var nth: c_int = 32; // SIMD width

    while (nth < @as(c_int, @intCast(args.ne00)) and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    nth = @min(nth, maxThreads(pipeline));
    nth = @min(nth, @as(c_int, @intCast(args.ne00)));

    const smem = pipeline.smem;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(s0.ne[1]),
        @intCast(s0.ne[2]),
        @intCast(s0.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_cumsum` (ggml-metal-ops.cpp:1023 @c1d0e7a00).
///
/// Up to three dispatches: a per-block scan, then — only when the row
/// exceeds one threadgroup — a scan of the block totals and an add of
/// those back into the result. The scratch sits immediately after `dst`,
/// which is why `backend.zig`'s `get_alloc_size` doubles the allocation
/// for `CUMSUM`.
pub export fn ggml_metal_op_cumsum(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);

    const pipeline_blk = library.ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    var nth: c_int = 1;
    while (nth < ne00 and 2 * nth <= maxThreads(pipeline_blk)) {
        nth *= 2;
    }

    impl.assert(ne00 <= nth * nth, "ne00 <= nth*nth");

    const net0: i64 = @divTrunc(@as(i64, ne00) + nth - 1, nth);
    const net1: i64 = ne01;
    const net2: i64 = ne02;
    const net3: i64 = ne03;

    const nbt0: u64 = @sizeOf(f32);
    const nbt1: u64 = @as(u64, @intCast(net0)) * nbt0;
    const nbt2: u64 = @as(u64, @intCast(net1)) * nbt1;
    const nbt3: u64 = @as(u64, @intCast(net2)) * nbt2;

    const smem = impl.pad(32 * @sizeOf(f32), 16);

    const bid_src0 = getBufferId(s0);
    const bid_dst = getBufferId(op);

    var bid_tmp = bid_dst;
    bid_tmp.offs += @intCast(c.ggml_nbytes(op));

    {
        var args: kargs.cumsum_blk = .{
            .ne00 = ne00,
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .net0 = net0,
            .net1 = net1,
            .net2 = net2,
            .net3 = net3,
            .nbt0 = nbt0,
            .nbt1 = nbt1,
            .nbt2 = nbt2,
            .nbt3 = nbt3,
            .outb = ne00 > nth,
        };

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        setBytes(enc, &args, 0);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 2);
        mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @intCast(net0 * ne01),
            ne02,
            ne03,
            nth,
            1,
            1,
        );
    }

    if (ne00 > nth) {
        _ = concurrencyReset(ctx);

        {
            var args: kargs.cumsum_blk = .{
                .ne00 = @intCast(net0),
                .ne01 = @intCast(net1),
                .ne02 = @intCast(net2),
                .ne03 = @intCast(net3),
                .nb00 = nbt0,
                .nb01 = nbt1,
                .nb02 = nbt2,
                .nb03 = nbt3,
                .net0 = net0,
                .net1 = net1,
                .net2 = net2,
                .net3 = net3,
                .nbt0 = nbt0,
                .nbt1 = nbt1,
                .nbt2 = nbt2,
                .nbt3 = nbt3,
                .outb = false,
            };

            mc.ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            setBytes(enc, &args, 0);
            mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
            mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 2);
            mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 3);

            mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @intCast(net1),
                @intCast(net2),
                @intCast(net3),
                nth,
                1,
                1,
            );
        }

        _ = concurrencyReset(ctx);

        {
            const pipeline_add = library.ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            var args: kargs.cumsum_add = .{
                .ne00 = ne00,
                .ne01 = ne01,
                .ne02 = ne02,
                .ne03 = ne03,
                .nb00 = s0.nb[0],
                .nb01 = s0.nb[1],
                .nb02 = s0.nb[2],
                .nb03 = s0.nb[3],
                .net0 = net0,
                .net1 = net1,
                .net2 = net2,
                .net3 = net3,
                .nbt0 = nbt0,
                .nbt1 = nbt1,
                .nbt2 = nbt2,
                .nbt3 = nbt3,
            };

            mc.ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            setBytes(enc, &args, 0);
            mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
            mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @intCast(net0 * ne01),
                ne02,
                ne03,
                nth,
                1,
                1,
            );
        }
    }

    return 1;
}

/// Ports `ggml_metal_op_lightning_indexer` (ggml-metal-ops.cpp:1315
/// @c1d0e7a00).
///
/// Note the C binds the buffers **before** setting the pipeline and the
/// bytes, where every other encoder sets the pipeline first. Order is
/// preserved.
pub export fn ggml_metal_op_lightning_indexer(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const enc = ctx.enc;

    impl.assert(op.op == c.GGML_OP_LIGHTNING_INDEXER, "op->op == GGML_OP_LIGHTNING_INDEXER");

    const q = src(op, 0);
    const k = src(op, 1);
    const w = src(op, 2);
    const m = src(op, 3);

    impl.assert(q.type == c.GGML_TYPE_F32, "q->type == GGML_TYPE_F32");
    impl.assert(
        k.type == c.GGML_TYPE_F32 or
            k.type == c.GGML_TYPE_F16 or
            k.type == c.GGML_TYPE_BF16 or
            k.type == c.GGML_TYPE_Q4_0 or
            k.type == c.GGML_TYPE_Q4_1 or
            k.type == c.GGML_TYPE_Q5_0 or
            k.type == c.GGML_TYPE_Q5_1 or
            k.type == c.GGML_TYPE_Q8_0,
        "k->type is one of f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0",
    );
    impl.assert(w.type == c.GGML_TYPE_F32, "w->type == GGML_TYPE_F32");
    impl.assert(m.type == c.GGML_TYPE_F16, "m->type == GGML_TYPE_F16");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    impl.assert(q.ne[0] == fc.OP_LIGHTNING_INDEXER_DK, "q->ne[0] == OP_LIGHTNING_INDEXER_DK");
    impl.assert(q.ne[1] == fc.OP_LIGHTNING_INDEXER_NH, "q->ne[1] == OP_LIGHTNING_INDEXER_NH");

    var args: kargs.lightning_indexer = .{
        .n_kv = @intCast(k.ne[2]),
        .n_batch = @intCast(q.ne[2]),
        .mask_ne3 = @intCast(m.ne[3]),
        .nb1 = op.nb[1],
        .nb3 = op.nb[3],
        .nbq1 = q.nb[1],
        .nbq2 = q.nb[2],
        .nbq3 = q.nb[3],
        .nbk2 = k.nb[2],
        .nbk3 = k.nb[3],
        .nbw1 = w.nb[1],
        .nbw3 = w.nb[3],
        .nbm1 = m.nb[1],
        .nbm3 = m.nb[3],
    };

    setBuffer(enc, q, 1);
    setBuffer(enc, k, 2);
    setBuffer(enc, w, 3);
    setBuffer(enc, m, 4);
    setBuffer(enc, op, 5);

    const nsg = fc.OP_LIGHTNING_INDEXER_NSG;
    const nkptg = fc.OP_LIGHTNING_INDEXER_NKPSG * nsg;
    const nbptg = fc.OP_LIGHTNING_INDEXER_NBPTG;

    const pipeline = library.ggml_metal_library_get_pipeline_lightning_indexer(ctx.lib, op);
    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(@divTrunc(k.ne[2] + nkptg - 1, nkptg)),
        @intCast(@divTrunc(q.ne[2] + nbptg - 1, nbptg)),
        @intCast(q.ne[3]),
        32,
        nsg,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_dsv4_hc` (ggml-metal-ops.cpp:1381 @c1d0e7a00).
///
/// Three ops behind one entry point, each with its own `kargs` struct and
/// its own source naming. Note the pipeline is set **before** the switch,
/// so all three arms share it.
pub export fn ggml_metal_op_dsv4_hc(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const enc = ctx.enc;
    const pipeline = library.ggml_metal_library_get_pipeline_dsv4_hc(ctx.lib, op.op);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);

    switch (op.op) {
        c.GGML_OP_DSV4_HC_COMB => {
            const mixes = src(op, 0);
            const scale = src(op, 1);
            const base = src(op, 2);

            impl.assert(mixes.type == c.GGML_TYPE_F32, "mixes->type == GGML_TYPE_F32");
            impl.assert(scale.type == c.GGML_TYPE_F32, "scale->type == GGML_TYPE_F32");
            impl.assert(base.type == c.GGML_TYPE_F32, "base->type == GGML_TYPE_F32");
            impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
            impl.assert(mixes.ne[0] == 24, "mixes->ne[0] == 24");
            impl.assert(op.ne[0] == 4 and op.ne[1] == 4, "op->ne[0] == 4 && op->ne[1] == 4");

            var args: kargs.dsv4_hc_comb = .{
                .n_tokens = @intCast(mixes.ne[1]),
                .n_iter = impl.getOpParamsI32(op, 1),
                .nb_m0 = mixes.nb[0],
                .nb_m1 = mixes.nb[1],
                .nb_s0 = scale.nb[0],
                .nb_b0 = base.nb[0],
                .nb_d0 = op.nb[0],
                .nb_d1 = op.nb[1],
                .nb_d2 = op.nb[2],
                .eps = impl.getOpParamsF32(op, 0),
            };

            setBytes(enc, &args, 0);
            setBuffer(enc, mixes, 1);
            setBuffer(enc, scale, 2);
            setBuffer(enc, base, 3);
            setBuffer(enc, op, 4);

            // One SIMDgroup owns one 4x4 Sinkhorn matrix. Packing up to four
            // independent tokens per threadgroup keeps both decode and prompt
            // dispatches compact without any threadgroup-memory synchronization.
            const nsg = @min(@as(c_int, 4), args.n_tokens);
            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(args.n_tokens + nsg - 1, nsg),
                1,
                1,
                32,
                nsg,
                1,
            );
        },
        c.GGML_OP_DSV4_HC_PRE => {
            const x = src(op, 0);
            const weights = src(op, 1);

            impl.assert(x.type == c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
            impl.assert(weights.type == c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
            impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
            impl.assert(x.ne[1] == 4, "x->ne[1] == 4");

            var args: kargs.dsv4_hc_pre = .{
                .n_embd = @intCast(x.ne[0]),
                .n_tokens = @intCast(x.ne[2]),
                .nb_x0 = x.nb[0],
                .nb_x1 = x.nb[1],
                .nb_x2 = x.nb[2],
                .nb_w0 = weights.nb[0],
                .nb_w1 = weights.nb[1],
                .nb_d0 = op.nb[0],
                .nb_d1 = op.nb[1],
            };

            setBytes(enc, &args, 0);
            setBuffer(enc, x, 1);
            setBuffer(enc, weights, 2);
            setBuffer(enc, op, 3);

            const n_tiles = @divTrunc(args.n_embd + 31, 32);
            const nsg = @min(@as(c_int, 4), n_tiles);
            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(n_tiles + nsg - 1, nsg),
                args.n_tokens,
                1,
                32,
                nsg,
                1,
            );
        },
        c.GGML_OP_DSV4_HC_POST => {
            const x = src(op, 0);
            const residual = src(op, 1);
            const post = src(op, 2);
            const comb = src(op, 3);

            impl.assert(x.type == c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
            impl.assert(residual.type == c.GGML_TYPE_F32, "residual->type == GGML_TYPE_F32");
            impl.assert(post.type == c.GGML_TYPE_F32, "post->type == GGML_TYPE_F32");
            impl.assert(comb.type == c.GGML_TYPE_F32, "comb->type == GGML_TYPE_F32");
            impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
            impl.assert(residual.ne[1] == 4, "residual->ne[1] == 4");

            var args: kargs.dsv4_hc_post = .{
                .n_embd = @intCast(x.ne[0]),
                .n_tokens = @intCast(x.ne[1]),
                .nb_x0 = x.nb[0],
                .nb_x1 = x.nb[1],
                .nb_r0 = residual.nb[0],
                .nb_r1 = residual.nb[1],
                .nb_r2 = residual.nb[2],
                .nb_p0 = post.nb[0],
                .nb_p1 = post.nb[1],
                .nb_c0 = comb.nb[0],
                .nb_c1 = comb.nb[1],
                .nb_c2 = comb.nb[2],
                .nb_d0 = op.nb[0],
                .nb_d1 = op.nb[1],
                .nb_d2 = op.nb[2],
            };

            setBytes(enc, &args, 0);
            setBuffer(enc, x, 1);
            setBuffer(enc, residual, 2);
            setBuffer(enc, post, 3);
            setBuffer(enc, comb, 4);
            setBuffer(enc, op, 5);

            const n_tiles = @divTrunc(args.n_embd + 31, 32);
            const nsg = @min(@as(c_int, 4), n_tiles);
            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(n_tiles + nsg - 1, nsg),
                args.n_tokens,
                1,
                32,
                nsg,
                1,
            );
        },
        else => impl.abort("metal: no dsv4_hc encoder for this op"),
    }

    return 1;
}

/// Ports `ggml_metal_op_soft_max` (ggml-metal-ops.cpp:1512 @c1d0e7a00).
///
/// `src[1]` (mask) and `src[2]` (sinks) are both optional; when absent
/// the C binds `src[0]` in their slot rather than a null buffer.
pub export fn ggml_metal_op_soft_max(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = srcOpt(op, 1);
    const s2 = srcOpt(op, 2);

    // The C `memcpy`s both out of `op_params` at indices 0 and 1.
    const scale = impl.getOpParamsF32(op, 0);
    const max_bias = impl.getOpParamsF32(op, 1);

    const n_head: u32 = @intCast(s0.ne[2]);
    const n_head_log2: i32 = @as(i32, 1) << @intCast(@as(u32, @intFromFloat(@floor(std.math.log2(@as(f32, @floatFromInt(n_head)))))));

    const m0 = std.math.pow(f32, 2.0, -(max_bias) / @as(f32, @floatFromInt(n_head_log2)));
    const m1 = std.math.pow(f32, 2.0, -(max_bias / 2.0) / @as(f32, @floatFromInt(n_head_log2)));

    // softmax

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);

    var args: kargs.soft_max = .{
        .ne00 = ne00,
        .ne01 = ne01,
        .ne02 = ne02,
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne11 = if (s1) |t| @intCast(t.ne[1]) else 0,
        .ne12 = if (s1) |t| @intCast(t.ne[2]) else 0,
        .ne13 = if (s1) |t| @intCast(t.ne[3]) else 0,
        .nb11 = if (s1) |t| t.nb[1] else 0,
        .nb12 = if (s1) |t| t.nb[2] else 0,
        .nb13 = if (s1) |t| t.nb[3] else 0,
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .scale = scale,
        .max_bias = max_bias,
        .m0 = m0,
        .m1 = m1,
        .n_head_log2 = n_head_log2,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_soft_max(lib, op);

    var nth: c_int = 32; // SIMD width

    if (@rem(ne00, 4) == 0) {
        while (nth < @divTrunc(ne00, 4) and nth * ne01 * ne02 * ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 and nth * ne01 * ne02 * ne03 < 256) {
            nth *= 2;
        }
    }

    const smem = pipeline.smem;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    if (s1) |t| {
        setBuffer(enc, t, 2);
    } else {
        setBuffer(enc, s0, 2);
    }
    if (s2) |t| {
        setBuffer(enc, t, 3);
    } else {
        setBuffer(enc, s0, 3);
    }
    setBuffer(enc, op, 4);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_ssm_conv` (ggml-metal-ops.cpp:1602 @c1d0e7a00).
pub export fn ggml_metal_op_ssm_conv(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    var args: kargs.ssm_conv = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .ne10 = @intCast(s1.ne[0]),
        .ne11 = @intCast(s1.ne[1]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
    };

    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne1: c_int = @intCast(op.ne[1]);

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        const BATCH_SIZE: c_int = if (ne1 > 128)
            256
        else if (ne1 > 64)
            128
        else if (ne1 > 32)
            64
        else if (ne1 > 16)
            32
        else if (ne1 > 8)
            16
        else if (ne1 > 4)
            8
        else
            2;

        const pipeline = library.ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE);

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const n_token_batches = @divTrunc(ne1 + BATCH_SIZE - 1, BATCH_SIZE);
        mc.ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        const pipeline = library.ggml_metal_library_get_pipeline_ssm_conv(lib, op);

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        mc.ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);
    }

    return 1;
}

/// Ports `ggml_metal_op_ssm_scan` (ggml-metal-ops.cpp:1675 @c1d0e7a00).
///
/// Seven sources, and the `ns*` fields are stride *ratios* — `nb12/nb10`
/// and friends — element counts rather than byte counts.
pub export fn ggml_metal_op_ssm_scan(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = src(op, 2);
    const src3 = src(op, 3);
    const src4 = src(op, 4);
    const src5 = src(op, 5);
    const src6 = src(op, 6);

    impl.assert(op.src[3] != null, "src3");
    impl.assert(op.src[4] != null, "src4");
    impl.assert(op.src[5] != null, "src5");
    impl.assert(op.src[6] != null, "src6");
    _ = src6;

    const d_state: i64 = s0.ne[0];
    const d_inner: i64 = s0.ne[1];
    const n_head: i64 = s0.ne[2];
    const n_group: i64 = src4.ne[1];
    const n_seq_tokens: i64 = s1.ne[2];
    const n_seqs: i64 = s1.ne[3];
    const K: i64 = impl.getOpParamsI32(op, 0);

    impl.assert(K >= 1, "K >= 1");
    impl.assert(
        c.ggml_nelements(s1) + K * d_state * d_inner * n_head * n_seqs == c.ggml_nelements(op),
        "ggml_nelements(op->src[1]) + K*d_state*d_inner*n_head*n_seqs == ggml_nelements(op)",
    );

    var args: kargs.ssm_scan = .{
        .d_state = @intCast(d_state),
        .d_inner = @intCast(d_inner),
        .n_head = @intCast(n_head),
        .n_group = @intCast(n_group),
        .n_seq_tokens = @intCast(n_seq_tokens),
        .n_seqs = @intCast(n_seqs),
        .K = @intCast(K),
        .s_off = @intCast(@as(usize, @intCast(c.ggml_nelements(s1))) * @sizeOf(f32)),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .ns12 = @intCast(s1.nb[2] / s1.nb[0]),
        .nb13 = s1.nb[3],
        .nb20 = s2.nb[0],
        .nb21 = s2.nb[1],
        .ns21 = @intCast(s2.nb[1] / s2.nb[0]),
        .nb22 = s2.nb[2],
        .ne30 = @intCast(src3.ne[0]),
        .nb31 = src3.nb[1],
        .nb41 = src4.nb[1],
        .nb42 = src4.nb[2],
        .ns42 = @intCast(src4.nb[2] / src4.nb[0]),
        .nb43 = src4.nb[3],
        .nb51 = src5.nb[1],
        .nb52 = src5.nb[2],
        .ns52 = @intCast(src5.nb[2] / src5.nb[0]),
        .nb53 = src5.nb[3],
        .nb0 = op.nb[0],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_ssm_scan(lib, op);

    impl.assert(
        d_state <= maxThreads(pipeline),
        "d_state <= max_theads_per_threadgroup",
    );

    const smem = pipeline.smem;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    for (0..7) |j| {
        setBuffer(enc, src(op, j), @intCast(j + 1));
    }
    setBuffer(enc, op, 8);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(d_inner),
        @intCast(n_head),
        @intCast(n_seqs),
        @intCast(d_state),
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_rwkv` (ggml-metal-ops.cpp:1778 @c1d0e7a00).
///
/// The one encoder with **no `kargs` struct**: it passes `B`, `T`, `C`
/// and `H` as four separate `set_bytes` of an `int64_t` each.
pub export fn ggml_metal_op_rwkv(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const B: i64 = if (op.op == c.GGML_OP_RWKV_WKV6) src(op, 5).ne[1] else src(op, 6).ne[1];
    const T: i64 = s0.ne[2];
    const C: i64 = op.ne[0];
    const H: i64 = s0.ne[1];

    const pipeline = library.ggml_metal_library_get_pipeline_rwkv(lib, op);

    var ida: c_int = 0;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    for (0..6) |j| {
        setBuffer(enc, src(op, j), ida);
        ida += 1;
    }
    if (op.op == c.GGML_OP_RWKV_WKV7) {
        setBuffer(enc, src(op, 6), ida);
        ida += 1;
    }
    setBuffer(enc, op, ida);
    ida += 1;
    setBytes(enc, &B, ida);
    ida += 1;
    setBytes(enc, &T, ida);
    ida += 1;
    setBytes(enc, &C, ida);
    ida += 1;
    setBytes(enc, &H, ida);
    ida += 1;

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(B * H),
        1,
        1,
        @intCast(@divTrunc(C, H)),
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_gated_delta_net` (ggml-metal-ops.cpp:1819
/// @c1d0e7a00).
pub export fn ggml_metal_op_gated_delta_net(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = src(op, 2);

    const pipeline = library.ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    var ida: c_int = 0;

    var args: kargs.gated_delta_net = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne10 = @intCast(s1.ne[0]),
        .ne11 = @intCast(s1.ne[1]),
        .ne12 = @intCast(s1.ne[2]),
        .ne13 = @intCast(s1.ne[3]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .nb13 = s1.nb[3],
        .ne20 = @intCast(s2.ne[0]),
        .ne21 = @intCast(s2.ne[1]),
        .ne22 = @intCast(s2.ne[2]),
        .ne23 = @intCast(s2.ne[3]),
        .nb20 = s2.nb[0],
        .nb21 = s2.nb[1],
        .nb22 = s2.nb[2],
        .nb23 = s2.nb[3],
        .ns02 = @intCast(s0.nb[2] / @sizeOf(f32)),
        .ns12 = @intCast(s1.nb[2] / @sizeOf(f32)),
        .ns22 = @intCast(s2.nb[2] / @sizeOf(f32)),
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, ida);
    ida += 1;
    // q, k, v, gate, beta, state, dst -- the C's own labels
    for (0..6) |j| {
        setBuffer(enc, src(op, j), ida);
        ida += 1;
    }
    setBuffer(enc, op, ida);
    ida += 1;

    const nsg = pipeline.nsg;

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(@divTrunc(s2.ne[0], nsg)),
        @intCast(s2.ne[1]),
        @intCast(s2.ne[3]),
        32,
        nsg,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_solve_tri` (ggml-metal-ops.cpp:1894 @c1d0e7a00).
pub export fn ggml_metal_op_solve_tri(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    var args: kargs.solve_tri = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne10 = @intCast(s1.ne[0]),
        .ne11 = @intCast(s1.ne[1]),
        .ne12 = @intCast(s1.ne[2]),
        .ne13 = @intCast(s1.ne[3]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .nb13 = s1.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_solve_tri(lib, op);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    const nsg = pipeline.nsg;

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    const ne10: c_int = @intCast(s1.ne[0]);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @divTrunc(ne10 + nsg - 1, nsg),
        @intCast(s0.ne[2]),
        @intCast(s0.ne[3]),
        32,
        nsg,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_mul_mat` (ggml-metal-ops.cpp:2300 @c1d0e7a00).
///
/// Three kernels behind one op, chosen in order: the small-batch
/// `mul_mv_ext` for `ne11` in [2, 8] on a 128-aligned row, then the
/// simdgroup-matrix `mul_mm` once the batch clears `ne11_mm_min`, else
/// the row-at-a-time `mul_mv`. A Hadamard hint short-circuits to `fwht`
/// before any of them.
pub export fn ggml_metal_op_mul_mat(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const hint = impl.getOpParamsI32(op, 1);

    if (hint == c.GGML_HINT_SRC0_IS_HADAMARD) {
        if (s1.type == c.GGML_TYPE_F32 and
            op.type == c.GGML_TYPE_F32 and
            c.ggml_is_contiguous(s1) and
            c.ggml_is_contiguous(op) and
            c.ggml_are_same_shape(s1, op) and
            fwhtSupportedSize(s1.ne[0]))
        {
            return ggml_metal_op_fwht(ctx, idx);
        }
    }
    const props_dev = mc.ggml_metal_device_get_props(ctx.dev);

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);
    const ne10: c_int = @intCast(s1.ne[0]);
    const ne11: c_int = @intCast(s1.ne[1]);
    const ne12: c_int = @intCast(s1.ne[2]);
    const ne13: c_int = @intCast(s1.ne[3]);

    impl.assert(ne00 == ne10, "ne00 == ne10");

    impl.assert(@rem(ne12, ne02) == 0, "ne12 % ne02 == 0");
    impl.assert(@rem(ne13, ne03) == 0, "ne13 % ne03 == 0");

    const r2: i16 = @intCast(@divTrunc(ne12, ne02));
    const r3: i16 = @intCast(@divTrunc(ne13, ne03));

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const ne11_mm_min: c_int = 8;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    //
    // the C writes this as one nested condition; it is split into its
    // two type groups here, which are the only things that differ
    // between them besides the `ne11` window.
    const t0_bs2 = s0.type == c.GGML_TYPE_F32 or // TODO: helper function
        s0.type == c.GGML_TYPE_F16 or
        s0.type == c.GGML_TYPE_BF16 or
        s0.type == c.GGML_TYPE_Q1_0 or
        s0.type == c.GGML_TYPE_Q2_0 or
        s0.type == c.GGML_TYPE_Q4_0 or
        s0.type == c.GGML_TYPE_Q4_1 or
        s0.type == c.GGML_TYPE_Q5_0 or
        s0.type == c.GGML_TYPE_Q5_1 or
        s0.type == c.GGML_TYPE_Q8_0 or
        s0.type == c.GGML_TYPE_MXFP4 or
        s0.type == c.GGML_TYPE_IQ4_NL;

    const t0_bs4 = s0.type == c.GGML_TYPE_Q4_K or
        s0.type == c.GGML_TYPE_Q5_K or
        s0.type == c.GGML_TYPE_Q6_K or
        s0.type == c.GGML_TYPE_Q2_K or
        s0.type == c.GGML_TYPE_Q3_K;

    const small_batch = s1.type == c.GGML_TYPE_F32 and (@rem(ne00, 128) == 0) and
        ((t0_bs2 and (ne11 >= 2 and ne11 <= 8)) or
            (t0_bs4 and (ne11 >= 4 and ne11 <= 8)));

    if (small_batch) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const nsg: c_int = 2; // num simdgroups per threadgroup

        // num threads along row per simdgroup
        var nxpsg: i16 = 0;
        if (@rem(ne00, 256) == 0 and ne11 < 3) {
            nxpsg = 16;
        } else if (@rem(ne00, 128) == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const nypsg: i16 = @divTrunc(32, nxpsg); // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const r0ptg: i16 = nypsg * @as(i16, @intCast(nsg)); // num src0 rows per threadgroup
        var r1ptg: i16 = 4; // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        switch (ne11) {
            2 => r1ptg = 2,
            3, 6 => r1ptg = 3,
            4, 7, 8 => r1ptg = 4,
            5 => r1ptg = 5,
            else => impl.abort("unsupported ne11"),
        }

        const pipeline = library.ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg);

        var args: kargs.mul_mv_ext = .{
            .ne00 = ne00,
            .ne01 = ne01,
            .ne02 = ne02,
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne10 = ne10,
            .ne11 = ne11,
            .ne12 = ne12,
            .nb10 = s1.nb[0],
            .nb11 = s1.nb[1],
            .nb12 = s1.nb[2],
            .nb13 = s1.nb[3],
            .ne0 = @intCast(op.ne[0]),
            .ne1 = @intCast(op.ne[1]),
            .r2 = r2,
            .r3 = r3,
        };

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @divTrunc(ne01 + r0ptg - 1, r0ptg),
            @divTrunc(ne11 + r1ptg - 1, r1ptg),
            ne12 * ne13,
            32,
            nsg,
            1,
        );
    } else if (!c.ggml_is_transposed(s0) and
        !c.ggml_is_transposed(s1) and
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        props_dev.has_simdgroup_mm and ne00 >= 64 and ne11 > ne11_mm_min)
    {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        // (the C leaves an alignment assert commented out here)

        const pipeline = library.ggml_metal_library_get_pipeline_mul_mm(lib, op);

        var args: kargs.mul_mm = .{
            .ne00 = ne00,
            .ne02 = ne02,
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne12 = ne12,
            .nb10 = s1.nb[0],
            .nb11 = s1.nb[1],
            .nb12 = s1.nb[2],
            .nb13 = s1.nb[3],
            .ne0 = @intCast(op.ne[0]),
            .ne1 = @intCast(op.ne[1]),
            .r2 = r2,
            .r3 = r3,
        };

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        const smem = pipeline.smem;

        mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const nr0 = pipeline.nr0;
        const nr1 = pipeline.nr1;
        const nsg = pipeline.nsg;

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @divTrunc(ne11 + nr1 - 1, nr1),
            @divTrunc(ne01 + nr0 - 1, nr0),
            ne12 * ne13,
            32,
            nsg,
            1,
        );
    } else {
        const pipeline = library.ggml_metal_library_get_pipeline_mul_mv(lib, op);

        const nr0 = pipeline.nr0;
        const nr1 = pipeline.nr1;
        const nsg = pipeline.nsg;

        const smem = pipeline.smem;

        var args: kargs.mul_mv = .{
            .ne00 = ne00,
            .ne01 = ne01,
            .ne02 = ne02,
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne10 = ne10,
            .ne11 = ne11,
            .ne12 = ne12,
            .nb10 = s1.nb[0],
            .nb11 = s1.nb[1],
            .nb12 = s1.nb[2],
            .nb13 = s1.nb[3],
            .ne0 = @intCast(op.ne[0]),
            .ne1 = @intCast(op.ne[1]),
            .nr0 = nr0,
            .r2 = r2,
            .r3 = r3,
        };

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (s0.type == c.GGML_TYPE_F32 or
            s0.type == c.GGML_TYPE_F16 or
            s0.type == c.GGML_TYPE_BF16 or
            s0.type == c.GGML_TYPE_Q8_0)
        {
            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(ne01 + nr0 - 1, nr0),
                @divTrunc(ne11 + nr1 - 1, nr1),
                ne12 * ne13,
                32,
                nsg,
                1,
            );
        } else {
            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(ne01 + nr0 * nsg - 1, nr0 * nsg),
                @divTrunc(ne11 + nr1 - 1, nr1),
                ne12 * ne13,
                32,
                nsg,
                1,
            );
        }
    }

    return 1;
}

/// Ports `ggml_metal_op_mul_mat_id` (ggml-metal-ops.cpp:2561 @c1d0e7a00).
///
/// Above a batch of `ne21_mm_id_min` it builds an expert->token id map
/// into scratch after `dst`, barriers, then runs the simdgroup-matrix
/// kernel over it; below that it falls to `mul_mv_id`, one row at a time.
pub export fn ggml_metal_op_mul_mat_id(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const props_dev = mc.ggml_metal_device_get_props(ctx.dev);

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = src(op, 2);

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);
    const ne10: c_int = @intCast(s1.ne[0]);
    const ne11: c_int = @intCast(s1.ne[1]);
    const ne12: c_int = @intCast(s1.ne[2]);
    const ne13: c_int = @intCast(s1.ne[3]);
    const ne20: c_int = @intCast(s2.ne[0]);
    const ne21: c_int = @intCast(s2.ne[1]);

    // src2 = ids
    impl.assert(s2.type == c.GGML_TYPE_I32, "op->src[2]->type == GGML_TYPE_I32");

    impl.assert(!c.ggml_is_transposed(s0), "!ggml_is_transposed(op->src[0])");
    impl.assert(!c.ggml_is_transposed(s1), "!ggml_is_transposed(op->src[1])");

    impl.assert(ne03 == 1, "ne03 == 1");
    impl.assert(ne13 == 1, "ne13 == 1");

    const bid_src0 = getBufferId(s0);
    const bid_src1 = getBufferId(s1);
    const bid_src2 = getBufferId(s2);
    const bid_dst = getBufferId(op);

    const r2: u32 = 1;
    const r3: u32 = 1;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    // ne20 = n_used_experts
    // ne21 = n_rows (batch size)
    const ne21_mm_id_min: c_int = 32;

    if (props_dev.has_simdgroup_mm and ne00 >= 64 and (ne21 >= ne21_mm_id_min)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        // (the C leaves an alignment assert commented out here)

        // extra buffers for intermediate id mapping
        var bid_tpe = bid_dst;
        bid_tpe.offs += @intCast(c.ggml_nbytes(op));

        var bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            // the C initialises this one positionally
            var args: kargs.mul_mm_id_map0 = .{
                .ne02 = ne02,
                .ne10 = ne10,
                .ne11 = ne11, // n_expert_used (bcast)
                .nb11 = s1.nb[1],
                .nb12 = s1.nb[2],
                .ne21 = ne21, // n_tokens
                .ne20 = ne20, // n_expert_used
                .nb21 = s2.nb[1],
            };

            const pipeline = library.ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const smem = pipeline.smem;

            impl.assert(ne02 <= maxThreads(pipeline), "ne02 <= max_theads_per_threadgroup");

            impl.assert(smem <= props_dev.max_theadgroup_memory_size, "smem <= max_theadgroup_memory_size");

            mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
            setBytes(enc, &args, 0);
            mc.ggml_metal_encoder_set_buffer(enc, bid_src2, 1);
            mc.ggml_metal_encoder_set_buffer(enc, bid_tpe, 2);
            mc.ggml_metal_encoder_set_buffer(enc, bid_ids, 3);

            mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            mc.ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        _ = concurrencyReset(ctx);

        {
            const pipeline = library.ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            var args: kargs.mul_mm_id = .{
                .ne00 = ne00,
                .ne02 = ne02,
                .nb01 = s0.nb[1],
                .nb02 = s0.nb[2],
                .nb03 = s0.nb[3],
                .ne11 = ne11, // n_expert_used (bcast)
                .nb10 = s1.nb[0],
                .nb11 = s1.nb[1],
                .nb12 = s1.nb[2],
                .nb13 = s1.nb[3],
                .ne20 = ne20, // n_expert_used
                .ne21 = ne21, // n_tokens
                .ne0 = @intCast(op.ne[0]),
                .ne1 = @intCast(op.ne[1]),
                .r2 = r2,
                .r3 = r3,
            };

            mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
            setBytes(enc, &args, 0);
            mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
            mc.ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
            mc.ggml_metal_encoder_set_buffer(enc, bid_tpe, 3);
            mc.ggml_metal_encoder_set_buffer(enc, bid_ids, 4);
            mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 5);

            const smem = pipeline.smem;

            mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(ne21 + 31, 32),
                @divTrunc(ne01 + 63, 64),
                ne02,
                128,
                1,
                1,
            );
        }
    } else {
        const pipeline = library.ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const nr0 = pipeline.nr0;
        const nr1 = pipeline.nr1;
        const nsg = pipeline.nsg;

        const smem = pipeline.smem;

        var args: kargs.mul_mv_id = .{
            .nei0 = ne20,
            .nei1 = ne21,
            .nbi1 = s2.nb[1],
            .ne00 = ne00,
            .ne01 = ne01,
            .ne02 = ne02,
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .ne10 = ne10,
            .ne11 = ne11,
            .ne12 = ne12,
            .ne13 = ne13,
            .nb10 = s1.nb[0],
            .nb11 = s1.nb[1],
            .nb12 = s1.nb[2],
            .ne0 = @intCast(op.ne[0]),
            .ne1 = @intCast(op.ne[1]),
            .nb1 = op.nb[1],
            .nr0 = nr0,
        };

        if (c.ggml_is_quantized(s0.type)) {
            impl.assert(ne00 >= nsg * nr0, "ne00 >= nsg*nr0");
        }

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 3);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const _ne1: i64 = 1;
        const ne123: i64 = @as(i64, ne20) * @as(i64, ne21);

        mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (s0.type == c.GGML_TYPE_F32 or
            s0.type == c.GGML_TYPE_F16 or
            s0.type == c.GGML_TYPE_BF16 or
            s0.type == c.GGML_TYPE_Q8_0)
        {
            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(ne01 + nr0 - 1, nr0),
                @intCast(@divTrunc(_ne1 + nr1 - 1, nr1)),
                @intCast(ne123),
                32,
                nsg,
                1,
            );
        } else {
            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(ne01 + nr0 * nsg - 1, nr0 * nsg),
                @intCast(@divTrunc(_ne1 + nr1 - 1, nr1)),
                @intCast(ne123),
                32,
                nsg,
                1,
            );
        }
    }

    return 1;
}

/// Ports `ggml_metal_op_get_rows` (ggml-metal-ops.cpp:1166 @c1d0e7a00).
pub export fn ggml_metal_op_get_rows(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const pipeline = library.ggml_metal_library_get_pipeline_get_rows(lib, s0.type);

    const ne00: i32 = @intCast(s0.ne[0]);

    var args: kargs.get_rows = .{
        .ne00t = if (c.ggml_is_quantized(s0.type)) @divTrunc(ne00, 16) else ne00,
        .ne00 = ne00,
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne10 = @intCast(s1.ne[0]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    const nth = @min(args.ne00t, maxThreads(pipeline));

    const nw0 = @divTrunc(args.ne00t + nth - 1, nth);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        nw0 * @as(c_int, @intCast(s1.ne[0])),
        @intCast(s1.ne[1]),
        @intCast(s1.ne[2]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_set_rows` (ggml-metal-ops.cpp:1211 @c1d0e7a00).
pub export fn ggml_metal_op_set_rows(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const pipeline = library.ggml_metal_library_get_pipeline_set_rows(lib, op);

    const nk0: i32 = @intCast(@divTrunc(op.ne[0], c.ggml_blck_size(op.type)));

    var nth: c_int = 32; // SIMD width

    while (nth < nk0 and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    var nrptg: c_int = 1;
    if (nth > nk0) {
        nrptg = @divTrunc(nth + nk0 - 1, nk0);
        nth = nk0;

        if (nrptg * nth > maxThreads(pipeline)) {
            nrptg -= 1;
        }
    }

    nth = @min(nth, nk0);

    var args: kargs.set_rows = .{
        .nk0 = nk0,
        .ne01 = @intCast(s0.ne[1]),
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne11 = @intCast(s1.ne[1]),
        .ne12 = @intCast(s1.ne[2]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    const ne01: c_int = @intCast(s0.ne[1]);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @divTrunc(ne01 + nrptg - 1, nrptg),
        @intCast(s0.ne[2]),
        @intCast(s0.ne[3]),
        nth,
        nrptg,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_diag` (ggml-metal-ops.cpp:1273 @c1d0e7a00).
pub export fn ggml_metal_op_diag(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    var args: kargs.diag = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_diag(lib, op);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(op.ne[1]),
        @intCast(op.ne[2]),
        @intCast(op.ne[3]),
        32,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_l2_norm` (ggml-metal-ops.cpp:3789 @c1d0e7a00).
pub export fn ggml_metal_op_l2_norm(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const bid_src0 = getBufferId(s0);
    const bid_dst = getBufferId(op);

    const eps = impl.getOpParamsF32(op, 0);

    const ne00: i32 = @intCast(s0.ne[0]);

    var args: kargs.l2_norm = .{
        .ne00 = ne00,
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .eps = eps,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_l2_norm(lib, op);

    if (pipeline.c4) {
        args.ne00 = @divTrunc(ne00, 4);
        args.ne0 = @divTrunc(@as(i32, @intCast(op.ne[0])), 4);
    }

    var nth: c_int = 32; // SIMD width

    while (nth < ne00 and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    nth = @min(nth, maxThreads(pipeline));

    const smem = pipeline.smem;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(s0.ne[1]),
        @intCast(s0.ne[2]),
        @intCast(s0.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_group_norm` (ggml-metal-ops.cpp:3857 @c1d0e7a00).
pub export fn ggml_metal_op_group_norm(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ngrp = impl.getOpParamsI32(op, 0);

    // The C reads `eps` from `op_params + 1`, so index 1.
    const eps = impl.getOpParamsF32(op, 1);

    var args: kargs.group_norm = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .ngrp = ngrp,
        .eps = eps,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_group_norm(lib, op);

    // The C leaves the usual widen-to-SIMD loop commented out here and
    // keeps `nth` at one SIMD width; not reproduced beyond this note.
    const nth: c_int = 32; // SIMD width

    const smem = pipeline.smem;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_norm` (ggml-metal-ops.cpp:3908 @c1d0e7a00).
///
/// The first **fusion** encoder: it may swallow a following `MUL` and
/// then an `ADD` into one kernel, and returns how many nodes it
/// consumed. The `nef*`/`nbf*` array fields carry one entry per fused
/// operand, which is why `kargs_norm` has `[3]` arrays where every other
/// struct has scalars.
pub export fn ggml_metal_op_norm(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const use_fusion = ctx.use_fusion;

    const debug_fusion = ctx.debug_fusion;

    const s0 = src(op, 0);

    const eps = impl.getOpParamsF32(op, 0);

    const bid_src0 = getBufferId(s0);
    var bid_dst = getBufferId(op);

    const ne00: i32 = @intCast(s0.ne[0]);

    var args: kargs.norm = .{
        .ne00 = ne00,
        .ne00_t = if (@rem(ne00, 4) == 0) @divTrunc(ne00, 4) else ne00,
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .eps = eps,
        // The C's `{ ne01 }` sets element 0 and zero-fills the rest.
        .nef1 = .{ @intCast(s0.ne[1]), 0, 0 },
        .nef2 = .{ @intCast(s0.ne[2]), 0, 0 },
        .nef3 = .{ @intCast(s0.ne[3]), 0, 0 },
        .nbf1 = .{ s0.nb[1], 0, 0 },
        .nbf2 = .{ s0.nb[2], 0, 0 },
        .nbf3 = .{ s0.nb[3], 0, 0 },
    };

    var fops: [8]c.enum_ggml_op = undefined;

    var n_fuse: c_int = 1;

    var bid_fuse = [2]mc.BufferId{ bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b)
    // d[2] = add(d[1], c)
    if (use_fusion) {
        fops[0] = op.op;
        fops[1] = c.GGML_OP_MUL;
        fops[2] = c.GGML_OP_ADD;

        n_fuse = 0;
        while (n_fuse <= 1) : (n_fuse += 1) {
            if (!ctx.canFuse(idx + n_fuse, fops[@intCast(n_fuse)..], 2)) {
                break;
            }

            const f0 = ctx.node(idx + n_fuse);
            const f1 = ctx.node(idx + n_fuse + 1);

            if (f0 != impl.one(Tensor, f1.src[0])) {
                break;
            }

            const f1s1 = src(f1, 1);

            if (f1s1.ne[0] != op.ne[0]) {
                break;
            }

            if (!c.ggml_is_contiguous_rows(f1s1)) {
                break;
            }

            if (f1.type != c.GGML_TYPE_F32) {
                break;
            }

            //ctx->fuse_cnt[f1->op]++;

            bid_fuse[@intCast(n_fuse)] = getBufferId(f1s1);

            const k: usize = @intCast(n_fuse + 1);

            args.nef1[k] = @intCast(f1s1.ne[1]);
            args.nef2[k] = @intCast(f1s1.ne[2]);
            args.nef3[k] = @intCast(f1s1.ne[3]);

            args.nbf1[k] = f1s1.nb[1];
            args.nbf2[k] = f1s1.nb[2];
            args.nbf3[k] = f1s1.nb[3];
        }

        n_fuse += 1;

        if (debug_fusion > 1 and n_fuse > 1) {
            if (n_fuse == 2) {
                impl.logDebug("%s: fuse: %s + MUL\n", .{ "ggml_metal_op_norm", c.ggml_op_name(op.op) });
            }
            if (n_fuse == 3) {
                impl.logDebug("%s: fuse: %s + MUL + ADD\n", .{ "ggml_metal_op_norm", c.ggml_op_name(op.op) });
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = getBufferId(ctx.node(idx + n_fuse - 1));

        var i: c_int = 1;
        while (i < n_fuse) : (i += 1) {
            if (!concurrencyCheck(ctx, ctx.node(idx + i))) {
                _ = concurrencyReset(ctx);

                break;
            }
        }
    }

    const pipeline = library.ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    var nth: c_int = 32; // SIMD width

    while (nth < args.ne00_t and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    nth = @min(nth, maxThreads(pipeline));
    nth = @min(nth, @divTrunc(args.ne00_t + 31, 32) * 32);

    const smem = pipeline.smem;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    mc.ggml_metal_encoder_set_buffer(enc, bid_fuse[0], 2);
    mc.ggml_metal_encoder_set_buffer(enc, bid_fuse[1], 3);
    mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 4);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(s0.ne[1]),
        @intCast(s0.ne[2]),
        @intCast(s0.ne[3]),
        nth,
        1,
        1,
    );

    return n_fuse;
}

/// Ports `ggml_metal_op_rope` (ggml-metal-ops.cpp:4046 @c1d0e7a00).
pub export fn ggml_metal_op_rope(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = srcOpt(op, 2);

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne02: i64 = s0.ne[2];
    const ne10: i64 = s1.ne[0];

    // make sure we have one or more position id(ne10) per token(ne02)
    impl.assert(@rem(ne10, ne02) == 0, "ne10 % ne02 == 0");
    impl.assert(ne10 >= ne02, "ne10 >= ne02");

    const nth = @min(@as(c_int, 1024), ne00);

    const n_past = impl.getOpParamsI32(op, 0);
    const n_dims = impl.getOpParamsI32(op, 1);
    //const mode = impl.getOpParamsI32(op, 2);
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const n_ctx_orig = impl.getOpParamsI32(op, 4);

    const freq_base = impl.getOpParamsF32(op, 5);
    const freq_scale = impl.getOpParamsF32(op, 6);
    const ext_factor = impl.getOpParamsF32(op, 7);
    const attn_factor = impl.getOpParamsF32(op, 8);
    const beta_fast = impl.getOpParamsF32(op, 9);
    const beta_slow = impl.getOpParamsF32(op, 10);

    // mrope
    const sect_0 = impl.getOpParamsI32(op, 11);
    const sect_1 = impl.getOpParamsI32(op, 12);
    const sect_2 = impl.getOpParamsI32(op, 13);
    const sect_3 = impl.getOpParamsI32(op, 14);

    const n_offs = impl.getOpParamsI32(op, 15);

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const inplace = op.data == s0.data;

    var args: kargs.rope = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .n_past = n_past,
        .n_dims = n_dims,
        .n_offs = n_offs,
        .n_ctx_orig = n_ctx_orig,
        .freq_base = freq_base,
        .freq_scale = freq_scale,
        .ext_factor = ext_factor,
        .attn_factor = attn_factor,
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .sect_0 = sect_0,
        .sect_1 = sect_1,
        .sect_2 = sect_2,
        .sect_3 = sect_3,
        .src2 = s2 != null,
        .inplace = inplace,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_rope(lib, op);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    if (s2) |t| {
        setBuffer(enc, t, 3);
    } else {
        setBuffer(enc, s0, 3);
    }
    setBuffer(enc, op, 4);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(s0.ne[1]),
        @intCast(s0.ne[2]),
        @intCast(s0.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_im2col` (ggml-metal-ops.cpp:4149 @c1d0e7a00).
pub export fn ggml_metal_op_im2col(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const s0 = impl.getOpParamsI32(op, 0);
    const s1 = impl.getOpParamsI32(op, 1);
    const p0 = impl.getOpParamsI32(op, 2);
    const p1 = impl.getOpParamsI32(op, 3);
    const d0 = impl.getOpParamsI32(op, 4);
    const d1 = impl.getOpParamsI32(op, 5);

    const is_2D = impl.getOpParamsI32(op, 6) == 1;

    const N: i32 = @intCast(src1.ne[if (is_2D) 3 else 2]);
    const IC: i32 = @intCast(src1.ne[if (is_2D) 2 else 1]);
    const IH: i32 = if (is_2D) @intCast(src1.ne[1]) else 1;
    const IW: i32 = @intCast(src1.ne[0]);

    const KH: i32 = if (is_2D) @intCast(src0.ne[1]) else 1;
    const KW: i32 = @intCast(src0.ne[0]);

    const OH: i32 = if (is_2D) @intCast(op.ne[2]) else 1;
    const OW: i32 = @intCast(op.ne[1]);

    const CHW: i32 = IC * KH * KW;

    const ofs0: u64 = src1.nb[if (is_2D) 3 else 2] / 4;
    const ofs1: u64 = src1.nb[if (is_2D) 2 else 1] / 4;

    var args: kargs.im2col = .{
        .ofs0 = ofs0,
        .ofs1 = ofs1,
        .IW = IW,
        .IH = IH,
        .CHW = CHW,
        .s0 = s0,
        .s1 = s1,
        .p0 = p0,
        .p1 = p1,
        .d0 = d0,
        .d1 = d1,
        .N = N,
        .KH = KH,
        .KW = KW,
        .KHW = KH * KW,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_im2col(lib, op);

    if (KH * KW <= maxThreads(pipeline)) {
        const ntptg0: u64 = @min(
            @as(u64, @intCast(@divTrunc(maxThreads(pipeline), KH * KW))),
            @as(u64, @intCast(N)),
        );

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, src1, 1);
        setBuffer(enc, op, 2);

        mc.ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, @intCast(ntptg0), KH, KW);
    } else {
        const n_threads: u64 = @min(@as(u64, @intCast(maxThreads(pipeline))), @as(u64, @intCast(N)));
        const quotient: i64 = @divTrunc(@as(i64, N), @as(i64, @intCast(n_threads))) +
            (if (@rem(@as(i64, N), @as(i64, @intCast(n_threads))) > 0) @as(i64, 1) else 0);

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, src1, 1);
        setBuffer(enc, op, 2);

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @intCast(quotient * @as(i64, CHW)),
            OH,
            OW,
            @intCast(n_threads),
            1,
            1,
        );
    }

    return 1;
}

/// Ports `ggml_metal_op_conv_2d` (ggml-metal-ops.cpp:4229 @c1d0e7a00).
pub export fn ggml_metal_op_conv_2d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
    impl.assert(
        src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_F32,
        "op->src[0]->type is f16 or f32",
    );

    const s0 = impl.getOpParamsI32(op, 0);
    const s1 = impl.getOpParamsI32(op, 1);
    const p0 = impl.getOpParamsI32(op, 2);
    const p1 = impl.getOpParamsI32(op, 3);
    const d0 = impl.getOpParamsI32(op, 4);
    const d1 = impl.getOpParamsI32(op, 5);

    var args: kargs.conv_2d = .{
        .nb00 = src0.nb[0],
        .nb01 = src0.nb[1],
        .nb02 = src0.nb[2],
        .nb03 = src0.nb[3],
        .nb10 = src1.nb[0],
        .nb11 = src1.nb[1],
        .nb12 = src1.nb[2],
        .nb13 = src1.nb[3],
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .IW = @intCast(src1.ne[0]),
        .IH = @intCast(src1.ne[1]),
        .KW = @intCast(src0.ne[0]),
        .KH = @intCast(src0.ne[1]),
        .IC = @intCast(src0.ne[2]),
        .OC = @intCast(src0.ne[3]),
        .OW = @intCast(op.ne[0]),
        .OH = @intCast(op.ne[1]),
        .N = @intCast(op.ne[3]),
        .s0 = s0,
        .s1 = s1,
        .p0 = p0,
        .p1 = p1,
        .d0 = d0,
        .d1 = d1,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_conv_2d(lib, op);

    var nth = maxThreads(pipeline);
    nth = @min(nth, 256);
    nth = @max(nth, 1);

    const n_out: u64 = @intCast(c.ggml_nelements(op));

    var tg: u64 = @divTrunc(n_out + @as(u64, @intCast(nth)) - 1, @as(u64, @intCast(nth)));
    tg = @max(tg, 1);
    tg = @min(tg, @as(u64, std.math.maxInt(c_int)));

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(tg), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_conv_2d_dw` (ggml-metal-ops.cpp:4307 @c1d0e7a00).
///
/// **The C writes `/*.nb02 =*/ nb03` here** — the `nb02` field is fed the
/// weight tensor's `nb[3]`, not its `nb[2]`. Reproduced as written. Which
/// it is cannot be settled from this side: the kernel reads the field by
/// offset, so if this is an upstream slip it is one the kernel has been
/// compiled against. `make node-diff ARGS=--gpu` would be the arbiter,
/// and `CONV_2D_DW` is not in a Qwen3.5 graph.
pub export fn ggml_metal_op_conv_2d_dw(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
    impl.assert(
        src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_F32,
        "op->src[0]->type is f16 or f32",
    );

    const s0 = impl.getOpParamsI32(op, 0);
    const s1 = impl.getOpParamsI32(op, 1);
    const p0 = impl.getOpParamsI32(op, 2);
    const p1 = impl.getOpParamsI32(op, 3);
    const d0 = impl.getOpParamsI32(op, 4);
    const d1 = impl.getOpParamsI32(op, 5);

    var args: kargs.conv_2d_dw = .{
        .nb00 = src0.nb[0],
        .nb01 = src0.nb[1],
        .nb02 = src0.nb[3], // the C's `/*.nb02 =*/ nb03` -- see the note above
        .nb10 = src1.nb[0],
        .nb11 = src1.nb[1],
        .nb12 = src1.nb[2],
        .nb13 = src1.nb[3],
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .IW = @intCast(src1.ne[0]),
        .IH = @intCast(src1.ne[1]),
        .KW = @intCast(src0.ne[0]),
        .KH = @intCast(src0.ne[1]),
        .C = @intCast(src1.ne[2]),
        .OW = @intCast(op.ne[0]),
        .OH = @intCast(op.ne[1]),
        .N = @intCast(src1.ne[3]),
        .s0 = s0,
        .s1 = s1,
        .p0 = p0,
        .p1 = p1,
        .d0 = d0,
        .d1 = d1,
    };

    const use_tiled = src1.nb[2] < src1.nb[0];

    const pipeline = library.ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

    var nth = maxThreads(pipeline);
    nth = @min(nth, 256);
    nth = @max(nth, 1);

    const OW: i32 = @intCast(op.ne[0]);
    const OH: i32 = @intCast(op.ne[1]);
    const C: i32 = @intCast(src1.ne[2]);
    const N: i32 = @intCast(src1.ne[3]);

    const tg_x: c_int = if (use_tiled)
        @divTrunc(C + nth - 1, nth)
    else
        @divTrunc(OW + nth - 1, nth);
    const tg_y: c_int = OH;
    const tg_z: c_int = if (use_tiled) OW * N else C * N;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_conv_transpose_1d` (ggml-metal-ops.cpp:4458
/// @c1d0e7a00).
pub export fn ggml_metal_op_conv_transpose_1d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    // `s0` is the C's name for the stride op-param; the tensors are
    // reached through `src0`/`src1` here so the two do not collide.
    const s0 = impl.getOpParamsI32(op, 0);

    const IC: i32 = @intCast(src1.ne[1]);
    const IL: i32 = @intCast(src1.ne[0]);

    const K: i32 = @intCast(src0.ne[0]);

    const OL: i32 = @intCast(op.ne[0]);
    const OC: i32 = @intCast(op.ne[1]);

    var args: kargs.conv_transpose_1d = .{
        .IC = IC,
        .IL = IL,
        .K = K,
        .s0 = s0,
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_conv_transpose_2d` (ggml-metal-ops.cpp:4593
/// @c1d0e7a00).
pub export fn ggml_metal_op_conv_transpose_2d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const s0 = impl.getOpParamsI32(op, 0);

    const IC: i32 = @intCast(src1.ne[2]);
    const IH: i32 = @intCast(src1.ne[1]);
    const IW: i32 = @intCast(src1.ne[0]);

    const KH: i32 = @intCast(src0.ne[1]);
    const KW: i32 = @intCast(src0.ne[0]);

    const OW: i32 = @intCast(op.ne[0]);
    const OH: i32 = @intCast(op.ne[1]);
    const OC: i32 = @intCast(op.ne[2]);

    var args: kargs.conv_transpose_2d = .{
        .IC = IC,
        .IH = IH,
        .IW = IW,
        .KH = KH,
        .KW = KW,
        .OC = OC,
        .s0 = s0,
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    // Metal requires buffer size to be multiple of 16 bytes
    const smem = impl.pad(@as(usize, @intCast(KW)) * @as(usize, @intCast(KH)) * @sizeOf(f32), 16);
    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC, KW, KH, 1);

    return 1;
}

/// Ports `ggml_metal_op_col2im_1d` (ggml-metal-ops.cpp:4503 @c1d0e7a00).
pub export fn ggml_metal_op_col2im_1d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);

    const s0 = impl.getOpParamsI32(op, 0);
    const OC = impl.getOpParamsI32(op, 1);
    const p0 = impl.getOpParamsI32(op, 2);

    const K_OC: i32 = @intCast(src0.ne[0]);
    const T_in: i32 = @intCast(src0.ne[1]);
    const K: i32 = @divTrunc(K_OC, OC);
    const T_out: i32 = @intCast(op.ne[0]);

    var args: kargs.col2im_1d = .{
        .T_in = T_in,
        .T_out = T_out,
        .OC = OC,
        .K = K,
        .K_OC = K_OC,
        .s0 = s0,
        .p0 = p0,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    const total = T_out * OC;
    const nth: i32 = 256;
    const ntg = @divTrunc(total + nth - 1, nth);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_conv_3d` (ggml-metal-ops.cpp:4387 @c1d0e7a00).
///
/// The C's initialiser is positional from `s0` onwards rather than
/// designated; the field names here come from the struct's declaration
/// order, which `kargs.zig` carries from the header.
pub export fn ggml_metal_op_conv_3d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    // 2. Extract hyperparams from op_params
    const s0 = impl.getOpParamsI32(op, 0);
    const s1 = impl.getOpParamsI32(op, 1);
    const s2 = impl.getOpParamsI32(op, 2);
    const p0 = impl.getOpParamsI32(op, 3);
    const p1 = impl.getOpParamsI32(op, 4);
    const p2 = impl.getOpParamsI32(op, 5);
    const d0 = impl.getOpParamsI32(op, 6);
    const d1 = impl.getOpParamsI32(op, 7);
    const d2 = impl.getOpParamsI32(op, 8);
    const IC = impl.getOpParamsI32(op, 9);
    const N = impl.getOpParamsI32(op, 10);
    const OC = impl.getOpParamsI32(op, 11);

    // 3. Build the parameter struct
    var args: kargs.conv_3d = .{
        .IW = @intCast(src1.ne[0]),
        .IH = @intCast(src1.ne[1]),
        .ID = @intCast(src1.ne[2]),
        .OW = @intCast(op.ne[0]),
        .OH = @intCast(op.ne[1]),
        .OD = @intCast(op.ne[2]),
        .KW = @intCast(src0.ne[0]),
        .KH = @intCast(src0.ne[1]),
        .KD = @intCast(src0.ne[2]),
        .s0 = s0,
        .s1 = s1,
        .s2 = s2,
        .p0 = p0,
        .p1 = p1,
        .p2 = p2,
        .d0 = d0,
        .d1 = d1,
        .d2 = d2,
        .IC = IC,
        .N = N,
        .OC = OC,
        .nb00 = src0.nb[0], // Weight strides
        .nb01 = src0.nb[1],
        .nb02 = src0.nb[2],
        .nb03 = src0.nb[3],
        .nb10 = src1.nb[0], // Input strides
        .nb11 = src1.nb[1],
        .nb12 = src1.nb[2],
        .nb13 = src1.nb[3],
        .nb0 = op.nb[0], // Output strides
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    // 4. Fetch the JIT pipeline
    const pipeline = library.ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    const nth0: c_int = 32; // Standard SIMD width for Apple Silicon
    const nth1: c_int = 1;
    const nth2: c_int = 1;

    const spatial_volume: i64 = @as(i64, args.OW) * @as(i64, args.OH) * @as(i64, args.OD);

    const ntg0: c_int = @intCast(@divTrunc(spatial_volume + nth0 - 1, nth0));
    const ntg1: c_int = args.OC;
    const ntg2: c_int = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    return 1;
}

/// Ports `ggml_metal_op_upscale` (ggml-metal-ops.cpp:4649 @c1d0e7a00).
pub export fn ggml_metal_op_upscale(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ne00: i64 = s0.ne[0];
    const ne01: i64 = s0.ne[1];
    const ne0: i64 = op.ne[0];
    const ne1: i64 = op.ne[1];
    const ne2: i64 = op.ne[2];
    const ne3: i64 = op.ne[3];

    var sf0: f32 = @as(f32, @floatFromInt(ne0)) / @as(f32, @floatFromInt(s0.ne[0]));
    var sf1: f32 = @as(f32, @floatFromInt(ne1)) / @as(f32, @floatFromInt(s0.ne[1]));
    const sf2: f32 = @as(f32, @floatFromInt(ne2)) / @as(f32, @floatFromInt(s0.ne[2]));
    const sf3: f32 = @as(f32, @floatFromInt(ne3)) / @as(f32, @floatFromInt(s0.ne[3]));

    const mode_flags = impl.getOpParamsI32(op, 0);

    var poffs: f32 = 0.5;

    if ((mode_flags & c.GGML_SCALE_FLAG_ALIGN_CORNERS) != 0) {
        poffs = 0.0;
        sf0 = if (ne0 > 1 and ne00 > 1)
            @as(f32, @floatFromInt(ne0 - 1)) / @as(f32, @floatFromInt(ne00 - 1))
        else
            sf0;
        sf1 = if (ne1 > 1 and ne01 > 1)
            @as(f32, @floatFromInt(ne1 - 1)) / @as(f32, @floatFromInt(ne01 - 1))
        else
            sf1;
    }

    var args: kargs.upscale = .{
        .ne00 = ne00,
        .ne01 = ne01,
        .ne02 = s0.ne[2],
        .ne03 = s0.ne[3],
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = ne0,
        .ne1 = ne1,
        .ne2 = ne2,
        .ne3 = ne3,
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .sf0 = sf0,
        .sf1 = sf1,
        .sf2 = sf2,
        .sf3 = sf3,
        .poffs = poffs,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_upscale(lib, op);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(ne0)));

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(ne1),
        @intCast(ne2),
        @intCast(ne3),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_pad` (ggml-metal-ops.cpp:4766 @c1d0e7a00).
pub export fn ggml_metal_op_pad(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ne00: i64 = s0.ne[0];
    const ne0: i64 = op.ne[0];

    var args: kargs.pad = .{
        .ne00 = ne00,
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = ne0,
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_pad(lib, op);

    if (pipeline.c4) {
        args.ne00 = @divTrunc(ne00, 4);
        args.ne0 = @divTrunc(ne0, 4);
    }

    // `kargs_pad` is one of the structs whose `ne` fields are `int64_t`,
    // where the C's `GGML_TENSOR_LOCALS` locals are `int32_t` — it widens
    // on the way in and truncates back to `int` here. Both narrowings are
    // written out rather than left implicit.
    const nth_max = @min(@as(c_int, 64), maxThreads(pipeline));
    const nth: c_int = @min(@as(c_int, @intCast(args.ne0)), nth_max);
    const nk0: c_int = @divTrunc(@as(c_int, @intCast(args.ne0)) + 1024 - 1, 1024); // note: 1024 is hardcoded in the kernel!

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        nk0 * @as(c_int, @intCast(op.ne[1])),
        @intCast(op.ne[2]),
        @intCast(op.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_pad_reflect_1d` (ggml-metal-ops.cpp:4817
/// @c1d0e7a00).
pub export fn ggml_metal_op_pad_reflect_1d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    var args: kargs.pad_reflect_1d = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .p0 = impl.getOpParamsI32(op, 0),
        .p1 = impl.getOpParamsI32(op, 1),
    };

    const pipeline = library.ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const nth = @min(@as(c_int, 1024), @as(c_int, @intCast(op.ne[0])));

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(op.ne[1]),
        @intCast(op.ne[2]),
        @intCast(op.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_roll` (ggml-metal-ops.cpp:4713 @c1d0e7a00).
pub export fn ggml_metal_op_roll(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);

    const s0 = impl.getOpParamsI32(op, 0);
    const s1 = impl.getOpParamsI32(op, 1);
    const s2 = impl.getOpParamsI32(op, 2);
    const s3 = impl.getOpParamsI32(op, 3);

    var args: kargs.roll = .{
        .ne00 = @intCast(src0.ne[0]),
        .ne01 = @intCast(src0.ne[1]),
        .ne02 = @intCast(src0.ne[2]),
        .ne03 = @intCast(src0.ne[3]),
        .nb00 = src0.nb[0],
        .nb01 = src0.nb[1],
        .nb02 = src0.nb[2],
        .nb03 = src0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .s0 = s0,
        .s1 = s1,
        .s2 = s2,
        .s3 = s3,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_roll(lib, op);

    const nth = @min(@as(c_int, 1024), @as(c_int, @intCast(op.ne[0])));

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(op.ne[1]),
        @intCast(op.ne[2]),
        @intCast(op.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_arange` (ggml-metal-ops.cpp:4863 @c1d0e7a00).
pub export fn ggml_metal_op_arange(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    // The C `memcpy`s these out of `op_params` as `float`, reading
    // indices 0 and 2 -- not 0 and 1.
    const start = impl.getOpParamsF32(op, 0);
    const step = impl.getOpParamsF32(op, 2);

    var args: kargs.arange = .{
        .ne0 = @intCast(op.ne[0]),
        .start = start,
        .step = step,
    };

    const ne0: c_int = @intCast(op.ne[0]);
    const nth = @min(@as(c_int, 1024), ne0);

    const pipeline = library.ggml_metal_library_get_pipeline_arange(lib, op);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, op, 1);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_timestep_embedding` (ggml-metal-ops.cpp:4897
/// @c1d0e7a00).
pub export fn ggml_metal_op_timestep_embedding(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const dim = impl.getOpParamsI32(op, 0);
    const max_period = impl.getOpParamsI32(op, 1);

    var args: kargs.timestep_embedding = .{
        .nb1 = op.nb[1],
        .dim = dim,
        .max_period = max_period,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const nth = @max(@as(c_int, 1), @min(@as(c_int, 1024), @divTrunc(dim, 2)));

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(s0.ne[0]), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_argsort` (ggml-metal-ops.cpp:4970 @c1d0e7a00).
///
/// A bitonic sort of power-of-two blocks, then a merge ladder that
/// ping-pongs between `dst` and the scratch immediately after it. The
/// parity test up front decides which of the two the first pass writes
/// to, so the last merge lands in `dst`.
pub export fn ggml_metal_op_argsort(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);

    const pipeline = library.ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    var nth: c_int = 1;
    while (nth < ne00 and 2 * nth <= maxThreads(pipeline)) {
        nth *= 2;
    }

    const npr = @divTrunc(ne00 + nth - 1, nth);

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const smem = impl.pad(@as(usize, @intCast(nth)) * @sizeOf(i32), 16);

    const bid_src0 = getBufferId(s0);
    var bid_dst = getBufferId(op);

    var bid_tmp = bid_dst;
    bid_tmp.offs += @intCast(c.ggml_nbytes(op));

    if (@rem(@as(c_int, @intFromFloat(@ceil(@log(@as(f64, @floatFromInt(npr))) / @log(@as(f64, 2))))), 2) == 1) {
        std.mem.swap(mc.BufferId, &bid_dst, &bid_tmp);
    }

    var args: kargs.argsort = .{
        .ne00 = ne00,
        .ne01 = ne01,
        .ne02 = ne02,
        .ne03 = ne03,
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .top_k = nth,
    };

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, npr * ne01, ne02, ne03, nth, 1, 1);

    const pipeline_merge = library.ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    var len: c_int = nth;

    while (len < ne00) {
        _ = concurrencyReset(ctx);

        var args_merge: kargs.argsort_merge = .{
            .ne00 = ne00,
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne0 = @intCast(op.ne[0]),
            .ne1 = @intCast(op.ne[1]),
            .ne2 = @intCast(op.ne[2]),
            .ne3 = @intCast(op.ne[3]),
            .top_k = ne00,
            .len = len,
        };

        // merges per row
        const nm = @divTrunc(ne00 + 2 * len - 1, 2 * len);

        const nth_merge = @min(@as(c_int, 512), maxThreads(pipeline_merge));

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        setBytes(enc, &args_merge, 0);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);
        mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 3);

        mc.ggml_metal_encoder_dispatch_threadgroups(enc, nm * ne01, ne02, ne03, nth_merge, 1, 1);

        std.mem.swap(mc.BufferId, &bid_dst, &bid_tmp);

        len <<= 1;
    }

    return 1;
}

/// Ports `ggml_metal_op_top_k` (ggml-metal-ops.cpp:5077 @c1d0e7a00).
///
/// `argsort`'s structure, but each block keeps only its own top-k, so
/// `args.ne0` is recomputed to the total kept rather than the row width,
/// and the final merge narrows to `top_k`. It reuses
/// `kargs_argsort`/`kargs_argsort_merge`.
pub export fn ggml_metal_op_top_k(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    impl.assert(c.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);
    const ne1: c_int = @intCast(op.ne[1]);
    const ne2: c_int = @intCast(op.ne[2]);
    const ne3: c_int = @intCast(op.ne[3]);

    const pipeline = library.ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    var nth: c_int = 1;
    while (nth < ne00 and 2 * nth <= maxThreads(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const npr = @divTrunc(ne00 + nth - 1, nth);

    const smem = impl.pad(@as(usize, @intCast(nth)) * @sizeOf(i32), 16);

    const bid_src0 = getBufferId(s0);
    var bid_dst = getBufferId(op);

    var bid_tmp = bid_dst;
    bid_tmp.offs += @sizeOf(i32) * @as(usize, @intCast(c.ggml_nelements(s0)));

    if (@rem(@as(c_int, @intFromFloat(@ceil(@log(@as(f64, @floatFromInt(npr))) / @log(@as(f64, 2))))), 2) == 1) {
        std.mem.swap(mc.BufferId, &bid_dst, &bid_tmp);
    }

    const top_k: c_int = @intCast(op.ne[0]);

    var args: kargs.argsort = .{
        .ne00 = ne00,
        .ne01 = ne01,
        .ne02 = ne02,
        .ne03 = ne03,
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = ne1,
        .ne2 = ne2,
        .ne3 = ne3,
        .top_k = @min(nth, top_k), // for each block, keep just the top_k indices
    };

    if (npr > 1) {
        args.ne0 = (npr - 1) * args.top_k + @min(ne00 - (npr - 1) * nth, args.top_k);
    }

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, npr * ne01, ne02, ne03, nth, 1, 1);

    const pipeline_merge = library.ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    var len: c_int = args.top_k;

    while (len < args.ne0) {
        _ = concurrencyReset(ctx);

        // merges per row
        const nm = @divTrunc(args.ne0 + 2 * len - 1, 2 * len);

        const nth_merge = @min(@as(c_int, 512), @min(len, maxThreads(pipeline_merge)));

        var args_merge: kargs.argsort_merge = .{
            .ne00 = ne00,
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne0 = args.ne0,
            .ne1 = ne1,
            .ne2 = ne2,
            .ne3 = ne3,
            .top_k = if (nm == 1) top_k else args.ne0, // the final merge outputs top_k elements
            .len = len,
        };

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        setBytes(enc, &args_merge, 0);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);
        mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 3);

        mc.ggml_metal_encoder_dispatch_threadgroups(enc, nm * ne01, ne02, ne03, nth_merge, 1, 1);

        std.mem.swap(mc.BufferId, &bid_dst, &bid_tmp);

        len <<= 1;
    }

    return 1;
}

/// Ports `ggml_metal_op_tri` (ggml-metal-ops.cpp:5189 @c1d0e7a00).
pub export fn ggml_metal_op_tri(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ne00: c_int = @intCast(s0.ne[0]);

    var args: kargs.tri = .{
        .ne00 = ne00,
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_tri(lib, op);

    var nth: c_int = 32; // SIMD width

    while (nth < ne00 and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    nth = @min(nth, maxThreads(pipeline));
    nth = @min(nth, ne00);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(s0.ne[1]),
        @intCast(s0.ne[2]),
        @intCast(s0.ne[3]),
        nth,
        1,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_flash_attn_ext` (ggml-metal-ops.cpp:2999
/// @c1d0e7a00).
///
/// The largest encoder in the file, and up to **five** dispatches: an
/// optional KV dequantise to F16, an optional KV-tail pad, an optional
/// mask-block scan, the attention kernel itself, and — on the vector
/// path with more than one workgroup — a reduce over the per-workgroup
/// partials. Four scratch regions sit after `dst`, which is what
/// `backend.zig`'s `get_alloc_size` reserves.
///
/// Several of the C's branches are commented out to a constant
/// (`if (false)`, `if (true)`, a forced `has_kvpad`) so that scratch is
/// always reserved and graphs are not reallocated. Those are reproduced
/// as the live arm only, each noted where it occurs.
pub export fn ggml_metal_op_flash_attn_ext(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const props_dev = mc.ggml_metal_device_get_props(ctx.dev);

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = src(op, 2);
    const s3o = srcOpt(op, 3);

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);
    const ne10: c_int = @intCast(s1.ne[0]);
    const ne11: c_int = @intCast(s1.ne[1]);
    const ne12: c_int = @intCast(s1.ne[2]);
    const ne13: c_int = @intCast(s1.ne[3]);
    const ne20: c_int = @intCast(s2.ne[0]);
    const ne21: c_int = @intCast(s2.ne[1]);
    const ne22: c_int = @intCast(s2.ne[2]);
    const ne23: c_int = @intCast(s2.ne[3]);

    // src[3] is the mask and may be absent; `GGML_TENSOR_LOCALS` is
    // null-guarded, so these are zero when it is.
    const ne30: c_int = if (s3o) |t| @intCast(t.ne[0]) else 0;
    const ne31: c_int = if (s3o) |t| @intCast(t.ne[1]) else 0;
    const ne32: c_int = if (s3o) |t| @intCast(t.ne[2]) else 0;
    const ne33: c_int = if (s3o) |t| @intCast(t.ne[3]) else 0;
    const nb31: u64 = if (s3o) |t| t.nb[1] else 0;
    const nb32: u64 = if (s3o) |t| t.nb[2] else 0;
    const nb33: u64 = if (s3o) |t| t.nb[3] else 0;

    impl.assert(@rem(ne00, 4) == 0, "ne00 % 4 == 0");

    impl.assert(s0.type == c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    impl.assert(s1.type == s2.type, "op->src[1]->type == op->src[2]->type");

    impl.assert(ne11 == ne21, "ne11 == ne21");
    impl.assert(ne12 == ne22, "ne12 == ne22");

    impl.assert(s3o == null or s3o.?.type == c.GGML_TYPE_F16, "!op->src[3] || op->src[3]->type == GGML_TYPE_F16");
    impl.assert(
        s3o == null or s3o.?.ne[1] >= s0.ne[1],
        "the Flash-Attention Metal kernel requires the mask to be at least n_queries big",
    );

    var scale = impl.getOpParamsF32(op, 0);
    const max_bias = impl.getOpParamsF32(op, 1);
    const logit_softcap = impl.getOpParamsF32(op, 2);

    if (logit_softcap != 0.0) {
        scale /= logit_softcap;
    }

    const has_mask = op.src[3] != null;
    const has_sinks = op.src[4] != null;
    const has_bias = max_bias != 0.0;
    const has_scap = logit_softcap != 0.0;

    const n_head: u32 = @intCast(s0.ne[2]);
    const n_head_log2: i32 = @as(i32, 1) << @intCast(@as(u32, @intFromFloat(@floor(std.math.log2(@as(f32, @floatFromInt(n_head)))))));

    const m0 = std.math.pow(f32, 2.0, -(max_bias) / @as(f32, @floatFromInt(n_head_log2)));
    const m1 = std.math.pow(f32, 2.0, -(max_bias / 2.0) / @as(f32, @floatFromInt(n_head_log2)));

    impl.assert(ne01 < 65536, "ne01 < 65536");

    const bid_src0 = getBufferId(s0);
    const bid_src1 = getBufferId(s1);
    const bid_src2 = getBufferId(s2);
    const bid_src3 = if (has_mask) getBufferId(s3o.?) else bid_src0;
    const bid_src4 = if (has_sinks) getBufferId(src(op, 4)) else bid_src0;

    const bid_dst = getBufferId(op);

    var bid_pad = bid_dst;
    bid_pad.offs += @intCast(c.ggml_nbytes(op));

    var bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    var bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    var bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    const use_kv_f16 = useKvF16(op);

    var bid_k = bid_src1;
    var bid_v = bid_src2;

    var nb10_attn: u64 = s1.nb[0];
    var nb11_attn: u64 = s1.nb[1];
    var nb12_attn: u64 = s1.nb[2];
    var nb13_attn: u64 = s1.nb[3];
    var nb20_attn: u64 = s2.nb[0];
    var nb21_attn: u64 = s2.nb[1];
    var nb22_attn: u64 = s2.nb[2];
    var nb23_attn: u64 = s2.nb[3];
    _ = &nb20_attn;

    if (use_kv_f16) {
        impl.assert(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0, "extra_kv_f16(op) != 0");

        const v_is_view_of_k = vIsViewOfK(op);

        const nblocks1_64: i64 = @divTrunc(s1.ne[0], c.ggml_blck_size(s1.type)) * s1.ne[1] * s1.ne[2] * s1.ne[3];
        impl.assert(nblocks1_64 <= std.math.maxInt(i32), "nblocks1_64 <= INT32_MAX");
        const nblocks1: i32 = @intCast(nblocks1_64);

        var bid_v_f16 = bid_kv_f16;
        bid_v_f16.offs += kvF16KSize(op);

        const pipeline0 = library.ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
        const nth = @min(maxThreads(pipeline0), @as(c_int, 256));

        // K
        var args_k: kargs.flash_attn_ext_kv_f16 = .{
            .ne0 = ne10,
            .ne1 = ne11,
            .ne2 = ne12,
            .ne3 = ne13,
            .nb0 = s1.nb[0],
            .nb1 = s1.nb[1],
            .nb2 = s1.nb[2],
            .nb3 = s1.nb[3],
            .nblocks = nblocks1,
        };

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline0);
        setBytes(enc, &args_k, 0);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src1, 1);
        mc.ggml_metal_encoder_set_buffer(enc, bid_kv_f16, 2);

        mc.ggml_metal_encoder_dispatch_threadgroups(enc, @divTrunc(nblocks1 + nth - 1, nth), 1, 1, nth, 1, 1);

        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
        if (!v_is_view_of_k) {
            const nblocks2_64: i64 = @divTrunc(s2.ne[0], c.ggml_blck_size(s2.type)) * s2.ne[1] * s2.ne[2] * s2.ne[3];
            impl.assert(nblocks2_64 <= std.math.maxInt(i32), "nblocks2_64 <= INT32_MAX");
            const nblocks2: i32 = @intCast(nblocks2_64);

            var args_v: kargs.flash_attn_ext_kv_f16 = .{
                .ne0 = ne20,
                .ne1 = ne21,
                .ne2 = ne22,
                .ne3 = ne23,
                .nb0 = s2.nb[0],
                .nb1 = s2.nb[1],
                .nb2 = s2.nb[2],
                .nb3 = s2.nb[3],
                .nblocks = nblocks2,
            };

            mc.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args_v, 0);
            mc.ggml_metal_encoder_set_buffer(enc, bid_src2, 1);
            mc.ggml_metal_encoder_set_buffer(enc, bid_v_f16, 2);

            mc.ggml_metal_encoder_dispatch_threadgroups(enc, @divTrunc(nblocks2 + nth - 1, nth), 1, 1, nth, 1, 1);
        }

        // the pad and attention kernels read the dequantized KV
        _ = concurrencyReset(ctx);

        bid_k = bid_kv_f16;
        bid_v = if (v_is_view_of_k) bid_k else bid_v_f16;

        // contiguous F16 layout of the dequantized K
        nb10_attn = @sizeOf(c.ggml_fp16_t);
        nb11_attn = nb10_attn * @as(u64, @intCast(ne10));
        nb12_attn = nb11_attn * @as(u64, @intCast(ne11));
        nb13_attn = nb12_attn * @as(u64, @intCast(ne12));

        // if V is a view of K, the dequantized V is read from the dequantized K with K's strides
        if (v_is_view_of_k) {
            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } else {
            // contiguous F16 layout of the dequantized V
            nb20_attn = @sizeOf(c.ggml_fp16_t);
            nb21_attn = nb20_attn * @as(u64, @intCast(ne20));
            nb22_attn = nb21_attn * @as(u64, @intCast(ne21));
            nb23_attn = nb22_attn * @as(u64, @intCast(ne22));
        }
    }

    if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const nqptg: c_int = fc.OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const ncpsg: c_int = fc.OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        comptime {
            if (fc.OP_FLASH_ATTN_EXT_NQPSG > 32 or
                @rem(fc.OP_FLASH_ATTN_EXT_NQPSG, 8) != 0 or
                @rem(fc.OP_FLASH_ATTN_EXT_NCPSG, 32) != 0)
            {
                @compileError("nqptg <= 32 && nqptg % 8 == 0 && ncpsg % 32 == 0");
            }
        }

        var need_sync = false;

        const has_kvpad = @rem(ne11, ncpsg) != 0;

        if (has_kvpad) {
            impl.assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0, "extra_pad(op) != 0");

            var args0: kargs.flash_attn_ext_pad = .{
                .ne11 = ne11,
                .ne_12_2 = ne12,
                .ne_12_3 = ne13,
                .nb11 = nb11_attn,
                .nb12 = nb12_attn,
                .nb13 = nb13_attn,
                .nb21 = nb21_attn,
                .nb22 = nb22_attn,
                .nb23 = nb23_attn,
                .ne31 = ne31,
                .ne32 = ne32,
                .ne33 = ne33,
                .nb31 = nb31,
                .nb32 = nb32,
                .nb33 = nb33,
            };

            const pipeline0 = library.ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            mc.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args0, 0);
            mc.ggml_metal_encoder_set_buffer(enc, bid_k, 1);
            mc.ggml_metal_encoder_set_buffer(enc, bid_v, 2);
            mc.ggml_metal_encoder_set_buffer(enc, bid_src3, 3);
            mc.ggml_metal_encoder_set_buffer(enc, bid_pad, 4);

            impl.assert(ne12 == ne22, "ne12 == ne22");
            impl.assert(ne13 == ne23, "ne13 == ne23");

            mc.ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, @max(ne12, ne32), @max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            impl.assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0, "extra_blk(op) != 0");

            var args0: kargs.flash_attn_ext_blk = .{
                .ne01 = ne01,
                .ne30 = ne30,
                .ne31 = ne31,
                .ne32 = ne32,
                .ne33 = ne33,
                .nb31 = nb31,
                .nb32 = nb32,
                .nb33 = nb33,
            };

            const pipeline0 = library.ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            mc.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args0, 0);
            mc.ggml_metal_encoder_set_buffer(enc, bid_src3, 1);
            mc.ggml_metal_encoder_set_buffer(enc, bid_blk, 2);

            const nblk1: i32 = @divTrunc(ne01 + nqptg - 1, nqptg);
            const nblk0: i32 = @divTrunc(ne30 + ncpsg - 1, ncpsg);

            mc.ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32 * ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            _ = concurrencyReset(ctx);
        }

        const is_q: c_int = if (!use_kv_f16 and c.ggml_is_quantized(s1.type)) 1 else 0;

        // 2*(2*ncpsg)
        // ncpsg soft_max values + ncpsg mask values
        //
        // 16*32*(nsg)
        // the shared memory needed for the simdgroups to load the KV cache
        // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
        //
        // the C's FATTN_SMEM(nsg) macro
        const fattnSmem = struct {
            fn f(ne00_: c_int, ne20_: c_int, ncpsg_: c_int, nqptg_: c_int, is_q_: c_int, nsg_: i32) usize {
                const terms = nqptg_ * (ne00_ + 2 * @as(c_int, @intCast(impl.pad(@intCast(ne20_), 64))) + 2 * (2 * ncpsg_)) +
                    is_q_ * (16 * 32 * nsg_);
                return impl.pad(@as(usize, @intCast(terms)) * (@sizeOf(f32) / 2), 16);
            }
        }.f;

        // simdgroups per threadgroup (a.k.a. warps)
        const nsg: i32 = if (ne00 >= 512) 8 else 4;

        const smem = fattnSmem(ne00, ne20, ncpsg, nqptg, is_q, nsg);

        const ns10: i32 = @intCast(nb11_attn / nb10_attn);
        const ns20: i32 = @intCast(nb21_attn / nb20_attn);

        var args: kargs.flash_attn_ext = .{
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne11 = ne11,
            .ne_12_2 = ne12,
            .ne_12_3 = ne13,
            .ns10 = ns10,
            .nb11 = nb11_attn,
            .nb12 = nb12_attn,
            .nb13 = nb13_attn,
            .ns20 = ns20,
            .nb21 = nb21_attn,
            .nb22 = nb22_attn,
            .nb23 = nb23_attn,
            .ne31 = ne31,
            .ne32 = ne32,
            .ne33 = ne33,
            .nb31 = nb31,
            .nb32 = nb32,
            .nb33 = nb33,
            .ne1 = @intCast(op.ne[1]),
            .ne2 = @intCast(op.ne[2]),
            .ne3 = @intCast(op.ne[3]),
            .scale = scale,
            .max_bias = max_bias,
            .m0 = m0,
            .m1 = m1,
            .n_head_log2 = n_head_log2,
            .logit_softcap = logit_softcap,
        };

        const pipeline = library.ggml_metal_library_get_pipeline_flash_attn_ext(
            lib,
            op,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            nsg,
            use_kv_f16,
            ns10,
            ns20,
        );

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        mc.ggml_metal_encoder_set_buffer(enc, bid_k, 2);
        mc.ggml_metal_encoder_set_buffer(enc, bid_v, 3);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src3, 4);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src4, 5);
        mc.ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
        mc.ggml_metal_encoder_set_buffer(enc, bid_blk, 7);
        mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 8);

        mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @divTrunc(ne01 + nqptg - 1, nqptg),
            ne02,
            ne03,
            32,
            nsg,
            1,
        );
    } else {
        // half4x4 kernel
        var cfg = tuning.pick(
            @intFromEnum(props_dev.device_id),
            props_dev.gpu_family,
            @intCast(s1.type),
            ne00,
            ne20, // dk, dv (ne00 == dk for FA)
            ne11,
            ne01,
        );
        var nqptg: c_int = cfg.Q; // queries per threadgroup
        const ncpsg: c_int = fc.OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const nhptg: c_int = 1; // heads per threadgroup

        impl.assert(nqptg <= 32, "nqptg <= 32");
        impl.assert(nqptg == 1 or nqptg == 2 or nqptg == 4, "only instantiated Q values");
        comptime {
            if (@rem(fc.OP_FLASH_ATTN_EXT_VEC_NCPSG, 32) != 0) {
                @compileError("ncpsg % 32 == 0");
            }
        }

        var need_sync = false;

        const has_kvpad = @rem(ne11, ncpsg) != 0;

        if (has_kvpad) {
            impl.assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0, "extra_pad(op) != 0");

            var args0: kargs.flash_attn_ext_pad = .{
                .ne11 = ne11,
                .ne_12_2 = ne12,
                .ne_12_3 = ne13,
                .nb11 = nb11_attn,
                .nb12 = nb12_attn,
                .nb13 = nb13_attn,
                .nb21 = nb21_attn,
                .nb22 = nb22_attn,
                .nb23 = nb23_attn,
                .ne31 = ne31,
                .ne32 = ne32,
                .ne33 = ne33,
                .nb31 = nb31,
                .nb32 = nb32,
                .nb33 = nb33,
            };

            const pipeline0 = library.ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            mc.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args0, 0);
            mc.ggml_metal_encoder_set_buffer(enc, bid_k, 1);
            mc.ggml_metal_encoder_set_buffer(enc, bid_v, 2);
            mc.ggml_metal_encoder_set_buffer(enc, bid_src3, 3);
            mc.ggml_metal_encoder_set_buffer(enc, bid_pad, 4);

            impl.assert(ne12 == ne22, "ne12 == ne22");
            impl.assert(ne13 == ne23, "ne13 == ne23");

            mc.ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, @max(ne12, ne32), @max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            _ = concurrencyReset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        impl.assert(ne10 >= ne20, "ne10 >= ne20");

        // ne00 + 2*ncpsg*(nsg)
        // for each query, we load it as f16 in shared memory (ne00)
        // and store the soft_max values and the mask
        //
        // ne20*(nsg)
        // each simdgroup has a full f32 head vector in shared mem to accumulate results
        //
        // the C's FATTN_SMEM(nsg) macro -- a different one from the
        // non-vec arm above
        const fattnSmem = struct {
            fn f(ne00_: c_int, ne20_: c_int, ncpsg_: c_int, nqptg_: c_int, nsg_: i64) usize {
                const inner = @as(usize, impl.pad(@intCast(ne00_), 128)) +
                    @as(usize, @intCast(4 * ncpsg_)) +
                    2 * @as(usize, impl.pad(@intCast(ne20_), 128));
                return impl.pad(inner * @as(usize, @intCast(nsg_)) * @as(usize, @intCast(nqptg_)) * (@sizeOf(f32) / 2), 16);
            }
        }.f;

        var nsg: i64 = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        var nwg: i32 = 1;
        // the C's `if (false)` arm -- a single workgroup writing straight
        // to dst -- is disabled upstream, so only the else is live
        {
            nwg = 32;
            nsg = 1;
            while (2 * nwg * nsg * ncpsg < ne11 and nsg < 4) {
                nsg *= 2;
            }
        }

        // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
        if (fattnSmem(ne00, ne20, ncpsg, nqptg, nsg) > props_dev.max_theadgroup_memory_size) {
            cfg = tuning.baselineCfg(ne00, ne20);
            nqptg = cfg.Q; // = 1
        }

        const ns10: i32 = @intCast(nb11_attn / nb10_attn);
        const ns20: i32 = @intCast(nb21_attn / nb20_attn);

        var args: kargs.flash_attn_ext_vec = .{
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne11 = ne11,
            .ne_12_2 = ne12,
            .ne_12_3 = ne13,
            .ns10 = ns10,
            .nb11 = nb11_attn,
            .nb12 = nb12_attn,
            .nb13 = nb13_attn,
            .ns20 = ns20,
            .nb21 = nb21_attn,
            .nb22 = nb22_attn,
            .nb23 = nb23_attn,
            .ne31 = ne31,
            .ne32 = ne32,
            .ne33 = ne33,
            .nb31 = nb31,
            .nb32 = nb32,
            .nb33 = nb33,
            .ne1 = @intCast(op.ne[1]),
            .ne2 = @intCast(op.ne[2]),
            .ne3 = @intCast(op.ne[3]),
            .scale = scale,
            .max_bias = max_bias,
            .m0 = m0,
            .m1 = m1,
            .n_head_log2 = n_head_log2,
            .logit_softcap = logit_softcap,
        };

        const pipeline = library.ggml_metal_library_get_pipeline_flash_attn_ext_vec(
            lib,
            op,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            nqptg,
            cfg.NE,
            @intCast(nsg),
            nwg,
            use_kv_f16,
            ns10,
            ns20,
        );

        impl.assert(nsg * 32 <= maxThreads(pipeline), "nsg*32 <= max_theads_per_threadgroup");

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        mc.ggml_metal_encoder_set_buffer(enc, bid_k, 2);
        mc.ggml_metal_encoder_set_buffer(enc, bid_v, 3);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src3, 4);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src4, 5);

        const smem = fattnSmem(ne00, ne20, ncpsg, nqptg, nsg);

        impl.assert(smem <= props_dev.max_theadgroup_memory_size, "smem <= max_theadgroup_memory_size");

        if (nwg == 1) {
            impl.assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) == 0, "extra_tmp(op) == 0");

            // using 1 workgroup -> write the result directly into dst
            mc.ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(ne01 + nqptg - 1, nqptg),
                @divTrunc(ne02 + nhptg - 1, nhptg),
                ne03 * nwg,
                32,
                @intCast(nsg),
                1,
            );
        } else {
            // sanity checks
            impl.assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0, "extra_tmp(op) != 0");

            impl.assert(
                @as(i64, ne01) * ne02 * ne03 == op.ne[1] * op.ne[2] * op.ne[3],
                "ne01*ne02*ne03 == ne1*ne2*ne3",
            );
            impl.assert(
                @as(u64, @intCast(op.ne[1] * op.ne[2] * op.ne[3])) <= (@as(u64, 1) << 31),
                "(uint64_t)ne1*ne2*ne3 <= (1u << 31)",
            );

            // write the results from each workgroup into a temp buffer
            mc.ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            mc.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(ne01 + nqptg - 1, nqptg),
                @divTrunc(ne02 + nhptg - 1, nhptg),
                ne03 * nwg,
                32,
                @intCast(nsg),
                1,
            );

            // sync the 2 kernels
            _ = concurrencyReset(ctx);

            // reduce the results from the workgroups
            {
                const nrows: i32 = @intCast(op.ne[1] * op.ne[2] * op.ne[3]);

                var args0: kargs.flash_attn_ext_vec_reduce = .{
                    .nrows = nrows,
                };

                const pipeline0 = library.ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                mc.ggml_metal_encoder_set_pipeline(enc, pipeline0);
                setBytes(enc, &args0, 0);
                mc.ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
                mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

                mc.ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32 * nwg, 1, 1);
            }
        }
    }

    return 1;
}

/// Ports `ggml_metal_op_set` (ggml-metal-ops.cpp:1951 @c1d0e7a00).
///
/// Two `cpy` dispatches when not in-place: src0 into dst, then src1 into
/// a window of dst at `offs` with the `pnb*` strides. Note the second
/// `args` describes **src1**'s shape in the `ne0*` fields.
pub export fn ggml_metal_op_set(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const bid_src0 = getBufferId(s0);
    const bid_src1 = getBufferId(s1);
    var bid_dst = getBufferId(op);

    const pnb1: usize = @intCast(impl.getOpParamsI32(op, 0));
    const pnb2: usize = @intCast(impl.getOpParamsI32(op, 1));
    const pnb3: usize = @intCast(impl.getOpParamsI32(op, 2));
    const offs: usize = @intCast(impl.getOpParamsI32(op, 3));

    const inplace = impl.getOpParamsI32(op, 4) != 0;

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        const pipeline_cpy = library.ggml_metal_library_get_pipeline_cpy(lib, s0.type, op.type);

        var args_cpy: kargs.cpy = .{
            .nk0 = s0.ne[0],
            .ne00 = s0.ne[0],
            .ne01 = s0.ne[1],
            .ne02 = s0.ne[2],
            .ne03 = s0.ne[3],
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne0 = op.ne[0],
            .ne1 = op.ne[1],
            .ne2 = op.ne[2],
            .ne3 = op.ne[3],
            .nb0 = op.nb[0],
            .nb1 = op.nb[1],
            .nb2 = op.nb[2],
            .nb3 = op.nb[3],
        };

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline_cpy);
        setBytes(enc, &args_cpy, 0);
        mc.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

        const nth_cpy = @min(maxThreads(pipeline_cpy), @as(c_int, @intCast(s0.ne[0])));

        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @intCast(s0.ne[1]),
            @intCast(s0.ne[2]),
            @intCast(s0.ne[3]),
            nth_cpy,
            1,
            1,
        );

        _ = concurrencyReset(ctx);
    }

    const pipeline = library.ggml_metal_library_get_pipeline_cpy(lib, s1.type, op.type);

    const ne10: i64 = s1.ne[0];

    impl.assert(
        @rem(ne10, c.ggml_blck_size(s1.type)) == 0,
        "ne10 % ggml_blck_size(op->src[1]->type) == 0",
    );

    var nk0: i64 = ne10;
    if (c.ggml_is_quantized(s1.type)) {
        nk0 = @divTrunc(ne10, 16);
    } else if (c.ggml_is_quantized(op.type)) {
        nk0 = @divTrunc(ne10, c.ggml_blck_size(op.type));
    }

    var nth: c_int = @intCast(@min(nk0 * s1.ne[1], 256));

    // when rows are small, we can batch them together in a single threadgroup
    var nrptg: c_int = 1;

    // TODO: relax this constraint in the future
    if (c.ggml_blck_size(s1.type) == 1 and c.ggml_blck_size(op.type) == 1) {
        if (nth > @as(c_int, @intCast(nk0))) {
            nrptg = @divTrunc(nth + @as(c_int, @intCast(nk0)) - 1, @as(c_int, @intCast(nk0)));
            nth = @intCast(nk0);

            if (nrptg * nth > 256) {
                nrptg -= 1;
            }
        }
    }

    nth = @min(nth, @as(c_int, @intCast(nk0)));

    var args: kargs.cpy = .{
        .nk0 = nk0,
        .ne00 = s1.ne[0],
        .ne01 = s1.ne[1],
        .ne02 = s1.ne[2],
        .ne03 = s1.ne[3],
        .nb00 = s1.nb[0],
        .nb01 = s1.nb[1],
        .nb02 = s1.nb[2],
        .nb03 = s1.nb[3],
        .ne0 = s1.ne[0],
        .ne1 = s1.ne[1],
        .ne2 = s1.ne[2],
        .ne3 = s1.ne[3],
        .nb0 = @intCast(c.ggml_element_size(op)),
        .nb1 = pnb1,
        .nb2 = pnb2,
        .nb3 = pnb3,
    };

    const nw0: c_int = if (nrptg == 1)
        @divTrunc(@as(c_int, @intCast(nk0)) + nth - 1, nth)
    else
        1;

    bid_dst.offs += offs;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    mc.ggml_metal_encoder_set_buffer(enc, bid_src1, 1);
    mc.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    const ne11: c_int = @intCast(s1.ne[1]);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @divTrunc(nw0 * (ne11 + nrptg - 1), nrptg),
        @intCast(s1.ne[2]),
        @intCast(s1.ne[3]),
        nth,
        nrptg,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_cpy` (ggml-metal-ops.cpp:2079 @c1d0e7a00).
pub export fn ggml_metal_op_cpy(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const pipeline = library.ggml_metal_library_get_pipeline_cpy(lib, s0.type, op.type);

    const ne00: i64 = s0.ne[0];

    impl.assert(
        @rem(ne00, c.ggml_blck_size(s0.type)) == 0,
        "ne00 % ggml_blck_size(op->src[0]->type) == 0",
    );

    var nk0: i64 = ne00;
    if (c.ggml_is_quantized(s0.type)) {
        nk0 = @divTrunc(ne00, 16);
    } else if (c.ggml_is_quantized(op.type)) {
        nk0 = @divTrunc(ne00, c.ggml_blck_size(op.type));
    }

    var nth: c_int = @intCast(@min(nk0 * s0.ne[1], 256));

    // when rows are small, we can batch them together in a single threadgroup
    var nrptg: c_int = 1;

    // TODO: relax this constraint in the future
    if (c.ggml_blck_size(s0.type) == 1 and c.ggml_blck_size(op.type) == 1) {
        if (nth > @as(c_int, @intCast(nk0))) {
            nrptg = @divTrunc(nth + @as(c_int, @intCast(nk0)) - 1, @as(c_int, @intCast(nk0)));
            nth = @intCast(nk0);

            if (nrptg * nth > 256) {
                nrptg -= 1;
            }
        }
    }

    nth = @min(nth, @as(c_int, @intCast(nk0)));

    var args: kargs.cpy = .{
        .nk0 = nk0,
        .ne00 = ne00,
        .ne01 = s0.ne[1],
        .ne02 = s0.ne[2],
        .ne03 = s0.ne[3],
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne0 = op.ne[0],
        .ne1 = op.ne[1],
        .ne2 = op.ne[2],
        .ne3 = op.ne[3],
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
    };

    const nw0: c_int = if (nrptg == 1)
        @divTrunc(@as(c_int, @intCast(nk0)) + nth - 1, nth)
    else
        1;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    const ne01: c_int = @intCast(s0.ne[1]);

    mc.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @divTrunc(nw0 * (ne01 + nrptg - 1), nrptg),
        @intCast(s0.ne[2]),
        @intCast(s0.ne[3]),
        nth,
        nrptg,
        1,
    );

    return 1;
}

/// Ports `ggml_metal_op_pool_1d` (ggml-metal-ops.cpp:2152 @c1d0e7a00).
pub export fn ggml_metal_op_pool_1d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);

    const op_pool: c.enum_ggml_op_pool = @intCast(impl.getOpParamsI32(op, 0));

    const k0 = impl.getOpParamsI32(op, 1);
    const s0 = impl.getOpParamsI32(op, 2);
    const p0 = impl.getOpParamsI32(op, 3);

    const IW: i64 = src0.ne[0];
    const OW: i64 = op.ne[0];

    const np: i64 = c.ggml_nelements(op);

    var args_pool_1d: kargs.pool_1d = .{
        .k0 = k0,
        .s0 = s0,
        .p0 = p0,
        .IW = IW,
        .OW = OW,
        .np = np,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(np)));
    const ntg = @divTrunc(np + nth - 1, nth);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args_pool_1d, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(ntg), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_pool_2d` (ggml-metal-ops.cpp:2240 @c1d0e7a00).
pub export fn ggml_metal_op_pool_2d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);

    const op_pool: c.enum_ggml_op_pool = @intCast(impl.getOpParamsI32(op, 0));

    const k0 = impl.getOpParamsI32(op, 1);
    const k1 = impl.getOpParamsI32(op, 2);
    const s0 = impl.getOpParamsI32(op, 3);
    const s1 = impl.getOpParamsI32(op, 4);
    const p0 = impl.getOpParamsI32(op, 5);
    const p1 = impl.getOpParamsI32(op, 6);

    const IH: i64 = src0.ne[1];
    const IW: i64 = src0.ne[0];

    const N: i64 = op.ne[3];
    const OC: i64 = op.ne[2];
    const OH: i64 = op.ne[1];
    const OW: i64 = op.ne[0];

    const np: i64 = N * OC * OH * OW;

    var args_pool_2d: kargs.pool_2d = .{
        .k0 = k0,
        .k1 = k1,
        .s0 = s0,
        .s1 = s1,
        .p0 = p0,
        .p1 = p1,
        .IH = IH,
        .IW = IW,
        .OH = OH,
        .OW = OW,
        .np = np,
    };

    const pipeline = library.ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(np)));
    const ntg = @divTrunc(np + nth - 1, nth);

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args_pool_2d, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(ntg), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_argmax` (ggml-metal-ops.cpp:4931 @c1d0e7a00).
pub export fn ggml_metal_op_argmax(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ne00: i32 = @intCast(s0.ne[0]);
    const ne01: i32 = @intCast(s0.ne[1]);
    const ne02: i32 = @intCast(s0.ne[2]);
    const ne03: i32 = @intCast(s0.ne[3]);

    var args: kargs.argmax = .{
        .ne00 = ne00,
        .nb01 = s0.nb[1],
    };

    const pipeline = library.ggml_metal_library_get_pipeline_argmax(lib, op);

    const nrows: i64 = c.ggml_nrows(s0);

    var nth: i32 = 32; // SIMD width
    while (nth < ne00 and nth * ne01 * ne02 * ne03 < 256) {
        nth *= 2;
    }

    const smem = pipeline.smem;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(nrows), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_opt_step_adamw` (ggml-metal-ops.cpp:5240
/// @c1d0e7a00).
pub export fn ggml_metal_op_opt_step_adamw(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const pipeline = library.ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const np: i64 = c.ggml_nelements(s0);
    var args: kargs.opt_step_adamw = .{
        .np = np,
    };

    var ida: c_int = 0;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, ida);
    ida += 1;
    // The C writes out five `set_buffer` calls for src[0..4].
    for (0..5) |j| {
        setBuffer(enc, src(op, j), ida);
        ida += 1;
    }

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(s0.ne[0])));
    const n = @divTrunc(np + nth - 1, nth);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_opt_step_sgd` (ggml-metal-ops.cpp:5276 @c1d0e7a00).
pub export fn ggml_metal_op_opt_step_sgd(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const pipeline = library.ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const np: i64 = c.ggml_nelements(s0);
    var args: kargs.opt_step_sgd = .{
        .np = np,
    };

    var ida: c_int = 0;

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, ida);
    ida += 1;
    setBuffer(enc, s0, ida);
    ida += 1;
    setBuffer(enc, src(op, 1), ida);
    ida += 1;
    setBuffer(enc, src(op, 2), ida);
    ida += 1;

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(s0.ne[0])));
    const n = @divTrunc(np + nth - 1, nth);

    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_count_equal` (ggml-metal-ops.cpp:5310 @c1d0e7a00).
///
/// Two dispatches with a concurrency reset between them: the destination
/// is zeroed first, then atomically accumulated into, so the second pass
/// must not overlap the first.
pub export fn ggml_metal_op_count_equal(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    {
        var args: kargs.memset = .{ .val = 0 };

        const pipeline = library.ggml_metal_library_get_pipeline_memset(lib, op);

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, op, 1);

        mc.ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    _ = concurrencyReset(ctx);

    {
        var args: kargs.count_equal = .{
            .ne00 = @intCast(s0.ne[0]),
            .ne01 = @intCast(s0.ne[1]),
            .ne02 = @intCast(s0.ne[2]),
            .ne03 = @intCast(s0.ne[3]),
            .nb00 = s0.nb[0],
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .nb10 = s1.nb[0],
            .nb11 = s1.nb[1],
            .nb12 = s1.nb[2],
            .nb13 = s1.nb[3],
        };

        const pipeline = library.ggml_metal_library_get_pipeline_count_equal(lib, op);

        const smem = pipeline.smem;

        const nth = 32 * pipeline.nsg;

        impl.assert(nth <= maxThreads(pipeline), "nth <= max_theads_per_threadgroup");

        mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        mc.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        mc.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @intCast(s0.ne[1]),
            @intCast(s0.ne[2]),
            @intCast(s0.ne[3]),
            nth,
            1,
            1,
        );
    }

    return 1;
}

/// Ports `ggml_metal_fwht_supported_size` (ggml-metal-ops.cpp:2201
/// @c1d0e7a00).
///
/// supported FWHT sizes, must stay in sync with the
/// kernel_fwht_f32_<N> templates in ggml-metal.metal
fn fwhtSupportedSize(n: i64) bool {
    return n == 64 or n == 128 or n == 256 or n == 512;
}

/// Ports `ggml_metal_op_fwht` (ggml-metal-ops.cpp:2205 @c1d0e7a00).
///
/// Not reached from the dispatch switch: `ggml_metal_op_mul_mat` calls it
/// at its line 2315. It is an export of this translation unit all the
/// same. Note it reads `src[1]`, not `src[0]`.
pub export fn ggml_metal_op_fwht(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src1 = src(op, 1);

    const n: i64 = src1.ne[0];
    const nrows: i64 = c.ggml_nrows(src1);

    var args: kargs.fwht = .{
        .nrows = @intCast(nrows),
    };

    const pipeline = library.ggml_metal_library_get_pipeline_fwht(lib, @intCast(n));

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src1, 1);
    setBuffer(enc, op, 2);

    const th_max = maxThreads(pipeline);
    const simd_size: c_int = 32;

    var sg_per_tg: c_int = 2;
    sg_per_tg = @min(sg_per_tg, @divTrunc(th_max, simd_size));
    sg_per_tg = @max(sg_per_tg, 1);

    const n_tg = @divTrunc(nrows + @as(i64, sg_per_tg) - 1, @as(i64, sg_per_tg));
    mc.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n_tg), 1, 1, 32 * sg_per_tg, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_can_fuse_snake` (ggml-metal-ops.cpp:3553
/// @c1d0e7a00).
///
/// Recognises the `mul -> sin -> sqr -> mul -> add` chain that the fused
/// `snake` kernel implements. Every clause is the C's, in order.
fn canFuseSnake(ctx: *Op, idx: c_int) bool {
    const snake_ops = [5]c.enum_ggml_op{
        c.GGML_OP_MUL, c.GGML_OP_SIN, c.GGML_OP_SQR, c.GGML_OP_MUL, c.GGML_OP_ADD,
    };

    if (ctx.node(idx).op != c.GGML_OP_MUL or !ctx.canFuse(idx, &snake_ops, 5)) {
        return false;
    }

    const mul0 = ctx.node(idx + 0);
    const sin_node = ctx.node(idx + 1);
    const sqr = ctx.node(idx + 2);
    const mul1 = ctx.node(idx + 3);
    const add = ctx.node(idx + 4);

    // x carries the full activation shape, a is the broadcast operand
    const x = if (c.ggml_are_same_shape(mul0, src(mul0, 0))) src(mul0, 0) else src(mul0, 1);
    const a = if (x == src(mul0, 0)) src(mul0, 1) else src(mul0, 0);

    // mul1 reads sqr and inv_b in either operand order
    const inv_b = if (src(mul1, 0) == sqr) src(mul1, 1) else src(mul1, 0);

    // closure check: the trailing add reads the same x as the leading mul
    const x_in_add = if (src(add, 0) == mul1) src(add, 1) else src(add, 0);

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

/// Ports `ggml_metal_op_snake_fused` (ggml-metal-ops.cpp:4546 @c1d0e7a00).
///
/// Dispatch the fused snake kernel from the matched mul -> sin -> sqr ->
/// mul -> add chain. `idx` points at the leading mul. The caller has
/// validated the chain. Returns **5**: it consumes the whole chain.
pub export fn ggml_metal_op_snake_fused(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const lib = ctx.lib;
    const enc = ctx.enc;

    const mul0 = ctx.node(idx + 0);
    const sqr = ctx.node(idx + 2);
    const mul1 = ctx.node(idx + 3);
    const add = ctx.node(idx + 4);

    const x = if (c.ggml_are_same_shape(mul0, src(mul0, 0))) src(mul0, 0) else src(mul0, 1);
    const a = if (x == src(mul0, 0)) src(mul0, 1) else src(mul0, 0);
    const inv_b = if (src(mul1, 0) == sqr) src(mul1, 1) else src(mul1, 0);

    const T: c_int = @intCast(x.ne[0]);
    const C: c_int = @intCast(x.ne[1]);
    const total = T * C;

    // the encode loop pre-checked the leading mul only, check the rest of the chain
    var i: c_int = 1;
    while (i < 5) : (i += 1) {
        if (!concurrencyCheck(ctx, ctx.node(idx + i))) {
            _ = concurrencyReset(ctx);

            break;
        }
    }

    const pipeline = library.ggml_metal_library_get_pipeline_snake(lib, x.type);

    var args: kargs.snake = .{
        .T = T,
        .C = C,
    };

    mc.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, x, 1);
    setBuffer(enc, a, 2);
    setBuffer(enc, inv_b, 3);
    setBuffer(enc, add, 4);

    const nth: c_int = 256;
    const ntg = @divTrunc(total + nth - 1, nth);
    mc.ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 5;
}

/// Ports `ggml_metal_op_mul_mat_id_extra_tpe` (ggml-metal-ops.cpp:2544
/// @c1d0e7a00).
///
/// Scratch for the tokens-per-expert table, allocated after `dst` by
/// `backend.zig`'s `get_alloc_size`.
pub export fn ggml_metal_op_mul_mat_id_extra_tpe(op: *const Tensor) callconv(.c) usize {
    impl.assert(op.op == c.GGML_OP_MUL_MAT_ID, "op->op == GGML_OP_MUL_MAT_ID");

    const ne02: i64 = src(op, 0).ne[2]; // n_expert

    return c.ggml_type_size(c.GGML_TYPE_I32) * @as(usize, @intCast(ne02));
}

/// Ports `ggml_metal_op_mul_mat_id_extra_ids` (ggml-metal-ops.cpp:2552
/// @c1d0e7a00).
pub export fn ggml_metal_op_mul_mat_id_extra_ids(op: *const Tensor) callconv(.c) usize {
    impl.assert(op.op == c.GGML_OP_MUL_MAT_ID, "op->op == GGML_OP_MUL_MAT_ID");

    const ne02: i64 = src(op, 0).ne[2]; // n_expert
    const ne21: i64 = src(op, 2).ne[1]; // n_token

    return c.ggml_type_size(c.GGML_TYPE_I32) * @as(usize, @intCast(ne02)) * @as(usize, @intCast(ne21));
}

/// Ports `ggml_metal_op_flash_attn_ext_use_vec` (ggml-metal-ops.cpp:2795
/// @c1d0e7a00).
pub export fn ggml_metal_op_flash_attn_ext_use_vec(op: *const Tensor) callconv(.c) bool {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const s0 = src(op, 0);

    const ne00: i64 = s0.ne[0]; // head size
    const ne01: i64 = s0.ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) and (@rem(ne00, 32) == 0);
}

/// Ports `ggml_metal_op_flash_attn_ext_use_kv_f16`
/// (ggml-metal-ops.cpp:2807 @c1d0e7a00).
///
/// ref: https://github.com/ggml-org/llama.cpp/pull/27390
/// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
fn useKvF16(op: *const Tensor) bool {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
    // TODO: tune per device
    if (src(op, 0).ne[1] < 32) {
        return false;
    }

    return switch (src(op, 1).type) {
        c.GGML_TYPE_Q4_0,
        c.GGML_TYPE_Q4_1,
        c.GGML_TYPE_Q5_0,
        c.GGML_TYPE_Q5_1,
        c.GGML_TYPE_Q8_0,
        => true,
        else => false,
    };
}

/// Ports `ggml_metal_op_flash_attn_ext_v_is_view_of_k`
/// (ggml-metal-ops.cpp:2832 @c1d0e7a00).
///
/// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
/// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
/// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
fn vIsViewOfK(op: *const Tensor) bool {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const K = src(op, 1);
    const V = src(op, 2);

    if (V.view_src == null) {
        return false;
    }
    const v_view = impl.one(Tensor, V.view_src);

    return v_view == K or
        (K.view_src != null and v_view == impl.one(Tensor, K.view_src) and V.view_offs == K.view_offs);
}

/// Ports `ggml_metal_op_flash_attn_ext_kv_f16_k_size`
/// (ggml-metal-ops.cpp:2842 @c1d0e7a00).
///
/// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
fn kvF16KSize(op: *const Tensor) usize {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const s1 = src(op, 1);

    return impl.pad(@sizeOf(c.ggml_fp16_t) * @as(usize, @intCast(s1.ne[0] * s1.ne[1] * s1.ne[2] * s1.ne[3])), 16);
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_pad`
/// (ggml-metal-ops.cpp:2850 @c1d0e7a00).
pub export fn ggml_metal_op_flash_attn_ext_extra_pad(op: *const Tensor) callconv(.c) usize {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const s1 = src(op, 1);
    const s2 = src(op, 2);
    const s3 = srcOpt(op, 3);

    var res: usize = 0;

    const has_mask = op.src[3] != null;
    const use_kv_f16 = useKvF16(op);

    // when the KV is dequantized to F16, the pad kernel copies the tail chunk from the F16 scratch buffer
    // note: when V is a view of K, the dequantized V is read from the dequantized K with K's row stride
    const v_is_view_of_k = use_kv_f16 and vIsViewOfK(op);
    var nb11_pad: u64 = s1.nb[1];
    var nb21_pad: u64 = s2.nb[1];

    if (use_kv_f16) {
        nb11_pad = @sizeOf(c.ggml_fp16_t) * @as(u64, @intCast(s1.ne[0]));
        nb21_pad = @sizeOf(c.ggml_fp16_t) * @as(u64, @intCast(if (v_is_view_of_k) s1.ne[0] else s2.ne[0]));
    }

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    //
    // Both are `impl_c.zig` constants, so the C's run-time `GGML_ASSERT`
    // becomes a compile-time one here.
    comptime {
        if (fc.OP_FLASH_ATTN_EXT_NCPSG < fc.OP_FLASH_ATTN_EXT_VEC_NCPSG) {
            @compileError("OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG");
        }
    }

    // The C's `if (ggml_metal_op_flash_attn_ext_use_vec(op))` is
    // commented out to `if (false)`, so only the `else` arm is live --
    // "always reserve the padding space to avoid graph reallocations".
    // `has_kvpad` is likewise forced true there.
    {
        const mask_term: u64 = if (has_mask) blk: {
            const m = s3.?;
            break :blk @as(u64, @intCast(c.ggml_type_size(c.GGML_TYPE_F16))) *
                @as(u64, @intCast(m.ne[1] * m.ne[2] * m.ne[3]));
        } else 0;

        res += @as(usize, @intCast(fc.OP_FLASH_ATTN_EXT_NCPSG)) * @as(usize, @intCast(nb11_pad *
            @as(u64, @intCast(s1.ne[2] * s1.ne[3])) +
            nb21_pad * @as(u64, @intCast(s2.ne[2] * s2.ne[3])) +
            mask_term));
    }

    return res;
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_blk`
/// (ggml-metal-ops.cpp:2908 @c1d0e7a00).
pub export fn ggml_metal_op_flash_attn_ext_extra_blk(op: *const Tensor) callconv(.c) usize {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const s0 = src(op, 0);

    var res: usize = 0;

    const has_mask = op.src[3] != null;

    if (!has_mask) {
        return res;
    }

    const s3 = src(op, 3);

    const is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    // (the C's early return for `is_vec` is commented out)

    const nqptg: c_int = if (is_vec) fc.OP_FLASH_ATTN_EXT_VEC_NQPSG else fc.OP_FLASH_ATTN_EXT_NQPSG;
    const ncpsg: c_int = if (is_vec) fc.OP_FLASH_ATTN_EXT_VEC_NCPSG else fc.OP_FLASH_ATTN_EXT_NCPSG;

    const ne1: i64 = @divTrunc(s0.ne[1] + nqptg - 1, nqptg);
    const ne0: i64 = @divTrunc(s3.ne[0] + ncpsg - 1, ncpsg);

    res += impl.pad(@as(usize, @intCast(c.ggml_type_size(c.GGML_TYPE_I8))) *
        @as(usize, @intCast(ne0 * ne1 * s3.ne[2] * s3.ne[3])), 32);

    return res;
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_tmp`
/// (ggml-metal-ops.cpp:2947 @c1d0e7a00).
pub export fn ggml_metal_op_flash_attn_ext_extra_tmp(op: *const Tensor) callconv(.c) usize {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const s0 = src(op, 0);
    const s2 = src(op, 2);

    var res: usize = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    // (the C's `use_vec` test is commented out to `if (true)`)
    {
        const nwg: i64 = 32;
        const ne01_max: i64 = @min(s0.ne[1], 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += @as(usize, @intCast(c.ggml_type_size(c.GGML_TYPE_F32))) *
            @as(usize, @intCast(ne01_max * s0.ne[2] * s0.ne[3] * nwg * (s2.ne[0] + 2)));
    }

    return res;
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_kv_f16`
/// (ggml-metal-ops.cpp:2976 @c1d0e7a00).
pub export fn ggml_metal_op_flash_attn_ext_extra_kv_f16(op: *const Tensor) callconv(.c) usize {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    // note: always reserve the temp buffer to avoid graph reallocations
    // (the C's `!use_kv_f16` early return is commented out)

    const s2 = src(op, 2);

    const k_size = kvF16KSize(op);

    // when V is a view of K, the dequantized V is a view of the dequantized K
    const v_is_view_of_k = vIsViewOfK(op);
    if (v_is_view_of_k) {
        return k_size;
    }

    const v_size = impl.pad(@sizeOf(c.ggml_fp16_t) *
        @as(usize, @intCast(s2.ne[0] * s2.ne[1] * s2.ne[2] * s2.ne[3])), 16);

    return k_size + v_size;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
