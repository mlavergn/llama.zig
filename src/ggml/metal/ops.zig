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
//! # Status: incomplete, and deliberately not in the barrel
//!
//! `module.zig` does **not** import this file yet. A translation unit is
//! ported all-or-nothing and swapped only when complete, because the
//! linker's missing-symbol errors are the completeness check and two
//! definitions of a symbol in one archive is not a link error — the
//! linker picks one silently. Until the 66th export is written, wiring
//! this in would both break the build and, worse, could appear to work.
//!
//! Done: the context and its lifecycle, the buffer-id and concurrency
//! helpers, `ggml_metal_op_encode`, and `ggml_metal_op_encode_impl` with
//! its full dispatch switch — verified to cover exactly the C's 73 op
//! labels, no more and no fewer.
//!
//! Not done: the **53 per-op encoder bodies**, which are stubs that abort,
//! and **9 further exports** the switch does not reach —
//! `ggml_metal_op_fwht`, `_snake_fused`, `_flash_attn_ext_use_vec`, the
//! four `_flash_attn_ext_extra_*` and the two `_mul_mat_id_extra_*`. Six
//! of those nine are called by `backend.zig`, which is already swapped
//! in and currently reaches the C++ ones.
//!
//! # Why the stubs are not exported
//!
//! They are private Zig functions, not `pub export fn ggml_metal_op_*`,
//! on purpose. Exporting them would make `scripts/port-coverage` able to
//! read **66 / 66 symbols (100%)** for a file whose every kernel dispatch
//! aborts. That is precisely what `repack.cpp` did: with its dispatch
//! stubbed, coverage read `36 / 36 (100%)` and `node-diff` failed on the
//! first `MUL_MAT`. The count must not be reachable until the bodies
//! are.
//!
//! `encoders_implemented` below is the comptime guard for the swap, in
//! the same shape as `repack.dispatch_implemented` — which
//! `scripts/cluster-check` asserts so that a symbol count cannot stand
//! alone again.

const std = @import("std");

const impl = @import("../impl.zig");
const c = impl.c;

const mc = @import("device_c.zig");
const fc = @import("impl_c.zig");
const kargs = @import("kargs.zig");
const library = @import("library.zig");
const common = @import("common.zig");
const graph = @import("../graph.zig");

const Tensor = c.ggml_tensor;

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
        c.GGML_OP_CONCAT => encodeConcat(ctx, idx),
        c.GGML_OP_ADD, c.GGML_OP_SUB, c.GGML_OP_MUL, c.GGML_OP_DIV => encodeBin(ctx, idx),
        c.GGML_OP_ADD_ID => encodeAddId(ctx, idx),
        c.GGML_OP_REPEAT => encodeRepeat(ctx, idx),
        c.GGML_OP_ACC => encodeAcc(ctx, idx),
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
        => encodeUnary(ctx, idx),
        c.GGML_OP_SILU_BACK => encodeSiluBack(ctx, idx),
        c.GGML_OP_GLU => encodeGlu(ctx, idx),
        c.GGML_OP_SUM => encodeSum(ctx, idx),
        c.GGML_OP_SUM_ROWS, c.GGML_OP_MEAN => encodeSumRows(ctx, idx),
        c.GGML_OP_CUMSUM => encodeCumsum(ctx, idx),
        c.GGML_OP_LIGHTNING_INDEXER => encodeLightningIndexer(ctx, idx),
        c.GGML_OP_DSV4_HC_COMB,
        c.GGML_OP_DSV4_HC_PRE,
        c.GGML_OP_DSV4_HC_POST,
        => encodeDsv4Hc(ctx, idx),
        c.GGML_OP_SOFT_MAX => encodeSoftMax(ctx, idx),
        c.GGML_OP_SSM_CONV => encodeSsmConv(ctx, idx),
        c.GGML_OP_SSM_SCAN => encodeSsmScan(ctx, idx),
        c.GGML_OP_RWKV_WKV6, c.GGML_OP_RWKV_WKV7 => encodeRwkv(ctx, idx),
        c.GGML_OP_GATED_DELTA_NET => encodeGatedDeltaNet(ctx, idx),
        c.GGML_OP_SOLVE_TRI => encodeSolveTri(ctx, idx),
        c.GGML_OP_MUL_MAT => encodeMulMat(ctx, idx),
        c.GGML_OP_MUL_MAT_ID => encodeMulMatId(ctx, idx),
        c.GGML_OP_GET_ROWS => encodeGetRows(ctx, idx),
        c.GGML_OP_SET_ROWS => encodeSetRows(ctx, idx),
        c.GGML_OP_DIAG => encodeDiag(ctx, idx),
        c.GGML_OP_L2_NORM => encodeL2Norm(ctx, idx),
        c.GGML_OP_GROUP_NORM => encodeGroupNorm(ctx, idx),
        c.GGML_OP_NORM, c.GGML_OP_RMS_NORM => encodeNorm(ctx, idx),
        c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => encodeRope(ctx, idx),
        c.GGML_OP_IM2COL => encodeIm2col(ctx, idx),
        c.GGML_OP_CONV_2D => encodeConv2d(ctx, idx),
        c.GGML_OP_CONV_2D_DW => encodeConv2dDw(ctx, idx),
        c.GGML_OP_CONV_TRANSPOSE_1D => encodeConvTranspose1d(ctx, idx),
        c.GGML_OP_CONV_TRANSPOSE_2D => encodeConvTranspose2d(ctx, idx),
        c.GGML_OP_COL2IM_1D => encodeCol2im1d(ctx, idx),
        c.GGML_OP_CONV_3D => encodeConv3d(ctx, idx),
        c.GGML_OP_UPSCALE => encodeUpscale(ctx, idx),
        c.GGML_OP_PAD => encodePad(ctx, idx),
        c.GGML_OP_PAD_REFLECT_1D => encodePadReflect1d(ctx, idx),
        c.GGML_OP_ROLL => encodeRoll(ctx, idx),
        c.GGML_OP_ARANGE => encodeArange(ctx, idx),
        c.GGML_OP_TIMESTEP_EMBEDDING => encodeTimestepEmbedding(ctx, idx),
        c.GGML_OP_ARGSORT => encodeArgsort(ctx, idx),
        c.GGML_OP_TOP_K => encodeTopK(ctx, idx),
        c.GGML_OP_TRI => encodeTri(ctx, idx),
        c.GGML_OP_FLASH_ATTN_EXT => encodeFlashAttnExt(ctx, idx),
        c.GGML_OP_SET => encodeSet(ctx, idx),
        c.GGML_OP_DUP, c.GGML_OP_CPY, c.GGML_OP_CONT => encodeCpy(ctx, idx),
        c.GGML_OP_POOL_1D => encodePool1d(ctx, idx),
        c.GGML_OP_POOL_2D => encodePool2d(ctx, idx),
        c.GGML_OP_ARGMAX => encodeArgmax(ctx, idx),
        c.GGML_OP_OPT_STEP_ADAMW => encodeOptStepAdamw(ctx, idx),
        c.GGML_OP_OPT_STEP_SGD => encodeOptStepSgd(ctx, idx),
        c.GGML_OP_COUNT_EQUAL => encodeCountEqual(ctx, idx),
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
pub const encoders_implemented = false;

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

fn encodeConcat(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("concat");
}

fn encodeBin(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("bin");
}

fn encodeAddId(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("add_id");
}

fn encodeRepeat(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("repeat");
}

fn encodeAcc(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("acc");
}

fn encodeUnary(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("unary");
}

fn encodeSiluBack(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("silu_back");
}

fn encodeGlu(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("glu");
}

fn encodeSum(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("sum");
}

fn encodeSumRows(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("sum_rows");
}

fn encodeCumsum(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("cumsum");
}

fn encodeLightningIndexer(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("lightning_indexer");
}

fn encodeDsv4Hc(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("dsv4_hc");
}

fn encodeSoftMax(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("soft_max");
}

fn encodeSsmConv(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("ssm_conv");
}

fn encodeSsmScan(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("ssm_scan");
}

fn encodeRwkv(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("rwkv");
}

fn encodeGatedDeltaNet(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("gated_delta_net");
}

fn encodeSolveTri(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("solve_tri");
}

fn encodeMulMat(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("mul_mat");
}

fn encodeMulMatId(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("mul_mat_id");
}

fn encodeGetRows(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("get_rows");
}

fn encodeSetRows(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("set_rows");
}

fn encodeDiag(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("diag");
}

fn encodeL2Norm(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("l2_norm");
}

fn encodeGroupNorm(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("group_norm");
}

fn encodeNorm(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("norm");
}

fn encodeRope(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("rope");
}

fn encodeIm2col(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("im2col");
}

fn encodeConv2d(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("conv_2d");
}

fn encodeConv2dDw(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("conv_2d_dw");
}

fn encodeConvTranspose1d(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("conv_transpose_1d");
}

fn encodeConvTranspose2d(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("conv_transpose_2d");
}

fn encodeCol2im1d(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("col2im_1d");
}

fn encodeConv3d(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("conv_3d");
}

fn encodeUpscale(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("upscale");
}

fn encodePad(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("pad");
}

fn encodePadReflect1d(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("pad_reflect_1d");
}

fn encodeRoll(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("roll");
}

fn encodeArange(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("arange");
}

fn encodeTimestepEmbedding(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("timestep_embedding");
}

fn encodeArgsort(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("argsort");
}

fn encodeTopK(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("top_k");
}

fn encodeTri(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("tri");
}

fn encodeFlashAttnExt(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("flash_attn_ext");
}

fn encodeSet(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("set");
}

fn encodeCpy(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("cpy");
}

fn encodePool1d(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("pool_1d");
}

fn encodePool2d(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("pool_2d");
}

fn encodeArgmax(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("argmax");
}

fn encodeOptStepAdamw(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("opt_step_adamw");
}

fn encodeOptStepSgd(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("opt_step_sgd");
}

fn encodeCountEqual(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("count_equal");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
