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

/// Ports `ggml_metal_op_concat` (ggml-metal-ops.cpp:540 @c1d0e7a00).
fn encodeConcat(ctx: *Op, idx: c_int) c_int {
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

/// Ports `ggml_metal_op_repeat` (ggml-metal-ops.cpp:609 @c1d0e7a00).
fn encodeRepeat(ctx: *Op, idx: c_int) c_int {
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
fn encodeAcc(ctx: *Op, idx: c_int) c_int {
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
fn encodeUnary(ctx: *Op, idx: c_int) c_int {
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
fn encodeSiluBack(ctx: *Op, idx: c_int) c_int {
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
fn encodeGlu(ctx: *Op, idx: c_int) c_int {
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
fn encodeSum(ctx: *Op, idx: c_int) c_int {
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
fn encodeSumRows(ctx: *Op, idx: c_int) c_int {
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
fn encodeCumsum(ctx: *Op, idx: c_int) c_int {
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
fn encodeLightningIndexer(ctx: *Op, idx: c_int) c_int {
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

fn encodeDsv4Hc(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("dsv4_hc");
}

/// Ports `ggml_metal_op_soft_max` (ggml-metal-ops.cpp:1512 @c1d0e7a00).
///
/// `src[1]` (mask) and `src[2]` (sinks) are both optional; when absent
/// the C binds `src[0]` in their slot rather than a null buffer.
fn encodeSoftMax(ctx: *Op, idx: c_int) c_int {
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
fn encodeSsmConv(ctx: *Op, idx: c_int) c_int {
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
fn encodeSsmScan(ctx: *Op, idx: c_int) c_int {
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
fn encodeRwkv(ctx: *Op, idx: c_int) c_int {
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
fn encodeGatedDeltaNet(ctx: *Op, idx: c_int) c_int {
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
fn encodeSolveTri(ctx: *Op, idx: c_int) c_int {
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

/// Ports `ggml_metal_op_get_rows` (ggml-metal-ops.cpp:1166 @c1d0e7a00).
fn encodeGetRows(ctx: *Op, idx: c_int) c_int {
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
fn encodeSetRows(ctx: *Op, idx: c_int) c_int {
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
fn encodeDiag(ctx: *Op, idx: c_int) c_int {
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
fn encodeL2Norm(ctx: *Op, idx: c_int) c_int {
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
fn encodeGroupNorm(ctx: *Op, idx: c_int) c_int {
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
fn encodeNorm(ctx: *Op, idx: c_int) c_int {
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
fn encodeRope(ctx: *Op, idx: c_int) c_int {
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
fn encodeIm2col(ctx: *Op, idx: c_int) c_int {
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
fn encodeConv2d(ctx: *Op, idx: c_int) c_int {
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
fn encodeConv2dDw(ctx: *Op, idx: c_int) c_int {
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
fn encodeConvTranspose1d(ctx: *Op, idx: c_int) c_int {
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
fn encodeConvTranspose2d(ctx: *Op, idx: c_int) c_int {
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
fn encodeCol2im1d(ctx: *Op, idx: c_int) c_int {
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
fn encodeConv3d(ctx: *Op, idx: c_int) c_int {
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
fn encodeUpscale(ctx: *Op, idx: c_int) c_int {
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
fn encodePad(ctx: *Op, idx: c_int) c_int {
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
fn encodePadReflect1d(ctx: *Op, idx: c_int) c_int {
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
fn encodeRoll(ctx: *Op, idx: c_int) c_int {
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
fn encodeArange(ctx: *Op, idx: c_int) c_int {
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
fn encodeTimestepEmbedding(ctx: *Op, idx: c_int) c_int {
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
fn encodeArgsort(ctx: *Op, idx: c_int) c_int {
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

fn encodeTopK(ctx: *Op, idx: c_int) c_int {
    _ = ctx;
    _ = idx;
    todo.notPorted("top_k");
}

/// Ports `ggml_metal_op_tri` (ggml-metal-ops.cpp:5189 @c1d0e7a00).
fn encodeTri(ctx: *Op, idx: c_int) c_int {
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

/// Ports `ggml_metal_op_cpy` (ggml-metal-ops.cpp:2079 @c1d0e7a00).
fn encodeCpy(ctx: *Op, idx: c_int) c_int {
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
fn encodePool1d(ctx: *Op, idx: c_int) c_int {
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
fn encodePool2d(ctx: *Op, idx: c_int) c_int {
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
fn encodeArgmax(ctx: *Op, idx: c_int) c_int {
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
fn encodeOptStepAdamw(ctx: *Op, idx: c_int) c_int {
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
fn encodeOptStepSgd(ctx: *Op, idx: c_int) c_int {
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
fn encodeCountEqual(ctx: *Op, idx: c_int) c_int {
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

/// Ports `ggml_metal_op_fwht` (ggml-metal-ops.cpp:2205 @c1d0e7a00).
///
/// Not reached from the dispatch switch: `ggml_metal_op_mul_mat` calls it
/// at its line 2315. It is an export of this translation unit all the
/// same. Note it reads `src[1]`, not `src[0]`.
fn encodeFwht(ctx: *Op, idx: c_int) c_int {
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

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
