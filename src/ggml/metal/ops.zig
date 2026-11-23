//! The Metal op encoder: one graph node to one kernel dispatch.
//!
//! # Provenance
//!
//! Ports `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` at v1.0.0
//! (`b1b770d90`).
//!
//! # What this file does
//!
//! For each node of a graph it picks a pipeline (`library.zig`), fills the
//! kernel's argument block (`kargs.zig`), binds the buffers, and
//! dispatches. The encoder itself lives in `ggml-metal-device.m`, which
//! stays Objective-C and is reached through
//! `device_c.zig`.
//!
//! # The encoders are exported under the C's names
//!
//! Each `ggml_metal_op_*` is a `pub export fn`, so the linker's
//! missing-symbol errors are what proved the port complete: all **72**
//! of the translation unit's unmangled exports resolve. They were
//! deliberately *private* while bodies were still stubs, because
//! `scripts/port-coverage` would otherwise have read `66 / 66 (100%)`
//! for a file whose every dispatch aborted — which is exactly what
//! `repack.cpp` did before `scripts/cluster-check` existed.
//!
//! `encoders_implemented` is the comptime guard for that, in the same
//! shape as `repack.dispatch_implemented`.

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

/// The allocator ported ggml uses — libc's, which is what a C entry point
/// can reach, so memory crossing the C ABI can be freed by either side.
/// Ports `GGML_METAL_MIN_THREADGROUPS` (ggml-metal-ops.cpp:20 @b1b770d90):
/// the threadgroups needed to fill the GPU.
const min_threadgroups: i64 = 128;

/// `op->src[i]`, **narrowed**.
///
/// `op.src[i]` is a `[*c]ggml_tensor`, and Zig 0.16 types `p.*.ne[0]` on a
/// C pointer as the whole `[4]i64`, a miscompile.
/// `.?` does **not** narrow a `[*c]`; `impl.one` does.
inline fn src(op: *const ggml.impl.ggml_c.ggml_tensor, i: usize) *const ggml.impl.ggml_c.ggml_tensor {
    return ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, op.src[i]);
}

/// `op->src[i]` when it may be absent.
inline fn srcOpt(op: *const ggml.impl.ggml_c.ggml_tensor, i: usize) ?*const ggml.impl.ggml_c.ggml_tensor {
    return if (op.src[i] == null) null else ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, op.src[i]);
}

// Where the C reads a possibly-absent source through
// `GGML_TENSOR_LOCALS(int32_t, ne1, op->src[1], ne)`, the macro is
// **null-guarded** -- `GGML_TENSOR_LOCALS_1` (ggml.h:125 @b1b770d90)
// expands to `(pointer) ? (pointer)->array[0] : 0`. So an absent source
// contributes **zeros** to the `kargs`, not garbage and not a
// dereference, which is why `encodeSoftMax` writes
// `if (s1) |t| t.nb[1] else 0` rather than asserting it is present.

/// Ports `ggml_metal_get_buffer_id` (ggml-metal-ops.cpp:22 @b1b770d90).
///
/// Return: the Metal buffer and offset backing `t`, or a null handle when
/// `t` is absent. A view reads its source's buffer.
fn getBufferId(t: ?*const ggml.impl.ggml_c.ggml_tensor) ggml.metal.device_c.BufferId {
    const tensor = t orelse return .{ .metal = null, .offs = 0 };

    const buffer: *ggml.impl.ggml_c.ggml_backend_buffer = if (tensor.view_src != null)
        ggml.impl.one(ggml.impl.ggml_c.ggml_backend_buffer, ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor.view_src).buffer)
    else
        ggml.impl.one(ggml.impl.ggml_c.ggml_backend_buffer, tensor.buffer);

    const ctx: *ggml.metal.device_c.Buffer = @ptrCast(@alignCast(buffer.context.?));

    return ggml.metal.device_c.ggml_metal_buffer_get_id(ctx, tensor);
}

/// The C's `ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), idx)`,
/// with the size taken from the type rather than restated.
///
/// The encoder copies the bytes, so the pointer does not outlive the call
/// and a stack `args` is correct — as it is in the C.
inline fn setBytes(enc: *ggml.metal.device_c.Encoder, args: anytype, idx: c_int) void {
    const T = @TypeOf(args.*);
    ggml.metal.device_c.ggml_metal_encoder_set_bytes(enc, @ptrCast(@constCast(args)), @sizeOf(T), idx);
}

/// `ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(t), idx)`.
inline fn setBuffer(enc: *ggml.metal.device_c.Encoder, t: ?*const ggml.impl.ggml_c.ggml_tensor, idx: c_int) void {
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, getBufferId(t), idx);
}

/// `ggml_metal_pipeline_max_theads_per_threadgroup`, which the encoders
/// clamp thread counts against. The misspelling is upstream's.
inline fn maxThreads(pipeline: ggml.metal.device_c.PipelineWithParams) c_int {
    return ggml.metal.device_c.ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
}

/// Ports `struct ggml_metal_op` (ggml-metal-ops.cpp:34 @b1b770d90).
///
/// The C's one STL member is `std::vector<int> idxs`, the non-empty node
/// indices; here it is a heap slice sized once from `gf->n_nodes`, which
/// is the bound the C reserves to.
pub const Op = struct {
    dev: *ggml.metal.device_c.Device,
    lib: *ggml.metal.device_c.Library,
    enc: *ggml.metal.device_c.Encoder,
    mem_ranges: ?*ggml.metal.common.MemRanges,

    /// shared fusion debugging context, owned by the device
    finfo: *ggml.metal.fusion.Info,

    use_concurrency: bool,
    use_capture: bool,

    debug_graph: c_int,

    gf: *ggml.impl.CGraph,

    idx_start: c_int,
    idx_end: c_int,

    /// non-empty node indices
    idxs: []c_int,
    n_idxs: usize,

    /// Ports `ggml_metal_op`'s constructor (ggml-metal-ops.cpp:35
    /// @b1b770d90).
    fn init(
        dev: *ggml.metal.device_c.Device,
        cmd_buf: ggml.metal.device_c.CmdBuf,
        gf: *ggml.impl.CGraph,
        finfo: *ggml.metal.fusion.Info,
        idx_start: c_int,
        idx_end: c_int,
        use_concurrency: bool,
        use_capture: bool,
        debug_graph: c_int,
    ) *Op {
        const self = std.heap.c_allocator.create(Op) catch ggml.impl.abort("metal: out of memory allocating the op context");

        self.* = .{
            .dev = dev,
            .lib = ggml.metal.device_c.ggml_metal_device_get_library(dev) orelse ggml.impl.abort("metal: device has no library"),
            .enc = ggml.metal.device_c.ggml_metal_encoder_init(cmd_buf, use_concurrency) orelse
                ggml.impl.abort("metal: could not create a command encoder"),
            .mem_ranges = ggml.metal.common.ggml_mem_ranges_init(debug_graph),
            .finfo = finfo,
            .use_concurrency = use_concurrency,
            .use_capture = use_capture,
            .debug_graph = debug_graph,
            .gf = gf,
            .idx_start = idx_start,
            .idx_end = idx_end,
            .idxs = std.heap.c_allocator.alloc(c_int, @intCast(@max(gf.n_nodes, 0))) catch
                ggml.impl.abort("metal: out of memory allocating the node index list"),
            .n_idxs = 0,
        };

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        var i = idx_start;
        while (i < idx_end) : (i += 1) {
            const n = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, ggml.graph.ggml_graph_node(@ptrCast(gf), i));
            if (!ggml.impl.opIsEmpty(n.op) and !ggml.types.ggml_is_empty(n)) {
                self.idxs[self.n_idxs] = i;
                self.n_idxs += 1;
            }
        }

        return self;
    }

    /// Ports `ggml_metal_op`'s destructor (ggml-metal-ops.cpp:69
    /// @b1b770d90).
    fn deinit(self: *Op) void {
        ggml.metal.device_c.ggml_metal_encoder_end_encoding(self.enc);
        ggml.metal.device_c.ggml_metal_encoder_free(self.enc);
        if (self.mem_ranges) |mr| ggml.metal.common.ggml_mem_ranges_free(mr);
        std.heap.c_allocator.free(self.idxs);
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `n_nodes` (ggml-metal-ops.cpp:75 @b1b770d90).
    fn nNodes(self: *const Op) c_int {
        return @intCast(self.n_idxs);
    }

    /// Ports `node` (ggml-metal-ops.cpp:79 @b1b770d90).
    fn node(self: *const Op, i: c_int) *ggml.impl.ggml_c.ggml_tensor {
        ggml.impl.assert(i >= 0 and i < self.nNodes(), "i >= 0 && i < (int) idxs.size()");
        return ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, ggml.graph.ggml_graph_node(@ptrCast(self.gf), self.idxs[@intCast(i)]));
    }

    /// Ports `can_fuse` (ggml-metal-ops.cpp:86 @b1b770d90): the
    /// fusion table's longest pattern starting at `j0`, over this
    /// encoder's non-empty nodes only, so a run never crosses into
    /// another encoder's split.
    ///
    /// `i0` is renamed `j0`: Zig reserves every `iN` as an integer type
    /// name. Digit for digit, so the index arithmetic reads against the C.
    ///
    /// Return: the matching pattern, or null; `n_out` receives the
    /// number of nodes it covers.
    fn canFuse(self: *const Op, j0: c_int, mode: ggml.metal.fusion.Mode, n_out: *c_int) ?*const ggml.metal.fusion.Fusion {
        // the C's `assert(use_fusion())` and `assert(i0 >= 0 && i0 < n_nodes())`
        // compile out in the reference's release build, as here

        return ggml.metal.fusion.ggml_metal_fusion_next(self.gf, self.idxs.ptr, self.nNodes(), j0, mode, n_out);
    }

    /// Ports `use_fusion` (ggml-metal-ops.cpp:95 @b1b770d90): whether to
    /// attempt fusion. The toggle lives in the device's shared fusion
    /// info, set from `GGML_METAL_FUSION_DISABLE` and overridable through
    /// `ggml_backend_fusion_set_enabled`.
    fn useFusion(self: *const Op) bool {
        return ggml.metal.fusion.ggml_metal_fusion_info_enabled(self.finfo);
    }

    /// Ports `count_fusions` (ggml-metal-ops.cpp:100 @b1b770d90): record
    /// that a fusion fired, indexed by its table entry.
    fn countFusions(self: *const Op, f: *const ggml.metal.fusion.Fusion) void {
        ggml.metal.fusion.ggml_metal_fusion_info_count_fusion(self.finfo, f);
    }

    /// `ggml_metal_fusion_info_debug(ctx->finfo)`, which the encoders read
    /// where they used to read `ctx->debug_fusion`.
    fn debugFusion(self: *const Op) c_int {
        return ggml.metal.fusion.ggml_metal_fusion_info_debug(self.finfo);
    }
};

/// Ports `ggml_metal_op_init` (ggml-metal-ops.cpp:127 @b1b770d90).
pub fn ggml_metal_op_init(
    dev: *ggml.metal.device_c.Device,
    cmd_buf: ggml.metal.device_c.CmdBuf,
    gf: *ggml.impl.CGraph,
    finfo: *ggml.metal.fusion.Info,
    idx_start: c_int,
    idx_end: c_int,
    use_concurrency: bool,
    use_capture: bool,
    debug_graph: c_int,
) callconv(.c) *Op {
    return Op.init(
        dev,
        cmd_buf,
        gf,
        finfo,
        idx_start,
        idx_end,
        use_concurrency,
        use_capture,
        debug_graph,
    );
}

/// Ports `ggml_metal_op_free` (ggml-metal-ops.cpp:151 @b1b770d90).
pub fn ggml_metal_op_free(ctx: *Op) callconv(.c) void {
    ctx.deinit();
}

/// Ports `ggml_metal_op_n_nodes` (ggml-metal-ops.cpp:155 @b1b770d90).
pub fn ggml_metal_op_n_nodes(ctx: *Op) callconv(.c) c_int {
    return ctx.nNodes();
}

/// Ports `ggml_metal_op_concurrency_reset` (ggml-metal-ops.cpp:159
/// @b1b770d90).
fn concurrencyReset(ctx: *Op) bool {
    const mr = ctx.mem_ranges orelse return true;

    ggml.metal.device_c.ggml_metal_encoder_memory_barrier(ctx.enc);

    ggml.metal.common.ggml_mem_ranges_reset(mr);

    return true;
}

/// Ports `ggml_metal_op_concurrency_check` (ggml-metal-ops.cpp:171
/// @b1b770d90).
fn concurrencyCheck(ctx: *Op, node: *const ggml.impl.ggml_c.ggml_tensor) bool {
    const mr = ctx.mem_ranges orelse return false;

    return ggml.metal.common.ggml_mem_ranges_check(mr, node);
}

/// Ports `ggml_metal_op_encode` (ggml-metal-ops.cpp:546 @b1b770d90).
///
/// Return: how many nodes were consumed — more than one when the encoder
/// fused a run of them.
pub fn ggml_metal_op_encode(ctx: *Op, idx: c_int) callconv(.c) c_int {
    if (ctx.use_capture) {
        ggml.metal.device_c.ggml_metal_encoder_debug_group_push(ctx.enc, ggml.types.ggml_op_desc(ctx.node(idx)));
    }

    const res = encodeImpl(ctx, idx);
    if (idx + res > ctx.nNodes()) {
        ggml.impl.abort("fusion error: nodes spanning multiple encoders have been fused. " ++
            "this indicates a bug in the fusion logic " ++
            "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx.use_capture) {
        ggml.metal.device_c.ggml_metal_encoder_debug_group_pop(ctx.enc);
    }

    return res;
}

/// Ports `ggml_metal_op_concurrency_add` (ggml-metal-ops.cpp:179
/// @b1b770d90).
fn concurrencyAdd(ctx: *Op, node: *const ggml.impl.ggml_c.ggml_tensor) bool {
    const mr = ctx.mem_ranges orelse return true;

    return ggml.metal.common.ggml_mem_ranges_add(mr, node);
}

/// Ports the dispatch switch of `ggml_metal_op_encode_impl`
/// (ggml-metal-ops.cpp:187 @b1b770d90), which begins at its line 286.
///
/// Each arm is written as `=> encodeX(ctx, idx)` rather than the C's
/// `n_fuse = …; break;`, so a missing arm is a compile error here where
/// the C would fall to its `default` and abort at run time.
fn dispatch(ctx: *Op, node: *const ggml.impl.ggml_c.ggml_tensor, idx: c_int) c_int {
    return switch (node.op) {
        ggml.impl.ggml_c.GGML_OP_CONCAT => ggml_metal_op_concat(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_SUB, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_DIV => ggml_metal_op_bin(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_ADD_ID => ggml_metal_op_add_id(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_REPEAT => ggml_metal_op_repeat(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_ACC => ggml_metal_op_acc(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SCALE,
        ggml.impl.ggml_c.GGML_OP_FILL,
        ggml.impl.ggml_c.GGML_OP_CLAMP,
        ggml.impl.ggml_c.GGML_OP_LEAKY_RELU,
        ggml.impl.ggml_c.GGML_OP_SQR,
        ggml.impl.ggml_c.GGML_OP_SQRT,
        ggml.impl.ggml_c.GGML_OP_SIN,
        ggml.impl.ggml_c.GGML_OP_COS,
        ggml.impl.ggml_c.GGML_OP_LOG,
        ggml.impl.ggml_c.GGML_OP_UNARY,
        => ggml_metal_op_unary(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SILU_BACK => ggml_metal_op_silu_back(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_GLU => ggml_metal_op_glu(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SUM => ggml_metal_op_sum(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_MEAN => ggml_metal_op_sum_rows(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_CUMSUM => ggml_metal_op_cumsum(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER => ggml_metal_op_lightning_indexer(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_TOP_K_MASK => ggml_metal_op_top_k_mask(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_COMB,
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_PRE,
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST,
        => ggml_metal_op_dsv4_hc(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX => ggml_metal_op_soft_max(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SSM_CONV => ggml_metal_op_ssm_conv(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SSM_SCAN => ggml_metal_op_ssm_scan(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV6, ggml.impl.ggml_c.GGML_OP_RWKV_WKV7 => ggml_metal_op_rwkv(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET => ggml_metal_op_gated_delta_net(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SOLVE_TRI => ggml_metal_op_solve_tri(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_MUL_MAT => ggml_metal_op_mul_mat(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID => ggml_metal_op_mul_mat_id(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_GET_ROWS => ggml_metal_op_get_rows(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SET_ROWS => ggml_metal_op_set_rows(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_DIAG => ggml_metal_op_diag(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_L2_NORM => ggml_metal_op_l2_norm(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_GROUP_NORM => ggml_metal_op_group_norm(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_NORM, ggml.impl.ggml_c.GGML_OP_RMS_NORM => ggml_metal_op_norm(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_ROPE_BACK => ggml_metal_op_rope(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_IM2COL => ggml_metal_op_im2col(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_CONV_2D => ggml_metal_op_conv_2d(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_CONV_2D_DW => ggml_metal_op_conv_2d_dw(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D => ggml_metal_op_conv_transpose_1d(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D => ggml_metal_op_conv_transpose_2d(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_COL2IM_1D => ggml_metal_op_col2im_1d(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_CONV_3D => ggml_metal_op_conv_3d(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_UPSCALE => ggml_metal_op_upscale(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_PAD => ggml_metal_op_pad(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D => ggml_metal_op_pad_reflect_1d(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_ROLL => ggml_metal_op_roll(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_ARANGE => ggml_metal_op_arange(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_TIMESTEP_EMBEDDING => ggml_metal_op_timestep_embedding(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_ARGSORT => ggml_metal_op_argsort(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_TOP_K => ggml_metal_op_top_k(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_TRI => ggml_metal_op_tri(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT => ggml_metal_op_flash_attn_ext(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_SET => ggml_metal_op_set(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_DUP, ggml.impl.ggml_c.GGML_OP_CPY, ggml.impl.ggml_c.GGML_OP_CONT => ggml_metal_op_cpy(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_POOL_1D => ggml_metal_op_pool_1d(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_POOL_2D => ggml_metal_op_pool_2d(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_ARGMAX => ggml_metal_op_argmax(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_ADAMW => ggml_metal_op_opt_step_adamw(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_SGD => ggml_metal_op_opt_step_sgd(ctx, idx),
        ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL => ggml_metal_op_count_equal(ctx, idx),
        else => {
            ggml.impl.logError("%s: error: node %3d, op = %8s not implemented\n", .{
                "ggml_metal_op_encode_impl", idx, ggml.types.ggml_op_name(node.op),
            });
            ggml.impl.abort("metal: op not implemented");
        },
    };
}

/// Ports `ggml_metal_op_encode_impl` (ggml-metal-ops.cpp:187 @b1b770d90).
fn encodeImpl(ctx: *Op, idx: c_int) c_int {
    const node = ctx.node(idx);

    if (ggml.types.ggml_is_empty(node)) {
        return 1;
    }

    switch (node.op) {
        ggml.impl.ggml_c.GGML_OP_NONE,
        ggml.impl.ggml_c.GGML_OP_RESHAPE,
        ggml.impl.ggml_c.GGML_OP_VIEW,
        ggml.impl.ggml_c.GGML_OP_TRANSPOSE,
        ggml.impl.ggml_c.GGML_OP_PERMUTE,
        => {
            // noop -> next node
            if (ctx.debug_graph > 0) {
                ggml.impl.logDebug("%s: node[%5d] - %-12s %s\n", .{
                    "ggml_metal_op_encode_impl", idx, ggml.types.ggml_op_name(node.op), "(noop)",
                });
            }
            return 1;
        },
        else => {},
    }

    if (!ggml.metal.device_c.ggml_metal_device_supports_op(ctx.dev, node)) {
        ggml.impl.logError("%s: error: unsupported op '%s'\n", .{
            "ggml_metal_op_encode_impl", ggml.types.ggml_op_desc(node),
        });
        ggml.impl.abort("unsupported op");
    }

    if ((node.flags & ggml.impl.ggml_c.GGML_TENSOR_FLAG_COMPUTE) == 0) {
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
        var is_concurrent = concurrencyCheck(ctx, node);

        if (is_concurrent and ctx.useFusion()) {
            var n_fuse: c_int = 1;
            if (ctx.canFuse(idx, .full, &n_fuse) != null) {
                // fused kernels write to the last node of the group, not necessarily to the first node's dst
                is_concurrent = ggml.metal.common.ggml_mem_ranges_check(ctx.mem_ranges.?, ctx.node(idx + n_fuse - 1));
            }
        }

        if (!is_concurrent) {
            _ = concurrencyReset(ctx);
        }

        if (ctx.debug_graph > 0) {
            ggml.impl.logDebug("%s: node[%5d] - %-12s %-12s %s\n", .{
                "ggml_metal_op_encode_impl",
                idx,
                ggml.types.ggml_op_name(node.op),
                ggml.context.ggml_get_name(node),
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
            ggml.impl.logDebug("%s:               fuse %d ops\n", .{ "ggml_metal_op_encode_impl", n_fuse });
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
/// (ggml-metal-ops.cpp:187 @b1b770d90), at its line 251 — lifted out so
/// the control flow above reads.
///
/// The C expands ten `GGML_TENSOR_LOCALS` here and prints four sources
/// and the node; the loop is equivalent and spells the same format.
fn debugLogShapes(node: *const ggml.impl.ggml_c.ggml_tensor) void {
    for (0..4) |j| {
        const s = srcOpt(node, j) orelse continue;
        ggml.impl.logDebug(
            "%s: src%d - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n",
            .{
                "ggml_metal_op_encode_impl", @as(c_int, @intCast(j)), ggml.types.ggml_type_name(s.type),
                s.ne[0],                     s.ne[1],                 s.ne[2],
                s.ne[3],                     s.nb[0],                 s.nb[1],
                s.nb[2],                     s.nb[3],                 @as(c_int, if (ggml.types.ggml_is_contiguous(s)) 1 else 0),
                &s.name,
            },
        );
    }

    ggml.impl.logDebug(
        "%s: node - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n",
        .{
            "ggml_metal_op_encode_impl", ggml.types.ggml_type_name(node.type), node.ne[0],
            node.ne[1],                  node.ne[2],                           node.ne[3],
            node.nb[0],                  node.nb[1],                           node.nb[2],
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
        ggml.impl.abort("metal: ggml_metal_op_" ++ what ++ " is not ported yet");
    }
};

/// Ports `ggml_metal_op_concat` (ggml-metal-ops.cpp:564 @b1b770d90).
///
/// A quantized type concatenates whole blocks: its row lengths are passed
/// in blocks, so `nb0 == type_size` is the block's size.
pub fn ggml_metal_op_concat(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const dim = ggml.impl.getOpParamsI32(op, 0);

    const is_q = ggml.types.ggml_is_quantized(op.type);

    // for quantized types, concat is done at the block level (nb0 == type_size == block size)
    var ne00_arg: i32 = @intCast(s0.ne[0]);
    var ne10_arg: i32 = @intCast(s1.ne[0]);
    var ne0_arg: i32 = @intCast(op.ne[0]);
    if (is_q) {
        const blck: i32 = @intCast(ggml.types.ggml_blck_size(op.type));
        ggml.impl.assert(@rem(ne00_arg, blck) == 0, "ne00 % blck == 0");
        ggml.impl.assert(@rem(ne10_arg, blck) == 0, "ne10 % blck == 0");
        ggml.impl.assert(@rem(ne0_arg, blck) == 0, "ne0  % blck == 0");
        ne00_arg = @divTrunc(ne00_arg, blck);
        ne10_arg = @divTrunc(ne10_arg, blck);
        ne0_arg = @divTrunc(ne0_arg, blck);
    }

    const ne1: c_int = @intCast(op.ne[1]);

    const nth = @min(@as(c_int, 256), ne0_arg);

    // when rows are small, we can batch them together in a single threadgroup
    var nrptg: c_int = 1;
    if (nth < 256) {
        nrptg = @min(@divTrunc(256 + nth - 1, nth), ne1);
        if (nrptg * nth > 256) {
            nrptg = @divTrunc(@as(c_int, 256), nth);
        }
    }

    const nw0 = @divTrunc(ne1 + nrptg - 1, nrptg);

    // split long rows across threadgroups when there are too few rows to fill the GPU
    const n_rows: i64 = @as(i64, nw0) * op.ne[2] * op.ne[3];

    var nc0: c_int = 1;
    if (nrptg == 1) {
        nc0 = @intCast(@max(1, @min(@divTrunc(@as(i64, ne0_arg) + nth - 1, nth), @divTrunc(min_threadgroups + n_rows - 1, n_rows))));
    }

    const args: ggml.metal.kargs.concat = .{
        .ne00 = ne00_arg,
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .ne10 = ne10_arg,
        .ne11 = @intCast(s1.ne[1]),
        .ne12 = @intCast(s1.ne[2]),
        .ne13 = @intCast(s1.ne[3]),
        .nb10 = s1.nb[0],
        .nb11 = s1.nb[1],
        .nb12 = s1.nb[2],
        .nb13 = s1.nb[3],
        .ne0 = ne0_arg,
        .ne1 = @intCast(op.ne[1]),
        .ne2 = @intCast(op.ne[2]),
        .ne3 = @intCast(op.ne[3]),
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .dim = dim,
        .nc0 = nc0,
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_concat(lib, op.type);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, nw0 * nc0, @intCast(op.ne[2]), @intCast(op.ne[3]), nth, nrptg, 1);

    return 1;
}

/// Ports `ggml_metal_op_bin` (ggml-metal-ops.cpp:4351 @b1b770d90).
///
/// Fuses a run of up to **seven** `ADD`s into one kernel by passing each
/// operand's offset in `o1[]`; all of them must live in the same Metal
/// buffer, which is why `bid_src1.offs` is zeroed and the offsets become
/// relative to it. A `MUL` opening the snake pattern goes to
/// `ggml_metal_op_snake_fused` instead, and one opening the MoE output
/// reduction to `ggml_metal_op_moe_reduce`. Which run fuses is the fusion
/// table's answer (`fusion.zig`), the same one the optimizer packed by.
pub fn ggml_metal_op_bin(ctx: *Op, idx: c_int) callconv(.c) c_int {
    var n_fuse: c_int = 1;
    var fus: ?*const ggml.metal.fusion.Fusion = null;

    const use_fusion = ctx.useFusion();

    if (use_fusion) {
        var n: c_int = 1;
        fus = ctx.canFuse(idx, .full, &n);
        n_fuse = n;

        // snake activation autofuse: mul -> sin -> sqr -> mul -> add
        if (fus) |f| {
            if (ggml.metal.fusion.ggml_metal_fusion_get_id(f) == .snake) {
                ctx.countFusions(f);
                return ggml_metal_op_snake_fused(ctx, idx);
            }
        }

        // MoE output reduction: experts * weights -> weighted sum
        if (fus) |f| {
            if (ggml.metal.fusion.ggml_metal_fusion_get_id(f) == .moe_reduce) {
                ctx.countFusions(f);
                return ggml_metal_op_moe_reduce(ctx, idx);
            }
        }
    }

    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const debug_fusion = ctx.debugFusion();

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");
    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s1), "ggml_is_contiguous_rows(op->src[1])");

    const bid_src0 = getBufferId(s0);
    var bid_src1 = getBufferId(s1);
    var bid_dst = getBufferId(op);

    const ne00: i32 = @intCast(s0.ne[0]);
    const ne10: i32 = @intCast(s1.ne[0]);
    const ne0: i32 = @intCast(op.ne[0]);

    var args: ggml.metal.kargs.bin = .{
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

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion and fus != null and ggml.metal.fusion.ggml_metal_fusion_get_id(fus.?) == .add_chain) {
        // the offsets of the fused addends are relative to the start of the src1 buffer
        var i: c_int = 1;
        while (i < n_fuse) : (i += 1) {
            args.o1[@intCast(i)] = getBufferId(src(ctx.node(idx + i), 1)).offs;
        }

        ctx.countFusions(fus.?);

        if (debug_fusion > 1) {
            ggml.impl.logDebug("%s: fuse: ADD x %d\n", .{ "ggml_metal_op_bin", n_fuse });
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

    if (pipeline.cnt) {
        // A threadgroup holds up to 256 threads.
        const nth: c_int = @min(@as(c_int, 256), args.ne0);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @divTrunc(args.ne0 + nth - 1, nth),
            @intCast(ggml.types.ggml_nrows(op)),
            1,
            nth,
            1,
            1,
        );
    } else {
        const nth_max = @min(@as(c_int, 256), maxThreads(pipeline));

        var nth: c_int = 1;

        while (2 * nth < args.ne0 and nth < nth_max) {
            nth *= 2;
        }

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_add_id` (ggml-metal-ops.cpp:3154 @b1b770d90).
///
/// Short, despite the 250 lines the file gives it -- the rest of that
/// span is the `flash_attn_ext` helper group that follows it.
pub fn ggml_metal_op_add_id(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = src(op, 2);

    ggml.impl.assert(s0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    ggml.impl.assert(s1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    ggml.impl.assert(s2.type == ggml.impl.ggml_c.GGML_TYPE_I32, "op->src[2]->type == GGML_TYPE_I32");
    ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    var args: ggml.metal.kargs.add_id = .{
        .ne0 = @intCast(op.ne[0]),
        .ne1 = @intCast(op.ne[1]),
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb11 = s1.nb[1],
        .nb21 = s2.nb[1],
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_base(lib, ggml.impl.ggml_c.GGML_OP_ADD_ID);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, s2, 3);
    setBuffer(enc, op, 4);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(s0.ne[0])));

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_repeat` (ggml-metal-ops.cpp:658 @b1b770d90).
pub fn ggml_metal_op_repeat(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_repeat(lib, op.type);

    var args: ggml.metal.kargs.repeat = .{
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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(op.ne[0])));

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_acc` (ggml-metal-ops.cpp:704 @b1b770d90).
///
/// Two dispatches when not in-place: a `cpy` of src0 into dst, then an
/// `ADD` of src1 into a *window* of it — which is what the `pnb*` strides
/// and `offs` describe. Note the `bin` args deliberately feed `src[1]`'s
/// shape into the `ne0*` fields and the window strides into `nb0*`.
pub fn ggml_metal_op_acc(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    ggml.impl.assert(s0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    ggml.impl.assert(s1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");
    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s1), "ggml_is_contiguous_rows(op->src[1])");

    const pnb1: usize = @intCast(ggml.impl.getOpParamsI32(op, 0));
    const pnb2: usize = @intCast(ggml.impl.getOpParamsI32(op, 1));
    const pnb3: usize = @intCast(ggml.impl.getOpParamsI32(op, 2));
    const offs: usize = @intCast(ggml.impl.getOpParamsI32(op, 3));

    const inplace = ggml.impl.getOpParamsI32(op, 4) != 0;

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_cpy(lib, s0.type, op.type);

        var args: ggml.metal.kargs.cpy = .{
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, op, 2);

        const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(s0.ne[0])));

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

    var args: ggml.metal.kargs.bin = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_bin_one(lib, ggml.impl.ggml_c.GGML_OP_ADD);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    const nth_max = @min(@as(c_int, 256), maxThreads(pipeline));

    var nth: c_int = 1;

    while (2 * nth < args.ne0 and nth < nth_max) {
        nth *= 2;
    }

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_unary` (ggml-metal-ops.cpp:821 @b1b770d90).
///
/// One kernel for nine ops plus twenty-two `GGML_UNARY_OP_*`; the op's
/// own parameters are packed into the six trailing float fields, which
/// start at zero and are filled only by the arms that use them. A
/// `SOFTPLUS` or `SIGMOID` (#31615) may also anchor a `TOPK_MOE`
/// fusion (the DeepSeek-V4 and GLM routings), and is handed to
/// `ggml_metal_op_topk_moe` when it does.
pub fn ggml_metal_op_unary(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    // the sigmoid and sqrt(softplus) TOPK_MOE fusions anchor at the gating unary node
    if (op.op == ggml.impl.ggml_c.GGML_OP_UNARY and ctx.useFusion() and
        (ggml.types.ggml_get_unary_op(op) == ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS or ggml.types.ggml_get_unary_op(op) == ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID))
    {
        var n: c_int = 1;
        if (ctx.canFuse(idx, .full, &n)) |f| {
            if (ggml.metal.fusion.ggml_metal_fusion_get_id(f) == .topk_moe) {
                return ggml_metal_op_topk_moe(ctx, idx);
            }
        }
    }

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const bid_src0 = getBufferId(s0);
    const bid_dst = getBufferId(op);

    const ne00: i32 = @intCast(s0.ne[0]);
    const ne0: i32 = @intCast(op.ne[0]);

    var args: ggml.metal.kargs.unary = .{
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

    if (op.op == ggml.impl.ggml_c.GGML_OP_LEAKY_RELU) {
        args.slope = ggml.impl.getOpParamsF32(op, 0);
    }

    if (op.op == ggml.impl.ggml_c.GGML_OP_SCALE) {
        args.scale = ggml.impl.getOpParamsF32(op, 0);
        args.bias = ggml.impl.getOpParamsF32(op, 1);
    }

    if (op.op == ggml.impl.ggml_c.GGML_OP_FILL) {
        args.val = ggml.impl.getOpParamsF32(op, 0);
    }

    if (op.op == ggml.impl.ggml_c.GGML_OP_CLAMP) {
        args.min = ggml.impl.getOpParamsF32(op, 0);
        args.max = ggml.impl.getOpParamsF32(op, 1);
    }

    if (op.op == ggml.impl.ggml_c.GGML_OP_UNARY and ggml.types.ggml_get_unary_op(op) == ggml.impl.ggml_c.GGML_UNARY_OP_XIELU) {
        args.slope = ggml.impl.getOpParamsF32(op, 1); // alpha_n
        args.scale = ggml.impl.getOpParamsF32(op, 2); // alpha_p
        args.bias = ggml.impl.getOpParamsF32(op, 3); // beta
        args.val = ggml.impl.getOpParamsF32(op, 4); // eps
    }

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_unary(lib, op);

    if (pipeline.c4) {
        args.ne00 = @divTrunc(ne00, 4);
        args.ne0 = @divTrunc(ne0, 4);
    }

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    if (pipeline.cnt) {
        const n: c_int = @intCast(if (pipeline.c4)
            @divTrunc(ggml.types.ggml_nelements(op), 4)
        else
            ggml.types.ggml_nelements(op));

        // A threadgroup holds up to 256 threads.
        const nth: c_int = @min(@as(c_int, 256), n);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @divTrunc(n + nth - 1, nth), 1, 1, nth, 1, 1);
    } else {
        const nth_max = @min(@as(c_int, 256), maxThreads(pipeline));
        const nth = @min(args.ne00, nth_max);
        const nk0 = @divTrunc(args.ne00 + nth - 1, nth);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_silu_back` (ggml-metal-ops.cpp:4492 @b1b770d90).
pub fn ggml_metal_op_silu_back(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_silu_back(lib, op);

    const ne: i64 = ggml.types.ggml_nelements(op);

    var args: ggml.metal.kargs.silu_back = .{
        .ne = ne,
    };

    var arg_idx: c_int = 0;

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
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

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n), 1, 1, @intCast(nth), 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_glu` (ggml-metal-ops.cpp:926 @b1b770d90).
///
/// `src[1]` is optional: without it the kernel reads both halves out of
/// `src[0]`, which is what `i00`/`i10` select — and note the C passes
/// **0** for both when `src[1]` is present, having just computed them.
pub fn ggml_metal_op_glu(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = srcOpt(op, 1);

    if (src1) |s1| {
        ggml.impl.assert(ggml.types.ggml_are_same_shape(src0, s1), "ggml_are_same_shape(op->src[0], op->src[1])");
    }

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_glu(lib, op);

    const swp = ggml.impl.getOpParamsI32(op, 1);
    const alpha = ggml.impl.getOpParamsF32(op, 2);
    const limit = ggml.impl.getOpParamsF32(op, 3);

    const ne00: i32 = @intCast(src0.ne[0]);
    const ne0: i32 = @intCast(op.ne[0]);

    // `i00`/`i10` are reserved integer type names in Zig, renamed digit
    // for digit. The `kargs` fields
    // keep the C's spelling.
    const j00: i32 = if (swp != 0) ne0 else 0;
    const j10: i32 = if (swp != 0) 0 else ne0;

    var args: ggml.metal.kargs.glu = .{
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

    const nrows: i64 = ggml.types.ggml_nrows(src0);

    // split each row over several threadgroups: with a single row (decode) one threadgroup would run on one GPU core
    const nth: i32 = @max(1, @min(maxThreads(pipeline), @min(@as(i32, 256), ne0)));
    const nk: i32 = @divTrunc(ne0 + nth - 1, nth);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    if (src1) |s1| {
        setBuffer(enc, s1, 2);
    } else {
        setBuffer(enc, src0, 2);
    }
    setBuffer(enc, op, 3);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, nk, @intCast(nrows), 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_sum` (ggml-metal-ops.cpp:986 @b1b770d90).
pub fn ggml_metal_op_sum(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const n: u64 = @intCast(ggml.types.ggml_nelements(s0));

    var args: ggml.metal.kargs.sum = .{
        .np = n,
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_sum(lib, op);

    var nth: c_int = 32; // SIMD width

    while (nth < @as(c_int, @intCast(n)) and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    nth = @min(nth, maxThreads(pipeline));
    nth = @min(nth, @as(c_int, @intCast(n)));

    const nsg = @divTrunc(nth + 31, 32);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, stdlib.num.pad(usize, @as(usize, @intCast(nsg)) * @sizeOf(f32), 16), 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_sum_rows` (ggml-metal-ops.cpp:1023 @b1b770d90).
pub fn ggml_metal_op_sum_rows(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const bid_src0 = getBufferId(s0);
    const bid_dst = getBufferId(op);

    const ne00: i64 = s0.ne[0];
    const ne0: i64 = op.ne[0];

    var args: ggml.metal.kargs.sum_rows = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_sum_rows(lib, op);

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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_cumsum` (ggml-metal-ops.cpp:1088 @b1b770d90).
///
/// Up to three dispatches: a per-block scan, then — only when the row
/// exceeds one threadgroup — a scan of the block totals and an add of
/// those back into the result. The scratch sits immediately after `dst`,
/// which is why `backend.zig`'s `get_alloc_size` doubles the allocation
/// for `CUMSUM`.
pub fn ggml_metal_op_cumsum(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);

    const pipeline_blk = ggml.metal.library.ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    var nth: c_int = 1;
    while (nth < ne00 and 2 * nth <= maxThreads(pipeline_blk)) {
        nth *= 2;
    }

    ggml.impl.assert(ne00 <= nth * nth, "ne00 <= nth*nth");

    const net0: i64 = @divTrunc(@as(i64, ne00) + nth - 1, nth);
    const net1: i64 = ne01;
    const net2: i64 = ne02;
    const net3: i64 = ne03;

    const nbt0: u64 = @sizeOf(f32);
    const nbt1: u64 = @as(u64, @intCast(net0)) * nbt0;
    const nbt2: u64 = @as(u64, @intCast(net1)) * nbt1;
    const nbt3: u64 = @as(u64, @intCast(net2)) * nbt2;

    const smem = stdlib.num.pad(usize, 32 * @sizeOf(f32), 16);

    const bid_src0 = getBufferId(s0);
    const bid_dst = getBufferId(op);

    var bid_tmp = bid_dst;
    bid_tmp.offs += @intCast(ggml.types.ggml_nbytes(op));

    {
        var args: ggml.metal.kargs.cumsum_blk = .{
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        setBytes(enc, &args, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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
            var args: ggml.metal.kargs.cumsum_blk = .{
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

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            setBytes(enc, &args, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 2);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 3);

            ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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
            const pipeline_add = ggml.metal.library.ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            var args: ggml.metal.kargs.cumsum_add = .{
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

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            setBytes(enc, &args, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_lightning_indexer` (ggml-metal-ops.cpp:1380
/// @b1b770d90).
///
/// The head count is a function constant of the pipeline; `q` is not
/// asserted to have `OP_LIGHTNING_INDEXER_NH` heads.
///
/// Note the C binds the buffers **before** setting the pipeline and the
/// bytes, where every other encoder sets the pipeline first. Order is
/// preserved.
pub fn ggml_metal_op_lightning_indexer(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const enc = ctx.enc;

    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER, "op->op == GGML_OP_LIGHTNING_INDEXER");

    const q = src(op, 0);
    const k = src(op, 1);
    const w = src(op, 2);
    const m = src(op, 3);

    ggml.impl.assert(q.type == ggml.impl.ggml_c.GGML_TYPE_F32, "q->type == GGML_TYPE_F32");
    ggml.impl.assert(
        k.type == ggml.impl.ggml_c.GGML_TYPE_F32 or
            k.type == ggml.impl.ggml_c.GGML_TYPE_F16 or
            k.type == ggml.impl.ggml_c.GGML_TYPE_BF16 or
            k.type == ggml.impl.ggml_c.GGML_TYPE_Q4_0 or
            k.type == ggml.impl.ggml_c.GGML_TYPE_Q4_1 or
            k.type == ggml.impl.ggml_c.GGML_TYPE_Q5_0 or
            k.type == ggml.impl.ggml_c.GGML_TYPE_Q5_1 or
            k.type == ggml.impl.ggml_c.GGML_TYPE_Q8_0,
        "k->type is one of f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0",
    );
    ggml.impl.assert(w.type == ggml.impl.ggml_c.GGML_TYPE_F32, "w->type == GGML_TYPE_F32");
    ggml.impl.assert(m.type == ggml.impl.ggml_c.GGML_TYPE_F16, "m->type == GGML_TYPE_F16");
    ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    ggml.impl.assert(q.ne[0] == ggml.metal.impl_c.OP_LIGHTNING_INDEXER_DK, "q->ne[0] == OP_LIGHTNING_INDEXER_DK");

    var args: ggml.metal.kargs.lightning_indexer = .{
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

    const nsg = ggml.metal.impl_c.OP_LIGHTNING_INDEXER_NSG;
    const nkptg = ggml.metal.impl_c.OP_LIGHTNING_INDEXER_NKPSG * nsg;
    const nbptg = ggml.metal.impl_c.OP_LIGHTNING_INDEXER_NBPTG;

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_lightning_indexer(ctx.lib, op);
    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_top_k_mask` (ggml-metal-ops.cpp:1445 @b1b770d90).
///
/// `src[0]` is the mask, `src[1]` the indices kept per row, and three
/// optional sources shape them: per-index scores (`src[2]`), block ids
/// (`src[3]`), and indices kept regardless (`src[4]`). Two kernels: a fill
/// of all of `dst`, four elements at a time when the size and the buffer's
/// offset allow, a barrier, then one threadgroup per row of `dst` that
/// overwrites the kept cells with the mask's values. An absent source binds
/// the mask in its slot, and its strides are zero.
pub fn ggml_metal_op_top_k_mask(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const mask = src(op, 0);
    const top_k = src(op, 1);
    const score = srcOpt(op, 2);
    const blocks = srcOpt(op, 3);
    const extra = srcOpt(op, 4);

    ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F16 or op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32");
    ggml.impl.assert(mask.type == op.type, "mask->type == op->type");
    ggml.impl.assert(top_k.type == ggml.impl.ggml_c.GGML_TYPE_I32, "top_k->type == GGML_TYPE_I32");
    ggml.impl.assert(score == null or score.?.type == ggml.impl.ggml_c.GGML_TYPE_F32, "!score  || score->type  == GGML_TYPE_F32");
    ggml.impl.assert(blocks == null or blocks.?.type == ggml.impl.ggml_c.GGML_TYPE_I32, "!blocks || blocks->type == GGML_TYPE_I32");
    ggml.impl.assert(extra == null or extra.?.type == ggml.impl.ggml_c.GGML_TYPE_I32, "!extra  || extra->type  == GGML_TYPE_I32");

    const n_kv: i32 = @truncate(mask.ne[0]);
    const n_top: i32 = @truncate(top_k.ne[0]);
    const n_pick: i32 = if (blocks) |b| @truncate(b.ne[1]) else if (score) |s| @truncate(s.ne[0]) else n_kv;
    const n_blk: i32 = if (blocks) |b| @truncate(b.ne[0]) else 0;
    const n_extra: i32 = if (extra) |e| @truncate(e.ne[0]) else 0;

    const bid = getBufferId(op);

    // the vector fill stores 4 elements at a time
    const ts: i32 = @intCast(ggml.types.ggml_type_size(op.type));
    const n_elems: i64 = ggml.types.ggml_nelements(op);

    const has_score = score != null;
    const has_blocks = blocks != null;
    const has_extra = extra != null;

    const vec = @rem(n_elems, 4) == 0 and bid.offs % @as(usize, @intCast(4 * ts)) == 0;

    var args: ggml.metal.kargs.top_k_mask = .{
        .n_kv = n_kv,
        .m1 = @truncate(mask.ne[1]),
        .m2 = @truncate(mask.ne[2]),
        .m3 = @truncate(mask.ne[3]),
        .n_top = n_top,
        .n_pick = n_pick,
        .n_blk = n_blk,
        .n_extra = n_extra,
        .n_elems = n_elems,
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .nbm1 = mask.nb[1],
        .nbm2 = mask.nb[2],
        .nbm3 = mask.nb[3],
        .nbk0 = top_k.nb[0],
        .nbk1 = top_k.nb[1],
        .nbk2 = top_k.nb[2],
        .nbk3 = top_k.nb[3],
        .nbs0 = if (score) |s| s.nb[0] else 0,
        .nbs1 = if (score) |s| s.nb[1] else 0,
        .nbs2 = if (score) |s| s.nb[2] else 0,
        .nbs3 = if (score) |s| s.nb[3] else 0,
        .nbb0 = if (blocks) |b| b.nb[0] else 0,
        .nbb1 = if (blocks) |b| b.nb[1] else 0,
        .nbe0 = if (extra) |e| e.nb[0] else 0,
        .nbe1 = if (extra) |e| e.nb[1] else 0,
        .nbe2 = if (extra) |e| e.nb[2] else 0,
        .nbe3 = if (extra) |e| e.nb[3] else 0,
    };

    // the fill covers the whole dst, the scatter overwrites the kept cells with the mask values
    {
        const pipeline_fill = ggml.metal.library.ggml_metal_library_get_pipeline_top_k_mask(lib, op, true, false, false, false, vec);

        const n_fill: i64 = @divTrunc(n_elems, @as(i64, if (vec) 4 else 1));
        const nth_max = maxThreads(pipeline_fill);

        var nth: c_int = 32; // SIMD width

        while (nth < nth_max and nth < n_fill) {
            nth *= 2;
        }

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_fill);
        setBytes(enc, &args, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid, 1);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(@divTrunc(n_fill + nth - 1, nth)), 1, 1, nth, 1, 1);

        _ = concurrencyReset(ctx);
    }

    {
        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_top_k_mask(lib, op, false, has_score, has_blocks, has_extra, false);

        const n_max = n_top + n_extra;
        const nth_max = maxThreads(pipeline);

        var nth: c_int = 32; // SIMD width

        while (nth < nth_max and nth < n_max) {
            nth *= 2;
        }

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, mask, 1);
        setBuffer(enc, top_k, 2);
        setBuffer(enc, score orelse mask, 3);
        setBuffer(enc, blocks orelse mask, 4);
        setBuffer(enc, extra orelse mask, 5);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid, 6);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(op.ne[1]), @intCast(op.ne[2]), @intCast(op.ne[3]), nth, 1, 1);
    }

    return 1;
}

/// Ports `ggml_metal_op_dsv4_hc` (ggml-metal-ops.cpp:1563 @b1b770d90).
///
/// Three ops behind one entry point, each with its own `kargs` struct and
/// its own source naming. Note the pipeline is set **before** the switch,
/// so all three arms share it. `HC_PRE` takes any number of
/// hyper-connections (a function constant, not an assertion), a `scale`
/// and the gated kernel from op_params slot 1, and `HC_POST` an absent
/// `comb`, which binds the destination one slot earlier.
pub fn ggml_metal_op_dsv4_hc(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const enc = ctx.enc;
    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_dsv4_hc(ctx.lib, op);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);

    switch (op.op) {
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_COMB => {
            const mixes = src(op, 0);
            const scale = src(op, 1);
            const base = src(op, 2);

            ggml.impl.assert(mixes.type == ggml.impl.ggml_c.GGML_TYPE_F32, "mixes->type == GGML_TYPE_F32");
            ggml.impl.assert(scale.type == ggml.impl.ggml_c.GGML_TYPE_F32, "scale->type == GGML_TYPE_F32");
            ggml.impl.assert(base.type == ggml.impl.ggml_c.GGML_TYPE_F32, "base->type == GGML_TYPE_F32");
            ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
            ggml.impl.assert(mixes.ne[0] == 24, "mixes->ne[0] == 24");
            ggml.impl.assert(op.ne[0] == 4 and op.ne[1] == 4, "op->ne[0] == 4 && op->ne[1] == 4");

            var args: ggml.metal.kargs.dsv4_hc_comb = .{
                .n_tokens = @intCast(mixes.ne[1]),
                .n_iter = ggml.impl.getOpParamsI32(op, 1),
                .nb_m0 = mixes.nb[0],
                .nb_m1 = mixes.nb[1],
                .nb_s0 = scale.nb[0],
                .nb_b0 = base.nb[0],
                .nb_d0 = op.nb[0],
                .nb_d1 = op.nb[1],
                .nb_d2 = op.nb[2],
                .eps = ggml.impl.getOpParamsF32(op, 0),
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
            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(args.n_tokens + nsg - 1, nsg),
                1,
                1,
                32,
                nsg,
                1,
            );
        },
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_PRE => {
            const x = src(op, 0);
            const weights = src(op, 1);

            ggml.impl.assert(x.type == ggml.impl.ggml_c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
            ggml.impl.assert(weights.type == ggml.impl.ggml_c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
            ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

            var args: ggml.metal.kargs.dsv4_hc_pre = .{
                .n_embd = @intCast(x.ne[0]),
                .n_tokens = @intCast(x.ne[2]),
                .nb_x0 = x.nb[0],
                .nb_x1 = x.nb[1],
                .nb_x2 = x.nb[2],
                .nb_w0 = weights.nb[0],
                .nb_w1 = weights.nb[1],
                .nb_w2 = weights.nb[2],
                .nb_d0 = op.nb[0],
                .nb_d1 = op.nb[1],
                .scale = ggml.impl.getOpParamsF32(op, 0),
            };

            setBytes(enc, &args, 0);
            setBuffer(enc, x, 1);
            setBuffer(enc, weights, 2);
            setBuffer(enc, op, 3);

            const n_tiles = @divTrunc(args.n_embd + 31, 32);
            const nsg = @min(@as(c_int, 4), n_tiles);
            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(n_tiles + nsg - 1, nsg),
                args.n_tokens,
                1,
                32,
                nsg,
                1,
            );
        },
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST => {
            const x = src(op, 0);
            const residual = src(op, 1);
            const post = src(op, 2);
            const comb = srcOpt(op, 3);

            ggml.impl.assert(x.type == ggml.impl.ggml_c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
            ggml.impl.assert(residual.type == ggml.impl.ggml_c.GGML_TYPE_F32, "residual->type == GGML_TYPE_F32");
            ggml.impl.assert(post.type == ggml.impl.ggml_c.GGML_TYPE_F32, "post->type == GGML_TYPE_F32");
            ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
            ggml.impl.assert(residual.ne[1] == 4, "residual->ne[1] == 4");

            var args: ggml.metal.kargs.dsv4_hc_post = .{
                .n_embd = @intCast(x.ne[0]),
                .n_tokens = @intCast(x.ne[1]),
                .nb_x0 = x.nb[0],
                .nb_x1 = x.nb[1],
                .nb_r0 = residual.nb[0],
                .nb_r1 = residual.nb[1],
                .nb_r2 = residual.nb[2],
                .nb_p0 = post.nb[0],
                .nb_p1 = post.nb[1],
                .nb_c0 = if (comb) |t| t.nb[0] else 0,
                .nb_c1 = if (comb) |t| t.nb[1] else 0,
                .nb_c2 = if (comb) |t| t.nb[2] else 0,
                .nb_d0 = op.nb[0],
                .nb_d1 = op.nb[1],
                .nb_d2 = op.nb[2],
            };

            setBytes(enc, &args, 0);
            setBuffer(enc, x, 1);
            setBuffer(enc, residual, 2);
            setBuffer(enc, post, 3);
            if (comb) |t| {
                setBuffer(enc, t, 4);
                setBuffer(enc, op, 5);
            } else {
                setBuffer(enc, op, 4);
            }

            const n_tiles = @divTrunc(args.n_embd + 31, 32);
            const nsg = @min(@as(c_int, 4), n_tiles);
            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(n_tiles + nsg - 1, nsg),
                args.n_tokens,
                1,
                32,
                nsg,
                1,
            );
        },
        else => ggml.impl.abort("metal: no dsv4_hc encoder for this op"),
    }

    return 1;
}

/// Ports `ggml_metal_op_soft_max` (ggml-metal-ops.cpp:1698 @b1b770d90).
///
/// A `SOFT_MAX` opening the MoE routing pattern goes to
/// `ggml_metal_op_topk_moe` instead.
///
/// `src[1]` (mask) and `src[2]` (sinks) are both optional; when absent
/// the C binds `src[0]` in their slot rather than a null buffer.
pub fn ggml_metal_op_soft_max(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    if (ctx.useFusion()) {
        var n: c_int = 1;
        if (ctx.canFuse(idx, .full, &n)) |f| {
            if (ggml.metal.fusion.ggml_metal_fusion_get_id(f) == .topk_moe) {
                return ggml_metal_op_topk_moe(ctx, idx);
            }
        }
    }

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = srcOpt(op, 1);
    const s2 = srcOpt(op, 2);

    // The C `memcpy`s both out of `op_params` at indices 0 and 1.
    const scale = ggml.impl.getOpParamsF32(op, 0);
    const max_bias = ggml.impl.getOpParamsF32(op, 1);

    const n_head: u32 = @intCast(s0.ne[2]);
    const n_head_log2: i32 = @as(i32, 1) << @intCast(@as(u32, @intFromFloat(@floor(std.math.log2(@as(f32, @floatFromInt(n_head)))))));

    const m0 = std.math.pow(f32, 2.0, -(max_bias) / @as(f32, @floatFromInt(n_head_log2)));
    const m1 = std.math.pow(f32, 2.0, -(max_bias / 2.0) / @as(f32, @floatFromInt(n_head_log2)));

    // softmax

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);

    var args: ggml.metal.kargs.soft_max = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_soft_max(lib, op);

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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
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

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_ssm_conv` (ggml-metal-ops.cpp:1796 @b1b770d90).
///
/// Fuses a following SILU (`SSM_CONV_SILU`): the kernel applies it and
/// writes to the UNARY's destination. The kernel width `nc`
/// (`ne10`) is a function constant, not a kernel argument.
pub fn ggml_metal_op_ssm_conv(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const ne10: i32 = @intCast(s1.ne[0]);

    var n_fuse: c_int = 1;
    var use_silu = false;

    if (ctx.useFusion()) {
        var n: c_int = 1;
        if (ctx.canFuse(idx, .full, &n)) |f| {
            if (ggml.metal.fusion.ggml_metal_fusion_get_id(f) == .ssm_conv_silu) {
                n_fuse = n;
                use_silu = true;

                ctx.countFusions(f);
            }
        }
    }

    var args: ggml.metal.kargs.ssm_conv = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
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

    const bid_dst = getBufferId(if (n_fuse > 1) ctx.node(idx + n_fuse - 1) else op);

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

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE, ne10, use_silu);

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const n_token_batches = @divTrunc(ne1 + BATCH_SIZE - 1, BATCH_SIZE);
        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_ssm_conv(lib, op, ne10, use_silu);

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);
    }

    if (n_fuse > 1 and ctx.debugFusion() > 1) {
        ggml.impl.logDebug("%s: fuse: SSM_CONV + UNARY\n", .{"ggml_metal_op_ssm_conv"});
    }

    return n_fuse;
}

/// Ports `ggml_metal_op_ssm_scan` (ggml-metal-ops.cpp:1888 @b1b770d90).
///
/// Seven sources, and the `ns*` fields are stride *ratios* — `nb12/nb10`
/// and friends — element counts rather than byte counts.
///
/// Mamba-2 heads of `OP_SSM_SCAN_SSD_HD` take the chunked SSD kernel on
/// simdgroup matrices for whole chunks of `OP_SSM_SCAN_SSD_CS` tokens, and
/// the sequential kernel's `_tail` variant for what is left -- which always
/// includes the last `K` tokens when `K > 1`, since only the sequential
/// kernel writes the rollback snapshots.
pub fn ggml_metal_op_ssm_scan(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;
    const props_dev = ggml.metal.device_c.ggml_metal_device_get_props(ctx.dev);

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = src(op, 2);
    const src3 = src(op, 3);
    const src4 = src(op, 4);
    const src5 = src(op, 5);
    const src6 = src(op, 6);

    ggml.impl.assert(op.src[3] != null, "src3");
    ggml.impl.assert(op.src[4] != null, "src4");
    ggml.impl.assert(op.src[5] != null, "src5");
    ggml.impl.assert(op.src[6] != null, "src6");
    _ = src6;

    const d_state: i64 = s0.ne[0];
    const d_inner: i64 = s0.ne[1];
    const n_head: i64 = s0.ne[2];
    const n_group: i64 = src4.ne[1];
    const n_seq_tokens: i64 = s1.ne[2];
    const n_seqs: i64 = s1.ne[3];
    const K: i64 = ggml.impl.getOpParamsI32(op, 0);

    ggml.impl.assert(K >= 1, "K >= 1");
    ggml.impl.assert(
        ggml.types.ggml_nelements(s1) + K * d_state * d_inner * n_head * n_seqs == ggml.types.ggml_nelements(op),
        "ggml_nelements(op->src[1]) + K*d_state*d_inner*n_head*n_seqs == ggml_nelements(op)",
    );

    var args: ggml.metal.kargs.ssm_scan = .{
        .d_state = @intCast(d_state),
        .d_inner = @intCast(d_inner),
        .n_head = @intCast(n_head),
        .n_group = @intCast(n_group),
        .n_seq_tokens = @intCast(n_seq_tokens),
        .n_seq_tokens_total = @intCast(n_seq_tokens),
        .token_offset = 0,
        .n_seqs = @intCast(n_seqs),
        .K = @intCast(K),
        .s_off = @intCast(@as(usize, @intCast(ggml.types.ggml_nelements(s1))) * @sizeOf(f32)),
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

    const CHUNK: i64 = ggml.metal.impl_c.OP_SSM_SCAN_SSD_CS;

    const snap_reserve: i64 = if (K > 1) K else 0; // tokens reserved for sequential kernel rollback snapshots
    const mma_tokens: i64 = @divTrunc(n_seq_tokens - snap_reserve, CHUNK) * CHUNK; // largest multiple of CHUNK that leaves snap_reserve for the tail
    const use_mma =
        mma_tokens > 0 and
        src3.ne[0] == 1 and // checks that A tensor is set to scalar decay per head (A shape {1, n_head})
        props_dev.has_simdgroup_mm and // hardware check for M1 or newer
        @rem(d_state, 8) == 0 and // d_state must be multiple of 8 to align with simdgroup_float 8x8 tiles
        d_inner == ggml.metal.impl_c.OP_SSM_SCAN_SSD_HD; // mma kernel is specialized for the Mamba-2 head dim; this checks it

    // The C's `dispatch` lambda, which captures `args` by reference: the
    // tail reads the token count and offset set just before it.
    const Dispatch = struct {
        fn run(o: *const ggml.impl.ggml_c.ggml_tensor, e: *ggml.metal.device_c.Encoder, props: *const ggml.metal.device_c.DeviceProps, a: *const ggml.metal.kargs.ssm_scan, pipeline: ggml.metal.device_c.PipelineWithParams, nth: i64, n_tg_x: i64, nh: i64, ns: i64) void {
            ggml.impl.assert(nth <= maxThreads(pipeline), "nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)");
            ggml.impl.assert(pipeline.smem <= props.max_theadgroup_memory_size, "pipeline.smem <= props_dev->max_theadgroup_memory_size");

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(e, pipeline);
            setBytes(e, a, 0);
            for (0..7) |j| {
                setBuffer(e, src(o, j), @intCast(j + 1));
            }
            setBuffer(e, o, 8);
            ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(e, pipeline.smem, 0);
            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(e, @intCast(n_tg_x), @intCast(nh), @intCast(ns), @intCast(nth), 1, 1);
        }
    };

    if (!use_mma) {
        Dispatch.run(op, enc, props_dev, &args, ggml.metal.library.ggml_metal_library_get_pipeline_ssm_scan(lib, op, false), d_state, d_inner, n_head, n_seqs);
        return 1;
    }

    args.n_seq_tokens = mma_tokens;
    Dispatch.run(op, enc, props_dev, &args, ggml.metal.library.ggml_metal_library_get_pipeline_ssm_scan_ssd_mma(lib, op), ggml.metal.impl_c.OP_SSM_SCAN_SSD_NSG * 32, 1, n_head, n_seqs);

    if (mma_tokens < n_seq_tokens) {
        _ = concurrencyReset(ctx);

        args.n_seq_tokens = n_seq_tokens - mma_tokens;
        args.token_offset = mma_tokens;
        Dispatch.run(op, enc, props_dev, &args, ggml.metal.library.ggml_metal_library_get_pipeline_ssm_scan(lib, op, true), d_state, d_inner, n_head, n_seqs);
    }

    return 1;
}

/// Ports `ggml_metal_op_rwkv` (ggml-metal-ops.cpp:2021 @b1b770d90).
///
/// The one encoder with **no `kargs` struct**: it passes `B`, `T`, `C`
/// and `H` as four separate `set_bytes` of an `int64_t` each.
pub fn ggml_metal_op_rwkv(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const B: i64 = if (op.op == ggml.impl.ggml_c.GGML_OP_RWKV_WKV6) src(op, 5).ne[1] else src(op, 6).ne[1];
    const T: i64 = s0.ne[2];
    const C: i64 = op.ne[0];
    const H: i64 = s0.ne[1];

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_rwkv(lib, op);

    var ida: c_int = 0;

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    for (0..6) |j| {
        setBuffer(enc, src(op, j), ida);
        ida += 1;
    }
    if (op.op == ggml.impl.ggml_c.GGML_OP_RWKV_WKV7) {
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

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_gated_delta_net` (ggml-metal-ops.cpp:2062
/// @b1b770d90).
///
/// Fused with the `CPY` that follows it (`GDN_CACHE` in `fusion.zig`),
/// the kernel writes the state snapshots straight into the recurrent
/// cache and the copy is skipped: `nb_out` is then the cache's stride in
/// floats, and 0 keeps the unfused layout, snapshots after the scores.
pub fn ggml_metal_op_gated_delta_net(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const use_fusion = ctx.useFusion();
    const debug_fusion = ctx.debugFusion();

    const s0 = src(op, 0);
    const s1 = src(op, 1);
    const s2 = src(op, 2);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    // when fused with the trailing cache cpy, the snapshots are written straight into the
    // recurrent cache and the cpy is skipped (see GGML_METAL_FUSION_GDN_CACHE)
    var bid_out = getBufferId(op);
    var nb_out: u64 = 0;
    var n_fuse: c_int = 1;

    if (use_fusion) {
        var n: c_int = 1;
        const fus = ctx.canFuse(idx, .full, &n);

        if (fus != null and ggml.metal.fusion.ggml_metal_fusion_get_id(fus.?) == .gdn_cache) {
            const dst_cache = src(ctx.node(idx + 1), 1); // cache view

            bid_out = getBufferId(dst_cache);
            nb_out = dst_cache.nb[2] / @sizeOf(f32);
            n_fuse = 2;

            ctx.countFusions(fus.?);

            if (debug_fusion > 1) {
                ggml.impl.logDebug("%s: fuse: GATED_DELTA_NET + CPY\n", .{"ggml_metal_op_gated_delta_net"});
            }
        }
    }

    var ida: c_int = 0;

    var args: ggml.metal.kargs.gated_delta_net = .{
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
        .nb_out = nb_out,
    };

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, ida);
    ida += 1;
    // q, k, v, gate, beta, state, dst (attn), state_out -- the C's own labels
    for (0..6) |j| {
        setBuffer(enc, src(op, j), ida);
        ida += 1;
    }
    setBuffer(enc, op, ida);
    ida += 1;
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_out, ida);
    ida += 1;

    const nsg = pipeline.nsg;

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @intCast(@divTrunc(s2.ne[0], nsg)),
        @intCast(s2.ne[1]),
        @intCast(s2.ne[3]),
        32,
        nsg,
        1,
    );

    return n_fuse;
}

/// Ports `ggml_metal_op_solve_tri` (ggml-metal-ops.cpp:2166 @b1b770d90).
pub fn ggml_metal_op_solve_tri(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    var args: ggml.metal.kargs.solve_tri = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    const nsg = pipeline.nsg;

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    const ne10: c_int = @intCast(s1.ne[0]);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_try_fusion` (ggml-metal-ops.cpp:2579 @b1b770d90):
/// the number of nodes from `idx` on that the fusion table fuses as
/// pattern `id` into one dispatch, 1 if it fuses none.
fn tryFusion(ctx: *Op, idx: c_int, id: ggml.metal.fusion.Id) c_int {
    if (!ctx.useFusion()) {
        return 1;
    }

    var n: c_int = 1;
    const fus = ctx.canFuse(idx, .full, &n) orelse return 1;
    if (ggml.metal.fusion.ggml_metal_fusion_get_id(fus) != id) {
        return 1;
    }

    ctx.countFusions(fus);

    if (ctx.debugFusion() > 1) {
        ggml.impl.logDebug("%s: fuse: %s to %s, %d nodes\n", .{
            "ggml_metal_op_try_fusion",
            ggml.types.ggml_op_name(ctx.node(idx).op),
            ggml.types.ggml_op_name(ctx.node(idx + n - 1).op),
            n,
        });
    }

    return n;
}

/// Ports `ggml_metal_op_mul_mat_mma` (ggml-metal-ops.cpp:2599 @b1b770d90).
///
/// The few-row MMA mat-mul (`mul_mv_mma.metal`). When the
/// fusion table fuses the `ADD` after this node (`MUL_MAT_ADD`), the
/// kernel adds the residual in its store: `dst` becomes the `ADD`, and the
/// residual is the `ADD`'s other operand, chosen by identity with this
/// node, because the residual can itself be a `MUL_MAT`
/// output, which a test on `op` would mistake for this one. Buffer 4 is
/// bound either way -- to `dst` itself when nothing is fused.
fn mulMatMma(ctx: *Op, idx: c_int) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    // GGML_TENSOR_LOCALS(int32_t, ...) narrows each ne
    const ne00: i32 = @truncate(s0.ne[0]);
    const ne01: i32 = @truncate(s0.ne[1]);
    const ne02: i32 = @truncate(s0.ne[2]);
    const ne03: i32 = @truncate(s0.ne[3]);
    const ne10: i32 = @truncate(s1.ne[0]);
    const ne11: i32 = @truncate(s1.ne[1]);
    const ne12: i32 = @truncate(s1.ne[2]);
    const ne13: i32 = @truncate(s1.ne[3]);

    // the MMA store adds the residual when the table fuses the ADD after this mat-mul
    const n_fuse = tryFusion(ctx, idx, .mul_mat_add);
    const fuse_add = n_fuse > 1;

    var dst: *const ggml.impl.ggml_c.ggml_tensor = op;
    var res: *const ggml.impl.ggml_c.ggml_tensor = dst;

    if (fuse_add) {
        dst = ctx.node(idx + n_fuse - 1);
        // the residual is the other operand of the ADD, by identity: it can itself be a MUL_MAT output
        res = if (dst.src[0] == op) src(dst, 1) else src(dst, 0);
    }

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_mul_mv_mma_auto(lib, op, fuse_add);

    var args: ggml.metal.kargs.mul_mv_ext = .{
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
        .ne0 = @truncate(op.ne[0]),
        .ne1 = @truncate(op.ne[1]),
        .r2 = @truncate(@divTrunc(ne12, ne02)),
        .r3 = @truncate(@divTrunc(ne13, ne03)),
    };

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, dst, 3);
    setBuffer(enc, res, 4);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    const rows0 = pipeline.nr0;
    const rows1 = pipeline.nr1;

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
        enc,
        @divTrunc(ne01 + rows0 - 1, rows0),
        @divTrunc(ne11 + rows1 - 1, rows1),
        ne12 * ne13,
        32,
        pipeline.nsg,
        1,
    );

    return n_fuse;
}

/// Ports `ggml_metal_op_mul_mat_mv` (ggml-metal-ops.cpp:2665 @b1b770d90):
/// the generic mat-vec kernel, `ggml_metal_op_mul_mat`'s last arm.
fn mulMatMv(ctx: *Op, idx: c_int) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const ne00: i32 = @truncate(s0.ne[0]);
    const ne01: i32 = @truncate(s0.ne[1]);
    const ne02: i32 = @truncate(s0.ne[2]);
    const ne03: i32 = @truncate(s0.ne[3]);
    const ne10: i32 = @truncate(s1.ne[0]);
    const ne11: i32 = @truncate(s1.ne[1]);
    const ne12: i32 = @truncate(s1.ne[2]);
    const ne13: i32 = @truncate(s1.ne[3]);

    const r2: i16 = @truncate(@divTrunc(ne12, ne02));
    const r3: i16 = @truncate(@divTrunc(ne13, ne03));

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_mul_mv(lib, op);

    const nr0 = pipeline.nr0;
    const nr1 = pipeline.nr1;
    const nsg = pipeline.nsg;

    const smem = pipeline.smem;

    var args: ggml.metal.kargs.mul_mv = .{
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
        .ne0 = @truncate(op.ne[0]),
        .ne1 = @truncate(op.ne[1]),
        .nr0 = nr0,
        .r2 = r2,
        .r3 = r3,
    };

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    if (s0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_BF16 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q8_0)
    {
        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @divTrunc(ne01 + nr0 - 1, nr0),
            @divTrunc(ne11 + nr1 - 1, nr1),
            ne12 * ne13,
            32,
            nsg,
            1,
        );
    } else {
        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @divTrunc(ne01 + nr0 * nsg - 1, nr0 * nsg),
            @divTrunc(ne11 + nr1 - 1, nr1),
            ne12 * ne13,
            32,
            nsg,
            1,
        );
    }

    return 1;
}

/// Ports `ggml_metal_op_mul_mat` (ggml-metal-ops.cpp:2730 @b1b770d90).
///
/// Five kernels behind one op, chosen in order: a Hadamard hint goes to
/// `fwht`; a few F32 `src1` rows go to the MMA kernels
/// (`mulMatMma`, which may fuse the following `ADD`); then the
/// small-batch `mul_mv_ext` for `ne11` in [2, 8] on a 128-aligned row,
/// the simdgroup-matrix `mul_mm` where `ggml_metal_op_mul_mat_use_mm`
/// says, else the row-at-a-time `mul_mv` (`mulMatMv`).
pub fn ggml_metal_op_mul_mat(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const props_dev = ggml.metal.device_c.ggml_metal_device_get_props(ctx.dev);

    if (ggml.metal.common.ggml_metal_op_mul_mat_use_fwht(op, props_dev.max_theadgroup_memory_size)) {
        return ggml_metal_op_fwht(ctx, idx);
    }

    if (props_dev.has_simdgroup_mm and ggml.metal.common.ggml_metal_op_mul_mat_use_mma(op)) {
        return mulMatMma(ctx, idx);
    }

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);
    const ne10: c_int = @intCast(s1.ne[0]);
    const ne11: c_int = @intCast(s1.ne[1]);
    const ne12: c_int = @intCast(s1.ne[2]);
    const ne13: c_int = @intCast(s1.ne[3]);

    ggml.impl.assert(ne00 == ne10, "ne00 == ne10");

    ggml.impl.assert(@rem(ne12, ne02) == 0, "ne12 % ne02 == 0");
    ggml.impl.assert(@rem(ne13, ne03) == 0, "ne13 % ne03 == 0");

    const r2: i16 = @intCast(@divTrunc(ne12, ne02));
    const r3: i16 = @intCast(@divTrunc(ne13, ne03));

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    //
    // the C writes this as one nested condition; it is split into its
    // two type groups here, which are the only things that differ
    // between them besides the `ne11` window.
    const t0_bs2 = s0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or // TODO: helper function
        s0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_BF16 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q1_0 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q2_0 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q4_0 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q4_1 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q5_0 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q5_1 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q8_0 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_MXFP4 or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_IQ4_NL;

    const t0_bs4 = s0.type == ggml.impl.ggml_c.GGML_TYPE_Q4_K or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q5_K or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q6_K or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q2_K or
        s0.type == ggml.impl.ggml_c.GGML_TYPE_Q3_K;

    const small_batch = s1.type == ggml.impl.ggml_c.GGML_TYPE_F32 and (@rem(ne00, 128) == 0) and
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
            else => ggml.impl.abort("unsupported ne11"),
        }

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg);

        var args: ggml.metal.kargs.mul_mv_ext = .{
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @divTrunc(ne01 + r0ptg - 1, r0ptg),
            @divTrunc(ne11 + r1ptg - 1, r1ptg),
            ne12 * ne13,
            32,
            nsg,
            1,
        );
    } else if (ggml.metal.common.ggml_metal_op_mul_mat_use_mm(op, props_dev.has_simdgroup_mm)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        // (the C leaves an alignment assert commented out here)

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_mul_mm(lib, op);

        var args: ggml.metal.kargs.mul_mm = .{
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        const smem = pipeline.smem;

        ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const nr0 = pipeline.nr0;
        const nr1 = pipeline.nr1;
        const nsg = pipeline.nsg;

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
            enc,
            @divTrunc(ne11 + nr1 - 1, nr1),
            @divTrunc(ne01 + nr0 - 1, nr0),
            ne12 * ne13,
            32,
            nsg,
            1,
        );
    } else {
        return mulMatMv(ctx, idx);
    }

    return 1;
}

/// Ports `ggml_metal_op_mul_mat_id` (ggml-metal-ops.cpp:2951 @b1b770d90).
///
/// Where `ggml_metal_op_mul_mat_id_use_mm` says, it builds an expert->token id map
/// into scratch after `dst`, barriers, then runs the simdgroup-matrix
/// kernel over it; below that it falls to `mul_mv_id`, one row at a time.
/// The map ("fix mul_mat_id with duplicate ids") is a list of
/// 32-entry tiles, each naming its expert, and the matrix kernel's grid is
/// one threadgroup per tile up to `mulMatIdMaxTiles` rather than one per
/// (32 tokens, expert); `ne02` reaches the map kernel as a function
/// constant instead of an argument.
///
/// An id may be `-1`: the row belongs to no expert, the map
/// skips it, and `kernel_mul_mm_id_zero` zeroes it beside the map, ahead of
/// the barrier -- the rows it writes are disjoint from the matrix kernel's.
pub fn ggml_metal_op_mul_mat_id(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const props_dev = ggml.metal.device_c.ggml_metal_device_get_props(ctx.dev);

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
    ggml.impl.assert(s2.type == ggml.impl.ggml_c.GGML_TYPE_I32, "op->src[2]->type == GGML_TYPE_I32");

    ggml.impl.assert(!ggml.types.ggml_is_transposed(s0), "!ggml_is_transposed(op->src[0])");
    ggml.impl.assert(!ggml.types.ggml_is_transposed(s1), "!ggml_is_transposed(op->src[1])");

    ggml.impl.assert(ne03 == 1, "ne03 == 1");
    ggml.impl.assert(ne13 == 1, "ne13 == 1");

    const bid_src0 = getBufferId(s0);
    const bid_src1 = getBufferId(s1);
    const bid_src2 = getBufferId(s2);
    const bid_dst = getBufferId(op);

    const r2: u32 = 1;
    const r3: u32 = 1;

    if (ggml.metal.common.ggml_metal_op_mul_mat_id_use_mm(op, props_dev.has_simdgroup_mm)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        // (the C leaves an alignment assert commented out here)

        // extra buffers for intermediate id mapping
        var bid_tpe = bid_dst;
        bid_tpe.offs += @intCast(ggml.types.ggml_nbytes(op));

        var bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            // the C initialises this one positionally
            var args: ggml.metal.kargs.mul_mm_id_map0 = .{
                .ne10 = ne10,
                .ne11 = ne11, // n_expert_used (bcast)
                .nb11 = s1.nb[1],
                .nb12 = s1.nb[2],
                .ne21 = ne21, // n_tokens
                .ne20 = ne20, // n_expert_used
                .nb21 = s2.nb[1],
            };

            const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const smem = pipeline.smem;

            ggml.impl.assert(ne02 <= maxThreads(pipeline), "ne02 <= max_theads_per_threadgroup");

            ggml.impl.assert(smem <= props_dev.max_theadgroup_memory_size, "smem <= max_theadgroup_memory_size");

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
            setBytes(enc, &args, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src2, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tpe, 2);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_ids, 3);

            ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        // zero the rows that no expert owns - disjoint from what kernel_mul_mm_id writes
        {
            var args: ggml.metal.kargs.mul_mm_id_zero = .{
                .ne0 = @intCast(op.ne[0]),
                .nb1 = op.nb[1],
                .nb2 = op.nb[2],
                .nb21 = s2.nb[1],
            };

            const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_mul_mm_id_zero(lib);

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
            setBytes(enc, &args, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src2, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ne20, ne21, 1, 32, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        _ = concurrencyReset(ctx);

        {
            const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            var args: ggml.metal.kargs.mul_mm_id = .{
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
                .ne0 = @intCast(op.ne[0]),
                .ne1 = @intCast(op.ne[1]),
                .r2 = r2,
                .r3 = r3,
            };

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
            setBytes(enc, &args, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tpe, 3);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_ids, 4);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 5);

            const smem = pipeline.smem;

            ggml.impl.assert(smem <= props_dev.max_theadgroup_memory_size, "smem <= props_dev->max_theadgroup_memory_size");

            ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            // the C's `(int)` cast of an `int64_t`
            const max_tiles: c_int = @truncate(mulMatIdMaxTiles(op));

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, max_tiles, @divTrunc(ne01 + 63, 64), 1, 128, 1, 1);
        }
    } else {
        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const nr0 = pipeline.nr0;
        const nr1 = pipeline.nr1;
        const nsg = pipeline.nsg;

        const smem = pipeline.smem;

        var args: ggml.metal.kargs.mul_mv_id = .{
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

        if (ggml.types.ggml_is_quantized(s0.type)) {
            ggml.impl.assert(ne00 >= nsg * nr0, "ne00 >= nsg*nr0");
        }

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 3);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const _ne1: i64 = 1;
        const ne123: i64 = @as(i64, ne20) * @as(i64, ne21);

        ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (s0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or
            s0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or
            s0.type == ggml.impl.ggml_c.GGML_TYPE_BF16 or
            s0.type == ggml.impl.ggml_c.GGML_TYPE_Q8_0)
        {
            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
                enc,
                @divTrunc(ne01 + nr0 - 1, nr0),
                @intCast(@divTrunc(_ne1 + nr1 - 1, nr1)),
                @intCast(ne123),
                32,
                nsg,
                1,
            );
        } else {
            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_get_rows` (ggml-metal-ops.cpp:1231 @b1b770d90).
pub fn ggml_metal_op_get_rows(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_get_rows(lib, s0.type);

    const ne00: i32 = @intCast(s0.ne[0]);

    var args: ggml.metal.kargs.get_rows = .{
        .ne00t = if (ggml.types.ggml_is_quantized(s0.type)) @divTrunc(ne00, 16) else ne00,
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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_set_rows` (ggml-metal-ops.cpp:1276 @b1b770d90).
pub fn ggml_metal_op_set_rows(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_set_rows(lib, op);

    const nk0: i32 = @intCast(@divTrunc(op.ne[0], ggml.types.ggml_blck_size(op.type)));

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

    var args: ggml.metal.kargs.set_rows = .{
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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    setBuffer(enc, op, 3);

    const ne01: c_int = @intCast(s0.ne[1]);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_diag` (ggml-metal-ops.cpp:1338 @b1b770d90).
pub fn ggml_metal_op_diag(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    var args: ggml.metal.kargs.diag = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_diag(lib, op);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_l2_norm` (ggml-metal-ops.cpp:4522 @b1b770d90).
pub fn ggml_metal_op_l2_norm(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const bid_src0 = getBufferId(s0);
    const bid_dst = getBufferId(op);

    const eps = ggml.impl.getOpParamsF32(op, 0);

    const ne00: i32 = @intCast(s0.ne[0]);

    var args: ggml.metal.kargs.l2_norm = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_l2_norm(lib, op);

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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_group_norm` (ggml-metal-ops.cpp:4590 @b1b770d90).
///
/// The kernel takes the source's batch stride and all
/// of `dst`'s, and one threadgroup row per batch (`ne03`), so a source with
/// `ne03 > 1` or non-contiguous rows normalises correctly (#31852).
pub fn ggml_metal_op_group_norm(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ngrp = ggml.impl.getOpParamsI32(op, 0);

    // The C reads `eps` from `op_params + 1`, so index 1.
    const eps = ggml.impl.getOpParamsF32(op, 1);

    var args: ggml.metal.kargs.group_norm = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .nb00 = s0.nb[0],
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .nb1 = op.nb[1],
        .nb2 = op.nb[2],
        .nb3 = op.nb[3],
        .ngrp = ngrp,
        .eps = eps,
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_group_norm(lib, op);

    // The C leaves the usual widen-to-SIMD loop commented out here and
    // keeps `nth` at one SIMD width; not reproduced beyond this note.
    const nth: c_int = 32; // SIMD width

    const smem = pipeline.smem;

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, @intCast(s0.ne[3]), nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_norm` (ggml-metal-ops.cpp:4645 @b1b770d90).
///
/// The first **fusion** encoder: it may swallow a following `MUL` and
/// then an `ADD` into one kernel, or a `SCALE`, whose
/// factor rides in `args.scale` to the `_use_scale` specialisation of the
/// MUL kernel; it returns how many nodes it consumed. The `nef*`/`nbf*` array fields carry one entry per fused
/// operand, which is why `kargs_norm` has `[3]` arrays where every other
/// struct has scalars.
pub fn ggml_metal_op_norm(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const use_fusion = ctx.useFusion();

    const debug_fusion = ctx.debugFusion();

    const s0 = src(op, 0);

    const eps = ggml.impl.getOpParamsF32(op, 0);

    const bid_src0 = getBufferId(s0);
    var bid_dst = getBufferId(op);

    const ne00: i32 = @intCast(s0.ne[0]);

    var args: ggml.metal.kargs.norm = .{
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
        .scale = 1.0,
    };

    var n_fuse: c_int = 1;
    var fused_norm_scale = false;

    var bid_fuse = [2]ggml.metal.device_c.BufferId{ bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b) or scale(d[0])
    // d[2] = add(d[1], c)
    if (use_fusion) {
        var n: c_int = 1;
        const fus = ctx.canFuse(idx, .full, &n);

        if (fus != null and (ggml.metal.fusion.ggml_metal_fusion_get_id(fus.?) == .norm_mul or ggml.metal.fusion.ggml_metal_fusion_get_id(fus.?) == .norm_mul_add)) {
            n_fuse = n;

            ctx.countFusions(fus.?);

            var i: c_int = 1;
            while (i < n_fuse) : (i += 1) {
                const fn1 = src(ctx.node(idx + i), 1);
                const k: usize = @intCast(i);

                bid_fuse[k - 1] = getBufferId(fn1);

                args.nef1[k] = @intCast(fn1.ne[1]);
                args.nef2[k] = @intCast(fn1.ne[2]);
                args.nef3[k] = @intCast(fn1.ne[3]);

                args.nbf1[k] = fn1.nb[1];
                args.nbf2[k] = fn1.nb[2];
                args.nbf3[k] = fn1.nb[3];
            }

            if (debug_fusion > 1) {
                if (n_fuse == 2) {
                    ggml.impl.logDebug("%s: fuse: %s + MUL\n", .{ "ggml_metal_op_norm", ggml.types.ggml_op_name(op.op) });
                }
                if (n_fuse == 3) {
                    ggml.impl.logDebug("%s: fuse: %s + MUL + ADD\n", .{ "ggml_metal_op_norm", ggml.types.ggml_op_name(op.op) });
                }
            }
        }

        if (fus != null and ggml.metal.fusion.ggml_metal_fusion_get_id(fus.?) == .norm_scale) {
            n_fuse = n;
            fused_norm_scale = true;

            ctx.countFusions(fus.?);

            const scale_node = ctx.node(idx + 1);
            args.scale = ggml.impl.getOpParamsF32(scale_node, 0);

            if (debug_fusion > 1) {
                ggml.impl.logDebug("%s: fuse: %s + SCALE\n", .{ "ggml_metal_op_norm", ggml.types.ggml_op_name(op.op) });
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

    const pipeline = if (fused_norm_scale)
        ggml.metal.library.ggml_metal_library_get_pipeline_norm_scale(lib, op)
    else
        ggml.metal.library.ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    var nth: c_int = 32; // SIMD width

    while (nth < args.ne00_t and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    nth = @min(nth, maxThreads(pipeline));
    nth = @min(nth, @divTrunc(args.ne00_t + 31, 32) * 32);

    const smem = pipeline.smem;

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_fuse[0], 2);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_fuse[1], 3);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 4);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_rope` (ggml-metal-ops.cpp:4779 @b1b770d90).
pub fn ggml_metal_op_rope(ctx: *Op, idx: c_int) callconv(.c) c_int {
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
    ggml.impl.assert(@rem(ne10, ne02) == 0, "ne10 % ne02 == 0");
    ggml.impl.assert(ne10 >= ne02, "ne10 >= ne02");

    const nth = @min(@as(c_int, 1024), ne00);

    const n_past = ggml.impl.getOpParamsI32(op, 0);
    const n_dims = ggml.impl.getOpParamsI32(op, 1);
    //const mode = impl.getOpParamsI32(op, 2);
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const n_ctx_orig = ggml.impl.getOpParamsI32(op, 4);

    const freq_base = ggml.impl.getOpParamsF32(op, 5);
    const freq_scale = ggml.impl.getOpParamsF32(op, 6);
    const ext_factor = ggml.impl.getOpParamsF32(op, 7);
    const attn_factor = ggml.impl.getOpParamsF32(op, 8);
    const beta_fast = ggml.impl.getOpParamsF32(op, 9);
    const beta_slow = ggml.impl.getOpParamsF32(op, 10);

    // mrope
    const sect_0 = ggml.impl.getOpParamsI32(op, 11);
    const sect_1 = ggml.impl.getOpParamsI32(op, 12);
    const sect_2 = ggml.impl.getOpParamsI32(op, 13);
    const sect_3 = ggml.impl.getOpParamsI32(op, 14);

    const n_offs = ggml.impl.getOpParamsI32(op, 15);

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const inplace = op.data == s0.data;

    var args: ggml.metal.kargs.rope = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_rope(lib, op);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, s1, 2);
    if (s2) |t| {
        setBuffer(enc, t, 3);
    } else {
        setBuffer(enc, s0, 3);
    }
    setBuffer(enc, op, 4);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_im2col` (ggml-metal-ops.cpp:4882 @b1b770d90).
pub fn ggml_metal_op_im2col(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const s0 = ggml.impl.getOpParamsI32(op, 0);
    const s1 = ggml.impl.getOpParamsI32(op, 1);
    const p0 = ggml.impl.getOpParamsI32(op, 2);
    const p1 = ggml.impl.getOpParamsI32(op, 3);
    const d0 = ggml.impl.getOpParamsI32(op, 4);
    const d1 = ggml.impl.getOpParamsI32(op, 5);

    const is_2D = ggml.impl.getOpParamsI32(op, 6) == 1;

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

    var args: ggml.metal.kargs.im2col = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_im2col(lib, op);

    if (KH * KW <= maxThreads(pipeline)) {
        const ntptg0: u64 = @min(
            @as(u64, @intCast(@divTrunc(maxThreads(pipeline), KH * KW))),
            @as(u64, @intCast(N)),
        );

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, src1, 1);
        setBuffer(enc, op, 2);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, @intCast(ntptg0), KH, KW);
    } else {
        const n_threads: u64 = @min(@as(u64, @intCast(maxThreads(pipeline))), @as(u64, @intCast(N)));
        const quotient: i64 = @divTrunc(@as(i64, N), @as(i64, @intCast(n_threads))) +
            (if (@rem(@as(i64, N), @as(i64, @intCast(n_threads))) > 0) @as(i64, 1) else 0);

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, src1, 1);
        setBuffer(enc, op, 2);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_conv_2d` (ggml-metal-ops.cpp:4962 @b1b770d90).
pub fn ggml_metal_op_conv_2d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    ggml.impl.assert(ggml.types.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
    ggml.impl.assert(
        src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        "op->src[0]->type is f16 or f32",
    );

    const s0 = ggml.impl.getOpParamsI32(op, 0);
    const s1 = ggml.impl.getOpParamsI32(op, 1);
    const p0 = ggml.impl.getOpParamsI32(op, 2);
    const p1 = ggml.impl.getOpParamsI32(op, 3);
    const d0 = ggml.impl.getOpParamsI32(op, 4);
    const d1 = ggml.impl.getOpParamsI32(op, 5);

    var args: ggml.metal.kargs.conv_2d = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_conv_2d(lib, op);

    var nth = maxThreads(pipeline);
    nth = @min(nth, 256);
    nth = @max(nth, 1);

    const n_out: u64 = @intCast(ggml.types.ggml_nelements(op));

    var tg: u64 = @divTrunc(n_out + @as(u64, @intCast(nth)) - 1, @as(u64, @intCast(nth)));
    tg = @max(tg, 1);
    tg = @min(tg, @as(u64, std.math.maxInt(c_int)));

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(tg), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_conv_2d_dw` (ggml-metal-ops.cpp:5040 @b1b770d90).
///
/// **The C writes `/*.nb02 =*/ nb03` here** — the `nb02` field is fed the
/// weight tensor's `nb[3]`, not its `nb[2]`. Reproduced as written. Which
/// it is cannot be settled from this side: the kernel reads the field by
/// offset, so if this is an upstream slip it is one the kernel has been
/// compiled against. `make node-diff ARGS=--gpu` would be the arbiter,
/// and `CONV_2D_DW` is not in a Qwen3.5 graph.
pub fn ggml_metal_op_conv_2d_dw(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    ggml.impl.assert(op.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");
    ggml.impl.assert(
        src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        "op->src[0]->type is f16 or f32",
    );

    const s0 = ggml.impl.getOpParamsI32(op, 0);
    const s1 = ggml.impl.getOpParamsI32(op, 1);
    const p0 = ggml.impl.getOpParamsI32(op, 2);
    const p1 = ggml.impl.getOpParamsI32(op, 3);
    const d0 = ggml.impl.getOpParamsI32(op, 4);
    const d1 = ggml.impl.getOpParamsI32(op, 5);

    var args: ggml.metal.kargs.conv_2d_dw = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_conv_transpose_1d` (ggml-metal-ops.cpp:5191
/// @b1b770d90).
pub fn ggml_metal_op_conv_transpose_1d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    // `s0` is the C's name for the stride op-param; the tensors are
    // reached through `src0`/`src1` here so the two do not collide.
    const s0 = ggml.impl.getOpParamsI32(op, 0);

    const IC: i32 = @intCast(src1.ne[1]);
    const IL: i32 = @intCast(src1.ne[0]);

    const K: i32 = @intCast(src0.ne[0]);

    const OL: i32 = @intCast(op.ne[0]);
    const OC: i32 = @intCast(op.ne[1]);

    var args: ggml.metal.kargs.conv_transpose_1d = .{
        .IC = IC,
        .IL = IL,
        .K = K,
        .s0 = s0,
        .nb0 = op.nb[0],
        .nb1 = op.nb[1],
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_conv_transpose_2d` (ggml-metal-ops.cpp:5326
/// @b1b770d90).
///
/// One threadgroup per output pixel and channel, for each of the `N`
/// images in the batch.
pub fn ggml_metal_op_conv_transpose_2d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const s0 = ggml.impl.getOpParamsI32(op, 0);

    const IC: i32 = @intCast(src1.ne[2]);
    const IH: i32 = @intCast(src1.ne[1]);
    const IW: i32 = @intCast(src1.ne[0]);

    const KH: i32 = @intCast(src0.ne[1]);
    const KW: i32 = @intCast(src0.ne[0]);

    const OW: i32 = @intCast(op.ne[0]);
    const OH: i32 = @intCast(op.ne[1]);
    const OC: i32 = @intCast(op.ne[2]);
    const N: i32 = @intCast(src(op, 1).ne[3]);

    var args: ggml.metal.kargs.conv_transpose_2d = .{
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
        .nb3 = op.nb[3],
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    // Metal requires buffer size to be multiple of 16 bytes
    const smem = stdlib.num.pad(usize, @as(usize, @intCast(KW)) * @as(usize, @intCast(KH)) * @sizeOf(f32), 16);
    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC * N, KW, KH, 1);

    return 1;
}

/// Ports `ggml_metal_op_col2im_1d` (ggml-metal-ops.cpp:5236 @b1b770d90).
pub fn ggml_metal_op_col2im_1d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);

    const s0 = ggml.impl.getOpParamsI32(op, 0);
    const OC = ggml.impl.getOpParamsI32(op, 1);
    const p0 = ggml.impl.getOpParamsI32(op, 2);

    const K_OC: i32 = @intCast(src0.ne[0]);
    const T_in: i32 = @intCast(src0.ne[1]);
    const K: i32 = @divTrunc(K_OC, OC);
    const T_out: i32 = @intCast(op.ne[0]);

    var args: ggml.metal.kargs.col2im_1d = .{
        .T_in = T_in,
        .T_out = T_out,
        .OC = OC,
        .K = K,
        .K_OC = K_OC,
        .s0 = s0,
        .p0 = p0,
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    const total = T_out * OC;
    const nth: i32 = 256;
    const ntg = @divTrunc(total + nth - 1, nth);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_conv_3d` (ggml-metal-ops.cpp:5120 @b1b770d90).
///
/// The C's initialiser is positional from `s0` onwards rather than
/// designated; the field names here come from the struct's declaration
/// order, which `kargs.zig` carries from the header.
pub fn ggml_metal_op_conv_3d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    // 2. Extract hyperparams from op_params
    const s0 = ggml.impl.getOpParamsI32(op, 0);
    const s1 = ggml.impl.getOpParamsI32(op, 1);
    const s2 = ggml.impl.getOpParamsI32(op, 2);
    const p0 = ggml.impl.getOpParamsI32(op, 3);
    const p1 = ggml.impl.getOpParamsI32(op, 4);
    const p2 = ggml.impl.getOpParamsI32(op, 5);
    const d0 = ggml.impl.getOpParamsI32(op, 6);
    const d1 = ggml.impl.getOpParamsI32(op, 7);
    const d2 = ggml.impl.getOpParamsI32(op, 8);
    const IC = ggml.impl.getOpParamsI32(op, 9);
    const N = ggml.impl.getOpParamsI32(op, 10);
    const OC = ggml.impl.getOpParamsI32(op, 11);

    // 3. Build the parameter struct
    var args: ggml.metal.kargs.conv_3d = .{
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
    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    const nth0: c_int = 32; // Standard SIMD width for Apple Silicon
    const nth1: c_int = 1;
    const nth2: c_int = 1;

    const spatial_volume: i64 = @as(i64, args.OW) * @as(i64, args.OH) * @as(i64, args.OD);

    const ntg0: c_int = @intCast(@divTrunc(spatial_volume + nth0 - 1, nth0));
    const ntg1: c_int = args.OC;
    const ntg2: c_int = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, src1, 2);
    setBuffer(enc, op, 3);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    return 1;
}

/// Ports `ggml_metal_op_upscale` (ggml-metal-ops.cpp:5384 @b1b770d90).
pub fn ggml_metal_op_upscale(ctx: *Op, idx: c_int) callconv(.c) c_int {
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

    const mode_flags = ggml.impl.getOpParamsI32(op, 0);

    var poffs: f32 = 0.5;

    if ((mode_flags & ggml.impl.ggml_c.GGML_SCALE_FLAG_ALIGN_CORNERS) != 0) {
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

    var args: ggml.metal.kargs.upscale = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_upscale(lib, op);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(ne0)));

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_pad` (ggml-metal-ops.cpp:5501 @b1b770d90).
pub fn ggml_metal_op_pad(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ne00: i64 = s0.ne[0];
    const ne0: i64 = op.ne[0];

    const args: ggml.metal.kargs.pad = .{
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
        .lp0 = ggml.impl.getOpParamsI32(op, 0),
        .lp1 = ggml.impl.getOpParamsI32(op, 2),
        .lp2 = ggml.impl.getOpParamsI32(op, 4),
        .lp3 = ggml.impl.getOpParamsI32(op, 6),
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_pad(lib, op);

    // `kargs_pad` is one of the structs whose `ne` fields are `int64_t`,
    // where the C's `GGML_TENSOR_LOCALS` locals are `int32_t` — it widens
    // on the way in and truncates back to `int` here. Both narrowings are
    // written out rather than left implicit.
    const nth_max = @min(@as(c_int, 64), maxThreads(pipeline));
    const nth: c_int = @min(@as(c_int, @intCast(args.ne0)), nth_max);
    const nk0: c_int = @divTrunc(@as(c_int, @intCast(args.ne0)) + 1024 - 1, 1024); // note: 1024 is hardcoded in the kernel!

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_pad_reflect_1d` (ggml-metal-ops.cpp:5551
/// @b1b770d90).
pub fn ggml_metal_op_pad_reflect_1d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    var args: ggml.metal.kargs.pad_reflect_1d = .{
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
        .p0 = ggml.impl.getOpParamsI32(op, 0),
        .p1 = ggml.impl.getOpParamsI32(op, 1),
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const nth = @min(@as(c_int, 1024), @as(c_int, @intCast(op.ne[0])));

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_roll` (ggml-metal-ops.cpp:5448 @b1b770d90).
pub fn ggml_metal_op_roll(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);

    const s0 = ggml.impl.getOpParamsI32(op, 0);
    const s1 = ggml.impl.getOpParamsI32(op, 1);
    const s2 = ggml.impl.getOpParamsI32(op, 2);
    const s3 = ggml.impl.getOpParamsI32(op, 3);

    var args: ggml.metal.kargs.roll = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_roll(lib, op);

    const nth = @min(@as(c_int, 1024), @as(c_int, @intCast(op.ne[0])));

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_arange` (ggml-metal-ops.cpp:5597 @b1b770d90).
pub fn ggml_metal_op_arange(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    // The C `memcpy`s these out of `op_params` as `float`, reading
    // indices 0 and 2 -- not 0 and 1.
    const start = ggml.impl.getOpParamsF32(op, 0);
    const step = ggml.impl.getOpParamsF32(op, 2);

    var args: ggml.metal.kargs.arange = .{
        .ne0 = @intCast(op.ne[0]),
        .start = start,
        .step = step,
    };

    const ne0: c_int = @intCast(op.ne[0]);
    const nth = @min(@as(c_int, 1024), ne0);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_arange(lib, op);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, op, 1);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_timestep_embedding` (ggml-metal-ops.cpp:5631
/// @b1b770d90).
pub fn ggml_metal_op_timestep_embedding(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const dim = ggml.impl.getOpParamsI32(op, 0);
    const max_period = ggml.impl.getOpParamsI32(op, 1);

    var args: ggml.metal.kargs.timestep_embedding = .{
        .nb1 = op.nb[1],
        .dim = dim,
        .max_period = max_period,
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const nth = @max(@as(c_int, 1), @min(@as(c_int, 1024), @divTrunc(dim, 2)));

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(s0.ne[0]), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_argsort` (ggml-metal-ops.cpp:5762 @b1b770d90).
///
/// A bitonic sort of power-of-two blocks, then a merge ladder that
/// ping-pongs between `dst` and the scratch immediately after it. The
/// parity test up front decides which of the two the first pass writes
/// to, so the last merge lands in `dst`.
pub fn ggml_metal_op_argsort(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    var nth: c_int = 1;
    while (nth < ne00 and 2 * nth <= maxThreads(pipeline)) {
        nth *= 2;
    }

    const npr = @divTrunc(ne00 + nth - 1, nth);

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const smem = stdlib.num.pad(usize, @as(usize, @intCast(nth)) * @sizeOf(i32), 16);

    const bid_src0 = getBufferId(s0);
    var bid_dst = getBufferId(op);

    var bid_tmp = bid_dst;
    bid_tmp.offs += @intCast(ggml.types.ggml_nbytes(op));

    if (@rem(@as(c_int, @intFromFloat(@ceil(@log(@as(f64, @floatFromInt(npr))) / @log(@as(f64, 2))))), 2) == 1) {
        std.mem.swap(ggml.metal.device_c.BufferId, &bid_dst, &bid_tmp);
    }

    var args: ggml.metal.kargs.argsort = .{
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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, npr * ne01, ne02, ne03, nth, 1, 1);

    const pipeline_merge = ggml.metal.library.ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    var len: c_int = nth;

    while (len < ne00) {
        _ = concurrencyReset(ctx);

        var args_merge: ggml.metal.kargs.argsort_merge = .{
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        setBytes(enc, &args_merge, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 3);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, nm * ne01, ne02, ne03, nth_merge, 1, 1);

        std.mem.swap(ggml.metal.device_c.BufferId, &bid_dst, &bid_tmp);

        len <<= 1;
    }

    return 1;
}

/// Ports `ggml_metal_op_topk_moe` (ggml-metal-ops.cpp:6028 @b1b770d90).
///
/// The fused MoE routing (`TOPK_MOE` in `fusion.zig`): the gating over the
/// expert logits, its top-k selection and the gathered weights --
/// optionally renormalised by their sum, optionally scaled -- in one kernel
/// that writes both the expert ids (the ARGSORT's VIEW) and the weights
/// (the last node). Reached from `ggml_metal_op_soft_max`, and (#31615)
/// from `ggml_metal_op_unary`: `gating` is 0 for softmax, 1 for
/// `sqrt(softplus)` (DeepSeek-V4) and 2 for sigmoid (GLM), the unary chains
/// may add a selection bias (`prob_bias`, buffer 4), and the
/// renormalisation's denominator is either a clamp (softmax) or a `SCALE`
/// whose bias rides in `args.bias`. The nodes are found by op rather than
/// by position, since the chains differ in length.
///
/// Return: the nodes consumed; 1 when the pattern does not match here,
/// which the caller never asks.
pub fn ggml_metal_op_topk_moe(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const lib = ctx.lib;
    const enc = ctx.enc;

    var n_fuse: c_int = 1;
    const f = ctx.canFuse(idx, .full, &n_fuse) orelse return 1;
    if (ggml.metal.fusion.ggml_metal_fusion_get_id(f) != .topk_moe) {
        return 1;
    }

    const first = ctx.node(idx);
    const logits = src(first, 0);

    // the unary chains score with sigmoid or sqrt(softplus) and rank biased scores
    const with_unary = first.op == ggml.impl.ggml_c.GGML_OP_UNARY;
    const gating: i32 = if (!with_unary) 0 else if (ggml.types.ggml_get_unary_op(first) == ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS) 1 else 2;

    var get_rows: ?*ggml.impl.ggml_c.ggml_tensor = null;
    var sum_rows: ?*ggml.impl.ggml_c.ggml_tensor = null;
    var prob_bias: ?*ggml.impl.ggml_c.ggml_tensor = null;
    var sum_rows_pos: c_int = -1;
    var j: c_int = 0;
    while (j < n_fuse) : (j += 1) {
        const node = ctx.node(idx + j);
        if (node.op == ggml.impl.ggml_c.GGML_OP_GET_ROWS) get_rows = node;
        if (node.op == ggml.impl.ggml_c.GGML_OP_ADD) prob_bias = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, node.src[1]);
        if (node.op == ggml.impl.ggml_c.GGML_OP_SUM_ROWS) {
            sum_rows = node;
            sum_rows_pos = j;
        }
    }

    const ids = src(get_rows.?, 1);
    const weights = ctx.node(idx + n_fuse - 1);

    const with_norm = sum_rows != null;
    const with_scale = weights.op == ggml.impl.ggml_c.GGML_OP_SCALE;

    const n_expert: i64 = logits.ne[0];
    const n_tokens: i64 = logits.ne[1];
    const n_expert_used: i64 = ids.ne[0];

    var clamp: f32 = -std.math.inf(f32);
    var bias: f32 = 0.0;
    if (with_norm) {
        const denom_node = ctx.node(idx + sum_rows_pos + 1);
        if (denom_node.op == ggml.impl.ggml_c.GGML_OP_SCALE) {
            bias = ggml.impl.getOpParamsF32(denom_node, 1);
        } else {
            clamp = ggml.impl.getOpParamsF32(denom_node, 0);
        }
    }

    var scale: f32 = 1.0;
    if (with_scale) {
        scale = ggml.impl.getOpParamsF32(weights, 0);
    }

    var args: ggml.metal.kargs.topk_moe = .{
        .ne01 = @truncate(n_tokens),
        .nb01 = logits.nb[1],
        .nb1_ids = ids.nb[1],
        .clamp = clamp,
        .bias = bias,
        .scale = scale,
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_topk_moe(lib, @truncate(n_expert), @truncate(n_expert_used), with_norm, gating, prob_bias != null);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, logits, 1);
    setBuffer(enc, weights, 2);
    setBuffer(enc, ids, 3);
    setBuffer(enc, prob_bias orelse logits, 4);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n_tokens), 1, 1, 32, 1, 1);

    ctx.countFusions(f);

    if (ctx.debugFusion() > 1) {
        const gate_name: [*:0]const u8 = if (gating == 1) "SOFTPLUS + SQRT" else if (gating == 2) "SIGMOID" else "SOFT_MAX";
        ggml.impl.logDebug("%s: fuse: %s + ARGSORT + GET_ROWS\n", .{ "ggml_metal_op_topk_moe", gate_name });
    }

    return n_fuse;
}

/// Ports `ggml_metal_op_moe_reduce` (ggml-metal-ops.cpp:6113 @b1b770d90).
///
/// The fused MoE output reduction (`MOE_REDUCE` in `fusion.zig`): the
/// experts' outputs times their routing weights, summed over the experts,
/// in one kernel. Reached from `ggml_metal_op_bin` on the leading `MUL`.
///
/// Return: the nodes consumed; 1 when the pattern does not match here.
pub fn ggml_metal_op_moe_reduce(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const lib = ctx.lib;
    const enc = ctx.enc;

    var n_fuse: c_int = 1;
    const f = ctx.canFuse(idx, .full, &n_fuse) orelse return 1;
    if (ggml.metal.fusion.ggml_metal_fusion_get_id(f) != .moe_reduce) {
        return 1;
    }

    const mul = ctx.node(idx);
    const experts = src(mul, 0);
    const weights = src(mul, 1);
    const dst = ctx.node(idx + n_fuse - 1);

    var args: ggml.metal.kargs.moe_reduce = .{
        .ne00 = @truncate(experts.ne[0]),
        .ne02 = @truncate(experts.ne[2]),
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_moe_reduce(lib, @truncate(experts.ne[1]));

    const nth = @min(@as(c_int, 256), maxThreads(pipeline));
    const n_col_tiles = @divTrunc(args.ne00 + nth - 1, nth);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, experts, 1);
    setBuffer(enc, weights, 2);
    setBuffer(enc, dst, 3);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, args.ne02, n_col_tiles, 1, nth, 1, 1);

    ctx.countFusions(f);

    if (ctx.debugFusion() > 1) {
        ggml.impl.logDebug("%s: fuse: MOE_REDUCE\n", .{"ggml_metal_op_moe_reduce"});
    }

    return n_fuse;
}

/// Ports `ggml_metal_op_top_k` (ggml-metal-ops.cpp:6155 @b1b770d90).
///
/// Two paths. Radix-select has a fixed one-workgroup-per-row cost that only
/// pays off for long rows, many rows or a large `k`; otherwise the bitonic
/// sort and merge wins.
pub fn ggml_metal_op_top_k(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    // radix-select has a fixed single-workgroup-per-row cost (~50-60us) that is only
    // amortized for long rows, many rows, or a large k; otherwise the bitonic path wins
    const ncols: c_int = @intCast(src(op, 0).ne[0]);
    const k: c_int = @intCast(op.ne[0]);
    const nrows: c_int = @intCast(ggml.types.ggml_nrows(src(op, 0)));

    const use_radix =
        ncols > 2048 and (k > 64 or (nrows > 4 and ncols >= 8192));

    if (use_radix) {
        topKRadix(ctx, idx);
    } else {
        topKBitonic(ctx, idx);
    }

    return 1;
}

/// Ports `ggml_metal_op_top_k_radix` (ggml-metal-ops.cpp:5984 @b1b770d90).
///
/// One workgroup per row. Each float maps to an order-preserving unsigned
/// key; four radix-8 histogram passes find the k-th largest, then the top-k
/// indices are compacted.
fn topKRadix(ctx: *Op, idx: c_int) void {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_top_k_radix(lib, op);

    // one workgroup per row; radix-select the k-th largest value
    const nth = @min(@as(c_int, 1024), maxThreads(pipeline));

    var args: ggml.metal.kargs.top_k = .{
        .ne00 = @intCast(s0.ne[0]),
        .ne01 = @intCast(s0.ne[1]),
        .ne02 = @intCast(s0.ne[2]),
        .ne03 = @intCast(s0.ne[3]),
        .nb01 = s0.nb[1],
        .nb02 = s0.nb[2],
        .nb03 = s0.nb[3],
        .top_k = @intCast(op.ne[0]),
    };

    // shared memory: 256-entry histogram + bucket/above scalars
    const smem_histo = stdlib.num.pad(usize, 256 * @sizeOf(u32), 16);
    const smem_bucket = stdlib.num.pad(usize, @sizeOf(u32), 16);
    const smem_above = stdlib.num.pad(usize, @sizeOf(u32), 16);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_histo, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_bucket, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_above, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(s0.ne[1]), @intCast(s0.ne[2]), @intCast(s0.ne[3]), nth, 1, 1);
}

/// Ports `ggml_metal_op_top_k_bitonic` (ggml-metal-ops.cpp:5871 @b1b770d90).
///
/// `argsort`'s structure, but each block keeps only its own top-k, so
/// `args.ne0` is recomputed to the total kept rather than the row width,
/// and the final merge narrows to `top_k`. It reuses
/// `kargs_argsort`/`kargs_argsort_merge`. The fallback for small `k` over
/// few rows, where one radix workgroup per row cannot reach enough
/// parallelism.
fn topKBitonic(ctx: *Op, idx: c_int) void {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    ggml.impl.assert(ggml.types.ggml_is_contiguous_rows(s0), "ggml_is_contiguous_rows(op->src[0])");

    const ne00: c_int = @intCast(s0.ne[0]);
    const ne01: c_int = @intCast(s0.ne[1]);
    const ne02: c_int = @intCast(s0.ne[2]);
    const ne03: c_int = @intCast(s0.ne[3]);
    const ne1: c_int = @intCast(op.ne[1]);
    const ne2: c_int = @intCast(op.ne[2]);
    const ne3: c_int = @intCast(op.ne[3]);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    var nth: c_int = 1;
    while (nth < ne00 and 2 * nth <= maxThreads(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const npr = @divTrunc(ne00 + nth - 1, nth);

    const smem = stdlib.num.pad(usize, @as(usize, @intCast(nth)) * @sizeOf(i32), 16);

    const bid_src0 = getBufferId(s0);
    var bid_dst = getBufferId(op);

    var bid_tmp = bid_dst;
    bid_tmp.offs += @sizeOf(i32) * @as(usize, @intCast(ggml.types.ggml_nelements(s0)));

    if (@rem(@as(c_int, @intFromFloat(@ceil(@log(@as(f64, @floatFromInt(npr))) / @log(@as(f64, 2))))), 2) == 1) {
        std.mem.swap(ggml.metal.device_c.BufferId, &bid_dst, &bid_tmp);
    }

    const top_k: c_int = @intCast(op.ne[0]);

    var args: ggml.metal.kargs.argsort = .{
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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, npr * ne01, ne02, ne03, nth, 1, 1);

    const pipeline_merge = ggml.metal.library.ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    var len: c_int = args.top_k;

    while (len < args.ne0) {
        _ = concurrencyReset(ctx);

        // merges per row
        const nm = @divTrunc(args.ne0 + 2 * len - 1, 2 * len);

        const nth_merge = @min(@as(c_int, 512), @min(len, maxThreads(pipeline_merge)));

        var args_merge: ggml.metal.kargs.argsort_merge = .{
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        setBytes(enc, &args_merge, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 3);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, nm * ne01, ne02, ne03, nth_merge, 1, 1);

        std.mem.swap(ggml.metal.device_c.BufferId, &bid_dst, &bid_tmp);

        len <<= 1;
    }
}

/// Ports `ggml_metal_op_tri` (ggml-metal-ops.cpp:6176 @b1b770d90).
pub fn ggml_metal_op_tri(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ne00: c_int = @intCast(s0.ne[0]);

    var args: ggml.metal.kargs.tri = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_tri(lib, op);

    var nth: c_int = 32; // SIMD width

    while (nth < ne00 and nth < maxThreads(pipeline)) {
        nth *= 2;
    }

    nth = @min(nth, maxThreads(pipeline));
    nth = @min(nth, ne00);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_flash_attn_ext` (ggml-metal-ops.cpp:3560
/// @b1b770d90).
///
/// The largest encoder in the file, and up to **five** dispatches: an
/// optional KV dequantise to F16, an optional KV-tail pad, an optional
/// mask-block scan, the attention kernel itself, and — on the vector
/// path with more than one workgroup — a reduce over the per-workgroup
/// partials. A third attention arm comes first: the
/// tensor-API kernel (`useTensor`), on devices that have it, for F16 or BF16 KV
/// (or quantized KV converted to F16) whose length
/// needs no pad. Four scratch regions sit after `dst`, which is what
/// `backend.zig`'s `get_alloc_size` reserves.
///
/// With `ggml_flash_attn_ext_set_n_kv_max` set, a **sparse** vector path
/// first gathers each mask row's finite entries into an index list (a fifth
/// scratch region), then runs the vector kernel over those indices only.
///
/// Several of the C's branches are commented out to a constant
/// (`if (false)`, `if (true)`, a forced `has_kvpad`) so that scratch is
/// always reserved and graphs are not reallocated. Those are reproduced
/// as the live arm only, each noted where it occurs.
///
/// `src[5]` may hold `ggml_flash_attn_ext_rows`'s K/V row
/// lists, one per query slice. The vector kernels read rows through the
/// lists (`has_rows`); the other paths first gather the referenced rows
/// into contiguous F16 buffers -- a sixth scratch region, after the sparse
/// index -- dequantizing quantized KV to F16 first, and then attend over
/// them as an ordinary dense op.
pub fn ggml_metal_op_flash_attn_ext(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const props_dev = ggml.metal.device_c.ggml_metal_device_get_props(ctx.dev);

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

    ggml.impl.assert(@rem(ne00, 4) == 0, "ne00 % 4 == 0");

    ggml.impl.assert(s0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    ggml.impl.assert(s1.type == s2.type, "op->src[1]->type == op->src[2]->type");

    ggml.impl.assert(ne11 == ne21, "ne11 == ne21");
    ggml.impl.assert(ne12 == ne22, "ne12 == ne22");

    ggml.impl.assert(s3o == null or s3o.?.type == ggml.impl.ggml_c.GGML_TYPE_F16, "!op->src[3] || op->src[3]->type == GGML_TYPE_F16");
    ggml.impl.assert(
        s3o == null or s3o.?.ne[1] >= s0.ne[1],
        "the Flash-Attention Metal kernel requires the mask to be at least n_queries big",
    );

    // K/V row lists (ggml_flash_attn_ext_rows): src[5] holds one list of cache rows per q slice
    // the attention loops then run over the mask columns and read K/V rows through the lists
    const kv_rows_opt = srcOpt(op, 5);
    const has_rows = kv_rows_opt != null;

    ggml.impl.assert(!has_rows or (kv_rows_opt.?.type == ggml.impl.ggml_c.GGML_TYPE_I32 and ggml.types.ggml_is_contiguous(kv_rows_opt.?)), "!has_rows || (kv_rows->type == GGML_TYPE_I32 && ggml_is_contiguous(kv_rows))");
    ggml.impl.assert(!has_rows or (kv_rows_opt.?.ne[1] == ne03 and kv_rows_opt.?.ne[2] == 1 and kv_rows_opt.?.ne[3] == 1), "!has_rows || (kv_rows->ne[1] == ne03 && kv_rows->ne[2] == 1 && kv_rows->ne[3] == 1)");
    ggml.impl.assert(!has_rows or op.src[3] != null, "!has_rows || op->src[3]");

    // the loop bound over the mask columns / K/V row lists
    const ne11_attn: i64 = if (kv_rows_opt) |kr| kr.ne[0] else ne11;

    // the row lists are padded by the KV cache to a multiple of 256, so the pad path never triggers
    ggml.impl.assert(!has_rows or @rem(ne11_attn, ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NCPSG) == 0, "!has_rows || ne11_attn % OP_FLASH_ATTN_EXT_NCPSG == 0");
    ggml.impl.assert(!has_rows or @rem(ne11_attn, ggml.metal.impl_c.OP_FLASH_ATTN_EXT_VEC_NCPSG) == 0, "!has_rows || ne11_attn % OP_FLASH_ATTN_EXT_VEC_NCPSG == 0");

    var scale = ggml.impl.getOpParamsF32(op, 0);
    const max_bias = ggml.impl.getOpParamsF32(op, 1);
    const logit_softcap = ggml.impl.getOpParamsF32(op, 2);

    if (logit_softcap != 0.0) {
        scale /= logit_softcap;
    }

    const has_mask = op.src[3] != null;
    const has_sinks = op.src[4] != null;

    ggml.impl.assert(!has_rows or has_mask, "!has_rows || has_mask");
    const has_bias = max_bias != 0.0;
    const has_scap = logit_softcap != 0.0;

    const n_head: u32 = @intCast(s0.ne[2]);
    const n_head_log2: i32 = @as(i32, 1) << @intCast(@as(u32, @intFromFloat(@floor(std.math.log2(@as(f32, @floatFromInt(n_head)))))));

    const m0 = std.math.pow(f32, 2.0, -(max_bias) / @as(f32, @floatFromInt(n_head_log2)));
    const m1 = std.math.pow(f32, 2.0, -(max_bias / 2.0) / @as(f32, @floatFromInt(n_head_log2)));

    ggml.impl.assert(ne01 < 65536, "ne01 < 65536");

    const bid_src0 = getBufferId(s0);
    const bid_src1 = getBufferId(s1);
    const bid_src2 = getBufferId(s2);
    const bid_src3 = if (has_mask) getBufferId(s3o.?) else bid_src0;
    const bid_src4 = if (has_sinks) getBufferId(src(op, 4)) else bid_src0;

    const bid_dst = getBufferId(op);

    var bid_pad = bid_dst;
    bid_pad.offs += @intCast(ggml.types.ggml_nbytes(op));

    var bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    var bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    var bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    // sparse path: gather the finite mask entries into index lists and run the vec kernels over them
    const n_kv_max_sparse = nKvMaxSparse(op);
    const use_sparse = n_kv_max_sparse > 0;
    const n_kv_max_padded: c_int = if (use_sparse) @intCast(stdlib.num.pad(usize, @intCast(n_kv_max_sparse), ggml.metal.impl_c.OP_FLASH_ATTN_EXT_VEC_NCPSG)) else 0;

    const use_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // with kv_rows, the non-vec path gathers the referenced rows into a contiguous F16 buffer and
    // attends over it with the F16 kernel - quantized KV is first dequantized with the F16 pass
    const rows_gather = has_rows and !use_vec;

    // the vec kernels dequantize the KV inline; no need for the F16 dequant pass in the sparse path
    const dequant_kv = !use_sparse and (if (rows_gather)
        ggml.types.ggml_is_quantized(s1.type)
    else
        useKvF16(op));

    const use_kv_f16 = !use_sparse and (if (rows_gather) true else dequant_kv);

    var bid_idx = bid_kv_f16;
    bid_idx.offs += ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

    // contiguous K/V buffers holding the rows referenced by the kv_rows lists
    var bid_rows = bid_idx;
    bid_rows.offs += ggml_metal_op_flash_attn_ext_extra_idx(op);

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

    if (dequant_kv) {
        ggml.impl.assert(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0, "extra_kv_f16(op) != 0");

        const v_is_view_of_k = vIsViewOfK(op);

        const nblocks1_64: i64 = @divTrunc(s1.ne[0], ggml.types.ggml_blck_size(s1.type)) * s1.ne[1] * s1.ne[2] * s1.ne[3];
        ggml.impl.assert(nblocks1_64 <= std.math.maxInt(i32), "nblocks1_64 <= INT32_MAX");
        const nblocks1: i32 = @intCast(nblocks1_64);

        var bid_v_f16 = bid_kv_f16;
        bid_v_f16.offs += kvF16KSize(op);

        const pipeline0 = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
        const nth = @min(maxThreads(pipeline0), @as(c_int, 256));

        // K
        var args_k: ggml.metal.kargs.flash_attn_ext_kv_f16 = .{
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
        setBytes(enc, &args_k, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src1, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_kv_f16, 2);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @divTrunc(nblocks1 + nth - 1, nth), 1, 1, nth, 1, 1);

        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
        if (!v_is_view_of_k) {
            const nblocks2_64: i64 = @divTrunc(s2.ne[0], ggml.types.ggml_blck_size(s2.type)) * s2.ne[1] * s2.ne[2] * s2.ne[3];
            ggml.impl.assert(nblocks2_64 <= std.math.maxInt(i32), "nblocks2_64 <= INT32_MAX");
            const nblocks2: i32 = @intCast(nblocks2_64);

            var args_v: ggml.metal.kargs.flash_attn_ext_kv_f16 = .{
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

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args_v, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src2, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_v_f16, 2);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @divTrunc(nblocks2 + nth - 1, nth), 1, 1, nth, 1, 1);
        }

        // the pad and attention kernels read the dequantized KV
        _ = concurrencyReset(ctx);

        bid_k = bid_kv_f16;
        bid_v = if (v_is_view_of_k) bid_k else bid_v_f16;

        // contiguous F16 layout of the dequantized K
        nb10_attn = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
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
            nb20_attn = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
            nb21_attn = nb20_attn * @as(u64, @intCast(ne20));
            nb22_attn = nb21_attn * @as(u64, @intCast(ne21));
            nb23_attn = nb22_attn * @as(u64, @intCast(ne22));
        }
    }

    // with kv_rows, gather the cache rows referenced by the lists into contiguous F16 buffers
    // the attention then runs as a standard dense op over the gathered rows (tensor or half8x8 kernel)
    if (rows_gather) {
        // the C's `assert(extra_rows(op) != 0)` compiles out in the reference

        const v_is_view_of_k = vIsViewOfK(op);

        ggml.impl.assert(@rem(ne20, 4) == 0, "ne20 % 4 == 0");

        const k_size: usize = stdlib.num.pad(usize, @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) * @as(usize, @intCast(ne10)) * @as(usize, @intCast(ne11_attn)) * @as(usize, @intCast(ne12)) * @as(usize, @intCast(ne03)), 16);

        var args0: ggml.metal.kargs.flash_attn_ext_rows = .{
            .ne11 = @intCast(ne11_attn),
            .ne_12_2 = ne12,
            .ne10 = ne10,
            .ne20 = if (v_is_view_of_k) 0 else ne20, // the gathered V is a view of the gathered K
            .v_off = k_size,
            .nb11 = nb11_attn,
            .nb12 = nb12_attn,
            .nb21 = nb21_attn,
            .nb22 = nb22_attn,
        };

        const pipeline0 = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_rows(lib, op);
        const nth = @min(maxThreads(pipeline0), @as(c_int, 64));

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
        setBytes(enc, &args0, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_k, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_v, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, getBufferId(kv_rows_opt.?), 3);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_rows, 4);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(ne11_attn), ne03, 1, nth, 1, 1);

        _ = concurrencyReset(ctx);

        // contiguous F16 layout of the gathered rows: [DK, n_rows, n_head_kv, n_streams]
        bid_k = bid_rows;

        nb10_attn = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
        nb11_attn = nb10_attn * @as(u64, @intCast(ne10));
        nb12_attn = nb11_attn * @as(u64, @intCast(ne11_attn));
        nb13_attn = nb12_attn * @as(u64, @intCast(ne12));

        // if V is a view of K, the gathered V is read from the gathered K with K's strides
        if (v_is_view_of_k) {
            bid_v = bid_k;

            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } else {
            bid_v = bid_rows;
            bid_v.offs += k_size;

            nb20_attn = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t);
            nb21_attn = nb20_attn * @as(u64, @intCast(ne20));
            nb22_attn = nb21_attn * @as(u64, @intCast(ne11_attn));
            nb23_attn = nb22_attn * @as(u64, @intCast(ne22));
        }
    }

    if (!use_sparse and useTensor(op, props_dev.has_tensor)) {
        // tensor API kernel
        const nqptg: c_int = if (ne00 >= 512) ggml.metal.impl_c.OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE else ggml.metal.impl_c.OP_FLASH_ATTN_EXT_TENSOR_NQPSG; // queries per threadgroup
        const ncpsg: c_int = ggml.metal.impl_c.OP_FLASH_ATTN_EXT_TENSOR_NCPSG; // cache values per threadgroup
        const nsg: c_int = ggml.metal.impl_c.OP_FLASH_ATTN_EXT_TENSOR_NSG;

        if (has_mask) {
            // the C's `assert(extra_blk(op) != 0)` compiles out in the reference

            var args0: ggml.metal.kargs.flash_attn_ext_blk = .{
                .ne01 = ne01,
                .ne30 = ne30,
                .ne31 = ne31,
                .ne32 = ne32,
                .ne33 = ne33,
                .nb31 = nb31,
                .nb32 = nb32,
                .nb33 = nb33,
            };

            const pipeline0 = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args0, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src3, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_blk, 2);

            const nblk1: i32 = @divTrunc(ne01 + nqptg - 1, nqptg);
            const nblk0: i32 = @divTrunc(ne30 + ncpsg - 1, ncpsg);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32 * ne33, 32, 1, 1);

            _ = concurrencyReset(ctx);
        }

        const ns10: i32 = @intCast(nb11_attn / nb10_attn);
        const ns20: i32 = @intCast(nb21_attn / nb20_attn);

        var args: ggml.metal.kargs.flash_attn_ext = .{
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne11 = @intCast(ne11_attn),
            .ne_12_2 = ne12,
            .ne_12_3 = if (has_rows) ne03 else ne13, // the gathered K/V holds one copy per stream
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

        // shared memory layout: queries (16-bit), scores (float), probabilities (16-bit), row scale (float), rescale flag (int)
        const q: usize = @intCast(nqptg);
        const smem = stdlib.num.pad(
            usize,
            q * @as(usize, @intCast(ne00)) * @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) +
                q * @as(usize, @intCast(ncpsg)) * (@sizeOf(f32) + @sizeOf(ggml.impl.ggml_c.ggml_fp16_t)) +
                q * @sizeOf(f32) + @sizeOf(i32),
            16,
        );

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_tensor(lib, op, has_mask, has_sinks, has_bias, has_scap);

        ggml.impl.assert(nsg * 32 <= maxThreads(pipeline), "nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)");
        ggml.impl.assert(smem <= props_dev.max_theadgroup_memory_size, "smem <= props_dev->max_theadgroup_memory_size");

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_k, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_v, 3);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src3, 4);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src4, 5);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_blk, 6);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

        ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @divTrunc(ne01 + nqptg - 1, nqptg), ne02, ne03, 32, nsg, 1);
    } else if (!use_sparse and !ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const nqptg: c_int = ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const ncpsg: c_int = ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        comptime {
            if (ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NQPSG > 32 or
                @rem(ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NQPSG, 8) != 0 or
                @rem(ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NCPSG, 32) != 0)
            {
                @compileError("nqptg <= 32 && nqptg % 8 == 0 && ncpsg % 32 == 0");
            }
        }

        var need_sync = false;

        const has_kvpad = @rem(ne11_attn, ncpsg) != 0;

        if (has_kvpad) {
            ggml.impl.assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0, "extra_pad(op) != 0");

            var args0: ggml.metal.kargs.flash_attn_ext_pad = .{
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

            const pipeline0 = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args0, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_k, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_v, 2);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src3, 3);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_pad, 4);

            ggml.impl.assert(ne12 == ne22, "ne12 == ne22");
            ggml.impl.assert(ne13 == ne23, "ne13 == ne23");

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, @max(ne12, ne32), @max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            ggml.impl.assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0, "extra_blk(op) != 0");

            var args0: ggml.metal.kargs.flash_attn_ext_blk = .{
                .ne01 = ne01,
                .ne30 = ne30,
                .ne31 = ne31,
                .ne32 = ne32,
                .ne33 = ne33,
                .nb31 = nb31,
                .nb32 = nb32,
                .nb33 = nb33,
            };

            const pipeline0 = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args0, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src3, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_blk, 2);

            const nblk1: i32 = @divTrunc(ne01 + nqptg - 1, nqptg);
            const nblk0: i32 = @divTrunc(ne30 + ncpsg - 1, ncpsg);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32 * ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            _ = concurrencyReset(ctx);
        }

        const is_q: c_int = if (!use_kv_f16 and ggml.types.ggml_is_quantized(s1.type)) 1 else 0;

        // shared memory layout (halfs unless noted):
        //   queries/attn/result: Q*(DK + 2*PAD2(DV,64) + 4*C)
        //   quantized KV scratch: 16*32*NSG (only when is_q)
        //
        // the C's `fa_smem` lambda
        const dv_pad: i64 = @intCast(stdlib.num.pad(usize, @intCast(ne20), 64));

        const faSmem = struct {
            fn f(ne00_: c_int, dv_pad_: i64, ncpsg_: c_int, nqptg_: c_int, is_q_: c_int, nsg_: i32) usize {
                const smem_half: usize = @intCast(nqptg_ * (ne00_ + 2 * dv_pad_ + 4 * ncpsg_) + is_q_ * (16 * 32 * nsg_));
                return stdlib.num.pad(usize, smem_half * @sizeOf(ggml.impl.ggml_c.ggml_fp16_t), 16);
            }
        }.f;

        // simdgroups per threadgroup (a.k.a. warps)
        const nsg: i32 = if (ne00 >= 512) 8 else 4;

        const smem = faSmem(ne00, dv_pad, ncpsg, nqptg, is_q, nsg);

        const ns10: i32 = @intCast(nb11_attn / nb10_attn);
        const ns20: i32 = @intCast(nb21_attn / nb20_attn);

        var args: ggml.metal.kargs.flash_attn_ext = .{
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne11 = @intCast(ne11_attn),
            .ne_12_2 = ne12,
            .ne_12_3 = if (has_rows) ne03 else ne13, // the gathered K/V holds one copy per stream
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

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext(
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_k, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_v, 3);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src3, 4);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src4, 5);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_blk, 7);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 8);

        ggml.impl.assert(smem <= props_dev.max_theadgroup_memory_size, "smem <= props_dev->max_theadgroup_memory_size");

        ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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
        // sparse: the index lists are per query row, so a threadgroup can share KV with Q == 1 only
        var cfg = if (use_sparse)
            ggml.metal.tuning.baselineCfg(ne00, ne20)
        else
            ggml.metal.tuning.pick(
                props_dev.gpu_family,
                @intCast(s1.type),
                ne00,
                ne20, // dk, dv (ne00 == dk for FA)
                ne11,
                ne01,
            );

        var nqptg: c_int = cfg.Q; // queries per threadgroup
        const ncpsg: c_int = ggml.metal.impl_c.OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const nhptg: c_int = 1; // heads per threadgroup

        ggml.impl.assert(nqptg <= 32, "nqptg <= 32");
        ggml.impl.assert(nqptg == 1 or nqptg == 2 or nqptg == 4, "only instantiated Q values");
        comptime {
            if (@rem(ggml.metal.impl_c.OP_FLASH_ATTN_EXT_VEC_NCPSG, 32) != 0) {
                @compileError("ncpsg % 32 == 0");
            }
        }

        var need_sync = false;

        const has_kvpad = !use_sparse and @rem(ne11_attn, ncpsg) != 0;

        if (use_sparse) {
            ggml.impl.assert(ggml_metal_op_flash_attn_ext_extra_idx(op) != 0, "extra_idx(op) != 0");

            ggml.impl.assert(ne30 == ne11, "ne30 == ne11");

            var args0: ggml.metal.kargs.flash_attn_ext_vec_idx = .{
                .ne30 = ne30,
                .ne31 = ne31,
                .ne32 = ne32,
                .ne33 = ne33,
                .nb31 = nb31,
                .nb32 = nb32,
                .nb33 = nb33,
                .n_kv_max = n_kv_max_sparse,
                .n_kv_max_padded = n_kv_max_padded,
            };

            const pipeline0 = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args0, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src3, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_idx, 2);

            var nth = @min(maxThreads(pipeline0), @as(c_int, 256));
            nth = @max(@as(c_int, 32), @divTrunc(nth, 32) * 32);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nth, 1, 1);

            need_sync = true;
        }

        if (has_kvpad) {
            ggml.impl.assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0, "extra_pad(op) != 0");

            var args0: ggml.metal.kargs.flash_attn_ext_pad = .{
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

            const pipeline0 = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
            setBytes(enc, &args0, 0);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_k, 1);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_v, 2);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src3, 3);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_pad, 4);

            ggml.impl.assert(ne12 == ne22, "ne12 == ne22");
            ggml.impl.assert(ne13 == ne23, "ne13 == ne23");

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, @max(ne12, ne32), @max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            _ = concurrencyReset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        ggml.impl.assert(ne10 >= ne20, "ne10 >= ne20");

        // shared memory layout (halfs unless noted):
        //   queries:      Q*NSG*PAD2(ne00, 128)
        //   attn + mask:  NSG*4*Q*C
        //   results:      2*NSG*Q*PAD2(ne20, 128)
        //   sparse idx:   NSG*C ints (when use_sparse, or when gathering K/V rows)
        //
        // the C's `fa_vec_smem` lambda
        const dk_pad: usize = stdlib.num.pad(usize, @intCast(ne00), 128);
        const dv_pad: usize = stdlib.num.pad(usize, @intCast(ne20), 128);

        const faVecSmem = struct {
            fn f(dk_pad_: usize, dv_pad_: usize, ncpsg_: c_int, use_sparse_: bool, nsg_: i64, nqptg_: i32) usize {
                const smem_half: usize = (dk_pad_ + @as(usize, @intCast(4 * ncpsg_)) + 2 * dv_pad_) * @as(usize, @intCast(nqptg_)) * @as(usize, @intCast(nsg_));
                const smem_idx: usize = if (use_sparse_) @as(usize, @intCast(nsg_)) * @as(usize, @intCast(ncpsg_)) * @sizeOf(c_int) else 0;
                return stdlib.num.pad(usize, smem_half * @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) + smem_idx, 16);
            }
        }.f;

        var nsg: i64 = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        var nwg: i32 = 1;
        if (use_sparse) {
            if (ne01 > 32) {
                // large sparse batch
                nwg = 1;
                nsg = 1;
                if (n_kv_max_padded == 640) {
                    nsg = 4; // 640 % (4*32) == 0
                } else {
                    while (2 * nwg * nsg * ncpsg < n_kv_max_padded and nsg < 4) {
                        nsg *= 2;
                    }
                }
            } else {
                // small sparse batch
                nwg = 32;
                nsg = 1;
                while (2 * nwg * nsg * ncpsg < n_kv_max_padded and nsg < 4) {
                    nsg *= 2;
                }
            }
        } else {
            nwg = 32;
            nsg = 1;
            while (2 * nwg * nsg * ncpsg < ne11_attn and nsg < 4) {
                nsg *= 2;
            }
        }

        // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
        if (faVecSmem(dk_pad, dv_pad, ncpsg, use_sparse or has_rows, nsg, nqptg) > props_dev.max_theadgroup_memory_size) {
            cfg = ggml.metal.tuning.baselineCfg(ne00, ne20);
            nqptg = cfg.Q; // = 1
        }

        const ns10: i32 = @intCast(nb11_attn / nb10_attn);
        const ns20: i32 = @intCast(nb21_attn / nb20_attn);

        var args: ggml.metal.kargs.flash_attn_ext_vec = .{
            .ne01 = ne01,
            .ne02 = ne02,
            .ne03 = ne03,
            .nb01 = s0.nb[1],
            .nb02 = s0.nb[2],
            .nb03 = s0.nb[3],
            .ne11 = if (use_sparse) n_kv_max_padded else @as(i32, @intCast(ne11_attn)),
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
            .n_kv_max_padded = n_kv_max_padded,
        };

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_vec(
            lib,
            op,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            use_sparse,
            has_rows,
            nqptg,
            cfg.NE,
            @intCast(nsg),
            nwg,
            use_kv_f16,
            ns10,
            ns20,
        );

        ggml.impl.assert(nsg * 32 <= maxThreads(pipeline), "nsg*32 <= max_theads_per_threadgroup");

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_k, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_v, 3);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src3, 4);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src4, 5);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, if (use_sparse) bid_idx else (if (has_rows) getBufferId(kv_rows_opt.?) else bid_src0), 8);

        const smem = faVecSmem(dk_pad, dv_pad, ncpsg, use_sparse or has_rows, nsg, nqptg);

        ggml.impl.assert(smem <= props_dev.max_theadgroup_memory_size, "smem <= props_dev->max_theadgroup_memory_size");

        if (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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
            ggml.impl.assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0, "extra_tmp(op) != 0");

            ggml.impl.assert(
                @as(i64, ne01) * ne02 * ne03 == op.ne[1] * op.ne[2] * op.ne[3],
                "ne01*ne02*ne03 == ne1*ne2*ne3",
            );
            ggml.impl.assert(
                @as(u64, @intCast(op.ne[1] * op.ne[2] * op.ne[3])) <= (@as(u64, 1) << 31),
                "(uint64_t)ne1*ne2*ne3 <= (1u << 31)",
            );

            // write the results from each workgroup into a temp buffer
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

                var args0: ggml.metal.kargs.flash_attn_ext_vec_reduce = .{
                    .nrows = nrows,
                };

                const pipeline0 = ggml.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline0);
                setBytes(enc, &args0, 0);
                ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
                ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

                ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32 * nwg, 1, 1);
            }
        }
    }

    return 1;
}

/// Ports `ggml_metal_op_set` (ggml-metal-ops.cpp:2223 @b1b770d90).
///
/// Two `cpy` dispatches when not in-place: src0 into dst, then src1 into
/// a window of dst at `offs` with the `pnb*` strides. Note the second
/// `args` describes **src1**'s shape in the `ne0*` fields.
pub fn ggml_metal_op_set(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    const bid_src0 = getBufferId(s0);
    const bid_src1 = getBufferId(s1);
    var bid_dst = getBufferId(op);

    const pnb1: usize = @intCast(ggml.impl.getOpParamsI32(op, 0));
    const pnb2: usize = @intCast(ggml.impl.getOpParamsI32(op, 1));
    const pnb3: usize = @intCast(ggml.impl.getOpParamsI32(op, 2));
    const offs: usize = @intCast(ggml.impl.getOpParamsI32(op, 3));

    const inplace = ggml.impl.getOpParamsI32(op, 4) != 0;

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        const pipeline_cpy = ggml.metal.library.ggml_metal_library_get_pipeline_cpy(lib, s0.type, op.type);

        var args_cpy: ggml.metal.kargs.cpy = .{
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

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_cpy);
        setBytes(enc, &args_cpy, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

        const nth_cpy = @min(maxThreads(pipeline_cpy), @as(c_int, @intCast(s0.ne[0])));

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_cpy(lib, s1.type, op.type);

    const ne10: i64 = s1.ne[0];

    ggml.impl.assert(
        @rem(ne10, ggml.types.ggml_blck_size(s1.type)) == 0,
        "ne10 % ggml_blck_size(op->src[1]->type) == 0",
    );

    var nk0: i64 = ne10;
    if (ggml.types.ggml_is_quantized(s1.type)) {
        nk0 = @divTrunc(ne10, 16);
    } else if (ggml.types.ggml_is_quantized(op.type)) {
        nk0 = @divTrunc(ne10, ggml.types.ggml_blck_size(op.type));
    }

    var nth: c_int = @intCast(@min(nk0 * s1.ne[1], 256));

    // when rows are small, we can batch them together in a single threadgroup
    var nrptg: c_int = 1;

    // TODO: relax this constraint in the future
    if (ggml.types.ggml_blck_size(s1.type) == 1 and ggml.types.ggml_blck_size(op.type) == 1) {
        if (nth > @as(c_int, @intCast(nk0))) {
            nrptg = @divTrunc(nth + @as(c_int, @intCast(nk0)) - 1, @as(c_int, @intCast(nk0)));
            nth = @intCast(nk0);

            if (nrptg * nth > 256) {
                nrptg -= 1;
            }
        }
    }

    nth = @min(nth, @as(c_int, @intCast(nk0)));

    var args: ggml.metal.kargs.cpy = .{
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
        .nb0 = @intCast(ggml.types.ggml_element_size(op)),
        .nb1 = pnb1,
        .nb2 = pnb2,
        .nb3 = pnb3,
    };

    const nw0: c_int = if (nrptg == 1)
        @divTrunc(@as(c_int, @intCast(nk0)) + nth - 1, nth)
    else
        1;

    bid_dst.offs += offs;

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_src1, 1);
    ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);

    const ne11: c_int = @intCast(s1.ne[1]);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_cpy` (ggml-metal-ops.cpp:2351 @b1b770d90).
pub fn ggml_metal_op_cpy(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_cpy(lib, s0.type, op.type);

    const ne00: i64 = s0.ne[0];

    ggml.impl.assert(
        @rem(ne00, ggml.types.ggml_blck_size(s0.type)) == 0,
        "ne00 % ggml_blck_size(op->src[0]->type) == 0",
    );

    var nk0: i64 = ne00;
    if (ggml.types.ggml_is_quantized(s0.type)) {
        nk0 = @divTrunc(ne00, 16);
    } else if (ggml.types.ggml_is_quantized(op.type)) {
        nk0 = @divTrunc(ne00, ggml.types.ggml_blck_size(op.type));
    }

    var nth: c_int = @intCast(@min(nk0 * s0.ne[1], 256));

    // when rows are small, we can batch them together in a single threadgroup
    var nrptg: c_int = 1;

    // TODO: relax this constraint in the future
    if (ggml.types.ggml_blck_size(s0.type) == 1 and ggml.types.ggml_blck_size(op.type) == 1) {
        if (nth > @as(c_int, @intCast(nk0))) {
            nrptg = @divTrunc(nth + @as(c_int, @intCast(nk0)) - 1, @as(c_int, @intCast(nk0)));
            nth = @intCast(nk0);

            if (nrptg * nth > 256) {
                nrptg -= 1;
            }
        }
    }

    nth = @min(nth, @as(c_int, @intCast(nk0)));

    var args: ggml.metal.kargs.cpy = .{
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

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    const ne01: c_int = @intCast(s0.ne[1]);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_pool_1d` (ggml-metal-ops.cpp:2424 @b1b770d90).
///
/// One threadgroup makes one output: its threads split the
/// window, and the reduction goes through threadgroup memory, a sum and a
/// count of `f32`s for `AVG` and one partial for `MAX`.
pub fn ggml_metal_op_pool_1d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);

    const op_pool: ggml.impl.ggml_c.enum_ggml_op_pool = @intCast(ggml.impl.getOpParamsI32(op, 0));

    const k0 = ggml.impl.getOpParamsI32(op, 1);
    const s0 = ggml.impl.getOpParamsI32(op, 2);
    const p0 = ggml.impl.getOpParamsI32(op, 3);

    const IW: i64 = src0.ne[0];
    const OW: i64 = op.ne[0];

    const np: i64 = ggml.types.ggml_nelements(op);

    var args_pool_1d: ggml.metal.kargs.pool_1d = .{
        .k0 = k0,
        .s0 = s0,
        .p0 = p0,
        .IW = IW,
        .OW = OW,
        .np = np,
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const nth = @min(@as(c_int, 256), @max(@as(c_int, 1), k0));
    const ntg: c_int = @truncate(np);

    const smem: usize = if (op_pool == ggml.impl.ggml_c.GGML_OP_POOL_AVG)
        2 * 32 * @sizeOf(f32)
    else
        32 * @sizeOf(f32);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args_pool_1d, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, op, 2);
    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_pool_2d` (ggml-metal-ops.cpp:2518 @b1b770d90).
pub fn ggml_metal_op_pool_2d(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src0 = src(op, 0);

    const op_pool: ggml.impl.ggml_c.enum_ggml_op_pool = @intCast(ggml.impl.getOpParamsI32(op, 0));

    const k0 = ggml.impl.getOpParamsI32(op, 1);
    const k1 = ggml.impl.getOpParamsI32(op, 2);
    const s0 = ggml.impl.getOpParamsI32(op, 3);
    const s1 = ggml.impl.getOpParamsI32(op, 4);
    const p0 = ggml.impl.getOpParamsI32(op, 5);
    const p1 = ggml.impl.getOpParamsI32(op, 6);

    const IH: i64 = src0.ne[1];
    const IW: i64 = src0.ne[0];

    const N: i64 = op.ne[3];
    const OC: i64 = op.ne[2];
    const OH: i64 = op.ne[1];
    const OW: i64 = op.ne[0];

    const np: i64 = N * OC * OH * OW;

    var args_pool_2d: ggml.metal.kargs.pool_2d = .{
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(np)));
    const ntg = @divTrunc(np + nth - 1, nth);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args_pool_2d, 0);
    setBuffer(enc, src0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(ntg), 1, 1, nth, 1, 1);

    return 1;
}

/// How many parts `ggml_metal_op_argmax` splits a row of `ne00` into: one
/// up to `N_ARGMAX_CHUNK`, then a part per chunk, up to `N_ARGMAX_NPART_MAX`.
/// The C writes this twice, in `ggml_metal_op_argmax_extra` and
/// `ggml_metal_op_argmax` (ggml-metal-ops.cpp:5665, 5684 @b1b770d90).
fn argmaxNpart(ne00: i64) c_int {
    if (ne00 > ggml.metal.impl_c.N_ARGMAX_CHUNK) {
        return @truncate(@min(ggml.metal.impl_c.N_ARGMAX_NPART_MAX, @divTrunc(ne00 + ggml.metal.impl_c.N_ARGMAX_CHUNK - 1, ggml.metal.impl_c.N_ARGMAX_CHUNK)));
    }

    return 1;
}

/// Ports `ggml_metal_op_argmax_extra` (ggml-metal-ops.cpp:5665 @b1b770d90):
/// a partial max and a partial argmax per part of every row, after `dst`,
/// and nothing for a row short enough to take one pass. The C's `assert`
/// on the op compiles out in the reference and is not ported.
pub fn ggml_metal_op_argmax_extra(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    const ne00: i64 = src(op, 0).ne[0];
    const ne01: i64 = src(op, 0).ne[1];

    const npart = argmaxNpart(ne00);

    if (npart == 1) {
        return 0;
    }

    // size_t * int * int64_t: the C converts the signed operands to unsigned
    return (@sizeOf(f32) + @sizeOf(i32)) *% @as(usize, @bitCast(@as(i64, npart))) *% @as(usize, @bitCast(ne01));
}

/// Ports `ggml_metal_op_argmax` (ggml-metal-ops.cpp:5684 @b1b770d90).
///
/// A row over `N_ARGMAX_CHUNK` takes two kernels: one
/// threadgroup per part per row into the scratch `ggml_metal_op_argmax_extra`
/// sized, a barrier, then one threadgroup per row over the parts.
pub fn ggml_metal_op_argmax(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const ne00: i32 = @intCast(s0.ne[0]);
    const ne01: i32 = @intCast(s0.ne[1]);
    const ne02: i32 = @intCast(s0.ne[2]);
    const ne03: i32 = @intCast(s0.ne[3]);

    const nrows: i64 = ggml.types.ggml_nrows(s0);

    const npart = argmaxNpart(ne00);

    var args: ggml.metal.kargs.argmax = .{
        .ne00 = ne00,
        .nb01 = s0.nb[1],
        .npart = npart,
        .ne01 = ne01,
    };

    const smem: usize = 32 * (@sizeOf(f32) + @sizeOf(i32));

    if (npart > 1) {
        const pipeline_part = ggml.metal.library.ggml_metal_library_get_pipeline_argmax_part(lib);
        const pipeline_final = ggml.metal.library.ggml_metal_library_get_pipeline_argmax_final(lib);

        const bid_dst = getBufferId(op);
        var bid_tmp = bid_dst;
        bid_tmp.offs += @intCast(ggml.types.ggml_nbytes(op));

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_part);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, npart, @intCast(nrows), 1, 256, 1, 1);

        _ = concurrencyReset(ctx);

        var nth: c_int = 32; // SIMD width
        while (nth < npart and nth < 256) {
            nth *= 2;
        }

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline_final);
        setBytes(enc, &args, 0);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
        ggml.metal.device_c.ggml_metal_encoder_set_buffer(enc, bid_dst, 2);
        ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(nrows), 1, 1, nth, 1, 1);

        return 1;
    }

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_argmax(lib, op);

    var nth: i32 = 32; // SIMD width
    while (nth < ne00 and nth * ne01 * ne02 * ne03 < 256) {
        nth *= 2;
    }

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, s0, 1);
    setBuffer(enc, op, 2);

    ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(nrows), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_opt_step_adamw` (ggml-metal-ops.cpp:6227
/// @b1b770d90).
pub fn ggml_metal_op_opt_step_adamw(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const np: i64 = ggml.types.ggml_nelements(s0);
    var args: ggml.metal.kargs.opt_step_adamw = .{
        .np = np,
    };

    var ida: c_int = 0;

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, ida);
    ida += 1;
    // The C writes out five `set_buffer` calls for src[0..4].
    for (0..5) |j| {
        setBuffer(enc, src(op, j), ida);
        ida += 1;
    }

    const nth = @min(maxThreads(pipeline), @as(c_int, @intCast(s0.ne[0])));
    const n = @divTrunc(np + nth - 1, nth);

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_opt_step_sgd` (ggml-metal-ops.cpp:6263 @b1b770d90).
pub fn ggml_metal_op_opt_step_sgd(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const np: i64 = ggml.types.ggml_nelements(s0);
    var args: ggml.metal.kargs.opt_step_sgd = .{
        .np = np,
    };

    var ida: c_int = 0;

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
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

    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n), 1, 1, nth, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_count_equal` (ggml-metal-ops.cpp:6297 @b1b770d90).
///
/// Two dispatches with a concurrency reset between them: the destination
/// is zeroed first, then atomically accumulated into, so the second pass
/// must not overlap the first.
pub fn ggml_metal_op_count_equal(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const s0 = src(op, 0);
    const s1 = src(op, 1);

    {
        var args: ggml.metal.kargs.memset = .{ .val = 0 };

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_memset(lib, op);

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, op, 1);

        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    _ = concurrencyReset(ctx);

    {
        var args: ggml.metal.kargs.count_equal = .{
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

        const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_count_equal(lib, op);

        const smem = pipeline.smem;

        const nth = 32 * pipeline.nsg;

        ggml.impl.assert(nth <= maxThreads(pipeline), "nth <= max_theads_per_threadgroup");

        ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
        setBytes(enc, &args, 0);
        setBuffer(enc, s0, 1);
        setBuffer(enc, s1, 2);
        setBuffer(enc, op, 3);

        ggml.metal.device_c.ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(
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

/// Ports `ggml_metal_op_fwht` (ggml-metal-ops.cpp:2476 @b1b770d90).
///
/// Not reached from the dispatch switch: `ggml_metal_op_mul_mat` calls it
/// at its line 2597. Widths of `GGML_METAL_FWHT_TG_MIN_N` and
/// up run the threadgroup kernel, one row per threadgroup. It is an export of this translation unit all the
/// same. Note it reads `src[1]`, not `src[0]`, and an F16
/// `src[1]` too, which picks the kernel by name. The size check moved to
/// `ggml-metal-common.cpp` with `ggml_metal_op_mul_mat_use_fwht`.
pub fn ggml_metal_op_fwht(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const op = ctx.node(idx);

    const lib = ctx.lib;
    const enc = ctx.enc;

    const src1 = src(op, 1);

    const n: i64 = src1.ne[0];
    const nrows: i64 = ggml.types.ggml_nrows(src1);

    var args: ggml.metal.kargs.fwht = .{
        .nrows = @intCast(nrows),
    };

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_fwht(lib, @intCast(n), src1.type);

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, src1, 1);
    setBuffer(enc, op, 2);

    const th_max = maxThreads(pipeline);
    const simd_size: c_int = 32;

    if (n >= ggml.metal.impl_c.GGML_METAL_FWHT_TG_MIN_N) {
        ggml.impl.assert(th_max >= ggml.metal.impl_c.GGML_METAL_FWHT_TG_NT, "th_max >= GGML_METAL_FWHT_TG_NT");
        ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(nrows), 1, 1, ggml.metal.impl_c.GGML_METAL_FWHT_TG_NT, 1, 1);

        return 1;
    }

    var sg_per_tg: c_int = 2;
    sg_per_tg = @min(sg_per_tg, @divTrunc(th_max, simd_size));
    sg_per_tg = @max(sg_per_tg, 1);

    const n_tg = @divTrunc(nrows + @as(i64, sg_per_tg) - 1, @as(i64, sg_per_tg));
    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, @intCast(n_tg), 1, 1, 32 * sg_per_tg, 1, 1);

    return 1;
}

/// Ports `ggml_metal_op_snake_fused` (ggml-metal-ops.cpp:5279 @b1b770d90).
///
/// Dispatch the fused snake kernel from the matched mul -> sin -> sqr ->
/// mul -> add chain. `idx` points at the leading mul. The caller has
/// validated the chain. Returns **5**: it consumes the whole chain.
pub fn ggml_metal_op_snake_fused(ctx: *Op, idx: c_int) callconv(.c) c_int {
    const lib = ctx.lib;
    const enc = ctx.enc;

    const mul0 = ctx.node(idx + 0);
    const sqr = ctx.node(idx + 2);
    const mul1 = ctx.node(idx + 3);
    const add = ctx.node(idx + 4);

    const x = if (ggml.types.ggml_are_same_shape(mul0, src(mul0, 0))) src(mul0, 0) else src(mul0, 1);
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

    const pipeline = ggml.metal.library.ggml_metal_library_get_pipeline_snake(lib, x.type);

    var args: ggml.metal.kargs.snake = .{
        .T = T,
        .C = C,
    };

    ggml.metal.device_c.ggml_metal_encoder_set_pipeline(enc, pipeline);
    setBytes(enc, &args, 0);
    setBuffer(enc, x, 1);
    setBuffer(enc, a, 2);
    setBuffer(enc, inv_b, 3);
    setBuffer(enc, add, 4);

    const nth: c_int = 256;
    const ntg = @divTrunc(total + nth - 1, nth);
    ggml.metal.device_c.ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 5;
}

/// Ports `ggml_metal_op_mul_mat_id_max_tiles` (ggml-metal-ops.cpp:2918
/// @b1b770d90).
///
/// The upper bound on the number of entry tiles `kernel_mul_mm_id_map0`
/// emits: the full 32-row tiles over all `ne20*ne21` entries plus at most
/// one partial tile per expert, and never more tiles than entries. It
/// sizes both the row map in scratch and the matrix kernel's grid.
fn mulMatIdMaxTiles(op: *const ggml.impl.ggml_c.ggml_tensor) i64 {
    const ne02: i64 = src(op, 0).ne[2]; // n_expert
    const ne20: i64 = src(op, 2).ne[0]; // n_expert_used
    const ne21: i64 = src(op, 2).ne[1]; // n_token

    const n_rows = ne20 * ne21;
    const tiles = @divTrunc(n_rows + 31, 32) + (if (ne02 < n_rows) ne02 else n_rows) - 1;

    return if (n_rows < tiles) n_rows else tiles;
}

/// Ports `ggml_metal_op_mul_mat_id_extra_tpe` (ggml-metal-ops.cpp:2929
/// @b1b770d90).
///
/// Scratch for the row map of `kernel_mul_mm_id`, allocated after `dst` by
/// `backend.zig`'s `get_alloc_size`: the total tile count, then one
/// (expert, first entry, valid rows) descriptor per tile. The C's `assert`
/// on the op compiles out in the reference and is not ported.
pub fn ggml_metal_op_mul_mat_id_extra_tpe(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    const max_tiles = mulMatIdMaxTiles(op);

    // size_t * int64_t: the C converts the signed operand to unsigned
    return ggml.types.ggml_type_size(ggml.impl.ggml_c.GGML_TYPE_I32) *% @as(usize, @bitCast(1 + 3 * max_tiles));
}

/// Ports `ggml_metal_op_mul_mat_id_extra_ids` (ggml-metal-ops.cpp:2939
/// @b1b770d90).
///
/// One entry per (expert used, token) pair of ids plus one padding slot
/// per expert (see `kernel_mul_mm_id_map0`). The C's `assert` on the op
/// compiles out in the reference and is not ported.
pub fn ggml_metal_op_mul_mat_id_extra_ids(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    const ne02: i64 = src(op, 0).ne[2]; // n_expert
    const ne20: i64 = src(op, 2).ne[0]; // n_expert_used
    const ne21: i64 = src(op, 2).ne[1]; // n_token

    return ggml.types.ggml_type_size(ggml.impl.ggml_c.GGML_TYPE_I32) *% @as(usize, @bitCast(ne20 * ne21 + ne02));
}

/// Ports `ggml_metal_op_flash_attn_ext_use_vec` (ggml-metal-ops.cpp:3200
/// @b1b770d90).
pub fn ggml_metal_op_flash_attn_ext_use_vec(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const s0 = src(op, 0);

    const ne00: i64 = s0.ne[0]; // head size
    const ne01: i64 = s0.ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) and (@rem(ne00, 32) == 0);
}

/// Ports `ggml_metal_op_flash_attn_ext_use_kv_f16`
/// (ggml-metal-ops.cpp:3212 @b1b770d90).
///
/// ref: https://github.com/ggml-org/llama.cpp/pull/27390
/// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
fn useKvF16(op: *const ggml.impl.ggml_c.ggml_tensor) bool {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
    // TODO: tune per device
    // large heads need the upfront dequant to fit the non-vec threadgroup memory
    if (src(op, 0).ne[1] < 32 and
        (src(op, 0).ne[0] < 512 or ggml_metal_op_flash_attn_ext_use_vec(op)))
    {
        return false;
    }

    return switch (src(op, 1).type) {
        ggml.impl.ggml_c.GGML_TYPE_Q4_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0,
        => true,
        else => false,
    };
}

/// Ports `ggml_metal_op_flash_attn_ext_use_tensor`
/// (ggml-metal-ops.cpp:3236 @b1b770d90).
///
/// Whether the tensor-API kernel (`kernel_flash_attn_ext_tensor_f16_*`
/// and `_bf16_*`) takes this op: a device with
/// the tensor API, a batch too big for the vec kernel, one of six head-size
/// pairs, enough work to fill the GPU, F16 or BF16 K (or a quantized K the
/// KV-to-F16 pass converts) and
/// a KV length a whole number of 64-wide chunks -- the length
/// of the `kv_rows` lists when `src[5]` holds them, since the attention then
/// runs over the gathered rows. The C's `assert` on the op compiles out in
/// the reference and is not ported.
fn useTensor(op: *const ggml.impl.ggml_c.ggml_tensor, has_tensor: bool) bool {
    if (!has_tensor or ggml_metal_op_flash_attn_ext_use_vec(op)) {
        return false;
    }

    const ne01: i64 = src(op, 0).ne[1];
    const ne02: i64 = src(op, 0).ne[2];
    const ne03: i64 = src(op, 0).ne[3];

    const dk: i64 = src(op, 1).ne[0];
    const dv: i64 = src(op, 2).ne[0];

    const dk_dv_ok = (dk == 64 and dv == 64) or
        (dk == 128 and dv == 128) or
        (dk == 192 and dv == 128) or
        (dk == 256 and dv == 256) or
        (dk == 512 and dv == 512) or
        (dk == 576 and dv == 512);

    if (!dk_dv_ok) {
        return false;
    }

    // large heads use fewer queries per threadgroup, so that the queries fit in threadgroup memory
    const nqptg: i64 = if (dk >= 512) ggml.metal.impl_c.OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE else ggml.metal.impl_c.OP_FLASH_ATTN_EXT_TENSOR_NQPSG;

    // few heads and small batches do not fill the GPU - the half8x8 kernel is faster there
    // TODO: tune per device
    if (@divTrunc(ne01 + nqptg - 1, nqptg) * ne02 * ne03 * dk < 8192) {
        return false;
    }

    if (src(op, 1).type != ggml.impl.ggml_c.GGML_TYPE_F16 and src(op, 1).type != ggml.impl.ggml_c.GGML_TYPE_BF16 and !useKvF16(op)) {
        return false;
    }

    // with kv_rows, the attention runs over the compacted row lists (padded to 256), not over the cache
    const ne11_attn: i64 = if (srcOpt(op, 5)) |kv_rows| kv_rows.ne[0] else src(op, 1).ne[1];

    return @rem(ne11_attn, ggml.metal.impl_c.OP_FLASH_ATTN_EXT_TENSOR_NCPSG) == 0;
}

/// Ports `ggml_metal_op_flash_attn_ext_n_kv_max_sparse`
/// (ggml-metal-ops.cpp:3283 @b1b770d90).
///
/// The `n_kv_max` hint when the sparse path is available for this op, or 0.
/// The mask (`src[3]`) stays the single source of truth: its finite entries
/// are the valid KV positions, and `n_kv_max` only bounds their number per
/// mask row, to size the index lists. Only the `(type, dk, dv)` the vector
/// kernel is instantiated for qualify.
fn nKvMaxSparse(op: *const ggml.impl.ggml_c.ggml_tensor) c_int {
    // K/V row lists are handled by the dense kernels, not by the sparse path
    if (op.src[5] != null) {
        return 0;
    }

    const n_kv_max: c_int = ggml.impl.getOpParamsI32(op, 4);

    if (n_kv_max <= 0) {
        return 0;
    }

    // the sparse indices are gathered from the mask
    if (op.src[3] == null) {
        return 0;
    }

    // bound the size of the index lists
    if (n_kv_max > 4096) {
        return 0;
    }

    // vec kernel instantiations exist for these (type, dk, dv) combinations only
    const dk = src(op, 1).ne[0];
    const dv = src(op, 2).ne[0];

    const dk_dv_ok = (dk == 32 and dv == 32) or
        (dk == 64 and dv == 64) or
        (dk == 96 and dv == 96) or
        (dk == 96 and dv == 64) or
        (dk == 128 and dv == 128) or
        (dk == 128 and dv == 96) or
        (dk == 192 and dv == 128) or
        (dk == 192 and dv == 192) or
        (dk == 256 and dv == 256) or
        (dk == 320 and dv == 256) or
        (dk == 512 and dv == 512) or
        (dk == 576 and dv == 512);

    if (!dk_dv_ok) {
        return 0;
    }

    return switch (src(op, 1).type) {
        ggml.impl.ggml_c.GGML_TYPE_F16,
        ggml.impl.ggml_c.GGML_TYPE_BF16,
        ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0,
        => n_kv_max,
        else => 0,
    };
}

/// Ports `ggml_metal_op_flash_attn_ext_v_is_view_of_k`
/// (ggml-metal-ops.cpp:3349 @b1b770d90).
///
/// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
/// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
/// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
fn vIsViewOfK(op: *const ggml.impl.ggml_c.ggml_tensor) bool {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const K = src(op, 1);
    const V = src(op, 2);

    if (V.view_src == null) {
        return false;
    }
    const v_view = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, V.view_src);

    return v_view == K or
        (K.view_src != null and v_view == ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, K.view_src) and V.view_offs == K.view_offs);
}

/// Ports `ggml_metal_op_flash_attn_ext_kv_f16_k_size`
/// (ggml-metal-ops.cpp:3359 @b1b770d90).
///
/// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
fn kvF16KSize(op: *const ggml.impl.ggml_c.ggml_tensor) usize {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const s1 = src(op, 1);

    return stdlib.num.pad(usize, @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) * @as(usize, @intCast(s1.ne[0] * s1.ne[1] * s1.ne[2] * s1.ne[3])), 16);
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_pad`
/// (ggml-metal-ops.cpp:3367 @b1b770d90).
pub fn ggml_metal_op_flash_attn_ext_extra_pad(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

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
        nb11_pad = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) * @as(u64, @intCast(s1.ne[0]));
        nb21_pad = @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) * @as(u64, @intCast(if (v_is_view_of_k) s1.ne[0] else s2.ne[0]));
    }

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    //
    // Both are `impl_c.zig` constants, so the C's run-time `GGML_ASSERT`
    // becomes a compile-time one here.
    comptime {
        if (ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NCPSG < ggml.metal.impl_c.OP_FLASH_ATTN_EXT_VEC_NCPSG) {
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
            break :blk @as(u64, @intCast(ggml.types.ggml_type_size(ggml.impl.ggml_c.GGML_TYPE_F16))) *
                @as(u64, @intCast(m.ne[1] * m.ne[2] * m.ne[3]));
        } else 0;

        res += @as(usize, @intCast(ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NCPSG)) * @as(usize, @intCast(nb11_pad *
            @as(u64, @intCast(s1.ne[2] * s1.ne[3])) +
            nb21_pad * @as(u64, @intCast(s2.ne[2] * s2.ne[3])) +
            mask_term));
    }

    return res;
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_blk`
/// (ggml-metal-ops.cpp:3425 @b1b770d90).
pub fn ggml_metal_op_flash_attn_ext_extra_blk(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

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

    const nqptg: c_int = if (is_vec) ggml.metal.impl_c.OP_FLASH_ATTN_EXT_VEC_NQPSG else ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NQPSG;
    const ncpsg: c_int = if (is_vec) ggml.metal.impl_c.OP_FLASH_ATTN_EXT_VEC_NCPSG else ggml.metal.impl_c.OP_FLASH_ATTN_EXT_NCPSG;

    const ne1: i64 = @divTrunc(s0.ne[1] + nqptg - 1, nqptg);
    const ne0: i64 = @divTrunc(s3.ne[0] + ncpsg - 1, ncpsg);

    res += stdlib.num.pad(usize, @as(usize, @intCast(ggml.types.ggml_type_size(ggml.impl.ggml_c.GGML_TYPE_I8))) *
        @as(usize, @intCast(ne0 * ne1 * s3.ne[2] * s3.ne[3])), 32);

    return res;
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_tmp`
/// (ggml-metal-ops.cpp:3464 @b1b770d90).
pub fn ggml_metal_op_flash_attn_ext_extra_tmp(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

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
        res += @as(usize, @intCast(ggml.types.ggml_type_size(ggml.impl.ggml_c.GGML_TYPE_F32))) *
            @as(usize, @intCast(ne01_max * s0.ne[2] * s0.ne[3] * nwg * (s2.ne[0] + 2)));
    }

    return res;
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_kv_f16`
/// (ggml-metal-ops.cpp:3493 @b1b770d90).
pub fn ggml_metal_op_flash_attn_ext_extra_kv_f16(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    // note: always reserve the temp buffer to avoid graph reallocations
    // (the C's `!use_kv_f16` early return is commented out)

    const s2 = src(op, 2);

    const k_size = kvF16KSize(op);

    // when V is a view of K, the dequantized V is a view of the dequantized K
    const v_is_view_of_k = vIsViewOfK(op);
    if (v_is_view_of_k) {
        return k_size;
    }

    const v_size = stdlib.num.pad(usize, @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) *
        @as(usize, @intCast(s2.ne[0] * s2.ne[1] * s2.ne[2] * s2.ne[3])), 16);

    return k_size + v_size;
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_idx`
/// (ggml-metal-ops.cpp:3518 @b1b770d90).
///
/// The sparse index lists: one list of KV indices per mask row, padded with
/// -1 to a multiple of `OP_FLASH_ATTN_EXT_VEC_NCPSG`. Zero when the op does
/// not take the sparse path.
pub fn ggml_metal_op_flash_attn_ext_extra_idx(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    ggml.impl.assert(op.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const n_kv_max = nKvMaxSparse(op);

    if (n_kv_max <= 0) {
        return 0;
    }

    // `nKvMaxSparse` returned non-zero, so the mask is there.
    const s3 = src(op, 3);
    const ne31: usize = @intCast(s3.ne[1]);
    const ne32: usize = @intCast(s3.ne[2]);
    const ne33: usize = @intCast(s3.ne[3]);

    const n_kv_max_padded = stdlib.num.pad(usize, @intCast(n_kv_max), ggml.metal.impl_c.OP_FLASH_ATTN_EXT_VEC_NCPSG);

    return stdlib.num.pad(usize, @sizeOf(i32) * n_kv_max_padded * ne31 * ne32 * ne33, 16);
}

/// Ports `ggml_metal_op_flash_attn_ext_extra_rows`
/// (ggml-metal-ops.cpp:3536 @b1b770d90).
///
/// The contiguous F16 buffers holding the K/V rows a `kv_rows` op's lists
/// reference: K, then V unless V is a view of K (then the gathered V is
/// read from the gathered K). Zero without `src[5]`. Reserved whatever
/// path the op takes, as the C does: the vec path reads the rows through
/// the lists and never touches it. The C's `assert` on the op compiles out
/// in the reference and is not ported.
pub fn ggml_metal_op_flash_attn_ext_extra_rows(op: *const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    const kv_rows = srcOpt(op, 5) orelse return 0;

    const s1 = src(op, 1);
    const s2 = src(op, 2);

    // The C's `GGML_TENSOR_LOCALS(int32_t, ...)`, of which these are used.
    const ne10: i32 = @intCast(s1.ne[0]);
    const ne12: i32 = @intCast(s1.ne[2]);
    const ne20: i32 = @intCast(s2.ne[0]);
    const ne22: i32 = @intCast(s2.ne[2]);

    const n_rows: i64 = kv_rows.ne[0];
    const n_slices: i64 = kv_rows.ne[1];

    // note: when V is a view of K, the gathered V is read from the gathered K, so only K is gathered
    const v_is_view_of_k = vIsViewOfK(op);

    const k_size: usize = stdlib.num.pad(usize, @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) * @as(usize, @intCast(ne10)) * @as(usize, @intCast(n_rows)) * @as(usize, @intCast(ne12)) * @as(usize, @intCast(n_slices)), 16);
    const v_size: usize = if (v_is_view_of_k) 0 else stdlib.num.pad(usize, @sizeOf(ggml.impl.ggml_c.ggml_fp16_t) * @as(usize, @intCast(ne20)) * @as(usize, @intCast(n_rows)) * @as(usize, @intCast(ne22)) * @as(usize, @intCast(n_slices)), 16);

    return k_size + v_size;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
