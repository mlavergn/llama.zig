//! The op layer's core: the push-constant helpers every op shares,
//! `ggml_vk_op_f32` that most ops dispatch through, and the switch from a
//! graph node to its op.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.0
//! (`5266f24da`). The C++ specialises two templates per push-constant
//! struct -- `init_pushconst_fastdiv` and `init_pushconst_tensor_offsets`
//! -- and the port switches on the struct's type at comptime.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const common = m.common;
const allocator = common.allocator;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;
const Pipeline = m.pipeline.Pipeline;
const Subbuffer = m.buffer.Subbuffer;
const Tensor = c.ggml_tensor;
const CGraph = impl.CGraph;
const push = m.push;

/// `(uint32_t) x` of an `int64_t` or `size_t`: the low 32 bits.
pub inline fn u32Of(x: anytype) u32 {
    return @truncate(@as(u64, @bitCast(@as(i64, @intCast(x)))));
}

/// `(uint32_t) nb / type_size`: the C++ narrows the stride first.
pub inline fn strideIn(nb: usize, type_size: u32) u32 {
    return @as(u32, @truncate(nb)) / type_size;
}

pub fn ceilDiv(a: anytype, b: anytype) @TypeOf(a, b) {
    return @divTrunc(a + b - 1, b);
}

pub fn typeSize(t: *const Tensor) u32 {
    return @intCast(c.ggml_type_size(t.type));
}

// -----------------------------------------------------------------------------
// Fast division

/// Ports `init_fastdiv_values` (ggml-vulkan.cpp:1585 @5266f24da): `mp` and
/// `L` with `n / d == (mulhi(n, mp) + n) >> L`.
pub fn initFastdivValues(d: u32, mp: *u32, L: *u32) void {
    // compute L = ceil(log2(d));
    L.* = 0;
    while (L.* < 32 and (@as(u64, 1) << @intCast(L.*)) < d) L.* += 1;
    // `uint64_t` arithmetic: the product wraps when L is 32.
    mp.* = @truncate(((@as(u64, 1) << 32) *% ((@as(u64, 1) << @intCast(L.*)) - d)) / d + 1);
}

/// Ports `pack_fastdiv_L` (ggml-vulkan.cpp:1596 @5266f24da).
pub fn packFastdivL(L0: u32, L1: u32, L2: u32) u32 {
    return L0 | (L1 << 8) | (L2 << 16);
}

/// Ports `init_pushconst_fastdiv` and its specialisations for
/// `vk_op_unary_push_constants`, `vk_op_glu_push_constants`,
/// `vk_op_count_experts_push_constants`, `vk_op_conv2d_push_constants`,
/// `vk_op_conv3d_push_constants` and `vk_op_sum_rows_push_constants`
/// (ggml-vulkan.cpp:1600, 1605, 1625, 1632, 1978, 2013, 2074 @5266f24da).
pub fn initPushconstFastdiv(p: anytype) void {
    const T = @TypeOf(p.*);
    if (T == push.Unary) {
        var l: [6]u32 = undefined;
        initFastdivValues(p.ne02 *% p.ne01 *% p.ne00, &p.ne0_012mp, &l[0]);
        initFastdivValues(p.ne01 *% p.ne00, &p.ne0_01mp, &l[1]);
        initFastdivValues(p.ne00, &p.ne0_0mp, &l[2]);
        initFastdivValues(p.ne12 *% p.ne11 *% p.ne10, &p.ne1_012mp, &l[3]);
        initFastdivValues(p.ne11 *% p.ne10, &p.ne1_01mp, &l[4]);
        initFastdivValues(p.ne10, &p.ne1_0mp, &l[5]);
        p.ne0_Ls = packFastdivL(l[0], l[1], l[2]);
        p.ne1_Ls = packFastdivL(l[3], l[4], l[5]);
    } else if (T == push.Glu) {
        // GLU linearizes over dst, then uses dst coordinates for src0/src1.
        initFastdivValues(p.ne22 *% p.ne21 *% p.ne20, &p.ne2_012mp, &p.ne2_012L);
        initFastdivValues(p.ne21 *% p.ne20, &p.ne2_01mp, &p.ne2_01L);
        initFastdivValues(p.ne20, &p.ne2_0mp, &p.ne2_0L);
    } else if (T == push.CountExperts) {
        initFastdivValues(p.ne00, &p.ne00mp, &p.ne00L);
    } else if (T == push.Conv2d) {
        initFastdivValues(p.OW, &p.OWmp, &p.OWL);
        initFastdivValues(p.OW *% p.OH, &p.OWOHmp, &p.OWOHL);
    } else if (T == push.Conv3d) {
        initFastdivValues(p.OW, &p.OWmp, &p.OWL);
        initFastdivValues(p.OW *% p.OH, &p.OWOHmp, &p.OWOHL);
        initFastdivValues(p.OW *% p.OH *% p.OD, &p.OWOHODmp, &p.OWOHODL);
    } else if (T == push.SumRows) {
        initFastdivValues(p.ne01 *% p.ne02, &p.ne0_12mp, &p.ne0_12L);
        initFastdivValues(p.ne01, &p.ne0_1mp, &p.ne0_1L);
    }
}

// -----------------------------------------------------------------------------
// Push-constant initialisers

/// Ports `vk_op_unary_push_constants_init` (ggml-vulkan.cpp:1495 @5266f24da).
/// `ne` 0 means the destination's element count.
pub fn unaryPushConstantsInit(src0: *const Tensor, dst: *const Tensor, ne_in: i64) push.Unary {
    impl.assert(ne_in != 0 or c.ggml_nelements(src0) == c.ggml_nelements(dst), "ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst))");
    const ne = if (ne_in != 0) ne_in else c.ggml_nelements(dst);
    impl.assert(ne <= std.math.maxInt(u32), "ne <= (int64_t)std::numeric_limits<uint32_t>::max()");

    var p: push.Unary = .{};
    p.ne = @intCast(ne);
    const ts0 = c.ggml_type_size(src0.type);
    p.ne00 = u32Of(src0.ne[0]);
    p.ne01 = u32Of(src0.ne[1]);
    p.ne02 = u32Of(src0.ne[2]);
    p.ne03 = u32Of(src0.ne[3]);
    p.nb00 = @truncate(src0.nb[0] / ts0);
    p.nb01 = @truncate(src0.nb[1] / ts0);
    p.nb02 = @truncate(src0.nb[2] / ts0);
    p.nb03 = @truncate(src0.nb[3] / ts0);
    const tsd = c.ggml_type_size(dst.type);
    p.ne10 = u32Of(dst.ne[0]);
    p.ne11 = u32Of(dst.ne[1]);
    p.ne12 = u32Of(dst.ne[2]);
    p.ne13 = u32Of(dst.ne[3]);
    p.nb10 = @truncate(dst.nb[0] / tsd);
    p.nb11 = @truncate(dst.nb[1] / tsd);
    p.nb12 = @truncate(dst.nb[2] / tsd);
    p.nb13 = @truncate(dst.nb[3] / tsd);
    return p; // offsets are initialized later in ggml_vk_op
}

/// Ports `vk_op_pad_push_constants_init` (ggml-vulkan.cpp:1539 @5266f24da).
pub fn padPushConstantsInit(src0: *const Tensor, dst: *const Tensor) push.Pad {
    const ne = c.ggml_nelements(dst);
    impl.assert(ne <= std.math.maxInt(u32), "ne <= (int64_t)std::numeric_limits<uint32_t>::max()");
    var p: push.Pad = .{};
    p.ne = @intCast(ne);
    const ts0 = c.ggml_type_size(src0.type);
    p.ne00 = u32Of(src0.ne[0]);
    p.ne01 = u32Of(src0.ne[1]);
    p.ne02 = u32Of(src0.ne[2]);
    p.ne03 = u32Of(src0.ne[3]);
    p.nb00 = @truncate(src0.nb[0] / ts0);
    p.nb01 = @truncate(src0.nb[1] / ts0);
    p.nb02 = @truncate(src0.nb[2] / ts0);
    p.nb03 = @truncate(src0.nb[3] / ts0);
    const tsd = c.ggml_type_size(dst.type);
    p.ne10 = u32Of(dst.ne[0]);
    p.ne11 = u32Of(dst.ne[1]);
    p.ne12 = u32Of(dst.ne[2]);
    p.ne13 = u32Of(dst.ne[3]);
    p.nb10 = @truncate(dst.nb[0] / tsd);
    p.nb11 = @truncate(dst.nb[1] / tsd);
    p.nb12 = @truncate(dst.nb[2] / tsd);
    p.nb13 = @truncate(dst.nb[3] / tsd);
    const op: [*]const u32 = @ptrCast(&dst.op_params);
    p.lp0 = op[0];
    p.rp0 = op[1];
    p.lp1 = op[2];
    p.rp1 = op[3];
    p.lp2 = op[4];
    p.rp2 = op[5];
    p.lp3 = op[6];
    p.rp3 = op[7];
    p.circular = op[8];
    return p; // fastdiv values and offsets are initialized later in ggml_vk_op
}

/// Ports `vk_op_sum_rows_push_constants_init` (ggml-vulkan.cpp:2058 @5266f24da).
pub fn sumRowsPushConstantsInit(src: *const Tensor, dst: *const Tensor, n_cols: i64) push.SumRows {
    const type_size: u32 = @sizeOf(f32);
    var p: push.SumRows = .{};
    p.n_cols = u32Of(n_cols);
    p.ne01 = u32Of(src.ne[1]);
    p.ne02 = u32Of(src.ne[2]);
    p.nb01 = strideIn(src.nb[1], type_size);
    p.nb02 = strideIn(src.nb[2], type_size);
    p.nb03 = strideIn(src.nb[3], type_size);
    p.nb11 = strideIn(dst.nb[1], type_size);
    p.nb12 = strideIn(dst.nb[2], type_size);
    p.nb13 = strideIn(dst.nb[3], type_size);
    p.weight = 1.0;
    return p;
}

/// The `vk_op_binary_push_constants` initialiser most binary ops spell out:
/// every shape and stride in elements, `ne` first and the params last.
pub fn binaryPc(ne: u32, src0: *const Tensor, src1: *const Tensor, dst: *const Tensor, param1: f32, param2: f32, param3: i32) push.Binary {
    const s0 = typeSize(src0);
    const s1 = typeSize(src1);
    const sd = typeSize(dst);
    return .{
        .ne = ne,
        .ne00 = u32Of(src0.ne[0]),
        .ne01 = u32Of(src0.ne[1]),
        .ne02 = u32Of(src0.ne[2]),
        .ne03 = u32Of(src0.ne[3]),
        .nb00 = strideIn(src0.nb[0], s0),
        .nb01 = strideIn(src0.nb[1], s0),
        .nb02 = strideIn(src0.nb[2], s0),
        .nb03 = strideIn(src0.nb[3], s0),
        .ne10 = u32Of(src1.ne[0]),
        .ne11 = u32Of(src1.ne[1]),
        .ne12 = u32Of(src1.ne[2]),
        .ne13 = u32Of(src1.ne[3]),
        .nb10 = strideIn(src1.nb[0], s1),
        .nb11 = strideIn(src1.nb[1], s1),
        .nb12 = strideIn(src1.nb[2], s1),
        .nb13 = strideIn(src1.nb[3], s1),
        .ne20 = u32Of(dst.ne[0]),
        .ne21 = u32Of(dst.ne[1]),
        .ne22 = u32Of(dst.ne[2]),
        .ne23 = u32Of(dst.ne[3]),
        .nb20 = strideIn(dst.nb[0], sd),
        .nb21 = strideIn(dst.nb[1], sd),
        .nb22 = strideIn(dst.nb[2], sd),
        .nb23 = strideIn(dst.nb[3], sd),
        .misalign_offsets = 0,
        .param1 = param1,
        .param2 = param2,
        .param3 = param3,
    };
}

// -----------------------------------------------------------------------------
// Misalignment offsets

fn misalignIn(ctx: *const BackendCtx, t: *const Tensor, unit: u32) u32 {
    return m.context.getMisalignBytes(ctx, t) / unit;
}

/// Ports `init_pushconst_tensor_offsets` (ggml-vulkan.cpp:2530 @5266f24da), the
/// primary template, and its specialisations for the MatVecP021, MatVecNc,
/// Fwht, Unary, Glu, SumRows, Pad, Im2col3d, Binary, Concat, Upscale and
/// Rope structs (`init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`
/// (ggml-vulkan.cpp:2545, 2557, 2569, 12041, 12052, 12067, 12078, 12089, 12100, 12113, 12125, 12137 @5266f24da)):
/// the elements by which each binding's start was rounded down, packed
/// as the struct's shader expects; for a struct without them, the
/// assertion that nothing was misaligned.
pub fn initPushconstTensorOffsets(ctx: *const BackendCtx, p: anytype, src0: ?*const Tensor, src1: ?*const Tensor, src2: ?*const Tensor, src3: ?*const Tensor, dst: ?*const Tensor) void {
    const T = @TypeOf(p.*);
    if (T == push.MatVecP021 or T == push.MatVecNc) {
        p.b_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        p.d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else if (T == push.Fwht) {
        p.src_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        p.dst_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else if (T == push.Unary or T == push.SumRows or T == push.Pad) {
        const a_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        const d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
        p.misalign_offsets = (a_offset << 16) | d_offset;
    } else if (T == push.Glu) {
        const a_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        const b_offset = if (src1) |s1| misalignIn(ctx, s1, typeSize(s1)) else a_offset;
        const d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
        impl.assert(a_offset < (1 << 8), "a_offset < (1u << 8)");
        impl.assert(b_offset < (1 << 8), "b_offset < (1u << 8)");
        impl.assert(d_offset < (1 << 8), "d_offset < (1u << 8)");
        p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;
    } else if (T == push.Im2col3d) {
        const a_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        const d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
        p.misalign_offsets = (a_offset << 16) | d_offset;
    } else if (T == push.Binary) {
        const a_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        const b_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        const d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
        impl.assert(dst.?.op != c.GGML_OP_GET_ROWS or (a_offset == 0 and b_offset == 0 and d_offset == 0), "dst->op != GGML_OP_GET_ROWS || (a_offset == 0 && b_offset == 0 && d_offset == 0)");
        p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;
    } else if (T == push.Concat) {
        const unit_size = m.op_pipeline.concatUnitSize(dst.?.type);
        p.misalign_offsets = (misalignIn(ctx, src0.?, unit_size) << 16) | (misalignIn(ctx, src1.?, unit_size) << 8) | misalignIn(ctx, dst.?, unit_size);
    } else if (T == push.Upscale or T == push.Rope) {
        p.a_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        p.d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else {
        impl.assert(src0 == null or m.context.getMisalignBytes(ctx, src0.?) == 0, "!src0 || get_misalign_bytes(ctx, src0) == 0");
        impl.assert(src1 == null or m.context.getMisalignBytes(ctx, src1.?) == 0, "!src1 || get_misalign_bytes(ctx, src1) == 0");
        impl.assert(src2 == null or m.context.getMisalignBytes(ctx, src2.?) == 0, "!src2 || get_misalign_bytes(ctx, src2) == 0");
        impl.assert(src3 == null or m.context.getMisalignBytes(ctx, src3.?) == 0, "!src3 || get_misalign_bytes(ctx, src3) == 0");
        impl.assert(dst == null or m.context.getMisalignBytes(ctx, dst.?) == 0, "!dst  || get_misalign_bytes(ctx, dst) == 0");
    }
}

// -----------------------------------------------------------------------------
// The generic dispatch

/// The `{ 512, 512, z }` split of a one-dimensional count every op uses.
pub fn split512(ne: u32) [3]u32 {
    if (ne > 262144) return .{ 512, 512, ceilDiv(ne, 262144) };
    if (ne > 512) return .{ 512, ceilDiv(ne, 512), 1 };
    return .{ ne, 1, 1 };
}

/// Ports `ggml_vk_op_f32` (ggml-vulkan.cpp:12147 @5266f24da): picks the
/// op's pipeline, finishes the push constants, sizes the dispatch for the
/// op, and binds the sources and destination as the op's shader expects.
pub fn opF32(ctx: *BackendCtx, subctx: *Context, src0: ?*const Tensor, src1: ?*const Tensor, src2: ?*const Tensor, src3: ?*const Tensor, dst: *Tensor, op: c.enum_ggml_op, pc_in: anytype) void {
    var pc = pc_in;
    const PC = @TypeOf(pc);
    const s0 = src0.?;
    impl.assert(op == c.GGML_OP_GET_ROWS or op == c.GGML_OP_CPY or op == c.GGML_OP_CONCAT or (!c.ggml_is_quantized(s0.type) and (src1 == null or !c.ggml_is_quantized(src1.?.type))), "op == GGML_OP_GET_ROWS || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type)))");
    impl.assert(dst.buffer != null, "dst->buffer != nullptr");
    const ne00: u64 = @bitCast(s0.ne[0]);
    const ne01: u64 = @bitCast(s0.ne[1]);
    const ne02: u64 = @bitCast(s0.ne[2]);
    const ne03: u64 = @bitCast(s0.ne[3]);

    const use_src1 = src1 != null;
    const ne10: u64 = if (use_src1) @bitCast(src1.?.ne[0]) else 0;
    const ne11: u64 = if (use_src1) @bitCast(src1.?.ne[1]) else 0;
    const ne12: u64 = if (use_src1) @bitCast(src1.?.ne[2]) else 0;
    const ne13: u64 = if (use_src1) @bitCast(src1.?.ne[3]) else 0;
    const use_src2 = src2 != null;
    const use_src3 = src3 != null;

    initPushconstFastdiv(&pc);

    const pipeline = m.op_pipeline.getPipeline(ctx, s0, src1, src2, dst, op) orelse {
        impl.fprintfStderr("ggml_vulkan: Error: Missing op: %s for %s", .{ c.ggml_op_name(op), c.ggml_type_name(s0.type) });
        if (src1) |s1| impl.fprintfStderr(" and %s", .{c.ggml_type_name(s1.type)});
        impl.fprintfStderr(" to %s\n", .{c.ggml_type_name(dst.type)});
        impl.abort("fatal error");
    };

    m.context.requestDescriptorSets(ctx, pipeline, 1);

    const src0_buf = m.context.tensorSubbuffer(ctx, s0, true);
    const src1_buf: Subbuffer = if (use_src1) m.context.tensorSubbuffer(ctx, src1.?, true) else .{};
    const src2_buf: Subbuffer = if (use_src2) m.context.tensorSubbuffer(ctx, src2.?, true) else .{};
    const src3_buf: Subbuffer = if (use_src3) m.context.tensorSubbuffer(ctx, src3.?, true) else .{};
    var dst_buf = m.context.tensorSubbuffer(ctx, dst, true);

    // Compute misalignment offset for descriptors and store it in in push constants.
    initPushconstTensorOffsets(ctx, &pc, s0, src1, src2, src3, dst);

    const limits = ctx.device.properties.limits;
    var elements: [3]u32 = undefined;
    switch (op) {
        c.GGML_OP_NORM, c.GGML_OP_RMS_NORM_BACK, c.GGML_OP_L2_NORM, c.GGML_OP_SOFT_MAX, c.GGML_OP_SOFT_MAX_BACK, c.GGML_OP_SUM_ROWS, c.GGML_OP_CUMSUM, c.GGML_OP_MEAN, c.GGML_OP_ARGMAX => {
            elements = split512(u32Of(c.ggml_nrows(s0)));
        },
        c.GGML_OP_SOLVE_TRI => elements = split512(@truncate(ne02 *% ne03)),
        c.GGML_OP_RMS_NORM => {
            if (ctx.do_add_rms_partials) {
                // Run one element per thread, 128 threads per workgroup
                elements = .{ @truncate(ceilDiv(ne00, 128)), 1, 1 };
            } else {
                elements = .{ @truncate(ne01), @truncate(ne02), @truncate(ne03) };
            }
        },
        // We use GGML_OP_SUM_ROWS with 1 row.
        c.GGML_OP_SUM => elements = .{ 1, 1, 1 },
        c.GGML_OP_GROUP_NORM => {
            const num_groups: u32 = @bitCast(dst.op_params[0]);
            elements = .{ num_groups *% u32Of(s0.ne[3]), 1, 1 };
        },
        c.GGML_OP_DIAG_MASK_INF => elements = .{ u32Of(c.ggml_nrows(s0)), @truncate(ne00), 1 },
        c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => {
            var nrows = u32Of(c.ggml_nrows(s0));
            var z: u32 = 1;
            if (nrows > limits.maxComputeWorkGroupCount[0]) {
                z = ceilDiv(nrows, 32768);
                nrows = 32768;
            }
            elements = .{ nrows, @truncate(ne00), z };
        },
        c.GGML_OP_GET_ROWS => {
            elements = .{ @truncate(ne00), @truncate(ne10), @truncate(ne11 *% ne12) };
            elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
            elements[2] = @min(elements[2], limits.maxComputeWorkGroupCount[2]);
        },
        c.GGML_OP_GET_ROWS_BACK => {
            elements = .{ u32Of(dst.ne[0]), u32Of(dst.ne[1]), 1 };
            elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
        },
        c.GGML_OP_ARGSORT => impl.abort("GGML_ASSERT(0) failed"),
        c.GGML_OP_IM2COL => {
            const s1 = src1.?;
            const is_2D = dst.op_params[6] == 1;
            const IC = u32Of(s1.ne[if (is_2D) 2 else 1]);
            const KH: u32 = if (is_2D) u32Of(s0.ne[1]) else 1;
            const KW = u32Of(s0.ne[0]);
            const OH: u32 = if (is_2D) u32Of(dst.ne[2]) else 1;
            const OW = u32Of(dst.ne[1]);
            const batch = u32Of(s1.ne[if (is_2D) 3 else 2]);
            const CHW = IC *% KH *% KW;
            // Cap X workgroups to limit concurrent IC channel reads. AMD prefers a lower limit.
            const min_cap: u32 = if (ctx.device.vendor_id == common.vendor_id.amd) 512 else 4096;
            const x_elements = @min(CHW, @max(min_cap, OW *% KH *% KW));
            elements = .{ x_elements, OW, OH *% batch };
            elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
            elements[2] = @min(elements[2], limits.maxComputeWorkGroupCount[2]);
        },
        c.GGML_OP_IM2COL_3D => {
            const IC: u32 = @bitCast(dst.op_params[9]);
            const N: u32 = @truncate(ne13 / IC);
            const KD: u32 = @truncate(ne02);
            const KH: u32 = @truncate(ne01);
            const KW: u32 = @truncate(ne00);
            const OD: u32 = @truncate(@as(u64, @bitCast(dst.ne[3])) / N);
            const OH = u32Of(dst.ne[2]);
            const OW = u32Of(dst.ne[1]);
            elements = .{ IC *% KD *% KH *% KW, OW, N *% OD *% OH };
            elements[2] = @min(elements[2], limits.maxComputeWorkGroupCount[2]);
        },
        c.GGML_OP_TIMESTEP_EMBEDDING => {
            const dim: u32 = @bitCast(dst.op_params[0]);
            elements = .{ (dim +% 1) / 2, u32Of(s0.ne[0]), 1 };
        },
        // parallelize in {Cout, 1, 1}
        c.GGML_OP_CONV_TRANSPOSE_1D => elements = .{ u32Of(s0.ne[1]), 1, 1 },
        c.GGML_OP_COL2IM_1D => elements = .{ u32Of(dst.ne[0]), u32Of(dst.ne[1]), 1 },
        c.GGML_OP_POOL_1D => {
            const N = u32Of(dst.ne[3] *% dst.ne[2]);
            elements = .{ N *% u32Of(dst.ne[1]) *% u32Of(dst.ne[0]), 1, 1 };
        },
        c.GGML_OP_POOL_2D => elements = .{ u32Of(dst.ne[3]) *% u32Of(dst.ne[2]) *% u32Of(dst.ne[1]) *% u32Of(dst.ne[0]), 1, 1 },
        c.GGML_OP_CONV_2D, c.GGML_OP_CONV_TRANSPOSE_2D => {
            if (PC != push.Conv2d) impl.abort("invalid push constant type for CONV_2D");
            const NPQ = pc.N *% pc.OH *% pc.OW;
            const shape = m.op_pipeline.convSelectShape(ctx, pc.Cout, NPQ);
            const npq_blocks = ceilDiv(NPQ, m.backend.conv_block_sizes[@backingInt(shape)].NPQ);
            elements = .{ pc.Cout, npq_blocks, 1 };
            if (elements[1] > 512) {
                elements[2] = ceilDiv(elements[1], 512);
                elements[1] = 512;
            }
        },
        c.GGML_OP_CONV_3D => {
            if (PC != push.Conv3d) impl.abort("invalid push constant type for CONV_3D");
            const NPQ = pc.N *% pc.OD *% pc.OH *% pc.OW;
            const shape = m.op_pipeline.convSelectShape(ctx, pc.OC, NPQ);
            const npq_blocks = ceilDiv(NPQ, m.backend.conv_block_sizes[@backingInt(shape)].NPQ);
            elements = .{ pc.OC, npq_blocks, 1 };
            if (elements[1] > 512) {
                elements[2] = ceilDiv(elements[1], 512);
                elements[1] = 512;
            }
        },
        c.GGML_OP_ADD, c.GGML_OP_SUB, c.GGML_OP_DIV, c.GGML_OP_MUL, c.GGML_OP_ADD1, c.GGML_OP_OUT_PROD, c.GGML_OP_ARANGE, c.GGML_OP_FILL, c.GGML_OP_SCALE, c.GGML_OP_SQR, c.GGML_OP_SQRT, c.GGML_OP_SIN, c.GGML_OP_COS, c.GGML_OP_LOG, c.GGML_OP_TRI, c.GGML_OP_DIAG, c.GGML_OP_CLAMP, c.GGML_OP_LEAKY_RELU, c.GGML_OP_PAD, c.GGML_OP_PAD_REFLECT_1D, c.GGML_OP_ROLL, c.GGML_OP_REPEAT, c.GGML_OP_REPEAT_BACK, c.GGML_OP_CPY, c.GGML_OP_CONCAT, c.GGML_OP_UPSCALE, c.GGML_OP_UNARY, c.GGML_OP_GLU, c.GGML_OP_CONV_2D_DW => {
            var ne = u32Of(c.ggml_nelements(dst));
            if (op == c.GGML_OP_CPY and c.ggml_is_quantized(s0.type) and c.ggml_is_quantized(dst.type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= u32Of(c.ggml_blck_size(s0.type));
                const ts = typeSize(s0);
                if (ts % 4 == 0) ne *%= ts / 4 else ne *%= ts / 2;
            }
            if (op == c.GGML_OP_CONCAT and c.ggml_is_quantized(dst.type)) {
                ne = ne / u32Of(c.ggml_blck_size(dst.type)) *% typeSize(dst) / m.op_pipeline.concatUnitSize(dst.type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            if (op == c.GGML_OP_CPY and !c.ggml_is_quantized(s0.type) and c.ggml_is_quantized(dst.type)) {
                ne = ceilDiv(ne, u32Of(c.ggml_blck_size(dst.type)));
            }
            elements = split512(ne);

            const dev = ctx.device;
            if (pipeline == dev.pipeline_cpy_transpose_02_32 or pipeline == dev.pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = u32Of(ceilDiv(dst.ne[0], 32));
                elements[1] = u32Of(ceilDiv(dst.ne[2], 32));
                elements[2] = u32Of(dst.ne[1] *% dst.ne[3]);
                elements[0] = @min(elements[0], limits.maxComputeWorkGroupCount[0]);
                elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
                elements[2] = @min(elements[2], limits.maxComputeWorkGroupCount[2]);
            } else if (pipeline == dev.pipeline_cpy_transpose_32 or pipeline == dev.pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = u32Of(ceilDiv(dst.ne[0], 32));
                elements[1] = u32Of(ceilDiv(dst.ne[1], 32));
                elements[2] = u32Of(dst.ne[2] *% dst.ne[3]);
                elements[0] = @min(elements[0], limits.maxComputeWorkGroupCount[0]);
                elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
                elements[2] = @min(elements[2], limits.maxComputeWorkGroupCount[2]);
            }
        },
        c.GGML_OP_ADD_ID => elements = .{ @truncate(ne01), @truncate(ne02), 1 },
        c.GGML_OP_SET_ROWS => {
            var ne = u32Of(c.ggml_nelements(s0));
            if (c.ggml_is_quantized(dst.type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = ceilDiv(ne, 32 *% u32Of(c.ggml_blck_size(dst.type)));
            } else {
                // scalar types do one element per thread, running 512 threads
                ne = ceilDiv(ne, 512);
            }
            elements = split512(ne);
        },
        c.GGML_OP_SSM_CONV => elements = .{ u32Of(s0.ne[1]), u32Of(dst.ne[1]), u32Of(dst.ne[2]) },
        else => elements = .{ u32Of(c.ggml_nelements(s0)), 1, 1 },
    }

    const pcb = std.mem.asBytes(&pc);
    if (op == c.GGML_OP_ADD or op == c.GGML_OP_RMS_NORM) {
        var a_buf = src0_buf;
        if (ctx.do_add_rms_partials) a_buf = m.context.subbuffer(ctx, ctx.prealloc_add_rms_partials.?, ctx.prealloc_size_add_rms_partials_offset);
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, dst_buf, a_buf }, pcb, elements);
    } else if (op == c.GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, if (use_src1) src1_buf else src0_buf, dst_buf }, pcb, elements);
    } else if (op == c.GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, if (use_src1) src1_buf else src0_buf, if (use_src2) src2_buf else src0_buf, dst_buf }, pcb, elements);
    } else if (op == c.GGML_OP_ROPE or op == c.GGML_OP_ROPE_BACK) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, if (use_src2) src2_buf else src0_buf, dst_buf, if (use_src3) src3_buf else src0_buf }, pcb, elements);
    } else if (op == c.GGML_OP_IM2COL or op == c.GGML_OP_IM2COL_3D) {
        // buffer device address path doesn't use dst buffer
        if (ctx.device.shader_int64 and ctx.device.buffer_device_address) dst_buf.size = 1;
        // im2col uses only src1 and dst buffers
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src1_buf, dst_buf }, pcb, elements);
    } else if (op == c.GGML_OP_COUNT_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        m.submit.bufferMemsetAsync(subctx, dst_buf.buffer.?, dst_buf.offset, 0, dst_buf.size);
        m.context.syncBuffers(ctx, subctx);
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, dst_buf }, pcb, elements);
    } else if (op == c.GGML_OP_OPT_STEP_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, src2_buf }, pcb, elements);
    } else if (use_src3) {
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pcb, elements);
    } else if (use_src2) {
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, src2_buf, dst_buf }, pcb, elements);
    } else if (use_src1) {
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, dst_buf }, pcb, elements);
    } else {
        m.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, dst_buf }, pcb, elements);
    }
}

// -----------------------------------------------------------------------------
// Shared helpers

/// Ports `ggml_vk_use_mul_mat_vec_id` (ggml-vulkan.cpp:10786 @5266f24da).
pub fn useMulMatVecId(cgraph: *const CGraph, node_idx: c_int) bool {
    const dst = cgraph.nodes[@intCast(node_idx)].?;
    const src0 = dst.src[0].*;
    const src2 = dst.src[2].*;
    return src2.ne[1] <= 8 and (src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16 or c.ggml_is_quantized(src0.type));
}

/// Ports `ggml_vk_rms_num_partials` (ggml-vulkan.cpp:13421 @5266f24da).
pub fn rmsNumPartials(ctx: *const BackendCtx, node: *const Tensor) u32 {
    const ne = u32Of(node.ne[0]);
    const denom = ctx.device.pipeline_add_rms[0][0][0].?.wg_denoms[0];
    return ceilDiv(ne, denom);
}

/// Ports `ggml_vk_rms_partials_size` (ggml-vulkan.cpp:13428 @5266f24da).
pub fn rmsPartialsSize(ctx: *const BackendCtx, node: *const Tensor) u32 {
    const num_partials = rmsNumPartials(ctx, node);
    const a = ctx.device.partials_binding_alignment;
    // ROUNDUP_POW2
    return (num_partials *% @sizeOf(u32) + a - 1) & ~(a - 1);
}

// -----------------------------------------------------------------------------
// The node switch

/// The `switch (node->op)` of `ggml_vk_build_graph` (ggml-vulkan.cpp:15667 @5266f24da).
///
/// Return: false where the C++ returns false, for an op it has no
/// dispatch for.
pub fn dispatchNode(ctx: *BackendCtx, compute_ctx: *Context, cgraph: *CGraph, node_idx: c_int, node: *Tensor, src0_: ?*Tensor, src1_: ?*Tensor, src2_: ?*Tensor, src3_: ?*Tensor) bool {
    const B = m.ops_basic;
    const R = m.ops_reduce;
    const M = m.ops_more;
    const V = m.ops_conv;
    const src0 = src0_;
    const src1 = src1_;
    switch (node.op) {
        c.GGML_OP_REPEAT, c.GGML_OP_REPEAT_BACK => B.unaryDstCount(ctx, compute_ctx, src0.?, node, node.op),
        c.GGML_OP_ACC, c.GGML_OP_SET => B.acc(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_GET_ROWS => if (ctx.fused_topk_qsa) R.topkQsa(ctx, compute_ctx, cgraph, node_idx) else B.getRows(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_GET_ROWS_BACK => B.getRowsBack(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_ADD => if (ctx.num_additional_fused_ops != 0) B.multiAdd(ctx, compute_ctx, cgraph, node_idx) else B.add(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_OUT_PROD => B.outProd(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_SUB, c.GGML_OP_DIV, c.GGML_OP_ADD1 => B.binary(ctx, compute_ctx, src0.?, src1.?, node, node.op),
        c.GGML_OP_MUL => if (ctx.num_additional_fused_ops != 0) B.snakeFused(ctx, compute_ctx, cgraph, node_idx) else B.binary(ctx, compute_ctx, src0.?, src1.?, node, c.GGML_OP_MUL),
        c.GGML_OP_ADD_ID => B.addId(ctx, compute_ctx, src0.?, src1.?, src2_.?, node),
        c.GGML_OP_CONCAT => B.concat(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_UPSCALE => B.upscale(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_ARANGE, c.GGML_OP_FILL => B.fillLike(ctx, compute_ctx, node, node.op),
        c.GGML_OP_SCALE, c.GGML_OP_CLAMP => B.unary2(ctx, compute_ctx, src0.?, node, node.op),
        c.GGML_OP_SQR, c.GGML_OP_SQRT, c.GGML_OP_SIN, c.GGML_OP_COS, c.GGML_OP_LOG => B.unaryPlain(ctx, compute_ctx, src0.?, node, node.op),
        c.GGML_OP_TRI => B.tri(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_DIAG => B.unaryDstCount(ctx, compute_ctx, src0.?, node, c.GGML_OP_DIAG),
        c.GGML_OP_PAD => B.pad(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_PAD_REFLECT_1D => B.padReflect1d(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_ROLL => B.roll(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_CPY, c.GGML_OP_CONT, c.GGML_OP_DUP => B.cpy(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_SET_ROWS => B.setRows(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_SILU_BACK => B.siluBack(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_NORM, c.GGML_OP_L2_NORM => B.normEps(ctx, compute_ctx, src0.?, node, node.op),
        c.GGML_OP_GROUP_NORM => B.groupNorm(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_RMS_NORM => M.rmsNorm(ctx, compute_ctx, cgraph, node_idx),
        c.GGML_OP_RMS_NORM_BACK => B.rmsNormBack(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_UNARY => {
            if (ctx.fused_topk_moe_mode != .count) {
                R.topkMoe(ctx, compute_ctx, cgraph, node_idx);
                return true;
            }
            switch (c.ggml_get_unary_op(node)) {
                c.GGML_UNARY_OP_ELU,
                c.GGML_UNARY_OP_EXP,
                c.GGML_UNARY_OP_EXPM1,
                c.GGML_UNARY_OP_SILU,
                c.GGML_UNARY_OP_GELU,
                c.GGML_UNARY_OP_GELU_ERF,
                c.GGML_UNARY_OP_GELU_QUICK,
                c.GGML_UNARY_OP_RELU,
                c.GGML_UNARY_OP_NEG,
                c.GGML_UNARY_OP_TANH,
                c.GGML_UNARY_OP_SIGMOID,
                c.GGML_UNARY_OP_HARDSIGMOID,
                c.GGML_UNARY_OP_HARDSWISH,
                c.GGML_UNARY_OP_ABS,
                c.GGML_UNARY_OP_SOFTPLUS,
                c.GGML_UNARY_OP_STEP,
                c.GGML_UNARY_OP_ROUND,
                c.GGML_UNARY_OP_CEIL,
                c.GGML_UNARY_OP_FLOOR,
                c.GGML_UNARY_OP_TRUNC,
                c.GGML_UNARY_OP_SGN,
                => B.unaryPlain(ctx, compute_ctx, src0.?, node, c.GGML_OP_UNARY),
                c.GGML_UNARY_OP_XIELU => B.xielu(ctx, compute_ctx, src0.?, node),
                else => return false,
            }
        },
        c.GGML_OP_GLU => switch (c.ggml_get_glu_op(node)) {
            c.GGML_GLU_OP_GEGLU, c.GGML_GLU_OP_REGLU, c.GGML_GLU_OP_SWIGLU, c.GGML_GLU_OP_SWIGLU_OAI, c.GGML_GLU_OP_GEGLU_ERF, c.GGML_GLU_OP_GEGLU_QUICK, c.GGML_GLU_OP_SWIGLU_CLAMP => B.glu(ctx, compute_ctx, src0.?, src1, node),
            else => return false,
        },
        c.GGML_OP_DIAG_MASK_INF => B.diagMaskInf(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_SOFT_MAX => if (ctx.fused_topk_moe_mode != .count) R.topkMoe(ctx, compute_ctx, cgraph, node_idx) else R.softMax(ctx, compute_ctx, src0.?, src1, src2_, node),
        c.GGML_OP_SOFT_MAX_BACK => R.softMaxBack(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_ROPE => R.rope(ctx, compute_ctx, cgraph, node_idx, false),
        c.GGML_OP_ROPE_BACK => R.rope(ctx, compute_ctx, cgraph, node_idx, true),
        c.GGML_OP_ARGSORT => if (ctx.fused_topk_moe_mode != .count) R.topkMoe(ctx, compute_ctx, cgraph, node_idx) else R.argsort(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_TOP_K => R.topk(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_SUM, c.GGML_OP_SUM_ROWS, c.GGML_OP_MEAN => R.sumRows(ctx, compute_ctx, src0.?, node, node.op),
        c.GGML_OP_CUMSUM => R.cumsum(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_ARGMAX => R.argmax(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_CROSS_ENTROPY_LOSS => R.crossEntropyLoss(ctx, compute_ctx, node),
        c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => R.crossEntropyLossBack(ctx, compute_ctx, node),
        c.GGML_OP_COUNT_EQUAL => R.countEqual(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_SOLVE_TRI => R.solveTri(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_IM2COL => V.im2col(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_IM2COL_3D => V.im2col3d(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_TIMESTEP_EMBEDDING => V.timestepEmbedding(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_COL2IM_1D => V.col2im1d(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_CONV_TRANSPOSE_1D => V.convTranspose1d(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_POOL_1D => V.pool1d(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_POOL_2D => V.pool2d(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_CONV_2D, c.GGML_OP_CONV_TRANSPOSE_2D => V.conv2d(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_CONV_3D => V.conv3d(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_CONV_2D_DW => V.conv2dDw(ctx, compute_ctx, src0.?, src1.?, node),
        c.GGML_OP_LEAKY_RELU => V.leakyRelu(ctx, compute_ctx, src0.?, node),
        c.GGML_OP_MUL_MAT => m.ops_matmul.mulMat(ctx, compute_ctx, cgraph, node_idx),
        c.GGML_OP_MUL_MAT_ID => m.ops_matmul_id.mulMatId(ctx, compute_ctx, cgraph, node_idx),
        c.GGML_OP_FLASH_ATTN_EXT => m.ops_flash_attn.flashAttn(ctx, compute_ctx, src0.?, src1.?, src2_.?, src3_, node.src[4], node),
        c.GGML_OP_RWKV_WKV6 => M.rwkvWkv(ctx, compute_ctx, node, 6),
        c.GGML_OP_RWKV_WKV7 => M.rwkvWkv(ctx, compute_ctx, node, 7),
        c.GGML_OP_GATED_LINEAR_ATTN => M.gatedLinearAttn(ctx, compute_ctx, node),
        c.GGML_OP_LIGHTNING_INDEXER => M.lightningIndexer(ctx, compute_ctx, node),
        c.GGML_OP_GATED_DELTA_NET => M.gatedDeltaNet(ctx, compute_ctx, node),
        c.GGML_OP_SSM_SCAN => M.ssmScan(ctx, compute_ctx, node),
        c.GGML_OP_SSM_CONV => M.ssmConv(ctx, compute_ctx, cgraph, node_idx),
        c.GGML_OP_OPT_STEP_ADAMW => M.optStepAdamw(ctx, compute_ctx, node),
        c.GGML_OP_OPT_STEP_SGD => M.optStepSgd(ctx, compute_ctx, src0.?, src1.?, src2_.?, node),
        else => return false,
    }
    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "initFastdivValues divides every numerator exactly" {
    for ([_]u32{ 1, 2, 3, 7, 10, 64, 100, 1000, 65537, 0x7fffffff }) |d| {
        var mp: u32 = 0;
        var L: u32 = 0;
        initFastdivValues(d, &mp, &L);
        for ([_]u32{ 0, 1, d -% 1, d, d + 1, 12345, 0x7ffffffe }) |n| {
            const hi: u32 = @truncate((@as(u64, n) * mp) >> 32);
            try std.testing.expectEqual(n / d, @as(u32, @truncate((@as(u64, hi) + n) >> @intCast(L))));
        }
    }
}
