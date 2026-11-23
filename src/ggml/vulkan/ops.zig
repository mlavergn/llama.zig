//! The op layer's core: the push-constant helpers every op shares,
//! `ggml_vk_op_f32` that most ops dispatch through, and the switch from a
//! graph node to its op.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-push-constants.h` at v1.0.0
//! (`b1b770d90`). The C++ specialises two templates per push-constant struct --
//! `init_pushconst_fastdiv` and `init_pushconst_tensor_offsets` -- and the
//! port switches on the struct's type at comptime.

const std = @import("std");
const ggml = @import("../module.zig");

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

pub fn typeSize(t: *const ggml.impl.ggml_c.ggml_tensor) u32 {
    return @intCast(ggml.types.ggml_type_size(t.type));
}

// -----------------------------------------------------------------------------
// Fast division

/// Ports `init_fastdiv_values` (ggml-vulkan-push-constants.h:363 @b1b770d90): `mp` and
/// `L` with `n / d == (mulhi(n, mp) + n) >> L`.
pub fn initFastdivValues(d: u32, mp: *u32, L: *u32) void {
    // compute L = ceil(log2(d));
    L.* = 0;
    while (L.* < 32 and (@as(u64, 1) << @intCast(L.*)) < d) L.* += 1;
    // `uint64_t` arithmetic: the product wraps when L is 32.
    mp.* = @truncate(((@as(u64, 1) << 32) *% ((@as(u64, 1) << @intCast(L.*)) - d)) / d + 1);
}

/// Ports `pack_fastdiv_L` (ggml-vulkan-push-constants.h:374 @b1b770d90).
pub fn packFastdivL(L0: u32, L1: u32, L2: u32) u32 {
    return L0 | (L1 << 8) | (L2 << 16);
}

/// Ports `init_pushconst_fastdiv` and its specialisations for
/// `vk_op_unary_push_constants`, `vk_op_glu_push_constants`,
/// `vk_op_count_experts_push_constants`, `vk_op_conv2d_push_constants`,
/// `vk_op_conv3d_push_constants` and `vk_op_sum_rows_push_constants`
/// (ggml-vulkan-push-constants.h:378, 383, 403, 410, 759, 798, 860 @b1b770d90).
pub fn initPushconstFastdiv(p: anytype) void {
    const T = @TypeOf(p.*);
    if (T == ggml.vulkan.push.Unary) {
        var l: [6]u32 = undefined;
        initFastdivValues(p.ne02 *% p.ne01 *% p.ne00, &p.ne0_012mp, &l[0]);
        initFastdivValues(p.ne01 *% p.ne00, &p.ne0_01mp, &l[1]);
        initFastdivValues(p.ne00, &p.ne0_0mp, &l[2]);
        initFastdivValues(p.ne12 *% p.ne11 *% p.ne10, &p.ne1_012mp, &l[3]);
        initFastdivValues(p.ne11 *% p.ne10, &p.ne1_01mp, &l[4]);
        initFastdivValues(p.ne10, &p.ne1_0mp, &l[5]);
        p.ne0_Ls = packFastdivL(l[0], l[1], l[2]);
        p.ne1_Ls = packFastdivL(l[3], l[4], l[5]);
    } else if (T == ggml.vulkan.push.Glu) {
        // GLU linearizes over dst, then uses dst coordinates for src0/src1.
        initFastdivValues(p.ne22 *% p.ne21 *% p.ne20, &p.ne2_012mp, &p.ne2_012L);
        initFastdivValues(p.ne21 *% p.ne20, &p.ne2_01mp, &p.ne2_01L);
        initFastdivValues(p.ne20, &p.ne2_0mp, &p.ne2_0L);
    } else if (T == ggml.vulkan.push.CountExperts) {
        initFastdivValues(p.ne00, &p.ne00mp, &p.ne00L);
    } else if (T == ggml.vulkan.push.Conv2d) {
        initFastdivValues(p.OW, &p.OWmp, &p.OWL);
        initFastdivValues(p.OW *% p.OH, &p.OWOHmp, &p.OWOHL);
    } else if (T == ggml.vulkan.push.Conv3d) {
        initFastdivValues(p.OW, &p.OWmp, &p.OWL);
        initFastdivValues(p.OW *% p.OH, &p.OWOHmp, &p.OWOHL);
        initFastdivValues(p.OW *% p.OH *% p.OD, &p.OWOHODmp, &p.OWOHODL);
    } else if (T == ggml.vulkan.push.SumRows) {
        initFastdivValues(p.ne01 *% p.ne02, &p.ne0_12mp, &p.ne0_12L);
        initFastdivValues(p.ne01, &p.ne0_1mp, &p.ne0_1L);
    }
}

// -----------------------------------------------------------------------------
// Push-constant initialisers

/// Ports `vk_op_unary_push_constants_init` (ggml-vulkan-push-constants.h:279 @b1b770d90).
/// `ne` 0 means the destination's element count.
pub fn unaryPushConstantsInit(src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *const ggml.impl.ggml_c.ggml_tensor, ne_in: i64) ggml.vulkan.push.Unary {
    ggml.impl.assert(ne_in != 0 or ggml.types.ggml_nelements(src0) == ggml.types.ggml_nelements(dst), "ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst))");
    const ne = if (ne_in != 0) ne_in else ggml.types.ggml_nelements(dst);
    ggml.impl.assert(ne <= std.math.maxInt(u32), "ne <= (int64_t)std::numeric_limits<uint32_t>::max()");

    var p: ggml.vulkan.push.Unary = .{};
    p.ne = @intCast(ne);
    const ts0 = ggml.types.ggml_type_size(src0.type);
    p.ne00 = u32Of(src0.ne[0]);
    p.ne01 = u32Of(src0.ne[1]);
    p.ne02 = u32Of(src0.ne[2]);
    p.ne03 = u32Of(src0.ne[3]);
    p.nb00 = @truncate(src0.nb[0] / ts0);
    p.nb01 = @truncate(src0.nb[1] / ts0);
    p.nb02 = @truncate(src0.nb[2] / ts0);
    p.nb03 = @truncate(src0.nb[3] / ts0);
    const tsd = ggml.types.ggml_type_size(dst.type);
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

/// Ports `vk_op_pad_push_constants_init` (ggml-vulkan-push-constants.h:323 @b1b770d90).
pub fn padPushConstantsInit(src0: *const ggml.impl.ggml_c.ggml_tensor, dst: *const ggml.impl.ggml_c.ggml_tensor) ggml.vulkan.push.Pad {
    const ne = ggml.types.ggml_nelements(dst);
    ggml.impl.assert(ne <= std.math.maxInt(u32), "ne <= (int64_t)std::numeric_limits<uint32_t>::max()");
    var p: ggml.vulkan.push.Pad = .{};
    p.ne = @intCast(ne);
    const ts0 = ggml.types.ggml_type_size(src0.type);
    p.ne00 = u32Of(src0.ne[0]);
    p.ne01 = u32Of(src0.ne[1]);
    p.ne02 = u32Of(src0.ne[2]);
    p.ne03 = u32Of(src0.ne[3]);
    p.nb00 = @truncate(src0.nb[0] / ts0);
    p.nb01 = @truncate(src0.nb[1] / ts0);
    p.nb02 = @truncate(src0.nb[2] / ts0);
    p.nb03 = @truncate(src0.nb[3] / ts0);
    const tsd = ggml.types.ggml_type_size(dst.type);
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

/// Ports `vk_op_sum_rows_push_constants_init` (ggml-vulkan-push-constants.h:843 @b1b770d90).
pub fn sumRowsPushConstantsInit(src: *const ggml.impl.ggml_c.ggml_tensor, dst: *const ggml.impl.ggml_c.ggml_tensor, n_cols: i64) ggml.vulkan.push.SumRows {
    const type_size: u32 = @sizeOf(f32);
    var p: ggml.vulkan.push.SumRows = .{};
    p.n_cols = u32Of(n_cols);
    p.ne01 = u32Of(src.ne[1]);
    p.ne02 = u32Of(src.ne[2]);
    p.ne03 = u32Of(src.ne[3]);
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
pub fn binaryPc(ne: u32, src0: *const ggml.impl.ggml_c.ggml_tensor, src1: *const ggml.impl.ggml_c.ggml_tensor, dst: *const ggml.impl.ggml_c.ggml_tensor, param1: f32, param2: f32, param3: i32) ggml.vulkan.push.Binary {
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

fn misalignIn(ctx: *const ggml.vulkan.context.BackendCtx, t: *const ggml.impl.ggml_c.ggml_tensor, unit: u32) u32 {
    return ggml.vulkan.context.getMisalignBytes(ctx, t) / unit;
}

/// Ports `init_pushconst_tensor_offsets` (ggml-vulkan-push-constants.h:901 @b1b770d90), the
/// primary template, and its specialisations for the MatVecP021, MatVecNc,
/// Fwht, Dsv4HcComb, Dsv4HcPre, Dsv4HcPost, Unary, Glu, SumRows, Pad,
/// Conv2d, Conv3d, Im2col3d, Binary, Concat, Upscale and Rope structs
/// (`init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`, `init_pushconst_tensor_offsets`,
/// `init_pushconst_tensor_offsets`
/// (ggml-vulkan-push-constants.h:916, 928, 940, 949, 958, 967, 1004, 1015, 1030, 1041, 1052, 1061, 1070, 1081, 1096, 1108, 1120 @b1b770d90)):
/// the elements by which each binding's start was rounded down, packed
/// as the struct's shader expects; for a struct without them, the
/// assertion that nothing was misaligned.
pub fn initPushconstTensorOffsets(ctx: *const ggml.vulkan.context.BackendCtx, p: anytype, src0: ?*const ggml.impl.ggml_c.ggml_tensor, src1: ?*const ggml.impl.ggml_c.ggml_tensor, src2: ?*const ggml.impl.ggml_c.ggml_tensor, src3: ?*const ggml.impl.ggml_c.ggml_tensor, dst: ?*const ggml.impl.ggml_c.ggml_tensor) void {
    const T = @TypeOf(p.*);
    if (T == ggml.vulkan.push.MatVecP021 or T == ggml.vulkan.push.MatVecNc) {
        p.b_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        p.d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else if (T == ggml.vulkan.push.Fwht) {
        p.src_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        p.dst_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else if (T == ggml.vulkan.push.Dsv4HcComb) {
        p.m_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        p.s_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        p.b_offset = misalignIn(ctx, src2.?, typeSize(src2.?));
        p.d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else if (T == ggml.vulkan.push.Dsv4HcPre) {
        p.x_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        p.w_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        p.d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else if (T == ggml.vulkan.push.Dsv4HcPost) {
        p.x_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        p.r_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        p.p_offset = misalignIn(ctx, src2.?, typeSize(src2.?));
        p.c_offset = if (src3) |s3| misalignIn(ctx, s3, typeSize(s3)) else 0;
        p.d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else if (T == ggml.vulkan.push.Unary or T == ggml.vulkan.push.SumRows or T == ggml.vulkan.push.Pad) {
        const a_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        const d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
        p.misalign_offsets = (a_offset << 16) | d_offset;
    } else if (T == ggml.vulkan.push.Glu) {
        const a_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        const b_offset = if (src1) |s1| misalignIn(ctx, s1, typeSize(s1)) else a_offset;
        const d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
        ggml.impl.assert(a_offset < (1 << 8), "a_offset < (1u << 8)");
        ggml.impl.assert(b_offset < (1 << 8), "b_offset < (1u << 8)");
        ggml.impl.assert(d_offset < (1 << 8), "d_offset < (1u << 8)");
        p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;
    } else if (T == ggml.vulkan.push.Conv2d or T == ggml.vulkan.push.Conv3d) {
        p.knl_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        p.src_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        p.dst_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else if (T == ggml.vulkan.push.Im2col3d) {
        const a_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        const d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
        p.misalign_offsets = (a_offset << 16) | d_offset;
    } else if (T == ggml.vulkan.push.Binary) {
        const a_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        const b_offset = misalignIn(ctx, src1.?, typeSize(src1.?));
        const d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
        ggml.impl.assert(a_offset <= 0xFFFF, "a_offset <= 0xFFFF");
        ggml.impl.assert(b_offset <= 0xFF, "b_offset <= 0xFF");
        ggml.impl.assert(d_offset <= 0xFF, "d_offset <= 0xFF");
        p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;
    } else if (T == ggml.vulkan.push.Concat) {
        const unit_size = ggml.vulkan.op_pipeline.concatUnitSize(dst.?.type);
        p.misalign_offsets = (misalignIn(ctx, src0.?, unit_size) << 16) | (misalignIn(ctx, src1.?, unit_size) << 8) | misalignIn(ctx, dst.?, unit_size);
    } else if (T == ggml.vulkan.push.Upscale or T == ggml.vulkan.push.Rope) {
        p.a_offset = misalignIn(ctx, src0.?, typeSize(src0.?));
        p.d_offset = misalignIn(ctx, dst.?, typeSize(dst.?));
    } else {
        ggml.impl.assert(src0 == null or ggml.vulkan.context.getMisalignBytes(ctx, src0.?) == 0, "!src0 || get_misalign_bytes(ctx, src0) == 0");
        ggml.impl.assert(src1 == null or ggml.vulkan.context.getMisalignBytes(ctx, src1.?) == 0, "!src1 || get_misalign_bytes(ctx, src1) == 0");
        ggml.impl.assert(src2 == null or ggml.vulkan.context.getMisalignBytes(ctx, src2.?) == 0, "!src2 || get_misalign_bytes(ctx, src2) == 0");
        ggml.impl.assert(src3 == null or ggml.vulkan.context.getMisalignBytes(ctx, src3.?) == 0, "!src3 || get_misalign_bytes(ctx, src3) == 0");
        ggml.impl.assert(dst == null or ggml.vulkan.context.getMisalignBytes(ctx, dst.?) == 0, "!dst  || get_misalign_bytes(ctx, dst) == 0");
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

/// Ports `ggml_vk_op_f32` (ggml-vulkan.cpp:9873 @b1b770d90): picks the
/// op's pipeline, finishes the push constants, sizes the dispatch for the
/// op, and binds the sources and destination as the op's shader expects.
pub fn opF32(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: ?*const ggml.impl.ggml_c.ggml_tensor, src1: ?*const ggml.impl.ggml_c.ggml_tensor, src2: ?*const ggml.impl.ggml_c.ggml_tensor, src3: ?*const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op, pc_in: anytype) void {
    opF32Override(ctx, subctx, src0, src1, src2, src3, dst, op, pc_in, null);
}

/// `ggml_vk_op_f32` with its `pipeline_override` argument: a fused op
/// dispatched through another op's sizing and bindings.
pub fn opF32Override(ctx: *ggml.vulkan.context.BackendCtx, subctx: *ggml.vulkan.submit.Context, src0: ?*const ggml.impl.ggml_c.ggml_tensor, src1: ?*const ggml.impl.ggml_c.ggml_tensor, src2: ?*const ggml.impl.ggml_c.ggml_tensor, src3: ?*const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, op: ggml.impl.ggml_c.enum_ggml_op, pc_in: anytype, pipeline_override: ?*ggml.vulkan.pipeline.Pipeline) void {
    var pc = pc_in;
    const PC = @TypeOf(pc);
    const s0 = src0.?;
    ggml.impl.assert(op == ggml.impl.ggml_c.GGML_OP_GET_ROWS or op == ggml.impl.ggml_c.GGML_OP_CPY or op == ggml.impl.ggml_c.GGML_OP_CONCAT or (!ggml.types.ggml_is_quantized(s0.type) and (src1 == null or !ggml.types.ggml_is_quantized(src1.?.type))), "op == GGML_OP_GET_ROWS || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type)))");
    ggml.impl.assert(dst.buffer != null, "dst->buffer != nullptr");
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

    const pipeline = pipeline_override orelse ggml.vulkan.op_pipeline.getPipeline(ctx, s0, src1, src2, dst, op) orelse {
        ggml.impl.fprintfStderr("ggml_vulkan: Error: Missing op: %s for %s", .{ ggml.types.ggml_op_name(op), ggml.types.ggml_type_name(s0.type) });
        if (src1) |s1| ggml.impl.fprintfStderr(" and %s", .{ggml.types.ggml_type_name(s1.type)});
        ggml.impl.fprintfStderr(" to %s\n", .{ggml.types.ggml_type_name(dst.type)});
        ggml.impl.abort("fatal error");
    };

    ggml.vulkan.context.requestDescriptorSets(ctx, pipeline, 1);

    const src0_buf = ggml.vulkan.context.tensorSubbuffer(ctx, s0, true);
    const src1_buf: ggml.vulkan.buffer.Subbuffer = if (use_src1) ggml.vulkan.context.tensorSubbuffer(ctx, src1.?, true) else .{};
    const src2_buf: ggml.vulkan.buffer.Subbuffer = if (use_src2) ggml.vulkan.context.tensorSubbuffer(ctx, src2.?, true) else .{};
    const src3_buf: ggml.vulkan.buffer.Subbuffer = if (use_src3) ggml.vulkan.context.tensorSubbuffer(ctx, src3.?, true) else .{};
    var dst_buf = ggml.vulkan.context.tensorSubbuffer(ctx, dst, true);

    // Compute misalignment offset for descriptors and store it in in push constants.
    initPushconstTensorOffsets(ctx, &pc, s0, src1, src2, src3, dst);

    const limits = ctx.device.properties.limits;
    var elements: [3]u32 = undefined;
    switch (op) {
        ggml.impl.ggml_c.GGML_OP_NORM, ggml.impl.ggml_c.GGML_OP_RMS_NORM_BACK, ggml.impl.ggml_c.GGML_OP_L2_NORM, ggml.impl.ggml_c.GGML_OP_SOFT_MAX, ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK, ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_CUMSUM, ggml.impl.ggml_c.GGML_OP_MEAN, ggml.impl.ggml_c.GGML_OP_ARGMAX => {
            elements = split512(u32Of(ggml.types.ggml_nrows(s0)));
        },
        ggml.impl.ggml_c.GGML_OP_SOLVE_TRI => elements = split512(@truncate(ne02 *% ne03)),
        ggml.impl.ggml_c.GGML_OP_RMS_NORM => {
            if (ctx.do_add_rms_partials) {
                // Run one element per thread, 128 threads per workgroup
                elements = .{ @truncate(ceilDiv(ne00, 128)), 1, 1 };
            } else {
                elements = .{ @truncate(ne01), @truncate(ne02), @truncate(ne03) };
                elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
                elements[2] = @min(elements[2], limits.maxComputeWorkGroupCount[2]);
            }
        },
        // We use GGML_OP_SUM_ROWS with 1 row.
        ggml.impl.ggml_c.GGML_OP_SUM => elements = .{ 1, 1, 1 },
        ggml.impl.ggml_c.GGML_OP_GROUP_NORM => {
            const total_workgroups: u32 = @as(u32, @bitCast(dst.op_params[0])) *% u32Of(s0.ne[3]);
            const groups_x = @min(total_workgroups, limits.maxComputeWorkGroupCount[0]);
            const groups_y_needed = ceilDiv(@as(u64, total_workgroups), groups_x);
            const groups_y: u32 = @intCast(@min(groups_y_needed, limits.maxComputeWorkGroupCount[1]));
            const groups_xy = @as(u64, groups_x) * groups_y;
            elements = .{ groups_x, groups_y, @truncate(ceilDiv(@as(u64, total_workgroups), groups_xy)) };
            ggml.impl.assert(elements[2] <= limits.maxComputeWorkGroupCount[2], "elements[2] <= ctx->device->properties.limits.maxComputeWorkGroupCount[2]");
        },
        ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_ROPE_BACK => {
            var nrows = u32Of(ggml.types.ggml_nrows(s0));
            var z: u32 = 1;
            if (nrows > limits.maxComputeWorkGroupCount[0]) {
                z = ceilDiv(nrows, 32768);
                nrows = 32768;
            }
            elements = .{ nrows, @truncate(ne00), z };
        },
        ggml.impl.ggml_c.GGML_OP_GET_ROWS => {
            elements = .{ @truncate(ne00), @truncate(ne10), @truncate(ne11 *% ne12) };
            elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
            elements[2] = @min(elements[2], limits.maxComputeWorkGroupCount[2]);
        },
        ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK => {
            elements = .{ u32Of(dst.ne[0]), u32Of(dst.ne[1]), 1 };
            elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
        },
        ggml.impl.ggml_c.GGML_OP_ARGSORT => ggml.impl.abort("GGML_ASSERT(0) failed"),
        ggml.impl.ggml_c.GGML_OP_IM2COL => {
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
            const min_cap: u32 = if (ctx.device.vendor_id == ggml.vulkan.common.vendor_id.amd) 512 else 4096;
            const x_elements = @min(CHW, @max(min_cap, OW *% KH *% KW));
            elements = .{ x_elements, OW, OH *% batch };
            elements[1] = @min(elements[1], limits.maxComputeWorkGroupCount[1]);
            elements[2] = @min(elements[2], limits.maxComputeWorkGroupCount[2]);
        },
        ggml.impl.ggml_c.GGML_OP_IM2COL_3D => {
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
        ggml.impl.ggml_c.GGML_OP_TIMESTEP_EMBEDDING => {
            const dim: u32 = @bitCast(dst.op_params[0]);
            elements = .{ (dim +% 1) / 2, u32Of(s0.ne[0]), 1 };
        },
        // parallelize in {Cout, 1, 1}
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D => elements = .{ u32Of(s0.ne[1]), 1, 1 },
        ggml.impl.ggml_c.GGML_OP_COL2IM_1D => elements = .{ u32Of(dst.ne[0]), u32Of(dst.ne[1]), 1 },
        ggml.impl.ggml_c.GGML_OP_POOL_1D => {
            const N = u32Of(dst.ne[3] *% dst.ne[2]);
            elements = .{ N *% u32Of(dst.ne[1]) *% u32Of(dst.ne[0]), 1, 1 };
        },
        ggml.impl.ggml_c.GGML_OP_POOL_2D => {
            const outputs: u32 = @truncate(@as(u64, @bitCast(ggml.types.ggml_nelements(dst))));
            const outputs_per_workgroup: u32 = if (pipeline == ctx.device.pipeline_pool2d_subgroup_f32) 128 / ctx.device.subgroup_size else 512;
            const workgroups = ceilDiv(outputs, outputs_per_workgroup);
            const groups_y = ceilDiv(workgroups, limits.maxComputeWorkGroupCount[0]);
            const groups_x = ceilDiv(workgroups, groups_y);
            elements = .{ groups_x, groups_y, 1 };
        },
        ggml.impl.ggml_c.GGML_OP_CONV_2D, ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D => {
            if (PC != ggml.vulkan.push.Conv2d) ggml.impl.abort("invalid push constant type for CONV_2D");
            const NPQ = pc.N *% pc.OH *% pc.OW;
            const shape = ggml.vulkan.op_pipeline.convSelectShape(ctx, pc.Cout, NPQ);
            const npq_blocks = ceilDiv(NPQ, ggml.vulkan.backend.conv_block_sizes[@backingInt(shape)].NPQ);
            elements = .{ pc.Cout, npq_blocks, 1 };
            if (elements[1] > 512) {
                elements[2] = ceilDiv(elements[1], 512);
                elements[1] = 512;
            }
        },
        ggml.impl.ggml_c.GGML_OP_CONV_3D => {
            if (PC != ggml.vulkan.push.Conv3d) ggml.impl.abort("invalid push constant type for CONV_3D");
            const NPQ = pc.N *% pc.OD *% pc.OH *% pc.OW;
            const shape = ggml.vulkan.op_pipeline.conv3dSelectShape(ctx, pc.OC, NPQ);
            const npq_blocks = ceilDiv(NPQ, ggml.vulkan.backend.conv_block_sizes[@backingInt(shape)].NPQ);
            elements = .{ pc.OC, npq_blocks, 1 };
            if (elements[1] > 512) {
                elements[2] = ceilDiv(elements[1], 512);
                elements[1] = 512;
            }
        },
        ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_SUB, ggml.impl.ggml_c.GGML_OP_DIV, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_OUT_PROD, ggml.impl.ggml_c.GGML_OP_ARANGE, ggml.impl.ggml_c.GGML_OP_FILL, ggml.impl.ggml_c.GGML_OP_SCALE, ggml.impl.ggml_c.GGML_OP_SQR, ggml.impl.ggml_c.GGML_OP_SQRT, ggml.impl.ggml_c.GGML_OP_SIN, ggml.impl.ggml_c.GGML_OP_COS, ggml.impl.ggml_c.GGML_OP_LOG, ggml.impl.ggml_c.GGML_OP_TRI, ggml.impl.ggml_c.GGML_OP_DIAG, ggml.impl.ggml_c.GGML_OP_CLAMP, ggml.impl.ggml_c.GGML_OP_LEAKY_RELU, ggml.impl.ggml_c.GGML_OP_PAD, ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D, ggml.impl.ggml_c.GGML_OP_ROLL, ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_REPEAT_BACK, ggml.impl.ggml_c.GGML_OP_CPY, ggml.impl.ggml_c.GGML_OP_CONCAT, ggml.impl.ggml_c.GGML_OP_UPSCALE, ggml.impl.ggml_c.GGML_OP_UNARY, ggml.impl.ggml_c.GGML_OP_GLU, ggml.impl.ggml_c.GGML_OP_CONV_2D_DW => blk: {
            // dispatched per 32x32 tile, not per element
            if (op == ggml.impl.ggml_c.GGML_OP_CONCAT and
                (pipeline == ctx.device.pipeline_concat_transpose_i8 or
                    pipeline == ctx.device.pipeline_concat_transpose_i16 or
                    pipeline == ctx.device.pipeline_concat_transpose_i32))
            {
                const tiles_y: u32 = @truncate(ceilDiv(@as(u64, @bitCast(dst.ne[0])), 32));
                const groups_y = @min(tiles_y, limits.maxComputeWorkGroupCount[1]);
                elements = .{ u32Of(src1.?.ne[1]), groups_y * 32, ceilDiv(tiles_y, groups_y) };
                break :blk;
            }
            var ne = u32Of(ggml.types.ggml_nelements(dst));
            if (op == ggml.impl.ggml_c.GGML_OP_CPY and ggml.types.ggml_is_quantized(s0.type) and ggml.types.ggml_is_quantized(dst.type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= u32Of(ggml.types.ggml_blck_size(s0.type));
                const ts = typeSize(s0);
                if (ts % 4 == 0) ne *%= ts / 4 else ne *%= ts / 2;
            }
            if (op == ggml.impl.ggml_c.GGML_OP_CONCAT and ggml.types.ggml_is_quantized(dst.type)) {
                ne = ne / u32Of(ggml.types.ggml_blck_size(dst.type)) *% typeSize(dst) / ggml.vulkan.op_pipeline.concatUnitSize(dst.type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            if (op == ggml.impl.ggml_c.GGML_OP_CPY and !ggml.types.ggml_is_quantized(s0.type) and ggml.types.ggml_is_quantized(dst.type)) {
                ne = ceilDiv(ne, u32Of(ggml.types.ggml_blck_size(dst.type)));
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
        ggml.impl.ggml_c.GGML_OP_ADD_ID => elements = .{ @truncate(ne01), @truncate(ne02), 1 },
        ggml.impl.ggml_c.GGML_OP_SET_ROWS => {
            var ne = u32Of(ggml.types.ggml_nelements(s0));
            if (ggml.types.ggml_is_quantized(dst.type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = ceilDiv(ne, 32 *% u32Of(ggml.types.ggml_blck_size(dst.type)));
            } else {
                // scalar types do one element per thread, running 512 threads
                ne = ceilDiv(ne, 512);
            }
            elements = split512(ne);
        },
        ggml.impl.ggml_c.GGML_OP_SSM_CONV => elements = .{ u32Of(s0.ne[1]), u32Of(dst.ne[1]), u32Of(dst.ne[2]) },
        else => elements = .{ u32Of(ggml.types.ggml_nelements(s0)), 1, 1 },
    }

    const pcb = std.mem.asBytes(&pc);
    if (op == ggml.impl.ggml_c.GGML_OP_ADD or op == ggml.impl.ggml_c.GGML_OP_RMS_NORM) {
        var a_buf = src0_buf;
        if (ctx.do_add_rms_partials) a_buf = ggml.vulkan.context.subbuffer(ctx, ctx.prealloc_add_rms_partials.?, ctx.prealloc_size_add_rms_partials_offset);
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, dst_buf, a_buf }, pcb, elements);
    } else if (op == ggml.impl.ggml_c.GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, if (use_src1) src1_buf else src0_buf, dst_buf }, pcb, elements);
    } else if (op == ggml.impl.ggml_c.GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, if (use_src1) src1_buf else src0_buf, if (use_src2) src2_buf else src0_buf, dst_buf }, pcb, elements);
    } else if (op == ggml.impl.ggml_c.GGML_OP_ROPE or op == ggml.impl.ggml_c.GGML_OP_ROPE_BACK) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, if (use_src2) src2_buf else src0_buf, dst_buf, if (use_src3) src3_buf else src0_buf }, pcb, elements);
    } else if (op == ggml.impl.ggml_c.GGML_OP_IM2COL or op == ggml.impl.ggml_c.GGML_OP_IM2COL_3D) {
        // buffer device address path doesn't use dst buffer
        if (ctx.device.shader_int64 and ctx.device.buffer_device_address) dst_buf.size = 1;
        // im2col uses only src1 and dst buffers
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src1_buf, dst_buf }, pcb, elements);
    } else if (op == ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml.vulkan.submit.bufferMemsetAsync(subctx, dst_buf.buffer.?, dst_buf.offset, 0, dst_buf.size);
        ggml.vulkan.context.syncBuffers(ctx, subctx);
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, dst_buf }, pcb, elements);
    } else if (op == ggml.impl.ggml_c.GGML_OP_OPT_STEP_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, src2_buf }, pcb, elements);
    } else if (use_src3) {
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pcb, elements);
    } else if (use_src2) {
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, src2_buf, dst_buf }, pcb, elements);
    } else if (use_src1) {
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, src1_buf, dst_buf }, pcb, elements);
    } else {
        ggml.vulkan.context.dispatchPipeline(ctx, subctx, pipeline, &.{ src0_buf, dst_buf }, pcb, elements);
    }
}

// -----------------------------------------------------------------------------
// Shared helpers

/// Ports the tensor overload of `ggml_vk_use_mul_mat_vec_id`
/// (ggml-vulkan.cpp:8290 @b1b770d90).
pub fn useMulMatVecIdTensor(dst: *const ggml.impl.ggml_c.ggml_tensor) bool {
    const src0 = dst.src[0].*;
    const ids = dst.src[2].*;
    return ids.ne[1] <= 8 and (src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or ggml.types.ggml_is_quantized(src0.type));
}

/// Ports `ggml_vk_use_mul_mat_vec_id` (ggml-vulkan.cpp:8296 @b1b770d90).
pub fn useMulMatVecId(cgraph: *const ggml.impl.CGraph, node_idx: c_int) bool {
    return useMulMatVecIdTensor(cgraph.nodes[@intCast(node_idx)].?);
}

/// Ports `ggml_vk_rms_num_partials` (ggml-vulkan.cpp:11204 @b1b770d90).
pub fn rmsNumPartials(ctx: *const ggml.vulkan.context.BackendCtx, node: *const ggml.impl.ggml_c.ggml_tensor) u32 {
    const ne = u32Of(node.ne[0]);
    const denom = ctx.device.pipeline_add_rms[0][0][0].?.wg_denoms[0];
    return ceilDiv(ne, denom);
}

/// Ports `ggml_vk_rms_partials_size` (ggml-vulkan.cpp:11211 @b1b770d90).
pub fn rmsPartialsSize(ctx: *const ggml.vulkan.context.BackendCtx, node: *const ggml.impl.ggml_c.ggml_tensor) u32 {
    const num_partials = rmsNumPartials(ctx, node);
    const a = ctx.device.partials_binding_alignment;
    // ROUNDUP_POW2
    return (num_partials *% @sizeOf(u32) + a - 1) & ~(a - 1);
}

// -----------------------------------------------------------------------------
// The node switch

/// The `switch (node->op)` of `ggml_vk_build_graph` (ggml-vulkan.cpp:12698 @b1b770d90).
///
/// Return: false where the C++ returns false, for an op it has no
/// dispatch for.
pub fn dispatchNode(ctx: *ggml.vulkan.context.BackendCtx, compute_ctx: *ggml.vulkan.submit.Context, cgraph: *ggml.impl.CGraph, node_idx: c_int, node: *ggml.impl.ggml_c.ggml_tensor, src0_: ?*ggml.impl.ggml_c.ggml_tensor, src1_: ?*ggml.impl.ggml_c.ggml_tensor, src2_: ?*ggml.impl.ggml_c.ggml_tensor, src3_: ?*ggml.impl.ggml_c.ggml_tensor) bool {
    const B = ggml.vulkan.ops_basic;
    const R = ggml.vulkan.ops_reduce;
    const M = ggml.vulkan.ops_more;
    const V = ggml.vulkan.ops_conv;
    const src0 = src0_;
    const src1 = src1_;
    switch (node.op) {
        ggml.impl.ggml_c.GGML_OP_REPEAT, ggml.impl.ggml_c.GGML_OP_REPEAT_BACK => B.unaryDstCount(ctx, compute_ctx, src0.?, node, node.op),
        ggml.impl.ggml_c.GGML_OP_ACC, ggml.impl.ggml_c.GGML_OP_SET => B.acc(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_GET_ROWS => if (ctx.fused_topk_qsa) R.topkQsa(ctx, compute_ctx, cgraph, node_idx) else B.getRows(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK => B.getRowsBack(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_ADD => if (ctx.num_additional_fused_ops != 0) B.multiAdd(ctx, compute_ctx, cgraph, node_idx) else B.add(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_OUT_PROD => B.outProd(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_SUB, ggml.impl.ggml_c.GGML_OP_DIV => B.binary(ctx, compute_ctx, src0.?, src1.?, node, node.op),
        ggml.impl.ggml_c.GGML_OP_MUL => if (ctx.num_additional_fused_ops != 0) B.snakeFused(ctx, compute_ctx, cgraph, node_idx) else B.binary(ctx, compute_ctx, src0.?, src1.?, node, ggml.impl.ggml_c.GGML_OP_MUL),
        ggml.impl.ggml_c.GGML_OP_ADD_ID => B.addId(ctx, compute_ctx, src0.?, src1.?, src2_.?, node),
        ggml.impl.ggml_c.GGML_OP_CONCAT => B.concat(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_UPSCALE => B.upscale(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_ARANGE, ggml.impl.ggml_c.GGML_OP_FILL => B.fillLike(ctx, compute_ctx, node, node.op),
        ggml.impl.ggml_c.GGML_OP_SCALE => if (ctx.fused_hc_post_gate) {
            const hc_post = cgraph.nodes[@intCast(node_idx + ctx.num_additional_fused_ops)].?;
            ggml.vulkan.ops_matmul.dsv4HcPost(ctx, compute_ctx, hc_post.src[0], hc_post.src[1], hc_post.src[2], hc_post.src[3], hc_post, node);
        } else B.unary2(ctx, compute_ctx, src0.?, node, ggml.impl.ggml_c.GGML_OP_SCALE),
        ggml.impl.ggml_c.GGML_OP_CLAMP => B.unary2(ctx, compute_ctx, src0.?, node, node.op),
        ggml.impl.ggml_c.GGML_OP_SQR, ggml.impl.ggml_c.GGML_OP_SQRT, ggml.impl.ggml_c.GGML_OP_SIN, ggml.impl.ggml_c.GGML_OP_COS, ggml.impl.ggml_c.GGML_OP_LOG => B.unaryPlain(ctx, compute_ctx, src0.?, node, node.op),
        ggml.impl.ggml_c.GGML_OP_TRI => B.tri(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_DIAG => B.unaryDstCount(ctx, compute_ctx, src0.?, node, ggml.impl.ggml_c.GGML_OP_DIAG),
        ggml.impl.ggml_c.GGML_OP_PAD => B.pad(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D => B.padReflect1d(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_ROLL => B.roll(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_CPY, ggml.impl.ggml_c.GGML_OP_CONT, ggml.impl.ggml_c.GGML_OP_DUP => B.cpy(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_SET_ROWS => B.setRows(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_SILU_BACK => B.siluBack(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_NORM, ggml.impl.ggml_c.GGML_OP_L2_NORM => B.normEps(ctx, compute_ctx, src0.?, node, node.op),
        ggml.impl.ggml_c.GGML_OP_GROUP_NORM => B.groupNorm(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_RMS_NORM => M.rmsNorm(ctx, compute_ctx, cgraph, node_idx),
        ggml.impl.ggml_c.GGML_OP_RMS_NORM_BACK => B.rmsNormBack(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_UNARY => {
            if (ctx.fused_topk_moe_mode != .count) {
                R.topkMoe(ctx, compute_ctx, cgraph, node_idx);
                return true;
            }
            if (ctx.num_additional_fused_ops != 0) {
                B.unaryMul(ctx, compute_ctx, cgraph, node_idx);
                return true;
            }
            switch (ggml.types.ggml_get_unary_op(node)) {
                ggml.impl.ggml_c.GGML_UNARY_OP_ELU,
                ggml.impl.ggml_c.GGML_UNARY_OP_EXP,
                ggml.impl.ggml_c.GGML_UNARY_OP_EXPM1,
                ggml.impl.ggml_c.GGML_UNARY_OP_SILU,
                ggml.impl.ggml_c.GGML_UNARY_OP_GELU,
                ggml.impl.ggml_c.GGML_UNARY_OP_GELU_ERF,
                ggml.impl.ggml_c.GGML_UNARY_OP_GELU_QUICK,
                ggml.impl.ggml_c.GGML_UNARY_OP_RELU,
                ggml.impl.ggml_c.GGML_UNARY_OP_NEG,
                ggml.impl.ggml_c.GGML_UNARY_OP_TANH,
                ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID,
                ggml.impl.ggml_c.GGML_UNARY_OP_HARDSIGMOID,
                ggml.impl.ggml_c.GGML_UNARY_OP_HARDSWISH,
                ggml.impl.ggml_c.GGML_UNARY_OP_ABS,
                ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS,
                ggml.impl.ggml_c.GGML_UNARY_OP_STEP,
                ggml.impl.ggml_c.GGML_UNARY_OP_ROUND,
                ggml.impl.ggml_c.GGML_UNARY_OP_CEIL,
                ggml.impl.ggml_c.GGML_UNARY_OP_FLOOR,
                ggml.impl.ggml_c.GGML_UNARY_OP_TRUNC,
                ggml.impl.ggml_c.GGML_UNARY_OP_SGN,
                => B.unaryPlain(ctx, compute_ctx, src0.?, node, ggml.impl.ggml_c.GGML_OP_UNARY),
                ggml.impl.ggml_c.GGML_UNARY_OP_XIELU => B.xielu(ctx, compute_ctx, src0.?, node),
                else => return false,
            }
        },
        ggml.impl.ggml_c.GGML_OP_GLU => switch (ggml.types.ggml_get_glu_op(node)) {
            ggml.impl.ggml_c.GGML_GLU_OP_GEGLU, ggml.impl.ggml_c.GGML_GLU_OP_REGLU, ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU, ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_OAI, ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_ERF, ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_QUICK, ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_CLAMP => B.glu(ctx, compute_ctx, src0.?, src1, node),
            else => return false,
        },
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX => if (ctx.fused_topk_moe_mode != .count) R.topkMoe(ctx, compute_ctx, cgraph, node_idx) else R.softMax(ctx, compute_ctx, src0.?, src1, src2_, node),
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK => R.softMaxBack(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_ROPE => R.rope(ctx, compute_ctx, cgraph, node_idx, false),
        ggml.impl.ggml_c.GGML_OP_ROPE_BACK => R.rope(ctx, compute_ctx, cgraph, node_idx, true),
        ggml.impl.ggml_c.GGML_OP_ARGSORT => if (ctx.fused_topk_moe_mode != .count) R.topkMoe(ctx, compute_ctx, cgraph, node_idx) else R.argsort(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_TOP_K => R.topk(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_SUM, ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_MEAN => R.sumRows(ctx, compute_ctx, src0.?, node, node.op),
        ggml.impl.ggml_c.GGML_OP_CUMSUM => R.cumsum(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_COMB => ggml.vulkan.ops_matmul.dsv4HcComb(ctx, compute_ctx, src0.?, src1.?, src2_.?, node),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_PRE => ggml.vulkan.ops_matmul.dsv4HcPre(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST => ggml.vulkan.ops_matmul.dsv4HcPost(ctx, compute_ctx, src0.?, src1.?, src2_.?, src3_, node, null),
        ggml.impl.ggml_c.GGML_OP_ARGMAX => R.argmax(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS => R.crossEntropyLoss(ctx, compute_ctx, node),
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => R.crossEntropyLossBack(ctx, compute_ctx, node),
        ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL => R.countEqual(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_SOLVE_TRI => R.solveTri(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_IM2COL => V.im2col(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_IM2COL_3D => V.im2col3d(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_TIMESTEP_EMBEDDING => V.timestepEmbedding(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_COL2IM_1D => V.col2im1d(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D => V.convTranspose1d(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_POOL_1D => V.pool1d(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_POOL_2D => V.pool2d(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_CONV_2D, ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D => V.conv2d(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_CONV_3D => V.conv3d(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_CONV_2D_DW => V.conv2dDw(ctx, compute_ctx, src0.?, src1.?, node),
        ggml.impl.ggml_c.GGML_OP_LEAKY_RELU => V.leakyRelu(ctx, compute_ctx, src0.?, node),
        ggml.impl.ggml_c.GGML_OP_MUL_MAT => ggml.vulkan.ops_matmul.mulMat(ctx, compute_ctx, cgraph, node_idx),
        ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID => ggml.vulkan.ops_matmul_id.mulMatId(ctx, compute_ctx, cgraph, node_idx),
        ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT => ggml.vulkan.ops_flash_attn.flashAttn(ctx, compute_ctx, src0.?, src1.?, src2_.?, src3_, node.src[4], node),
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV6 => M.rwkvWkv(ctx, compute_ctx, node, 6),
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV7 => M.rwkvWkv(ctx, compute_ctx, node, 7),
        ggml.impl.ggml_c.GGML_OP_GATED_LINEAR_ATTN => M.gatedLinearAttn(ctx, compute_ctx, node),
        ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER => M.lightningIndexer(ctx, compute_ctx, node),
        ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET => M.gatedDeltaNet(ctx, compute_ctx, node),
        ggml.impl.ggml_c.GGML_OP_SSM_SCAN => M.ssmScan(ctx, compute_ctx, node),
        ggml.impl.ggml_c.GGML_OP_SSM_CONV => M.ssmConv(ctx, compute_ctx, cgraph, node_idx),
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_ADAMW => M.optStepAdamw(ctx, compute_ctx, node),
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_SGD => M.optStepSgd(ctx, compute_ctx, src0.?, src1.?, src2_.?, node),
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
