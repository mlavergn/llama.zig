//! Broadcasting binary ops: `ADD`, `SUB`, `MUL`, `DIV`, `REPEAT`,
//! `REPEAT_BACK`, and the fused add/mul chains `graph.zig`'s fusion builds.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/binbcast.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/binbcast.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels -- `k_bin_bcast`, `k_bin_bcast_unravel`, `k_repeat_back` -- stay
//! CUDA.
//!
//! # The parameter pack
//!
//! `launch_bin_bcast_pack` is instantiated with `std::make_index_sequence
//! <n_fuse>`, and the kernel takes one more `const src1_t *` per index:
//! `n_fuse` is 1 for a plain op, so even `ADD` passes `src[1]` twice -- once
//! as `src1` and once in the pack, which is the one the kernel reads.
//! `REPEAT` uses `n_fuse == 0` and reads `src1`. The key's trailing `, >`
//! for an empty pack is how `cu++filt` writes it.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// The four `__device__` ops a launch is instantiated with, plus repeat.
pub const Op = enum {
    repeat,
    add,
    sub,
    mul,
    div,

    fn name(self: Op) []const u8 {
        return switch (self) {
            .repeat => "op_repeat",
            .add => "op_add",
            .sub => "op_sub",
            .mul => "op_mul",
            .div => "op_div",
        };
    }
};

fn binKernel(comptime unravel: bool, comptime op: Op, comptime S0: type, comptime S1: type, comptime D: type, comptime n_fuse: usize) kernels.Kernel {
    const key = comptime blk: {
        var k: []const u8 = (if (unravel) "k_bin_bcast_unravel" else "k_bin_bcast") ++ "<&" ++ op.name() ++ ", " ++
            kernels.typeName(S0) ++ ", " ++ kernels.typeName(S1) ++ ", " ++ kernels.typeName(D) ++ ", ";
        if (n_fuse == 0) {
            k = k ++ ">";
        } else {
            for (0..n_fuse) |i| k = k ++ (if (i == 0) "" else ", ") ++ kernels.constPtr(S1);
            k = k ++ ">";
        }
        break :blk k;
    };
    return kernels.get("binbcast", key);
}

const u32_max = std.math.maxInt(u32);

/// Ports `launch_bin_bcast_pack` (binbcast.cu:168 @c1d0e7a00).
fn launchBinBcastPack(
    comptime op: Op,
    comptime S0: type,
    comptime S1: type,
    comptime D: type,
    comptime n_fuse: usize,
    src0: *const Tensor,
    src1: *const Tensor,
    dst: *const Tensor,
    src0_dd: ?*const anyopaque,
    src1_dd: ?*const anyopaque,
    dst_dd: ?*anyopaque,
    stream: cu.cudaStream_t,
) void {
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];

    const nr = [4]i64{
        @as(c_int, @truncate(@divTrunc(ne10, dst.ne[0]))),
        @as(c_int, @truncate(@divTrunc(ne11, dst.ne[1]))),
        @as(c_int, @truncate(@divTrunc(ne12, dst.ne[2]))),
        @as(c_int, @truncate(@divTrunc(ne13, dst.ne[3]))),
    };

    var cne: [4]i64 = dst.ne;
    var cne0: [4]i64 = src0.ne;
    var cne1: [4]i64 = src1.ne;

    var cnb: [4]usize = dst.nb;
    var cnb0: [4]usize = src0.nb;
    var cnb1: [4]usize = src1.nb;

    const collapse = struct {
        fn f(x: *[4]i64) void {
            x[0] *= x[1];
            x[1] = x[2];
            x[2] = x[3];
            x[3] = 1;
        }
    }.f;
    const collapseNb = struct {
        fn f(nb: *[4]usize, x: *const [4]i64) void {
            nb[1] *= @intCast(x[1]);
            nb[2] *= @intCast(x[2]);
            nb[3] *= @intCast(x[3]);
        }
    }.f;

    if (c.ggml_is_contiguous(src0) and c.ggml_is_contiguous(src1) and !c.ggml_is_permuted(src0) and !c.ggml_is_permuted(src1)) {
        for (0..4) |i| {
            if (nr[i] != 1) break;
            if (i > 0) {
                collapseNb(&cnb, &cne);
                collapseNb(&cnb0, &cne0);
                collapseNb(&cnb1, &cne1);
                collapse(&cne);
                collapse(&cne0);
                collapse(&cne1);
            }
        }
    }

    const ne0 = cne[0];
    const ne1 = cne[1];
    const ne2 = cne[2];
    const ne3 = cne[3];

    const nb0 = cnb[0];
    const nb1 = cnb[1];
    const nb2 = cnb[2];
    const nb3 = cnb[3];
    const nb00 = cnb0[0];
    const nb01 = cnb0[1];
    const nb02 = cnb0[2];
    const nb03 = cnb0[3];
    const nb10 = cnb1[0];
    const nb11 = cnb1[1];
    const nb12 = cnb1[2];
    const nb13 = cnb1[3];

    const s1 = nb1 / @sizeOf(D);
    const s2 = nb2 / @sizeOf(D);
    const s3 = nb3 / @sizeOf(D);

    const s10 = nb10 / @sizeOf(S1);
    const s11 = nb11 / @sizeOf(S1);
    const s12 = nb12 / @sizeOf(S1);
    const s13 = nb13 / @sizeOf(S1);

    const s00 = nb00 / @sizeOf(S0);
    const s01 = nb01 / @sizeOf(S0);
    const s02 = nb02 / @sizeOf(S0);
    const s03 = nb03 / @sizeOf(S0);

    impl.assert(ne0 <= u32_max, "ne0 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(ne1 <= u32_max, "ne1 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(ne2 <= u32_max, "ne2 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(ne3 <= u32_max, "ne3 <= std::numeric_limits<uint32_t>::max()");

    impl.assert(s1 <= u32_max, "s1  <= std::numeric_limits<uint32_t>::max()");
    impl.assert(s2 <= u32_max, "s2  <= std::numeric_limits<uint32_t>::max()");
    impl.assert(s3 <= u32_max, "s3  <= std::numeric_limits<uint32_t>::max()");

    impl.assert(s00 <= u32_max, "s00 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(s01 <= u32_max, "s01 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(s02 <= u32_max, "s02 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(s03 <= u32_max, "s03 <= std::numeric_limits<uint32_t>::max()");

    impl.assert(s10 <= u32_max, "s10 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(s11 <= u32_max, "s11 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(s12 <= u32_max, "s12 <= std::numeric_limits<uint32_t>::max()");
    impl.assert(s13 <= u32_max, "s13 <= std::numeric_limits<uint32_t>::max()");

    impl.assert(cne1[0] <= u32_max, "cne1[0] <= std::numeric_limits<uint32_t>::max()");
    impl.assert(cne1[1] <= u32_max, "cne1[1] <= std::numeric_limits<uint32_t>::max()");
    impl.assert(cne1[2] <= u32_max, "cne1[2] <= std::numeric_limits<uint32_t>::max()");
    impl.assert(cne1[3] <= u32_max, "cne1[3] <= std::numeric_limits<uint32_t>::max()");

    impl.assert(nb0 % @sizeOf(D) == 0, "nb0 % sizeof(dst_t) == 0");
    impl.assert(nb1 % @sizeOf(D) == 0, "nb1 % sizeof(dst_t) == 0");
    impl.assert(nb2 % @sizeOf(D) == 0, "nb2 % sizeof(dst_t) == 0");
    impl.assert(nb3 % @sizeOf(D) == 0, "nb3 % sizeof(dst_t) == 0");

    impl.assert(nb00 % @sizeOf(S0) == 0, "nb00 % sizeof(src0_t) == 0");
    impl.assert(nb01 % @sizeOf(S0) == 0, "nb01 % sizeof(src0_t) == 0");
    impl.assert(nb02 % @sizeOf(S0) == 0, "nb02 % sizeof(src0_t) == 0");
    impl.assert(nb03 % @sizeOf(S0) == 0, "nb03 % sizeof(src0_t) == 0");

    impl.assert(nb10 % @sizeOf(S1) == 0, "nb10 % sizeof(src1_t) == 0");
    impl.assert(nb11 % @sizeOf(S1) == 0, "nb11 % sizeof(src1_t) == 0");
    impl.assert(nb12 % @sizeOf(S1) == 0, "nb12 % sizeof(src1_t) == 0");
    impl.assert(nb13 % @sizeOf(S1) == 0, "nb13 % sizeof(src1_t) == 0");

    impl.assert(ne2 * ne3 <= u32_max, "ne2 * ne3 <= std::numeric_limits<unsigned int>::max()");

    const block_size = 128;

    const hne0: i64 = @max(@divTrunc(ne0, 2), 1);

    // `dim3 block_dims` with each member assigned through
    // `std::min<unsigned int>`, which converts both arguments first.
    var block_dims: kernels.Dim3 = .{};
    block_dims.x = @min(@as(u32, @truncate(@as(u64, @bitCast(hne0)))), block_size);
    block_dims.y = @min(@as(u32, @truncate(@as(u64, @bitCast(ne1)))), block_size / block_dims.x);
    block_dims.z = @min(@min(@as(u32, @truncate(@as(u64, @bitCast(ne2 * ne3)))), block_size / block_dims.x / block_dims.y), 64);

    const block_nums: kernels.Dim3 = .{
        .x = @truncate(@as(u64, @bitCast(@divTrunc(hne0 + block_dims.x - 1, block_dims.x)))),
        .y = @truncate(@as(u64, @bitCast(@divTrunc(ne1 + block_dims.y - 1, block_dims.y)))),
        .z = @truncate(@as(u64, @bitCast(@divTrunc(ne2 * ne3 + block_dims.z - 1, block_dims.z)))),
    };

    const fd10 = common.initFastdivValues(@as(u32, @truncate(@as(u64, @bitCast(cne1[0])))));
    const fd11 = common.initFastdivValues(@as(u32, @truncate(@as(u64, @bitCast(cne1[1])))));
    const fd12 = common.initFastdivValues(@as(u32, @truncate(@as(u64, @bitCast(cne1[2])))));
    const fd13 = common.initFastdivValues(@as(u32, @truncate(@as(u64, @bitCast(cne1[3])))));

    // `(const src1_t *) dst->src[I + 1]->data...`
    var pack: [n_fuse]?*const anyopaque = undefined;
    inline for (0..n_fuse) |i| pack[i] = dst.src[i + 1].*.data;

    const p0: ?*const anyopaque = src0_dd;
    const p1: ?*const anyopaque = src1_dd;
    const pd: ?*anyopaque = dst_dd;

    if (block_nums.z > 65535 or block_nums.y > 65535) {
        const block_num: i64 = @divTrunc(ne0 * ne1 * ne2 * ne3 + block_size - 1, block_size);

        impl.assert(block_num <= u32_max, "block_num              <= std::numeric_limits<uint32_t>::max()");
        impl.assert(block_num * block_size <= u32_max, "block_num * block_size <= std::numeric_limits<uint32_t>::max()");
        impl.assert(ne0 * ne1 <= u32_max, "ne0 * ne1              <= std::numeric_limits<uint32_t>::max()");
        impl.assert(ne0 * ne1 * ne2 <= u32_max, "ne0 * ne1 * ne2        <= std::numeric_limits<uint32_t>::max()");

        const prod_012 = common.initFastdivValues(@intCast(ne0 * ne1 * ne2));
        const prod_01 = common.initFastdivValues(@intCast(ne0 * ne1));
        const ne0_fastdiv = common.initFastdivValues(@intCast(ne0));
        const ne1_fastdiv = common.initFastdivValues(@intCast(ne1));
        const ne2_fastdiv = common.initFastdivValues(@intCast(ne2));

        kernels.launchPdl(binKernel(true, op, S0, S1, D, n_fuse), .{
            .grid = .{ .x = @intCast(block_num) },
            .block = .{ .x = block_size },
            .stream = stream,
        }, kernels.append(.{
            p0,                      p1,                      pd,
            ne0_fastdiv,             ne1_fastdiv,             ne2_fastdiv,
            @as(u32, @intCast(ne3)), prod_012,                prod_01,
            fd10,                    fd11,                    fd12,
            fd13,                    @as(u32, @intCast(s1)),  @as(u32, @intCast(s2)),
            @as(u32, @intCast(s3)),  @as(u32, @intCast(s00)), @as(u32, @intCast(s01)),
            @as(u32, @intCast(s02)), @as(u32, @intCast(s03)), @as(u32, @intCast(s10)),
            @as(u32, @intCast(s11)), @as(u32, @intCast(s12)), @as(u32, @intCast(s13)),
        }, n_fuse, ?*const anyopaque, pack));
    } else {
        impl.assert(@as(i64, block_nums.x) * block_dims.x <= u32_max, "int64_t(block_nums.x) * block_dims.x <= std::numeric_limits<uint32_t>::max()");
        impl.assert(@as(i64, block_nums.y) * block_dims.y <= u32_max, "int64_t(block_nums.y) * block_dims.y <= std::numeric_limits<uint32_t>::max()");
        impl.assert(@as(i64, block_nums.z) * block_dims.z <= u32_max, "int64_t(block_nums.z) * block_dims.z <= std::numeric_limits<uint32_t>::max()");

        const ne3_fastdiv = common.initFastdivValues(@intCast(ne3));
        kernels.launchPdl(binKernel(false, op, S0, S1, D, n_fuse), .{
            .grid = block_nums,
            .block = block_dims,
            .stream = stream,
        }, kernels.append(.{
            p0,                      p1,                      pd,
            @as(u32, @intCast(ne0)), @as(u32, @intCast(ne1)), @as(u32, @intCast(ne2)),
            ne3_fastdiv,             fd10,                    fd11,
            fd12,                    fd13,                    @as(u32, @intCast(s1)),
            @as(u32, @intCast(s2)),  @as(u32, @intCast(s3)),  @as(u32, @intCast(s00)),
            @as(u32, @intCast(s01)), @as(u32, @intCast(s02)), @as(u32, @intCast(s03)),
            @as(u32, @intCast(s10)), @as(u32, @intCast(s11)), @as(u32, @intCast(s12)),
            @as(u32, @intCast(s13)),
        }, n_fuse, ?*const anyopaque, pack));
    }
}

/// Ports `ggml_cuda_op_bin_bcast` (binbcast.cu:412 @c1d0e7a00) with the
/// `bin_bcast_cuda<op, n_fuse>` functor it is instantiated with
/// (binbcast.cu:355 @c1d0e7a00).
fn opBinBcast(comptime op: Op, comptime n_fuse: usize, src0: *const Tensor, src1: *const Tensor, dst: *const Tensor, src0_dd: ?*const anyopaque, src1_dd: ?*const anyopaque, dst_dd: ?*anyopaque, stream: cu.cudaStream_t) void {
    impl.assert(src1.type == c.GGML_TYPE_F32 or src1.type == c.GGML_TYPE_F16, "src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16");

    const F32 = c.GGML_TYPE_F32;
    const F16 = c.GGML_TYPE_F16;
    if (src0.type == F32 and dst.type == F32) {
        launchBinBcastPack(op, f32, f32, f32, n_fuse, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
    } else if (src0.type == F16 and src1.type == F16 and dst.type == F16) {
        launchBinBcastPack(op, half, half, half, n_fuse, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
    } else if (src0.type == F16 and src1.type == F32 and dst.type == F16) {
        launchBinBcastPack(op, half, f32, half, n_fuse, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
    } else if (src0.type == F16 and dst.type == F32) {
        launchBinBcastPack(op, half, f32, f32, n_fuse, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
    } else {
        impl.logError("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", .{ "ggml_cuda_op_bin_bcast", c.ggml_type_name(dst.type), c.ggml_type_name(src0.type), c.ggml_type_name(src1.type) });
        impl.abort("fatal error");
    }
}

fn srcOf(dst: *const Tensor, i: usize) *const Tensor {
    return impl.one(Tensor, dst.src[i]);
}

/// Ports `ggml_cuda_op_repeat` (binbcast.cu:433 @c1d0e7a00): `src0` is `dst`
/// itself, read through a null pointer, so the kernel starts every element
/// from zero.
pub fn opRepeat(ctx: *common.Context, dst: *Tensor) void {
    opBinBcast(.repeat, 0, dst, srcOf(dst, 0), dst, null, srcOf(dst, 0).data, dst.data, ctx.stream());
}

/// Ports `ggml_cuda_op_add`, `ggml_cuda_op_sub`, `ggml_cuda_op_mul` and
/// `ggml_cuda_op_div` (binbcast.cu:437, 441, 445, 449 @c1d0e7a00).
pub fn opBinary(comptime op: Op, ctx: *common.Context, dst: *Tensor) void {
    const s0 = srcOf(dst, 0);
    const s1 = srcOf(dst, 1);
    opBinBcast(op, 1, s0, s1, dst, s0.data, s1.data, dst.data, ctx.stream());
}

/// Ports `ggml_cuda_op_fused_binbcast_impl` (binbcast.cu:454 @c1d0e7a00).
fn fusedBinBcastImpl(comptime op: Op, comptime n_fuse: usize, ctx: *common.Context, dst: *Tensor) void {
    const stream = ctx.stream();
    const src0 = srcOf(dst, 0);
    const src1 = srcOf(dst, 1);

    const F32 = c.GGML_TYPE_F32;
    const F16 = c.GGML_TYPE_F16;
    if (src0.type == F32 and dst.type == F32) {
        launchBinBcastPack(op, f32, f32, f32, n_fuse, src0, src1, dst, src0.data, src1.data, dst.data, stream);
    } else if (src0.type == F16 and src1.type == F16 and dst.type == F16) {
        launchBinBcastPack(op, half, half, half, n_fuse, src0, src1, dst, src0.data, src1.data, dst.data, stream);
    } else if (src0.type == F16 and src1.type == F32 and dst.type == F16) {
        launchBinBcastPack(op, half, f32, half, n_fuse, src0, src1, dst, src0.data, src1.data, dst.data, stream);
    } else if (src0.type == F16 and dst.type == F32) {
        launchBinBcastPack(op, half, f32, f32, n_fuse, src0, src1, dst, src0.data, src1.data, dst.data, stream);
    } else {
        impl.logError("%s: unsupported types for fusion: dst: %s, src0: %s, src1: %s\n", .{ "ggml_cuda_op_fused_binbcast_impl", c.ggml_type_name(dst.type), c.ggml_type_name(src0.type), c.ggml_type_name(src1.type) });
        impl.abort("fatal error");
    }
}

/// Ports `ggml_cuda_op_fused_add` and `ggml_cuda_op_fused_mul`
/// (binbcast.cu:485, 515 @c1d0e7a00).
pub fn opFused(comptime op: Op, ctx: *common.Context, dst: *Tensor, n_fuse: c_int) void {
    impl.assert(2 <= n_fuse and n_fuse <= 8, "2 <= n_fuse && n_fuse <= 8");
    switch (n_fuse) {
        inline 2...8 => |n| fusedBinBcastImpl(op, n, ctx, dst),
        else => impl.assert(false, "false && \"Unsupported n_fuse value\""),
    }
}

/// Ports `ggml_cuda_op_repeat_back` (binbcast.cu:545 @c1d0e7a00) with
/// `repeat_back_cuda` (binbcast.cu:400 @c1d0e7a00), which launches with
/// `<<<>>>`, not PDL.
pub fn opRepeatBack(ctx: *common.Context, dst: *Tensor) void {
    const src0 = srcOf(dst, 0);

    impl.assert(src0.type == dst.type, "src0->type == dst->type");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    impl.assert(c.ggml_can_repeat(dst, src0), "ggml_can_repeat(dst, src0)");

    const stream = ctx.stream();
    const ne0 = dst.ne[0];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];

    impl.assert(ne2 * ne3 <= (1 << 15), "ne2*ne3 <= (1 << 15)");

    const ts = c.ggml_type_size(src0.type);
    const s00 = src0.nb[0] / ts;
    const s01 = src0.nb[1] / ts;
    const s02 = src0.nb[2] / ts;
    const s03 = src0.nb[3] / ts;

    switch (dst.type) {
        c.GGML_TYPE_F32 => {
            kernels.launch(kernels.get("binbcast", "k_repeat_back<float>"), .{
                .grid = kernels.Dim3.of(@divTrunc(ne0 + common.warp_size - 1, common.warp_size), ne1, ne2 * ne3),
                .block = .{ .x = common.warp_size },
                .stream = stream,
            }, .{
                @as(?*const anyopaque, src0.data), @as(?*anyopaque, dst.data),
                src0.ne[0],                        src0.ne[1],
                src0.ne[2],                        src0.ne[3],
                s00,                               s01,
                s02,                               s03,
                ne0,                               ne1,
                ne2,                               ne3,
            });
        },
        else => impl.assert(false, "false"),
    }
}
