//! The flash-attention launcher every kernel family shares: the scratch
//! space for F16 copies of K and V, the occupancy-driven grid choice, the
//! KV-mask scan, and the stream-k fixup or parallel-block combine that
//! follows the main kernel.
//!
//! # Provenance
//!
//! Ported from the host half of
//! `llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh` at v0.4.0 (`5266f24da`),
//! plus `amd_mfma_available` and `amd_wmma_available` from
//! `llama.cpp/ggml/src/ggml-cuda/common.cuh`. The kernels --
//! `flash_attn_mask_to_KV_max`, `flash_attn_stream_k_fixup_uniform`,
//! `flash_attn_stream_k_fixup_general` and `flash_attn_combine_results` --
//! stay CUDA. The sparse K/V path v0.4.0 added (`use_sparse`) reuses the
//! `KV_max` buffer for the indices `fattn.compactMask` writes, which is
//! `fattn.cu`'s and launches from that translation unit.
//!
//! # Which translation unit
//!
//! `launch_fattn` is a template, instantiated in whichever translation unit
//! instantiates the case that calls it, and the helper kernels it launches
//! are `static`: every `template-instances/fattn-*.cu` carries its own copy.
//! `launchFattn` therefore takes the caller's translation unit and launches
//! the helpers from it, as the C++ does.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const convert = @import("convert.zig");
const fattn = @import("fattn.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

extern fn powf(x: f32, y: f32) f32;
extern fn log2f(x: f32) f32;
extern fn floorf(x: f32) f32;

/// Ports `FATTN_KQ_STRIDE` (fattn-common.cuh:9 @5266f24da).
pub const kq_stride = 256;

/// `sizeof(half)`.
const sizeof_half = 2;
/// `sizeof(half2)`.
const sizeof_half2 = 4;
/// `sizeof(float2)`.
const sizeof_float2 = 8;

/// `float2`, as the kernels' `dst_meta` element.
const Float2 = extern struct { x: f32, y: f32 };

// -----------------------------------------------------------------------------
// AMD predicates

/// `GGML_CUDA_CC_CDNA1` (common.cuh:72 @5266f24da).
const cc_cdna1 = common.cc.offset_amd + 0x908;
/// `GGML_CUDA_CC_RDNA1` (common.cuh:78 @5266f24da).
const cc_rdna1 = common.cc.offset_amd + 0x1010;
/// `GGML_CUDA_CC_RDNA3` (common.cuh:80 @5266f24da).
const cc_rdna3 = common.cc.offset_amd + 0x1100;
/// `GGML_CUDA_CC_RDNA3_5` (common.cuh:81 @5266f24da).
const cc_rdna3_5 = common.cc.offset_amd + 0x1150;
/// `GGML_CUDA_CC_RDNA4` (common.cuh:82 @5266f24da).
const cc_rdna4 = common.cc.offset_amd + 0x1200;

/// Ports `amd_mfma_available` (common.cuh:333 @5266f24da), the
/// `!GGML_HIP_NO_MMQ_MFMA` arm: `GGML_CUDA_CC_IS_CDNA(cc)`. False for every
/// CUDA compute capability.
pub fn amdMfmaAvailable(x: c_int) bool {
    return x >= cc_cdna1 and x < cc_rdna1;
}

/// Ports `amd_wmma_available` (common.cuh:341 @5266f24da):
/// `GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc)`. False for every
/// CUDA compute capability.
pub fn amdWmmaAvailable(x: c_int) bool {
    const rdna3_0 = x >= cc_rdna3 and x < cc_rdna3_5;
    const rdna3_5 = x >= cc_rdna3_5 and x < cc_rdna4;
    return x >= cc_rdna4 or (rdna3_0 or rdna3_5);
}

// -----------------------------------------------------------------------------
// C conversions

/// A C++ implicit conversion of an integer to `int32_t`: modular.
fn i32Of(x: anytype) i32 {
    return switch (@TypeOf(x)) {
        usize, u64 => @bitCast(@as(u32, @truncate(x))),
        i64 => @truncate(x),
        i32, c_int => x,
        else => @compileError("i32Of: " ++ @typeName(@TypeOf(x))),
    };
}

/// A C++ implicit conversion of `size_t` to `int64_t`.
fn i64Of(x: usize) i64 {
    return @bitCast(@as(u64, x));
}

/// A C++ implicit conversion of an integer to `unsigned int`: modular.
fn u32Of(x: anytype) u32 {
    return switch (@TypeOf(x)) {
        usize, u64 => @truncate(x),
        i64 => @truncate(@as(u64, @bitCast(x))),
        i32, c_int => @bitCast(x),
        else => @compileError("u32Of: " ++ @typeName(@TypeOf(x))),
    };
}

// -----------------------------------------------------------------------------
// F16 scratch

/// Ports `ggml_cuda_flash_attn_ext_f16_extra_data` (fattn-common.cuh:47 @5266f24da).
pub const F16ExtraData = struct {
    K: usize = 0,
    V: usize = 0,
    end: usize = 0,
};

/// Ports `ggml_cuda_flash_attn_ext_get_f16_extra_data` (fattn-common.cuh:53 @5266f24da):
/// where in `dst`'s allocation, past `dst`'s own bytes, the F16 copies of K
/// and V go. Computed from `dst->data`, so the buffer type's size (asked
/// with whatever `data` the tensor has then) and the launch agree only
/// because both pad the end to 128 bytes.
///
/// Parameters:
/// - `dst`: a `GGML_OP_FLASH_ATTN_EXT` node.
/// - `need_f16_K`, `need_f16_V`: whether the kernel reads K and V as F16.
///
/// Return: the addresses of the two copies (0 when not needed) and the end
/// of the region.
pub fn getF16ExtraData(dst: *const Tensor, need_f16_K: bool, need_f16_V: bool) F16ExtraData {
    impl.assert(dst.op == c.GGML_OP_FLASH_ATTN_EXT, "dst->op == GGML_OP_FLASH_ATTN_EXT");

    const K_opt: ?*const Tensor = dst.src[1];
    const V_opt: ?*const Tensor = dst.src[2];

    impl.assert(K_opt != null, "K != nullptr");
    impl.assert(V_opt != null, "V != nullptr");
    const K = K_opt.?;
    const V = V_opt.?;

    const V_is_K_view = vIsKView(K, V);

    var data: F16ExtraData = .{};
    data.end = @intFromPtr(dst.data) +% c.ggml_nbytes(dst);

    if (need_f16_K and K.type != c.GGML_TYPE_F16) {
        data.end = impl.pad(data.end, 128);
        data.K = data.end;
        data.end +%= @as(usize, @bitCast(c.ggml_nelements(K))) *% c.ggml_type_size(c.GGML_TYPE_F16);
    }

    if (need_f16_V and V.type != c.GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = impl.pad(data.end, 128);
            data.V = data.end;
            data.end +%= @as(usize, @bitCast(c.ggml_nelements(V))) *% c.ggml_type_size(c.GGML_TYPE_F16);
        }
    }

    return data;
}

/// `V->view_src && (V->view_src == K || (V->view_src == K->view_src &&
/// V->view_offs == K->view_offs))`, as both `get_f16_extra_data` and
/// `launch_fattn` spell it.
fn vIsKView(K: *const Tensor, V: *const Tensor) bool {
    const V_src: ?*const Tensor = V.view_src;
    const K_src: ?*const Tensor = K.view_src;
    return V_src != null and (V_src.? == K or (V_src == K_src and V.view_offs == K.view_offs));
}

// -----------------------------------------------------------------------------
// The launcher

/// `CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel,
/// block_size, nbytes_shared))`, on the current device. The runtime call
/// forwards to the driver's, which takes the kernel's function in the
/// current context.
fn occupancyMaxActiveBlocksPerMultiprocessor(k: kernels.Kernel, block_size: c_int, nbytes_shared: usize) c_int {
    var f: cu.CUfunction = null;
    common.cuCheck(cu.cuKernelGetFunction(&f, kernels.handle(k)), "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared)", @src());
    var n: c_int = 0;
    common.cuCheck(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, f, block_size, nbytes_shared), "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared)", @src());
    return n;
}

/// The arguments of `launch_fattn` that are not template parameters.
pub const LaunchArgs = struct {
    /// The main kernel, `fattn_kernel_t`.
    kernel: kernels.Kernel,
    nwarps: c_int,
    nbytes_shared: usize,
    nbatch_fa: c_int,
    need_f16_K: bool,
    need_f16_V: bool,
    stream_k: bool,
    /// Gather K/V through the sparse indices `fattn.compactMask` writes
    /// from the mask's finite entries, `ggml_flash_attn_ext_set_n_kv_max`'s
    /// op_params slot 4 of them per mask row; no default, as the C has none.
    use_sparse: bool,
    warp_size: c_int = common.warp_size,
};

/// Ports `launch_fattn` (fattn-common.cuh:976 @5266f24da).
///
/// Parameters:
/// - `library`: the translation unit the calling case is instantiated in;
///   the helper kernels come from it.
/// - `DV`, `ncols1`, `ncols2`: the C's template parameters.
/// - `ctx`, `dst`: the backend context and the `FLASH_ATTN_EXT` node.
/// - `a`: the C's remaining arguments.
pub fn launchFattn(
    comptime library: []const u8,
    comptime DV: comptime_int,
    comptime ncols1: comptime_int,
    comptime ncols2: comptime_int,
    ctx: *common.Context,
    dst: *Tensor,
    a: LaunchArgs,
) void {
    const ncols = ncols1 * ncols2;

    const Q = impl.one(Tensor, dst.src[0]);
    const K = impl.one(Tensor, dst.src[1]);
    const V = impl.one(Tensor, dst.src[2]);

    const V_is_K_view = vIsKView(K, V);

    const mask: ?*const Tensor = dst.src[3];
    const sinks: ?*const Tensor = dst.src[4];

    const KQV = dst;

    impl.assert(Q.type == c.GGML_TYPE_F32, "Q->type == GGML_TYPE_F32");
    impl.assert(KQV.type == c.GGML_TYPE_F32, "KQV->type == GGML_TYPE_F32");

    impl.assert(Q.nb[0] == c.ggml_element_size(Q), "Q->nb[0] == ggml_element_size(Q)");
    impl.assert(K.nb[0] == c.ggml_element_size(K), "K->nb[0] == ggml_element_size(K)");
    impl.assert(V.nb[0] == c.ggml_element_size(V), "V->nb[0] == ggml_element_size(V)");

    impl.assert(mask == null or mask.?.type == c.GGML_TYPE_F16, "!mask || mask->type == GGML_TYPE_F16");

    const pool = ctx.pool();
    const main_stream = ctx.stream();
    const id = common.getDevice();
    const cc = common.info().devices[@intCast(id)].cc;
    const nsm = common.info().devices[@intCast(id)].nsm;

    const f16_extra = getF16ExtraData(KQV, a.need_f16_K, a.need_f16_V);

    var KV_max = common.PoolAlloc(c_int).init(pool);
    defer KV_max.deinit();
    var dst_tmp = common.PoolAlloc(f32).init(pool);
    defer dst_tmp.deinit();
    var dst_tmp_meta = common.PoolAlloc(Float2).init(pool);
    defer dst_tmp_meta.deinit();

    var K_data: [*c]const u8 = @ptrCast(K.data);
    var nb11: usize = K.nb[1];
    var nb12: usize = K.nb[2];
    var nb13: usize = K.nb[3];

    var V_data: [*c]const u8 = @ptrCast(V.data);
    var nb21: usize = V.nb[1];
    var nb22: usize = V.nb[2];
    var nb23: usize = V.nb[3];

    if (a.need_f16_K and K.type != c.GGML_TYPE_F16) {
        const bs: usize = @intCast(c.ggml_blck_size(K.type));
        const ts: usize = c.ggml_type_size(K.type);

        impl.assert(f16_extra.K != 0, "f16_extra.K != 0");
        const K_f16: ?*anyopaque = @ptrFromInt(f16_extra.K);
        if (c.ggml_is_contiguously_allocated(K)) {
            const to_fp16 = convert.getToFp16(K.type).?;
            to_fp16(K_data, K_f16, c.ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * sizeof_half / ts;
            nb12 = nb12 * bs * sizeof_half / ts;
            nb13 = nb13 * bs * sizeof_half / ts;
        } else {
            impl.assert(K.nb[0] == ts, "K->nb[0] == ts");
            const to_fp16 = convert.getToFp16Nc(K.type).?;
            const s01 = i64Of(nb11 / ts);
            const s02 = i64Of(nb12 / ts);
            const s03 = i64Of(nb13 / ts);
            to_fp16(K_data, K_f16, K.ne[0], K.ne[1], K.ne[2], K.ne[3], s01, s02, s03, main_stream);

            nb11 = @as(usize, @bitCast(K.ne[0])) *% sizeof_half;
            nb12 = @as(usize, @bitCast(K.ne[1])) *% nb11;
            nb13 = @as(usize, @bitCast(K.ne[2])) *% nb12;
        }
        K_data = @ptrCast(K_f16);
    }

    if (a.need_f16_V and V.type != c.GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21 = nb11;
            nb22 = nb12;
            nb23 = nb13;
        } else {
            const bs: usize = @intCast(c.ggml_blck_size(V.type));
            const ts: usize = c.ggml_type_size(V.type);

            impl.assert(f16_extra.V != 0, "f16_extra.V != 0");
            const V_f16: ?*anyopaque = @ptrFromInt(f16_extra.V);
            if (c.ggml_is_contiguously_allocated(V)) {
                const to_fp16 = convert.getToFp16(V.type).?;
                to_fp16(V_data, V_f16, c.ggml_nelements(V), main_stream);
                V_data = @ptrCast(V_f16);

                nb21 = nb21 * bs * sizeof_half / ts;
                nb22 = nb22 * bs * sizeof_half / ts;
                nb23 = nb23 * bs * sizeof_half / ts;
            } else {
                impl.assert(V.nb[0] == ts, "V->nb[0] == ts");
                const to_fp16 = convert.getToFp16Nc(V.type).?;
                const s01 = i64Of(nb21 / ts);
                const s02 = i64Of(nb22 / ts);
                const s03 = i64Of(nb23 / ts);
                to_fp16(V_data, V_f16, V.ne[0], V.ne[1], V.ne[2], V.ne[3], s01, s02, s03, main_stream);

                nb21 = @as(usize, @bitCast(V.ne[0])) *% sizeof_half;
                nb22 = @as(usize, @bitCast(V.ne[1])) *% nb21;
                nb23 = @as(usize, @bitCast(V.ne[2])) *% nb22;
            }
            V_data = @ptrCast(V_f16);
        }
    }

    const ntiles_x: c_int = i32Of(@divTrunc(Q.ne[1] + ncols1 - 1, ncols1));
    const gqa_ratio: c_int = i32Of(@divTrunc(Q.ne[2], K.ne[2]));
    const ntiles_z_gqa: c_int = @divTrunc(gqa_ratio + ncols2 - 1, ncols2);
    const ntiles_dst: c_int = i32Of(@as(i64, ntiles_x * ntiles_z_gqa) * K.ne[2] * Q.ne[3]);

    const n_kv_max: i32 = if (a.use_sparse) impl.getOpParamsI32(KQV, 4) else 0;
    if (a.use_sparse) {
        impl.assert(mask != null, "mask != nullptr");
        impl.assert(n_kv_max > 0, "n_kv_max > 0");
        const m = mask.?;
        const mask_rows: usize = @as(usize, @bitCast(m.ne[1])) *% @as(usize, @bitCast(m.ne[3]));

        _ = KV_max.alloc(@as(usize, @intCast(n_kv_max)) *% mask_rows);
        fattn.compactMask(m, KV_max.ptr, n_kv_max, main_stream);
    }

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (!a.use_sparse and mask != null and @rem(K.ne[1], kq_stride) == 0 and (Q.ne[1] >= 1024 or Q.ne[3] > 1)) {
        const m = mask.?;
        const s31 = i64Of(m.nb[1] / sizeof_half2);
        const s33 = i64Of(m.nb[3] / sizeof_half2);

        const blocks_num_KV_max: kernels.Dim3 = .{ .x = u32Of(ntiles_x), .y = u32Of(Q.ne[3]), .z = 1 };
        const block_dim_KV_max: kernels.Dim3 = .{ .x = kq_stride / 2, .y = 1, .z = 1 };

        const ne_KV_max: c_int = @bitCast(blocks_num_KV_max.x *% blocks_num_KV_max.y);
        const iter_k: c_int = i32Of(@divTrunc(K.ne[1], kq_stride));

        _ = KV_max.alloc(@intCast(ne_KV_max));
        kernels.launchPdl(kernels.get(library, kernels.tmpl("flash_attn_mask_to_KV_max", &.{kernels.int(ncols1)})), .{
            .grid = blocks_num_KV_max,
            .block = block_dim_KV_max,
            .shmem = 0,
            .stream = main_stream,
        }, .{ @as(?*const anyopaque, m.data), @as(?[*]c_int, KV_max.ptr), iter_k, s31, s33 });
    }

    const block_dim: kernels.Dim3 = .{ .x = u32Of(a.warp_size), .y = u32Of(a.nwarps), .z = 1 };
    var max_blocks_per_sm: c_int = 1; // Max. number of active blocks limited by occupancy.
    max_blocks_per_sm = occupancyMaxActiveBlocksPerMultiprocessor(a.kernel, @bitCast(block_dim.x *% block_dim.y *% block_dim.z), a.nbytes_shared);
    impl.assert(max_blocks_per_sm > 0, "max_blocks_per_sm > 0");
    var parallel_blocks: c_int = max_blocks_per_sm;

    const n_kv: i64 = if (a.use_sparse) n_kv_max else K.ne[1];
    const ntiles_KV: c_int = i32Of(@divTrunc(n_kv + a.nbatch_fa - 1, a.nbatch_fa)); // Max. number of parallel blocks limited by KV cache length.

    var blocks_num: kernels.Dim3 = .{};
    if (a.stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const max_blocks = max_blocks_per_sm * nsm;
        const tiles_nwaves = @divTrunc(ntiles_dst + max_blocks - 1, max_blocks);
        const tiles_efficiency_percent = @divTrunc(100 * ntiles_dst, max_blocks * tiles_nwaves);

        const use_stream_k = cc >= common.cc.ada_lovelace or amdWmmaAvailable(cc) or tiles_efficiency_percent < 75;

        blocks_num.x = u32Of(ntiles_dst);
        blocks_num.y = 1;
        blocks_num.z = 1;

        if (use_stream_k) {
            const nblocks_stream_k_raw = @min(max_blocks, ntiles_KV * ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const nblocks_stream_k_rounded = @divTrunc(nblocks_stream_k_raw, ntiles_dst) * ntiles_dst;
            const max_efficiency_loss_percent = 5;
            const efficiency_loss_percent: c_int = if (nblocks_stream_k_rounded > 0)
                @divTrunc(100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded), nblocks_stream_k_raw)
            else
                100;
            const nblocks_stream_k = if (efficiency_loss_percent <= max_efficiency_loss_percent)
                nblocks_stream_k_rounded
            else
                nblocks_stream_k_raw;

            blocks_num.x = u32Of(nblocks_stream_k);
        }

        if (u32Of(ntiles_dst) % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            _ = dst_tmp_meta.alloc(@as(usize, blocks_num.x) * ncols * (2 + DV / 2));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = @min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const blocks_per_wave = nsm * max_blocks_per_sm;
        var nwaves_best: c_int = 0;
        var efficiency_percent_best: c_int = 0;
        var parallel_blocks_test = parallel_blocks;
        while (parallel_blocks_test <= ntiles_KV) : (parallel_blocks_test += 1) {
            const nblocks_total = ntiles_dst * parallel_blocks_test;
            const nwaves = @divTrunc(nblocks_total + blocks_per_wave - 1, blocks_per_wave);
            const efficiency_percent = @divTrunc(100 * nblocks_total, nwaves * blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 and nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = u32Of(ntiles_x);
        blocks_num.y = u32Of(parallel_blocks);
        blocks_num.z = u32Of(@as(i64, ntiles_z_gqa) * K.ne[2] * Q.ne[3]);

        if (parallel_blocks > 1) {
            _ = dst_tmp.alloc(@bitCast(@as(i64, parallel_blocks) * c.ggml_nelements(KQV)));
            _ = dst_tmp_meta.alloc(@bitCast(@as(i64, parallel_blocks) * c.ggml_nrows(KQV)));
        }
    }

    var scale = impl.getOpParamsF32(KQV, 0);
    const max_bias = impl.getOpParamsF32(KQV, 1);
    const logit_softcap = impl.getOpParamsF32(KQV, 2);

    if (logit_softcap != 0.0) {
        scale /= logit_softcap;
    }

    const n_head: u32 = u32Of(Q.ne[2]);
    const n_head_log2: u32 = @as(u32, 1) << @intFromFloat(floorf(log2f(@floatFromInt(n_head))));

    const m0 = powf(2.0, -(max_bias) / @as(f32, @floatFromInt(n_head_log2)));
    const m1 = powf(2.0, -(max_bias / 2.0) / @as(f32, @floatFromInt(n_head_log2)));

    // TODO other tensor dimensions after removal of WMMA kernel:
    const ne01 = common.initFastdivValues(@bitCast(Q.ne[1]));

    impl.assert(@rem(@as(c_int, @bitCast(block_dim.x)), a.warp_size) == 0, "block_dim.x % warp_size == 0");

    const dst_main: ?[*]f32 = if (!a.stream_k and parallel_blocks > 1) dst_tmp.ptr else @ptrCast(@alignCast(KQV.data));
    kernels.launchPdl(a.kernel, .{
        .grid = blocks_num,
        .block = block_dim,
        .shmem = a.nbytes_shared,
        .stream = main_stream,
    }, .{
        @as([*c]const u8, @ptrCast(Q.data)),
        K_data,
        V_data,
        if (mask) |m| @as([*c]const u8, @ptrCast(m.data)) else @as([*c]const u8, null),
        if (sinks) |s| @as([*c]const u8, @ptrCast(s.data)) else @as([*c]const u8, null),
        @as(?[*]c_int, KV_max.ptr),
        dst_main,
        @as(?[*]Float2, dst_tmp_meta.ptr),
        scale,
        max_bias,
        m0,
        m1,
        n_head_log2,
        logit_softcap,
        i32Of(Q.ne[0]),
        ne01,
        i32Of(Q.ne[2]),
        i32Of(Q.ne[3]),
        i32Of(Q.nb[1]),
        i32Of(Q.nb[2]),
        i32Of(Q.nb[3]),
        i32Of(K.ne[0]),
        i32Of(n_kv),
        i32Of(K.ne[2]),
        i32Of(K.ne[3]),
        i32Of(nb11),
        i32Of(nb12),
        i64Of(nb13),
        i32Of(nb21),
        i32Of(nb22),
        i64Of(nb23),
        if (mask) |m| i32Of(m.ne[1]) else @as(i32, 0),
        if (mask) |m| i32Of(m.ne[2]) else @as(i32, 0),
        if (mask) |m| i32Of(m.ne[3]) else @as(i32, 0),
        if (mask) |m| i32Of(m.nb[1]) else @as(i32, 0),
        if (mask) |m| i32Of(m.nb[2]) else @as(i32, 0),
        if (mask) |m| i64Of(m.nb[3]) else @as(i64, 0),
    });

    const KQV_data: ?[*]f32 = @ptrCast(@alignCast(KQV.data));
    if (a.stream_k) {
        if (@rem(@as(c_int, @bitCast(blocks_num.x)), ntiles_dst) == 0 and @as(c_int, @bitCast(blocks_num.x)) > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const nblocks_sk: c_int = @bitCast(blocks_num.x);
            const bpt = @divTrunc(nblocks_sk, ntiles_dst);

            const fd0 = common.initFastdivValues(@bitCast(@as(i64, ntiles_x * ntiles_z_gqa) * K.ne[2]));
            const fd1 = common.initFastdivValues(@intCast(ntiles_x * ntiles_z_gqa));
            const fd2 = common.initFastdivValues(@intCast(ntiles_x));

            const block_dim_combine: kernels.Dim3 = .{ .x = DV, .y = 1, .z = 1 };
            const blocks_num_combine: kernels.Dim3 = .{ .x = u32Of(ntiles_dst), .y = ncols1, .z = ncols2 };

            kernels.launchPdl(kernels.get(library, kernels.tmpl("flash_attn_stream_k_fixup_uniform", &.{ kernels.int(DV), kernels.int(ncols1), kernels.int(ncols2) })), .{
                .grid = blocks_num_combine,
                .block = block_dim_combine,
                .shmem = 0,
                .stream = main_stream,
            }, .{ KQV_data, @as(?[*]const Float2, dst_tmp_meta.ptr), i32Of(Q.ne[1]), i32Of(Q.ne[2]), i32Of(K.ne[2]), nblocks_sk, gqa_ratio, bpt, fd0, fd1, fd2 });
        } else if (u32Of(ntiles_dst) % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const total_work = ntiles_KV * ntiles_dst;

            const fd_k_j_z_ne12 = common.initFastdivValues(@bitCast(@as(i64, ntiles_KV * ntiles_x * ntiles_z_gqa) * K.ne[2]));
            const fd_k_j_z = common.initFastdivValues(@intCast(ntiles_KV * ntiles_x * ntiles_z_gqa));
            const fd_k_j = common.initFastdivValues(@intCast(ntiles_KV * ntiles_x));
            const fd_k = common.initFastdivValues(@intCast(ntiles_KV));

            const block_dim_combine: kernels.Dim3 = .{ .x = DV, .y = 1, .z = 1 };
            const blocks_num_combine: kernels.Dim3 = .{ .x = blocks_num.x, .y = ncols1, .z = ncols2 };

            kernels.launchPdl(kernels.get(library, kernels.tmpl("flash_attn_stream_k_fixup_general", &.{ kernels.int(DV), kernels.int(ncols1), kernels.int(ncols2) })), .{
                .grid = blocks_num_combine,
                .block = block_dim_combine,
                .shmem = 0,
                .stream = main_stream,
            }, .{ KQV_data, @as(?[*]const Float2, dst_tmp_meta.ptr), i32Of(Q.ne[1]), i32Of(Q.ne[2]), gqa_ratio, total_work, fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k });
        }
    } else if (parallel_blocks > 1) {
        const block_dim_combine: kernels.Dim3 = .{ .x = DV, .y = 1, .z = 1 };
        const blocks_num_combine: kernels.Dim3 = .{ .x = u32Of(Q.ne[1]), .y = u32Of(Q.ne[2]), .z = u32Of(Q.ne[3]) };
        const nbytes_shared_combine: usize = @as(usize, @intCast(parallel_blocks)) * sizeof_float2;

        kernels.launchPdl(kernels.get(library, kernels.tmpl("flash_attn_combine_results", &.{kernels.int(DV)})), .{
            .grid = blocks_num_combine,
            .block = block_dim_combine,
            .shmem = nbytes_shared_combine,
            .stream = main_stream,
        }, .{ @as(?[*]const f32, dst_tmp.ptr), @as(?[*]const Float2, dst_tmp_meta.ptr), KQV_data, parallel_blocks });
    }
}
