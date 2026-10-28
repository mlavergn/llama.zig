//! The tensor-core flash-attention kernel, `flash_attn_ext_f16`: its
//! configuration lookup and its launch.
//!
//! # Provenance
//!
//! Ported from the host half of
//! `llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh` at v0.4.0 (`5266f24da`),
//! and the host half of `llama.cpp/ggml/src/ggml-cuda/fattn-swizzle.cuh`,
//! new at v0.4.0 (`5266f24da`).
//! The configuration tables are generated into `fattn_config_table.zig` by
//! `scripts/gen-fattn-config`. The kernel stays CUDA, one instantiation set
//! per `(ncols1, ncols2)` in
//! `template-instances/fattn-mma-f16-instance-ncols1_<n1>-ncols2_<n2>.cu`.
//!
//! # Which translation unit
//!
//! `fattn-mma-f16.cuh` declares most `ggml_cuda_flash_attn_ext_mma_f16_case`
//! instantiations `extern`, so `fattn.cu` calls the instance file's copy.
//! The ones it does not declare -- every `DKQ = 192` case and `<576, 512, 2,
//! 4>` -- `fattn.cu` instantiates implicitly, and its own copy (with its own
//! `static` kernels) is the one its calls reach. `library` encodes that.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const fattn_common = @import("fattn_common.zig");
const table = @import("fattn_config_table.zig");
const fattn = @import("fattn.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// `sizeof(half2)`.
const sizeof_half2 = 4;

/// Ports `fattn_mma_config` (fattn-mma-f16.cuh:11 @5266f24da).
pub const Config = struct {
    /// Number of threads per CUDA block.
    nthreads: c_int,
    /// Targeted occupancy for the MMA kernel.
    occupancy: c_int,
    /// Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    nbatch_fa: c_int,
    /// Number of K half2 values in direction of DKQ to load in parallel.
    nbatch_K2: c_int,
    /// Number of V half2 values in direction of DV to load in parallel.
    nbatch_V2: c_int,
    /// Number of VKQ half2 values in direction of DV to combine in parallel.
    nbatch_combine: c_int,
    /// Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    nstages_target: c_int,
    /// Whether the Q values should be kept permanently in registers.
    Q_in_reg: bool,
};

/// A family's table: the first case matching `(DKQ, DV, ncols)`, else its
/// default or the table it falls through to.
fn lookup(t: *const table.MmaTable, DKQ: c_int, DV: c_int, ncols: c_int) Config {
    for (t.cases) |e| {
        if (DKQ == e.DKQ and DV == e.DV and ncols == e.ncols) return e.config;
    }
    if (t.default) |d| return d;
    return lookup(t.fallthrough.?, DKQ, DV, ncols);
}

/// Ports the host `ggml_cuda_fattn_mma_get_config` (fattn-mma-f16.cuh:232 @5266f24da).
/// The `amd_mfma_available` and `amd_wmma_available` arms, which pick the
/// CDNA and RDNA tables, are false for every CUDA compute capability; those
/// tables are not generated.
fn getConfig(DKQ: c_int, DV: c_int, ncols: c_int, cc: c_int) Config {
    if (common.ampereMmaAvailable(cc)) {
        return lookup(&table.mma_ampere, DKQ, DV, ncols);
    }
    if (common.turingMmaAvailable(cc)) {
        return lookup(&table.mma_turing, DKQ, DV, ncols);
    }
    impl.assert(common.voltaMmaAvailable(cc), "volta_mma_available(cc)");
    return lookup(&table.mma_volta, DKQ, DV, ncols);
}

/// Ports `get_cols_per_warp` (fattn-mma-f16.cuh:338 @5266f24da).
fn getColsPerWarp(cc: c_int) c_int {
    if (common.turingMmaAvailable(cc) or fattn_common.amdWmmaAvailable(cc) or fattn_common.amdMfmaAvailable(cc)) {
        return 16;
    } else {
        // Volta
        return 32;
    }
}

/// Ports the host `ggml_cuda_fattn_mma_get_nstages` (fattn-mma-f16.cuh:349 @5266f24da),
/// with `ggml_cuda_fattn_mma_get_nstages_target` (fattn-mma-f16.cuh:314 @5266f24da)
/// inlined.
fn getNstages(DKQ: c_int, DV: c_int, ncols1: c_int, ncols2: c_int, cc: c_int) c_int {
    return if (common.cpAsyncAvailable(cc) and ncols2 >= 2) getConfig(DKQ, DV, ncols1 * ncols2, cc).nstages_target else 0;
}

/// The translation unit that holds `ggml_cuda_flash_attn_ext_mma_f16_case
/// <DKQ, DV, ncols1, ncols2>` as `fattn.cu` reaches it: the instance file
/// for every case `fattn-mma-f16.cuh` declares `extern` -- its
/// `DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2` lines and the `extern` list after
/// them -- and `fattn` itself for the rest.
fn library(comptime DKQ: comptime_int, comptime DV: comptime_int, comptime ncols1: comptime_int, comptime ncols2: comptime_int) []const u8 {
    const all_ncols2_dkq = [_]comptime_int{ 64, 80, 96, 112, 128, 256 };
    const extern_list = [_][4]comptime_int{
        .{ 512, 512, 4, 2 },  .{ 512, 512, 8, 2 },  .{ 512, 512, 16, 2 }, .{ 512, 512, 32, 2 },
        .{ 512, 512, 2, 4 },  .{ 512, 512, 4, 4 },  .{ 512, 512, 8, 4 },  .{ 512, 512, 16, 4 },
        .{ 512, 512, 1, 8 },  .{ 512, 512, 2, 8 },  .{ 512, 512, 4, 8 },  .{ 512, 512, 8, 8 },
        .{ 576, 512, 1, 16 }, .{ 576, 512, 2, 16 }, .{ 576, 512, 4, 16 }, .{ 320, 256, 1, 32 },
        .{ 320, 256, 2, 32 }, .{ 576, 512, 4, 4 },  .{ 576, 512, 8, 4 },  .{ 576, 512, 16, 4 },
        .{ 576, 512, 1, 32 }, .{ 576, 512, 2, 32 },
    };
    const declared = blk: {
        // `DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)` for ncols in
        // {8, 16, 32, 64}: every `ncols2` in {1, 2, 4, 8, 16} with
        // `ncols1 = ncols / ncols2`.
        for (all_ncols2_dkq) |d| {
            if (DKQ == d and DV == d) {
                for ([_]comptime_int{ 8, 16, 32, 64 }) |ncols| {
                    for ([_]comptime_int{ 1, 2, 4, 8, 16 }) |n2| {
                        if (ncols1 == ncols / n2 and ncols2 == n2) break :blk true;
                    }
                }
            }
        }
        for (extern_list) |e| {
            if (DKQ == e[0] and DV == e[1] and ncols1 == e[2] and ncols2 == e[3]) break :blk true;
        }
        break :blk false;
    };
    return if (declared)
        std.fmt.comptimePrint("fattn-mma-f16-instance-ncols1_{d}-ncols2_{d}", .{ ncols1, ncols2 })
    else
        "fattn";
}

/// The host half of `ggml_cuda_fattn_smem_swizzle` (fattn-swizzle.cuh):
/// a K/V shared-memory tile whose row is a multiple of 32 `half2` is
/// XOR-swizzled on Turing and later instead of padded, so its stride drops
/// the `+ 4`. The kernels compute the same stride at compile time; the host
/// needs it to size the dynamic shared memory.
const smem_swizzle = struct {
    /// Ports `bank_aligned` (fattn-swizzle.cuh:11 @5266f24da).
    fn bankAligned(nbatch_2: c_int) bool {
        return nbatch_2 >= 32 and @rem(nbatch_2, 32) == 0;
    }

    /// Ports the host `enabled` (fattn-swizzle.cuh:24 @5266f24da), the
    /// non-HIP arm.
    fn enabled(nbatch_2: c_int, cc: c_int) bool {
        return common.turingMmaAvailable(cc) and bankAligned(nbatch_2);
    }

    /// Ports the host `tile_stride` (fattn-swizzle.cuh:38 @5266f24da).
    fn tileStride(nbatch_2: c_int, cc: c_int) c_int {
        return if (enabled(nbatch_2, cc)) nbatch_2 else nbatch_2 + 4;
    }
};

/// Ports `ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse` (fattn-mma-f16.cuh:1758 @5266f24da):
/// the `(DKQ, DV, ncols1, ncols2)` instantiations that have a sparse-gather
/// kernel, `use_sparse = true`. `fattn.zig`'s `ncols1` switch asks it too.
pub fn mayUseSparse(comptime DKQ: comptime_int, comptime DV: comptime_int, comptime ncols1: comptime_int, comptime ncols2: comptime_int) bool {
    return (DKQ == 512 and DV == 512 and ncols1 == 1 and ncols2 == 8) or
        (DKQ == 576 and DV == 512 and ncols1 == 1 and ncols2 == 16);
}

/// `flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse>`.
fn kernel(comptime DKQ: comptime_int, comptime DV: comptime_int, comptime ncols1: comptime_int, comptime ncols2: comptime_int, comptime use_logit_softcap: bool, comptime V_is_K_view: bool, comptime use_sparse: bool) kernels.Kernel {
    return kernels.get(library(DKQ, DV, ncols1, ncols2), kernels.tmpl("flash_attn_ext_f16", &.{
        kernels.int(DKQ),
        kernels.int(DV),
        kernels.int(ncols1),
        kernels.int(ncols2),
        kernels.boolean(use_logit_softcap),
        kernels.boolean(V_is_K_view),
        kernels.boolean(use_sparse),
    }));
}

/// Ports `ggml_cuda_flash_attn_ext_mma_f16_case` (fattn-mma-f16.cuh:1967 @5266f24da).
///
/// The C raises each kernel's dynamic shared-memory limit once per device
/// behind a function-local `static bool[GGML_CUDA_MAX_DEVICES]`, one per
/// instantiation, softcap and sparse branch; `kernels.setSharedMemoryLimit`
/// keeps the same once-per-(device, kernel) flag. The C's sparse arm
/// raises the limit under `!GGML_USE_MUSA` implicitly, being inside the
/// `!GGML_USE_HIP && !GGML_USE_MUSA` block; neither is defined here.
pub fn mmaF16Case(comptime DKQ: comptime_int, comptime DV: comptime_int, comptime ncols1: comptime_int, comptime ncols2: comptime_int, ctx: *common.Context, dst: *Tensor) void {
    const KQV = dst;
    const id = common.getDevice();
    const cc = common.info().devices[@intCast(id)].cc;

    const ncols = ncols1 * ncols2;

    const config = getConfig(DKQ, DV, ncols, cc);
    const nthreads = config.nthreads;
    const nbatch_fa = config.nbatch_fa;
    const nbatch_K2 = config.nbatch_K2;
    const nbatch_V2 = config.nbatch_V2;
    const nbatch_combine = config.nbatch_combine;
    const Q_in_reg = config.Q_in_reg;
    const nstages = getNstages(DKQ, DV, ncols1, ncols2, cc);

    const cols_per_warp = @min(ncols, getColsPerWarp(cc));
    const warp_size_host = common.info().devices[@intCast(ctx.device)].warp_size;
    const nwarps = @divTrunc(nthreads, warp_size_host);

    const V_is_K_view = DKQ == 576; // Guaranteed by the kernel selection logic in fattn.cu

    // KV tile strides must match flash_attn_ext_f16_iter / _process_tile.
    const stride_tile_K = smem_swizzle.tileStride(nbatch_K2, cc);
    const stride_tile_V = if (V_is_K_view) stride_tile_K else smem_swizzle.tileStride(nbatch_V2, cc);
    const nbytes_shared_KV_1stage: usize = @as(usize, @intCast(nbatch_fa * @max(stride_tile_K, stride_tile_V))) * sizeof_half2;
    const nbytes_shared_KV_2stage: usize = @as(usize, @intCast(nbatch_fa * (stride_tile_K + stride_tile_V))) * sizeof_half2;
    const nbytes_shared_Q: usize = @as(usize, ncols * (DKQ / 2 + 4)) * sizeof_half2;
    const nbytes_shared_mask: usize = @as(usize, @intCast(ncols1 * (@divTrunc(nbatch_fa, 2) + 4))) * sizeof_half2;
    const nbytes_shared_combine: usize = @as(usize, @intCast(nwarps * cols_per_warp * (nbatch_combine + 4))) * sizeof_half2;

    const nbytes_shared_KV = if (nstages <= 1) nbytes_shared_KV_1stage else nbytes_shared_KV_2stage;

    const nbytes_shared_total = @max(nbytes_shared_combine, if (Q_in_reg)
        @max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask)
    else
        nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    const logit_softcap = impl.getOpParamsF32(KQV, 2);

    var fattn_kernel: kernels.Kernel = undefined;
    var use_sparse = false;
    if (logit_softcap == 0.0) {
        const use_logit_softcap = false;
        if (comptime mayUseSparse(DKQ, DV, ncols1, ncols2)) {
            if (fattn.mmaF16ShallUseSparse(ctx, dst)) {
                const use_sparse_kernel = true;
                fattn_kernel = comptime kernel(DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel);
                use_sparse = true;
                kernels.setSharedMemoryLimit(fattn_kernel, nbytes_shared_total);
            } else {
                const use_sparse_kernel = false;
                fattn_kernel = comptime kernel(DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel);
                kernels.setSharedMemoryLimit(fattn_kernel, nbytes_shared_total);
            }
        } else {
            const use_sparse_kernel = false;
            fattn_kernel = comptime kernel(DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel);
            kernels.setSharedMemoryLimit(fattn_kernel, nbytes_shared_total);
        }
    } else {
        const use_logit_softcap = true;
        const use_sparse_kernel = false;
        fattn_kernel = comptime kernel(DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel);
        kernels.setSharedMemoryLimit(fattn_kernel, nbytes_shared_total);
    }

    fattn_common.launchFattn(comptime library(DKQ, DV, ncols1, ncols2), DV, ncols1, ncols2, ctx, dst, .{
        .kernel = fattn_kernel,
        .nwarps = nwarps,
        .nbytes_shared = nbytes_shared_total,
        .nbatch_fa = nbatch_fa,
        .need_f16_K = true,
        .need_f16_V = true,
        .stream_k = true,
        .use_sparse = use_sparse,
        .warp_size = warp_size_host,
    });
}
