//! The generic flash-attention kernel, `flash_attn_tile`, for GPUs and head
//! sizes the tensor-core kernel does not serve.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fattn-tile.cu` and the host
//! half of `llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh` at v0.3.0
//! (`c1d0e7a00`), the non-HIP arms. The configuration tables are generated
//! into `fattn_config_table.zig` by `scripts/gen-fattn-config`. The kernel
//! stays CUDA, one instantiation set per head size in
//! `template-instances/fattn-tile-instance-dkq<DKQ>-dv<DV>.cu`, which every
//! `ggml_cuda_flash_attn_ext_tile_case` is declared `extern` against.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const fattn_common = @import("fattn_common.zig");
const table = @import("fattn_config_table.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// The packed configuration of one table entry, as
/// `GGML_CUDA_FATTN_TILE_CONFIG_CASE` (fattn-tile.cuh:12 @c1d0e7a00) returns it.
fn pack(e: table.TileCase) u32 {
    return (e.nthreads << 0) | (e.occupancy << 10) | (e.nbatch_fa << 14) | (e.nbatch_K << 23);
}

/// A family's table: the first case matching `(DKQ, DV, ncols)`, else 0.
fn lookup(cases: []const table.TileCase, DKQ: c_int, DV: c_int, ncols: c_int) u32 {
    for (cases) |e| {
        if (DKQ == e.DKQ and DV == e.DV and ncols == e.ncols) return pack(e);
    }
    return 0;
}

/// Ports the host `ggml_cuda_fattn_tile_get_config` (fattn-tile.cuh:315 @c1d0e7a00).
/// The AMD arm picks the `amd` or `amd_rdna` table for an AMD compute
/// capability, which a CUDA device never has; those tables are not
/// generated, and that arm aborts instead.
fn getConfig(DKQ: c_int, DV: c_int, ncols: c_int, cc: c_int) u32 {
    if (common.cc.isAmd(cc)) {
        impl.abort("ggml_cuda_fattn_tile_get_config: the AMD tables are not ported");
    }
    if (common.fastFp16Available(cc)) {
        return lookup(&table.tile_nvidia_fp16, DKQ, DV, ncols);
    }
    return lookup(&table.tile_nvidia_fp32, DKQ, DV, ncols);
}

/// Ports the host `ggml_cuda_fattn_tile_get_nthreads` (fattn-tile.cuh:344 @c1d0e7a00).
fn getNthreads(DKQ: c_int, DV: c_int, ncols: c_int, cc: c_int) c_int {
    return @intCast((getConfig(DKQ, DV, ncols, cc) >> 0) & ((1 << 10) - 1));
}

/// Ports the host `ggml_cuda_fattn_tile_get_occupancy` (fattn-tile.cuh:352 @c1d0e7a00).
/// The kernel reads its own copy; the host has no caller, as in the C.
fn getOccupancy(DKQ: c_int, DV: c_int, ncols: c_int, cc: c_int) c_int {
    return @intCast((getConfig(DKQ, DV, ncols, cc) >> 10) & ((1 << 4) - 1));
}

/// Ports the host `ggml_cuda_fattn_tile_get_nbatch_fa` (fattn-tile.cuh:360 @c1d0e7a00).
fn getNbatchFa(DKQ: c_int, DV: c_int, ncols: c_int, cc: c_int) c_int {
    return @intCast((getConfig(DKQ, DV, ncols, cc) >> 14) & ((1 << 9) - 1));
}

/// Ports the host `ggml_cuda_fattn_tile_get_nbatch_K` (fattn-tile.cuh:368 @c1d0e7a00).
/// The kernel reads its own copy; the host has no caller, as in the C.
fn getNbatchK(DKQ: c_int, DV: c_int, ncols: c_int, cc: c_int) c_int {
    return @intCast((getConfig(DKQ, DV, ncols, cc) >> 23) & ((1 << 9) - 1));
}

comptime {
    _ = &getOccupancy;
    _ = &getNbatchK;
}

/// One arm of `launch_fattn_tile_switch_ncols1`: the kernel for
/// `cols_per_block` columns, launched through `launch_fattn`.
fn launchCols(comptime DKQ: comptime_int, comptime DV: comptime_int, comptime ncols2: comptime_int, comptime use_logit_softcap: bool, comptime cols_per_block: comptime_int, ctx: *common.Context, dst: *Tensor, cc: c_int, warp_size: c_int) void {
    const library = comptime std.fmt.comptimePrint("fattn-tile-instance-dkq{d}-dv{d}", .{ DKQ, DV });
    const nbytes_shared: usize = 0;
    const nwarps = @divTrunc(getNthreads(DKQ, DV, cols_per_block, cc), warp_size);
    const nbatch_fa = getNbatchFa(DKQ, DV, cols_per_block, cc);
    const fattn_kernel = comptime kernels.get(library, kernels.tmpl("flash_attn_tile", &.{
        kernels.int(DKQ),
        kernels.int(DV),
        kernels.int(cols_per_block / ncols2),
        kernels.int(ncols2),
        kernels.boolean(use_logit_softcap),
    }));
    fattn_common.launchFattn(library, DV, cols_per_block / ncols2, ncols2, ctx, dst, .{
        .kernel = fattn_kernel,
        .nwarps = nwarps,
        .nbytes_shared = nbytes_shared,
        .nbatch_fa = nbatch_fa,
        .need_f16_K = true,
        .need_f16_V = true,
        .stream_k = false,
        .warp_size = warp_size,
    });
}

/// Ports `launch_fattn_tile_switch_ncols1` (fattn-tile.cuh:1149 @c1d0e7a00),
/// the non-HIP arm: no 64-column variant.
fn switchNcols1(comptime DKQ: comptime_int, comptime DV: comptime_int, comptime ncols2: comptime_int, comptime use_logit_softcap: bool, ctx: *common.Context, dst: *Tensor) void {
    const Q = impl.one(Tensor, dst.src[0]);

    const id = common.getDevice();
    const cc = common.info().devices[@intCast(id)].cc;
    const warp_size: c_int = 32;

    if (DKQ <= 256) {
        if (Q.ne[1] > 16 / ncols2) {
            launchCols(DKQ, DV, ncols2, use_logit_softcap, 32, ctx, dst, cc, warp_size);
            return;
        }
    }

    if (ncols2 <= 16) {
        if (Q.ne[1] > 8 / ncols2) {
            launchCols(DKQ, DV, ncols2, use_logit_softcap, 16, ctx, dst, cc, warp_size);
            return;
        }
    }

    if (ncols2 <= 8) {
        if (Q.ne[1] > 4 / ncols2) {
            launchCols(DKQ, DV, ncols2, use_logit_softcap, 8, ctx, dst, cc, warp_size);
            return;
        }
    }

    if (ncols2 <= 4) {
        if (Q.ne[1] > 2 / ncols2) {
            launchCols(DKQ, DV, ncols2, use_logit_softcap, 4, ctx, dst, cc, warp_size);
            return;
        }
    }

    if (ncols2 <= 2) {
        launchCols(DKQ, DV, ncols2, use_logit_softcap, 2, ctx, dst, cc, warp_size);
        return;
    }

    impl.abort("fatal error");
}

/// Ports `launch_fattn_tile_switch_ncols2` (fattn-tile.cuh:1237 @c1d0e7a00),
/// the non-HIP arm.
fn switchNcols2(comptime DKQ: comptime_int, comptime DV: comptime_int, comptime use_logit_softcap: bool, ctx: *common.Context, dst: *Tensor) void {
    const KQV = dst;
    const Q = impl.one(Tensor, dst.src[0]);
    const K = impl.one(Tensor, dst.src[1]);
    const mask: ?*const Tensor = dst.src[3];

    const max_bias = impl.getOpParamsF32(KQV, 1);

    impl.assert(@rem(Q.ne[2], K.ne[2]) == 0, "Q->ne[2] % K->ne[2] == 0");
    const gqa_ratio: c_int = @truncate(@divTrunc(Q.ne[2], K.ne[2]));

    // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
    // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
    const nvidia = common.cc.isNvidia(common.info().devices[@intCast(common.getDevice())].cc);
    const gqa_limit: c_int = if (nvidia and gqa_ratio <= 4 and DV <= 256) 16 else std.math.maxInt(c_int);
    const use_gqa_opt = mask != null and max_bias == 0.0 and Q.ne[1] <= gqa_limit and @rem(K.ne[1], fattn_common.kq_stride) == 0;

    if (DKQ == 320) {
        // This branch is only used for Mistral Small 4 which has a GQA ratio of 32.
        // On AMD, simply use that GQA ratio with 32 columns / block since we always have enough SRAM.
        // On NVIDIA however, the tile kernel is only used for GPUs that can't use the mma kernel (Pascal and older).
        // Therefore, use a GQA ratio of 16 with 16 columns / block to stay below 48 kiB of SRAM / block.
        if (use_gqa_opt and @rem(gqa_ratio, 16) == 0) {
            switchNcols1(DKQ, DV, 16, use_logit_softcap, ctx, dst);
            return;
        }
        impl.abort("flash-attn tile (320/256): expected GQA ratio multiple of 32");
    }

    if (DKQ == 576) {
        if (use_gqa_opt and @rem(gqa_ratio, 16) == 0) {
            switchNcols1(DKQ, DV, 16, use_logit_softcap, ctx, dst);
            return;
        }
        if (use_gqa_opt and @rem(gqa_ratio, 4) == 0) {
            switchNcols1(DKQ, DV, 4, use_logit_softcap, ctx, dst);
            return;
        }
    }

    if (DKQ == 192) {
        // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
        if (use_gqa_opt and @rem(gqa_ratio, 16) == 0) {
            switchNcols1(DKQ, DV, 16, use_logit_softcap, ctx, dst);
            return;
        }
        if (use_gqa_opt and @rem(gqa_ratio, 8) == 0) {
            switchNcols1(DKQ, DV, 8, use_logit_softcap, ctx, dst);
            return;
        }
        impl.abort("flash-attn tile (192/128): expected GQA ratio multiple of 8");
    }

    if (DKQ <= 512 and DKQ != 320 and DKQ != 192) {
        if (use_gqa_opt and @rem(gqa_ratio, 8) == 0) {
            switchNcols1(DKQ, DV, 8, use_logit_softcap, ctx, dst);
            return;
        }

        if (use_gqa_opt and @rem(gqa_ratio, 4) == 0) {
            switchNcols1(DKQ, DV, 4, use_logit_softcap, ctx, dst);
            return;
        }

        if (use_gqa_opt and @rem(gqa_ratio, 2) == 0) {
            switchNcols1(DKQ, DV, 2, use_logit_softcap, ctx, dst);
            return;
        }

        if (DV <= 256) {
            switchNcols1(DKQ, DV, 1, use_logit_softcap, ctx, dst);
            return;
        }
    }
    impl.abort("fatal error");
}

/// Ports `ggml_cuda_flash_attn_ext_tile_case` (fattn-tile.cuh:1323 @c1d0e7a00).
fn tileCase(comptime DKQ: comptime_int, comptime DV: comptime_int, ctx: *common.Context, dst: *Tensor) void {
    const KQV = dst;

    const logit_softcap = impl.getOpParamsF32(KQV, 2);

    if (logit_softcap == 0.0) {
        const use_logit_softcap = false;
        switchNcols2(DKQ, DV, use_logit_softcap, ctx, dst);
    } else {
        const use_logit_softcap = true;
        switchNcols2(DKQ, DV, use_logit_softcap, ctx, dst);
    }
}

/// Ports `ggml_cuda_flash_attn_ext_tile` (fattn-tile.cu:4 @c1d0e7a00).
pub fn flashAttnExtTile(ctx: *common.Context, dst: *Tensor) void {
    const K = impl.one(Tensor, dst.src[1]);
    const V = impl.one(Tensor, dst.src[2]);
    switch (K.ne[0]) {
        40 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(40, 40, ctx, dst);
        },
        64 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(64, 64, ctx, dst);
        },
        72 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(72, 72, ctx, dst);
        },
        80 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(80, 80, ctx, dst);
        },
        96 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(96, 96, ctx, dst);
        },
        112 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(112, 112, ctx, dst);
        },
        128 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(128, 128, ctx, dst);
        },
        192 => {
            impl.assert(V.ne[0] == 128, "V->ne[0] == 128");
            tileCase(192, 128, ctx, dst);
        },
        256 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(256, 256, ctx, dst);
        },
        320 => {
            impl.assert(V.ne[0] == 256, "V->ne[0] == 256");
            tileCase(320, 256, ctx, dst);
        },
        512 => {
            impl.assert(V.ne[0] == K.ne[0], "V->ne[0] == K->ne[0]");
            tileCase(512, 512, ctx, dst);
        },
        576 => {
            impl.assert(V.ne[0] == 512, "V->ne[0] == 512");
            tileCase(576, 512, ctx, dst);
        },
        else => {
            impl.abort("Unsupported head size");
        },
    }
}
