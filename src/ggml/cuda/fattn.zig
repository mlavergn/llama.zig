//! `GGML_OP_FLASH_ATTN_EXT`: which of the three kernel families a node
//! takes, and the dispatch into each.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fattn.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/fattn.cuh` at v0.4.0 (`5266f24da`); the
//! sparse K/V path (`compactMask`, `mmaF16ShallUseSparse`) is new at v0.4.0
//! (`5266f24da`). The
//! families are in `fattn_mma.zig` (tensor cores), `fattn_tile.zig`
//! (generic) and `fattn_vec.zig` (one or two query columns); the launcher
//! they share is `fattn_common.zig`.
//!
//! # Configuration
//!
//! The reference build defines `GGML_CUDA_FA_ALL_QUANTS`
//! (`build/llamacpp.zig`, `cuda_defines`), so every K/V type pair has a
//! vector instance and K and V may differ. `FLASH_ATTN_AVAILABLE` is
//! defined: neither `GGML_CUDA_NO_FA` nor MUSA.
//!
//! # The scratch space
//!
//! A K or V that is not F16 is converted into the destination tensor's own
//! allocation, past its bytes: `getAllocSize` is what the buffer type
//! reserves, and it must be exactly the C's, because the launch writes up
//! to that end.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const fattn_common = @import("fattn_common.zig");
const fattn_mma = @import("fattn_mma.zig");
const fattn_tile = @import("fattn_tile.zig");
const fattn_vec = @import("fattn_vec.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// `GGML_CUDA_FA_ALL_QUANTS`, which the reference build defines.
const fa_all_quants = true;

const kq_stride = fattn_common.kq_stride;

/// `sizeof(half)`.
const sizeof_half = 2;

/// Ports `ggml_cuda_flash_attn_ext_compact_mask` (fattn.cu:92 @5266f24da),
/// the non-HIP, non-MUSA arm: for every mask row, the indices of its finite
/// entries, at most `n_kv_max` of them, padded with -1. The kernel,
/// `flash_attn_mask_to_sparse_indices` (fattn.cu:10 @5266f24da), stays
/// CUDA; it is in this translation unit whichever instance file's
/// `launch_fattn` asks for it, since the C's function is not a template.
pub fn compactMask(mask: *const Tensor, indices: ?[*]i32, n_kv_max: i32, stream: common.cu.cudaStream_t) void {
    const s31: i64 = @bitCast(mask.nb[1] / sizeof_half);
    const s33: i64 = @bitCast(mask.nb[3] / sizeof_half);
    // `dim3`'s members are `unsigned int`.
    const blocks_num: kernels.Dim3 = .{
        .x = @truncate(@as(u64, @bitCast(mask.ne[1]))),
        .y = @truncate(@as(u64, @bitCast(mask.ne[3]))),
        .z = 1,
    };
    const block_dim: kernels.Dim3 = .{ .x = 256, .y = 1, .z = 1 };
    kernels.launchPdl(comptime kernels.get("fattn", "flash_attn_mask_to_sparse_indices"), .{
        .grid = blocks_num,
        .block = block_dim,
        .shmem = 0,
        .stream = stream,
    }, .{ @as(?*const anyopaque, mask.data), indices, @as(c_int, @truncate(mask.ne[0])), n_kv_max, s31, s33 });
    common.cudaCheck(common.cu.cudaGetLastError(), "cudaGetLastError()", @src());
}

/// Ports `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` (fattn.cu:109 @5266f24da),
/// the non-HIP, non-MUSA arm: whether a node takes the sparse-gather
/// kernel, given an instantiation that has one (`fattn_mma.mayUseSparse`).
pub fn mmaF16ShallUseSparse(ctx: *common.Context, dst: *const Tensor) bool {
    const Q = impl.one(Tensor, dst.src[0]);
    const K = impl.one(Tensor, dst.src[1]);
    const mask_opt: ?*const Tensor = dst.src[3];
    const cc = common.info().devices[@intCast(ctx.device)].cc;

    const max_bias = impl.getOpParamsF32(dst, 1);
    const logit_softcap = impl.getOpParamsF32(dst, 2);

    const n_kv_max = impl.getOpParamsI32(dst, 4);
    const mask = mask_opt orelse return false;
    return common.cc.isNvidia(cc) and common.turingMmaAvailable(cc) and
        n_kv_max > 0 and max_bias == 0.0 and logit_softcap == 0.0 and
        mask.ne[0] == K.ne[1] and mask.ne[1] >= Q.ne[1] and mask.ne[2] == 1 and
        K.ne[1] >= @max(@as(i64, 4096), 2 * @as(i64, n_kv_max));
}

/// Ports `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` (fattn.cu:133 @5266f24da).
fn mmaF16SwitchNcols1(comptime DKQ: comptime_int, comptime DV: comptime_int, comptime ncols2: comptime_int, ctx: *common.Context, dst: *Tensor) void {
    const cc = common.info().devices[@intCast(common.getDevice())].cc;
    const Q = impl.one(Tensor, dst.src[0]);

    if (comptime fattn_mma.mayUseSparse(DKQ, DV, 1, ncols2)) {
        if (mmaF16ShallUseSparse(ctx, dst)) {
            fattn_mma.mmaF16Case(DKQ, DV, 1, ncols2, ctx, dst);
            return;
        }
    }

    if (ncols2 <= 8) {
        if (common.turingMmaAvailable(cc) and Q.ne[1] <= 8 / ncols2) {
            fattn_mma.mmaF16Case(DKQ, DV, 8 / ncols2, ncols2, ctx, dst);
            return;
        }
    }

    if (ncols2 <= 16) {
        if (Q.ne[1] <= 16 / ncols2) {
            fattn_mma.mmaF16Case(DKQ, DV, 16 / ncols2, ncols2, ctx, dst);
            return;
        }
    }

    if (Q.ne[1] <= 32 / ncols2 or (common.cc.isNvidia(cc) and common.highestCompiledArch(cc) == common.cc.turing) or
        (common.cc.isAmd(cc) and DKQ > 256))
    {
        fattn_mma.mmaF16Case(DKQ, DV, 32 / ncols2, ncols2, ctx, dst);
        return;
    }

    fattn_mma.mmaF16Case(DKQ, DV, 64 / ncols2, ncols2, ctx, dst);
}

/// `use_gqa_opt` as `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2` and
/// `ggml_cuda_get_best_fattn_kernel` both compute it: no GQA optimization
/// for a node with no mask, ALiBi, an unpadded K/V, or a non-quantized
/// operand with a row stride not a multiple of 16 bytes.
fn gqaOpt(base: bool, Q: *const Tensor, K: *const Tensor, V: *const Tensor, mask: ?*const Tensor) bool {
    var use_gqa_opt = base;
    for ([_]?*const Tensor{ Q, K, V, mask }) |t_opt| {
        const t = t_opt orelse continue;
        if (c.ggml_is_quantized(t.type)) {
            continue;
        }
        for (1..c.GGML_MAX_DIMS) |i| {
            if (t.nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }
    return use_gqa_opt;
}

/// Ports `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2` (fattn.cu:170 @5266f24da).
fn mmaF16SwitchNcols2(comptime DKQ: comptime_int, comptime DV: comptime_int, ctx: *common.Context, dst: *Tensor) void {
    const cc = common.info().devices[@intCast(common.getDevice())].cc;
    const KQV = dst;
    const Q = impl.one(Tensor, dst.src[0]);
    const K = impl.one(Tensor, dst.src[1]);
    const V = impl.one(Tensor, dst.src[2]);
    const mask: ?*const Tensor = dst.src[3];

    const max_bias = impl.getOpParamsF32(KQV, 1);

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    const use_gqa_opt = gqaOpt(mask != null and max_bias == 0.0 and @rem(K.ne[1], kq_stride) == 0, Q, K, V, mask);

    impl.assert(@rem(Q.ne[2], K.ne[2]) == 0, "Q->ne[2] % K->ne[2] == 0");
    const gqa_ratio: c_int = @truncate(@divTrunc(Q.ne[2], K.ne[2]));

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == common.cc.volta) {
        if (use_gqa_opt and @rem(gqa_ratio, 8) == 0) {
            mmaF16SwitchNcols1(DKQ, DV, 8, ctx, dst);
            return;
        }

        if (use_gqa_opt and @rem(gqa_ratio, 4) == 0) {
            mmaF16SwitchNcols1(DKQ, DV, 4, ctx, dst);
            return;
        }

        if (DKQ <= 256) {
            if (use_gqa_opt and @rem(gqa_ratio, 2) == 0) {
                mmaF16SwitchNcols1(DKQ, DV, 2, ctx, dst);
                return;
            }

            mmaF16SwitchNcols1(DKQ, DV, 1, ctx, dst);
            return;
        } else {
            impl.abort("fatal error");
        }
    }

    if (use_gqa_opt and gqa_ratio > 4) {
        mmaF16SwitchNcols1(DKQ, DV, 8, ctx, dst);
        return;
    }

    if (use_gqa_opt and gqa_ratio > 2) {
        mmaF16SwitchNcols1(DKQ, DV, 4, ctx, dst);
        return;
    }

    if (use_gqa_opt and gqa_ratio > 1) {
        mmaF16SwitchNcols1(DKQ, DV, 2, ctx, dst);
        return;
    }

    if (DKQ <= 256) {
        mmaF16SwitchNcols1(DKQ, DV, 1, ctx, dst);
    } else {
        impl.abort("fatal error");
    }
}

/// Ports `ggml_cuda_flash_attn_ext_mma_f16` (fattn.cu:246 @5266f24da).
fn mmaF16(ctx: *common.Context, dst: *Tensor) void {
    const cc = common.info().devices[@intCast(common.getDevice())].cc;
    const KQV = dst;
    const Q = impl.one(Tensor, dst.src[0]);
    const K = impl.one(Tensor, dst.src[1]);
    const V = impl.one(Tensor, dst.src[2]);
    const mask: ?*const Tensor = dst.src[3];

    switch (Q.ne[0]) {
        64 => {
            impl.assert(V.ne[0] == 64, "V->ne[0] == 64");
            mmaF16SwitchNcols2(64, 64, ctx, dst);
        },
        80 => {
            impl.assert(V.ne[0] == 80, "V->ne[0] == 80");
            mmaF16SwitchNcols2(80, 80, ctx, dst);
        },
        96 => {
            impl.assert(V.ne[0] == 96, "V->ne[0] == 96");
            mmaF16SwitchNcols2(96, 96, ctx, dst);
        },
        112 => {
            impl.assert(V.ne[0] == 112, "V->ne[0] == 112");
            mmaF16SwitchNcols2(112, 112, ctx, dst);
        },
        128 => {
            impl.assert(V.ne[0] == 128, "V->ne[0] == 128");
            mmaF16SwitchNcols2(128, 128, ctx, dst);
        },
        192 => {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            impl.assert(V.ne[0] == 128, "V->ne[0] == 128");
            const max_bias = impl.getOpParamsF32(KQV, 1);
            const use_gqa_opt = mask != null and max_bias == 0.0;
            impl.assert(use_gqa_opt, "use_gqa_opt");
            impl.assert(@rem(Q.ne[2], K.ne[2]) == 0, "Q->ne[2] % K->ne[2] == 0");
            const gqa_ratio: c_int = @truncate(@divTrunc(Q.ne[2], K.ne[2]));
            if (@rem(gqa_ratio, 16) == 0) {
                mmaF16SwitchNcols1(192, 128, 16, ctx, dst);
            } else {
                impl.assert(@rem(gqa_ratio, 8) == 0, "gqa_ratio % 8 == 0");
                mmaF16SwitchNcols1(192, 128, 8, ctx, dst);
            }
        },
        256 => {
            impl.assert(V.ne[0] == 256, "V->ne[0] == 256");
            mmaF16SwitchNcols2(256, 256, ctx, dst);
        },
        320 => {
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            impl.assert(V.ne[0] == 256, "V->ne[0] == 256");
            const max_bias = impl.getOpParamsF32(KQV, 1);

            const use_gqa_opt = mask != null and max_bias == 0.0;
            impl.assert(use_gqa_opt, "use_gqa_opt");
            impl.assert(@rem(Q.ne[2], K.ne[2]) == 0, "Q->ne[2] % K->ne[2] == 0");
            const gqa_ratio: c_int = @truncate(@divTrunc(Q.ne[2], K.ne[2]));
            impl.assert(@rem(gqa_ratio, 32) == 0, "gqa_ratio % 32 == 0");

            mmaF16SwitchNcols1(320, 256, 32, ctx, dst);
        },
        512 => {
            impl.assert(V.ne[0] == 512, "V->ne[0] == 512");
            mmaF16SwitchNcols2(512, 512, ctx, dst);
        },
        576 => {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            impl.assert(V.ne[0] == 512, "V->ne[0] == 512");
            const max_bias = impl.getOpParamsF32(KQV, 1);

            const use_gqa_opt = mask != null and max_bias == 0.0;
            impl.assert(use_gqa_opt, "use_gqa_opt");

            impl.assert(@rem(Q.ne[2], K.ne[2]) == 0, "Q->ne[2] % K->ne[2] == 0");
            const gqa_ratio: c_int = @truncate(@divTrunc(Q.ne[2], K.ne[2]));
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= common.cc.dgx_spark) {
                    if (Q.ne[1] <= 8) {
                        mmaF16SwitchNcols1(576, 512, 16, ctx, dst);
                        return;
                    }
                    mmaF16SwitchNcols1(576, 512, 4, ctx, dst);
                    return;
                }
                if (cc >= common.cc.blackwell) {
                    if (Q.ne[1] <= 4 and K.ne[1] >= 65536) {
                        mmaF16SwitchNcols1(576, 512, 16, ctx, dst);
                        return;
                    }
                    mmaF16SwitchNcols1(576, 512, 4, ctx, dst);
                    return;
                }
                if (cc >= common.cc.ada_lovelace) {
                    if (Q.ne[1] <= 4) {
                        mmaF16SwitchNcols1(576, 512, 16, ctx, dst);
                        return;
                    }
                    mmaF16SwitchNcols1(576, 512, 4, ctx, dst);
                    return;
                }
                if (cc >= common.cc.turing) {
                    if (Q.ne[1] <= 4) {
                        if (K.ne[1] <= 16384) {
                            mmaF16SwitchNcols1(576, 512, 16, ctx, dst);
                            return;
                        }
                        mmaF16SwitchNcols1(576, 512, 32, ctx, dst);
                        return;
                    }
                    mmaF16SwitchNcols1(576, 512, 4, ctx, dst);
                    return;
                }
                // Volta:
                mmaF16SwitchNcols1(576, 512, 4, ctx, dst);
            } else if (@rem(gqa_ratio, 16) == 0) {
                mmaF16SwitchNcols1(576, 512, 16, ctx, dst);
            } else {
                mmaF16SwitchNcols1(576, 512, 4, ctx, dst);
            }
        },
        else => {
            impl.abort("fatal error");
        },
    }
}

/// The K/V type pairs `ggml_cuda_flash_attn_ext_vec` tries, in its order:
/// the `GGML_CUDA_FA_ALL_QUANTS` arm, every V type outermost, every K type
/// within.
const vec_types = [_]c.enum_ggml_type{
    c.GGML_TYPE_F16,
    c.GGML_TYPE_Q4_0,
    c.GGML_TYPE_Q4_1,
    c.GGML_TYPE_Q5_0,
    c.GGML_TYPE_Q5_1,
    c.GGML_TYPE_Q8_0,
    c.GGML_TYPE_BF16,
};

/// Ports `ggml_cuda_flash_attn_ext_vec` (fattn.cu:392 @5266f24da), with
/// `FATTN_VEC_CASE` (fattn.cu:377 @5266f24da) unrolled over the
/// `GGML_CUDA_FA_ALL_QUANTS` list and `FATTN_VEC_CASES_ALL_D`'s head sizes.
fn vec(ctx: *common.Context, dst: *Tensor) void {
    comptime std.debug.assert(fa_all_quants);
    const Q = impl.one(Tensor, dst.src[0]);
    const K = impl.one(Tensor, dst.src[1]);
    const V = impl.one(Tensor, dst.src[2]);

    inline for (vec_types) |type_V| {
        inline for (vec_types) |type_K| {
            inline for (.{ 64, 128, 256 }) |D| {
                const type_K_okay = K.type == type_K or (K.type == c.GGML_TYPE_F32 and type_K == c.GGML_TYPE_F16);
                const type_V_okay = V.type == type_V or (V.type == c.GGML_TYPE_F32 and type_V == c.GGML_TYPE_F16);
                if (Q.ne[0] == D and type_K_okay and type_V_okay) {
                    fattn_vec.vecCase(D, type_K, type_V, ctx, dst);
                    return;
                }
            }
        }
    }

    impl.abort("fatal error");
}

/// Ports `best_fattn_kernel` (fattn.cu:464 @5266f24da): the best
/// FlashAttention kernel for a specific GPU.
const BestKernel = enum(c_int) {
    none = 0,
    tile = 200,
    vec = 100,
    mma_f16 = 400,
};

/// Ports `ggml_cuda_fattn_kv_type_supported` (fattn.cu:471 @5266f24da).
fn kvTypeSupported(t: c.enum_ggml_type) bool {
    return switch (t) {
        c.GGML_TYPE_F32, c.GGML_TYPE_F16 => true,
        c.GGML_TYPE_Q4_1, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q5_1 => fa_all_quants,
        c.GGML_TYPE_Q4_0, c.GGML_TYPE_Q8_0, c.GGML_TYPE_BF16 => true,
        else => false,
    };
}

/// Ports `ggml_cuda_get_best_fattn_kernel` (fattn.cu:491 @5266f24da).
fn getBestKernel(device: c_int, dst: *const Tensor) BestKernel {
    const KQV = dst;
    const Q = impl.one(Tensor, dst.src[0]);
    const K = impl.one(Tensor, dst.src[1]);
    const V = impl.one(Tensor, dst.src[2]);
    const mask: ?*const Tensor = dst.src[3];

    const gqa_ratio: c_int = @truncate(@divTrunc(Q.ne[2], K.ne[2]));
    impl.assert(@rem(Q.ne[2], K.ne[2]) == 0, "Q->ne[2] % K->ne[2] == 0");

    const max_bias = impl.getOpParamsF32(KQV, 1);

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    const gqa_opt_applies = gqaOpt(gqa_ratio >= 2 and mask != null and max_bias == 0.0 and @rem(K.ne[1], kq_stride) == 0, Q, K, V, mask);

    const cc = common.info().devices[@intCast(device)].cc;

    switch (K.ne[0]) {
        40, 64, 72, 80, 96, 128, 112, 256 => {
            if (V.ne[0] != K.ne[0]) {
                return .none;
            }
        },
        192 => {
            if (V.ne[0] != 128 or !gqa_opt_applies) {
                return .none;
            }
            if (@rem(gqa_ratio, 8) != 0) {
                return .none;
            }
        },
        320 => {
            if (V.ne[0] != 256 or !gqa_opt_applies) {
                return .none;
            }
            if (@rem(gqa_ratio, 32) != 0) {
                return .none;
            }
        },
        512 => {
            if (V.ne[0] != K.ne[0]) {
                return .none;
            }
            if (!gqa_opt_applies) {
                return .none;
            }
        },
        576 => {
            if (V.ne[0] != 512) {
                return .none;
            }
            if (!gqa_opt_applies) {
                return .none;
            }
        },
        else => return .none,
    }

    if (!fa_all_quants) {
        if (K.type != V.type) {
            return .none;
        }
    }

    if (!kvTypeSupported(K.type) or !kvTypeSupported(V.type)) {
        return .none;
    }

    if (mask != null and mask.?.ne[2] != 1) {
        return .none;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const can_use_vector_kernel = Q.ne[0] <= 256 and @rem(Q.ne[0], 64) == 0 and Q.ne[0] != 192 and @rem(K.ne[1], kq_stride) == 0;

    // If Turing tensor cores are available, use them:
    if (common.turingMmaAvailable(cc) and Q.ne[0] != 40 and Q.ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!c.ggml_is_quantized(K.type) and !c.ggml_is_quantized(V.type)) {
                if (cc >= common.cc.ada_lovelace and Q.ne[1] == 1 and Q.ne[3] == 1 and !(gqa_ratio > 4 and K.ne[1] >= 8192)) {
                    return .vec;
                }
            } else {
                if (cc >= common.cc.ada_lovelace) {
                    if (Q.ne[1] <= 2) {
                        return .vec;
                    }
                } else {
                    if (Q.ne[1] == 1) {
                        return .vec;
                    }
                }
            }
            if (!gqa_opt_applies and Q.ne[1] == 1) {
                return .vec;
            }
        }
        return .mma_f16;
    }

    const ncols2_max: c_int = if (Q.ne[0] == 320) 32 else if (Q.ne[0] == 576 or Q.ne[0] == 192) 16 else 8;
    var gqa_ratio_eff: c_int = 1;
    while (@rem(gqa_ratio, 2 * gqa_ratio_eff) == 0 and gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (common.voltaMmaAvailable(cc) and Q.ne[0] != 40 and Q.ne[0] != 72) {
        if (can_use_vector_kernel and Q.ne[1] * gqa_ratio_eff <= 2) {
            return .vec;
        }
        if (Q.ne[1] * gqa_ratio_eff <= 16) {
            return .tile; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return .mma_f16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((fattn_common.amdMfmaAvailable(cc) and Q.ne[0] <= 256) and Q.ne[0] != 40 and Q.ne[0] != 72) {
        if ((Q.ne[0] <= 64 and Q.ne[1] * gqa_ratio_eff > 8)) {
            return .mma_f16;
        }
        if ((Q.ne[0] <= 128 and Q.ne[1] * gqa_ratio_eff > 16)) {
            return .mma_f16;
        }
        if ((Q.ne[0] <= 256 and Q.ne[1] * gqa_ratio_eff > 64)) {
            return .mma_f16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((fattn_common.amdWmmaAvailable(cc) and gqa_opt_applies and Q.ne[0] <= 128) and Q.ne[0] != 40 and Q.ne[0] != 72 and Q.ne[1] * gqa_ratio_eff > 8) {
        return .mma_f16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!c.ggml_is_quantized(K.type) and !c.ggml_is_quantized(V.type)) {
            if (Q.ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return .vec;
                }
            }
        } else {
            if (Q.ne[1] <= 2) {
                return .vec;
            }
        }
    }
    return .tile;
}

/// Ports `ggml_cuda_flash_attn_ext_get_alloc_size` (fattn.cu:669 @5266f24da):
/// `dst`'s bytes plus room for the F16 copies of K and V the chosen kernel
/// needs.
///
/// Parameters:
/// - `device`: the buffer type's device.
/// - `dst`: a `GGML_OP_FLASH_ATTN_EXT` node.
///
/// Return: the allocation size in bytes.
pub fn getAllocSize(device: c_int, dst: *const Tensor) usize {
    impl.assert(dst.op == c.GGML_OP_FLASH_ATTN_EXT, "dst->op == GGML_OP_FLASH_ATTN_EXT");

    const K: ?*const Tensor = dst.src[1];
    const V: ?*const Tensor = dst.src[2];

    impl.assert(K != null, "K != nullptr");
    impl.assert(V != null, "V != nullptr");

    const kernel = getBestKernel(device, dst);

    var need_f16_K = false;
    var need_f16_V = false;

    switch (kernel) {
        .tile, .mma_f16 => {
            need_f16_K = true;
            need_f16_V = true;
        },
        .vec => {
            need_f16_K = K.?.type == c.GGML_TYPE_F32;
            need_f16_V = V.?.type == c.GGML_TYPE_F32;
        },
        .none => {},
    }

    const f16_extra = fattn_common.getF16ExtraData(dst, need_f16_K, need_f16_V);

    return f16_extra.end -% @intFromPtr(dst.data);
}

/// Ports `ggml_cuda_flash_attn_ext` (fattn.cu:703 @5266f24da).
pub fn flashAttnExt(ctx: *common.Context, dst: *Tensor) void {
    common.setDevice(ctx.device);
    switch (getBestKernel(common.getDevice(), dst)) {
        .none => impl.abort("fatal error"),
        .tile => fattn_tile.flashAttnExtTile(ctx, dst),
        .vec => vec(ctx, dst),
        .mma_f16 => mmaF16(ctx, dst),
    }
}

/// Ports `ggml_cuda_flash_attn_ext_supported` (fattn.cu:720 @5266f24da).
pub fn supported(device: c_int, dst: *const Tensor) bool {
    return getBestKernel(device, dst) != .none;
}
