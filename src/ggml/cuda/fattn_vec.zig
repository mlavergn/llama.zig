//! The vector flash-attention kernel, `flash_attn_ext_vec`, for one or two
//! query columns: decoding.
//!
//! # Provenance
//!
//! Ported from the host half of
//! `llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` at v0.4.1 (`b29c606e2`).
//! The kernel stays CUDA, one instantiation set per (K type, V type) in
//! `template-instances/fattn-vec-instance-<K>-<V>.cu`, which every
//! `ggml_cuda_flash_attn_ext_vec_case` is declared `extern` against.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const fattn_common = @import("fattn_common.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `ggml_cuda_fattn_vec_get_nthreads_host` (fattn-vec.cuh:4 @b29c606e2).
fn getNthreadsHost(cc: c_int) c_int {
    _ = cc;
    return 128;
}

/// The instance file's spelling of a K or V type.
fn typeStem(comptime t: c.enum_ggml_type) []const u8 {
    return switch (t) {
        c.GGML_TYPE_F16 => "f16",
        c.GGML_TYPE_Q4_0 => "q4_0",
        c.GGML_TYPE_Q4_1 => "q4_1",
        c.GGML_TYPE_Q5_0 => "q5_0",
        c.GGML_TYPE_Q5_1 => "q5_1",
        c.GGML_TYPE_Q8_0 => "q8_0",
        c.GGML_TYPE_BF16 => "bf16",
        else => @compileError("no fattn-vec instance for this type"),
    };
}

/// Ports `ggml_cuda_flash_attn_ext_vec_case_impl` (fattn-vec.cuh:532 @b29c606e2).
fn vecCaseImpl(comptime D: comptime_int, comptime cols_per_block: comptime_int, comptime type_K: c.enum_ggml_type, comptime type_V: c.enum_ggml_type, comptime use_logit_softcap: bool, ctx: *common.Context, dst: *Tensor) void {
    const library = comptime std.fmt.comptimePrint("fattn-vec-instance-{s}-{s}", .{ typeStem(type_K), typeStem(type_V) });
    const cc = common.info().devices[@intCast(common.getDevice())].cc;

    const nthreads = getNthreadsHost(cc);
    const nwarps = @divTrunc(nthreads, common.warp_size);
    const fattn_kernel = comptime kernels.get(library, kernels.tmpl("flash_attn_ext_vec", &.{
        kernels.int(D),
        kernels.int(cols_per_block),
        kernels.ggmlType(type_K),
        kernels.ggmlType(type_V),
        kernels.boolean(use_logit_softcap),
    }));
    const need_f16_K = type_K == c.GGML_TYPE_F16;
    const need_f16_V = type_V == c.GGML_TYPE_F16;
    const nbytes_shared: usize = 0;
    fattn_common.launchFattn(library, D, cols_per_block, 1, ctx, dst, .{
        .kernel = fattn_kernel,
        .nwarps = nwarps,
        .nbytes_shared = nbytes_shared,
        .nbatch_fa = D,
        .need_f16_K = need_f16_K,
        .need_f16_V = need_f16_V,
        .stream_k = false,
        .use_sparse = false,
    });
}

/// Ports `ggml_cuda_flash_attn_ext_vec_case` (fattn-vec.cuh:545 @b29c606e2).
pub fn vecCase(comptime D: comptime_int, comptime type_K: c.enum_ggml_type, comptime type_V: c.enum_ggml_type, ctx: *common.Context, dst: *Tensor) void {
    const KQV = dst;
    const Q = impl.one(Tensor, dst.src[0]);

    const logit_softcap = impl.getOpParamsF32(KQV, 2);

    if (Q.ne[1] == 1) {
        const cols_per_block = 1;
        if (logit_softcap == 0.0) {
            const use_logit_softcap = false;
            vecCaseImpl(D, cols_per_block, type_K, type_V, use_logit_softcap, ctx, dst);
        } else {
            const use_logit_softcap = true;
            vecCaseImpl(D, cols_per_block, type_K, type_V, use_logit_softcap, ctx, dst);
        }
        return;
    }

    const cols_per_block = 2;
    if (logit_softcap == 0.0) {
        const use_logit_softcap = false;
        vecCaseImpl(D, cols_per_block, type_K, type_V, use_logit_softcap, ctx, dst);
    } else {
        const use_logit_softcap = true;
        vecCaseImpl(D, cols_per_block, type_K, type_V, use_logit_softcap, ctx, dst);
    }
}
