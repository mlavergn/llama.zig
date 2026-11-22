//! `GGML_OP_GATED_LINEAR_ATTN`: gated linear attention's recurrence.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/gla.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/gla.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `gated_linear_attn_f32`, stays CUDA.

const ggml = @import("../module.zig");
/// An `int64_t` into a `dim3` member, which is `unsigned int`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// An `int64_t` into an `int` parameter.
fn int(x: i64) c_int {
    return @truncate(x);
}

/// Ports `ggml_cuda_op_gated_linear_attn` (gla.cu:64 @b1b770d90).
pub fn opGatedLinearAttn(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const k_d: ?*const anyopaque = dst.src[0].*.data;
    const v_d: ?*const anyopaque = dst.src[1].*.data;
    const r_d: ?*const anyopaque = dst.src[2].*.data;
    const td_d: ?*const anyopaque = dst.src[3].*.data;
    const s_d: ?*const anyopaque = dst.src[4].*.data;

    const B = dst.src[4].*.ne[1];
    const T = dst.src[0].*.ne[2];
    const C = dst.ne[0];
    const H = dst.src[0].*.ne[1];

    const scale = ggml.impl.getOpParamsF32(dst, 0);

    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    ggml.impl.assert(dst.src[4].*.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->src[4]->type == GGML_TYPE_F32");
    ggml.impl.assert(@rem(C, H) == 0, "C % H == 0");
    ggml.impl.assert(@divTrunc(C, H) == 64 or @divTrunc(C, H) == 128, "C / H == 64 || C / H == 128");

    const p: ggml.cuda.kernels.LaunchParams = .{ .grid = .{ .x = dim(B * H) }, .block = .{ .x = dim(@divTrunc(C, H)) }, .shmem = 0, .stream = stream };
    const args = .{ int(B), int(T), int(C), int(H), scale, k_d, v_d, r_d, td_d, s_d, dst_d };
    if (@divTrunc(C, H) == 64) {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("gla", "gated_linear_attn_f32<(int)64>"), p, args);
    } else {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("gla", "gated_linear_attn_f32<(int)128>"), p, args);
    }
}
