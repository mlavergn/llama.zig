//! `supports_op` for the CUDA device.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v0.3.0
//! (`c1d0e7a00`): `ggml_backend_cuda_device_supports_op`, the CUDA arms
//! (the `GGML_USE_MUSA` exclusions compile out; `GGML_CUDA_USE_CUB` is
//! defined, so `TOP_K` and `ARGSORT` are unconditional).
//!
//! `cuda-ops-diff` asks this and the reference the same question for every
//! case it builds.

const std = @import("std");
const impl = @import("../impl.zig");
const lightning_indexer = @import("lightning_indexer.zig");
const backend = @import("backend.zig");
const fattn = @import("fattn.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// The device vtable's `supports_op`: `deviceSupportsOp` across the C ABI.
pub fn supportsOp(dev: c.ggml_backend_dev_t, op_c: [*c]const Tensor) callconv(.c) bool {
    const op = impl.one(Tensor, @constCast(op_c));
    return deviceSupportsOp(dev, op);
}

fn src(op: *const Tensor, i: usize) *const Tensor {
    return impl.one(Tensor, op.src[i]);
}

/// `ggml_cuda_lightning_indexer_supported`; see `lightning_indexer.zig`.
const lightningIndexerSupported = lightning_indexer.supported;
/// Ports `ggml_backend_cuda_device_supports_op` (ggml-cuda.cu:4867 @c1d0e7a00).
pub fn deviceSupportsOp(dev: c.ggml_backend_dev_t, op: *const Tensor) bool {
    const dev_ctx = backend.deviceCtx(dev);

    // check if all the sources are allocated on this device
    for (op.src) |s_opt| {
        const s = s_opt orelse continue;
        const buf = s.*.buffer orelse continue;
        const buft = impl.one(c.struct_ggml_backend_buffer, buf).buft;
        if (backend.buftIsCuda(buft)) {
            if (backend.buftDevice(buft) != dev_ctx.device) return false;
        }
    }

    switch (op.op) {
        c.GGML_OP_UNARY => switch (c.ggml_get_unary_op(op)) {
            c.GGML_UNARY_OP_ABS,
            c.GGML_UNARY_OP_SGN,
            c.GGML_UNARY_OP_NEG,
            c.GGML_UNARY_OP_STEP,
            c.GGML_UNARY_OP_GELU,
            c.GGML_UNARY_OP_SILU,
            c.GGML_UNARY_OP_RELU,
            c.GGML_UNARY_OP_SIGMOID,
            c.GGML_UNARY_OP_HARDSIGMOID,
            c.GGML_UNARY_OP_HARDSWISH,
            c.GGML_UNARY_OP_GELU_ERF,
            c.GGML_UNARY_OP_GELU_QUICK,
            c.GGML_UNARY_OP_TANH,
            c.GGML_UNARY_OP_EXP,
            c.GGML_UNARY_OP_EXPM1,
            c.GGML_UNARY_OP_SOFTPLUS,
            c.GGML_UNARY_OP_ELU,
            c.GGML_UNARY_OP_XIELU,
            c.GGML_UNARY_OP_FLOOR,
            c.GGML_UNARY_OP_CEIL,
            c.GGML_UNARY_OP_ROUND,
            c.GGML_UNARY_OP_TRUNC,
            => return c.ggml_is_contiguous(src(op, 0)),
            else => return false,
        },
        c.GGML_OP_GLU => switch (c.ggml_get_glu_op(op)) {
            c.GGML_GLU_OP_REGLU,
            c.GGML_GLU_OP_GEGLU,
            c.GGML_GLU_OP_SWIGLU,
            c.GGML_GLU_OP_SWIGLU_OAI,
            c.GGML_GLU_OP_GEGLU_ERF,
            c.GGML_GLU_OP_GEGLU_QUICK,
            => return c.ggml_is_contiguous_1(src(op, 0)),
            else => return false,
        },
        c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT_ID => {
            const a = src(op, 0);
            const b = src(op, 1);
            if (a.nb[0] != c.ggml_element_size(a) or b.nb[0] != c.ggml_element_size(b)) return false;
            if (b.type == c.GGML_TYPE_F16 and a.type != c.GGML_TYPE_F16) return false;
            return switch (a.type) {
                c.GGML_TYPE_F32,
                c.GGML_TYPE_F16,
                c.GGML_TYPE_Q1_0,
                c.GGML_TYPE_Q2_0,
                c.GGML_TYPE_Q4_0,
                c.GGML_TYPE_Q4_1,
                c.GGML_TYPE_Q5_0,
                c.GGML_TYPE_Q5_1,
                c.GGML_TYPE_Q8_0,
                c.GGML_TYPE_MXFP4,
                c.GGML_TYPE_NVFP4,
                c.GGML_TYPE_Q2_K,
                c.GGML_TYPE_Q3_K,
                c.GGML_TYPE_Q4_K,
                c.GGML_TYPE_Q5_K,
                c.GGML_TYPE_Q6_K,
                c.GGML_TYPE_Q8_K,
                c.GGML_TYPE_IQ1_M,
                c.GGML_TYPE_IQ1_S,
                c.GGML_TYPE_IQ2_S,
                c.GGML_TYPE_IQ2_XS,
                c.GGML_TYPE_IQ2_XXS,
                c.GGML_TYPE_IQ3_S,
                c.GGML_TYPE_IQ3_XXS,
                c.GGML_TYPE_IQ4_NL,
                c.GGML_TYPE_IQ4_XS,
                c.GGML_TYPE_BF16,
                => true,
                else => false,
            };
        },
        c.GGML_OP_OUT_PROD => return op.type == c.GGML_TYPE_F32 and src(op, 0).type == c.GGML_TYPE_F32 and src(op, 1).type == c.GGML_TYPE_F32,
        c.GGML_OP_GET_ROWS => return switch (src(op, 0).type) {
            c.GGML_TYPE_F16,
            c.GGML_TYPE_F32,
            c.GGML_TYPE_BF16,
            c.GGML_TYPE_I32,
            c.GGML_TYPE_Q1_0,
            c.GGML_TYPE_Q2_0,
            c.GGML_TYPE_Q4_0,
            c.GGML_TYPE_Q4_1,
            c.GGML_TYPE_Q5_0,
            c.GGML_TYPE_Q5_1,
            c.GGML_TYPE_Q8_0,
            c.GGML_TYPE_Q2_K,
            c.GGML_TYPE_Q3_K,
            c.GGML_TYPE_Q4_K,
            c.GGML_TYPE_Q5_K,
            c.GGML_TYPE_Q6_K,
            c.GGML_TYPE_IQ2_XXS,
            c.GGML_TYPE_IQ2_XS,
            c.GGML_TYPE_IQ2_S,
            c.GGML_TYPE_IQ3_XXS,
            c.GGML_TYPE_IQ3_S,
            c.GGML_TYPE_IQ1_S,
            c.GGML_TYPE_IQ1_M,
            c.GGML_TYPE_IQ4_XS,
            => true,
            // 32-value sub-blocks, the row size does not guarantee the QK_K
            // super-blocks the get_rows kernel iterates on
            c.GGML_TYPE_IQ4_NL, c.GGML_TYPE_MXFP4 => @mod(src(op, 0).ne[0], c.QK_K) == 0,
            else => false,
        },
        c.GGML_OP_GET_ROWS_BACK => return op.type == c.GGML_TYPE_F32 and src(op, 0).type == c.GGML_TYPE_F32 and op.ne[2] == 1 and op.ne[3] == 1,
        c.GGML_OP_SET_ROWS => {
            const t = op.type;
            const s0 = src(op, 0).type;
            const s1 = src(op, 1).type;
            return (((t == c.GGML_TYPE_F32 or t == c.GGML_TYPE_F16 or t == c.GGML_TYPE_BF16 or
                t == c.GGML_TYPE_Q4_0 or t == c.GGML_TYPE_Q4_1 or t == c.GGML_TYPE_Q5_0 or
                t == c.GGML_TYPE_Q5_1 or t == c.GGML_TYPE_Q8_0 or t == c.GGML_TYPE_IQ4_NL) and s0 == c.GGML_TYPE_F32) or
                (t == c.GGML_TYPE_F16 and s0 == c.GGML_TYPE_F16)) and
                (s1 == c.GGML_TYPE_I64 or s1 == c.GGML_TYPE_I32);
        },
        c.GGML_OP_SET => {
            const t = op.type;
            return (t == c.GGML_TYPE_F32 or t == c.GGML_TYPE_I32) and t == src(op, 0).type and t == src(op, 1).type;
        },
        c.GGML_OP_CPY => {
            const s0 = src(op, 0).type;
            const s1 = src(op, 1).type;
            const F32 = c.GGML_TYPE_F32;
            if ((s0 == F32 or s0 == c.GGML_TYPE_BF16 or s0 == c.GGML_TYPE_F16) and
                (s1 == F32 or s1 == c.GGML_TYPE_BF16 or s1 == c.GGML_TYPE_F16)) return true;
            const pairs = [_][2]c.enum_ggml_type{
                .{ F32, c.GGML_TYPE_Q8_0 },   .{ c.GGML_TYPE_Q8_0, F32 },
                .{ F32, c.GGML_TYPE_Q4_0 },   .{ c.GGML_TYPE_Q4_0, F32 },
                .{ F32, c.GGML_TYPE_Q4_1 },   .{ c.GGML_TYPE_Q4_1, F32 },
                .{ F32, c.GGML_TYPE_Q5_0 },   .{ c.GGML_TYPE_Q5_0, F32 },
                .{ F32, c.GGML_TYPE_Q5_1 },   .{ c.GGML_TYPE_Q5_1, F32 },
                .{ F32, c.GGML_TYPE_IQ4_NL }, .{ F32, c.GGML_TYPE_I32 },
                .{ c.GGML_TYPE_I32, F32 },    .{ c.GGML_TYPE_I32, c.GGML_TYPE_I32 },
            };
            for (pairs) |p| {
                if (s0 == p[0] and s1 == p[1]) return true;
            }
            return s0 == s1 and c.ggml_is_contiguous(src(op, 0)) and c.ggml_is_contiguous(src(op, 1));
        },
        c.GGML_OP_DUP => {
            const s0 = src(op, 0).type;
            return s0 != c.GGML_TYPE_I32 and s0 != c.GGML_TYPE_I16;
        },
        c.GGML_OP_ARGMAX, c.GGML_OP_COUNT_EQUAL => return true,
        c.GGML_OP_REPEAT => {
            // the CUDA REPEAT path only implements F32/F16
            const s0 = src(op, 0).type;
            return s0 == c.GGML_TYPE_F32 or s0 == c.GGML_TYPE_F16;
        },
        c.GGML_OP_REPEAT_BACK => return op.type == c.GGML_TYPE_F32 and (src(op, 0).ne[2] * src(op, 0).ne[3]) <= (1 << 15),
        c.GGML_OP_CONCAT => {
            const s0 = src(op, 0);
            const s1 = src(op, 1);
            const t0 = s0.type;
            const dim = op.op_params[0];
            if (t0 != s1.type or t0 != op.type) return false;
            if (c.ggml_is_quantized(t0)) {
                const layout_ok = if (dim == 3)
                    c.ggml_is_contiguous(s0) and c.ggml_is_contiguous(s1)
                else
                    c.ggml_is_contiguous_to_3(s0) and c.ggml_is_contiguous_to_3(s1);
                return layout_ok and @mod(s0.ne[0], c.ggml_blck_size(t0)) == 0 and @mod(s1.ne[0], c.ggml_blck_size(t0)) == 0;
            }
            const ts = c.ggml_type_size(t0);
            return c.ggml_blck_size(t0) == 1 and (ts == 1 or ts == 2 or ts == 4 or ts == 8);
        },
        c.GGML_OP_CONV_TRANSPOSE_1D => return src(op, 0).type == c.GGML_TYPE_F32 and src(op, 1).type == c.GGML_TYPE_F32,
        c.GGML_OP_COL2IM_1D => {
            const t0 = src(op, 0).type;
            return (t0 == c.GGML_TYPE_F32 or t0 == c.GGML_TYPE_F16 or t0 == c.GGML_TYPE_BF16) and
                op.type == t0 and c.ggml_is_contiguous(src(op, 0)) and c.ggml_is_contiguous(op);
        },
        c.GGML_OP_SILU_BACK => return c.ggml_is_contiguous(src(op, 0)) and src(op, 0).type == c.GGML_TYPE_F32,
        c.GGML_OP_NORM, c.GGML_OP_RMS_NORM, c.GGML_OP_L2_NORM => return c.ggml_is_contiguous_rows(src(op, 0)),
        c.GGML_OP_RMS_NORM_BACK => return c.ggml_is_contiguous(src(op, 0)),
        c.GGML_OP_NONE,
        c.GGML_OP_RESHAPE,
        c.GGML_OP_VIEW,
        c.GGML_OP_PERMUTE,
        c.GGML_OP_TRANSPOSE,
        c.GGML_OP_ADD_ID,
        c.GGML_OP_ADD1,
        c.GGML_OP_SCALE,
        c.GGML_OP_SQR,
        c.GGML_OP_SQRT,
        c.GGML_OP_SIN,
        c.GGML_OP_COS,
        c.GGML_OP_CLAMP,
        c.GGML_OP_LOG,
        => return true,
        c.GGML_OP_ADD, c.GGML_OP_SUB, c.GGML_OP_MUL, c.GGML_OP_DIV => {
            const ok = struct {
                fn f(t: c.enum_ggml_type) bool {
                    return t == c.GGML_TYPE_F32 or t == c.GGML_TYPE_F16;
                }
            }.f;
            return ok(src(op, 0).type) and ok(src(op, 1).type) and ok(op.type);
        },
        c.GGML_OP_SSM_SCAN => {
            const K = impl.getOpParamsI32(op, 0);
            if (src(op, 3).ne[0] == 1) {
                // Mamba2: d_state == 128 || d_state == 256, d_head % 16 == 0
                return (src(op, 0).ne[0] == 128 or src(op, 0).ne[0] == 256) and @mod(src(op, 0).ne[1], 16) == 0;
            }
            if (K > 1) return false;
            // Mamba: d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1
            return src(op, 0).ne[0] == 16 and src(op, 0).ne[1] == 1 and @mod(src(op, 0).ne[2], 128) == 0 and src(op, 4).ne[1] == 1;
        },
        // assumes d_inner % threads == 0
        c.GGML_OP_SSM_CONV => return @mod(src(op, 0).ne[1], 128) == 0,
        c.GGML_OP_CONT, c.GGML_OP_DIAG_MASK_INF, c.GGML_OP_SOFT_MAX => return true,
        c.GGML_OP_SOFT_MAX_BACK => return impl.getOpParamsF32(op, 1) == 0.0,
        c.GGML_OP_ROLL => return src(op, 0).type == c.GGML_TYPE_F32 and c.ggml_is_contiguous(src(op, 0)),
        c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => return src(op, 0).nb[0] == c.ggml_type_size(src(op, 0).type) and c.ggml_is_contiguous_2(src(op, 0)),
        c.GGML_OP_IM2COL, c.GGML_OP_IM2COL_3D, c.GGML_OP_CONV_2D => return c.ggml_is_contiguous(src(op, 0)) and c.ggml_is_contiguous(src(op, 1)),
        c.GGML_OP_CONV_2D_DW => return src(op, 0).type == c.GGML_TYPE_F32,
        c.GGML_OP_CONV_TRANSPOSE_2D, c.GGML_OP_POOL_1D, c.GGML_OP_POOL_2D => return true,
        c.GGML_OP_ACC => return c.ggml_is_contiguous(src(op, 0)) and c.ggml_is_contiguous(src(op, 1)),
        c.GGML_OP_SUM => return c.ggml_is_contiguous_rows(src(op, 0)),
        c.GGML_OP_TOP_K, c.GGML_OP_ARGSORT => return true,
        c.GGML_OP_SUM_ROWS, c.GGML_OP_MEAN, c.GGML_OP_GROUP_NORM => return c.ggml_is_contiguous(src(op, 0)),
        c.GGML_OP_PAD => return true,
        c.GGML_OP_UPSCALE,
        c.GGML_OP_PAD_REFLECT_1D,
        c.GGML_OP_ARANGE,
        c.GGML_OP_TIMESTEP_EMBEDDING,
        c.GGML_OP_LEAKY_RELU,
        c.GGML_OP_RWKV_WKV6,
        c.GGML_OP_GATED_LINEAR_ATTN,
        c.GGML_OP_RWKV_WKV7,
        => return true,
        c.GGML_OP_GATED_DELTA_NET => return true,
        c.GGML_OP_DSV4_HC_COMB => return src(op, 0).type == c.GGML_TYPE_F32 and src(op, 1).type == c.GGML_TYPE_F32 and
            src(op, 2).type == c.GGML_TYPE_F32 and op.type == c.GGML_TYPE_F32,
        c.GGML_OP_DSV4_HC_PRE => return src(op, 0).type == c.GGML_TYPE_F32 and src(op, 1).type == c.GGML_TYPE_F32 and
            op.type == c.GGML_TYPE_F32,
        c.GGML_OP_DSV4_HC_POST => return src(op, 0).type == c.GGML_TYPE_F32 and src(op, 1).type == c.GGML_TYPE_F32 and
            src(op, 2).type == c.GGML_TYPE_F32 and src(op, 3).type == c.GGML_TYPE_F32 and op.type == c.GGML_TYPE_F32,
        c.GGML_OP_FLASH_ATTN_EXT => return fattn.supported(dev_ctx.device, op),
        c.GGML_OP_CROSS_ENTROPY_LOSS,
        c.GGML_OP_CROSS_ENTROPY_LOSS_BACK,
        c.GGML_OP_OPT_STEP_ADAMW,
        c.GGML_OP_OPT_STEP_SGD,
        c.GGML_OP_FILL,
        c.GGML_OP_CUMSUM,
        c.GGML_OP_TRI,
        c.GGML_OP_DIAG,
        c.GGML_OP_SOLVE_TRI,
        => return true,
        c.GGML_OP_LIGHTNING_INDEXER => return lightningIndexerSupported(dev_ctx.device, op),
        else => return false,
    }
}
