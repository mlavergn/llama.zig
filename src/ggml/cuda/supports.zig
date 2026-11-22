//! `supports_op` for the CUDA device.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v1.0.0
//! (`b1b770d90`): `ggml_backend_cuda_device_supports_op`, the CUDA arms
//! (the `GGML_USE_MUSA` exclusions compile out; `GGML_CUDA_USE_CUB` is
//! defined, so `ARGSORT` takes its CUB arm -- the shared-memory bound
//! is the bitonic-only one -- and both it and `TOP_K`
//! accept any row of at most `INT_MAX` elements).
//! `GATED_DELTA_NET` has no `GGML_USE_MUSA` exclusion, and is `true` here.
//!
//! `cuda-ops-diff` asks this and the reference the same question for every
//! case it builds.

const std = @import("std");
const ggml = @import("../module.zig");

/// The device vtable's `supports_op`: `deviceSupportsOp` across the C ABI.
pub fn supportsOp(dev: ggml.impl.c.ggml_backend_dev_t, op_c: [*c]const ggml.impl.c.ggml_tensor) callconv(.c) bool {
    const op = ggml.impl.one(ggml.impl.c.ggml_tensor, @constCast(op_c));
    return deviceSupportsOp(dev, op);
}

fn src(op: *const ggml.impl.c.ggml_tensor, i: usize) *const ggml.impl.c.ggml_tensor {
    return ggml.impl.one(ggml.impl.c.ggml_tensor, op.src[i]);
}

/// `ggml_cuda_lightning_indexer_supported`; see `lightning_indexer.zig`.
/// Ports `ggml_backend_cuda_device_supports_op` (ggml-cuda.cu:6195 @b1b770d90).
/// The `F8_E4M3` type in `MUL_MAT`, `GET_ROWS`, `SET_ROWS` and
/// `CPY`; `MUL_MAT`'s `src1` precision (`ggml_prec_get_src`) gates F32 and
/// BF16 requests; `SSM_SCAN` takes `d_state == 64`; `ADD1` and
/// `DIAG_MASK_INF` are not in ggml.
/// `UNARY` takes any strides for F16, F32 and BF16 (the kernels
/// index by stride now), `CLAMP` and `ROLL` take non-contiguous views,
/// `SWIGLU_OAI` refuses F16, and `TOP_K`/`ARGSORT` are bounded by
/// `INT_MAX` columns.
pub fn deviceSupportsOp(dev: ggml.impl.c.ggml_backend_dev_t, op: *const ggml.impl.c.ggml_tensor) bool {
    const dev_ctx = ggml.cuda.backend.deviceCtx(dev);

    // check if all the sources are allocated on this device
    for (op.src) |s_opt| {
        const s = s_opt orelse continue;
        const buf = s.*.buffer orelse continue;
        const buft = ggml.impl.one(ggml.impl.c.struct_ggml_backend_buffer, buf).buft;
        if (ggml.cuda.backend.buftIsCuda(buft)) {
            if (ggml.cuda.backend.buftDevice(buft) != dev_ctx.device) return false;
        }
    }

    switch (op.op) {
        ggml.impl.c.GGML_OP_UNARY => switch (ggml.impl.c.ggml_get_unary_op(op)) {
            ggml.impl.c.GGML_UNARY_OP_ABS,
            ggml.impl.c.GGML_UNARY_OP_SGN,
            ggml.impl.c.GGML_UNARY_OP_NEG,
            ggml.impl.c.GGML_UNARY_OP_STEP,
            ggml.impl.c.GGML_UNARY_OP_GELU,
            ggml.impl.c.GGML_UNARY_OP_SILU,
            ggml.impl.c.GGML_UNARY_OP_RELU,
            ggml.impl.c.GGML_UNARY_OP_SIGMOID,
            ggml.impl.c.GGML_UNARY_OP_HARDSIGMOID,
            ggml.impl.c.GGML_UNARY_OP_HARDSWISH,
            ggml.impl.c.GGML_UNARY_OP_GELU_ERF,
            ggml.impl.c.GGML_UNARY_OP_GELU_QUICK,
            ggml.impl.c.GGML_UNARY_OP_TANH,
            ggml.impl.c.GGML_UNARY_OP_EXP,
            ggml.impl.c.GGML_UNARY_OP_EXPM1,
            ggml.impl.c.GGML_UNARY_OP_SOFTPLUS,
            ggml.impl.c.GGML_UNARY_OP_ELU,
            ggml.impl.c.GGML_UNARY_OP_XIELU,
            ggml.impl.c.GGML_UNARY_OP_FLOOR,
            ggml.impl.c.GGML_UNARY_OP_CEIL,
            ggml.impl.c.GGML_UNARY_OP_ROUND,
            ggml.impl.c.GGML_UNARY_OP_TRUNC,
            => {
                if (src(op, 0).type == ggml.impl.c.GGML_TYPE_BF16 and ggml.impl.c.ggml_get_unary_op(op) == ggml.impl.c.GGML_UNARY_OP_XIELU) return false;
                return src(op, 0).type == ggml.impl.c.GGML_TYPE_F16 or src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 or src(op, 0).type == ggml.impl.c.GGML_TYPE_BF16;
            },
            else => return false,
        },
        ggml.impl.c.GGML_OP_GLU => switch (ggml.impl.c.ggml_get_glu_op(op)) {
            ggml.impl.c.GGML_GLU_OP_REGLU,
            ggml.impl.c.GGML_GLU_OP_GEGLU,
            ggml.impl.c.GGML_GLU_OP_SWIGLU,
            ggml.impl.c.GGML_GLU_OP_SWIGLU_OAI,
            ggml.impl.c.GGML_GLU_OP_GEGLU_ERF,
            ggml.impl.c.GGML_GLU_OP_GEGLU_QUICK,
            ggml.impl.c.GGML_GLU_OP_SWIGLU_CLAMP,
            => {
                if (src(op, 0).type == ggml.impl.c.GGML_TYPE_BF16 and
                    (ggml.impl.c.ggml_get_glu_op(op) == ggml.impl.c.GGML_GLU_OP_SWIGLU_OAI or ggml.impl.c.ggml_get_glu_op(op) == ggml.impl.c.GGML_GLU_OP_SWIGLU_CLAMP))
                {
                    return false;
                }
                if (src(op, 0).type == ggml.impl.c.GGML_TYPE_F16 and ggml.impl.c.ggml_get_glu_op(op) == ggml.impl.c.GGML_GLU_OP_SWIGLU_OAI) return false;
                return ggml.impl.c.ggml_is_contiguous_1(src(op, 0));
            },
            else => return false,
        },
        ggml.impl.c.GGML_OP_MUL_MAT, ggml.impl.c.GGML_OP_MUL_MAT_ID => {
            const a = src(op, 0);
            const b = src(op, 1);
            if (a.nb[0] != ggml.impl.c.ggml_element_size(a) or b.nb[0] != ggml.impl.c.ggml_element_size(b)) return false;
            if (b.type == ggml.impl.c.GGML_TYPE_F16 and a.type != ggml.impl.c.GGML_TYPE_F16 and !ggml.cuda.fwht.opMulMatUseFwht(op)) return false;
            const cc_dev = ggml.cuda.common.info().devices[@intCast(dev_ctx.device)].cc;
            const f32_src1_ok = ggml.cuda.common.cc.isNvidia(cc_dev) and cc_dev < ggml.cuda.common.cc.ampere;
            const prec_src1 = ggml.impl.c.ggml_prec_get_src(op, 1);
            const fp8_id_mmvf = op.op == ggml.impl.c.GGML_OP_MUL_MAT_ID and
                a.type == ggml.impl.c.GGML_TYPE_F8_E4M3 and b.type == ggml.impl.c.GGML_TYPE_F32 and op.type == ggml.impl.c.GGML_TYPE_F32 and
                ggml.cuda.mmvf.shouldUseMmvf(a, cc_dev, ggml.cuda.common.info().devices[@intCast(dev_ctx.device)].warp_size, b.ne[2]);
            if (prec_src1 == ggml.impl.c.GGML_PREC_F32 and !fp8_id_mmvf and (op.op == ggml.impl.c.GGML_OP_MUL_MAT_ID or !f32_src1_ok or a.type != ggml.impl.c.GGML_TYPE_F32)) {
                return false;
            }
            if (prec_src1 == ggml.impl.c.GGML_PREC_BF16 and (op.op == ggml.impl.c.GGML_OP_MUL_MAT_ID or a.type != ggml.impl.c.GGML_TYPE_F32)) {
                return false;
            }
            return switch (a.type) {
                ggml.impl.c.GGML_TYPE_F32,
                ggml.impl.c.GGML_TYPE_F16,
                ggml.impl.c.GGML_TYPE_Q1_0,
                ggml.impl.c.GGML_TYPE_Q2_0,
                ggml.impl.c.GGML_TYPE_Q4_0,
                ggml.impl.c.GGML_TYPE_Q4_1,
                ggml.impl.c.GGML_TYPE_Q5_0,
                ggml.impl.c.GGML_TYPE_Q5_1,
                ggml.impl.c.GGML_TYPE_Q8_0,
                ggml.impl.c.GGML_TYPE_MXFP4,
                ggml.impl.c.GGML_TYPE_NVFP4,
                ggml.impl.c.GGML_TYPE_F8_E4M3,
                ggml.impl.c.GGML_TYPE_Q2_K,
                ggml.impl.c.GGML_TYPE_Q3_K,
                ggml.impl.c.GGML_TYPE_Q4_K,
                ggml.impl.c.GGML_TYPE_Q5_K,
                ggml.impl.c.GGML_TYPE_Q6_K,
                ggml.impl.c.GGML_TYPE_Q8_K,
                ggml.impl.c.GGML_TYPE_IQ1_M,
                ggml.impl.c.GGML_TYPE_IQ1_S,
                ggml.impl.c.GGML_TYPE_IQ2_S,
                ggml.impl.c.GGML_TYPE_IQ2_XS,
                ggml.impl.c.GGML_TYPE_IQ2_XXS,
                ggml.impl.c.GGML_TYPE_IQ3_S,
                ggml.impl.c.GGML_TYPE_IQ3_XXS,
                ggml.impl.c.GGML_TYPE_IQ4_NL,
                ggml.impl.c.GGML_TYPE_IQ4_XS,
                ggml.impl.c.GGML_TYPE_BF16,
                => true,
                else => false,
            };
        },
        ggml.impl.c.GGML_OP_OUT_PROD => return op.type == ggml.impl.c.GGML_TYPE_F32 and src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 and src(op, 1).type == ggml.impl.c.GGML_TYPE_F32,
        ggml.impl.c.GGML_OP_GET_ROWS => return switch (src(op, 0).type) {
            ggml.impl.c.GGML_TYPE_F16,
            ggml.impl.c.GGML_TYPE_F32,
            ggml.impl.c.GGML_TYPE_BF16,
            ggml.impl.c.GGML_TYPE_I32,
            ggml.impl.c.GGML_TYPE_Q1_0,
            ggml.impl.c.GGML_TYPE_Q2_0,
            ggml.impl.c.GGML_TYPE_Q4_0,
            ggml.impl.c.GGML_TYPE_Q4_1,
            ggml.impl.c.GGML_TYPE_Q5_0,
            ggml.impl.c.GGML_TYPE_Q5_1,
            ggml.impl.c.GGML_TYPE_Q8_0,
            ggml.impl.c.GGML_TYPE_F8_E4M3,
            ggml.impl.c.GGML_TYPE_Q2_K,
            ggml.impl.c.GGML_TYPE_Q3_K,
            ggml.impl.c.GGML_TYPE_Q4_K,
            ggml.impl.c.GGML_TYPE_Q5_K,
            ggml.impl.c.GGML_TYPE_Q6_K,
            ggml.impl.c.GGML_TYPE_IQ2_XXS,
            ggml.impl.c.GGML_TYPE_IQ2_XS,
            ggml.impl.c.GGML_TYPE_IQ2_S,
            ggml.impl.c.GGML_TYPE_IQ3_XXS,
            ggml.impl.c.GGML_TYPE_IQ3_S,
            ggml.impl.c.GGML_TYPE_IQ1_S,
            ggml.impl.c.GGML_TYPE_IQ1_M,
            ggml.impl.c.GGML_TYPE_IQ4_XS,
            => true,
            // 32-value sub-blocks, the row size does not guarantee the QK_K
            // super-blocks the get_rows kernel iterates on
            ggml.impl.c.GGML_TYPE_IQ4_NL, ggml.impl.c.GGML_TYPE_MXFP4 => @mod(src(op, 0).ne[0], ggml.impl.c.QK_K) == 0,
            else => false,
        },
        ggml.impl.c.GGML_OP_GET_ROWS_BACK => return op.type == ggml.impl.c.GGML_TYPE_F32 and src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 and op.ne[2] == 1 and op.ne[3] == 1,
        ggml.impl.c.GGML_OP_SET_ROWS => {
            const t = op.type;
            const s0 = src(op, 0).type;
            const s1 = src(op, 1).type;
            return (((t == ggml.impl.c.GGML_TYPE_F32 or t == ggml.impl.c.GGML_TYPE_F16 or t == ggml.impl.c.GGML_TYPE_BF16 or t == ggml.impl.c.GGML_TYPE_F8_E4M3 or
                t == ggml.impl.c.GGML_TYPE_Q4_0 or t == ggml.impl.c.GGML_TYPE_Q4_1 or t == ggml.impl.c.GGML_TYPE_Q5_0 or
                t == ggml.impl.c.GGML_TYPE_Q5_1 or t == ggml.impl.c.GGML_TYPE_Q8_0 or t == ggml.impl.c.GGML_TYPE_IQ4_NL) and s0 == ggml.impl.c.GGML_TYPE_F32) or
                (t == ggml.impl.c.GGML_TYPE_F16 and s0 == ggml.impl.c.GGML_TYPE_F16)) and
                (s1 == ggml.impl.c.GGML_TYPE_I64 or s1 == ggml.impl.c.GGML_TYPE_I32);
        },
        ggml.impl.c.GGML_OP_SET => {
            const t = op.type;
            return (t == ggml.impl.c.GGML_TYPE_F32 or t == ggml.impl.c.GGML_TYPE_I32) and t == src(op, 0).type and t == src(op, 1).type;
        },
        ggml.impl.c.GGML_OP_CPY => {
            const s0 = src(op, 0).type;
            const s1 = src(op, 1).type;
            const F32 = ggml.impl.c.GGML_TYPE_F32;
            const F8 = ggml.impl.c.GGML_TYPE_F8_E4M3;
            if (s0 == F8 and s1 == F8) return true;
            if ((s0 == F32 or s0 == ggml.impl.c.GGML_TYPE_F16 or s0 == ggml.impl.c.GGML_TYPE_BF16) and s1 == F8) return true;
            if (s0 == F8 and s1 == F32) return true;
            if ((s0 == F32 or s0 == ggml.impl.c.GGML_TYPE_BF16 or s0 == ggml.impl.c.GGML_TYPE_F16) and
                (s1 == F32 or s1 == ggml.impl.c.GGML_TYPE_BF16 or s1 == ggml.impl.c.GGML_TYPE_F16)) return true;
            const pairs = [_][2]ggml.impl.c.enum_ggml_type{
                .{ F32, ggml.impl.c.GGML_TYPE_Q8_0 },   .{ ggml.impl.c.GGML_TYPE_Q8_0, F32 },
                .{ F32, ggml.impl.c.GGML_TYPE_Q4_0 },   .{ ggml.impl.c.GGML_TYPE_Q4_0, F32 },
                .{ F32, ggml.impl.c.GGML_TYPE_Q4_1 },   .{ ggml.impl.c.GGML_TYPE_Q4_1, F32 },
                .{ F32, ggml.impl.c.GGML_TYPE_Q5_0 },   .{ ggml.impl.c.GGML_TYPE_Q5_0, F32 },
                .{ F32, ggml.impl.c.GGML_TYPE_Q5_1 },   .{ ggml.impl.c.GGML_TYPE_Q5_1, F32 },
                .{ F32, ggml.impl.c.GGML_TYPE_IQ4_NL }, .{ F32, ggml.impl.c.GGML_TYPE_I32 },
                .{ ggml.impl.c.GGML_TYPE_I32, F32 },    .{ ggml.impl.c.GGML_TYPE_I32, ggml.impl.c.GGML_TYPE_I32 },
            };
            for (pairs) |p| {
                if (s0 == p[0] and s1 == p[1]) return true;
            }
            return s0 == s1 and ggml.impl.c.ggml_is_contiguous(src(op, 0)) and ggml.impl.c.ggml_is_contiguous(src(op, 1));
        },
        ggml.impl.c.GGML_OP_DUP => return true,
        ggml.impl.c.GGML_OP_ARGMAX, ggml.impl.c.GGML_OP_COUNT_EQUAL => return true,
        ggml.impl.c.GGML_OP_REPEAT => {
            // the CUDA REPEAT path only implements F32/F16
            const s0 = src(op, 0).type;
            return s0 == ggml.impl.c.GGML_TYPE_F32 or s0 == ggml.impl.c.GGML_TYPE_F16;
        },
        ggml.impl.c.GGML_OP_REPEAT_BACK => return op.type == ggml.impl.c.GGML_TYPE_F32 and (src(op, 0).ne[2] * src(op, 0).ne[3]) <= (1 << 15),
        ggml.impl.c.GGML_OP_CONCAT => {
            const s0 = src(op, 0);
            const s1 = src(op, 1);
            const t0 = s0.type;
            const dim = op.op_params[0];
            if (t0 != s1.type or t0 != op.type) return false;
            if (ggml.impl.c.ggml_is_quantized(t0)) {
                const layout_ok = if (dim == 3)
                    ggml.impl.c.ggml_is_contiguous(s0) and ggml.impl.c.ggml_is_contiguous(s1)
                else
                    ggml.impl.c.ggml_is_contiguous_to_3(s0) and ggml.impl.c.ggml_is_contiguous_to_3(s1);
                return layout_ok and @mod(s0.ne[0], ggml.impl.c.ggml_blck_size(t0)) == 0 and @mod(s1.ne[0], ggml.impl.c.ggml_blck_size(t0)) == 0;
            }
            const ts = ggml.impl.c.ggml_type_size(t0);
            return ggml.impl.c.ggml_blck_size(t0) == 1 and (ts == 1 or ts == 2 or ts == 4 or ts == 8);
        },
        ggml.impl.c.GGML_OP_CONV_TRANSPOSE_1D => return src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 and src(op, 1).type == ggml.impl.c.GGML_TYPE_F32,
        ggml.impl.c.GGML_OP_COL2IM_1D => {
            const t0 = src(op, 0).type;
            return (t0 == ggml.impl.c.GGML_TYPE_F32 or t0 == ggml.impl.c.GGML_TYPE_F16 or t0 == ggml.impl.c.GGML_TYPE_BF16) and
                op.type == t0 and ggml.impl.c.ggml_is_contiguous(src(op, 0)) and ggml.impl.c.ggml_is_contiguous(op);
        },
        ggml.impl.c.GGML_OP_SILU_BACK => return ggml.impl.c.ggml_is_contiguous(src(op, 0)) and src(op, 0).type == ggml.impl.c.GGML_TYPE_F32,
        ggml.impl.c.GGML_OP_NORM, ggml.impl.c.GGML_OP_RMS_NORM, ggml.impl.c.GGML_OP_L2_NORM => return ggml.impl.c.ggml_is_contiguous_rows(src(op, 0)),
        ggml.impl.c.GGML_OP_RMS_NORM_BACK => return ggml.impl.c.ggml_is_contiguous(src(op, 0)),
        ggml.impl.c.GGML_OP_NONE,
        ggml.impl.c.GGML_OP_RESHAPE,
        ggml.impl.c.GGML_OP_VIEW,
        ggml.impl.c.GGML_OP_PERMUTE,
        ggml.impl.c.GGML_OP_TRANSPOSE,
        ggml.impl.c.GGML_OP_ADD_ID,
        ggml.impl.c.GGML_OP_SQR,
        ggml.impl.c.GGML_OP_SQRT,
        ggml.impl.c.GGML_OP_SIN,
        ggml.impl.c.GGML_OP_COS,
        ggml.impl.c.GGML_OP_LOG,
        => return true,
        ggml.impl.c.GGML_OP_SCALE => return (src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 or src(op, 0).type == ggml.impl.c.GGML_TYPE_BF16) and op.type == src(op, 0).type and
            ggml.impl.c.ggml_is_contiguous_rows(src(op, 0)),
        ggml.impl.c.GGML_OP_CLAMP => return ggml.impl.c.ggml_is_contiguous_rows(src(op, 0)),
        ggml.impl.c.GGML_OP_ADD, ggml.impl.c.GGML_OP_SUB, ggml.impl.c.GGML_OP_MUL, ggml.impl.c.GGML_OP_DIV => {
            if (src(op, 0).type == ggml.impl.c.GGML_TYPE_BF16 or src(op, 1).type == ggml.impl.c.GGML_TYPE_BF16 or op.type == ggml.impl.c.GGML_TYPE_BF16) {
                return src(op, 0).type == ggml.impl.c.GGML_TYPE_BF16 and op.type == ggml.impl.c.GGML_TYPE_BF16 and
                    (src(op, 1).type == ggml.impl.c.GGML_TYPE_BF16 or src(op, 1).type == ggml.impl.c.GGML_TYPE_F32);
            }
            const ok = struct {
                fn f(t: ggml.impl.c.enum_ggml_type) bool {
                    return t == ggml.impl.c.GGML_TYPE_F32 or t == ggml.impl.c.GGML_TYPE_F16;
                }
            }.f;
            return ok(src(op, 0).type) and ok(src(op, 1).type) and ok(op.type);
        },
        ggml.impl.c.GGML_OP_SSM_SCAN => {
            const K = ggml.impl.getOpParamsI32(op, 0);
            if (src(op, 3).ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 64 || d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const d_state = src(op, 0).ne[0];
                return (d_state == 64 or d_state == 96 or d_state == 128 or d_state == 256) and @mod(src(op, 0).ne[1], 16) == 0;
            }
            if (K > 1) return false;
            // Mamba: d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1
            return src(op, 0).ne[0] == 16 and src(op, 0).ne[1] == 1 and @mod(src(op, 0).ne[2], 128) == 0 and src(op, 4).ne[1] == 1;
        },
        // assumes d_inner % threads == 0
        ggml.impl.c.GGML_OP_SSM_CONV => return @mod(src(op, 0).ne[1], 128) == 0,
        ggml.impl.c.GGML_OP_CONT, ggml.impl.c.GGML_OP_SOFT_MAX => return true,
        ggml.impl.c.GGML_OP_SOFT_MAX_BACK => return ggml.impl.getOpParamsF32(op, 1) == 0.0,
        ggml.impl.c.GGML_OP_ROLL => return src(op, 0).type == ggml.impl.c.GGML_TYPE_F32,
        ggml.impl.c.GGML_OP_ROPE, ggml.impl.c.GGML_OP_ROPE_BACK => return src(op, 0).nb[0] == ggml.impl.c.ggml_type_size(src(op, 0).type) and ggml.impl.c.ggml_is_contiguous_2(src(op, 0)),
        ggml.impl.c.GGML_OP_IM2COL, ggml.impl.c.GGML_OP_IM2COL_3D, ggml.impl.c.GGML_OP_CONV_2D => return ggml.impl.c.ggml_is_contiguous(src(op, 0)) and ggml.impl.c.ggml_is_contiguous(src(op, 1)),
        ggml.impl.c.GGML_OP_CONV_3D => return (src(op, 0).type == ggml.impl.c.GGML_TYPE_F16 or src(op, 0).type == ggml.impl.c.GGML_TYPE_F32) and
            src(op, 1).type == ggml.impl.c.GGML_TYPE_F32 and op.type == ggml.impl.c.GGML_TYPE_F32 and
            ggml.impl.c.ggml_is_contiguous(src(op, 0)) and ggml.impl.c.ggml_is_contiguous(src(op, 1)) and ggml.impl.c.ggml_is_contiguous(op),
        ggml.impl.c.GGML_OP_CONV_2D_DW => return (src(op, 0).type == ggml.impl.c.GGML_TYPE_F16 or src(op, 0).type == ggml.impl.c.GGML_TYPE_F32) and
            src(op, 1).type == ggml.impl.c.GGML_TYPE_F32 and op.type == ggml.impl.c.GGML_TYPE_F32,
        ggml.impl.c.GGML_OP_CONV_TRANSPOSE_2D, ggml.impl.c.GGML_OP_POOL_1D, ggml.impl.c.GGML_OP_POOL_2D => return true,
        ggml.impl.c.GGML_OP_ACC => return ggml.impl.c.ggml_is_contiguous(src(op, 0)) and ggml.impl.c.ggml_is_contiguous(src(op, 1)),
        ggml.impl.c.GGML_OP_SUM => return ggml.impl.c.ggml_is_contiguously_allocated(src(op, 0)),
        ggml.impl.c.GGML_OP_TOP_K, ggml.impl.c.GGML_OP_ARGSORT => return src(op, 0).ne[0] <= std.math.maxInt(c_int),
        ggml.impl.c.GGML_OP_SUM_ROWS => return src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 and op.type == ggml.impl.c.GGML_TYPE_F32 and ggml.impl.c.ggml_is_contiguous_rows(src(op, 0)),
        ggml.impl.c.GGML_OP_MEAN => return src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 and op.type == ggml.impl.c.GGML_TYPE_F32 and ggml.impl.c.ggml_is_contiguous_rows(src(op, 0)),
        ggml.impl.c.GGML_OP_GROUP_NORM => return ggml.impl.c.ggml_is_contiguous(src(op, 0)),
        ggml.impl.c.GGML_OP_PAD => return true,
        ggml.impl.c.GGML_OP_UPSCALE,
        ggml.impl.c.GGML_OP_PAD_REFLECT_1D,
        ggml.impl.c.GGML_OP_ARANGE,
        ggml.impl.c.GGML_OP_TIMESTEP_EMBEDDING,
        ggml.impl.c.GGML_OP_LEAKY_RELU,
        ggml.impl.c.GGML_OP_RWKV_WKV6,
        ggml.impl.c.GGML_OP_GATED_LINEAR_ATTN,
        ggml.impl.c.GGML_OP_RWKV_WKV7,
        => return true,
        ggml.impl.c.GGML_OP_GATED_DELTA_NET => return true,
        ggml.impl.c.GGML_OP_DSV4_HC_COMB => return src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 and src(op, 1).type == ggml.impl.c.GGML_TYPE_F32 and
            src(op, 2).type == ggml.impl.c.GGML_TYPE_F32 and op.type == ggml.impl.c.GGML_TYPE_F32,
        ggml.impl.c.GGML_OP_DSV4_HC_PRE => return src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 and src(op, 1).type == ggml.impl.c.GGML_TYPE_F32 and
            op.type == ggml.impl.c.GGML_TYPE_F32,
        ggml.impl.c.GGML_OP_DSV4_HC_POST => return src(op, 0).type == ggml.impl.c.GGML_TYPE_F32 and src(op, 1).type == ggml.impl.c.GGML_TYPE_F32 and
            src(op, 2).type == ggml.impl.c.GGML_TYPE_F32 and (op.src[3] == null or src(op, 3).type == ggml.impl.c.GGML_TYPE_F32) and op.type == ggml.impl.c.GGML_TYPE_F32,
        ggml.impl.c.GGML_OP_FLASH_ATTN_EXT => return ggml.cuda.fattn.supported(dev_ctx.device, op),
        ggml.impl.c.GGML_OP_CROSS_ENTROPY_LOSS,
        ggml.impl.c.GGML_OP_CROSS_ENTROPY_LOSS_BACK,
        ggml.impl.c.GGML_OP_OPT_STEP_ADAMW,
        ggml.impl.c.GGML_OP_OPT_STEP_SGD,
        ggml.impl.c.GGML_OP_FILL,
        ggml.impl.c.GGML_OP_CUMSUM,
        ggml.impl.c.GGML_OP_TRI,
        ggml.impl.c.GGML_OP_DIAG,
        ggml.impl.c.GGML_OP_SOLVE_TRI,
        => return true,
        ggml.impl.c.GGML_OP_LIGHTNING_INDEXER => return ggml.cuda.lightning_indexer.supported(dev_ctx.device, op),
        ggml.impl.c.GGML_OP_TOP_K_MASK => return op.type == ggml.impl.c.GGML_TYPE_F16 or op.type == ggml.impl.c.GGML_TYPE_F32,
        else => return false,
    }
}
