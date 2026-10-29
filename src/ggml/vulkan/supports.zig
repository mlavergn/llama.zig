//! Which ops the Vulkan device takes: the reference's predicate, op by op.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.1
//! (`b29c606e2`): `ggml_backend_vk_device_supports_op`. The device vtable
//! answers this AND-ed with `ops.ported`, so the scheduler sends nothing to
//! host code the port does not have yet.

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const F32 = c.GGML_TYPE_F32;
const F16 = c.GGML_TYPE_F16;
const BF16 = c.GGML_TYPE_BF16;
const I32 = c.GGML_TYPE_I32;

fn s(op: *const Tensor, i: usize) *const Tensor {
    return op.src[i].?;
}

fn isAny(t: c.ggml_type, comptime set: []const c.ggml_type) bool {
    inline for (set) |x| if (t == x) return true;
    return false;
}

/// The quantized types every matmul and get_rows path has shaders for.
const quant_types = [_]c.ggml_type{
    c.GGML_TYPE_Q1_0,    c.GGML_TYPE_Q2_0,   c.GGML_TYPE_Q4_0,  c.GGML_TYPE_Q4_1,    c.GGML_TYPE_Q5_0,  c.GGML_TYPE_Q5_1,   c.GGML_TYPE_Q8_0,
    c.GGML_TYPE_Q2_K,    c.GGML_TYPE_Q3_K,   c.GGML_TYPE_Q4_K,  c.GGML_TYPE_Q5_K,    c.GGML_TYPE_Q6_K,  c.GGML_TYPE_IQ1_S,  c.GGML_TYPE_IQ1_M,
    c.GGML_TYPE_IQ2_XXS, c.GGML_TYPE_IQ2_XS, c.GGML_TYPE_IQ2_S, c.GGML_TYPE_IQ3_XXS, c.GGML_TYPE_IQ3_S, c.GGML_TYPE_IQ4_XS, c.GGML_TYPE_IQ4_NL,
    c.GGML_TYPE_MXFP4,   c.GGML_TYPE_NVFP4,  c.GGML_TYPE_TQ1_0, c.GGML_TYPE_TQ2_0,
};

/// The types a CPY converts f32 to and from.
const cpy_types = [_]c.ggml_type{ c.GGML_TYPE_Q1_0, c.GGML_TYPE_Q2_0, c.GGML_TYPE_Q4_0, c.GGML_TYPE_Q4_1, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q5_1, c.GGML_TYPE_Q8_0, c.GGML_TYPE_IQ4_NL };

/// Ports `ggml_backend_vk_device_supports_op` (ggml-vulkan.cpp:19142 @b29c606e2).
pub fn supportsOp(dev_idx: usize, op: *const Tensor) bool {
    const device = m.device.getDevice(dev_idx);

    const uses_bda = (op.op == c.GGML_OP_IM2COL or op.op == c.GGML_OP_IM2COL_3D) and device.shader_int64 and device.buffer_device_address;
    const sizeOk = struct {
        fn f(dv: *const m.device.Device, bda: bool, tensor_size: usize) bool {
            if (tensor_size > dv.max_buffer_size) return false;
            // BDA im2col and 64-bit indexing ignore maxStorageBufferRange.
            if (!bda and !dv.shader_64b_indexing and tensor_size > dv.properties.limits.maxStorageBufferRange) return false;
            return true;
        }
    }.f;
    // reject any tensors larger than the max buffer size
    for (0..c.GGML_MAX_SRC) |i| {
        if (op.src[i]) |src| if (!sizeOk(device, uses_bda, c.ggml_nbytes(src))) return false;
    }
    if (!sizeOk(device, uses_bda, c.ggml_nbytes(op))) return false;

    switch (op.op) {
        c.GGML_OP_UNARY => switch (c.ggml_get_unary_op(op)) {
            c.GGML_UNARY_OP_EXP,
            c.GGML_UNARY_OP_EXPM1,
            c.GGML_UNARY_OP_ELU,
            c.GGML_UNARY_OP_GELU,
            c.GGML_UNARY_OP_GELU_ERF,
            c.GGML_UNARY_OP_GELU_QUICK,
            c.GGML_UNARY_OP_SILU,
            c.GGML_UNARY_OP_RELU,
            c.GGML_UNARY_OP_XIELU,
            c.GGML_UNARY_OP_NEG,
            c.GGML_UNARY_OP_TANH,
            c.GGML_UNARY_OP_SIGMOID,
            c.GGML_UNARY_OP_HARDSIGMOID,
            c.GGML_UNARY_OP_HARDSWISH,
            c.GGML_UNARY_OP_ABS,
            c.GGML_UNARY_OP_SOFTPLUS,
            c.GGML_UNARY_OP_STEP,
            c.GGML_UNARY_OP_ROUND,
            c.GGML_UNARY_OP_CEIL,
            c.GGML_UNARY_OP_FLOOR,
            c.GGML_UNARY_OP_TRUNC,
            c.GGML_UNARY_OP_SGN,
            => return (s(op, 0).type == F32 or s(op, 0).type == F16) and (op.type == F32 or op.type == F16) and s(op, 0).type == op.type,
            else => return false,
        },
        c.GGML_OP_GLU => switch (c.ggml_get_glu_op(op)) {
            c.GGML_GLU_OP_GEGLU, c.GGML_GLU_OP_REGLU, c.GGML_GLU_OP_SWIGLU, c.GGML_GLU_OP_SWIGLU_OAI, c.GGML_GLU_OP_GEGLU_ERF, c.GGML_GLU_OP_GEGLU_QUICK, c.GGML_GLU_OP_SWIGLU_CLAMP => {
                return (s(op, 0).type == F32 or s(op, 0).type == F16) and (op.type == F32 or op.type == F16) and
                    s(op, 0).type == op.type and (op.src[1] == null or s(op, 1).type == s(op, 0).type);
            },
            else => return false,
        },
        c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT_ID => {
            const t0 = s(op, 0).type;
            if (op.op == c.GGML_OP_MUL_MAT_ID) {
                const ti: usize = @intCast(t0);
                // Not enough shared memory for row_ids and the result tile: fall back to CPU
                if (!device.mul_mat_id_s[ti] and !device.mul_mat_id_m[ti] and !device.mul_mat_id_l[ti]) return false;
            }
            if (!(isAny(t0, &.{ F32, F16, BF16 }) or isAny(t0, &quant_types))) return false;
            const a = if (op.op == c.GGML_OP_MUL_MAT) s(op, 0) else s(op, 2);
            const b = s(op, 1);
            if (a.ne[3] != b.ne[3]) return false;
            if (!(m.op_pipeline.dim01Contiguous(s(op, 0)) or isAny(t0, &.{ F32, F16, BF16 })) or
                !(m.op_pipeline.dim01Contiguous(s(op, 1)) or isAny(s(op, 1).type, &.{ F32, F16 }))) return false;
            // No bf16 x f16 shader, nor an fp16->bf16 copy shader.
            if (t0 == BF16 and s(op, 1).type == F16) return false;
            return true;
        },
        c.GGML_OP_FLASH_ATTN_EXT => {
            const HSK = m.ops.u32Of(s(op, 1).ne[0]);
            const HSV = m.ops.u32Of(s(op, 2).ne[0]);
            if (HSK % 8 != 0 or HSV % 8 != 0) return false;
            if (op.src[4] != null and s(op, 4).type != F32) return false;
            if (s(op, 0).type != F32) return false;
            if (op.type != F32) return false;
            if (op.src[3] != null and s(op, 3).type != F16) return false;
            const kv = [_]c.ggml_type{ F32, F16, BF16, c.GGML_TYPE_Q8_0, c.GGML_TYPE_Q5_1, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q4_1, c.GGML_TYPE_Q4_0, c.GGML_TYPE_IQ4_NL };
            if (!isAny(s(op, 1).type, &kv) or !isAny(s(op, 2).type, &kv)) return false;
            if ((s(op, 1).type == BF16) != (s(op, 2).type == BF16)) return false;
            // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
            if (!device.coopmat2 and !(device.subgroup_shuffle and device.subgroup_vote)) return false;
            return true;
        },
        c.GGML_OP_GET_ROWS => return isAny(s(op, 0).type, &.{ F32, F16, BF16, I32 }) or isAny(s(op, 0).type, &quant_types),
        c.GGML_OP_GET_ROWS_BACK => return op.type == F32 and s(op, 0).type == F32,
        c.GGML_OP_SET_ROWS => {
            if ((s(op, 0).type != F32 and s(op, 0).type != F16) or (s(op, 1).type != I32 and s(op, 1).type != c.GGML_TYPE_I64)) return false;
            return isAny(op.type, &.{ F32, F16, BF16 }) or isAny(op.type, &cpy_types);
        },
        c.GGML_OP_CONT, c.GGML_OP_CPY, c.GGML_OP_DUP => {
            const t0 = s(op, 0).type;
            const t1 = if (op.src[1]) |s1| s1.*.type else t0;
            if (t0 == F32 and (isAny(t1, &.{ F32, F16, BF16 }) or isAny(t1, &cpy_types))) return true;
            if (t1 == F32 and (isAny(t0, &.{ F16, BF16 }) or isAny(t0, &cpy_types))) return true;
            if (t0 == F16 and t1 == F16) return true;
            if ((t0 == F32 and t1 == I32) or (t0 == I32 and t1 == F32)) return true;
            // A copy to the same type: not quantized, or quantized and
            // contiguous, through the f16 or f32 shaders.
            if (t0 == t1 and (!c.ggml_is_quantized(t0) or (c.ggml_is_contiguous(s(op, 0)) and c.ggml_is_contiguous(op))) and c.ggml_type_size(t0) % 2 == 0) return true;
            return false;
        },
        c.GGML_OP_REPEAT => {
            const sz = c.ggml_type_size(op.type);
            return sz == c.ggml_type_size(s(op, 0).type) and (sz == @sizeOf(f32) or sz == 2);
        },
        c.GGML_OP_REPEAT_BACK => return op.type == F32 and s(op, 0).type == F32,
        c.GGML_OP_ROPE => return c.ggml_is_contiguous_rows(op) and c.ggml_is_contiguous_rows(s(op, 0)),
        c.GGML_OP_ROPE_BACK, c.GGML_OP_NONE, c.GGML_OP_RESHAPE, c.GGML_OP_VIEW, c.GGML_OP_PERMUTE, c.GGML_OP_TRANSPOSE, c.GGML_OP_RMS_NORM => return true,
        c.GGML_OP_GROUP_NORM => return c.ggml_is_contiguous(s(op, 0)),
        c.GGML_OP_NORM, c.GGML_OP_L2_NORM => return s(op, 0).type == F32 and op.type == F32,
        c.GGML_OP_ADD, c.GGML_OP_SUB, c.GGML_OP_MUL, c.GGML_OP_DIV => return (s(op, 0).type == F32 or s(op, 0).type == F16) and (s(op, 1).type == F32 or s(op, 1).type == F16) and (op.type == F32 or op.type == F16),
        c.GGML_OP_ADD_ID => return s(op, 0).type == F32 and s(op, 1).type == F32 and s(op, 2).type == I32 and op.type == F32,
        c.GGML_OP_SILU_BACK, c.GGML_OP_RMS_NORM_BACK => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32,
        c.GGML_OP_SQR, c.GGML_OP_SQRT, c.GGML_OP_SIN, c.GGML_OP_COS, c.GGML_OP_CLAMP, c.GGML_OP_LEAKY_RELU, c.GGML_OP_LOG, c.GGML_OP_TRI, c.GGML_OP_DIAG => {
            return (s(op, 0).type == F32 or s(op, 0).type == F16) and op.type == s(op, 0).type;
        },
        c.GGML_OP_OPT_STEP_ADAMW, c.GGML_OP_OPT_STEP_SGD => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32,
        c.GGML_OP_OUT_PROD => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32 and c.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == F32 and op.type == F32,
        c.GGML_OP_ARGSORT => {
            if (!c.ggml_is_contiguous(op) or !c.ggml_is_contiguous(s(op, 0))) return false;
            // pipeline_argsort_large_f32 requires vulkan memory model.
            if (device.vulkan_memory_model) return true;
            return op.ne[0] <= (@as(i64, 1) << @intCast(device.max_workgroup_size_log2));
        },
        c.GGML_OP_TOP_K => {
            if (!c.ggml_is_contiguous(op) or !c.ggml_is_contiguous(s(op, 0))) return false;
            // large k falls back to radix-select
            if (m.ops_reduce.topkTournamentFits(device, m.ops.u32Of(op.ne[0]))) return true;
            return device.pipeline_topk_radix_f32 != null;
        },
        c.GGML_OP_UPSCALE => {
            if (op.op_params[0] & c.GGML_SCALE_FLAG_ANTIALIAS != 0) {
                if (op.op_params[0] & 0xFF != c.GGML_SCALE_MODE_BILINEAR) return false;
            }
            return s(op, 0).type == F32;
        },
        c.GGML_OP_ACC => return s(op, 0).type == F32 and s(op, 1).type == F32 and op.type == F32,
        c.GGML_OP_SET => return s(op, 0).type == s(op, 1).type and s(op, 0).type == op.type and (s(op, 0).type == F32 or s(op, 0).type == I32),
        c.GGML_OP_CONCAT => return m.op_pipeline.concatSupported(s(op, 0), s(op, 1), op),
        c.GGML_OP_ADD1 => {
            const t0 = s(op, 0).type;
            const t1 = s(op, 1).type;
            return (t0 == F32 and t1 == F32) or (t0 == F16 and t1 == F32) or (t0 == F16 and t1 == F16);
        },
        c.GGML_OP_ARANGE => return op.type == F32,
        c.GGML_OP_FILL => return op.type == F32 or op.type == F16,
        c.GGML_OP_SCALE => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32,
        c.GGML_OP_PAD, c.GGML_OP_PAD_REFLECT_1D, c.GGML_OP_ROLL => return s(op, 0).type == F32,
        c.GGML_OP_DIAG_MASK_INF => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32,
        c.GGML_OP_SOFT_MAX => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32 and (op.src[1] == null or s(op, 1).type == F32 or s(op, 1).type == F16),
        c.GGML_OP_SOFT_MAX_BACK => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32 and c.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == F32,
        c.GGML_OP_SUM, c.GGML_OP_SUM_ROWS, c.GGML_OP_MEAN => return s(op, 0).type == F32 and c.ggml_is_contiguous_rows(s(op, 0)),
        c.GGML_OP_CUMSUM => {
            if (device.subgroup_arithmetic and device.subgroup_require_full_support) return s(op, 0).type == F32 and c.ggml_is_contiguous_rows(s(op, 0));
            return false;
        },
        c.GGML_OP_DSV4_HC_COMB, c.GGML_OP_DSV4_HC_PRE, c.GGML_OP_DSV4_HC_POST => {
            if (op.type != F32) return false;
            for (0..c.GGML_MAX_SRC) |i| {
                if (op.src[i]) |src| if (src.*.type != F32) return false;
            }
            // hc is hardcoded to 4 in the shaders. ggml only constrains it
            // to 4 for COMB, so PRE/POST have to be checked here.
            if (op.op == c.GGML_OP_DSV4_HC_PRE and s(op, 0).ne[1] != 4) return false;
            if (op.op == c.GGML_OP_DSV4_HC_POST and s(op, 1).ne[1] != 4) return false;
            if (op.op == c.GGML_OP_DSV4_HC_COMB) return device.pipeline_dsv4_hc_comb_f32 != null;
            return true;
        },
        c.GGML_OP_SOLVE_TRI => {
            if (op.type != F32 or s(op, 0).type != F32) return false;
            const N = m.ops.u32Of(s(op, 0).ne[0]);
            const K = m.ops.u32Of(s(op, 1).ne[0]);
            // K dimension limited to workgroup size
            if (K > @as(u32, 1) << @intCast(device.max_workgroup_size_log2)) return false;
            const batch_N = device.properties.limits.maxComputeSharedMemorySize / ((N +% K) *% @sizeOf(f32));
            return batch_N != 0;
        },
        c.GGML_OP_ARGMAX => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32,
        c.GGML_OP_CROSS_ENTROPY_LOSS => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32 and
            c.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == F32 and
            c.ggml_are_same_shape(s(op, 0), s(op, 1)) and
            c.ggml_is_contiguous(op) and c.ggml_is_scalar(op) and op.type == F32,
        c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32 and c.ggml_is_scalar(s(op, 0)) and
            c.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == F32 and
            c.ggml_is_contiguous(s(op, 2)) and s(op, 2).type == F32 and
            c.ggml_are_same_shape(s(op, 1), s(op, 2)) and
            c.ggml_are_same_shape(s(op, 1), op) and
            c.ggml_is_contiguous(op) and op.type == F32,
        c.GGML_OP_COUNT_EQUAL => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == I32 and c.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == I32,
        c.GGML_OP_IM2COL => return c.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == F32 and (op.type == F32 or op.type == F16),
        c.GGML_OP_IM2COL_3D => return s(op, 1).type == F32 and (op.type == F32 or op.type == F16),
        c.GGML_OP_TIMESTEP_EMBEDDING => return s(op, 0).type == F32,
        c.GGML_OP_CONV_2D_DW => return (s(op, 0).type == F32 or s(op, 0).type == F16) and s(op, 1).type == F32,
        c.GGML_OP_POOL_1D, c.GGML_OP_POOL_2D => return c.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == F32,
        // all inputs are contiguous, see ggml.c
        c.GGML_OP_RWKV_WKV6, c.GGML_OP_RWKV_WKV7 => return true,
        // the shader block size is hardcoded to head_size 64
        c.GGML_OP_GATED_LINEAR_ATTN => return s(op, 0).type == F32 and op.type == F32 and s(op, 0).ne[0] == 64,
        c.GGML_OP_LIGHTNING_INDEXER => {
            const q = s(op, 0);
            const k = s(op, 1);
            const w = s(op, 2);
            const mk = s(op, 3);

            // the q/w/m types and the shape relationships between q, k, w, m and dst
            // are already asserted in ggml_lightning_indexer()
            if (!m.device.lightningIndexerKTypeSupported(k.type) or !device.fp16) return false;

            // the shader block size is hardcoded to head size 128
            if (q.ne[0] != 128) return false;

            // the shader indexes the buffers by element stride, and is dispatched
            // without allow_misalign
            for ([_]*const Tensor{ q, k, w, mk, op }) |t| {
                const ts = c.ggml_type_size(t.type);
                if (t.nb[0] != ts or
                    (m.buffer.tensorOffset(t) + t.view_offs) % device.properties.limits.minStorageBufferOffsetAlignment != 0)
                {
                    return false;
                }
                // the strides get scaled down from bytes, so the division must be exact
                for (1..c.GGML_MAX_DIMS) |i| {
                    if (t.nb[i] % ts != 0) return false;
                }
            }
            return true;
        },
        c.GGML_OP_GATED_DELTA_NET => {
            const S_v = m.ops.u32Of(s(op, 2).ne[0]);
            if (S_v != 16 and S_v != 32 and S_v != 64 and S_v != 128) return false;
            for (0..6) |i| {
                if (op.src[i] == null or s(op, i).type != F32) return false;
            }
            return op.type == F32;
        },
        c.GGML_OP_SSM_SCAN => {
            for (0..6) |i| {
                if (op.src[i]) |src| if (c.ggml_is_quantized(src.*.type)) return false;
            }
            if (op.src[6] != null and s(op, 6).type != I32) return false;
            if (s(op, 0).type != F32 or op.type != F32) return false;
            const d_state = m.ops.u32Of(s(op, 0).ne[0]);
            const head_dim = m.ops.u32Of(s(op, 0).ne[1]);
            const is_mamba2 = op.src[3] != null and s(op, 3).nb[1] == @sizeOf(f32);
            if (!is_mamba2) return false;
            if ((d_state != 128 and d_state != 256) or head_dim % 16 != 0) return false;
            if (@as(usize, d_state) * @sizeOf(f32) > device.properties.limits.maxComputeSharedMemorySize) return false;
            if (!device.subgroup_basic) return false;
            return true;
        },
        c.GGML_OP_SSM_CONV => return s(op, 0).type == F32,
        c.GGML_OP_CONV_TRANSPOSE_1D => return s(op, 0).type == F32 and s(op, 1).type == F32,
        c.GGML_OP_COL2IM_1D => return isAny(s(op, 0).type, &.{ F32, F16, BF16 }) and op.type == s(op, 0).type and c.ggml_is_contiguous(s(op, 0)) and c.ggml_is_contiguous(op),
        c.GGML_OP_CONV_2D, c.GGML_OP_CONV_TRANSPOSE_2D => {
            const transpose = op.op == c.GGML_OP_CONV_TRANSPOSE_2D;
            const cout = if (!transpose) s(op, 0).ne[3] else s(op, 0).ne[2];
            const cin = if (!transpose) s(op, 0).ne[2] else s(op, 0).ne[3];
            // Channel-contiguous format is not supported yet.
            return (s(op, 0).type == F32 or s(op, 0).type == F16) and
                (s(op, 0).nb[0] == @sizeOf(f32) or s(op, 0).nb[0] == @sizeOf(c.ggml_fp16_t)) and
                s(op, 1).type == F32 and op.type == F32 and cout == op.ne[2] and cin == s(op, 1).ne[2] and
                c.ggml_is_contiguous(s(op, 0)) and c.ggml_is_contiguous(s(op, 1)) and c.ggml_is_contiguous(op);
        },
        c.GGML_OP_CONV_3D => return (s(op, 0).type == F32 or s(op, 0).type == F16) and s(op, 1).type == F32 and op.type == F32 and
            c.ggml_is_contiguous(s(op, 0)) and c.ggml_is_contiguous(s(op, 1)) and c.ggml_is_contiguous(op),
        else => return false,
    }
}

extern fn log2f(x: f32) f32;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
