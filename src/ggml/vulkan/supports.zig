//! Which ops the Vulkan device takes: the reference's predicate, op by op.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v1.0.0
//! (`b1b770d90`): `ggml_backend_vk_device_supports_op`. The device vtable
//! answers this AND-ed with `ops.ported`, so the scheduler sends nothing to
//! host code the port does not have yet.

const std = @import("std");
const ggml = @import("../module.zig");

fn s(op: *const ggml.impl.ggml_c.ggml_tensor, i: usize) *const ggml.impl.ggml_c.ggml_tensor {
    return op.src[i].?;
}

fn isAny(t: ggml.impl.ggml_c.ggml_type, comptime set: []const ggml.impl.ggml_c.ggml_type) bool {
    inline for (set) |x| if (t == x) return true;
    return false;
}

/// The quantized types every matmul and get_rows path has shaders for.
const quant_types = [_]ggml.impl.ggml_c.ggml_type{
    ggml.impl.ggml_c.GGML_TYPE_Q1_0,    ggml.impl.ggml_c.GGML_TYPE_Q2_0,   ggml.impl.ggml_c.GGML_TYPE_Q4_0,  ggml.impl.ggml_c.GGML_TYPE_Q4_1,    ggml.impl.ggml_c.GGML_TYPE_Q5_0,  ggml.impl.ggml_c.GGML_TYPE_Q5_1,   ggml.impl.ggml_c.GGML_TYPE_Q8_0,
    ggml.impl.ggml_c.GGML_TYPE_Q2_K,    ggml.impl.ggml_c.GGML_TYPE_Q3_K,   ggml.impl.ggml_c.GGML_TYPE_Q4_K,  ggml.impl.ggml_c.GGML_TYPE_Q5_K,    ggml.impl.ggml_c.GGML_TYPE_Q6_K,  ggml.impl.ggml_c.GGML_TYPE_IQ1_S,  ggml.impl.ggml_c.GGML_TYPE_IQ1_M,
    ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS, ggml.impl.ggml_c.GGML_TYPE_IQ2_XS, ggml.impl.ggml_c.GGML_TYPE_IQ2_S, ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS, ggml.impl.ggml_c.GGML_TYPE_IQ3_S, ggml.impl.ggml_c.GGML_TYPE_IQ4_XS, ggml.impl.ggml_c.GGML_TYPE_IQ4_NL,
    ggml.impl.ggml_c.GGML_TYPE_MXFP4,   ggml.impl.ggml_c.GGML_TYPE_NVFP4,  ggml.impl.ggml_c.GGML_TYPE_TQ1_0, ggml.impl.ggml_c.GGML_TYPE_TQ2_0,
};

/// The types a CPY converts f32 to and from.
const cpy_types = [_]ggml.impl.ggml_c.ggml_type{ ggml.impl.ggml_c.GGML_TYPE_Q1_0, ggml.impl.ggml_c.GGML_TYPE_Q2_0, ggml.impl.ggml_c.GGML_TYPE_Q4_0, ggml.impl.ggml_c.GGML_TYPE_Q4_1, ggml.impl.ggml_c.GGML_TYPE_Q5_0, ggml.impl.ggml_c.GGML_TYPE_Q5_1, ggml.impl.ggml_c.GGML_TYPE_Q8_0, ggml.impl.ggml_c.GGML_TYPE_IQ4_NL };

/// Ports `ggml_backend_vk_device_supports_op` (ggml-vulkan.cpp:15895 @b1b770d90).
pub fn supportsOp(dev_idx: usize, op: *const ggml.impl.ggml_c.ggml_tensor) bool {
    const device = ggml.vulkan.device.getDevice(dev_idx);

    const uses_bda = (op.op == ggml.impl.ggml_c.GGML_OP_IM2COL or op.op == ggml.impl.ggml_c.GGML_OP_IM2COL_3D) and device.shader_int64 and device.buffer_device_address;
    const sizeOk = struct {
        fn f(dv: *const ggml.vulkan.device.Device, bda: bool, tensor_size: usize) bool {
            if (tensor_size > dv.max_buffer_size) return false;
            // BDA im2col and 64-bit indexing ignore maxStorageBufferRange.
            if (!bda and !dv.shader_64b_indexing and tensor_size > dv.properties.limits.maxStorageBufferRange) return false;
            return true;
        }
    }.f;
    // non-compute ops are never executed here; their size is bounded by the buffer allocation limit
    const non_compute = ggml.impl.opIsEmpty(op.op);
    for (0..ggml.impl.ggml_c.GGML_MAX_SRC) |i| {
        if (op.src[i]) |src| if (!non_compute and !sizeOk(device, uses_bda, ggml.types.ggml_nbytes(src))) return false;
    }
    if (!non_compute and !sizeOk(device, uses_bda, ggml.types.ggml_nbytes(op))) return false;

    switch (op.op) {
        ggml.impl.ggml_c.GGML_OP_UNARY => switch (ggml.types.ggml_get_unary_op(op)) {
            ggml.impl.ggml_c.GGML_UNARY_OP_EXP,
            ggml.impl.ggml_c.GGML_UNARY_OP_EXPM1,
            ggml.impl.ggml_c.GGML_UNARY_OP_ELU,
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU,
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU_ERF,
            ggml.impl.ggml_c.GGML_UNARY_OP_GELU_QUICK,
            ggml.impl.ggml_c.GGML_UNARY_OP_SILU,
            ggml.impl.ggml_c.GGML_UNARY_OP_RELU,
            ggml.impl.ggml_c.GGML_UNARY_OP_XIELU,
            ggml.impl.ggml_c.GGML_UNARY_OP_NEG,
            ggml.impl.ggml_c.GGML_UNARY_OP_TANH,
            ggml.impl.ggml_c.GGML_UNARY_OP_SIGMOID,
            ggml.impl.ggml_c.GGML_UNARY_OP_HARDSIGMOID,
            ggml.impl.ggml_c.GGML_UNARY_OP_HARDSWISH,
            ggml.impl.ggml_c.GGML_UNARY_OP_ABS,
            ggml.impl.ggml_c.GGML_UNARY_OP_SOFTPLUS,
            ggml.impl.ggml_c.GGML_UNARY_OP_STEP,
            ggml.impl.ggml_c.GGML_UNARY_OP_ROUND,
            ggml.impl.ggml_c.GGML_UNARY_OP_CEIL,
            ggml.impl.ggml_c.GGML_UNARY_OP_FLOOR,
            ggml.impl.ggml_c.GGML_UNARY_OP_TRUNC,
            ggml.impl.ggml_c.GGML_UNARY_OP_SGN,
            => return (s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F16) and (op.type == ggml.impl.ggml_c.GGML_TYPE_F32 or op.type == ggml.impl.ggml_c.GGML_TYPE_F16) and s(op, 0).type == op.type,
            else => return false,
        },
        ggml.impl.ggml_c.GGML_OP_GLU => switch (ggml.types.ggml_get_glu_op(op)) {
            ggml.impl.ggml_c.GGML_GLU_OP_GEGLU, ggml.impl.ggml_c.GGML_GLU_OP_REGLU, ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU, ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_OAI, ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_ERF, ggml.impl.ggml_c.GGML_GLU_OP_GEGLU_QUICK, ggml.impl.ggml_c.GGML_GLU_OP_SWIGLU_CLAMP => {
                return (s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F16) and (op.type == ggml.impl.ggml_c.GGML_TYPE_F32 or op.type == ggml.impl.ggml_c.GGML_TYPE_F16) and
                    s(op, 0).type == op.type and (op.src[1] == null or s(op, 1).type == s(op, 0).type);
            },
            else => return false,
        },
        ggml.impl.ggml_c.GGML_OP_MUL_MAT, ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID => {
            const t0 = s(op, 0).type;
            if (op.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID) {
                if (!ggml.vulkan.ops.useMulMatVecIdTensor(op)) {
                    const ids = s(op, 2);
                    // Shaders store row IDs as uint16_t slot and token indices.
                    if (ids.ne[0] > @as(i64, std.math.maxInt(u16)) + 1 or ids.ne[1] > @as(i64, std.math.maxInt(u16)) + 1) return false;
                }
                const ti: usize = @intCast(t0);
                // Not enough shared memory for row_ids and the result tile: fall back to CPU
                if (!device.mul_mat_id_s[ti] and !device.mul_mat_id_m[ti] and !device.mul_mat_id_l[ti]) return false;
            }
            if (ggml.ops.ggml_prec_get_src(op, 1) == ggml.impl.ggml_c.GGML_PREC_BF16 or ggml.ops.ggml_prec_get_src(op, 1) == ggml.impl.ggml_c.GGML_PREC_F32) return false;
            if (!(isAny(t0, &.{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_BF16 }) or isAny(t0, &quant_types))) return false;
            // MUL_MAT broadcasts src0 over dim 3 like dim 2, MUL_MAT_ID needs the ids to match src1
            if (op.op == ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID and s(op, 2).ne[3] != s(op, 1).ne[3]) return false;
            if (!(ggml.vulkan.op_pipeline.dim01Contiguous(s(op, 0)) or isAny(t0, &.{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_BF16 })) or
                !(ggml.vulkan.op_pipeline.dim01Contiguous(s(op, 1)) or isAny(s(op, 1).type, &.{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16 }))) return false;
            // No bf16 x f16 shader, nor an fp16->bf16 copy shader.
            if (t0 == ggml.impl.ggml_c.GGML_TYPE_BF16 and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F16) return false;
            // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
            if (s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_BF16 and t0 != ggml.impl.ggml_c.GGML_TYPE_BF16 and op.op != ggml.impl.ggml_c.GGML_OP_MUL_MAT) return false;
            return true;
        },
        ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT => {
            // K/V row lists (ggml_flash_attn_ext_rows) are not supported
            if (op.src[5] != null) return false;
            const HSK = ggml.vulkan.ops.u32Of(s(op, 1).ne[0]);
            const HSV = ggml.vulkan.ops.u32Of(s(op, 2).ne[0]);
            if (HSK % 8 != 0 or HSV % 8 != 0) return false;
            if (op.src[4] != null and s(op, 4).type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            if (s(op, 0).type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            if (op.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            if (op.src[3] != null and s(op, 3).type != ggml.impl.ggml_c.GGML_TYPE_F16) return false;
            const kv = [_]ggml.impl.ggml_c.ggml_type{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_BF16, ggml.impl.ggml_c.GGML_TYPE_Q8_0, ggml.impl.ggml_c.GGML_TYPE_Q5_1, ggml.impl.ggml_c.GGML_TYPE_Q5_0, ggml.impl.ggml_c.GGML_TYPE_Q4_1, ggml.impl.ggml_c.GGML_TYPE_Q4_0, ggml.impl.ggml_c.GGML_TYPE_IQ4_NL };
            if (!isAny(s(op, 1).type, &kv) or !isAny(s(op, 2).type, &kv)) return false;
            if ((s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_BF16) != (s(op, 2).type == ggml.impl.ggml_c.GGML_TYPE_BF16)) return false;
            // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
            if (!device.coopmat2 and !(device.subgroup_shuffle and device.subgroup_vote)) return false;
            return true;
        },
        ggml.impl.ggml_c.GGML_OP_GET_ROWS => return isAny(s(op, 0).type, &.{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_BF16, ggml.impl.ggml_c.GGML_TYPE_I32 }) or isAny(s(op, 0).type, &quant_types),
        ggml.impl.ggml_c.GGML_OP_GET_ROWS_BACK => return op.type == ggml.impl.ggml_c.GGML_TYPE_F32 and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_SET_ROWS => {
            if ((s(op, 0).type != ggml.impl.ggml_c.GGML_TYPE_F32 and s(op, 0).type != ggml.impl.ggml_c.GGML_TYPE_F16) or (s(op, 1).type != ggml.impl.ggml_c.GGML_TYPE_I32 and s(op, 1).type != ggml.impl.ggml_c.GGML_TYPE_I64)) return false;
            return isAny(op.type, &.{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_BF16 }) or isAny(op.type, &cpy_types);
        },
        ggml.impl.ggml_c.GGML_OP_CONT, ggml.impl.ggml_c.GGML_OP_CPY, ggml.impl.ggml_c.GGML_OP_DUP => {
            const t0 = s(op, 0).type;
            const t1 = if (op.src[1]) |s1| s1.*.type else t0;
            if (t0 == ggml.impl.ggml_c.GGML_TYPE_F32 and (isAny(t1, &.{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_BF16 }) or isAny(t1, &cpy_types))) return true;
            if (t1 == ggml.impl.ggml_c.GGML_TYPE_F32 and (isAny(t0, &.{ ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_BF16 }) or isAny(t0, &cpy_types))) return true;
            if (t0 == ggml.impl.ggml_c.GGML_TYPE_F16 and t1 == ggml.impl.ggml_c.GGML_TYPE_F16) return true;
            if ((t0 == ggml.impl.ggml_c.GGML_TYPE_F32 and t1 == ggml.impl.ggml_c.GGML_TYPE_I32) or (t0 == ggml.impl.ggml_c.GGML_TYPE_I32 and t1 == ggml.impl.ggml_c.GGML_TYPE_F32)) return true;
            // A copy to the same type: not quantized, or quantized and
            // contiguous, through the f16 or f32 shaders.
            if (t0 == t1 and (!ggml.types.ggml_is_quantized(t0) or (ggml.types.ggml_is_contiguous(s(op, 0)) and ggml.types.ggml_is_contiguous(op))) and ggml.types.ggml_type_size(t0) % 2 == 0) return true;
            return false;
        },
        ggml.impl.ggml_c.GGML_OP_REPEAT => {
            const sz = ggml.types.ggml_type_size(op.type);
            return sz == ggml.types.ggml_type_size(s(op, 0).type) and (sz == @sizeOf(f32) or sz == 2);
        },
        ggml.impl.ggml_c.GGML_OP_REPEAT_BACK => return op.type == ggml.impl.ggml_c.GGML_TYPE_F32 and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_ROPE => return ggml.types.ggml_is_contiguous_rows(op) and ggml.types.ggml_is_contiguous_rows(s(op, 0)),
        ggml.impl.ggml_c.GGML_OP_ROPE_BACK, ggml.impl.ggml_c.GGML_OP_NONE, ggml.impl.ggml_c.GGML_OP_RESHAPE, ggml.impl.ggml_c.GGML_OP_VIEW, ggml.impl.ggml_c.GGML_OP_PERMUTE, ggml.impl.ggml_c.GGML_OP_TRANSPOSE, ggml.impl.ggml_c.GGML_OP_RMS_NORM => return true,
        ggml.impl.ggml_c.GGML_OP_GROUP_NORM => return ggml.types.ggml_is_contiguous(s(op, 0)),
        ggml.impl.ggml_c.GGML_OP_NORM, ggml.impl.ggml_c.GGML_OP_L2_NORM => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_ADD, ggml.impl.ggml_c.GGML_OP_SUB, ggml.impl.ggml_c.GGML_OP_MUL, ggml.impl.ggml_c.GGML_OP_DIV => return (s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F16) and (s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F16) and (op.type == ggml.impl.ggml_c.GGML_TYPE_F32 or op.type == ggml.impl.ggml_c.GGML_TYPE_F16),
        ggml.impl.ggml_c.GGML_OP_ADD_ID => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and s(op, 2).type == ggml.impl.ggml_c.GGML_TYPE_I32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_SILU_BACK, ggml.impl.ggml_c.GGML_OP_RMS_NORM_BACK => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_SQR, ggml.impl.ggml_c.GGML_OP_SQRT, ggml.impl.ggml_c.GGML_OP_SIN, ggml.impl.ggml_c.GGML_OP_COS, ggml.impl.ggml_c.GGML_OP_CLAMP, ggml.impl.ggml_c.GGML_OP_LEAKY_RELU, ggml.impl.ggml_c.GGML_OP_LOG, ggml.impl.ggml_c.GGML_OP_TRI, ggml.impl.ggml_c.GGML_OP_DIAG => {
            return (s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F16) and op.type == s(op, 0).type;
        },
        ggml.impl.ggml_c.GGML_OP_OPT_STEP_ADAMW, ggml.impl.ggml_c.GGML_OP_OPT_STEP_SGD => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_OUT_PROD => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.types.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_ARGSORT => {
            if (!ggml.types.ggml_is_contiguous(op) or !ggml.types.ggml_is_contiguous(s(op, 0))) return false;
            // pipeline_argsort_large_f32 requires vulkan memory model.
            if (device.vulkan_memory_model) return true;
            return op.ne[0] <= (@as(i64, 1) << @intCast(@min(device.max_workgroup_size_log2, ggml.vulkan.device.num_argsort_pipelines - 1)));
        },
        ggml.impl.ggml_c.GGML_OP_TOP_K => {
            if (!ggml.types.ggml_is_contiguous(op) or !ggml.types.ggml_is_contiguous(s(op, 0))) return false;
            // large k falls back to radix-select
            if (ggml.vulkan.ops_reduce.topkTournamentFits(device, ggml.vulkan.ops.u32Of(op.ne[0]))) return true;
            return device.pipeline_topk_radix_f32 != null;
        },
        ggml.impl.ggml_c.GGML_OP_UPSCALE => {
            if (op.op_params[0] & ggml.impl.ggml_c.GGML_SCALE_FLAG_ANTIALIAS != 0) {
                if (op.op_params[0] & 0xFF != ggml.impl.ggml_c.GGML_SCALE_MODE_BILINEAR) return false;
            }
            return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32;
        },
        ggml.impl.ggml_c.GGML_OP_ACC => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_SET => return s(op, 0).type == s(op, 1).type and s(op, 0).type == op.type and (s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_I32),
        ggml.impl.ggml_c.GGML_OP_CONCAT => return ggml.vulkan.op_pipeline.concatSupported(s(op, 0), s(op, 1), op),
        ggml.impl.ggml_c.GGML_OP_ARANGE => return op.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_FILL => return op.type == ggml.impl.ggml_c.GGML_TYPE_F32 or op.type == ggml.impl.ggml_c.GGML_TYPE_F16,
        ggml.impl.ggml_c.GGML_OP_SCALE => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_PAD, ggml.impl.ggml_c.GGML_OP_PAD_REFLECT_1D, ggml.impl.ggml_c.GGML_OP_ROLL => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and (op.src[1] == null or s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F16),
        ggml.impl.ggml_c.GGML_OP_SOFT_MAX_BACK => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.types.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_SUM => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.types.ggml_is_contiguously_allocated(s(op, 0)),
        ggml.impl.ggml_c.GGML_OP_SUM_ROWS, ggml.impl.ggml_c.GGML_OP_MEAN => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.types.ggml_is_contiguous_rows(s(op, 0)),
        ggml.impl.ggml_c.GGML_OP_CUMSUM => {
            if (device.subgroup_arithmetic and device.subgroup_require_full_support) return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.types.ggml_is_contiguous_rows(s(op, 0));
            return false;
        },
        ggml.impl.ggml_c.GGML_OP_DSV4_HC_COMB, ggml.impl.ggml_c.GGML_OP_DSV4_HC_PRE, ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST => {
            if (op.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            for (0..ggml.impl.ggml_c.GGML_MAX_SRC) |i| {
                if (op.src[i]) |src| if (src.*.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            }
            // hc is hardcoded to 4 in the shaders. ggml only constrains it
            // to 4 for COMB, so PRE/POST have to be checked here.
            if (op.op == ggml.impl.ggml_c.GGML_OP_DSV4_HC_PRE and s(op, 0).ne[1] != 4) return false;
            if (op.op == ggml.impl.ggml_c.GGML_OP_DSV4_HC_POST and s(op, 1).ne[1] != 4) return false;
            if (op.op == ggml.impl.ggml_c.GGML_OP_DSV4_HC_COMB) return device.pipeline_dsv4_hc_comb_f32 != null;
            return true;
        },
        ggml.impl.ggml_c.GGML_OP_SOLVE_TRI => {
            if (op.type != ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            const N = ggml.vulkan.ops.u32Of(s(op, 0).ne[0]);
            const K = ggml.vulkan.ops.u32Of(s(op, 1).ne[0]);
            // K dimension limited to workgroup size
            if (K > @as(u32, 1) << @intCast(device.max_workgroup_size_log2)) return false;
            const batch_N = device.properties.limits.maxComputeSharedMemorySize / ((N +% K) *% @sizeOf(f32));
            return batch_N != 0;
        },
        ggml.impl.ggml_c.GGML_OP_ARGMAX => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and
            ggml.types.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and
            ggml.types.ggml_are_same_shape(s(op, 0), s(op, 1)) and
            ggml.types.ggml_is_contiguous(op) and ggml.types.ggml_is_scalar(op) and op.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and ggml.types.ggml_is_scalar(s(op, 0)) and
            ggml.types.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and
            ggml.types.ggml_is_contiguous(s(op, 2)) and s(op, 2).type == ggml.impl.ggml_c.GGML_TYPE_F32 and
            ggml.types.ggml_are_same_shape(s(op, 1), s(op, 2)) and
            ggml.types.ggml_are_same_shape(s(op, 1), op) and
            ggml.types.ggml_is_contiguous(op) and op.type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_COUNT_EQUAL => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_I32 and ggml.types.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_I32,
        ggml.impl.ggml_c.GGML_OP_IM2COL => return ggml.types.ggml_is_contiguous(s(op, 1)) and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and (op.type == ggml.impl.ggml_c.GGML_TYPE_F32 or op.type == ggml.impl.ggml_c.GGML_TYPE_F16),
        ggml.impl.ggml_c.GGML_OP_IM2COL_3D => return s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and (op.type == ggml.impl.ggml_c.GGML_TYPE_F32 or op.type == ggml.impl.ggml_c.GGML_TYPE_F16),
        ggml.impl.ggml_c.GGML_OP_TIMESTEP_EMBEDDING => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_CONV_2D_DW => return (s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F16) and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_POOL_1D, ggml.impl.ggml_c.GGML_OP_POOL_2D => return ggml.types.ggml_is_contiguous(s(op, 0)) and s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        // all inputs are contiguous, see ggml.c
        ggml.impl.ggml_c.GGML_OP_RWKV_WKV6, ggml.impl.ggml_c.GGML_OP_RWKV_WKV7 => return true,
        // the shader block size is hardcoded to head_size 64
        ggml.impl.ggml_c.GGML_OP_GATED_LINEAR_ATTN => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32 and s(op, 0).ne[0] == 64,
        ggml.impl.ggml_c.GGML_OP_LIGHTNING_INDEXER => {
            const q = s(op, 0);
            const k = s(op, 1);
            const w = s(op, 2);
            const mk = s(op, 3);

            // the q/w/m types and the shape relationships between q, k, w, m and dst
            // are already asserted in ggml_lightning_indexer()
            if (!ggml.vulkan.device.lightningIndexerKTypeSupported(k.type) or !device.fp16) return false;

            // the shader block size is hardcoded to head size 128
            if (q.ne[0] != 128) return false;

            // the shader indexes the buffers by element stride, and is dispatched
            // without allow_misalign
            for ([_]*const ggml.impl.ggml_c.ggml_tensor{ q, k, w, mk, op }) |t| {
                const ts = ggml.types.ggml_type_size(t.type);
                if (t.nb[0] != ts or
                    (ggml.vulkan.buffer.tensorOffset(t) + t.view_offs) % device.properties.limits.minStorageBufferOffsetAlignment != 0)
                {
                    return false;
                }
                // the strides get scaled down from bytes, so the division must be exact
                for (1..ggml.impl.ggml_c.GGML_MAX_DIMS) |i| {
                    if (t.nb[i] % ts != 0) return false;
                }
            }
            return true;
        },
        ggml.impl.ggml_c.GGML_OP_GATED_DELTA_NET => {
            const S_v = ggml.vulkan.ops.u32Of(s(op, 2).ne[0]);
            if (S_v != 16 and S_v != 32 and S_v != 64 and S_v != 128) return false;
            for (0..6) |i| {
                if (op.src[i] == null or s(op, i).type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            }
            return op.type == ggml.impl.ggml_c.GGML_TYPE_F32;
        },
        ggml.impl.ggml_c.GGML_OP_SSM_SCAN => {
            for (0..6) |i| {
                if (op.src[i]) |src| if (ggml.types.ggml_is_quantized(src.*.type)) return false;
            }
            if (op.src[6] != null and s(op, 6).type != ggml.impl.ggml_c.GGML_TYPE_I32) return false;
            if (s(op, 0).type != ggml.impl.ggml_c.GGML_TYPE_F32 or op.type != ggml.impl.ggml_c.GGML_TYPE_F32) return false;
            const d_state = ggml.vulkan.ops.u32Of(s(op, 0).ne[0]);
            const head_dim = ggml.vulkan.ops.u32Of(s(op, 0).ne[1]);
            const is_mamba2 = op.src[3] != null and s(op, 3).nb[1] == @sizeOf(f32);
            if (!is_mamba2) return false;
            if ((d_state != 128 and d_state != 256) or head_dim % 16 != 0) return false;
            if (@as(usize, d_state) * @sizeOf(f32) > device.properties.limits.maxComputeSharedMemorySize) return false;
            if (!device.subgroup_basic) return false;
            return true;
        },
        ggml.impl.ggml_c.GGML_OP_SSM_CONV => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_1D => return s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32,
        ggml.impl.ggml_c.GGML_OP_COL2IM_1D => return isAny(s(op, 0).type, &.{ ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_F16, ggml.impl.ggml_c.GGML_TYPE_BF16 }) and op.type == s(op, 0).type and ggml.types.ggml_is_contiguous(s(op, 0)) and ggml.types.ggml_is_contiguous(op),
        ggml.impl.ggml_c.GGML_OP_CONV_2D, ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D => {
            const transpose = op.op == ggml.impl.ggml_c.GGML_OP_CONV_TRANSPOSE_2D;
            const cout = if (!transpose) s(op, 0).ne[3] else s(op, 0).ne[2];
            const cin = if (!transpose) s(op, 0).ne[2] else s(op, 0).ne[3];
            // Channel-contiguous format is not supported yet.
            return (s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F16) and
                (s(op, 0).nb[0] == @sizeOf(f32) or s(op, 0).nb[0] == @sizeOf(ggml.impl.ggml_c.ggml_fp16_t)) and
                s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32 and cout == op.ne[2] and cin == s(op, 1).ne[2] and
                ggml.types.ggml_is_contiguous(s(op, 0)) and ggml.types.ggml_is_contiguous(s(op, 1)) and ggml.types.ggml_is_contiguous(op);
        },
        ggml.impl.ggml_c.GGML_OP_CONV_3D => return (s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F32 or s(op, 0).type == ggml.impl.ggml_c.GGML_TYPE_F16) and s(op, 1).type == ggml.impl.ggml_c.GGML_TYPE_F32 and op.type == ggml.impl.ggml_c.GGML_TYPE_F32 and
            ggml.types.ggml_is_contiguous(s(op, 0)) and ggml.types.ggml_is_contiguous(s(op, 1)) and ggml.types.ggml_is_contiguous(op),
        else => return false,
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
