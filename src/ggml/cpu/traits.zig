//! The CPU backend's per-type dispatch table: how to quantise a row of floats
//! into each type, and how to take a dot product against it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v1.0.0, `b1b770d90`),
//! the section beginning at line 218. Each declaration names the C it replaces
//! and the line it began at.
//!
//! # The kernels are still C
//!
//! Every `quantize_row_*` and `ggml_vec_dot_*` below lives in
//! `ggml-cpu/quants.c` and `ggml-cpu/arch/arm/quants.c`, ported in
//! `cpu/quants/`.
//!
//! They are declared by hand rather than reached through `@cImport` because
//! `quants.h` lives under `ggml/src/ggml-cpu/`, a directory the ported
//! module's include path deliberately does not carry -- the headers there
//! reach `<arm_neon.h>` the same way `ggml-impl.h` does.

const std = @import("std");
const ggml = @import("../module.zig");
const builtin = @import("builtin");

/// Ports the `__ARM_FEATURE_MATMUL_INT8` predicate the table branches on.
///
/// With i8mm the `q4_0`, `q4_1`, `q8_0`, `q4_K` and `q6_K` kernels take two
/// rows per call and `mul_mat` tiles accordingly. The baseline (M1) has no
/// i8mm; `-Dcpu=apple_m3` does. Whichever it is, this and the kernels'
/// two-row arms (`quants/arm/mmla.zig`) read the same `builtin.cpu`, so they
/// agree. They must: `nrows = 2` against a kernel with no two-row arm
/// computes the first row and leaves the second unwritten -- measured, every
/// second output `0x00000000` in `make ops-diff GATE_CPU=apple_m3` before
/// those arms were ported, with ReleaseFast having compiled out the
/// `nrc == 1` assertion that would have said so.
const has_matmul_int8 = std.Target.aarch64.featureSetHas(builtin.cpu.features, .i8mm);

/// The number of rows a `vec_dot` for one of the i8mm-capable types takes per
/// call.
const mmla_rows: i64 = if (has_matmul_int8) 2 else 1;

// -----------------------------------------------------------------------------
// The still-C kernels
//
// `quants.c` and `arch/arm/quants.c`. Signatures come from
// `ggml-cpu/quants.h`.

// -----------------------------------------------------------------------------
// The table

/// Builds one row of the table.
///
/// The C writes designated initialisers and lets the rest zero, which reads as
/// "unset" for `from_float` and `vec_dot` but as a real value for `nrows` --
/// `q8_1` and `q8_K` genuinely differ there. So every field is spelled out
/// here and the defaults are the C's zeros, not a guess.
///
/// Parameters:
/// - `from_float`: quantiser for a row of `f32`, or null if the type has none.
/// - `vec_dot`: dot product against a row of this type, or null.
/// - `vec_dot_type`: the type `vec_dot` expects its second operand in.
/// - `nrows`: rows consumed per `vec_dot` call.
///
/// Return: the entry, by value.
fn entry(
    from_float: ?*const anyopaque,
    vec_dot: ?*const anyopaque,
    vec_dot_type: c_uint,
    nrows: i64,
) ggml.impl.ggml_c.ggml_type_traits_cpu {
    return .{
        .from_float = @ptrCast(@alignCast(from_float)),
        .vec_dot = @ptrCast(@alignCast(vec_dot)),
        .vec_dot_type = vec_dot_type,
        .nrows = nrows,
    };
}

/// Ports `type_traits_cpu` (ggml-cpu.c:218 @b1b770d90).
///
/// Read directly by `mul_mat`, `mul_mat_id` and `ggml_graph_plan` as well as
/// through `ggml_get_type_traits_cpu`, so it is `pub` rather than file-local.
/// Types absent from the C's initialiser list -- `I8`, `I16`, `I64`, `F64` and
/// the rest -- are all-zero here as they are there.
pub const table: [ggml.impl.ggml_c.GGML_TYPE_COUNT]ggml.impl.ggml_c.ggml_type_traits_cpu = blk: {
    var t: [ggml.impl.ggml_c.GGML_TYPE_COUNT]ggml.impl.ggml_c.ggml_type_traits_cpu = @splat(entry(null, null, ggml.impl.ggml_c.GGML_TYPE_F32, 0));

    t[ggml.impl.ggml_c.GGML_TYPE_F32] = entry(&ggml.cpu.convert.ggml_cpu_fp32_to_fp32, &ggml.cpu.vec.ggml_vec_dot_f32, ggml.impl.ggml_c.GGML_TYPE_F32, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_F16] = entry(&ggml.cpu.convert.ggml_cpu_fp32_to_fp16, &ggml.cpu.vec.ggml_vec_dot_f16_f32, ggml.impl.ggml_c.GGML_TYPE_F32, 1);

    t[ggml.impl.ggml_c.GGML_TYPE_Q1_0] = entry(&ggml.cpu.quants.rows.quantize_row_q1_0, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_q1_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_Q2_0] = entry(&ggml.cpu.quants.rows.quantize_row_q2_0, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_q2_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_Q4_0] = entry(&ggml.cpu.quants.rows.quantize_row_q4_0, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_q4_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mmla_rows);
    t[ggml.impl.ggml_c.GGML_TYPE_Q4_1] = entry(&ggml.cpu.quants.rows.quantize_row_q4_1, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_q4_1_q8_1, ggml.impl.ggml_c.GGML_TYPE_Q8_1, mmla_rows);
    t[ggml.impl.ggml_c.GGML_TYPE_Q5_0] = entry(&ggml.cpu.quants.rows.quantize_row_q5_0, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_q5_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_Q5_1] = entry(&ggml.cpu.quants.rows.quantize_row_q5_1, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_q5_1_q8_1, ggml.impl.ggml_c.GGML_TYPE_Q8_1, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_Q8_0] = entry(&ggml.cpu.quants.arch.rows.quantize_row_q8_0, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_q8_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mmla_rows);

    // No `vec_dot`: q8_1 is only ever a right-hand operand.
    t[ggml.impl.ggml_c.GGML_TYPE_Q8_1] = entry(&ggml.cpu.quants.arch.rows.quantize_row_q8_1, null, ggml.impl.ggml_c.GGML_TYPE_Q8_1, 1);

    t[ggml.impl.ggml_c.GGML_TYPE_MXFP4] = entry(&ggml.cpu.quants.rows.quantize_row_mxfp4, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_mxfp4_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_NVFP4] = entry(&ggml.cpu.quants.rows.quantize_row_nvfp4, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_nvfp4_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_F8_E4M3] = entry(&ggml.cpu.quants.rows.quantize_row_f8_e4m3, &ggml.cpu.quants.legacy.ggml_vec_dot_f8_e4m3_f32, ggml.impl.ggml_c.GGML_TYPE_F32, 1);

    t[ggml.impl.ggml_c.GGML_TYPE_Q2_K] = entry(&ggml.cpu.quants.rows.quantize_row_q2_K, &ggml.cpu.quants.arch.k.ggml_vec_dot_q2_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_Q3_K] = entry(&ggml.cpu.quants.rows.quantize_row_q3_K, &ggml.cpu.quants.arch.k.ggml_vec_dot_q3_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_Q4_K] = entry(&ggml.cpu.quants.rows.quantize_row_q4_K, &ggml.cpu.quants.arch.k.ggml_vec_dot_q4_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, mmla_rows);
    t[ggml.impl.ggml_c.GGML_TYPE_Q5_K] = entry(&ggml.cpu.quants.rows.quantize_row_q5_K, &ggml.cpu.quants.arch.k.ggml_vec_dot_q5_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_Q6_K] = entry(&ggml.cpu.quants.rows.quantize_row_q6_K, &ggml.cpu.quants.arch.k.ggml_vec_dot_q6_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, mmla_rows);

    // The i-quants have no `from_float` here. For iq2_xxs, iq2_xs, iq1_s and
    // iq1_m the C sets it to NULL outright; for iq3_xxs, iq3_s and iq2_s it
    // comments the line out, because those quantisers need the tables
    // `ggml_quantize_init` builds and so cannot be called through this table.
    t[ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS] = entry(null, &ggml.cpu.quants.arch.iq.ggml_vec_dot_iq2_xxs_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_IQ2_XS] = entry(null, &ggml.cpu.quants.arch.iq.ggml_vec_dot_iq2_xs_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS] = entry(null, &ggml.cpu.quants.arch.iq.ggml_vec_dot_iq3_xxs_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_IQ3_S] = entry(null, &ggml.cpu.quants.arch.iq.ggml_vec_dot_iq3_s_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_IQ2_S] = entry(null, &ggml.cpu.quants.arch.iq.ggml_vec_dot_iq2_s_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_IQ1_S] = entry(null, &ggml.cpu.quants.arch.iq.ggml_vec_dot_iq1_s_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_IQ1_M] = entry(null, &ggml.cpu.quants.arch.iq.ggml_vec_dot_iq1_m_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);

    t[ggml.impl.ggml_c.GGML_TYPE_IQ4_NL] = entry(&ggml.cpu.quants.rows.quantize_row_iq4_nl, &ggml.cpu.quants.arch.legacy.ggml_vec_dot_iq4_nl_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_IQ4_XS] = entry(&ggml.cpu.quants.rows.quantize_row_iq4_xs, &ggml.cpu.quants.arch.iq.ggml_vec_dot_iq4_xs_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);

    // q8_K is a right-hand operand only, and the C leaves `nrows` at zero
    // here rather than the 1 every other entry carries. Kept as written.
    t[ggml.impl.ggml_c.GGML_TYPE_Q8_K] = entry(&ggml.cpu.quants.arch.rows.quantize_row_q8_K, null, ggml.impl.ggml_c.GGML_TYPE_F32, 0);

    t[ggml.impl.ggml_c.GGML_TYPE_BF16] = entry(&ggml.cpu.convert.ggml_cpu_fp32_to_bf16, &ggml.cpu.vec.ggml_vec_dot_bf16, ggml.impl.ggml_c.GGML_TYPE_BF16, 1);

    t[ggml.impl.ggml_c.GGML_TYPE_TQ1_0] = entry(&ggml.cpu.quants.rows.quantize_row_tq1_0, &ggml.cpu.quants.arch.ternary.ggml_vec_dot_tq1_0_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);
    t[ggml.impl.ggml_c.GGML_TYPE_TQ2_0] = entry(&ggml.cpu.quants.rows.quantize_row_tq2_0, &ggml.cpu.quants.arch.ternary.ggml_vec_dot_tq2_0_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, 1);

    t[ggml.impl.ggml_c.GGML_TYPE_I32] = entry(&ggml.cpu.convert.ggml_cpu_fp32_to_i32, null, ggml.impl.ggml_c.GGML_TYPE_F32, 0);

    break :blk t;
};

/// Ports `ggml_get_type_traits_cpu` (ggml-cpu.c:427 @b1b770d90).
///
/// Parameters:
/// - `@"type"`: the tensor type to look up.
///
/// Return: a borrowed pointer into `table`, valid for the life of the process.
pub fn ggml_get_type_traits_cpu(@"type": ggml.impl.ggml_c.enum_ggml_type) callconv(.c) *const ggml.impl.ggml_c.ggml_type_traits_cpu {
    return &table[@intCast(@"type")];
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the table covers every type the C lists" {
    // Types outside the C's initialiser list stay all-zero, and `mul_mat`
    // dereferences `vec_dot` without checking, so a stray entry is a crash.
    try std.testing.expect(table[ggml.impl.ggml_c.GGML_TYPE_F32].vec_dot != null);
    try std.testing.expect(table[ggml.impl.ggml_c.GGML_TYPE_Q4_K].vec_dot != null);
    try std.testing.expect(table[ggml.impl.ggml_c.GGML_TYPE_I8].vec_dot == null);
    try std.testing.expect(table[ggml.impl.ggml_c.GGML_TYPE_I8].from_float == null);
}

test "the right-hand-only types carry the C's nrows" {
    // q8_1 is 1 and q8_K is 0 in the C. They look like the same case and are
    // not, so both are asserted rather than assumed.
    try std.testing.expectEqual(@as(i64, 1), table[ggml.impl.ggml_c.GGML_TYPE_Q8_1].nrows);
    try std.testing.expectEqual(@as(i64, 0), table[ggml.impl.ggml_c.GGML_TYPE_Q8_K].nrows);
    try std.testing.expect(table[ggml.impl.ggml_c.GGML_TYPE_Q8_1].vec_dot == null);
    try std.testing.expect(table[ggml.impl.ggml_c.GGML_TYPE_Q8_K].vec_dot == null);
}

test "vec_dot_type points somewhere that can quantise" {
    // Every type with a `vec_dot` needs its `vec_dot_type` to have a
    // `from_float`, or `mul_mat` cannot build the right-hand operand.
    for (table, 0..) |t, i| {
        if (t.vec_dot == null) continue;
        const rhs = table[@intCast(t.vec_dot_type)];
        try std.testing.expect(rhs.from_float != null);
        _ = i;
    }
}

test "the i8mm row count matches the build" {
    try std.testing.expectEqual(@as(i64, if (has_matmul_int8) 2 else 1), mmla_rows);
    try std.testing.expectEqual(mmla_rows, table[ggml.impl.ggml_c.GGML_TYPE_Q4_0].nrows);
    try std.testing.expectEqual(@as(i64, 1), table[ggml.impl.ggml_c.GGML_TYPE_Q5_0].nrows);
}
