//! The CPU backend's per-type dispatch table: how to quantise a row of floats
//! into each type, and how to take a dot product against it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! the section beginning at line 214. Each declaration names the C it replaces
//! and the line it began at.
//!
//! # The kernels are still C
//!
//! Every `quantize_row_*` and `ggml_vec_dot_*` below lives in
//! `ggml-cpu/quants.c` and `ggml-cpu/arch/arm/quants.c`, which Stage 3 step 5
//! ports. Until then this file is the boundary: the table is Zig, the
//! functions it points at are not.
//!
//! They are declared by hand rather than reached through `@cImport` because
//! `quants.h` lives under `ggml/src/ggml-cpu/`, a directory the ported
//! module's include path deliberately does not carry -- the headers there
//! reach `<arm_neon.h>` the same way `ggml-impl.h` does.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const convert = @import("convert.zig");
const c = impl.c;

const Traits = c.ggml_type_traits_cpu;
const FromFloat = c.ggml_from_float_t;
const VecDot = c.ggml_vec_dot_t;

/// Ports the `__ARM_FEATURE_MATMUL_INT8` predicate the table branches on.
///
/// With i8mm the `q4_0`, `q4_1`, `q8_0`, `q4_K` and `q6_K` kernels take two
/// rows per call and `mul_mat` tiles accordingly. This build has no i8mm --
/// neither does `zig cc`'s, which is what compiles the kernels themselves, so
/// the two agree. They must: `nrows = 2` against a kernel built without the
/// instruction would read past the row it was handed.
const has_matmul_int8 = std.Target.aarch64.featureSetHas(builtin.cpu.features, .i8mm);

/// The number of rows a `vec_dot` for one of the i8mm-capable types takes per
/// call.
const mmla_rows: i64 = if (has_matmul_int8) 2 else 1;

// -----------------------------------------------------------------------------
// The still-C kernels
//
// `quants.c` and `arch/arm/quants.c` at v0.3.0. Signatures come from
// `ggml-cpu/quants.h`.

const quants = struct {
    extern fn quantize_row_q1_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q2_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q4_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q4_1(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q5_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q5_1(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q8_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q8_1(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_mxfp4(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_nvfp4(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q2_K(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q3_K(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q4_K(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q5_K(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q6_K(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q8_K(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_tq1_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_tq2_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_iq4_nl(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_iq4_xs(x: [*c]const f32, y: ?*anyopaque, k: i64) void;

    extern fn ggml_vec_dot_q1_0_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q2_0_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q4_0_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q4_1_q8_1(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q5_0_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q5_1_q8_1(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q8_0_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_mxfp4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_nvfp4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q2_K_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q3_K_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q4_K_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q5_K_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_q6_K_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_tq1_0_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_tq2_0_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq2_xxs_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq2_xs_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq2_s_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq3_xxs_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq3_s_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq1_s_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq1_m_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq4_nl_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_iq4_xs_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize, nrc: c_int) void;
};

/// The three float dot products, from `ggml-cpu/vec.cpp` (still C++).
///
/// Their first pointer argument is typed rather than `void *`, so the C casts
/// them to `ggml_vec_dot_t` when it stores them. That cast is `@ptrCast` here.
const vec = struct {
    extern fn ggml_vec_dot_f32(n: c_int, s: [*c]f32, bs: usize, x: [*c]const f32, bx: usize, y: [*c]const f32, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_f16(n: c_int, s: [*c]f32, bs: usize, x: [*c]c.ggml_fp16_t, bx: usize, y: [*c]c.ggml_fp16_t, by: usize, nrc: c_int) void;
    extern fn ggml_vec_dot_bf16(n: c_int, s: [*c]f32, bs: usize, x: [*c]c.ggml_bf16_t, bx: usize, y: [*c]c.ggml_bf16_t, by: usize, nrc: c_int) void;
};

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
) Traits {
    return .{
        .from_float = @ptrCast(@alignCast(from_float)),
        .vec_dot = @ptrCast(@alignCast(vec_dot)),
        .vec_dot_type = vec_dot_type,
        .nrows = nrows,
    };
}

/// Ports `type_traits_cpu` (ggml-cpu.c:214 @c1d0e7a00).
///
/// Read directly by `mul_mat`, `mul_mat_id` and `ggml_graph_plan` as well as
/// through `ggml_get_type_traits_cpu`, so it is `pub` rather than file-local.
/// Types absent from the C's initialiser list -- `I8`, `I16`, `I64`, `F64` and
/// the rest -- are all-zero here as they are there.
pub const table: [c.GGML_TYPE_COUNT]Traits = blk: {
    var t: [c.GGML_TYPE_COUNT]Traits = @splat(entry(null, null, c.GGML_TYPE_F32, 0));

    t[c.GGML_TYPE_F32] = entry(&convert.ggml_cpu_fp32_to_fp32, &vec.ggml_vec_dot_f32, c.GGML_TYPE_F32, 1);
    t[c.GGML_TYPE_F16] = entry(&convert.ggml_cpu_fp32_to_fp16, &vec.ggml_vec_dot_f16, c.GGML_TYPE_F16, 1);

    t[c.GGML_TYPE_Q1_0] = entry(&quants.quantize_row_q1_0, &quants.ggml_vec_dot_q1_0_q8_0, c.GGML_TYPE_Q8_0, 1);
    t[c.GGML_TYPE_Q2_0] = entry(&quants.quantize_row_q2_0, &quants.ggml_vec_dot_q2_0_q8_0, c.GGML_TYPE_Q8_0, 1);
    t[c.GGML_TYPE_Q4_0] = entry(&quants.quantize_row_q4_0, &quants.ggml_vec_dot_q4_0_q8_0, c.GGML_TYPE_Q8_0, mmla_rows);
    t[c.GGML_TYPE_Q4_1] = entry(&quants.quantize_row_q4_1, &quants.ggml_vec_dot_q4_1_q8_1, c.GGML_TYPE_Q8_1, mmla_rows);
    t[c.GGML_TYPE_Q5_0] = entry(&quants.quantize_row_q5_0, &quants.ggml_vec_dot_q5_0_q8_0, c.GGML_TYPE_Q8_0, 1);
    t[c.GGML_TYPE_Q5_1] = entry(&quants.quantize_row_q5_1, &quants.ggml_vec_dot_q5_1_q8_1, c.GGML_TYPE_Q8_1, 1);
    t[c.GGML_TYPE_Q8_0] = entry(&quants.quantize_row_q8_0, &quants.ggml_vec_dot_q8_0_q8_0, c.GGML_TYPE_Q8_0, mmla_rows);

    // No `vec_dot`: q8_1 is only ever a right-hand operand.
    t[c.GGML_TYPE_Q8_1] = entry(&quants.quantize_row_q8_1, null, c.GGML_TYPE_Q8_1, 1);

    t[c.GGML_TYPE_MXFP4] = entry(&quants.quantize_row_mxfp4, &quants.ggml_vec_dot_mxfp4_q8_0, c.GGML_TYPE_Q8_0, 1);
    t[c.GGML_TYPE_NVFP4] = entry(&quants.quantize_row_nvfp4, &quants.ggml_vec_dot_nvfp4_q8_0, c.GGML_TYPE_Q8_0, 1);

    t[c.GGML_TYPE_Q2_K] = entry(&quants.quantize_row_q2_K, &quants.ggml_vec_dot_q2_K_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_Q3_K] = entry(&quants.quantize_row_q3_K, &quants.ggml_vec_dot_q3_K_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_Q4_K] = entry(&quants.quantize_row_q4_K, &quants.ggml_vec_dot_q4_K_q8_K, c.GGML_TYPE_Q8_K, mmla_rows);
    t[c.GGML_TYPE_Q5_K] = entry(&quants.quantize_row_q5_K, &quants.ggml_vec_dot_q5_K_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_Q6_K] = entry(&quants.quantize_row_q6_K, &quants.ggml_vec_dot_q6_K_q8_K, c.GGML_TYPE_Q8_K, mmla_rows);

    // The i-quants have no `from_float` here. For iq2_xxs, iq2_xs, iq1_s and
    // iq1_m the C sets it to NULL outright; for iq3_xxs, iq3_s and iq2_s it
    // comments the line out, because those quantisers need the tables
    // `ggml_quantize_init` builds and so cannot be called through this table.
    t[c.GGML_TYPE_IQ2_XXS] = entry(null, &quants.ggml_vec_dot_iq2_xxs_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_IQ2_XS] = entry(null, &quants.ggml_vec_dot_iq2_xs_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_IQ3_XXS] = entry(null, &quants.ggml_vec_dot_iq3_xxs_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_IQ3_S] = entry(null, &quants.ggml_vec_dot_iq3_s_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_IQ2_S] = entry(null, &quants.ggml_vec_dot_iq2_s_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_IQ1_S] = entry(null, &quants.ggml_vec_dot_iq1_s_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_IQ1_M] = entry(null, &quants.ggml_vec_dot_iq1_m_q8_K, c.GGML_TYPE_Q8_K, 1);

    t[c.GGML_TYPE_IQ4_NL] = entry(&quants.quantize_row_iq4_nl, &quants.ggml_vec_dot_iq4_nl_q8_0, c.GGML_TYPE_Q8_0, 1);
    t[c.GGML_TYPE_IQ4_XS] = entry(&quants.quantize_row_iq4_xs, &quants.ggml_vec_dot_iq4_xs_q8_K, c.GGML_TYPE_Q8_K, 1);

    // q8_K is a right-hand operand only, and the C leaves `nrows` at zero
    // here rather than the 1 every other entry carries. Kept as written.
    t[c.GGML_TYPE_Q8_K] = entry(&quants.quantize_row_q8_K, null, c.GGML_TYPE_F32, 0);

    t[c.GGML_TYPE_BF16] = entry(&convert.ggml_cpu_fp32_to_bf16, &vec.ggml_vec_dot_bf16, c.GGML_TYPE_BF16, 1);

    t[c.GGML_TYPE_TQ1_0] = entry(&quants.quantize_row_tq1_0, &quants.ggml_vec_dot_tq1_0_q8_K, c.GGML_TYPE_Q8_K, 1);
    t[c.GGML_TYPE_TQ2_0] = entry(&quants.quantize_row_tq2_0, &quants.ggml_vec_dot_tq2_0_q8_K, c.GGML_TYPE_Q8_K, 1);

    t[c.GGML_TYPE_I32] = entry(&convert.ggml_cpu_fp32_to_i32, null, c.GGML_TYPE_F32, 0);

    break :blk t;
};

/// Ports `ggml_get_type_traits_cpu` (ggml-cpu.c:417 @c1d0e7a00).
///
/// Parameters:
/// - `@"type"`: the tensor type to look up.
///
/// Return: a borrowed pointer into `table`, valid for the life of the process.
pub export fn ggml_get_type_traits_cpu(@"type": c.enum_ggml_type) *const Traits {
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
    try std.testing.expect(table[c.GGML_TYPE_F32].vec_dot != null);
    try std.testing.expect(table[c.GGML_TYPE_Q4_K].vec_dot != null);
    try std.testing.expect(table[c.GGML_TYPE_I8].vec_dot == null);
    try std.testing.expect(table[c.GGML_TYPE_I8].from_float == null);
}

test "the right-hand-only types carry the C's nrows" {
    // q8_1 is 1 and q8_K is 0 in the C. They look like the same case and are
    // not, so both are asserted rather than assumed.
    try std.testing.expectEqual(@as(i64, 1), table[c.GGML_TYPE_Q8_1].nrows);
    try std.testing.expectEqual(@as(i64, 0), table[c.GGML_TYPE_Q8_K].nrows);
    try std.testing.expect(table[c.GGML_TYPE_Q8_1].vec_dot == null);
    try std.testing.expect(table[c.GGML_TYPE_Q8_K].vec_dot == null);
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
    try std.testing.expectEqual(mmla_rows, table[c.GGML_TYPE_Q4_0].nrows);
    try std.testing.expectEqual(@as(i64, 1), table[c.GGML_TYPE_Q5_0].nrows);
}
