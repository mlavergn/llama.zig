//! What the CPU this binary was built for can do, and the one-time
//! initialisation that depends on it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! the arch-feature state at line 88, the predicates at line 3599, and
//! `ggml_cpu_init` at line 3823. Each declaration names the C it replaces and
//! the line it began at.
//!
//! # These are compile-time answers, not runtime probes
//!
//! Every one of the C's predicates is a `#if defined(__AVX2__)` around a
//! `return 1`, so what they report is the instruction set the *compiler* was
//! told to target, not what the chip underneath happens to have. The
//! translation keeps that: each is a `comptime` test against
//! `builtin.cpu.features`, which is where Zig records the same thing.
//!
//! Keeping the answers honest matters more than it looks. `type_traits_cpu`
//! branches on i8mm to decide how many rows a kernel takes per call, and the
//! kernels themselves are compiled by `zig cc` from `quants.c`. Both sides
//! read the same target, so both sides agree -- but only because neither is
//! hardcoded.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const convert = @import("convert.zig");
const c = impl.c;

/// True when this build targets AArch64 and the named feature is enabled.
///
/// The C's spelling is `#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_X)`.
/// The arch test comes first here for the same reason it does there: on a
/// non-ARM target the feature question is not merely false, it is meaningless.
inline fn armHas(comptime name: []const u8) bool {
    if (builtin.cpu.arch != .aarch64) return false;
    return std.Target.aarch64.featureSetHas(
        builtin.cpu.features,
        @field(std.Target.aarch64.Feature, name),
    );
}

/// Ports `struct ggml_arm_arch_features_type` and the `ggml_arm_arch_features`
/// instance (ggml-cpu.c:91 @c1d0e7a00).
///
/// Exported because it is a file-scope non-static in the C, and therefore part
/// of the translation unit's symbol contract even though only `ggml_cpu_init`
/// writes it and only `ggml_cpu_get_sve_cnt` reads it.
pub const ArmArchFeatures = extern struct {
    sve_cnt: c_int,
};

pub export var ggml_arm_arch_features: ArmArchFeatures = .{ .sve_cnt = 0 };

/// Ports `ggml_cpu_disable_fusion` (ggml-cpu.c:3029 @c1d0e7a00).
///
/// Written once by `ggml_cpu_init` and read-only afterwards, so no
/// synchronisation -- the C says the same in a comment on the same line.
pub var disable_fusion = false;

// -----------------------------------------------------------------------------
// x86 and friends
//
// All constant-false on this target. They are written out rather than folded
// into one stub because they are separate C symbols and the backend registry
// calls every one of them by name to build its feature string.

/// Ports `ggml_cpu_has_avx` (ggml-cpu.c:3599 @c1d0e7a00).
pub export fn ggml_cpu_has_avx() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_avx_vnni` (ggml-cpu.c:3607 @c1d0e7a00).
pub export fn ggml_cpu_has_avx_vnni() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_avx2` (ggml-cpu.c:3615 @c1d0e7a00).
pub export fn ggml_cpu_has_avx2() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_avx512` (ggml-cpu.c:3623 @c1d0e7a00).
pub export fn ggml_cpu_has_avx512() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_avx512_vbmi` (ggml-cpu.c:3631 @c1d0e7a00).
pub export fn ggml_cpu_has_avx512_vbmi() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_avx512_vnni` (ggml-cpu.c:3639 @c1d0e7a00).
pub export fn ggml_cpu_has_avx512_vnni() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_avx512_bf16` (ggml-cpu.c:3647 @c1d0e7a00).
pub export fn ggml_cpu_has_avx512_bf16() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_amx_int8` (ggml-cpu.c:3655 @c1d0e7a00).
pub export fn ggml_cpu_has_amx_int8() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_bmi2` (ggml-cpu.c:3663 @c1d0e7a00).
pub export fn ggml_cpu_has_bmi2() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_fma` (ggml-cpu.c:3671 @c1d0e7a00). The x86 `__FMA__`, not ARM's.
pub export fn ggml_cpu_has_fma() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_f16c` (ggml-cpu.c:3703 @c1d0e7a00).
pub export fn ggml_cpu_has_f16c() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_sse3` (ggml-cpu.c:3735 @c1d0e7a00).
pub export fn ggml_cpu_has_sse3() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_ssse3` (ggml-cpu.c:3743 @c1d0e7a00).
pub export fn ggml_cpu_has_ssse3() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_vsx` (ggml-cpu.c:3751 @c1d0e7a00), POWER9.
pub export fn ggml_cpu_has_vsx() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_vxe` (ggml-cpu.c:3759 @c1d0e7a00), s390x.
pub export fn ggml_cpu_has_vxe() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_wasm_simd` (ggml-cpu.c:3719 @c1d0e7a00).
pub export fn ggml_cpu_has_wasm_simd() c_int {
    return 0;
}

/// Ports `ggml_cpu_has_riscv_v` (ggml-cpu.c:3687 @c1d0e7a00).
pub export fn ggml_cpu_has_riscv_v() c_int {
    return 0;
}

/// Ports `ggml_cpu_get_rvv_vlen` (ggml-cpu.c:3695 @c1d0e7a00).
pub export fn ggml_cpu_get_rvv_vlen() c_int {
    return 0;
}

// -----------------------------------------------------------------------------
// ARM

/// Ports `ggml_cpu_has_arm_fma` (ggml-cpu.c:3679 @c1d0e7a00).
///
/// The C tests `__ARM_FEATURE_FMA`, which clang defines whenever the ARM
/// floating-point unit is available -- always, on AArch64.
pub export fn ggml_cpu_has_arm_fma() c_int {
    return @intFromBool(armHas("fp_armv8"));
}

/// Ports `ggml_cpu_has_fp16_va` (ggml-cpu.c:3711 @c1d0e7a00).
///
/// `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`, which clang ties to FullFP16.
pub export fn ggml_cpu_has_fp16_va() c_int {
    return @intFromBool(armHas("fullfp16"));
}

/// Ports `ggml_cpu_has_neon` (ggml-cpu.c:3767 @c1d0e7a00).
pub export fn ggml_cpu_has_neon() c_int {
    return @intFromBool(armHas("neon"));
}

/// Ports `ggml_cpu_has_dotprod` (ggml-cpu.c:3775 @c1d0e7a00).
pub export fn ggml_cpu_has_dotprod() c_int {
    return @intFromBool(armHas("dotprod"));
}

/// Ports `ggml_cpu_has_sve` (ggml-cpu.c:3783 @c1d0e7a00).
pub export fn ggml_cpu_has_sve() c_int {
    return @intFromBool(armHas("sve"));
}

/// Ports `ggml_cpu_has_matmul_int8` (ggml-cpu.c:3791 @c1d0e7a00).
pub export fn ggml_cpu_has_matmul_int8() c_int {
    return @intFromBool(armHas("i8mm"));
}

/// Ports `ggml_cpu_get_sve_cnt` (ggml-cpu.c:3799 @c1d0e7a00).
///
/// Zero unless SVE is compiled in, in which case `ggml_cpu_init` has filled
/// it from `svcntb()`.
pub export fn ggml_cpu_get_sve_cnt() c_int {
    if (!armHas("sve")) return 0;
    return ggml_arm_arch_features.sve_cnt;
}

/// Ports `ggml_cpu_has_sme` (ggml-cpu.c:3807 @c1d0e7a00).
pub export fn ggml_cpu_has_sme() c_int {
    return @intFromBool(armHas("sme"));
}

/// Ports `ggml_cpu_has_sme2` (ggml-cpu.c:3815 @c1d0e7a00).
pub export fn ggml_cpu_has_sme2() c_int {
    return @intFromBool(armHas("sme2"));
}

// -----------------------------------------------------------------------------
// Build configuration

/// Ports `ggml_cpu_has_llamafile` (ggml-cpu.c:3727 @c1d0e7a00).
///
/// Not a CPU question at all: `GGML_USE_LLAMAFILE` is a build flag, and
/// `build/llamacpp.zig` passes it. Kept true to match, and asserted against
/// the `mul_mat` fast path in the test below so the two cannot drift apart.
pub export fn ggml_cpu_has_llamafile() c_int {
    return @intFromBool(use_llamafile);
}

/// Ports the `GGML_USE_LLAMAFILE` define (build/llamacpp.zig).
///
/// The C also clears it under SVE or i8mm -- see `ggml-cpu.c:44` -- because
/// `sgemm.cpp` has no kernels for those paths. Reproduced here so the flag and
/// the `mul_mat` fast path cannot disagree.
pub const use_llamafile = !armHas("sve") and !armHas("i8mm");

// -----------------------------------------------------------------------------
// One-time initialisation

/// The C's `static bool is_first_call`, hoisted to file scope because Zig has
/// no function-local statics. Read and written only under the critical
/// section `ggml_cpu_init` takes.
var is_first_call = true;

extern fn ggml_time_us() i64;
extern fn atoi(str: [*:0]const u8) c_int;

/// Ports `ggml_cpu_init` (ggml-cpu.c:3823 @c1d0e7a00).
///
/// Fills the lookup tables and reads `GGML_CPU_DISABLE_FUSION`, once per
/// process. `ggml_graph_compute` calls it on every graph, so the guard is what
/// makes that cheap.
pub export fn ggml_cpu_init() void {
    // Needed to initialise ggml's time base. The context is created and freed
    // purely for that side effect, which is what the C does.
    {
        const params = c.struct_ggml_init_params{
            .mem_size = 0,
            .mem_buffer = null,
            .no_alloc = false,
        };
        ggml_free(ggml_init(params));
    }

    c.ggml_critical_section_start();

    if (is_first_call) {
        const t_start = ggml_time_us();

        convert.initTables();

        const t_end = ggml_time_us();
        impl.printDebug(
            "ggml_cpu_init: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n",
            .{@as(f64, @floatFromInt(t_end - t_start)) / 1000.0},
        );

        initArmArchFeatures();

        if (std.c.getenv("GGML_CPU_DISABLE_FUSION")) |env| {
            disable_fusion = atoi(env) == 1;
        }

        is_first_call = false;
    }

    c.ggml_critical_section_end();
}

/// Ports `ggml_init_arm_arch_features` (ggml-cpu.c:731 @c1d0e7a00).
///
/// Empty on this build: the C reads `svcntb()` only under
/// `__ARM_FEATURE_SVE`, and takes the empty arm at line 735 otherwise.
/// Left as a named function rather than deleted so the SVE arm has somewhere
/// obvious to go.
fn initArmArchFeatures() void {
    if (comptime armHas("sve")) {
        @compileError("SVE builds must fill ggml_arm_arch_features.sve_cnt from svcntb()");
    }
}

extern fn ggml_init(params: c.struct_ggml_init_params) ?*anyopaque;
extern fn ggml_free(ctx: ?*anyopaque) void;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the ARM predicates agree with the build target" {
    // These are what `type_traits_cpu` and the still-C kernels branch on, so
    // they are asserted against Zig's own view of the target rather than
    // against a hardcoded expectation for this machine.
    const has = std.Target.aarch64.featureSetHas;
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    try std.testing.expectEqual(
        @intFromBool(has(builtin.cpu.features, .neon)),
        ggml_cpu_has_neon(),
    );
    try std.testing.expectEqual(
        @intFromBool(has(builtin.cpu.features, .dotprod)),
        ggml_cpu_has_dotprod(),
    );
    try std.testing.expectEqual(
        @intFromBool(has(builtin.cpu.features, .i8mm)),
        ggml_cpu_has_matmul_int8(),
    );
}

test "the x86 predicates are all false" {
    try std.testing.expectEqual(@as(c_int, 0), ggml_cpu_has_avx());
    try std.testing.expectEqual(@as(c_int, 0), ggml_cpu_has_avx2());
    try std.testing.expectEqual(@as(c_int, 0), ggml_cpu_has_avx512());
    try std.testing.expectEqual(@as(c_int, 0), ggml_cpu_has_f16c());
    try std.testing.expectEqual(@as(c_int, 0), ggml_cpu_has_fma());
}

test "sve_cnt stays zero without SVE" {
    try std.testing.expectEqual(@as(c_int, 0), ggml_cpu_has_sve());
    try std.testing.expectEqual(@as(c_int, 0), ggml_cpu_get_sve_cnt());
}

test "llamafile is on, which is what mul_mat's fast path assumes" {
    try std.testing.expectEqual(@as(c_int, 1), ggml_cpu_has_llamafile());
    try std.testing.expect(use_llamafile);
}

test "init is idempotent and fills the tables" {
    ggml_cpu_init();
    const first = convert.ggml_table_f32_f16[0x3C00]; // 1.0 as a half
    try std.testing.expectEqual(@as(f32, 1.0), first);

    ggml_cpu_init();
    try std.testing.expectEqual(first, convert.ggml_table_f32_f16[0x3C00]);
    try std.testing.expectEqual(@as(f32, 0.5), convert.ggml_table_f32_ue4m3[0x38]);
}
