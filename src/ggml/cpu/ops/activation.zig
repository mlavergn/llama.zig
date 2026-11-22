//! `unary`, `glu`, `silu_back` and `leaky_relu`: the activations `ops.cpp`
//! keeps for itself, and the two dispatchers that route every unary and
//! gated-linear-unit op.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v1.0.0 (`b1b770d90`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # What lives here and what does not
//!
//! The C splits the unary ops across two translation units. `gelu`,
//! `gelu_erf`, `gelu_quick` and `silu` are here, because they call `vec.h`
//! kernels with lookup tables or NEON arms; the other eighteen are
//! `unary-ops.cpp`'s, already `cpu/unary_ops.zig`, and `ggml_compute_forward_unary`
//! reaches them across the C ABI exactly as the C++ does. `fill` and `tri` sit
//! between `gelu` and `gelu_erf` in the C but belong to `sort.zig`'s family in
//! this split, so they are not here.
//!
//! # One kernel body, many entry points
//!
//! The C writes each activation out twice, `_f32` and `_f16`, and the copies
//! differ only in the element type and the `ggml_vec_*` they call — verified
//! line by line, not assumed. The unary four collapse to `unaryRows` and the
//! six gated units to `gluRows`, each over a comptime element type and kernel.
//! `swiglu_oai` is the exception: it has no `vec.h` kernel and only an `f32`
//! variant, so it is written out.
//!
//! # `NDEBUG`
//!
//! Every kernel ends with an `#ifndef NDEBUG` loop asserting the output is
//! finite. This target defines `NDEBUG`, so those loops are dead and not
//! ported; the plain `assert`s at the top are dead for the same reason. Only
//! `GGML_ASSERT` survives into the build, and only that is ported.
//!
//! # Loop index names
//!
//! `i1`, `i2`, `i3` are Zig integer type names. Renamed `j1`, `j2`, `j3`,
//! digit for digit.

const std = @import("std");
const ggml = @import("../../module.zig");
const stdlib = @import("stdlib");

// -----------------------------------------------------------------------------
// The unary kernels `unary-ops.cpp` provides
//
// Ported in `cpu/unary_ops.zig`. Declared rather than imported so the
// dispatcher below reads like the C's, which calls them across a
// translation-unit boundary too.

// -----------------------------------------------------------------------------
// The `vec.h` kernels, at one signature
//
// Most of `vecinline.zig` takes an `i64` count; the four `vec.cpp` exports take
// a `c_int`. These adapt the latter so `unaryRows` and `gluRows` can take any
// of them as one comptime parameter.

fn siluF32(n: i64, y: [*]f32, x: [*]const f32) void {
    ggml.cpu.vec.ggml_vec_silu_f32(@intCast(n), y, x);
}

fn swigluF32(n: i64, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    ggml.cpu.vec.ggml_vec_swiglu_f32(@intCast(n), y, x, g);
}

fn geluQuickF32(n: i64, y: [*]f32, x: [*]const f32) void {
    ggml.cpu.vec.ggml_vec_gelu_quick_f32(@intCast(n), y, x);
}

fn gegluQuickF32(n: i64, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    ggml.cpu.vec.ggml_vec_geglu_quick_f32(@intCast(n), y, x, g);
}

// -----------------------------------------------------------------------------
// gelu, gelu_erf, gelu_quick, silu

/// Ports `ggml_compute_forward_gelu_f32` and `ggml_compute_forward_gelu_f16`
/// (ops.cpp:2055, 2102 @b1b770d90), and with them
/// `ggml_compute_forward_gelu_erf_f32`, `ggml_compute_forward_gelu_erf_f16`,
/// `ggml_compute_forward_gelu_quick_f32`,
/// `ggml_compute_forward_gelu_quick_f16`, `ggml_compute_forward_silu_f32`
/// and `ggml_compute_forward_silu_f16`
/// (ops.cpp:2295, 2342, 2424, 2471, 2553, 2600 @b1b770d90), which are the
/// same body over a different `ggml_vec_*`, and
/// `ggml_compute_forward_unary_bf16_rows` (ops.cpp:2015 @b1b770d90), the C's
/// template for the BF16 arms, whose body is this one over `ggml_bf16_t`.
///
/// Rows are split across threads; within a row the kernel runs over `nc`
/// contiguous elements.
fn unaryRows(
    comptime T: type,
    comptime kernel: anytype,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    const l = ggml.cpu.defs.UnaryLocals.of(src0, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nc: i64 = @as(c_int, @truncate(src0.ne[0]));
    const nr: i64 = @as(c_int, @truncate(ggml.impl.c.ggml_nrows(src0)));

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j3 = @divTrunc(ir, l.ne02 * l.ne01);
        const j2 = @divTrunc(ir - j3 * l.ne02 * l.ne01, l.ne01);
        const j1 = ir - j3 * l.ne02 * l.ne01 - j2 * l.ne01;

        const y: [*]T = @ptrCast(@alignCast(dd + @as(usize, @intCast(j3)) * l.nb3 + @as(usize, @intCast(j2)) * l.nb2 + @as(usize, @intCast(j1)) * l.nb1));
        const x: [*]const T = @ptrCast(@alignCast(s0 + @as(usize, @intCast(j3)) * l.nb03 + @as(usize, @intCast(j2)) * l.nb02 + @as(usize, @intCast(j1)) * l.nb01));
        kernel(nc, y, x);
    }
}

/// The type switch every activation dispatcher opens with: `f32`, `f16` or
/// `bf16`, and `GGML_ABORT` on anything else.
fn unaryByType(
    comptime k32: anytype,
    comptime k16: anytype,
    comptime kb16: anytype,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.c.GGML_TYPE_F32 => unaryRows(f32, k32, params, dst),
        ggml.impl.c.GGML_TYPE_F16 => unaryRows(ggml.impl.c.ggml_fp16_t, k16, params, dst),
        ggml.impl.c.GGML_TYPE_BF16 => unaryRows(ggml.impl.c.ggml_bf16_t, kb16, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_gelu` (ops.cpp:2156 @b1b770d90) and
/// `ggml_compute_forward_gelu_bf16` (ops.cpp:2150 @b1b770d90).
fn gelu(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    unaryByType(ggml.cpu.ops.vecinline.gelu_f32, ggml.cpu.ops.vecinline.gelu_f16, ggml.cpu.ops.vecinline.gelu_bf16, params, dst);
}

/// Ports `ggml_compute_forward_gelu_erf` (ops.cpp:2396 @b1b770d90) and
/// `ggml_compute_forward_gelu_erf_bf16` (ops.cpp:2390 @b1b770d90).
fn geluErf(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    unaryByType(ggml.cpu.ops.vecinline.gelu_erf_f32, ggml.cpu.ops.vecinline.gelu_erf_f16, ggml.cpu.ops.vecinline.gelu_erf_bf16, params, dst);
}

/// Ports `ggml_compute_forward_gelu_quick` (ops.cpp:2525 @b1b770d90) and
/// `ggml_compute_forward_gelu_quick_bf16` (ops.cpp:2519 @b1b770d90).
fn geluQuick(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    unaryByType(geluQuickF32, ggml.cpu.ops.vecinline.gelu_quick_f16, ggml.cpu.ops.vecinline.gelu_quick_bf16, params, dst);
}

/// Ports `ggml_compute_forward_silu` (ops.cpp:2654 @b1b770d90) and
/// `ggml_compute_forward_silu_bf16` (ops.cpp:2648 @b1b770d90).
///
/// The `f32` arm reaches `ggml_vec_silu_f32`, exported by `cpu/vec.zig` with
/// its NEON polynomial `exp`; the `f16` and `bf16` arms are the scalar
/// `vec.h` inlines with libm's `expf`. They are not the same function of
/// `x`, as in the C.
fn silu(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    unaryByType(siluF32, ggml.cpu.ops.vecinline.silu_f16, ggml.cpu.ops.vecinline.silu_bf16, params, dst);
}

// -----------------------------------------------------------------------------
// leaky_relu

/// Ports `ggml_compute_forward_leaky_relu_f32` and
/// `ggml_compute_forward_leaky_relu_f16` (ops.cpp:2681, 2711 @b1b770d90).
///
/// Single-threaded: every thread but the first returns at once. Rows are
/// addressed by `nb[1]` alone, the C assuming (in an `assert` compiled out
/// here) that both tensors are contiguous in their upper dimensions.
fn leakyReluRows(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    if (params.ith != 0) return;

    const n: i64 = @as(c_int, @truncate(ggml.impl.c.ggml_nrows(src0)));
    const nc: i64 = @as(c_int, @truncate(src0.ne[0]));

    const negative_slope: f32 = ggml.impl.getOpParamsF32(dst, 0);

    const s0: [*]u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var i: i64 = 0;
    while (i < n) : (i += 1) {
        const y: [*]T = @ptrCast(@alignCast(dd + @as(usize, @intCast(i)) * dst.nb[1]));
        const x: [*]const T = @ptrCast(@alignCast(s0 + @as(usize, @intCast(i)) * src0.nb[1]));
        if (T == f32) ggml.cpu.ops.vecinline.leaky_relu_f32(nc, y, x, negative_slope) else ggml.cpu.ops.vecinline.leaky_relu_f16(nc, y, x, negative_slope);
    }
}

/// Ports `ggml_compute_forward_leaky_relu` (ops.cpp:2741 @b1b770d90).
pub export fn ggml_compute_forward_leaky_relu(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.c.GGML_TYPE_F32 => leakyReluRows(f32, params, dst),
        ggml.impl.c.GGML_TYPE_F16 => leakyReluRows(ggml.impl.c.ggml_fp16_t, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// silu_back

/// Ports `ggml_compute_forward_silu_back_f32` and
/// `ggml_compute_forward_silu_back_f16` (ops.cpp:2765, 2808 @b1b770d90).
///
/// Note the operand order: `src[0]` is the incoming gradient and `src[1]` the
/// forward input, and the row count comes from `src[1]`.
fn siluBackRows(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    const grad = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nc: i64 = @as(c_int, @truncate(src1.ne[0]));
    const nr: i64 = @as(c_int, @truncate(ggml.impl.c.ggml_nrows(src1)));

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const dd: [*]u8 = @ptrCast(dst.data.?);
    const s1: [*]u8 = @ptrCast(src1.data.?);
    const gd: [*]u8 = @ptrCast(grad.data.?);

    var j1: i64 = ir0;
    while (j1 < ir1) : (j1 += 1) {
        const row: usize = @intCast(j1);
        const dx: [*]T = @ptrCast(@alignCast(dd + row * dst.nb[1]));
        const x: [*]const T = @ptrCast(@alignCast(s1 + row * src1.nb[1]));
        const dy: [*]const T = @ptrCast(@alignCast(gd + row * grad.nb[1]));
        if (T == f32) ggml.cpu.ops.vecinline.silu_backward_f32(nc, dx, x, dy) else ggml.cpu.ops.vecinline.silu_backward_f16(nc, dx, x, dy);
    }
}

/// Ports `ggml_compute_forward_silu_back` (ops.cpp:2852 @b1b770d90).
pub export fn ggml_compute_forward_silu_back(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.c.GGML_TYPE_F32 => siluBackRows(f32, params, dst),
        ggml.impl.c.GGML_TYPE_F16 => siluBackRows(ggml.impl.c.ggml_fp16_t, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// The gated linear units

/// The row geometry every GLU kernel opens with: where the two halves come
/// from, how wide the output is, and which rows this thread owns.
///
/// With one input the gate is the other half of the same row, and
/// `op_params[1]` (`swapped`) says which half is which. With two the halves
/// are separate tensors and `swapped` is ignored.
const GluRows = struct {
    src0_d: [*]u8,
    src1_d: [*]u8,
    src0_o: usize,
    src1_o: usize,
    has_src1: bool,
    swapped: bool,
    nc: i64,
    ir0: i64,
    ir1: i64,

    fn of(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) GluRows {
        const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
        const src1: ?*ggml.impl.c.ggml_tensor = dst.src[1];
        const src0_d: [*]u8 = @ptrCast(src0.data.?);
        const src1_d: [*]u8 = @ptrCast((if (src1) |s| s.data else src0.data).?);
        const src0_o = src0.nb[1];
        const src1_o = if (src1) |s| s.nb[1] else src0.nb[1];

        ggml.impl.assert(ggml.impl.c.ggml_is_contiguous_1(src0), "ggml_is_contiguous_1(src0)");
        ggml.impl.assert(ggml.impl.c.ggml_is_contiguous_1(dst), "ggml_is_contiguous_1(dst)");

        if (src1) |s| {
            ggml.impl.assert(ggml.impl.c.ggml_is_contiguous_1(s), "ggml_is_contiguous_1(src1)");
            ggml.impl.assert(src0.type == s.type, "src0->type == src1->type");
        }

        const ith: i64 = params.ith;
        const nth: i64 = params.nth;

        const nc: i64 = @as(c_int, @truncate(if (src1 != null) src0.ne[0] else @divTrunc(src0.ne[0], 2)));
        const nr: i64 = @as(c_int, @truncate(ggml.impl.c.ggml_nrows(src0)));

        ggml.impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
        ggml.impl.assert(ggml.impl.c.ggml_nrows(dst) == nr, "ggml_nrows(dst) == nr");

        const swapped = ggml.impl.getOpParamsI32(dst, 1);

        // rows per thread
        const dr = @divTrunc(nr + nth - 1, nth);

        // row range for this thread
        const ir0 = dr * ith;
        const ir1 = @min(ir0 + dr, nr);

        return .{
            .src0_d = src0_d,
            .src1_d = src1_d,
            .src0_o = src0_o,
            .src1_o = src1_o,
            .has_src1 = src1 != null,
            .swapped = swapped != 0,
            .nc = nc,
            .ir0 = ir0,
            .ir1 = ir1,
        };
    }

    /// Return: the value and gate rows for row `j1`, as `src0_p` and `src1_p`.
    fn halves(self: GluRows, comptime T: type, j1: i64) struct { [*]T, [*]T } {
        const row: usize = @intCast(j1);
        var src0_p: [*]T = @ptrCast(@alignCast(self.src0_d + row * self.src0_o));
        var src1_p: [*]T = @ptrCast(@alignCast(self.src1_d + row * self.src1_o));

        if (!self.has_src1) {
            const nc: usize = @intCast(self.nc);
            src0_p += if (self.swapped) nc else 0;
            src1_p += if (self.swapped) 0 else nc;
        }
        return .{ src0_p, src1_p };
    }
};

/// Ports `ggml_compute_forward_reglu_f32` and `ggml_compute_forward_reglu_f16`
/// (ops.cpp:2929, 2988 @b1b770d90), and with them
/// `ggml_compute_forward_geglu_f32`, `ggml_compute_forward_geglu_f16`,
/// `ggml_compute_forward_swiglu_f32`, `ggml_compute_forward_swiglu_f16`,
/// `ggml_compute_forward_geglu_erf_f32`,
/// `ggml_compute_forward_geglu_erf_f16`,
/// `ggml_compute_forward_geglu_quick_f32` and
/// `ggml_compute_forward_geglu_quick_f16`
/// (ops.cpp:3082, 3141, 3235, 3294, 3679, 3738, 3832, 3891 @b1b770d90),
/// which differ from it only in the `ggml_vec_*` called, and
/// `ggml_compute_forward_glu_bf16_rows` (ops.cpp:2878 @b1b770d90), the C's
/// template for the BF16 arms: the same body and assertions over
/// `ggml_bf16_t`.
fn gluRows(
    comptime T: type,
    comptime kernel: anytype,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
) void {
    const g = GluRows.of(params, dst);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var j1: i64 = g.ir0;
    while (j1 < g.ir1) : (j1 += 1) {
        const src0_p, const src1_p = g.halves(T, j1);
        kernel(g.nc, @ptrCast(@alignCast(dd + @as(usize, @intCast(j1)) * dst.nb[1])), src0_p, src1_p);
    }
}

/// The type switch every GLU dispatcher opens with.
fn gluByType(
    comptime k32: anytype,
    comptime k16: anytype,
    comptime kb16: anytype,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.c.GGML_TYPE_F32 => gluRows(f32, k32, params, dst),
        ggml.impl.c.GGML_TYPE_F16 => gluRows(ggml.impl.c.ggml_fp16_t, k16, params, dst),
        ggml.impl.c.GGML_TYPE_BF16 => gluRows(ggml.impl.c.ggml_bf16_t, kb16, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_reglu` (ops.cpp:3054 @b1b770d90) and
/// `ggml_compute_forward_reglu_bf16` (ops.cpp:3048 @b1b770d90).
fn reglu(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    gluByType(ggml.cpu.ops.vecinline.reglu_f32, ggml.cpu.ops.vecinline.reglu_f16, ggml.cpu.ops.vecinline.reglu_bf16, params, dst);
}

/// Ports `ggml_compute_forward_geglu` (ops.cpp:3207 @b1b770d90) and
/// `ggml_compute_forward_geglu_bf16` (ops.cpp:3201 @b1b770d90).
fn geglu(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    gluByType(ggml.cpu.ops.vecinline.geglu_f32, ggml.cpu.ops.vecinline.geglu_f16, ggml.cpu.ops.vecinline.geglu_bf16, params, dst);
}

/// Ports `ggml_compute_forward_swiglu` (ops.cpp:3360 @b1b770d90) and
/// `ggml_compute_forward_swiglu_bf16` (ops.cpp:3354 @b1b770d90).
fn swiglu(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    gluByType(swigluF32, ggml.cpu.ops.vecinline.swiglu_f16, ggml.cpu.ops.vecinline.swiglu_bf16, params, dst);
}

/// Ports `ggml_compute_forward_geglu_erf` (ops.cpp:3804 @b1b770d90) and
/// `ggml_compute_forward_geglu_erf_bf16` (ops.cpp:3798 @b1b770d90).
fn gegluErf(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    gluByType(ggml.cpu.ops.vecinline.geglu_erf_f32, ggml.cpu.ops.vecinline.geglu_erf_f16, ggml.cpu.ops.vecinline.geglu_erf_bf16, params, dst);
}

/// Ports `ggml_compute_forward_geglu_quick` (ops.cpp:3957 @b1b770d90) and
/// `ggml_compute_forward_geglu_quick_bf16` (ops.cpp:3951 @b1b770d90).
fn gegluQuick(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    gluByType(gegluQuickF32, ggml.cpu.ops.vecinline.geglu_quick_f16, ggml.cpu.ops.vecinline.geglu_quick_bf16, params, dst);
}

/// Ports `ggml_compute_forward_swiglu_oai_f32` (ops.cpp:3388 @b1b770d90) and
/// `ggml_compute_forward_swiglu_oai_f16` (ops.cpp:3455 @b1b770d90).
///
/// The clamped SwiGLU of gpt-oss: the value is capped at `limit` from above
/// only, the gate on both sides, and the sigmoid is scaled by `alpha`. No
/// expression in the loop adds a product, so nothing here contracts. The f16
/// variant widens through `GGML_CPU_FP16_TO_FP32` and narrows
/// through `GGML_CPU_FP32_TO_FP16`, unlike `swiglu_clamp`'s, which uses the
/// non-CPU macros. Its `#ifndef NDEBUG` NaN/Inf scan is compiled out in the
/// reference.
fn swigluOaiRows(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    const g = GluRows.of(params, dst);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    const alpha = ggml.impl.getOpParamsF32(dst, 2);
    const limit = ggml.impl.getOpParamsF32(dst, 3);

    const nc: usize = @intCast(g.nc);

    var j1: i64 = g.ir0;
    while (j1 < g.ir1) : (j1 += 1) {
        const src0_p, const src1_p = g.halves(T, j1);
        const dst_p: [*]T = @ptrCast(@alignCast(dd + @as(usize, @intCast(j1)) * dst.nb[1]));

        for (0..nc) |k| {
            if (T == f32) {
                const x = stdlib.libcxx.min(f32, src0_p[k], limit);
                const y = stdlib.libcxx.clamp(f32, src1_p[k], -limit, limit);
                const out_glu = x / (1.0 + stdlib.libm.expf(alpha * (-x)));
                dst_p[k] = out_glu * (y + 1.0);
            } else {
                const x = stdlib.libcxx.min(f32, ggml.cpu.convert.cpuFp16ToFp32(src0_p[k]), limit);
                const y = stdlib.libcxx.clamp(f32, ggml.cpu.convert.cpuFp16ToFp32(src1_p[k]), -limit, limit);
                const out_glu = x / (1.0 + stdlib.libm.expf(alpha * (-x)));
                dst_p[k] = ggml.cpu.convert.cpuFp32ToFp16(out_glu * (y + 1.0));
            }
        }
    }
}

/// Ports `ggml_compute_forward_swiglu_oai` (ops.cpp:3522 @b1b770d90).
fn swigluOai(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.c.GGML_TYPE_F32 => swigluOaiRows(f32, params, dst),
        ggml.impl.c.GGML_TYPE_F16 => swigluOaiRows(ggml.impl.c.ggml_fp16_t, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_swiglu_clamp_f32` and
/// `ggml_compute_forward_swiglu_clamp_f16` (ops.cpp:3546, 3605 @b1b770d90).
///
/// SwiGLU with the gate capped at `limit` from above and the value clamped
/// on both sides; unlike `swiglu_oai` the sigmoid is unscaled and the value
/// is not offset by one. No expression adds a product, so nothing contracts.
/// The C's `#ifndef NDEBUG` NaN/Inf scan is compiled out in the reference.
fn swigluClampRows(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    const g = GluRows.of(params, dst);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    const limit = ggml.impl.getOpParamsF32(dst, 3);

    const nc: usize = @intCast(g.nc);

    var j1: i64 = g.ir0;
    while (j1 < g.ir1) : (j1 += 1) {
        const src0_p, const src1_p = g.halves(T, j1);
        const dst_p: [*]T = @ptrCast(@alignCast(dd + @as(usize, @intCast(j1)) * dst.nb[1]));

        for (0..nc) |k| {
            if (T == f32) {
                const gate = stdlib.libcxx.min(f32, src0_p[k], limit);
                const up = stdlib.libcxx.clamp(f32, src1_p[k], -limit, limit);
                dst_p[k] = gate / (1.0 + stdlib.libm.expf(-gate)) * up;
            } else {
                const gate = stdlib.libcxx.min(f32, ggml.impl.fp16ToFp32(src0_p[k]), limit);
                const up = stdlib.libcxx.clamp(f32, ggml.impl.fp16ToFp32(src1_p[k]), -limit, limit);
                dst_p[k] = ggml.impl.fp32ToFp16(gate / (1.0 + stdlib.libm.expf(-gate)) * up);
            }
        }
    }
}

/// Ports `ggml_compute_forward_swiglu_clamp` (ops.cpp:3664 @b1b770d90).
fn swigluClamp(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.c.GGML_TYPE_F32 => swigluClampRows(f32, params, dst),
        ggml.impl.c.GGML_TYPE_F16 => swigluClampRows(ggml.impl.c.ggml_fp16_t, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// The dispatchers

/// Ports `ggml_compute_forward_unary` (ops.cpp:10245 @b1b770d90).
pub export fn ggml_compute_forward_unary(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) callconv(.c) void {
    const op = ggml.impl.c.ggml_get_unary_op(dst);

    switch (op) {
        ggml.impl.c.GGML_UNARY_OP_ABS => ggml.cpu.unary_ops.entry(.abs)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_SGN => ggml.cpu.unary_ops.entry(.sgn)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_NEG => ggml.cpu.unary_ops.entry(.neg)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_STEP => ggml.cpu.unary_ops.entry(.step)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_TANH => ggml.cpu.unary_ops.entry(.tanh)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_ELU => ggml.cpu.unary_ops.entry(.elu)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_RELU => ggml.cpu.unary_ops.entry(.relu)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_SIGMOID => ggml.cpu.unary_ops.entry(.sigmoid)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_GELU => gelu(params, dst),
        ggml.impl.c.GGML_UNARY_OP_GELU_ERF => geluErf(params, dst),
        ggml.impl.c.GGML_UNARY_OP_GELU_QUICK => geluQuick(params, dst),
        ggml.impl.c.GGML_UNARY_OP_SILU => silu(params, dst),
        ggml.impl.c.GGML_UNARY_OP_HARDSWISH => ggml.cpu.unary_ops.entry(.hardswish)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_HARDSIGMOID => ggml.cpu.unary_ops.entry(.hardsigmoid)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_EXP => ggml.cpu.unary_ops.entry(.exp)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_FLOOR => ggml.cpu.unary_ops.entry(.floor)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_CEIL => ggml.cpu.unary_ops.entry(.ceil)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_ROUND => ggml.cpu.unary_ops.entry(.round)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_TRUNC => ggml.cpu.unary_ops.entry(.trunc)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_XIELU => ggml.cpu.unary_ops.ggml_compute_forward_xielu(params, dst),
        ggml.impl.c.GGML_UNARY_OP_EXPM1 => ggml.cpu.unary_ops.entry(.expm1)(params, dst),
        ggml.impl.c.GGML_UNARY_OP_SOFTPLUS => ggml.cpu.unary_ops.entry(.softplus)(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_glu` (ops.cpp:10349 @b1b770d90).
pub export fn ggml_compute_forward_glu(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) callconv(.c) void {
    const op = ggml.impl.c.ggml_get_glu_op(dst);

    switch (op) {
        ggml.impl.c.GGML_GLU_OP_REGLU => reglu(params, dst),
        ggml.impl.c.GGML_GLU_OP_GEGLU => geglu(params, dst),
        ggml.impl.c.GGML_GLU_OP_SWIGLU => swiglu(params, dst),
        ggml.impl.c.GGML_GLU_OP_SWIGLU_OAI => swigluOai(params, dst),
        ggml.impl.c.GGML_GLU_OP_GEGLU_ERF => gegluErf(params, dst),
        ggml.impl.c.GGML_GLU_OP_GEGLU_QUICK => gegluQuick(params, dst),
        ggml.impl.c.GGML_GLU_OP_SWIGLU_CLAMP => swigluClamp(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "std::min and std::clamp pass a NaN through, as libc++'s do" {
    const nan = std.math.nan(f32);
    try std.testing.expect(std.math.isNan(stdlib.libcxx.min(f32, nan, 1.0)));
    try std.testing.expect(std.math.isNan(stdlib.libcxx.clamp(f32, nan, -1.0, 1.0)));
    try std.testing.expectEqual(@as(f32, 1.0), stdlib.libcxx.min(f32, 3.0, 1.0));
    try std.testing.expectEqual(@as(f32, -1.0), stdlib.libcxx.clamp(f32, -3.0, -1.0, 1.0));
    try std.testing.expectEqual(@as(f32, 0.5), stdlib.libcxx.clamp(f32, 0.5, -1.0, 1.0));
}
