//! `unary`, `glu`, `silu_back` and `leaky_relu`: the activations `ops.cpp`
//! keeps for itself, and the two dispatchers that route every unary and
//! gated-linear-unit op.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
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
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;
const fp16 = c.ggml_fp16_t;

extern fn expf(x: f32) f32;

// -----------------------------------------------------------------------------
// The unary kernels `unary-ops.cpp` provides
//
// Ported in `cpu/unary_ops.zig`. Declared rather than imported so the
// dispatcher below reads like the C's, which calls them across a
// translation-unit boundary too.

extern fn ggml_compute_forward_abs(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_sgn(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_neg(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_step(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_tanh(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_elu(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_relu(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_sigmoid(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_hardswish(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_hardsigmoid(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_exp(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_floor(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_ceil(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_round(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_trunc(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_xielu(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_expm1(params: *const ComputeParams, dst: *Tensor) void;
extern fn ggml_compute_forward_softplus(params: *const ComputeParams, dst: *Tensor) void;

// -----------------------------------------------------------------------------
// The `vec.h` kernels, at one signature
//
// Most of `vecinline.zig` takes an `i64` count; the two `vec.cpp` exports take
// a `c_int`. These adapt the latter so `unaryRows` and `gluRows` can take any
// of them as one comptime parameter.

fn siluF32(n: i64, y: [*]f32, x: [*]const f32) void {
    vec.silu_f32(@intCast(n), y, x);
}

fn swigluF32(n: i64, y: [*]f32, x: [*]const f32, g: [*]const f32) void {
    vec.swiglu_f32(@intCast(n), y, x, g);
}

// -----------------------------------------------------------------------------
// gelu, gelu_erf, gelu_quick, silu

/// Ports `ggml_compute_forward_gelu_f32` and `ggml_compute_forward_gelu_f16`
/// (ops.cpp:2111, 2158 @c1d0e7a00), and with them
/// `ggml_compute_forward_gelu_erf_f32`, `ggml_compute_forward_gelu_erf_f16`,
/// `ggml_compute_forward_gelu_quick_f32`,
/// `ggml_compute_forward_gelu_quick_f16`, `ggml_compute_forward_silu_f32`
/// and `ggml_compute_forward_silu_f16`
/// (ops.cpp:2341, 2388, 2460, 2507, 2579, 2626 @c1d0e7a00), which are the
/// same body over a different `ggml_vec_*`.
///
/// Rows are split across threads; within a row the kernel runs over `nc`
/// contiguous elements.
fn unaryRows(
    comptime T: type,
    comptime kernel: anytype,
    params: *const ComputeParams,
    dst: *Tensor,
) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    const l = common.UnaryLocals.of(src0, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nc: i64 = @as(c_int, @truncate(src0.ne[0]));
    const nr: i64 = @as(c_int, @truncate(c.ggml_nrows(src0)));

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

/// The type switch every activation dispatcher opens with: `f32` or `f16`,
/// and `GGML_ABORT` on anything else.
fn unaryByType(
    comptime k32: anytype,
    comptime k16: anytype,
    params: *const ComputeParams,
    dst: *Tensor,
) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => unaryRows(f32, k32, params, dst),
        c.GGML_TYPE_F16 => unaryRows(fp16, k16, params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_gelu` (ops.cpp:2206 @c1d0e7a00).
fn gelu(params: *const ComputeParams, dst: *Tensor) void {
    unaryByType(vec.gelu_f32, vec.gelu_f16, params, dst);
}

/// Ports `ggml_compute_forward_gelu_erf` (ops.cpp:2436 @c1d0e7a00).
fn geluErf(params: *const ComputeParams, dst: *Tensor) void {
    unaryByType(vec.gelu_erf_f32, vec.gelu_erf_f16, params, dst);
}

/// Ports `ggml_compute_forward_gelu_quick` (ops.cpp:2555 @c1d0e7a00).
fn geluQuick(params: *const ComputeParams, dst: *Tensor) void {
    unaryByType(vec.gelu_quick_f32, vec.gelu_quick_f16, params, dst);
}

/// Ports `ggml_compute_forward_silu` (ops.cpp:2674 @c1d0e7a00).
///
/// The `f32` arm reaches `ggml_vec_silu_f32`, exported by `cpu/vec.zig` with
/// its NEON polynomial `exp`; the `f16` arm is the scalar `vec.h` inline with
/// libm's `expf`. The two are not the same function of `x`, as in the C.
fn silu(params: *const ComputeParams, dst: *Tensor) void {
    unaryByType(siluF32, vec.silu_f16, params, dst);
}

// -----------------------------------------------------------------------------
// leaky_relu

/// Ports `ggml_compute_forward_leaky_relu_f32` and
/// `ggml_compute_forward_leaky_relu_f16` (ops.cpp:2697, 2727 @c1d0e7a00).
///
/// Single-threaded: every thread but the first returns at once. Rows are
/// addressed by `nb[1]` alone, the C assuming (in an `assert` compiled out
/// here) that both tensors are contiguous in their upper dimensions.
fn leakyReluRows(comptime T: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    const n: i64 = @as(c_int, @truncate(c.ggml_nrows(src0)));
    const nc: i64 = @as(c_int, @truncate(src0.ne[0]));

    const negative_slope: f32 = impl.getOpParamsF32(dst, 0);

    const s0: [*]u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var i: i64 = 0;
    while (i < n) : (i += 1) {
        const y: [*]T = @ptrCast(@alignCast(dd + @as(usize, @intCast(i)) * dst.nb[1]));
        const x: [*]const T = @ptrCast(@alignCast(s0 + @as(usize, @intCast(i)) * src0.nb[1]));
        if (T == f32) vec.leaky_relu_f32(nc, y, x, negative_slope) else vec.leaky_relu_f16(nc, y, x, negative_slope);
    }
}

/// Ports `ggml_compute_forward_leaky_relu` (ops.cpp:2757 @c1d0e7a00).
pub export fn ggml_compute_forward_leaky_relu(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => leakyReluRows(f32, params, dst),
        c.GGML_TYPE_F16 => leakyReluRows(fp16, params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// silu_back

/// Ports `ggml_compute_forward_silu_back_f32` and
/// `ggml_compute_forward_silu_back_f16` (ops.cpp:2781, 2824 @c1d0e7a00).
///
/// Note the operand order: `src[0]` is the incoming gradient and `src[1]` the
/// forward input, and the row count comes from `src[1]`.
fn siluBackRows(comptime T: type, params: *const ComputeParams, dst: *Tensor) void {
    const grad = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nc: i64 = @as(c_int, @truncate(src1.ne[0]));
    const nr: i64 = @as(c_int, @truncate(c.ggml_nrows(src1)));

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
        if (T == f32) vec.silu_backward_f32(nc, dx, x, dy) else vec.silu_backward_f16(nc, dx, x, dy);
    }
}

/// Ports `ggml_compute_forward_silu_back` (ops.cpp:2868 @c1d0e7a00).
pub export fn ggml_compute_forward_silu_back(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => siluBackRows(f32, params, dst),
        c.GGML_TYPE_F16 => siluBackRows(fp16, params, dst),
        else => impl.abort("fatal error"),
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

    fn of(params: *const ComputeParams, dst: *Tensor) GluRows {
        const src0 = impl.one(Tensor, dst.src[0]);
        const src1: ?*Tensor = dst.src[1];
        const src0_d: [*]u8 = @ptrCast(src0.data.?);
        const src1_d: [*]u8 = @ptrCast((if (src1) |s| s.data else src0.data).?);
        const src0_o = src0.nb[1];
        const src1_o = if (src1) |s| s.nb[1] else src0.nb[1];

        impl.assert(c.ggml_is_contiguous_1(src0), "ggml_is_contiguous_1(src0)");
        impl.assert(c.ggml_is_contiguous_1(dst), "ggml_is_contiguous_1(dst)");

        if (src1) |s| {
            impl.assert(c.ggml_is_contiguous_1(s), "ggml_is_contiguous_1(src1)");
            impl.assert(src0.type == s.type, "src0->type == src1->type");
        }

        const ith: i64 = params.ith;
        const nth: i64 = params.nth;

        const nc: i64 = @as(c_int, @truncate(if (src1 != null) src0.ne[0] else @divTrunc(src0.ne[0], 2)));
        const nr: i64 = @as(c_int, @truncate(c.ggml_nrows(src0)));

        impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
        impl.assert(c.ggml_nrows(dst) == nr, "ggml_nrows(dst) == nr");

        const swapped = impl.getOpParamsI32(dst, 1);

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
/// (ops.cpp:2892, 2951 @c1d0e7a00), and with them
/// `ggml_compute_forward_geglu_f32`, `ggml_compute_forward_geglu_f16`,
/// `ggml_compute_forward_swiglu_f32`, `ggml_compute_forward_swiglu_f16`,
/// `ggml_compute_forward_geglu_erf_f32`,
/// `ggml_compute_forward_geglu_erf_f16`,
/// `ggml_compute_forward_geglu_quick_f32` and
/// `ggml_compute_forward_geglu_quick_f16`
/// (ops.cpp:3035, 3094, 3178, 3237, 3408, 3467, 3551, 3610 @c1d0e7a00),
/// which differ from it only in the `ggml_vec_*` called.
fn gluRows(
    comptime T: type,
    comptime kernel: anytype,
    params: *const ComputeParams,
    dst: *Tensor,
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
    params: *const ComputeParams,
    dst: *Tensor,
) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => gluRows(f32, k32, params, dst),
        c.GGML_TYPE_F16 => gluRows(fp16, k16, params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_reglu` (ops.cpp:3011 @c1d0e7a00).
fn reglu(params: *const ComputeParams, dst: *Tensor) void {
    gluByType(vec.reglu_f32, vec.reglu_f16, params, dst);
}

/// Ports `ggml_compute_forward_geglu` (ops.cpp:3154 @c1d0e7a00).
fn geglu(params: *const ComputeParams, dst: *Tensor) void {
    gluByType(vec.geglu_f32, vec.geglu_f16, params, dst);
}

/// Ports `ggml_compute_forward_swiglu` (ops.cpp:3297 @c1d0e7a00).
fn swiglu(params: *const ComputeParams, dst: *Tensor) void {
    gluByType(swigluF32, vec.swiglu_f16, params, dst);
}

/// Ports `ggml_compute_forward_geglu_erf` (ops.cpp:3527 @c1d0e7a00).
fn gegluErf(params: *const ComputeParams, dst: *Tensor) void {
    gluByType(vec.geglu_erf_f32, vec.geglu_erf_f16, params, dst);
}

/// Ports `ggml_compute_forward_geglu_quick` (ops.cpp:3670 @c1d0e7a00).
fn gegluQuick(params: *const ComputeParams, dst: *Tensor) void {
    gluByType(vec.geglu_quick_f32, vec.geglu_quick_f16, params, dst);
}

/// `std::min` with the C++'s NaN behaviour; see `common.stdMin`.
const stdMin = common.stdMin;

/// Ports `std::clamp(v, lo, hi)` as libc++ defines it:
/// `v < lo ? lo : (hi < v ? hi : v)`. A NaN `v` passes through.
inline fn stdClamp(v: f32, lo: f32, hi: f32) f32 {
    return if (v < lo) lo else if (hi < v) hi else v;
}

/// Ports `ggml_compute_forward_swiglu_oai_f32` (ops.cpp:3321 @c1d0e7a00).
///
/// The clamped SwiGLU of gpt-oss: the value is capped at `limit` from above
/// only, the gate on both sides, and the sigmoid is scaled by `alpha`. No
/// expression in the loop adds a product, so nothing here contracts.
fn swigluOaiF32(params: *const ComputeParams, dst: *Tensor) void {
    const g = GluRows.of(params, dst);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    const alpha = impl.getOpParamsF32(dst, 2);
    const limit = impl.getOpParamsF32(dst, 3);

    const nc: usize = @intCast(g.nc);

    var j1: i64 = g.ir0;
    while (j1 < g.ir1) : (j1 += 1) {
        const src0_p, const src1_p = g.halves(f32, j1);
        const dst_p: [*]f32 = @ptrCast(@alignCast(dd + @as(usize, @intCast(j1)) * dst.nb[1]));

        for (0..nc) |k| {
            const x = stdMin(src0_p[k], limit);
            const y = stdClamp(src1_p[k], -limit, limit);
            const out_glu = x / (1.0 + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.0);
        }
    }
}

/// Ports `ggml_compute_forward_swiglu_oai` (ops.cpp:3388 @c1d0e7a00).
fn swigluOai(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => swigluOaiF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// The dispatchers

/// Ports `ggml_compute_forward_unary` (ops.cpp:9998 @c1d0e7a00).
pub export fn ggml_compute_forward_unary(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const op = c.ggml_get_unary_op(dst);

    switch (op) {
        c.GGML_UNARY_OP_ABS => ggml_compute_forward_abs(params, dst),
        c.GGML_UNARY_OP_SGN => ggml_compute_forward_sgn(params, dst),
        c.GGML_UNARY_OP_NEG => ggml_compute_forward_neg(params, dst),
        c.GGML_UNARY_OP_STEP => ggml_compute_forward_step(params, dst),
        c.GGML_UNARY_OP_TANH => ggml_compute_forward_tanh(params, dst),
        c.GGML_UNARY_OP_ELU => ggml_compute_forward_elu(params, dst),
        c.GGML_UNARY_OP_RELU => ggml_compute_forward_relu(params, dst),
        c.GGML_UNARY_OP_SIGMOID => ggml_compute_forward_sigmoid(params, dst),
        c.GGML_UNARY_OP_GELU => gelu(params, dst),
        c.GGML_UNARY_OP_GELU_ERF => geluErf(params, dst),
        c.GGML_UNARY_OP_GELU_QUICK => geluQuick(params, dst),
        c.GGML_UNARY_OP_SILU => silu(params, dst),
        c.GGML_UNARY_OP_HARDSWISH => ggml_compute_forward_hardswish(params, dst),
        c.GGML_UNARY_OP_HARDSIGMOID => ggml_compute_forward_hardsigmoid(params, dst),
        c.GGML_UNARY_OP_EXP => ggml_compute_forward_exp(params, dst),
        c.GGML_UNARY_OP_FLOOR => ggml_compute_forward_floor(params, dst),
        c.GGML_UNARY_OP_CEIL => ggml_compute_forward_ceil(params, dst),
        c.GGML_UNARY_OP_ROUND => ggml_compute_forward_round(params, dst),
        c.GGML_UNARY_OP_TRUNC => ggml_compute_forward_trunc(params, dst),
        c.GGML_UNARY_OP_XIELU => ggml_compute_forward_xielu(params, dst),
        c.GGML_UNARY_OP_EXPM1 => ggml_compute_forward_expm1(params, dst),
        c.GGML_UNARY_OP_SOFTPLUS => ggml_compute_forward_softplus(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_glu` (ops.cpp:10102 @c1d0e7a00).
pub export fn ggml_compute_forward_glu(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const op = c.ggml_get_glu_op(dst);

    switch (op) {
        c.GGML_GLU_OP_REGLU => reglu(params, dst),
        c.GGML_GLU_OP_GEGLU => geglu(params, dst),
        c.GGML_GLU_OP_SWIGLU => swiglu(params, dst),
        c.GGML_GLU_OP_SWIGLU_OAI => swigluOai(params, dst),
        c.GGML_GLU_OP_GEGLU_ERF => gegluErf(params, dst),
        c.GGML_GLU_OP_GEGLU_QUICK => gegluQuick(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "std::min and std::clamp pass a NaN through, as libc++'s do" {
    const nan = std.math.nan(f32);
    try std.testing.expect(std.math.isNan(stdMin(nan, 1.0)));
    try std.testing.expect(std.math.isNan(stdClamp(nan, -1.0, 1.0)));
    try std.testing.expectEqual(@as(f32, 1.0), stdMin(3.0, 1.0));
    try std.testing.expectEqual(@as(f32, -1.0), stdClamp(-3.0, -1.0, 1.0));
    try std.testing.expectEqual(@as(f32, 0.5), stdClamp(0.5, -1.0, 1.0));
}
