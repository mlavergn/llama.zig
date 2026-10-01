//! The twenty-three element-wise unary CPU kernels: abs, sgn, neg, step, tanh,
//! elu, relu, sigmoid, hardsigmoid, exp, hardswish, sqr, sqrt, sin, cos, log,
//! expm1, softplus, floor, ceil, round, trunc and xielu.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-cpu/unary-ops.cpp` — the kernels
//! - `llama.cpp/ggml/src/ggml-cpu/common.h`      — `type_conversion_table`
//!                                                 and `get_thread_range`
//!
//! Both at v0.3.0 (`c1d0e7a00`). Each declaration below names the C++ it
//! replaces and the line it began at. This file exports the same 23 C symbols
//! with the same signatures.
//!
//! # libm, not Zig's builtins
//!
//! Every transcendental here goes through an `extern` declaration of the libm
//! function the C calls — `expf`, `tanhf`, `logf` and the rest — not through
//! `@exp`, `@tanh` or `std.math`. Zig's builtins lower to LLVM intrinsics,
//! which are free to differ from the platform's libm in the last bit, and
//! there is no oracle here that would catch it: `make backend-ops` compares
//! CPU against Metal with a tolerance, not bit-for-bit. Calling the same
//! function the C calls removes the question rather than measuring it.
//!
//! # The one functor
//!
//! Twenty-two ops are plain `float -> float` and go through `unaryOp`. `xielu`
//! is not: it reads four values from `op_params` and the C++ wraps them in a
//! capturing lambda passed to a second template family
//! (`apply_unary_op_functor`). Zig has no capturing closures, so the captures
//! become a struct with an `apply` method — and the two families stay separate
//! here because **their assertions differ**: the plain path requires
//! `ggml_is_contiguous_rows`, the functor path only `ggml_is_contiguous_1`.
//! Merging them would silently tighten or loosen one of the two.
//!
//! # Not ported: `unary_op_params`
//!
//! `unary-ops.cpp:155` defines a third template family over
//! `float (*)(float, ggml_tensor *)`. Nothing instantiates it, so the C++
//! emits no code for it and it has no symbol. Recorded here so its absence is
//! a decision rather than an oversight.

const std = @import("std");
const impl = @import("../impl.zig");
const defs = @import("defs.zig");
const c = impl.c;

const Tensor = defs.Tensor;
const ComputeParams = defs.ComputeParams;

// -----------------------------------------------------------------------------
// libm
//
// Declared rather than imported: these come from `<math.h>` via the C's
// headers, and `src/ggml/impl.zig`'s `@cImport` does not take it.

extern fn fabsf(x: f32) f32;
extern fn tanhf(x: f32) f32;
extern fn expf(x: f32) f32;
extern fn expm1f(x: f32) f32;
extern fn logf(x: f32) f32;
extern fn sqrtf(x: f32) f32;
extern fn sinf(x: f32) f32;
extern fn cosf(x: f32) f32;
extern fn floorf(x: f32) f32;
extern fn ceilf(x: f32) f32;
extern fn roundf(x: f32) f32;
extern fn truncf(x: f32) f32;
extern fn fminf(x: f32, y: f32) f32;
extern fn fmaxf(x: f32, y: f32) f32;

// -----------------------------------------------------------------------------
// The scalar operations
//
// Ports `op_abs` through `op_trunc` (unary-ops.cpp:3 to :96 @c1d0e7a00). The
// C++ passes each as a function-pointer template argument; an enum selects
// them at comptime here, as in `binary_ops.zig`.

/// Ports the scalar op functions of `unary-ops.cpp`
/// (unary-ops.cpp:3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 64, 68,
/// 72, 76, 80, 84, 88, 92, 96 @c1d0e7a00); names and numbers pair up in order.
const Op = enum {
    abs,
    sgn,
    neg,
    step,
    tanh,
    elu,
    relu,
    sigmoid,
    hardsigmoid,
    exp,
    hardswish,
    sqr,
    sqrt,
    sin,
    cos,
    log,
    expm1,
    softplus,
    floor,
    ceil,
    round,
    trunc,

    inline fn apply(comptime op: Op, x: f32) f32 {
        return switch (op) {
            .abs => fabsf(x),
            .sgn => if (x > 0.0) 1.0 else (if (x < 0.0) -1.0 else 0.0),
            .neg => -x,
            .step => if (x > 0.0) 1.0 else 0.0,
            .tanh => tanhf(x),
            .elu => if (x > 0.0) x else expm1f(x),
            .relu => if (x > 0.0) x else 0.0,
            .sigmoid => 1.0 / (1.0 + expf(-x)),
            .hardsigmoid => fminf(1.0, fmaxf(0.0, (x + 3.0) / 6.0)),
            .exp => expf(x),
            .hardswish => x * fminf(1.0, fmaxf(0.0, (x + 3.0) / 6.0)),
            .sqr => x * x,
            .sqrt => sqrtf(x),
            .sin => sinf(x),
            .cos => cosf(x),
            .log => logf(x),
            // Note this is `expf(x) - 1`, *not* `expm1f(x)`: the C spells it
            // out and the two differ near zero. `elu` above does use `expm1f`.
            .expm1 => expf(x) - 1.0,
            .softplus => if (x > 20.0) x else logf(1.0 + expf(x)),
            .floor => floorf(x),
            .ceil => ceilf(x),
            .round => roundf(x),
            .trunc => truncf(x),
        };
    }
};

/// Ports `op_xielu` (unary-ops.cpp:55 @c1d0e7a00).
///
/// The four parameters are the C++ lambda's captures, read from `op_params`
/// by `ggml_compute_forward_xielu`.
const Xielu = struct {
    alpha_n: f32,
    alpha_p: f32,
    beta: f32,
    eps: f32,

    /// **Both branches name an FMA, and that is measured rather than
    /// guessed.** `scripts/ops-diff` caught this: the C compiles at
    /// `-ffp-contract=on` and fuses, strict Zig did not, and the result was
    /// 1 ULP out. The gate that passed it — `backend-ops`, at NMSE 1e-7 —
    /// cannot see a last-bit difference.
    ///
    /// Which multiply clang fuses is the question, since two feed each add.
    /// Measured over 200k inputs: for `x > 0`, `fma(alpha_p*x, x, beta*x)`
    /// matches all 99,596 positive samples where fusing the `beta` term
    /// matches 91% of them. That is the **left** operand of the `+`, and the
    /// negative branch follows the same rule — there both fusings agree on
    /// every sample, so consistency decides it rather than the data.
    inline fn apply(self: Xielu, x: f32) f32 {
        if (x > 0.0) {
            return @mulAdd(f32, self.alpha_p * x, x, self.beta * x);
        }
        const min_x_eps = fminf(x, self.eps);
        return @mulAdd(f32, expm1f(min_x_eps) - x, self.alpha_n, self.beta * x);
    }
};

// -----------------------------------------------------------------------------
// Element conversion and thread range
//
// The same two helpers `binary_ops.zig` ports from `ggml-cpu/common.h`; they
// are duplicated rather than shared because each file states its own
// provenance and the C has them in a header both include.

/// Ports the `to_f32` half of `type_conversion_table`
/// (ggml-cpu/common.h:48 @c1d0e7a00).
inline fn toF32(comptime T: type, v: T) f32 {
    return switch (T) {
        f32 => v,
        c.ggml_fp16_t => impl.fp16ToFp32(v),
        c.ggml_bf16_t => impl.bf16ToFp32(v.bits),
        else => @compileError("no conversion for " ++ @typeName(T)),
    };
}

/// Ports the `from_f32` half of `type_conversion_table`
/// (ggml-cpu/common.h:48 @c1d0e7a00).
inline fn fromF32(comptime T: type, v: f32) T {
    return switch (T) {
        f32 => v,
        c.ggml_fp16_t => impl.fp32ToFp16(v),
        c.ggml_bf16_t => .{ .bits = impl.fp32ToBf16(v) },
        else => @compileError("no conversion for " ++ @typeName(T)),
    };
}

/// Ports `get_thread_range` (ggml-cpu/common.h:74 @c1d0e7a00).
fn threadRange(params: *const ComputeParams, src0: *const Tensor) struct { i64, i64 } {
    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr = c.ggml_nrows(src0);

    const dr = @divTrunc(nr + nth - 1, nth);

    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    return .{ ir0, ir1 };
}

// -----------------------------------------------------------------------------
// The kernels
//
// The C's row indices `i01`, `i02`, `i03` are `j01`, `j02`, `j03` here: Zig
// parses `iNN` as an integer type name. Renamed digit for digit, as
// `cpu/mulmat.zig` and `cpu/binary_ops.zig` do.

/// Ports `vec_unary_op` (unary-ops.cpp:101 @c1d0e7a00).
inline fn vecUnaryOp(
    comptime op: Op,
    comptime Src0: type,
    comptime Dst: type,
    n: i64,
    y: [*]Dst,
    x: [*]const Src0,
) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        y[i] = fromF32(Dst, op.apply(toF32(Src0, x[i])));
    }
}

/// Ports `apply_unary_op` (unary-ops.cpp:111 @c1d0e7a00).
///
/// Asserts **`ggml_is_contiguous_rows`** on both tensors; the functor variant
/// below asserts only `ggml_is_contiguous_1`.
fn applyUnaryOp(
    comptime op: Op,
    comptime Src0: type,
    comptime Dst: type,
    params: *const ComputeParams,
    dst: *Tensor,
) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(
        c.ggml_is_contiguous_rows(src0) and c.ggml_is_contiguous_rows(dst) and c.ggml_are_same_shape(src0, dst),
        "ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst)",
    );

    const l = defs.BinaryLocals.of(src0, src0, dst);

    impl.assert(l.nb0 == @sizeOf(Dst), "nb0 == sizeof(dst_t)");
    impl.assert(l.nb00 == @sizeOf(Src0), "nb00 == sizeof(src0_t)");

    const ir0, const ir1 = threadRange(params, src0);

    var ir = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne02 * l.ne01, l.ne01);
        const j01 = ir - j03 * l.ne02 * l.ne01 - j02 * l.ne01;

        const dst_bytes = @as([*]u8, @ptrCast(dst.data)) +
            @as(usize, @intCast(j03)) * l.nb3 + @as(usize, @intCast(j02)) * l.nb2 + @as(usize, @intCast(j01)) * l.nb1;
        const src0_bytes = @as([*]const u8, @ptrCast(src0.data)) +
            @as(usize, @intCast(j03)) * l.nb03 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j01)) * l.nb01;

        vecUnaryOp(op, Src0, Dst, l.ne0, @ptrCast(@alignCast(dst_bytes)), @ptrCast(@alignCast(src0_bytes)));
    }
}

/// Ports `unary_op` (unary-ops.cpp:137 @c1d0e7a00).
///
/// Five type pairs, in the C's order. The C prints to `stderr` with `fprintf`
/// rather than going through `GGML_LOG_ERROR`, then aborts; `impl.logError`
/// reaches the same stream through ggml's own handler.
fn unaryOp(comptime op: Op, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    const F16 = c.ggml_fp16_t;
    const BF16 = c.ggml_bf16_t;

    if (src0.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32) {
        applyUnaryOp(op, f32, f32, params, dst); // all f32
    } else if (src0.type == c.GGML_TYPE_F16 and dst.type == c.GGML_TYPE_F16) {
        applyUnaryOp(op, F16, F16, params, dst); // all f16
    } else if (src0.type == c.GGML_TYPE_BF16 and dst.type == c.GGML_TYPE_BF16) {
        applyUnaryOp(op, BF16, BF16, params, dst); // all bf16
    } else if (src0.type == c.GGML_TYPE_BF16 and dst.type == c.GGML_TYPE_F32) {
        applyUnaryOp(op, BF16, f32, params, dst);
    } else if (src0.type == c.GGML_TYPE_F16 and dst.type == c.GGML_TYPE_F32) {
        applyUnaryOp(op, F16, f32, params, dst);
    } else {
        impl.logError(
            "%s: unsupported types: dst: %s, src0: %s\n",
            .{ "unary_op", c.ggml_type_name(dst.type), c.ggml_type_name(src0.type) },
        );
        impl.abort("fatal error");
    }
}

/// Ports `vec_unary_op_functor` (unary-ops.cpp:180 @c1d0e7a00).
inline fn vecUnaryOpFunctor(
    comptime Src0: type,
    comptime Dst: type,
    n: i64,
    y: [*]Dst,
    x: [*]const Src0,
    op: anytype,
) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        y[i] = fromF32(Dst, op.apply(toF32(Src0, x[i])));
    }
}

/// Ports `apply_unary_op_functor` (unary-ops.cpp:191 @c1d0e7a00).
///
/// Asserts **`ggml_is_contiguous_1`**, a weaker condition than the plain
/// path's `ggml_is_contiguous_rows`. That difference is the C's and is kept.
fn applyUnaryOpFunctor(
    comptime Src0: type,
    comptime Dst: type,
    params: *const ComputeParams,
    dst: *Tensor,
    op: anytype,
) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(
        c.ggml_is_contiguous_1(src0) and c.ggml_is_contiguous_1(dst) and c.ggml_are_same_shape(src0, dst),
        "ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst)",
    );

    const l = defs.BinaryLocals.of(src0, src0, dst);

    impl.assert(l.nb0 == @sizeOf(Dst), "nb0 == sizeof(dst_t)");
    impl.assert(l.nb00 == @sizeOf(Src0), "nb00 == sizeof(src0_t)");

    const ir0, const ir1 = threadRange(params, src0);

    var ir = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne02 * l.ne01, l.ne01);
        const j01 = ir - j03 * l.ne02 * l.ne01 - j02 * l.ne01;

        const dst_bytes = @as([*]u8, @ptrCast(dst.data)) +
            @as(usize, @intCast(j03)) * l.nb3 + @as(usize, @intCast(j02)) * l.nb2 + @as(usize, @intCast(j01)) * l.nb1;
        const src0_bytes = @as([*]const u8, @ptrCast(src0.data)) +
            @as(usize, @intCast(j03)) * l.nb03 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j01)) * l.nb01;

        vecUnaryOpFunctor(Src0, Dst, l.ne0, @ptrCast(@alignCast(dst_bytes)), @ptrCast(@alignCast(src0_bytes)), op);
    }
}

/// Ports `unary_op_functor` (unary-ops.cpp:217 @c1d0e7a00).
fn unaryOpFunctor(params: *const ComputeParams, dst: *Tensor, op: anytype) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    const F16 = c.ggml_fp16_t;
    const BF16 = c.ggml_bf16_t;

    if (src0.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32) {
        applyUnaryOpFunctor(f32, f32, params, dst, op); // all f32
    } else if (src0.type == c.GGML_TYPE_F16 and dst.type == c.GGML_TYPE_F16) {
        applyUnaryOpFunctor(F16, F16, params, dst, op); // all f16
    } else if (src0.type == c.GGML_TYPE_BF16 and dst.type == c.GGML_TYPE_BF16) {
        applyUnaryOpFunctor(BF16, BF16, params, dst, op); // all bf16
    } else if (src0.type == c.GGML_TYPE_BF16 and dst.type == c.GGML_TYPE_F32) {
        applyUnaryOpFunctor(BF16, f32, params, dst, op);
    } else if (src0.type == c.GGML_TYPE_F16 and dst.type == c.GGML_TYPE_F32) {
        applyUnaryOpFunctor(F16, f32, params, dst, op);
    } else {
        impl.logError(
            "%s: unsupported types: dst: %s, src0: %s\n",
            .{ "unary_op_functor", c.ggml_type_name(dst.type), c.ggml_type_name(src0.type) },
        );
        impl.abort("fatal error");
    }
}

// -----------------------------------------------------------------------------
// The exported entry points
//
// Ports `ggml_compute_forward_abs` through `ggml_compute_forward_trunc`
// (unary-ops.cpp:237 to :321 @c1d0e7a00), each a one-line call into `unaryOp`.
// Generated from the enum so the list cannot drift out of step with it; the
// C writes all twenty-two out by hand.

comptime {
    for (@typeInfo(Op).@"enum".fields) |field| {
        const op: Op = @enumFromInt(field.value);
        const S = struct {
            fn f(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
                unaryOp(op, params, dst);
            }
        };
        @export(&S.f, .{ .name = "ggml_compute_forward_" ++ field.name, .linkage = .strong });
    }
}

/// Ports `ggml_compute_forward_xielu` (unary-ops.cpp:325 @c1d0e7a00).
///
/// The only entry point that reads `op_params`, and the only one that goes
/// through the functor family. Note the C reads indices **1 to 4**, not 0 to 3.
pub export fn ggml_compute_forward_xielu(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const op: Xielu = .{
        .alpha_n = impl.getOpParamsF32(dst, 1),
        .alpha_p = impl.getOpParamsF32(dst, 2),
        .beta = impl.getOpParamsF32(dst, 3),
        .eps = impl.getOpParamsF32(dst, 4),
    };
    unaryOpFunctor(params, dst, op);
}
