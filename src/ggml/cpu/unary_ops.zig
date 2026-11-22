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
//! Both at v1.0.0 (`b1b770d90`). Each declaration below names the C++ it
//! replaces and the line it began at. This file exports the same 23 C symbols
//! with the same signatures.
//!
//! # libm, not Zig's builtins
//!
//! Every transcendental here goes through an `extern` declaration of the libm
//! function the C calls — `expf`, `tanhf`, `logf`, `log1pf` and the rest — not through
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

const ggml = @import("../module.zig");

const stdlib = @import("stdlib");

// -----------------------------------------------------------------------------
// libm
//
// Declared rather than imported: these come from `<math.h>` via the C's
// headers, and `src/ggml/impl.zig`'s `@cImport` does not take it.

// -----------------------------------------------------------------------------
// The scalar operations
//
// Ports `op_abs` through `op_trunc` (unary-ops.cpp:3 to :96 @b1b770d90). The
// C++ passes each as a function-pointer template argument; an enum selects
// them at comptime here, as in `binary_ops.zig`.

/// Ports `op_abs`, `op_sgn`, `op_neg`, `op_step`, `op_tanh`, `op_elu`,
/// `op_relu`, `op_sigmoid`, `op_hardsigmoid`, `op_exp`, `op_hardswish`,
/// `op_sqr`, `op_sqrt`, `op_sin`, `op_cos`, `op_log`, `op_expm1`,
/// `op_softplus`, `op_floor`, `op_ceil`, `op_round` and `op_trunc`
/// (unary-ops.cpp:3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 64, 68,
/// 72, 76, 80, 84, 88, 92, 96 @b1b770d90); names and numbers pair up in order.
pub const Op = enum {
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
            .abs => @abs(@as(f32, x)),
            .sgn => if (x > 0.0) 1.0 else (if (x < 0.0) -1.0 else 0.0),
            .neg => -x,
            .step => if (x > 0.0) 1.0 else 0.0,
            .tanh => stdlib.libm.tanhf(x),
            .elu => if (x > 0.0) x else stdlib.libm.expm1f(x),
            .relu => if (x > 0.0) x else 0.0,
            .sigmoid => 1.0 / (1.0 + stdlib.libm.expf(-x)),
            .hardsigmoid => stdlib.libm.fminf(1.0, stdlib.libm.fmaxf(0.0, (x + 3.0) / 6.0)),
            .exp => stdlib.libm.expf(x),
            .hardswish => x * stdlib.libm.fminf(1.0, stdlib.libm.fmaxf(0.0, (x + 3.0) / 6.0)),
            .sqr => x * x,
            .sqrt => @sqrt(@as(f32, x)),
            .sin => stdlib.libm.sinf(x),
            .cos => stdlib.libm.cosf(x),
            .log => stdlib.libm.logf(x),
            // Note this is `libm.expf(x) - 1`, *not* `libm.expm1f(x)`: the C spells it
            // out and the two differ near zero. `elu` above does use `expm1f`.
            .expm1 => stdlib.libm.expf(x) - 1.0,
            .softplus => if (x > 20.0) x else stdlib.libm.log1pf(stdlib.libm.expf(x)),
            .floor => @floor(@as(f32, x)),
            .ceil => @ceil(@as(f32, x)),
            .round => @round(@as(f32, x)),
            .trunc => @trunc(@as(f32, x)),
        };
    }
};

/// Ports `op_xielu` (unary-ops.cpp:55 @b1b770d90).
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
        const min_x_eps = stdlib.libm.fminf(x, self.eps);
        return @mulAdd(f32, stdlib.libm.expm1f(min_x_eps) - x, self.alpha_n, self.beta * x);
    }
};

// -----------------------------------------------------------------------------
// Element conversion and thread range
//
// -----------------------------------------------------------------------------
// The kernels
//
// The C's row indices `i01`, `i02`, `i03` are `j01`, `j02`, `j03` here: Zig
// parses `iNN` as an integer type name. Renamed digit for digit, as
// `cpu/mulmat.zig` and `cpu/binary_ops.zig` do.

/// Ports `vec_unary_op` (unary-ops.cpp:101 @b1b770d90).
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
        y[i] = ggml.cpu.ops.common.fromF32(Dst, op.apply(ggml.cpu.ops.common.toF32(Src0, x[i])));
    }
}

/// Ports `apply_unary_op` (unary-ops.cpp:111 @b1b770d90).
///
/// Asserts **`ggml_is_contiguous_rows`** on both tensors; the functor variant
/// below asserts only `ggml_is_contiguous_1`.
fn applyUnaryOp(
    comptime op: Op,
    comptime Src0: type,
    comptime Dst: type,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    ggml.impl.assert(
        ggml.impl.c.ggml_is_contiguous_rows(src0) and ggml.impl.c.ggml_is_contiguous_rows(dst) and ggml.impl.c.ggml_are_same_shape(src0, dst),
        "ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst)",
    );

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src0, dst);

    ggml.impl.assert(l.nb0 == @sizeOf(Dst), "nb0 == sizeof(dst_t)");
    ggml.impl.assert(l.nb00 == @sizeOf(Src0), "nb00 == sizeof(src0_t)");

    const ir0, const ir1 = ggml.cpu.ops.common.getThreadRange(params, src0);

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

/// Ports `unary_op` (unary-ops.cpp:137 @b1b770d90).
///
/// Five type pairs, in the C's order. The C prints to `stderr` with `fprintf`
/// rather than going through `GGML_LOG_ERROR`, then aborts, and so does this:
/// `impl.fprintfStderr`, which a log callback does not intercept.
fn unaryOp(comptime op: Op, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    const F16 = ggml.impl.c.ggml_fp16_t;
    const BF16 = ggml.impl.c.ggml_bf16_t;

    if (src0.type == ggml.impl.c.GGML_TYPE_F32 and dst.type == ggml.impl.c.GGML_TYPE_F32) {
        applyUnaryOp(op, f32, f32, params, dst); // all f32
    } else if (src0.type == ggml.impl.c.GGML_TYPE_F16 and dst.type == ggml.impl.c.GGML_TYPE_F16) {
        applyUnaryOp(op, F16, F16, params, dst); // all f16
    } else if (src0.type == ggml.impl.c.GGML_TYPE_BF16 and dst.type == ggml.impl.c.GGML_TYPE_BF16) {
        applyUnaryOp(op, BF16, BF16, params, dst); // all bf16
    } else if (src0.type == ggml.impl.c.GGML_TYPE_BF16 and dst.type == ggml.impl.c.GGML_TYPE_F32) {
        applyUnaryOp(op, BF16, f32, params, dst);
    } else if (src0.type == ggml.impl.c.GGML_TYPE_F16 and dst.type == ggml.impl.c.GGML_TYPE_F32) {
        applyUnaryOp(op, F16, f32, params, dst);
    } else {
        ggml.impl.fprintfStderr(
            "%s: unsupported types: dst: %s, src0: %s\n",
            .{ "unary_op", ggml.impl.c.ggml_type_name(dst.type), ggml.impl.c.ggml_type_name(src0.type) },
        );
        ggml.impl.abort("fatal error");
    }
}

/// Ports `vec_unary_op_functor` (unary-ops.cpp:180 @b1b770d90).
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
        y[i] = ggml.cpu.ops.common.fromF32(Dst, op.apply(ggml.cpu.ops.common.toF32(Src0, x[i])));
    }
}

/// Ports `apply_unary_op_functor` (unary-ops.cpp:191 @b1b770d90).
///
/// Asserts **`ggml_is_contiguous_1`**, a weaker condition than the plain
/// path's `ggml_is_contiguous_rows`. That difference is the C's and is kept.
fn applyUnaryOpFunctor(
    comptime Src0: type,
    comptime Dst: type,
    params: *const ggml.cpu.defs.ComputeParams,
    dst: *ggml.impl.c.ggml_tensor,
    op: anytype,
) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    ggml.impl.assert(
        ggml.impl.c.ggml_is_contiguous_1(src0) and ggml.impl.c.ggml_is_contiguous_1(dst) and ggml.impl.c.ggml_are_same_shape(src0, dst),
        "ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst)",
    );

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src0, dst);

    ggml.impl.assert(l.nb0 == @sizeOf(Dst), "nb0 == sizeof(dst_t)");
    ggml.impl.assert(l.nb00 == @sizeOf(Src0), "nb00 == sizeof(src0_t)");

    const ir0, const ir1 = ggml.cpu.ops.common.getThreadRange(params, src0);

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

/// Ports `unary_op_functor` (unary-ops.cpp:217 @b1b770d90).
fn unaryOpFunctor(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor, op: anytype) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    const F16 = ggml.impl.c.ggml_fp16_t;
    const BF16 = ggml.impl.c.ggml_bf16_t;

    if (src0.type == ggml.impl.c.GGML_TYPE_F32 and dst.type == ggml.impl.c.GGML_TYPE_F32) {
        applyUnaryOpFunctor(f32, f32, params, dst, op); // all f32
    } else if (src0.type == ggml.impl.c.GGML_TYPE_F16 and dst.type == ggml.impl.c.GGML_TYPE_F16) {
        applyUnaryOpFunctor(F16, F16, params, dst, op); // all f16
    } else if (src0.type == ggml.impl.c.GGML_TYPE_BF16 and dst.type == ggml.impl.c.GGML_TYPE_BF16) {
        applyUnaryOpFunctor(BF16, BF16, params, dst, op); // all bf16
    } else if (src0.type == ggml.impl.c.GGML_TYPE_BF16 and dst.type == ggml.impl.c.GGML_TYPE_F32) {
        applyUnaryOpFunctor(BF16, f32, params, dst, op);
    } else if (src0.type == ggml.impl.c.GGML_TYPE_F16 and dst.type == ggml.impl.c.GGML_TYPE_F32) {
        applyUnaryOpFunctor(F16, f32, params, dst, op);
    } else {
        ggml.impl.fprintfStderr(
            "%s: unsupported types: dst: %s, src0: %s\n",
            .{ "unary_op_functor", ggml.impl.c.ggml_type_name(dst.type), ggml.impl.c.ggml_type_name(src0.type) },
        );
        ggml.impl.abort("fatal error");
    }
}

// -----------------------------------------------------------------------------
// The exported entry points
//
// Ports `ggml_compute_forward_abs` through `ggml_compute_forward_trunc`
// (unary-ops.cpp:237 to :321 @b1b770d90), each a one-line call into `unaryOp`.
// Generated from the enum so the list cannot drift out of step with it; the
// C writes all twenty-two out by hand.

comptime {
    for (@typeInfo(Op).@"enum".field_names) |name| {
        @export(entry(@field(Op, name)), .{ .name = "ggml_compute_forward_" ++ name, .linkage = .strong });
    }
}

/// `ggml_compute_forward_<op>`, the entry point exported above, for callers
/// in Zig.
///
/// Parameters:
///   op: the unary op.
///
/// Return: the one function exported under that name.
pub fn entry(comptime op: Op) *const fn (*const ggml.cpu.defs.ComputeParams, *ggml.impl.c.ggml_tensor) callconv(.c) void {
    return &struct {
        fn f(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) callconv(.c) void {
            unaryOp(op, params, dst);
        }
    }.f;
}

/// Ports `ggml_compute_forward_xielu` (unary-ops.cpp:325 @b1b770d90).
///
/// The only entry point that reads `op_params`, and the only one that goes
/// through the functor family. Note the C reads indices **1 to 4**, not 0 to 3.
pub export fn ggml_compute_forward_xielu(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.c.ggml_tensor) callconv(.c) void {
    const op: Xielu = .{
        .alpha_n = ggml.impl.getOpParamsF32(dst, 1),
        .alpha_p = ggml.impl.getOpParamsF32(dst, 2),
        .beta = ggml.impl.getOpParamsF32(dst, 3),
        .eps = ggml.impl.getOpParamsF32(dst, 4),
    };
    unaryOpFunctor(params, dst, op);
}
