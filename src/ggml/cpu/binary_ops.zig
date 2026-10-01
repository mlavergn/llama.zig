//! The four element-wise binary CPU kernels: add, sub, mul and div.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-cpu/binary-ops.cpp` — the kernels
//! - `llama.cpp/ggml/src/ggml-cpu/common.h`       — `type_conversion_table`
//!                                                  and `get_thread_range`
//!
//! Both at v0.3.0 (`c1d0e7a00`). Each declaration below names the C++ it
//! replaces and the line it began at. This file exports the same four C
//! symbols with the same signatures, so `ops.cpp` and the ported `ggml-cpu.c`
//! dispatch into it unchanged.
//!
//! # Templates become comptime, and that is the whole translation
//!
//! The C++ is three nested function templates — over the scalar operation, and
//! over the source and destination element types — expanded by a seven-arm
//! `if` chain on the runtime type triple. Zig's `comptime` parameters express
//! that directly: `applyBinaryOp` takes the same four compile-time arguments
//! and `binaryOp` is the same seven-arm chain. Nothing is dispatched at run
//! time that the C++ dispatches at compile time, or the reverse.
//!
//! # The Accelerate path is live and must stay
//!
//! `build/llamacpp.zig` defines `GGML_USE_ACCELERATE`, so the all-`f32` case
//! calls vDSP — `vDSP_vadd`, `vDSP_vsub`, `vDSP_vmul`, `vDSP_vdiv` — rather
//! than the scalar loop. These are element-wise with no accumulation, so they
//! are IEEE-exact and the choice costs nothing numerically; it is taken for
//! throughput, and dropping it would be a silent performance regression that
//! no gate in this project measures.
//!
//! The C selects the vDSP function by comparing the template's `op` pointer
//! against `op_add` and friends. A comptime enum is the same decision made at
//! the same time, without the function-pointer identity.

const std = @import("std");
const impl = @import("../impl.zig");
const defs = @import("defs.zig");
const c = impl.c;

const Tensor = defs.Tensor;
const ComputeParams = defs.ComputeParams;

// -----------------------------------------------------------------------------
// Accelerate
//
// Declared rather than imported: `src/ggml/impl.zig`'s `@cImport` does not take
// `<Accelerate/Accelerate.h>`, and these four are its only use here. The
// signature is vDSP's `(src1, stride1, src2, stride2, dst, strideD, n)`.

/// Mirrors `vDSP_Stride` (`vDSP.h`), the `long` element stride vDSP takes.
const Stride = c_long;

/// Mirrors `vDSP_Length`, the `unsigned long` element count.
const Length = c_ulong;

extern fn vDSP_vadd(a: [*]const f32, ia: Stride, b: [*]const f32, ib: Stride, dst: [*]f32, id: Stride, n: Length) void;
extern fn vDSP_vsub(a: [*]const f32, ia: Stride, b: [*]const f32, ib: Stride, dst: [*]f32, id: Stride, n: Length) void;
extern fn vDSP_vmul(a: [*]const f32, ia: Stride, b: [*]const f32, ib: Stride, dst: [*]f32, id: Stride, n: Length) void;
extern fn vDSP_vdiv(a: [*]const f32, ia: Stride, b: [*]const f32, ib: Stride, dst: [*]f32, id: Stride, n: Length) void;

// -----------------------------------------------------------------------------
// The scalar operations

/// Ports `op_add`, `op_sub`, `op_mul` and `op_div`
/// (binary-ops.cpp:9, 13, 17, 21 @c1d0e7a00).
///
/// The C++ passes these as function-pointer template arguments, which is how
/// it both selects the arithmetic and, under Accelerate, identifies which vDSP
/// routine to use. An enum carries both jobs here and keeps the vDSP lookup a
/// comptime switch rather than a pointer comparison.
const Op = enum {
    add,
    sub,
    mul,
    div,

    /// Applies the operation to two `f32`.
    inline fn apply(comptime op: Op, a: f32, b: f32) f32 {
        return switch (op) {
            .add => a + b,
            .sub => a - b,
            .mul => a * b,
            .div => a / b,
        };
    }

    /// The vDSP routine for this operation, used only on the all-`f32` path.
    inline fn vdsp(comptime op: Op) @TypeOf(&vDSP_vadd) {
        return switch (op) {
            .add => &vDSP_vadd,
            .sub => &vDSP_vsub,
            .mul => &vDSP_vmul,
            .div => &vDSP_vdiv,
        };
    }
};

/// Mirrors `GGML_USE_ACCELERATE`, which `build/llamacpp.zig` defines for this
/// target. A `comptime` branch rather than a runtime one, so the `extern`
/// references are not emitted where nothing provides them.
const use_accelerate = @import("builtin").os.tag == .macos;

// -----------------------------------------------------------------------------
// Element conversion

/// Ports `type_conversion_table` (ggml-cpu/common.h:48 @c1d0e7a00), the four
/// specialisations collapsed into two comptime functions.
///
/// Note **`vDSP_vsub`'s operand order is reversed** relative to the scalar
/// path, which the call site handles; nothing here depends on it.
inline fn toF32(comptime T: type, v: T) f32 {
    return switch (T) {
        f32 => v,
        c.ggml_fp16_t => impl.fp16ToFp32(v),
        c.ggml_bf16_t => impl.bf16ToFp32(v.bits),
        i32 => @floatFromInt(v),
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
        i32 => @intFromFloat(v),
        else => @compileError("no conversion for " ++ @typeName(T)),
    };
}

// -----------------------------------------------------------------------------
// The kernels

/// Ports `get_thread_range` (ggml-cpu/common.h:74 @c1d0e7a00).
///
/// Return: `.{ ir0, ir1 }`, the half-open row range this thread owns. The
/// last thread gets a short range rather than the division being exact.
fn threadRange(params: *const ComputeParams, src0: *const Tensor) struct { i64, i64 } {
    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr = c.ggml_nrows(src0);

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    return .{ ir0, ir1 };
}

/// Ports `vec_binary_op_contiguous` (binary-ops.cpp:26 @c1d0e7a00).
inline fn vecBinaryOpContiguous(
    comptime op: Op,
    comptime Src0: type,
    comptime Src1: type,
    comptime Dst: type,
    n: i64,
    z: [*]Dst,
    x: [*]const Src0,
    y: [*]const Src1,
) void {
    var i: usize = 0;
    while (i < @as(usize, @intCast(n))) : (i += 1) {
        z[i] = fromF32(Dst, op.apply(toF32(Src0, x[i]), toF32(Src1, y[i])));
    }
}

/// Ports `vec_binary_op_non_contiguous` (binary-ops.cpp:37 @c1d0e7a00).
///
/// `y` is indexed modulo `ne10` and by byte stride `nb10`, which is how a row
/// of `src1` broadcasts across a longer row of `src0`.
///
/// The C's index `i10` is `j10` here: Zig parses `iNN` as an integer type, so
/// every one of this file's row indices collides. Renamed digit for digit, as
/// `cpu/mulmat.zig` does for `i11`/`i12`/`i13`.
inline fn vecBinaryOpNonContiguous(
    comptime op: Op,
    comptime Src0: type,
    comptime Src1: type,
    comptime Dst: type,
    n: i64,
    ne10: i64,
    nb10: usize,
    z: [*]Dst,
    x: [*]const Src0,
    y: [*]const u8,
) void {
    var i: i64 = 0;
    while (i < n) : (i += 1) {
        const j10 = @mod(i, ne10);
        const y_ptr: *const Src1 = @ptrCast(@alignCast(y + @as(usize, @intCast(j10)) * nb10));
        const idx: usize = @intCast(i);
        z[idx] = fromF32(Dst, op.apply(toF32(Src0, x[idx]), toF32(Src1, y_ptr.*)));
    }
}

/// Ports `apply_binary_op` (binary-ops.cpp:50 @c1d0e7a00).
///
/// Walks this thread's rows of `src0`, broadcasting `src1` across the trailing
/// dimensions by modulo.
fn applyBinaryOp(
    comptime op: Op,
    comptime Src0: type,
    comptime Src1: type,
    comptime Dst: type,
    params: *const ComputeParams,
    dst: *Tensor,
) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(
        c.ggml_can_repeat(src1, src0) and c.ggml_are_same_shape(src0, dst),
        "ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst)",
    );

    const l = defs.BinaryLocals.of(src0, src1, dst);

    impl.assert(l.nb0 == @sizeOf(Dst), "nb0 == sizeof(dst_t)");
    impl.assert(l.nb00 == @sizeOf(Src0), "nb00 == sizeof(src0_t)");

    const ir0, const ir1 = threadRange(params, src0);
    const is_src1_contiguous_rows = c.ggml_is_contiguous_rows(src1);

    // The C++ picks the vDSP routine here, by comparing the template's `op`
    // pointer; `Op.vdsp` makes the same choice at compile time. The type guard
    // is the `if constexpr` at binary-ops.cpp:99.
    const all_f32 = Src0 == f32 and Src1 == f32 and Dst == f32;

    var ir = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne02 * l.ne01, l.ne01);
        const j01 = ir - j03 * l.ne02 * l.ne01 - j02 * l.ne01;

        const j13 = @mod(j03, l.ne13);
        const j12 = @mod(j02, l.ne12);
        const j11 = @mod(j01, l.ne11);

        const dst_bytes = @as([*]u8, @ptrCast(dst.data)) +
            @as(usize, @intCast(j03)) * l.nb3 + @as(usize, @intCast(j02)) * l.nb2 + @as(usize, @intCast(j01)) * l.nb1;
        const src0_bytes = @as([*]const u8, @ptrCast(src0.data)) +
            @as(usize, @intCast(j03)) * l.nb03 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j01)) * l.nb01;
        const src1_bytes = @as([*]const u8, @ptrCast(src1.data)) +
            @as(usize, @intCast(j13)) * l.nb13 + @as(usize, @intCast(j12)) * l.nb12 + @as(usize, @intCast(j11)) * l.nb11;

        const dst_ptr: [*]Dst = @ptrCast(@alignCast(dst_bytes));
        const src0_ptr: [*]const Src0 = @ptrCast(@alignCast(src0_bytes));

        if (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const src1_ptr: [*]const Src1 = @ptrCast(@alignCast(src1_bytes));
            const nr0 = @divTrunc(l.ne00, l.ne10);

            var r: i64 = 0;
            while (r < nr0) : (r += 1) {
                const off: usize = @intCast(r * l.ne10);
                if (use_accelerate and all_f32) {
                    // vDSP takes (src1, src0) in this order; the C++ passes
                    // them the same way round, and for `vsub` that is what
                    // makes `src0 - src1` come out, not the reverse.
                    op.vdsp()(
                        @ptrCast(src1_ptr),
                        1,
                        @ptrCast(src0_ptr + off),
                        1,
                        @ptrCast(dst_ptr + off),
                        1,
                        @intCast(l.ne10),
                    );
                    continue;
                }
                vecBinaryOpContiguous(op, Src0, Src1, Dst, l.ne10, dst_ptr + off, src0_ptr + off, src1_ptr);
            }
        } else {
            vecBinaryOpNonContiguous(op, Src0, Src1, Dst, l.ne0, l.ne10, l.nb10, dst_ptr, src0_ptr, src1_bytes);
        }
    }
}

/// Ports `binary_op` (binary-ops.cpp:116 @c1d0e7a00).
///
/// The seven supported type triples, in the C's order. Anything else aborts,
/// as the C does — the TODO above it in the C++ about using the traits table
/// instead is reproduced as written rather than acted on.
fn binaryOp(comptime op: Op, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const F16 = c.ggml_fp16_t;
    const BF16 = c.ggml_bf16_t;

    if (src0.type == c.GGML_TYPE_F32 and src1.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32) {
        applyBinaryOp(op, f32, f32, f32, params, dst); // all f32
    } else if (src0.type == c.GGML_TYPE_F16 and src1.type == c.GGML_TYPE_F16 and dst.type == c.GGML_TYPE_F16) {
        applyBinaryOp(op, F16, F16, F16, params, dst); // all f16
    } else if (src0.type == c.GGML_TYPE_BF16 and src1.type == c.GGML_TYPE_BF16 and dst.type == c.GGML_TYPE_BF16) {
        applyBinaryOp(op, BF16, BF16, BF16, params, dst); // all bf16
    } else if (src0.type == c.GGML_TYPE_BF16 and src1.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_BF16) {
        applyBinaryOp(op, BF16, f32, BF16, params, dst);
    } else if (src0.type == c.GGML_TYPE_BF16 and src1.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32) {
        applyBinaryOp(op, BF16, f32, f32, params, dst);
    } else if (src0.type == c.GGML_TYPE_F16 and src1.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F16) {
        applyBinaryOp(op, F16, f32, F16, params, dst);
    } else if (src0.type == c.GGML_TYPE_F16 and src1.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32) {
        applyBinaryOp(op, F16, f32, f32, params, dst);
    } else {
        impl.logError(
            "%s: unsupported types: dst: %s, src0: %s, src1: %s\n",
            .{ "binary_op", c.ggml_type_name(dst.type), c.ggml_type_name(src0.type), c.ggml_type_name(src1.type) },
        );
        impl.abort("unsupported types");
    }
}

/// Ports `ggml_compute_forward_add_non_quantized` (binary-ops.cpp:140 @c1d0e7a00).
pub export fn ggml_compute_forward_add_non_quantized(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    binaryOp(.add, params, dst);
}

/// Ports `ggml_compute_forward_sub` (binary-ops.cpp:144 @c1d0e7a00).
pub export fn ggml_compute_forward_sub(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    binaryOp(.sub, params, dst);
}

/// Ports `ggml_compute_forward_mul` (binary-ops.cpp:148 @c1d0e7a00).
pub export fn ggml_compute_forward_mul(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    binaryOp(.mul, params, dst);
}

/// Ports `ggml_compute_forward_div` (binary-ops.cpp:152 @c1d0e7a00).
pub export fn ggml_compute_forward_div(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    binaryOp(.div, params, dst);
}
