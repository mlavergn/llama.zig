//! Helpers shared by the `ops.cpp` kernels: the element conversion table and
//! the row range a thread owns.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-cpu/common.h` — `type_conversion_table` and
//!   `get_thread_range`
//!
//! at v0.3.0 (`c1d0e7a00`).
//!
//! `cpu/binary_ops.zig` and `cpu/unary_ops.zig` each carry their own copy of
//! these two, because each states its own provenance and the C has them in a
//! header all three translation units include. This is the copy for the
//! `ops.cpp` port, whose files share it through `module.zig`.

const std = @import("std");
const impl = @import("../../impl.zig");
const defs = @import("../defs.zig");

const c = impl.c;

pub const Tensor = defs.Tensor;
pub const ComputeParams = defs.ComputeParams;
pub const UnaryLocals = defs.UnaryLocals;
pub const BinaryLocals = defs.BinaryLocals;
pub const TernaryLocals = defs.TernaryLocals;

/// Ports the `to_f32` half of `type_conversion_table`
/// (ggml-cpu/common.h:48 @c1d0e7a00).
///
/// Parameters:
/// - `T`: the element type, one of `f32`, `ggml_fp16_t`, `ggml_bf16_t`, `i32`.
/// - `v`: the value to widen.
///
/// Return: `v` as an `f32`.
pub inline fn toF32(comptime T: type, v: T) f32 {
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
///
/// Parameters:
/// - `T`: the element type to produce.
/// - `v`: the `f32` to narrow.
///
/// Return: `v` as a `T`.
pub inline fn fromF32(comptime T: type, v: f32) T {
    return switch (T) {
        f32 => v,
        c.ggml_fp16_t => impl.fp32ToFp16(v),
        c.ggml_bf16_t => .{ .bits = impl.fp32ToBf16(v) },
        // `f32_to_i32` is a C cast, which truncates toward zero and is
        // undefined out of range; `@intFromFloat` would panic instead, so the
        // saturating form is the one that matches.
        i32 => @intFromFloat(std.math.clamp(@trunc(v), -2147483648.0, 2147483647.0)),
        else => @compileError("no conversion for " ++ @typeName(T)),
    };
}

/// Ports `get_thread_range` (ggml-cpu/common.h:74 @c1d0e7a00).
///
/// Parameters:
/// - `params`: the thread index and count.
/// - `src0`: the tensor whose rows are being split.
///
/// Return: the half-open row range `[ir0, ir1)` this thread owns.
pub fn getThreadRange(params: *const ComputeParams, src0: *const Tensor) struct { i64, i64 } {
    const ith: i64 = params.ith;
    const nth: i64 = params.nth;
    const nr: i64 = @intCast(c.ggml_nrows(src0));
    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    return .{ ir0, @min(ir0 + dr, nr) };
}

/// The C's `MIN` over `i64`, kept as a named helper so the ported loops read
/// like the C's.
pub inline fn min64(a: i64, b: i64) i64 {
    return @min(a, b);
}

/// `CACHE_LINE_SIZE_F32` (ggml-cpu/ops.h:21 @c1d0e7a00), the stride between
/// per-thread scratch slots in `params.wdata`.
///
/// `CACHE_LINE_SIZE` is 64 on this target -- the `#else` arm of ops.h:12-19,
/// the POWER9 arm being the one that uses 256.
pub const cache_line_size_f32: usize = 64 / @sizeOf(f32);
