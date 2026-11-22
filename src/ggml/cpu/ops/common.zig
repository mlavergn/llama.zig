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
//! at v1.0.0 (`b1b770d90`).
//!
//! `cpu/binary_ops.zig` and `cpu/unary_ops.zig` each carry their own copy of
//! these two, because each states its own provenance and the C has them in a
//! header all three translation units include. This is the copy for the
//! `ops.cpp` port, whose files share it through `module.zig`.

const std = @import("std");
const libm = @import("stdlib").libm;
const builtin = @import("builtin");
const impl = @import("../../impl.zig");
const convert = @import("../convert.zig");
const defs = @import("../defs.zig");
const threading = @import("../threading.zig");
const simd = @import("../simd.zig");

const c = impl.c;

pub const Tensor = defs.Tensor;
pub const ComputeParams = defs.ComputeParams;
pub const UnaryLocals = defs.UnaryLocals;
pub const BinaryLocals = defs.BinaryLocals;
pub const TernaryLocals = defs.TernaryLocals;

// -----------------------------------------------------------------------------
// Addressing and comparison helpers
//
// **Not ports.** The C writes these inline at every use -- `i01*nb01` for an
// offset, `(char *) t->data + off` for an access -- and spelling that out in
// Zig costs an `@intCast` and an `@alignCast` each time, which buries the
// index arithmetic the C's is meant to be read against.
//
// Each of the `ops.cpp` files grew its own copy while the port was being
// written in parallel; these are the single definitions they now alias. The
// names stay short for the same reason the C's expressions are terse.

/// A byte offset: index times stride, as the C writes `i01*nb01`.
pub inline fn byteOff(i: i64, nb: usize) usize {
    return @as(usize, @intCast(i)) * nb;
}

/// A stride widened to `i64`, for offset arithmetic the C does in signed
/// values.
pub inline fn sz(nb: usize) i64 {
    return @intCast(nb);
}

/// A typed many-pointer at a byte offset from a tensor's `data`.
///
/// The offset is `anytype` because the C mixes `int64_t` and `size_t` at
/// these sites and the port follows it rather than normalising.
pub inline fn ptr(comptime T: type, base: ?*anyopaque, byte_off: anytype) [*]T {
    const b: [*]u8 = @ptrCast(base.?);
    return @ptrCast(@alignCast(b + @as(usize, @intCast(byte_off))));
}

/// As `ptr`, for a single element rather than a run of them.
pub inline fn ref(comptime T: type, base: ?*anyopaque, byte_off: anytype) *T {
    const b: [*]u8 = @ptrCast(base.?);
    return @ptrCast(@alignCast(b + @as(usize, @intCast(byte_off))));
}

/// `ggml_barrier(params->threadpool)`, which the C writes directly.
pub inline fn barrier(params: *const ComputeParams) void {
    threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
}

/// Ports the `to_f32` half of `type_conversion_table`
/// (ggml-cpu/common.h:56 @b1b770d90), with `f8_e4m3_to_f32`
/// (ggml-cpu/common.h:46 @b1b770d90) for `ggml_fp8_e4m3_t`.
///
/// Parameters:
/// - `T`: the element type, one of `f32`, `ggml_fp16_t`, `ggml_bf16_t`,
///   `ggml_fp8_e4m3_t`, `i32`.
/// - `v`: the value to widen.
///
/// Return: `v` as an `f32`.
pub inline fn toF32(comptime T: type, v: T) f32 {
    return switch (T) {
        f32 => v,
        c.ggml_fp16_t => impl.fp16ToFp32(v),
        c.ggml_bf16_t => impl.bf16ToFp32(v.bits),
        c.ggml_fp8_e4m3_t => impl.f8E4m3ToFp32(v.bits),
        i32 => @floatFromInt(v),
        else => @compileError("no conversion for " ++ @typeName(T)),
    };
}

/// Ports the `from_f32` half of `type_conversion_table`
/// (ggml-cpu/common.h:56 @b1b770d90), with `f32_to_f8_e4m3`
/// (ggml-cpu/common.h:50 @b1b770d90) for `ggml_fp8_e4m3_t`.
///
/// Parameters:
/// - `T`: the element type to produce.
/// - `v`: the `f32` to narrow.
///
/// Return: `v` as a `T`.
pub inline fn fromF32(comptime T: type, v: f32) T {
    return switch (T) {
        f32 => v,
        c.ggml_fp16_t => convert.cpuFp32ToFp16(v),
        c.ggml_bf16_t => .{ .bits = impl.fp32ToBf16(v) },
        c.ggml_fp8_e4m3_t => .{ .bits = impl.fp32ToF8E4m3(v) },
        // `f32_to_i32` is a C cast, which truncates toward zero and is
        // undefined out of range or on a NaN; `truncTo` gives what aarch64's
        // `fcvtzs` gives (saturated, NaN to 0) where `@intFromFloat` traps
        i32 => impl.truncTo(i32, v),
        else => @compileError("no conversion for " ++ @typeName(T)),
    };
}

/// Ports `get_thread_range` (ggml-cpu/common.h:88 @b1b770d90).
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

/// `CACHE_LINE_SIZE_F32` (ggml-cpu/ops.h:20 @b1b770d90), the stride between
/// per-thread scratch slots in `params.wdata`.
///
/// `CACHE_LINE_SIZE` is 64 on this target -- the `#else` arm of ops.h:12-18,
/// POWER9 using 128 and s390x 256. That is so in C and C++
/// translation units alike: the `std::hardware_destructive_interference_size`
/// arm, which a C++ unit took depending on include order, is gone.
pub const cache_line_size_f32: usize = 64 / @sizeOf(f32);
