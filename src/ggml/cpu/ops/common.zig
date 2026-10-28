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
//! at v0.4.0 (`5266f24da`).
//!
//! `cpu/binary_ops.zig` and `cpu/unary_ops.zig` each carry their own copy of
//! these two, because each states its own provenance and the C has them in a
//! header all three translation units include. This is the copy for the
//! `ops.cpp` port, whose files share it through `module.zig`.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../../impl.zig");
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

/// Ports `std::min(a, b)` as libc++ defines it: `(b < a) ? b : a`.
///
/// **Not `@min`, which drops a NaN.** The C++'s returns `a` whenever the
/// comparison is false, so a NaN input passes through. Several of these
/// kernels feed it values that can be NaN.
pub inline fn stdMin(a: f32, b: f32) f32 {
    return if (b < a) b else a;
}

/// Ports `std::max(a, b)`: `(a < b) ? b : a`. Not `@max`, for the reason
/// given on `stdMin`.
pub inline fn stdMax(a: f32, b: f32) f32 {
    return if (a < b) b else a;
}

/// `ggml_barrier(params->threadpool)`, which the C writes directly.
pub inline fn barrier(params: *const ComputeParams) void {
    threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
}

/// Ports the `to_f32` half of `type_conversion_table`
/// (ggml-cpu/common.h:48 @5266f24da).
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
/// (ggml-cpu/common.h:48 @5266f24da).
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

/// Ports `get_thread_range` (ggml-cpu/common.h:74 @5266f24da).
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

/// `CACHE_LINE_SIZE_F32` (ggml-cpu/ops.h:21 @5266f24da), the stride between
/// per-thread scratch slots in `params.wdata`.
///
/// `CACHE_LINE_SIZE` is 64 on this target -- the `#else` arm of ops.h:12-19,
/// the POWER9 arm being the one that uses 256.
pub const cache_line_size_f32: usize = 64 / @sizeOf(f32);

/// A sine and cosine of the same argument, computed as the reference build
/// computes them.
pub const SinCos = struct { sin: f32, cos: f32 };

/// Mirrors the `__float2` that `__sincosf_stret` returns (math.h).
const Float2 = extern struct { sin: f32, cos: f32 };

extern fn __sincosf_stret(x: f32) Float2;
extern fn sinf(x: f32) f32;
extern fn cosf(x: f32) f32;

/// `sinf(x)` and `cosf(x)` **as clang emits them when both appear**.
///
/// Not a port of anything: the C calls `sinf` and `cosf`. But clang folds a
/// `sinf`/`cosf` pair on one argument into a single libcall -- on Darwin,
/// `__sincosf_stret` -- and Apple's combined routine does not round as the
/// separate ones do. Measured over 10.9M arguments: `sin` differs in 4.0% of
/// them, `cos` in 0.5%. The reference `ops.o` imports `___sincosf_stret` and
/// neither `_sinf` nor `_cosf`, so the port has to call the same routine to
/// produce the same bits. It is what `make ops-diff` caught on every f32 rope
/// and on `timestep_embedding`.
///
/// **Off Darwin there is no fold**, and the pair stays two calls. Measured
/// on the `aarch64-linux-gnu` reference: its `ops.o` imports `sinf` and
/// `cosf` and no `sincosf` at all. Linux defaults to `-fmath-errno`
/// (`simd.math_errno`), so the calls may set `errno` and clang will not
/// combine them; with `-fno-math-errno` the same pair on Linux becomes
/// `sincosf`, and with `-fmath-errno` on Darwin it stays `sinf` and `cosf`.
/// An earlier version of this function assumed `sincosf` off Darwin, which
/// matched nothing the reference calls.
///
/// Parameters:
/// - `x`: the argument.
///
/// Return: both values, by value.
pub inline fn sinCos(x: f32) SinCos {
    if (comptime !simd.math_errno) {
        const r = __sincosf_stret(x);
        return .{ .sin = r.sin, .cos = r.cos };
    }
    return .{ .sin = sinf(x), .cos = cosf(x) };
}
