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
const builtin = @import("builtin");
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

/// A sine and cosine of the same argument, computed as the reference build
/// computes them.
pub const SinCos = struct { sin: f32, cos: f32 };

/// Mirrors the `__float2` that `__sincosf_stret` returns (math.h).
const Float2 = extern struct { sin: f32, cos: f32 };

extern fn __sincosf_stret(x: f32) Float2;
extern fn sincosf(x: f32, s: *f32, cc: *f32) void;

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
/// Off Darwin clang emits `sincosf` for the same pair, so that is the
/// fallback; Stage 5 has to re-measure it against glibc rather than assume.
///
/// Parameters:
/// - `x`: the argument.
///
/// Return: both values, by value.
pub inline fn sinCos(x: f32) SinCos {
    if (comptime builtin.os.tag.isDarwin()) {
        const r = __sincosf_stret(x);
        return .{ .sin = r.sin, .cos = r.cos };
    }
    var sv: f32 = undefined;
    var cv: f32 = undefined;
    sincosf(x, &sv, &cv);
    return .{ .sin = sv, .cos = cv };
}

/// Whether the reference compiler runs the **vectorized** form of a plain C
/// loop that reduces `acc += a[i] * b[i]` while storing to `store[i]`.
///
/// The loop vectorizer turns such a loop into a strict in-order reduction:
/// groups of four products are rounded (`fmul.4s`) and added one lane at a
/// time, and only what is left over runs scalar, fused (`fmadd`). But it only
/// takes that form when two runtime guards pass, read off the LLVM IR of the
/// same loop compiled by the same toolchain:
///
/// - more than three iterations, else the scalar loop runs throughout; and
/// - for every array the loop loads, `(store_row - load_row)` as an
///   **unsigned** byte difference is at least 64 -- the vector body's
///   footprint. A load that starts up to 63 bytes below the store could be
///   overwritten before it is read, so the loop falls back to scalar.
///
/// The second guard is not academic. `ssm_scan` reads the previous state from
/// the buffer it is writing for every token after the first (`s0 = s`), so
/// the difference is zero and those tokens take the all-fused scalar loop
/// while the first one does not.
///
/// Parameters:
/// - `trip`: the loop's iteration count.
/// - `store`: the first address the loop stores to.
/// - `loads`: the first address of each array the loop loads.
///
/// Return: true when the vectorized form runs.
pub inline fn strictLoopVectorized(trip: i64, store: *const anyopaque, loads: []const *const anyopaque) bool {
    if (trip <= 3) return false;
    const st = @intFromPtr(store);
    for (loads) |l| {
        if (st -% @intFromPtr(l) < 64) return false;
    }
    return true;
}
