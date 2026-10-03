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
const threading = @import("../threading.zig");
const neon = @import("../quants/arm/neon.zig");

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

// -----------------------------------------------------------------------------
// The NEON SIMD mapping
//
// `simd-mappings.h` is macro-only, so these are the NEON arm of those macros
// written out once. `recurrent.zig` and `ssm.zig` both run the same
// four-accumulator reduction; it was open-coded in one and named in the other
// while the port was written in parallel.

pub const f32x4 = neon.f32x4;

/// `GGML_F32_STEP` and `GGML_F32_EPR` (simd-mappings.h:337, 338 @c1d0e7a00),
/// the NEON arm: four `float32x4_t` per step.
pub const f32_step: usize = 16;
pub const f32_epr: usize = 4;

/// Ports the NEON `GGML_F32x4_REDUCE` (simd-mappings.h:349 @c1d0e7a00):
/// halve, halve, then one **pairwise** `vaddvq_f32`.
///
/// The pairwise step is load-bearing. `neon.addvq_f32` reduces as
/// `(l0+l1)+(l2+l3)`; `@reduce(.Add, ...)` is ordered and differs on 23.5% of
/// random lane vectors. See `CLAUDE.md`, "Porting notes".
///
/// The C widens the result to `ggml_float` and every caller here narrows it
/// straight back to `float`, which is lossless, so the round trip is left
/// out.
pub inline fn f32VecReduce(x: *[4]f32x4) f32 {
    x[0] = x[0] + x[2];
    x[1] = x[1] + x[3];
    x[0] = x[0] + x[1];
    return neon.addvq_f32(x[0]);
}

/// A four-lane load, as `GGML_F32x4_LOAD` (simd-mappings.h:343 @c1d0e7a00).
pub inline fn load4(p: [*]const f32) f32x4 {
    return p[0..4].*;
}

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
