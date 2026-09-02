//! Types and constants the CPU backend shares, restated from headers that
//! `@cImport` cannot reach.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h` — `ggml_compute_params`
//! - `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`      — `ggml_threadpool` and
//!                                                   `ggml_compute_state`
//! - `llama.cpp/ggml/src/ggml-cpu/ops.h`           — `CACHE_LINE_SIZE`
//! - `llama.cpp/ggml/include/ggml.h`               — the tensor-locals macros
//!
//! all at v0.3.0 (`c1d0e7a00`). Each declaration names the C it came from and
//! the line it began at, using the bare filename.
//!
//! # Why these are hand-written
//!
//! `ggml-cpu-impl.h` includes `ggml-impl.h`, which includes `<arm_neon.h>` —
//! the same reason `src/ggml/impl.zig` exists. `ggml_threadpool` is worse: the
//! C only ever declares it in `ggml.h`, defining it inside `ggml-cpu.c`, so
//! there is no header to read it from at all. That opacity is also a licence:
//! nothing outside this port sees the layout, so the mutex and condition
//! variable can be Zig's rather than `pthread_t`-shaped.

const std = @import("std");
const impl = @import("../impl.zig");
const c = impl.c;

pub const Tensor = c.ggml_tensor;
pub const CGraph = impl.CGraph;

/// Ports `CACHE_LINE_SIZE` (ops.h:17 @c1d0e7a00), the non-POWER, non-s390x arm.
pub const cache_line_size = 64;

/// Ports `CACHE_LINE_SIZE_F32` (ops.h:21 @c1d0e7a00).
pub const cache_line_size_f32 = cache_line_size / @sizeOf(f32);

/// Ports `GGML_CACHE_LINE` (ggml-cpu.c:60 @c1d0e7a00). The C hardcodes 64 with a note
/// that it wants `std::hardware_destructive_interference_size`; `ops.h` picks
/// the same number for this architecture, so the two agree here.
pub const cache_align = 64;

/// Ports `GGML_THREADPOOL_N_THREADS_MASK` (ggml-cpu.c:205 @c1d0e7a00).
pub const n_threads_mask: c_int = 0xffff;

/// Ports `GGML_THREADPOOL_N_THREADS_BITS` (ggml-cpu.c:206 @c1d0e7a00).
pub const n_threads_bits = 16;

/// Ports `struct ggml_compute_params` (ggml-cpu-impl.h:18 @c1d0e7a00).
///
/// Passed by pointer to every `ggml_compute_forward_*` in the still-C++
/// `ops.cpp`, so this layout is a hard ABI contract, not an internal choice.
pub const ComputeParams = extern struct {
    /// Thread index.
    ith: c_int,
    /// Number of threads.
    nth: c_int,
    /// Size of the work buffer shared by all threads.
    wsize: usize,
    wdata: ?*anyopaque,
    threadpool: ?*Threadpool,
    /// Take the reference path, skipping fused and vectorised kernels.
    use_ref: bool,
};

/// Ports `struct ggml_threadpool` (ggml-cpu.c:480 @c1d0e7a00).
///
/// Opaque to everything outside this translation unit — `ggml.h` only forward
/// declares it — so this is a plain Zig struct rather than an `extern` one,
/// and the synchronisation primitives are `std.Thread`'s.
///
/// The `align(cache_align)` on the three hot counters is the C's
/// `GGML_CACHE_ALIGN`: `n_barrier`, `n_barrier_passed` and `current_chunk` are
/// hammered by every thread, and sharing a cache line between them costs more
/// than the padding does.
pub const Threadpool = struct {
    mutex: std.c.pthread_mutex_t,
    cond: std.c.pthread_cond_t,

    cgraph: ?*const CGraph,
    cplan: ?*const c.ggml_cplan,

    /// Holds both the graph counter and the active thread count: the low
    /// `n_threads_bits` are the thread count, the rest a monotonic graph id.
    /// One atomic covers both so a worker can read them without tearing.
    n_graph: std.atomic.Value(c_int),
    n_barrier: std.atomic.Value(c_int) align(cache_align),
    n_barrier_passed: std.atomic.Value(c_int) align(cache_align),
    /// The chunk `ggml_compute_forward_mul_mat` hands out next.
    current_chunk: std.atomic.Value(c_int) align(cache_align),

    /// Atomic as an annotation for thread-sanitizer, per the C.
    stop: std.atomic.Value(bool),
    pause: std.atomic.Value(bool),
    abort: std.atomic.Value(c_int),

    workers: ?[*]ComputeState,
    n_threads: c_int,
    prio: i32,
    poll: u32,

    ec: c.enum_ggml_status,
};

/// Ports `struct ggml_compute_state` (ggml-cpu.c:507 @c1d0e7a00), the per-thread half of
/// the threadpool.
pub const ComputeState = struct {
    thrd: std.Thread,
    last_graph: c_int,
    pending: bool,

    cpumask: [c.GGML_MAX_N_THREADS]bool,
    threadpool: *Threadpool,
    ith: c_int,
};

/// Ports `ggml_thread_cpu_relax` (ggml-cpu.c:520 @c1d0e7a00), the aarch64 arm.
///
/// `std.atomic.spinLoopHint` emits the same instruction, but not the memory
/// clobber: the C's barrier loop re-reads `n_barrier_passed` every iteration
/// and must not have that read hoisted out.
pub inline fn cpuRelax() void {
    asm volatile ("yield" ::: .{ .memory = true });
}

/// The locals `GGML_TENSOR_BINARY_OP_LOCALS` (ggml.h:320 @c1d0e7a00) declares.
///
/// The macro drops twenty-four names into the enclosing scope. Zig has no
/// equivalent, and spelling them out at each of the four call sites invites a
/// transposed digit, so they are gathered into one struct built by `of`.
pub const BinaryLocals = struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: usize,
    nb01: usize,
    nb02: usize,
    nb03: usize,

    ne10: i64,
    ne11: i64,
    ne12: i64,
    ne13: i64,
    nb10: usize,
    nb11: usize,
    nb12: usize,
    nb13: usize,

    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb0: usize,
    nb1: usize,
    nb2: usize,
    nb3: usize,

    /// Parameters:
    /// - `src0`, `src1`: the two inputs, supplying the `*0*` and `*1*` names.
    /// - `dst`: the result, supplying the unsuffixed names.
    ///
    /// Return: the twenty-four locals, by value.
    pub fn of(src0: *const Tensor, src1: *const Tensor, dst: *const Tensor) BinaryLocals {
        return .{
            .ne00 = src0.ne[0],
            .ne01 = src0.ne[1],
            .ne02 = src0.ne[2],
            .ne03 = src0.ne[3],
            .nb00 = src0.nb[0],
            .nb01 = src0.nb[1],
            .nb02 = src0.nb[2],
            .nb03 = src0.nb[3],

            .ne10 = src1.ne[0],
            .ne11 = src1.ne[1],
            .ne12 = src1.ne[2],
            .ne13 = src1.ne[3],
            .nb10 = src1.nb[0],
            .nb11 = src1.nb[1],
            .nb12 = src1.nb[2],
            .nb13 = src1.nb[3],

            .ne0 = dst.ne[0],
            .ne1 = dst.ne[1],
            .ne2 = dst.ne[2],
            .ne3 = dst.ne[3],
            .nb0 = dst.nb[0],
            .nb1 = dst.nb[1],
            .nb2 = dst.nb[2],
            .nb3 = dst.nb[3],
        };
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "compute params match the C layout" {
    // `ops.cpp` and `sgemm.cpp` are still C++ and read this struct directly,
    // so a mismatch here is a wrong answer rather than a link error.
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ComputeParams, "ith"));
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(ComputeParams, "nth"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(ComputeParams, "wsize"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(ComputeParams, "wdata"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(ComputeParams, "threadpool"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(ComputeParams, "use_ref"));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(ComputeParams));
}

test "the hot threadpool counters do not share a cache line" {
    const barrier = @offsetOf(Threadpool, "n_barrier");
    const passed = @offsetOf(Threadpool, "n_barrier_passed");
    const chunk = @offsetOf(Threadpool, "current_chunk");

    try std.testing.expect(passed - barrier >= cache_align);
    try std.testing.expect(chunk - passed >= cache_align);
}
