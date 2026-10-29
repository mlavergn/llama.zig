//! The global critical section ggml uses to guard its process-wide tables.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-threading.cpp` (v0.4.1, `b29c606e2`),
//! 12 lines of C++. Each declaration below names the C++ it replaces and the
//! line it began at. This file exports the same C symbols with the same
//! signatures, so the C and C++ that call into it link unchanged.
//!
//! **The first C++ translation unit ported.** It is here as the warm-up: the
//! contract is three symbols, the callers are all already Zig, and it proves
//! the Stage 3 swap mechanism carries over to a `.cpp` file unchanged.
//!
//! # Why libc++'s primitive and not a Zig mutex
//!
//! The C++ holds a namespace-scope `std::mutex` and calls `.lock()` and
//! `.unlock()`. libc++'s `std::mutex` *is* a `pthread_mutex_t` initialised to
//! `PTHREAD_MUTEX_INITIALIZER` on POSIX and an `SRWLOCK` on Windows, so
//! `impl.CxxMutex`, which drives exactly those, is the closer translation
//! rather than a substitution.
//!
//! It is also the only workable one. Decision 39: Zig 0.16's `std.Io.Mutex`
//! takes an `Io` on every lock, and a function entered through a C ABI has
//! none to thread through. `src/ggml/cpu/threading.zig` reaches the same
//! conclusion for the threadpool's mutex and condition variable.
//!
//! # One deliberate difference
//!
//! The C++ object is a global with a destructor, so its `~mutex()` runs at
//! exit through `__cxa_atexit`. This port does not destroy the mutex. That is
//! not an omission: destroying a process-wide lock during exit, while another
//! thread may still be inside it, is a hazard the C++ inherits from RAII and
//! gains nothing from. Nothing observes the difference — the process is
//! leaving.
//!
//! # Callers
//!
//! All four are already ported Zig: `ggml_new_context`/`ggml_free`
//! (`context.zig`), the quantization type registry (`quantize.zig`), and CPU
//! feature detection (`cpu/features.zig`). `kleidiai.cpp` also calls it
//! upstream, but that backend is not compiled here.

const std = @import("std");
const impl = @import("impl.zig");

/// Mirrors `ggml_critical_section_mutex` (ggml-threading.cpp:4 @b29c606e2).
///
/// Exported rather than file-private because the C++ global it replaces is
/// exported too, and `scripts/port-coverage` compares the whole symbol list.
/// Nothing outside this file references it, in the C++ or here.
///
/// `.{}` is a static initialisation with no runtime setup — the same as
/// libc++'s `constexpr mutex()`.
export var ggml_critical_section_mutex: impl.CxxMutex = .{};

/// Ports `ggml_critical_section_start` (ggml-threading.cpp:6 @b29c606e2).
///
/// Return: nothing. Blocks until the lock is held.
export fn ggml_critical_section_start() callconv(.c) void {
    // `std::mutex::lock` throws on error; a C ABI has nowhere to put that, and
    // the only documented failures here (deadlock, invalid mutex) are bugs
    // rather than conditions. The C++ would terminate; this returns having not
    // acquired it, which is the same unrecoverable state.
    ggml_critical_section_mutex.lock();
}

/// Ports `ggml_critical_section_end` (ggml-threading.cpp:10 @b29c606e2).
///
/// Return: nothing.
export fn ggml_critical_section_end() callconv(.c) void {
    ggml_critical_section_mutex.unlock();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the critical section is reentrant across sequential acquisitions" {
    // A plain (non-recursive) mutex, so this is lock/unlock twice in sequence,
    // not a nested lock. It is the shape every caller uses.
    ggml_critical_section_start();
    ggml_critical_section_end();
    ggml_critical_section_start();
    ggml_critical_section_end();
}

test "the critical section actually excludes" {
    // Without mutual exclusion the increments race and the total comes out
    // short. Negative-tested by removing the lock: it fails.
    const Counter = struct {
        var value: u64 = 0;

        fn bump(times: usize) void {
            for (0..times) |_| {
                ggml_critical_section_start();
                defer ggml_critical_section_end();
                // Deliberately non-atomic: the lock is what makes it safe.
                value += 1;
            }
        }
    };

    Counter.value = 0;
    const per_thread = 20_000;
    var threads: [4]std.Thread = undefined;
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, Counter.bump, .{per_thread});
    for (threads) |t| t.join();

    try std.testing.expectEqual(@as(u64, threads.len * per_thread), Counter.value);
}
