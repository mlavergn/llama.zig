//! The global critical section ggml uses to guard its process-wide tables.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-threading.cpp` (v1.0.0, `b1b770d90`),
//! 12 lines of C++. Each declaration below names the C++ it replaces and the
//! line it began at. This file exports the same C symbols with the same
//! signatures, so the C and C++ that call into it link unchanged.
//!
//! **The first C++ translation unit ported.** It is here as the warm-up: the
//! contract is three symbols, the callers are all already Zig, and it proves
//! the C-ABI swap carries over to a `.cpp` file unchanged.
//!
//! # The mutex
//!
//! The C++ holds a namespace-scope `std::mutex` and calls `.lock()` and
//! `.unlock()`. Here it is `impl.Mutex`, the port's futex mutex: no C
//! library, and no `std.Io`, which a function entered through a C ABI has
//! none of.
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
const stdlib = @import("stdlib");
const mod = @import("module.zig");

/// Mirrors `ggml_critical_section_mutex` (ggml-threading.cpp:4 @b1b770d90).
///
/// Exported rather than file-private because the C++ global it replaces is
/// exported too, and `scripts/port-coverage` compares the whole symbol list.
/// Nothing outside this file references it, in the C++ or here.
///
/// libc++'s `constexpr mutex()`.
export var ggml_critical_section_mutex: stdlib.sync.Mutex = .{};

/// Ports `ggml_critical_section_start` (ggml-threading.cpp:6 @b1b770d90).
///
/// Return: nothing. Blocks until the lock is held.
pub fn ggml_critical_section_start() callconv(.c) void {
    // `std::mutex::lock` throws on error; a C ABI has nowhere to put that, and
    // the only documented failures here (deadlock, invalid mutex) are bugs
    // rather than conditions. The C++ would terminate; this returns having not
    // acquired it, which is the same unrecoverable state.
    ggml_critical_section_mutex.lock();
}

/// Ports `ggml_critical_section_end` (ggml-threading.cpp:10 @b1b770d90).
///
/// Return: nothing.
pub fn ggml_critical_section_end() callconv(.c) void {
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
