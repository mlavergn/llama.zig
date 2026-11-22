//! What the port asks about threads: how many CPUs `std::thread` says the
//! machine has.
//!
//! # Provenance
//!
//! **Not a port.** `hardwareConcurrency` is libc++'s
//! `thread::hardware_concurrency` (`src/thread.cpp`):
//! `sysconf(_SC_NPROCESSORS_ONLN)` where the C library defines it -- every
//! CPU online, whatever the process's affinity mask -- and `GetSystemInfo`'s
//! `dwNumberOfProcessors` on Windows, which std reads from the process
//! environment block. std's own count differs on Linux (it honours the
//! affinity mask) and on Darwin (`hw.logicalcpu`), so `sysconf` stays the
//! C library's until std reads the online count.

const std = @import("std");
const mod = @import("module.zig");
const builtin = @import("builtin");

/// `std::thread::hardware_concurrency()`.
///
/// Return: the CPU count, 0 when it cannot be read.
pub fn hardwareConcurrency() u32 {
    if (comptime builtin.os.tag == .windows) return @intCast(std.Thread.getCpuCount() catch 0);
    const rc = mod.libc.sysconf(sc_nprocessors_onln);
    return if (rc < 0) 0 else @intCast(rc);
}

/// `pthread_setaffinity_np(pthread_self(), size, set)` as glibc runs it:
/// the calling thread's `sched_setaffinity`, the error number returned
/// rather than left in `errno`. Linux only.
///
/// Parameters:
///   size: the bytes of `set` the kernel reads.
///   set: a `cpu_set_t` of at least `size` bytes.
///
/// Return: 0, or the error number.
pub fn setAffinity(size: usize, set: *const anyopaque) c_int {
    const linux = std.os.linux;
    const rc = linux.syscall3(.sched_setaffinity, 0, size, @intFromPtr(set));
    return @intCast(@backingInt(linux.errno(rc)));
}

/// `pthread_getaffinity_np(pthread_self(), size, set)` as glibc runs it:
/// `sched_getaffinity`, the bytes past what the kernel wrote zeroed. Linux
/// only.
///
/// Parameters:
///   size: the bytes of `set` to fill.
///   set: a `cpu_set_t` of at least `size` bytes.
///
/// Return: 0, or the error number.
pub fn getAffinity(size: usize, set: *anyopaque) c_int {
    const linux = std.os.linux;
    const rc = linux.syscall3(.sched_getaffinity, 0, size, @intFromPtr(set));
    const e = linux.errno(rc);
    if (e != .SUCCESS) return @intCast(@backingInt(e));
    if (rc < size) @memset(@as([*]u8, @ptrCast(set))[rc..size], 0);
    return 0;
}

/// `_SC_NPROCESSORS_ONLN`, which std's `_SC` lacks on Darwin.
const sc_nprocessors_onln: c_int = if (builtin.os.tag.isDarwin()) 58 else 84;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the machine has a CPU" {
    try std.testing.expect(hardwareConcurrency() >= 1);
}
