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
const builtin = @import("builtin");

/// `std::thread::hardware_concurrency()`.
///
/// Return: the CPU count, 0 when it cannot be read.
pub fn hardwareConcurrency() u32 {
    if (comptime builtin.os.tag == .windows) return @intCast(std.Thread.getCpuCount() catch 0);
    const rc = libc.sysconf(sc_nprocessors_onln);
    return if (rc < 0) 0 else @intCast(rc);
}

/// `_SC_NPROCESSORS_ONLN`, which std's `_SC` lacks on Darwin.
const sc_nprocessors_onln: c_int = if (builtin.os.tag.isDarwin()) 58 else 84;

const libc = @import("libc.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the machine has a CPU" {
    try std.testing.expect(hardwareConcurrency() >= 1);
}
