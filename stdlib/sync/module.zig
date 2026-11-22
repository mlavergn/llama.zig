//! Barrel for `stdlib/sync/`: the port's mutex and condition variable, on
//! futexes, with no C library under them.
//!
//! **Not a port.** Every lock in the port is one of these: never
//! `pthread_mutex_t`, `pthread_cond_t`, SRW locks or `std.Io.Mutex`.

pub const futex = @import("futex.zig");
pub const Mutex = @import("mutex.zig").Mutex;
pub const Condition = @import("condition.zig").Condition;

// ---------------------------------------------------------------------------
// Unit Tests

test {
    @import("std").testing.refAllDecls(@This());
}
