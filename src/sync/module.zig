//! Barrel for `src/sync/`: the port's mutex and condition variable, on futexes,
//! and the process's standard output and error, with no C library under them.
//!
//! **Not a port.** It is its own module because ggml, libllama, libmtmd and
//! the tools are separate modules, and a Zig module cannot import a file
//! outside its own directory.

pub const futex = @import("futex.zig");
pub const Mutex = @import("mutex.zig").Mutex;
pub const Condition = @import("condition.zig").Condition;
pub const stdio = @import("stdio.zig");

// ---------------------------------------------------------------------------
// Unit Tests

test {
    @import("std").testing.refAllDecls(@This());
}
