//! Barrel for `stdlib/`: what upstream gets from the C library and the C++
//! standard library -- locks, standard streams, files, the file system,
//! clocks, the environment, number parsing, strings -- as one module on
//! Zig's `std`, shared by ggml, libllama, libmtmd, the CLI, the tools, the
//! server and the gates.
//!
//! **Not a port**, except where a file says so: `subproc.zig` ports
//! upstream's `common_subproc` and the vendored `subprocess.h` under it,
//! the one process layer upstream has. It is its own module because those
//! are separate modules,
//! and a Zig module cannot import a file outside its own directory. Every
//! component takes the same instance, so a lock or a stream buffer is one
//! object in a process.
//!
//! # Rules
//!
//! - One canonical helper per behaviour. Where upstream does two different
//!   things (a stream read that keeps what it read before an error, a read
//!   that fails whole), each has its own name, saying which C or C++ idiom
//!   it reproduces; a caller picks the one upstream's code at that site
//!   uses. Never a local copy.
//! - Library-safe throughout: no `@panic`, no `std.debug.print`, no
//!   `std.log`; an `Io` is `std.Io.Threaded.global_single_threaded`, never
//!   `Threaded.init`.
//! - Imports only `std` and `builtin`.

pub const sync = @import("sync/module.zig");
pub const stdio = @import("stdio.zig");
pub const errno = @import("errno.zig");
pub const file = @import("file.zig");
pub const fs = @import("fs.zig");
pub const path = @import("path.zig");
pub const clock = @import("clock.zig");
pub const env = @import("env.zig");
pub const num = @import("num.zig");
pub const strings = @import("strings.zig");
pub const libcxx = @import("libcxx.zig");
pub const libm = @import("libm.zig");
pub const libc = @import("libc.zig");
pub const thread = @import("thread.zig");
pub const tty = @import("tty.zig");
pub const subproc = @import("subproc.zig");

// ---------------------------------------------------------------------------
// Unit Tests

test {
    @import("std").testing.refAllDecls(@This());
}
