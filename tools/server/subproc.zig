//! A child server process as router mode watches it: its combined output
//! read without blocking, and a waiter that sleeps until one of several
//! children has output or is woken.
//!
//! # Provenance
//!
//! Ports `server_subproc` and `server_subproc::waiter` from
//! `llama.cpp/tools/server/server-common.cpp` and
//! `llama.cpp/tools/server/server-common.h` (llama.cpp v0.5.0,
//! `7fe450e19`), their POSIX arms. Each declaration names what it ports
//! and the line it began at.
//!
//! # Generic over `common_subproc`
//!
//! `server_subproc` wraps a `common_subproc` (`common/subproc.cpp`, Stage 10
//! step 6, with router mode in step 7). It is written against the members
//! it calls -- `alive()`, `terminate()`, `join()` and `stdoutFile()`, the
//! child's combined output as a C `FILE *` or null -- so `Subproc(P)` is
//! the C++ struct over any such `P`. Windows' arms (`PeekNamedPipe`) wait
//! for the Windows server.

const std = @import("std");
const tools = @import("tools_common");
const common = @import("common.zig");

const log = tools.log;

extern fn fileno(f: *anyopaque) c_int;
extern fn fcntl(fd: c_int, cmd: c_int, ...) c_int;
extern fn read(fd: c_int, buf: [*]u8, len: usize) isize;
extern fn write(fd: c_int, buf: [*]const u8, len: usize) isize;
extern fn close(fd: c_int) c_int;
extern fn pipe(fds: *[2]c_int) c_int;
extern fn strerror(e: c_int) [*:0]const u8;

const F_GETFL: c_int = 3;
const F_SETFL: c_int = 4;
const O_NONBLOCK: c_int = if (@import("builtin").os.tag.isDarwin()) 0x0004 else 0o4000;

fn errno() c_int {
    return std.c._errno().*;
}

fn setNonBlocking(fd: c_int) void {
    _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, @as(c_int, 0)) | O_NONBLOCK);
}

/// Ports `server_subproc` (server-common.h:619 @7fe450e19) over a
/// `common_subproc`-like `P` (see the file's doc comment).
pub fn Subproc(comptime P: type) type {
    return struct {
        const Self = @This();

        sproc: P,
        stopped: std.atomic.Value(bool) = .init(false), // set by the monitor once the process exited and was reaped

        out_handle: isize = -1, // fd on POSIX, HANDLE on Windows; taken lazily from sproc

        /// Ports `is_alive` (server-common.h:623 @7fe450e19).
        pub fn isAlive(self: *Self) bool {
            return self.sproc.alive();
        }

        /// Ports `terminate` (server-common.h:624 @7fe450e19).
        pub fn terminate(self: *Self) void {
            self.sproc.terminate();
        }

        /// Ports `join` (server-common.h:625 @7fe450e19).
        pub fn join(self: *Self) c_int {
            return self.sproc.join();
        }

        /// Ports `server_subproc`'s `has_output` (server-common.cpp:1864 @7fe450e19): true when the
        /// child's combined output pipe is available; the first call makes
        /// it non-blocking.
        pub fn hasOutput(self: *Self) bool {
            if (self.out_handle >= 0) {
                return true;
            }
            const f = self.sproc.stdoutFile() orelse return false; // combined stdout/stderr
            const fd = fileno(f);
            if (fd >= 0) {
                setNonBlocking(fd);
                self.out_handle = fd;
            }
            return self.out_handle >= 0;
        }

        /// Ports `server_subproc`'s `read_output` (server-common.cpp:1887 @7fe450e19): a
        /// non-blocking read.
        ///
        /// Return: the bytes read, 0 when nothing is available, -1 when the
        /// pipe is closed or broken.
        pub fn readOutput(self: *Self, buf: []u8) c_int {
            if (!self.hasOutput()) {
                return -1;
            }
            while (true) {
                const r = read(@intCast(self.out_handle), buf.ptr, buf.len);
                if (r > 0) {
                    return @intCast(r);
                }
                if (r == 0) {
                    return -1; // EOF
                }
                const e = errno();
                if (e == @backingInt(std.posix.E.INTR)) {
                    continue;
                }
                if (e == @backingInt(std.posix.E.AGAIN)) {
                    return 0;
                }
                return -1;
            }
        }
    };
}

/// Ports `server_subproc`'s `waiter` (server-common.h:635 @7fe450e19): a
/// self-pipe to wake a `poll` on the children's output.
pub const Waiter = struct {
    wake_fd: [2]c_int = .{ -1, -1 }, // POSIX self-pipe

    /// Ports the constructor of `server_subproc`'s `waiter` (server-common.cpp:1938 @7fe450e19).
    pub fn init() Waiter {
        var fds: [2]c_int = undefined;
        common.ggmlAssert(pipe(&fds) == 0, "server-common.cpp", 1937, "pipe(fds) == 0");
        for (fds) |fd| {
            setNonBlocking(fd);
        }
        return .{ .wake_fd = fds };
    }

    /// Ports the destructor of `server_subproc`'s `waiter` (server-common.cpp:1938 @7fe450e19).
    pub fn deinit(self: *Waiter) void {
        _ = close(self.wake_fd[0]);
        _ = close(self.wake_fd[1]);
    }

    /// Ports `server_subproc::waiter`'s `wake` (server-common.cpp:1945 @7fe450e19): thread-safe.
    pub fn wake(self: *Waiter) void {
        const ch: u8 = 1;
        _ = write(self.wake_fd[1], (&ch)[0..1].ptr, 1);
    }

    /// Ports `server_subproc::waiter`'s `wait` (server-common.cpp:1952 @7fe450e19): until a
    /// child has output (or a broken pipe), `wake()` is called, or
    /// `timeout_ms` passes (forever when negative).
    ///
    /// Parameters:
    /// - `procs`: pointers to `Subproc(P)`s.
    /// - `ready`: set per child, same length as `procs`.
    pub fn wait(self: *Waiter, gpa: std.mem.Allocator, procs: anytype, ready: []bool, timeout_ms: i64) !void {
        @memset(ready, false);
        var pfds: std.ArrayList(std.posix.pollfd) = .empty;
        defer pfds.deinit(gpa);
        try pfds.ensureTotalCapacity(gpa, procs.len + 1);
        pfds.appendAssumeCapacity(.{ .fd = self.wake_fd[0], .events = std.posix.POLL.IN, .revents = 0 });
        for (procs) |p| {
            pfds.appendAssumeCapacity(.{ .fd = if (p.hasOutput()) @intCast(p.out_handle) else -1, .events = std.posix.POLL.IN, .revents = 0 }); // poll() skips negative fds
        }
        const timeout: c_int = if (timeout_ms < 0) -1 else @intCast(@min(timeout_ms, std.math.maxInt(c_int)));
        const r = std.c.poll(pfds.items.ptr, @intCast(pfds.items.len), timeout);
        if (r < 0 and errno() != @backingInt(std.posix.E.INTR)) {
            log.err("%s: poll() failed: %s\n", .{ "wait", strerror(errno()) });
        }
        if (pfds.items[0].revents != 0) {
            var buf: [64]u8 = undefined;
            while (read(self.wake_fd[0], &buf, buf.len) > 0) {}
        }
        for (procs, 0..) |_, i| {
            ready[i] = pfds.items[i + 1].fd < 0 or pfds.items[i + 1].revents != 0;
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

extern fn fdopen(fd: c_int, mode: [*:0]const u8) ?*anyopaque;
extern fn fclose(f: *anyopaque) c_int;

/// A `common_subproc` stand-in over a pipe: its read end is the "child's"
/// output.
const PipeProc = struct {
    file: ?*anyopaque,
    pub fn alive(_: *PipeProc) bool {
        return true;
    }
    pub fn terminate(_: *PipeProc) void {}
    pub fn join(_: *PipeProc) c_int {
        return 0;
    }
    pub fn stdoutFile(self: *PipeProc) ?*anyopaque {
        return self.file;
    }
};

test "read_output is non-blocking, and the waiter wakes on output and on wake()" {
    var fds: [2]c_int = undefined;
    try std.testing.expectEqual(@as(c_int, 0), pipe(&fds));
    defer _ = close(fds[1]);
    var sp: Subproc(PipeProc) = .{ .sproc = .{ .file = fdopen(fds[0], "r") } };
    defer _ = fclose(sp.sproc.file.?);
    var buf: [16]u8 = undefined;
    try std.testing.expectEqual(@as(c_int, 0), sp.readOutput(&buf));

    var w = Waiter.init();
    defer w.deinit();
    var ready = [_]bool{false};
    const procs = [_]*Subproc(PipeProc){&sp};
    w.wake();
    try w.wait(std.testing.allocator, &procs, &ready, 1000);
    try std.testing.expect(!ready[0]);

    _ = write(fds[1], "hi", 2);
    try w.wait(std.testing.allocator, &procs, &ready, 1000);
    try std.testing.expect(ready[0]);
    try std.testing.expectEqual(@as(c_int, 2), sp.readOutput(&buf));
}
