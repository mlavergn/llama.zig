//! A child server process as router mode watches it: its two output pipes
//! -- state commands on stdout, logs on stderr -- read without blocking, a
//! waiter that sleeps until one of several children has output or is
//! woken, and the child's side of the split, `server_reserve_stdout`.
//!
//! # Provenance
//!
//! Ports `server_subproc`, `server_subproc::waiter`,
//! `server_subproc_stream` and `server_reserve_stdout` from
//! `llama.cpp/tools/server/server-common.cpp` and
//! `llama.cpp/tools/server/server-common.h` (llama.cpp v0.6.0,
//! `d81235049`), their POSIX arms. Each declaration names what it ports
//! and the line it began at.
//!
//! # Generic over `common_subproc`
//!
//! `server_subproc` wraps a `common_subproc` (`common/subproc.cpp`, Stage 10
//! step 6, with router mode in step 7). It is written against the members
//! it calls -- `alive()`, `terminate()`, `join()`, `stdoutFile()` and
//! `stderrFile()`, the child's two output pipes as C `FILE *`s or null -- so `Subproc(P)` is
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

/// Ports `server_subproc_stream` (server-common.h:657 @d81235049): a
/// child server writes its state commands to stdout and its logs to stderr.
pub const Stream = enum(u1) {
    stdout,
    stderr,
};

/// `SERVER_SUBPROC_STREAMS` (server-common.h:660 @d81235049).
pub const n_streams = 2;

extern fn dup2(old: c_int, new: c_int) c_int;
extern fn fflush(f: ?*anyopaque) c_int;
extern fn fdopen(fd: c_int, mode: [*:0]const u8) ?*anyopaque;
const F_DUPFD_CLOEXEC: c_int = if (@import("builtin").os.tag.isDarwin()) 67 else 1030;

/// Ports `server_reserve_stdout` (server-common.cpp:1929 @d81235049), its
/// POSIX arm: gives the current stdout to the caller as a stream of its
/// own, and sends everything else written to stdout to stderr. The
/// reserved stream is not inherited, so grandchildren get stdout and
/// stderr of their own.
pub fn reserveStdout() *anyopaque {
    _ = fflush(tools.io.cStdout());
    const fd = fcntl(fileno(tools.io.cStdout()), F_DUPFD_CLOEXEC, @as(c_int, 0));
    common.ggmlAssert(fd >= 0, "server-common.cpp", 1941, "fd >= 0");
    _ = dup2(fileno(tools.io.cStderr()), fileno(tools.io.cStdout()));
    const f = fdopen(fd, "w");
    common.ggmlAssert(f != null, "server-common.cpp", 1945, "f");
    return f.?;
}

/// Ports `server_subproc` (server-common.h:668 @d81235049) over a
/// `common_subproc`-like `P` (see the file's doc comment).
pub fn Subproc(comptime P: type) type {
    return struct {
        const Self = @This();

        sproc: P,
        stopped: std.atomic.Value(bool) = .init(false), // set by the monitor once the process exited and was reaped

        out_handles: [n_streams]isize = .{ -1, -1 }, // fd on POSIX, HANDLE on Windows; taken lazily from sproc
        out_closed: [n_streams]bool = .{ false, false },

        /// Ports `is_alive` (server-common.h:672 @d81235049).
        pub fn isAlive(self: *Self) bool {
            return self.sproc.alive();
        }

        /// Ports `terminate` (server-common.h:673 @d81235049).
        pub fn terminate(self: *Self) void {
            self.sproc.terminate();
        }

        /// Ports `join` (server-common.h:674 @d81235049).
        pub fn join(self: *Self) c_int {
            return self.sproc.join();
        }

        /// Ports `server_subproc`'s `has_output` (server-common.cpp:1949 @d81235049): true when
        /// both output pipes of the child are available; the first call
        /// makes each non-blocking.
        pub fn hasOutput(self: *Self) bool {
            const files = [n_streams]?*anyopaque{ @ptrCast(self.sproc.stdoutFile()), @ptrCast(self.sproc.stderrFile()) };
            for (0..n_streams) |i| {
                if (self.out_handles[i] >= 0) {
                    continue;
                }
                const f = files[i] orelse return false;
                const fd = fileno(f);
                if (fd < 0) {
                    return false;
                }
                setNonBlocking(fd);
                self.out_handles[i] = fd;
            }
            return true;
        }

        /// Ports `server_subproc`'s `output_closed` (server-common.cpp:1976 @d81235049):
        /// true once both output pipes are closed.
        pub fn outputClosed(self: *const Self) bool {
            return self.out_closed[@backingInt(Stream.stdout)] and self.out_closed[@backingInt(Stream.stderr)];
        }

        /// Ports `server_subproc`'s `read_output` (server-common.cpp:1980 @d81235049): a
        /// non-blocking read from one output pipe.
        ///
        /// Return: the bytes read, 0 when nothing is available, -1 when the
        /// pipe is closed or broken.
        pub fn readOutput(self: *Self, stream: Stream, buf: []u8) c_int {
            const s = @backingInt(stream);
            if (self.out_closed[s]) {
                return -1;
            }
            if (!self.hasOutput()) {
                self.out_closed[s] = true;
                return -1;
            }
            while (true) {
                const r = read(@intCast(self.out_handles[s]), buf.ptr, buf.len);
                if (r > 0) {
                    return @intCast(r);
                }
                const e = errno();
                if (r < 0 and e == @backingInt(std.posix.E.INTR)) {
                    continue;
                }
                if (r < 0 and e == @backingInt(std.posix.E.AGAIN)) {
                    return 0;
                }
                self.out_closed[s] = true; // EOF or error
                return -1;
            }
        }
    };
}

/// Ports `server_subproc`'s `waiter` (server-common.h:687 @d81235049): a
/// self-pipe to wake a `poll` on the children's output.
pub const Waiter = struct {
    wake_fd: [2]c_int = .{ -1, -1 }, // POSIX self-pipe

    /// Ports the constructor of `server_subproc`'s `waiter` (server-common.cpp:2023 @d81235049).
    pub fn init() Waiter {
        var fds: [2]c_int = undefined;
        common.ggmlAssert(pipe(&fds) == 0, "server-common.cpp", 2026, "pipe(fds) == 0");
        for (fds) |fd| {
            setNonBlocking(fd);
        }
        return .{ .wake_fd = fds };
    }

    /// Ports the destructor of `server_subproc`'s `waiter` (server-common.cpp:2035 @d81235049).
    pub fn deinit(self: *Waiter) void {
        _ = close(self.wake_fd[0]);
        _ = close(self.wake_fd[1]);
    }

    /// Ports `server_subproc::waiter`'s `wake` (server-common.cpp:2042 @d81235049): thread-safe.
    pub fn wake(self: *Waiter) void {
        const ch: u8 = 1;
        _ = write(self.wake_fd[1], (&ch)[0..1].ptr, 1);
    }

    /// Ports `server_subproc::waiter`'s `wait` (server-common.cpp:2049 @d81235049), its POSIX
    /// arm: until a child has output on an open pipe (or a broken pipe),
    /// `wake()` is called, or `timeout_ms` passes (forever when negative).
    ///
    /// Parameters:
    /// - `procs`: pointers to `Subproc(P)`s.
    /// - `ready`: set per child, same length as `procs`.
    pub fn wait(self: *Waiter, gpa: std.mem.Allocator, procs: anytype, ready: []bool, timeout_ms: i64) !void {
        @memset(ready, false);
        var pfds: std.ArrayList(std.posix.pollfd) = .empty;
        defer pfds.deinit(gpa);
        try pfds.ensureTotalCapacity(gpa, procs.len * n_streams + 1);
        pfds.appendAssumeCapacity(.{ .fd = self.wake_fd[0], .events = std.posix.POLL.IN, .revents = 0 });
        for (procs) |p| {
            const open = p.hasOutput();
            for (0..n_streams) |s| {
                pfds.appendAssumeCapacity(.{ .fd = if (open and !p.out_closed[s]) @intCast(p.out_handles[s]) else -1, .events = std.posix.POLL.IN, .revents = 0 }); // poll() skips negative fds
            }
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
        for (procs, 0..) |p, i| {
            ready[i] = !p.hasOutput();
            for (0..n_streams) |s| {
                ready[i] = ready[i] or pfds.items[1 + i * n_streams + s].revents != 0;
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

extern fn fclose(f: *anyopaque) c_int;

/// A `common_subproc` stand-in over a pipe: its read end is the "child's"
/// output.
const PipeProc = struct {
    file: ?*anyopaque,
    err: ?*anyopaque,
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
    pub fn stderrFile(self: *PipeProc) ?*anyopaque {
        return self.err;
    }
};

test "read_output is non-blocking per pipe, and the waiter wakes on output and on wake()" {
    var fds: [2]c_int = undefined;
    var efds: [2]c_int = undefined;
    try std.testing.expectEqual(@as(c_int, 0), pipe(&fds));
    try std.testing.expectEqual(@as(c_int, 0), pipe(&efds));
    defer _ = close(fds[1]);
    var sp: Subproc(PipeProc) = .{ .sproc = .{ .file = fdopen(fds[0], "r"), .err = fdopen(efds[0], "r") } };
    defer _ = fclose(sp.sproc.file.?);
    defer _ = fclose(sp.sproc.err.?);
    var buf: [16]u8 = undefined;
    try std.testing.expectEqual(@as(c_int, 0), sp.readOutput(.stdout, &buf));
    try std.testing.expectEqual(@as(c_int, 0), sp.readOutput(.stderr, &buf));

    var w = Waiter.init();
    defer w.deinit();
    var ready = [_]bool{false};
    const procs = [_]*Subproc(PipeProc){&sp};
    w.wake();
    try w.wait(std.testing.allocator, &procs, &ready, 1000);
    try std.testing.expect(!ready[0]);

    _ = write(efds[1], "log", 3);
    try w.wait(std.testing.allocator, &procs, &ready, 1000);
    try std.testing.expect(ready[0]);
    try std.testing.expectEqual(@as(c_int, 0), sp.readOutput(.stdout, &buf));
    try std.testing.expectEqual(@as(c_int, 3), sp.readOutput(.stderr, &buf));

    _ = write(fds[1], "hi", 2);
    try w.wait(std.testing.allocator, &procs, &ready, 1000);
    try std.testing.expect(ready[0]);
    try std.testing.expectEqual(@as(c_int, 2), sp.readOutput(.stdout, &buf));

    // one pipe closed is not the output closed
    _ = close(efds[1]);
    try std.testing.expectEqual(@as(c_int, -1), sp.readOutput(.stderr, &buf));
    try std.testing.expect(!sp.outputClosed());
    try std.testing.expectEqual(@as(c_int, -1), sp.readOutput(.stderr, &buf));
}
