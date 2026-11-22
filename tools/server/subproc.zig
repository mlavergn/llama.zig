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
//! `llama.cpp/tools/server/server-common.h` (llama.cpp v1.0.0,
//! `b1b770d90`), their POSIX arms. Each declaration names what it ports
//! and the line it began at.
//!
//! # Generic over `common_subproc`
//!
//! `server_subproc` wraps a `common_subproc` (`common/subproc.cpp`), for
//! the built-in tools and router mode. It is written against the members
//! it calls -- `alive()`, `terminate()`, `join()`, `stdoutFile()` and
//! `stderrFile()`, the child's two output pipes as `std.Io.File`s or null
//! -- so `Subproc(P)` is the C++ struct over any such `P`. Windows' arms (`PeekNamedPipe`) wait
//! for the Windows server.

const std = @import("std");
const stdlib = @import("stdlib");
const tools = @import("tools_common");

const fd_t = std.posix.fd_t;

/// A pipe end as std reads it.
fn pipeFile(fd: fd_t) std.Io.File {
    return .{ .handle = fd, .flags = .{ .nonblocking = false } };
}

/// Reads what is there without blocking: EINTR is retried.
///
/// Return: the bytes read; null when nothing is available (EAGAIN); 0 at
/// end of file or on an error.
fn readNonBlocking(fd: fd_t, buf: []u8) ?usize {
    return pipeFile(fd).readStreaming(stdlib.file.defaultIo(), &.{buf}) catch |err| switch (err) {
        error.WouldBlock => null,
        else => 0,
    };
}

/// Ports `server_subproc_stream` (server-common.h:695 @b1b770d90): a
/// child server writes its state commands to stdout and its logs to stderr.
pub const Stream = enum(u1) {
    stdout,
    stderr,
};

/// `SERVER_SUBPROC_STREAMS` (server-common.h:698 @b1b770d90).
pub const n_streams = 2;

/// Ports `server_reserve_stdout` (server-common.cpp:2120 @b1b770d90), its
/// POSIX arm: gives the current stdout to the caller as a stream of its
/// own, and sends everything else written to stdout to stderr. The
/// reserved stream is not inherited, so grandchildren get stdout and
/// stderr of their own.
///
/// Return: the reserved stream, the C's `fdopen(fd, "w")` without a
/// `FILE`; that call's `GGML_ASSERT(f)` has nothing left to fail.
pub fn reserveStdout() std.Io.File {
    stdlib.stdio.stdout().flush();
    // `fileno(stdout)` and `fileno(stderr)`
    const stdout_fd: fd_t = 1;
    const stderr_fd: fd_t = 2;
    const fd = std.posix.system.fcntl(stdout_fd, std.posix.F.DUPFD_CLOEXEC, @as(c_int, 0));
    tools.ggml.ggmlAssert(std.posix.errno(fd) == .SUCCESS, "server-common.cpp", 2132, "fd >= 0");
    _ = std.posix.system.dup2(stderr_fd, stdout_fd);
    return pipeFile(@intCast(fd));
}

/// Ports `server_subproc` (server-common.h:706 @b1b770d90) over a
/// `common_subproc`-like `P` (see the file's doc comment).
pub fn Subproc(comptime P: type) type {
    return struct {
        const Self = @This();

        sproc: P,
        stopped: std.atomic.Value(bool) = .init(false), // set by the monitor once the process exited and was reaped

        out_handles: [n_streams]isize = .{ -1, -1 }, // fd on POSIX, HANDLE on Windows; taken lazily from sproc
        out_closed: [n_streams]bool = .{ false, false },

        /// Ports `is_alive` (server-common.h:710 @b1b770d90).
        pub fn isAlive(self: *Self) bool {
            return self.sproc.alive();
        }

        /// Ports `terminate` (server-common.h:711 @b1b770d90).
        pub fn terminate(self: *Self) void {
            self.sproc.terminate();
        }

        /// Ports `join` (server-common.h:712 @b1b770d90).
        pub fn join(self: *Self) c_int {
            return self.sproc.join();
        }

        /// Ports `server_subproc`'s `has_output` (server-common.cpp:2140 @b1b770d90): true when
        /// both output pipes of the child are available; the first call
        /// makes each non-blocking.
        pub fn hasOutput(self: *Self) bool {
            const files = [n_streams]?std.Io.File{ self.sproc.stdoutFile(), self.sproc.stderrFile() };
            for (0..n_streams) |i| {
                if (self.out_handles[i] >= 0) {
                    continue;
                }
                const f = files[i] orelse return false;
                stdlib.subproc.setNonBlocking(f.handle);
                self.out_handles[i] = f.handle;
            }
            return true;
        }

        /// Ports `server_subproc`'s `output_closed` (server-common.cpp:2167 @b1b770d90):
        /// true once both output pipes are closed.
        pub fn outputClosed(self: *const Self) bool {
            return self.out_closed[@backingInt(Stream.stdout)] and self.out_closed[@backingInt(Stream.stderr)];
        }

        /// Ports `server_subproc`'s `read_output` (server-common.cpp:2171 @b1b770d90): a
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
            const r = readNonBlocking(@intCast(self.out_handles[s]), buf) orelse return 0;
            if (r > 0) {
                return @intCast(r);
            }
            self.out_closed[s] = true; // EOF or error
            return -1;
        }
    };
}

/// Ports `server_subproc`'s `waiter` (server-common.h:725 @b1b770d90): a
/// self-pipe to wake a `poll` on the children's output.
pub const Waiter = struct {
    wake_fd: [2]fd_t = .{ -1, -1 }, // POSIX self-pipe

    /// Ports the constructor of `server_subproc`'s `waiter` (server-common.cpp:2214 @b1b770d90).
    pub fn init() Waiter {
        // pipe(), not close-on-exec: the C's children inherit it too
        const fds = std.Io.Threaded.pipe2(.{}) catch [2]fd_t{ -1, -1 };
        tools.ggml.ggmlAssert(fds[0] >= 0, "server-common.cpp", 2217, "pipe(fds) == 0");
        for (fds) |fd| {
            stdlib.subproc.setNonBlocking(fd);
        }
        return .{ .wake_fd = fds };
    }

    /// Ports the destructor of `server_subproc`'s `waiter` (server-common.cpp:2226 @b1b770d90).
    pub fn deinit(self: *Waiter) void {
        _ = std.posix.system.close(self.wake_fd[0]);
        _ = std.posix.system.close(self.wake_fd[1]);
    }

    /// Ports `server_subproc::waiter`'s `wake` (server-common.cpp:2233 @b1b770d90): thread-safe.
    pub fn wake(self: *Waiter) void {
        const ch = [1]u8{1};
        _ = std.posix.system.write(self.wake_fd[1], &ch, 1);
    }

    /// Ports `server_subproc::waiter`'s `wait` (server-common.cpp:2240 @b1b770d90), its POSIX
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
        // the C library's poll on Darwin, the system call on Linux
        const e = std.posix.errno(std.posix.system.poll(pfds.items.ptr, @intCast(pfds.items.len), @intCast(timeout)));
        if (e != .SUCCESS and e != .INTR) {
            tools.log.err("%s: poll() failed: %s\n", .{ "wait", stdlib.errno.strerror(@backingInt(e)) });
        }
        if (pfds.items[0].revents != 0) {
            var buf: [64]u8 = undefined;
            while ((readNonBlocking(self.wake_fd[0], &buf) orelse 0) > 0) {}
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

/// A `common_subproc` stand-in over a pipe: its read end is the "child's"
/// output.
const PipeProc = struct {
    file: ?std.Io.File,
    err: ?std.Io.File,
    pub fn alive(_: *PipeProc) bool {
        return true;
    }
    pub fn terminate(_: *PipeProc) void {}
    pub fn join(_: *PipeProc) c_int {
        return 0;
    }
    pub fn stdoutFile(self: *PipeProc) ?std.Io.File {
        return self.file;
    }
    pub fn stderrFile(self: *PipeProc) ?std.Io.File {
        return self.err;
    }
};

test "read_output is non-blocking per pipe, and the waiter wakes on output and on wake()" {
    const io = std.testing.io;
    const fds = try std.Io.Threaded.pipe2(.{});
    const efds = try std.Io.Threaded.pipe2(.{});
    defer pipeFile(fds[1]).close(io);
    var sp: Subproc(PipeProc) = .{ .sproc = .{ .file = pipeFile(fds[0]), .err = pipeFile(efds[0]) } };
    defer sp.sproc.file.?.close(io);
    defer sp.sproc.err.?.close(io);
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

    try pipeFile(efds[1]).writeStreamingAll(io, "log");
    try w.wait(std.testing.allocator, &procs, &ready, 1000);
    try std.testing.expect(ready[0]);
    try std.testing.expectEqual(@as(c_int, 0), sp.readOutput(.stdout, &buf));
    try std.testing.expectEqual(@as(c_int, 3), sp.readOutput(.stderr, &buf));

    try pipeFile(fds[1]).writeStreamingAll(io, "hi");
    try w.wait(std.testing.allocator, &procs, &ready, 1000);
    try std.testing.expect(ready[0]);
    try std.testing.expectEqual(@as(c_int, 2), sp.readOutput(.stdout, &buf));

    // one pipe closed is not the output closed
    pipeFile(efds[1]).close(io);
    try std.testing.expectEqual(@as(c_int, -1), sp.readOutput(.stderr, &buf));
    try std.testing.expect(!sp.outputClosed());
    try std.testing.expectEqual(@as(c_int, -1), sp.readOutput(.stderr, &buf));
}
