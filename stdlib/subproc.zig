//! A child process with its standard streams on pipes: what the server's
//! built-in tools, its MCP clients and router mode spawn.
//!
//! # Provenance
//!
//! Ports `common_subproc` from `llama.cpp/common/subproc.cpp` and
//! `llama.cpp/common/subproc.h`, and the POSIX arm of the parts of
//! `llama.cpp/vendor/sheredom/subprocess.h` it calls --
//! `subprocess_create_ex`, `subprocess_join`, `subprocess_destroy`,
//! `subprocess_terminate`, `subprocess_alive` and the pipe helpers -- at
//! llama.cpp v1.0.0 (`b1b770d90`). Each declaration names what it ports
//! and the line it began at.
//!
//! # Which arm
//!
//! `LLAMA_SUBPROCESS` is defined wherever the reference builds (every
//! platform but iOS, `llama.cpp.zmake`'s `subprocess_defines`), so
//! `is_supported()` is true there and false on iOS, where `create` fails.
//! Windows' arm (`CreateProcessW`) is not ported: the server's Windows
//! build waits on it.
//!
//! # Over std
//!
//! The child is spawned by `std.process.spawn`, not `posix_spawn`: std
//! forks and execs, with the pipes created here and handed to it, so a
//! spawn that fails closes them as `subprocess_create_ex` does. One
//! difference remains: when the program cannot be executed, std returns
//! the error without the child's pid, so that child is never reaped and
//! stays a zombie until the server exits (glibc's and Darwin's
//! `posix_spawn` leave none). The pid is then the C's: `kill(SIGKILL)` and
//! `waitpid`, through `std.posix`, so `terminate` can run on another thread
//! while one reads the child's output, as upstream's timeout does.
//!
//! A program name without a `/` is searched in PATH under
//! `option_search_user_path` (`posix_spawnp`, std's `.detect`) and taken as
//! a path otherwise (`posix_spawn`). PATH and an inherited environment
//! come from `env`.

const std = @import("std");
const mod = @import("module.zig");
const builtin = @import("builtin");

const fd_t = std.posix.fd_t;
const pid_t = std.posix.pid_t;
const EXIT_FAILURE: c_int = 1;

/// The `Io` that spawns children: `init_single_threaded` (`Threaded.init`
/// would install SIGPIPE and SIGIO handlers) with the process environment,
/// where std reads PATH and what a child inherits, and an allocator, where
/// std builds the child's `argv` and environment before it forks.
var spawner_state: std.Io.Threaded = .init_single_threaded;
var spawner_ready: std.atomic.Value(bool) = .init(false);
var spawner_mutex: mod.sync.Mutex = .{};

fn spawner() std.Io {
    if (!spawner_ready.load(.acquire)) {
        spawner_mutex.lock();
        defer spawner_mutex.unlock();
        if (!spawner_ready.load(.acquire)) {
            const environ = mod.env.processEnviron();
            spawner_state.environ = .{ .process_environ = environ };
            spawner_state.environ_initialized = environ.block.isEmpty();
            spawner_state.allocator = std.heap.page_allocator;
            spawner_ready.store(true, .release);
        }
    }
    return spawner_state.io();
}

fn closeFd(fd: fd_t) void {
    _ = std.posix.system.close(fd);
}

fn pipeFile(fd: fd_t) std.Io.File {
    return .{ .handle = fd, .flags = .{ .nonblocking = false } };
}

/// `fcntl(fd, F_SETFL, flags | O_NONBLOCK)`, for the server's polled
/// pipes; an error leaves the descriptor as it was, as the C's unchecked
/// call does.
///
/// Parameters:
/// - `fd`: a pipe end, `std.Io.File.handle`.
pub fn setNonBlocking(fd: fd_t) void {
    const fl = std.posix.system.fcntl(fd, std.posix.F.GETFL, @as(c_int, 0));
    if (std.posix.errno(fl) != .SUCCESS) return;
    const nonblock: u32 = @bitCast(std.posix.O{ .NONBLOCK = true });
    _ = std.posix.system.fcntl(fd, std.posix.F.SETFL, @as(c_int, @intCast(fl)) | @as(c_int, @bitCast(nonblock)));
}

/// Ports `subprocess_option_e` (subprocess.h:75 @b1b770d90).
pub const option_combined_stdout_stderr: c_int = 0x1;
pub const option_inherit_environment: c_int = 0x2;
pub const option_enable_async: c_int = 0x4;
pub const option_no_window: c_int = 0x8;
pub const option_search_user_path: c_int = 0x10;
pub const option_enable_async_no_wait: c_int = 0x20;

/// Ports `subprocess_error_e` (subprocess.h:101 @b1b770d90).
const err_invalid_options: c_int = -2;
const err_invalid_environment: c_int = -3;
const err_not_found: c_int = -4;
const err_permission_denied: c_int = -5;
const err_no_memory: c_int = -6;
const err_pipe: c_int = -7;
const err_spawn: c_int = -8;

/// Ports `subprocess_s` (subprocess.h:563 @b1b770d90), its POSIX members.
const Process = struct {
    stdin_file: ?std.Io.File = null,
    stdout_file: ?std.Io.File = null,
    stderr_file: ?std.Io.File = null, // stdout_file's when combined
    child: pid_t = 0,
    return_status: c_int = 0,
    alive: c_int = 0,
    no_wait: c_int = 0,
};

/// Ports `subprocess_error_from_errno` (subprocess.h:622 @b1b770d90), over
/// std's errors for the same `errno`s; `subprocess_error_spawn` for one
/// with no class of its own, as each spawn step of `subprocess_create_ex`
/// does.
fn spawnError(err: anyerror) c_int {
    return switch (err) {
        error.FileNotFound, error.NotDir => err_not_found,
        error.AccessDenied, error.PermissionDenied => err_permission_denied,
        error.ProcessFdQuotaExceeded, error.SystemFdQuotaExceeded, error.SystemResources, error.OutOfMemory => err_no_memory,
        else => err_spawn,
    };
}

/// Ports `subprocess_fds_above_std` (subprocess.h:732 @b1b770d90): moves a
/// pipe end off 0, 1 or 2, keeping it close-on-exec, so the child's
/// `dup2` onto its standard streams never meets it there.
fn fdsAboveStd(fds: *[2]fd_t) error{Pipe}!void {
    for (0..2) |index| {
        if (fds[index] > 2) continue;
        const moved = std.posix.system.fcntl(fds[index], std.posix.F.DUPFD_CLOEXEC, @as(c_int, 3));
        if (std.posix.errno(moved) != .SUCCESS) {
            closeFd(fds[0]);
            closeFd(fds[1]);
            fds.* = .{ -1, -1 };
            return error.Pipe;
        }
        closeFd(fds[index]);
        fds[index] = @intCast(moved);
    }
}

/// Ports `subprocess_pipe_cloexec` (subprocess.h:774 @b1b770d90): a pipe
/// whose ends later children do not inherit.
fn pipeCloexec() error{Pipe}![2]fd_t {
    var fds = std.Io.Threaded.pipe2(.{ .CLOEXEC = true }) catch return error.Pipe;
    try fdsAboveStd(&fds);
    return fds;
}

/// Ports `subprocess_create_ex` (subprocess.h:826 @b1b770d90), its POSIX
/// arm, over `std.process.spawn`.
///
/// Parameters:
/// - `argv`: the command line; `argv[0]` is the program.
/// - `environment`: replaces the child's, or null.
/// - `process_cwd`: the child's working directory, or null for ours.
///
/// Return: 0, or a `subprocess_error_e`.
fn createEx(argv: []const []const u8, options: c_int, environment: ?*const std.process.Environ.Map, process_cwd: ?[]const u8, out: *Process) c_int {
    var stdinfd: [2]fd_t = .{ -1, -1 };
    var stdoutfd: [2]fd_t = .{ -1, -1 };
    var stderrfd: [2]fd_t = .{ -1, -1 };
    var result: c_int = 0;

    const async_no_wait = option_enable_async_no_wait == (options & option_enable_async_no_wait);
    if (async_no_wait and option_enable_async != (options & option_enable_async)) return err_invalid_options;
    const inherit = option_inherit_environment == (options & option_inherit_environment);
    if (inherit and environment != null) return err_invalid_environment;
    const combined = option_combined_stdout_stderr == (options & option_combined_stdout_stderr);

    out.* = .{};

    body: {
        stdinfd = pipeCloexec() catch {
            result = err_pipe;
            break :body;
        };
        stdoutfd = pipeCloexec() catch {
            result = err_pipe;
            break :body;
        };
        if (!combined) {
            stderrfd = pipeCloexec() catch {
                result = err_pipe;
                break :body;
            };
        }

        // no environment and no inheritance: an empty one (null inherits the spawner's)
        const empty: std.process.Environ.Map = .init(std.mem.Allocator.failing);
        const used_environment: ?*const std.process.Environ.Map = if (environment) |e| e else if (inherit) null else &empty;

        const child = std.process.spawn(spawner(), .{
            .exe = if (option_search_user_path == (options & option_search_user_path)) .detect else .{ .path = .cwd() },
            .argv = argv,
            .cwd = if (process_cwd) |d| .{ .path = d } else .inherit,
            .environ_map = used_environment,
            // the child's ends; std `dup2`s them onto its 0, 1 and 2, and the
            // close-on-exec originals (and our ends) close at its exec
            .stdin = .{ .file = pipeFile(stdinfd[0]) },
            .stdout = .{ .file = pipeFile(stdoutfd[1]) },
            .stderr = .{ .file = pipeFile(if (combined) stdoutfd[1] else stderrfd[1]) },
            .create_no_window = option_no_window == (options & option_no_window),
        }) catch |err| {
            result = spawnError(err);
            break :body;
        };

        // Close the child's ends, store ours
        closeFd(stdinfd[0]);
        stdinfd[0] = -1;
        out.stdin_file = pipeFile(stdinfd[1]);
        stdinfd[1] = -1;
        closeFd(stdoutfd[1]);
        stdoutfd[1] = -1;
        out.stdout_file = pipeFile(stdoutfd[0]);
        stdoutfd[0] = -1;
        // Set non blocking if we are async and asked not to wait.
        if (async_no_wait) setNonBlocking(out.stdout_file.?.handle);
        if (combined) {
            out.stderr_file = out.stdout_file;
        } else {
            closeFd(stderrfd[1]);
            stderrfd[1] = -1;
            out.stderr_file = pipeFile(stderrfd[0]);
            stderrfd[0] = -1;
            if (async_no_wait) setNonBlocking(out.stderr_file.?.handle);
        }

        // Store the child's pid
        out.child = child.id.?;
        out.alive = 1;
        out.no_wait = @intFromBool(async_no_wait);
    }

    // cleanup:
    for ([_]fd_t{ stdinfd[0], stdinfd[1], stdoutfd[0], stdoutfd[1], stderrfd[0], stderrfd[1] }) |fd| {
        if (fd != -1) closeFd(fd);
    }
    return result;
}

/// `WIFEXITED` and `WEXITSTATUS`: `EXIT_FAILURE` when a signal ended it.
fn exitStatus(status: c_int) c_int {
    const s: u32 = @bitCast(status);
    if (std.posix.W.IFEXITED(s)) return std.posix.W.EXITSTATUS(s);
    return EXIT_FAILURE;
}

/// `waitpid(pid, &status, flags)`, the C library's on Darwin and the
/// system call on Linux, as `std.posix.system` resolves it.
///
/// Return: the pid reaped, 0 when `WNOHANG` found it running, -1 on an
/// error (EINTR included, as the C's single call).
fn waitPid(pid: pid_t, status: *c_int, flags: u32) pid_t {
    const rc = std.posix.system.waitpid(pid, status, @intCast(flags));
    if (std.posix.errno(rc) != .SUCCESS) return -1;
    return @intCast(rc);
}

/// Ports `subprocess_join` (subprocess.h:1903 @b1b770d90), its POSIX arm.
fn processJoin(p: *Process, out_return_code: ?*c_int) c_int {
    if (p.stdin_file) |f| {
        f.close(spawner());
        p.stdin_file = null;
    }
    if (p.child != 0) {
        var status: c_int = 0;
        if (p.child != waitPid(p.child, &status, 0)) return -1;
        p.child = 0;
        p.return_status = exitStatus(status);
        p.alive = 0;
    }
    if (out_return_code) |r| r.* = p.return_status;
    return 0;
}

/// Ports `subprocess_destroy` (subprocess.h:1963 @b1b770d90).
fn processDestroy(p: *Process) void {
    const io = spawner();
    if (p.stdin_file) |f| {
        f.close(io);
        p.stdin_file = null;
    }
    if (p.stdout_file) |f| {
        f.close(io);
        if (p.stderr_file) |e| {
            if (e.handle != f.handle) e.close(io);
        }
        p.stdout_file = null;
        p.stderr_file = null;
    }
}

/// Ports `subprocess_alive` (subprocess.h:2140 @b1b770d90): reaps the child
/// when it has exited.
fn processAlive(p: *Process) c_int {
    var is_alive = p.alive;
    if (is_alive == 0) return 0;
    var status: c_int = 0;
    is_alive = @intFromBool(waitPid(p.child, &status, std.posix.W.NOHANG) == 0);
    // If the process was successfully waited on we need to cleanup now.
    if (is_alive == 0) {
        p.return_status = exitStatus(status);
        // Since we've already successfully waited on the process, we need to wipe
        // the child now.
        p.child = 0;
        if (processJoin(p, null) != 0) return -1;
    }
    if (is_alive == 0) p.alive = 0;
    return is_alive;
}

/// Ports `common_subproc` (subproc.h:22 @b1b770d90): a RAII wrapper around
/// `subprocess_s`. Not copyable: hold it in place.
pub const Subproc = struct {
    proc: Process = .{},
    is_created: std.atomic.Value(bool) = .init(false),

    /// Ports `common_subproc`'s `is_supported` (subproc.cpp:3 @b1b770d90): true
    /// where the reference defines `LLAMA_SUBPROCESS`.
    pub fn isSupported() bool {
        return comptime builtin.os.tag != .ios and builtin.os.tag != .windows;
    }

    /// Ports `common_subproc`'s destructor (subproc.cpp:23 @b1b770d90).
    pub fn deinit(self: *Subproc) void {
        if (self.is_created.load(.seq_cst)) {
            processDestroy(&self.proc);
            self.is_created.store(false, .seq_cst);
        }
    }

    /// Ports `common_subproc`'s `create` (subproc.cpp:30 @b1b770d90): spawns a
    /// child; a non-empty `env` replaces its environment (do not combine
    /// with `option_inherit_environment`).
    ///
    /// Parameters:
    /// - `args`: the command line; `args[0]` is the program.
    /// - `options`: `option_*` bits.
    /// - `env`: `KEY=VALUE` strings, or empty.
    /// - `cwd`: the child's working directory, or null for ours.
    ///
    /// Return: whether the child was spawned.
    pub fn create(self: *Subproc, args: []const []const u8, options: c_int, env_vars: []const []const u8, cwd: ?[]const u8) bool {
        if (comptime !isSupported()) return false;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        var environment: std.process.Environ.Map = .init(arena.allocator());
        if (env_vars.len != 0) toEnvironMap(&environment, env_vars) catch return false;
        const result = createEx(args, options, if (env_vars.len == 0) null else &environment, cwd, &self.proc);
        self.is_created.store(result == 0, .seq_cst);
        return result == 0;
    }

    /// Ports `to_cstr_vec` (subproc.cpp:13 @b1b770d90) for the environment:
    /// the `KEY=VALUE` strings as std's map. A name given twice keeps its
    /// first value, the one the child's `getenv` reads from the C's vector;
    /// an entry without `=`, which `getenv` never matches, is left out.
    fn toEnvironMap(m: *std.process.Environ.Map, v: []const []const u8) std.mem.Allocator.Error!void {
        for (v) |entry| {
            const eq = std.mem.findScalar(u8, entry, '=') orelse continue;
            if (eq == 0 or m.contains(entry[0..eq])) continue;
            try m.put(entry[0..eq], entry[eq + 1 ..]);
        }
    }

    /// Ports `common_subproc`'s `has_handle` (subproc.cpp:49 @b1b770d90).
    fn hasHandle(self: *const Subproc) bool {
        if (!self.is_created.load(.seq_cst)) return false;
        return self.proc.child > 0;
    }

    /// Ports `common_subproc`'s `alive` (subproc.cpp:60 @b1b770d90).
    pub fn alive(self: *Subproc) bool {
        if (comptime !isSupported()) return false;
        return self.is_created.load(.seq_cst) and processAlive(&self.proc) != 0;
    }

    /// Ports `common_subproc`'s `stdin_file` (subproc.cpp:64 @b1b770d90).
    pub fn stdinFile(self: *Subproc) ?std.Io.File {
        return if (self.is_created.load(.seq_cst)) self.proc.stdin_file else null;
    }

    /// Ports `common_subproc`'s `stdout_file` (subproc.cpp:68 @b1b770d90).
    pub fn stdoutFile(self: *Subproc) ?std.Io.File {
        return if (self.is_created.load(.seq_cst)) self.proc.stdout_file else null;
    }

    /// Ports `common_subproc`'s `stderr_file` (subproc.cpp:72 @b1b770d90), over
    /// `subprocess_stderr` (subprocess.h:1895 @b1b770d90): null when stderr
    /// is combined with stdout.
    pub fn stderrFile(self: *Subproc) ?std.Io.File {
        if (!self.is_created.load(.seq_cst)) return null;
        const err = self.proc.stderr_file orelse return null;
        if (self.proc.stdout_file) |out| {
            if (out.handle == err.handle) return null;
        }
        return err;
    }

    /// Ports `common_subproc`'s `close_stdin` (subproc.cpp:76 @b1b770d90):
    /// EOF to the child while it runs.
    pub fn closeStdin(self: *Subproc) void {
        if (self.is_created.load(.seq_cst)) {
            if (self.proc.stdin_file) |f| {
                f.close(spawner());
                self.proc.stdin_file = null;
            }
        }
    }

    /// Ports `common_subproc`'s `terminate` (subproc.cpp:83 @b1b770d90), over
    /// `subprocess_terminate` (subprocess.h:2002 @b1b770d90): `SIGKILL`.
    pub fn terminate(self: *Subproc) void {
        if (comptime !isSupported()) return;
        if (self.hasHandle()) _ = std.posix.system.kill(self.proc.child, .KILL);
    }

    /// Ports `common_subproc`'s `join` (subproc.cpp:89 @b1b770d90): waits for
    /// the child, releases it.
    ///
    /// Return: its exit code -- `EXIT_FAILURE` when a signal ended it, -1
    /// when nothing was created.
    pub fn join(self: *Subproc) c_int {
        var exit_code: c_int = -1;
        if (comptime !isSupported()) return exit_code;
        if (self.is_created.load(.seq_cst)) {
            _ = processJoin(&self.proc, &exit_code);
            processDestroy(&self.proc);
            self.is_created.store(false, .seq_cst);
        }
        return exit_code;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a child's combined output, its exit code, and a working directory" {
    var p: Subproc = .{};
    defer p.deinit();
    try std.testing.expect(p.create(&.{ "sh", "-c", "pwd; echo err >&2; exit 3" }, option_combined_stdout_stderr | option_inherit_environment | option_search_user_path, &.{}, "/"));
    p.closeStdin();
    try std.testing.expect(p.stderrFile() == null);
    var buf: [64]u8 = undefined;
    var n: usize = 0;
    while (true) n += p.stdoutFile().?.readStreaming(std.testing.io, &.{buf[n..]}) catch break;
    try std.testing.expectEqualStrings("/\nerr\n", buf[0..n]);
    try std.testing.expectEqual(@as(c_int, 3), p.join());
}

test "a replaced environment, separate stderr, and stdin through to the child" {
    var p: Subproc = .{};
    defer p.deinit();
    // without option_search_user_path the program is a path, never a name
    try std.testing.expect(!p.create(&.{ "sh", "-c", "true" }, 0, &.{"A=1"}, null));
    try std.testing.expect(p.create(&.{ "/bin/sh", "-c", "read x; echo \"$x $A $B\"; echo e >&2" }, 0, &.{ "A=first", "A=second", "B=2", "junk" }, null));
    try p.stdinFile().?.writeStreamingAll(std.testing.io, "in\n");
    p.closeStdin();
    var buf: [64]u8 = undefined;
    var n: usize = 0;
    while (true) n += p.stdoutFile().?.readStreaming(std.testing.io, &.{buf[n..]}) catch break;
    try std.testing.expectEqualStrings("in first 2\n", buf[0..n]);
    n = 0;
    while (true) n += p.stderrFile().?.readStreaming(std.testing.io, &.{buf[n..]}) catch break;
    try std.testing.expectEqualStrings("e\n", buf[0..n]);
    try std.testing.expectEqual(@as(c_int, 0), p.join());
}

test "a program that is not there fails to spawn" {
    var p: Subproc = .{};
    defer p.deinit();
    try std.testing.expect(!p.create(&.{"/nonexistent/program"}, option_inherit_environment, &.{}, null));
    try std.testing.expectEqual(@as(c_int, -1), p.join());
}

test "a killed child exits with EXIT_FAILURE and is no longer alive" {
    var p: Subproc = .{};
    defer p.deinit();
    try std.testing.expect(p.create(&.{ "sleep", "10" }, option_search_user_path, &.{"A=1"}, null));
    try std.testing.expect(p.alive());
    p.terminate();
    try std.testing.expectEqual(@as(c_int, 1), p.join());
    try std.testing.expect(!p.alive());
}
