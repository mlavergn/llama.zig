//! A child process with its standard streams on pipes: what the server's
//! built-in tools, its MCP clients and router mode spawn.
//!
//! # Provenance
//!
//! Ports `common_subproc` from `llama.cpp/common/subproc.cpp` and
//! `llama.cpp/common/subproc.h`, and the POSIX arm of the parts of
//! `llama.cpp/vendor/sheredom/subprocess.h` it calls --
//! `subprocess_create_ex` over `posix_spawn`, `subprocess_join`,
//! `subprocess_destroy`, `subprocess_terminate`, `subprocess_alive` and the
//! pipe helpers -- at llama.cpp v0.5.0 (`7fe450e19`). Each declaration names
//! what it ports and the line it began at.
//!
//! # Which arm
//!
//! `LLAMA_SUBPROCESS` is defined wherever the reference builds (every
//! platform but iOS, `llama.cpp.zmake`'s `subprocess_defines`), so
//! `is_supported()` is true there and false on iOS, where `create` fails.
//! Of `subprocess.h`'s three spawn paths, Darwin and glibc take
//! `posix_spawn`: `SUBPROCESS_SPAWN_VIA_FORK` is only AIX, OpenBSD and old
//! NetBSD. The working directory goes through
//! `posix_spawn_file_actions_addchdir_np` on both; a macOS 26 deployment
//! target spells it `posix_spawn_file_actions_addchdir`, which libSystem
//! implements as the same call. Windows' arm (`CreateProcessW`) is not
//! ported: the server's Windows build waits on it.

const std = @import("std");
const builtin = @import("builtin");

/// `FILE *`.
pub const File = anyopaque;

extern fn fdopen(fd: c_int, mode: [*:0]const u8) ?*File;
extern fn fclose(f: *File) c_int;
extern fn fileno(f: *File) c_int;
extern fn fcntl(fd: c_int, cmd: c_int, ...) c_int;
extern fn close(fd: c_int) c_int;
extern fn pipe(fds: *[2]c_int) c_int;
extern fn pipe2(fds: *[2]c_int, flags: c_int) c_int;
extern fn kill(pid: c_int, sig: c_int) c_int;
extern fn waitpid(pid: c_int, status: ?*c_int, options: c_int) c_int;

/// `posix_spawn_file_actions_t`: a pointer on Darwin, an 80-byte struct in
/// glibc. Opaque storage big enough for either.
const SpawnActions = extern struct { storage: [128]u8 align(16) = undefined };

extern fn posix_spawn_file_actions_init(a: *SpawnActions) c_int;
extern fn posix_spawn_file_actions_destroy(a: *SpawnActions) c_int;
extern fn posix_spawn_file_actions_addclose(a: *SpawnActions, fd: c_int) c_int;
extern fn posix_spawn_file_actions_adddup2(a: *SpawnActions, fd: c_int, newfd: c_int) c_int;
extern fn posix_spawn_file_actions_addchdir_np(a: *SpawnActions, path: [*:0]const u8) c_int;
extern fn posix_spawn(pid: *c_int, path: [*:0]const u8, actions: *const SpawnActions, attr: ?*const anyopaque, argv: [*]const ?[*:0]const u8, envp: [*]const ?[*:0]const u8) c_int;
extern fn posix_spawnp(pid: *c_int, file: [*:0]const u8, actions: *const SpawnActions, attr: ?*const anyopaque, argv: [*]const ?[*:0]const u8, envp: [*]const ?[*:0]const u8) c_int;

const F_DUPFD: c_int = 0;
const F_GETFD: c_int = 1;
const F_SETFD: c_int = 2;
const F_GETFL: c_int = 3;
const F_SETFL: c_int = 4;
const FD_CLOEXEC: c_int = 1;
const O_CLOEXEC: c_int = if (builtin.os.tag.isDarwin()) 0x1000000 else 0o2000000;
const O_NONBLOCK: c_int = if (builtin.os.tag.isDarwin()) 0x0004 else 0o4000;
const WNOHANG: c_int = 1;
const STDIN_FILENO: c_int = 0;
const STDOUT_FILENO: c_int = 1;
const STDERR_FILENO: c_int = 2;
const EXIT_FAILURE: c_int = 1;

fn errno() c_int {
    return std.c._errno().*;
}

fn setErrno(e: c_int) void {
    std.c._errno().* = e;
}

/// Ports `subprocess_option_e` (subprocess.h:75 @7fe450e19).
pub const option_combined_stdout_stderr: c_int = 0x1;
pub const option_inherit_environment: c_int = 0x2;
pub const option_enable_async: c_int = 0x4;
pub const option_no_window: c_int = 0x8;
pub const option_search_user_path: c_int = 0x10;
pub const option_enable_async_no_wait: c_int = 0x20;

/// Ports `subprocess_error_e` (subprocess.h:101 @7fe450e19).
const err_unknown: c_int = -1;
const err_invalid_options: c_int = -2;
const err_invalid_environment: c_int = -3;
const err_not_found: c_int = -4;
const err_permission_denied: c_int = -5;
const err_no_memory: c_int = -6;
const err_pipe: c_int = -7;
const err_spawn: c_int = -8;
const err_not_supported: c_int = -9;

/// Ports `subprocess_s` (subprocess.h:563 @7fe450e19), its POSIX members.
const Process = struct {
    stdin_file: ?*File = null,
    stdout_file: ?*File = null,
    stderr_file: ?*File = null,
    child: c_int = 0,
    return_status: c_int = 0,
    alive: c_int = 0,
    no_wait: c_int = 0,
};

/// Ports `subprocess_error_from_errno` (subprocess.h:622 @7fe450e19).
fn errorFromErrno(e: c_int) c_int {
    const E = std.posix.E;
    return switch (e) {
        @backingInt(E.NOENT), @backingInt(E.NOTDIR) => err_not_found,
        @backingInt(E.ACCES), @backingInt(E.PERM) => err_permission_denied,
        @backingInt(E.MFILE), @backingInt(E.NFILE), @backingInt(E.NOMEM) => err_no_memory,
        @backingInt(E.NOSYS) => err_not_supported,
        else => err_unknown,
    };
}

/// Ports `subprocess_fds_above_std` (subprocess.h:732 @7fe450e19): moves a
/// pipe end off 0, 1 or 2, keeping it close-on-exec.
fn fdsAboveStd(fds: *[2]c_int) c_int {
    for (0..2) |index| {
        if (fds[index] > STDERR_FILENO) continue;
        var moved = fcntl(fds[index], F_DUPFD, STDERR_FILENO + 1);
        if (moved != -1) {
            const fd_flags = fcntl(moved, F_GETFD, @as(c_int, 0));
            if (fd_flags == -1 or fcntl(moved, F_SETFD, fd_flags | FD_CLOEXEC) == -1) {
                const saved = errno();
                _ = close(moved);
                setErrno(saved);
                moved = -1;
            }
        }
        if (moved == -1) {
            const saved = errno();
            _ = close(fds[0]);
            _ = close(fds[1]);
            fds.* = .{ -1, -1 };
            setErrno(saved);
            return -1;
        }
        _ = close(fds[index]);
        fds[index] = moved;
    }
    return 0;
}

/// Ports `subprocess_pipe_cloexec` (subprocess.h:774 @7fe450e19): a pipe
/// whose ends later children do not inherit.
fn pipeCloexec(fds: *[2]c_int) c_int {
    if (comptime builtin.os.tag == .linux or builtin.os.tag == .freebsd or builtin.os.tag == .netbsd or builtin.os.tag == .openbsd) {
        if (pipe2(fds, O_CLOEXEC) == 0) return fdsAboveStd(fds);
        // Older kernels can lack pipe2 even when the C library declares it.
        if (errno() != @backingInt(std.posix.E.NOSYS)) return -1;
    }
    if (pipe(fds) != 0) return -1;
    for (0..2) |index| {
        const fd_flags = fcntl(fds[index], F_GETFD, @as(c_int, 0));
        if (fd_flags == -1 or fcntl(fds[index], F_SETFD, fd_flags | FD_CLOEXEC) == -1) {
            const saved = errno();
            _ = close(fds[0]);
            _ = close(fds[1]);
            fds.* = .{ -1, -1 };
            setErrno(saved);
            return -1;
        }
    }
    return fdsAboveStd(fds);
}

/// `posix_error` into a `subprocess_error_e`, `subprocess_error_spawn` for
/// an errno with no class of its own: the `goto cleanup` arm every spawn
/// step of `subprocess_create_ex` repeats.
fn spawnError(posix_error: c_int) c_int {
    const r = errorFromErrno(posix_error);
    return if (r == err_unknown) err_spawn else r;
}

/// Ports `subprocess_create_ex` (subprocess.h:826 @7fe450e19), its
/// `posix_spawn` arm.
fn createEx(command_line: [*]const ?[*:0]const u8, options: c_int, environment: ?[*]const ?[*:0]const u8, process_cwd: ?[*:0]const u8, out: *Process) c_int {
    var stdinfd: [2]c_int = .{ -1, -1 };
    var stdoutfd: [2]c_int = .{ -1, -1 };
    var stderrfd: [2]c_int = .{ -1, -1 };
    var result: c_int = err_unknown;
    var saved_errno: c_int = 0;
    var child: c_int = 0;
    const empty_environment = [_]?[*:0]const u8{null};
    var actions_created = false;
    var actions: SpawnActions = .{};

    const async_no_wait = option_enable_async_no_wait == (options & option_enable_async_no_wait);
    if (async_no_wait and option_enable_async != (options & option_enable_async)) {
        setErrno(@backingInt(std.posix.E.INVAL));
        return err_invalid_options;
    }
    if (option_inherit_environment == (options & option_inherit_environment)) {
        if (environment != null) {
            setErrno(@backingInt(std.posix.E.INVAL));
            return err_invalid_environment;
        }
    }

    out.* = .{};

    body: {
        if (pipeCloexec(&stdinfd) != 0) {
            saved_errno = errno();
            result = err_pipe;
            break :body;
        }
        if (pipeCloexec(&stdoutfd) != 0) {
            saved_errno = errno();
            result = err_pipe;
            break :body;
        }
        const combined = option_combined_stdout_stderr == (options & option_combined_stdout_stderr);
        if (!combined) {
            if (pipeCloexec(&stderrfd) != 0) {
                saved_errno = errno();
                result = err_pipe;
                break :body;
            }
        }

        const used_environment: [*]const ?[*:0]const u8 = if (environment) |e|
            e
        else if (option_inherit_environment == (options & option_inherit_environment))
            @ptrCast(std.c.environ)
        else
            &empty_environment;

        var posix_error = posix_spawn_file_actions_init(&actions);
        if (posix_error != 0) {
            saved_errno = posix_error;
            result = spawnError(posix_error);
            break :body;
        }
        actions_created = true;

        // Set working directory
        if (process_cwd) |cwd| {
            posix_error = posix_spawn_file_actions_addchdir_np(&actions, cwd);
            if (posix_error != 0) {
                saved_errno = posix_error;
                result = spawnError(posix_error);
                break :body;
            }
        }

        // Close the stdin write end, map the read end to stdin
        posix_error = posix_spawn_file_actions_addclose(&actions, stdinfd[1]);
        if (posix_error == 0) posix_error = posix_spawn_file_actions_adddup2(&actions, stdinfd[0], STDIN_FILENO);
        // Close the stdout read end, map the write end to stdout
        if (posix_error == 0) posix_error = posix_spawn_file_actions_addclose(&actions, stdoutfd[0]);
        if (posix_error == 0) posix_error = posix_spawn_file_actions_adddup2(&actions, stdoutfd[1], STDOUT_FILENO);
        if (posix_error == 0) {
            if (combined) {
                posix_error = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
            } else {
                // Close the stderr read end, map the write end to stderr
                posix_error = posix_spawn_file_actions_addclose(&actions, stderrfd[0]);
                if (posix_error == 0) posix_error = posix_spawn_file_actions_adddup2(&actions, stderrfd[1], STDERR_FILENO);
            }
        }
        if (posix_error != 0) {
            saved_errno = posix_error;
            result = spawnError(posix_error);
            break :body;
        }

        posix_error = if (option_search_user_path == (options & option_search_user_path))
            posix_spawnp(&child, command_line[0].?, &actions, null, command_line, used_environment)
        else
            posix_spawn(&child, command_line[0].?, &actions, null, command_line, used_environment);
        if (posix_error != 0) {
            saved_errno = posix_error;
            result = spawnError(posix_error);
            break :body;
        }

        // Close the stdin read end, store the stdin write end
        _ = close(stdinfd[0]);
        stdinfd[0] = -1;
        out.stdin_file = fdopen(stdinfd[1], "wb");
        if (out.stdin_file == null) {
            saved_errno = errno();
            result = errorFromErrno(saved_errno);
            break :body;
        }
        stdinfd[1] = -1;

        // Close the stdout write end, store the stdout read end
        _ = close(stdoutfd[1]);
        stdoutfd[1] = -1;
        out.stdout_file = fdopen(stdoutfd[0], "rb");
        if (out.stdout_file == null) {
            saved_errno = errno();
            result = errorFromErrno(saved_errno);
            break :body;
        }
        stdoutfd[0] = -1;

        // Set non blocking if we are async and asked not to wait.
        if (async_no_wait) {
            const fd = fileno((out.stdout_file.?));
            _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, @as(c_int, 0)) | O_NONBLOCK);
        }

        if (combined) {
            out.stderr_file = out.stdout_file;
        } else {
            // Close the stderr write end, store the stderr read end
            _ = close(stderrfd[1]);
            stderrfd[1] = -1;
            out.stderr_file = fdopen(stderrfd[0], "rb");
            if (out.stderr_file == null) {
                saved_errno = errno();
                result = errorFromErrno(saved_errno);
                break :body;
            }
            stderrfd[0] = -1;
            if (async_no_wait) {
                const fd = fileno((out.stderr_file.?));
                _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, @as(c_int, 0)) | O_NONBLOCK);
            }
        }

        // Store the child's pid
        out.child = child;
        child = 0;
        out.alive = 1;
        out.no_wait = @intFromBool(async_no_wait);
        result = 0;
    }

    // cleanup:
    if (result != 0 and saved_errno == 0) saved_errno = errno();
    if (result == err_unknown and saved_errno != 0) result = errorFromErrno(saved_errno);
    if (actions_created) _ = posix_spawn_file_actions_destroy(&actions);
    if (result != 0) {
        if (child != 0) {
            _ = kill(child, 9);
            _ = waitpid(child, null, 0);
        }
        if (out.stdin_file) |f| {
            _ = fclose(f);
            out.stdin_file = null;
        }
        if (out.stdout_file) |f| {
            _ = fclose(f);
            if (out.stderr_file) |e| {
                if (e != f) _ = fclose(e);
            }
            out.stdout_file = null;
            out.stderr_file = null;
        }
    }
    for ([_]c_int{ stdinfd[0], stdinfd[1], stdoutfd[0], stdoutfd[1], stderrfd[0], stderrfd[1] }) |fd| {
        if (fd != -1) _ = close(fd);
    }
    if (result != 0 and saved_errno != 0) setErrno(saved_errno);
    return result;
}

/// `WIFEXITED` and `WEXITSTATUS`, as `<sys/wait.h>` defines them on
/// Darwin and glibc alike.
fn exitStatus(status: c_int) c_int {
    if ((status & 0x7f) == 0) return (status >> 8) & 0xff;
    return EXIT_FAILURE;
}

/// Ports `subprocess_join` (subprocess.h:1903 @7fe450e19), its POSIX arm.
fn processJoin(p: *Process, out_return_code: ?*c_int) c_int {
    if (p.stdin_file) |f| {
        _ = fclose(f);
        p.stdin_file = null;
    }
    if (p.child != 0) {
        var status: c_int = 0;
        if (p.child != waitpid(p.child, &status, 0)) return -1;
        p.child = 0;
        p.return_status = exitStatus(status);
        p.alive = 0;
    }
    if (out_return_code) |r| r.* = p.return_status;
    return 0;
}

/// Ports `subprocess_destroy` (subprocess.h:1963 @7fe450e19).
fn processDestroy(p: *Process) void {
    if (p.stdin_file) |f| {
        _ = fclose(f);
        p.stdin_file = null;
    }
    if (p.stdout_file) |f| {
        _ = fclose(f);
        if (p.stderr_file) |e| {
            if (e != f) _ = fclose(e);
        }
        p.stdout_file = null;
        p.stderr_file = null;
    }
}

/// Ports `subprocess_alive` (subprocess.h:2140 @7fe450e19): reaps the child
/// when it has exited.
fn processAlive(p: *Process) c_int {
    var is_alive = p.alive;
    if (is_alive == 0) return 0;
    var status: c_int = 0;
    is_alive = @intFromBool(waitpid(p.child, &status, WNOHANG) == 0);
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

/// Ports `common_subproc` (subproc.h:22 @7fe450e19): a RAII wrapper around
/// `subprocess_s`. Not copyable: hold it in place.
pub const Subproc = struct {
    proc: Process = .{},
    is_created: std.atomic.Value(bool) = .init(false),

    /// Ports `common_subproc`'s `is_supported` (subproc.cpp:3 @7fe450e19): true
    /// where the reference defines `LLAMA_SUBPROCESS`.
    pub fn isSupported() bool {
        return comptime builtin.os.tag != .ios and builtin.os.tag != .windows;
    }

    /// Ports `common_subproc`'s destructor (subproc.cpp:23 @7fe450e19).
    pub fn deinit(self: *Subproc) void {
        if (self.is_created.load(.seq_cst)) {
            processDestroy(&self.proc);
            self.is_created.store(false, .seq_cst);
        }
    }

    /// Ports `common_subproc`'s `create` (subproc.cpp:30 @7fe450e19): spawns a
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
    pub fn create(self: *Subproc, args: []const []const u8, options: c_int, env: []const []const u8, cwd: ?[]const u8) bool {
        if (!isSupported()) return false;
        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const argv = toCstrVec(a, args) catch return false;
        const cwd_z: ?[*:0]const u8 = if (cwd) |d| (a.dupeSentinel(u8, d, 0) catch return false).ptr else null;
        var result: c_int = undefined;
        if (env.len == 0 and cwd == null) {
            result = createEx(argv, options, null, null, &self.proc);
        } else {
            const envp = toCstrVec(a, env) catch return false;
            result = createEx(argv, options, if (env.len == 0) null else envp, cwd_z, &self.proc);
        }
        self.is_created.store(result == 0, .seq_cst);
        return result == 0;
    }

    /// Ports `to_cstr_vec` (subproc.cpp:13 @7fe450e19): a null-terminated
    /// `char *` array.
    fn toCstrVec(a: std.mem.Allocator, v: []const []const u8) ![*]const ?[*:0]const u8 {
        const r = try a.alloc(?[*:0]const u8, v.len + 1);
        for (v, 0..) |s, i| r[i] = (try a.dupeSentinel(u8, s, 0)).ptr;
        r[v.len] = null;
        return r.ptr;
    }

    /// Ports `common_subproc`'s `has_handle` (subproc.cpp:49 @7fe450e19).
    fn hasHandle(self: *const Subproc) bool {
        if (!self.is_created.load(.seq_cst)) return false;
        return self.proc.child > 0;
    }

    /// Ports `common_subproc`'s `alive` (subproc.cpp:60 @7fe450e19).
    pub fn alive(self: *Subproc) bool {
        return self.is_created.load(.seq_cst) and processAlive(&self.proc) != 0;
    }

    /// Ports `common_subproc`'s `stdin_file` (subproc.cpp:64 @7fe450e19).
    pub fn stdinFile(self: *Subproc) ?*File {
        return if (self.is_created.load(.seq_cst)) self.proc.stdin_file else null;
    }

    /// Ports `common_subproc`'s `stdout_file` (subproc.cpp:68 @7fe450e19).
    pub fn stdoutFile(self: *Subproc) ?*File {
        return if (self.is_created.load(.seq_cst)) self.proc.stdout_file else null;
    }

    /// Ports `common_subproc`'s `stderr_file` (subproc.cpp:72 @7fe450e19), over
    /// `subprocess_stderr` (subprocess.h:1895 @7fe450e19): null when stderr
    /// is combined with stdout.
    pub fn stderrFile(self: *Subproc) ?*File {
        if (!self.is_created.load(.seq_cst)) return null;
        if (self.proc.stdout_file != self.proc.stderr_file) return self.proc.stderr_file;
        return null;
    }

    /// Ports `common_subproc`'s `close_stdin` (subproc.cpp:76 @7fe450e19):
    /// EOF to the child while it runs.
    pub fn closeStdin(self: *Subproc) void {
        if (self.is_created.load(.seq_cst)) {
            if (self.proc.stdin_file) |f| {
                _ = fclose(f);
                self.proc.stdin_file = null;
            }
        }
    }

    /// Ports `common_subproc`'s `terminate` (subproc.cpp:83 @7fe450e19), over
    /// `subprocess_terminate` (subprocess.h:2002 @7fe450e19): `SIGKILL`.
    pub fn terminate(self: *Subproc) void {
        if (self.hasHandle()) _ = kill(self.proc.child, 9);
    }

    /// Ports `common_subproc`'s `join` (subproc.cpp:89 @7fe450e19): waits for
    /// the child, releases it.
    ///
    /// Return: its exit code -- `EXIT_FAILURE` when a signal ended it, -1
    /// when nothing was created.
    pub fn join(self: *Subproc) c_int {
        var exit_code: c_int = -1;
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

extern fn fwrite(p: [*]const u8, size: usize, n: usize, f: *File) usize;
extern fn read(fd: c_int, buf: [*]u8, n: usize) isize;

test "a child's combined output, its exit code, and a working directory" {
    var p: Subproc = .{};
    defer p.deinit();
    try std.testing.expect(p.create(&.{ "sh", "-c", "pwd; echo err >&2; exit 3" }, option_combined_stdout_stderr | option_inherit_environment | option_search_user_path, &.{}, "/"));
    p.closeStdin();
    var buf: [64]u8 = undefined;
    var n: usize = 0;
    while (true) {
        const r = read(fileno((p.stdoutFile().?)), buf[n..].ptr, buf.len - n);
        if (r <= 0) break;
        n += @intCast(r);
    }
    try std.testing.expectEqualStrings("/\nerr\n", buf[0..n]);
    try std.testing.expectEqual(@as(c_int, 3), p.join());
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
