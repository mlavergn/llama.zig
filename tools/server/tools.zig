//! The server's built-in tools (`--tools`) and the MCP servers' tools
//! behind them, on `GET /tools` and `POST /tools`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-tools.cpp` and
//! `llama.cpp/tools/server/server-tools.h` (llama.cpp v0.7.0, `620d2225c`):
//! the seven tools, `tools_io` with its host (`tools_io_basic`) and isolate
//! (`tools_io_isolate`, `tools_io_container`, `tools_io_ssh`)
//! implementations, `run_subprocess`, the `--tools-runtime` runtimes,
//! `server_mcp_tool`, and `server_tools` with its two handlers. Also
//! `glob_match` and `glob_class_match` from `llama.cpp/common/common.cpp`,
//! which only these tools call. Each declaration names what it ports and
//! the line it began at.
//!
//! # Its shape
//!
//! `server_tool` and its eight subclasses are one `Tool` with a `Kind`;
//! `tools_io` and its four subclasses are one `Io`, a tagged union of the
//! host and an isolate, the isolate's transport (`<engine> exec` or `ssh`)
//! a further union. A C++ throw is an error with its `what()` in
//! `common.what` (`error.JsonError`'s in `cli/json`'s buffer). Every JSON
//! value a call builds lives in that call's arena. `std::filesystem` is
//! ported as far as the tools use it, for POSIX paths:
//! `lexically_normal`, `operator/`, `filename()`, `directory_iterator`
//! (readdir order), `is_directory`/`is_regular_file`/`file_size` (through
//! `stat`) and `create_directories`.
//!
//! # POSIX only
//!
//! The `_WIN32` arms -- the OEM code page conversion, `cmd /c`, NTFS's
//! case-folded junk names, junction detection -- are not ported, and only
//! Darwin has run this.

const std = @import("std");
const json = @import("json");
const tc = @import("tools_common");
const libcxx = @import("llama").libcxx;
const regex = @import("llama").regex;
const http = @import("http.zig");
const common = @import("common.zig");
const queue = @import("queue.zig");
const mcp = @import("mcp.zig");

const Value = json.Value;
const Allocator = std.mem.Allocator;
const Subproc = tc.subproc.Subproc;
const srv = common.srv;
const throw = common.throw;

const gpa = std.heap.c_allocator;

extern fn fileno(f: *anyopaque) c_int;
extern fn fwrite(p: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern fn fflush(f: *anyopaque) c_int;
extern fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern fn getcwd(buf: [*]u8, size: usize) ?[*:0]u8;
extern fn mkdir(path: [*:0]const u8, mode: c_uint) c_int;
extern fn unlink(path: [*:0]const u8) c_int;
extern fn strtoull(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_ulonglong;
extern fn pthread_self() usize;

fn errno() c_int {
    return std.c._errno().*;
}

/// A NUL-terminated copy for a `%s` argument, freed with `common.freeZ`.
fn z(s: []const u8) [*:0]const u8 {
    return common.dupZ(s);
}

// -----------------------------------------------------------------------------
// internal helpers

/// Ports `SERVER_TOOL_GIT_LS_FILES_MAX_OUTPUT` (server-tools.cpp:88 @620d2225c).
const git_ls_files_max_output: usize = 8 * 1024 * 1024; // 8 MB
/// Ports `SERVER_TOOL_LIST_ENTRIES_TIMEOUT` (server-tools.cpp:90 @620d2225c): the
/// budget for one listing call, shared by the git and walker paths.
const list_entries_timeout: c_int = 15; // seconds

/// Ports `list_kind` (server-tools.cpp:93 @620d2225c): the entry kinds a directory
/// listing may return.
pub const ListKind = enum { files, dirs, all };

/// Ports `home_dir` (server-tools.cpp:112 @620d2225c): `HOME`. The C++ reads it
/// once, at first use; nothing here sets it, so reading it each time
/// gives the same answer.
fn homeDir() []const u8 {
    const h = std.c.getenv("HOME") orelse return "";
    return std.mem.span(h);
}

/// Ports `expand_home` (server-tools.cpp:127 @620d2225c): a leading `~` or `~/`.
fn expandHome(a: Allocator, path: []const u8) ![]const u8 {
    if (path.len == 0 or path[0] != '~') return path;
    if (path.len > 1 and path[1] != '/' and path[1] != '\\') return path;
    const home = homeDir();
    if (home.len == 0) return path;
    return std.mem.concat(a, u8, &.{ home, path[1..] });
}

/// Ports `entry_depth` (server-tools.cpp:136 @620d2225c): `"a/b/c"` is 3.
fn entryDepth(rel: []const u8) c_int {
    return 1 + @as(c_int, @intCast(std.mem.count(u8, rel, "/")));
}

/// Ports `SERVER_TOOL_JUNK_DIR_NAMES` (server-tools.cpp:142 @620d2225c): directories a
/// listing reports but never descends into.
const junk_dir_names = [_][]const u8{
    ".git",   ".svn",  ".hg",     "node_modules", "__pycache__",
    ".venv",  "venv",  "dist",    "build",        "target",
    ".cache", ".idea", ".vscode",
};

fn isJunk(name: []const u8) bool {
    for (junk_dir_names) |n| {
        if (std.mem.eql(u8, n, name)) return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// std::filesystem, as far as the tools use it (POSIX paths)

fn isAbsolute(p: []const u8) bool {
    return p.len != 0 and p[0] == '/';
}

/// `fs::path::operator/`: `rhs` replaces an absolute; otherwise one
/// separator joins them unless `lhs` is empty or already ends in one.
fn pathJoin(a: Allocator, lhs: []const u8, rhs: []const u8) ![]u8 {
    if (isAbsolute(rhs)) return a.dupe(u8, rhs);
    if (lhs.len == 0 or lhs[lhs.len - 1] == '/') return std.mem.concat(a, u8, &.{ lhs, rhs });
    return std.mem.concat(a, u8, &.{ lhs, "/", rhs });
}

/// `fs::path::filename()`: the last element, empty after a trailing
/// separator.
fn pathFilename(p: []const u8) []const u8 {
    const slash = std.mem.findScalarLast(u8, p, '/') orelse return p;
    return p[slash + 1 ..];
}

/// `fs::path::parent_path()` of a path whose last element is a name.
fn pathParent(p: []const u8) []const u8 {
    const slash = std.mem.findScalarLast(u8, p, '/') orelse return "";
    var end = slash;
    while (end > 0 and p[end - 1] == '/') end -= 1;
    if (end == 0) return p[0..1];
    return p[0..end];
}

/// `fs::path::lexically_normal()`, as libc++ implements it for POSIX
/// paths: separators collapsed, `.` dropped, `name/..` folded, `..` at the
/// root dropped, a trailing separator kept only after a name.
fn lexicallyNormal(a: Allocator, p: []const u8) ![]u8 {
    if (p.len == 0) return a.dupe(u8, "");
    const PartKind = enum { root, name, dotdot };
    const Part = struct { s: []const u8, k: PartKind };
    var parts: std.ArrayList(Part) = .empty;
    var maybe_need_trailing_sep = false;
    var i: usize = 0;
    if (p[0] == '/') {
        try parts.append(a, .{ .s = "/", .k = .root });
        while (i < p.len and p[i] == '/') i += 1;
    }
    while (i < p.len) {
        const start = i;
        while (i < p.len and p[i] != '/') i += 1;
        const name = p[start..i];
        const trailing_sep = blk: {
            var j = i;
            while (j < p.len and p[j] == '/') j += 1;
            const had_sep = j > i;
            i = j;
            break :blk had_sep and j == p.len;
        };
        const last: ?PartKind = if (parts.items.len == 0) null else parts.items[parts.items.len - 1].k;
        if (std.mem.eql(u8, name, ".")) {
            maybe_need_trailing_sep = true;
        } else if (std.mem.eql(u8, name, "..")) {
            if (last == .name) {
                _ = parts.pop();
            } else if (last != .root) {
                try parts.append(a, .{ .s = "..", .k = .dotdot });
            }
            maybe_need_trailing_sep = last == .name;
        } else {
            try parts.append(a, .{ .s = name, .k = .name });
            maybe_need_trailing_sep = false;
        }
        if (trailing_sep) maybe_need_trailing_sep = true;
    }
    if (parts.items.len == 0) return a.dupe(u8, ".");
    const need_trailing_sep = maybe_need_trailing_sep and parts.items[parts.items.len - 1].k == .name;
    var result: []u8 = try a.dupe(u8, "");
    for (parts.items) |part| result = try pathJoin(a, result, part.s);
    if (need_trailing_sep) result = try std.mem.concat(a, u8, &.{ result, "/" });
    return result;
}

/// `fs::current_path(ec)`.
fn currentPath(a: Allocator) ?[]u8 {
    var buf: [4096]u8 = undefined;
    const r = getcwd(&buf, buf.len) orelse return null;
    return a.dupe(u8, std.mem.span(r)) catch null;
}

const FileType = enum { none, regular, directory, other };

/// `fs::status(p, ec).type()`, following links.
fn statType(path: []const u8, size: ?*u64) FileType {
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= buf.len) return .none;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const st = tc.fs_stat.stat(@ptrCast(&buf), true) orelse return .none;
    if (size) |s| s.* = @intCast(st.size);
    return switch (st.mode & tc.fs_stat.IFMT) {
        tc.fs_stat.IFREG => .regular,
        tc.fs_stat.IFDIR => .directory,
        else => .other,
    };
}

/// `fs::create_directories(p, ec)`.
///
/// Return: false on an error.
fn createDirectories(a: Allocator, p: []const u8) bool {
    if (p.len == 0) return true;
    switch (statType(p, null)) {
        .directory => return true,
        .none => {},
        else => return false,
    }
    const parent = pathParent(p);
    if (parent.len != 0 and parent.len < p.len and !createDirectories(a, parent)) return false;
    const pz = a.dupeSentinel(u8, p, 0) catch return false;
    if (mkdir(pz.ptr, 0o777) == 0) return true;
    return statType(p, null) == .directory;
}

/// The bytes of a whole file, as `std::ifstream` and `ss << f.rdbuf()`
/// read them: null when it cannot be opened; a read error ends the copy.
fn readWhole(a: Allocator, path: []const u8) ?[]u8 {
    const pz = a.dupeSentinel(u8, path, 0) catch return null;
    const fd = std.c.open(pz.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = read(fd, &buf, buf.len);
        if (n <= 0) break;
        out.appendSlice(a, buf[0..@intCast(n)]) catch return null;
    }
    return out.items;
}

// -----------------------------------------------------------------------------
// tools_io

/// Ports `tools_io`'s `exec_result` (server-tools.cpp:149 @620d2225c).
const ExecResult = struct {
    output: []const u8 = "",
    exit_code: c_int = -1,
    timed_out: bool = false,
};

/// Ports `tools_io`'s `list_entry` (server-tools.cpp:164 @620d2225c).
const ListEntry = struct {
    rel: []const u8, // '/'-separated, relative to `base`
    is_dir: bool = false,
};

/// Ports `tools_io`'s `list_result` (server-tools.cpp:168 @620d2225c).
const ListResult = struct {
    entries: std.ArrayList(ListEntry) = .empty,
    err: []const u8 = "", // set when `base` is not a directory
    truncated: bool = false, // set when the walk could not see everything
};

/// The `on_chunk` callback of `run`: each chunk of output as it is read;
/// false terminates the process early.
const OnChunk = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, chunk: []const u8) bool,
};

/// Ports `run_subprocess` (server-tools.cpp:187 @620d2225c): runs `args` with a
/// supervisory timeout, writing `stdin_data` first and reading raw output
/// as it comes, past the size cap so the child never blocks on a full pipe.
fn runSubprocess(
    a: Allocator,
    args: []const []const u8,
    max_output: usize,
    timeout_secs: c_int,
    on_chunk: ?OnChunk,
    combine_stderr: bool,
    cwd: []const u8,
    stdin_data: ?[]const u8,
) !ExecResult {
    var res: ExecResult = .{};

    var proc: Subproc = .{};
    defer proc.deinit();

    var options = tc.subproc.option_no_window | tc.subproc.option_inherit_environment | tc.subproc.option_search_user_path;
    if (combine_stderr) options |= tc.subproc.option_combined_stdout_stderr;

    if (!proc.create(args, options, &.{}, if (cwd.len == 0) null else cwd)) {
        res.output = "failed to spawn process";
        return res;
    }

    const Timeout = struct {
        done: std.atomic.Value(bool) = .init(false),
        timed_out: std.atomic.Value(bool) = .init(false),
        proc: *Subproc,
        deadline: i128,

        fn run(self: *@This()) void {
            while (!self.done.load(.seq_cst)) {
                if (mcp.nowNs() >= self.deadline) {
                    self.timed_out.store(true, .seq_cst);
                    self.proc.terminate();
                    return;
                }
                mcp.sleepMs(100);
            }
        }
    };
    var tm: Timeout = .{ .proc = &proc, .deadline = mcp.nowNs() + @as(i128, timeout_secs) * std.time.ns_per_s };
    const timeout_thread = std.Thread.spawn(.{}, Timeout.run, .{&tm}) catch null;

    // write stdin before reading stdout, the child drains stdin as it goes
    // always close stdin, a transport client waits forever if its stdin pipe stays open
    if (proc.stdinFile()) |in| {
        if (stdin_data) |d| {
            // a short write is not an error by itself, the exit code below decides
            if (d.len != 0) _ = fwrite(d.ptr, 1, d.len, in);
        }
        _ = fflush(in);
    }
    proc.closeStdin();

    var output: std.ArrayList(u8) = .empty;
    var truncated = false;
    if (proc.stdoutFile()) |f| {
        // read raw bytes, not lines: the output can hold NUL and must arrive as soon as it is ready
        // keep draining past the size cap, else the child blocks on a full pipe
        var buf: [4096]u8 = undefined;
        const fd = fileno(f);
        while (true) {
            var n = read(fd, &buf, buf.len);
            while (n < 0 and errno() == @backingInt(std.posix.E.INTR)) n = read(fd, &buf, buf.len);
            if (n <= 0) break;
            if (truncated) continue;
            const len: usize = @intCast(n);
            if (output.items.len + len <= max_output) {
                try output.appendSlice(a, buf[0..len]);
                if (on_chunk) |cb| {
                    if (!cb.f(cb.ctx, buf[0..len])) {
                        proc.terminate();
                        break;
                    }
                }
            } else {
                const remaining = max_output - output.items.len;
                try output.appendSlice(a, buf[0..remaining]);
                if (on_chunk) |cb| {
                    if (remaining > 0) _ = cb.f(cb.ctx, buf[0..remaining]);
                }
                truncated = true;
            }
        }
    }

    tm.done.store(true, .seq_cst);
    if (timeout_thread) |t| t.join();

    res.exit_code = proc.join();

    res.output = output.items; // console_output_to_utf8: the identity off Windows
    res.timed_out = tm.timed_out.load(.seq_cst);
    if (truncated) res.output = try std.mem.concat(a, u8, &.{ res.output, "\n[output truncated]" });
    return res;
}

/// Ports `tools_io_basic` (server-tools.cpp:298 @620d2225c): the tools run on
/// the host. `cwd`, if set, resolves relative paths and is where `run`
/// runs.
const Basic = struct {
    cwd: []const u8 = "",

    /// Ports `tools_io_basic`'s `resolve` (server-tools.cpp:305 @620d2225c): `~`
    /// expanded, then against `cwd` (or the server's working directory), so
    /// the result is always absolute; `.` and `..` dropped.
    fn resolve(self: *const Basic, a: Allocator, path: []const u8) ![]const u8 {
        const p = try expandHome(a, path);

        var full: []const u8 = p;
        if (!isAbsolute(full)) {
            if (self.cwd.len == 0) {
                const cur = currentPath(a) orelse return p;
                full = try pathJoin(a, cur, full);
            } else {
                full = try pathJoin(a, self.cwd, full);
            }
        }

        // drop "." and ".." so they never reach git or the client
        full = try lexicallyNormal(a, full);
        // a trailing ".." normalizes to a path that ends with a separator
        if (full.len > 1 and full[full.len - 1] == '/') full = full[0 .. full.len - 1];
        return full;
    }

    /// Ports `tools_io_basic`'s `is_directory` (server-tools.cpp:329 @620d2225c).
    fn isDirectory(self: *const Basic, a: Allocator, path: []const u8) !bool {
        return statType(try self.resolve(a, path), null) == .directory;
    }

    /// Ports `tools_io_basic`'s `is_regular_file` (server-tools.cpp:334 @620d2225c).
    fn isRegularFile(self: *const Basic, a: Allocator, path: []const u8) !bool {
        return statType(try self.resolve(a, path), null) == .regular;
    }

    /// Ports `tools_io_basic`'s `file_size` (server-tools.cpp:339 @620d2225c): a
    /// regular file's size; anything else is an error.
    fn fileSize(self: *const Basic, a: Allocator, path: []const u8, out_size: *u64) !bool {
        var size: u64 = 0;
        const ty = statType(try self.resolve(a, path), &size);
        out_size.* = if (ty == .regular) size else std.math.maxInt(u64);
        return ty == .regular;
    }

    /// Ports `tools_io_basic`'s `read_file` (server-tools.cpp:345 @620d2225c).
    fn readFile(self: *const Basic, a: Allocator, path: []const u8) !?[]const u8 {
        return readWhole(a, try self.resolve(a, path));
    }

    /// Ports `tools_io_basic`'s `write_file` (server-tools.cpp:354 @620d2225c): parent
    /// directories created first.
    fn writeFile(self: *const Basic, a: Allocator, path: []const u8, content: []const u8) !bool {
        const fpath = try self.resolve(a, path);
        const parent = pathParent(fpath);
        if (parent.len != 0) {
            if (!createDirectories(a, parent)) return false;
        }
        const pz = try a.dupeSentinel(u8, fpath, 0);
        const fd = std.c.open(pz.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o666));
        if (fd < 0) return false;
        defer _ = std.c.close(fd);
        var off: usize = 0;
        while (off < content.len) {
            const n = write(fd, content[off..].ptr, content.len - off);
            if (n <= 0) return false;
            off += @intCast(n);
        }
        return true;
    }

    /// Ports `tools_io_basic`'s `list_entries` (server-tools.cpp:367 @620d2225c): `git
    /// ls-files` for files inside a repository, the walker otherwise.
    fn listEntries(self: *const Basic, a: Allocator, base: []const u8, max_depth: c_int, kind: ListKind) !ListResult {
        var out: ListResult = .{};

        if (statType(base, null) != .directory) {
            out.err = "path does not exist or is not a directory";
            return out;
        }

        const deadline = mcp.nowNs() + @as(i128, list_entries_timeout) * std.time.ns_per_s;

        // git ls-files cannot list directories; use the walker when they are requested
        if (kind == .files) {
            const res = try self.run(a, &.{ "git", "-C", base, "ls-files", "--cached", "--others", "--exclude-standard" }, git_ls_files_max_output, list_entries_timeout, null);

            if (res.exit_code == 0 and !res.timed_out) {
                var it = GetLines.init(res.output);
                while (it.next()) |raw| {
                    var line = raw;
                    if (line.len != 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
                    if (line.len == 0) continue;
                    const l = try a.dupe(u8, line);
                    std.mem.replaceScalar(u8, l, '\\', '/');
                    if (max_depth > 0 and entryDepth(l) > max_depth) continue;
                    if (try self.isRegularFile(a, try pathJoin(a, base, l))) {
                        try out.entries.append(a, .{ .rel = l, .is_dir = false });
                    }
                }
                return out;
            }
        }

        out.entries = try listEntriesFallback(a, base, max_depth, kind, deadline, &out.truncated);
        return out;
    }

    /// Ports `tools_io_basic`'s `run` (server-tools.cpp:404 @620d2225c): stdout and
    /// stderr combined, in `cwd`.
    fn run(self: *const Basic, a: Allocator, args: []const []const u8, max_output: usize, timeout_secs: c_int, on_chunk: ?OnChunk) !ExecResult {
        return runSubprocess(a, args, max_output, timeout_secs, on_chunk, true, self.cwd, null);
    }

    /// Ports `tools_io_basic`'s `list_entries_fallback` (server-tools.cpp:457 @620d2225c): a
    /// depth-first walk that lists junk directories without entering them
    /// and never follows a link.
    fn listEntriesFallback(a: Allocator, base: []const u8, max_depth: c_int, kind: ListKind, deadline: i128, truncated: *bool) !std.ArrayList(ListEntry) {
        var result: std.ArrayList(ListEntry) = .empty;

        const Frame = struct { dir: []const u8, rel_dir: []const u8, depth: c_int };
        var stack: std.ArrayList(Frame) = .empty;
        try stack.append(a, .{ .dir = base, .rel_dir = "", .depth = 0 });

        while (stack.pop()) |frame| {
            if (mcp.nowNs() >= deadline) {
                truncated.* = true;
                return result;
            }

            const dz = try a.dupeSentinel(u8, frame.dir, 0);
            const dirp = std.c.opendir(dz.ptr) orelse {
                // permission errors are skipped (skip_permission_denied); anything else is a subtree the caller never sees
                if (errno() != @backingInt(std.posix.E.ACCES)) truncated.* = true;
                continue;
            };
            defer _ = std.c.closedir(dirp);
            while (true) {
                std.c._errno().* = 0;
                const ent = std.c.readdir(dirp) orelse {
                    if (errno() != 0) truncated.* = true;
                    break;
                };
                if (mcp.nowNs() >= deadline) {
                    truncated.* = true;
                    return result;
                }
                const fname_z: [*:0]const u8 = @ptrCast(&ent.name);
                const fname = std.mem.span(fname_z);
                if (std.mem.eql(u8, fname, ".") or std.mem.eql(u8, fname, "..")) continue;
                const entry_path = try pathJoin(a, frame.dir, fname);
                const ty = statType(entry_path, null);
                if (ty == .none) continue; // the status could not be read
                if (ty == .directory) {
                    const rel = try pathJoin(a, frame.rel_dir, fname);
                    if (kind == .dirs or kind == .all) try result.append(a, .{ .rel = rel, .is_dir = true });
                    // junk directories stay selectable but are never walked: they can be enormous
                    if (isJunk(fname)) continue;
                    if (!isLink(a, entry_path) and (max_depth == 0 or frame.depth + 1 < max_depth)) {
                        try stack.append(a, .{ .dir = entry_path, .rel_dir = rel, .depth = frame.depth + 1 });
                    }
                } else if (ty == .regular) {
                    if (kind == .files or kind == .all) try result.append(a, .{ .rel = try pathJoin(a, frame.rel_dir, fname), .is_dir = false });
                }
            }
        }

        return result;
    }

    /// Ports `tools_io_basic`'s `is_link` (server-tools.cpp:416 @620d2225c): a link can
    /// point back to an ancestor, so it is never walked.
    fn isLink(a: Allocator, path: []const u8) bool {
        const pz = a.dupeSentinel(u8, path, 0) catch return true;
        const st = tc.fs_stat.stat(pz.ptr, false) orelse return true;
        return (st.mode & tc.fs_stat.IFMT) == tc.fs_stat.IFLNK;
    }
};

/// `std::getline` over a string: the lines between `'\n'`s, the last one
/// even without one, none after a final `'\n'`.
const GetLines = struct {
    text: []const u8,
    pos: usize = 0,

    fn init(text: []const u8) GetLines {
        return .{ .text = text };
    }

    fn next(self: *GetLines) ?[]const u8 {
        if (self.pos >= self.text.len) return null;
        const start = self.pos;
        if (std.mem.findScalarPos(u8, self.text, start, '\n')) |nl| {
            self.pos = nl + 1;
            return self.text[start..nl];
        }
        self.pos = self.text.len;
        return self.text[start..];
    }
};

/// Ports `SERVER_TOOL_ISOLATE_EXEC_TIMEOUT` (server-tools.cpp:518 @620d2225c): auxiliary
/// isolate calls; `exec_shell_command` uses its own.
const isolate_exec_timeout: c_int = 15; // seconds
/// Ports `SERVER_TOOL_ISOLATE_READ_FILE_MAX_SIZE` (server-tools.cpp:519 @620d2225c).
const isolate_read_file_max_size: usize = 64 * 1024 * 1024; // 64 MB

/// Ports `tools_io_isolate` (server-tools.cpp:524 @620d2225c): every operation as a
/// command inside an isolate -- a container (`tools_io_container`) or a
/// remote host (`tools_io_ssh`) -- that provides `sh`, `cat`, `wc`,
/// `mkdir`, `dirname`, `find` and `timeout`. Paths are POSIX.
const Isolate = struct {
    cwd: []const u8 = "",
    transport: Transport,

    const Transport = union(enum) {
        container: struct { bin: []const u8, container_id: []const u8 },
        ssh: struct { target: []const u8 },
    };

    /// Ports `tools_io_isolate`'s `resolve` (server-tools.cpp:531 @620d2225c): against
    /// `cwd` when relative and `cwd` is set.
    fn resolve(self: *const Isolate, a: Allocator, path: []const u8) ![]const u8 {
        if (self.cwd.len == 0 or (path.len != 0 and path[0] == '/')) return path;
        return std.mem.concat(a, u8, &.{ self.cwd, "/", path });
    }

    /// Ports `tools_io_isolate`'s `is_directory` (server-tools.cpp:538 @620d2225c).
    fn isDirectory(self: *const Isolate, a: Allocator, path: []const u8) !bool {
        return self.shellTest(a, "-d", try self.resolve(a, path));
    }

    /// Ports `tools_io_isolate`'s `is_regular_file` (server-tools.cpp:542 @620d2225c).
    fn isRegularFile(self: *const Isolate, a: Allocator, path: []const u8) !bool {
        return self.shellTest(a, "-f", try self.resolve(a, path));
    }

    /// Ports `tools_io_isolate`'s `file_size` (server-tools.cpp:546 @620d2225c): `wc
    /// -c`, read with `std::stoull`.
    fn fileSize(self: *const Isolate, a: Allocator, path: []const u8, out_size: *u64) !bool {
        const res = try self.exec(a, &.{ "sh", "-c", "wc -c < \"$1\"", "_", try self.resolve(a, path) }, 64, true);
        if (res.exit_code != 0 or res.timed_out) return false;
        const sz = try a.dupeSentinel(u8, res.output, 0);
        var end: ?[*:0]const u8 = null;
        std.c._errno().* = 0;
        const v = strtoull(sz.ptr, &end, 10);
        if (end == null or end.? == sz.ptr or errno() == @backingInt(std.posix.E.RANGE)) return false;
        out_size.* = v;
        return true;
    }

    /// Ports `tools_io_isolate`'s `read_file` (server-tools.cpp:558 @620d2225c): `cat`,
    /// with stderr kept out of the bytes.
    fn readFile(self: *const Isolate, a: Allocator, path: []const u8) !?[]const u8 {
        const res = try self.exec(a, &.{ "cat", "--", try self.resolve(a, path) }, isolate_read_file_max_size, false);
        if (res.exit_code != 0 or res.timed_out) return null;
        return res.output;
    }

    /// Ports `tools_io_isolate`'s `write_file` (server-tools.cpp:566 @620d2225c): the
    /// content travels on stdin.
    fn writeFile(self: *const Isolate, a: Allocator, path: []const u8, content: []const u8) !bool {
        const argv = try self.buildArgv(a, &.{ "sh", "-c", "mkdir -p \"$(dirname \"$1\")\" && cat > \"$1\"", "_", try self.resolve(a, path) }, true);
        const res = try runSubprocess(a, argv, 4096, isolate_exec_timeout, null, true, "", content);
        return res.exit_code == 0 and !res.timed_out;
    }

    /// Ports `tools_io_isolate`'s `list_entries` (server-tools.cpp:575 @620d2225c).
    fn listEntries(self: *const Isolate, a: Allocator, base: []const u8, max_depth: c_int, kind: ListKind) !ListResult {
        var out: ListResult = .{};

        const abs_base = try self.resolve(a, base);
        if (!try self.isDirectory(a, base)) {
            out.err = "path does not exist or is not a directory";
            return out;
        }

        // git ls-files cannot list directories; use the walker when they are requested
        if (kind == .files) {
            const res = try self.exec(a, &.{ "sh", "-c", "cd \"$1\" && git ls-files --cached --others --exclude-standard", "_", abs_base }, git_ls_files_max_output, true);
            if (res.exit_code == 0 and !res.timed_out) {
                for ((try splitLines(a, res.output, false)).items) |rel| {
                    if (max_depth > 0 and entryDepth(rel) > max_depth) continue;
                    try out.entries.append(a, .{ .rel = rel, .is_dir = false });
                }
                return out;
            }
        }

        if (kind == .dirs or kind == .all) {
            for ((try self.findEntries(a, abs_base, max_depth, true, &out.truncated)).items) |rel| {
                try out.entries.append(a, .{ .rel = rel, .is_dir = true });
            }
        }
        if (kind == .files or kind == .all) {
            for ((try self.findEntries(a, abs_base, max_depth, false, &out.truncated)).items) |rel| {
                try out.entries.append(a, .{ .rel = rel, .is_dir = false });
            }
        }
        return out;
    }

    /// Ports `tools_io_isolate`'s `run` (server-tools.cpp:615 @620d2225c): wrapped in an
    /// in-isolate `timeout`, since killing the host-side client does not
    /// kill the process tree inside.
    fn run(self: *const Isolate, a: Allocator, args: []const []const u8, max_output: usize, timeout_secs: c_int, on_chunk: ?OnChunk) !ExecResult {
        var inner: std.ArrayList([]const u8) = .empty;
        try inner.append(a, "timeout");
        try inner.append(a, try std.fmt.allocPrint(a, "{d}s", .{timeout_secs}));
        try inner.appendSlice(a, args);
        // small buffer over timeout_secs so the in-isolate `timeout` has a chance to exit cleanly
        // before the host-side supervisory timeout forcibly kills the client
        const argv = try self.buildArgv(a, try self.withCwd(a, inner.items), true);
        return runSubprocess(a, argv, max_output, timeout_secs +% 5, on_chunk, true, "", null);
    }

    /// Ports `tools_io_container`'s `build_argv` (server-tools.cpp:724 @620d2225c), and
    /// `tools_io_ssh`'s `build_argv` (server-tools.cpp:758 @620d2225c): the host-side
    /// argv that runs `inner` in the isolate.
    fn buildArgv(self: *const Isolate, a: Allocator, inner: []const []const u8, needs_stdin: bool) ![]const []const u8 {
        var argv: std.ArrayList([]const u8) = .empty;
        switch (self.transport) {
            .container => |c| {
                try argv.appendSlice(a, &.{ c.bin, "exec" });
                if (needs_stdin) try argv.append(a, "-i");
                try argv.append(a, c.container_id);
                try argv.appendSlice(a, inner);
            },
            .ssh => |s| {
                // the remote shell re-parses the command line, so `inner` travels as one quoted word
                try argv.appendSlice(a, &sshArgv());
                if (!needs_stdin) try argv.append(a, "-n");
                try argv.append(a, s.target);
                try argv.append(a, try shellQuoteJoin(a, inner));
            },
        }
        return argv.items;
    }

    /// Ports `tools_io_isolate`'s `shell_quote_join` (server-tools.cpp:635 @620d2225c): one
    /// string a POSIX shell re-parses into exactly `argv`.
    fn shellQuoteJoin(a: Allocator, argv: []const []const u8) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (argv) |arg| {
            if (out.items.len != 0) try out.append(a, ' ');
            try out.append(a, '\'');
            for (arg) |ch| {
                // a single quote cannot be escaped inside single quotes: close, escape, reopen
                if (ch == '\'') try out.appendSlice(a, "'\\''") else try out.append(a, ch);
            }
            try out.append(a, '\'');
        }
        return out.items;
    }

    /// Ports `tools_io_isolate`'s `with_cwd` (server-tools.cpp:655 @620d2225c): the working
    /// directory set in the command itself.
    fn withCwd(self: *const Isolate, a: Allocator, inner: []const []const u8) ![]const []const u8 {
        if (self.cwd.len == 0) return inner;
        // 127 is what a shell reports for a command it could not run
        var out: std.ArrayList([]const u8) = .empty;
        try out.appendSlice(a, &.{ "sh", "-c", "cd \"$1\" || exit 127; shift; exec \"$@\"", "_", self.cwd });
        try out.appendSlice(a, inner);
        return out.items;
    }

    /// Ports `tools_io_isolate`'s `exec` (server-tools.cpp:665 @620d2225c).
    fn exec(self: *const Isolate, a: Allocator, inner: []const []const u8, max_output: usize, combine_stderr: bool) !ExecResult {
        return runSubprocess(a, try self.buildArgv(a, inner, false), max_output, isolate_exec_timeout, null, combine_stderr, "", null);
    }

    /// Ports `tools_io_isolate`'s `shell_run` (server-tools.cpp:671 @620d2225c).
    fn shellRun(self: *const Isolate, a: Allocator, inner: []const []const u8) !bool {
        const res = try self.exec(a, inner, 4096, true);
        return res.exit_code == 0 and !res.timed_out;
    }

    /// Ports `tools_io_isolate`'s `shell_test` (server-tools.cpp:676 @620d2225c).
    fn shellTest(self: *const Isolate, a: Allocator, flag: []const u8, path: []const u8) !bool {
        return self.shellRun(a, &.{ "sh", "-c", try std.mem.concat(a, u8, &.{ "[ ", flag, " \"$1\" ]" }), "_", path });
    }

    /// Ports `tools_io_isolate`'s `split_lines` (server-tools.cpp:680 @620d2225c).
    fn splitLines(a: Allocator, text: []const u8, strip_dot_slash: bool) !std.ArrayList([]const u8) {
        var result: std.ArrayList([]const u8) = .empty;
        var it = GetLines.init(text);
        while (it.next()) |raw| {
            var line = raw;
            if (line.len != 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
            if (line.len == 0) continue;
            if (strip_dot_slash and std.mem.startsWith(u8, line, "./")) line = line[2..];
            const l = try a.dupe(u8, line);
            std.mem.replaceScalar(u8, l, '\\', '/');
            try result.append(a, l);
        }
        return result;
    }

    /// Ports `tools_io_isolate`'s `find_entries` (server-tools.cpp:696 @620d2225c): one
    /// `find` pass; junk directories listed but never entered.
    fn findEntries(self: *const Isolate, a: Allocator, abs_base: []const u8, max_depth: c_int, dirs: bool, truncated: *bool) !std.ArrayList([]const u8) {
        var prune_expr: std.ArrayList(u8) = .empty;
        for (junk_dir_names) |n| {
            if (prune_expr.items.len != 0) try prune_expr.appendSlice(a, " -o ");
            try prune_expr.appendSlice(a, "-name ");
            try prune_expr.appendSlice(a, n);
        }

        var cmd: std.ArrayList(u8) = .empty;
        try cmd.appendSlice(a, "cd \"$1\" && find . -mindepth 1");
        if (max_depth > 0) try cmd.print(a, " -maxdepth {d}", .{max_depth});
        try cmd.print(a, " \\( {s} \\) -prune", .{prune_expr.items});
        try cmd.appendSlice(a, if (dirs) " -print -o -type d -print" else " -o -type f -print");

        const res = try self.exec(a, &.{ "sh", "-c", cmd.items, "_", abs_base }, git_ls_files_max_output, true);
        truncated.* = truncated.* or res.timed_out;
        return splitLines(a, res.output, true);
    }

    /// Ports `tools_io_ssh`'s `ssh_argv` (server-tools.cpp:774 @620d2225c): no console, so
    /// key-based auth only and a host key trusted beforehand.
    fn sshArgv() [9][]const u8 {
        return .{ "ssh", "-o", "BatchMode=yes", "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no", "-o", "StrictHostKeyChecking=yes" };
    }
};

/// Ports `tools_io_ssh`'s `is_valid_target` (server-tools.cpp:748 @620d2225c): a target
/// starting with `-` would be an ssh option.
fn sshIsValidTarget(target: []const u8) bool {
    if (target.len == 0 or target[0] == '-') return false;
    for (target) |ch| {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '-' or ch == '_' or ch == '@')) return false;
    }
    return true;
}

/// Ports `container_runtime_spec` (server-tools.cpp:786 @620d2225c):
/// `"<engine>:<image>"` spawns a container, `"<engine>-container:<id>"`
/// attaches to one.
const ContainerSpec = struct {
    bin: []const u8 = "",
    arg: []const u8 = "", // image name when spawning, container id when attaching
    attach: bool = false,

    /// Ports `container_runtime_spec`'s `parse` (server-tools.cpp:791 @620d2225c).
    fn parse(spec: []const u8) ?ContainerSpec {
        // docker and podman take the same verbs, hence a single implementation
        for ([_][]const u8{ "docker", "podman" }) |bin| {
            if (std.mem.startsWith(u8, spec, bin) and std.mem.startsWith(u8, spec[bin.len..], "-container:")) {
                return .{ .bin = bin, .arg = spec[bin.len + "-container:".len ..], .attach = true };
            }
            if (std.mem.startsWith(u8, spec, bin) and std.mem.startsWith(u8, spec[bin.len..], ":")) {
                return .{ .bin = bin, .arg = spec[bin.len + 1 ..], .attach = false };
            }
        }
        return null;
    }

    /// Ports `container_runtime_spec`'s `is_valid_id` (server-tools.cpp:811 @620d2225c): an
    /// id starting with `-` would be an engine option.
    fn isValidId(id: []const u8) bool {
        if (id.len == 0 or !std.ascii.isAlphanumeric(id[0])) return false;
        for (id) |ch| {
            if (!(std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '-' or ch == '_')) return false;
        }
        return true;
    }
};

/// Ports `tools_io` (server-tools.cpp:147 @620d2225c): where a tool's file and
/// process operations run.
const Io = union(enum) {
    basic: Basic,
    isolate: Isolate,

    fn resolve(self: *const Io, a: Allocator, path: []const u8) ![]const u8 {
        return switch (self.*) {
            inline else => |*x| x.resolve(a, path),
        };
    }
    fn isDirectory(self: *const Io, a: Allocator, path: []const u8) !bool {
        return switch (self.*) {
            inline else => |*x| x.isDirectory(a, path),
        };
    }
    fn isRegularFile(self: *const Io, a: Allocator, path: []const u8) !bool {
        return switch (self.*) {
            inline else => |*x| x.isRegularFile(a, path),
        };
    }
    fn fileSize(self: *const Io, a: Allocator, path: []const u8, out: *u64) !bool {
        return switch (self.*) {
            inline else => |*x| x.fileSize(a, path, out),
        };
    }
    fn readFile(self: *const Io, a: Allocator, path: []const u8) !?[]const u8 {
        return switch (self.*) {
            inline else => |*x| x.readFile(a, path),
        };
    }
    fn writeFile(self: *const Io, a: Allocator, path: []const u8, content: []const u8) !bool {
        return switch (self.*) {
            inline else => |*x| x.writeFile(a, path, content),
        };
    }
    fn listEntries(self: *const Io, a: Allocator, base: []const u8, max_depth: c_int, kind: ListKind) !ListResult {
        return switch (self.*) {
            inline else => |*x| x.listEntries(a, base, max_depth, kind),
        };
    }
    fn run(self: *const Io, a: Allocator, args: []const []const u8, max_output: usize, timeout_secs: c_int, on_chunk: ?OnChunk) !ExecResult {
        return switch (self.*) {
            inline else => |*x| x.run(a, args, max_output, timeout_secs, on_chunk),
        };
    }
};

/// Ports `make_tools_io` (server-tools.cpp:821 @620d2225c): the host for an empty
/// `runtime`, an attached container or an ssh host otherwise; anything
/// else throws rather than fall back to the host.
fn makeToolsIo(a: Allocator, params: *const Value) !Io {
    const cwd = try a.dupe(u8, common.jsonValue(params, "cwd", []const u8, ""));
    const runtime = try a.dupe(u8, common.jsonValue(params, "runtime", []const u8, ""));
    if (runtime.len == 0) {
        // an empty runtime runs the tools on the host
        return .{ .basic = .{ .cwd = cwd } };
    }
    if (ContainerSpec.parse(runtime)) |container| {
        // spawning belongs to the runtime that owns the container, a tool call only attaches
        if (!container.attach) return throw(error.RuntimeError, "tool runtime must name a running container: {s}", .{runtime});
        if (!ContainerSpec.isValidId(container.arg)) return throw(error.RuntimeError, "invalid container id: {s}", .{container.arg});
        return .{ .isolate = .{ .cwd = cwd, .transport = .{ .container = .{ .bin = container.bin, .container_id = container.arg } } } };
    }
    const ssh_prefix = "ssh:";
    if (std.mem.startsWith(u8, runtime, ssh_prefix)) {
        const target = runtime[ssh_prefix.len..];
        if (!sshIsValidTarget(target)) return throw(error.RuntimeError, "invalid ssh target: {s}", .{target});
        return .{ .isolate = .{ .cwd = cwd, .transport = .{ .ssh = .{ .target = target } } } };
    }
    // do not fall back to the host, the caller asked for an isolate
    return throw(error.RuntimeError, "unknown tool runtime: {s}", .{runtime});
}

// -----------------------------------------------------------------------------
// glob_match (common/common.cpp)

/// A byte as the C's (plain, signed on Darwin) `char` compares it.
fn cch(b: u8) c_char {
    return @bitCast(b);
}

/// Ports `glob_class_match` (common/common.cpp:703 @620d2225c): whether `c` is in
/// the class `pattern[0..class_end]` (after its `[`).
fn globClassMatch(ch: u8, pattern: []const u8, class_end: usize) bool {
    var class_start: usize = 0;
    var negated = false;

    if (pattern[class_start] == '!') {
        negated = true;
        class_start += 1;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if (pattern[class_start] == ']' or pattern[class_start] == '-') {
        if (class_start < class_end and pattern[class_start] == ch) return !negated;
        class_start += 1;
    }

    var matched = false;
    while (class_start < class_end) {
        if (class_start + 2 < class_end and pattern[class_start + 1] == '-' and pattern[class_start + 2] != ']') {
            const start_char = cch(pattern[class_start]);
            const end_char = cch(pattern[class_start + 2]);
            if (cch(ch) >= start_char and cch(ch) <= end_char) {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if (pattern[class_start] == ch) {
                matched = true;
                break;
            }
            class_start += 1;
        }
    }
    return if (negated) !matched else matched;
}

/// `pattern[i]` and `str[i]` with the C string's terminating NUL.
fn at0(s: []const u8, i: usize) u8 {
    return if (i < s.len) s[i] else 0;
}

/// Ports `glob_match` (common/common.cpp:744 @620d2225c): `*` matches non-`/`
/// characters, `**` anything including `/`, `[]` a character class. Over
/// C strings, so the first NUL ends either.
fn globMatch(pattern: []const u8, str: []const u8) bool {
    const p0 = at0(pattern, 0);
    if (p0 == 0) return at0(str, 0) == 0;
    if (p0 == '*' and at0(pattern, 1) == '*') {
        const p = pattern[2..];
        if (globMatch(p, str)) return true;
        if (at0(str, 0) != 0) return globMatch(pattern, str[1..]);
        return false;
    }
    if (p0 == '*') {
        const p = pattern[1..];
        var s = str;
        while (at0(s, 0) != 0 and s[0] != '/') : (s = s[1..]) {
            if (globMatch(p, s)) return true;
        }
        return globMatch(p, s);
    }
    if (p0 == '?' and at0(str, 0) != 0 and str[0] != '/') return globMatch(pattern[1..], str[1..]);
    if (p0 == '[') {
        var class_end: usize = 1;
        // If first character after '[' is ']' or '-', treat it as literal
        if (at0(pattern, class_end) == ']' or at0(pattern, class_end) == '-') class_end += 1;
        while (at0(pattern, class_end) != 0 and pattern[class_end] != ']') class_end += 1;
        if (at0(pattern, class_end) == ']') {
            if (at0(str, 0) == 0) return false;
            const matched = globClassMatch(str[0], pattern[1..], class_end - 1);
            return matched and globMatch(pattern[class_end + 1 ..], str[1..]);
        } else {
            if (at0(str, 0) == '[') return globMatch(pattern[1..], str[1..]);
            return false;
        }
    }
    if (p0 == at0(str, 0)) return globMatch(pattern[1..], str[1..]);
    return false;
}

/// Ports `path_glob_match` (server-tools.cpp:852 @620d2225c): no `/` in the pattern
/// matches the basename at any depth; otherwise the full relative path,
/// anchored with `**/` unless it already is.
fn pathGlobMatch(a: Allocator, pattern: []const u8, rel_path: []const u8) !bool {
    if (std.mem.findScalar(u8, pattern, '/') == null) return globMatch(pattern, pathFilename(rel_path));
    if (std.mem.eql(u8, pattern, "**") or std.mem.startsWith(u8, pattern, "**/") or std.mem.startsWith(u8, pattern, "/")) {
        return globMatch(pattern, rel_path);
    }
    return globMatch(try std.mem.concat(a, u8, &.{ "**/", pattern }), rel_path);
}

// -----------------------------------------------------------------------------
// the tools

/// Ports `SERVER_TOOL_READ_FILE_MAX_SIZE` (server-tools.cpp:866 @620d2225c).
const read_file_max_size: usize = 16 * 1024; // 16 KB
/// Ports `SERVER_TOOL_READ_FILE_MAX_SIZE_BASE64` (server-tools.cpp:867 @620d2225c).
const read_file_max_size_base64: usize = 32 * 1024 * 1024; // 32 MB
/// Ports `SERVER_TOOL_FILE_SEARCH_MAX_RESULTS` (server-tools.cpp:972 @620d2225c).
const file_search_max_results: c_int = 100;
/// Ports `SERVER_TOOL_GREP_SEARCH_MAX_RESULTS` (server-tools.cpp:1090 @620d2225c).
const grep_search_max_results: usize = 100;
/// Ports `SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE` (server-tools.cpp:1242 @620d2225c).
const exec_shell_command_max_output_size: usize = 16 * 1024; // 16 KB
/// Ports `SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT` (server-tools.cpp:1243 @620d2225c).
const exec_shell_command_max_timeout: c_int = 60; // seconds
/// Ports `SERVER_TOOL_GET_INFO_MAX_OUTPUT` (server-tools.cpp:1698 @620d2225c).
const get_info_max_output: usize = 4096;
/// Ports `SERVER_TOOL_GET_INFO_TIMEOUT` (server-tools.cpp:1699 @620d2225c).
const get_info_timeout: c_int = 5; // seconds

/// The definitions, as JSON text in the C++ initializer lists' order (an
/// ordered object keeps insertion order, so parsing the text builds the
/// same value).
const defs = struct {
    /// Ports `server_tool_read_file`'s `get_definition` (server-tools.cpp:877 @620d2225c).
    const read_file =
        \\{"type":"function","function":{"name":"read_file","description":"Read the contents of a file. Optionally specify a 1-based line range. If append_loc is true, each line is prefixed with its line number (e.g. \"1→...\").","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file"},"start_line":{"type":"integer","description":"First line to read, 1-based (default: 1)"},"end_line":{"type":"integer","description":"Last line to read, 1-based inclusive (default: end of file)"},"append_loc":{"type":"boolean","description":"Prefix each line with its line number"}},"required":["path"]}}}
    ;
    /// Ports `server_tool_file_glob_search`'s `get_definition` (server-tools.cpp:985 @620d2225c).
    const file_glob_search = std.fmt.comptimePrint(
        \\{{"type":"function","function":{{"name":"file_glob_search","description":"Recursively search for files matching a glob pattern under a directory. Automatically skips files ignored by .gitignore (when the directory is inside a git repo) and common junk directories (.git, node_modules, build, dist, etc.) otherwise. A pattern with no '/' (e.g. \"*.cpp\") matches the file's basename at any depth. A pattern containing '/' matches the full relative path; unless already anchored with \"**/\" or a leading '/', it is automatically prefixed with \"**/\". Use type=\"dir\" or \"all\" to also list directories; directory entries are suffixed with '/' in the output. Note: directory listings do not apply .gitignore filtering.","parameters":{{"type":"object","properties":{{"path":{{"type":"string","description":"Base directory to search in"}},"include":{{"type":"string","description":"Glob pattern for files to include (e.g. \"*.cpp\" or \"src/**/*.cpp\"). Default: **"}},"exclude":{{"type":"string","description":"Glob pattern for files to exclude"}},"type":{{"type":"string","description":"Entry type to return: \"file\" (default), \"dir\" or \"all\""}},"max_depth":{{"type":"integer","description":"Maximum depth to descend into subdirectories (default: 0 = unlimited; 1 = direct children only)"}},"limit":{{"type":"integer","description":"Maximum number of results to return, capped at {d} (default {d})"}}}},"required":["path"]}}}}}}
    , .{ file_search_max_results, file_search_max_results });
    /// Ports `server_tool_grep_search`'s `get_definition` (server-tools.cpp:1100 @620d2225c).
    const grep_search =
        \\{"type":"function","function":{"name":"grep_search","description":"Search for a pattern in files under a path. Returns matching lines with file paths (and, unless searching a single file, paths relative to the given directory). Automatically skips files ignored by .gitignore (when the directory is inside a git repo) and common junk directories (.git, node_modules, build, dist, etc.) otherwise. include/exclude: a pattern with no '/' matches the basename at any depth; a pattern containing '/' matches the full relative path (auto-anchored with \"**/\" unless already anchored).","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File or directory to search in"},"pattern":{"type":"string","description":"Pattern to search for (regular expression unless literal is true)"},"include":{"type":"string","description":"Glob pattern to filter files (default: **)"},"exclude":{"type":"string","description":"Glob pattern to exclude files"},"return_line_numbers":{"type":"boolean","description":"If true, include line numbers in results"},"literal":{"type":"boolean","description":"Treat pattern as a literal string instead of a regular expression (default: false)"},"ignore_case":{"type":"boolean","description":"Case-insensitive search (default: false)"},"context_lines":{"type":"integer","description":"Number of lines of context to show before and after each match (default: 0)"}},"required":["path","pattern"]}}}
    ;
    /// Ports `server_tool_exec_shell_command`'s `get_definition` (server-tools.cpp:1254 @620d2225c).
    const exec_shell_command = std.fmt.comptimePrint(
        \\{{"type":"function","function":{{"name":"exec_shell_command","description":"Execute a shell command and return its output (stdout and stderr combined).","parameters":{{"type":"object","properties":{{"command":{{"type":"string","description":"Shell command to execute"}},"timeout":{{"type":"integer","description":"Timeout in seconds (default 10, max {d})"}},"max_output_size":{{"type":"integer","description":"Maximum output size in bytes (default {d})"}}}},"required":["command"]}}}}}}
    , .{ exec_shell_command_max_timeout, exec_shell_command_max_output_size });
    /// Ports `server_tool_write_file`'s `get_definition` (server-tools.cpp:1332 @620d2225c).
    const write_file =
        \\{"type":"function","function":{"name":"write_file","description":"Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path of the file to write"},"content":{"type":"string","description":"Content to write"}},"required":["path","content"]}}}
    ;
    /// Ports `server_tool_edit_file`'s `get_definition` (server-tools.cpp:1375 @620d2225c).
    const edit_file =
        \\{"type":"function","function":{"name":"edit_file","description":"Edit a file using exact text replacement. Each edits[].old_text must be unique in the file and is matched against the original content, not incrementally. Merge nearby changes into one edit instead of overlapping edits. Use write_file to replace the whole file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to edit"},"edits":{"type":"array","description":"One or more exact text replacements to apply","items":{"type":"object","properties":{"old_text":{"type":"string","description":"Exact text to find; must be unique in the file and must not overlap with other edits"},"new_text":{"type":"string","description":"Text to replace old_text with"}},"required":["old_text","new_text"]}}},"required":["path","edits"]}}}
    ;
    /// Ports `server_tool_get_info`'s `get_definition` (server-tools.cpp:1709 @620d2225c).
    const get_info =
        \\{"type":"function","function":{"name":"get_info","description":"Returns runtime info: the OS name/version and the current working directory","parameters":{"type":"object","properties":{}}}}
    ;
};

/// Which tool a `Tool` is: the seven built in, or one from an MCP server.
pub const Kind = enum { read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_info, mcp };

/// Ports `server_tool`'s `stream` (server-tools.h:23 @620d2225c): where a
/// streaming tool pushes its output.
pub const Stream = struct {
    qr: *ResultQueue,
    id: i32,
    alive: ?mcp.StopFn, // true while the client is there

    /// Ports `server_tool`'s `stream`'s `push` (server-tools.cpp:1776 @620d2225c).
    fn push(self: *const Stream, chunk: []const u8) !void {
        if (chunk.len == 0) return;
        const r = try StreamResult.create();
        r.id = self.id;
        r.chunk = try gpa.dupe(u8, chunk);
        try self.qr.send(r);
    }

    fn isAlive(self: *const Stream) bool {
        return if (self.alive) |f| f.call() else true;
    }
};

/// Ports `server_tool` (server-tools.h:12 @620d2225c) and its subclasses, one
/// per `Kind`; `server_mcp_tool` (server-tools.cpp:1802 @620d2225c) is `.mcp`.
pub const Tool = struct {
    name: []const u8,
    display_name: []const u8,
    permission_write: bool = false,
    support_stream: bool = false, // if true, output can be streamed
    uses_cwd: bool = false, // if true, the tool resolves paths and runs against the working directory
    kind: Kind,

    // server_mcp_tool
    server_name: []const u8 = "",
    tool_name: []const u8 = "",
    def: ?mcp.ToolDef = null,
    mcp_mgr: ?*mcp.Mcp = null,

    /// Ports `server_tool`'s `type` (server-tools.h:21 @620d2225c), and
    /// `server_mcp_tool`'s `type` (server-tools.cpp:1820 @620d2225c).
    fn typeName(self: *const Tool) []const u8 {
        return if (self.kind == .mcp) "mcp" else "server";
    }

    /// Ports each tool's `get_definition`, `server_mcp_tool`'s
    /// `get_definition` (server-tools.cpp:1822 @620d2225c) among them.
    fn getDefinition(self: *const Tool, a: Allocator) !Value {
        const text = switch (self.kind) {
            .read_file => defs.read_file,
            .file_glob_search => defs.file_glob_search,
            .grep_search => defs.grep_search,
            .exec_shell_command => defs.exec_shell_command,
            .write_file => defs.write_file,
            .edit_file => defs.edit_file,
            .get_info => defs.get_info,
            .mcp => {
                var schema = try self.def.?.input_schema.clone(a);
                if (schema.isNull() or !schema.isObject()) schema = Value.emptyObject();
                var func = common.Obj.init(a);
                try func.put("name", self.name);
                try func.put("description", self.def.?.description);
                try func.put("parameters", schema);
                var o = common.Obj.init(a);
                try o.put("type", "function");
                try o.put("function", func.done());
                return o.done();
            },
        };
        return json.parse(a, text);
    }

    /// Ports `server_tool`'s `to_json` (server-tools.cpp:75 @620d2225c).
    fn toJson(self: *const Tool, a: Allocator) !Value {
        var perms = common.Obj.init(a);
        try perms.put("write", self.permission_write);
        var o = common.Obj.init(a);
        try o.put("display_name", self.display_name);
        try o.put("tool", self.name);
        try o.put("type", self.typeName());
        try o.put("permissions", perms.done());
        try o.put("uses_cwd", self.uses_cwd);
        try o.put("definition", try self.getDefinition(a));
        return o.done();
    }

    /// Ports each tool's `invoke`: the result, or null when it streamed.
    ///
    /// Return: allocated by `a`.
    fn invoke(self: *const Tool, a: Allocator, params: *const Value, st: ?*const Stream) !Value {
        return switch (self.kind) {
            .read_file => invokeReadFile(a, params),
            .file_glob_search => invokeFileGlobSearch(a, params),
            .grep_search => invokeGrepSearch(a, params),
            .exec_shell_command => invokeExecShellCommand(a, params, st),
            .write_file => invokeWriteFile(a, params),
            .edit_file => invokeEditFile(a, params),
            .get_info => invokeGetInfo(a, params),
            // Ports `server_mcp_tool`'s `invoke` (server-tools.cpp:1837 @620d2225c).
            .mcp => self.mcp_mgr.?.callTool(a, self.server_name, self.tool_name, params, null),
        };
    }
};

/// `{"error": msg}`.
fn errorObj(a: Allocator, msg: []const u8) !Value {
    var o = common.Obj.init(a);
    try o.put("error", msg);
    return o.done();
}

/// `{"plain_text_response": text}`.
fn plainText(a: Allocator, text: []const u8) !Value {
    var o = common.Obj.init(a);
    try o.put("plain_text_response", text);
    return o.done();
}

/// `params.at(key).get<std::string>()`.
fn atString(params: *const Value, key: []const u8) ![]const u8 {
    return (try params.at(key)).get([]const u8);
}

/// Ports `server_tool_read_file`'s `invoke` (server-tools.cpp:898 @620d2225c): a
/// line range, optionally numbered; or, asked by the `x-resp-type` header,
/// the whole file in base64.
fn invokeReadFile(a: Allocator, params: *const Value) !Value {
    const path = try atString(params, "path");
    const start_line = common.jsonValue(params, "start_line", c_int, 1);
    const end_line = common.jsonValue(params, "end_line", c_int, -1); // -1 = no limit
    const append_loc = common.jsonValue(params, "append_loc", bool, false);
    // comes from the x-resp-type header, the model cannot ask for it
    const as_base64 = std.mem.eql(u8, common.jsonValue(params, "resp_type", []const u8, ""), "base64");

    const io = try makeToolsIo(a, params);

    var file_size: u64 = 0;
    if (!try io.fileSize(a, path, &file_size)) {
        return errorObj(a, try std.mem.concat(a, u8, &.{ "cannot stat file: ", path }));
    }

    if (as_base64) {
        if (file_size > read_file_max_size_base64) {
            return errorObj(a, try std.fmt.allocPrint(a, "file too large ({d} bytes, max {d})", .{ file_size, read_file_max_size_base64 }));
        }
        const content = (try io.readFile(a, path)) orelse return errorObj(a, try std.mem.concat(a, u8, &.{ "failed to open file: ", path }));
        const enc = std.base64.standard.Encoder;
        const out = try a.alloc(u8, enc.calcSize(content.len));
        _ = enc.encode(out, content);
        var o = common.Obj.init(a);
        try o.put("base64", out);
        try o.put("size_bytes", @as(usize, content.len));
        return o.done();
    }

    if (file_size > read_file_max_size and end_line == -1) {
        return errorObj(a, try std.fmt.allocPrint(a, "file too large ({d} bytes, max {d}). Use start_line/end_line to read a portion.", .{ file_size, read_file_max_size }));
    }

    const content = (try io.readFile(a, path)) orelse return errorObj(a, try std.mem.concat(a, u8, &.{ "failed to open file: ", path }));

    var result: std.ArrayList(u8) = .empty;
    var lineno: c_int = 0;
    var it = GetLines.init(content);
    while (it.next()) |line| {
        lineno +%= 1;
        if (lineno < start_line) continue;
        if (end_line != -1 and lineno > end_line) break;

        const out_line = if (append_loc)
            try std.fmt.allocPrint(a, "{d}\u{2192}{s}\n", .{ lineno, line })
        else
            try std.mem.concat(a, u8, &.{ line, "\n" });

        if (result.items.len + out_line.len > read_file_max_size) {
            try result.appendSlice(a, "[output truncated]");
            break;
        }
        try result.appendSlice(a, out_line);
    }

    return plainText(a, result.items);
}

/// Ports `server_tool_file_glob_search`'s `invoke` (server-tools.cpp:1015 @620d2225c).
fn invokeFileGlobSearch(a: Allocator, params: *const Value) !Value {
    const io = try makeToolsIo(a, params);

    const path = try atString(params, "path");

    const base = try io.resolve(a, path);
    const include = common.jsonValue(params, "include", []const u8, "**");
    const exclude = common.jsonValue(params, "exclude", []const u8, "");
    const ty = common.jsonValue(params, "type", []const u8, "file");
    const max_depth = @max(0, common.jsonValue(params, "max_depth", c_int, 0));
    const limit_req = common.jsonValue(params, "limit", c_int, file_search_max_results);
    if (limit_req < 1) {
        return errorObj(a, try std.fmt.allocPrint(a, "invalid limit: {d} (expected 1 or more)", .{limit_req}));
    }
    const limit = @min(limit_req, file_search_max_results);

    const kind: ListKind = if (std.mem.eql(u8, ty, "file"))
        .files
    else if (std.mem.eql(u8, ty, "dir"))
        .dirs
    else if (std.mem.eql(u8, ty, "all"))
        .all
    else
        return errorObj(a, try std.mem.concat(a, u8, &.{ "invalid type: ", ty, " (expected \"file\", \"dir\" or \"all\")" }));

    const listing = try io.listEntries(a, base, max_depth, kind);
    if (listing.err.len != 0) {
        return errorObj(a, try std.mem.concat(a, u8, &.{ listing.err, ": ", path }));
    }

    var matches: std.ArrayList(ListEntry) = .empty;
    for (listing.entries.items) |entry| {
        if (!try pathGlobMatch(a, include, entry.rel)) continue;
        if (exclude.len != 0 and try pathGlobMatch(a, exclude, entry.rel)) continue;
        try matches.append(a, entry);
    }

    const total = matches.items.len;
    const shown = @min(total, @as(usize, @intCast(limit)));

    var output_text: std.ArrayList(u8) = .empty;
    var entries_json = Value.emptyArray();
    for (matches.items[0..shown]) |m| {
        try output_text.print(a, "{s}{s}\n", .{ m.rel, if (m.is_dir) "/" else "" });
        var e = common.Obj.init(a);
        try e.put("path", m.rel);
        try e.put("type", if (m.is_dir) "dir" else "file");
        try entries_json.pushBack(a, e.done());
    }

    try output_text.print(a, "\n---\nTotal matches: {d}\n", .{total});
    if (total > shown) {
        try output_text.print(a, "[{d} results limit reached ({d} total matches). Refine the glob pattern to narrow the search.]\n", .{ shown, total });
    }
    if (listing.truncated) {
        try output_text.appendSlice(a, "[results truncated: time budget or unreadable directory]\n");
    }

    // `base` is always absolute (resolve falls back to the server cwd), so
    // API clients (e.g. the web UI picker) can join the relative entries
    // into absolute paths.
    var o = common.Obj.init(a);
    try o.put("plain_text_response", output_text.items);
    try o.put("entries", entries_json);
    try o.put("base", base);
    return o.done();
}

/// Ports `server_tool_grep_search`'s `invoke` (server-tools.cpp:1130 @620d2225c): an
/// ECMAScript regex (or a literal) over every line of the files under a
/// path, with context lines.
fn invokeGrepSearch(a: Allocator, params: *const Value) !Value {
    const path = try atString(params, "path");
    const pat_str = try atString(params, "pattern");
    const include = common.jsonValue(params, "include", []const u8, "**");
    const exclude = common.jsonValue(params, "exclude", []const u8, "");
    const show_lineno = common.jsonValue(params, "return_line_numbers", bool, false);
    const literal = common.jsonValue(params, "literal", bool, false);
    const ignore_case = common.jsonValue(params, "ignore_case", bool, false);
    const ctx_lines = @max(0, common.jsonValue(params, "context_lines", c_int, 0));

    var pattern_src: []const u8 = pat_str;
    if (literal) {
        const specials = "\\^$.|?*+()[]{}";
        var escaped: std.ArrayList(u8) = .empty;
        for (pat_str) |ch| {
            if (std.mem.findScalar(u8, specials, ch) != null) try escaped.append(a, '\\');
            try escaped.append(a, ch);
        }
        pattern_src = escaped.items;
    }

    var pattern = regex.Regex.compileWith(pattern_src, .{ .icase = ignore_case }) catch |e| {
        return errorObj(a, try std.mem.concat(a, u8, &.{ "invalid regex: ", regex.errorText(e) }));
    };
    defer pattern.deinit();

    const io = try makeToolsIo(a, params);

    // collect (absolute_path, display_path) pairs to search
    const FileEntry = struct { abs: []const u8, display: []const u8 };
    var files: std.ArrayList(FileEntry) = .empty;

    const abs_path = try io.resolve(a, path);
    if (try io.isRegularFile(a, abs_path)) {
        try files.append(a, .{ .abs = abs_path, .display = path });
    } else if (try io.isDirectory(a, abs_path)) {
        const listing = try io.listEntries(a, abs_path, 0, .files);
        if (listing.err.len != 0) {
            return errorObj(a, try std.mem.concat(a, u8, &.{ listing.err, ": ", path }));
        }
        for (listing.entries.items) |entry| {
            if (!try pathGlobMatch(a, include, entry.rel)) continue;
            if (exclude.len != 0 and try pathGlobMatch(a, exclude, entry.rel)) continue;
            try files.append(a, .{ .abs = try pathJoin(a, abs_path, entry.rel), .display = entry.rel });
        }
    } else {
        return errorObj(a, try std.mem.concat(a, u8, &.{ "path does not exist: ", path }));
    }

    var output_text: std.ArrayList(u8) = .empty;
    var total: usize = 0;
    var limit_reached = false;
    const show_num = show_lineno or ctx_lines > 0;

    var m = regex.Match.init();
    defer m.deinit();
    for (files.items) |file_entry| {
        if (limit_reached) break;
        const content = (try io.readFile(a, file_entry.abs)) orelse continue;
        var lines: std.ArrayList([]const u8) = .empty;
        var it = GetLines.init(content);
        while (it.next()) |line| try lines.append(a, line);

        for (lines.items, 0..) |line, i| {
            if (total >= grep_search_max_results) {
                limit_reached = true;
                break;
            }
            const found = pattern.trySearch(line, &m) catch |e| return throw(error.RuntimeError, "{s}", .{regex.errorText(e)});
            if (!found) continue;

            const ii: i64 = @intCast(i);
            const ctx_start: i64 = if (ctx_lines > 0) @max(0, ii - ctx_lines) else ii;
            const ctx_end: i64 = if (ctx_lines > 0) @min(@as(i64, @intCast(lines.items.len)) - 1, ii + ctx_lines) else ii;

            var j = ctx_start;
            while (j <= ctx_end) : (j += 1) {
                const is_match = j == ii;
                try output_text.appendSlice(a, file_entry.display);
                try output_text.append(a, if (is_match) ':' else '-');
                if (show_num) {
                    try output_text.print(a, "{d}", .{j + 1});
                    try output_text.append(a, if (is_match) ':' else '-');
                }
                try output_text.appendSlice(a, lines.items[@intCast(j)]);
                try output_text.append(a, '\n');
            }
            if (ctx_lines > 0) try output_text.appendSlice(a, "--\n");
            total += 1;
        }
    }

    try output_text.print(a, "\n---\nTotal matches: {d}\n", .{total});
    if (limit_reached) {
        try output_text.print(a, "[{d} matches limit reached. Narrow the path/pattern/include to see more.]\n", .{grep_search_max_results});
    }

    return plainText(a, output_text.items);
}

/// Ports `server_tool_exec_shell_command`'s `invoke` (server-tools.cpp:1273 @620d2225c):
/// `sh -c` with a timeout and an output cap; with a stream, each chunk as
/// it comes and the exit code last.
fn invokeExecShellCommand(a: Allocator, params: *const Value, st: ?*const Stream) !Value {
    const command = try atString(params, "command");
    var timeout = common.jsonValue(params, "timeout", c_int, 10);
    var max_output: usize = @bitCast(@as(isize, common.jsonValue(params, "max_output_size", c_int, @intCast(exec_shell_command_max_output_size))));

    timeout = @min(timeout, exec_shell_command_max_timeout);
    max_output = @min(max_output, exec_shell_command_max_output_size);

    // an isolate is always POSIX regardless of host OS, so it always gets `sh -c`
    const args = [_][]const u8{ "sh", "-c", command };

    const io = try makeToolsIo(a, params);

    if (st) |s| {
        const Cb = struct {
            fn f(ctx: *anyopaque, chunk: []const u8) bool {
                const stream: *const Stream = @ptrCast(@alignCast(ctx));
                stream.push(chunk) catch |err| switch (err) {
                    error.OutOfMemory => {},
                };
                return stream.isAlive();
            }
        };
        const res = try io.run(a, &args, max_output, timeout, .{ .ctx = @constCast(s), .f = Cb.f });
        if (!s.isAlive()) return .null;
        var tail = try std.fmt.allocPrint(a, "\n[exit code: {d}]", .{res.exit_code});
        if (res.timed_out) tail = try std.mem.concat(a, u8, &.{ tail, " [exit due to timed out]" });
        try s.push(tail);
        return .null;
    }

    const res = try io.run(a, &args, max_output, timeout, null);

    var text_output = try std.fmt.allocPrint(a, "{s}\n[exit code: {d}]", .{ res.output, res.exit_code });
    if (res.timed_out) text_output = try std.mem.concat(a, u8, &.{ text_output, " [exit due to timed out]" });

    return plainText(a, text_output);
}

/// Ports `server_tool_write_file`'s `invoke` (server-tools.cpp:1350 @620d2225c).
fn invokeWriteFile(a: Allocator, params: *const Value) !Value {
    const path = try atString(params, "path");
    const content = try atString(params, "content");

    const io = try makeToolsIo(a, params);
    if (!try io.writeFile(a, path, content)) {
        return errorObj(a, try std.mem.concat(a, u8, &.{ "failed to write file: ", path }));
    }

    var o = common.Obj.init(a);
    try o.put("result", "file written successfully");
    try o.put("path", path);
    try o.put("bytes", @as(usize, content.len));
    return o.done();
}

/// Ports `server_tool_edit_file`'s `matched_edit` (server-tools.cpp:1598 @620d2225c).
const MatchedEdit = struct {
    edit_index: usize,
    match_index: usize, // offset into the "base content"
    match_length: usize,
    new_text: []const u8,
};

/// `std::string::npos`.
const npos = std.math.maxInt(usize);

/// `s.substr(pos, n)`: throws `std::out_of_range` past the end.
fn substr(s: []const u8, pos: usize, n: usize) ![]const u8 {
    if (pos > s.len) return throw(error.RuntimeError, "basic_string", .{});
    return s[pos..@min(s.len, pos +| n)];
}

/// `s.find(needle)`, or `npos`.
fn find(s: []const u8, needle: []const u8) usize {
    return std.mem.find(u8, s, needle) orelse npos;
}

/// Ports `server_tool_edit_file`'s `normalize_line_for_fuzzy_match`
/// (server-tools.cpp:1498 @620d2225c): trailing whitespace stripped, smart quotes,
/// dashes and spaces made ASCII.
fn normalizeLineForFuzzyMatch(a: Allocator, line: []const u8) ![]const u8 {
    var end = line.len;
    while (end > 0 and (line[end - 1] == ' ' or line[end - 1] == '\t' or line[end - 1] == '\r')) end -= 1;
    var s: []const u8 = line[0..end];

    const R = struct {
        fn replaceAll(al: Allocator, str: []const u8, from: []const u8, to: []const u8) ![]const u8 {
            if (from.len == 0 or std.mem.find(u8, str, from) == null) return str;
            return std.mem.replaceOwned(u8, al, str, from, to);
        }
    };

    // smart single quotes -> '
    for ([_]u8{ 0x98, 0x99, 0x9A, 0x9B }) |b| s = try R.replaceAll(a, s, &.{ 0xE2, 0x80, b }, "'");
    // smart double quotes -> "
    for ([_]u8{ 0x9C, 0x9D, 0x9E, 0x9F }) |b| s = try R.replaceAll(a, s, &.{ 0xE2, 0x80, b }, "\"");
    // various dashes -> -
    var b: u8 = 0x90;
    while (b <= 0x95) : (b += 1) s = try R.replaceAll(a, s, &.{ 0xE2, 0x80, b }, "-");
    s = try R.replaceAll(a, s, "\xE2\x88\x92", "-"); // minus sign
    // special spaces -> ' '
    s = try R.replaceAll(a, s, "\xC2\xA0", " "); // no-break space
    b = 0x82;
    while (b <= 0x8A) : (b += 1) s = try R.replaceAll(a, s, &.{ 0xE2, 0x80, b }, " ");
    s = try R.replaceAll(a, s, "\xE2\x80\xAF", " "); // narrow no-break space
    s = try R.replaceAll(a, s, "\xE2\x81\x9F", " "); // medium mathematical space
    s = try R.replaceAll(a, s, "\xE3\x80\x80", " "); // ideographic space
    return s;
}

/// Ports `server_tool_edit_file`'s `normalize_for_fuzzy_match` (server-tools.cpp:1540 @620d2225c):
/// every line normalized, line count and positions kept.
fn normalizeForFuzzyMatch(a: Allocator, content: []const u8) ![]const u8 {
    var result: std.ArrayList(u8) = .empty;
    var start: usize = 0;
    while (true) {
        const nl = std.mem.findScalarPos(u8, content, start, '\n');
        const line = if (nl) |n| content[start..n] else content[start..];
        try result.appendSlice(a, try normalizeLineForFuzzyMatch(a, line));
        const n = nl orelse break;
        try result.append(a, '\n');
        start = n + 1;
    }
    return result.items;
}

/// Ports `server_tool_edit_file`'s `split_lines_with_endings` (server-tools.cpp:1557 @620d2225c).
fn splitLinesWithEndings(a: Allocator, content: []const u8) !std.ArrayList([]const u8) {
    var lines: std.ArrayList([]const u8) = .empty;
    var start: usize = 0;
    while (start < content.len) {
        const nl = std.mem.findScalarPos(u8, content, start, '\n') orelse {
            try lines.append(a, content[start..]);
            break;
        };
        try lines.append(a, content[start .. nl + 1]);
        start = nl + 1;
    }
    return lines;
}

/// Ports `server_tool_edit_file`'s `line_span` (server-tools.cpp:1572 @620d2225c).
const LineSpan = struct { start: usize, end: usize };

/// Ports `server_tool_edit_file`'s `get_line_spans` (server-tools.cpp:1577 @620d2225c).
fn getLineSpans(a: Allocator, content: []const u8) !std.ArrayList(LineSpan) {
    var spans: std.ArrayList(LineSpan) = .empty;
    var offset: usize = 0;
    for ((try splitLinesWithEndings(a, content)).items) |line| {
        try spans.append(a, .{ .start = offset, .end = offset + line.len });
        offset += line.len;
    }
    return spans;
}

/// Ports `server_tool_edit_file`'s `count_occurrences` (server-tools.cpp:1588 @620d2225c):
/// non-overlapping.
fn countOccurrences(content: []const u8, needle: []const u8) usize {
    if (needle.len == 0) return 0;
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.findPos(u8, content, pos, needle)) |p| {
        count += 1;
        pos = p + needle.len;
    }
    return count;
}

/// Ports `server_tool_edit_file`'s `apply_replacements` (server-tools.cpp:1606 @620d2225c):
/// sorted, non-overlapping replacements applied from the back.
fn applyReplacements(a: Allocator, content: []const u8, replacements: []const MatchedEdit, offset: usize) ![]const u8 {
    var result = content;
    var i = replacements.len;
    while (i > 0) {
        i -= 1;
        const it = replacements[i];
        const local_index = it.match_index -% offset;
        result = try std.mem.concat(a, u8, &.{ try substr(result, 0, local_index), it.new_text, try substr(result, local_index +% it.match_length, npos) });
    }
    return result;
}

/// Ports `server_tool_edit_file`'s `get_replacement_line_range` (server-tools.cpp:1619 @620d2225c):
/// a replacement's byte range widened to the lines it touches.
fn getReplacementLineRange(lines: []const LineSpan, match_index: usize, match_length: usize, out_start_line: *usize, out_end_line: *usize) bool {
    const replacement_start = match_index;
    const replacement_end = match_index +% match_length;

    var start_line: usize = npos;
    for (lines, 0..) |l, i| {
        if (replacement_start >= l.start and replacement_start < l.end) {
            start_line = i;
            break;
        }
    }
    if (start_line == npos) return false;

    var end_line = start_line;
    while (end_line < lines.len and lines[end_line].end < replacement_end) end_line += 1;
    if (end_line >= lines.len) return false;

    out_start_line.* = start_line;
    out_end_line.* = end_line + 1;
    return true;
}

/// Ports `server_tool_edit_file`'s `apply_replacements_preserving_unchanged_lines`
/// (server-tools.cpp:1647 @620d2225c): as `apply_replacements`, but untouched
/// lines come from the original content.
fn applyReplacementsPreservingUnchangedLines(a: Allocator, original_content: []const u8, base_content: []const u8, replacements: []const MatchedEdit) ![]const u8 {
    const original_lines = (try splitLinesWithEndings(a, original_content)).items;
    const base_lines = (try getLineSpans(a, base_content)).items;

    const Group = struct {
        start_line: usize,
        end_line: usize, // exclusive
        reps: std.ArrayList(MatchedEdit),
    };
    var groups: std.ArrayList(Group) = .empty;

    for (replacements) |rep| {
        var start_line: usize = 0;
        var end_line: usize = 0;
        _ = getReplacementLineRange(base_lines, rep.match_index, rep.match_length, &start_line, &end_line);
        if (groups.items.len != 0 and start_line < groups.items[groups.items.len - 1].end_line) {
            const g = &groups.items[groups.items.len - 1];
            g.end_line = @max(g.end_line, end_line);
            try g.reps.append(a, rep);
        } else {
            var reps: std.ArrayList(MatchedEdit) = .empty;
            try reps.append(a, rep);
            try groups.append(a, .{ .start_line = start_line, .end_line = end_line, .reps = reps });
        }
    }

    var original_line_index: usize = 0;
    var result: std.ArrayList(u8) = .empty;
    for (groups.items) |g| {
        var i = original_line_index;
        while (i < g.start_line) : (i += 1) try result.appendSlice(a, original_lines[i]);

        const group_start_offset = base_lines[g.start_line].start;
        const group_end_offset = base_lines[g.end_line -% 1].end;
        const slice = try substr(base_content, group_start_offset, group_end_offset -% group_start_offset);
        try result.appendSlice(a, try applyReplacements(a, slice, g.reps.items, group_start_offset));

        original_line_index = g.end_line;
    }
    var i = original_line_index;
    while (i < original_lines.len) : (i += 1) try result.appendSlice(a, original_lines[i]);

    return result.items;
}

/// Ports `server_tool_edit_file`'s `invoke` (server-tools.cpp:1407 @620d2225c): every
/// edit matched against the original content, exactly or -- when any one
/// has no exact match -- after normalizing whitespace and typography.
fn invokeEditFile(a: Allocator, params: *const Value) !Value {
    const path = try atString(params, "path");
    const edits_json = try params.at("edits");

    if (!edits_json.isArray() or edits_json.empty()) {
        return errorObj(a, "\"edits\" must be a non-empty array");
    }

    const EditReq = struct { old_text: []const u8, new_text: []const u8 };
    var edits: std.ArrayList(EditReq) = .empty;
    for (edits_json.array.items) |*e| {
        const er: EditReq = .{ .old_text = try atString(e, "old_text"), .new_text = try atString(e, "new_text") };
        if (er.old_text.len == 0) {
            return errorObj(a, try std.fmt.allocPrint(a, "edits[{d}].old_text must not be empty", .{edits.items.len}));
        }
        try edits.append(a, er);
    }

    const io = try makeToolsIo(a, params);
    const original_content = (try io.readFile(a, path)) orelse return errorObj(a, try std.mem.concat(a, u8, &.{ "failed to open file: ", path }));

    // does any old_text need fuzzy matching (no exact match found)?
    var any_fuzzy = false;
    for (edits.items, 0..) |e, i| {
        if (find(original_content, e.old_text) != npos) continue;
        const fuzzy_content = try normalizeForFuzzyMatch(a, original_content);
        const fuzzy_old = try normalizeForFuzzyMatch(a, e.old_text);
        if (find(fuzzy_content, fuzzy_old) == npos) {
            return errorObj(a, try std.fmt.allocPrint(a, "could not find edits[{d}].old_text in {s}, it must match the file's current content exactly", .{ i, path }));
        }
        any_fuzzy = true;
    }

    const base_content = if (any_fuzzy) try normalizeForFuzzyMatch(a, original_content) else original_content;

    // uniqueness check always uses fuzzy-normalized text, so a whitespace-only duplicate still counts
    var matched: std.ArrayList(MatchedEdit) = .empty;
    for (edits.items, 0..) |e, i| {
        const needle = if (any_fuzzy) try normalizeForFuzzyMatch(a, e.old_text) else e.old_text;
        const occurrences = countOccurrences(try normalizeForFuzzyMatch(a, original_content), try normalizeForFuzzyMatch(a, e.old_text));
        if (occurrences > 1) {
            return errorObj(a, try std.fmt.allocPrint(a, "found {d} occurrences of edits[{d}].old_text in {s}, it must be unique", .{ occurrences, i, path }));
        }
        const idx = find(base_content, needle);
        try matched.append(a, .{ .edit_index = i, .match_index = idx, .match_length = needle.len, .new_text = e.new_text });
    }

    const Less = struct {
        fn f(_: void, x: MatchedEdit, y: MatchedEdit) bool {
            return x.match_index < y.match_index;
        }
    };
    libcxx.sort(MatchedEdit, matched.items, {}, Less.f);
    var i: usize = 1;
    while (i < matched.items.len) : (i += 1) {
        const prev = matched.items[i - 1];
        if (prev.match_index +% prev.match_length > matched.items[i].match_index) {
            return errorObj(a, try std.fmt.allocPrint(a, "edits[{d}] and edits[{d}] overlap in {s}; merge them into one edit or target disjoint regions", .{ prev.edit_index, matched.items[i].edit_index, path }));
        }
    }

    const new_content = if (any_fuzzy)
        try applyReplacementsPreservingUnchangedLines(a, original_content, base_content, matched.items)
    else
        try applyReplacements(a, base_content, matched.items, 0);

    if (std.mem.eql(u8, new_content, original_content)) {
        return errorObj(a, "no changes made: the replacement(s) produced identical content");
    }

    if (!try io.writeFile(a, path, new_content)) {
        return errorObj(a, try std.mem.concat(a, u8, &.{ "failed to write file: ", path }));
    }

    var o = common.Obj.init(a);
    try o.put("result", "file edited successfully");
    try o.put("path", path);
    try o.put("edits_applied", @as(c_int, @intCast(matched.items.len)));
    return o.done();
}

/// Ports `server_tool_get_info`'s `invoke` (server-tools.cpp:1723 @620d2225c): `uname
/// -a` and the working directory.
fn invokeGetInfo(a: Allocator, params: *const Value) !Value {
    const io = try makeToolsIo(a, params);

    // inside an isolate, we always use the linux command
    const res = try io.run(a, &.{ "uname", "-a" }, get_info_max_output, get_info_timeout, null);
    // a failed spawn or a timeout leaves a diagnostic in res.output, which is not an OS name
    const os_info = if (res.exit_code == 0 and !res.timed_out) tc.strings.strip(res.output) else "unknown";

    var cwd: []const u8 = common.jsonValue(params, "cwd", []const u8, "");
    if (cwd.len == 0) {
        if (common.jsonValue(params, "runtime", []const u8, "").len == 0) {
            cwd = currentPath(a) orelse "";
        } else {
            const pwd = try io.run(a, &.{"pwd"}, get_info_max_output, get_info_timeout, null);
            cwd = if (pwd.exit_code == 0 and !pwd.timed_out) tc.strings.strip(pwd.output) else "unknown";
        }
    }

    var o = common.Obj.init(a);
    try o.put("os", os_info);
    try o.put("cwd", cwd);
    return o.done();
}

// -----------------------------------------------------------------------------
// streaming

/// Ports `server_tool_stream_result` (server-tools.cpp:1758 @620d2225c): one chunk of a
/// streamed tool's output, or the end.
pub const StreamResult = struct {
    id: i32 = -1,
    chunk: []u8 = &.{},
    done: bool = false,
    error_msg: []u8 = &.{},

    fn create() !*StreamResult {
        const r = try gpa.create(StreamResult);
        r.* = .{};
        return r;
    }

    pub fn destroy(self: *StreamResult, al: Allocator) void {
        al.free(self.chunk);
        al.free(self.error_msg);
        al.destroy(self);
    }

    /// Ports `server_tool_stream_result`'s `to_json` (server-tools.cpp:1763 @620d2225c).
    fn toJson(self: *const StreamResult, a: Allocator) !Value {
        var o = common.Obj.init(a);
        if (!self.done) {
            try o.put("chunk", self.chunk);
        } else {
            try o.put("done", true);
            if (self.error_msg.len != 0) try o.put("error", self.error_msg);
        }
        return o.done();
    }
};

/// `server_response` over tool stream results.
pub const ResultQueue = queue.Response(*StreamResult);

/// Ports `server_tools_res` (server-tools.cpp:1784 @620d2225c): a response whose
/// destructor joins the streaming worker and stops waiting for its id.
const ToolsRes = struct {
    base: http.Res,
    worker: ?std.Thread = null,
    qr: ?*ResultQueue = null, // set only for streaming responses
    id: i32 = -1,

    // the stream's state, owned here
    owner: ?*Tools = null,
    worker_arena: ?*std.heap.ArenaAllocator = null,

    const vtable: http.Res.VTable = .{ .on_complete = http.Res.noComplete, .destroy = destroyRes };

    fn create() !*ToolsRes {
        const self = try gpa.create(ToolsRes);
        self.* = .{ .base = .{ .arena = std.heap.ArenaAllocator.init(gpa), .vtable = &vtable } };
        return self;
    }

    /// Ports `server_tools_res`'s destructor (server-tools.cpp:1789 @620d2225c).
    fn destroyRes(res: *http.Res) void {
        const self: *ToolsRes = @fieldParentPtr("base", res);
        if (self.worker) |t| t.join();
        if (self.qr) |q| q.removeWaitingTaskId(self.id);
        if (self.worker_arena) |wa| {
            wa.deinit();
            gpa.destroy(wa);
        }
        self.base.arena.deinit();
        gpa.destroy(self);
    }
};

// -----------------------------------------------------------------------------
// runtimes

/// Ports `server_tools_runtime` (server-tools.cpp:1844 @620d2225c), and its two
/// subclasses: `server_tools_static_runtime` (server-tools.cpp:1851 @620d2225c), a
/// target that already exists, and `server_tools_container_runtime`
/// (server-tools.cpp:1861 @620d2225c), a container this server spawns and owns.
const Runtime = union(enum) {
    static: []const u8,
    container: *ContainerRuntime,

    /// Ports `server_tools_static_runtime`'s `spec` (server-tools.cpp:1853 @620d2225c)
    /// and `server_tools_container_runtime`'s `spec` (server-tools.cpp:1885 @620d2225c):
    /// the runtime string `make_tools_io` takes, once per tool call.
    ///
    /// Return: owned by `a`.
    fn spec(self: Runtime, a: Allocator) ![]const u8 {
        return switch (self) {
            .static => |s| s,
            .container => |c| c.spec(a),
        };
    }

    fn deinit(self: Runtime) void {
        switch (self) {
            .static => |s| gpa.free(s),
            .container => |c| c.destroy(),
        }
    }
};

/// Ports `server_tools_container_runtime` (server-tools.cpp:1861 @620d2225c): owns
/// the container `--tools-runtime "<engine>:<image>"` names, spawned here
/// and stopped when the server exits.
const ContainerRuntime = struct {
    bin: []const u8,
    image: []const u8,
    container_id: []u8 = &.{},
    proc: Subproc = .{}, // `<engine> run` client that keeps the container alive
    mutex: mcp.Mutex = .{},

    /// Ports `server_tools_container_runtime`'s constructor (server-tools.cpp:1864 @620d2225c).
    fn create(spec_str: []const u8) !*ContainerRuntime {
        const parsed = ContainerSpec.parse(spec_str) orelse
            return throw(error.RuntimeError, "unknown --tools-runtime option: {s}", .{spec_str});
        if (parsed.arg.len == 0) {
            return throw(error.RuntimeError, "--tools-runtime {s}:<image> requires an image name", .{parsed.bin});
        }
        const self = try gpa.create(ContainerRuntime);
        self.* = .{ .bin = parsed.bin, .image = try gpa.dupe(u8, parsed.arg) };
        self.spawn() catch |e| {
            gpa.free(self.image);
            gpa.destroy(self);
            return e;
        };
        return self;
    }

    /// Ports `server_tools_container_runtime`'s destructor (server-tools.cpp:1878 @620d2225c): closing stdin signals
    /// the container's shell to exit; `--rm` then removes it.
    fn destroy(self: *ContainerRuntime) void {
        self.proc.closeStdin();
        _ = self.proc.join();
        self.proc.deinit();
        gpa.free(self.container_id);
        gpa.free(self.image);
        gpa.destroy(self);
    }

    /// Ports `server_tools_container_runtime`'s `spec` (server-tools.cpp:1885 @620d2225c):
    /// respawns a container that died on its own.
    fn spec(self: *ContainerRuntime, a: Allocator) ![]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!self.proc.alive()) {
            const b = z(self.bin);
            defer common.freeZ(b);
            const c = z(self.container_id);
            defer common.freeZ(c);
            srv(.wrn, "spec", "%s tools runtime container \"%s\" died, respawning\n", .{ b, c });
            try self.spawn();
        }
        return std.mem.concat(a, u8, &.{ self.bin, "-container:", self.container_id });
    }

    /// Ports `server_tools_container_runtime`'s `spawn` (server-tools.cpp:1903 @620d2225c):
    /// `<engine> run --rm -i --cidfile <tmp> <image> sh`, its stdin kept
    /// open so the shell blocks and the container stays up.
    fn spawn(self: *ContainerRuntime) !void {
        // create() writes over the handle it is given, so the previous one is released first
        _ = self.proc.join();

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cidfile = try pathJoin(a, tempDirectoryPath(), try std.fmt.allocPrint(a, "llama-tools-runtime-cid-{d}.tmp", .{pthread_self()}));
        const cz = try a.dupeSentinel(u8, cidfile, 0);
        _ = unlink(cz.ptr);

        const args = [_][]const u8{ self.bin, "run", "--rm", "-i", "--cidfile", cidfile, self.image, "sh" };
        const options = tc.subproc.option_no_window | tc.subproc.option_inherit_environment | tc.subproc.option_search_user_path;
        if (!self.proc.create(&args, options, &.{}, null)) {
            return throw(error.RuntimeError, "failed to spawn {s} container for tools runtime (image: {s})", .{ self.bin, self.image });
        }

        var cid: []const u8 = "";
        var i: usize = 0;
        while (i < 100 and cid.len == 0) : (i += 1) {
            if (readWhole(a, cidfile)) |text| {
                cid = if (std.mem.findScalar(u8, text, '\n')) |nl| text[0..nl] else text;
            }
            if (cid.len == 0) mcp.sleepMs(100);
        }
        _ = unlink(cz.ptr);
        if (cid.len == 0) {
            self.proc.terminate();
            return throw(error.RuntimeError, "timed out waiting for {s} container to start (image: {s})", .{ self.bin, self.image });
        }
        gpa.free(self.container_id);
        self.container_id = try gpa.dupe(u8, cid);
    }
};

/// `fs::temp_directory_path(ec)`: `TMPDIR`, `TMP`, `TEMP`, `TEMPDIR` or
/// `/tmp`, which must be a directory; empty on an error.
fn tempDirectoryPath() []const u8 {
    var p: []const u8 = "/tmp";
    for ([_][*:0]const u8{ "TMPDIR", "TMP", "TEMP", "TEMPDIR" }) |name| {
        if (std.c.getenv(name)) |v| {
            p = std.mem.span(v);
            break;
        }
    }
    if (statType(p, null) != .directory) return "";
    return p;
}

/// Ports `make_tools_runtime` (server-tools.cpp:1988 @620d2225c): the
/// `"<engine>:<image>"` form owns a container; anything else names an
/// existing target, validated now.
fn makeToolsRuntime(spec_str: []const u8) !Runtime {
    if (ContainerSpec.parse(spec_str)) |parsed| {
        if (!parsed.attach) return .{ .container = try ContainerRuntime.create(spec_str) };
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var params = Value.emptyObject();
    try params.set(arena.allocator(), "runtime", try Value.initString(arena.allocator(), spec_str));
    _ = try makeToolsIo(arena.allocator(), &params); // nothing to own, just reject a bad spec now
    return .{ .static = try gpa.dupe(u8, spec_str) };
}

// -----------------------------------------------------------------------------
// server_tools

/// Ports `find_tool` (server-tools.cpp:1935 @620d2225c).
fn findTool(tools: []const *Tool, name: []const u8, require_stream: bool) !*Tool {
    for (tools) |t| {
        if (std.mem.eql(u8, t.name, name)) {
            if (require_stream and !t.support_stream) {
                return throw(error.InvalidArgument, "tool \"{s}\" does not support stream = true", .{name});
            }
            return t;
        }
    }
    return throw(error.InvalidArgument, "unknown tool \"{s}\"", .{name});
}

/// Ports `build_tools` (server-tools.cpp:1951 @620d2225c), with each tool's constructor:
/// `server_tool_read_file` (server-tools.cpp:870 @620d2225c),
/// `server_tool_file_glob_search` (server-tools.cpp:978 @620d2225c),
/// `server_tool_grep_search` (server-tools.cpp:1093 @620d2225c),
/// `server_tool_exec_shell_command` (server-tools.cpp:1246 @620d2225c),
/// `server_tool_write_file` (server-tools.cpp:1325 @620d2225c),
/// `server_tool_edit_file` (server-tools.cpp:1368 @620d2225c),
/// `server_tool_get_info` (server-tools.cpp:1702 @620d2225c).
fn buildTools() ![7]*Tool {
    const all = [7]Tool{
        .{ .name = "read_file", .display_name = "Read file", .uses_cwd = true, .kind = .read_file },
        .{ .name = "file_glob_search", .display_name = "File search", .uses_cwd = true, .kind = .file_glob_search },
        .{ .name = "grep_search", .display_name = "Grep search", .uses_cwd = true, .kind = .grep_search },
        .{ .name = "exec_shell_command", .display_name = "Execute shell command", .uses_cwd = true, .permission_write = true, .support_stream = true, .kind = .exec_shell_command },
        .{ .name = "write_file", .display_name = "Write file", .uses_cwd = true, .permission_write = true, .kind = .write_file },
        .{ .name = "edit_file", .display_name = "Edit file", .uses_cwd = true, .permission_write = true, .kind = .edit_file },
        .{ .name = "get_info", .display_name = "Get Runtime Info", .uses_cwd = true, .kind = .get_info },
    };
    var out: [7]*Tool = undefined;
    for (all, 0..) |t, i| {
        out[i] = try gpa.create(Tool);
        out[i].* = t;
    }
    return out;
}

/// Ports `get_header` (server-tools.cpp:1973 @620d2225c) over `str_to_lower`
/// (server-tools.cpp:1967 @620d2225c): the first header whose name matches
/// without case, in the map's order.
fn getHeader(headers: *const http.StringMap, key: []const u8) []const u8 {
    for (headers.entries.items) |h| {
        if (std.ascii.eqlIgnoreCase(h.key, key)) return h.value;
    }
    return "";
}

/// Ports `server_tools` (server-tools.h:36 @620d2225c): the enabled tools,
/// the queue their streams go through, and the `--tools-runtime` isolate
/// every call is routed through.
pub const Tools = struct {
    tools: std.ArrayList(*Tool) = .empty,

    // for streaming
    queue_res: ResultQueue,
    res_id: std.atomic.Value(i32) = .init(0),

    // set when --tools-runtime is configured; routes every tool call through an isolate
    runtime: ?Runtime = null,

    /// Ports `server_tools`'s constructor (server-tools.cpp:1983 @620d2225c).
    pub fn init() !Tools {
        return .{ .queue_res = try ResultQueue.init(gpa) };
    }

    /// Ports `server_tools`'s destructor (server-tools.cpp:1984 @620d2225c): the runtime's
    /// container stopped.
    pub fn deinit(self: *Tools) void {
        if (self.runtime) |r| r.deinit();
        for (self.tools.items) |t| {
            if (t.def) |*d| {
                var dd = d.*;
                mcp.freeDef(&dd);
                gpa.free(t.name);
            }
            gpa.destroy(t);
        }
        self.tools.deinit(gpa);
        self.queue_res.deinit();
    }

    /// Ports `server_tools`'s `setup` (server-tools.cpp:1997 @620d2225c): the runtime,
    /// the enabled built-in tools (`all` for every one), then the MCP
    /// servers' tools, skipping a name already taken.
    pub fn setup(self: *Tools, enabled_tools: []const []const u8, mcp_mgr: *mcp.Mcp, tools_runtime: []const u8) !void {
        if (tools_runtime.len != 0) self.runtime = try makeToolsRuntime(tools_runtime);

        if (enabled_tools.len != 0) {
            if (!Subproc.isSupported()) return throw(error.RuntimeError, "subprocess is not enabled on this build", .{});

            var all_tools = try buildTools();
            var taken: [7]bool = @splat(false);
            defer for (all_tools, taken) |t, k| {
                if (!k) gpa.destroy(t);
            };

            // validate that every requested tool is known
            for (enabled_tools) |name| {
                if (std.mem.eql(u8, name, "all")) continue;
                var known = false;
                for (all_tools) |t| known = known or std.mem.eql(u8, t.name, name);
                if (!known) {
                    var names: std.ArrayList(u8) = .empty;
                    defer names.deinit(gpa);
                    for (all_tools, 0..) |t, i| {
                        if (i != 0) try names.appendSlice(gpa, ", ");
                        try names.appendSlice(gpa, t.name);
                    }
                    return throw(error.RuntimeError, "unknown tool \"{s}\". available tools: {s}", .{ name, names.items });
                }
            }

            self.tools.clearRetainingCapacity();
            for (&all_tools, 0..) |t, i| {
                var enabled = false;
                for (enabled_tools) |name| enabled = enabled or std.mem.eql(u8, name, t.name) or std.mem.eql(u8, name, "all");
                if (enabled) {
                    try self.tools.append(gpa, t);
                    taken[i] = true;
                }
            }
        }

        // append MCP tools, skipping any that collide with a server tool or another MCP tool of the same "<server>_<tool>" name
        if (!mcp_mgr.empty()) {
            var seen_names: std.StringHashMapUnmanaged(void) = .empty;
            defer seen_names.deinit(gpa);
            for (self.tools.items) |t| try seen_names.put(gpa, t.name, {});
            var n_added: usize = 0;
            const list = try mcp_mgr.listTools();
            defer if (list.len != 0) gpa.free(list);
            for (list) |*def| {
                const mcp_name = try std.mem.concat(gpa, u8, &.{ def.server_name, "_", def.name });
                if (seen_names.contains(mcp_name)) {
                    const n = z(mcp_name);
                    defer common.freeZ(n);
                    const s = z(def.server_name);
                    defer common.freeZ(s);
                    srv(.wrn, "setup", "MCP tool \"%s\" from server \"%s\" collides with an existing tool, skipping\n", .{ n, s });
                    gpa.free(mcp_name);
                    mcp.freeDef(def);
                    continue;
                }
                try seen_names.put(gpa, mcp_name, {});
                // Ports `server_mcp_tool`'s constructor (server-tools.cpp:1808 @620d2225c).
                const t = try gpa.create(Tool);
                t.* = .{
                    .name = mcp_name,
                    .display_name = mcp_name,
                    .kind = .mcp,
                    .server_name = def.server_name,
                    .tool_name = def.name,
                    .def = def.*,
                    .mcp_mgr = mcp_mgr,
                };
                try self.tools.append(gpa, t);
                n_added += 1;
            }
            if (n_added > 0) srv(.inf, "setup", "Added %zu MCP tools\n", .{n_added});
        }
    }

    /// `handle_get`, bound to this.
    pub fn handleGet(self: *Tools) http.Handler {
        return .{ .ctx = self, .func = handleGetImpl };
    }

    /// `handle_post`, bound to this.
    pub fn handlePost(self: *Tools) http.Handler {
        return .{ .ctx = self, .func = handlePostImpl };
    }

    /// Ports the `handle_get` lambda (server-tools.cpp:2061 @620d2225c): every tool's
    /// `to_json`.
    fn handleGetImpl(ctx: ?*anyopaque, _: *const http.Req) anyerror!*http.Res {
        const self: *Tools = @ptrCast(@alignCast(ctx.?));
        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();
        build: {
            var result = Value.emptyArray();
            for (self.tools.items) |t| {
                const v = t.toJson(a) catch |e| {
                    try serverError(res, e);
                    break :build;
                };
                try result.pushBack(a, v);
            }
            try res.setData(try common.safeJsonToStr(a, &result));
        }
        return res;
    }

    /// The handlers' `catch (const std::exception & e)`: logged, 500.
    fn serverError(res: *http.Res, e: anyerror) !void {
        const a = res.allocator();
        const w = common.exception.what(e);
        const wz = try a.dupeSentinel(u8, w, 0);
        srv(.err, "operator()", "got exception: %s\n", .{wz.ptr});
        res.status = 500;
        var v = try common.formatErrorResponse(a, w, .server);
        try res.setData(try common.safeJsonToStr(a, &v));
    }

    /// Ports the `handle_post` lambda (server-tools.cpp:2077 @620d2225c): one tool call,
    /// its `cwd`, `runtime` and `resp_type` taken from the `x-tool-cwd`,
    /// `x-tool-runtime` and `x-resp-type` headers and never from the body;
    /// streamed through `queue_res` when asked.
    fn handlePostImpl(ctx: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
        const self: *Tools = @ptrCast(@alignCast(ctx.?));
        const res = try ToolsRes.create();
        errdefer res.base.destroy();
        const a = res.base.allocator();
        self.post(res, req) catch |e| switch (e) {
            error.JsonError => {
                res.base.status = 400;
                var v = try common.formatErrorResponse(a, json.what(), .invalid_request);
                try res.base.setData(try common.safeJsonToStr(a, &v));
            },
            error.InvalidArgument => {
                res.base.status = 404;
                var v = try common.formatErrorResponse(a, common.what(e), .invalid_request);
                try res.base.setData(try common.safeJsonToStr(a, &v));
            },
            else => try serverError(&res.base, e),
        };
        return &res.base;
    }

    fn post(self: *Tools, res: *ToolsRes, req: *const http.Req) !void {
        const a = res.base.allocator();
        var body = try json.parse(a, req.body);
        const tool_name = try (try body.at("tool")).get([]const u8);
        var params = if (body.contains("params")) try (try body.at("params")).clone(a) else Value.emptyObject();
        const stream = try body.value("stream", bool, false);

        // accept x-tool-cwd header to override of the process
        if (params.contains("cwd")) try params.erase(a, "cwd");
        const cwd = getHeader(&req.headers, "x-tool-cwd");
        if (cwd.len != 0) try params.set(a, "cwd", try Value.initString(a, cwd));

        // accept x-tool-runtime header to route tool I/O through an isolate, e.g. "docker-container:<id>";
        // falls back to the --tools-runtime isolate, if configured
        if (params.contains("runtime")) try params.erase(a, "runtime");
        const runtime_header = getHeader(&req.headers, "x-tool-runtime");
        if (runtime_header.len != 0) {
            try params.set(a, "runtime", try Value.initString(a, runtime_header));
        } else if (self.runtime) |r| {
            try params.set(a, "runtime", try Value.initString(a, try r.spec(a)));
        }

        // x-resp-type header is only used by read_file for now
        if (params.contains("resp_type")) try params.erase(a, "resp_type");
        const resp_type = getHeader(&req.headers, "x-resp-type");
        if (resp_type.len != 0) try params.set(a, "resp_type", try Value.initString(a, resp_type));

        const tool = try findTool(self.tools.items, tool_name, stream);

        if (stream) {
            const id = self.res_id.fetchAdd(1, .seq_cst);
            try self.queue_res.addWaitingTaskId(id);
            res.qr = &self.queue_res;
            res.id = id;

            const wa = try gpa.create(std.heap.ArenaAllocator);
            wa.* = std.heap.ArenaAllocator.init(gpa);
            res.worker_arena = wa;
            const job = try wa.allocator().create(Worker);
            job.* = .{ .tools = self, .id = id, .req = req, .tool = tool, .params = try params.clone(wa.allocator()), .arena = wa.allocator() };
            res.worker = try std.Thread.spawn(.{}, Worker.run, .{job});

            res.base.content_type = "text/event-stream";
            res.base.status = 200;
            const next = try a.create(Next);
            next.* = .{ .tools = self, .id = id };
            res.base.next = .{ .ctx = next, .func = Next.call };
        } else {
            const result = try tool.invoke(a, &params, null);
            res.base.status = 200;
            try res.base.setData(try common.safeJsonToStr(a, &result));
        }
    }

    /// The streaming worker's lambda: invokes the tool, then sends the
    /// final result with any exception's text.
    const Worker = struct {
        tools: *Tools,
        id: i32,
        req: *const http.Req,
        tool: *Tool,
        params: Value,
        arena: Allocator,

        fn alive(ctx: ?*anyopaque) bool {
            const req: *const http.Req = @ptrCast(@alignCast(ctx.?));
            return !req.should_stop.call();
        }

        fn run(self: *Worker) void {
            const st: Stream = .{ .qr = &self.tools.queue_res, .id = self.id, .alive = .{ .ctx = @constCast(self.req), .f = alive } };
            const done = StreamResult.create() catch return;
            _ = self.tool.invoke(self.arena, &self.params, &st) catch |e| {
                done.error_msg = gpa.dupe(u8, if (e == error.JsonError or e == error.RuntimeError or e == error.InvalidArgument or e == error.OutOfMemory)
                    common.exception.what(e)
                else
                    "An unknown error occurred") catch &.{};
            };
            done.id = st.id;
            done.done = true;
            st.qr.send(done) catch |err| switch (err) {
                error.OutOfMemory => {},
            };
        }
    };

    /// The `next` lambda of a streamed call: one result, as an SSE event.
    const Next = struct {
        tools: *Tools,
        id: i32,

        fn call(ctx: ?*anyopaque, out: *std.ArrayList(u8), al: Allocator) anyerror!bool {
            const self: *Next = @ptrCast(@alignCast(ctx.?));
            var ids = try queue.IdSet.init(gpa);
            defer ids.deinit();
            try ids.insert(self.id);
            const result = self.tools.queue_res.recv(&ids);
            defer result.destroy(gpa);
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const v = try result.toJson(arena.allocator());
            try out.appendSlice(al, "data: ");
            try out.appendSlice(al, try common.safeJsonToStr(arena.allocator(), &v));
            try out.appendSlice(al, "\n\n");
            if (result.done) {
                self.tools.queue_res.removeWaitingTaskId(self.id);
                return false;
            }
            return true;
        }
    };
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "lexically_normal and resolve agree with std::filesystem on POSIX paths" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("/a/c", try lexicallyNormal(a, "/a/./b/../c"));
    try std.testing.expectEqualStrings("/a/", try lexicallyNormal(a, "/a/b/.."));
    try std.testing.expectEqualStrings("/", try lexicallyNormal(a, "/.."));
    try std.testing.expectEqualStrings("../x", try lexicallyNormal(a, "../x"));
    try std.testing.expectEqualStrings(".", try lexicallyNormal(a, "a/.."));
    const io: Basic = .{ .cwd = "/w" };
    try std.testing.expectEqualStrings("/w/s", try io.resolve(a, "s/"));
    try std.testing.expectEqualStrings("/", try io.resolve(a, "/x/.."));
}

test "glob_match: * stops at /, ** does not, classes and ranges" {
    try std.testing.expect(globMatch("*.cpp", "a.cpp"));
    try std.testing.expect(!globMatch("*.cpp", "x/a.cpp"));
    try std.testing.expect(globMatch("**/*.cpp", "x/y/a.cpp"));
    try std.testing.expect(globMatch("[a-c]?", "bz"));
    try std.testing.expect(!globMatch("[!a-c]?", "bz"));
    try std.testing.expect(globMatch("[]]", "]"));
}

test "edit_file's fuzzy normalization keeps lines and drops trailing blanks" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("a\n'b'\n", try normalizeForFuzzyMatch(a, "a  \n\xE2\x80\x98b\xE2\x80\x99\t\n"));
}
