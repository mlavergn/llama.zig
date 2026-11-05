//! The Hugging Face hub cache: the layout `huggingface_hub` writes
//! (`models--<user>--<repo>/{refs,blobs,snapshots}`), what is in it, how a
//! downloaded blob becomes a snapshot entry, and the hub API calls that
//! list a repository's files.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/hf-cache.cpp` and `llama.cpp/common/hf-cache.h`
//! (v0.7.0, `620d2225c`) whole, their POSIX arms. Each declaration names
//! the C++ it replaces and the line it began at.
//!
//! # `std::filesystem`
//!
//! The C++ leans on `std::filesystem`; what it uses is here as functions
//! over byte strings with POSIX's separator: `operator/`, `parent_path`,
//! `lexically_normal`, `lexically_relative`, `relative`,
//! `create_directories`, `remove_all`, and `directory_iterator` and
//! `recursive_directory_iterator` in `readdir`'s order, which libc++'s
//! iterators follow. A recursive iteration does not descend into a
//! symbolic link to a directory, as `directory_options::none` does not.
//!
//! # The network
//!
//! `api_get` goes through `download.zig`'s `common_http_client`, so the
//! hub is reached over plain HTTP only: `https://huggingface.co/` -- the
//! default endpoint -- is refused with upstream's message for a build
//! without SSL, exactly as the reference refuses it. `MODEL_ENDPOINT` or
//! `HF_ENDPOINT` pointing at a plain-HTTP mirror works on both sides.

const std = @import("std");
const builtin = @import("builtin");
const tc = @import("tools_common");
const json = @import("json");
const common = @import("common.zig");
const download = @import("download.zig");

const log = tc.log;
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// A thrown exception: `RuntimeError` and `InvalidArgument` carry their
/// `what()` in `common.what`, `JsonError` in `json.what`.
pub const Error = common.Error;

// -----------------------------------------------------------------------------
// std::filesystem, as the C++ uses it

/// `p / q` (POSIX): `q` when it is absolute, else `p` and `q` with one
/// separator between them unless `p` is empty or already ends in one.
pub fn pathJoin(a: Allocator, p: []const u8, q: []const u8) Allocator.Error![]u8 {
    if (q.len != 0 and q[0] == '/') return a.dupe(u8, q);
    if (p.len == 0 or p[p.len - 1] == '/') return std.mem.concat(a, u8, &.{ p, q });
    return std.mem.concat(a, u8, &.{ p, "/", q });
}

/// `path::filename()`: what follows the last separator.
pub fn pathFilename(p: []const u8) []const u8 {
    const i = std.mem.lastIndexOfScalar(u8, p, '/') orelse return p;
    return p[i + 1 ..];
}

/// `path::parent_path()`: the path without its last element; a path whose
/// last element is its root keeps the root.
pub fn pathParent(p: []const u8) []const u8 {
    const i = std.mem.lastIndexOfScalar(u8, p, '/') orelse return "";
    // the separators before the filename go with it
    var j = i;
    while (j > 0 and p[j - 1] == '/') j -= 1;
    if (j == 0) return p[0..1]; // the root directory
    return p[0..j];
}

/// A path's elements as `path::iterator` gives them: `/` for the root
/// directory, then each filename, and an empty element for a trailing
/// separator.
fn pathElements(a: Allocator, p: []const u8) Allocator.Error![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    if (p.len == 0) return out.items;
    var i: usize = 0;
    if (p[0] == '/') {
        try out.append(a, "/");
        while (i < p.len and p[i] == '/') i += 1;
    }
    while (i < p.len) {
        const start = i;
        while (i < p.len and p[i] != '/') i += 1;
        try out.append(a, p[start..i]);
        if (i < p.len) {
            while (i < p.len and p[i] == '/') i += 1;
            if (i == p.len) try out.append(a, "");
        }
    }
    return out.items;
}

/// Ports `path::lexically_normal()` ([fs.path.generic]): no `.`, no
/// `name/..`, no `..` right after the root, one separator between
/// elements, a trailing separator kept, and `.` for nothing.
pub fn lexicallyNormal(a: Allocator, p: []const u8) Allocator.Error![]u8 {
    if (p.len == 0) return a.dupe(u8, "");
    const elems = try pathElements(a, p);
    const has_root = elems.len != 0 and std.mem.eql(u8, elems[0], "/");
    var names: std.ArrayList([]const u8) = .empty;
    var trailing = false;
    for (elems[@intFromBool(has_root)..], 0..) |e, k| {
        const is_last = k + @intFromBool(has_root) == elems.len - 1;
        if (e.len == 0) {
            trailing = true;
            continue;
        }
        if (std.mem.eql(u8, e, ".")) {
            // a dot element is removed; the separator after it is kept when last
            if (is_last) trailing = true;
            continue;
        }
        if (std.mem.eql(u8, e, "..")) {
            if (names.items.len != 0 and !std.mem.eql(u8, names.items[names.items.len - 1], "..")) {
                _ = names.pop();
                if (is_last) trailing = true;
                continue;
            }
            if (has_root) {
                if (is_last) trailing = names.items.len != 0;
                continue; // a root's parent is the root
            }
            try names.append(a, e);
            trailing = false;
            continue;
        }
        trailing = false;
        try names.append(a, e);
    }
    // a last ".." loses the separator after it
    if (names.items.len != 0 and std.mem.eql(u8, names.items[names.items.len - 1], "..")) trailing = false;
    var out: std.ArrayList(u8) = .empty;
    if (has_root) try out.append(a, '/');
    for (names.items, 0..) |n, k| {
        if (k != 0) try out.append(a, '/');
        try out.appendSlice(a, n);
    }
    if (trailing and names.items.len != 0) try out.append(a, '/');
    if (out.items.len == 0) try out.append(a, '.');
    return out.items;
}

/// Ports `path::lexically_relative(base)` ([fs.path.gen]) for two absolute
/// paths: the path that, appended to `base`, names `p`.
pub fn lexicallyRelative(a: Allocator, p: []const u8, base: []const u8) Allocator.Error![]u8 {
    const pe = try pathElements(a, p);
    const be = try pathElements(a, base);
    var i: usize = 0;
    while (i < pe.len and i < be.len and std.mem.eql(u8, pe[i], be[i])) i += 1;
    if (i == pe.len and i == be.len) return a.dupe(u8, ".");
    var n: i64 = 0;
    for (be[i..]) |e| {
        if (std.mem.eql(u8, e, "..")) {
            n -= 1;
        } else if (e.len != 0 and !std.mem.eql(u8, e, ".")) {
            n += 1;
        }
    }
    if (n < 0) return a.dupe(u8, "");
    if (n == 0 and (i == pe.len or pe[i].len == 0)) return a.dupe(u8, ".");
    var out: std.ArrayList(u8) = .empty;
    var k: i64 = 0;
    while (k < n) : (k += 1) {
        if (out.items.len != 0) try out.append(a, '/');
        try out.appendSlice(a, "..");
    }
    for (pe[i..]) |e| {
        if (out.items.len != 0 and out.items[out.items.len - 1] != '/') try out.append(a, '/');
        try out.appendSlice(a, e);
    }
    return out.items;
}

fn zpath(a: Allocator, p: []const u8) Allocator.Error![:0]u8 {
    return a.dupeSentinel(u8, p, 0);
}

extern fn realpath(path: [*:0]const u8, resolved: ?[*]u8) ?[*:0]u8;
extern fn symlink(target: [*:0]const u8, link: [*:0]const u8) c_int;
extern fn rename(old: [*:0]const u8, new: [*:0]const u8) c_int;
extern fn unlink(path: [*:0]const u8) c_int;
extern fn rmdir(path: [*:0]const u8) c_int;
extern fn readlink(path: [*:0]const u8, buf: [*]u8, len: usize) isize;
extern fn strerror(e: c_int) [*:0]const u8;
extern fn remove(path: [*:0]const u8) c_int;
extern fn getuid() c_uint;
extern fn getpwuid(uid: c_uint) ?*anyopaque;

fn errno() c_int {
    return std.c._errno().*;
}

fn statMode(a: Allocator, p: []const u8, follow: bool) ?u32 {
    const z = zpath(a, p) catch return null;
    defer a.free(z);
    const st = tc.fs_stat.stat(z.ptr, follow) orelse return null;
    return st.mode;
}

/// `std::filesystem::exists`.
pub fn exists(a: Allocator, p: []const u8) bool {
    return statMode(a, p, true) != null;
}

/// `std::filesystem::is_directory`.
pub fn isDirectory(a: Allocator, p: []const u8) bool {
    const m = statMode(a, p, true) orelse return false;
    return m & tc.fs_stat.IFMT == tc.fs_stat.IFDIR;
}

/// `std::filesystem::is_regular_file`.
pub fn isRegularFile(a: Allocator, p: []const u8) bool {
    const m = statMode(a, p, true) orelse return false;
    return m & tc.fs_stat.IFMT == tc.fs_stat.IFREG;
}

/// `std::filesystem::is_symlink`.
pub fn isSymlink(a: Allocator, p: []const u8) bool {
    const m = statMode(a, p, false) orelse return false;
    return m & tc.fs_stat.IFMT == tc.fs_stat.IFLNK;
}

/// `std::filesystem::create_directories(p, ec)`.
///
/// Return: the `errno` of the first failure, or 0.
pub fn createDirectories(a: Allocator, p: []const u8) c_int {
    if (p.len == 0) return 0;
    if (isDirectory(a, p)) return 0;
    const parent = pathParent(p);
    if (parent.len != 0 and parent.len < p.len) {
        const rc = createDirectories(a, parent);
        if (rc != 0) return rc;
    }
    const z = zpath(a, p) catch return @backingInt(std.c.E.NOMEM);
    defer a.free(z);
    if (std.c.mkdir(z.ptr, 0o777) != 0) {
        const e = errno();
        if (e == @backingInt(std.c.E.EXIST) and isDirectory(a, p)) return 0;
        return e;
    }
    return 0;
}

/// `std::filesystem::remove_all(p, ec)`: the count of entries removed, or
/// the `errno` of a failure.
pub fn removeAll(a: Allocator, p: []const u8) struct { n: u64, err: c_int } {
    const m = statMode(a, p, false) orelse return .{ .n = 0, .err = 0 };
    var n: u64 = 0;
    if (m & tc.fs_stat.IFMT == tc.fs_stat.IFDIR) {
        const entries = listDir(a, p) catch return .{ .n = n, .err = @backingInt(std.c.E.NOMEM) };
        for (entries) |name| {
            const child = pathJoin(a, p, name) catch return .{ .n = n, .err = @backingInt(std.c.E.NOMEM) };
            const r = removeAll(a, child);
            n += r.n;
            if (r.err != 0) return .{ .n = n, .err = r.err };
        }
        const z = zpath(a, p) catch return .{ .n = n, .err = @backingInt(std.c.E.NOMEM) };
        if (rmdir(z.ptr) != 0) return .{ .n = n, .err = errno() };
    } else {
        const z = zpath(a, p) catch return .{ .n = n, .err = @backingInt(std.c.E.NOMEM) };
        if (unlink(z.ptr) != 0) return .{ .n = n, .err = errno() };
    }
    return .{ .n = n + 1, .err = 0 };
}

/// `std::filesystem::remove(p, ec)`.
///
/// Return: the `errno` of a failure other than the path not existing, or 0.
pub fn removeFile(a: Allocator, p: []const u8) c_int {
    const z = zpath(a, p) catch return @backingInt(std.c.E.NOMEM);
    defer a.free(z);
    if (remove(z.ptr) != 0) {
        const e = errno();
        if (e == @backingInt(std.c.E.NOENT)) return 0;
        return e;
    }
    return 0;
}

/// A directory's entry names, `.` and `..` aside, in `readdir`'s order.
pub fn listDir(a: Allocator, p: []const u8) Allocator.Error![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    const z = try zpath(a, p);
    const dir = std.c.opendir(z.ptr) orelse return out.items;
    defer _ = std.c.closedir(dir);
    while (std.c.readdir(dir)) |ent| {
        const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        try out.append(a, try a.dupe(u8, name));
    }
    return out.items;
}

/// `std::filesystem::weakly_canonical` of a path whose every element
/// exists: `realpath`.
fn canonical(a: Allocator, p: []const u8) ?[]u8 {
    const z = zpath(a, p) catch return null;
    defer a.free(z);
    const r = realpath(z.ptr, null) orelse return null;
    defer std.c.free(r);
    return a.dupe(u8, std.mem.span(r)) catch null;
}

/// `std::error_code::message()` for an `errno`.
pub fn errMessage(e: c_int) [*:0]const u8 {
    return strerror(e);
}

/// `std::filesystem::absolute`: a relative path under the working
/// directory.
fn absolute(a: Allocator, p: []const u8) Allocator.Error![]u8 {
    if (p.len != 0 and p[0] == '/') return a.dupe(u8, p);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.c.getcwd(&buf, buf.len) orelse return a.dupe(u8, p);
    return pathJoin(a, std.mem.span(@as([*:0]const u8, @ptrCast(cwd))), p);
}

// -----------------------------------------------------------------------------
// hf_cache

/// The cache directory, decided on the first call.
var cache_dir_value: ?[]const u8 = null;
var cache_dir_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;

/// Ports `get_cache_directory` (hf-cache.cpp:33 @620d2225c): the first of
/// `LLAMA_CACHE`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `$HF_HOME/hub`,
/// `$XDG_CACHE_HOME/huggingface/hub`, `$HOME/.cache/huggingface/hub` and
/// the password database's home, decided on the first call.
fn getCacheDirectory() Error![]const u8 {
    _ = std.c.pthread_mutex_lock(&cache_dir_mutex);
    defer _ = std.c.pthread_mutex_unlock(&cache_dir_mutex);
    if (cache_dir_value) |v| return v;
    const a = std.heap.c_allocator;
    const Entry = struct { var_: [*:0]const u8, path: []const u8 };
    const entries = [_]Entry{
        .{ .var_ = "LLAMA_CACHE", .path = "" },
        .{ .var_ = "HF_HUB_CACHE", .path = "" },
        .{ .var_ = "HUGGINGFACE_HUB_CACHE", .path = "" },
        .{ .var_ = "HF_HOME", .path = "hub" },
        .{ .var_ = "XDG_CACHE_HOME", .path = "huggingface/hub" },
        .{ .var_ = "HOME", .path = ".cache/huggingface/hub" },
    };
    for (entries) |entry| {
        if (std.c.getenv(entry.var_)) |p| {
            const base = std.mem.span(p);
            if (base.len == 0) continue;
            cache_dir_value = if (entry.path.len == 0) try a.dupe(u8, base) else try pathJoin(a, base, entry.path);
            return cache_dir_value.?;
        }
    }
    if (comptime builtin.os.tag != .windows) {
        if (getpwuid(getuid())) |pw| {
            if (passwdDir(pw)) |dir| {
                if (dir.len != 0) {
                    cache_dir_value = try pathJoin(a, try pathJoin(a, dir, ".cache"), "huggingface/hub");
                    return cache_dir_value.?;
                }
            }
        }
    }
    return common.throw(error.RuntimeError, "Failed to determine HF cache directory", .{});
}

/// `pw->pw_dir`: at a different offset in Darwin's `struct passwd` and
/// glibc's.
fn passwdDir(pw: *anyopaque) ?[]const u8 {
    const p: [*]const ?[*:0]const u8 = @ptrCast(@alignCast(pw));
    // Darwin: name, passwd, uid+gid, change (time_t), class, gecos, dir
    // glibc:  name, passwd, uid+gid, gecos, dir
    const idx: usize = if (comptime builtin.os.tag.isDarwin()) 6 else 4;
    const d = p[idx] orelse return null;
    return std.mem.span(d);
}

/// Ports `folder_name_to_repo` (hf-cache.cpp:68 @620d2225c): `models--a--b`
/// as `a/b`, anything else as nothing.
fn folderNameToRepo(a: Allocator, folder: []const u8) Allocator.Error![]u8 {
    const prefix = "models--";
    if (!std.mem.startsWith(u8, folder, prefix)) {
        return a.dupe(u8, "");
    }
    return std.mem.replaceOwned(u8, a, folder[prefix.len..], "--", "/");
}

/// Ports `repo_to_folder_name` (hf-cache.cpp:78 @620d2225c).
fn repoToFolderName(a: Allocator, repo_id: []const u8) Allocator.Error![]u8 {
    const result = try std.mem.concat(a, u8, &.{ "models--", repo_id });
    return std.mem.replaceOwned(u8, a, result, "/", "--");
}

/// Ports `get_repo_path` (hf-cache.cpp:85 @620d2225c).
fn getRepoPath(a: Allocator, repo_id: []const u8) Error![]u8 {
    return pathJoin(a, try getCacheDirectory(), try repoToFolderName(a, repo_id));
}

/// Ports `is_hex_char` (hf-cache.cpp:89 @620d2225c).
fn isHexChar(ch: u8) bool {
    return (ch >= 'A' and ch <= 'F') or
        (ch >= 'a' and ch <= 'f') or
        (ch >= '0' and ch <= '9');
}

/// Ports `is_hex_string` (hf-cache.cpp:95 @620d2225c).
fn isHexString(s: []const u8, expected_len: usize) bool {
    if (s.len != expected_len) {
        return false;
    }
    for (s) |ch| {
        if (!isHexChar(ch)) {
            return false;
        }
    }
    return true;
}

/// Ports `is_alphanum` (hf-cache.cpp:107 @620d2225c).
fn isAlphanum(ch: u8) bool {
    return (ch >= 'A' and ch <= 'Z') or
        (ch >= 'a' and ch <= 'z') or
        (ch >= '0' and ch <= '9');
}

/// Ports `is_special_char` (hf-cache.cpp:113 @620d2225c).
fn isSpecialChar(ch: u8) bool {
    return ch == '/' or ch == '.' or ch == '-';
}

/// Ports `is_valid_repo_id` (hf-cache.cpp:120 @620d2225c): base chars
/// `[A-Za-z0-9_]` are always valid, special chars `[/.-]` must be
/// surrounded by base chars, exactly one `/` required.
pub fn isValidRepoId(repo_id: []const u8) bool {
    if (repo_id.len == 0 or repo_id.len > 256) {
        return false;
    }
    var slash: c_int = 0;
    var special = true;

    for (repo_id) |ch| {
        if (isAlphanum(ch) or ch == '_') {
            special = false;
        } else if (isSpecialChar(ch)) {
            if (special) {
                return false;
            }
            slash += @intFromBool(ch == '/');
            special = true;
        } else {
            return false;
        }
    }
    return !special and slash == 1;
}

/// Ports `is_valid_hf_token` (hf-cache.cpp:143 @620d2225c).
fn isValidHfToken(token: []const u8) bool {
    if (token.len < 37 or token.len > 256 or
        !std.mem.startsWith(u8, token, "hf_"))
    {
        return false;
    }
    for (token[3..]) |ch| {
        if (!isAlphanum(ch)) {
            return false;
        }
    }
    return true;
}

/// Ports `is_valid_commit` (hf-cache.cpp:156 @620d2225c).
fn isValidCommit(hash: []const u8) bool {
    return isHexString(hash, 40);
}

/// Ports `is_valid_oid` (hf-cache.cpp:160 @620d2225c).
fn isValidOid(oid: []const u8) bool {
    return isHexString(oid, 40) or isHexString(oid, 64);
}

/// Ports `is_valid_subpath` (hf-cache.cpp:164 @620d2225c): `subpath`, a
/// relative path, stays under `path` once normalized.
fn isValidSubpath(a: Allocator, path: []const u8, subpath: []const u8) Allocator.Error!bool {
    if (subpath.len != 0 and subpath[0] == '/') {
        return false; // never do a / b with b absolute
    }
    const b = try lexicallyNormal(a, try absolute(a, path));
    const t = try lexicallyNormal(a, try pathJoin(a, b, subpath));
    const be = try pathElements(a, b);
    const te = try pathElements(a, t);
    var i: usize = 0;
    while (i < be.len and i < te.len and std.mem.eql(u8, be[i], te[i])) i += 1;

    return i == be.len;
}

/// An `std::ofstream` written whole: false when it could not be opened or
/// written.
pub fn writeFile(a: Allocator, path: []const u8, data: []const u8) bool {
    const z = zpath(a, path) catch return false;
    defer a.free(z);
    const fd = std.c.open(z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o666));
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    var off: usize = 0;
    while (off < data.len) {
        const n = std.c.write(fd, data[off..].ptr, data.len - off);
        if (n <= 0) return false;
        off += @intCast(n);
    }
    return true;
}

/// The first line of a file, as `std::getline` reads it; null when the
/// file cannot be opened, empty when it is.
pub fn readFirstLine(a: Allocator, path: []const u8) Allocator.Error!?[]u8 {
    const z = try zpath(a, path);
    defer a.free(z);
    const fd = std.c.open(z.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n <= 0) break;
        const chunk = buf[0..@intCast(n)];
        if (std.mem.indexOfScalar(u8, chunk, '\n')) |nl| {
            try out.appendSlice(a, chunk[0..nl]);
            return out.items;
        }
        try out.appendSlice(a, chunk);
    }
    return out.items;
}

/// Ports `api_get` (hf-cache.cpp:175 @620d2225c): a hub API GET, its body
/// parsed as JSON; any other status throws with the body's `error`, or the
/// body itself.
fn apiGet(a: Allocator, url: []const u8, token: []const u8) Error!Value {
    var http = try download.commonHttpClient(a, url);
    defer http.deinit();

    var headers: download.Headers = .{};
    _ = try headers.emplace(a, "User-Agent", try std.mem.concat(a, u8, &.{ "llama-cpp/", download.buildInfo() }));
    _ = try headers.emplace(a, "Accept", "application/json");

    if (isValidHfToken(token)) {
        _ = try headers.emplace(a, "Authorization", try std.mem.concat(a, u8, &.{ "Bearer ", token }));
    } else if (token.len != 0) {
        log.wrn("%s: invalid token, authentication disabled\n", .{@as([*:0]const u8, "api_get")});
    }

    const result = http.get(a, http.parts.path, &headers);
    if (result.res) |res| {
        var body: []const u8 = res.body.items;

        if (res.status == 200) {
            return json.parse(a, res.body.items);
        }
        // `common_json::parse(res->body)["error"].get<std::string>()`, any
        // throw swallowed
        if (json.parse(a, res.body.items)) |v| {
            var parsed = v;
            if (parsed.isObject() or parsed.isNull()) {
                if (parsed.index(a, "error")) |e| {
                    if (e.get([]const u8)) |s| body = s else |_| {}
                } else |_| {}
            }
        } else |_| {}

        return common.throw(error.RuntimeError, "GET failed ({d}): {s}", .{ res.status, body });
    } else {
        return common.throw(error.RuntimeError, "HTTPLIB failed: {s}", .{download.errorToString(result.err)});
    }
}

/// `catch (const common_json_error & e)` then `catch (const std::exception & e)`
/// around a hub call: the error's text logged under `func`.
fn logCaught(comptime func: [*:0]const u8, e: anyerror) void {
    const a = std.heap.c_allocator;
    switch (e) {
        error.JsonError => {
            const z = a.dupeSentinel(u8, json.what(), 0) catch return;
            defer a.free(z);
            log.err("%s: JSON error: %s\n", .{ func, z.ptr });
        },
        else => {
            const z = a.dupeSentinel(u8, common.what(e), 0) catch return;
            defer a.free(z);
            log.err("%s: error: %s\n", .{ func, z.ptr });
        },
    }
}

/// Ports `get_repo_commit` (hf-cache.cpp:206 @620d2225c): the commit of the
/// `main` branch, or of the first valid one, recorded under `refs/`; empty
/// on any failure, which is logged.
fn getRepoCommit(a: Allocator, repo_id: []const u8, token: []const u8) Allocator.Error![]const u8 {
    const func = "get_repo_commit";
    return getRepoCommitImpl(a, repo_id, token) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        logCaught(func, e);
        return "";
    };
}

fn getRepoCommitImpl(a: Allocator, repo_id: []const u8, token: []const u8) Error![]const u8 {
    const func = "get_repo_commit";
    const endpoint = try download.commonGetModelEndpoint(a);
    var js = try apiGet(a, try std.mem.concat(a, u8, &.{ endpoint, "api/models/", repo_id, "/refs" }), token);

    if (!js.isObject() or
        !js.contains("branches") or !(try js.at("branches")).isArray())
    {
        log.wrn("%s: missing 'branches' for '%s'\n", .{ func, (try zpath(a, repo_id)).ptr });
        return "";
    }

    const refs_path = try pathJoin(a, try getRepoPath(a, repo_id), "refs");
    var name: []const u8 = "";
    var commit: []const u8 = "";

    for ((try js.at("branches")).array.items) |*branch| {
        if (!branch.isObject() or
            !branch.contains("name") or !(try branch.at("name")).isString() or
            !branch.contains("targetCommit") or !(try branch.at("targetCommit")).isString())
        {
            continue;
        }
        const _name = try (try branch.at("name")).get([]const u8);
        const _commit = try (try branch.at("targetCommit")).get([]const u8);

        if (!isValidCommit(_commit)) {
            log.wrn("%s: skip invalid commit: %s\n", .{ func, (try zpath(a, _commit)).ptr });
            continue;
        }
        const candidate = _name; // fs::u8path(_name): the same bytes on POSIX

        if (!try isValidSubpath(a, refs_path, candidate)) {
            log.wrn("%s: skip invalid branch: %s\n", .{ func, (try zpath(a, _name)).ptr });
            continue;
        }

        if (std.mem.eql(u8, _name, "main")) {
            name = _name;
            commit = _commit;
            break;
        }

        if (name.len == 0 or commit.len == 0) {
            name = _name;
            commit = _commit;
        }
    }

    if (name.len == 0 or commit.len == 0) {
        log.wrn("%s: no valid branch for '%s'\n", .{ func, (try zpath(a, repo_id)).ptr });
        return "";
    }

    try download.fsWriteAtomic(a, try pathJoin(a, refs_path, name), commit);
    return try a.dupe(u8, commit);
}

/// Ports `hf_file` (hf-cache.h:10 @620d2225c).
pub const HfFile = struct {
    path: []const u8 = "",
    url: []const u8 = "",
    local_path: []const u8 = "",
    final_path: []const u8 = "",
    oid: []const u8 = "",
    repo_id: []const u8 = "",
};

/// Ports `hf_files` (hf-cache.h:19 @620d2225c).
pub const HfFiles = []HfFile;

/// Ports `get_repo_files` (hf-cache.cpp:273 @620d2225c): every file of the
/// repository at its current commit, from the hub's tree API; a blob not
/// yet in a snapshot downloads to `blobs/<oid>`. Empty on any failure,
/// which is logged.
pub fn getRepoFiles(a: Allocator, repo_id: []const u8, token: []const u8) Error!HfFiles {
    const func = "get_repo_files";
    if (!isValidRepoId(repo_id)) {
        log.wrn("%s: invalid repository: %s\n", .{ func, (try zpath(a, repo_id)).ptr });
        return &.{};
    }

    const commit = try getRepoCommit(a, repo_id, token);
    if (commit.len == 0) {
        log.wrn("%s: failed to resolve commit for %s\n", .{ func, (try zpath(a, repo_id)).ptr });
        return &.{};
    }

    const blobs_path = try pathJoin(a, try getRepoPath(a, repo_id), "blobs");
    const commit_path = try pathJoin(a, try pathJoin(a, try getRepoPath(a, repo_id), "snapshots"), commit);

    var files: std.ArrayList(HfFile) = .empty;

    getRepoFilesImpl(a, repo_id, token, commit, blobs_path, commit_path, &files) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        logCaught(func, e);
    };
    return files.items;
}

fn getRepoFilesImpl(a: Allocator, repo_id: []const u8, token: []const u8, commit: []const u8, blobs_path: []const u8, commit_path: []const u8, files: *std.ArrayList(HfFile)) Error!void {
    const func = "get_repo_files";
    const endpoint = try download.commonGetModelEndpoint(a);
    var js = try apiGet(a, try std.mem.concat(a, u8, &.{ endpoint, "api/models/", repo_id, "/tree/", commit, "?recursive=true" }), token);

    if (!js.isArray()) {
        log.wrn("%s: response is not an array for '%s'\n", .{ func, (try zpath(a, repo_id)).ptr });
        files.clearRetainingCapacity();
        return;
    }

    for (js.array.items) |*item| {
        if (!item.isObject() or
            !item.contains("type") or !(try item.at("type")).isString() or !std.mem.eql(u8, try (try item.at("type")).get([]const u8), "file") or
            !item.contains("path") or !(try item.at("path")).isString())
        {
            continue;
        }

        var file: HfFile = .{};
        file.repo_id = repo_id;
        file.path = try (try item.at("path")).get([]const u8);

        if (!try isValidSubpath(a, commit_path, file.path)) {
            log.wrn("%s: skip invalid path: %s\n", .{ func, (try zpath(a, file.path)).ptr });
            continue;
        }

        if (item.contains("lfs") and (try item.at("lfs")).isObject()) {
            const lfs = try item.at("lfs");
            if (lfs.contains("oid") and (try lfs.at("oid")).isString()) {
                file.oid = try (try lfs.at("oid")).get([]const u8);
            }
        } else if (item.contains("oid") and (try item.at("oid")).isString()) {
            file.oid = try (try item.at("oid")).get([]const u8);
        }

        if (file.oid.len != 0 and !isValidOid(file.oid)) {
            log.wrn("%s: skip invalid oid: %s\n", .{ func, (try zpath(a, file.oid)).ptr });
            continue;
        }

        file.url = try std.mem.concat(a, u8, &.{ endpoint, repo_id, "/resolve/", commit, "/", file.path });

        const final_path = try pathJoin(a, commit_path, file.path);
        file.final_path = final_path;

        if (file.oid.len != 0 and !exists(a, final_path)) {
            file.local_path = try pathJoin(a, blobs_path, file.oid);
        } else {
            file.local_path = file.final_path;
        }

        try files.append(a, file);
    }
}

/// Ports `get_cached_ref` (hf-cache.cpp:353 @620d2225c): the commit `refs/main`
/// names, or the first valid one under `refs/`.
fn getCachedRef(a: Allocator, repo_path: []const u8) Allocator.Error![]const u8 {
    const refs_path = try pathJoin(a, repo_path, "refs");
    if (!isDirectory(a, refs_path)) {
        return "";
    }
    var fallback: []const u8 = "";

    for (try listDir(a, refs_path)) |name| {
        const entry = try pathJoin(a, refs_path, name);
        if (!isRegularFile(a, entry)) {
            continue;
        }
        const commit = (try readFirstLine(a, entry)) orelse continue;
        if (commit.len == 0) continue;
        if (!isValidCommit(commit)) {
            log.wrn("%s: skip invalid commit: %s\n", .{ @as([*:0]const u8, "get_cached_ref"), (try zpath(a, commit)).ptr });
            continue;
        }
        if (std.mem.eql(u8, name, "main")) {
            return commit;
        }
        if (fallback.len == 0) {
            fallback = commit;
        }
    }
    return fallback;
}

/// `recursive_directory_iterator` over `root`: each entry's path, a
/// directory before what it holds, in `readdir`'s order; a symbolic link to
/// a directory is listed and not entered.
fn walk(a: Allocator, root: []const u8, out: *std.ArrayList([]const u8)) Allocator.Error!void {
    for (try listDir(a, root)) |name| {
        const p = try pathJoin(a, root, name);
        try out.append(a, p);
        const m = statMode(a, p, false) orelse continue;
        if (m & tc.fs_stat.IFMT == tc.fs_stat.IFDIR) try walk(a, p, out);
    }
}

/// Ports `get_cache_path` (hf-cache.cpp:64 @620d2225c): the HuggingFace hub
/// cache path, as UTF-8.
///
/// Return: process-lifetime; the error `get_cache_directory` throws when no
/// home directory is known.
pub fn getCachePath() Error![]const u8 {
    return getCacheDirectory();
}

/// Ports `get_cached_files` (hf-cache.cpp:383 @620d2225c): every file in
/// the snapshot of each cached repository's current commit, or of `repo_id`'s
/// alone.
pub fn getCachedFiles(a: Allocator, repo_id: []const u8) Error!HfFiles {
    const cache_dir = try getCacheDirectory();
    if (!exists(a, cache_dir)) {
        return &.{};
    }

    if (repo_id.len != 0 and !isValidRepoId(repo_id)) {
        log.wrn("%s: invalid repository: %s\n", .{ @as([*:0]const u8, "get_cached_files"), (try zpath(a, repo_id)).ptr });
        return &.{};
    }

    var files: std.ArrayList(HfFile) = .empty;

    for (try listDir(a, cache_dir)) |name| {
        const repo = try pathJoin(a, cache_dir, name);
        if (!isDirectory(a, repo)) {
            continue;
        }
        const snapshots_path = try pathJoin(a, repo, "snapshots");

        if (!exists(a, snapshots_path)) {
            continue;
        }
        const _repo_id = try folderNameToRepo(a, name);

        if (!isValidRepoId(_repo_id)) {
            continue;
        }
        if (repo_id.len != 0 and !std.mem.eql(u8, _repo_id, repo_id)) {
            continue;
        }
        const commit = try getCachedRef(a, repo);
        const commit_path = try pathJoin(a, snapshots_path, commit);

        if (commit.len == 0 or !isDirectory(a, commit_path)) {
            continue;
        }
        var entries: std.ArrayList([]const u8) = .empty;
        try walk(a, commit_path, &entries);
        for (entries.items) |entry| {
            if (!isRegularFile(a, entry) and !isSymlink(a, entry)) {
                continue;
            }
            const path = try lexicallyRelative(a, entry, commit_path);

            if (path.len != 0) {
                try files.append(a, .{
                    .repo_id = _repo_id,
                    .path = path,
                    .local_path = entry,
                    .final_path = entry,
                });
            }
        }
    }

    return files.items;
}

var symlinks_disabled: std.atomic.Value(bool) = .init(false);

/// Ports `finalize_file` (hf-cache.cpp:440 @620d2225c): a downloaded blob
/// linked into its snapshot (relative symbolic link), or moved, or copied
/// once links have failed.
///
/// Return: the snapshot path, owned by `a`.
pub fn finalizeFile(a: Allocator, file: *const HfFile) Allocator.Error![]const u8 {
    const func = "finalize_file";
    const local_path = file.local_path;
    const final_path = file.final_path;

    if (std.mem.eql(u8, local_path, final_path) or exists(a, final_path)) {
        return file.final_path;
    }

    if (!exists(a, local_path)) {
        return file.final_path;
    }

    _ = createDirectories(a, pathParent(final_path));

    var ec: c_int = 0;
    if (!symlinks_disabled.load(.seq_cst)) {
        // fs::relative(local_path, final_path.parent_path(), ec)
        const cl = canonical(a, local_path);
        const cb = canonical(a, pathParent(final_path));
        if (cl != null and cb != null) {
            const target = try lexicallyRelative(a, cl.?, cb.?);
            const zt = try zpath(a, target);
            const zf = try zpath(a, final_path);
            if (symlink(zt.ptr, zf.ptr) != 0) ec = errno();
        } else {
            ec = @backingInt(std.c.E.NOENT);
        }
        if (ec == 0) {
            return file.final_path;
        }
    }

    if (!symlinks_disabled.swap(true, .seq_cst)) {
        log.wrn("%s: failed to create symlink: %s\n", .{ func, strerror(ec) });
        log.wrn("%s: switching to degraded mode\n", .{func});
    }

    const zl = try zpath(a, local_path);
    const zf = try zpath(a, final_path);
    if (rename(zl.ptr, zf.ptr) != 0) {
        ec = errno();
        log.wrn("%s: failed to move file to snapshots: %s\n", .{ func, strerror(ec) });
        if (!copyFile(a, local_path, final_path)) {
            log.err("%s: failed to copy file to snapshots: %s\n", .{ func, strerror(errno()) });
        }
    }
    return file.final_path;
}

/// `std::filesystem::copy` of one regular file.
fn copyFile(a: Allocator, from: []const u8, to: []const u8) bool {
    const zf = zpath(a, from) catch return false;
    const fd = std.c.open(zf.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    var data: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n < 0) return false;
        if (n == 0) break;
        data.appendSlice(a, buf[0..@intCast(n)]) catch return false;
    }
    return writeFile(a, to, data.items);
}

/// Ports `remove_cached_repo` (hf-cache.cpp:483 @620d2225c): the whole
/// cached directory of a repository.
///
/// Return: whether anything was removed.
pub fn removeCachedRepo(a: Allocator, repo_id: []const u8) Error!bool {
    const func = "remove_cached_repo";
    if (!isValidRepoId(repo_id)) {
        log.wrn("%s: invalid repository: %s\n", .{ func, (try zpath(a, repo_id)).ptr });
        return false;
    }
    const repo_path = try getRepoPath(a, repo_id);
    const r = removeAll(a, repo_path);
    if (r.err != 0) {
        log.err("%s: failed to remove repo cache %s: %s\n", .{ func, (try zpath(a, repo_path)).ptr, strerror(r.err) });
        return false;
    }
    return r.n > 0;
}

/// `std::filesystem::read_symlink`.
pub fn readSymlink(a: Allocator, p: []const u8) ?[]u8 {
    const z = zpath(a, p) catch return null;
    defer a.free(z);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = readlink(z.ptr, &buf, buf.len);
    if (n < 0) return null;
    return a.dupe(u8, buf[0..@intCast(n)]) catch null;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "lexically_normal and lexically_relative" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("/a/c", try lexicallyNormal(a, "/a/b/../c"));
    try std.testing.expectEqualStrings("/a/b/", try lexicallyNormal(a, "/a/b/."));
    try std.testing.expectEqualStrings("/", try lexicallyNormal(a, "/.."));
    try std.testing.expectEqualStrings("../x", try lexicallyNormal(a, "../x"));
    try std.testing.expectEqualStrings(".", try lexicallyNormal(a, "a/.."));
    try std.testing.expectEqualStrings("../../blobs/x", try lexicallyRelative(a, "/c/r/blobs/x", "/c/r/snapshots/abc"));
    try std.testing.expectEqualStrings("sub/f.gguf", try lexicallyRelative(a, "/c/s/sub/f.gguf", "/c/s"));
    try std.testing.expect(try isValidSubpath(a, "/c/refs", "main"));
    try std.testing.expect(!try isValidSubpath(a, "/c/refs", "../x"));
    try std.testing.expect(isValidRepoId("ggml-org/models"));
    try std.testing.expect(!isValidRepoId("ggml-org//models"));
}
