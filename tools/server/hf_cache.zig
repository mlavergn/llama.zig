//! The Hugging Face hub cache: the layout `huggingface_hub` writes
//! (`models--<user>--<repo>/{refs,blobs,snapshots}`), what is in it, how a
//! downloaded blob becomes a snapshot entry, and the hub API calls that
//! list a repository's files.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/hf-cache.cpp` and `llama.cpp/common/hf-cache.h`
//! (v1.0.0, `b1b770d90`) whole, their POSIX arms. Each declaration names
//! the C++ it replaces and the line it began at.
//!
//! # `std::filesystem`
//!
//! The C++ leans on `std::filesystem`; its paths are `stdlib.path`'s and
//! its queries `stdlib.fs`'s. `recursive_directory_iterator` is `walk`, in
//! `readdir`'s order, which libc++'s iterators follow; it does not descend
//! into a symbolic link to a directory, as `directory_options::none` does
//! not.
//!
//! # The network
//!
//! `api_get` goes through `download.zig`'s `common_http_client`, so the
//! hub is reached over plain HTTP only: `https://huggingface.co/` -- the
//! default endpoint -- is refused with upstream's message for a build
//! without SSL, exactly as the reference refuses it. `MODEL_ENDPOINT` or
//! `HF_ENDPOINT` pointing at a plain-HTTP mirror works on both sides.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");
const tc = @import("tools_common");
const json = @import("json");

const Allocator = std.mem.Allocator;

/// A thrown exception: `RuntimeError` and `InvalidArgument` carry their
/// `what()` in `common.what`, `JsonError` in `json.what`.
pub const Error = mod.common.Error;

// -----------------------------------------------------------------------------
// hf_cache

fn zpath(a: Allocator, p: []const u8) Allocator.Error![:0]u8 {
    return a.dupeSentinel(u8, p, 0);
}

// -----------------------------------------------------------------------------
// hf_cache

/// The cache directory, decided on the first call.
var cache_dir_value: ?[]const u8 = null;
var cache_dir_mutex: stdlib.sync.Mutex = .{};

/// Ports `get_cache_directory` (hf-cache.cpp:34 @b1b770d90): the first of
/// `LLAMA_CACHE`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `$HF_HOME/hub`,
/// `$XDG_CACHE_HOME/huggingface/hub`, `$HOME/.cache/huggingface/hub` and
/// the password database's home, decided on the first call.
fn getCacheDirectory() Error![]const u8 {
    cache_dir_mutex.lock();
    defer cache_dir_mutex.unlock();
    if (cache_dir_value) |v| return v;
    const a = std.heap.c_allocator;
    const Entry = struct { var_: [:0]const u8, path: []const u8 };
    const entries = [_]Entry{
        .{ .var_ = "LLAMA_CACHE", .path = "" },
        .{ .var_ = "HF_HUB_CACHE", .path = "" },
        .{ .var_ = "HUGGINGFACE_HUB_CACHE", .path = "" },
        .{ .var_ = "HF_HOME", .path = "hub" },
        .{ .var_ = "XDG_CACHE_HOME", .path = "huggingface/hub" },
        .{ .var_ = "HOME", .path = ".cache/huggingface/hub" },
    };
    for (entries) |entry| {
        if (stdlib.env.get(entry.var_)) |base| {
            if (base.len == 0) continue;
            cache_dir_value = if (entry.path.len == 0) try a.dupe(u8, base) else try stdlib.path.join(a, base, entry.path);
            return cache_dir_value.?;
        }
    }
    if (stdlib.env.passwdHome()) |dir| {
        if (dir.len != 0) {
            cache_dir_value = try stdlib.path.join(a, try stdlib.path.join(a, dir, ".cache"), "huggingface/hub");
            return cache_dir_value.?;
        }
    }
    return mod.common.throw(error.RuntimeError, "Failed to determine HF cache directory", .{});
}

/// Ports `folder_name_to_repo` (hf-cache.cpp:69 @b1b770d90): `models--a--b`
/// as `a/b`, anything else as nothing.
fn folderNameToRepo(a: Allocator, folder: []const u8) Allocator.Error![]u8 {
    const prefix = "models--";
    if (!std.mem.startsWith(u8, folder, prefix)) {
        return a.dupe(u8, "");
    }
    return std.mem.replaceOwned(u8, a, folder[prefix.len..], "--", "/");
}

/// Ports `repo_to_folder_name` (hf-cache.cpp:79 @b1b770d90).
fn repoToFolderName(a: Allocator, repo_id: []const u8) Allocator.Error![]u8 {
    const result = try std.mem.concat(a, u8, &.{ "models--", repo_id });
    return std.mem.replaceOwned(u8, a, result, "/", "--");
}

/// Ports `get_repo_path` (hf-cache.cpp:86 @b1b770d90).
fn getRepoPath(a: Allocator, repo_id: []const u8) Error![]u8 {
    return stdlib.path.join(a, try getCacheDirectory(), try repoToFolderName(a, repo_id));
}

/// Ports `is_hex_char` (hf-cache.cpp:90 @b1b770d90).
fn isHexChar(ch: u8) bool {
    return (ch >= 'A' and ch <= 'F') or
        (ch >= 'a' and ch <= 'f') or
        (ch >= '0' and ch <= '9');
}

/// Ports `is_hex_string` (hf-cache.cpp:96 @b1b770d90).
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

/// Ports `is_alphanum` (hf-cache.cpp:108 @b1b770d90).
fn isAlphanum(ch: u8) bool {
    return (ch >= 'A' and ch <= 'Z') or
        (ch >= 'a' and ch <= 'z') or
        (ch >= '0' and ch <= '9');
}

/// Ports `is_special_char` (hf-cache.cpp:114 @b1b770d90).
fn isSpecialChar(ch: u8) bool {
    return ch == '/' or ch == '.' or ch == '-';
}

/// Ports `is_valid_repo_id` (hf-cache.cpp:121 @b1b770d90): base chars
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

/// Ports `is_valid_hf_token` (hf-cache.cpp:144 @b1b770d90).
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

/// Ports `is_valid_commit` (hf-cache.cpp:157 @b1b770d90).
fn isValidCommit(hash: []const u8) bool {
    return isHexString(hash, 40);
}

/// Ports `is_valid_oid` (hf-cache.cpp:161 @b1b770d90).
fn isValidOid(oid: []const u8) bool {
    return isHexString(oid, 40) or isHexString(oid, 64);
}

/// Ports `get_lock_path` (hf-cache.cpp:165 @b1b770d90): the lock file of a
/// blob, shared with the hf CLI: `<cache>/.locks/<repo folder>/<oid>.lock`. A
/// pure path computation: creates nothing.
///
/// Return: an empty string on an invalid repo or oid.
pub fn getLockPath(a: Allocator, repo_id: []const u8, oid: []const u8) Error![]const u8 {
    if (!isValidRepoId(repo_id) or !isValidOid(oid)) return "";
    const dir = try stdlib.path.join(a, try stdlib.path.join(a, try getCachePath(), ".locks"), try repoToFolderName(a, repo_id));
    return stdlib.path.join(a, dir, try std.mem.concat(a, u8, &.{ oid, ".lock" }));
}

/// Ports `is_valid_subpath` (hf-cache.cpp:173 @b1b770d90): `subpath`, a
/// relative path, stays under `path` once normalized.
fn isValidSubpath(a: Allocator, path: []const u8, subpath: []const u8) Allocator.Error!bool {
    if (subpath.len != 0 and subpath[0] == '/') {
        return false; // never do a / b with b absolute
    }
    const b = try stdlib.path.normal(a, try stdlib.fs.absolute(a, path));
    const t = try stdlib.path.normal(a, try stdlib.path.join(a, b, subpath));
    const be = try stdlib.path.elements(a, b);
    const te = try stdlib.path.elements(a, t);
    var i: usize = 0;
    while (i < be.len and i < te.len and std.mem.eql(u8, be[i], te[i])) i += 1;

    return i == be.len;
}

/// Ports `api_get` (hf-cache.cpp:184 @b1b770d90): a hub API GET, its body
/// parsed as JSON; any other status throws with the body's `error`, or the
/// body itself.
fn apiGet(a: Allocator, url: []const u8, token: []const u8) Error!json.Value {
    var http = try mod.download.commonHttpClient(a, url);
    defer http.deinit();

    var headers: mod.download.Headers = .{};
    _ = try headers.emplace(a, "User-Agent", try std.mem.concat(a, u8, &.{ "llama-cpp/", mod.download.buildInfo() }));
    _ = try headers.emplace(a, "Accept", "application/json");

    if (isValidHfToken(token)) {
        _ = try headers.emplace(a, "Authorization", try std.mem.concat(a, u8, &.{ "Bearer ", token }));
    } else if (token.len != 0) {
        tc.log.wrn("%s: invalid token, authentication disabled\n", .{@as([*:0]const u8, "api_get")});
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

        return mod.common.throw(error.RuntimeError, "GET failed ({d}): {s}", .{ res.status, body });
    } else {
        return mod.common.throw(error.RuntimeError, "HTTPLIB failed: {s}", .{mod.download.errorToString(result.err)});
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
            tc.log.err("%s: JSON error: %s\n", .{ func, z.ptr });
        },
        else => {
            const z = a.dupeSentinel(u8, mod.common.what(e), 0) catch return;
            defer a.free(z);
            tc.log.err("%s: error: %s\n", .{ func, z.ptr });
        },
    }
}

/// Ports `get_repo_commit` (hf-cache.cpp:215 @b1b770d90): the commit of the
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
    const endpoint = try mod.download.commonGetModelEndpoint(a);
    var js = try apiGet(a, try std.mem.concat(a, u8, &.{ endpoint, "api/models/", repo_id, "/refs" }), token);

    if (!js.isObject() or
        !js.contains("branches") or !(try js.at("branches")).isArray())
    {
        tc.log.wrn("%s: missing 'branches' for '%s'\n", .{ func, (try zpath(a, repo_id)).ptr });
        return "";
    }

    const refs_path = try stdlib.path.join(a, try getRepoPath(a, repo_id), "refs");
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
            tc.log.wrn("%s: skip invalid commit: %s\n", .{ func, (try zpath(a, _commit)).ptr });
            continue;
        }
        const candidate = _name; // fs::u8path(_name): the same bytes on POSIX

        if (!try isValidSubpath(a, refs_path, candidate)) {
            tc.log.wrn("%s: skip invalid branch: %s\n", .{ func, (try zpath(a, _name)).ptr });
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
        tc.log.wrn("%s: no valid branch for '%s'\n", .{ func, (try zpath(a, repo_id)).ptr });
        return "";
    }

    try mod.download.fsWriteAtomic(a, try stdlib.path.join(a, refs_path, name), commit);
    return try a.dupe(u8, commit);
}

/// Ports `hf_file` (hf-cache.h:11 @b1b770d90).
pub const HfFile = struct {
    path: []const u8 = "",
    url: []const u8 = "",
    local_path: []const u8 = "",
    final_path: []const u8 = "",
    oid: []const u8 = "",
    repo_id: []const u8 = "",
};

/// Ports `hf_files` (hf-cache.h:20 @b1b770d90).
pub const HfFiles = []HfFile;

/// Ports `get_repo_files` (hf-cache.cpp:282 @b1b770d90): every file of the
/// repository at its current commit, from the hub's tree API; a blob not
/// yet in a snapshot downloads to `blobs/<oid>`. Empty on any failure,
/// which is logged.
pub fn getRepoFiles(a: Allocator, repo_id: []const u8, token: []const u8) Error!HfFiles {
    const func = "get_repo_files";
    if (!isValidRepoId(repo_id)) {
        tc.log.wrn("%s: invalid repository: %s\n", .{ func, (try zpath(a, repo_id)).ptr });
        return &.{};
    }

    const commit = try getRepoCommit(a, repo_id, token);
    if (commit.len == 0) {
        tc.log.wrn("%s: failed to resolve commit for %s\n", .{ func, (try zpath(a, repo_id)).ptr });
        return &.{};
    }

    const blobs_path = try stdlib.path.join(a, try getRepoPath(a, repo_id), "blobs");
    const commit_path = try stdlib.path.join(a, try stdlib.path.join(a, try getRepoPath(a, repo_id), "snapshots"), commit);

    var files: std.ArrayList(HfFile) = .empty;

    getRepoFilesImpl(a, repo_id, token, commit, blobs_path, commit_path, &files) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        logCaught(func, e);
    };
    return files.items;
}

fn getRepoFilesImpl(a: Allocator, repo_id: []const u8, token: []const u8, commit: []const u8, blobs_path: []const u8, commit_path: []const u8, files: *std.ArrayList(HfFile)) Error!void {
    const func = "get_repo_files";
    const endpoint = try mod.download.commonGetModelEndpoint(a);
    var js = try apiGet(a, try std.mem.concat(a, u8, &.{ endpoint, "api/models/", repo_id, "/tree/", commit, "?recursive=true" }), token);

    if (!js.isArray()) {
        tc.log.wrn("%s: response is not an array for '%s'\n", .{ func, (try zpath(a, repo_id)).ptr });
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
            tc.log.wrn("%s: skip invalid path: %s\n", .{ func, (try zpath(a, file.path)).ptr });
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
            tc.log.wrn("%s: skip invalid oid: %s\n", .{ func, (try zpath(a, file.oid)).ptr });
            continue;
        }

        file.url = try std.mem.concat(a, u8, &.{ endpoint, repo_id, "/resolve/", commit, "/", file.path });

        const final_path = try stdlib.path.join(a, commit_path, file.path);
        file.final_path = final_path;

        if (file.oid.len != 0 and !stdlib.fs.exists(final_path)) {
            file.local_path = try stdlib.path.join(a, blobs_path, file.oid);
        } else {
            file.local_path = file.final_path;
        }

        try files.append(a, file);
    }
}

/// Ports `get_cached_ref` (hf-cache.cpp:362 @b1b770d90): the commit `refs/main`
/// names, or the first valid one under `refs/`.
fn getCachedRef(a: Allocator, repo_path: []const u8) Allocator.Error![]const u8 {
    const refs_path = try stdlib.path.join(a, repo_path, "refs");
    if (!stdlib.fs.isDirectory(refs_path)) {
        return "";
    }
    var fallback: []const u8 = "";

    for (try stdlib.fs.listNames(a, refs_path)) |name| {
        const entry = try stdlib.path.join(a, refs_path, name);
        if (!stdlib.fs.isRegularFile(entry)) {
            continue;
        }
        const commit = (try stdlib.file.readFirstLine(stdlib.file.defaultIo(), a, entry)) orelse continue;
        if (commit.len == 0) continue;
        if (!isValidCommit(commit)) {
            tc.log.wrn("%s: skip invalid commit: %s\n", .{ @as([*:0]const u8, "get_cached_ref"), (try zpath(a, commit)).ptr });
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
    for (try stdlib.fs.listNames(a, root)) |name| {
        const p = try stdlib.path.join(a, root, name);
        try out.append(a, p);
        const st = stdlib.fs.stat(p, false) orelse continue;
        if (st.mode & stdlib.fs.IFMT == stdlib.fs.IFDIR) try walk(a, p, out);
    }
}

/// Ports `get_cache_path` (hf-cache.cpp:65 @b1b770d90): the HuggingFace hub
/// cache path, as UTF-8.
///
/// Return: process-lifetime; the error `get_cache_directory` throws when no
/// home directory is known.
pub fn getCachePath() Error![]const u8 {
    return getCacheDirectory();
}

/// Ports `get_cached_files` (hf-cache.cpp:392 @b1b770d90): every file in
/// the snapshot of each cached repository's current commit, or of `repo_id`'s
/// alone.
pub fn getCachedFiles(a: Allocator, repo_id: []const u8) Error!HfFiles {
    const cache_dir = try getCacheDirectory();
    if (!stdlib.fs.exists(cache_dir)) {
        return &.{};
    }

    if (repo_id.len != 0 and !isValidRepoId(repo_id)) {
        tc.log.wrn("%s: invalid repository: %s\n", .{ @as([*:0]const u8, "get_cached_files"), (try zpath(a, repo_id)).ptr });
        return &.{};
    }

    var files: std.ArrayList(HfFile) = .empty;

    for (try stdlib.fs.listNames(a, cache_dir)) |name| {
        const repo = try stdlib.path.join(a, cache_dir, name);
        if (!stdlib.fs.isDirectory(repo)) {
            continue;
        }
        const snapshots_path = try stdlib.path.join(a, repo, "snapshots");

        if (!stdlib.fs.exists(snapshots_path)) {
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
        const commit_path = try stdlib.path.join(a, snapshots_path, commit);

        if (commit.len == 0 or !stdlib.fs.isDirectory(commit_path)) {
            continue;
        }
        var entries: std.ArrayList([]const u8) = .empty;
        try walk(a, commit_path, &entries);
        for (entries.items) |entry| {
            if (!stdlib.fs.isRegularFile(entry) and !stdlib.fs.isSymlink(entry)) {
                continue;
            }
            const path = try stdlib.path.relative(a, entry, commit_path);

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

/// Ports `finalize_file` (hf-cache.cpp:448 @b1b770d90): a downloaded blob
/// linked into its snapshot (relative symbolic link), or moved, or copied
/// once links have failed. A completed snapshot is returned without the
/// lock, so a read-only cache works; otherwise the blob's lock, shared with
/// huggingface_hub, is taken first.
///
/// Parameters:
/// - `keep_waiting`: polled while the lock is busy, and once it is taken.
///
/// Return: the snapshot path, owned by `a`; throws when the lock cannot be
/// taken or `keep_waiting` returns false.
pub fn finalizeFile(a: Allocator, file: *const HfFile, keep_waiting: ?mod.file_lock.KeepWaiting) Error![]const u8 {
    const func = "finalize_file";
    const local_path = file.local_path;
    const final_path = file.final_path;

    if (std.mem.eql(u8, local_path, final_path) or stdlib.fs.exists(final_path)) {
        return file.final_path;
    }

    if (!stdlib.fs.exists(local_path)) {
        return file.final_path;
    }

    // the lock file is shared with huggingface_hub: another process may be
    // finalizing the same blob
    var lock = mod.file_lock.FileLock.init(a, try getLockPath(a, file.repo_id, file.oid));
    defer lock.deinit();
    if (!try lock.acquire(keep_waiting)) {
        // without the lock, another process could move the blob while we check it
        return mod.common.throw(error.RuntimeError, "failed to lock cache for finalization: {s}", .{final_path});
    }

    // acquire() only polls keep_waiting while waiting for a busy lock
    if (keep_waiting) |kw| if (!kw.call()) {
        return mod.common.throw(error.RuntimeError, "download cancelled before finalization: {s}", .{final_path});
    };

    // re-check under the lock: the winner may have created the link already
    if (stdlib.fs.exists(final_path)) {
        return file.final_path;
    }

    _ = stdlib.fs.createDirectories(stdlib.path.parent(final_path));

    var ec: c_int = 0;
    if (!symlinks_disabled.load(.seq_cst)) {
        // fs::relative(local_path, final_path.parent_path(), ec)
        const cl = stdlib.fs.realPathAlloc(a, local_path);
        const cb = stdlib.fs.realPathAlloc(a, stdlib.path.parent(final_path));
        if (cl != null and cb != null) {
            const target = try stdlib.path.relative(a, cl.?, cb.?);
            ec = stdlib.fs.symlink(target, final_path);
        } else {
            ec = @backingInt(std.c.E.NOENT);
        }
        if (ec == 0) {
            return file.final_path;
        }
    }

    if (!symlinks_disabled.swap(true, .seq_cst)) {
        tc.log.wrn("%s: failed to create symlink: %s\n", .{ func, stdlib.errno.strerror(ec) });
        tc.log.wrn("%s: switching to degraded mode\n", .{func});
    }

    ec = stdlib.fs.rename(local_path, final_path);
    if (ec != 0) {
        tc.log.wrn("%s: failed to move file to snapshots: %s\n", .{ func, stdlib.errno.strerror(ec) });
        ec = stdlib.fs.copyFile(a, local_path, final_path);
        if (ec != 0) {
            tc.log.err("%s: failed to copy file to snapshots: %s\n", .{ func, stdlib.errno.strerror(ec) });
        }
    }
    return file.final_path;
}

/// Ports `remove_cached_repo` (hf-cache.cpp:509 @b1b770d90): the whole
/// cached directory of a repository.
///
/// Return: whether anything was removed.
pub fn removeCachedRepo(a: Allocator, repo_id: []const u8) Error!bool {
    const func = "remove_cached_repo";
    if (!isValidRepoId(repo_id)) {
        tc.log.wrn("%s: invalid repository: %s\n", .{ func, (try zpath(a, repo_id)).ptr });
        return false;
    }
    const repo_path = try getRepoPath(a, repo_id);
    const r = stdlib.fs.removeAll(a, repo_path);
    if (r.err != 0) {
        tc.log.err("%s: failed to remove repo cache %s: %s\n", .{ func, (try zpath(a, repo_path)).ptr, stdlib.errno.strerror(r.err) });
        return false;
    }
    return r.n > 0;
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
    try std.testing.expectEqualStrings("/a/c", try stdlib.path.normal(a, "/a/b/../c"));
    try std.testing.expectEqualStrings("/a/b/", try stdlib.path.normal(a, "/a/b/."));
    try std.testing.expectEqualStrings("/", try stdlib.path.normal(a, "/.."));
    try std.testing.expectEqualStrings("../x", try stdlib.path.normal(a, "../x"));
    try std.testing.expectEqualStrings(".", try stdlib.path.normal(a, "a/.."));
    try std.testing.expectEqualStrings("../../blobs/x", try stdlib.path.relative(a, "/c/r/blobs/x", "/c/r/snapshots/abc"));
    try std.testing.expectEqualStrings("sub/f.gguf", try stdlib.path.relative(a, "/c/s/sub/f.gguf", "/c/s"));
    try std.testing.expect(try isValidSubpath(a, "/c/refs", "main"));
    try std.testing.expect(!try isValidSubpath(a, "/c/refs", "../x"));
    try std.testing.expect(isValidRepoId("ggml-org/models"));
    try std.testing.expect(!isValidRepoId("ggml-org//models"));
}
