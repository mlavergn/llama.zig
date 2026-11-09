//! The model downloader: a URL (`-mu`), a Hugging Face repository (`-hf`)
//! or a Docker model (`-dr`) to a file in the cache, with `.etag`
//! revalidation, resumable `.downloadInProgress` files and three attempts;
//! what is already cached and how to forget it; and the plan that picks a
//! repository's model, mmproj and draft sidecars.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/download.cpp` and `llama.cpp/common/download.h`
//! (v0.8.0, `8642541b2`) whole, with what they and router mode need from
//! the rest of `common/`: `common_models_handler_init`, `_is_preset_repo`
//! and `_apply` and their helpers from `common/arg.cpp`; the client helpers
//! of `common/http.h` (`common_http_parse_url`, `common_http_client`,
//! `common_http_show_masked_url`, `common_http_get_free_port`); and
//! `fs_get_cache_directory`, `fs_get_cache_file`, `fs_write_atomic`,
//! `common_get_path_from_env`, `get_home_directory` and
//! `common_get_model_endpoint` from `common/common.cpp`. Each declaration names the C++ it replaces and the
//! line it began at.
//!
//! # No SSL
//!
//! The reference defines no `CPPHTTPLIB_OPENSSL_SUPPORT`, so
//! `common_http_client` refuses an `https` URL with its "HTTPS is not
//! supported. Please rebuild with one of: ..." message, and so does this:
//! `-hf` against `huggingface.co` and every Docker model fail there on both
//! sides. A plain-HTTP endpoint (`MODEL_ENDPOINT`, a `-mu` URL) downloads.
//!
//! # Threads
//!
//! `common_download_run_tasks` downloads every file at once, one
//! `std::async` thread each; so does this, one thread per task. The tasks
//! are de-duplicated through a `std::unordered_map` keyed by local path and
//! downloaded in its iteration order, which `libcxx.StringMap` reproduces.

const std = @import("std");
const builtin = @import("builtin");
const tc = @import("tools_common");
const json = @import("json");
const llama = @import("llama");
const regex = llama.regex;
const libcxx = llama.libcxx;
const common = @import("common.zig");
const hf_cache = @import("hf_cache.zig");
const client = @import("httplib/client.zig");
const file_lock = @import("file_lock.zig");

const log = tc.log;
const Params = tc.params.Params;
const ModelParams = tc.params.Model;
const SpeculativeType = tc.params.SpeculativeType;
const Allocator = std.mem.Allocator;
const HfFile = hf_cache.HfFile;

pub const Error = common.Error;
const throw = common.throw;

fn zs(a: Allocator, s: []const u8) Allocator.Error![*:0]const u8 {
    return (try a.dupeSentinel(u8, s, 0)).ptr;
}

/// Ports `llama_build_info` (build-info.cpp.in:27 @cd959bae9): `b<number>-<commit>`.
pub fn buildInfo() []const u8 {
    return @import("result.zig").build_info;
}

// -----------------------------------------------------------------------------
// common/common.cpp

/// Ports `common_get_model_endpoint` (common/common.cpp:1580 @cd959bae9):
/// `MODEL_ENDPOINT`, else `HF_ENDPOINT`, else the hub, ending in `/`.
pub fn commonGetModelEndpoint(a: Allocator) Allocator.Error![]const u8 {
    var endpoint: []const u8 = getEnv("MODEL_ENDPOINT");
    if (endpoint.len == 0) {
        // the HF_ENDPOINT variable is respected for backward compatibility
        endpoint = getEnv("HF_ENDPOINT");
    }
    if (endpoint.len == 0) {
        return "https://huggingface.co/";
    }
    if (endpoint[endpoint.len - 1] != '/') {
        return std.mem.concat(a, u8, &.{ endpoint, "/" });
    }
    return endpoint;
}

/// Ports `common_get_env` (common/common.cpp:969 @cd959bae9).
fn getEnv(name: [*:0]const u8) []const u8 {
    const v = std.c.getenv(name) orelse return "";
    return std.mem.span(v);
}

/// Ports `common_get_path_from_env` (common/common.cpp:986 @cd959bae9),
/// its POSIX arm: an unset variable gives an empty path.
fn getPathFromEnv(name: [*:0]const u8) []const u8 {
    return getEnv(name);
}

/// Ports `get_home_directory` (common/common.cpp:998 @cd959bae9): `HOME`,
/// else the password database's home.
fn getHomeDirectory() Error![]const u8 {
    const home = getPathFromEnv("HOME");
    if (home.len != 0) {
        return home;
    }
    const pw = std.c.getpwuid(std.c.getuid());
    const dir = if (pw) |p| p.dir else null;
    if (dir == null or dir.?[0] == 0) {
        return throw(error.RuntimeError, "Failed to find $HOME directory", .{});
    }
    return std.mem.span(dir.?);
}

/// Ports `fs_get_cache_directory` (common/common.cpp:1011 @cd959bae9), its
/// macOS and Linux arms: `LLAMA_CACHE` as it is, else the platform's cache
/// directory with `llama.cpp` appended -- a path, with no trailing
/// separator added.
pub fn fsGetCacheDirectory(a: Allocator) Error![]const u8 {
    var cache_directory: []const u8 = getPathFromEnv("LLAMA_CACHE");
    if (cache_directory.len != 0) {
        return cache_directory;
    }
    if (comptime builtin.os.tag.isDarwin()) {
        cache_directory = try hf_cache.pathJoin(a, try getHomeDirectory(), "Library/Caches");
    } else {
        cache_directory = getPathFromEnv("XDG_CACHE_HOME");
        if (cache_directory.len == 0) {
            cache_directory = try hf_cache.pathJoin(a, try getHomeDirectory(), ".cache");
        }
    }
    return hf_cache.pathJoin(a, cache_directory, "llama.cpp");
}

/// Ports `fs_get_cache_file` (common/common.cpp:1037 @8642541b2): `filename`
/// in the cache directory, which is created.
pub fn fsGetCacheFile(a: Allocator, filename: []const u8) Error![]const u8 {
    common.ggmlAssert(std.mem.indexOfScalar(u8, filename, '/') == null, "common/common.cpp", 1049, "filename.find(DIRECTORY_SEPARATOR) == std::string::npos");
    const cache_directory = try fsGetCacheDirectory(a);
    // `common_create_directories` (common/common.h:933 @cd959bae9): on Linux through
    // `path / ""`, which makes no difference here
    if (hf_cache.createDirectories(a, cache_directory) != 0) {
        return throw(error.RuntimeError, "failed to create cache directory: {s}", .{cache_directory});
    }
    return hf_cache.pathJoin(a, cache_directory, filename);
}

/// Ports `fs_write_atomic` (common/common.cpp:926 @8642541b2): through a
/// temporary file renamed into place, the parent directories created first
/// (their failure ignored). A random suffix gives every writer its own
/// temporary file, across processes and containers.
pub fn fsWriteAtomic(a: Allocator, path: []const u8, data: []const u8) Error!void {
    var id_bytes: [8]u8 = undefined;
    @import("httplib/ws.zig").fillRandomBytes(&id_bytes);
    const path_tmp = try std.fmt.allocPrint(a, "{s}.{d}.tmp", .{ path, std.mem.readInt(u64, &id_bytes, .little) });

    const parent = hf_cache.pathParent(path);
    if (parent.len != 0) {
        _ = hf_cache.createDirectories(a, parent);
    }

    const fail = !hf_cache.writeFile(a, path_tmp, data);

    var ec = false;
    if (!fail) {
        ec = cRename(try zs(a, path_tmp), try zs(a, path)) != 0;
    }

    if (fail or ec) {
        _ = hf_cache.removeFile(a, path_tmp);
        return throw(error.RuntimeError, "failed to write file: {s}", .{path});
    }
}

// -----------------------------------------------------------------------------
// common/http.h

/// Ports `common_http_url` (http.h:17 @cd959bae9).
pub const HttpUrl = struct {
    scheme: []const u8 = "",
    user: []const u8 = "",
    password: []const u8 = "",
    host: []const u8 = "",
    port: c_int = 0,
    path: []const u8 = "",
};

/// Ports `common_http_format_host` (http.h:27 @cd959bae9): an IPv6 literal
/// bracketed for a URL's authority.
pub fn commonHttpFormatHost(a: Allocator, host: []const u8) Allocator.Error![]const u8 {
    if (std.mem.indexOfScalar(u8, host, ':') != null) return std.mem.concat(a, u8, &.{ "[", host, "]" });
    return host;
}

/// Ports `common_http_parse_url` (http.h:31 @cd959bae9).
pub fn commonHttpParseUrl(a: Allocator, url: []const u8) Error!HttpUrl {
    var parts: HttpUrl = .{};
    const scheme_end = std.mem.indexOf(u8, url, "://") orelse {
        return throw(error.RuntimeError, "invalid URL: no scheme", .{});
    };
    parts.scheme = url[0..scheme_end];

    if (!std.mem.eql(u8, parts.scheme, "http") and !std.mem.eql(u8, parts.scheme, "https")) {
        return throw(error.RuntimeError, "unsupported URL scheme: {s}", .{parts.scheme});
    }

    var rest = url[scheme_end + 3 ..];

    if (std.mem.indexOfScalar(u8, rest, '@')) |at_pos| {
        const auth = rest[0..at_pos];
        if (std.mem.indexOfScalar(u8, auth, ':')) |colon_pos| {
            parts.user = auth[0..colon_pos];
            parts.password = auth[colon_pos + 1 ..];
        } else {
            parts.user = auth;
        }
        rest = rest[at_pos + 1 ..];
    }

    if (std.mem.indexOfScalar(u8, rest, '/')) |slash_pos| {
        parts.host = rest[0..slash_pos];
        parts.path = rest[slash_pos..];
    } else {
        parts.host = rest;
        parts.path = "/";
    }

    // split the authority into host and optional port, a bracketed IPv6 literal keeps its inner colons (RFC 3986)
    var port_str: []const u8 = "";
    if (parts.host.len != 0 and parts.host[0] == '[') {
        const close = std.mem.indexOfScalar(u8, parts.host, ']') orelse {
            return throw(error.RuntimeError, "invalid IPv6 URL authority: {s}", .{parts.host});
        };
        const after = parts.host[close + 1 ..];
        if (after.len != 0 and after[0] == ':') {
            port_str = after[1..];
        }
        parts.host = parts.host[1..close];
    } else if (std.mem.indexOfScalar(u8, parts.host, ':')) |colon_pos| {
        port_str = parts.host[colon_pos + 1 ..];
        parts.host = parts.host[0..colon_pos];
    }

    if (port_str.len != 0) {
        const z = try a.dupeSentinel(u8, port_str, 0);
        parts.port = tc.arg.stoi(z) catch return throw(error.InvalidArgument, "{s}", .{tc.arg.what()});
    } else if (std.mem.eql(u8, parts.scheme, "http")) {
        parts.port = 80;
    } else if (std.mem.eql(u8, parts.scheme, "https")) {
        parts.port = 443;
    } else {
        return throw(error.RuntimeError, "unsupported URL scheme: {s}", .{parts.scheme});
    }

    return parts;
}

/// Ports `common_http_is_loopback` (http.h:105 @cd959bae9): a loopback
/// destination; a remote proxy can never reach the client machine's own
/// loopback, so these hosts are always contacted directly, before `NO_PROXY`
/// is consulted (similar to Go's net/http).
fn commonHttpIsLoopback(a: Allocator, host: []const u8) Allocator.Error!bool {
    const h = try asciiLower(a, host);

    if (std.mem.eql(u8, h, "localhost") or std.mem.eql(u8, h, "::1") or std.mem.eql(u8, h, "0.0.0.0")) {
        return true;
    }

    if (h.len > 10 and std.mem.eql(u8, h[h.len - 10 ..], ".localhost")) {
        return true;
    }

    // IPv4 literal: exactly four all-digit octets with values 0-255, loopback is 127.0.0.0/8
    var octets: [4]c_int = @splat(0);
    var pos: usize = 0;
    var n: usize = 0;
    var complete = false;
    while (n < 4 and pos <= h.len) {
        const dot = std.mem.indexOfScalarPos(u8, h, pos, '.') orelse h.len;
        if (dot == pos or dot - pos > 3) {
            return false;
        }
        var val: c_int = 0;
        for (h[pos..dot]) |ch| {
            if (ch < '0' or ch > '9') {
                return false;
            }
            val = val * 10 + (ch - '0');
        }
        if (val > 255) {
            return false;
        }
        octets[n] = val;
        n += 1;
        if (dot == h.len) {
            complete = n == 4;
            break;
        }
        pos = dot + 1;
    }

    return complete and octets[0] == 127;
}

/// The `lower` lambda of `common_http_is_loopback` and
/// `common_http_proxy_bypass`.
fn asciiLower(a: Allocator, s: []const u8) Allocator.Error![]u8 {
    const out = try a.dupe(u8, s);
    for (out) |*ch| {
        if (ch.* >= 'A' and ch.* <= 'Z') ch.* += 'a' - 'A';
    }
    return out;
}

/// Ports `common_http_proxy_bypass` (http.h:161 @cd959bae9): a host against a
/// `NO_PROXY` list: comma separated entries, `*` bypasses the proxy for every
/// host, an entry matches the host itself or any of its subdomains, a
/// leading dot is optional, ports are ignored.
fn commonHttpProxyBypass(a: Allocator, no_proxy: []const u8, host: []const u8) Allocator.Error!bool {
    const h = try asciiLower(a, host);

    var pos: usize = 0;
    while (pos < no_proxy.len) {
        const end = std.mem.indexOfScalarPos(u8, no_proxy, pos, ',') orelse no_proxy.len;

        var entry: []const u8 = try asciiLower(a, no_proxy[pos..end]);
        pos = end + 1;

        entry = std.mem.trim(u8, entry, " \t");
        if (entry.len == 0) {
            continue;
        }

        if (std.mem.eql(u8, entry, "*")) {
            return true;
        }

        if (entry[0] == '.') {
            entry = entry[1..];
        }

        if (entry.len == 0) {
            continue;
        }

        if (std.mem.eql(u8, h, entry)) {
            return true;
        }

        if (h.len > entry.len and h[h.len - entry.len - 1] == '.' and std.mem.endsWith(u8, h, entry)) {
            return true;
        }
    }

    return false;
}

/// The `getenv_s` lambda of `common_http_client`: the upper-case name first.
fn getenvS(upper: [*:0]const u8, lower: [*:0]const u8) []const u8 {
    const v = std.c.getenv(upper) orelse std.c.getenv(lower) orelse return "";
    return std.mem.span(v);
}

/// Ports `common_http_client` (http.h:214 @cd959bae9): a client for the
/// URL's host, following redirects, with its credentials, through the proxy
/// `HTTP_PROXY` names. `https` is refused: the reference builds without SSL.
pub fn commonHttpClient(a: Allocator, url: []const u8) Error!HttpClient {
    const parts = try commonHttpParseUrl(a, url);

    if (parts.host.len == 0) {
        return throw(error.RuntimeError, "error: invalid URL format", .{});
    }

    // #ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (std.mem.eql(u8, parts.scheme, "https")) {
        return throw(error.RuntimeError, "HTTPS is not supported. Please rebuild with one of:\n" ++
            "  -DLLAMA_BUILD_BORINGSSL=ON\n" ++
            "  -DLLAMA_BUILD_LIBRESSL=ON\n" ++
            "  -DLLAMA_OPENSSL=ON (default, requires OpenSSL dev files installed)", .{});
    }

    var cli = try HttpClient.init(a, parts);

    if (parts.user.len != 0) {
        cli.setBasicAuth(parts.user, parts.password);
    }

    cli.setFollowLocation(true);

    // Honor HTTP_PROXY / HTTPS_PROXY environment variables (both lower- and upper-case).
    const proxy_url = getenvS("HTTP_PROXY", "http_proxy");

    if (proxy_url.len != 0 and !try commonHttpIsLoopback(a, parts.host) and
        !try commonHttpProxyBypass(a, getenvS("NO_PROXY", "no_proxy"), parts.host))
    {
        if (commonHttpParseUrl(a, proxy_url)) |proxy| {
            cli.setProxy(proxy.host, proxy.port);
            if (proxy.user.len != 0) {
                cli.setProxyBasicAuth(proxy.user, proxy.password);
            }
        } else |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            // fall back to a direct connection, the URL itself is not logged as it can carry credentials
            log.wrn("%s: ignoring malformed proxy URL from the environment: %s\n", .{ @as([*:0]const u8, "common_http_client"), try zs(a, common.what(e)) });
        }
    }

    return cli;
}

/// Ports `common_http_show_masked_url` (http.h:270 @cd959bae9): the URL
/// without its credentials.
pub fn commonHttpShowMaskedUrl(a: Allocator, parts: *const HttpUrl) Allocator.Error![]const u8 {
    return std.mem.concat(a, u8, &.{ parts.scheme, "://", if (parts.user.len == 0) "" else "****:****@", try commonHttpFormatHost(a, parts.host), parts.path });
}

/// Ports `common_http_get_free_port` (http.h:274 @cd959bae9), its POSIX
/// arm: a port the kernel picks for a socket bound to any address.
///
/// Return: the port, or -1.
pub fn commonHttpGetFreePort() c_int {
    const sock = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (sock < 0) {
        return -1;
    }
    defer _ = std.c.close(sock);

    var serv_addr: std.c.sockaddr.in = .{ .port = 0, .addr = 0 };
    serv_addr.family = std.c.AF.INET;

    if (std.c.bind(sock, @ptrCast(&serv_addr), @sizeOf(std.c.sockaddr.in)) != 0) {
        return -1;
    }

    var namelen: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
    if (std.c.getsockname(sock, @ptrCast(&serv_addr), &namelen) != 0) {
        return -1;
    }

    return std.mem.bigToNative(u16, serv_addr.port);
}

// -----------------------------------------------------------------------------
// download.h

/// Ports `common_download_progress` (download.h:14 @cd959bae9).
pub const Progress = struct {
    url: []const u8 = "",
    downloaded: usize = 0,
    total: usize = 0,
    cached: bool = false,
};

/// Ports `common_download_callback` (download.h:21 @cd959bae9): its four
/// virtual members.
pub const Callback = struct {
    ctx: *anyopaque,
    on_start: *const fn (ctx: *anyopaque, p: *const Progress) void,
    on_update: *const fn (ctx: *anyopaque, p: *const Progress) void,
    on_done: *const fn (ctx: *anyopaque, p: *const Progress, ok: bool) void,
    is_cancelled: *const fn (ctx: *anyopaque) bool = notCancelled,

    fn notCancelled(_: *anyopaque) bool {
        return false;
    }
};

/// Ports `common_header` and `common_header_list` (download.h:11, 12
/// @cd959bae9).
pub const HeaderList = []const struct { []const u8, []const u8 };

/// Ports `common_remote_params` (download.h:30 @cd959bae9).
pub const RemoteParams = struct {
    headers: HeaderList = &.{},
    timeout: c_long = 0, // in seconds, 0 means no timeout
    max_size: c_long = 0, // unlimited if 0
};

/// Ports `common_cached_model_info` (download.h:45 @cd959bae9).
pub const CachedModelInfo = struct {
    repo: []const u8,
    tag: []const u8,

    /// Ports `to_string` (download.h:48 @cd959bae9), a member of
    /// `common_cached_model_info`.
    pub fn toString(self: CachedModelInfo, a: Allocator) Allocator.Error![]const u8 {
        return std.mem.concat(a, u8, &.{ self.repo, ":", self.tag });
    }
};

/// Ports `common_download_opts` (download.h:54 @cd959bae9).
pub const Opts = struct {
    bearer_token: []const u8 = "",
    headers: HeaderList = &.{},
    offline: bool = false,
    download_mmproj: bool = false,
    download_mtp: bool = false,
    download_eagle3: bool = false,
    download_dflash: bool = false,
    download_dspark: bool = false,
    callback: ?*const Callback = null,
};

/// `std::function<void()> on_done` of a task: what it sets once its file
/// is in place.
pub const OnDone = struct {
    ctx: ?*anyopaque = null,
    func: ?*const fn (ctx: ?*anyopaque) anyerror!void = null,

    fn call(self: OnDone) anyerror!void {
        if (self.func) |f| try f(self.ctx);
    }
};

/// Ports `common_download_task` (download.h:66 @cd959bae9).
pub const Task = struct {
    opts: Opts = .{},
    url: []const u8 = "",
    local_path: []const u8 = "",
    on_done: OnDone = .{},
    is_hf: bool = false,
    lock_path: []const u8 = "",
    alt_path: []const u8 = "", // HF snapshot path, also treated as cached (the blob may be moved there)

    /// Ports the constructor taking an `hf_file` (download.h:76
    /// @cd959bae9).
    pub fn fromHf(a: Allocator, f: *const HfFile, opts: Opts, on_done: OnDone) Error!Task {
        return .{
            .opts = opts,
            .url = f.url,
            .local_path = f.local_path,
            .on_done = on_done,
            .is_hf = true,
            .lock_path = try hf_cache.getLockPath(a, f.repo_id, f.oid),
            .alt_path = f.final_path,
        };
    }
};

/// Ports `common_download_hf_plan` (download.h:125 @cd959bae9).
pub const HfPlan = struct {
    primary: HfFile = .{},
    model_files: []HfFile = &.{},
    mmproj: HfFile = .{},
    mtp: HfFile = .{},
    eagle3: HfFile = .{},
    dflash: HfFile = .{},
    dspark: HfFile = .{},
    preset: HfFile = .{}, // if set, only this file is downloaded
};

/// Ports `common_download_sidecar` (download.h:139 @cd959bae9): a sidecar
/// kind a download can pull next to its model: mmproj, and the draft heads of
/// `--spec-type draft-<name>`. `opt` and `file` name the fields of `Opts` and
/// `HfPlan`, the C's pointers to members.
pub const Sidecar = struct {
    name: []const u8, // the kind, as the API names it
    keyword: []const u8, // the part of a file name that marks a file of this kind
    draft: bool, // a draft head, resolved at the tag even without a model
    opt: []const u8, // the option that requests it
    file: []const u8, // the plan file it resolves to
};

/// Ports `common_download_partial` (download.h:159 @cd959bae9): a file a
/// plan still has to download, at the path its partial takes while it is
/// incomplete; `kind` is the sidecar it belongs to, empty for the model
/// files.
pub const Partial = struct {
    kind: []const u8 = "",
    path: []const u8 = "",
};

// -----------------------------------------------------------------------------
// download.cpp

/// Ports `PARTIAL_SUFFIX` (download.cpp:44 @cd959bae9): the suffix of the
/// file a download writes until it completes.
const partial_suffix = ".downloadInProgress";

/// Ports `write_etag` (download.cpp:46 @cd959bae9).
fn writeEtag(a: Allocator, path: []const u8, etag: []const u8) Error!void {
    const etag_path = try std.mem.concat(a, u8, &.{ path, ".etag" });
    try fsWriteAtomic(a, etag_path, etag);
    log.dbg("%s: file etag saved: %s\n", .{ @as([*:0]const u8, "write_etag"), try zs(a, etag_path) });
}

/// Ports `read_etag` (download.cpp:52 @cd959bae9): the first line of
/// `<path>.etag`, or nothing.
fn readEtag(a: Allocator, path: []const u8) Allocator.Error![]const u8 {
    const etag_path = try std.mem.concat(a, u8, &.{ path, ".etag" });
    if (!hf_cache.exists(a, etag_path)) {
        return "";
    }
    const etag = (try hf_cache.readFirstLine(a, etag_path)) orelse {
        log.err("%s: could not open .etag file for reading: %s\n", .{ @as([*:0]const u8, "read_etag"), try zs(a, etag_path) });
        return "";
    };
    return etag;
}

/// Ports `is_http_status_ok` (download.cpp:67 @cd959bae9).
fn isHttpStatusOk(status: c_int) bool {
    return status >= 200 and status < 400;
}

/// Ports `common_download_split_repo_tag` (download.cpp:71 @cd959bae9):
/// `user/model:tag` into the repository and the tag, which may be empty.
pub fn splitRepoTag(a: Allocator, hf_repo_with_tag: []const u8) Error!struct { []const u8, []const u8 } {
    const parts = try tc.strings.split(a, hf_repo_with_tag, ':');
    const tag: []const u8 = if (parts.items.len > 1) parts.items[parts.items.len - 1] else "";
    const hf_repo = parts.items[0];
    const sp = try tc.strings.split(a, hf_repo, '/');
    if (sp.items.len != 2) {
        return throw(error.InvalidArgument, "error: invalid HF repo format, expected <user>/<model>[:quant]\n", .{});
    }
    return .{ hf_repo, tag };
}

/// Ports `url_get_filename` (download.cpp:82 @cd959bae9): the file name from a
/// URL, without the query string.
fn urlGetFilename(url: []const u8) []const u8 {
    var f = url;
    if (std.mem.lastIndexOfScalar(u8, f, '/')) |pos| {
        f = f[pos + 1 ..];
    }
    if (std.mem.indexOfScalar(u8, f, '?')) |pos| {
        f = f[0..pos];
    }
    return f;
}

/// Ports `ProgressBar` (download.cpp:93 @cd959bae9): one line per file
/// being downloaded, redrawn in place, when stdout is a terminal.
const ProgressBar = struct {
    var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
    var lines: std.AutoHashMapUnmanaged(*const ProgressBar, c_int) = .empty;
    var max_line: c_int = 0;

    filename: std.ArrayList(u8) = .empty,
    len: usize = 0,

    fn cleanup(line: *const ProgressBar) void {
        _ = lines.remove(line);
        if (lines.count() == 0) {
            max_line = 0;
        }
    }

    fn isOutputATty() bool {
        return std.c.isatty(1) != 0;
    }

    fn callback(self: *ProgressBar) Callback {
        return .{ .ctx = self, .on_start = onStart, .on_update = onUpdate, .on_done = onDone };
    }

    fn deinit(self: *ProgressBar) void {
        self.filename.deinit(std.heap.c_allocator);
    }

    fn onStart(ctx: *anyopaque, p: *const Progress) void {
        const self: *ProgressBar = @ptrCast(@alignCast(ctx));
        const a = std.heap.c_allocator;
        const f = urlGetFilename(p.url);

        self.filename.clearRetainingCapacity();
        self.filename.appendSlice(a, f) catch return;
        var i: usize = 0;
        while (i < self.filename.items.len) : (i += 1) {
            if ((self.filename.items[i] & 0xC0) != 0x80) {
                const was = self.len;
                self.len += 1;
                if (was == 39) {
                    self.filename.shrinkRetainingCapacity(i);
                    self.filename.appendSlice(a, "…") catch |err| switch (err) {
                        error.OutOfMemory => {},
                    };
                    break;
                }
            }
        }
    }

    fn onDone(ctx: *anyopaque, _: *const Progress, _: bool) void {
        const self: *ProgressBar = @ptrCast(@alignCast(ctx));
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        cleanup(self);
    }

    fn onUpdate(ctx: *anyopaque, p: *const Progress) void {
        const self: *ProgressBar = @ptrCast(@alignCast(ctx));
        if (p.total == 0 or !isOutputATty()) {
            return;
        }

        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);

        const io = tc.io;
        if (!lines.contains(self)) {
            lines.put(std.heap.c_allocator, self, max_line) catch return;
            max_line += 1;
            io.printf("\n", .{});
        }
        const lines_up = max_line - lines.get(self).?;

        const bar: usize = (55 - self.len) * 2;
        const pct: usize = (100 * p.downloaded) / p.total;
        const pos: usize = (bar * p.downloaded) / p.total;

        if (lines_up > 0) {
            io.printf("\x1b[%dA", .{lines_up});
        }
        io.printf("\rDownloading %.*s ", .{ @as(c_int, @intCast(self.filename.items.len)), self.filename.items.ptr });

        var i: usize = 0;
        while (i < bar) : (i += 2) {
            io.printf("%s", .{@as([*:0]const u8, if (i + 1 < pos) "─" else if (i < pos) "╴" else " ")});
        }
        io.printf("%4zu%%\x1b[K", .{pct});

        if (lines_up > 0) {
            io.printf("\x1b[%dB", .{lines_up});
        }
        io.printf("\r", .{});
        io.flushStdout();

        if (p.downloaded == p.total) {
            cleanup(self);
        }
    }
};

/// The two `httplib::Client::Get` handlers of `common_pull_file`.
const PullState = struct {
    a: Allocator,
    fd: c_int,
    fd_open: bool = true,
    path_tmp: []const u8,
    p: *Progress,
    callback: ?*const Callback,
    progress_step: usize = 0,

    fn onResponse(ctx: *anyopaque, response: *const Response) bool {
        const self: *PullState = @ptrCast(@alignCast(ctx));
        const func = "common_pull_file";
        const p = self.p;
        if (p.downloaded > 0 and response.status != 206) {
            log.wrn("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", .{ func, response.status });
            if (response.status == 416 or response.status == 200) {
                // the kept partial cannot be resumed: 416 for a complete or stale one,
                // 200 from a server that ignores the range; drop it, so the next
                // attempt starts over instead of wedging on the same answer
                _ = std.c.close(self.fd);
                self.fd_open = false;
                _ = cRemove(zs(self.a, self.path_tmp) catch return false);
            }
            return false;
        }
        if (p.downloaded == 0 and response.status != 200) {
            log.wrn("%s: download received non-successful status code: %d\n", .{ func, response.status });
            return false;
        }
        if (p.total == 0 and response.hasHeader("Content-Length")) {
            if (stoull(self.a, response.getHeaderValue("Content-Length"))) |content_length| {
                p.total = p.downloaded + content_length;
            } else |_| {
                log.wrn("%s: invalid Content-Length header: %s\n", .{ func, @as([*:0]const u8, "stoull: no conversion") });
            }
        }
        return true;
    }

    fn onData(ctx: *anyopaque, data: []const u8) bool {
        const self: *PullState = @ptrCast(@alignCast(ctx));
        const func = "common_pull_file";
        var off: usize = 0;
        while (off < data.len) {
            const n = std.c.write(self.fd, data[off..].ptr, data.len - off);
            if (n <= 0) {
                log.err("%s: error writing to file: %s\n", .{ func, zs(self.a, self.path_tmp) catch "" });
                return false;
            }
            off += @intCast(n);
        }
        const p = self.p;
        p.downloaded += data.len;
        self.progress_step += data.len;

        if (self.progress_step >= p.total / 1000 or p.downloaded == p.total) {
            if (self.callback) |cb| {
                cb.on_update(cb.ctx, p);
                if (cb.is_cancelled(cb.ctx) and (p.total == 0 or p.downloaded != p.total)) {
                    return false;
                }
            }
            self.progress_step = 0;
        }
        return true;
    }
};

/// `std::stoull` on a header value; `error.Throw` where it throws (no
/// digits, out of range).
fn stoull(a: Allocator, s: []const u8) !usize {
    const z = try a.dupeSentinel(u8, s, 0);
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtoull(z.ptr, &end, 10);
    if (end == z.ptr) return error.Throw;
    if (std.c._errno().* == @backingInt(std.c.E.RANGE)) return error.Throw;
    return @intCast(v);
}
const c_stdio = struct {
    extern "c" fn remove(path: [*:0]const u8) c_int;
    extern "c" fn rename(old: [*:0]const u8, new: [*:0]const u8) c_int;
};
fn cRemove(p: [*:0]const u8) c_int {
    return c_stdio.remove(p);
}
fn cRename(o: [*:0]const u8, n: [*:0]const u8) c_int {
    return c_stdio.rename(o, n);
}
extern fn strtoull(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_ulonglong;

/// Ports `common_pull_file` (download.cpp:175 @cd959bae9): a GET appended
/// to `path_tmp`, from `p.downloaded` with a `Range` when the server takes
/// ranges.
///
/// Return: false, having logged why, when the request did not complete or
/// the file did not close cleanly.
fn commonPullFile(a: Allocator, cli: *HttpClient, resolve_path: []const u8, path_tmp: []const u8, supports_ranges: bool, p: *Progress, callback: ?*const Callback) Allocator.Error!bool {
    const func = "common_pull_file";
    const z = try a.dupeSentinel(u8, path_tmp, 0);
    const fd = std.c.open(z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o666));
    if (fd < 0) {
        log.err("%s: error opening local file for writing: %s\n", .{ func, z.ptr });
        return false;
    }
    var state: PullState = .{ .a = a, .fd = fd, .path_tmp = path_tmp, .p = p, .callback = callback };
    defer if (state.fd_open) {
        _ = std.c.close(fd);
    };

    var headers: Headers = .{};
    if (supports_ranges and p.downloaded > 0) {
        _ = try headers.emplace(a, "Range", try std.fmt.allocPrint(a, "bytes={d}-", .{p.downloaded}));
    }

    const res = cli.getStream(a, resolve_path, &headers, &state, PullState.onResponse, PullState.onData);

    if (res.res == null) {
        log.err("%s: download failed: %s (status: %d)\n", .{ func, (try zs(a, errorToString(res.err))), @as(c_int, -1) });
        return false;
    }

    state.fd_open = false;
    if (std.c.close(fd) != 0) {
        log.err("%s: error closing file: %s\n", .{ func, z.ptr });
        return false;
    }

    return true;
}

/// What `common_download_file_single_online`'s lambdas capture.
const OnlineState = struct {
    a: Allocator,
    url: []const u8,
    opts: *const Opts,
    path: []const u8,
    alt_path: []const u8,
    lock: *file_lock.FileLock,
    p: *Progress,
    waited: bool = false,

    /// The `finish` lambda: the completion callback may itself take the same
    /// lock (the HF cache finalizes blobs under it), so every exit goes
    /// through here.
    fn finish(self: *const OnlineState, status: c_int) c_int {
        self.lock.close();
        if (self.opts.callback) |cb| {
            cb.on_done(cb.ctx, self.p, isHttpStatusOk(status));
        }
        return status;
    }

    /// The `cache_hit` lambda: `alt_path` is the HF snapshot, where the blob
    /// may have been moved when symlinks are unavailable.
    fn cacheHit(self: *const OnlineState) bool {
        return hf_cache.exists(self.a, self.path) or
            (self.alt_path.len != 0 and hf_cache.exists(self.a, self.alt_path));
    }

    /// The lambda `lock.acquire` polls.
    fn keepWaiting(ctx: ?*anyopaque) bool {
        const self: *OnlineState = @ptrCast(@alignCast(ctx.?));
        if (!self.waited) {
            const name = zs(self.a, urlGetFilename(self.url)) catch "";
            log.inf("%s: file '%s' is being downloaded by another process, waiting...\n", .{ @as([*:0]const u8, "operator()"), name });
            self.waited = true;
        }
        if (self.opts.callback) |cb| return !cb.is_cancelled(cb.ctx);
        return true;
    }
};

/// Ports `common_download_file_single_online` (download.cpp:264
/// @cd959bae9): one file, under a lock another process downloading it also
/// takes, revalidated against its `.etag` by a HEAD, then up to three
/// attempts, each resuming what the last one left.
///
/// Parameters:
/// - `lock_path`: when empty, `<path>.lock`.
/// - `alt_path`: an additional path treated as present in the cache.
///
/// Return: the HEAD's status, 304 for a cached file, or -1.
fn downloadFileSingleOnline(a: Allocator, url: []const u8, path: []const u8, opts: *const Opts, skip_etag: bool, lock_path: []const u8, alt_path: []const u8) Error!c_int {
    const func = "common_download_file_single_online";
    const max_attempts = 3;
    const retry_delay_seconds = 2;

    const path_temporary = try std.mem.concat(a, u8, &.{ path, partial_suffix });
    const path_lock = if (lock_path.len == 0) try std.mem.concat(a, u8, &.{ path, ".lock" }) else lock_path;

    var p: Progress = .{};
    p.url = url;

    if (opts.callback) |cb| {
        cb.on_start(cb.ctx, &p);
    }

    var lock = file_lock.FileLock.init(a, path_lock);
    defer lock.deinit();
    var st: OnlineState = .{ .a = a, .url = url, .opts = opts, .path = path, .alt_path = alt_path, .lock = &lock, .p = &p };

    if (skip_etag and st.cacheHit()) {
        log.dbg("%s: using cached file: %s\n", .{ func, try zs(a, path) });
        return st.finish(304); // 304 Not Modified - fake cached response
    }

    // take the lock before touching the cache, so two processes cannot race on the same file
    if (!try lock.acquire(.{ .ctx = &st, .func = OnlineState.keepWaiting })) {
        return st.finish(-1);
    }

    // under the lock: another process may have finished while we waited
    const file_exists = st.cacheHit();

    if (file_exists and skip_etag) {
        log.dbg("%s: using cached file: %s\n", .{ func, try zs(a, path) });
        return st.finish(304); // 304 Not Modified - fake cached response
    }

    var cli = try commonHttpClient(a, url);
    defer cli.deinit();
    const parts = cli.parts;

    var headers: Headers = .{};
    for (opts.headers) |h| {
        _ = try headers.emplace(a, h[0], h[1]);
    }
    if (headers.find("User-Agent") == null) {
        _ = try headers.emplace(a, "User-Agent", try std.mem.concat(a, u8, &.{ "llama-cpp/", buildInfo() }));
    }
    if (opts.bearer_token.len != 0) {
        _ = try headers.emplace(a, "Authorization", try std.mem.concat(a, u8, &.{ "Bearer ", opts.bearer_token }));
    }
    cli.setDefaultHeaders(headers);

    var last_etag: []const u8 = "";
    if (file_exists) {
        last_etag = try readEtag(a, path);
    } else {
        log.dbg("%s: no previous model file found %s\n", .{ func, try zs(a, path) });
    }

    const head = cli.head(a, parts.path);
    if (head.res == null or head.res.?.status < 200 or head.res.?.status >= 300) {
        log.trc("%s: HEAD failed, status: %d\n", .{ func, if (head.res != null) head.res.?.status else @as(c_int, -1) });
        if (file_exists) {
            log.trc("%s: using cached file (HEAD failed): %s\n", .{ func, try zs(a, path) });
            return st.finish(304); // 304 Not Modified - fake cached response
        }
        return st.finish(if (head.res != null) head.res.?.status else -1);
    }
    const hres = head.res.?;

    var etag: []const u8 = "";
    if (hres.hasHeader("ETag")) {
        etag = hres.getHeaderValue("ETag");
    }

    if (hres.hasHeader("Content-Length")) {
        if (stoull(a, hres.getHeaderValue("Content-Length"))) |v| {
            p.total = v;
        } else |_| {
            log.wrn("%s: invalid Content-Length in HEAD response: %s\n", .{ func, @as([*:0]const u8, "stoull: no conversion") });
        }
    }

    var supports_ranges = false;
    if (hres.hasHeader("Accept-Ranges")) {
        supports_ranges = !std.mem.eql(u8, hres.getHeaderValue("Accept-Ranges"), "none");
    }

    if (file_exists) {
        if (etag.len == 0) {
            log.dbg("%s: using cached file (no server etag): %s\n", .{ func, try zs(a, path) });
            return st.finish(304); // 304 Not Modified - fake cached response
        }
        if (last_etag.len != 0 and std.mem.eql(u8, last_etag, etag)) {
            log.dbg("%s: using cached file (same etag): %s\n", .{ func, try zs(a, path) });
            return st.finish(304); // 304 Not Modified - fake cached response
        }
        // pass this point, the file exists but is different from the server version, so we need to redownload it
        if (cRemove(try zs(a, path)) != 0) {
            log.err("%s: unable to delete file: %s\n", .{ func, try zs(a, path) });
            return st.finish(-1);
        }
    }

    { // silent
        _ = hf_cache.createDirectories(a, hf_cache.pathParent(path));
    }

    var success = false;
    var delay: c_int = retry_delay_seconds;

    if (!skip_etag and etag.len != 0) {
        if (hf_cache.exists(a, path_temporary) and !std.mem.eql(u8, try readEtag(a, path_temporary), etag)) {
            log.wrn("%s: partial file does not match the server etag, discarding it: %s\n", .{ func, try zs(a, path_temporary) });
            if (cRemove(try zs(a, path_temporary)) != 0) {
                log.err("%s: unable to delete file: %s\n", .{ func, try zs(a, path_temporary) });
                return st.finish(-1);
            }
        }
        try writeEtag(a, path_temporary, etag);
    }

    if (opts.callback) |cb| {
        // report the resume offset, so a resumed file does not start at 0
        p.downloaded = if (supports_ranges and hf_cache.exists(a, path_temporary)) fileSize(a, path_temporary) else 0;
        cb.on_update(cb.ctx, &p);
    }

    var i: c_int = 0;
    while (i < max_attempts) : (i += 1) {
        if (opts.callback) |cb| if (cb.is_cancelled(cb.ctx)) {
            break;
        };
        if (i != 0) {
            log.wrn("%s: retrying after %d seconds...\n", .{ func, delay });
            sleepSeconds(delay);
            delay *= retry_delay_seconds;
        }

        var existing_size: usize = 0;

        if (hf_cache.exists(a, path_temporary)) {
            if (supports_ranges) {
                existing_size = fileSize(a, path_temporary);
            } else if (cRemove(try zs(a, path_temporary)) != 0) {
                log.err("%s: unable to delete file: %s\n", .{ func, try zs(a, path_temporary) });
                break;
            }
        }

        p.downloaded = existing_size;

        log.dbg("%s: downloading from %s to %s (etag:%s)...\n", .{ func, try zs(a, try commonHttpShowMaskedUrl(a, &parts)), try zs(a, path_temporary), try zs(a, etag) });

        if (try commonPullFile(a, &cli, parts.path, path_temporary, supports_ranges, &p, opts.callback)) {
            if (cRename(try zs(a, path_temporary), try zs(a, path)) != 0) {
                log.err("%s: unable to rename file: %s to %s\n", .{ func, try zs(a, path_temporary), try zs(a, path) });
                break;
            }
            if (etag.len != 0 and !skip_etag) {
                try writeEtag(a, path, etag);
                _ = cRemove(try zs(a, try std.mem.concat(a, u8, &.{ path_temporary, ".etag" })));
            }
            success = true;
            break;
        }
    }

    // a stopped download keeps path_temporary, so the next attempt resumes from it
    if (!success) {
        log.err("%s: download failed after %d attempts\n", .{ func, @as(c_int, max_attempts) });
    }

    return st.finish(if (success) hres.status else -1);
}

fn sleepSeconds(s: c_int) void {
    var ts: std.c.timespec = .{ .sec = s, .nsec = 0 };
    while (std.c.nanosleep(&ts, &ts) != 0) {}
}

fn fileSize(a: Allocator, p: []const u8) usize {
    const z = a.dupeSentinel(u8, p, 0) catch return 0;
    const st = tc.fs_stat.stat(z.ptr, true) orelse return 0;
    return @intCast(st.size);
}

/// The content receiver of `common_remote_get_content`.
const ContentState = struct {
    a: Allocator,
    buf: std.ArrayList(u8) = .empty,
    max_size: c_long,

    fn onData(ctx: *anyopaque, data: []const u8) bool {
        const self: *ContentState = @ptrCast(@alignCast(ctx));
        self.buf.appendSlice(self.a, data) catch return false;
        return self.max_size == 0 or
            self.buf.items.len <= @as(usize, @intCast(self.max_size));
    }

    fn onResponse(_: *anyopaque, _: *const Response) bool {
        return true;
    }
};

/// Ports `common_remote_get_content` (download.cpp:465 @cd959bae9): a GET's
/// status and body, cut off past `max_size`.
pub fn remoteGetContent(a: Allocator, url: []const u8, params: *const RemoteParams) Error!struct { status: c_long, body: []u8 } {
    var cli = try commonHttpClient(a, url);
    defer cli.deinit();
    const parts = cli.parts;

    var headers: Headers = .{};
    for (params.headers) |h| {
        _ = try headers.emplace(a, h[0], h[1]);
    }
    if (headers.find("User-Agent") == null) {
        _ = try headers.emplace(a, "User-Agent", try std.mem.concat(a, u8, &.{ "llama-cpp/", buildInfo() }));
    }

    if (params.timeout > 0) {
        cli.setReadTimeout(params.timeout, 0);
        cli.setWriteTimeout(params.timeout, 0);
    }

    var state: ContentState = .{ .a = a, .max_size = params.max_size };
    const res = cli.getStream(a, parts.path, &headers, &state, null, ContentState.onData);

    if (res.res == null) {
        return throw(error.RuntimeError, "error: cannot make GET request", .{});
    }

    return .{ .status = res.res.?.status, .body = state.buf.items };
}

/// Ports `common_download_file_single` (download.cpp:499 @cd959bae9): one
/// file, or the cached copy when offline.
///
/// Return: an HTTP status, 304 for a cached file, or -1.
pub fn downloadFileSingle(a: Allocator, url: []const u8, path: []const u8, opts: *const Opts, skip_etag: bool, lock_path: []const u8, alt_path: []const u8) Error!c_int {
    const func = "common_download_file_single";
    if (!opts.offline) {
        var tty_cb_state: ProgressBar = .{};
        defer tty_cb_state.deinit();
        const tty_cb = tty_cb_state.callback();
        var online_opts = opts.*;
        if (online_opts.callback == null) {
            online_opts.callback = &tty_cb;
        }
        return downloadFileSingleOnline(a, url, path, &online_opts, skip_etag, lock_path, alt_path);
    }

    if (!hf_cache.exists(a, path)) {
        log.err("%s: required file is not available in cache (offline mode): %s\n", .{ func, try zs(a, path) });
        return -1;
    }

    log.dbg("%s: using cached file (offline mode): %s\n", .{ func, try zs(a, path) });

    // notify the callback that the file was cached
    if (opts.callback) |cb| {
        var p: Progress = .{};
        p.url = url;
        p.cached = true;
        cb.on_start(cb.ctx, &p);
        cb.on_done(cb.ctx, &p, true);
    }

    return 304; // Not Modified - fake cached response
}

/// Ports `gguf_split_info` (download.cpp:533 @cd959bae9).
const SplitInfo = struct {
    prefix: []const u8 = "", // tag included
    tag: []const u8 = "",
    index: c_int = 0,
    count: c_int = 0,
};

/// Ports `get_gguf_split_info` (download.cpp:540 @cd959bae9): a `.gguf`
/// path's stem, split index and count, and its quant tag in upper case.
fn getGgufSplitInfo(a: Allocator, path: []const u8) Error!SplitInfo {
    var re_split = regex.Regex.compileWith("^(.+)-([0-9]{5})-of-([0-9]{5})$", .{ .icase = true }) catch |err| switch (err) {
        error.Range, error.Collate, error.Ctype, error.Escape, error.Backref, error.Brack, error.Paren, error.Brace, error.BadBrace, error.BadRepeat, error.Complexity, error.Parse => unreachable,
    };
    defer re_split.deinit();
    var re_tag = regex.Regex.compileWith("[-.]([A-Z0-9_]+)$", .{ .icase = true }) catch |err| switch (err) {
        error.Range, error.Collate, error.Ctype, error.Escape, error.Backref, error.Brack, error.Paren, error.Brace, error.BadBrace, error.BadRepeat, error.Complexity, error.Parse => unreachable,
    };
    defer re_tag.deinit();
    var m = regex.Match.init();
    defer m.deinit();

    var prefix = tc.strings.removeSuffix(path, ".gguf") orelse return .{};

    var index: c_int = 1;
    var count: c_int = 1;

    if (re_split.match(prefix, &m)) {
        index = try stoiThrow(a, prefix[m.position(2)..][0..m.length(2)]);
        count = try stoiThrow(a, prefix[m.position(3)..][0..m.length(3)]);
        prefix = prefix[m.position(1)..][0..m.length(1)];
    }

    var tag: []const u8 = "";
    if (re_tag.search(prefix, &m)) {
        const t = try a.dupe(u8, prefix[m.position(1)..][0..m.length(1)]);
        for (t) |*ch| ch.* = std.ascii.toUpper(ch.*);
        tag = t;
    }

    return .{ .prefix = prefix, .tag = tag, .index = index, .count = count };
}

/// `std::stoi`, its throw as `std::invalid_argument` or `std::out_of_range`
/// (both caught as `std::exception`).
fn stoiThrow(a: Allocator, s: []const u8) Error!c_int {
    const z = try a.dupeSentinel(u8, s, 0);
    return tc.arg.stoi(z) catch return throw(error.InvalidArgument, "{s}", .{tc.arg.what()});
}

/// Ports `extract_quant_bits` (download.cpp:571 @cd959bae9): Q4_0 -> 4,
/// F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8, etc.
fn extractQuantBits(a: Allocator, filename: []const u8) Error!c_int {
    const split = try getGgufSplitInfo(a, filename);

    const pos = std.mem.indexOfAny(u8, split.tag, "0123456789") orelse return 0;

    return stoiThrow(a, split.tag[pos..]);
}

/// Ports `get_split_files` (download.cpp:582 @cd959bae9): every shard of the
/// split `file` belongs to.
fn getSplitFiles(a: Allocator, files: []const HfFile, file: *const HfFile) Error![]HfFile {
    const split = try getGgufSplitInfo(a, file.path);

    if (split.count <= 1) {
        const out = try a.alloc(HfFile, 1);
        out[0] = file.*;
        return out;
    }
    var result: std.ArrayList(HfFile) = .empty;

    for (files) |f| {
        const split_f = try getGgufSplitInfo(a, f.path);
        if (split_f.count == split.count and std.mem.eql(u8, split_f.prefix, split.prefix)) {
            try result.append(a, f);
        }
    }
    return result.items;
}

/// Ports `find_best_sibling` (download.cpp:603 @cd959bae9): the best sibling
/// GGUF whose filename contains `keyword` (e.g. "mmproj" / "mtp"),
/// preferring deeper shared directory prefix with the model, then exact
/// `tag` match, then closest quantization to the tag when given, or to the
/// model otherwise.
fn findBestSibling(a: Allocator, files: []const HfFile, model: []const u8, keyword: []const u8, tag: []const u8) Error!HfFile {
    var best: HfFile = .{};
    var best_depth: usize = 0;
    var best_diff: c_int = 0;
    var best_exact = false;
    var found = false;

    const tag_upper = try a.dupe(u8, tag);
    for (tag_upper) |*ch| ch.* = std.ascii.toUpper(ch.*);

    var model_bits: c_int = 0;
    if (tag_upper.len != 0) {
        const pos = std.mem.indexOfAny(u8, tag_upper, "0123456789");
        model_bits = if (pos) |p| try stoiThrow(a, tag_upper[p..]) else 0;
    } else {
        model_bits = try extractQuantBits(a, model);
    }
    const model_parts = (try tc.strings.split(a, model, '/')).items;
    const model_dir = model_parts.len - 1;

    for (files) |f| {
        if (!std.mem.endsWith(u8, f.path, ".gguf") or
            std.mem.indexOf(u8, f.path, keyword) == null)
        {
            continue;
        }

        const sib_parts = (try tc.strings.split(a, f.path, '/')).items;
        const sib_dir = sib_parts.len - 1;

        // std::mismatch(model_parts.begin(), model_dir, sib_parts.begin(), sib_dir)
        var dir: usize = 0;
        while (dir < model_dir and dir < sib_dir and std.mem.eql(u8, model_parts[dir], sib_parts[dir])) dir += 1;
        if (dir != sib_dir) {
            continue;
        }

        const depth = dir;
        const bits = try extractQuantBits(a, f.path);
        const diff: c_int = @intCast(@abs(bits - model_bits));

        const path_upper = try a.dupe(u8, f.path);
        for (path_upper) |*ch| ch.* = std.ascii.toUpper(ch.*);
        const exact = tag_upper.len != 0 and std.mem.indexOf(u8, path_upper, try std.mem.concat(a, u8, &.{ "-", tag_upper, "." })) != null;

        if (!found or depth > best_depth or
            (depth == best_depth and exact and !best_exact) or
            (depth == best_depth and exact == best_exact and diff < best_diff))
        {
            best = f;
            best_depth = depth;
            best_diff = diff;
            best_exact = exact;
            found = true;
        }
    }
    return best;
}

/// Ports `sidecars` (download.cpp:666 @cd959bae9), with
/// `common_download_sidecars` (download.cpp:674 @cd959bae9): every sidecar
/// kind, the single list the resolution, the listing and the removal follow.
pub const sidecars = [_]Sidecar{
    .{ .name = "mmproj", .keyword = "mmproj", .draft = false, .opt = "download_mmproj", .file = "mmproj" },
    .{ .name = "mtp", .keyword = "mtp-", .draft = true, .opt = "download_mtp", .file = "mtp" },
    .{ .name = "dflash", .keyword = "dflash-", .draft = true, .opt = "download_dflash", .file = "dflash" },
    .{ .name = "eagle3", .keyword = "eagle3-", .draft = true, .opt = "download_eagle3", .file = "eagle3" },
    .{ .name = "dspark", .keyword = "dspark-", .draft = true, .opt = "download_dspark", .file = "dspark" },
};

/// Ports `common_download_opts_add_sidecar` (download.cpp:678 @cd959bae9):
/// selects the sidecar `kind` in `opts`.
///
/// Return: false for an unknown kind.
pub fn optsAddSidecar(opts: *Opts, kind: []const u8) bool {
    inline for (sidecars) |sc| {
        if (std.mem.eql(u8, kind, sc.name)) {
            @field(opts, sc.opt) = true;
            return true;
        }
    }
    return false;
}

/// `opts.*s.opt = true` for every kind: what a removal's cached plan sets.
fn optsAddAllSidecars(opts: *Opts) void {
    inline for (sidecars) |sc| @field(opts, sc.opt) = true;
}

/// Ports `common_download_plan_sidecars` (download.cpp:688 @cd959bae9): the
/// sidecar kinds `plan` resolves a file for.
pub fn planSidecars(a: Allocator, plan: *const HfPlan) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    inline for (sidecars) |sc| {
        if (@field(plan, sc.file).path.len != 0) try out.append(a, sc.name);
    }
    return out.items;
}

/// Ports `common_download_plan_partials` (download.cpp:698 @cd959bae9): the
/// partials the download of `plan` writes, for the files not in the cache
/// yet: a file in the cache resolves to its snapshot entry, a file still to
/// download to its blob.
pub fn planPartials(a: Allocator, plan: *const HfPlan) Allocator.Error![]Partial {
    var out: std.ArrayList(Partial) = .empty;
    const Add = struct {
        fn f(ar: Allocator, list: *std.ArrayList(Partial), kind: []const u8, file: *const HfFile) Allocator.Error!void {
            if (file.local_path.len != 0 and !std.mem.eql(u8, file.local_path, file.final_path)) {
                try list.append(ar, .{ .kind = kind, .path = try std.mem.concat(ar, u8, &.{ file.local_path, partial_suffix }) });
            }
        }
    };
    for (plan.model_files) |*f| try Add.f(a, &out, "", f);
    inline for (sidecars) |sc| try Add.f(a, &out, sc.name, &@field(plan, sc.file));
    return out.items;
}

/// Ports `name_is_sidecar` (download.cpp:718 @cd959bae9): true when `name`
/// carries the keyword of a sidecar kind.
fn nameIsSidecar(name: []const u8) bool {
    inline for (sidecars) |sc| {
        if (std.mem.indexOf(u8, name, sc.keyword) != null) return true;
    }
    return false;
}

/// Ports `gguf_filename_is_model` (download.cpp:727 @cd959bae9): a `.gguf`
/// that is not a projector, imatrix or draft sidecar.
fn ggufFilenameIsModel(filepath: []const u8) bool {
    if (!std.mem.endsWith(u8, filepath, ".gguf")) {
        return false;
    }

    var filename = filepath;
    if (std.mem.lastIndexOfScalar(u8, filename, '/')) |pos| {
        filename = filename[pos + 1 ..];
    }

    return std.mem.indexOf(u8, filename, "imatrix") == null and !nameIsSidecar(filename);
}

/// `std::regex_search(text, std::regex(pattern, std::regex::icase))`; a
/// pattern that does not compile throws the `regex_error`.
fn icaseSearch(pattern: []const u8, text: []const u8) Error!bool {
    var re = regex.Regex.compileWith(pattern, .{ .icase = true }) catch |e| return throw(error.RuntimeError, "{s}", .{regex.errorText(e)});
    defer re.deinit();
    var m = regex.Match.init();
    defer m.deinit();
    return re.search(text, &m);
}

/// Ports `find_best_model` (download.cpp:740 @8642541b2): the first model
/// file whose name carries the tag (by default Q4_K_M, then Q8_0), the
/// first shard of a split; with no tag, any model file.
fn findBestModel(a: Allocator, files: []const HfFile, tag: []const u8) Error!HfFile {
    var tags: []const []const u8 = undefined;

    if (tag.len != 0) {
        tags = &.{tag};
    } else {
        tags = &.{ "Q4_K_M", "Q8_0" };
    }

    for (tags) |t| {
        const pattern = try std.mem.concat(a, u8, &.{ t, "[.-]" });
        for (files) |f| {
            if (ggufFilenameIsModel(f.path) and
                try icaseSearch(pattern, f.path))
            {
                const split = try getGgufSplitInfo(a, f.path);
                if (split.count > 1 and split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    // fallback to first available model only if tag is empty
    if (tag.len == 0) {
        for (files) |f| {
            if (ggufFilenameIsModel(f.path)) {
                const split = try getGgufSplitInfo(a, f.path);
                if (split.count > 1 and split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    return .{};
}

/// Ports `list_available_gguf_files` (download.cpp:781 @cd959bae9).
fn listAvailableGgufFiles(a: Allocator, files: []const HfFile) Allocator.Error!void {
    log.inf("Available GGUF files:\n", .{});
    for (files) |f| {
        if (std.mem.endsWith(u8, f.path, ".gguf")) {
            log.inf(" - %s\n", .{try zs(a, f.path)});
        }
    }
}

/// Ports `common_download_get_hf_plan` (download.cpp:790 @cd959bae9): which
/// files of a repository to fetch -- its `preset.ini` alone, or the model
/// (every shard) and the sidecars asked for.
pub fn getHfPlan(a: Allocator, model: *const ModelParams, opts: *const Opts) Error!HfPlan {
    const func = "common_download_get_hf_plan";
    var plan: HfPlan = .{};
    var all: []HfFile = &.{};

    const repo, const tag = try splitRepoTag(a, model.hf_repo);

    if (!opts.offline) {
        all = try hf_cache.getRepoFiles(a, repo, opts.bearer_token);
    }
    if (all.len == 0) {
        all = try hf_cache.getCachedFiles(a, repo);
    }
    if (all.len == 0) {
        return plan;
    }

    // if preset.ini exists in the repo root, download only that file
    for (all) |f| {
        if (std.mem.eql(u8, f.path, "preset.ini")) {
            plan.preset = f;
            return plan;
        }
    }

    var primary: HfFile = .{};

    if (model.hf_file.len != 0) {
        for (all) |f| {
            if (std.mem.eql(u8, f.path, model.hf_file)) {
                primary = f;
                break;
            }
        }
        if (primary.path.len == 0) {
            log.err("%s: file '%s' not found in repository\n", .{ func, try zs(a, model.hf_file) });
            try listAvailableGgufFiles(a, all);
            return plan;
        }
    } else {
        primary = try findBestModel(a, all, tag);
        // a requested draft can resolve on its own, without a full model of the same tag
        var want_draft = false;
        inline for (sidecars) |sc| {
            if (sc.draft and @field(opts, sc.opt)) want_draft = true;
        }
        if (primary.path.len == 0 and !want_draft) {
            log.err("%s: no GGUF files found in repository %s\n", .{ func, try zs(a, repo) });
            try listAvailableGgufFiles(a, all);
            return plan;
        }
    }

    if (primary.path.len != 0) {
        plan.primary = primary;
        plan.model_files = try getSplitFiles(a, all, &primary);
    }

    // a draft resolves at the tag, the mmproj pairs with the model
    inline for (sidecars) |sc| {
        if (@field(opts, sc.opt)) {
            if (sc.draft) {
                @field(plan, sc.file) = try findBestSibling(a, all, primary.path, sc.keyword, tag);
            } else if (primary.path.len != 0) {
                @field(plan, sc.file) = try findBestSibling(a, all, primary.path, sc.keyword, "");
            }
        }
    }

    var has_draft = false;
    inline for (sidecars) |sc| {
        if (sc.draft and @field(plan, sc.file).local_path.len != 0) has_draft = true;
    }
    if (primary.path.len == 0 and !has_draft) {
        log.err("%s: no GGUF files found in repository %s\n", .{ func, try zs(a, repo) });
        try listAvailableGgufFiles(a, all);
    }

    return plan;
}

/// One `std::async` download of `common_download_run_tasks`.
const TaskThread = struct {
    task: *const Task,
    status: c_int = -1,
    err: ?anyerror = null,
    what: []const u8 = "",

    fn run(self: *TaskThread) void {
        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena.deinit();
        const t = self.task;
        self.status = downloadFileSingle(arena.allocator(), t.url, t.local_path, &t.opts, t.is_hf, t.lock_path, t.alt_path) catch |e| {
            self.err = e;
            self.what = std.heap.c_allocator.dupe(u8, common.what(e)) catch "";
            return;
        };
    }
};

/// Ports `common_download_run_tasks` (download.cpp:869 @cd959bae9): every
/// task at once, then each result in order; the first failure throws.
pub fn runTasks(a: Allocator, tasks: []const Task) Error!void {
    const jobs = try a.alloc(TaskThread, tasks.len);
    const threads = try a.alloc(?std.Thread, tasks.len);
    for (tasks, 0..) |*task, i| {
        jobs[i] = .{ .task = task };
        threads[i] = std.Thread.spawn(.{}, TaskThread.run, .{&jobs[i]}) catch null;
        if (threads[i] == null) jobs[i].run();
    }

    var failed: ?Error = null;
    for (jobs, 0..) |*job, i| {
        if (threads[i]) |t| t.join();
        if (failed != null) continue;
        const url = tasks[i].url;
        // `futures[i].get()` rethrows what the download threw
        if (job.err) |e| {
            failed = switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                error.JsonError => error.JsonError,
                error.InvalidArgument => throw(error.InvalidArgument, "{s}", .{job.what}),
                else => throw(error.RuntimeError, "{s}", .{job.what}),
            };
            continue;
        }
        const status = job.status;
        const is_ok = isHttpStatusOk(status);
        if (!is_ok) {
            failed = throw(error.RuntimeError, "Download '{s}' failed with status code: {d}", .{ url, status });
        }
    }
    if (failed) |e| return e;
}

/// Ports `common_download_get_all_parts` (download.cpp:889 @cd959bae9):
/// every shard's URL of a split GGUF, or the URL alone.
pub fn getAllParts(a: Allocator, url: []const u8) Error![]const []const u8 {
    const split = try getGgufSplitInfo(a, url);

    if (split.count <= 1) {
        const out = try a.alloc([]const u8, 1);
        out[0] = url;
        return out;
    }

    var parts: std.ArrayList([]const u8) = .empty;
    var i: c_int = 1;
    while (i <= split.count) : (i += 1) {
        const suffix = try std.fmt.allocPrint(a, "-{d:0>5}-of-{d:0>5}.gguf", .{ @as(u32, @intCast(i)), @as(u32, @intCast(split.count)) });
        try parts.append(a, try std.mem.concat(a, u8, &.{ split.prefix, suffix }));
    }
    return parts.items;
}

//
// Docker registry functions
//

/// Ports `common_docker_get_token` (download.cpp:908 @cd959bae9).
fn dockerGetToken(a: Allocator, repo: []const u8) Error![]const u8 {
    const url = try std.mem.concat(a, u8, &.{ "https://auth.docker.io/token?service=registry.docker.io&scope=repository:", repo, ":pull" });

    const params: RemoteParams = .{};
    const res = try remoteGetContent(a, url, &params);

    if (res.status != 200) {
        return throw(error.RuntimeError, "Failed to get Docker registry token, HTTP code: {d}", .{res.status});
    }

    var response = try json.parse(a, res.body);

    if (!response.contains("token")) {
        return throw(error.RuntimeError, "Docker registry token response missing 'token' field", .{});
    }

    return (try response.index(a, "token")).get([]const u8);
}

/// Ports `common_docker_resolve_model` (download.cpp:927 @8642541b2): a
/// Docker model's GGUF layer downloaded to the cache.
///
/// Return: the local path.
pub fn dockerResolveModel(a: Allocator, docker: []const u8) Error![]const u8 {
    const func = "common_docker_resolve_model";
    // Parse ai/smollm2:135M-Q4_0
    var repo: []const u8 = undefined;
    var tag: []const u8 = undefined;
    if (std.mem.indexOfScalar(u8, docker, ':')) |colon_pos| {
        repo = docker[0..colon_pos];
        tag = docker[colon_pos + 1 ..];
    } else {
        repo = docker;
        tag = "latest";
    }

    // ai/ is the default
    if (std.mem.indexOfScalar(u8, docker, '/') == null) {
        repo = try std.mem.concat(a, u8, &.{ "ai/", repo });
    }

    log.inf("%s: Downloading Docker Model: %s:%s\n", .{ func, try zs(a, repo), try zs(a, tag) });
    return dockerResolveImpl(a, repo, tag) catch |e| {
        if (e != error.OutOfMemory) {
            log.err("%s: Docker Model download failed: %s\n", .{ func, try zs(a, common.what(e)) });
        }
        return e;
    };
}

fn dockerResolveImpl(a: Allocator, repo: []const u8, tag: []const u8) Error![]const u8 {
    const func = "common_docker_resolve_model";
    const token = try dockerGetToken(a, repo); // Get authentication token

    // Get manifest
    // TODO: cache the manifest response so that it appears in the model list
    const url_prefix = try std.mem.concat(a, u8, &.{ "https://registry-1.docker.io/v2/", repo });
    const manifest_url = try std.mem.concat(a, u8, &.{ url_prefix, "/manifests/", tag });
    const manifest_headers = [_]struct { []const u8, []const u8 }{
        .{ "Authorization", try std.mem.concat(a, u8, &.{ "Bearer ", token }) },
        .{ "Accept", "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json" },
    };
    const manifest_params: RemoteParams = .{ .headers = &manifest_headers };
    const manifest_res = try remoteGetContent(a, manifest_url, &manifest_params);
    if (manifest_res.status != 200) {
        return throw(error.RuntimeError, "Failed to get Docker manifest, HTTP code: {d}", .{manifest_res.status});
    }

    var manifest = try json.parse(a, manifest_res.body);
    var gguf_digest: []const u8 = ""; // Find the GGUF layer
    if (manifest.contains("layers")) {
        for ((try manifest.index(a, "layers")).array.items) |*layer| {
            if (layer.contains("mediaType")) {
                const media_type = try (try layer.at("mediaType")).get([]const u8);
                if (std.mem.eql(u8, media_type, "application/vnd.docker.ai.gguf.v3") or
                    std.mem.indexOf(u8, media_type, "gguf") != null)
                {
                    gguf_digest = try (try layer.at("digest")).get([]const u8);
                    break;
                }
            }
        }
    }

    if (gguf_digest.len == 0) {
        return throw(error.RuntimeError, "No GGUF layer found in Docker manifest", .{});
    }

    // Validate & normalize digest
    gguf_digest = try validateOciDigest(a, gguf_digest);
    log.dbg("%s: Using validated digest: %s\n", .{ func, try zs(a, gguf_digest) });

    // Prepare local filename
    var model_filename = try a.dupe(u8, repo);
    std.mem.replaceScalar(u8, model_filename, '/', '_');
    model_filename = try std.mem.concat(a, u8, &.{ model_filename, "_", tag, ".gguf" });
    const local_path = try fsGetCacheFile(a, model_filename);

    const blob_url = try std.mem.concat(a, u8, &.{ url_prefix, "/blobs/", gguf_digest });
    const opts: Opts = .{ .bearer_token = token };
    const http_status = try downloadFileSingle(a, blob_url, local_path, &opts, false, "", "");
    if (!isHttpStatusOk(http_status)) {
        return throw(error.RuntimeError, "Failed to download Docker Model", .{});
    }

    log.inf("%s: Downloaded Docker Model to: %s\n", .{ func, try zs(a, local_path) });
    return local_path;
}

/// The `validate_oci_digest` (download.cpp:949 @cd959bae9) lambda of
/// `common_docker_resolve_model`: `sha256:` and 64 hex digits, lower-cased.
fn validateOciDigest(a: Allocator, digest: []const u8) Error![]const u8 {
    var re = regex.Regex.compile("^sha256:([a-fA-F0-9]{64})$") catch |err| switch (err) {
        error.Range, error.Collate, error.Ctype, error.Escape, error.Backref, error.Brack, error.Paren, error.Brace, error.BadBrace, error.BadRepeat, error.Complexity, error.Parse => unreachable,
    };
    defer re.deinit();
    var m = regex.Match.init();
    defer m.deinit();
    if (!re.match(digest, &m)) {
        return throw(error.RuntimeError, "Invalid OCI digest format received in manifest: {s}", .{digest});
    }
    // normalize hex to lowercase
    const normalized = try a.dupe(u8, digest);
    for (normalized[7..]) |*ch| ch.* = std.ascii.toLower(ch.*);
    return normalized;
}

/// Ports `is_listed_model` (download.cpp:1028 @cd959bae9): a cached file that
/// lists a model under its tag: the first split of a model, not a sidecar.
fn isListedModel(split: *const SplitInfo) bool {
    return split.index == 1 and split.tag.len != 0 and !nameIsSidecar(split.prefix);
}

/// Ports `common_list_cached_models` (download.cpp:1032 @cd959bae9): each
/// cached repository and tag with a first shard that is a model.
pub fn listCachedModels(a: Allocator) Error![]CachedModelInfo {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var result: std.ArrayList(CachedModelInfo) = .empty;

    const files = try hf_cache.getCachedFiles(a, "");

    for (files) |f| {
        const split = try getGgufSplitInfo(a, f.path);
        if (!isListedModel(&split)) {
            continue;
        }
        const key = try std.mem.concat(a, u8, &.{ f.repo_id, ":", split.tag });
        const gop = try seen.getOrPut(a, key);
        if (!gop.found_existing) {
            try result.append(a, .{ .repo = f.repo_id, .tag = split.tag });
        }
    }

    return result.items;
}

/// Ports `common_download_resolve_path` (download.cpp:1051 @cd959bae9): the
/// cached file a repository resolves to, without the network.
pub fn resolvePath(a: Allocator, hf_repo_with_tag: []const u8, hf_file: []const u8) Error![]const u8 {
    const repo, const tag = try splitRepoTag(a, hf_repo_with_tag);

    const files = try hf_cache.getCachedFiles(a, repo);
    if (files.len == 0) {
        return "";
    }

    if (hf_file.len != 0) {
        for (files) |f| {
            if (std.mem.eql(u8, f.path, hf_file)) {
                return f.local_path;
            }
        }
        return "";
    }

    return (try findBestModel(a, files, tag)).local_path;
}

/// Ports `files_in_use` (download.cpp:1073 @cd959bae9): the files `-hf`
/// loads from the cache, the model with every sidecar, for each cached tag of
/// `repo_id` other than `tag`.
fn filesInUse(a: Allocator, repo_id: []const u8, tag: []const u8) Error!std.StringHashMapUnmanaged(void) {
    var opts: Opts = .{};
    opts.offline = true;
    optsAddAllSidecars(&opts);

    var tags: std.StringHashMapUnmanaged(void) = .empty;
    var used: std.StringHashMapUnmanaged(void) = .empty;

    for (try hf_cache.getCachedFiles(a, repo_id)) |f| {
        const split = try getGgufSplitInfo(a, f.path);
        if (!isListedModel(&split) or std.mem.eql(u8, split.tag, tag) or (try tags.getOrPut(a, split.tag)).found_existing) {
            continue;
        }

        var model: ModelParams = .{};
        model.hf_repo = try std.mem.concat(a, u8, &.{ repo_id, ":", split.tag });

        const plan = try getHfPlan(a, &model, &opts);

        for (plan.model_files) |m| {
            try used.put(a, m.local_path, {});
        }
        inline for (sidecars) |sc| {
            const sf = &@field(plan, sc.file);
            if (sf.local_path.len != 0) {
                try used.put(a, sf.local_path, {});
            }
        }
    }
    return used;
}

/// Ports `common_download_remove` (download.cpp:1109 @cd959bae9): a cached
/// repository, or only its files of one tag and the blobs nothing else
/// links to, except the files another cached tag of the repository still
/// loads.
///
/// Return: whether anything was removed.
pub fn remove(a: Allocator, hf_repo_with_tag: []const u8) Error!bool {
    const repo_id, const tag = try splitRepoTag(a, hf_repo_with_tag);

    if (tag.len == 0) {
        return hf_cache.removeCachedRepo(a, repo_id);
    }

    const tag_upper = try a.dupe(u8, tag);
    for (tag_upper) |*ch| ch.* = std.ascii.toUpper(ch.*);

    const files = try hf_cache.getCachedFiles(a, repo_id);
    if (files.len == 0) {
        return false;
    }

    // collect snapshot entries whose tag matches, but a file another tag loads stays
    const used = try filesInUse(a, repo_id, tag_upper);

    var to_remove: std.ArrayList([]const u8) = .empty;
    for (files) |f| {
        const split = try getGgufSplitInfo(a, f.path);
        if (std.mem.eql(u8, split.tag, tag_upper) and !used.contains(f.local_path)) {
            try to_remove.append(a, f.local_path);
        }
    }

    if (to_remove.items.len == 0) {
        return false;
    }

    return removeSnapshotEntries(a, repo_id, to_remove.items);
}

/// Ports `remove_snapshot_entries` (download.cpp:1147 @cd959bae9): snapshot
/// entries of `repo_id`, then the blobs no remaining entry references.
fn removeSnapshotEntries(a: Allocator, repo_id: []const u8, to_remove: []const []const u8) Error!bool {
    const func = "remove_snapshot_entries";

    // resolve blob paths from symlinks before deleting snapshot entries
    var blobs_to_check: std.ArrayList([]const u8) = .empty;
    for (to_remove) |p| {
        if (hf_cache.isSymlink(a, p)) {
            if (hf_cache.readSymlink(a, p)) |target| {
                try blobs_to_check.append(a, try hf_cache.lexicallyNormal(a, try hf_cache.pathJoin(a, hf_cache.pathParent(p), target)));
            }
        }
    }

    // remove snapshot entries
    for (to_remove) |p| {
        const ec = hf_cache.removeFile(a, p);
        if (ec != 0) {
            log.wrn("%s: failed to remove %s: %s\n", .{ func, try zs(a, p), hf_cache.errMessage(ec) });
        }
    }

    if (blobs_to_check.items.len == 0) {
        return true;
    }

    // collect blobs still referenced by remaining snapshot entries
    var still_referenced: std.StringHashMapUnmanaged(void) = .empty;
    for (try hf_cache.getCachedFiles(a, repo_id)) |f| {
        const p = f.local_path;
        if (hf_cache.isSymlink(a, p)) {
            if (hf_cache.readSymlink(a, p)) |target| {
                try still_referenced.put(a, try hf_cache.lexicallyNormal(a, try hf_cache.pathJoin(a, hf_cache.pathParent(p), target)), {});
            }
        }
    }

    // remove orphaned blobs
    for (blobs_to_check.items) |blob| {
        if (!still_referenced.contains(blob)) {
            const ec = hf_cache.removeFile(a, blob);
            if (ec != 0) {
                log.wrn("%s: failed to remove blob %s: %s\n", .{ func, try zs(a, blob), hf_cache.errMessage(ec) });
            }
        }
    }

    return true;
}

/// Ports `common_download_remove_sidecars` (download.cpp:1202 @cd959bae9):
/// the cached sidecar files `opts` selects for `user/model:tag`, as the
/// cached plan resolves them, leaving the model files in place; a sidecar
/// file another tag shares goes for every tag that resolves it.
///
/// Return: true if anything was removed.
pub fn removeSidecars(a: Allocator, hf_repo_with_tag: []const u8, opts: *const Opts) Error!bool {
    var model: ModelParams = .{};
    model.hf_repo = hf_repo_with_tag;

    var cached = opts.*;
    cached.offline = true;

    const plan = try getHfPlan(a, &model, &cached);

    var to_remove: std.ArrayList([]const u8) = .empty;
    inline for (sidecars) |sc| {
        const f = &@field(plan, sc.file);
        if (f.local_path.len != 0) {
            try to_remove.append(a, f.local_path);
        }
    }
    if (to_remove.items.len == 0) {
        return false;
    }

    const repo_id, _ = try splitRepoTag(a, hf_repo_with_tag);
    return removeSnapshotEntries(a, repo_id, to_remove.items);
}

// -----------------------------------------------------------------------------
// common_models_handler (arg.cpp)

/// Ports `mmproj_examples` (arg.cpp:36 @cd959bae9): the programs that
/// download a model's projector.
const mmproj_examples = [_]tc.params.Example{ .mtmd, .server, .cli, .tts };

/// Ports `common_models_handler` (arg.h:140 @cd959bae9).
pub const ModelsHandler = struct {
    plan: HfPlan = .{},
    plan_spec: HfPlan = .{},
    opts: Opts = .{},
};

/// Ports `get_default_local_path` (arg.cpp:351 @8642541b2): a URL's last
/// path element, in the cache.
fn getDefaultLocalPath(a: Allocator, url: []const u8) Error![]const u8 {
    var f = (try tc.strings.split(a, url, '#')).items[0];
    f = (try tc.strings.split(a, f, '?')).items[0];
    const sp = (try tc.strings.split(a, f, '/')).items;
    return fsGetCacheFile(a, sp[sp.len - 1]);
}

/// Ports `spec_types_is_default` (arg.cpp:333 @cd959bae9).
fn specTypesIsDefault(params: *const Params) bool {
    const t = params.speculative.types.items;
    return t.len == 1 and t[0] == .none;
}

fn hasSpecType(params: *const Params, t: SpeculativeType) bool {
    for (params.speculative.types.items) |x| if (x == t) return true;
    return false;
}

/// Ports `common_models_handler_init` (arg.cpp:337 @cd959bae9): the
/// download options and, for a repository, its plan -- which reaches the
/// hub, but downloads nothing.
pub fn modelsHandlerInit(a: Allocator, params: *const Params, curr_ex: tc.params.Example) Error!ModelsHandler {
    var plan: HfPlan = .{};
    var plan_spec: HfPlan = .{};
    var opts: Opts = .{};

    const spec_type_draft_mtp = hasSpecType(params, .draft_mtp);
    const spec_type_draft_dflash = hasSpecType(params, .draft_dflash);
    const spec_type_draft_eagle3 = hasSpecType(params, .draft_eagle3);
    const spec_type_draft_dspark = hasSpecType(params, .draft_dspark);

    // only download mmproj if the current example is using it
    var use_mmproj = false;
    for (mmproj_examples) |ex| {
        if (curr_ex == ex) {
            use_mmproj = true;
            break;
        }
    }
    if (curr_ex == .download) {
        use_mmproj = true;
    }

    opts.bearer_token = params.hf_token;
    opts.offline = params.offline;
    opts.download_mtp = spec_type_draft_mtp;
    opts.download_eagle3 = spec_type_draft_eagle3;
    opts.download_dflash = spec_type_draft_dflash;
    opts.download_dspark = spec_type_draft_dspark;
    opts.download_mmproj = use_mmproj and !params.no_mmproj and
        params.mmproj.path.len == 0 and params.mmproj.url.len == 0;

    if (params.model.hf_repo.len != 0) {
        plan = try getHfPlan(a, &params.model, &opts);
    }

    if (params.speculative.draft.mparams.hf_repo.len != 0) {
        // without a requested type, discover every sidecar the draft repo ships to infer the type later
        var opts_spec = opts;
        if (specTypesIsDefault(params)) {
            opts_spec.download_mtp = true;
            opts_spec.download_dflash = true;
            opts_spec.download_eagle3 = true;
            opts_spec.download_dspark = true;
        }
        plan_spec = try getHfPlan(a, &params.speculative.draft.mparams, &opts_spec);
    }

    return .{ .plan = plan, .plan_spec = plan_spec, .opts = opts };
}

/// Ports `common_models_handler_is_preset_repo` (arg.cpp:398 @cd959bae9).
pub fn modelsHandlerIsPresetRepo(handler: *const ModelsHandler) bool {
    return handler.plan.preset.url.len != 0;
}

/// Ports `build_url_tasks` (arg.cpp:402 @cd959bae9): one task per shard of
/// a URL, under `-m`'s directory when given, else the cache.
fn buildUrlTasks(a: Allocator, model: *const ModelParams, opts: Opts) Error![]Task {
    const parts = try getAllParts(a, model.url);
    var tasks: std.ArrayList(Task) = .empty;

    // single-part: download straight to model.path if the user gave one (-m), else the cache default
    if (parts.len == 1) {
        try tasks.append(a, .{
            .url = parts[0],
            .local_path = if (model.path.len == 0) try getDefaultLocalPath(a, parts[0]) else model.path,
            .opts = opts,
        });
        return tasks.items;
    }

    // multi-part: place each part under the user's -m directory (if given), else the cache default
    var base_dir: []const u8 = "";
    if (model.path.len != 0) {
        const pos = std.mem.lastIndexOfScalar(u8, model.path, '/');
        base_dir = if (pos) |p| model.path[0..p] else ".";
    }

    for (parts) |part| {
        var local = try getDefaultLocalPath(a, part);
        if (base_dir.len != 0) {
            const pos = std.mem.lastIndexOfScalar(u8, local, '/');
            const name = if (pos) |p| local[p + 1 ..] else local;
            local = try std.mem.concat(a, u8, &.{ base_dir, "/", name });
        }
        try tasks.append(a, .{ .url = part, .opts = opts, .local_path = local });
    }
    return tasks.items;
}

/// What a task's `on_done` lambda in `common_models_handler_apply` captured.
const Done = struct {
    a: Allocator,
    params: *Params,
    file: ?*const HfFile = null,
    kind: enum { model_path, finalize_only, model_primary, spec_sidecar, plan_sidecar, mmproj, preset } = .finalize_only,
    /// for `model_path`: the first part's path
    first_path: []const u8 = "",
    /// for `model_primary`: the model whose path it sets
    target: ?*ModelParams = null,
    /// `callback`, which the `finalize` lambda captures.
    callback: ?*const Callback = null,

    /// The lambda `finalize` passes `finalize_file`: another process may
    /// hold the cache lock during finalization, so the download callback
    /// cancels that wait too.
    fn keepWaiting(ctx: ?*anyopaque) bool {
        const self: *Done = @ptrCast(@alignCast(ctx.?));
        if (self.callback) |cb| return !cb.is_cancelled(cb.ctx);
        return true;
    }

    /// The `finalize` lambda (arg.cpp:451 @cd959bae9).
    fn finalize(self: *Done) Error![]const u8 {
        return hf_cache.finalizeFile(self.a, self.file.?, .{ .ctx = self, .func = keepWaiting });
    }

    fn call(ctx: ?*anyopaque) anyerror!void {
        const self: *Done = @ptrCast(@alignCast(ctx.?));
        const params = self.params;
        switch (self.kind) {
            .model_path => params.model.path = self.first_path,
            .finalize_only => _ = try self.finalize(),
            .model_primary => self.target.?.path = try self.finalize(),
            .spec_sidecar => {
                // only use the discovered sidecar when no draft path is set yet
                if (params.speculative.draft.mparams.path.len == 0) {
                    params.speculative.draft.mparams.path = try self.finalize();
                } else {
                    _ = try self.finalize();
                }
            },
            .plan_sidecar => {
                // only fall back to the discovered sidecar when no draft was explicitly provided
                if (params.speculative.draft.mparams.empty()) {
                    params.speculative.draft.mparams.path = try self.finalize();
                } else {
                    _ = try self.finalize();
                }
            },
            .mmproj => params.mmproj.path = try self.finalize(),
            .preset => {
                // if HF repo is a preset repo, we simply run server in router mode with the preset.ini file
                params.models_preset_hf = params.model.hf_repo; // only for showing a warning
                params.models_preset = try self.finalize();
                // clear the model so the server starts in router mode
                params.model.path = "";
                params.model.hf_repo = "";
                params.model.docker_repo = "";
            },
        }
    }
};

fn makeOnDone(a: Allocator, d: Done) Allocator.Error!OnDone {
    const p = try a.create(Done);
    p.* = d;
    return .{ .ctx = p, .func = Done.call };
}

/// Ports `common_models_handler_apply` (arg.cpp:440 @cd959bae9): every file
/// the handler planned and the URLs `params` names, downloaded at once
/// unless offline, then the paths set in `params`.
///
/// Parameters:
/// - `a`: owns the paths `params` is given; must outlive `params`' use.
pub fn modelsHandlerApply(a: Allocator, handler: *ModelsHandler, params: *Params, callback: ?*const Callback) Error!void {
    var tasks: std.ArrayList(Task) = .empty;

    const plan = &handler.plan;
    const plan_spec = &handler.plan_spec;

    var opts = handler.opts; // copy
    opts.callback = callback;

    // handle plain "url" if needed
    for ([_]*ModelParams{ &params.model, &params.mmproj, &params.speculative.draft.mparams }) |model| {
        if (model.url.len != 0) {
            if (model.path.len == 0) {
                model.path = try getDefaultLocalPath(a, model.url);
            }
        }
    }

    // optionally, if docker repo is set, resolve it
    if (params.model.docker_repo.len != 0) {
        params.model.url = try dockerResolveModel(a, params.model.docker_repo);
        params.model.path = try getDefaultLocalPath(a, params.model.url);
    }

    // handle plain "url" tasks (non-hf)
    if (params.model.url.len != 0) {
        const url_tasks = try buildUrlTasks(a, &params.model, opts);
        // the first part is what gets loaded, so point params.model.path at it
        if (url_tasks.len != 0) {
            const first_path = url_tasks[0].local_path;
            url_tasks[0].on_done = try makeOnDone(a, .{ .a = a, .params = params, .kind = .model_path, .first_path = first_path });
        }
        try tasks.appendSlice(a, url_tasks);
    }
    if (params.mmproj.url.len != 0) {
        try tasks.append(a, .{ .url = params.mmproj.url, .local_path = params.mmproj.path, .opts = opts });
    }
    var had_spec_url = false;
    if (params.speculative.draft.mparams.url.len != 0) {
        try tasks.append(a, .{ .url = params.speculative.draft.mparams.url, .local_path = params.speculative.draft.mparams.path, .opts = opts });
        had_spec_url = true;
    }

    // an explicit draft file selection (e.g. -md with -hfd) disables the sidecar resolution of the draft repo
    if (params.speculative.draft.mparams.hf_file.len != 0) {
        plan_spec.mtp = .{};
        plan_spec.dflash = .{};
        plan_spec.eagle3 = .{};
        plan_spec.dspark = .{};
    }

    // infer the speculative type from the sidecar shipped by the draft repo when none is requested
    if (specTypesIsDefault(params)) {
        if (plan_spec.mtp.local_path.len != 0) {
            try setSpecTypes(params, .draft_mtp);
            plan_spec.dspark = .{};
            plan_spec.dflash = .{};
            plan_spec.eagle3 = .{};
        } else if (plan_spec.dspark.local_path.len != 0) {
            // dspark outranks dflash, its sidecar carries the extra Markov head
            try setSpecTypes(params, .draft_dspark);
            plan_spec.dflash = .{};
            plan_spec.eagle3 = .{};
        } else if (plan_spec.dflash.local_path.len != 0) {
            try setSpecTypes(params, .draft_dflash);
            plan_spec.eagle3 = .{};
        } else if (plan_spec.eagle3.local_path.len != 0) {
            try setSpecTypes(params, .draft_eagle3);
        }
    }

    // infer the speculative type from the draft GGUF metadata when none is requested
    // note: reads only the first split - sharded drafts need an explicit --spec-type
    if (specTypesIsDefault(params) and params.speculative.draft.mparams.path.len != 0) {
        const z = try a.dupeSentinel(u8, params.speculative.draft.mparams.path, 0);
        const types_gguf = tc.speculative.typesFromGguf(a, z) catch |e| return mapAny(e);
        if (types_gguf.len != 0) {
            params.speculative.types.clearRetainingCapacity();
            try params.speculative.types.appendSlice(params.arena, types_gguf);
        }
    }

    // when a sidecar type is requested, the draft repo resolves to its sidecar instead of a full model
    const spec_sidecar_found = plan_spec.mtp.local_path.len != 0 or
        plan_spec.dflash.local_path.len != 0 or
        plan_spec.eagle3.local_path.len != 0 or
        plan_spec.dspark.local_path.len != 0;
    for ([_]*HfFile{ &plan_spec.mtp, &plan_spec.dflash, &plan_spec.eagle3, &plan_spec.dspark }) |f| {
        if (f.local_path.len != 0 and !had_spec_url) {
            try tasks.append(a, try Task.fromHf(a, f, opts, try makeOnDone(a, .{ .a = a, .params = params, .file = f, .kind = .spec_sidecar, .callback = callback })));
        }
    }

    // a wired draft sidecar counts as an explicit draft for the main plan fallback below
    if (spec_sidecar_found) {
        had_spec_url = true;
    }

    // handle plan_spec (e.g. --spec-draft-hf)
    if (plan_spec.model_files.len != 0 and !had_spec_url and !spec_sidecar_found) {
        try addTasks(a, &tasks, opts, params, plan_spec.model_files, &plan_spec.primary, &params.speculative.draft.mparams, callback);
        had_spec_url = true;
    }

    if (plan.model_files.len != 0) {
        try addTasks(a, &tasks, opts, params, plan.model_files, &plan.primary, &params.model, callback);
    }
    if (plan.mmproj.local_path.len != 0) {
        try tasks.append(a, try Task.fromHf(a, &plan.mmproj, opts, try makeOnDone(a, .{ .a = a, .params = params, .file = &plan.mmproj, .kind = .mmproj, .callback = callback })));
    }
    for ([_]*HfFile{ &plan.mtp, &plan.dflash, &plan.eagle3, &plan.dspark }) |f| {
        if (f.local_path.len != 0 and !had_spec_url) {
            try tasks.append(a, try Task.fromHf(a, f, opts, try makeOnDone(a, .{ .a = a, .params = params, .file = f, .kind = .plan_sidecar, .callback = callback })));
        }
    }
    if (plan.preset.local_path.len != 0) {
        try tasks.append(a, try Task.fromHf(a, &plan.preset, opts, try makeOnDone(a, .{ .a = a, .params = params, .file = &plan.preset, .kind = .preset, .callback = callback })));
    }

    // run all tasks in parallel
    if (!params.offline) {
        // if duplicated files are found, only download once (but still call on_done for each task)
        var unique_tasks = try libcxx.StringMap(*Task).init(a);
        for (tasks.items) |*task| {
            // `if (it == end) unique_tasks[task.local_path] = &task;`
            _ = try unique_tasks.getOrPut(task.local_path, task);
        }
        var unique_tasks_vec: std.ArrayList(Task) = .empty;
        var it = unique_tasks.iterator();
        while (it.next()) |e| {
            log.dbg("download task: %s -> %s\n", .{ try zs(a, e.value.*.url), try zs(a, e.value.*.local_path) });
            try unique_tasks_vec.append(a, e.value.*.*);
        }
        try runTasks(a, unique_tasks_vec.items);
    }

    // download successful, update params with the downloaded paths
    for (tasks.items) |*task| {
        task.on_done.call() catch |e| return mapAny(e);
    }
}

fn setSpecTypes(params: *Params, t: SpeculativeType) Allocator.Error!void {
    params.speculative.types.clearRetainingCapacity();
    try params.speculative.types.append(params.arena, t);
}

fn mapAny(e: anyerror) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
        error.InvalidArgument => error.InvalidArgument,
        error.RuntimeError => error.RuntimeError,
        error.Throw => throw(error.RuntimeError, "{s}", .{tc.arg.what()}),
        else => throw(error.RuntimeError, "{s}", .{@errorName(e)}),
    };
}

/// The `add_tasks` (arg.cpp:505 @cd959bae9) lambda of
/// `common_models_handler_apply`: every shard, the primary one setting
/// `model`'s path.
fn addTasks(a: Allocator, tasks: *std.ArrayList(Task), opts: Opts, params: *Params, model_files: []HfFile, primary: *const HfFile, model: *ModelParams, callback: ?*const Callback) Error!void {
    for (model_files) |*model_file| {
        const is_primary = std.mem.eql(u8, model_file.path, primary.path);
        try tasks.append(a, try Task.fromHf(a, model_file, opts, try makeOnDone(a, .{
            .a = a,
            .params = params,
            .file = model_file,
            .callback = callback,
            // the primary file is the first split (00001-of), use it as model path
            .kind = if (is_primary) .model_primary else .finalize_only,
            .target = model,
        })));
    }
}

// -----------------------------------------------------------------------------
// The HTTP client the downloader drives: cpp-httplib's `Client`, as
// `common_http_client` configures it.

pub const Headers = client.HeadersMap;
pub const Response = client.Response;
pub const ErrorCode = client.ErrorCode;
pub const errorToString = client.errorToString;

/// The pair `common_http_client` returns: the client and the parsed URL.
pub const HttpClient = struct {
    parts: HttpUrl,
    cli: client.Client,

    fn init(a: Allocator, parts: HttpUrl) Error!HttpClient {
        // `httplib::Client cli(scheme + "://" + host + ":" + port)`
        return .{ .parts = parts, .cli = try client.Client.init(a, parts.host, parts.port) };
    }

    pub fn deinit(self: *HttpClient) void {
        self.cli.deinit();
    }

    fn setBasicAuth(self: *HttpClient, user: []const u8, password: []const u8) void {
        self.cli.setBasicAuth(user, password);
    }

    fn setProxy(self: *HttpClient, host: []const u8, port: c_int) void {
        self.cli.setProxy(host, port);
    }

    fn setProxyBasicAuth(self: *HttpClient, user: []const u8, password: []const u8) void {
        self.cli.setProxyBasicAuth(user, password);
    }

    fn setFollowLocation(self: *HttpClient, on: bool) void {
        self.cli.setFollowLocation(on);
    }

    fn setDefaultHeaders(self: *HttpClient, headers: Headers) void {
        self.cli.setDefaultHeaders(headers);
    }

    fn setReadTimeout(self: *HttpClient, sec: c_long, usec: c_long) void {
        self.cli.setReadTimeout(sec, usec);
    }

    fn setWriteTimeout(self: *HttpClient, sec: c_long, usec: c_long) void {
        self.cli.setWriteTimeout(sec, usec);
    }

    /// `cli.Get(path, headers)`: the whole body.
    pub fn get(self: *HttpClient, a: Allocator, path: []const u8, headers: *const Headers) client.Result {
        return self.cli.get(a, path, headers);
    }

    /// `cli.Head(path)`.
    pub fn head(self: *HttpClient, a: Allocator, path: []const u8) client.Result {
        return self.cli.head(a, path);
    }

    /// `cli.Get(path, headers, response_handler, content_receiver)`.
    pub fn getStream(
        self: *HttpClient,
        a: Allocator,
        path: []const u8,
        headers: *const Headers,
        ctx: *anyopaque,
        on_response: ?*const fn (ctx: *anyopaque, r: *const Response) bool,
        on_data: *const fn (ctx: *anyopaque, data: []const u8) bool,
    ) client.Result {
        return self.cli.getStream(a, path, headers, ctx, on_response, on_data);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the proxy's exceptions: loopback and NO_PROXY" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "localhost", "LOCALHOST", "::1", "0.0.0.0", "a.localhost", "127.0.0.1", "127.255.1.2" }) |h| {
        try std.testing.expect(try commonHttpIsLoopback(a, h));
    }
    for ([_][]const u8{ "128.0.0.1", "127.0.0", "127.0.0.1.2", "127.0.0.256", "1270.0.0.1", "example.com", "" }) |h| {
        try std.testing.expect(!try commonHttpIsLoopback(a, h));
    }
    try std.testing.expect(try commonHttpProxyBypass(a, "*", "x"));
    try std.testing.expect(try commonHttpProxyBypass(a, " example.com ,other", "EXAMPLE.com"));
    try std.testing.expect(try commonHttpProxyBypass(a, ".example.com", "a.example.com"));
    try std.testing.expect(try commonHttpProxyBypass(a, "example.com", "a.example.com"));
    try std.testing.expect(!try commonHttpProxyBypass(a, "example.com", "aexample.com"));
    try std.testing.expect(!try commonHttpProxyBypass(a, ",.,", "x"));
}

test "split info" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = try getGgufSplitInfo(a, "dir/model-Q4_K_M-00002-of-00003.gguf");
    try std.testing.expectEqualStrings("dir/model-Q4_K_M", s.prefix);
    try std.testing.expectEqualStrings("Q4_K_M", s.tag);
    try std.testing.expectEqual(@as(c_int, 2), s.index);
    try std.testing.expectEqual(@as(c_int, 3), s.count);
    const p = try getAllParts(a, "http://h/x-00001-of-00002.gguf");
    try std.testing.expectEqualStrings("http://h/x-00002-of-00002.gguf", p[1]);
}
