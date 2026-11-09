//! The option table, part three: from `--embd-normalize` to
//! `--log-timestamps`, arg.cpp's lines 3289 to 3983 -- the server's block, the
//! chat and reasoning options, and logging.
//!
//! # Provenance
//!
//! Ports that stretch of `common_params_parser_init` (arg.cpp:1344 @cd959bae9)
//! from `llama.cpp/common/arg.cpp` (v0.8.0, `8642541b2`), as
//! `arg_options.zig` describes, with the helpers its handlers call:
//! `common_reasoning_format_from_name` from `llama.cpp/common/chat.cpp`,
//! `fs_is_directory` from `llama.cpp/common/common.cpp`, and libc++'s
//! `std::filesystem::create_directories` as `--log-prompts-dir` reaches it.
//! Each names what it ports and the line it began at.
//!
//! The JSON these options read and write -- `--chat-template-kwargs`,
//! `--reasoning-effort` -- goes through `cli/json/`, the port of
//! `common_json`, reached as `cli_fit`'s `json`.

const std = @import("std");
const fs_stat = @import("fs_stat.zig");
const c = @import("tools_h");
const arg = @import("arg.zig");
const params_mod = @import("params.zig");
const log = @import("log.zig");
const io = @import("io.zig");
const options1 = @import("arg_options.zig");
const options2 = @import("arg_options2.zig");
const json = @import("cli_fit").json;

const Params = params_mod.Params;
const Arg = arg.Arg;
const Context = arg.Context;
const throw = arg.throw;
const throwStr = arg.throwStr;
const S = [:0]const u8;

/// `"enabled"` or `"disabled"`.
fn ed(b: bool) [*:0]const u8 {
    return if (b) "enabled" else "disabled";
}

/// `s` with its NUL, for a `%s`.
fn z(p: *Params, s: []const u8) ![*:0]const u8 {
    return (try p.arena.dupeSentinel(u8, s, 0)).ptr;
}

/// A `common_json_error` as the handler's exception: `e.what()`.
fn jsonThrow(e: json.Error) anyerror {
    return switch (e) {
        error.JsonError => throwStr(json.what()),
        error.OutOfMemory => error.OutOfMemory,
    };
}

/// Ports `common_reasoning_format_from_name` (chat.cpp:1026 @cd959bae9).
fn reasoningFormatFromName(format: []const u8) !params_mod.ReasoningFormat {
    if (std.mem.eql(u8, format, "none")) return .none;
    if (std.mem.eql(u8, format, "auto")) return .auto;
    if (std.mem.eql(u8, format, "deepseek")) return .deepseek;
    if (std.mem.eql(u8, format, "deepseek-legacy")) return .deepseek_legacy;
    var buf: [512]u8 = undefined;
    return throwStr(std.fmt.bufPrint(&buf, "Unknown reasoning format: {s}", .{format}) catch "Unknown reasoning format");
}

/// `stat`'s answer, as libc++'s `posix_stat` classifies it.
const Status = enum { unknown, not_found, directory, other };

fn statOf(path: [:0]const u8, err: *c_int) Status {
    var st: fs_stat.Info = .{};
    const e = fs_stat.statErrno(path.ptr, true, &st);
    if (e == 0) return if (st.mode & fs_stat.IFMT == fs_stat.IFDIR) .directory else .other;
    if (e == @backingInt(std.c.E.NOENT) or e == @backingInt(std.c.E.NOTDIR)) return .not_found;
    err.* = e;
    return .unknown;
}

/// Ports `fs_is_directory` (common/common.cpp:964 @cd959bae9).
fn fsIsDirectory(p: *Params, path: []const u8) !bool {
    var e: c_int = 0;
    return statOf(try p.arena.dupeSentinel(u8, path, 0), &e) == .directory;
}

/// `path::parent_path` for a POSIX path: everything before the last
/// element, its trailing separators dropped, the root kept.
fn parentPath(path: []const u8) []const u8 {
    // no relative path: the root, or nothing
    if (std.mem.indexOfNone(u8, path, "/") == null) return path;
    if (path[path.len - 1] == '/') {
        // the last element is the empty filename after the separator
        return std.mem.trimEnd(u8, path, "/");
    }
    const cut = std.mem.lastIndexOfScalar(u8, path, '/') orelse return "";
    const parent = std.mem.trimEnd(u8, path[0 .. cut + 1], "/");
    return if (parent.len == 0) "/" else parent;
}

/// Reproduces libc++'s `__create_directories`, line 471 of
/// `src/filesystem/operations.cpp` in Zig's bundled libc++ 22: `mkdir -p`,
/// the error an `errno`.
fn createDirectories(a: std.mem.Allocator, path: []const u8) !c_int {
    const pz = try a.dupeSentinel(u8, path, 0);
    var e: c_int = 0;
    switch (statOf(pz, &e)) {
        .unknown => return e,
        .directory => return 0,
        .other => return @backingInt(std.c.E.EXIST),
        .not_found => {},
    }
    const parent = parentPath(path);
    if (parent.len != 0) {
        const parz = try a.dupeSentinel(u8, parent, 0);
        switch (statOf(parz, &e)) {
            .unknown => return e,
            .not_found => {
                if (std.mem.eql(u8, parent, path)) return @backingInt(std.c.E.INVAL);
                const r = try createDirectories(a, parent);
                if (r != 0) return r;
            },
            .other => return @backingInt(std.c.E.NOTDIR),
            .directory => {},
        }
    }
    // `__create_directory`
    if (std.c.mkdir(pz.ptr, 0o777) == 0) return 0;
    const mec = std.c._errno().*;
    if (mec != @backingInt(std.c.E.EXIST)) return mec;
    var ignored: c_int = 0;
    if (statOf(pz, &ignored) != .directory) return mec;
    return 0;
}

extern fn strerror(errnum: c_int) [*:0]const u8;

/// The handlers of `--slot-save-path` and `--media-path`: a directory, with
/// a separator at its end.
fn directoryWithSeparator(q: *Params, value: S) ![]const u8 {
    if (!try fsIsDirectory(q, value)) {
        var buf: [4096]u8 = undefined;
        return throwStr(std.fmt.bufPrint(&buf, "not a directory: {s}", .{value}) catch "not a directory");
    }
    // if doesn't end with DIRECTORY_SEPARATOR, add it
    if (value.len != 0 and value[value.len - 1] != '/') return std.mem.concat(q.arena, u8, &.{ value, "/" });
    return arg.dupe(q, value);
}

/// The table, part three.
pub fn add(ctx: *Context) !void {
    const p = ctx.params;

    // arg.cpp:3314
    try ctx.add((Arg{ .args = &.{"--embd-normalize"}, .value_hint = "N", .help = try ctx.fmt("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", .{p.embd_normalize}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.embd_normalize = value;
        }
    }.f } }).setExamples(&.{ .embedding, .server, .debug }));
    // arg.cpp:3342
    try ctx.add((Arg{ .args = &.{"--host"}, .value_hint = "HOST", .help = try ctx.fmt("IP addresses to listen on, comma-separated, or UNIX socket paths ending in .sock; with multiple TCP addresses, :: binds IPv6 only; overlapping addresses result in undefined behavior (default: %s)", .{try z(p, p.hostnames.items[0])}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.hostnames.clearRetainingCapacity();
            const hosts = try options1.parseCsvRow(q.arena, value);
            for (hosts.items) |h| {
                const host = @import("strings.zig").strip(h);
                if (host.len != 0) try q.hostnames.append(q.arena, host);
            }
            if (q.hostnames.items.len == 0) return throwStr("--host requires at least one address");
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_HOST"));
    // arg.cpp:3358
    try ctx.add((Arg{ .args = &.{"--port"}, .value_hint = "PORT", .help = try ctx.fmt("port to listen (default: %d)", .{p.port}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.port = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_PORT"));
    // arg.cpp:3365
    try ctx.add((Arg{ .args = &.{"--reuse-port"}, .help = try ctx.fmt("allow multiple sockets to bind to the same port (default: %s)", .{ed(p.reuse_port)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.reuse_port = true;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_REUSE_PORT"));
    // arg.cpp:3372
    try ctx.add((Arg{ .args = &.{"--path"}, .value_hint = "PATH", .help = try ctx.fmt("path to serve static files from (default: %s)", .{try z(p, p.public_path)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.public_path = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_STATIC_PATH"));
    // arg.cpp:3379
    try ctx.add((Arg{ .args = &.{"--cors-origins"}, .value_hint = "ORIGINS", .help = try ctx.fmt("comma-separated list of allowed origins for CORS (default: %s)\nif set to special value 'localhost', reflect the Origin header only if it is localhost", .{try z(p, p.cors_origins)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cors_origins = try arg.dupe(q, value);
            q.cors_origins_explicit = true;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_CORS_ORIGINS"));
    // arg.cpp:3390
    try ctx.add((Arg{ .args = &.{"--cors-methods"}, .value_hint = "METHODS", .help = try ctx.fmt("comma-separated list of allowed methods for CORS (default: %s)", .{try z(p, p.cors_methods)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cors_methods = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_CORS_METHODS"));
    // arg.cpp:3397
    try ctx.add((Arg{ .args = &.{"--cors-headers"}, .value_hint = "HEADERS", .help = try ctx.fmt("comma-separated list of allowed headers for CORS (default: %s)", .{try z(p, p.cors_headers)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cors_headers = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_CORS_HEADERS"));
    // arg.cpp:3404
    try ctx.add((Arg{ .args = &.{"--cors-credentials"}, .args_neg = &.{"--no-cors-credentials"}, .help = try ctx.fmt("whether to allow credentials for CORS (default: %s)\nnote: if this is enabled and --cors-origins is set to * (default), the Origin header will be echoed back, and credentials will always be allowed", .{ed(p.cors_credentials)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.cors_credentials = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_CORS_CREDENTIALS"));
    // arg.cpp:3415
    try ctx.add((Arg{ .args = &.{"--api-prefix"}, .value_hint = "PREFIX", .help = try ctx.fmt("prefix path the server serves from, without the trailing slash (default: %s)", .{try z(p, p.api_prefix)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.api_prefix = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_API_PREFIX"));
    // arg.cpp:3422
    try ctx.add((Arg{ .args = &.{ "--ui-config", "--webui-config" }, .value_hint = "JSON", .help = "JSON that provides default UI settings (overrides UI defaults)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.ui_config_json = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_UI_CONFIG"));
    // arg.cpp:3429
    try ctx.add((Arg{ .args = &.{ "--ui-config-file", "--webui-config-file" }, .value_hint = "PATH", .help = "JSON file that provides default UI settings (overrides UI defaults)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.ui_config_json = try options1.readFile(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_UI_CONFIG_FILE"));
    // arg.cpp:3436
    try ctx.add((Arg{ .args = &.{ "--ui-mcp-proxy", "--webui-mcp-proxy" }, .args_neg = &.{ "--no-ui-mcp-proxy", "--no-webui-mcp-proxy" }, .help = "experimental: whether to enable MCP CORS proxy - do not enable in untrusted environments (default: disabled)", .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.ui_mcp_proxy = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_UI_MCP_PROXY"));
    // arg.cpp:3444
    try ctx.add((Arg{ .args = &.{"--tools"}, .value_hint = "TOOL1,TOOL2,...", .help = "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted environments (default: no tools)\nspecify \"all\" to enable all tools\navailable tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_info\nnote: for security reasons, this will limit --cors-origins to localhost by default", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const items = try options1.parseCsvRow(q.arena, value);
            q.server_tools = .empty;
            for (items.items) |item| try q.server_tools.append(q.arena, item);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_TOOLS"));
    // arg.cpp:3454
    try ctx.add((Arg{ .args = &.{"--tools-runtime"}, .value_hint = "OPTION", .help = "experimental: run tools in a separate runtime environment (default: none, use host environment)\navailable options:\n  'docker:<image>', 'podman:<image>': spin up a new container and reuse it for all invocations, clean up on server exit\n  'docker-container:<id>', 'podman-container:<id>': use an existing container by ID, won't stop on server exit\n  'ssh:<target>': run tools on a remote POSIX host over SSH, key-based auth and a trusted host key are required\n", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.server_tools_runtime = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_TOOLS_RUNTIME"));
    // arg.cpp:3465
    try ctx.add((Arg{ .args = &.{"--mcp-servers-config"}, .value_hint = "PATH", .help = "experimental: path to JSON file with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\nnote: for security reasons, this will limit --cors-origins to localhost by default", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.mcp_servers_config = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_MCP_SERVERS_CONFIG"));
    // arg.cpp:3473
    try ctx.add((Arg{ .args = &.{"--mcp-servers-json"}, .value_hint = "JSON", .help = "experimental: inline JSON with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\nnote: for security reasons, this will limit --cors-origins to localhost by default", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.mcp_servers_json = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_MCP_SERVERS_JSON"));
    // arg.cpp:3481
    try ctx.add((Arg{
        .args = &.{ "-ag", "--agent" },
        .args_neg = &.{ "-no-ag", "--no-agent" },
        .help = "whether to enable CORS proxy and all built-in tools - do not enable in untrusted environments (default: disabled)\nnote: for security reasons, this will limit --cors-origins to localhost by default",
        .handler = .{
            .bool_ = struct {
                fn f(q: *Params, value: bool) !void {
                    q.server_tools = .empty;
                    if (value) {
                        try q.server_tools.append(q.arena, "all");
                        q.ui_mcp_proxy = true;
                    } else {
                        q.ui_mcp_proxy = false;
                    }
                    // note: do not modify cors_origins here, as the options are not evaluated in order (user may explicitly set --cors-origins before --agent)
                }
            }.f,
        },
    }).setExamples(&.{.server}).setEnv("LLAMA_ARG_AGENT"));
    // arg.cpp:3497
    try ctx.add((Arg{ .args = &.{ "--ui", "--webui" }, .args_neg = &.{ "--no-ui", "--no-webui" }, .help = try ctx.fmt("whether to enable the Web UI (default: %s)", .{ed(p.ui)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.ui = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_UI"));
    // arg.cpp:3505
    try ctx.add((Arg{ .args = &.{"--connect"}, .help = try ctx.fmt("(experimental) open a peer-to-peer tunnel for server using llama-connect (default: %s)", .{ed(p.server_connect)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.server_connect = true;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_CONNECT"));
    // arg.cpp:3512
    try ctx.add((Arg{ .args = &.{"--connect-code"}, .value_hint = "CODE", .help = "manually specify a 40-character code for --connect (default: generate a new one for each run)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            var buf: std.ArrayList(u8) = .empty;
            const a = q.arena;
            for (value) |ch| {
                if (ch != '-' and ch != ' ') try buf.append(a, ch);
            }
            if (buf.items.len != 40) {
                arg.setWhat("error: invalid connect code '%s', must be 40 characters\n", .{try z(q, value)});
                return error.Throw;
            }
            q.server_connect_code = buf.items;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_CONNECT_CODE"));
    // arg.cpp:3525
    try ctx.add((Arg{ .args = &.{ "--embedding", "--embeddings" }, .help = try ctx.fmt("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", .{ed(p.embedding)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.embedding = true;
        }
    }.f } }).setExamples(&.{ .server, .debug }).setEnv("LLAMA_ARG_EMBEDDINGS"));
    // arg.cpp:3532
    try ctx.add((Arg{ .args = &.{ "--rerank", "--reranking" }, .help = try ctx.fmt("enable reranking endpoint on server (default: %s)", .{@as([*:0]const u8, "disabled")}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.embedding = true;
            q.pooling_type = c.LLAMA_POOLING_TYPE_RANK;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_RERANKING"));
    // arg.cpp:3540
    try ctx.add((Arg{ .args = &.{"--api-key"}, .value_hint = "KEY", .help = "API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const items = try options1.parseCsvRow(q.arena, value);
            for (items.items) |key| {
                if (key.len != 0) try q.api_keys.append(q.arena, key);
            }
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_API_KEY"));
    // arg.cpp:3551
    try ctx.add((Arg{
        .args = &.{"--api-key-file"},
        .value_hint = "FNAME",
        .help = "path to file containing API keys, one per line; lines starting with a hash are treated as comments (default: none)",
        .handler = .{
            .string = struct {
                fn f(q: *Params, value: S) !void {
                    const content = try options1.readFile(q, value);
                    // `std::getline`: no empty line after a final '\n'
                    var it = std.mem.splitScalar(u8, content, '\n');
                    while (it.next()) |key| {
                        if (key.len != 0 and key[0] != '#') try q.api_keys.append(q.arena, key);
                    }
                }
            }.f,
        },
    }).setExamples(&.{.server}).setEnv("LLAMA_ARG_API_KEY_FILE"));
    // arg.cpp:3568
    try ctx.add((Arg{ .args = &.{"--ssl-key-file"}, .value_hint = "FNAME", .help = "path to file a PEM-encoded SSL private key", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.ssl_file_key = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_SSL_KEY_FILE"));
    // arg.cpp:3575
    try ctx.add((Arg{ .args = &.{"--ssl-cert-file"}, .value_hint = "FNAME", .help = "path to file a PEM-encoded SSL certificate", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.ssl_file_cert = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_SSL_CERT_FILE"));
    // arg.cpp:3582
    try ctx.add((Arg{ .args = &.{"--chat-template-kwargs"}, .value_hint = "STRING", .help = "sets additional params for the json template parser, must be a valid json object string, e.g. '{\"key1\":\"value1\",\"key2\":\"value2\"}'", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            var parsed = json.parse(q.arena, value) catch |e| return jsonThrow(e);
            var it = parsed.items();
            while (it.next()) |item| {
                if (std.mem.eql(u8, item.key, "enable_thinking")) {
                    log.wrn("Setting 'enable_thinking' via --chat-template-kwargs is deprecated. Use --reasoning on / --reasoning off instead.\n", .{});
                }
                if (std.mem.eql(u8, item.key, "preserve_reasoning")) {
                    log.wrn("Setting 'preserve_reasoning' via --chat-template-kwargs is deprecated. Use --reasoning-preserve / --no-reasoning-preserve instead.\n", .{});
                }
                const dumped = json.dump(q.arena, item.value, -1) catch |e| return jsonThrow(e);
                try q.default_template_kwargs.put(q.arena, try arg.dupe(q, item.key), dumped);
            }
        }
    }.f } }).setExamples(&.{ .server, .cli }).setEnv("LLAMA_ARG_CHAT_TEMPLATE_KWARGS"));
    // arg.cpp:3600
    try ctx.add((Arg{ .args = &.{ "-to", "--timeout" }, .value_hint = "N", .help = try ctx.fmt("server read/write timeout in seconds (default: %d)", .{p.timeout_read}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.timeout_read = value;
            q.timeout_write = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_TIMEOUT"));
    // arg.cpp:3608
    try ctx.add((Arg{ .args = &.{"--sse-ping-interval"}, .value_hint = "N", .help = try ctx.fmt("server SSE ping interval in seconds (-1 = disabled, default: %d)", .{p.sse_ping_interval}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.sse_ping_interval = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_SSE_PING_INTERVAL"));
    // arg.cpp:3615
    try ctx.add((Arg{ .args = &.{"--threads-http"}, .value_hint = "N", .help = try ctx.fmt("number of threads used to process HTTP requests (default: %d)", .{p.n_threads_http}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_threads_http = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_THREADS_HTTP"));
    // arg.cpp:3622
    try ctx.add((Arg{ .args = &.{"--cache-prompt"}, .args_neg = &.{"--no-cache-prompt"}, .help = try ctx.fmt("whether to enable prompt caching (default: %s)", .{ed(p.cache_prompt)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.cache_prompt = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_CACHE_PROMPT"));
    // arg.cpp:3630
    try ctx.add((Arg{ .args = &.{"--cache-reuse"}, .value_hint = "N", .help = try ctx.fmt("min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled (default: %d)\n[(card)](https://ggml.ai/f0.png)", .{p.n_cache_reuse}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_cache_reuse = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_CACHE_REUSE"));
    // arg.cpp:3640
    try ctx.add((Arg{ .args = &.{"--metrics"}, .help = try ctx.fmt("enable prometheus compatible metrics endpoint (default: %s)", .{ed(p.endpoint_metrics)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.endpoint_metrics = true;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_ENDPOINT_METRICS"));
    // arg.cpp:3647
    try ctx.add((Arg{ .args = &.{"--props"}, .help = try ctx.fmt("enable changing global properties via POST /props (default: %s)", .{ed(p.endpoint_props)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.endpoint_props = true;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_ENDPOINT_PROPS"));
    // arg.cpp:3654
    try ctx.add((Arg{ .args = &.{"--slots"}, .args_neg = &.{"--no-slots"}, .help = try ctx.fmt("expose slots monitoring endpoint (default: %s)", .{ed(p.endpoint_slots)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.endpoint_slots = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_ENDPOINT_SLOTS"));
    // arg.cpp:3662
    try ctx.add((Arg{ .args = &.{"--slot-save-path"}, .value_hint = "PATH", .help = "path to save slot kv cache (default: disabled)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.slot_save_path = try directoryWithSeparator(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_SLOT_SAVE_PATH"));
    // arg.cpp:3676
    try ctx.add((Arg{ .args = &.{"--media-path"}, .value_hint = "PATH", .help = "directory for loading local media files; files can be accessed via file:// URLs using relative paths (default: disabled)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.media_path = try directoryWithSeparator(q, value);
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:3690
    try ctx.add((Arg{ .args = &.{"--models-dir"}, .value_hint = "PATH", .help = "directory containing models for the router server (default: disabled)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.models_dir = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_MODELS_DIR"));
    // arg.cpp:3697
    try ctx.add((Arg{ .args = &.{"--models-preset"}, .value_hint = "PATH", .help = "path to INI file containing model presets for the router server (default: disabled)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.models_preset = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_MODELS_PRESET"));
    // arg.cpp:3704
    try ctx.add((Arg{ .args = &.{"--models-max"}, .value_hint = "N", .help = try ctx.fmt("for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)", .{p.models_max}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.models_max = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_MODELS_MAX"));
    // arg.cpp:3711
    try ctx.add((Arg{ .args = &.{"--models-memory-margin"}, .value_hint = "N", .help = try ctx.fmt("for router server, MiB of device memory to keep free per device, unload LRU models until a new one fits (default: %d, 0 = disabled)", .{p.models_memory_margin}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.models_memory_margin = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_MODELS_MEMORY_MARGIN"));
    // arg.cpp:3718
    try ctx.add((Arg{ .args = &.{"--models-autoload"}, .args_neg = &.{"--no-models-autoload"}, .help = try ctx.fmt("for router server, whether to automatically load models (default: %s)", .{ed(p.models_autoload)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.models_autoload = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_MODELS_AUTOLOAD"));
    // arg.cpp:3726
    try ctx.add((Arg{ .args = &.{"--jinja"}, .args_neg = &.{"--no-jinja"}, .help = try ctx.fmt("whether to use jinja template engine for chat (default: %s)", .{@as([*:0]const u8, if (p.use_jinja) "enabled" else "disabled")}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.use_jinja = value;
        }
    }.f } }).setExamples(&.{ .server, .completion, .cli, .mtmd }).setEnv("LLAMA_ARG_JINJA"));
    // arg.cpp:3734
    try ctx.add((Arg{ .args = &.{"--reasoning-format"}, .value_hint = "FORMAT", .help = "controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:\n- none: leaves thoughts unparsed in `message.content`\n- deepseek: puts thoughts in `message.reasoning_content`\n- deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`\n(default: auto)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.reasoning_format = try reasoningFormatFromName(value);
        }
    }.f } }).setExamples(&.{ .server, .completion, .cli }).setEnv("LLAMA_ARG_THINK"));
    // arg.cpp:3745
    try ctx.add((Arg{ .args = &.{ "-rea", "--reasoning" }, .value_hint = "[on|off|auto]", .help = "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (arg.isTruthy(value)) {
                q.enable_reasoning = 1;
                try q.default_template_kwargs.put(q.arena, "enable_thinking", "true");
            } else if (arg.isFalsey(value)) {
                q.enable_reasoning = 0;
                try q.default_template_kwargs.put(q.arena, "enable_thinking", "false");
            } else if (arg.isAutoy(value)) {
                q.enable_reasoning = -1;
            } else {
                return throw("error: unknown value for --reasoning: '%s'\n", .{value.ptr});
            }
        }
    }.f } }).setExamples(&.{ .server, .completion, .cli }).setEnv("LLAMA_ARG_REASONING"));
    // arg.cpp:3763
    try ctx.add((Arg{ .args = &.{"--reasoning-effort"}, .value_hint = "LEVEL", .help = "reasoning effort level given to the chat template: 'default' to keep the template default,\nor a level such as 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max' (default: default)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "default")) {
                q.default_template_kwargs.erase("reasoning_effort");
            } else {
                const v = json.Value.initString(q.arena, value) catch |e| return jsonThrow(e);
                const dumped = json.dump(q.arena, &v, -1) catch |e| return jsonThrow(e);
                try q.default_template_kwargs.put(q.arena, "reasoning_effort", dumped);
            }
        }
    }.f } }).setExamples(&.{ .server, .completion, .cli }).setEnv("LLAMA_ARG_REASONING_EFFORT"));
    // arg.cpp:3775
    try ctx.add((Arg{ .args = &.{"--reasoning-budget"}, .value_hint = "N", .help = "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < -1) return throwStr("invalid value");
            q.sampling.reasoning_budget_tokens = value;
        }
    }.f } }).setExamples(&.{ .server, .completion, .cli }).setEnv("LLAMA_ARG_THINK_BUDGET"));
    // arg.cpp:3783
    try ctx.add((Arg{ .args = &.{"--reasoning-budget-message"}, .value_hint = "MESSAGE", .help = "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.reasoning_budget_message = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .server, .completion, .cli }).setEnv("LLAMA_ARG_THINK_BUDGET_MESSAGE"));
    // arg.cpp:3790
    try ctx.add((Arg{ .args = &.{"--reasoning-preserve"}, .args_neg = &.{"--no-reasoning-preserve"}, .help = "preserve reasoning trace in the full history, not just the last assistant message (default: enabled)\ncompatible with certain templates having 'supports_preserve_reasoning' capability\nexample: https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking", .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            try q.default_template_kwargs.put(q.arena, "preserve_reasoning", if (value) "true" else "false");
            q.preserve_reasoning_specified = true;
        }
    }.f } }).setExamples(&.{ .server, .completion, .cli }).setEnv("LLAMA_ARG_REASONING_PRESERVE"));
    // arg.cpp:3805
    try ctx.add((Arg{ .args = &.{"--chat-template"}, .value_hint = "JINJA_TEMPLATE", .help = try ctx.fmt("set custom jinja chat template (default: template taken from model's metadata)\nif suffix/prefix are specified, template will be disabled\nonly commonly used templates are accepted (unless --jinja is set before this flag):\nlist of built-in templates:\n%s", .{(try options2.listBuiltinChatTemplates(p.arena)).ptr}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.chat_template = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .completion, .cli, .server, .mtmd }).setEnv("LLAMA_ARG_CHAT_TEMPLATE"));
    // arg.cpp:3817
    try ctx.add((Arg{ .args = &.{"--chat-template-file"}, .value_hint = "JINJA_TEMPLATE_FILE", .help = try ctx.fmt("set custom jinja chat template file (default: template taken from model's metadata)\nif suffix/prefix are specified, template will be disabled\nonly commonly used templates are accepted (unless --jinja is set before this flag):\nlist of built-in templates:\n%s", .{(try options2.listBuiltinChatTemplates(p.arena)).ptr}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.chat_template = try options1.readFile(q, value);
        }
    }.f } }).setExamples(&.{ .completion, .cli, .server }).setEnv("LLAMA_ARG_CHAT_TEMPLATE_FILE"));
    // arg.cpp:3829
    try ctx.add((Arg{ .args = &.{"--skip-chat-parsing"}, .args_neg = &.{"--no-skip-chat-parsing"}, .help = "force a pure content parser, even if a Jinja template is specified; model will output everything in the content section, including any reasoning and/or tool calls (default: disabled)", .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.force_pure_content_parser = value;
        }
    }.f } }).setExamples(&.{ .completion, .cli, .server }).setEnv("LLAMA_ARG_SKIP_CHAT_PARSING"));
    // arg.cpp:3840
    try ctx.add((Arg{ .args = &.{"--prefill-assistant"}, .args_neg = &.{"--no-prefill-assistant"}, .help = "whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\nwhen this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\n", .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.prefill_assistant = value;
        }
    }.f } }).setExamples(&.{.server}).setEnv("LLAMA_ARG_PREFILL_ASSISTANT"));
    // arg.cpp:3851
    try ctx.add((Arg{ .args = &.{ "-sps", "--slot-prompt-similarity" }, .value_hint = "SIMILARITY", .help = try ctx.fmt("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", .{@as(f64, p.slot_prompt_similarity)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.slot_prompt_similarity = try arg.stof(value);
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:3858
    try ctx.add((Arg{ .args = &.{"--lora-init-without-apply"}, .help = try ctx.fmt("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", .{ed(p.lora_init_without_apply)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.lora_init_without_apply = true;
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:3865
    try ctx.add((Arg{ .args = &.{"--sleep-idle-seconds"}, .value_hint = "SECONDS", .help = try ctx.fmt("number of seconds of idleness after which the server will sleep (default: %d; -1 = disabled)", .{p.sleep_idle_seconds}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value == 0 or value < -1) return throwStr("invalid value: cannot be 0 or less than -1");
            q.sleep_idle_seconds = value;
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:3882
    const pos_z = try p.arena.dupeSentinel(u8, p.cvector_positive_file, 0);
    try ctx.add((Arg{ .args = &.{"--positive-file"}, .value_hint = "FNAME", .help = try ctx.fmt("positive prompts file, one prompt per line (default: '%s')", .{pos_z.ptr}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cvector_positive_file = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3889
    const neg_z = try p.arena.dupeSentinel(u8, p.cvector_negative_file, 0);
    try ctx.add((Arg{ .args = &.{"--negative-file"}, .value_hint = "FNAME", .help = try ctx.fmt("negative prompts file, one prompt per line (default: '%s')", .{neg_z.ptr}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cvector_negative_file = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3896
    try ctx.add((Arg{ .args = &.{"--pca-batch"}, .value_hint = "N", .help = try ctx.fmt("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", .{p.n_pca_batch}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_pca_batch = value;
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3903
    try ctx.add((Arg{ .args = &.{"--pca-iter"}, .value_hint = "N", .help = try ctx.fmt("number of iterations used for PCA (default: %d)", .{p.n_pca_iterations}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_pca_iterations = value;
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3910
    try ctx.add((Arg{ .args = &.{"--method"}, .value_hint = "{pca, mean}", .help = "dimensionality reduction method to be used (default: pca)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "pca")) {
                q.cvector_dimre_method = .pca;
            } else if (std.mem.eql(u8, value, "mean")) {
                q.cvector_dimre_method = .mean;
            } else return throwStr("invalid value");
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3919
    try ctx.add((Arg{ .args = &.{"--output-format"}, .value_hint = "{md,jsonl}", .help = "output format for batched-bench results (default: md)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "jsonl")) {
                q.batched_bench_output_jsonl = true;
            } else if (std.mem.eql(u8, value, "md")) {
                q.batched_bench_output_jsonl = false;
            } else return throwStr("invalid value");
        }
    }.f } }).setExamples(&.{.bench}));
    // arg.cpp:3928
    try ctx.add(.{ .args = &.{"--log-disable"}, .help = "Log disable", .handler = .{ .void_ = struct {
        fn f(_: *Params) !void {
            log.pause();
        }
    }.f } });
    // arg.cpp:3935
    try ctx.add((Arg{ .args = &.{"--log-file"}, .value_hint = "FNAME", .help = "Log to file", .handler = .{ .string = struct {
        fn f(_: *Params, value: S) !void {
            log.setFile(value);
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_FILE"));
    // arg.cpp:3942
    try ctx.add((Arg{ .args = &.{"--log-jsonl"}, .args_neg = &.{"--no-log-jsonl"}, .help = "Log as JSONL (one JSON object per line) to stdout, this also disables colored logging (default: disabled)", .handler = .{ .bool_ = struct {
        fn f(_: *Params, value: bool) !void {
            log.setJsonl(value);
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_JSONL"));
    // arg.cpp:3950
    try ctx.add((Arg{ .args = &.{"--log-prompts-dir"}, .value_hint = "PATH", .help = "Log prompts to directory (auto-created if not present; only used for debugging, default: disabled)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.path_prompts_log_dir = try arg.dupe(q, value);
            const e = try createDirectories(q.arena, value);
            if (e != 0) {
                io.eprintf("warning: failed to create prompts-log-dir '%s': %s\n", .{ value.ptr, strerror(e) });
            }
        }
    }.f } }).setExamples(&.{ .server, .cli }));
    // arg.cpp:3962
    try ctx.add((Arg{ .args = &.{"--log-colors"}, .value_hint = "[on|off|auto]", .help = "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n'auto' enables colors when output is to a terminal", .handler = .{ .string = struct {
        fn f(_: *Params, value: S) !void {
            if (arg.isTruthy(value)) {
                log.setColors(true);
            } else if (arg.isFalsey(value)) {
                log.setColors(false);
            } else if (arg.isAutoy(value)) {
                log.setColors(null);
            } else return throw("error: unknown value for --log-colors: '%s'\n", .{value.ptr});
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_COLORS"));
    // arg.cpp:3979
    try ctx.add(.{ .args = &.{ "-v", "--verbose", "--log-verbose" }, .help = "Set verbosity level to infinity (i.e. log all messages, useful for debugging)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.verbosity = std.math.maxInt(i32);
            log.verbosity_thold = std.math.maxInt(i32);
        }
    }.f } });
    // arg.cpp:3987
    try ctx.add((Arg{ .args = &.{"--offline"}, .help = "Offline mode: forces use of cache, prevents network access", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.offline = true;
        }
    }.f } }).setExamples(&.{ .common, .download, .tokenize }).setEnv("LLAMA_ARG_OFFLINE"));
    // arg.cpp:3994
    try ctx.add((Arg{ .args = &.{ "-lv", "--verbosity", "--log-verbosity" }, .value_hint = "N", .help = try ctx.fmt("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n - 0: generic output\n - 1: error\n - 2: warning\n - 3: info\n - 4: trace (more info)\n - 5: debug\n(default: %d)\n", .{p.verbosity}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.verbosity = value;
            log.verbosity_thold = value;
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_VERBOSITY"));
    // arg.cpp:4009
    try ctx.add((Arg{ .args = &.{"--log-prefix"}, .args_neg = &.{"--no-log-prefix"}, .help = "Enable prefix in log messages", .handler = .{ .bool_ = struct {
        fn f(_: *Params, value: bool) !void {
            log.setPrefix(value);
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_PREFIX"));
    // arg.cpp:4017
    try ctx.add((Arg{ .args = &.{"--log-timestamps"}, .args_neg = &.{"--no-log-timestamps"}, .help = "Enable timestamps in log messages", .handler = .{ .bool_ = struct {
        fn f(_: *Params, value: bool) !void {
            log.setTimestamps(value);
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_TIMESTAMPS"));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "parentPath is path::parent_path's" {
    const cases = [_][2][]const u8{
        .{ "a/b", "a" },   .{ "a/b/", "a/b" }, .{ "a//b", "a" }, .{ "a", "" },
        .{ "/a", "/" },    .{ "/", "/" },      .{ "", "" },      .{ "a/b//", "a/b" },
        .{ "/a/b", "/a" },
    };
    for (cases) |cs| try std.testing.expectEqualStrings(cs[1], parentPath(cs[0]));
}
