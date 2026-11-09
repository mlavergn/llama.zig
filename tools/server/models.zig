//! Router mode: with no model on the command line, `llama-server` serves a
//! set of models -- from the cache, a `--models-dir` and a
//! `--models-preset` INI file -- each in a child `llama-server` it spawns
//! on demand, proxies requests to, and stops when it is unloaded, evicted
//! for `--models-max`, or the router exits. A child reports its state on a
//! pipe; the router reports it on `/models` and `/models/sse`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-models.cpp` and
//! `llama.cpp/tools/server/server-models.h` (v0.8.0, `8642541b2`): the
//! monitor thread, `server_models`, `server_models_routes` with every
//! handler, and the helpers above them; and `server_role`, the router's
//! protocol with a child, from `llama.cpp/tools/server/server-common.cpp` and
//! `llama.cpp/tools/server/server-common.h`, with `get_server_exec_path`.
//! `server_models_sched` is `models_sched.zig`. `server_http_proxy`, which
//! the CORS proxy also constructs, is `proxy.zig`. Each declaration names the
//! C++ it replaces and the line it began at.
//!
//! # Its shape
//!
//! `server_models::mapping`, a `std::map<std::string, instance_t>`, is a
//! list of heap instances kept in name order. Each instance owns an arena
//! its `meta` lives in; replacing the instance (`mapping[name] = inst`)
//! frees the old arena, and updating a field in place allocates the new
//! value there. The child process is the C++'s `shared_ptr<server_subproc>`
//! -- shared between the instance and the monitor -- as a reference-counted
//! `SharedProc`. `get_meta`'s `std::optional<server_model_meta>` copy is a
//! deep copy into the caller's allocator.
//!
//! Every method that the C++ guards with `mutex` takes the same lock here;
//! the condition variable is a pthread one, as `queue.zig`'s are.

const std = @import("std");
const builtin = @import("builtin");
const tc = @import("tools_common");
const json = @import("json");
const common = @import("common.zig");
const http = @import("http.zig");
const preset_mod = @import("preset.zig");
const download = @import("download.zig");
const hf_cache = @import("hf_cache.zig");
const subproc_mod = @import("subproc.zig");
const queue_mod = @import("queue.zig");
const result_mod = @import("result.zig");
const stream = @import("stream.zig");
const context = @import("context.zig");
const proxy_mod = @import("proxy.zig");
const client = @import("httplib/client.zig");
const ws_mod = @import("httplib/ws.zig");
const models_sched = @import("models_sched.zig");

const c = tc.c;
const log = tc.log;
const Params = tc.params.Params;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Preset = preset_mod.Preset;
const Presets = preset_mod.Presets;
const Error = common.Error;
const throw = common.throw;
const srv = common.srv;
const gpa = context.gpa;
const Result = result_mod.Result;
const ResultQueue = queue_mod.Response(*Result);

/// Ports `DEFAULT_STOP_TIMEOUT` (server-models.cpp:40 @8642541b2): seconds.
const default_stop_timeout = 10;

/// Ports `SERVER_ENV_ROUTER_PORT` and `SERVER_ENV_CHILD_ROLE`
/// (server-common.h:765, 766 @8642541b2): set by the router in the
/// environment of every child. The first's presence makes the process a
/// child; the second is `server_role_type_to_str`, an instance when unset.
pub const env_router_port = "LLAMA_SERVER_ROUTER_PORT";
pub const env_child_role = "LLAMA_SERVER_CHILD_ROLE";

/// Ports `CMD_ROUTER_TO_CHILD_EXIT` and `CMD_CHILD_TO_ROUTER_STATE`
/// (server-common.h:770, 771 @8642541b2).
const cmd_router_to_child_exit = "cmd_router_to_child:exit";
const cmd_child_to_router_state = "cmd_child_to_router:state:"; // followed by json string

/// Ports `CHILD_ADDR` (server-common.h:775 @8642541b2): the address a child
/// binds to, as the router may listen on 0.0.0.0.
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
const child_addr = "127.0.0.1";

/// Ports `STREAM_LOOKUP_TIMEOUT_MS` (server-models.cpp:266 @8642541b2): a short
/// loopback budget for the resumable stream router to child JSON calls
/// (probe, lookup, delete), distinct from `--timeout` which only applies to
/// the generation proxy.
const stream_lookup_timeout_ms = 250;

fn zs(a: Allocator, s: []const u8) Allocator.Error![*:0]const u8 {
    return (try a.dupeSentinel(u8, s, 0)).ptr;
}

fn timeMs() i64 {
    return c.ggml_time_ms();
}

/// A `std::condition_variable` paired with `server_models::mutex`.
const Mutex = struct {
    m: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    pub fn lock(self: *Mutex) void {
        _ = std.c.pthread_mutex_lock(&self.m);
    }
    pub fn unlock(self: *Mutex) void {
        _ = std.c.pthread_mutex_unlock(&self.m);
    }
};

const Cond = struct {
    c: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
    pub fn wait(self: *Cond, m: *Mutex) void {
        _ = std.c.pthread_cond_wait(&self.c, &m.m);
    }
    fn waitForMs(self: *Cond, m: *Mutex, ms: u64) void {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
        const ns: u64 = @as(u64, @intCast(ts.nsec)) + (ms % 1000) * std.time.ns_per_ms;
        ts.sec += @intCast(ms / 1000 + ns / std.time.ns_per_s);
        ts.nsec = @intCast(ns % std.time.ns_per_s);
        _ = std.c.pthread_cond_timedwait(&self.c, &m.m, &ts);
    }
    pub fn notifyAll(self: *Cond) void {
        _ = std.c.pthread_cond_broadcast(&self.c);
    }
};

// -----------------------------------------------------------------------------
// the child process

/// `server_subproc` over `common_subproc`.
pub const ServerSubproc = subproc_mod.Subproc(tc.subproc.Subproc);

/// `std::shared_ptr<server_subproc>`: shared with the monitor thread.
pub const SharedProc = struct {
    refs: std.atomic.Value(u32) = .init(1),
    proc: ServerSubproc,

    pub fn create() Allocator.Error!*SharedProc {
        const self = try gpa.create(SharedProc);
        self.* = .{ .proc = .{ .sproc = .{} } };
        return self;
    }

    pub fn acquire(self: *SharedProc) *SharedProc {
        _ = self.refs.fetchAdd(1, .monotonic);
        return self;
    }

    pub fn release(self: *SharedProc) void {
        if (self.refs.fetchSub(1, .acq_rel) == 1) {
            self.proc.sproc.deinit();
            gpa.destroy(self);
        }
    }
};

extern fn fprintf(stream: *anyopaque, format: [*:0]const u8, ...) c_int;
extern fn time(t: ?*i64) i64;
extern fn fflush(stream: ?*anyopaque) c_int;
extern fn fclose(stream: *anyopaque) c_int;

/// Ports `request_child_exit` (server-models.cpp:43 @8642541b2): the exit
/// command on the child's stdin. SIGPIPE is ignored by the server.
fn requestChildExit(proc: *ServerSubproc) void {
    if (proc.sproc.stdinFile()) |stdin_file| {
        _ = fprintf(stdin_file, "%s\n", @as([*:0]const u8, cmd_router_to_child_exit));
        _ = fflush(stdin_file);
    }
}

// -----------------------------------------------------------------------------
// server_model_meta

/// Ports `server_model_status` (server-models.h:35 @8642541b2).
///
/// state diagram:
///
/// DOWNLOADING ──► DOWNLOADED ──► (replaced by new instance)
///
/// UNLOADED ──► LOADING ──► LOADED ◄──── SLEEPING
///  ▲            │            │               ▲
///  └───failed───┘            │               │
///  ▲                         └──sleeping─────┘
///  └────────unloaded─────────┘
pub const Status = enum {
    downloading,
    incomplete, // a download that never finished, listed without an instance
    downloaded,
    unloaded,
    loading,
    loaded,
    sleeping,
};

/// Ports `server_model_status_to_string` (server-models.h:52 @8642541b2).
pub fn statusToString(status: Status) []const u8 {
    return switch (status) {
        .downloading => "downloading",
        .incomplete => "incomplete",
        .downloaded => "downloaded",
        .unloaded => "unloaded",
        .loading => "loading",
        .loaded => "loaded",
        .sleeping => "sleeping",
    };
}

/// Ports `server_model_source` (server-models.h:46 @8642541b2).
pub const Source = enum { preset, models_dir, cache };

/// Ports `server_model_source_to_string` (server-models.h:65 @8642541b2).
pub fn sourceToString(source: Source) []const u8 {
    return switch (source) {
        .preset => "preset",
        .models_dir => "models_dir",
        .cache => "cache",
    };
}

/// Ports `server_role_type` (server-common.h:736 @8642541b2): what this
/// llama-server process does. A router spawns the other roles as child
/// processes and talks to them over their stdin/stdout.
pub const RoleType = enum {
    router, // no model: lists the models, spawns instances on demand and proxies requests to them
    instance, // serves one model, standalone or as a child of a router
    download, // child of a router: download a model, report the progress, exit
    measure, // child of a router: report the memory the model needs without loading it, exit
};

/// Ports `server_role_type_to_str` (server-common.h:743 @8642541b2).
pub fn roleTypeToStr(role: RoleType) []const u8 {
    return switch (role) {
        .router => "router",
        .instance => "instance",
        .download => "download",
        .measure => "measure",
    };
}

/// Ports `server_role_type_from_str` (server-common.h:754 @8642541b2): the
/// role a router asks of a child; anything it does not name is an instance.
fn roleTypeFromStr(str: []const u8) RoleType {
    if (std.mem.eql(u8, str, "download")) return .download;
    if (std.mem.eql(u8, str, "measure")) return .measure;
    return .instance;
}

/// A `std::set<std::string>`: sorted, unique.
const StringSet = struct {
    items: std.ArrayList([]const u8) = .empty,

    fn insert(self: *StringSet, a: Allocator, s: []const u8) Allocator.Error!void {
        var lo: usize = 0;
        var hi = self.items.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (std.mem.order(u8, self.items.items[mid], s) == .lt) lo = mid + 1 else hi = mid;
        }
        if (lo < self.items.items.len and std.mem.eql(u8, self.items.items[lo], s)) return;
        try self.items.insert(a, lo, try a.dupe(u8, s));
    }

    pub fn contains(self: *const StringSet, s: []const u8) bool {
        for (self.items.items) |x| if (std.mem.eql(u8, x, s)) return true;
        return false;
    }

    fn clone(self: *const StringSet, a: Allocator) Allocator.Error!StringSet {
        var out: StringSet = .{};
        for (self.items.items) |s| try out.items.append(a, try a.dupe(u8, s));
        return out;
    }

    fn clear(self: *StringSet) void {
        self.items = .empty;
    }

    fn joinComma(self: *const StringSet, a: Allocator) Allocator.Error![]const u8 {
        return std.mem.join(a, ", ", self.items.items);
    }
};

/// Ports `server_model_meta` (server-models.h:74 @8642541b2). Its strings
/// live in the allocator it was built or cloned with. `architecture` is
/// the C++'s default (`server_model_architecture_json(false, false, false,
/// {"text"})`) from the first `update_caps`, which `add_model` always runs,
/// or from `Instance.create`.
pub const Meta = struct {
    source: Source = .cache,
    preset: Preset = .{},
    name: []const u8 = "",
    aliases: StringSet = .{}, // additional names that resolve to this model
    tags: StringSet = .{}, // informational tags, not used for routing
    port: c_int = 0,
    status: Status = .unloaded,
    last_used: i64 = 0, // for LRU unloading
    args: []const []const u8 = &.{}, // args passed to the model instance, will be populated by render_args()
    loaded_info: Value = .null, // info to be reflected via /v1/models endpoint ; if in DOWNLOADING state, it should contain download progress info
    progress: Value = .null, // reflect load or download progress info, if any
    exit_code: c_int = 0, // exit code of the model instance process (only valid if status == FAILED)
    stop_timeout: c_int = 0, // seconds to wait before force-killing the model instance during shutdown
    hidden: bool = false, // hidden from GET /models, but still accept if requested
    architecture: Value = .null,
    n_ctx_train: u32 = 0, // trained context, read from the GGUF metadata; 0 when unknown

    // only populated with --models-memory-margin, see server_models_sched
    memory: models_sched.MemoryInfo = .{},

    /// The C++ copy: every string and value deep-copied into `a`.
    pub fn clone(self: *const Meta, a: Allocator) Error!Meta {
        var out = self.*;
        out.preset = try presetCall(self.preset.clone(a));
        out.name = try a.dupe(u8, self.name);
        out.aliases = try self.aliases.clone(a);
        out.tags = try self.tags.clone(a);
        const args = try a.alloc([]const u8, self.args.len);
        for (self.args, 0..) |s, i| args[i] = try a.dupe(u8, s);
        out.args = args;
        out.loaded_info = try self.loaded_info.clone(a);
        out.progress = try self.progress.clone(a);
        out.architecture = try self.architecture.clone(a);
        out.memory = try self.memory.clone(a);
        return out;
    }

    /// Ports `is_ready` (server-models.h:95 @8642541b2).
    pub fn isReady(self: *const Meta) bool {
        return self.status == .loaded;
    }

    /// Ports `is_running` (server-models.h:99 @8642541b2).
    pub fn isRunning(self: *const Meta) bool {
        return self.status == .loaded or self.status == .loading or self.status == .sleeping;
    }

    /// Ports `is_ready_or_sleep` (server-models.h:103 @8642541b2).
    pub fn isReadyOrSleep(self: *const Meta) bool {
        return self.status == .loaded or self.status == .sleeping;
    }

    /// Ports `is_failed` (server-models.h:107 @8642541b2).
    pub fn isFailed(self: *const Meta) bool {
        return self.status == .unloaded and self.exit_code != 0;
    }

    /// Ports `server_model_meta`'s `update_args` (server-models.cpp:312
    /// @8642541b2): the child's command line from the preset, with the
    /// router's host, the child's port, its name as alias and the router's
    /// log colours.
    pub fn updateArgs(self: *Meta, a: Allocator, ctx_preset: *const preset_mod.Context, bin_path: []const u8) Error!void {
        // update params
        unsetReservedArgs(a, &self.preset, false);
        try presetCall(self.preset.setOption(a, ctx_preset, "LLAMA_ARG_HOST", child_addr));
        try presetCall(self.preset.setOption(a, ctx_preset, "LLAMA_ARG_PORT", try std.fmt.allocPrint(a, "{d}", .{self.port})));
        try presetCall(self.preset.setOption(a, ctx_preset, "LLAMA_ARG_ALIAS", self.name));
        // the child output goes through the router to its terminal, so it follows the router colors
        try presetCall(self.preset.setOption(a, ctx_preset, "LLAMA_ARG_LOG_COLORS", if (log.getColors()) "on" else "off"));
        // TODO: maybe validate preset before rendering ?
        // render args
        var args = try presetCall(self.preset.toArgs(a, bin_path));
        for (args.items) |*s| s.* = try a.dupe(u8, s.*);

        // unified binary dispatches by subcommand, re-inject it right after the
        // binary path so the child starts as 'llama serve ...' not 'llama ...'
        if (std.c.getenv("LLAMA_APP_CMD")) |app_cmd| {
            if (app_cmd[0] != 0 and bin_path.len != 0) {
                try args.insert(a, 1, try a.dupe(u8, std.mem.span(app_cmd)));
            }
        }
        self.args = args.items;
    }

    /// Ports `server_model_meta`'s `update_caps` (server-models.cpp:332
    /// @8642541b2): the model's input modalities from its projector and
    /// its output modalities from its GGUF metadata, resolved from the
    /// cache without downloading. `base` is the router's own parameters:
    /// its model and `--no-mmproj` apply to every child.
    pub fn updateCaps(self: *Meta, a: Allocator, base: *const Params) Error!void {
        // reset to the default so a failed refresh cannot keep old values
        self.architecture = try common.serverModelArchitectureJson(a, false, false, false, &.{"text"});
        self.n_ctx_train = 0;

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const sa = arena.allocator();

        // resolve the model file offline; do not download
        var params = Params.init(sa) catch return error.OutOfMemory;
        params.model = base.model;
        // --no-mmproj applies to child models and blocks auto-attached projectors
        params.no_mmproj = base.no_mmproj;
        self.resolveModel(sa, &params) catch |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            const w = if (e == error.Throw) tc.arg.what() else context.whatOf(e);
            log.wrn("failed to resolve the model of '%s': %s\n", .{ try zs(sa, self.name), try zs(sa, w) });
            return;
        };

        var inp_image = false;
        var inp_audio = false;
        var gen_audio = false;
        if (!params.no_mmproj and params.mmproj.path.len != 0) {
            // mtmd_get_cap_from_file() returns no capabilities on a bad file, it does not throw
            const caps = c.mtmd_get_cap_from_file(try zs(sa, params.mmproj.path));
            inp_image = caps.inp_vision;
            inp_audio = caps.inp_audio;
            gen_audio = caps.gen_audio;
        }

        // read the output modalities and the trained context from the GGUF metadata
        var output_modalities: []const []const u8 = &.{"text"};
        if (params.model.path.len != 0) {
            const info = tc.common.getGgufInfo(params.model.path);
            // same rule as mtmd_helper_get_capabilities()
            const can_chat = !gen_audio or info.has_chat_tmpl;
            output_modalities = common.serverModelOutputModalities(info.decision_type, can_chat, gen_audio);
            self.n_ctx_train = info.n_ctx_train;
        }

        // offline discovery cannot see video; a loaded model reports it
        self.architecture = try common.serverModelArchitectureJson(a, inp_image, inp_audio, false, output_modalities);
    }

    fn resolveModel(self: *Meta, a: Allocator, params: *Params) anyerror!void {
        try self.preset.applyToParams(params, &.{
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_MMPROJ_AUTO",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_FILE",
        });
        params.offline = true;
        var handler = try download.modelsHandlerInit(a, params, .server);
        try download.modelsHandlerApply(a, &handler, params, null);
    }
};

/// The `architecture` arm of `update_status` (server-models.cpp:1397
/// @8642541b2), from its line 1416: each of the loaded child's `input_modalities` and
/// `output_modalities` that is an array of strings replaces the router's.
fn mergeChildArchitecture(a: Allocator, arch: *Value, loaded_info: *const Value) !void {
    if (!loaded_info.contains("architecture") or !(try loaded_info.at("architecture")).isObject()) {
        return;
    }
    const child_arch = try loaded_info.at("architecture");
    for ([_][]const u8{ "input_modalities", "output_modalities" }) |key| {
        if (!child_arch.contains(key) or !(try child_arch.at(key)).isArray()) {
            continue;
        }
        var modalities = Value.emptyArray();
        var valid = true;
        for ((try child_arch.at(key)).array.items) |*m| {
            if (!m.isString()) {
                valid = false;
                break;
            }
            try modalities.pushBack(a, try Value.initString(a, m.string));
        }
        if (valid) {
            try arch.set(a, key, modalities);
        }
    }
}

/// A preset method's `error.Throw` as the `std::runtime_error` it is.
fn presetCall(r: anytype) Error!@typeInfo(@TypeOf(r)).error_union.payload {
    return r catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Throw => throw(error.RuntimeError, "{s}", .{tc.arg.what()}),
    };
}

/// Ports `unset_reserved_args` (server-models.cpp:268 @8642541b2): the
/// options a child must not inherit.
fn unsetReservedArgs(a: Allocator, preset: *Preset, unset_model_args: bool) void {
    preset.unsetOption(a, "LLAMA_ARG_SSL_KEY_FILE");
    preset.unsetOption(a, "LLAMA_ARG_SSL_CERT_FILE");
    preset.unsetOption(a, "LLAMA_API_KEY");
    preset.unsetOption(a, "LLAMA_ARG_API_KEY_FILE");
    preset.unsetOption(a, "LLAMA_ARG_MODELS_DIR");
    preset.unsetOption(a, "LLAMA_ARG_MODELS_MAX");
    preset.unsetOption(a, "LLAMA_ARG_MODELS_MEMORY_MARGIN");
    preset.unsetOption(a, "LLAMA_ARG_MODELS_PRESET");
    preset.unsetOption(a, "LLAMA_ARG_MODELS_AUTOLOAD");
    preset.unsetOption(a, "LLAMA_ARG_CONNECT");
    preset.unsetOption(a, "LLAMA_ARG_CONNECT_CODE");
    if (unset_model_args) {
        preset.unsetOption(a, "LLAMA_ARG_MODEL");
        preset.unsetOption(a, "LLAMA_ARG_MMPROJ");
        preset.unsetOption(a, "LLAMA_ARG_ALIAS");
        preset.unsetOption(a, "LLAMA_ARG_HF_REPO");
    }
}

extern fn _NSGetExecutablePath(buf: [*]u8, size: *u32) c_int;
extern fn realpath(path: [*:0]const u8, resolved: ?[*]u8) ?[*:0]u8;
extern fn readlink(path: [*:0]const u8, buf: [*]u8, len: usize) isize;

/// Ports `get_server_exec_path` (server-common.cpp:52 @8642541b2), its
/// macOS and Linux arms: this program's path, with symbolic links resolved
/// on macOS.
pub fn getServerExecPath(a: Allocator) Error![]const u8 {
    if (comptime builtin.os.tag.isDarwin()) {
        var small_path: [std.c.PATH_MAX]u8 = undefined;
        var size: u32 = small_path.len;

        if (_NSGetExecutablePath(&small_path, &size) == 0) {
            // resolve any symlinks to get absolute path
            const p: [*:0]const u8 = @ptrCast(&small_path);
            if (realpath(p, null)) |r| {
                defer std.c.free(r);
                return a.dupe(u8, std.mem.span(r));
            }
            return a.dupe(u8, std.mem.span(p));
        } else {
            // buffer was too small, allocate required size and call again
            const buf = try a.alloc(u8, size);
            if (_NSGetExecutablePath(buf.ptr, &size) == 0) {
                const p: [*:0]const u8 = @ptrCast(buf.ptr);
                if (realpath(p, null)) |r| {
                    defer std.c.free(r);
                    return a.dupe(u8, std.mem.span(r));
                }
                return a.dupe(u8, std.mem.span(p));
            }
            return throw(error.RuntimeError, "_NSGetExecutablePath failed after buffer resize", .{});
        }
    } else {
        var path: [4096]u8 = undefined;
        const count = readlink("/proc/self/exe", &path, path.len);
        if (count <= 0) {
            return throw(error.RuntimeError, "failed to resolve /proc/self/exe", .{});
        }
        return a.dupe(u8, path[0..@intCast(count)]);
    }
}

extern var environ: [*:null]?[*:0]u8;

/// Ports `get_environment` (server-models.cpp:288 @8642541b2), its POSIX
/// arm.
fn getEnvironment(a: Allocator) Allocator.Error![]const []const u8 {
    var env: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (environ[i]) |e| : (i += 1) {
        try env.append(a, try a.dupe(u8, std.mem.span(e)));
    }
    return env.items;
}

// -----------------------------------------------------------------------------
// server_monitor

/// Ports `server_monitor` (server-models.cpp:52 @8642541b2): one thread
/// watching every child at once -- forwarding its output, handing its state
/// lines to the router, force-killing it past its stop deadline, and
/// reaping it.
const Monitor = struct {
    const max_line = 1024 * 1024;

    /// Ports `child_t` (server-models.cpp:86 @8642541b2).
    const Watched = struct {
        name: []const u8 = "",
        proc: *SharedProc,
        role: RoleType = .instance,
        port: c_int = 0,
        buf: [subproc_mod.n_streams]std.ArrayList(u8) = .{ .empty, .empty }, // partial line of each pipe
        eof: bool = false, // output closed, waiting for the process to be reaped
        deadline: i64 = 0, // force-kill time in ms, 0 when no stop is pending

        fn deinit(self: *Watched) void {
            gpa.free(self.name);
            for (&self.buf) |*b| b.deinit(gpa);
            self.proc.release();
        }
    };

    /// Ports `cmd_t` (server-models.cpp:96 @8642541b2).
    const Cmd = struct {
        type: enum { watch, stop, quit },
        child: ?Watched = null,
        name: []const u8 = "",
        stop_timeout: c_int = 0,
        send_exit: bool = false,
    };

    models: *Models,
    mu: Mutex = .{},
    cmds: std.ArrayList(Cmd) = .empty,
    children: std.ArrayList(Watched) = .empty, // monitor thread only
    waiter: subproc_mod.Waiter,
    th: ?std.Thread = null,

    /// Ports the constructor of `server_monitor` (server-models.cpp:53
    /// @8642541b2): starts the thread.
    fn create(models: *Models) !*Monitor {
        const self = try gpa.create(Monitor);
        self.* = .{ .models = models, .waiter = subproc_mod.Waiter.init() };
        self.th = try std.Thread.spawn(.{}, run, .{self});
        return self;
    }

    /// Ports the destructor of `server_monitor` (server-models.cpp:57
    /// @8642541b2).
    fn destroy(self: *Monitor) void {
        self.push(.{ .type = .quit }) catch |err| switch (err) {
            error.OutOfMemory => {},
        };
        if (self.th) |t| t.join();
        for (self.children.items) |*ch| ch.deinit();
        self.children.deinit(gpa);
        self.cmds.deinit(gpa);
        self.waiter.deinit();
        gpa.destroy(self);
    }

    /// Ports `watch` (server-models.cpp:64 @8642541b2), a member of
    /// `server_monitor`: thread-safe. `kill_after` above 0 force-kills the
    /// child after that many seconds.
    fn watch(self: *Monitor, name: []const u8, proc: *SharedProc, role: RoleType, port: c_int, kill_after: c_int) !void {
        var ch: Watched = .{ .name = try gpa.dupe(u8, name), .proc = proc.acquire(), .role = role, .port = port };
        if (kill_after > 0) {
            ch.deadline = timeMs() + @as(i64, kill_after) * 1000;
        }
        if (!ch.proc.proc.hasOutput()) {
            const z = try zs(gpa, name);
            defer gpa.free(std.mem.span(z));
            srv(.err, "watch", "failed to get stdout/stderr of child process for name=%s\n", .{z});
            ch.eof = true;
        }
        try self.push(.{ .type = .watch, .child = ch });
    }

    /// Ports `stop` (server-models.cpp:81 @8642541b2), a member of
    /// `server_monitor`: thread-safe.
    fn stop(self: *Monitor, name: []const u8, stop_timeout: c_int, send_exit: bool) !void {
        try self.push(.{ .type = .stop, .name = try gpa.dupe(u8, name), .stop_timeout = stop_timeout, .send_exit = send_exit });
    }

    /// Ports `push` (server-models.cpp:104 @8642541b2), a member of
    /// `server_monitor`.
    fn push(self: *Monitor, cmd: Cmd) !void {
        {
            self.mu.lock();
            defer self.mu.unlock();
            try self.cmds.append(gpa, cmd);
        }
        self.waiter.wake();
    }

    /// Ports `handle_commands` (server-models.cpp:113 @8642541b2), a member
    /// of `server_monitor`.
    ///
    /// Return: true if the loop should exit.
    fn handleCommands(self: *Monitor) bool {
        var batch: std.ArrayList(Cmd) = .empty;
        defer batch.deinit(gpa);
        {
            self.mu.lock();
            defer self.mu.unlock();
            std.mem.swap(std.ArrayList(Cmd), &batch, &self.cmds);
        }
        var quit = false;
        for (batch.items) |*cmd| {
            if (quit) {
                if (cmd.child) |*ch| ch.deinit();
                if (cmd.type == .stop) gpa.free(cmd.name);
                continue;
            }
            switch (cmd.type) {
                .watch => self.children.append(gpa, cmd.child.?) catch |err| switch (err) {
                    error.OutOfMemory => {},
                },
                .stop => {
                    defer gpa.free(cmd.name);
                    // the newest child with this name is the one the registry knows
                    var i = self.children.items.len;
                    while (i > 0) {
                        i -= 1;
                        const it = &self.children.items[i];
                        // a measure child has the same name, skip it
                        if (!std.mem.eql(u8, it.name, cmd.name) or it.role == .measure) {
                            continue;
                        }
                        if (cmd.send_exit and !it.eof) {
                            requestChildExit(&it.proc.proc);
                        }
                        it.deadline = timeMs() + @as(i64, cmd.stop_timeout) * 1000;
                        break;
                    }
                },
                .quit => quit = true,
            }
        }
        return quit;
    }

    /// Ports `read_output` (server-models.cpp:146 @8642541b2), a member of
    /// `server_monitor`: what the child wrote on both pipes -- its commands
    /// handled and its logs forwarded, line by line.
    fn readOutput(self: *Monitor, ch: *Watched) void {
        for ([_]subproc_mod.Stream{ .stdout, .stderr }) |which| {
            self.readStream(ch, which);
        }
        ch.eof = ch.proc.proc.outputClosed();
    }

    /// Ports `read_stream` (server-models.cpp:153 @8642541b2), a member of
    /// `server_monitor`.
    fn readStream(self: *Monitor, ch: *Watched, which: subproc_mod.Stream) void {
        var chunk: [4096]u8 = undefined;
        const buf = &ch.buf[@backingInt(which)];
        var closed = false;
        while (true) {
            const n = ch.proc.proc.readOutput(which, &chunk);
            if (n < 0) {
                closed = true;
                break;
            }
            if (n == 0) {
                break;
            }
            buf.appendSlice(gpa, chunk[0..@intCast(n)]) catch |err| switch (err) {
                error.OutOfMemory => {},
            };
            var start: usize = 0;
            while (true) {
                const nl = std.mem.indexOfScalarPos(u8, buf.items, start, '\n') orelse break;
                self.onLine(ch, which, buf.items[start .. nl + 1]);
                start = nl + 1;
            }
            std.mem.copyForwards(u8, buf.items[0 .. buf.items.len - start], buf.items[start..]);
            buf.shrinkRetainingCapacity(buf.items.len - start);
            if (buf.items.len > max_line) {
                buf.clearRetainingCapacity(); // a child that never writes a newline must not grow this without bound
            }
        }
        if (closed and buf.items.len != 0) {
            self.onLine(ch, which, buf.items);
            buf.clearRetainingCapacity();
        }
    }

    /// Ports `on_line` (server-models.cpp:187 @8642541b2), a member of
    /// `server_monitor`: stderr is the child's log, stdout its commands.
    fn onLine(self: *Monitor, ch: *Watched, which: subproc_mod.Stream, line: []const u8) void {
        const z = gpa.dupeSentinel(u8, line, 0) catch return;
        defer gpa.free(z);
        if (which == .stderr) {
            log.out("[%5d] %s", .{ ch.port, z.ptr }); // forward log
        } else if (std.mem.startsWith(u8, line, cmd_child_to_router_state)) {
            log.dbg("[%5d] %s", .{ ch.port, z.ptr }); // prevent spamming the log
            self.models.handleChildState(ch.name, line);
        } else {
            srv(.wrn, "on_line", "[%5d] unexpected output on the command pipe: %s", .{ ch.port, z.ptr });
        }
    }

    /// Ports `run` (server-models.cpp:198 @8642541b2), a member of
    /// `server_monitor`.
    fn run(self: *Monitor) void {
        var procs: std.ArrayList(*ServerSubproc) = .empty;
        defer procs.deinit(gpa);
        var owners: std.ArrayList(usize) = .empty;
        defer owners.deinit(gpa);
        var ready: std.ArrayList(bool) = .empty;
        defer ready.deinit(gpa);
        while (true) {
            if (self.handleCommands()) {
                return;
            }

            // wait for output, a wakeup, or the next deadline;
            // a child whose output closed is polled for its exit every 50 ms
            var now = timeMs();
            var timeout: i64 = -1;
            for (self.children.items) |ch| {
                if (ch.eof) {
                    timeout = if (timeout < 0) 50 else @min(timeout, 50);
                }
                if (ch.deadline != 0) {
                    const d = @max(0, ch.deadline - now);
                    timeout = if (timeout < 0) d else @min(timeout, d);
                }
            }
            procs.clearRetainingCapacity();
            owners.clearRetainingCapacity();
            for (self.children.items, 0..) |*ch, i| {
                if (!ch.eof) {
                    procs.append(gpa, &ch.proc.proc) catch |err| switch (err) {
                        error.OutOfMemory => {},
                    };
                    owners.append(gpa, i) catch |err| switch (err) {
                        error.OutOfMemory => {},
                    };
                }
            }
            ready.resize(gpa, procs.items.len) catch |err| switch (err) {
                error.OutOfMemory => {},
            };
            self.waiter.wait(gpa, procs.items, ready.items, timeout) catch |err| switch (err) {
                error.OutOfMemory => {},
            };
            for (owners.items, 0..) |idx, i| {
                if (ready.items[i]) {
                    self.readOutput(&self.children.items[idx]);
                }
            }

            // deadlines and exits
            now = timeMs();
            var i: usize = 0;
            while (i < self.children.items.len) {
                const it = &self.children.items[i];
                if (it.deadline != 0 and now >= it.deadline and !it.proc.proc.stopped.load(.acquire)) {
                    const z = zs(gpa, it.name) catch "";
                    srv(.wrn, "run", "force-killing child process for name=%s after timeout\n", .{z});
                    it.proc.proc.terminate();
                    it.deadline = 0;
                }
                if (it.eof and !it.proc.proc.isAlive()) {
                    const exit_code = it.proc.proc.join();
                    it.proc.proc.stopped.store(true, .release);
                    self.models.onChildExit(it.name, it.proc, it.role, exit_code);
                    const z = zs(gpa, it.name) catch "";
                    srv(.inf, "run", "instance name=%s exited with status %d\n", .{ z, exit_code });
                    var removed = self.children.orderedRemove(i);
                    removed.deinit();
                } else {
                    i += 1;
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// server_models

/// Ports `instance_t` (server-models.h:169 @8642541b2), a member of
/// `server_models`.
pub const Instance = struct {
    arena: std.heap.ArenaAllocator,
    subproc: *SharedProc, // shared with the monitor thread
    meta: Meta,
    req_count: c_int = 0, // number of active proxy requests

    fn create(meta_src: *const Meta, subproc: *SharedProc) !*Instance {
        const self = try gpa.create(Instance);
        self.* = .{ .arena = .init(gpa), .subproc = subproc, .meta = undefined };
        errdefer {
            self.arena.deinit();
            gpa.destroy(self);
        }
        self.meta = try meta_src.clone(self.arena.allocator());
        // the member's initializer in the C++: a meta built by hand, as a
        // download's, has been through no `update_caps`
        if (self.meta.architecture == .null) {
            self.meta.architecture = try common.serverModelArchitectureJson(self.arena.allocator(), false, false, false, &.{"text"});
        }
        return self;
    }

    fn destroy(self: *Instance) void {
        self.subproc.release();
        self.arena.deinit();
        gpa.destroy(self);
    }

    pub fn alloc(self: *Instance) Allocator {
        return self.arena.allocator();
    }

    /// Ports `request_exit` (server-models.cpp:425 @8642541b2), a member of
    /// `instance_t`: ask the child to exit (it handles the command on its
    /// stdin, see `server_role::setup`).
    fn requestExit(self: *const Instance) void {
        requestChildExit(&self.subproc.proc);
    }
};

/// Ports `conv_model_tracker` (server-models.h:202 @8642541b2): conv_id ->
/// model name that currently serves its stream session, lets the resumable
/// stream routes go straight to the owning child instead of polling every
/// one. Populated when `proxy_request` forwards a POST carrying an
/// `X-Conversation-Id`; best effort. Owns its lock.
const ConvModelTracker = struct {
    const EntryT = struct {
        model: []const u8,
        ticket: u64,
    };
    mu: Mutex = .{},
    next_ticket: u64 = 1,
    map: std.StringHashMapUnmanaged(EntryT) = .empty,

    /// Ports `remember` (server-models.h:206 @8642541b2), a member of
    /// `conv_model_tracker`: the ticket of this registration, 0 when nothing
    /// was registered. Erasing or replacing the entry invalidates the
    /// ticket, which is how a stop cancels a request parked in the model
    /// load wait.
    fn remember(self: *ConvModelTracker, conv_id: []const u8, model: []const u8) !u64 {
        if (conv_id.len == 0 or model.len == 0) {
            return 0;
        }
        self.mu.lock();
        defer self.mu.unlock();
        const ticket = self.next_ticket;
        self.next_ticket += 1;
        const gop = try self.map.getOrPut(gpa, conv_id);
        if (gop.found_existing) {
            gpa.free(gop.value_ptr.model);
        } else {
            gop.key_ptr.* = try gpa.dupe(u8, conv_id);
        }
        gop.value_ptr.* = .{ .model = try gpa.dupe(u8, model), .ticket = ticket };
        return ticket;
    }

    /// Ports `alive` (server-models.h:217 @8642541b2), a member of
    /// `conv_model_tracker`: false means a stop erased the entry or a newer
    /// request replaced it.
    fn alive(self: *ConvModelTracker, conv_id: []const u8, ticket: u64) bool {
        self.mu.lock();
        defer self.mu.unlock();
        const e = self.map.get(conv_id) orelse return false;
        return e.ticket == ticket;
    }

    /// Ports `lookup` (server-models.h:223 @8642541b2), a member of
    /// `conv_model_tracker`.
    ///
    /// Return: a copy owned by `a`, or null.
    fn lookup(self: *ConvModelTracker, a: Allocator, conv_id: []const u8) !?[]const u8 {
        if (conv_id.len == 0) {
            return null;
        }
        self.mu.lock();
        defer self.mu.unlock();
        const e = self.map.get(conv_id) orelse return null;
        return try a.dupe(u8, e.model);
    }

    /// Ports `forget` (server-models.h:235 @8642541b2), a member of
    /// `conv_model_tracker`.
    fn forget(self: *ConvModelTracker, conv_id: []const u8) void {
        if (conv_id.len == 0) {
            return;
        }
        self.mu.lock();
        defer self.mu.unlock();
        if (self.map.fetchRemove(conv_id)) |kv| {
            gpa.free(kv.key);
            gpa.free(kv.value.model);
        }
    }
};

/// Ports `load_options` (server-models.h:322 @8642541b2), a member of
/// `server_models`.
pub const LoadOptions = struct {
    role: RoleType = .instance,
    // used for spawning a downloading child process
    custom_meta: ?*const Meta = null,
    // download mode: the sidecar kinds pulled with the model, as -hf would, and
    // the partial files the download plan writes
    sidecars: []const []const u8 = &.{},
    partials: []const download.Partial = &.{},
};

/// Ports `update_status_args` (server-models.h:339 @8642541b2), a member of
/// `server_models`.
pub const UpdateStatusArgs = struct {
    status: Status,
    exit_code: c_int = 0, // only valid if status == UNLOADED
    loaded_info: ?*const Value = null,
    progress: ?*const Value = null,
    memory: ?*const models_sched.ServerModelMemory = null, // what the instance allocated, sent with READY
};

/// Ports `server_unfinished_download` (server-models.h:117 @8642541b2): a
/// download that never finished: the tag it was asked under, the sidecar
/// kinds it pulls and the partial files its plan writes.
pub const UnfinishedDownload = struct {
    model: []const u8 = "",
    sidecars: []const []const u8 = &.{},
    partials: []const download.Partial = &.{},

    /// The C++ copy.
    fn clone(self: *const UnfinishedDownload, a: Allocator) Allocator.Error!UnfinishedDownload {
        const sidecars = try a.alloc([]const u8, self.sidecars.len);
        for (self.sidecars, sidecars) |s, *d| d.* = try a.dupe(u8, s);
        const partials = try a.alloc(download.Partial, self.partials.len);
        for (self.partials, partials) |s, *d| d.* = .{ .kind = try a.dupe(u8, s.kind), .path = try a.dupe(u8, s.path) };
        return .{ .model = try a.dupe(u8, self.model), .sidecars = sidecars, .partials = partials };
    }
};

/// Ports `has_sidecar` (server-models.cpp:487 @8642541b2).
fn hasSidecar(sidecars: []const []const u8, kind: []const u8) bool {
    for (sidecars) |s| if (std.mem.eql(u8, s, kind)) return true;
    return false;
}

/// Ports `check_sidecar` (server-models.cpp:492 @8642541b2): reject a
/// sidecar kind no download can pull.
fn checkSidecar(a: Allocator, kind: []const u8) Error!void {
    var kinds: [download.sidecars.len][]const u8 = undefined;
    for (download.sidecars, 0..) |s, i| kinds[i] = s.name;
    if (!hasSidecar(&kinds, kind)) {
        return throw(error.InvalidArgument, "unknown sidecar '{s}', expected one of {s}", .{ kind, try std.mem.join(a, ", ", &kinds) });
    }
}

/// Ports `parse_sidecars` (server-models.cpp:503 @8642541b2): the sidecars a
/// download asks for; by default the ones `-hf` pulls without a draft type.
fn parseSidecars(a: Allocator, body: *const Value) Error![]const []const u8 {
    var defaults: std.ArrayList([]const u8) = .empty;
    for (download.sidecars) |s| {
        if (!s.draft) {
            try defaults.append(a, s.name);
        }
    }
    var sidecars: []const []const u8 = defaults.items;
    if (body.contains("sidecars") and !(try body.at("sidecars")).isNull()) {
        const given = try body.at("sidecars");
        const well_typed = given.isArray() and for (given.array.items) |*v| {
            if (!v.isString()) break false;
        } else true;
        // `json_value` logs a wrong type and takes the default
        const list = try common.jsonValueStrings(a, body, "sidecars");
        if (well_typed) sidecars = list;
    }
    for (sidecars) |kind| {
        try checkSidecar(a, kind);
    }
    return sidecars;
}

/// Ports `sidecar_download_opts` (server-models.cpp:518 @8642541b2): the
/// download options of a model with its sidecars, the plan a removal purges
/// partials from.
fn sidecarDownloadOpts(params: *const Params, sidecars: []const []const u8) download.Opts {
    var opts: download.Opts = .{};
    opts.bearer_token = params.hf_token;
    for (sidecars) |kind| {
        _ = download.optsAddSidecar(&opts, kind);
    }
    return opts;
}

/// Ports `model_hf_repo` (server-models.cpp:529 @8642541b2): the HF repo a
/// model resolves from: a preset names its own, a cached model is listed
/// under it, and a local file or a models dir entry has none.
fn modelHfRepo(meta: *const Meta) []const u8 {
    if (meta.preset.getOption("LLAMA_ARG_HF_REPO")) |hf_repo| {
        if (hf_repo.len != 0) return hf_repo;
    }
    return if (meta.source == .cache) meta.name else "";
}

/// Ports `cached_sidecars` (server-models.cpp:538 @8642541b2): the sidecar
/// kinds of a HF repo present on disk, as the offline plan resolves them.
fn cachedSidecars(a: Allocator, hf_repo: []const u8) Error![]const []const u8 {
    var model: tc.params.Model = .{};
    model.hf_repo = hf_repo;

    var opts: download.Opts = .{};
    opts.offline = true;
    inline for (download.sidecars) |s| {
        @field(opts, s.opt) = true;
    }

    return download.planSidecars(a, &(try download.getHfPlan(a, &model, &opts)));
}

/// Ports `set_child_env` (server-models.cpp:552 @8642541b2): replace any
/// inherited value of `key` in a child environment.
fn setChildEnv(a: Allocator, env: *std.ArrayList([]const u8), key: []const u8, value: []const u8) Allocator.Error!void {
    const prefix = try std.mem.concat(a, u8, &.{ key, "=" });
    var i: usize = 0;
    while (i < env.items.len) {
        if (std.mem.startsWith(u8, env.items[i], prefix)) {
            _ = env.orderedRemove(i);
        } else {
            i += 1;
        }
    }
    try env.append(a, try std.mem.concat(a, u8, &.{ prefix, value }));
}

/// Ports `unfinished_downloads_path` (server-models.cpp:560 @8642541b2):
/// where the unfinished downloads are remembered, inside the model cache.
fn unfinishedDownloadsPath(a: Allocator) Error![]const u8 {
    return std.mem.concat(a, u8, &.{ try hf_cache.getCachePath(), "/unfinished-downloads.json" });
}

/// Ports `server_unfinished_downloads` (server-models.h:128 @8642541b2): the
/// downloads that never finished, recorded next to the model cache. The
/// cache names partial files after their content hash, so this record is the
/// only trace of what is still incomplete: it lets a restarted router list,
/// resume or remove them. A partial another recorded download also writes is
/// never deleted. Thread-safe. Its strings live in an arena of its own, for
/// the router's life: a record is a few short strings.
pub const UnfinishedDownloads = struct {
    mutex: Mutex = .{},
    downloads: std.ArrayList(UnfinishedDownload) = .empty,
    arena: std.heap.ArenaAllocator = .init(std.heap.c_allocator),

    fn alloc(self: *UnfinishedDownloads) Allocator {
        return self.arena.allocator();
    }

    fn deinit(self: *UnfinishedDownloads) void {
        self.arena.deinit();
    }

    /// Ports `load` (server-models.cpp:564 @8642541b2), a member of
    /// `server_unfinished_downloads`: read the record an earlier run left.
    fn load(self: *UnfinishedDownloads) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const path = try unfinishedDownloadsPath(sa);
        const z = try sa.dupeSentinel(u8, path, 0);
        const raw = readAll(sa, z) orelse return;
        self.mutex.lock();
        defer self.mutex.unlock();
        self.parse(sa, raw) catch |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            log.wrn("failed to read unfinished downloads: %s\n", .{try zs(sa, context.whatOf(e))});
        };
    }

    fn parse(self: *UnfinishedDownloads, sa: Allocator, raw: []const u8) Error!void {
        var data = try json.parse(sa, raw);
        // `data.value("downloads", json::array())`, on anything but an object a type_error
        if (!data.isObject()) {
            return json.errors.fail(.type_error, 306, "cannot use value() with {s}", .{data.typeName()});
        }
        if (!data.contains("downloads")) return;
        const list = try data.at("downloads");
        if (!list.isArray()) return;
        const a = self.alloc();
        for (list.array.items) |*d| {
            // `d.value("partials", ...)` below, on anything but an object
            if (!d.isObject()) {
                return json.errors.fail(.type_error, 306, "cannot use value() with {s}", .{d.typeName()});
            }
            var download_: UnfinishedDownload = .{};
            const sidecars = try common.jsonValueStrings(sa, d, "sidecars");
            download_.model = common.jsonValue(d, "model", []const u8, "");
            var partials: std.ArrayList(download.Partial) = .empty;
            if (d.contains("partials")) {
                const ps = try d.at("partials");
                if (ps.isArray()) {
                    for (ps.array.items) |*p| {
                        try partials.append(sa, .{ .kind = common.jsonValue(p, "kind", []const u8, ""), .path = common.jsonValue(p, "path", []const u8, "") });
                    }
                }
            }
            download_.sidecars = sidecars;
            download_.partials = partials.items;
            if (download_.model.len != 0) {
                try self.downloads.append(a, try download_.clone(a));
            }
        }
    }

    /// Ports `save` (server-models.cpp:589 @8642541b2), a member of
    /// `server_unfinished_downloads`: write the record back; the caller holds
    /// the mutex.
    fn save(self: *UnfinishedDownloads) void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        self.saveImpl(sa) catch |e| {
            log.wrn("failed to record unfinished downloads: %s\n", .{zs(sa, context.whatOf(e)) catch ""});
        };
    }

    fn saveImpl(self: *UnfinishedDownloads, sa: Allocator) Error!void {
        var list = Value.emptyArray();
        for (self.downloads.items) |*d| {
            var partials = Value.emptyArray();
            for (d.partials) |p| {
                var po = common.Obj.init(sa);
                try po.put("kind", p.kind);
                try po.put("path", p.path);
                try partials.pushBack(sa, po.done());
            }
            var o = common.Obj.init(sa);
            try o.put("model", d.model);
            try o.put("sidecars", try stringsArray(sa, d.sidecars));
            try o.put("partials", partials);
            try list.pushBack(sa, o.done());
        }
        var data = common.Obj.init(sa);
        try data.put("downloads", list);
        var v = data.done();
        try download.fsWriteAtomic(sa, try unfinishedDownloadsPath(sa), try json.dump(sa, &v, -1));
    }

    /// Ports `delete_partials` (server-models.cpp:606 @8642541b2), a member
    /// of `server_unfinished_downloads`: the partials of `parts` no recorded
    /// download writes; the caller holds the mutex.
    fn deletePartials(self: *UnfinishedDownloads, parts: []const download.Partial) void {
        for (parts) |p| {
            const shared = for (self.downloads.items) |*d| {
                const found = for (d.partials) |q| {
                    if (std.mem.eql(u8, q.path, p.path)) break true;
                } else false;
                if (found) break true;
            } else false;
            if (shared) {
                continue;
            }
            _ = hf_cache.removeFile(gpa, p.path);
        }
    }

    fn findIdx(self: *const UnfinishedDownloads, model: []const u8) ?usize {
        for (self.downloads.items, 0..) |*d, i| {
            if (std.mem.eql(u8, d.model, model)) return i;
        }
        return null;
    }

    /// Ports `add` (server-models.cpp:621 @8642541b2), a member of
    /// `server_unfinished_downloads`: record a download that starts,
    /// replacing an earlier record of the same model, and delete the partials
    /// of that record the new plan no longer writes.
    fn add(self: *UnfinishedDownloads, dl: *const UnfinishedDownload) Allocator.Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();
        var stale: []const download.Partial = &.{};
        const copy = try dl.clone(self.alloc());
        if (self.findIdx(dl.model)) |i| {
            stale = self.downloads.items[i].partials;
            self.downloads.items[i] = copy;
        } else {
            try self.downloads.append(self.alloc(), copy);
        }
        self.save();
        self.deletePartials(stale);
    }

    /// Ports `forget` (server-models.cpp:635 @8642541b2), a member of
    /// `server_unfinished_downloads`: forget a download whose files are
    /// complete.
    fn forget(self: *UnfinishedDownloads, model: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.findIdx(model)) |i| {
            _ = self.downloads.orderedRemove(i);
            self.save();
        }
    }

    /// Ports `remove` (server-models.cpp:644 @8642541b2), a member of
    /// `server_unfinished_downloads`: forget a download and delete its
    /// partials.
    ///
    /// Return: false when it is not recorded.
    fn remove(self: *UnfinishedDownloads, model: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const i = self.findIdx(model) orelse return false;
        const partials = self.downloads.items[i].partials;
        _ = self.downloads.orderedRemove(i);
        self.save();
        self.deletePartials(partials);
        return true;
    }

    /// Ports `remove_sidecar` (server-models.cpp:657 @8642541b2), a member of
    /// `server_unfinished_downloads`: drop the sidecar `kind` from a download
    /// and delete its partials.
    ///
    /// Return: false when the download is not recorded with that kind.
    fn removeSidecar(self: *UnfinishedDownloads, model: []const u8, kind: []const u8) Allocator.Error!bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const i = self.findIdx(model) orelse return false;
        const it = &self.downloads.items[i];
        if (!hasSidecar(it.sidecars, kind)) {
            return false;
        }
        const a = self.alloc();
        var sidecars: std.ArrayList([]const u8) = .empty;
        for (it.sidecars) |s| {
            if (!std.mem.eql(u8, s, kind)) try sidecars.append(a, s);
        }
        it.sidecars = sidecars.items;

        var kept: std.ArrayList(download.Partial) = .empty;
        var dropped: std.ArrayList(download.Partial) = .empty;
        for (it.partials) |p| {
            try (if (std.mem.eql(u8, p.kind, kind)) &dropped else &kept).append(a, p);
        }
        it.partials = kept.items;

        self.save();
        self.deletePartials(dropped.items);
        return true;
    }

    /// Ports `get_all` (server-models.cpp:678 @8642541b2), a member of
    /// `server_unfinished_downloads`: every recorded download.
    ///
    /// Return: a copy, owned by `a`.
    fn getAll(self: *UnfinishedDownloads, a: Allocator) Allocator.Error![]UnfinishedDownload {
        self.mutex.lock();
        defer self.mutex.unlock();
        const out = try a.alloc(UnfinishedDownload, self.downloads.items.len);
        for (self.downloads.items, out) |*s, *d| d.* = try s.clone(a);
        return out;
    }
};

/// `std::ifstream in(path)`, read whole: null when it does not open.
fn readAll(a: Allocator, path: [:0]const u8) ?[]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n <= 0) break;
        out.appendSlice(a, buf[0..@intCast(n)]) catch return null;
    }
    return out.items;
}

/// Ports `server_models` (server-models.h:163 @8642541b2).
pub const Models = struct {
    mutex: Mutex = .{},
    cv: Cond = .{},
    mapping: std.ArrayList(*Instance) = .empty,

    // downloads that never finished, with the sidecar kinds they pull
    unfinished_downloads: UnfinishedDownloads = .{},

    // models asked to stop, still counted as running until the monitor records their exit
    stopping_models: StringSet = .{},
    stopping_arena: std.heap.ArenaAllocator = .init(std.heap.c_allocator),

    // set to true while load_models() is executing a reload; load() will wait until clear
    is_reloading: bool = false,

    // if true, the next get_meta() will trigger a reload of model list
    need_reload: bool = false,

    // models marked with load-on-startup, unset once load_startup_models() drains it
    // no value means the startup phase is over, so a reload must not queue anything
    startup_models: ?std.ArrayList([]const u8) = .empty,

    ctx_preset: *preset_mod.Context,

    base_params: Params,
    bin_path: []const u8 = "",
    base_env: []const []const u8 = &.{},
    base_preset: Preset = .{}, // base preset from llama-server CLI args

    // request queue (models_max) and memory accounting
    sched: models_sched.Sched,

    // if true, add some delay to simulate works (useful for testing)
    debug_fake_timing: bool = false,

    // conv_id -> model tracker for the resumable stream routes, owns its lock
    conv_models: ConvModelTracker = .{},

    sse: ResultQueue, // for real-time updates via SSE endpoint

    // one thread watching every child; the destructor joins the thread
    monitor: ?*Monitor = null,

    /// Long-lived strings: the binary, the environment.
    arena: std.heap.ArenaAllocator = .init(std.heap.c_allocator),

    /// Ports the constructor of `server_models` (server-models.cpp:393
    /// @8642541b2). `self` must not move afterwards.
    pub fn init(self: *Models, params: *const Params, argv: []const [:0]const u8) !void {
        self.* = .{
            .ctx_preset = try preset_mod.Context.create(gpa, .server),
            .base_params = params.*,
            .sched = .{ .models = self },
            .sse = try ResultQueue.init(gpa),
        };
        const a = self.arena.allocator();
        self.base_env = try getEnvironment(a);
        self.base_preset = try presetCall(self.ctx_preset.loadFromArgs(argv));
        self.monitor = try Monitor.create(self);
        // propagate base params to child
        unsetReservedArgs(gpa, &self.base_preset, true);

        // do not propagate these options, but allow preset to explicitly set them
        self.base_preset.unsetOption(gpa, "LLAMA_ARG_LOG_FILE");
        // set binary path
        self.bin_path = getServerExecPath(a) catch |e| blk: {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            const w = try zs(a, common.what(e));
            srv(.wrn, "server_models", "failed to get server executable path: %s\n", .{w});
            srv(.wrn, "server_models", "using original argv[0] as fallback: %s\n", .{argv[0].ptr});
            break :blk argv[0];
        };

        try self.unfinished_downloads.load();
        try self.loadModels();
        self.debug_fake_timing = std.c.getenv("LLAMA_SERVER_DEBUG_FAKE_TIMING") != null and std.c.getenv("LLAMA_SERVER_DEBUG_FAKE_TIMING").?[0] != 0;
    }

    /// The destructor: the monitor is joined first, as the C++'s member
    /// order makes it.
    pub fn deinit(self: *Models) void {
        if (self.monitor) |m| m.destroy();
        self.monitor = null;
        for (self.mapping.items) |inst| inst.destroy();
        self.mapping.deinit(gpa);
        self.unfinished_downloads.deinit();
        self.base_preset.deinit(gpa);
        self.sse.deinit();
        self.ctx_preset.destroy();
        self.stopping_arena.deinit();
        self.arena.deinit();
    }

    fn lowerBound(self: *const Models, name: []const u8) usize {
        var lo: usize = 0;
        var hi = self.mapping.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (std.mem.order(u8, self.mapping.items[mid].meta.name, name) == .lt) lo = mid + 1 else hi = mid;
        }
        return lo;
    }

    /// `mapping.find(name)`.
    pub fn find(self: *const Models, name: []const u8) ?*Instance {
        const i = self.lowerBound(name);
        if (i < self.mapping.items.len and std.mem.eql(u8, self.mapping.items[i].meta.name, name)) return self.mapping.items[i];
        return null;
    }

    /// `mapping[name] = inst`: replaces (and frees) an existing instance.
    fn putInstance(self: *Models, name: []const u8, inst: *Instance) !void {
        const i = self.lowerBound(name);
        if (i < self.mapping.items.len and std.mem.eql(u8, self.mapping.items[i].meta.name, name)) {
            self.mapping.items[i].destroy();
            self.mapping.items[i] = inst;
            return;
        }
        try self.mapping.insert(gpa, i, inst);
    }

    /// `mapping[name]`: the instance, created empty (with its name) when
    /// missing, as `operator[]` default-constructs one.
    pub fn getOrCreate(self: *Models, name: []const u8) !*Instance {
        if (self.find(name)) |inst| return inst;
        const meta: Meta = .{ .name = name };
        const inst = try Instance.create(&meta, try SharedProc.create());
        // `instance_t{}` holds no process; the empty one stands for it
        try self.putInstance(name, inst);
        return inst;
    }

    /// `mapping.erase(it)`.
    fn eraseAt(self: *Models, i: usize) void {
        self.mapping.items[i].destroy();
        _ = self.mapping.orderedRemove(i);
    }

    /// Ports `add_model` (server-models.cpp:429 @8642541b2): not thread-safe,
    /// the caller holds the mutex.
    fn addModel(self: *Models, meta: *Meta) Error!void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        if (self.find(meta.name) != null) {
            return throw(error.RuntimeError, "model '{s}' appears multiple times", .{meta.name});
        }

        // check model name does not conflict with existing aliases
        for (self.mapping.items) |inst| {
            if (inst.meta.aliases.contains(meta.name)) {
                return throw(error.RuntimeError, "model name '{s}' conflicts with alias of model '{s}'", .{ meta.name, inst.meta.name });
            }
        }

        // parse aliases from preset's --alias option (comma-separated)
        if (meta.preset.getOption("LLAMA_ARG_ALIAS")) |alias_str| {
            if (alias_str.len != 0) {
                for ((try tc.strings.split(sa, alias_str, ',')).items) |alias_raw| {
                    const alias = tc.strings.strip(alias_raw);
                    if (alias.len != 0) {
                        try meta.aliases.insert(sa, alias);
                    }
                }
            }
        }

        // parse tags from preset's --tags option (comma-separated)
        if (meta.preset.getOption("LLAMA_ARG_TAGS")) |tags_str| {
            if (tags_str.len != 0) {
                for ((try tc.strings.split(sa, tags_str, ',')).items) |tag_raw| {
                    const tag = tc.strings.strip(tag_raw);
                    if (tag.len != 0) {
                        try meta.tags.insert(sa, tag);
                    }
                }
            }
        }

        // validate aliases do not conflict with existing names or aliases
        for (meta.aliases.items.items) |alias| {
            if (self.find(alias) != null) {
                return throw(error.RuntimeError, "alias '{s}' for model '{s}' conflicts with existing model name", .{ alias, meta.name });
            }
            for (self.mapping.items) |inst| {
                if (inst.meta.aliases.contains(alias)) {
                    return throw(error.RuntimeError, "alias '{s}' for model '{s}' conflicts with alias of model '{s}'", .{ alias, meta.name, inst.meta.name });
                }
            }
        }

        try meta.updateArgs(sa, self.ctx_preset, self.bin_path); // render args
        try meta.updateCaps(sa, &self.base_params);
        const inst = try Instance.create(meta, try SharedProc.create());
        try self.putInstance(inst.meta.name, inst);
    }

    /// Ports `notify_sse` (server-models.cpp:714 @8642541b2): an event to
    /// every `/models/sse` client.
    pub fn notifySse(self: *Models, event: []const u8, model_id: []const u8, data: ?*const Value) void {
        self.notifySseImpl(event, model_id, data) catch |err| switch (err) {
            error.OutOfMemory, error.InvalidArgument, error.RuntimeError, error.JsonError => {},
        };
    }

    fn notifySseImpl(self: *Models, event: []const u8, model_id: []const u8, data: ?*const Value) !void {
        const result = try Result.create(gpa, .router);
        errdefer result.destroy(gpa);
        const a = result.alloc();
        var o = common.Obj.init(a);
        try o.put("model", model_id);
        try o.put("event", event);
        result.kind.router.data = o.done();
        if (data) |d| if (!d.isNull()) {
            try result.kind.router.data.set(a, "data", try d.clone(a));
        };
        const dump = try common.safeJsonToStr(a, &result.kind.router.data);
        srv(.dbg, "notify_sse", "notifying SSE clients about event '%s' for model '%s': %s\n", .{ try zs(a, event), try zs(a, model_id), try zs(a, dump) });
        try self.sse.broadcast(result);
    }

    /// Ports `load_models` (server-models.cpp:727 @8642541b2): (re-)load the
    /// list of models from various sources and prepare the metadata
    /// mapping -- on the first call, simply populate it; on a later one, a
    /// running model whose source changed or went is unloaded, and the
    /// others are added or updated.
    pub fn loadModels(self: *Models) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const ctx = self.ctx_preset;

        // Phase 1: load presets from all sources - pure I/O, no lock needed
        // 1. cached models
        var cached_models = try presetCall(ctx.loadFromCache());
        defer cached_models.deinit(gpa);
        srv(.trc, "load_models", "Loaded %zu cached model presets from %s\n", .{ cached_models.count(), try zs(sa, try hf_cache.getCachePath()) });
        // 2. local models from --models-dir
        var local_models: Presets = .{};
        defer local_models.deinit(gpa);
        if (self.base_params.models_dir.len != 0) {
            local_models = try presetCall(ctx.loadFromModelsDir(self.base_params.models_dir));
            srv(.trc, "load_models", "Loaded %zu local model presets from %s\n", .{ local_models.count(), try zs(sa, self.base_params.models_dir) });
        }
        // 3. custom-path models from presets
        var global: Preset = .{};
        defer global.deinit(gpa);
        var custom_presets: Presets = .{};
        defer custom_presets.deinit(gpa);
        if (self.base_params.models_preset.len != 0) {
            custom_presets = try presetCall(ctx.loadFromIni(self.base_params.models_preset, &global));
            srv(.trc, "load_models", "Loaded %zu custom model presets from %s\n", .{ custom_presets.count(), try zs(sa, self.base_params.models_preset) });
        }

        // cascade, apply global preset first
        {
            const t1 = try presetCall(ctx.cascade(&global, &cached_models));
            cached_models.deinit(gpa);
            cached_models = t1;
            const t2 = try presetCall(ctx.cascade(&global, &local_models));
            local_models.deinit(gpa);
            local_models = t2;
            const t3 = try presetCall(ctx.cascade(&global, &custom_presets));
            custom_presets.deinit(gpa);
            custom_presets = t3;
        }

        // note: if a model exists in both cached and local, local takes precedence
        var final_presets: Presets = .{};
        defer final_presets.deinit(gpa);
        var source_map: std.StringHashMapUnmanaged(Source) = .empty;
        for (cached_models.entries.items) |*p| {
            try final_presets.putOwned(gpa, try p.clone(gpa));
            try source_map.put(sa, p.name, .cache);
        }
        for (local_models.entries.items) |*p| {
            try final_presets.putOwned(gpa, try p.clone(gpa));
            try source_map.put(sa, p.name, .models_dir);
        }
        for (custom_presets.entries.items) |*custom| {
            if (final_presets.get(custom.name)) |target| {
                try presetCall(target.merge(gpa, custom));
            } else {
                try final_presets.putOwned(gpa, try custom.clone(gpa));
            }
            try source_map.put(sa, custom.name, .preset);
        }

        // overlay router's own CLI args on top of every model preset so that
        // e.g. `llama-server --temp 0` is honoured by all child processes
        for (final_presets.entries.items) |*p| {
            try presetCall(p.merge(gpa, &self.base_preset));
        }

        const Src = struct {
            map: *std.StringHashMapUnmanaged(Source),
            fn get(s: @This(), name: []const u8) Source {
                return s.map.get(name) orelse .preset;
            }
        };
        const get_source: Src = .{ .map = &source_map };

        // hide cache models whose resolved file is already used by a preset with dedup-cache-models enabled
        var hidden_models: StringSet = .{};
        {
            var preset_paths: StringSet = .{};
            // the `add_hf_path` lambda (server-models.cpp:785 @8642541b2)
            const AddHfPath = struct {
                fn f(ar: Allocator, paths: *StringSet, preset: *const Preset, repo_key: []const u8, file_key: []const u8) !void {
                    const hf_repo = preset.getOption(repo_key) orelse return;
                    if (hf_repo.len == 0) return;
                    const hf_file = preset.getOption(file_key) orelse "";
                    const path = try download.resolvePath(ar, hf_repo, hf_file);
                    if (path.len != 0) {
                        try paths.insert(ar, path);
                    }
                }
            };
            for (custom_presets.entries.items) |*p| {
                const val = p.getOption(preset_mod.env_dedup_cache_models) orelse continue;
                if (!tc.arg.isTruthy(val)) continue;
                try AddHfPath.f(sa, &preset_paths, p, "LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_FILE");
                try AddHfPath.f(sa, &preset_paths, p, "LLAMA_ARG_SPEC_DRAFT_HF_REPO", "LLAMA_ARG_SPEC_DRAFT_MODEL");
            }
            if (preset_paths.items.items.len != 0) {
                for (cached_models.entries.items) |*p| {
                    if (get_source.get(p.name) != .cache) {
                        continue; // merged with another source, not a pure cache entry
                    }
                    const path = try download.resolvePath(sa, p.name, "");
                    if (path.len != 0 and preset_paths.contains(path)) {
                        srv(.inf, "load_models", "hiding cache model name=%s (deduplicated by a preset)\n", .{try zs(sa, p.name)});
                        try hidden_models.insert(sa, p.name);
                    }
                }
            }
        }

        // Phase 2: acquire the lock once for all mapping mutations.
        // We temporarily release it only when calling functions that acquire it internally (unload)
        self.mutex.lock();
        var locked = true;
        defer if (locked) self.mutex.unlock();

        self.need_reload = false;
        const is_first_load = self.mapping.items.len == 0;

        if (is_first_load) {
            // FIRST LOAD: add all models, then unlock for autoloading
            for (final_presets.entries.items) |*p| {
                var meta: Meta = .{
                    .source = get_source.get(p.name),
                    .preset = try p.clone(sa),
                    .name = p.name,
                    .stop_timeout = default_stop_timeout,
                };
                try self.addModel(&meta);
            }
            self.applyStopTimeout();
            self.applyHidden(&hidden_models);
            try self.logAvailableModels();
            try self.sched.measureUnmeasured();

            // skipped on reload, see startup_models
            if (self.startup_models != null) {
                var models_to_load: std.ArrayList([]const u8) = .empty;
                for (self.mapping.items) |inst| {
                    const val = inst.meta.preset.getOption(preset_mod.env_load_on_startup) orelse continue;
                    if (tc.arg.isTruthy(val)) {
                        try models_to_load.append(gpa, try gpa.dupe(u8, inst.meta.name));
                    }
                }
                if (@as(c_int, @intCast(models_to_load.items.len)) > self.base_params.models_max) {
                    return throw(error.RuntimeError, "number of models to load on startup ({d}) exceeds models_max ({d})", .{ models_to_load.items.len, self.base_params.models_max });
                }

                // to be lazy-loaded after main() setup phase is completed
                self.startup_models = models_to_load;
            }

            self.mutex.unlock();
            locked = false;
        } else {
            // RELOAD: diff the new preset list against the current mapping and reconcile
            self.is_reloading = true;

            // find running models whose source was removed or whose preset changed
            var to_unload: std.ArrayList([]const u8) = .empty;
            for (self.mapping.items) |inst| {
                if (!inst.meta.isRunning()) continue;
                if (final_presets.get(inst.meta.name)) |p| {
                    if (!try presetOptionsEqualForCompare(sa, &inst.meta.preset, p)) {
                        try to_unload.append(sa, try sa.dupe(u8, inst.meta.name)); // preset changed
                    }
                } else {
                    try to_unload.append(sa, try sa.dupe(u8, inst.meta.name)); // removed from source
                }
            }

            // unload() acquires the lock internally, so release before each call
            for (to_unload.items) |name| {
                srv(.inf, "load_models", "(reload) unloading model name=%s (source updated or removed)\n", .{try zs(sa, name)});
                self.mutex.unlock();
                self.unload(name) catch |err| switch (err) {
                    error.OutOfMemory => {},
                };
                self.mutex.lock();
            }

            // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
            while (true) {
                var done = true;
                for (to_unload.items) |name| {
                    if (self.find(name)) |inst| if (inst.meta.isRunning()) {
                        done = false;
                    };
                }
                if (done) break;
                self.cv.wait(&self.mutex);
            }

            // erase models no longer in any source
            var i: usize = 0;
            while (i < self.mapping.items.len) {
                const inst = self.mapping.items[i];
                if (inst.meta.status == .downloading) {
                    i += 1; // download thread is still busy, skip
                } else if (inst.meta.status == .downloaded) {
                    // download finished, safe to erase
                    self.eraseAt(i);
                } else if (final_presets.get(inst.meta.name) == null) {
                    srv(.inf, "load_models", "(reload) removing model name=%s (no longer in source)\n", .{try zs(sa, inst.meta.name)});
                    self.eraseAt(i);
                } else {
                    i += 1;
                }
            }

            // update presets for non-running models still in source
            for (self.mapping.items) |inst| {
                if (inst.meta.isRunning()) continue;
                const p = final_presets.get(inst.meta.name) orelse continue; // erased above
                const ia = inst.alloc();

                if (!try presetOptionsEqualForCompare(sa, &inst.meta.preset, p)) {
                    // different args, different memory: measure again
                    inst.meta.memory = .{};
                }
                inst.meta.preset = try p.clone(ia);

                // re-parse aliases, then validate against other models
                var new_aliases: StringSet = .{};
                if (inst.meta.preset.getOption("LLAMA_ARG_ALIAS")) |alias_str| {
                    if (alias_str.len != 0) {
                        for ((try tc.strings.split(sa, alias_str, ',')).items) |alias_raw| {
                            const alias = tc.strings.strip(alias_raw);
                            if (alias.len != 0) try new_aliases.insert(sa, alias);
                        }
                    }
                }
                inst.meta.aliases.clear();
                for (new_aliases.items.items) |alias| {
                    var conflict = false;
                    for (self.mapping.items) |other| {
                        if (std.mem.eql(u8, other.meta.name, inst.meta.name)) continue;
                        if (std.mem.eql(u8, other.meta.name, alias) or other.meta.aliases.contains(alias)) {
                            srv(.wrn, "load_models", "(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n", .{ try zs(sa, alias), try zs(sa, inst.meta.name), try zs(sa, other.meta.name) });
                            conflict = true;
                            break;
                        }
                    }
                    if (!conflict) try inst.meta.aliases.insert(ia, alias);
                }

                // re-parse tags
                inst.meta.tags.clear();
                if (inst.meta.preset.getOption("LLAMA_ARG_TAGS")) |tags_str| {
                    if (tags_str.len != 0) {
                        for ((try tc.strings.split(sa, tags_str, ',')).items) |tag_raw| {
                            const tag = tc.strings.strip(tag_raw);
                            if (tag.len != 0) try inst.meta.tags.insert(ia, tag);
                        }
                    }
                }

                inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
                try inst.meta.updateArgs(ia, self.ctx_preset, self.bin_path);
                try inst.meta.updateCaps(ia, &self.base_params);
            }

            // add models that are new in this reload, load-on-startup is not honored here since a
            // reload never spawns an instance
            for (final_presets.entries.items) |*p| {
                if (self.find(p.name) == null) {
                    var meta: Meta = .{
                        .source = get_source.get(p.name),
                        .preset = try p.clone(sa),
                        .name = p.name,
                        .stop_timeout = default_stop_timeout,
                    };
                    try self.addModel(&meta);
                }
            }

            self.applyStopTimeout();
            self.applyHidden(&hidden_models);
            try self.sched.measureUnmeasured();

            // clear reload flag under the lock, this releases the load() calls waiting on !is_reloading
            self.is_reloading = false;
            self.cv.notifyAll();

            try self.logAvailableModels();

            self.mutex.unlock();
            locked = false;

            self.notifySse("models_reload", "*", null);
        }
    }

    /// The `log_available_models` (server-models.cpp:828 @8642541b2) lambda
    /// of `load_models`: each model with where it came from.
    fn logAvailableModels(self: *Models) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        srv(.inf, "operator()", "Available models (%zu):\n", .{self.mapping.items.len});
        if (self.mapping.items.len == 0) {
            srv(.inf, "operator()", "%s", .{@as([*:0]const u8, "  no models found on the system (visit https://llama.app/models for suggestions)\n")});
        } else {
            for (self.mapping.items) |inst| {
                const source = sourceToString(inst.meta.source);

                var info: std.ArrayList(u8) = .empty;
                if (inst.meta.aliases.items.items.len != 0) {
                    try info.print(sa, " (aliases: {s})", .{try inst.meta.aliases.joinComma(sa)});
                }
                if (inst.meta.tags.items.items.len != 0) {
                    try info.print(sa, " [tags: {s}]", .{try inst.meta.tags.joinComma(sa)});
                }

                srv(.inf, "operator()", "  [%10s] %s%s\n", .{ try zs(sa, source), try zs(sa, inst.meta.name), try zs(sa, info.items) });
            }
        }
    }

    /// The `apply_stop_timeout` (server-models.cpp:844 @8642541b2) lambda of
    /// `load_models`.
    fn applyStopTimeout(self: *Models) void {
        for (self.mapping.items) |inst| {
            if (inst.meta.preset.getOption(preset_mod.env_stop_timeout)) |val| {
                const z = gpa.dupeSentinel(u8, val, 0) catch continue;
                defer gpa.free(z);
                inst.meta.stop_timeout = tc.arg.stoi(z) catch blk: {
                    srv(.wrn, "operator()", "invalid stop-timeout value '%s' for model '%s', using default %d seconds\n", .{ z.ptr, zs(inst.alloc(), inst.meta.name) catch "", @as(c_int, default_stop_timeout) });
                    break :blk default_stop_timeout;
                };
            }
        }
    }

    /// The `apply_hidden` (server-models.cpp:858 @8642541b2) lambda of
    /// `load_models`.
    fn applyHidden(self: *Models, hidden_models: *const StringSet) void {
        for (self.mapping.items) |inst| {
            inst.meta.hidden = hidden_models.contains(inst.meta.name);
        }
    }

    /// Ports `load_startup_models` (server-models.cpp:1064 @8642541b2):
    /// lazy-load the models marked `load-on-startup`, after `main()`'s setup.
    pub fn loadStartupModels(self: *Models) !void {
        var to_load: std.ArrayList([]const u8) = .empty;
        defer {
            for (to_load.items) |s| gpa.free(s);
            to_load.deinit(gpa);
        }
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.startup_models == null) {
                return; // already drained
            }
            to_load = self.startup_models.?;
            self.startup_models = null;
        }
        for (to_load.items) |name| {
            const z = try zs(gpa, name);
            defer gpa.free(std.mem.span(z));
            srv(.inf, "load_startup_models", "(startup) loading model %s\n", .{z});
            try self.load(name, .{});
        }
    }

    /// Ports `has_model` (server-models.cpp:1089 @8642541b2): by name or
    /// alias (thread-safe).
    pub fn hasModel(self: *Models, name: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.find(name) != null) {
            return true;
        }
        for (self.mapping.items) |inst| {
            if (inst.meta.aliases.contains(name)) {
                return true;
            }
        }
        return false;
    }

    /// Ports `get_meta` (server-models.cpp:1102 @8642541b2): a copy of a
    /// model's metadata, by name or alias, reloading the list first when a
    /// download finished (thread-safe).
    pub fn getMeta(self: *Models, a: Allocator, name: []const u8) !?Meta {
        self.mutex.lock();
        if (self.need_reload) {
            self.mutex.unlock();
            try self.loadModels();
            self.mutex.lock();
        }
        defer self.mutex.unlock();

        if (self.find(name)) |inst| {
            return try inst.meta.clone(a);
        }
        for (self.mapping.items) |inst| {
            if (inst.meta.aliases.contains(name)) {
                return try inst.meta.clone(a);
            }
        }
        return null;
    }

    /// Ports `get_all_meta` (server-models.cpp:1122 @8642541b2): a copy of
    /// every model's metadata (thread-safe).
    pub fn getAllMeta(self: *Models, a: Allocator) ![]Meta {
        self.mutex.lock();
        if (self.need_reload) {
            self.mutex.unlock();
            try self.loadModels();
            self.mutex.lock();
        }
        defer self.mutex.unlock();

        const result = try a.alloc(Meta, self.mapping.items.len);
        for (self.mapping.items, 0..) |inst, i| {
            result[i] = try inst.meta.clone(a);
        }
        return result;
    }

    /// Ports `unfinished_downloads_get` (server-models.cpp:683 @8642541b2):
    /// downloads that never finished, with their sidecar kinds (thread-safe).
    pub fn unfinishedDownloadsGet(self: *Models, a: Allocator) Allocator.Error![]UnfinishedDownload {
        return self.unfinished_downloads.getAll(a);
    }

    /// Ports `unfinished_downloads_remove` (server-models.cpp:687
    /// @8642541b2): drop a download that will never be resumed, its record
    /// and its partial files; throws for a name that is not recorded
    /// (thread-safe).
    pub fn unfinishedDownloadsRemove(self: *Models, name: []const u8) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        // under the lock, so a download of the same tag cannot start while its files go
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.find(name) != null) {
            return throw(error.InvalidArgument, "model '{s}' is downloading", .{name});
        }
        // a name with no record is no model: its files belong to the cached tags of the repo
        if (!self.unfinished_downloads.remove(name)) {
            return throw(error.RuntimeError, "model name={s} is not found", .{name});
        }
        const ok = try download.remove(sa, name);
        srv(.inf, "unfinished_downloads_remove", "removing unfinished download name=%s (%s)\n", .{ try zs(sa, name), @as([*:0]const u8, if (ok) "succeeded" else "nothing cached") });
    }

    /// Ports `remove_sidecar` (server-models.cpp:701 @8642541b2): the sidecar
    /// `kind` of a model that is not in use, its files and partials
    /// (thread-safe).
    pub fn removeSidecar(self: *Models, name: []const u8, kind: []const u8) !bool {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.find(name)) |it| {
            if (it.meta.isRunning() or it.meta.status == .downloading) {
                return throw(error.InvalidArgument, "model '{s}' is in use, unload it first", .{name});
            }
        }
        // a sidecar that only ever landed as a partial is still something removed
        var ok = try self.unfinished_downloads.removeSidecar(name, kind);
        const opts = sidecarDownloadOpts(&self.base_params, &.{kind});
        ok = (try download.removeSidecars(sa, name, &opts)) or ok;
        srv(.inf, "remove_sidecar", "removing sidecar %s of model name=%s (%s)\n", .{ try zs(sa, kind), try zs(sa, name), @as([*:0]const u8, if (ok) "succeeded" else "nothing cached") });
        return ok;
    }

    /// Ports `load` (server-models.cpp:1142 @8642541b2), and the overload
    /// without options: spawn a child for the model (thread-safe); a model
    /// not unloaded is left alone.
    pub fn load(self: *Models, name: []const u8, opts: LoadOptions) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        // the fake load time of a model; a download worker loads none
        if (self.debug_fake_timing and opts.role == .instance) {
            // do not hold the mutex here, other requests must keep making progress
            sleepMs(2000);
        }

        if (opts.custom_meta == null) {
            if (!self.hasModel(name)) {
                return throw(error.RuntimeError, "model name={s} is not found", .{name});
            }
            if (opts.role == .instance) {
                try self.sched.makeRoom(name);
            }
        }

        self.mutex.lock();
        defer self.mutex.unlock();
        // edge case: block until any in-progress reload has finished so we always load
        // against the freshest preset and a consistent mapping state
        while (self.is_reloading) self.cv.wait(&self.mutex);

        // a download over an existing entry is decided under the lock, so a load of the
        // same model cannot start in between: a paused download or an idle cached model
        // is downloaded again with its files kept, a download on its way and a running
        // model are refused, and a preset or local model is not a download
        if (opts.role == .download) {
            if (self.find(name)) |it| {
                const cur = &it.meta;
                if (cur.status == .downloading) {
                    return throw(error.InvalidArgument, "model '{s}' is already downloading", .{name});
                }
                if (cur.isRunning()) {
                    return throw(error.InvalidArgument, "model '{s}' is running, unload it first", .{name});
                }
                if (cur.source != .cache) {
                    return throw(error.InvalidArgument, "model '{s}' already exists", .{name});
                }
                srv(.inf, "load", "resuming download for model '%s'\n", .{try zs(sa, name)});
            }

            // two downloads writing the same partial would interleave their bytes
            for (try self.unfinished_downloads.getAll(sa)) |*d| {
                const other = self.find(d.model);
                if (std.mem.eql(u8, d.model, name) or other == null or other.?.meta.status != .downloading) {
                    continue;
                }
                for (d.partials) |p| {
                    for (opts.partials) |q| {
                        if (std.mem.eql(u8, q.path, p.path)) {
                            return throw(error.InvalidArgument, "model '{s}' shares files with the download of '{s}' on its way, retry once it is done", .{ name, d.model });
                        }
                    }
                }
            }
        }

        var meta: Meta = if (opts.custom_meta) |m| try m.clone(sa) else try (try self.getOrCreate(name)).meta.clone(sa);
        if (meta.status != .unloaded) {
            srv(.inf, "load", "model %s is not ready\n", .{try zs(sa, name)});
            return;
        }

        // Re-check capacity under the lock to prevent concurrent loads from
        // exceeding models_max. Without this, the window between make_room()
        // releasing its lock and this lock_guard acquiring allows multiple
        // threads to each observe capacity and all proceed to load.
        // Download workers do not use models_max slots.
        if (opts.role == .instance and self.sched.limitsExceeded(name, false)) {
            return throw(error.RuntimeError, "model limit reached, try again later", .{});
        }

        // prepare new instance info
        meta.port = download.commonHttpGetFreePort();
        meta.status = .loading;
        meta.loaded_info = .null;
        meta.last_used = timeMs();

        if (meta.port <= 0) {
            return throw(error.RuntimeError, "failed to get a port number", .{});
        }

        const proc = try SharedProc.create();
        var proc_owned = true;
        defer if (proc_owned) proc.release();
        {
            srv(.inf, "load", "spawning server instance with name=%s on port %d\n", .{ try zs(sa, meta.name), meta.port });

            try meta.updateArgs(sa, self.ctx_preset, self.bin_path); // render args

            const child_args = meta.args; // copy
            var child_env: std.ArrayList([]const u8) = .empty;
            try child_env.appendSlice(sa, self.base_env); // copy
            try child_env.append(sa, try std.fmt.allocPrint(sa, "{s}={d}", .{ env_router_port, self.base_params.port }));

            if (opts.role == .download) {
                meta.status = .downloading;
                try child_env.append(sa, try std.mem.concat(sa, u8, &.{ env_child_role, "=", roleTypeToStr(.download) }));
                try child_env.append(sa, try std.mem.concat(sa, u8, &.{ "LLAMA_ARG_HF_REPO=", name }));

                // the sidecars as -hf would pull them: the draft types, and the mmproj switch
                // for the one that is not a draft
                var mmproj = false;
                var spec_types: std.ArrayList(u8) = .empty;
                for (download.sidecars) |sc| {
                    if (!hasSidecar(opts.sidecars, sc.name)) {
                        continue;
                    }
                    if (sc.draft) {
                        if (spec_types.items.len != 0) try spec_types.append(sa, ',');
                        try spec_types.appendSlice(sa, "draft-");
                        try spec_types.appendSlice(sa, sc.name);
                    } else {
                        mmproj = true;
                    }
                }
                try setChildEnv(sa, &child_env, "LLAMA_ARG_MMPROJ_AUTO", if (mmproj) "1" else "0");
                try setChildEnv(sa, &child_env, "LLAMA_ARG_SPEC_TYPE", if (spec_types.items.len == 0) "none" else spec_types.items);
            }

            srv(.inf, "load", "%s", .{@as([*:0]const u8, "spawning server instance with args:\n")});
            for (child_args) |arg| {
                srv(.inf, "load", "  %s\n", .{try zs(sa, arg)});
            }
            meta.args = child_args; // save for debugging

            // the child writes its commands to stdout and its logs to stderr
            const options = tc.subproc.option_no_window;
            if (!proc.proc.sproc.create(child_args, options, child_env.items, null)) {
                return throw(error.RuntimeError, "failed to spawn server instance", .{});
            }
        }

        // remember the download: the partial files carry a content hash, not the name
        if (opts.role == .download) {
            try self.unfinished_downloads.add(&.{ .model = name, .sidecars = opts.sidecars, .partials = opts.partials });
        }

        // old process should have exited already, but just in case, we clean it up here
        if (self.find(name)) |old| {
            if (old.subproc.proc.isAlive()) {
                srv(.wrn, "load", "old process for model name=%s is still alive, this is unexpected\n", .{try zs(sa, name)});
                old.subproc.proc.terminate(); // force kill
            }
        }

        {
            var o = common.Obj.init(sa);
            try o.put("status", statusToString(meta.status));
            const v = o.done();
            self.notifySse("model_status", name, &v);
        }

        const port = meta.port;
        const inst = try Instance.create(&meta, proc);
        proc_owned = false;
        try self.putInstance(name, inst);
        if (opts.role == .instance) {
            try self.sched.onSpawned(name);
        }
        try self.monitor.?.watch(name, proc, opts.role, port, 0);
        self.cv.notifyAll();
    }

    /// Ports `request_stop` (server-models.cpp:1301 @8642541b2): ask the
    /// monitor to stop a running instance; `send_exit` is false for a child
    /// already force-killed. The caller holds the mutex.
    pub fn requestStop(self: *Models, name: []const u8, send_exit: bool) void {
        const it = self.find(name) orelse return;
        if (self.stopping_models.contains(name)) {
            return;
        }
        self.stopping_models.insert(self.stopping_arena.allocator(), name) catch return;
        self.monitor.?.stop(name, it.meta.stop_timeout, send_exit) catch |err| switch (err) {
            error.OutOfMemory => {},
        };
    }

    /// Ports `watch_child` (server-models.cpp:1310 @8642541b2): let the
    /// monitor watch a child; `kill_after` above 0 force-kills it after that
    /// many seconds.
    pub fn watchChild(self: *Models, name: []const u8, proc: *SharedProc, role: RoleType, port: c_int, kill_after: c_int) !void {
        try self.monitor.?.watch(name, proc, role, port, kill_after);
    }

    /// Ports `on_child_exit` (server-models.cpp:1314 @8642541b2): called by
    /// the monitor once a child exited and was reaped. The name leaves
    /// `stopping_models` only once its entry is unloaded -- here for a
    /// download or a stale entry, in `update_status` otherwise -- so a
    /// request never sees it ready in between.
    fn onChildExit(self: *Models, name: []const u8, proc: *SharedProc, role: RoleType, exit_code: c_int) void {
        if (role == .measure) {
            self.sched.onMeasureExit(name, proc, exit_code);
            return;
        }
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            const it = self.find(name);
            if (it == null or it.?.subproc != proc) {
                self.eraseStopping(name);
                return; // entry erased, or a newer instance took the name
            }
        }
        if (role == .download) {
            // instance will be cleaned up on next load_models() call
            self.mutex.lock();
            defer self.mutex.unlock();
            self.eraseStopping(name);
            self.cv.notifyAll();
        } else {
            self.updateStatus(name, .{ .status = .unloaded, .exit_code = exit_code });
        }
    }

    /// `stopping_models.count(name) > 0`; `mutex` is held.
    fn isStopping(self: *Models, name: []const u8) bool {
        return self.stopping_models.contains(name);
    }

    fn eraseStopping(self: *Models, name: []const u8) void {
        for (self.stopping_models.items.items, 0..) |s, i| {
            if (std.mem.eql(u8, s, name)) {
                _ = self.stopping_models.items.orderedRemove(i);
                return;
            }
        }
    }

    /// Ports `unload` (server-models.cpp:1340 @8642541b2): stop the model's
    /// child, or cancel its download (thread-safe).
    pub fn unload(self: *Models, name: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const z = try zs(gpa, name);
        defer gpa.free(std.mem.span(z));
        if (self.find(name)) |it| {
            if (it.meta.status == .downloading) {
                srv(.inf, "unload", "cancelling download for model name=%s\n", .{z});
                it.requestExit();
                // for convenience, we wait the status change here
                self.waitLocked(name, struct {
                    fn p(m: *const Meta) bool {
                        return m.status != .downloading;
                    }
                }.p);
            } else if (it.meta.isRunning()) {
                srv(.inf, "unload", "stopping model instance name=%s\n", .{z});
                const loading = it.meta.status == .loading;
                if (loading) {
                    // special case: if model is in loading state, unloading means force-killing it
                    srv(.wrn, "unload", "model name=%s is still loading, force-killing\n", .{z});
                    it.subproc.proc.terminate();
                }
                self.requestStop(name, !loading);
                // status change will be handled by the monitor
            } else {
                srv(.wrn, "unload", "model instance name=%s is not running\n", .{z});
            }
        }
    }

    /// Ports `unload_all` (server-models.cpp:1367 @8642541b2): stop every
    /// child and wait for each to exit; the monitor force-kills the ones
    /// that ignore the exit command.
    pub fn unloadAll(self: *Models) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.sched.cancelMeasurements();
        for (self.mapping.items) |inst| {
            const z = zs(gpa, inst.meta.name) catch continue;
            defer gpa.free(std.mem.span(z));
            if (inst.meta.status == .downloading) {
                srv(.inf, "unload_all", "cancelling download for model name=%s\n", .{z});
                inst.requestExit();
            } else if (inst.meta.isRunning()) {
                srv(.inf, "unload_all", "stopping model instance name=%s\n", .{z});
                const loading = inst.meta.status == .loading;
                if (loading) {
                    inst.subproc.proc.terminate();
                }
                self.requestStop(inst.meta.name, !loading);
            }
        }
        // wait for every child to exit, the monitor force-kills the ones that ignore the exit command
        while (true) {
            var done = !self.sched.isMeasuring();
            for (self.mapping.items) |inst| {
                if (inst.meta.isRunning() or inst.meta.status == .downloading) {
                    done = false;
                }
            }
            if (done) break;
            self.cv.wait(&self.mutex);
        }
    }

    /// Ports `update_status` (server-models.cpp:1397 @8642541b2): a model's
    /// status, and the event on `/models/sse` (thread-safe).
    pub fn updateStatus(self: *Models, name: []const u8, args: UpdateStatusArgs) void {
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.find(name)) |it| {
                const meta = &it.meta;
                meta.status = args.status;
                meta.exit_code = args.exit_code;
                if (args.status == .unloaded) {
                    self.eraseStopping(name);
                }
                if (args.loaded_info) |li| if (!li.isNull()) {
                    meta.loaded_info = li.clone(it.alloc()) catch meta.loaded_info;
                    // the child replaces both arrays in full; a bad or missing value changes nothing
                    mergeChildArchitecture(it.alloc(), &meta.architecture, li) catch |err| switch (err) {
                        error.OutOfMemory, error.JsonError => {},
                    };
                };
                if (args.progress) |pr| if (!pr.isNull()) {
                    meta.progress = pr.clone(it.alloc()) catch meta.progress;
                };
                switch (args.status) {
                    .loaded => self.sched.onReady(name, args.memory) catch |err| switch (err) {
                        else => {},
                    },
                    .sleeping => self.sched.onSleeping(name),
                    .unloaded => self.sched.onUnloaded(name),
                    else => {},
                }
                // a model that comes up idle or goes down changes the slot count for queued requests
                self.sched.tick();
            }
        }
        // broadcast status change to SSE
        {
            var scratch = std.heap.ArenaAllocator.init(gpa);
            defer scratch.deinit();
            const sa = scratch.allocator();
            var data = common.Obj.init(sa);
            data.put("status", statusToString(args.status)) catch return;
            if (args.status == .unloaded) {
                data.put("exit_code", args.exit_code) catch return;
            }
            if (args.loaded_info) |li| if (!li.isNull()) {
                data.put("info", li.clone(sa) catch return) catch return;
            };
            if (args.progress) |pr| if (!pr.isNull()) {
                data.put("progress", pr.clone(sa) catch return) catch return;
            };
            if (args.memory) |m| {
                data.put("memory", m.toJson(sa) catch return) catch return;
            }
            const v = data.done();
            // note: notify_sse doesn't acquire the lock, so no deadlock here
            self.notifySse("status_change", name, &v);
        }
        self.mutex.lock();
        self.cv.notifyAll();
        self.mutex.unlock();
    }

    /// Ports `update_download_progress` (server-models.cpp:1466 @8642541b2):
    /// a download child's progress, or its end, which marks the instance to
    /// be erased on the next reload.
    pub fn updateDownloadProgress(self: *Models, name: []const u8, progress: *const download.Progress, done: bool, ok: bool) void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        var curr: Value = .null;
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.find(name)) |it| {
                if (done) {
                    // mark the instance to be erased on next load_models() call
                    it.meta.status = .downloaded;
                    self.need_reload = true;

                    // a download that failed keeps its tag: its partial files are still
                    // there and re-posting the tag resumes them
                    if (ok) {
                        self.unfinished_downloads.forget(name);
                    }
                } else {
                    const ia = it.alloc();
                    const info = &it.meta.loaded_info;
                    if (info.isNull()) info.* = Value.emptyObject();
                    if (!info.contains("progress")) {
                        info.set(ia, "progress", Value.emptyObject()) catch |err| switch (err) {
                            error.OutOfMemory, error.JsonError => {},
                        };
                    }
                    var entry = common.Obj.init(ia);
                    entry.put("done", progress.downloaded) catch |err| switch (err) {
                        error.OutOfMemory, error.InvalidArgument, error.RuntimeError, error.JsonError => {},
                    };
                    entry.put("total", progress.total) catch |err| switch (err) {
                        error.OutOfMemory, error.InvalidArgument, error.RuntimeError, error.JsonError => {},
                    };
                    if (info.index(ia, "progress")) |pv| {
                        pv.set(ia, progress.url, entry.done()) catch |err| switch (err) {
                            error.OutOfMemory, error.JsonError => {},
                        };
                    } else |_| {}
                    curr = info.clone(sa) catch .null; // copy
                }
            }
        }
        if (done) {
            self.mutex.lock();
            self.cv.notifyAll(); // notify in case unload() is waiting for download to be cancelled
            self.mutex.unlock();
            // `{}` as a `const json &` is a null json: no "data"
            self.notifySse(if (ok) "download_finished" else "download_failed", name, null);
        } else {
            self.notifySse("download_progress", name, &curr);
        }
    }

    /// Ports `remove` (server-models.cpp:1503 @8642541b2): a cached model off
    /// the disk and out of the list; only cache models can be removed
    /// (thread-safe).
    pub fn remove(self: *Models, name: []const u8) !bool {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const z = try zs(sa, name);
        // do everything under one lock acquisition; avoid get_meta() /
        // unload() because they can trigger load_models() which erases
        // transient DOWNLOADING / DOWNLOADED entries as a side-effect
        self.mutex.lock();
        defer self.mutex.unlock();

        const it = self.find(name) orelse {
            return throw(error.RuntimeError, "model name={s} is not found", .{name});
        };
        if (it.meta.source != .cache) {
            return throw(error.RuntimeError, "model name={s} is not removable (not from cache)", .{name});
        }

        if (it.meta.status == .downloading) {
            // cancel in-flight download
            srv(.inf, "remove", "cancelling download for model name=%s\n", .{z});
            it.requestExit();
        } else if (it.meta.isRunning()) {
            // stop running instance
            srv(.inf, "remove", "stopping model instance name=%s\n", .{z});
            const loading = it.meta.status == .loading;
            if (loading) {
                it.subproc.proc.terminate();
            }
            self.requestStop(name, !loading);
        }

        // wait until the child is gone
        self.waitLocked(name, struct {
            fn p(m: *const Meta) bool {
                return m.status == .unloaded or m.status == .downloaded;
            }
        }.p);

        // load_models() may have erased the entry during the wait; the files go under the
        // lock, so a download of the same tag cannot start meanwhile
        const idx = self.lowerBound(name);
        if (idx < self.mapping.items.len and std.mem.eql(u8, self.mapping.items[idx].meta.name, name)) {
            self.eraseAt(idx);
        }
        _ = self.unfinished_downloads.remove(name);

        // best-effort: a cancelled download may have no cached files
        const ok = try download.remove(sa, name);
        srv(.inf, "remove", "removing model name=%s from cache (%s)\n", .{ z, @as([*:0]const u8, if (ok) "succeeded" else "nothing cached") });
        self.notifySse("model_remove", name, null); // `{}` as a `const json &`: a null json
        return true;
    }

    /// Ports `wait` (server-models.cpp:1554 @8642541b2), the overload
    /// taking a held lock: until `predicate` holds, or the model is gone.
    fn waitLocked(self: *Models, name: []const u8, comptime predicate: fn (*const Meta) bool) void {
        while (true) {
            if (self.find(name)) |it| {
                if (predicate(&it.meta)) return;
            } else {
                // model was removed from mapping by another code path (e.g. load_models()).
                // nothing left to wait for - tell the caller to proceed.
                return;
            }
            self.cv.wait(&self.mutex);
        }
    }

    /// Ports `ensure_model_ready` (server-models.cpp:1567 @8642541b2): load
    /// the model if it is not, through the `models_max` queue -- whose entry
    /// protects it from eviction until its waiters leave -- and block until
    /// it is up (thread-safe). A stopping instance takes no new request: its
    /// waiters queue for the next one.
    ///
    /// Return: false if the model was ready (or sleeping), true after
    /// waiting; throws if the load fails or `should_stop` fires.
    pub fn ensureModelReady(self: *Models, name: []const u8, should_stop: ?http.ShouldStop) !bool {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const z = try zs(sa, name);
        const meta = (try self.getMeta(sa, name)) orelse {
            return throw(error.RuntimeError, "model name={s} is not found", .{name});
        };
        const stopping = blk: {
            self.mutex.lock();
            defer self.mutex.unlock();
            break :blk self.isStopping(name);
        };
        if (!stopping and meta.isReady()) {
            return false; // ready for taking requests
        }
        if (!stopping and meta.status == .sleeping) {
            return false; // child is sleeping but still running; new request will wake it up
        }

        var queued = false;
        var did_load = false;
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.find(name)) |it| if (it.meta.status == .unloaded) {
                // the queue entry protects the model from eviction until its waiters leave
                try self.sched.join(name);
                self.sched.tick();
                queued = true;
            };
        }

        // while queued, this is also where the load happens: the head of the queue does it
        srv(.inf, "ensure_model_ready", "waiting until model name=%s is fully loaded...\n", .{z});
        self.mutex.lock();
        defer self.mutex.unlock();

        self.ensureWait(sa, name, should_stop, &queued, &did_load) catch |e| {
            if (queued) {
                self.sched.leave(name);
                queued = false;
            }
            self.sched.tick(); // a slot freed for this waiter goes to the next one
            return e;
        };
        if (queued) {
            self.sched.leave(name);
            queued = false;
        }

        return true;
    }

    /// The waiting loop of `ensure_model_ready`, from `saw_loading`
    /// (server-models.cpp:1608 @8642541b2) on, with the mutex held.
    fn ensureWait(self: *Models, sa: Allocator, name: []const u8, should_stop: ?http.ShouldStop, queued: *bool, did_load: *bool) !void {
        const z = try zs(sa, name);
        var saw_loading = false;
        while (true) {
            const it = self.find(name) orelse break; // removed by another code path, nothing to wait for
            if (self.isStopping(name)) {
                // a stopping instance takes no new request, the next instance serves it
                if (!queued.*) {
                    try self.sched.join(name);
                    self.sched.tick();
                    queued.* = true;
                }
                if (should_stop) |ss| if (ss.call()) {
                    return throw(error.RuntimeError, "request cancelled while waiting for model name={s}", .{name});
                };
                self.cv.waitForMs(&self.mutex, 200);
                continue;
            }
            const status = it.meta.status;

            if (status == .loaded or status == .sleeping) {
                break;
            }
            if (status == .downloading or status == .downloaded) {
                break; // do not wait on a download child
            }
            if (status == .loading) {
                saw_loading = true;
            } else if (status == .unloaded) {
                if (did_load.* or saw_loading) {
                    // a spawn happened and the instance came back down
                    if (it.meta.isFailed()) {
                        return throw(error.RuntimeError, "model name={s} failed to load", .{name});
                    }
                    break; // unloaded by another code path, caller reports "not running"
                }
                if (!queued.*) {
                    break; // not queued, and the load someone else started fell over
                }
            }

            if (should_stop) |ss| if (ss.call()) {
                // if a model was evicted for us, the free slot goes to the next waiter
                return throw(error.RuntimeError, "request cancelled while waiting for model name={s}", .{name});
            };

            // our turn: our model is at the head, and a slot really did free up
            if (status == .unloaded and self.sched.tryClaim(name)) {
                self.mutex.unlock();
                var ok = true;
                srv(.inf, "ensure_model_ready", "slot available, loading queued model name=%s\n", .{z});
                if (self.load(name, .{})) {
                    did_load.* = true;
                } else |e| {
                    if (e == error.OutOfMemory) {
                        self.mutex.lock();
                        return e;
                    }
                    // lost a race for the slot, stay in line and retry
                    srv(.wrn, "ensure_model_ready", "queued load of name=%s did not go through: %s\n", .{ z, try zs(sa, context.whatOf(e)) });
                    ok = false;
                }
                self.mutex.lock();
                self.sched.claimDone(name, ok);
                self.sched.tick();
                continue;
            }

            self.cv.waitForMs(&self.mutex, 200);
        }
    }

    /// Ports `proxy_request` (server-models.cpp:1686 @8642541b2): the request
    /// forwarded to the model's child; a detached request (one belonging to
    /// a replay session) ignores its client hanging up.
    pub fn proxyRequest(self: *Models, req: *const http.Req, method: []const u8, name: []const u8, update_last_used: bool, detached: bool) !*http.Res {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const meta = (try self.getMeta(sa, name)) orelse {
            return throw(error.RuntimeError, "model name={s} is not found", .{name});
        };
        if (!meta.isRunning()) {
            return throw(error.InvalidArgument, "model name={s} is not running", .{name});
        }
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            const inst = try self.getOrCreate(name);
            if (update_last_used) {
                inst.meta.last_used = timeMs();
            }
            inst.req_count += 1;
        }
        if (self.debug_fake_timing) {
            // sleep after req_count++, so the model counts as busy while we wait here
            sleepMs(2000);
        }
        srv(.inf, "proxy_request", "proxying request to model %s on port %d\n", .{ try zs(sa, name), meta.port });
        var proxy_path = try sa.dupe(u8, req.path);
        if (req.query_string.len != 0) {
            proxy_path = try std.mem.concat(sa, u8, &.{ proxy_path, "?", req.query_string });
        }
        const res = try proxy_mod.Proxy.create(
            method,
            "http",
            child_addr,
            meta.port,
            proxy_path,
            &req.headers,
            req.body,
            &req.files,
            // a detached request belongs to a replay session
            if (detached) .{ .req = null, .never = true } else req.should_stop,
            self.base_params.timeout_read,
            self.base_params.timeout_write,
        );

        const cleanup = try gpa.create(ProxyCleanup);
        cleanup.* = .{ .models = self, .name = try gpa.dupe(u8, name) };
        const p: *proxy_mod.Proxy = @fieldParentPtr("base", res);
        p.cleanup = .{ .ctx = cleanup, .func = ProxyCleanup.run };

        return res;
    }

    /// Ports `proxy_ws` (server-models.cpp:1741 @8642541b2): forward a
    /// websocket connection to the child, returning when either side closes.
    pub fn proxyWs(self: *Models, req: *const http.Req, conn: *http.WsConn, name: []const u8) !void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();

        const SendError = struct {
            fn f(a: Allocator, cn: *http.WsConn, message: []const u8) !void {
                var err = common.Obj.init(a);
                try err.put("type", "server_error");
                try err.put("code", @as(Value, .null));
                try err.put("message", message);
                try err.put("param", @as(Value, .null));
                var o = common.Obj.init(a);
                try o.put("type", "error");
                try o.put("error", err.done());
                var v = o.done();
                _ = cn.send(try common.safeJsonToStr(a, &v));
                cn.close();
            }
        };

        const meta = try self.getMeta(sa, name);
        if (meta == null or !meta.?.isRunning()) {
            try SendError.f(sa, conn, try std.fmt.allocPrint(sa, "model name={s} is not running", .{name}));
            return;
        }
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            const inst = try self.getOrCreate(name);
            inst.meta.last_used = timeMs();
            inst.req_count += 1;
        }
        srv(.inf, "proxy_ws", "proxying websocket to model %s on port %d\n", .{ try zs(sa, name), meta.?.port });

        var url = try std.fmt.allocPrint(sa, "ws://{s}:{d}{s}", .{ child_addr, meta.?.port, req.path });
        if (req.query_string.len != 0) {
            url = try std.mem.concat(sa, u8, &.{ url, "?", req.query_string });
        }
        // the client library adds its own upgrade headers
        var headers: client.HeadersMap = .{};
        for (req.headers.entries.items) |e| {
            const lower = try sa.dupe(u8, e.key);
            for (lower) |*ch| ch.* = std.ascii.toLower(ch.*);
            if (std.mem.eql(u8, lower, "host") or std.mem.eql(u8, lower, "connection") or std.mem.eql(u8, lower, "upgrade") or std.mem.startsWith(u8, lower, "sec-websocket-")) {
                continue;
            }
            _ = try headers.emplace(sa, e.key, e.value);
        }

        {
            var upstream = ws_mod.WebSocketClient.init(gpa, url, headers);
            defer upstream.deinit();
            if (!try upstream.connect()) {
                try SendError.f(sa, conn, try std.fmt.allocPrint(sa, "failed to connect to model name={s}", .{name}));
            } else {
                // client to child
                const Pump = struct {
                    fn run(cn: *http.WsConn, up: *ws_mod.WebSocketClient) void {
                        var msg: std.ArrayList(u8) = .empty;
                        defer msg.deinit(gpa);
                        while (true) {
                            const res = cn.read(gpa, &msg) catch 0;
                            if (res == 0 or !(if (res == 1) up.send(msg.items) else up.sendBinary(msg.items))) {
                                break;
                            }
                        }
                        up.close();
                    }
                };
                const pump = try std.Thread.spawn(.{}, Pump.run, .{ conn, &upstream });
                // child to client
                var msg: std.ArrayList(u8) = .empty;
                defer msg.deinit(gpa);
                while (true) {
                    const res = upstream.read(gpa, &msg) catch .fail;
                    if (res == .fail or !(if (res == .text) conn.send(msg.items) else conn.sendBinary(msg.items))) {
                        break;
                    }
                }
                conn.close();
                pump.join();
            }
        }

        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.find(name)) |it| {
            if (it.req_count > 0) {
                it.req_count -= 1;
                if (it.req_count == 0) {
                    self.sched.tick();
                }
            }
        }
    }

    /// Ports `handle_child_state` (server-models.cpp:1816 @8642541b2): a
    /// state line from a child -- `CMD_CHILD_TO_ROUTER_STATE` and a JSON
    /// object -- called from the monitor thread.
    ///
    /// payload per state:
    ///     state = loading     -> payload = {} (TODO: add progress info)
    ///     state = ready       -> payload = model_info (json) with a "memory" field, or {} if wakeup from sleeping
    ///     state = sleeping    -> payload = {}
    ///     state = measuring   -> payload = {result: "measured", min, max} or {result: "failed", error}
    fn handleChildState(self: *Models, name: []const u8, raw_input: []const u8) void {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        var state: context.ServerState = undefined;
        var payload: Value = .null;

        parse: {
            var data = json.parse(sa, raw_input[cmd_child_to_router_state.len..]) catch {
                srv(.err, "handle_child_state", "failed to parse child state update for name=%s: %s\n", .{ zs(sa, name) catch "", zs(sa, json.what()) catch "" });
                return;
            };
            const s = common.jsonValue(&data, "state", []const u8, "");
            state = serverStateFromStr(s);
            // `json{}` is a null json
            payload = common.jsonValueJson(sa, &data, "payload", .null) catch return;
            break :parse;
        }

        switch (state) {
            .downloading => {
                const result = common.jsonValue(&payload, "result", []const u8, "");
                const url = common.jsonValue(&payload, "url", []const u8, "");
                if (std.mem.eql(u8, result, "download_finished")) {
                    self.updateDownloadProgress(name, &.{}, true, true);
                    self.requestExitLocked(name);
                } else if (std.mem.eql(u8, result, "download_failed")) {
                    self.updateDownloadProgress(name, &.{}, true, false);
                    self.requestExitLocked(name);
                } else if (url.len != 0) {
                    var p: download.Progress = .{};
                    p.url = url;
                    p.downloaded = common.jsonValue(&payload, "downloaded", usize, 0);
                    p.total = common.jsonValue(&payload, "total", usize, 0);
                    self.updateDownloadProgress(name, &p, false, true);
                }
            },
            .loading => {
                self.updateStatus(name, .{
                    .status = .loading,
                    .exit_code = 0,
                    .loaded_info = null, // no loaded_info yet
                    .progress = &payload,
                });
            },
            .measuring => {
                self.sched.handleMeasured(name, &payload);
            },
            .ready => {
                var memory: ?models_sched.ServerModelMemory = null;
                var info = payload;
                if (info.contains("memory")) {
                    if (info.at("memory")) |jm| {
                        memory = models_sched.ServerModelMemory.fromJson(sa, jm) catch |e| blk: {
                            srv(.wrn, "handle_child_state", "malformed memory report from model name=%s: %s\n", .{ zs(sa, name) catch "", zs(sa, context.whatOf(e)) catch "" });
                            break :blk null;
                        };
                    } else |_| {}
                    info.erase(sa, "memory") catch |e| switch (e) {
                        error.OutOfMemory, error.JsonError => {},
                    };
                }
                self.updateStatus(name, .{
                    .status = .loaded,
                    .exit_code = 0,
                    // note: payload can be empty if this is a wakeup from sleep
                    .loaded_info = if (info.size() > 0) &info else null,
                    // "reset progress info": `{}` is a null json, which
                    // update_status skips, so the progress stays
                    .progress = null,
                    .memory = if (memory) |*m| m else null,
                });
            },
            .sleeping => {
                self.updateStatus(name, .{ .status = .sleeping });
            },
        }
    }

    /// The `request_exit` (server-models.cpp:1834 @8642541b2) lambda of
    /// `handle_child_state`.
    fn requestExitLocked(self: *Models, name: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.find(name)) |it| {
            it.requestExit();
        }
    }
};

/// The `cleanup` (server-models.cpp:1727 @8642541b2) lambda `proxy_request`
/// gives the proxy: the request is over.
const ProxyCleanup = struct {
    models: *Models,
    name: []const u8,

    fn run(ctx: ?*anyopaque) void {
        const self: *ProxyCleanup = @ptrCast(@alignCast(ctx.?));
        defer {
            gpa.free(self.name);
            gpa.destroy(self);
        }
        const models = self.models;
        models.mutex.lock();
        defer models.mutex.unlock();
        if (models.find(self.name)) |it| {
            if (it.req_count > 0) {
                it.req_count -= 1;
                if (it.req_count == 0) {
                    models.sched.tick();
                }
            }
        }
    }
};

/// Ports `server_state_from_str` (server-common.h:797 @8642541b2): an
/// unknown name aborts.
fn serverStateFromStr(str: []const u8) context.ServerState {
    if (std.mem.eql(u8, str, "downloading")) return .downloading;
    if (std.mem.eql(u8, str, "measuring")) return .measuring;
    if (std.mem.eql(u8, str, "loading")) return .loading;
    if (std.mem.eql(u8, str, "ready")) return .ready;
    if (std.mem.eql(u8, str, "sleeping")) return .sleeping;
    _ = c.ggml_abort("tools/server/server-common.h", 803, "GGML_ASSERT(%s) failed", "false && \"invalid server_state string\"");
    unreachable;
}

/// The `preset_options_for_compare` (server-models.cpp:864 @8642541b2)
/// lambda of `load_models`: `update_args()` injects HOST, PORT, ALIAS and
/// LOG_COLORS, so they are stripped before comparing presets.
fn presetOptionsEqualForCompare(a: Allocator, x: *const Preset, y: *const Preset) !bool {
    var px = try x.clone(a);
    var py = try y.clone(a);
    for ([_]*Preset{ &px, &py }) |p| {
        p.unsetOption(a, "LLAMA_ARG_HOST");
        p.unsetOption(a, "LLAMA_ARG_PORT");
        p.unsetOption(a, "LLAMA_ARG_ALIAS");
        p.unsetOption(a, "LLAMA_ARG_LOG_COLORS");
    }
    // `std::map<common_arg, std::string>::operator==`: same size, and each
    // pair equal in order (the keys by `common_arg::operator==`)
    if (px.options.items.len != py.options.items.len) return false;
    for (px.options.items, py.options.items) |ox, oy| {
        if (ox.opt.args.len == 0 or oy.opt.args.len == 0) return false;
        if (!std.mem.eql(u8, ox.opt.args[0], oy.opt.args[0])) return false;
        if (!std.mem.eql(u8, ox.value, oy.value)) return false;
    }
    return true;
}

fn sleepMs(ms: u64) void {
    var ts: std.c.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * std.time.ns_per_ms) };
    while (std.c.nanosleep(&ts, &ts) != 0) {}
}

// -----------------------------------------------------------------------------
// server_role

/// Ports `server_role` (server-common.h:807 @8642541b2): the role of this
/// process and, when it is a child of a router, its side of the protocol.
/// Construct it with `init` before anything is written to stdout, and
/// `deinit` it.
pub const Role = struct {
    // serializes the notify_to_router writes
    mtx_stdout: Mutex = .{},
    cmd_out: ?*anyopaque = null, // the stdout the router reads the commands from

    /// Ports the constructor of `server_role` (server-common.cpp:2223
    /// @8642541b2): in a child, keeps stdout for the commands to the
    /// router, so it is created before anything is written; everything
    /// else written to stdout goes to stderr with the logs.
    pub fn init() Role {
        var self: Role = .{};
        if (self.isManagedByRouter()) {
            self.cmd_out = subproc_mod.reserveStdout();
        }
        return self;
    }

    /// Ports the destructor of `server_role` (server-common.cpp:2229
    /// @8642541b2).
    pub fn deinit(self: *Role) void {
        if (self.cmd_out) |f| {
            _ = fclose(f);
            self.cmd_out = null;
        }
    }

    /// Ports `is_managed_by_router` (server-common.cpp:2235 @8642541b2):
    /// true if the current process was spawned by a router.
    pub fn isManagedByRouter(_: *const Role) bool {
        return std.c.getenv(env_router_port) != null;
    }

    /// Ports `get_role` (server-common.cpp:2239 @8642541b2).
    pub fn getRole(_: *const Role) RoleType {
        const role = std.c.getenv(env_child_role);
        return roleTypeFromStr(if (role) |r| std.mem.span(r) else "");
    }

    /// `std::function<void(int)>` for `setup`'s shutdown handler.
    pub const ShutdownHandler = struct {
        ctx: ?*anyopaque,
        func: *const fn (ctx: ?*anyopaque, signal: c_int) void,
    };

    /// Ports `setup` (server-common.cpp:2244 @8642541b2): a thread reading
    /// stdin until the router's exit command, which calls the shutdown
    /// handler, or EOF, which exits.
    ///
    /// Return: the thread, to be joined by the caller.
    pub fn setup(_: *Role, shutdown_handler: ShutdownHandler) ?std.Thread {
        // setup thread for monitoring stdin
        return std.Thread.spawn(.{}, struct {
            fn run(h: ShutdownHandler) void {
                // wait for EOF on stdin
                srv(.inf, "operator()", "%s", .{@as([*:0]const u8, "child server monitoring thread started, waiting for EOF on stdin...\n")});
                var eof = false;
                var line: std.ArrayList(u8) = .empty;
                defer line.deinit(gpa);
                outer: while (true) {
                    line.clearRetainingCapacity();
                    // std::getline(std::cin, line)
                    var got_any = false;
                    while (true) {
                        var ch: [1]u8 = undefined;
                        const n = std.c.read(0, &ch, 1);
                        if (n <= 0) {
                            if (!got_any) {
                                // EOF detected, that means the router server is unexpectedly exit or killed
                                eof = true;
                                break :outer;
                            }
                            break;
                        }
                        got_any = true;
                        if (ch[0] == '\n') break;
                        line.append(gpa, ch[0]) catch |err| switch (err) {
                            error.OutOfMemory => {},
                        };
                    }
                    if (std.mem.indexOf(u8, line.items, cmd_router_to_child_exit) != null) {
                        srv(.inf, "operator()", "%s", .{@as([*:0]const u8, "exit command received, exiting...\n")});
                        h.func(h.ctx, 0);
                        break;
                    }
                }
                if (eof) {
                    srv(.inf, "operator()", "%s", .{@as([*:0]const u8, "EOF on stdin detected, forcing shutdown...\n")});
                    std.c.exit(1);
                }
            }
        }.run, .{shutdown_handler}) catch null;
    }

    /// Ports `notify_to_router` (server-common.cpp:2270 @8642541b2): a state
    /// line on the reserved stdout, which the router reads.
    pub fn notifyToRouter(self: *Role, state: []const u8, payload: *const Value) void {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var o = common.Obj.init(a);
        o.put("state", state) catch return;
        o.put("payload", payload.clone(a) catch return) catch return;
        var data = o.done();
        const dump = common.safeJsonToStr(a, &data) catch return;
        const dz = a.dupeSentinel(u8, dump, 0) catch return;
        self.mtx_stdout.lock();
        defer self.mtx_stdout.unlock();
        // fprintf(nullptr) crashes in the C++ too: a child is always constructed with its stream
        _ = fprintf(self.cmd_out.?, "%s%s\n", @as([*:0]const u8, cmd_child_to_router_state), dz.ptr);
        _ = fflush(self.cmd_out);
    }

    /// The `set_state_callback` (server.cpp:667 @8642541b2) lambda a child
    /// gives its context: each state to the router.
    pub fn stateCallback(self: *Role) context.StateCallback {
        return .{ .ctx = self, .func = struct {
            fn f(ctx: ?*anyopaque, state: context.ServerState, payload: *const Value) void {
                const ch: *Role = @ptrCast(@alignCast(ctx.?));
                ch.notifyToRouter(context.serverStateToStr(state), payload);
            }
        }.f };
    }
};

// -----------------------------------------------------------------------------
// server_models_routes

/// Ports `sse_client_id_counter` (server-models.cpp:1904 @8642541b2).
var sse_client_id_counter: std.atomic.Value(c_int) = .init(0);

/// Ports `server_models_sse_client` (server-models.cpp:1905 @8642541b2): like
/// `server_response_reader`, but without a task queue.
const SseClient = struct {
    queue_results: *ResultQueue,
    client_id: c_int,
    ids: queue_mod.IdSet,

    fn init(q: *ResultQueue) !SseClient {
        const client_id = sse_client_id_counter.fetchAdd(1, .monotonic);
        srv(.dbg, "server_models_sse_client", "new SSE client connected, assigned client_id=%d\n", .{client_id});
        try q.addWaitingTaskId(client_id);
        var ids = try queue_mod.IdSet.init(gpa);
        try ids.insert(client_id);
        return .{ .queue_results = q, .client_id = client_id, .ids = ids };
    }

    fn deinit(self: *SseClient) void {
        srv(.dbg, "~server_models_sse_client", "SSE client disconnected, removing client_id=%d\n", .{self.client_id});
        self.queue_results.removeWaitingTaskId(self.client_id);
        self.ids.deinit();
    }

    /// Ports `next` (server-models.cpp:1920 @8642541b2), a member of
    /// `server_models_sse_client`.
    ///
    /// Return: null if `should_stop` holds before a result arrives.
    fn next(self: *SseClient, routes: *ModelsRoutes, should_stop: http.ShouldStop) ?*Result {
        while (true) {
            const http_polling_seconds = 1; // check should_stop every 1 second
            const result = self.queue_results.recvWithTimeout(&self.ids, http_polling_seconds) orelse {
                // timeout, check stop condition
                if (routes.stopping.load(.monotonic) or should_stop.call()) {
                    return null;
                }
                // continue waiting otherwise
                continue;
            };
            if (log.verbosity_thold >= log.level_debug) {
                var arena = std.heap.ArenaAllocator.init(gpa);
                defer arena.deinit();
                const a = arena.allocator();
                if (result.toJson(a)) |v| {
                    var vv = v;
                    const d = common.safeJsonToStr(a, &vv) catch "";
                    srv(.dbg, "next", "recv result for client_id=%d: %s\n", .{ self.client_id, zs(a, d) catch "" });
                } else |_| {}
            }
            return result;
        }
        // should not reach here
    }
};

/// Ports `res_ok` (server-models.cpp:1939 @8642541b2).
fn resOk(res: *http.Res, response_data: *const Value) !void {
    res.status = 200;
    try res.setData(try common.safeJsonToStr(res.allocator(), response_data));
}

/// Ports `res_err` (server-models.cpp:1944 @8642541b2).
fn resErr(res: *http.Res, error_data: *const Value) !void {
    const a = res.allocator();
    res.status = common.jsonValue(error_data, "code", c_int, 500);
    var o = common.Obj.init(a);
    try o.put("error", try error_data.clone(a));
    var v = o.done();
    try res.setData(try common.safeJsonToStr(a, &v));
}

/// Ports `router_validate_model` (server-models.cpp:1949 @8642541b2): the
/// name resolved to its model's, or an error response.
fn routerValidateModel(name: *[]const u8, models: *Models, models_autoload: bool, res: *http.Res) !bool {
    const a = res.allocator();
    if (name.len == 0) {
        var e = try common.formatErrorResponse(a, "model name is missing from the request", .invalid_request);
        try resErr(res, &e);
        return false;
    }
    const meta = (try models.getMeta(a, name.*)) orelse {
        var e = try common.formatErrorResponse(a, try std.fmt.allocPrint(a, "model '{s}' not found", .{name.*}), .invalid_request);
        try resErr(res, &e);
        return false;
    };
    // resolve alias to canonical model name
    name.* = meta.name;
    if (!models_autoload and !meta.isRunning()) {
        var e = try common.formatErrorResponse(a, "model is not loaded", .invalid_request);
        try resErr(res, &e);
        return false;
    }
    return true;
}

/// Ports `is_autoload` (server-models.cpp:1968 @8642541b2).
fn isAutoload(params: *const Params, req: *const http.Req) bool {
    const autoload = req.getParam("autoload", "");
    if (autoload.len == 0) {
        return params.models_autoload;
    } else {
        return std.mem.eql(u8, autoload, "true") or std.mem.eql(u8, autoload, "1");
    }
}

/// Ports `encode_qs` (server-models.cpp:1979 @8642541b2): percent encode one
/// query or path component.
fn encodeQs(a: Allocator, in: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(a, in.len * 3);
    for (in) |ch| {
        const safe = (ch >= 'A' and ch <= 'Z') or (ch >= 'a' and ch <= 'z') or (ch >= '0' and ch <= '9') or
            ch == '-' or ch == '_' or ch == '.' or ch == '~';
        if (safe) {
            out.appendAssumeCapacity(ch);
        } else {
            out.appendSliceAssumeCapacity(&[_]u8{ '%', "0123456789ABCDEF"[ch >> 4], "0123456789ABCDEF"[ch & 15] });
        }
    }
    return out.items;
}

/// Ports `resolve_child_for_conv` (server-models.cpp:2000 @8642541b2): the
/// ready child that owns a conversation's stream session, through the
/// conv_id -> model map.
fn resolveChildForConv(a: Allocator, models: *Models, conversation_id: []const u8) !?Meta {
    if (conversation_id.len == 0) {
        return null;
    }
    const tracked = (try models.conv_models.lookup(a, conversation_id)) orelse return null;
    const meta = try models.getMeta(a, tracked);
    if (meta) |m| if (m.isReady()) {
        return m;
    };
    return null;
}

/// Ports `server_models_routes` (server-models.h:393 @8642541b2).
pub const ModelsRoutes = struct {
    params: Params,
    ui_settings: Value = Value.emptyObject(), // Primary: new name
    ui_arena: std.heap.ArenaAllocator = .init(std.heap.c_allocator),
    stopping: std.atomic.Value(bool) = .init(false), // for graceful disconnecting SSE clients during shutdown
    models: Models,

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    get_router_props: http.Handler = undefined,
    proxy_get: http.Handler = undefined,
    proxy_post: http.Handler = undefined,
    proxy_ws: http.WsHandler = undefined,
    get_router_models: http.Handler = undefined,
    post_router_models_load: http.Handler = undefined,
    post_router_models_unload: http.Handler = undefined,
    // management API
    get_router_models_sse: http.Handler = undefined,
    post_router_models: http.Handler = undefined,
    del_router_models: http.Handler = undefined,

    // router side handlers for the resumable streaming routes. each resolves the child that owns
    // a conversation through the conv_id -> model map, no probing or fan out
    router_stream_get: http.Handler = undefined,
    router_streams_lookup: http.Handler = undefined,
    router_stream_delete: http.Handler = undefined,

    /// Ports the constructor of `server_models_routes` (server-models.h:393
    /// @8642541b2). `self` must not move afterwards.
    pub fn init(self: *ModelsRoutes, params: *const Params, argv: []const [:0]const u8) !void {
        self.* = .{ .params = params.*, .models = undefined };
        try self.models.init(params, argv);
        errdefer self.models.deinit();
        const cfg = self.params.ui_config_json;
        if (cfg.len != 0) {
            const json_settings = json.parse(self.ui_arena.allocator(), cfg) catch |e| {
                const w = try zs(self.ui_arena.allocator(), context.whatOf(e));
                log.err("%s: failed to parse UI config: %s\n", .{ @as([*:0]const u8, "server_models_routes"), w });
                return e;
            };
            self.ui_settings = json_settings;
        }
        try self.initRoutes();
    }

    pub fn deinit(self: *ModelsRoutes) void {
        self.models.deinit();
        self.ui_arena.deinit();
    }

    fn handler(self: *ModelsRoutes, comptime f: fn (*ModelsRoutes, *const http.Req) anyerror!*http.Res) http.Handler {
        return .{ .ctx = self, .func = struct {
            fn call(ctx: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
                return f(@ptrCast(@alignCast(ctx.?)), req);
            }
        }.call };
    }

    /// Ports `init_routes` (server-models.cpp:2016 @8642541b2).
    fn initRoutes(self: *ModelsRoutes) !void {
        if (!tc.subproc.Subproc.isSupported()) {
            return throw(error.RuntimeError, "subprocess is not enabled on this build", .{});
        }

        self.get_router_props = self.handler(getRouterProps);
        self.proxy_get = self.handler(proxyGet);
        self.proxy_post = self.handler(proxyPost);
        self.proxy_ws = .{ .ctx = self, .func = proxyWsRoute };
        self.post_router_models_load = self.handler(postRouterModelsLoad);
        self.get_router_models = self.handler(getRouterModels);
        self.post_router_models_unload = self.handler(postRouterModelsUnload);
        self.get_router_models_sse = self.handler(getRouterModelsSse);
        self.post_router_models = self.handler(postRouterModels);
        self.del_router_models = self.handler(delRouterModels);
        self.router_stream_get = self.handler(routerStreamGet);
        self.router_streams_lookup = self.handler(routerStreamsLookup);
        self.router_stream_delete = self.handler(routerStreamDelete);
    }

    /// The `get_router_props` lambda (server-models.cpp:2021 @8642541b2).
    fn getRouterProps(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const name = req.getParam("model", "");
        if (name.len == 0) {
            // main instance
            const res = try http.Res.create(gpa);
            errdefer res.destroy();
            const a = res.allocator();
            var dgs = common.Obj.init(a);
            try dgs.put("params", @as(Value, .null));
            try dgs.put("n_ctx", @as(c_int, 0));
            var o = common.Obj.init(a);
            // TODO: add support for this on web UI
            try o.put("role", "router");
            try o.put("max_instances", self.params.models_max);
            try o.put("models_autoload", self.params.models_autoload);
            // this is a dummy response to make sure the UI doesn't break
            try o.put("model_alias", "llama-server");
            try o.put("model_path", "none");
            try o.put("default_generation_settings", dgs.done());
            // New key
            try o.put("ui_settings", try self.ui_settings.clone(a));
            try o.put("build_info", result_mod.build_info);
            try o.put("cors_proxy_enabled", self.params.ui_mcp_proxy);
            if (self.params.server_connect_code.len != 0) {
                try o.put("connect_code", self.params.server_connect_code);
            }
            const v = o.done();
            try resOk(res, &v);
            return res;
        }
        return self.proxyGet(req);
    }

    /// The `proxy_get` lambda (server-models.cpp:2052 @8642541b2).
    fn proxyGet(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const method = "GET";
        var name: []const u8 = req.getParam("model", "");
        const autoload = isAutoload(&self.params, req);
        const error_res = try http.Res.create(gpa);
        var keep_error = true;
        defer if (!keep_error) error_res.destroy();
        errdefer if (keep_error) error_res.destroy();
        if (!try routerValidateModel(&name, &self.models, autoload, error_res)) {
            return error_res;
        }
        keep_error = false;
        // `name` lives in the error response's arena until it goes
        const canonical = try gpa.dupe(u8, name);
        defer gpa.free(canonical);
        if (autoload) {
            _ = try self.models.ensureModelReady(canonical, req.should_stop);
        }
        return self.models.proxyRequest(req, method, canonical, false, false);
    }

    /// The `proxy_post` lambda (server-models.cpp:2084 @8642541b2).
    fn proxyPost(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const method = "POST";
        const error_res = try http.Res.create(gpa);
        var keep_error = true;
        defer if (!keep_error) error_res.destroy();
        errdefer if (keep_error) error_res.destroy();
        const a = error_res.allocator();
        var body = try json.parse(a, req.body);
        var name: []const u8 = common.jsonValue(&body, "model", []const u8, "");
        const autoload = isAutoload(&self.params, req);
        if (!try routerValidateModel(&name, &self.models, autoload, error_res)) {
            return error_res;
        }
        // remember which child serves this conversation so the stream routes can route straight
        // to it without polling, keyed on the exact conv id from the header. registered before
        // the load wait so a stop issued while the model loads can erase the entry and cancel
        // this request instead of leaving an orphan generation
        const conv_id = stream.convIdFromHeaders(&req.headers);
        const ticket = try self.models.conv_models.remember(conv_id, name);
        // a dead socket must not cancel a session request, only a stop does (checked right below)
        const should_stop: ?http.ShouldStop = if (ticket == 0) req.should_stop else null;
        const waited = autoload and try self.models.ensureModelReady(name, should_stop);
        if (ticket != 0 and !self.models.conv_models.alive(conv_id, ticket)) {
            srv(.inf, "operator()", "request for conv_id=%s cancelled while model name=%s was loading\n", .{ try zs(a, conv_id), try zs(a, name) });
            var e = try common.formatErrorResponse(a, "request cancelled by a stop while the model was loading", .invalid_request);
            try resErr(error_res, &e);
            return error_res;
        }
        keep_error = false;
        const canonical = try gpa.dupe(u8, name);
        defer gpa.free(canonical);
        // a session request that waited for a load detaches from the client socket: the
        // client may have dropped during the wait (page reload) and the session buffer must
        // still receive the generation for a later resume
        return self.models.proxyRequest(req, method, canonical, true, waited and ticket != 0); // update last usage for POST request only
    }

    /// The `proxy_ws` lambda (server-models.cpp:2066 @8642541b2). The C++'s
    /// handler has no `ex_wrapper`: an exception ends the connection, as a
    /// return from here does.
    fn proxyWsRoute(ctx: ?*anyopaque, req: *const http.Req, conn: *http.WsConn) void {
        const self: *ModelsRoutes = @ptrCast(@alignCast(ctx.?));
        self.proxyWsImpl(req, conn) catch |e| switch (e) {
            else => {},
        };
    }

    fn proxyWsImpl(self: *ModelsRoutes, req: *const http.Req, conn: *http.WsConn) !void {
        var name: []const u8 = req.getParam("model", "");
        const autoload = isAutoload(&self.params, req);
        const error_res = try http.Res.create(gpa);
        defer error_res.destroy();
        if (!try routerValidateModel(&name, &self.models, autoload, error_res)) {
            // error_res holds {"error": {...}}, send it as a websocket error event
            const a = error_res.allocator();
            var err = try json.parse(a, error_res.data.items);
            try err.set(a, "type", try Value.initString(a, "error"));
            _ = conn.send(try common.safeJsonToStr(a, &err));
            conn.close();
            return;
        }
        if (autoload) {
            _ = try self.models.ensureModelReady(name, req.should_stop);
        }
        try self.models.proxyWs(req, conn, name);
    }

    /// The `post_router_models_load` lambda (server-models.cpp:2115 @8642541b2).
    fn postRouterModelsLoad(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();
        var body = try json.parse(a, req.body);
        const name = common.jsonValue(&body, "model", []const u8, "");
        const meta = (try self.models.getMeta(a, name)) orelse {
            var e = try common.formatErrorResponse(a, "model is not found", .not_found);
            try resErr(res, &e);
            return res;
        };
        if (meta.isRunning()) {
            var e = try common.formatErrorResponse(a, "model is already running", .invalid_request);
            try resErr(res, &e);
            return res;
        }
        try self.models.load(meta.name, .{});
        var o = common.Obj.init(a);
        try o.put("success", true);
        const v = o.done();
        try resOk(res, &v);
        return res;
    }

    /// The `get_router_models` lambda (server-models.cpp:2133 @8642541b2).
    fn getRouterModels(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const reload = req.getParam("reload", "").len != 0;
        if (reload) {
            try self.models.loadModels();
        }
        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();
        var models_json = Value.emptyArray();
        const all_models = try self.models.getAllMeta(a);
        const t = time(null);
        // a download on its way has no files to resolve yet, only the kinds it pulls
        const unfinished = try self.models.unfinishedDownloadsGet(a);

        for (all_models) |*meta| {
            if (meta.hidden) {
                continue; // cache model deduplicated by a preset
            }
            var status = common.Obj.init(a);
            try status.put("value", statusToString(meta.status));
            try status.put("args", try stringsArray(a, meta.args));
            if (meta.preset.name.len != 0) {
                var preset_copy = try meta.preset.clone(a);
                unsetReservedArgs(a, &preset_copy, false);
                preset_copy.unsetOption(a, "LLAMA_ARG_HOST");
                preset_copy.unsetOption(a, "LLAMA_ARG_PORT");
                preset_copy.unsetOption(a, "LLAMA_ARG_ALIAS");
                preset_copy.unsetOption(a, "LLAMA_ARG_TAGS");
                try status.put("preset", try preset_copy.toIni(a));
            }
            if (meta.isFailed()) {
                try status.put("exit_code", meta.exit_code);
                try status.put("failed", true);
            }

            var model_info = common.Obj.init(a);
            try model_info.put("id", meta.name);
            try model_info.put("aliases", try stringsArray(a, meta.aliases.items.items));
            try model_info.put("tags", try stringsArray(a, meta.tags.items.items));
            try model_info.put("object", "model"); // for OAI-compat
            try model_info.put("owned_by", "llamacpp"); // for OAI-compat
            try model_info.put("created", @as(i64, t)); // for OAI-compat
            try model_info.put("status", status.done());
            try model_info.put("architecture", try meta.architecture.clone(a));
            try model_info.put("source", sourceToString(meta.source));
            try model_info.put("can_remove", meta.source == .cache);
            // {"need_download", meta.need_download},
            // TODO: add other fields from the GGUF metadata

            if (meta.n_ctx_train > 0) {
                try model_info.put("context_length", meta.n_ctx_train);
            }

            if (self.models.sched.memoryEnabled()) {
                try model_info.put("memory", try meta.memory.toJson(a));
            }

            // the sidecars a model has on disk, resolved as -hf would load them, for a
            // model that resolves from a HF repo. A download on its way reports the
            // kinds it pulls, as an unfinished one does
            const it_unfinished: ?*const UnfinishedDownload = for (unfinished) |*d| {
                if (std.mem.eql(u8, d.model, meta.name)) break d;
            } else null;
            const hf_repo = modelHfRepo(meta);

            if (it_unfinished != null and meta.status == .downloading) {
                try model_info.put("sidecars", try stringsArray(a, it_unfinished.?.sidecars));
            } else if (hf_repo.len != 0) {
                const cached = try cachedSidecars(a, hf_repo);
                try model_info.put("sidecars", try stringsArray(a, cached));

                // the sidecars of an unfinished download of this model that are not on
                // disk yet: a POST with the kind resumes one, a DELETE with it drops it
                if (it_unfinished) |d| {
                    var incomplete: std.ArrayList([]const u8) = .empty;
                    for (d.partials) |p| {
                        if (p.kind.len != 0 and !hasSidecar(incomplete.items, p.kind) and !hasSidecar(cached, p.kind)) {
                            try incomplete.append(a, p.kind);
                        }
                    }
                    if (incomplete.items.len != 0) {
                        try model_info.put("incomplete_sidecars", try stringsArray(a, incomplete.items));
                    }
                }
            }

            var info = model_info.done();

            // merge with loaded_info from the child process if available
            if (meta.isRunning() and meta.loaded_info.isObject()) {
                var i: usize = 0;
                while (i < meta.loaded_info.size()) : (i += 1) {
                    const key = try meta.loaded_info.keyAt(i);
                    if (!info.contains(key)) {
                        try info.set(a, key, try (try meta.loaded_info.at(key)).clone(a));
                    }
                }
            }
            try models_json.pushBack(a, info);
        }

        // a download that never finished has no instance: report it as incomplete so
        // the UI can offer to resume it, without starting anything
        for (unfinished) |*d| {
            if (self.models.hasModel(d.model)) {
                continue; // complete, or downloading right now
            }
            var status = common.Obj.init(a);
            try status.put("value", statusToString(.incomplete));
            try status.put("args", Value.emptyArray());
            var architecture = common.Obj.init(a);
            try architecture.put("input_modalities", try stringsArray(a, &.{"text"}));
            try architecture.put("output_modalities", try stringsArray(a, &.{"text"}));
            var mi = common.Obj.init(a);
            try mi.put("id", d.model);
            try mi.put("aliases", Value.emptyArray());
            try mi.put("tags", Value.emptyArray());
            try mi.put("object", "model");
            try mi.put("owned_by", "llamacpp");
            try mi.put("created", @as(i64, t));
            try mi.put("status", status.done());
            try mi.put("architecture", architecture.done());
            try mi.put("source", sourceToString(.cache));
            try mi.put("can_remove", true);
            try mi.put("sidecars", try stringsArray(a, d.sidecars));
            try models_json.pushBack(a, mi.done());
        }

        var o = common.Obj.init(a);
        try o.put("data", models_json);
        try o.put("object", "list");
        const v = o.done();
        try resOk(res, &v);
        return res;
    }

    /// The `post_router_models_unload` lambda (server-models.cpp:2263 @8642541b2).
    fn postRouterModelsUnload(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();
        var body = try json.parse(a, req.body);
        const name = common.jsonValue(&body, "model", []const u8, "");
        const model = (try self.models.getMeta(a, name)) orelse {
            var e = try common.formatErrorResponse(a, "model is not found", .invalid_request);
            try resErr(res, &e);
            return res;
        };
        if (!model.isRunning() and model.status != .downloading) {
            var e = try common.formatErrorResponse(a, "model is not running", .invalid_request);
            try resErr(res, &e);
            return res;
        }
        try self.models.unload(model.name);
        var o = common.Obj.init(a);
        try o.put("success", true);
        const v = o.done();
        try resOk(res, &v);
        return res;
    }

    /// The `get_router_models_sse` lambda (server-models.cpp:2281 @8642541b2).
    fn getRouterModelsSse(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const sse = try gpa.create(SseRes);
        errdefer gpa.destroy(sse);
        sse.* = .{
            .res = .{ .arena = .init(gpa), .vtable = &SseRes.vtable },
            .routes = self,
            .client = try SseClient.init(&self.models.sse),
            .should_stop = req.should_stop,
        };
        sse.res.status = 200;
        sse.res.content_type = "text/event-stream";
        sse.res.next = .{ .ctx = sse, .func = SseRes.next };
        return &sse.res;
    }

    /// The `post_router_models` lambda (server-models.cpp:2299 @8642541b2):
    /// download a model from the hub in a child, with its sidecars.
    fn postRouterModels(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();

        var body = try json.parse(a, req.body);
        const name = common.jsonValue(&body, "model", []const u8, "");
        if (name.len == 0) {
            return throw(error.InvalidArgument, "model must be a non-empty string", .{});
        }
        const sidecars = try parseSidecars(a, &body);

        var model: tc.params.Model = .{};
        model.hf_repo = name;

        // validate by fetching metadata, through the plan the download follows
        const opts = sidecarDownloadOpts(&self.params, sidecars);
        const plan = download.getHfPlan(a, &model, &opts) catch |e| {
            srv(.err, "operator()", "unknown error while validating model '%s'\n", .{try zs(a, name)});
            // other exceptions will be handled by the outer ex_wrapper()
            return e;
        };

        // a tag that resolves no file has nothing to download
        if (plan.preset.path.len == 0 and plan.primary.path.len == 0 and (try download.planSidecars(a, &plan)).len == 0) {
            return throw(error.InvalidArgument, "model '{s}' resolves no file to download", .{name});
        }

        // then, proceed with the actual download; load() decides over an existing entry
        srv(.inf, "operator()", "starting download for model '%s'\n", .{try zs(a, name)});
        {
            const custom: Meta = .{ .source = .cache, .name = name };
            try self.models.load(name, .{
                .role = .download,
                .custom_meta = &custom,
                .sidecars = sidecars,
                .partials = try download.planPartials(a, &plan),
            });
        }

        var o = common.Obj.init(a);
        try o.put("success", true);
        const v = o.done();
        try resOk(res, &v);
        return res;
    }

    /// The `del_router_models` lambda (server-models.cpp:2344 @8642541b2).
    fn delRouterModels(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();

        const name = req.getParam("model", "");
        if (name.len == 0) {
            return throw(error.InvalidArgument, "model must be a non-empty string", .{});
        }

        // a sidecar alone goes, the model keeps its files
        const sidecar = req.getParam("sidecar", "");
        if (sidecar.len != 0) {
            try checkSidecar(a, sidecar);
            var o = common.Obj.init(a);
            try o.put("success", try self.models.removeSidecar(name, sidecar));
            const v = o.done();
            try resOk(res, &v);
            return res;
        }

        // a download listed from an earlier run has no instance, only its tag and
        // its partial files
        if (!self.models.hasModel(name)) {
            try self.models.unfinishedDownloadsRemove(name);
            var o = common.Obj.init(a);
            try o.put("success", true);
            const v = o.done();
            try resOk(res, &v);
            return res;
        }

        _ = try self.models.remove(name); // throws on error

        var o = common.Obj.init(a);
        try o.put("success", true);
        const v = o.done();
        try resOk(res, &v);
        return res;
    }

    /// The `router_stream_get` lambda (server-models.cpp:2374 @8642541b2):
    /// `GET /v1/stream?conv_id=<id>&from=N`, proxied to the child that owns
    /// the conversation.
    fn routerStreamGet(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const res = try http.Res.create(gpa);
        var keep = true;
        defer if (!keep) res.destroy();
        errdefer if (keep) res.destroy();
        const a = res.allocator();
        const conv_id = req.getParam("conv_id", "");
        if (conv_id.len == 0) {
            var e = try common.formatErrorResponse(a, "Missing conversation id in path", .invalid_request);
            try resErr(res, &e);
            return res;
        }
        const owner = (try resolveChildForConv(a, &self.models, conv_id)) orelse {
            // a registered conv whose model is still loading earns a retry: the session appears
            // once the load ends and the pending request reaches the child
            const tracked = try self.models.conv_models.lookup(a, conv_id);
            const meta = if (tracked) |t| try self.models.getMeta(a, t) else null;
            const transient = if (meta) |m| (m.status == .loading or m.status == .downloading or m.status == .downloaded) else false;
            if (transient) {
                var e = try common.formatErrorResponse(a, "Stream owner model is loading, retry later", .unavailable);
                try resErr(res, &e);
            } else {
                var e = try common.formatErrorResponse(a, "Stream not found or expired", .not_found);
                try resErr(res, &e);
            }
            return res;
        };
        const from = req.getParam("from", "");
        var child_path = try std.mem.concat(a, u8, &.{ "/v1/stream?conv_id=", try encodeQs(a, conv_id) });
        if (from.len != 0) {
            child_path = try std.mem.concat(a, u8, &.{ child_path, "&from=", from });
        }
        srv(.trc, "operator()", "proxying stream resume to model %s on port %d, path=%s\n", .{ try zs(a, owner.name), owner.port, try zs(a, child_path) });
        const proxy = try proxy_mod.Proxy.create(
            "GET",
            "http",
            child_addr,
            owner.port,
            child_path,
            &req.headers,
            req.body,
            &req.files,
            req.should_stop,
            self.params.timeout_read,
            self.params.timeout_write,
        );
        keep = false;
        return proxy;
    }

    /// The `router_streams_lookup` lambda (server-models.cpp:2421
    /// @8642541b2): `POST /v1/streams/lookup`, each id asked of only the
    /// child that owns it.
    fn routerStreamsLookup(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();
        var requested: std.ArrayList([]const u8) = .empty;
        parse: {
            var body = json.parse(a, req.body) catch {
                const v = Value.emptyArray();
                try resOk(res, &v);
                return res;
            };
            if (body.isObject() and body.contains("conversation_ids") and (try body.at("conversation_ids")).isArray()) {
                for ((try body.at("conversation_ids")).array.items) |*v| {
                    if (v.isString() and (try v.get([]const u8)).len != 0) {
                        try requested.append(a, try v.get([]const u8));
                    }
                }
            }
            break :parse;
        }

        // group requested ids by the child port that owns them, drop ids that map to nothing
        // (`std::unordered_map<int, json>`, iterated in libc++'s order)
        var per_child = try llama_libcxx.I32Map(usize).init(a);
        var lists: std.ArrayList(Value) = .empty;
        for (requested.items) |cid| {
            const owner = (try resolveChildForConv(a, &self.models, cid)) orelse continue;
            const idx = per_child.get(owner.port) orelse blk: {
                try lists.append(a, Value.emptyArray());
                try per_child.put(owner.port, lists.items.len - 1);
                break :blk lists.items.len - 1;
            };
            try lists.items[idx].pushBack(a, try Value.initString(a, cid));
        }

        var aggregated = Value.emptyArray();
        var it = per_child.iterator();
        while (it.next()) |e| {
            const port: c_int = e.key;
            var child_body = common.Obj.init(a);
            try child_body.put("conversation_ids", lists.items[e.value]);
            var cb = child_body.done();
            const dump = try json.dump(a, &cb, -1);
            var cli = try client.Client.init(a, child_addr, port);
            defer cli.deinit();
            cli.setConnectionTimeout(0, stream_lookup_timeout_ms * 1000);
            cli.setReadTimeout(0, stream_lookup_timeout_ms * 1000);
            cli.setWriteTimeout(0, stream_lookup_timeout_ms * 1000);
            const resp = cli.post(a, "/v1/streams/lookup", dump, "application/json");
            if (resp.res == null or resp.res.?.status != 200) {
                continue;
            }
            var child_arr = json.parse(a, resp.res.?.body.items) catch continue;
            if (!child_arr.isArray()) {
                continue;
            }
            for (child_arr.array.items) |*entry| {
                if (entry.isObject()) {
                    try aggregated.pushBack(a, try entry.clone(a));
                }
            }
        }
        try resOk(res, &aggregated);
        return res;
    }

    /// The `router_stream_delete` lambda (server-models.cpp:2481
    /// @8642541b2): `DELETE /v1/stream?conv_id=<id>`, forwarded to the
    /// owning child only.
    fn routerStreamDelete(self: *ModelsRoutes, req: *const http.Req) anyerror!*http.Res {
        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();
        const conv_id = req.getParam("conv_id", "");
        if (conv_id.len == 0) {
            var e = try common.formatErrorResponse(a, "Missing conversation id in path", .invalid_request);
            try resErr(res, &e);
            return res;
        }
        const child_path = try std.mem.concat(a, u8, &.{ "/v1/stream?conv_id=", try encodeQs(a, conv_id) });
        if (try resolveChildForConv(a, &self.models, conv_id)) |owner| {
            var cli = try client.Client.init(a, child_addr, owner.port);
            defer cli.deinit();
            cli.setConnectionTimeout(0, stream_lookup_timeout_ms * 1000);
            cli.setReadTimeout(0, stream_lookup_timeout_ms * 1000);
            cli.setWriteTimeout(0, stream_lookup_timeout_ms * 1000);
            _ = cli.delete(a, child_path); // the child logs its own miss when the session is unknown there
        } else if (try self.models.conv_models.lookup(a, conv_id)) |tracked| {
            // the entry exists but its model is still loading: the forget below erases it,
            // which cancels the request parked in proxy_post before the generation starts
            srv(.inf, "operator()", "router stop for conv_id=%s while model name=%s is loading, cancelling the pending request\n", .{ try zs(a, conv_id), try zs(a, tracked) });
        } else {
            srv(.wrn, "operator()", "router stop for unknown conv_id=%s, no owning child in the conv map\n", .{try zs(a, conv_id)});
        }
        // drop the tracking entry, the session is being torn down
        self.models.conv_models.forget(conv_id);
        res.status = 204;
        res.content_type = "application/json";
        return res;
    }
};

const llama_libcxx = @import("llama").libcxx;

/// The response `get_router_models_sse` returns: events from the router's
/// SSE queue until the client goes or the router stops.
const SseRes = struct {
    res: http.Res,
    routes: *ModelsRoutes,
    client: SseClient,
    should_stop: http.ShouldStop,

    const vtable: http.Res.VTable = .{ .on_complete = http.Res.noComplete, .destroy = destroy };

    fn destroy(r: *http.Res) void {
        const self: *SseRes = @fieldParentPtr("res", r);
        self.client.deinit();
        self.res.arena.deinit();
        gpa.destroy(self);
    }

    /// The `next` (server-models.cpp:2286 @8642541b2) lambda of
    /// `get_router_models_sse`.
    fn next(ctx: ?*anyopaque, out: *std.ArrayList(u8), a: Allocator) anyerror!bool {
        const self: *SseRes = @ptrCast(@alignCast(ctx.?));
        const result = self.client.next(self.routes, self.should_stop) orelse {
            return false; // client disconnected or should_stop
        };
        defer result.destroy(gpa);
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        var v = try result.toJson(scratch.allocator());
        const d = try common.safeJsonToStr(scratch.allocator(), &v);
        try out.appendSlice(a, "data: ");
        try out.appendSlice(a, d);
        try out.appendSlice(a, "\n\n");
        return true; // listen for the next event
    }
};

/// `json(std::vector<std::string>)` / `json(std::set<std::string>)`.
fn stringsArray(a: Allocator, items: []const []const u8) !Value {
    var arr = Value.emptyArray();
    for (items) |s| try arr.pushBack(a, try Value.initString(a, s));
    return arr;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "encode_qs keeps the unreserved bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("a-b_c.d~e%20%2F%3A", try encodeQs(arena.allocator(), "a-b_c.d~e /:"));
}
