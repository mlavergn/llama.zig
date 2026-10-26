//! `common_params_parse`: the command line every `common`-based tool shares.
//!
//! # Provenance
//!
//! Ports `common_arg`, `common_params_parse_ex`, `common_params_print_usage`,
//! `common_params_parse` and their helpers from `llama.cpp/common/arg.cpp`
//! (v0.3.0, `c1d0e7a00`). The option table itself is `arg_options.zig`.
//! Each declaration names what it ports and the line it began at.
//!
//! # Exceptions
//!
//! A handler that throws in the C -- `std::invalid_argument`,
//! `std::runtime_error`, or what `std::stoi` and friends throw -- returns
//! `error.Throw` here, with the message the C's `e.what()` gives in `what`.
//! The parser wraps it exactly as the C's `catch` does.
//!
//! # What is refused
//!
//! The system and user config files (`common_params_apply_system_config`,
//! upstream's INI presets) and the model downloader (`-hf`, `-mu`, `-dr`)
//! are not ported: the first refuses to run when a config file exists, the
//! second when a download would be needed. The downloader would be an
//! external dependency.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("tools_h");
const params_mod = @import("params.zig");
const io = @import("io.zig");
const log = @import("log.zig");
const build_info = @import("build_info.zig");
const options = @import("arg_options.zig");

pub const Params = params_mod.Params;
pub const Example = params_mod.Example;
pub const ExampleSet = std.EnumSet(Example);

// -----------------------------------------------------------------------------
// Exceptions

/// `e.what()` of the last throw.
pub var what_buf: [4096]u8 = undefined;
pub var what_len: usize = 0;

/// The thrown message.
pub fn what() []const u8 {
    return what_buf[0..what_len];
}

/// `throw std::...(string_format(fmt, args))`: formats with the C's
/// `snprintf`, so `%d` and `%s` read as the C's do.
pub fn throw(comptime fmt: [*:0]const u8, args: anytype) error{Throw} {
    const n = @call(.auto, snprintf, .{ &what_buf, what_buf.len, fmt } ++ args);
    what_len = @min(@as(usize, @intCast(@max(n, 0))), what_buf.len - 1);
    return error.Throw;
}

/// Records `e.what()` without throwing, for a message the caller returns
/// with a different error.
pub fn setWhat(comptime fmt: [*:0]const u8, args: anytype) void {
    const n = @call(.auto, snprintf, .{ &what_buf, what_buf.len, fmt } ++ args);
    what_len = @min(@as(usize, @intCast(@max(n, 0))), what_buf.len - 1);
}

/// `throw ...(s)` for a string already built.
pub fn throwStr(s: []const u8) error{Throw} {
    what_len = @min(s.len, what_buf.len - 1);
    @memcpy(what_buf[0..what_len], s[0..what_len]);
    return error.Throw;
}

extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;
extern fn strtol(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_long;
extern fn strtoul(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_ulong;
extern fn strtof(s: [*:0]const u8, end: *?[*:0]const u8) f32;
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

fn errnoRange() bool {
    return std.c._errno().* == @intFromEnum(std.c.E.RANGE);
}

/// `std::stoi`, as libc++ throws (string.cpp's `as_integer_helper`).
pub fn stoi(s: [:0]const u8) error{Throw}!c_int {
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtol(s.ptr, &end, 10);
    if (errnoRange()) return throwStr("stoi: out of range");
    if (end == s.ptr) return throwStr("stoi: no conversion");
    if (v < std.math.minInt(c_int) or v > std.math.maxInt(c_int)) return throwStr("stoi: out of range");
    return @intCast(v);
}

/// `std::stoul`.
pub fn stoul(s: [:0]const u8) error{Throw}!c_ulong {
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtoul(s.ptr, &end, 10);
    if (errnoRange()) return throwStr("stoul: out of range");
    if (end == s.ptr) return throwStr("stoul: no conversion");
    return v;
}

/// `std::stof`.
pub fn stof(s: [:0]const u8) error{Throw}!f32 {
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtof(s.ptr, &end);
    if (errnoRange()) return throwStr("stof: out of range");
    if (end == s.ptr) return throwStr("stof: no conversion");
    return v;
}

/// `std::string(s)` for a value the handler keeps.
pub fn dupe(p: *Params, s: []const u8) ![:0]const u8 {
    return p.arena.dupeSentinel(u8, s, 0);
}

// -----------------------------------------------------------------------------
// common_arg

/// The five handler signatures, `handler_void`, `handler_string`,
/// `handler_str_str`, `handler_int` and `handler_bool`
/// (arg.h:34, 35, 36, 37, 38 @c1d0e7a00).
pub const Handler = union(enum) {
    void_: *const fn (*Params) anyerror!void,
    string: *const fn (*Params, [:0]const u8) anyerror!void,
    str_str: *const fn (*Params, [:0]const u8, [:0]const u8) anyerror!void,
    int: *const fn (*Params, c_int) anyerror!void,
    bool_: *const fn (*Params, bool) anyerror!void,
};

/// Ports `common_arg` (arg.h:22 @c1d0e7a00).
pub const Arg = struct {
    examples: ExampleSet = ExampleSet.initOne(.common),
    excludes: ExampleSet = ExampleSet.empty,
    args: []const [:0]const u8,
    args_neg: []const [:0]const u8 = &.{},
    value_hint: ?[:0]const u8 = null, // help text or example for arg value
    value_hint_2: ?[:0]const u8 = null, // for second arg value
    env: ?[:0]const u8 = null,
    help: []const u8,
    is_sampling: bool = false, // is current arg a sampling param?
    is_spec: bool = false, // is current arg a speculative decoding param?
    is_preset_only: bool = false, // is current arg preset-only (not treated as CLI arg)
    handler: Handler,

    /// Ports `set_examples` (arg.cpp:86 @c1d0e7a00), a member of `common_arg`.
    pub fn setExamples(self: Arg, examples: []const Example) Arg {
        var r = self;
        r.examples = ExampleSet.empty;
        for (examples) |e| r.examples.insert(e);
        return r;
    }

    /// Ports `set_excludes` (arg.cpp:91 @c1d0e7a00), a member of `common_arg`.
    pub fn setExcludes(self: Arg, excludes: []const Example) Arg {
        var r = self;
        r.excludes = ExampleSet.empty;
        for (excludes) |e| r.excludes.insert(e);
        return r;
    }

    /// Ports `set_env` (arg.cpp:96 @c1d0e7a00), a member of `common_arg`. The C appends
    /// `\n(env: NAME)` to the help here; `toString` does.
    pub fn setEnv(self: Arg, env: [:0]const u8) Arg {
        var r = self;
        r.env = env;
        return r;
    }

    /// Ports `set_sampling` (arg.cpp:102 @c1d0e7a00), a member of `common_arg`.
    pub fn setSampling(self: Arg) Arg {
        var r = self;
        r.is_sampling = true;
        return r;
    }

    /// Ports `set_spec` (arg.cpp:107 @c1d0e7a00), a member of `common_arg`.
    pub fn setSpec(self: Arg) Arg {
        var r = self;
        r.is_spec = true;
        return r;
    }

    /// Ports `set_preset_only` (arg.cpp:112 @c1d0e7a00), a member of `common_arg`.
    pub fn setPresetOnly(self: Arg) Arg {
        var r = self;
        r.is_preset_only = true;
        return r;
    }

    /// Ports `in_example` (arg.cpp:117 @c1d0e7a00), a member of `common_arg`.
    pub fn inExample(self: *const Arg, ex: Example) bool {
        return self.examples.contains(ex);
    }

    /// Ports `is_exclude` (arg.cpp:121 @c1d0e7a00), a member of `common_arg`.
    pub fn isExclude(self: *const Arg, ex: Example) bool {
        return self.excludes.contains(ex);
    }

    /// The `LLAMA_ARG_NO_` spelling of `env` (arg.cpp:129 @c1d0e7a00).
    fn negEnv(self: *const Arg, buf: []u8) ?[:0]const u8 {
        const env = self.env orelse return null;
        if (!std.mem.startsWith(u8, env, "LLAMA_ARG_")) return std.fmt.bufPrintSentinel(buf, "{s}", .{env}, 0) catch null;
        return std.fmt.bufPrintSentinel(buf, "LLAMA_ARG_NO_{s}", .{env["LLAMA_ARG_".len..]}, 0) catch null;
    }

    /// Ports `get_value_from_env` (arg.cpp:125 @c1d0e7a00), a member of `common_arg`.
    pub fn getValueFromEnv(self: *const Arg) ?[:0]const u8 {
        const env = self.env orelse return null;
        if (self.args_neg.len != 0) {
            // for compatibility, we need to check LLAMA_ARG_NO_ env as well
            var buf: [256]u8 = undefined;
            if (self.negEnv(&buf)) |neg| {
                if (getenv(neg.ptr) != null) return "0"; // falsey
            }
        }
        if (getenv(env.ptr)) |v| return std.mem.span(v);
        return null;
    }

    /// Ports `has_value_from_env` (arg.cpp:145 @c1d0e7a00), a member of `common_arg`.
    pub fn hasValueFromEnv(self: *const Arg) bool {
        const env = self.env orelse return false;
        if (self.args_neg.len != 0) {
            var buf: [256]u8 = undefined;
            if (self.negEnv(&buf)) |neg| {
                if (getenv(neg.ptr) != null) return true;
            }
        }
        return getenv(env.ptr) != null;
    }

    /// Ports `to_string` (arg.cpp:184 @c1d0e7a00), a member of `common_arg`: the option
    /// as `--help` lists it.
    pub fn toString(self: *const Arg, a: std.mem.Allocator) ![]u8 {
        // params for printing to console
        const n_leading_spaces = 40;
        const n_char_per_line_help = 70; // TODO: detect this based on current console
        const leading_spaces: [n_leading_spaces]u8 = @splat(' ');

        var ss: std.ArrayList(u8) = .empty;
        const n_all = self.args.len + self.args_neg.len;
        var k: usize = 0;
        while (k < n_all) : (k += 1) {
            const arg = if (k < self.args.len) self.args[k] else self.args_neg[k - self.args.len];
            // `arg == all_args.front()` compares strings, so a later spelling
            // equal to the first takes the first's branch too
            const front = if (self.args.len > 0) self.args[0] else self.args_neg[0];
            const back = if (self.args_neg.len > 0) self.args_neg[self.args_neg.len - 1] else self.args[self.args.len - 1];
            if (std.mem.eql(u8, arg, front)) {
                if (n_all == 1) {
                    try ss.appendSlice(a, arg);
                } else {
                    // first arg is usually abbreviation, we need padding to make it more beautiful
                    try ss.appendSlice(a, arg);
                    try ss.appendSlice(a, ", ");
                    const tmp_len: i64 = @intCast(arg.len + 2);
                    try ss.appendNTimes(a, ' ', @intCast(@max(0, 7 - tmp_len)));
                }
            } else {
                try ss.appendSlice(a, arg);
                if (!std.mem.eql(u8, arg, back)) try ss.appendSlice(a, ", ");
            }
        }
        if (self.value_hint) |h| {
            try ss.append(a, ' ');
            try ss.appendSlice(a, h);
        }
        if (self.value_hint_2) |h| {
            try ss.append(a, ' ');
            try ss.appendSlice(a, h);
        }
        if (ss.items.len > n_leading_spaces - 3) {
            // current line is too long, add new line
            try ss.append(a, '\n');
            try ss.appendSlice(a, &leading_spaces);
        } else {
            // padding between arg and help, same line
            try ss.appendNTimes(a, ' ', n_leading_spaces - ss.items.len);
        }
        // `set_env` appended this to the help when it was called
        // (arg.cpp:96 @c1d0e7a00); nothing changes the help after it.
        const help = if (self.env) |env| try std.mem.concat(a, u8, &.{ self.help, "\n(env: ", env, ")" }) else self.help;
        const help_lines = try breakStrIntoLines(a, help, n_char_per_line_help);
        for (help_lines.items, 0..) |line, i| {
            if (i != 0) try ss.appendSlice(a, &leading_spaces);
            try ss.appendSlice(a, line);
            try ss.append(a, '\n');
        }
        return ss.items;
    }
};

/// Ports `break_str_into_lines` (arg.cpp:157 @c1d0e7a00): each line of
/// `input` as `std::getline` reads them; one longer than the limit is
/// re-flowed word by word, words as `operator>>` reads them.
fn breakStrIntoLines(a: std.mem.Allocator, input: []const u8, max_char_per_line: usize) !std.ArrayList([]const u8) {
    var result: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, input, '\n');
    var idx: usize = 0;
    while (it.next()) |l| {
        // `std::getline` yields no final empty line after a trailing '\n'
        idx += l.len + 1;
        if (l.len == 0 and idx > input.len) break;
        if (l.len <= max_char_per_line) {
            try result.append(a, l);
        } else {
            var current_line: std.ArrayList(u8) = .empty;
            var words = std.mem.tokenizeAny(u8, l, " \t\n\x0b\x0c\r");
            while (words.next()) |word| {
                const sep: usize = @intFromBool(current_line.items.len != 0);
                if (current_line.items.len + sep + word.len > max_char_per_line) {
                    if (current_line.items.len != 0) try result.append(a, current_line.items);
                    current_line = .empty;
                    try current_line.appendSlice(a, word);
                } else {
                    if (current_line.items.len != 0) try current_line.append(a, ' ');
                    try current_line.appendSlice(a, word);
                }
            }
            if (current_line.items.len != 0) try result.append(a, current_line.items);
        }
    }
    return result;
}

/// Ports `is_truthy` (arg.cpp:1331 @c1d0e7a00), a member of `common_arg_utils`.
pub fn isTruthy(value: []const u8) bool {
    return eqlAny(value, &.{ "on", "enabled", "true", "1" });
}

/// Ports `is_falsey` (arg.cpp:1335 @c1d0e7a00), a member of `common_arg_utils`.
pub fn isFalsey(value: []const u8) bool {
    return eqlAny(value, &.{ "off", "disabled", "false", "0" });
}

/// Ports `is_autoy` (arg.cpp:1339 @c1d0e7a00), a member of `common_arg_utils`.
pub fn isAutoy(value: []const u8) bool {
    return eqlAny(value, &.{ "auto", "-1" });
}

fn eqlAny(v: []const u8, list: []const []const u8) bool {
    for (list) |s| if (std.mem.eql(u8, v, s)) return true;
    return false;
}

/// Ports `parse_bool_value` (arg.cpp:333 @c1d0e7a00).
fn parseBoolValue(value: []const u8) error{Throw}!bool {
    if (isTruthy(value)) return true;
    if (isFalsey(value)) return false;
    var buf: [512]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "invalid boolean value: {s}", .{value}) catch "invalid boolean value";
    return throwStr(msg);
}

// -----------------------------------------------------------------------------
// common_params_context

/// Ports `common_params_context` (arg.h:115 @c1d0e7a00).
pub const Context = struct {
    ex: Example,
    params: *Params,
    options: std.ArrayList(Arg) = .empty,
    print_usage: ?*const fn (argv: []const [:0]const u8) void = null,

    /// The `add_opt` lambda (arg.cpp:1432 @c1d0e7a00): keep an option when
    /// it belongs to this example, or to all of them and this is not
    /// `download`, unless this example is excluded.
    pub fn add(self: *Context, arg: Arg) !void {
        // download only exposes the handful of args explicitly tagged for it
        const inherit_common = self.ex != .download;
        if ((arg.inExample(self.ex) or (inherit_common and arg.inExample(.common))) and !arg.isExclude(self.ex)) {
            try self.options.append(self.params.arena, arg);
        }
    }

    /// `string_format(...)` for a help text, built when the table is.
    pub fn fmt(self: *Context, comptime f: [*:0]const u8, args: anytype) ![]const u8 {
        const a = self.params.arena;
        const n: usize = @intCast(@call(.auto, snprintf, .{ null, 0, f } ++ args));
        const buf = try a.alloc(u8, n + 1);
        _ = @call(.auto, snprintf, .{ buf.ptr, n + 1, f } ++ args);
        return buf[0..n];
    }
};

/// Ports `common_params_parser_init` (arg.cpp:1391 @c1d0e7a00): the
/// per-example defaults, then the table.
pub fn parserInit(params: *Params, ex: Example, print_usage: ?*const fn ([]const [:0]const u8) void) !Context {
    // per-example default params
    switch (ex) {
        .tokenize => params.parse_special = true, // parse special tokens by default, like the old tokenize tool
        else => {},
    }

    params.use_color = log.ttyCanUseColors();

    var ctx: Context = .{ .ex = ex, .params = params, .print_usage = print_usage };
    try options.addAll(&ctx);
    return ctx;
}

// -----------------------------------------------------------------------------
// Parsing

/// The config files `common_params_apply_system_config` reads
/// (arg.cpp:719 @c1d0e7a00). None may exist: their INI presets are not
/// ported.
fn refuseSystemConfig(a: std.mem.Allocator) !void {
    var paths: std.ArrayList([:0]const u8) = .empty;
    if (comptime builtin.os.tag == .windows) {
        if (getenv("PROGRAMDATA")) |pd| try paths.append(a, try std.fmt.allocPrintSentinel(a, "{s}\\llama.cpp\\config.ini", .{pd}, 0));
        if (getenv("APPDATA")) |ad| try paths.append(a, try std.fmt.allocPrintSentinel(a, "{s}\\llama.cpp\\config.ini", .{ad}, 0));
    } else {
        try paths.append(a, "/etc/llama.cpp/config.ini");
        if (getenv("XDG_CONFIG_HOME")) |x| {
            try paths.append(a, try std.fmt.allocPrintSentinel(a, "{s}/llama.cpp/config.ini", .{x}, 0));
        } else if (getenv("HOME")) |h| {
            try paths.append(a, try std.fmt.allocPrintSentinel(a, "{s}/.config/llama.cpp/config.ini", .{h}, 0));
        }
    }
    for (paths.items) |p| {
        var st: std.c.Stat = undefined;
        if (std.c.stat(p.ptr, &st) == 0) {
            io.eprintf("error: config file %s found, but config files are not supported by this port\n", .{p.ptr});
            exitC(1);
        }
    }
}

extern fn exit(code: c_int) noreturn;
fn exitC(code: c_int) noreturn {
    exit(code);
}

/// Ports `postprocess_cpu_params` (common/common.cpp:288 @c1d0e7a00).
pub fn postprocessCpuParams(cpuparams: *params_mod.CpuParams, role_model: ?*const params_mod.CpuParams) void {
    var n_set: i32 = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model) |r| {
            cpuparams.* = r.*;
        } else {
            cpuparams.n_threads = @import("cpu.zig").numMath();
        }
    }

    for (cpuparams.cpumask) |b| {
        if (b) n_set += 1;
    }

    if (n_set != 0 and n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        log.wrn("cmn  %12.*s: Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", .{ @as(c_int, 12), "postprocess_cpu_params", n_set, cpuparams.n_threads });
    }
}

/// The two kinds of failure `common_params_parse` tells apart: an invalid
/// argument, which returns false, and anything else, which exits 1.
const ParseError = error{ InvalidArgument, Other };

/// Ports `common_params_parse_ex` (arg.cpp:762 @c1d0e7a00).
fn parseEx(argv: []const [:0]const u8, ctx: *Context) !void {
    const params = ctx.params;
    const a = params.arena;

    // setup log directly from params.verbosity: see tools/cli/cli.cpp
    log.verbosity_thold = params.verbosity;

    // config file applies first, so env variables and CLI arguments override it
    try refuseSystemConfig(a);

    // handle environment variables
    for (ctx.options.items) |*opt| {
        const value = opt.getValueFromEnv() orelse continue;
        (blk: {
            switch (opt.handler) {
                .void_ => |h| if (isTruthy(value)) h(params) catch |e| break :blk e,
                .int => |h| h(params, stoi(value) catch |e| break :blk e) catch |e| break :blk e,
                .bool_ => |h| h(params, parseBoolValue(value) catch |e| break :blk e) catch |e| break :blk e,
                .string => |h| h(params, value) catch |e| break :blk e,
                .str_str => {},
            }
            break :blk {};
        }) catch |e| {
            if (e != error.Throw) return e;
            var inner: [4096]u8 = undefined;
            const w = what();
            @memcpy(inner[0..w.len], w);
            setWhat("error while handling environment variable \"%s\": %.*s\n\n", .{ opt.env.?.ptr, @as(c_int, @intCast(w.len)), &inner });
            return error.InvalidArgument;
        };
    }

    var seen_args: std.StringHashMapUnmanaged(void) = .empty;

    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg_buf = try a.dupeSentinel(u8, argv[i], 0);
        if (std.mem.startsWith(u8, arg_buf, "--")) std.mem.replaceScalar(u8, arg_buf, '_', '-');
        const arg: [:0]const u8 = arg_buf;

        // `arg_to_options`: a later option's spelling replaces an earlier one's
        var found: ?*Arg = null;
        var is_positive = true;
        for (ctx.options.items) |*opt| {
            for (opt.args) |s| if (std.mem.eql(u8, s, arg)) {
                found = opt;
                is_positive = true;
            };
            for (opt.args_neg) |s| if (std.mem.eql(u8, s, arg)) {
                found = opt;
                is_positive = false;
            };
        }
        const opt = found orelse {
            setWhat("error: invalid argument: %s", .{arg.ptr});
            return error.InvalidArgument;
        };
        const gop = try seen_args.getOrPut(a, arg);
        if (gop.found_existing) {
            const skip = std.mem.eql(u8, arg, "--spec-type");
            if (!skip) log.wrn("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\n", .{arg.ptr});
        }
        if (opt.hasValueFromEnv()) {
            io.eprintf("warn: %s environment variable is set, but will be overwritten by command line argument %s\n", .{ opt.env.?.ptr, arg.ptr });
        }
        (blk: {
            switch (opt.handler) {
                .void_ => |h| {
                    h(params) catch |e| break :blk e;
                    break :blk {};
                },
                .bool_ => |h| {
                    h(params, is_positive) catch |e| break :blk e;
                    break :blk {};
                },
                else => {},
            }
            // arg with single value
            if (i + 1 >= argv.len) break :blk throwStr("expected value for argument");
            i += 1;
            const val = argv[i];
            switch (opt.handler) {
                .int => |h| {
                    h(params, stoi(val) catch |e| break :blk e) catch |e| break :blk e;
                    break :blk {};
                },
                .string => |h| {
                    h(params, val) catch |e| break :blk e;
                    break :blk {};
                },
                else => {},
            }
            // arg with 2 values
            if (i + 1 >= argv.len) break :blk throwStr("expected value for argument");
            i += 1;
            const val2 = argv[i];
            switch (opt.handler) {
                .str_str => |h| h(params, val, val2) catch |e| break :blk e,
                else => {},
            }
            break :blk {};
        }) catch |e| {
            if (e != error.Throw) return e;
            const w = try a.dupe(u8, what());
            const usage = try opt.toString(a);
            setWhat("error while handling argument \"%s\": %.*s\n\nusage:\n%.*s\n\nto show complete usage, run with -h", .{
                arg.ptr, @as(c_int, @intCast(w.len)), w.ptr, @as(c_int, @intCast(usage.len)), usage.ptr,
            });
            return error.InvalidArgument;
        };
    }

    // TODO: remove this check after deprecating --mmap|mlock|dio
    const lm = seen_args.contains("-lm") or seen_args.contains("--load-mode");
    var old = false;
    for ([_][]const u8{ "--mlock", "--mmap", "--no-mmap", "-dio", "--direct-io", "-ndio", "--no-direct-io" }) |n| {
        if (seen_args.contains(n)) old = true;
    }
    if (lm and old) {
        log.wrn("DEPRECATED: `--load-mode` and `--mlock`/`--mmap`/`--direct-io` should not be combined; only the last flag on the command line will take effect\n", .{});
    }

    postprocessCpuParams(&params.cpuparams, null);
    postprocessCpuParams(&params.cpuparams_batch, &params.cpuparams);

    // handle model and download: the download paths are refused
    if (params.model.url.len != 0 or params.model.hf_repo.len != 0 or params.model.docker_repo.len != 0) {
        setWhat("error: model downloads (-hf, -mu, -dr) are not supported by this port; pass a local file with -m\n", .{});
        return error.InvalidArgument;
    }

    // model is required (except for server)
    const can_skip_model = params.usage or params.completion;
    if (!can_skip_model and params.model.path.len == 0) {
        setWhat("error: --model is required\n", .{});
        return error.InvalidArgument;
    }

    if (params.escape) {
        params.prompt = try processEscapes(a, params.prompt);
        for (params.sampling.dry_sequence_breakers.items) |*s| s.* = try processEscapes(a, s.*);
    }

    if (params.kv_overrides.items.len != 0) {
        try params.kv_overrides.append(a, std.mem.zeroes(c.llama_model_kv_override));
    }

    // pad tensor_buft_overrides for llama_params_fit:
    const ntbo = c.llama_max_tensor_buft_overrides();
    while (params.tensor_buft_overrides.items.len < ntbo) {
        try params.tensor_buft_overrides.append(a, .{ .pattern = null, .buft = null });
    }
}

/// Ports `string_process_escapes` (common/common.cpp:645 @c1d0e7a00).
pub fn processEscapes(a: std.mem.Allocator, input: []const u8) ![]const u8 {
    const out = try a.alloc(u8, input.len);
    const input_len = input.len;
    var output_idx: usize = 0;

    var input_idx: usize = 0;
    while (input_idx < input_len) : (input_idx += 1) {
        if (input[input_idx] == '\\' and input_idx + 1 < input_len) {
            input_idx += 1;
            switch (input[input_idx]) {
                'n' => {
                    out[output_idx] = '\n';
                    output_idx += 1;
                },
                'r' => {
                    out[output_idx] = '\r';
                    output_idx += 1;
                },
                't' => {
                    out[output_idx] = '\t';
                    output_idx += 1;
                },
                '\'' => {
                    out[output_idx] = '\'';
                    output_idx += 1;
                },
                '"' => {
                    out[output_idx] = '"';
                    output_idx += 1;
                },
                '\\' => {
                    out[output_idx] = '\\';
                    output_idx += 1;
                },
                'x' => {
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const x = [3:0]u8{ input[input_idx + 1], input[input_idx + 2], 0 };
                        var err_p: ?[*:0]const u8 = null;
                        const val = strtol(&x, &err_p, 16);
                        if (err_p == @as([*:0]const u8, &x) + 2) {
                            input_idx += 2;
                            out[output_idx] = @truncate(@as(c_ulong, @bitCast(val)));
                            output_idx += 1;
                            continue;
                        }
                    }
                    // fall through
                    out[output_idx] = '\\';
                    out[output_idx + 1] = input[input_idx];
                    output_idx += 2;
                },
                else => |ch| {
                    out[output_idx] = '\\';
                    out[output_idx + 1] = ch;
                    output_idx += 2;
                },
            }
        } else {
            out[output_idx] = input[input_idx];
            output_idx += 1;
        }
    }

    return out[0..output_idx];
}

/// Ports `common_params_print_usage` (arg.cpp:966 @c1d0e7a00).
fn printUsage(ctx: *Context) !void {
    const a = ctx.params.arena;
    var common_options: std.ArrayList(*Arg) = .empty;
    var sampling_options: std.ArrayList(*Arg) = .empty;
    var spec_options: std.ArrayList(*Arg) = .empty;
    var specific_options: std.ArrayList(*Arg) = .empty;
    for (ctx.options.items) |*opt| {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sampling) {
            try sampling_options.append(a, opt);
        } else if (opt.is_spec) {
            try spec_options.append(a, opt);
        } else if (opt.inExample(ctx.ex)) {
            try specific_options.append(a, opt);
        } else {
            try common_options.append(a, opt);
        }
    }
    var first = true;
    const sections = [_]struct { [*:0]const u8, *std.ArrayList(*Arg) }{
        .{ "common params", &common_options },
        .{ "sampling params", &sampling_options },
        .{ "speculative params", &spec_options },
        .{ "example-specific params", &specific_options },
    };
    for (sections) |sec| {
        if (sec[1].items.len == 0) continue;
        io.printf("%s----- %s -----\n\n", .{ @as([*:0]const u8, if (first) "" else "\n\n"), sec[0] });
        first = false;
        for (sec[1].items) |opt| {
            const s = try opt.toString(a);
            io.printf("%.*s", .{ @as(c_int, @intCast(s.len)), s.ptr });
        }
    }
}

/// Ports `common_params_parse` (arg.cpp:1277 @c1d0e7a00).
///
/// Parameters:
/// - `argv`: the command line, `argv[0]` included.
/// - `params`: the tool's defaults in, the parsed values out.
/// - `ex`: which tool's options apply.
/// - `print_usage`: the tool's own usage, printed after the options.
///
/// Return: false for an invalid argument, having printed why. `--help`
/// prints and exits 0; any other failure prints and exits 1, as the C does.
pub fn parse(argv: []const [:0]const u8, params: *Params, ex: Example, print_usage: ?*const fn ([]const [:0]const u8) void) !bool {
    var ctx = try parserInit(params, ex, print_usage);
    const params_org = params.*; // the example can modify the default params

    parseEx(argv, &ctx) catch |e| switch (e) {
        error.InvalidArgument => {
            io.eprintf("%.*s\n", .{ @as(c_int, @intCast(what_len)), &what_buf });
            params.* = params_org;
            return false;
        },
        error.Other => {
            io.eprintf("%.*s\n", .{ @as(c_int, @intCast(what_len)), &what_buf });
            exitC(1); // for other exceptions, we exit with status code 1
        },
        else => |other| return other,
    };
    if (params.usage) {
        try printUsage(&ctx);
        if (ctx.print_usage) |pu| pu(argv);
        exitC(0);
    }
    if (params.completion) {
        io.eprintf("error: --completion-bash is not supported by this port\n", .{});
        exitC(1);
    }
    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
