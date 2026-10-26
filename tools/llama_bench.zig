//! `llama-bench`: time prompt processing and generation over the cross
//! product of the parameters given, and print one row per test.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/llama-bench/llama-bench.cpp` and
//! `tools/llama-bench/main.cpp` at v0.3.0 (`c1d0e7a00`). Each declaration
//! names the C++ it replaces and the line it began at.
//!
//! # Its own command line
//!
//! llama-bench parses its arguments itself rather than through
//! `common_params_parse`: every test parameter takes a comma-separated list
//! and ranges (`parse_int_range`, an `std::regex` matched here by hand with
//! the same backtracking), booleans go through `std::istringstream >>
//! bool`, and `-ts` splits on `std::sregex_token_iterator` over `[;/]+`.
//! `-hf` downloads a model upstream; no download is ported (`SPEC.md`), so
//! it fails as upstream does when the download fails.

const std = @import("std");
const builtin = @import("builtin");
const tc = @import("tools_common");
const fit = @import("cli_fit");
const c = tc.c;
const io = tc.io;
const log = tc.log;
const arg = tc.arg;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

const gpa = std.heap.c_allocator;

extern fn fprintf(stream: *anyopaque, format: [*:0]const u8, ...) c_int;
extern fn fflush(stream: ?*anyopaque) c_int;
extern fn rand() c_int;
extern fn strtoull(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_ulonglong;
extern fn time(t: ?*i64) i64;
extern fn gmtime(t: *const i64) ?*anyopaque;
extern fn strftime(buf: [*]u8, max: usize, format: [*:0]const u8, tm: ?*const anyopaque) usize;
extern fn setlocale(category: c_int, locale: ?[*:0]const u8) ?[*:0]const u8;
extern fn sleep(seconds: c_uint) c_uint;

const Override = c.llama_model_tensor_buft_override;

fn fput(f: *anyopaque, comptime fmt: [*:0]const u8, args: anytype) void {
    _ = @call(.auto, fprintf, .{ f, fmt } ++ args);
}

/// `GGML_ASSERT`, which upstream keeps in every build mode.
fn ggmlAssert(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort("llama-bench.cpp", line, "GGML_ASSERT(%s) failed", expr);
}

/// Ports `get_time_ns` (llama-bench.cpp:40 @c1d0e7a00).
fn getTimeNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Ports `tensor_buft_override_equal` (llama-bench.cpp:45 @c1d0e7a00).
fn overrideEqual(a: Override, b: Override) bool {
    if (a.pattern != b.pattern) {
        // cString comparison that may be null
        if (a.pattern == null or b.pattern == null) return false;
        if (!std.mem.eql(u8, std.mem.span(a.pattern), std.mem.span(b.pattern))) return false;
    }
    return a.buft == b.buft;
}

/// Ports `vec_tensor_buft_override_equal` (llama-bench.cpp:61 @c1d0e7a00).
fn overridesEqual(a: []const Override, b: []const Override) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!overrideEqual(x, y)) return false;
    }
    return true;
}

/// Ports `vec_vec_tensor_buft_override_equal` (llama-bench.cpp:73 @c1d0e7a00).
fn overrideListsEqual(a: []const []const Override, b: []const []const Override) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!overridesEqual(x, y)) return false;
    }
    return true;
}

/// `join` (llama-bench.cpp:85 @c1d0e7a00): each value as
/// `std::ostream <<` writes it -- a `double` as `%g`, a `bool` as `0` or `1`.
fn join(a: std.mem.Allocator, comptime T: type, values: []const T, delim: []const u8) ![:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (values, 0..) |v, i| {
        switch (T) {
            []const u8, [:0]const u8 => try out.appendSlice(a, v),
            [*:0]const u8 => try out.appendSlice(a, std.mem.span(v)),
            bool => try out.append(a, if (v) '1' else '0'),
            f64 => {
                var buf: [64]u8 = undefined;
                const n = snprintf(&buf, buf.len, "%g", v);
                try out.appendSlice(a, buf[0..@intCast(n)]);
            },
            else => try out.print(a, "{d}", .{v}),
        }
        if (i < values.len - 1) try out.appendSlice(a, delim);
    }
    return a.dupeSentinel(u8, out.items, 0);
}

extern fn snprintf(buf: [*]u8, size: usize, format: [*:0]const u8, ...) c_int;

/// Ports `avg` (llama-bench.cpp:102 @c1d0e7a00).
fn avgU64(v: []const u64) u64 {
    if (v.len == 0) return 0;
    var sum: u64 = 0;
    for (v) |x| sum +%= x;
    return sum / v.len;
}

fn avgF64(v: []const f64) f64 {
    if (v.len == 0) return 0;
    var sum: f64 = 0;
    for (v) |x| sum += x;
    return sum / @as(f64, @floatFromInt(v.len));
}

/// Ports `stdev` (llama-bench.cpp:110 @c1d0e7a00), over `uint64_t`:
/// integer arithmetic throughout, as the template instantiates it.
fn stdevU64(v: []const u64) u64 {
    if (v.len <= 1) return 0;
    const n: u64 = v.len;
    const mean = avgU64(v);
    var sq_sum: u64 = 0;
    for (v) |x| sq_sum +%= x *% x;
    const inner = sq_sum / (n - 1) -% mean *% mean *% n / (n - 1);
    return @intFromFloat(@sqrt(@as(f64, @floatFromInt(inner))));
}

fn stdevF64(v: []const f64) f64 {
    if (v.len <= 1) return 0;
    const n: f64 = @floatFromInt(v.len);
    const mean = avgF64(v);
    var sq_sum: f64 = 0;
    for (v) |x| sq_sum = @mulAdd(f64, x, x, sq_sum);
    return @sqrt(sq_sum / (n - 1) - mean * mean * n / (n - 1));
}

/// Ports `get_cpu_info` and `get_gpu_info` (llama-bench.cpp:120, 132 @c1d0e7a00).
fn devInfo(a: std.mem.Allocator, gpu: bool) ![:0]u8 {
    var list: std.ArrayList([*:0]const u8) = .empty;
    var i: usize = 0;
    while (i < c.ggml_backend_dev_count()) : (i += 1) {
        const dev = c.ggml_backend_dev_get(i);
        const t = c.ggml_backend_dev_type(dev);
        const want = if (gpu)
            t == c.GGML_BACKEND_DEVICE_TYPE_GPU or t == c.GGML_BACKEND_DEVICE_TYPE_IGPU
        else
            t == c.GGML_BACKEND_DEVICE_TYPE_CPU or t == c.GGML_BACKEND_DEVICE_TYPE_ACCEL;
        if (want) try list.append(a, c.ggml_backend_dev_description(dev));
    }
    return join(a, [*:0]const u8, list.items, ", ");
}

/// Ports `string_strip` (common/common.cpp:458 @c1d0e7a00).
fn strip(s: []const u8) []const u8 {
    var start: usize = 0;
    var end = s.len;
    while (start < end and isspace(s[start])) start += 1;
    while (end > start and isspace(s[end - 1])) end -= 1;
    return s[start..end];
}

fn isspace(ch: u8) bool {
    return ch == ' ' or (ch >= '\t' and ch <= '\r');
}

/// What a thrown `std::invalid_argument` carries out of a parser here.
const Thrown = error{Throw};

/// Ports `parse_devices_arg` (llama-bench.cpp:144 @c1d0e7a00).
fn parseDevicesArg(a: std.mem.Allocator, value: []const u8) !std.ArrayList(c.ggml_backend_dev_t) {
    var devices: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    const trimmed = strip(value);
    if (trimmed.len == 0) return arg.throwStr("no devices specified");
    if (std.mem.eql(u8, trimmed, "auto")) return devices;

    const dev_names = try tc.strings.split(a, trimmed, '/');
    if (dev_names.items.len == 1 and std.mem.eql(u8, strip(dev_names.items[0]), "none")) {
        try devices.append(a, null);
        return devices;
    }

    for (dev_names.items) |name| {
        const dev_name = strip(name);
        if (dev_name.len == 0) return arg.throwStr("invalid device specification");
        const z = try a.dupeSentinel(u8, dev_name, 0);
        const dev = c.ggml_backend_dev_by_name(z.ptr);
        if (dev == null or c.ggml_backend_dev_type(dev) == c.GGML_BACKEND_DEVICE_TYPE_CPU) {
            return arg.throw("invalid device: %s", .{z.ptr});
        }
        try devices.append(a, dev);
    }

    try devices.append(a, null);
    return devices;
}

/// Ports `devices_to_string` (llama-bench.cpp:198 @c1d0e7a00).
fn devicesToString(a: std.mem.Allocator, devices: []const c.ggml_backend_dev_t) ![:0]const u8 {
    if (devices.len == 0) return "auto";
    if (devices.len == 1 and devices[0] == null) return "none";
    var names: std.ArrayList([*:0]const u8) = .empty;
    for (devices) |dev| {
        if (dev == null) break;
        try names.append(a, c.ggml_backend_dev_name(dev));
    }
    return join(a, [*:0]const u8, names.items, "/");
}

/// Ports `output_formats` (llama-bench.cpp:219 @c1d0e7a00).
const OutputFormat = enum { none, csv, json, jsonl, markdown, sql };

/// Ports `output_format_str` (llama-bench.cpp:221 @c1d0e7a00).
fn outputFormatStr(f: OutputFormat) [*:0]const u8 {
    return switch (f) {
        .none => "none",
        .csv => "csv",
        .json => "json",
        .jsonl => "jsonl",
        .markdown => "md",
        .sql => "sql",
    };
}

/// Ports `output_format_from_str` (llama-bench.cpp:240 @c1d0e7a00).
fn outputFormatFromStr(s: []const u8) ?OutputFormat {
    const map = [_]struct { []const u8, OutputFormat }{ .{ "none", .none }, .{ "csv", .csv }, .{ "json", .json }, .{ "jsonl", .jsonl }, .{ "md", .markdown }, .{ "sql", .sql } };
    for (map) |m| {
        if (std.mem.eql(u8, s, m[0])) return m[1];
    }
    return null;
}

/// Ports `split_mode_str` (llama-bench.cpp:259 @c1d0e7a00).
fn splitModeStr(mode: c.llama_split_mode) [*:0]const u8 {
    return switch (mode) {
        c.LLAMA_SPLIT_MODE_NONE => "none",
        c.LLAMA_SPLIT_MODE_LAYER => "layer",
        c.LLAMA_SPLIT_MODE_ROW => "row",
        c.LLAMA_SPLIT_MODE_TENSOR => "tensor",
        else => blk: {
            _ = c.ggml_abort("llama-bench.cpp", 270, "invalid split mode");
            break :blk "";
        },
    };
}

/// `std::stoi` on a run of digits the range pattern matched, which can
/// only fail by being out of range.
fn stoiDigits(a: std.mem.Allocator, s: []const u8) !c_int {
    return arg.stoi(try a.dupeSentinel(u8, s, 0));
}

/// Ports `parse_int_range` (llama-bench.cpp:280 @c1d0e7a00): repeated
/// `regex_search` of `^(-?\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$)` from
/// where the last match ended. Each attempt takes the longest form whose
/// end is followed by `,` or the end, as the backtracking regex does; the
/// operator class includes `|`, which the loop then rejects.
fn parseIntRange(a: std.mem.Allocator, s: []const u8, allow_negative: bool) !std.ArrayList(c_int) {
    var result: std.ArrayList(c_int) = .empty;
    var pos: usize = 0;
    const digitsAt = struct {
        fn f(str: []const u8, at: usize) usize {
            var k = at;
            while (k < str.len and std.ascii.isDigit(str[k])) k += 1;
            return k;
        }
    }.f;
    const termAt = struct {
        fn f(str: []const u8, at: usize) ?usize {
            if (at == str.len) return at;
            if (str[at] == ',') return at + 1;
            return null;
        }
    }.f;
    while (pos <= s.len) {
        // (-?\d+)
        var p = pos;
        if (allow_negative and p < s.len and s[p] == '-') p += 1;
        const d0 = digitsAt(s, p);
        if (d0 == p) break;
        const first_s = s[pos..d0];

        var last_s: ?[]const u8 = null;
        var op: u8 = '+';
        var step_s: ?[]const u8 = null;
        var end: ?usize = null;
        // (?:-(\d+)(?:([\+|\*])(\d+))?)? then (?:,|$)
        if (d0 < s.len and s[d0] == '-') {
            const d1 = digitsAt(s, d0 + 1);
            if (d1 > d0 + 1) {
                if (d1 < s.len and (s[d1] == '+' or s[d1] == '|' or s[d1] == '*')) {
                    const d2 = digitsAt(s, d1 + 1);
                    if (d2 > d1 + 1) {
                        if (termAt(s, d2)) |e| {
                            last_s = s[d0 + 1 .. d1];
                            op = s[d1];
                            step_s = s[d1 + 1 .. d2];
                            end = e;
                        }
                    }
                }
                if (end == null) {
                    if (termAt(s, d1)) |e| {
                        last_s = s[d0 + 1 .. d1];
                        end = e;
                    }
                }
            }
        }
        if (end == null) {
            end = termAt(s, d0) orelse break;
        }

        const first = try stoiDigits(a, first_s);
        const last = if (last_s) |l| try stoiDigits(a, l) else first;
        const step = if (step_s) |st| try stoiDigits(a, st) else 1;

        var i = first;
        while (i <= last) {
            try result.append(a, i);
            const prev_i = i;
            if (op == '+') {
                i +%= step;
            } else if (op == '*') {
                i *%= step;
            } else {
                return arg.throwStr("invalid range format");
            }
            if (i <= prev_i) return arg.throwStr("invalid range");
        }
        pos = end.?;
        if (pos == s.len) break;
    }

    if (pos != s.len) return arg.throwStr("invalid range format");
    return result;
}

/// `string_split` (common/common.h:786 @c1d0e7a00): each token through
/// `std::istringstream >> bool` without `boolalpha` -- a `long`, where 0 is
/// false, 1 is true, any other number true, and no number at all false.
fn splitBool(a: std.mem.Allocator, s: []const u8) !std.ArrayList(bool) {
    var out: std.ArrayList(bool) = .empty;
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |tok| {
        // `std::getline` yields no empty last token after a final ','
        if (it.index == null and tok.len == 0) break;
        var k: usize = 0;
        while (k < tok.len and isspace(tok[k])) k += 1;
        const st = k;
        if (k < tok.len and (tok[k] == '+' or tok[k] == '-')) k += 1;
        const ds = k;
        while (k < tok.len and std.ascii.isDigit(tok[k])) k += 1;
        const v: bool = if (k == ds) false else blk: {
            const n = std.fmt.parseInt(i64, tok[st..k], 10) catch break :blk true;
            break :blk n != 0;
        };
        try out.append(a, v);
    }
    return out;
}

/// `std::stoull`, as libc++ throws.
fn stoull(a: std.mem.Allocator, s: []const u8) !u64 {
    const z = try a.dupeSentinel(u8, s, 0);
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtoull(z.ptr, &end, 10);
    if (std.c._errno().* == @backingInt(std.c.E.RANGE)) return arg.throwStr("stoull: out of range");
    if (end == z.ptr) return arg.throwStr("stoull: no conversion");
    return v;
}

/// `std::sregex_token_iterator(s, std::regex("[;/]+"), -1)`: the text
/// between runs of `;` and `/`, a leading empty token when `s` starts with
/// one, no trailing empty token, and one empty token for an empty `s`.
fn splitSemiSlash(a: std.mem.Allocator, s: []const u8) !std.ArrayList([]const u8) {
    var out: std.ArrayList([]const u8) = .empty;
    if (s.len == 0) {
        try out.append(a, s);
        return out;
    }
    var i: usize = 0;
    var tok_start: usize = 0;
    while (i < s.len) {
        if (s[i] == ';' or s[i] == '/') {
            try out.append(a, s[tok_start..i]);
            while (i < s.len and (s[i] == ';' or s[i] == '/')) i += 1;
            tok_start = i;
        } else {
            i += 1;
        }
    }
    if (tok_start < s.len) try out.append(a, s[tok_start..]);
    return out;
}

/// Ports `cmd_params` (llama-bench.cpp:322 @c1d0e7a00).
const CmdParams = struct {
    model: std.ArrayList([]const u8) = .empty,
    hf_repo: std.ArrayList([]const u8) = .empty,
    hf_file: std.ArrayList([]const u8) = .empty,
    hf_token: []const u8 = "",
    offline: bool = false,
    n_prompt: std.ArrayList(c_int) = .empty,
    n_gen: std.ArrayList(c_int) = .empty,
    n_pg: std.ArrayList([2]c_int) = .empty,
    n_depth: std.ArrayList(c_int) = .empty,
    n_batch: std.ArrayList(c_int) = .empty,
    n_ubatch: std.ArrayList(c_int) = .empty,
    type_k: std.ArrayList(c.ggml_type) = .empty,
    type_v: std.ArrayList(c.ggml_type) = .empty,
    n_threads: std.ArrayList(c_int) = .empty,
    cpu_mask: std.ArrayList([]const u8) = .empty,
    cpu_strict: std.ArrayList(bool) = .empty,
    poll: std.ArrayList(c_int) = .empty,
    n_gpu_layers: std.ArrayList(c_int) = .empty,
    n_cpu_moe: std.ArrayList(c_int) = .empty,
    split_mode: std.ArrayList(c.llama_split_mode) = .empty,
    load_mode: std.ArrayList(c.llama_load_mode) = .empty,
    main_gpu: std.ArrayList(c_int) = .empty,
    no_kv_offload: std.ArrayList(bool) = .empty,
    flash_attn: std.ArrayList(c.llama_flash_attn_type) = .empty,
    devices: std.ArrayList([]const c.ggml_backend_dev_t) = .empty,
    tensor_split: std.ArrayList([]const f32) = .empty,
    tensor_buft_overrides: std.ArrayList([]const Override) = .empty,
    embeddings: std.ArrayList(bool) = .empty,
    no_op_offload: std.ArrayList(bool) = .empty,
    no_host: std.ArrayList(bool) = .empty,
    fit_params_target: std.ArrayList(usize) = .empty,
    fit_params_min_ctx: std.ArrayList(u32) = .empty,
    numa: c.ggml_numa_strategy = c.GGML_NUMA_STRATEGY_DISABLED,
    reps: c_int = 5,
    prio: c.ggml_sched_priority = c.GGML_SCHED_PRIO_NORMAL,
    delay: c_int = 0,
    verbose: bool = false,
    progress: bool = false,
    no_warmup: bool = false,
    output_format: OutputFormat = .markdown,
    output_format_stderr: OutputFormat = .none,
};

/// Ports `cmd_params_defaults` (llama-bench.cpp:366 @c1d0e7a00).
const Defaults = struct {
    model: []const []const u8,
    n_prompt: []const c_int,
    n_gen: []const c_int,
    n_pg: []const [2]c_int,
    n_depth: []const c_int,
    n_batch: []const c_int,
    n_ubatch: []const c_int,
    type_k: []const c.ggml_type,
    type_v: []const c.ggml_type,
    n_threads: []const c_int,
    cpu_mask: []const []const u8,
    cpu_strict: []const bool,
    poll: []const c_int,
    n_gpu_layers: []const c_int,
    n_cpu_moe: []const c_int,
    split_mode: []const c.llama_split_mode,
    load_mode: []const c.llama_load_mode,
    main_gpu: []const c_int,
    no_kv_offload: []const bool,
    flash_attn: []const c.llama_flash_attn_type,
    devices: []const []const c.ggml_backend_dev_t,
    tensor_split: []const []const f32,
    tensor_buft_overrides: []const []const Override,
    embeddings: []const bool,
    no_op_offload: []const bool,
    no_host: []const bool,
    fit_params_target: []const usize,
    fit_params_min_ctx: []const u32,
    reps: c_int = 5,
    prio: c.ggml_sched_priority = c.GGML_SCHED_PRIO_NORMAL,
    delay: c_int = 0,
    output_format: OutputFormat = .markdown,
    output_format_stderr: OutputFormat = .none,

    fn init(a: std.mem.Allocator) !Defaults {
        const n_threads = try a.alloc(c_int, 1);
        n_threads[0] = tc.cpu.numMath();
        const ts = try a.alloc(f32, c.llama_max_devices());
        @memset(ts, 0);
        const tss = try a.alloc([]const f32, 1);
        tss[0] = ts;
        const ot = try a.alloc(Override, 1);
        ot[0] = .{ .pattern = null, .buft = null };
        const ots = try a.alloc([]const Override, 1);
        ots[0] = ot;
        return .{
            .model = &.{"models/7B/ggml-model-q4_0.gguf"},
            .n_prompt = &.{512},
            .n_gen = &.{128},
            .n_pg = &.{},
            .n_depth = &.{0},
            .n_batch = &.{2048},
            .n_ubatch = &.{512},
            .type_k = &.{c.GGML_TYPE_F16},
            .type_v = &.{c.GGML_TYPE_F16},
            .n_threads = n_threads,
            .cpu_mask = &.{"0x0"},
            .cpu_strict = &.{false},
            .poll = &.{50},
            .n_gpu_layers = &.{-1},
            .n_cpu_moe = &.{0},
            .split_mode = &.{c.LLAMA_SPLIT_MODE_LAYER},
            .load_mode = &.{c.LLAMA_LOAD_MODE_AUTO},
            .main_gpu = &.{0},
            .no_kv_offload = &.{false},
            .flash_attn = &.{c.LLAMA_FLASH_ATTN_TYPE_AUTO},
            .devices = &.{&.{}},
            .tensor_split = tss,
            .tensor_buft_overrides = ots,
            .embeddings = &.{false},
            .no_op_offload = &.{false},
            .no_host = &.{false},
            .fit_params_target = &.{0},
            .fit_params_min_ctx = &.{0},
        };
    }
};

var defaults: Defaults = undefined;

fn typeNames(a: std.mem.Allocator, types: []const c.ggml_type) ![:0]u8 {
    var names: std.ArrayList([*:0]const u8) = .empty;
    for (types) |t| try names.append(a, c.ggml_type_name(t));
    return join(a, [*:0]const u8, names.items, ",");
}

/// Ports `print_usage` (llama-bench.cpp:410 @c1d0e7a00).
fn printUsage(a: std.mem.Allocator, argv0: [*:0]const u8) !void {
    const d = &defaults;
    io.printf("usage: %s [options]\n", .{argv0});
    io.printf("\n", .{});
    io.printf("options:\n", .{});
    io.printf("  -h, --help\n", .{});
    io.printf("  --numa <distribute|isolate|numactl>         numa mode (default: disabled)\n", .{});
    io.printf("  -r, --repetitions <n>                       number of times to repeat each test (default: %d)\n", .{d.reps});
    io.printf("  --prio <-1|0|1|2|3>                         process/thread priority (default: %d)\n", .{@as(c_int, @intCast(d.prio))});
    io.printf("  --delay <0...N> (seconds)                   delay between each test (default: %d)\n", .{d.delay});
    io.printf("  -o, --output <csv|json|jsonl|md|sql>        output format printed to stdout (default: %s)\n", .{outputFormatStr(d.output_format)});
    io.printf("  -oe, --output-err <csv|json|jsonl|md|sql>   output format printed to stderr (default: %s)\n", .{outputFormatStr(d.output_format_stderr)});
    io.printf("  --list-devices                              list available devices and exit\n", .{});
    io.printf("  -v, --verbose                               verbose output\n", .{});
    io.printf("  --progress                                  print test progress indicators\n", .{});
    io.printf("  --no-warmup                                 skip warmup runs before benchmarking\n", .{});
    io.printf("  -fitt, --fit-target <MiB>                   fit model to device memory with this margin per device in MiB (default: off)\n", .{});
    io.printf("  -fitc, --fit-ctx <n>                        minimum ctx size for --fit-target (default: 4096)\n", .{});
    if (c.llama_supports_rpc()) {
        io.printf("  -rpc, --rpc <rpc_servers>                   register RPC devices (comma separated)\n", .{});
    }
    io.printf("\n", .{});
    io.printf("test parameters:\n", .{});
    io.printf("  -m, --model <filename>                            (default: %s)\n", .{(try join(a, []const u8, d.model, ",")).ptr});
    io.printf("  -hf, -hfr, --hf-repo <user>/<model>[:quant]       Hugging Face model repository; quant is optional, case-insensitive\n", .{});
    io.printf("                                                    default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\n", .{});
    io.printf("                                                    example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n", .{});
    io.printf("                                                    (default: unused)\n", .{});
    io.printf("  -hff, --hf-file <file>                            Hugging Face model file. If specified, it will override the quant in --hf-repo\n", .{});
    io.printf("                                                    (default: unused)\n", .{});
    io.printf("  -hft, --hf-token <token>                          Hugging Face access token\n", .{});
    io.printf("                                                    (default: value from HF_TOKEN environment variable)\n", .{});
    io.printf("  --offline                                         Offline mode: forces use of cache, prevents network access\n", .{});
    io.printf("                                                    (default: disabled)\n", .{});
    io.printf("  -p, --n-prompt <n>                                (default: %s)\n", .{(try join(a, c_int, d.n_prompt, ",")).ptr});
    io.printf("  -n, --n-gen <n>                                   (default: %s)\n", .{(try join(a, c_int, d.n_gen, ",")).ptr});
    io.printf("  -pg <pp,tg>                                       (default: %s)\n", .{@as([*:0]const u8, "")});
    io.printf("  -d, --n-depth <n>                                 (default: %s)\n", .{(try join(a, c_int, d.n_depth, ",")).ptr});
    io.printf("  -b, --batch-size <n>                              (default: %s)\n", .{(try join(a, c_int, d.n_batch, ",")).ptr});
    io.printf("  -ub, --ubatch-size <n>                            (default: %s)\n", .{(try join(a, c_int, d.n_ubatch, ",")).ptr});
    io.printf("  -ctk, --cache-type-k <t>                          (default: %s)\n", .{(try typeNames(a, d.type_k)).ptr});
    io.printf("  -ctv, --cache-type-v <t>                          (default: %s)\n", .{(try typeNames(a, d.type_v)).ptr});
    io.printf("  -t, --threads <n>                                 (default: %s)\n", .{(try join(a, c_int, d.n_threads, ",")).ptr});
    io.printf("  -C, --cpu-mask <hex,hex>                          (default: %s)\n", .{(try join(a, []const u8, d.cpu_mask, ",")).ptr});
    io.printf("  --cpu-strict <0|1>                                (default: %s)\n", .{(try join(a, bool, d.cpu_strict, ",")).ptr});
    io.printf("  --poll <0...100>                                  (default: %s)\n", .{(try join(a, c_int, d.poll, ",")).ptr});
    io.printf("  -ngl, --n-gpu-layers <n>                          (default: %s)\n", .{(try join(a, c_int, d.n_gpu_layers, ",")).ptr});
    io.printf("  -ncmoe, --n-cpu-moe <n>                           (default: %s)\n", .{(try join(a, c_int, d.n_cpu_moe, ",")).ptr});
    io.printf("  -sm, --split-mode <none|layer|row|tensor>         (default: %s)\n", .{splitModeStr(d.split_mode[0])});
    io.printf("  -mg, --main-gpu <i>                               (default: %s)\n", .{(try join(a, c_int, d.main_gpu, ",")).ptr});
    io.printf("  -nkvo, --no-kv-offload <0|1>                      (default: %s)\n", .{(try join(a, bool, d.no_kv_offload, ",")).ptr});
    io.printf("  -fa, --flash-attn <on|off|auto>                   (default: %s)\n", .{c.llama_flash_attn_type_name(d.flash_attn[0])});
    io.printf("  -dev, --device <dev0/dev1/...>                    (default: auto)\n", .{});
    io.printf("  -lm, --load-mode <auto|none|mmap|mlock|mmap+mlock|dio> (default: %s)\n", .{c.llama_load_mode_name(d.load_mode[0])});
    io.printf("  -mmp, --mmap <0|1>                                (DEPRECATED IN FAVOUR OF --load-mode)\n", .{});
    io.printf("  -dio, --direct-io <0|1>                           (DEPRECATED IN FAVOUR OF --load-mode)\n", .{});
    io.printf("  -embd, --embeddings <0|1>                         (default: %s)\n", .{(try join(a, bool, d.embeddings, ",")).ptr});
    io.printf("  -ts, --tensor-split <ts0/ts1/..>                  (default: 0)\n", .{});
    io.printf("  -ot --override-tensor <tensor name pattern>=<buffer type>;...\n", .{});
    io.printf("                                                    (default: disabled)\n", .{});
    io.printf("  -nopo, --no-op-offload <0|1>                      (default: 0)\n", .{});
    io.printf("  --no-host <0|1>                                   (default: %s)\n", .{(try join(a, bool, d.no_host, ",")).ptr});
    io.printf("\n", .{});
    io.printf("Multiple values can be given for each parameter by separating them with ','\n" ++
        "or by specifying the parameter multiple times. Ranges can be given as\n" ++
        "'first-last' or 'first-last+step' or 'first-last*mult'.\n", .{});
}

/// Ports `ggml_type_from_name` (llama-bench.cpp:478 @c1d0e7a00).
fn typeFromName(s: []const u8) ?c.ggml_type {
    const map = [_]struct { []const u8, c.ggml_type }{
        .{ "f16", c.GGML_TYPE_F16 },   .{ "bf16", c.GGML_TYPE_BF16 },     .{ "q8_0", c.GGML_TYPE_Q8_0 },
        .{ "q4_0", c.GGML_TYPE_Q4_0 }, .{ "q4_1", c.GGML_TYPE_Q4_1 },     .{ "q5_0", c.GGML_TYPE_Q5_0 },
        .{ "q5_1", c.GGML_TYPE_Q5_1 }, .{ "iq4_nl", c.GGML_TYPE_IQ4_NL },
    };
    for (map) |m| {
        if (std.mem.eql(u8, s, m[0])) return m[1];
    }
    return null;
}

/// `exit` from the parser, flushing as the C runtime does.
fn exitWith(code: u8) noreturn {
    io.flushStdout();
    tc.cExit(code);
}

const eql = struct {
    fn f(x: []const u8, ys: []const []const u8) bool {
        for (ys) |y| {
            if (std.mem.eql(u8, x, y)) return true;
        }
        return false;
    }
}.f;
// every option that takes a value
const NextVal = struct {
    fn get(av: []const [:0]const u8, idx: *usize) ?[:0]const u8 {
        idx.* += 1;
        if (idx.* >= av.len) return null;
        return av[idx.*];
    }
};

/// One option of `parse_cmd_params` (llama-bench.cpp:507 @c1d0e7a00):
/// `error.Invalid` where upstream sets `invalid_param` and breaks, and
/// `error.Throw` where a parse throws into the loop's `catch`.
fn parseArg(a: std.mem.Allocator, argv: []const [:0]const u8, i: *usize, cur: []const u8, params: *CmdParams, invalid_param: *bool) !void {
    const split_delim = ',';
    if (eql(cur, &.{ "-h", "--help" })) {
        try printUsage(a, argv[0].ptr);
        exitWith(0);
    } else if (eql(cur, &.{ "-m", "--model" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.model.appendSlice(a, (try tc.strings.split(a, v, split_delim)).items);
    } else if (eql(cur, &.{ "-hf", "-hfr", "--hf-repo" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.hf_repo.appendSlice(a, (try tc.strings.split(a, v, split_delim)).items);
    } else if (eql(cur, &.{ "-hff", "--hf-file" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.hf_file.appendSlice(a, (try tc.strings.split(a, v, split_delim)).items);
    } else if (eql(cur, &.{ "-hft", "--hf-token" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        params.hf_token = v;
    } else if (eql(cur, &.{"--offline"})) {
        params.offline = true;
    } else if (eql(cur, &.{ "-p", "--n-prompt" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.n_prompt.appendSlice(a, (try parseIntRange(a, v, false)).items);
    } else if (eql(cur, &.{ "-n", "--n-gen" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.n_gen.appendSlice(a, (try parseIntRange(a, v, false)).items);
    } else if (eql(cur, &.{"-pg"})) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        const p = try tc.strings.split(a, v, ',');
        if (p.items.len != 2) return error.Invalid;
        const x = try arg.stoi(try a.dupeSentinel(u8, p.items[0], 0));
        const y = try arg.stoi(try a.dupeSentinel(u8, p.items[1], 0));
        try params.n_pg.append(a, .{ x, y });
    } else if (eql(cur, &.{ "-d", "--n-depth" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.n_depth.appendSlice(a, (try parseIntRange(a, v, false)).items);
    } else if (eql(cur, &.{ "-b", "--batch-size" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.n_batch.appendSlice(a, (try parseIntRange(a, v, false)).items);
    } else if (eql(cur, &.{ "-ub", "--ubatch-size" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.n_ubatch.appendSlice(a, (try parseIntRange(a, v, false)).items);
    } else if (eql(cur, &.{ "-ctk", "--cache-type-k", "-ctv", "--cache-type-v" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        var types: std.ArrayList(c.ggml_type) = .empty;
        for ((try tc.strings.split(a, v, split_delim)).items) |t| {
            try types.append(a, typeFromName(t) orelse return error.Invalid);
        }
        const is_k = eql(cur, &.{ "-ctk", "--cache-type-k" });
        try (if (is_k) &params.type_k else &params.type_v).appendSlice(a, types.items);
    } else if (eql(cur, &.{ "-dev", "--device" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        for ((try tc.strings.split(a, v, split_delim)).items) |combo| {
            const devs = parseDevicesArg(a, combo) catch |e| switch (e) {
                error.Throw => {
                    io.eprintf("error: %.*s\n", .{ @as(c_int, @intCast(arg.what().len)), arg.what().ptr });
                    return error.Invalid;
                },
                else => |other| return other,
            };
            try params.devices.append(a, devs.items);
        }
    } else if (eql(cur, &.{"--list-devices"})) {
        try tc.arg_options2.printAvailableDevices(a);
        exitWith(0);
    } else if (eql(cur, &.{ "-t", "--threads" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.n_threads.appendSlice(a, (try parseIntRange(a, v, false)).items);
    } else if (eql(cur, &.{ "-C", "--cpu-mask" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.cpu_mask.appendSlice(a, (try tc.strings.split(a, v, split_delim)).items);
    } else if (eql(cur, &.{"--cpu-strict"})) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.cpu_strict.appendSlice(a, (try splitBool(a, v)).items);
    } else if (eql(cur, &.{"--poll"})) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.poll.appendSlice(a, (try parseIntRange(a, v, false)).items);
    } else if (eql(cur, &.{ "-ngl", "--n-gpu-layers" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.n_gpu_layers.appendSlice(a, (try parseIntRange(a, v, true)).items);
    } else if (eql(cur, &.{ "-ncmoe", "--n-cpu-moe" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.n_cpu_moe.appendSlice(a, (try parseIntRange(a, v, false)).items);
    } else if (c.llama_supports_rpc() and eql(cur, &.{ "-rpc", "--rpc" })) {
        // no RPC backend is built: `llama_supports_rpc` is false
        return error.Invalid;
    } else if (eql(cur, &.{ "-sm", "--split-mode" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        var modes: std.ArrayList(c.llama_split_mode) = .empty;
        for ((try tc.strings.split(a, v, split_delim)).items) |m| {
            const mode: c.llama_split_mode = if (std.mem.eql(u8, m, "none"))
                c.LLAMA_SPLIT_MODE_NONE
            else if (std.mem.eql(u8, m, "layer"))
                c.LLAMA_SPLIT_MODE_LAYER
            else if (std.mem.eql(u8, m, "row"))
                c.LLAMA_SPLIT_MODE_ROW
            else if (std.mem.eql(u8, m, "tensor"))
                c.LLAMA_SPLIT_MODE_TENSOR
            else
                return error.Invalid;
            try modes.append(a, mode);
        }
        try params.split_mode.appendSlice(a, modes.items);
    } else if (eql(cur, &.{ "-lm", "--load-mode" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        var modes: std.ArrayList(c.llama_load_mode) = .empty;
        const names = [_]struct { []const u8, c.llama_load_mode }{
            .{ "auto", c.LLAMA_LOAD_MODE_AUTO },   .{ "none", c.LLAMA_LOAD_MODE_NONE },             .{ "mmap", c.LLAMA_LOAD_MODE_MMAP },
            .{ "mlock", c.LLAMA_LOAD_MODE_MLOCK }, .{ "mmap+mlock", c.LLAMA_LOAD_MODE_MMAP_MLOCK }, .{ "dio", c.LLAMA_LOAD_MODE_DIRECT_IO },
        };
        for ((try tc.strings.split(a, v, split_delim)).items) |m| {
            var found: ?c.llama_load_mode = null;
            for (names) |n| {
                if (std.mem.eql(u8, m, n[0])) found = n[1];
            }
            try modes.append(a, found orelse return error.Invalid);
        }
        try params.load_mode.appendSlice(a, modes.items);
    } else if (eql(cur, &.{ "-mg", "--main-gpu" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        params.main_gpu = try parseIntRange(a, v, false);
    } else if (eql(cur, &.{ "-nkvo", "--no-kv-offload" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.no_kv_offload.appendSlice(a, (try splitBool(a, v)).items);
    } else if (eql(cur, &.{"--numa"})) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        if (std.mem.eql(u8, v, "distribute") or v.len == 0) {
            params.numa = c.GGML_NUMA_STRATEGY_DISTRIBUTE;
        } else if (std.mem.eql(u8, v, "isolate")) {
            params.numa = c.GGML_NUMA_STRATEGY_ISOLATE;
        } else if (std.mem.eql(u8, v, "numactl")) {
            params.numa = c.GGML_NUMA_STRATEGY_NUMACTL;
        } else return error.Invalid;
    } else if (eql(cur, &.{ "-fa", "--flash-attn" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        var types: std.ArrayList(c.llama_flash_attn_type) = .empty;
        for ((try tc.strings.split(a, v, split_delim)).items) |t| {
            const ty: c.llama_flash_attn_type = if (arg.isTruthy(t))
                c.LLAMA_FLASH_ATTN_TYPE_ENABLED
            else if (arg.isFalsey(t))
                c.LLAMA_FLASH_ATTN_TYPE_DISABLED
            else if (arg.isAutoy(t))
                c.LLAMA_FLASH_ATTN_TYPE_AUTO
            else
                return error.Invalid;
            try types.append(a, ty);
        }
        try params.flash_attn.appendSlice(a, types.items);
    } else if (eql(cur, &.{ "-mmp", "--mmap", "-dio", "--direct-io" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        const dio = std.mem.eql(u8, cur, "-dio") or std.mem.eql(u8, cur, "--direct-io");
        if (dio) {
            log.wrn("DEPRECATED: -dio and --direct-io are deprecated in favour of --load-mode. Please use --load-mode dio instead.\n", .{});
        } else {
            log.wrn("DEPRECATED: -mmp and --mmap are deprecated in favour of --load-mode. Please use --load-mode mmap instead.\n", .{});
        }
        for ((try splitBool(a, v)).items) |m| {
            try params.load_mode.append(a, if (m) (if (dio) c.LLAMA_LOAD_MODE_DIRECT_IO else c.LLAMA_LOAD_MODE_MMAP) else c.LLAMA_LOAD_MODE_NONE);
        }
    } else if (eql(cur, &.{ "-embd", "--embeddings" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.embeddings.appendSlice(a, (try splitBool(a, v)).items);
    } else if (eql(cur, &.{ "-nopo", "--no-op-offload" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.no_op_offload.appendSlice(a, (try splitBool(a, v)).items);
    } else if (eql(cur, &.{"--no-host"})) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        try params.no_host.appendSlice(a, (try splitBool(a, v)).items);
    } else if (eql(cur, &.{ "-ts", "--tensor-split" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        for ((try tc.strings.split(a, v, split_delim)).items) |ts| {
            // split string by ; and /
            const split_arg = try splitSemiSlash(a, ts);
            ggmlAssert(split_arg.items.len <= c.llama_max_devices(), 913, "split_arg.size() <= llama_max_devices()");

            const tensor_split = try a.alloc(f32, c.llama_max_devices());
            for (tensor_split, 0..) |*t, k| {
                t.* = if (k < split_arg.items.len) try arg.stof(try a.dupeSentinel(u8, split_arg.items[k], 0)) else 0.0;
            }
            try params.tensor_split.append(a, tensor_split);
        }
    } else if (eql(cur, &.{ "-ot", "--override-tensor" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        if (try parseOverrides(a, v, params)) {
            // upstream breaks out of its group loop only: the
            // argument loop goes on, and the error names the last
            // argument it read
            invalid_param.* = true;
        }
    } else if (eql(cur, &.{ "-r", "--repetitions" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        params.reps = try arg.stoi(v);
    } else if (eql(cur, &.{"--prio"})) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        params.prio = @intCast(try arg.stoi(v));
    } else if (eql(cur, &.{"--delay"})) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        params.delay = try arg.stoi(v);
    } else if (eql(cur, &.{ "-o", "--output" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        if (outputFormatFromStr(v)) |f| {
            params.output_format = f;
            invalid_param.* = false;
        } else invalid_param.* = true;
    } else if (eql(cur, &.{ "-oe", "--output-err" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        if (outputFormatFromStr(v)) |f| {
            params.output_format_stderr = f;
            invalid_param.* = false;
        } else invalid_param.* = true;
    } else if (eql(cur, &.{ "-v", "--verbose" })) {
        params.verbose = true;
    } else if (eql(cur, &.{"--progress"})) {
        params.progress = true;
    } else if (eql(cur, &.{"--no-warmup"})) {
        params.no_warmup = true;
    } else if (eql(cur, &.{ "-fitt", "--fit-target" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        for ((try tc.strings.split(a, v, split_delim)).items) |x| try params.fit_params_target.append(a, try stoull(a, x));
    } else if (eql(cur, &.{ "-fitc", "--fit-ctx" })) {
        const v = NextVal.get(argv, i) orelse return error.Invalid;
        for ((try tc.strings.split(a, v, split_delim)).items) |x| {
            try params.fit_params_min_ctx.append(a, @truncate(try arg.stoul(try a.dupeSentinel(u8, x, 0))));
        }
    } else {
        return error.Invalid;
    }
}

/// Ports `parse_cmd_params` (llama-bench.cpp:507 @c1d0e7a00).
fn parseCmdParams(a: std.mem.Allocator, argv: []const [:0]const u8) !CmdParams {
    var params: CmdParams = .{};
    var argbuf: []const u8 = "";
    var invalid_param = false;

    if (std.c.getenv("HF_TOKEN")) |env| params.hf_token = std.mem.span(env);

    var i: usize = 1;
    outer: while (i < argv.len) : (i += 1) {
        const cur = try a.dupe(u8, argv[i]);
        if (std.mem.startsWith(u8, cur, "--")) std.mem.replaceScalar(u8, cur, '_', '-');
        argbuf = cur;
        parseArg(a, argv, &i, cur, &params, &invalid_param) catch |e| switch (e) {
            error.Invalid => {
                invalid_param = true;
                break :outer;
            },
            error.Throw => {
                io.eprintf("error: %.*s\n", .{ @as(c_int, @intCast(arg.what().len)), arg.what().ptr });
                invalid_param = true;
                break :outer;
            },
            else => |other| return other,
        };
    }

    if (invalid_param) {
        io.eprintf("error: invalid parameter for argument: %.*s\n", .{ @as(c_int, @intCast(argbuf.len)), argbuf.ptr });
        try printUsage(a, argv[0].ptr);
        exitWith(1);
    }

    if (params.hf_repo.items.len != 0) {
        // no downloader is ported: upstream's path when the download fails
        io.eprintf("error: failed to download model from HuggingFace\n", .{});
        exitWith(1);
    }

    // set defaults
    const d = &defaults;
    inline for (.{
        "model",         "n_prompt",   "n_gen",        "n_pg",         "n_depth",               "n_batch",           "n_ubatch",
        "type_k",        "type_v",     "n_gpu_layers", "n_cpu_moe",    "split_mode",            "load_mode",         "main_gpu",
        "no_kv_offload", "flash_attn", "devices",      "tensor_split", "tensor_buft_overrides", "embeddings",        "no_op_offload",
        "no_host",       "n_threads",  "cpu_mask",     "cpu_strict",   "poll",                  "fit_params_target", "fit_params_min_ctx",
    }) |f| {
        if (@field(params, f).items.len == 0) try @field(params, f).appendSlice(a, @field(d, f));
    }

    return params;
}

/// The `-ot` branch of `parse_cmd_params` (llama-bench.cpp:507 @c1d0e7a00).
///
/// Return: whether the value was invalid.
fn parseOverrides(a: std.mem.Allocator, value_in: []const u8, params: *CmdParams) !bool {
    // `std::map<std::string, buft>`: by name, sorted
    const Entry = struct { name: []const u8, buft: c.ggml_backend_buffer_type_t };
    var buft_list: std.ArrayList(Entry) = .empty;
    var k: usize = 0;
    while (k < c.ggml_backend_dev_count()) : (k += 1) {
        const dev = c.ggml_backend_dev_get(k);
        const buft = c.ggml_backend_dev_buffer_type(dev);
        if (buft != null) {
            const name = std.mem.span(c.ggml_backend_buft_name(buft));
            var found = false;
            for (buft_list.items) |*e| {
                if (std.mem.eql(u8, e.name, name)) {
                    e.buft = buft;
                    found = true;
                }
            }
            if (!found) try buft_list.append(a, .{ .name = name, .buft = buft });
        }
    }
    std.mem.sort(Entry, buft_list.items, {}, struct {
        fn lt(_: void, x: Entry, y: Entry) bool {
            return std.mem.order(u8, x.name, y.name) == .lt;
        }
    }.lt);

    const strcspn = struct {
        fn f(s: []const u8, set: []const u8) usize {
            return std.mem.indexOfAny(u8, s, set) orelse s.len;
        }
    }.f;

    var value = value_in;
    var group_span = strcspn(value, ",");
    var last_group = false;
    while (true) {
        if (group_span == 0) {
            // Adds an empty override-tensors for an empty span
            const empty = try a.alloc(Override, 1);
            empty[0] = .{ .pattern = null, .buft = null };
            try params.tensor_buft_overrides.append(a, empty);
            if (group_span == value.len) {
                last_group = true;
            } else {
                value = value[group_span + 1 ..];
                group_span = strcspn(value, ",");
            }
            if (last_group) break;
            continue;
        }
        var group = value[0..group_span];
        if (group_span == value.len) {
            last_group = true;
            value = value[group_span..];
        } else {
            value = value[group_span + 1 ..];
        }
        var overrides: std.ArrayList(Override) = .empty;
        var span = strcspn(group, ";");
        while (span > 0) {
            const ov = group[0..span];
            group = if (span < group.len) group[span + 1 ..] else group[span..];
            const name_span = strcspn(ov, "=");
            if (name_span >= span) return true;
            const tensor_name = try a.dupeSentinel(u8, ov[0..name_span], 0);
            const buffer_type = ov[name_span + 1 ..];
            var buft: ?c.ggml_backend_buffer_type_t = null;
            for (buft_list.items) |e| {
                if (std.mem.eql(u8, e.name, buffer_type)) buft = e.buft;
            }
            if (buft == null) {
                io.printf("error: unrecognized buffer type '%.*s'\n", .{ @as(c_int, @intCast(buffer_type.len)), buffer_type.ptr });
                io.printf("Available buffer types:\n", .{});
                for (buft_list.items) |e| io.printf("  %s\n", .{c.ggml_backend_buft_name(e.buft)});
                return true;
            }
            try overrides.append(a, .{ .pattern = tensor_name.ptr, .buft = buft.? });
            span = strcspn(group, ";");
        }
        try overrides.append(a, .{ .pattern = null, .buft = null });
        try params.tensor_buft_overrides.append(a, overrides.items);
        group_span = strcspn(value, ",");
        if (last_group) break;
    }
    return false;
}

/// Ports `cmd_params_instance` (llama-bench.cpp:1189 @c1d0e7a00).
const Instance = struct {
    model: []const u8,
    n_prompt: c_int,
    n_gen: c_int,
    n_depth: c_int,
    n_batch: c_int,
    n_ubatch: c_int,
    type_k: c.ggml_type,
    type_v: c.ggml_type,
    n_threads: c_int,
    cpu_mask: []const u8,
    cpu_strict: bool,
    poll: c_int,
    n_gpu_layers: c_int,
    n_cpu_moe: c_int,
    split_mode: c.llama_split_mode,
    load_mode: c.llama_load_mode,
    main_gpu: c_int,
    no_kv_offload: bool,
    flash_attn: c.llama_flash_attn_type,
    devices: []const c.ggml_backend_dev_t,
    tensor_split: []const f32,
    tensor_buft_overrides: []const Override,
    embeddings: bool,
    no_op_offload: bool,
    no_host: bool,
    fit_target: usize,
    fit_min_ctx: u32,

    /// The `static` vectors `to_llama_mparams` keeps its merged overrides in.
    var merged: std.ArrayList(Override) = .empty;
    var patterns: std.ArrayList([:0]u8) = .empty;

    /// Ports `to_llama_mparams` (llama-bench.cpp:1218 @c1d0e7a00), a member of `cmd_params_instance`.
    fn toLlamaMparams(self: *const Instance) !c.llama_model_params {
        var mparams = c.llama_model_default_params();

        mparams.n_gpu_layers = self.n_gpu_layers;
        if (self.devices.len != 0) mparams.devices = @constCast(self.devices.ptr);
        mparams.split_mode = self.split_mode;
        mparams.load_mode = self.load_mode;
        mparams.main_gpu = self.main_gpu;
        mparams.tensor_split = self.tensor_split.ptr;
        mparams.no_host = self.no_host;

        if (self.n_cpu_moe <= 0) {
            if (self.tensor_buft_overrides.len == 0) {
                mparams.tensor_buft_overrides = null;
            } else {
                ggmlAssert(self.tensor_buft_overrides[self.tensor_buft_overrides.len - 1].pattern == null, 1235, "tensor_buft_overrides.back().pattern == nullptr && \"Tensor buffer overrides not terminated with empty pattern\"");
                mparams.tensor_buft_overrides = self.tensor_buft_overrides.ptr;
            }
        } else {
            merged.clearRetainingCapacity();
            patterns.clearRetainingCapacity();

            var ots = self.tensor_buft_overrides;
            if (ots.len != 0 and ots[ots.len - 1].pattern == null) ots = ots[0 .. ots.len - 1];
            try merged.appendSlice(gpa, ots);

            var i: c_int = 0;
            while (i < self.n_cpu_moe) : (i += 1) {
                // `llm_ffn_exps_block_regex` (common/common.h:1116 @c1d0e7a00)
                const pat = try std.fmt.allocPrintSentinel(gpa, "blk\\.{d}\\.ffn_(up|down|gate|gate_up)_(ch|)exps", .{i}, 0);
                try patterns.append(gpa, pat);
                try merged.append(gpa, .{ .pattern = pat.ptr, .buft = c.ggml_backend_cpu_buffer_type() });
            }

            try merged.append(gpa, .{ .pattern = null, .buft = null });

            mparams.tensor_buft_overrides = merged.items.ptr;
        }

        return mparams;
    }

    /// Ports `equal_mparams` (llama-bench.cpp:1270 @c1d0e7a00), a member of `cmd_params_instance`.
    fn equalMparams(self: *const Instance, other: *const Instance) bool {
        return std.mem.eql(u8, self.model, other.model) and self.n_gpu_layers == other.n_gpu_layers and self.n_cpu_moe == other.n_cpu_moe and
            self.split_mode == other.split_mode and
            self.main_gpu == other.main_gpu and std.mem.eql(f32, self.tensor_split, other.tensor_split) and
            self.load_mode == other.load_mode and std.mem.eql(c.ggml_backend_dev_t, self.devices, other.devices) and self.no_host == other.no_host and
            overridesEqual(self.tensor_buft_overrides, other.tensor_buft_overrides);
    }

    /// Ports `to_llama_cparams` (llama-bench.cpp:1278 @c1d0e7a00), a member of `cmd_params_instance`.
    fn toLlamaCparams(self: *const Instance) c.llama_context_params {
        var cparams = c.llama_context_default_params();

        cparams.n_ctx = @intCast(self.n_prompt + self.n_gen + self.n_depth);
        cparams.n_batch = @intCast(self.n_batch);
        cparams.n_ubatch = @intCast(self.n_ubatch);
        cparams.type_k = self.type_k;
        cparams.type_v = self.type_v;
        cparams.offload_kqv = !self.no_kv_offload;
        cparams.flash_attn_type = self.flash_attn;
        cparams.embeddings = self.embeddings;
        cparams.op_offload = !self.no_op_offload;
        cparams.swa_full = false;

        return cparams;
    }
};

/// Ports `get_cmd_params_instances` (llama-bench.cpp:1296 @c1d0e7a00):
/// ordered to minimize the number of times each model is reloaded.
fn getInstances(a: std.mem.Allocator, p: *const CmdParams) !std.ArrayList(Instance) {
    var instances: std.ArrayList(Instance) = .empty;
    for (p.model.items) |m| for (p.fit_params_target.items) |fpt| for (p.fit_params_min_ctx.items) |fpc|
        for (p.n_gpu_layers.items) |nl| for (p.n_cpu_moe.items) |ncmoe| for (p.split_mode.items) |sm|
            for (p.load_mode.items) |lm| for (p.main_gpu.items) |mg| for (p.devices.items) |devs|
                for (p.tensor_split.items) |ts| for (p.tensor_buft_overrides.items) |ot| for (p.no_host.items) |noh|
                    for (p.embeddings.items) |embd| for (p.no_op_offload.items) |nopo| for (p.n_batch.items) |nb|
                        for (p.n_ubatch.items) |nub| for (p.type_k.items) |tk| for (p.type_v.items) |tv|
                            for (p.no_kv_offload.items) |nkvo| for (p.flash_attn.items) |fa| for (p.n_threads.items) |nt|
                                for (p.cpu_mask.items) |cm| for (p.cpu_strict.items) |cs| for (p.n_depth.items) |nd|
                                    for (p.poll.items) |pl| {
                                        const base: Instance = .{
                                            .model = m,
                                            .n_prompt = 0,
                                            .n_gen = 0,
                                            .n_depth = nd,
                                            .n_batch = nb,
                                            .n_ubatch = nub,
                                            .type_k = tk,
                                            .type_v = tv,
                                            .n_threads = nt,
                                            .cpu_mask = cm,
                                            .cpu_strict = cs,
                                            .poll = pl,
                                            .n_gpu_layers = nl,
                                            .n_cpu_moe = ncmoe,
                                            .split_mode = sm,
                                            .load_mode = lm,
                                            .main_gpu = mg,
                                            .no_kv_offload = nkvo,
                                            .flash_attn = fa,
                                            .devices = devs,
                                            .tensor_split = ts,
                                            .tensor_buft_overrides = ot,
                                            .embeddings = embd,
                                            .no_op_offload = nopo,
                                            .no_host = noh,
                                            .fit_target = fpt,
                                            .fit_min_ctx = fpc,
                                        };
                                        for (p.n_prompt.items) |n_prompt| {
                                            if (n_prompt == 0) continue;
                                            var inst = base;
                                            inst.n_prompt = n_prompt;
                                            try instances.append(a, inst);
                                        }
                                        for (p.n_gen.items) |n_gen| {
                                            if (n_gen == 0) continue;
                                            var inst = base;
                                            inst.n_gen = n_gen;
                                            try instances.append(a, inst);
                                        }
                                        for (p.n_pg.items) |n_pg| {
                                            if (n_pg[0] == 0 and n_pg[1] == 0) continue;
                                            var inst = base;
                                            inst.n_prompt = n_pg[0];
                                            inst.n_gen = n_pg[1];
                                            try instances.append(a, inst);
                                        }
                                    };
    return instances;
}

/// Ports `test` (llama-bench.cpp:1439 @c1d0e7a00).
const Test = struct {
    cpu_info: [:0]const u8,
    gpu_info: [:0]const u8,
    inst: Instance,
    model_type: [:0]const u8,
    model_size: u64,
    model_n_params: u64,
    test_time: [:0]const u8,
    samples_ns: std.ArrayList(u64) = .empty,

    fn init(a: std.mem.Allocator, inst: *const Instance, lmodel: ?*const c.llama_model) !Test {
        var buf: [128]u8 = undefined;
        _ = c.llama_model_desc(lmodel, &buf, buf.len);
        const model_type = try a.dupeSentinel(u8, std.mem.sliceTo(&buf, 0), 0);
        // RFC 3339 date-time format
        const t = time(null);
        const n = strftime(&buf, buf.len, "%FT%TZ", gmtime(&t));
        return .{
            .cpu_info = try devInfo(a, false),
            .gpu_info = try devInfo(a, true),
            .inst = inst.*,
            .model_type = model_type,
            .model_size = c.llama_model_size(lmodel),
            .model_n_params = c.llama_model_n_params(lmodel),
            .test_time = try a.dupeSentinel(u8, buf[0..n], 0),
        };
    }

    fn avgNs(self: *const Test) u64 {
        return avgU64(self.samples_ns.items);
    }

    fn stdevNs(self: *const Test) u64 {
        return stdevU64(self.samples_ns.items);
    }

    /// Ports `get_ts` (llama-bench.cpp:1525 @c1d0e7a00), a member of `test`.
    fn getTs(self: *const Test, a: std.mem.Allocator) ![]f64 {
        const n_tokens: f64 = @floatFromInt(self.inst.n_prompt + self.inst.n_gen);
        const ts = try a.alloc(f64, self.samples_ns.items.len);
        for (self.samples_ns.items, 0..) |t, i| ts[i] = 1e9 * n_tokens / @as(f64, @floatFromInt(t));
        return ts;
    }

    fn avgTs(self: *const Test, a: std.mem.Allocator) !f64 {
        return avgF64(try self.getTs(a));
    }

    fn stdevTs(self: *const Test, a: std.mem.Allocator) !f64 {
        return stdevF64(try self.getTs(a));
    }

    /// Ports `get_backend` (llama-bench.cpp:1537 @c1d0e7a00), a member of `test`.
    fn getBackend(a: std.mem.Allocator) ![:0]const u8 {
        var backends: std.ArrayList([*:0]const u8) = .empty;
        var rpc_used = false;
        var i: usize = 0;
        while (i < c.ggml_backend_reg_count()) : (i += 1) {
            const reg = c.ggml_backend_reg_get(i);
            const name = std.mem.span(c.ggml_backend_reg_name(reg));
            if (std.mem.startsWith(u8, name, "RPC")) {
                if (c.ggml_backend_reg_dev_count(reg) > 0) rpc_used = true;
            } else if (!std.mem.eql(u8, name, "CPU")) {
                try backends.append(a, c.ggml_backend_reg_name(reg));
            }
        }
        if (rpc_used) try backends.append(a, "RPC");
        return if (backends.items.len == 0) "CPU" else join(a, [*:0]const u8, backends.items, ",");
    }

    /// Ports `get_fields` (llama-bench.cpp:1559 @c1d0e7a00), a member of `test`.
    const fields = [_][]const u8{
        "build_commit",          "build_number",  "cpu_info",     "gpu_info",       "backends",
        "model_filename",        "model_type",    "model_size",   "model_n_params", "n_batch",
        "n_ubatch",              "n_threads",     "cpu_mask",     "cpu_strict",     "poll",
        "type_k",                "type_v",        "n_gpu_layers", "n_cpu_moe",      "split_mode",
        "main_gpu",              "no_kv_offload", "flash_attn",   "devices",        "tensor_split",
        "tensor_buft_overrides", "load_mode",     "embeddings",   "no_op_offload",  "no_host",
        "fit_target",            "fit_min_ctx",   "n_prompt",     "n_gen",          "n_depth",
        "test_time",             "avg_ns",        "stddev_ns",    "avg_ts",         "stddev_ts",
    };

    const FieldType = enum { string, boolean, int, float };

    /// Ports `get_field_type` (llama-bench.cpp:1576 @c1d0e7a00), a member of `test`.
    fn getFieldType(field: []const u8) FieldType {
        const ints = [_][]const u8{ "build_number", "n_batch", "n_ubatch", "n_threads", "poll", "model_size", "model_n_params", "n_gpu_layers", "main_gpu", "n_prompt", "n_gen", "n_depth", "avg_ns", "stddev_ns", "no_op_offload", "n_cpu_moe", "fit_target", "fit_min_ctx", "flash_attn" };
        for (ints) |f| {
            if (std.mem.eql(u8, field, f)) return .int;
        }
        const bools = [_][]const u8{ "f16_kv", "no_kv_offload", "cpu_strict", "embeddings", "no_host" };
        for (bools) |f| {
            if (std.mem.eql(u8, field, f)) return .boolean;
        }
        if (std.mem.eql(u8, field, "avg_ts") or std.mem.eql(u8, field, "stddev_ts")) return .float;
        return .string;
    }

    /// Ports `get_values` (llama-bench.cpp:1597 @c1d0e7a00), a member of `test`.
    fn getValues(self: *const Test, a: std.mem.Allocator) ![]const []const u8 {
        const in = &self.inst;
        var tensor_split_str: std.ArrayList(u8) = .empty;
        var tensor_buft_overrides_str: std.ArrayList(u8) = .empty;
        var max_nonzero: usize = 0;
        for (0..c.llama_max_devices()) |i| {
            if (in.tensor_split[i] > 0) max_nonzero = i;
        }
        for (0..max_nonzero + 1) |i| {
            var buf: [32]u8 = undefined;
            const n = snprintf(&buf, buf.len, "%.2f", @as(f64, in.tensor_split[i]));
            try tensor_split_str.appendSlice(a, buf[0..@intCast(n)]);
            if (i < max_nonzero) try tensor_split_str.append(a, '/');
        }
        const ots = in.tensor_buft_overrides;
        if (ots.len == 1) {
            // Last element of tensor_buft_overrides is always a null pattern
            // so if it is only one element long, it must be a null pattern.
            ggmlAssert(ots[0].pattern == null, 1617, "tensor_buft_overrides[0].pattern == nullptr");
            try tensor_buft_overrides_str.appendSlice(a, "none");
        } else {
            for (0..ots.len -% 1) |i| {
                // Last element of tensor_buft_overrides is always a null pattern
                if (ots[i].pattern == null) {
                    try tensor_buft_overrides_str.appendSlice(a, "none");
                } else {
                    try tensor_buft_overrides_str.appendSlice(a, std.mem.span(ots[i].pattern));
                    try tensor_buft_overrides_str.append(a, '=');
                    try tensor_buft_overrides_str.appendSlice(a, std.mem.span(c.ggml_backend_buft_name(ots[i].buft)));
                }
                if (i + 2 < ots.len) try tensor_buft_overrides_str.append(a, ';');
            }
        }
        const S = struct {
            fn d(al: std.mem.Allocator, v: anytype) ![]const u8 {
                return std.fmt.allocPrint(al, "{d}", .{v});
            }
            fn b(v: bool) []const u8 {
                return if (v) "1" else "0";
            }
            fn f(al: std.mem.Allocator, v: f64) ![]const u8 {
                var buf: [64]u8 = undefined;
                const n = snprintf(&buf, buf.len, "%f", v);
                return al.dupe(u8, buf[0..@intCast(n)]);
            }
        };
        const values = try a.alloc([]const u8, fields.len);
        const vals = [_][]const u8{
            tc.build_info.commit,
            try S.d(a, tc.build_info.build_number),
            self.cpu_info,
            self.gpu_info,
            try getBackend(a),
            in.model,
            self.model_type,
            try S.d(a, self.model_size),
            try S.d(a, self.model_n_params),
            try S.d(a, in.n_batch),
            try S.d(a, in.n_ubatch),
            try S.d(a, in.n_threads),
            in.cpu_mask,
            S.b(in.cpu_strict),
            try S.d(a, in.poll),
            std.mem.span(c.ggml_type_name(in.type_k)),
            std.mem.span(c.ggml_type_name(in.type_v)),
            try S.d(a, in.n_gpu_layers),
            try S.d(a, in.n_cpu_moe),
            std.mem.span(splitModeStr(in.split_mode)),
            try S.d(a, in.main_gpu),
            S.b(in.no_kv_offload),
            try S.d(a, @as(c_int, @intCast(in.flash_attn))),
            try devicesToString(a, in.devices),
            tensor_split_str.items,
            tensor_buft_overrides_str.items,
            std.mem.span(c.llama_load_mode_name(in.load_mode)),
            S.b(in.embeddings),
            S.b(in.no_op_offload),
            S.b(in.no_host),
            try S.d(a, in.fit_target),
            try S.d(a, in.fit_min_ctx),
            try S.d(a, in.n_prompt),
            try S.d(a, in.n_gen),
            try S.d(a, in.n_depth),
            self.test_time,
            try S.d(a, self.avgNs()),
            try S.d(a, self.stdevNs()),
            try S.f(a, try self.avgTs(a)),
            try S.f(a, try self.stdevTs(a)),
        };
        @memcpy(values, &vals);
        return values;
    }

    /// `get_map` (llama-bench.cpp:1677 @c1d0e7a00), a member of `test`, as a lookup.
    fn valueOf(values: []const []const u8, field: []const u8) ?[]const u8 {
        for (fields, 0..) |f, i| {
            if (std.mem.eql(u8, f, field)) return values[i];
        }
        return null;
    }
};

/// Ports `escape_json` (llama-bench.cpp:1728 @c1d0e7a00). `char` is signed
/// on the targets built here but aarch64 Linux, so a byte past 0x7f is
/// `<= 0x1f` there and prints sign-extended, cut to the buffer.
fn escapeJson(a: std.mem.Allocator, value: []const u8) ![]const u8 {
    var escaped: std.ArrayList(u8) = .empty;
    for (value) |byte| {
        const ch: c_int = @as(c_char, @bitCast(byte));
        if (byte == '"') {
            try escaped.appendSlice(a, "\\\"");
        } else if (byte == '\\') {
            try escaped.appendSlice(a, "\\\\");
        } else if (ch <= 0x1f) {
            var buf: [8]u8 = undefined;
            const n = snprintf(&buf, buf.len, "\\u%04x", ch);
            try escaped.appendSlice(a, buf[0..@min(@as(usize, @intCast(n)), buf.len - 1)]);
        } else {
            try escaped.append(a, byte);
        }
    }
    return escaped.items;
}

/// Ports `format_json_value` (llama-bench.cpp:1746 @c1d0e7a00).
fn formatJsonValue(a: std.mem.Allocator, field: []const u8, value: []const u8) ![]const u8 {
    return switch (Test.getFieldType(field)) {
        .string => std.mem.concat(a, u8, &.{ "\"", try escapeJson(a, value), "\"" }),
        .boolean => if (std.mem.eql(u8, value, "0")) "false" else "true",
        else => value,
    };
}

/// Ports `get_field_width` (llama-bench.cpp:1811 @c1d0e7a00), a member of `markdown_printer`.
fn fieldWidth(field: []const u8) c_int {
    const widths = [_]struct { []const u8, c_int }{
        .{ "model", -30 },       .{ "t/s", 20 },       .{ "size", 10 },      .{ "params", 10 },   .{ "n_gpu_layers", 3 },
        .{ "n_threads", 7 },     .{ "n_batch", 7 },    .{ "n_ubatch", 8 },   .{ "type_k", 6 },    .{ "type_v", 6 },
        .{ "split_mode", 6 },    .{ "load_mode", 10 }, .{ "flash_attn", 3 }, .{ "devices", -12 }, .{ "test", 15 },
        .{ "no_op_offload", 4 }, .{ "no_host", 4 },
    };
    for (widths) |w| {
        if (std.mem.eql(u8, field, w[0])) return w[1];
    }
    const width: c_int = @max(@as(c_int, @intCast(field.len)), 10);
    if (Test.getFieldType(field) == .string) return -width;
    return width;
}

/// Ports `get_field_display_name` (llama-bench.cpp:1866 @c1d0e7a00), a member of `markdown_printer`.
fn fieldDisplayName(field: []const u8) []const u8 {
    const names = [_]struct { []const u8, []const u8 }{
        .{ "n_gpu_layers", "ngl" }, .{ "split_mode", "sm" },    .{ "n_threads", "threads" }, .{ "no_kv_offload", "nkvo" },
        .{ "flash_attn", "fa" },    .{ "load_mode", "lm" },     .{ "embeddings", "embd" },   .{ "no_op_offload", "nopo" },
        .{ "no_host", "noh" },      .{ "devices", "dev" },      .{ "tensor_split", "ts" },   .{ "tensor_buft_overrides", "ot" },
        .{ "fit_target", "fitt" },  .{ "fit_min_ctx", "fitc" },
    };
    for (names) |n| {
        if (std.mem.eql(u8, field, n[0])) return n[1];
    }
    return field;
}

/// Ports `printer` (llama-bench.cpp:1690 @c1d0e7a00) and its five subclasses.
const Printer = struct {
    format: OutputFormat,
    fout: *anyopaque,
    a: std.mem.Allocator,
    /// `json_printer::first`.
    first: bool = true,
    /// `markdown_printer::fields`.
    md_fields: std.ArrayList([]const u8) = .empty,

    fn printHeader(self: *Printer, params: *const CmdParams) !void {
        const f = self.fout;
        const a = self.a;
        switch (self.format) {
            .none => {},
            .csv => fput(f, "%s\n", .{(try join(a, []const u8, &Test.fields, ",")).ptr}),
            .json => fput(f, "[\n", .{}),
            .jsonl => {},
            .sql => {
                fput(f, "CREATE TABLE IF NOT EXISTS llama_bench (\n", .{});
                for (Test.fields, 0..) |field, i| {
                    const ty: [*:0]const u8 = switch (Test.getFieldType(field)) {
                        .string => "TEXT",
                        .boolean, .int => "INTEGER",
                        .float => "REAL",
                    };
                    fput(f, "  %.*s %s%s\n", .{ @as(c_int, @intCast(field.len)), field.ptr, ty, @as([*:0]const u8, if (i < Test.fields.len - 1) "," else "") });
                }
                fput(f, ");\n", .{});
                fput(f, "\n", .{});
            },
            .markdown => try self.markdownHeader(params),
        }
    }

    /// Ports `print_header` (llama-bench.cpp:1912 @c1d0e7a00), a member of `markdown_printer`.
    fn markdownHeader(self: *Printer, params: *const CmdParams) !void {
        const a = self.a;
        const d = &defaults;
        const fl = &self.md_fields;
        // select fields to print
        try fl.appendSlice(a, &.{ "model", "size", "params", "backend" });
        const backend = try Test.getBackend(a);
        const is_cpu_backend = std.mem.indexOf(u8, backend, "CPU") != null or
            std.mem.indexOf(u8, backend, "BLAS") != null or
            std.mem.indexOf(u8, backend, "ZenDNN") != null;
        if (!is_cpu_backend) try fl.append(a, "n_gpu_layers");
        const differs = struct {
            fn f(comptime T: type, v: []const T, def: []const T) bool {
                return v.len > 1 or !std.mem.eql(T, v, def);
            }
        }.f;
        const differsStr = struct {
            fn f(v: []const []const u8, def: []const []const u8) bool {
                if (v.len > 1 or v.len != def.len) return true;
                for (v, def) |x, y| {
                    if (!std.mem.eql(u8, x, y)) return true;
                }
                return false;
            }
        }.f;
        const differsNested = struct {
            fn f(comptime T: type, v: []const []const T, def: []const []const T) bool {
                if (v.len > 1 or v.len != def.len) return true;
                for (v, def) |x, y| {
                    if (!std.mem.eql(T, x, y)) return true;
                }
                return false;
            }
        }.f;
        if (differs(c_int, params.n_cpu_moe.items, d.n_cpu_moe)) try fl.append(a, "n_cpu_moe");
        if (differs(c_int, params.n_threads.items, d.n_threads) or is_cpu_backend) try fl.append(a, "n_threads");
        if (differsStr(params.cpu_mask.items, d.cpu_mask)) try fl.append(a, "cpu_mask");
        if (differs(bool, params.cpu_strict.items, d.cpu_strict)) try fl.append(a, "cpu_strict");
        if (differs(c_int, params.poll.items, d.poll)) try fl.append(a, "poll");
        if (differs(c_int, params.n_batch.items, d.n_batch)) try fl.append(a, "n_batch");
        if (differs(c_int, params.n_ubatch.items, d.n_ubatch)) try fl.append(a, "n_ubatch");
        if (differs(c.ggml_type, params.type_k.items, d.type_k)) try fl.append(a, "type_k");
        if (differs(c.ggml_type, params.type_v.items, d.type_v)) try fl.append(a, "type_v");
        if (differs(c_int, params.main_gpu.items, d.main_gpu)) try fl.append(a, "main_gpu");
        if (differs(c.llama_split_mode, params.split_mode.items, d.split_mode)) try fl.append(a, "split_mode");
        if (differs(bool, params.no_kv_offload.items, d.no_kv_offload)) try fl.append(a, "no_kv_offload");
        if (differs(c.llama_flash_attn_type, params.flash_attn.items, d.flash_attn)) try fl.append(a, "flash_attn");
        if (differsNested(c.ggml_backend_dev_t, params.devices.items, d.devices)) try fl.append(a, "devices");
        if (differsNested(f32, params.tensor_split.items, d.tensor_split)) try fl.append(a, "tensor_split");
        if (params.tensor_buft_overrides.items.len > 1 or !overrideListsEqual(params.tensor_buft_overrides.items, d.tensor_buft_overrides)) try fl.append(a, "tensor_buft_overrides");
        if (differs(c.llama_load_mode, params.load_mode.items, d.load_mode)) try fl.append(a, "load_mode");
        if (differs(bool, params.embeddings.items, d.embeddings)) try fl.append(a, "embeddings");
        if (differs(bool, params.no_op_offload.items, d.no_op_offload)) try fl.append(a, "no_op_offload");
        if (differs(bool, params.no_host.items, d.no_host)) try fl.append(a, "no_host");
        if (differs(usize, params.fit_params_target.items, d.fit_params_target)) try fl.append(a, "fit_target");
        if (differs(u32, params.fit_params_min_ctx.items, d.fit_params_min_ctx)) try fl.append(a, "fit_min_ctx");
        try fl.appendSlice(a, &.{ "test", "t/s" });

        const f = self.fout;
        fput(f, "|", .{});
        for (fl.items) |field| {
            const name = fieldDisplayName(field);
            fput(f, " %*.*s |", .{ fieldWidth(field), @as(c_int, @intCast(name.len)), name.ptr });
        }
        fput(f, "\n", .{});
        fput(f, "|", .{});
        for (fl.items) |field| {
            const width = fieldWidth(field);
            const dashes = try a.alloc(u8, @intCast(@abs(width) - 1));
            @memset(dashes, '-');
            fput(f, " %.*s%s |", .{ @as(c_int, @intCast(dashes.len)), dashes.ptr, @as([*:0]const u8, if (width > 0) ":" else "-") });
        }
        fput(f, "\n", .{});
    }

    fn printTest(self: *Printer, t: *const Test) !void {
        const f = self.fout;
        const a = self.a;
        switch (self.format) {
            .none => {},
            .csv => {
                const values = try t.getValues(a);
                var escaped = try a.alloc([]const u8, values.len);
                for (values, 0..) |v, i| {
                    // `escape_csv` (llama-bench.cpp:1703 @c1d0e7a00), a member of `csv_printer`
                    var e: std.ArrayList(u8) = .empty;
                    try e.append(a, '"');
                    for (v) |ch| {
                        if (ch == '"') try e.append(a, '"');
                        try e.append(a, ch);
                    }
                    try e.append(a, '"');
                    escaped[i] = e.items;
                }
                const line = try join(a, []const u8, escaped, ",");
                fput(f, "%s\n", .{line.ptr});
            },
            .json => {
                if (self.first) self.first = false else fput(f, ",\n", .{});
                fput(f, "  {\n", .{});
                const values = try t.getValues(a);
                for (Test.fields, values) |field, v| {
                    const jv = try formatJsonValue(a, field, v);
                    fput(f, "    \"%.*s\": %.*s,\n", .{ @as(c_int, @intCast(field.len)), field.ptr, @as(c_int, @intCast(jv.len)), jv.ptr });
                }
                fput(f, "    \"samples_ns\": [ %s ],\n", .{(try join(a, u64, t.samples_ns.items, ", ")).ptr});
                fput(f, "    \"samples_ts\": [ %s ]\n", .{(try join(a, f64, try t.getTs(a), ", ")).ptr});
                fput(f, "  }", .{});
                _ = fflush(f);
            },
            .jsonl => {
                fput(f, "{", .{});
                const values = try t.getValues(a);
                for (Test.fields, values) |field, v| {
                    const jv = try formatJsonValue(a, field, v);
                    fput(f, "\"%.*s\": %.*s, ", .{ @as(c_int, @intCast(field.len)), field.ptr, @as(c_int, @intCast(jv.len)), jv.ptr });
                }
                fput(f, "\"samples_ns\": [ %s ],", .{(try join(a, u64, t.samples_ns.items, ", ")).ptr});
                fput(f, "\"samples_ts\": [ %s ]", .{(try join(a, f64, try t.getTs(a), ", ")).ptr});
                fput(f, "}\n", .{});
                _ = fflush(f);
            },
            .sql => {
                fput(f, "INSERT INTO llama_bench (%s) ", .{(try join(a, []const u8, &Test.fields, ", ")).ptr});
                fput(f, "VALUES (", .{});
                const values = try t.getValues(a);
                for (values, 0..) |v, i| {
                    fput(f, "'%.*s'%s", .{ @as(c_int, @intCast(v.len)), v.ptr, @as([*:0]const u8, if (i < values.len - 1) ", " else "") });
                }
                fput(f, ");\n", .{});
            },
            .markdown => try self.markdownTest(t),
        }
    }

    /// Ports `print_test` (llama-bench.cpp:2006 @c1d0e7a00), a member of `markdown_printer`.
    fn markdownTest(self: *Printer, t: *const Test) !void {
        const a = self.a;
        const f = self.fout;
        const values = try t.getValues(a);
        fput(f, "|", .{});
        for (self.md_fields.items) |field| {
            var buf: [128]u8 = undefined;
            var value: []const u8 = "";
            if (std.mem.eql(u8, field, "model")) {
                value = t.model_type;
            } else if (std.mem.eql(u8, field, "size")) {
                const n = if (t.model_size < 1024 * 1024 * 1024)
                    snprintf(&buf, buf.len, "%.2f MiB", @as(f64, @floatFromInt(t.model_size)) / 1024.0 / 1024.0)
                else
                    snprintf(&buf, buf.len, "%.2f GiB", @as(f64, @floatFromInt(t.model_size)) / 1024.0 / 1024.0 / 1024.0);
                value = try a.dupe(u8, buf[0..@intCast(n)]);
            } else if (std.mem.eql(u8, field, "params")) {
                const n = if (t.model_n_params < 1000 * 1000 * 1000)
                    snprintf(&buf, buf.len, "%.2f M", @as(f64, @floatFromInt(t.model_n_params)) / 1e6)
                else
                    snprintf(&buf, buf.len, "%.2f B", @as(f64, @floatFromInt(t.model_n_params)) / 1e9);
                value = try a.dupe(u8, buf[0..@intCast(n)]);
            } else if (std.mem.eql(u8, field, "backend")) {
                value = try Test.getBackend(a);
            } else if (std.mem.eql(u8, field, "test")) {
                const in = &t.inst;
                var n = if (in.n_prompt > 0 and in.n_gen == 0)
                    snprintf(&buf, buf.len, "pp%d", in.n_prompt)
                else if (in.n_gen > 0 and in.n_prompt == 0)
                    snprintf(&buf, buf.len, "tg%d", in.n_gen)
                else
                    snprintf(&buf, buf.len, "pp%d+tg%d", in.n_prompt, in.n_gen);
                if (in.n_depth > 0) {
                    const len: usize = @intCast(n);
                    n += snprintf(buf[len..].ptr, buf.len - len, " @ d%d", in.n_depth);
                }
                value = try a.dupe(u8, buf[0..@intCast(n)]);
            } else if (std.mem.eql(u8, field, "t/s")) {
                const n = snprintf(&buf, buf.len, "%.2f ± %.2f", try t.avgTs(a), try t.stdevTs(a));
                value = try a.dupe(u8, buf[0..@intCast(n)]);
            } else if (Test.valueOf(values, field)) |v| {
                value = v;
            } else {
                exitWith(1);
            }

            var width = fieldWidth(field);
            if (std.mem.eql(u8, field, "t/s")) {
                // HACK: the utf-8 character is 2 bytes
                width += 1;
            }
            fput(f, " %*.*s |", .{ width, @as(c_int, @intCast(value.len)), value.ptr });
        }
        fput(f, "\n", .{});
    }

    fn printFooter(self: *Printer) void {
        switch (self.format) {
            .json => fput(self.fout, "\n]\n", .{}),
            .markdown => fput(self.fout, "\nbuild: %.*s (%d)\n", .{ @as(c_int, @intCast(tc.build_info.commit.len)), tc.build_info.commit.ptr, @as(c_int, tc.build_info.build_number) }),
            else => {},
        }
    }
};

/// Ports `ctx_state` (llama-bench.cpp:2108 @c1d0e7a00).
const CtxState = struct {
    depth: c_int = 0, // in tokens
    buf: std.ArrayList(u8) = .empty, // the llama_context state buffer
};

/// Ports `test_prompt` (llama-bench.cpp:2114 @c1d0e7a00).
fn testPrompt(ctx: ?*c.llama_context, n_prompt: c_int, n_batch: c_int, n_threads: c_int) !bool {
    c.llama_set_n_threads(ctx, n_threads, n_threads);

    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);
    const n_vocab = c.llama_vocab_n_tokens(vocab);

    const tokens = try gpa.alloc(c.llama_token, @intCast(n_batch));
    defer gpa.free(tokens);
    @memset(tokens, 0);

    var n_processed: c_int = 0;

    while (n_processed < n_prompt) {
        const n_tokens = @min(n_prompt - n_processed, n_batch);
        tokens[0] = if (n_processed == 0 and c.llama_vocab_get_add_bos(vocab)) c.llama_vocab_bos(vocab) else @rem(rand(), n_vocab);
        var i: usize = 1;
        while (i < n_tokens) : (i += 1) tokens[i] = @rem(rand(), n_vocab);
        const res = c.llama_decode(ctx, c.llama_batch_get_one(tokens.ptr, n_tokens));
        if (res != 0) {
            io.eprintf("%s: failed to decode prompt batch, res = %d\n", .{ "test_prompt", res });
            return false;
        }
        n_processed += n_tokens;
    }

    c.llama_synchronize(ctx);
    return true;
}

/// Ports `test_gen` (llama-bench.cpp:2143 @c1d0e7a00).
fn testGen(ctx: ?*c.llama_context, n_gen: c_int, n_threads: c_int) bool {
    c.llama_set_n_threads(ctx, n_threads, n_threads);

    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);
    const n_vocab = c.llama_vocab_n_tokens(vocab);

    var token: c.llama_token = if (c.llama_vocab_get_add_bos(vocab)) c.llama_vocab_bos(vocab) else @rem(rand(), n_vocab);

    var i: c_int = 0;
    while (i < n_gen) : (i += 1) {
        const res = c.llama_decode(ctx, c.llama_batch_get_one(&token, 1));
        if (res != 0) {
            io.eprintf("%s: failed to decode generation batch, res = %d\n", .{ "test_gen", res });
            return false;
        }
        c.llama_synchronize(ctx);
        token = @rem(rand(), n_vocab);
    }
    return true;
}

/// Ports `llama_null_log_callback` (llama-bench.cpp:2164 @c1d0e7a00).
fn nullLogCallback(_: c.ggml_log_level, _: [*c]const u8, _: ?*anyopaque) callconv(.c) void {}

/// The cleanup every failed test step does before `exit(1)`.
fn failTest(comptime msg: [*:0]const u8, ctx: ?*c.llama_context, lmodel: ?*c.llama_model) noreturn {
    io.eprintf(msg, .{"llama_bench"});
    c.llama_free(ctx);
    c.llama_model_free(lmodel);
    exitWith(1);
}

/// Ports `llama_bench` (llama-bench.cpp:2191 @c1d0e7a00).
fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    const func = "llama_bench";
    _ = setlocale(4, "C"); // LC_NUMERIC
    // try to set locale for unicode characters in markdown
    _ = setlocale(2, ".UTF-8"); // LC_CTYPE

    // the C's `NDEBUG` and `__OPTIMIZE__`, as Zig sets them for C
    if (builtin.mode == .debug or builtin.mode == .safe) {
        io.eprintf("warning: asserts enabled, performance may be affected\n", .{});
    }
    if (builtin.mode == .debug) {
        io.eprintf("warning: debug build, performance may be affected\n", .{});
    }

    // initialize backends
    c.ggml_backend_load_all();

    defaults = try Defaults.init(a);
    const params = try parseCmdParams(a, argv);

    const cpu_dev = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == null) {
        io.eprintf("%s: error: CPU backend is not loaded\n", .{func});
        return 1;
    }

    // initialize llama.cpp
    if (!params.verbose) c.llama_log_set(nullLogCallback, null);
    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    if (!tc.init.setProcessPriority(params.prio)) {
        io.eprintf("%s: error: failed to set process priority\n", .{func});
        return 1;
    }

    // initialize printer
    var p: ?Printer = if (params.output_format == .none) null else .{ .format = params.output_format, .fout = io.cStdout(), .a = a };
    var p_err: ?Printer = if (params.output_format_stderr == .none) null else .{ .format = params.output_format_stderr, .fout = io.cStderr(), .a = a };

    if (p) |*pr| try pr.printHeader(&params);
    if (p_err) |*pr| try pr.printHeader(&params);

    const instances = try getInstances(a, &params);

    var lmodel: ?*c.llama_model = null;
    var prev_inst: ?*const Instance = null;

    // store the llama_context state at the previous depth that we performed a test
    var cstate: CtxState = .{};

    var params_idx: c_int = 0;
    const params_count = instances.items.len;
    for (instances.items) |*inst| {
        params_idx += 1;
        if (params.progress) io.eprintf("llama-bench: benchmark %d/%zu: starting\n", .{ params_idx, params_count });
        var mparams = try inst.toLlamaMparams();
        var cparams = inst.toLlamaCparams();

        const do_fit = inst.fit_target != defaults.fit_params_target[0] or inst.fit_min_ctx != defaults.fit_params_min_ctx[0];

        const fit_tensor_split = try a.alloc(f32, c.llama_max_devices());
        @memset(fit_tensor_split, 0);
        const fit_overrides = try a.alloc(Override, c.llama_max_tensor_buft_overrides());
        @memset(fit_overrides, .{ .pattern = null, .buft = null });

        if (do_fit) {
            // free the previous model so fit sees full free VRAM
            if (lmodel) |m| {
                c.llama_model_free(m);
                lmodel = null;
                prev_inst = null;
            }

            // use default n_gpu_layers and n_ctx so common_fit_params can adjust them
            mparams.n_gpu_layers = c.llama_model_default_params().n_gpu_layers;
            mparams.tensor_split = fit_tensor_split.ptr;
            mparams.tensor_buft_overrides = fit_overrides.ptr;
            cparams.n_ctx = 0;

            const margins = try a.alloc(usize, c.llama_max_devices());
            @memset(margins, inst.fit_target *% 1024 *% 1024);

            const n_ctx_needed: u32 = @bitCast(inst.n_prompt + inst.n_gen + inst.n_depth);
            cparams.n_ctx = @max(cparams.n_ctx, n_ctx_needed);

            var patterns = try fit.Patterns.init(a);
            const model_z = try a.dupeSentinel(u8, inst.model, 0);
            _ = try fit.fitParams(
                init.gpa,
                &patterns,
                .{ .verbosity = std.math.maxInt(i32), .sink = tc.common.fitLogSink },
                model_z.ptr,
                @ptrCast(&mparams),
                @ptrCast(&cparams),
                fit_tensor_split.ptr,
                @ptrCast(fit_overrides.ptr),
                margins.ptr,
                inst.fit_min_ctx,
                null,
                if (params.verbose) c.GGML_LOG_LEVEL_DEBUG else c.GGML_LOG_LEVEL_ERROR,
            );
        }

        // keep the same model between tests when possible
        if (lmodel == null or prev_inst == null or !inst.equalMparams(prev_inst.?)) {
            if (lmodel) |m| c.llama_model_free(m);

            const model_z = try a.dupeSentinel(u8, inst.model, 0);
            lmodel = c.llama_model_load_from_file(model_z.ptr, mparams);
            if (lmodel == null) {
                io.eprintf("%s: error: failed to load model '%s'\n", .{ func, model_z.ptr });
                return 1;
            }
            prev_inst = inst;
        }

        const ctx = c.llama_init_from_model(lmodel, cparams) orelse {
            const model_z = try a.dupeSentinel(u8, inst.model, 0);
            io.eprintf("%s: error: failed to create context with model '%s'\n", .{ func, model_z.ptr });
            c.llama_model_free(lmodel);
            return 1;
        };

        var t = try Test.init(a, inst, lmodel);

        c.llama_memory_clear(c.llama_get_memory(ctx), false);

        // cool off before the test
        if (params.delay != 0) _ = sleep(@intCast(params.delay));

        var tpp = c.ggml_threadpool_params_default(inst.n_threads);
        if (!tc.arg_options.parseCpuMask(inst.cpu_mask, &tpp.cpumask)) {
            const z = try a.dupeSentinel(u8, inst.cpu_mask, 0);
            io.eprintf("%s: failed to parse cpu-mask: %s\n", .{ func, z.ptr });
            c.llama_free(ctx);
            c.llama_model_free(lmodel);
            exitWith(1);
        }
        tpp.strict_cpu = inst.cpu_strict;
        tpp.poll = @intCast(inst.poll);
        tpp.prio = params.prio;

        const threadpool = c.ggml_threadpool_new(&tpp) orelse {
            io.eprintf("%s: threadpool create failed : n_threads %d\n", .{ func, tpp.n_threads });
            c.llama_free(ctx);
            c.llama_model_free(lmodel);
            exitWith(1);
        };

        c.llama_attach_threadpool(ctx, threadpool, null);

        // warmup run
        if (!params.no_warmup) {
            if (inst.n_prompt > 0) {
                if (params.progress) io.eprintf("llama-bench: benchmark %d/%zu: warmup prompt run\n", .{ params_idx, params_count });
                if (!try testPrompt(ctx, inst.n_prompt, inst.n_batch, inst.n_threads)) failTest("%s: error: failed to run prompt warmup\n", ctx, lmodel);
            }
            if (inst.n_gen > 0) {
                if (params.progress) io.eprintf("llama-bench: benchmark %d/%zu: warmup generation run\n", .{ params_idx, params_count });
                if (!testGen(ctx, 1, inst.n_threads)) failTest("%s: error: failed to run gen warmup\n", ctx, lmodel);
            }
        }

        var i: c_int = 0;
        while (i < params.reps) : (i += 1) {
            c.llama_memory_clear(c.llama_get_memory(ctx), false);

            if (inst.n_depth > 0) {
                var is_cached = inst.n_depth == cstate.depth;

                if (is_cached) {
                    // if previously we have computed at this depth, just restore the state
                    const ret = c.llama_state_seq_set_data(ctx, cstate.buf.items.ptr, cstate.buf.items.len, 0);
                    // if the old state is incompatible with the current context - reprocess from scratch
                    if (ret == 0) is_cached = false;
                }

                if (!is_cached) {
                    if (params.progress) io.eprintf("llama-bench: benchmark %d/%zu: depth run %d/%d\n", .{ params_idx, params_count, i + 1, params.reps });
                    if (!try testPrompt(ctx, inst.n_depth, inst.n_batch, inst.n_threads)) failTest("%s: error: failed to run depth\n", ctx, lmodel);

                    // store the context state for reuse in later runs
                    cstate.depth = inst.n_depth;
                    try cstate.buf.resize(gpa, c.llama_state_seq_get_size(ctx, 0));
                    _ = c.llama_state_seq_get_data(ctx, cstate.buf.items.ptr, cstate.buf.items.len, 0);
                } else {
                    if (params.progress) io.eprintf("llama-bench: benchmark %d/%zu: depth run %d/%d (cached)\n", .{ params_idx, params_count, i + 1, params.reps });
                }
            }

            const t_start = getTimeNs();

            if (inst.n_prompt > 0) {
                if (params.progress) io.eprintf("llama-bench: benchmark %d/%zu: prompt run %d/%d\n", .{ params_idx, params_count, i + 1, params.reps });
                if (!try testPrompt(ctx, inst.n_prompt, inst.n_batch, inst.n_threads)) failTest("%s: error: failed to run prompt\n", ctx, lmodel);
            }
            if (inst.n_gen > 0) {
                if (params.progress) io.eprintf("llama-bench: benchmark %d/%zu: generation run %d/%d\n", .{ params_idx, params_count, i + 1, params.reps });
                if (!testGen(ctx, inst.n_gen, inst.n_threads)) failTest("%s: error: failed to run gen\n", ctx, lmodel);
            }

            const t_ns = getTimeNs() - t_start;
            try t.samples_ns.append(a, t_ns);
        }

        if (p) |*pr| {
            try pr.printTest(&t);
            _ = fflush(pr.fout);
        }
        if (p_err) |*pr| {
            try pr.printTest(&t);
            _ = fflush(pr.fout);
        }

        c.llama_perf_context_print(ctx);

        c.llama_free(ctx);

        c.ggml_threadpool_free(threadpool);
    }

    c.llama_model_free(lmodel);

    if (p) |*pr| pr.printFooter();
    if (p_err) |*pr| pr.printFooter();

    c.llama_backend_free();

    return 0;
}

/// Ports `main` (tools/llama-bench/main.cpp:3 @c1d0e7a00).
pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(init, arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
