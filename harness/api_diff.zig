//! Compares the port's `llama.h` entry points in `src/llama/api/` --
//! `llama.zig`, `model.zig`, `vocab.zig`, `impl.zig`, `batch.zig`,
//! `chat.zig`, `cparams.zig` and `model_loader.zig` -- with the C++ that
//! defines the same names in `libllama.a`.
//!
//! **Not a port.** The harness behind `scripts/api-diff`. Both sides live in
//! one process: every function is called once through `llama.h` and once
//! through the port's function of the same name, which must have the
//! translated C signature exactly -- `F` below fails to compile otherwise.
//! The internals behind these entry points are gated elsewhere
//! (`vocab-diff`, `loader-diff`, `chat-diff`, `batch-diff`); this is about
//! the C boundary:
//!
//! - **strings into a caller's buffer**, at every size around the answer
//!   -- null with 0, 1, one short, exact, one over, large -- with the
//!   buffer pre-filled, so a stray byte or a missing terminator shows;
//! - **return codes**: lengths, negative lengths, -1, 0;
//! - **what each load logs**, through every loading entry point and its
//!   errors, and every model and vocabulary getter on what it loaded;
//! - **every metadata key and value by index**, and by key;
//! - **every token** through the C API: text, score bits, attributes,
//!   flags and its piece under `special` x `lstrip` x buffer size;
//! - **tokenize and detokenize** of upstream's vocab test strings and
//!   random text under every flag combination and buffer size;
//! - **the built-in chat templates**, each by name, null, unknown and raw
//!   template text, into small and large buffers;
//! - **split names**, **ftype names**, **load-mode names**, the logger;
//! - **what terminates**: an uncaught C++ exception or a `GGML_ASSERT`
//!   at the boundary, run in forked children, must kill both sides with
//!   the same message.
//!
//! Every file loads vocabulary-only through `llama_model_load_from_file`
//! on both sides; the Qwen3.5 models, whose architecture is ported, load in
//! full too. The loading entry points' variants run on the first of them.
//!
//! Usage: api_diff <tmpdir> <file.gguf | corpus.inp>...

const std = @import("std");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;
const M = harness_root.model;
const api = harness_root.api;
const Token = c.llama_token;

const gpa = std.heap.c_allocator;

extern fn _Z18llama_log_internal14ggml_log_levelPKcz(level: c.enum_ggml_log_level, format: [*:0]const u8, ...) void;

extern "c" fn fork() c_int;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn dup2(old: c_int, new: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*c.FILE;
extern "c" fn fclose(f: *c.FILE) c_int;

var checks: usize = 0;
var failures: usize = 0;
var case_failures: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    case_failures += 1;
    if (case_failures <= 8) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

// -----------------------------------------------------------------------------
// The two sides

const Side = enum { ref, port };

const port_mods = .{ api.llama, api.model, api.vocab, api.impl, api.batch, api.chat, api.cparams, api.model_loader };

/// The function `name` on side `s`: `llama.h`'s, or the port's, which must
/// have exactly the translated type.
fn F(comptime s: Side, comptime name: []const u8) *const @TypeOf(@field(c, name)) {
    if (s == .ref) return &@field(c, name);
    inline for (port_mods) |m| {
        if (@hasDecl(m, name)) return &@field(m, name);
    }
    @compileError("the port has no " ++ name);
}

// -----------------------------------------------------------------------------
// Transcripts

var rec: ?*std.ArrayList(u8) = null;

fn logCb(_: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const r = rec orelse return;
    r.appendSlice(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
}

/// `n` copies of `ch`, as a string literal.
fn rep(comptime ch: u8, comptime n: usize) *const [n:0]u8 {
    return comptime blk: {
        const a: [n:0]u8 = @splat(ch);
        break :blk &a;
    };
}

fn w(out: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) void {
    out.print(gpa, fmt ++ "\n", args) catch unreachable;
}

fn str(p: [*c]const u8) []const u8 {
    if (p == null) return "(null)";
    return std.mem.span(@as([*:0]const u8, @ptrCast(p)));
}

/// Lines that move with time or with what the process allocated.
fn normalize(log: []const u8) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, log, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "ggml_metal_library_compile_pipeline:")) continue;
        if (!first) out.append(gpa, '\n') catch unreachable;
        first = false;
        // A device's free memory is the system's, at that moment.
        if (std.mem.endsWith(u8, line, " MiB free")) {
            if (std.mem.findLast(u8, line, " - ")) |at| {
                out.appendSlice(gpa, line[0..at]) catch unreachable;
                out.appendSlice(gpa, " - N MiB free") catch unreachable;
                continue;
            }
        }
        // CUDA on a unified-memory device reads the system's free memory
        // from /proc/meminfo, as `context_diff.zig` notes.
        if (std.mem.find(u8, line, "available_memory_kb: ")) |at| {
            out.appendSlice(gpa, line[0 .. at + "available_memory_kb: ".len]) catch unreachable;
            out.appendSlice(gpa, "N") catch unreachable;
            continue;
        }
        if (std.mem.startsWith(u8, line, "ggml_metal_log_allocated_size:")) {
            if (std.mem.findScalar(u8, line, '(')) |open| {
                out.appendSlice(gpa, line[0..open]) catch unreachable;
                continue;
            }
        }
        out.appendSlice(gpa, line) catch unreachable;
    }
    return out;
}

/// Compares two transcripts line by line, reporting the first difference.
fn sameTranscript(name: []const u8, a: []const u8, b: []const u8) void {
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return;
        checks += 1;
        if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
            fail("{s}: line {d}:\n    ref  '{s}'\n    port '{s}'", .{ name, line, la orelse "<end>", lb orelse "<end>" });
            return;
        }
    }
}

/// Runs `script` on both sides and compares the transcripts.
fn both(name: []const u8, comptime script: anytype, args: anytype) void {
    var a: std.ArrayList(u8) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(gpa);
    @call(.auto, script, .{ Side.ref, &a } ++ args);
    @call(.auto, script, .{ Side.port, &b } ++ args);
    sameTranscript(name, a.items, b.items);
}

const sentinel: u8 = 0xA5;

/// A buffer of `n` bytes, pre-filled, with room to see writes past `n`.
fn scratchBuf(n: usize) []u8 {
    const b = gpa.alloc(u8, n + 16) catch unreachable;
    @memset(b, sentinel);
    return b;
}

/// The whole pre-filled buffer, escaped.
fn dumpBuf(out: *std.ArrayList(u8), b: []const u8) void {
    out.print(gpa, "  buf \"{f}\"\n", .{std.zig.fmtString(b)}) catch unreachable;
}

// -----------------------------------------------------------------------------
// Terminations, in forked children

/// What a child that should have died printed: the uncaught exception's
/// type and message, or `ggml_abort`'s message after its `file:line: `,
/// which names the C++ source on one side and the port's citation on the
/// other. Null when it did not die of SIGABRT.
fn dies(body: *const fn (ctx: usize) void, ctx: usize, out: []u8) ?[]const u8 {
    var fds: [2]c_int = undefined;
    if (pipe(&fds) != 0) return null;
    const pid = fork();
    if (pid == 0) {
        _ = dup2(fds[1], 2);
        _ = close(fds[0]);
        body(ctx);
        _exit(0);
    }
    _ = close(fds[1]);
    var len: usize = 0;
    while (len < out.len) {
        const got = read(fds[0], out[len..].ptr, out.len - len);
        if (got <= 0) break;
        len += @intCast(got);
    }
    _ = close(fds[0]);
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    if (status & 0x7f != 6) return null;
    var lines = std.mem.splitScalar(u8, out[0..len], '\n');
    while (lines.next()) |line| {
        if (std.mem.find(u8, line, "terminating due to uncaught exception")) |at| return line[at..];
        // `file:line: message`
        const colon = std.mem.findScalar(u8, line, ':') orelse continue;
        const rest = line[colon + 1 ..];
        const colon2 = std.mem.find(u8, rest, ": ") orelse continue;
        if (colon2 == 0) continue;
        for (rest[0..colon2]) |ch| if (!std.ascii.isDigit(ch)) break else return rest[colon2 + 2 ..];
    }
    return "(died, no message)";
}

fn sameDeath(name: []const u8, ref: *const fn (usize) void, port: *const fn (usize) void, ctx: usize) void {
    var a: [8192]u8 = undefined;
    var b: [8192]u8 = undefined;
    const da = dies(ref, ctx, &a);
    const db = dies(port, ctx, &b);
    if (std.c.getenv("API_DIFF_VERBOSE") != null) std.debug.print("    death {s}: ref '{s}' port '{s}'\n", .{ name, da orelse "(lived)", db orelse "(lived)" });
    checks += 1;
    if (da == null) {
        fail("{s}: the C++ did not abort", .{name});
        return;
    }
    if (db == null) {
        fail("{s}: the port did not abort (C++: {s})", .{ name, da.? });
        return;
    }
    if (!std.mem.eql(u8, da.?, db.?)) fail("{s}: death differs:\n    ref  '{s}'\n    port '{s}'", .{ name, da.?, db.? });
}

// -----------------------------------------------------------------------------
// Everything that needs no model

fn staticScript(comptime s: Side, out: *std.ArrayList(u8)) void {
    // Names.
    for ([_]c.enum_llama_flash_attn_type{ c.LLAMA_FLASH_ATTN_TYPE_AUTO, c.LLAMA_FLASH_ATTN_TYPE_DISABLED, c.LLAMA_FLASH_ATTN_TYPE_ENABLED }) |t| {
        w(out, "flash_attn_type_name {d} {s}", .{ t, str(F(s, "llama_flash_attn_type_name")(t)) });
    }
    var lm: c.enum_llama_load_mode = 0;
    while (lm <= c.LLAMA_LOAD_MODE_DIRECT_IO) : (lm += 1) {
        const n = F(s, "llama_load_mode_name")(lm);
        w(out, "load_mode_name {d} {s} -> {d}", .{ lm, str(n), F(s, "llama_load_mode_from_str")(n) });
    }
    var ft: c_uint = 0;
    while (ft < 64) : (ft += 1) {
        w(out, "ftype_name {d} {s} | {s}", .{ ft, str(F(s, "llama_ftype_name")(ft)), str(F(s, "llama_ftype_name")(ft | c.LLAMA_FTYPE_GUESSED)) });
    }
    for ([_]c_uint{ 999, 1023, 2048, 0xffffffff, 0xfffffbff }) |v| w(out, "ftype_name {d} {s}", .{ v, str(F(s, "llama_ftype_name")(v)) });
    var mk: c_uint = 0;
    while (mk < 16) : (mk += 1) w(out, "meta_key_str {d} {s}", .{ mk, str(F(s, "llama_model_meta_key_str")(mk)) });
    w(out, "meta_key_str big {s}", .{str(F(s, "llama_model_meta_key_str")(0xffffffff))});

    // Constants and capabilities.
    w(out, "chain_default no_perf {}", .{F(s, "llama_sampler_chain_default_params")().no_perf});
    const mp = F(s, "llama_model_default_params")();
    w(out, "model_default_params {x}", .{std.mem.asBytes(&mp)});
    w(out, "max_devices {d} max_buft_overrides {d} max_parallel {d}", .{ F(s, "llama_max_devices")(), F(s, "llama_max_tensor_buft_overrides")(), F(s, "llama_max_parallel_sequences")() });
    w(out, "supports mmap {} mlock {} gpu {} rpc {}", .{ F(s, "llama_supports_mmap")(), F(s, "llama_supports_mlock")(), F(s, "llama_supports_gpu_offload")(), F(s, "llama_supports_rpc")() });
    w(out, "version {s}", .{str(F(s, "llama_version")())});
    w(out, "system_info {s}", .{str(F(s, "llama_print_system_info")())});
    w(out, "system_info again {s}", .{str(F(s, "llama_print_system_info")())});

    // Built-in template names.
    var names: [80][*c]const u8 = undefined;
    for ([_]usize{ 0, 1, 7, 53, 54, 55, 80 }) |len| {
        @memset(&names, null);
        const n = F(s, "llama_chat_builtin_templates")(if (len == 0) null else &names, len);
        w(out, "builtin_templates len {d} -> {d}", .{ len, n });
        for (names[0..@min(len, names.len)]) |p| w(out, "  {s}", .{str(p)});
    }
    w(out, "builtin_templates null,0 -> {d}", .{F(s, "llama_chat_builtin_templates")(null, 0)});

    splitScript(s, out);
    batchScript(s, out);

    F(s, "llama_numa_init")(c.GGML_NUMA_STRATEGY_DISABLED);
    F(s, "llama_backend_free")();
    F(s, "llama_backend_init")();
    w(out, "time_us positive {}", .{F(s, "llama_time_us")() > 0});
}

fn splitScript(comptime s: Side, out: *std.ArrayList(u8)) void {
    const long = "/very/long/" ++ rep('x', 300);
    const prefixes = [_][:0]const u8{ "", "a", "/models/m", "/models/m-00001-of-00002.gguf", long };
    const nos = [_]i32{ -1, 0, 1, 7, 99998, 99999, 100000, std.math.minInt(i32), std.math.maxInt(i32) - 1 };
    const counts = [_]i32{ 0, 1, 2, 8, 99999, 100000, -1, std.math.minInt(i32), std.math.maxInt(i32) };
    const lens = [_]usize{ 0, 1, 10, 23, 24, 25, 40, 1024 };
    for (prefixes) |p| for (nos) |no| for (counts) |cnt| for (lens) |len| {
        const b = scratchBuf(len);
        defer gpa.free(b);
        const r = F(s, "llama_split_path")(if (len == 0) null else b.ptr, len, p.ptr, no, cnt);
        w(out, "split_path '{s}' {d} {d} {d} -> {d}", .{ p[0..@min(p.len, 20)], no, cnt, len, r });
        dumpBuf(out, b);
    };

    // Prefixes: a real split name for each (no, count), and near misses.
    var path_buf: [512]u8 = undefined;
    for (prefixes) |p| for (nos) |no| for (counts) |cnt| {
        const n = c.snprintf(&path_buf, path_buf.len, "%s-%05d-of-%05d.gguf", p.ptr, no +% 1, cnt);
        if (n <= 0 or n >= path_buf.len) continue;
        const real = path_buf[0..@intCast(n) :0];
        for ([_]i32{ no, no +% 1 }) |ask| for (lens) |len| {
            const b = scratchBuf(len);
            defer gpa.free(b);
            const r = F(s, "llama_split_prefix")(if (len == 0) null else b.ptr, len, real.ptr, ask, cnt);
            w(out, "split_prefix {d} {d} {d} {d} -> {d}", .{ no, ask, cnt, len, r });
            dumpBuf(out, b);
        };
    };
    for ([_][:0]const u8{ "", "x", "-00001-of-00002.gguf", "a-00001-of-00002.gguf", "a-00001-of-00002.ggu", "a-1-of-2.gguf" }) |p| for (lens) |len| {
        const b = scratchBuf(len);
        defer gpa.free(b);
        const r = F(s, "llama_split_prefix")(if (len == 0) null else b.ptr, len, p.ptr, 0, 2);
        w(out, "split_prefix '{s}' {d} -> {d}", .{ p, len, r });
        dumpBuf(out, b);
    };
}

fn batchScript(comptime s: Side, out: *std.ArrayList(u8)) void {
    var toks = [_]Token{ 1, 2, 3 };
    const one = F(s, "llama_batch_get_one")(&toks, 3);
    w(out, "get_one n {d} token {} embd {} pos {} n_seq_id {} seq_id {} logits {}", .{
        one.n_tokens, one.token == @as([*c]Token, &toks), one.embd == null, one.pos == null, one.n_seq_id == null, one.seq_id == null, one.logits == null,
    });
    const shapes = [_][3]i32{ .{ 0, 0, 1 }, .{ 1, 0, 1 }, .{ 5, 0, 2 }, .{ 3, 8, 1 }, .{ 512, 0, 4 }, .{ 2, 4, 0 } };
    for (shapes) |sh| {
        const b = F(s, "llama_batch_init")(sh[0], sh[1], sh[2]);
        w(out, "init {d},{d},{d}: n {d} token {} embd {} pos {} n_seq_id {} seq_id {} logits {} term {}", .{
            sh[0],            sh[1],            sh[2],                             b.n_tokens,
            b.token != null,  b.embd != null,   b.pos != null,                     b.n_seq_id != null,
            b.seq_id != null, b.logits != null, b.seq_id[@intCast(sh[0])] == null,
        });
        var k: usize = 0;
        while (k < sh[0]) : (k += 1) {
            if (b.seq_id[k] == null) w(out, "  seq_id[{d}] null", .{k});
        }
        // Each side frees what the other allocated: both are malloc.
        F(if (s == .ref) .port else .ref, "llama_batch_free")(b);
    }
    F(s, "llama_batch_free")(std.mem.zeroes(c.llama_batch));
}

// -----------------------------------------------------------------------------
// Chat templates

const conversation = [_]c.llama_chat_message{
    .{ .role = "system", .content = "You are a helpful assistant. Réponds en français." },
    .{ .role = "user", .content = "Hello! What is 2+2?\n" },
    .{ .role = "assistant", .content = "  It is 4.  " },
    .{ .role = "user", .content = "And \xe2\x9c\x93 the capital of France?" },
};

fn chatScript(comptime s: Side, out: *std.ArrayList(u8), tmpls: []const ?[:0]const u8) void {
    const apply = F(s, "llama_chat_apply_template");
    for (tmpls) |t| {
        const tp: [*c]const u8 = if (t) |x| x.ptr else null;
        for ([_]usize{ 0, 1, 2, 4 }) |n_msg| for ([_]bool{ false, true }) |add_ass| {
            const msgs: [*c]const c.llama_chat_message = if (n_msg == 0) null else &conversation;
            const r = apply(tp, msgs, n_msg, add_ass, null, 0);
            w(out, "apply '{s}' n {d} ass {} -> {d}", .{ if (t) |x| x[0..@min(x.len, 40)] else "(null)", n_msg, add_ass, r });
            if (r < 0) continue;
            const ru: usize = @intCast(r);
            for ([_]usize{ 1, ru -| 1, ru, ru + 1, ru + 40 }) |len| {
                const b = scratchBuf(len + 8);
                defer gpa.free(b);
                const r2 = apply(tp, msgs, n_msg, add_ass, b.ptr, @intCast(len));
                w(out, "  len {d} -> {d}", .{ len, r2 });
                dumpBuf(out, b);
            }
            const b = scratchBuf(8);
            defer gpa.free(b);
            w(out, "  len -3 -> {d}", .{apply(tp, msgs, n_msg, add_ass, b.ptr, -3)});
            dumpBuf(out, b);
        };
    }
}

// -----------------------------------------------------------------------------
// Logging

var cap_level: c.enum_ggml_log_level = 0;

fn capLevel(level: c.enum_ggml_log_level, text: [*c]const u8, ud: ?*anyopaque) callconv(.c) void {
    cap_level = level;
    logCb(level, text, ud);
}

fn logScript(comptime s: Side, out: *std.ArrayList(u8)) void {
    var mine: std.ArrayList(u8) = .empty;
    defer mine.deinit(gpa);
    const saved = rec;
    defer rec = saved;
    rec = &mine;

    var tag: u8 = 1;
    F(s, "llama_log_set")(capLevel, &tag);
    var cb: c.ggml_log_callback = null;
    var ud: ?*anyopaque = null;
    F(s, "llama_log_get")(&cb, &ud);
    w(out, "get after set: cb {} ud {}", .{ cb == @as(c.ggml_log_callback, capLevel), ud == @as(?*anyopaque, &tag) });
    // Through the other side's getter: both read ggml's.
    F(if (s == .ref) .port else .ref, "llama_log_get")(&cb, &ud);
    w(out, "other get: cb {} ud {}", .{ cb == @as(c.ggml_log_callback, capLevel), ud == @as(?*anyopaque, &tag) });

    const long = rep('y', 300);
    const msgs = [_][:0]const u8{ "", "x", rep('a', 126), rep('a', 127), rep('a', 128), rep('a', 129), long };
    for (msgs) |m| for ([_]c_uint{ c.GGML_LOG_LEVEL_NONE, c.GGML_LOG_LEVEL_DEBUG, c.GGML_LOG_LEVEL_INFO, c.GGML_LOG_LEVEL_WARN, c.GGML_LOG_LEVEL_ERROR, c.GGML_LOG_LEVEL_CONT }) |lvl| {
        mine.clearRetainingCapacity();
        cap_level = 99;
        if (s == .ref) {
            _Z18llama_log_internal14ggml_log_levelPKcz(lvl, "%s|%d|%.3f|%zu|%c\n", m.ptr, @as(c_int, -42), @as(f64, 2.5), @as(usize, 1) << 40, @as(c_int, 'Z'));
        } else {
            harness_root.log.logInternal(lvl, "%s|%d|%.3f|%zu|%c\n", .{ m.ptr, @as(c_int, -42), @as(f64, 2.5), @as(usize, 1) << 40, @as(c_int, 'Z') });
        }
        w(out, "log {d} -> level {d} \"{f}\"", .{ lvl, cap_level, std.zig.fmtString(mine.items) });
    };

    F(s, "llama_log_set")(null, null);
    F(s, "llama_log_get")(&cb, &ud);
    w(out, "get after null: cb null {} ud null {}", .{ cb == null, ud == null });
    if (s == .port) w(out, "default restored {}", .{harness_root.log.g_logger_state.log_callback == @as(c.ggml_log_callback, harness_root.log.callbackDefault)}) else w(out, "default restored true", .{});
    c.llama_log_set(logCb, null);
    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
}

fn refDefaultLog(_: usize) void {
    c.llama_log_set(null, null);
    _Z18llama_log_internal14ggml_log_levelPKcz(c.GGML_LOG_LEVEL_INFO, "x:1: to stderr %d\n", @as(c_int, 7));
    c.ggml_abort("x", 1, "done");
}

fn portDefaultLog(_: usize) void {
    api.impl.llama_log_set(null, null);
    harness_root.log.logInternal(c.GGML_LOG_LEVEL_INFO, "x:1: to stderr %d\n", .{@as(c_int, 7)});
    c.ggml_abort("x", 1, "done");
}

// -----------------------------------------------------------------------------
// Models

fn modelScript(comptime s: Side, out: *std.ArrayList(u8), m: ?*c.struct_llama_model) void {
    const mc: ?*const c.struct_llama_model = m;
    w(out, "vocab non-null {}", .{F(s, "llama_model_get_vocab")(mc) != null});
    {
        const n_layer = F(s, "llama_model_n_layer")(mc);
        w(out, "n_ctx_train {d} n_embd {d} inp {d} out {d} n_layer {d} nextn {d} n_swa {d}", .{
            F(s, "llama_model_n_ctx_train")(mc), F(s, "llama_model_n_embd")(mc), F(s, "llama_model_n_embd_inp")(mc),
            F(s, "llama_model_n_embd_out")(mc),  n_layer,                        F(s, "llama_model_n_layer_nextn")(mc),
            F(s, "llama_model_n_swa")(mc),
        });
        if (n_layer > 0) w(out, "n_head {d} n_head_kv {d} dep {d}", .{ F(s, "llama_model_n_head")(mc), F(s, "llama_model_n_head_kv")(mc), F(s, "llama_n_head")(mc) });
        w(out, "deprecated {d} {d} {d}", .{ F(s, "llama_n_ctx_train")(mc), F(s, "llama_n_embd")(mc), F(s, "llama_n_layer")(mc) });
        w(out, "rope_freq_scale_train {x}", .{@as(u32, @bitCast(F(s, "llama_model_rope_freq_scale_train")(mc)))});
        w(out, "ftype {d} size {d} n_params {d} dec_start {d}", .{ F(s, "llama_model_ftype")(mc), F(s, "llama_model_size")(mc), F(s, "llama_model_n_params")(mc), F(s, "llama_model_decoder_start_token")(mc) });
        var dlen: i32 = F(s, "llama_model_desc")(mc, null, 0);
        w(out, "desc len {d}", .{dlen});
        if (dlen < 0) dlen = 0;
        const dl: usize = @intCast(dlen);
        for ([_]usize{ 1, 5, dl -| 1, dl, dl + 1, 256 }) |len| {
            const b = scratchBuf(len);
            defer gpa.free(b);
            w(out, "  desc {d} -> {d}", .{ len, F(s, "llama_model_desc")(mc, b.ptr, len) });
            dumpBuf(out, b);
        }
    }
    w(out, "rope_type {d}", .{F(s, "llama_model_rope_type")(mc)});
    w(out, "encoder {} decoder {} recurrent {} hybrid {} diffusion {}", .{
        F(s, "llama_model_has_encoder")(mc), F(s, "llama_model_has_decoder")(mc),  F(s, "llama_model_is_recurrent")(mc),
        F(s, "llama_model_is_hybrid")(mc),   F(s, "llama_model_is_diffusion")(mc),
    });
    const n_cls = F(s, "llama_model_n_cls_out")(mc);
    w(out, "n_cls_out {d}", .{n_cls});
    var i: u32 = 0;
    while (i < n_cls + 2) : (i += 1) w(out, "  cls_label {d} {s}", .{ i, str(F(s, "llama_model_cls_label")(mc, i)) });
    w(out, "  cls_label max {s}", .{str(F(s, "llama_model_cls_label")(mc, std.math.maxInt(u32)))});

    for ([_]?[:0]const u8{ null, "tool_use", "default", "rag", "", "no-such-template" }) |name| {
        w(out, "chat_template {s}: \"{f}\"", .{ name orelse "(null)", std.zig.fmtString(str(F(s, "llama_model_chat_template")(mc, if (name) |n| n.ptr else null))) });
    }

    // Metadata.
    const count = F(s, "llama_model_meta_count")(mc);
    w(out, "meta_count {d}", .{count});
    var idx: i32 = -2;
    var big: [65536]u8 = undefined;
    while (idx < count + 2) : (idx += 1) {
        for ([2][]const u8{ "key", "val" }) |which| {
            const f = if (which[0] == 'k') F(s, "llama_model_meta_key_by_index") else F(s, "llama_model_meta_val_str_by_index");
            const n = f(mc, idx, null, 0);
            w(out, "meta {s} {d} -> {d}", .{ which, idx, n });
            const nu: usize = if (n > 0) @intCast(n) else 0;
            for ([_]usize{ 1, 7, nu -| 1, nu, nu + 1 }) |len| {
                const b = scratchBuf(len);
                defer gpa.free(b);
                w(out, "  {d} -> {d}", .{ len, f(mc, idx, b.ptr, len) });
                dumpBuf(out, b);
            }
            if (idx >= 0 and idx < count and which[0] == 'k') {
                // The value by its key.
                _ = f(mc, idx, &big, big.len);
                const key: [*:0]const u8 = @ptrCast(&big);
                const vn = F(s, "llama_model_meta_val_str")(mc, key, null, 0);
                const vb = scratchBuf(@intCast(@max(vn, 0) + 1));
                defer gpa.free(vb);
                w(out, "  by key -> {d} {d}", .{ vn, F(s, "llama_model_meta_val_str")(mc, key, vb.ptr, vb.len - 16) });
                dumpBuf(out, vb);
            }
        }
    }
    var mk: c_uint = 0;
    while (mk < 14) : (mk += 1) {
        const key = F(s, "llama_model_meta_key_str")(mk);
        if (key == null) continue;
        const b = scratchBuf(64);
        defer gpa.free(b);
        w(out, "meta_key {d} -> {d}", .{ mk, F(s, "llama_model_meta_val_str")(mc, key, b.ptr, 64) });
        dumpBuf(out, b);
    }
    for ([_][:0]const u8{ "", "no.such.key", "general.architecture", "tokenizer.ggml.tokens" }) |key| for ([_]usize{ 0, 1, 4096 }) |len| {
        const b = scratchBuf(len);
        defer gpa.free(b);
        w(out, "meta_val '{s}' {d} -> {d}", .{ key, len, F(s, "llama_model_meta_val_str")(mc, key.ptr, if (len == 0) null else b.ptr, len) });
        dumpBuf(out, b);
    };
}

// -----------------------------------------------------------------------------
// Vocabularies

fn vocabScalars(comptime s: Side, out: *std.ArrayList(u8), v: ?*const c.struct_llama_vocab) void {
    const names = .{
        "llama_vocab_bos",     "llama_vocab_eos",     "llama_vocab_eot",      "llama_vocab_cls",     "llama_vocab_sep",
        "llama_vocab_nl",      "llama_vocab_pad",     "llama_vocab_mask",     "llama_vocab_fim_pre", "llama_vocab_fim_suf",
        "llama_vocab_fim_mid", "llama_vocab_fim_pad", "llama_vocab_fim_rep",  "llama_vocab_fim_sep", "llama_token_bos",
        "llama_token_eos",     "llama_token_eot",     "llama_token_cls",      "llama_token_sep",     "llama_token_nl",
        "llama_token_pad",     "llama_token_fim_pre", "llama_token_fim_suf",  "llama_token_fim_mid", "llama_token_fim_pad",
        "llama_token_fim_rep", "llama_token_fim_sep", "llama_vocab_n_tokens", "llama_n_vocab",
    };
    inline for (names) |n| w(out, "{s} {d}", .{ n, F(s, n)(v) });
    w(out, "type {d} add bos {} eos {} sep {} dep {} {}", .{
        F(s, "llama_vocab_type")(v),        F(s, "llama_vocab_get_add_bos")(v), F(s, "llama_vocab_get_add_eos")(v),
        F(s, "llama_vocab_get_add_sep")(v), F(s, "llama_add_bos_token")(v),     F(s, "llama_add_eos_token")(v),
    });
    var n_sup: i32 = -7;
    const sup = F(s, "llama_vocab_get_suppress_tokens")(v, &n_sup);
    w(out, "suppress null {} n {d}", .{ sup == null, n_sup });
    if (sup != null) for (sup[0..@intCast(n_sup)]) |t| w(out, "  {d}", .{t});
    w(out, "suppress null-count null {}", .{F(s, "llama_vocab_get_suppress_tokens")(v, null) == null});
}

fn tokenToPiece(comptime s: Side, v: ?*const c.struct_llama_vocab, tok: Token, b: []u8, len: i32, lstrip: i32, special: bool) i32 {
    return F(s, "llama_token_to_piece")(v, tok, if (len == 0) null else b.ptr, len, lstrip, special);
}

/// Every token, both sides at once.
fn checkTokens(name: []const u8, rv: ?*const c.struct_llama_vocab, pv: ?*const c.struct_llama_vocab) void {
    const n: i32 = c.llama_vocab_n_tokens(rv);
    const is_rwkv = c.llama_vocab_type(rv) == c.LLAMA_VOCAB_TYPE_RWKV;
    var ba: [512]u8 = undefined;
    var bb: [512]u8 = undefined;
    var t: i32 = 0;
    while (t < n) : (t += 1) {
        checks += 1;
        const ta = str(c.llama_vocab_get_text(rv, t));
        const tb = str(F(.port, "llama_vocab_get_text")(pv, t));
        const tb2 = str(F(.port, "llama_token_get_text")(pv, t));
        if (!std.mem.eql(u8, ta, tb) or !std.mem.eql(u8, ta, tb2)) fail("{s}: token {d} text differs", .{ name, t });
        const sa: u32 = @bitCast(c.llama_vocab_get_score(rv, t));
        if (sa != @as(u32, @bitCast(F(.port, "llama_vocab_get_score")(pv, t))) or sa != @as(u32, @bitCast(F(.port, "llama_token_get_score")(pv, t))))
            fail("{s}: token {d} score differs", .{ name, t });
        const aa = c.llama_vocab_get_attr(rv, t);
        if (aa != F(.port, "llama_vocab_get_attr")(pv, t) or aa != F(.port, "llama_token_get_attr")(pv, t)) fail("{s}: token {d} attr differs", .{ name, t });
        const ea = c.llama_vocab_is_eog(rv, t);
        if (ea != F(.port, "llama_vocab_is_eog")(pv, t) or ea != F(.port, "llama_token_is_eog")(pv, t)) fail("{s}: token {d} eog differs", .{ name, t });
        const ca = c.llama_vocab_is_control(rv, t);
        if (ca != F(.port, "llama_vocab_is_control")(pv, t) or ca != F(.port, "llama_token_is_control")(pv, t)) fail("{s}: token {d} control differs", .{ name, t });

        for ([_]bool{ false, true }) |special| for ([_]i32{ 0, 1, 3 }) |lstrip| {
            @memset(&ba, sentinel);
            @memset(&bb, sentinel);
            const full = tokenToPiece(.ref, rv, t, &ba, 256, lstrip, special);
            const fullp = tokenToPiece(.port, pv, t, &bb, 256, lstrip, special);
            checks += 1;
            if (full != fullp or !std.mem.eql(u8, &ba, &bb)) {
                fail("{s}: token {d} piece (256, lstrip {d}, special {}) -> {d} vs {d}", .{ name, t, lstrip, special, full, fullp });
                continue;
            }
            const need: i32 = if (full > 0) full else 1;
            for ([_]i32{ 0, need - 1, need, -1 }) |len| {
                if (len < 0 and is_rwkv) continue; // the C++ copies past the buffer
                @memset(&ba, sentinel);
                @memset(&bb, sentinel);
                const ra = tokenToPiece(.ref, rv, t, &ba, len, lstrip, special);
                const rb = tokenToPiece(.port, pv, t, &bb, len, lstrip, special);
                checks += 1;
                if (ra != rb or !std.mem.eql(u8, &ba, &bb)) fail("{s}: token {d} piece ({d}, lstrip {d}, special {}) -> {d} vs {d}", .{ name, t, len, lstrip, special, ra, rb });
            }
        };
    }
    // Ids past the vocabulary that do not throw.
    for ([_]Token{ n, n + 5, -1, std.math.minInt(Token) }) |x| {
        checks += 1;
        if (c.llama_vocab_is_eog(rv, x) != F(.port, "llama_vocab_is_eog")(pv, x)) fail("{s}: is_eog({d}) differs", .{ name, x });
    }
}

fn tokenizeOn(comptime s: Side, v: ?*const c.struct_llama_vocab, text: []const u8, buf: []Token, n_max: i32, add: bool, parse: bool) i32 {
    return F(s, "llama_tokenize")(v, if (text.len == 0) null else text.ptr, @intCast(text.len), if (n_max <= 0) null else buf.ptr, n_max, add, parse);
}

fn detokenizeOn(comptime s: Side, v: ?*const c.struct_llama_vocab, toks: []const Token, n: i32, buf: []u8, len: i32, rm: bool, unparse: bool) i32 {
    return F(s, "llama_detokenize")(v, if (toks.len == 0) null else toks.ptr, n, if (len <= 0) null else buf.ptr, len, rm, unparse);
}

fn checkDetokenize(name: []const u8, rv: ?*const c.struct_llama_vocab, pv: ?*const c.struct_llama_vocab, toks: []const Token) void {
    var ba: [8192]u8 = undefined;
    var bb: [8192]u8 = undefined;
    for ([_]bool{ false, true }) |rm| for ([_]bool{ false, true }) |unparse| {
        const nt: i32 = @intCast(toks.len);
        const need = detokenizeOn(.ref, rv, toks, nt, &ba, 0, rm, unparse);
        const needu: i32 = if (need < 0) -need else need;
        for ([_]i32{ 0, needu - 1, needu, needu + 1, 4096 }) |len| {
            if (len < 0 or len > 4096) continue;
            @memset(&ba, sentinel);
            @memset(&bb, sentinel);
            const ra = detokenizeOn(.ref, rv, toks, nt, &ba, len, rm, unparse);
            const rb = detokenizeOn(.port, pv, toks, nt, &bb, len, rm, unparse);
            checks += 1;
            if (ra != rb or !std.mem.eql(u8, &ba, &bb)) fail("{s}: detokenize {d} tokens into {d} (rm {}, unparse {}) -> {d} vs {d}", .{ name, toks.len, len, rm, unparse, ra, rb });
        }
        // A negative count is no tokens; a negative length with none left
        // is 0.
        for ([_]i32{ -1, -100 }) |n| {
            const ra = detokenizeOn(.ref, rv, toks, n, &ba, 64, rm, unparse);
            const rb = detokenizeOn(.port, pv, toks, n, &bb, 64, rm, unparse);
            checks += 1;
            if (ra != rb) fail("{s}: detokenize n {d} -> {d} vs {d}", .{ name, n, ra, rb });
        }
        const ra = detokenizeOn(.ref, rv, &.{}, 0, &ba, -5, rm, unparse);
        const rb = detokenizeOn(.port, pv, &.{}, 0, &bb, -5, rm, unparse);
        checks += 1;
        if (ra != rb) fail("{s}: detokenize nothing into -5 -> {d} vs {d}", .{ name, ra, rb });
    };
}

fn checkText(name: []const u8, rv: ?*const c.struct_llama_vocab, pv: ?*const c.struct_llama_vocab, text: []const u8) void {
    var ta: [4096]Token = undefined;
    var tb: [4096]Token = undefined;
    for ([_]bool{ false, true }) |add| for ([_]bool{ false, true }) |parse| {
        const need = tokenizeOn(.ref, rv, text, &ta, 0, add, parse);
        const needu: i32 = if (need < 0) -need else need;
        if (needu > 4000) continue;
        for ([_]i32{ 0, needu - 1, needu, needu + 3, -1 }) |n_max| {
            if (n_max == -1 and needu == 0) continue;
            @memset(std.mem.sliceAsBytes(&ta), sentinel);
            @memset(std.mem.sliceAsBytes(&tb), sentinel);
            const ra = tokenizeOn(.ref, rv, text, &ta, n_max, add, parse);
            const rb = tokenizeOn(.port, pv, text, &tb, n_max, add, parse);
            checks += 1;
            if (ra != rb or !std.mem.eql(Token, &ta, &tb)) {
                fail("{s}: tokenize \"{f}\" (add {}, parse {}, max {d}) -> {d} vs {d}", .{ name, std.zig.fmtString(text[0..@min(text.len, 60)]), add, parse, n_max, ra, rb });
                return;
            }
        }
        const r = tokenizeOn(.ref, rv, text, &ta, 4096, add, parse);
        if (r > 0) checkDetokenize(name, rv, pv, ta[0..@intCast(r)]);
    };
}

fn buildCorpus(rng: std.Random, rv: ?*const c.struct_llama_vocab, base: []const []const u8, out: *std.ArrayList([]u8)) void {
    for (base) |s| out.append(gpa, gpa.dupe(u8, s) catch unreachable) catch unreachable;
    const n: i32 = c.llama_vocab_n_tokens(rv);
    // Special and user-defined token texts, spliced into random text.
    var specials: std.ArrayList([]const u8) = .empty;
    defer specials.deinit(gpa);
    var t: i32 = 0;
    while (t < n) : (t += 1) {
        const a = c.llama_vocab_get_attr(rv, t);
        if (a & (c.LLAMA_TOKEN_ATTR_CONTROL | c.LLAMA_TOKEN_ATTR_USER_DEFINED) != 0) specials.append(gpa, str(c.llama_vocab_get_text(rv, t))) catch unreachable;
    }
    const alphabet = " \t\nabcxyzABCXYZ0123456789.,!?'\"-_()[]<>|/\\\xc3\xa9\xe2\x80\x99\xf0\x9f\x98\x80\xe4\xb8\xad";
    for (0..60) |_| {
        var s: std.ArrayList(u8) = .empty;
        const len = rng.uintLessThan(usize, 80);
        for (0..len) |_| {
            if (specials.items.len != 0 and rng.uintLessThan(u8, 12) == 0) {
                if (rng.boolean()) s.append(gpa, ' ') catch unreachable;
                s.appendSlice(gpa, specials.items[rng.uintLessThan(usize, specials.items.len)]) catch unreachable;
                if (rng.boolean()) s.append(gpa, ' ') catch unreachable;
            } else if (rng.uintLessThan(u8, 30) == 0) {
                s.append(gpa, rng.int(u8)) catch unreachable; // invalid UTF-8 too
            } else {
                s.append(gpa, alphabet[rng.uintLessThan(usize, alphabet.len)]) catch unreachable;
            }
        }
        out.append(gpa, s.toOwnedSlice(gpa) catch unreachable) catch unreachable;
    }
}

fn checkVocab(name: []const u8, rv: ?*const c.struct_llama_vocab, pv: ?*const c.struct_llama_vocab, corpus: []const []const u8) void {
    {
        var a: std.ArrayList(u8) = .empty;
        defer a.deinit(gpa);
        var b: std.ArrayList(u8) = .empty;
        defer b.deinit(gpa);
        vocabScalars(.ref, &a, rv);
        vocabScalars(.port, &b, pv);
        sameTranscript(name, a.items, b.items);
    }
    checkTokens(name, rv, pv);

    var prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(0, name));
    const rng = prng.random();
    var texts: std.ArrayList([]u8) = .empty;
    defer {
        for (texts.items) |s| gpa.free(s);
        texts.deinit(gpa);
    }
    buildCorpus(rng, rv, corpus, &texts);
    for (texts.items) |s| checkText(name, rv, pv, s);

    // Random ids, every one in range.
    const n: i32 = c.llama_vocab_n_tokens(rv);
    var toks: [64]Token = undefined;
    for (0..40) |_| {
        const k = rng.uintLessThan(usize, toks.len);
        for (toks[0..k]) |*t| t.* = rng.intRangeLessThan(Token, 0, n);
        if (k > 0 and rng.boolean()) toks[0] = c.llama_vocab_bos(rv);
        if (k > 1 and rng.boolean()) toks[k - 1] = c.llama_vocab_eos(rv);
        for (toks[0..k]) |*t| if (t.* < 0 or t.* >= n) {
            t.* = 0;
        };
        checkDetokenize(name, rv, pv, toks[0..k]);
    }
}

// The terminations, through one vocabulary the parent set.
var death_rv: ?*const c.struct_llama_vocab = null;
var death_pv: ?*const c.struct_llama_vocab = null;

fn refPieceOob(k: usize) void {
    var b: [16]u8 = undefined;
    _ = c.llama_token_to_piece(death_rv, @intCast(k), &b, 16, 0, true);
}
fn portPieceOob(k: usize) void {
    var b: [16]u8 = undefined;
    _ = F(.port, "llama_token_to_piece")(death_pv, @intCast(k), &b, 16, 0, true);
}
fn refTextOob(k: usize) void {
    _ = c.llama_vocab_get_text(death_rv, @intCast(k));
}
fn portTextOob(k: usize) void {
    _ = F(.port, "llama_vocab_get_text")(death_pv, @intCast(k));
}
fn refScoreOob(k: usize) void {
    _ = c.llama_vocab_get_score(death_rv, @intCast(k));
}
fn portScoreOob(k: usize) void {
    _ = F(.port, "llama_vocab_get_score")(death_pv, @intCast(k));
}
fn refAttrOob(k: usize) void {
    _ = c.llama_token_get_attr(death_rv, @intCast(k));
}
fn portAttrOob(k: usize) void {
    _ = F(.port, "llama_token_get_attr")(death_pv, @intCast(k));
}
fn refTokNeg(_: usize) void {
    var t: [8]Token = undefined;
    _ = c.llama_tokenize(death_rv, "abc", -1, &t, 8, false, false);
}
fn portTokNeg(_: usize) void {
    var t: [8]Token = undefined;
    _ = F(.port, "llama_tokenize")(death_pv, "abc", -1, &t, 8, false, false);
}
fn refDetokNeg(_: usize) void {
    const t = [_]Token{ 5, 6 };
    _ = c.llama_detokenize(death_rv, &t, 2, null, -1, false, false);
}
fn portDetokNeg(_: usize) void {
    const t = [_]Token{ 5, 6 };
    _ = F(.port, "llama_detokenize")(death_pv, &t, 2, null, -1, false, false);
}
fn refDetokOob(k: usize) void {
    const t = [_]Token{@intCast(k)};
    var b: [16]u8 = undefined;
    _ = c.llama_detokenize(death_rv, &t, 1, &b, 16, false, true);
}
fn portDetokOob(k: usize) void {
    const t = [_]Token{@intCast(k)};
    var b: [16]u8 = undefined;
    _ = F(.port, "llama_detokenize")(death_pv, &t, 1, &b, 16, false, true);
}

fn checkVocabDeaths(name: []const u8, n_vocab: i32) void {
    var nb: [128]u8 = undefined;
    const oob: usize = @intCast(n_vocab);
    sameDeath(std.fmt.bufPrint(&nb, "{s}: piece of {d}", .{ name, oob }) catch name, refPieceOob, portPieceOob, oob);
    sameDeath(std.fmt.bufPrint(&nb, "{s}: text of {d}", .{ name, oob }) catch name, refTextOob, portTextOob, oob);
    sameDeath(std.fmt.bufPrint(&nb, "{s}: score of {d}", .{ name, oob + 100 }) catch name, refScoreOob, portScoreOob, oob + 100);
    sameDeath(std.fmt.bufPrint(&nb, "{s}: attr of -1", .{name}) catch name, refAttrOob, portAttrOob, @bitCast(@as(isize, -1)));
    sameDeath(std.fmt.bufPrint(&nb, "{s}: tokenize length -1", .{name}) catch name, refTokNeg, portTokNeg, 0);
    sameDeath(std.fmt.bufPrint(&nb, "{s}: detokenize into -1", .{name}) catch name, refDetokNeg, portDetokNeg, 0);
    sameDeath(std.fmt.bufPrint(&nb, "{s}: detokenize {d}", .{ name, oob }) catch name, refDetokOob, portDetokOob, oob);
}

// Other terminations.
fn refBadLoadMode(_: usize) void {
    _ = c.llama_load_mode_from_str("mmap ");
}
fn portBadLoadMode(_: usize) void {
    _ = F(.port, "llama_load_mode_from_str")("mmap ");
}
fn refLoadModeName(_: usize) void {
    _ = c.llama_load_mode_name(6);
}
fn portLoadModeName(_: usize) void {
    _ = F(.port, "llama_load_mode_name")(6);
}
fn refFaName(_: usize) void {
    _ = c.llama_flash_attn_type_name(2);
}
fn portFaName(_: usize) void {
    _ = F(.port, "llama_flash_attn_type_name")(2);
}
fn refUserNull(_: usize) void {
    _ = c.llama_model_init_from_user(null, null, null, c.llama_model_default_params());
}
fn portUserNull(_: usize) void {
    _ = F(.port, "llama_model_init_from_user")(null, null, null, c.llama_model_default_params());
}

// -----------------------------------------------------------------------------
// Loading

const Loaded = struct { rm: ?*c.struct_llama_model, pm: ?*c.struct_llama_model };

/// Loads through `name` on both sides with the same arguments, and compares
/// what each logged and whether each loaded.
fn loadBoth(label: []const u8, comptime name: []const u8, args: anytype) Loaded {
    var la: std.ArrayList(u8) = .empty;
    defer la.deinit(gpa);
    var lb: std.ArrayList(u8) = .empty;
    defer lb.deinit(gpa);
    rec = &la;
    const rm = @call(.auto, F(.ref, name), args);
    rec = &lb;
    const pm = @call(.auto, F(.port, name), args);
    rec = null;
    checks += 1;
    if ((rm == null) != (pm == null)) fail("{s}: C++ loaded {} port loaded {}", .{ label, rm != null, pm != null });
    var na = normalize(la.items);
    defer na.deinit(gpa);
    var nb = normalize(lb.items);
    defer nb.deinit(gpa);
    sameTranscript(label, na.items, nb.items);
    return .{ .rm = rm, .pm = pm };
}

fn freeBoth(l: Loaded) void {
    if (l.rm) |m| c.llama_model_free(m);
    if (l.pm) |m| F(.port, "llama_model_free")(m);
}

fn compareModels(label: []const u8, rm: *c.struct_llama_model, pm: *c.struct_llama_model) void {
    var a: std.ArrayList(u8) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(gpa);
    modelScript(.ref, &a, rm);
    modelScript(.port, &b, pm);
    sameTranscript(label, a.items, b.items);
}

fn checkLoadVariants(path: [:0]const u8) void {
    var mp = c.llama_model_default_params();
    mp.vocab_only = true;
    var l = loadBoth("load_model_from_file (deprecated)", "llama_load_model_from_file", .{ path.ptr, mp });
    if (l.rm != null and l.pm != null) compareModels("deprecated load", l.rm.?, l.pm.?);
    freeBoth(l);

    var paths = [_][*c]const u8{ path.ptr, "/nonexistent-split.gguf" };
    l = loadBoth("load_from_splits [path]", "llama_model_load_from_splits", .{ @as([*c][*c]const u8, &paths), @as(usize, 1), mp });
    if (l.rm != null and l.pm != null) compareModels("splits load", l.rm.?, l.pm.?);
    freeBoth(l);
    freeBoth(loadBoth("load_from_splits [path, missing]", "llama_model_load_from_splits", .{ @as([*c][*c]const u8, &paths), @as(usize, 2), mp }));
    freeBoth(loadBoth("load_from_splits []", "llama_model_load_from_splits", .{ @as([*c][*c]const u8, &paths), @as(usize, 0), mp }));
    var empty = [_][*c]const u8{""};
    freeBoth(loadBoth("load_from_splits [\"\"]", "llama_model_load_from_splits", .{ @as([*c][*c]const u8, &empty), @as(usize, 1), mp }));

    freeBoth(loadBoth("load_from_file \"\"", "llama_model_load_from_file", .{ "", mp }));
    freeBoth(loadBoth("load_from_file missing", "llama_model_load_from_file", .{ "/nonexistent/model.gguf", mp }));
    freeBoth(loadBoth("load_from_file a corpus file", "llama_model_load_from_file", .{ "llama.cpp/models/ggml-vocab-llama-spm.gguf.inp", mp }));
    freeBoth(loadBoth("load_from_file_ptr null", "llama_model_load_from_file_ptr", .{ @as([*c]c.FILE, null), mp }));

    {
        // Each side reads its own handle from the start.
        var la: std.ArrayList(u8) = .empty;
        defer la.deinit(gpa);
        var lb: std.ArrayList(u8) = .empty;
        defer lb.deinit(gpa);
        const fa = fopen(path.ptr, "rb").?;
        defer _ = fclose(fa);
        const fb = fopen(path.ptr, "rb").?;
        defer _ = fclose(fb);
        rec = &la;
        const rm = c.llama_model_load_from_file_ptr(@ptrCast(fa), mp);
        rec = &lb;
        const pm = F(.port, "llama_model_load_from_file_ptr")(@ptrCast(fb), mp);
        rec = null;
        checks += 1;
        if ((rm == null) != (pm == null)) fail("load_from_file_ptr: C++ {} port {}", .{ rm != null, pm != null });
        sameTranscript("load_from_file_ptr", la.items, lb.items);
        if (rm != null and pm != null) compareModels("file_ptr load", rm.?, pm.?);
        freeBoth(.{ .rm = rm, .pm = pm });
    }
    {
        const meta = c.gguf_init_from_file(path.ptr, .{ .no_alloc = true, .ctx = null }).?;
        defer c.gguf_free(meta);
        l = loadBoth("init_from_user", "llama_model_init_from_user", .{ meta, @as(c.llama_model_set_tensor_data_t, null), @as(?*anyopaque, null), mp });
        if (l.rm != null and l.pm != null) compareModels("init_from_user", l.rm.?, l.pm.?);
        freeBoth(l);
    }
    {
        // A whole model from edited metadata, its tensors zeroed by the
        // caller: what a vocabulary-only load never reads -- classifier
        // labels, named templates, the sampling keys -- on the full path.
        const meta = c.gguf_init_from_file(path.ptr, .{ .no_alloc = true, .ctx = null }).?;
        defer c.gguf_free(meta);
        addMeta(meta, "qwen35");
        var fp = c.llama_model_default_params();
        fp.n_gpu_layers = 0;
        l = loadBoth("init_from_user, edited, full", "llama_model_init_from_user", .{ meta, @as(c.llama_model_set_tensor_data_t, zeroTensor), @as(?*anyopaque, null), fp });
        if (l.rm != null and l.pm != null) {
            compareModels("init_from_user, edited, full", l.rm.?, l.pm.?);
        } else {
            fail("init_from_user, edited, full: C++ {} port {}", .{ l.rm != null, l.pm != null });
        }
        freeBoth(l);
    }
    F(.port, "llama_model_free")(null);
    F(.port, "llama_free_model")(null);
}

fn zeroTensor(t: [*c]c.ggml_tensor, _: ?*anyopaque) callconv(.c) void {
    c.ggml_backend_tensor_memset(t, 0, 0, c.ggml_nbytes(t));
}

// -----------------------------------------------------------------------------
// Synthesized files

fn setStrs(ctx: *c.gguf_context, key: [:0]const u8, strs: []const [:0]const u8) void {
    var ptrs: std.ArrayList([*c]const u8) = .empty;
    defer ptrs.deinit(gpa);
    for (strs) |x| ptrs.append(gpa, x.ptr) catch unreachable;
    c.gguf_set_arr_str(ctx, key.ptr, @ptrCast(ptrs.items.ptr), ptrs.items.len);
}

/// The metadata no shipped file has, for an architecture named `arch`.
fn addMeta(ctx: *c.gguf_context, comptime arch: []const u8) void {
    const labels_key = arch ++ ".classifier.output_labels";
    setStrs(ctx, labels_key, &.{ "negative", "positive", "", "n\xc3\xa9utre \"q\"" });
    c.gguf_set_val_str(ctx, "tokenizer.chat_template", "{% for m in messages %}<|im_start|>{{ m.role }}{% endfor %}");
    c.gguf_set_val_str(ctx, "tokenizer.chat_template.tool_use", "[INST] tools [/INST]");
    c.gguf_set_val_str(ctx, "tokenizer.chat_template.rag", "");
    c.gguf_set_val_str(ctx, "general.sampling.sequence", "top_k;top_p;temp");
    c.gguf_set_val_i32(ctx, "general.sampling.top_k", 40);
    c.gguf_set_val_f32(ctx, "general.sampling.top_p", 0.95);
    c.gguf_set_val_f32(ctx, "general.sampling.temp", 0.7);
    c.gguf_set_val_i32(ctx, "general.sampling.mirostat", 2);
    c.gguf_set_val_f32(ctx, "general.sampling.mirostat_eta", 1.0e-7);
    c.gguf_set_val_bool(ctx, "test.bool", true);
    c.gguf_set_val_u64(ctx, "test.u64", std.math.maxInt(u64));
    c.gguf_set_val_i64(ctx, "test.i64", std.math.minInt(i64));
    c.gguf_set_val_f64(ctx, "test.f64", -1.0e300);
    c.gguf_set_val_i8(ctx, "test.i8", -128);
    c.gguf_set_val_u16(ctx, "test.u16", 65535);
    c.gguf_set_val_str(ctx, "test.quoted", "a \"b\" \\c\nd\te");
    c.gguf_set_val_str(ctx, "test.empty", "");
    const long = rep('L', 9000);
    c.gguf_set_val_str(ctx, "test.long", long);
}

/// Upstream's SPM vocabulary with what no shipped file has: classifier
/// labels, named chat templates, the sampling keys `llama_model_meta_key_str`
/// names, and values that are long, quoted, empty or of every scalar type.
fn synthMeta(dir: []const u8, base: [:0]const u8) [:0]const u8 {
    const ctx = c.gguf_init_from_file(base.ptr, .{ .no_alloc = true, .ctx = null }).?;
    defer c.gguf_free(ctx);
    addMeta(ctx, "llama");
    const path = gpa.printSentinel("{s}/meta-rich.gguf", .{dir}, 0) catch unreachable;
    if (!c.gguf_write_to_file(ctx, path.ptr, true)) @panic("cannot write a synthetic file");
    return path;
}

// -----------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: api_diff <tmpdir> <file.gguf | corpus.inp>...\n", .{});
        std.process.exit(2);
    }

    var corpus: std.ArrayList([]const u8) = .empty;
    var files: std.ArrayList([:0]const u8) = .empty;
    for (args[2..]) |a| {
        if (std.mem.endsWith(u8, a, ".inp")) {
            const f = std.c.fopen(a.ptr, "rb") orelse return error.CannotOpen;
            var buf: std.ArrayList(u8) = .empty;
            var chunk: [65536]u8 = undefined;
            while (true) {
                const n = std.c.fread(&chunk, 1, chunk.len, f);
                if (n == 0) break;
                try buf.appendSlice(gpa, chunk[0..n]);
            }
            _ = std.c.fclose(f);
            var it = std.mem.splitSequence(u8, buf.items, "\n__ggml_vocab_test__\n");
            while (it.next()) |s| try corpus.append(gpa, s);
        } else {
            try files.append(gpa, try gpa.dupeSentinel(u8, a, 0));
        }
    }

    c.llama_log_set(logCb, null);

    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    c.llama_backend_init();

    for (files.items) |f| {
        if (std.mem.endsWith(u8, f, "ggml-vocab-llama-spm.gguf")) {
            try files.append(gpa, synthMeta(args[1], f));
            break;
        }
    }

    // No model.
    case_failures = 0;
    both("statics", staticScript, .{});
    both("logging", logScript, .{});
    sameDeath("default logger", refDefaultLog, portDefaultLog, 0);
    sameDeath("load_mode_from_str unknown", refBadLoadMode, portBadLoadMode, 0);
    sameDeath("load_mode_name 6", refLoadModeName, portLoadModeName, 0);
    sameDeath("flash_attn_type_name 2", refFaName, portFaName, 0);
    sameDeath("init_from_user null", refUserNull, portUserNull, 0);
    std.debug.print("  {s:<40} {s:<4} {d:>10} checks\n", .{ "statics, logging, terminations", if (case_failures == 0) "ok" else "FAIL", checks });

    // Chat templates: every built-in name, null, nothing, raw texts.
    var tmpls: std.ArrayList(?[:0]const u8) = .empty;
    {
        var names: [128][*c]const u8 = undefined;
        const n: usize = @intCast(c.llama_chat_builtin_templates(&names, names.len));
        for (names[0..n]) |p| try tmpls.append(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(p))));
        for ([_][:0]const u8{ "", "bogus", "chatml ", "<|im_start|>", "[INST]", "<start_of_turn>", "<|user|>", "### Instruction:", "{{ bos_token }}" }) |t| try tmpls.append(gpa, t);
        try tmpls.append(gpa, null);
    }

    var tally_loaded: usize = 0;
    var variants_on: ?[:0]const u8 = null;
    for (files.items) |path| {
        case_failures = 0;
        const name = std.Io.Dir.path.basename(path);
        var mp = c.llama_model_default_params();
        mp.vocab_only = true;
        const l = loadBoth(name, "llama_model_load_from_file", .{ path.ptr, mp });
        defer freeBoth(l);
        if (l.rm == null or l.pm == null) {
            fail("{s}: not loaded on both sides", .{name});
            continue;
        }
        const rm = l.rm.?;
        const pm = l.pm.?;
        tally_loaded += 1;
        const is_model = c.llama_model_n_params(rm) > 0;
        if (variants_on == null and is_model) variants_on = path;
        compareModels(name, rm, pm);
        checkVocab(name, c.llama_model_get_vocab(rm), F(.port, "llama_model_get_vocab")(pm), corpus.items);
        // The model's own template, as raw text.
        if (c.llama_model_chat_template(rm, null)) |t| try tmpls.append(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(t))));
        death_rv = c.llama_model_get_vocab(rm);
        death_pv = F(.port, "llama_model_get_vocab")(pm);
        checkVocabDeaths(name, c.llama_vocab_n_tokens(death_rv));

        // A model, loaded in full.
        if (is_model) {
            var fp = c.llama_model_default_params();
            fp.vocab_only = false;
            var label: [256]u8 = undefined;
            const lab = std.fmt.bufPrint(&label, "{s} (full)", .{name}) catch name;
            const full = loadBoth(lab, "llama_model_load_from_file", .{ path.ptr, fp });
            if (full.rm != null and full.pm != null) compareModels(lab, full.rm.?, full.pm.?);
            freeBoth(full);
        }
        std.debug.print("  {s:<40} {s:<4} {d:>10} checks\n", .{ name, if (case_failures == 0) "ok" else "FAIL", checks });
    }

    case_failures = 0;
    both("chat templates", chatScript, .{tmpls.items});
    std.debug.print("  {s:<40} {s:<4} {d:>10} checks, {d} templates\n", .{ "chat_apply_template", if (case_failures == 0) "ok" else "FAIL", checks, tmpls.items.len });

    if (variants_on orelse if (files.items.len != 0) files.items[0] else null) |p| {
        case_failures = 0;
        checkLoadVariants(p);
        std.debug.print("  {s:<40} {s:<4} {d:>10} checks\n", .{ "loading entry points", if (case_failures == 0) "ok" else "FAIL", checks });
    } else {
        fail("no file the port loads: the loading entry points went unchecked", .{});
    }

    std.debug.print("api-diff: {d} files; {d} checks, {d} failures\n", .{ tally_loaded, checks, failures });
    if (failures != 0) std.process.exit(1);
}
