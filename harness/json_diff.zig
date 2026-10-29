//! Compares `cli/json/` with upstream's `common_json` over nlohmann.
//!
//! **Not a port.** The harness behind `scripts/json-diff`. A program -- a
//! byte string of operations over eight registers -- is run twice in one
//! process: by `ref_json_run` in `harness/json_ref.cpp`, which is
//! `common/json.cpp` compiled unmodified over the vendored nlohmann, and by
//! `zigRun` below, which makes the same calls on the port. Each operation
//! leaves one record -- `ok`, a dump, a query's answer, or `E:` and the
//! `what()` of what was thrown -- and every record must be identical.
//!
//! Five corpora:
//!
//! - **documents**: random valid JSON text -- every value type, deep and
//!   shallow nesting, escapes, surrogate pairs, raw UTF-8, duplicate keys,
//!   random whitespace -- parsed, then dumped at indents -1, 0, 1, 2 and 4
//!   (`dump`, `dump_safe`, and nlohmann's `dump` with `ensure_ascii` and
//!   each error handler), queried, iterated and read with every `get<T>()`.
//! - **malformed**: the same documents with bytes deleted, inserted,
//!   replaced or cut off, a BOM in front, a NUL inside; the error text, or
//!   the discard, must match.
//! - **numbers**: integer text at and one past the int64 and uint64
//!   bounds, `-0`, denormals, overflow, long mantissas and exponents, and
//!   random doubles written shortest, at random precision and in fixed
//!   notation -- through `parse`, then `dump` and `get<T>()`.
//! - **doubles**: random bit patterns -- every class, denormals, integers,
//!   powers of ten and two and their neighbours, short decimals -- dumped:
//!   Grisu2's digits, `-0.0`, NaN and the infinities as `null`.
//! - **mutations**: random sequences of every accessor and modifier
//!   `json.h` offers over random values, strings and keys with invalid
//!   UTF-8 among them, so `dump` must throw where upstream throws.

const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");

extern fn ref_json_run(prog: [*]const u8, len: usize, out: *[*]u8) usize;

const Value = json.Value;

const Op = enum(u8) {
    parse = 0,
    parse_nt,
    set_null,
    set_bool,
    set_int,
    set_uint,
    set_double,
    set_string,
    set_array,
    set_object,
    dump,
    dump_safe,
    dump_raw,
    query,
    get,
    at_key,
    at_idx,
    index_key,
    index_idx,
    contains,
    value,
    front,
    back,
    clear,
    erase_key,
    erase_idx,
    push,
    insert,
    set_item,
    assign,
    eq,
    iter,
    set_float,
    push_obj,
    const_index,
};

const n_get_types = 20;
const n_value_types = 9;

// -----------------------------------------------------------------------------
// programs

const Prog = struct {
    buf: std.ArrayList(u8) = .empty,
    gpa: std.mem.Allocator,

    fn op(p: *Prog, o: Op) void {
        p.buf.append(p.gpa, @backingInt(o)) catch oom();
    }
    fn u8_(p: *Prog, v: u8) void {
        p.buf.append(p.gpa, v) catch oom();
    }
    fn u32_(p: *Prog, v: u32) void {
        p.buf.appendSlice(p.gpa, std.mem.asBytes(&v)) catch oom();
    }
    fn u64_(p: *Prog, v: u64) void {
        p.buf.appendSlice(p.gpa, std.mem.asBytes(&v)) catch oom();
    }
    fn str(p: *Prog, s: []const u8) void {
        p.u32_(@intCast(s.len));
        p.buf.appendSlice(p.gpa, s) catch oom();
    }
    fn i32_(p: *Prog, v: i32) void {
        p.u32_(@bitCast(v));
    }
    fn reset(p: *Prog) void {
        p.buf.clearRetainingCapacity();
    }
};

fn oom() noreturn {
    std.debug.print("FAIL: out of memory\n", .{});
    std.process.exit(2);
}

const Reader = struct {
    p: []const u8,
    i: usize = 0,
    fn u8_(r: *Reader) u8 {
        r.i += 1;
        return r.p[r.i - 1];
    }
    fn u32_(r: *Reader) u32 {
        const v = std.mem.readInt(u32, r.p[r.i..][0..4], .little);
        r.i += 4;
        return v;
    }
    fn u64_(r: *Reader) u64 {
        const v = std.mem.readInt(u64, r.p[r.i..][0..8], .little);
        r.i += 8;
        return v;
    }
    fn str(r: *Reader) []const u8 {
        const n = r.u32_();
        const s = r.p[r.i .. r.i + n];
        r.i += n;
        return s;
    }
};

// -----------------------------------------------------------------------------
// the port's side

const Out = struct {
    buf: std.ArrayList(u8) = .empty,
    gpa: std.mem.Allocator,
    cur: usize = 0,

    fn begin(o: *Out) void {
        o.cur = o.buf.items.len;
        o.buf.appendSlice(o.gpa, &[_]u8{ 0, 0, 0, 0 }) catch oom();
    }
    fn end(o: *Out) void {
        const n: u32 = @intCast(o.buf.items.len - o.cur - 4);
        std.mem.writeInt(u32, o.buf.items[o.cur..][0..4], n, .little);
    }
    fn add(o: *Out, s: []const u8) void {
        o.buf.appendSlice(o.gpa, s) catch oom();
    }
    fn print(o: *Out, comptime fmt: []const u8, args: anytype) void {
        o.buf.print(o.gpa, fmt, args) catch oom();
    }
    fn rec(o: *Out, s: []const u8) void {
        o.begin();
        o.add(s);
        o.end();
    }
};

fn showFloat(o: *Out, x: anytype) void {
    switch (@TypeOf(x)) {
        f32 => o.print("f{x:0>16}", .{@as(u32, @bitCast(x))}),
        f64 => o.print("d{x:0>16}", .{@as(u64, @bitCast(x))}),
        else => unreachable,
    }
}

fn show(o: *Out, x: anytype) void {
    const T = @TypeOf(x);
    if (T == bool) return o.add(if (x) "true" else "false");
    if (T == []const u8) {
        o.add("s");
        return o.add(x);
    }
    switch (@typeInfo(T)) {
        .float => showFloat(o, x),
        .int => o.print("{d}", .{x}),
        else => @compileError("show"),
    }
}

/// `float_ub` of the reference: a float an integer `static_cast` cannot
/// hold, compared only on AArch64.
fn floatUb(v: *const Value, comptime T: type) bool {
    if (builtin.cpu.arch == .aarch64) return false;
    if (v.* != .number_float) return false;
    const d = v.number_float;
    const lo: f64 = @floatFromInt(std.math.minInt(T));
    const hi: f64 = @floatFromInt(std.math.maxInt(T));
    return !(d > lo - 1.0 and d < hi + 1.0);
}

fn getInt(o: *Out, v: *const Value, comptime T: type) json.Error!void {
    if (floatUb(v, T)) return o.add("ub");
    show(o, try v.get(T));
}

fn getVec(o: *Out, v: *const Value, comptime T: type) json.Error!void {
    const xs = try v.getSlice(o.gpa, T);
    defer o.gpa.free(xs);
    o.add("[");
    for (xs) |x| {
        show(o, x);
        o.add(",");
    }
    o.add("]");
}

fn anyUb(v: *const Value, comptime T: type) bool {
    var tmp = v.*;
    var it = tmp.items();
    while (it.next()) |item| if (floatUb(item.value, T)) return true;
    return false;
}

fn lessEntry(_: void, a: Value.MapEntry(usize), b: Value.MapEntry(usize)) bool {
    return switch (std.mem.order(u8, a.key, b.key)) {
        .lt => true,
        .gt => false,
        .eq => a.value < b.value,
    };
}

fn doGet(o: *Out, v: *const Value, t: u8) json.Error!void {
    switch (t) {
        0 => show(o, try v.get(bool)),
        1 => try getInt(o, v, c_int),
        2 => try getInt(o, v, c_uint),
        3 => try getInt(o, v, c_long),
        4 => try getInt(o, v, c_ulong),
        5 => try getInt(o, v, c_longlong),
        6 => try getInt(o, v, c_ulonglong),
        7 => show(o, try v.get(f32)),
        8 => show(o, try v.get(f64)),
        9, 19 => show(o, try v.get([]const u8)),
        10 => try getVec(o, v, f32),
        11 => try getVec(o, v, []const u8),
        12 => {
            const xs = try v.getStringSet(o.gpa);
            defer o.gpa.free(xs);
            o.add("[");
            for (xs) |x| {
                show(o, x);
                o.add(",");
            }
            o.add("]");
        },
        13 => {
            if (anyUb(v, i32)) return o.add("ub");
            try getVec(o, v, c_int);
        },
        14 => {
            if (anyUb(v, u64)) return o.add("ub");
            try getVec(o, v, usize);
        },
        15 => {
            if (v.* == .object and anyUb(v, u64)) return o.add("ub");
            const es = try v.getMap(o.gpa, usize);
            defer o.gpa.free(es);
            std.mem.sort(Value.MapEntry(usize), es, {}, lessEntry);
            o.add("{");
            for (es) |e| {
                o.add(e.key);
                o.add("=");
                show(o, e.value);
                o.add(",");
            }
            o.add("}");
        },
        16 => try getInt(o, v, i64),
        17 => try getInt(o, v, u32),
        18 => try getInt(o, v, usize),
        else => o.add("?"),
    }
}

fn dumpSafeInto(o: *Out, v: *const Value) void {
    json.dumpTo(&o.buf, o.gpa, v, .{ .error_handler = .replace }) catch oom();
}

fn doValue(o: *Out, v: *const Value, key: []const u8, t: u8) json.Error!void {
    switch (t) {
        0 => show(o, try v.value(key, bool, false)),
        1 => show(o, try v.value(key, c_int, -7)),
        2 => show(o, try v.value(key, c_uint, 7)),
        3 => show(o, try v.value(key, c_longlong, -9)),
        4 => show(o, try v.value(key, f64, 2.5)),
        5 => show(o, try v.value(key, f32, 1.25)),
        6 => show(o, try v.value(key, []const u8, "dflt")),
        7 => show(o, try v.value(key, []const u8, "cstr")),
        8 => {
            // value(key, const common_json &): the member itself, or the default
            if (v.contains(key)) {
                dumpSafeInto(o, try v.at(key));
            } else {
                var def = Value.emptyArray();
                defer def.deinit(o.gpa);
                try def.pushBack(o.gpa, try Value.from(o.gpa, 1));
                try def.pushBack(o.gpa, try Value.from(o.gpa, "x"));
                dumpSafeInto(o, &def);
            }
        },
        else => o.add("?"),
    }
}

fn query(o: *Out, v: *const Value) void {
    o.add(if (v.isNull()) "N" else "-");
    o.add(if (v.isObject()) "O" else "-");
    o.add(if (v.isArray()) "A" else "-");
    o.add(if (v.isString()) "S" else "-");
    o.add(if (v.isBoolean()) "B" else "-");
    o.add(if (v.isNumber()) "#" else "-");
    o.add(if (v.isNumberInteger()) "I" else "-");
    o.add(if (v.isNumberFloat()) "F" else "-");
    o.add(if (v.isDiscarded()) "D" else "-");
    o.print(" size={d}", .{v.size()});
    o.add(if (v.empty()) " empty" else " full");
}

fn handlerOf(h: u8) json.ErrorHandler {
    return switch (h) {
        0 => .strict,
        1 => .replace,
        else => .ignore,
    };
}

fn emptyContainer(v: *const Value) bool {
    return (v.isArray() or v.isObject()) and v.empty();
}

/// One operation on the port; the record is opened by the caller and an
/// error is turned into `E:` there.
fn step(o: *Out, R: *[8]Value, in: *Reader, op: Op) json.Error!void {
    const a = o.gpa;
    switch (op) {
        .parse => {
            const r = in.u8_();
            const s = in.str();
            const v = try json.parse(a, s);
            R[r].assign(a, v);
            o.add("ok");
        },
        .parse_nt => {
            const r = in.u8_();
            const s = in.str();
            R[r].assign(a, try json.parseNoThrow(a, s));
            o.add(if (R[r].isDiscarded()) "discarded" else "ok");
        },
        .set_null => {
            R[in.u8_()].assign(a, .null);
            o.add("ok");
        },
        .set_bool => {
            const r = in.u8_();
            R[r].assign(a, try Value.from(a, in.u8_() != 0));
            o.add("ok");
        },
        .set_int => {
            const r = in.u8_();
            R[r].assign(a, try Value.from(a, @as(i64, @bitCast(in.u64_()))));
            o.add("ok");
        },
        .set_uint => {
            const r = in.u8_();
            R[r].assign(a, try Value.from(a, in.u64_()));
            o.add("ok");
        },
        .set_double => {
            const r = in.u8_();
            R[r].assign(a, try Value.from(a, @as(f64, @bitCast(in.u64_()))));
            o.add("ok");
        },
        .set_float => {
            const r = in.u8_();
            R[r].assign(a, try Value.from(a, @as(f32, @bitCast(in.u32_()))));
            o.add("ok");
        },
        .set_string => {
            const r = in.u8_();
            R[r].assign(a, try Value.from(a, in.str()));
            o.add("ok");
        },
        .set_array => {
            R[in.u8_()].assign(a, Value.emptyArray());
            o.add("ok");
        },
        .set_object => {
            R[in.u8_()].assign(a, Value.emptyObject());
            o.add("ok");
        },
        .dump, .dump_safe => {
            const r = in.u8_();
            const indent: i32 = @bitCast(in.u32_());
            o.add("d");
            try json.dumpTo(&o.buf, a, &R[r], .{
                .indent = indent,
                .error_handler = if (op == .dump) .strict else .replace,
            });
        },
        .dump_raw => {
            const r = in.u8_();
            const indent: i32 = @bitCast(in.u32_());
            const ascii = in.u8_() != 0;
            const h = in.u8_();
            const ichar = in.u8_();
            o.add("d");
            try json.dumpTo(&o.buf, a, &R[r], .{
                .indent = indent,
                .indent_char = ichar,
                .ensure_ascii = ascii,
                .error_handler = handlerOf(h),
            });
        },
        .query => query(o, &R[in.u8_()]),
        .get => {
            const r = in.u8_();
            try doGet(o, &R[r], in.u8_());
        },
        .at_key, .const_index => {
            const d = in.u8_();
            const r = in.u8_();
            const k = in.str();
            const tmp = try (try R[r].at(k)).clone(a);
            R[d].assign(a, tmp);
            o.add("ok");
        },
        .at_idx => {
            const d = in.u8_();
            const r = in.u8_();
            const i = in.u64_();
            const tmp = try (try R[r].atIndex(i)).clone(a);
            R[d].assign(a, tmp);
            o.add("ok");
        },
        .index_key => {
            const r = in.u8_();
            const k = in.str();
            const s = in.u8_();
            var tmp = try R[s].clone(a);
            const slot = R[r].index(a, k) catch |e| {
                tmp.deinit(a);
                return e;
            };
            slot.assign(a, tmp);
            o.add("ok");
        },
        .index_idx => {
            const r = in.u8_();
            const i = in.u64_();
            const s = in.u8_();
            var tmp = try R[s].clone(a);
            const slot = R[r].indexAt(a, i) catch |e| {
                tmp.deinit(a);
                return e;
            };
            slot.assign(a, tmp);
            o.add("ok");
        },
        .contains => {
            const r = in.u8_();
            const k = in.str();
            o.print("{s}{d}", .{ if (R[r].contains(k)) "1" else "0", R[r].count(k) });
        },
        .value => {
            const r = in.u8_();
            const k = in.str();
            try doValue(o, &R[r], k, in.u8_());
        },
        .front, .back => {
            const d = in.u8_();
            const r = in.u8_();
            if (emptyContainer(&R[r])) return o.add("skip");
            const src = if (op == .front) try R[r].front() else try R[r].back();
            const tmp = try src.clone(a);
            R[d].assign(a, tmp);
            o.add("ok");
        },
        .clear => {
            R[in.u8_()].clear(a);
            o.add("ok");
        },
        .erase_key => {
            const r = in.u8_();
            try R[r].erase(a, in.str());
            o.add("ok");
        },
        .erase_idx => {
            const r = in.u8_();
            try R[r].eraseIndex(a, in.u64_());
            o.add("ok");
        },
        .push => {
            const r = in.u8_();
            const s = in.u8_();
            try R[r].pushBack(a, try R[s].clone(a));
            o.add("ok");
        },
        .push_obj => {
            const r = in.u8_();
            const k = in.str();
            const s = in.u8_();
            var obj = Value.emptyObject();
            try obj.set(a, k, try R[s].clone(a));
            try R[r].pushBack(a, obj);
            o.add("ok");
        },
        .insert => {
            const r = in.u8_();
            const s = in.u8_();
            try R[r].insert(a, &R[s]);
            o.add("ok");
        },
        .set_item => {
            const r = in.u8_();
            const k = in.str();
            const s = in.u8_();
            try R[r].set(a, k, try R[s].clone(a));
            o.add("ok");
        },
        .assign => {
            const d = in.u8_();
            const s = in.u8_();
            const tmp = try R[s].clone(a);
            R[d].assign(a, tmp);
            o.add("ok");
        },
        .eq => {
            const r = in.u8_();
            const s = in.u8_();
            const e = R[r].eql(&R[s]);
            o.add(if (e) "1" else "0");
            o.add(if (!e) "1" else "0");
        },
        .iter => {
            const r = in.u8_();
            const v = &R[r];
            const n = v.size();
            for (0..n) |i| {
                const elem: *const Value = switch (v.*) {
                    .object => |*m| &m.members.items[i].value,
                    .array => |*arr| &arr.items[i],
                    else => v,
                };
                dumpSafeInto(o, elem);
                if (v.keyAt(i)) |k| {
                    o.add("@");
                    o.add(k);
                } else |e| switch (e) {
                    error.JsonError => {
                        o.add("!");
                        o.add(json.what());
                    },
                    error.OutOfMemory => oom(),
                }
                o.add("|");
            }
            var it = v.items();
            while (it.next()) |item| {
                o.add(item.key);
                o.add("=");
                dumpSafeInto(o, item.value);
                o.add("|");
            }
        },
    }
}

fn zigRun(gpa: std.mem.Allocator, prog: []const u8) []u8 {
    var o: Out = .{ .gpa = gpa };
    var R: [8]Value = @splat(.null);
    defer for (&R) |*v| v.deinit(gpa);
    var in: Reader = .{ .p = prog };
    while (in.i < prog.len) {
        const op: Op = @fromBackingInt(@intCast(in.u8_()));
        o.begin();
        step(&o, &R, &in, op) catch |e| switch (e) {
            error.JsonError => {
                // the record so far is partial output; the C++ threw before
                // returning any of it
                o.buf.shrinkRetainingCapacity(o.cur + 4);
                o.add("E:");
                o.add(json.what());
            },
            error.OutOfMemory => oom(),
        };
        o.end();
    }
    return o.buf.toOwnedSlice(gpa) catch oom();
}

// -----------------------------------------------------------------------------
// comparison

var checks: usize = 0;
var failures: usize = 0;

/// What the reference answered, by kind, so a pass can be read for what
/// it covered: errors, dumps, skipped `front`/`back` on empty containers,
/// and the float-to-integer casts left out off AArch64.
var n_errors: usize = 0;
var n_dumps: usize = 0;
var n_skips: usize = 0;
var n_ub: usize = 0;

/// Every distinct error message with its digits and quoted text dropped,
/// under `JSON_DIFF_STATS=1`.
var kinds: std.StringHashMapUnmanaged(usize) = .empty;

fn tally(x: []const u8) void {
    if (std.mem.startsWith(u8, x, "E:")) {
        n_errors += 1;
        if (std.c.getenv("JSON_DIFF_STATS") == null) return;
        var buf: [256]u8 = undefined;
        var n: usize = 0;
        var quoted = false;
        for (x) |c| {
            if (c == '\'') quoted = !quoted;
            if (quoted or std.ascii.isDigit(c)) continue;
            if (n < buf.len) {
                buf[n] = c;
                n += 1;
            }
        }
        const gop = kinds.getOrPut(std.heap.c_allocator, buf[0..n]) catch oom();
        if (!gop.found_existing) {
            gop.key_ptr.* = std.heap.c_allocator.dupe(u8, buf[0..n]) catch oom();
            gop.value_ptr.* = 0;
        }
        gop.value_ptr.* += 1;
    } else if (x.len > 0 and x[0] == 'd') {
        n_dumps += 1;
    } else if (std.mem.eql(u8, x, "skip")) {
        n_skips += 1;
    } else if (std.mem.eql(u8, x, "ub")) {
        n_ub += 1;
    }
}

fn nextRecord(buf: []const u8, i: *usize) ?[]const u8 {
    if (i.* >= buf.len) return null;
    const n = std.mem.readInt(u32, buf[i.*..][0..4], .little);
    const s = buf[i.* + 4 .. i.* + 4 + n];
    i.* += 4 + n;
    return s;
}

/// Runs `prog` on both sides and compares record by record.
fn compare(gpa: std.mem.Allocator, corpus: []const u8, prog: []const u8) void {
    var ref_ptr: [*]u8 = undefined;
    const ref_len = ref_json_run(prog.ptr, prog.len, &ref_ptr);
    defer std.c.free(ref_ptr);
    const ref = ref_ptr[0..ref_len];
    const port = zigRun(gpa, prog);
    defer gpa.free(port);

    var ri: usize = 0;
    var zi: usize = 0;
    var k: usize = 0;
    while (true) : (k += 1) {
        const r = nextRecord(ref, &ri);
        const z = nextRecord(port, &zi);
        if (r == null and z == null) break;
        checks += 1;
        if (r) |x| tally(x);
        if (r != null and z != null and std.mem.eql(u8, r.?, z.?)) continue;
        failures += 1;
        if (failures <= 10) {
            std.debug.print("FAIL {s} record {d}:\n  ref:  \"{f}\"\n  port: \"{f}\"\n", .{
                corpus,                                                             k,
                std.zig.fmtString(if (r) |x| x[0..@min(x.len, 400)] else "<none>"), std.zig.fmtString(if (z) |x| x[0..@min(x.len, 400)] else "<none>"),
            });
            dumpProgram(prog, k);
        }
        if (r == null or z == null) break;
    }
}

/// Prints the operation behind record `k` (each operation leaves one).
fn dumpProgram(prog: []const u8, k: usize) void {
    var in: Reader = .{ .p = prog };
    var n: usize = 0;
    while (in.i < prog.len) : (n += 1) {
        const start = in.i;
        const op: Op = @fromBackingInt(@intCast(in.u8_()));
        skipOperands(&in, op);
        if (n == k) {
            std.debug.print("  op {s}: \"{f}\"\n", .{ @tagName(op), std.zig.fmtString(prog[start..@min(in.i, start + 400)]) });
            return;
        }
    }
}

fn skipOperands(in: *Reader, op: Op) void {
    switch (op) {
        .parse, .parse_nt, .set_string => {
            _ = in.u8_();
            _ = in.str();
        },
        .set_null, .set_array, .set_object, .query, .clear, .iter => _ = in.u8_(),
        .set_bool, .get => in.i += 2,
        .set_int, .set_uint, .set_double => in.i += 9,
        .set_float => in.i += 5,
        .dump, .dump_safe => in.i += 5,
        .dump_raw => in.i += 8,
        .at_key, .const_index => {
            in.i += 2;
            _ = in.str();
        },
        .at_idx => in.i += 10,
        .index_key, .set_item, .push_obj => {
            _ = in.u8_();
            _ = in.str();
            _ = in.u8_();
        },
        .index_idx => in.i += 10,
        .contains, .erase_key => {
            _ = in.u8_();
            _ = in.str();
        },
        .value => {
            _ = in.u8_();
            _ = in.str();
            _ = in.u8_();
        },
        .front, .back, .push, .insert, .assign, .eq => in.i += 2,
        .erase_idx => in.i += 9,
    }
}

// -----------------------------------------------------------------------------
// generators

const Gen = struct {
    r: std.Random,
    gpa: std.mem.Allocator,
    out: std.ArrayList(u8) = .empty,

    fn put(g: *Gen, s: []const u8) void {
        g.out.appendSlice(g.gpa, s) catch oom();
    }
    fn byte(g: *Gen, b: u8) void {
        g.out.append(g.gpa, b) catch oom();
    }
    fn print(g: *Gen, comptime fmt: []const u8, args: anytype) void {
        g.out.print(g.gpa, fmt, args) catch oom();
    }
    fn chance(g: *Gen, n: u32) bool {
        return g.r.uintLessThan(u32, n) == 0;
    }

    fn ws(g: *Gen) void {
        if (!g.chance(3)) return;
        const n = g.r.uintLessThan(usize, 3) + 1;
        for (0..n) |_| g.byte(" \t\n\r"[g.r.uintLessThan(usize, 4)]);
    }

    fn utf8(g: *Gen, cp: u21) void {
        var b: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &b) catch unreachable;
        g.put(b[0..n]);
    }

    fn hex4(g: *Gen, v: u16) void {
        if (g.chance(2)) g.print("\\u{X:0>4}", .{v}) else g.print("\\u{x:0>4}", .{v});
    }

    fn randomCodepoint(g: *Gen) u21 {
        return switch (g.r.uintLessThan(u8, 5)) {
            0 => g.r.intRangeAtMost(u21, 0x80, 0x7FF),
            1 => g.r.intRangeAtMost(u21, 0x800, 0xD7FF),
            2 => g.r.intRangeAtMost(u21, 0xE000, 0xFFFF),
            3 => g.r.intRangeAtMost(u21, 0x10000, 0x10FFFF),
            else => @as(u21, 0x7F),
        };
    }

    /// A string literal's body, every kind of escape and raw UTF-8.
    fn stringBody(g: *Gen) void {
        const n = g.r.uintLessThan(usize, 10);
        for (0..n) |_| {
            switch (g.r.uintLessThan(u8, 9)) {
                0, 1 => {
                    var c = g.r.intRangeAtMost(u8, 0x20, 0x7E);
                    if (c == '"' or c == '\\') c = 'q';
                    g.byte(c);
                },
                2 => g.put(([_][]const u8{ "\\\"", "\\\\", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t" })[g.r.uintLessThan(usize, 8)]),
                3 => g.hex4(g.r.intRangeAtMost(u16, 0, 0x1F)),
                4 => {
                    const cp = g.randomCodepoint();
                    if (cp <= 0xFFFF) g.hex4(@intCast(cp)) else {
                        const v = cp - 0x10000;
                        g.hex4(@intCast(0xD800 + (v >> 10)));
                        g.hex4(@intCast(0xDC00 + (v & 0x3FF)));
                    }
                },
                5, 6 => g.utf8(g.randomCodepoint()),
                7 => g.put(([_][]const u8{ "a", "b", "\\u0061", "key", "k\\u0065y", "\xc3\xa9", "\\u00e9", "\xf0\x9f\x98\x80", "\\ud83d\\ude00" })[g.r.uintLessThan(usize, 9)]),
                else => g.byte(0x7F),
            }
        }
    }

    /// A key from a small set, often spelled differently, so duplicates
    /// are common.
    fn key(g: *Gen) void {
        g.byte('"');
        if (g.chance(3)) {
            g.stringBody();
        } else {
            g.put(([_][]const u8{ "a", "b", "\\u0061", "c", "", "\\u0000", "\xc3\xa9", "\\u00E9", "a\\/b", "a/b" })[g.r.uintLessThan(usize, 10)]);
        }
        g.byte('"');
    }

    fn digits(g: *Gen, n: usize) void {
        for (0..n) |_| g.byte('0' + g.r.uintLessThan(u8, 10));
    }

    fn randomDouble(g: *Gen) f64 {
        return @bitCast(randomDoubleBits(g.r));
    }

    /// Number text: bounds, edge spellings, random shapes, real doubles.
    fn number(g: *Gen) void {
        switch (g.r.uintLessThan(u8, 8)) {
            0 => g.put(edge_numbers[g.r.uintLessThan(usize, edge_numbers.len)]),
            1 => g.print("{d}", .{g.r.int(i64)}),
            2 => g.print("{d}", .{g.r.int(u64)}),
            3 => g.print("{d}", .{g.r.intRangeAtMost(i32, -1000, 1000)}),
            4, 5 => {
                const d = g.randomDouble();
                if (!std.math.isFinite(d)) return g.put("0.5");
                switch (g.r.uintLessThan(u8, 4)) {
                    0 => g.print("{e}", .{d}),
                    1 => g.print("{e:.0}", .{d}),
                    2 => switch (g.r.uintLessThan(u8, 4)) {
                        0 => g.print("{e:.3}", .{d}),
                        1 => g.print("{e:.10}", .{d}),
                        2 => g.print("{e:.16}", .{d}),
                        else => g.print("{e:.25}", .{d}),
                    },
                    else => if (@abs(d) < 1e30) g.print("{d}", .{d}) else g.print("{e}", .{d}),
                }
            },
            else => {
                if (g.chance(2)) g.byte('-');
                if (g.chance(4)) g.byte('0') else {
                    g.byte('1' + g.r.uintLessThan(u8, 9));
                    g.digits(g.r.uintLessThan(usize, 25));
                }
                if (g.chance(2)) {
                    g.byte('.');
                    g.digits(g.r.uintLessThan(usize, 25) + 1);
                }
                if (g.chance(2)) {
                    g.byte(if (g.chance(2)) 'e' else 'E');
                    switch (g.r.uintLessThan(u8, 3)) {
                        0 => g.byte('+'),
                        1 => g.byte('-'),
                        else => {},
                    }
                    g.digits(g.r.uintLessThan(usize, 4) + 1);
                }
            },
        }
    }

    /// A valid document.
    fn value(g: *Gen, depth: usize) void {
        const leaf = depth >= 6 or g.chance(@intCast(depth + 2));
        if (leaf) {
            switch (g.r.uintLessThan(u8, 6)) {
                0 => g.put("null"),
                1 => g.put(if (g.chance(2)) "true" else "false"),
                2, 3 => g.number(),
                else => {
                    g.byte('"');
                    g.stringBody();
                    g.byte('"');
                },
            }
            return;
        }
        if (g.chance(2)) {
            g.byte('[');
            const n = g.r.uintLessThan(usize, 6);
            for (0..n) |i| {
                if (i > 0) g.byte(',');
                g.ws();
                g.value(depth + 1);
                g.ws();
            }
            if (n == 0) g.ws();
            g.byte(']');
        } else {
            g.byte('{');
            const n = g.r.uintLessThan(usize, 6);
            for (0..n) |i| {
                if (i > 0) g.byte(',');
                g.ws();
                g.key();
                g.ws();
                g.byte(':');
                g.ws();
                g.value(depth + 1);
                g.ws();
            }
            if (n == 0) g.ws();
            g.byte('}');
        }
    }

    fn document(g: *Gen) []const u8 {
        g.out.clearRetainingCapacity();
        if (g.chance(40)) {
            // deep nesting: the serializer's indent string grows past 512
            const depth = g.r.intRangeAtMost(usize, 100, 400);
            for (0..depth) |i| g.byte(if (i % 3 == 0) '{' else '[');
            g.byte('1');
            var i = depth;
            while (i > 0) {
                i -= 1;
                if (i % 3 == 0) g.byte('}') else g.byte(']');
            }
            // objects need keys: rebuild as `{"k":` openers
            const s = g.out.items;
            var fixed: std.ArrayList(u8) = .empty;
            for (s) |c| {
                if (c == '{') fixed.appendSlice(g.gpa, "{\"k\":") catch oom() else fixed.append(g.gpa, c) catch oom();
            }
            g.out.deinit(g.gpa);
            g.out = fixed;
            return g.out.items;
        }
        g.ws();
        g.value(0);
        g.ws();
        return g.out.items;
    }

    /// Breaks a document: bytes deleted, inserted, replaced, or cut off.
    fn mangle(g: *Gen) void {
        const n = g.r.uintLessThan(usize, 3) + 1;
        for (0..n) |_| {
            const len = g.out.items.len;
            switch (g.r.uintLessThan(u8, 6)) {
                0 => if (len > 0) {
                    _ = g.out.orderedRemove(g.r.uintLessThan(usize, len));
                },
                1, 2 => g.out.insert(g.gpa, g.r.uintLessThan(usize, len + 1), interesting[g.r.uintLessThan(usize, interesting.len)]) catch oom(),
                3 => if (len > 0) {
                    g.out.items[g.r.uintLessThan(usize, len)] = interesting[g.r.uintLessThan(usize, interesting.len)];
                },
                4 => g.out.shrinkRetainingCapacity(g.r.uintLessThan(usize, len + 1)),
                else => if (len > 0) {
                    g.out.items[g.r.uintLessThan(usize, len)] = g.r.int(u8);
                },
            }
        }
        if (g.chance(20)) g.out.insertSlice(g.gpa, 0, ([_][]const u8{ "\xEF\xBB\xBF", "\xEF\xBB", "\xEF", "\xEF\xBF\xBF" })[g.r.uintLessThan(usize, 4)]) catch oom();
    }
};

const interesting = [_]u8{ '{', '}', '[', ']', ':', ',', '"', '\\', '0', '-', '.', 'e', 'E', '+', 't', 'n', 'f', 'u', 0, 1, 0x1f, '\n', '\r', ' ', 0x7f, 0x80, 0xC3, 0xE2, 0xED, 0xF0, 0xF4, 0xF5, 0xFF, 0xEF, 'x' };

const edge_numbers = [_][]const u8{
    "0",                       "-0",                      "0.0",                     "-0.0",                 "0e0",                                                     "-0e-0",                                                   "0E+00",
    "1",                       "-1",                      "9223372036854775807",     "9223372036854775808",  "-9223372036854775808",                                    "-9223372036854775809",                                    "18446744073709551615",
    "18446744073709551616",    "-18446744073709551615",   "-18446744073709551616",   "18446744073709551614", "99999999999999999999",                                    "100000000000000000000000000000000000000000000000000",     "1e308",
    "1.7976931348623157e308",  "1.7976931348623158e308",  "1.7976931348623159e308",  "1e309",                "-1e309",                                                  "1e400",                                                   "1e99999999999999999999",
    "2.2250738585072011e-308", "2.2250738585072014e-308", "2.225073858507201e-308",  "4.9e-324",             "5e-324",                                                  "2.4703282292062327e-324",                                 "2.4703282292062328e-324",
    "2e-324",                  "1e-400",                  "1e-99999999999999999999", "0.1",                  "0.2",                                                     "0.30000000000000004",                                     "1E+2",
    "1e-2",                    "123456789012345678",      "1234567890123456789",     "12345678901234567890", "1e15",                                                    "1e16",                                                    "1e-4",
    "1e-5",                    "0.0001",                  "0.00001",                 "123456789012345.6",    "1234567890123456.7",                                      "9007199254740993",                                        "9007199254740993.0",
    "4294967295",              "4294967296",              "-2147483648",             "-2147483649",          "2147483647.5",                                            "-2147483648.5",                                           "3.0e10",
    "-3.0e10",                 "1.8446744073709552e19",   "-9.3e18",                 "65535",                "1.00000000000000011102230246251565404236316680908203125", "1.00000000000000011102230246251565404236316680908203126",
};

/// Bit patterns for `dump`: every class of double.
fn randomDoubleBits(r: std.Random) u64 {
    return switch (r.uintLessThan(u8, 10)) {
        0, 1 => r.int(u64),
        // normal, random mantissa, an exponent near 1
        2 => (@as(u64, r.intRangeAtMost(u11, 990, 1080)) << 52) | (r.int(u64) & 0xFFFFFFFFFFFFF) | (@as(u64, r.int(u1)) << 63),
        // denormals and the smallest normals
        3 => (r.int(u64) & 0x801FFFFFFFFFFFFF) >> @intCast(r.uintLessThan(u6, 50)),
        // integers
        4 => @bitCast(@as(f64, @floatFromInt(r.int(i64) >> @intCast(r.uintLessThan(u6, 63))))),
        // powers of ten and two, and their neighbours
        5 => blk: {
            const p = std.math.pow(f64, 10.0, @floatFromInt(r.intRangeAtMost(i32, -323, 308)));
            const b: u64 = @bitCast(p);
            break :blk b +% @as(u64, @bitCast(@as(i64, r.intRangeAtMost(i8, -2, 2))));
        },
        6 => blk: {
            const b: u64 = @as(u64, r.intRangeAtMost(u11, 1, 2046)) << 52;
            break :blk b +% @as(u64, @bitCast(@as(i64, r.intRangeAtMost(i8, -2, 2))));
        },
        // short decimals, where Grisu2 is most often not shortest
        7, 8 => blk: {
            var buf: [64]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}e{d}", .{ r.intRangeAtMost(u32, 1, 999999), r.intRangeAtMost(i32, -330, 300) }) catch unreachable;
            break :blk @bitCast(std.fmt.parseFloat(f64, s) catch unreachable);
        },
        // the specials
        else => ([_]u64{ 0, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0xFFF8000000000001, 1, 0x7FEFFFFFFFFFFFFF, 0x0010000000000000, 0x000FFFFFFFFFFFFF, 0x3FF0000000000000, 0x3FB999999999999A })[r.uintLessThan(usize, 12)],
    };
}

/// Every read of register 0: the dumps, the queries, the iteration, every
/// `get<T>()`.
fn readAll(p: *Prog, reg: u8) void {
    for ([_]i32{ -1, 0, 1, 2, 4 }) |ind| {
        p.op(.dump);
        p.u8_(reg);
        p.i32_(ind);
        p.op(.dump_safe);
        p.u8_(reg);
        p.i32_(ind);
    }
    for ([_]i32{ -1, 0, 2 }) |ind| {
        for (0..3) |h| {
            p.op(.dump_raw);
            p.u8_(reg);
            p.i32_(ind);
            p.u8_(1);
            p.u8_(@intCast(h));
            p.u8_(' ');
        }
        p.op(.dump_raw);
        p.u8_(reg);
        p.i32_(ind);
        p.u8_(0);
        p.u8_(2);
        p.u8_('\t');
    }
    p.op(.query);
    p.u8_(reg);
    p.op(.iter);
    p.u8_(reg);
    for (0..n_get_types) |t| {
        p.op(.get);
        p.u8_(reg);
        p.u8_(@intCast(t));
    }
}

const mutation_keys = [_][]const u8{ "a", "b", "c", "", "\xc3\xa9", "a\x80", "\xff", "k\x00z", "0", "1" };

fn randomBytes(r: std.Random, buf: []u8) []const u8 {
    const n = r.uintLessThan(usize, 8);
    for (buf[0..n]) |*b| b.* = switch (r.uintLessThan(u8, 6)) {
        0 => r.int(u8),
        1 => 0x80 | r.uintLessThan(u8, 0x40),
        2 => 0xC2 + r.uintLessThan(u8, 0x33),
        3 => r.uintLessThan(u8, 0x20),
        else => 'a' + r.uintLessThan(u8, 26),
    };
    // sometimes a whole valid multibyte sequence
    if (n >= 4 and r.uintLessThan(u8, 2) == 0) @memcpy(buf[0..4], "\xf0\x9f\x98\x80");
    return buf[0..n];
}

/// A random program of every accessor and modifier.
fn mutationProgram(p: *Prog, r: std.Random) void {
    var sbuf: [16]u8 = undefined;
    const n = r.intRangeAtMost(usize, 10, 60);
    for (0..n) |_| {
        const reg = r.uintLessThan(u8, 4);
        const src = r.uintLessThan(u8, 4);
        const key = if (r.uintLessThan(u8, 8) == 0) randomBytes(r, &sbuf) else mutation_keys[r.uintLessThan(usize, mutation_keys.len)];
        const idx: u64 = if (r.uintLessThan(u8, 20) == 0) r.int(u64) else r.uintLessThan(u64, 5);
        switch (r.uintLessThan(u8, 36)) {
            0 => {
                p.op(.set_null);
                p.u8_(reg);
            },
            1 => {
                p.op(.set_bool);
                p.u8_(reg);
                p.u8_(r.int(u1));
            },
            2 => {
                p.op(.set_int);
                p.u8_(reg);
                p.u64_(if (r.boolean()) r.int(u64) else @bitCast(@as(i64, r.intRangeAtMost(i64, -5, 5))));
            },
            3 => {
                p.op(.set_uint);
                p.u8_(reg);
                p.u64_(if (r.boolean()) r.int(u64) else r.uintLessThan(u64, 5));
            },
            4 => {
                p.op(.set_double);
                p.u8_(reg);
                p.u64_(randomDoubleBits(r));
            },
            5 => {
                p.op(.set_float);
                p.u8_(reg);
                p.u32_(r.int(u32));
            },
            6, 7 => {
                p.op(.set_string);
                p.u8_(reg);
                p.str(randomBytes(r, &sbuf));
            },
            8 => {
                p.op(.set_array);
                p.u8_(reg);
            },
            9 => {
                p.op(.set_object);
                p.u8_(reg);
            },
            10 => {
                p.op(.dump);
                p.u8_(reg);
                p.i32_(r.intRangeAtMost(i32, -1, 3));
            },
            11 => {
                p.op(.dump_safe);
                p.u8_(reg);
                p.i32_(r.intRangeAtMost(i32, -1, 3));
            },
            12 => {
                p.op(.dump_raw);
                p.u8_(reg);
                p.i32_(r.intRangeAtMost(i32, -1, 2));
                p.u8_(r.int(u1));
                p.u8_(r.uintLessThan(u8, 3));
                p.u8_(' ');
            },
            13 => {
                p.op(.query);
                p.u8_(reg);
            },
            14 => {
                p.op(.get);
                p.u8_(reg);
                p.u8_(r.uintLessThan(u8, n_get_types));
            },
            15 => {
                p.op(.at_key);
                p.u8_(src);
                p.u8_(reg);
                p.str(key);
            },
            16 => {
                p.op(.at_idx);
                p.u8_(src);
                p.u8_(reg);
                p.u64_(idx);
            },
            17, 18 => {
                p.op(.index_key);
                p.u8_(reg);
                p.str(key);
                p.u8_(src);
            },
            19 => {
                p.op(.index_idx);
                p.u8_(reg);
                p.u64_(r.uintLessThan(u64, 6));
                p.u8_(src);
            },
            20 => {
                p.op(.contains);
                p.u8_(reg);
                p.str(key);
            },
            21 => {
                p.op(.value);
                p.u8_(reg);
                p.str(key);
                p.u8_(r.uintLessThan(u8, n_value_types));
            },
            22 => {
                p.op(.front);
                p.u8_(src);
                p.u8_(reg);
            },
            23 => {
                p.op(.back);
                p.u8_(src);
                p.u8_(reg);
            },
            24 => {
                p.op(.clear);
                p.u8_(reg);
            },
            25 => {
                p.op(.erase_key);
                p.u8_(reg);
                p.str(key);
            },
            26 => {
                p.op(.erase_idx);
                p.u8_(reg);
                p.u64_(idx);
            },
            27, 28 => {
                p.op(.push);
                p.u8_(reg);
                p.u8_(src);
            },
            29 => {
                p.op(.insert);
                p.u8_(reg);
                p.u8_(src);
            },
            30, 31 => {
                p.op(.set_item);
                p.u8_(reg);
                p.str(key);
                p.u8_(src);
            },
            32 => {
                p.op(.assign);
                p.u8_(reg);
                p.u8_(src);
            },
            33 => {
                p.op(.eq);
                p.u8_(reg);
                p.u8_(src);
            },
            34 => {
                p.op(.iter);
                p.u8_(reg);
            },
            else => if (r.boolean()) {
                p.op(.push_obj);
                p.u8_(reg);
                p.str(key);
                p.u8_(src);
            } else {
                p.op(.const_index);
                p.u8_(src);
                p.u8_(reg);
                p.str(key);
            },
        }
    }
    for (0..4) |reg| readAll(p, @intCast(reg));
    // compare every pair, both ways
    for (0..4) |x| for (0..4) |y| {
        p.op(.eq);
        p.u8_(@intCast(x));
        p.u8_(@intCast(y));
    };
}

/// The handwritten texts: each of the lexer's and parser's messages, and
/// the quirks they show.
const fixed_texts = [_][]const u8{
    "",                     " ",                                 "\x00",                        "1\x00garbage",                   "[1, tru]",                    "[1,\n tru\n]",       "nul",                             "fals",
    "tru",                  "truex",                             "{\"a\" 1}",                   "{\"a\":1,}",                     "[1,]",                        "[,1]",               "{,}",                             "{1:2}",
    "1 2",                  "\"\\x\"",                           "\"\\u12\"",                   "\"\\u12G4\"",                    "\"\\ud800\"",                 "\"\\ud800\\u0041\"", "\"\\ud800x\"",                    "\"\\udc00\"",
    "\"abc",                "\"a\tb\"",                          "\"a\x01b\"",                  "\"a\x1fb\"",                     "\"\x80\"",                    "\"\xc0\xaf\"",       "\"\xe0\x80\x80\"",                "\"\xed\xa0\x80\"",
    "\"\xf0\x80\x80\x80\"", "\"\xf4\x90\x80\x80\"",              "\"\xf5\"",                    "\"\xc3\"",                       "\"\xe2\x82\"",                "-",                  "-a",                              "1.",
    "1.e5",                 "1e",                                "1e+",                         "1e+a",                           "01",                          "-01",                ".5",                              "+1",
    "\xEF\xBB\xBF{}",       "\xEF\xBB\xBF",                      "\xEF\xBB",                    "\xEF",                           " \xEF\xBB\xBF1",              "[\n\n\n1,\n\n]",     "{\"a\":{\"b\":[1,{\"c\":}]}}",    "\n\n\n",
    "[1\n,\r\n2]x",         "{\"a\":1,\"a\":2,\"b\":3,\"a\":4}", "{\"a\":[1],\"a\":{\"x\":1}}", "\"\\u0000\"",                    "[\"\\u001f\\u007f\\u0080\"]", "\"\\uD83D\\uDE00\"", "{\"\":0}",                        "[[[[[]]]]]",
    "1e400",                "-1e400",                            "[1e400]",                     "{\"x\":1e999}",                  "123abc",                      "[1 2]",              "{\"a\":1 \"b\":2}",               "]",
    "}",                    ":",                                 ",",                           "\"\\/\\b\\f\\n\\r\\t\\\"\\\\\"", "\t\r\n 7 \t\r\n",             "[true,false,null]",  "{\"\xc3\xa9\":\"\xe2\x82\xac\"}", "\"\\u00e9\\u20AC\"",
};

pub fn main() !void {
    const gpa = std.heap.c_allocator;
    var prng = std.Random.DefaultPrng.init(0x15005eed);
    const r = prng.random();
    var p: Prog = .{ .gpa = gpa };
    var g: Gen = .{ .r = r, .gpa = gpa };

    // the handwritten texts
    for (fixed_texts) |t| {
        p.reset();
        p.op(.parse);
        p.u8_(0);
        p.str(t);
        p.op(.parse_nt);
        p.u8_(1);
        p.str(t);
        readAll(&p, 0);
        readAll(&p, 1);
        compare(gpa, "fixed", p.buf.items);
    }

    // documents, valid and broken
    for (0..12000) |i| {
        _ = g.document();
        if (i % 2 == 1) g.mangle();
        p.reset();
        p.op(.parse);
        p.u8_(0);
        p.str(g.out.items);
        p.op(.parse_nt);
        p.u8_(1);
        p.str(g.out.items);
        readAll(&p, 0);
        p.op(.dump);
        p.u8_(1);
        p.i32_(-1);
        compare(gpa, if (i % 2 == 1) "malformed" else "documents", p.buf.items);
    }

    // numbers through the lexer
    for (edge_numbers) |t| {
        p.reset();
        for ([_][]const u8{ "", "-", " ", "[" }) |pre| {
            var buf: [128]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{s}{s}{s}", .{ pre, t, if (pre.len == 1 and pre[0] == '[') "]" else "" }) catch unreachable;
            p.op(.parse);
            p.u8_(0);
            p.str(s);
            readAll(&p, 0);
        }
        compare(gpa, "numbers", p.buf.items);
    }
    for (0..400) |_| {
        p.reset();
        for (0..100) |_| {
            g.out.clearRetainingCapacity();
            g.number();
            p.op(.parse);
            p.u8_(0);
            p.str(g.out.items);
            p.op(.query);
            p.u8_(0);
            p.op(.dump);
            p.u8_(0);
            p.i32_(-1);
            for ([_]u8{ 1, 2, 5, 6, 7, 8, 16 }) |t| {
                p.op(.get);
                p.u8_(0);
                p.u8_(t);
            }
        }
        compare(gpa, "numbers", p.buf.items);
    }

    // doubles by their bits
    for (0..200) |_| {
        p.reset();
        for (0..10000) |_| {
            if (r.uintLessThan(u8, 8) == 0) {
                p.op(.set_float);
                p.u8_(0);
                p.u32_(r.int(u32));
            } else {
                p.op(.set_double);
                p.u8_(0);
                p.u64_(randomDoubleBits(r));
            }
            p.op(.dump);
            p.u8_(0);
            p.i32_(-1);
        }
        compare(gpa, "doubles", p.buf.items);
    }

    // mutations
    for (0..20000) |_| {
        p.reset();
        mutationProgram(&p, r);
        compare(gpa, "mutations", p.buf.items);
    }

    if (failures != 0) {
        std.debug.print("FAIL: {d} of {d} checks differ\n", .{ failures, checks });
        std.process.exit(1);
    }
    var it = kinds.iterator();
    while (it.next()) |e| std.debug.print("{d:>9}  {s}\n", .{ e.value_ptr.*, e.key_ptr.* });
    std.debug.print("PASS: {d} checks ({d} errors, {d} dumps, {d} skipped, {d} ub), cli/json identical to common_json over nlohmann\n", .{ checks, n_errors, n_dumps, n_skips, n_ub });
}
