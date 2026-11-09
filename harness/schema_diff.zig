//! `make schema-diff`: `cli/json/schema.zig` and
//! `cli/json/schema_to_grammar.zig` against upstream's
//! `json-schema.cpp` and `json-schema-to-grammar.cpp`.
//!
//! **Not a port.** Both sides run the same bytecode: `harness/schema_ref.cpp`
//! interprets it over the C++ compiled unmodified, this file over the port,
//! and every record must be byte-identical -- each grammar, the class and
//! `what()` of every exception (`IA:` `std::invalid_argument`, `RE:`
//! `std::runtime_error`, `JE:` `common_json_error`), the warning line
//! `check_errors` writes to stderr, the schema tree read back through
//! every accessor of `json-schema.h`, and every name `build_grammar`'s
//! builder hands back.
//!
//! The inputs: every case of upstream's `tests/test-json-schema-to-grammar.cpp`,
//! taken from the test itself (`ref_schema_upstream`) rather than copied;
//! those cases mutated; random schemas over every keyword the builder reads,
//! malformed ones included; an integer-bound sweep; a pattern sweep; random
//! literals; and random `build_grammar` scripts.
//!
//! The reference runs in forked children, a batch at a time: an `allOf`
//! over a `$ref` whose target is a `$ref` back to itself recurses without
//! bound in `_visit_all_of`'s `add_component`, upstream and port alike. A
//! batch whose child dies is rerun one case per child; a case that kills
//! the reference is skipped on both sides and counted.
//!
//! Usage: schema_diff [case substring]
//!   SCHEMA_DIFF_SEED=n    the generators' seed (default 1)
//!   SCHEMA_DIFF_SCALE=n   multiplies every generated count (default 1)
//!   SCHEMA_DIFF_STATS=1   lists every distinct error and warning seen

const std = @import("std");
const builtin = @import("builtin");
const win_child = @import("win_child.zig");
const json = @import("json");
const schema = json.schema;
const s2g = json.schema_to_grammar;

extern fn ref_schema_run(prog: [*]const u8, len: usize, out_len: *usize) [*]u8;
extern fn ref_schema_upstream(out_len: *usize) [*]u8;

extern "c" fn free(p: ?*anyopaque) void;
extern "c" fn fork() c_int;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

const gpa = std.heap.smp_allocator;

const OP_CASE: u8 = 0;
const OP_J2G: u8 = 1;
const OP_DOC: u8 = 2;
const OP_BUILD: u8 = 3;
const OP_LIT: u8 = 4;
const STEP_RULE: u8 = 0;
const STEP_SCHEMA: u8 = 1;

fn oom() noreturn {
    std.debug.print("FAIL: out of memory\n", .{});
    std.process.exit(2);
}

// -----------------------------------------------------------------------------
// Programs

const Prog = struct {
    buf: std.ArrayList(u8) = .empty,

    fn u8_(self: *Prog, v: u8) void {
        self.buf.append(gpa, v) catch oom();
    }
    fn u32_(self: *Prog, v: u32) void {
        self.buf.appendSlice(gpa, std.mem.asBytes(&v)) catch oom();
    }
    fn str(self: *Prog, s: []const u8) void {
        self.u32_(@intCast(s.len));
        self.buf.appendSlice(gpa, s) catch oom();
    }
};

const Case = struct {
    label: []const u8,
    /// the schema or literal the case is about, for a failure report
    input: []const u8,
    prog: []u8,
};

var cases: std.ArrayList(Case) = .empty;

fn beginCase(p: *Prog) void {
    p.u8_(OP_CASE);
    p.u32_(@intCast(cases.items.len));
}

fn addCase(label: []const u8, input: []const u8, p: *Prog) void {
    cases.append(gpa, .{
        .label = label,
        .input = gpa.dupe(u8, input) catch oom(),
        .prog = p.buf.toOwnedSlice(gpa) catch oom(),
    }) catch oom();
}

/// A schema run both ways: through the JSON entry point and through a
/// parsed document read back node by node.
fn schemaCase(label: []const u8, text: []const u8) void {
    var p: Prog = .{};
    beginCase(&p);
    p.u8_(OP_J2G);
    p.str(text);
    p.u8_(1);
    p.u8_(OP_DOC);
    p.str(text);
    addCase(label, text, &p);
}

fn literalCase(s: []const u8) void {
    var p: Prog = .{};
    beginCase(&p);
    p.u8_(OP_LIT);
    p.str(s);
    addCase("literal", s, &p);
}

const Step = union(enum) {
    rule: struct { name: []const u8, rule: []const u8 },
    schema: struct { name: []const u8, path: []const u32 },
};

fn buildCase(label: []const u8, text: []const u8, dotall: bool, steps: []const Step) void {
    var p: Prog = .{};
    beginCase(&p);
    p.u8_(OP_BUILD);
    p.str(text);
    p.u8_(@intFromBool(dotall));
    p.u32_(@intCast(steps.len));
    for (steps) |s| switch (s) {
        .rule => |r| {
            p.u8_(STEP_RULE);
            p.str(r.name);
            p.str(r.rule);
        },
        .schema => |sc| {
            p.u8_(STEP_SCHEMA);
            p.str(sc.name);
            p.u32_(@intCast(sc.path.len));
            for (sc.path) |k| p.u32_(k);
        },
    };
    addCase(label, text, &p);
}

// -----------------------------------------------------------------------------
// The port's interpreter: harness/schema_ref.cpp's `run_one`, over the port

const Reader = struct {
    p: []const u8,
    i: usize = 0,

    fn u8_(self: *Reader) u8 {
        self.i += 1;
        return self.p[self.i - 1];
    }
    fn u32_(self: *Reader) u32 {
        const v = std.mem.readInt(u32, self.p[self.i..][0..4], .little);
        self.i += 4;
        return v;
    }
    fn str(self: *Reader) []const u8 {
        const n = self.u32_();
        const s = self.p[self.i..][0..n];
        self.i += n;
        return s;
    }
};

const Records = std.ArrayList([]u8);

fn rec(out: *Records, parts: []const []const u8) void {
    out.append(gpa, std.mem.concat(gpa, u8, parts) catch oom()) catch oom();
}

var warning_buf: std.ArrayList(u8) = .empty;

fn captureWarning(line: []const u8) void {
    warning_buf.appendSlice(gpa, line) catch oom();
}

fn guardText(err: schema.Error) []const u8 {
    return switch (err) {
        error.InvalidArgument => "IA:",
        error.RuntimeError => "RE:",
        error.JsonError => "JE:",
        error.OutOfMemory => oom(),
    };
}

fn guardResult(out: *Records, res: schema.Error![]u8) void {
    if (res) |g| {
        defer gpa.free(g);
        rec(out, &.{ "G:", g });
    } else |e| rec(out, &.{ guardText(e), schema.what() });
}

/// `children` in harness/schema_ref.cpp.
fn children(n: *const schema.Node, buf: *std.ArrayList(*const schema.Node)) void {
    buf.clearRetainingCapacity();
    switch (n.*) {
        .any_of, .all_of, .tuple => |kids| for (kids) |k| buf.append(gpa, k) catch oom(),
        .array => |arr| buf.append(gpa, arr.items) catch oom(),
        .object => |obj| {
            for (obj.properties) |p| buf.append(gpa, p.schema) catch oom();
            if (obj.additional_properties) |ap| buf.append(gpa, ap) catch oom();
        },
        else => {},
    }
}

/// `describe` in harness/schema_ref.cpp.
fn describe(n: *const schema.Node, depth: usize, out: *std.ArrayList(u8)) void {
    const w = struct {
        fn s(o: *std.ArrayList(u8), t: []const u8) void {
            o.appendSlice(gpa, t) catch oom();
        }
        fn f(o: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) void {
            o.print(gpa, fmt, args) catch oom();
        }
    };
    out.appendNTimes(gpa, ' ', depth) catch oom();
    w.s(out, schema.kindName(n.kind()));
    // a $ref chain past COMMON_RECURSIVE_LIMIT throws out of both questions
    if (n.valueTypes()) |types| {
        w.s(out, " types=");
        for (0..7) |ti| {
            const t: schema.ValueType = @fromBackingInt(@intCast(ti));
            if (types.has(t)) {
                w.s(out, schema.typeName(t));
                w.s(out, ",");
            }
        }
        w.f(out, " empty={d}", .{@intFromBool(types.empty())});
    } else |_| {
        w.s(out, " types!");
        w.s(out, schema.what());
    }
    if (n.mayBeString()) |str| {
        w.f(out, " str={d}", .{@intFromBool(str)});
    } else |_| {
        w.s(out, " str!");
        w.s(out, schema.what());
    }
    switch (n.*) {
        .ref => |r| {
            w.s(out, " ref=");
            w.s(out, r.ref);
            w.s(out, " target=");
            w.s(out, if (r.target) |t| schema.kindName(t.kind()) else "null");
        },
        .@"const" => |*v| {
            const d = json.dumpSafe(gpa, v, -1) catch oom();
            defer gpa.free(d);
            w.s(out, " value=");
            w.s(out, d);
        },
        .@"enum" => |vs| {
            w.s(out, " values=");
            for (vs) |*v| {
                const d = json.dumpSafe(gpa, v, -1) catch oom();
                defer gpa.free(d);
                w.s(out, d);
                w.s(out, ",");
            }
        },
        .integer => |i| w.f(out, " min={d} max={d}", .{ i.minimum, i.maximum }),
        .string => |str| {
            w.f(out, " format={d} min={d} max={d} pattern=", .{ @backingInt(str.format), str.min_length, str.max_length });
            w.s(out, str.pattern);
        },
        .array => |arr| w.f(out, " min={d} max={d}", .{ arr.min_items, arr.max_items }),
        .object => |obj| {
            for (obj.properties) |p| {
                w.s(out, " prop=");
                w.s(out, p.name);
                if (p.required) w.s(out, "!");
            }
            w.s(out, if (obj.additional_properties != null) " additional" else " closed");
        },
        else => {},
    }
    w.s(out, "\n");
    var kids: std.ArrayList(*const schema.Node) = .empty;
    defer kids.deinit(gpa);
    children(n, &kids);
    for (kids.items) |k| describe(k, depth + 1, out);
}

fn select(root: *const schema.Node, path: []const u32) *const schema.Node {
    var s = root;
    var kids: std.ArrayList(*const schema.Node) = .empty;
    defer kids.deinit(gpa);
    for (path) |k| {
        children(s, &kids);
        if (kids.items.len == 0) break;
        s = kids.items[k % kids.items.len];
    }
    return s;
}

const BuildCtx = struct {
    steps: []const Step,
    doc: ?*const schema.Document,
    names: *std.ArrayList([]u8),

    fn run(self: *const BuildCtx, b: *s2g.GrammarBuilder) schema.Error!void {
        for (self.steps) |s| switch (s) {
            .rule => |r| {
                const n = try b.addRule(r.name, r.rule);
                self.names.append(gpa, gpa.dupe(u8, n) catch oom()) catch oom();
            },
            .schema => |sc| if (self.doc) |d| {
                const n = try b.addSchema(sc.name, select(d.root, sc.path));
                self.names.append(gpa, gpa.dupe(u8, n) catch oom()) catch oom();
            },
        };
    }
};

fn runPort(prog: []const u8, out: *Records) void {
    var r: Reader = .{ .p = prog };
    while (r.i < prog.len) {
        switch (r.u8_()) {
            OP_CASE => {
                const id = r.u32_();
                out.append(gpa, std.fmt.allocPrint(gpa, "#{d}", .{id}) catch oom()) catch oom();
            },
            OP_J2G => {
                const text = r.str();
                const force = r.u8_() != 0;
                var v = json.parse(gpa, text) catch |e| switch (e) {
                    error.JsonError => {
                        rec(out, &.{ "P:", json.what() });
                        continue;
                    },
                    error.OutOfMemory => oom(),
                };
                defer v.deinit(gpa);
                warning_buf.clearRetainingCapacity();
                guardResult(out, s2g.jsonSchemaToGrammar(gpa, &v, force));
                rec(out, &.{ "W:", warning_buf.items });
            },
            OP_DOC => {
                const text = r.str();
                var v = json.parse(gpa, text) catch |e| switch (e) {
                    error.JsonError => {
                        rec(out, &.{ "P:", json.what() });
                        continue;
                    },
                    error.OutOfMemory => oom(),
                };
                defer v.deinit(gpa);
                var doc = schema.fromJsonSchema(gpa, &v) catch |e| {
                    rec(out, &.{ guardText(e), schema.what() });
                    continue;
                };
                defer doc.deinit();
                var tree: std.ArrayList(u8) = .empty;
                defer tree.deinit(gpa);
                describe(doc.root, 0, &tree);
                for (doc.refs) |e| {
                    tree.print(gpa, "refs[{s}]={s}\n", .{ e.key, schema.kindName(e.node.kind()) }) catch oom();
                }
                rec(out, &.{ "G:", tree.items });
                warning_buf.clearRetainingCapacity();
                guardResult(out, s2g.documentToGrammar(gpa, &doc));
                rec(out, &.{ "W:", warning_buf.items });
            },
            OP_BUILD => {
                const text = r.str();
                const dotall = r.u8_() != 0;
                const n = r.u32_();
                const steps = gpa.alloc(Step, n) catch oom();
                defer {
                    for (steps) |s| if (s == .schema) gpa.free(s.schema.path);
                    gpa.free(steps);
                }
                for (steps) |*s| {
                    const kind = r.u8_();
                    const name = r.str();
                    if (kind == STEP_RULE) {
                        s.* = .{ .rule = .{ .name = name, .rule = r.str() } };
                    } else {
                        const path = gpa.alloc(u32, r.u32_()) catch oom();
                        for (path) |*k| k.* = r.u32_();
                        s.* = .{ .schema = .{ .name = name, .path = path } };
                    }
                }
                var v: json.Value = .null;
                defer v.deinit(gpa);
                var doc: ?schema.Document = null;
                defer if (doc) |*d| d.deinit();
                if (text.len != 0) {
                    v = json.parse(gpa, text) catch |e| switch (e) {
                        error.JsonError => {
                            rec(out, &.{ "JE:", json.what() });
                            continue;
                        },
                        error.OutOfMemory => oom(),
                    };
                    doc = schema.fromJsonSchema(gpa, &v) catch |e| {
                        rec(out, &.{ guardText(e), schema.what() });
                        continue;
                    };
                    rec(out, &.{"G:"});
                }
                var names: std.ArrayList([]u8) = .empty;
                defer {
                    for (names.items) |x| gpa.free(x);
                    names.deinit(gpa);
                }
                const ctx: BuildCtx = .{ .steps = steps, .doc = if (doc) |*d| d else null, .names = &names };
                warning_buf.clearRetainingCapacity();
                const res = s2g.buildGrammar(gpa, .{ .dotall = dotall }, &ctx, BuildCtx.run);
                for (names.items) |x| rec(out, &.{ "S:", x });
                guardResult(out, res);
                rec(out, &.{ "W:", warning_buf.items });
            },
            OP_LIT => {
                const s = r.str();
                const g = s2g.gbnfFormatLiteral(gpa, s) catch oom();
                defer gpa.free(g);
                rec(out, &.{g});
            },
            else => unreachable,
        }
    }
}

// -----------------------------------------------------------------------------
// The reference, in forked children

fn splitRecords(buf: []const u8, out: *Records) void {
    var i: usize = 0;
    while (i < buf.len) {
        const n = std.mem.readInt(u32, buf[i..][0..4], .little);
        out.append(gpa, gpa.dupe(u8, buf[i + 4 ..][0..n]) catch oom()) catch oom();
        i += 4 + n;
    }
}

/// Runs `prog` through the reference in a child. Return: the records, or
/// null when the child died.
fn runRefForked(prog: []const u8, out: *Records) bool {
    if (comptime builtin.os.tag == .windows) {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        if (!win_child.run("schema", prog, &buf, 0)) return false;
        splitRecords(buf.items, out);
        return true;
    }
    var fds: [2]c_int = undefined;
    if (pipe(&fds) != 0) oom();
    const pid = fork();
    if (pid < 0) oom();
    if (pid == 0) {
        _ = close(fds[0]);
        var len: usize = 0;
        const p = ref_schema_run(prog.ptr, prog.len, &len);
        var off: usize = 0;
        while (off < len) {
            const k = write(fds[1], p + off, len - off);
            if (k <= 0) _exit(3);
            off += @intCast(k);
        }
        _exit(0);
    }
    _ = close(fds[1]);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    var chunk: [65536]u8 = undefined;
    while (true) {
        const k = read(fds[0], &chunk, chunk.len);
        if (k <= 0) break;
        buf.appendSlice(gpa, chunk[0..@intCast(k)]) catch oom();
    }
    _ = close(fds[0]);
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    if (status != 0) return false;
    splitRecords(buf.items, out);
    return true;
}

// -----------------------------------------------------------------------------
// Generators

var prng: std.Random.DefaultPrng = undefined;

fn rnd() std.Random {
    return prng.random();
}

fn below(n: usize) usize {
    return rnd().uintLessThan(usize, n);
}

fn chance(pct: usize) bool {
    return below(100) < pct;
}

fn pick(comptime T: type, xs: []const T) T {
    return xs[below(xs.len)];
}

const Text = std.ArrayList(u8);

fn put(t: *Text, s: []const u8) void {
    t.appendSlice(gpa, s) catch oom();
}

fn putf(t: *Text, comptime fmt: []const u8, args: anytype) void {
    t.print(gpa, fmt, args) catch oom();
}

/// A JSON string literal of `s`, which must be valid UTF-8.
fn putStr(t: *Text, s: []const u8) void {
    put(t, "\"");
    for (s) |c| switch (c) {
        '"' => put(t, "\\\""),
        '\\' => put(t, "\\\\"),
        0...0x1f => putf(t, "\\u{x:0>4}", .{c}),
        else => t.append(gpa, c) catch oom(),
    };
    put(t, "\"");
}

const names_pool = [_][]const u8{
    "a",      "b",     "c",     "name",   "age",       "*",                "",        "kv",      "root", "char",
    "string", "value", "a-b",   "a b",    "a.b",       "\u{65e5}\u{672c}", "a\"b",    "a]b",     "a\\b", "\x00x",
    "x\ny",   "dot",   "space", "a-kv",   "a-rest",    "item",             "tuple-0", "number",  "date", "uuid",
    "-",      "ab",    "abc",   "\u{e9}", "\u{1f600}", "additional",       "1",       "integer", "null", "boolean",
    // upper case, so a rule order that ignored case would show
    "A",      "B",     "Name",  "aB",     "Ab",        "ZZ",               "a_B",     "Zeta",    "Root", "Char",
};

const type_names = [_][]const u8{
    "string", "number", "integer", "boolean", "null", "object", "array", "", "date", "int", "Object", "any",
};

/// A type name: one of the seven, nearly always.
fn typeName() []const u8 {
    return if (chance(92)) type_names[below(7)] else pick([]const u8, &type_names);
}

const formats = [_][]const u8{
    "date", "time", "date-time", "uuid", "uuid1", "uuid3", "uuid5", "uuid0", "uuid6", "uuidx", "uuid12", "", "email", "DATE",
};

fn putName(t: *Text) void {
    putStr(t, pick([]const u8, &names_pool));
}

/// A count as `get_count` reads it: mostly fine, sometimes not.
fn putCount(t: *Text) void {
    switch (below(14)) {
        0...6 => putf(t, "{d}", .{below(6)}),
        7 => put(t, "-1"),
        8 => put(t, "1.5"),
        9 => put(t, "\"2\""),
        10 => put(t, "true"),
        11 => put(t, "4294967297"),
        12 => put(t, pick([]const u8, &.{ "2147483648", "18446744073709551615", "-2147483649", "2147483647" })),
        else => put(t, "null"),
    }
}

/// An integer bound: small and large, the edges of `int64_t`, an unsigned
/// one past it, fractions, and non-numbers.
fn putBound(t: *Text) void {
    switch (below(24)) {
        0...5 => putf(t, "{d}", .{@as(i64, @intCast(below(2001))) - 1000}),
        6...9 => {
            const digits = 1 + below(19);
            var v: i64 = 0;
            for (0..digits) |_| v = v *% 10 +% @as(i64, @intCast(below(10)));
            if (chance(50)) v = -%v;
            putf(t, "{d}", .{v});
        },
        10 => put(t, pick([]const u8, &.{ "0", "1", "9", "10", "99", "100", "-1", "-9", "-10", "-100", "1000000000000000" })),
        11 => put(t, pick([]const u8, &.{ "9223372036854775807", "-9223372036854775808", "-9223372036854775807", "9223372036854775806" })),
        12 => put(t, pick([]const u8, &.{ "18446744073709551615", "9223372036854775808", "10000000000000000000" })),
        13...15 => put(t, pick([]const u8, &.{ "2.5", "-2.5", "0.5", "-0.5", "1e3", "-1e3", "0.0", "-0.0", "99.999", "1.0" })),
        16 => if (builtin.cpu.arch == .aarch64)
            // (int64_t) of a double out of range: AArch64 saturates, and so
            // does the port; anywhere else it is undefined, so not asked
            put(t, pick([]const u8, &.{ "1e20", "-1e20", "9.3e18", "-9.3e18", "1e308" }))
        else
            put(t, "7"),
        17 => put(t, pick([]const u8, &.{ "\"5\"", "true", "null", "[1]", "{}" })),
        else => putf(t, "{d}", .{@as(i64, @intCast(below(41))) - 20}),
    }
}

const pattern_tokens = [_][]const u8{
    "a",            "b",              "z",             "0",              "9",             " ",     "\"",    "\u{e9}", "\x00",  "-",
    ".",            ".",              "\\d",           "\\w",            "\\s",           "\\.",   "\\x41", "\\x4",   "\\xZZ", "\\u00e9",
    "\\U0001F600",  "\\t",            "\\n",           "\\\"",           "\\[",           "\\]",   "\\-",   "\\\\",   "\\/",   "\\^",
    "\\$",          "\\(",            "\\)",           "\\|",            "\\{",           "\\*",   "\\+",   "\\?",    "[a-z]", "[^0-9]",
    "[\\]]",        "[\\d]",          "[\\x41-\\x5A]", "[",              "]",             "(",     ")",     "(?:",    "(?=",   "(?<n>",
    "|",            "*",              "+",             "?",              "{2}",           "{1,3}", "{,3}",  "{2,}",   "{}",    "{a}",
    "{1,2,3}",      "{-1}",           "{ 2}",          "{99999999999}",  "{",             "}",     "^",     "$",      "\\",    "{0}",
    "{0,1}",        "{1,}",           "{0,}",          "{3,1}",          "[.]",           "[\"]",  "\\x",   "a{2}b",  "(ab)",  "(a|b)",
    // the edges of `int`, which `std::stoi` takes and refuses
    "{2147483647}", "{0,2147483647}", "{2147483648}",  "{-2147483648,}", "{-2147483649}", "{ +3}",
};

fn genPattern(t: *Text) void {
    const anchored = chance(85);
    if (anchored) put(t, "^");
    const n = below(9);
    var depth: usize = 0;
    for (0..n) |_| {
        if (chance(8)) {
            put(t, "(");
            depth += 1;
        } else if (depth > 0 and chance(10)) {
            put(t, ")");
            depth -= 1;
        } else {
            put(t, pick([]const u8, &pattern_tokens));
        }
    }
    if (chance(80)) {
        while (depth > 0) : (depth -= 1) put(t, ")");
    }
    if (anchored or chance(10)) put(t, "$");
}

/// A random JSON value, for `const` and `enum`.
fn putValue(t: *Text, depth: usize) void {
    switch (below(if (depth > 1) 6 else 8)) {
        0 => put(t, "null"),
        1 => put(t, pick([]const u8, &.{ "true", "false" })),
        2 => putBound(t),
        3, 4 => putStr(t, pick([]const u8, &.{ "a", "", "x\"y", "back\\slash", "new\nline", "\u{e9}t\u{e9}", "-", "]", "\r", "\x00", "tab\t" })),
        5 => put(t, pick([]const u8, &.{ "1.5", "-0.0", "1e100", "3.0", "0.1" })),
        6 => {
            put(t, "[");
            for (0..below(3)) |k| {
                if (k > 0) put(t, ",");
                putValue(t, depth + 1);
            }
            put(t, "]");
        },
        else => {
            put(t, "{");
            for (0..below(3)) |k| {
                if (k > 0) put(t, ",");
                putName(t);
                put(t, ":");
                putValue(t, depth + 1);
            }
            put(t, "}");
        },
    }
}

const ref_targets = [_][]const u8{
    "#/$defs/a",      "#/$defs/b",              "#/$defs/c",           "#/$defs/missing",                  "#/definitions/a",
    "#/properties/a", "#/$defs/a/anyOf/0",      "#/$defs/arr/1",       "#/$defs/arr/-1",                   "#/$defs/arr/ 1",
    "#/$defs/arr/1x", "#/$defs/arr/99",         "#/$defs/arr/+0",      "#/$defs/arr/99999999999999999999", "#/$defs/arr/",
    "#",              "#/",                     "http://x/y.json",     "x",                                "#/$defs/a b",
    "#/$defs/a~1b",   "#/$defs/a/properties/a", "#/$defs/arr/0/items", "#/$defs/\u{e9}",                   "#/$defs/a/items",
    "#/$defs",        "#/$defs/b/allOf/0",
};

/// A random schema. `depth` bounds nesting.
fn genSchema(t: *Text, depth: usize) void {
    if (chance(if (depth == 0) 1 else 3)) {
        put(t, pick([]const u8, &.{ "true", "1", "\"x\"", "null", "[]", "[{}]" }));
        return;
    }
    if (depth > 3 or chance(10)) {
        // a leaf
        switch (below(9)) {
            0 => put(t, "{}"),
            1 => {
                put(t, "{\"type\":");
                putStr(t, typeName());
                put(t, "}");
            },
            2 => {
                put(t, "{\"$ref\":");
                if (chance(90)) putStr(t, pick([]const u8, &ref_targets)) else put(t, "5");
                put(t, "}");
            },
            3 => {
                put(t, "{\"const\":");
                putValue(t, 0);
                put(t, "}");
            },
            4 => {
                put(t, "{\"type\":\"string\",\"pattern\":");
                var p: Text = .empty;
                defer p.deinit(gpa);
                genPattern(&p);
                putStr(t, p.items);
                put(t, "}");
            },
            5 => {
                put(t, "{\"type\":\"integer\",\"minimum\":");
                putBound(t);
                put(t, "}");
            },
            6 => {
                put(t, "{\"format\":");
                putStr(t, pick([]const u8, &formats));
                put(t, "}");
            },
            7 => put(t, "{\"type\":\"object\"}"),
            else => put(t, "{\"type\":\"array\"}"),
        }
        return;
    }

    put(t, "{");
    var first = true;
    const key = struct {
        fn k(tt: *Text, f: *bool, name: []const u8) void {
            if (!f.*) put(tt, ",");
            f.* = false;
            putStr(tt, name);
            put(tt, ":");
        }
    }.k;

    // the type, sometimes a list, sometimes nonsense
    if (chance(70)) {
        key(t, &first, "type");
        switch (below(10)) {
            0...5 => putStr(t, pick([]const u8, &type_names)),
            6, 7 => {
                put(t, "[");
                for (0..below(4)) |k| {
                    if (k > 0) put(t, ",");
                    if (chance(95)) putStr(t, typeName()) else put(t, pick([]const u8, &.{ "5", "null", "[\"string\"]", "[]" }));
                }
                put(t, "]");
            },
            8 => put(t, "null"),
            else => if (chance(30)) put(t, pick([]const u8, &.{ "5", "{}", "true" })) else putStr(t, typeName()),
        }
    }

    var defs_written = false;
    const n_keys = 1 + below(5);
    for (0..n_keys) |_| {
        switch (below(26)) {
            0, 1, 2 => {
                key(t, &first, "properties");
                if (chance(5)) {
                    put(t, pick([]const u8, &.{ "[]", "1", "null" }));
                } else {
                    put(t, "{");
                    for (0..below(4)) |k| {
                        if (k > 0) put(t, ",");
                        putName(t);
                        put(t, ":");
                        genSchema(t, depth + 1);
                    }
                    put(t, "}");
                }
            },
            3, 4 => {
                key(t, &first, "required");
                if (chance(10)) {
                    put(t, pick([]const u8, &.{ "\"a\"", "true", "{}" }));
                } else {
                    put(t, "[");
                    for (0..below(4)) |k| {
                        if (k > 0) put(t, ",");
                        if (chance(90)) putName(t) else put(t, "1");
                    }
                    put(t, "]");
                }
            },
            5, 6 => {
                key(t, &first, "additionalProperties");
                switch (below(6)) {
                    0 => put(t, "true"),
                    1 => put(t, "false"),
                    2, 3 => genSchema(t, depth + 1),
                    else => put(t, pick([]const u8, &.{ "1", "\"x\"", "null", "[]" })),
                }
            },
            7, 8 => {
                key(t, &first, pick([]const u8, &.{ "items", "items", "prefixItems" }));
                if (chance(30)) {
                    put(t, "[");
                    for (0..below(4)) |k| {
                        if (k > 0) put(t, ",");
                        genSchema(t, depth + 1);
                    }
                    put(t, "]");
                } else genSchema(t, depth + 1);
            },
            9 => {
                key(t, &first, pick([]const u8, &.{ "minItems", "maxItems" }));
                putCount(t);
            },
            10 => {
                key(t, &first, pick([]const u8, &.{ "minLength", "maxLength" }));
                putCount(t);
            },
            11 => {
                key(t, &first, "pattern");
                if (chance(95)) {
                    var p: Text = .empty;
                    defer p.deinit(gpa);
                    genPattern(&p);
                    putStr(t, p.items);
                } else put(t, "5");
            },
            12 => {
                key(t, &first, "format");
                if (chance(95)) putStr(t, pick([]const u8, &formats)) else put(t, "1");
            },
            13, 14 => {
                key(t, &first, pick([]const u8, &.{ "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum" }));
                putBound(t);
            },
            15 => {
                key(t, &first, "const");
                putValue(t, 0);
            },
            16, 17 => {
                key(t, &first, "enum");
                if (chance(10)) {
                    put(t, pick([]const u8, &.{ "[]", "1", "\"a\"" }));
                } else {
                    put(t, "[");
                    for (0..1 + below(4)) |k| {
                        if (k > 0) put(t, ",");
                        putValue(t, 0);
                    }
                    put(t, "]");
                }
            },
            18, 19, 20 => {
                key(t, &first, pick([]const u8, &.{ "anyOf", "oneOf", "allOf", "allOf" }));
                if (chance(8)) {
                    put(t, pick([]const u8, &.{ "[]", "{}", "1" }));
                } else {
                    put(t, "[");
                    for (0..1 + below(3)) |k| {
                        if (k > 0) put(t, ",");
                        genSchema(t, depth + 1);
                    }
                    put(t, "]");
                }
            },
            21 => {
                key(t, &first, "$ref");
                if (chance(95)) putStr(t, pick([]const u8, &ref_targets)) else put(t, pick([]const u8, &.{ "5", "null", "[\"#/$defs/a\"]" }));
            },
            22, 23 => if (depth == 0 and !defs_written) {
                defs_written = true;
                key(t, &first, pick([]const u8, &.{ "$defs", "$defs", "definitions" }));
                putDefs(t);
            },
            24 => {
                key(t, &first, pick([]const u8, &.{ "description", "title", "$schema", "unknownKeyword", "default" }));
                putValue(t, 0);
            },
            else => {
                key(t, &first, "type");
                putStr(t, typeName());
            },
        }
    }
    if (depth == 0 and !defs_written and chance(70)) {
        key(t, &first, "$defs");
        putDefs(t);
    }
    put(t, "}");
}

/// The `$defs` the generated `$ref`s point into.
fn putDefs(t: *Text) void {
    put(t, "{");
    var k: usize = 0;
    for ([_][]const u8{ "a", "b", "c", "arr", "a b", "a~1b", "\u{e9}" }) |name| {
        if (!chance(if (name.len == 1 or std.mem.eql(u8, name, "arr")) 85 else 30)) continue;
        if (k > 0) put(t, ",");
        k += 1;
        putStr(t, name);
        put(t, ":");
        if (std.mem.eql(u8, name, "arr")) {
            put(t, "[");
            for (0..below(3)) |j| {
                if (j > 0) put(t, ",");
                genSchema(t, 2);
            }
            put(t, "]");
        } else genSchema(t, 1);
    }
    put(t, "}");
}

/// Mutates a parsed schema in place: replace, delete or insert at a random
/// node.
fn mutate(v: *json.Value, depth: usize) void {
    switch (v.*) {
        .object => |*m| {
            const n = m.members.items.len;
            if (n != 0 and depth < 6 and chance(60)) {
                mutate(&m.members.items[below(n)].value, depth + 1);
                return;
            }
            switch (below(4)) {
                0 => if (n != 0) {
                    const k = m.members.items[below(n)].key;
                    const kk = gpa.dupe(u8, k) catch oom();
                    defer gpa.free(kk);
                    _ = m.erase(gpa, kk);
                },
                1 => if (n != 0) {
                    var t: Text = .empty;
                    defer t.deinit(gpa);
                    putValue(&t, 0);
                    const nv = json.parse(gpa, t.items) catch unreachable;
                    var slot = &m.members.items[below(n)].value;
                    slot.deinit(gpa);
                    slot.* = nv;
                },
                else => {
                    var t: Text = .empty;
                    defer t.deinit(gpa);
                    const kw = pick([]const u8, &.{ "type", "minimum", "maximum", "pattern", "format", "required", "additionalProperties", "items", "enum", "const", "$ref", "allOf", "anyOf", "minItems", "maxLength", "exclusiveMinimum" });
                    if (std.mem.eql(u8, kw, "$ref")) putStr(&t, pick([]const u8, &ref_targets)) else if (chance(50)) genSchema(&t, 3) else putValue(&t, 0);
                    const nv = json.parse(gpa, t.items) catch unreachable;
                    v.set(gpa, kw, nv) catch oom();
                },
            }
        },
        .array => |*a| {
            if (a.items.len != 0 and chance(70)) {
                mutate(&a.items[below(a.items.len)], depth + 1);
            } else if (a.items.len != 0 and chance(50)) {
                var x = a.orderedRemove(below(a.items.len));
                x.deinit(gpa);
            } else {
                var t: Text = .empty;
                defer t.deinit(gpa);
                genSchema(&t, 3);
                a.append(gpa, json.parse(gpa, t.items) catch unreachable) catch oom();
            }
        },
        else => {
            var t: Text = .empty;
            defer t.deinit(gpa);
            genSchema(&t, 3);
            v.deinit(gpa);
            v.* = json.parse(gpa, t.items) catch unreachable;
        },
    }
}

const rule_names = [_][]const u8{
    "root", "",  "a",    "a",     "b",  "dot", "space", "string", "char", "value", "a b", "a-b", "x.y", "\u{65e5}", "a0", "a-0", "item", "-", "\x00", "ref-defs-a",
    "A",    "B", "Root", "Space", "aB",
};
const rule_bodies = [_][]const u8{
    "\"a\"", "\"b\"", "\"a\"", "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}", "[^\\x0A\\x0D]", "[\\U00000000-\\U0010FFFF]", "", "x y", "\"\\\"\" char* \"\\\"\"",
};

fn genBuild(texts: []const []const u8) void {
    const text: []const u8 = if (chance(15)) "" else if (chance(50)) texts[below(texts.len)] else blk: {
        var t: Text = .empty;
        genSchema(&t, 0);
        break :blk t.items;
    };
    var steps: std.ArrayList(Step) = .empty;
    defer steps.deinit(gpa);
    for (0..1 + below(6)) |_| {
        if (chance(50)) {
            steps.append(gpa, .{ .rule = .{ .name = pick([]const u8, &rule_names), .rule = pick([]const u8, &rule_bodies) } }) catch oom();
        } else {
            const path = gpa.alloc(u32, below(4)) catch oom();
            for (path) |*k| k.* = rnd().int(u32) % 8;
            steps.append(gpa, .{ .schema = .{ .name = pick([]const u8, &rule_names), .path = path } }) catch oom();
        }
    }
    buildCase("build", text, chance(50), steps.items);
}

/// `n` copies of `head`, `core`, `n` copies of `tail`.
fn nested(head: []const u8, core: []const u8, tail: []const u8, n: usize) []const u8 {
    var t: Text = .empty;
    for (0..n) |_| put(&t, head);
    put(&t, core);
    for (0..n) |_| put(&t, tail);
    return t.items;
}

/// A chain of `n` `$ref`s, listed tail first in an `anyOf` so the builder
/// reaches each target already built: its recursion stays shallow, and the
/// chain's length is left for `value_types`, `may_be_string` and `visit`.
fn refChain(n: usize, comptime wrap_all_of: bool) []const u8 {
    var t: Text = .empty;
    put(&t, if (wrap_all_of) "{\"allOf\": [" else "{\"anyOf\": [");
    var i = n + 1;
    while (i > 0) {
        i -= 1;
        putf(&t, "{{\"$ref\": \"#/$defs/a{d}\"}}", .{i});
        if (i != 0) put(&t, ", ");
    }
    put(&t, "], \"$defs\": {");
    for (0..n) |k| putf(&t, "\"a{d}\": {{\"$ref\": \"#/$defs/a{d}\"}}, ", .{ k, k + 1 });
    putf(&t, "\"a{d}\": {{\"type\": \"string\"}}}}}}", .{n});
    return t.items;
}

/// As `refChain`, the tail-first list hidden from `visit`: `_visit_all_of`
/// ignores an array component, so the chain is entered at its head, where
/// each `$ref` visits the next and only `visit`'s own guard stops it.
fn refChainVisit(n: usize) []const u8 {
    var t: Text = .empty;
    put(&t, "{\"allOf\": [{\"type\": \"array\", \"items\": {\"anyOf\": [");
    var i = n + 1;
    while (i > 0) {
        i -= 1;
        putf(&t, "{{\"$ref\": \"#/$defs/a{d}\"}}", .{i});
        if (i != 0) put(&t, ", ");
    }
    put(&t, "]}}, {\"type\": \"object\", \"properties\": {\"x\": {\"$ref\": \"#/$defs/a0\"}}}], \"$defs\": {");
    for (0..n) |k| putf(&t, "\"a{d}\": {{\"$ref\": \"#/$defs/a{d}\"}}, ", .{ k, k + 1 });
    putf(&t, "\"a{d}\": {{\"type\": \"string\"}}}}}}", .{n});
    return t.items;
}

/// v0.8.0's recursion limits, on either side of each.
fn depthCases() void {
    for ([_]usize{ 126, 127, 128, 129, 400 }) |n| {
        const label = std.fmt.allocPrint(gpa, "depth-{d}", .{n}) catch oom();
        schemaCase(label, nested("{\"type\": \"array\", \"items\": ", "{\"type\": \"string\"}", "}", n));
        schemaCase(label, nested("{\"anyOf\": [", "{\"type\": \"string\"}", "]}", n));
        schemaCase(label, nested("{\"type\": \"object\", \"properties\": {\"k\": ", "{\"type\": \"string\"}", "}}", n));
        schemaCase(label, refChain(n, false));
        schemaCase(label, refChain(n, true));
        schemaCase(label, refChainVisit(n));

        // one level of the name's trie per char
        var t: Text = .empty;
        put(&t, "{\"type\": \"object\", \"properties\": {\"");
        t.appendNTimes(gpa, 'a', n) catch oom();
        put(&t, "\": {\"type\": \"string\"}}, \"additionalProperties\": true}");
        schemaCase(label, t.items);

        var p: Text = .empty;
        put(&p, "{\"type\": \"string\", \"pattern\": \"^");
        put(&p, nested("(", "a", ")", n));
        put(&p, "$\"}");
        schemaCase(label, p.items);
    }
}

fn handwritten() void {
    depthCases();
    const xs = [_][]const u8{
        // a ref cycle through allOf: the C++ recurses without bound
        \\{"allOf": [{"$ref": "#/$defs/a"}], "$defs": {"a": {"$ref": "#/$defs/a"}}}
        ,
        \\{"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}}
        ,
        \\{"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}}
        ,
        \\{"type": "integer", "minimum": -9223372036854775807}
        ,
        \\{"type": "integer", "maximum": -9223372036854775808}
        ,
        \\{"type": "integer", "minimum": -5, "maximum": -9223372036854775808}
        ,
        \\{"type": "integer", "exclusiveMinimum": 9223372036854775807}
        ,
        \\{"type": "integer", "exclusiveMaximum": -9223372036854775808}
        ,
        \\{"type": "integer", "minimum": 5, "maximum": 3}
        ,
        \\{"type": "integer", "minimum": 100, "maximum": 5}
        ,
        \\{"type": "integer", "minimum": 5, "maximum": -3}
        ,
        \\{"type": "string", "pattern": "^a\u0000b\\q$"}
        ,
        \\{"type": "string", "pattern": "^[\\\u0000]$"}
        ,
        \\{"type": "string", "pattern": "^(((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((a)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))$"}
        ,
        \\{"type": "object", "properties": {"*": {"type": "string"}, "a": {}}, "additionalProperties": {"type": "integer"}}
        ,
        \\{"type": "object", "properties": {"": {}, "a": {}}, "additionalProperties": true}
        ,
        \\{"allOf": [{"properties": {"a": {}}}, {"properties": {"a": {"type": "string"}}, "required": ["a"]}, {"anyOf": [{"properties": {"b": {}}}]}]}
        ,
        \\{"allOf": [{"enum": [1, 2, 2]}, {"enum": [2, 1]}]}
        ,
        \\{"allOf": [{"enum": ["b", "a"]}, {"enum": ["a", "b"]}, {"$ref": "#/$defs/e"}], "$defs": {"e": {"enum": ["b"]}}}
        ,
        \\{"type": ["string", "null", "integer"], "minimum": 3, "maxLength": 2}
        ,
        \\{"type": [["string"]]}
        ,
        \\{"type": [5], "const": 1}
        ,
        \\{"$defs": {"arr": [{"type": "string"}, {"type": "integer"}]}, "anyOf": [{"$ref": "#/$defs/arr/1"}, {"$ref": "#/$defs/arr/-0"}, {"$ref": "#/$defs/arr/ 0"}]}
        ,
        \\{"$defs": {"arr": [{"type": "string"}]}, "$ref": "#/$defs/arr/18446744073709551616"}
        ,
        \\{"$defs": {"arr": [{"type": "string"}]}, "$ref": "#/$defs/arr/-18446744073709551615"}
        ,
        \\{"items": [{"type": "string"}], "minItems": -1}
        ,
        \\{"type": "array", "items": {"type": "string"}, "minItems": 4294967297, "maxItems": 4294967298}
        ,
        \\{"type": "string", "minLength": 2147483648}
        ,
        \\{"type": "string", "pattern": "^a{2147483647}b{,2147483647}c{-2147483648,}d{2147483648}$"}
        ,
        // one edge to a pattern: `{2147483648}` makes a whole pattern unsupported
        \\{"type": "string", "pattern": "^a{2147483647}$"}
        ,
        \\{"type": "string", "pattern": "^b{,2147483647}$"}
        ,
        \\{"type": "string", "pattern": "^c{-2147483648,}$"}
        ,
        \\{"type": "string", "pattern": "^d{2147483648}$"}
        ,
        \\{"type": "string", "pattern": "^(a|b){2}(a|b){3}(a|b){2}[x]{1}[x]{1}$"}
        ,
        \\{"type": "string", "pattern": "^\"\\\"[\"]$"}
        ,
        \\{"properties": {"a": {"type": "string", "pattern": "^."}, "b": {"pattern": "a$"}, "c": {"pattern": "^(?=a)$"}}}
        ,
        \\{"type": "string", "format": "uuid5", "pattern": ""}
        ,
        \\{"type": "string", "format": "date", "pattern": "^a$"}
        ,
        \\{"additionalProperties": 1}
        ,
        \\{"additionalProperties": true, "allOf": [{}]}
        ,
        \\{"type": "object", "allOf": [{"properties": {"x": {}}}], "additionalProperties": true}
        ,
        \\{"type": "string", "allOf": [{"type": "string"}, {}]}
        ,
        \\{"oneOf": [{"type": "string"}], "anyOf": []}
        ,
        \\{"properties": {"\u65e5": {}, "\u65e5\u672c": {}, "a\u00e9": {}}, "additionalProperties": {}}
        ,
        \\{"properties": {"root": {"type": "string"}, "char": {"enum": [1]}, "space": {}}}
        ,
    };
    for (xs) |x| schemaCase("handwritten", x);
    // not JSON at all: the parse error, before any schema
    for ([_][]const u8{ "", "{", "{\"type\": }", "[1,]", "{\"a\" 1}", "nul", "\"\\x\"", "{\"type\":\"string\"} x" }) |x| {
        schemaCase("malformed", x);
    }
}

// -----------------------------------------------------------------------------
// The run

/// Phrases some reference record must contain, so a generator that drifts
/// away from an arm fails here rather than passing on nothing.
const required_phrases = [_][]const u8{
    "schema must be an object",                           "$ref must be a string",
    "unsupported $ref",                                   "is out of range",
    " not found",                                         "must be an array of schemas",
    "must not be empty",                                  "properties must be an object",
    "additionalProperties must be a boolean or a schema", "must be a non-negative integer",
    "must be a number",                                   "format must be a string",
    "pattern must be a string",                           "type must not be empty",
    "enum must be a non-empty array",                     "type must be a string or an array of strings",
    "unrecognized type",                                  "At least one of min_value or max_value must be set",
    "not anchored with '^' and '$'",                      "unsupported group syntax",
    "pattern nesting too deep",                           "unbalanced parentheses",
    "anchor inside the pattern",                          "unsupported escape in character class",
    "unterminated character class",                       "nothing to repeat",
    "unterminated curly brackets",                        "wrong number of values in curly brackets",
    "invalid number in curly brackets",                   "trailing backslash",
    "unsupported escape: ",                               "unsupported character: ",
    "WARNING: JSON schema conversion was incomplete",     "Invalid pattern",
    "JSON schema conversion failed",                      "dot ::= [\\U00000000-\\U0010FFFF]",
    "dot ::= [^\\x0A\\x0D]",                              "-additional-k ::= [\"]",
    "date-time-string ::=",                               "uuid ::=",
    "-rest ::=",                                          "P:[json.exception.parse_error",
    "JSON schema is nested more than 128 levels deep",    "JSON schema property name is longer than 128 chars",
    "types!JSON schema is nested",                        "str!JSON schema is nested",
};

fn envUsize(name: [*:0]const u8, def: usize) usize {
    const v = getenv(name) orelse return def;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch def;
}

fn printEscaped(s: []const u8) void {
    const max = 400;
    for (s[0..@min(s.len, max)]) |c| {
        if (c == '\n') std.debug.print("\\n", .{}) else if (c < 0x20 or c == 0x7f) std.debug.print("\\x{x:0>2}", .{c}) else std.debug.print("{c}", .{c});
    }
    if (s.len > max) std.debug.print("...", .{});
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    // Windows: this process may be the reference's child (win_child.zig)
    if (comptime builtin.os.tag == .windows) if (win_child.childArgs(argv)) |c| {
        var len: usize = 0;
        const p = ref_schema_run(c.input.ptr, c.input.len, &len);
        std.process.exit(if (win_child.writeFile(c.out.ptr, p[0..len])) 0 else 3);
    };
    const filter: ?[]const u8 = if (argv.len > 1) argv[1] else null;

    const seed = envUsize("SCHEMA_DIFF_SEED", 1);
    const scale = envUsize("SCHEMA_DIFF_SCALE", 1);
    prng = std.Random.DefaultPrng.init(seed);
    s2g.warning_sink = captureWarning;

    // upstream's cases, from the test itself
    var upstream_texts: std.ArrayList([]const u8) = .empty;
    {
        var len: usize = 0;
        const p = ref_schema_upstream(&len);
        defer free(p);
        var recs: Records = .empty;
        splitRecords(p[0..len], &recs);
        var k: usize = 0;
        var mismatches: usize = 0;
        while (k + 5 <= recs.items.len) : (k += 5) {
            if (!std.mem.eql(u8, recs.items[k + 4], "ok")) {
                std.debug.print("upstream case '{s}': the reference does not reproduce its expectation\n", .{recs.items[k]});
                mismatches += 1;
            }
            schemaCase(recs.items[k], recs.items[k + 1]);
            upstream_texts.append(gpa, recs.items[k + 1]) catch oom();
        }
        if (mismatches != 0 or upstream_texts.items.len == 0) {
            std.debug.print("FAIL: {d} upstream cases, {d} not reproduced by the reference\n", .{ upstream_texts.items.len, mismatches });
            std.process.exit(1);
        }
        std.debug.print("upstream: {d} cases, each reproduced by the reference\n", .{upstream_texts.items.len});
    }
    // the two document-level checks of upstream's `main`, as builds
    buildCase("upstream: parsed document",
        \\{"$ref": "#/$defs/node", "$defs": {"node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/node"}, "leaf": {}}, "additionalProperties": false}}}
    , false, &.{.{ .schema = .{ .name = "root", .path = &.{} } }});
    buildCase("upstream: sub-schema $ref",
        \\{"type": "object", "properties": {"item": {"$ref": "#/$defs/item"}}, "$defs": {"item": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}}}
    , false, &.{.{ .schema = .{ .name = "root", .path = &.{0} } }});

    handwritten();

    // upstream's cases, mutated
    for (upstream_texts.items) |text| {
        for (0..80 * scale) |_| {
            var v = json.parse(gpa, text) catch continue;
            defer v.deinit(gpa);
            for (0..1 + below(3)) |_| mutate(&v, 0);
            const out = json.dump(gpa, &v, -1) catch oom();
            schemaCase("mutated", out);
            gpa.free(out);
        }
    }

    // random schemas
    for (0..60000 * scale) |_| {
        var t: Text = .empty;
        genSchema(&t, 0);
        schemaCase("random", t.items);
        t.deinit(gpa);
    }

    // integer bounds
    for (0..30000 * scale) |_| {
        var t: Text = .empty;
        put(&t, "{\"type\":\"integer\"");
        const keys = [_][]const u8{ "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum" };
        for (keys) |kname| {
            if (!chance(45)) continue;
            put(&t, ",");
            putStr(&t, kname);
            put(&t, ":");
            if (chance(85)) {
                const digits = 1 + below(19);
                var v: i64 = 0;
                for (0..digits) |_| v = v *% 10 +% @as(i64, @intCast(below(10)));
                if (chance(40)) v = -%v;
                putf(&t, "{d}", .{v});
            } else putBound(&t);
        }
        put(&t, "}");
        schemaCase("integer", t.items);
        t.deinit(gpa);
    }

    // patterns
    for (0..40000 * scale) |_| {
        var p: Text = .empty;
        genPattern(&p);
        var t: Text = .empty;
        put(&t, "{\"type\":\"string\",\"pattern\":");
        putStr(&t, p.items);
        put(&t, "}");
        schemaCase("pattern", t.items);
        if (chance(10)) {
            // the dot under both settings
            buildCase("pattern-build", t.items, chance(50), &.{.{ .schema = .{ .name = "p", .path = &.{} } }});
        }
        t.deinit(gpa);
        p.deinit(gpa);
    }

    // literals, every byte
    for (0..5000 * scale) |_| {
        var s: [24]u8 = undefined;
        const n = below(s.len);
        for (s[0..n]) |*c| c.* = if (chance(40)) pick(u8, "\r\n\"\\-]a ") else rnd().int(u8);
        literalCase(s[0..n]);
    }

    // build_grammar scripts
    for (0..20000 * scale) |_| genBuild(upstream_texts.items);

    // -------------------------------------------------------------------------
    // run both sides

    var checks: usize = 0;
    var failures: usize = 0;
    var crashed: usize = 0;
    var n_grammars: usize = 0;
    var n_errors: usize = 0;
    var n_warnings: usize = 0;
    var n_names: usize = 0;
    var phrase_seen: [required_phrases.len]bool = @splat(false);
    var stats: std.StringHashMapUnmanaged(usize) = .empty;
    const want_stats = getenv("SCHEMA_DIFF_STATS") != null;

    const batch = 400;
    var start: usize = 0;
    while (start < cases.items.len) : (start += batch) {
        const end = @min(start + batch, cases.items.len);
        var ref_by_case = gpa.alloc(?Records, end - start) catch oom();
        defer gpa.free(ref_by_case);
        @memset(ref_by_case, null);

        var prog: std.ArrayList(u8) = .empty;
        defer prog.deinit(gpa);
        for (cases.items[start..end]) |c| {
            if (filter) |f| if (std.mem.indexOf(u8, c.label, f) == null) continue;
            prog.appendSlice(gpa, c.prog) catch oom();
        }
        var all: Records = .empty;
        if (runRefForked(prog.items, &all)) {
            var cur: ?usize = null;
            for (all.items) |r| {
                if (r.len > 0 and r[0] == '#') {
                    const id = std.fmt.parseInt(usize, r[1..], 10) catch unreachable;
                    cur = id - start;
                    ref_by_case[cur.?] = .empty;
                }
                ref_by_case[cur.?].?.append(gpa, r) catch oom();
            }
        } else {
            // a case killed the reference: one child per case
            for (cases.items[start..end], 0..) |c, k| {
                if (filter) |f| if (std.mem.indexOf(u8, c.label, f) == null) continue;
                var one: Records = .empty;
                if (runRefForked(c.prog, &one)) {
                    ref_by_case[k] = one;
                } else {
                    crashed += 1;
                    if (crashed <= 3) {
                        std.debug.print("note: the reference crashed on case {d} ({s}), skipped: ", .{ start + k, c.label });
                        printEscaped(c.input);
                        std.debug.print("\n", .{});
                    }
                }
            }
        }

        for (cases.items[start..end], 0..) |c, k| {
            const ref = ref_by_case[k] orelse continue;
            var port: Records = .empty;
            runPort(c.prog, &port);
            const n = @max(ref.items.len, port.items.len);
            for (0..n) |j| {
                const a: []const u8 = if (j < ref.items.len) ref.items[j] else "<missing>";
                const b: []const u8 = if (j < port.items.len) port.items[j] else "<missing>";
                if (a.len > 0 and a[0] == '#') continue;
                checks += 1;
                if (std.mem.startsWith(u8, a, "G:") and std.mem.indexOf(u8, a, " ::= ") != null) n_grammars += 1;
                if (std.mem.startsWith(u8, a, "S:")) n_names += 1;
                if (std.mem.startsWith(u8, a, "IA:") or std.mem.startsWith(u8, a, "RE:") or std.mem.startsWith(u8, a, "JE:")) {
                    n_errors += 1;
                    if (want_stats) {
                        const key = a[0..@min(a.len, 90)];
                        const gop = stats.getOrPut(gpa, gpa.dupe(u8, key) catch oom()) catch oom();
                        if (!gop.found_existing) gop.value_ptr.* = 0;
                        gop.value_ptr.* += 1;
                    }
                }
                if (std.mem.startsWith(u8, a, "W:") and a.len > 2) n_warnings += 1;
                for (required_phrases, 0..) |ph, pi| {
                    if (!phrase_seen[pi] and std.mem.indexOf(u8, a, ph) != null) phrase_seen[pi] = true;
                }
                if (!std.mem.eql(u8, a, b)) {
                    failures += 1;
                    if (failures <= 8) {
                        std.debug.print("FAIL case {d} ({s}) record {d}\n  input: ", .{ start + k, c.label, j });
                        printEscaped(c.input);
                        std.debug.print("\n  ref:   ", .{});
                        printEscaped(a);
                        std.debug.print("\n  port:  ", .{});
                        printEscaped(b);
                        std.debug.print("\n", .{});
                    }
                }
            }
            for (port.items) |x| gpa.free(x);
            port.deinit(gpa);
        }
        for (all.items) |x| gpa.free(x);
        all.deinit(gpa);
    }

    if (want_stats) {
        var it = stats.iterator();
        while (it.next()) |e| {
            std.debug.print("{d: >7}  ", .{e.value_ptr.*});
            printEscaped(e.key_ptr.*);
            std.debug.print("\n", .{});
        }
    }

    var missing: usize = 0;
    if (filter == null) {
        for (required_phrases, 0..) |ph, pi| {
            if (!phrase_seen[pi]) {
                std.debug.print("FAIL: no reference record contains \"{s}\": a generator has drifted\n", .{ph});
                missing += 1;
            }
        }
    }

    if (failures != 0 or missing != 0) {
        std.debug.print("FAIL: {d} of {d} checks differ over {d} cases\n", .{ failures, checks, cases.items.len });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} checks over {d} cases ({d} grammars, {d} errors, {d} warnings, {d} builder names, {d} reference crashes skipped), cli/json/schema*.zig identical to json-schema*.cpp\n", .{ checks, cases.items.len, n_grammars, n_errors, n_warnings, n_names, crashed });
}
