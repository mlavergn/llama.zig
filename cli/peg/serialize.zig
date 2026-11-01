//! A PEG arena to JSON and back: how a chat template's parser travels from
//! `common_chat_templates_apply` to `common_chat_parse` as a string.
//!
//! # Provenance
//!
//! Ports `serialize_parser_variant`, `deserialize_parser_variant`,
//! `common_peg_arena::to_json`, `from_json`, `save` and `load` from
//! `llama.cpp/common/peg-parser.cpp` (llama.cpp v0.4.1, `b29c606e2`). Each
//! declaration names what it ports and the line it began at.
//!
//! The rules are written in the rule map's iteration order, libc++'s, and
//! read back by inserting them in the object's order, as nlohmann's
//! `from_json` for a map does; a schema parser comes back without its
//! schema, as in the C++.

const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const arena_mod = @import("arena.zig");

const Allocator = std.mem.Allocator;
const Error = errors.Error;
const Arena = arena_mod.Arena;
const Parser = arena_mod.Parser;
const ParserId = arena_mod.ParserId;
const CharRange = arena_mod.CharRange;
const Value = json.Value;
const invalid_parser_id = arena_mod.invalid_parser_id;

fn je(e: json.Error) Error {
    return errors.fromJson(e);
}

fn str(gpa: Allocator, s: []const u8) Error!Value {
    return Value.initString(gpa, s) catch |e| je(e);
}

fn put(gpa: Allocator, o: *Value, key: []const u8, v: Value) Error!void {
    var val = v;
    o.set(gpa, key, val) catch |e| {
        val.deinit(gpa);
        return je(e);
    };
}

fn idArray(gpa: Allocator, ids: []const ParserId) Error!Value {
    var arr = Value.emptyArray();
    errdefer arr.deinit(gpa);
    for (ids) |id| arr.pushBack(gpa, .{ .number_unsigned = id }) catch |e| return je(e);
    return arr;
}

fn strArray(gpa: Allocator, list: []const []const u8) Error!Value {
    var arr = Value.emptyArray();
    errdefer arr.deinit(gpa);
    for (list) |s| arr.pushBack(gpa, try str(gpa, s)) catch |e| return je(e);
    return arr;
}

fn typed(gpa: Allocator, t: []const u8) Error!Value {
    var o = Value.emptyObject();
    errdefer o.deinit(gpa);
    try put(gpa, &o, "type", try str(gpa, t));
    return o;
}

/// Ports `serialize_parser_variant` (peg-parser.cpp:1791 @b29c606e2).
///
/// Return: owned by the caller.
fn serializeParser(gpa: Allocator, parser: *const Parser) Error!Value {
    const name: []const u8 = switch (parser.*) {
        .epsilon => "epsilon",
        .start => "start",
        .end => "end",
        .literal => "literal",
        .sequence => "sequence",
        .choice => "choice",
        .repetition => "repetition",
        .and_ => "and",
        .not_ => "not",
        .any => "any",
        .space => "space",
        .chars => "chars",
        .string => "string",
        .until => "until",
        .schema => "schema",
        .rule => "rule",
        .ref => "ref",
        .atomic => "atomic",
        .tag => "tag",
        .gbnf => "gbnf",
        .ac => "ac",
    };
    var o = try typed(gpa, name);
    errdefer o.deinit(gpa);
    switch (parser.*) {
        .epsilon, .start, .end, .any, .space => {},
        .literal => |l| try put(gpa, &o, "literal", try str(gpa, l)),
        .sequence, .choice => |c| try put(gpa, &o, "children", try idArray(gpa, c)),
        .repetition => |p| {
            try put(gpa, &o, "child", .{ .number_unsigned = p.child });
            try put(gpa, &o, "min_count", .{ .number_integer = p.min_count });
            try put(gpa, &o, "max_count", .{ .number_integer = p.max_count });
        },
        .and_, .not_, .atomic => |c| try put(gpa, &o, "child", .{ .number_unsigned = c }),
        .chars => |p| {
            var ranges = Value.emptyArray();
            errdefer ranges.deinit(gpa);
            for (p.ranges) |r| {
                var ro = Value.emptyObject();
                errdefer ro.deinit(gpa);
                try put(gpa, &ro, "start", .{ .number_unsigned = r.start });
                try put(gpa, &ro, "end", .{ .number_unsigned = r.end });
                ranges.pushBack(gpa, ro) catch |e| return je(e);
            }
            try put(gpa, &o, "pattern", try str(gpa, p.pattern));
            try put(gpa, &o, "ranges", ranges);
            try put(gpa, &o, "negated", .{ .boolean = p.negated });
            try put(gpa, &o, "min_count", .{ .number_integer = p.min_count });
            try put(gpa, &o, "max_count", .{ .number_integer = p.max_count });
        },
        .string => |d| try put(gpa, &o, "delimiter", try str(gpa, &.{d})),
        .until => |d| try put(gpa, &o, "delimiters", try strArray(gpa, d)),
        .schema => |p| {
            try put(gpa, &o, "child", .{ .number_unsigned = p.child });
            try put(gpa, &o, "name", try str(gpa, p.name));
            try put(gpa, &o, "raw", .{ .boolean = p.raw });
        },
        .rule => |p| {
            try put(gpa, &o, "name", try str(gpa, p.name));
            try put(gpa, &o, "child", .{ .number_unsigned = p.child });
            try put(gpa, &o, "trigger", .{ .boolean = p.trigger });
        },
        .ref => |n| try put(gpa, &o, "name", try str(gpa, n)),
        .tag => |p| {
            try put(gpa, &o, "child", .{ .number_unsigned = p.child });
            try put(gpa, &o, "tag", try str(gpa, p.tag));
        },
        .gbnf => |p| {
            try put(gpa, &o, "child", .{ .number_unsigned = p.child });
            try put(gpa, &o, "grammar", try str(gpa, p.grammar));
        },
        .ac => |p| {
            try put(gpa, &o, "child", .{ .number_unsigned = p.child });
            try put(gpa, &o, "delimiters", try strArray(gpa, p.delimiters));
        },
    }
    return o;
}

/// Ports `to_json` (peg-parser.cpp:1873 @b29c606e2), a member of
/// `common_peg_arena`.
///
/// Return: owned by the caller.
pub fn toJson(gpa: Allocator, arena: *const Arena) Error!Value {
    var parsers = Value.emptyArray();
    errdefer parsers.deinit(gpa);
    for (arena.parsers.items) |*p| parsers.pushBack(gpa, try serializeParser(gpa, p)) catch |e| return je(e);

    var rules = Value.emptyObject();
    errdefer rules.deinit(gpa);
    var it = arena.rules.iterator();
    while (it.next()) |e| try put(gpa, &rules, e.key, .{ .number_unsigned = e.value.* });

    var o = Value.emptyObject();
    errdefer o.deinit(gpa);
    try put(gpa, &o, "parsers", parsers);
    try put(gpa, &o, "rules", rules);
    try put(gpa, &o, "root", .{ .number_unsigned = arena.root });
    return o;
}

/// Ports `save` (peg-parser.cpp:2086 @b29c606e2): `to_json().dump()`.
///
/// Return: owned by the caller; `JsonError` when a string in it is not
/// UTF-8.
pub fn save(gpa: Allocator, arena: *const Arena) Error![]u8 {
    var v = try toJson(gpa, arena);
    defer v.deinit(gpa);
    return json.dump(gpa, &v, -1) catch |e| je(e);
}

/// `j[key]` on a `const common_json`: `at(key)`.
fn idx(j: *const Value, key: []const u8) Error!*Value {
    return j.at(key) catch |e| je(e);
}

/// `j[key].get<T>()`.
fn getAt(j: *const Value, key: []const u8, comptime T: type) Error!T {
    return (try idx(j, key)).get(T) catch |e| je(e);
}

/// Element `i` of a range-for over a `common_json`: an object's member
/// values, an array's elements, a plain value itself once.
fn elemAt(j: *Value, i: usize) *Value {
    return switch (j.*) {
        .object => |*m| &m.members.items[i].value,
        .array => |*a| &a.items[i],
        else => j,
    };
}

fn fail(text: []const u8) Error {
    return errors.raise(error.RuntimeError, &.{text});
}

/// Ports `deserialize_parser_variant` (peg-parser.cpp:1885 @b29c606e2).
///
/// Return: the parser; its strings and lists are allocated in `arena`.
fn deserializeParser(arena: *Arena, j: *const Value) Error!Parser {
    const a = arena.alloc();
    if (!j.contains("type") or !(try idx(j, "type")).isString()) return fail("Parser variant JSON missing or invalid 'type' field");
    const t = try getAt(j, "type", []const u8);

    if (std.mem.eql(u8, t, "epsilon")) return .epsilon;
    if (std.mem.eql(u8, t, "start")) return .start;
    if (std.mem.eql(u8, t, "end")) return .end;
    if (std.mem.eql(u8, t, "literal")) {
        if (!j.contains("literal") or !(try idx(j, "literal")).isString()) return fail("literal parser missing or invalid 'literal' field");
        return .{ .literal = try a.dupe(u8, try getAt(j, "literal", []const u8)) };
    }
    if (std.mem.eql(u8, t, "sequence") or std.mem.eql(u8, t, "choice")) {
        const seq = t[0] == 's';
        if (!j.contains("children") or !(try idx(j, "children")).isArray()) {
            return fail(if (seq) "sequence parser missing or invalid 'children' field" else "choice parser missing or invalid 'children' field");
        }
        const ids = (try idx(j, "children")).getSlice(a, usize) catch |e| return je(e);
        return if (seq) .{ .sequence = ids } else .{ .choice = ids };
    }
    if (std.mem.eql(u8, t, "repetition")) {
        if (!j.contains("child") or !j.contains("min_count") or !j.contains("max_count")) return fail("repetition parser missing required fields");
        const child = try getAt(j, "child", usize);
        const min = try getAt(j, "min_count", c_int);
        const max = try getAt(j, "max_count", c_int);
        return .{ .repetition = .{ .child = child, .min_count = min, .max_count = max } };
    }
    if (std.mem.eql(u8, t, "and")) {
        if (!j.contains("child")) return fail("and parser missing 'child' field");
        return .{ .and_ = try getAt(j, "child", usize) };
    }
    if (std.mem.eql(u8, t, "not")) {
        if (!j.contains("child")) return fail("not parser missing 'child' field");
        return .{ .not_ = try getAt(j, "child", usize) };
    }
    if (std.mem.eql(u8, t, "any")) return .any;
    if (std.mem.eql(u8, t, "space")) return .space;
    if (std.mem.eql(u8, t, "chars")) {
        if (!j.contains("pattern") or !j.contains("ranges") or !j.contains("negated") or
            !j.contains("min_count") or !j.contains("max_count")) return fail("chars parser missing required fields");
        const pattern = try a.dupe(u8, try getAt(j, "pattern", []const u8));
        const negated = try getAt(j, "negated", bool);
        const min = try getAt(j, "min_count", c_int);
        const max = try getAt(j, "max_count", c_int);
        var ranges: std.ArrayList(CharRange) = .empty;
        const rj = try idx(j, "ranges");
        for (0..rj.size()) |i| {
            const r = elemAt(rj, i);
            if (!r.contains("start") or !r.contains("end")) return fail("char_range missing 'start' or 'end' field");
            const s = try getAt(r, "start", u32);
            const e = try getAt(r, "end", u32);
            try ranges.append(a, .{ .start = s, .end = e });
        }
        return .{ .chars = .{ .pattern = pattern, .ranges = ranges.items, .negated = negated, .min_count = min, .max_count = max } };
    }
    if (std.mem.eql(u8, t, "string")) {
        if (!j.contains("delimiter")) return fail("string parser missing delimiter field.");
        const d = try getAt(j, "delimiter", []const u8);
        if (d.len == 0) return fail("string parser delimiter is empty.");
        return .{ .string = d[0] };
    }
    if (std.mem.eql(u8, t, "until")) {
        if (!j.contains("delimiters") or !(try idx(j, "delimiters")).isArray()) return fail("until parser missing or invalid 'delimiters' field");
        return .{ .until = try dupeStrings(a, try idx(j, "delimiters")) };
    }
    if (std.mem.eql(u8, t, "schema")) {
        if (!j.contains("child") or !j.contains("name") or !j.contains("raw")) return fail("schema parser missing required fields");
        const child = try getAt(j, "child", usize);
        const name = try a.dupe(u8, try getAt(j, "name", []const u8));
        const raw = try getAt(j, "raw", bool);
        return .{ .schema = .{ .child = child, .name = name, .raw = raw } };
    }
    if (std.mem.eql(u8, t, "rule")) {
        if (!j.contains("name") or !j.contains("child") or !j.contains("trigger")) return fail("rule parser missing required fields");
        const name = try a.dupe(u8, try getAt(j, "name", []const u8));
        const child = try getAt(j, "child", usize);
        const trigger = try getAt(j, "trigger", bool);
        return .{ .rule = .{ .name = name, .child = child, .trigger = trigger } };
    }
    if (std.mem.eql(u8, t, "ref")) {
        if (!j.contains("name") or !(try idx(j, "name")).isString()) return fail("ref parser missing or invalid 'name' field");
        return .{ .ref = try a.dupe(u8, try getAt(j, "name", []const u8)) };
    }
    if (std.mem.eql(u8, t, "atomic")) {
        if (!j.contains("child")) return fail("tag parser missing required fields");
        return .{ .atomic = try getAt(j, "child", usize) };
    }
    if (std.mem.eql(u8, t, "tag")) {
        if (!j.contains("child") or !j.contains("tag")) return fail("tag parser missing required fields");
        const child = try getAt(j, "child", usize);
        return .{ .tag = .{ .child = child, .tag = try a.dupe(u8, try getAt(j, "tag", []const u8)) } };
    }
    if (std.mem.eql(u8, t, "gbnf")) {
        if (!j.contains("child") or !j.contains("grammar")) return fail("gbnf parser missing required fields");
        const child = try getAt(j, "child", usize);
        return .{ .gbnf = .{ .child = child, .grammar = try a.dupe(u8, try getAt(j, "grammar", []const u8)) } };
    }
    if (std.mem.eql(u8, t, "ac")) {
        if (!j.contains("child") or !j.contains("delimiters") or !(try idx(j, "delimiters")).isArray() or (try idx(j, "delimiters")).empty()) {
            return fail("ac parser requires 'child' and a non-empty 'delimiters' array");
        }
        const child = try getAt(j, "child", usize);
        return .{ .ac = .{ .child = child, .delimiters = try dupeStrings(a, try idx(j, "delimiters")) } };
    }
    return errors.raise(error.RuntimeError, &.{ "Unknown parser type: ", t });
}

/// `get<std::vector<std::string>>()`, copied into `a`.
fn dupeStrings(a: Allocator, v: *const Value) Error![]const []const u8 {
    const list = v.getSlice(a, []const u8) catch |e| return je(e);
    for (list) |*s| s.* = try a.dupe(u8, s.*);
    return list;
}

/// Ports `from_json` (peg-parser.cpp:2051 @b29c606e2), a member of
/// `common_peg_arena`. Nothing checks that a child id is in range; a parse
/// that reaches one fails with `OutOfRange`, as `std::vector::at` throws.
///
/// Return: the arena, owned by the caller.
pub fn fromJson(gpa: Allocator, j: *const Value) Error!Arena {
    if (!j.contains("parsers") or !(try idx(j, "parsers")).isArray()) return fail("JSON missing or invalid 'parsers' array");
    if (!j.contains("rules") or !(try idx(j, "rules")).isObject()) return fail("JSON missing or invalid 'rules' object");
    if (!j.contains("root")) return fail("JSON missing 'root' field");

    var arena = try Arena.init(gpa);
    errdefer arena.deinit();

    const parsers_json = try idx(j, "parsers");
    try arena.parsers.ensureTotalCapacityPrecise(gpa, parsers_json.size());
    for (0..parsers_json.size()) |i| {
        const p = try deserializeParser(&arena, elemAt(parsers_json, i));
        arena.parsers.appendAssumeCapacity(p);
    }

    const rules = (try idx(j, "rules")).getMap(gpa, usize) catch |e| return je(e);
    defer gpa.free(rules);
    for (rules) |r| _ = try arena.rules.getOrPut(r.key, r.value);

    var it = arena.rules.iterator();
    while (it.next()) |e| {
        if (e.value.* >= arena.parsers.items.len) {
            var buf: [24]u8 = undefined;
            const n = std.fmt.bufPrint(&buf, "{d}", .{e.value.*}) catch unreachable;
            return errors.raise(error.RuntimeError, &.{ "Rule '", e.key, "' references invalid parser ID: ", n });
        }
    }

    arena.root = try getAt(j, "root", usize);
    if (arena.root != invalid_parser_id and arena.root >= arena.parsers.items.len) {
        var buf: [24]u8 = undefined;
        const n = std.fmt.bufPrint(&buf, "{d}", .{arena.root}) catch unreachable;
        return errors.raise(error.RuntimeError, &.{ "Root references invalid parser ID: ", n });
    }
    return arena;
}

/// Ports `load` (peg-parser.cpp:2090 @b29c606e2): `*this =
/// from_json(common_json::parse(data))`; on failure the arena is unchanged.
pub fn load(arena: *Arena, data: []const u8) Error!void {
    const gpa = arena.gpa;
    var v = json.parse(gpa, data) catch |e| return je(e);
    defer v.deinit(gpa);
    const next = try fromJson(gpa, &v);
    arena.deinit();
    arena.* = next;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an arena saved and loaded keeps every parser and rule" {
    const builder = @import("builder.zig");
    const a = std.testing.allocator;
    var arena = try builder.buildPegParser(a, {}, struct {
        fn f(_: void, b: *builder.Builder) Error!arena_mod.ParserId {
            return b.plus(try b.chars("[a-z\\u00e9]", 1, -1), try b.tag("t", try b.json()));
        }
    }.f);
    defer arena.deinit();
    const s1 = try save(a, &arena);
    defer a.free(s1);
    var other = try Arena.init(a);
    defer other.deinit();
    try load(&other, s1);
    const s2 = try save(a, &other);
    defer a.free(s2);
    // the same text but for the rules' order: reinserted, they iterate
    // differently, as the C++'s do
    try std.testing.expectEqual(s1.len, s2.len);
    try std.testing.expectEqualStrings(s1[0..std.mem.indexOf(u8, s1, "\"rules\"").?], s2[0..std.mem.indexOf(u8, s2, "\"rules\"").?]);
    try std.testing.expectError(error.RuntimeError, load(&other, "{\"parsers\":[{\"type\":\"x\"}],\"rules\":{},\"root\":0}"));
    try std.testing.expectEqualStrings("Unknown parser type: x", errors.what(error.RuntimeError));
}
