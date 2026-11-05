//! A PEG arena as JSON, for comparing two arenas in `make peg-diff` and
//! `make chat-parse-diff`.
//!
//! **Not a port.** Upstream's `common_peg_arena` serialized itself with
//! `save` (peg-parser.cpp, until v0.6.0) so a chat template's parser could
//! travel as a string; v0.7.0 removed that and passes the arena itself,
//! and the port followed (`cli/peg/` has no serializer). The gates still
//! compare whole arenas, so this keeps the old form, and
//! `harness_peg_save` in `harness/peg_rec.h` writes the same for upstream's
//! arena through its public members. One change from the old form: the rule
//! map, whose libc++ iteration order no caller can see any more, is written
//! sorted by name.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const errors = peg.errors;
const arena_mod = peg.arena;

const Allocator = std.mem.Allocator;
const Error = errors.Error;
const Arena = arena_mod.Arena;
const Parser = arena_mod.Parser;
const ParserId = arena_mod.ParserId;
const Value = json.Value;

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

/// One parser, as the old `serialize_parser_variant` wrote it.
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

/// The arena, as the old `to_json` wrote it, its rules sorted by name.
///
/// Return: owned by the caller.
pub fn toJson(gpa: Allocator, arena: *const Arena) Error!Value {
    var parsers = Value.emptyArray();
    errdefer parsers.deinit(gpa);
    for (arena.parsers.items) |*p| parsers.pushBack(gpa, try serializeParser(gpa, p)) catch |e| return je(e);

    const Entry = struct { key: []const u8, id: ParserId };
    var entries: std.ArrayList(Entry) = .empty;
    defer entries.deinit(gpa);
    var it = arena.rules.iterator();
    while (it.next()) |e| try entries.append(gpa, .{ .key = e.key, .id = e.value.* });
    std.mem.sort(Entry, entries.items, {}, struct {
        fn lt(_: void, x: Entry, y: Entry) bool {
            return std.mem.order(u8, x.key, y.key) == .lt;
        }
    }.lt);
    var rules = Value.emptyObject();
    errdefer rules.deinit(gpa);
    for (entries.items) |e| try put(gpa, &rules, e.key, .{ .number_unsigned = e.id });

    var o = Value.emptyObject();
    errdefer o.deinit(gpa);
    try put(gpa, &o, "parsers", parsers);
    try put(gpa, &o, "rules", rules);
    try put(gpa, &o, "root", .{ .number_unsigned = arena.root });
    return o;
}

/// The old `save`: `to_json().dump()`.
///
/// Return: owned by the caller; `JsonError` when a string in it is not
/// UTF-8.
pub fn save(gpa: Allocator, arena: *const Arena) Error![]u8 {
    var v = try toJson(gpa, arena);
    defer v.deinit(gpa);
    return json.dump(gpa, &v, -1) catch |e| je(e);
}
