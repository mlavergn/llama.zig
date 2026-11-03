//! A PEG arena as GBNF: every rule reachable from the root -- or, lazily,
//! from the trigger rules -- written through the JSON schema converter's
//! grammar builder, with `until` and `ac` parsers emitted as the DFA of an
//! Aho-Corasick automaton.
//!
//! # Provenance
//!
//! Ports `common_peg_arena::build_grammar`, `collect_reachable_rules` and the
//! GBNF helpers `gbnf_escape_char_class`, `gbnf_char_class`,
//! `gbnf_ac_grammar`, `gbnf_excluding_grammar` and `gbnf_including_grammar`
//! from `llama.cpp/common/peg-parser.cpp` (llama.cpp v0.6.0, `d81235049`).
//! Each declaration names what it ports and the line it began at.
//!
//! The rules go to the builder in the rule map's iteration order -- libc++'s
//! `std::unordered_map`, reproduced by `libcxx.StringMap` -- which decides
//! which of two colliding names keeps its own.

const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const arena_mod = @import("arena.zig");

const Allocator = std.mem.Allocator;
const Error = errors.Error;
const Arena = arena_mod.Arena;
const Parser = arena_mod.Parser;
const ParserId = arena_mod.ParserId;
const invalid_parser_id = arena_mod.invalid_parser_id;
const s2g = json.schema_to_grammar;
const trie = json.trie;

/// The JSON schema converter's builder, which the grammar is written to.
pub const GrammarBuilder = s2g.GrammarBuilder;

/// Ports `gbnf_escape_char_class` (peg-parser.cpp:1389 @d81235049).
fn escapeCharClass(out: *std.ArrayList(u8), a: Allocator, c: u32) Error!void {
    if (c == '-' or c == ']' or c == '[' or c == '\\') return out.appendSlice(a, &.{ '\\', @intCast(c) });
    if (c == '\n') return out.appendSlice(a, "\\n");
    if (c == '\t') return out.appendSlice(a, "\\t");
    if (c == '\r') return out.appendSlice(a, "\\r");
    if (c >= 0x20 and c <= 0x7e) return out.append(a, @intCast(c));
    if (c <= 0xff) return out.print(a, "\\x{X:0>2}", .{c});
    if (c <= 0xffff) return out.print(a, "\\u{X:0>4}", .{c});
    return out.print(a, "\\U{X:0>8}", .{c});
}

/// Ports `gbnf_char_class` (peg-parser.cpp:1439 @d81235049).
fn charClass(a: Allocator, chars: []const u32, negate: bool) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, if (negate) "[^" else "[");
    for (chars) |ch| try escapeCharClass(&out, a, ch);
    try out.append(a, ']');
    return out.items;
}

/// One destination state and the characters that reach it.
const Bucket = struct { dest: usize, chars: std.ArrayList(u32) };

const Mode = enum { excluding, including };

/// Ports `gbnf_ac_grammar` (peg-parser.cpp:1447 @d81235049), with the two
/// rule bodies of `gbnf_excluding_grammar` (peg-parser.cpp:1502 @d81235049)
/// and `gbnf_including_grammar` (peg-parser.cpp:1524 @d81235049) chosen by
/// `mode`: one rule per non-matching state of the automaton over
/// `strings`.
///
/// Return: the start state's rule name, allocated with `a`.
fn acGrammar(a: Allocator, builder: *GrammarBuilder, prefix: []const u8, strings: []const []const u8, mode: Mode) Error![]const u8 {
    var ac = try trie.AhoCorasick.init(a, strings);
    defer ac.deinit();

    const S = struct {
        fn name(al: Allocator, pfx: []const u8, s: usize) Error![]const u8 {
            if (s == 0) return pfx;
            if (s < 10) return std.fmt.allocPrint(al, "{s}-0{d}", .{ pfx, s });
            return std.fmt.allocPrint(al, "{s}-{d}", .{ pfx, s });
        }
    };

    for (0..ac.numStates()) |q| {
        if (ac.isTerminal(q)) continue;
        // a `std::map` keyed by destination: kept sorted by it
        var buckets: std.ArrayList(Bucket) = .empty;
        var completing: std.ArrayList(u32) = .empty;
        var specific: std.ArrayList(u32) = .empty;
        for (ac.alphabet) |c| {
            const d = ac.next(q, c);
            if (ac.isTerminal(d)) {
                try completing.append(a, c);
                try specific.append(a, c);
            } else if (d != 0) {
                var at: usize = 0;
                while (at < buckets.items.len and buckets.items[at].dest < d) at += 1;
                if (at == buckets.items.len or buckets.items[at].dest != d) {
                    try buckets.insert(a, at, .{ .dest = d, .chars = .empty });
                }
                try buckets.items[at].chars.append(a, c);
                try specific.append(a, c);
            }
        }

        var rhs: std.ArrayList(u8) = .empty;
        switch (mode) {
            .excluding => {
                try rhs.appendSlice(a, "|");
                for (buckets.items) |b| {
                    try rhs.print(a, " {s} {s} |", .{ try charClass(a, b.chars.items, false), try S.name(a, prefix, b.dest) });
                }
                try rhs.print(a, " {s} {s}", .{ try charClass(a, specific.items, true), try S.name(a, prefix, 0) });
            },
            .including => {
                var alts: std.ArrayList([]const u8) = .empty;
                if (completing.items.len > 0) try alts.append(a, try charClass(a, completing.items, false));
                for (buckets.items) |b| {
                    try alts.append(a, try std.fmt.allocPrint(a, "{s} {s}", .{ try charClass(a, b.chars.items, false), try S.name(a, prefix, b.dest) }));
                }
                try alts.append(a, try std.fmt.allocPrint(a, "{s} {s}", .{ try charClass(a, specific.items, true), try S.name(a, prefix, 0) }));
                try arena_mod.joinTo(a, &rhs, alts.items, " | ");
            },
        }
        _ = builder.addRule(try S.name(a, prefix, q), rhs.items) catch |e| return errors.fromSchema(e);
    }

    if (ac.isTerminal(0)) _ = builder.addRule(prefix, "|") catch |e| return errors.fromSchema(e);
    return S.name(a, prefix, 0);
}

/// Ports `collect_reachable_rules` (peg-parser.cpp:1545 @d81235049): the
/// names of every rule reachable from `rule`, references followed.
fn collectReachableRules(a: Allocator, arena: *const Arena, rule: ParserId, reachable: *std.StringHashMapUnmanaged(void)) Error!void {
    var visited: std.StringHashMapUnmanaged(void) = .empty;
    try collectVisit(a, arena, rule, reachable, &visited);
}

fn collectVisit(a: Allocator, arena: *const Arena, id: ParserId, reachable: *std.StringHashMapUnmanaged(void), visited: *std.StringHashMapUnmanaged(void)) Error!void {
    const parser = try arena.get(id);
    switch (parser.*) {
        .epsilon, .start, .end, .until, .literal, .chars, .space, .any, .string => {},
        .sequence, .choice => |children| for (children) |c| try collectVisit(a, arena, c, reachable, visited),
        .rule => |p| {
            if (!visited.contains(p.name)) {
                try visited.put(a, p.name, {});
                try reachable.put(a, p.name, {});
                try collectVisit(a, arena, p.child, reachable, visited);
            }
        },
        .ref => |name| try collectVisit(a, arena, try arena.getRule(name), reachable, visited),
        else => try collectVisit(a, arena, parser.childPtr().?.*, reachable, visited),
    }
}

/// Builds `common_peg_arena::build_grammar`'s `to_gbnf` and its two
/// lambdas over one arena and builder.
const Generator = struct {
    a: Allocator,
    arena: *const Arena,
    builder: *GrammarBuilder,

    /// `schema_delegates`: a raw string value is parsed by the child rather
    /// than constrained by the schema.
    fn schemaDelegates(s: Parser.Schema) bool {
        const node = s.node orelse return true;
        return s.raw and node.mayBeString();
    }

    /// `effective_parser`: through tags, atomics and delegating schemas.
    fn effective(self: *const Generator, id_in: ParserId) Error!*const Parser {
        var id = id_in;
        while (true) {
            const p = try self.arena.get(id);
            switch (p.*) {
                .tag => |t| id = t.child,
                .atomic => |c| id = c,
                .schema => |s| if (schemaDelegates(s)) {
                    id = s.child;
                } else return p,
                else => return p,
            }
        }
    }

    fn isGroup(p: *const Parser) bool {
        return p.* == .choice or p.* == .sequence;
    }

    fn repeatSuffix(self: *const Generator, base: []const u8, min: c_int, max: c_int) Error![]const u8 {
        const a = self.a;
        if (min == 0 and max == 1) return std.fmt.allocPrint(a, "{s}?", .{base});
        if (min == 0 and max == -1) return std.fmt.allocPrint(a, "{s}*", .{base});
        if (min == 1 and max == -1) return std.fmt.allocPrint(a, "{s}+", .{base});
        if (max == -1) return std.fmt.allocPrint(a, "{s}{{{d},}}", .{ base, min });
        if (min == max) {
            if (min == 1) return base;
            return std.fmt.allocPrint(a, "{s}{{{d}}}", .{ base, min });
        }
        return std.fmt.allocPrint(a, "{s}{{{d},{d}}}", .{ base, min, max });
    }

    /// `to_gbnf`: one parser as a GBNF expression.
    fn toGbnf(self: *const Generator, id: ParserId) Error![]const u8 {
        const a = self.a;
        const parser = try self.arena.get(id);
        switch (parser.*) {
            .epsilon, .start, .end => return "",
            .literal => |l| return s2g.gbnfFormatLiteral(a, l) catch |e| return errors.fromSchema(e),
            .sequence => |children| {
                var s: std.ArrayList(u8) = .empty;
                for (children) |child| {
                    const child_gbnf = try self.toGbnf(child);
                    if (child_gbnf.len == 0) continue;
                    if (s.items.len > 0) try s.append(a, ' ');
                    if (isGroup(try self.effective(child))) {
                        try s.print(a, "({s})", .{child_gbnf});
                    } else {
                        try s.appendSlice(a, child_gbnf);
                    }
                }
                return s.items;
            },
            .choice => |children| {
                var s: std.ArrayList(u8) = .empty;
                for (children) |child| {
                    if (s.items.len > 0) try s.appendSlice(a, " | ");
                    const child_gbnf = try self.toGbnf(child);
                    if ((try self.effective(child)).* == .choice) {
                        try s.print(a, "({s})", .{child_gbnf});
                    } else {
                        try s.appendSlice(a, child_gbnf);
                    }
                }
                return s.items;
            },
            .repetition => |p| {
                var child_gbnf = try self.toGbnf(p.child);
                if (isGroup(try self.effective(p.child))) child_gbnf = try std.fmt.allocPrint(a, "({s})", .{child_gbnf});
                return self.repeatSuffix(child_gbnf, p.min_count, p.max_count);
            },
            .and_, .not_ => return "",
            .any => return ".",
            .space => return "space",
            .chars => |p| return self.repeatSuffix(p.pattern, p.min_count, p.max_count),
            .string => |d| return std.fmt.allocPrint(a, "( [^{c}\\\\] | \"\\\\\" ( [{c}\\\\/ bfnrt] | \"u\" [0-9a-fA-F]{{4}} ) )*", .{ d, d }),
            .until => |delims| {
                if (delims.len == 0) return ".*";
                return acGrammar(a, self.builder, try std.fmt.allocPrint(a, "until-{d}", .{id}), delims, .excluding);
            },
            .schema => |p| {
                if (schemaDelegates(p)) return self.toGbnf(p.child);
                return self.builder.addSchema(p.name, p.node.?) catch |e| return errors.fromSchema(e);
            },
            .rule => |p| return p.name,
            .ref => |name| return name,
            .tag => |p| return self.toGbnf(p.child),
            .atomic => |c| return self.toGbnf(c),
            .gbnf => |p| return p.grammar,
            .ac => |p| return acGrammar(a, self.builder, try std.fmt.allocPrint(a, "ac-{d}", .{id}), p.delimiters, .including),
        }
    }
};

fn lessBytes(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

/// Ports `build_grammar` (peg-parser.cpp:1606 @d81235049), a member of
/// `common_peg_arena`: every reachable rule, then `root` -- the root parser,
/// or, lazily, the trigger rules as alternatives.
///
/// Parameters:
/// - `arena`: the parsers.
/// - `builder`: the grammar being assembled.
/// - `lazy`: start from the trigger rules rather than the root.
pub fn buildGrammar(arena: *const Arena, builder: *GrammarBuilder, lazy: bool) Error!void {
    var scratch = std.heap.ArenaAllocator.init(arena.gpa);
    defer scratch.deinit();
    const a = scratch.allocator();
    const gen: Generator = .{ .a = a, .arena = arena, .builder = builder };

    var reachable: std.StringHashMapUnmanaged(void) = .empty;
    if (lazy) {
        var it = arena.rules.iterator();
        while (it.next()) |e| {
            const parser = try arena.get(e.value.*);
            if (parser.* == .rule and parser.rule.trigger) {
                try reachable.put(a, e.key, {});
                try collectReachableRules(a, arena, e.value.*, &reachable);
            }
        }
    } else {
        try collectReachableRules(a, arena, arena.root, &reachable);
    }

    var it = arena.rules.iterator();
    while (it.next()) |e| {
        if (!reachable.contains(e.key)) continue;
        const parser = try arena.get(e.value.*);
        if (parser.* == .rule) {
            const rhs = try gen.toGbnf(parser.rule.child);
            _ = builder.addRule(parser.rule.name, rhs) catch |err| return errors.fromSchema(err);
        }
    }

    if (lazy) {
        var names: std.ArrayList([]const u8) = .empty;
        var it2 = arena.rules.iterator();
        while (it2.next()) |e| {
            const parser = try arena.get(e.value.*);
            if (parser.* == .rule and parser.rule.trigger) try names.append(a, parser.rule.name);
        }
        std.mem.sort([]const u8, names.items, {}, lessBytes);
        var rhs: std.ArrayList(u8) = .empty;
        try arena_mod.joinTo(a, &rhs, names.items, " | ");
        _ = builder.addRule("root", rhs.items) catch |err| return errors.fromSchema(err);
    } else if (arena.root != invalid_parser_id) {
        const rhs = try gen.toGbnf(arena.root);
        _ = builder.addRule("root", rhs) catch |err| return errors.fromSchema(err);
    }
}

/// `build_grammar([&](const common_grammar_builder & b) { arena.build_grammar(b, lazy); })`:
/// the whole grammar, through `json-schema-to-grammar.cpp`'s `build_grammar`.
///
/// Parameters:
/// - `gpa`: allocates the result.
/// - `arena`: the parsers.
/// - `lazy`: start from the trigger rules.
/// - `options`: the converter's, `dotall` for patterns.
///
/// Return: the grammar, owned by the caller.
pub fn grammar(gpa: Allocator, arena: *const Arena, lazy: bool, options: s2g.Options) Error![]u8 {
    const Ctx = struct {
        arena: *const Arena,
        lazy: bool,
        err: ?Error = null,

        fn cb(self: *@This(), b: *GrammarBuilder) json.schema.Error!void {
            buildGrammar(self.arena, b, self.lazy) catch |e| {
                self.err = e;
                return error.RuntimeError;
            };
        }
    };
    var ctx: Ctx = .{ .arena = arena, .lazy = lazy };
    return s2g.buildGrammar(gpa, options, &ctx, Ctx.cb) catch |e| {
        // an error of ours already carries its text; one of the
        // converter's carries `json.schema.what()`
        if (ctx.err) |ours| return ours;
        return errors.fromSchema(e);
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an until parser is the complement of its delimiters' automaton" {
    const builder = @import("builder.zig");
    const a = std.testing.allocator;
    var arena = try builder.buildPegParser(a, {}, struct {
        fn f(_: void, b: *builder.Builder) Error!ParserId {
            return b.plus(try b.until("ab"), try b.literal("ab"));
        }
    }.f);
    defer arena.deinit();
    const g = try grammar(a, &arena, false, .{});
    defer a.free(g);
    try std.testing.expectEqualStrings(
        \\root ::= until-0 "ab"
        \\space ::= | " " | "\n"{1,2} [ \t]{0,20}
        \\until-0 ::= | [a] until-0-01 | [^a] until-0
        \\until-0-01 ::= | [a] until-0-01 | [^ab] until-0
        \\
    , g);
}
