//! The PEG engine: the parser variants, the arena that holds them, the parse
//! context and its AST, and the executor that runs a parser over input --
//! fully or partially, a stream that may not have ended yet.
//!
//! # Provenance
//!
//! Ports the parser types, `common_peg_ast_arena`, `common_peg_parse_context`,
//! `common_peg_arena` and `parser_executor` from
//! `llama.cpp/common/peg-parser.h` and `llama.cpp/common/peg-parser.cpp`
//! (llama.cpp v0.7.0, `620d2225c`). Each declaration names what it ports and
//! the line it began at. The builder is `builder.zig`, the GBNF export
//! `gbnf.zig`, the serialization `serialize.zig`.
//!
//! # Ownership
//!
//! An `Arena` owns its parsers and everything they hold, in one arena
//! allocator; the rule map is libc++'s `std::unordered_map` layout
//! (`libcxx.StringMap`), because `to_json` writes the rules in its
//! iteration order and `build_grammar` adds them in it. A `ParseContext`
//! owns a copy of its input, the AST, and the scratch lists results carry;
//! an AST node's `text` is a slice of the context's input, as the C++'s is a
//! `std::string_view` of it.
//!
//! `std::visit` over the variant is a `switch` over `Parser`, whose tags are
//! in the variant's order.

const std = @import("std");
const json = @import("json");
const libcxx = @import("llama").libcxx;
const errors = @import("errors.zig");
const trie = json.trie;
const schema = json.schema;

const Allocator = std.mem.Allocator;
pub const Error = errors.Error;
const raise = errors.raise;

/// Ports `common_peg_parser_id` (peg-parser.h:20 @620d2225c).
pub const ParserId = usize;
/// Ports `COMMON_PEG_INVALID_PARSER_ID` (peg-parser.h:21 @620d2225c).
pub const invalid_parser_id: ParserId = std.math.maxInt(usize);
/// Ports `common_peg_ast_id` (peg-parser.h:23 @620d2225c).
pub const AstId = usize;
/// Ports `COMMON_PEG_INVALID_AST_ID` (peg-parser.h:24 @620d2225c).
pub const invalid_ast_id: AstId = std.math.maxInt(usize);

/// Ports `common_peg_parse_result_type` (peg-parser.h:68 @620d2225c).
pub const ResultType = enum(c_int) {
    fail = 0,
    success = 1,
    need_more_input = 2,

    /// Ports `common_peg_parse_result_type_name` (peg-parser.cpp:21 @620d2225c).
    pub fn name(self: ResultType) []const u8 {
        return switch (self) {
            .fail => "fail",
            .success => "success",
            .need_more_input => "need_more_input",
        };
    }
};

/// Ports `common_peg_parse_flags` (peg-parser.h:163 @620d2225c), as a bit
/// set.
pub const Flags = struct {
    pub const none: u32 = 0;
    pub const lenient: u32 = 1 << 0;
    pub const debug: u32 = 1 << 1;
};

/// Ports `common_peg_invalid_utf8` (peg-parser.h:77 @620d2225c): a run of
/// input bytes that does not decode as UTF-8, by its position in the input.
pub const InvalidUtf8 = struct {
    pos: usize,
    len: usize,
};

/// Ports `common_peg_ast_node` (peg-parser.h:82 @620d2225c).
pub const AstNode = struct {
    id: AstId,
    rule: []const u8,
    tag: []const u8,
    start: usize,
    end: usize,
    /// a slice of the parse context's input
    text: []const u8,
    children: []const AstId,
    is_partial: bool = false,
    /// the invalid UTF-8 inside the node, in ascending order
    invalid_utf8: []const InvalidUtf8 = &.{},

    /// Ports `sanitized_text` (peg-parser.cpp:169 @620d2225c), a member of
    /// `common_peg_ast_node`: the text with every invalid run replaced by
    /// U+FFFD.
    ///
    /// Parameters:
    /// - `gpa`: allocates the result.
    ///
    /// Return: owned by the caller.
    pub fn sanitizedText(self: *const AstNode, gpa: Allocator) Allocator.Error![]u8 {
        if (self.invalid_utf8.len == 0) return gpa.dupe(u8, self.text);
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.ensureTotalCapacity(gpa, self.text.len + 2 * self.invalid_utf8.len);
        var seg_start = self.start;
        for (self.invalid_utf8) |invalid| {
            try out.appendSlice(gpa, self.text[seg_start - self.start ..][0 .. invalid.pos - seg_start]);
            try out.appendSlice(gpa, "\xEF\xBF\xBD");
            seg_start = invalid.pos + invalid.len;
        }
        try out.appendSlice(gpa, self.text[seg_start - self.start ..][0 .. self.end - seg_start]);
        return out.toOwnedSlice(gpa);
    }
};

/// Ports `common_peg_ast_arena` (peg-parser.h:104 @620d2225c): the nodes a
/// parse produced, by id.
pub const AstArena = struct {
    gpa: Allocator,
    nodes: std.ArrayList(AstNode) = .empty,
    /// the nodes' names and child lists
    mem: std.heap.ArenaAllocator,

    /// Return: an empty arena; `deinit` frees it.
    pub fn init(gpa: Allocator) AstArena {
        return .{ .gpa = gpa, .mem = .init(gpa) };
    }

    /// Frees every node.
    pub fn deinit(self: *AstArena) void {
        self.nodes.deinit(self.gpa);
        self.mem.deinit();
    }

    /// Ports `add_node` (peg-parser.h:107 @620d2225c), a member of
    /// `common_peg_ast_arena`. The names, the child list and the invalid
    /// runs are copied.
    ///
    /// Return: the new node's id.
    pub fn addNode(
        self: *AstArena,
        rule: []const u8,
        tag: []const u8,
        start: usize,
        end: usize,
        text: []const u8,
        children: []const AstId,
        is_partial: bool,
        invalid_utf8: []const InvalidUtf8,
    ) Error!AstId {
        const a = self.mem.allocator();
        const id = self.nodes.items.len;
        try self.nodes.append(self.gpa, .{
            .id = id,
            .rule = try a.dupe(u8, rule),
            .tag = try a.dupe(u8, tag),
            .start = start,
            .end = end,
            .text = text,
            .children = try a.dupe(AstId, children),
            .is_partial = is_partial,
            .invalid_utf8 = try a.dupe(InvalidUtf8, invalid_utf8),
        });
        return id;
    }

    /// Ports `get` (peg-parser.h:122 @620d2225c), a member of
    /// `common_peg_ast_arena`: `nodes_.at(id)`.
    ///
    /// Return: borrowed until the next `addNode` or `clear`; `OutOfRange`
    /// past the end.
    pub fn get(self: *const AstArena, id: AstId) Error!*const AstNode {
        if (id >= self.nodes.items.len) return errors.outOfRange();
        return &self.nodes.items[id];
    }

    /// Ports `find_by_tag` (peg-parser.cpp:137 @620d2225c): the first
    /// descendant with `tag`, depth first, `max_depth` levels down.
    ///
    /// Return: its id, or `invalid_ast_id`.
    pub fn findByTag(self: *const AstArena, parent: *const AstNode, tag: []const u8, max_depth: c_int) Error!AstId {
        for (parent.children) |child_id| {
            const child = try self.get(child_id);
            if (std.mem.eql(u8, child.tag, tag)) return child_id;
            if (max_depth > 1) {
                const r = try self.findByTag(child, tag, max_depth - 1);
                if (r != invalid_ast_id) return r;
            }
        }
        return invalid_ast_id;
    }

    /// Ports `find_by_rule` (peg-parser.cpp:153 @620d2225c): as `findByTag`,
    /// matching the rule name.
    ///
    /// Return: its id, or `invalid_ast_id`.
    pub fn findByRule(self: *const AstArena, parent: *const AstNode, rule: []const u8, max_depth: c_int) Error!AstId {
        for (parent.children) |child_id| {
            const child = try self.get(child_id);
            if (std.mem.eql(u8, child.rule, rule)) return child_id;
            if (max_depth > 1) {
                const r = try self.findByRule(child, rule, max_depth - 1);
                if (r != invalid_ast_id) return r;
            }
        }
        return invalid_ast_id;
    }

    /// Ports `visit` (peg-parser.cpp:188 @620d2225c), the overload taking an
    /// id: the node, then its children, depth first.
    ///
    /// Parameters:
    /// - `id`: where to start; `invalid_ast_id` visits nothing.
    /// - `ctx`, `f`: the visitor, called with each node.
    pub fn visit(self: *const AstArena, id: AstId, ctx: anytype, comptime f: fn (@TypeOf(ctx), *const AstNode) Error!void) Error!void {
        if (id == invalid_ast_id) return;
        const node = try self.get(id);
        try f(ctx, node);
        for (node.children) |child| try self.visit(child, ctx, f);
    }

    /// Ports `visit` (peg-parser.cpp:199 @620d2225c), the overload taking a
    /// result: every node it carries.
    pub fn visitResult(self: *const AstArena, result: *const ParseResult, ctx: anytype, comptime f: fn (@TypeOf(ctx), *const AstNode) Error!void) Error!void {
        for (result.nodes) |node| try self.visit(node, ctx, f);
    }

    /// Ports `size` (peg-parser.h:127 @620d2225c).
    pub fn size(self: *const AstArena) usize {
        return self.nodes.items.len;
    }

    /// Ports `clear` (peg-parser.h:129 @620d2225c).
    pub fn clear(self: *AstArena) void {
        self.nodes.clearRetainingCapacity();
        _ = self.mem.reset(.retain_capacity);
    }

    /// Ports `dump` (peg-parser.cpp:857 @620d2225c): every node and, under
    /// it, its children, indented -- so a node appears once for itself and
    /// once more under each ancestor.
    ///
    /// Return: owned by the caller.
    pub fn dump(self: *const AstArena, gpa: Allocator) Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        for (self.nodes.items) |*node| try self.bfsNode(gpa, &out, node, 0);
        return out.toOwnedSlice(gpa);
    }

    /// Ports `bfs_node` (peg-parser.cpp:840 @620d2225c).
    fn bfsNode(self: *const AstArena, gpa: Allocator, out: *std.ArrayList(u8), node: *const AstNode, indent: c_int) Error!void {
        var i: c_int = 0;
        while (i < indent) : (i += 1) try out.appendSlice(gpa, "  ");
        try out.print(gpa, "NODE {d}", .{node.id});
        if (node.rule.len > 0) try out.print(gpa, " (rule {s})", .{node.rule});
        if (node.tag.len > 0) try out.print(gpa, " (tag {s})", .{node.tag});
        try out.print(gpa, " ['{s}']\n", .{node.text});
        for (node.children) |child| try self.bfsNode(gpa, out, try self.get(child), indent + 1);
    }
};

/// Ports `common_peg_parse_result` (peg-parser.h:137 @620d2225c).
pub const ParseResult = struct {
    type: ResultType = .fail,
    start: usize = 0,
    end: usize = 0,
    /// borrowed from the parse context's scratch space
    nodes: []const AstId = &.{},
    /// the invalid UTF-8 this result consumed, carried up to the enclosing
    /// AST nodes; borrowed from the parse context's scratch space
    invalid_utf8: []const InvalidUtf8 = &.{},

    fn at(t: ResultType, start: usize) ParseResult {
        return .{ .type = t, .start = start, .end = start };
    }

    fn span(t: ResultType, start: usize, end: usize) ParseResult {
        return .{ .type = t, .start = start, .end = end };
    }

    pub fn fail(self: ParseResult) bool {
        return self.type == .fail;
    }

    pub fn needMoreInput(self: ParseResult) bool {
        return self.type == .need_more_input;
    }

    pub fn success(self: ParseResult) bool {
        return self.type == .success;
    }
};

/// Ports `common_peg_parse_context` (peg-parser.h:185 @620d2225c).
pub const ParseContext = struct {
    gpa: Allocator,
    input: []u8, // [h,   e,  l,  l,  o,  _,  w,  o,  r,  l,  d]
    /// One entry per input byte: the token that begins there, -1
    /// (`LLAMA_TOKEN_NULL`) inside a token's text; empty when the input
    /// came without tokens. Nothing in the parser reads it yet, as in
    /// upstream.
    tokens: []i32 = &.{}, // [id, -1, -1, -1, -1, id, -1, -1, -1, -1, -1]
    flags: u32,
    ast: AstArena,
    parse_depth: c_int = 0,
    /// the node lists results carry
    scratch: std.heap.ArenaAllocator,

    /// Ports `common_peg_parse_context` (peg-parser.h:196 @620d2225c), the
    /// constructor taking input, which it copies.
    ///
    /// Return: the context; `deinit` frees it.
    pub fn init(gpa: Allocator, input: []const u8, flags: u32) Error!ParseContext {
        return .{
            .gpa = gpa,
            .input = try gpa.dupe(u8, input),
            .flags = flags,
            .ast = .init(gpa),
            .scratch = .init(gpa),
        };
    }

    /// Ports `common_peg_parse_context` (peg-parser.h:199 @620d2225c), the
    /// constructor taking input and its tokens, both copied: `tokens` is
    /// empty or one per input byte, a `GGML_ASSERT` kept in every build
    /// mode.
    ///
    /// Return: the context; `deinit` frees it.
    pub fn initTokens(gpa: Allocator, input: []const u8, tokens: []const i32, flags: u32) Error!ParseContext {
        if (!(tokens.len == 0 or tokens.len == input.len)) {
            const text = "GGML_ASSERT(this->tokens.empty() || this->tokens.size() == this->input.size()) failed\n";
            _ = std.c.write(2, text.ptr, text.len);
            std.c.abort();
        }
        var self = try init(gpa, input, flags);
        errdefer self.deinit();
        self.tokens = try gpa.dupe(i32, tokens);
        return self;
    }

    /// Frees the input copy, the AST and the scratch space.
    pub fn deinit(self: *ParseContext) void {
        self.gpa.free(self.tokens);
        self.gpa.free(self.input);
        self.ast.deinit();
        self.scratch.deinit();
    }

    pub fn isLenient(self: *const ParseContext) bool {
        return self.flags & Flags.lenient != 0;
    }

    pub fn isDebug(self: *const ParseContext) bool {
        return self.flags & Flags.debug != 0;
    }
};

/// Ports `char_range` (peg-parser.h:248 @620d2225c), a member of
/// `common_peg_chars_parser`.
pub const CharRange = struct {
    start: u32,
    end: u32,

    pub fn contains(self: CharRange, cp: u32) bool {
        return cp >= self.start and cp <= self.end;
    }
};

/// A schema document several parsers -- and arenas -- may share, as the
/// C++ shares a `common_chat_schema_document_ptr`.
pub const SharedDoc = struct {
    gpa: Allocator,
    doc: schema.Document,
    refs: usize = 1,

    /// Parses `value` into a document of its own, held once.
    ///
    /// Return: the document, released with `release`.
    pub fn fromJson(gpa: Allocator, value: *const json.Value) Error!*SharedDoc {
        const doc = schema.fromJsonSchema(gpa, value) catch |e| return errors.fromSchema(e);
        const self = try gpa.create(SharedDoc);
        self.* = .{ .gpa = gpa, .doc = doc };
        return self;
    }

    pub fn retain(self: *SharedDoc) *SharedDoc {
        self.refs += 1;
        return self;
    }

    pub fn release(self: *SharedDoc) void {
        self.refs -= 1;
        if (self.refs > 0) return;
        self.doc.deinit();
        self.gpa.destroy(self);
    }
};

/// Ports `common_peg_parser_variant` (peg-parser.h:309 @620d2225c), each
/// arm one of the variant's structs, in its order.
pub const Parser = union(enum) {
    /// `common_peg_epsilon_parser`
    epsilon,
    /// `common_peg_start_parser`
    start,
    /// `common_peg_end_parser`
    end,
    /// `common_peg_literal_parser`
    literal: []const u8,
    /// `common_peg_sequence_parser`
    sequence: []ParserId,
    /// `common_peg_choice_parser`
    choice: []ParserId,
    /// `common_peg_repetition_parser`; a max of -1 is unbounded
    repetition: struct { child: ParserId, min_count: c_int, max_count: c_int },
    /// `common_peg_and_parser`
    and_: ParserId,
    /// `common_peg_not_parser`
    not_: ParserId,
    /// `common_peg_any_parser`
    any,
    /// `common_peg_space_parser`
    space,
    /// `common_peg_chars_parser`
    chars: Chars,
    /// `common_peg_string_parser`
    string: u8,
    /// `common_peg_until_parser`
    until: []const []const u8,
    /// `common_peg_schema_parser`
    schema: Schema,
    /// `common_peg_rule_parser`
    rule: struct { name: []const u8, child: ParserId, trigger: bool },
    /// `common_peg_ref_parser`
    ref: []const u8,
    /// `common_peg_atomic_parser`
    atomic: ParserId,
    /// `common_peg_tag_parser`
    tag: struct { child: ParserId, tag: []const u8 },
    /// `common_peg_gbnf_parser`
    gbnf: struct { child: ParserId, grammar: []const u8 },
    /// `common_peg_ac_parser`
    ac: struct { child: ParserId, delimiters: []const []const u8 },

    /// Ports `common_peg_chars_parser` (peg-parser.h:247 @620d2225c).
    pub const Chars = struct {
        pattern: []const u8,
        ranges: []const CharRange,
        negated: bool,
        min_count: c_int,
        max_count: c_int,
    };

    /// Ports `common_peg_schema_parser` (peg-parser.h:269 @620d2225c). A
    /// deserialized one has neither document nor node.
    pub const Schema = struct {
        child: ParserId,
        name: []const u8,
        doc: ?*SharedDoc = null,
        node: ?*const schema.Node = null,
        raw: bool,
    };

    /// The single child of the arms that have one.
    pub fn childPtr(self: *Parser) ?*ParserId {
        return switch (self.*) {
            .repetition => |*p| &p.child,
            .and_, .not_, .atomic => |*c| c,
            .tag => |*p| &p.child,
            .gbnf => |*p| &p.child,
            .ac => |*p| &p.child,
            .rule => |*p| &p.child,
            .schema => |*p| &p.child,
            else => null,
        };
    }
};

/// Ports `common_peg_arena` (peg-parser.h:333 @620d2225c): the parsers, the
/// named rules and the root.
pub const Arena = struct {
    gpa: Allocator,
    parsers: std.ArrayList(Parser) = .empty,
    rules: libcxx.StringMap(ParserId),
    root: ParserId = invalid_parser_id,
    /// every string and list the parsers hold
    mem: std.heap.ArenaAllocator,

    /// Return: an empty arena; `deinit` frees it.
    pub fn init(gpa: Allocator) Error!Arena {
        return .{ .gpa = gpa, .rules = try .init(gpa), .mem = .init(gpa) };
    }

    /// Frees every parser and releases the schema documents they hold.
    pub fn deinit(self: *Arena) void {
        for (self.parsers.items) |p| switch (p) {
            .schema => |s| if (s.doc) |d| d.release(),
            else => {},
        };
        self.parsers.deinit(self.gpa);
        self.rules.deinit();
        self.mem.deinit();
    }

    /// The C++ copy of a `common_peg_arena`: every parser's payload copied
    /// into a new arena, the schema documents shared, the rules inserted in
    /// the source's iteration order. libc++'s copy of an `unordered_map`
    /// rehashes to the source's bucket count first, which this does not;
    /// the order only matters to a grammar built from the copy, and nothing
    /// builds one from a copy.
    ///
    /// Return: owned by `gpa`; `deinit` frees it.
    pub fn clone(self: *const Arena, gpa: Allocator) Error!Arena {
        var out = try Arena.init(gpa);
        errdefer out.deinit();
        const a = out.alloc();
        try out.parsers.ensureTotalCapacity(gpa, self.parsers.items.len);
        for (self.parsers.items) |p| {
            const q: Parser = switch (p) {
                .epsilon, .start, .end, .any, .space, .string, .repetition, .and_, .not_, .atomic => p,
                .literal => |l| .{ .literal = try a.dupe(u8, l) },
                .sequence => |c| .{ .sequence = try a.dupe(ParserId, c) },
                .choice => |c| .{ .choice = try a.dupe(ParserId, c) },
                .chars => |c| .{ .chars = .{
                    .pattern = try a.dupe(u8, c.pattern),
                    .ranges = try a.dupe(CharRange, c.ranges),
                    .negated = c.negated,
                    .min_count = c.min_count,
                    .max_count = c.max_count,
                } },
                .until => |d| .{ .until = try dupeStrings(a, d) },
                .schema => |sc| .{ .schema = .{
                    .child = sc.child,
                    .name = try a.dupe(u8, sc.name),
                    .doc = if (sc.doc) |d| d.retain() else null,
                    .node = sc.node,
                    .raw = sc.raw,
                } },
                .rule => |r| .{ .rule = .{ .name = try a.dupe(u8, r.name), .child = r.child, .trigger = r.trigger } },
                .ref => |r| .{ .ref = try a.dupe(u8, r) },
                .tag => |t| .{ .tag = .{ .child = t.child, .tag = try a.dupe(u8, t.tag) } },
                .gbnf => |g| .{ .gbnf = .{ .child = g.child, .grammar = try a.dupe(u8, g.grammar) } },
                .ac => |c| .{ .ac = .{ .child = c.child, .delimiters = try dupeStrings(a, c.delimiters) } },
            };
            out.parsers.appendAssumeCapacity(q);
        }
        var it = self.rules.iterator();
        while (it.next()) |e| try out.addRule(e.key, e.value.*);
        out.root = self.root;
        return out;
    }

    fn dupeStrings(a: Allocator, list: []const []const u8) Error![]const []const u8 {
        const out = try a.alloc([]const u8, list.len);
        for (list, out) |s, *o| o.* = try a.dupe(u8, s);
        return out;
    }

    /// The allocator parser payloads live in; never stored, since an arena
    /// moves.
    pub fn alloc(self: *Arena) Allocator {
        return self.mem.allocator();
    }

    /// Ports `get` (peg-parser.h:339 @620d2225c), a member of
    /// `common_peg_arena`: `parsers_.at(id)`.
    ///
    /// Return: borrowed until the next `addParser`; `OutOfRange` past the
    /// end.
    pub fn get(self: *const Arena, id: ParserId) Error!*Parser {
        if (id >= self.parsers.items.len) return errors.outOfRange();
        return &self.parsers.items[id];
    }

    /// Ports `size` (peg-parser.h:342 @620d2225c), a member of `common_peg_arena`.
    pub fn size(self: *const Arena) usize {
        return self.parsers.items.len;
    }

    /// Ports `empty` (peg-parser.h:343 @620d2225c), a member of `common_peg_arena`.
    pub fn empty(self: *const Arena) bool {
        return self.parsers.items.len == 0;
    }

    /// Ports `add_parser` (peg-parser.cpp:207 @620d2225c).
    ///
    /// Return: the new parser's id.
    pub fn addParser(self: *Arena, parser: Parser) Error!ParserId {
        const id = self.parsers.items.len;
        try self.parsers.append(self.gpa, parser);
        return id;
    }

    /// Ports `add_rule` (peg-parser.cpp:213 @620d2225c): `rules_[name] = id`.
    pub fn addRule(self: *Arena, name: []const u8, id: ParserId) Error!void {
        const slot = try self.rules.getOrPut(name, id);
        slot.value.* = id;
    }

    /// Ports `get_rule` (peg-parser.cpp:217 @620d2225c).
    ///
    /// Return: the rule's parser id; `RuntimeError` when there is none.
    pub fn getRule(self: *const Arena, name: []const u8) Error!ParserId {
        if (self.rules.get(name)) |id| return id.*;
        return raise(error.RuntimeError, &.{ "Rule not found: ", name });
    }

    /// Ports `has_rule` (peg-parser.h:346 @620d2225c).
    pub fn hasRule(self: *const Arena, name: []const u8) bool {
        return self.rules.get(name) != null;
    }

    /// Ports `parse` (peg-parser.cpp:818 @620d2225c), the overload taking
    /// the root.
    ///
    /// Return: the result; its node list is borrowed from `ctx`.
    pub fn parse(self: *const Arena, ctx: *ParseContext, start: usize) Error!ParseResult {
        if (self.root == invalid_parser_id) return raise(error.RuntimeError, &.{"No root parser set"});
        return self.parseAt(self.root, ctx, start);
    }

    /// Ports `parse` (peg-parser.cpp:825 @620d2225c), the overload taking
    /// an id.
    pub fn parseAt(self: *const Arena, id: ParserId, ctx: *ParseContext, start: usize) Error!ParseResult {
        const parser = try self.get(id);
        var exec: Executor = .{ .arena = self, .ctx = ctx, .start_pos = start };
        return exec.run(parser.*);
    }

    /// Ports `resolve_ref` (peg-parser.cpp:832 @620d2225c).
    fn resolveRef(self: *Arena, id: ParserId) Error!ParserId {
        const parser = try self.get(id);
        return switch (parser.*) {
            .ref => |name| self.getRule(name),
            else => id,
        };
    }

    /// Ports `resolve_refs` (peg-parser.cpp:865 @620d2225c): every child
    /// that is a reference becomes the rule it names, and so does the root.
    pub fn resolveRefs(self: *Arena) Error!void {
        for (0..self.parsers.items.len) |i| {
            switch (self.parsers.items[i]) {
                .sequence, .choice => |children| for (children) |*c| {
                    c.* = try self.resolveRef(c.*);
                },
                else => if (self.parsers.items[i].childPtr()) |c| {
                    c.* = try self.resolveRef(c.*);
                },
            }
        }
        if (self.root != invalid_parser_id) self.root = try self.resolveRef(self.root);
    }

    /// Ports `dump` (peg-parser.cpp:914 @620d2225c): a parser as text, a
    /// parser met twice on the way down printed as `[cycle]`.
    ///
    /// Return: owned by the caller.
    pub fn dump(self: *const Arena, gpa: Allocator, id: ParserId) Error![]u8 {
        var visited: std.AutoHashMapUnmanaged(ParserId, void) = .empty;
        defer visited.deinit(gpa);
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try self.dumpImpl(gpa, &out, id, &visited);
        return out.toOwnedSlice(gpa);
    }

    /// Ports `dump_impl` (peg-parser.cpp:919 @620d2225c).
    fn dumpImpl(self: *const Arena, gpa: Allocator, out: *std.ArrayList(u8), id: ParserId, visited: *std.AutoHashMapUnmanaged(ParserId, void)) Error!void {
        if (visited.contains(id)) return out.appendSlice(gpa, "[cycle]");
        try visited.put(gpa, id, {});
        const parser = try self.get(id);
        switch (parser.*) {
            .epsilon => try out.appendSlice(gpa, "Epsilon"),
            .start => try out.appendSlice(gpa, "Start"),
            .end => try out.appendSlice(gpa, "End"),
            .literal => |l| try out.print(gpa, "Literal({s})", .{l}),
            .sequence, .choice => |children| {
                try out.appendSlice(gpa, if (parser.* == .sequence) "Sequence(" else "Choice(");
                for (children, 0..) |c, i| {
                    if (i > 0) try out.appendSlice(gpa, ", ");
                    try self.dumpImpl(gpa, out, c, visited);
                }
                try out.appendSlice(gpa, ")");
            },
            .repetition => |p| {
                try out.appendSlice(gpa, "Repetition(");
                try self.dumpImpl(gpa, out, p.child, visited);
                if (p.max_count == -1) {
                    try out.print(gpa, ", {d}, unbounded)", .{p.min_count});
                } else {
                    try out.print(gpa, ", {d}, {d})", .{ p.min_count, p.max_count });
                }
            },
            .and_ => |c| try self.dumpWrapped(gpa, out, "And(", c, visited),
            .not_ => |c| try self.dumpWrapped(gpa, out, "Not(", c, visited),
            .atomic => |c| try self.dumpWrapped(gpa, out, "Atomic(", c, visited),
            .gbnf => |p| {
                try out.print(gpa, "Gbnf({s}, ", .{p.grammar});
                try self.dumpImpl(gpa, out, p.child, visited);
                try out.appendSlice(gpa, ")");
            },
            .ac => |p| {
                try out.appendSlice(gpa, "Ac(");
                try joinTo(gpa, out, p.delimiters, " | ");
                try out.appendSlice(gpa, ", ");
                try self.dumpImpl(gpa, out, p.child, visited);
                try out.appendSlice(gpa, ")");
            },
            .any => try out.appendSlice(gpa, "Any"),
            .space => try out.appendSlice(gpa, "Space"),
            .chars => |p| {
                if (p.max_count == -1) {
                    try out.print(gpa, "CharRepeat({s}, {d}, unbounded)", .{ p.pattern, p.min_count });
                } else {
                    try out.print(gpa, "CharRepeat({s}, {d}, {d})", .{ p.pattern, p.min_count, p.max_count });
                }
            },
            .string => |d| try out.print(gpa, "String({c})", .{d}),
            .until => |delims| {
                try out.appendSlice(gpa, "Until(");
                try joinTo(gpa, out, delims, " | ");
                try out.appendSlice(gpa, ")");
            },
            .schema => |p| {
                try out.appendSlice(gpa, "Schema(");
                try self.dumpImpl(gpa, out, p.child, visited);
                try out.print(gpa, ", {s})", .{if (p.node) |n| schema.kindName(n.kind()) else "null"});
            },
            .rule => |p| {
                try out.print(gpa, "Rule({s}, ", .{p.name});
                try self.dumpImpl(gpa, out, p.child, visited);
                try out.appendSlice(gpa, ")");
            },
            .ref => |name| try out.print(gpa, "Ref({s})", .{name}),
            .tag => |p| {
                // a fresh `dump`, not `dump_impl`: the cycle check restarts
                const inner = try self.dump(gpa, p.child);
                defer gpa.free(inner);
                try out.print(gpa, "Tag({s}, {s})", .{ p.tag, inner });
            },
        }
    }

    fn dumpWrapped(self: *const Arena, gpa: Allocator, out: *std.ArrayList(u8), open: []const u8, child: ParserId, visited: *std.AutoHashMapUnmanaged(ParserId, void)) Error!void {
        try out.appendSlice(gpa, open);
        try self.dumpImpl(gpa, out, child, visited);
        try out.appendSlice(gpa, ")");
    }
};

/// `string_join` (common.cpp), appended.
pub fn joinTo(gpa: Allocator, out: *std.ArrayList(u8), parts: []const []const u8, sep: []const u8) Error!void {
    for (parts, 0..) |p, i| {
        if (i > 0) try out.appendSlice(gpa, sep);
        try out.appendSlice(gpa, p);
    }
}

/// A string as `printf`'s `%s` prints its `c_str()`: up to the first NUL.
pub fn cstr(s: []const u8) []const u8 {
    return if (std.mem.indexOfScalar(u8, s, 0)) |n| s[0..n] else s;
}

/// `std::isspace` in the `"C"` locale.
pub fn isCSpace(c: u8) bool {
    return c == ' ' or (c >= '\t' and c <= '\r');
}

/// Ports `is_hex_digit` (peg-parser.cpp:30 @620d2225c).
pub fn isHexDigit(c: u8) bool {
    return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
}

/// `debug_indent`'s spaces, for any depth a stack can reach.
const spaces: [1 << 16]u8 = @splat(' ');

/// Ports `parser_executor` (peg-parser.cpp:225 @620d2225c): one parser run
/// at one position.
const Executor = struct {
    arena: *const Arena,
    ctx: *ParseContext,
    start_pos: usize,

    fn res(t: ResultType, start: usize) ParseResult {
        return ParseResult.at(t, start);
    }

    fn span(t: ResultType, start: usize, end: usize) ParseResult {
        return ParseResult.span(t, start, end);
    }

    fn withNodes(self: *Executor, t: ResultType, start: usize, end: usize, nodes: []const AstId, invalid_utf8: []const InvalidUtf8) Error!ParseResult {
        _ = self;
        return .{ .type = t, .start = start, .end = end, .nodes = nodes, .invalid_utf8 = invalid_utf8 };
    }

    fn scratch(self: *Executor) Allocator {
        return self.ctx.scratch.allocator();
    }

    fn run(self: *Executor, parser: Parser) Error!ParseResult {
        return switch (parser) {
            .epsilon => res(.success, self.start_pos),
            .start => res(if (self.start_pos == 0) .success else .fail, self.start_pos),
            .end => res(if (self.start_pos >= self.ctx.input.len) .success else .fail, self.start_pos),
            .literal => |l| self.literal(l),
            .sequence => |c| self.sequence(c),
            .choice => |c| self.choice(c),
            .repetition => |p| self.repetition(p.child, p.min_count, p.max_count),
            .and_ => |c| self.andParser(c),
            .not_ => |c| self.notParser(c),
            .any => self.any(),
            .space => self.space(),
            .chars => |p| self.chars(p),
            .string => |d| self.string(d),
            .until => |d| self.until(d),
            .schema => |p| self.arena.parseAt(p.child, self.ctx, self.start_pos),
            .rule => |p| self.named(p.child, p.name, ""),
            .tag => |p| self.tagged(p.child, p.tag),
            .ref => |name| self.ref(name),
            .atomic => |c| self.atomic(c),
            .gbnf => |p| self.arena.parseAt(p.child, self.ctx, self.start_pos),
            .ac => |p| self.arena.parseAt(p.child, self.ctx, self.start_pos),
        };
    }

    /// Ports `debug_indent` (peg-parser.cpp:233 @620d2225c).
    fn indent(self: *const Executor) []const u8 {
        const n: usize = @intCast(@max(self.ctx.parse_depth, 0) * 2);
        return spaces[0..@min(n, spaces.len)];
    }

    /// Ports `debug_input_snippet` (peg-parser.cpp:235 @620d2225c).
    fn snippet(self: *const Executor, gpa: Allocator, pos: usize) Error![]u8 {
        const len = 60;
        const input = self.ctx.input;
        if (pos >= input.len) return gpa.dupe(u8, "<EOF>");
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        for (input[pos..@min(pos + len, input.len)]) |c| switch (c) {
            '\n' => try out.appendSlice(gpa, "\\n"),
            '\r' => try out.appendSlice(gpa, "\\r"),
            '\t' => try out.appendSlice(gpa, "\\t"),
            else => try out.append(gpa, c),
        };
        if (pos + len < input.len) try out.appendSlice(gpa, "...");
        return out.toOwnedSlice(gpa);
    }

    fn dumpOf(self: *const Executor, id: ParserId) Error![]u8 {
        return self.arena.dump(std.heap.smp_allocator, id);
    }

    /// Ports the executor's arm for `common_peg_literal_parser` (peg-parser.cpp:277 @620d2225c).
    fn literal(self: *Executor, lit: []const u8) ParseResult {
        var pos = self.start_pos;
        const input = self.ctx.input;
        for (lit) |c| {
            if (pos >= input.len) {
                if (!self.ctx.isLenient()) return res(.fail, self.start_pos);
                return span(.need_more_input, self.start_pos, pos);
            }
            if (input[pos] != c) return res(.fail, self.start_pos);
            pos += 1;
        }
        return span(.success, self.start_pos, pos);
    }

    /// Ports the executor's arm for `common_peg_sequence_parser` (peg-parser.cpp:295 @620d2225c).
    fn sequence(self: *Executor, children: []const ParserId) Error!ParseResult {
        const ctx = self.ctx;
        const a = std.heap.smp_allocator;
        if (ctx.isDebug()) {
            const s = try self.snippet(a, self.start_pos);
            defer a.free(s);
            try errors.logDbg("{s}SEQ start at {d} '{s}' ({d} children)\n", .{ self.indent(), self.start_pos, cstr(s), children.len });
        }
        ctx.parse_depth += 1;

        var pos = self.start_pos;
        var nodes: std.ArrayList(AstId) = .empty;
        var invalid_utf8: std.ArrayList(InvalidUtf8) = .empty;
        for (children, 0..) |child_id, i| {
            if (ctx.isDebug()) {
                const d = try self.dumpOf(child_id);
                defer a.free(d);
                try errors.eprint("{s}SEQ child {d}: {s}\n", .{ self.indent(), i, cstr(d) });
            }
            const result = try self.arena.parseAt(child_id, ctx, pos);
            if (ctx.isDebug()) {
                try errors.eprint("{s}SEQ child {d}: {s} at {d}->{d}\n", .{ self.indent(), i, result.type.name(), result.start, result.end });
            }
            if (result.fail()) {
                ctx.parse_depth -= 1;
                if (ctx.isDebug()) try errors.eprint("{s}SEQ -> FAIL\n", .{self.indent()});
                return span(.fail, self.start_pos, result.end);
            }
            if (result.nodes.len > 0) try nodes.appendSlice(self.scratch(), result.nodes);
            try invalid_utf8.appendSlice(self.scratch(), result.invalid_utf8);
            if (result.needMoreInput()) {
                ctx.parse_depth -= 1;
                if (ctx.isDebug()) try errors.eprint("{s}SEQ -> NEED_MORE\n", .{self.indent()});
                return self.withNodes(.need_more_input, self.start_pos, result.end, nodes.items, invalid_utf8.items);
            }
            pos = result.end;
        }
        ctx.parse_depth -= 1;
        if (ctx.isDebug()) try errors.eprint("{s}SEQ -> SUCCESS at {d}->{d}\n", .{ self.indent(), self.start_pos, pos });
        return self.withNodes(.success, self.start_pos, pos, nodes.items, invalid_utf8.items);
    }

    /// Ports the executor's arm for `common_peg_choice_parser` (peg-parser.cpp:349 @620d2225c).
    fn choice(self: *Executor, children: []const ParserId) Error!ParseResult {
        const ctx = self.ctx;
        const a = std.heap.smp_allocator;
        if (ctx.isDebug()) {
            const s = try self.snippet(a, self.start_pos);
            defer a.free(s);
            try errors.eprint("{s}CHOICE start at {d} '{s}' ({d} options)\n", .{ self.indent(), self.start_pos, cstr(s), children.len });
        }
        ctx.parse_depth += 1;
        const pos = self.start_pos;
        for (children, 0..) |child_id, i| {
            if (ctx.isDebug()) {
                const d = try self.dumpOf(child_id);
                defer a.free(d);
                try errors.eprint("{s}CHOICE option {d}: {s}\n", .{ self.indent(), i, cstr(d) });
            }
            const result = try self.arena.parseAt(child_id, ctx, pos);
            if (ctx.isDebug()) try errors.eprint("{s}CHOICE option {d}: {s}\n", .{ self.indent(), i, result.type.name() });
            if (!result.fail()) {
                ctx.parse_depth -= 1;
                if (ctx.isDebug()) try errors.eprint("{s}CHOICE -> {s} (option {d})\n", .{ self.indent(), result.type.name(), i });
                return result;
            }
        }
        ctx.parse_depth -= 1;
        if (ctx.isDebug()) try errors.eprint("{s}CHOICE -> FAIL (no options matched)\n", .{self.indent()});
        return res(.fail, self.start_pos);
    }

    /// Ports the executor's arm for `common_peg_repetition_parser` (peg-parser.cpp:384 @620d2225c).
    fn repetition(self: *Executor, child: ParserId, min_count: c_int, max_count: c_int) Error!ParseResult {
        const ctx = self.ctx;
        const a = std.heap.smp_allocator;
        if (ctx.isDebug()) {
            const s = try self.snippet(a, self.start_pos);
            defer a.free(s);
            try errors.eprint("{s}REPEAT start at {d} '{s}' (min={d}, max={d})\n", .{ self.indent(), self.start_pos, cstr(s), min_count, max_count });
        }
        ctx.parse_depth += 1;

        var pos = self.start_pos;
        var match_count: c_int = 0;
        var nodes: std.ArrayList(AstId) = .empty;
        var invalid_utf8: std.ArrayList(InvalidUtf8) = .empty;
        while (max_count == -1 or match_count < max_count) {
            if (pos >= ctx.input.len) {
                if (ctx.isDebug()) try errors.eprint("{s}REPEAT: at end of input, count={d}\n", .{ self.indent(), match_count });
                break;
            }
            const result = try self.arena.parseAt(child, ctx, pos);
            if (ctx.isDebug()) {
                try errors.eprint("{s}REPEAT iter {d}: {s} at {d}->{d}, nodes={d}\n", .{ self.indent(), match_count, result.type.name(), result.start, result.end, result.nodes.len });
                const d = try self.dumpOf(child);
                defer a.free(d);
                try errors.eprint("{s}REPEAT CHILD: {s}\n", .{ self.indent(), cstr(d) });
            }
            if (result.success()) {
                if (result.end == pos) {
                    if (ctx.isDebug()) try errors.eprint("{s}  REPEAT: empty match, stopping\n", .{self.indent()});
                    break;
                }
                if (result.nodes.len > 0) try nodes.appendSlice(self.scratch(), result.nodes);
                try invalid_utf8.appendSlice(self.scratch(), result.invalid_utf8);
                pos = result.end;
                match_count += 1;
                continue;
            }
            if (result.needMoreInput()) {
                if (result.nodes.len > 0) try nodes.appendSlice(self.scratch(), result.nodes);
                try invalid_utf8.appendSlice(self.scratch(), result.invalid_utf8);
                ctx.parse_depth -= 1;
                if (ctx.isDebug()) try errors.eprint("{s}REPEAT -> NEED_MORE (count={d}, nodes={d})\n", .{ self.indent(), match_count, nodes.items.len });
                return self.withNodes(.need_more_input, self.start_pos, result.end, nodes.items, invalid_utf8.items);
            }
            if (ctx.isDebug()) try errors.eprint("{s}REPEAT: child failed, stopping\n", .{self.indent()});
            break;
        }

        if (min_count > 0 and match_count < min_count) {
            ctx.parse_depth -= 1;
            if (pos >= ctx.input.len and ctx.isLenient()) {
                if (ctx.isDebug()) try errors.eprint("{s}REPEAT -> NEED_MORE (not enough matches: {d} < {d})\n", .{ self.indent(), match_count, min_count });
                return self.withNodes(.need_more_input, self.start_pos, pos, nodes.items, invalid_utf8.items);
            }
            if (ctx.isDebug()) try errors.eprint("{s}REPEAT -> FAIL (not enough matches: {d} < {d})\n", .{ self.indent(), match_count, min_count });
            return span(.fail, self.start_pos, pos);
        }

        ctx.parse_depth -= 1;
        if (ctx.isDebug()) try errors.eprint("{s}REPEAT -> SUCCESS (count={d}, nodes={d})\n", .{ self.indent(), match_count, nodes.items.len });
        return self.withNodes(.success, self.start_pos, pos, nodes.items, invalid_utf8.items);
    }

    /// Ports the executor's arm for `common_peg_and_parser` (peg-parser.cpp:478 @620d2225c).
    fn andParser(self: *Executor, child: ParserId) Error!ParseResult {
        const result = try self.arena.parseAt(child, self.ctx, self.start_pos);
        return res(result.type, self.start_pos);
    }

    /// Ports the executor's arm for `common_peg_not_parser` (peg-parser.cpp:484 @620d2225c).
    fn notParser(self: *Executor, child: ParserId) Error!ParseResult {
        const result = try self.arena.parseAt(child, self.ctx, self.start_pos);
        if (result.success()) return res(.fail, self.start_pos);
        if (result.needMoreInput()) return res(.need_more_input, self.start_pos);
        return res(.success, self.start_pos);
    }

    /// Ports the executor's arm for `common_peg_any_parser` (peg-parser.cpp:501 @620d2225c):
    /// one UTF-8 code point.
    fn any(self: *Executor) ParseResult {
        const r = trie.parseUtf8Codepoint(self.ctx.input, self.start_pos);
        switch (r.status) {
            .incomplete => return res(if (self.ctx.isLenient()) .need_more_input else .fail, self.start_pos),
            .invalid => return res(.fail, self.start_pos),
            .success => return span(.success, self.start_pos, self.start_pos + r.bytes_consumed),
        }
    }

    /// Ports the executor's arm for `common_peg_space_parser` (peg-parser.cpp:517 @620d2225c).
    fn space(self: *Executor) ParseResult {
        var pos = self.start_pos;
        while (pos < self.ctx.input.len and isCSpace(self.ctx.input[pos])) pos += 1;
        return span(.success, self.start_pos, pos);
    }

    /// Ports the executor's arm for `common_peg_chars_parser` (peg-parser.cpp:531 @620d2225c).
    fn chars(self: *Executor, p: Parser.Chars) ParseResult {
        const ctx = self.ctx;
        var pos = self.start_pos;
        var match_count: c_int = 0;
        while (p.max_count == -1 or match_count < p.max_count) {
            const r = trie.parseUtf8Codepoint(ctx.input, pos);
            if (r.status == .incomplete) {
                if (match_count >= p.min_count) return span(.success, self.start_pos, pos);
                if (!ctx.isLenient()) return res(.fail, self.start_pos);
                return span(.need_more_input, self.start_pos, pos);
            }
            if (r.status == .invalid) {
                if (match_count >= p.min_count) return span(.success, self.start_pos, pos);
                return res(.fail, self.start_pos);
            }
            var matches = false;
            for (p.ranges) |range| {
                if (range.contains(r.codepoint)) {
                    matches = true;
                    break;
                }
            }
            if (p.negated) matches = !matches;
            if (!matches) break;
            pos += r.bytes_consumed;
            match_count += 1;
        }
        if (match_count < p.min_count) {
            if (pos >= ctx.input.len and ctx.isLenient()) return span(.need_more_input, self.start_pos, pos);
            return span(.fail, self.start_pos, pos);
        }
        return span(.success, self.start_pos, pos);
    }

    /// Ports `handle_escape_sequence` (peg-parser.cpp:595 @620d2225c).
    fn handleEscapeSequence(ctx: *ParseContext, start: usize, pos: *usize, delimiter: u8) ParseResult {
        const save = pos.*;
        pos.* += 1;
        if (pos.* >= ctx.input.len) {
            if (!ctx.isLenient()) return res(.fail, start);
            pos.* = save;
            return span(.need_more_input, start, pos.*);
        }
        const c = ctx.input[pos.*];
        if (c == delimiter or c == '\\' or c == '/' or c == 'b' or c == 'f' or c == 'n' or c == 'r' or c == 't') {
            pos.* += 1;
            return span(.success, start, pos.*);
        }
        if (c == 'u') {
            const result = handleUnicodeEscape(ctx, start, pos);
            if (result.needMoreInput()) {
                pos.* = save;
                return span(.need_more_input, start, pos.*);
            }
            return result;
        }
        return res(.fail, start);
    }

    /// Ports `handle_unicode_escape` (peg-parser.cpp:626 @620d2225c).
    fn handleUnicodeEscape(ctx: *ParseContext, start: usize, pos: *usize) ParseResult {
        pos.* += 1;
        for (0..4) |_| {
            if (pos.* >= ctx.input.len) {
                if (!ctx.isLenient()) return res(.fail, start);
                return span(.need_more_input, start, pos.*);
            }
            if (!isHexDigit(ctx.input[pos.*])) return res(.fail, start);
            pos.* += 1;
        }
        return span(.success, start, pos.*);
    }

    /// Ports the executor's arm for `common_peg_string_parser` (peg-parser.cpp:643 @620d2225c):
    /// string content up to an unescaped delimiter, which it leaves.
    fn string(self: *Executor, delimiter: u8) ParseResult {
        const ctx = self.ctx;
        var pos = self.start_pos;
        while (pos < ctx.input.len) {
            const c = ctx.input[pos];
            if (c == delimiter) return span(.success, self.start_pos, pos);
            if (c == '\\') {
                const result = handleEscapeSequence(ctx, self.start_pos, &pos, delimiter);
                if (!result.success()) return result;
            } else {
                const r = trie.parseUtf8Codepoint(ctx.input, pos);
                if (r.status == .incomplete) {
                    if (!ctx.isLenient()) return res(.fail, self.start_pos);
                    return span(.need_more_input, self.start_pos, pos);
                }
                if (r.status == .invalid) return res(.fail, self.start_pos);
                pos += r.bytes_consumed;
            }
        }
        if (!ctx.isLenient()) return span(.fail, self.start_pos, pos);
        return span(.need_more_input, self.start_pos, pos);
    }

    /// Ports the executor's arm for `common_peg_until_parser` (peg-parser.cpp:685 @620d2225c):
    /// everything before the first delimiter, or before a delimiter's prefix
    /// at the end of the input. Bytes that do not decode are consumed and
    /// recorded -- a delimiter cannot start inside them -- except a sequence
    /// cut short by the end of a lenient parse's input, which may still
    /// arrive.
    fn until(self: *Executor, delimiters: []const []const u8) Error!ParseResult {
        const ctx = self.ctx;
        var matcher = try trie.Trie.init(std.heap.smp_allocator, delimiters);
        defer matcher.deinit();
        var pos = self.start_pos;
        var last_valid_pos = self.start_pos;
        var invalid_utf8: std.ArrayList(InvalidUtf8) = .empty;
        while (pos < ctx.input.len) {
            const r = trie.parseUtf8Codepoint(ctx.input, pos);
            if (r.status == .incomplete and ctx.isLenient()) {
                return self.withNodes(.need_more_input, self.start_pos, last_valid_pos, &.{}, invalid_utf8.items);
            }
            if (r.status != .success) {
                try invalid_utf8.append(self.scratch(), .{ .pos = pos, .len = r.bytes_consumed });
                pos += r.bytes_consumed;
                last_valid_pos = pos;
                continue;
            }
            switch (matcher.checkAt(ctx.input, pos)) {
                .complete_match, .partial_match => return self.withNodes(.success, self.start_pos, pos, &.{}, invalid_utf8.items),
                .no_match => {},
            }
            pos += r.bytes_consumed;
            last_valid_pos = pos;
        }
        if (last_valid_pos == ctx.input.len and ctx.isLenient()) return self.withNodes(.need_more_input, self.start_pos, last_valid_pos, &.{}, invalid_utf8.items);
        return self.withNodes(.success, self.start_pos, last_valid_pos, &.{}, invalid_utf8.items);
    }

    /// Ports the executor's arms for `common_peg_rule_parser` (peg-parser.cpp:738 @620d2225c)
    /// and `common_peg_tag_parser` (peg-parser.cpp:765 @620d2225c): the
    /// child, and a node for it unless it failed; the node and the result
    /// both carry the child's invalid UTF-8.
    fn named(self: *Executor, child: ParserId, rule: []const u8, tag: []const u8) Error!ParseResult {
        const ctx = self.ctx;
        const result = try self.arena.parseAt(child, ctx, self.start_pos);
        if (result.fail()) return result;
        var text: []const u8 = "";
        if (result.start < ctx.input.len) {
            const n = result.end -% result.start;
            text = ctx.input[result.start..][0..@min(n, ctx.input.len - result.start)];
        }
        const node_id = try ctx.ast.addNode(rule, tag, result.start, result.end, text, result.nodes, result.needMoreInput(), result.invalid_utf8);
        const nodes = try self.scratch().alloc(AstId, 1);
        nodes[0] = node_id;
        return .{ .type = result.type, .start = result.start, .end = result.end, .nodes = nodes, .invalid_utf8 = result.invalid_utf8 };
    }

    fn tagged(self: *Executor, child: ParserId, tag: []const u8) Error!ParseResult {
        if (self.ctx.isDebug()) try errors.eprint("{s}TAG: {s}\n", .{ self.indent(), cstr(tag) });
        return self.named(child, "", tag);
    }

    /// Ports the executor's arm for `common_peg_ref_parser` (peg-parser.cpp:795 @620d2225c).
    fn ref(self: *Executor, name: []const u8) Error!ParseResult {
        const rule_id = try self.arena.getRule(name);
        return self.arena.parseAt(rule_id, self.ctx, self.start_pos);
    }

    /// Ports the executor's arm for `common_peg_atomic_parser` (peg-parser.cpp:800 @620d2225c):
    /// a partial match carries no nodes.
    fn atomic(self: *Executor, child: ParserId) Error!ParseResult {
        var result = try self.arena.parseAt(child, self.ctx, self.start_pos);
        if (result.needMoreInput()) result.nodes = &.{};
        return result;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a literal parses fully, partially, or not at all" {
    const a = std.testing.allocator;
    var arena = try Arena.init(a);
    defer arena.deinit();
    const lit = try arena.addParser(.{ .literal = "hello" });
    const tag = try arena.addParser(.{ .tag = .{ .child = lit, .tag = "greeting" } });
    arena.root = tag;

    var full = try ParseContext.init(a, "hello world", Flags.none);
    defer full.deinit();
    const r1 = try arena.parse(&full, 0);
    try std.testing.expect(r1.success());
    try std.testing.expectEqual(@as(usize, 5), r1.end);
    try std.testing.expectEqualStrings("hello", (try full.ast.get(r1.nodes[0])).text);

    var part = try ParseContext.init(a, "hel", Flags.lenient);
    defer part.deinit();
    const r2 = try arena.parse(&part, 0);
    try std.testing.expect(r2.needMoreInput());
    try std.testing.expect((try part.ast.get(r2.nodes[0])).is_partial);

    var none = try ParseContext.init(a, "hel", Flags.none);
    defer none.deinit();
    try std.testing.expect((try arena.parse(&none, 0)).fail());
}
