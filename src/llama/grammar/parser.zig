//! The GBNF parser: grammar text to rules of elements.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-grammar.cpp` and `llama-grammar.h`
//! (v0.3.0, `c1d0e7a00`). Each declaration below names the C++ it replaces
//! and the line it began at.
//!
//! # Exceptions
//!
//! The C++ parser throws `std::runtime_error` from anywhere in the descent
//! and `llama_grammar_parser::parse` catches it, prints it, and clears the
//! rules. Here every parse step returns `error.Parse` with the exception's
//! text left in `Parser.err`, and `parse` prints and clears exactly as the
//! C++ does.
//!
//! # `std::map` semantics
//!
//! `symbol_ids` is an `std::map<std::string, uint32_t>`. Only two of its
//! behaviours are observable and both are kept: `emplace` leaves an existing
//! key alone, and `operator[]` assignment overwrites one without growing the
//! map, so the next generated id repeats. The map's sorted order is never
//! read for anything but printing, which this port does not reach.

const std = @import("std");
const impl = @import("../impl.zig");

const c = impl.c;
const src = "llama-grammar.cpp";

/// Ports `llama_gretype` (llama-grammar.h:13 @c1d0e7a00).
pub const GreType = enum(c_uint) {
    end = 0,
    alt = 1,
    rule_ref = 2,
    char = 3,
    char_not = 4,
    char_rng_upper = 5,
    char_alt = 6,
    char_any = 7,
    token = 8,
    token_not = 9,
};

/// Ports `llama_grammar_element` (llama-grammar.h:47 @c1d0e7a00).
pub const Element = extern struct {
    type: GreType,
    /// A code point, rule id or token id, by `type`.
    value: u32,
};

/// Ports `llama_grammar_rule` (llama-grammar.h:64 @c1d0e7a00).
pub const Rule = std.ArrayList(Element);

/// Ports `MAX_REPETITION_THRESHOLD` (llama-grammar.cpp:13 @c1d0e7a00).
const MAX_REPETITION_THRESHOLD: u64 = 2000;

const ParseError = error{Parse};

/// Ports the `const char *` overload of `decode_utf8` (llama-grammar.cpp:19 @c1d0e7a00).
///
/// Assumes valid UTF-8 but stops at a NUL.
///
/// Parameters:
/// - `s`: the input, NUL-terminated.
/// - `pos`: the index of the first byte.
///
/// Return: the code point and the index after it.
pub fn decodeUtf8At(s: [*:0]const u8, pos: usize) struct { u32, usize } {
    const lookup = [_]u32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    const first_byte = s[pos];
    const highbits = first_byte >> 4;
    const len = lookup[highbits];
    const mask: u8 = @truncate((@as(u32, 1) << @intCast(8 - len)) - 1);
    var value: u32 = first_byte & mask;
    const end = pos + len;
    var p = pos + 1;
    while (p < end and s[p] != 0) : (p += 1) {
        value = (value << 6) + (s[p] & 0x3F);
    }
    return .{ value, p };
}

/// Ports `is_digit_char` (llama-grammar.cpp:94 @c1d0e7a00).
fn isDigitChar(ch: u8) bool {
    return '0' <= ch and ch <= '9';
}

/// Ports `is_word_char` (llama-grammar.cpp:98 @c1d0e7a00).
fn isWordChar(ch: u8) bool {
    return ('a' <= ch and ch <= 'z') or ('A' <= ch and ch <= 'Z') or ch == '-' or isDigitChar(ch);
}

/// Ports `parse_space` (llama-grammar.cpp:125 @c1d0e7a00).
fn parseSpace(s: [*:0]const u8, start: usize, newline_ok: bool) usize {
    var pos = start;
    while (s[pos] == ' ' or s[pos] == '\t' or s[pos] == '#' or
        (newline_ok and (s[pos] == '\r' or s[pos] == '\n')))
    {
        if (s[pos] == '#') {
            while (s[pos] != 0 and s[pos] != '\r' and s[pos] != '\n') pos += 1;
        } else {
            pos += 1;
        }
    }
    return pos;
}

/// Ports `llama_grammar_parser` (llama-grammar.h:86 @c1d0e7a00).
pub const Parser = struct {
    vocab: ?*const c.struct_llama_vocab,
    /// Owned keys.
    symbol_ids: std.StringHashMapUnmanaged(u32) = .empty,
    rules: std.ArrayList(Rule) = .empty,

    /// The text of the exception a failed step would have thrown.
    err: std.ArrayList(u8) = .empty,

    /// Ports the constructor of `llama_grammar_parser` (llama-grammar.h:92 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `vocab`: needed only for `<token>` references; may be null.
    ///
    /// Return: an empty parser.
    pub fn init(vocab: ?*const c.struct_llama_vocab) Parser {
        return .{ .vocab = vocab };
    }

    /// Frees the rules, the symbol table and the error text.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Parser) void {
        var it = self.symbol_ids.keyIterator();
        while (it.next()) |k| impl.allocator.free(k.*);
        self.symbol_ids.deinit(impl.allocator);
        self.clearRules();
        self.rules.deinit(impl.allocator);
        self.err.deinit(impl.allocator);
    }

    fn clearRules(self: *Parser) void {
        for (self.rules.items) |*r| r.deinit(impl.allocator);
        self.rules.clearRetainingCapacity();
    }

    /// Records an exception's text and returns the error that stands for it.
    fn fail(self: *Parser, comptime fmt: []const u8, args: anytype) ParseError {
        self.err.clearRetainingCapacity();
        self.err.print(impl.allocator, fmt, args) catch impl.abort(src, 0, "out of memory");
        return error.Parse;
    }

    fn oom() noreturn {
        impl.abort(src, 0, "out of memory");
    }

    /// Ports `llama_grammar_parser`'s `get_symbol_id` (llama-grammar.cpp:416 @c1d0e7a00).
    ///
    /// Return: the name's id, assigning the next one when it is new.
    fn getSymbolId(self: *Parser, name: []const u8) u32 {
        const next_id: u32 = @intCast(self.symbol_ids.count());
        if (self.symbol_ids.get(name)) |id| return id;
        const key = impl.allocator.dupe(u8, name) catch oom();
        self.symbol_ids.put(impl.allocator, key, next_id) catch oom();
        return next_id;
    }

    /// Ports `llama_grammar_parser`'s `generate_symbol_id` (llama-grammar.cpp:422 @c1d0e7a00).
    ///
    /// Return: the map's size before the call, which `operator[]` assigns to
    /// `base_name + '_' + id` -- overwriting, not growing, if that name
    /// already exists.
    fn generateSymbolId(self: *Parser, base_name: []const u8) u32 {
        const next_id: u32 = @intCast(self.symbol_ids.count());
        const name = impl.allocator.print("{s}_{d}", .{ base_name, next_id }) catch oom();
        const gop = self.symbol_ids.getOrPut(impl.allocator, name) catch oom();
        if (gop.found_existing) {
            impl.allocator.free(name);
        }
        gop.value_ptr.* = next_id;
        return next_id;
    }

    /// Ports `llama_grammar_parser`'s `add_rule` (llama-grammar.cpp:428 @c1d0e7a00).
    fn addRule(self: *Parser, rule_id: u32, rule: []const Element) void {
        while (self.rules.items.len <= rule_id) {
            self.rules.append(impl.allocator, .empty) catch oom();
        }
        const r = &self.rules.items[rule_id];
        r.clearRetainingCapacity();
        r.appendSlice(impl.allocator, rule) catch oom();
    }

    /// Ports `parse_hex` (llama-grammar.cpp:102 @c1d0e7a00).
    fn parseHex(self: *Parser, s: [*:0]const u8, start: usize, size: usize) ParseError!struct { u32, usize } {
        var pos = start;
        const end = start + size;
        var value: u32 = 0;
        while (pos < end and s[pos] != 0) : (pos += 1) {
            value <<= 4;
            const ch = s[pos];
            if ('a' <= ch and ch <= 'f') {
                value += ch - 'a' + 10;
            } else if ('A' <= ch and ch <= 'F') {
                value += ch - 'A' + 10;
            } else if ('0' <= ch and ch <= '9') {
                value += ch - '0';
            } else {
                break;
            }
        }
        if (pos != end) {
            return self.fail("expecting {d} hex chars at {s}", .{ size, std.mem.span(s + start) });
        }
        return .{ value, pos };
    }

    /// Ports `parse_name` (llama-grammar.cpp:140 @c1d0e7a00).
    fn parseName(self: *Parser, s: [*:0]const u8, start: usize) ParseError!usize {
        var pos = start;
        while (isWordChar(s[pos])) pos += 1;
        if (pos == start) {
            return self.fail("expecting name at {s}", .{std.mem.span(s + start)});
        }
        return pos;
    }

    /// Ports `parse_int` (llama-grammar.cpp:151 @c1d0e7a00).
    fn parseInt(self: *Parser, s: [*:0]const u8, start: usize) ParseError!usize {
        var pos = start;
        while (isDigitChar(s[pos])) pos += 1;
        if (pos == start) {
            return self.fail("expecting integer at {s}", .{std.mem.span(s + start)});
        }
        return pos;
    }

    /// Ports `parse_char` (llama-grammar.cpp:162 @c1d0e7a00).
    fn parseChar(self: *Parser, s: [*:0]const u8, pos: usize) ParseError!struct { u32, usize } {
        if (s[pos] == '\\') {
            switch (s[pos + 1]) {
                'x' => return self.parseHex(s, pos + 2, 2),
                'u' => return self.parseHex(s, pos + 2, 4),
                'U' => return self.parseHex(s, pos + 2, 8),
                't' => return .{ '\t', pos + 2 },
                'r' => return .{ '\r', pos + 2 },
                'n' => return .{ '\n', pos + 2 },
                '\\', '"', '[', ']', '-' => return .{ s[pos + 1], pos + 2 },
                else => return self.fail("unknown escape at {s}", .{std.mem.span(s + pos)}),
            }
        } else if (s[pos] != 0) {
            return decodeUtf8At(s, pos);
        }
        return self.fail("unexpected end of input", .{});
    }

    /// `std::stoul`/`std::stoull` over a run of digits: out of range throws.
    fn parseU64(self: *Parser, digits: []const u8, comptime what: []const u8) ParseError!u64 {
        return std.fmt.parseInt(u64, digits, 10) catch self.fail(what ++ ": out of range", .{});
    }

    /// Ports `parse_token` (llama-grammar.cpp:186 @c1d0e7a00).
    ///
    /// `<[id]>` names a token by id; `<text>` tokenizes `<text>` with special
    /// tokens parsed and requires exactly one token.
    fn parseToken(self: *Parser, s: [*:0]const u8, start: usize) ParseError!struct { u32, usize } {
        var pos = start;
        if (s[pos] != '<') {
            return self.fail("expecting '<' at {s}", .{std.mem.span(s + pos)});
        }
        pos += 1;

        if (s[pos] == '[') {
            pos += 1;
            const int_end = try self.parseInt(s, pos);
            // `std::stoul` into a `uint32_t`: parsed as `unsigned long`, then
            // narrowed.
            const token_id: u32 = @truncate(try self.parseU64(s[pos..int_end], "stoul"));
            pos = int_end;
            if (s[pos] != ']') {
                return self.fail("expecting ']' at {s}", .{std.mem.span(s + pos)});
            }
            pos += 1;
            if (s[pos] != '>') {
                return self.fail("expecting '>' at {s}", .{std.mem.span(s + pos)});
            }
            pos += 1;
            return .{ token_id, pos };
        }

        if (self.vocab == null) {
            return self.fail("no vocab to parse token at {s}", .{std.mem.span(s + start)});
        }

        while (s[pos] != 0 and s[pos] != '>') pos += 1;
        if (s[pos] != '>') {
            return self.fail("expecting '>' at {s}", .{std.mem.span(s + pos)});
        }
        pos += 1;

        var tokens: [2]c.llama_token = undefined;
        const n_tokens = c.llama_tokenize(self.vocab, s + start, @intCast(pos - start), &tokens, 2, false, true);
        if (n_tokens != 1) {
            return self.fail("invalid token '{s}'", .{s[start..pos]});
        }
        return .{ @bitCast(tokens[0]), pos };
    }

    /// Ports `llama_grammar_parser`'s `parse_alternates` (llama-grammar.cpp:435 @c1d0e7a00).
    fn parseAlternates(self: *Parser, s: [*:0]const u8, start: usize, rule_name: []const u8, rule_id: u32, is_nested: bool) ParseError!usize {
        var rule: Rule = .empty;
        defer rule.deinit(impl.allocator);
        var pos = try self.parseSequence(s, start, rule_name, &rule, is_nested);
        while (s[pos] == '|') {
            rule.append(impl.allocator, .{ .type = .alt, .value = 0 }) catch oom();
            pos = parseSpace(s, pos + 1, true);
            pos = try self.parseSequence(s, pos, rule_name, &rule, is_nested);
        }
        rule.append(impl.allocator, .{ .type = .end, .value = 0 }) catch oom();
        self.addRule(rule_id, rule.items);
        return pos;
    }

    /// Ports the `handle_repetitions` lambda in
    /// `llama_grammar_parser::parse_sequence` (llama-grammar.cpp:463 @c1d0e7a00).
    ///
    /// Rewrites the last symbol into its `m` required copies and `n - m`
    /// nested optional rules, or one recursive rule for no maximum.
    fn handleRepetitions(
        self: *Parser,
        s: [*:0]const u8,
        pos: usize,
        rule_name: []const u8,
        rule: *Rule,
        last_sym_start: usize,
        n_prev_rules: *u64,
        min_times: u64,
        max_times: u64,
    ) ParseError!void {
        const no_max = max_times == std.math.maxInt(u64);
        if (last_sym_start == rule.items.len) {
            return self.fail("expecting preceding item to */+/?/{{ at {s}", .{std.mem.span(s + pos)});
        }

        var prev_rule: Rule = .empty;
        defer prev_rule.deinit(impl.allocator);
        prev_rule.appendSlice(impl.allocator, rule.items[last_sym_start..]) catch oom();

        var total_rules: u64 = 1;
        if (!no_max and max_times > 0) {
            total_rules = max_times;
        } else if (min_times > 0) {
            total_rules = min_times;
        }

        if (n_prev_rules.* *% total_rules >= MAX_REPETITION_THRESHOLD) {
            return self.fail("number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please reduce the number of repetitions or rule complexity", .{});
        }

        if (min_times == 0) {
            rule.shrinkRetainingCapacity(last_sym_start);
        } else {
            var i: u64 = 1;
            while (i < min_times) : (i += 1) {
                rule.appendSlice(impl.allocator, prev_rule.items) catch oom();
            }
        }

        var last_rec_rule_id: u32 = 0;
        const n_opt: u64 = if (no_max) 1 else max_times -% min_times;

        var rec_rule: Rule = .empty;
        defer rec_rule.deinit(impl.allocator);
        rec_rule.appendSlice(impl.allocator, prev_rule.items) catch oom();
        var i: u64 = 0;
        while (i < n_opt) : (i += 1) {
            rec_rule.shrinkRetainingCapacity(prev_rule.items.len);
            const rec_rule_id = self.generateSymbolId(rule_name);
            if (i > 0 or no_max) {
                rec_rule.append(impl.allocator, .{ .type = .rule_ref, .value = if (no_max) rec_rule_id else last_rec_rule_id }) catch oom();
            }
            rec_rule.append(impl.allocator, .{ .type = .alt, .value = 0 }) catch oom();
            rec_rule.append(impl.allocator, .{ .type = .end, .value = 0 }) catch oom();
            self.addRule(rec_rule_id, rec_rule.items);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) {
            rule.append(impl.allocator, .{ .type = .rule_ref, .value = last_rec_rule_id }) catch oom();
        }
        n_prev_rules.* *%= total_rules;
        impl.assert(n_prev_rules.* >= 1, src, 527, "n_prev_rules >= 1");
    }

    /// Ports `llama_grammar_parser`'s `parse_sequence` (llama-grammar.cpp:452 @c1d0e7a00).
    fn parseSequence(self: *Parser, s: [*:0]const u8, start: usize, rule_name: []const u8, rule: *Rule, is_nested: bool) ParseError!usize {
        var last_sym_start = rule.items.len;
        var pos = start;
        var n_prev_rules: u64 = 1;

        while (s[pos] != 0) {
            if (s[pos] == '"') { // literal string
                pos += 1;
                last_sym_start = rule.items.len;
                n_prev_rules = 1;
                while (s[pos] != '"') {
                    if (s[pos] == 0) {
                        return self.fail("unexpected end of input", .{});
                    }
                    const ch, const next = try self.parseChar(s, pos);
                    pos = next;
                    rule.append(impl.allocator, .{ .type = .char, .value = ch }) catch oom();
                }
                pos = parseSpace(s, pos + 1, is_nested);
            } else if (s[pos] == '[') { // char range(s)
                pos += 1;
                var start_type: GreType = .char;
                if (s[pos] == '^') {
                    pos += 1;
                    start_type = .char_not;
                }
                last_sym_start = rule.items.len;
                n_prev_rules = 1;
                while (s[pos] != ']') {
                    if (s[pos] == 0) {
                        return self.fail("unexpected end of input", .{});
                    }
                    const ch, const next = try self.parseChar(s, pos);
                    pos = next;
                    const ty: GreType = if (last_sym_start < rule.items.len) .char_alt else start_type;

                    rule.append(impl.allocator, .{ .type = ty, .value = ch }) catch oom();
                    if (s[pos] == '-' and s[pos + 1] != ']') {
                        if (s[pos + 1] == 0) {
                            return self.fail("unexpected end of input", .{});
                        }
                        const endch, const endnext = try self.parseChar(s, pos + 1);
                        pos = endnext;
                        rule.append(impl.allocator, .{ .type = .char_rng_upper, .value = endch }) catch oom();
                    }
                }
                pos = parseSpace(s, pos + 1, is_nested);
            } else if (s[pos] == '<' or s[pos] == '!') { // token
                var ty: GreType = .token;
                if (s[pos] == '!') { // token inverse
                    ty = .token_not;
                    pos += 1;
                }
                const token_id, const token_end = try self.parseToken(s, pos);
                last_sym_start = rule.items.len;
                n_prev_rules = 1;
                rule.append(impl.allocator, .{ .type = ty, .value = token_id }) catch oom();
                pos = parseSpace(s, token_end, is_nested);
            } else if (isWordChar(s[pos])) { // rule reference
                const name_end = try self.parseName(s, pos);
                const ref_rule_id = self.getSymbolId(s[pos..name_end]);
                pos = parseSpace(s, name_end, is_nested);
                last_sym_start = rule.items.len;
                n_prev_rules = 1;
                rule.append(impl.allocator, .{ .type = .rule_ref, .value = ref_rule_id }) catch oom();
            } else if (s[pos] == '(') { // grouping
                pos = parseSpace(s, pos + 1, true);
                const n_rules_before: u32 = @intCast(self.symbol_ids.count());
                const sub_rule_id = self.generateSymbolId(rule_name);
                pos = try self.parseAlternates(s, pos, rule_name, sub_rule_id, true);
                // `std::max(1u, (uint32_t)symbol_ids.size() - n_rules_before)`,
                // unsigned.
                n_prev_rules = @max(1, @as(u32, @intCast(self.symbol_ids.count())) -% n_rules_before);
                last_sym_start = rule.items.len;
                rule.append(impl.allocator, .{ .type = .rule_ref, .value = sub_rule_id }) catch oom();
                if (s[pos] != ')') {
                    return self.fail("expecting ')' at {s}", .{std.mem.span(s + pos)});
                }
                pos = parseSpace(s, pos + 1, is_nested);
            } else if (s[pos] == '.') { // any char
                last_sym_start = rule.items.len;
                n_prev_rules = 1;
                rule.append(impl.allocator, .{ .type = .char_any, .value = 0 }) catch oom();
                pos = parseSpace(s, pos + 1, is_nested);
            } else if (s[pos] == '*') {
                pos = parseSpace(s, pos + 1, is_nested);
                try self.handleRepetitions(s, pos, rule_name, rule, last_sym_start, &n_prev_rules, 0, std.math.maxInt(u64));
            } else if (s[pos] == '+') {
                pos = parseSpace(s, pos + 1, is_nested);
                try self.handleRepetitions(s, pos, rule_name, rule, last_sym_start, &n_prev_rules, 1, std.math.maxInt(u64));
            } else if (s[pos] == '?') {
                pos = parseSpace(s, pos + 1, is_nested);
                try self.handleRepetitions(s, pos, rule_name, rule, last_sym_start, &n_prev_rules, 0, 1);
            } else if (s[pos] == '{') {
                pos = parseSpace(s, pos + 1, is_nested);

                if (!isDigitChar(s[pos])) {
                    return self.fail("expecting an int at {s}", .{std.mem.span(s + pos)});
                }
                const int_end = try self.parseInt(s, pos);
                const min_times = try self.parseU64(s[pos..int_end], "stoull");
                pos = parseSpace(s, int_end, is_nested);

                var max_times: u64 = std.math.maxInt(u64);

                if (s[pos] == '}') {
                    max_times = min_times;
                    pos = parseSpace(s, pos + 1, is_nested);
                } else if (s[pos] == ',') {
                    pos = parseSpace(s, pos + 1, is_nested);

                    if (isDigitChar(s[pos])) {
                        const int_end2 = try self.parseInt(s, pos);
                        max_times = try self.parseU64(s[pos..int_end2], "stoull");
                        pos = parseSpace(s, int_end2, is_nested);
                    }

                    if (s[pos] != '}') {
                        return self.fail("expecting '}}' at {s}", .{std.mem.span(s + pos)});
                    }
                    pos = parseSpace(s, pos + 1, is_nested);
                } else {
                    return self.fail("expecting ',' at {s}", .{std.mem.span(s + pos)});
                }
                if (min_times > MAX_REPETITION_THRESHOLD) {
                    return self.fail("number of repetitions exceeds sane defaults, please reduce the number of repetitions", .{});
                }
                if (max_times != std.math.maxInt(u64) and max_times > MAX_REPETITION_THRESHOLD) {
                    max_times = std.math.maxInt(u64);
                }
                try self.handleRepetitions(s, pos, rule_name, rule, last_sym_start, &n_prev_rules, min_times, max_times);
            } else {
                break;
            }
        }
        return pos;
    }

    /// Ports `llama_grammar_parser`'s `parse_rule` (llama-grammar.cpp:666 @c1d0e7a00).
    fn parseRule(self: *Parser, s: [*:0]const u8, start: usize) ParseError!usize {
        const name_end = try self.parseName(s, start);
        var pos = parseSpace(s, name_end, false);
        const name = s[start..name_end];
        const rule_id = self.getSymbolId(name);

        if (!(s[pos] == ':' and s[pos + 1] == ':' and s[pos + 2] == '=')) {
            return self.fail("expecting ::= at {s}", .{std.mem.span(s + pos)});
        }
        pos = parseSpace(s, pos + 3, true);

        pos = try self.parseAlternates(s, pos, name, rule_id, false);

        if (s[pos] == '\r') {
            pos += if (s[pos + 1] == '\n') 2 else 1;
        } else if (s[pos] == '\n') {
            pos += 1;
        } else if (s[pos] != 0) {
            return self.fail("expecting newline or end at {s}", .{std.mem.span(s + pos)});
        }
        return parseSpace(s, pos, true);
    }

    fn parseAll(self: *Parser, s: [*:0]const u8) ParseError!void {
        var pos = parseSpace(s, 0, true);
        while (s[pos] != 0) {
            pos = try self.parseRule(s, pos);
        }
        // Every referenced rule must be defined.
        for (self.rules.items) |rule| {
            if (rule.items.len == 0) {
                return self.fail("Undefined rule", .{});
            }
            for (rule.items) |elem| {
                if (elem.type == .rule_ref) {
                    if (elem.value >= self.rules.items.len or self.rules.items[elem.value].items.len == 0) {
                        var it = self.symbol_ids.iterator();
                        while (it.next()) |kv| {
                            if (kv.value_ptr.* == elem.value) {
                                return self.fail("Undefined rule identifier '{s}'", .{kv.key_ptr.*});
                            }
                        }
                    }
                }
            }
        }
    }

    /// Ports `llama_grammar_parser`'s `parse` (llama-grammar.cpp:690 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `s`: the grammar text.
    ///
    /// Return: true on success; on failure, prints the error and the grammar
    /// to stderr, clears the rules, and returns false.
    pub fn parse(self: *Parser, s: [*:0]const u8) bool {
        self.parseAll(s) catch {
            std.debug.print("{s}: error parsing grammar: {s}\n\n{s}\n", .{ "parse", self.err.items, std.mem.span(s) });
            self.clearRules();
            return false;
        };
        return true;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a two-alternative grammar parses into one rule" {
    var p = Parser.init(null);
    defer p.deinit();
    try std.testing.expect(p.parse("root ::= \"a\" | \"b\""));
    try std.testing.expectEqual(@as(usize, 1), p.rules.items.len);
    const r = p.rules.items[0].items;
    try std.testing.expectEqual(@as(usize, 4), r.len);
    try std.testing.expectEqual(GreType.char, r[0].type);
    try std.testing.expectEqual(@as(u32, 'a'), r[0].value);
    try std.testing.expectEqual(GreType.alt, r[1].type);
    try std.testing.expectEqual(GreType.end, r[3].type);
}

test "repetition generates numbered helper rules" {
    var p = Parser.init(null);
    defer p.deinit();
    try std.testing.expect(p.parse("root ::= \"x\"{1,3}"));
    // root plus two optional tails, root_1 and root_2.
    try std.testing.expectEqual(@as(usize, 3), p.rules.items.len);
    try std.testing.expect(p.symbol_ids.contains("root_1"));
    try std.testing.expect(p.symbol_ids.contains("root_2"));
}

test "an undefined rule fails and clears the rules" {
    var p = Parser.init(null);
    defer p.deinit();
    try std.testing.expect(!p.parse("root ::= missing"));
    try std.testing.expectEqual(@as(usize, 0), p.rules.items.len);
}
