//! The grammar runtime: pushdown stacks over parsed rules, constraining
//! which tokens the sampler may choose.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-grammar.cpp` and `llama-grammar.h`
//! (v0.4.1, `b29c606e2`). Each declaration below names the C++ it replaces
//! and the line it began at.
//!
//! # Contract
//!
//! `llama-grammar.cpp` exports nothing with C linkage; its only caller is
//! the sampler, through the five `llama_grammar_*_impl` functions. Those
//! and the types they share are what this file provides, to
//! `sampler/grammar.zig`. The rest of `llama-grammar.h` -- `accept_str`,
//! `get_rules`, the rule-array `init_impl` overload, `print` -- serves
//! upstream's tests, which this project does not build.
//!
//! # Pointers into rules
//!
//! As in the C++, a stack is a list of pointers to elements inside the rules
//! it walks. The rules are allocated once per grammar and never resized, so
//! the pointers stay valid; `clone` remaps them into the copy's rules exactly
//! as `llama_grammar_clone_impl` does.
//!
//! # Order
//!
//! `llama_grammar_advance_stack` dedups through an `std::set` ordered by
//! pointer address. Only membership is read, so a hash set stands in; the
//! **order stacks are produced in** -- a LIFO work list -- is kept, because
//! it is the order every later step walks them in.
//!
//! # Termination where the C++ throws
//!
//! An empty stack set after accepting a token throws `std::runtime_error`
//! out through `llama_sampler_accept`, a C entry point, and terminates. Here
//! it terminates through `impl.throwRuntime` with the same text.

const std = @import("std");
const impl = @import("../impl.zig");
const parser = @import("parser.zig");
const regex = @import("../regex.zig");

const c = impl.c;
const src = "llama-grammar.cpp";

pub const Element = parser.Element;
pub const GreType = parser.GreType;
const ElemPtr = [*]const Element;
const Token = c.llama_token;

/// Ports `llama_grammar_stack` (llama-grammar.h:65 @b29c606e2).
pub const Stack = std.ArrayList(ElemPtr);
/// Ports `llama_grammar_stacks` (llama-grammar.h:68 @b29c606e2).
pub const Stacks = std.ArrayList(Stack);

/// Ports `llama_partial_utf8` (llama-grammar.h:52 @b29c606e2).
pub const PartialUtf8 = struct {
    value: u32 = 0,
    /// Bytes still to come; -1 marks an invalid sequence.
    n_remain: c_int = 0,
};

/// Ports `llama_grammar_candidate` (llama-grammar.h:57 @b29c606e2).
const Candidate = struct {
    index: usize,
    code_points: [*]const u32,
    partial_utf8: PartialUtf8,
    id: Token,
};

fn oom() noreturn {
    impl.abort(src, 0, "out of memory");
}

fn freeStacks(stacks: *Stacks) void {
    for (stacks.items) |*s| s.deinit(impl.allocator);
    stacks.deinit(impl.allocator);
}

fn copyStack(s: []const ElemPtr) Stack {
    var out: Stack = .empty;
    out.appendSlice(impl.allocator, s) catch oom();
    return out;
}

/// `std::find(stacks.begin(), stacks.end(), stack) != stacks.end()`.
fn containsStack(stacks: []const Stack, s: []const ElemPtr) bool {
    for (stacks) |t| {
        if (std.mem.eql(ElemPtr, t.items, s)) return true;
    }
    return false;
}

/// Ports `llama_grammar_trigger_pattern` (llama-grammar.h:119 @b29c606e2).
pub const TriggerPattern = struct {
    pattern: []u8,
    re: regex.Regex,

    /// Ports `llama_grammar_trigger_pattern`'s `find` (llama-grammar.cpp:378 @b29c606e2).
    ///
    /// A pattern that starts with `^` and ends with `$` is first tried as a
    /// whole-input match; otherwise, or when that fails, it is searched for.
    ///
    /// Parameters:
    /// - `input`: the buffered text.
    ///
    /// Return: where the grammar should start reading -- the first non-empty
    /// capture group, else the whole match -- or null for no match.
    pub fn find(self: *const TriggerPattern, input: []const u8) ?usize {
        if (self.pattern.len != 0 and self.pattern[0] == '^' and self.pattern[self.pattern.len - 1] == '$') {
            var m = regex.Match.init();
            defer m.deinit();
            if (self.re.match(input, &m)) {
                return findStartPos(&m);
            }
        }

        var m = regex.Match.init();
        defer m.deinit();
        if (self.re.search(input, &m)) {
            return findStartPos(&m);
        }

        return null;
    }

    fn findStartPos(m: *const regex.Match) usize {
        var i: usize = 1;
        while (i < m.size()) : (i += 1) {
            if (m.length(i) > 0) return m.position(i);
        }
        return m.position(0);
    }

    fn deinit(self: *TriggerPattern) void {
        impl.allocator.free(self.pattern);
        self.re.deinit();
    }

    fn clone(self: *const TriggerPattern) TriggerPattern {
        return .{
            .pattern = impl.allocator.dupe(u8, self.pattern) catch oom(),
            .re = self.re.clone(),
        };
    }
};

/// Ports `llama_grammar`'s `token_pos` (llama-grammar.h:128 @b29c606e2).
const TokenPos = struct {
    token: Token,
    start: usize,
    end: usize,
};

/// Ports `llama_grammar` (llama-grammar.h:126 @b29c606e2).
pub const Grammar = struct {
    vocab: ?*const c.struct_llama_vocab,

    /// Owned; never resized, so the stacks' pointers into it stay valid.
    rules: [][]Element,
    stacks: Stacks,

    /// The tail of a UTF-8 sequence the last token left unfinished.
    partial_utf8: PartialUtf8,

    lazy: bool = false,
    awaiting_trigger: bool = false,
    trigger_buffer: std.ArrayList(u8) = .empty,
    trigger_buffer_positions: std.ArrayList(TokenPos) = .empty,
    trigger_tokens: std.ArrayList(Token) = .empty,
    trigger_patterns: std.ArrayList(TriggerPattern) = .empty,
};

fn freeRules(rules: [][]Element) void {
    for (rules) |r| impl.allocator.free(r);
    impl.allocator.free(rules);
}

/// Ports the `std::string` overload of `decode_utf8` (llama-grammar.cpp:34 @b29c606e2).
///
/// Decodes `src_bytes` -- up to its first NUL, as the C++ reads it through
/// `c_str()` -- continuing any sequence `partial_start` left open.
///
/// Parameters:
/// - `src_bytes`: the text.
/// - `partial_start`: the unfinished sequence carried in.
/// - `out`: receives the code points and a terminating 0.
///
/// Return: the sequence left unfinished at the end.
fn decodeUtf8(src_bytes: []const u8, partial_start: PartialUtf8, out: *std.ArrayList(u32)) PartialUtf8 {
    const lookup = [_]c_int{ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    out.clearRetainingCapacity();
    out.ensureTotalCapacity(impl.allocator, src_bytes.len + 1) catch oom();

    var pos: usize = 0;
    const at = struct {
        fn f(b: []const u8, p: usize) u8 {
            return if (p < b.len) b[p] else 0;
        }
    }.f;

    var value: u32 = partial_start.value;
    var n_remain: c_int = partial_start.n_remain;

    while (at(src_bytes, pos) != 0 and n_remain > 0) {
        const next_byte = at(src_bytes, pos);
        if ((next_byte >> 6) != 2) {
            out.append(impl.allocator, 0) catch oom();
            return .{ .value = 0, .n_remain = -1 };
        }
        value = (value << 6) + (next_byte & 0x3F);
        pos += 1;
        n_remain -= 1;
    }

    if (partial_start.n_remain > 0 and n_remain == 0) {
        out.append(impl.allocator, value) catch oom();
    }

    while (at(src_bytes, pos) != 0) {
        const first_byte = at(src_bytes, pos);
        const highbits = first_byte >> 4;
        n_remain = lookup[highbits] - 1;

        if (n_remain < 0) {
            out.clearRetainingCapacity();
            out.append(impl.allocator, 0) catch oom();
            return .{ .value = 0, .n_remain = n_remain };
        }

        const mask: u8 = @truncate((@as(u32, 1) << @intCast(7 - n_remain)) - 1);
        value = first_byte & mask;

        pos += 1;
        while (at(src_bytes, pos) != 0 and n_remain > 0) {
            value = (value << 6) + (at(src_bytes, pos) & 0x3F);
            pos += 1;
            n_remain -= 1;
        }
        if (n_remain == 0) {
            out.append(impl.allocator, value) catch oom();
        }
    }
    out.append(impl.allocator, 0) catch oom();

    return .{ .value = value, .n_remain = n_remain };
}

/// Ports `llama_grammar_is_end_of_sequence` (llama-grammar.cpp:751 @b29c606e2).
fn isEndOfSequence(pos: ElemPtr) bool {
    return switch (pos[0].type) {
        .end, .alt => true,
        else => false,
    };
}

/// Ports `llama_grammar_match_char` (llama-grammar.cpp:761 @b29c606e2).
///
/// Return: whether `chr` satisfies the char range at `pos_in`, and the
/// element after the range.
fn matchChar(pos_in: ElemPtr, chr: u32) struct { bool, ElemPtr } {
    var pos = pos_in;
    var found = false;
    const is_positive_char = pos[0].type == .char or pos[0].type == .char_any;

    impl.assert(is_positive_char or pos[0].type == .char_not, src, 767, "is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT");

    while (true) {
        if (pos[1].type == .char_rng_upper) {
            found = found or (pos[0].value <= chr and chr <= pos[1].value);
            pos += 2;
        } else if (pos[0].type == .char_any) {
            found = true;
            pos += 1;
        } else {
            found = found or pos[0].value == chr;
            pos += 1;
        }
        if (pos[0].type != .char_alt) break;
    }

    return .{ found == is_positive_char, pos };
}

/// Ports `llama_grammar_match_partial_char` (llama-grammar.cpp:791 @b29c606e2).
///
/// Return: whether some completion of `partial_utf8` could satisfy the range.
fn matchPartialChar(pos_in: ElemPtr, partial_utf8: PartialUtf8) bool {
    var pos = pos_in;
    const is_positive_char = pos[0].type == .char or pos[0].type == .char_any;
    impl.assert(is_positive_char or pos[0].type == .char_not, src, 795, "is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT");

    const partial_value = partial_utf8.value;
    const n_remain = partial_utf8.n_remain;

    if (n_remain < 0 or (n_remain == 1 and partial_value < 2)) return false;

    const shift: u5 = @intCast(n_remain * 6);
    var low: u32 = partial_value << shift;
    const high: u32 = low | ((@as(u32, 1) << shift) - 1);

    if (low == 0) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }

    while (true) {
        if (pos[1].type == .char_rng_upper) {
            if (pos[0].value <= high and low <= pos[1].value) return is_positive_char;
            pos += 2;
        } else if (pos[0].type == .char_any) {
            return true;
        } else {
            if (low <= pos[0].value and pos[0].value <= high) return is_positive_char;
            pos += 1;
        }
        if (pos[0].type != .char_alt) break;
    }

    return !is_positive_char;
}

/// Ports `llama_grammar_match_token` (llama-grammar.cpp:841 @b29c606e2).
fn matchToken(pos: ElemPtr, token: Token) bool {
    impl.assert(pos[0].type == .token or pos[0].type == .token_not, src, 844, "pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT");
    if (pos[0].type == .token) return pos[0].value == @as(u32, @bitCast(token));
    if (pos[0].type == .token_not) return pos[0].value != @as(u32, @bitCast(token));
    return false;
}

/// Hashes a stack by its pointers, for `advanceStack`'s `seen` set.
const StackSetCtx = struct {
    pub fn hash(_: StackSetCtx, k: []const ElemPtr) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(k));
    }
    pub fn eql(_: StackSetCtx, a: []const ElemPtr, b: []const ElemPtr) bool {
        return std.mem.eql(ElemPtr, a, b);
    }
};

/// Ports `llama_grammar_advance_stack` (llama-grammar.cpp:856 @b29c606e2).
///
/// Expands `stack` through rule references until every resulting stack is
/// empty or has a terminal on top, appending the new ones to `new_stacks`.
fn advanceStack(rules: []const []Element, stack: []const ElemPtr, new_stacks: *Stacks) void {
    var todo: Stacks = .empty;
    defer freeStacks(&todo);
    todo.append(impl.allocator, copyStack(stack)) catch oom();

    var seen: std.HashMapUnmanaged([]const ElemPtr, void, StackSetCtx, 80) = .empty;
    defer {
        var it = seen.keyIterator();
        while (it.next()) |k| impl.allocator.free(k.*);
        seen.deinit(impl.allocator);
    }

    while (todo.items.len != 0) {
        var curr_stack = todo.pop().?;

        if (seen.contains(curr_stack.items)) {
            curr_stack.deinit(impl.allocator);
            continue;
        }
        const key = impl.allocator.dupe(ElemPtr, curr_stack.items) catch oom();
        seen.put(impl.allocator, key, {}) catch oom();

        if (curr_stack.items.len == 0) {
            if (!containsStack(new_stacks.items, curr_stack.items)) {
                new_stacks.append(impl.allocator, curr_stack) catch oom();
            } else {
                curr_stack.deinit(impl.allocator);
            }
            continue;
        }

        const pos = curr_stack.items[curr_stack.items.len - 1];

        switch (pos[0].type) {
            .rule_ref => {
                const rule_id: usize = pos[0].value;
                var subpos: ElemPtr = rules[rule_id].ptr;
                while (true) {
                    // The stack without its top...
                    var next_stack = copyStack(curr_stack.items[0 .. curr_stack.items.len - 1]);
                    if (!isEndOfSequence(pos + 1)) {
                        // ...then what follows the reference, if anything...
                        next_stack.append(impl.allocator, pos + 1) catch oom();
                    }
                    if (!isEndOfSequence(subpos)) {
                        // ...then this alternative, if it is non-empty.
                        next_stack.append(impl.allocator, subpos) catch oom();
                    }
                    todo.append(impl.allocator, next_stack) catch oom();
                    while (!isEndOfSequence(subpos)) subpos += 1;
                    if (subpos[0].type == .alt) {
                        subpos += 1;
                    } else {
                        break;
                    }
                }
                curr_stack.deinit(impl.allocator);
            },
            .char, .char_not, .char_any, .token, .token_not => {
                if (!containsStack(new_stacks.items, curr_stack.items)) {
                    new_stacks.append(impl.allocator, curr_stack) catch oom();
                } else {
                    curr_stack.deinit(impl.allocator);
                }
            },
            else => impl.abort(src, 935, "fatal error"),
        }
    }
}

/// Ports `llama_grammar_reject_candidates` (llama-grammar.cpp:940 @b29c606e2).
///
/// Return: the candidates no stack accepts; owned by the caller.
fn rejectCandidates(rules: []const []Element, stacks: []const Stack, candidates: []const Candidate) std.ArrayList(Candidate) {
    impl.assert(stacks.len != 0, src, 943, "!stacks.empty()");

    if (candidates.len == 0) return .empty;

    var rejects = rejectCandidatesForStack(rules, stacks[0].items, candidates);

    for (stacks[1..]) |s| {
        const next = rejectCandidatesForStack(rules, s.items, rejects.items);
        rejects.deinit(impl.allocator);
        rejects = next;
    }

    return rejects;
}

/// Ports `llama_grammar_detect_left_recursion` (llama-grammar.cpp:959 @b29c606e2).
fn detectLeftRecursion(rules: []const []Element, rule_index: usize, rules_visited: []bool, rules_in_progress: []bool, rules_may_be_empty: []bool) bool {
    if (rules_in_progress[rule_index]) return true;

    rules_in_progress[rule_index] = true;

    const rule = rules[rule_index];

    // Can the rule produce the empty string?
    var at_rule_start = true;
    for (0..rule.len) |i| {
        if (isEndOfSequence(rule[i..].ptr)) {
            if (at_rule_start) {
                rules_may_be_empty[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Recurse into the leftmost non-terminals, and the next one while the
    // previous may be empty.
    var recurse_into_nonterminal = true;
    for (0..rule.len) |i| {
        if (rule[i].type == .rule_ref and recurse_into_nonterminal) {
            if (detectLeftRecursion(rules, rule[i].value, rules_visited, rules_in_progress, rules_may_be_empty)) {
                return true;
            }
            if (!rules_may_be_empty[rule[i].value]) {
                recurse_into_nonterminal = false;
            }
        } else if (isEndOfSequence(rule[i..].ptr)) {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    rules_in_progress[rule_index] = false;
    rules_visited[rule_index] = true;

    return false;
}

/// Ports `llama_grammar_accept_chr` (llama-grammar.cpp:1020 @b29c606e2).
fn acceptChr(g: *Grammar, stack: []const ElemPtr, chr: u32, new_stacks: *Stacks) void {
    if (stack.len == 0) return;

    const pos = stack[stack.len - 1];

    // A token rule is not advanced by characters.
    if (pos[0].type == .token or pos[0].type == .token_not) return;

    const matched, const next = matchChar(pos, chr);
    if (matched) {
        var new_stack = copyStack(stack[0 .. stack.len - 1]);
        defer new_stack.deinit(impl.allocator);
        if (!isEndOfSequence(next)) {
            new_stack.append(impl.allocator, next) catch oom();
        }
        advanceStack(g.rules, new_stack.items, new_stacks);
    }
}

/// Ports `llama_grammar_reject_candidates_for_stack` (llama-grammar.cpp:1057 @b29c606e2).
///
/// Return: the candidates this stack rejects; owned by the caller.
fn rejectCandidatesForStack(rules: []const []Element, stack: []const ElemPtr, candidates: []const Candidate) std.ArrayList(Candidate) {
    var rejects: std.ArrayList(Candidate) = .empty;
    rejects.ensureTotalCapacity(impl.allocator, candidates.len) catch oom();

    if (stack.len == 0) {
        for (candidates) |tok| {
            if (tok.code_points[0] != 0 or tok.partial_utf8.n_remain != 0) {
                rejects.appendAssumeCapacity(tok);
            }
        }
        return rejects;
    }

    const stack_pos = stack[stack.len - 1];

    // A token rule on top: only the token id matters.
    if (stack_pos[0].type == .token or stack_pos[0].type == .token_not) {
        for (candidates) |tok| {
            if (tok.code_points[0] == 0) {
                // A token consumed by char rules ends here; reject only if it
                // ended mid-sequence.
                if (tok.partial_utf8.n_remain != 0) rejects.appendAssumeCapacity(tok);
            } else if (!matchToken(stack_pos, tok.id)) {
                rejects.appendAssumeCapacity(tok);
            }
        }
        return rejects;
    }

    var next_candidates: std.ArrayList(Candidate) = .empty;
    defer next_candidates.deinit(impl.allocator);
    next_candidates.ensureTotalCapacity(impl.allocator, candidates.len) catch oom();

    for (candidates) |tok| {
        if (tok.code_points[0] == 0) {
            if (tok.partial_utf8.n_remain != 0 and !matchPartialChar(stack_pos, tok.partial_utf8)) {
                rejects.appendAssumeCapacity(tok);
            }
        } else if (matchChar(stack_pos, tok.code_points[0])[0]) {
            next_candidates.appendAssumeCapacity(.{ .index = tok.index, .code_points = tok.code_points + 1, .partial_utf8 = tok.partial_utf8, .id = tok.id });
        } else {
            rejects.appendAssumeCapacity(tok);
        }
    }

    const stack_pos_after = matchChar(stack_pos, 0)[1];

    var stack_after = copyStack(stack[0 .. stack.len - 1]);
    defer stack_after.deinit(impl.allocator);
    if (!isEndOfSequence(stack_pos_after)) {
        stack_after.append(impl.allocator, stack_pos_after) catch oom();
    }
    var next_stacks: Stacks = .empty;
    defer freeStacks(&next_stacks);
    advanceStack(rules, stack_after.items, &next_stacks);

    var next_rejects = rejectCandidates(rules, next_stacks.items, next_candidates.items);
    defer next_rejects.deinit(impl.allocator);
    for (next_rejects.items) |tok| {
        rejects.append(impl.allocator, .{ .index = tok.index, .code_points = tok.code_points - 1, .partial_utf8 = tok.partial_utf8, .id = tok.id }) catch oom();
    }

    return rejects;
}

/// Ports the grammar-text overload of `llama_grammar_init_impl`
/// (llama-grammar.cpp:1211 @b29c606e2).
///
/// Parameters:
/// - `vocab`: the vocabulary; borrowed for the grammar's lifetime.
/// - `grammar_str`: the GBNF text.
/// - `grammar_root`: the start symbol.
/// - `lazy`: wait for a trigger before constraining.
/// - `trigger_patterns`: regexes, compiled with ECMAScript semantics; an
///   invalid one terminates, as the C++'s `std::regex` constructor throws.
/// - `trigger_tokens`: tokens that trigger immediately.
///
/// Return: the grammar, or null after logging why it could not be built.
pub fn initImpl(
    vocab: ?*const c.struct_llama_vocab,
    grammar_str: [*:0]const u8,
    grammar_root: [*:0]const u8,
    lazy: bool,
    trigger_patterns: []const [*:0]const u8,
    trigger_tokens: []const Token,
) ?*Grammar {
    var p = parser.Parser.init(vocab);
    defer p.deinit();

    if (!p.parse(grammar_str) or p.rules.items.len == 0) {
        impl.logError("failed to parse grammar\n", .{});
        return null;
    }

    const start_rule_index: usize = p.symbol_ids.get(std.mem.span(grammar_root)) orelse {
        impl.logError("grammar does not contain a '%s' symbol\n", .{grammar_root});
        return null;
    };

    const n_rules = p.rules.items.len;

    // Copy each rule up to its END, then END.
    const vec_rules = impl.allocator.alloc([]Element, n_rules) catch oom();
    for (p.rules.items, 0..) |r, i| {
        var n: usize = 0;
        while (r.items[n].type != .end) n += 1;
        const copy = impl.allocator.alloc(Element, n + 1) catch oom();
        @memcpy(copy[0..n], r.items[0..n]);
        copy[n] = .{ .type = .end, .value = 0 };
        vec_rules[i] = copy;
    }

    // Left recursion would make `advanceStack` loop.
    {
        const rules_visited = impl.allocator.alloc(bool, n_rules) catch oom();
        defer impl.allocator.free(rules_visited);
        const rules_in_progress = impl.allocator.alloc(bool, n_rules) catch oom();
        defer impl.allocator.free(rules_in_progress);
        const rules_may_be_empty = impl.allocator.alloc(bool, n_rules) catch oom();
        defer impl.allocator.free(rules_may_be_empty);
        @memset(rules_visited, false);
        @memset(rules_in_progress, false);
        @memset(rules_may_be_empty, false);
        for (0..n_rules) |i| {
            if (rules_visited[i]) continue;
            if (detectLeftRecursion(vec_rules, i, rules_visited, rules_in_progress, rules_may_be_empty)) {
                impl.logError("unsupported grammar, left recursion detected for nonterminal at index %zu\n", .{i});
                freeRules(vec_rules);
                return null;
            }
        }
    }

    // One initial stack per alternative of the start rule.
    var stacks: Stacks = .empty;
    var pos: ElemPtr = vec_rules[start_rule_index].ptr;
    while (true) {
        var stack: Stack = .empty;
        defer stack.deinit(impl.allocator);
        if (!isEndOfSequence(pos)) {
            stack.append(impl.allocator, pos) catch oom();
        }
        advanceStack(vec_rules, stack.items, &stacks);
        while (!isEndOfSequence(pos)) pos += 1;
        if (pos[0].type == .alt) {
            pos += 1;
        } else {
            break;
        }
    }

    const g = impl.allocator.create(Grammar) catch oom();
    g.* = .{
        .vocab = vocab,
        .rules = vec_rules,
        .stacks = stacks,
        .partial_utf8 = .{},
        .lazy = lazy,
        .awaiting_trigger = lazy,
    };
    g.trigger_tokens.appendSlice(impl.allocator, trigger_tokens) catch oom();
    for (trigger_patterns) |pat| {
        const text = std.mem.span(pat);
        const re = regex.Regex.compile(text) catch |err| regex.throwRegexError(err);
        g.trigger_patterns.append(impl.allocator, .{
            .pattern = impl.allocator.dupe(u8, text) catch oom(),
            .re = re,
        }) catch oom();
    }
    return g;
}

/// Ports `llama_grammar_free_impl` (llama-grammar.cpp:1317 @b29c606e2).
///
/// Parameters:
/// - `g`: the grammar, or null.
///
/// Return: nothing.
pub fn freeImpl(g: ?*Grammar) void {
    const grammar = g orelse return;
    freeRules(grammar.rules);
    freeStacks(&grammar.stacks);
    grammar.trigger_buffer.deinit(impl.allocator);
    grammar.trigger_buffer_positions.deinit(impl.allocator);
    grammar.trigger_tokens.deinit(impl.allocator);
    for (grammar.trigger_patterns.items) |*tp| tp.deinit();
    grammar.trigger_patterns.deinit(impl.allocator);
    impl.allocator.destroy(grammar);
}

/// Ports `llama_grammar_clone_impl` (llama-grammar.cpp:1325 @b29c606e2).
///
/// Copies everything, then repoints each stack element from the source's
/// rules to the copy's.
///
/// Parameters:
/// - `grammar`: the grammar to copy.
///
/// Return: the copy, freed with `freeImpl`.
pub fn cloneImpl(grammar: *const Grammar) *Grammar {
    const rules = impl.allocator.alloc([]Element, grammar.rules.len) catch oom();
    for (grammar.rules, 0..) |r, i| rules[i] = impl.allocator.dupe(Element, r) catch oom();

    var stacks: Stacks = .empty;
    stacks.ensureTotalCapacity(impl.allocator, grammar.stacks.items.len) catch oom();
    for (grammar.stacks.items) |s| {
        const ns = copyStack(s.items);
        for (ns.items) |*e| {
            for (grammar.rules, 0..) |r, ir0| {
                const base = @intFromPtr(r.ptr);
                const addr = @intFromPtr(e.*);
                if (addr >= base and addr < base + r.len * @sizeOf(Element)) {
                    e.* = rules[ir0].ptr + (addr - base) / @sizeOf(Element);
                }
            }
        }
        stacks.appendAssumeCapacity(ns);
    }

    const result = impl.allocator.create(Grammar) catch oom();
    result.* = .{
        .vocab = grammar.vocab,
        .rules = rules,
        .stacks = stacks,
        .partial_utf8 = grammar.partial_utf8,
        .lazy = grammar.lazy,
        .awaiting_trigger = grammar.awaiting_trigger,
    };
    result.trigger_buffer.appendSlice(impl.allocator, grammar.trigger_buffer.items) catch oom();
    result.trigger_buffer_positions.appendSlice(impl.allocator, grammar.trigger_buffer_positions.items) catch oom();
    result.trigger_tokens.appendSlice(impl.allocator, grammar.trigger_tokens.items) catch oom();
    for (grammar.trigger_patterns.items) |*tp| {
        result.trigger_patterns.append(impl.allocator, tp.clone()) catch oom();
    }
    return result;
}

/// `vocab->token_to_piece(id)`: the cached piece, which the C API returns
/// with `lstrip = 0, special = true`.
fn tokenToPiece(vocab: ?*const c.struct_llama_vocab, token: Token, buf: *std.ArrayList(u8)) []const u8 {
    if (buf.items.len < 64) buf.resize(impl.allocator, 64) catch oom();
    var n = c.llama_token_to_piece(vocab, token, buf.items.ptr, @intCast(buf.items.len), 0, true);
    if (n < 0) {
        buf.resize(impl.allocator, @intCast(-n)) catch oom();
        n = c.llama_token_to_piece(vocab, token, buf.items.ptr, @intCast(buf.items.len), 0, true);
    }
    return buf.items[0..@intCast(n)];
}

/// Ports `llama_grammar_apply_impl` (llama-grammar.cpp:1355 @b29c606e2).
///
/// Sets every candidate the grammar cannot continue with to -inf. EOG is
/// allowed only when some stack is complete.
///
/// Parameters:
/// - `grammar`: the grammar.
/// - `cur_p`: the candidates.
///
/// Return: nothing.
pub fn applyImpl(grammar: *const Grammar, cur_p: *c.llama_token_data_array) void {
    impl.assert(grammar.vocab != null, src, 1355, "grammar.vocab != nullptr");

    if (grammar.awaiting_trigger) return;

    var allow_eog = false;
    for (grammar.stacks.items) |s| {
        if (s.items.len == 0) {
            allow_eog = true;
            break;
        }
    }

    // Each decoded piece is its own allocation, so the candidates' pointers
    // into them stay valid as more are added.
    var candidates_decoded: std.ArrayList(std.ArrayList(u32)) = .empty;
    defer {
        for (candidates_decoded.items) |*d| d.deinit(impl.allocator);
        candidates_decoded.deinit(impl.allocator);
    }
    candidates_decoded.ensureTotalCapacity(impl.allocator, cur_p.size) catch oom();

    var candidates_grammar: std.ArrayList(Candidate) = .empty;
    defer candidates_grammar.deinit(impl.allocator);
    candidates_grammar.ensureTotalCapacity(impl.allocator, cur_p.size) catch oom();

    var piece_buf: std.ArrayList(u8) = .empty;
    defer piece_buf.deinit(impl.allocator);

    for (0..cur_p.size) |i| {
        const id = cur_p.data[i].id;
        const piece = tokenToPiece(grammar.vocab, id, &piece_buf);

        if (c.llama_vocab_is_eog(grammar.vocab, id)) {
            if (!allow_eog) cur_p.data[i].logit = -std.math.inf(f32);
        } else if (piece.len == 0 or piece[0] == 0) {
            cur_p.data[i].logit = -std.math.inf(f32);
        } else {
            var decoded: std.ArrayList(u32) = .empty;
            const partial = decodeUtf8(piece, grammar.partial_utf8, &decoded);
            candidates_decoded.appendAssumeCapacity(decoded);
            candidates_grammar.appendAssumeCapacity(.{
                .index = i,
                .code_points = candidates_decoded.items[candidates_decoded.items.len - 1].items.ptr,
                .partial_utf8 = partial,
                .id = id,
            });
        }
    }

    var rejects = rejectCandidates(grammar.rules, grammar.stacks.items, candidates_grammar.items);
    defer rejects.deinit(impl.allocator);
    for (rejects.items) |reject| {
        cur_p.data[reject.index].logit = -std.math.inf(f32);
    }
}

/// Ports `llama_grammar_accept_impl` (llama-grammar.cpp:1398 @b29c606e2).
///
/// While a lazy grammar waits, tokens are buffered; a trigger token, or a
/// trigger pattern matching the buffer, starts it, replaying the buffered
/// text from where the match says the grammar begins.
///
/// Parameters:
/// - `grammar`: the grammar.
/// - `token`: the token chosen.
///
/// Return: nothing.
pub fn acceptImpl(grammar: *Grammar, token: Token) void {
    impl.assert(grammar.vocab != null, src, 1398, "grammar.vocab != nullptr");

    var piece_buf: std.ArrayList(u8) = .empty;
    defer piece_buf.deinit(impl.allocator);
    const piece = tokenToPiece(grammar.vocab, token, &piece_buf);

    if (grammar.awaiting_trigger) {
        if (std.mem.findScalar(Token, grammar.trigger_tokens.items, token) != null) {
            grammar.awaiting_trigger = false;
            grammar.trigger_buffer.clearRetainingCapacity();
            acceptToken(grammar, token, piece);
            const z = impl.allocator.dupeSentinel(u8, piece, 0) catch oom();
            defer impl.allocator.free(z);
            impl.logDebug("Grammar triggered on token %u (`%s`)", .{ @as(c_uint, @bitCast(token)), z.ptr });
            return;
        } else {
            const start_pos = grammar.trigger_buffer.items.len;
            grammar.trigger_buffer_positions.append(impl.allocator, .{ .token = token, .start = start_pos, .end = start_pos + piece.len }) catch oom();
            grammar.trigger_buffer.appendSlice(impl.allocator, piece) catch oom();

            for (grammar.trigger_patterns.items) |*trigger_pattern| {
                if (trigger_pattern.find(grammar.trigger_buffer.items)) |start| {
                    grammar.awaiting_trigger = false;

                    // Replay the tokens that overlap [start, end).
                    for (grammar.trigger_buffer_positions.items) |tp| {
                        if (tp.end <= start) continue;

                        const piece_start = if (tp.start < start) start else tp.start;
                        const piece_len = tp.end - piece_start;
                        const tok_piece = impl.allocator.dupe(u8, grammar.trigger_buffer.items[piece_start .. piece_start + piece_len]) catch oom();
                        defer impl.allocator.free(tok_piece);
                        acceptToken(grammar, tp.token, tok_piece);
                    }

                    const constrained_str = impl.allocator.dupeSentinel(u8, grammar.trigger_buffer.items[start..], 0) catch oom();
                    defer impl.allocator.free(constrained_str);
                    grammar.trigger_buffer.clearRetainingCapacity();
                    grammar.trigger_buffer_positions.clearRetainingCapacity();
                    impl.logDebug("Grammar triggered on regex: '%s'\n", .{constrained_str.ptr});
                    return;
                }
            }
            const z = impl.allocator.dupeSentinel(u8, piece, 0) catch oom();
            defer impl.allocator.free(z);
            impl.logDebug("Grammar still awaiting trigger after token %d (`%s`)\n", .{ token, z.ptr });
            return;
        }
    }

    if (c.llama_vocab_is_eog(grammar.vocab, token)) {
        for (grammar.stacks.items) |s| {
            if (s.items.len == 0) return;
        }
        impl.abort(src, 1450, "fatal error");
    }

    acceptToken(grammar, token, piece);
}

/// Ports `llama_grammar_accept_token` (llama-grammar.cpp:1472 @b29c606e2).
///
/// Advances every stack over `piece` (or over the token itself, for a token
/// rule on top) and keeps the distinct survivors.
///
/// Parameters:
/// - `grammar`: the grammar.
/// - `token`: the token.
/// - `piece`: its text, possibly a suffix of it when replaying a trigger.
///
/// Return: nothing; terminates if no stack survives.
pub fn acceptToken(grammar: *Grammar, token: Token, piece: []const u8) void {
    var code_points: std.ArrayList(u32) = .empty;
    defer code_points.deinit(impl.allocator);
    const partial = decodeUtf8(piece, grammar.partial_utf8, &code_points);

    var stacks_new: Stacks = .empty;
    stacks_new.ensureTotalCapacity(impl.allocator, grammar.stacks.items.len) catch oom();

    for (grammar.stacks.items) |stack| {
        if (stack.items.len == 0) continue;

        const pos = stack.items[stack.items.len - 1];

        if (pos[0].type == .token or pos[0].type == .token_not) {
            if (matchToken(pos, token)) {
                var new_stack = copyStack(stack.items[0 .. stack.items.len - 1]);
                defer new_stack.deinit(impl.allocator);
                if (!isEndOfSequence(pos + 1)) {
                    new_stack.append(impl.allocator, pos + 1) catch oom();
                }
                advanceStack(grammar.rules, new_stack.items, &stacks_new);
            }
        } else {
            var current_stacks: Stacks = .empty;
            defer freeStacks(&current_stacks);
            current_stacks.append(impl.allocator, copyStack(stack.items)) catch oom();

            for (code_points.items[0 .. code_points.items.len - 1]) |cp| {
                var next_stacks: Stacks = .empty;

                for (current_stacks.items) |cur_stack| {
                    acceptChr(grammar, cur_stack.items, cp, &next_stacks);
                }

                freeStacks(&current_stacks);
                current_stacks = next_stacks;
                if (current_stacks.items.len == 0) break;
            }

            for (current_stacks.items) |surviving| {
                if (!containsStack(stacks_new.items, surviving.items)) {
                    stacks_new.append(impl.allocator, copyStack(surviving.items)) catch oom();
                }
            }
        }
    }

    freeStacks(&grammar.stacks);
    grammar.stacks = stacks_new;
    grammar.partial_utf8 = partial;

    if (grammar.stacks.items.len == 0) {
        var msg: std.ArrayList(u8) = .empty;
        defer msg.deinit(impl.allocator);
        msg.print(impl.allocator, "Unexpected empty grammar stack after accepting piece: {s} ({d})", .{ piece, token }) catch oom();
        impl.throwRuntime(src, 1522, msg.items);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "decoding carries an unfinished sequence into the next piece" {
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(impl.allocator);
    // "é" is C3 A9; split across two pieces.
    const p1 = decodeUtf8("a\xC3", .{}, &out);
    try std.testing.expectEqual(@as(c_int, 1), p1.n_remain);
    try std.testing.expectEqualSlices(u32, &.{ 'a', 0 }, out.items);
    const p2 = decodeUtf8("\xA9b", p1, &out);
    try std.testing.expectEqual(@as(c_int, 0), p2.n_remain);
    try std.testing.expectEqualSlices(u32, &.{ 0xE9, 'b', 0 }, out.items);
}
