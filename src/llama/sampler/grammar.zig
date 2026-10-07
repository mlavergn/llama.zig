//! The grammar sampler, over the ported grammar runtime.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v0.3.0, `c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//! The grammar itself is `src/llama/grammar/`.

const std = @import("std");
const impl = @import("../impl.zig");
const core = @import("core.zig");
const grammar = @import("../grammar/grammar.zig");

const c = impl.c;
const Sampler = core.Sampler;
const SamplerI = core.SamplerI;
const TokenDataArray = core.TokenDataArray;
const Token = core.Token;
const src = core.src;

/// Ports `llama_sampler_grammar` (llama-sampler.cpp:2660 @c1d0e7a00).
const Gram = struct {
    vocab: ?*const c.struct_llama_vocab,

    /// Kept to rebuild the grammar on reset; empty when there is none.
    grammar_str: [:0]u8,
    grammar_root: [:0]u8,

    grammar: ?*grammar.Grammar,
};

fn emptyZ() [:0]u8 {
    return impl.allocator.dupeZ(u8, "") catch impl.abort(src, 0, "out of memory");
}

/// Ports `llama_sampler_grammar_name` (llama-sampler.cpp:2669 @c1d0e7a00).
fn grammarName(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "grammar";
}

/// Ports `llama_sampler_grammar_accept_impl` (llama-sampler.cpp:2673 @c1d0e7a00).
fn grammarAccept(smpl: [*c]Sampler, token: Token) callconv(.c) void {
    const ctx = core.ctxOf(Gram, smpl);
    if (ctx.grammar) |g| grammar.acceptImpl(g, token);
}

/// Ports `llama_sampler_grammar_apply` (llama-sampler.cpp:2680 @c1d0e7a00).
fn grammarApply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(Gram, smpl);
    if (ctx.grammar) |g| grammar.applyImpl(g, cur_p);
}

/// Ports `llama_sampler_grammar_reset` (llama-sampler.cpp:2700 @c1d0e7a00).
///
/// Rebuilds the grammar from its text, keeping its trigger configuration.
fn grammarReset(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(Gram, smpl);
    const g = ctx.grammar orelse return;

    var trigger_patterns_c: std.ArrayList([*:0]const u8) = .empty;
    defer trigger_patterns_c.deinit(impl.allocator);
    var owned: std.ArrayList([:0]u8) = .empty;
    defer {
        for (owned.items) |z| impl.allocator.free(z);
        owned.deinit(impl.allocator);
    }
    for (g.trigger_patterns.items) |tp| {
        const z = impl.allocator.dupeZ(u8, tp.pattern) catch impl.abort(src, 2709, "out of memory");
        owned.append(impl.allocator, z) catch impl.abort(src, 2709, "out of memory");
        trigger_patterns_c.append(impl.allocator, z.ptr) catch impl.abort(src, 2709, "out of memory");
    }

    const grammar_new = grammar.initImpl(g.vocab, ctx.grammar_str.ptr, ctx.grammar_root.ptr, g.lazy, trigger_patterns_c.items, g.trigger_tokens.items);

    grammar.freeImpl(g);
    ctx.grammar = grammar_new;
}

/// Ports `llama_sampler_grammar_clone` (llama-sampler.cpp:2720 @c1d0e7a00).
fn grammarClone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    const ctx = core.ctxOf(Gram, smpl);

    const result = initGrammarImpl(ctx.vocab, null, null, false, &.{}, &.{}, &.{});
    impl.assert(result != null, src, 2724, "result");

    const result_ctx = core.ctxOf(Gram, result);
    if (ctx.grammar) |g| {
        impl.allocator.free(result_ctx.grammar_str);
        impl.allocator.free(result_ctx.grammar_root);
        result_ctx.grammar_str = impl.allocator.dupeZ(u8, ctx.grammar_str) catch impl.abort(src, 2731, "out of memory");
        result_ctx.grammar_root = impl.allocator.dupeZ(u8, ctx.grammar_root) catch impl.abort(src, 2732, "out of memory");

        result_ctx.grammar = grammar.cloneImpl(g);
    }

    return result;
}

/// Ports `llama_sampler_grammar_free` (llama-sampler.cpp:2741 @c1d0e7a00).
fn grammarFree(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(Gram, smpl);

    if (ctx.grammar) |g| grammar.freeImpl(g);

    impl.allocator.free(ctx.grammar_str);
    impl.allocator.free(ctx.grammar_root);
    core.destroy(Gram, ctx);
}

/// Ports `llama_sampler_grammar_i` (llama-sampler.cpp:2751 @c1d0e7a00).
var grammar_iface: SamplerI = .{
    .name = grammarName,
    .accept = grammarAccept,
    .apply = grammarApply,
    .reset = grammarReset,
    .clone = grammarClone,
    .free = grammarFree,
};

/// The characters `special_chars` (llama-sampler.cpp:2787 @c1d0e7a00), in
/// `llama_sampler_init_grammar_impl`, matches: `[.^$|()*+?\[\]{}\\]`; `regex_replace` with `"\\$0"` prefixes
/// each with a backslash.
fn isRegexSpecial(ch: u8) bool {
    return std.mem.indexOfScalar(u8, ".^$|()*+?[]{}\\", ch) != null;
}

/// Ports `llama_sampler_init_grammar_impl` (llama-sampler.cpp:2766 @c1d0e7a00).
///
/// An empty or null grammar text gives a sampler with no grammar, which
/// constrains nothing. Trigger words, the deprecated form, become one
/// pattern that captures from the first of them.
fn initGrammarImpl(
    vocab: ?*const c.struct_llama_vocab,
    grammar_str: ?[*:0]const u8,
    grammar_root: ?[*:0]const u8,
    lazy: bool,
    trigger_words: []const [*:0]const u8,
    trigger_tokens: []const Token,
    trigger_patterns: []const [*:0]const u8,
) [*c]Sampler {
    if (grammar_str != null and grammar_str.?[0] != 0) {
        var g: ?*grammar.Grammar = null;
        if (trigger_words.len > 0) {
            impl.assert(trigger_patterns.len == 0, src, 2784, "trigger_patterns == nullptr && num_trigger_patterns == 0");
            var trigger_pattern: std.ArrayList(u8) = .empty;
            defer trigger_pattern.deinit(impl.allocator);
            trigger_pattern.appendSlice(impl.allocator, "[\\s\\S]*?(") catch impl.abort(src, 2785, "out of memory");
            for (trigger_words, 0..) |word, i| {
                if (i > 0) trigger_pattern.append(impl.allocator, '|') catch impl.abort(src, 2789, "out of memory");
                for (std.mem.span(word)) |ch| {
                    if (isRegexSpecial(ch)) trigger_pattern.append(impl.allocator, '\\') catch impl.abort(src, 2791, "out of memory");
                    trigger_pattern.append(impl.allocator, ch) catch impl.abort(src, 2791, "out of memory");
                }
            }
            trigger_pattern.appendSlice(impl.allocator, ")[\\s\\S]*") catch impl.abort(src, 2793, "out of memory");
            trigger_pattern.append(impl.allocator, 0) catch impl.abort(src, 2793, "out of memory");

            const tmp = [_][*:0]const u8{@ptrCast(trigger_pattern.items.ptr)};
            g = grammar.initImpl(vocab, grammar_str.?, grammar_root.?, lazy, &tmp, trigger_tokens);
        } else {
            g = grammar.initImpl(vocab, grammar_str.?, grammar_root.?, lazy, trigger_patterns, trigger_tokens);
        }
        if (g == null) return null;

        return core.llama_sampler_init(&grammar_iface, core.create(Gram, .{
            .vocab = vocab,
            .grammar_str = impl.allocator.dupeZ(u8, std.mem.span(grammar_str.?)) catch impl.abort(src, 2802, "out of memory"),
            .grammar_root = impl.allocator.dupeZ(u8, std.mem.span(grammar_root.?)) catch impl.abort(src, 2803, "out of memory"),
            .grammar = g,
        }));
    }

    return core.llama_sampler_init(&grammar_iface, core.create(Gram, .{
        .vocab = vocab,
        .grammar_str = emptyZ(),
        .grammar_root = emptyZ(),
        .grammar = null,
    }));
}

/// `(ptr, n)` as a slice, treating a null pointer as empty.
fn cSlice(comptime T: type, ptr: [*c]const T, n: usize) []const T {
    if (ptr == null or n == 0) return &.{};
    return ptr[0..n];
}

/// Ports `llama_sampler_init_grammar` (llama-sampler.cpp:2825 @c1d0e7a00).
///
/// Parameters:
/// - `vocab`: the vocabulary; borrowed for the sampler's lifetime.
/// - `grammar_str`: GBNF text; empty gives a sampler that constrains nothing.
/// - `grammar_root`: the start symbol.
///
/// Return: the sampler, or null when the grammar does not parse.
pub export fn llama_sampler_init_grammar(vocab: ?*const c.struct_llama_vocab, grammar_str: [*c]const u8, grammar_root: [*c]const u8) callconv(.c) [*c]Sampler {
    return initGrammarImpl(vocab, @ptrCast(grammar_str), @ptrCast(grammar_root), false, &.{}, &.{}, &.{});
}

/// Ports `llama_sampler_init_grammar_lazy` (llama-sampler.cpp:2832 @c1d0e7a00).
///
/// Deprecated upstream in favour of the patterns form.
///
/// Return: the sampler, or null when the grammar does not parse.
pub export fn llama_sampler_init_grammar_lazy(
    vocab: ?*const c.struct_llama_vocab,
    grammar_str: [*c]const u8,
    grammar_root: [*c]const u8,
    trigger_words: [*c]const [*c]const u8,
    num_trigger_words: usize,
    trigger_tokens: [*c]const Token,
    num_trigger_tokens: usize,
) callconv(.c) [*c]Sampler {
    return initGrammarImpl(
        vocab,
        @ptrCast(grammar_str),
        @ptrCast(grammar_root),
        true,
        @ptrCast(cSlice([*c]const u8, trigger_words, num_trigger_words)),
        cSlice(Token, trigger_tokens, num_trigger_tokens),
        &.{},
    );
}

/// Ports `llama_sampler_init_grammar_lazy_patterns` (llama-sampler.cpp:2843 @c1d0e7a00).
///
/// Parameters:
/// - `vocab`: the vocabulary; borrowed for the sampler's lifetime.
/// - `grammar_str`: GBNF text.
/// - `grammar_root`: the start symbol.
/// - `trigger_patterns`: ECMAScript regexes; the grammar starts at the first
///   non-empty capture group of the first that matches the output so far.
/// - `num_trigger_patterns`: how many.
/// - `trigger_tokens`: tokens that start the grammar at themselves.
/// - `num_trigger_tokens`: how many.
///
/// Return: the sampler, or null when the grammar does not parse.
pub export fn llama_sampler_init_grammar_lazy_patterns(
    vocab: ?*const c.struct_llama_vocab,
    grammar_str: [*c]const u8,
    grammar_root: [*c]const u8,
    trigger_patterns: [*c]const [*c]const u8,
    num_trigger_patterns: usize,
    trigger_tokens: [*c]const Token,
    num_trigger_tokens: usize,
) callconv(.c) [*c]Sampler {
    return initGrammarImpl(
        vocab,
        @ptrCast(grammar_str),
        @ptrCast(grammar_root),
        true,
        &.{},
        cSlice(Token, trigger_tokens, num_trigger_tokens),
        @ptrCast(cSlice([*c]const u8, trigger_patterns, num_trigger_patterns)),
    );
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
