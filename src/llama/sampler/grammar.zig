//! The grammar sampler, over the ported grammar runtime.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v1.0.0, `b1b770d90`).
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

/// Ports `llama_sampler_grammar` (llama-sampler.cpp:2667 @b1b770d90).
const Gram = struct {
    vocab: ?*const c.struct_llama_vocab,

    /// Kept to rebuild the grammar on reset; empty when there is none.
    grammar_str: [:0]u8,
    grammar_root: [:0]u8,

    grammar: ?*grammar.Grammar,
};

fn emptyZ() [:0]u8 {
    return impl.allocator.dupeSentinel(u8, "", 0) catch impl.abort(src, 0, "out of memory");
}

/// Ports `llama_sampler_grammar_name` (llama-sampler.cpp:2676 @b1b770d90).
fn grammarName(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "grammar";
}

/// Ports `llama_sampler_grammar_accept_impl` (llama-sampler.cpp:2680 @b1b770d90).
fn grammarAccept(smpl: [*c]Sampler, token: Token) callconv(.c) void {
    const ctx = core.ctxOf(Gram, smpl);
    if (ctx.grammar) |g| grammar.acceptImpl(g, token);
}

/// Ports `llama_sampler_grammar_apply` (llama-sampler.cpp:2687 @b1b770d90).
fn grammarApply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(Gram, smpl);
    if (ctx.grammar) |g| grammar.applyImpl(g, cur_p);
}

/// Ports `llama_sampler_grammar_reset` (llama-sampler.cpp:2705 @b1b770d90).
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
        const z = impl.allocator.dupeSentinel(u8, tp.pattern, 0) catch impl.abort(src, 2709, "out of memory");
        owned.append(impl.allocator, z) catch impl.abort(src, 2709, "out of memory");
        trigger_patterns_c.append(impl.allocator, z.ptr) catch impl.abort(src, 2709, "out of memory");
    }

    const grammar_new = grammar.initImpl(g.vocab, ctx.grammar_str.ptr, ctx.grammar_root.ptr, g.lazy, trigger_patterns_c.items, g.trigger_tokens.items);

    grammar.freeImpl(g);
    ctx.grammar = grammar_new;
}

/// Ports `llama_sampler_grammar_clone` (llama-sampler.cpp:2725 @b1b770d90).
fn grammarClone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    const ctx = core.ctxOf(Gram, smpl);

    const result = initGrammarImpl(ctx.vocab, null, null, false, &.{}, &.{});
    impl.assert(result != null, src, 2729, "result");

    const result_ctx = core.ctxOf(Gram, result);
    if (ctx.grammar) |g| {
        impl.allocator.free(result_ctx.grammar_str);
        impl.allocator.free(result_ctx.grammar_root);
        result_ctx.grammar_str = impl.allocator.dupeSentinel(u8, ctx.grammar_str, 0) catch impl.abort(src, 2736, "out of memory");
        result_ctx.grammar_root = impl.allocator.dupeSentinel(u8, ctx.grammar_root, 0) catch impl.abort(src, 2737, "out of memory");

        result_ctx.grammar = grammar.cloneImpl(g);
    }

    return result;
}

/// Ports `llama_sampler_grammar_free` (llama-sampler.cpp:2746 @b1b770d90).
fn grammarFree(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(Gram, smpl);

    if (ctx.grammar) |g| grammar.freeImpl(g);

    impl.allocator.free(ctx.grammar_str);
    impl.allocator.free(ctx.grammar_root);
    core.destroy(Gram, ctx);
}

/// Ports `llama_sampler_grammar_i` (llama-sampler.cpp:2756 @b1b770d90).
var grammar_iface: SamplerI = .{
    .name = grammarName,
    .accept = grammarAccept,
    .apply = grammarApply,
    .reset = grammarReset,
    .clone = grammarClone,
    .free = grammarFree,
};

/// Ports `llama_sampler_init_grammar_impl` (llama-sampler.cpp:2771 @b1b770d90).
///
/// An empty or null grammar text gives a sampler with no grammar, which
/// constrains nothing.
fn initGrammarImpl(
    vocab: ?*const c.struct_llama_vocab,
    grammar_str: ?[*:0]const u8,
    grammar_root: ?[*:0]const u8,
    lazy: bool,
    trigger_tokens: []const Token,
    trigger_patterns: []const [*:0]const u8,
) [*c]Sampler {
    if (grammar_str != null and grammar_str.?[0] != 0) {
        const g = grammar.initImpl(vocab, grammar_str.?, grammar_root.?, lazy, trigger_patterns, trigger_tokens);
        if (g == null) return null;

        return core.llama_sampler_init(&grammar_iface, core.create(Gram, .{
            .vocab = vocab,
            .grammar_str = impl.allocator.dupeSentinel(u8, std.mem.span(grammar_str.?), 0) catch impl.abort(src, 2786, "out of memory"),
            .grammar_root = impl.allocator.dupeSentinel(u8, std.mem.span(grammar_root.?), 0) catch impl.abort(src, 2787, "out of memory"),
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

/// Ports `llama_sampler_init_grammar` (llama-sampler.cpp:2809 @b1b770d90).
///
/// Parameters:
/// - `vocab`: the vocabulary; borrowed for the sampler's lifetime.
/// - `grammar_str`: GBNF text; empty gives a sampler that constrains nothing.
/// - `grammar_root`: the start symbol.
///
/// Return: the sampler, or null when the grammar does not parse.
pub export fn llama_sampler_init_grammar(vocab: ?*const c.struct_llama_vocab, grammar_str: [*c]const u8, grammar_root: [*c]const u8) callconv(.c) [*c]Sampler {
    return initGrammarImpl(vocab, @ptrCast(grammar_str), @ptrCast(grammar_root), false, &.{}, &.{});
}

/// Ports `llama_sampler_init_grammar_lazy_patterns` (llama-sampler.cpp:2816 @b1b770d90).
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
        cSlice(Token, trigger_tokens, num_trigger_tokens),
        @ptrCast(cSlice([*c]const u8, trigger_patterns, num_trigger_patterns)),
    );
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
