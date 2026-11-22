//! The grammar sampler, over the ported grammar runtime.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v1.0.0, `b1b770d90`).
//! Each declaration below names the C++ it replaces and the line it began at.
//! The grammar itself is `src/llama/grammar/`.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_sampler_grammar` (llama-sampler.cpp:2667 @b1b770d90).
const Gram = struct {
    vocab: ?*const llama.impl.c.struct_llama_vocab,

    /// Kept to rebuild the grammar on reset; empty when there is none.
    grammar_str: [:0]u8,
    grammar_root: [:0]u8,

    grammar: ?*llama.grammar.grammar.Grammar,
};

fn emptyZ() [:0]u8 {
    return std.heap.c_allocator.dupeSentinel(u8, "", 0) catch llama.impl.abort(llama.sampler.core.src, 0, "out of memory");
}

/// Ports `llama_sampler_grammar_name` (llama-sampler.cpp:2676 @b1b770d90).
fn grammarName(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "grammar";
}

/// Ports `llama_sampler_grammar_accept_impl` (llama-sampler.cpp:2680 @b1b770d90).
fn grammarAccept(smpl: [*c]llama.impl.c.struct_llama_sampler, token: llama.impl.c.llama_token) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Gram, smpl);
    if (ctx.grammar) |g| llama.grammar.grammar.acceptImpl(g, token);
}

/// Ports `llama_sampler_grammar_apply` (llama-sampler.cpp:2687 @b1b770d90).
fn grammarApply(smpl: [*c]llama.impl.c.struct_llama_sampler, cur_p: [*c]llama.impl.c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Gram, smpl);
    if (ctx.grammar) |g| llama.grammar.grammar.applyImpl(g, cur_p);
}

/// Ports `llama_sampler_grammar_reset` (llama-sampler.cpp:2705 @b1b770d90).
///
/// Rebuilds the grammar from its text, keeping its trigger configuration.
fn grammarReset(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Gram, smpl);
    const g = ctx.grammar orelse return;

    var trigger_patterns_c: std.ArrayList([*:0]const u8) = .empty;
    defer trigger_patterns_c.deinit(std.heap.c_allocator);
    var owned: std.ArrayList([:0]u8) = .empty;
    defer {
        for (owned.items) |z| std.heap.c_allocator.free(z);
        owned.deinit(std.heap.c_allocator);
    }
    for (g.trigger_patterns.items) |tp| {
        const z = std.heap.c_allocator.dupeSentinel(u8, tp.pattern, 0) catch llama.impl.abort(llama.sampler.core.src, 2709, "out of memory");
        owned.append(std.heap.c_allocator, z) catch llama.impl.abort(llama.sampler.core.src, 2709, "out of memory");
        trigger_patterns_c.append(std.heap.c_allocator, z.ptr) catch llama.impl.abort(llama.sampler.core.src, 2709, "out of memory");
    }

    const grammar_new = llama.grammar.grammar.initImpl(g.vocab, ctx.grammar_str.ptr, ctx.grammar_root.ptr, g.lazy, trigger_patterns_c.items, g.trigger_tokens.items);

    llama.grammar.grammar.freeImpl(g);
    ctx.grammar = grammar_new;
}

/// Ports `llama_sampler_grammar_clone` (llama-sampler.cpp:2725 @b1b770d90).
fn grammarClone(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(Gram, smpl);

    const result = initGrammarImpl(ctx.vocab, null, null, false, &.{}, &.{});
    llama.impl.assert(result != null, llama.sampler.core.src, 2729, "result");

    const result_ctx = llama.sampler.core.ctxOf(Gram, result);
    if (ctx.grammar) |g| {
        std.heap.c_allocator.free(result_ctx.grammar_str);
        std.heap.c_allocator.free(result_ctx.grammar_root);
        result_ctx.grammar_str = std.heap.c_allocator.dupeSentinel(u8, ctx.grammar_str, 0) catch llama.impl.abort(llama.sampler.core.src, 2736, "out of memory");
        result_ctx.grammar_root = std.heap.c_allocator.dupeSentinel(u8, ctx.grammar_root, 0) catch llama.impl.abort(llama.sampler.core.src, 2737, "out of memory");

        result_ctx.grammar = llama.grammar.grammar.cloneImpl(g);
    }

    return result;
}

/// Ports `llama_sampler_grammar_free` (llama-sampler.cpp:2746 @b1b770d90).
fn grammarFree(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Gram, smpl);

    if (ctx.grammar) |g| llama.grammar.grammar.freeImpl(g);

    std.heap.c_allocator.free(ctx.grammar_str);
    std.heap.c_allocator.free(ctx.grammar_root);
    llama.sampler.core.destroy(Gram, ctx);
}

/// Ports `llama_sampler_grammar_i` (llama-sampler.cpp:2756 @b1b770d90).
var grammar_iface: llama.impl.c.struct_llama_sampler_i = .{
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
    vocab: ?*const llama.impl.c.struct_llama_vocab,
    grammar_str: ?[*:0]const u8,
    grammar_root: ?[*:0]const u8,
    lazy: bool,
    trigger_tokens: []const llama.impl.c.llama_token,
    trigger_patterns: []const [*:0]const u8,
) [*c]llama.impl.c.struct_llama_sampler {
    if (grammar_str != null and grammar_str.?[0] != 0) {
        const g = llama.grammar.grammar.initImpl(vocab, grammar_str.?, grammar_root.?, lazy, trigger_patterns, trigger_tokens);
        if (g == null) return null;

        return llama.sampler.core.llama_sampler_init(&grammar_iface, llama.sampler.core.create(Gram, .{
            .vocab = vocab,
            .grammar_str = std.heap.c_allocator.dupeSentinel(u8, std.mem.span(grammar_str.?), 0) catch llama.impl.abort(llama.sampler.core.src, 2786, "out of memory"),
            .grammar_root = std.heap.c_allocator.dupeSentinel(u8, std.mem.span(grammar_root.?), 0) catch llama.impl.abort(llama.sampler.core.src, 2787, "out of memory"),
            .grammar = g,
        }));
    }

    return llama.sampler.core.llama_sampler_init(&grammar_iface, llama.sampler.core.create(Gram, .{
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
pub export fn llama_sampler_init_grammar(vocab: ?*const llama.impl.c.struct_llama_vocab, grammar_str: [*c]const u8, grammar_root: [*c]const u8) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
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
    vocab: ?*const llama.impl.c.struct_llama_vocab,
    grammar_str: [*c]const u8,
    grammar_root: [*c]const u8,
    trigger_patterns: [*c]const [*c]const u8,
    num_trigger_patterns: usize,
    trigger_tokens: [*c]const llama.impl.c.llama_token,
    num_trigger_tokens: usize,
) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    return initGrammarImpl(
        vocab,
        @ptrCast(grammar_str),
        @ptrCast(grammar_root),
        true,
        cSlice(llama.impl.c.llama_token, trigger_tokens, num_trigger_tokens),
        @ptrCast(cSlice([*c]const u8, trigger_patterns, num_trigger_patterns)),
    );
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
