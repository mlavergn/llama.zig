//! `llama.h`'s vocabulary entry points: token properties, special tokens,
//! tokenizing and detokenizing.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` (v1.0.0, `b1b770d90`): the
//! C API at the end of the file, from `llama_vocab_n_tokens` through
//! `llama_detokenize` -- its 27 unmangled exports -- with the forwarding
//! members of `llama_vocab` they go through. The vocabulary is
//! `vocab/vocab.zig`.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.
//!
//! # Exceptions at the C boundary
//!
//! Where the C++ throws -- `id_to_token.at` for a token past the
//! vocabulary, a tokenizer's own `std::runtime_error`, `std::string`'s
//! `length_error` for a negative length -- nothing between it and the C
//! caller catches, and the process terminates. The port's vocabulary
//! returns `error.Thrown`, and these entry points terminate with it.
//!
//! # Lengths a C caller can get wrong
//!
//! The C++ takes `(buf, length)` and never forms a slice; the port's
//! vocabulary takes slices. A negative length becomes an empty slice, which
//! gives the C++'s answer everywhere but one place: an RWKV piece into a
//! negative `length`, where the C++ compares against `(size_t)length` and
//! copies past the buffer. The port reports the length needed instead.

const std = @import("std");
const llama = @import("../module.zig");

/// The port's vocabulary behind a `llama_vocab *`: the one inside a model,
/// as `llama_model_get_vocab` hands it out.
pub fn vocabOf(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) *const llama.vocab.Vocab {
    return @ptrCast(@alignCast(vocab.?));
}

/// `(ptr, n)` as a slice; empty for a non-positive `n`, so a null pointer
/// with no length is never sliced.
fn cSlice(comptime T: type, ptr: [*c]T, n: i32) []T {
    if (n <= 0) return &.{};
    return ptr[0..@intCast(n)];
}

/// Ports `llama_vocab`'s `token_get_text` (llama-vocab.cpp:4477 @b1b770d90).
fn tokenGetText(v: *const llama.vocab.Vocab, id: llama.impl.llama_c.llama_token) [*c]const u8 {
    llama.impl.assert(v.type != llama.vocab.vocab.type_none, "llama-vocab.cpp", 4478, "pimpl->type != LLAMA_VOCAB_TYPE_NONE");
    var exc: llama.impl.Exception = .{};
    _ = v.tokenDataAt(id, &exc) catch exc.terminate("llama-vocab.cpp", 4479);
    return v.text(id).ptr;
}

/// Ports `llama_vocab`'s `token_get_score` (llama-vocab.cpp:4482 @b1b770d90).
fn tokenGetScore(v: *const llama.vocab.Vocab, id: llama.impl.llama_c.llama_token) f32 {
    llama.impl.assert(v.type != llama.vocab.vocab.type_none, "llama-vocab.cpp", 4483, "pimpl->type != LLAMA_VOCAB_TYPE_NONE");
    var exc: llama.impl.Exception = .{};
    const d = v.tokenDataAt(id, &exc) catch exc.terminate("llama-vocab.cpp", 4484);
    return d.score;
}

/// Ports `llama_vocab_n_tokens` (llama-vocab.cpp:4719 @b1b770d90).
pub fn llama_vocab_n_tokens(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) i32 {
    return @bitCast(vocabOf(vocab).nTokens());
}

/// Ports `llama_vocab_type` (llama-vocab.cpp:4723 @b1b770d90).
pub fn llama_vocab_type(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.enum_llama_vocab_type {
    return vocabOf(vocab).type;
}

/// Ports `llama_vocab_get_text` (llama-vocab.cpp:4727 @b1b770d90).
///
/// Return: the token's text, borrowed from the vocabulary.
pub fn llama_vocab_get_text(vocab: ?*const llama.impl.llama_c.struct_llama_vocab, token: llama.impl.llama_c.llama_token) callconv(.c) [*c]const u8 {
    return tokenGetText(vocabOf(vocab), token);
}

/// Ports `llama_vocab_get_score` (llama-vocab.cpp:4731 @b1b770d90).
pub fn llama_vocab_get_score(vocab: ?*const llama.impl.llama_c.struct_llama_vocab, token: llama.impl.llama_c.llama_token) callconv(.c) f32 {
    return tokenGetScore(vocabOf(vocab), token);
}

/// Ports `llama_vocab_get_attr` (llama-vocab.cpp:4735 @b1b770d90).
pub fn llama_vocab_get_attr(vocab: ?*const llama.impl.llama_c.struct_llama_vocab, token: llama.impl.llama_c.llama_token) callconv(.c) llama.impl.llama_c.enum_llama_token_attr {
    var exc: llama.impl.Exception = .{};
    return vocabOf(vocab).tokenGetAttr(token, &exc) catch exc.terminate("llama-vocab.cpp", 4736);
}

/// Ports `llama_vocab_is_eog` (llama-vocab.cpp:4739 @b1b770d90).
pub fn llama_vocab_is_eog(vocab: ?*const llama.impl.llama_c.struct_llama_vocab, token: llama.impl.llama_c.llama_token) callconv(.c) bool {
    return vocabOf(vocab).isEog(token);
}

/// Ports `llama_vocab_is_control` (llama-vocab.cpp:4743 @b1b770d90): no
/// bounds check, as the C++'s `id_to_token[id]` has none.
pub fn llama_vocab_is_control(vocab: ?*const llama.impl.llama_c.struct_llama_vocab, token: llama.impl.llama_c.llama_token) callconv(.c) bool {
    return vocabOf(vocab).isControl(token);
}

/// Ports `llama_vocab_bos` (llama-vocab.cpp:4747 @b1b770d90).
pub fn llama_vocab_bos(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_bos_id;
}

/// Ports `llama_vocab_eos` (llama-vocab.cpp:4751 @b1b770d90).
pub fn llama_vocab_eos(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_eos_id;
}

/// Ports `llama_vocab_eot` (llama-vocab.cpp:4755 @b1b770d90).
pub fn llama_vocab_eot(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_eot_id;
}

/// Ports `llama_vocab_sep` (llama-vocab.cpp:4759 @b1b770d90).
pub fn llama_vocab_sep(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_sep_id;
}

/// Ports `llama_vocab_nl` (llama-vocab.cpp:4763 @b1b770d90).
pub fn llama_vocab_nl(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).linefeed_id;
}

/// Ports `llama_vocab_pad` (llama-vocab.cpp:4767 @b1b770d90).
pub fn llama_vocab_pad(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_pad_id;
}

/// Ports `llama_vocab_get_add_bos` (llama-vocab.cpp:4771 @b1b770d90).
pub fn llama_vocab_get_add_bos(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) bool {
    return vocabOf(vocab).add_bos;
}

/// Ports `llama_vocab_get_add_eos` (llama-vocab.cpp:4775 @b1b770d90).
pub fn llama_vocab_get_add_eos(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) bool {
    return vocabOf(vocab).add_eos;
}

/// Ports `llama_vocab_get_add_sep` (llama-vocab.cpp:4779 @b1b770d90).
pub fn llama_vocab_get_add_sep(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) bool {
    return vocabOf(vocab).add_sep;
}

/// Ports `llama_vocab_get_suppress_tokens` (llama-vocab.cpp:4783 @b1b770d90).
///
/// Parameters:
/// - `vocab`: the vocabulary.
/// - `n_suppress_tokens`: receives the count, when not null.
///
/// Return: the tokens, borrowed from the vocabulary; null, as libc++'s
/// `data()` of a vector that never allocated is, when there are none.
pub fn llama_vocab_get_suppress_tokens(vocab: ?*const llama.impl.llama_c.struct_llama_vocab, n_suppress_tokens: [*c]i32) callconv(.c) [*c]const llama.impl.llama_c.llama_token {
    const tokens = &vocabOf(vocab).suppress_tokens;
    if (n_suppress_tokens != null) n_suppress_tokens.* = @intCast(tokens.items.len);
    if (tokens.capacity == 0) return null;
    return tokens.items.ptr;
}

/// Ports `llama_vocab_fim_pre` (llama-vocab.cpp:4791 @b1b770d90).
pub fn llama_vocab_fim_pre(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_fim_pre_id;
}

/// Ports `llama_vocab_fim_suf` (llama-vocab.cpp:4795 @b1b770d90).
pub fn llama_vocab_fim_suf(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_fim_suf_id;
}

/// Ports `llama_vocab_fim_mid` (llama-vocab.cpp:4799 @b1b770d90).
pub fn llama_vocab_fim_mid(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_fim_mid_id;
}

/// Ports `llama_vocab_fim_pad` (llama-vocab.cpp:4803 @b1b770d90).
pub fn llama_vocab_fim_pad(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_fim_pad_id;
}

/// Ports `llama_vocab_fim_rep` (llama-vocab.cpp:4807 @b1b770d90).
pub fn llama_vocab_fim_rep(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_fim_rep_id;
}

/// Ports `llama_vocab_fim_sep` (llama-vocab.cpp:4811 @b1b770d90).
pub fn llama_vocab_fim_sep(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_fim_sep_id;
}

/// Ports `llama_vocab_mask` (llama-vocab.cpp:4815 @b1b770d90).
pub fn llama_vocab_mask(vocab: ?*const llama.impl.llama_c.struct_llama_vocab) callconv(.c) llama.impl.llama_c.llama_token {
    return vocabOf(vocab).special_mask_id;
}

/// Ports `llama_tokenize` (llama-vocab.cpp:4823 @b1b770d90), through
/// `llama_vocab`'s C-buffer `tokenize` (llama-vocab.cpp:4645 @b1b770d90).
///
/// Return: the number of tokens written; minus the number needed when
/// `n_tokens_max` is too small; `INT32_MIN` past `int32_t`.
pub fn llama_tokenize(
    vocab: ?*const llama.impl.llama_c.struct_llama_vocab,
    text: [*c]const u8,
    text_len: i32,
    tokens: [*c]llama.impl.llama_c.llama_token,
    n_tokens_max: i32,
    add_special: bool,
    parse_special: bool,
) callconv(.c) i32 {
    var exc: llama.impl.Exception = .{};
    // `std::string(text, text_len)`: a negative length is a huge `size_t`.
    if (text_len < 0) {
        exc.set(.length_error, "basic_string", .{});
        exc.terminate("llama-vocab.cpp", 4652);
    }
    const t: []const u8 = if (text_len == 0) "" else text[0..@intCast(text_len)];
    return tokenize(vocab, t, cSlice(llama.impl.llama_c.llama_token, tokens, n_tokens_max), add_special, parse_special, &exc) catch
        exc.terminate("llama-vocab.cpp", 4831);
}

/// `llama_tokenize` for Zig callers. Where upstream's tokenizer throws -- a
/// `std::out_of_range` from a vocabulary with no byte token for a byte of
/// the text -- this returns `error.Thrown`, with the exception's class and
/// text in `exc`; the C entry point above terminates, as an exception
/// leaving a C function does. The callers whose upstream counterparts
/// catch it call this one: the server's `common_tokenize`, under its
/// handlers' `ex_wrapper`, and libmtmd's `mtmd_tokenize`. Not part of
/// `llama.h`, so `exports.zig` leaves it alone.
///
/// Parameters:
/// - `vocab`: the vocabulary.
/// - `text`: the text to tokenize.
/// - `tokens`: where the tokens go; empty to ask for the count.
/// - `add_special`, `parse_special`: as `llama_tokenize`'s.
/// - `exc`: receives the exception on `error.Thrown`.
///
/// Return: as `llama_tokenize`'s, or `error.Thrown`.
pub fn tokenize(
    vocab: ?*const llama.impl.llama_c.struct_llama_vocab,
    text: []const u8,
    tokens: []llama.impl.llama_c.llama_token,
    add_special: bool,
    parse_special: bool,
    exc: *llama.impl.Exception,
) llama.impl.Thrown!i32 {
    return vocabOf(vocab).tokenizeInto(text, tokens, add_special, parse_special, exc);
}

/// Ports `llama_token_to_piece` (llama-vocab.cpp:4834 @b1b770d90). Writes
/// no NUL.
///
/// Return: the bytes written, or minus the bytes needed.
pub fn llama_token_to_piece(
    vocab: ?*const llama.impl.llama_c.struct_llama_vocab,
    token: llama.impl.llama_c.llama_token,
    buf: [*c]u8,
    length: i32,
    lstrip: i32,
    special: bool,
) callconv(.c) i32 {
    var exc: llama.impl.Exception = .{};
    return tokenToPiece(vocab, token, cSlice(u8, buf, length), lstrip, special, &exc) catch
        exc.terminate("llama-vocab.cpp", 4841);
}

/// `llama_token_to_piece` for Zig callers, as `tokenize` is for
/// `llama_tokenize`: a token id outside the vocabulary, where upstream
/// throws `std::out_of_range`, is `error.Thrown` with the exception in
/// `exc`. The server's `tokens_to_str` takes token ids from the client
/// (`/detokenize`), and upstream's `ex_wrapper` answers those with 500.
///
/// Parameters:
/// - `vocab`: the vocabulary.
/// - `token`: the token.
/// - `buf`: where the piece goes; empty to ask for its length.
/// - `lstrip`, `special`: as `llama_token_to_piece`'s.
/// - `exc`: receives the exception on `error.Thrown`.
///
/// Return: as `llama_token_to_piece`'s, or `error.Thrown`.
pub fn tokenToPiece(
    vocab: ?*const llama.impl.llama_c.struct_llama_vocab,
    token: llama.impl.llama_c.llama_token,
    buf: []u8,
    lstrip: i32,
    special: bool,
    exc: *llama.impl.Exception,
) llama.impl.Thrown!i32 {
    return vocabOf(vocab).tokenToPiece(token, buf, lstrip, special, exc);
}

/// Ports `llama_detokenize` (llama-vocab.cpp:4844 @b1b770d90). Writes no
/// NUL.
///
/// A negative `text_len_max` is the C++'s `avail` starting negative: its
/// `GGML_ASSERT(avail >= 0)` fires on the first token left once
/// `remove_special` has dropped the BOS and EOS, and with none left the
/// answer is 0.
///
/// Return: the bytes written, or minus the bytes needed.
pub fn llama_detokenize(
    vocab: ?*const llama.impl.llama_c.struct_llama_vocab,
    tokens: [*c]const llama.impl.llama_c.llama_token,
    n_tokens: i32,
    text: [*c]u8,
    text_len_max: i32,
    remove_special: bool,
    unparse_special: bool,
) callconv(.c) i32 {
    const v = vocabOf(vocab);
    const toks: []const llama.impl.llama_c.llama_token = if (n_tokens <= 0) &.{} else tokens[0..@intCast(n_tokens)];
    if (text_len_max < 0) {
        if (v.type == llama.vocab.vocab.type_none) return 0;
        llama.impl.assert(v.tokenizer != .none, "llama-vocab.cpp", 4221, "tokenizer && \"Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.\"");
        var rest = toks;
        if (remove_special and v.add_bos and rest.len > 0 and rest[0] == v.special_bos_id) rest = rest[1..];
        if (remove_special and v.add_eos and rest.len > 0 and rest[rest.len - 1] == v.special_eos_id) rest = rest[0 .. rest.len - 1];
        llama.impl.assert(rest.len == 0, "llama-vocab.cpp", 4248, "avail >= 0");
        return 0;
    }
    var exc: llama.impl.Exception = .{};
    return detokenize(vocab, toks, cSlice(u8, text, text_len_max), remove_special, unparse_special, &exc) catch
        exc.terminate("llama-vocab.cpp", 4852);
}

/// `llama_detokenize` for Zig callers, as `tokenize` is for
/// `llama_tokenize`: a token id outside the vocabulary is `error.Thrown`
/// with the exception in `exc`. Takes a slice, so the C entry point's
/// negative-length arm has no counterpart here.
///
/// Parameters:
/// - `vocab`: the vocabulary.
/// - `tokens`: the tokens.
/// - `text`: where the text goes; empty to ask for its length.
/// - `remove_special`, `unparse_special`: as `llama_detokenize`'s.
/// - `exc`: receives the exception on `error.Thrown`.
///
/// Return: as `llama_detokenize`'s, or `error.Thrown`.
pub fn detokenize(
    vocab: ?*const llama.impl.llama_c.struct_llama_vocab,
    tokens: []const llama.impl.llama_c.llama_token,
    text: []u8,
    remove_special: bool,
    unparse_special: bool,
    exc: *llama.impl.Exception,
) llama.impl.Thrown!i32 {
    return vocabOf(vocab).detokenize(tokens, text, remove_special, unparse_special, exc);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
