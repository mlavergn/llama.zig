//! `llama.h`'s vocabulary entry points: token properties, special tokens,
//! tokenizing and detokenizing.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` (v0.3.0, `c1d0e7a00`): the
//! C API at the end of the file, from `llama_vocab_n_tokens` through
//! `llama_detokenize` -- its 49 unmangled exports -- with the forwarding
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
const impl = @import("../impl.zig");
const vm = @import("../vocab/module.zig");

const c = impl.c;
const Vocab = vm.Vocab;
const Token = c.llama_token;

/// The port's vocabulary behind a `llama_vocab *`: the one inside a model,
/// as `llama_model_get_vocab` hands it out.
pub fn vocabOf(vocab: ?*const c.struct_llama_vocab) *const Vocab {
    return @ptrCast(@alignCast(vocab.?));
}

/// `(ptr, n)` as a slice; empty for a non-positive `n`, so a null pointer
/// with no length is never sliced.
fn cSlice(comptime T: type, ptr: [*c]T, n: i32) []T {
    if (n <= 0) return &.{};
    return ptr[0..@intCast(n)];
}

/// Ports `llama_vocab`'s `token_get_text` (llama-vocab.cpp:3957 @c1d0e7a00).
fn tokenGetText(v: *const Vocab, id: Token) [*c]const u8 {
    impl.assert(v.type != vm.vocab.type_none, "llama-vocab.cpp", 3958, "pimpl->type != LLAMA_VOCAB_TYPE_NONE");
    var exc: impl.Exception = .{};
    _ = v.tokenDataAt(id, &exc) catch exc.terminate("llama-vocab.cpp", 3959);
    return v.text(id).ptr;
}

/// Ports `llama_vocab`'s `token_get_score` (llama-vocab.cpp:3962 @c1d0e7a00).
fn tokenGetScore(v: *const Vocab, id: Token) f32 {
    impl.assert(v.type != vm.vocab.type_none, "llama-vocab.cpp", 3963, "pimpl->type != LLAMA_VOCAB_TYPE_NONE");
    var exc: impl.Exception = .{};
    const d = v.tokenDataAt(id, &exc) catch exc.terminate("llama-vocab.cpp", 3964);
    return d.score;
}

/// Ports `llama_vocab_n_tokens` (llama-vocab.cpp:4195 @c1d0e7a00).
pub fn llama_vocab_n_tokens(vocab: ?*const c.struct_llama_vocab) callconv(.c) i32 {
    return @bitCast(vocabOf(vocab).nTokens());
}

/// Ports `llama_n_vocab` (llama-vocab.cpp:4200 @c1d0e7a00).
pub fn llama_n_vocab(vocab: ?*const c.struct_llama_vocab) callconv(.c) i32 {
    return llama_vocab_n_tokens(vocab);
}

/// Ports `llama_vocab_type` (llama-vocab.cpp:4204 @c1d0e7a00).
pub fn llama_vocab_type(vocab: ?*const c.struct_llama_vocab) callconv(.c) c.enum_llama_vocab_type {
    return vocabOf(vocab).type;
}

/// Ports `llama_vocab_get_text` (llama-vocab.cpp:4208 @c1d0e7a00).
///
/// Return: the token's text, borrowed from the vocabulary.
pub fn llama_vocab_get_text(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) [*c]const u8 {
    return tokenGetText(vocabOf(vocab), token);
}

/// Ports `llama_vocab_get_score` (llama-vocab.cpp:4212 @c1d0e7a00).
pub fn llama_vocab_get_score(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) f32 {
    return tokenGetScore(vocabOf(vocab), token);
}

/// Ports `llama_vocab_get_attr` (llama-vocab.cpp:4216 @c1d0e7a00).
pub fn llama_vocab_get_attr(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) c.enum_llama_token_attr {
    var exc: impl.Exception = .{};
    return vocabOf(vocab).tokenGetAttr(token, &exc) catch exc.terminate("llama-vocab.cpp", 4217);
}

/// Ports `llama_vocab_is_eog` (llama-vocab.cpp:4220 @c1d0e7a00).
pub fn llama_vocab_is_eog(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) bool {
    return vocabOf(vocab).isEog(token);
}

/// Ports `llama_vocab_is_control` (llama-vocab.cpp:4224 @c1d0e7a00): no
/// bounds check, as the C++'s `id_to_token[id]` has none.
pub fn llama_vocab_is_control(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) bool {
    return vocabOf(vocab).isControl(token);
}

/// Ports `llama_vocab_bos` (llama-vocab.cpp:4228 @c1d0e7a00).
pub fn llama_vocab_bos(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_bos_id;
}

/// Ports `llama_vocab_eos` (llama-vocab.cpp:4232 @c1d0e7a00).
pub fn llama_vocab_eos(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_eos_id;
}

/// Ports `llama_vocab_eot` (llama-vocab.cpp:4236 @c1d0e7a00).
pub fn llama_vocab_eot(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_eot_id;
}

/// Ports `llama_vocab_cls` (llama-vocab.cpp:4241 @c1d0e7a00): deprecated,
/// and BOS.
pub fn llama_vocab_cls(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_bos_id;
}

/// Ports `llama_vocab_sep` (llama-vocab.cpp:4245 @c1d0e7a00).
pub fn llama_vocab_sep(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_sep_id;
}

/// Ports `llama_vocab_nl` (llama-vocab.cpp:4249 @c1d0e7a00).
pub fn llama_vocab_nl(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).linefeed_id;
}

/// Ports `llama_vocab_pad` (llama-vocab.cpp:4253 @c1d0e7a00).
pub fn llama_vocab_pad(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_pad_id;
}

/// Ports `llama_vocab_get_add_bos` (llama-vocab.cpp:4257 @c1d0e7a00).
pub fn llama_vocab_get_add_bos(vocab: ?*const c.struct_llama_vocab) callconv(.c) bool {
    return vocabOf(vocab).add_bos;
}

/// Ports `llama_vocab_get_add_eos` (llama-vocab.cpp:4261 @c1d0e7a00).
pub fn llama_vocab_get_add_eos(vocab: ?*const c.struct_llama_vocab) callconv(.c) bool {
    return vocabOf(vocab).add_eos;
}

/// Ports `llama_vocab_get_add_sep` (llama-vocab.cpp:4265 @c1d0e7a00).
pub fn llama_vocab_get_add_sep(vocab: ?*const c.struct_llama_vocab) callconv(.c) bool {
    return vocabOf(vocab).add_sep;
}

/// Ports `llama_vocab_get_suppress_tokens` (llama-vocab.cpp:4269 @c1d0e7a00).
///
/// Parameters:
/// - `vocab`: the vocabulary.
/// - `n_suppress_tokens`: receives the count, when not null.
///
/// Return: the tokens, borrowed from the vocabulary; null, as libc++'s
/// `data()` of a vector that never allocated is, when there are none.
pub fn llama_vocab_get_suppress_tokens(vocab: ?*const c.struct_llama_vocab, n_suppress_tokens: [*c]i32) callconv(.c) [*c]const Token {
    const tokens = &vocabOf(vocab).suppress_tokens;
    if (n_suppress_tokens != null) n_suppress_tokens.* = @intCast(tokens.items.len);
    if (tokens.capacity == 0) return null;
    return tokens.items.ptr;
}

/// Ports `llama_vocab_fim_pre` (llama-vocab.cpp:4277 @c1d0e7a00).
pub fn llama_vocab_fim_pre(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_fim_pre_id;
}

/// Ports `llama_vocab_fim_suf` (llama-vocab.cpp:4281 @c1d0e7a00).
pub fn llama_vocab_fim_suf(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_fim_suf_id;
}

/// Ports `llama_vocab_fim_mid` (llama-vocab.cpp:4285 @c1d0e7a00).
pub fn llama_vocab_fim_mid(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_fim_mid_id;
}

/// Ports `llama_vocab_fim_pad` (llama-vocab.cpp:4289 @c1d0e7a00).
pub fn llama_vocab_fim_pad(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_fim_pad_id;
}

/// Ports `llama_vocab_fim_rep` (llama-vocab.cpp:4293 @c1d0e7a00).
pub fn llama_vocab_fim_rep(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_fim_rep_id;
}

/// Ports `llama_vocab_fim_sep` (llama-vocab.cpp:4297 @c1d0e7a00).
pub fn llama_vocab_fim_sep(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_fim_sep_id;
}

/// Ports `llama_vocab_mask` (llama-vocab.cpp:4301 @c1d0e7a00).
pub fn llama_vocab_mask(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return vocabOf(vocab).special_mask_id;
}

/// Ports `llama_token_get_text` (llama-vocab.cpp:4306 @c1d0e7a00).
pub fn llama_token_get_text(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) [*c]const u8 {
    return llama_vocab_get_text(vocab, token);
}

/// Ports `llama_token_get_score` (llama-vocab.cpp:4311 @c1d0e7a00).
pub fn llama_token_get_score(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) f32 {
    return llama_vocab_get_score(vocab, token);
}

/// Ports `llama_token_get_attr` (llama-vocab.cpp:4316 @c1d0e7a00).
pub fn llama_token_get_attr(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) c.enum_llama_token_attr {
    return llama_vocab_get_attr(vocab, token);
}

/// Ports `llama_token_is_eog` (llama-vocab.cpp:4321 @c1d0e7a00).
pub fn llama_token_is_eog(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) bool {
    return llama_vocab_is_eog(vocab, token);
}

/// Ports `llama_token_is_control` (llama-vocab.cpp:4326 @c1d0e7a00).
pub fn llama_token_is_control(vocab: ?*const c.struct_llama_vocab, token: Token) callconv(.c) bool {
    return llama_vocab_is_control(vocab, token);
}

/// Ports `llama_token_bos` (llama-vocab.cpp:4331 @c1d0e7a00).
pub fn llama_token_bos(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_bos(vocab);
}

/// Ports `llama_token_eos` (llama-vocab.cpp:4336 @c1d0e7a00).
pub fn llama_token_eos(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_eos(vocab);
}

/// Ports `llama_token_eot` (llama-vocab.cpp:4341 @c1d0e7a00).
pub fn llama_token_eot(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_eot(vocab);
}

/// Ports `llama_token_cls` (llama-vocab.cpp:4346 @c1d0e7a00): BOS.
pub fn llama_token_cls(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_bos(vocab);
}

/// Ports `llama_token_sep` (llama-vocab.cpp:4352 @c1d0e7a00).
pub fn llama_token_sep(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_sep(vocab);
}

/// Ports `llama_token_nl` (llama-vocab.cpp:4357 @c1d0e7a00).
pub fn llama_token_nl(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_nl(vocab);
}

/// Ports `llama_token_pad` (llama-vocab.cpp:4362 @c1d0e7a00).
pub fn llama_token_pad(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_pad(vocab);
}

/// Ports `llama_add_bos_token` (llama-vocab.cpp:4367 @c1d0e7a00).
pub fn llama_add_bos_token(vocab: ?*const c.struct_llama_vocab) callconv(.c) bool {
    return llama_vocab_get_add_bos(vocab);
}

/// Ports `llama_add_eos_token` (llama-vocab.cpp:4372 @c1d0e7a00).
pub fn llama_add_eos_token(vocab: ?*const c.struct_llama_vocab) callconv(.c) bool {
    return llama_vocab_get_add_eos(vocab);
}

/// Ports `llama_token_fim_pre` (llama-vocab.cpp:4377 @c1d0e7a00).
pub fn llama_token_fim_pre(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_fim_pre(vocab);
}

/// Ports `llama_token_fim_suf` (llama-vocab.cpp:4382 @c1d0e7a00).
pub fn llama_token_fim_suf(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_fim_suf(vocab);
}

/// Ports `llama_token_fim_mid` (llama-vocab.cpp:4387 @c1d0e7a00).
pub fn llama_token_fim_mid(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_fim_mid(vocab);
}

/// Ports `llama_token_fim_pad` (llama-vocab.cpp:4392 @c1d0e7a00).
pub fn llama_token_fim_pad(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_fim_pad(vocab);
}

/// Ports `llama_token_fim_rep` (llama-vocab.cpp:4397 @c1d0e7a00).
pub fn llama_token_fim_rep(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_fim_rep(vocab);
}

/// Ports `llama_token_fim_sep` (llama-vocab.cpp:4402 @c1d0e7a00).
pub fn llama_token_fim_sep(vocab: ?*const c.struct_llama_vocab) callconv(.c) Token {
    return llama_vocab_fim_sep(vocab);
}

/// Ports `llama_tokenize` (llama-vocab.cpp:4410 @c1d0e7a00), through
/// `llama_vocab`'s C-buffer `tokenize` (llama-vocab.cpp:4121 @c1d0e7a00).
///
/// Return: the number of tokens written; minus the number needed when
/// `n_tokens_max` is too small; `INT32_MIN` past `int32_t`.
pub fn llama_tokenize(
    vocab: ?*const c.struct_llama_vocab,
    text: [*c]const u8,
    text_len: i32,
    tokens: [*c]Token,
    n_tokens_max: i32,
    add_special: bool,
    parse_special: bool,
) callconv(.c) i32 {
    var exc: impl.Exception = .{};
    // `std::string(text, text_len)`: a negative length is a huge `size_t`.
    if (text_len < 0) {
        exc.set(.length_error, "basic_string", .{});
        exc.terminate("llama-vocab.cpp", 4128);
    }
    const t: []const u8 = if (text_len == 0) "" else text[0..@intCast(text_len)];
    return vocabOf(vocab).tokenizeInto(t, cSlice(Token, tokens, n_tokens_max), add_special, parse_special, &exc) catch
        exc.terminate("llama-vocab.cpp", 4418);
}

/// Ports `llama_token_to_piece` (llama-vocab.cpp:4421 @c1d0e7a00). Writes
/// no NUL.
///
/// Return: the bytes written, or minus the bytes needed.
pub fn llama_token_to_piece(
    vocab: ?*const c.struct_llama_vocab,
    token: Token,
    buf: [*c]u8,
    length: i32,
    lstrip: i32,
    special: bool,
) callconv(.c) i32 {
    var exc: impl.Exception = .{};
    return vocabOf(vocab).tokenToPiece(token, cSlice(u8, buf, length), lstrip, special, &exc) catch
        exc.terminate("llama-vocab.cpp", 4428);
}

/// Ports `llama_detokenize` (llama-vocab.cpp:4431 @c1d0e7a00). Writes no
/// NUL.
///
/// A negative `text_len_max` is the C++'s `avail` starting negative: its
/// `GGML_ASSERT(avail >= 0)` fires on the first token left once
/// `remove_special` has dropped the BOS and EOS, and with none left the
/// answer is 0.
///
/// Return: the bytes written, or minus the bytes needed.
pub fn llama_detokenize(
    vocab: ?*const c.struct_llama_vocab,
    tokens: [*c]const Token,
    n_tokens: i32,
    text: [*c]u8,
    text_len_max: i32,
    remove_special: bool,
    unparse_special: bool,
) callconv(.c) i32 {
    const v = vocabOf(vocab);
    const toks: []const Token = if (n_tokens <= 0) &.{} else tokens[0..@intCast(n_tokens)];
    if (text_len_max < 0) {
        if (v.type == vm.vocab.type_none) return 0;
        impl.assert(v.tokenizer != .none, "llama-vocab.cpp", 3712, "tokenizer && \"Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.\"");
        var rest = toks;
        if (remove_special and v.add_bos and rest.len > 0 and rest[0] == v.special_bos_id) rest = rest[1..];
        if (remove_special and v.add_eos and rest.len > 0 and rest[rest.len - 1] == v.special_eos_id) rest = rest[0 .. rest.len - 1];
        impl.assert(rest.len == 0, "llama-vocab.cpp", 3735, "avail >= 0");
        return 0;
    }
    var exc: impl.Exception = .{};
    return v.detokenize(toks, cSlice(u8, text, text_len_max), remove_special, unparse_special, &exc) catch
        exc.terminate("llama-vocab.cpp", 4439);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
