//! The byte-level BPE tokenizer: pre-tokenize with the model's patterns,
//! then merge each piece by rank.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` at v0.8.0 (`8642541b2`): the
//! tokenizer and its pattern table, the session, and the two sessions that
//! extend it, `hybriddna` and `whitespace`.
//!
//! The pattern table below is **generated from the C++ literals** -- every
//! `regex_exprs = { ... }` in `llm_tokenizer_bpe`'s constructor, byte for
//! byte -- and `scripts/unicode-diff` extracts the same literals at run time
//! and checks `unicode_regex_split` on each. Regenerate rather than edit.
//!
//! The work queue's comparator orders on rank and then `left`, so ties are
//! only between stale entries, which the text check discards whichever way
//! they come out; it is `libcxx.PriorityQueue` anyway, so the heap is
//! libc++'s.

const std = @import("std");
const impl = @import("../impl.zig");
const libcxx = @import("../libcxx.zig");
const unicode = @import("../unicode.zig");
const vm = @import("module.zig");

const allocator = impl.allocator;
const Vocab = vm.Vocab;
const Token = vm.Token;
const token_null = vm.vocab.token_null;
const PreType = vm.vocab.PreType;

fn oom() noreturn {
    impl.abort("llama-vocab.cpp", 0, "out of memory");
}

/// Ports `dna_kmer_marker` (llama-vocab.cpp:1820 @8642541b2): U+E000.
pub const dna_kmer_marker = "\xee\x80\x80";

/// The hybriddna session's constants: `open_tag`, `close_tag` and `k`
/// (llama-vocab.cpp:2010, 2011, 2012 @8642541b2).
const dna_open_tag = "<dna>";
const dna_close_tag = "</dna>";
const dna_k = 6;

const Patterns = struct {
    exprs: []const []const u8,
    byte_encode: bool,
};

/// Ports `llm_tokenizer_bpe`'s constructor's `switch` (llama-vocab.cpp:284
/// @8642541b2), generated.
fn patterns(pre: PreType) Patterns {
    return switch (pre) {
        .flint => .{ .exprs = &.{
            "\\p{N}{1,3}",
            "[\xe4\xb8\x80-\xe9\xbe\xa5\xe3\x81\x80-\xe3\x82\x9f\xe3\x82\xa0-\xe3\x83\xbf]+",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .llama3 => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .jais2 => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",
        }, .byte_encode = true },
        .dbrx, .smaug => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .deepseek_llm => .{ .exprs = &.{
            "[\r\n]",
            "\\s?[A-Za-z\xc2\xb5\xc3\x80-\xc3\x96\xc3\x98-\xc3\xb6\xc3\xb8-\xc6\xba\xc6\xbc-\xc6\xbf\xc7\x84-\xca\x93\xca\x95-\xca\xaf\xcd\xb0-\xcd\xb3\xcd\xb6\xcd\xb7\xcd\xbb-\xcd\xbd\xcd\xbf\xce\x86\xce\x88-\xce\x8a\xce\x8c\xce\x8e-\xce\xa1\xce\xa3-\xcf\xb5\xcf\xb7-\xd2\x81\xd2\x8a-\xd4\xaf\xd4\xb1-\xd5\x96\xe1\x82\xa0-\xe1\x83\x85\xe1\x8e\xa0-\xe1\x8f\xb5\xe1\x8f\xb8-\xe1\x8f\xbd\xe1\xb2\x90-\xe1\xb2\xba\xe1\xb2\xbd-\xe1\xb2\xbf\xe1\xb4\x80-\xe1\xb4\xab\xe1\xb5\xab-\xe1\xb5\xb7\xe1\xb5\xb9-\xe1\xb6\x9a\xe1\xb8\x80-\xe1\xbc\x95\xe1\xbc\x98-\xe1\xbc\x9d\xe1\xbc\xa0-\xe1\xbd\x85\xe1\xbd\x88-\xe1\xbd\x8d\xe1\xbd\x90-\xe1\xbd\x97\xe1\xbd\x99\xe1\xbd\x9b\xe1\xbd\x9d\xe1\xbd\x9f-\xe1\xbd\xbd\xe1\xbe\x80-\xe1\xbe\xb4\xe1\xbe\xb6-\xe1\xbe\xbc\xe1\xbe\xbe\xe1\xbf\x82-\xe1\xbf\x84\xe1\xbf\x86-\xe1\xbf\x8c\xe1\xbf\x90-\xe1\xbf\x93\xe1\xbf\x96-\xe1\xbf\x9b\xe1\xbf\xa0-\xe1\xbf\xac\xe1\xbf\xb2-\xe1\xbf\xb4\xe1\xbf\xb6-\xe1\xbf\xbc\xe2\x84\x82\xe2\x84\x87\xe2\x84\x8a-\xe2\x84\x93\xe2\x84\x95\xe2\x84\x99-\xe2\x84\x9d\xe2\x84\xa4\xe2\x84\xa6\xe2\x84\xa8\xe2\x84\xaa-\xe2\x84\xad\xe2\x84\xaf-\xe2\x84\xb4\xe2\x84\xb9\xe2\x84\xbc-\xe2\x84\xbf\xe2\x85\x85-\xe2\x85\x89\xe2\x85\x8e\xe2\x86\x83\xe2\x86\x84\xe2\xb0\x80-\xe2\xb1\xbb\xe2\xb1\xbe-\xe2\xb3\xa4\xe2\xb3\xab-\xe2\xb3\xae\xe2\xb3\xb2\xe2\xb3\xb3\xea\x99\x80-\xea\x99\xad\xea\x9a\x80-\xea\x9a\x9b\xea\x9c\xa2-\xea\x9d\xaf\xea\x9d\xb1-\xea\x9e\x87\xea\x9e\x8b-\xea\x9e\x8e\xea\xad\xb0-\xea\xae\xbf\xef\xac\x80-\xef\xac\x86\xef\xac\x93-\xef\xac\x97\xef\xbc\xa1-\xef\xbc\xba\xef\xbd\x81-\xef\xbd\x9a\xf0\x90\x90\x80-\xf0\x90\x91\x8f\xf0\x90\x92\xb0-\xf0\x90\x93\x93\xf0\x90\x93\x98-\xf0\x90\x93\xbb\xf0\x90\xb2\x80-\xf0\x90\xb2\xb2\xf0\x90\xb3\x80-\xf0\x90\xb3\xb2\xf0\x91\xa2\xa0-\xf0\x91\xa3\x9f\xf0\x9e\xa4\x80-\xf0\x9e\xa5\x83]+",
            "\\s?[!-/:-~\xef\xbc\x81-\xef\xbc\x8f\xef\xbc\x9a-\xef\xbd\x9e\xe2\x80\x98-\xe2\x80\x9f\xe3\x80\x80-\xe3\x80\x82]+",
            "\\s+$",
            "[\xe4\xb8\x80-\xe9\xbe\xa5\xe0\xa0\x80-\xe4\xb8\x80\xea\xb0\x80-\xed\x9f\xbf]+",
            "\\p{N}+",
        }, .byte_encode = true },
        .deepseek3_llm, .hunyuan_dense, .joyai_llm, .hy_v4 => .{ .exprs = &.{
            "\\p{N}{1,3}",
            "[\xe4\xb8\x80-\xe9\xbe\xa5\xe3\x81\x80-\xe3\x82\x9f\xe3\x82\xa0-\xe3\x83\xbf]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .spark2_5 => .{ .exprs = &.{
            "\\p{N}{1,3}",
            "[\xe4\xb8\x80-\xe9\xbe\xa5\xe3\x81\x80-\xe3\x82\x9f\xe3\x82\xa0-\xe3\x83\xbf]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
            "\\p{N}",
        }, .byte_encode = true },
        .youtu => .{ .exprs = &.{
            "[\xea\xb0\x80-\xed\x9e\xa3\xe3\x84\xb1-\xe3\x86\x8e]+|[\xef\xbc\x81\xe2\x80\xa6\xe2\x80\x9c\xe2\x80\x9d\xe2\x80\x98\xe2\x80\x99\xe2\x80\x94\xef\xbc\x9a\xef\xbc\x9b\xef\xbc\x8c\xe3\x80\x81-\xe3\x80\xbf\xef\xb8\xb0-\xef\xb9\x8f]+|[\xe3\x84\x85-\xe3\x84\xaf]+|[\xe4\xb8\x80-\xe9\xbe\xa5\xe3\x81\x80-\xe3\x82\x9f\xe3\x82\xa0-\xe3\x83\xbf]+",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .deepseek_coder => .{ .exprs = &.{
            "[\r\n]",
            "\\s?\\p{L}+",
            "\\s?\\p{P}+",
            "[\xe4\xb8\x80-\xe9\xbe\xa5\xe0\xa0\x80-\xe4\xb8\x80\xea\xb0\x80-\xed\x9f\xbf]+",
            "\\p{N}",
        }, .byte_encode = true },
        .falcon => .{ .exprs = &.{
            "[\\p{P}\\$\\+<=>\\^~\\|`]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "[0-9][0-9][0-9]",
        }, .byte_encode = true },
        .starcoder, .refact, .command_r, .smollm, .codeshell, .exaone, .minerva, .mellum2 => .{ .exprs = &.{
            "\\p{N}",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        }, .byte_encode = true },
        .gpt2, .mpt, .olmo, .jais, .trillion, .granite_docling => .{ .exprs = &.{
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        }, .byte_encode = true },
        .stablelm2, .qwen2, .hunyuan, .solar_open => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .qwen35 => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .poro, .bloom, .gpt3_finnish => .{ .exprs = &.{
            " ?[^(\\s|.,!?\xe2\x80\xa6\xe3\x80\x82\xef\xbc\x8c\xe3\x80\x81\xe0\xa5\xa4\xdb\x94\xd8\x8c)]+",
        }, .byte_encode = true },
        .chatglm4 => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .viking => .{ .exprs = &.{
            " ?[^(\\s|.,!?\xe2\x80\xa6\xe3\x80\x82\xef\xbc\x8c\xe3\x80\x81\xe0\xa5\xa4\xdb\x94\xd8\x8c)]+",
            "\\p{N}",
        }, .byte_encode = true },
        .tekken, .nemotron_3_5_embd => .{ .exprs = &.{
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .chameleon => .{ .exprs = &.{
            "<sentinel:[0-9]+>",
            "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
            "([\\t\\n]|    |  )",
            "\\p{N}",
            "[\\p{P}!-/:-@\\[-`{-~]",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        }, .byte_encode = true },
        .gpt4o, .minimax_m2 => .{ .exprs = &.{
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .granite_emb_multi => .{ .exprs = &.{
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .tiny_aya => .{ .exprs = &.{
            "\\d{1,3}(?=(?:\\d{3})*\\b)",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .kimi_k2 => .{ .exprs = &.{
            "\\p{Han}+",
        }, .byte_encode = true },
        .superbpe => .{ .exprs = &.{
            "\\p{N}+",
            "(?=(\\d{3})+(?!\\d))",
        }, .byte_encode = true },
        .bailingmoe => .{ .exprs = &.{
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .seed_coder => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .ufakzeka => .{ .exprs = &.{
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .grok_2 => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .afmoe => .{ .exprs = &.{
            "\\p{AFMoE_digits}",
            "[\xe4\xb8\x80-\xe9\xbf\xbf\xe3\x90\x80-\xe4\xb6\xbf\xe8\xb1\x88-\xef\xab\xbf\xe3\x81\x80-\xe3\x82\x9f\xe3\x82\xa0-\xe3\x83\xbf\xef\xbd\xa5-\xef\xbe\x9f\xe2\xbc\x80-\xe2\xbf\x9f\xe0\xb9\x80-\xe0\xb9\xbf\xe0\xba\x80-\xe0\xbb\xbf\xe1\x9e\x80-\xe1\x9f\xbf\xe1\x80\x80-\xe1\x82\x9f\xea\xa9\xa0-\xea\xa9\xbf\xea\xa7\xa0-\xea\xa7\xbf\xea\xb0\x80-\xed\x9e\xaf\xe1\x84\x80-\xe1\x87\xbf]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .laguna => .{ .exprs = &.{
            "[^\\n]+|[\\n]+",
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .exaone_moe => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .gemma4 => .{ .exprs = &.{
            "[^\\n]+|[\\n]+",
        }, .byte_encode = false },
        // as Gemma 4's; `tokenize` splits the words
        .mmbert => .{ .exprs = &.{
            "[^\\n]+|[\\n]+",
        }, .byte_encode = false },
        .sarvam_moe, .domyn_small => .{ .exprs = &.{
            "[^\\n]+|[\\n]+",
        }, .byte_encode = false },
        .minicpm5 => .{ .exprs = &.{
            "\\p{N}{1,3}",
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .k2_horizon => .{ .exprs = &.{
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        }, .byte_encode = true },
        .soprano => .{ .exprs = &.{
            "\\p{N}",
            "\\s+|[\\p{L}\\p{N}_]+|[^\\p{L}\\p{N}_\\s]+",
        }, .byte_encode = false },
        .whitespace => .{ .exprs = &.{
            "\\S+",
        }, .byte_encode = false },
        else => .{ .exprs = &.{
            "[\\p{P}\\$\\+<=>\\^~\\|]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "\\p{N}+",
            "[0-9][0-9][0-9]",
        }, .byte_encode = true },
    };
}

/// Ports `llm_tokenizer_bpe` (llama-vocab.cpp:281 @8642541b2).
pub const Tokenizer = struct {
    regex_exprs: []const []const u8,
    /// GPT-2 byte encoding; false for SPM-style BPE over raw UTF-8.
    byte_encode: bool,

    /// Return: the tokenizer for `vocab`'s pre-tokenizer.
    pub fn init(vocab: *const Vocab) Tokenizer {
        impl.assert(vocab.type == vm.vocab.type_bpe, "llama-vocab.cpp", 283, "vocab.get_type() == LLAMA_VOCAB_TYPE_BPE");
        const p = patterns(vocab.pre_type);
        return .{ .regex_exprs = p.exprs, .byte_encode = p.byte_encode };
    }

    /// Return: nothing; the patterns are static.
    pub fn deinit(_: *Tokenizer) void {}
};

/// Ports `llm_bigram_bpe` (llama-vocab.cpp:265 @8642541b2). `text` is a
/// span of the word, which the two symbols are adjacent parts of.
const Bigram = struct {
    left: i32,
    right: i32,
    text: []const u8,
    rank: i32,
    size: usize,
};

/// `llm_bigram_bpe`'s `comparator`.
fn bigramLess(_: void, l: Bigram, r: Bigram) bool {
    return l.rank > r.rank or (l.rank == r.rank and l.left > r.left);
}

/// `llm_symbol` over a word: `text` is a slice of it.
const Symbol = struct {
    prev: i32,
    next: i32,
    text: []const u8,
};

/// Which session `llama_vocab::impl`'s `tokenize` builds.
const Mode = enum { plain, hybriddna, whitespace };

/// Ports `llm_tokenizer_bpe_session` (llama-vocab.cpp:603 @8642541b2),
/// and through `mode` the two sessions derived from it.
pub const Session = struct {
    vocab: *const Vocab,
    tokenizer: *const Tokenizer,
    mode: Mode,
    symbols: std.ArrayList(Symbol) = .empty,
    symbols_final: std.ArrayList(Symbol) = .empty,
    work_queue: libcxx.PriorityQueue(Bigram, void, bigramLess) = .{ .ctx = {} },

    // `llm_tokenizer_hybriddna_session`'s members.
    dna_begin_id: Token = token_null,
    dna_end_id: Token = token_null,
    dna_oov_id: Token = token_null,
    pad_char: u8 = 'A',

    /// The session `llama_vocab::impl`'s `tokenize` makes for `vocab`'s
    /// tokenizer model (llama-vocab.cpp:3870 @8642541b2).
    ///
    /// Return: the session; free it with `deinit`.
    pub fn init(vocab: *const Vocab, tokenizer: *const Tokenizer) Session {
        const model = vocab.tokenizer_model.items;
        const mode: Mode = if (std.mem.eql(u8, model, "hybriddna")) .hybriddna else if (std.mem.eql(u8, model, "whitespace")) .whitespace else .plain;
        return .{ .vocab = vocab, .tokenizer = tokenizer, .mode = mode };
    }

    /// Return: nothing.
    pub fn deinit(self: *Session) void {
        self.symbols.deinit(allocator);
        self.symbols_final.deinit(allocator);
        self.work_queue.deinit(allocator);
    }

    /// Ports `llm_tokenizer_bpe_session`'s `append_bos`.
    pub fn appendBos(self: *const Session, output: *std.ArrayList(Token)) void {
        if (self.vocab.add_bos) {
            impl.assert(self.vocab.special_bos_id != token_null, "llama-vocab.cpp", 614, "vocab.token_bos() != LLAMA_TOKEN_NULL");
            output.append(allocator, self.vocab.special_bos_id) catch oom();
        }
    }

    /// Ports `llm_tokenizer_bpe_session`'s `append_eos`.
    pub fn appendEos(self: *const Session, output: *std.ArrayList(Token)) void {
        if (self.vocab.add_eos) {
            impl.assert(self.vocab.special_eos_id != token_null, "llama-vocab.cpp", 623, "vocab.token_eos() != LLAMA_TOKEN_NULL");
            output.append(allocator, self.vocab.special_eos_id) catch oom();
        }
    }

    /// Ports `llm_tokenizer_bpe_session`'s `check_double_bos_eos`.
    pub fn checkDoubleBosEos(self: *const Session, output: []const Token) void {
        if (self.vocab.add_bos and output.len >= 2 and output[1] == self.vocab.special_bos_id) {
            impl.logWarn("%s: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?\n", .{"check_double_bos_eos"});
        }
        if (self.vocab.add_eos and output.len >= 2 and output[output.len - 2] == self.vocab.special_eos_id) {
            impl.logWarn("%s: Added a EOS token to the prompt as specified by the model but the prompt also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. Are you sure this is what you want?\n", .{"check_double_bos_eos"});
        }
    }

    /// The session's `tokenize`, dispatched as the C++'s virtual is.
    ///
    /// Return: nothing, or `error.Thrown` from the pre-tokenizer.
    pub fn tokenize(self: *Session, text: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
        return switch (self.mode) {
            .plain => self.tokenizeBpe(text, output, exc),
            .hybriddna => self.tokenizeHybridDna(text, output, exc),
            .whitespace => self.tokenizeWhitespace(text, output, exc),
        };
    }

    /// Ports `llm_tokenizer_bpe_session`'s `add_new_bigram`.
    fn addNewBigram(self: *Session, left: i32, right: i32) void {
        if (left == -1 or right == -1) return;
        const lt = self.symbols.items[@intCast(left)].text;
        const rt = self.symbols.items[@intCast(right)].text;
        const rank_found = self.vocab.findBpeRank(lt, rt);
        if (rank_found < 0) return;
        // The symbols are adjacent in the word, so `left_token +
        // right_token` is one span of it.
        const t = lt.ptr[0 .. lt.len + rt.len];
        self.work_queue.push(allocator, .{ .left = left, .right = right, .text = t, .rank = rank_found, .size = lt.len + rt.len }) catch oom();
    }

    /// Ports `llm_tokenizer_bpe_session`'s `tokenize` (llama-vocab.cpp:645
    /// @8642541b2). Nemotron 3.5 Embed drops the zero-width non-joiner after
    /// `[` and `<`; Soprano folds whitespace runs to one space and lowercases.
    /// mmBERT's Metaspace pre-tokenizer: a text starts with an escaped space,
    /// and each escaped space starts a new word.
    fn tokenizeBpe(self: *Session, text: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
        const vocab = self.vocab;
        const tok_pre = vocab.pre_type;

        var normalized_text: std.ArrayList(u8) = .empty;
        defer normalized_text.deinit(allocator);
        var split_text = text;
        if (tok_pre == .nemotron_3_5_embd) {
            const zwnj = "\xe2\x80\x8c";
            normalized_text.appendSlice(allocator, text) catch oom();
            vm.vocab.replaceAll(&normalized_text, "[" ++ zwnj, "[");
            vm.vocab.replaceAll(&normalized_text, "<" ++ zwnj, "<");
            split_text = normalized_text.items;
        } else if (tok_pre == .soprano) {
            var space = false;
            const cpts = unicode.cptsFromUtf8(text);
            defer allocator.free(cpts);
            for (cpts) |cpt| {
                if (unicode.cptFlagsFromCpt(cpt).is_whitespace) {
                    if (!space) normalized_text.append(allocator, ' ') catch oom();
                    space = true;
                } else {
                    // Unicode full lowercase expands LATIN CAPITAL LETTER I
                    // WITH DOT ABOVE.
                    if (cpt == 0x0130) {
                        normalized_text.appendSlice(allocator, "i\xcc\x87") catch oom();
                    } else {
                        unicode.appendCptUtf8(&normalized_text, unicode.tolower(cpt)) catch
                            return exc.throw(.invalid_argument, "invalid codepoint", .{});
                    }
                    space = false;
                }
            }
            split_text = normalized_text.items;
        }

        var word_collection = unicode.regexSplit(split_text, self.tokenizer.regex_exprs, self.tokenizer.byte_encode) catch |err| return switch (err) {
            error.RegexFailed => exc.throw(.runtime_error, "Failed to process regex", .{}),
            error.CategoryWithNonAscii => exc.throw(.runtime_error, "Regex includes both unicode categories and non-ASCII characters - not supported", .{}),
            error.InvalidCodepoint => exc.throw(.invalid_argument, "invalid codepoint", .{}),
        };
        defer word_collection.deinit();

        var final_prev_index: i32 = -1;

        self.symbols_final.clearRetainingCapacity();

        if (tok_pre == .mmbert) {
            const space = "\xe2\x96\x81";
            var words: unicode.Words = .{};
            var tmp: std.ArrayList(u8) = .empty;
            defer tmp.deinit(allocator);
            for (0..word_collection.len()) |w| {
                const word0 = word_collection.get(w);
                if (std.mem.findNone(u8, word0, "\n") == null) {
                    words.buf.appendSlice(allocator, word0) catch oom();
                    words.ends.append(allocator, words.buf.items.len) catch oom();
                    continue;
                }
                tmp.clearRetainingCapacity();
                if (!std.mem.startsWith(u8, word0, space)) tmp.appendSlice(allocator, space) catch oom();
                tmp.appendSlice(allocator, word0) catch oom();
                const word = tmp.items;
                var start: usize = 0;
                while (start < word.len) {
                    const end = std.mem.findPos(u8, word, start + space.len, space) orelse word.len;
                    words.buf.appendSlice(allocator, word[start..end]) catch oom();
                    words.ends.append(allocator, words.buf.items.len) catch oom();
                    start = end;
                }
            }
            word_collection.deinit();
            word_collection = words;
        }

        for (0..word_collection.len()) |w| {
            const word = word_collection.get(w);
            self.work_queue.clear();
            self.symbols.clearRetainingCapacity();

            var index: i32 = 0;
            var offset: usize = 0;

            if (vocab.ignore_merges and vocab.textToToken(word) != token_null) {
                self.symbols.append(allocator, .{ .prev = -1, .next = -1, .text = word }) catch oom();
                offset = word.len;
            } else if ((tok_pre == .gemma4 or tok_pre == .mmbert) and std.mem.findNone(u8, word, "\n") == null) {
                // fix for gemma 4, ref: https://github.com/ggml-org/llama.cpp/pull/21343
                if (vocab.textToToken(word) != token_null) {
                    self.symbols.append(allocator, .{ .prev = -1, .next = -1, .text = word }) catch oom();
                    offset = word.len;
                }
            }

            while (offset < word.len) {
                const char_len = @min(word.len - offset, unicode.lenUtf8(word[offset]));
                const sym: Symbol = .{
                    .text = word[offset..][0..char_len],
                    .prev = index - 1,
                    .next = if (offset + char_len == word.len) -1 else index + 1,
                };
                offset += char_len;
                index += 1;
                self.symbols.append(allocator, sym) catch oom();
            }
            var i: i32 = 1;
            while (i < @as(i32, @intCast(self.symbols.items.len))) : (i += 1) self.addNewBigram(i - 1, i);

            // build token(s)
            while (!self.work_queue.empty()) {
                const bigram = self.work_queue.pop();
                const left_symbol = &self.symbols.items[@intCast(bigram.left)];
                const right_symbol = &self.symbols.items[@intCast(bigram.right)];

                if (left_symbol.text.len == 0 or right_symbol.text.len == 0) continue;
                const lt = left_symbol.text;
                const rt = right_symbol.text;
                if (lt.len + rt.len != bigram.text.len or
                    !std.mem.eql(u8, lt, bigram.text[0..lt.len]) or
                    !std.mem.eql(u8, rt, bigram.text[lt.len..]))
                {
                    continue; // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.text = lt.ptr[0 .. lt.len + rt.len];
                right_symbol.text = rt[0..0];

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) self.symbols.items[@intCast(right_symbol.next)].prev = bigram.left;

                const prev = left_symbol.prev;
                const next = left_symbol.next;
                self.addNewBigram(prev, bigram.left);
                self.addNewBigram(bigram.left, next);
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (self.symbols.items) |*sym| {
                if (sym.text.len > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) self.symbols_final.items[@intCast(final_prev_index)].next = @intCast(self.symbols_final.items.len);
                    self.symbols_final.append(allocator, sym.*) catch oom();
                    final_prev_index = @intCast(self.symbols_final.items.len - 1);
                }
            }
        }

        std.mem.swap(std.ArrayList(Symbol), &self.symbols, &self.symbols_final);
        self.symbols_final.clearRetainingCapacity();
        self.symbols_final.appendSlice(allocator, self.symbols.items) catch oom();

        if (self.symbols.items.len != 0) {
            var i: i32 = 0;
            while (i != -1) : (i = self.symbols.items[@intCast(i)].next) {
                const symbol = self.symbols.items[@intCast(i)];
                if (symbol.text.len == 0) continue;

                const str = symbol.text;
                const token = vocab.textToToken(str);

                if (token == token_null and tok_pre == .soprano) {
                    output.append(allocator, vocab.special_unk_id) catch oom();
                } else if (token == token_null) {
                    for (str) |ch| {
                        var token_multibyte: Token = token_null;
                        if (self.tokenizer.byte_encode) {
                            token_multibyte = vocab.textToToken(&[_]u8{ch});
                        } else {
                            // For non-byte-encoded BPE (e.g. gemma-4), byte tokens use <0xXX> format
                            const hex = "0123456789ABCDEF";
                            token_multibyte = vocab.textToToken(&[_]u8{ '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>' });
                        }
                        if (token_multibyte != token_null) output.append(allocator, token_multibyte) catch oom();
                    }
                } else {
                    output.append(allocator, token) catch oom();
                }
            }
        }
    }

    /// Ports `llm_tokenizer_hybriddna_session`'s `tokenize` (llama-vocab.cpp:1827
    /// @8642541b2), which mirrors HybridDNATokenizer.encode: text inside
    /// `<dna>...</dna>` is cut into fixed 6-mers, everything else goes
    /// through the base BPE, with optional tagging of untagged DNA
    /// (`auto_dna_tags`, Carbon-14B) and tail="pad_n" vocabularies, whose
    /// dedicated tokens pad a trailing partial k-mer with N instead of A.
    fn tokenizeHybridDna(self: *Session, text: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
        const vocab = self.vocab;

        self.dna_begin_id = vocab.textToToken(dna_open_tag);
        self.dna_end_id = vocab.textToToken(dna_close_tag);
        self.dna_oov_id = vocab.textToToken("<oov>");

        // Fall back to plain BPE if the DNA pieces aren't in the vocab.
        if (self.dna_begin_id == token_null or self.dna_end_id == token_null or self.dna_oov_id == token_null) {
            return self.tokenizeBpe(text, output, exc);
        }

        self.pad_char = if (vocab.textToToken("ANNNNN" ++ dna_kmer_marker) != token_null) 'N' else 'A';

        // split on explicit tags as HybridDNATokenizer._split_by_dna_tags
        // does; only the text outside them is auto-tagged
        const npos = std.math.maxInt(usize);
        var pos: usize = 0;
        while (pos < text.len) {
            const start = std.mem.findPos(u8, text, pos, dna_open_tag) orelse npos;
            const end = std.mem.findPos(u8, text, pos, dna_close_tag) orelse npos;

            if (start == npos and end == npos) {
                try self.tokenizeDnaText(text[pos..], output, exc);
                break;
            }
            if (start == npos or end < start) {
                // as in _split_by_dna_tags, text before a </dna> without a
                // matching <dna> is DNA (no begin token)
                self.tokenizeDna(text[pos..end], false, true, output);
                pos = end + dna_close_tag.len;
                continue;
            }

            try self.tokenizeDnaText(text[pos..start], output, exc);

            const content_start = start + dna_open_tag.len;
            if (end == npos) {
                // an unterminated <dna> runs to the end of the text (no end
                // token)
                self.tokenizeDna(text[content_start..], true, false, output);
                break;
            }
            self.tokenizeDna(text[content_start..end], true, true, output);
            pos = end + dna_close_tag.len;
        }
    }

    /// `is_letter`.
    fn isLetter(ch: u8) bool {
        return (ch >= 'A' and ch <= 'Z') or (ch >= 'a' and ch <= 'z');
    }

    /// `is_acgt`.
    fn isAcgt(ch: u8) bool {
        return ch == 'A' or ch == 'C' or ch == 'G' or ch == 'T';
    }

    /// `to_upper`.
    fn toUpper(ch: u8) u8 {
        return if (ch >= 'a' and ch <= 'z') ch - 32 else ch;
    }

    /// `is_iupac`: IUPAC nucleotide codes, either case
    /// (HybridDNATokenizer.IUPAC_RUN_RE).
    fn isIupac(ch: u8) bool {
        return std.mem.findScalar(u8, "ACGTRYSWKMBDHVN", toUpper(ch)) != null;
    }

    /// Ports `llm_tokenizer_hybriddna_session`'s `normalize_spaced_dna`, which
    /// is HybridDNATokenizer._normalize_spaced_dna: join displays like
    /// "ACGTA CGTAC ..." when they hold at least twelve bases (at least six
    /// chunks of 1-5 uppercase ACGT separated by spaces or tabs, not touching
    /// a letter).
    fn normalizeSpacedDna(text: []const u8, out: *std.ArrayList(u8)) void {
        const isBlank = struct {
            fn f(ch: u8) bool {
                return ch == ' ' or ch == '\t';
            }
        }.f;
        var seq: std.ArrayList(u8) = .empty;
        defer seq.deinit(allocator);
        var p: usize = 0;
        while (p < text.len) {
            if (isAcgt(text[p]) and (p == 0 or !isLetter(text[p - 1]))) {
                var match_end: ?usize = null;
                var chunk = p;
                var n_chunks: usize = 1;
                while (true) : (n_chunks += 1) {
                    var chunk_end = chunk;
                    while (chunk_end < text.len and chunk_end - chunk < 5 and isAcgt(text[chunk_end])) chunk_end += 1;
                    if (chunk_end == chunk) break;
                    if (n_chunks >= 6 and (chunk_end == text.len or !isLetter(text[chunk_end]))) {
                        match_end = chunk_end; // the regex is greedy, so the last valid end wins
                    }
                    var next = chunk_end;
                    while (next < text.len and isBlank(text[next])) next += 1;
                    if (next == chunk_end) break;
                    chunk = next;
                }
                if (match_end) |me| {
                    seq.clearRetainingCapacity();
                    for (text[p..me]) |ch| {
                        if (!isBlank(ch)) seq.append(allocator, ch) catch oom();
                    }
                    out.appendSlice(allocator, if (seq.items.len >= 12) seq.items else text[p..me]) catch oom();
                    p = me;
                    continue;
                }
            }
            out.append(allocator, text[p]) catch oom();
            p += 1;
        }
    }

    /// Ports `llm_tokenizer_hybriddna_session`'s `tokenize_text`: text outside
    /// explicit tags. With `auto_dna_tags`, tag DNA runs as
    /// HybridDNATokenizer._auto_tag_text_segment does (pure ACGT runs of at
    /// least 6 bases, or runs of at least 12 with at most 20% IUPAC ambiguity
    /// codes).
    fn tokenizeDnaText(self: *Session, raw: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
        if (!self.vocab.dna_auto_tags) {
            if (raw.len != 0) try self.tokenizeBpe(raw, output, exc);
            return;
        }

        var normalized: std.ArrayList(u8) = .empty;
        defer normalized.deinit(allocator);
        normalizeSpacedDna(raw, &normalized);
        const text = normalized.items;

        var prev: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            if (!isIupac(text[i])) {
                i += 1;
                continue;
            }
            var j = i;
            var n_ambiguous: usize = 0;
            while (j < text.len and isIupac(text[j])) : (j += 1) {
                n_ambiguous += @intFromBool(!isAcgt(toUpper(text[j])));
            }
            const len = j - i;
            if ((n_ambiguous == 0 and len >= 6) or (n_ambiguous > 0 and len >= 12 and 5 * n_ambiguous <= len)) {
                if (i > prev) try self.tokenizeBpe(text[prev..i], output, exc);
                self.tokenizeDna(text[i..j], true, true, output);
                prev = j;
            }
            i = j;
        }
        if (text.len > prev) try self.tokenizeBpe(text[prev..], output, exc);
    }

    /// Ports `llm_tokenizer_hybriddna_session`'s `tokenize_dna`, which is
    /// HybridDNATokenizer._process_dna_sequence: strip, uppercase, then one
    /// token per k bases. A k-mer with anything but ACGT is `<oov>`, and a
    /// trailing partial k-mer is right-padded with `pad_char`.
    fn tokenizeDna(self: *Session, content: []const u8, has_start: bool, has_end: bool, output: *std.ArrayList(Token)) void {
        if (has_start) output.append(allocator, self.dna_begin_id) catch oom();

        const cpts = unicode.cptsFromUtf8(content);
        defer allocator.free(cpts);
        var first: usize = 0;
        var last: usize = cpts.len;
        while (first < last and unicode.cptFlagsFromCpt(cpts[first]).is_whitespace) first += 1;
        while (last > first and unicode.cptFlagsFromCpt(cpts[last - 1]).is_whitespace) last -= 1;

        var i = first;
        while (i + dna_k <= last) : (i += dna_k) {
            output.append(allocator, self.kmerToken(cpts[i .. i + dna_k])) catch oom();
        }
        if (i < last) output.append(allocator, self.kmerToken(cpts[i..last])) catch oom();

        if (has_end) output.append(allocator, self.dna_end_id) catch oom();
    }

    /// `kmer_token`: the k-mer's token, carrying the reserved marker suffix so
    /// it never resolves to a base-vocab token.
    fn kmerToken(self: *const Session, kmer_cpts: []const u32) Token {
        var kmer: [dna_k + dna_kmer_marker.len]u8 = undefined;
        for (kmer_cpts, 0..) |cpt, i| {
            const ch: u32 = if (cpt < 128) toUpper(@intCast(cpt)) else cpt;
            if (ch > 127 or !isAcgt(@intCast(ch))) return self.dna_oov_id;
            kmer[i] = @intCast(ch);
        }
        @memset(kmer[kmer_cpts.len..dna_k], self.pad_char);
        kmer[dna_k..].* = dna_kmer_marker.*;
        const tok = self.vocab.textToToken(&kmer);
        return if (tok != token_null) tok else self.dna_oov_id;
    }

    /// Ports `llm_tokenizer_whitespace_session`'s `tokenize` (llama-vocab.cpp:2025
    /// @8642541b2): whitespace dropped, each run between lowercased if the
    /// normalizer says so, and BPE'd alone.
    fn tokenizeWhitespace(self: *Session, text: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
        const lowercase = self.vocab.normalizer_opts.lowercase;
        var segment: std.ArrayList(u8) = .empty;
        defer segment.deinit(allocator);

        const cpts = unicode.cptsFromUtf8(text);
        defer allocator.free(cpts);
        for (cpts) |cpt| {
            if (unicode.cptFlagsFromCpt(cpt).is_whitespace) {
                if (segment.items.len != 0) {
                    try self.tokenizeBpe(segment.items, output, exc);
                    segment.clearRetainingCapacity();
                }
            } else {
                unicode.appendCptUtf8(&segment, if (lowercase) unicode.tolower(cpt) else cpt) catch
                    return exc.throw(.invalid_argument, "invalid codepoint", .{});
            }
        }
        if (segment.items.len != 0) try self.tokenizeBpe(segment.items, output, exc);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
