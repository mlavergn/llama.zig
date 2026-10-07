// llama.cpp's unicode.cpp behind a C ABI, for scripts/unicode-diff.
//
// The reference half of the oracle for `src/llama/unicode.zig`. The
// script compiles `llama.cpp/src/unicode.cpp` and `unicode-data.cpp`
// unmodified alongside this file, against the SDK's libc++ -- the same
// `std::regex` the shipped library runs -- and links the lot into
// `harness/unicode_diff.zig`, which calls the Zig port on the same input.
// Every function here is a thin call into `unicode.h` that turns a C++
// result or exception into plain data. Nothing else.

#include "unicode.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

uint16_t ref_uni_flags(uint32_t cpt) {
    return unicode_cpt_flags_from_cpt(cpt).as_uint();
}

uint32_t ref_uni_tolower(uint32_t cpt) {
    return unicode_tolower(cpt);
}

int ref_uni_is_han(uint32_t cpt) {
    return unicode_cpt_is_han(cpt) ? 1 : 0;
}

size_t ref_uni_len_utf8(char c) {
    return unicode_len_utf8(c);
}

void ref_uni_nfd(const uint32_t * in, size_t n, uint32_t * out) {
    std::vector<uint32_t> v(in, in + n);
    std::vector<uint32_t> r = unicode_cpts_normalize_nfd(v);
    std::memcpy(out, r.data(), n * sizeof(uint32_t));
}

// Return: the length written to `out`, or -1 for std::invalid_argument.
int ref_uni_cpt_to_utf8(uint32_t cpt, char * out) {
    try {
        std::string s = unicode_cpt_to_utf8(cpt);
        std::memcpy(out, s.data(), s.size());
        return (int) s.size();
    } catch (const std::invalid_argument &) {
        return -1;
    }
}

// Return: the number of code points; `out` must hold `n`.
size_t ref_uni_cpts_from_utf8(const char * s, size_t n, uint32_t * out) {
    std::vector<uint32_t> r = unicode_cpts_from_utf8(std::string(s, n));
    std::memcpy(out, r.data(), r.size() * sizeof(uint32_t));
    return r.size();
}

// Decodes at `*offset`. Return: 0 with `*cpt` set and `*offset` advanced,
// or -1 for std::invalid_argument with `*offset` as the C++ left it.
int ref_uni_cpt_from_utf8(const char * s, size_t n, size_t * offset, uint32_t * cpt) {
    std::string str(s, n);
    size_t off = *offset;
    try {
        *cpt = unicode_cpt_from_utf8(str, off);
        *offset = off;
        return 0;
    } catch (const std::invalid_argument &) {
        *offset = off;
        return -1;
    }
}

// Return: the flags, or -1 for std::invalid_argument.
int ref_uni_flags_from_utf8(const char * s, size_t n) {
    try {
        return unicode_cpt_flags_from_utf8(std::string(s, n)).as_uint();
    } catch (const std::invalid_argument &) {
        return -1;
    }
}

int ref_uni_byte_to_utf8(uint8_t b, char * out) {
    std::string s = unicode_byte_to_utf8(b);
    std::memcpy(out, s.data(), s.size());
    return (int) s.size();
}

// Return: the byte, or -1 for std::out_of_range.
int ref_uni_utf8_to_byte(const char * s, size_t n) {
    try {
        return unicode_utf8_to_byte(std::string(s, n));
    } catch (const std::out_of_range &) {
        return -1;
    }
}

// Runs unicode_regex_split. On success `*out` is a malloc'd buffer of
// pieces, each a native-endian uint32 length and its bytes, and `*out_len`
// its size.
//
// Return: the number of pieces, or -1 for runtime_error("Failed to process
// regex"), -2 for any other runtime_error, -3 for invalid_argument, -4 for
// anything else.
long ref_uni_regex_split(const char * text, size_t text_len, const char * const * exprs, const size_t * expr_lens,
                         size_t n_exprs, int byte_encode, char ** out, size_t * out_len) {
    std::vector<std::string> ex;
    for (size_t i = 0; i < n_exprs; i++) ex.emplace_back(exprs[i], expr_lens[i]);
    std::vector<std::string> words;
    try {
        words = unicode_regex_split(std::string(text, text_len), ex, byte_encode != 0);
    } catch (const std::runtime_error & e) {
        return std::strcmp(e.what(), "Failed to process regex") == 0 ? -1 : -2;
    } catch (const std::invalid_argument &) {
        return -3;
    } catch (...) {
        return -4;
    }
    size_t total = 0;
    for (const auto & w : words) total += 4 + w.size();
    char * buf = (char *) std::malloc(total ? total : 1);
    size_t at = 0;
    for (const auto & w : words) {
        uint32_t len = (uint32_t) w.size();
        std::memcpy(buf + at, &len, 4);
        std::memcpy(buf + at + 4, w.data(), w.size());
        at += 4 + w.size();
    }
    *out = buf;
    *out_len = total;
    return (long) words.size();
}

void ref_uni_free(char * p) {
    std::free(p);
}

}
