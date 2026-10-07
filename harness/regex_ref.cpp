// libc++'s std::regex behind a C ABI, for scripts/regex-diff.
//
// The reference half of the oracle for `src/llama/regex.zig`: compiled
// against the SDK's `c++/v1` headers and libc++ dylib -- what
// `llama-grammar.cpp` and `unicode.cpp` use -- and linked into
// `harness/regex_diff.zig`, which runs the Zig engine on the same pattern
// and input and compares. It must stay a direct use of the standard
// library and nothing more.

#include <cstddef>
#include <cstdint>
#include <regex>
#include <string>

namespace {

template <typename CharT>
std::basic_string<CharT> str(const void * p, size_t n) {
    const CharT * s = (const CharT *) p;
    return std::basic_string<CharT>(s, s + n);
}

std::wstring wstr(const void * p, size_t n) {
    const uint32_t * s = (const uint32_t *) p;
    std::wstring w;
    w.reserve(n);
    for (size_t i = 0; i < n; i++) w.push_back((wchar_t) s[i]);
    return w;
}

template <typename CharT>
int run(const std::basic_string<CharT> & pat, const std::basic_string<CharT> & input, int nosubs, int mode,
        size_t begin, size_t end, long * out, int max_out) {
    using Re = std::basic_regex<CharT>;
    using It = typename std::basic_string<CharT>::const_iterator;
    Re re;
    try {
        auto flags = std::regex_constants::ECMAScript;
        if (nosubs) flags = std::regex_constants::optimize | std::regex_constants::nosubs;
        re = Re(pat, flags);
    } catch (const std::regex_error & e) {
        return -(int) e.code();
    }
    try {
        if (mode == 2) {
            // regex_iterator over [begin, end), as unicode_regex_split_stl.
            std::regex_iterator<It> it(input.begin() + begin, input.begin() + end, re);
            std::regex_iterator<It> stop;
            int n = 0;
            while (it != stop && n < max_out) {
                std::match_results<It> m = *it;
                out[2 * n] = (long) m.position();
                out[2 * n + 1] = (long) m.length();
                n++;
                ++it;
            }
            return 1 + n;
        }
        std::match_results<It> m;
        bool ok = mode == 0 ? std::regex_search(input, m, re) : std::regex_match(input, m, re);
        if (!ok) return 0;
        int n = (int) m.size();
        for (int i = 0; i < n && i < max_out; i++) {
            out[2 * i] = (long) m.position(i);
            out[2 * i + 1] = m[i].matched ? (long) m.length(i) : -1;
        }
        return 1 + n;
    } catch (const std::regex_error & e) {
        return -100 - (int) e.code();
    }
}

}

extern "C" {

// Compiles `pat`, then runs regex_search (mode 0) or regex_match (mode 1)
// over `in`.
//
// Return: -code for a regex_error at construction, -100-code for one while
// matching, 0 for no match, or 1 + the number of sub-matches; `out` gets
// (position, length) per sub-match, length -1 when it did not participate.
int ref_regex(const char * pat, size_t plen, const char * in, size_t ilen, int mode, long * out, int max_groups) {
    return run<char>(str<char>(pat, plen), str<char>(in, ilen), 0, mode, 0, ilen, out, max_groups);
}

// The general form. `wide` selects std::wregex, with `pat` and `in` as
// arrays of uint32 code points; `nosubs` compiles with `optimize | nosubs`;
// mode 2 walks a regex_iterator over `in[begin, end)` and returns 1 + the
// number of matches, `out` getting (position, length) for each.
int ref_regex2(const void * pat, size_t plen, const void * in, size_t ilen, int wide, int nosubs, int mode,
               size_t begin, size_t end, long * out, int max_out) {
    if (wide) return run<wchar_t>(wstr(pat, plen), wstr(in, ilen), nosubs, mode, begin, end, out, max_out);
    return run<char>(str<char>(pat, plen), str<char>(in, ilen), nosubs, mode, begin, end, out, max_out);
}

// For every code point c below `n`, whether the wregex `pat` matches the
// one-character string c in full: one byte per code point into `out`.
// Return: 0, or -code if the pattern does not compile.
int ref_wsweep(const uint32_t * pat, size_t plen, uint32_t n, uint8_t * out) {
    std::wregex re;
    try {
        re = std::wregex(wstr(pat, plen));
    } catch (const std::regex_error & e) {
        return -(int) e.code();
    }
    std::wstring s(1, L'\0');
    for (uint32_t c = 0; c < n; c++) {
        s[0] = (wchar_t) c;
        out[c] = std::regex_match(s, re) ? 1 : 0;
    }
    return 0;
}

// For every code point c below `n`, the error code (0 for none) of
// compiling the wregex `pre` c `post`: what a character does in a position
// the parser reads through the traits -- a brace count, a hex digit, a
// class or collating name.
void ref_wcompile_sweep(const uint32_t * pre, size_t prelen, const uint32_t * post, size_t postlen, uint32_t n,
                        uint8_t * out) {
    std::wstring a = wstr(pre, prelen), b = wstr(post, postlen);
    for (uint32_t c = 0; c < n; c++) {
        std::wstring p = a;
        p.push_back((wchar_t) c);
        p += b;
        try {
            std::wregex re(p);
            out[c] = 0;
        } catch (const std::regex_error & e) {
            out[c] = (uint8_t) e.code();
        }
    }
}

}
