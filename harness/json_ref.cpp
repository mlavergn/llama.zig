// Upstream's common_json behind a C ABI, for scripts/json-diff.
//
// The reference half of the oracle for `cli/json/`: `common/json.cpp` is
// compiled unmodified against the vendored nlohmann (`vendor/nlohmann/
// json.hpp`) and linked beside this file. `ref_run` interprets a program --
// the same bytecode `harness/json_diff.zig` runs through the Zig port --
// over eight `common_json` registers and returns one record per operation:
// a dump, a query's answer, or the `what()` of whatever was thrown.
//
// It must stay a direct use of `common_json`. The one exception is
// `DUMP_RAW`, which calls nlohmann's own `dump` with the `ensure_ascii` and
// `error_handler` arguments `common_json` does not expose, reaching the
// `ordered_json` inside a `common_json` the way `json.cpp` itself does.

#include "json.h"
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// `json.cpp` defines JSON_ASSERT as GGML_ASSERT; nothing else of ggml is linked.
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}

namespace {

enum op : uint8_t {
    OP_PARSE = 0, OP_PARSE_NT, OP_SET_NULL, OP_SET_BOOL, OP_SET_INT, OP_SET_UINT, OP_SET_DOUBLE, OP_SET_STRING,
    OP_SET_ARRAY, OP_SET_OBJECT, OP_DUMP, OP_DUMP_SAFE, OP_DUMP_RAW, OP_QUERY, OP_GET, OP_AT_KEY, OP_AT_IDX,
    OP_INDEX_KEY, OP_INDEX_IDX, OP_CONTAINS, OP_VALUE, OP_FRONT, OP_BACK, OP_CLEAR, OP_ERASE_KEY, OP_ERASE_IDX,
    OP_PUSH, OP_INSERT, OP_SET_ITEM, OP_ASSIGN, OP_EQ, OP_ITER, OP_SET_FLOAT, OP_PUSH_OBJ, OP_CONST_INDEX,
};

struct reader {
    const uint8_t * p;
    const uint8_t * end;
    uint8_t  u8()  { return *p++; }
    uint32_t u32() { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
    uint64_t u64() { uint64_t v; memcpy(&v, p, 8); p += 8; return v; }
    std::string str() { uint32_t n = u32(); std::string s((const char *) p, n); p += n; return s; }
};

struct writer {
    std::string out;
    void rec(const std::string & s) {
        uint32_t n = (uint32_t) s.size();
        out.append((const char *) &n, 4);
        out.append(s);
    }
};

std::string hex64(uint64_t v) {
    char b[32];
    snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}
std::string fbits(double d) { uint64_t u; memcpy(&u, &d, 8); return "d" + hex64(u); }
std::string fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return "f" + hex64(u); }

template <typename T> std::string show(const T & v) { return std::to_string(v); }
template <> std::string show(const bool & v) { return v ? "true" : "false"; }
template <> std::string show(const float & v) { return fbits(v); }
template <> std::string show(const double & v) { return fbits(v); }
template <> std::string show(const std::string & v) { return "s" + v; }

template <typename T> std::string show_vec(const T & v) {
    std::string s = "[";
    for (const auto & x : v) { s += show(x); s += ","; }
    return s + "]";
}

// a float that a static_cast to an integer type cannot hold is undefined
// behaviour; AArch64 saturates, and the port does what AArch64 does, so it
// is compared there and skipped elsewhere
bool float_ub(const common_json & j, double lo, double hi) {
#if defined(__aarch64__)
    (void) j; (void) lo; (void) hi;
    return false;
#else
    if (!j.is_number_float()) return false;
    double d = j.get<double>();
    return !(d > lo - 1.0 && d < hi + 1.0);
#endif
}

template <typename T> std::string get_int(const common_json & j) {
    if (float_ub(j, (double) std::numeric_limits<T>::min(), (double) std::numeric_limits<T>::max())) return "ub";
    return show(j.get<T>());
}

std::string do_get(const common_json & j, uint8_t t) {
    switch (t) {
        case 0:  return show(j.get<bool>());
        case 1:  return get_int<int>(j);
        case 2:  return get_int<unsigned int>(j);
        case 3:  return get_int<long>(j);
        case 4:  return get_int<unsigned long>(j);
        case 5:  return get_int<long long>(j);
        case 6:  return get_int<unsigned long long>(j);
        case 7:  return show(j.get<float>());
        case 8:  return show(j.get<double>());
        case 9:  return show(j.get<std::string>());
        case 10: return show_vec(j.get<std::vector<float>>());
        case 11: return show_vec(j.get<std::vector<std::string>>());
        case 12: return show_vec(j.get<std::set<std::string>>());
        case 13: {
            for (const auto & x : j) if (float_ub(x, INT32_MIN, INT32_MAX)) return "ub";
            return show_vec(j.get<std::vector<int>>());
        }
        case 14: {
            for (const auto & x : j) if (float_ub(x, 0, 18446744073709551615.0)) return "ub";
            return show_vec(j.get<std::vector<size_t>>());
        }
        case 15: {
            if (j.is_object()) for (const auto & x : j) if (float_ub(x, 0, 18446744073709551615.0)) return "ub";
            auto m = j.get<std::unordered_map<std::string, size_t>>();
            std::vector<std::pair<std::string, size_t>> v(m.begin(), m.end());
            std::sort(v.begin(), v.end());
            std::string s = "{";
            for (const auto & kv : v) { s += kv.first; s += "="; s += show(kv.second); s += ","; }
            return s + "}";
        }
        case 16: return get_int<int64_t>(j);
        case 17: return get_int<uint32_t>(j);
        case 18: return get_int<size_t>(j);
        case 19: return show(std::string(j));   // operator std::string
    }
    return "?";
}

std::string do_value(const common_json & j, const std::string & key, uint8_t t) {
    switch (t) {
        case 0: return show(j.value(key, false));
        case 1: return show(j.value(key, -7));
        case 2: return show(j.value(key, 7u));
        case 3: return show(j.value(key, (long long) -9));
        case 4: return show(j.value(key, 2.5));
        case 5: return show(j.value(key, 1.25f));
        case 6: return show(j.value(key, std::string("dflt")));
        case 7: return show(j.value(key, "cstr"));
        case 8: return j.value(key, common_json::array({1, "x"})).dump_safe();
    }
    return "?";
}

std::string query(const common_json & j) {
    std::string s;
    s += j.is_null() ? "N" : "-";
    s += j.is_object() ? "O" : "-";
    s += j.is_array() ? "A" : "-";
    s += j.is_string() ? "S" : "-";
    s += j.is_boolean() ? "B" : "-";
    s += j.is_number() ? "#" : "-";
    s += j.is_number_integer() ? "I" : "-";
    s += j.is_number_float() ? "F" : "-";
    s += j.is_discarded() ? "D" : "-";
    s += " size=" + std::to_string(j.size());
    s += j.empty() ? " empty" : " full";
    return s;
}

nlohmann::ordered_json::error_handler_t handler(uint8_t h) {
    switch (h) {
        case 0: return nlohmann::ordered_json::error_handler_t::strict;
        case 1: return nlohmann::ordered_json::error_handler_t::replace;
        default: return nlohmann::ordered_json::error_handler_t::ignore;
    }
}

bool empty_container(const common_json & j) {
    return (j.is_array() || j.is_object()) && j.empty();
}

void run(reader & in, writer & w) {
    common_json R[8];
    while (in.p < in.end) {
        uint8_t op = in.u8();
        try {
            switch (op) {
                case OP_PARSE: {
                    uint8_t r = in.u8();
                    std::string s = in.str();
                    R[r] = common_json::parse(s);
                    w.rec("ok");
                    break;
                }
                case OP_PARSE_NT: {
                    uint8_t r = in.u8();
                    std::string s = in.str();
                    R[r] = common_json::parse_no_throw(s);
                    w.rec(R[r].is_discarded() ? "discarded" : "ok");
                    break;
                }
                case OP_SET_NULL:   { uint8_t r = in.u8(); R[r] = nullptr; w.rec("ok"); break; }
                case OP_SET_BOOL:   { uint8_t r = in.u8(); R[r] = in.u8() != 0; w.rec("ok"); break; }
                case OP_SET_INT:    { uint8_t r = in.u8(); R[r] = (int64_t) in.u64(); w.rec("ok"); break; }
                case OP_SET_UINT:   { uint8_t r = in.u8(); R[r] = (uint64_t) in.u64(); w.rec("ok"); break; }
                case OP_SET_DOUBLE: { uint8_t r = in.u8(); uint64_t b = in.u64(); double d; memcpy(&d, &b, 8); R[r] = d; w.rec("ok"); break; }
                case OP_SET_FLOAT:  { uint8_t r = in.u8(); uint32_t b = in.u32(); float f; memcpy(&f, &b, 4); R[r] = f; w.rec("ok"); break; }
                case OP_SET_STRING: { uint8_t r = in.u8(); R[r] = in.str(); w.rec("ok"); break; }
                case OP_SET_ARRAY:  { uint8_t r = in.u8(); R[r] = common_json::array(); w.rec("ok"); break; }
                case OP_SET_OBJECT: { uint8_t r = in.u8(); R[r] = common_json::object(); w.rec("ok"); break; }
                case OP_DUMP: {
                    uint8_t r = in.u8();
                    int indent = (int) in.u32();
                    w.rec("d" + R[r].dump(indent));
                    break;
                }
                case OP_DUMP_SAFE: {
                    uint8_t r = in.u8();
                    int indent = (int) in.u32();
                    w.rec("d" + R[r].dump_safe(indent));
                    break;
                }
                case OP_DUMP_RAW: {
                    uint8_t r = in.u8();
                    int indent = (int) in.u32();
                    bool ascii = in.u8() != 0;
                    uint8_t h = in.u8();
                    char ichar = (char) in.u8();
                    const auto & j = *reinterpret_cast<const nlohmann::ordered_json *>(&R[r]);
                    w.rec("d" + j.dump(indent, ichar, ascii, handler(h)));
                    break;
                }
                case OP_QUERY: { uint8_t r = in.u8(); w.rec(query(R[r])); break; }
                case OP_GET: {
                    uint8_t r = in.u8();
                    uint8_t t = in.u8();
                    w.rec(do_get(R[r], t));
                    break;
                }
                case OP_AT_KEY: {
                    uint8_t d = in.u8(), r = in.u8();
                    std::string k = in.str();
                    common_json tmp = R[r].at(k);
                    R[d] = tmp;
                    w.rec("ok");
                    break;
                }
                case OP_AT_IDX: {
                    uint8_t d = in.u8(), r = in.u8();
                    uint64_t i = in.u64();
                    common_json tmp = R[r].at((size_t) i);
                    R[d] = tmp;
                    w.rec("ok");
                    break;
                }
                case OP_CONST_INDEX: {
                    // a const operator[] cannot add: it throws like at()
                    uint8_t d = in.u8(), r = in.u8();
                    std::string k = in.str();
                    const common_json & cr = R[r];
                    common_json tmp = cr[k];
                    R[d] = tmp;
                    w.rec("ok");
                    break;
                }
                case OP_INDEX_KEY: {
                    uint8_t r = in.u8();
                    std::string k = in.str();
                    uint8_t s = in.u8();
                    common_json tmp = R[s];
                    R[r][k] = std::move(tmp);
                    w.rec("ok");
                    break;
                }
                case OP_INDEX_IDX: {
                    uint8_t r = in.u8();
                    uint64_t i = in.u64();
                    uint8_t s = in.u8();
                    common_json tmp = R[s];
                    R[r][(size_t) i] = std::move(tmp);
                    w.rec("ok");
                    break;
                }
                case OP_CONTAINS: {
                    uint8_t r = in.u8();
                    std::string k = in.str();
                    w.rec(std::string(R[r].contains(k) ? "1" : "0") + std::to_string(R[r].count(k)));
                    break;
                }
                case OP_VALUE: {
                    uint8_t r = in.u8();
                    std::string k = in.str();
                    uint8_t t = in.u8();
                    w.rec(do_value(R[r], k, t));
                    break;
                }
                case OP_FRONT:
                case OP_BACK: {
                    uint8_t d = in.u8(), r = in.u8();
                    if (empty_container(R[r])) { w.rec("skip"); break; }
                    common_json tmp = op == OP_FRONT ? R[r].front() : R[r].back();
                    R[d] = tmp;
                    w.rec("ok");
                    break;
                }
                case OP_CLEAR: { uint8_t r = in.u8(); R[r].clear(); w.rec("ok"); break; }
                case OP_ERASE_KEY: {
                    uint8_t r = in.u8();
                    std::string k = in.str();
                    R[r].erase(k);
                    w.rec("ok");
                    break;
                }
                case OP_ERASE_IDX: {
                    uint8_t r = in.u8();
                    uint64_t i = in.u64();
                    R[r].erase((size_t) i);
                    w.rec("ok");
                    break;
                }
                case OP_PUSH: {
                    uint8_t r = in.u8(), s = in.u8();
                    R[r].push_back(R[s]);
                    w.rec("ok");
                    break;
                }
                case OP_PUSH_OBJ: {
                    uint8_t r = in.u8();
                    std::string k = in.str();
                    uint8_t s = in.u8();
                    R[r].push_back({ { k, R[s] } });
                    w.rec("ok");
                    break;
                }
                case OP_INSERT: {
                    uint8_t r = in.u8(), s = in.u8();
                    R[r].insert(R[s]);
                    w.rec("ok");
                    break;
                }
                case OP_SET_ITEM: {
                    uint8_t r = in.u8();
                    std::string k = in.str();
                    uint8_t s = in.u8();
                    R[r].set({ k, R[s] });
                    w.rec("ok");
                    break;
                }
                case OP_ASSIGN: { uint8_t d = in.u8(), s = in.u8(); R[d] = R[s]; w.rec("ok"); break; }
                case OP_EQ: {
                    uint8_t r = in.u8(), s = in.u8();
                    w.rec(std::string(R[r] == R[s] ? "1" : "0") + (R[r] != R[s] ? "1" : "0"));
                    break;
                }
                case OP_ITER: {
                    uint8_t r = in.u8();
                    std::string s;
                    for (auto it = R[r].begin(); it != R[r].end(); ++it) {
                        s += (*it).dump_safe();
                        try {
                            s += "@" + it.key();
                        } catch (const std::exception & e) {
                            s += "!" + std::string(e.what());
                        }
                        s += "|";
                    }
                    for (const auto & [k, v] : R[r].items()) {
                        s += k + "=" + v.dump_safe() + "|";
                    }
                    w.rec(s);
                    break;
                }
                default:
                    w.rec("bad op");
                    return;
            }
        } catch (const std::exception & e) {
            w.rec(std::string("E:") + e.what());
        }
    }
}

} // namespace

extern "C" size_t ref_json_run(const uint8_t * prog, size_t len, char ** out) {
    reader in{ prog, prog + len };
    writer w;
    run(in, w);
    char * buf = (char *) malloc(w.out.size() + 1);
    memcpy(buf, w.out.data(), w.out.size());
    *out = buf;
    return w.out.size();
}
