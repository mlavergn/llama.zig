// Upstream's JSON schema to grammar behind a C ABI, for scripts/schema-diff.
//
// The reference half of the oracle for `cli/json/schema.zig` and
// `cli/json/schema_to_grammar.zig`: `common/json-schema.cpp`,
// `common/json-schema-to-grammar.cpp`, `common/json.cpp`, `common/trie.cpp`,
// `common/unicode.cpp` and `common/common.cpp` are compiled unmodified and
// linked beside this file. `ref_schema_run` interprets a program -- the same
// bytecode `harness/schema_diff.zig` runs through the Zig port -- and
// returns one record per result: a grammar, the class and `what()` of an
// exception, the warning line `check_errors` writes to stderr, a schema
// tree read back through `json-schema.h`, or a name `build_grammar`'s
// builder returned.
//
// Upstream's own cases are not copied: `tests/test-json-schema-to-grammar.cpp`
// is included below with its `main` renamed, and `ref_schema_upstream`
// drives its `test_all` to hand every case to the harness.

#include "json-schema-to-grammar.h"
#include "json-schema.h"
#include "json.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#ifdef _WIN32
// MinGW has `dup2`, `lseek` and `ftruncate` but no `pread`: the capture
// below reads its scratch file from the start through these, and
// `/dev/null` is `NUL` (scripts/schema-diff and scripts/peg-diff under
// WIN_X86).
#include <io.h>
static long long harness_pread(int fd, void * buf, size_t n, long long off) {
    if (_lseeki64(fd, off, SEEK_SET) != off) return -1;
    return _read(fd, buf, (unsigned) n);
}
#define pread harness_pread
#define HARNESS_DEVNULL "NUL"
#else
#define HARNESS_DEVNULL "/dev/null"
#endif

// Renamed, its `main` is never called: it falls off its end, which only
// `main` may do. scripts/schema-diff runs the test as its own program.
#define main schema_diff_upstream_test_main
#include "../llama.cpp/tests/test-json-schema-to-grammar.cpp"
#undef main

namespace {

enum op : uint8_t {
    OP_CASE = 0,    // u32 id
    OP_J2G  = 1,    // str schema, u8 force_gbnf
    OP_DOC  = 2,    // str schema
    OP_BUILD = 3,   // str schema, u8 dotall, u32 n, steps
    OP_LIT  = 4,    // str literal
};

enum step : uint8_t {
    STEP_RULE = 0,    // str name, str rule
    STEP_SCHEMA = 1,  // str name, u32 n, u32 path[n]
};

struct reader {
    const uint8_t * p;
    uint8_t  u8()  { return *p++; }
    uint32_t u32() { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
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

// stderr, captured: `check_errors` writes its warning with fprintf, so fd 2
// is pointed at a scratch file around each conversion and read back.
struct stderr_capture {
    FILE * tmp   = nullptr;
    int    saved = -1;

    void begin() {
        if (!tmp) tmp = tmpfile();
        fflush(stderr);
        ftruncate(fileno(tmp), 0);
        lseek(fileno(tmp), 0, SEEK_SET);
        saved = dup(2);
        dup2(fileno(tmp), 2);
    }

    std::string end() {
        fflush(stderr);
        dup2(saved, 2);
        close(saved);
        off_t n = lseek(fileno(tmp), 0, SEEK_END);
        std::string s((size_t) n, '\0');
        if (n > 0 && pread(fileno(tmp), &s[0], (size_t) n, 0) != n) s = "<read error>";
        return s;
    }
};

stderr_capture cap;

template <typename F> std::string guard(F f) {
    try {
        return "G:" + f();
    } catch (const std::invalid_argument & e) {
        return std::string("IA:") + e.what();
    } catch (const common_json_error & e) {
        return std::string("JE:") + e.what();
    } catch (const std::runtime_error & e) {
        return std::string("RE:") + e.what();
    } catch (const std::exception & e) {
        return std::string("EX:") + e.what();
    }
}

// the children a path steps through, as harness/schema_diff.zig enumerates them
std::vector<const common_chat_schema *> children(const common_chat_schema & s) {
    std::vector<const common_chat_schema *> out;
    switch (s.kind()) {
        case common_chat_schema::KIND_ANY_OF:
            for (const auto & c : static_cast<const common_chat_schema_any_of &>(s).children) out.push_back(c.get());
            break;
        case common_chat_schema::KIND_ALL_OF:
            for (const auto & c : static_cast<const common_chat_schema_all_of &>(s).children) out.push_back(c.get());
            break;
        case common_chat_schema::KIND_ARRAY:
            out.push_back(static_cast<const common_chat_schema_array &>(s).items.get());
            break;
        case common_chat_schema::KIND_TUPLE:
            for (const auto & c : static_cast<const common_chat_schema_tuple &>(s).items) out.push_back(c.get());
            break;
        case common_chat_schema::KIND_OBJECT: {
            const auto & o = static_cast<const common_chat_schema_object &>(s);
            for (const auto & p : o.properties) out.push_back(p.schema.get());
            if (o.additional_properties) out.push_back(o.additional_properties.get());
            break;
        }
        default:
            break;
    }
    return out;
}

// one line per node, preorder, through every accessor json-schema.h has
void describe(const common_chat_schema & s, int depth, std::string & out) {
    out.append((size_t) depth, ' ');
    out += common_chat_schema::kind_name(s.kind());
    char buf[128];
    const auto types = s.value_types();
    std::string ts;
    for (int t = common_chat_schema::TYPE_NULL; t <= common_chat_schema::TYPE_OBJECT; t++) {
        if (types.has((common_chat_schema::value_type) t)) {
            ts += common_chat_schema::type_name((common_chat_schema::value_type) t);
            ts += ',';
        }
    }
    snprintf(buf, sizeof buf, " types=%s empty=%d str=%d", ts.c_str(), (int) types.empty(), (int) s.may_be_string());
    out += buf;
    switch (s.kind()) {
        case common_chat_schema::KIND_REF: {
            const auto & r = static_cast<const common_chat_schema_ref &>(s);
            out += " ref=" + r.ref + " target=" + (r.target ? common_chat_schema::kind_name(r.target->kind()) : "null");
            break;
        }
        case common_chat_schema::KIND_CONST:
            out += " value=" + static_cast<const common_chat_schema_const &>(s).value.dump_safe();
            break;
        case common_chat_schema::KIND_ENUM:
            out += " values=";
            for (const auto & v : static_cast<const common_chat_schema_enum &>(s).values) out += v.dump_safe() + ",";
            break;
        case common_chat_schema::KIND_INTEGER: {
            const auto & n = static_cast<const common_chat_schema_integer &>(s);
            snprintf(buf, sizeof buf, " min=%lld max=%lld", (long long) n.minimum, (long long) n.maximum);
            out += buf;
            break;
        }
        case common_chat_schema::KIND_STRING: {
            const auto & n = static_cast<const common_chat_schema_string &>(s);
            snprintf(buf, sizeof buf, " format=%d min=%d max=%d pattern=", (int) n.format, n.min_length, n.max_length);
            out += buf;
            out += n.pattern;
            break;
        }
        case common_chat_schema::KIND_ARRAY: {
            const auto & n = static_cast<const common_chat_schema_array &>(s);
            snprintf(buf, sizeof buf, " min=%d max=%d", n.min_items, n.max_items);
            out += buf;
            break;
        }
        case common_chat_schema::KIND_OBJECT: {
            const auto & o = static_cast<const common_chat_schema_object &>(s);
            for (const auto & p : o.properties) out += " prop=" + p.name + (p.required ? "!" : "");
            out += o.additional_properties ? " additional" : " closed";
            break;
        }
        default:
            break;
    }
    out += '\n';
    for (const auto * c : children(s)) describe(*c, depth + 1, out);
}

const common_chat_schema * select(const common_chat_schema * s, const std::vector<uint32_t> & path) {
    for (uint32_t k : path) {
        auto kids = children(*s);
        if (kids.empty()) break;
        s = kids[k % kids.size()];
    }
    return s;
}

void run_one(reader & r, writer & w) {
    switch (r.u8()) {
        case OP_CASE: {
            w.rec("#" + std::to_string(r.u32()));
            break;
        }
        case OP_J2G: {
            const std::string text = r.str();
            const bool force = r.u8() != 0;
            common_json j;
            try {
                j = common_json::parse(text);
            } catch (const std::exception & e) {
                w.rec(std::string("P:") + e.what());
                break;
            }
            cap.begin();
            std::string res = guard([&] { return json_schema_to_grammar(j, force); });
            std::string warn = cap.end();
            w.rec(res);
            w.rec("W:" + warn);
            break;
        }
        case OP_DOC: {
            const std::string text = r.str();
            common_json j;
            try {
                j = common_json::parse(text);
            } catch (const std::exception & e) {
                w.rec(std::string("P:") + e.what());
                break;
            }
            common_chat_schema_document doc;
            std::string tree = guard([&] {
                doc = common_chat_schema_from_json(j);
                std::string out;
                describe(*doc.root, 0, out);
                for (const auto & e : doc.refs) {
                    out += "refs[" + e.first + "]=" + common_chat_schema::kind_name(e.second->kind()) + "\n";
                }
                return out;
            });
            w.rec(tree);
            if (tree.compare(0, 2, "G:") != 0) break;
            cap.begin();
            std::string res = guard([&] { return json_schema_to_grammar(doc); });
            std::string warn = cap.end();
            w.rec(res);
            w.rec("W:" + warn);
            break;
        }
        case OP_BUILD: {
            const std::string text = r.str();
            const bool dotall = r.u8() != 0;
            struct stepdef { uint8_t kind; std::string name, rule; std::vector<uint32_t> path; };
            std::vector<stepdef> steps(r.u32());
            for (auto & s : steps) {
                s.kind = r.u8();
                s.name = r.str();
                if (s.kind == STEP_RULE) {
                    s.rule = r.str();
                } else {
                    s.path.resize(r.u32());
                    for (auto & k : s.path) k = r.u32();
                }
            }
            common_chat_schema_document doc;
            if (!text.empty()) {
                std::string built = guard([&] {
                    doc = common_chat_schema_from_json(common_json::parse(text));
                    return std::string();
                });
                w.rec(built);
                if (built != "G:") break;
            }
            std::vector<std::string> names;
            cap.begin();
            std::string res = guard([&] {
                return build_grammar([&](const common_grammar_builder & b) {
                    for (const auto & s : steps) {
                        if (s.kind == STEP_RULE) {
                            names.push_back(b.add_rule(s.name, s.rule));
                        } else if (doc.root) {
                            names.push_back(b.add_schema(s.name, *select(doc.root.get(), s.path)));
                        }
                    }
                }, common_grammar_options { dotall });
            });
            std::string warn = cap.end();
            for (const auto & n : names) w.rec("S:" + n);
            w.rec(res);
            w.rec("W:" + warn);
            break;
        }
        case OP_LIT: {
            const std::string s = r.str();
            w.rec(gbnf_format_literal(s));
            break;
        }
        default:
            fprintf(stderr, "schema_ref: bad op\n");
            abort();
    }
}

uint8_t * give(const std::string & s, size_t * len) {
    auto * p = (uint8_t *) malloc(s.size() ? s.size() : 1);
    memcpy(p, s.data(), s.size());
    *len = s.size();
    return p;
}

} // namespace

// Runs a program; the records are returned in a malloc'd buffer.
extern "C" uint8_t * ref_schema_run(const uint8_t * prog, size_t len, size_t * out_len) {
    reader r { prog };
    writer w;
    while (r.p < prog + len) run_one(r, w);
    return give(w.out, out_len);
}

// Every case of upstream's test, through its own `test_all`: per case the
// name, the schema, the expected status ("S"/"F"), the expected grammar,
// and whether this reference reproduces the expectation the way upstream's
// runner judges it ("ok"/"MISMATCH").
extern "C" uint8_t * ref_schema_upstream(size_t * out_len) {
    writer w;
    cap.begin();
    test_all("schema-diff", [&](const TestCase & tc) {
        w.rec(tc.name);
        w.rec(tc.schema);
        w.rec(tc.expected_status == SUCCESS ? "S" : "F");
        w.rec(tc.expected_grammar);
        bool ok;
        try {
            const std::string g = json_schema_to_grammar(common_json::parse(tc.schema), true);
            ok = tc.expected_status == SUCCESS && trim(g) == trim(tc.expected_grammar);
        } catch (const std::invalid_argument &) {
            ok = tc.expected_status == FAILURE;
        }
        w.rec(ok ? "ok" : "MISMATCH");
    });
    cap.end();
    return give(w.out, out_len);
}
