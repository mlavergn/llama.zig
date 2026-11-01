// Upstream's PEG parser behind a C ABI, for scripts/peg-diff.
//
// The reference half of the oracle for `cli/peg/` and
// `cli/common_chat/chat_peg.zig`: upstream's `common/peg-parser.cpp` and
// `common/chat-peg-parser.cpp`, unmodified, from `libllama-common-ref.a`.
// `ref_peg_run` interprets a program -- the same bytecode
// `harness/peg_diff.zig` runs through the port -- and returns one record per
// result: the build's outcome, the arena's dump, its saved JSON, its GBNF
// both ways, the arena loaded back and saved again, and for every input
// every parse -- complete, lenient, every prefix -- with its AST, the three
// chat mappers' messages, the tags, and the debug trace written to stderr.
// Every exception is a record too, its class and `what()`.
//
// Upstream's own cases are not copied: its PEG tests are compiled with
// `harness/peg_rec.h` force-included, which records every parser they build
// and every input they parse, and `ref_peg_upstream` runs them and hands the
// recordings over -- after replaying each through upstream's builder and
// requiring the arena it gives to save exactly as the test's own did.

#define PEG_REC_NO_RENAME
#include "peg_rec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>

#include <fcntl.h>
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

// the tests' own mains, renamed by scripts/peg-diff
int peg_test_main(int argc, char ** argv);
int peg_chat_test_main(int argc, char ** argv);

namespace {

struct reader {
    const uint8_t * p;
    uint8_t  u8()  { return *p++; }
    uint32_t u32() { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
    int32_t  i32() { int32_t v; memcpy(&v, p, 4); p += 4; return v; }
    std::string str() { uint32_t n = u32(); std::string s((const char *) p, n); p += n; return s; }
    std::vector<std::string> strs() {
        std::vector<std::string> v(u32());
        for (auto & s : v) s = str();
        return v;
    }
};

struct writer {
    std::string out;
    void u32(uint32_t n) { out.append((const char *) &n, 4); }
    void rec(const std::string & s) {
        u32((uint32_t) s.size());
        out.append(s);
    }
};

uint8_t * give(const std::string & s, size_t * out_len) {
    auto * buf = (uint8_t *) malloc(s.size() ? s.size() : 1);
    memcpy(buf, s.data(), s.size());
    *out_len = s.size();
    return buf;
}

std::string err(const std::exception_ptr & ep) {
    try {
        std::rethrow_exception(ep);
    } catch (const std::invalid_argument & e) {
        return std::string("IA:") + e.what();
    } catch (const common_json_error & e) {
        return std::string("JE:") + e.what();
    } catch (const std::out_of_range & e) {
        return std::string("OR:") + e.what();
    } catch (const std::runtime_error & e) {
        return std::string("RE:") + e.what();
    } catch (const std::exception & e) {
        return std::string("EX:") + e.what();
    }
}

template <typename F> std::string guard(F f) {
    try {
        return "G:" + f();
    } catch (...) {
        return "E:" + err(std::current_exception());
    }
}

// a harness failure, not upstream's
struct harness_error : std::exception {
    std::string msg;
    explicit harness_error(std::string m) : msg(std::move(m)) {}
    const char * what() const noexcept override { return msg.c_str(); }
};

// stderr, captured: the debug trace is written with fprintf
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

// The description interpreter: every call through upstream's chat builder.
struct interp {
    common_chat_peg_builder &                        pb;
    std::vector<std::optional<common_peg_parser_id>> slots;
    uint32_t                                         n = 0;

    explicit interp(common_chat_peg_builder & pb) : pb(pb) {}

    common_peg_parser get(uint32_t s) {
        if (s >= slots.size() || !slots[s]) throw harness_error("HE:unset slot " + std::to_string(s));
        return common_peg_parser(*slots[s], pb);
    }
    void put(uint32_t s, const common_peg_parser & p) {
        if (slots.size() <= s) slots.resize(s + 1);
        slots[s] = p.id();
    }
    std::vector<common_peg_parser> list(reader & r, bool exec) {
        uint32_t k = r.u32();
        std::vector<uint32_t> ss(k);
        for (auto & x : ss) x = r.u32();
        std::vector<common_peg_parser> out;
        if (exec) for (auto x : ss) out.push_back(get(x));
        return out;
    }

    // the ops until a body's end; returns the slot the body ended on
    uint32_t body(reader & r, bool exec) {
        while (*r.p != PEG_BODY_END) step(r, exec);
        r.u8();
        return r.u32();
    }

    // one call: its arguments always read, made only when `exec`
    void step(reader & r, bool exec) {
        const uint8_t code = r.u8();
        const uint32_t self = n++;
        auto unary = [&](auto f) {
            uint32_t a = r.u32();
            if (exec) put(self, f(get(a)));
        };
        auto nullary = [&](auto f) {
            if (exec) put(self, f());
        };
        switch (code) {
            case PEG_EPS: nullary([&] { return pb.eps(); }); break;
            case PEG_START: nullary([&] { return pb.start(); }); break;
            case PEG_END: nullary([&] { return pb.end(); }); break;
            case PEG_LITERAL: { auto s = r.str(); nullary([&] { return pb.literal(s); }); break; }
            case PEG_SEQ_EMPTY: nullary([&] { return pb.sequence(); }); break;
            case PEG_SEQ: { auto l = list(r, exec); nullary([&] { return pb.sequence(l); }); break; }
            case PEG_CHOICE_EMPTY: nullary([&] { return pb.choice(); }); break;
            case PEG_CHOICE: { auto l = list(r, exec); nullary([&] { return pb.choice(l); }); break; }
            case PEG_ONE_OR_MORE: unary([&](const common_peg_parser & p) { return pb.one_or_more(p); }); break;
            case PEG_ZERO_OR_MORE: unary([&](const common_peg_parser & p) { return pb.zero_or_more(p); }); break;
            case PEG_OPTIONAL: unary([&](const common_peg_parser & p) { return pb.optional(p); }); break;
            case PEG_PEEK: unary([&](const common_peg_parser & p) { return pb.peek(p); }); break;
            case PEG_NEGATE: unary([&](const common_peg_parser & p) { return pb.negate(p); }); break;
            case PEG_ANY: nullary([&] { return pb.any(); }); break;
            case PEG_CHARS: {
                auto s = r.str();
                int mn = r.i32(), mx = r.i32();
                nullary([&] { return pb.chars(s, mn, mx); });
                break;
            }
            case PEG_REF: { auto s = r.str(); nullary([&] { return pb.ref(s); }); break; }
            case PEG_SPACE: nullary([&] { return pb.space(); }); break;
            case PEG_UNTIL: { auto s = r.str(); nullary([&] { return pb.until(s); }); break; }
            case PEG_UNTIL_ONE_OF: { auto v = r.strs(); nullary([&] { return pb.until_one_of(v); }); break; }
            case PEG_REST: nullary([&] { return pb.rest(); }); break;
            case PEG_REPEAT: {
                uint32_t a = r.u32();
                int mn = r.i32(), mx = r.i32();
                if (exec) put(self, pb.repeat(get(a), mn, mx));
                break;
            }
            case PEG_REPEAT_N: {
                uint32_t a = r.u32();
                int k = r.i32();
                if (exec) put(self, pb.repeat(get(a), k));
                break;
            }
            case PEG_DQ_STRING: nullary([&] { return pb.double_quoted_string(); }); break;
            case PEG_SQ_STRING: nullary([&] { return pb.single_quoted_string(); }); break;
            case PEG_QUOTED_STRING: nullary([&] { return pb.quoted_string(); }); break;
            case PEG_STRING_CONTENT: { char d = (char) r.u8(); nullary([&] { return pb.string_content(d); }); break; }
            case PEG_JSON: nullary([&] { return pb.json(); }); break;
            case PEG_JSON_OBJECT: nullary([&] { return pb.json_object(); }); break;
            case PEG_JSON_STRING: nullary([&] { return pb.json_string(); }); break;
            case PEG_JSON_ARRAY: nullary([&] { return pb.json_array(); }); break;
            case PEG_JSON_NUMBER: nullary([&] { return pb.json_number(); }); break;
            case PEG_JSON_BOOL: nullary([&] { return pb.json_bool(); }); break;
            case PEG_JSON_NULL: nullary([&] { return pb.json_null(); }); break;
            case PEG_JSON_MEMBER: {
                auto k = r.str();
                uint32_t a = r.u32();
                if (exec) put(self, pb.json_member(k, get(a)));
                break;
            }
            case PEG_PY_VALUE: nullary([&] { return pb.python_value(); }); break;
            case PEG_PY_DICT: nullary([&] { return pb.python_dict(); }); break;
            case PEG_PY_STRING: nullary([&] { return pb.python_string(); }); break;
            case PEG_PY_ARRAY: nullary([&] { return pb.python_array(); }); break;
            case PEG_PY_NUMBER: nullary([&] { return pb.python_number(); }); break;
            case PEG_PY_BOOL: nullary([&] { return pb.python_bool(); }); break;
            case PEG_PY_NULL: nullary([&] { return pb.python_null(); }); break;
            case PEG_MARKER: nullary([&] { return pb.marker(); }); break;
            case PEG_SCHEMA: {
                uint32_t a = r.u32();
                auto name = r.str();
                auto text = r.str();
                bool raw = r.u8();
                if (exec) put(self, pb.schema(get(a), name, common_json::parse(text), raw));
                break;
            }
            case PEG_RULE: {
                auto name = r.str();
                uint32_t a = r.u32();
                bool trig = r.u8();
                if (exec) put(self, pb.rule(name, get(a), trig));
                break;
            }
            case PEG_TRIGGER_RULE: {
                auto name = r.str();
                uint32_t a = r.u32();
                if (exec) put(self, pb.trigger_rule(name, get(a)));
                break;
            }
            case PEG_RULE_FN:
            case PEG_TRIGGER_RULE_FN: {
                auto name = r.str();
                bool trig = code == PEG_RULE_FN ? (bool) r.u8() : true;
                if (!exec) {
                    body(r, false);
                    break;
                }
                bool called = false;
                auto fn = [&]() -> common_peg_parser {
                    called = true;
                    uint32_t s = body(r, true);
                    if (s == PEG_NO_SLOT) throw harness_error("HE:a body that returned nothing");
                    return get(s);
                };
                common_peg_parser res = code == PEG_RULE_FN ? pb.rule(name, fn, trig) : pb.trigger_rule(name, fn);
                if (!called) body(r, false);
                put(self, res);
                break;
            }
            case PEG_ATOMIC: unary([&](const common_peg_parser & p) { return pb.atomic(p); }); break;
            case PEG_TAG: {
                auto t = r.str();
                uint32_t a = r.u32();
                if (exec) put(self, pb.tag(t, get(a)));
                break;
            }
            case PEG_GBNF: {
                uint32_t a = r.u32();
                auto g = r.str();
                if (exec) put(self, pb.gbnf(get(a), g));
                break;
            }
            case PEG_AC: {
                uint32_t a = r.u32();
                auto v = r.strs();
                if (exec) put(self, pb.ac(get(a), v));
                break;
            }
            case PEG_AC_ONE: {
                uint32_t a = r.u32();
                auto d = r.str();
                if (exec) put(self, pb.ac(get(a), d));
                break;
            }
            case PEG_OP_PLUS: case PEG_OP_SHL: case PEG_OP_OR: case PEG_OP_PLUS_ASSIGN: case PEG_OP_OR_ASSIGN: {
                uint32_t a = r.u32(), b = r.u32();
                if (!exec) break;
                common_peg_parser x = get(a), y = get(b);
                switch (code) {
                    case PEG_OP_PLUS: put(self, x + y); break;
                    case PEG_OP_SHL: put(self, x << y); break;
                    case PEG_OP_OR: put(self, x | y); break;
                    case PEG_OP_PLUS_ASSIGN: x += y; put(self, x); break;
                    default: x |= y; put(self, x); break;
                }
                break;
            }
            case PEG_OP_PLUS_LIT: case PEG_OP_SHL_LIT: case PEG_OP_OR_LIT: {
                uint32_t a = r.u32();
                auto s = r.str();
                if (!exec) break;
                common_peg_parser x = get(a);
                put(self, code == PEG_OP_PLUS_LIT ? x + s : code == PEG_OP_SHL_LIT ? (x << s) : (x | s));
                break;
            }
            case PEG_OP_LIT_PLUS: case PEG_OP_LIT_SHL: case PEG_OP_LIT_OR: {
                auto s = r.str();
                uint32_t a = r.u32();
                if (!exec) break;
                common_peg_parser x = get(a);
                put(self, code == PEG_OP_LIT_PLUS ? s + x : code == PEG_OP_LIT_SHL ? (s << x) : (s | x));
                break;
            }
            case PEG_REASONING_BLOCK: unary([&](const common_peg_parser & p) { return pb.reasoning_block(p); }); break;
            case PEG_REASONING: unary([&](const common_peg_parser & p) { return pb.reasoning(p); }); break;
            case PEG_CONTENT: unary([&](const common_peg_parser & p) { return pb.content(p); }); break;
            case PEG_TAG_SAFE: {
                auto t = r.str();
                auto m = r.str();
                uint32_t a = r.u32();
                if (exec) put(self, pb.tag_with_safe_content(t, m, get(a)));
                break;
            }
            case PEG_TOOL: unary([&](const common_peg_parser & p) { return pb.tool(p); }); break;
            case PEG_TOOL_OPEN: unary([&](const common_peg_parser & p) { return pb.tool_open(p); }); break;
            case PEG_TOOL_CLOSE: unary([&](const common_peg_parser & p) { return pb.tool_close(p); }); break;
            case PEG_TOOL_ID: unary([&](const common_peg_parser & p) { return pb.tool_id(p); }); break;
            case PEG_TOOL_NAME: unary([&](const common_peg_parser & p) { return pb.tool_name(p); }); break;
            case PEG_TOOL_ARGS: unary([&](const common_peg_parser & p) { return pb.tool_args(p); }); break;
            case PEG_TOOL_ARG: unary([&](const common_peg_parser & p) { return pb.tool_arg(p); }); break;
            case PEG_TOOL_ARG_OPEN: unary([&](const common_peg_parser & p) { return pb.tool_arg_open(p); }); break;
            case PEG_TOOL_ARG_CLOSE: unary([&](const common_peg_parser & p) { return pb.tool_arg_close(p); }); break;
            case PEG_TOOL_ARG_NAME: unary([&](const common_peg_parser & p) { return pb.tool_arg_name(p); }); break;
            case PEG_TOOL_ARG_VALUE: unary([&](const common_peg_parser & p) { return pb.tool_arg_value(p); }); break;
            case PEG_TOOL_ARG_STRING_VALUE: unary([&](const common_peg_parser & p) { return pb.tool_arg_string_value(p); }); break;
            case PEG_TOOL_ARG_JSON_VALUE: unary([&](const common_peg_parser & p) { return pb.tool_arg_json_value(p); }); break;
            case PEG_PERMUTE: {
                auto pre = r.str();
                auto l = list(r, exec);
                if (exec) put(self, pb.permute(pre, l));
                break;
            }
            case PEG_PREFIX: {
                auto s = r.str();
                auto d = r.str();
                nullary([&] { return pb.prefix(s, d); });
                break;
            }
            case PEG_OPTSPACE: { auto s = r.str(); nullary([&] { return pb.optspace(s); }); break; }
            case PEG_STD_JSON_TOOLS: {
                auto ss = r.str(), se = r.str(), tools = r.str();
                bool par = r.u8(), force = r.u8();
                auto nk = r.str(), ak = r.str();
                bool aw = r.u8(), fik = r.u8();
                auto ck = r.str(), gk = r.str();
                auto order = r.strs();
                bool oai = r.u8();
                if (exec) put(self, pb.standard_json_tools(ss, se, common_json::parse(tools), par, force, nk, ak, aw, fik, ck, gk, order, oai));
                break;
            }
            case PEG_STD_CONSTRUCTED_TOOLS: {
                std::map<std::string, std::string> markers;
                uint32_t k = r.u32();
                for (uint32_t i = 0; i < k; i++) {
                    auto key = r.str();
                    auto val = r.str();
                    markers.emplace(key, val);
                }
                auto tools = r.str();
                bool par = r.u8(), force = r.u8();
                if (exec) put(self, pb.standard_constructed_tools(markers, common_json::parse(tools), par, force));
                break;
            }
            case PEG_PY_STYLE_TOOLS: {
                auto tools = r.str();
                bool par = r.u8(), lit = r.u8();
                if (exec) put(self, pb.python_style_tool_calls(common_json::parse(tools), par, lit));
                break;
            }
            case PEG_SET_ROOT: {
                uint32_t a = r.u32();
                if (exec) pb.set_root(get(a));
                break;
            }
            default:
                throw harness_error("HE:unknown op " + std::to_string(code));
        }
    }
};

// Every call of a description, through upstream's builder, then built.
common_peg_arena build_from(const std::string & ops) {
    common_chat_peg_builder pb;
    interp it(pb);
    reader r { (const uint8_t *) ops.data() };
    const uint8_t * end = r.p + ops.size();
    while (r.p < end) it.step(r, true);
    return pb.build();
}

std::string tool_calls_text(const common_chat_msg & m) {
    std::string s = m.content + "\x1f" + m.reasoning_content + "\x1f" + std::to_string(m.tool_calls.size());
    for (const auto & tc : m.tool_calls) s += "\x1e" + tc.name + "\x1d" + tc.arguments + "\x1d" + tc.id;
    return s;
}

std::string result_text(const common_peg_parse_result & r) {
    std::string s = std::string(common_peg_parse_result_type_name(r.type)) + " " + std::to_string(r.start) + " " + std::to_string(r.end);
    for (auto n : r.nodes) s += " " + std::to_string(n);
    return s;
}

std::string grammar(const common_peg_arena & a, bool lazy) {
    return build_grammar([&](const common_grammar_builder & b) { a.build_grammar(b, lazy); });
}

// One parse and everything read off it.
void parse_records(const common_peg_arena & arena, const std::string & input, common_peg_parse_flags flags, const std::string & tag, writer & w) {
    common_peg_parse_context ctx(input, flags);
    common_peg_parse_result result;
    try {
        result = arena.parse(ctx);
    } catch (...) {
        w.rec(tag + "E:" + err(std::current_exception()));
        return;
    }
    w.rec(tag + result_text(result));
    w.rec("A:" + guard([&] { return ctx.ast.dump(); }));
    for (int k = 0; k < 3; k++) {
        std::string out = guard([&] {
            common_chat_msg msg;
            std::unique_ptr<common_chat_peg_mapper> m;
            if (k == 0) m = std::make_unique<common_chat_peg_mapper>(msg);
            if (k == 1) m = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
            if (k == 2) m = std::make_unique<common_chat_peg_minimax_m3_mapper>(msg);
            m->from_ast(ctx.ast, result);
            return tool_calls_text(msg);
        });
        w.rec("M" + std::to_string(k) + ":" + out);
    }
}

std::string tags_text(const tagged_parse_result & t) {
    std::string s = result_text(t.result);
    for (const auto & [k, v] : t.tags) s += "\x1e" + k + "\x1d" + v;
    return s;
}

void run_case(reader & r, writer & w) {
    const uint32_t id = r.u32();
    const uint8_t kind = r.u8();
    const std::string text = r.str();
    const uint32_t n_inputs = r.u32();
    std::vector<std::pair<std::string, uint8_t>> inputs;
    for (uint32_t i = 0; i < n_inputs; i++) {
        auto s = r.str();
        uint8_t modes = r.u8();
        inputs.emplace_back(s, modes);
    }

    writer cw;
    std::optional<common_peg_arena> arena;
    try {
        if (kind == 3) {
            common_peg_arena a;
            a.load(text);
            arena = std::move(a);
        } else {
            arena = build_from(text);
        }
        cw.rec("B:ok");
    } catch (...) {
        cw.rec("B:E:" + err(std::current_exception()));
    }
    if (arena) {
        const auto & a = *arena;
        cw.rec("N:" + std::to_string(a.size()));
        cw.rec("D:" + guard([&] { return a.dump(a.root()); }));
        std::string saved = guard([&] { return a.save(); });
        cw.rec("S:" + saved);
        cw.rec("G:" + guard([&] { return grammar(a, false); }));
        cw.rec("L:" + guard([&] { return grammar(a, true); }));
        if (saved.rfind("G:", 0) == 0) {
            cw.rec("R:" + guard([&] {
                common_peg_arena b;
                b.load(saved.substr(2));
                return b.save();
            }));
        }
        for (const auto & [input, modes] : inputs) {
            cw.rec("I:" + input);
            if (modes & 1) parse_records(a, input, COMMON_PEG_PARSE_FLAG_NONE, "P0:", cw);
            if (modes & 2) parse_records(a, input, COMMON_PEG_PARSE_FLAG_LENIENT, "P1:", cw);
            if (modes & 4) {
                size_t stride = input.size() <= 160 ? 1 : input.size() / 120;
                for (size_t k = 0; k < input.size(); k += stride) {
                    parse_records(a, input.substr(0, k), COMMON_PEG_PARSE_FLAG_LENIENT, "Q" + std::to_string(k) + ":", cw);
                }
            }
            if (modes & 8) {
                cap.begin();
                std::string res;
                try {
                    common_peg_parse_context ctx(input, COMMON_PEG_PARSE_FLAG_LENIENT | COMMON_PEG_PARSE_FLAG_DEBUG);
                    res = result_text(a.parse(ctx));
                } catch (...) {
                    res = "E:" + err(std::current_exception());
                }
                std::string trace = cap.end();
                cw.rec("X:" + res + "\n" + trace);
            }
            if (modes & 16) {
                tagged_peg_parser tp { a };
                cw.rec("T0:" + guard([&] { return tags_text(tp.parse_and_extract(input)); }));
                cw.rec("T1:" + guard([&] { return tags_text(tp.parse_and_extract(input, COMMON_PEG_PARSE_FLAG_LENIENT)); }));
                cw.rec("T2:" + guard([&] { return tags_text(tp.parse_anywhere_and_extract(input)); }));
            }
        }
    }
    w.u32(id);
    w.out += cw.out;
    w.rec("\x01");  // end of case
}

}  // namespace

// A program: cases until the end; per case its id, then its records, then
// a record holding a single \x01.
extern "C" uint8_t * ref_peg_run(const uint8_t * prog, size_t len, size_t * out_len) {
    reader r { prog };
    writer w;
    while (r.p < prog + len) run_case(r, w);
    return give(w.out, out_len);
}

// Upstream's PEG tests, run with the recorder: the number of failures each
// reported, then per parser built the kind, the description, and the
// inputs it was parsed over. A description that, replayed through
// upstream's builder, does not save as the test's own arena did is a
// harness failure, reported in place of the corpus.
extern "C" uint8_t * ref_peg_upstream(size_t * out_len) {
    writer w;
    int saved_out = dup(1);
    int devnull = open(HARNESS_DEVNULL, O_WRONLY);
    fflush(stdout);
    std::cout.flush();
    dup2(devnull, 1);
    cap.begin();
    char arg0[] = "peg-diff";
    char * argv[] = { arg0, nullptr };
    int f1 = peg_test_main(1, argv);
    int f2 = peg_chat_test_main(1, argv);
    std::cout.flush();
    fflush(stdout);
    cap.end();
    dup2(saved_out, 1);
    close(saved_out);
    close(devnull);

    w.rec(std::to_string(f1) + " " + std::to_string(f2));
    std::string mismatch;
    for (size_t i = 0; i < g_peg_rec.cases.size(); i++) {
        const auto & c = g_peg_rec.cases[i];
        std::string replay;
        try {
            replay = rec_save(build_from(c.ops));
        } catch (...) {
            replay = "E:" + err(std::current_exception());
        }
        if (replay != c.saved && mismatch.empty()) mismatch = "case " + std::to_string(i) + ": " + replay.substr(0, 200) + " vs " + c.saved.substr(0, 200);
    }
    w.rec(mismatch);
    w.u32((uint32_t) g_peg_rec.cases.size());
    for (const auto & c : g_peg_rec.cases) {
        w.rec(std::string(1, (char) c.kind));
        w.rec(c.ops);
        w.u32((uint32_t) c.inputs.size());
        for (const auto & s : c.inputs) w.rec(s);
    }
    return give(w.out, out_len);
}
