// A recording stand-in for upstream's PEG builder, for scripts/peg-diff.
//
// Not a port. Upstream's PEG tests (tests/test-peg-parser.cpp and the files
// under tests/peg-parser/, tests/test-chat-peg-parser.cpp) build parsers in
// C++ lambdas, so their cases are code, not data. This header is force-
// included ahead of each of them (`-include harness/peg_rec.h`): it pulls
// in the real headers first, then renames -- for the rest of the test's own
// translation unit only -- the builder, the parser handle, the parse context
// and the three `build_*` helpers to the `rec_*` types below. Each of those
// forwards every call to the real builder and writes the call into a
// description, the bytecode `harness/peg_ref.cpp` and `harness/peg_diff.zig`
// both interpret, and every input a parse context is built from is noted
// against the description being built. The tests still run against the real
// parser and must still pass; the descriptions and inputs become peg-diff's
// upstream cases, replayed through upstream's builder and the port's alike.
//
// A handle's slot is the ordinal of the call that made it, counted over
// every call written, so a replay that skips a rule body -- a rule built by
// a function is built once per name -- leaves those slots empty on both
// sides alike.

#pragma once

#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "peg-parser.h"

#include "../llama.cpp/tests/testing.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// The opcodes, shared with harness/peg_ref.cpp and harness/peg_diff.zig.
enum peg_op : uint8_t {
    PEG_EPS = 0, PEG_START, PEG_END, PEG_LITERAL, PEG_SEQ_EMPTY, PEG_SEQ, PEG_CHOICE_EMPTY, PEG_CHOICE,
    PEG_ONE_OR_MORE, PEG_ZERO_OR_MORE, PEG_OPTIONAL, PEG_PEEK, PEG_NEGATE, PEG_ANY, PEG_CHARS, PEG_REF,
    PEG_SPACE, PEG_UNTIL, PEG_UNTIL_ONE_OF, PEG_REST, PEG_REPEAT, PEG_REPEAT_N, PEG_DQ_STRING, PEG_SQ_STRING,
    PEG_QUOTED_STRING, PEG_STRING_CONTENT, PEG_JSON, PEG_JSON_OBJECT, PEG_JSON_STRING, PEG_JSON_ARRAY,
    PEG_JSON_NUMBER, PEG_JSON_BOOL, PEG_JSON_NULL, PEG_JSON_MEMBER, PEG_PY_VALUE, PEG_PY_DICT, PEG_PY_STRING,
    PEG_PY_ARRAY, PEG_PY_NUMBER, PEG_PY_BOOL, PEG_PY_NULL, PEG_MARKER, PEG_SCHEMA, PEG_RULE, PEG_RULE_FN,
    PEG_TRIGGER_RULE, PEG_TRIGGER_RULE_FN, PEG_ATOMIC, PEG_TAG, PEG_GBNF, PEG_AC, PEG_AC_ONE,
    PEG_OP_PLUS, PEG_OP_SHL, PEG_OP_OR, PEG_OP_PLUS_LIT, PEG_OP_SHL_LIT, PEG_OP_OR_LIT, PEG_OP_LIT_PLUS,
    PEG_OP_LIT_SHL, PEG_OP_LIT_OR, PEG_OP_PLUS_ASSIGN, PEG_OP_OR_ASSIGN,
    PEG_REASONING_BLOCK, PEG_REASONING, PEG_CONTENT, PEG_TAG_SAFE, PEG_TOOL, PEG_TOOL_OPEN, PEG_TOOL_CLOSE,
    PEG_TOOL_ID, PEG_TOOL_NAME, PEG_TOOL_ARGS, PEG_TOOL_ARG, PEG_TOOL_ARG_OPEN, PEG_TOOL_ARG_CLOSE,
    PEG_TOOL_ARG_NAME, PEG_TOOL_ARG_VALUE, PEG_TOOL_ARG_STRING_VALUE, PEG_TOOL_ARG_JSON_VALUE, PEG_PERMUTE,
    PEG_PREFIX, PEG_OPTSPACE, PEG_STD_JSON_TOOLS, PEG_STD_CONSTRUCTED_TOOLS, PEG_PY_STYLE_TOOLS,
    PEG_SET_ROOT,
    PEG_OP_COUNT,
    PEG_BODY_END = 0xFE,  // not a call: closes a rule body, u32 slot of what it returned
};

constexpr uint32_t PEG_NO_SLOT = 0xFFFFFFFFu;

// One description being written, and every one written so far.
struct peg_rec_case {
    uint8_t                  kind = 0;  // 0 build_peg_parser, 1 build_chat_peg_parser, 2 build_tagged_peg_parser
    std::string              ops;
    uint32_t                 n_ops = 0;
    std::string              saved;     // the arena's save(), "S:..." or the exception
    std::vector<std::string> inputs;
};

struct peg_rec_state {
    std::vector<peg_rec_case> cases;
    bool                      active = false;  // inside a build_* call

    peg_rec_case & cur() { return cases.back(); }

    void u8(uint8_t v) { cur().ops.push_back((char) v); }
    void u32(uint32_t v) { cur().ops.append((const char *) &v, 4); }
    void i32(int32_t v) { cur().ops.append((const char *) &v, 4); }
    void str(const std::string & s) {
        u32((uint32_t) s.size());
        cur().ops.append(s);
    }
    void strs(const std::vector<std::string> & v) {
        u32((uint32_t) v.size());
        for (const auto & s : v) str(s);
    }
    // writes an opcode; returns the call's slot
    uint32_t op(uint8_t code) {
        u8(code);
        return cur().n_ops++;
    }

    void note_input(const std::string & input) {
        if (cases.empty()) return;
        auto & in = cur().inputs;
        for (const auto & x : in) if (x == input) return;
        in.push_back(input);
    }
};

inline peg_rec_state g_peg_rec;

struct rec_builder;

// common_peg_parser, recorded
struct rec_parser {
    common_peg_parser_id id_;
    uint32_t             slot;
    rec_builder *        b;

    rec_parser(common_peg_parser_id id, uint32_t slot, rec_builder & b) : id_(id), slot(slot), b(&b) {}

    common_peg_parser_id id() const { return id_; }
    operator common_peg_parser_id() const { return id_; }

    rec_parser & operator+=(const rec_parser & other);
    rec_parser & operator|=(const rec_parser & other);
    rec_parser operator+(const rec_parser & other) const;
    rec_parser operator<<(const rec_parser & other) const;
    rec_parser operator|(const rec_parser & other) const;
    rec_parser operator+(const char * s) const { return lit_op(PEG_OP_PLUS_LIT, s); }
    rec_parser operator+(const std::string & s) const { return lit_op(PEG_OP_PLUS_LIT, s); }
    rec_parser operator<<(const char * s) const { return lit_op(PEG_OP_SHL_LIT, s); }
    rec_parser operator<<(const std::string & s) const { return lit_op(PEG_OP_SHL_LIT, s); }
    rec_parser operator|(const char * s) const { return lit_op(PEG_OP_OR_LIT, s); }
    rec_parser operator|(const std::string & s) const { return lit_op(PEG_OP_OR_LIT, s); }

    rec_parser lit_op(uint8_t code, const std::string & s) const;
};

struct rec_builder {
    common_chat_peg_builder & real;

    explicit rec_builder(common_chat_peg_builder & real) : real(real) {}

    common_peg_parser h(const rec_parser & p) { return common_peg_parser(p.id_, real); }
    rec_parser wrap(const common_peg_parser & p, uint32_t slot) { return rec_parser(p.id(), slot, *this); }

    rec_parser nullary(uint8_t code, const std::function<common_peg_parser()> & f) {
        uint32_t s = g_peg_rec.op(code);
        return wrap(f(), s);
    }
    rec_parser unary(uint8_t code, const rec_parser & p, const std::function<common_peg_parser(const common_peg_parser &)> & f) {
        uint32_t s = g_peg_rec.op(code);
        g_peg_rec.u32(p.slot);
        return wrap(f(h(p)), s);
    }
    std::vector<common_peg_parser> hs(const std::vector<rec_parser> & ps) {
        std::vector<common_peg_parser> out;
        for (const auto & p : ps) out.push_back(h(p));
        return out;
    }
    void slots(const std::vector<rec_parser> & ps) {
        g_peg_rec.u32((uint32_t) ps.size());
        for (const auto & p : ps) g_peg_rec.u32(p.slot);
    }

    rec_parser eps() { return nullary(PEG_EPS, [&] { return real.eps(); }); }
    rec_parser start() { return nullary(PEG_START, [&] { return real.start(); }); }
    rec_parser end() { return nullary(PEG_END, [&] { return real.end(); }); }
    rec_parser literal(const std::string & s) {
        uint32_t w = g_peg_rec.op(PEG_LITERAL);
        g_peg_rec.str(s);
        return wrap(real.literal(s), w);
    }
    rec_parser sequence() { return nullary(PEG_SEQ_EMPTY, [&] { return real.sequence(); }); }
    rec_parser sequence(const std::vector<rec_parser> & ps) {
        uint32_t w = g_peg_rec.op(PEG_SEQ);
        slots(ps);
        return wrap(real.sequence(hs(ps)), w);
    }
    rec_parser sequence(std::initializer_list<rec_parser> ps) { return sequence(std::vector<rec_parser>(ps)); }
    rec_parser choice() { return nullary(PEG_CHOICE_EMPTY, [&] { return real.choice(); }); }
    rec_parser choice(const std::vector<rec_parser> & ps) {
        uint32_t w = g_peg_rec.op(PEG_CHOICE);
        slots(ps);
        return wrap(real.choice(hs(ps)), w);
    }
    rec_parser choice(std::initializer_list<rec_parser> ps) { return choice(std::vector<rec_parser>(ps)); }
    rec_parser one_or_more(const rec_parser & p) { return unary(PEG_ONE_OR_MORE, p, [&](const common_peg_parser & x) { return real.one_or_more(x); }); }
    rec_parser zero_or_more(const rec_parser & p) { return unary(PEG_ZERO_OR_MORE, p, [&](const common_peg_parser & x) { return real.zero_or_more(x); }); }
    rec_parser optional(const rec_parser & p) { return unary(PEG_OPTIONAL, p, [&](const common_peg_parser & x) { return real.optional(x); }); }
    rec_parser peek(const rec_parser & p) { return unary(PEG_PEEK, p, [&](const common_peg_parser & x) { return real.peek(x); }); }
    rec_parser negate(const rec_parser & p) { return unary(PEG_NEGATE, p, [&](const common_peg_parser & x) { return real.negate(x); }); }
    rec_parser any() { return nullary(PEG_ANY, [&] { return real.any(); }); }
    rec_parser chars(const std::string & classes, int min = 1, int max = -1) {
        uint32_t w = g_peg_rec.op(PEG_CHARS);
        g_peg_rec.str(classes);
        g_peg_rec.i32(min);
        g_peg_rec.i32(max);
        return wrap(real.chars(classes, min, max), w);
    }
    rec_parser ref(const std::string & name) {
        uint32_t w = g_peg_rec.op(PEG_REF);
        g_peg_rec.str(name);
        return wrap(real.ref(name), w);
    }
    rec_parser space() { return nullary(PEG_SPACE, [&] { return real.space(); }); }
    rec_parser until(const std::string & d) {
        uint32_t w = g_peg_rec.op(PEG_UNTIL);
        g_peg_rec.str(d);
        return wrap(real.until(d), w);
    }
    rec_parser until_one_of(const std::vector<std::string> & ds) {
        uint32_t w = g_peg_rec.op(PEG_UNTIL_ONE_OF);
        g_peg_rec.strs(ds);
        return wrap(real.until_one_of(ds), w);
    }
    rec_parser rest() { return nullary(PEG_REST, [&] { return real.rest(); }); }
    rec_parser repeat(const rec_parser & p, int min, int max) {
        uint32_t w = g_peg_rec.op(PEG_REPEAT);
        g_peg_rec.u32(p.slot);
        g_peg_rec.i32(min);
        g_peg_rec.i32(max);
        return wrap(real.repeat(h(p), min, max), w);
    }
    rec_parser repeat(const rec_parser & p, int n) {
        uint32_t w = g_peg_rec.op(PEG_REPEAT_N);
        g_peg_rec.u32(p.slot);
        g_peg_rec.i32(n);
        return wrap(real.repeat(h(p), n), w);
    }
    rec_parser double_quoted_string() { return nullary(PEG_DQ_STRING, [&] { return real.double_quoted_string(); }); }
    rec_parser single_quoted_string() { return nullary(PEG_SQ_STRING, [&] { return real.single_quoted_string(); }); }
    rec_parser quoted_string() { return nullary(PEG_QUOTED_STRING, [&] { return real.quoted_string(); }); }
    rec_parser string_content(char d) {
        uint32_t w = g_peg_rec.op(PEG_STRING_CONTENT);
        g_peg_rec.u8((uint8_t) d);
        return wrap(real.string_content(d), w);
    }
    rec_parser json() { return nullary(PEG_JSON, [&] { return real.json(); }); }
    rec_parser json_object() { return nullary(PEG_JSON_OBJECT, [&] { return real.json_object(); }); }
    rec_parser json_string() { return nullary(PEG_JSON_STRING, [&] { return real.json_string(); }); }
    rec_parser json_array() { return nullary(PEG_JSON_ARRAY, [&] { return real.json_array(); }); }
    rec_parser json_number() { return nullary(PEG_JSON_NUMBER, [&] { return real.json_number(); }); }
    rec_parser json_bool() { return nullary(PEG_JSON_BOOL, [&] { return real.json_bool(); }); }
    rec_parser json_null() { return nullary(PEG_JSON_NULL, [&] { return real.json_null(); }); }
    rec_parser json_member(const std::string & key, const rec_parser & p) {
        uint32_t w = g_peg_rec.op(PEG_JSON_MEMBER);
        g_peg_rec.str(key);
        g_peg_rec.u32(p.slot);
        return wrap(real.json_member(key, h(p)), w);
    }
    rec_parser python_value() { return nullary(PEG_PY_VALUE, [&] { return real.python_value(); }); }
    rec_parser python_dict() { return nullary(PEG_PY_DICT, [&] { return real.python_dict(); }); }
    rec_parser python_string() { return nullary(PEG_PY_STRING, [&] { return real.python_string(); }); }
    rec_parser python_array() { return nullary(PEG_PY_ARRAY, [&] { return real.python_array(); }); }
    rec_parser python_number() { return nullary(PEG_PY_NUMBER, [&] { return real.python_number(); }); }
    rec_parser python_bool() { return nullary(PEG_PY_BOOL, [&] { return real.python_bool(); }); }
    rec_parser python_null() { return nullary(PEG_PY_NULL, [&] { return real.python_null(); }); }
    rec_parser marker() { return nullary(PEG_MARKER, [&] { return real.marker(); }); }
    rec_parser schema(const rec_parser & p, const std::string & name, const common_json & schema, bool raw = false) {
        uint32_t w = g_peg_rec.op(PEG_SCHEMA);
        g_peg_rec.u32(p.slot);
        g_peg_rec.str(name);
        g_peg_rec.str(schema.dump());
        g_peg_rec.u8(raw);
        return wrap(real.schema(h(p), name, schema, raw), w);
    }

    rec_parser rule(const std::string & name, const rec_parser & p, bool trigger = false) {
        uint32_t w = g_peg_rec.op(PEG_RULE);
        g_peg_rec.str(name);
        g_peg_rec.u32(p.slot);
        g_peg_rec.u8(trigger);
        return wrap(real.rule(name, h(p), trigger), w);
    }
    template <typename F, typename std::enable_if<!std::is_convertible<F, const rec_parser &>::value, int>::type = 0>
    rec_parser rule(const std::string & name, F && f, bool trigger = false) {
        return rule_fn(PEG_RULE_FN, name, std::function<rec_parser()>(std::forward<F>(f)), trigger);
    }
    rec_parser trigger_rule(const std::string & name, const rec_parser & p) {
        uint32_t w = g_peg_rec.op(PEG_TRIGGER_RULE);
        g_peg_rec.str(name);
        g_peg_rec.u32(p.slot);
        return wrap(real.trigger_rule(name, h(p)), w);
    }
    template <typename F, typename std::enable_if<!std::is_convertible<F, const rec_parser &>::value, int>::type = 0>
    rec_parser trigger_rule(const std::string & name, F && f) {
        return rule_fn(PEG_TRIGGER_RULE_FN, name, std::function<rec_parser()>(std::forward<F>(f)), true);
    }
    rec_parser rule_fn(uint8_t code, const std::string & name, const std::function<rec_parser()> & f, bool trigger) {
        uint32_t w = g_peg_rec.op(code);
        g_peg_rec.str(name);
        if (code == PEG_RULE_FN) g_peg_rec.u8(trigger);
        bool called = false;
        auto body = [&]() -> common_peg_parser {
            called = true;
            rec_parser inner = f();
            g_peg_rec.u8(PEG_BODY_END);
            g_peg_rec.u32(inner.slot);
            return h(inner);
        };
        common_peg_parser r = code == PEG_RULE_FN ? real.rule(name, body, trigger) : real.trigger_rule(name, body);
        if (!called) {
            g_peg_rec.u8(PEG_BODY_END);
            g_peg_rec.u32(PEG_NO_SLOT);
        }
        return wrap(r, w);
    }
    rec_parser atomic(const rec_parser & p) { return unary(PEG_ATOMIC, p, [&](const common_peg_parser & x) { return real.atomic(x); }); }
    rec_parser tag(const std::string & t, const rec_parser & p) {
        uint32_t w = g_peg_rec.op(PEG_TAG);
        g_peg_rec.str(t);
        g_peg_rec.u32(p.slot);
        return wrap(real.tag(t, h(p)), w);
    }
    rec_parser gbnf(const rec_parser & p, const std::string & grammar) {
        uint32_t w = g_peg_rec.op(PEG_GBNF);
        g_peg_rec.u32(p.slot);
        g_peg_rec.str(grammar);
        return wrap(real.gbnf(h(p), grammar), w);
    }
    rec_parser ac(const rec_parser & p, const std::vector<std::string> & ds) {
        uint32_t w = g_peg_rec.op(PEG_AC);
        g_peg_rec.u32(p.slot);
        g_peg_rec.strs(ds);
        return wrap(real.ac(h(p), ds), w);
    }
    rec_parser ac(const rec_parser & p, const std::string & d) {
        uint32_t w = g_peg_rec.op(PEG_AC_ONE);
        g_peg_rec.u32(p.slot);
        g_peg_rec.str(d);
        return wrap(real.ac(h(p), d), w);
    }

    // chat
    rec_parser reasoning_block(const rec_parser & p) { return unary(PEG_REASONING_BLOCK, p, [&](const common_peg_parser & x) { return real.reasoning_block(x); }); }
    rec_parser reasoning(const rec_parser & p) { return unary(PEG_REASONING, p, [&](const common_peg_parser & x) { return real.reasoning(x); }); }
    rec_parser content(const rec_parser & p) { return unary(PEG_CONTENT, p, [&](const common_peg_parser & x) { return real.content(x); }); }
    rec_parser tag_with_safe_content(const std::string & tag_name, const std::string & marker, const rec_parser & p) {
        uint32_t w = g_peg_rec.op(PEG_TAG_SAFE);
        g_peg_rec.str(tag_name);
        g_peg_rec.str(marker);
        g_peg_rec.u32(p.slot);
        return wrap(real.tag_with_safe_content(tag_name, marker, h(p)), w);
    }
    rec_parser tool(const rec_parser & p) { return unary(PEG_TOOL, p, [&](const common_peg_parser & x) { return real.tool(x); }); }
    rec_parser tool_open(const rec_parser & p) { return unary(PEG_TOOL_OPEN, p, [&](const common_peg_parser & x) { return real.tool_open(x); }); }
    rec_parser tool_close(const rec_parser & p) { return unary(PEG_TOOL_CLOSE, p, [&](const common_peg_parser & x) { return real.tool_close(x); }); }
    rec_parser tool_id(const rec_parser & p) { return unary(PEG_TOOL_ID, p, [&](const common_peg_parser & x) { return real.tool_id(x); }); }
    rec_parser tool_name(const rec_parser & p) { return unary(PEG_TOOL_NAME, p, [&](const common_peg_parser & x) { return real.tool_name(x); }); }
    rec_parser tool_args(const rec_parser & p) { return unary(PEG_TOOL_ARGS, p, [&](const common_peg_parser & x) { return real.tool_args(x); }); }
    rec_parser tool_arg(const rec_parser & p) { return unary(PEG_TOOL_ARG, p, [&](const common_peg_parser & x) { return real.tool_arg(x); }); }
    rec_parser tool_arg_open(const rec_parser & p) { return unary(PEG_TOOL_ARG_OPEN, p, [&](const common_peg_parser & x) { return real.tool_arg_open(x); }); }
    rec_parser tool_arg_close(const rec_parser & p) { return unary(PEG_TOOL_ARG_CLOSE, p, [&](const common_peg_parser & x) { return real.tool_arg_close(x); }); }
    rec_parser tool_arg_name(const rec_parser & p) { return unary(PEG_TOOL_ARG_NAME, p, [&](const common_peg_parser & x) { return real.tool_arg_name(x); }); }
    rec_parser tool_arg_value(const rec_parser & p) { return unary(PEG_TOOL_ARG_VALUE, p, [&](const common_peg_parser & x) { return real.tool_arg_value(x); }); }
    rec_parser tool_arg_string_value(const rec_parser & p) { return unary(PEG_TOOL_ARG_STRING_VALUE, p, [&](const common_peg_parser & x) { return real.tool_arg_string_value(x); }); }
    rec_parser tool_arg_json_value(const rec_parser & p) { return unary(PEG_TOOL_ARG_JSON_VALUE, p, [&](const common_peg_parser & x) { return real.tool_arg_json_value(x); }); }
    rec_parser permute(const std::string & prefix, const std::vector<rec_parser> & ps) {
        uint32_t w = g_peg_rec.op(PEG_PERMUTE);
        g_peg_rec.str(prefix);
        slots(ps);
        return wrap(real.permute(prefix, hs(ps)), w);
    }
    rec_parser prefix(const std::string & s, const std::string & delimiter = {}) {
        uint32_t w = g_peg_rec.op(PEG_PREFIX);
        g_peg_rec.str(s);
        g_peg_rec.str(delimiter);
        return wrap(real.prefix(s, delimiter), w);
    }
    rec_parser optspace(const std::string & t) {
        uint32_t w = g_peg_rec.op(PEG_OPTSPACE);
        g_peg_rec.str(t);
        return wrap(real.optspace(t), w);
    }
    rec_parser standard_json_tools(const std::string & section_start, const std::string & section_end, const common_json & tools,
                                   bool parallel, bool force, const std::string & name_key = "", const std::string & args_key = "",
                                   bool array_wrapped = false, bool function_is_key = false, const std::string & call_id_key = "",
                                   const std::string & gen_call_id_key = "", const std::vector<std::string> & parameters_order = {},
                                   bool accept_openai_wrapper = false) {
        uint32_t w = g_peg_rec.op(PEG_STD_JSON_TOOLS);
        g_peg_rec.str(section_start);
        g_peg_rec.str(section_end);
        g_peg_rec.str(tools.dump());
        g_peg_rec.u8(parallel);
        g_peg_rec.u8(force);
        g_peg_rec.str(name_key);
        g_peg_rec.str(args_key);
        g_peg_rec.u8(array_wrapped);
        g_peg_rec.u8(function_is_key);
        g_peg_rec.str(call_id_key);
        g_peg_rec.str(gen_call_id_key);
        g_peg_rec.strs(parameters_order);
        g_peg_rec.u8(accept_openai_wrapper);
        return wrap(real.standard_json_tools(section_start, section_end, tools, parallel, force, name_key, args_key, array_wrapped,
                                             function_is_key, call_id_key, gen_call_id_key, parameters_order, accept_openai_wrapper), w);
    }
    rec_parser standard_constructed_tools(const std::map<std::string, std::string> & markers, const common_json & tools, bool parallel, bool force) {
        uint32_t w = g_peg_rec.op(PEG_STD_CONSTRUCTED_TOOLS);
        g_peg_rec.u32((uint32_t) markers.size());
        for (const auto & [k, v] : markers) {
            g_peg_rec.str(k);
            g_peg_rec.str(v);
        }
        g_peg_rec.str(tools.dump());
        g_peg_rec.u8(parallel);
        g_peg_rec.u8(force);
        return wrap(real.standard_constructed_tools(markers, tools, parallel, force), w);
    }
    rec_parser python_style_tool_calls(const common_json & tools, bool parallel, bool allow_json_literals) {
        uint32_t w = g_peg_rec.op(PEG_PY_STYLE_TOOLS);
        g_peg_rec.str(tools.dump());
        g_peg_rec.u8(parallel);
        g_peg_rec.u8(allow_json_literals);
        return wrap(real.python_style_tool_calls(tools, parallel, allow_json_literals), w);
    }

    void set_root(const rec_parser & p) {
        g_peg_rec.op(PEG_SET_ROOT);
        g_peg_rec.u32(p.slot);
        real.set_root(h(p));
    }
};

inline rec_parser rec_bin(uint8_t code, const rec_parser & a, const rec_parser & b,
                          const std::function<common_peg_parser(const common_peg_parser &, const common_peg_parser &)> & f) {
    uint32_t w = g_peg_rec.op(code);
    g_peg_rec.u32(a.slot);
    g_peg_rec.u32(b.slot);
    return a.b->wrap(f(a.b->h(a), a.b->h(b)), w);
}

inline rec_parser & rec_parser::operator+=(const rec_parser & o) {
    *this = rec_bin(PEG_OP_PLUS_ASSIGN, *this, o, [](const common_peg_parser & x, const common_peg_parser & y) {
        common_peg_parser r = x;
        r += y;
        return r;
    });
    return *this;
}
inline rec_parser & rec_parser::operator|=(const rec_parser & o) {
    *this = rec_bin(PEG_OP_OR_ASSIGN, *this, o, [](const common_peg_parser & x, const common_peg_parser & y) {
        common_peg_parser r = x;
        r |= y;
        return r;
    });
    return *this;
}
inline rec_parser rec_parser::operator+(const rec_parser & o) const {
    return rec_bin(PEG_OP_PLUS, *this, o, [](const common_peg_parser & x, const common_peg_parser & y) { return x + y; });
}
inline rec_parser rec_parser::operator<<(const rec_parser & o) const {
    return rec_bin(PEG_OP_SHL, *this, o, [](const common_peg_parser & x, const common_peg_parser & y) { return x << y; });
}
inline rec_parser rec_parser::operator|(const rec_parser & o) const {
    return rec_bin(PEG_OP_OR, *this, o, [](const common_peg_parser & x, const common_peg_parser & y) { return x | y; });
}
inline rec_parser rec_parser::lit_op(uint8_t code, const std::string & s) const {
    uint32_t w = g_peg_rec.op(code);
    g_peg_rec.u32(slot);
    g_peg_rec.str(s);
    common_peg_parser x = b->h(*this);
    switch (code) {
        case PEG_OP_PLUS_LIT: return b->wrap(x + s, w);
        case PEG_OP_SHL_LIT:  return b->wrap(x << s, w);
        default:              return b->wrap(x | s, w);
    }
}

inline rec_parser rec_lit_left(uint8_t code, const std::string & s, const rec_parser & p) {
    uint32_t w = g_peg_rec.op(code);
    g_peg_rec.str(s);
    g_peg_rec.u32(p.slot);
    common_peg_parser x = p.b->h(p);
    switch (code) {
        case PEG_OP_LIT_PLUS: return p.b->wrap(s + x, w);
        case PEG_OP_LIT_SHL:  return p.b->wrap(s << x, w);
        default:              return p.b->wrap(s | x, w);
    }
}
inline rec_parser operator+(const char * s, const rec_parser & p) { return rec_lit_left(PEG_OP_LIT_PLUS, s, p); }
inline rec_parser operator+(const std::string & s, const rec_parser & p) { return rec_lit_left(PEG_OP_LIT_PLUS, s, p); }
inline rec_parser operator<<(const char * s, const rec_parser & p) { return rec_lit_left(PEG_OP_LIT_SHL, s, p); }
inline rec_parser operator<<(const std::string & s, const rec_parser & p) { return rec_lit_left(PEG_OP_LIT_SHL, s, p); }
inline rec_parser operator|(const char * s, const rec_parser & p) { return rec_lit_left(PEG_OP_LIT_OR, s, p); }
inline rec_parser operator|(const std::string & s, const rec_parser & p) { return rec_lit_left(PEG_OP_LIT_OR, s, p); }

inline std::string rec_save(const common_peg_arena & a) {
    try {
        return "S:" + a.save();
    } catch (const std::exception & e) {
        return std::string("E:") + e.what();
    }
}

template <typename F> common_peg_arena rec_build(uint8_t kind, F && fn) {
    g_peg_rec.cases.emplace_back();
    g_peg_rec.cur().kind = kind;
    common_chat_peg_builder real;
    rec_builder b(real);
    b.set_root(fn(b));
    auto arena = real.build();
    g_peg_rec.cur().saved = rec_save(arena);
    return arena;
}

template <typename F> common_peg_arena rec_build_peg_parser(F && fn) { return rec_build(0, std::forward<F>(fn)); }
template <typename F> common_peg_arena rec_build_chat_peg_parser(F && fn) { return rec_build(1, std::forward<F>(fn)); }

// tagged_peg_parser, its inputs noted
struct rec_tagged_peg_parser {
    tagged_peg_parser real;

    rec_tagged_peg_parser & withDebug() { real.withDebug(); return *this; }
    rec_tagged_peg_parser & withoutDebug() { real.withoutDebug(); return *this; }

    tagged_parse_result parse_and_extract(const std::string & input, common_peg_parse_flags extra = COMMON_PEG_PARSE_FLAG_NONE) const {
        g_peg_rec.note_input(input);
        return real.parse_and_extract(input, extra);
    }
    tagged_parse_result parse_anywhere_and_extract(const std::string & input) const {
        g_peg_rec.note_input(input);
        return real.parse_anywhere_and_extract(input);
    }
};

template <typename F> rec_tagged_peg_parser rec_build_tagged_peg_parser(F && fn) {
    return rec_tagged_peg_parser{ tagged_peg_parser{ rec_build(2, std::forward<F>(fn)) } };
}

// common_peg_parse_context, its input noted
struct rec_parse_context : common_peg_parse_context {
    rec_parse_context(common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE) : common_peg_parse_context(flags) {}
    rec_parse_context(const std::string & input, common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE)
        : common_peg_parse_context(input, flags) {
        g_peg_rec.note_input(input);
    }
};

#ifndef PEG_REC_NO_RENAME
#define common_peg_parser rec_parser
#define common_peg_parser_builder rec_builder
#define common_chat_peg_builder rec_builder
#define build_peg_parser rec_build_peg_parser
#define build_chat_peg_parser rec_build_chat_peg_parser
#define build_tagged_peg_parser rec_build_tagged_peg_parser
#define tagged_peg_parser rec_tagged_peg_parser
#define common_peg_parse_context rec_parse_context
#endif
