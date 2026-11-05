// A recording shim over upstream's chat API, for scripts/chat-parse-diff.
//
// Not a port. Upstream's chat tests (tests/test-chat.cpp,
// tests/test-chat-template.cpp, tests/test-chat-auto-parser.cpp) build their
// cases in C++ -- conversations, tools, parser parameters -- so there is no
// data to hand over. This header is force-included ahead of each of them
// (`-include harness/chat_parse_rec.h`): it pulls in the real headers first,
// then renames, for the rest of the test's own translation unit only,
// `common_chat_templates_init`, `common_chat_templates_apply`,
// `common_chat_parse`, `common_chat_peg_parse`, and the auto-parser helpers
// `calculate_diff_split`, `until_common_prefix`, `after_common_suffix`,
// `segmentize_markers` and `compare_variants`, and the `common_chat_session`
// class, to the `rec_*` wrappers below. Each calls the real function and appends what it was given, as one
// JSON line, to the file named by `CHAT_PARSE_REC`: the records
// `harness/chat_parse_ref.cpp` and `harness/chat_parse_diff.zig` both replay.
// The tests still run against the real code and must still pass.
//
// A parse's input is written as hex (`input_hex`): a test may cut one
// mid-character, and a record must carry the bytes, not their JSON.

#pragma once

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "log.h"
#include "peg-parser.h"
#include "peg_save.h"

#if __has_include("server-chat.h")
#include "server-chat.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace chat_parse_rec {

inline FILE * out() {
    static FILE * f = [] {
        const char * path = std::getenv("CHAT_PARSE_REC");
        return path ? std::fopen(path, "a") : nullptr;
    }();
    return f;
}

inline void write(const common_json & j) {
    FILE * f = out();
    if (!f) {
        return;
    }
    std::string s = j.dump_safe();
    std::fwrite(s.data(), 1, s.size(), f);
    std::fputc('\n', f);
    std::fflush(f);
}

inline std::string hex(const std::string & s) {
    static const char * digits = "0123456789abcdef";
    std::string h;
    h.reserve(s.size() * 2);
    for (unsigned char c : s) {
        h += digits[c >> 4];
        h += digits[c & 15];
    }
    return h;
}

inline int & next_id() {
    static int id = 0;
    return id;
}

inline std::map<const common_chat_templates *, int> & ids() {
    static std::map<const common_chat_templates *, int> m;
    return m;
}

inline common_json msg(const common_chat_msg & m) {
    common_json j = common_json::object();
    j["role"]    = m.role;
    j["content"] = m.content;
    common_json parts = common_json::array();
    for (const auto & p : m.content_parts) {
        parts.push_back(common_json::array({ p.type, p.text }));
    }
    j["parts"] = parts;
    common_json calls = common_json::array();
    for (const auto & tc : m.tool_calls) {
        calls.push_back(common_json::array({ tc.name, tc.arguments, tc.id }));
    }
    j["tool_calls"]        = calls;
    j["reasoning_content"] = m.reasoning_content;
    j["tool_name"]         = m.tool_name;
    j["tool_call_id"]      = m.tool_call_id;
    return j;
}

inline common_json inputs(const common_chat_templates_inputs & in) {
    common_json j = common_json::object();
    common_json msgs = common_json::array();
    for (const auto & m : in.messages) {
        msgs.push_back(msg(m));
    }
    j["messages"]              = msgs;
    j["grammar"]               = in.grammar;
    j["json_schema"]           = in.json_schema;
    j["add_generation_prompt"] = in.add_generation_prompt;
    j["continue"]              = (int) in.continue_final_message;
    j["use_jinja"]             = in.use_jinja;
    common_json tools = common_json::array();
    for (const auto & t : in.tools) {
        tools.push_back(common_json::array({ t.name, t.description, t.parameters }));
    }
    j["tools"]            = tools;
    j["tool_choice"]      = (int) in.tool_choice;
    j["parallel"]         = in.parallel_tool_calls;
    j["reasoning_format"] = (int) in.reasoning_format;
    j["enable_thinking"]  = in.enable_thinking;
    common_json kwargs = common_json::array();
    for (const auto & [k, v] : in.chat_template_kwargs) {
        kwargs.push_back(common_json::array({ k, v }));
    }
    j["kwargs"]             = kwargs;
    j["add_bos"]            = in.add_bos;
    j["add_eos"]            = in.add_eos;
    j["force_pure_content"] = in.force_pure_content;
    return j;
}

inline common_json template_params(const ::template_params & p) {
    common_json j = common_json::object();
    j["messages"]              = p.messages;
    j["tools"]                 = p.tools;
    j["add_generation_prompt"] = p.add_generation_prompt;
    j["enable_thinking"]       = p.enable_thinking;
    if (p.extra_context) {
        j["extra_context"] = *p.extra_context;
    }
    return j;
}

inline std::string saved(const common_peg_arena & arena) {
    return arena.empty() ? std::string() : harness_peg_save(arena);
}

inline void parse_record(const std::string & parser, common_chat_format format, const std::string & gen,
                         const std::string & input, bool is_partial) {
    common_json j = common_json::object();
    j["op"]        = "parse";
    j["format"]    = (int) format;
    j["gen"]       = gen;
    j["parser"]    = parser;
    j["input_hex"] = hex(input);
    j["partial"]   = is_partial;
    write(j);
}

inline void parse_record(const common_peg_arena & arena, const common_chat_input & input, bool is_partial,
                         const common_chat_parser_params & params) {
    parse_record(saved(arena), params.format, params.generation_prompt.text, input.text, is_partial);
}

}  // namespace chat_parse_rec

inline common_chat_templates_ptr rec_common_chat_templates_init(const struct llama_model * model,
                                                                const std::string &        tmpl,
                                                                const std::string &        bos = "",
                                                                const std::string &        eos = "") {
    int id = chat_parse_rec::next_id()++;
    if (!model) {
        common_json j = common_json::object();
        j["op"]  = "tmpl";
        j["id"]  = id;
        j["src"] = tmpl;
        j["bos"] = bos;
        j["eos"] = eos;
        chat_parse_rec::write(j);
    }
    auto ptr = common_chat_templates_init(model, tmpl, bos, eos);
    if (!model) {
        chat_parse_rec::ids()[ptr.get()] = id;
    }
    return ptr;
}

inline common_chat_params rec_common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                                          const struct common_chat_templates_inputs & inputs) {
    auto it = chat_parse_rec::ids().find(tmpls);
    if (it == chat_parse_rec::ids().end()) {
        return common_chat_templates_apply(tmpls, inputs);
    }
    common_json j = common_json::object();
    j["op"] = "apply";
    j["t"]  = it->second;
    j["in"] = chat_parse_rec::inputs(inputs);
    try {
        auto params = common_chat_templates_apply(tmpls, inputs);
        // which apply a parser came from, for the script to pair parses with it
        j["parser_out"] = chat_parse_rec::saved(params.parser);
        j["format_out"] = (int) params.format;
        j["gen_out"]    = params.generation_prompt;
        chat_parse_rec::write(j);
        return params;
    } catch (...) {
        chat_parse_rec::write(j);
        throw;
    }
}

inline common_chat_msg rec_common_chat_peg_parse(const common_peg_arena & src_parser, const common_chat_input & input,
                                                 bool is_partial, const common_chat_parser_params & params) {
    chat_parse_rec::parse_record(src_parser, input, is_partial, params);
    return common_chat_peg_parse(src_parser, input, is_partial, params);
}

inline common_chat_msg rec_common_chat_parse(const common_chat_input & input, bool is_partial,
                                             const common_chat_parser_params & params) {
    chat_parse_rec::parse_record(params.parser, input, is_partial, params);
    return common_chat_parse(input, is_partial, params);
}

// A chat session (v0.7.0) applies and parses inside chat.cpp, where nothing
// is renamed: its constructor is recorded as the apply it makes, and every
// feed and finish as the parse of the reply so far it makes -- the
// constructor's own parse of a continued message's prefill included.
class rec_common_chat_session : public common_chat_session {
  public:
    rec_common_chat_session() = default;

    rec_common_chat_session(const common_chat_templates *        tmpls,
                            const llama_vocab *                  vocab,
                            const common_chat_templates_inputs & inputs,
                            const common_chat_session_params &   params = {})
    try : common_chat_session(tmpls, vocab, inputs, params) {
        auto it = chat_parse_rec::ids().find(tmpls);
        if (it == chat_parse_rec::ids().end()) {
            return;
        }
        recorded = true;
        parser_text = chat_parse_rec::saved(parser());
        common_json j = common_json::object();
        j["op"]         = "apply";
        j["t"]          = it->second;
        j["in"]         = chat_parse_rec::inputs(inputs);
        j["parser_out"] = parser_text;
        j["format_out"] = (int) format();
        j["gen_out"]    = generation_prompt();
        chat_parse_rec::write(j);
        if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !params.echo) {
            chat_parse_rec::parse_record(parser_text, format(), generation_prompt(), "", true);
        }
    } catch (...) {
        auto it = chat_parse_rec::ids().find(tmpls);
        if (it != chat_parse_rec::ids().end()) {
            common_json j = common_json::object();
            j["op"] = "apply";
            j["t"]  = it->second;
            j["in"] = chat_parse_rec::inputs(inputs);
            chat_parse_rec::write(j);
        }
        throw;
    }

    const common_chat_msg & feed(const common_chat_input & chunk) {
        reply += chunk.text;
        if (recorded) {
            chat_parse_rec::parse_record(parser_text, format(), generation_prompt(), reply, true);
        }
        return common_chat_session::feed(chunk);
    }

    const common_chat_msg & finish(const common_chat_input & chunk = {}) {
        reply += chunk.text;
        if (recorded) {
            chat_parse_rec::parse_record(parser_text, format(), generation_prompt(), reply, false);
        }
        return common_chat_session::finish(chunk);
    }

  private:
    bool        recorded = false;
    std::string parser_text;
    std::string reply;
};

inline diff_split rec_calculate_diff_split(const std::string & left, const std::string & right) {
    common_json j = common_json::object();
    j["op"] = "diff";
    j["l"]  = left;
    j["r"]  = right;
    chat_parse_rec::write(j);
    return calculate_diff_split(left, right);
}

inline std::string rec_until_common_prefix(const std::string & full, const std::string & left, const std::string & right) {
    common_json j = common_json::object();
    j["op"]   = "ucp";
    j["full"] = full;
    j["l"]    = left;
    j["r"]    = right;
    chat_parse_rec::write(j);
    return until_common_prefix(full, left, right);
}

inline std::string rec_after_common_suffix(const std::string & full, const std::string & left, const std::string & right) {
    common_json j = common_json::object();
    j["op"]   = "acs";
    j["full"] = full;
    j["l"]    = left;
    j["r"]    = right;
    chat_parse_rec::write(j);
    return after_common_suffix(full, left, right);
}

inline std::vector<segment> rec_segmentize_markers(const std::string & text) {
    common_json j = common_json::object();
    j["op"]   = "seg";
    j["text"] = text;
    chat_parse_rec::write(j);
    return segmentize_markers(text);
}

inline std::optional<compare_variants_result> rec_compare_variants(const common_chat_template &                   tmpl,
                                                                   const template_params &                        params_A,
                                                                   const std::function<void(template_params &)> & modifier) {
    template_params params_B = params_A;
    if (modifier) {
        modifier(params_B);
    }
    common_json j = common_json::object();
    j["op"]  = "cv";
    j["src"] = tmpl.source();
    j["bos"] = tmpl.bos_token();
    j["eos"] = tmpl.eos_token();
    j["a"]   = chat_parse_rec::template_params(params_A);
    j["b"]   = chat_parse_rec::template_params(params_B);
    chat_parse_rec::write(j);
    return autoparser::compare_variants(tmpl, params_A, modifier);
}

#define common_chat_templates_init  rec_common_chat_templates_init
#define common_chat_templates_apply rec_common_chat_templates_apply
#define common_chat_peg_parse       rec_common_chat_peg_parse
#define common_chat_parse           rec_common_chat_parse
#define calculate_diff_split        rec_calculate_diff_split
#define until_common_prefix         rec_until_common_prefix
#define after_common_suffix         rec_after_common_suffix
#define segmentize_markers          rec_segmentize_markers
#define compare_variants            rec_compare_variants
#define common_chat_session         rec_common_chat_session
