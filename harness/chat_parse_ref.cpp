// Upstream's chat templates and output parsing behind a record driver, for
// scripts/chat-parse-diff.
//
// Not a port. The reference half of the oracle for `cli/common_chat/`'s
// templates, auto-parser, per-model parsers and parse: it links upstream's
// `llama-common` (`common/chat.cpp`, `chat-diff-analyzer.cpp`,
// `chat-auto-parser-*.cpp`, `parsers/*.cpp`, `jinja/`, all unmodified) and
// replays each record through the calls the port reproduces, writing one
// transcript; `harness/chat_parse_diff.zig` writes the same transcript
// through the port. Line for line they must agree, after the script
// normalises the text of a Jinja engine failure (the engines word them
// differently).
//
// Input: one JSON record per line (see scripts/chat-parse-diff):
//   tmpl    {id, src, bos, eos}          common_chat_templates_init, its caps,
//                                        source, flags, example, verify, and
//                                        the differential analysis of the template
//   apply   {t, in, keep?}               common_chat_templates_apply
//   parse   {a, input_hex, partial}      common_chat_peg_parse with the parser
//                                        and parser params apply record `a`
//                                        produced (a saved parser cannot be
//                                        loaded since v0.7.0)
//   diff, ucp, acs, seg, cv              the auto-parser helpers
// Output: per record `== <n> <op>`, then `key value` lines, every string
// escaped onto one line; `E <class> <what>` for an exception; `L <level>
// <text>` for every INF/WRN/ERR line common_log was given.
//
// Both clocks are pinned (`time`, `clock_gettime(CLOCK_REALTIME)`), so
// `strftime_now` and `common_chat_extra_context`'s dates cannot move
// between the two sides.

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "json.h"
#include "llama.h"
#include "log.h"
#include "peg-parser.h"
#include "peg_save.h"

#include <dlfcn.h>
#include <time.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define FIXED_TIME ((time_t) 1700000000)

extern "C" time_t time(time_t * t) {
    if (t) {
        *t = FIXED_TIME;
    }
    return FIXED_TIME;
}

extern "C" int clock_gettime(clockid_t id, struct timespec * ts) {
    if (id == CLOCK_REALTIME) {
        ts->tv_sec  = FIXED_TIME;
        ts->tv_nsec = 0;
        return 0;
    }
    using fn_t      = int (*)(clockid_t, struct timespec *);
    static fn_t real = (fn_t) dlsym(RTLD_NEXT, "clock_gettime");
    return real(id, ts);
}

static FILE * out = nullptr;
static std::string log_path;
static size_t log_seen = 0;

static std::string esc(const std::string & s) {
    std::string r;
    r.reserve(s.size() + 2);
    for (unsigned char c : s) {
        if (c == '\\') {
            r += "\\\\";
        } else if (c < 0x20 || c >= 0x7f) {
            char buf[5];
            snprintf(buf, sizeof(buf), "\\x%02x", c);
            r += buf;
        } else {
            r += (char) c;
        }
    }
    return r;
}

static void line(const std::string & s) {
    fwrite(s.data(), 1, s.size(), out);
    fputc('\n', out);
}

static void kv(const char * key, const std::string & s) {
    line(std::string(key) + " " + esc(s));
}

static void flag(const char * key, bool b) {
    line(std::string(key) + " " + (b ? "1" : "0"));
}

template <typename F> static void guarded(F f) {
    try {
        f();
    } catch (const common_json_error & e) {
        line("E json_error " + esc(e.what()));
    } catch (const std::invalid_argument & e) {
        line("E invalid_argument " + esc(e.what()));
    } catch (const std::exception & e) {
        line("E runtime_error " + esc(e.what()));
    }
}

// Every common_log line written since the last call, as `L <level> <text>`.
static void flush_logs() {
    common_log_flush(common_log_main());
    std::ifstream f(log_path, std::ios::binary);
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (all.size() <= log_seen) {
        return;
    }
    std::string fresh = all.substr(log_seen);
    log_seen          = all.size();
    size_t pos        = 0;
    while (pos < fresh.size()) {
        size_t nl = fresh.find('\n', pos);
        if (nl == std::string::npos) {
            nl = fresh.size();
        }
        std::string one = fresh.substr(pos, nl - pos);
        pos             = nl + 1;
        if (one.empty()) {
            continue;
        }
        auto        j     = common_json::parse(one);
        std::string level = j.at("level").get<std::string>();
        if (level == "debug") {
            continue;
        }
        const char * tag = level == "info" ? "I" : level == "warn" ? "W" : level == "error" ? "E" : "?";
        line(std::string("L ") + tag + " " + esc(j.at("msg").get<std::string>()));
    }
}

static std::string unhex(const std::string & h) {
    std::string s;
    s.reserve(h.size() / 2);
    for (size_t i = 0; i + 1 < h.size(); i += 2) {
        s += (char) std::stoi(h.substr(i, 2), nullptr, 16);
    }
    return s;
}

static common_chat_msg read_msg(const common_json & j) {
    common_chat_msg m;
    m.role    = j.at("role").get<std::string>();
    m.content = j.at("content").get<std::string>();
    for (const auto & p : j.at("parts")) {
        m.content_parts.push_back({ p.at((size_t) 0).get<std::string>(), p.at((size_t) 1).get<std::string>(),
                                    p.size() > 2 ? p.at((size_t) 2).get<std::string>() : std::string() });
    }
    for (const auto & tc : j.at("tool_calls")) {
        m.tool_calls.push_back({ tc.at((size_t) 0).get<std::string>(), tc.at((size_t) 1).get<std::string>(),
                                 tc.at((size_t) 2).get<std::string>() });
    }
    m.reasoning_content = j.at("reasoning_content").get<std::string>();
    m.tool_name         = j.at("tool_name").get<std::string>();
    m.tool_call_id      = j.at("tool_call_id").get<std::string>();
    if (j.contains("extra_json")) {
        m.extra_json = j.at("extra_json").get<std::string>();
    }
    return m;
}

static common_chat_templates_inputs read_inputs(const common_json & j) {
    common_chat_templates_inputs in;
    for (const auto & m : j.at("messages")) {
        in.messages.push_back(read_msg(m));
    }
    in.grammar                = j.at("grammar").get<std::string>();
    in.json_schema            = j.at("json_schema").get<std::string>();
    in.add_generation_prompt  = j.at("add_generation_prompt").get<bool>();
    in.continue_final_message = (common_chat_continuation) j.at("continue").get<int>();
    in.use_jinja              = j.at("use_jinja").get<bool>();
    for (const auto & t : j.at("tools")) {
        in.tools.push_back({ t.at((size_t) 0).get<std::string>(), t.at((size_t) 1).get<std::string>(),
                             t.at((size_t) 2).get<std::string>(), t.size() > 3 && t.at((size_t) 3).get<bool>() });
    }
    in.tool_choice         = (common_chat_tool_choice) j.at("tool_choice").get<int>();
    in.parallel_tool_calls = j.at("parallel").get<bool>();
    in.reasoning_format    = (common_reasoning_format) j.at("reasoning_format").get<int>();
    in.enable_thinking     = j.at("enable_thinking").get<bool>();
    for (const auto & kv : j.at("kwargs")) {
        in.chat_template_kwargs[kv.at((size_t) 0).get<std::string>()] = kv.at((size_t) 1).get<std::string>();
    }
    in.add_bos            = j.at("add_bos").get<bool>();
    in.add_eos            = j.at("add_eos").get<bool>();
    in.force_pure_content = j.at("force_pure_content").get<bool>();
    return in;
}

static template_params read_template_params(const common_json & j) {
    template_params p;
    p.messages              = j.at("messages");
    p.tools                 = j.at("tools");
    p.add_generation_prompt = j.at("add_generation_prompt").get<bool>();
    p.enable_thinking       = j.at("enable_thinking").get<bool>();
    if (j.contains("extra_context")) {
        p.extra_context = j.at("extra_context");
    }
    return p;
}

// The prompt: its text, then every part as upstream produced it, an empty
// one included.
static void show_prompt(const common_chat_prompt & p) {
    kv("prompt", p.str());
    for (const auto & part : p.parts) {
        kv(part.is_input ? "prompt.input" : "prompt.template", part.text);
    }
}

static void show_params(const common_chat_params & p) {
    guarded([&] { kv("format", common_chat_format_name(p.format)); });
    show_prompt(p.prompt);
    kv("grammar", p.grammar);
    flag("lazy", p.grammar_lazy);
    kv("gen", p.generation_prompt);
    flag("thinking", p.supports_thinking);
    kv("think_start", p.thinking_start_tag);
    for (const auto & t : p.thinking_end_tags) {
        kv("think_end", t);
    }
    for (const auto & t : p.grammar_triggers) {
        line("trigger " + std::to_string((int) t.type) + " " + std::to_string(t.token) + " " + esc(t.value));
    }
    for (const auto & t : p.preserved_tokens) {
        kv("preserved", t);
    }
    for (const auto & t : p.additional_stops) {
        kv("stop", t);
    }
    guarded([&] { kv("parser", p.parser.empty() ? std::string() : harness_peg_save(p.parser)); });
    // the form `common_chat_msg_delimiters::to_json` wrote until v0.7.0 removed it
    guarded([&] {
        common_json d = common_json::array();
        for (const auto & x : p.message_delimiters.delimiters) {
            d.push_back({ { "role", common_chat_role_to_string(x.role) }, { "delimiter", x.delimiter } });
        }
        kv("delims", d.dump());
    });
}

static void show_msg(const common_chat_msg & m) {
    kv("role", m.role);
    kv("content", m.content);
    for (const auto & p : m.content_parts) {
        kv("part.type", p.type);
        kv("part.text", p.text);
    }
    for (const auto & tc : m.tool_calls) {
        kv("tc.name", tc.name);
        kv("tc.arguments", tc.arguments);
        kv("tc.id", tc.id);
    }
    kv("reasoning", m.reasoning_content);
    kv("tool_name", m.tool_name);
    kv("tool_call_id", m.tool_call_id);
}

static void show_analysis(const autoparser::autoparser & a) {
    auto mode = [](auto m) {
        std::ostringstream os;
        os << m;
        return os.str();
    };
    kv("an.user_start", a.user_start);
    kv("an.assistant_start", a.assistant_start);
    kv("an.reasoning.mode", mode(a.reasoning.mode));
    kv("an.reasoning.start", a.reasoning.start);
    kv("an.reasoning.end", a.reasoning.end);
    kv("an.content.mode", mode(a.content.mode));
    kv("an.content.start", a.content.start);
    kv("an.content.end", a.content.end);
    const auto & f = a.tools.format;
    kv("an.tools.mode", mode(f.mode));
    kv("an.tools.section_start", f.section_start);
    kv("an.tools.section_end", f.section_end);
    kv("an.tools.per_call_start", f.per_call_start);
    kv("an.tools.per_call_end", f.per_call_end);
    flag("an.tools.fun_name_is_key", f.fun_name_is_key);
    flag("an.tools.tools_array_wrapped", f.tools_array_wrapped);
    flag("an.tools.openai_wrapper_trigger", f.openai_wrapper_trigger);
    kv("an.tools.function_field", f.function_field);
    kv("an.tools.name_field", f.name_field);
    kv("an.tools.args_field", f.args_field);
    kv("an.tools.id_field", f.id_field);
    kv("an.tools.gen_id_field", f.gen_id_field);
    for (const auto & p : f.parameter_order) {
        kv("an.tools.parameter_order", p);
    }
    const auto & fn = a.tools.function;
    kv("an.function.name_prefix", fn.name_prefix);
    kv("an.function.name_suffix", fn.name_suffix);
    kv("an.function.args_separator", fn.args_separator);
    kv("an.function.close", fn.close);
    const auto & ar = a.tools.arguments;
    kv("an.args.start", ar.start);
    kv("an.args.end", ar.end);
    kv("an.args.name_prefix", ar.name_prefix);
    kv("an.args.name_suffix", ar.name_suffix);
    kv("an.args.value_prefix", ar.value_prefix);
    kv("an.args.value_suffix", ar.value_suffix);
    kv("an.args.separator", ar.separator);
    flag("an.args.tolerate_intertag_whitespace", ar.tolerate_intertag_whitespace);
    kv("an.call_id.pos", mode(a.tools.call_id.pos));
    kv("an.call_id.prefix", a.tools.call_id.prefix);
    kv("an.call_id.suffix", a.tools.call_id.suffix);
    for (const auto & t : a.preserved_tokens) {
        kv("an.preserved", t);
    }
    for (const auto & t : a.additional_stops) {
        kv("an.stop", t);
    }
}

static void show_diff(const diff_split & d) {
    kv("prefix", d.prefix);
    kv("suffix", d.suffix);
    kv("left", d.left);
    kv("right", d.right);
}

struct state {
    std::map<int, common_chat_templates_ptr>  tmpls;
    std::map<int, common_chat_params>         applies;
};

static void run(state & st, size_t n, const common_json & r) {
    const std::string op = r.at("op").get<std::string>();
    line("== " + std::to_string(n) + " " + op);
    if (op == "tmpl") {
        int         id  = r.at("id").get<int>();
        std::string src = r.at("src").get<std::string>();
        std::string bos = r.at("bos").get<std::string>();
        std::string eos = r.at("eos").get<std::string>();
        // With a "model", the vocabulary's BOS, EOS and add-BOS/EOS flags
        // replace the given ones, as they do for a loaded model.
        llama_model * model = nullptr;
        if (r.contains("model")) {
            static std::map<std::string, llama_model *> models;
            const std::string path = r.at("model").get<std::string>();
            if (!models.count(path)) {
                llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
                llama_model_params mp = llama_model_default_params();
                mp.vocab_only = true;
                models[path] = llama_model_load_from_file(path.c_str(), mp);
            }
            model = models[path];
        }
        try {
            st.tmpls[id] = common_chat_templates_init(model, src, bos, eos);
        } catch (const std::exception &) {
            line("init_failed");
            flush_logs();
            return;
        }
        const auto * t = st.tmpls[id].get();
        const common_json caps = common_chat_templates_get_caps(t);
        for (const auto & [k, v] : caps.items()) {
            if (v.is_boolean()) {
                line("caps " + k + "=" + (v.get<bool>() ? "1" : "0"));
            }
        }
        kv("caps_json", caps.dump());
        kv("source", common_chat_templates_source(t));
        kv("tool_use_source", common_chat_templates_source(t, "tool_use"));
        flag("explicit", common_chat_templates_was_explicit(t));
        for (const char * language : { "", "fr", "FR", "xx" }) {
            auto asr = common_chat_get_asr_prompt(t, language);
            kv("asr.system", asr.system);
            kv("asr.user", asr.user);
            flag("asr.has_language", asr.has_language);
        }
        guarded([&] { kv("tts", common_chat_get_tts_prompt(t, "Say this")); });
        guarded([&] { kv("tts", common_chat_get_tts_prompt(t, "Say \"that\"", "fr", "alice")); });
        guarded([&] { flag("enable_thinking", common_chat_templates_support_enable_thinking(t)); });
        guarded([&] { kv("example", common_chat_format_example(t, true, {})); });
        flag("verify", common_chat_verify_template(src, true));
        guarded([&] {
            common_chat_template tmpl(src, bos, eos);
            autoparser::autoparser a;
            a.analyze_template(tmpl);
            show_analysis(a);
        });
    } else if (op == "apply") {
        int  t    = r.at("t").get<int>();
        bool keep = r.contains("keep") && r.at("keep").get<bool>();
        auto it   = st.tmpls.find(t);
        if (it == st.tmpls.end()) {
            line("no_templates");
        } else {
            guarded([&] {
                auto in     = read_inputs(r.at("in"));
                auto params = common_chat_templates_apply(it->second.get(), in);
                show_params(params);
                if (keep) {
                    st.applies[(int) n] = params;
                }
            });
        }
    } else if (op == "parse") {
        std::string input = unhex(r.at("input_hex").get<std::string>());
        bool        partial = r.at("partial").get<bool>();
        auto it = st.applies.find(r.at("a").get<int>());
        if (it == st.applies.end()) {
            line("no_apply");
            flush_logs();
            return;
        }
        common_chat_parser_params params(it->second);
        guarded([&] { show_msg(common_chat_peg_parse(it->second.parser, common_chat_input(input), partial, params)); });
    } else if (op == "diff") {
        show_diff(calculate_diff_split(r.at("l").get<std::string>(), r.at("r").get<std::string>()));
    } else if (op == "ucp") {
        kv("r", until_common_prefix(r.at("full").get<std::string>(), r.at("l").get<std::string>(), r.at("r").get<std::string>()));
    } else if (op == "acs") {
        kv("r", after_common_suffix(r.at("full").get<std::string>(), r.at("l").get<std::string>(), r.at("r").get<std::string>()));
    } else if (op == "seg") {
        for (const auto & s : segmentize_markers(r.at("text").get<std::string>())) {
            kv(s.type == segment_type::TEXT ? "TEXT" : "MARKER", s.value);
        }
    } else if (op == "cv") {
        guarded([&] {
            common_chat_template tmpl(r.at("src").get<std::string>(), r.at("bos").get<std::string>(),
                                      r.at("eos").get<std::string>());
            auto pa  = read_template_params(r.at("a"));
            auto pb  = read_template_params(r.at("b"));
            auto res = autoparser::compare_variants(tmpl, pa, [&](template_params & p) { p = pb; });
            if (!res) {
                line("none");
                return;
            }
            show_diff(res->diff);
            kv("output_A", res->output_A);
            kv("output_B", res->output_B);
        });
    } else {
        line("unknown_op");
    }
    flush_logs();
}

// Whether a record before the start of a resumed run rebuilds state the
// records after it read: templates, kept applies.
static bool builds_state(const common_json & r) {
    std::string op = r.at("op").get<std::string>();
    return op == "tmpl" || (op == "apply" && r.contains("keep"));
}

// Usage: chat_parse_ref <records> <transcript> <log scratch file> [start].
// From `start` the transcript is appended to, the records before it replayed
// without output where they build state; CHAT_PARSE_SKIP lists records
// (comma-separated) a previous run crashed on, never run again. Each record
// is flushed as it finishes, so a crash leaves every finished record behind.
int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s <records> <transcript> <log scratch file> [start]\n", argv[0]);
        return 2;
    }
    size_t start = argc == 5 ? (size_t) std::stoull(argv[4]) : 0;
    std::set<size_t> skip;
    if (const char * list = getenv("CHAT_PARSE_SKIP")) {
        std::string s = list;
        size_t      pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos) {
                comma = s.size();
            }
            if (comma > pos) {
                skip.insert((size_t) std::stoull(s.substr(pos, comma - pos)));
            }
            pos = comma + 1;
        }
    }
    FILE * real_out = fopen(argv[2], start == 0 ? "wb" : "ab");
#ifdef _WIN32
    FILE * null_out = fopen("NUL", "wb");
#else
    FILE * null_out = fopen("/dev/null", "wb");
#endif
    log_path = argv[3];
    common_log_set_jsonl(true);
    common_log_set_colors(common_log_main(), LOG_COLORS_DISABLED);
    common_log_set_prefix(common_log_main(), false);
    common_log_set_timestamps(common_log_main(), false);
    common_log_set_file(common_log_main(), log_path.c_str());

    std::ifstream in(argv[1], std::ios::binary);
    std::string   text;
    state         st;
    size_t        n = 0;
    for (; std::getline(in, text); n++) {
        if (text.empty() || skip.count(n)) {
            continue;
        }
        auto r = common_json::parse(text);
        if (n < start && !builds_state(r)) {
            continue;
        }
        out = n < start ? null_out : real_out;
        run(st, n, r);
        fflush(out);
    }
    fclose(real_out);
    common_log_flush(common_log_main());
    return 0;
}
