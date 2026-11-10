// Not a port: upstream's chat-template engine behind a C ABI, for
// `harness/jinja_diff.zig`, the harness behind `scripts/jinja-diff`.
//
// `llama.cpp/common/jinja` is the Jinja engine upstream's `llama-cli` renders
// chat templates with; ours renders them through `zigjinja` (PLAN.md,
// Decisions 25 and 27). This compiles the unmodified engine and renders a
// template the way `common_chat_template` and
// `common_chat_template_direct_apply_impl` do (chat.cpp:1425, 1042
// @cd959bae9): `lexer::tokenize`, `parse_from_tokens`, a `context` over the
// lexer's source, `global_from_json` with input marking on, then `bos_token`
// and `eos_token` set again unmarked -- they come from the model, not the
// user -- `runtime::execute` and `gather_string_parts`.
//
// It returns the rendered text and, per byte, whether it came from input --
// upstream's `is_input` -- so the harness can check ours never trusts a byte
// upstream marks as input.

#include "jinja/caps.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"
#include "json.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

static char * dup_bytes(const std::string & s) {
    char * p = static_cast<char *>(std::malloc(s.size() + 1));
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
    return p;
}

// Renders `tmpl` with the variables in the JSON object `vars`.
//
// Returns 0 with `*out` the text and `*input` one byte per text byte, 1 if the
// template fails to lex or parse, 2 if it raises or fails while rendering;
// on failure `*out` holds the message. Free every pointer with jinja_ref_free.
//
// `*parts` lists the gathered parts as upstream produced them, an empty one
// included: "I<len>," or "T<len>," each, NUL-terminated.
extern "C" int jinja_ref_render(const char * tmpl, size_t tmpl_len, const char * vars, size_t vars_len,
                               char ** out, size_t * out_len, unsigned char ** input, char ** parts) {
    *out     = nullptr;
    *out_len = 0;
    *input   = nullptr;
    *parts   = nullptr;

    jinja::program prog;
    std::string source;
    try {
        jinja::lexer lexer;
        auto lexed = lexer.tokenize(std::string(tmpl, tmpl_len));
        prog       = jinja::parse_from_tokens(lexed);
        source     = lexed.source;
    } catch (const std::exception & e) {
        *out = dup_bytes(e.what());
        return 1;
    }

    try {
        jinja::context ctx(source);
        const common_json inp = common_json::parse(std::string(vars, vars_len));
        jinja::global_from_json(ctx, inp, true);
        for (const char * name : { "bos_token", "eos_token" }) {
            if (inp.contains(name)) {
                ctx.set_val(name, jinja::mk_val<jinja::value_string>(inp.at(name).get<std::string>()));
            }
        }
        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(prog);
        const jinja::string text   = jinja::runtime::gather_string_parts(results)->as_string();

        std::string flat;
        std::string mask;
        std::string list;
        for (const auto & part : text.parts) {
            flat += part.val;
            mask.append(part.val.size(), part.is_input ? 1 : 0);
            list += (part.is_input ? "I" : "T") + std::to_string(part.val.size()) + ",";
        }
        *out     = dup_bytes(flat);
        *out_len = flat.size();
        *input   = reinterpret_cast<unsigned char *>(dup_bytes(mask));
        *parts   = dup_bytes(list);
        return 0;
    } catch (const std::exception & e) {
        *out = dup_bytes(e.what());
        return 2;
    }
}

// The capabilities `jinja::caps_get` reads off `tmpl`, as
// `common_chat_template`'s constructor computes them (chat.cpp:1425 @cd959bae9),
// in `caps::to_map`'s order: one byte each, and `levels` (holding `levels_cap`
// bytes) the reasoning effort levels, comma-separated, NUL-terminated. Returns
// 1 if the template fails to lex or parse.
extern "C" int jinja_ref_caps(const char * tmpl, size_t tmpl_len, unsigned char * out, char * levels, size_t levels_cap) {
    try {
        jinja::lexer lexer;
        auto lexed = lexer.tokenize(std::string(tmpl, tmpl_len));
        jinja::program prog = jinja::parse_from_tokens(lexed);
        size_t i = 0;
        const jinja::caps c = jinja::caps_get(prog);
        for (const auto & [name, value] : c.to_map()) {
            out[i++] = value ? 1 : 0;
        }
        std::string joined;
        for (const auto & level : c.reasoning_effort_levels) {
            joined += (joined.empty() ? "" : ",") + level;
        }
        std::snprintf(levels, levels_cap, "%s", joined.c_str());
        return 0;
    } catch (const std::exception &) {
        return 1;
    }
}

extern "C" void jinja_ref_free(void * p) {
    std::free(p);
}
