// Not a port: the reference side of `scripts/cnv-diff`, a conversation
// driven through upstream's chat stack and upstream's libllama.
//
// Our `llama-cli -cnv` holds a conversation (SPEC.md, section 5.3): each
// line of stdin is a user turn, the whole history is re-rendered through the
// chat template every turn, the new render is diffed against the tokens
// already decoded and only their common prefix is kept, and a memory that
// refuses `llama_memory_seq_rm` -- Qwen3.5's hybrid cache does -- is cleared
// and decoded again. Each reply is read back by the template's parser and
// its reasoning shown apart from the answer (SPEC.md 5.3, "Replies").
// `parity-cli` covers one-shot prompts only, so nothing else runs that loop
// against a reference.
//
// This is that loop written a second time, from SPEC.md, against the stock
// pieces -- `llama.cpp.zmake`'s libllama decodes, and upstream's
// `llama-common` (`libllama-common-ref.a`, `zig build tools-ref`) does the
// rest, as upstream's `llama-cli` reaches it through the server it runs:
//   - `common_chat_templates_init` and `common_chat_templates_apply`
//     (chat.cpp:757, 1424 @b29c606e2) render the request
//     `oaicompat_chat_params_parse` (tools/server/server-common.cpp:1151
//     @b29c606e2) builds from a body of `messages` alone;
//   - `common_sampler` (common/sampling.cpp:187 @b29c606e2) samples, with the
//     reasoning budget the server's schema builds (server-schema.cpp:387
//     @b29c606e2);
//   - each piece is held back and cut at stop strings as `process_token`
//     (server-context.cpp:1833 @b29c606e2) does, then parsed with
//     `common_chat_parse` and diffed with `compute_diffs` as
//     `update_chat_msg` (server-task.cpp:162 @b29c606e2) does;
//   - the display is upstream's own `ui::assistant_turn` and `ui::show_error`
//     (tools/cli/cli-ui.h @b29c606e2) over `console`, included, not copied.
// Deliberately not our CLI recompiled -- that would compare the CLI against
// itself.
//
// What it reproduces, and must keep reproducing:
//   - every byte our CLI writes: replies on stdout; the loading line, the
//     banner, the `> ` marker, the slash-command answers, the cut-off notes
//     and the error sentence on stderr;
//   - the trust split: upstream's input marks, runs of one mark tokenized
//     separately, `parse_special` off for input. Upstream's own prompt is a
//     plain string, so the marks come from a second render of the same
//     bindings `common_chat_template_direct_apply_impl` (chat.cpp:905
//     @b29c606e2) makes, through `common/jinja`, which must equal
//     `common_chat_templates_apply`'s prompt byte for byte or the run fails.
//     `bos_token` and `eos_token` are re-trusted only when a part is exactly
//     their text; a message reading exactly `</s>` would be told apart by
//     ours and not here, so the scripts never send one;
//   - the cache report our CLI writes under `LLAMAZIG_CNV_TRACE` -- tokens
//     kept and tokens decoded, per turn -- the one place prefix reuse shows;
//   - `--fit`, as upstream's `common_init_result` runs it before loading:
//     `common_fit_params` (common/fit.cpp:879 @b29c606e2) over the buffers
//     `common_params` lends the model parameters, at `llama-cli`'s log
//     verbosity, its error line flushed to stderr before the load; and the
//     context size and GPU layers it left, which the trace reports once;
//   - greedy sampling: `common_sampler` at temperature 0, where our CLI runs
//     upstream's default chain at `--temp 0`.
//
// Usage: conversation_ref <model.gguf> <n_ctx> <n_predict> <system> <seed prompt>
//                         [reuse|fresh [template file|-]] [-- <chat flags>]
//   An empty `n_ctx` is the CLI without `-c`, which the fit may shrink; `0`
//   is an explicit `-c 0`, which it may not. CNV_REF_FIT=off is `--fit
//   off`, CNV_REF_FIT_TARGET_MIB=<n> is `-fitt <n>`.
//   `system` and `seed prompt` may be empty. `fresh` clears the memory and
//   decodes the whole render every turn instead of keeping the prefix, so
//   the script can tell whether prefix reuse changes the computation. A
//   template file replaces the model's template, as `--chat-template-file`
//   does ours; `-` is none. The chat flags are llama-cli's, read as
//   `common/arg.cpp`'s handlers read them: --reasoning-format, -rea /
//   --reasoning, --reasoning-budget, --reasoning-budget-message,
//   --reasoning-effort, --[no-]reasoning-preserve, --[no-]skip-chat-parsing,
//   --chat-template-kwargs and -co / --color.
// Reads turns from stdin; exits 1 with the CLI's sentence on a prompt that
// does not fit.

#include "chat.h"
#include "common.h"
#include "console.h"
#include "fit.h"
#include "log.h"
#include "sampling.h"
#include "jinja/caps.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"
#include "json.h"
#include "llama.h"

#include "cli-ui.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = common_json;

struct run {
    std::string text;
    bool trusted;
};

// The chat options llama-cli takes, as common_params holds them.
struct chat_opts {
    common_reasoning_format            reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    int                                enable_reasoning = -1;
    int                                budget           = -1;
    std::string                        budget_message;
    bool                               skip_parsing     = false;
    std::map<std::string, std::string> kwargs;
    bool                               color            = false;
};

static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static std::string format_time(const char * format) {
    auto        now        = std::chrono::system_clock::now();
    auto        time       = std::chrono::system_clock::to_time_t(now);
    auto        local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format);
    return ss.str();
}

// The bindings `common_chat_template_direct_apply_impl` makes for this
// request, rendered through `common/jinja` with every bound string marked
// input, split into runs of one trust level as cli/conversation.zig splits
// ours: adjacent parts with the same mark merge, and a bound token stays a
// run of its own.
static bool render_marked(const std::string & src, const std::vector<common_chat_msg> & history,
                          const std::string & bos, const std::string & eos, bool add_bos, bool add_eos,
                          bool enable_thinking, const std::map<std::string, std::string> & kwargs,
                          std::vector<run> & runs) {
    runs.clear();
    try {
        jinja::lexer lexer;
        auto lexed = lexer.tokenize(src);
        jinja::program prog = jinja::parse_from_tokens(lexed);
        jinja::context ctx(lexed.source);

        json extra = json::object();
        extra["datetime"]    = format_time("%b %d %Y");
        extra["date_string"] = format_time("%d %b %Y");
        for (const auto & [k, v] : kwargs) {
            extra[k] = json::parse(v);
        }
        json inp = json{
            {"messages", common_chat_msgs_to_json_oaicompat(history)},
            {"bos_token", bos},
            {"eos_token", eos},
            {"enable_thinking", enable_thinking},
        };
        for (const auto & [k, v] : extra.items()) {
            inp[k] = v;
        }
        inp["add_generation_prompt"] = true;
        if (inp.contains("preserve_reasoning") && inp["preserve_reasoning"].is_boolean()) {
            jinja::caps_apply_preserve_reasoning(ctx, inp["preserve_reasoning"].get<bool>());
        }
        if (inp.contains("reasoning_effort") && inp["reasoning_effort"].is_string() && !inp["reasoning_effort"].empty()) {
            jinja::caps_apply_reasoning_effort(ctx, inp["reasoning_effort"].get<std::string>());
        }
        jinja::global_from_json(ctx, inp, true);
        jinja::runtime runtime(ctx);
        const jinja::value  results = runtime.execute(prog);
        const jinja::string text    = jinja::runtime::gather_string_parts(results)->as_string();
        bool last_token = false;
        for (const auto & part : text.parts) {
            if (part.val.empty()) continue;
            const bool token   = part.is_input && ((!bos.empty() && part.val == bos) || (!eos.empty() && part.val == eos));
            const bool trusted = !part.is_input || token;
            if (!runs.empty() && runs.back().trusted == trusted && !token && !last_token) {
                runs.back().text += part.val;
            } else {
                runs.push_back({ part.val, trusted });
            }
            last_token = token;
        }
    } catch (const std::exception &) {
        return false;
    }
    // `add_bos`/`add_eos`: the vocabulary adds them, so the render loses them.
    if (add_bos && !bos.empty() && !runs.empty() && string_starts_with(runs.front().text, bos)) {
        runs.front().text.erase(0, bos.size());
        if (runs.front().text.empty()) runs.erase(runs.begin());
    }
    if (add_eos && !eos.empty() && !runs.empty() && string_ends_with(runs.back().text, eos)) {
        runs.back().text.resize(runs.back().text.size() - eos.size());
        if (runs.back().text.empty()) runs.pop_back();
    }
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::vector<run> & runs) {
    std::vector<llama_token> out;
    for (const auto & r : runs) {
        const int n = -llama_tokenize(vocab, r.text.data(), (int) r.text.size(), nullptr, 0, false, r.trusted);
        std::vector<llama_token> part(n > 0 ? n : 0);
        const int got = llama_tokenize(vocab, r.text.data(), (int) r.text.size(), part.data(), (int) part.size(), false, r.trusted);
        part.resize(got > 0 ? got : 0);
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

// cli/session.zig's `warmUp`, as harness/raw_completion.c has it.
static void warm_up(llama_model * model, llama_context * ctx, const llama_vocab * vocab, int n_batch) {
    llama_token tmp[2];
    int n = 0;
    const llama_token bos = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    if (bos != LLAMA_TOKEN_NULL) tmp[n++] = bos;
    if (eos != LLAMA_TOKEN_NULL) tmp[n++] = eos;
    if (n == 0) { tmp[0] = 0; n = 1; }
    if (llama_model_has_encoder(model)) {
        llama_encode(ctx, llama_batch_get_one(tmp, n));
        llama_token start = llama_model_decoder_start_token(model);
        if (start == LLAMA_TOKEN_NULL) start = bos;
        tmp[0] = start;
        n = 1;
    }
    if (llama_model_has_decoder(model)) {
        const int count = n < n_batch ? n : n_batch;
        llama_decode(ctx, llama_batch_get_one(tmp, count > 0 ? count : 1));
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);
    llama_perf_context_reset(ctx);
}

static std::string trim(const std::string & s) {
    const char * ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

static bool starts_with(const std::string & s, const char * p) {
    return s.compare(0, strlen(p), p) == 0;
}

static int fail(const char * sentence) {
    fprintf(stderr, "error: %s\n", sentence);
    return 1;
}

// tools/server/server-common.cpp:903 @b29c606e2, which is not in llama-common.
static size_t validate_utf8(const std::string & text) {
    size_t len = text.size();
    if (len == 0) return 0;
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        if ((c & 0xE0) == 0xC0) {
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            if (i < 4) return len - i;
        }
    }
    return len;
}

// server_slot::find_stopping_strings (server-context.cpp:567 @b29c606e2).
static size_t find_stopping_strings(const std::vector<std::string> & stops, const std::string & text,
                                    size_t last_token_size, bool is_full_stop) {
    size_t stop_pos = std::string::npos;
    for (const std::string & word : stops) {
        size_t pos;
        if (is_full_stop) {
            const size_t tmp      = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
            pos = text.find(word, from_pos);
        } else {
            pos = string_find_partial_stop(text, word);
        }
        if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
            stop_pos = pos;
        }
    }
    return stop_pos;
}

// The chat flags, as common/arg.cpp's handlers read them.
static bool parse_chat_flags(int argc, char ** argv, int i, chat_opts & o) {
    auto truthy = [](const std::string & v) { return v == "on" || v == "enabled" || v == "true" || v == "1"; };
    auto falsey = [](const std::string & v) { return v == "off" || v == "disabled" || v == "false" || v == "0"; };
    for (; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--reasoning-format") {
            o.reasoning_format = common_reasoning_format_from_name(next());
        } else if (a == "-rea" || a == "--reasoning") {
            const std::string v = next();
            if (truthy(v)) {
                o.enable_reasoning = 1;
                o.kwargs["enable_thinking"] = "true";
            } else if (falsey(v)) {
                o.enable_reasoning = 0;
                o.kwargs["enable_thinking"] = "false";
            } else {
                o.enable_reasoning = -1;
            }
        } else if (a == "--reasoning-budget") {
            o.budget = std::stoi(next());
        } else if (a == "--reasoning-budget-message") {
            o.budget_message = next();
        } else if (a == "--reasoning-effort") {
            const std::string v = next();
            if (v == "default") {
                o.kwargs.erase("reasoning_effort");
            } else {
                o.kwargs["reasoning_effort"] = json(v).dump();
            }
        } else if (a == "--reasoning-preserve" || a == "--no-reasoning-preserve") {
            o.kwargs["preserve_reasoning"] = a == "--reasoning-preserve" ? "true" : "false";
        } else if (a == "--skip-chat-parsing" || a == "--no-skip-chat-parsing") {
            o.skip_parsing = a == "--skip-chat-parsing";
        } else if (a == "--chat-template-kwargs") {
            auto parsed = json::parse(next());
            for (const auto & item : parsed.items()) {
                o.kwargs[item.key()] = item.value().dump();
            }
        } else if (a == "-co" || a == "--color") {
            o.color = truthy(next());
        } else {
            fprintf(stderr, "conversation_ref: unknown chat flag %s\n", a.c_str());
            return false;
        }
    }
    if (!o.kwargs.count("preserve_reasoning")) {
        o.kwargs["preserve_reasoning"] = "true";
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <n_ctx> <n_predict> <system> <seed prompt> [reuse|fresh [template file|-]] [-- <chat flags>]\n", argv[0]);
        return 2;
    }
    int n_pos = argc;
    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) { n_pos = i; break; }
    }
    const char *      model_path = argv[1];
    const bool        n_ctx_set  = argv[2][0] != '\0';
    const int         n_ctx_arg  = atoi(argv[2]);
    const int         n_predict  = atoi(argv[3]);
    const std::string system     = argv[4];
    std::string       seed       = argv[5];
    const bool        fresh      = n_pos > 6 && strcmp(argv[6], "fresh") == 0;
    const bool        trace      = getenv("LLAMAZIG_CNV_TRACE") != nullptr;
    const char *      tmpl_path  = n_pos > 7 && strcmp(argv[7], "-") != 0 ? argv[7] : nullptr;

    chat_opts opts;
    if (!parse_chat_flags(argc, argv, n_pos + 1, opts)) return 2;

    common_log_set_verbosity_thold(LOG_LEVEL_ERROR); // tools/cli/cli.cpp:36 @b29c606e2
    console::init(/* simple_io */ true, opts.color);
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    fprintf(stderr, "loading %s ...\n", model_path);

    // The buffers `common_params` lends the model parameters
    // (common/common.cpp:1679 @b29c606e2), padded as common/arg.cpp:941 @b29c606e2 pads
    // the overrides.
    std::vector<float> tensor_split(128, 0.0f);
    std::vector<llama_model_tensor_buft_override> tbo(llama_max_tensor_buft_overrides(), { nullptr, nullptr });
    llama_model_params mp = llama_model_default_params();
    mp.tensor_split          = tensor_split.data();
    mp.tensor_buft_overrides = tbo.data();

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx_set ? (uint32_t) n_ctx_arg : 0;
    cp.n_batch = 2048;
    cp.no_perf = true;
    // server_output_limits (tools/server/server-context.cpp:41 @b29c606e2):
    // one sequence, no draft.
    cp.n_outputs_max         = 1;
    cp.n_outputs_max_per_seq = 1;

    const char * fit_env = getenv("CNV_REF_FIT");
    if (!fit_env || strcmp(fit_env, "off") != 0) {
        const char *        target_env = getenv("CNV_REF_FIT_TARGET_MIB");
        const size_t        target_mib = target_env ? strtoull(target_env, nullptr, 10) : 1024;
        std::vector<size_t> margins(llama_max_devices(), target_mib * 1024 * 1024);
        common_fit_params(model_path, &mp, &cp, tensor_split.data(), tbo.data(), margins.data(),
            n_ctx_set && n_ctx_arg == 0 ? UINT32_MAX : 4096, nullptr, GGML_LOG_LEVEL_ERROR);
        common_log_flush(common_log_main());
    }

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return fail("could not load the model -- check the path and that it is a GGUF file");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return fail("could not create a context -- try a smaller -c");

    warm_up(model, ctx, vocab, (int) cp.n_batch);
    if (trace) fprintf(stderr, "[ctx: %u, gpu layers: %d]\n", llama_n_ctx(ctx), mp.n_gpu_layers);

    std::string tmpl_override;
    if (tmpl_path) {
        FILE * f = fopen(tmpl_path, "rb");
        if (!f) return fail("cannot read the template file");
        char   buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) tmpl_override.append(buf, n);
        fclose(f);
    }

    // The server's chat options (server-context.cpp:1449 @b29c606e2).
    common_chat_templates_ptr tmpls;
    bool enable_thinking_default = false;
    try {
        tmpls = common_chat_templates_init(model, tmpl_override);
        enable_thinking_default = opts.enable_reasoning != 0 && common_chat_templates_support_enable_thinking(tmpls.get());
    } catch (const std::exception & e) {
        return fail(e.what());
    }
    const std::string tmpl_src = common_chat_templates_source(tmpls.get());
    const auto        caps     = common_chat_templates_get_caps(tmpls.get());
    const std::string bos      = llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL ? common_token_to_piece(vocab, llama_vocab_bos(vocab), true) : "";
    const std::string eos      = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL ? common_token_to_piece(vocab, llama_vocab_eos(vocab), true) : "";
    const uint32_t    n_ctx    = llama_n_ctx(ctx);

    std::vector<common_chat_msg> history;
    std::vector<llama_token>     decoded;
    auto add_system = [&]() {
        if (!system.empty()) {
            common_chat_msg m;
            m.role    = "system";
            m.content = system;
            history.push_back(m);
        }
    };
    add_system();

    fprintf(stderr, "conversation mode -- /exit to quit, /clear to reset, /regen to retry\n");
    char line_buf[4096];
    bool have_seed = !seed.empty();
    // A failure ends the loop rather than returning: upstream's Metal device
    // asserts in a static destructor if the context outlives `main`.
    int rc = 0;

    while (true) {
        std::string line;
        if (have_seed) {
            line      = seed;
            have_seed = false;
        } else {
            fprintf(stderr, "\n> ");
            fflush(stderr);
            if (!fgets(line_buf, sizeof line_buf, stdin)) break;
            line = line_buf;
        }

        const std::string turn = trim(line);
        if (turn.empty()) continue;
        if (starts_with(turn, "/exit")) break;
        if (starts_with(turn, "/clear")) {
            history.clear();
            add_system();
            fprintf(stderr, "chat history cleared\n");
            continue;
        }
        if (starts_with(turn, "/regen")) {
            if (history.empty() || history.back().role != "assistant") {
                fprintf(stderr, "nothing to regenerate\n");
                continue;
            }
            history.pop_back();
        } else {
            common_chat_msg m;
            m.role    = "user";
            m.content = turn;
            history.push_back(m);
        }

        // The request (server-common.cpp:1151 @b29c606e2).
        common_chat_templates_inputs inputs;
        inputs.messages              = history;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.parallel_tool_calls   = caps.at("supports_parallel_tool_calls");
        inputs.reasoning_format      = opts.reasoning_format;
        inputs.enable_thinking       = enable_thinking_default;
        inputs.chat_template_kwargs  = opts.kwargs;
        inputs.force_pure_content    = opts.skip_parsing;
        auto et = opts.kwargs.find("enable_thinking");
        if (et != opts.kwargs.end()) {
            if (et->second == "true") inputs.enable_thinking = true;
            else if (et->second == "false") inputs.enable_thinking = false;
        }
        common_chat_params params;
        try {
            params = common_chat_templates_apply(tmpls.get(), inputs);
        } catch (const std::exception & e) {
            rc = fail(e.what());
            break;
        }

        std::vector<run> runs;
        if (!render_marked(tmpl_src, history, bos, eos, llama_vocab_get_add_bos(vocab), llama_vocab_get_add_eos(vocab),
                           inputs.enable_thinking, opts.kwargs, runs)) {
            rc = fail("TemplateFailed");
            break;
        }
        {
            std::string joined;
            for (const auto & r : runs) joined += r.text;
            if (joined != params.prompt) {
                if (string_starts_with(params.prompt, joined)) {
                    runs.push_back({ params.prompt.substr(joined.size()), true });
                } else {
                    fprintf(stderr, "conversation_ref: the marked render is not common_chat_templates_apply's prompt\n");
                    rc = 3;
                    break;
                }
            }
        }
        const std::vector<llama_token> tokens = tokenize(vocab, runs);
        if (tokens.size() >= n_ctx) { rc = fail("the prompt is longer than the context -- raise -c or shorten it"); break; }

        // SPEC.md 5.3: keep the longest common prefix, but never all of it --
        // an empty suffix leaves nothing to sample from.
        size_t keep = 0;
        while (keep < tokens.size() && keep < decoded.size() && tokens[keep] == decoded[keep]) keep++;
        if (keep == tokens.size() && keep > 0) keep--;
        if (fresh) keep = 0;
        if (keep < decoded.size() || fresh) {
            llama_memory_t mem = llama_get_memory(ctx);
            if (keep == 0 || !llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1)) {
                llama_memory_clear(mem, true);
                keep = 0;
            }
            decoded.resize(keep);
        }

        std::vector<llama_token> suffix(tokens.begin() + (long) decoded.size(), tokens.end());
        if (trace) fprintf(stderr, "[turn: kept %zu, decoding %zu]\n", decoded.size(), suffix.size());
        decoded.insert(decoded.end(), suffix.begin(), suffix.end());

        // The server task (server-schema.cpp @b29c606e2): the generation
        // prompt, the preserved tokens that are one token, and the budget
        // fields when the template has an end-of-thinking tag.
        common_params_sampling sp;
        sp.temp              = 0.0f;
        sp.generation_prompt = params.generation_prompt;
        for (const auto & t : params.preserved_tokens) {
            auto ids = common_tokenize(vocab, t, false, true);
            if (ids.size() == 1) sp.preserved_tokens.insert(ids[0]);
        }
        sp.reasoning_budget_tokens = opts.budget;
        if (!params.thinking_end_tags.empty()) {
            sp.reasoning_budget_start = common_tokenize(vocab, params.thinking_start_tag, false, true);
            for (const auto & tag : params.thinking_end_tags) {
                if (!tag.empty()) sp.reasoning_budget_end.push_back(common_tokenize(vocab, tag, false, true));
            }
            if (!sp.reasoning_budget_end.empty()) {
                llama_tokens end_tag = sp.reasoning_budget_end.front();
                if (!opts.budget_message.empty()) {
                    llama_tokens msg = common_tokenize(vocab, opts.budget_message, false, true);
                    end_tag.insert(end_tag.begin(), msg.begin(), msg.end());
                }
                sp.reasoning_budget_forced = std::move(end_tag);
            }
        }
        common_sampler * smpl = common_sampler_init(model, sp);

        common_chat_parser_params pp(params);
        pp.reasoning_format     = opts.reasoning_format;
        pp.reasoning_in_content = opts.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
        if (!params.parser.empty()) pp.parser.load(params.parser);

        std::string     text;     // slot.generated_text
        size_t          n_sent = 0;
        std::string     sent;     // task_result_state::generated_text
        common_chat_msg msg_prv;  // task_result_state::chat_msg
        std::string     content, reasoning;
        bool            failed = false;

        enum { EOG, N_PREDICT, CONTEXT_FULL, WORD, REJECTED } stop = N_PREDICT;
        {
            ui::assistant_turn a;
            // update_chat_msg and generate_completion's handling of the deltas.
            auto update = [&](bool is_partial) -> bool {
                try {
                    common_chat_msg new_msg = common_chat_parse(sent, is_partial, pp);
                    if (!new_msg.empty()) {
                        auto diffs = common_chat_msg_diff::compute_diffs(msg_prv, new_msg);
                        msg_prv = new_msg;
                        for (const auto & d : diffs) {
                            if (!d.reasoning_content_delta.empty()) {
                                reasoning += d.reasoning_content_delta;
                                a.push(ui::ASSISTANT_DISPLAY_MODE_REASONING, d.reasoning_content_delta);
                            }
                            if (!d.content_delta.empty()) {
                                content += d.content_delta;
                                a.push(ui::ASSISTANT_DISPLAY_MODE_CONTENT, d.content_delta);
                            }
                        }
                    }
                    return true;
                } catch (const std::exception & e) {
                    failed = true;
                    ui::show_error(e.what());
                    return false;
                }
            };

            llama_token id;
            llama_batch batch     = llama_batch_get_one(suffix.data(), (int32_t) suffix.size());
            int         generated = 0;
            while (n_predict < 0 || generated < n_predict) {
                if (llama_decode(ctx, batch) != 0) { rc = fail("the model failed to decode"); break; }
                id = common_sampler_sample(smpl, ctx, -1, false);
                common_sampler_accept(smpl, id, true);
                if (llama_vocab_is_eog(vocab, id)) { stop = EOG; break; }
                generated++;

                const std::string token_str = common_token_to_piece(vocab, id, sp.preserved_tokens.count(id) != 0);
                text += token_str;
                bool stopped = false;
                if (validate_utf8(text) == text.size()) {
                    size_t pos = std::min(n_sent, text.size());
                    const std::string str_test = text.substr(pos);
                    bool send = true;
                    size_t stop_pos = find_stopping_strings(params.additional_stops, str_test, token_str.size(), true);
                    if (stop_pos != std::string::npos) {
                        text.erase(pos + stop_pos);
                        stopped = true;
                    } else if (find_stopping_strings(params.additional_stops, str_test, token_str.size(), false) != std::string::npos) {
                        send = false;
                    }
                    if (send && text.size() > n_sent) {
                        sent += text.substr(n_sent);
                        n_sent = text.size();
                        if (!update(true)) { stop = REJECTED; break; }
                    }
                }
                if (stopped) { stop = WORD; break; }

                // The last token of a reply cut short is never decoded.
                if (n_predict >= 0 && generated >= n_predict) break;
                if (decoded.size() + 1 >= n_ctx) { stop = CONTEXT_FULL; break; }
                decoded.push_back(id);
                batch = llama_batch_get_one(&id, 1);
            }
            if (rc == 0 && !failed) update(false);
        }
        common_sampler_free(smpl);
        if (rc != 0) break;
        if (stop == N_PREDICT) fprintf(stderr, "[reply cut off: -n %d tokens reached]\n", n_predict);
        if (stop == CONTEXT_FULL) fprintf(stderr, "[reply cut off: context full; /clear to start over, or raise -c]\n");

        common_chat_msg reply;
        reply.role              = "assistant";
        reply.content           = content;
        reply.reasoning_content = reasoning;
        history.push_back(reply);
    }
    if (rc == 0) fprintf(stderr, "\n");
    fflush(stdout);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
