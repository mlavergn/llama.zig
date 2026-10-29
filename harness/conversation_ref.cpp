// Not a port: the reference side of `scripts/cnv-diff`, a conversation
// driven through upstream's chat-template engine and upstream's libllama.
//
// Our `llama-cli -cnv` holds a conversation (SPEC.md, section 5.3): each
// line of stdin is a user turn, the whole history is re-rendered through the
// chat template every turn, the new render is diffed against the tokens
// already decoded and only their common prefix is kept, and a memory that
// refuses `llama_memory_seq_rm` -- Qwen3.5's hybrid cache does -- is cleared
// and decoded again. `parity-cli` covers one-shot prompts only, so nothing
// else runs that loop against a reference.
//
// This is that loop written a second time, from SPEC.md, against the stock
// pieces: `llama.cpp/common/jinja` renders, as in `harness/jinja_ref.cpp`
// (chat.h:58, chat.cpp:905 @b29c606e2), and `llama.cpp.zmake`'s libllama
// decodes. Deliberately not our CLI recompiled -- that would compare the CLI
// against itself.
//
// What it reproduces, and must keep reproducing:
//   - every byte our CLI writes: replies on stdout, each ended by a newline;
//     the loading line, the banner, the `> ` marker, the slash-command
//     answers, the cut-off notes and the error sentence on stderr;
//   - the trust split: upstream's input marks, runs of one mark tokenized
//     separately, `parse_special` off for input. `bos_token` and `eos_token`
//     are re-trusted only when a part is exactly their text, which is what
//     ours does for the bound tokens; a message reading exactly `</s>` would
//     be told apart by ours and not here, so the scripts never send one;
//   - the cache report our CLI writes under `LLAMAZIG_CNV_TRACE` -- tokens
//     kept and tokens decoded, per turn -- the one place prefix reuse shows;
//   - `--fit`, as upstream's `common_init_result` runs it before loading:
//     `common_fit_params` (common/fit.cpp:879 @b29c606e2) over the buffers
//     `common_params` lends the model parameters, at `llama-cli`'s log
//     verbosity, its error line flushed to stderr before the load; and the
//     context size and GPU layers it left, which the trace reports once;
//   - greedy sampling, where our CLI runs upstream's default chain at
//     `--temp 0` -- the equivalence `scripts/parity-cli` relies on.
//
// Usage: conversation_ref <model.gguf> <n_ctx> <n_predict> <system> <seed prompt> [reuse|fresh [template file]]
//   An empty `n_ctx` is the CLI without `-c`, which the fit may shrink; `0`
//   is an explicit `-c 0`, which it may not. CNV_REF_FIT=off is `--fit
//   off`, CNV_REF_FIT_TARGET_MIB=<n> is `-fitt <n>`.
//   `system` and `seed prompt` may be empty. `fresh` clears the memory and
//   decodes the whole render every turn instead of keeping the prefix, so
//   the script can tell whether prefix reuse changes the computation. A
//   template file replaces the model's template, as `--chat-template-file`
//   does ours.
// Reads turns from stdin; exits 1 with the CLI's sentence on a prompt that
// does not fit.

#include "fit.h"
#include "log.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"
#include "json.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

struct message {
    std::string role;
    std::string content;
};

struct run {
    std::string text;
    bool trusted;
};

static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static std::string json_string(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out + "\"";
}

// The render, split into runs of one trust level, as cli/chat.zig's
// `segment` splits ours: adjacent parts with the same mark merge, and a bound
// token stays a run of its own.
static bool render(const std::string & tmpl, const std::vector<message> & history,
                   const std::string & bos, const std::string & eos, std::vector<run> & runs) {
    std::string vars = "{\"messages\":[";
    for (size_t i = 0; i < history.size(); i++) {
        if (i) vars += ",";
        vars += "{\"role\":" + json_string(history[i].role) + ",\"content\":" + json_string(history[i].content) + "}";
    }
    vars += "],\"add_generation_prompt\":true,\"bos_token\":" + json_string(bos) + ",\"eos_token\":" + json_string(eos) + "}";

    runs.clear();
    try {
        jinja::lexer lexer;
        auto lexed = lexer.tokenize(tmpl);
        jinja::program prog = jinja::parse_from_tokens(lexed);
        jinja::context ctx(lexed.source);
        jinja::global_from_json(ctx, common_json::parse(vars), true);
        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(prog);
        const jinja::string text   = jinja::runtime::gather_string_parts(results)->as_string();
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
    size_t total = 0;
    for (const auto & r : runs) total += r.text.size();
    return total > 0 || history.empty();
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

static std::string token_text(const llama_vocab * vocab, llama_token t) {
    if (t == LLAMA_TOKEN_NULL) return "";
    const char * s = llama_vocab_get_text(vocab, t);
    return s ? s : "";
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

// `common.cpp` defines this, and `log.cpp` asks it once when the logger
// starts; compiling `common.cpp` for one predicate would bring most of
// `common/` with it. stderr is a file under `scripts/cnv-diff`, where the
// CLI's own answer is no colour too.
bool tty_can_use_colors() {
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <n_ctx> <n_predict> <system> <seed prompt> [reuse|fresh [template file]]\n", argv[0]);
        return 2;
    }
    const char *      model_path = argv[1];
    const bool        n_ctx_set  = argv[2][0] != '\0';
    const int         n_ctx_arg  = atoi(argv[2]);
    const int         n_predict  = atoi(argv[3]);
    const std::string system     = argv[4];
    std::string       seed       = argv[5];
    const bool        fresh      = argc > 6 && strcmp(argv[6], "fresh") == 0;
    const bool        trace      = getenv("LLAMAZIG_CNV_TRACE") != nullptr;
    const char *      tmpl_path  = argc > 7 ? argv[7] : nullptr;

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
        common_log_set_verbosity_thold(LOG_LEVEL_ERROR); // tools/cli/cli.cpp:36 @b29c606e2
        common_fit_params(model_path, &mp, &cp, tensor_split.data(), tbo.data(), margins.data(),
            n_ctx_set && n_ctx_arg == 0 ? UINT32_MAX : 4096, nullptr, GGML_LOG_LEVEL_ERROR);
        common_log_flush(common_log_main());
    }

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return fail("could not load the model -- check the path and that it is a GGUF file");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return fail("could not create a context -- try a smaller -c");

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    warm_up(model, ctx, vocab, (int) cp.n_batch);
    if (trace) fprintf(stderr, "[ctx: %u, gpu layers: %d]\n", llama_n_ctx(ctx), mp.n_gpu_layers);

    std::string tmpl;
    if (tmpl_path) {
        FILE * f = fopen(tmpl_path, "rb");
        if (!f) return fail("cannot read the template file");
        char   buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) tmpl.append(buf, n);
        fclose(f);
    } else {
        const char * tmpl_c = llama_model_chat_template(model, nullptr);
        if (!tmpl_c) return fail("the model has no chat template -- pass --chat-template or --chat-template-file");
        tmpl = tmpl_c;
    }
    const std::string bos  = token_text(vocab, llama_vocab_bos(vocab));
    const std::string eos  = token_text(vocab, llama_vocab_eos(vocab));
    const uint32_t    n_ctx = llama_n_ctx(ctx);

    std::vector<message>     history;
    std::vector<llama_token> decoded;
    if (!system.empty()) history.push_back({ "system", system });

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
            if (!system.empty()) history.push_back({ "system", system });
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
            history.push_back({ "user", turn });
        }

        std::vector<run> runs;
        if (!render(tmpl, history, bos, eos, runs)) { rc = fail("TemplateFailed"); break; }
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

        std::string  reply;
        llama_token  id;
        llama_batch  batch     = llama_batch_get_one(suffix.data(), (int32_t) suffix.size());
        int          generated = 0;
        enum { EOG, N_PREDICT, CONTEXT_FULL } stop = N_PREDICT;
        while (n_predict < 0 || generated < n_predict) {
            if (llama_decode(ctx, batch) != 0) { rc = fail("the model failed to decode"); break; }
            id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) { stop = EOG; break; }
            char piece[256];
            const int n = llama_token_to_piece(vocab, id, piece, sizeof piece, 0, true);
            if (n > 0) {
                reply.append(piece, (size_t) n);
                fwrite(piece, 1, (size_t) n, stdout);
                fflush(stdout);
            }
            generated++;
            decoded.push_back(id);
            if (decoded.size() >= n_ctx) { stop = CONTEXT_FULL; break; }
            batch = llama_batch_get_one(&id, 1);
        }
        if (rc != 0) break;
        fputs("\n", stdout);
        fflush(stdout);
        if (stop == N_PREDICT) fprintf(stderr, "[reply cut off: -n %d tokens reached]\n", n_predict);
        if (stop == CONTEXT_FULL) fprintf(stderr, "[reply cut off: context full; /clear to start over, or raise -c]\n");
        history.push_back({ "assistant", reply });
    }
    if (rc == 0) fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
