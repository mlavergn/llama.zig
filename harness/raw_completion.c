// Raw completion through libllama's C ABI: prompt in, tokens out, greedy.
//
// This is the *reference* side of `scripts/parity-cli`, and the thing
// `make ref-raw` runs. It exists because upstream's `llama-cli` has no raw
// completion mode -- its `-st` runs a single turn of a *conversation*, which
// applies the model's chat template and therefore feeds the model a different
// prompt. Comparing our CLI against that measures the template, not the port.
//
// So the honest reference is this: the same tokenize/decode/sample loop our
// CLI runs, linked against stock C libraries.
//
// Usage: raw_completion <model.gguf> <prompt> <n_predict> [flags] [temp] [seed]
//
// `flags` is a comma-separated list, empty or absent for the defaults:
//   no-prompt    do not echo the prompt before the completion
//   no-timings   do not print the throughput line
//
// Both default to *on*, matching our CLI, so `make ref` and `make port`
// produce the same shape of output and can be compared. `scripts/parity-cli`
// passes both suppressors, because it diffs completions alone.

#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// True when `flags` contains `name` as a comma-separated token.
//
// A plain `strcmp` would do for one flag and did until there were two; a
// substring search would match `no-prompt` inside a hypothetical
// `no-prompt-echo`, so the boundaries are checked.
static int has_flag(const char *flags, const char *name) {
    if (!flags) return 0;
    const size_t len = strlen(name);
    for (const char *p = flags; *p; ) {
        const char *end = strchr(p, ',');
        const size_t n = end ? (size_t)(end - p) : strlen(p);
        if (n == len && strncmp(p, name, len) == 0) return 1;
        if (!end) break;
        p = end + 1;
    }
    return 0;
}

// Ports the warmup in cli/session.zig:warmUp, which in turn follows
// `common_init_from_params`.
//
// Not optional for a timing comparison. Without it the first `llama_decode`
// carries Metal pipeline compilation and first-touch page faults, and they
// land in the prompt counter -- so `make ref` would look far slower than
// `make port` for reasons that have nothing to do with the port.
static void warm_up(struct llama_model *model, struct llama_context *ctx,
                    const struct llama_vocab *vocab, int n_batch) {
    llama_token tmp[2];
    int n = 0;

    // Qwen has no BOS, hence the fallback to token 0 rather than assuming one.
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

    // Drop the warmup tokens, or they prefix the real prompt.
    llama_memory_clear(llama_get_memory(ctx), true);
    // Wait for the GPU before resetting, or work still in flight lands in the
    // counters afterwards.
    llama_synchronize(ctx);
    llama_perf_context_reset(ctx);
}

int main(int argc, char **argv) {
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token toks[2048];
    int n = llama_tokenize(vocab, argv[2], (int)strlen(argv[2]), toks, 2048, true, true);
    if (n < 0) return 1;
    const char *flags = argc > 4 ? argv[4] : "";
    const int show_timings = !has_flag(flags, "no-timings");

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048;
    // `llama_context_default_params` sets `no_perf = true`, so without this
    // the counters stay at zero and every rate comes out 0.0 -- the same trap
    // cli/session.zig documents.
    cp.no_perf = !show_timings;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    if (show_timings) warm_up(model, ctx, vocab, cp.n_batch);
    // The same sampler chain cli/session.zig builds, with the same defaults
    // from common/common.h. Greedy alone would make this incomparable to
    // `make port`, which uses upstream's defaults.
    //
    // temp and seed come from argv so a comparison can be made reproducible;
    // everything else is fixed at upstream's default.
    const float temp = argc > 5 ? (float) atof(argv[5]) : 0.80f;
    const uint32_t seed = argc > 6 ? (uint32_t) strtoul(argv[6], NULL, 10) : LLAMA_DEFAULT_SEED;

    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    struct llama_sampler *smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.00f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 0));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 0));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    // Echo the prompt, matching upstream's `--display-prompt` default and our
    // CLI's, so `make port` and `make ref` produce literally identical text.
    // `scripts/parity-cli` passes a fourth argument to suppress it, because it
    // compares completions alone.
    if (!has_flag(flags, "no-prompt")) {
        fputs(argv[2], stdout);
    }

    struct llama_batch batch = llama_batch_get_one(toks, n);
    llama_token id;
    for (int i = 0; i < atoi(argv[3]); i++) {
        if (llama_decode(ctx, batch)) return 1;
        id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        char buf[256];
        int k = llama_token_to_piece(vocab, id, buf, sizeof buf, 0, true);
        if (k > 0) fwrite(buf, 1, (size_t)k, stdout);
        batch = llama_batch_get_one(&id, 1);
    }
    printf("\n");

    // The same line cli/session.zig:reportTimings writes, computed the same
    // way `llama_perf_context_print` does (llama-context.cpp:4163). On stdout,
    // because that is where our CLI puts it.
    if (show_timings) {
        const struct llama_perf_context_data d = llama_perf_context(ctx);
        const double pp = d.t_p_eval_ms > 0 ? 1e3 / d.t_p_eval_ms * (double) d.n_p_eval : 0.0;
        const double tg = d.t_eval_ms   > 0 ? 1e3 / d.t_eval_ms   * (double) d.n_eval   : 0.0;
        printf("\n[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n", pp, tg);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
