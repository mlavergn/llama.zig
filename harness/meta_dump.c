/* Runs a real model tensor-parallel through the meta backend and prints what
 * it computed and what it logged.
 *
 * The end-to-end half of `scripts/meta-diff`. Built twice -- once against
 * `zig-out/lib/libggml.a` as it is, whose meta backend is the C++
 * `ggml-backend-meta.cpp`, and once against a copy with that object deleted
 * and `src/ggml/backend_meta.zig` linked in its place -- with the same
 * `libllama.a` on both sides. So the meta backend is the only variable, and
 * libllama's own split-state callback, `llama_meta_device_get_split_state`,
 * is what assigns every weight its split: the split states this gate covers
 * are the ones libllama really produces for the model, segmented Qwen3.5
 * linear-attention weights included.
 *
 * This machine has one GPU, so "several devices" means the same device more
 * than once, which the meta backend accepts: each listing gets its own
 * backend, its own buffers and its own slice, and the all-reduce between them
 * runs exactly as it would between cards.
 *
 * argv: <model.gguf> <devices> [opts]
 *   devices  one letter per device: m = Metal (MTL0), c = CPU.
 *   opts     comma-separated, any of:
 *              split=a:b:...   tensor_split proportions, one per device
 *              nofa            flash attention off (soft_max path)
 *              ubatch=N        n_ubatch (default: n_batch)
 *
 * Output, all on stdout:
 *   - every log line except the ones that carry addresses or timings, which
 *     differ run to run on either side. With GGML_META_DEBUG=1 in the
 *     environment this includes one `SPLIT_STATE` line per tensor the meta
 *     backend derives a split for: its sources' axes and per-device extents
 *     and its own.
 *   - per decode: the FNV-1a of every output position's logits and the
 *     greedy token.
 */

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* Lines are assembled from fragments (the CONT level) before filtering. */
static char line_buf[1 << 16];
static size_t line_len = 0;

/* "0x" and at least eight hex digits. A bare "0x" will not do: a split
 * state's per-device extent prints as `0x1` for an empty slice, and the
 * first version of this filter silently dropped every SPLIT_STATE line of a
 * run with a zero `tensor_split` entry -- the gate passed on nothing. */
static int has_address(const char * s) {
    for (const char * p = strstr(s, "0x"); p; p = strstr(p + 2, "0x")) {
        int n = 0;
        while ((p[2 + n] >= '0' && p[2 + n] <= '9') || (p[2 + n] >= 'a' && p[2 + n] <= 'f')) n++;
        if (n >= 8) return 1;
    }
    return 0;
}

static int volatile_line(const char * s) {
    /* The meta backend's own derivations are never filtered. */
    if (strstr(s, "SPLIT_STATE")) return 0;
    /* Addresses, timings and free-memory figures: different every run. */
    if (has_address(s)) return 1;
    static const char * const skip[] = {
        " ms", " sec", "t/s", "took", "MiB free", "free ", "recommendedMaxWorkingSetSize",
        "load time", "eval time", "total time", "mmap", "loaded meta data",
    };
    for (size_t i = 0; i < sizeof(skip)/sizeof(skip[0]); i++) {
        if (strstr(s, skip[i])) return 1;
    }
    return 0;
}

static void flush_line(void) {
    line_buf[line_len] = 0;
    if (line_len > 0 && !volatile_line(line_buf)) {
        printf("log: %s", line_buf);
        if (line_buf[line_len - 1] != '\n') printf("\n");
    }
    line_len = 0;
}

static void on_log(enum ggml_log_level level, const char * text, void * ud) {
    (void) level; (void) ud;
    for (const char * p = text; *p; p++) {
        if (line_len + 1 < sizeof(line_buf)) line_buf[line_len++] = *p;
        if (*p == '\n') flush_line();
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <devices> [opts]\n", argv[0]);
        return 2;
    }
    const char * opts = argc > 3 ? argv[3] : "";

    llama_log_set(on_log, NULL);
    llama_backend_init();

    ggml_backend_dev_t devs[17] = {0};
    int nd = 0;
    for (const char * p = argv[2]; *p && nd < 16; p++) {
        if (*p == 'm') devs[nd++] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (*p == 'c') devs[nd++] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }

    float split[16] = {0};
    int have_split = 0;
    const char * s = strstr(opts, "split=");
    if (s) {
        s += 6;
        for (int i = 0; i < nd; i++) {
            split[i] = strtof(s, (char **) &s);
            if (*s == ':') s++;
        }
        have_split = 1;
    }
    int n_ubatch = 0;
    const char * u = strstr(opts, "ubatch=");
    if (u) n_ubatch = atoi(u + 7);

    struct llama_model_params mp = llama_model_default_params();
    mp.devices = devs;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.n_gpu_layers = 99;
    if (have_split) mp.tensor_split = split;
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { flush_line(); return 1; }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "The capital of France is Paris, and the capital of Italy is";
    llama_token toks[512];
    const int n = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks, 512, true, true);
    if (n < 0) return 1;

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 256;
    if (n_ubatch) cp.n_ubatch = n_ubatch;
    cp.n_threads = cp.n_threads_batch = 4;
    if (strstr(opts, "nofa")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { flush_line(); return 1; }

    const int nv = llama_vocab_n_tokens(vocab);

    /* The prompt with logits at every position, then single-token steps: two
     * graph shapes, so the meta backend rebuilds its subgraphs and flips its
     * compute containers between them. */
    struct llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b)) { flush_line(); return 1; }
    for (int i = 0; i < n; i++) {
        const float * lg = llama_get_logits_ith(ctx, i);
        printf("prompt pos %d logits %016llx\n", i, (unsigned long long) fnv1a(lg, (size_t) nv * sizeof(float)));
    }
    llama_batch_free(b);

    llama_token next = 0;
    {
        const float * lg = llama_get_logits_ith(ctx, -1);
        for (int i = 1; i < nv; i++) if (lg[i] > lg[next]) next = i;
    }
    for (int step = 0; step < 6; step++) {
        printf("step %d token %d\n", step, next);
        if (llama_decode(ctx, llama_batch_get_one(&next, 1))) { flush_line(); return 1; }
        const float * lg = llama_get_logits_ith(ctx, -1);
        printf("step %d logits %016llx\n", step, (unsigned long long) fnv1a(lg, (size_t) nv * sizeof(float)));
        llama_token best = 0;
        for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
        next = best;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    flush_line();
    return 0;
}
