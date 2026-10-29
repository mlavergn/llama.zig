/* Runs every public sampler on fixed inputs and prints what it did, on bits.
 *
 * Built twice by scripts/sampler-diff -- once against the stock libllama from
 * `llama.cpp.zmake`, once against ours -- and the two outputs are diffed. It
 * talks to libllama through `llama.h` alone, so the same source links against
 * either side unchanged; that is what lets it gate `llama-sampler.cpp` and
 * `llama-grammar.cpp` across the swap.
 *
 * Why nothing else can do this
 * ----------------------------
 * `make parity-cli` builds one sampler chain -- penalties, top-k, top-p,
 * min-p, temp-ext, dist -- and checks the tokens it picks. That leaves most of
 * the file unreached: typical, XTC, top-n-sigma, both mirostats, DRY,
 * adaptive-p, logit-bias, infill, grammar and the lazy grammar, and the whole
 * backend-sampling path, which the CLI never configures. And a token is a
 * coarse readout: a sampler that computes the wrong probabilities and still
 * picks the same token passes it. This prints the whole candidate array --
 * every id, logit and probability -- after every call.
 *
 * Five sections:
 *
 *  - **synthetic**: the samplers that need no vocabulary, over seven input
 *    distributions and twelve sizes, through apply, accept, clone, reset and
 *    copy. The sizes straddle every threshold in the file -- the `<= 128`
 *    partial-sort cut, top-p's `> 1024` bucket path, libc++'s insertion-sort
 *    limit of 24 and its ninther threshold of 128.
 *  - **chain**: the chain API itself -- get, remove, n, name, seed, perf.
 *  - **vocab**: grammar, lazy grammar, DRY and infill, which need a real
 *    vocabulary, loaded with `vocab_only`.
 *  - **backend**: backend sampling through a real context, which builds ggml
 *    graphs from the sampler and runs them on Metal.
 *  - **generate**: the CLI's chain at temperature 0.8 over a real decode,
 *    through `llama_sampler_sample`'s raw-logits path.
 *
 * Inputs deliberately include **ties**. `std::sort` and `std::partial_sort`
 * leave the order of equal elements to the library, and the samplers sort on
 * logits; the port reproduces libc++'s algorithms exactly rather than any
 * sort, and only inputs with ties can show the difference.
 *
 * Determinism: every sampler gets an explicit seed. `LLAMA_DEFAULT_SEED` reads
 * the clock, so the one check that uses it prints only whether the seed it got
 * differs from the default.
 *
 * Usage: sampler_dump <model.gguf> [synthetic|chain|vocab|backend|generate]...
 * SAMPLER_FULL=<substring> prints every element of the matching records.
 */

#include "llama.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Input generation and output
// ---------------------------------------------------------------------------

static uint64_t rng_state;

static uint64_t rng_next(void) {
    // xorshift64*: fixed, platform-independent, and nothing like the
    // mt19937 the samplers use, so the inputs cannot correlate with the
    // draws being tested.
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}

static float rng_unit(void) {
    return (float) (rng_next() >> 40) * (1.0f / 16777216.0f);
}

enum pattern {
    PAT_RAND1,   // uniform in [-1, 1)
    PAT_RAND30,  // uniform in [-30, 30): beyond the partial sort's [-10, 10] buckets
    PAT_TIES,    // eight levels only, so most logits tie
    PAT_NEGINF,  // a quarter -inf, as a masking sampler upstream leaves them
    PAT_EQUAL,   // every logit equal
    PAT_PEAK,    // one logit far above the rest
    PAT_SPREAD,  // two scales mixed, crossing the bucket edges
    PAT_COUNT,
};

static const char * pattern_name[PAT_COUNT] = {
    "rand1", "rand30", "ties", "neginf", "equal", "peak", "spread",
};

static const size_t sizes[] = { 1, 2, 3, 5, 6, 24, 25, 129, 1000, 1025, 4096, 32000 };
#define N_SIZES (sizeof(sizes) / sizeof(sizes[0]))

// The synthetic vocabulary size. Ids stay below it; the penalty and
// logit-bias samplers take it as `n_vocab`.
#define SYN_VOCAB 32000

static llama_token_data * buf;
static size_t buf_cap;

static void ensure_buf(size_t n) {
    if (n > buf_cap) {
        buf = realloc(buf, n * sizeof(*buf));
        buf_cap = n;
    }
}

/* Fills `n` candidates. Ids are a permutation of 0..n-1 when `shuffle` is
 * set and the identity otherwise -- logit-bias has a fast path for the
 * identity layout and a search path for everything else. */
static void fill(llama_token_data_array * arr, size_t n, int pat, uint64_t seed, int shuffle) {
    ensure_buf(n);
    rng_state = 0x9E3779B97F4A7C15ULL ^ (seed * 0x100000001B3ULL) ^ ((uint64_t) pat << 32) ^ n;
    if (rng_state == 0) rng_state = 1;

    for (size_t i = 0; i < n; i++) {
        buf[i].id = (llama_token) i;
    }
    if (shuffle) {
        for (size_t i = n; i > 1; i--) {
            size_t j = (size_t) (rng_next() % i);
            llama_token t = buf[i - 1].id;
            buf[i - 1].id = buf[j].id;
            buf[j].id = t;
        }
    }

    for (size_t i = 0; i < n; i++) {
        float v = 0.0f;
        switch (pat) {
            case PAT_RAND1:  v = rng_unit() * 2.0f - 1.0f; break;
            case PAT_RAND30: v = rng_unit() * 60.0f - 30.0f; break;
            case PAT_TIES:   v = (float) (int) (rng_unit() * 8.0f) * 0.5f - 2.0f; break;
            case PAT_NEGINF: v = (i > 0 && rng_unit() < 0.25f) ? -INFINITY : rng_unit() * 20.0f - 10.0f; break;
            case PAT_EQUAL:  v = 0.25f; break;
            case PAT_PEAK:   v = rng_unit() * 2.0f - 1.0f; break;
            case PAT_SPREAD: v = (rng_unit() * 2.0f - 1.0f) * (rng_unit() < 0.5f ? 3.0f : 15.0f); break;
        }
        buf[i].logit = v;
        buf[i].p = 0.0f;
    }
    if (pat == PAT_PEAK) {
        buf[n / 3].logit = 12.0f;
    }

    arr->data = buf;
    arr->size = n;
    arr->selected = -1;
    arr->sorted = false;
}

static uint64_t fnv(uint64_t h, const void * p, size_t n) {
    const unsigned char * b = p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001B3ULL;
    }
    return h;
}

static const char * full_filter;

/* One record: the array's size, selection, sorted flag and a hash over every
 * candidate's id, logit bits and probability bits. */
static void rec(const char * label, const llama_token_data_array * a) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < a->size; i++) {
        h = fnv(h, &a->data[i].id, sizeof(a->data[i].id));
        h = fnv(h, &a->data[i].logit, sizeof(float));
        h = fnv(h, &a->data[i].p, sizeof(float));
    }
    llama_token sel_id = (a->selected >= 0 && (size_t) a->selected < a->size) ? a->data[a->selected].id : -1;
    printf("%s size=%zu sel=%lld tok=%d sorted=%d hash=%016llx\n",
        label, a->size, (long long) a->selected, sel_id, a->sorted ? 1 : 0, (unsigned long long) h);

    if (full_filter && strstr(label, full_filter)) {
        for (size_t i = 0; i < a->size; i++) {
            uint32_t lb, pb;
            memcpy(&lb, &a->data[i].logit, 4);
            memcpy(&pb, &a->data[i].p, 4);
            printf("  [%zu] id=%d logit=%08x p=%08x\n", i, a->data[i].id, lb, pb);
        }
    }
}

/* The token a step accepts: the selection when there is one, else the first
 * candidate. */
static llama_token pick(const llama_token_data_array * a) {
    if (a->selected >= 0 && (size_t) a->selected < a->size) {
        return a->data[a->selected].id;
    }
    return a->size > 0 ? a->data[0].id : 0;
}

// ---------------------------------------------------------------------------
// Synthetic section
// ---------------------------------------------------------------------------

/* Builds sampler `i`, or returns NULL past the end. Every seeded sampler gets
 * an explicit seed. The `?name` entries are each sampler's disabled form,
 * which upstream turns into an empty placeholder. */
static struct llama_sampler * make_synthetic(int i, const char ** label) {
    static const llama_logit_bias biases[] = {
        { 0, 3.0f }, { 7, -INFINITY }, { 12, 0.5f }, { 40000, 1.0f }, { 5, -2.0f }, { 23, 0.25f },
    };
    struct llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    cp.no_perf = true;
    struct llama_sampler * c;

    switch (i) {
        case  0: *label = "greedy";               return llama_sampler_init_greedy();
        case  1: *label = "dist42";               return llama_sampler_init_dist(42);
        case  2: *label = "dist1234";             return llama_sampler_init_dist(1234);
        case  3: *label = "top_k1";               return llama_sampler_init_top_k(1);
        case  4: *label = "top_k5";               return llama_sampler_init_top_k(5);
        case  5: *label = "top_k40";              return llama_sampler_init_top_k(40);
        case  6: *label = "top_k200";             return llama_sampler_init_top_k(200);
        case  7: *label = "top_k100000";          return llama_sampler_init_top_k(100000);
        case  8: *label = "top_k0";               return llama_sampler_init_top_k(0);
        case  9: *label = "top_p0.5";             return llama_sampler_init_top_p(0.5f, 1);
        case 10: *label = "top_p0.9";             return llama_sampler_init_top_p(0.9f, 1);
        case 11: *label = "top_p0.95k3";          return llama_sampler_init_top_p(0.95f, 3);
        case 12: *label = "top_p0";               return llama_sampler_init_top_p(0.0f, 1);
        case 13: *label = "top_p0.999k0";         return llama_sampler_init_top_p(0.999f, 0);
        case 14: *label = "top_p1";               return llama_sampler_init_top_p(1.0f, 1);
        case 15: *label = "min_p0.05";            return llama_sampler_init_min_p(0.05f, 1);
        case 16: *label = "min_p0.2k3";           return llama_sampler_init_min_p(0.2f, 3);
        case 17: *label = "min_p0.9k0";           return llama_sampler_init_min_p(0.9f, 0);
        case 18: *label = "min_p1";               return llama_sampler_init_min_p(1.0f, 1);
        case 19: *label = "min_p0";               return llama_sampler_init_min_p(0.0f, 1);
        case 20: *label = "min_p0.5k100";         return llama_sampler_init_min_p(0.5f, 100);
        case 21: *label = "typical0.5";           return llama_sampler_init_typical(0.5f, 1);
        case 22: *label = "typical0.95k3";        return llama_sampler_init_typical(0.95f, 3);
        case 23: *label = "typical0.2k0";         return llama_sampler_init_typical(0.2f, 0);
        case 24: *label = "temp0";                return llama_sampler_init_temp(0.0f);
        case 25: *label = "temp0.5";              return llama_sampler_init_temp(0.5f);
        case 26: *label = "temp1.5";              return llama_sampler_init_temp(1.5f);
        case 27: *label = "temp-1";               return llama_sampler_init_temp(-1.0f);
        case 28: *label = "temp1";                return llama_sampler_init_temp(1.0f);
        case 29: *label = "temp_ext0.8";          return llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f);
        case 30: *label = "temp_ext0.8d0.5";      return llama_sampler_init_temp_ext(0.8f, 0.5f, 1.0f);
        case 31: *label = "temp_ext1.2d0.3e2";    return llama_sampler_init_temp_ext(1.2f, 0.3f, 2.0f);
        case 32: *label = "temp_ext0.5d1e0.5";    return llama_sampler_init_temp_ext(0.5f, 1.0f, 0.5f);
        case 33: *label = "temp_ext0d0";          return llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f);
        case 34: *label = "xtc1t0.1";             return llama_sampler_init_xtc(1.0f, 0.1f, 1, 7);
        case 35: *label = "xtc0.5t0.05k2";        return llama_sampler_init_xtc(0.5f, 0.05f, 2, 99);
        case 36: *label = "xtc0";                 return llama_sampler_init_xtc(0.0f, 0.1f, 1, 7);
        case 37: *label = "top_n_sigma1";         return llama_sampler_init_top_n_sigma(1.0f);
        case 38: *label = "top_n_sigma2.5";       return llama_sampler_init_top_n_sigma(2.5f);
        case 39: *label = "top_n_sigma0";         return llama_sampler_init_top_n_sigma(0.0f);
        case 40: *label = "mirostat_t5";          return llama_sampler_init_mirostat(SYN_VOCAB, 5, 5.0f, 0.1f, 100);
        case 41: *label = "mirostat_t3m10";       return llama_sampler_init_mirostat(SYN_VOCAB, 11, 3.0f, 0.2f, 10);
        case 42: *label = "mirostat2_t5";         return llama_sampler_init_mirostat_v2(5, 5.0f, 0.1f);
        case 43: *label = "mirostat2_t3";         return llama_sampler_init_mirostat_v2(11, 3.0f, 0.2f);
        case 44: *label = "penalties64r1.1";      return llama_sampler_init_penalties(SYN_VOCAB, 64, 1.1f, 0.0f, 0.0f);
        case 45: *label = "penalties16all";       return llama_sampler_init_penalties(SYN_VOCAB, 16, 1.3f, 0.5f, 0.25f);
        case 46: *label = "penalties-1f";         return llama_sampler_init_penalties(SYN_VOCAB, -1, 1.0f, 0.7f, 0.0f);
        case 47: *label = "penalties0";           return llama_sampler_init_penalties(SYN_VOCAB, 0, 1.5f, 0.0f, 0.0f);
        case 48: *label = "penalties4neg";        return llama_sampler_init_penalties(SYN_VOCAB, 4, 0.8f, -0.3f, 1.5f);
        case 49: *label = "adaptive_p0.5";        return llama_sampler_init_adaptive_p(0.5f, 0.9f, 3);
        case 50: *label = "adaptive_p-1";         return llama_sampler_init_adaptive_p(-1.0f, 0.9f, 3);
        case 51: *label = "adaptive_p0.2d1.5";    return llama_sampler_init_adaptive_p(0.2f, 1.5f, 8);
        case 52: *label = "logit_bias";           return llama_sampler_init_logit_bias(SYN_VOCAB, 6, biases);
        case 53: *label = "logit_bias0";          return llama_sampler_init_logit_bias(SYN_VOCAB, 0, biases);
        case 54:
            *label = "chain_cli";
            c = llama_sampler_chain_init(cp);
            llama_sampler_chain_add(c, llama_sampler_init_penalties(SYN_VOCAB, 64, 1.1f, 0.0f, 0.0f));
            llama_sampler_chain_add(c, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 0));
            llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 0));
            llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(42));
            return c;
        case 55:
            *label = "chain_greedy";
            c = llama_sampler_chain_init(cp);
            llama_sampler_chain_add(c, llama_sampler_init_greedy());
            return c;
        case 56:
            *label = "chain_all";
            c = llama_sampler_chain_init(cp);
            llama_sampler_chain_add(c, llama_sampler_init_logit_bias(SYN_VOCAB, 6, biases));
            llama_sampler_chain_add(c, llama_sampler_init_penalties(SYN_VOCAB, 32, 1.2f, 0.1f, 0.1f));
            llama_sampler_chain_add(c, llama_sampler_init_top_n_sigma(1.5f));
            llama_sampler_chain_add(c, llama_sampler_init_top_k(100));
            llama_sampler_chain_add(c, llama_sampler_init_typical(0.9f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.9f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_min_p(0.02f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_xtc(1.0f, 0.1f, 1, 3));
            llama_sampler_chain_add(c, llama_sampler_init_temp_ext(1.0f, 0.4f, 1.0f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(5));
            return c;
        case 57:
            *label = "chain_adaptive";
            c = llama_sampler_chain_init(cp);
            llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_adaptive_p(0.5f, 0.9f, 77));
            return c;
        case 58:
            *label = "chain_mirostat";
            c = llama_sampler_chain_init(cp);
            llama_sampler_chain_add(c, llama_sampler_init_temp(0.9f));
            llama_sampler_chain_add(c, llama_sampler_init_mirostat_v2(21, 4.0f, 0.15f));
            return c;
        default:
            return NULL;
    }
}

/* A history with plenty of repeats, for the penalty samplers. Ids stay
 * below `n` so they name real candidates. */
static void warm(struct llama_sampler * s, size_t n) {
    for (int h = 0; h < 24; h++) {
        llama_sampler_accept(s, (llama_token) ((size_t) (h * h * 7 + 3) % 17 % n));
    }
}

static void run_synthetic(void) {
    char label[256];
    for (int i = 0;; i++) {
        const char * name = NULL;
        struct llama_sampler * probe = make_synthetic(i, &name);
        if (!probe) break;
        printf("synthetic %s name=%s seed_set=%d\n", name, llama_sampler_name(probe),
            llama_sampler_get_seed(probe) != LLAMA_DEFAULT_SEED);
        llama_sampler_free(probe);

        for (int pat = 0; pat < PAT_COUNT; pat++) {
            for (size_t si = 0; si < N_SIZES; si++) {
                const size_t n = sizes[si];
                const int shuffle = (pat + (int) si) % 2;
                llama_token_data_array a;

                struct llama_sampler * s = make_synthetic(i, &name);
                warm(s, n);

                for (int step = 0; step < 3; step++) {
                    fill(&a, n, pat, (uint64_t) step, shuffle);
                    llama_sampler_apply(s, &a);
                    snprintf(label, sizeof label, "syn %s %s n=%zu step%d", name, pattern_name[pat], n, step);
                    rec(label, &a);
                    llama_sampler_accept(s, pick(&a));
                }

                struct llama_sampler * c = llama_sampler_clone(s);

                fill(&a, n, pat, 3, shuffle);
                llama_sampler_apply(s, &a);
                snprintf(label, sizeof label, "syn %s %s n=%zu orig", name, pattern_name[pat], n);
                rec(label, &a);

                fill(&a, n, pat, 3, shuffle);
                llama_sampler_apply(c, &a);
                snprintf(label, sizeof label, "syn %s %s n=%zu clone", name, pattern_name[pat], n);
                rec(label, &a);

                llama_sampler_reset(s);
                fill(&a, n, pat, 0, shuffle);
                llama_sampler_apply(s, &a);
                snprintf(label, sizeof label, "syn %s %s n=%zu reset", name, pattern_name[pat], n);
                rec(label, &a);
                llama_sampler_accept(s, pick(&a));

                // copy carries s's state into c, through copy_state where the
                // sampler has one and through clone-and-transplant where not.
                llama_sampler_copy(s, c);
                fill(&a, n, pat, 4, shuffle);
                llama_sampler_apply(s, &a);
                snprintf(label, sizeof label, "syn %s %s n=%zu copy-src", name, pattern_name[pat], n);
                rec(label, &a);
                fill(&a, n, pat, 4, shuffle);
                llama_sampler_apply(c, &a);
                snprintf(label, sizeof label, "syn %s %s n=%zu copy-dst", name, pattern_name[pat], n);
                rec(label, &a);

                llama_sampler_free(c);
                llama_sampler_free(s);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chain section
// ---------------------------------------------------------------------------

static const char * custom_name(const struct llama_sampler * smpl) {
    (void) smpl;
    return "custom";
}

static void custom_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    (void) smpl;
    for (size_t i = 0; i < cur_p->size; i++) {
        cur_p->data[i].logit *= 0.5f;
    }
}

static struct llama_sampler_i custom_iface = {
    /* .name              = */ custom_name,
    /* .accept            = */ NULL,
    /* .apply             = */ custom_apply,
    /* .reset             = */ NULL,
    /* .clone             = */ NULL,
    /* .free              = */ NULL,
    /* .backend_init      = */ NULL,
    /* .backend_accept    = */ NULL,
    /* .backend_apply     = */ NULL,
    /* .backend_set_input = */ NULL,
    /* .backend_reset     = */ NULL,
    /* .copy_state        = */ NULL,
};

static void run_chain(void) {
    struct llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    printf("chain default no_perf=%d\n", cp.no_perf);

    cp.no_perf = false;
    struct llama_sampler * c = llama_sampler_chain_init(cp);
    llama_sampler_chain_add(c, llama_sampler_init_top_k(10));
    llama_sampler_chain_add(c, llama_sampler_init_top_p(1.0f, 1));
    llama_sampler_chain_add(c, llama_sampler_init_typical(1.0f, 1));
    llama_sampler_chain_add(c, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(c, llama_sampler_init_mirostat(SYN_VOCAB, 17, 5.0f, 0.1f, 100));
    llama_sampler_chain_add(c, llama_sampler_init_dist(19));

    printf("chain n=%d name=%s\n", llama_sampler_chain_n(c), llama_sampler_name(c));
    for (int i = -2; i <= llama_sampler_chain_n(c); i++) {
        struct llama_sampler * e = llama_sampler_chain_get(c, i);
        printf("chain get(%d)=%s\n", i, e ? llama_sampler_name(e) : "(null)");
    }
    printf("chain seed=%u\n", llama_sampler_get_seed(c));

    llama_token_data_array a;
    char label[64];
    for (int step = 0; step < 4; step++) {
        fill(&a, 1000, PAT_RAND1, (uint64_t) step, 1);
        llama_sampler_apply(c, &a);
        snprintf(label, sizeof label, "chain step%d", step);
        rec(label, &a);
        llama_sampler_accept(c, pick(&a));
    }
    printf("chain perf n_sample=%d\n", llama_perf_sampler(c).n_sample);
    llama_perf_sampler_reset(c);
    printf("chain perf-reset n_sample=%d\n", llama_perf_sampler(c).n_sample);

    struct llama_sampler * removed = llama_sampler_chain_remove(c, 4);
    printf("chain remove(4)=%s n=%d\n", removed ? llama_sampler_name(removed) : "(null)", llama_sampler_chain_n(c));
    llama_sampler_free(removed);
    removed = llama_sampler_chain_remove(c, 99);
    printf("chain remove(99)=%s\n", removed ? llama_sampler_name(removed) : "(null)");
    printf("chain seed-after-remove=%u\n", llama_sampler_get_seed(c));

    // Not a chain: get returns NULL.
    struct llama_sampler * g = llama_sampler_init_greedy();
    printf("chain get-on-greedy=%s\n", llama_sampler_chain_get(g, -1) ? "non-null" : "(null)");
    printf("chain seed-greedy=%u\n", llama_sampler_get_seed(g));
    llama_sampler_free(g);

    // A default seed reads the clock; only its difference from the default
    // is reproducible.
    struct llama_sampler * d = llama_sampler_init_dist(LLAMA_DEFAULT_SEED);
    printf("chain dist-default seed_is_default=%d\n", llama_sampler_get_seed(d) == LLAMA_DEFAULT_SEED);
    llama_sampler_free(d);

    // A user sampler with no context clones through the generic path.
    struct llama_sampler * u = llama_sampler_init(&custom_iface, NULL);
    struct llama_sampler * uc = llama_sampler_clone(u);
    printf("chain custom name=%s clone-name=%s\n", llama_sampler_name(u), llama_sampler_name(uc));
    fill(&a, 32, PAT_RAND1, 0, 0);
    llama_sampler_apply(uc, &a);
    rec("chain custom-apply", &a);
    llama_sampler_free(uc);
    llama_sampler_free(u);

    // Null tolerance in the dispatchers.
    llama_sampler_accept(NULL, 0);
    llama_sampler_apply(NULL, &a);
    llama_sampler_reset(NULL);
    printf("chain null-clone=%s\n", llama_sampler_clone(NULL) ? "non-null" : "(null)");
    llama_sampler_free(NULL);
    llama_sampler_copy(NULL, c);

    struct llama_sampler * cc = llama_sampler_clone(c);
    printf("chain clone n=%d\n", llama_sampler_chain_n(cc));
    llama_sampler_free(cc);
    llama_sampler_free(c);
}

// ---------------------------------------------------------------------------
// Vocab section
// ---------------------------------------------------------------------------

// The JSON rules without a root, so two grammars can share them.
#define JSON_RULES \
    "value  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\n" \
    "object ::=\n" \
    "  \"{\" ws (\n" \
    "            string \":\" ws value\n" \
    "    (\",\" ws string \":\" ws value)*\n" \
    "  )? \"}\" ws\n" \
    "array  ::=\n" \
    "  \"[\" ws (\n" \
    "            value\n" \
    "    (\",\" ws value)*\n" \
    "  )? \"]\" ws\n" \
    "string ::=\n" \
    "  \"\\\"\" (\n" \
    "    [^\"\\\\\\x7F\\x00-\\x1F] |\n" \
    "    \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4}) # escapes\n" \
    "  )* \"\\\"\" ws\n" \
    "number ::= (\"-\"? ([0-9] | [1-9] [0-9]{0,15})) (\".\" [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws\n" \
    "# Optional space: by convention, applied in this grammar after literal chars when allowed\n" \
    "ws ::= | \" \" | \"\\n\" [ \\t]{0,20}\n"

static const char * json_gbnf = "root ::= object\n" JSON_RULES;

// The tool-call form: the grammar itself starts with the trigger text, so a
// trigger token is fed to a grammar that accepts it.
static const char * tool_gbnf = "root ::= \"<tool_call>\" object\n" JSON_RULES;


struct grammar_case {
    const char * name;
    const char * text;
    const char * root;
};

static const struct grammar_case grammars[] = {
    { "yesno",   "root ::= \"yes\" | \"no\"", "root" },
    { "arith",   "root ::= expr\nexpr ::= term ([-+*/] term)*\nterm ::= [0-9]+ | \"(\" expr \")\"", "root" },
    { "json",    NULL, "root" },
    { "unicode", "root ::= [а-яА-Я]+ \" \" [^\\n]{1,5} \"!\" | \"日本\" [\\u3040-\\u309F]*", "root" },
    { "repeat",  "root ::= \"a\"{2,4} \"b\"? \"c\"* [xyz]{3} .", "root" },
    { "nested",  "root ::= (\"<\" (item (\",\" item)*)? \">\")+\nitem ::= [a-z]+ | root", "root" },
    { "tokens",  "root ::= <[1]> | !<[2]> \"x\"", "root" },
    { "anychar", "root ::= . . \"z\"? [\\x20-\\x7E]*", "root" },
    { "altroot", "start ::= \"hi\" | \"ho\"\nroot ::= start", "start" },
    { "empty",   "", "root" },
    { "badref",  "root ::= missing", "root" },
    { "badsyn",  "root ::= \"unterminated", "root" },
    { "leftrec", "root ::= root \"a\" | \"b\"", "root" },
    { "noroot",  "start ::= \"x\"", "root" },
    { "badesc",  "root ::= \"\\q\"", "root" },
    { "toomany", "root ::= \"a\"{3000}", "root" },
    // Exactly at MAX_REPETITION_THRESHOLD: accepted since v0.4.1, whose
    // check is `>` where v0.4.0's was `>=`.
    { "edge2000", "root ::= \"a\"{2000}", "root" },
    // Ranges just above the 2- and 3-byte UTF-8 floors, which a token ending
    // in a bare 0xE0 or 0xF0 lead byte is checked against as a partial
    // sequence.
    { "partial3", "root ::= [\\u0800-\\u0840]+ \"!\"", "root" },
    { "partial4", "root ::= [\\U00010000-\\U00010040]+ \"!\"", "root" },
};
#define N_GRAMMARS (sizeof(grammars) / sizeof(grammars[0]))

static llama_token * vocab_ids;
static int32_t n_vocab;

/* Candidates for a vocab-section step: every token, with logits from the
 * synthetic generator. */
static void fill_vocab(llama_token_data_array * a, uint64_t seed) {
    fill(a, (size_t) n_vocab, PAT_RAND1, seed, 0);
}

/* The grammar walk's choice: the best candidate the grammar left finite. */
static llama_token best_finite(const llama_token_data_array * a) {
    llama_token best = -1;
    float best_l = -INFINITY;
    for (size_t i = 0; i < a->size; i++) {
        if (a->data[i].logit > best_l) {
            best_l = a->data[i].logit;
            best = a->data[i].id;
        }
    }
    return best;
}

static void walk_grammar(const char * name, struct llama_sampler * s, const struct llama_vocab * vocab, int steps) {
    char label[128];
    llama_token_data_array a;
    struct llama_sampler * c = NULL;
    for (int step = 0; step < steps; step++) {
        fill_vocab(&a, (uint64_t) (100 + step));
        llama_sampler_apply(s, &a);
        snprintf(label, sizeof label, "grammar %s step%d", name, step);
        rec(label, &a);
        llama_token t = best_finite(&a);
        printf("grammar %s step%d chose=%d\n", name, step, t);
        if (t < 0 || llama_vocab_is_eog(vocab, t)) break;
        llama_sampler_accept(s, t);
        if (step == 2) {
            c = llama_sampler_clone(s);
        }
    }
    if (c) {
        fill_vocab(&a, 200);
        llama_sampler_apply(c, &a);
        snprintf(label, sizeof label, "grammar %s clone", name);
        rec(label, &a);
        llama_sampler_free(c);
    }
    llama_sampler_reset(s);
    fill_vocab(&a, 100);
    llama_sampler_apply(s, &a);
    snprintf(label, sizeof label, "grammar %s reset", name);
    rec(label, &a);
}

/* Feeds a lazy grammar the tokens of `text` and records what it allows
 * after each one. */
static void feed_lazy(const char * name, struct llama_sampler * s, const struct llama_vocab * vocab, const char * text) {
    llama_token toks[256];
    int n = llama_tokenize(vocab, text, (int32_t) strlen(text), toks, 256, false, true);
    char label[128];
    llama_token_data_array a;
    for (int i = 0; i < n; i++) {
        llama_sampler_accept(s, toks[i]);
        fill_vocab(&a, (uint64_t) (300 + i));
        llama_sampler_apply(s, &a);
        snprintf(label, sizeof label, "lazy %s tok%d=%d", name, i, toks[i]);
        rec(label, &a);
    }
}

static void run_vocab(const struct llama_vocab * vocab) {
    n_vocab = llama_vocab_n_tokens(vocab);
    printf("vocab n=%d eos=%d eot=%d\n", n_vocab, llama_vocab_eos(vocab), llama_vocab_eot(vocab));
    (void) vocab_ids;

    for (size_t g = 0; g < N_GRAMMARS; g++) {
        const char * text = grammars[g].text ? grammars[g].text : json_gbnf;
        struct llama_sampler * s = llama_sampler_init_grammar(vocab, text, grammars[g].root);
        printf("grammar %s init=%s\n", grammars[g].name, s ? llama_sampler_name(s) : "(null)");
        if (!s) continue;
        walk_grammar(grammars[g].name, s, vocab, 8);
        llama_sampler_free(s);
    }

    // Lazy grammars. The trigger patterns run through std::regex upstream.
    // Ends where the JSON does: a lazy grammar that has triggered rejects
    // anything after it, and upstream throws on that rather than reporting it.
    static const char * lazy_text = "Let me check that. <tool_call>{\"name\": \"x\", \"v\": [1, 2]}";
    {
        const char * pats[] = { "[\\s\\S]*?<tool_call>(\\{)[\\s\\S]*" };
        struct llama_sampler * s = llama_sampler_init_grammar_lazy_patterns(vocab, json_gbnf, "root", pats, 1, NULL, 0);
        printf("lazy search init=%s\n", s ? "ok" : "(null)");
        if (s) { feed_lazy("search", s, vocab, lazy_text); llama_sampler_free(s); }
    }
    {
        const char * pats[] = { "^[\\s\\S]*?<tool_call>(\\{)[\\s\\S]*$" };
        struct llama_sampler * s = llama_sampler_init_grammar_lazy_patterns(vocab, json_gbnf, "root", pats, 1, NULL, 0);
        printf("lazy anchored init=%s\n", s ? "ok" : "(null)");
        if (s) { feed_lazy("anchored", s, vocab, lazy_text); llama_sampler_free(s); }
    }
    {
        const char * pats[] = { "never(\\{)", "(\\{\")" };
        struct llama_sampler * s = llama_sampler_init_grammar_lazy_patterns(vocab, json_gbnf, "root", pats, 2, NULL, 0);
        printf("lazy multi init=%s\n", s ? "ok" : "(null)");
        if (s) { feed_lazy("multi", s, vocab, lazy_text); llama_sampler_free(s); }
    }
    {
        // A trigger token, when the vocabulary has one for the text.
        llama_token tt[4];
        int nt = llama_tokenize(vocab, "<tool_call>", 11, tt, 4, false, true);
        printf("lazy trigger-token n=%d\n", nt);
        if (nt == 1) {
            struct llama_sampler * s = llama_sampler_init_grammar_lazy_patterns(vocab, tool_gbnf, "root", NULL, 0, tt, 1);
            if (s) { feed_lazy("token", s, vocab, lazy_text); llama_sampler_free(s); }
        }
    }
    {
        // The deprecated trigger-word form, which escapes the words into a
        // regex with std::regex_replace.
        const char * words[] = { "<tool_call>", "a.b*(c)" };
        struct llama_sampler * s = llama_sampler_init_grammar_lazy(vocab, tool_gbnf, "root", words, 2, NULL, 0);
        printf("lazy words init=%s\n", s ? "ok" : "(null)");
        if (s) {
            feed_lazy("words", s, vocab, lazy_text);
            struct llama_sampler * c = llama_sampler_clone(s);
            llama_token_data_array a;
            fill_vocab(&a, 400);
            llama_sampler_apply(c, &a);
            rec("lazy words clone", &a);
            llama_sampler_reset(s);
            feed_lazy("words-reset", s, vocab, "plain text <tool_call>{\"k\": true}");
            llama_sampler_free(c);
            llama_sampler_free(s);
        }
    }

    // DRY: the breakers are matched against every token's text at init.
    {
        const char * breakers[] = { "\n", ":", "\"", "*", "" };
        struct dry_case { const char * name; float mult, base; int allowed, last_n; } cases[] = {
            { "dry-neg1",    0.8f, 1.75f, 2, -1 },  // clamps to 0: disabled
            { "dry0.8",      0.8f, 1.75f, 2, 64 },
            { "dry2last16",  2.0f, 1.1f,  1, 16 },
            { "dry-big",     5.0f, 3.0f,  2, 64 },
            { "dry-off",     0.0f, 1.75f, 2, 64 },
            // A base and multiplier whose powers round differently in float
            // and double at an exponent this history reaches (3), so the
            // `double` `pow` is visible. 1.37 and 0.31 differ only at 2, 5
            // and 7, which it does not, and the float form passed.
            { "dry-odd",     0.77f, 1.13f, 1, 64 },
        };
        static const char * history = "the cat sat on the mat: the cat sat on the mat: \"the cat sat on the";
        llama_token hist[256];
        int nh = llama_tokenize(vocab, history, (int32_t) strlen(history), hist, 256, false, false);
        for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
            struct llama_sampler * s = llama_sampler_init_dry(vocab, cases[k].mult, cases[k].base,
                cases[k].allowed, cases[k].last_n, breakers, 5);
            printf("dry %s name=%s\n", cases[k].name, llama_sampler_name(s));
            char label[128];
            llama_token_data_array a;
            for (int i = 0; i < nh; i++) {
                llama_sampler_accept(s, hist[i]);
                if (i % 4 == 3 || i == nh - 1) {
                    fill_vocab(&a, (uint64_t) (500 + i));
                    llama_sampler_apply(s, &a);
                    snprintf(label, sizeof label, "dry %s after%d", cases[k].name, i);
                    rec(label, &a);

                    // All-zero logits: `0 - penalty` keeps every bit of the
                    // penalty, where subtracting it from a logit near 1
                    // rounds most last-bit differences away.
                    fill_vocab(&a, (uint64_t) (500 + i));
                    for (size_t t = 0; t < a.size; t++) a.data[t].logit = 0.0f;
                    llama_sampler_apply(s, &a);
                    snprintf(label, sizeof label, "dry %s zeros%d", cases[k].name, i);
                    rec(label, &a);
                }
            }
            struct llama_sampler * c = llama_sampler_clone(s);
            fill_vocab(&a, 600);
            llama_sampler_apply(c, &a);
            snprintf(label, sizeof label, "dry %s clone", cases[k].name);
            rec(label, &a);
            llama_sampler_reset(s);
            fill_vocab(&a, 600);
            llama_sampler_apply(s, &a);
            snprintf(label, sizeof label, "dry %s reset", cases[k].name);
            rec(label, &a);
            llama_sampler_free(c);
            llama_sampler_free(s);
        }
    }

    // Infill: candidates sharing prefixes, plus EOG tokens.
    {
        static const char * words[] = { " hel", " hell", " hello", " help", " world", " wor", " dummy", " a", " ab", " abc" };
        llama_token ids[64];
        int n_ids = 0;
        for (size_t w = 0; w < sizeof(words) / sizeof(words[0]); w++) {
            llama_token t[8];
            int nt = llama_tokenize(vocab, words[w], (int32_t) strlen(words[w]), t, 8, false, false);
            if (nt >= 1) ids[n_ids++] = t[0];
        }
        ids[n_ids++] = llama_vocab_eos(vocab);
        if (llama_vocab_eot(vocab) != LLAMA_TOKEN_NULL) ids[n_ids++] = llama_vocab_eot(vocab);

        struct llama_sampler * s = llama_sampler_init_infill(vocab);
        char label[64];
        for (int variant = 0; variant < 14; variant++) {
            llama_token_data_array a;
            fill(&a, (size_t) n_ids, PAT_RAND1, (uint64_t) variant, 0);
            for (int i = 0; i < n_ids; i++) {
                a.data[i].id = ids[i];
            }
            if (variant == 1) a.data[n_ids - 1].logit = 8.0f;   // EOG dominates
            if (variant == 2) a.data[2].logit = 6.0f;           // " hello" dominates
            if (variant == 3) for (int i = 0; i < n_ids; i++) a.data[i].logit = 0.0f;
            if (variant == 4) a.data[n_ids - 1].logit = -INFINITY;
            if (variant == 5) { a.data[0].logit = 3.0f; a.data[n_ids - 1].logit = 2.9f; }
            // EOG mass low and a few strong text candidates, so the merge and
            // both thresholds run rather than the early EOG return.
            if (variant >= 6) {
                for (int i = 0; i < n_ids; i++) a.data[i].logit = -4.0f + 0.1f * (float) i;
                a.data[n_ids - 1].logit = -9.0f;
                if (n_ids > 2) a.data[n_ids - 2].logit = -9.0f;
                a.data[(variant + 1) % (n_ids - 2)].logit = 2.0f;
                a.data[(variant + 3) % (n_ids - 2)].logit = 1.4f + 0.2f * (float) (variant - 6);
                a.data[(variant + 4) % (n_ids - 2)].logit = 1.1f;
            }
            // Three text candidates near 0.45, 0.32 and 0.23 of the mass, so
            // the first cut (0.2) and the second (one over the survivors
            // plus one) both decide something.
            if (variant >= 10) {
                for (int i = 0; i < n_ids; i++) a.data[i].logit = -6.0f;
                a.data[(variant + 0) % (n_ids - 2)].logit = 2.0f;
                a.data[(variant + 2) % (n_ids - 2)].logit = 1.65f;
                a.data[(variant + 5) % (n_ids - 2)].logit = 1.32f - 0.05f * (float) (variant - 10);
            }
            llama_sampler_apply(s, &a);
            snprintf(label, sizeof label, "infill variant%d", variant);
            rec(label, &a);
        }
        struct llama_sampler * c = llama_sampler_clone(s);
        printf("infill clone name=%s\n", llama_sampler_name(c));
        llama_sampler_free(c);
        llama_sampler_free(s);
    }

    // logit-bias over the real vocabulary, where ids match indices.
    {
        llama_logit_bias lb[] = { { 0, 1.0f }, { (llama_token) (n_vocab - 1), -5.0f }, { 1234, 2.5f } };
        struct llama_sampler * s = llama_sampler_init_logit_bias(n_vocab, 3, lb);
        llama_token_data_array a;
        fill_vocab(&a, 700);
        llama_sampler_apply(s, &a);
        rec("vocab logit_bias", &a);
        llama_sampler_free(s);
    }
}

// ---------------------------------------------------------------------------
// Backend section
// ---------------------------------------------------------------------------

static uint64_t hash_floats(const float * p, uint32_t n) {
    return p ? fnv(0xCBF29CE484222325ULL, p, (size_t) n * sizeof(float)) : 0;
}

static uint64_t hash_tokens(const llama_token * p, uint32_t n) {
    return p ? fnv(0xCBF29CE484222325ULL, p, (size_t) n * sizeof(llama_token)) : 0;
}

static struct llama_sampler * make_backend(int i, const char ** label, int32_t nv, uint32_t * outputs) {
    struct llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    cp.no_perf = true;
    struct llama_sampler * c = llama_sampler_chain_init(cp);
    static const llama_logit_bias lb[] = { { 11, 4.0f }, { 13, -INFINITY }, { 220, 1.5f } };
    *outputs = 1;
    switch (i) {
        case 0:
            *label = "greedy";
            llama_sampler_chain_add(c, llama_sampler_init_greedy());
            return c;
        case 1:
            *label = "dist";
            llama_sampler_chain_add(c, llama_sampler_init_dist(42));
            return c;
        case 2:
            *label = "topk-temp-dist";
            llama_sampler_chain_add(c, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(c, llama_sampler_init_temp(0.8f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(7));
            return c;
        case 3:
            *label = "full";
            llama_sampler_chain_add(c, llama_sampler_init_penalties(nv, 64, 1.2f, 0.1f, 0.1f));
            llama_sampler_chain_add(c, llama_sampler_init_top_k(50));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.9f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.8f, 0.3f, 1.0f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(3));
            return c;
        case 4:
            *label = "bias-topk-greedy";
            llama_sampler_chain_add(c, llama_sampler_init_logit_bias(nv, 3, lb));
            llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
            llama_sampler_chain_add(c, llama_sampler_init_greedy());
            return c;
        case 5:
            *label = "split";
            // typical has no backend form, so everything from it on runs on
            // the CPU over the backend's output.
            llama_sampler_chain_add(c, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(c, llama_sampler_init_typical(0.9f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_dist(9));
            return c;
        case 6:
            *label = "temp0-minp";
            llama_sampler_chain_add(c, llama_sampler_init_min_p(0.1f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(1));
            return c;
        case 7:
            *label = "multi-output";
            // More than one output per sequence: dist turns transactional and
            // penalties declines the backend.
            *outputs = 4;
            llama_sampler_chain_add(c, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(c, llama_sampler_init_dist(11));
            return c;
        default:
            llama_sampler_free(c);
            return NULL;
    }
}

static void run_backend(struct llama_model * model) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t nv = llama_vocab_n_tokens(vocab);

    static const char * prompt = "The capital of France is";
    llama_token toks[64];
    const int np = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt), toks, 64, true, false);

    for (int i = 0;; i++) {
        const char * name = NULL;
        uint32_t outputs = 1;
        struct llama_sampler * chain = make_backend(i, &name, nv, &outputs);
        if (!chain) break;

        struct llama_sampler_seq_config cfg = { 0, chain };
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.samplers = &cfg;
        cp.n_samplers = 1;
        cp.n_outputs_max_per_seq = outputs;

        struct llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            printf("backend %s ctx=(null)\n", name);
            llama_sampler_free(chain);
            continue;
        }
        for (int k = 0; k < llama_sampler_chain_n(chain); k++) {
            printf("backend %s elem%d=%s\n", name, k, llama_sampler_name(llama_sampler_chain_get(chain, k)));
        }

        struct llama_batch b = llama_batch_init(np, 0, 1);
        for (int k = 0; k < np; k++) {
            b.token[k] = toks[k];
            b.pos[k] = k;
            b.n_seq_id[k] = 1;
            b.seq_id[k][0] = 0;
            b.logits[k] = k >= np - (int) outputs;
        }
        b.n_tokens = np;
        if (llama_decode(ctx, b) != 0) {
            printf("backend %s decode failed\n", name);
        }
        llama_batch_free(b);

        int pos = np;
        for (int step = 0; step < 8; step++) {
            const int n_out = step == 0 ? (int) outputs : 1;
            llama_token chosen = -1;
            for (int o = 0; o < n_out; o++) {
                // `_ith` takes a position in the batch, not an output number.
                const int idx = n_out == 1 ? -1 : np - n_out + o;
                const llama_token st = llama_get_sampled_token_ith(ctx, idx);
                const uint32_t npr = llama_get_sampled_probs_count_ith(ctx, idx);
                const uint32_t nlg = llama_get_sampled_logits_count_ith(ctx, idx);
                printf("backend %s step%d out%d sampled=%d probs=%u/%016llx logits=%u/%016llx cands=%016llx\n",
                    name, step, o, st,
                    npr, (unsigned long long) hash_floats(llama_get_sampled_probs_ith(ctx, idx), npr),
                    nlg, (unsigned long long) hash_floats(llama_get_sampled_logits_ith(ctx, idx), nlg),
                    (unsigned long long) hash_tokens(llama_get_sampled_candidates_ith(ctx, idx), npr ? npr : nlg));
                chosen = llama_sampler_sample(chain, ctx, idx);
                printf("backend %s step%d out%d token=%d\n", name, step, o, chosen);
            }
            struct llama_batch nb = llama_batch_get_one(&chosen, 1);
            (void) pos;
            pos++;
            if (llama_decode(ctx, nb) != 0) {
                printf("backend %s step%d decode failed\n", name, step);
                break;
            }
        }

        llama_free(ctx);
        llama_sampler_free(chain);
    }
}

// ---------------------------------------------------------------------------
// Generate section

/* The CLI's own chain -- penalties, top-k, top-p, min-p, temp-ext, dist --
 * at temperature 0.8 through `llama_sampler_sample`'s raw-logits path, the
 * one every non-greedy CLI token takes. `make parity-cli` runs only
 * `--temp 0`, where `dist` never draws. */
static void run_generate(struct llama_model * model) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    static const char * prompts[] = { "The capital of France is", "Once upon a time", "def quicksort(arr):" };
    static const uint32_t seeds[] = { 1, 42, 2026 };
    for (size_t pi = 0; pi < sizeof(prompts) / sizeof(prompts[0]); pi++) {
        for (size_t si = 0; si < sizeof(seeds) / sizeof(seeds[0]); si++) {
            struct llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 512;
            cp.n_batch = 512;
            struct llama_context * ctx = llama_init_from_model(model, cp);
            struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
            sp.no_perf = true;
            struct llama_sampler * c = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(c, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.1f, 0.0f, 0.0f));
            llama_sampler_chain_add(c, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 0));
            llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 0));
            llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(seeds[si]));

            llama_token toks[64];
            const int n = llama_tokenize(vocab, prompts[pi], (int32_t) strlen(prompts[pi]), toks, 64, true, false);
            struct llama_batch b = llama_batch_get_one(toks, n);
            printf("generate p%zu seed=%u:", pi, seeds[si]);
            for (int step = 0; step < 48; step++) {
                if (llama_decode(ctx, b) != 0) { printf(" decode-failed"); break; }
                llama_token t = llama_sampler_sample(c, ctx, -1);
                printf(" %d", t);
                if (llama_vocab_is_eog(vocab, t)) break;
                toks[0] = t;
                b = llama_batch_get_one(toks, 1);
            }
            printf("\n");
            llama_sampler_free(c);
            llama_free(ctx);
        }
    }
}

// ---------------------------------------------------------------------------

static void quiet(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [synthetic|chain|vocab|backend|generate]...\n", argv[0]);
        return 2;
    }
    full_filter = getenv("SAMPLER_FULL");
    if (!getenv("SAMPLER_VERBOSE")) {
        llama_log_set(quiet, NULL);
    }
    llama_backend_init();

    int want_syn = argc == 2, want_chain = argc == 2, want_vocab = argc == 2, want_backend = argc == 2, want_gen = argc == 2;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "synthetic")) want_syn = 1;
        else if (!strcmp(argv[i], "chain")) want_chain = 1;
        else if (!strcmp(argv[i], "vocab")) want_vocab = 1;
        else if (!strcmp(argv[i], "backend")) want_backend = 1;
        else if (!strcmp(argv[i], "generate")) want_gen = 1;
        else { fprintf(stderr, "unknown section %s\n", argv[i]); return 2; }
    }

    if (want_syn) run_synthetic();
    if (want_chain) run_chain();

    if (want_vocab) {
        struct llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        struct llama_model * model = llama_model_load_from_file(argv[1], mp);
        if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
        run_vocab(llama_model_get_vocab(model));
        llama_model_free(model);
    }

    if (want_backend || want_gen) {
        struct llama_model_params mp = llama_model_default_params();
        struct llama_model * model = llama_model_load_from_file(argv[1], mp);
        if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
        if (want_backend) run_backend(model);
        if (want_gen) run_generate(model);
        llama_model_free(model);
    }

    free(buf);
    llama_backend_free();
    return 0;
}
