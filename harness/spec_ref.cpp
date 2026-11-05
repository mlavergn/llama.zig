// Not a port. The reference side of `scripts/spec-diff`: upstream's
// `common/` -- sampling.cpp, reasoning-budget.cpp, speculative.cpp and the
// three n-gram stores -- compiled unmodified into libllama-common-ref, over
// libllama-ref. `harness/spec_diff.zig` is the same script over the port;
// both write one transcript to stdout and log through common_log to stderr,
// and the two streams must be identical.
//
// Usage: spec_ref api|run <common args...>
//   api  the model-free entry points: type names, n_max, synthetic rates,
//        output limits, types_from_gguf (SPEC_GGUFS, colon-separated), the
//        n-gram stores on synthetic token streams (every cache's iteration
//        order and saved bytes), the reasoning budget's state machine, the
//        sampler's parameter print and name tables.
//   run  a speculative decode over the target (-m) and the draft (-md, if
//        any), as llama-speculative-simple drives it, on -np sequences:
//        every draft, every accepted run, the target sampler's top
//        candidates on bits, clone/copy, the statistics. The prompt (-p) is
//        split at "|" into one prompt per sequence; "#ids:1,2,3" gives token
//        ids for a model with no vocabulary. SPEC_RBUDGET=<n> gives the
//        target sampler a reasoning budget over the tokens of SPEC_RB_START,
//        SPEC_RB_END and SPEC_RB_FORCED (default "<think>", "</think>",
//        " ok</think>").

#include "arg.h"
#include "common.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "reasoning-budget.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"
#include "llama.h"

#include <cinttypes>
#include <climits>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static void print_tokens(const char * label, const std::vector<llama_token> & v) {
    printf("%s [", label);
    for (size_t i = 0; i < v.size(); i++) printf(i ? ",%d" : "%d", v[i]);
    printf("]");
}

// the synthetic token streams: an LCG with copied segments, so n-grams repeat
struct stream_gen {
    uint32_t x;
    explicit stream_gen(uint32_t seed) : x(seed) {}
    uint32_t next() { x = x * 1103515245u + 12345u; return (x >> 16) & 0x7fff; }
    std::vector<llama_token> make(size_t n, int vocab, int rep) {
        std::vector<llama_token> v;
        while (v.size() < n) {
            if (v.size() > 8 && (int)(next() % 100) < rep) {
                size_t len = 2 + next() % 10;
                size_t from = next() % (v.size() - 1);
                for (size_t k = 0; k < len && v.size() < n; k++) v.push_back(v[(from + k) % v.size()]);
            } else {
                v.push_back((llama_token)(next() % vocab));
            }
        }
        return v;
    }
};

static void dump_cache(const char * label, const common_ngram_cache & nc) {
    printf("%s size=%zu buckets=%zu\n", label, nc.size(), nc.bucket_count());
    for (const auto & it : nc) {
        printf("  [%d,%d,%d,%d] n=%zu b=%zu:", it.first.tokens[0], it.first.tokens[1], it.first.tokens[2], it.first.tokens[3], it.second.size(), it.second.bucket_count());
        for (const auto & p : it.second) printf(" %d:%d", p.first, p.second);
        printf("\n");
    }
}

static uint64_t file_hash(const std::string & path, size_t * size) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { *size = 0; return 0; }
    std::vector<uint8_t> buf;
    uint8_t tmp[4096];
    size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    fclose(f);
    *size = buf.size();
    return fnv(buf.data(), buf.size());
}

static void api_types() {
    printf("all_types %s\n", common_speculative_all_types_str());
    const std::vector<std::vector<std::string>> lists = {
        {"ngram-mod"}, {"none", "ngram-mod"}, {"ngram-mod", "none"}, {"draft-simple", "ngram-cache", "ngram-map-k"},
        {"bogus"}, {"ngram-map-k4v", "Ngram-mod"}, {}, {"draft-eagle3", "draft-mtp", "draft-dflash", "draft-dspark", "ngram-simple"},
        {"ngram-map-k", ""},
    };
    for (const auto & l : lists) {
        printf("from_names");
        for (const auto & s : l) printf(" '%s'", s.c_str());
        try {
            auto t = common_speculative_types_from_names(l);
            printf(" -> '%s' (%zu)\n", common_speculative_type_name_str(t).c_str(), t.size());
        } catch (const std::exception & e) {
            printf(" -> invalid_argument: %s\n", e.what());
        }
    }
    for (const char * n : {"none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "draft-dspark", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache", "x", "", "NONE"}) {
        printf("from_name '%s' -> %d\n", n, (int) common_speculative_type_from_name(n));
    }
    for (int t = 0; t <= COMMON_SPECULATIVE_TYPE_COUNT; t++) {
        printf("to_str %d -> %s\n", t, common_speculative_type_to_str((common_speculative_type) t).c_str());
    }
}

static void api_n_max() {
    for (int k = 0; k < 24; k++) {
        common_params_speculative p;
        p.types.clear();
        for (int t = 0; t < COMMON_SPECULATIVE_TYPE_COUNT; t++) if ((k * 7 + t * 3) % 5 == 0) p.types.push_back((common_speculative_type) t);
        p.draft.n_max = (k % 7) - 2;
        p.ngram_simple.size_m = (uint16_t) (k * 3);
        p.ngram_map_k.size_m = (uint16_t) (k * 5 % 17);
        p.ngram_map_k4v.size_m = (uint16_t) (k * 11 % 23);
        p.ngram_mod.n_max = (k % 9) * 9 - 10;
        printf("n_max types='%s' -> %d\n", common_speculative_type_name_str(p.types).c_str(), common_speculative_n_max(&p));
    }
}

static void api_synth() {
    struct C { double len; std::vector<double> rates; int n_max; };
    const double nan = std::nan("");
    std::vector<C> cases = {
        {-1.0, {}, 3}, {2.5, {}, 3}, {1.0, {}, 3}, {4.0, {}, 3}, {4.5, {}, 3}, {nan, {}, 3}, {0.5, {}, 3},
        {-1.0, {0.9, 0.8, 0.5}, 3}, {-1.0, {0.9, 0.8}, 3}, {-1.0, {0.5, 0.8, 0.9}, 3}, {-1.0, {1.5, 0.8, 0.5}, 3},
        {2.0, {0.5, 0.4, 0.3}, 3}, {2.0, {}, 0}, {-1.0, {0.5}, 0}, {1.7, {}, 8}, {1.0000001, {}, 2}, {3.999999, {}, 3},
        {-1.0, {1.0, 1.0, 0.0}, 3}, {-1.0, {nan, 0.1, 0.0}, 3}, {INFINITY, {}, 4}, {-2.0, {}, 4}, {7.25, {}, 16},
    };
    for (const auto & cs : cases) {
        common_params_speculative p;
        p.synth_len = cs.len;
        p.synth_rates = cs.rates;
        printf("synth len=%a rates=%zu n_max=%d ->", cs.len, cs.rates.size(), cs.n_max);
        try {
            auto r = common_speculative_synth_rates_resolve(&p, cs.n_max);
            for (double x : r) printf(" %a", x);
            printf("\n");
        } catch (const std::exception & e) {
            printf(" invalid_argument: %s\n", e.what());
        }
    }
    for (int k = 0; k < 12; k++) {
        auto l = common_speculative_get_output_limits((k * 37) % 50, k % 5, k - 3);
        printf("limits %d %d %d -> %d %d\n", (k * 37) % 50, k % 5, k - 3, l.total, l.per_seq);
    }
}

static void api_gguf() {
    const char * env = getenv("SPEC_GGUFS");
    if (!env) return;
    std::string s = env;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(':', a);
        if (b == std::string::npos) b = s.size();
        std::string path = s.substr(a, b - a);
        if (!path.empty()) {
            auto t = common_speculative_types_from_gguf(path);
            printf("from_gguf %s -> '%s'\n", path.c_str(), common_speculative_type_name_str(t).c_str());
            // v0.7.0: common_get_gguf_info replaces common_get_decision_type(fname)
            const auto info = common_get_gguf_info(path);
            printf("gguf_info %s -> decision %d n_ctx_train %u\n", path.c_str(), (int) info.decision_type, info.n_ctx_train);
        }
        a = b + 1;
    }
}

static void api_ngram_simple() {
    stream_gen g(7);
    for (int k = 0; k < 40; k++) {
        auto toks = g.make(20 + k * 5, 6 + k % 20, 30 + k);
        common_ngram_simple_config cfg = { (uint16_t) (1 + k % 4), (uint16_t) (1 + k % 9) };
        llama_token sampled = toks.back();
        toks.pop_back();
        auto d = common_ngram_simple_draft(cfg, toks, sampled);
        printf("simple %d n=%d m=%d", k, cfg.size_ngram, cfg.size_mgram);
        print_tokens("", d);
        printf("\n");
    }
}

static void dump_map(const common_ngram_map & m) {
    size_t nz = 0;
    for (auto v : m.key_map) if (v) nz++;
    printf("  map keys=%zu last_begin=%zu created=%d key_idx=%zu val_idx=%u check=%zu kmlast=%u nz=%zu kmhash=%016" PRIx64 "\n",
        m.keys.size(), m.size_last_begin, (int) m.last_draft_created, m.last_draft_key_idx, m.last_draft_value_idx, m.idx_last_check,
        m.key_map_last_idx, nz, fnv(m.key_map.data(), m.key_map.size() * sizeof(uint32_t)));
    for (const auto & k : m.keys) {
        printf("   key %zu %zu %u:", k.key_idx, k.stat_idx, k.key_num);
        for (const auto & v : k.values) printf(" (%zu,%u,%d)", v.value_idx, v.value_num, v.n_accepted);
        printf("\n");
    }
}

static void api_ngram_map() {
    stream_gen g(11);
    for (int k = 0; k < 16; k++) {
        const bool key_only = k % 2 == 0;
        common_ngram_map m((uint16_t) (2 + k % 3), (uint16_t) (2 + k % 5), key_only, (uint16_t) (1 + k % 3));
        auto all = g.make(400, 5 + k % 7, 60);
        size_t len = 30 + k;
        common_ngram_map_begin(m, std::vector<llama_token>(all.begin(), all.begin() + len));
        printf("map %d n=%d m=%d key_only=%d hits=%d\n", k, m.size_key, m.size_value, (int) key_only, m.min_hits);
        for (int step = 0; step < 40 && len + 1 < all.size(); step++) {
            std::vector<llama_token> inp(all.begin(), all.begin() + len);
            std::vector<llama_token> draft;
            common_ngram_map_draft(m, inp, all[len], draft);
            printf("  step %d len %zu", step, len);
            print_tokens(" draft", draft);
            printf("\n");
            size_t acc = 0;
            while (acc < draft.size() && len + 1 + acc < all.size() && all[len + 1 + acc] == draft[acc]) acc++;
            common_ngram_map_accept(m, (uint16_t) acc);
            len += 1 + acc;
            if (step % 13 == 12) {
                // a shorter history, as after a reasoning block is dropped
                size_t back = len > 40 ? len - 25 : len;
                common_ngram_map_begin(m, std::vector<llama_token>(all.begin(), all.begin() + back));
                len = back;
                printf("  rebegin %zu\n", back);
            }
        }
        dump_map(m);
    }
}

static void api_ngram_mod() {
    stream_gen g(13);
    for (int k = 0; k < 6; k++) {
        common_ngram_mod mod((uint16_t) (2 + k), 97 + k * 31);
        auto toks = g.make(300, 4 + k * 3, 50);
        size_t n = mod.get_n();
        for (size_t i = 0; i + n < toks.size(); i++) mod.add(toks.data() + i);
        printf("mod %d n=%zu used=%zu size=%zu bytes=%zu:", k, n, mod.get_used(), mod.size(), mod.size_bytes());
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i + n < toks.size(); i++) {
            int32_t v = mod.get(toks.data() + i);
            size_t ix = mod.idx(toks.data() + i);
            h = fnv(&v, sizeof(v), h);
            h = fnv(&ix, sizeof(ix), h);
        }
        printf(" %016" PRIx64 "\n", h);
        mod.reset();
        printf("  reset used=%zu\n", mod.get_used());
    }
}

static void api_ngram_cache(const std::string & dir) {
    stream_gen g(17);
    for (int k = 0; k < 8; k++) {
        common_ngram_cache ctx, dyn, sta;
        auto a = g.make(200 + k * 40, 6 + k * 4, 55);
        auto b = g.make(150, 6 + k * 4, 55);
        common_ngram_cache_update(ctx, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, a, (int) a.size(), false);
        common_ngram_cache_update(sta, 2, 2, b, (int) b.size(), false);
        // grow `a` and update incrementally, as the drafter does
        auto a2 = a;
        auto more = g.make(30, 6 + k * 4, 55);
        a2.insert(a2.end(), more.begin(), more.end());
        common_ngram_cache_update(ctx, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, a2, (int) more.size(), false);
        common_ngram_cache_update(dyn, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, b, (int) b.size(), false);
        printf("cache %d\n", k);
        dump_cache(" ctx", ctx);
        dump_cache(" sta", sta);
        for (int d = 0; d < 20; d++) {
            std::vector<llama_token> inp(a2.begin(), a2.begin() + 10 + (d * 37) % (a2.size() - 10));
            std::vector<llama_token> draft = { inp.back() };
            inp.pop_back();
            inp.push_back(draft[0]);
            common_ngram_cache_draft(inp, draft, 1 + d % 9, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, ctx, dyn, sta);
            printf(" draft %d", d);
            print_tokens("", draft);
            printf("\n");
        }
        const std::string f1 = dir + "/cache_" + std::to_string(k) + ".bin";
        common_ngram_cache_save(ctx, f1);
        size_t sz = 0;
        uint64_t h = file_hash(f1, &sz);
        printf(" saved %zu %016" PRIx64 "\n", sz, h);
        auto loaded = common_ngram_cache_load(f1);
        dump_cache(" loaded", loaded);
        common_ngram_cache_merge(loaded, dyn);
        dump_cache(" merged", loaded);
        const std::string f2 = dir + "/merged_" + std::to_string(k) + ".bin";
        common_ngram_cache_save(loaded, f2);
        h = file_hash(f2, &sz);
        printf(" saved merged %zu %016" PRIx64 "\n", sz, h);
        auto copy = loaded;
        dump_cache(" copy", copy);
    }
    try {
        common_ngram_cache_load(dir + "/does-not-exist.bin");
    } catch (const std::exception & e) {
        printf("load missing: %s\n", e.what());
    }
}

static const char * rb_state(common_reasoning_budget_state s) {
    switch (s) {
        case REASONING_BUDGET_IDLE: return "idle";
        case REASONING_BUDGET_COUNTING: return "counting";
        case REASONING_BUDGET_FORCING: return "forcing";
        case REASONING_BUDGET_WAITING_UTF8: return "waiting_utf8";
        case REASONING_BUDGET_DONE: return "done";
    }
    return "?";
}

static void rb_print(const char * label, llama_sampler * s) {
    printf("  %s %s", label, rb_state(common_reasoning_budget_get_state(s)));
    const auto * em = common_reasoning_budget_get_end_match(s);
    if (em) print_tokens(" end", *em);
    llama_token_data d[12];
    for (int i = 0; i < 12; i++) d[i] = { i, 1.0f + i, 0.0f };
    llama_token_data_array a = { d, 12, -1, false };
    llama_sampler_apply(s, &a);
    printf(" apply:");
    for (int i = 0; i < 12; i++) if (std::isinf(d[i].logit)) printf(" -%d", i);
    printf("\n");
}

static void api_reasoning_budget(const llama_vocab * vocab) {
    stream_gen g(19);
    const std::vector<std::vector<llama_tokens>> starts = { {{1, 2}}, {{1}, {1, 2}, {}, {1}}, {{9, 1, 2}, {2}} };
    const std::vector<std::vector<llama_tokens>> ends = { {{3, 4}, {5}}, {{4}}, {{3, 4, 5}, {4, 5}, {3, 4}} };
    const std::vector<llama_tokens> forced = { {7, 3, 4}, {}, {8, 5} };
    const int budgets[] = { 0, 1, 2, 5, INT_MAX, -3 };
    int n = 0;
    for (size_t si = 0; si < starts.size(); si++) {
        for (int bi = 0; bi < 6; bi++) {
            for (int st = 0; st < 5; st += 1) {
                const bool use_vocab = (n % 3) != 0;
                llama_sampler * s = common_reasoning_budget_init(use_vocab ? vocab : nullptr, starts[si], ends[si], forced[si], budgets[bi], (common_reasoning_budget_state) st);
                printf("rb %d start=%zu budget=%d init=%s vocab=%d name=%s\n", n, si, budgets[bi], rb_state((common_reasoning_budget_state) st), (int) use_vocab, llama_sampler_name(s));
                llama_sampler * cl = nullptr;
                auto toks = g.make(60, 12, 20);
                for (size_t i = 0; i < toks.size(); i++) {
                    // non-ASCII tokens of the real vocabulary, for the UTF-8 wait
                    const llama_token t = (i % 11 == 10 && use_vocab) ? (llama_token) (100000 + (g.next() % 50000)) : toks[i];
                    llama_sampler_accept(s, t);
                    printf("  tok %d ->", t);
                    rb_print("", s);
                    if (i == 20) cl = llama_sampler_clone(s);
                    if (i == 30) printf("  force %d\n", (int) common_reasoning_budget_force(s));
                    if (i == 40 && cl) {
                        llama_sampler_copy(cl, s);
                        rb_print("copied", s);
                    }
                    if (i == 50) {
                        llama_sampler_reset(s);
                        rb_print("reset", s);
                    }
                }
                if (cl) {
                    llama_sampler_accept(cl, 1);
                    llama_sampler_accept(cl, 2);
                    rb_print("clone", cl);
                    llama_sampler_free(cl);
                }
                llama_sampler_free(s);
                n++;
            }
        }
    }
    printf("rb null %s %d %d\n", rb_state(common_reasoning_budget_get_state(nullptr)), (int) (common_reasoning_budget_get_end_match(nullptr) == nullptr), (int) common_reasoning_budget_force(nullptr));
}

// every one- to four-byte string over the bytes that decide a UTF-8 boundary
static void api_utf8() {
    const unsigned char bytes[] = { 0x00, 0x41, 0x7f, 0x80, 0xbf, 0xc0, 0xc3, 0xdf, 0xe0, 0xe6, 0xef, 0xf0, 0xf4, 0xf7, 0xf8, 0xff };
    const size_t nb = sizeof(bytes);
    uint64_t h = 1469598103934665603ULL;
    size_t n_complete = 0, n = 0;
    for (size_t len = 0; len <= 4; len++) {
        size_t total = 1;
        for (size_t k = 0; k < len; k++) total *= nb;
        for (size_t idx = 0; idx < total; idx++) {
            std::string str;
            size_t x = idx;
            for (size_t k = 0; k < len; k++) { str.push_back((char) bytes[x % nb]); x /= nb; }
            const bool c = common_utf8_is_complete(str);
            n_complete += c;
            n++;
            const uint8_t b = c;
            h = fnv(&b, 1, h);
        }
    }
    printf("utf8 %zu strings, %zu complete, %016" PRIx64 "\n", n, n_complete, h);
}

static void api_sampler_tables() {
    common_params_sampling p;
    printf("print default\n%s\n", p.print().c_str());
    p.penalty_last_n = 13; p.penalty_repeat = 1.1f; p.top_k = 7; p.top_p = 0.5f; p.temp = 0.123456f; p.mirostat = 2; p.adaptive_target = 0.33f;
    printf("print set\n%s\n", p.print().c_str());
    const std::vector<std::string> names = { "top_k", "TOP-K", "topk", "nucleus", "temp", "typ", "typ_p", "Min-P", "minp", "dry", "xtc", "infill", "penalties", "adaptive-p", "adaptivep", "top_n_sigma", "top-n-sigma", "topnsigma", "bogus", "", "top_k " };
    for (auto t : common_sampler_types_from_names(names)) printf("name -> %s %c\n", common_sampler_type_to_str(t).c_str(), common_sampler_type_to_chr(t));
    for (auto t : common_sampler_types_from_chars("dkypsmtxiea?Kz")) printf("chr -> %s\n", common_sampler_type_to_str(t).c_str());
    for (int t = 0; t <= 13; t++) printf("type %d %s %c\n", t, common_sampler_type_to_str((common_sampler_type) t).c_str(), common_sampler_type_to_chr((common_sampler_type) t));
}

// v0.7.0's common_sampling_add_preserved_tokens and common_sampling_add_grammar_triggers
static void print_sampling_lists(const common_params_sampling & s) {
    printf("  preserved");
    for (auto t : s.preserved_tokens) printf(" %d", t);
    printf("\n  triggers");
    for (const auto & t : s.grammar_triggers) printf(" {%d '%s' %d}", (int) t.type, t.value.c_str(), t.token);
    printf("\n");
}

static void api_sampling_lists(const llama_vocab * vocab) {
    if (!vocab) return;
    const std::vector<std::string> pres = { "<think>", "</think>", "hello world", "<|im_start|>", "", "<tool_call>", "</think>", " the", "the" };
    for (int k = 0; k < 6; k++) {
        common_params_sampling s;
        s.grammar_lazy = k % 2 == 1;
        std::vector<std::string> p(pres.begin(), pres.begin() + (k * 2) % pres.size());
        common_sampling_add_preserved_tokens(s, vocab, p);
        std::vector<common_grammar_trigger> trig;
        if (k >= 1) trig.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<think>" });
        if (k >= 2) trig.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "hello world" });
        if (k >= 3) trig.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "a|b" });
        if (k >= 3) { common_grammar_trigger t; t.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN; t.value = "x"; t.token = 42; trig.push_back(t); }
        if (k == 4) trig.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "the" });
        if (k == 5) trig.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, "^x$" });
        if (k == 1) trig.clear();
        printf("sampling lists %d lazy=%d\n", k, (int) s.grammar_lazy);
        try {
            common_sampling_add_grammar_triggers(s, vocab, trig);
        } catch (const std::exception & e) {
            printf("  runtime_error: %s\n", e.what());
        }
        print_sampling_lists(s);
    }
}

// the self-speculative drafters through the public API, without a model
static void api_spec() {
    stream_gen g(23);
    using T = common_speculative_type;
    const std::vector<std::vector<T>> combos = {
        {COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE}, {COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K}, {COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
        {COMMON_SPECULATIVE_TYPE_NGRAM_MOD}, {COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
        {COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
        {COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V}, {COMMON_SPECULATIVE_TYPE_NONE},
        {COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3}, {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
        {COMMON_SPECULATIVE_TYPE_NGRAM_MOD}, {COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    };
    for (size_t ci = 0; ci < combos.size(); ci++) {
        common_params_speculative p;
        p.types = combos[ci];
        p.ngram_simple.size_n = (uint16_t) (2 + ci % 3);
        p.ngram_simple.size_m = (uint16_t) (3 + ci % 5);
        p.ngram_map_k.size_n = 2;
        p.ngram_map_k.size_m = 4;
        p.ngram_map_k4v.size_n = 2;
        p.ngram_map_k4v.size_m = 3;
        p.ngram_map_k4v.min_hits = (uint16_t) (1 + ci % 2);
        p.ngram_mod.n_match = (int32_t) (2 + ci % 3);
        p.ngram_mod.n_max = ci == 11 ? 4 : 6;
        p.ngram_mod.n_min = ci == 10 ? 3 : 1;
        printf("spec %zu '%s'", ci, common_speculative_type_name_str(p.types).c_str());
        common_speculative * spec = nullptr;
        try {
            spec = common_speculative_init(p, 2);
        } catch (const std::exception & e) {
            printf(" exception: %s\n", e.what());
            continue;
        }
        printf(" -> %s n_max=%d\n", spec ? "yes" : "null", common_speculative_n_max(spec));
        if (!spec) continue;
        std::vector<std::vector<llama_token>> all(2), prompt(2), draft(2);
        size_t len[2] = { 40, 52 };
        for (int s = 0; s < 2; s++) {
            all[s] = g.make(700, 6 + (int) ci + s, 75);
            prompt[s].assign(all[s].begin(), all[s].begin() + len[s]);
            common_speculative_begin(spec, s, prompt[s]);
        }
        for (int step = 0; step < 70; step++) {
            for (int s = 0; s < 2; s++) {
                prompt[s].assign(all[s].begin(), all[s].begin() + len[s]);
                draft[s].clear();
                common_speculative_get_draft_params(spec, s) = { s == 1 && step % 9 == 4 ? false : true, step % 7 == 3 ? 2 : -1, (llama_pos) len[s], all[s][len[s]], &prompt[s], &draft[s] };
            }
            common_speculative_draft(spec);
            for (int s = 0; s < 2; s++) {
                size_t acc = 0;
                while (acc < draft[s].size() && all[s][len[s] + 1 + acc] == draft[s][acc]) acc++;
                if (step >= 12 && step < 24) acc = 0;
                // a quarter accepted exactly, the streak's boundary
                if (ci == 11) acc = draft[s].size() == 4 ? 1 : 0;
                printf("  step %d seq %d len %zu", step, s, len[s]);
                print_tokens(" draft", draft[s]);
                printf(" acc %zu\n", acc);
                common_speculative_accept(spec, s, (uint16_t) acc);
                len[s] += 1 + acc;
                if (step % 29 == 28) {
                    len[s] -= 20;
                    prompt[s].assign(all[s].begin(), all[s].begin() + len[s]);
                    common_speculative_begin(spec, s, prompt[s]);
                }
            }
        }
        common_speculative_print_stats(spec);
        common_speculative_free(spec);
    }
}

static void run_api(const common_params & params, const llama_vocab * vocab) {
    api_types();
    api_n_max();
    api_synth();
    api_gguf();
    api_ngram_simple();
    api_ngram_map();
    api_ngram_mod();
    const char * dir = getenv("SPEC_TMP");
    api_ngram_cache(dir ? dir : ".");
    api_reasoning_budget(vocab);
    api_utf8();
    api_sampler_tables();
    api_sampling_lists(vocab);
    // v0.7.0's common_log_set_stdo: off, a line is written nowhere; on again, the next one is
    common_log_set_stdo(common_log_main(), false);
    LOG_INF("%s: stdo off, not written\n", __func__);
    common_log_set_stdo(common_log_main(), true);
    LOG_INF("%s: stdo on, written\n", __func__);
    api_spec();
    (void) params;
}

static std::vector<llama_token> prompt_tokens(llama_context * ctx, const std::string & text) {
    if (text.rfind("#ids:", 0) == 0) {
        std::vector<llama_token> r;
        const char * p = text.c_str() + 5;
        while (*p) {
            r.push_back((llama_token) strtol(p, (char **) &p, 10));
            if (*p == ',') p++;
        }
        return r;
    }
    return common_tokenize(ctx, text, true, true);
}

static void print_cands(common_sampler * smpl, int s) {
    auto * cur = common_sampler_get_candidates(smpl, true);
    printf("  cand seq %d size %zu sel %" PRId64 " sorted %d:", s, cur->size, cur->selected, (int) cur->sorted);
    for (size_t i = 0; i < std::min<size_t>(cur->size, 6); i++) {
        float p = cur->data[i].p, l = cur->data[i].logit;
        uint32_t pb, lb;
        memcpy(&pb, &p, 4);
        memcpy(&lb, &l, 4);
        printf(" %d:%08x:%08x", cur->data[i].id, pb, lb);
    }
    printf("\n");
}

static int run_decode(common_params & params) {
    // --spec-synth-len and --spec-synth-rates are the server's options, not this example's
    if (const char * v = getenv("SPEC_SYNTH_LEN")) params.speculative.synth_len = atof(v);
    if (const char * v = getenv("SPEC_SYNTH_RATES")) {
        for (const char * p = v; *p;) {
            char * end;
            params.speculative.synth_rates.push_back(strtod(p, &end));
            p = *end == ',' ? end + 1 : end;
        }
    }
    const auto output_limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;
    printf("limits %d %d n_max(params)=%d\n", output_limits.total, output_limits.per_seq, common_speculative_n_max(&params.speculative));

    auto init_tgt = common_init_from_params(params);
    llama_model * model_tgt = init_tgt->model();
    llama_context * ctx_tgt = init_tgt->context();
    if (!model_tgt || !ctx_tgt) {
        printf("target failed\n");
        return 1;
    }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    common_speculative_init_result_ptr spec_init;
    {
        common_params params_dft = common_base_params_to_speculative(params);
        printf("params_dft model='%s' ngl=%d split=%d ctk=%d ctv=%d outs=%d per_seq=%d embd=%d pool=%d threads=%d/%d\n",
            params_dft.model.path.c_str(), params_dft.n_gpu_layers, (int) params_dft.split_mode, (int) params_dft.cache_type_k, (int) params_dft.cache_type_v,
            params_dft.n_outputs_max, params_dft.n_outputs_max_per_seq, (int) params_dft.embedding, (int) params_dft.pooling_type,
            params_dft.cpuparams.n_threads, params_dft.cpuparams_batch.n_threads);
        // v0.7.0: the draft's own n_ubatch and no MoE cache, beside the target's context parameters
        const llama_context_params cp_tgt = common_context_params_to_llama(params);
        const llama_context_params cp_dft = common_context_params_to_llama(params_dft);
        printf("cparams ubatch=%u/%u moe_cache=%zu/%zu\n", cp_tgt.n_ubatch, cp_dft.n_ubatch, cp_tgt.moe_cache_size, cp_dft.moe_cache_size);
        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
        printf("draft model=%d ctx=%d\n", (int) (spec_init->model() != nullptr), (int) (spec_init->context() != nullptr));
        // SPEC_SEQ_RM: the server's probe of both contexts, before anything is decoded; a draft
        // context with a state input decodes it with zeros (v0.7.0's common_batch_set_zero_state)
        if (getenv("SPEC_SEQ_RM")) {
            printf("seq_rm tgt=%d\n", (int) common_context_can_seq_rm(ctx_tgt));
            if (spec_init->context()) printf("seq_rm dft=%d\n", (int) common_context_can_seq_rm(spec_init->context()));
        }
    }
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    const int n_seq = std::max(1, params.n_parallel);
    std::vector<std::vector<llama_token>> inp(n_seq);
    {
        std::vector<std::string> parts;
        std::string s = params.prompt;
        size_t a = 0;
        while (true) {
            size_t b = s.find('|', a);
            parts.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
            if (b == std::string::npos) break;
            a = b + 1;
        }
        for (int i = 0; i < n_seq; i++) {
            inp[i] = prompt_tokens(ctx_tgt, parts[i % parts.size()]);
            print_tokens("prompt", inp[i]);
            printf("\n");
        }
    }

    // the target samplers, with an optional reasoning budget
    std::vector<common_sampler *> smpl(n_seq);
    if (const char * rb = getenv("SPEC_RBUDGET")) {
        auto tok = [&](const char * env, const char * def) {
            const char * v = getenv(env);
            return common_tokenize(ctx_tgt, v ? v : def, false, true);
        };
        params.sampling.reasoning_budget_tokens = atoi(rb);
        params.sampling.reasoning_budget_start = tok("SPEC_RB_START", "<think>");
        params.sampling.reasoning_budget_end = { tok("SPEC_RB_END", "</think>") };
        params.sampling.reasoning_budget_forced = tok("SPEC_RB_FORCED", " ok</think>");
    }
    if (const char * gp = getenv("SPEC_GEN_PROMPT")) params.sampling.generation_prompt = gp;
    // a lazy grammar's triggers and a tool-call grammar, which no option sets
    if (const char * t = getenv("SPEC_TRIGGER"))         { params.sampling.grammar_lazy = true; params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, t}); }
    if (const char * t = getenv("SPEC_TRIGGER_PATTERN")) { params.sampling.grammar_lazy = true; params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, t}); }
    if (const char * t = getenv("SPEC_TRIGGER_FULL"))    { params.sampling.grammar_lazy = true; params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, t}); }
    if (const char * t = getenv("SPEC_TRIGGER_TOKEN"))   { params.sampling.grammar_lazy = true; common_grammar_trigger tr; tr.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN; tr.token = atoi(t); params.sampling.grammar_triggers.push_back(tr); }
    if (getenv("SPEC_TOOL_GRAMMAR") && !params.sampling.grammar.grammar.empty()) params.sampling.grammar.type = COMMON_GRAMMAR_TYPE_TOOL_CALLS;
    // --n-probs is the server's and completion's option, not this example's
    if (const char * v = getenv("SPEC_N_PROBS")) params.sampling.n_probs = atoi(v);
    for (int s = 0; s < n_seq; s++) {
        try {
            smpl[s] = common_sampler_init(model_tgt, params.sampling);
        } catch (const std::exception & e) {
            printf("exception: %s\n", e.what());
            return 3;
        }
        printf("sampler %d: %s seed=%u\n", s, common_sampler_print(smpl[s]).c_str(), common_sampler_get_seed(smpl[s]));
    }

    // -lcs and -lcd are the lookup and server examples' options, not this one's
    if (const char * v = getenv("SPEC_LCS")) params.speculative.ngram_cache.lookup_cache_static = v;
    if (const char * v = getenv("SPEC_LCD")) params.speculative.ngram_cache.lookup_cache_dynamic = v;
    common_speculative * spec = nullptr;
    try {
        spec = common_speculative_init(params.speculative, n_seq);
    } catch (const std::exception & e) {
        printf("exception: %s\n", e.what());
        return 3;
    }
    printf("spec %s n_max=%d synth=%zu:", spec ? "yes" : "null", common_speculative_n_max(spec), spec ? common_speculative_get_synth_probs(spec).size() : 0);
    if (spec) for (double p : common_speculative_get_synth_probs(spec)) printf(" %a", p);
    printf("\n");

    // prefill every sequence but its last token
    {
        int n = 0;
        for (auto & v : inp) n += (int) v.size();
        (void) n;
        common_batch b(ctx_tgt);
        // SPEC_PREFILL_EMBD: the prefill as embedding rows, as an image's are, a
        // fixed pattern per position (v0.7.0's MTP drafter catches them up too)
        std::vector<float> embd;
        const bool as_embd = getenv("SPEC_PREFILL_EMBD") != nullptr;
        if (as_embd) {
            const size_t n_embd = (size_t) llama_model_n_embd_inp(model_tgt);
            size_t n_rows = 0;
            for (int s = 0; s < n_seq; s++) n_rows += inp[s].size() > 0 ? inp[s].size() - 1 : 0;
            embd.resize(n_rows * n_embd);
            for (size_t r = 0; r < n_rows; r++) {
                for (size_t j = 0; j < n_embd; j++) embd[r * n_embd + j] = (float) (int) ((r * 31 + j * 7) % 97) / 97.0f - 0.5f;
            }
            size_t r = 0;
            for (int s = 0; s < n_seq; s++) {
                for (size_t i = 0; i + 1 < inp[s].size(); i++, r++) {
                    const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
                    b.add_embd({ embd.data() + r * n_embd, 1, n_embd }, pos, s, false);
                }
            }
        } else {
            for (int s = 0; s < n_seq; s++) {
                for (size_t i = 0; i + 1 < inp[s].size(); i++) b.add(inp[s][i], (llama_pos) i, s, false);
            }
        }
        int rc = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, b.get());
        printf("prefill rc=%d\n", rc);
        printf("process prefill %d\n", (int) common_speculative_process(spec, b));
    }

    std::vector<llama_token> id_last(n_seq);
    std::vector<llama_tokens> prompt_tgt(n_seq);
    std::vector<int> n_past(n_seq), n_gen(n_seq, 0);
    std::vector<bool> done(n_seq, false);
    std::vector<llama_tokens> draft(n_seq);
    for (int s = 0; s < n_seq; s++) {
        id_last[s] = inp[s].back();
        prompt_tgt[s].assign(inp[s].begin(), inp[s].end() - 1);
        n_past[s] = (int) inp[s].size() - 1;
        common_speculative_begin(spec, s, prompt_tgt[s]);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    common_batch batch_tgt(ctx_tgt);
    const int n_predict = params.n_predict < 0 ? 32 : params.n_predict;
    // verify by rejection, as the server does at temp > 0: the drafters keep their candidates
    const bool reject = getenv("SPEC_REJECT") != nullptr;
    std::vector<std::vector<std::vector<llama_token_data>>> draft_q(n_seq);

    for (int it = 0; it < 4 * n_predict; it++) {
        bool any = false;
        for (int s = 0; s < n_seq; s++) {
            if (done[s]) continue;
            any = true;
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) / n_seq - n_past[s] - 2;
            n_draft_max = std::min(n_draft_max, n_predict - n_gen[s] - 1);
            n_draft_max = std::max(n_draft_max, 0);
            draft[s].clear();
            draft_q[s].clear();
            if (spec) {
                auto & dp = common_speculative_get_draft_params(spec, s);
                dp = { true, n_draft_max, n_past[s], id_last[s], &prompt_tgt[s], &draft[s] };
                if (reject) {
                    dp.result_q = &draft_q[s];
                    dp.temp     = params.sampling.temp;
                    dp.seed     = params.sampling.seed;
                }
            }
        }
        if (!any) break;
        common_speculative_draft(spec);

        batch_tgt.clear();
        std::vector<std::vector<int>> idxs(n_seq);
        for (int s = 0; s < n_seq; s++) {
            if (done[s]) continue;
            printf("it %d seq %d pos %d last %d", it, s, n_past[s], id_last[s]);
            print_tokens(" draft", draft[s]);
            if (reject) {
                uint64_t h = fnv(nullptr, 0);
                for (const auto & q : draft_q[s]) {
                    const size_t n = q.size();
                    h = fnv(&n, sizeof(n), h);
                    if (n) h = fnv(q.data(), n * sizeof(llama_token_data), h);
                }
                printf(" q %zu %016" PRIx64, draft_q[s].size(), h);
            }
            printf("\n");
            if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), s, n_past[s], -1);
            idxs[s].push_back(batch_tgt.size());
            batch_tgt.add(id_last[s], n_past[s], s, true);
            for (size_t i = 0; i < draft[s].size(); i++) {
                idxs[s].push_back(batch_tgt.size());
                batch_tgt.add(draft[s][i], n_past[s] + 1 + (int) i, s, true);
            }
        }
        int rc = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        if (rc != 0) { printf("decode rc=%d\n", rc); break; }
        if (!common_speculative_process(spec, batch_tgt)) { printf("process failed\n"); break; }

        for (int s = 0; s < n_seq; s++) {
            if (done[s]) continue;
            common_sampler * keep = (it % 3 == 1) ? common_sampler_clone(smpl[s]) : nullptr;
            // the server's rule: rejection only over candidates for every drafted token
            const bool use_rejection = reject && !draft_q[s].empty() && draft_q[s].size() == draft[s].size();
            auto ids = use_rejection
                ? common_sampler_sample_and_accept_n_rejection(smpl[s], ctx_tgt, idxs[s], draft[s], draft_q[s])
                : common_sampler_sample_and_accept_n(smpl[s], ctx_tgt, idxs[s], draft[s]);
            if (use_rejection) printf("  rejection\n");
            print_tokens("  ids", ids);
            printf(" acc %zu last %d\n", ids.size() - 1, common_sampler_last(smpl[s]));
            print_cands(smpl[s], s);
            if (keep) {
                // the clone, given the same samples, reaches the same state; copied back it must agree
                for (auto t : ids) common_sampler_accept(keep, t, true);
                common_sampler_copy(keep, smpl[s]);
                common_sampler_free(keep);
                if (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE) printf("  prev '%s'\n", common_sampler_prev_str(smpl[s], ctx_tgt, 6).c_str());
            }
            if (spec) common_speculative_accept(spec, s, (uint16_t) (ids.size() - 1));
            std::vector<uint8_t> st;
            if (common_speculative_get_state(spec, s, st)) {
                printf("  state %zu %016" PRIx64 "\n", st.size(), fnv(st.data(), st.size()));
                common_speculative_set_state(spec, s, st);
            }
            prompt_tgt[s].push_back(id_last[s]);
            prompt_tgt[s].insert(prompt_tgt[s].end(), ids.begin(), ids.end() - 1);
            n_past[s] += (int) ids.size();
            id_last[s] = ids.back();
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), s, n_past[s], -1);
            n_gen[s] += (int) ids.size();
            if (n_gen[s] >= n_predict || llama_vocab_is_eog(vocab, id_last[s])) done[s] = true;
        }
    }
    for (int s = 0; s < n_seq; s++) {
        printf("seq %d generated %d", s, n_gen[s]);
        print_tokens(" all", prompt_tgt[s]);
        printf("\n");
        if (s == 0 && getenv("SPEC_RBUDGET")) printf("rb force %d\n", (int) common_sampler_reasoning_budget_force(smpl[s]));
    }
    common_speculative_print_stats(spec);
    common_speculative_free(spec);
    for (auto * s : smpl) common_sampler_free(s);
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) return 2;
    const std::string mode = argv[1];
    argv[1] = argv[0];

    common_params params;
    common_init();
    try {
        if (!common_params_parse(argc - 1, argv + 1, params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;
    } catch (const std::exception & e) {
        printf("parse: %s\n", e.what());
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    int rc = 0;
    if (mode == "api") {
        // a vocabulary for the reasoning budget's UTF-8 checks
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * m = llama_model_load_from_file(params.model.path.c_str(), mp);
        run_api(params, m ? llama_model_get_vocab(m) : nullptr);
        if (m) llama_model_free(m);
    } else {
        try {
            rc = run_decode(params);
        } catch (const std::exception & e) {
            printf("exception: %s\n", e.what());
            rc = 3;
        }
    }
    fflush(stdout);
    common_log_pause(common_log_main());
    llama_backend_free();
    return rc;
}
