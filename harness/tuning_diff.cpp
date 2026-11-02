/* Sweep the flash-attention tuning lookup against the reference, exhaustively.
 *
 * **Not a port.** This is a gate, driven by `scripts/tuning-diff`.
 *
 * Why it exists: `src/ggml/metal/tuning_table.zig` is 936 rows transcribed
 * from a generated C table. `make node-diff ARGS=--gpu` only exercises the
 * handful of (dtype, dk, dv, ne11, ne01) a Qwen3.5 decode reaches on this
 * one SKU, and `backend-ops` only those its FA cases cover. Neither can
 * tell a dropped or mistyped row from a correct one.
 *
 * So the whole key space is swept instead: every GPU family the table
 * names and every one around them, families past int8_t that the C's cast
 * wraps onto a tuned one, every KV type the table mentions plus ones it
 * does not, every (dk, dv) pair the Metal kernels instantiate and some they
 * do not, and both sides of every bucket edge. Since v0.5.0 the table is
 * keyed by family (7 to 10) rather than by SKU, with no fallback. The reference is compiled with its seven entry points renamed
 * via -D, so both implementations live in one process.
 */
#include "ggml-metal-tuning.h"

#include <cstdio>
#include <cstdint>
#include <vector>

/* Both sides, declared explicitly.
 *
 * **This file must be compiled WITHOUT the -D renames.** They are what
 * turns the reference's definitions into `ref_*`; applying them here
 * renames these declarations too, so every comparison calls the reference
 * twice and passes no matter what the port does. Measured: with the
 * renames on, four injected faults -- a dropped table row, a bucket edge
 * off by one, a missing `baseline_ne` pair, and a dropped family
 * fallback -- all passed. `scripts/tuning-diff` therefore renames only
 * the reference translation unit.
 *
 * The `ref_*` names are spelled out here rather than obtained from the
 * header, which is the whole point: the header gives the real names, and
 * those are ours. */
namespace ggml_metal_tuning {
    int          fa_vec_ne11_bucket(int64_t);
    int          fa_vec_ne01_bucket(int64_t);
    int          fa_vec_baseline_ne(int, int);
    fa_vec_cfg_t fa_vec_baseline_cfg(int, int);
    void         fa_vec_set_override(fa_vec_cfg_t);
    void         fa_vec_clear_override();
    fa_vec_cfg_t fa_vec_pick(int, int, int, int, int64_t, int64_t);

    int          ref_fa_vec_ne11_bucket(int64_t);
    int          ref_fa_vec_ne01_bucket(int64_t);
    int          ref_fa_vec_baseline_ne(int, int);
    fa_vec_cfg_t ref_fa_vec_baseline_cfg(int, int);
    void         ref_fa_vec_set_override(fa_vec_cfg_t);
    void         ref_fa_vec_clear_override();
    fa_vec_cfg_t ref_fa_vec_pick(int, int, int, int, int64_t, int64_t);
}

static long checks = 0;
static long fails  = 0;

static void cmp(const char *what, int got, int want) {
    checks++;
    if (got != want) {
        if (fails < 12) printf("  %-58s got %d, want %d   WRONG\n", what, got, want);
        fails++;
    }
}

static void cmpcfg(const char *what, ggml_metal_tuning::fa_vec_cfg_t got,
                                     ggml_metal_tuning::fa_vec_cfg_t want) {
    checks++;
    if (got.Q != want.Q || got.NE != want.NE) {
        if (fails < 12) printf("  %-58s got {%d,%d}, want {%d,%d}   WRONG\n",
                               what, got.Q, got.NE, want.Q, want.NE);
        fails++;
    }
}

int main() {
    using namespace ggml_metal_tuning;
    char lbl[128];

    /* bucket functions: both sides of every edge, plus extremes */
    {
        std::vector<int64_t> ns = { 0, 1, 2, 3, 4, 5, 6, 1023, 1024, 1025,
                                    4095, 4096, 4097, 16383, 16384, 16385,
                                    1 << 20, (int64_t) 1 << 40 };
        for (int64_t n : ns) {
            snprintf(lbl, sizeof lbl, "ne11_bucket(%lld)", (long long) n);
            cmp(lbl, fa_vec_ne11_bucket(n), ref_fa_vec_ne11_bucket(n));
            snprintf(lbl, sizeof lbl, "ne01_bucket(%lld)", (long long) n);
            cmp(lbl, fa_vec_ne01_bucket(n), ref_fa_vec_ne01_bucket(n));
        }
    }

    /* baseline_ne / baseline_cfg over every head-size pair that matters,
     * plus pairs with no instantiation so the default arm is covered */
    const int dims[] = { 0, 1, 32, 40, 64, 80, 96, 112, 128, 192, 256, 320, 512, 576, 1024 };
    for (int dk : dims) {
        for (int dv : dims) {
            snprintf(lbl, sizeof lbl, "baseline_ne(%d,%d)", dk, dv);
            cmp(lbl, fa_vec_baseline_ne(dk, dv), ref_fa_vec_baseline_ne(dk, dv));
            snprintf(lbl, sizeof lbl, "baseline_cfg(%d,%d)", dk, dv);
            cmpcfg(lbl, fa_vec_baseline_cfg(dk, dv), ref_fa_vec_baseline_cfg(dk, dv));
        }
    }

    /* fa_vec_pick over the whole key space the table can express */
    const int types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                          GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                          GGML_TYPE_BF16, GGML_TYPE_Q4_K };
    const int64_t ne11s[] = { 0, 1, 1023, 1024, 1025, 2048, 4095, 4096, 4097, 8192,
                              16383, 16384, 16385, 65536, (int64_t) 1 << 40 };
    const int64_t ne01s[] = { 0, 1, 2, 3, 4, 5, 6, 16, (int64_t) 1 << 33 };
    const int dks[] = { 0, 32, 64, 72, 80, 96, 112, 128, 192, 256, 320, 512, 576, 576 + 65536 };
    const int dvs[] = { 0, 32, 64, 72, 96, 128, 192, 256, 320, 512, 512 + 65536 };

    // Every family from below 0 to past the tuned ones -- an untuned family
    // must miss the table on both sides -- plus values past int8_t that the
    // C's `(int8_t) gpu_family` wraps onto 7..10 (263 is 7, -246 is 10):
    // a port that range-checked or saturated instead would differ there.
    // dk and dv past int16_t do the same for `(int16_t) dk`. The table's
    // families at v0.5.0 are 7..10; `gen-tuning-table` stops on any other.
    std::vector<int> families;
    for (int fam = -2; fam <= 40; fam++) families.push_back(fam);
    for (int fam : { 127, 128, 255, 256, 263, 264, 265, 266, -246, -249, 7 + 1024, 10 - 4096 })
        families.push_back(fam);
    for (int fam : families) {
        for (int ty : types) {
            for (int dk : dks) {
                for (int dv : dvs) {
                    for (int64_t n11 : ne11s) {
                        for (int64_t n01 : ne01s) {
                            auto a =     fa_vec_pick(fam, ty, dk, dv, n11, n01);
                            auto b = ref_fa_vec_pick(fam, ty, dk, dv, n11, n01);
                            checks++;
                            if (a.Q != b.Q || a.NE != b.NE) {
                                if (fails < 12) {
                                    printf("  pick(fam=%d ty=%d dk=%d dv=%d ne11=%lld ne01=%lld)"
                                           " got {%d,%d}, want {%d,%d}   WRONG\n",
                                           fam, ty, dk, dv, (long long) n11, (long long) n01,
                                           a.Q, a.NE, b.Q, b.NE);
                                }
                                fails++;
                            }
                        }
                    }
                }
            }
        }
    }

    /* The sweep above compares, but a table whose every answer equals
     * the baseline would pass it without testing a row. Count how many
     * lookups in the tuned families the reference answers with something
     * other than the baseline, at one representative ne11/ne01 per
     * bucket: the table must be reachable from the grid. */
    {
        const int64_t ne11_of[] = { 2048, 8192, 32768 };   // buckets 1, 2, 3
        const int64_t ne01_of[] = { 1, 2, 3, 4, 5 };        // buckets 0..4
        long tuned = 0;
        for (int fam = 7; fam <= 10; fam++) {
            for (int ty : types) {
                for (int dk : dks) {
                    for (int dv : dvs) {
                        const auto base = ref_fa_vec_baseline_cfg(dk, dv);
                        for (int64_t n11 : ne11_of) {
                            for (int64_t n01 : ne01_of) {
                                auto b = ref_fa_vec_pick(fam, ty, dk, dv, n11, n01);
                                if (b.Q != base.Q || b.NE != base.NE) tuned++;
                            }
                        }
                    }
                }
            }
        }
        // A sweep whose tuned answers all equal the baseline proves nothing
        // about the table: at least some lookups must land on a tuned row.
        checks++;
        if (tuned < 1000) {
            printf("  %-58s only %ld lookups differ from baseline   WRONG\n", "table reached", tuned);
            fails++;
        }
        printf("lookups that a tuned row answered differently from baseline: %ld\n", tuned);
    }

    /* the override path, and that clearing it restores the table */
    {
        fa_vec_cfg_t forced { (int8_t) 7, (int8_t) 3 };
        fa_vec_set_override(forced);
        ref_fa_vec_set_override(forced);
        cmpcfg("pick under override",
               fa_vec_pick(9, GGML_TYPE_F16, 128, 128, 8192, 1),
               ref_fa_vec_pick(9, GGML_TYPE_F16, 128, 128, 8192, 1));
        checks++;
        if (fa_vec_pick(9, GGML_TYPE_F16, 128, 128, 8192, 1).Q != 7) {
            printf("  %-58s override not applied   WRONG\n", "pick under override");
            fails++;
        }
        fa_vec_clear_override();
        ref_fa_vec_clear_override();
        cmpcfg("pick after clear",
               fa_vec_pick(9, GGML_TYPE_F16, 128, 128, 8192, 1),
               ref_fa_vec_pick(9, GGML_TYPE_F16, 128, 128, 8192, 1));
    }

    printf("\n");
    if (fails == 0) {
        printf("PASS: %ld tuning lookups identical, ported vs reference\n", checks);
        return 0;
    }
    printf("FAIL: %ld of %ld tuning lookups differ\n", fails, checks);
    return 1;
}
