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
 * So the whole key space is swept instead: every device id, every KV type
 * the table mentions plus ones it does not, every (dk, dv) pair the Metal
 * kernels instantiate, both sides of every bucket edge, and the family
 * fallback. The reference is compiled with its seven entry points renamed
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
    fa_vec_cfg_t fa_vec_pick(enum ggml_metal_device_id, int, int, int, int, int64_t, int64_t);

    int          ref_fa_vec_ne11_bucket(int64_t);
    int          ref_fa_vec_ne01_bucket(int64_t);
    int          ref_fa_vec_baseline_ne(int, int);
    fa_vec_cfg_t ref_fa_vec_baseline_cfg(int, int);
    void         ref_fa_vec_set_override(fa_vec_cfg_t);
    void         ref_fa_vec_clear_override();
    fa_vec_cfg_t ref_fa_vec_pick(enum ggml_metal_device_id, int, int, int, int, int64_t, int64_t);
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
    const int types[] = { GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                          GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                          GGML_TYPE_BF16, GGML_TYPE_Q4_K };
    const int64_t ne11s[] = { 1, 1023, 1024, 2048, 4095, 4096, 8192, 16383, 16384, 65536 };
    const int64_t ne01s[] = { 1, 2, 3, 4, 5, 6, 16 };
    const int families[] = { 0, 7, 8, 9, 10 };

    for (int dev = 0; dev <= GGML_METAL_DEVICE_M5_ULTRA; dev++) {
        for (int fam : families) {
            for (int ty : types) {
                for (int dk : { 32, 64, 96, 128, 192, 256, 320, 512, 576 }) {
                    for (int dv : { 32, 64, 96, 128, 192, 256, 512 }) {
                        for (int64_t n11 : ne11s) {
                            for (int64_t n01 : ne01s) {
                                auto a =     fa_vec_pick((enum ggml_metal_device_id) dev, fam, ty, dk, dv, n11, n01);
                                auto b = ref_fa_vec_pick((enum ggml_metal_device_id) dev, fam, ty, dk, dv, n11, n01);
                                checks++;
                                if (a.Q != b.Q || a.NE != b.NE) {
                                    if (fails < 12) {
                                        printf("  pick(dev=%d fam=%d ty=%d dk=%d dv=%d ne11=%lld ne01=%lld)"
                                               " got {%d,%d}, want {%d,%d}   WRONG\n",
                                               dev, fam, ty, dk, dv, (long long) n11, (long long) n01,
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
    }

    /* the override path, and that clearing it restores the table */
    {
        fa_vec_cfg_t forced { (int8_t) 7, (int8_t) 3 };
        fa_vec_set_override(forced);
        ref_fa_vec_set_override(forced);
        cmpcfg("pick under override",
               fa_vec_pick(GGML_METAL_DEVICE_M4_MAX, 9, GGML_TYPE_F16, 128, 128, 8192, 1),
               ref_fa_vec_pick(GGML_METAL_DEVICE_M4_MAX, 9, GGML_TYPE_F16, 128, 128, 8192, 1));
        checks++;
        if (fa_vec_pick(GGML_METAL_DEVICE_M4_MAX, 9, GGML_TYPE_F16, 128, 128, 8192, 1).Q != 7) {
            printf("  %-58s override not applied   WRONG\n", "pick under override");
            fails++;
        }
        fa_vec_clear_override();
        ref_fa_vec_clear_override();
        cmpcfg("pick after clear",
               fa_vec_pick(GGML_METAL_DEVICE_M4_MAX, 9, GGML_TYPE_F16, 128, 128, 8192, 1),
               ref_fa_vec_pick(GGML_METAL_DEVICE_M4_MAX, 9, GGML_TYPE_F16, 128, 128, 8192, 1));
    }

    printf("\n");
    if (fails == 0) {
        printf("PASS: %ld tuning lookups identical, ported vs reference\n", checks);
        return 0;
    }
    printf("FAIL: %ld of %ld tuning lookups differ\n", fails, checks);
    return 1;
}
