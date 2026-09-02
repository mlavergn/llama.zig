// Localise a dot-product mismatch by asking the shipped kernel for every
// prefix of a row.
//
// Why this exists
// ---------------
// A golden mismatch says the kernel is wrong; it does not say where. The
// prefix sums do: `nblk = 1` exercises only the scalar tail, `nblk = 2` only
// one unrolled iteration, and the first prefix that diverges says which.
//
// It was written for `ggml_vec_dot_iq4_nl_q8_0`, whose result differed from
// the port by one ULP. The prefixes showed the divergence at `nblk = 2` --
// before any accumulation order could matter -- which meant the per-iteration
// *term* was wrong, not the summation. Eight candidate orderings later,
// exactly one reproduced all sixteen values: `-ffp-contract=on` fuses the left
// multiply into the inner add, and fuses the tail's multiply into its
// accumulate.
//
// This is the difference between naming fusion sites soundly and guessing at
// them. `src/ggml/quants/helpers.zig` records why guessing failed there; here
// there are sixteen equations per kernel.
//
// Usage: scripts/vecdot-prefix [kernel]
//   with no argument, lists the kernels it knows.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NELEM 512

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

struct entry {
    const char *   name;
    enum ggml_type xt;
    enum ggml_type yt;
    int            blk;   // elements per x block, so a prefix is a block count
    dot_fn         fn;
};

static const struct entry entries[] = {
    { "q1_0",    GGML_TYPE_Q1_0,    GGML_TYPE_Q8_0, QK1_0,     ggml_vec_dot_q1_0_q8_0    },
    { "q2_0",    GGML_TYPE_Q2_0,    GGML_TYPE_Q8_0, QK2_0,     ggml_vec_dot_q2_0_q8_0    },
    { "q4_0",    GGML_TYPE_Q4_0,    GGML_TYPE_Q8_0, QK8_0,     ggml_vec_dot_q4_0_q8_0    },
    { "q4_1",    GGML_TYPE_Q4_1,    GGML_TYPE_Q8_1, QK8_1,     ggml_vec_dot_q4_1_q8_1    },
    { "q5_0",    GGML_TYPE_Q5_0,    GGML_TYPE_Q8_0, QK8_0,     ggml_vec_dot_q5_0_q8_0    },
    { "q5_1",    GGML_TYPE_Q5_1,    GGML_TYPE_Q8_1, QK8_1,     ggml_vec_dot_q5_1_q8_1    },
    { "q8_0",    GGML_TYPE_Q8_0,    GGML_TYPE_Q8_0, QK8_0,     ggml_vec_dot_q8_0_q8_0    },
    { "mxfp4",   GGML_TYPE_MXFP4,   GGML_TYPE_Q8_0, QK_MXFP4,  ggml_vec_dot_mxfp4_q8_0   },
    { "nvfp4",   GGML_TYPE_NVFP4,   GGML_TYPE_Q8_0, QK_NVFP4,  ggml_vec_dot_nvfp4_q8_0   },
    { "q2_K",    GGML_TYPE_Q2_K,    GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_q2_K_q8_K    },
    { "q3_K",    GGML_TYPE_Q3_K,    GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_q3_K_q8_K    },
    { "q4_K",    GGML_TYPE_Q4_K,    GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_q4_K_q8_K    },
    { "q5_K",    GGML_TYPE_Q5_K,    GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_q5_K_q8_K    },
    { "q6_K",    GGML_TYPE_Q6_K,    GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_q6_K_q8_K    },
    { "tq1_0",   GGML_TYPE_TQ1_0,   GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_tq1_0_q8_K   },
    { "tq2_0",   GGML_TYPE_TQ2_0,   GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_tq2_0_q8_K   },
    { "iq2_xxs", GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_iq2_xxs_q8_K },
    { "iq2_xs",  GGML_TYPE_IQ2_XS,  GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_iq2_xs_q8_K  },
    { "iq2_s",   GGML_TYPE_IQ2_S,   GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_iq2_s_q8_K   },
    { "iq3_xxs", GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_iq3_xxs_q8_K },
    { "iq3_s",   GGML_TYPE_IQ3_S,   GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_iq3_s_q8_K   },
    { "iq1_s",   GGML_TYPE_IQ1_S,   GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_iq1_s_q8_K   },
    { "iq1_m",   GGML_TYPE_IQ1_M,   GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_iq1_m_q8_K   },
    { "iq4_nl",  GGML_TYPE_IQ4_NL,  GGML_TYPE_Q8_0, QK4_NL,    ggml_vec_dot_iq4_nl_q8_0  },
    { "iq4_xs",  GGML_TYPE_IQ4_XS,  GGML_TYPE_Q8_K, QK_K,      ggml_vec_dot_iq4_xs_q8_K  },
};

#define NENTRIES (sizeof(entries) / sizeof(entries[0]))

static uint32_t st = 1;
static float nx(void) { st = 1103515245u * st + 12345u; return ((float) (st >> 16) / 32768.0f) - 1.0f; }

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <kernel>\nknown:", argv[0]);
        for (size_t i = 0; i < NENTRIES; i++) fprintf(stderr, " %s", entries[i].name);
        fprintf(stderr, "\n");
        return 2;
    }

    const struct entry * en = NULL;
    for (size_t i = 0; i < NENTRIES; i++) {
        if (strcmp(argv[1], entries[i].name) == 0) { en = &entries[i]; break; }
    }
    if (!en) { fprintf(stderr, "unknown kernel: %s\n", argv[1]); return 2; }

    ggml_cpu_init();

    static float xf[NELEM], yf[NELEM], im[NELEM];
    static uint8_t xq[NELEM * 4], yq[NELEM * 4];

    // The `random` pattern from harness/vecdot_golden.c, so a prefix run and a
    // golden run see the same data.
    st = 1;
    for (int i = 0; i < NELEM; i++) { xf[i] = nx(); yf[i] = nx(); }
    for (int i = 0; i < NELEM; i++) im[i] = 1.0f;

    memset(xq, 0, sizeof xq);
    memset(yq, 0, sizeof yq);
    ggml_quantize_chunk(en->xt, xf, xq, 0, 1, NELEM, im);
    switch (en->yt) {
        case GGML_TYPE_Q8_0: quantize_row_q8_0_ref(yf, (void *) yq, NELEM); break;
        case GGML_TYPE_Q8_1: quantize_row_q8_1_ref(yf, (void *) yq, NELEM); break;
        case GGML_TYPE_Q8_K: quantize_row_q8_K_ref(yf, (void *) yq, NELEM); break;
        default: fprintf(stderr, "unexpected vec_dot_type\n"); return 1;
    }

    const int max_blk = NELEM / en->blk;
    printf("%s: %d blocks of %d elements\n", en->name, max_blk, en->blk);
    printf("blocks  bits         value\n");
    for (int nblk = 1; nblk <= max_blk; nblk++) {
        float s = 0;
        en->fn(nblk * en->blk, &s, 0, xq, 0, yq, 0, 1);
        uint32_t b;
        memcpy(&b, &s, sizeof b);
        printf("%6d  0x%08X   %.9g\n", nblk, b, (double) s);
    }
    return 0;
}
