// Capture the reference behaviour of every `ggml_vec_dot_*_generic` in
// ggml-cpu/quants.c.
//
// Emits a Zig source file of golden values: for each quantized type, the exact
// bits of the float that `ggml_vec_dot_<type>_<vec_dot_type>_generic` writes,
// for each of several input patterns.
//
// Why goldens are the only option here
// ------------------------------------
// On this target nothing calls these functions. `arch-fallback.h` renames
// nothing from quants.c on ARM, and ggml-cpu/arch/arm/quants.c supplies all 28
// real entry points, so the `_generic` names are exported and unreachable.
// `test-backend-ops` cannot reach them, and neither can token parity. Values
// captured from the C are the whole gate.
//
// Bits, not floats
// ----------------
// The result is recorded as the raw u32 so the comparison is exact rather than
// approximate. A dot product that is right to six decimals and wrong in the
// last bit is a porting bug, and this is meant to catch porting bugs.
//
// Inputs are a fixed LCG so this is reproducible, and captured once at the
// pinned commit.
//
// Build and run: see scripts/vecdot-golden

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

// 512 elements is divisible by every block size in play: 256 (QK_K), 128
// (QK1_0), 64 (QK2_0, QK_NVFP4) and 32 (the legacy formats). Two K-quant
// super-blocks, so a bug in the per-block loop shows up rather than being
// masked by there being only one block.
#define NELEM 512

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*row_fn)(const float *, void *, int64_t);

static uint64_t fnv(const void * p, size_t n) {
    const uint8_t * b = p;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// The row quantizers `type_traits_cpu` puts in its `from_float` slots. These
// are live on every target: a quantized mul_mat with an F32 right-hand operand
// stages it through one of them, so a wrong byte here is a wrong answer
// everywhere. Captured as a checksum of the output bytes.
struct row_entry {
    const char *   name;
    enum ggml_type t;
    row_fn         generic;
    row_fn         arm;
};

struct entry {
    const char *      name;   // the Zig field name
    enum ggml_type    xt;     // left operand type
    enum ggml_type    yt;     // right operand type, the C's vec_dot_type
    dot_fn            generic; // ggml-cpu/quants.c
    dot_fn            arm;     // ggml-cpu/arch/arm/quants.c
};

// The pairings come from `type_traits_cpu` in ggml-cpu.c. Spelled out rather
// than read from `ggml_get_type_traits_cpu` so this harness does not depend on
// the file being ported.
static const struct entry entries[] = {
    { "q1_0",     GGML_TYPE_Q1_0,     GGML_TYPE_Q8_0,  ggml_vec_dot_q1_0_q8_0_generic, ggml_vec_dot_q1_0_q8_0 },
    { "q2_0",     GGML_TYPE_Q2_0,     GGML_TYPE_Q8_0,  ggml_vec_dot_q2_0_q8_0_generic, ggml_vec_dot_q2_0_q8_0 },
    { "q4_0",     GGML_TYPE_Q4_0,     GGML_TYPE_Q8_0,  ggml_vec_dot_q4_0_q8_0_generic, ggml_vec_dot_q4_0_q8_0 },
    { "q4_1",     GGML_TYPE_Q4_1,     GGML_TYPE_Q8_1,  ggml_vec_dot_q4_1_q8_1_generic, ggml_vec_dot_q4_1_q8_1 },
    { "q5_0",     GGML_TYPE_Q5_0,     GGML_TYPE_Q8_0,  ggml_vec_dot_q5_0_q8_0_generic, ggml_vec_dot_q5_0_q8_0 },
    { "q5_1",     GGML_TYPE_Q5_1,     GGML_TYPE_Q8_1,  ggml_vec_dot_q5_1_q8_1_generic, ggml_vec_dot_q5_1_q8_1 },
    { "q8_0",     GGML_TYPE_Q8_0,     GGML_TYPE_Q8_0,  ggml_vec_dot_q8_0_q8_0_generic, ggml_vec_dot_q8_0_q8_0 },
    { "mxfp4",    GGML_TYPE_MXFP4,    GGML_TYPE_Q8_0,  ggml_vec_dot_mxfp4_q8_0_generic, ggml_vec_dot_mxfp4_q8_0 },
    { "nvfp4",    GGML_TYPE_NVFP4,    GGML_TYPE_Q8_0,  ggml_vec_dot_nvfp4_q8_0_generic, ggml_vec_dot_nvfp4_q8_0 },
    { "q2_K",     GGML_TYPE_Q2_K,     GGML_TYPE_Q8_K,  ggml_vec_dot_q2_K_q8_K_generic, ggml_vec_dot_q2_K_q8_K },
    { "q3_K",     GGML_TYPE_Q3_K,     GGML_TYPE_Q8_K,  ggml_vec_dot_q3_K_q8_K_generic, ggml_vec_dot_q3_K_q8_K },
    { "q4_K",     GGML_TYPE_Q4_K,     GGML_TYPE_Q8_K,  ggml_vec_dot_q4_K_q8_K_generic, ggml_vec_dot_q4_K_q8_K },
    { "q5_K",     GGML_TYPE_Q5_K,     GGML_TYPE_Q8_K,  ggml_vec_dot_q5_K_q8_K_generic, ggml_vec_dot_q5_K_q8_K },
    { "q6_K",     GGML_TYPE_Q6_K,     GGML_TYPE_Q8_K,  ggml_vec_dot_q6_K_q8_K_generic, ggml_vec_dot_q6_K_q8_K },
    { "tq1_0",    GGML_TYPE_TQ1_0,    GGML_TYPE_Q8_K,  ggml_vec_dot_tq1_0_q8_K_generic, ggml_vec_dot_tq1_0_q8_K },
    { "tq2_0",    GGML_TYPE_TQ2_0,    GGML_TYPE_Q8_K,  ggml_vec_dot_tq2_0_q8_K_generic, ggml_vec_dot_tq2_0_q8_K },
    { "iq2_xxs",  GGML_TYPE_IQ2_XXS,  GGML_TYPE_Q8_K,  ggml_vec_dot_iq2_xxs_q8_K_generic, ggml_vec_dot_iq2_xxs_q8_K },
    { "iq2_xs",   GGML_TYPE_IQ2_XS,   GGML_TYPE_Q8_K,  ggml_vec_dot_iq2_xs_q8_K_generic, ggml_vec_dot_iq2_xs_q8_K },
    { "iq2_s",    GGML_TYPE_IQ2_S,    GGML_TYPE_Q8_K,  ggml_vec_dot_iq2_s_q8_K_generic, ggml_vec_dot_iq2_s_q8_K },
    { "iq3_xxs",  GGML_TYPE_IQ3_XXS,  GGML_TYPE_Q8_K,  ggml_vec_dot_iq3_xxs_q8_K_generic, ggml_vec_dot_iq3_xxs_q8_K },
    { "iq3_s",    GGML_TYPE_IQ3_S,    GGML_TYPE_Q8_K,  ggml_vec_dot_iq3_s_q8_K_generic, ggml_vec_dot_iq3_s_q8_K },
    { "iq1_s",    GGML_TYPE_IQ1_S,    GGML_TYPE_Q8_K,  ggml_vec_dot_iq1_s_q8_K_generic, ggml_vec_dot_iq1_s_q8_K },
    { "iq1_m",    GGML_TYPE_IQ1_M,    GGML_TYPE_Q8_K,  ggml_vec_dot_iq1_m_q8_K_generic, ggml_vec_dot_iq1_m_q8_K },
    { "iq4_nl",   GGML_TYPE_IQ4_NL,   GGML_TYPE_Q8_0,  ggml_vec_dot_iq4_nl_q8_0_generic, ggml_vec_dot_iq4_nl_q8_0 },
    { "iq4_xs",   GGML_TYPE_IQ4_XS,   GGML_TYPE_Q8_K,  ggml_vec_dot_iq4_xs_q8_K_generic, ggml_vec_dot_iq4_xs_q8_K },
};

#define NENTRIES (sizeof(entries) / sizeof(entries[0]))

static const struct row_entry row_entries[] = {
    { "q8_0", GGML_TYPE_Q8_0, quantize_row_q8_0_generic, quantize_row_q8_0 },
    { "q8_1", GGML_TYPE_Q8_1, quantize_row_q8_1_generic, quantize_row_q8_1 },
    { "q8_K", GGML_TYPE_Q8_K, quantize_row_q8_K_generic, quantize_row_q8_K },
};

#define NROWENTRIES (sizeof(row_entries) / sizeof(row_entries[0]))

// The input patterns.
//
// Chosen so that only `zeros` produces a zero result. An earlier set had
// `alternating` cancelling to exactly zero for 22 of the 25 kernels and `tiny`
// underflowing the fp16 delta to zero for all 25 -- 72 of 125 values were
// 0x00000000, which a kernel that did nothing but `*s = 0` would have passed.
enum pattern { P_RANDOM, P_ZEROS, P_SIGNS, P_OPPOSED, P_LOPSIDED, P_TIES, NPATTERNS };

static const char * pattern_names[NPATTERNS] = {
    "random", "zeros", "signs", "opposed", "lopsided", "ties",
};

static uint32_t lcg_state = 1;

static float lcg_next(void) {
    lcg_state = 1103515245u * lcg_state + 12345u;
    return ((float) (lcg_state >> 16) / 32768.0f) - 1.0f;
}

// Fills `x` and `y` for one pattern. Both operands are generated, because a
// dot product has two and an asymmetric bug (reading x's scale for y's) is
// invisible when they are the same data.
static void fill(enum pattern p, float * x, float * y, int n) {
    lcg_state = 1;
    for (int i = 0; i < n; i++) {
        switch (p) {
            case P_RANDOM:
                x[i] = lcg_next();
                y[i] = lcg_next();
                break;
            case P_ZEROS:
                // amax == 0 on both sides: every symmetric quantizer's
                // divide-by-zero guard, and a dot product of nothing.
                x[i] = 0.0f;
                y[i] = 0.0f;
                break;
            case P_SIGNS:
                // Fixed signs with asymmetric magnitudes and coprime periods,
                // so the nibble unpacking's sign handling is exercised without
                // the sum cancelling.
                x[i] = (i % 2) ? 1.0f : -0.5f;
                y[i] = (i % 3) ? 0.75f : -1.0f;
                break;
            case P_OPPOSED:
                // Near-total cancellation: the sum passes through ~0 while
                // the terms stay large.
                x[i] = lcg_next();
                y[i] = -x[i];
                break;
            case P_LOPSIDED:
                // Four orders of magnitude apart, with x's fp16 delta down in
                // the subnormals but not underflowed. Reading one operand's
                // scale for the other's shows up here and nowhere else.
                x[i] = lcg_next() * 1e-4f;
                y[i] = lcg_next() * 4.0f;
                break;
            case P_TIES:
                // Exact halves at the quantization step, which no
                // pseudorandom input ever hits. Element 0 is 127 so that
                // `amax == 127`, hence `d == 1` and `id == 1`, hence
                // `x * id == x` exactly -- and every other element is an
                // exact `n + 0.5`.
                //
                // Without this, round-half-to-even and round-half-away are
                // indistinguishable: swapping `fcvtns` for a `round()` was
                // measured to pass every other pattern.
                if (i == 0) {
                    x[i] = 127.0f;
                    y[i] = 127.0f;
                } else {
                    const float half = (float) ((i % 9) - 4) + 0.5f;
                    x[i] = half;
                    y[i] = -half;
                }
                break;
            default:
                x[i] = 0.0f;
                y[i] = 0.0f;
                break;
        }
    }
}

// Quantizes the left operand.
//
// Everything goes through `ggml_quantize_chunk` because it calls
// `ggml_quantize_init`, and the i-quants cannot be quantized at all without
// their lookup tables.
//
// The importance matrix is all-ones and passed for every type, not just the
// ones `ggml_quantize_requires_imatrix` names. What this harness needs from x
// is deterministic *bytes*; how they were arrived at does not matter, and one
// path for all 25 types is less to get wrong than two.
static void quant_x(enum ggml_type t, const float * src, void * dst, int n) {
    static float imatrix[NELEM];
    for (int i = 0; i < n; i++) {
        imatrix[i] = 1.0f;
    }
    ggml_quantize_chunk(t, src, dst, 0, 1, n, imatrix);
}

// Quantizes the right operand.
//
// None of the three go through `ggml_quantize_chunk`: it has no case for q8_1
// or q8_K at all, and using the reference quantizer for all three keeps the
// three paths identical.
static void quant_y(enum ggml_type t, const float * src, void * dst, int n) {
    switch (t) {
        case GGML_TYPE_Q8_0: quantize_row_q8_0_ref(src, dst, n); break;
        case GGML_TYPE_Q8_1: quantize_row_q8_1_ref(src, dst, n); break;
        case GGML_TYPE_Q8_K: quantize_row_q8_K_ref(src, dst, n); break;
        default:             fprintf(stderr, "unexpected vec_dot_type\n"); exit(1);
    }
}

// Which of the two translation units' kernels to capture.
enum which { W_GENERIC, W_ARM };

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "generic";
    enum which w = strcmp(mode, "arm") == 0 ? W_ARM : W_GENERIC;
    const char * unit = w == W_ARM ? "ggml-cpu/arch/arm/quants.c" : "ggml-cpu/quants.c";

    static float xf[NELEM];
    static float yf[NELEM];
    static uint8_t xq[NELEM * 4];
    static uint8_t yq[NELEM * 4];

    // Fills the fp16, e8m0 and ue4m3 lookup tables. Not optional: the NEON
    // `nvfp4` kernel reads `ggml_table_f32_ue4m3` through
    // `GGML_CPU_UE4M3_TO_FP32`, and without this every one of its scales is
    // zero -- which was captured as six all-zero goldens before this call was
    // added, values a kernel that did nothing would have passed.
    ggml_cpu_init();

    printf("//! Golden values for the ported `%s` dot products.\n", unit);
    printf("//!\n");
    printf("//! # Provenance\n");
    printf("//!\n");
    printf("//! **Not a port, and not hand-written.** Generated by\n");
    printf("//! `harness/vecdot_golden.c` run against the C reference build at\n");
    printf("//! v0.3.0 (`c1d0e7a00`). Regenerate with `scripts/vecdot-golden`.\n");
    printf("//!\n");
    printf("//! # Why these exist\n");
    printf("//!\n");
    if (w == W_ARM) {
        printf("//! These kernels are live: `type_traits_cpu` points at them and\n");
        printf("//! every quantized `mul_mat` on the CPU goes through one. So\n");
        printf("//! `test-backend-ops` and token parity do reach them -- but both\n");
        printf("//! compare with a tolerance or at token granularity, and a\n");
        printf("//! last-bit difference in a dot product survives either. These\n");
        printf("//! values do not.\n");
    } else {
        printf("//! Nothing on this target calls `ggml_vec_dot_*_generic`:\n");
        printf("//! `arch-fallback.h` renames nothing from `quants.c` on ARM, and\n");
        printf("//! `arch/arm/quants.c` supplies all 28 real entry points. So\n");
        printf("//! `test-backend-ops` cannot reach these and neither can token\n");
        printf("//! parity. These values, captured from the C, are the only gate\n");
        printf("//! the generic kernels have.\n");
    }
    printf("//!\n");
    printf("//! Each value is the **raw bits** of the `f32` the C writes, so the\n");
    printf("//! comparison is exact. A dot product right to six decimals and\n");
    printf("//! wrong in the last bit is a porting bug.\n");
    printf("//!\n");
    printf("//! # The five input patterns\n");
    printf("//!\n");
    printf("//! - `random`      the common path\n");
    printf("//! - `zeros`       both operands all-zero: the divide-by-zero guards\n");
    printf("//! - `signs`       fixed signs, asymmetric magnitudes, coprime\n");
    printf("//!                 periods -- sign handling without cancellation\n");
    printf("//! - `opposed`     y = -x, so the sum cancels to near zero while the\n");
    printf("//!                 terms stay large\n");
    printf("//! - `lopsided`    operands four orders of magnitude apart, x's fp16\n");
    printf("//!                 delta subnormal -- catches a swapped scale\n");
    printf("//!\n");
    printf("//! Only `zeros` should produce a zero result. If another pattern\n");
    printf("//! does, it is testing less than it looks like it is.\n");
    printf("\n");
    printf("/// One kernel's result for each input pattern, as raw `f32` bits.\n");
    printf("pub const Dot = struct {\n");
    for (int p = 0; p < NPATTERNS; p++) {
        printf("    %s: u32,\n", pattern_names[p]);
    }
    printf("};\n");
    printf("\n");
    printf("/// Elements per dot product. Divisible by every block size in play.\n");
    printf("pub const nelem = %d;\n", NELEM);
    printf("\n");

    printf("/// One row quantizer's output for each input pattern, as an FNV-1a\n");
    printf("/// checksum of the bytes it writes.\n");
    printf("pub const Row = struct {\n");
    for (int p = 0; p < NPATTERNS; p++) {
        printf("    %s: u64,\n", pattern_names[p]);
    }
    printf("};\n\n");

    for (size_t e = 0; e < NROWENTRIES; e++) {
        const struct row_entry * re = &row_entries[e];
        printf("pub const row_%s = Row{\n", re->name);
        for (int p = 0; p < NPATTERNS; p++) {
            fill((enum pattern) p, xf, yf, NELEM);
            memset(xq, 0, sizeof(xq));
            (w == W_ARM ? re->arm : re->generic)(xf, xq, NELEM);
            const size_t bytes = ggml_row_size(re->t, NELEM);
            printf("    .%s = 0x%016llX,\n", pattern_names[p],
                   (unsigned long long) fnv(xq, bytes));
        }
        printf("};\n\n");
    }

    for (size_t e = 0; e < NENTRIES; e++) {
        const struct entry * en = &entries[e];

            printf("pub const %s%s = Dot{\n", en->name, "");

        for (int p = 0; p < NPATTERNS; p++) {
            fill((enum pattern) p, xf, yf, NELEM);

            memset(xq, 0, sizeof(xq));
            memset(yq, 0, sizeof(yq));

            quant_x(en->xt, xf, xq, NELEM);
            quant_y(en->yt, yf, yq, NELEM);

            float s = 0.0f;
            (w == W_ARM ? en->arm : en->generic)(NELEM, &s, 0, xq, 0, yq, 0, 1);

            uint32_t bits;
            memcpy(&bits, &s, sizeof(bits));
            printf("    .%s = 0x%08X, // % .9g\n", pattern_names[p], bits, (double) s);
        }

        printf("};\n\n");
    }

    return 0;
}
