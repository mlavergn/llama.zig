// Capture the reference behaviour of every ggml-quants.c entry point.
//
// Emits a Zig source file of golden values: for each type, a checksum of the
// bytes `quantize_row_*_ref` produces from a fixed input, a checksum of the
// floats `dequantize_row_*` produces back, and the same for the chunk-level
// `quantize_*` with and without an importance matrix.
//
// Why checksums captured from C rather than a round-trip test
// -----------------------------------------------------------
// A round-trip test -- quantize then dequantize and check the error is small --
// compares the port against itself and passes even when both halves are wrong
// in the same way. This project has produced four such false passes already.
// These values come from the C, so the ported code is checked against the thing
// it is meant to reproduce.
//
// The inputs are deterministic (a fixed LCG) so this is reproducible, and
// captured once at the pinned commit.
//
// Build and run: see scripts/quants-golden

#include "ggml.h"
#include "ggml-quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

// Rows are 256 elements so every block size divides them: the K-quants need
// 256, the legacy formats 32, and QK_K is 256.
#define NPR   256
#define NROWS 4
#define NELEM (NPR * NROWS)

static uint64_t fnv(const void *p, size_t n) {
    const uint8_t *b = p;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// A fixed LCG rather than rand(): the values must not depend on the platform's
// libc, or the goldens stop being reproducible.
static uint32_t rng_state = 12345;
static float next_float(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    // Roughly [-2, 2), which spans the range these kernels care about without
    // being so wide that every block saturates.
    return ((float)(rng_state >> 8) / (float)(1 << 24)) * 4.0f - 2.0f;
}

static float  src[NELEM];
static float  imatrix[NPR];
static uint8_t dst[NELEM * 8];
static float  deq_buf[NELEM];

// Input shapes worth capturing.
//
// A single pseudo-random buffer exercises the common path and nothing else. In
// particular `amax` is never zero with random input, so the `id = d ? 1/d : 0`
// guard at the top of every symmetric quantizer -- the divide-by-zero
// guard -- is never taken. An all-zero block is not exotic: a pruned row or a
// padded tail produces one, and it would have been the first thing to break in
// production while every test stayed green.
typedef enum {
    PAT_RANDOM,      // the common path
    PAT_ZEROS,       // amax == 0: takes the divide-by-zero guard
    PAT_CONSTANT,    // min == max: the asymmetric formats' degenerate range
    PAT_OUTLIER,     // one huge value among tiny ones: extreme scale ratio,
                     // and the clamp on the quantized index
    PAT_ALTERNATING, // +v, -v, +v...: sign handling, and max != amax
    PAT_COUNT
} pattern_t;

static const char *pattern_name(pattern_t p) {
    switch (p) {
        case PAT_RANDOM:      return "random";
        case PAT_ZEROS:       return "zeros";
        case PAT_CONSTANT:    return "constant";
        case PAT_OUTLIER:     return "outlier";
        case PAT_ALTERNATING: return "alternating";
        default:              return "?";
    }
}

static void fill_src(pattern_t p) {
    switch (p) {
        case PAT_ZEROS:
            for (int i = 0; i < NELEM; i++) src[i] = 0.0f;
            break;
        case PAT_CONSTANT:
            for (int i = 0; i < NELEM; i++) src[i] = 0.5f;
            break;
        case PAT_OUTLIER:
            for (int i = 0; i < NELEM; i++) src[i] = 0.001f;
            // One per row, off the block boundary so it is not always the
            // first element of a block.
            for (int r = 0; r < NROWS; r++) src[r*NPR + 7] = 100.0f;
            break;
        case PAT_ALTERNATING:
            for (int i = 0; i < NELEM; i++) src[i] = (i & 1) ? -1.25f : 1.25f;
            break;
        case PAT_RANDOM:
        default:
            rng_state = 12345;
            for (int i = 0; i < NELEM; i++) src[i] = next_float();
            break;
    }
}

// Generated from its own stream, so it is identical for every pattern. Sharing
// one RNG with `fill_src` would make the importance matrix depend on how many
// values the pattern happened to draw.
static void fill_imatrix(void) {
    rng_state = 999;
    // An importance matrix is non-negative; upstream uses activation
    // magnitudes, so positive values in a similar range.
    for (int i = 0; i < NPR; i++) imatrix[i] = (next_float() + 2.0f) * 0.5f + 0.01f;
}

static FILE *out;

typedef void (*ref_fn)(const float *, void *, int64_t);
typedef void (*deq_fn)(const void *, float *, int64_t);
typedef size_t (*chunk_fn)(const float *, void *, int64_t, int64_t, const float *);

// One type's record. Driving this by symbol name rather than through the traits
// table is deliberate: `from_float_ref` is NULL for the formats that require an
// importance matrix (IQ2_XXS, IQ2_XS, IQ1_S, IQ1_M), and Q8_1 and Q8_K are
// missing one side or the other, so a traits-driven loop silently skips six of
// the twenty-seven.
static void emit_one(const char *name, enum ggml_type t, pattern_t pat,
                     ref_fn ref, deq_fn deq, chunk_fn chunk) {
    // The i-quants index a shared codebook built lazily. `ggml_quantize_chunk`
    // does this itself, but `quantize_row_*_ref` does not, so it has to happen
    // here or the first i-quant dereferences a null table.
    ggml_quantize_init(t);
    fill_src(pat);
    fill_imatrix();

    const size_t row_size = ggml_row_size(t, NPR);

    fprintf(out, "    .{ .name = \"%s\", .pattern = \"%s\", .type = %d, .row_size = %zu",
            name, pattern_name(pat), (int)t, row_size);

    // The bytes the dequantizer is fed. Prefer the reference quantizer; the
    // imatrix-only formats have none, so their bytes come from the chunk
    // entry point instead -- which is the only way to make one for them.
    unsigned char *bytes = NULL;
    if (ref) {
        memset(dst, 0, sizeof dst);
        ref(src, dst, NELEM);
        fprintf(out, ", .ref = 0x%016llx",
                (unsigned long long)fnv(dst, row_size * NROWS));
        bytes = dst;
    } else if (chunk) {
        memset(dst, 0, sizeof dst);
        chunk(src, dst, NROWS, NPR, imatrix);
        bytes = dst;
    }

    if (deq && bytes) {
        memset(deq_buf, 0, sizeof deq_buf);
        deq(bytes, deq_buf, NELEM);
        fprintf(out, ", .deq = 0x%016llx",
                (unsigned long long)fnv(deq_buf, sizeof(float) * NELEM));
    }

    if (chunk) {
        // Without an imatrix. The formats that require one are called with it
        // anyway below; here they would abort, so they are skipped.
        if (!ggml_quantize_requires_imatrix(t)) {
            memset(dst, 0, sizeof dst);
            const size_t n0 = chunk(src, dst, NROWS, NPR, NULL);
            fprintf(out, ", .chunk = 0x%016llx, .chunk_bytes = %zu",
                    (unsigned long long)fnv(dst, n0), n0);
        }
        memset(dst, 0, sizeof dst);
        const size_t n1 = chunk(src, dst, NROWS, NPR, imatrix);
        fprintf(out, ", .chunk_imatrix = 0x%016llx",
                (unsigned long long)fnv(dst, n1));
        if (ggml_quantize_requires_imatrix(t)) {
            fprintf(out, ", .chunk_bytes = %zu", n1);
        }
    }

    fprintf(out, " },\n");
}

static void emit(const char *name, enum ggml_type t,
                 ref_fn ref, deq_fn deq, chunk_fn chunk) {
    for (pattern_t pat = 0; pat < PAT_COUNT; pat++) {
        emit_one(name, t, pat, ref, deq, chunk);
    }
}

int main(int argc, char **argv) {
    out = argc > 1 ? fopen(argv[1], "w") : stdout;
    if (!out) return 1;

    fprintf(out,
        "//! Golden values for the ported `ggml-quants.c`.\n"
        "//!\n"
        "//! # Provenance\n"
        "//!\n"
        "//! **Not a port, and not hand-written.** Generated by\n"
        "//! `harness/quants_golden.c` run against the C reference build at\n"
        "//! v0.3.0 (`c1d0e7a00`). Regenerate with `scripts/quants-golden`.\n"
        "//!\n"
        "//! # Why these exist\n"
        "//!\n"
        "//! A quantizer is easy to test wrongly: quantize, dequantize, check the\n"
        "//! error is small, and the test passes with both halves broken in the\n"
        "//! same way. These checksums come from the C, so the port is measured\n"
        "//! against what it is replacing rather than against itself.\n"
        "//!\n"
        "//! `ref` is FNV-1a over the bytes `quantize_row_*_ref` writes for a\n"
        "//! fixed input; `deq` over the floats `dequantize_row_*` gives back;\n"
        "//! `chunk` and `chunk_imatrix` over `quantize_*` without and with an\n"
        "//! importance matrix. A null field means the C has no such entry point\n"
        "//! for that type, which is itself part of the contract.\n"
        "//!\n"
        "//! # The five input patterns\n"
        "//!\n"
        "//! Each type is captured against every one of them. Random input alone\n"
        "//! leaves `amax` never zero, so the divide-by-zero guard at the top of\n"
        "//! every symmetric quantizer would go untested -- and an all-zero block\n"
        "//! is what a pruned row or a padded tail produces.\n"
        "//!\n"
        "//! - `random`      the common path\n"
        "//! - `zeros`       amax == 0, taking that guard\n"
        "//! - `constant`    min == max, the asymmetric formats degenerate range\n"
        "//! - `outlier`     one huge value among tiny ones: extreme scale ratio\n"
        "//! - `alternating` +v, -v, ...: sign handling, and max != amax\n"
        "\n"
        "/// One type's golden values.\n"
        "pub const Golden = struct {\n"
        "    name: []const u8,\n"
        "    pattern: []const u8,\n"
        "    type: c_int,\n"
        "    row_size: usize,\n"
        "    ref: ?u64 = null,\n"
        "    deq: ?u64 = null,\n"
        "    chunk: ?u64 = null,\n"
        "    chunk_imatrix: ?u64 = null,\n"
        "    chunk_bytes: usize = 0,\n"
        "};\n"
        "\n"
        "/// Input shape the values were captured with. The test must use the\n"
        "/// same, and the same generator, or nothing matches.\n"
        "pub const n_per_row: usize = %d;\n"
        "pub const n_rows: usize = %d;\n"
        "\n"
        "pub const all = [_]Golden{\n", NPR, NROWS);

#define R(n)  ((ref_fn)   quantize_row_##n##_ref)
#define D(n)  ((deq_fn)   dequantize_row_##n)
#define C(n)  ((chunk_fn) quantize_##n)

    emit("Q4_0",    GGML_TYPE_Q4_0,    R(q4_0),  D(q4_0),  C(q4_0));
    emit("Q4_1",    GGML_TYPE_Q4_1,    R(q4_1),  D(q4_1),  C(q4_1));
    emit("Q5_0",    GGML_TYPE_Q5_0,    R(q5_0),  D(q5_0),  C(q5_0));
    emit("Q5_1",    GGML_TYPE_Q5_1,    R(q5_1),  D(q5_1),  C(q5_1));
    emit("Q8_0",    GGML_TYPE_Q8_0,    R(q8_0),  D(q8_0),  C(q8_0));
    // Q8_1 has a reference quantizer but no dequantizer: it is an intermediate
    // the dot-product kernels produce, never a stored weight format.
    emit("Q8_1",    GGML_TYPE_Q8_1,    R(q8_1),  NULL,     NULL);
    emit("Q1_0",    GGML_TYPE_Q1_0,    R(q1_0),  D(q1_0),  C(q1_0));
    emit("Q2_0",    GGML_TYPE_Q2_0,    R(q2_0),  D(q2_0),  C(q2_0));
    emit("MXFP4",   GGML_TYPE_MXFP4,   R(mxfp4), D(mxfp4), C(mxfp4));
    emit("NVFP4",   GGML_TYPE_NVFP4,   R(nvfp4), D(nvfp4), C(nvfp4));

    emit("Q2_K",    GGML_TYPE_Q2_K,    R(q2_K),  D(q2_K),  C(q2_K));
    emit("Q3_K",    GGML_TYPE_Q3_K,    R(q3_K),  D(q3_K),  C(q3_K));
    emit("Q4_K",    GGML_TYPE_Q4_K,    R(q4_K),  D(q4_K),  C(q4_K));
    emit("Q5_K",    GGML_TYPE_Q5_K,    R(q5_K),  D(q5_K),  C(q5_K));
    emit("Q6_K",    GGML_TYPE_Q6_K,    R(q6_K),  D(q6_K),  C(q6_K));
    // Q8_K is the activation side of the K-quant dot products; no chunk form.
    emit("Q8_K",    GGML_TYPE_Q8_K,    R(q8_K),  D(q8_K),  NULL);

    emit("TQ1_0",   GGML_TYPE_TQ1_0,   R(tq1_0), D(tq1_0), C(tq1_0));
    emit("TQ2_0",   GGML_TYPE_TQ2_0,   R(tq2_0), D(tq2_0), C(tq2_0));

    // The four with no reference quantizer: an importance matrix is not
    // optional for them, so the only way in is the chunk entry point.
    emit("IQ2_XXS", GGML_TYPE_IQ2_XXS, NULL,        D(iq2_xxs), C(iq2_xxs));
    emit("IQ2_XS",  GGML_TYPE_IQ2_XS,  NULL,        D(iq2_xs),  C(iq2_xs));
    emit("IQ1_S",   GGML_TYPE_IQ1_S,   NULL,        D(iq1_s),   C(iq1_s));
    emit("IQ1_M",   GGML_TYPE_IQ1_M,   NULL,        D(iq1_m),   C(iq1_m));

    emit("IQ2_S",   GGML_TYPE_IQ2_S,   R(iq2_s),    D(iq2_s),   C(iq2_s));
    emit("IQ3_XXS", GGML_TYPE_IQ3_XXS, R(iq3_xxs),  D(iq3_xxs), C(iq3_xxs));
    emit("IQ3_S",   GGML_TYPE_IQ3_S,   R(iq3_s),    D(iq3_s),   C(iq3_s));
    emit("IQ4_NL",  GGML_TYPE_IQ4_NL,  R(iq4_nl),   D(iq4_nl),  C(iq4_nl));
    emit("IQ4_XS",  GGML_TYPE_IQ4_XS,  R(iq4_xs),   D(iq4_xs),  C(iq4_xs));

#undef R
#undef D
#undef C

    fprintf(out, "};\n");

    // ggml_validate_row_data is the one entry point that is not a
    // (de)quantizer. Both answers are captured: clean data must validate, and a
    // NaN scale must not -- a validator that always returns true would pass the
    // first check alone.
    fprintf(out, "\n/// `ggml_validate_row_data` on the same fixed input.\n");
    fprintf(out, "pub const validate = struct {\n");
    {
        fill_src(PAT_RANDOM);
        memset(dst, 0, sizeof dst);
        quantize_row_q4_0_ref(src, (block_q4_0 *)dst, NELEM);
        const size_t nbytes = ggml_row_size(GGML_TYPE_Q4_0, NPR) * NROWS;
        fprintf(out, "    pub const q4_0_clean = %s;\n",
                ggml_validate_row_data(GGML_TYPE_Q4_0, dst, nbytes) ? "true" : "false");
        dst[0] = 0xff; dst[1] = 0x7f; // half-precision NaN in the first scale
        fprintf(out, "    pub const q4_0_nan_scale = %s;\n",
                ggml_validate_row_data(GGML_TYPE_Q4_0, dst, nbytes) ? "true" : "false");
    }
    fprintf(out, "};\n");

    if (out != stdout) fclose(out);
    return 0;
}
