// Diff every x86 quant kernel against the reference C, on bits: the 27
// exports of `arch/x86/quants.c`, and `ggml_vec_dot_q2_0_q8_0`, which on x86
// is `quants.c`'s generic body under the real name (`arch-fallback.h:86`).
//
// Not a port. This is a gate, driven by `scripts/x86-quants-diff`.
//
// The x86 kernels need a per-kernel oracle for the same reason the repack
// kernels did: `ops-diff` reaches each through one `mul_mat` shape per type,
// at ~2 minutes a pass under QEMU, and `node-diff` stops at the first node.
// Here the reference translation units are compiled with every unmangled
// export renamed to `ref_*` (the `-D` rename `scripts/repack-diff` uses), so
// both implementations link into one process and are called back to back on
// identical bytes.
//
// The inputs are random bytes. These kernels are integer dot products with a
// float epilogue and every bit pattern of a quantized block is a valid one, so
// only the scale fields are fixed up -- each `ggml_half` to a finite value,
// E8M0 and UE4M3 bytes away from their NaN encodings -- to keep the epilogue
// off NaN and Inf, where comparing bits would stop meaning anything. The
// `q8_K` operand's `bsums` are the real sums of its `qs`, as the quantizer
// writes them.
//
// Every line it prints stays under 80 columns: on the Windows laptop its
// output comes back through ConPTY, which wraps at the console's width.
//
// Usage: x86_quants_diff [trials [kernel substring]]   (default 300 trials)

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*vec_dot_t)(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
typedef void (*quantize_t)(const float * x, void * y, int64_t k);

#define DECL_DOT(name) \
    void name(int, float *, size_t, const void *, size_t, const void *, size_t, int); \
    void ref_##name(int, float *, size_t, const void *, size_t, const void *, size_t, int);
#define DECL_QUANT(name) \
    void name(const float *, void *, int64_t); \
    void ref_##name(const float *, void *, int64_t);

DECL_DOT(ggml_vec_dot_q1_0_q8_0)
DECL_DOT(ggml_vec_dot_q2_0_q8_0)
DECL_DOT(ggml_vec_dot_q4_0_q8_0)
DECL_DOT(ggml_vec_dot_q4_1_q8_1)
DECL_DOT(ggml_vec_dot_q5_0_q8_0)
DECL_DOT(ggml_vec_dot_q5_1_q8_1)
DECL_DOT(ggml_vec_dot_q8_0_q8_0)
DECL_DOT(ggml_vec_dot_mxfp4_q8_0)
DECL_DOT(ggml_vec_dot_nvfp4_q8_0)
DECL_DOT(ggml_vec_dot_iq4_nl_q8_0)
DECL_DOT(ggml_vec_dot_q2_K_q8_K)
DECL_DOT(ggml_vec_dot_q3_K_q8_K)
DECL_DOT(ggml_vec_dot_q4_K_q8_K)
DECL_DOT(ggml_vec_dot_q5_K_q8_K)
DECL_DOT(ggml_vec_dot_q6_K_q8_K)
DECL_DOT(ggml_vec_dot_tq1_0_q8_K)
DECL_DOT(ggml_vec_dot_tq2_0_q8_K)
DECL_DOT(ggml_vec_dot_iq1_s_q8_K)
DECL_DOT(ggml_vec_dot_iq1_m_q8_K)
DECL_DOT(ggml_vec_dot_iq2_xxs_q8_K)
DECL_DOT(ggml_vec_dot_iq2_xs_q8_K)
DECL_DOT(ggml_vec_dot_iq2_s_q8_K)
DECL_DOT(ggml_vec_dot_iq3_xxs_q8_K)
DECL_DOT(ggml_vec_dot_iq3_s_q8_K)
DECL_DOT(ggml_vec_dot_iq4_xs_q8_K)
DECL_QUANT(quantize_row_q8_0)
DECL_QUANT(quantize_row_q8_1)
DECL_QUANT(quantize_row_q8_K)

// -----------------------------------------------------------------------------
// Random numbers: splitmix64 from a fixed seed, so a run reproduces exactly.

static uint64_t rng_state;

static uint64_t rng(void) {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static float rngf(void) { return (float)(rng() >> 40) / (float)(1 << 24); } // [0, 1)

static void fill_bytes(uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)rng();
}

// A finite half, sign random, magnitude 2^-14 .. 2^3: wide enough that the
// epilogue's roundings vary, narrow enough that nothing overflows.
static uint16_t rand_half(void) {
    const uint16_t exp = (uint16_t)(1 + rng() % 18); // biased 1..18
    const uint16_t man = (uint16_t)(rng() & 0x3ff);
    const uint16_t sgn = (uint16_t)((rng() & 1) << 15);
    return (uint16_t)(sgn | (exp << 10) | man);
}

static void set_half(void * blk, size_t off) {
    const uint16_t h = rand_half();
    memcpy((uint8_t *)blk + off, &h, 2);
}

// -----------------------------------------------------------------------------
// The left operand of each dot product.

enum ytype { Y_Q8_0, Y_Q8_1, Y_Q8_K };

typedef struct {
    const char * name;
    vec_dot_t port, ref;
    size_t blk;      // bytes per block
    int qk;          // weights per block
    enum ytype y;
    void (*fixup)(void * blk);
} dot_case;

static void fix_d(void * b)        { set_half(b, 0); }
static void fix_dm(void * b)       { set_half(b, 0); set_half(b, 2); }
static void fix_q2_K(void * b)     { set_half(b, offsetof(block_q2_K, dm)); set_half(b, offsetof(block_q2_K, dm) + 2); }
static void fix_q3_K(void * b)     { set_half(b, offsetof(block_q3_K, d)); }
static void fix_q4_K(void * b)     { set_half(b, offsetof(block_q4_K, dm)); set_half(b, offsetof(block_q4_K, dm) + 2); }
static void fix_q5_K(void * b)     { set_half(b, offsetof(block_q5_K, dm)); set_half(b, offsetof(block_q5_K, dm) + 2); }
static void fix_q6_K(void * b)     { set_half(b, offsetof(block_q6_K, d)); }
static void fix_tq1_0(void * b)    { set_half(b, offsetof(block_tq1_0, d)); }
static void fix_tq2_0(void * b)    { set_half(b, offsetof(block_tq2_0, d)); }
static void fix_iq1_s(void * b)    { set_half(b, offsetof(block_iq1_s, d)); }
static void fix_iq2_xxs(void * b)  { set_half(b, offsetof(block_iq2_xxs, d)); }
static void fix_iq2_xs(void * b)   { set_half(b, offsetof(block_iq2_xs, d)); }
static void fix_iq2_s(void * b)    { set_half(b, offsetof(block_iq2_s, d)); }
static void fix_iq3_xxs(void * b)  { set_half(b, offsetof(block_iq3_xxs, d)); }
static void fix_iq3_s(void * b)    { set_half(b, offsetof(block_iq3_s, d)); }
static void fix_iq4_xs(void * b)   { set_half(b, offsetof(block_iq4_xs, d)); }

// E8M0: 2^(e-127); 0xff is NaN. Kept to 2^-17 .. 2^13.
static void fix_mxfp4(void * b) { ((block_mxfp4 *)b)->e = (uint8_t)(110 + rng() % 30); }

// UE4M3: 0x7f and 0xff are NaN; the sign bit is ignored. Any other byte.
static void fix_nvfp4(void * b) {
    block_nvfp4 * x = b;
    for (int i = 0; i < QK_NVFP4 / QK_NVFP4_SUB; i++) {
        uint8_t v = (uint8_t)rng();
        if ((v & 0x7f) == 0x7f) v ^= 1;
        x->d[i] = v;
    }
}

// iq1_m keeps its half in the top nibble of each of the four `uint16_t`
// scale words.
static void fix_iq1_m(void * b) {
    block_iq1_m * x = b;
    const uint16_t h = rand_half();
    uint16_t sc[4];
    memcpy(sc, x->scales, sizeof sc);
    for (int i = 0; i < 4; i++) sc[i] = (uint16_t)((sc[i] & 0x0fff) | (((h >> (4 * i)) & 0xf) << 12));
    memcpy(x->scales, sc, sizeof sc);
}

#define DOT(type, ytype_, qk_, fix) \
    { #type, ggml_vec_dot_##type, ref_ggml_vec_dot_##type, 0, qk_, ytype_, fix }

// Block sizes are filled in main(); the macro above cannot name the struct
// from a kernel name.
static dot_case dots[] = {
    DOT(q1_0_q8_0,     Y_Q8_0, QK1_0,    fix_d),
    DOT(q2_0_q8_0,     Y_Q8_0, QK2_0,    fix_d),
    DOT(q4_0_q8_0,     Y_Q8_0, QK4_0,    fix_d),
    DOT(q4_1_q8_1,     Y_Q8_1, QK4_1,    fix_dm),
    DOT(q5_0_q8_0,     Y_Q8_0, QK5_0,    fix_d),
    DOT(q5_1_q8_1,     Y_Q8_1, QK5_1,    fix_dm),
    DOT(q8_0_q8_0,     Y_Q8_0, QK8_0,    fix_d),
    DOT(mxfp4_q8_0,    Y_Q8_0, QK_MXFP4, fix_mxfp4),
    DOT(nvfp4_q8_0,    Y_Q8_0, QK_NVFP4, fix_nvfp4),
    DOT(iq4_nl_q8_0,   Y_Q8_0, QK4_NL,   fix_d),
    DOT(q2_K_q8_K,     Y_Q8_K, QK_K,     fix_q2_K),
    DOT(q3_K_q8_K,     Y_Q8_K, QK_K,     fix_q3_K),
    DOT(q4_K_q8_K,     Y_Q8_K, QK_K,     fix_q4_K),
    DOT(q5_K_q8_K,     Y_Q8_K, QK_K,     fix_q5_K),
    DOT(q6_K_q8_K,     Y_Q8_K, QK_K,     fix_q6_K),
    DOT(tq1_0_q8_K,    Y_Q8_K, QK_K,     fix_tq1_0),
    DOT(tq2_0_q8_K,    Y_Q8_K, QK_K,     fix_tq2_0),
    DOT(iq1_s_q8_K,    Y_Q8_K, QK_K,     fix_iq1_s),
    DOT(iq1_m_q8_K,    Y_Q8_K, QK_K,     fix_iq1_m),
    DOT(iq2_xxs_q8_K,  Y_Q8_K, QK_K,     fix_iq2_xxs),
    DOT(iq2_xs_q8_K,   Y_Q8_K, QK_K,     fix_iq2_xs),
    DOT(iq2_s_q8_K,    Y_Q8_K, QK_K,     fix_iq2_s),
    DOT(iq3_xxs_q8_K,  Y_Q8_K, QK_K,     fix_iq3_xxs),
    DOT(iq3_s_q8_K,    Y_Q8_K, QK_K,     fix_iq3_s),
    DOT(iq4_xs_q8_K,   Y_Q8_K, QK_K,     fix_iq4_xs),
};

static const size_t dot_blk[] = {
    sizeof(block_q1_0), sizeof(block_q2_0), sizeof(block_q4_0), sizeof(block_q4_1), sizeof(block_q5_0),
    sizeof(block_q5_1), sizeof(block_q8_0), sizeof(block_mxfp4), sizeof(block_nvfp4),
    sizeof(block_iq4_nl), sizeof(block_q2_K), sizeof(block_q3_K), sizeof(block_q4_K),
    sizeof(block_q5_K), sizeof(block_q6_K), sizeof(block_tq1_0), sizeof(block_tq2_0),
    sizeof(block_iq1_s), sizeof(block_iq1_m), sizeof(block_iq2_xxs), sizeof(block_iq2_xs),
    sizeof(block_iq2_s), sizeof(block_iq3_xxs), sizeof(block_iq3_s), sizeof(block_iq4_xs),
};

// -----------------------------------------------------------------------------
// The right operand: a quantized activation row.
//
// `mode` 0 draws `qs` over the whole `int8_t` range, -128 included, which the
// quantizers never write; mode 1 keeps to [-127, 127], which they do; mode 2
// pins every byte to an extreme, where `maddubs`' saturation would show.

static int8_t rand_q8(int mode) {
    switch (mode) {
        case 0:  return (int8_t)rng();
        case 1:  return (int8_t)((int)(rng() % 255) - 127);
        default: return (rng() & 1) ? 127 : -127;
    }
}

static void fill_y(void * vy, enum ytype t, int n, int mode) {
    if (t == Y_Q8_0) {
        block_q8_0 * y = vy;
        for (int b = 0; b < n / QK8_0; b++) {
            y[b].d = rand_half();
            for (int i = 0; i < QK8_0; i++) y[b].qs[i] = rand_q8(mode);
        }
    } else if (t == Y_Q8_1) {
        block_q8_1 * y = vy;
        for (int b = 0; b < n / QK8_1; b++) {
            y[b].d = rand_half();
            y[b].s = rand_half();
            for (int i = 0; i < QK8_1; i++) y[b].qs[i] = rand_q8(mode);
        }
    } else {
        block_q8_K * y = vy;
        for (int b = 0; b < n / QK_K; b++) {
            y[b].d = ldexpf(rngf() + 0.5f, -(int)(rng() % 12)) * ((rng() & 1) ? 1.0f : -1.0f);
            for (int i = 0; i < QK_K; i++) y[b].qs[i] = rand_q8(mode);
            for (int g = 0; g < QK_K / 16; g++) {
                int sum = 0;
                for (int i = 0; i < 16; i++) sum += y[b].qs[16 * g + i];
                y[b].bsums[g] = (int16_t)sum;
            }
        }
    }
}

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static int run_dot(const dot_case * dc, int trials) {
    static const int nblocks[] = { 1, 2, 3, 4, 5, 8, 16, 33 };
    int failures = 0, checks = 0;
    for (size_t li = 0; li < sizeof nblocks / sizeof nblocks[0]; li++) {
        int n = dc->qk * nblocks[li];
        // q8_K rows are whole super-blocks.
        if (dc->y == Y_Q8_K && n % QK_K) continue;
        const size_t xbytes = dc->blk * (size_t)(n / dc->qk);
        const size_t ybytes = (dc->y == Y_Q8_K ? sizeof(block_q8_K) * (size_t)(n / QK_K)
                            : dc->y == Y_Q8_1 ? sizeof(block_q8_1) * (size_t)(n / QK8_1)
                            : sizeof(block_q8_0) * (size_t)(n / QK8_0));
        uint8_t * x = malloc(xbytes);
        uint8_t * y = malloc(ybytes);
        for (int t = 0; t < trials; t++) {
            fill_bytes(x, xbytes);
            for (int b = 0; b < n / dc->qk; b++) dc->fixup(x + dc->blk * (size_t)b);
            fill_y(y, dc->y, n, t % 3);
            float a = 0, r = 0;
            dc->ref(n, &r, 0, x, 0, y, 0, 1);
            dc->port(n, &a, 0, x, 0, y, 0, 1);
            checks++;
            if (bits(a) != bits(r)) {
                if (failures < 3) {
                    printf("  %s n=%d t%d m%d: ref %08x port %08x\n", dc->name, n, t, t % 3, bits(r), bits(a));
                }
                failures++;
            }
        }
        free(x);
        free(y);
    }
    printf("%-6s %-14s %d / %d\n", failures ? "FAIL" : "ok", dc->name, checks - failures, checks);
    return failures;
}

// -----------------------------------------------------------------------------
// The row quantizers.

typedef struct {
    const char * name;
    quantize_t port, ref;
    int qk;
    size_t blk;
} quant_case;

static const quant_case quants[] = {
    { "q8_0", quantize_row_q8_0, ref_quantize_row_q8_0, QK8_0, sizeof(block_q8_0) },
    { "q8_1", quantize_row_q8_1, ref_quantize_row_q8_1, QK8_1, sizeof(block_q8_1) },
    { "q8_K", quantize_row_q8_K, ref_quantize_row_q8_K, QK_K,  sizeof(block_q8_K) },
};

// Pattern 0: uniform at a random amplitude. Pattern 1: half-integers under an
// exact unit scale, so every scaled value is a rounding tie. Pattern 2: one
// large element among small ones. Pattern 3: an all-zero block among others.
static void fill_x(float * x, int k, int qk, int pattern) {
    const float amp = ldexpf(1.0f, (int)(rng() % 40) - 20);
    for (int i = 0; i < k; i++) x[i] = (2 * rngf() - 1) * amp;
    if (pattern == 1) {
        for (int i = 0; i < k; i++) x[i] = (float)((int)(rng() % 253) - 126) * 0.5f;
        for (int b = 0; b < k / qk; b++) x[b * qk + (int)(rng() % (uint64_t)qk)] = (rng() & 1) ? 127.0f : -127.0f;
    } else if (pattern == 2) {
        for (int b = 0; b < k / qk; b++) x[b * qk + (int)(rng() % (uint64_t)qk)] = amp * 1000.0f;
    } else if (pattern == 3) {
        const int b = (int)(rng() % (uint64_t)(k / qk));
        for (int i = 0; i < qk; i++) x[b * qk + i] = 0.0f;
    }
}

static int run_quant(const quant_case * qc, int trials) {
    int failures = 0, checks = 0;
    for (int nb = 1; nb <= 8; nb++) {
        const int k = qc->qk * nb;
        float * x = malloc(sizeof(float) * (size_t)k);
        uint8_t * a = malloc(qc->blk * (size_t)nb);
        uint8_t * r = malloc(qc->blk * (size_t)nb);
        for (int t = 0; t < trials; t++) {
            fill_x(x, k, qc->qk, t % 4);
            memset(a, 0xa5, qc->blk * (size_t)nb);
            memset(r, 0xa5, qc->blk * (size_t)nb);
            qc->ref(x, r, k);
            qc->port(x, a, k);
            checks++;
            if (memcmp(a, r, qc->blk * (size_t)nb) != 0) {
                if (failures < 3) {
                    size_t at = 0;
                    while (a[at] == r[at]) at++;
                    printf("  %s k=%d t%d p%d: byte %zu: ref %02x port %02x\n", qc->name, k, t, t % 4, at, r[at], a[at]);
                }
                failures++;
            }
        }
        free(x);
        free(a);
        free(r);
    }
    printf("%-6s quantize_row_%-4s %d / %d\n", failures ? "FAIL" : "ok", qc->name, checks - failures, checks);
    return failures;
}

int main(int argc, char ** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 300;
    const char * only = argc > 2 ? argv[2] : NULL;
    // The E8M0 and UE4M3 tables the fp4 kernels read are filled here.
    ggml_cpu_init();
    rng_state = 0x5eed5eed5eed5eedull;

    int failed = 0, total = 0;
    for (size_t i = 0; i < sizeof dots / sizeof dots[0]; i++) {
        dots[i].blk = dot_blk[i];
        if (only && !strstr(dots[i].name, only)) continue;
        total++;
        failed += run_dot(&dots[i], trials) != 0;
    }
    for (size_t i = 0; i < sizeof quants / sizeof quants[0]; i++) {
        if (only && !strstr(quants[i].name, only)) continue;
        total++;
        failed += run_quant(&quants[i], trials) != 0;
    }
    if (failed) {
        printf("FAIL: %d of %d x86 kernels differ from the reference\n", failed, total);
        return 1;
    }
    printf("PASS: %d x86 kernels identical to the reference\n", total);
    return 0;
}
