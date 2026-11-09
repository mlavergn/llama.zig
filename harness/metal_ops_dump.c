/* Executes Metal op encoders on fixed inputs and prints the output bits.
 *
 * Built twice by scripts/metal-ops-diff -- once against the ported library,
 * once against `llama.cpp.zmake`'s -- and the two outputs are diffed: the
 * same driver, the same `.m` files and the same embedded kernels on both
 * sides, so the host layer in `src/ggml/metal/` is the only variable.
 *
 * Why this is needed
 * ------------------
 * `make backend-ops` runs Metal against the CPU with a tolerance, so it
 * cannot tell a reduction split over the wrong thread count from a right
 * one, and its largest ARGMAX row takes two parts of the 256 v0.8.0's
 * two-stage kernel can use. `make node-diff ARGS=--gpu` is exact but reaches
 * only what one decode reaches: nothing there calls TOP_K_MASK, POOL_1D or
 * a MUL_MAT_ID with `-1` ids. `make metal-diff` compares the vtables and
 * dispatches nothing. New at v0.8.0, for the encoders that release moved.
 *
 * What a line means
 * -----------------
 * Every `dst` is filled with a sentinel before the graph runs. A fresh
 * buffer is all zeros, and that is the answer `kernel_mul_mm_id_zero` gives
 * -- without the sentinel, dropping its dispatch passes. `untouched=` counts
 * what survived, and the script requires zero of the reference; `supported=`
 * must be 1. A case the backend refuses or a kernel that wrote nothing is a
 * failed gate, not a pass.
 *
 * Quantized weights are block bytes made here, not `ggml_quantize_chunk`'s:
 * the two builds' quantizers do not agree in the last bit, and that would be
 * a difference in the quantizers. See harness/ops_dump.c.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ggml_backend_t backend;
static struct ggml_context * ctx;
static ggml_backend_buffer_t buffer;

/* The same LCG as harness/vec_golden.c, re-seeded per case. */
static uint64_t rng;

static uint32_t rnd(void) {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t) (rng >> 33);
}

static float frnd(void) {
    return (float) rnd() / (float) (1u << 30) - 1.0f;
}

static void begin(uint64_t seed) {
    struct ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), NULL, true };
    ctx = ggml_init(ip);
    rng = seed * 2654435761u + 1;
}

static void alloc_all(void) {
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { fprintf(stderr, "allocation failed\n"); exit(1); }
}

static void fill_floats(struct ggml_tensor * t, float amp) {
    const int64_t n = ggml_nelements(t);
    float * f = malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) f[i] = amp * frnd();
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f, 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t * h = malloc(n * sizeof(ggml_fp16_t));
        ggml_fp32_to_fp16_row(f, h, n);
        ggml_backend_tensor_set(t, h, 0, n * sizeof(ggml_fp16_t));
        free(h);
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_bf16_t * h = malloc(n * sizeof(ggml_bf16_t));
        ggml_fp32_to_bf16_row(f, h, n);
        ggml_backend_tensor_set(t, h, 0, n * sizeof(ggml_bf16_t));
        free(h);
    } else {
        /* q4_0 and q8_0 both open on an fp16 scale */
        const size_t bs = ggml_type_size(t->type);
        const size_t nb = ggml_nbytes(t);
        uint8_t * q = malloc(nb);
        for (size_t i = 0; i < nb; i++) q[i] = (uint8_t) rnd();
        for (size_t i = 0; i + bs <= nb; i += bs) { q[i] = 0x00; q[i + 1] = 0x20; } /* 2^-7 */
        ggml_backend_tensor_set(t, q, 0, nb);
        free(q);
    }
    free(f);
}

static void fill_i32(struct ggml_tensor * t, const int32_t * v) {
    ggml_backend_tensor_set(t, v, 0, ggml_nbytes(t));
}

/* Cells and picks sometimes out of range, as test_top_k_mask's are: the op skips them. */
static void fill_cells(struct ggml_tensor * t, int32_t range) {
    const int64_t n = ggml_nelements(t);
    int32_t * v = malloc(n * sizeof(int32_t));
    for (int64_t i = 0; i < n; i++) {
        const uint32_t r = rnd() % 16;
        v[i] = r == 0 ? range : r == 1 ? -1 : (int32_t) (rnd() % (uint32_t) range);
    }
    fill_i32(t, v);
    free(v);
}

static uint64_t fnv(const uint8_t * p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

static void finish(const char * label, struct ggml_tensor * out) {
    const int supported = ggml_backend_supports_op(backend, out);
    const size_t nbytes = ggml_nbytes(out);
    const size_t ts = ggml_type_size(out->type);

    /* The scratch after `dst`, read through the host pointer of a shared
     * buffer. Two-stage ARGMAX sized this region and nothing else says it
     * used it: with the parts written over `dst` instead, the answer only
     * goes wrong if a row's final pass loses a race. */
    const size_t alloc = ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(out->buffer), out);
    const size_t extra = alloc - nbytes;
    const int shared = strstr(ggml_backend_buffer_name(out->buffer), "Private") == NULL;
    uint8_t * scratch = shared ? (uint8_t *) out->data + nbytes : NULL;
    if (scratch) memset(scratch, 0xAB, extra);

    uint8_t * sent = malloc(nbytes);
    memset(sent, 0xAB, nbytes);
    uint8_t * got = malloc(nbytes);
    memset(got, 0xAB, nbytes);

    int status = -1;
    if (supported) {
        ggml_backend_tensor_set(out, sent, 0, nbytes);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        status = ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);

        ggml_backend_tensor_get(out, got, 0, nbytes);
    }

    const uint64_t h = fnv(got, nbytes, 0xcbf29ce484222325ull);
    const uint64_t hs = scratch ? fnv(scratch, extra, 0xcbf29ce484222325ull) : 0;

    size_t untouched = 0;
    for (size_t i = 0; i + ts <= nbytes; i += ts) untouched += memcmp(got + i, sent, ts) == 0;

    printf("%-52s supported=%d status=%d type=%-4s ne=[%lld,%lld,%lld,%lld] untouched=%zu hash=%016llx extra=%zu shash=%016llx",
           label, supported, status, ggml_type_name(out->type),
           (long long) out->ne[0], (long long) out->ne[1], (long long) out->ne[2], (long long) out->ne[3],
           untouched, (unsigned long long) h, extra, (unsigned long long) hs);
    for (size_t i = 0; i < 4 && (i + 1) * ts <= nbytes; i++) {
        uint32_t w = 0;
        memcpy(&w, got + i * ts, ts);
        printf(" %0*x", (int) ts * 2, w);
    }
    printf("\n");

    free(sent);
    free(got);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    buffer = NULL;
    ctx = NULL;
}

/* A row either side of N_ARGMAX_CHUNK, of every part count's edge, and of
 * N_ARGMAX_NPART_MAX (past 256 parts, a part is more than one chunk). */
static void argmax_cases(void) {
    const int64_t ne00s[] = { 100, 4095, 4096, 4097, 5438, 8192, 8193, 65536, 1048576, 1048577, 1228800 };
    const int64_t ne01s[] = { 1, 3, 7 };
    char label[128];

    for (size_t a = 0; a < sizeof ne00s / sizeof *ne00s; a++) {
        for (size_t b = 0; b < sizeof ne01s / sizeof *ne01s; b++) {
            const int64_t ne00 = ne00s[a], ne01 = ne01s[b];

            /* tie 0 is plain, 1 has every row's maximum twice, in two parts */
            for (int tie = 0; tie < 2; tie++) {
                begin(1000 + a * 10 + b * 2 + tie);
                struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne01);
                struct ggml_tensor * out = ggml_argmax(ctx, x);
                alloc_all();

                float * v = malloc(ne00 * ne01 * sizeof(float));
                for (int64_t i = 0; i < ne00 * ne01; i++) v[i] = frnd();
                if (tie) {
                    for (int64_t r = 0; r < ne01; r++) {
                        v[r * ne00 + (ne00 - 1 - r)] = 2.0f;
                        v[r * ne00 + ne00 / 3 + r] = 2.0f;
                    }
                }
                ggml_backend_tensor_set(x, v, 0, ggml_nbytes(x));
                free(v);

                snprintf(label, sizeof label, "ARGMAX ne00=%lld ne01=%lld tie=%d", (long long) ne00, (long long) ne01, tie);
                finish(label, out);
            }
        }
    }

    /* rows padded apart: `nb01` is not `ne00*4`, and only the kargs say so */
    for (int i = 0; i < 2; i++) {
        const int64_t ne00 = i ? 40000 : 3000, ne01 = 3, pad = 72;
        begin(1500 + i);
        struct ggml_tensor * p = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00 + pad, ne01);
        struct ggml_tensor * x = ggml_view_2d(ctx, p, ne00, ne01, p->nb[1], 0);
        struct ggml_tensor * out = ggml_argmax(ctx, x);
        alloc_all();
        fill_floats(p, 1.0f);
        snprintf(label, sizeof label, "ARGMAX padded rows ne00=%lld", (long long) ne00);
        finish(label, out);
    }
}

struct tkm {
    int64_t n_kv, n_blk, n_pick, n_top, n_extra, nb, nh, ns;
    int score, bcast;
    int64_t mh; /* mask heads: 0 is one, broadcast over `nh` */
};

/* `dst_offs` places `dst` that many bytes into its buffer: nothing but a
 * hand-placed tensor reaches the scalar fill of an even element count. */
static void top_k_mask_case(enum ggml_type type, struct tkm c, size_t dst_offs, uint64_t seed) {
    char label[128];
    begin(seed);

    const int64_t n_pick = c.n_blk ? c.n_pick : c.n_kv;

    struct ggml_tensor * m = c.bcast ? ggml_new_tensor_1d(ctx, type, c.n_kv)
                                     : ggml_new_tensor_4d(ctx, type, c.n_kv, c.nb, c.mh ? c.mh : 1, c.ns);
    struct ggml_tensor * top_k = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, c.n_top, c.nb, c.nh, c.ns);
    struct ggml_tensor * s = c.score ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_pick, c.nb, c.nh, c.ns) : NULL;
    struct ggml_tensor * blocks = c.n_blk ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_blk, n_pick) : NULL;
    struct ggml_tensor * extra = c.n_extra ? ggml_new_tensor_4d(ctx, GGML_TYPE_I32, c.n_extra, c.nb, c.nh, c.ns) : NULL;
    struct ggml_tensor * out = ggml_top_k_mask(ctx, m, top_k, s, blocks, extra);

    if (dst_offs) {
        struct ggml_tensor * ins[] = { m, top_k, s, blocks, extra };
        size_t need = 4096 + ggml_nbytes(out);
        for (int i = 0; i < 5; i++) if (ins[i]) need += ggml_nbytes(ins[i]) + 256;
        buffer = ggml_backend_alloc_buffer(backend, need);

        struct ggml_tallocr ta = ggml_tallocr_new(buffer);
        for (int i = 0; i < 5; i++) if (ins[i]) ggml_tallocr_alloc(&ta, ins[i]);

        char * at = (char *) ggml_backend_buffer_get_base(buffer) + ta.offset + dst_offs;
        ggml_backend_tensor_alloc(buffer, out, at);
    } else {
        alloc_all();
    }

    fill_floats(m, 1.0f);
    fill_cells(top_k, (int32_t) n_pick);
    if (s) {
        const int64_t n = ggml_nelements(s);
        float * v = malloc(n * sizeof(float));
        for (int64_t i = 0; i < n; i++) v[i] = rnd() % 4 == 0 ? -INFINITY : frnd();
        ggml_backend_tensor_set(s, v, 0, n * sizeof(float));
        free(v);
    }
    if (blocks) fill_cells(blocks, (int32_t) c.n_kv);
    if (extra) fill_cells(extra, (int32_t) c.n_kv);

    snprintf(label, sizeof label, "TOP_K_MASK %s kv=%lld blk=%lld top=%lld ex=%lld sc=%d %lldx%lldx%lld bc=%d offs=%zu",
             ggml_type_name(type), (long long) c.n_kv, (long long) c.n_blk, (long long) c.n_top, (long long) c.n_extra,
             c.score, (long long) c.nb, (long long) c.nh, (long long) c.ns, c.bcast, dst_offs);
    finish(label, out);
}

static void top_k_mask_cases(void) {
    const enum ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_F32 };
    uint64_t seed = 2000;

    for (int t = 0; t < 2; t++) {
        /* each of the scatter's eight specialisations, even and odd sizes */
        for (int bits = 0; bits < 8; bits++) {
            const int score = bits & 1, has_blk = (bits >> 1) & 1, has_extra = (bits >> 2) & 1;
            struct tkm even = { 256, has_blk ? 4 : 0, 64, 16, has_extra ? 3 : 0, 5, 1, 1, score, 0 };
            struct tkm odd  = { 1001, has_blk ? 3 : 0, 70, 13, has_extra ? 7 : 0, 7, 1, 1, score, 0 };
            top_k_mask_case(types[t], even, 0, seed++);
            top_k_mask_case(types[t], odd, 0, seed++);
        }

        /* the head and stream dimensions, broadcast mask rows */
        struct tkm heads   = { 512, 8, 16, 32, 7, 5, 4, 2, 1, 0 };
        struct tkm bcast   = { 512, 4, 32, 16, 2, 6, 2, 3, 1, 1 };
        /* more kept cells than one threadgroup has threads */
        struct tkm wide    = { 4096, 0, 0, 1500, 300, 3, 1, 1, 0, 0 };
        struct tkm widebl  = { 4096, 4, 1024, 1100, 3, 2, 1, 2, 1, 0 };
        /* one element short of a whole vector, and exactly one */
        struct tkm tiny    = { 3, 0, 0, 2, 0, 1, 1, 1, 0, 0 };
        struct tkm tiny4   = { 4, 0, 0, 2, 1, 1, 1, 1, 1, 0 };
        /* one mask per head, then a mask for each pair of heads */
        struct tkm mheads  = { 512, 8, 16, 32, 7, 5, 4, 2, 1, 0, 4 };
        struct tkm mpairs  = { 512, 0, 0, 32, 0, 5, 4, 2, 0, 0, 2 };
        top_k_mask_case(types[t], mheads, 0, seed++);
        top_k_mask_case(types[t], mpairs, 0, seed++);
        top_k_mask_case(types[t], heads, 0, seed++);
        top_k_mask_case(types[t], bcast, 0, seed++);
        top_k_mask_case(types[t], wide, 0, seed++);
        top_k_mask_case(types[t], widebl, 0, seed++);
        top_k_mask_case(types[t], tiny, 0, seed++);
        top_k_mask_case(types[t], tiny4, 0, seed++);

        /* a fill long enough to still be running when the scatter starts,
         * if nothing holds it back */
        struct tkm big     = { 262144, 0, 0, 64, 16, 48, 1, 1, 0, 0 };
        top_k_mask_case(types[t], big, 0, seed++);

        /* a `dst` off the vector store's alignment, by one element */
        struct tkm even = { 256, 4, 64, 16, 3, 5, 1, 1, 1, 0 };
        top_k_mask_case(types[t], even, ggml_type_size(types[t]), seed++);
        top_k_mask_case(types[t], even, 2 * ggml_type_size(types[t]), seed++);
        top_k_mask_case(types[t], even, 4 * ggml_type_size(types[t]), seed++);
    }
}

/* Since v0.8.0 a threadgroup per output and a window split over its
 * threads: AVG sums in a different order at any other thread count. */
static void pool_1d_cases(void) {
    const int k0s[] = { 1, 2, 3, 31, 32, 33, 255, 256, 257, 700 };
    const int s0s[] = { 1, 2, 5 };
    char label[128];
    uint64_t seed = 3000;

    for (int op = 0; op < 2; op++) {
        for (size_t a = 0; a < sizeof k0s / sizeof *k0s; a++) {
            for (size_t b = 0; b < sizeof s0s / sizeof *s0s; b++) {
                for (int p0 = 0; p0 < 2; p0++) {
                    const int k0 = k0s[a], s0 = s0s[b];
                    if (p0 > k0 / 2) continue;
                    begin(seed++);
                    struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k0 + 37, 3, 2);
                    struct ggml_tensor * out = ggml_pool_1d(ctx, x, op == 0 ? GGML_OP_POOL_AVG : GGML_OP_POOL_MAX, k0, s0, p0);
                    alloc_all();
                    fill_floats(x, 1.0f);
                    snprintf(label, sizeof label, "POOL_1D %s k0=%d s0=%d p0=%d", op == 0 ? "avg" : "max", k0, s0, p0);
                    finish(label, out);
                }
            }
        }
    }
}

/* ids with `-1` slots, on the map path (many tokens) and the mul_mv path
 * (one or two), with a token that has no expert at all. `n_used` 9 is the
 * map kernel v0.8.0 added. */
static void fill_ids(struct ggml_tensor * ids, int n_mats, int skip) {
    const int64_t n_used = ids->ne[0], n_tokens = ids->ne[1];
    int32_t * v = malloc(n_used * n_tokens * sizeof(int32_t));
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t u = 0; u < n_used; u++) {
            int32_t id = (int32_t) (rnd() % (uint32_t) n_mats);
            if (skip && (rnd() % 4 == 0 || t == n_tokens / 2)) id = -1;
            v[t * n_used + u] = id;
        }
    }
    fill_i32(ids, v);
    free(v);
}

static void mul_mat_id_cases(void) {
    const enum ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 };
    const int n_useds[] = { 1, 2, 4, 9 };
    const int n_tokenss[] = { 1, 2, 7, 33, 130 };
    char label[128];
    uint64_t seed = 4000;

    for (int t = 0; t < 4; t++) {
        for (int u = 0; u < 4; u++) {
            for (int n = 0; n < 5; n++) {
                for (int bcast = 0; bcast < 2; bcast++) {
                    for (int skip = 0; skip < 2; skip++) {
                        const int n_mats = 16, n_used = n_useds[u], n_tokens = n_tokenss[n];
                        begin(seed++);
                        struct ggml_tensor * w = ggml_new_tensor_3d(ctx, types[t], 128, 96, n_mats);
                        struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, bcast ? 1 : n_used, n_tokens);
                        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
                        struct ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, ids);
                        alloc_all();
                        fill_floats(w, 0.5f);
                        fill_floats(x, 1.0f);
                        fill_ids(ids, n_mats, skip);
                        snprintf(label, sizeof label, "MUL_MAT_ID %s used=%d tokens=%d bcast=%d skip=%d",
                                 ggml_type_name(types[t]), n_used, n_tokens, bcast, skip);
                        finish(label, out);
                    }
                }
            }
        }
    }

    /* a row off the 32-element block: its own pipeline, and with a key
     * that drops `bci` the aligned case's would answer */
    for (int t = 0; t < 2; t++) {
        begin(seed++);
        struct ggml_tensor * w = ggml_new_tensor_3d(ctx, types[t], 100, 96, 16);
        struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 100, 4, 64);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 64);
        struct ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, ids);
        alloc_all();
        fill_floats(w, 0.5f);
        fill_floats(x, 1.0f);
        fill_ids(ids, 16, 1);
        snprintf(label, sizeof label, "MUL_MAT_ID %s k=100", ggml_type_name(types[t]));
        finish(label, out);
    }

    /* F32 `src[1]` precision: through v0.7.0 its own specialisation and two more kernels */
    for (int skip = 0; skip < 2; skip++) {
        begin(seed++);
        struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 96, 16);
        struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 4, 64);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 64);
        struct ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, ids);
        ggml_prec_set_src(out, GGML_PREC_F32, 1);
        alloc_all();
        fill_floats(w, 0.5f);
        fill_floats(x, 1.0f);
        fill_ids(ids, 16, skip);
        snprintf(label, sizeof label, "MUL_MAT_ID prec=f32 skip=%d", skip);
        finish(label, out);
    }

    for (int skip = 0; skip < 2; skip++) {
        begin(seed++);
        struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 96, 4, 9);
        struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 96, 16);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 9);
        struct ggml_tensor * out = ggml_add_id(ctx, a, b, ids);
        alloc_all();
        fill_floats(a, 1.0f);
        fill_floats(b, 1.0f);
        fill_ids(ids, 16, skip);
        snprintf(label, sizeof label, "ADD_ID skip=%d", skip);
        finish(label, out);
    }
}

/* v0.8.0's constructor takes F16 and BF16, and `supports_op` says so. */
static void scale_cases(void) {
    const enum ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 };
    char label[128];

    for (int t = 0; t < 3; t++) {
        for (int n = 0; n < 2; n++) {
            begin(5000 + t * 2 + n);
            struct ggml_tensor * x = ggml_new_tensor_2d(ctx, types[t], n ? 1001 : 1024, 3);
            struct ggml_tensor * out = ggml_scale_bias(ctx, x, 1.5f, 0.25f);
            alloc_all();
            fill_floats(x, 1.0f);
            snprintf(label, sizeof label, "SCALE %s ne0=%lld", ggml_type_name(types[t]), (long long) x->ne[0]);
            finish(label, out);
        }
    }
}

int main(void) {
    backend = ggml_backend_metal_init();
    if (!backend) { fprintf(stderr, "no Metal backend\n"); return 1; }

    argmax_cases();
    top_k_mask_cases();
    pool_1d_cases();
    mul_mat_id_cases();
    scale_cases();

    ggml_backend_free(backend);
    return 0;
}
