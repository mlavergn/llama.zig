/* Executes each CPU op kernel on fixed inputs and prints the output bits.
 *
 * Built twice by scripts/ops-diff -- once against the ported libraries, once
 * against the reference C -- and the two outputs are diffed. Identical output
 * means the ported kernels compute the same bits the C ones do.
 *
 * Why this is needed, and why `graph_dump.c` is not enough
 * -------------------------------------------------------
 * `scripts/graph-diff` runs with `no_alloc = true`: it checks the *structure*
 * a constructor produces -- op, shape, strides, op_params, src[] -- and never
 * computes anything. It is blind to a kernel that builds the right graph and
 * fills it with the wrong numbers.
 *
 * `make backend-ops` does execute, but it compares CPU against Metal with
 * **NMSE against a 1e-7 threshold**. Measured on unary-ops.cpp: a `softplus`
 * cutoff moved from 20 to 2 gives an NMSE around 8e-9 and passes. It catches
 * a kernel wrong across the input domain and misses one wrong in a band of
 * it. Its `--diff` mode does not help either -- `test-backend-ops` prints the
 * error value only when it exceeds the threshold, so on a clean run the log
 * carries no computed values and the diff is structural.
 *
 * And on a Metal machine the CPU kernels do not run during inference at all,
 * so `make port` and `make parity-cli` cannot see them -- measured, by putting
 * an abort in one and watching generation finish.
 *
 * So this is the only value-level oracle for ggml-cpu. It compares on **bits**.
 *
 * Determinism
 * -----------
 * One thread. The CPU backend splits rows across threads and sums per-thread
 * chunks, so a different thread count is a different summation order and a
 * different last bit. One thread also makes the comparison independent of the
 * machine it runs on.
 *
 * Inputs come from the same LCG as harness/vec_golden.c, re-seeded per node so
 * each op is independent of the ones before it -- several ops here are
 * in-place and would otherwise leave their operands modified.
 */

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static struct ggml_context * ctx;

/* Thread count, from OPS_THREADS (default 1). scripts/ops-diff runs the whole
 * suite at 1 and again at 3: at one thread every kernel's `ith`/`nth` split is
 * the trivial one, so a wrong row partition -- the kind of bug a port of a
 * threaded kernel actually has -- cannot show. Three is still deterministic:
 * the split depends on the count, not on which thread wins a chunk. */
static int n_threads = 1;

/* Quantized inputs are shared through a file rather than quantized twice.
 *
 * `ggml_quantize_chunk` is part of the library under test, and the two builds
 * do not agree: measured, Q6_K bytes differ between the port and the stock C
 * while Q4_K and Q8_0 match. That is a known and accepted consequence --
 * CLAUDE.md, "Float contraction": the quantizers deliberately leave their FMA
 * sites unnamed because there is no oracle for them, and the deliverable reads
 * model files and never writes one.
 *
 * But it means that quantizing on both sides feeds the two mul_mat kernels
 * *different weights*, and the comparison stops being about the kernel. So the
 * reference run writes its quantized blocks here and the ported run reads them
 * back, and both kernels see the same bytes. */
static FILE * qfile    = NULL;
static int    qwriting = 0;

/* The LCG of harness/vec_golden.c. Values in [-1, 1), scaled by `fill_amp`.
 *
 * The amplitude is not decoration. Several kernels branch on a threshold --
 * `softplus` switches to the identity above 20, `exp` saturates, `clamp` has
 * its bounds -- and at amplitude 1 no input ever reaches one. Measured: with
 * [-1, 1) inputs only, moving the `softplus` cutoff from 20 to 2 **escapes
 * this gate**, because both arms then take the same branch for every value.
 * The activation ops are therefore run twice, at 1 and at 30. */
static float fill_amp = 1.0f;

static uint32_t lcg_state = 1;
static float lcg_next(void) {
    lcg_state = 1103515245u * lcg_state + 12345u;
    return (((float) (lcg_state >> 16) / 32768.0f) - 1.0f) * fill_amp;
}

/* Index tensors need values a kernel can actually use: an out-of-range row
 * index is a read out of bounds, not a wrong answer. `bound` is the number of
 * rows the consuming op will index into; 0 means "any non-negative int is
 * fine", which is the case for RoPE positions. */
static void fill_leaf(struct ggml_tensor * t, int64_t bound) {
    const int64_t n = ggml_nelements(t);

    switch (t->type) {
        case GGML_TYPE_F32: {
            float * d = (float *) t->data;
            for (int64_t i = 0; i < n; i++) d[i] = lcg_next();
            break;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t * d = (ggml_fp16_t *) t->data;
            for (int64_t i = 0; i < n; i++) d[i] = ggml_fp32_to_fp16(lcg_next());
            break;
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t * d = (ggml_bf16_t *) t->data;
            for (int64_t i = 0; i < n; i++) d[i] = ggml_fp32_to_bf16(lcg_next());
            break;
        }
        case GGML_TYPE_I32: {
            int32_t * d = (int32_t *) t->data;
            for (int64_t i = 0; i < n; i++)
                d[i] = bound > 0 ? (int32_t) (((uint64_t) i * 7919u) % (uint64_t) bound)
                                 : (int32_t) i;
            break;
        }
        case GGML_TYPE_I64: {
            int64_t * d = (int64_t *) t->data;
            for (int64_t i = 0; i < n; i++)
                d[i] = bound > 0 ? (int64_t) (((uint64_t) i * 7919u) % (uint64_t) bound)
                                 : i;
            break;
        }
        default: {
            /* Quantized: quantize real floats rather than scribbling bytes.
             * Random bytes can land a NaN or an absurd scale in a block
             * header, which makes the comparison about denormal handling
             * instead of about the kernel. */
            const int64_t nrow = ggml_nrows(t);
            const int64_t ne0  = t->ne[0];
            if (qfile && !qwriting) {
                /* Replay the reference's bytes; see the note on `qfile`. */
                if (fread(t->data, 1, ggml_nbytes(t), qfile) != ggml_nbytes(t)) {
                    fprintf(stderr, "short read of quantized inputs\n");
                    exit(2);
                }
                /* Keep the LCG in step with the writing run, which drew
                 * ne0 values per row before quantizing them. */
                for (int64_t r = 0; r < nrow; r++)
                    for (int64_t i = 0; i < ne0; i++) (void) lcg_next();
                break;
            }
            float * tmp = (float *) malloc((size_t) ne0 * sizeof(float));
            /* IQ2_XXS, IQ2_XS and IQ1_S refuse to quantize without an
             * importance matrix; uniform weights are the neutral one. */
            float * imat = NULL;
            if (ggml_quantize_requires_imatrix(t->type)) {
                imat = (float *) malloc((size_t) ne0 * sizeof(float));
                for (int64_t i = 0; i < ne0; i++) imat[i] = 1.0f;
            }
            for (int64_t r = 0; r < nrow; r++) {
                for (int64_t i = 0; i < ne0; i++) tmp[i] = lcg_next();
                ggml_quantize_chunk(t->type, tmp, (char *) t->data + r * t->nb[1],
                                    0, 1, ne0, imat);
            }
            free(imat);
            free(tmp);
            if (qfile && qwriting) fwrite(t->data, 1, ggml_nbytes(t), qfile);
            break;
        }
    }
}

/* There is no public accessor for a graph's leafs, so the inputs are found by
 * walking `src[]` and `view_src` from the output node. A tensor that owns its
 * data and has no op is an input; everything else a kernel will write.
 *
 * The traversal order fixes which LCG draw each input gets, so it has to be
 * the same on both sides -- which it is, being the same code. */
static const struct ggml_tensor * seen[4096];
static int seen_n = 0;

static int already_seen(const struct ggml_tensor * t) {
    for (int i = 0; i < seen_n; i++) if (seen[i] == t) return 1;
    if (seen_n < (int) (sizeof seen / sizeof *seen)) seen[seen_n++] = t;
    return 0;
}

static void fill_inputs(struct ggml_tensor * t, int64_t bound) {
    if (t == NULL || already_seen(t)) return;
    if (t->view_src) fill_inputs(t->view_src, bound);
    for (int i = 0; i < GGML_MAX_SRC; i++) fill_inputs(t->src[i], bound);
    if (t->op == GGML_OP_NONE && t->view_src == NULL) fill_leaf(t, bound);
}

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* Builds a one-node graph, fills every leaf, computes on the CPU with one
 * thread, and prints the output bits. */
static void run_amp(const char * label, struct ggml_tensor * node, int64_t bound, float amp) {
    fill_amp = amp;
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, node);

    lcg_state = 1;
    seen_n = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        fill_inputs(ggml_graph_node(gf, i), bound);
    }

    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        printf("%-34s amp=%-4g STATUS=%d\n", label, (double) amp, (int) st);
        return;
    }

    const size_t nb = ggml_nbytes(node);
    printf("%-34s amp=%-4g type=%-8s ne=[%lld,%lld,%lld,%lld] bytes=%zu hash=%016llx",
           label, (double) amp, ggml_type_name(node->type),
           (long long) node->ne[0], (long long) node->ne[1],
           (long long) node->ne[2], (long long) node->ne[3],
           nb, (unsigned long long) fnv1a(node->data, nb));

    /* A hash says "differs" and nothing else. The first few elements say
     * *how*, which is the difference between a diff you can act on and one
     * you have to bisect. */
    if (node->type == GGML_TYPE_F32) {
        const float * d = (const float *) node->data;
        const int64_t n = ggml_nelements(node);
        printf(" head=");
        for (int64_t i = 0; i < 4 && i < n; i++) {
            uint32_t u; memcpy(&u, &d[i], 4);
            printf("%08x,", u);
        }
    }
    printf("\n");

    /* OPS_FULL=<substring>: also print every element of any matching op, so
     * a failing hash can be localised to the elements that moved. */
    const char * full = getenv("OPS_FULL");
    if (full && strstr(label, full) && node->type == GGML_TYPE_F32) {
        const float * d = (const float *) node->data;
        for (int64_t i = 0; i < ggml_nelements(node); i++) {
            uint32_t u; memcpy(&u, &d[i], 4);
            printf("  [%lld] %08x %.9g\n", (long long) i, u, (double) d[i]);
        }
    }
}

#define R(expr)      run_amp(#expr, (expr), 0, 1.0f)
#define RB(expr, b)  run_amp(#expr, (expr), (b), 1.0f)
/* Wide inputs, for the kernels that branch on a threshold. */
#define RW(expr)     run_amp(#expr, (expr), 0, 30.0f)

/* argv: [write-q|read-q <path>] -- see the note on `qfile`. */
/* The map_custom and custom ops call back into the caller. These split rows
 * by `ith`/`nth` the way a real kernel would, so the dispatch's thread
 * arguments are part of what is compared. */
static void custom1_fn(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * ud) {
    (void) ud;
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = ith; i < n; i += nth) ((float *) dst->data)[i] = 2.0f * ((const float *) a->data)[i] + (float) ith;
}
static void custom2_fn(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * ud) {
    (void) ud;
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = ith; i < n; i += nth) ((float *) dst->data)[i] = ((const float *) a->data)[i] * ((const float *) b->data)[i];
}
static void custom3_fn(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * ud) {
    (void) ud;
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = ith; i < n; i += nth)
        ((float *) dst->data)[i] = ((const float *) a->data)[i] - ((const float *) b->data)[i] * ((const float *) c->data)[i];
}
static void custom_fn(struct ggml_tensor * dst, int ith, int nth, void * ud) {
    const int64_t n = ggml_nelements(dst);
    const struct ggml_tensor * a = dst->src[0];
    for (int64_t i = ith; i < n; i += nth) ((float *) dst->data)[i] = ((const float *) a->data)[i] + *(const float *) ud;
}

int main(int argc, char ** argv) {
    {
        const char * env = getenv("OPS_THREADS");
        if (env && atoi(env) > 0) n_threads = atoi(env);
    }
    if (argc == 3) {
        qwriting = strcmp(argv[1], "write-q") == 0;
        qfile = fopen(argv[2], qwriting ? "wb" : "rb");
        if (!qfile) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    }

    struct ggml_init_params p = { 1024u * 1024u * 1024u, NULL, false /*alloc*/ };
    ctx = ggml_init(p);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    /* The f16 and quant lookup tables have to be live before any kernel runs. */
    ggml_cpu_init();

    struct ggml_tensor * a2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    struct ggml_tensor * b2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    struct ggml_tensor * row  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    struct ggml_tensor * a3   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 2);
    struct ggml_tensor * a4   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 8, 2);
    struct ggml_tensor * sq   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    struct ggml_tensor * vec  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    struct ggml_tensor * scal = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    puts("=== elementwise ===");
    R(ggml_add(ctx, a2, b2));
    R(ggml_add(ctx, a2, row));          /* broadcast */
    R(ggml_add1(ctx, a2, scal));
    R(ggml_sub(ctx, a2, b2));
    R(ggml_mul(ctx, a2, b2));
    R(ggml_div(ctx, a2, b2));
    R(ggml_sqr(ctx, a2));
    R(ggml_sqrt(ctx, ggml_abs(ctx, a2)));
    R(ggml_log(ctx, ggml_abs(ctx, a2)));
    R(ggml_sin(ctx, a2));
    R(ggml_cos(ctx, a2));
    R(ggml_abs(ctx, a2));
    R(ggml_neg(ctx, a2));
    R(ggml_step(ctx, a2));
    R(ggml_scale(ctx, a2, 2.5f));
    R(ggml_scale_bias(ctx, a2, 2.5f, -1.25f));
    R(ggml_clamp(ctx, a2, -0.5f, 0.5f));

    puts("=== activations ===");
    R(ggml_relu(ctx, a2));
    R(ggml_gelu(ctx, a2));
    R(ggml_gelu_quick(ctx, a2));
    R(ggml_gelu_erf(ctx, a2));
    R(ggml_silu(ctx, a2));
    R(ggml_tanh(ctx, a2));
    R(ggml_sigmoid(ctx, a2));
    R(ggml_hardswish(ctx, a2));
    R(ggml_hardsigmoid(ctx, a2));
    R(ggml_elu(ctx, a2));
    R(ggml_exp(ctx, a2));
    R(ggml_softplus(ctx, a2));
    R(ggml_expm1(ctx, a2));
    R(ggml_xielu(ctx, a2, 0.5f, 1.5f, 0.25f, 1e-6f));
    R(ggml_leaky_relu(ctx, a2, 0.01f, false));
    R(ggml_silu_back(ctx, a2, b2));

    puts("=== activations, wide inputs ===");
    /* At amplitude 1 no input reaches a branch threshold, and a fault in one
     * of those branches escapes -- measured, with `softplus`. */
    RW(ggml_relu(ctx, a2));
    RW(ggml_gelu(ctx, a2));
    RW(ggml_gelu_quick(ctx, a2));
    RW(ggml_gelu_erf(ctx, a2));
    RW(ggml_silu(ctx, a2));
    RW(ggml_tanh(ctx, a2));
    RW(ggml_sigmoid(ctx, a2));
    RW(ggml_hardswish(ctx, a2));
    RW(ggml_hardsigmoid(ctx, a2));
    RW(ggml_elu(ctx, a2));
    RW(ggml_exp(ctx, a2));
    RW(ggml_softplus(ctx, a2));
    RW(ggml_expm1(ctx, a2));
    RW(ggml_xielu(ctx, a2, 0.5f, 1.5f, 0.25f, 1e-6f));
    RW(ggml_leaky_relu(ctx, a2, 0.01f, false));
    RW(ggml_clamp(ctx, a2, -0.5f, 0.5f));
    RW(ggml_step(ctx, a2));
    RW(ggml_abs(ctx, a2));
    RW(ggml_sqr(ctx, a2));
    RW(ggml_sin(ctx, a2));
    RW(ggml_cos(ctx, a2));
    RW(ggml_soft_max(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8)));

    puts("=== gated linear units ===");
    R(ggml_swiglu(ctx, a2));
    R(ggml_swiglu_swapped(ctx, a2));
    R(ggml_swiglu_split(ctx, a2, b2));
    R(ggml_geglu(ctx, a2));
    R(ggml_reglu(ctx, a2));
    R(ggml_geglu_erf(ctx, a2));
    R(ggml_geglu_quick(ctx, a2));
    R(ggml_swiglu_oai(ctx, a2, b2, 1.702f, 7.0f));

    puts("=== reductions ===");
    R(ggml_sum(ctx, a3));
    R(ggml_sum_rows(ctx, a3));
    R(ggml_mean(ctx, a3));
    R(ggml_argmax(ctx, a2));
    {   /* I32 only -- ops.cpp:1684 aborts on anything else */
        struct ggml_tensor * i2a = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 4);
        struct ggml_tensor * i2b = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 4);
        RB(ggml_count_equal(ctx, i2a, i2b), 3);
    }
    R(ggml_cumsum(ctx, a2));

    puts("=== normalisation ===");
    R(ggml_norm(ctx, a2, 1e-5f));
    R(ggml_rms_norm(ctx, a2, 1e-5f));
    R(ggml_rms_norm_back(ctx, a2, b2, 1e-5f));
    R(ggml_group_norm(ctx, a4, 4, 1e-6f));
    R(ggml_l2_norm(ctx, a2, 1e-5f));

    puts("=== matmul ===");
    R(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 16), a2));
    R(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 16), a2));
    R(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 16),
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4)));
    R(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 256, 16),
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4)));
    R(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 256, 16),
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4)));
    R(ggml_out_prod(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8),
                         ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 8)));

    puts("=== matmul, every quantized type ===");
    {
        /* n >= 8, which is the only shape that reaches llamafile_sgemm's F16
         * kernel: sgemm.cpp:3975 returns false below 8, and the sweep below
         * tops out at 7. Measured before porting sgemm.cpp -- the F32 and
         * Q0 paths were already covered at n = 4 and n = 7, the F16 one was
         * not covered at all. F32 at n = 8 comes along for the width. */
        run_amp("mul_mat f16 x f32 [512 x 16] x 8",
                ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 512, 16),
                                  ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 8)), 0, 1.0f);
        run_amp("mul_mat f32 x f32 [512 x 16] x 8",
                ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 16),
                                  ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 8)), 0, 1.0f);
    }

    {
        /* Every type with a CPU vec_dot, at 1 column (the decode path) and 7
         * (a short prompt). Only Q4_K, Q6_K and Q8_0 were here at first, and
         * a Q5_K divergence surfaced end to end instead -- measured, as the
         * first differing node of a CPU-only Qwen3.5 decode. */
        static const enum ggml_type qtypes[] = {
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
            GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
            GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_NVFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0,
            GGML_TYPE_BF16, GGML_TYPE_F16,
        };
        for (size_t qi = 0; qi < sizeof qtypes / sizeof qtypes[0]; qi++) {
            for (int nc = 1; nc <= 7; nc += 6) {
                char label[96];
                snprintf(label, sizeof label, "mul_mat %s x f32 [512 x 16] x %d", ggml_type_name(qtypes[qi]), nc);
                run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, qtypes[qi], 512, 16),
                                                 ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, nc)), 0, 1.0f);
            }
        }
    }

    puts("=== layout ===");
    R(ggml_cont(ctx, a2));
    R(ggml_cont(ctx, ggml_transpose(ctx, a2)));
    R(ggml_cast(ctx, a2, GGML_TYPE_F16));
    R(ggml_cpy(ctx, a2, b2));
    R(ggml_cont(ctx, ggml_permute(ctx, a4, 1, 2, 0, 3)));
    R(ggml_set(ctx, a2, vec, 32, 64, 96, 16));
    R(ggml_diag(ctx, vec));
    R(ggml_diag_mask_inf(ctx, sq, 3));
    R(ggml_diag_mask_zero(ctx, sq, 3));

    puts("=== softmax ===");
    {
        struct ggml_tensor * logits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 4);
        struct ggml_tensor * mask   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 32);
        R(ggml_soft_max(ctx, logits));
        R(ggml_soft_max_ext(ctx, logits, mask, 0.125f, 8.0f));
    }

    puts("=== rope ===");
    {
        struct ggml_tensor * q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 32, 6);
        struct ggml_tensor * pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
        struct ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        R(ggml_rope(ctx, q, pos, 128, GGML_ROPE_TYPE_NEOX));
        R(ggml_rope_ext(ctx, q, pos, freqs, 128, GGML_ROPE_TYPE_NEOX, 4096,
                        500000.0f, 0.5f, 1.0f, 0.75f, 32.0f, 1.0f));
    }

    puts("=== gather ===");
    {
        struct ggml_tensor * tbl = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 100);
        struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7);
        RB(ggml_get_rows(ctx, tbl, ids), 100);
        struct ggml_tensor * tf  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 100);
        struct ggml_tensor * id2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7);
        RB(ggml_get_rows(ctx, tf, id2), 100);
    }

    puts("=== sorting and generation ===");
    R(ggml_argsort(ctx, a2, GGML_SORT_ORDER_DESC));
    R(ggml_argsort(ctx, a2, GGML_SORT_ORDER_ASC));
    R(ggml_top_k(ctx, a2, 4));
    R(ggml_arange(ctx, 0.0f, 10.0f, 2.5f));
    R(ggml_timestep_embedding(ctx, vec, 64, 10000));
    R(ggml_roll(ctx, a2, 3, -2, 0, 0));
    R(ggml_tri(ctx, sq, GGML_TRI_TYPE_LOWER));
    R(ggml_fill(ctx, a2, -1.5f));

    puts("=== padding and resampling ===");
    R(ggml_upscale(ctx, a4, 2, GGML_SCALE_MODE_NEAREST));
    R(ggml_upscale_ext(ctx, a4, 24, 20, 8, 2, GGML_SCALE_MODE_BILINEAR));
    R(ggml_pad(ctx, a4, 2, 3, 0, 0));
    R(ggml_pad_reflect_1d(ctx, a2, 3, 5));

    puts("=== pooling ===");
    R(ggml_pool_1d(ctx, a4, GGML_OP_POOL_AVG, 4, 4, 0));
    R(ggml_pool_2d(ctx, a4, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0));


    /* ------------------------------------------------------------------
     * Everything below was added before the bulk of ops.cpp was ported, so
     * that the port lands on a gate rather than having one fitted to it.
     * Shapes follow tests/test-backend-ops.cpp where it has a case.
     * ------------------------------------------------------------------ */

    puts("=== repeat and concat ===");
    R(ggml_repeat(ctx, row, a2));
    R(ggml_repeat(ctx, a2, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 8, 2, 3)));
    R(ggml_repeat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4), ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 16, 8, 2)));
    R(ggml_repeat_back(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 8, 2, 3), a2));
    R(ggml_concat(ctx, a3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 4, 2), 0));
    R(ggml_concat(ctx, a3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 3, 2), 1));
    R(ggml_concat(ctx, a3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 5), 2));
    R(ggml_concat(ctx, a4, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 8, 1), 3));
    R(ggml_concat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4), ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 3), 1));
    R(ggml_concat(ctx, ggml_transpose(ctx, a2), ggml_transpose(ctx, b2), 0));   /* non-contiguous */

    puts("=== accumulate and indexed add ===");
    R(ggml_acc(ctx, a2, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2), a2->nb[1], a2->nb[2], a2->nb[3], a2->nb[1] + 2*sizeof(float)));
    RB(ggml_add_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 3), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 5),
                   ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3)), 5);

    puts("=== fused rms_norm * weight ===");
    R(ggml_mul(ctx, ggml_rms_norm(ctx, a2, 1e-5f), row));
    R(ggml_mul(ctx, ggml_rms_norm(ctx, a3, 1e-6f), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)));

    puts("=== scatter ===");
    RB(ggml_get_rows_back(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 7), ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7),
                          ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 10)), 10);
    RB(ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 10), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3),
                     ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3)), 10);
    RB(ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 10), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3),
                     ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3)), 10);
    RB(ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 10), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3),
                     ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3)), 10);
    {
        struct ggml_tensor * tf = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 100);
        RB(ggml_get_rows(ctx, tf, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7)), 100);
        struct ggml_tensor * ti = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 16, 100);
        RB(ggml_get_rows(ctx, ti, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7)), 100);
    }

    puts("=== backward passes ===");
    R(ggml_soft_max_ext_back(ctx, a2, b2, 0.5f, 0.0f));
    {
        struct ggml_tensor * q   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 6);
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
        R(ggml_rope_ext_back(ctx, q, pos, NULL, 64, GGML_ROPE_TYPE_NEOX, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
        R(ggml_rope_ext_back(ctx, q, pos, NULL, 64, GGML_ROPE_TYPE_NORMAL, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
    }
    {
        struct ggml_tensor * af = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 2, 1);
        struct ggml_tensor * g  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 2, 1);
        R(ggml_pool_2d_back(ctx, g, af, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0));
        R(ggml_pool_2d_back(ctx, g, af, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0));
    }

    puts("=== rope variants ===");
    {
        struct ggml_tensor * q   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 6);
        struct ggml_tensor * qh  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 4, 6);
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
        struct ggml_tensor * p4  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 24);
        int sections[GGML_MROPE_SECTIONS] = { 12, 10, 10, 0 };
        int vsect[GGML_MROPE_SECTIONS]    = { 16, 16, 0, 0 };
        R(ggml_rope_ext(ctx, q, pos, NULL, 64, GGML_ROPE_TYPE_NORMAL, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
        R(ggml_rope_ext(ctx, q, pos, NULL, 48, GGML_ROPE_TYPE_NEOX, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)); /* partial */
        R(ggml_rope_ext(ctx, qh, pos, NULL, 64, GGML_ROPE_TYPE_NEOX, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
        R(ggml_rope_ext(ctx, q, pos, NULL, 64, GGML_ROPE_TYPE_NEOX, 2048, 10000.0f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f)); /* yarn */
        R(ggml_rope_multi(ctx, q, p4, NULL, 64, sections, GGML_ROPE_TYPE_MROPE, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
        R(ggml_rope_multi(ctx, q, p4, NULL, 64, sections, GGML_ROPE_TYPE_IMROPE, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
        R(ggml_rope_multi(ctx, q, p4, NULL, 32, vsect, GGML_ROPE_TYPE_VISION, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
        R(ggml_rope_multi(ctx, qh, p4, NULL, 64, sections, GGML_ROPE_TYPE_IMROPE, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
    }

    puts("=== activations, f16 ===");
    {
        struct ggml_tensor * h2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4);
        struct ggml_tensor * g2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4);
        R(ggml_gelu(ctx, h2));
        R(ggml_gelu_quick(ctx, h2));
        R(ggml_gelu_erf(ctx, h2));
        R(ggml_silu(ctx, h2));
        R(ggml_relu(ctx, h2));
        R(ggml_tanh(ctx, h2));
        R(ggml_leaky_relu(ctx, h2, 0.01f, false));
        R(ggml_silu_back(ctx, h2, g2));
        R(ggml_swiglu(ctx, h2));
        R(ggml_geglu(ctx, h2));
        R(ggml_reglu(ctx, h2));
        R(ggml_geglu_erf(ctx, h2));
        R(ggml_geglu_quick(ctx, h2));
        R(ggml_swiglu_split(ctx, h2, g2));
        RW(ggml_gelu(ctx, h2));
        R(ggml_clamp(ctx, h2, -0.5f, 0.5f));
        R(ggml_soft_max(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 33, 5)));
        R(ggml_norm(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 33, 5), 1e-5f));
        {   /* rows contiguous, tensor not: norm's other path */
            struct ggml_tensor * big = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 4);
            R(ggml_norm(ctx, ggml_view_2d(ctx, big, 8, 4, big->nb[1], 0), 1e-5f));
            R(ggml_rms_norm(ctx, ggml_view_2d(ctx, big, 8, 4, big->nb[1], 0), 1e-5f));
        }
        R(ggml_out_prod(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 8), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 8)));
    }

    puts("=== convolution ===");
    {
        struct ggml_tensor * k2  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 3, 4);
        struct ggml_tensor * k2f = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 3, 4);
        struct ggml_tensor * x2  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 10, 8, 3, 2);
        R(ggml_im2col(ctx, k2, x2, 1, 1, 1, 1, 1, 1, true, GGML_TYPE_F32));
        R(ggml_im2col(ctx, k2, x2, 2, 1, 0, 1, 1, 2, true, GGML_TYPE_F16));
        R(ggml_im2col(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 3, 4), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 12, 3, 2),
                      1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32));
        {
            int64_t ne[4] = { 10, 8, 3, 2 };
            /* (grad, kernel), as ggml.c:7032 calls it. ggml.h labels the
             * first argument "convolution kernel", but the kernel reads src[0]
             * as the gradient; passed the other way round it reads past the
             * end of the kernel tensor, and the output varied run to run. */
            R(ggml_im2col_back(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 27, 10, 8, 2), k2f, ne, 1, 1, 1, 1, 1, 1, true));
        }
        R(ggml_im2col_3d(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 3, 4), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 6, 6, 5, 2),
                         2, 1, 1, 1, 1, 1, 1, 1, 1, 1, GGML_TYPE_F32));
        R(ggml_conv_transpose_1d(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 2), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 7, 2), 2, 0, 1));
        R(ggml_conv_transpose_1d(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 4, 2), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 7, 2), 1, 0, 1));
        R(ggml_conv_transpose_2d_p0(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 4, 2), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 5, 2, 1), 2));
        R(ggml_conv_2d_direct(ctx, k2f, x2, 1, 1, 1, 1, 1, 1));
        R(ggml_conv_2d_direct(ctx, k2, x2, 2, 1, 0, 1, 1, 1));
        R(ggml_conv_2d_dw_direct(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, 3), x2, 1, 1, 1, 1, 1, 1));
        R(ggml_conv_2d_dw_direct(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, 3), x2, 2, 1, 0, 1, 1, 1));
        R(ggml_conv_3d_direct(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 3, 2 * 3), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 6, 6, 5, 2 * 2),
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3));
        R(ggml_col2im_1d(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * 4, 6), 2, 4, 1));
    }

    puts("=== attention ===");
    {
        struct ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 5, 4, 1);
        struct ggml_tensor * q1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 4, 1);
        struct ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 32, 2, 1);
        struct ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 32, 2, 1);
        struct ggml_tensor * k8 = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, 64, 32, 2, 1);
        struct ggml_tensor * v8 = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, 64, 32, 2, 1);
        struct ggml_tensor * kf = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 32, 2, 1);
        struct ggml_tensor * vf = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 32, 2, 1);
        /* A causal mask with real -inf entries, so the kernels' skip of a
         * fully masked score is exercised rather than never taken. */
        struct ggml_tensor * m  = ggml_cast(ctx, ggml_diag_mask_inf(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 5), 27), GGML_TYPE_F16);
        struct ggml_tensor * m1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 1, 1, 1);
        R(ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 0.0f));
        R(ggml_flash_attn_ext(ctx, q, k, v, NULL, 0.125f, 0.0f, 0.0f));
        R(ggml_flash_attn_ext(ctx, q1, k, v, m1, 0.125f, 0.0f, 0.0f));
        R(ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 8.0f, 0.0f));   /* alibi */
        R(ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 30.0f));  /* softcap */
        R(ggml_flash_attn_ext(ctx, q, k8, v8, m, 0.125f, 0.0f, 0.0f));
        R(ggml_flash_attn_ext(ctx, q, kf, vf, m, 0.125f, 0.0f, 0.0f));
        {
            struct ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(o, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4));
            run_amp("flash_attn_ext + sinks", o, 0, 1.0f);
        }
        {
            struct ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
            run_amp("flash_attn_ext prec=f32", o, 0, 1.0f);
        }
        /* Longer, to reach any tiled or chunked path. */
        {
            struct ggml_tensor * ql = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 40, 4, 1);
            struct ggml_tensor * kl = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 300, 2, 1);
            struct ggml_tensor * vl = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 300, 2, 1);
            struct ggml_tensor * ml = ggml_cast(ctx, ggml_diag_mask_inf(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 300, 40), 260), GGML_TYPE_F16);
            R(ggml_flash_attn_ext(ctx, ql, kl, vl, ml, 0.0883883f, 0.0f, 0.0f));
        }
        /* The two fast paths, each with its own entry condition (ops.cpp's
         * dispatcher): split-KV for a single query row against at least 512
         * keys, tiled for at least 64 query rows. Neither is reached by the
         * shapes above. */
        for (int kt = 0; kt < 2; kt++) {
            const enum ggml_type tkv = kt == 0 ? GGML_TYPE_F16 : GGML_TYPE_F32;
            struct ggml_tensor * qs = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 4, 1);
            struct ggml_tensor * ks = ggml_new_tensor_4d(ctx, tkv, 64, 600, 2, 1);
            struct ggml_tensor * vs = ggml_new_tensor_4d(ctx, tkv, 64, 600, 2, 1);
            struct ggml_tensor * ms = ggml_cast(ctx, ggml_diag_mask_inf(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 600, 1), 550), GGML_TYPE_F16);
            run_amp(kt == 0 ? "flash_attn_ext split-kv f16" : "flash_attn_ext split-kv f32",
                    ggml_flash_attn_ext(ctx, qs, ks, vs, ms, 0.125f, 0.0f, 0.0f), 0, 1.0f);
            {
                struct ggml_tensor * o = ggml_flash_attn_ext(ctx, qs, ks, vs, ms, 0.125f, 0.0f, 30.0f);
                ggml_flash_attn_ext_add_sinks(o, ggml_scale(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4), 4.0f));
                run_amp(kt == 0 ? "flash_attn_ext split-kv f16 sinks+softcap" : "flash_attn_ext split-kv f32 sinks+softcap", o, 0, 1.0f);
            }

            struct ggml_tensor * qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 70, 4, 1);
            struct ggml_tensor * kt_ = ggml_new_tensor_4d(ctx, tkv, 64, 150, 2, 1);
            struct ggml_tensor * vt = ggml_new_tensor_4d(ctx, tkv, 64, 150, 2, 1);
            struct ggml_tensor * mt = ggml_cast(ctx, ggml_diag_mask_inf(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 150, 70), 80), GGML_TYPE_F16);
            run_amp(kt == 0 ? "flash_attn_ext tiled f16" : "flash_attn_ext tiled f32",
                    ggml_flash_attn_ext(ctx, qt, kt_, vt, mt, 0.125f, 0.0f, 0.0f), 0, 1.0f);
            run_amp(kt == 0 ? "flash_attn_ext tiled f16 nomask" : "flash_attn_ext tiled f32 nomask",
                    ggml_flash_attn_ext(ctx, qt, kt_, vt, NULL, 0.125f, 0.0f, 0.0f), 0, 1.0f);
            {
                /* Sinks in [-4, 4): at [-1, 1) a sink seldom beats the row's
                 * max score, the `ms = 1` branch is exact, and an unfused
                 * `S*ms + vs` escaped -- measured. */
                struct ggml_tensor * o = ggml_flash_attn_ext(ctx, qt, kt_, vt, mt, 0.125f, 8.0f, 30.0f);
                ggml_flash_attn_ext_add_sinks(o, ggml_scale(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4), 4.0f));
                run_amp(kt == 0 ? "flash_attn_ext tiled f16 alibi+softcap+sinks" : "flash_attn_ext tiled f32 alibi+softcap+sinks", o, 0, 1.0f);
            }
        }
        /* No flash_attn_back case: its constructor aborts at ggml.c:5523,
         * "TODO: adapt to ggml_flash_attn_ext() changes", so no public graph
         * reaches the kernel. */
        {
            struct ggml_tensor * iq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 4, 3, 1);
            struct ggml_tensor * iw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 3, 1, 1);
            struct ggml_tensor * im = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 3, 1, 1);
            R(ggml_lightning_indexer(ctx, iq, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 1, 16, 1), iw, im));
            R(ggml_lightning_indexer(ctx, iq, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 1, 16, 1), iw, im));
        }
    }

    puts("=== state space and recurrent ===");
    R(ggml_ssm_conv(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 + 5, 16, 2), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 16)));
    R(ggml_ssm_conv(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 + 1, 16, 1), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 16)));
    {
        /* Mamba-2: d_state 16, head_dim 8, 4 heads, 2 groups. */
        const int ds = 16, hd = 8, nh = 4, ng = 2, ns = 2;
        for (int t = 0; t < 3; t++) {
            const int nt = t == 0 ? 5 : t == 1 ? 1 : 300;  /* prefill, decode, long */
            RB(ggml_ssm_scan(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds, hd, nh, ns),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, nh, nt, ns),
                                  ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nh, nt, ns),
                                  ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nh),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds, ng, nt, ns),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds, ng, nt, ns),
                                  ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ns), 1), ns);
        }
        /* Rollback snapshots, K = 3. */
        RB(ggml_ssm_scan(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds, hd, nh, ns),
                              ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, nh, 6, ns),
                              ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nh, 6, ns),
                              ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nh),
                              ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds, ng, 6, ns),
                              ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds, ng, 6, ns),
                              ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ns), 3), ns);
        /* d_state off the 16-wide step, so the scalar tail runs: 20 leaves
         * 4 (vectorizable), 19 leaves 3 (scalar), 27 leaves 11. */
        for (int t = 0; t < 3; t++) {
            const int dst_ = t == 0 ? 20 : t == 1 ? 19 : 27;
            RB(ggml_ssm_scan(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dst_, hd, nh, ns),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, nh, 3, ns),
                                  ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nh, 3, ns),
                                  ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nh),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dst_, ng, 3, ns),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dst_, ng, 3, ns),
                                  ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ns), 1), ns);
        }
        /* Mamba-1: head_dim 1, A per state. */
        for (int t = 0; t < 3; t++) {
            const int d1 = t == 0 ? ds : t == 1 ? 3 : 10;   /* main loop, scalar, vector epilogue */
            RB(ggml_ssm_scan(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d1, 1, 32, 1),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 32, 4, 1),
                                  ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 4, 1),
                                  ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d1, 32),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d1, 1, 4, 1),
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d1, 1, 4, 1),
                                  ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), 1), 1);
        }
    }
    {
        /* Gated delta net, the Qwen3.5 linear-attention layer. Inputs are
         * shaped the way the model shapes them: q and k L2-normalised, g a
         * negative log-decay, beta in (0, 1). */
        const int hs = 16, hc = 2, ns = 2;
        for (int t = 0; t < 4; t++) {
            const int nt   = t == 0 ? 5 : t == 1 ? 1 : t == 2 ? 70 : 4;
            const int kda  = t == 3;
            const int vrep = t == 3 ? 2 : 1;
            struct ggml_tensor * q = ggml_l2_norm(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, hc, nt, ns), 1e-6f);
            struct ggml_tensor * k = ggml_l2_norm(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, hc, nt, ns), 1e-6f);
            struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, hc * vrep, nt, ns);
            struct ggml_tensor * g = ggml_scale_bias(ctx, ggml_abs(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kda ? hs : 1, hc * vrep, nt, ns)), -5.0f, -1e-4f);
            struct ggml_tensor * beta  = ggml_sigmoid(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, hc * vrep, nt, ns));
            struct ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, hs, hc * vrep, ns);
            R(ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1));
        }
        {
            struct ggml_tensor * q = ggml_l2_norm(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, hc, 6, ns), 1e-6f);
            struct ggml_tensor * k = ggml_l2_norm(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, hc, 6, ns), 1e-6f);
            struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, hc, 6, ns);
            struct ggml_tensor * g = ggml_scale_bias(ctx, ggml_abs(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, hc, 6, ns)), -5.0f, -1e-4f);
            struct ggml_tensor * beta  = ggml_sigmoid(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, hc, 6, ns));
            struct ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, hs, hc, ns);
            R(ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 3));   /* snapshots */
        }
    }
    {
        const int hs = 64, hc = 2, nt = 6, ns = 2;
        #define T3() ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, hc, nt)
        struct ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hs * hs * hc, ns);
        R(ggml_rwkv_wkv6(ctx, T3(), T3(), T3(), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hs, hc), T3(), s));
        R(ggml_rwkv_wkv7(ctx, T3(), T3(), T3(), T3(), ggml_l2_norm(ctx, T3(), 1e-7f), ggml_l2_norm(ctx, T3(), 1e-7f), s));
        R(ggml_gated_linear_attn(ctx, T3(), T3(), T3(), T3(), s, 0.125f));
        #undef T3
    }
    {
        const int hc = 4, nt = 5, ne = 8;
        R(ggml_dsv4_hc_comb(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (2 + hc) * hc, nt), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3),
                            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (2 + hc) * hc), 1e-6f, 4));
        R(ggml_dsv4_hc_pre(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt)));
        R(ggml_dsv4_hc_post(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nt), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt),
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, nt)));
    }

    puts("=== windows and relative position ===");
    R(ggml_win_part(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 10, 9), 4));
    R(ggml_win_unpart(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 4, 9), 10, 9, 4));
    R(ggml_get_rel_pos(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 9), 5, 5));
    R(ggml_add_rel_pos(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 2), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 4, 2),
                       ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 4, 2)));

    puts("=== transforms ===");
    {
        struct ggml_tensor * h = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3));
        ggml_mul_mat_set_hint(h, GGML_HINT_SRC0_IS_HADAMARD);
        run_amp("mul_mat hint=hadamard (fwht)", h, 0, 1.0f);
    }
    {
        /* Lower-triangular with a diagonal bounded away from zero. */
        struct ggml_tensor * lt = ggml_tri(ctx, ggml_scale_bias(ctx, ggml_abs(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 11, 11, 2)), 0.9f, 0.1f),
                                           GGML_TRI_TYPE_LOWER_DIAG);
        R(ggml_solve_tri(ctx, lt, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 11, 2), true, true, false));
    }

    puts("=== training ===");
    {
        struct ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 6);
        struct ggml_tensor * labels = ggml_soft_max(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 6));
        R(ggml_cross_entropy_loss(ctx, logits, labels));
        R(ggml_cross_entropy_loss_back(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), logits, labels));
        struct ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_set_param(w);   /* the optimizer steps assert it */
        struct ggml_tensor * gr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        struct ggml_tensor * mm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        struct ggml_tensor * vv = ggml_abs(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4));
        R(ggml_opt_step_adamw(ctx, w, gr, mm, vv, ggml_abs(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7))));
        struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_set_param(w2);
        R(ggml_opt_step_sgd(ctx, w2, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4),
                            ggml_abs(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2))));
    }

    puts("=== custom ===");
    {
        static float bias = 0.75f;
        R(ggml_map_custom1(ctx, a2, custom1_fn, GGML_N_TASKS_MAX, NULL));
        R(ggml_map_custom2(ctx, a2, b2, custom2_fn, 2, NULL));
        R(ggml_map_custom3(ctx, a2, b2, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4), custom3_fn, GGML_N_TASKS_MAX, NULL));
        struct ggml_tensor * args[1] = { a2 };
        R(ggml_custom_4d(ctx, GGML_TYPE_F32, 8, 4, 1, 1, args, 1, custom_fn, GGML_N_TASKS_MAX, &bias));
    }

    puts("=== sorting, more ===");
    R(ggml_argsort(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 3), GGML_SORT_ORDER_ASC));
    R(ggml_top_k(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 3), 10));
    R(ggml_upscale(ctx, a4, 2, GGML_SCALE_MODE_BILINEAR));
    R(ggml_interpolate(ctx, a4, 24, 20, 8, 2, GGML_SCALE_MODE_BICUBIC));
    R(ggml_interpolate(ctx, a4, 24, 20, 8, 2, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS));
    R(ggml_pad_ext(ctx, a4, 1, 2, 3, 0, 0, 0, 0, 0));

    if (qfile) fclose(qfile);
    ggml_free(ctx);
    return 0;
}
