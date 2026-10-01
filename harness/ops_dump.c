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
            for (int64_t r = 0; r < nrow; r++) {
                for (int64_t i = 0; i < ne0; i++) tmp[i] = lcg_next();
                ggml_quantize_chunk(t->type, tmp, (char *) t->data + r * t->nb[1],
                                    0, 1, ne0, NULL);
            }
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

    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
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
}

#define R(expr)      run_amp(#expr, (expr), 0, 1.0f)
#define RB(expr, b)  run_amp(#expr, (expr), (b), 1.0f)
/* Wide inputs, for the kernels that branch on a threshold. */
#define RW(expr)     run_amp(#expr, (expr), 0, 30.0f)

/* argv: [write-q|read-q <path>] -- see the note on `qfile`. */
int main(int argc, char ** argv) {
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

    if (qfile) fclose(qfile);
    ggml_free(ctx);
    return 0;
}
