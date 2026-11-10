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
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <math.h>
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
/* Leaves a case wants filled by a rule, picked by the tensor's name:
 *
 *   wild:<n>  I32, spanning [-2, n + 2): both sides of a range the op skips
 *             out of, which a fill inside the bound never reaches.
 *   skip:<n>  I32, as `bound` would fill it over [0, n), but with -1 in about
 *             a third of the slots and in all of row 1, an expert id meaning
 *             "no expert".
 *   score     F32 with -inf and NaN mixed in, the two a pick is dead on.
 */
static int fill_named(struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    int bound = 0;

    if (t->type == GGML_TYPE_I32 && sscanf(t->name, "wild:%d", &bound) == 1) {
        int32_t * d = (int32_t *) t->data;
        for (int64_t i = 0; i < n; i++) d[i] = (int32_t) (((uint64_t) i * 7919u) % (uint64_t) (bound + 4)) - 2;
        return 1;
    }
    if (t->type == GGML_TYPE_I32 && sscanf(t->name, "skip:%d", &bound) == 1) {
        int32_t * d = (int32_t *) t->data;
        for (int64_t i = 0; i < n; i++) {
            const int skip = ((uint64_t) i * 2654435761u >> 9) % 3 == 0 || i / t->ne[0] == 1;
            d[i] = skip ? -1 : (int32_t) (((uint64_t) i * 7919u) % (uint64_t) bound);
        }
        return 1;
    }
    if (t->type == GGML_TYPE_F32 && strcmp(t->name, "score") == 0) {
        float * d = (float *) t->data;
        for (int64_t i = 0; i < n; i++) {
            d[i] = lcg_next();
            if (i % 5 == 3) d[i] = -INFINITY;
            if (i % 7 == 2) d[i] = NAN;
        }
        return 1;
    }
    return 0;
}

static void fill_leaf(struct ggml_tensor * t, int64_t bound) {
    const int64_t n = ggml_nelements(t);

    if (fill_named(t)) return;

    switch (t->type) {
        case GGML_TYPE_F32: {
            float * d = (float *) t->data;
            for (int64_t i = 0; i < n; i++) d[i] = lcg_next();
            break;
        }
        case GGML_TYPE_F64: {
            double * d = (double *) t->data;
            for (int64_t i = 0; i < n; i++) d[i] = lcg_next();
            break;
        }
        case GGML_TYPE_I8: {
            int8_t * d = (int8_t *) t->data;
            for (int64_t i = 0; i < n; i++) d[i] = (int8_t) (lcg_next() * 127.0f);
            break;
        }
        case GGML_TYPE_I16: {
            int16_t * d = (int16_t *) t->data;
            for (int64_t i = 0; i < n; i++) d[i] = (int16_t) (lcg_next() * 32767.0f);
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

/* Set around a run_amp call to compute with `cplan.use_ref`, the CPU's
 * reference-only switch: it skips the dispatches a kernel takes over the
 * general path -- the Hadamard hint's fwht, the IQ panel GEMM -- so the
 * general path behind them is what gets compared. */
static int run_use_ref = 0;

/* Builds a one-node graph, fills every leaf, computes on the CPU with one
 * thread, and prints the output bits. */
static void run_amp(const char * label, struct ggml_tensor * node, int64_t bound, float amp) {
    fill_amp = amp;
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, node);

    /* A sentinel under the result, so an element a kernel never writes shows
     * as the sentinel and not as whatever the context's memory held -- which
     * is usually zero, and zero is what MUL_MAT_ID must write to a skipped
     * slot. A view's data is its source's, which the fill below owns. */
    if (node->view_src == NULL) memset(node->data, 0xA5, ggml_nbytes(node));

    lcg_state = 1;
    seen_n = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        fill_inputs(ggml_graph_node(gf, i), bound);
    }

    /* The plan is part of what is compared: `ggml_get_n_tasks` decides how
     * many threads run each node and `ggml_graph_plan` the work buffer, and
     * neither shows in a result a single-op graph computes -- an op planned
     * for one thread computes the same bits. */
    const struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, NULL);

    enum ggml_status st;
    if (run_use_ref) {
        struct ggml_cplan cp = ggml_graph_plan(gf, n_threads, NULL);
        cp.use_ref = true;
        void * work = cp.work_size > 0 ? malloc(cp.work_size) : NULL;
        cp.work_data = (uint8_t *) work;
        st = ggml_graph_compute(gf, &cp);
        free(work);
    } else {
        st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    }
    if (st != GGML_STATUS_SUCCESS) {
        printf("%-34s amp=%-4g STATUS=%d\n", label, (double) amp, (int) st);
        return;
    }

    const size_t nb = ggml_nbytes(node);
    printf("%-34s amp=%-4g plan=%d/%zu type=%-8s ne=[%lld,%lld,%lld,%lld] bytes=%zu hash=%016llx",
           label, (double) amp, plan.n_threads, plan.work_size, ggml_type_name(node->type),
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

/* A tensor named for `fill_named`. */
static struct ggml_tensor * named(struct ggml_tensor * t, const char * fmt, int bound) {
    return ggml_format_name(t, fmt, bound);
}

/* v0.8.0's TOP_K_MASK. Every optional source present and absent, over both
 * mask types, a mask that broadcasts and one that does not; then the same
 * with every source a strided view, so each stride the kernel reads is one
 * the contiguous cases would not notice it had swapped. */
static void top_k_mask_cases(void) {
    enum { n_kv = 70, n_top = 6, ne1 = 4, ne2 = 3, ne3 = 2, n_pick = 11, n_blk = 5, n_extra = 3 };

    static const enum ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16 };
    static const int64_t bcast[][3] = { { ne1, ne2, ne3 }, { 2, 1, 2 }, { 1, 3, 1 } };

    for (size_t ti = 0; ti < sizeof types / sizeof types[0]; ti++) {
        for (size_t bi = 0; bi < sizeof bcast / sizeof bcast[0]; bi++) {
            for (int bits = 0; bits < 8; bits++) {
                const int has_score = bits & 1, has_blocks = bits & 2, has_extra = bits & 4;
                const int picks = has_score || has_blocks ? n_pick : n_kv;
                char label[96];
                snprintf(label, sizeof label, "top_k_mask %s m=[%lld,%lld,%lld] score=%d blocks=%d extra=%d",
                         ggml_type_name(types[ti]), (long long) bcast[bi][0], (long long) bcast[bi][1],
                         (long long) bcast[bi][2], !!has_score, !!has_blocks, !!has_extra);
                struct ggml_tensor * mask = ggml_new_tensor_4d(ctx, types[ti], n_kv, bcast[bi][0], bcast[bi][1], bcast[bi][2]);
                struct ggml_tensor * top  = named(ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_top, ne1, ne2, ne3), "wild:%d", picks);
                struct ggml_tensor * sc   = has_score  ? named(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_pick, ne1, ne2, ne3), "score", 0) : NULL;
                struct ggml_tensor * bl   = has_blocks ? named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_blk, n_pick), "wild:%d", n_kv) : NULL;
                struct ggml_tensor * ex   = has_extra  ? named(ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_extra, ne1, ne2, ne3), "wild:%d", n_kv) : NULL;
                run_amp(label, ggml_top_k_mask(ctx, mask, top, sc, bl, ex), 0, 1.0f);
            }
        }

        /* Every operand strided: a mask row inside a wider one, and the rest
         * transposed, so nb[0] is not the element size. */
        struct ggml_tensor * wide = ggml_new_tensor_4d(ctx, types[ti], n_kv + 9, 2, 1, 2);
        struct ggml_tensor * mask = ggml_view_4d(ctx, wide, n_kv, 2, 1, 2, wide->nb[1], wide->nb[2], wide->nb[3], 3*wide->nb[0]);
        struct ggml_tensor * top  = ggml_permute(ctx, named(ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne1, n_top, ne2, ne3), "wild:%d", n_pick), 1, 0, 2, 3);
        struct ggml_tensor * sc   = ggml_permute(ctx, named(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne1, n_pick, ne2, ne3), "score", 0), 1, 0, 2, 3);
        struct ggml_tensor * bl   = ggml_transpose(ctx, named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_pick, n_blk), "wild:%d", n_kv));
        struct ggml_tensor * ex   = ggml_permute(ctx, named(ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne1, n_extra, ne2, ne3), "wild:%d", n_kv), 1, 0, 2, 3);
        char label[96];
        snprintf(label, sizeof label, "top_k_mask %s, every source strided", ggml_type_name(types[ti]));
        run_amp(label, ggml_top_k_mask(ctx, mask, top, sc, bl, ex), 0, 1.0f);
    }
}

/* v0.8.0: an id of -1 skips the slot. `ggml-cpu.c` zeroes the row of the
 * result, `repack.cpp` does too on its own path, and ADD_ID copies `a`.
 *
 * The repack half needs a weight in a real CPU_REPACK buffer: `run_amp`'s
 * tensors never are, and `repack-diff` calls kernels, not `forward_mul_mat_id`.
 * A type this CPU has no repack for says so and is skipped, on both sides. */
static void skipped_slot_cases(void) {
    static const enum ggml_type types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL,
    };
    static const int nb11[] = { 1, 3 };
    for (size_t ti = 0; ti < sizeof types / sizeof types[0]; ti++) {
        for (size_t k = 0; k < sizeof nb11 / sizeof nb11[0]; k++) {
            char label[96];
            snprintf(label, sizeof label, "mul_mat_id skip %s b=[%d, 9]", ggml_type_name(types[ti]), nb11[k]);
            run_amp(label, ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, types[ti], 256, 16, 4),
                                                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, nb11[k], 9),
                                                named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 9), "skip:%d", 4)), 0, 1.0f);
        }
    }

    /* The long name is the gate: row -1 of `b` is the end of its own
     * `ggml_tensor`, and with an empty name that is zeros, so a kernel that
     * indexes with the -1 adds nothing and passes. */
    run_amp("add_id skip", ggml_add_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 13, 4, 5),
                                            named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 13, 6), "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", 0),
                                            named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 5), "skip:%d", 6)), 0, 1.0f);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    ggml_backend_buffer_type_t * (*get_extra)(ggml_backend_dev_t) =
        (ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack = NULL;
    for (ggml_backend_buffer_type_t * b = get_extra ? get_extra(dev) : NULL; b && *b; b++) {
        if (strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) repack = *b;
    }
    if (repack == NULL) {
        puts("mul_mat_id skip repack: no CPU_REPACK buffer type");
        ggml_backend_free(backend);
        return;
    }

    static const enum ggml_type rtypes[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL };
    for (size_t ti = 0; ti < sizeof rtypes / sizeof rtypes[0]; ti++) {
        const int64_t k = 512, n = 16, n_as = 4, n_used = 3, n_tok = 9;

        struct ggml_init_params wp = { 16u * 1024u * 1024u, NULL, true };
        struct ggml_context * wctx = ggml_init(wp);
        struct ggml_context * ictx = ggml_init(wp);
        struct ggml_tensor * as = ggml_new_tensor_3d(wctx, rtypes[ti], k, n, n_as);
        struct ggml_tensor * b  = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, k, 1, n_tok);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, n_used, n_tok);
        struct ggml_tensor * out = ggml_mul_mat_id(ictx, as, b, ids);

        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, repack);
        ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, ggml_backend_cpu_buffer_type());

        /* `set_tensor` reads `extra` unchecked, and a type this CPU has no
         * repack for has none: `supports_op` keeps such a weight out of the
         * buffer in a real load. */
        if (as->extra == NULL) {
            printf("mul_mat_id skip repack %-8s not selected on this target\n", ggml_type_name(rtypes[ti]));
            ggml_backend_buffer_free(wbuf);
            ggml_backend_buffer_free(ibuf);
            ggml_free(wctx);
            ggml_free(ictx);
            continue;
        }

        /* The quantized bytes go through a scratch tensor, for `fill_leaf`'s
         * shared-bytes replay; then the buffer's set_tensor repacks them. */
        struct ggml_tensor * qs = ggml_new_tensor_3d(ctx, rtypes[ti], k, n, n_as);
        fill_amp = 1.0f;
        lcg_state = 1;
        fill_leaf(qs, 0);
        ggml_backend_tensor_set(as, qs->data, 0, ggml_nbytes(as));

        struct ggml_tensor * bs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tok);
        fill_leaf(bs, 0);
        ggml_backend_tensor_set(b, bs->data, 0, ggml_nbytes(b));

        struct ggml_tensor * is = named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok), "skip:%d", (int) n_as);
        fill_leaf(is, 0);
        ggml_backend_tensor_set(ids, is->data, 0, ggml_nbytes(ids));

        memset(out->data, 0xA5, ggml_nbytes(out));

        struct ggml_cgraph * gf = ggml_new_graph(ictx);
        ggml_build_forward_expand(gf, out);
        const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
        printf("mul_mat_id skip repack %-8s status=%d hash=%016llx\n", ggml_type_name(rtypes[ti]),
               (int) st, (unsigned long long) fnv1a(out->data, ggml_nbytes(out)));

        ggml_backend_buffer_free(wbuf);
        ggml_backend_buffer_free(ibuf);
        ggml_free(wctx);
        ggml_free(ictx);
    }
    ggml_backend_free(backend);
}

/* v0.8.0's concat, one body for every type: a tile of rows at a time, memcpy
 * for a source with contiguous rows and element by element for a strided
 * one at each width. More than GGML_CONCAT_TILE (32) rows, so a tile ends on
 * the tile size, on the end of a thread's share, and for dim 1 on the end of
 * src0. */
static void concat_cases(void) {
    static const enum ggml_type types[] = {
        GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F64,
    };
    for (size_t ti = 0; ti < sizeof types / sizeof types[0]; ti++) {
        const enum ggml_type t = types[ti];
        const char * tn = ggml_type_name(t);
        char label[96];
        for (int dim = 0; dim < 4; dim++) {
            int64_t a[4] = { 9, 40, 3, 2 }, b[4] = { 9, 40, 3, 2 };
            b[dim] = dim == 0 ? 7 : dim == 1 ? 51 : 2;
            snprintf(label, sizeof label, "concat %s dim %d", tn, dim);
            run_amp(label, ggml_concat(ctx, ggml_new_tensor(ctx, t, 4, a), ggml_new_tensor(ctx, t, 4, b), dim), 0, 1.0f);

            /* src0 strided, src1 strided, and both: a transposed view of the
             * tensor with its first two dimensions swapped. */
            for (int which = 1; which < 4; which++) {
                int64_t at[4] = { a[1], a[0], a[2], a[3] }, bt[4] = { b[1], b[0], b[2], b[3] };
                struct ggml_tensor * s0 = which & 1 ? ggml_permute(ctx, ggml_new_tensor(ctx, t, 4, at), 1, 0, 2, 3) : ggml_new_tensor(ctx, t, 4, a);
                struct ggml_tensor * s1 = which & 2 ? ggml_permute(ctx, ggml_new_tensor(ctx, t, 4, bt), 1, 0, 2, 3) : ggml_new_tensor(ctx, t, 4, b);
                snprintf(label, sizeof label, "concat %s dim %d strided %d", tn, dim, which);
                run_amp(label, ggml_concat(ctx, s0, s1, dim), 0, 1.0f);
            }
        }
    }

    static const enum ggml_type qtypes[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K };
    for (size_t ti = 0; ti < sizeof qtypes / sizeof qtypes[0]; ti++) {
        const enum ggml_type t = qtypes[ti];
        const int64_t w = ggml_blck_size(t);
        char label[96];
        for (int dim = 0; dim < 4; dim++) {
            int64_t a[4] = { 2*w, 40, 3, 2 }, b[4] = { 2*w, 40, 3, 2 };
            b[dim] = dim == 0 ? w : dim == 1 ? 51 : 2;
            snprintf(label, sizeof label, "concat %s dim %d", ggml_type_name(t), dim);
            run_amp(label, ggml_concat(ctx, ggml_new_tensor(ctx, t, 4, a), ggml_new_tensor(ctx, t, 4, b), dim), 0, 1.0f);
        }
    }
}

/* v0.8.0: SCALE needs only contiguous rows, so a view with gaps between its
 * rows reaches the kernel, with src0's strides unlike dst's; in place, dst's
 * are src0's. A view that keeps i2 and i3 apart, and a permutation. */
static void scale_view_cases(void) {
    static const enum ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_BF16 };
    for (size_t ti = 0; ti < sizeof types / sizeof types[0]; ti++) {
        const enum ggml_type t = types[ti];
        const char * tn = ggml_type_name(t);
        char label[96];
        struct ggml_tensor * big = ggml_new_tensor_4d(ctx, t, 19, 7, 3, 3);
        #define V() ggml_view_4d(ctx, big, 13, 5, 2, 2, big->nb[1], big->nb[2], big->nb[3], big->nb[1] + 4*big->nb[0])
        #define P() ggml_permute(ctx, ggml_new_tensor_4d(ctx, t, 13, 3, 5, 2), 0, 2, 1, 3)
        #define C(what, expr) snprintf(label, sizeof label, "scale %s %s", tn, what); run_amp(label, (expr), 0, 1.0f)
        C("view",              ggml_scale(ctx, V(), 2.5f));
        C("view bias",         ggml_scale_bias(ctx, V(), 2.5f, -1.25f));
        C("view inplace",      ggml_scale_inplace(ctx, V(), 2.5f));
        C("view bias inplace", ggml_scale_bias_inplace(ctx, V(), 2.5f, -1.25f));
        C("permuted",          ggml_scale(ctx, P(), 0.37f));
        C("permuted bias",     ggml_scale_bias(ctx, P(), 0.37f, 0.11f));
        C("one column",        ggml_scale(ctx, ggml_transpose(ctx, ggml_new_tensor_2d(ctx, t, 6, 1)), 2.5f));
        #undef C
        #undef P
        #undef V
    }
}

/* v0.8.0: `GGML_CPU_FP32_TO_FP16` is the hardware on NEON, where it had been
 * the software routine, so the bits move on a NaN. Every f32 whose low half
 * sits on a rounding boundary, through the row converter, whose tail is the
 * macro; and the two tables `ggml_cpu_init` fills through it. */
extern ggml_fp16_t ggml_table_gelu_f16[1 << 16];
extern ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

static void narrowing_cases(void) {
    static const uint32_t lows[] = { 0x0000, 0x0001, 0x0fff, 0x1000, 0x1001, 0x1fff, 0x2000, 0x2001, 0xffff };
    static float       x[1 << 16];
    static ggml_fp16_t y[1 << 16];
    for (size_t li = 0; li < sizeof lows / sizeof lows[0]; li++) {
        for (uint32_t hi = 0; hi < 1u << 16; hi++) {
            const uint32_t u = hi << 16 | lows[li];
            memcpy(&x[hi], &u, sizeof u);
        }
        ggml_cpu_fp32_to_fp16(x, y, 1 << 16);
        printf("fp32_to_fp16 low=%04x hash=%016llx\n", lows[li], (unsigned long long) fnv1a(y, sizeof y));
    }
    printf("table gelu hash=%016llx\n", (unsigned long long) fnv1a(ggml_table_gelu_f16, sizeof ggml_table_gelu_f16));
    printf("table gelu_quick hash=%016llx\n", (unsigned long long) fnv1a(ggml_table_gelu_quick_f16, sizeof ggml_table_gelu_quick_f16));
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

    /* The instruction set each library was compiled for, so a run at
     * GATE_CPU=apple_m3 against a library built for the baseline -- or one
     * side built for each -- shows as a diff on the first line rather than
     * passing on the kernels it never reached. scripts/ops-diff also checks
     * the value against the compiler's own answer for the CPU. */
    printf("cpu: dotprod=%d matmul_int8=%d sve=%d llamafile=%d\n",
           ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(), ggml_cpu_has_sve(),
           ggml_cpu_has_llamafile());

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
    {   /* v0.7.0 (#29517): a strided view and a permuted one */
        struct ggml_tensor * fb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 12, 6, 3, 2);
        R(ggml_clamp(ctx, ggml_view_4d(ctx, fb, 8, 4, 3, 2, fb->nb[1], fb->nb[2], fb->nb[3], fb->nb[1]), -0.5f, 0.5f));
        R(ggml_clamp(ctx, ggml_permute(ctx, fb, 0, 2, 1, 3), -0.25f, 0.25f));
        R(ggml_clamp(ctx, ggml_view_3d(ctx, fb, 5, 6, 6, fb->nb[1], fb->nb[2], 0), -0.1f, 0.6f));
    }
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
    /* v0.7.0 (#29200): an F16 arm, split and fused, and wide inputs so the
     * `limit` clamps on both operands fire */
    {
        struct ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, 5);
        struct ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, 5);
        R(ggml_swiglu_oai(ctx, h, g, 1.702f, 7.0f));
        R(ggml_swiglu_oai(ctx, h, NULL, 1.702f, 7.0f));
        RW(ggml_swiglu_oai(ctx, h, g, 1.702f, 7.0f));
        RW(ggml_swiglu_oai(ctx, h, NULL, 0.5f, 3.0f));
        RW(ggml_swiglu_oai(ctx, a2, NULL, 0.5f, 3.0f));
    }

    puts("=== reductions ===");
    R(ggml_sum(ctx, a3));
    /* v0.8.0: SUM on rows that are not contiguous */
    R(ggml_sum(ctx, ggml_transpose(ctx, a2)));
    RW(ggml_sum(ctx, ggml_permute(ctx, a4, 1, 2, 0, 3)));
    RW(ggml_sum(ctx, ggml_view_3d(ctx, a4, 5, 3, 2, a4->nb[1], a4->nb[2], 3*a4->nb[0])));
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

        /* v0.6.0: on x86 tinyBLAS no longer refuses `k % KN` (sgemm.cpp:568)
         * and gemm_bloc finishes a row with a masked load_partial step
         * (sgemm.cpp:654) -- reached only when ne00 is not a multiple of 8
         * (AVX2's KN; 16 at AVX-512), which no other float mul_mat here is.
         * 36 is four whole blocks of 8 plus 4 (two of 16 plus 4); 5 is the
         * tail alone. 16 rows, so matmul's m % 4 tiling accepts it; F16 and
         * BF16 reach the kernel after src1 is converted to their
         * vec_dot_type. Elsewhere these run through vec_dot. */
        static const enum ggml_type ftypes[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 };
        static const int kt[] = { 36, 5 };
        for (size_t fi = 0; fi < sizeof ftypes / sizeof ftypes[0]; fi++) {
            for (size_t ki = 0; ki < sizeof kt / sizeof kt[0]; ki++) {
                char label[96];
                snprintf(label, sizeof label, "mul_mat %s x f32 [%d x 16] x 8 (k tail)", ggml_type_name(ftypes[fi]), kt[ki]);
                run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, ftypes[fi], kt[ki], 16),
                                                 ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kt[ki], 8)), 0, 1.0f);
            }
        }
    }

    {
        /* Every type with a CPU vec_dot, at 1 column (the decode path), 2 and
         * 7 (a short prompt). Only Q4_K, Q6_K and Q8_0 were here at first,
         * and a Q5_K divergence surfaced end to end instead -- measured, as
         * the first differing node of a CPU-only Qwen3.5 decode.
         *
         * 2 is for i8mm. With __ARM_FEATURE_MATMUL_INT8 the q4_0, q4_1,
         * q8_0, q4_K and q6_K kernels take two rows per call, and
         * ggml_compute_forward_mul_mat falls back to one whenever a column
         * count or a chunk is odd -- so 1 and 7 alone never reach the
         * two-row arm at all. */
        static const enum ggml_type qtypes[] = {
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
            GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
            GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_NVFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0,
            GGML_TYPE_BF16, GGML_TYPE_F16,
        };
        static const int ncols[] = { 1, 2, 7 };
        for (size_t qi = 0; qi < sizeof qtypes / sizeof qtypes[0]; qi++) {
            for (size_t ci = 0; ci < sizeof ncols / sizeof ncols[0]; ci++) {
                const int nc = ncols[ci];
                char label[96];
                snprintf(label, sizeof label, "mul_mat %s x f32 [512 x 16] x %d", ggml_type_name(qtypes[qi]), nc);
                run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, qtypes[qi], 512, 16),
                                                 ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, nc)), 0, 1.0f);
            }
        }
    }

    puts("=== matmul, the tiled path ===");
    {
        /* tiled/tiled.cpp (v0.6.0, replacing iqp.cpp): the K-quants' and grid
         * IQ types' mul_mat and mul_mat_id as an integer GEMM, which x86
         * takes from AVX2 on once src1 has at least 8 columns (or an expert
         * at least 8 routed rows) -- shapes the sweep above, at 1, 2 and 7
         * columns, never reaches. Elsewhere these run through vec_dot.
         *
         * MUL_MAT: 33 columns is the standard path (more than 16, a ragged
         * 16-column band); 40 rows end on a half microtile. 9 and 16 columns
         * are the narrow path, whose K extent is 512 at ne00 = 512 (two
         * slabs), 768 at 768 (three) and 256 at 1280 (one slab, which takes
         * the NK=1 microtile on the narrow path). [9, 2] clamps a window at
         * the batch boundary; [512 x 160] x 64 gives three threads several
         * chunks each. MUL_MAT_ID: ids (i * 7919) % 4 route 2 x 15 rows as
         * 8, 7, 7, 8 (two experts tiled, two below the floor, in one node),
         * 2 x 40 as 20 each (the standard path), the second once broadcast
         * (ne11 = 1); 160 rows per expert are three row groups of 64. */
        static const enum ggml_type ttypes[] = {
            GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K,
            GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
        };
        for (size_t qi = 0; qi < sizeof ttypes / sizeof ttypes[0]; qi++) {
            const enum ggml_type t = ttypes[qi];
            const char * tn = ggml_type_name(t);
            char label[96];

            snprintf(label, sizeof label, "tiled mul_mat %s [512 x 40] x 33", tn);
            run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, t, 512, 40),
                                             ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 33)), 0, 1.0f);
            static const int nk[] = { 512, 768, 1280 };
            static const int ncols[] = { 9, 16 };
            for (size_t ki = 0; ki < sizeof nk / sizeof nk[0]; ki++) {
                for (size_t ci = 0; ci < sizeof ncols / sizeof ncols[0]; ci++) {
                    snprintf(label, sizeof label, "tiled mul_mat %s [%d x 24] x %d", tn, nk[ki], ncols[ci]);
                    run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, t, nk[ki], 24),
                                                     ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nk[ki], ncols[ci])), 0, 1.0f);
                }
            }
            snprintf(label, sizeof label, "tiled mul_mat %s [512 x 160] x 64", tn);
            run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, t, 512, 160),
                                             ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 64)), 0, 1.0f);
            snprintf(label, sizeof label, "tiled mul_mat %s [512 x 16] x [9, 2]", tn);
            run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, t, 512, 16),
                                             ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 9, 2)), 0, 1.0f);

            static const int ntok[] = { 15, 40, 40 };
            static const int nb11[] = { 2, 2, 1 };
            static const int nrow[] = { 16, 160, 16 };
            for (size_t k = 0; k < sizeof ntok / sizeof ntok[0]; k++) {
                snprintf(label, sizeof label, "tiled mul_mat_id %s [512 x %d x 4] b=[%d, %d]", tn, nrow[k], nb11[k], ntok[k]);
                run_amp(label, ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, t, 512, nrow[k], 4),
                                                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, nb11[k], ntok[k]),
                                                    ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, ntok[k])), 4, 1.0f);
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
    /* v0.7.0 (#30526) splits pad's rows over threads and copies a contiguous
     * row whole: padding on every axis, a negative (cropping) pad on axis 0,
     * a non-contiguous source, and the circular arm */
    R(ggml_pad_ext(ctx, a4, 1, 2, 1, 1, 1, 0, 0, 1));
    R(ggml_pad_ext(ctx, a4, -2, 1, 0, 2, -1, 0, 0, 0));
    R(ggml_pad_ext(ctx, a4, 3, -4, 0, 0, 0, 0, 0, 0));
    R(ggml_pad_ext(ctx, ggml_transpose(ctx, a2), 1, 2, 2, 1, 0, 0, 0, 0));
    R(ggml_pad_ext(ctx, ggml_permute(ctx, a3, 0, 2, 1, 3), 2, 0, 1, 0, 1, 1, 0, 0));
    R(ggml_pad_ext_circular(ctx, a4, 1, 2, 3, 1, 1, 0, 0, 1));
    R(ggml_pad_circular(ctx, a3, 2, 1, 1, 0));
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
    concat_cases();

    puts("=== sparse attention mask ===");
    top_k_mask_cases();

    puts("=== an id of -1 ===");
    skipped_slot_cases();

    puts("=== scale on strided rows ===");
    scale_view_cases();

    puts("=== f32 to f16 ===");
    narrowing_cases();

    puts("=== quick gelu, f32 ===");
    {
        /* v0.8.0 computes it in f32 where a table lost precision: a vector
         * body and a scalar tail that round differently, so a width with a
         * remainder; and past the amplitude where exp overflows, where
         * `ggml_v_expf` leaves its fast path. */
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 13, 5);
        struct ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 13, 5);
        struct ggml_tensor * xg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 26, 5);
        static const float amps[] = { 1.0f, 30.0f, 120.0f };
        for (size_t i = 0; i < sizeof amps / sizeof amps[0]; i++) {
            run_amp("gelu_quick 13", ggml_gelu_quick(ctx, x), 0, amps[i]);
            run_amp("geglu_quick 13", ggml_geglu_quick(ctx, xg), 0, amps[i]);
            run_amp("geglu_quick_split 13", ggml_geglu_quick_split(ctx, x, g), 0, amps[i]);
        }
    }

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
    /* v0.7.0 (#30454): SET_ROWS is planned for every thread, and an F16
     * source converts through a per-thread row of the work buffer */
    RB(ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 12), ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 7),
                     ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 7)), 12);
    RB(ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 12), ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 7),
                     ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7)), 12);
    RB(ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 12), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 7),
                     ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 7)), 12);
    {
        struct ggml_tensor * tf = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 100);
        RB(ggml_get_rows(ctx, tf, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7)), 100);
        struct ggml_tensor * ti = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 16, 100);
        RB(ggml_get_rows(ctx, ti, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7)), 100);
    }

    puts("=== backward passes ===");
    R(ggml_soft_max_ext_back(ctx, a2, b2, 0.5f, 0.0f));
    /* 0.5 scales exactly, so where the scale is applied cannot show; 0.37
     * is what tells v0.6.0's `scale * (dy - dot) * y` from v0.5.0's order. */
    R(ggml_soft_max_ext_back(ctx, a2, b2, 0.37f, 0.0f));
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
        /* v0.7.0 (#29517): CLAMP on a non-contiguous view, rows addressed
         * through all three outer strides */
        {
            struct ggml_tensor * hb = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 12, 6, 3);
            R(ggml_clamp(ctx, ggml_view_3d(ctx, hb, 8, 4, 3, hb->nb[1], hb->nb[2], 2 * hb->nb[0]), -0.5f, 0.5f));
            R(ggml_clamp(ctx, ggml_permute(ctx, hb, 0, 2, 1, 3), -0.25f, 0.25f));
        }
        R(ggml_soft_max(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 33, 5)));
        R(ggml_norm(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 33, 5), 1e-5f));
        {   /* rows contiguous, tensor not: norm's other path */
            struct ggml_tensor * big = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 4);
            R(ggml_norm(ctx, ggml_view_2d(ctx, big, 8, 4, big->nb[1], 0), 1e-5f));
            R(ggml_rms_norm(ctx, ggml_view_2d(ctx, big, 8, 4, big->nb[1], 0), 1e-5f));
        }
        R(ggml_out_prod(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 8), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 8)));
    }

    puts("=== activations, bf16 ===");
    {
        /* v0.6.0's BF16 arms of the unary and GLU ops and of scale: a scalar
         * loop each, f32 math narrowed back to BF16. 8 rows, so three
         * threads split them unevenly. */
        struct ggml_tensor * hb = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 64, 8);
        struct ggml_tensor * gb = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 64, 8);
        R(ggml_gelu(ctx, hb));
        R(ggml_gelu_quick(ctx, hb));
        R(ggml_gelu_erf(ctx, hb));
        R(ggml_silu(ctx, hb));
        RW(ggml_gelu(ctx, hb));
        RW(ggml_gelu_quick(ctx, hb));
        RW(ggml_gelu_erf(ctx, hb));
        RW(ggml_silu(ctx, hb));
        R(ggml_swiglu(ctx, hb));
        R(ggml_swiglu_swapped(ctx, hb));
        R(ggml_geglu(ctx, hb));
        R(ggml_reglu(ctx, hb));
        R(ggml_geglu_erf(ctx, hb));
        R(ggml_geglu_quick(ctx, hb));
        R(ggml_swiglu_split(ctx, hb, gb));
        R(ggml_geglu_split(ctx, hb, gb));
        R(ggml_reglu_split(ctx, hb, gb));
        R(ggml_geglu_erf_split(ctx, hb, gb));
        R(ggml_geglu_quick_split(ctx, hb, gb));
        RW(ggml_geglu(ctx, hb));
        RW(ggml_swiglu_split(ctx, hb, gb));
        RW(ggml_geglu_quick_split(ctx, hb, gb));
        R(ggml_scale(ctx, hb, 2.5f));
        R(ggml_scale_bias(ctx, hb, 2.5f, -1.25f));
        RW(ggml_scale_bias(ctx, hb, 0.37f, 0.11f));
        {   /* A fused and an unfused `s*x + b` differ in the last f32 bit,
             * which BF16 keeps only where it straddles a rounding boundary:
             * about one element in 10^5, so 2^20 of them -- the case that
             * can tell `scaleBf16`'s fusion is right. gelu's `1 + A*x*x`
             * the same way. */
            struct ggml_tensor * big = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 1024, 1024);
            R(ggml_scale_bias(ctx, big, 0.37f, 0.11f));
            R(ggml_gelu(ctx, big));
            RW(ggml_geglu_split(ctx, big, big));
        }
        /* BF16 src1 widened into the F32 work buffer: only against an F32
         * src0, whose vec_dot_type is F32 (ggml-cpu.cpp's supports_op). */
        R(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 16), ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 64, 5)));
        R(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 16), ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, 64, 9, 2)));
    }

    puts("=== convolution ===");
    {
        /* v0.7.0 (#28738): batched input, whose rows used to come out in
         * the wrong order */
        R(ggml_conv_1d(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 4, 5), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 17, 4, 3), 1, 1, 1));
        R(ggml_conv_1d(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 5), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 17, 4, 2), 2, 0, 2));
    }
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
        /* Head size 36: not a whole number of f32 vectors on x86 (8 or 16
         * lanes), so before v0.6.0 x86 took the one-chunk path here, and
         * now takes the tiled one with simd_gemm's masked tail; on ARM (4
         * lanes) it was tiled all along. 150 keys end on a ragged KV tile,
         * which with softcap is where v0.6.0 moved the softcap ahead of the
         * -inf padding. */
        for (int kt = 0; kt < 2; kt++) {
            const enum ggml_type tkv = kt == 0 ? GGML_TYPE_F16 : GGML_TYPE_F32;
            struct ggml_tensor * qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 36, 70, 4, 1);
            struct ggml_tensor * kt_ = ggml_new_tensor_4d(ctx, tkv, 36, 150, 2, 1);
            struct ggml_tensor * vt = ggml_new_tensor_4d(ctx, tkv, 36, 150, 2, 1);
            struct ggml_tensor * mt = ggml_cast(ctx, ggml_diag_mask_inf(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 150, 70), 80), GGML_TYPE_F16);
            run_amp(kt == 0 ? "flash_attn_ext tiled d36 f16" : "flash_attn_ext tiled d36 f32",
                    ggml_flash_attn_ext(ctx, qt, kt_, vt, mt, 0.125f, 0.0f, 0.0f), 0, 1.0f);
            run_amp(kt == 0 ? "flash_attn_ext tiled d36 f16 softcap" : "flash_attn_ext tiled d36 f32 softcap",
                    ggml_flash_attn_ext(ctx, qt, kt_, vt, NULL, 0.125f, 0.0f, 30.0f), 0, 1.0f);
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

    /* v0.9.0: ggml_flash_attn_ext_rows. Each q slice i3 attends over its own
     * list of K/V rows, kv_rows[:, i3] (src[5]); a negative row is padding.
     * The kv_rows fill (`skip:`) puts -1 in a third of the slots and in all
     * of slice 1, so a slice with nothing to attend to is compared too. K and
     * V hold more rows than any list names, and the lists repeat and reorder
     * rows, so reading row ic instead of kv_rows[ic] shows. One case per
     * kernel path: the one-chunk path (few q rows; f16, q8_0 and f32 K/V),
     * the tiled path (64+ q rows, a ragged last KV tile), and a single q row
     * over 520 mask columns, which without kv_rows would take the split-KV
     * path -- kv_rows must keep it off. */
    puts("=== flash attention over K/V row lists ===");
    {
        const enum ggml_type kvt[] = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_F32 };
        for (int t = 0; t < 3; t++) {
            struct ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 5, 4, 3);
            struct ggml_tensor * k  = ggml_new_tensor_4d(ctx, kvt[t], 64, 45, 2, 1);
            struct ggml_tensor * v  = ggml_new_tensor_4d(ctx, kvt[t], 64, 45, 2, 1);
            struct ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 37, 5, 1, 3);
            struct ggml_tensor * kr = named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 37, 3), "skip:%d", 45);
            char label[96];
            snprintf(label, sizeof label, "flash_attn_ext_rows %s", ggml_type_name(kvt[t]));
            run_amp(label, ggml_flash_attn_ext_rows(ctx, q, k, v, m, kr, 0.125f, 0.0f, 0.0f), 0, 1.0f);
            snprintf(label, sizeof label, "flash_attn_ext_rows %s alibi+softcap", ggml_type_name(kvt[t]));
            run_amp(label, ggml_flash_attn_ext_rows(ctx, q, k, v, m, kr, 0.125f, 8.0f, 30.0f), 0, 1.0f);
        }
        for (int t = 0; t < 3; t += 2) {
            struct ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 70, 4, 2);
            struct ggml_tensor * k  = ggml_new_tensor_4d(ctx, kvt[t], 64, 160, 2, 1);
            struct ggml_tensor * v  = ggml_new_tensor_4d(ctx, kvt[t], 64, 160, 2, 1);
            struct ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 150, 70, 1, 2);
            struct ggml_tensor * kr = named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 150, 2), "skip:%d", 160);
            char label[96];
            snprintf(label, sizeof label, "flash_attn_ext_rows tiled %s", ggml_type_name(kvt[t]));
            run_amp(label, ggml_flash_attn_ext_rows(ctx, q, k, v, m, kr, 0.125f, 0.0f, 0.0f), 0, 1.0f);
            {
                struct ggml_tensor * o = ggml_flash_attn_ext_rows(ctx, q, k, v, m, kr, 0.125f, 8.0f, 30.0f);
                ggml_flash_attn_ext_add_sinks(o, ggml_scale(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4), 4.0f));
                snprintf(label, sizeof label, "flash_attn_ext_rows tiled %s alibi+softcap+sinks", ggml_type_name(kvt[t]));
                run_amp(label, o, 0, 1.0f);
            }
        }
        for (int t = 0; t < 3; t += 2) {
            struct ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 4, 1);
            struct ggml_tensor * k  = ggml_new_tensor_4d(ctx, kvt[t], 64, 600, 2, 1);
            struct ggml_tensor * v  = ggml_new_tensor_4d(ctx, kvt[t], 64, 600, 2, 1);
            struct ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 520, 1, 1, 1);
            struct ggml_tensor * kr = named(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 520, 1), "skip:%d", 600);
            char label[96];
            snprintf(label, sizeof label, "flash_attn_ext_rows 1 row %s", ggml_type_name(kvt[t]));
            run_amp(label, ggml_flash_attn_ext_rows(ctx, q, k, v, m, kr, 0.125f, 0.0f, 0.0f), 0, 1.0f);
        }
    }

    /* v0.9.0: GGML_TYPE_F8_E4M3 on the CPU. Its vec_dot takes f32 rows
     * directly (vec_dot_type F32) and runs 16 lanes at a time, then a scalar
     * tail, so the row lengths are chosen off multiples of 16; at amplitude
     * 30 the inputs reach the format's coarse top binades. The conversions
     * through dup go every way the kernel allows, and add, add1 and
     * out_prod take the dequantize-quantize path with its work buffer. */
    puts("=== f8_e4m3 ===");
    {
        static const int lens[] = { 16, 37, 515 };
        static const int ncols[] = { 1, 2, 7 };
        for (int li = 0; li < 3; li++) {
            for (int ci = 0; ci < 3; ci++) {
                char label[96];
                snprintf(label, sizeof label, "mul_mat f8_e4m3 x f32 [%d x 9] x %d", lens[li], ncols[ci]);
                run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, lens[li], 9),
                                                 ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lens[li], ncols[ci])), 0, 1.0f);
            }
            char label[96];
            snprintf(label, sizeof label, "mul_mat f8_e4m3 x f32 [%d x 9] x 3 amp30", lens[li]);
            run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, lens[li], 9),
                                             ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lens[li], 3)), 0, 30.0f);
        }
        /* An f8 value has 4 significant bits and the fill's f32 values at
         * most 16, so every product above is exact in f32 and a fused and an
         * unfused multiply-add agree -- measured: an unfused tail passed.
         * Scaling src1 by 1/3 first gives it a full 24-bit significand, so
         * the products round and where the fusion sits shows. */
        for (int li = 1; li < 3; li++) {
            for (int nc = 1; nc <= 2; nc++) {
                char label[96];
                snprintf(label, sizeof label, "mul_mat f8_e4m3 x f32/3 [%d x 9] x %d", lens[li], nc);
                run_amp(label, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, lens[li], 9),
                                                 ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lens[li], nc), 1.0f / 3.0f)), 0, 1.0f);
            }
        }
        const enum ggml_type ft[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 };
        for (int t = 0; t < 3; t++) {
            char label[96];
            snprintf(label, sizeof label, "cpy %s -> f8_e4m3", ggml_type_name(ft[t]));
            run_amp(label, ggml_cpy(ctx, ggml_new_tensor_2d(ctx, ft[t], 37, 6), ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 6)), 0, 30.0f);
            snprintf(label, sizeof label, "cpy f8_e4m3 -> %s", ggml_type_name(ft[t]));
            run_amp(label, ggml_cpy(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 6), ggml_new_tensor_2d(ctx, ft[t], 37, 6)), 0, 30.0f);
            snprintf(label, sizeof label, "cpy f8_e4m3 -> %s, transposed", ggml_type_name(ft[t]));
            run_amp(label, ggml_cpy(ctx, ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 6, 37)),
                                    ggml_new_tensor_2d(ctx, ft[t], 37, 6)), 0, 30.0f);
        }
        run_amp("cpy f32 -> f8_e4m3, amp 1000", ggml_cpy(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4), ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 64, 4)), 0, 1000.0f);
        run_amp("cpy f8_e4m3 -> f8_e4m3", ggml_cpy(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 6), ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 6)), 0, 1.0f);
        RB(ggml_get_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 20), ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7)), 20);
        RB(ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 10), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 37, 3),
                         ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3)), 10);
        R(ggml_add(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 6), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 37, 6)));
        R(ggml_add1(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 6), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)));
        R(ggml_out_prod(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 37, 8), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 8)));
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
        /* New at v0.5.0: the gated form -- a per-element sigmoid gate and a
         * scale in op_params -- and hc_post's identity comb (src[3] NULL).
         * The gate runs at amplitude 30 too, where the sigmoid saturates. */
        R(ggml_dsv4_hc_pre_gated(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt), 0.75f));
        RW(ggml_dsv4_hc_pre_gated(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt), 1.0f));
        R(ggml_dsv4_hc_pre_gated(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * ne + 1, hc, nt), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * ne + 1, hc, nt), -1.5f));
        R(ggml_dsv4_hc_post(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nt), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt),
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt), NULL));
        R(ggml_dsv4_hc_post(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * ne + 1, nt), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * ne + 1, hc, nt),
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt), NULL));
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
        /* New at v0.5.0: fwht reads F16 input, and the general mul_mat
         * widens an F16 src1 for an F32 src0 -- reached under the hint only
         * with use_ref, and without the hint directly. */
        struct ggml_tensor * h = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32), ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 5));
        ggml_mul_mat_set_hint(h, GGML_HINT_SRC0_IS_HADAMARD);
        run_amp("mul_mat hint=hadamard f16 (fwht)", h, 0, 1.0f);
        struct ggml_tensor * hr = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32), ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 5));
        ggml_mul_mat_set_hint(hr, GGML_HINT_SRC0_IS_HADAMARD);
        run_use_ref = 1;
        run_amp("mul_mat hint=hadamard f16 use_ref", hr, 0, 1.0f);
        run_use_ref = 0;
        run_amp("mul_mat f32 x f16", ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 40, 7), ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 40, 6, 2)), 0, 1.0f);
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
