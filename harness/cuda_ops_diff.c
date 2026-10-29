/*
 * The CUDA op oracle: upstream's test-backend-ops, run against the ported
 * CUDA backend with the *reference* CUDA backend standing where the CPU
 * normally does, compared on bits. **Not a port.** See scripts/cuda-ops-diff.
 *
 * `test-backend-ops.cpp` is compiled with two names renamed to the hooks
 * below:
 *
 * - `ggml_backend_compare_graph_backend` -> `cuda_diff_compare_graph_backend`.
 *   When backend 1 is the ported CUDA backend, the case runs on it and on
 *   the reference CUDA backend for the same device -- upstream's objects
 *   with their `ggml-cuda.h` exports renamed `ref_*` -- through the real
 *   comparison function. Every node the test's callback would see is
 *   compared byte for byte first; a single differing byte prints `BITDIFF`
 *   and fails the case, whatever its NMSE.
 * - `ggml_backend_supports_op` -> `cuda_diff_supports_op`. Every question
 *   the test asks the ported backend is asked of the reference too. The
 *   port claiming an op the reference refuses is an error (`SUPPORTS`);
 *   the reverse is an op whose host code is not ported yet, and is counted.
 *
 * Two departures from a straight `ggml_backend_compare_graph_backend`:
 *
 * - **Whole-graph cases give the reference a graph it can fuse.**
 *   `ggml_backend_graph_copy` (ggml-backend.cpp:2207 @b29c606e2) fills the
 *   copy's `nodes[]` directly and never builds its use counts, so the second
 *   backend of a comparison can never fuse -- harmless upstream, where that
 *   backend is the CPU, and fatal here, where it would compare the port's
 *   fused kernels against the reference's unfused ones. Measured: every fusion case
 *   "failed" until the port's fusion was switched off. So for a case with
 *   test nodes the copy is re-expanded node by node, in the original order,
 *   into a fresh graph: same nodes, same order, real use counts.
 * - **The copy keeps `ggml_cast`'s self-reference.** `ggml_cast` builds a
 *   `CPY` whose `src[1]` is the node itself, and `graph_copy_dup_tensor`
 *   (ggml-backend.cpp:2146 @b29c606e2) records a node's copy only after
 *   copying its sources, so the self-reference comes back as
 *   the still-empty slot: the copy's `src[1]` is NULL. Upstream's CPU never
 *   reads it; the reference CUDA backend dispatches `CPY` as
 *   `ggml_cuda_cpy(ctx, src[0], src[1])` and faults. Measured on v0.4.0's
 *   `test_topk_qsa`, the first whole-graph case with a cast. The copy's
 *   self-references are restored before it runs.
 * - **A difference the reference makes against itself is not the port's.**
 *   A case that differs is re-run reference against reference; if the
 *   reference disagrees with itself the case is reported `NONDET` and not
 *   failed. Two sources, both measured: CUB's float
 *   `DeviceScan::InclusiveSum`, which `CUMSUM` uses on long rows (43 of 49
 *   runs differ at 375960 elements), and the fused MMVQ for `MUL_MAT_ID`
 *   with lane scale, `ADD_ID` bias and a gate on NVFP4 -- unfused the
 *   reference agrees with itself, fused it does not, and where it differs
 *   moves from run to run.
 *
 * And `main` is ours: `test-backend-ops.cpp` is compiled with `main` renamed,
 * and the harness first runs `extra_cases` -- shapes the gate needs and
 * upstream's list does not have -- on fresh backends, then hands over.
 *
 * **Every `cudaMalloc` gets 1 MiB of slack.** Upstream's
 * `flash_attn_mask_to_KV_max` reads up to `ncols1 - 1` mask rows past the
 * mask's end when `Q->ne[1] % ncols1 != 0` (measured: up to 126 KiB at the
 * test shapes). Where the graph copy puts the mask last in its buffer, the
 * reference faults with an illegal address -- 62 of 432 such cases, varying
 * with what happens to be mapped. The rows it reads only feed rows nobody
 * keeps, so padding every allocation, on both sides alike, changes no
 * result. `cudaMalloc` is interposed below; the real one comes from
 * `dlsym(RTLD_NEXT)`.
 *
 * `CUDA_DIFF_SELF=1` puts a second reference backend where the port goes,
 * so the comparison machinery itself can be checked: that run must pass.
 *
 * At exit it prints the totals; scripts/cuda-ops-diff turns them into the
 * gate's verdict.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

ggml_backend_t ref_ggml_backend_cuda_init(int device);

/* See the file comment. `cudaError_t` is an `int`-sized enum. */
int cudaMalloc(void ** ptr, size_t size) {
    static int (*real)(void **, size_t) = NULL;
    if (real == NULL) {
        real = (int (*)(void **, size_t)) dlsym(RTLD_NEXT, "cudaMalloc");
        if (real == NULL) {
            fprintf(stderr, "cuda_ops_diff: cannot find the real cudaMalloc\n");
            abort();
        }
    }
    return real(ptr, size + (1u << 20));
}
ggml_backend_reg_t ref_ggml_backend_cuda_reg(void);

static long n_bitdiff_cases;
static long n_bitdiff_nodes;
static long n_compared_nodes;
static long n_supports_extra;   /* port yes, reference no: a bug */
static long n_supports_missing; /* reference yes, port no: not ported yet */
static long n_supports_asked;
static long n_nondet_cases;
static long n_fallback_graphs;
static _Thread_local bool last_fell_back;

static _Thread_local ggml_backend_t ref_backends[16];
static _Thread_local ggml_backend_t ref2_backends[16];

static int device_index(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_cuda_reg();
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            return (int) i;
        }
    }
    fprintf(stderr, "cuda_ops_diff: backend %s is not a CUDA device\n", ggml_backend_name(backend));
    abort();
}

static ggml_backend_t ref_for(ggml_backend_t port) {
    const int i = device_index(port);
    if (ref_backends[i] == NULL) {
        ref_backends[i] = ref_ggml_backend_cuda_init(i);
        if (ref_backends[i] == NULL) {
            fprintf(stderr, "cuda_ops_diff: reference CUDA backend %d failed to initialise\n", i);
            abort();
        }
    }
    return ref_backends[i];
}

static ggml_backend_t ref2_for(ggml_backend_t port) {
    const int i = device_index(port);
    if (ref2_backends[i] == NULL) {
        ref2_backends[i] = ref_ggml_backend_cuda_init(i);
    }
    return ref2_backends[i];
}

bool cuda_diff_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    const bool port = ggml_backend_supports_op(backend, op);
    if (!ggml_backend_is_cuda(backend)) {
        return port;
    }
    const bool ref = ggml_backend_supports_op(ref_for(backend), op);
    n_supports_asked++;
    if (port && !ref) {
        n_supports_extra++;
        printf("SUPPORTS: the port claims %s (%s), the reference does not\n", ggml_op_desc(op), op->name);
    } else if (!port && ref) {
        n_supports_missing++;
        printf("UNPORTED: %s\n", ggml_op_desc(op));
    }
    return port;
}

struct wrapped {
    ggml_backend_eval_callback callback;
    void * user_data;
    bool differs;
};

static bool bit_compare(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data) {
    struct wrapped * w = (struct wrapped *) user_data;
    const size_t n = ggml_nbytes(t1);
    if (n == ggml_nbytes(t2) && n > 0) {
        unsigned char * a = (unsigned char *) malloc(n);
        unsigned char * b = (unsigned char *) malloc(n);
        ggml_backend_tensor_get(t1, a, 0, n);
        ggml_backend_tensor_get(t2, b, 0, n);
        n_compared_nodes++;
        if (memcmp(a, b, n) != 0) {
            size_t first = 0;
            while (a[first] == b[first]) {
                first++;
            }
            printf("BITDIFF: node %d %s (%s), %zu bytes, first difference at byte %zu\n",
                   node_index, ggml_op_desc(t1), t1->name, n, first);
            n_bitdiff_nodes++;
            w->differs = true;
        }
        free(a);
        free(b);
    } else if (ggml_nbytes(t1) != ggml_nbytes(t2)) {
        printf("BITDIFF: node %d %s sizes %zu vs %zu\n", node_index, ggml_op_desc(t1), ggml_nbytes(t1), ggml_nbytes(t2));
        n_bitdiff_nodes++;
        w->differs = true;
    }
    return w->callback(node_index, t1, t2, w->user_data);
}

static bool no_callback(int i, struct ggml_tensor * t1, struct ggml_tensor * t2, void * ud) {
    (void) i; (void) t1; (void) t2; (void) ud;
    return true;
}

/* `ggml_backend_compare_graph_backend`'s whole-graph arm, with backend 2's
 * copy re-expanded so it carries use counts; see the file comment. */
static bool compare_whole_graph(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph,
                                ggml_backend_eval_callback callback, void * user_data,
                                struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }
    const int n = ggml_graph_n_nodes(graph);
    // Restore the self-references the copy dropped; see the file comment.
    for (int i = 0; i < n; i++) {
        struct ggml_tensor * orig = ggml_graph_node(graph, i);
        struct ggml_tensor * dup = ggml_graph_node(copy.graph, i);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (orig->src[j] == orig && dup->src[j] == NULL) {
                dup->src[j] = dup;
            }
        }
    }
    struct ggml_init_params params = { ggml_graph_overhead_custom(ggml_graph_size(graph), false) + 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * g2 = ggml_new_graph_custom(ctx, ggml_graph_size(graph), false);
    for (int i = 0; i < n; i++) {
        struct ggml_tensor * t = ggml_graph_node(copy.graph, i);
        // test-backend-ops appends sentinel tensors as op-NONE nodes, which
        // expanding would file as leafs; append them as the original did.
        // An op-NONE *parameter* is a node to the expansion already.
        if (t->op == GGML_OP_NONE && !(t->flags & GGML_TENSOR_FLAG_PARAM)) {
            ggml_graph_add_node(g2, t);
        } else {
            ggml_build_forward_expand(g2, t);
        }
    }
    // Use the re-expanded graph only where it reproduces the original node
    // for node; where expanding would pull in a tensor the original never
    // listed, fall back to the plain copy (which the reference cannot fuse,
    // so a fused port would show as a BITDIFF and be looked at).
    bool same = ggml_graph_n_nodes(g2) == n;
    for (int i = 0; same && i < n; i++) {
        same = ggml_graph_node(g2, i) == ggml_graph_node(copy.graph, i);
    }
    last_fell_back = !same;
    if (!same) {
        g2 = copy.graph;
        n_fallback_graphs++;
        printf("FALLBACK: whole-graph case compared on the plain copy\n");
    }

    ggml_backend_graph_compute(backend1, graph);
    ggml_backend_graph_compute(backend2, g2);

    bool verified = false;
    for (int i = 0; i < n; i++) {
        for (size_t j = 0; j < num_test_nodes; j++) {
            if (ggml_graph_node(graph, i) == test_nodes[j]) {
                callback(i, ggml_graph_node(graph, i), ggml_graph_node(g2, i), user_data);
                verified = true;
            }
        }
    }
    if (!verified) {
        fprintf(stderr, "cuda_ops_diff: no test node found in the graph\n");
        abort();
    }
    ggml_free(ctx);
    ggml_backend_graph_copy_free(copy);
    return true;
}

static bool compare(ggml_backend_t b1, ggml_backend_t b2, struct ggml_cgraph * graph,
                    ggml_backend_eval_callback callback, void * user_data,
                    struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    if (num_test_nodes != 0) {
        return compare_whole_graph(b1, b2, graph, callback, user_data, test_nodes, num_test_nodes);
    }
    return ggml_backend_compare_graph_backend(b1, b2, graph, callback, user_data, test_nodes, num_test_nodes);
}

bool cuda_diff_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph,
                                     ggml_backend_eval_callback callback, void * user_data,
                                     struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    if (!ggml_backend_is_cuda(backend1)) {
        return ggml_backend_compare_graph_backend(backend1, backend2, graph, callback, user_data, test_nodes, num_test_nodes);
    }
    struct wrapped w = { callback, user_data, false };
    // CUDA_DIFF_SELF=1 runs a second reference backend where the port would
    // be: the gate's own baseline, which must pass everywhere.
    static int self_mode = -1;
    if (self_mode < 0) {
        const char * e = getenv("CUDA_DIFF_SELF");
        self_mode = e != NULL && atoi(e) != 0;
    }
    ggml_backend_t first = self_mode ? ref2_for(backend1) : backend1;
    const bool ok = compare(first, ref_for(backend1), graph, bit_compare, &w, test_nodes, num_test_nodes);
    // A whole-graph case that fell back to the plain copy compares a fused
    // graph with an unfused one, so a self-difference there says nothing.
    if (w.differs && !(num_test_nodes != 0 && last_fell_back)) {
        struct wrapped self = { no_callback, NULL, false };
        compare(ref2_for(backend1), ref_for(backend1), graph, bit_compare, &self, test_nodes, num_test_nodes);
        if (self.differs) {
            n_nondet_cases++;
            printf("NONDET: the reference differs from itself on this case; not counted against the port\n");
            return ok;
        }
    }
    if (w.differs) {
        n_bitdiff_cases++;
    }
    return ok && !w.differs;
}

/* Upstream's `main`, renamed by `-Dmain=test_backend_ops_main`; C++ linkage. */
int test_backend_ops_main(int argc, char ** argv) __asm__("_Z21test_backend_ops_mainiPPc");

/* One extra case. `build` creates its inputs in `ctx` (every tensor it
 * creates with `ggml_new_tensor*` is an input, filled with the same
 * pseudo-random data on both sides), returns the tensor to expand, and
 * sets `*cmp` to the tensor to compare -- the output, or a cache the graph
 * writes into. */
typedef struct ggml_tensor * (*extra_build)(struct ggml_context * ctx, struct ggml_tensor ** cmp);

static struct ggml_tensor * build_mean_n(struct ggml_context * ctx, struct ggml_tensor ** cmp, int64_t ne0) {
    struct ggml_tensor * in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    return *cmp = ggml_mean(ctx, in);
}
static struct ggml_tensor * build_mean_65536(struct ggml_context * c, struct ggml_tensor ** t) { return build_mean_n(c, t, 65536); }
static struct ggml_tensor * build_mean_65537(struct ggml_context * c, struct ggml_tensor ** t) { return build_mean_n(c, t, 65537); }
static struct ggml_tensor * build_mean_262144(struct ggml_context * c, struct ggml_tensor ** t) { return build_mean_n(c, t, 262144); }

/* gated_delta_net with K = 3 snapshot slots over 4 tokens, its snapshot
 * tail copied into a recurrent cache through a strided view: the pattern
 * `ggml_cuda_try_gdn_cache_fusion` matches. The cache has a spare sequence
 * row, so its slot stride is not the contiguous one. */
static struct ggml_tensor * build_gdn_cache(struct ggml_context * ctx, struct ggml_tensor ** cmp) {
    const int64_t S = 32, H = 2, T = 4, NS = 2, K = 3;
    const int64_t D = S * S * H;
    struct ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, NS);
    struct ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, NS);
    struct ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, NS);
    struct ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, NS);
    struct ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, NS);
    struct ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, NS);
    struct ggml_tensor * cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NS + 1, K + 1);
    struct ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K);
    const size_t tail = ggml_row_size(GGML_TYPE_F32, S * H * T * NS);
    struct ggml_tensor * snap = ggml_view_3d(ctx, out, D, NS, K, D * sizeof(float), D * NS * sizeof(float), tail);
    struct ggml_tensor * dst  = ggml_view_3d(ctx, cache, D, NS, K, cache->nb[1], cache->nb[2], 0);
    *cmp = cache;
    return ggml_cpy(ctx, snap, dst);
}

static void fill(float * p, size_t n, uint32_t seed) {
    uint32_t x = seed;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        p[i] = (float) (x >> 8) / (float) (1u << 24) * 2.0f - 1.0f;
    }
}

/* Inputs by type: F32 pseudo-random in [-1, 1); quantized types quantized
 * from such data, so every block is valid; I32 is an expert-id table whose
 * name is `ids:<n_experts>`, each row a distinct rotation. */
static void fill_tensor(struct ggml_tensor * t, uint32_t seed) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        int n_experts = 0;
        if (sscanf(t->name, "ids:%d", &n_experts) != 1) {
            fprintf(stderr, "cuda_ops_diff: I32 input %s has no expert count\n", t->name);
            abort();
        }
        int32_t * ids = (int32_t *) malloc(ggml_nbytes(t));
        for (int64_t r = 0; r < t->ne[1]; r++) {
            for (int64_t j = 0; j < t->ne[0]; j++) {
                ids[r * t->ne[0] + j] = (int32_t) ((r * 3 + j) % n_experts);
            }
        }
        ggml_backend_tensor_set(t, ids, 0, ggml_nbytes(t));
        free(ids);
        return;
    }
    float * data = (float *) malloc(n * sizeof(float));
    fill(data, n, seed);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    } else {
        void * q = malloc(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, data, q, 0, n / t->ne[0], t->ne[0], NULL);
        ggml_backend_tensor_set(t, q, 0, ggml_nbytes(t));
        free(q);
    }
    free(data);
}

/* MUL_MAT_ID over 8 experts, 2 used per token, k = 256, at `n_tokens`. */
static struct ggml_tensor * build_mmid(struct ggml_context * ctx, struct ggml_tensor ** cmp, enum ggml_type type, int64_t n_tokens) {
    const int64_t n_mats = 8, n_used = 2, m = 64, k = 256;
    struct ggml_tensor * as  = ggml_new_tensor_3d(ctx, type, k, m, n_mats);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_format_name(ids, "ids:%d", (int) n_mats);
    struct ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    return *cmp = ggml_mul_mat_id(ctx, as, b, ids);
}
static struct ggml_tensor * build_mmid_q4_0_8(struct ggml_context * c, struct ggml_tensor ** t) { return build_mmid(c, t, GGML_TYPE_Q4_0, 8); }
static struct ggml_tensor * build_mmid_q4_0_9(struct ggml_context * c, struct ggml_tensor ** t) { return build_mmid(c, t, GGML_TYPE_Q4_0, 9); }
static struct ggml_tensor * build_mmid_q4_k_8(struct ggml_context * c, struct ggml_tensor ** t) { return build_mmid(c, t, GGML_TYPE_Q4_K, 8); }
static struct ggml_tensor * build_mmid_q4_k_9(struct ggml_context * c, struct ggml_tensor ** t) { return build_mmid(c, t, GGML_TYPE_Q4_K, 9); }

static void run_extra(ggml_backend_t backend, extra_build build, unsigned char ** result, size_t * nbytes, int * n_nodes) {
    struct ggml_init_params params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * cmp = NULL;
    struct ggml_tensor * root = build(ctx, &cmp);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, root);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    uint32_t seed = 12345;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op == GGML_OP_NONE && t->view_src == NULL) {
            fill_tensor(t, seed++);
        }
    }
    ggml_backend_graph_compute(backend, gf);
    *nbytes = ggml_nbytes(cmp);
    *result = (unsigned char *) malloc(*nbytes);
    ggml_backend_tensor_get(cmp, *result, 0, *nbytes);
    *n_nodes = ggml_graph_n_nodes(gf);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

/* Runs `build` on fresh port and reference backends and compares on bits.
 * Return: whether it matched. */
static bool extra_case(const char * name, extra_build build) {
    ggml_backend_t port = ggml_backend_cuda_init(0);
    ggml_backend_t ref  = ref_ggml_backend_cuda_init(0);
    unsigned char * a, * b;
    size_t na, nb;
    int nodes;
    run_extra(port, build, &a, &na, &nodes);
    run_extra(ref,  build, &b, &nb, &nodes);
    n_compared_nodes++;
    bool differs = na != nb || memcmp(a, b, na) != 0;
    if (differs && na == nb) {
        size_t first = 0;
        while (a[first] == b[first]) {
            first++;
        }
        printf("BITDIFF: extra %s, %zu bytes, first difference at byte %zu\n", name, na, first);
        n_bitdiff_nodes++;
    }
    printf("  EXTRA %s (%d nodes): %s\n", name, nodes, differs ? "FAIL" : "OK");
    if (differs) {
        n_bitdiff_cases++;
    }
    free(a);
    free(b);
    ggml_backend_free(port);
    ggml_backend_free(ref);
    return !differs;
}

/* Shapes and patterns upstream's list lacks. `MEAN` sends one row longer
 * than 65536 to `cub::DeviceReduce::Sum` when no CUDA graph exists yet --
 * which a fresh backend guarantees -- and upstream's longest single row is
 * 32769. No upstream case builds the gated-delta-net snapshot copy that
 * `ggml_cuda_try_gdn_cache_fusion` fuses. And upstream's test-mode
 * `MUL_MAT_ID` batches are 1, 4, 5, 17, 32 and 129, so none sits on the
 * MMVQ/MMQ boundary at 8 (`get_mmvq_mmid_max_batch`). */
static int extra_cases(const char * filter) {
    static const struct { const char * op; const char * name; extra_build build; } cases[] = {
        { "MEAN", "MEAN(ne=[65536,1,1,1])", build_mean_65536 },
        { "MEAN", "MEAN(ne=[65537,1,1,1])", build_mean_65537 },
        { "MEAN", "MEAN(ne=[262144,1,1,1])", build_mean_262144 },
        { "GATED_DELTA_NET", "GATED_DELTA_NET+CPY(K=3,n_tokens=4,strided cache)", build_gdn_cache },
        { "MUL_MAT_ID", "MUL_MAT_ID(q4_0,n_tokens=8)", build_mmid_q4_0_8 },
        { "MUL_MAT_ID", "MUL_MAT_ID(q4_0,n_tokens=9)", build_mmid_q4_0_9 },
        { "MUL_MAT_ID", "MUL_MAT_ID(q4_K,n_tokens=8)", build_mmid_q4_k_8 },
        { "MUL_MAT_ID", "MUL_MAT_ID(q4_K,n_tokens=9)", build_mmid_q4_k_9 },
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (filter == NULL || strcmp(filter, cases[i].op) == 0) {
            failed += !extra_case(cases[i].name, cases[i].build);
        }
    }
    return failed;
}

int main(int argc, char ** argv) {
    const char * filter = NULL;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "-o") == 0) {
            filter = argv[i + 1];
        }
    }
    extra_cases(filter);
    return test_backend_ops_main(argc, argv);
}

static void report(void) {
    printf("cuda_ops_diff: %ld nodes compared on bits, %ld differing in %ld cases, %ld reference-nondeterministic cases\n",
           n_compared_nodes, n_bitdiff_nodes, n_bitdiff_cases, n_nondet_cases);
    printf("cuda_ops_diff: %ld whole-graph cases compared on the plain copy\n", n_fallback_graphs);
    printf("cuda_ops_diff: supports_op asked %ld times: %ld claimed beyond the reference, %ld not ported yet\n",
           n_supports_asked, n_supports_extra, n_supports_missing);
}

__attribute__((constructor)) static void install_report(void) {
    atexit(report);
}
