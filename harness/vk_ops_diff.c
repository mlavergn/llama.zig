/*
 * The Vulkan op oracle: upstream's test-backend-ops, run against the ported
 * Vulkan backend with the *reference* Vulkan backend standing where the CPU
 * normally does, compared on bits -- `cuda-ops-diff` for Vulkan. **Not a
 * port.** See scripts/vk-ops-diff.
 *
 * `test-backend-ops.cpp` is compiled with two names renamed to the hooks
 * below:
 *
 * - `ggml_backend_compare_graph_backend` -> `vk_diff_compare_graph_backend`.
 *   When backend 1 is the ported Vulkan backend, the case runs on it and on
 *   the reference Vulkan backend for the same device -- upstream's
 *   `ggml-vulkan.cpp` with its `ggml-vulkan.h` exports renamed `ref_*` --
 *   through the real comparison function. Every node the test's callback
 *   would see is compared byte for byte first; a single differing byte
 *   prints `BITDIFF` and fails the case, whatever its NMSE.
 * - `ggml_backend_supports_op` -> `vk_diff_supports_op`. Every question the
 *   test asks the ported backend is asked of the reference too, and the
 *   answers must agree (`SUPPORTS`). Before the swap the port refused ops
 *   it had no host code for and those were only counted; since the swap
 *   its `supports_op` is the C++'s predicate alone, so a refusal the
 *   reference does not make is an error too.
 *
 * Both sides run the same SPIR-V: the port embeds the very generated files
 * the reference compiles (`build/vulkan_shaders.zig`), so any difference is
 * the host code -- a pipeline choice, a specialization constant, a push
 * constant, a dispatch size, a binding.
 *
 * As in `cuda_ops_diff.c`, whole-graph cases give the reference a
 * re-expanded copy it can fuse, and a case the reference cannot reproduce
 * against itself is reported `NONDET` rather than failed. Every copy gets
 * `ggml_cast`'s self-reference back, as there: the copy drops it, and
 * TOPK_QSA's fusion counts it -- `ggml_vk_can_fuse_topk_qsa` wants a use
 * count of 2 on the `CPY` -- so without it the reference never fused the
 * case, the port did, and the reference's unfused radix `TOP_K`, which
 * disagrees with itself there, excused the difference as `NONDET`.
 * Measured on v0.4.0's `test_topk_qsa`, through `GGML_VK_PERF_LOGGER`.
 * `VK_DIFF_SELF=1`
 * puts a second reference backend where the port goes: that run must pass.
 *
 * At exit it prints the totals; scripts/vk-ops-diff turns them into the
 * gate's verdict.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

ggml_backend_t ref_ggml_backend_vk_init(size_t dev_num);

ggml_backend_reg_t ref_ggml_backend_vk_reg(void);

static long n_bitdiff_cases;
static long n_bitdiff_nodes;
static long n_compared_nodes;
static long n_supports_extra;   /* port yes, reference no: a bug */
static long n_supports_missing; /* reference yes, port no: a bug */
static long n_supports_asked;
static long n_nondet_cases;
static long n_fallback_graphs;
static long n_topk_set_nodes;   /* TOP_K rows equal as sets, order and ties aside */
static _Thread_local bool last_fell_back;

static _Thread_local ggml_backend_t ref_backends[16];
static _Thread_local ggml_backend_t ref2_backends[16];

static int device_index(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            return (int) i;
        }
    }
    fprintf(stderr, "vk_ops_diff: backend %s is not a Vulkan device\n", ggml_backend_name(backend));
    abort();
}

static ggml_backend_t ref_for(ggml_backend_t port) {
    const int i = device_index(port);
    if (ref_backends[i] == NULL) {
        ref_backends[i] = ref_ggml_backend_vk_init(i);
        if (ref_backends[i] == NULL) {
            fprintf(stderr, "vk_ops_diff: reference Vulkan backend %d failed to initialise\n", i);
            abort();
        }
    }
    return ref_backends[i];
}

static ggml_backend_t ref2_for(ggml_backend_t port) {
    const int i = device_index(port);
    if (ref2_backends[i] == NULL) {
        ref2_backends[i] = ref_ggml_backend_vk_init(i);
    }
    return ref2_backends[i];
}

bool vk_diff_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    const bool port = ggml_backend_supports_op(backend, op);
    if (!ggml_backend_is_vk(backend)) {
        return port;
    }
    const bool ref = ggml_backend_supports_op(ref_for(backend), op);
    n_supports_asked++;
    if (port && !ref) {
        n_supports_extra++;
        printf("SUPPORTS: the port claims %s (%s), the reference does not\n", ggml_op_desc(op), op->name);
    } else if (!port && ref) {
        n_supports_missing++;
        printf("SUPPORTS: the port refuses %s (%s), the reference accepts it\n", ggml_op_desc(op), op->name);
    }
    return port;
}

struct wrapped {
    ggml_backend_eval_callback callback;
    void * user_data;
    bool differs;
};

static int cmp_i32(const void * x, const void * y) {
    const int32_t a = *(const int32_t *) x, b = *(const int32_t *) y;
    return (a > b) - (a < b);
}

/* A TOP_K's output order is unspecified -- upstream's own checks compare its
 * indices as a set -- and the radix path, fused into TOPK_QSA among others,
 * orders them differently from run to run, the reference against itself
 * included. Bits first; where they differ, each row is compared again as a
 * sorted set, so the elements chosen are still held to the reference while
 * their order alone is not reported as a difference, and not excused as
 * NONDET. Sorts both buffers in place. */
static bool topk_same_set(const struct ggml_tensor * t, unsigned char * a, unsigned char * b) {
    if (t->op != GGML_OP_TOP_K || t->type != GGML_TYPE_I32 || !ggml_is_contiguous(t)) {
        return false;
    }
    const int64_t k = t->ne[0];
    const int64_t rows = ggml_nrows(t);
    for (int64_t r = 0; r < rows; r++) {
        qsort((int32_t *) a + r * k, (size_t) k, sizeof(int32_t), cmp_i32);
        qsort((int32_t *) b + r * k, (size_t) k, sizeof(int32_t), cmp_i32);
    }
    return memcmp(a, b, ggml_nbytes(t)) == 0;
}

static int cmp_f32(const void * x, const void * y) {
    const float a = *(const float *) x, b = *(const float *) y;
    return (a > b) - (a < b);
}

/* With ties the elements chosen are unspecified too -- which of several
 * equal values makes the cut changes run to run on either side -- so where
 * the index sets differ the values they select are compared, per row, as
 * upstream's own `test_top_k` does for its `ties` cases. Only where the
 * input is a leaf each side holds: a fused TOPK_QSA never materializes its
 * top-k's input, and is built without ties. `a` and `b` are the two sides'
 * outputs; `t1` and `t2` their nodes. */
static bool topk_same_values(const struct ggml_tensor * t1, const struct ggml_tensor * t2,
                             const int32_t * a, const int32_t * b) {
    const struct ggml_tensor * s1 = t1->src[0];
    const struct ggml_tensor * s2 = t2->src[0];
    if (t1->op != GGML_OP_TOP_K || !s1 || !s2 || s1->op != GGML_OP_NONE || s1->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(s1) || !ggml_is_contiguous(s2) || ggml_nbytes(s1) != ggml_nbytes(s2) ||
        ggml_nrows(s1) != ggml_nrows(t1)) {
        return false;
    }
    const int64_t k = t1->ne[0];
    const int64_t ncols = s1->ne[0];
    const int64_t rows = ggml_nrows(t1);
    float * x1 = (float *) malloc(ggml_nbytes(s1));
    float * x2 = (float *) malloc(ggml_nbytes(s2));
    float * v1 = (float *) malloc((size_t) k * sizeof(float));
    float * v2 = (float *) malloc((size_t) k * sizeof(float));
    ggml_backend_tensor_get(s1, x1, 0, ggml_nbytes(s1));
    ggml_backend_tensor_get(s2, x2, 0, ggml_nbytes(s2));
    bool same = true;
    for (int64_t r = 0; same && r < rows; r++) {
        for (int64_t j = 0; j < k; j++) {
            const int32_t ia = a[r * k + j], ib = b[r * k + j];
            if (ia < 0 || ia >= ncols || ib < 0 || ib >= ncols) {
                same = false;
                break;
            }
            v1[j] = x1[r * ncols + ia];
            v2[j] = x2[r * ncols + ib];
        }
        if (same) {
            qsort(v1, (size_t) k, sizeof(float), cmp_f32);
            qsort(v2, (size_t) k, sizeof(float), cmp_f32);
            same = memcmp(v1, v2, (size_t) k * sizeof(float)) == 0;
        }
    }
    free(x1);
    free(x2);
    free(v1);
    free(v2);
    return same;
}

static bool bit_compare(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data) {
    struct wrapped * w = (struct wrapped *) user_data;
    const size_t n = ggml_nbytes(t1);
    if (n == ggml_nbytes(t2) && n > 0) {
        unsigned char * a = (unsigned char *) malloc(n);
        unsigned char * b = (unsigned char *) malloc(n);
        ggml_backend_tensor_get(t1, a, 0, n);
        ggml_backend_tensor_get(t2, b, 0, n);
        n_compared_nodes++;
        const bool differ = memcmp(a, b, n) != 0;
        if (differ && topk_same_set(t1, a, b)) {
            printf("TOPKSET: node %d %s (%s), %zu bytes, the same indices per row in a different order\n",
                   node_index, ggml_op_desc(t1), t1->name, n);
            n_topk_set_nodes++;
        } else if (differ && topk_same_values(t1, t2, (const int32_t *) a, (const int32_t *) b)) {
            printf("TOPKSET: node %d %s (%s), %zu bytes, the same values per row from tied indices\n",
                   node_index, ggml_op_desc(t1), t1->name, n);
            n_topk_set_nodes++;
        } else if (differ) {
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

/* `graph_copy_dup_tensor` (ggml-backend.cpp:2153 @5266f24da) records a
 * node's copy only after copying its sources, so `ggml_cast`'s `src[1]`,
 * the node itself, comes back NULL in the copy. Put it back; see the file
 * comment. */
static void restore_self_refs(struct ggml_cgraph * graph, struct ggml_cgraph * copy) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor * orig = ggml_graph_node(graph, i);
        struct ggml_tensor * dup = ggml_graph_node(copy, i);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (orig->src[j] == orig && dup->src[j] == NULL) {
                dup->src[j] = dup;
            }
        }
    }
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
    restore_self_refs(graph, copy.graph);
    const int n = ggml_graph_n_nodes(graph);
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
        fprintf(stderr, "vk_ops_diff: no test node found in the graph\n");
        abort();
    }
    ggml_free(ctx);
    ggml_backend_graph_copy_free(copy);
    return true;
}

static bool compare_on(ggml_backend_t b1, ggml_backend_t b2, struct ggml_cgraph * graph,
                       ggml_backend_eval_callback callback, void * user_data,
                       struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    if (num_test_nodes != 0) {
        return compare_whole_graph(b1, b2, graph, callback, user_data, test_nodes, num_test_nodes);
    }
    return ggml_backend_compare_graph_backend(b1, b2, graph, callback, user_data, test_nodes, num_test_nodes);
}

/* The test allocated the graph's tensors in the port's buffers. A reference
 * backend in backend 1's place -- `VK_DIFF_SELF`, and the NONDET re-run --
 * cannot compute on them: it reads a buffer's context as its own, and the
 * port's has a different layout. So it gets its own copy of the graph, and
 * the test nodes are found in the copy by position. */
static bool compare(ggml_backend_t b1, bool b1_is_port, ggml_backend_t b2, struct ggml_cgraph * graph,
                    ggml_backend_eval_callback callback, void * user_data,
                    struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    /* `ggml_backend_is_vk` cannot tell: both backends carry the same GUID. */
    if (b1_is_port) {
        return compare_on(b1, b2, graph, callback, user_data, test_nodes, num_test_nodes);
    }
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(b1, graph);
    if (copy.buffer == NULL) {
        return false;
    }
    restore_self_refs(graph, copy.graph);
    struct ggml_tensor const * mapped[64];
    if (num_test_nodes > sizeof(mapped) / sizeof(mapped[0])) {
        fprintf(stderr, "vk_ops_diff: too many test nodes\n");
        abort();
    }
    for (size_t j = 0; j < num_test_nodes; j++) {
        mapped[j] = NULL;
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            if (ggml_graph_node(graph, i) == test_nodes[j]) {
                mapped[j] = ggml_graph_node(copy.graph, i);
            }
        }
    }
    /* A plain copy has no use counts, so it cannot fuse; re-expand it the
     * way `compare_whole_graph` re-expands backend 2's, or a fused port
     * would be excused as NONDET against an unfused stand-in. */
    const int n = ggml_graph_n_nodes(copy.graph);
    struct ggml_init_params params = { ggml_graph_overhead_custom(ggml_graph_size(graph), false) + 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * g1 = ggml_new_graph_custom(ctx, ggml_graph_size(graph), false);
    for (int i = 0; i < n; i++) {
        struct ggml_tensor * t = ggml_graph_node(copy.graph, i);
        if (t->op == GGML_OP_NONE && !(t->flags & GGML_TENSOR_FLAG_PARAM)) {
            ggml_graph_add_node(g1, t);
        } else {
            ggml_build_forward_expand(g1, t);
        }
    }
    bool same = ggml_graph_n_nodes(g1) == n;
    for (int i = 0; same && i < n; i++) {
        same = ggml_graph_node(g1, i) == ggml_graph_node(copy.graph, i);
    }
    const bool ok = compare_on(b1, b2, same ? g1 : copy.graph, callback, user_data, mapped, num_test_nodes);
    ggml_free(ctx);
    ggml_backend_graph_copy_free(copy);
    return ok;
}

bool vk_diff_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph,
                                     ggml_backend_eval_callback callback, void * user_data,
                                     struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    if (!ggml_backend_is_vk(backend1)) {
        return ggml_backend_compare_graph_backend(backend1, backend2, graph, callback, user_data, test_nodes, num_test_nodes);
    }
    struct wrapped w = { callback, user_data, false };
    // VK_DIFF_SELF=1 runs a second reference backend where the port would
    // be: the gate's own baseline, which must pass everywhere.
    static int self_mode = -1;
    if (self_mode < 0) {
        const char * e = getenv("VK_DIFF_SELF");
        self_mode = e != NULL && atoi(e) != 0;
    }
    ggml_backend_t first = self_mode ? ref2_for(backend1) : backend1;
    const bool ok = compare(first, !self_mode, ref_for(backend1), graph, bit_compare, &w, test_nodes, num_test_nodes);
    // A whole-graph case that fell back to the plain copy compares a fused
    // graph with an unfused one, so a self-difference there says nothing.
    if (w.differs && !(num_test_nodes != 0 && last_fell_back)) {
        struct wrapped self = { no_callback, NULL, false };
        compare(ref2_for(backend1), false, ref_for(backend1), graph, bit_compare, &self, test_nodes, num_test_nodes);
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

static void fill(float * p, size_t n, uint32_t seed) {
    uint32_t x = seed;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        p[i] = (float) (x >> 8) / (float) (1u << 24) * 2.0f - 1.0f;
    }
}

/* Inputs by type: F32 pseudo-random in [-1, 1); quantized types quantized
 * from such data, so every block is valid; I32 is an expert-id table whose
 * name is `ids:<n_experts>`, each row a distinct rotation; I64 is a row
 * index list, reversed. */
static void fill_tensor(struct ggml_tensor * t, uint32_t seed) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        int n_experts = 0;
        if (sscanf(t->name, "ids:%d", &n_experts) != 1) {
            fprintf(stderr, "vk_ops_diff: I32 input %s has no expert count\n", t->name);
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
    if (t->type == GGML_TYPE_I64) {
        /* Row indices: a distinct permutation of ne[0] rows, reversed. */
        int64_t * rows = (int64_t *) malloc(ggml_nbytes(t));
        for (int64_t j = 0; j < n; j++) {
            rows[j] = n - 1 - j;
        }
        ggml_backend_tensor_set(t, rows, 0, ggml_nbytes(t));
        free(rows);
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
    ggml_backend_t port = ggml_backend_vk_init(0);
    ggml_backend_t ref  = ref_ggml_backend_vk_init(0);
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

/* rope (optionally after rms_norm * w) -> view -> set_rows, into an F16
 * leaf placed 3 elements into a cache. Upstream's ROPE_SET_ROWS and
 * RMS_NORM_MUL_ROPE always write a fresh tensor at offset 0, so the fused
 * dispatch's misaligned-destination arm (the rounded-down binding and
 * `d_offset`) never ran: zeroing `d_offset` passed all 144
 * RMS_NORM_MUL_ROPE cases. The destination must be a *leaf*: a VIEW node
 * there overlaps set_rows' output, and the overlap check turns the fusion
 * off on both sides. ggml-alloc never places a leaf off alignment, so no
 * model graph reaches this arm; a hand-placed tensor does, and the C++
 * handles it. The leaf is op NONE with a `view_src`, which
 * `ggml_backend_alloc_ctx_tensors` places through `ggml_backend_view_init`. */
static struct ggml_tensor * build_rope_set_rows_misaligned(struct ggml_context * ctx, struct ggml_tensor ** cmp, bool rms) {
    const int64_t hd = 64, nh = 4, nt = 8;
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, nt);
    if (rms) {
        struct ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        a = ggml_mul(ctx, ggml_rms_norm(ctx, a, 1e-6f), w);
    }
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
    ggml_set_name(pos, "ids:100");
    struct ggml_tensor * rope = ggml_rope(ctx, a, pos, (int) hd, GGML_ROPE_TYPE_NEOX);
    struct ggml_tensor * view = ggml_view_2d(ctx, rope, hd * nh, nt, rope->nb[2], 0);
    struct ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd * nh, nt + 2);
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd * nh, nt);
    dst->view_src = cache;
    dst->view_offs = 3 * ggml_type_size(GGML_TYPE_F16);
    struct ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
    *cmp = cache;
    return ggml_set_rows(ctx, dst, view, rows);
}
static struct ggml_tensor * build_rope_sr_mis(struct ggml_context * c, struct ggml_tensor ** t) { return build_rope_set_rows_misaligned(c, t, false); }
static struct ggml_tensor * build_rms_rope_sr_mis(struct ggml_context * c, struct ggml_tensor ** t) { return build_rope_set_rows_misaligned(c, t, true); }

/* Matrix-matrix shapes upstream's MUL_MAT list never reaches. Its cases
 * keep K small, so `ggml_vk_guess_split_k` (K >= 2048 and few enough tiles
 * to leave the GPU half empty) never split and its rounding passed when
 * changed; and its M and N stay small, so coopmat2's large-tile choice
 * (tiles that would overfill the GPU) was never made. */
static struct ggml_tensor * build_mm(struct ggml_context * ctx, struct ggml_tensor ** cmp, enum ggml_type type, int64_t m, int64_t n, int64_t k) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    struct ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    *cmp = out;
    return out;
}
static struct ggml_tensor * build_mm_split_k_f16(struct ggml_context * c, struct ggml_tensor ** t) { return build_mm(c, t, GGML_TYPE_F16, 128, 128, 4096); }
static struct ggml_tensor * build_mm_split_k_q4_0(struct ggml_context * c, struct ggml_tensor ** t) { return build_mm(c, t, GGML_TYPE_Q4_0, 128, 96, 8192); }
static struct ggml_tensor * build_mm_large_tiles(struct ggml_context * c, struct ggml_tensor ** t) { return build_mm(c, t, GGML_TYPE_F16, 1024, 1024, 256); }

/* MUL_MAT + ADD + ADD with the second bias in src[0]. The vector shader's
 * C++ takes the second bias from whichever operand is not the first ADD;
 * upstream's MUL_MAT_VEC_FUSION always puts it in src[1], so taking src[1]
 * unconditionally passed all 192 cases. */
static struct ggml_tensor * build_mm_bias_swapped(struct ggml_context * ctx, struct ggml_tensor ** cmp) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 64);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    struct ggml_tensor * b0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1);
    struct ggml_tensor * b1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1);
    struct ggml_tensor * add1 = ggml_add(ctx, ggml_mul_mat(ctx, a, b), b0);
    struct ggml_tensor * out = ggml_add(ctx, b1, add1);
    *cmp = out;
    return out;
}

/* Shapes and patterns upstream's list lacks; each later layer of the port
 * adds the cases its host code needs. */
static int extra_cases(const char * filter) {
    static const struct { const char * op; const char * name; extra_build build; } cases[] = {
        { "ROPE_SET_ROWS",     "rope+set_rows into a misaligned leaf",          build_rope_sr_mis },
        { "RMS_NORM_MUL_ROPE", "rms_norm*w+rope+set_rows into a misaligned leaf", build_rms_rope_sr_mis },
        { "MUL_MAT",           "f16 128x128x4096, split_k",                         build_mm_split_k_f16 },
        { "MUL_MAT",           "q4_0 128x96x8192, split_k",                         build_mm_split_k_q4_0 },
        { "MUL_MAT",           "f16 1024x1024x256, large tiles",                    build_mm_large_tiles },
        { "MUL_MAT_VEC_FUSION", "mul_mat+add+add, second bias in src[0]",           build_mm_bias_swapped },
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (cases[i].op == NULL) {
            continue;
        }
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
    printf("vk_ops_diff: %ld nodes compared on bits, %ld differing in %ld cases, %ld reference-nondeterministic cases\n",
           n_compared_nodes, n_bitdiff_nodes, n_bitdiff_cases, n_nondet_cases);
    printf("vk_ops_diff: %ld whole-graph cases compared on the plain copy\n", n_fallback_graphs);
    printf("vk_ops_diff: %ld TOP_K nodes equal as index or value sets, order and ties aside\n", n_topk_set_nodes);
    printf("vk_ops_diff: supports_op asked %ld times: %ld claimed beyond the reference, %ld refused that it accepts\n",
           n_supports_asked, n_supports_extra, n_supports_missing);
}

__attribute__((constructor)) static void install_report(void) {
    atexit(report);
}
