/*
 * The CUDA backend's vtables, asked the same questions on the port and on the
 * reference: `metal-diff` for CUDA. **Not a port.** See
 * scripts/cuda-backend-diff.
 *
 * Both registries live in one process -- the port's through
 * `ggml_backend_cuda_reg`, the reference's through `ref_ggml_backend_cuda_reg`
 * (upstream's objects with their `ggml-cuda.h` exports renamed, in
 * `libggml-cuda-ref.a`) -- and every answer is printed once per side as
 * `<side> <question> = <answer>`. The script strips the side and diffs.
 *
 * Why `cuda-ops-diff` and `cuda-node-diff` are not enough: they run graphs,
 * so they see only the answers a graph happens to need. `get_alloc_size` is
 * the clearest case. A buffer sized too small by the port shows nowhere
 * until something overruns it, and `metal-diff` measured exactly that on
 * Metal -- a 162x under-allocation for flash-attention scratch that
 * `node-diff` passed. Here:
 *
 * - the registry: name, device count, every `get_proc_address` name
 *   upstream answers plus one it does not, and the feature list;
 * - per device: name, description, type, props (memory total -- free memory
 *   moves under any run, so it is not compared -- device id and caps), both
 *   buffer types' name, alignment, max size and is-host, and `supports_buft`
 *   against its own, the host and the CPU buffer type;
 * - `offload_op` on both sides of `op_offload_min_batch_size` (32) for every
 *   arm of `get_op_batch_size`;
 * - `get_alloc_size` for every type at row lengths on both sides of the
 *   512-element `MATRIX_ROW_PADDING`, and for flash-attention nodes --
 *   whose size depends on `dst->data`, so each is asked at several
 *   addresses;
 * - `graph_optimize`'s allocation dependencies: every `add_alloc_dep` call
 *   it makes on top-k MoE gating chains (each gating function, with and
 *   without the bias, the norm and the scale, followed by a consumer or
 *   ending the graph) and on MoE weighted reductions. They only keep
 *   tensors allocated longer, so a missing one shows in no result until the
 *   allocator happens to reuse the memory -- and `test-backend-ops` never
 *   calls `graph_optimize` at all.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

ggml_backend_reg_t ref_ggml_backend_cuda_reg(void);

static const char * side;

#define SAY(...) do { printf("%s ", side); printf(__VA_ARGS__); printf("\n"); } while (0)

static void ask_reg(ggml_backend_reg_t reg) {
    SAY("reg name = %s", ggml_backend_reg_name(reg));
    SAY("reg dev_count = %zu", ggml_backend_reg_dev_count(reg));
    static const char * procs[] = {
        "ggml_backend_comm_init", "ggml_backend_comm_free", "ggml_backend_comm_allreduce_tensor",
        "ggml_backend_register_host_buffer", "ggml_backend_unregister_host_buffer",
        "ggml_backend_get_features", "ggml_backend_split_buffer_type", "no_such_function",
    };
    for (size_t i = 0; i < sizeof(procs) / sizeof(procs[0]); i++) {
        SAY("reg proc %s = %s", procs[i], ggml_backend_reg_get_proc_address(reg, procs[i]) ? "yes" : "null");
    }
    ggml_backend_get_features_t get_features =
        (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (get_features) {
        for (struct ggml_backend_feature * f = get_features(reg); f->name; f++) {
            SAY("reg feature %s = %s", f->name, f->value);
        }
    }
}

static void ask_buft(const char * what, ggml_backend_buffer_type_t buft) {
    SAY("%s name = %s", what, ggml_backend_buft_name(buft));
    SAY("%s alignment = %zu", what, ggml_backend_buft_get_alignment(buft));
    SAY("%s max_size = %zu", what, ggml_backend_buft_get_max_size(buft));
    SAY("%s is_host = %d", what, ggml_backend_buft_is_host(buft));
}

/* get_alloc_size for every type, at row lengths around the padding. */
static void ask_alloc_types(ggml_backend_buffer_type_t buft, struct ggml_context * ctx) {
    static const int64_t blocks[] = { 1, 2, 3, 7, 15, 16, 17, 33 };
    static const int64_t rows[] = { 1, 5 };
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const int64_t blck = ggml_blck_size((enum ggml_type) t);
        if (blck == 0 || ggml_type_size((enum ggml_type) t) == 0) {
            continue; // a removed type
        }
        int64_t ne0s[16];
        int n = 0;
        for (size_t i = 0; i < sizeof(blocks) / sizeof(blocks[0]); i++) {
            ne0s[n++] = blck * blocks[i];
        }
        for (int64_t e = 511; e <= 1537; e += 513) { // 511, 1024, 1537 rounded to a block
            ne0s[n++] = (e / blck + 1) * blck;
        }
        for (int i = 0; i < n; i++) {
            for (size_t r = 0; r < sizeof(rows) / sizeof(rows[0]); r++) {
                struct ggml_tensor * x = ggml_new_tensor_2d(ctx, (enum ggml_type) t, ne0s[i], rows[r]);
                SAY("alloc %s [%lld,%lld] = %zu", ggml_type_name((enum ggml_type) t),
                    (long long) ne0s[i], (long long) rows[r], ggml_backend_buft_get_alloc_size(buft, x));
            }
        }
    }
}

/* get_alloc_size for flash-attention nodes, each at several `data`
 * addresses: the scratch is padded from the real address. */
static void ask_alloc_fattn(ggml_backend_buffer_type_t buft, struct ggml_context * ctx) {
    static const struct { int64_t dk, dv; } heads[] = { { 64, 64 }, { 128, 128 }, { 256, 256 }, { 576, 512 }, { 40, 40 } };
    static const int64_t n_qs[] = { 1, 4, 32, 129 };
    static const int64_t n_kvs[] = { 256, 1024 };
    /* The last three pairs are outside upstream's default FA quants, so a
     * `-Dcuda-fa-quants` build without them takes the vec kernel's
     * f16_fallback; with only matched compiled pairs it was never asked. */
    static const struct { enum ggml_type k, v; } kv_types[] = {
        { GGML_TYPE_F16, GGML_TYPE_F16 }, { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 }, { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0 },
        { GGML_TYPE_Q5_1, GGML_TYPE_Q5_1 }, { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }, { GGML_TYPE_F16, GGML_TYPE_Q8_0 },
    };
    static const uintptr_t bases[] = { 0x10000000, 0x10000040, 0x10000080, 0x100000c4, 0x10000f00 };
    for (size_t h = 0; h < sizeof(heads) / sizeof(heads[0]); h++) {
        for (size_t iq = 0; iq < sizeof(n_qs) / sizeof(n_qs[0]); iq++) {
            for (size_t ik = 0; ik < sizeof(n_kvs) / sizeof(n_kvs[0]); ik++) {
                for (size_t it = 0; it < sizeof(kv_types) / sizeof(kv_types[0]); it++) {
                    for (int gqa = 1; gqa <= 4; gqa *= 4) {
                        const int64_t H = 8 * gqa, Hkv = 8, B = 1;
                        const int64_t n_q = n_qs[iq], n_kv = n_kvs[ik];
                        struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, heads[h].dk, n_q, H, B);
                        struct ggml_tensor * k = ggml_new_tensor_4d(ctx, kv_types[it].k, heads[h].dk, n_kv, Hkv, B);
                        struct ggml_tensor * v = ggml_new_tensor_4d(ctx, kv_types[it].v, heads[h].dv, n_kv, Hkv, B);
                        struct ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, 1);
                        struct ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 0.0f);
                        for (size_t b = 0; b < sizeof(bases) / sizeof(bases[0]); b++) {
                            fa->data = (void *) bases[b];
                            SAY("alloc fattn dk=%lld dv=%lld n_q=%lld n_kv=%lld k=%s v=%s gqa=%d data=%#lx = %zu",
                                (long long) heads[h].dk, (long long) heads[h].dv, (long long) n_q, (long long) n_kv,
                                ggml_type_name(kv_types[it].k), ggml_type_name(kv_types[it].v), gqa, (unsigned long) bases[b],
                                ggml_backend_buft_get_alloc_size(buft, fa));
                        }
                        fa->data = NULL;
                    }
                }
            }
        }
    }
}

/* offload_op on both sides of the batch threshold, every arm of
 * `get_op_batch_size`. */
static void ask_offload(ggml_backend_dev_t dev, struct ggml_context * ctx) {
    static const int64_t ns[] = { 1, 8, 31, 32, 33, 64 };
    for (size_t i = 0; i < sizeof(ns) / sizeof(ns[0]); i++) {
        const int64_t n = ns[i];
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, n);
        SAY("offload MUL_MAT ne1=%lld = %d", (long long) n, ggml_backend_dev_offload_op(dev, ggml_mul_mat(ctx, w, x)));

        struct ggml_tensor * as = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 256, 64, 8);
        struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, n);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n);
        SAY("offload MUL_MAT_ID ne2=%lld = %d", (long long) n, ggml_backend_dev_offload_op(dev, ggml_mul_mat_id(ctx, as, b, ids)));

        struct ggml_tensor * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, n);
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        SAY("offload ROPE ne2=%lld = %d", (long long) n, ggml_backend_dev_offload_op(dev, ggml_rope(ctx, r, pos, 64, 0)));

        struct ggml_tensor * e = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1000);
        struct ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        SAY("offload GET_ROWS n=%lld = %d", (long long) n, ggml_backend_dev_offload_op(dev, ggml_get_rows(ctx, e, rows)));

        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, n);
        SAY("offload ADD nrows=%lld = %d", (long long) n, ggml_backend_dev_offload_op(dev, ggml_add(ctx, a, a)));
    }
}

/* graph_optimize: the graph being optimized and the case's label, for the
 * recording callback. */
static struct ggml_cgraph * opt_graph;
static const char * opt_label;
static long opt_deps;
static long opt_deps_topk;

/* A tensor as the dependency report names it: its node index, else its name
 * (every leaf the cases build is named), else "null". */
static void tensor_label(char * buf, size_t n, const struct ggml_tensor * t) {
    if (t == NULL) {
        snprintf(buf, n, "null");
        return;
    }
    for (int i = 0; i < ggml_graph_n_nodes(opt_graph); i++) {
        if (ggml_graph_node(opt_graph, i) == t) {
            snprintf(buf, n, "node%d", i);
            return;
        }
    }
    snprintf(buf, n, "leaf:%s", ggml_get_name(t));
}

static void record_alloc_dep(void * user_data, struct ggml_tensor * tensor, struct ggml_tensor * until) {
    (void) user_data;
    char a[64], b[64];
    tensor_label(a, sizeof(a), tensor);
    tensor_label(b, sizeof(b), until);
    SAY("graph_optimize %s dep %s until %s", opt_label, a, b);
    opt_deps++;
}

enum gating { GATE_SOFTMAX, GATE_SIGMOID, GATE_SQRT_SOFTPLUS, GATE_SOFTMAX_WEIGHT };

/* test-backend-ops' `test_topk_moe` graph (test-backend-ops.cpp:6774
 * @7fe450e19), plus, unless `at_end`, a consumer of the ids and one of the
 * weights after the chain, as a MoE layer has. Its context memory is
 * zeroed, so `nodes[n_nodes]` -- which graph_optimize names as the
 * dependency target of a chain that ends the graph -- reads as null. */
static struct ggml_cgraph * build_topk_moe(struct ggml_context * ctx, int64_t n_expert, int64_t n_used, int64_t n_tokens,
                                           enum gating g, bool bias, bool norm, float scale_w, bool at_end) {
    struct ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tokens);
    ggml_set_name(logits, "logits");
    struct ggml_tensor * probs = g == GATE_SOFTMAX ? ggml_soft_max(ctx, logits) :
                                 g == GATE_SIGMOID ? ggml_sigmoid(ctx, logits) :
                                 g == GATE_SQRT_SOFTPLUS ? ggml_sqrt(ctx, ggml_softplus(ctx, logits)) : logits;
    struct ggml_tensor * selection = probs;
    if (bias) {
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
        ggml_set_name(b, "exp_probs_b");
        selection = ggml_add(ctx, probs, b);
    }
    struct ggml_tensor * ids = ggml_argsort_top_k(ctx, selection, (int) n_used);
    struct ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), ids);
    if (g == GATE_SOFTMAX_WEIGHT) {
        w = ggml_reshape_3d(ctx, ggml_soft_max(ctx, ggml_reshape_2d(ctx, w, n_used, n_tokens)), 1, n_used, n_tokens);
    }
    if (norm) {
        w = ggml_reshape_2d(ctx, w, n_used, n_tokens);
        struct ggml_tensor * sum = ggml_clamp(ctx, ggml_sum_rows(ctx, w), 6.103515625e-5f, INFINITY);
        w = ggml_reshape_3d(ctx, ggml_div(ctx, w, sum), 1, n_used, n_tokens);
    }
    if (scale_w != 0.0f) {
        w = ggml_scale(ctx, w, scale_w);
    }
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    if (at_end) {
        /* `ids` is reached through the GET_ROWS, after the RESHAPE, which
         * keeps the order the fusion matches. */
        ggml_build_forward_expand(gf, w);
        return gf;
    }
    struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, n_used, n_tokens);
    ggml_set_name(x, "x");
    struct ggml_tensor * table = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, n_expert, n_tokens);
    ggml_set_name(table, "table");
    ggml_build_forward_expand(gf, ggml_add(ctx, ggml_mul(ctx, x, w), ggml_get_rows(ctx, table, ids)));
    return gf;
}

/* test-backend-ops' `test_moe_reduce` graph (test-backend-ops.cpp:6873
 * @7fe450e19) in its test-mode node order, plus a consumer. */
static struct ggml_cgraph * build_moe_reduce(struct ggml_context * ctx, int64_t n_embd, int64_t n_used, int64_t n_tokens, bool expert_scale) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_used, n_tokens);
    ggml_set_name(experts, "experts");
    struct ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tokens);
    ggml_set_name(weights, "weights");
    struct ggml_tensor * scaled = experts;
    if (expert_scale) {
        struct ggml_tensor * es = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tokens);
        ggml_set_name(es, "expert_scale");
        scaled = ggml_mul(ctx, experts, es);
    }
    struct ggml_tensor * weighted = ggml_mul(ctx, scaled, weights);
    struct ggml_tensor * views[16];
    for (int64_t e = 0; e < n_used; e++) {
        views[e] = ggml_view_2d(ctx, weighted, n_embd, n_tokens, weighted->nb[2], e * weighted->nb[1]);
        ggml_build_forward_expand(gf, views[e]);
    }
    struct ggml_tensor * out = views[0];
    for (int64_t e = 1; e < n_used; e++) {
        out = ggml_add(ctx, out, views[e]);
        ggml_build_forward_expand(gf, out);
    }
    struct ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(y, "y");
    ggml_build_forward_expand(gf, ggml_add(ctx, out, y));
    return gf;
}

static void optimize_one(ggml_backend_t backend, const char * label, struct ggml_cgraph * gf) {
    opt_graph = gf;
    opt_label = label;
    const long before = opt_deps;
    struct ggml_backend_graph_optimize_params params = { record_alloc_dep, NULL };
    backend->iface.graph_optimize(backend, gf, &params);
    SAY("graph_optimize %s nodes = %d deps = %ld", label, ggml_graph_n_nodes(gf), opt_deps - before);
}

static void ask_graph_optimize(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (backend == NULL || backend->iface.graph_optimize == NULL) {
        SAY("graph_optimize = %s", backend == NULL ? "no backend" : "null");
        if (backend) ggml_backend_free(backend);
        return;
    }
    static const char * gate_names[] = { "softmax", "sigmoid", "sqrt_softplus", "softmax_weight" };
    static const struct { int64_t n_expert, n_used, n_tokens; } shapes[] = { { 32, 4, 8 }, { 256, 6, 3 }, { 31, 4, 8 } };
    const size_t mem = 16 * 1024 * 1024;
    void * buf = malloc(mem);
    char label[160];
    for (int g = 0; g < 4; g++) {
        for (size_t sh = 0; sh < sizeof(shapes) / sizeof(shapes[0]); sh++) {
            for (int v = 0; v < 16; v++) {
                const bool bias = v & 1, norm = v & 2, scale = v & 4, at_end = v & 8;
                /* A norm chain that ends the graph, with no SCALE after it:
                 * ggml_cuda_topk_moe_fusion reads nodes[n_nodes]->op there
                 * (ggml-cuda.cu:2963 @7fe450e19), a NULL in this zeroed
                 * graph, and upstream segfaults -- the port, ported as
                 * written, alike (measured on the GB10, each side alone). */
                if (norm && !scale && at_end) continue;
                memset(buf, 0, mem);
                struct ggml_init_params ip = { mem, buf, true };
                struct ggml_context * ctx = ggml_init(ip);
                struct ggml_cgraph * gf = build_topk_moe(ctx, shapes[sh].n_expert, shapes[sh].n_used, shapes[sh].n_tokens,
                                                         (enum gating) g, bias, norm, scale ? 2.5f : 0.0f, at_end);
                snprintf(label, sizeof(label), "topk_moe %s n_expert=%lld n_used=%lld bias=%d norm=%d scale=%d at_end=%d",
                         gate_names[g], (long long) shapes[sh].n_expert, (long long) shapes[sh].n_used, bias, norm, scale, at_end);
                const long before = opt_deps;
                optimize_one(backend, label, gf);
                if (strcmp(side, "ref") == 0) {
                    opt_deps_topk += opt_deps - before;
                }
                ggml_free(ctx);
            }
        }
    }
    for (int64_t n_used = 2; n_used <= 8; n_used *= 2) {
        for (int es = 0; es < 2; es++) {
            memset(buf, 0, mem);
            struct ggml_init_params ip = { mem, buf, true };
            struct ggml_context * ctx = ggml_init(ip);
            snprintf(label, sizeof(label), "moe_reduce n_used=%lld expert_scale=%d", (long long) n_used, es);
            optimize_one(backend, label, build_moe_reduce(ctx, 64, n_used, 5, es));
            ggml_free(ctx);
        }
    }
    free(buf);
    ggml_backend_free(backend);
}

static void ask_device(size_t i, ggml_backend_dev_t dev) {
    SAY("dev %zu name = %s", i, ggml_backend_dev_name(dev));
    SAY("dev %zu description = %s", i, ggml_backend_dev_description(dev));
    SAY("dev %zu type = %d", i, (int) ggml_backend_dev_type(dev));
    struct ggml_backend_dev_props p;
    memset(&p, 0, sizeof(p));
    ggml_backend_dev_get_props(dev, &p);
    SAY("dev %zu props name = %s", i, p.name);
    SAY("dev %zu props description = %s", i, p.description);
    SAY("dev %zu props memory_total = %zu", i, p.memory_total);
    SAY("dev %zu props type = %d", i, (int) p.type);
    SAY("dev %zu props device_id = %s", i, p.device_id ? p.device_id : "(null)");
    SAY("dev %zu props caps = async %d host_buffer %d buffer_from_host_ptr %d events %d mmap %d", i,
        p.caps.async, p.caps.host_buffer, p.caps.buffer_from_host_ptr, p.caps.events, p.caps.mmap_support);
    size_t free_mem, total;
    ggml_backend_dev_memory(dev, &free_mem, &total);
    SAY("dev %zu memory total = %zu", i, total);

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(dev);
    ask_buft("buft", buft);
    ask_buft("host_buft", host);
    SAY("dev %zu supports_buft own = %d", i, ggml_backend_dev_supports_buft(dev, buft));
    SAY("dev %zu supports_buft host = %d", i, ggml_backend_dev_supports_buft(dev, host));
    SAY("dev %zu supports_buft cpu = %d", i, ggml_backend_dev_supports_buft(dev, ggml_backend_cpu_buffer_type()));

    struct ggml_init_params params = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    ask_offload(dev, ctx);
    ask_alloc_types(buft, ctx);
    ask_alloc_types(host, ctx);
    ask_alloc_fattn(buft, ctx);
    ggml_free(ctx);
    ask_graph_optimize(dev);
}

static void ask(const char * name, ggml_backend_reg_t reg) {
    side = name;
    ask_reg(reg);
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
        ask_device(i, ggml_backend_reg_dev_get(reg, i));
    }
}

int main(void) {
    ask("port", ggml_backend_cuda_reg());
    ask("ref", ref_ggml_backend_cuda_reg());
    /* A gate that recorded no top-k dependency on the reference compared
     * nothing about the v0.5.0 arm: say so rather than pass. */
    if (opt_deps_topk == 0) {
        fprintf(stderr, "cuda_backend_diff: graph_optimize recorded no top-k MoE dependency\n");
        return 3;
    }
    return 0;
}
