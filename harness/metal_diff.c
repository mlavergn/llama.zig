/* Compare the ported Metal buffer-type and device vtables against the
 * reference, through the registry.
 *
 * **Not a port.** This is a gate, driven by `scripts/metal-diff`.
 *
 * Why it exists: `make node-diff ARGS=--gpu` runs the graph and hashes
 * every node, which proves the backend computes the right values. It does
 * **not** see most of `ggml-metal.cpp`. Measured, by injection:
 *
 *   - dropping the four FLASH_ATTN_EXT scratch terms from
 *     `get_alloc_size` -- node-diff passed, and the terms are non-zero on
 *     this graph, so the fault was real and invisible;
 *   - `get_alignment` 32 -> 16 and 32 -> 64 -- node-diff passed both,
 *     though a probe shows the function is on the path;
 *   - the CUMSUM/ARGSORT and MUL_MAT_ID arms, and `offload_op`'s
 *     MUL_MAT_ID case -- node-diff passed, but those are provably inert
 *     here: a Qwen3.5 decode contains none of those ops.
 *
 * So the vtables are asked directly instead, for tensors that reach every
 * arm of the switch, and compared against the reference's answers.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include "ggml-metal-fusion.h"
#include "ggml-backend-impl.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Ours, and the reference's with its six exports renamed by -D. */
extern ggml_backend_reg_t ggml_backend_metal_reg(void);
extern ggml_backend_reg_t ref_ggml_backend_metal_reg(void);

/* The fusion table: ours (`src/ggml/metal/fusion.zig`, declared by
 * `ggml-metal-fusion.h`) and the reference's `ggml-metal-fusion.cpp` with
 * its 17 exports renamed by -D. Each side's info object is its own struct,
 * so each is only ever handed to its own functions. */
/* The reordering pass (`ggml-metal-common.cpp`, ours `src/ggml/metal/common.zig`),
 * the reference's nine exports renamed ref_*. Its fusion queries are
 * shared -- both sides pack by our `ggml_metal_fusion_max`, compared above
 * -- so this compares the packing-aware reorder alone. */
extern void ggml_graph_optimize(struct ggml_cgraph * gf);

/* The op encoders' scratch-size queries (`ggml-metal-ops.cpp`), which
 * `get_alloc_size` adds up. The reference `ggml-metal.cpp` calls *ours*
 * -- they are shared symbols -- so the get_alloc_size comparison cannot
 * see them; the reference `ggml-metal-ops.cpp` is compiled with all its
 * exports renamed ref_* and these are asked directly. */
#define EXTRA(f) extern size_t f(const struct ggml_tensor *); extern size_t ref_##f(const struct ggml_tensor *);
EXTRA(ggml_metal_op_mul_mat_id_extra_tpe)
EXTRA(ggml_metal_op_mul_mat_id_extra_ids)
EXTRA(ggml_metal_op_flash_attn_ext_extra_pad)
EXTRA(ggml_metal_op_flash_attn_ext_extra_blk)
EXTRA(ggml_metal_op_flash_attn_ext_extra_tmp)
EXTRA(ggml_metal_op_flash_attn_ext_extra_kv_f16)
EXTRA(ggml_metal_op_flash_attn_ext_extra_idx)
EXTRA(ggml_metal_op_flash_attn_ext_extra_rows)
EXTRA(ggml_metal_op_argmax_extra)
#undef EXTRA
extern bool ggml_metal_op_flash_attn_ext_use_vec(const struct ggml_tensor *);
extern bool ref_ggml_metal_op_flash_attn_ext_use_vec(const struct ggml_tensor *);

/* The mat-mul dispatch predicates of `ggml-metal-common.cpp` (ours
 * `src/ggml/metal/common.zig`), the reference's renamed ref_*. Both the
 * encoder and the Objective-C `supports_op` ask these, and since v0.6.0
 * the fusion table's MUL_MAT + ADD check does too, so they are swept
 * directly: a wrong answer moves a MUL_MAT to another kernel, which only
 * `node-diff ARGS=--gpu` could otherwise see, and only for the shapes one
 * decode reaches. */
#define PRED(ret, f, ...) extern ret f(__VA_ARGS__); extern ret ref_##f(__VA_ARGS__);
PRED(bool,    ggml_metal_op_mul_mat_use_fwht,    const struct ggml_tensor *, size_t)
PRED(bool,    ggml_metal_op_mul_mat_use_mm,      const struct ggml_tensor *, bool)
PRED(bool,    ggml_metal_op_mul_mat_id_use_mm,   const struct ggml_tensor *, bool)
PRED(bool,    ggml_metal_op_mul_mat_use_mma,     const struct ggml_tensor *)
PRED(bool,    ggml_metal_op_mul_mat_may_use_mma, const struct ggml_tensor *)
PRED(int,     ggml_metal_mul_mv_mma_kind,        enum ggml_type, int)
PRED(int,     ggml_metal_mul_mv_mma_rt,          const struct ggml_tensor *)
PRED(int64_t, ggml_metal_mul_mv_mma_k_step,      enum ggml_type, int)
#undef PRED

/* Each side's backend, for its `graph_optimize` vtable entry. */
static ggml_backend_t backend_a = NULL, backend_b = NULL;
extern void ref_ggml_graph_optimize(struct ggml_cgraph * gf);

extern ggml_metal_fusion_id ref_ggml_metal_fusion_get_id(const struct ggml_metal_fusion * fusion);
extern void ref_ggml_metal_fusion_add_alloc_deps(void * user_data, void (*add_alloc_dep)(void *, struct ggml_tensor *, struct ggml_tensor *), struct ggml_cgraph * gf);
extern struct ggml_metal_fusion_info * ref_ggml_metal_fusion_info_init(bool enabled, int debug);
extern void         ref_ggml_metal_fusion_info_free(struct ggml_metal_fusion_info * finfo);
extern bool         ref_ggml_metal_fusion_info_enabled(const struct ggml_metal_fusion_info * finfo);
extern bool         ref_ggml_metal_fusion_info_stats(const struct ggml_metal_fusion_info * finfo);
extern int          ref_ggml_metal_fusion_info_debug(const struct ggml_metal_fusion_info * finfo);
extern int          ref_ggml_metal_fusion_info_n_fusions(const struct ggml_metal_fusion_info * finfo);
extern const char * ref_ggml_metal_fusion_info_label(const struct ggml_metal_fusion_info * finfo, int idx);
extern uint64_t     ref_ggml_metal_fusion_info_count(const struct ggml_metal_fusion_info * finfo, int idx);
extern void         ref_ggml_metal_fusion_info_count_fusion(struct ggml_metal_fusion_info * finfo, const struct ggml_metal_fusion * fusion);
extern void         ref_ggml_metal_fusion_info_set_enabled(struct ggml_metal_fusion_info * finfo, bool enabled);
extern void         ref_ggml_metal_fusion_info_stats_init(struct ggml_metal_fusion_info * finfo);
extern void         ref_ggml_metal_fusion_info_stats_reset(struct ggml_metal_fusion_info * finfo);
extern int          ref_ggml_metal_fusion_info_stats_get(const struct ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n);
extern void         ref_ggml_metal_fusion_info_labels_init(struct ggml_metal_fusion_info * finfo);
extern const ggml_metal_fusion * ref_ggml_metal_fusion_next(const struct ggml_cgraph * gf, const int * node_idxs, int n_idxs, int idx, ggml_metal_fusion_mode mode, int * n_out);
extern int          ref_ggml_metal_fusion_max(const struct ggml_cgraph * gf, int idx);

static int fails  = 0;
static int checks = 0;

static void cmp_sz(const char *what, size_t a, size_t b) {
    checks++;
    if (a != b) {
        printf("  %-52s ported %-12zu reference %-12zu WRONG\n", what, a, b);
        fails++;
    }
}

static void cmp_i(const char *what, long a, long b) {
    checks++;
    if (a != b) {
        printf("  %-52s ported %-12ld reference %-12ld WRONG\n", what, a, b);
        fails++;
    }
}

static void cmp_s(const char *what, const char *a, const char *b) {
    checks++;
    if (!a || !b || strcmp(a, b) != 0) {
        printf("  %-52s ported %-12s reference %-12s WRONG\n", what, a ? a : "(null)", b ? b : "(null)");
        fails++;
    }
}

/* ---- the fusion table ---------------------------------------------------
 *
 * `node-diff ARGS=--gpu` cannot see this: on Qwen3.5 a fused and an
 * unfused decode give the same logits bit for bit (measured), so a pattern
 * dropped from the table, or one that stops matching, changes no hash.
 * The table is asked directly instead -- its entries, its labels and
 * counters, and `ggml_metal_fusion_next` / `_max` at every position of
 * small graphs built to reach each check's arms, in both modes. The
 * graphs live in Metal buffers, because the FULL mode asks which buffer
 * an addend is in. */

/* Since v0.5.0 the table is opaque -- `ggml_metal_fusion_all` is gone and
 * the header declares `struct ggml_metal_fusion` without a body -- so an
 * entry's position is read back the only way the API allows: count it in
 * a stats-enabled info and see which counter moved. Both sides count by
 * the entry's offset into their own table. */
static struct ggml_metal_fusion_info * idx_info_a = NULL;
static struct ggml_metal_fusion_info * idx_info_b = NULL;

static int table_index(int ref, const ggml_metal_fusion * f) {
    if (f == NULL) return -1;
    struct ggml_metal_fusion_info * fi = ref ? idx_info_b : idx_info_a;
    const int n = ref ? ref_ggml_metal_fusion_info_n_fusions(fi) : ggml_metal_fusion_info_n_fusions(fi);
    if (ref) ref_ggml_metal_fusion_info_stats_reset(fi); else ggml_metal_fusion_info_stats_reset(fi);
    if (ref) ref_ggml_metal_fusion_info_count_fusion(fi, f); else ggml_metal_fusion_info_count_fusion(fi, f);
    for (int i = 0; i < n; i++) {
        const uint64_t k = ref ? ref_ggml_metal_fusion_info_count(fi, i) : ggml_metal_fusion_info_count(fi, i);
        if (k == 1) return i;
    }
    return -2; /* not in the table at all */
}

/* `ggml_metal_fusion_add_alloc_deps` reports (tensor, until) pairs through
 * a callback; each side's are recorded and compared in order. */
#define MAX_DEPS 512
struct dep_log { int n; struct ggml_tensor * t[MAX_DEPS]; struct ggml_tensor * u[MAX_DEPS]; };
static void dep_record(void * ud, struct ggml_tensor * t, struct ggml_tensor * u) {
    struct dep_log * l = (struct dep_log *) ud;
    if (l->n < MAX_DEPS) { l->t[l->n] = t; l->u[l->n] = u; }
    l->n++;
}
static int deps_seen = 0;
static int reorders_seen = 0;
static int backend_deps_seen = 0;

/* Every table entry each side handed out, by table index, as the probes
 * found them: the info object's counters are fed these. */
static const ggml_metal_fusion * entry_a[32];
static const ggml_metal_fusion * entry_b[32];
static int n_entry = 0;


/* How often the reference matched each table entry, per mode, and how
 * often STRUCTURAL matched where FULL refused: every entry must be reached
 * in both modes, or a pattern could be dropped from the table and nothing
 * here would notice. */
static int fusion_hits[2][32];
static int fusion_full_refusals = 0;

static ggml_backend_buffer_t fusion_bufs[64];
static int n_fusion_bufs = 0;

/* Every position of `gf`: `_max` on raw indices, `_next` on the
 * non-empty ones (as `ggml-metal-ops.cpp` filters them), both modes. */
static void fusion_probe_graph(const char * name, struct ggml_cgraph * gf) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    int idxs[256];
    int n_idxs = 0;
    for (int i = 0; i < n_nodes && n_idxs < 256; i++) {
        struct ggml_tensor * t = ggml_graph_node(gf, i);
        if (!ggml_op_is_empty(t->op) && !ggml_is_empty(t)) idxs[n_idxs++] = i;
    }

    char lbl[96];
    for (int i = 0; i < n_nodes; i++) {
        snprintf(lbl, sizeof lbl, "fusion_max(%s @%d)", name, i);
        cmp_i(lbl, ggml_metal_fusion_max(gf, i), ref_ggml_metal_fusion_max(gf, i));
    }
    for (int m = 0; m < 2; m++) {
        for (int i = 0; i < n_idxs; i++) {
            int na = -7, nb = -7;
            const ggml_metal_fusion * fa = ggml_metal_fusion_next(gf, idxs, n_idxs, i, (ggml_metal_fusion_mode) m, &na);
            const ggml_metal_fusion * fb = ref_ggml_metal_fusion_next(gf, idxs, n_idxs, i, (ggml_metal_fusion_mode) m, &nb);
            snprintf(lbl, sizeof lbl, "fusion_next(%s @%d %s)", name, i, m ? "full" : "struct");
            cmp_i(lbl, table_index(0, fa), table_index(1, fb));
            snprintf(lbl, sizeof lbl, "fusion_next n_out(%s @%d %s)", name, i, m ? "full" : "struct");
            cmp_i(lbl, na, nb);
            snprintf(lbl, sizeof lbl, "fusion_get_id(%s @%d %s)", name, i, m ? "full" : "struct");
            cmp_i(lbl, fa ? (long) ggml_metal_fusion_get_id(fa) : -1, fb ? (long) ref_ggml_metal_fusion_get_id(fb) : -1);
            const int ib = table_index(1, fb);
            if (ib >= 0 && ib < 32 && ib == table_index(0, fa)) { entry_a[ib] = fa; entry_b[ib] = fb; }
            if (ib >= 0 && ib < 32) fusion_hits[m][ib]++;
            if (m == 1) {
                int ns = 0;
                if (ref_ggml_metal_fusion_next(gf, idxs, n_idxs, i, GGML_METAL_FUSION_STRUCTURAL, &ns) != NULL && fb == NULL) {
                    fusion_full_refusals++;
                }
            }
        }
    }

    /* the alloc deps graph_optimize hands the scheduler */
    static struct dep_log da, db;
    da.n = 0; db.n = 0;
    ggml_metal_fusion_add_alloc_deps(&da, dep_record, gf);
    ref_ggml_metal_fusion_add_alloc_deps(&db, dep_record, gf);
    snprintf(lbl, sizeof lbl, "alloc_deps n(%s)", name);
    cmp_i(lbl, da.n, db.n);
    for (int k = 0; k < da.n && k < db.n && k < MAX_DEPS; k++) {
        snprintf(lbl, sizeof lbl, "alloc_dep[%d] tensor(%s)", k, name);
        cmp_i(lbl, (long) (intptr_t) da.t[k], (long) (intptr_t) db.t[k]);
        snprintf(lbl, sizeof lbl, "alloc_dep[%d] until(%s)", k, name);
        cmp_i(lbl, (long) (intptr_t) da.u[k], (long) (intptr_t) db.u[k]);
    }
    deps_seen += db.n;

    /* graph_optimize's order on a copy of the graph, each side its own */
    {
        struct ggml_init_params gp = { ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false) * 4 + 4096, NULL, true };
        struct ggml_context * gctx = ggml_init(gp);
        struct ggml_cgraph * ga = ggml_graph_dup(gctx, gf, false);
        struct ggml_cgraph * gb = ggml_graph_dup(gctx, gf, false);
        ggml_graph_optimize(ga);
        ref_ggml_graph_optimize(gb);
        int moved = 0;
        for (int i = 0; i < n_nodes; i++) {
            snprintf(lbl, sizeof lbl, "graph_optimize(%s)[%d]", name, i);
            cmp_i(lbl, (long) (intptr_t) ggml_graph_node(ga, i), (long) (intptr_t) ggml_graph_node(gb, i));
            if (ggml_graph_node(gb, i) != ggml_graph_node(gf, i)) moved = 1;
        }
        reorders_seen += moved;

        /* and the backend's own graph_optimize: alloc deps, then the
         * context's reorder, through each side's vtable */
        if (backend_a && backend_b) {
            struct ggml_cgraph * va = ggml_graph_dup(gctx, gf, false);
            struct ggml_cgraph * vb = ggml_graph_dup(gctx, gf, false);
            static struct dep_log la, lb;
            la.n = 0; lb.n = 0;
            struct ggml_backend_graph_optimize_params pa = { dep_record, &la };
            struct ggml_backend_graph_optimize_params pb = { dep_record, &lb };
            backend_a->iface.graph_optimize(backend_a, va, &pa);
            backend_b->iface.graph_optimize(backend_b, vb, &pb);
            snprintf(lbl, sizeof lbl, "backend graph_optimize deps n(%s)", name);
            cmp_i(lbl, la.n, lb.n);
            for (int k = 0; k < la.n && k < lb.n && k < MAX_DEPS; k++) {
                snprintf(lbl, sizeof lbl, "backend graph_optimize dep[%d](%s)", k, name);
                cmp_i(lbl, (long) (intptr_t) la.t[k] ^ (long) (intptr_t) la.u[k], (long) (intptr_t) lb.t[k] ^ (long) (intptr_t) lb.u[k]);
            }
            for (int i = 0; i < n_nodes; i++) {
                snprintf(lbl, sizeof lbl, "backend graph_optimize(%s)[%d]", name, i);
                cmp_i(lbl, (long) (intptr_t) ggml_graph_node(va, i), (long) (intptr_t) ggml_graph_node(vb, i));
            }
            backend_deps_seen += lb.n;
        }
        ggml_free(gctx);
    }
}

static struct ggml_context * fusion_ctx(void) {
    struct ggml_init_params ip = { 4u * 1024u * 1024u, NULL, true };
    return ggml_init(ip);
}

/* Allocate every tensor of `ctx` in one Metal buffer of its own. */
static void fusion_alloc(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf && n_fusion_bufs < 64) fusion_bufs[n_fusion_bufs++] = buf;
}

static void fusion_info_diff(void) {
    char lbl[96];
    const ggml_metal_fusion * first_a = entry_a[0];
    const ggml_metal_fusion * first_b = entry_b[0];
    for (int i = 0; i < n_entry; i++) {
        if (!entry_a[i] || !entry_b[i]) {
            checks++; fails++;
            printf("  fusion[%d]: no entry pointer recorded for the info checks\n", i);
            return;
        }
    }

    /* the info object: flags, labels, counters, stats_get's two forms */
    for (int debug = 0; debug <= 2; debug++) {
        for (int en = 0; en <= 1; en++) {
            struct ggml_metal_fusion_info * a = ggml_metal_fusion_info_init(en, debug);
            struct ggml_metal_fusion_info * b = ref_ggml_metal_fusion_info_init(en, debug);
            snprintf(lbl, sizeof lbl, "info(%d,%d) enabled", en, debug);   cmp_i(lbl, ggml_metal_fusion_info_enabled(a), ref_ggml_metal_fusion_info_enabled(b));
            snprintf(lbl, sizeof lbl, "info(%d,%d) stats", en, debug);     cmp_i(lbl, ggml_metal_fusion_info_stats(a), ref_ggml_metal_fusion_info_stats(b));
            snprintf(lbl, sizeof lbl, "info(%d,%d) debug", en, debug);     cmp_i(lbl, ggml_metal_fusion_info_debug(a), ref_ggml_metal_fusion_info_debug(b));
            snprintf(lbl, sizeof lbl, "info(%d,%d) n_fusions", en, debug); cmp_i(lbl, ggml_metal_fusion_info_n_fusions(a), ref_ggml_metal_fusion_info_n_fusions(b));
            snprintf(lbl, sizeof lbl, "info(%d,%d) stats_get(null)", en, debug);
            cmp_i(lbl, ggml_metal_fusion_info_stats_get(a, NULL, NULL, 0), ref_ggml_metal_fusion_info_stats_get(b, NULL, NULL, 0));

            /* counted before stats are on: must not count */
            ggml_metal_fusion_info_count_fusion(a, first_a);
            ref_ggml_metal_fusion_info_count_fusion(b, first_b);

            ggml_metal_fusion_info_stats_init(a);
            ref_ggml_metal_fusion_info_stats_init(b);
            ggml_metal_fusion_info_labels_init(a); /* idempotent */
            ref_ggml_metal_fusion_info_labels_init(b);
            ggml_metal_fusion_info_set_enabled(a, !en);
            ref_ggml_metal_fusion_info_set_enabled(b, !en);
            snprintf(lbl, sizeof lbl, "info(%d,%d) set_enabled", en, debug); cmp_i(lbl, ggml_metal_fusion_info_enabled(a), ref_ggml_metal_fusion_info_enabled(b));

            const int nf = ggml_metal_fusion_info_n_fusions(a);
            snprintf(lbl, sizeof lbl, "info(%d,%d) n_fusions after init", en, debug); cmp_i(lbl, nf, ref_ggml_metal_fusion_info_n_fusions(b));
            for (int i = 0; i < nf && i < n_entry; i++) {
                for (int k = 0; k <= i % 3; k++) {
                    ggml_metal_fusion_info_count_fusion(a, entry_a[i]);
                    ref_ggml_metal_fusion_info_count_fusion(b, entry_b[i]);
                }
            }
            ggml_metal_fusion_info_count_fusion(a, NULL);
            ref_ggml_metal_fusion_info_count_fusion(b, NULL);
            for (int i = 0; i < nf; i++) {
                snprintf(lbl, sizeof lbl, "info label[%d]", i); cmp_s(lbl, ggml_metal_fusion_info_label(a, i), ref_ggml_metal_fusion_info_label(b, i));
                snprintf(lbl, sizeof lbl, "info count[%d]", i); cmp_i(lbl, (long) ggml_metal_fusion_info_count(a, i), (long) ref_ggml_metal_fusion_info_count(b, i));
            }
            const char * la[32] = {0}; const char * lb[32] = {0};
            uint64_t ca[32] = {0}, cb[32] = {0};
            for (int cap = 0; cap <= nf + 1 && cap <= 32; cap += 5) {
                const int ra = ggml_metal_fusion_info_stats_get(a, la, ca, cap);
                const int rb = ref_ggml_metal_fusion_info_stats_get(b, lb, cb, cap);
                snprintf(lbl, sizeof lbl, "stats_get(cap=%d)", cap); cmp_i(lbl, ra, rb);
                for (int i = 0; i < ra && i < rb; i++) {
                    snprintf(lbl, sizeof lbl, "stats_get(cap=%d) label[%d]", cap, i); cmp_s(lbl, la[i], lb[i]);
                    snprintf(lbl, sizeof lbl, "stats_get(cap=%d) count[%d]", cap, i); cmp_i(lbl, (long) ca[i], (long) cb[i]);
                }
            }
            ggml_metal_fusion_info_stats_reset(a);
            ref_ggml_metal_fusion_info_stats_reset(b);
            for (int i = 0; i < nf; i++) {
                snprintf(lbl, sizeof lbl, "count after reset[%d]", i); cmp_i(lbl, (long) ggml_metal_fusion_info_count(a, i), (long) ref_ggml_metal_fusion_info_count(b, i));
            }
            ggml_metal_fusion_info_free(a);
            ref_ggml_metal_fusion_info_free(b);
        }
    }

}

/* ---- the mat-mul dispatch predicates -------------------------------------
 *
 * Every type, row lengths around each K step (32 and 64) and block size,
 * src1 rows across every per-type minimum and the 16-row maximum, F32 and
 * F16 src1, a src1 whose row stride is not 16-byte aligned, a batch past
 * INT16_MAX, a broadcast batch, and a src0 strided in K; then the Hadamard
 * hint over FWHT widths from 32 to 16384 against threadgroup-memory
 * budgets either side of each width's float[N]. */
static int mma_true = 0, mma_may_true = 0, fwht_true = 0, fwht_false_wide = 0;

static void mma_predicates_diff(void) {
    printf("=== Metal mat-mul dispatch predicates: ported vs reference ===\n");

    struct ggml_init_params ip = { 256u * 1024u * 1024u, NULL, true };
    char lbl[160];

    const int64_t ne00s[] = { 32, 64, 96, 128, 160, 256, 288, 320, 512, 768, 4096 };
    const int64_t ne11s[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32 };

    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const enum ggml_type type = (enum ggml_type) t;
        if (ggml_type_size(type) == 0 || ggml_blck_size(type) == 0) continue;

        for (int rt = 0; rt <= 3; rt++) {
            snprintf(lbl, sizeof lbl, "mma_kind(%s, rt=%d)", ggml_type_name(type), rt);
            cmp_i(lbl, ggml_metal_mul_mv_mma_kind(type, rt), ref_ggml_metal_mul_mv_mma_kind(type, rt));
            snprintf(lbl, sizeof lbl, "mma_k_step(%s, rt=%d)", ggml_type_name(type), rt);
            cmp_i(lbl, (long) ggml_metal_mul_mv_mma_k_step(type, rt), (long) ref_ggml_metal_mul_mv_mma_k_step(type, rt));
        }

        struct ggml_context * ctx = ggml_init(ip);
        for (size_t i = 0; i < sizeof ne00s / sizeof *ne00s; i++) {
            const int64_t ne00 = ne00s[i];
            if (ne00 % ggml_blck_size(type) != 0) continue;
            for (size_t j = 0; j < sizeof ne11s / sizeof *ne11s; j++) {
                const int64_t ne11 = ne11s[j];
                /* shape variants: 0 plain, 1 F16 src1, 2 src1 rows not 16-byte strided,
                 * 3 batch past INT16_MAX, 4 broadcast batch, 5 src0 strided in K */
                for (int v = 0; v < 6; v++) {
                    if (v >= 2 && j % 3 != 0) continue;
                    struct ggml_tensor * a = v == 3 ? ggml_new_tensor_3d(ctx, type, ne00, 64, 1)
                                           : v == 4 ? ggml_new_tensor_4d(ctx, type, ne00, 64, 2, 1)
                                           :          ggml_new_tensor_2d(ctx, type, ne00, 64);
                    if (v == 5) {
                        /* a src0 whose nb[0] is not the type size: no view
                         * constructor makes one, so the stride is set by hand */
                        for (int d = 0; d < 4; d++) a->nb[d] *= 2;
                    }
                    struct ggml_tensor * b;
                    if (v == 1) {
                        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne00, ne11);
                    } else if (v == 2) {
                        struct ggml_tensor * big = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00 + 1, ne11);
                        b = ggml_view_2d(ctx, big, ne00, ne11, big->nb[1], 0);
                    } else if (v == 3) {
                        b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne00, ne11, 40000);
                    } else if (v == 4) {
                        b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne00, ne11, 6, 3);
                    } else {
                        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne11);
                    }
                    struct ggml_tensor * mm = ggml_mul_mat(ctx, a, b);

                    snprintf(lbl, sizeof lbl, "use_mma(%s ne00=%lld ne11=%lld v%d)", ggml_type_name(type), (long long) ne00, (long long) ne11, v);
                    const bool um = ggml_metal_op_mul_mat_use_mma(mm);
                    cmp_i(lbl, um, ref_ggml_metal_op_mul_mat_use_mma(mm));
                    mma_true += um;
                    snprintf(lbl, sizeof lbl, "may_use_mma(%s ne00=%lld ne11=%lld v%d)", ggml_type_name(type), (long long) ne00, (long long) ne11, v);
                    const bool mu = ggml_metal_op_mul_mat_may_use_mma(mm);
                    cmp_i(lbl, mu, ref_ggml_metal_op_mul_mat_may_use_mma(mm));
                    mma_may_true += mu;
                    snprintf(lbl, sizeof lbl, "mma_rt(%s ne00=%lld ne11=%lld v%d)", ggml_type_name(type), (long long) ne00, (long long) ne11, v);
                    cmp_i(lbl, ggml_metal_mul_mv_mma_rt(mm), ref_ggml_metal_mul_mv_mma_rt(mm));
                    for (int sm = 0; sm <= 1; sm++) {
                        snprintf(lbl, sizeof lbl, "use_mm(%s ne00=%lld ne11=%lld v%d sm=%d)", ggml_type_name(type), (long long) ne00, (long long) ne11, v, sm);
                        cmp_i(lbl, ggml_metal_op_mul_mat_use_mm(mm, sm), ref_ggml_metal_op_mul_mat_use_mm(mm, sm));
                    }
                }
            }
        }
        ggml_free(ctx);
    }

    /* MUL_MAT_ID's matrix-matrix break-even, at expert-id rows either side
     * of 32 -- and, since v0.7.0, at expert counts either side of the 1024
     * the id-map kernel's one threadgroup takes */
    {
        struct ggml_context * ctx = ggml_init(ip);
        const int64_t ne21s[] = { 1, 31, 32, 33 };
        const int64_t ne00s2[] = { 32, 64 };
        const int64_t ne02s[] = { 4, 1023, 1024, 1025 };
        int id_mm_true = 0, id_mm_false_experts = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                for (int e = 0; e < 4; e++) {
                    struct ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne00s2[j], 16, ne02s[e]);
                    struct ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne00s2[j], 2, ne21s[i]);
                    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, ne21s[i]);
                    struct ggml_tensor * t = ggml_mul_mat_id(ctx, w, x, ids);
                    for (int sm = 0; sm <= 1; sm++) {
                        snprintf(lbl, sizeof lbl, "id_use_mm(ne00=%lld ne02=%lld ne21=%lld sm=%d)",
                                 (long long) ne00s2[j], (long long) ne02s[e], (long long) ne21s[i], sm);
                        const bool u = ggml_metal_op_mul_mat_id_use_mm(t, sm);
                        cmp_i(lbl, u, ref_ggml_metal_op_mul_mat_id_use_mm(t, sm));
                        id_mm_true += u;
                        id_mm_false_experts += !u && sm && ne00s2[j] >= 64 && ne21s[i] >= 32 && ne02s[e] > 1024;
                    }
                }
            }
        }
        ggml_free(ctx);
        checks += 2;
        if (id_mm_true == 0)          { printf("  id_use_mm never true\n"); fails++; }
        if (id_mm_false_experts == 0) { printf("  id_use_mm never refused an expert count\n"); fails++; }
    }

    /* the Hadamard hint: widths with and without a kernel, F32 and F16 src1
     * and a non-F32 result type, against budgets either side of float[N] */
    {
        struct ggml_context * ctx = ggml_init(ip);
        const int64_t ns[] = { 32, 64, 128, 256, 512, 768, 1024, 2048, 4096, 8192, 16384 };
        const size_t mems[] = { 0, 4095, 4096, 8191, 8192, 16384, 32767, 32768, 65536 };
        for (size_t i = 0; i < sizeof ns / sizeof *ns; i++) {
            for (int v = 0; v < 4; v++) {
                const enum ggml_type tb = v == 1 ? GGML_TYPE_F16 : GGML_TYPE_F32;
                struct ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ns[i], ns[i]);
                struct ggml_tensor * x = ggml_new_tensor_2d(ctx, tb, ns[i], 3);
                struct ggml_tensor * mm = ggml_mul_mat(ctx, h, x);
                if (v != 2) ggml_mul_mat_set_hint(mm, GGML_HINT_SRC0_IS_HADAMARD);
                if (v == 3) mm->type = GGML_TYPE_F16; /* the check reads op->type */
                for (size_t k = 0; k < sizeof mems / sizeof *mems; k++) {
                    snprintf(lbl, sizeof lbl, "use_fwht(n=%lld v%d tg_mem=%zu)", (long long) ns[i], v, mems[k]);
                    const bool f = ggml_metal_op_mul_mat_use_fwht(mm, mems[k]);
                    cmp_i(lbl, f, ref_ggml_metal_op_mul_mat_use_fwht(mm, mems[k]));
                    fwht_true += f;
                    if (!f && v == 0 && ns[i] >= 1024 && ns[i] <= 8192) fwht_false_wide++;
                }
            }
        }
        ggml_free(ctx);
    }

    /* each predicate must have answered yes somewhere, or the sweep tests nothing */
    checks += 4;
    if (mma_true == 0)        { printf("  use_mma never true\n"); fails++; }
    if (mma_may_true == 0)    { printf("  may_use_mma never true\n"); fails++; }
    if (fwht_true == 0)       { printf("  use_fwht never true\n"); fails++; }
    if (fwht_false_wide == 0) { printf("  use_fwht never refused a wide width for its budget\n"); fails++; }
}

/* MUL_MAT_ID's two id-map scratch regions, swept over expert counts,
 * experts used and tokens. Since v0.7.0 ("fix mul_mat_id with duplicate
 * ids") the first is a tile list bounded by
 * `ggml_metal_op_mul_mat_id_max_tiles`, whose three arms -- the bound
 * clamped to the entry count, the per-expert partial tiles bounded by the
 * experts, and by the entries -- each need a shape of their own; the
 * second holds one padding slot per expert. Both sides also allocate the
 * sum through `get_alloc_size`, compared too. A zero-token op is in the
 * sweep: the C's bound goes negative there and its `size_t` product wraps,
 * which the port must reproduce. */
static void mul_mat_id_extra_diff(ggml_backend_buffer_type_t bt_a, ggml_backend_buffer_type_t bt_b) {
    printf("=== Metal MUL_MAT_ID id-map scratch: ported vs reference ===\n");

    struct ggml_init_params ip = { 64u * 1024u * 1024u, NULL, true };
    char lbl[160];

    const int64_t ne02s[] = { 1, 2, 8, 32, 64, 128, 1024 };
    const int64_t ne20s[] = { 1, 2, 4, 8 };
    const int64_t ne21s[] = { 0, 1, 3, 7, 16, 31, 32, 33, 40, 64, 512 };
    int clamped = 0, by_experts = 0, by_entries = 0;

    for (size_t a = 0; a < sizeof ne02s / sizeof *ne02s; a++) {
        struct ggml_context * ctx = ggml_init(ip);
        for (size_t b = 0; b < sizeof ne20s / sizeof *ne20s; b++) {
            if (ne20s[b] > ne02s[a]) continue;
            for (size_t c = 0; c < sizeof ne21s / sizeof *ne21s; c++) {
                const int64_t ne02 = ne02s[a], ne20 = ne20s[b], ne21 = ne21s[c];
                struct ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 32, ne02);
                struct ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, ne20, ne21);
                struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne20, ne21);
                struct ggml_tensor * t   = ggml_mul_mat_id(ctx, w, x, ids);

                const int64_t n_rows = ne20*ne21;
                const int64_t tiles  = (n_rows + 31)/32 + (ne02 < n_rows ? ne02 : n_rows) - 1;
                clamped    += n_rows < tiles;
                by_experts += ne02 < n_rows;
                by_entries += ne02 >= n_rows && n_rows > 0;

                snprintf(lbl, sizeof lbl, "id extra_tpe(ne02=%lld ne20=%lld ne21=%lld)", (long long) ne02, (long long) ne20, (long long) ne21);
                cmp_sz(lbl, ggml_metal_op_mul_mat_id_extra_tpe(t), ref_ggml_metal_op_mul_mat_id_extra_tpe(t));
                snprintf(lbl, sizeof lbl, "id extra_ids(ne02=%lld ne20=%lld ne21=%lld)", (long long) ne02, (long long) ne20, (long long) ne21);
                cmp_sz(lbl, ggml_metal_op_mul_mat_id_extra_ids(t), ref_ggml_metal_op_mul_mat_id_extra_ids(t));
                snprintf(lbl, sizeof lbl, "id get_alloc_size(ne02=%lld ne20=%lld ne21=%lld)", (long long) ne02, (long long) ne20, (long long) ne21);
                cmp_sz(lbl, ggml_backend_buft_get_alloc_size(bt_a, t), ggml_backend_buft_get_alloc_size(bt_b, t));
            }
        }
        ggml_free(ctx);
    }

    /* each arm of the tile bound must have been reached, or the sweep tests less than it says */
    checks += 3;
    if (clamped == 0)    { printf("  max_tiles never clamped to the entry count\n"); fails++; }
    if (by_experts == 0) { printf("  max_tiles never bounded by the experts\n"); fails++; }
    if (by_entries == 0) { printf("  max_tiles never bounded by the entries\n"); fails++; }
}

/* The two-stage ARGMAX's scratch (v0.8.0): nothing up to N_ARGMAX_CHUNK, a
 * max and an index per part per row past it, the parts capped at
 * N_ARGMAX_NPART_MAX. Rows either side of every edge: the size is the only
 * thing that says the scratch exists, and a kernel writing past a short one
 * lands in the next tensor. */
static void argmax_extra_diff(ggml_backend_buffer_type_t bt_a, ggml_backend_buffer_type_t bt_b) {
    printf("=== Metal ARGMAX scratch: ported vs reference ===\n");

    struct ggml_init_params ip = { 16u * 1024u * 1024u, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);
    char lbl[160];

    const int64_t ne00s[] = { 1, 32, 4095, 4096, 4097, 8191, 8192, 8193, 12288, 12289, 65536,
                              1044480, 1044481, 1048575, 1048576, 1048577, 1052672, 1052673, 2097152, 16777216 };
    const int64_t ne01s[] = { 1, 2, 3, 7, 512, 100000 };
    int none = 0, split = 0, capped = 0;

    for (size_t a = 0; a < sizeof ne00s / sizeof *ne00s; a++) {
        for (size_t b = 0; b < sizeof ne01s / sizeof *ne01s; b++) {
            const int64_t ne00 = ne00s[a], ne01 = ne01s[b];
            struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne01);
            struct ggml_tensor * t = ggml_argmax(ctx, x);

            const size_t want = ref_ggml_metal_op_argmax_extra(t);
            none   += want == 0;
            split  += want != 0 && want < 8u*256u*(size_t) ne01;
            capped += want == 8u*256u*(size_t) ne01;

            snprintf(lbl, sizeof lbl, "argmax_extra(ne00=%lld ne01=%lld)", (long long) ne00, (long long) ne01);
            cmp_sz(lbl, ggml_metal_op_argmax_extra(t), want);
            snprintf(lbl, sizeof lbl, "argmax get_alloc_size(ne00=%lld ne01=%lld)", (long long) ne00, (long long) ne01);
            cmp_sz(lbl, ggml_backend_buft_get_alloc_size(bt_a, t), ggml_backend_buft_get_alloc_size(bt_b, t));
        }
    }

    ggml_free(ctx);

    /* each arm of the part count must have been reached */
    checks += 3;
    if (none == 0)   { printf("  no row took one pass\n"); fails++; }
    if (split == 0)  { printf("  no row split below the cap\n"); fails++; }
    if (capped == 0) { printf("  no row reached the part cap\n"); fails++; }
}

static void fusion_diff(ggml_backend_buffer_type_t buft) {
    printf("=== Metal fusion table: ported vs reference ===\n");

    /* The table's length and each entry's non-empty op sequence, through
     * the labels; the ids, the raw sequences, `unsafe` and the checks are
     * compared through what `_next`, `_get_id` and `_add_alloc_deps` say
     * on the graphs below, every entry reached in both modes. */
    idx_info_a = ggml_metal_fusion_info_init(true, 1);
    idx_info_b = ref_ggml_metal_fusion_info_init(true, 1);
    const int n_all = ggml_metal_fusion_info_n_fusions(idx_info_a);
    const int n_ref = ref_ggml_metal_fusion_info_n_fusions(idx_info_b);
    cmp_i("fusion table length", n_all, n_ref);
    char lbl[96];
    for (int i = 0; i < n_all && i < n_ref; i++) {
        snprintf(lbl, sizeof lbl, "fusion[%d] label", i);
        cmp_s(lbl, ggml_metal_fusion_info_label(idx_info_a, i), ref_ggml_metal_fusion_info_label(idx_info_b, i));
    }

    /* NORM / RMS_NORM + MUL (+ ADD): the plain chains, a weight of the wrong
     * width (it broadcasts, so it builds), a non-F32 result, a view packed
     * between the norm and the mul, an intermediate that is an output,
     * and an intermediate with a second consumer; then (v8) the MUL
     * placed overlapping the final ADD, as gallocr may place an
     * intermediate: FULL must still fuse, the overlap being internal. */
    for (int variant = 0; variant < 9; variant++) {
        struct ggml_context * ctx = fusion_ctx();
        const enum ggml_type tx = variant == 3 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, tx, 64, 4);
        struct ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, variant == 2 ? 32 : 64);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        struct ggml_tensor * n = (variant & 1) ? ggml_norm(ctx, x, 1e-5f) : ggml_rms_norm(ctx, x, 1e-5f);
        struct ggml_tensor * wv = variant == 4 ? ggml_reshape_2d(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64), 64, 1) : w;
        struct ggml_tensor * m = ggml_mul(ctx, n, wv);
        struct ggml_tensor * a = ggml_add(ctx, m, b);
        if (variant == 5) ggml_set_output(m);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, a);
        if (variant == 6) ggml_build_forward_expand(gf, ggml_scale(ctx, m, 2.0f));
        if (variant == 7) ggml_build_forward_expand(gf, ggml_add(ctx, ggml_rms_norm(ctx, a, 1e-5f), b));
        fusion_alloc(ctx, buft);
        if (variant == 8) m->data = (char *) a->data + 4;
        char name[32]; snprintf(name, sizeof name, "norm v%d", variant);
        fusion_probe_graph(name, gf);
        ggml_free(ctx);
    }

    /* ADD x N, N up to 9 (past the table's 7), with the addends in the
     * same buffer, in a second buffer (FULL refuses, STRUCTURAL packs),
     * with one addend of a different layout, with a chain broken by
     * reading the previous ADD as src1, and (v4) with no rows at all. */
    for (int variant = 0; variant < 5; variant++) {
        for (int len = 1; len <= 9; len++) {
            struct ggml_context * ctx  = fusion_ctx();
            struct ggml_context * ctx2 = fusion_ctx();
            /* v4: zero rows -- since v0.6.0 `_max` packs 0-element nodes
             * as it packs their non-empty counterparts */
            const int64_t rows = variant == 4 ? 0 : 4;
            struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, rows);
            struct ggml_tensor * c = x;
            for (int k = 0; k < len; k++) {
                struct ggml_context * actx = (variant == 1 && k == len / 2) ? ctx2 : ctx;
                struct ggml_tensor * bk = (variant == 2 && k == len - 1)
                    ? ggml_new_tensor_2d(actx, GGML_TYPE_F32, 32, 1)
                    : ggml_new_tensor_2d(actx, GGML_TYPE_F32, 32, rows);
                c = (variant == 3 && k == 2) ? ggml_add(ctx, bk, c) : ggml_add(ctx, c, bk);
            }
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, c);
            fusion_alloc(ctx, buft);
            fusion_alloc(ctx2, buft);
            char name[32]; snprintf(name, sizeof name, "add v%d x%d", variant, len);
            fusion_probe_graph(name, gf);
            ggml_free(ctx);
            ggml_free(ctx2);
        }
    }

    /* MUL + SIN + SQR + MUL + ADD: the snake, a full-width `a`, a 3-d
     * activation, F16, a wrongly shaped inv_b, the closing ADD's operands
     * swapped, and an ADD that closes on a different x. */
    for (int variant = 0; variant < 7; variant++) {
        struct ggml_context * ctx = fusion_ctx();
        const enum ggml_type tx = variant == 3 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        struct ggml_tensor * x  = variant == 2 ? ggml_new_tensor_3d(ctx, tx, 32, 8, 2) : ggml_new_tensor_2d(ctx, tx, 32, 8);
        struct ggml_tensor * x2 = ggml_new_tensor_2d(ctx, tx, 32, 8);
        struct ggml_tensor * al = variant == 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 8);
        struct ggml_tensor * ib = variant == 4 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 8);
        struct ggml_tensor * m0 = ggml_mul(ctx, x, al);
        struct ggml_tensor * s0 = ggml_sin(ctx, m0);
        struct ggml_tensor * q  = ggml_sqr(ctx, s0);
        struct ggml_tensor * m1 = ggml_mul(ctx, q, ib);
        struct ggml_tensor * out = variant == 5 ? ggml_add(ctx, m1, x) : variant == 6 ? ggml_add(ctx, x2, m1) : ggml_add(ctx, x, m1);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        fusion_alloc(ctx, buft);
        char name[32]; snprintf(name, sizeof name, "snake v%d", variant);
        fusion_probe_graph(name, gf);
        ggml_free(ctx);
    }

    /* GATED_DELTA_NET + CPY: K below, at and above n_tokens, either node
     * an output, a destination of the wrong stride, a source view at the
     * wrong offset, and the attention scores consumed after the copy. */
    for (int variant = 0; variant < 8; variant++) {
        struct ggml_context * ctx = fusion_ctx();
        const int64_t S_v = 16, H = 2, T = 3, B = 1;
        const int64_t K = variant == 1 ? 4 : variant == 2 ? 2 : 1;
        struct ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, B);
        struct ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, B);
        struct ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, B);
        struct ggml_tensor * g  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
        struct ggml_tensor * be = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
        struct ggml_tensor * st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, B);
        struct ggml_tensor * gdn = ggml_gated_delta_net(ctx, q, k, v, g, be, st, K);
        const int64_t D = S_v * S_v * H;
        const int64_t n_written = T < K ? T : K;
        const size_t tail = ggml_row_size(GGML_TYPE_F32, S_v * H * T * B) - (variant == 5 ? ggml_row_size(GGML_TYPE_F32, S_v) : 0);
        struct ggml_tensor * src = ggml_view_1d(ctx, gdn, D * B * n_written, tail);
        struct ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * B + (variant == 4 ? 16 : 0), 8);
        struct ggml_tensor * dst = ggml_view_3d(ctx, cache, D, B, n_written,
                                                ggml_row_size(GGML_TYPE_F32, variant == 4 ? D + 16 : D),
                                                cache->nb[1], 0);
        struct ggml_tensor * cp = ggml_cpy(ctx, src, dst);
        if (variant == 3) ggml_set_output(gdn);
        if (variant == 6) ggml_set_output(cp);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, cp);
        struct ggml_tensor * attn = ggml_view_1d(ctx, gdn, S_v * H * T * B, 0);
        ggml_build_forward_expand(gf, ggml_scale(ctx, attn, variant == 7 ? 0.5f : 1.0f));
        fusion_alloc(ctx, buft);
        char name[32]; snprintf(name, sizeof name, "gdn v%d", variant);
        fusion_probe_graph(name, gf);
        ggml_free(ctx);
    }

    /* NORM / RMS_NORM + SCALE (v0.5.0): both norms, an F16 norm, a scale
     * of something else after the norm, a reshape between them, the norm
     * also consumed elsewhere, a scale_bias (still a SCALE), and -- after
     * allocation -- the norm's input moved to overlap the output, which
     * FULL mode must refuse (`check_memory_ranges`) and STRUCTURAL not.
     * The input is an op there, not a leaf: the memory check skips
     * GGML_OP_NONE sources, so a leaf overlapping proves nothing. v8 hangs
     * a src[1] on the SCALE by hand -- no constructor does, but the check
     * asks. */
    for (int variant = 0; variant < 9; variant++) {
        struct ggml_context * ctx = fusion_ctx();
        const enum ggml_type tx = variant == 2 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        struct ggml_tensor * x0 = ggml_new_tensor_2d(ctx, tx, 64, 4);
        struct ggml_tensor * x = variant == 7 ? ggml_scale(ctx, x0, 1.0f) : x0;
        struct ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        struct ggml_tensor * n = (variant & 1) ? ggml_norm(ctx, x, 1e-5f) : ggml_rms_norm(ctx, x, 1e-5f);
        struct ggml_tensor * in = variant == 4 ? ggml_reshape_2d(ctx, n, 128, 2) : n;
        struct ggml_tensor * sc = variant == 3 ? ggml_scale(ctx, ggml_add(ctx, n, y), 0.5f)
                                : variant == 6 ? ggml_scale_bias(ctx, in, 0.5f, 1.0f)
                                :                ggml_scale(ctx, in, 0.25f + variant);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, sc);
        if (variant == 5) ggml_build_forward_expand(gf, ggml_scale(ctx, n, 3.0f));
        if (variant == 8) sc->src[1] = y;
        fusion_alloc(ctx, buft);
        if (variant == 7) x->data = (char *) sc->data + 16;
        char name[32]; snprintf(name, sizeof name, "norm_scale v%d", variant);
        fusion_probe_graph(name, gf);
        ggml_free(ctx);
    }

    /* SSM_CONV + UNARY(silu) (v0.5.0): silu, gelu (not silu), the conv
     * consumed elsewhere, a scale between them, an output overlapping
     * the conv's input. */
    for (int variant = 0; variant < 5; variant++) {
        struct ggml_context * ctx = fusion_ctx();
        const int64_t d_conv = 4, d_inner = 32, n_t = 3, n_s = 2;
        struct ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_conv - 1 + n_t, d_inner, n_s);
        struct ggml_tensor * cw0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
        struct ggml_tensor * cw = variant == 4 ? ggml_scale(ctx, cw0, 1.0f) : cw0;
        struct ggml_tensor * cv = ggml_ssm_conv(ctx, sx, cw);
        struct ggml_tensor * in = variant == 3 ? ggml_scale(ctx, cv, 2.0f) : cv;
        struct ggml_tensor * un = variant == 1 ? ggml_gelu(ctx, in) : ggml_silu(ctx, in);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, un);
        if (variant == 2) ggml_build_forward_expand(gf, ggml_scale(ctx, cv, 3.0f));
        fusion_alloc(ctx, buft);
        if (variant == 4) cw->data = (char *) un->data + 8;
        char name[32]; snprintf(name, sizeof name, "ssm_conv v%d", variant);
        fusion_probe_graph(name, gf);
        ggml_free(ctx);
    }

    /* MoE routing and output reduction (v0.5.0), built as llama-graph.cpp's
     * `build_moe_ffn` builds them: softmax gating, `ggml_argsort_top_k`,
     * the weights gathered, optionally renormalised by their clamped sum
     * and optionally scaled; then the experts' outputs times the weights,
     * one VIEW per expert expanded first, and an ADD chain. Over k = 1..9
     * (MOE_REDUCE covers 2..8) and the shapes and arms each check refuses:
     *   v0 plain        v1 scale        v2 norm         v3 norm+scale
     *   v4 softmax scale != 1           v5 softmax with a mask
     *   v6 n_expert past 1024           v7 norm without its trailing reshape
     *   v8 adds summed out of order     v9 a view at the wrong offset
     *   v10 non-F32 experts             v11 logits overlapping the ARGSORT,
     *   the routing's second output (FULL; only `outs` makes it one)
     *   v12 one expert view consumed elsewhere
     *   v13 the experts overlapping the reduction's output (FULL)
     *   v14 the MUL and its views overlapping the output: internal, so
     *   FULL must still fuse
     *   v15 the third ADD reading the first view instead of the running
     *   sum: src[1] is right, only the src[0] check refuses it
     *   v16 no tokens (a no-output ubatch): since v0.6.0 both checks
     *   accept it, so the packing does not depend on the batch
     *   v17, v18 the experts, then the weights, a leaf overlapping the
     *   output (FULL): the memory-range check skips a leaf, so since
     *   v0.8.0 only the reduction's own check refuses these
     *   v19, v20 the same, sharing the output's address exactly, which
     *   that check skips too
     *   v21 both leaves, clear of the output: FULL must still fuse
     */
    for (int variant = 0; variant < 22; variant++) {
        for (int k = 1; k <= 9; k++) {
            if (variant >= 4 && k != 4 && k != 8) continue;
            struct ggml_context * ctx = fusion_ctx();
            const int64_t n_embd = 32, n_tokens = variant == 16 ? 0 : 3;
            const int64_t n_expert = variant == 6 ? 1100 : 16;
            if (k > n_expert) { ggml_free(ctx); continue; }
            const int with_norm  = variant == 2 || variant == 3 || variant == 7;
            const int with_scale = variant == 1 || variant == 3;

            struct ggml_tensor * logits0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tokens);
            struct ggml_tensor * logits = variant == 11 ? ggml_scale(ctx, logits0, 1.0f) : logits0;
            struct ggml_tensor * mask   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tokens);
            struct ggml_tensor * probs  = variant == 4 ? ggml_soft_max_ext(ctx, logits, NULL, 0.5f, 0.0f)
                                        : variant == 5 ? ggml_soft_max_ext(ctx, logits, mask, 1.0f, 0.0f)
                                        :                ggml_soft_max(ctx, logits);
            struct ggml_tensor * probs_r = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
            struct ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, k);
            struct ggml_tensor * weights = ggml_get_rows(ctx, probs_r, selected);
            if (with_norm) {
                struct ggml_tensor * wr = ggml_reshape_2d(ctx, weights, k, n_tokens);
                struct ggml_tensor * sum = ggml_sum_rows(ctx, wr);
                sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
                wr = ggml_div(ctx, wr, sum);
                weights = variant == 7 ? wr : ggml_reshape_3d(ctx, wr, 1, k, n_tokens);
            }
            if (with_scale) weights = ggml_scale(ctx, weights, 2.5f);

            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, weights);

            /* the expert outputs: a stand-in for mul_mat_id's [n_embd, k, n_tokens] */
            struct ggml_tensor * emb = ggml_new_tensor_3d(ctx, variant == 10 ? GGML_TYPE_F16 : GGML_TYPE_F32, n_embd, n_expert, n_tokens);
            const int leaf_experts = variant == 17 || variant == 19 || variant == 21;
            const int leaf_weights = variant == 18 || variant == 20 || variant == 21;
            struct ggml_tensor * experts = leaf_experts ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, k, n_tokens)
                                                        : ggml_get_rows(ctx, emb, selected);
            if (variant == 10) experts = ggml_cast(ctx, experts, GGML_TYPE_F16);
            ggml_build_forward_expand(gf, experts);
            struct ggml_tensor * w3 = variant == 7 ? ggml_reshape_3d(ctx, weights, 1, k, n_tokens)
                                    : leaf_weights ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, n_tokens)
                                    : weights;
            struct ggml_tensor * mul = ggml_mul(ctx, experts, variant == 10 ? ggml_cast(ctx, w3, GGML_TYPE_F16) : w3);
            ggml_build_forward_expand(gf, mul);
            struct ggml_tensor * views[16];
            for (int i = 0; i < k; i++) {
                const size_t offs = (size_t) (variant == 9 && i == k - 1 ? i + 1 : i) * mul->nb[1];
                views[i] = ggml_view_2d(ctx, mul, n_embd, n_tokens, mul->nb[2], variant == 9 && i == k - 1 ? 0 : offs);
                ggml_build_forward_expand(gf, views[i]);
            }
            struct ggml_tensor * acc = views[0];
            for (int i = 1; i < k; i++) {
                if (variant == 15 && i == 2) {
                    ggml_build_forward_expand(gf, acc);
                    acc = ggml_add(ctx, views[0], views[i]);
                    continue;
                }
                acc = (variant == 8 && i == 2) ? ggml_add(ctx, views[i], acc) : ggml_add(ctx, acc, views[i]);
            }
            ggml_build_forward_expand(gf, acc);
            if (variant == 12 && k > 1) ggml_build_forward_expand(gf, ggml_scale(ctx, views[1], 2.0f));
            ggml_build_forward_expand(gf, ggml_scale(ctx, acc, 1.5f));
            /* room after the reduction's output: a tensor moved onto it must
             * still end inside the buffer, or `ggml_metal_buffer_get_id`
             * finds no buffer for it and it overlaps nothing */
            if (variant >= 17) ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16384);
            fusion_alloc(ctx, buft);
            if (variant == 11) logits->data = (char *) selected->src[0]->data + 4;
            if (variant == 13) experts->data = (char *) acc->data + 4;
            if (variant == 17) experts->data = (char *) acc->data + 4;
            if (variant == 18) w3->data = (char *) acc->data + 4;
            if (variant == 19) experts->data = acc->data;
            if (variant == 20) w3->data = acc->data;
            if (variant == 14) {
                mul->data = (char *) acc->data + 4;
                for (int i = 0; i < k; i++) views[i]->data = (char *) mul->data + views[i]->view_offs;
            }
            char name[32]; snprintf(name, sizeof name, "moe v%d k%d", variant, k);
            fusion_probe_graph(name, gf);
            ggml_free(ctx);
        }
    }

    /* MUL_MAT + ADD (v0.6.0): the residual added in the few-row MMA store.
     * The weight sits in a buffer marked WEIGHTS, as the model loader marks
     * them; STRUCTURAL packs on the operand and types alone, FULL also
     * wants the MMA path, a same-shape contiguous residual and no overlap
     * but in place. The variants:
     *   v0 Q4_0, 4 rows             v1 the residual in the weights buffer
     *   v2 32 rows (no MMA)         v3 a broadcast residual
     *   v4 Q2_K (no MMA kernel)     v5 F16 src1
     *   v6 add(mm, mm)              v7 an F16 residual
     *   v8 in place on the residual v9 add(r, mm)
     *   v10 the sum over src1       v11 the sum 4 bytes into the residual
     *   v12 the sum over src0       v13 F32 src0, 5 rows (below its 6)
     *   v14 F32 src0, 6 rows        v15 a transposed (non-contiguous) residual
     *   v16 F16 src0, 96 wide (not a whole K step)
     *   v17 two MUL_MAT + ADD in a residual chain
     *   v18 the MUL_MAT also consumed elsewhere
     *   v19 Q8_0 at 9 rows (two tiles: the generic kernel, not per block)
     *   v20 the sum ending exactly where the residual begins, and v21
     *   starting exactly where it ends: adjacent is not overlapping, so
     *   FULL must fuse both -- the edges of `ggml_metal_fusion_overlap` */
    for (int variant = 0; variant < 22; variant++) {
        struct ggml_context * ctx  = fusion_ctx();
        struct ggml_context * wctx = fusion_ctx();
        const int64_t ne01 = 64;
        int64_t ne00 = 256, ne11 = 4;
        enum ggml_type t0 = GGML_TYPE_Q4_0;
        if (variant == 2)  ne11 = 32;
        if (variant == 4)  t0 = GGML_TYPE_Q2_K;
        if (variant == 13) { t0 = GGML_TYPE_F32; ne11 = 5; }
        if (variant == 14) { t0 = GGML_TYPE_F32; ne11 = 6; }
        if (variant == 16) { t0 = GGML_TYPE_F16; ne00 = 96; }
        if (variant == 19) { t0 = GGML_TYPE_Q8_0; ne11 = 9; }

        struct ggml_tensor * w  = ggml_new_tensor_2d(variant == 12 ? ctx : wctx, t0, ne00, ne01);
        if (variant == 12) ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 64); /* keep the weights buffer */
        struct ggml_tensor * x  = ggml_new_tensor_2d(ctx, variant == 5 ? GGML_TYPE_F16 : GGML_TYPE_F32, ne00, ne11);
        /* v20: a tensor the size of the sum, allocated just before the residual */
        struct ggml_tensor * spacer = variant == 20 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne01, ne11) : NULL;
        struct ggml_tensor * r  = variant == 1  ? ggml_new_tensor_2d(wctx, GGML_TYPE_F32, ne01, ne11)
                                : variant == 3  ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne01, 1)
                                : variant == 7  ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne01, ne11)
                                : variant == 15 ? ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne11, ne01))
                                :                 ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne01, ne11);
        struct ggml_tensor * mm = ggml_mul_mat(ctx, w, x);
        struct ggml_tensor * add = variant == 6 ? ggml_add(ctx, mm, mm)
                                 : variant == 8 ? ggml_add_inplace(ctx, r, mm)
                                 : variant == 9 ? ggml_add(ctx, r, mm)
                                 :                ggml_add(ctx, mm, r);
        struct ggml_tensor * out = add;
        if (variant == 17) {
            struct ggml_tensor * w2 = ggml_new_tensor_2d(wctx, t0, ne01, ne01);
            out = ggml_add(ctx, ggml_mul_mat(ctx, w2, add), add);
        }
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_scale(ctx, out, 0.5f));
        if (variant == 18) ggml_build_forward_expand(gf, ggml_scale(ctx, mm, 2.0f));
        fusion_alloc(ctx, buft);
        {
            /* every variant puts something in wctx, so this is never the main buffer */
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
            if (!wbuf) { printf("FAIL: mul_mat_add v%d has no weights buffer\n", variant); fails++; checks++; }
            else {
                ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                if (n_fusion_bufs < 64) fusion_bufs[n_fusion_bufs++] = wbuf;
            }
        }
        if (variant == 10) add->data = x->data;
        if (variant == 11) add->data = (char *) r->data + 4;
        if (variant == 12) add->data = w->data;
        if (variant == 20) {
            if ((char *) spacer->data + ggml_nbytes(add) != (char *) r->data) { printf("FAIL: mul_mat_add v20: the spacer is not adjacent\n"); fails++; checks++; }
            add->data = spacer->data;
        }
        if (variant == 21) add->data = (char *) r->data + ggml_nbytes(r);
        char name[32]; snprintf(name, sizeof name, "mul_mat_add v%d", variant);
        fusion_probe_graph(name, gf);
        ggml_free(ctx);
        ggml_free(wctx);
    }

    /* graph_optimize's reorder (common.cpp): four independent chains of
     * RMS_NORM + MUL (one fused group each) then SCALE, summed, so the
     * pass has nodes to pull forward. In the variants a later node's
     * output is placed over a fused group's *internal* source (the norm
     * between RMS_NORM and MUL) or over its first node's output. Since
     * v0.5.0 a group's internal sources are not ranges it reads, and every
     * fused tensor is a range it writes (`h_add`/`h_check`); these are the
     * cases where that decides whether a node may move. */
    for (int variant = 0; variant < 6; variant++) {
        struct ggml_context * ctx = fusion_ctx();
        struct ggml_tensor * n[4], * m[4], * sc[4];
        for (int c = 0; c < 4; c++) {
            struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
            struct ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
            n[c] = ggml_rms_norm(ctx, x, 1e-5f);
            m[c] = ggml_mul(ctx, n[c], w);
            sc[c] = ggml_scale(ctx, m[c], 0.5f + c);
        }
        struct ggml_tensor * acc = ggml_add(ctx, ggml_add(ctx, sc[0], sc[1]), ggml_add(ctx, sc[2], sc[3]));
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        for (int c = 0; c < 4; c++) ggml_build_forward_expand(gf, sc[c]);
        ggml_build_forward_expand(gf, acc);
        fusion_alloc(ctx, buft);
        switch (variant) {
            case 1: sc[1]->data = n[0]->data; break;                 /* over an internal source */
            case 2: sc[2]->data = (char *) n[1]->data + 64; break;
            case 3: m[2]->data = n[3]->data; break;                  /* a group writing over another's internal */
            case 4: sc[3]->data = m[0]->data; break;                 /* over a group's output */
            case 5: n[2]->data = (char *) sc[0]->data + 32; break;   /* an internal over an earlier output */
            default: break;
        }
        char name[32]; snprintf(name, sizeof name, "reorder v%d", variant);
        fusion_probe_graph(name, gf);
        ggml_free(ctx);
    }

    /* a run longer than GGML_METAL_FUSION_MAX: chained patterns stop at 16 */
    {
        struct ggml_context * ctx = fusion_ctx();
        struct ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
        for (int k = 0; k < 40; k++) c = ggml_add(ctx, c, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4));
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);
        fusion_alloc(ctx, buft);
        fusion_probe_graph("add x40", gf);
        ggml_free(ctx);
    }

    for (int i = 0; i < n_fusion_bufs; i++) ggml_backend_buffer_free(fusion_bufs[i]);

    for (int m = 0; m < 2; m++) {
        for (int i = 0; i < n_ref && i < 32; i++) {
            checks++;
            if (fusion_hits[m][i] == 0) {
                printf("  fusion[%d] (%s) never matched in %s mode -- the probes do not reach it\n",
                       i, ref_ggml_metal_fusion_info_label(idx_info_b, i), m ? "full" : "structural");
                fails++;
            }
        }
    }
    checks++;
    if (backend_deps_seen == 0) {
        printf("  no backend graph_optimize produced an alloc dep\n");
        fails++;
    }
    checks++;
    if (reorders_seen == 0) {
        printf("  graph_optimize reordered no probe graph\n");
        fails++;
    }
    checks++;
    if (deps_seen == 0) {
        printf("  no probe produced an alloc dep\n");
        fails++;
    }

    n_entry = n_ref < 32 ? n_ref : 32;
    fusion_info_diff();
    checks++;
    if (fusion_full_refusals == 0) {
        printf("  no probe matched structurally and was refused in full mode\n");
        fails++;
    }
}

/* The three batched buffer slots (v0.9.0): `set_tensor_n`, `get_tensor_n`
 * and `cpy_tensor_n`. translate-c defaults an omitted slot to null, so an
 * iface that forgot them would still compile and `ggml-backend.cpp` would
 * quietly fall back to one call per tensor -- same bytes, other path. So
 * each side's buffer is asked which slots it has, and then driven through
 * them: a scattered write, a read back, a copy between two buffers of the
 * same type, and a copy from a CPU buffer, which `cpy_tensor_n` must
 * refuse before copying anything. Every byte is compared with what was
 * written, and the two sides' answers with each other. */
static void tensor_n_side(ggml_backend_buffer_type_t bt, long out[16], uint8_t *got, size_t ngot) {
    struct ggml_init_params ip = { 16u * 1024u * 1024u, NULL, true };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *t[4], *u[4];
    for (int i = 0; i < 4; i++) {
        t[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024 + 256*i);
        u[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024 + 256*i);
    }
    ggml_backend_buffer_t buf  = ggml_backend_alloc_ctx_tensors_from_buft(ctx, bt);
    struct ggml_context *ctx2 = ggml_init(ip);
    struct ggml_tensor *w[4], *h[4];
    for (int i = 0; i < 4; i++) {
        w[i] = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 1024 + 256*i);
    }
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors_from_buft(ctx2, bt);
    struct ggml_context *ctx3 = ggml_init(ip);
    for (int i = 0; i < 4; i++) {
        h[i] = ggml_new_tensor_1d(ctx3, GGML_TYPE_F32, 1024 + 256*i);
    }
    ggml_backend_buffer_t bufh = ggml_backend_alloc_ctx_tensors_from_buft(ctx3, ggml_backend_cpu_buffer_type());

    out[0] = buf->iface.set_tensor_n != NULL;
    out[1] = buf->iface.get_tensor_n != NULL;
    out[2] = buf->iface.cpy_tensor_n != NULL;
    out[3] = ggml_backend_buffer_is_host(buf);

    ggml_backend_buffer_clear(buf, 0x11);
    ggml_backend_buffer_clear(buf2, 0x22);

    /* scattered write: tensor i gets bytes [offs[i], offs[i]+sizes[i]) from data at doffs[i] */
    uint8_t data[32768];
    for (size_t i = 0; i < sizeof data; i++) data[i] = (uint8_t) (i*131 + 7);
    size_t offs[4]  = { 0, 64, 1000, 4 };
    size_t sizes[4] = { 4096, 0, 2048, 777*4 };
    size_t doffs[4] = { 100, 0, 9000, 20000 };
    ggml_backend_tensor_set_n(t, data, offs, sizes, doffs, 4);

    /* read back whole tensors */
    size_t roffs[4] = { 0, 0, 0, 0 }, rsizes[4], rdoffs[4];
    size_t at = 0;
    for (int i = 0; i < 4; i++) { rsizes[i] = ggml_nbytes(t[i]); rdoffs[i] = at; at += rsizes[i]; }
    if (at > ngot) at = ngot;
    memset(got, 0, ngot);
    ggml_backend_tensor_get_n((const struct ggml_tensor * const *) t, got, roffs, rsizes, rdoffs, 4);

    long ok = 1;
    for (int i = 0; i < 4; i++) {
        for (size_t b = 0; b < rsizes[i]; b++) {
            uint8_t want = (b >= offs[i] && b < offs[i] + sizes[i]) ? data[doffs[i] + b - offs[i]] : 0x11;
            if (got[rdoffs[i] + b] != want) { ok = 0; break; }
        }
    }
    out[4] = ok;

    /* copy within the Metal buffer type, then read the copies back */
    ggml_backend_tensor_copy_n((const struct ggml_tensor * const *) t, w, 4);
    uint8_t *got2 = got + ngot/2;
    ggml_backend_tensor_get_n((const struct ggml_tensor * const *) w, got2, roffs, rsizes, rdoffs, 4);
    out[5] = memcmp(got, got2, at) == 0;

    /* a CPU source: the Metal `cpy_tensor_n` refuses, the generic path copies */
    for (int i = 0; i < 4; i++) memset(h[i]->data, 0x5a + i, ggml_nbytes(h[i]));
    out[6] = buf->iface.cpy_tensor_n ? buf->iface.cpy_tensor_n(buf, (const struct ggml_tensor * const *) h, t, 4) : -1;
    ggml_backend_tensor_copy_n((const struct ggml_tensor * const *) h, t, 4);
    ggml_backend_tensor_get_n((const struct ggml_tensor * const *) t, got, roffs, rsizes, rdoffs, 4);
    ok = 1;
    for (int i = 0; i < 4; i++) {
        for (size_t b = 0; b < rsizes[i]; b++) {
            if (got[rdoffs[i] + b] != (uint8_t) (0x5a + i)) { ok = 0; break; }
        }
    }
    out[7] = ok;

    /* `cpy_tensor_n` itself, Metal to Metal */
    ggml_backend_buffer_clear(buf2, 0x33);
    out[8] = buf->iface.cpy_tensor_n ? buf2->iface.cpy_tensor_n(buf2, (const struct ggml_tensor * const *) t, w, 4) : -1;
    ggml_backend_tensor_get_n((const struct ggml_tensor * const *) w, got2, roffs, rsizes, rdoffs, 4);
    out[9] = memcmp(got, got2, at) == 0;

    ggml_backend_buffer_free(bufh);
    ggml_backend_buffer_free(buf2);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx3);
    ggml_free(ctx2);
    ggml_free(ctx);
    (void) u;
}

static void tensor_n_diff(ggml_backend_dev_t dev_a, ggml_backend_dev_t dev_b) {
    static uint8_t got_a[65536], got_b[65536];
    static const char * what[10] = {
        "set_tensor_n present", "get_tensor_n present", "cpy_tensor_n present", "is_host",
        "set_tensor_n/get_tensor_n round trip", "tensor_copy_n Metal->Metal",
        "cpy_tensor_n from CPU refused", "tensor_copy_n CPU->Metal",
        "cpy_tensor_n Metal->Metal", "cpy_tensor_n bytes",
    };
    /* the device's own buffer type: shared on this machine, private when
     * the script reruns this part under GGML_METAL_SHARED_BUFFERS_DISABLE
     * (the device reads it once per process), so both ifaces are reached */
    ggml_backend_buffer_type_t bts[2][1] = {
        { ggml_backend_dev_buffer_type(dev_a) },
        { ggml_backend_dev_buffer_type(dev_b) },
    };
    for (int k = 0; k < 1; k++) {
        long a[16] = {0}, b[16] = {0};
        tensor_n_side(bts[0][k], a, got_a, sizeof got_a);
        tensor_n_side(bts[1][k], b, got_b, sizeof got_b);
        for (int i = 0; i < 10; i++) {
            char lbl[96];
            snprintf(lbl, sizeof lbl, "%s %s", ggml_backend_buft_name(bts[1][k]), what[i]);
            cmp_i(lbl, a[i], b[i]);
        }
        /* every functional answer must also be the right one */
        const int must[] = { 4, 5, 7, 9 };
        for (int j = 0; j < 4; j++) {
            char lbl[96];
            snprintf(lbl, sizeof lbl, "%s %s (correct)", ggml_backend_buft_name(bts[1][k]), what[must[j]]);
            cmp_i(lbl, a[must[j]], 1);
        }
    }
}

int main(void) {
    ggml_backend_reg_t reg_a = ggml_backend_metal_reg();
    ggml_backend_reg_t reg_b = ref_ggml_backend_metal_reg();
    if (!reg_a || !reg_b) { printf("FAIL: no Metal registry\n"); return 1; }

    printf("=== Metal registry, device and buffer types: ported vs reference ===\n");

    cmp_s("reg get_name", ggml_backend_reg_name(reg_a), ggml_backend_reg_name(reg_b));
    cmp_sz("reg device_count", ggml_backend_reg_dev_count(reg_a), ggml_backend_reg_dev_count(reg_b));

    ggml_backend_dev_t dev_a = ggml_backend_reg_dev_get(reg_a, 0);
    ggml_backend_dev_t dev_b = ggml_backend_reg_dev_get(reg_b, 0);

    if (getenv("METAL_DIFF_TENSOR_N_ONLY")) {
        tensor_n_diff(dev_a, dev_b);
        printf("\n");
        if (fails == 0) {
            printf("PASS: %d batched buffer-slot answers identical, ported vs reference\n", checks);
            return 0;
        }
        printf("FAIL: %d of %d batched buffer-slot answers differ\n", fails, checks);
        return 1;
    }

    cmp_s("dev get_name", ggml_backend_dev_name(dev_a), ggml_backend_dev_name(dev_b));
    cmp_s("dev get_description", ggml_backend_dev_description(dev_a), ggml_backend_dev_description(dev_b));
    cmp_i("dev get_type", ggml_backend_dev_type(dev_a), ggml_backend_dev_type(dev_b));

    ggml_backend_buffer_type_t bt_a = ggml_backend_dev_buffer_type(dev_a);
    ggml_backend_buffer_type_t bt_b = ggml_backend_dev_buffer_type(dev_b);

    cmp_s("buft get_name", ggml_backend_buft_name(bt_a), ggml_backend_buft_name(bt_b));
    cmp_sz("buft get_alignment", ggml_backend_buft_get_alignment(bt_a), ggml_backend_buft_get_alignment(bt_b));
    cmp_sz("buft get_max_size", ggml_backend_buft_get_max_size(bt_a), ggml_backend_buft_get_max_size(bt_b));
    cmp_i("buft is_host", ggml_backend_buft_is_host(bt_a), ggml_backend_buft_is_host(bt_b));

    /* get_alloc_size over tensors reaching every arm of the switch, plus
     * `offload_op` and `supports_op` over the same set. no_alloc, so the
     * tensors are shapes and ops without any memory behind them. */
    struct ggml_init_params ip = { 64u * 1024u * 1024u, NULL, true };
    struct ggml_context *ctx = ggml_init(ip);
    if (!ctx) { printf("FAIL: no ggml context\n"); return 1; }

    struct ggml_tensor *probes[128];
    const char *labels[128];
    int n = 0;

    /* plain, no extra */
    probes[n] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);          labels[n] = "f32 2d";                 n++;
    probes[n] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 32, 4, 2);     labels[n] = "f16 4d";                 n++;
    probes[n] = ggml_new_tensor_1d(ctx, GGML_TYPE_Q4_K, 256);             labels[n] = "q4_K 1d";                n++;

    /* the arms */
    {
        struct ggml_tensor *a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);
        probes[n] = ggml_cumsum(ctx, a);                                   labels[n] = "CUMSUM";                 n++;
        probes[n] = ggml_argsort(ctx, a, GGML_SORT_ORDER_ASC);             labels[n] = "ARGSORT";                n++;
        probes[n] = ggml_top_k(ctx, a, 8);                                 labels[n] = "TOP_K";                  n++;
    }
    {
        /* MUL_MAT_ID and MUL_MAT across batch sizes that straddle
         * `op_offload_min_batch_size`, which `ggml-metal-device.m:1196`
         * defaults to 32.
         *
         * **The straddle is the point.** `offload_op` compares the batch
         * size against that threshold, and `get_op_batch_size` reads
         * `ne[1]` for MUL_MAT but `ne[2]` for MUL_MAT_ID. A first version
         * of this harness used one MUL_MAT_ID with ne[1]=2, ne[2]=16 --
         * both below 32, so swapping the two dimensions gave the same
         * answer and the injection escaped. The pairs below are chosen so
         * that reading the wrong dimension crosses the threshold. */
        const int64_t n_tok[] = { 1, 2, 64, 512 };
        const int64_t n_exp[] = { 64, 512, 1, 2 };
        for (int i = 0; i < 4; i++) {
            struct ggml_tensor *w   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 256, 128, 8);
            struct ggml_tensor *x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, n_tok[i], n_exp[i]);
            struct ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_tok[i], n_exp[i]);
            struct ggml_tensor *t = ggml_mul_mat_id(ctx, w, x, ids);
            if (t) {
                static char buf[4][48];
                snprintf(buf[i], sizeof buf[i], "MUL_MAT_ID ne1=%lld ne2=%lld",
                         (long long) t->ne[1], (long long) t->ne[2]);
                probes[n] = t; labels[n] = buf[i]; n++;
            }
        }
        for (int i = 0; i < 4; i++) {
            struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 128);
            struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, n_tok[i] * 8);
            struct ggml_tensor *t = ggml_mul_mat(ctx, w, x);
            if (t) {
                static char buf[4][48];
                snprintf(buf[i], sizeof buf[i], "MUL_MAT ne1=%lld", (long long) t->ne[1]);
                probes[n] = t; labels[n] = buf[i]; n++;
            }
        }
    }
    {
        /* FLASH_ATTN_EXT at a few shapes, since the extras are shape
         * dependent and one shape can have them all zero. */
        const int64_t dks[]  = { 64, 128, 128, 576 };
        const int64_t dvs[]  = { 64, 128, 128, 512 };
        const int64_t n11s[] = { 64, 512, 4096, 512 };
        const int64_t n01s[] = {  1,   1,   8,    1 };
        for (int i = 0; i < 4; i++) {
            struct ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dks[i], n01s[i], 8, 1);
            struct ggml_tensor *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dks[i], n11s[i], 8, 1);
            struct ggml_tensor *v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dvs[i], n11s[i], 8, 1);
            struct ggml_tensor *t = ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0f/8.0f, 0.0f, 0.0f);
            if (t) {
                static char buf[4][48];
                snprintf(buf[i], sizeof buf[i], "FLASH_ATTN_EXT dk=%lld ne11=%lld",
                         (long long) dks[i], (long long) n11s[i]);
                probes[n] = t; labels[n] = buf[i]; n++;
            }
        }
    }

    {
        /* sparse FLASH_ATTN_EXT (`n_kv_max` set, a mask present): the
         * vec kernel's (dk, dv) list decides whether the index scratch is
         * allocated -- v0.5.0 added (96, 64) -- and so do the bound and
         * the KV type. One pair off the list, one over the 4096 bound. */
        const int64_t dks[]  = { 96, 96, 128, 80, 128, 576, 128 };
        const int64_t dvs[]  = { 96, 64, 128, 80, 128, 512,  96 };
        const int32_t nkv[]  = { 256, 256, 512, 256, 5000, 1024, 256 };
        for (int i = 0; i < 7; i++) {
            struct ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dks[i], 1, 8, 1);
            struct ggml_tensor *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dks[i], 8192, 8, 1);
            struct ggml_tensor *v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dvs[i], 8192, 8, 1);
            struct ggml_tensor *m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8192, 1, 1, 1);
            struct ggml_tensor *t = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/8.0f, 0.0f, 0.0f);
            if (t) {
                ggml_flash_attn_ext_set_n_kv_max(t, nkv[i]);
                static char buf[7][64];
                snprintf(buf[i], sizeof buf[i], "FLASH_ATTN_EXT sparse dk=%lld dv=%lld n_kv_max=%d",
                         (long long) dks[i], (long long) dvs[i], nkv[i]);
                probes[n] = t; labels[n] = buf[i]; n++;
            }
        }
    }

    {
        /* quantized KV and the upfront dequantise to F16: since v0.7.0 a
         * batch under 32 rows dequantises when the head is 512 or wider
         * and the batch is too big for the vec kernel (20 rows), so the
         * pad scratch is sized for F16 rows there. Rows either side of
         * both thresholds, at head sizes either side of 512. */
        const int64_t dks[]  = { 576, 576, 576, 576, 512, 128, 128 };
        const int64_t dvs[]  = { 512, 512, 512, 512, 512, 128, 128 };
        const int64_t n01s[] = {   8,  19,  24,  32,  24,  24,  32 };
        for (int i = 0; i < 7; i++) {
            struct ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dks[i], n01s[i], 8, 1);
            struct ggml_tensor *k = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, dks[i], 4096, 1, 1);
            struct ggml_tensor *v = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, dvs[i], 4096, 1, 1);
            struct ggml_tensor *m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 4096, n01s[i], 1, 1);
            struct ggml_tensor *t = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/8.0f, 0.0f, 0.0f);
            if (t) {
                static char buf[7][64];
                snprintf(buf[i], sizeof buf[i], "FLASH_ATTN_EXT q8_0 dk=%lld ne01=%lld",
                         (long long) dks[i], (long long) n01s[i]);
                probes[n] = t; labels[n] = buf[i]; n++;
            }
        }
    }

    {
        /* FLASH_ATTN_EXT over K/V row lists (`ggml_flash_attn_ext_rows`,
         * v0.9.0): the gathered-rows scratch is K's rows plus V's, unless V
         * is a view of K; zero without `src[5]`. Several streams (q->ne[3])
         * since the lists are one per stream; the KV types `supports_op`
         * accepts and two it refuses; V a view of K at the same and at a
         * different offset; a batch on each side of the vec threshold; and
         * a sparse hint, which the row lists switch off. */
        const enum ggml_type kt[] = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_BF16,
                                      GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_Q4_1, GGML_TYPE_F16,
                                      GGML_TYPE_F16, GGML_TYPE_F32 };
        const int64_t dks[]  = { 128, 128,  64, 128, 576, 576,  96, 128, 128, 64 };
        const int64_t dvs[]  = { 128, 128,  64, 128, 512, 512,  96, 128,  98, 64 };
        const int64_t n01s[] = {   1,  32,  64,   1,   1,  32,   8,  40,   1, 1 };
        const int64_t n3s[]  = {   1,   2,   3,   1,   1,   2,   4,   1,   1, 1 };
        const int64_t nrws[] = { 256, 512, 768, 256, 256, 512, 256, 1024, 256, 256 };
        /* 0 separate V, 1 V a view of K at K's offset, 2 a view at another offset */
        const int     vkind[] = { 0, 0, 0, 0, 1, 2, 0, 0, 0, 0 };
        const int32_t nkv[]   = { 0, 0, 0, 0, 0, 0, 0, 256, 0, 0 };
        for (int i = 0; i < 10; i++) {
            struct ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dks[i], n01s[i], 8, n3s[i]);
            struct ggml_tensor *k = ggml_new_tensor_4d(ctx, kt[i], dks[i], 4096, 2, 1);
            struct ggml_tensor *v;
            if (vkind[i] == 0) {
                v = ggml_new_tensor_4d(ctx, kt[i], dvs[i], 4096, 2, 1);
            } else {
                v = ggml_view_4d(ctx, k, dvs[i], 4096, 2, 1, k->nb[1], k->nb[2], k->nb[3],
                                 vkind[i] == 1 ? 0 : 64*ggml_type_size(kt[i]));
            }
            struct ggml_tensor *m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nrws[i], n01s[i], 1, n3s[i]);
            struct ggml_tensor *kr = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nrws[i], n3s[i]);
            struct ggml_tensor *t  = ggml_flash_attn_ext_rows(ctx, q, k, v, m, kr, 1.0f/8.0f, 0.0f, 0.0f);
            if (t) {
                if (nkv[i]) ggml_flash_attn_ext_set_n_kv_max(t, nkv[i]);
                static char buf[10][80];
                snprintf(buf[i], sizeof buf[i], "FLASH_ATTN_EXT rows %s dk=%lld dv=%lld ne01=%lld ne3=%lld v=%d nkv=%d",
                         ggml_type_name(kt[i]), (long long) dks[i], (long long) dvs[i], (long long) n01s[i],
                         (long long) n3s[i], vkind[i], nkv[i]);
                probes[n] = t; labels[n] = buf[i]; n++;
            }
        }
    }

    {
        /* TOP_K_MASK (v0.8.0), each optional source on and off: no scratch,
         * and `supports_op`'s arm is the `.m`'s. */
        for (int bits = 0; bits < 16; bits++) {
            const enum ggml_type type = bits & 8 ? GGML_TYPE_F32 : GGML_TYPE_F16;
            struct ggml_tensor *m = ggml_new_tensor_2d(ctx, type, 256, 4);
            struct ggml_tensor *k = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 16, 4);
            struct ggml_tensor *bl = bits & 2 ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 64) : NULL;
            struct ggml_tensor *sc = bits & 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bl ? 64 : 256, 4) : NULL;
            struct ggml_tensor *ex = bits & 4 ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 4) : NULL;
            static char buf[16][48];
            snprintf(buf[bits], sizeof buf[bits], "TOP_K_MASK %s sc=%d bl=%d ex=%d", ggml_type_name(type),
                     !!(bits & 1), !!(bits & 2), !!(bits & 4));
            probes[n] = ggml_top_k_mask(ctx, m, k, sc, bl, ex); labels[n] = buf[bits]; n++;
        }
        /* ARGMAX either side of the two-stage threshold, through every vtable answer */
        const int64_t ne00s[] = { 4096, 4097, 2000000 };
        for (int i = 0; i < 3; i++) {
            static char buf[3][48];
            snprintf(buf[i], sizeof buf[i], "ARGMAX ne00=%lld", (long long) ne00s[i]);
            probes[n] = ggml_argmax(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00s[i], 3)); labels[n] = buf[i]; n++;
        }
    }

    char lbl[128];
    for (int i = 0; i < n; i++) {
        snprintf(lbl, sizeof lbl, "get_alloc_size(%s)", labels[i]);
        cmp_sz(lbl, ggml_backend_buft_get_alloc_size(bt_a, probes[i]),
                    ggml_backend_buft_get_alloc_size(bt_b, probes[i]));

        snprintf(lbl, sizeof lbl, "dev_supports_op(%s)", labels[i]);
        cmp_i(lbl, ggml_backend_dev_supports_op(dev_a, probes[i]),
                   ggml_backend_dev_supports_op(dev_b, probes[i]));

        snprintf(lbl, sizeof lbl, "dev_offload_op(%s)", labels[i]);
        cmp_i(lbl, ggml_backend_dev_offload_op(dev_a, probes[i]),
                   ggml_backend_dev_offload_op(dev_b, probes[i]));

#define CMP_EXTRA(f) snprintf(lbl, sizeof lbl, #f "(%s)", labels[i]); cmp_sz(lbl, f(probes[i]), ref_##f(probes[i]));
        if (probes[i]->op == GGML_OP_MUL_MAT_ID) {
            CMP_EXTRA(ggml_metal_op_mul_mat_id_extra_tpe)
            CMP_EXTRA(ggml_metal_op_mul_mat_id_extra_ids)
        }
        if (probes[i]->op == GGML_OP_ARGMAX) {
            CMP_EXTRA(ggml_metal_op_argmax_extra)
        }
        if (probes[i]->op == GGML_OP_FLASH_ATTN_EXT) {
            CMP_EXTRA(ggml_metal_op_flash_attn_ext_extra_pad)
            CMP_EXTRA(ggml_metal_op_flash_attn_ext_extra_blk)
            CMP_EXTRA(ggml_metal_op_flash_attn_ext_extra_tmp)
            CMP_EXTRA(ggml_metal_op_flash_attn_ext_extra_kv_f16)
            CMP_EXTRA(ggml_metal_op_flash_attn_ext_extra_idx)
            CMP_EXTRA(ggml_metal_op_flash_attn_ext_extra_rows)
            snprintf(lbl, sizeof lbl, "flash_attn_ext_use_vec(%s)", labels[i]);
            cmp_i(lbl, ggml_metal_op_flash_attn_ext_use_vec(probes[i]), ref_ggml_metal_op_flash_attn_ext_use_vec(probes[i]));
        }
#undef CMP_EXTRA
    }

    /* supports_buft both ways: each side must accept its own buffer type
     * and reject the CPU's. */
    cmp_i("dev_supports_buft(own)", ggml_backend_dev_supports_buft(dev_a, bt_a),
                                    ggml_backend_dev_supports_buft(dev_b, bt_b));
    cmp_i("dev_supports_buft(cpu)", ggml_backend_dev_supports_buft(dev_a, ggml_backend_cpu_buffer_type()),
                                    ggml_backend_dev_supports_buft(dev_b, ggml_backend_cpu_buffer_type()));

    ggml_free(ctx);

    /* the five generic fusion-debugging names, new at v0.4.1: each side's
     * registry must hand out an entry point for each, or neither */
    {
        const char * names[] = { "ggml_backend_fusion_get", "ggml_backend_fusion_stats_init",
                                 "ggml_backend_fusion_stats_reset", "ggml_backend_fusion_stats_get",
                                 "ggml_backend_fusion_set_enabled", "ggml_backend_fusion_bogus" };
        for (int i = 0; i < 6; i++) {
            char lbl[96];
            snprintf(lbl, sizeof lbl, "get_proc_address(%s)", names[i]);
            cmp_i(lbl, ggml_backend_reg_get_proc_address(reg_a, names[i]) != NULL,
                       ggml_backend_reg_get_proc_address(reg_b, names[i]) != NULL);
        }
    }

    tensor_n_diff(dev_a, dev_b);
    mma_predicates_diff();
    mul_mat_id_extra_diff(bt_a, bt_b);
    argmax_extra_diff(bt_a, bt_b);

    backend_a = ggml_backend_dev_init(dev_a, NULL);
    backend_b = ggml_backend_dev_init(dev_b, NULL);
    if (!backend_a || !backend_b) { printf("FAIL: could not init both Metal backends\n"); return 1; }

    fusion_diff(bt_a);

    ggml_backend_free(backend_a);
    ggml_backend_free(backend_b);

    printf("\n");
    if (fails == 0) {
        printf("PASS: %d Metal vtable answers identical, ported vs reference\n", checks);
        return 0;
    }
    printf("FAIL: %d of %d Metal vtable answers differ\n", fails, checks);
    return 1;
}
