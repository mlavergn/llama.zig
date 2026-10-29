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

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Ours, and the reference's with its six exports renamed by -D. */
extern ggml_backend_reg_t ggml_backend_metal_reg(void);
extern ggml_backend_reg_t ref_ggml_backend_metal_reg(void);

/* The fusion table: ours (`src/ggml/metal/fusion.zig`, declared by
 * `ggml-metal-fusion.h`) and the reference's `ggml-metal-fusion.cpp` with
 * its 17 exports renamed by -D. Each side's info object is its own struct,
 * so each is only ever handed to its own functions. */
extern const ggml_metal_fusion * ref_ggml_metal_fusion_all(int * n);
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

static int table_index(const ggml_metal_fusion * all, int n, const ggml_metal_fusion * f) {
    if (f == NULL) return -1;
    for (int i = 0; i < n; i++) if (&all[i] == f) return i;
    return -2; /* not in the table at all */
}

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
    int n_all = 0, n_ref = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n_all);
    const ggml_metal_fusion * ref = ref_ggml_metal_fusion_all(&n_ref);

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
            cmp_i(lbl, table_index(all, n_all, fa), table_index(ref, n_ref, fb));
            snprintf(lbl, sizeof lbl, "fusion_next n_out(%s @%d %s)", name, i, m ? "full" : "struct");
            cmp_i(lbl, na, nb);
            const int ib = table_index(ref, n_ref, fb);
            if (ib >= 0 && ib < 32) fusion_hits[m][ib]++;
            if (m == 1) {
                int ns = 0;
                if (ref_ggml_metal_fusion_next(gf, idxs, n_idxs, i, GGML_METAL_FUSION_STRUCTURAL, &ns) != NULL && fb == NULL) {
                    fusion_full_refusals++;
                }
            }
        }
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

static void fusion_diff(ggml_backend_buffer_type_t buft) {
    printf("=== Metal fusion table: ported vs reference ===\n");

    int n_all = 0, n_ref = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n_all);
    const ggml_metal_fusion * ref = ref_ggml_metal_fusion_all(&n_ref);
    cmp_i("fusion_all n", n_all, n_ref);
    char lbl[96];
    for (int i = 0; i < n_all && i < n_ref; i++) {
        snprintf(lbl, sizeof lbl, "fusion[%d] id", i);       cmp_i(lbl, all[i].id, ref[i].id);
        snprintf(lbl, sizeof lbl, "fusion[%d] n_ops", i);    cmp_i(lbl, all[i].n_ops, ref[i].n_ops);
        snprintf(lbl, sizeof lbl, "fusion[%d] unsafe", i);   cmp_i(lbl, all[i].unsafe, ref[i].unsafe);
        snprintf(lbl, sizeof lbl, "fusion[%d] has check", i); cmp_i(lbl, all[i].check != NULL, ref[i].check != NULL);
        for (int j = 0; j < all[i].n_ops && j < ref[i].n_ops; j++) {
            snprintf(lbl, sizeof lbl, "fusion[%d] ops[%d]", i, j); cmp_i(lbl, all[i].ops[j], ref[i].ops[j]);
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
            ggml_metal_fusion_info_count_fusion(a, &all[0]);
            ref_ggml_metal_fusion_info_count_fusion(b, &ref[0]);

            ggml_metal_fusion_info_stats_init(a);
            ref_ggml_metal_fusion_info_stats_init(b);
            ggml_metal_fusion_info_labels_init(a); /* idempotent */
            ref_ggml_metal_fusion_info_labels_init(b);
            ggml_metal_fusion_info_set_enabled(a, !en);
            ref_ggml_metal_fusion_info_set_enabled(b, !en);
            snprintf(lbl, sizeof lbl, "info(%d,%d) set_enabled", en, debug); cmp_i(lbl, ggml_metal_fusion_info_enabled(a), ref_ggml_metal_fusion_info_enabled(b));

            const int nf = ggml_metal_fusion_info_n_fusions(a);
            snprintf(lbl, sizeof lbl, "info(%d,%d) n_fusions after init", en, debug); cmp_i(lbl, nf, ref_ggml_metal_fusion_info_n_fusions(b));
            for (int i = 0; i < nf && i < n_all; i++) {
                for (int k = 0; k <= i % 3; k++) {
                    ggml_metal_fusion_info_count_fusion(a, &all[i]);
                    ref_ggml_metal_fusion_info_count_fusion(b, &ref[i]);
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

    /* NORM / RMS_NORM + MUL (+ ADD): the plain chains, a weight of the wrong
     * width (it broadcasts, so it builds), a non-F32 result, a view packed
     * between the norm and the mul, an intermediate that is an output,
     * and an intermediate with a second consumer. */
    for (int variant = 0; variant < 8; variant++) {
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
        char name[32]; snprintf(name, sizeof name, "norm v%d", variant);
        fusion_probe_graph(name, gf);
        ggml_free(ctx);
    }

    /* ADD x N, N up to 9 (past the table's 7), with the addends in the
     * same buffer, in a second buffer (FULL refuses, STRUCTURAL packs),
     * with one addend of a different layout, and with a chain broken by
     * reading the previous ADD as src1. */
    for (int variant = 0; variant < 4; variant++) {
        for (int len = 1; len <= 9; len++) {
            struct ggml_context * ctx  = fusion_ctx();
            struct ggml_context * ctx2 = fusion_ctx();
            struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
            struct ggml_tensor * c = x;
            for (int k = 0; k < len; k++) {
                struct ggml_context * actx = (variant == 1 && k == len / 2) ? ctx2 : ctx;
                struct ggml_tensor * bk = (variant == 2 && k == len - 1)
                    ? ggml_new_tensor_2d(actx, GGML_TYPE_F32, 32, 1)
                    : ggml_new_tensor_2d(actx, GGML_TYPE_F32, 32, 4);
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
                       i, ggml_op_name(ref[i].ops[0]), m ? "full" : "structural");
                fails++;
            }
        }
    }
    checks++;
    if (fusion_full_refusals == 0) {
        printf("  no probe matched structurally and was refused in full mode\n");
        fails++;
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

    struct ggml_tensor *probes[48];
    const char *labels[48];
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

    fusion_diff(bt_a);

    printf("\n");
    if (fails == 0) {
        printf("PASS: %d Metal vtable answers identical, ported vs reference\n", checks);
        return 0;
    }
    printf("FAIL: %d of %d Metal vtable answers differ\n", fails, checks);
    return 1;
}
