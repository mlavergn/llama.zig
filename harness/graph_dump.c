/* Dumps the structure a ggml op constructor produces, for diffing two builds.
 *
 * Token parity says "the model answered differently" and nothing more. This
 * says "node 47, ggml_rope_ext, op_params[5] differs" -- before any arithmetic
 * runs, so a shape or parameter mistake is named rather than inferred.
 *
 * Built twice by scripts/graph-diff: once against the ported libraries, once
 * against the reference. Identical output means the ported constructors build
 * the same graphs the C ones do.
 *
 * Scope is the constructors that *compute* something -- a shape, a stride, an
 * op_params slot. Pure getters are covered by unit tests against golden values
 * taken from the C, which is a cross-check this cannot improve on.
 */

#include "ggml.h"

#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static struct ggml_context * ctx;

/* Prints everything a backend reads off a node. op_params is dumped whole
 * because slot *position* is what goes wrong -- a value in the wrong index
 * looks fine in isolation. */
static void dump(const char * label, const struct ggml_tensor * t) {
    printf("%-28s op=%-22s type=%-8s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]",
           label, ggml_op_name(t->op), ggml_type_name(t->type),
           (long long) t->ne[0], (long long) t->ne[1],
           (long long) t->ne[2], (long long) t->ne[3],
           t->nb[0], t->nb[1], t->nb[2], t->nb[3]);

    printf(" params=");
    for (size_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        printf("%08x", ((const int32_t *) t->op_params)[i]);
    }

    /* Which inputs are wired, and in which slot. ggml_set_rows deliberately
     * uses an unusual order, so the pattern itself is worth recording. */
    printf(" src=");
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        printf("%c", t->src[i] ? '1' : '0');
    }

    printf(" view=%c flags=%d\n", t->view_src ? '1' : '0', t->flags);
}

#define D(expr) dump(#expr, (expr))

/* Runs `fn` in a forked child and reports whether it aborted: the C's
 * GGML_ASSERTs are part of what a constructor does, and an assertion the port
 * dropped would let it build a node the C refuses. */
static void dies(const char * label, void (*fn)(void)) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        fn();
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    printf("%-28s %s\n", label, WIFSIGNALED(st) ? "aborted" : "returned");
}

/* v0.7.0 (#30135): a negative offset passed through `size_t` used to wrap
 * into an out-of-bounds `int32_t` slot; `ggml_acc_impl` now asserts it. */
static void acc_negative_offset(void) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_acc(ctx, a, b, 32, 128, 128, (size_t) -4);
}
static void acc_offset_limit(void) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_acc(ctx, a, b, 32, 128, 128, (size_t) 1 << 30);
}
static void acc_offset_below_limit(void) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_acc_inplace(ctx, a, b, 32, 128, 128, ((size_t) 1 << 30) - 4);
}

/* v0.8.0: `ggml_scale_impl` asserts `ggml_is_contiguous_rows`, where it had
 * asserted `ggml_is_padded_1d`. A permutation that keeps dim 0 now builds; one
 * that moves it does not. */
static void scale_permuted_rows(void) {
    ggml_scale(ctx, ggml_permute(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 2), 0, 2, 1, 3), 2.0f);
}
static void scale_transposed(void) {
    ggml_scale(ctx, ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4)), 2.0f);
}

/* Every GGML_ASSERT of `ggml_top_k_mask`, each from a call that is otherwise
 * valid. */
#define TKM_MASK(t)  ggml_new_tensor_4d(ctx, (t), 64, 2, 1, 1)
#define TKM_TOP()    ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 8, 4, 3, 2)
#define TKM_SCORE(t) ggml_new_tensor_4d(ctx, (t), 16, 4, 3, 2)
#define TKM_BLOCKS() ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 16)
#define TKM_EXTRA()  ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2, 4, 3, 2)
static void tkm_ok(void)            { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), TKM_SCORE(GGML_TYPE_F32), TKM_BLOCKS(), TKM_EXTRA()); }
static void tkm_mask_type(void)     { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_BF16), TKM_TOP(), NULL, NULL, NULL); }
static void tkm_mask_strided(void)  { ggml_top_k_mask(ctx, ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 64)), TKM_TOP(), NULL, NULL, NULL); }
static void tkm_top_type(void)      { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 4, 3, 2), NULL, NULL, NULL); }
static void tkm_bcast1(void)        { ggml_top_k_mask(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 3, 1, 1), TKM_TOP(), NULL, NULL, NULL); }
static void tkm_bcast2(void)        { ggml_top_k_mask(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 2, 1), TKM_TOP(), NULL, NULL, NULL); }
static void tkm_bcast3(void)        { ggml_top_k_mask(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 1, 3), TKM_TOP(), NULL, NULL, NULL); }
static void tkm_score_type(void)    { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), TKM_SCORE(GGML_TYPE_F16), NULL, NULL); }
static void tkm_score_ne1(void)     { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 5, 3, 2), NULL, NULL); }
static void tkm_score_ne2(void)     { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 4, 1, 2), NULL, NULL); }
static void tkm_score_ne3(void)     { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 4, 3, 1), NULL, NULL); }
static void tkm_blocks_type(void)   { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), NULL, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 16), NULL); }
static void tkm_blocks_ne2(void)    { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), NULL, ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, 16, 2), NULL); }
static void tkm_blocks_ne3(void)    { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), NULL, ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 4, 16, 1, 2), NULL); }
static void tkm_blocks_score(void)  { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), TKM_SCORE(GGML_TYPE_F32), ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 15), NULL); }
static void tkm_extra_type(void)    { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), NULL, NULL, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 4, 3, 2)); }
static void tkm_extra_ne1(void)     { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), NULL, NULL, ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2, 5, 3, 2)); }
static void tkm_extra_ne2(void)     { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), NULL, NULL, ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2, 4, 1, 2)); }
static void tkm_extra_ne3(void)     { ggml_top_k_mask(ctx, TKM_MASK(GGML_TYPE_F32), TKM_TOP(), NULL, NULL, ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2, 4, 3, 1)); }

/* `ggml_prec_get_src` asserts its index; `idx` is out of range on each side. */
static struct ggml_tensor * prec_node;
static void prec_idx_negative(void) { ggml_prec_get_src(prec_node, -1); }
static void prec_idx_past_end(void) { ggml_prec_get_src(prec_node, GGML_MAX_SRC); }

/* v0.9.0: `ggml_new_tensor_impl` takes a view's strides and bounds the view by
 * the bytes those strides reach (`ggml_nbytes`), where it had bounded it by
 * its element count. Overlapping rows near the end of the source now build;
 * gapped rows that run past it, and an element count that fits while the
 * strides do not, no longer do. `ggml_view_impl` also refuses strides over a
 * source that is not contiguous along dim 0. */
static void view_overlap_tail(void) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    /* 4 rows of 8 at a 4-float stride: 128 bytes of elements, 80 reached */
    ggml_view_2d(ctx, a, 8, 4, 16, 32);
}
static void view_gap_past_end(void) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    /* 2 rows of 8 at a 112-byte stride: 64 bytes of elements, 144 reached */
    ggml_view_2d(ctx, a, 8, 2, 112, 0);
}
static void view_gap_fits(void) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_view_2d(ctx, a, 8, 2, 64, 0);
}
static void view_transposed_src(void) {
    struct ggml_tensor * a = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4));
    ggml_view_2d(ctx, a, 2, 2, 8, 0);
}
static void view_1d_transposed_src(void) {
    struct ggml_tensor * a = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4));
    ggml_view_1d(ctx, a, 4, 0);
}
static void view_4d_past_end(void) {
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 4, 4, 2, 2);
    ggml_view_4d(ctx, a, 4, 4, 2, 2, 8, 32, 72, 0);
}

/* v0.9.0: `ggml_roll` bounds each shift on both sides instead of through
 * `abs`, which is undefined for INT_MIN. */
static void roll_int_min(void) {
    ggml_roll(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4), -2147483647 - 1, 0, 0, 0);
}
static void roll_neg_edge(void) {
    ggml_roll(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4), -7, -3, 0, 0);
}
static void roll_neg_ne(void) {
    ggml_roll(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4), 0, -4, 0, 0);
}

/* v0.9.0: every GGML_ASSERT `ggml_flash_attn_ext_impl` adds for `kv_rows`,
 * each from a call that is otherwise valid. */
#define FAR_Q()    ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 4, 8, 3)
#define FAR_KV()   ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 40, 2, 1)
#define FAR_MASK() ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 4, 1, 3)
#define FAR_ROWS() ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 32, 3)
static void far_ok(void)         { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), FAR_KV(), FAR_MASK(), FAR_ROWS(), 0.125f, 0.0f, 0.0f); }
static void far_no_mask(void)    { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), FAR_KV(), NULL, FAR_ROWS(), 0.125f, 0.0f, 0.0f); }
static void far_mask_ne0(void)   { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), FAR_KV(), FAR_MASK(), ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 16, 3), 0.125f, 0.0f, 0.0f); }
static void far_rows_type(void)  { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), FAR_KV(), FAR_MASK(), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 3), 0.125f, 0.0f, 0.0f); }
static void far_rows_strided(void) { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), FAR_KV(), FAR_MASK(), ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 32)), 0.125f, 0.0f, 0.0f); }
static void far_rows_ne1(void)   { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), FAR_KV(), FAR_MASK(), ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 32, 2), 0.125f, 0.0f, 0.0f); }
static void far_rows_ne2(void)   { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), FAR_KV(), FAR_MASK(), ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 32, 3, 2), 0.125f, 0.0f, 0.0f); }
static void far_rows_ne3(void)   { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), FAR_KV(), FAR_MASK(), ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 32, 3, 1, 2), 0.125f, 0.0f, 0.0f); }
static void far_k_ne3(void)      { ggml_flash_attn_ext_rows(ctx, FAR_Q(), ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 40, 2, 3), FAR_KV(), FAR_MASK(), FAR_ROWS(), 0.125f, 0.0f, 0.0f); }
static void far_v_ne3(void)      { ggml_flash_attn_ext_rows(ctx, FAR_Q(), FAR_KV(), ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 40, 2, 3), FAR_MASK(), FAR_ROWS(), 0.125f, 0.0f, 0.0f); }
/* without kv_rows the q/k ne[3] check stays as it was */
static void fa_k_ne3(void)       { ggml_flash_attn_ext(ctx, FAR_Q(), FAR_KV(), FAR_KV(), NULL, 0.125f, 0.0f, 0.0f); }

/* ggml-backend-impl.h: which ops `get_alloc_size` may size past `ggml_nbytes`. */
bool ggml_op_alloc_size_may_expand(enum ggml_op op);

int main(void) {
    struct ggml_init_params p = { 256u * 1024u * 1024u, NULL, true /*no_alloc*/ };
    ctx = ggml_init(p);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    /* Fixed shapes, chosen to satisfy each constructor's assertions and to make
     * a wrong dimension visible rather than coincidentally right. */
    struct ggml_tensor * a2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    struct ggml_tensor * b2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    struct ggml_tensor * row  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    struct ggml_tensor * a3   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 2);
    struct ggml_tensor * a4   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 8, 2);
    struct ggml_tensor * sq   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    struct ggml_tensor * vec  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    struct ggml_tensor * scal = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    puts("=== tensor creation ===");
    D(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32));
    D(ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 4));
    D(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 8, 4, 2));
    D(ggml_new_tensor_4d(ctx, GGML_TYPE_Q6_K, 256, 4, 2, 1));
    D(ggml_dup_tensor(ctx, a2));
    D(ggml_view_tensor(ctx, a2));

    puts("=== elementwise ===");
    D(ggml_add(ctx, a2, b2));
    D(ggml_add(ctx, a2, row));          /* broadcast */
    D(ggml_add_inplace(ctx, a2, b2));
    D(ggml_add1(ctx, a2, scal));
    D(ggml_acc(ctx, a2, vec, 32, 64, 96, 16));
    dies("ggml_acc offset -4", acc_negative_offset);
    dies("ggml_acc offset 1<<30", acc_offset_limit);
    dies("ggml_acc offset (1<<30)-4", acc_offset_below_limit);
    D(ggml_sub(ctx, a2, b2));
    D(ggml_mul(ctx, a2, b2));
    D(ggml_div(ctx, a2, b2));
    D(ggml_sqr(ctx, a2));
    D(ggml_sqrt(ctx, a2));
    D(ggml_log(ctx, a2));
    D(ggml_sin(ctx, a2));
    D(ggml_cos(ctx, a2));

    puts("=== reductions ===");
    D(ggml_sum(ctx, a3));
    D(ggml_sum_rows(ctx, a3));
    D(ggml_mean(ctx, a3));
    D(ggml_argmax(ctx, a2));
    D(ggml_count_equal(ctx, a2, b2));
    D(ggml_cumsum(ctx, a2));

    puts("=== broadcast and concat ===");
    D(ggml_repeat(ctx, row, a2));
    D(ggml_repeat_4d(ctx, a2, 16, 8, 1, 1));
    D(ggml_repeat_back(ctx, a2, row));
    D(ggml_concat(ctx, a3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 3), 2));

    puts("=== activations ===");
    D(ggml_relu(ctx, a2));
    D(ggml_gelu(ctx, a2));
    D(ggml_silu(ctx, a2));
    D(ggml_tanh(ctx, a2));
    D(ggml_sigmoid(ctx, a2));
    D(ggml_hardswish(ctx, a2));
    D(ggml_relu_inplace(ctx, a2));
    D(ggml_xielu(ctx, a2, 0.5f, 1.5f, 0.25f, 1e-6f));
    D(ggml_leaky_relu(ctx, a2, 0.01f, false));
    D(ggml_silu_back(ctx, a2, b2));

    puts("=== gated linear units ===");
    D(ggml_swiglu(ctx, a2));
    D(ggml_swiglu_swapped(ctx, a2));
    D(ggml_swiglu_split(ctx, a2, b2));
    D(ggml_geglu(ctx, a2));
    D(ggml_reglu(ctx, a2));
    D(ggml_geglu_erf(ctx, a2));
    D(ggml_geglu_quick(ctx, a2));
    D(ggml_swiglu_oai(ctx, a2, b2, 1.702f, 7.0f));

    puts("=== normalisation ===");
    D(ggml_norm(ctx, a2, 1e-5f));
    D(ggml_rms_norm(ctx, a2, 1e-5f));
    D(ggml_rms_norm_back(ctx, a2, b2, 1e-5f));
    D(ggml_group_norm(ctx, a4, 4, 1e-6f));
    D(ggml_l2_norm(ctx, a2, 1e-5f));

    puts("=== matmul and scale ===");
    D(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 16), a2));
    D(ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 16),
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4)));
    D(ggml_out_prod(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8),
                         ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 8)));
    D(ggml_scale(ctx, a2, 2.5f));
    D(ggml_scale_bias(ctx, a2, 2.5f, -1.25f));
    D(ggml_scale(ctx, ggml_permute(ctx, a3, 0, 2, 1, 3), 2.5f));
    dies("scale permuted rows", scale_permuted_rows);
    dies("scale transposed", scale_transposed);

    puts("=== layout ===");
    D(ggml_reshape_1d(ctx, a2, 32));
    D(ggml_reshape_2d(ctx, a2, 4, 8));
    D(ggml_reshape_3d(ctx, a2, 4, 4, 2));
    D(ggml_reshape_4d(ctx, a2, 2, 2, 2, 4));
    D(ggml_cont(ctx, a2));
    D(ggml_cont_2d(ctx, a2, 4, 8));
    D(ggml_cast(ctx, a2, GGML_TYPE_F16));
    D(ggml_cpy(ctx, a2, b2));
    D(ggml_view_1d(ctx, a2, 8, 32));
    D(ggml_view_2d(ctx, a2, 8, 2, a2->nb[1], a2->nb[1]));
    D(ggml_view_3d(ctx, a3, 8, 2, 2, a3->nb[1], a3->nb[2], 0));
    D(ggml_view_4d(ctx, a4, 16, 8, 4, 1, a4->nb[1], a4->nb[2], a4->nb[3], 0));
    D(ggml_permute(ctx, a4, 1, 2, 0, 3));
    D(ggml_transpose(ctx, a2));
    D(ggml_set(ctx, a2, vec, 32, 64, 96, 16));
    D(ggml_set_1d(ctx, a2, vec, 16));
    D(ggml_set_2d(ctx, a2, vec, 32, 16));

    puts("=== gather and mask ===");
    {
        struct ggml_tensor * tbl = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 100);
        struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7);
        D(ggml_get_rows(ctx, tbl, ids));
        struct ggml_tensor * dst  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 100);
        struct ggml_tensor * data = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        struct ggml_tensor * idx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
        D(ggml_set_rows(ctx, dst, data, idx));
    }
    D(ggml_diag(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)));
    D(ggml_diag_mask_inf(ctx, sq, 3));
    D(ggml_diag_mask_zero(ctx, sq, 3));
    D(ggml_clamp(ctx, a2, -1.5f, 2.5f));

    puts("=== softmax ===");
    {
        struct ggml_tensor * logits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 4);
        struct ggml_tensor * mask   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 32);
        D(ggml_soft_max(ctx, logits));
        D(ggml_soft_max_ext(ctx, logits, mask, 0.125f, 8.0f));
        D(ggml_soft_max_ext_back(ctx, logits, logits, 0.125f, 8.0f));
    }

    puts("=== rope ===");
    {
        struct ggml_tensor * q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 32, 6);
        struct ggml_tensor * pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
        struct ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        int sections[4] = { 16, 24, 24, 0 };
        D(ggml_rope(ctx, q, pos, 128, GGML_ROPE_TYPE_NEOX));
        D(ggml_rope_ext(ctx, q, pos, freqs, 128, GGML_ROPE_TYPE_NEOX, 4096,
                        500000.0f, 0.5f, 1.0f, 0.75f, 32.0f, 1.0f));
        D(ggml_rope_ext_back(ctx, q, pos, NULL, 128, GGML_ROPE_TYPE_NEOX, 4096,
                             10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f));
        struct ggml_tensor * mpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 24);
        D(ggml_rope_multi(ctx, q, mpos, NULL, 128, sections, GGML_ROPE_TYPE_MROPE,
                          4096, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f));
        struct ggml_tensor * r = ggml_rope(ctx, q, pos, 128, GGML_ROPE_TYPE_NEOX);
        ggml_rope_set_offset(r, 17);
        dump("ggml_rope_set_offset", r);

        float dims[2];
        ggml_rope_yarn_corr_dims(128, 4096, 10000.0f, 32.0f, 1.0f, dims);
        printf("%-28s dims=[%.6f,%.6f]\n", "ggml_rope_yarn_corr_dims", dims[0], dims[1]);
    }

    puts("=== convolution and pooling ===");
    {
        struct ggml_tensor * k2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 4, 8);
        struct ggml_tensor * im = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 4, 2);
        struct ggml_tensor * k1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 4, 8);
        struct ggml_tensor * sg = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 4, 1);
        D(ggml_im2col(ctx, k2, im, 1, 1, 1, 1, 1, 1, true, GGML_TYPE_F16));
        D(ggml_im2col(ctx, k2, im, 2, 2, 1, 1, 1, 1, true, GGML_TYPE_F16));
        D(ggml_conv_1d(ctx, k1, sg, 1, 1, 1));
        D(ggml_conv_1d_ph(ctx, k1, sg, 1, 1));
        /* batched (v0.7.0, #28738): the matmul's rows are [OC, N, OL] and a
         * permute plus a cont bring them to [N, OC, OL]; the permute's
         * strides are what a wrong axis order would change */
        struct ggml_tensor * sg3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 4, 3);
        D(ggml_conv_1d(ctx, k1, sg3, 1, 1, 1));
        D(ggml_conv_1d(ctx, k1, sg3, 2, 0, 1)->src[0]);
        D(ggml_conv_1d(ctx, k1, sg3, 1, 2, 2)->src[0]->src[0]);
        D(ggml_conv_1d_ph(ctx, k1, sg3, 1, 1));
        /* depthwise: one kernel per channel, so ne[1] must be 1 */
        struct ggml_tensor * k1dw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 1, 4);
        D(ggml_conv_1d_dw(ctx, k1dw, sg, 1, 1, 1));
        D(ggml_conv_1d_dw_ph(ctx, k1dw, sg, 1, 1));
        D(ggml_conv_2d(ctx, k2, im, 1, 1, 1, 1, 1, 1));
        D(ggml_conv_2d_direct(ctx, k2, im, 1, 1, 1, 1, 1, 1));
        D(ggml_conv_2d_sk_p0(ctx, k2, im));
        D(ggml_conv_2d_s1_ph(ctx, k2, im));
        /* depthwise 2d: kernel [KW,KH,1,C] over an image with C channels */
        struct ggml_tensor * k2dw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 1, 8);
        struct ggml_tensor * im8  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 8, 1);
        D(ggml_conv_2d_dw(ctx, k2dw, im8, 1, 1, 1, 1, 1, 1));
        D(ggml_conv_2d_dw_direct(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, 8),
                                      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 8, 1),
                                 1, 1, 1, 1, 1, 1));
        D(ggml_col2im_1d(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 10), 2, 8, 1));
        D(ggml_conv_transpose_1d(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 4),
                                      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 4), 2, 0, 1));
        D(ggml_conv_transpose_2d_p0(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 4, 4, 8, 16),
                                         ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 16, 1), 2));
        D(ggml_im2col_3d(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 3, 8),
                              ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 16, 2),
                         2, 1, 1, 1, 1, 1, 1, 1, 1, 1, GGML_TYPE_F16));
        D(ggml_pool_1d(ctx, a4, GGML_OP_POOL_AVG, 4, 4, 0));
        D(ggml_pool_2d(ctx, a4, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0));
        D(ggml_pool_2d_back(ctx, ggml_pool_2d(ctx, a4, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0),
                            a4, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0));
    }

    puts("=== resampling and padding ===");
    D(ggml_upscale(ctx, a4, 2, GGML_SCALE_MODE_NEAREST));
    D(ggml_upscale_ext(ctx, a4, 24, 20, 8, 2, GGML_SCALE_MODE_BILINEAR));
    D(ggml_interpolate(ctx, a4, 8, 8, 8, 2,
                       GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS));
    D(ggml_pad(ctx, a4, 2, 3, 0, 0));
    D(ggml_pad_ext(ctx, a4, 1, 2, 3, 4, 0, 0, 0, 0));
    D(ggml_pad_circular(ctx, a4, 1, 0, 0, 0));
    D(ggml_pad_ext_circular(ctx, a4, 1, 1, 0, 0, 0, 0, 0, 0));
    D(ggml_pad_reflect_1d(ctx, a2, 3, 5));

    puts("=== sorting, generation, windows ===");
    D(ggml_argsort(ctx, a2, GGML_SORT_ORDER_DESC));
    D(ggml_argsort_top_k(ctx, a2, 4));
    D(ggml_top_k(ctx, a2, 4));

    puts("=== top_k_mask ===");
    {
        /* Each optional source present and absent, over both mask types and a
         * mask that broadcasts: `src=` is the pattern to read. */
        static const enum ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16 };
        for (int ti = 0; ti < 2; ti++) {
            for (int bits = 0; bits < 8; bits++) {
                struct ggml_tensor * mask = ggml_new_tensor_4d(ctx, types[ti], 64, bits & 1 ? 4 : 2, bits & 1 ? 3 : 1, 1);
                char label[64];
                snprintf(label, sizeof(label), "top_k_mask %s score=%d blocks=%d extra=%d",
                         ggml_type_name(types[ti]), bits & 1, !!(bits & 2), !!(bits & 4));
                dump(label, ggml_top_k_mask(ctx, mask, TKM_TOP(),
                                            bits & 1 ? TKM_SCORE(GGML_TYPE_F32) : NULL,
                                            bits & 2 ? TKM_BLOCKS() : NULL,
                                            bits & 4 ? TKM_EXTRA() : NULL));
            }
        }
        dies("top_k_mask valid", tkm_ok);
        dies("top_k_mask mask type", tkm_mask_type);
        dies("top_k_mask mask strided", tkm_mask_strided);
        dies("top_k_mask top_k type", tkm_top_type);
        dies("top_k_mask mask ne1", tkm_bcast1);
        dies("top_k_mask mask ne2", tkm_bcast2);
        dies("top_k_mask mask ne3", tkm_bcast3);
        dies("top_k_mask score type", tkm_score_type);
        dies("top_k_mask score ne1", tkm_score_ne1);
        dies("top_k_mask score ne2", tkm_score_ne2);
        dies("top_k_mask score ne3", tkm_score_ne3);
        dies("top_k_mask blocks type", tkm_blocks_type);
        dies("top_k_mask blocks ne2", tkm_blocks_ne2);
        dies("top_k_mask blocks ne3", tkm_blocks_ne3);
        dies("top_k_mask blocks score", tkm_blocks_score);
        dies("top_k_mask extra type", tkm_extra_type);
        dies("top_k_mask extra ne1", tkm_extra_ne1);
        dies("top_k_mask extra ne2", tkm_extra_ne2);
        dies("top_k_mask extra ne3", tkm_extra_ne3);
    }
    D(ggml_arange(ctx, 0.0f, 10.0f, 2.5f));
    D(ggml_timestep_embedding(ctx, vec, 64, 10000));
    D(ggml_roll(ctx, a2, 3, -2, 0, 0));
    D(ggml_tri(ctx, sq, GGML_TRI_TYPE_LOWER));
    D(ggml_fill(ctx, a2, -1.5f));
    D(ggml_win_part(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 14, 14, 1), 8));
    D(ggml_win_unpart(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 8, 8, 4), 14, 14, 8));
    D(ggml_get_rel_pos(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 15), 8, 8));

    puts("=== hyper-connections (dsv4) ===");
    {
        /* hc_pre's two forms share GGML_OP_DSV4_HC_PRE and differ only in
         * op_params -- scale in slot 0, gated in slot 1 -- and in the
         * weights' shape; hc_post's comb may be NULL since v0.5.0, which
         * shows in src=. The scales are chosen so a slot swap or a dropped
         * sign is visible: 1.0 is what the plain form writes. */
        const int hc = 4, nt = 5, ne = 8;
        struct ggml_tensor * x3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt);
        D(ggml_dsv4_hc_comb(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (2 + hc) * hc, nt), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3),
                            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (2 + hc) * hc), 1e-6f, 4));
        D(ggml_dsv4_hc_pre(ctx, x3, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt)));
        D(ggml_dsv4_hc_pre_gated(ctx, x3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt), 0.75f));
        D(ggml_dsv4_hc_pre_gated(ctx, x3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt), -2.5f));
        D(ggml_dsv4_hc_pre_gated(ctx, x3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne, hc, nt), 1.0f));
        D(ggml_dsv4_hc_post(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nt), x3,
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, nt)));
        D(ggml_dsv4_hc_post(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, nt), x3,
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt), NULL));
    }

    puts("=== flags ===");
    {
        struct ggml_tensor * f = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_set_input(f);   dump("ggml_set_input", f);
        struct ggml_tensor * v = ggml_reshape_2d(ctx, f, 4, 8);
        ggml_set_output(v);  dump("ggml_set_output(view)", v);
        dump("ggml_set_output(base)", f);
        struct ggml_tensor * leaf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_param(leaf); dump("ggml_set_param", leaf);
        struct ggml_tensor * loss = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_loss(loss);  dump("ggml_set_loss", loss);
    }

    puts("=== precision (ggml_prec_set_acc, ggml_prec_set_src) ===");
    {
        /* Each op gets a fresh node; every return value is printed beside the
         * node, so a refusal arm that writes anyway shows in the params and a
         * refusal reported as success shows in the return. */
        const enum ggml_prec precs[] = { GGML_PREC_UNDEFINED, GGML_PREC_F32, GGML_PREC_BF16,
                                         GGML_PREC_F16, GGML_PREC_Q8, GGML_PREC_Q4 };
        struct ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 16);
        struct ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4);
        struct ggml_tensor * bid = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 3);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3);
        struct ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 4, 8, 1);
        struct ggml_tensor * kk  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 32, 2, 1);
        struct ggml_tensor * vv  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 32, 2, 1);
        for (int p = 0; p < (int) (sizeof(precs) / sizeof(precs[0])); p++) {
            struct ggml_tensor * nodes[4] = {
                ggml_mul_mat(ctx, w, a2),
                ggml_mul_mat_id(ctx, as, bid, ids),
                ggml_flash_attn_ext(ctx, q, kk, vv, NULL, 0.125f, 0.0f, 0.0f),
                ggml_add(ctx, a2, b2),
            };
            for (int n = 0; n < 4; n++) {
                char label[64];
                const bool acc = ggml_prec_set_acc(nodes[n], precs[p]);
                snprintf(label, sizeof(label), "prec_set_acc(%d)=%d", (int) precs[p], acc);
                dump(label, nodes[n]);
                for (int idx = 0; idx < GGML_MAX_SRC; idx += 3) {
                    const bool src = ggml_prec_set_src(nodes[n], precs[p], idx);
                    snprintf(label, sizeof(label), "prec_set_src(%d,%d)=%d", (int) precs[p], idx, src);
                    dump(label, nodes[n]);
                }
                const bool src1 = ggml_prec_set_src(nodes[n], precs[p], 1);
                snprintf(label, sizeof(label), "prec_set_src(%d,1)=%d", (int) precs[p], src1);
                dump(label, nodes[n]);

                /* v0.8.0's getters read back what the setters above left. */
                printf("prec_get_acc=%d prec_get_src=", (int) ggml_prec_get_acc(nodes[n]));
                for (int idx = 0; idx < GGML_MAX_SRC; idx++) {
                    printf("%d,", (int) ggml_prec_get_src(nodes[n], idx));
                }
                printf("\n");
            }
        }
        prec_node = ggml_mul_mat(ctx, w, a2);
        dies("prec_get_src(-1)", prec_idx_negative);
        dies("prec_get_src(GGML_MAX_SRC)", prec_idx_past_end);
    }

    puts("=== v0.9.0: strided views ===");
    {
        struct ggml_tensor * v4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 4, 4, 2, 2);
        D(ggml_view_2d(ctx, a2, 8, 4, 16, 32));
        D(ggml_view_2d(ctx, a2, 4, 2, 64, 16));
        D(ggml_view_3d(ctx, a3, 4, 2, 2, 64, 128, 8));
        D(ggml_view_4d(ctx, v4, 2, 4, 2, 1, 8, 32, 64, 64));
        D(ggml_view_4d(ctx, v4, 4, 2, 2, 2, 8, 64, 32, 0));
        D(ggml_view_tensor(ctx, ggml_permute(ctx, a3, 1, 0, 2, 3)));
        D(ggml_view_1d(ctx, ggml_transpose(ctx, a2), 3, 4));
        D(ggml_reshape_2d(ctx, a3, 16, 4));
        dies("view overlap tail", view_overlap_tail);
        dies("view gap past end", view_gap_past_end);
        dies("view gap fits", view_gap_fits);
        dies("view of transposed src", view_transposed_src);
        dies("view_1d of transposed src", view_1d_transposed_src);
        dies("view_4d past end", view_4d_past_end);
    }

    puts("=== v0.9.0: roll bounds ===");
    dies("roll INT_MIN", roll_int_min);
    dies("roll -7,-3", roll_neg_edge);
    dies("roll -ne1", roll_neg_ne);
    D(ggml_roll(ctx, a2, -7, -3, 0, 0));

    puts("=== v0.9.0: flash_attn_ext_rows ===");
    {
        struct ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 4, 8, 3);
        struct ggml_tensor * kk   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 40, 2, 1);
        struct ggml_tensor * vv   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 48, 40, 2, 1);
        struct ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 4, 1, 3);
        struct ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 32, 3);
        D(ggml_flash_attn_ext_rows(ctx, q, kk, vv, mask, rows, 0.125f, 0.0f, 0.0f));
        D(ggml_flash_attn_ext_rows(ctx, q, kk, vv, mask, rows, 0.25f, 8.0f, 30.0f));
        dies("fa_rows valid", far_ok);
        dies("fa_rows no mask", far_no_mask);
        dies("fa_rows mask ne0", far_mask_ne0);
        dies("fa_rows rows type", far_rows_type);
        dies("fa_rows rows strided", far_rows_strided);
        dies("fa_rows rows ne1", far_rows_ne1);
        dies("fa_rows rows ne2", far_rows_ne2);
        dies("fa_rows rows ne3", far_rows_ne3);
        dies("fa_rows k ne3", far_k_ne3);
        dies("fa_rows v ne3", far_v_ne3);
        dies("fa k ne3 != q ne3", fa_k_ne3);
    }

    puts("=== v0.9.0: f8_e4m3 ===");
    {
        struct ggml_tensor * f8 = ggml_new_tensor_3d(ctx, GGML_TYPE_F8_E4M3, 32, 6, 2);
        struct ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 5, 2);
        struct ggml_tensor * ix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
        D(f8);
        D(ggml_mul_mat(ctx, f8, x));
        D(ggml_get_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 32, 6), ix));
        D(ggml_cpy(ctx, x, ggml_new_tensor_3d(ctx, GGML_TYPE_F8_E4M3, 32, 5, 2)));
        D(ggml_view_2d(ctx, f8, 16, 3, f8->nb[1] * 2, 8));
        D(ggml_add(ctx, f8, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 6, 2)));
        printf("row_size(f8,33)=%zu nbytes=%zu ftype=%d\n", ggml_row_size(GGML_TYPE_F8_E4M3, 33), ggml_nbytes(f8),
               (int) ggml_ftype_to_ggml_type(GGML_FTYPE_MOSTLY_F8_E4M3));
    }

    puts("=== op tables ===");
    for (int op = 0; op < GGML_OP_COUNT; op++) {
        printf("op %3d %-24s %-46s expand=%d\n", op, ggml_op_name((enum ggml_op) op), ggml_op_symbol((enum ggml_op) op),
               ggml_op_alloc_size_may_expand((enum ggml_op) op));
    }

    puts("=== type and shape queries ===");
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        printf("type %2d %-12s blck=%lld size=%zu quant=%d\n", t,
               ggml_type_name((enum ggml_type) t),
               (long long) ggml_blck_size((enum ggml_type) t),
               ggml_type_size((enum ggml_type) t),
               ggml_is_quantized((enum ggml_type) t));
    }

    ggml_free(ctx);
    return 0;
}
