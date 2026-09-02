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
    D(ggml_arange(ctx, 0.0f, 10.0f, 2.5f));
    D(ggml_timestep_embedding(ctx, vec, 64, 10000));
    D(ggml_roll(ctx, a2, 3, -2, 0, 0));
    D(ggml_tri(ctx, sq, GGML_TRI_TYPE_LOWER));
    D(ggml_fill(ctx, a2, -1.5f));
    D(ggml_win_part(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 14, 14, 1), 8));
    D(ggml_win_unpart(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 8, 8, 4), 14, 14, 8));
    D(ggml_get_rel_pos(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 15), 8, 8));

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
