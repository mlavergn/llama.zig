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

#include <stdio.h>
#include <string.h>

/* Ours, and the reference's with its six exports renamed by -D. */
extern ggml_backend_reg_t ggml_backend_metal_reg(void);
extern ggml_backend_reg_t ref_ggml_backend_metal_reg(void);

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

    printf("\n");
    if (fails == 0) {
        printf("PASS: %d Metal vtable answers identical, ported vs reference\n", checks);
        return 0;
    }
    printf("FAIL: %d of %d Metal vtable answers differ\n", fails, checks);
    return 1;
}
