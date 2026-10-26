/*
 * Writes LoRA adapters for a base model, for scripts/tools-diff export-lora:
 * good ones over an attention, an FFN and the token-embedding tensor, and one
 * bad file per error export-lora reports.
 *
 * **Not a port.** Our own harness, over ggml's GGUF API.
 *
 * Usage: make_lora <base.gguf> <outdir>
 *
 * Shapes follow export-lora's merge: for a weight W of ne = {in, out},
 * lora_a is {in, r} and lora_b is {r, out}; for token_embd, lora_a is
 * {r, n_vocab} and lora_b is {r, n_embd}. Values come from a fixed LCG, so
 * every run writes the same bytes.
 */
#include "ggml.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned g_seed = 1;
static float rnd(void) {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((float) ((g_seed >> 8) & 0xffff) / 65535.0f - 0.5f) * 0.1f;
}

struct pick { const char * name; int64_t ne0, ne1; };

static void fill(struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        for (int64_t i = 0; i < n; i++) ((float *) t->data)[i] = rnd();
    } else if (t->type == GGML_TYPE_F16) {
        for (int64_t i = 0; i < n; i++) ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(rnd());
    } else {
        float * tmp = malloc(n * sizeof(float));
        for (int64_t i = 0; i < n; i++) tmp[i] = rnd();
        ggml_quantize_chunk(t->type, tmp, t->data, 0, n / t->ne[0], t->ne[0], NULL);
        free(tmp);
    }
}

/* One adapter file. flags select the bad variants. */
enum { BAD_NONE, BAD_GENERAL_TYPE, BAD_ADAPTER_TYPE, BAD_ARCH, BAD_MISSING_B, BAD_QUANT, SUBSET };

static int write_adapter(const char * path, const char * arch, struct pick * picks, int n_picks,
                         enum ggml_type type, float alpha, int bad) {
    const int r = 4;
    struct ggml_init_params ip = { 64 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", bad == BAD_GENERAL_TYPE ? "model" : "adapter");
    gguf_set_val_str(g, "adapter.type", bad == BAD_ADAPTER_TYPE ? "control_vector" : "lora");
    gguf_set_val_str(g, "general.architecture", bad == BAD_ARCH ? "not-an-arch" : arch);
    if (alpha != 0.0f) gguf_set_val_f32(g, "adapter.lora.alpha", alpha);

    const int n = bad == SUBSET ? 1 : n_picks;
    for (int i = 0; i < n; i++) {
        const int tok = strncmp(picks[i].name, "token_embd", 10) == 0;
        const int64_t a0 = tok ? r : picks[i].ne0, a1 = tok ? picks[i].ne1 : r;
        const int64_t b0 = r,                      b1 = tok ? picks[i].ne0 : picks[i].ne1;
        const enum ggml_type ta = (bad == BAD_QUANT && i == 0) ? GGML_TYPE_Q8_0 : type;
        struct ggml_tensor * ta_t = ggml_new_tensor_2d(ctx, ta, ta == GGML_TYPE_Q8_0 ? (a0 + 31) / 32 * 32 : a0, a1);
        struct ggml_tensor * tb_t = ggml_new_tensor_2d(ctx, type, b0, b1);
        char name[GGML_MAX_NAME];
        snprintf(name, sizeof(name), "%s.lora_a", picks[i].name);
        ggml_set_name(ta_t, name);
        snprintf(name, sizeof(name), "%s.lora_b", picks[i].name);
        ggml_set_name(tb_t, name);
        fill(ta_t);
        fill(tb_t);
        gguf_add_tensor(g, ta_t);
        if (!(bad == BAD_MISSING_B && i == n - 1)) gguf_add_tensor(g, tb_t);
    }
    const bool ok = gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <base.gguf> <outdir>\n", argv[0]);
        return 2;
    }
    struct ggml_context * meta = NULL;
    struct gguf_init_params gp = { true, &meta };
    struct gguf_context * base = gguf_init_from_file(argv[1], gp);
    if (!base) return 1;
    const char * arch = gguf_get_val_str(base, gguf_find_key(base, "general.architecture"));

    /* the first attention and FFN weights, and the token embedding */
    struct pick picks[3];
    int n = 0;
    const char * want[3] = { "attn_q.weight", "ffn_down.weight", "token_embd.weight" };
    for (int w = 0; w < 3; w++) {
        for (struct ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
            if (ggml_n_dims(t) == 2 && strstr(t->name, want[w])) {
                picks[n].name = strdup(t->name);
                picks[n].ne0 = t->ne[0];
                picks[n].ne1 = t->ne[1];
                n++;
                break;
            }
        }
    }

    char path[4096];
    int rc = 0;
#define OUT(f, ...) (snprintf(path, sizeof(path), "%s/%s", argv[2], f), rc |= write_adapter(path, arch, picks, n, __VA_ARGS__))
    OUT("good.gguf",        GGML_TYPE_F32, 8.0f, BAD_NONE);
    OUT("good-f16.gguf",    GGML_TYPE_F16, 0.0f, BAD_NONE);
    OUT("subset.gguf",      GGML_TYPE_F32, 8.0f, SUBSET);
    OUT("not-adapter.gguf", GGML_TYPE_F32, 8.0f, BAD_GENERAL_TYPE);
    OUT("not-lora.gguf",    GGML_TYPE_F32, 8.0f, BAD_ADAPTER_TYPE);
    OUT("wrong-arch.gguf",  GGML_TYPE_F32, 8.0f, BAD_ARCH);
    OUT("missing-b.gguf",   GGML_TYPE_F32, 8.0f, BAD_MISSING_B);
    OUT("quant.gguf",       GGML_TYPE_F32, 8.0f, BAD_QUANT);
    printf("wrote adapters for %d tensors of %s\n", n, arch);
    return rc;
}
