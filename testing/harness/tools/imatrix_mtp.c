// Gives a Qwen3.5 arch fixture an MTP (NextN) block, for imatrix's
// `--nextn` and `-md` paths: test-llama-archs writes no fixture with one.
// The block at index n_layer copies the first full-attention layer's
// tensors -- an MTP block is a full-attention decoder block -- and adds the
// three NextN tensors (eh_proj, enorm, hnorm) with small random values;
// block_count grows by one and nextn_predict_layers is 1.
//
// Writes two files: the model with its MTP block, and an MTP-only draft --
// the same metadata, the non-block tensors and the MTP block alone, no
// `blk.0.` tensor -- which is what `-md` takes.
//
// Usage: imatrix_mtp <in.gguf> <with-nextn.gguf> <draft.gguf>
#include "ggml.h"
#include "gguf.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 12345;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) / 16777216.0f - 0.5f) * 0.2f;
}

static size_t type_size(enum gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 4;
    }
}

static struct ggml_tensor * new_f32(struct ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1, float base) {
    struct ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1) : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) { d[i] = base + frand(); }
    return t;
}

int main(int argc, char ** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s in.gguf with-nextn.gguf draft.gguf\n", argv[0]); return 2; }
    struct ggml_context * ctx = NULL;
    struct gguf_init_params ip = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx };
    struct gguf_context * src = gguf_init_from_file(argv[1], ip);
    if (!src) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

    const char * arch = gguf_get_val_str(src, gguf_find_key(src, "general.architecture"));
    char key[256];
    snprintf(key, sizeof key, "%s.block_count", arch);
    const uint32_t n_layer = gguf_get_val_u32(src, gguf_find_key(src, key));
    snprintf(key, sizeof key, "%s.embedding_length", arch);
    const uint32_t n_embd = gguf_get_val_u32(src, gguf_find_key(src, key));

    int full = -1;
    for (uint32_t il = 0; il < n_layer && full < 0; il++) {
        char name[128];
        snprintf(name, sizeof name, "blk.%u.attn_q.weight", il);
        if (ggml_get_tensor(ctx, name)) { full = (int) il; }
    }
    if (full < 0) { fprintf(stderr, "no full-attention layer in %s\n", argv[1]); return 1; }

    // the MTP block's tensors, in a context of their own
    const int64_t n_src = gguf_get_n_tensors(src);
    struct ggml_init_params gp = { /*.mem_size =*/ 64*1024*1024, /*.mem_buffer =*/ NULL, /*.no_alloc =*/ false };
    struct ggml_context * mtp = ggml_init(gp);
    char prefix[64];
    snprintf(prefix, sizeof prefix, "blk.%d.", full);
    for (int64_t i = 0; i < n_src; i++) {
        const char * name = gguf_get_tensor_name(src, i);
        if (strncmp(name, prefix, strlen(prefix)) != 0) { continue; }
        struct ggml_tensor * t = ggml_get_tensor(ctx, name);
        struct ggml_tensor * d = ggml_dup_tensor(mtp, t);
        char nn[128];
        snprintf(nn, sizeof nn, "blk.%u.%s", n_layer, name + strlen(prefix));
        ggml_set_name(d, nn);
        memcpy(d->data, t->data, ggml_nbytes(t));
    }
    snprintf(key, sizeof key, "blk.%u.nextn.eh_proj.weight", n_layer);
    new_f32(mtp, key, 2 * (int64_t) n_embd, n_embd, 0.0f);
    snprintf(key, sizeof key, "blk.%u.nextn.enorm.weight", n_layer);
    new_f32(mtp, key, n_embd, 0, 1.0f);
    snprintf(key, sizeof key, "blk.%u.nextn.hnorm.weight", n_layer);
    new_f32(mtp, key, n_embd, 0, 1.0f);

    for (int out = 0; out < 2; out++) {
        struct gguf_context * dst = gguf_init_empty();
        gguf_set_kv(dst, src);
        snprintf(key, sizeof key, "%s.block_count", arch);
        gguf_set_val_u32(dst, key, n_layer + 1);
        snprintf(key, sizeof key, "%s.nextn_predict_layers", arch);
        gguf_set_val_u32(dst, key, 1);
        // per-layer arrays grow by the full-attention layer's entry: the MTP
        // block is one (not recurrent, its head counts)
        for (int64_t k = 0; k < gguf_get_n_kv(src); k++) {
            if (gguf_get_kv_type(src, k) != GGUF_TYPE_ARRAY) { continue; }
            const enum gguf_type at = gguf_get_arr_type(src, k);
            if (at == GGUF_TYPE_STRING || at == GGUF_TYPE_ARRAY) { continue; }
            if (gguf_get_arr_n(src, k) != n_layer) { continue; }
            const size_t es = type_size(at);
            char * buf = malloc(es * (n_layer + 1));
            memcpy(buf, gguf_get_arr_data(src, k), es * n_layer);
            memcpy(buf + es * n_layer, (const char *) gguf_get_arr_data(src, k) + es * full, es);
            gguf_set_arr_data(dst, gguf_get_key(src, k), at, buf, n_layer + 1);
            free(buf);
        }
        for (int64_t i = 0; i < n_src; i++) {
            const char * name = gguf_get_tensor_name(src, i);
            if (out == 1 && strncmp(name, "blk.", 4) == 0) { continue; }
            gguf_add_tensor(dst, ggml_get_tensor(ctx, name));
        }
        for (struct ggml_tensor * t = ggml_get_first_tensor(mtp); t; t = ggml_get_next_tensor(mtp, t)) {
            gguf_add_tensor(dst, t);
        }
        if (!gguf_write_to_file(dst, argv[2 + out], false)) { fprintf(stderr, "cannot write %s\n", argv[2 + out]); return 1; }
        gguf_free(dst);
    }

    ggml_free(mtp);
    gguf_free(src);
    ggml_free(ctx);
    return 0;
}
