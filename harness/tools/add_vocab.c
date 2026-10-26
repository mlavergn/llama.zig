// Gives an arch fixture a tokenizer: test-llama-archs writes them with
// `no_vocab`, which cannot tokenize text, so a tool gate that reads a text
// file cannot run on them. This rewrites the 128-token vocabulary as SPM --
// <unk>, <s>, </s>, "▁", printable ASCII, the newline as a byte token, the
// rest unused -- and copies everything else. For imatrix's MUL_MAT_ID path,
// which no local model with a vocabulary reaches.
//
// Usage: add_vocab <in.gguf> <out.gguf>
#include "ggml.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s in.gguf out.gguf\n", argv[0]); return 2; }
    struct ggml_context * ctx = NULL;
    struct gguf_init_params ip = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx };
    struct gguf_context * src = gguf_init_from_file(argv[1], ip);
    if (!src) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

    enum { N = 128 };
    static char text[N][16];
    const char * tokens[N];
    float scores[N];
    int32_t types[N];
    int n = 0;
    strcpy(text[n], "<unk>"); types[n] = 2; n++;
    strcpy(text[n], "<s>");   types[n] = 3; n++;
    strcpy(text[n], "</s>");  types[n] = 3; n++;
    strcpy(text[n], "\xe2\x96\x81"); types[n] = 1; n++;
    for (int ch = 0x21; ch <= 0x7e; ch++) { text[n][0] = (char) ch; text[n][1] = 0; types[n] = 1; n++; }
    strcpy(text[n], "<0x0A>"); types[n] = 6; n++;
    while (n < N) { snprintf(text[n], sizeof text[n], "<unused%d>", n); types[n] = 5; n++; }
    for (int i = 0; i < N; i++) { tokens[i] = text[i]; scores[i] = -(float) i; }

    struct gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    gguf_set_val_str(dst, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(dst, "tokenizer.ggml.tokens", tokens, N);
    gguf_set_arr_data(dst, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, N);
    gguf_set_arr_data(dst, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, N);
    gguf_set_val_u32(dst, "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(dst, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(dst, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_bool(dst, "tokenizer.ggml.add_bos_token", true);
    gguf_set_val_bool(dst, "tokenizer.ggml.add_eos_token", false);

    for (int64_t i = 0; i < gguf_get_n_tensors(src); i++) {
        gguf_add_tensor(dst, ggml_get_tensor(ctx, gguf_get_tensor_name(src, i)));
    }
    if (!gguf_write_to_file(dst, argv[2], false)) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    gguf_free(dst);
    gguf_free(src);
    ggml_free(ctx);
    return 0;
}
