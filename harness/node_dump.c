/* Hashes the output of every node of one llama_decode, in execution order.
 *
 * Built twice by scripts/node-diff -- once against the ported libraries, once
 * against llama.cpp.zmake -- and the two listings are diffed. The first line
 * that differs names the op that first computed a different bit, in a real
 * model's real graph.
 *
 * Why this exists
 * ---------------
 * Token parity is coarse: at --temp 0 a last-bit difference has to move an
 * argmax before it shows, and usually it does not. `make ops-diff` is exact
 * but sees only the shapes and types its author thought of. This is exact
 * *and* sees what the model actually runs. It found the Q5_K, TQ1_0 and TQ2_0
 * dot-product epilogues as the first divergent MUL_MAT of a CPU-only Qwen3.5
 * decode, which the vec_dot goldens had passed.
 *
 * argv: <model.gguf> <prompt> <n_threads> [gpu]
 *   The model is loaded on the CPU device alone unless `gpu` is given: on a
 *   Metal machine the CPU kernels are otherwise never on the path.
 *
 * Output: one line per node -- index, op, name, type, the buffer holding the
 * output (which says which backend ran it), shape, FNV-1a of the output
 * bytes, and the op and type of each source.
 */

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int node_index = 0;

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* The scheduler asks first (`ask`), then calls again once the node is
 * computed. Asking for everything costs a graph split per node, which is slow
 * and is exactly what makes every output observable. */
static bool observe(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (ask) return true;

    const size_t nb = ggml_nbytes(t);
    uint64_t h;
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        h = fnv1a(t->data, nb);
    } else {
        void * tmp = malloc(nb);
        ggml_backend_tensor_get(t, tmp, 0, nb);
        h = fnv1a(tmp, nb);
        free(tmp);
    }

    printf("%5d %-14s %-32s %-6s %-8s [%lld,%lld,%lld,%lld] %016llx",
           node_index++, ggml_op_desc(t), t->name, ggml_type_name(t->type),
           t->buffer ? ggml_backend_buffer_name(t->buffer) : "-",
           (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
           (unsigned long long) h);
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
        printf(" s%d=%s:%s", i, ggml_op_desc(t->src[i]), ggml_type_name(t->src[i]->type));
    }
    printf("\n");
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> <n_threads> [gpu]\n", argv[0]);
        return 2;
    }
    const int gpu = argc > 4 && strcmp(argv[4], "gpu") == 0;

    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu_only[2] = { NULL, NULL };
    if (gpu) {
        mp.n_gpu_layers = 99;
    } else {
        cpu_only[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        mp.devices = cpu_only;
        mp.n_gpu_layers = 0;
    }
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token toks[512];
    const int n = llama_tokenize(vocab, argv[2], (int) strlen(argv[2]), toks, 512, true, true);
    if (n < 0) return 1;

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_threads = cp.n_threads_batch = atoi(argv[3]);
    if (!gpu) {
        cp.offload_kqv = false;
        cp.op_offload = false;
    }
    cp.cb_eval = observe;
    cp.cb_eval_user_data = NULL;

    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    /* The prompt, then one decode step: the batched path and the single-row
     * path take different kernels (flash attention's split-KV, the
     * recurrent layers' one-token branch). */
    if (llama_decode(ctx, llama_batch_get_one(toks, n))) return 1;
    llama_token next = toks[n - 1];
    if (llama_decode(ctx, llama_batch_get_one(&next, 1))) return 1;

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
