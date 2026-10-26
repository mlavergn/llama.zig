// The reference side of `make smoke-macos` and `make smoke-sim`.
//
// The driver `scripts/parity-port` writes inline, kept here as a file: greedy
// decoding through `llama.h`, `cpu` restricting the model to the CPU device.
// The Makefile links it against the port's own libraries (`zig build
// reference --release=fast -Dblas`), and `smoke.swift` runs the identical
// loop through `llama.framework` -- so any difference in output is the
// packaging, not the port.
//
// Usage: driver <model.gguf> <prompt> <n_predict> [cpu|gpu]

#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    // "cpu": the CPU device alone, so no op can be scheduled onto Metal --
    // n_gpu_layers = 0 would still leave the device in the list.
    ggml_backend_dev_t cpu_only[2] = { NULL, NULL };
    const int cpu = argc > 4 && strcmp(argv[4], "cpu") == 0;
    if (cpu) {
        cpu_only[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        mp.devices = cpu_only;
        mp.n_gpu_layers = 0;
    }
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token toks[512];
    int n = llama_tokenize(vocab, argv[2], (int)strlen(argv[2]), toks, 512, true, true);
    if (n < 0) return 1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512;
    if (cpu) {
        cp.offload_kqv = false;
        cp.op_offload = false;
    }
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    struct llama_batch batch = llama_batch_get_one(toks, n);
    llama_token id;
    for (int i = 0; i < atoi(argv[3]); i++) {
        if (llama_decode(ctx, batch)) return 1;
        id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        char buf[256];
        int k = llama_token_to_piece(vocab, id, buf, sizeof buf, 0, true);
        if (k > 0) fwrite(buf, 1, (size_t)k, stdout);
        batch = llama_batch_get_one(&id, 1);
    }
    printf("\n");
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
