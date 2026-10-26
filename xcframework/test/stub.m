// The link gate for every slice: `@import llama;` and one call per library.
//
// Linked with `-F <slice> -framework llama` and nothing else -- no system
// frameworks named -- so it links only if the module map's autolinking
// supplies Metal, Accelerate and Foundation.

@import llama;

int main(void) {
    llama_backend_init();
    const char *info = llama_print_system_info();
    struct ggml_init_params params = { 1 << 20, NULL, false };
    struct ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
    struct gguf_context *g = gguf_init_empty();
    gguf_free(g);
    struct mtmd_context_params mp = mtmd_context_params_default();
    const char *marker = mtmd_default_marker();
    llama_backend_free();
    return info == 0 || marker == 0 || mp.n_threads == 0;
}
