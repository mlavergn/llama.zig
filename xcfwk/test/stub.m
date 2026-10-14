// The link gate for every slice: `@import llamazig;` and one call per library.
//
// Linked with `-F <slice> -framework llamazig` and nothing else -- no `-lc++`,
// no system frameworks named -- so it links only if the module map's
// autolinking supplies Metal, Accelerate and Foundation, and the prelinked
// object carries its own libc++.

@import llamazig;

int main(void) {
    llama_backend_init();
    const char *info = llama_print_system_info();
    struct ggml_init_params params = { 1 << 20, NULL, false };
    struct ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
    struct gguf_context *g = gguf_init_empty();
    gguf_free(g);
    llama_backend_free();
    return info == 0;
}
