/*
 * `llama.h`, and through it `ggml.h`, `ggml-backend.h`, `ggml-cpu.h` and
 * `ggml-opt.h`; and `gguf.h`, which the vocabulary and the model loader read
 * metadata through. Translated for `src/llama/impl.zig` as the `llama_c`
 * module (`b.addTranslateC` in `build/llamacpp.zig`).
 *
 * **Not a port.**
 */
#include "llama.h"
#include "gguf.h"
