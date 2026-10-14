/*
 * libllama's C ABI, translated once for `cli/c.zig` (and `harness/smoke.zig`)
 * as the `llama_h` module (`b.addTranslateC` in `build.zig`).
 *
 * **Not a port.**
 */
#include "llama.h"
/* `ggml_threadpool_new`, for the persistent CPU threadpool `Session` attaches,
 * as upstream's `common_threadpools::init` does. */
#include "ggml-cpu.h"
