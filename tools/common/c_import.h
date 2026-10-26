/*
 * The C ABI the ported tools call, translated once as the `tools_h` module
 * (`b.addTranslateC` in `build/tools.zig`): libllama, ggml's tensor and GGUF
 * APIs, the backends, and the C library's stdio, whose `printf` formats the
 * tools keep so their output is upstream's byte for byte.
 *
 * **Not a port.**
 */
#include <stdio.h>
#include <stdlib.h>
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
