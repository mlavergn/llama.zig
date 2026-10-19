// The three variadic entry points of ggml, for targets where Zig cannot
// define a C variadic function.
//
// Not a port in the usual sense, and not compiled everywhere. Zig 0.17's LLVM
// backend refuses `std.lang.VaList` on aarch64 outside Apple and Windows,
// and on x86_64 Windows ("disabled due to miscompilations"), so
// `ggml_abort`, `ggml_log_internal`
// and `ggml_format_name` -- which the C++ half of the build calls with
// varargs -- are defined here on those targets and nowhere else. Each does
// only the `va_list` handling and hands the formatted string back to the Zig
// in `src/ggml/runtime.zig`. `impl.zig_varargs` and `needsVarargsShim` in
// `build/llamacpp.zig` decide when this file is used, and must agree.
//
// Drawn from llama.cpp/ggml/src/ggml.c (v0.3.0, c1d0e7a00):
// `ggml_abort` (ggml.c:252), `ggml_log_internal_v` (ggml.c:286),
// `ggml_log_internal` (ggml.c:306), `ggml_format_name` (ggml.c:1954).

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"

// Defined in src/ggml/runtime.zig.
GGML_NORETURN void ggml_zig_abort_message(const char * message);
void ggml_zig_log_emit(enum ggml_log_level level, const char * text);

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);

    char message[2048];
    int offset = snprintf(message, sizeof(message), "%s:%d: ", file, line);

    va_list args;
    va_start(args, fmt);
    if (offset > 0 && (size_t) offset < sizeof(message)) {
        vsnprintf(message + offset, sizeof(message) - offset, fmt, args);
    }
    va_end(args);

    ggml_zig_abort_message(message);
}

void ggml_log_internal(enum ggml_log_level level, const char * format, ...) {
    if (format == NULL) {
        return;
    }
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        ggml_zig_log_emit(level, buffer);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        if (buffer2) {
            vsnprintf(buffer2, len + 1, format, args_copy);
            buffer2[len] = 0;
            ggml_zig_log_emit(level, buffer2);
            free(buffer2);
        }
    }
    va_end(args_copy);
    va_end(args);
}

struct ggml_tensor * ggml_format_name(struct ggml_tensor * tensor, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, sizeof(tensor->name), fmt, args);
    va_end(args);
    return tensor;
}
