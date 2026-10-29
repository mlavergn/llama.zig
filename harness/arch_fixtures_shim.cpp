// Not a port. The nine `common` symbols `llama.cpp/tests/test-llama-archs.cpp`
// reaches, so that file can be compiled unmodified against `libllama-ref.a`
// alone, without upstream's `common` library -- which llama.cpp.zmake builds
// as an intermediate and does not install, and which drags in httplib, the
// chat parsers and jinja for nine functions.
//
// `scripts/gen-arch-fixtures` builds the test with this beside it and runs
// its `-o <dir>` mode: upstream's own fixture code writes one small random
// model per architecture through `llama_model_saver`, and those files are
// what `scripts/arch-model-diff` loads through the C++ and the port.
//
// `common_batch_add` is copied from `llama.cpp/common/common.cpp` at v0.4.1
// (`b29c606e2`); the logging is reduced to stderr, which is all the test
// needs: it only ever logs progress and errors.

#include "common.h"
#include "log.h"

#include <cstdarg>
#include <cstdio>

// The threshold the LOG_* macros compare against, as `common/log.cpp` keeps
// it (log.cpp:30, 32, 36 @b29c606e2): since v0.4.0 the test raises it to
// LOG_LEVEL_DEBUG at start-up and filters libllama's log by
// `common_log_get_verbosity` itself.
int common_log_verbosity_thold = LOG_DEFAULT_LLAMA;

int common_log_get_verbosity_thold(void) {
    return common_log_verbosity_thold;
}

void common_log_set_verbosity_thold(int verbosity) {
    common_log_verbosity_thold = verbosity;
}

// Copied from `common/log.cpp` (log.cpp:529 @b29c606e2).
int common_log_get_verbosity(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return LOG_LEVEL_DEBUG;
        case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_WARN:  return LOG_LEVEL_WARN;
        case GGML_LOG_LEVEL_ERROR: return LOG_LEVEL_ERROR;
        case GGML_LOG_LEVEL_CONT:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_NONE:
        default:
            return LOG_LEVEL_OUTPUT;
    }
}

struct common_log * common_log_main() {
    return nullptr;
}

void common_log_add(struct common_log * /*log*/, enum ggml_log_level /*level*/, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

void common_log_flush(struct common_log * /*log*/) {
    fflush(stderr);
}

void common_init() {
    llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);
}

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}
