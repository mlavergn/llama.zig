// Not a port. The six `common` symbols `llama.cpp/tests/test-llama-archs.cpp`
// reaches, so that file can be compiled unmodified against `libllama-ref.a`
// alone, without upstream's `common` library -- which llama.cpp.zmake builds
// as an intermediate and does not install, and which drags in httplib, the
// chat parsers and jinja for six functions.
//
// `scripts/gen-arch-fixtures` builds the test with this beside it and runs
// its `-o <dir>` mode: upstream's own fixture code writes one small random
// model per architecture through `llama_model_saver`, and those files are
// what `scripts/arch-model-diff` loads through the C++ and the port.
//
// `common_batch_add` is copied from `llama.cpp/common/common.cpp` at v0.3.0
// (`c1d0e7a00`); the logging is reduced to stderr, which is all the test
// needs: it only ever logs progress and errors.

#include "common.h"
#include "log.h"

#include <cstdarg>
#include <cstdio>

// The test only compares this against a log level; LOG_DEFAULT_LLAMA is
// common/log.cpp's initial value.
int common_log_get_verbosity_thold(void) {
    return LOG_DEFAULT_LLAMA;
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
