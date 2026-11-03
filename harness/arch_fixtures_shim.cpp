// Not a port. The twelve `common` symbols `llama.cpp/tests/test-llama-archs.cpp`
// reaches, so that file can be compiled unmodified against `libllama-ref.a`
// alone, without upstream's `common` library -- which llama.cpp.zmake builds
// as an intermediate and does not install, and which drags in httplib, the
// chat parsers and jinja for twelve functions.
//
// `scripts/gen-arch-fixtures` builds the test with this beside it and runs
// its `-o <dir>` mode: upstream's own fixture code writes one small random
// model per architecture through `llama_model_saver`, and those files are
// what `scripts/arch-model-diff` loads through the C++ and the port.
//
// `common_batch`'s four members the test links against are copied from
// `llama.cpp/common/common.cpp` at v0.6.0 (`d81235049`); the logging is
// reduced to stderr, which is all the test needs: it only ever logs progress
// and errors.

#include "common.h"
#include "log.h"
#include "llama-ext.h"

#include <cstdarg>
#include <cstdio>

// The threshold the LOG_* macros compare against, as `common/log.cpp` keeps
// it (log.cpp:17, 19, 23 @d81235049): since v0.4.0 the test raises it to
// LOG_LEVEL_DEBUG at start-up and filters libllama's log by
// `common_log_get_verbosity` itself.
int common_log_verbosity_thold = LOG_DEFAULT_LLAMA;

int common_log_get_verbosity_thold(void) {
    return common_log_verbosity_thold;
}

void common_log_set_verbosity_thold(int verbosity) {
    common_log_verbosity_thold = verbosity;
}

// Copied from `common/log.cpp` (log.cpp:532 @d81235049).
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

// Copied from `common/common.cpp` (common.cpp:2145, 2150, 2154, 2202 @d81235049).
// Only `test_backends` reaches them, never the `-o` path the fixtures use.
common_batch::common_batch(llama_context * ctx) : batch(llama_batch_ext_init(ctx)) {
    const auto rope_type = llama_model_rope_type(llama_get_model(ctx));
    n_pos = rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? GGML_MROPE_SECTIONS : 1;
}

void common_batch::clear() {
    tokens.clear();
}

int32_t common_batch::add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output) {
    tokens.push_back({ id, { pos, 0, 0, 0 }, seq_id, output, { nullptr, 0, 0 }, {} });
    return size() - 1;
}

llama_batch_ext * common_batch::get_sub_batch(int32_t off, int32_t n) {
    GGML_ASSERT(batch && "common_batch was not initialized with a context");
    GGML_ASSERT(off >= 0 && n >= 0 && off + n <= size());

    llama_batch_ext * res = batch.get();
    llama_batch_ext_clear(res);

    for (int32_t i = off; i < off + n; ++i) {
        const token & t = tokens[i];

        int32_t idx;
        if (t.id != LLAMA_TOKEN_NULL) {
            idx = llama_batch_ext_add_token(res, t.seq_id, t.id);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add token %d at index %d (error %d, n = %d)\n", __func__, t.id, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
            if (t.embd.data && !llama_batch_ext_set_embd_token(res, idx, t.embd)) {
                GGML_ABORT("%s: failed to set the embedding of token %d at index %d\n", __func__, t.id, i);
            }
        } else {
            idx = llama_batch_ext_add_embd(res, t.seq_id, t.embd);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add embedding at index %d (error %d, n = %d)\n", __func__, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
        }
        GGML_ASSERT(idx == i - off);

        for (const llama_seq_id seq_id : t.seq_ids_extra) {
            if (!llama_batch_ext_add_seq(res, idx, seq_id)) {
                GGML_ABORT("%s: failed to add seq %d to the entry at index %d\n", __func__, seq_id, i);
            }
        }
        if (t.output) {
            llama_batch_ext_set_output_logits(res, idx, true);
        }
        if (t.decision_order != 0) {
            llama_batch_ext_set_decision_order(res, idx, (llama_decision_order) t.decision_order);
        }
    }

    return res;
}
