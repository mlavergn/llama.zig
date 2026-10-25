// C entry points over upstream's `common/fit.cpp`, for harness/fit_diff.zig.
//
// **Not a port.** `common_fit_params` and its siblings have C++ linkage, and
// `common_log` -- where `fit.cpp` writes its trace -- is a C++ singleton with
// a worker thread. These wrap both in a C ABI the Zig harness can call:
// `fit.cpp` and `log.cpp` compile unmodified beside this file, over
// `libllama-ref.a`.
//
// The log goes to a file as well as the terminal (`common_log_set_file`), with
// the level prefix on and colours off, so the harness can compare the text
// and the level of every line; `common_log_flush` waits for the worker before
// the file is read.
//
// Used by harness/fit_diff.zig through scripts/fit-diff.

#include "fit.h"
#include "log.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>

// `common.cpp` defines this, and `log.cpp` asks it once when the logger
// starts; compiling `common.cpp` for one predicate would bring most of
// `common/` with it. The harness turns colours off regardless.
bool tty_can_use_colors() {
    return false;
}

extern "C" {

// Points `common_log` at `path`, prefixed and uncoloured, at `verbosity`.
void fit_ref_log_setup(const char * path, int verbosity) {
    common_log * log = common_log_main();
    common_log_set_colors(log, LOG_COLORS_DISABLED);
    common_log_set_prefix(log, true);
    common_log_set_timestamps(log, false);
    common_log_set_file(log, path);
    common_log_set_verbosity_thold(verbosity);
}

// Waits until every pending message has been written.
void fit_ref_log_flush(void) {
    common_log_flush(common_log_main());
}

// `common_fit_params`, status as an int.
int fit_ref_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        const char * extra_path,
        llama_model_params * extra_mparams,
        llama_context_params * extra_cparams,
        bool extra_shares_model,
        int log_level) {
    common_fit_extra_model extra = { extra_path, extra_mparams, extra_cparams, extra_shares_model };
    return (int) common_fit_params(path_model, mparams, cparams, tensor_split, tensor_buft_overrides,
        margins, n_ctx_min, extra_path ? &extra : nullptr, (ggml_log_level) log_level);
}

// `common_get_device_memory_data`, flattened: five int64 per entry (total,
// free, model, context, compute) into `out`, at most `cap` entries, and the
// devices into `devs`. Returns the entry count, or -1 where it throws.
int fit_ref_device_memory(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        int64_t * out, int cap,
        ggml_backend_dev_t * devs, int * n_devs,
        uint32_t * hp, int log_level) {
    std::vector<ggml_backend_dev_t> d;
    common_device_memory_data_vec v;
    try {
        v = common_get_device_memory_data(path_model, mparams, cparams, d, hp[0], hp[1], hp[2], (ggml_log_level) log_level);
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        return -1;
    }
    for (size_t i = 0; i < v.size() && (int) i < cap; i++) {
        out[5*i + 0] = v[i].total;
        out[5*i + 1] = v[i].free;
        out[5*i + 2] = (int64_t) v[i].model;
        out[5*i + 3] = (int64_t) v[i].context;
        out[5*i + 4] = (int64_t) v[i].compute;
    }
    *n_devs = (int) d.size();
    for (size_t i = 0; i < d.size(); i++) {
        devs[i] = d[i];
    }
    return (int) v.size();
}

// `common_fit_print`, its stdout captured into `path`. Returns 0, or -1 where
// it throws.
int fit_ref_print(const char * path_model, llama_model_params * mparams, llama_context_params * cparams, const char * path) {
    fflush(stdout);
    const int saved = dup(1);
    FILE * f = fopen(path, "w");
    dup2(fileno(f), 1);
    int rc = 0;
    try {
        common_fit_print(path_model, mparams, cparams);
    } catch (const std::exception &) {
        rc = -1;
    }
    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    fclose(f);
    return rc;
}

}
