// The reference side of `mtmd_get_memory_usage` for `make mtmd-diff`.
//
// **Not a port.** A harness. Upstream's `mtmd_get_memory_usage` has C++
// linkage and, since v0.6.0, returns a struct holding a `std::map`, so the
// Zig harness cannot call it directly; this flattens the result into
// arrays, in the map's order (by device pointer).

#include "mtmd.h"

#include <map>

extern "C" int ref_mtmd_get_memory_usage(const char * fname, struct mtmd_context_params params,
                                         ggml_backend_dev_t * devs, size_t * sizes, int cap,
                                         int * image_max_tokens, bool * use_non_causal) {
    mtmd_memory_usage r = mtmd_get_memory_usage(fname, params);
    int n = 0;
    for (const auto & [dev, size] : r.backend_mem_usage) {
        if (n < cap) {
            devs[n]  = dev;
            sizes[n] = size;
        }
        n++;
    }
    *image_max_tokens = r.image_max_tokens;
    *use_non_causal   = r.use_non_causal;
    return n;
}
