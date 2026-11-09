// The reference side of `scripts/context-diff` that `llama.h` cannot reach.
//
// **Not a port.** `llama_get_memory_breakdown` (llama-ext.h) returns a
// `std::map`, which only C++ can walk; this copies it into flat arrays for
// the Zig harness, in the map's key order.

#include "llama-ext.h"

#include <cstddef>

extern "C" {

// Writes up to `cap` entries: bufts[i], then model, context and compute at
// sizes[3*i..]. Returns the map's size.
size_t context_ref_breakdown(const llama_context * ctx, ggml_backend_buffer_type_t * bufts, size_t * sizes, size_t cap) {
    const llama_memory_breakdown mb = llama_get_memory_breakdown(ctx);
    size_t i = 0;
    for (const auto & [buft, data] : mb) {
        if (i < cap) {
            bufts[i]       = buft;
            sizes[3*i + 0] = data.model;
            sizes[3*i + 1] = data.context;
            sizes[3*i + 2] = data.compute;
        }
        i++;
    }
    return mb.size();
}

}
