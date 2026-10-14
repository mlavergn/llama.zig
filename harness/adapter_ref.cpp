// The reference side of `scripts/adapter-diff` that `llama.h` cannot reach.
//
// **Not a port.** `llama_context` keeps its active LoRA adapters in a
// `llama_adapter_loras` -- `std::unordered_map<llama_adapter_lora *, float>`
// -- and the graph adds their deltas in that map's iteration order, which
// libc++ derives from the pointers' hashes. Nothing in `llama.h` reports it.
// This builds the same map type from the same insertions
// `llama_context::set_adapters_lora` makes (llama-context.cpp:1266), so the
// harness can read the order the C++ context uses, and the order the real
// template gives the port's own pointers.

#include "llama-adapter.h"

#include <cstddef>

extern "C" {

// Inserts every (keys[i], scales[i]) with a non-zero scale, in order, as
// set_adapters_lora does, and writes the map's iteration order. Returns the
// map's size; *out_bc receives its bucket count.
size_t adapter_ref_loras_order(void * const * keys, const float * scales, size_t n,
                               void ** out_keys, float * out_scales, size_t * out_bc) {
    llama_adapter_loras loras;
    for (size_t i = 0; i < n; i++) {
        if (scales[i] != 0.0f) {
            loras.insert({(llama_adapter_lora *) keys[i], scales[i]});
        }
    }
    size_t k = 0;
    for (const auto & it : loras) {
        out_keys[k]   = it.first;
        out_scales[k] = it.second;
        k++;
    }
    *out_bc = loras.bucket_count();
    return k;
}

// loras.find(key): 1 and the scale when present, else 0.
int adapter_ref_loras_find(void * const * keys, const float * scales, size_t n, void * key, float * out_scale) {
    llama_adapter_loras loras;
    for (size_t i = 0; i < n; i++) {
        if (scales[i] != 0.0f) {
            loras.insert({(llama_adapter_lora *) keys[i], scales[i]});
        }
    }
    auto it = loras.find((llama_adapter_lora *) key);
    if (it == loras.end()) {
        return 0;
    }
    *out_scale = it->second;
    return 1;
}

}
