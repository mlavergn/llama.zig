// The reference side of `scripts/moe-slru-diff`.
//
// **Not a port.** `moe_cache_slru` lives in an anonymous namespace in
// llama-moe-cache.cpp, so nothing outside that translation unit can name it.
// This file includes the unmodified source, with `llama_moe_cache` renamed so
// the class does not collide with libllama-ref.a's, and wraps the struct in
// plain C functions.

#define llama_moe_cache ref_slru_llama_moe_cache
#include "llama-moe-cache.cpp"
#undef llama_moe_cache

extern "C" {

void * slru_ref_new(int n_layer, int n_expert, int n_slots) {
    auto * s = new moe_cache_slru();
    s->init(n_layer, n_expert, n_slots);
    return s;
}

void slru_ref_free(void * p) {
    delete (moe_cache_slru *) p;
}

void slru_ref_set_map(void * p, int il, int32_t * data) {
    ((moe_cache_slru *) p)->slot_map[il] = data;
}

// Returns plan()'s result; fills holds expert, slot pairs, *n_fills of them.
int slru_ref_plan(void * p, int il, const int32_t * ids, size_t n_ids, int32_t * fills, size_t * n_fills, size_t * n_hit) {
    static std::vector<moe_cache_slru::fill> out;
    const bool ok = ((moe_cache_slru *) p)->plan(il, ids, n_ids, out, *n_hit);
    *n_fills = out.size();
    for (size_t i = 0; i < out.size(); ++i) {
        fills[2*i + 0] = out[i].expert;
        fills[2*i + 1] = out[i].slot;
    }
    return ok;
}

}
