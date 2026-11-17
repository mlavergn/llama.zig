// The reference side of `scripts/moe-slru-diff`.
//
// **Not a port.** `moe_cache_lru` lives in an anonymous namespace in
// llama-moe-cache.cpp, so nothing outside that translation unit can name it.
// This file includes the unmodified source, with `llama_moe_cache` renamed so
// the class does not collide with libllama-ref.a's, and wraps the struct in
// plain C functions. The `n_cpu` callback is `impl::copy`'s lambda
// (llama-moe-cache.cpp:808 @b1b770d90), over a share and a carry the caller
// owns.

#define llama_moe_cache ref_slru_llama_moe_cache
#include "llama-moe-cache.cpp"
#undef llama_moe_cache

extern "C" {

void * slru_ref_new(int n_layer, int n_expert, int n_slots) {
    auto * s = new moe_cache_lru();
    s->init(n_layer, n_expert, n_slots);
    return s;
}

void slru_ref_free(void * p) {
    delete (moe_cache_lru *) p;
}

void slru_ref_set_map(void * p, int il, int32_t * data) {
    ((moe_cache_lru *) p)->slot_map[il] = data;
}

int slru_ref_empty(void * p) {
    return ((moe_cache_lru *) p)->empty;
}

int slru_ref_n_filled(void * p) {
    return ((moe_cache_lru *) p)->n_filled;
}

// Returns plan()'s result; fills and cpu hold expert, slot pairs, *n_fills and
// *n_cpu of them; *acc is the carry of the CPU share, updated as copy() does.
int slru_ref_plan(void * p, int il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero,
        uint32_t share_q, uint32_t * acc,
        int32_t * fills, size_t * n_fills, int32_t * cpu, size_t * n_cpu_out, size_t * n_hit) {
    static std::vector<moe_cache_lru::fill> out;
    static std::vector<moe_cache_lru::fill> out_cpu;
    auto n_cpu = [&](size_t n_miss) {
        const uint64_t a = (uint64_t) n_miss*share_q + *acc;
        *acc = a & 0xffff;
        return (size_t) (a >> 16);
    };
    const bool ok = ((moe_cache_lru *) p)->plan(il, ids, n_ids, n_tokens, n_used, n_zero, n_cpu, out, out_cpu, *n_hit);
    *n_fills = out.size();
    for (size_t i = 0; i < out.size(); ++i) {
        fills[2*i + 0] = out[i].expert;
        fills[2*i + 1] = out[i].slot;
    }
    *n_cpu_out = out_cpu.size();
    for (size_t i = 0; i < out_cpu.size(); ++i) {
        cpu[2*i + 0] = out_cpu[i].expert;
        cpu[2*i + 1] = out_cpu[i].slot;
    }
    return ok;
}

}
