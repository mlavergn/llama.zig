// The reference side of `scripts/arch-diff`.
//
// **Not a port.** `llama-arch.cpp`'s functions have C++ linkage and take
// `std::string`, so the Zig harness cannot call them; this file wraps each in
// a C function over plain buffers. It is compiled against the unmodified
// `llama-arch.h` and linked with the reference `libllama.a`, so what it
// reports is what upstream's `llama-arch.o` does.
//
// A function the C++ makes throw reports the exception's `what()` and returns
// -1. `LLM_TN_IMPL::str` aborts instead of throwing for a tensor with no name,
// which only a separate process can observe; the harness forks for those.

#include "llama-arch.h"
#include "llama-ext.h"

#include <cstring>
#include <stdexcept>
#include <string>

static int put(const std::string & s, char * buf, size_t n) {
    if (n == 0) {
        return (int) s.size();
    }
    size_t k = s.size() < n - 1 ? s.size() : n - 1;
    memcpy(buf, s.data(), k);
    buf[k] = 0;
    return (int) s.size();
}

extern "C" {

int ref_arch_unknown(void) {
    return LLM_ARCH_UNKNOWN;
}

const char * ref_arch_name(int arch) {
    return llm_arch_name((llm_arch) arch);
}

int ref_arch_from_string(const char * name) {
    return llm_arch_from_string(name);
}

int ref_arch_all(int * out, int cap) {
    std::vector<llm_arch> all = llm_arch_all();
    for (size_t i = 0; i < all.size() && (int) i < cap; i++) {
        out[i] = all[i];
    }
    return (int) all.size();
}

int ref_arch_flags(int arch) {
    const llm_arch a = (llm_arch) arch;
    return (llm_arch_is_recurrent(a)         ? 1  : 0) |
           (llm_arch_is_hybrid(a)            ? 2  : 0) |
           (llm_arch_is_diffusion(a)         ? 4  : 0) |
           (llm_arch_supports_rs_rollback(a) ? 8  : 0) |
           (llm_arch_supports_sm_tensor(a)   ? 16 : 0) |
           (llm_arch_supports_mixed_batch(a) ? 32 : 0);
}

// llama-ext.h's architecture queries, which forward to the llm_* functions.
int ref_ext_arch_flags(int arch) {
    const llm_arch a = (llm_arch) arch;
    return (llama_arch_supports_sm_tensor(a)   ? 1 : 0) |
           (llama_arch_supports_mixed_batch(a) ? 2 : 0) |
           (llama_arch_supports_save(a)        ? 4 : 0);
}

const char * ref_ext_arch_name(int arch) {
    return llama_arch_name((llm_arch) arch);
}

int ref_ext_arch_from_string(const char * name) {
    return llama_arch_from_string(name);
}

int ref_ext_arch_all(int * out, int cap) {
    std::vector<llm_arch> all = llama_arch_all();
    for (size_t i = 0; i < all.size() && (int) i < cap; i++) {
        out[i] = all[i];
    }
    return (int) all.size();
}

int ref_ext_kv_name(int arch, int kv, char * buf, size_t n) {
    return put(llama_arch_kv_name((llm_arch) arch, (llm_kv) kv), buf, n);
}

// LLM_KV(arch, suffix)(kv). Returns the name's length, or -1 with what() in buf.
int ref_kv_name(int arch, int kv, const char * suffix, char * buf, size_t n) {
    try {
        return put(LLM_KV((llm_arch) arch, suffix)((llm_kv) kv), buf, n);
    } catch (const std::exception & e) {
        put(e.what(), buf, n);
        return -1;
    }
}

// LLM_TN(arch)(tensor, suffix, bid, xid).str(). Aborts for a tensor with no name.
int ref_tensor_name(int arch, int tensor, const char * suffix, int bid, int xid, char * buf, size_t n) {
    const LLM_TN tn((llm_arch) arch);
    return put(tn((llm_tensor) tensor, suffix, bid, xid).str(), buf, n);
}

// llm_tensor_info_for. Returns 0, or -1 with what() in buf.
int ref_tensor_info(int tensor, int * layer, int * op, char * buf, size_t n) {
    try {
        const llm_tensor_info & info = llm_tensor_info_for((llm_tensor) tensor);
        *layer = info.layer;
        *op    = info.op;
        return 0;
    } catch (const std::exception & e) {
        put(e.what(), buf, n);
        return -1;
    }
}

}
