// The reference side of `scripts/chat-diff`.
//
// **Not a port.** `llm_chat_detect_template`, `llm_chat_template_from_str`
// and `llm_chat_apply_template` have C++ linkage and take `std::string`s and a
// `std::vector`, so the Zig harness reaches the copies in `libllama.a`
// through these C wrappers. The public `llama_chat_apply_template` and
// `llama_chat_builtin_templates` it calls directly.

#include "llama-chat.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

// llm_chat_detect_template on exactly `n` bytes.
int ref_detect(const char * p, size_t n) {
    return (int) llm_chat_detect_template(std::string(p, n));
}

// llm_chat_template_from_str on exactly `n` bytes; -1 where it throws
// std::out_of_range.
int ref_from_str(const char * p, size_t n) {
    try {
        return (int) llm_chat_template_from_str(std::string(p, n));
    } catch (const std::out_of_range &) {
        return -1;
    }
}

// llm_chat_apply_template with `dest` holding `init` beforehand, so a -1
// that leaves it alone can be told from one that clears it. The result is
// malloc'd into `*out`; free it with ref_free_buf.
int32_t ref_apply(int tmpl, const llama_chat_message * chat, size_t n, bool add_ass,
                  const char * init, char ** out, size_t * out_len) {
    std::vector<const llama_chat_message *> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = &chat[i];
    }
    std::string dest(init);
    int32_t r = llm_chat_apply_template((llm_chat_template) tmpl, v, dest, add_ass);
    *out = (char *) malloc(dest.size() + 1);
    memcpy(*out, dest.data(), dest.size());
    (*out)[dest.size()] = 0;
    *out_len = dest.size();
    return r;
}

void ref_free_buf(char * p) {
    free(p);
}

}
