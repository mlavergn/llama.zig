// An app's own C++, on the system libc++, beside the framework's hidden copy.
//
// The framework carries Zig's libc++ and libc++abi, localized by `ld -r`, so
// the process holds two of each. This proves they do not meet: the app's
// containers and exceptions run on the system's, and a C++ exception thrown
// and caught *inside* the framework -- `llama_model_load_from_file` on a
// missing file -- unwinds through its own personality routine and comes back
// as a null return.

#include <llamazig/llama.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    std::vector<std::string> words = {"gamma", "alpha", "beta"};
    std::sort(words.begin(), words.end());

    bool caught = false;
    try {
        throw std::runtime_error("app-side");
    } catch (const std::runtime_error &e) {
        caught = std::string(e.what()) == "app-side";
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model *model = llama_model_load_from_file("/nonexistent/model.gguf", mp);
    llama_backend_free();

    const bool ok = caught && model == nullptr && words.front() == "alpha";
    std::printf("coexist: app exception %s, framework exception %s, std::sort %s\n",
                caught ? "ok" : "FAILED", model == nullptr ? "ok" : "FAILED",
                words.front() == "alpha" ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
