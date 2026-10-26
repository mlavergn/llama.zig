// An app's own C++, on the system libc++, beside the framework.
//
// The framework needs no C++ runtime of its own -- libllama is Zig -- while
// upstream's module map autolinks libc++ for the app. This proves the two
// sit together: the app's containers and exceptions run on the system's, and
// the framework's failure path -- `llama_model_load_from_file` on a missing
// file -- comes back as a null return.

#include <llama/llama.h>

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
