// Not a port. Writes the per-architecture fixtures `test-llama-archs.cpp`'s
// own `-o` mode will not: the architectures `llama_model_saver` refuses to
// save from a loaded model (`llama_model_saver_supports_arch`), the ones its
// `arch_supported` marks FIXME, and the four its loops skip outright.
//
// `llama.cpp/tests/test-llama-archs.cpp` is included textually, unmodified,
// with its `main` renamed, so its static `get_gguf_ctx`, `set_tensor_data`,
// `moe_mandatory` and `moe_implemented` are upstream's definition of the
// fixture here exactly as in `scripts/gen-arch-fixtures`. What differs is
// only how the file is written: the saver rebuilds the metadata from the
// loaded model's hparams, which is what it cannot do for these
// architectures; this keeps `get_gguf_ctx`'s metadata as built, loads the
// model through the C++ on the CPU to create and fill its tensors, and adds
// each tensor by name, once, as `llama_model_saver::add_tensor` does.
//
// One addition to `get_gguf_ctx`'s metadata, for the BERT family alone:
// `tokenizer.ggml.token_type_count` = 2. Their `load_arch_tensors` sizes the
// token-type embedding from it and throws without it, which is all
// upstream's "TODO vocab" stands for: `llama_vocab::impl::load` reads the
// count before its `no_vocab` return, so no vocabulary is needed. And for
// `nomic-bert-moe`, whose MoE upstream's rules leave out, a second fixture
// with `moe_every_n_layers` = 2, so that layer 1 routes to experts.
//
// And one variant, `--variant hs64`, for the RWKV family: Metal's WKV
// kernels need a head size of 64 (`GGML_ASSERT(C / H == 64)`), and
// `get_gguf_ctx`'s n_embd 256 over two heads is 128, so on the GPU every
// case aborts in ggml before the architecture's code is reached. hs64 sets
// `wkv.head_size` 64 over four heads, and for RWKV6-Qwen2 a
// `token_shift_count` of 1, without which the C++ asserts `n_embd ==
// n_embd_r()` itself. RWKV-7 and ARWKV-7 additionally get the LoRA ranks
// they read, and `--variant hs64nogate` writes ARWKV-7 without a gate rank,
// its gateless form. Written as `<arch>-<variant>-dense.gguf`.
//
// Usage: arch_fixtures_extra <out-dir> <seed> [--variant hs64] <arch>...
// Prints one line per fixture: `wrote <path>` or `failed <arch> (<config>): <why>`.

#define main test_llama_archs_main
#include "test-llama-archs.cpp"
#undef main

#include "llama-model.h"

#include <cstdlib>

static bool is_bert_family(llm_arch arch) {
    return arch == LLM_ARCH_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 ||
           arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE;
}

static bool write_fixture(llm_arch arch, bool moe, size_t seed, const std::string & dir, const std::string & variant) {
    gguf_context_ptr meta = get_gguf_ctx(arch, moe);
    if (variant == "hs64") {
        const std::string a = llm_arch_name(arch);
        gguf_set_val_u32(meta.get(), (a + ".wkv.head_size").c_str(), 64);
        gguf_set_val_u32(meta.get(), (a + ".attention.head_count").c_str(), 4);
        gguf_set_val_u32(meta.get(), (a + ".attention.head_count_kv").c_str(), 4);
        if (arch == LLM_ARCH_RWKV6QWEN2) {
            gguf_set_val_u32(meta.get(), (a + ".token_shift_count").c_str(), 1);
        }
    }
    // RWKV-7 and ARWKV-7 read four LoRA ranks `get_gguf_ctx` never writes,
    // which is the whole of why the C++ cannot build them from it.
    if ((variant == "hs64" || variant == "hs64nogate") && (arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7)) {
        const std::string a = llm_arch_name(arch);
        gguf_set_val_u32(meta.get(), (a + ".wkv.head_size").c_str(), 64);
        gguf_set_val_u32(meta.get(), (a + ".attention.head_count").c_str(), 4);
        gguf_set_val_u32(meta.get(), (a + ".attention.head_count_kv").c_str(), 4);
        gguf_set_val_u32(meta.get(), (a + ".attention.decay_lora_rank").c_str(), 32);
        gguf_set_val_u32(meta.get(), (a + ".attention.iclr_lora_rank").c_str(), 16);
        gguf_set_val_u32(meta.get(), (a + ".attention.value_residual_mix_lora_rank").c_str(), 16);
        // ARWKV-7 without a gate rank is its gateless form.
        if (variant == "hs64") {
            gguf_set_val_u32(meta.get(), (a + ".attention.gate_lora_rank").c_str(), 32);
        }
        gguf_set_val_u32(meta.get(), (a + ".token_shift_count").c_str(), arch == LLM_ARCH_RWKV7 ? 2 : 1);
    }
    // Since v0.4.1 Gemma 4 reads its SWA pattern with `get_arr`, which
    // refuses the scalar `get_gguf_ctx` still writes; upstream's own loop
    // skips the architecture, so nothing there noticed. The array is the
    // recipe's comment: every fifth layer full attention.
    if (arch == LLM_ARCH_GEMMA4) {
        const std::string a = llm_arch_name(arch);
        const uint32_t n_layer = gguf_get_val_u32(meta.get(), gguf_find_key(meta.get(), (a + ".block_count").c_str()));
        std::vector<char> pattern(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern[il] = il % 5 != 4;
        }
        gguf_set_arr_data(meta.get(), (a + ".attention.sliding_window_pattern").c_str(), GGUF_TYPE_BOOL, pattern.data(), pattern.size());
    }
    if (is_bert_family(arch)) {
        gguf_set_val_u32(meta.get(), "tokenizer.ggml.token_type_count", 2);
    }
    if (arch == LLM_ARCH_NOMIC_BERT_MOE && moe) {
        const std::string key = std::string(llm_arch_name(arch)) + ".moe_every_n_layers";
        gguf_set_val_u32(meta.get(), key.c_str(), 2);
    }

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;

    size_t tmp = seed;
    llama_model * model = llama_model_init_from_user(meta.get(), set_tensor_data, &tmp, mp);
    if (!model) {
        printf("failed %s (%s): the C++ cannot build the model\n", llm_arch_name(arch), moe ? "MoE" : "dense");
        return false;
    }

    gguf_context * out = gguf_init_empty();
    // Tensors made here, alive until the file is written.
    std::vector<ggml_context *> keep;
    gguf_set_kv(out, meta.get());
    for (const auto & [name, t] : llama_internal_get_tensor_map(model)) {
        if (gguf_find_tensor(out, name.c_str()) >= 0) {
            continue;
        }
        // ARWKV-7 without a gate has five lerps, not six: upstream loads
        // `time_mix_lerp_fused` six wide, and on failure five. Building the
        // model makes the six-wide one, so the file gets its first five.
        if (arch == LLM_ARCH_ARWKV7 && variant == "hs64nogate" && name.find("time_mix_lerp_fused") != std::string::npos) {
            ggml_init_params ip = { ggml_nbytes(t), nullptr, false };
            ip.mem_size = ggml_nbytes(t) + ggml_tensor_overhead();
            ggml_context * tctx = ggml_init(ip);
            ggml_tensor * five = ggml_new_tensor_4d(tctx, t->type, t->ne[0], t->ne[1], t->ne[2], 5);
            ggml_set_name(five, t->name);
            ggml_backend_tensor_get(t, five->data, 0, ggml_nbytes(five));
            gguf_add_tensor(out, five);
            keep.push_back(tctx);
            continue;
        }
        // RWKV6-Qwen2 with `time_mix_first` takes the non-QRWKV path, which
        // reads `time_mix_ln` it never loads: the C++ crashes. Building the
        // model creates the optional tensor; the file leaves it out.
        if (arch == LLM_ARCH_RWKV6QWEN2 && name.find("time_mix_first") != std::string::npos) {
            continue;
        }
        gguf_add_tensor(out, t);
    }

    const std::string path = dir + "/" + llm_arch_name(arch) + (variant.empty() ? "" : "-" + variant) + (moe ? "-moe.gguf" : "-dense.gguf");
    const bool ok = gguf_write_to_file(out, path.c_str(), false);
    gguf_free(out);
    for (ggml_context * k : keep) ggml_free(k);
    llama_model_free(model);
    if (!ok) {
        printf("failed %s (%s): cannot write %s\n", llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
        return false;
    }
    printf("wrote %s\n", path.c_str());
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <out-dir> <seed> <arch>...\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);

    const std::string dir = argv[1];
    const size_t seed = std::strtoull(argv[2], nullptr, 10);
    std::string variant;
    int first = 3;
    if (argc > 4 && std::string(argv[3]) == "--variant") {
        variant = argv[4];
        first = 5;
    }
    int failed = 0;
    for (int i = first; i < argc; i++) {
        const llm_arch arch = llm_arch_from_string(argv[i]);
        if (arch == LLM_ARCH_UNKNOWN) {
            printf("failed %s: unknown architecture\n", argv[i]);
            failed++;
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch) && arch != LLM_ARCH_NOMIC_BERT_MOE) continue;
            if (!moe && moe_mandatory(arch)) continue;
            try {
                if (!write_fixture(arch, moe, seed, dir, variant)) failed++;
            } catch (const std::exception & e) {
                printf("failed %s (%s): %s\n", argv[i], moe ? "MoE" : "dense", e.what());
                failed++;
            }
        }
    }
    llama_backend_free();
    return failed == 0 ? 0 : 1;
}
