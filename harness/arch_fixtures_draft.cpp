// Not a port. Writes fixtures for the three architectures upstream's
// `test-llama-archs.cpp` cannot: graniteswitch, eagle3 and dflash.
//
// `llama.cpp/tests/test-llama-archs.cpp` is included unmodified, as
// `harness/arch_fixtures_extra.cpp` does, so `get_gguf_ctx` and
// `set_tensor_data` are upstream's. Each fixture adds to that metadata only
// what the architecture's `load_arch_hparams` requires and the recipe lacks,
// builds the model through the C++ on the CPU, and writes its tensors. Two
// things the user-data path cannot do are done by hand afterwards:
// `get_tensor_meta` sees no tensors there, so eagle3's own `token_embd`, and
// the optional `d2t` and DSpark tensors, are added to the file as extra
// tensors filled the same way. Each written file is then reloaded from disk
// through the C++ and decoded once, so a fixture the C++ refuses is reported.
//
// Usage: arch_fixtures_draft <out-dir> <seed>, as scripts/gen-arch-fixtures runs it.

#define main test_llama_archs_main
#include "test-llama-archs.cpp"
#undef main

#include "llama-model.h"

#include <cstdlib>
#include <functional>
#include <set>

static std::string kv(llm_arch arch, const char * suffix) {
    return std::string(llm_arch_name(arch)) + "." + suffix;
}

// set_tensor_data's distribution, written to host memory.
static void fill_host(ggml_tensor * t, size_t seed) {
    std::hash<std::string> hasher;
    seed ^= hasher(t->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        d[i] = dis(gen);
    }
}

struct Extra {
    ggml_context * ctx = nullptr;
    std::vector<ggml_tensor *> tensors;
    std::set<std::string> replaces;

    Extra() {
        ggml_init_params p = { 64*1024*1024, nullptr, false };
        ctx = ggml_init(p);
    }
    ~Extra() { ggml_free(ctx); }

    ggml_tensor * f32(const char * name, std::initializer_list<int64_t> ne, size_t seed) {
        int64_t n[4] = {1, 1, 1, 1};
        int i = 0;
        for (auto x : ne) n[i++] = x;
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(), n);
        ggml_set_name(t, name);
        fill_host(t, seed);
        tensors.push_back(t);
        return t;
    }
};

static bool decode_check(const std::string & path) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        printf("failed %s: the C++ cannot load it\n", path.c_str());
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 0;
    cp.n_ubatch = 64;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        printf("failed %s: the C++ cannot create a context\n", path.c_str());
        llama_model_free(model);
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch b = llama_batch_init(32, 0, 1);
    for (int k = 0; k < 24; k++) {
        common_batch_add(b, (k * 37 + 11) % n_vocab, k, {0}, k == 23);
    }
    const int ret = llama_decode(ctx, b);
    bool ok = ret == 0;
    if (ok) {
        const float * l = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int j = 1; j < n_vocab; j++) if (l[j] > l[best]) best = j;
        printf("checked %s: decode ok, argmax %d (logit %g)\n", path.c_str(), best, l[best]);
    } else {
        printf("failed %s: decode returned %d\n", path.c_str(), ret);
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

static bool write_fixture(llm_arch arch, const std::string & name, size_t seed, const std::string & dir,
        const std::function<void(gguf_context *)> & edit_meta,
        const std::function<void(Extra &, llama_model *)> & add_tensors, bool moe = false) {
    gguf_context_ptr meta = get_gguf_ctx(arch, moe);
    edit_meta(meta.get());

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;

    size_t tmp = seed;
    llama_model * model = llama_model_init_from_user(meta.get(), set_tensor_data, &tmp, mp);
    if (!model) {
        printf("failed %s: the C++ cannot build the model\n", name.c_str());
        return false;
    }

    Extra extra;
    add_tensors(extra, model);

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, meta.get());
    for (const auto & [tname, t] : llama_internal_get_tensor_map(model)) {
        if (gguf_find_tensor(out, tname.c_str()) >= 0 || extra.replaces.count(tname)) {
            continue;
        }
        gguf_add_tensor(out, t);
    }
    for (ggml_tensor * t : extra.tensors) {
        gguf_add_tensor(out, t);
    }

    const std::string path = dir + "/" + name + ".gguf";
    const bool ok = gguf_write_to_file(out, path.c_str(), false);
    gguf_free(out);
    llama_model_free(model);
    if (!ok) {
        printf("failed %s: cannot write %s\n", name.c_str(), path.c_str());
        return false;
    }
    printf("wrote %s\n", path.c_str());
    return decode_check(path);
}

static void none_tensors(Extra &, llama_model *) {}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <out-dir> <seed>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);

    const std::string dir = argv[1];
    const size_t seed = std::strtoull(argv[2], nullptr, 10);
    int failed = 0;
    auto run = [&](auto && f) {
        try {
            if (!f()) failed++;
        } catch (const std::exception & e) {
            printf("failed: %s\n", e.what());
            failed++;
        }
    };

    // graniteswitch: two adapters of rank 8. The activation tokens are two
    // of the ids `harness/context_diff.zig` decodes, (k*37 + 11) % 128 at
    // k = 1 and k = 6, so the in-graph router switches mid-prompt.
    const llm_arch gs = LLM_ARCH_GRANITE_SWITCH;
    run([&] {
        return write_fixture(gs, "graniteswitch-dense", seed, dir, [&](gguf_context * m) {
            gguf_set_val_u32(m, kv(gs, "adapters.count").c_str(), 2);
            gguf_set_val_u32(m, kv(gs, "adapters.lora_rank").c_str(), 8);
            const int32_t act[2] = {48, 105};
            const int32_t sub[2] = {3, 7};
            gguf_set_arr_data(m, kv(gs, "adapters.token_ids_activate").c_str(), GGUF_TYPE_INT32, act, 2);
            gguf_set_arr_data(m, kv(gs, "adapters.token_ids_substitute").c_str(), GGUF_TYPE_INT32, sub, 2);
        }, none_tensors);
    });

    // eagle3: one decoder layer, three target layers of a 128-wide target,
    // and its own token embeddings.
    const llm_arch e3 = LLM_ARCH_EAGLE3;
    auto e3_meta = [&](gguf_context * m) {
        gguf_set_val_u32(m, "eagle3.block_count", 1);
        const int32_t tl[3] = {0, 1, 2};
        gguf_set_arr_data(m, kv(e3, "target_layers").c_str(), GGUF_TYPE_INT32, tl, 3);
        gguf_set_val_u32(m, kv(e3, "target_hidden_size").c_str(), 128);
    };
    run([&] {
        return write_fixture(e3, "eagle3-dense", seed, dir, e3_meta, [&](Extra & x, llama_model *) {
            x.f32("token_embd.weight", {256, 128}, seed);
        });
    });
    // eagle3 with a 64-token draft vocabulary mapped through d2t, and both
    // norm flags set.
    run([&] {
        return write_fixture(e3, "eagle3-d2t", seed, dir, [&](gguf_context * m) {
            e3_meta(m);
            gguf_set_val_bool(m, kv(e3, "norm_before_residual").c_str(), true);
            gguf_set_val_bool(m, kv(e3, "norm_before_fc").c_str(), true);
        }, [&](Extra & x, llama_model *) {
            x.f32("token_embd.weight", {256, 128}, seed);
            x.replaces.insert("output.weight");
            x.f32("output.weight", {256, 64}, seed);
            ggml_tensor * d2t = ggml_new_tensor_1d(x.ctx, GGML_TYPE_I64, 64);
            ggml_set_name(d2t, "d2t");
            for (int j = 0; j < 64; j++) ((int64_t *) d2t->data)[j] = (j * 5 + 3) % 128;
            x.tensors.push_back(d2t);
        });
    });

    // dflash: two target layers of the draft's own width. The recipe's
    // scalar sliding-window pattern is not the array dflash reads, so the
    // dense fixture drops the window and the iSWA one gives a pattern.
    const llm_arch df = LLM_ARCH_DFLASH;
    auto df_meta = [&](gguf_context * m) {
        const int32_t tl[2] = {1, 2};
        gguf_set_arr_data(m, kv(df, "target_layers").c_str(), GGUF_TYPE_INT32, tl, 2);
    };
    run([&] {
        return write_fixture(df, "dflash-dense", seed, dir, [&](gguf_context * m) {
            df_meta(m);
            gguf_remove_key(m, kv(df, "attention.sliding_window").c_str());
        }, none_tensors);
    });
    run([&] {
        return write_fixture(df, "dflash-iswa", seed, dir, [&](gguf_context * m) {
            df_meta(m);
            const uint32_t pat[2] = {1, 0};
            gguf_set_arr_data(m, kv(df, "attention.sliding_window_pattern").c_str(), GGUF_TYPE_UINT32, pat, 2);
        }, none_tensors);
    });
    // DSpark: the Markov and confidence heads over a 64-token draft
    // vocabulary, blocks of 8, a bonus anchor slot.
    auto dspark = [&](const char * name, bool anchor_key) {
        return write_fixture(df, name, seed, dir, [&](gguf_context * m) {
            df_meta(m);
            gguf_remove_key(m, kv(df, "attention.sliding_window").c_str());
            gguf_set_val_u32(m, "dflash.block_size", 8);
            if (anchor_key) gguf_set_val_bool(m, "dflash.sample_from_anchor", false);
        }, [&](Extra & x, llama_model *) {
            x.replaces.insert("output.weight");
            x.f32("output.weight", {256, 64}, seed);
            ggml_tensor * d2t = ggml_new_tensor_1d(x.ctx, GGML_TYPE_I64, 64);
            ggml_set_name(d2t, "d2t");
            for (int j = 0; j < 64; j++) ((int64_t *) d2t->data)[j] = (j * 7 + 1) % 128;
            x.tensors.push_back(d2t);
            x.f32("markov_w1.weight", {16, 128}, seed);
            x.f32("markov_w2.weight", {16, 64}, seed);
            x.f32("conf_proj.weight", {256 + 16, 1}, seed);
            x.f32("conf_proj.bias", {1}, seed);
        });
    };
    // DFlash's DeepSeek-V4 DSpark backbone: the MoE recipe, with what
    // load_arch_hparams' DSV4 branch requires -- hyper-connections, the
    // grouped output LoRA, sqrtsoftplus scoring -- set as test-llama-archs
    // sets them for deepseek4, eight heads of 64 sharing one KV head, with a 32-wide RoPE tail.
    run([&] {
        return write_fixture(df, "dflash-dsv4-moe", seed, dir, [&](gguf_context * m) {
            df_meta(m);
            gguf_set_val_u32(m, kv(df, "attention.head_count").c_str(), 8);
            gguf_set_val_u32(m, kv(df, "attention.head_count_kv").c_str(), 1);
            gguf_set_val_u32(m, kv(df, "attention.key_length").c_str(), 64);
            gguf_set_val_u32(m, kv(df, "attention.value_length").c_str(), 64);
            gguf_set_val_u32(m, kv(df, "rope.dimension_count").c_str(), 32);
            gguf_set_val_u32(m, kv(df, "attention.q_lora_rank").c_str(), 64);
            gguf_set_val_u32(m, kv(df, "hyper_connection.count").c_str(), 4);
            gguf_set_val_u32(m, kv(df, "hyper_connection.sinkhorn_iterations").c_str(), 2);
            gguf_set_val_f32(m, kv(df, "hyper_connection.epsilon").c_str(), 1.0e-6f);
            gguf_set_val_u32(m, kv(df, "attention.output_group_count").c_str(), 8);
            gguf_set_val_u32(m, kv(df, "attention.output_lora_rank").c_str(), 32);
            gguf_set_val_f32(m, kv(df, "swiglu_clamp_exp").c_str(), 10.0f);
            gguf_set_val_f32(m, kv(df, "expert_weights_scale").c_str(), 1.0f);
            gguf_set_val_bool(m, kv(df, "expert_weights_norm").c_str(), true);
            gguf_set_val_u32(m, kv(df, "expert_gating_func").c_str(), 4);
        }, none_tensors, true);
    });
    run([&] { return dspark("dflash-dspark-bonus", true); });
    run([&] { return dspark("dflash-dspark", false); });

    llama_backend_free();
    return failed == 0 ? 0 : 1;
}
