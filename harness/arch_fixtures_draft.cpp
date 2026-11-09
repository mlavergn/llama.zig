// Not a port. Writes fixtures for the five architectures upstream's
// `test-llama-archs.cpp` cannot: graniteswitch, eagle3, dflash, clef and
// vllm-decision2 -- and variants of architectures it can: Qwen3.5 with fused
// QKV, a ModernBERT and an LFM2 decision model. Clef is built as the qwen35 it
// extends and written under its own name (`rename_prefix`), its decision
// head added as extra tensors, because its `load_arch_tensors` sizes the
// head from tensors it finds in the file; vllm-decision2 likewise over
// qwen3.
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
#include "llama-ext.h"

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
    std::normal_distribution<float> dis(0.0f, 0.1f); // test-llama-archs' default -d since v0.5.0
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

// `common_batch_add` as `common/common.cpp` had it at v0.5.0, which v0.6.0
// replaced with `common_batch` over `llama_batch_ext`; a plain
// `llama_batch` is still what `llama_decode` takes.
static void batch_add(llama_batch & batch, llama_token id, llama_pos pos, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id  [batch.n_tokens][0] = 0;
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

// `embeddings`: a model with no LM head (clef, a ModernBERT decision
// model), which is checked for an embeddings output instead of logits.
static bool decode_check(const std::string & path, bool embeddings = false) {
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
    cp.embeddings = embeddings;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        printf("failed %s: the C++ cannot create a context\n", path.c_str());
        llama_model_free(model);
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch b = llama_batch_init(32, 0, 1);
    for (int k = 0; k < 24; k++) {
        batch_add(b, (k * 37 + 11) % n_vocab, k, k == 23);
    }
    const int ret = llama_decode(ctx, b);
    bool ok = ret == 0;
    if (ok && embeddings) {
        const float * e = llama_get_embeddings(ctx);
        ok = e != nullptr;
        if (ok) {
            printf("checked %s: decode ok, embedding[0] %g\n", path.c_str(), e[0]);
        } else {
            printf("failed %s: no embeddings\n", path.c_str());
        }
    } else if (ok) {
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

// LiliCorr's lattice (models/dflash.cpp:651 @8642541b2) builds only for a
// token batch of exactly one block per sequence that carries state
// embeddings, as common/speculative.cpp drafts: so a LiliCorr fixture is
// also run that way once, and must produce the lattice.
static bool lilicorr_check(const std::string & path, int block_size, int top_k) {
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
    llama_set_embeddings_nextn(ctx, true, false);
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> state((size_t) block_size * n_embd);
    for (size_t i = 0; i < state.size(); i++) {
        state[i] = (float) ((i * 2654435761u) % 2003) / 1001.0f - 1.0f;
    }
    llama_batch_ext * eb = llama_batch_ext_init(ctx);
    bool ok = eb != nullptr;
    for (int k = 0; ok && k < block_size; k++) {
        const int32_t idx = llama_batch_ext_add_token(eb, 0, (k * 37 + 11) % n_vocab);
        const llama_pos pos[4] = { k, 0, 0, 0 };
        ok = idx >= 0 && llama_batch_ext_set_pos(eb, idx, pos) &&
            llama_batch_ext_set_embd_state(eb, idx, { state.data() + (size_t) k * n_embd, 1, (size_t) n_embd });
    }
    const int ret = ok ? llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, eb) : -100;
    const float * lattice = ret == 0 ? llama_get_embeddings_nextn(ctx) : nullptr;
    if (lattice) {
        // row 1: the first drafted position's best candidate, and its score from the anchor
        printf("checked %s: lilicorr lattice ok, row 1 candidate %g score %g\n", path.c_str(), lattice[n_embd], lattice[n_embd + top_k]);
    } else {
        printf("failed %s: lilicorr process returned %d\n", path.c_str(), ret);
    }
    if (eb) llama_batch_ext_free(eb);
    llama_free(ctx);
    llama_model_free(model);
    return lattice != nullptr;
}

static bool write_fixture(llm_arch arch, const std::string & name, size_t seed, const std::string & dir,
        const std::function<void(gguf_context *)> & edit_meta,
        const std::function<void(Extra &, llama_model *)> & add_tensors, bool moe = false,
        const std::function<void(gguf_context *)> & edit_out = nullptr, bool embeddings = false,
        bool check = true) {
    gguf_context_ptr meta = get_gguf_ctx(arch, moe);
    edit_meta(meta.get());

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;

    // test-llama-archs' default -d (0.1) since v0.5.0
    tensor_data_params tp = { seed, 0.1f };
    llama_model * model = llama_model_init_from_user(meta.get(), set_tensor_data, &tp, mp);
    if (!model) {
        printf("failed %s: the C++ cannot build the model\n", name.c_str());
        return false;
    }

    Extra extra;
    add_tensors(extra, model);

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, meta.get());
    if (edit_out) {
        edit_out(out);
    }
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
    if (!check) {
        return true;
    }
    return decode_check(path, embeddings);
}

static void none_tensors(Extra &, llama_model *) {}

// Moves every key under `from` to the same name under `to`, so a model
// built as one architecture can be written as another that loads the same
// backbone (clef over qwen35).
static void rename_prefix(gguf_context * g, const std::string & from, const std::string & to) {
    std::vector<std::string> keys;
    for (int64_t i = 0; i < gguf_get_n_kv(g); i++) {
        const std::string k = gguf_get_key(g, i);
        if (k.rfind(from, 0) == 0) {
            keys.push_back(k);
        }
    }
    for (const std::string & k : keys) {
        const int64_t i = gguf_find_key(g, k.c_str());
        const std::string nk = to + k.substr(from.size());
        const char * n = nk.c_str();
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (g, n, gguf_get_val_u8  (g, i)); break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (g, n, gguf_get_val_i8  (g, i)); break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (g, n, gguf_get_val_u16 (g, i)); break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (g, n, gguf_get_val_i16 (g, i)); break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (g, n, gguf_get_val_u32 (g, i)); break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (g, n, gguf_get_val_i32 (g, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (g, n, gguf_get_val_f32 (g, i)); break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (g, n, gguf_get_val_u64 (g, i)); break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (g, n, gguf_get_val_i64 (g, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (g, n, gguf_get_val_f64 (g, i)); break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(g, n, gguf_get_val_bool(g, i)); break;
            // Copied out first: adding a key may move the storage they point into.
            case GGUF_TYPE_STRING: {
                const std::string v = gguf_get_val_str(g, i);
                gguf_set_val_str(g, n, v.c_str());
            } break;
            case GGUF_TYPE_ARRAY: {
                const gguf_type at = gguf_get_arr_type(g, i);
                const size_t    na = gguf_get_arr_n(g, i);
                if (at == GGUF_TYPE_STRING) {
                    std::vector<std::string> v(na);
                    std::vector<const char *> p(na);
                    for (size_t j = 0; j < na; j++) v[j] = gguf_get_arr_str(g, i, j);
                    for (size_t j = 0; j < na; j++) p[j] = v[j].c_str();
                    gguf_set_arr_str(g, n, p.data(), na);
                } else {
                    const char * d = (const char *) gguf_get_arr_data(g, i);
                    const size_t es = at == GGUF_TYPE_UINT64 || at == GGUF_TYPE_INT64 || at == GGUF_TYPE_FLOAT64 ? 8
                                    : at == GGUF_TYPE_UINT32 || at == GGUF_TYPE_INT32 || at == GGUF_TYPE_FLOAT32 ? 4
                                    : at == GGUF_TYPE_UINT16 || at == GGUF_TYPE_INT16 ? 2 : 1;
                    const std::vector<char> v(d, d + na*es);
                    gguf_set_arr_data(g, n, at, v.data(), na);
                }
            } break;
            default: GGML_ABORT("rename_prefix: unknown type of %s", k.c_str());
        }
        gguf_remove_key(g, k.c_str());
    }
}

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
    // Since v0.8.0 upstream's main fills the pool `set_tensor_data` draws from,
    // at test-llama-archs' default -d.
    init_normal_pool(seed, 0.1f);
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
    //
    // Since v0.4.0 the recipe writes rope sections for every architecture,
    // and DFlash takes M-RoPE whenever it has them (llama_model_rope_type).
    // Every DFlash fixture drops them. With M-RoPE an embedding batch must
    // carry four positions per token, and the feature injection the harness
    // decodes carries one: the C++ batch allocator then reads past the
    // positions it was given (llama-batch.cpp:786), and the two sides hash
    // whatever lies there. DSV4 could not keep them anyway: DeepSeek-V4's
    // graph ropes with ggml_rope_ext, and the C++ aborts on an M-RoPE type.
    const llm_arch df = LLM_ARCH_DFLASH;
    auto df_meta = [&](gguf_context * m) {
        const int32_t tl[2] = {1, 2};
        gguf_set_arr_data(m, kv(df, "target_layers").c_str(), GGUF_TYPE_INT32, tl, 2);
        gguf_remove_key(m, kv(df, "rope.dimension_sections").c_str());
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
    // DFlash2: the dynamic convolution around every attention and FFN block,
    // the target's logit transforms, and the selector lattice -- kernel 3
    // over groups of 64 channels, a rank-16 selector over the top 4
    // candidates, blocks of 4. The selector tensors are extra tensors:
    // load_arch_tensors creates them only when the file already has
    // `selector_hidden.weight`. Since v0.7.0 it creates the conv tensors
    // from the conv keys alone (models/dflash.cpp:159, 334 @8642541b2), so
    // the user-data path builds those.
    run([&] {
        return write_fixture(df, "dflash2", seed, dir, [&](gguf_context * m) {
            df_meta(m);
            gguf_remove_key(m, kv(df, "attention.sliding_window").c_str());
            gguf_set_val_u32(m, kv(df, "block_size").c_str(), 4);
            gguf_set_val_u32(m, kv(df, "conv_kernel_size").c_str(), 3);
            gguf_set_val_u32(m, kv(df, "conv_group_size").c_str(), 64);
            gguf_set_val_u32(m, kv(df, "selector_rank").c_str(), 16);
            gguf_set_val_u32(m, kv(df, "selector_top_k").c_str(), 4);
            gguf_set_val_f32(m, kv(df, "logit_scale").c_str(), 0.5f);
            gguf_set_val_f32(m, kv(df, "final_logit_softcapping").c_str(), 30.0f);
        }, [&](Extra & x, llama_model *) {
            x.f32("selector_predecessor.weight", {16, 128}, seed);
            x.f32("selector_successor.weight",   {16, 128}, seed);
            x.f32("selector_hidden.weight",      {256, 16}, seed);
        });
    });
    // DFlash2's dynamic convolution without the selector: since v0.7.0 the
    // conv keys alone make load_arch_tensors create the conv tensors
    // (models/dflash.cpp:159, 334 @8642541b2), where before they needed
    // `selector_hidden.weight` in the file.
    run([&] {
        return write_fixture(df, "dflash2-conv", seed, dir, [&](gguf_context * m) {
            df_meta(m);
            gguf_remove_key(m, kv(df, "attention.sliding_window").c_str());
            gguf_set_val_u32(m, kv(df, "block_size").c_str(), 4);
            gguf_set_val_u32(m, kv(df, "conv_kernel_size").c_str(), 3);
            gguf_set_val_u32(m, kv(df, "conv_group_size").c_str(), 64);
        }, none_tensors);
    });
    // LiliCorr, new at v0.7.0 (models/dflash.cpp:7, 127, 651 @8642541b2): a
    // lattice scorer over each block's top-4 candidates, blocks of 4. Its
    // keys are all it needs: load_arch_tensors creates every lilicorr.* and
    // blk.{n_layer_all + i}.* tensor from them, so the user-data path builds
    // the head. `dflash-lilicorr` has a 64-wide head over two heads and one
    // layer, so the token embeddings go through token_proj; `-wide` is as
    // wide as the decoder (no token_proj), with two layers over four heads,
    // the top 3 of blocks of 5. Each is also run once as a LiliCorr draft
    // block (lilicorr_check), which the plain decode check never reaches.
    auto lilicorr = [&](const char * name, uint32_t hidden, uint32_t heads, uint32_t layers, uint32_t top_k, uint32_t block) {
        const bool ok = write_fixture(df, name, seed, dir, [&](gguf_context * m) {
            df_meta(m);
            gguf_remove_key(m, kv(df, "attention.sliding_window").c_str());
            gguf_set_val_u32(m, kv(df, "block_size").c_str(), block);
            gguf_set_val_u32(m, "lilicorr.hidden_size",     hidden);
            gguf_set_val_u32(m, "lilicorr.candidate_topk",  top_k);
            gguf_set_val_u32(m, "lilicorr.num_heads",       heads);
            gguf_set_val_u32(m, "lilicorr.num_layers",      layers);
            gguf_set_val_f32(m, "lilicorr.vector_eps",      1.0e-6f);
            gguf_set_val_f32(m, "lilicorr.logit_scale",     8.0f);
            gguf_set_val_u32(m, "lilicorr.factor_dim",      32);
            gguf_set_val_u32(m, "lilicorr.mlp_hidden_size", 96);
        }, none_tensors);
        return ok && lilicorr_check(dir + "/" + name + ".gguf", (int) block, (int) top_k);
    };
    run([&] { return lilicorr("dflash-lilicorr", 64, 2, 1, 4, 4); });
    run([&] { return lilicorr("dflash-lilicorr-wide", 256, 4, 2, 3, 5); });
    // A Gemma-style DFlash backbone, new at v0.5.0 (models/dflash.cpp:7, 127,
    // 817 @8642541b2): an embedding and an attention scale, a GELU FFN, K
    // and V sharing one projection (no attn_v, V RMS-normed bare). The
    // recipe already writes the post norms, the per-layer output scale and
    // rope_freqs, so every DFlash fixture has those. Since v0.6.0 it also
    // carries an attention value scale, which scales the attention output;
    // the other DFlash fixtures have none, so both arms run. Until v0.7.0 it
    // had no output head and the token embedding stood in for it; #30111
    // removed that fallback (a draft without a head shares the target's
    // through ctx_other, and Gemma DSpark conversions now require
    // lm_head.weight), so it keeps the recipe's head, and `dflash-nohead`
    // holds the fallback's removal.
    run([&] {
        return write_fixture(df, "dflash-gemma", seed, dir, [&](gguf_context * m) {
            df_meta(m);
            gguf_remove_key(m, kv(df, "attention.sliding_window").c_str());
            gguf_set_val_f32(m, kv(df, "embedding_scale").c_str(), 16.0f);
            gguf_set_val_f32(m, kv(df, "attention.scale").c_str(), 0.0625f);
            gguf_set_val_f32(m, kv(df, "attention.value_scale").c_str(), 0.75f);
            gguf_set_val_str(m, kv(df, "hidden_activation").c_str(), "gelu_pytorch_tanh");
        }, [&](Extra & x, llama_model * model) {
            for (const auto & [tname, t] : llama_internal_get_tensor_map(model)) {
                if (tname.find(".attn_v.") != std::string::npos) {
                    x.replaces.insert(tname);
                }
            }
        });
    });
    // A DFlash with token embeddings and no output head (#30111,
    // models/dflash.cpp:253 @8642541b2): its decoder takes the target's
    // head through ctx_other and asserts on one (dflash.cpp:1056) without
    // it. Until v0.7.0 its own token embedding stood in. A token decode
    // without a target aborts in the C++, so it gets no decode check.
    run([&] {
        return write_fixture(df, "dflash-nohead", seed, dir, [&](gguf_context * m) {
            df_meta(m);
            gguf_remove_key(m, kv(df, "attention.sliding_window").c_str());
        }, [&](Extra & x, llama_model *) {
            x.replaces.insert("output.weight");
        }, false, nullptr, false, false);
    });
    // Qwen3.5 with its full-attention layers' Q (and Q gate), K and V fused
    // into one attn_qkv, which create_tensor_qkv takes when the file has it.
    // v0.5.0 splits that tensor across devices by its own segments
    // (llama-model.cpp:639, 729 @8642541b2): the recipe writes the three
    // separately, so without this no fixture reaches them.
    run([&] {
        return write_fixture(LLM_ARCH_QWEN35, "qwen35-fusedqkv", seed, dir, [&](gguf_context *) {
        }, [&](Extra & x, llama_model * model) {
            const auto & tmap = llama_internal_get_tensor_map(model);
            auto find = [&](const std::string & n) -> ggml_tensor * {
                for (const auto & [tname, t] : tmap) {
                    if (tname == n) return t;
                }
                return nullptr;
            };
            const int n_layer = llama_model_n_layer(model);
            for (int il = 0; il < n_layer; il++) {
                const std::string blk = "blk." + std::to_string(il) + ".";
                ggml_tensor * q = find(blk + "attn_q.weight");
                ggml_tensor * k = find(blk + "attn_k.weight");
                ggml_tensor * v = find(blk + "attn_v.weight");
                if (!q || !k || !v) {
                    continue; // a recurrent layer
                }
                for (const auto & [tname, t] : tmap) {
                    if (tname.rfind(blk + "attn_q.", 0) == 0 || tname.rfind(blk + "attn_k.", 0) == 0 ||
                            tname.rfind(blk + "attn_v.", 0) == 0) {
                        x.replaces.insert(tname);
                    }
                }
                x.f32((blk + "attn_qkv.weight").c_str(), {q->ne[0], q->ne[1] + k->ne[1] + v->ne[1]}, seed);
            }
        });
    });
    // DSpark: the Markov and confidence heads over a 64-token draft
    // vocabulary, blocks of 8, a bonus anchor slot.
    // Since v0.4.0 the confidence head is optional; one fixture leaves it out.
    auto dspark = [&](const char * name, bool anchor_key, bool with_conf = true) {
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
            if (with_conf) {
                x.f32("conf_proj.weight", {256 + 16, 1}, seed);
                x.f32("conf_proj.bias", {1}, seed);
            }
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
    // DeepSeek-V4.1's DSpark draft, new at v0.8.0 (models/dflash.cpp:1294
    // @8642541b2): the same backbone with single-pass hyper-connections, no
    // hc_head, and a window one wider. With a Markov head in `-dspark`.
    auto dsv41 = [&](const char * name, bool markov) {
        return write_fixture(df, name, seed, dir, [&](gguf_context * m) {
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
            gguf_set_val_bool(m, kv(df, "hyper_connection.single_pass").c_str(), true);
            gguf_set_val_u32(m, kv(df, "attention.output_group_count").c_str(), 8);
            gguf_set_val_u32(m, kv(df, "attention.output_lora_rank").c_str(), 32);
            gguf_set_val_f32(m, kv(df, "swiglu_clamp_exp").c_str(), 10.0f);
            gguf_set_val_f32(m, kv(df, "expert_weights_scale").c_str(), 1.0f);
            gguf_set_val_bool(m, kv(df, "expert_weights_norm").c_str(), true);
            gguf_set_val_u32(m, kv(df, "expert_gating_func").c_str(), 4);
            if (markov) gguf_set_val_u32(m, "dflash.block_size", 8);
        }, [&](Extra & x, llama_model *) {
            if (!markov) return;
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
        }, true);
    };
    run([&] { return dsv41("dflash-dsv41-moe", false); });
    run([&] { return dsv41("dflash-dsv41-dspark-moe", true); });
    run([&] { return dspark("dflash-dspark-bonus", true); });
    run([&] { return dspark("dflash-dspark", false); });
    run([&] { return dspark("dflash-dspark-noconf", false, false); });

    // clef, new at v0.6.0 (models/clef.cpp:8, 26 @8642541b2): a Qwen3.5
    // backbone with a joint decision head. Upstream's recipe skips it ("TODO
    // decision head tensors") and the user-data path cannot build it, since
    // load_arch_tensors sizes the head from tensors it finds in the file. So
    // the model is built as qwen35 -- whose recipe keys clef reads unchanged
    // -- and written as clef, the head's tensors added: a width of 64 over
    // two heads, a 96-wide FFN, one routing block and one joint block.
    run([&] {
        const int64_t n_embd = 256, n_embd_h = 64, n_ff_h = 96;
        return write_fixture(LLM_ARCH_QWEN35, "clef-dense", seed, dir, [&](gguf_context *) {
        }, [&](Extra & x, llama_model * model) {
            // qwen35's optional embeddings projection is n_embd wide in the
            // recipe; clef's output is one score, so the file carries none
            for (const auto & [tname, t] : llama_internal_get_tensor_map(model)) {
                if (tname.rfind("cls.", 0) == 0) x.replaces.insert(tname);
            }
            auto norm = [&](const std::string & n, int64_t size) {
                x.f32((n + ".weight").c_str(), {size}, seed);
                x.f32((n + ".bias").c_str(),   {size}, seed);
            };
            auto attn = [&](const std::string & p) {
                for (const char * s : {"q", "k", "v", "o"}) {
                    x.f32((p + s + ".weight").c_str(), {n_embd_h, n_embd_h}, seed);
                    x.f32((p + s + ".bias").c_str(),   {n_embd_h},           seed);
                }
            };
            for (int il = 0; il < 2; il++) {
                const std::string blk = "dec.blk." + std::to_string(il) + ".";
                if (il < 1) {
                    norm(blk + "cross_attn_norm_kv", n_embd_h);
                } else {
                    norm(blk + "attn_norm", n_embd_h);
                    attn(blk + "attn_");
                }
                norm(blk + "cross_attn_norm", n_embd_h);
                attn(blk + "cross_attn_");
                norm(blk + "ffn_norm", n_embd_h);
                x.f32((blk + "ffn_up.weight").c_str(),   {n_embd_h, n_ff_h}, seed);
                x.f32((blk + "ffn_up.bias").c_str(),     {n_ff_h},           seed);
                x.f32((blk + "ffn_down.weight").c_str(), {n_ff_h, n_embd_h}, seed);
                x.f32((blk + "ffn_down.bias").c_str(),   {n_embd_h},         seed);
            }
            norm("decision.hidden_norm",         n_embd);
            norm("decision.option_summary_norm", n_embd_h);
            norm("decision.field_norm",          n_embd_h);
            norm("decision.option_norm",         n_embd_h);
            for (const char * p : {"memory", "question", "option_question", "global", "option_context", "option_lexical"}) {
                x.f32((std::string("decision.proj_") + p + ".weight").c_str(), {n_embd, n_embd_h}, seed);
            }
            x.f32("decision.scales",            {3},                       seed);
            x.f32("token_types.weight",         {n_embd_h, 3},             seed);
            x.f32("decision.scorer.weight",     {4*n_embd_h, n_embd_h},    seed);
            x.f32("decision.scorer.bias",       {n_embd_h},                seed);
            x.f32("decision.scorer_out.weight", {n_embd_h, 1},             seed);
            x.f32("decision.scorer_out.bias",   {1},                       seed);
        }, false, [&](gguf_context * g) {
            rename_prefix(g, "qwen35.", "clef.");
            gguf_set_val_str(g, "general.architecture", "clef");
            gguf_set_val_u32(g, "clef.decision.routing_block_count", 1);
            gguf_set_val_u32(g, "clef.decision.block_count",         1);
            gguf_set_val_u32(g, "clef.decision.head_count",          2);
        }, true);
    });

    // vllm-decision2, new at v0.7.0 (models/vllm-decision2.cpp:7, 17
    // @8642541b2): a Qwen3 backbone with a decision head. Upstream's recipe
    // skips it ("TODO decision head tensors"), and its load_arch_tensors
    // sizes the head from `decision.key` in the file, so it is built as
    // qwen3 -- whose recipe keys, layer_norm_epsilon among them, it reads
    // unchanged -- and written as vllm-decision2, the head's tensors added
    // at a width of 64.
    run([&] {
        const int64_t n_embd = 256, n_embd_h = 64;
        return write_fixture(LLM_ARCH_QWEN3, "vllm-decision2-dense", seed, dir, [&](gguf_context *) {
        }, [&](Extra & x, llama_model * model) {
            // Since v0.8.0 qwen3 creates the optional dense_2 projection,
            // which the user-data path builds and vllm-decision2 does not
            // load.
            for (const auto & [tname, t] : llama_internal_get_tensor_map(model)) {
                if (tname.rfind("dense_2.", 0) == 0) x.replaces.insert(tname);
            }
            for (const char * n : {"decision.option_norm", "decision.query_norm"}) {
                x.f32((std::string(n) + ".weight").c_str(), {n_embd}, seed);
                x.f32((std::string(n) + ".bias").c_str(),   {n_embd}, seed);
            }
            x.f32("decision.key.weight",        {n_embd, n_embd_h}, seed);
            x.f32("decision.query.weight",      {n_embd, n_embd_h}, seed);
            x.f32("decision.option_mlp.weight", {n_embd, n_embd_h}, seed);
            x.f32("decision.option_mlp.bias",   {n_embd_h},         seed);
            x.f32("decision.query_mlp.weight",  {n_embd, n_embd_h}, seed);
            x.f32("decision.scorer_out.weight", {n_embd_h, 1},      seed);
        }, false, [&](gguf_context * g) {
            rename_prefix(g, "qwen3.", "vllm-decision2.");
            gguf_set_val_str(g, "general.architecture", "vllm-decision2");
        }, true);
    });

    // A ModernBERT decision model, new at v0.6.0 (models/modern-bert.cpp:6,
    // 61, 248 @8642541b2): the recipe's second block becomes the head's --
    // LayerNorms, QKV, output and FFN all with biases, the FFN not gated --
    // over three token types, scored through the classifier. The user-data
    // path builds all of it from the keys.
    run([&] {
        return write_fixture(LLM_ARCH_MODERN_BERT, "modern-bert-decision", seed, dir, [&](gguf_context * m) {
            // four blocks: the recipe's SWA pattern (2, dense first) makes
            // blocks 1 and 3 SWA, so the encoder keeps one and the head's
            // block is one load_arch_hparams must turn dense. With two, the
            // head takes the only SWA block and the C++ asserts on the
            // unused SWA mask's buffer.
            gguf_set_val_u32(m, "modern-bert.block_count", 4);
            gguf_set_val_u32(m, "modern-bert.decision.block_count", 1);
            gguf_set_val_u32(m, "tokenizer.ggml.token_type_count", 3);
        }, none_tensors, false, nullptr, true);
    });

    // An LFM2 decision model (LiquidAI's d1-omni), new at v0.7.0
    // (models/lfm2.cpp:29, 53, 317 @8642541b2): four blocks, the first two
    // -- an attention and a short convolution, as the recipe has them --
    // the non-causal trunk, the last two the head: LayerNorms, fused QKV,
    // output and an ungated FFN narrower than the trunk's, all with biases,
    // over three token types, scored through the classifier. No sliding
    // window: the trunk's no-cache attention input has no SWA mask. The
    // user-data path builds all of it from the keys.
    run([&] {
        const llm_arch lf = LLM_ARCH_LFM2;
        return write_fixture(lf, "lfm2-decision", seed, dir, [&](gguf_context * m) {
            gguf_set_val_u32(m, kv(lf, "block_count").c_str(), 4);
            gguf_set_val_u32(m, kv(lf, "decision.block_count").c_str(), 2);
            gguf_set_val_bool(m, kv(lf, "attention.causal").c_str(), false);
            gguf_remove_key(m, kv(lf, "attention.sliding_window").c_str());
            const uint32_t heads[4] = {2, 0, 2, 2};
            gguf_set_arr_data(m, kv(lf, "attention.head_count").c_str(), GGUF_TYPE_UINT32, heads, 4);
            gguf_set_arr_data(m, kv(lf, "attention.head_count_kv").c_str(), GGUF_TYPE_UINT32, heads, 4);
            const uint32_t ff[4] = {384, 384, 192, 192};
            gguf_set_arr_data(m, kv(lf, "feed_forward_length").c_str(), GGUF_TYPE_UINT32, ff, 4);
            gguf_set_val_u32(m, "tokenizer.ggml.token_type_count", 3);
        }, [&](Extra & x, llama_model * model) {
            // the user-data path also builds ColBERT's optional dense_2
            // projection, n_embd_out (3 scores here) wide, which no decision
            // model has: its graph would project the scores again
            for (const auto & [tname, t] : llama_internal_get_tensor_map(model)) {
                if (tname.rfind("dense_2.", 0) == 0) x.replaces.insert(tname);
            }
        }, false, nullptr, true);
    });

    // Voxtral Realtime's adaptive RMS norm, new at v0.8.0 (models/mistral3.cpp:58,
    // 99, 227 @8642541b2): Mistral 3 with a per-layer two-matrix MLP over a
    // sinusoid of the output delay. load_arch_tensors finds it in the file,
    // so the tensors are extra.
    run([&] {
        const llm_arch ms3 = LLM_ARCH_MISTRAL3;
        return write_fixture(ms3, "mistral3-ada", seed, dir, [&](gguf_context * m) {
            gguf_set_val_u32(m, kv(ms3, "realtime_delay_tokens").c_str(), 6);
        }, [&](Extra & x, llama_model * model) {
            const int n_layer = llama_model_n_layer(model);
            for (int il = 0; il < n_layer; il++) {
                const std::string blk = "blk." + std::to_string(il) + ".";
                x.f32((blk + "ffn_norm_ada_down.weight").c_str(), {256, 32}, seed);
                x.f32((blk + "ffn_norm_ada_up.weight").c_str(),   {32, 256}, seed);
            }
        });
    });

    // Qwen4exp with the merged weights the converter writes since v0.8.0
    // (`create_merged`, models/qwen4exp.cpp:235 @8642541b2): each pair of
    // matmuls that read one input in one tensor, the rows split in the graph.
    // The user-data path skips them, being virtual there.
    run([&] {
        return write_fixture(LLM_ARCH_QWEN4EXP, "qwen4exp-merged", seed, dir, [&](gguf_context *) {
        }, [&](Extra & x, llama_model * model) {
            const auto & tmap = llama_internal_get_tensor_map(model);
            auto find = [&](const std::string & n) -> ggml_tensor * {
                for (const auto & [tname, t] : tmap) {
                    if (tname == n) return t;
                }
                return nullptr;
            };
            const int n_layer = llama_model_n_layer(model);
            auto merge = [&](const std::string & merged, const std::string & a, const std::string & b) {
                ggml_tensor * ta = find(a + ".weight");
                ggml_tensor * tb = find(b + ".weight");
                if (!ta || !tb) return;
                // with the optional scales the user-data path builds
                for (const auto & [tname, t] : tmap) {
                    if (tname.rfind(a + ".", 0) == 0 || tname.rfind(b + ".", 0) == 0) x.replaces.insert(tname);
                }
                x.f32((merged + ".weight").c_str(), {ta->ne[0], ta->ne[1] + tb->ne[1]}, seed);
            };
            for (int il = 0; il < n_layer; il++) {
                const std::string blk = "blk." + std::to_string(il) + ".";
                merge(blk + "hc_attn_down_inject", blk + "hc_attn_down", blk + "hc_attn_inject");
                merge(blk + "hc_ffn_down_inject",  blk + "hc_ffn_down",  blk + "hc_ffn_inject");
                merge(blk + "attn_qkv_gate",       blk + "attn_qkv",     blk + "attn_gate");
                merge(blk + "ssm_beta_alpha",      blk + "ssm_beta",     blk + "ssm_alpha");
            }
        }, true);
    });

    llama_backend_free();
    return failed == 0 ? 0 : 1;
}
