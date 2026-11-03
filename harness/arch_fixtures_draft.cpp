// Not a port. Writes fixtures for the four architectures upstream's
// `test-llama-archs.cpp` cannot: graniteswitch, eagle3, dflash and clef --
// and variants of architectures it can: Qwen3.5 with fused QKV, and a
// ModernBERT decision model. Clef is built as the qwen35 it extends and
// written under its own name (`rename_prefix`), its decision head added as
// extra tensors, because its `load_arch_tensors` sizes the head from
// tensors it finds in the file.
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

static bool write_fixture(llm_arch arch, const std::string & name, size_t seed, const std::string & dir,
        const std::function<void(gguf_context *)> & edit_meta,
        const std::function<void(Extra &, llama_model *)> & add_tensors, bool moe = false,
        const std::function<void(gguf_context *)> & edit_out = nullptr, bool embeddings = false) {
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
    // candidates, blocks of 4. The conv and selector tensors are extra
    // tensors: load_arch_tensors creates them only when the file already has
    // `selector_hidden.weight`.
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
        }, [&](Extra & x, llama_model * model) {
            x.f32("selector_predecessor.weight", {16, 128}, seed);
            x.f32("selector_successor.weight",   {16, 128}, seed);
            x.f32("selector_hidden.weight",      {256, 16}, seed);
            const int n_layer = llama_model_n_layer(model);
            for (int il = 0; il < n_layer; il++) {
                const std::string blk = "blk." + std::to_string(il) + ".";
                x.f32((blk + "attn_conv_base").c_str(),        {256, 3, 2}, seed);
                x.f32((blk + "attn_conv_proj.weight").c_str(), {256, 24},   seed);
                x.f32((blk + "ffn_conv_base").c_str(),         {256, 3, 2}, seed);
                x.f32((blk + "ffn_conv_proj.weight").c_str(),  {256, 24},   seed);
            }
        });
    });
    // A Gemma-style DFlash backbone, new at v0.5.0 (models/dflash.cpp:7, 108,
    // 571 @d81235049): an embedding and an attention scale, a GELU FFN, K
    // and V sharing one projection (no attn_v, V RMS-normed bare), and no
    // output head -- the token embedding stands in for it. The recipe
    // already writes the post norms, the per-layer output scale and
    // rope_freqs, so every DFlash fixture has those. Since v0.6.0 it also
    // carries an attention value scale, which scales the attention output;
    // the other DFlash fixtures have none, so both arms run.
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
                const bool attn_v = tname.find(".attn_v.") != std::string::npos;
                const bool output = tname.rfind("output.", 0) == 0;
                if (attn_v || output) {
                    x.replaces.insert(tname);
                }
            }
        });
    });
    // Qwen3.5 with its full-attention layers' Q (and Q gate), K and V fused
    // into one attn_qkv, which create_tensor_qkv takes when the file has it.
    // v0.5.0 splits that tensor across devices by its own segments
    // (llama-model.cpp:610, 700 @d81235049): the recipe writes the three
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
    run([&] { return dspark("dflash-dspark-bonus", true); });
    run([&] { return dspark("dflash-dspark", false); });
    run([&] { return dspark("dflash-dspark-noconf", false, false); });

    // clef, new at v0.6.0 (models/clef.cpp:7, 25 @d81235049): a Qwen3.5
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

    // A ModernBERT decision model, new at v0.6.0 (models/modern-bert.cpp:6,
    // 55, 242 @d81235049): the recipe's second block becomes the head's --
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

    llama_backend_free();
    return failed == 0 ? 0 : 1;
}
