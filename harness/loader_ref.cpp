// The reference side of `scripts/loader-diff`.
//
// **Not a port.** `llama_model_load` is `static` in `llama.cpp`, and the
// public entry point around it also selects devices and loads tensors. This
// runs its metadata half -- the same calls, in the same order, with the same
// exception wrapping -- against the reference `libllama.a`, and stops where
// `load_tensors` would begin. Files with no tensors load as far as the C++
// gets with them, which is exactly as far as the port has to.
//
// It also hands the Zig harness the model's `llama_hparams` as bytes, with
// the layout `offsetof` gives for every member, so the port's mirror can be
// checked against the real struct before it is used to read it.

#include "llama-model-loader.h"
#include "llama-model.h"
#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void put(const std::string & s, char * buf, size_t n) {
    if (n == 0) {
        return;
    }
    size_t k = s.size() < n - 1 ? s.size() : n - 1;
    memcpy(buf, s.data(), k);
    buf[k] = 0;
}

extern "C" {

// The metadata half of llama_model_load (llama.cpp:315). Returns the model,
// or null with what llama_model_load would log after "error loading model: ".
llama_model * ref_load_metadata(const char * path, const llama_model_kv_override * overrides, int vocab_only, char * what, size_t n) {
    try {
        std::vector<std::string> splits;
        llama_model_params params = llama_model_default_params();
        params.vocab_only   = vocab_only != 0;
        params.kv_overrides = overrides;

        llama_model_loader ml(nullptr, nullptr, nullptr, path, splits, nullptr, LLAMA_LOAD_MODE_MMAP,
            false, params.no_alloc, false, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        llama_model * model = model_ptr.get();
        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch (const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch (const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", "llama_model_load");
        }
        return model_ptr.release();
    } catch (const std::exception & e) {
        put(e.what(), what, n);
        return nullptr;
    }
}

void ref_free(llama_model * model) {
    delete model;
}

const void * ref_hparams(const llama_model * model) {
    return &model->hparams;
}

int ref_type(const llama_model * model) {
    return model->type;
}

int ref_arch(const llama_model * model) {
    return model->arch;
}

const char * ref_name(const llama_model * model) {
    return model->name.c_str();
}

void ref_desc(const llama_model * model, char * buf, size_t n) {
    put(model->desc(), buf, n);
}

uint64_t ref_n_elements(const llama_model * model) {
    return model->n_elements();
}

size_t ref_n_bytes(const llama_model * model) {
    return model->size();
}

int ref_ftype(const llama_model * model) {
    return model->ftype();
}

size_t ref_n_labels(const llama_model * model) {
    return model->classifier_labels.size();
}

const char * ref_label(const llama_model * model, size_t i) {
    return model->classifier_labels[i].c_str();
}

// gguf_kv, in iteration order: the i-th key and value.
size_t ref_n_kv(const llama_model * model) {
    return model->gguf_kv.size();
}

int ref_kv(const llama_model * model, size_t i, const char ** key, const char ** value) {
    size_t k = 0;
    for (const auto & [kk, vv] : model->gguf_kv) {
        if (k++ == i) {
            *key   = kk.c_str();
            *value = vv.c_str();
            return 1;
        }
    }
    return 0;
}

// The layout of llama_hparams: one entry per member, from the generated list.
struct ref_field { const char * name; size_t offset; size_t size; };

#define F(name) { #name, offsetof(llama_hparams, name), sizeof(((llama_hparams *) nullptr)->name) },
static const ref_field ref_fields[] = {
#include "hparams_fields.inc"
};
#undef F

size_t ref_hparams_sizeof(void) {
    return sizeof(llama_hparams);
}

size_t ref_hparams_alignof(void) {
    return alignof(llama_hparams);
}

size_t ref_hparams_nfields(void) {
    return sizeof(ref_fields) / sizeof(ref_fields[0]);
}

const char * ref_hparams_field(size_t i, size_t * offset, size_t * size) {
    *offset = ref_fields[i].offset;
    *size   = ref_fields[i].size;
    return ref_fields[i].name;
}

// The layer members the port's `Layer` mirrors, by name. A name here that
// `llama_layer` lacks fails to compile, so the list cannot name a member
// that does not exist.
struct ref_layer_field { const char * name; size_t offset; };
#define F(m) { #m, offsetof(llama_layer, m) },
static const ref_layer_field ref_layer_fields[] = {
    F(attn_norm)
    F(attn_norm_b)
    F(attn_q_norm)
    F(attn_k_norm)
    F(attn_post_norm)
    F(ssm_norm)
    F(wq)
    F(wk)
    F(wv)
    F(wo)
    F(wqkv)
    F(wqkv_gate)
    F(wq_b)
    F(wk_b)
    F(wv_b)
    F(wo_b)
    F(wqkv_b)
    F(ffn_gate)
    F(ffn_down)
    F(ffn_up)
    F(ffn_gate_inp)
    F(ffn_gate_exps)
    F(ffn_down_exps)
    F(ffn_up_exps)
    F(ffn_gate_shexp)
    F(ffn_down_shexp)
    F(ffn_up_shexp)
    F(ssm_in)
    F(ssm_dt)
    F(ssm_out)
    F(ssm_conv1d)
    F(ssm_a)
    F(ssm_alpha)
    F(ssm_beta)
    F(rope_long)
    F(rope_short)
    F(rope_freqs)
    F(wq_s)
    F(wk_s)
    F(wv_s)
    F(wo_s)
    F(wqkv_s)
    F(wqkv_gate_s)
    F(ffn_gate_s)
    F(ffn_up_s)
    F(ffn_down_s)
    F(ffn_gate_shexp_s)
    F(ffn_up_shexp_s)
    F(ffn_down_shexp_s)
    F(ffn_gate_exps_s)
    F(ffn_down_exps_s)
    F(ffn_up_exps_s)
    F(ssm_in_s)
    F(ssm_out_s)
    F(ssm_alpha_s)
    F(ssm_beta_s)
    F(wq_in_s)
    F(wk_in_s)
    F(wv_in_s)
    F(wo_in_s)
    F(wqkv_in_s)
    F(wqkv_gate_in_s)
    F(ffn_gate_in_s)
    F(ffn_up_in_s)
    F(ffn_down_in_s)
    F(ffn_gate_exps_in_s)
    F(ffn_down_exps_in_s)
    F(ffn_up_exps_in_s)
    F(ffn_gate_shexp_in_s)
    F(ffn_up_shexp_in_s)
    F(ffn_down_shexp_in_s)
    F(ssm_in_in_s)
    F(ssm_out_in_s)
    F(ssm_alpha_in_s)
    F(ssm_beta_in_s)
    F(nextn.eh_proj)
    F(nextn.eh_proj_s)
    F(nextn.eh_proj_in_s)
    F(nextn.embed_tokens)
    F(nextn.enorm)
    F(nextn.hnorm)
    F(nextn.shared_head_head)
    F(nextn.shared_head_head_s)
    F(nextn.shared_head_head_in_s)
    F(nextn.shared_head_norm)
};
#undef F

// The model's tensor named by a `Layer` member, or null; "?" when the
// member is not in the list above.
const char * ref_layer_tensor(const llama_model * model, int il, const char * field) {
    for (const auto & f : ref_layer_fields) {
        if (strcmp(f.name, field) == 0) {
            const ggml_tensor * t = *(ggml_tensor * const *) ((const char *) &model->layers[il] + f.offset);
            return t ? t->name : nullptr;
        }
    }
    return "?";
}

const char * ref_model_tensor(const llama_model * model, const char * field) {
    const ggml_tensor * t = nullptr;
    if      (strcmp(field, "tok_embd")    == 0) t = model->tok_embd;
    else if (strcmp(field, "output_norm") == 0) t = model->output_norm;
    else if (strcmp(field, "output")      == 0) t = model->output;
    else if (strcmp(field, "output_s")    == 0) t = model->output_s;
    else if (strcmp(field, "output_in_s") == 0) t = model->output_in_s;
    else return "?";
    return t ? t->name : nullptr;
}

size_t ref_n_layers(const llama_model * model) {
    return model->layers.size();
}

size_t ref_n_tensors(const llama_model * model) {
    return model->tensors_by_name.size();
}

ggml_tensor * ref_tensor_at(const llama_model * model, size_t i, const char ** name) {
    *name = model->tensors_by_name[i].first.c_str();
    return model->tensors_by_name[i].second;
}

size_t ref_n_devices(const llama_model * model) {
    return model->devices.size();
}

ggml_backend_dev_t ref_device(const llama_model * model, size_t i) {
    return model->devices[i].dev;
}

ggml_backend_dev_t ref_dev_layer(const llama_model * model, int il) {
    return model->dev_layer(il);
}

ggml_backend_dev_t ref_dev_output(const llama_model * model) {
    return model->dev_output();
}

// memory_breakdown, flattened: the i-th entry's buffer type and size.
size_t ref_memory_breakdown(const llama_model * model, size_t i, ggml_backend_buffer_type_t * buft) {
    size_t k = 0;
    for (const auto & [b, n] : model->memory_breakdown()) {
        if (k++ == i) {
            *buft = b;
            return n;
        }
    }
    *buft = nullptr;
    return 0;
}

// FNV-1a over a tensor's data, read through its backend; a buffer that
// cannot be read through the backend (the CPU repack buffer has no
// get_tensor) is read in place. Used for both sides' tensors.
uint64_t ref_hash_tensor(const ggml_tensor * t) {
    const size_t n = ggml_nbytes(t);
    std::vector<uint8_t> buf(n);
    if (t->buffer && t->buffer->iface.get_tensor) {
        ggml_backend_tensor_get(t, buf.data(), 0, n);
    } else {
        memcpy(buf.data(), t->data, n);
    }
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t b : buf) {
        h ^= b;
        h *= 0x100000001b3ull;
    }
    return h;
}

}
