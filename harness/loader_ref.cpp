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

#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

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

}
