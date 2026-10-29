// Small random mmproj fixtures for every mtmd projector type, the text
// models they load against, and a validator that runs each through
// upstream's own libmtmd.
//
// **Not a port.** This is test harness: it plays for Stage 8's gates the
// role `tests/test-llama-archs.cpp` plays for `make arch-model-diff`. No
// mmproj file exists on this machine and upstream ships no generator, so
// each projector type's keys and tensor shapes here are read off
// `llama.cpp/tools/mtmd/clip.cpp` (`load_hparams`, `load_tensors`,
// `clip_n_mmproj_embd`) and the graph builders under `models/`, at
// v0.4.1 (`b29c606e2`). Weights are small and random (fixed seed per
// type), so a fixture checks plumbing and numerics, not output quality.
//
// Every projector targets n_embd 256, the arch fixtures' width. The text
// models are those arch fixtures (`zig-out/arch-fixtures`) with their
// `no_vocab` tokenizer rewritten as SPM: printable ASCII, a newline byte
// token and the special tokens `mtmd.cpp` and `mtmd-helper-gen.cpp` look
// up, so prompts tokenize and every `lookup_token` resolves.
//
// Usage:
//   mmproj-fixtures gen   <arch-fixtures-dir> <out-dir>   writes fixtures + manifest.txt
//   mmproj-fixtures check <out-dir>/manifest.txt [cpu|gpu|both] [type-filter]
//
// Manifest: one fixture per line, tab-separated:
//   projector  mmproj-path  text-model-path  modality  inputs
// where modality is vision|audio|vision+audio|tts-qwen3|tts-pocket, or
// expect-fail for a type upstream declares but cannot load, and inputs lists
// what to feed, `;`-separated: `img:WxH,...` for images, `aud:N,...` audio
// sample counts, `tts:N` generation steps. `check` runs each line in a
// child process (`one`), so an upstream abort is reported, not fatal.

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

// the KEY_* / TN_* names the loader reads, from the header that defines them
#include "clip-impl.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sys/wait.h>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

static std::string fmt(const char * f, ...) {
    va_list ap;
    va_start(ap, f);
    char buf[512];
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// -----------------------------------------------------------------------------
// GGUF writer

struct G {
    std::string name;
    gguf_context * g = nullptr;
    ggml_context * ctx = nullptr;
    std::mt19937 rng;
    std::set<std::string> seen;

    G(const std::string & n, uint32_t seed) : name(n), rng(seed) {
        g = gguf_init_empty();
        ggml_init_params p = { (size_t) 1536 * 1024 * 1024, nullptr, false };
        ctx = ggml_init(p);
        gguf_set_val_str(g, "general.architecture", "clip");
        gguf_set_val_str(g, KEY_NAME, ("mtmd fixture " + n).c_str());
        gguf_set_val_str(g, KEY_DESCRIPTION, "random weights, for tests");
    }
    ~G() {
        gguf_free(g);
        ggml_free(ctx);
    }

    void u32(const std::string & k, uint32_t v) { gguf_set_val_u32(g, k.c_str(), v); }
    void i32(const std::string & k, int32_t v) { gguf_set_val_i32(g, k.c_str(), v); }
    void f32(const std::string & k, float v) { gguf_set_val_f32(g, k.c_str(), v); }
    void b(const std::string & k, bool v) { gguf_set_val_bool(g, k.c_str(), v); }
    void str(const std::string & k, const std::string & v) { gguf_set_val_str(g, k.c_str(), v.c_str()); }
    void ai32(const std::string & k, const std::vector<int32_t> & v) {
        gguf_set_arr_data(g, k.c_str(), GGUF_TYPE_INT32, v.data(), v.size());
    }
    void af32(const std::string & k, const std::vector<float> & v) {
        gguf_set_arr_data(g, k.c_str(), GGUF_TYPE_FLOAT32, v.data(), v.size());
    }

    // a tensor of `ne` (ggml order), uniform in [base - scale, base + scale]
    ggml_tensor * t(const std::string & tname, std::vector<int64_t> ne, float scale = 0.05f, float base = 0.0f,
                    ggml_type type = GGML_TYPE_F32) {
        if (seen.count(tname)) {
            fprintf(stderr, "%s: duplicate tensor %s\n", name.c_str(), tname.c_str());
            exit(1);
        }
        seen.insert(tname);
        while (ne.size() < 4) ne.push_back(1);
        ggml_tensor * x = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(x, tname.c_str());
        std::uniform_real_distribution<float> d(base - scale, base + scale);
        const int64_t n = ggml_nelements(x);
        if (type == GGML_TYPE_F32) {
            float * p = (float *) x->data;
            for (int64_t i = 0; i < n; i++) p[i] = d(rng);
        } else if (type == GGML_TYPE_F16) {
            ggml_fp16_t * p = (ggml_fp16_t *) x->data;
            for (int64_t i = 0; i < n; i++) p[i] = ggml_fp32_to_fp16(d(rng));
        } else if (type == GGML_TYPE_I32) {
            int32_t * p = (int32_t *) x->data;
            for (int64_t i = 0; i < n; i++) p[i] = (int32_t) i;
        }
        gguf_add_tensor(g, x);
        return x;
    }
    // a matmul weight: rows of `in`, `out` of them, scaled for unit-ish activations
    ggml_tensor * w(const std::string & tname, int64_t in, int64_t out) {
        return t(tname, {in, out}, 1.0f / std::sqrt((float) in));
    }
    ggml_tensor * bias(const std::string & tname, int64_t n) { return t(tname, {n}, 0.02f); }
    ggml_tensor * norm(const std::string & tname, int64_t n) { return t(tname, {n}, 0.1f, 1.0f); }

    bool write(const std::string & path) {
        return gguf_write_to_file(g, path.c_str(), false);
    }
};

// -----------------------------------------------------------------------------
// shared building blocks

struct VitOpts {
    bool qkv = false;       // fused qkv
    bool attn_bias = true;  // q/k/v/o biases
    bool k_bias = true;     // whisper drops k's bias
    bool ln_bias = true;
    bool gated = false;     // ffn gate
    bool ffn_bias = true;
    int qk_norm = 0;        // 0 none, 1 full width, 2 per head
    bool ls = false;        // layer scales
    int n_head_kv = 0;      // GQA
    int d_head = 0;
    int n_expert = 0;       // MoE ffn on layers >= moe_from (dots3note)
    int moe_from = 0;
};

// the standard layer tensors `load_tensors` reads for every layer
static void vit_layers(G & g, const char * prefix, int n_layer, int E, int F, int n_head, const VitOpts & o = {}) {
    const int dh = o.d_head ? o.d_head : E / n_head;
    const int hkv = o.n_head_kv ? o.n_head_kv : n_head;
    for (int il = 0; il < n_layer; il++) {
        if (o.qkv) {
            g.w(fmt(TN_ATTN_QKV, prefix, il, "weight"), E, dh * (n_head + 2 * hkv));
            if (o.attn_bias) g.bias(fmt(TN_ATTN_QKV, prefix, il, "bias"), dh * (n_head + 2 * hkv));
        } else {
            g.w(fmt(TN_ATTN_Q, prefix, il, "weight"), E, dh * n_head);
            g.w(fmt(TN_ATTN_K, prefix, il, "weight"), E, dh * hkv);
            g.w(fmt(TN_ATTN_V, prefix, il, "weight"), E, dh * hkv);
            if (o.attn_bias) {
                g.bias(fmt(TN_ATTN_Q, prefix, il, "bias"), dh * n_head);
                if (o.k_bias) g.bias(fmt(TN_ATTN_K, prefix, il, "bias"), dh * hkv);
                g.bias(fmt(TN_ATTN_V, prefix, il, "bias"), dh * hkv);
            }
        }
        g.w(fmt(TN_ATTN_OUTPUT, prefix, il, "weight"), dh * n_head, E);
        if (o.attn_bias) g.bias(fmt(TN_ATTN_OUTPUT, prefix, il, "bias"), E);
        if (o.qk_norm == 1) {
            g.norm(fmt(TN_ATTN_Q_NORM, prefix, il, "weight"), dh * n_head);
            g.norm(fmt(TN_ATTN_K_NORM, prefix, il, "weight"), dh * hkv);
        } else if (o.qk_norm == 2) {
            g.norm(fmt(TN_ATTN_Q_NORM, prefix, il, "weight"), dh);
            g.norm(fmt(TN_ATTN_K_NORM, prefix, il, "weight"), dh);
        }
        if (o.ls) {
            g.norm(fmt(TN_LS_1, prefix, il, "weight"), E);
            g.norm(fmt(TN_LS_2, prefix, il, "weight"), E);
        }
        g.norm(fmt(TN_LN_1, prefix, il, "weight"), E);
        g.norm(fmt(TN_LN_2, prefix, il, "weight"), E);
        if (o.ln_bias) {
            g.bias(fmt(TN_LN_1, prefix, il, "bias"), E);
            g.bias(fmt(TN_LN_2, prefix, il, "bias"), E);
        }
        if (o.n_expert > 0 && il >= o.moe_from) {
            const int X = o.n_expert;
            g.w(fmt(TN_FFN_GATE_INP, prefix, il, "weight"), E, X);
            g.t(fmt(TN_FFN_GATE_EXPS, prefix, il, "weight"), {E, F, X}, 1.0f / std::sqrt((float) E));
            g.t(fmt(TN_FFN_UP_EXPS, prefix, il, "weight"), {E, F, X}, 1.0f / std::sqrt((float) E));
            g.t(fmt(TN_FFN_DOWN_EXPS, prefix, il, "weight"), {F, E, X}, 1.0f / std::sqrt((float) F));
            g.t(fmt(TN_FFN_EXP_PROBS_B, prefix, il, "weight"), {X}, 0.1f);
            continue;
        }
        g.w(fmt(TN_FFN_UP, prefix, il, "weight"), E, F);
        g.w(fmt(TN_FFN_DOWN, prefix, il, "weight"), F, E);
        if (o.gated) g.w(fmt(TN_FFN_GATE, prefix, il, "weight"), E, F);
        if (o.ffn_bias) {
            g.bias(fmt(TN_FFN_UP, prefix, il, "bias"), F);
            g.bias(fmt(TN_FFN_DOWN, prefix, il, "bias"), E);
            if (o.gated) g.bias(fmt(TN_FFN_GATE, prefix, il, "bias"), F);
        }
    }
}

// the modality keys and the common hparams `load_hparams` reads
static void hp(G & g, const char * prefix, int E, int F, int n_head, int n_layer, int proj_dim, float eps = 1e-6f) {
    g.u32(fmt(KEY_N_EMBD, prefix), E);
    g.u32(fmt(KEY_N_FF, prefix), F);
    g.u32(fmt(KEY_N_HEAD, prefix), n_head);
    g.u32(fmt(KEY_N_BLOCK, prefix), n_layer);
    g.u32(fmt(KEY_PROJ_DIM, prefix), proj_dim);
    g.f32(fmt(KEY_LAYER_NORM_EPS, prefix), eps);
}

static void vision(G & g, const std::string & proj, int image_size, int patch, bool gelu = true) {
    g.b(KEY_HAS_VISION_ENC, true);
    g.str(KEY_PROJ_TYPE, proj);
    g.u32(KEY_IMAGE_SIZE, image_size);
    g.u32(KEY_PATCH_SIZE, patch);
    g.af32(KEY_IMAGE_MEAN, {0.5f, 0.45f, 0.4f});
    g.af32(KEY_IMAGE_STD, {0.25f, 0.26f, 0.27f});
    if (gelu) g.b(KEY_USE_GELU, true);
}

static void audio(G & g, const std::string & proj, int n_mel) {
    g.b(KEY_HAS_AUDIO_ENC, true);
    g.str(KEY_PROJ_TYPE, proj);
    g.u32(KEY_A_NUM_MEL_BINS, n_mel);
}

// patch embedding conv [P, P, 3, E], its bias, and learned positions
static void patch_embd(G & g, int P, int E, int n_pos, bool bias = true) {
    g.t(TN_PATCH_EMBD, {P, P, 3, E}, 1.0f / std::sqrt((float) (P * P * 3)));
    if (bias) g.bias(TN_PATCH_BIAS, E);
    if (n_pos > 0) g.t(fmt(TN_POS_EMBD, "v"), {E, n_pos}, 0.02f);
}

static constexpr int N_TEXT = 256; // every text fixture's n_embd

// -----------------------------------------------------------------------------
// fixtures

struct Fixture {
    std::string type;     // projector name, as the manifest names it
    std::string text;     // text model key (see `text_models`)
    std::string modality; // vision | audio | tts-qwen3 | tts-pocket
    std::string inputs;
    std::function<void(G &)> build;
};

static std::vector<Fixture> fixtures() {
    std::vector<Fixture> f;
    const char * IMG = "img:64x64,96x48,40x80";
    const char * AUD = "aud:16000,40000";

    // --- siglip family (models/siglip.cpp)
    f.push_back({"gemma3", "qwen3", "vision", "img:64x64,100x60", [](G & g) {
        const int E = 64, P = 8, S = 64;
        vision(g, "gemma3", S, P);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        patch_embd(g, P, E, (S / P) * (S / P));
        vit_layers(g, "v", 2, E, 128, 4);
        g.norm(fmt(TN_LN_POST, "v", "weight"), E);
        g.bias(fmt(TN_LN_POST, "v", "bias"), E);
        g.norm(TN_MM_SOFT_EMB_N, E);
        g.t(TN_MM_INP_PROJ, {N_TEXT, E}, 0.1f);
    }});
    f.push_back({"idefics3", "qwen3", "vision", "img:64x64,130x70", [](G & g) {
        const int E = 64, P = 8, S = 32;
        vision(g, "idefics3", S, P);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        g.u32(KEY_PREPROC_IMAGE_SIZE, 64);
        patch_embd(g, P, E, (S / P) * (S / P));
        vit_layers(g, "v", 2, E, 128, 4);
        g.w(fmt(TN_MM_PROJECTOR, "weight"), E * 4, N_TEXT);
    }});
    f.push_back({"lfm2", "qwen3", "vision", "img:64x64,200x120", [](G & g) {
        const int E = 64, P = 8, S = 64;
        vision(g, "lfm2", S, P);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        patch_embd(g, P, E, (S / P) * (S / P));
        vit_layers(g, "v", 2, E, 128, 4);
        g.norm(TN_MM_INP_NORM, E * 4);
        g.bias(TN_MM_INP_NORM_B, E * 4);
        g.w(fmt(TN_LLAVA_PROJ, 1, "weight"), E * 4, 128);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), 128);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), 128, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});
    f.push_back({"janus_pro", "qwen3", "vision", "img:64x64", [](G & g) {
        const int E = 64, P = 8, S = 32;
        vision(g, "janus_pro", S, P);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        patch_embd(g, P, E, (S / P) * (S / P));
        vit_layers(g, "v", 2, E, 128, 4);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), E, 128);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), 128);
        g.w(fmt(TN_LLAVA_PROJ, 1, "weight"), 128, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), N_TEXT);
    }});

    // --- llava family (models/llava.cpp): q/k/v layers, class token,
    // positions by get_rows, a projector after the patches
    auto llava_body = [](G & g, const char * proj, int S, int P, int E, bool cls) {
        vision(g, proj, S, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        const int n_patches = (S / P) * (S / P);
        patch_embd(g, P, E, n_patches + (cls ? 1 : 0), false);
        if (cls) g.t(TN_CLASS_EMBD, {E}, 0.05f);
        g.norm(fmt(TN_LN_PRE, "v", "weight"), E);
        g.bias(fmt(TN_LN_PRE, "v", "bias"), E);
        vit_layers(g, "v", 2, E, 64, 2);
    };
    f.push_back({"mlp", "qwen3", "vision", "img:32x32,64x32,96x80", [llava_body](G & g) {
        // llava-1.6: grid pinpoints take the llava-uhd preprocessor
        const int E = 32;
        llava_body(g, "mlp", 32, 8, E, true);
        g.ai32(KEY_IMAGE_GRID_PINPOINTS, {32, 32, 32, 64, 64, 32, 64, 64});
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), E, 64);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), 64);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), 64, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});
    f.push_back({"mlp_norm", "qwen3", "vision", "img:32x32,50x30", [llava_body](G & g) {
        // the "mlp" type with a mm.3 tensor is the Yi-style mlp_norm
        const int E = 32;
        llava_body(g, "mlp", 32, 8, E, true);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), E, 64);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), 64);
        g.norm(fmt(TN_LLAVA_PROJ, 1, "weight"), 64);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), 64);
        g.w(fmt(TN_LLAVA_PROJ, 3, "weight"), 64, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 3, "bias"), N_TEXT);
        g.norm(fmt(TN_LLAVA_PROJ, 4, "weight"), N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 4, "bias"), N_TEXT);
    }});
    // LDP and LDPv2 hardcode a 24x24 patch grid
    f.push_back({"ldp", "qwen3", "vision", "img:96x96,60x90", [llava_body](G & g) {
        const int E = 32, D = N_TEXT;
        llava_body(g, "ldp", 96, 4, E, true);
        g.w(fmt(TN_MVLM_PROJ_MLP, 1, "weight"), E, D);
        g.bias(fmt(TN_MVLM_PROJ_MLP, 1, "bias"), D);
        g.w(fmt(TN_MVLM_PROJ_MLP, 3, "weight"), D, D);
        g.bias(fmt(TN_MVLM_PROJ_MLP, 3, "bias"), D);
        for (int blk = 1; blk <= 2; blk++) {
            // depthwise kernels are F16: ggml_conv_2d_dw's im2col is F16,
            // and the CPU multiplies it only by an F16 kernel
            g.t(fmt(TN_MVLM_PROJ_BLOCK, blk, 0, "0.weight"), {3, 3, 1, D}, 0.2f, 0.0f, GGML_TYPE_F16);
            g.norm(fmt(TN_MVLM_PROJ_BLOCK, blk, 0, "1.weight"), D);
            g.bias(fmt(TN_MVLM_PROJ_BLOCK, blk, 0, "1.bias"), D);
            g.w(fmt(TN_MVLM_PROJ_BLOCK, blk, 1, "fc1.weight"), D, D / 4);
            g.bias(fmt(TN_MVLM_PROJ_BLOCK, blk, 1, "fc1.bias"), D / 4);
            g.w(fmt(TN_MVLM_PROJ_BLOCK, blk, 1, "fc2.weight"), D / 4, D);
            g.bias(fmt(TN_MVLM_PROJ_BLOCK, blk, 1, "fc2.bias"), D);
            g.w(fmt(TN_MVLM_PROJ_BLOCK, blk, 2, "0.weight"), D, D);
            g.norm(fmt(TN_MVLM_PROJ_BLOCK, blk, 2, "1.weight"), D);
            g.bias(fmt(TN_MVLM_PROJ_BLOCK, blk, 2, "1.bias"), D);
        }
    }});
    f.push_back({"ldpv2", "qwen3", "vision", "img:96x96", [llava_body](G & g) {
        const int E = 32, D = N_TEXT;
        llava_body(g, "ldpv2", 96, 4, E, true);
        g.w(fmt(TN_MVLM_PROJ_MLP, 0, "weight"), E, D);
        g.bias(fmt(TN_MVLM_PROJ_MLP, 0, "bias"), D);
        g.w(fmt(TN_MVLM_PROJ_MLP, 2, "weight"), D, D);
        g.bias(fmt(TN_MVLM_PROJ_MLP, 2, "bias"), D);
        g.t(fmt(TN_MVLM_PROJ_PEG, 0, "weight"), {3, 3, 1, D}, 0.2f, 0.0f, GGML_TYPE_F16);
        g.bias(fmt(TN_MVLM_PROJ_PEG, 0, "bias"), D);
    }});
    f.push_back({"adapter", "qwen3", "vision", "img:32x32", [llava_body](G & g) {
        // glm_edge: no class token (the adapter needs a square patch grid)
        const int E = 32, Ea = 48, Hg = 64, F2 = 96;
        llava_body(g, "adapter", 32, 4, E, false);
        g.t(fmt(TN_GLM_ADAPER_CONV, "weight"), {2, 2, E, Ea}, 0.1f);
        g.bias(fmt(TN_GLM_ADAPER_CONV, "bias"), Ea);
        g.w(fmt(TN_GLM_ADAPTER_LINEAR, "weight"), Ea, Hg);
        g.norm(fmt(TN_GLM_ADAPTER_NORM_1, "weight"), Hg);
        g.bias(fmt(TN_GLM_ADAPTER_NORM_1, "bias"), Hg);
        g.w(fmt(TN_GLM_ADAPTER_D_H_2_4H, "weight"), Hg, F2);
        g.w(fmt(TN_GLM_ADAPTER_GATE, "weight"), Hg, F2);
        g.w(fmt(TN_GLM_ADAPTER_D_4H_2_H, "weight"), F2, N_TEXT);
        g.t(TN_TOK_GLM_BOI, {N_TEXT, 1}, 0.1f);
        g.t(TN_TOK_GLM_EOI, {N_TEXT, 1}, 0.1f);
    }});

    // --- 2D-rope ViTs with a patch merger
    auto pixtral_like = [](G & g, const char * proj, bool img_break) {
        const int E = 32, P = 8, H = 64;
        vision(g, proj, 64, P, !img_break);
        if (img_break) g.b(KEY_USE_SILU, true);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_SPATIAL_MERGE_SIZE, 2);
        patch_embd(g, P, E, 0, false);
        VitOpts o;
        o.attn_bias = false;
        o.ln_bias = false;
        o.ffn_bias = false;
        o.gated = true;
        vit_layers(g, "v", 2, E, 64, 2, o);
        g.norm(TN_MM_INP_NORM, E);
        g.w(fmt(TN_MM_PATCH_MERGER, "weight"), E * 4, E);
        g.w(fmt(TN_LLAVA_PROJ, 1, "weight"), E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
        if (img_break) g.t(TN_TOK_IMG_BREAK, {N_TEXT}, 0.1f);
    };
    f.push_back({"pixtral", "qwen3", "vision", "img:64x64,96x48", [pixtral_like](G & g) {
        pixtral_like(g, "pixtral", true);
    }});
    f.push_back({"lightonocr", "qwen3", "vision", "img:64x64,120x40", [pixtral_like](G & g) {
        g.u32(KEY_PREPROC_IMAGE_SIZE, 96);
        pixtral_like(g, "lightonocr", false);
    }});
    f.push_back({"internvl", "qwen3", "vision", "img:32x32,70x30", [](G & g) {
        const int E = 32, P = 8, S = 32, H = 64;
        vision(g, "internvl", S, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        g.u32(KEY_PREPROC_MAX_TILES, 4);
        patch_embd(g, P, E, (S / P) * (S / P) + 1);
        g.t(TN_CLASS_EMBD, {E}, 0.05f);
        VitOpts o;
        o.ls = true;
        vit_layers(g, "v", 2, E, 64, 2, o);
        g.norm(fmt(TN_MVLM_PROJ_MLP, 0, "weight"), 4 * E);
        g.bias(fmt(TN_MVLM_PROJ_MLP, 0, "bias"), 4 * E);
        g.w(fmt(TN_MVLM_PROJ_MLP, 1, "weight"), 4 * E, H);
        g.bias(fmt(TN_MVLM_PROJ_MLP, 1, "bias"), H);
        g.w(fmt(TN_MVLM_PROJ_MLP, 3, "weight"), H, N_TEXT);
        g.bias(fmt(TN_MVLM_PROJ_MLP, 3, "bias"), N_TEXT);
    }});
    f.push_back({"nemotron_v2_vl", "qwen3", "vision", "img:32x32,64x40", [](G & g) {
        const int E = 32, P = 8, S = 32, H = 64;
        vision(g, "nemotron_v2_vl", S, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        patch_embd(g, P, E, (S / P) * (S / P));
        g.t(TN_CLASS_EMBD, {E, 3}, 0.05f); // three registers
        vit_layers(g, "v", 2, E, 64, 2);
        g.norm(fmt(TN_MVLM_PROJ_MLP, 0, "weight"), 4 * E);
        g.w(fmt(TN_MVLM_PROJ_MLP, 1, "weight"), 4 * E, H);
        g.w(fmt(TN_MVLM_PROJ_MLP, 3, "weight"), H, N_TEXT);
    }});
    auto merge_proj = [](G & g, int E, int H, bool inp_norm) {
        if (inp_norm) {
            g.norm(TN_MM_INP_NORM, E);
            g.bias(TN_MM_INP_NORM_B, E);
        }
        g.w(fmt(TN_LLAVA_PROJ, 1, "weight"), 4 * E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    };
    f.push_back({"kimivl", "qwen3", "vision", "img:64x64,96x48", [merge_proj](G & g) {
        const int E = 32, P = 8;
        vision(g, "kimivl", 64, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        patch_embd(g, P, E, 8 * 8);
        vit_layers(g, "v", 2, E, 64, 2);
        merge_proj(g, E, 64, true);
    }});
    f.push_back({"paddleocr", "qwen3", "vision", "img:64x64,96x48", [merge_proj](G & g) {
        const int E = 64, P = 8;
        vision(g, "paddleocr", 64, P);
        hp(g, "vision", E, 64, 4, 2, N_TEXT);
        g.u32(KEY_IMAGE_MIN_PIXELS, 32 * 32);
        g.u32(KEY_IMAGE_MAX_PIXELS, 128 * 128);
        patch_embd(g, P, E, 8 * 8);
        vit_layers(g, "v", 2, E, 64, 4);
        merge_proj(g, E, 64, true);
    }});
    f.push_back({"phi4", "qwen3", "vision", "img:64x64,96x48", [](G & g) {
        const int E = 32, P = 8;
        vision(g, "phi4", 64, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_IMAGE_MIN_PIXELS, 32 * 32);
        g.u32(KEY_IMAGE_MAX_PIXELS, 128 * 128);
        patch_embd(g, P, E, 8 * 8);
        vit_layers(g, "v", 2, E, 64, 2);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), E, 64);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), 64);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), 64, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});

    // --- Qwen-VL family: two patch convs (temporal pairs), a 2x2 merger
    auto qwen2vl_like = [](G & g, const char * proj, bool rms, int wa_pattern) {
        const int E = 64, P = 8, H = 128;
        vision(g, proj, 64, P, false);
        g.b(KEY_USE_SILU, true);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_SPATIAL_MERGE_SIZE, 2);
        if (wa_pattern) g.u32(KEY_WIN_ATTN_PATTERN, wa_pattern);
        g.t(TN_PATCH_EMBD, {P, P, 3, E}, 0.05f);
        g.t(TN_PATCH_EMBD_1, {P, P, 3, E}, 0.05f);
        VitOpts o;
        o.ln_bias = !rms;
        o.gated = rms;
        vit_layers(g, "v", 2, E, 128, 4, o);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), 4 * E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    };
    f.push_back({"qwen2vl_merger", "qwen2vl", "vision", "img:64x64,96x48", [qwen2vl_like](G & g) {
        qwen2vl_like(g, "qwen2vl_merger", false, 0);
    }});
    f.push_back({"qwen2.5vl_merger", "qwen2vl", "vision", "img:64x64,128x80", [qwen2vl_like](G & g) {
        qwen2vl_like(g, "qwen2.5vl_merger", true, 2);
    }});
    f.push_back({"qwen3vl_merger", "qwen3vl-ds1", "vision", "img:64x64,96x48", [](G & g) {
        // one deepstack layer: the text model reads 2 x 256 per token
        const int E = 64, P = 8, H = 128;
        vision(g, "qwen3vl_merger", 64, P);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_SPATIAL_MERGE_SIZE, 2);
        patch_embd(g, P, E, 8 * 8);
        g.t(TN_PATCH_EMBD_1, {P, P, 3, E}, 0.05f);
        VitOpts o;
        o.qkv = true;
        vit_layers(g, "v", 2, E, 128, 4, o);
        g.norm(fmt(TN_DEEPSTACK_NORM, 0, "weight"), 4 * E);
        g.bias(fmt(TN_DEEPSTACK_NORM, 0, "bias"), 4 * E);
        g.w(fmt(TN_DEEPSTACK_FC1, 0, "weight"), 4 * E, H);
        g.bias(fmt(TN_DEEPSTACK_FC1, 0, "bias"), H);
        g.w(fmt(TN_DEEPSTACK_FC2, 0, "weight"), H, N_TEXT);
        g.bias(fmt(TN_DEEPSTACK_FC2, 0, "bias"), N_TEXT);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), 4 * E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});

    f.push_back({"exaone4_5", "qwen2vl", "vision", "img:64x64,96x48", [](G & g) {
        // GQA through a fused qkv, two patch convs, a 2x2 merger
        const int E = 64, P = 8, H = 128;
        vision(g, "exaone4_5", 64, P, false);
        g.b(KEY_USE_SILU, true);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(fmt(KEY_N_HEAD_KV, "vision"), 2);
        g.u32(KEY_IMAGE_MIN_PIXELS, 32 * 32);
        g.u32(KEY_IMAGE_MAX_PIXELS, 128 * 128);
        g.u32(KEY_WIN_ATTN_PATTERN, 2);
        g.t(TN_PATCH_EMBD, {P, P, 3, E}, 0.05f);
        g.t(TN_PATCH_EMBD_1, {P, P, 3, E}, 0.05f);
        VitOpts o;
        o.qkv = true;
        o.n_head_kv = 2;
        o.ln_bias = false;
        o.gated = true;
        vit_layers(g, "v", 2, E, 128, 4, o);
        g.norm(fmt(TN_LN_POST, "v", "weight"), E);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), 4 * E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});
    f.push_back({"glm4v", "qwen2vl", "vision", "img:64x64,96x48", [](G & g) {
        const int E = 64, P = 8, Em = 64, Hf = 96, Fm = 128;
        vision(g, "glm4v", 64, P, false);
        g.b(KEY_USE_SILU, true);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_SPATIAL_MERGE_SIZE, 2);
        patch_embd(g, P, E, 8 * 8);
        g.t(TN_PATCH_EMBD_1, {P, P, 3, E}, 0.05f);
        g.norm(fmt(TN_NORM_EMBD, "weight"), E);
        g.bias(fmt(TN_NORM_EMBD, "bias"), E);
        VitOpts o;
        o.ln_bias = false;
        o.gated = true;
        vit_layers(g, "v", 2, E, 128, 4, o);
        g.norm(fmt(TN_LN_POST, "v", "weight"), E);
        g.t(fmt(TN_MM_PATCH_MERGER, "weight"), {2, 2, E, Em}, 0.05f);
        g.bias(fmt(TN_MM_PATCH_MERGER, "bias"), Em);
        g.w(fmt(TN_MM_PROJECTOR, "weight"), Em, Hf);
        g.norm(fmt(TN_MM_POST_NORM, "weight"), Hf);
        g.bias(fmt(TN_MM_POST_NORM, "bias"), Hf);
        g.w(fmt(TN_MM_UP, "weight"), Hf, Fm);
        g.w(fmt(TN_MM_GATE, "weight"), Hf, Fm);
        g.w(fmt(TN_MM_DOWN, "weight"), Fm, N_TEXT);
    }});
    f.push_back({"cogvlm", "qwen3", "vision", "img:32x32,60x40", [](G & g) {
        const int E = 32, P = 8, S = 32, Hc = 64, F = 96;
        vision(g, "cogvlm", S, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        patch_embd(g, P, E, (S / P) * (S / P) + 1);
        g.t(TN_CLASS_EMBD, {E}, 0.05f);
        // the graph reads a fused qkv; the loader takes q/k/v as optional,
        // so without one it loads and then dereferences a null weight
        VitOpts o;
        o.qkv = true;
        vit_layers(g, "v", 2, E, 64, 2, o);
        g.w(fmt(TN_MM_PROJECTOR, "weight"), E, Hc);
        g.norm(fmt(TN_MM_POST_FC_NORM, "weight"), Hc);
        g.bias(fmt(TN_MM_POST_FC_NORM, "bias"), Hc);
        g.w(fmt(TN_MM_H_TO_4H, "weight"), Hc, F);
        g.w(fmt(TN_MM_GATE, "weight"), Hc, F);
        g.w(fmt(TN_MM_4H_TO_H, "weight"), F, N_TEXT);
        g.t(TN_TOK_BOI, {N_TEXT, 1}, 0.1f);
        g.t(TN_TOK_EOI, {N_TEXT, 1}, 0.1f);
    }});

    f.push_back({"llama4", "qwen3", "vision", "img:32x32,64x64,80x48", [](G & g) {
        // the patch embedding is a matmul over im2col: a 2-d [3 P P, E]
        const int E = 32, P = 8, S = 32, H = 64;
        vision(g, "llama4", S, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        g.t(TN_PATCH_EMBD, {3 * P * P, E}, 1.0f / std::sqrt(3.0f * P * P));
        g.t(fmt(TN_POS_EMBD, "v"), {E, (S / P) * (S / P) + 1}, 0.02f);
        g.t(TN_CLASS_EMBD, {E}, 0.05f);
        g.norm(fmt(TN_LN_PRE, "v", "weight"), E);
        g.bias(fmt(TN_LN_PRE, "v", "bias"), E);
        g.norm(fmt(TN_LN_POST, "v", "weight"), E);
        g.bias(fmt(TN_LN_POST, "v", "bias"), E);
        vit_layers(g, "v", 2, E, 64, 2);
        g.w(fmt(TN_MVLM_PROJ_MLP, 1, "weight"), 4 * E, H);
        g.w(fmt(TN_MVLM_PROJ_MLP, 2, "weight"), H, H);
        g.w(fmt(TN_MM_PROJECTOR, "weight"), H, N_TEXT);
    }});
    f.push_back({"kimik25", "qwen3", "vision", "img:64x64,96x48", [](G & g) {
        // positions are a 3-d [E, W, H] grid, resized per image
        const int E = 32, P = 8;
        vision(g, "kimik25", 64, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        g.u32(KEY_IMAGE_MIN_PIXELS, 32 * 32);
        g.u32(KEY_IMAGE_MAX_PIXELS, 128 * 128);
        patch_embd(g, P, E, 0);
        g.t(fmt(TN_POS_EMBD, "v"), {E, 8, 8}, 0.02f);
        vit_layers(g, "v", 2, E, 64, 2);
        g.norm(TN_MM_INP_NORM, E);
        g.bias(TN_MM_INP_NORM_B, E);
        g.w(fmt(TN_LLAVA_PROJ, 1, "weight"), 4 * E, 64);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), 64);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), 64, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});
    f.push_back({"youtuvl", "qwen2vl", "vision", "img:64x64,96x48,40x80", [](G & g) {
        // the patch embedding is linear; layer 1 is full attention, 0 windowed
        const int E = 64, P = 8, H = 128;
        vision(g, "youtuvl", 64, P);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_ATTN_WINDOW_SIZE, 32);
        g.ai32(KEY_WIN_ATTN_LAYER_INDEXES, {1});
        g.t(TN_PATCH_EMBD, {3 * P * P, E}, 1.0f / std::sqrt(3.0f * P * P));
        g.bias(TN_PATCH_BIAS, E);
        vit_layers(g, "v", 2, E, 128, 4);
        g.norm(fmt(TN_LN_POST, "v", "weight"), E);
        g.bias(fmt(TN_LN_POST, "v", "bias"), E);
        g.norm(TN_MM_INP_NORM, E);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), 4 * E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});
    f.push_back({"mimovl", "qwen2vl", "vision", "img:64x64,96x48", [](G & g) {
        // one layer of each window mode: full (-1), row (0), column (1)
        const int E = 64, P = 8, H = 128, NH = 4, NKV = 2;
        vision(g, "mimovl", 64, P, false);
        g.b(KEY_USE_SILU, true);
        hp(g, "vision", E, 128, NH, 3, N_TEXT);
        g.u32(fmt(KEY_N_HEAD_KV, "vision"), NKV);
        g.u32(KEY_ATTN_WINDOW_SIZE, 2);
        g.ai32(KEY_WA_PATTERN_MODE, {-1, 0, 1});
        g.u32(KEY_IMAGE_MIN_PIXELS, 32 * 32);
        g.u32(KEY_IMAGE_MAX_PIXELS, 128 * 128);
        g.t(TN_PATCH_EMBD, {P, P, 3, E}, 0.05f);
        g.t(TN_PATCH_EMBD_1, {P, P, 3, E}, 0.05f);
        VitOpts o;
        o.qkv = true;
        o.n_head_kv = NKV;
        o.ln_bias = false;
        o.gated = true;
        vit_layers(g, "v", 3, E, 128, NH, o);
        for (int il = 0; il < 3; il++) g.t(fmt(TN_ATTN_SINKS, "v", il), {NH}, 0.5f);
        g.norm(fmt(TN_LN_POST, "v", "weight"), E);
        g.bias(fmt(TN_LN_POST, "v", "bias"), E);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), 4 * E, H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
    }});
    f.push_back({"step3vl", "qwen3", "vision", "img:48x48,600x300", [](G & g) {
        // two stride-2 3x3 downsampling convs after the ViT. The crops are a
        // hardcoded 504 px and the token count floors 504 / (P * 4) where the
        // convs round up, so 504 / P must be a multiple of 4: P = 14
        const int E = 32, P = 14, S = 56, C1 = 48, C2 = 64;
        vision(g, "step3vl", S, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_PREPROC_IMAGE_SIZE, 1008);
        patch_embd(g, P, E, (S / P) * (S / P), false);
        g.norm(fmt(TN_LN_PRE, "v", "weight"), E);
        g.bias(fmt(TN_LN_PRE, "v", "bias"), E);
        vit_layers(g, "v", 2, E, 64, 2);
        g.t(fmt(TN_LLAVA_PROJ, 0, "weight"), {3, 3, E, C1}, 0.05f);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), C1);
        g.t(fmt(TN_LLAVA_PROJ, 1, "weight"), {3, 3, C1, C2}, 0.05f);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), C2);
        g.w(fmt(TN_MM_PROJECTOR, "weight"), C2, N_TEXT);
    }});
    f.push_back({"dots_ocr", "qwen3", "vision", "img:64x64,96x48", [](G & g) {
        const int E = 64, P = 8, H = 128;
        vision(g, "dots_ocr", 64, P, false);
        g.b(KEY_USE_SILU, true);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        g.u32(KEY_IMAGE_MIN_PIXELS, 32 * 32);
        g.u32(KEY_IMAGE_MAX_PIXELS, 128 * 128);
        patch_embd(g, P, E, 0);
        VitOpts o;
        o.ln_bias = false;
        o.gated = true;
        o.attn_bias = false;
        o.ffn_bias = false;
        vit_layers(g, "v", 2, E, 128, 4, o);
        g.norm(fmt(TN_MM_POST_NORM, "weight"), E);
        g.norm(TN_MM_INP_NORM, E);
        g.bias(TN_MM_INP_NORM_B, E);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), 4 * E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});

    f.push_back({"hunyuanvl", "hunyuanvl", "vision", "img:64x64,96x48", [](G & g) {
        // positions are resampled on the host from an 8x8 grid. The text
        // model must decode with M-RoPE: otherwise mtmd counts nx = all the
        // tokens, ny = 1, and its own BOI/EOI/newline count asserts
        // (mtmd.cpp:1484)
        const int E = 32, P = 8, C1 = 64, C2 = 96;
        vision(g, "hunyuanvl", 64, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_IMAGE_MIN_PIXELS, 32 * 32);
        g.u32(KEY_IMAGE_MAX_PIXELS, 128 * 128);
        patch_embd(g, P, E, 8 * 8);
        vit_layers(g, "v", 2, E, 64, 2);
        g.norm(fmt(TN_MM_PRE_NORM, "weight"), E);
        g.t(fmt(TN_LLAVA_PROJ, 0, "weight"), {2, 2, E, C1}, 0.08f);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), C1);
        g.t(fmt(TN_LLAVA_PROJ, 2, "weight"), {1, 1, C1, C2}, 0.1f);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), C2);
        g.t(TN_IMAGE_NEWLINE, {C2}, 0.1f);
        g.w(fmt(TN_MM_PROJECTOR, "weight"), C2, N_TEXT);
        g.bias(fmt(TN_MM_PROJECTOR, "bias"), N_TEXT);
        g.norm(fmt(TN_MM_POST_NORM, "weight"), N_TEXT);
        g.t(TN_MM_IMG_BEGIN, {N_TEXT}, 0.1f);
        g.t(TN_MM_IMG_END, {N_TEXT}, 0.1f);
    }});
    f.push_back({"minimax_m3", "qwen3", "vision", "img:64x64,96x48", [](G & g) {
        // 3-axis rope needs d_head >= 3 * axd: d_head 16
        const int E = 64, P = 8, H = 96, P2 = 64, M = 128;
        vision(g, "minimax_m3", 64, P);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.t(TN_PATCH_EMBD, {P, P, 3, E}, 0.05f);
        g.t(TN_PATCH_EMBD_1, {P, P, 3, E}, 0.05f);
        vit_layers(g, "v", 2, E, 128, 4);
        g.w(fmt(TN_LLAVA_PROJ, 1, "weight"), E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, P2);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), P2);
        g.w(fmt(TN_MM_MERGER_FC1, "weight"), 4 * P2, M);
        g.bias(fmt(TN_MM_MERGER_FC1, "bias"), M);
        g.w(fmt(TN_MM_MERGER_FC2, "weight"), M, N_TEXT);
        g.bias(fmt(TN_MM_MERGER_FC2, "bias"), N_TEXT);
    }});

    f.push_back({"dots3note_v", "qwen3", "vision", "img:64x64,96x48", [](G & g) {
        // dots_ocr's ViT with a MoE ffn on the second layer (top-2 of 4)
        const int E = 64, P = 8, H = 128;
        vision(g, "dots3note_v", 64, P, false);
        g.b(KEY_USE_SILU, true);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(KEY_SPATIAL_MERGE_SIZE, 2);
        g.u32(KEY_VISION_N_EXPERT_USED, 2);
        g.u32(KEY_IMAGE_MIN_PIXELS, 32 * 32);
        g.u32(KEY_IMAGE_MAX_PIXELS, 128 * 128);
        patch_embd(g, P, E, 0);
        VitOpts o;
        o.ln_bias = false;
        o.gated = true;
        o.attn_bias = false;
        o.ffn_bias = false;
        o.n_expert = 4;
        o.moe_from = 1;
        vit_layers(g, "v", 2, E, 128, 4, o);
        g.norm(fmt(TN_MM_POST_NORM, "weight"), E);
        g.norm(TN_MM_INP_NORM, E);
        g.bias(TN_MM_INP_NORM_B, E);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), 4 * E, H);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});

    f.push_back({"resampler", "qwen3", "vision", "img:32x32,96x48,40x100", [](G & g) {
        // minicpmv 2.6-style (version 4): a perceiver resampler with 16
        // queries; ViT positions are 70x70 bucket ids, so 4900 rows
        const int E = 32, P = 8, D = N_TEXT;
        vision(g, "resampler", 32, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.i32(KEY_MINICPMV_VERSION, 4);
        g.u32(KEY_MINICPMV_QUERY_NUM, 16);
        patch_embd(g, P, E, 70 * 70);
        vit_layers(g, "v", 2, E, 64, 2);
        g.norm(fmt(TN_LN_POST, "v", "weight"), E);
        g.bias(fmt(TN_LN_POST, "v", "bias"), E);
        g.t(TN_MINICPMV_POS_EMBD_K, {D, 64}, 0.02f);
        g.t(TN_MINICPMV_QUERY, {D, 16}, 0.1f);
        g.w(TN_MINICPMV_PROJ, D, D);
        g.w(TN_MINICPMV_KV_PROJ, E, D);
        for (const char * n : {"q", "k", "v", "out"}) {
            g.w(fmt(TN_MINICPMV_ATTN, n, "weight"), D, D);
            g.bias(fmt(TN_MINICPMV_ATTN, n, "bias"), D);
        }
        for (const char * n : {"q", "kv", "post"}) {
            g.norm(fmt(TN_MINICPMV_LN, n, "weight"), D);
            g.bias(fmt(TN_MINICPMV_LN, n, "bias"), D);
        }
    }});
    f.push_back({"minicpmv4_6", "qwen3", "vision", "img:64x64,96x48,40x100", [](G & g) {
        // the 4x merge: a windowed-attention ViT merger after layer 0, then
        // the final 2x2 merger
        const int E = 32, P = 8, F2 = 64, H = 128;
        vision(g, "minicpmv4_6", 64, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.ai32(KEY_WIN_ATTN_LAYER_INDEXES, {0});
        patch_embd(g, P, E, 70 * 70);
        vit_layers(g, "v", 2, E, 64, 2);
        g.norm(fmt(TN_LN_POST, "v", "weight"), E);
        g.bias(fmt(TN_LN_POST, "v", "bias"), E);
        g.norm(fmt(TN_VIT_MERGER_LN1, "weight"), E);
        g.bias(fmt(TN_VIT_MERGER_LN1, "bias"), E);
        for (const char * n : {TN_VIT_MERGER_ATTN_Q, TN_VIT_MERGER_ATTN_K, TN_VIT_MERGER_ATTN_V, TN_VIT_MERGER_ATTN_O}) {
            g.w(fmt(n, "weight"), E, E);
            g.bias(fmt(n, "bias"), E);
        }
        g.norm(fmt(TN_VIT_MERGER_DS_LN, "weight"), 4 * E);
        g.bias(fmt(TN_VIT_MERGER_DS_LN, "bias"), 4 * E);
        g.w(fmt(TN_VIT_MERGER_DS_UP, "weight"), 4 * E, F2);
        g.bias(fmt(TN_VIT_MERGER_DS_UP, "bias"), F2);
        g.w(fmt(TN_VIT_MERGER_DS_DOWN, "weight"), F2, E);
        g.bias(fmt(TN_VIT_MERGER_DS_DOWN, "bias"), E);
        g.norm(TN_MM_INP_NORM, 4 * E);
        g.bias(TN_MM_INP_NORM_B, 4 * E);
        g.w(fmt(TN_MM_UP, "weight"), 4 * E, H);
        g.bias(fmt(TN_MM_UP, "bias"), H);
        g.w(fmt(TN_MM_DOWN, "weight"), H, N_TEXT);
        g.bias(fmt(TN_MM_DOWN, "bias"), N_TEXT);
    }});

    f.push_back({"yasa2", "qwen3", "vision", "img:128x128,200x90", [](G & g) {
        // a ConvNeXt: one 4x4 patchify, a block at 32x32, a 2x2 downsample to
        // 16x16 and a block there, then an 8x8 average pool: 64 tokens
        const int P = 4, C0 = 16, C1 = 32, H = 128;
        vision(g, "yasa2", 128, P);
        hp(g, "vision", C1, 64, 1, 0, N_TEXT); // no ViT layers
        g.t(TN_PATCH_EMBD, {P, P, 3, C0}, 0.2f);
        g.bias(TN_PATCH_BIAS, C0);
        g.norm(TN_YASA_PATCH_LN_W, C0);
        g.bias(TN_YASA_PATCH_LN_B, C0);
        auto block = [&](int st, int C) {
            // depthwise kernels are F16, as for LDP
            g.t(fmt(TN_YASA_STAGE_BLK, st, 0, "dw", "weight"), {7, 7, 1, C}, 0.1f, 0.0f, GGML_TYPE_F16);
            g.bias(fmt(TN_YASA_STAGE_BLK, st, 0, "dw", "bias"), C);
            g.norm(fmt(TN_YASA_STAGE_BLK, st, 0, "ln", "weight"), C);
            g.bias(fmt(TN_YASA_STAGE_BLK, st, 0, "ln", "bias"), C);
            g.w(fmt(TN_YASA_STAGE_BLK, st, 0, "pw1", "weight"), C, 4 * C);
            g.bias(fmt(TN_YASA_STAGE_BLK, st, 0, "pw1", "bias"), 4 * C);
            g.norm(fmt(TN_YASA_STAGE_BLK, st, 0, "grn", "weight"), 4 * C);
            g.bias(fmt(TN_YASA_STAGE_BLK, st, 0, "grn", "bias"), 4 * C);
            g.w(fmt(TN_YASA_STAGE_BLK, st, 0, "pw2", "weight"), 4 * C, C);
            g.bias(fmt(TN_YASA_STAGE_BLK, st, 0, "pw2", "bias"), C);
        };
        block(0, C0);
        g.norm(fmt(TN_YASA_STAGE_DOWN_LN, 1, "weight"), C0);
        g.bias(fmt(TN_YASA_STAGE_DOWN_LN, 1, "bias"), C0);
        g.t(fmt(TN_YASA_STAGE_DOWN_CONV, 1, "weight"), {2, 2, C0, C1}, 0.1f);
        g.bias(fmt(TN_YASA_STAGE_DOWN_CONV, 1, "bias"), C1);
        block(1, C1);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), C1, H);
        g.bias(fmt(TN_LLAVA_PROJ, 0, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
    }});
    f.push_back({"muse-glimmer", "qwen2vl", "vision", "img:64x64,96x48", [](G & g) {
        // a 4x4 position grid sets 4x4-patch sparse windows: layer 0 is
        // windowed, the last layer global
        const int E = 32, P = 8, H = 64;
        vision(g, "muse-glimmer", 64, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        patch_embd(g, P, E, 4 * 4);
        vit_layers(g, "v", 2, E, 64, 2);
        g.w(fmt(TN_LLAVA_PROJ, 0, "weight"), 4 * E, H);
        g.w(fmt(TN_LLAVA_PROJ, 1, "weight"), H, H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
    }});

    // --- gemma4: x/y position tables [E, n, 2], 2-axis neox rope, avg pool
    f.push_back({"gemma4v", "qwen3", "vision", "img:64x64,120x80", [](G & g) {
        const int E = 32, P = 8;
        vision(g, "gemma4v", 64, P);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        g.t(TN_PATCH_EMBD, {P, P, 3, E}, 1.0f / std::sqrt(3.0f * P * P));
        g.t(fmt(TN_POS_EMBD, "v"), {E, 32, 2}, 0.02f);
        VitOpts o;
        o.attn_bias = false;
        o.ln_bias = false;
        o.ffn_bias = false;
        o.gated = true;
        o.qk_norm = 2;
        vit_layers(g, "v", 2, E, 64, 2, o);
        for (int il = 0; il < 2; il++) {
            g.norm(fmt(TN_ATTN_POST_NORM, "v", il, "weight"), E);
            g.norm(fmt(TN_FFN_POST_NORM, "v", il, "weight"), E);
        }
        g.t(TN_STD_BIAS, {E}, 0.05f);
        g.t(TN_STD_SCALE, {E}, 0.1f, 1.0f);
        g.w(TN_MM_INP_PROJ, E, N_TEXT);
        // Gemma4ClippableLinear: scalar clamps on the projection's input
        // and output, and on one layer's q
        g.t("mm.input_projection.input_max", {1}, 0.0f, 3.0f);
        g.t("mm.input_projection.input_min", {1}, 0.0f, -3.0f);
        g.t("mm.input_projection.output_max", {1}, 0.0f, 2.0f);
        g.t("mm.input_projection.output_min", {1}, 0.0f, -2.0f);
        g.t(fmt(TN_ATTN_Q, "v", 1, "input_max"), {1}, 0.0f, 1.5f);
    }});
    f.push_back({"gemma4uv", "qwen3", "vision", "img:64x64,120x80", [](G & g) {
        // encoder-free: patch size 8 x merge 2 = 16, norms around a matmul
        const int E = 64, P = 8, PP = 16;
        vision(g, "gemma4uv", 64, P);
        hp(g, "vision", E, 64, 2, 0, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, 2);
        g.norm(fmt(TN_PATCH_NORM, 1, "weight"), 3 * PP * PP);
        g.bias(fmt(TN_PATCH_NORM, 1, "bias"), 3 * PP * PP);
        g.w(TN_PATCH_EMBD, 3 * PP * PP, E);
        g.bias(TN_PATCH_BIAS, E);
        g.norm(fmt(TN_PATCH_NORM, 2, "weight"), E);
        g.bias(fmt(TN_PATCH_NORM, 2, "bias"), E);
        g.t(fmt(TN_POS_EMBD, "v"), {E, 32, 2}, 0.02f);
        g.norm(fmt(TN_PATCH_NORM, 3, "weight"), E);
        g.bias(fmt(TN_PATCH_NORM, 3, "bias"), E);
        g.w(TN_MM_INP_PROJ, E, N_TEXT);
    }});

    f.push_back({"granite4_vision", "qwen3", "vision", "img:64x64,128x64,100x140", [](G & g) {
        // anyres tiles through a SigLIP tower, then two window Q-Formers
        // (the Q-Former's head size is a hardcoded 64), one spatially
        // gathered and one average-pooled, concatenated to 2 x 128
        const int E = 64, P = 8, F = 128, PD = N_TEXT / 2;
        vision(g, "granite4_vision", 64, P);
        hp(g, "vision", E, F, 2, 2, PD);
        g.ai32(KEY_IMAGE_GRID_PINPOINTS, {64, 64, 64, 128, 128, 64, 128, 128});
        g.ai32(fmt(KEY_FEATURE_LAYERS, "vision"), {0, 1});
        g.ai32(KEY_PROJ_SPATIAL_OFFSETS, {0, -1});
        g.u32(KEY_PROJ_SAMPLE_QUERY_SIDE, 2);
        g.u32(KEY_PROJ_SAMPLE_WINDOW_SIDE, 4);
        patch_embd(g, P, E, 8 * 8);
        vit_layers(g, "v", 2, E, F, 2);
        g.t(TN_IMAGE_NEWLINE, {PD}, 0.1f);
        for (int bid = 0; bid < 2; bid++) {
            g.t(fmt(TN_MULTI_PROJ_IMG_POS, bid), {E, 16}, 0.05f);
            g.t(fmt(TN_MULTI_PROJ_QUERY, "v", bid), {E, 4}, 0.5f);
            g.w(fmt(TN_MULTI_PROJ_LINEAR, "v", bid, "weight"), E, PD);
            g.bias(fmt(TN_MULTI_PROJ_LINEAR, "v", bid, "bias"), PD);
            g.norm(fmt(TN_MULTI_PROJ_NORM, "v", bid, "weight"), E);
            g.bias(fmt(TN_MULTI_PROJ_NORM, "v", bid, "bias"), E);
            g.norm(fmt(TN_MULTI_PROJ_POST_NORM, "v", bid, "weight"), E);
            g.bias(fmt(TN_MULTI_PROJ_POST_NORM, "v", bid, "bias"), E);
            auto lin = [&](const char * n, int in, int out) {
                g.w(fmt(n, "v", bid, "weight"), in, out);
                g.bias(fmt(n, "v", bid, "bias"), out);
            };
            auto nrm = [&](const char * n) {
                g.norm(fmt(n, "v", bid, "weight"), E);
                g.bias(fmt(n, "v", bid, "bias"), E);
            };
            lin(TN_QF_SELF_ATTN_Q, E, E);
            lin(TN_QF_SELF_ATTN_K, E, E);
            lin(TN_QF_SELF_ATTN_V, E, E);
            lin(TN_QF_SELF_ATTN_O, E, E);
            nrm(TN_QF_SELF_ATTN_N);
            lin(TN_QF_CROSS_ATTN_Q, E, E);
            lin(TN_QF_CROSS_ATTN_K, E, E);
            lin(TN_QF_CROSS_ATTN_V, E, E);
            lin(TN_QF_CROSS_ATTN_O, E, E);
            nrm(TN_QF_CROSS_ATTN_N);
            lin(TN_QF_FFN_UP, E, F);
            lin(TN_QF_FFN_DOWN, F, E);
            nrm(TN_QF_FFN_NORM);
        }
    }});

    // --- DeepSeek-OCR: a SAM ViT-B at a hardcoded 1024 px / patch 16 (640
    // and 768 px tiles), its neck and two stride-2 convs, then a CLIP or a
    // Qwen2 encoder. The loader always reads 12 SAM blocks (n_sam_layers is
    // a model constant, not the block_count key), so all 12 are written;
    // the graph runs sam.block_count = 3, block 2 global over 64x64 patches
    auto sam = [](G & g, int E) {
        const int Es = 32, F = 64, Cn = 32, C2 = 48, W = 14;
        g.u32(KEY_SAM_N_BLOCK, 3);
        g.u32(KEY_SAM_N_HEAD, 2);
        g.u32(KEY_SAM_N_EMBD, Es);
        g.u32(KEY_ATTN_WINDOW_SIZE, W);
        g.t(fmt(TN_SAM_PATCH_EMBD, "weight"), {16, 16, 3, Es}, 1.0f / 27.7f);
        g.bias(fmt(TN_SAM_PATCH_EMBD, "bias"), Es);
        g.t(fmt(TN_SAM_POS_EMBD, "weight"), {Es, 64, 64}, 0.02f);
        for (int il = 0; il < 12; il++) {
            const bool global = il % 3 == 2;
            g.w(fmt(TN_SAM_ATTN_QKV, il, "weight"), Es, 3 * Es);
            g.bias(fmt(TN_SAM_ATTN_QKV, il, "bias"), 3 * Es);
            g.w(fmt(TN_SAM_ATTN_OUT, il, "weight"), Es, Es);
            g.bias(fmt(TN_SAM_ATTN_OUT, il, "bias"), Es);
            g.norm(fmt(TN_SAM_PRE_NORM, il, "weight"), Es);
            g.bias(fmt(TN_SAM_PRE_NORM, il, "bias"), Es);
            g.norm(fmt(TN_SAM_POST_NORM, il, "weight"), Es);
            g.bias(fmt(TN_SAM_POST_NORM, il, "bias"), Es);
            g.t(fmt(TN_SAM_ATTN_POS_H, il, "weight"), {Es / 2, global ? 127 : 2 * W - 1}, 0.05f);
            g.t(fmt(TN_SAM_ATTN_POS_W, il, "weight"), {Es / 2, global ? 127 : 2 * W - 1}, 0.05f);
            g.w(fmt(TN_SAM_FFN_UP, il, "weight"), Es, F);
            g.bias(fmt(TN_SAM_FFN_UP, il, "bias"), F);
            g.w(fmt(TN_SAM_FFN_DOWN, il, "weight"), F, Es);
            g.bias(fmt(TN_SAM_FFN_DOWN, il, "bias"), Es);
        }
        g.t(fmt(TN_SAM_NECK, 0, "weight"), {1, 1, Es, Cn}, 0.15f);
        g.norm(fmt(TN_SAM_NECK, 1, "weight"), Cn);
        g.bias(fmt(TN_SAM_NECK, 1, "bias"), Cn);
        g.t(fmt(TN_SAM_NECK, 2, "weight"), {3, 3, Cn, Cn}, 0.05f);
        g.norm(fmt(TN_SAM_NECK, 3, "weight"), Cn);
        g.bias(fmt(TN_SAM_NECK, 3, "bias"), Cn);
        g.t(fmt(TN_SAM_NET, 2, "weight"), {3, 3, Cn, C2}, 0.05f);
        g.t(fmt(TN_SAM_NET, 3, "weight"), {3, 3, C2, E}, 0.04f);
    };
    f.push_back({"deepseekocr", "qwen3", "vision", "img:64x64,1300x700", [sam](G & g) {
        const int E = 64;
        vision(g, "deepseekocr", 1024, 16);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        sam(g, E);
        g.t(fmt(TN_POS_EMBD, "v"), {E, 16 * 16 + 1}, 0.02f);
        g.t(TN_CLASS_EMBD, {E}, 0.05f);
        g.norm(fmt(TN_LN_PRE, "v", "weight"), E);
        g.bias(fmt(TN_LN_PRE, "v", "bias"), E);
        vit_layers(g, "v", 2, E, 128, 4);
        g.w(fmt(TN_MM_PROJECTOR, "weight"), 2 * E, N_TEXT);
        g.bias(fmt(TN_MM_PROJECTOR, "bias"), N_TEXT);
        g.t(TN_IMAGE_NEWLINE, {N_TEXT}, 0.1f);
        g.t(TN_IMAGE_SEPERATOR, {N_TEXT}, 0.1f);
    }});
    f.push_back({"deepseekocr2", "qwen3", "vision", "img:64x64,1300x700", [sam](G & g) {
        // a Qwen2 encoder (GQA, rope, causal-mask input) over SAM features
        // and 256 (global) or 144 (768 px tile) learned queries
        const int E = 64;
        vision(g, "deepseekocr2", 1024, 16);
        hp(g, "vision", E, 128, 4, 2, N_TEXT);
        g.u32(fmt(KEY_N_HEAD_KV, "vision"), 2);
        sam(g, E);
        VitOpts o;
        o.n_head_kv = 2;
        o.ln_bias = false;
        o.gated = true;
        o.ffn_bias = false;
        vit_layers(g, "v", 2, E, 128, 4, o);
        g.t(fmt(TN_RESMPL_QUERY, 1024, "weight"), {E, 256}, 0.1f);
        g.t(fmt(TN_RESMPL_QUERY, 768, "weight"), {E, 144}, 0.1f);
        g.w(fmt(TN_MM_PROJECTOR, "weight"), E, N_TEXT);
        g.bias(fmt(TN_MM_PROJECTOR, "bias"), N_TEXT);
        g.t(TN_IMAGE_SEPERATOR, {N_TEXT}, 0.1f);
    }});

    f.push_back({"gemma3nv", "qwen3", "vision", "img:64x64,300x200", [](G & g) {
        // MobileNetV5 at 512 px: a stem and four stride-2 stages (edge
        // residual; inverted residual; inverted residual + multi-query
        // attention; inverted residual), stages 2 and 3 fused at 32x32 and
        // pooled to 16x16. The token count is image_size / patch_size, so
        // patch_size is 2 to make it the 256 the graph produces.
        // clip_n_mmproj_embd reads mm.input_projection's ne[0] -- its input
        // width -- so the projection must be square: E = the text width
        const int E = N_TEXT, C0 = 8, C1 = 16, C2 = 24, C3 = 32, Cm = 64, NH = 2, D = 8;
        vision(g, "gemma3nv", 512, 2);
        hp(g, "vision", E, 64, 1, 0, N_TEXT);
        auto dw = [&](const std::string & n, int C) {
            // F16: ggml_conv_2d_dw's im2col is F16 (see LDP)
            g.t(n, {3, 3, 1, C}, 0.2f, 0.0f, GGML_TYPE_F16);
        };
        auto pw = [&](const std::string & n, int in, int out) {
            g.t(n, {1, 1, in, out}, 1.0f / std::sqrt((float) in));
        };
        auto ls = [&](const std::string & n, int C) { g.t(n, {1, 1, C}, 0.1f, 1.0f); };
        g.t(TN_MNV5_STEM_CONV, {3, 3, 3, C0}, 0.2f);
        g.t(TN_MNV5_STEM_BIAS, {1, 1, C0}, 0.02f);
        g.norm(TN_MNV5_STEM_BN, C0);
        // stage 0: edge residual
        g.t(fmt(TN_MNV5_BLK_S0_EXP_W, 0, 0), {3, 3, C0, 2 * C0}, 0.1f);
        g.norm(fmt(TN_MNV5_BLK_S0_BN1_W, 0, 0), 2 * C0);
        pw(fmt(TN_MNV5_BLK_S0_PWL_W, 0, 0), 2 * C0, C1);
        g.norm(fmt(TN_MNV5_BLK_S0_BN2_W, 0, 0), C1);
        // stages 1-3 open with an inverted residual
        auto uir = [&](int st, int blk, int in, int out) {
            dw(fmt(TN_MNV5_BLK_DW_START_W, st, blk), in);
            g.norm(fmt(TN_MNV5_BLK_DW_START_BN, st, blk), in);
            pw(fmt(TN_MNV5_BLK_PW_EXP_W, st, blk), in, 2 * in);
            g.norm(fmt(TN_MNV5_BLK_PW_EXP_BN, st, blk), 2 * in);
            dw(fmt(TN_MNV5_BLK_DW_MID_W, st, blk), 2 * in);
            g.norm(fmt(TN_MNV5_BLK_DW_MID_BN, st, blk), 2 * in);
            pw(fmt(TN_MNV5_BLK_PW_PROJ_W, st, blk), 2 * in, out);
            g.norm(fmt(TN_MNV5_BLK_PW_PROJ_BN, st, blk), out);
            ls(fmt(TN_MNV5_BLK_LAYER_SCALE, st, blk), out);
        };
        uir(1, 0, C1, C2);
        uir(2, 0, C2, C2);
        // stage 2, block 1: multi-query attention with strided k/v
        pw(fmt(TN_MNV5_ATTN_Q_W, 2, 1), C2, NH * D);
        pw(fmt(TN_MNV5_ATTN_K_W, 2, 1), C2, D);
        pw(fmt(TN_MNV5_ATTN_V_W, 2, 1), C2, D);
        pw(fmt(TN_MNV5_ATTN_O_W, 2, 1), NH * D, C2);
        dw(fmt(TN_MNV5_ATTN_K_DW, 2, 1), C2);
        g.norm(fmt(TN_MNV5_ATTN_K_NORM, 2, 1), C2);
        dw(fmt(TN_MNV5_ATTN_V_DW, 2, 1), C2);
        g.norm(fmt(TN_MNV5_ATTN_V_NORM, 2, 1), C2);
        g.norm(fmt(TN_MNV5_ATTN_NORM, 2, 1), C2);
        ls(fmt(TN_MNV5_BLK_LAYER_SCALE, 2, 1), C2);
        uir(3, 0, C2, C3);
        pw(TN_MNV5_MSFA_FFN_EXP_W, C2 + C3, Cm);
        g.norm(TN_MNV5_MSFA_FFN_EXP_BN, Cm);
        pw(TN_MNV5_MSFA_FFN_PROJ_W, Cm, E);
        g.norm(TN_MNV5_MSFA_FFN_PROJ_BN, E);
        g.norm(TN_MNV5_MSFA_NORM, E);
        g.norm(TN_MM_SOFT_EMB_N, E);
        g.w(TN_MM_INP_PROJ, E, N_TEXT);
    }});

    // --- whisper encoder family (models/whisper-enc.cpp)
    auto whisper = [](G & g, const char * proj, int E) {
        audio(g, proj, 32);
        hp(g, "audio", E, 128, 4, 2, N_TEXT);
        g.t(fmt(TN_CONV1D, 1, "weight"), {3, 32, E}, 0.1f);
        g.t(fmt(TN_CONV1D, 1, "bias"), {1, E}, 0.02f);
        g.t(fmt(TN_CONV1D, 2, "weight"), {3, E, E}, 0.05f);
        g.t(fmt(TN_CONV1D, 2, "bias"), {1, E}, 0.02f);
        g.t(fmt(TN_POS_EMBD, "a"), {E, 1500}, 0.02f);
        VitOpts o;
        o.k_bias = false;
        vit_layers(g, "a", 2, E, 128, 4, o);
        g.norm(fmt(TN_LN_POST, "a", "weight"), E);
        g.bias(fmt(TN_LN_POST, "a", "bias"), E);
    };
    f.push_back({"qwen2.5o", "qwen2vl", "vision+audio", "img:64x64,96x48;aud:16000", [whisper, qwen2vl_like](G & g) {
        // the legacy omni type: one file, both encoders; clip resolves it
        // to qwen2.5vl_merger for vision and qwen2a for audio
        qwen2vl_like(g, "qwen2.5o", true, 2);
        g.b(KEY_HAS_AUDIO_ENC, true);
        whisper(g, "qwen2.5o", 32);
        g.w(fmt(TN_MM_AUDIO_FC, "weight"), 32, N_TEXT);
        g.bias(fmt(TN_MM_AUDIO_FC, "bias"), N_TEXT);
    }});
    f.push_back({"qwen3a", "qwen3", "audio", AUD, [](G & g) {
        // three stride-2 conv2d over 100-frame chunks: 13 tokens a chunk
        const int E = 64, C = 16, NM = 32, H = 128;
        audio(g, "qwen3a", NM);
        hp(g, "audio", E, 128, 4, 2, N_TEXT);
        g.t(fmt(TN_CONV2D, 1, "weight"), {3, 3, 1, C}, 0.3f);
        g.bias(fmt(TN_CONV2D, 1, "bias"), C);
        g.t(fmt(TN_CONV2D, 2, "weight"), {3, 3, C, C}, 0.08f);
        g.bias(fmt(TN_CONV2D, 2, "bias"), C);
        g.t(fmt(TN_CONV2D, 3, "weight"), {3, 3, C, C}, 0.08f);
        g.bias(fmt(TN_CONV2D, 3, "bias"), C);
        g.w(fmt(TN_CONV_OUT, "weight"), C * (NM / 8), E);
        g.t(fmt(TN_POS_EMBD, "a"), {E, 13}, 0.02f);
        vit_layers(g, "a", 2, E, 128, 4);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), E, H);
        g.bias(fmt(TN_MM_AUDIO_MLP, 1, "bias"), H);
        g.w(fmt(TN_MM_AUDIO_MLP, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_MM_AUDIO_MLP, 2, "bias"), N_TEXT);
    }});
    f.push_back({"dots3note_a", "qwen3", "audio", AUD, [](G & g) {
        // qwen3a's conv stack over the whole clip, then a rope'd RMS ViT
        const int E = 64, C = 16, NM = 32, H = 128;
        audio(g, "dots3note_a", NM);
        hp(g, "audio", E, 128, 4, 2, N_TEXT);
        g.t(fmt(TN_CONV2D, 1, "weight"), {3, 3, 1, C}, 0.3f);
        g.bias(fmt(TN_CONV2D, 1, "bias"), C);
        g.t(fmt(TN_CONV2D, 2, "weight"), {3, 3, C, C}, 0.08f);
        g.bias(fmt(TN_CONV2D, 2, "bias"), C);
        g.t(fmt(TN_CONV2D, 3, "weight"), {3, 3, C, C}, 0.08f);
        g.bias(fmt(TN_CONV2D, 3, "bias"), C);
        g.w(fmt(TN_CONV_OUT, "weight"), C * (NM / 8), E);
        VitOpts o;
        o.ln_bias = false;
        vit_layers(g, "a", 2, E, 128, 4, o);
        g.norm(fmt(TN_MM_NORM_PRE, "weight"), E);
        g.bias(fmt(TN_MM_NORM_PRE, "bias"), E);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), E, H);
        g.bias(fmt(TN_MM_AUDIO_MLP, 1, "bias"), H);
        g.w(fmt(TN_MM_AUDIO_MLP, 3, "weight"), H, N_TEXT);
        g.bias(fmt(TN_MM_AUDIO_MLP, 3, "bias"), N_TEXT);
    }});
    f.push_back({"gemma4a", "qwen3", "audio", AUD, [](G & g) {
        // a conformer: two stride-2 conv2d, then per layer a half-step ffn,
        // chunked attention with a relative-position term, a gated depthwise
        // conv (ssm_conv, kernel 5) and a second half-step ffn
        const int E = 64, NH = 4, DH = 16, F = 128, NM = 32, C0 = 8, C1 = 16, Eo = 96;
        audio(g, "gemma4a", NM);
        hp(g, "audio", E, F, NH, 2, N_TEXT);
        g.t(fmt(TN_A_CONV1D, 0, "weight"), {3, 3, 1, C0}, 0.3f);
        g.norm(fmt(TN_A_CONV1D_NORM, 0, "weight"), C0);
        g.t(fmt(TN_A_CONV1D, 1, "weight"), {3, 3, C0, C1}, 0.1f);
        g.norm(fmt(TN_A_CONV1D_NORM, 1, "weight"), C1);
        g.w(fmt(TN_A_INP_PROJ, "weight"), C1 * (NM / 4), E);
        for (int il = 0; il < 2; il++) {
            const char * a = "a";
            g.norm(fmt(TN_FFN_NORM, a, il, "weight"), E);
            g.w(fmt(TN_FFN_UP, a, il, "weight"), E, F);
            g.w(fmt(TN_FFN_DOWN, a, il, "weight"), F, E);
            g.norm(fmt(TN_FFN_POST_NORM, a, il, "weight"), E);
            g.norm(fmt(TN_A_ATTN_PRE_NORM, a, il, "weight"), E);
            g.w(fmt(TN_ATTN_Q, a, il, "weight"), E, E);
            g.w(fmt(TN_ATTN_K, a, il, "weight"), E, E);
            g.w(fmt(TN_ATTN_V, a, il, "weight"), E, E);
            g.w(fmt(TN_ATTN_OUTPUT, a, il, "weight"), E, E);
            g.t(fmt(TN_A_PER_DIM_SCALE, a, il, "weight"), {DH}, 0.2f, 1.0f);
            g.t(fmt(TN_A_PER_DIM_K_SCALE, a, il, "weight"), {DH}, 0.2f, 1.0f);
            g.w(fmt(TN_A_ATTN_K_REL, a, il, "weight"), E, NH * DH);
            g.norm(fmt(TN_ATTN_POST_NORM, a, il, "weight"), E);
            g.norm(fmt(TN_CONV_NORM, a, il, "weight"), E); // loaded as norm_conv
            g.w(fmt(TN_CONV_PW1, a, il, "weight"), E, 2 * E);
            g.t(fmt(TN_CONV_DW, a, il, "weight"), {5, E}, 0.3f);
            g.norm(fmt(TN_NORM_CONV, a, il, "weight"), E); // loaded as conv_norm
            g.w(fmt(TN_CONV_PW2, a, il, "weight"), E, E);
            g.norm(fmt(TN_FFN_NORM_1, a, il, "weight"), E);
            g.w(fmt(TN_FFN_UP_1, a, il, "weight"), E, F);
            g.w(fmt(TN_FFN_DOWN_1, a, il, "weight"), F, E);
            g.norm(fmt(TN_A_FFN_POST_NORM_1, a, il, "weight"), E);
            g.norm(fmt(TN_LN_2, a, il, "weight"), E);
        }
        g.w(fmt(TN_A_OUT_PROJ, "weight"), E, Eo);
        g.bias(fmt(TN_A_OUT_PROJ, "bias"), Eo);
        g.norm(fmt(TN_A_MM_SOFT_EMB_N, "weight"), Eo);
        g.w(fmt(TN_A_MM_INP_PROJ, "weight"), Eo, N_TEXT);
        g.t("a.blk.0.attn_q.input_max", {1}, 0.0f, 2.0f);
        g.t("a.blk.0.attn_q.input_min", {1}, 0.0f, -2.0f);
    }});
    f.push_back({"lfm2a", "qwen3", "audio", AUD, [](G & g) {
        // FastConformer: a conv2d + two depthwise-separable stride-2 stages,
        // then rel-pos attention (a hardcoded 512-wide sinusoid), a gated
        // depthwise conv (kernel 9) and two half-step ffns
        const int E = 64, NH = 4, DH = 16, F = 128, NM = 32, C = 16, H = 128;
        audio(g, "lfm2a", NM);
        hp(g, "audio", E, F, NH, 2, N_TEXT, 1e-5f);
        g.t(fmt(TN_CONV1D, 0, "weight"), {3, 3, 1, C}, 0.3f);
        g.t(fmt(TN_CONV1D, 0, "bias"), {1, 1, C}, 0.02f);
        for (int i : {2, 5}) {
            g.t(fmt(TN_CONV1D, i, "weight"), {3, 3, 1, C}, 0.3f);
            g.t(fmt(TN_CONV1D, i, "bias"), {1, 1, C}, 0.02f);
        }
        for (int i : {3, 6}) {
            g.t(fmt(TN_CONV1D, i, "weight"), {1, 1, C, C}, 0.25f);
            g.t(fmt(TN_CONV1D, i, "bias"), {1, 1, C}, 0.02f);
        }
        g.w(fmt(TN_PRE_ENCODE_OUT, "weight"), C * (NM / 8), E);
        g.bias(fmt(TN_PRE_ENCODE_OUT, "bias"), E);
        // only its width is read: clip_n_mmproj_embd is position_embd->ne[0]
        // and ne[1] must cover n_frames / 2, warmup included (3000 frames)
        g.t(fmt(TN_POS_EMBD, "a"), {N_TEXT, 1500}, 0.02f);
        VitOpts o;
        vit_layers(g, "a", 2, E, F, NH, o);
        for (int il = 0; il < 2; il++) {
            const char * a = "a";
            g.norm(fmt(TN_FFN_NORM, a, il, "weight"), E);
            g.bias(fmt(TN_FFN_NORM, a, il, "bias"), E);
            g.norm(fmt(TN_FFN_NORM_1, a, il, "weight"), E);
            g.bias(fmt(TN_FFN_NORM_1, a, il, "bias"), E);
            g.w(fmt(TN_FFN_UP_1, a, il, "weight"), E, F);
            g.bias(fmt(TN_FFN_UP_1, a, il, "bias"), F);
            g.w(fmt(TN_FFN_DOWN_1, a, il, "weight"), F, E);
            g.bias(fmt(TN_FFN_DOWN_1, a, il, "bias"), E);
            g.t(fmt(TN_POS_BIAS_U, a, il), {DH, NH}, 0.05f);
            g.t(fmt(TN_POS_BIAS_V, a, il), {DH, NH}, 0.05f);
            g.norm(fmt(TN_NORM_CONV, a, il, "weight"), E);
            g.bias(fmt(TN_NORM_CONV, a, il, "bias"), E);
            g.w(fmt(TN_LINEAR_POS, a, il, "weight"), 512, E);
            g.norm(fmt(TN_CONV_NORM, a, il, "weight"), E);
            g.bias(fmt(TN_CONV_NORM, a, il, "bias"), E);
            g.t(fmt(TN_CONV_DW, a, il, "weight"), {9, E}, 0.2f);
            g.bias(fmt(TN_CONV_DW, a, il, "bias"), E);
            g.w(fmt(TN_CONV_PW1, a, il, "weight"), E, 2 * E);
            g.bias(fmt(TN_CONV_PW1, a, il, "bias"), 2 * E);
            g.w(fmt(TN_CONV_PW2, a, il, "weight"), E, E);
            g.bias(fmt(TN_CONV_PW2, a, il, "bias"), E);
        }
        g.norm(fmt(TN_MM_AUDIO_MLP, 0, "weight"), E);
        g.bias(fmt(TN_MM_AUDIO_MLP, 0, "bias"), E);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), E, H);
        g.bias(fmt(TN_MM_AUDIO_MLP, 1, "bias"), H);
        g.w(fmt(TN_MM_AUDIO_MLP, 3, "weight"), H, N_TEXT);
        g.bias(fmt(TN_MM_AUDIO_MLP, 3, "bias"), N_TEXT);
    }});
    f.push_back({"parakeet", "qwen3", "audio", AUD, [](G & g) {
        // NeMo FastConformer: mel filters and window ship as tensors, the
        // depthwise conv kernel is 9, batch norm is unfolded (mean / var)
        const int E = 64, NH = 4, DH = 16, F = 128, NM = 32, C = 16, H = 128, K = 9;
        audio(g, "parakeet", NM);
        hp(g, "audio", E, F, NH, 2, N_TEXT, 1e-5f);
        g.u32(KEY_AUDIO_SUBSMPL_FACTOR, 8);
        g.u32(KEY_A_CONV_KERNEL_SIZE, K);
        g.t(TN_MEL_FILTERS, {257, NM}, 0.01f, 0.01f);
        g.t(TN_WINDOW, {400}, 0.5f, 0.5f);
        g.t(fmt(TN_CONV1D, 0, "weight"), {3, 3, 1, C}, 0.3f);
        for (int i : {2, 5}) g.t(fmt(TN_CONV1D, i, "weight"), {3, 3, 1, C}, 0.3f);
        for (int i : {3, 6}) g.t(fmt(TN_CONV1D, i, "weight"), {1, 1, C, C}, 0.25f);
        for (int i : {0, 2, 3, 5, 6}) g.t(fmt(TN_CONV1D, i, "bias"), {1, 1, C}, 0.02f);
        g.w(fmt(TN_PRE_ENCODE_OUT, "weight"), C * (NM / 8), E);
        g.bias(fmt(TN_PRE_ENCODE_OUT, "bias"), E);
        VitOpts o;
        o.attn_bias = false;
        o.ffn_bias = false;
        vit_layers(g, "a", 2, E, F, NH, o);
        for (int il = 0; il < 2; il++) {
            const char * a = "a";
            g.w(fmt(TN_LINEAR_POS, a, il, "weight"), E, E);
            g.t(fmt(TN_POS_BIAS_U, a, il), {DH, NH}, 0.05f);
            g.t(fmt(TN_POS_BIAS_V, a, il), {DH, NH}, 0.05f);
            g.w(fmt(TN_CONV_PW1, a, il, "weight"), E, 2 * E);
            g.t(fmt(TN_CONV_DW, a, il, "weight"), {K, E}, 0.2f);
            g.norm(fmt(TN_CONV_NORM, a, il, "weight"), E);
            g.bias(fmt(TN_CONV_NORM, a, il, "bias"), E);
            g.t(fmt(TN_CONV_NORM_MEAN, a, il), {E}, 0.05f);
            g.t(fmt(TN_CONV_NORM_VAR, a, il), {E}, 0.2f, 1.0f);
            g.w(fmt(TN_CONV_PW2, a, il, "weight"), E, E);
            g.norm(fmt(TN_FFN_NORM, a, il, "weight"), E);
            g.bias(fmt(TN_FFN_NORM, a, il, "bias"), E);
            g.norm(fmt(TN_FFN_NORM_1, a, il, "weight"), E);
            g.bias(fmt(TN_FFN_NORM_1, a, il, "bias"), E);
            g.w(fmt(TN_FFN_UP_1, a, il, "weight"), E, F);
            g.w(fmt(TN_FFN_DOWN_1, a, il, "weight"), F, E);
            g.norm(fmt(TN_NORM_CONV, a, il, "weight"), E);
            g.bias(fmt(TN_NORM_CONV, a, il, "bias"), E);
        }
        g.norm(fmt(TN_MM_NORM_PRE, "weight"), E);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), E, H);
        g.w(fmt(TN_MM_AUDIO_MLP, 2, "weight"), H, N_TEXT);
        // loaded and required, never read by the graph
        g.w(fmt(TN_MVLM_PROJ_MLP, 0, "weight"), E, H);
        g.w(fmt(TN_MVLM_PROJ_MLP, 1, "weight"), H, H);
        g.w(fmt(TN_MVLM_PROJ_MLP, 3, "weight"), H, N_TEXT);
    }});
    f.push_back({"mimo_audio", "qwen3", "audio", AUD, [](G & g) {
        // whisper-style convs, a rope'd encoder (full / windowed / full) with
        // layer 2's output added back, a stride-2 downsample, a 2-level RVQ
        // whose code embeddings feed a 1-layer local transformer over
        // groups of 2
        const int E = 64, NH = 4, F = 128, NM = 32, H = 128, GS = 2;
        audio(g, "mimo_audio", NM);
        hp(g, "audio", E, F, NH, 3, N_TEXT);
        g.u32(KEY_A_RVQ_NUM_QUANTIZERS, 2);
        g.ai32(KEY_A_RVQ_CODEBOOK_SIZE, {16, 8});
        g.u32(KEY_A_ATTN_WINDOW_SIZE, 8);
        g.ai32(KEY_A_WA_PATTERN_MODE, {-1, 0, -1});
        g.u32(KEY_A_LOCAL_BLOCK_COUNT, 1);
        g.u32(KEY_A_LOCAL_GROUP_SIZE, GS);
        g.t(fmt(TN_CONV1D, 1, "weight"), {3, NM, E}, 0.1f);
        g.t(fmt(TN_CONV1D, 1, "bias"), {1, E}, 0.02f);
        g.t(fmt(TN_CONV1D, 2, "weight"), {3, E, E}, 0.05f);
        g.t(fmt(TN_CONV1D, 2, "bias"), {1, E}, 0.02f);
        vit_layers(g, "a", 3, E, F, NH);
        g.norm(fmt(TN_LN_POST, "a", "weight"), E);
        g.bias(fmt(TN_LN_POST, "a", "bias"), E);
        g.t(fmt(TN_A_DOWNSAMPLE_CONV, "weight"), {2, E, E}, 0.1f);
        g.norm(fmt(TN_A_DOWNSAMPLE_NORM, "weight"), E);
        g.bias(fmt(TN_A_DOWNSAMPLE_NORM, "bias"), E);
        g.t(fmt(TN_A_RVQ_CODEBOOK, "weight"), {E, 16, 2}, 1.0f);
        g.t(fmt(TN_MM_A_CODE_EMBD, "weight"), {E, 16, 2}, 0.1f);
        g.w(fmt(TN_MM_A_LOCAL_ATTN_Q, 0, "weight"), E, E);
        g.bias(fmt(TN_MM_A_LOCAL_ATTN_Q, 0, "bias"), E);
        g.w(fmt(TN_MM_A_LOCAL_ATTN_K, 0, "weight"), E, E);
        g.bias(fmt(TN_MM_A_LOCAL_ATTN_K, 0, "bias"), E);
        g.w(fmt(TN_MM_A_LOCAL_ATTN_V, 0, "weight"), E, E);
        g.bias(fmt(TN_MM_A_LOCAL_ATTN_V, 0, "bias"), E);
        g.w(fmt(TN_MM_A_LOCAL_ATTN_OUT, 0, "weight"), E, E);
        g.w(fmt(TN_MM_A_LOCAL_FFN_GATE, 0, "weight"), E, F);
        g.w(fmt(TN_MM_A_LOCAL_FFN_UP, 0, "weight"), E, F);
        g.w(fmt(TN_MM_A_LOCAL_FFN_DOWN, 0, "weight"), F, E);
        g.norm(fmt(TN_MM_A_LOCAL_LN1, 0, "weight"), E);
        g.norm(fmt(TN_MM_A_LOCAL_LN2, 0, "weight"), E);
        g.norm(fmt(TN_MM_A_LOCAL_NORM, "weight"), E);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), E * GS, H);
        g.w(fmt(TN_MM_AUDIO_MLP, 2, "weight"), H, N_TEXT);
    }});
    f.push_back({"granite_speech", "qwen3", "audio", AUD, [](G & g) {
        // a block-chunked conformer with Shaw relative positions, a CTC
        // side-branch at the middle layer, layer 1's output concatenated to
        // the last, then a windowed Q-Former (window 4, 2 queries)
        const int E = 64, NH = 4, DH = 16, F = 128, NM = 32, K = 9, M = 16, V = 32, CI = 2 * E;
        audio(g, "granite_speech", NM);
        hp(g, "audio", E, F, NH, 2, N_TEXT, 1e-5f);
        g.u32(KEY_A_CHUNK_SIZE, 8);
        g.u32(KEY_A_CONV_KERNEL_SIZE, K);
        g.u32(KEY_A_MAX_POS_EMB, M);
        g.u32(KEY_A_PROJ_WINDOW_SIZE, 4);
        g.u32(KEY_A_PROJ_DOWNSAMPLE_RATE, 2);
        g.u32(KEY_A_PROJ_HEAD_COUNT, 2);
        g.ai32(fmt(KEY_FEATURE_LAYERS, "audio"), {1});
        g.w(fmt(TN_INP_PROJ, "weight"), NM, E);
        g.bias(fmt(TN_INP_PROJ, "bias"), E);
        g.w(fmt(TN_CTC_OUT, "weight"), E, V);
        g.bias(fmt(TN_CTC_OUT, "bias"), V);
        g.w(fmt(TN_CTC_OUT_MID, "weight"), V, E);
        g.bias(fmt(TN_CTC_OUT_MID, "bias"), E);
        VitOpts o;
        o.attn_bias = false;
        vit_layers(g, "a", 2, E, F, NH, o);
        for (int il = 0; il < 2; il++) {
            const char * a = "a";
            g.bias(fmt(TN_ATTN_OUTPUT, a, il, "bias"), E);
            g.t(fmt(TN_ATTN_REL_POS_EMB, a, il), {DH, 2 * M + 1}, 0.1f);
            g.norm(fmt(TN_FFN_NORM, a, il, "weight"), E);
            g.bias(fmt(TN_FFN_NORM, a, il, "bias"), E);
            g.norm(fmt(TN_FFN_NORM_1, a, il, "weight"), E);
            g.bias(fmt(TN_FFN_NORM_1, a, il, "bias"), E);
            g.w(fmt(TN_FFN_UP_1, a, il, "weight"), E, F);
            g.bias(fmt(TN_FFN_UP_1, a, il, "bias"), F);
            g.w(fmt(TN_FFN_DOWN_1, a, il, "weight"), F, E);
            g.bias(fmt(TN_FFN_DOWN_1, a, il, "bias"), E);
            g.norm(fmt(TN_NORM_CONV, a, il, "weight"), E);
            g.bias(fmt(TN_NORM_CONV, a, il, "bias"), E);
            g.norm(fmt(TN_CONV_NORM, a, il, "weight"), E);
            g.bias(fmt(TN_CONV_NORM, a, il, "bias"), E);
            g.t(fmt(TN_CONV_DW, a, il, "weight"), {K, E}, 0.2f);
            g.w(fmt(TN_CONV_PW1, a, il, "weight"), E, 2 * E);
            g.bias(fmt(TN_CONV_PW1, a, il, "bias"), 2 * E);
            g.w(fmt(TN_CONV_PW2, a, il, "weight"), E, E);
            g.bias(fmt(TN_CONV_PW2, a, il, "bias"), E);
        }
        g.t(fmt(TN_QF_PROJ_QUERY, "a"), {E, 2}, 0.5f);
        g.norm(fmt(TN_QF_PROJ_NORM, "a", "weight"), E);
        g.bias(fmt(TN_QF_PROJ_NORM, "a", "bias"), E);
        g.w(fmt(TN_QF_PROJ_LINEAR, "a", "weight"), E, N_TEXT);
        g.bias(fmt(TN_QF_PROJ_LINEAR, "a", "bias"), N_TEXT);
        for (int il = 0; il < 2; il++) {
            auto lin = [&](const char * n, int in, int out) {
                g.w(fmt(n, "a", il, "weight"), in, out);
                g.bias(fmt(n, "a", il, "bias"), out);
            };
            auto nrm = [&](const char * n) {
                g.norm(fmt(n, "a", il, "weight"), E);
                g.bias(fmt(n, "a", il, "bias"), E);
            };
            lin(TN_QF_SELF_ATTN_Q, E, E);
            lin(TN_QF_SELF_ATTN_K, E, E);
            lin(TN_QF_SELF_ATTN_V, E, E);
            lin(TN_QF_SELF_ATTN_O, E, E);
            nrm(TN_QF_SELF_ATTN_N);
            lin(TN_QF_CROSS_ATTN_Q, E, E);
            lin(TN_QF_CROSS_ATTN_K, CI, E);
            lin(TN_QF_CROSS_ATTN_V, CI, E);
            lin(TN_QF_CROSS_ATTN_O, E, E);
            nrm(TN_QF_CROSS_ATTN_N);
            lin(TN_QF_FFN_UP, E, F);
            lin(TN_QF_FFN_DOWN, F, E);
            nrm(TN_QF_FFN_NORM);
        }
    }});
    f.push_back({"gemma3na", "qwen3", "expect-fail", AUD, [](G & g) {
        // declared in clip-impl.h and nowhere else: no load_hparams arm, no
        // load_tensors arm, no graph. load_hparams' shared default throws
        // "unknown vision projector type gemma3na" -- "vision" though this
        // is an audio encoder -- and mtmd_init_from_file returns null
        audio(g, "gemma3na", 32);
        hp(g, "audio", 64, 128, 4, 0, N_TEXT);
    }});
    f.push_back({"gemma4ua", "qwen3", "audio", AUD, [](G & g) {
        // encoder-free: 640-sample frames, an RMS norm and one matmul
        audio(g, "gemma4ua", 640);
        hp(g, "audio", 640, 64, 1, 0, N_TEXT);
        g.w(fmt(TN_A_MM_INP_PROJ, "weight"), 640, N_TEXT);
    }});
    f.push_back({"ultravox", "qwen3", "audio", AUD, [whisper](G & g) {
        const int E = 32, s = 4;
        whisper(g, "ultravox", E);
        g.u32(KEY_A_PROJ_STACK_FACTOR, s);
        g.norm(fmt(TN_MM_NORM_PRE, "weight"), s * E);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), s * E, 2 * 64);
        g.norm(fmt(TN_MM_NORM_MID, "weight"), 64);
        g.w(fmt(TN_MM_AUDIO_MLP, 2, "weight"), 64, N_TEXT);
    }});
    f.push_back({"qwen2a", "qwen3", "audio", AUD, [whisper](G & g) {
        const int E = 32;
        whisper(g, "qwen2a", E);
        g.w(fmt(TN_MM_AUDIO_FC, "weight"), E, N_TEXT);
        g.bias(fmt(TN_MM_AUDIO_FC, "bias"), N_TEXT);
    }});
    f.push_back({"voxtral", "qwen3", "audio", AUD, [whisper](G & g) {
        // stack factor 2, not 4: with avgpool the graph stacks 750 frames
        // and rounds up, where clip_n_output_tokens rounds down (187 vs 188
        // at factor 4) -- an upstream inconsistency, avoided here
        const int E = 32, s = 2;
        whisper(g, "voxtral", E);
        g.u32(KEY_A_PROJ_STACK_FACTOR, s);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), s * E, 64);
        g.w(fmt(TN_MM_AUDIO_MLP, 2, "weight"), 64, N_TEXT);
    }});
    f.push_back({"musicflamingo", "qwen3", "audio", AUD, [whisper](G & g) {
        const int E = 32;
        whisper(g, "musicflamingo", E);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), E, 64);
        g.bias(fmt(TN_MM_AUDIO_MLP, 1, "bias"), 64);
        g.w(fmt(TN_MM_AUDIO_MLP, 2, "weight"), 64, N_TEXT);
        g.bias(fmt(TN_MM_AUDIO_MLP, 2, "bias"), N_TEXT);
    }});
    f.push_back({"meralion", "qwen3", "audio", AUD, [whisper](G & g) {
        const int E = 32, s = 4;
        whisper(g, "meralion", E);
        g.u32(KEY_A_PROJ_STACK_FACTOR, s);
        g.norm(fmt(TN_MM_NORM_PRE, "weight"), s * E);
        g.bias(fmt(TN_MM_NORM_PRE, "bias"), s * E);
        g.w(fmt(TN_MM_AUDIO_MLP, 0, "weight"), s * E, 64);
        g.bias(fmt(TN_MM_AUDIO_MLP, 0, "bias"), 64);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), 64, 64);
        g.bias(fmt(TN_MM_AUDIO_MLP, 1, "bias"), 64);
        g.w(fmt(TN_MM_AUDIO_MLP, 2, "weight"), 64, 64);
        g.bias(fmt(TN_MM_AUDIO_MLP, 2, "bias"), 64);
        g.w(fmt(TN_MM_AUDIO_MLP, 3, "weight"), 64, N_TEXT);
        g.bias(fmt(TN_MM_AUDIO_MLP, 3, "bias"), N_TEXT);
    }});
    f.push_back({"glma", "qwen3", "audio", AUD, [whisper](G & g) {
        const int E = 32, s = 4;
        whisper(g, "glma", E);
        g.u32(KEY_A_PROJ_STACK_FACTOR, s);
        g.norm(fmt(TN_MM_NORM_PRE, "weight"), E);
        g.bias(fmt(TN_MM_NORM_PRE, "bias"), E);
        g.w(fmt(TN_MM_AUDIO_MLP, 1, "weight"), s * E, 64);
        g.bias(fmt(TN_MM_AUDIO_MLP, 1, "bias"), 64);
        g.w(fmt(TN_MM_AUDIO_MLP, 2, "weight"), 64, N_TEXT);
        g.bias(fmt(TN_MM_AUDIO_MLP, 2, "bias"), N_TEXT);
        g.t(TN_TOK_BOI, {N_TEXT, 1}, 0.1f);
        g.t(TN_TOK_EOI, {N_TEXT, 1}, 0.1f);
    }});

    // --- TTS: qwen3-tts (speaker encoder + code predictor / code2wav)
    // `shared`: the talker and the code predictor share a hidden size, so
    // the predictor has no proj_in (load_tensors reads it as optional)
    auto qwen3tts = [](G & g, bool shared) {
        // speaker encoder (audio modality, models/qwen3tts-spkenc.cpp)
        g.b(KEY_HAS_AUDIO_ENC, true);
        g.b(KEY_HAS_GEN_AUDIO_ENC, true);
        g.str(KEY_AUDIO_PROJ_TYPE, "qwen3tts_spkenc");
        g.str(KEY_GEN_AUDIO_PROJ_TYPE, "qwen3tts_gen");
        const int n_mel = 32, C = 32, Cs = C / 8, Cse = 16, Cm = 48, Ca = 16;
        g.u32(KEY_A_NUM_MEL_BINS, n_mel);
        hp(g, "audio", C, 64, 2, 3, N_TEXT);
        g.t(fmt(TN_CONV1D, 0, "weight"), {5, n_mel, C}, 0.1f);
        g.bias(fmt(TN_CONV1D, 0, "bias"), C);
        for (int il = 0; il < 3; il++) {
            const int bid = il + 1;
            g.t(fmt(TN_CONV_PW1, "a", bid, "weight"), {1, C, C}, 0.15f);
            g.bias(fmt(TN_CONV_PW1, "a", bid, "bias"), C);
            g.t(fmt(TN_CONV_PW2, "a", bid, "weight"), {1, C, C}, 0.15f);
            g.bias(fmt(TN_CONV_PW2, "a", bid, "bias"), C);
            g.t(fmt(TN_A_SE_CONV1, bid, "weight"), {1, C, Cse}, 0.15f);
            g.bias(fmt(TN_A_SE_CONV1, bid, "bias"), Cse);
            g.t(fmt(TN_A_SE_CONV2, bid, "weight"), {1, Cse, C}, 0.15f);
            g.bias(fmt(TN_A_SE_CONV2, bid, "bias"), C);
            for (int x = 0; x < 7; x++) {
                g.t(fmt(TN_A_CONV_RES2, bid, x, "weight"), {3, Cs, Cs}, 0.2f);
                g.bias(fmt(TN_A_CONV_RES2, bid, x, "bias"), Cs);
            }
        }
        g.t(fmt(TN_CONV_OUT, "weight"), {1, 3 * C, Cm}, 0.1f);
        g.bias(fmt(TN_CONV_OUT, "bias"), Cm);
        g.t(fmt(TN_A_ASP_TDNN, "weight"), {1, 3 * Cm, Ca}, 0.1f);
        g.bias(fmt(TN_A_ASP_TDNN, "bias"), Ca);
        g.t(fmt(TN_A_ASP_ATTN, "weight"), {1, Ca, Cm}, 0.1f);
        g.bias(fmt(TN_A_ASP_ATTN, "bias"), Cm);
        g.t(fmt(TN_MM_AUDIO_FC, "weight"), {1, 2 * Cm, N_TEXT}, 0.1f);
        g.bias(fmt(TN_MM_AUDIO_FC, "bias"), N_TEXT);

        // code predictor (gen modality, prefix "a.gen.code", models/qwen3tts-gen.cpp)
        const int E = shared ? N_TEXT : 64, H = 2, Hkv = 1, dh = 32, F = 128, V0 = 3968, Vh = 64, NA = 15;
        hp(g, "gen.audio", E, F, H, 2, N_TEXT);
        g.u32(fmt(KEY_N_HEAD_KV, "gen.audio"), Hkv);
        for (int il = 0; il < 2; il++) {
            const char * p = "a.gen.code";
            g.w(fmt(TN_ATTN_Q, p, il, "weight"), E, H * dh);
            g.w(fmt(TN_ATTN_K, p, il, "weight"), E, Hkv * dh);
            g.w(fmt(TN_ATTN_V, p, il, "weight"), E, Hkv * dh);
            g.w(fmt(TN_ATTN_OUTPUT, p, il, "weight"), H * dh, E);
            g.norm(fmt(TN_ATTN_Q_NORM, p, il, "weight"), dh);
            g.norm(fmt(TN_ATTN_K_NORM, p, il, "weight"), dh);
            g.norm(fmt(TN_LN_1, p, il, "weight"), E);
            g.norm(fmt(TN_LN_2, p, il, "weight"), E);
            g.w(fmt(TN_FFN_GATE, p, il, "weight"), E, F);
            g.w(fmt(TN_FFN_UP, p, il, "weight"), E, F);
            g.w(fmt(TN_FFN_DOWN, p, il, "weight"), F, E);
        }
        if (!shared) {
            g.w(fmt(TN_A_GEN_CODE_PROJ_IN, "weight"), N_TEXT, E);
            g.bias(fmt(TN_A_GEN_CODE_PROJ_IN, "bias"), E);
        }
        g.t(fmt(TN_A_GEN_CODE_EMBD, "weight"), {N_TEXT, Vh, NA}, 0.05f);
        g.t(fmt(TN_A_GEN_CODE_HEAD, "weight"), {E, Vh, NA}, 0.3f);
        g.t(fmt(TN_A_GEN_CODE_OUT_EMBD, "weight"), {N_TEXT, V0}, 0.05f);
        g.norm(fmt(TN_A_GEN_CODE_NORM, "weight"), E);

        // code2wav: the dimensions the loader hardcodes are layer and head
        // counts (8 layers, 16 heads, 2 upsample blocks, 4 DAC blocks of 3
        // residual units); every width comes from the tensors
        const int Dc = 16, Hq = 32, C1 = 32, Ht = 32, dht = 2, Ft = 64;
        g.w(fmt(TN_A_GEN_WAV_QUANT_FIRST_IN, "weight"), Hq, Dc);
        g.w(fmt(TN_A_GEN_WAV_QUANT_FIRST_OUT, "weight"), Dc, Hq);
        g.t(fmt(TN_A_GEN_WAV_QUANT_FIRST_CB, "weight"), {Dc, V0}, 0.5f);
        g.w(fmt(TN_A_GEN_WAV_QUANT_REST_IN, "weight"), Hq, Dc);
        g.w(fmt(TN_A_GEN_WAV_QUANT_REST_OUT, "weight"), Dc, Hq);
        g.t(fmt(TN_A_GEN_WAV_QUANT_REST_CB, "weight"), {Dc, Vh, NA}, 0.5f);
        g.t(fmt(TN_A_GEN_WAV_PRE_CONV, "weight"), {3, Hq, C1}, 0.1f);
        g.bias(fmt(TN_A_GEN_WAV_PRE_CONV, "bias"), C1);
        g.w(fmt(TN_A_GEN_WAV_TFM_IN_PROJ, "weight"), C1, Ht);
        g.bias(fmt(TN_A_GEN_WAV_TFM_IN_PROJ, "bias"), Ht);
        for (int il = 0; il < 8; il++) {
            const char * p = "a.gen.wav.tfm";
            g.w(fmt(TN_ATTN_Q, p, il, "weight"), Ht, 16 * dht);
            g.w(fmt(TN_ATTN_K, p, il, "weight"), Ht, 16 * dht);
            g.w(fmt(TN_ATTN_V, p, il, "weight"), Ht, 16 * dht);
            g.w(fmt(TN_ATTN_OUTPUT, p, il, "weight"), 16 * dht, Ht);
            g.norm(fmt(TN_LN_1, p, il, "weight"), Ht);
            g.norm(fmt(TN_LN_2, p, il, "weight"), Ht);
            g.t(fmt(TN_LS_1, p, il, "weight"), {Ht}, 0.05f, 0.1f);
            g.t(fmt(TN_LS_2, p, il, "weight"), {Ht}, 0.05f, 0.1f);
            g.w(fmt(TN_FFN_GATE, p, il, "weight"), Ht, Ft);
            g.w(fmt(TN_FFN_UP, p, il, "weight"), Ht, Ft);
            g.w(fmt(TN_FFN_DOWN, p, il, "weight"), Ft, Ht);
        }
        g.norm(fmt(TN_A_GEN_WAV_TFM_OUT_NORM, "weight"), Ht);
        g.w(fmt(TN_A_GEN_WAV_TFM_OUT_PROJ, "weight"), Ht, C1);
        g.bias(fmt(TN_A_GEN_WAV_TFM_OUT_PROJ, "bias"), C1);
        for (int il = 0; il < 2; il++) {
            g.t(fmt(TN_A_GEN_WAV_UP_CONV, il, "weight"), {2, C1, C1}, 0.15f);
            g.bias(fmt(TN_A_GEN_WAV_UP_CONV, il, "bias"), C1);
            g.t(fmt(TN_A_GEN_WAV_UP_DWCONV, il, "weight"), {7, 1, C1}, 0.2f);
            g.bias(fmt(TN_A_GEN_WAV_UP_DWCONV, il, "bias"), C1);
            g.norm(fmt(TN_A_GEN_WAV_UP_NORM, il, "weight"), C1);
            g.bias(fmt(TN_A_GEN_WAV_UP_NORM, il, "bias"), C1);
            g.w(fmt(TN_A_GEN_WAV_UP_PW1, il, "weight"), C1, 4 * C1);
            g.bias(fmt(TN_A_GEN_WAV_UP_PW1, il, "bias"), 4 * C1);
            g.w(fmt(TN_A_GEN_WAV_UP_PW2, il, "weight"), 4 * C1, C1);
            g.bias(fmt(TN_A_GEN_WAV_UP_PW2, il, "bias"), C1);
            g.t(fmt(TN_A_GEN_WAV_UP_GAMMA, il), {C1}, 0.05f, 0.1f);
        }
        const int Cd[5] = { 32, 32, 16, 16, 8 };
        g.t(fmt(TN_A_GEN_WAV_DAC_ENTRY, "weight"), {7, C1, Cd[0]}, 0.1f);
        g.bias(fmt(TN_A_GEN_WAV_DAC_ENTRY, "bias"), Cd[0]);
        for (int il = 0; il < 4; il++) {
            const int IC = Cd[il], OC = Cd[il + 1];
            g.t(fmt(TN_A_GEN_WAV_DAC_SNAKE, il, "alpha"), {IC}, 0.1f, 1.0f);
            g.t(fmt(TN_A_GEN_WAV_DAC_SNAKE, il, "beta"), {IC}, 0.1f, 1.0f);
            g.t(fmt(TN_A_GEN_WAV_DAC_CONV, il, "weight"), {4, OC, IC}, 0.15f); // kernel 4 = 2 x stride 2
            g.bias(fmt(TN_A_GEN_WAV_DAC_CONV, il, "bias"), OC);
            for (int ir = 0; ir < 3; ir++) {
                g.t(fmt(TN_A_GEN_WAV_DAC_RES_ACT1, il, ir, "alpha"), {OC}, 0.1f, 1.0f);
                g.t(fmt(TN_A_GEN_WAV_DAC_RES_ACT1, il, ir, "beta"), {OC}, 0.1f, 1.0f);
                g.t(fmt(TN_A_GEN_WAV_DAC_RES_CONV1, il, ir, "weight"), {7, OC, OC}, 0.1f);
                g.bias(fmt(TN_A_GEN_WAV_DAC_RES_CONV1, il, ir, "bias"), OC);
                g.t(fmt(TN_A_GEN_WAV_DAC_RES_ACT2, il, ir, "alpha"), {OC}, 0.1f, 1.0f);
                g.t(fmt(TN_A_GEN_WAV_DAC_RES_ACT2, il, ir, "beta"), {OC}, 0.1f, 1.0f);
                g.t(fmt(TN_A_GEN_WAV_DAC_RES_CONV2, il, ir, "weight"), {1, OC, OC}, 0.2f);
                g.bias(fmt(TN_A_GEN_WAV_DAC_RES_CONV2, il, ir, "bias"), OC);
            }
        }
        g.t(fmt(TN_A_GEN_WAV_DAC_POST_SNAKE, "alpha"), {Cd[4]}, 0.1f, 1.0f);
        g.t(fmt(TN_A_GEN_WAV_DAC_POST_SNAKE, "beta"), {Cd[4]}, 0.1f, 1.0f);
        g.t(fmt(TN_A_GEN_WAV_DAC_POST_CONV, "weight"), {7, Cd[4], 1}, 0.2f);
        g.bias(fmt(TN_A_GEN_WAV_DAC_POST_CONV, "bias"), 1);
    };
    f.push_back({"qwen3tts", "qwen3tts", "tts-qwen3", "tts:3", [qwen3tts](G & g) { qwen3tts(g, false); }});

    // --- TTS: pocket-tts (mimi speaker encoder + flow decoder / mimi decoder)
    f.push_back({"pockettts", "pockettts", "tts-pocket", "tts:3", [](G & g) {
        g.b(KEY_HAS_AUDIO_ENC, true);
        g.b(KEY_HAS_GEN_AUDIO_ENC, true);
        g.str(KEY_AUDIO_PROJ_TYPE, "pockettts_spkenc");
        g.str(KEY_GEN_AUDIO_PROJ_TYPE, "pockettts_gen");
        g.u32(KEY_A_NUM_MEL_BINS, 1);
        // speaker path: SEANet encoder (strides 4, 5, 6), a transformer, downsample 16
        const int D = 32, Ft = 64;
        hp(g, "audio", D, Ft, 2, 1, N_TEXT);
        {
            const int ch[4] = { 8, 16, 32, 64 };
            const int ratios[3] = { 4, 5, 6 };
            g.t(fmt(TN_A_SEANET_CONV_IN, "weight"), {7, 1, ch[0]}, 0.3f);
            g.bias(fmt(TN_A_SEANET_CONV_IN, "bias"), ch[0]);
            for (int i = 0; i < 3; i++) {
                const int C = ch[i];
                g.t(fmt(TN_A_SEANET_RES_CONV1, i, "weight"), {3, C, C / 2}, 0.2f);
                g.bias(fmt(TN_A_SEANET_RES_CONV1, i, "bias"), C / 2);
                g.t(fmt(TN_A_SEANET_RES_CONV2, i, "weight"), {1, C / 2, C}, 0.2f);
                g.bias(fmt(TN_A_SEANET_RES_CONV2, i, "bias"), C);
                g.t(fmt(TN_A_SEANET_SCALE_CONV, i, "weight"), {2 * ratios[i], C, ch[i + 1]}, 0.1f);
                g.bias(fmt(TN_A_SEANET_SCALE_CONV, i, "bias"), ch[i + 1]);
            }
            g.t(fmt(TN_A_SEANET_CONV_OUT, "weight"), {3, ch[3], D}, 0.1f);
            g.bias(fmt(TN_A_SEANET_CONV_OUT, "bias"), D);
        }
        VitOpts o;
        o.attn_bias = false;
        o.ffn_bias = false;
        o.ls = true;
        vit_layers(g, "a", 1, D, Ft, 2, o);
        g.t(fmt(TN_A_DOWNSAMPLE_CONV, "weight"), {32, D, D}, 0.05f);
        g.w(fmt(TN_A_SPEAKER_PROJ, "weight"), D, N_TEXT);

        // gen path (prefix "gen.audio"): flow net, then the mimi decoder
        const int L = 16, Cf = 32, nf = 8;
        hp(g, "gen.audio", D, Ft, 2, 1, N_TEXT);
        g.w(fmt(TN_A_GEN_FLOW_INPUT_PROJ, "weight"), L, Cf);
        g.bias(fmt(TN_A_GEN_FLOW_INPUT_PROJ, "bias"), Cf);
        g.w(fmt(TN_A_GEN_FLOW_COND_EMBD, "weight"), N_TEXT, Cf);
        g.bias(fmt(TN_A_GEN_FLOW_COND_EMBD, "bias"), Cf);
        g.w(fmt(TN_A_GEN_FLOW_FINAL_ADA, "weight"), Cf, 2 * Cf);
        g.bias(fmt(TN_A_GEN_FLOW_FINAL_ADA, "bias"), 2 * Cf);
        g.w(fmt(TN_A_GEN_FLOW_FINAL_PROJ, "weight"), Cf, L);
        g.bias(fmt(TN_A_GEN_FLOW_FINAL_PROJ, "bias"), L);
        for (int i = 0; i < 2; i++) {
            g.t(fmt(TN_A_GEN_FLOW_TIME_FREQS, i), {nf}, 0.5f, 0.5f);
            g.w(fmt(TN_A_GEN_FLOW_TIME_UP, i, "weight"), 2 * nf, Cf);
            g.bias(fmt(TN_A_GEN_FLOW_TIME_UP, i, "bias"), Cf);
            g.w(fmt(TN_A_GEN_FLOW_TIME_DOWN, i, "weight"), Cf, Cf);
            g.bias(fmt(TN_A_GEN_FLOW_TIME_DOWN, i, "bias"), Cf);
            g.norm(fmt(TN_A_GEN_FLOW_TIME_NORM, i), Cf);
        }
        for (int il = 0; il < 2; il++) {
            g.norm(fmt(TN_A_GEN_FLOW_BLK_NORM, il, "weight"), Cf);
            g.bias(fmt(TN_A_GEN_FLOW_BLK_NORM, il, "bias"), Cf);
            g.w(fmt(TN_A_GEN_FLOW_BLK_UP, il, "weight"), Cf, Cf);
            g.bias(fmt(TN_A_GEN_FLOW_BLK_UP, il, "bias"), Cf);
            g.w(fmt(TN_A_GEN_FLOW_BLK_DOWN, il, "weight"), Cf, Cf);
            g.bias(fmt(TN_A_GEN_FLOW_BLK_DOWN, il, "bias"), Cf);
            g.w(fmt(TN_A_GEN_FLOW_BLK_ADA, il, "weight"), Cf, 3 * Cf);
            g.bias(fmt(TN_A_GEN_FLOW_BLK_ADA, il, "bias"), 3 * Cf);
        }
        g.w(fmt(TN_A_GEN_OUT_EOS, "weight"), N_TEXT, 1);
        g.bias(fmt(TN_A_GEN_OUT_EOS, "bias"), 1);
        g.w(fmt(TN_A_GEN_INPUT_LINEAR, "weight"), L, N_TEXT);
        g.t(TN_A_GEN_EMB_MEAN, {L}, 0.1f);
        g.t(TN_A_GEN_EMB_STD, {L}, 0.1f, 1.0f);
        g.w(fmt(TN_A_GEN_WAV_QUANT_OUT, "weight"), L, D);
        g.t(fmt(TN_A_GEN_WAV_UPSAMPLE, "weight"), {32, 1, D}, 0.1f);
        {
            // decoder: strides 6, 5, 4 (the encoder's, reversed)
            const int ch[4] = { 64, 32, 16, 8 };
            const int strides[3] = { 6, 5, 4 };
            g.t(fmt(TN_A_GEN_WAV_SEANET_CONV_IN, "weight"), {7, D, ch[0]}, 0.1f);
            g.bias(fmt(TN_A_GEN_WAV_SEANET_CONV_IN, "bias"), ch[0]);
            for (int i = 0; i < 3; i++) {
                const int IC = ch[i], OC = ch[i + 1];
                g.t(fmt(TN_A_GEN_WAV_SEANET_SCALE_CONV, i, "weight"), {2 * strides[i], OC, IC}, 0.1f);
                g.bias(fmt(TN_A_GEN_WAV_SEANET_SCALE_CONV, i, "bias"), OC);
                g.t(fmt(TN_A_GEN_WAV_SEANET_RES_CONV1, i, "weight"), {3, OC, OC / 2}, 0.2f);
                g.bias(fmt(TN_A_GEN_WAV_SEANET_RES_CONV1, i, "bias"), OC / 2);
                g.t(fmt(TN_A_GEN_WAV_SEANET_RES_CONV2, i, "weight"), {1, OC / 2, OC}, 0.2f);
                g.bias(fmt(TN_A_GEN_WAV_SEANET_RES_CONV2, i, "bias"), OC);
            }
            g.t(fmt(TN_A_GEN_WAV_SEANET_CONV_OUT, "weight"), {3, ch[3], 1}, 0.2f);
            g.bias(fmt(TN_A_GEN_WAV_SEANET_CONV_OUT, "bias"), 1);
        }
        for (int il = 0; il < 1; il++) {
            const char * p = "a.gen.wav.tfm";
            g.norm(fmt(TN_LN_1, p, il, "weight"), D);
            g.bias(fmt(TN_LN_1, p, il, "bias"), D);
            g.w(fmt(TN_ATTN_Q, p, il, "weight"), D, D);
            g.w(fmt(TN_ATTN_K, p, il, "weight"), D, D);
            g.w(fmt(TN_ATTN_V, p, il, "weight"), D, D);
            g.w(fmt(TN_ATTN_OUTPUT, p, il, "weight"), D, D);
            g.t(fmt(TN_LS_1, p, il, "weight"), {D}, 0.05f, 0.1f);
            g.norm(fmt(TN_LN_2, p, il, "weight"), D);
            g.bias(fmt(TN_LN_2, p, il, "bias"), D);
            g.w(fmt(TN_FFN_UP, p, il, "weight"), D, Ft);
            g.w(fmt(TN_FFN_DOWN, p, il, "weight"), Ft, D);
            g.t(fmt(TN_LS_2, p, il, "weight"), {D}, 0.05f, 0.1f);
        }
    }});

    (void) IMG;
    // --- added at v0.4.0; last, so every earlier fixture keeps its seed
    // deepseek4v: an RMS-norm SwiGLU ViT with 2D M-RoPE, a 3x3 unfold
    // aligner, and the four sentinel embeddings get_rows lays out. The
    // sizes cover the min-pixel upscale (16x16), padding to the merge
    // (100x60), the extreme-aspect stretch (400x40) and the resize solver
    // (480x480, whose 20x20 aligner grid is over the 384-token cap).
    f.push_back({"deepseek4v", "qwen3", "vision", "img:64x64,100x60,16x16,400x40,480x480", [](G & g) {
        const int E = 32, P = 8, M = 3, H = 64;
        vision(g, "deepseek4v", 64, P, false);
        g.b(KEY_USE_SILU, true);
        hp(g, "vision", E, 64, 2, 2, N_TEXT);
        g.u32(KEY_PROJ_SCALE_FACTOR, M);
        g.u32(KEY_IMAGE_MIN_PIXELS, P * P * M * M * 4);
        patch_embd(g, P, E, 0, false);
        VitOpts o;
        o.ln_bias = false;
        o.gated = true;
        o.attn_bias = false;
        o.ffn_bias = false;
        vit_layers(g, "v", 2, E, 64, 2, o);
        g.w(fmt(TN_LLAVA_PROJ, 1, "weight"), E * M * M, H);
        g.bias(fmt(TN_LLAVA_PROJ, 1, "bias"), H);
        g.w(fmt(TN_LLAVA_PROJ, 2, "weight"), H, N_TEXT);
        g.bias(fmt(TN_LLAVA_PROJ, 2, "bias"), N_TEXT);
        g.t(TN_IMAGE_NEWLINE, {N_TEXT}, 0.1f);
        g.t(TN_TOK_IMG_START, {N_TEXT}, 0.1f);
        g.t(TN_TOK_IMG_END, {N_TEXT}, 0.1f);
        g.t(TN_TOK_IMG_PAD, {N_TEXT}, 0.1f);
    }});
    f.push_back({"qwen3tts_shared", "qwen3tts", "tts-qwen3", "tts:3", [qwen3tts](G & g) { qwen3tts(g, true); }});

    return f;
}

// -----------------------------------------------------------------------------
// text models: an arch fixture with an SPM vocabulary

static const std::vector<std::string> k_specials = {
    // mtmd.cpp's lookup_token
    "<image>", "</image>", "<slice>", "</slice>", "<im_start>", "<im_end>",
    "<patch_start>", "<patch_end>", "<patch_newline>", "<fake_token_around_image>",
    "<global-img>", "<|img_thumbnail|>", "\n\n",
    // mtmd.cpp's img_beg/img_end for gemma3, so the media markers tokenize as
    // special tokens on one fixture: the 128-token vocabulary has room for no
    // more, and every other marker tokenizes as characters either way
    "<start_of_image>", "<end_of_image>",
    // markers outside ASCII: the vocabulary's only byte token is <0x0A>, so
    // the SPM byte fallback would throw on them
    "<\xef\xbd\x9chy_place\xe2\x96\x81holder\xe2\x96\x81no\xe2\x96\x81" "100\xef\xbd\x9c>",
    "<\xef\xbd\x9chy_place\xe2\x96\x81holder\xe2\x96\x81no\xe2\x96\x81" "101\xef\xbd\x9c>",
    // mtmd-helper-gen.cpp: qwen3-tts and pocket-tts
    "<|codec_bos|>", "<|codec_eos_token|>", "<|codec_pad|>", "<|codec_think|>",
    "<|codec_think_bos|>", "<|codec_think_eos|>", "<tts_pad>", "<tts_text_bos>",
    "<tts_text_eod>", "<|codec_language_english|>", "<|bos_before_voice|>", "<|audio_bos|>",
};

static bool write_text_model(const std::string & in, const std::string & out, bool codec, int deepstack = 0, bool mrope = false) {
    ggml_context * ctx = nullptr;
    gguf_init_params ip = { false, &ctx };
    gguf_context * src = gguf_init_from_file(in.c_str(), ip);
    if (!src) {
        fprintf(stderr, "cannot read %s\n", in.c_str());
        return false;
    }
    const char * arch = gguf_get_val_str(src, gguf_find_key(src, "general.architecture"));
    const int64_t kv = gguf_find_key(src, (std::string(arch) + ".vocab_size").c_str());
    const int N = kv >= 0 ? (int) gguf_get_val_u32(src, kv) : 128;

    std::vector<std::string> text;
    std::vector<int32_t> types;
    auto add = [&](const std::string & s, int t) { text.push_back(s); types.push_back(t); };
    add("<unk>", 2);
    add("<s>", 3);
    add("</s>", 3);
    add("\xe2\x96\x81", 1);
    for (int ch = 0x21; ch <= 0x7e; ch++) add(std::string(1, (char) ch), 1);
    add("<0x0A>", 6);
    for (const auto & s : k_specials) add(s, 3);
    if (codec) {
        // qwen3-tts samples codec_0 + code: the rest of the vocabulary
        int i = 0;
        while ((int) text.size() < N) add(fmt("<|codec_%d|>", i++), 3);
    }
    while ((int) text.size() < N) add(fmt("<unused%d>", (int) text.size()), 5);
    if ((int) text.size() > N) {
        fprintf(stderr, "%s: vocabulary of %d cannot hold %zu tokens\n", in.c_str(), N, text.size());
        return false;
    }
    std::vector<const char *> ptrs;
    std::vector<float> scores;
    for (size_t i = 0; i < text.size(); i++) {
        ptrs.push_back(text[i].c_str());
        scores.push_back(-(float) i);
    }

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    gguf_set_val_str(dst, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(dst, "tokenizer.ggml.tokens", ptrs.data(), ptrs.size());
    gguf_set_arr_data(dst, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(dst, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(dst, "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(dst, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(dst, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_bool(dst, "tokenizer.ggml.add_bos_token", true);
    gguf_set_val_bool(dst, "tokenizer.ggml.add_eos_token", false);
    if (deepstack > 0) gguf_set_val_u32(dst, (std::string(arch) + ".n_deepstack_layers").c_str(), deepstack);
    if (mrope) {
        // hunyuan_vl decodes with M-RoPE only when its sections are set
        const int32_t sections[4] = { 4, 4, 4, 4 };
        gguf_set_arr_data(dst, (std::string(arch) + ".rope.dimension_sections").c_str(), GGUF_TYPE_INT32, sections, 4);
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(src); i++) {
        gguf_add_tensor(dst, ggml_get_tensor(ctx, gguf_get_tensor_name(src, i)));
    }
    const bool ok = gguf_write_to_file(dst, out.c_str(), false);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(ctx);
    return ok;
}

static int cmd_gen(const std::string & arch_dir, const std::string & out_dir, const std::string & filter) {
    struct Text { const char * key; const char * fixture; bool codec; int deepstack; bool mrope = false; };
    const Text texts[] = {
        { "qwen3", "qwen3-dense", false, 0 },
        { "qwen2vl", "qwen2vl-dense", false, 0 },
        { "qwen3vl", "qwen3vl-dense", false, 0 },
        { "qwen3vl-ds1", "qwen3vl-dense", false, 1 },
        { "hunyuanvl", "hunyuan_vl-dense", false, 0, true },
        { "qwen3tts", "qwen3tts-dense", true, 0 },
        { "pockettts", "pockettts-dense", false, 0 },
    };
    for (const auto & t : texts) {
        const std::string out = out_dir + "/text-" + t.key + ".gguf";
        if (!write_text_model(arch_dir + "/" + t.fixture + ".gguf", out, t.codec, t.deepstack, t.mrope)) return 1;
    }

    FILE * man = fopen((out_dir + "/manifest.txt").c_str(), "w");
    if (!man) {
        fprintf(stderr, "cannot write manifest\n");
        return 1;
    }
    uint32_t seed = 1;
    for (const auto & fx : fixtures()) {
        seed++;
        if (!filter.empty() && fx.type.find(filter) == std::string::npos) continue;
        G g(fx.type, seed);
        fx.build(g);
        const std::string path = out_dir + "/" + fx.type + ".gguf";
        if (!g.write(path)) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            return 1;
        }
        fprintf(man, "%s\t%s\t%s\t%s\t%s\n", fx.type.c_str(), path.c_str(),
                (out_dir + "/text-" + fx.text + ".gguf").c_str(), fx.modality.c_str(), fx.inputs.c_str());
        printf("wrote %s\n", path.c_str());
    }
    fclose(man);
    return 0;
}

// -----------------------------------------------------------------------------
// validation through upstream's libmtmd

static int g_quiet = 1;
static void log_cb(enum ggml_log_level level, const char * text, void *) {
    if (!g_quiet || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static bool check_embd(const float * e, size_t n, std::string & why) {
    if (!e || n == 0) {
        why = "no output";
        return false;
    }
    double s = 0;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(e[i])) {
            why = fmt("non-finite at %zu", i);
            return false;
        }
        s += std::fabs(e[i]);
    }
    if (s == 0) {
        why = "all zero";
        return false;
    }
    return true;
}

static std::vector<std::string> split(const std::string & s, char d) {
    std::vector<std::string> out;
    size_t p = 0, q;
    while ((q = s.find(d, p)) != std::string::npos) {
        out.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    out.push_back(s.substr(p));
    return out;
}

// encode every media chunk of one prompt; returns "" or the failure
static std::string run_media(mtmd_context * mctx, llama_model * model, std::vector<mtmd_bitmap *> & bitmaps) {
    std::string prompt = "describe:";
    for (size_t i = 0; i < bitmaps.size(); i++) prompt += std::string(" ") + mtmd_default_marker();
    prompt += " ok";
    mtmd_input_text text = { prompt.c_str(), prompt.size(), true, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    std::vector<const mtmd_bitmap *> bp(bitmaps.begin(), bitmaps.end());
    if (mtmd_tokenize(mctx, chunks, &text, bp.data(), bp.size()) != 0) {
        mtmd_input_chunks_free(chunks);
        return "tokenize failed";
    }
    const int n_embd = llama_model_n_embd_inp(model);
    std::string err;
    int n_media = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
        n_media++;
        if (mtmd_encode_chunk(mctx, ch) != 0) {
            err = fmt("encode failed on chunk %zu", i);
            break;
        }
        std::string why;
        const size_t n = (size_t) n_embd * mtmd_input_chunk_get_n_tokens(ch);
        if (!check_embd(mtmd_get_output_embd(mctx), n, why)) {
            err = fmt("chunk %zu: %s", i, why.c_str());
            break;
        }
    }
    if (err.empty() && n_media == 0) err = "no media chunk";
    mtmd_input_chunks_free(chunks);
    return err;
}

static std::string run_tts(mtmd_context * mctx, llama_model * model, bool qwen, int steps) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.embeddings = true;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) return "context failed";
    std::string err;
    mtmd_helper_gen_audio * gen = mtmd_helper_gen_audio_init(lctx, mctx);
    if (!gen) {
        llama_free(lctx);
        return "gen init failed";
    }
    // a speaker reference through the audio encoder
    std::vector<float> spk(24000);
    for (size_t i = 0; i < spk.size(); i++) spk[i] = 0.3f * std::sin(0.03f * i) + 0.1f * std::sin(0.31f * i);
    mtmd_bitmap * spk_bm = mtmd_bitmap_init_from_audio(spk.size(), spk.data());
    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id = 0;
    inp.prompt = "hello there, test.";
    inp.prompt_len = strlen(inp.prompt);
    inp.speaker_ref = spk_bm;
    inp.lang = "";
    inp.top_k = 8;
    inp.top_p = 1.0f;
    inp.seed = 42;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    if (mtmd_helper_gen_audio_set_input(gen, &inp) != 0) err = "set_input failed";
    while (err.empty()) {
        int32_t r = mtmd_helper_gen_audio_step_prompt(gen, 64);
        if (r < 0) err = "step_prompt failed";
        if (r <= 0) break;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token codec_0 = LLAMA_TOKEN_NULL;
    for (llama_token t = 0; t < llama_vocab_n_tokens(vocab) && qwen; t++) {
        if (std::string(llama_vocab_get_text(vocab, t)) == "<|codec_0|>") { codec_0 = t; break; }
    }
    const float * h = llama_get_embeddings_ith(lctx, -1);
    for (int s = 0; err.empty() && s < steps; s++) {
        // the backbone's token: a codec token for qwen3-tts, none for pocket-tts
        const llama_token sampled = qwen ? codec_0 + 7 * (s + 1) : LLAMA_TOKEN_NULL;
        const float * h_next = nullptr;
        bool stop = false;
        if (mtmd_helper_gen_audio_step_gen(gen, sampled, h, &h_next, &stop) != 0) {
            err = fmt("step_gen failed at %d", s);
            break;
        }
        if (!h_next) break;
        h = h_next;
    }
    if (err.empty()) {
        int32_t sr = 0;
        const char * data = nullptr;
        size_t len = 0;
        int64_t n_samples = 0;
        if (mtmd_helper_gen_audio_get_output(gen, &sr, &data, &len, &n_samples) != 0) {
            err = "get_output failed";
        } else if (len <= 44 || n_samples <= 0 || sr <= 0) {
            err = fmt("empty output (len %zu, samples %lld, sr %d)", len, (long long) n_samples, sr);
        } else {
            // 16-bit PCM after a 44-byte header: not silent
            const int16_t * pcm = (const int16_t *) (data + 44);
            int64_t nz = 0;
            for (int64_t i = 0; i < (int64_t) ((len - 44) / 2); i++) nz += pcm[i] != 0;
            if (nz == 0) err = "silent output";
        }
    }
    mtmd_bitmap_free(spk_bm);
    mtmd_helper_gen_audio_free(gen);
    llama_free(lctx);
    return err;
}

static std::string check_one(const std::vector<std::string> & m, bool gpu) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu ? 99 : 0;
    llama_model * model = llama_model_load_from_file(m[2].c_str(), mp);
    if (!model) return "text model load failed";
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.n_threads = 4;
    cp.warmup = true;
    mtmd_context * mctx = mtmd_init_from_file(m[1].c_str(), model, cp);
    if (!mctx) {
        llama_model_free(model);
        return "mtmd_init_from_file failed";
    }
    std::string err;
    const std::string & modality = m[3];
    std::mt19937 rng(7);
    for (const auto & item : split(m[4], ';')) {
        if (!err.empty()) break;
        const auto kv = split(item, ':');
        if (kv.size() != 2) continue;
        if (kv[0] == "img") {
            for (const auto & sz : split(kv[1], ',')) {
                const auto wh = split(sz, 'x');
                const int W = atoi(wh[0].c_str()), H = atoi(wh[1].c_str());
                std::vector<unsigned char> px((size_t) W * H * 3);
                for (auto & p : px) p = (unsigned char) (rng() & 0xff);
                std::vector<mtmd_bitmap *> bms = { mtmd_bitmap_init(W, H, px.data()) };
                err = run_media(mctx, model, bms);
                mtmd_bitmap_free(bms[0]);
                if (!err.empty()) {
                    err = sz + ": " + err;
                    break;
                }
            }
        } else if (kv[0] == "aud") {
            for (const auto & ns : split(kv[1], ',')) {
                const int N = atoi(ns.c_str());
                std::vector<float> pcm(N);
                for (int i = 0; i < N; i++) pcm[i] = 0.4f * std::sin(0.05f * i) + 0.05f * ((rng() % 1000) / 500.0f - 1.0f);
                std::vector<mtmd_bitmap *> bms = { mtmd_bitmap_init_from_audio(pcm.size(), pcm.data()) };
                err = run_media(mctx, model, bms);
                mtmd_bitmap_free(bms[0]);
                if (!err.empty()) {
                    err = "aud " + ns + ": " + err;
                    break;
                }
            }
        } else if (kv[0] == "tts") {
            err = run_tts(mctx, model, modality == "tts-qwen3", atoi(kv[1].c_str()));
        }
    }
    mtmd_free(mctx);
    llama_model_free(model);
    return err;
}

static void init_logs() {
    llama_log_set(log_cb, nullptr);
    mtmd_log_set(log_cb, nullptr);
    mtmd_helper_log_set(log_cb, nullptr);
    if (getenv("MMPROJ_VERBOSE")) g_quiet = 0;
    llama_backend_init();
}

// `one`: a single manifest line on one device, in this process. `check`
// runs each in a child so an upstream abort or crash is a result, not the
// end of the run; the last line is `RESULT ok` or `RESULT <why>`.
static int cmd_one(const std::vector<std::string> & m, bool gpu) {
    init_logs();
    const std::string err = check_one(m, gpu);
    printf("RESULT %s\n", err.empty() ? "ok" : err.c_str());
    fflush(stdout);
    return err.empty() ? 0 : 1;
}

static std::string g_self;

static std::string run_child(const std::vector<std::string> & m, bool gpu) {
    std::string cmd = "'" + g_self + "' one";
    for (size_t i = 0; i < 5; i++) cmd += " '" + m[i] + "'";
    cmd += gpu ? " gpu" : " cpu";
    cmd += " 2>&1";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) return "cannot start child";
    char buf[8192];
    std::string result, why;
    while (fgets(buf, sizeof buf, p)) {
        std::string l(buf);
        if (l.rfind("RESULT ", 0) == 0) {
            result = l.substr(7);
            while (!result.empty() && result.back() == '\n') result.pop_back();
            continue;
        }
        if (!g_quiet) fputs(buf, stderr);
        if (l.find("GGML_ASSERT") != std::string::npos || l.find("GGML_ABORT") != std::string::npos ||
            l.find(".cpp:") != std::string::npos || l.find("error") != std::string::npos) {
            why = l;
            while (!why.empty() && why.back() == '\n') why.pop_back();
        }
    }
    const int st = pclose(p);
    if (!result.empty()) return result == "ok" ? "" : result;
    if (WIFSIGNALED(st)) return fmt("crashed (signal %d): %s", WTERMSIG(st), why.c_str());
    return fmt("died (status %d): %s", WIFEXITED(st) ? WEXITSTATUS(st) : -1, why.c_str());
}

static int cmd_check(const std::string & manifest, const std::string & devs, const std::string & filter) {
    if (getenv("MMPROJ_VERBOSE")) g_quiet = 0;
    FILE * f = fopen(manifest.c_str(), "r");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", manifest.c_str());
        return 2;
    }
    char line[4096];
    int n = 0, n_fail = 0;
    while (fgets(line, sizeof line, f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;
        const auto m = split(s, '\t');
        if (m.size() < 5) continue;
        if (!filter.empty() && m[0].find(filter) == std::string::npos) continue;
        for (int pass = 0; pass < 2; pass++) {
            const bool gpu = pass == 1;
            if ((gpu && devs == "cpu") || (!gpu && devs == "gpu")) continue;
            n++;
            std::string err = run_child(m, gpu);
            if (m[3] == "expect-fail") {
                // a type upstream declares and cannot load: it must fail
                err = err.empty() ? "loaded, but upstream has no implementation to load it with" : "";
            }
            if (!err.empty()) n_fail++;
            printf("%-4s %-18s %s  %s\n", err.empty() ? "ok" : "FAIL", m[0].c_str(), gpu ? "gpu" : "cpu",
                   m[3] == "expect-fail" && err.empty() ? "(fails, as expected)" : err.c_str());
            fflush(stdout);
        }
    }
    fclose(f);
    printf("%s: %d of %d checks passed\n", n_fail ? "FAIL" : "PASS", n - n_fail, n);
    return n_fail ? 1 : 0;
}

int main(int argc, char ** argv) {
    g_self = argv[0];
    if (argc >= 8 && std::string(argv[1]) == "one") {
        return cmd_one({argv[2], argv[3], argv[4], argv[5], argv[6]}, std::string(argv[7]) == "gpu");
    }
    if (argc >= 4 && std::string(argv[1]) == "gen") return cmd_gen(argv[2], argv[3], argc > 4 ? argv[4] : "");
    if (argc >= 3 && std::string(argv[1]) == "check") {
        return cmd_check(argv[2], argc > 3 ? argv[3] : "both", argc > 4 ? argv[4] : "");
    }
    fprintf(stderr, "usage: %s gen <arch-fixtures-dir> <out-dir> [type] | check <manifest> [cpu|gpu|both] [type]\n", argv[0]);
    return 2;
}
