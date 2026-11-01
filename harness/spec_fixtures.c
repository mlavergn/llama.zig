// Not a port. Rewrites an arch fixture with a vocabulary, for
// `scripts/spec-diff`.
//
// The arch fixtures carry `tokenizer.ggml.model = no_vocab`, and
// llama-vocab.cpp returns before it reads any special token for that type,
// so a DFlash or DSpark draft has no mask token: every noise block decodes
// token -1 and fails, on both sides, and the drafters' reading of a block
// never runs. This copies the fixture -- every key and tensor -- and gives
// it an SPM vocabulary of the same size: `<unk>`, `<s>`, `</s>`, then
// `<tN>`, every score zero, with the last token as the mask.
//
// With --gemma4 it instead writes a Gemma 4 target with experts and its
// two-layer assistant, the pair `harness/gemma4_assistant_diff.zig`
// writes (its `writeTarget` and `writeAssistant`, the same metadata and
// tensors), for the MTP drafter's shared-KV mode; the data here is a
// fixed LCG rather than that harness's, which only both sides' agreeing
// needs.
//
// Usage: spec_fixtures <in.gguf> <out.gguf>
//        spec_fixtures --gemma4 <dir>

#include "gguf.h"
#include "ggml.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t lcg = 12345;

static void add(struct ggml_context * g, struct gguf_context * m, const char * name, int n_dims, int64_t a, int64_t b, int64_t c3) {
    const int64_t ne[3] = { a, b, c3 };
    struct ggml_tensor * t = ggml_new_tensor(g, GGML_TYPE_F32, n_dims, ne);
    ggml_set_name(t, name);
    const int norm = strstr(name, "norm") || strstr(name, "scale") || strstr(name, "rope_freqs");
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        lcg = lcg * 1103515245u + 12345u;
        const float r = (float) ((lcg >> 8) & 0xffff) / 65536.0f - 0.5f;
        d[i] = norm ? 1.0f + r * 0.2f : r * 0.5f;
    }
    gguf_add_tensor(m, t);
}

static void common_meta(struct gguf_context * g, const char * a) {
    char k[128];
#define U32(key, v) do { snprintf(k, sizeof(k), "%s." key, a); gguf_set_val_u32(g, k, v); } while (0)
#define F32(key, v) do { snprintf(k, sizeof(k), "%s." key, a); gguf_set_val_f32(g, k, v); } while (0)
    gguf_set_val_str(g, "general.architecture", a);
    gguf_set_val_str(g, "tokenizer.ggml.model", "no_vocab");
    U32("vocab_size", 128);
    U32("context_length", 256);
    U32("attention.head_count", 2);
    U32("attention.head_count_kv", 1);
    U32("attention.key_length", 64);
    U32("attention.value_length", 64);
    U32("attention.key_length_swa", 32);
    U32("attention.value_length_swa", 32);
    U32("rope.dimension_count", 64);
    U32("rope.dimension_count_swa", 32);
    F32("rope.freq_base", 1000000.0f);
    F32("rope.freq_base_swa", 10000.0f);
    F32("attention.layer_norm_rms_epsilon", 1e-6f);
    U32("attention.sliding_window", 16);
}

#define MU32(key, v) do { snprintf(k, sizeof(k), "%s." key, a); gguf_set_val_u32(m, k, v); } while (0)
#define MF32(key, v) do { snprintf(k, sizeof(k), "%s." key, a); gguf_set_val_f32(m, k, v); } while (0)

static int write_gemma4(const char * dir) {
    char path[1024], n[128], k[128];
    // the target: six layers, two reading earlier layers' KV, experts on 1, 3 and 5
    {
        struct gguf_context * m = gguf_init_empty();
        const char * a = "gemma4";
        common_meta(m, a);
        MU32("embedding_length", 128);
        MU32("feed_forward_length", 192);
        MU32("block_count", 6);
        MU32("attention.shared_kv_layers", 2);
        MU32("embedding_length_per_layer_input", 32);
        MF32("final_logit_softcapping", 30.0f);
        MU32("expert_count", 4);
        MU32("expert_used_count", 2);
        MU32("expert_feed_forward_length", 64);
        const bool swa[6] = { true, true, true, false, true, false };
        snprintf(k, sizeof(k), "%s.attention.sliding_window_pattern", a);
        gguf_set_arr_data(m, k, GGUF_TYPE_BOOL, swa, 6);
        struct ggml_init_params gp = { 32u << 20, NULL, false };
        struct ggml_context * g = ggml_init(gp);
        add(g, m, "token_embd.weight", 2, 128, 128, 1);
        add(g, m, "output_norm.weight", 1, 128, 1, 1);
        add(g, m, "per_layer_token_embd.weight", 2, 32 * 6, 128, 1);
        add(g, m, "per_layer_model_proj.weight", 2, 128, 32 * 6, 1);
        add(g, m, "per_layer_proj_norm.weight", 1, 32, 1, 1);
        bool rope_done = false;
        for (int il = 0; il < 6; il++) {
            const int64_t hd = swa[il] ? 32 : 64;
            const bool has_kv = il < 4;
#define T(fmt, ...) do { snprintf(n, sizeof(n), "blk.%d." fmt, il); add(g, m, n, __VA_ARGS__); } while (0)
            T("attn_norm.weight", 1, 128, 1, 1);
            T("attn_q.weight", 2, 128, hd * 2, 1);
            if (has_kv) T("attn_k.weight", 2, 128, hd, 1);
            if (il == 0 || il == 3) T("attn_v.weight", 2, 128, hd, 1);
            T("attn_output.weight", 2, hd * 2, 128, 1);
            T("attn_q_norm.weight", 1, hd, 1, 1);
            if (has_kv) T("attn_k_norm.weight", 1, hd, 1, 1);
            T("post_attention_norm.weight", 1, 128, 1, 1);
            if (il == 1 || il == 4) T("layer_output_scale.weight", 1, 1, 1, 1);
            if (!swa[il] && !rope_done) {
                add(g, m, "rope_freqs.weight", 1, hd / 2, 1, 1);
                rope_done = true;
            }
            T("ffn_norm.weight", 1, 128, 1, 1);
            T("ffn_gate.weight", 2, 128, 192, 1);
            T("ffn_up.weight", 2, 128, 192, 1);
            T("ffn_down.weight", 2, 192, 128, 1);
            T("post_ffw_norm.weight", 1, 128, 1, 1);
            if (il % 2 == 1) {
                T("ffn_gate_inp.weight", 2, 128, 4, 1);
                T("ffn_gate_inp.scale", 1, 128, 1, 1);
                T("pre_ffw_norm_2.weight", 1, 128, 1, 1);
                T("post_ffw_norm_1.weight", 1, 128, 1, 1);
                T("post_ffw_norm_2.weight", 1, 128, 1, 1);
                if (il == 1) {
                    T("ffn_gate_up_exps.weight", 3, 128, 64 * 2, 4);
                } else {
                    T("ffn_gate_exps.weight", 3, 128, 64, 4);
                    T("ffn_up_exps.weight", 3, 128, 64, 4);
                }
                T("ffn_down_exps.weight", 3, 64, 128, 4);
            }
            T("inp_gate.weight", 2, 128, 32, 1);
            T("proj.weight", 2, 32, 128, 1);
            T("post_norm.weight", 1, 128, 1, 1);
        }
        snprintf(path, sizeof(path), "%s/gemma4-moe.gguf", dir);
        if (!gguf_write_to_file(m, path, false)) return 1;
        ggml_free(g);
        gguf_free(m);
    }
    // the assistant: two next-token layers, SWA then full
    {
        struct gguf_context * m = gguf_init_empty();
        const char * a = "gemma4-assistant";
        common_meta(m, a);
        MU32("embedding_length", 64);
        MU32("embedding_length_out", 128);
        MU32("feed_forward_length", 96);
        MU32("block_count", 2);
        MU32("nextn_predict_layers", 2);
        const bool swa[2] = { true, false };
        snprintf(k, sizeof(k), "%s.attention.sliding_window_pattern", a);
        gguf_set_arr_data(m, k, GGUF_TYPE_BOOL, swa, 2);
        struct ggml_init_params gp = { 8u << 20, NULL, false };
        struct ggml_context * g = ggml_init(gp);
        add(g, m, "token_embd.weight", 2, 64, 128, 1);
        add(g, m, "output_norm.weight", 1, 64, 1, 1);
        add(g, m, "nextn.post_projection.weight", 2, 64, 128, 1);
        add(g, m, "nextn.pre_projection.weight", 2, 2 * 128, 64, 1);
        for (int il = 0; il < 2; il++) {
            const int64_t hd = il == 0 ? 32 : 64;
            T("attn_norm.weight", 1, 64, 1, 1);
            T("attn_q.weight", 2, 64, hd * 2, 1);
            T("attn_output.weight", 2, hd * 2, 64, 1);
            T("attn_q_norm.weight", 1, hd, 1, 1);
            T("post_attention_norm.weight", 1, 64, 1, 1);
            T("layer_output_scale.weight", 1, 1, 1, 1);
            if (il == 1) add(g, m, "rope_freqs.weight", 1, hd / 2, 1, 1);
            T("ffn_norm.weight", 1, 64, 1, 1);
            T("ffn_gate.weight", 2, 64, 96, 1);
            T("ffn_up.weight", 2, 64, 96, 1);
            T("ffn_down.weight", 2, 96, 64, 1);
            T("post_ffw_norm.weight", 1, 64, 1, 1);
        }
        snprintf(path, sizeof(path), "%s/gemma4-assistant.gguf", dir);
        if (!gguf_write_to_file(m, path, false)) return 1;
        ggml_free(g);
        gguf_free(m);
    }
    printf("%s: gemma4-moe.gguf, gemma4-assistant.gguf\n", dir);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--gemma4") == 0) return write_gemma4(argv[2]);
    if (argc != 3) {
        fprintf(stderr, "usage: %s in.gguf out.gguf\n", argv[0]);
        return 2;
    }
    struct ggml_context * data = NULL;
    struct gguf_init_params ip = { /*.no_alloc =*/ false, /*.ctx =*/ &data };
    struct gguf_context * in = gguf_init_from_file(argv[1], ip);
    if (!in) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }

    const char * arch = gguf_get_val_str(in, gguf_find_key(in, "general.architecture"));
    char key[256];
    snprintf(key, sizeof(key), "%s.vocab_size", arch);
    int64_t kid = gguf_find_key(in, key);
    uint32_t n_vocab = 0;
    if (kid >= 0) {
        n_vocab = gguf_get_val_u32(in, kid);
    } else {
        // the output projection's rows
        struct ggml_tensor * out = ggml_get_tensor(data, "output.weight");
        if (!out) out = ggml_get_tensor(data, "token_embd.weight");
        n_vocab = out ? (uint32_t) out->ne[1] : 0;
    }
    if (n_vocab < 8) {
        fprintf(stderr, "%s: no vocabulary size\n", argv[1]);
        return 1;
    }

    struct gguf_context * o = gguf_init_empty();
    gguf_set_kv(o, in);
    gguf_set_val_str(o, "tokenizer.ggml.model", "llama");
    gguf_remove_key(o, "tokenizer.ggml.pre");

    char ** toks = (char **) calloc(n_vocab, sizeof(char *));
    float * scores = (float *) calloc(n_vocab, sizeof(float));
    int32_t * types = (int32_t *) calloc(n_vocab, sizeof(int32_t));
    for (uint32_t i = 0; i < n_vocab; i++) {
        toks[i] = (char *) malloc(16);
        if (i == 0) strcpy(toks[i], "<unk>");
        else if (i == 1) strcpy(toks[i], "<s>");
        else if (i == 2) strcpy(toks[i], "</s>");
        else snprintf(toks[i], 16, "<t%u>", i);
        types[i] = i < 3 ? 3 : (i == n_vocab - 1 ? 4 : 1); // control, user-defined mask, normal
    }
    gguf_set_arr_str(o, "tokenizer.ggml.tokens", (const char **) toks, n_vocab);
    gguf_set_arr_data(o, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, n_vocab);
    gguf_set_arr_data(o, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, n_vocab);
    gguf_set_val_u32(o, "tokenizer.ggml.mask_token_id", n_vocab - 1);
    gguf_set_val_bool(o, "tokenizer.ggml.add_bos_token", false);

    for (int64_t i = 0; i < gguf_get_n_tensors(in); i++) {
        struct ggml_tensor * t = ggml_get_tensor(data, gguf_get_tensor_name(in, i));
        gguf_add_tensor(o, t);
    }
    if (!gguf_write_to_file(o, argv[2], false)) {
        fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    printf("%s: %u tokens, mask %u\n", argv[2], n_vocab, n_vocab - 1);
    return 0;
}
