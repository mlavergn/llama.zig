// The reference side of `make mtmd-audio-diff`: a C-ABI face on upstream's
// `tools/mtmd/mtmd-audio.cpp`, which is compiled unmodified beside this
// file with the flags `libmtmd-ref` uses (`build/mtmd_audio_diff.zig`).
//
// mtmd-audio.cpp reaches two things in clip.cpp: `clip_get_hparams`, which
// its preprocessors bind their hparams through, and `g_logger_state`, its
// logger. Both are defined here instead, so clip.cpp -- and the rest of
// libmtmd -- stays out of the link: the "clip_ctx" a preprocessor is built
// from is this file's `clip_hparams`, cast.
//
// **Not a port.**

#include "mtmd-audio.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct clip_logger_state g_logger_state = {clip_log_callback_default, NULL};

const clip_hparams * clip_get_hparams(const struct clip_ctx * ctx) {
    return reinterpret_cast<const clip_hparams *>(ctx);
}

extern "C" {

// hparams the harness fills field by field
void * ref_audio_hparams_new(void) { return new clip_hparams(); }
void ref_audio_hparams_free(void * hp) { delete (clip_hparams *) hp; }

void ref_audio_hparams_set(void * p, int n_mel_bins, int n_fft, int window_len, int hop_len,
                           int sample_rate, int chunk_len, int mimi_downsample) {
    auto * hp = (clip_hparams *) p;
    hp->n_mel_bins        = n_mel_bins;
    hp->audio_n_fft       = n_fft;
    hp->audio_window_len  = window_len;
    hp->audio_hop_len     = hop_len;
    hp->audio_sample_rate = sample_rate;
    hp->audio_chunk_len   = chunk_len;
    hp->mimi_downsample   = mimi_downsample;
}

void ref_audio_hparams_set_parakeet(void * p, const float * mel_filters, size_t n_filters,
                                    const float * window, size_t n_window) {
    auto * hp = (clip_hparams *) p;
    hp->mel_filters.assign(mel_filters, mel_filters + n_filters);
    hp->window.assign(window, window + n_window);
}

void ref_audio_set_log(ggml_log_callback cb, void * ud) {
    g_logger_state.log_callback = cb;
    g_logger_state.log_callback_user_data = ud;
}

// the eleven preprocessors, in mtmd-audio.h's order
static mtmd_audio_preprocessor * make(int kind, const clip_ctx * ctx) {
    switch (kind) {
        case 0:  return new mtmd_audio_preprocessor_whisper(ctx);
        case 1:  return new mtmd_audio_preprocessor_conformer(ctx);
        case 2:  return new mtmd_audio_preprocessor_granite_speech(ctx);
        case 3:  return new mtmd_audio_preprocessor_gemma4a(ctx);
        case 4:  return new mtmd_audio_preprocessor_gemma4ua(ctx);
        case 5:  return new mtmd_audio_preprocessor_qwen3a(ctx);
        case 6:  return new mtmd_audio_preprocessor_dots3note(ctx);
        case 7:  return new mtmd_audio_preprocessor_mimo_audio(ctx);
        case 8:  return new mtmd_audio_preprocessor_qwen3tts_spk(ctx);
        case 9:  return new mtmd_audio_preprocessor_pockettts(ctx);
        case 10: return new mtmd_audio_preprocessor_parakeet((clip_ctx *) ctx);
        default: return nullptr;
    }
}

struct ref_proc {
    std::unique_ptr<mtmd_audio_preprocessor> p;
};

void * ref_audio_proc_new(int kind, void * hp) {
    auto * r = new ref_proc();
    r->p.reset(make(kind, (const clip_ctx *) hp));
    r->p->initialize();
    return r;
}

void ref_audio_proc_free(void * r) { delete (ref_proc *) r; }

// Runs one preprocess. Returns ok, and through `out` a malloc'd blob the
// caller frees: n_mels, then per mel n_len, n_len_org, n_mel (as int64) and
// n_mel*n_len floats.
int ref_audio_proc_run(void * r, const float * samples, size_t n_samples, void ** out, size_t * out_len) {
    std::vector<mtmd_audio_mel> mels;
    bool ok = ((ref_proc *) r)->p->preprocess(samples, n_samples, mels);
    std::vector<unsigned char> blob;
    auto put = [&](const void * d, size_t n) {
        blob.insert(blob.end(), (const unsigned char *) d, (const unsigned char *) d + n);
    };
    int64_t n = (int64_t) mels.size();
    put(&n, 8);
    for (auto & m : mels) {
        put(&m.n_len, 8);
        put(&m.n_len_org, 8);
        put(&m.n_mel, 8);
        int64_t nd = (int64_t) m.data.size();
        put(&nd, 8);
        put(m.data.data(), m.data.size() * sizeof(float));
    }
    *out = malloc(blob.size());
    memcpy(*out, blob.data(), blob.size());
    *out_len = blob.size();
    return ok ? 1 : 0;
}

void * ref_istft_new(int n_fft, int hop) { return new mtmd_audio_streaming_istft(n_fft, hop); }
void ref_istft_free(void * s) { delete (mtmd_audio_streaming_istft *) s; }
void ref_istft_reset(void * s) { ((mtmd_audio_streaming_istft *) s)->reset(); }

// Returns the number of samples written to `out`, which holds n_fft floats.
size_t ref_istft_frame(void * s, const float * spec, float * out) {
    auto v = ((mtmd_audio_streaming_istft *) s)->process_frame(spec);
    memcpy(out, v.data(), v.size() * sizeof(float));
    return v.size();
}

size_t ref_istft_flush(void * s, float * out) {
    auto v = ((mtmd_audio_streaming_istft *) s)->flush();
    memcpy(out, v.data(), v.size() * sizeof(float));
    return v.size();
}

} // extern "C"
