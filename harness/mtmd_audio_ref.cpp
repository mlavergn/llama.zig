// The reference side of `make mtmd-audio-diff`: a C-ABI face on upstream's
// `tools/mtmd/mtmd-audio.cpp`, which is compiled unmodified beside this
// file with the flags `libmtmd-ref` uses (`build/mtmd_audio_diff.zig`).
//
// mtmd-audio.cpp reaches three things outside itself: `g_logger_state`, its
// logger (mtmd.cpp), `mtmd_audio_log_mel`, the log-mel graph
// (mtmd-subcontext.cpp), and the `mtmd_model` its preprocessors read their
// hparams from (mtmd-model.h, a struct whose constructor is inline). The
// logger is defined here; `mtmd_audio_log_mel` is upstream's own text,
// cut out of mtmd-subcontext.cpp at build time by
// `build/mtmd_audio_diff.zig`, so the rest of libmtmd stays out of the link.
// The hparams handle the harness fills is a `new mtmd_model()`.
//
// A preprocessor's public face is `preprocess` -- the entries'
// shapes, placeholders when it defers to the log-mel graph -- and `compute`,
// which runs the graph on `mel_config.sched`. The harness runs both and
// reports each stage, on a CPU scheduler of its own.
//
// **Not a port.**

#include "mtmd-audio.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct mtmd_logger_state g_logger_state = {mtmd_log_callback_default, NULL};

extern "C" {

// hparams the harness fills field by field, held in the model the
// preprocessors' constructors read them from
void * ref_audio_hparams_new(void) { return new mtmd_model(); }
void ref_audio_hparams_free(void * m) { delete (mtmd_model *) m; }

void ref_audio_hparams_set(void * p, int n_mel_bins, int n_fft, int window_len, int hop_len,
                           int sample_rate, int chunk_len, int mimi_downsample) {
    auto * hp = &((mtmd_model *) p)->hparams;
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
    auto * hp = &((mtmd_model *) p)->hparams;
    hp->mel_filters.assign(mel_filters, mel_filters + n_filters);
    hp->window.assign(window, window + n_window);
}

void ref_audio_hparams_set_patches(void * p, const int32_t * sizes, size_t n) {
    ((mtmd_model *) p)->hparams.audio_patch_sizes.assign(sizes, sizes + n);
}

void ref_audio_set_log(ggml_log_callback cb, void * ud) {
    g_logger_state.log_callback = cb;
    g_logger_state.log_callback_user_data = ud;
}

// the fourteen preprocessors, in mtmd-audio.h's order -- d1omni and
// two more last, as the Zig `audio.Kind` keeps it
static mtmd_audio_preprocessor * make(int kind, const mtmd_model * ctx) {
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
        case 10: return new mtmd_audio_preprocessor_parakeet(ctx);
        case 11: return new mtmd_audio_preprocessor_d1omni(ctx);
        case 12: return new mtmd_audio_preprocessor_lfm3a(ctx);
        case 13: return new mtmd_audio_preprocessor_muse_spark_audio(ctx);
        case 14: return new mtmd_audio_preprocessor_nemotron_transcribe(ctx);
        default: return nullptr;
    }
}

// one CPU scheduler for the whole run, as mtmd_backends builds one for a
// context with no accelerator
static ggml_backend_t        g_cpu   = nullptr;
static ggml_backend_sched_t  g_sched = nullptr;

static ggml_backend_sched_t sched_get(void) {
    if (!g_sched) {
        g_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_cpu);
        g_sched = ggml_backend_sched_new(&g_cpu, &buft, 1, 8192, false, true);
    }
    return g_sched;
}

struct ref_proc {
    std::unique_ptr<mtmd_audio_preprocessor> p;
};

static void * proc_init(mtmd_audio_preprocessor * p, int n_threads) {
    auto * r = new ref_proc();
    r->p.reset(p);
    r->p->mel_config.n_threads = n_threads;
    r->p->mel_config.sched     = sched_get();
    r->p->initialize();
    return r;
}

void * ref_audio_proc_new(int kind, void * m, int n_threads) {
    return proc_init(make(kind, (const mtmd_model *) m), n_threads);
}

void ref_audio_proc_free(void * r) { delete (ref_proc *) r; }

// The chatterbox front-end, `audio.Kind.chatterbox_ref` (15), takes a
// variant flag (`is_mtl`) the other preprocessors do not
void * ref_audio_proc_new_flag(int kind, void * m, int flag, int n_threads) {
    if (kind != 15) {
        return nullptr;
    }
    return proc_init(new mtmd_audio_preprocessor_chatterbox_ref((const mtmd_model *) m, flag != 0), n_threads);
}

// The blob both stages report: the number of entries, then per entry nx,
// ny, whether it is a placeholder and its float count (as int64), then the
// floats; then the kept samples' count and the samples, and needs_compute.
static void put_out(std::vector<unsigned char> & blob, const mtmd_audio_preproc_out & o) {
    auto put = [&](const void * d, size_t n) {
        blob.insert(blob.end(), (const unsigned char *) d, (const unsigned char *) d + n);
    };
    auto put64 = [&](int64_t v) { put(&v, 8); };
    put64((int64_t) o.entries.size());
    for (auto & e : o.entries) {
        put64(e.nx());
        put64(e.ny());
        put64(e.is_placeholder() ? 1 : 0);
        if (e.is_placeholder()) {
            put64(0);
        } else {
            const auto & b = e.get_ro_buf();
            put64((int64_t) b.size());
            put(b.data(), b.size() * sizeof(float));
        }
    }
    put64((int64_t) o.samples.size());
    put(o.samples.data(), o.samples.size() * sizeof(float));
    put64(o.needs_compute() ? 1 : 0);
}

// Runs one preprocess, then one compute. Returns the compute's ok, and
// through `out` a malloc'd blob the caller frees: both stages' reports.
int ref_audio_proc_run(void * r, const float * samples, size_t n_samples, void ** out, size_t * out_len) {
    auto & p = *((ref_proc *) r)->p;
    mtmd_audio_preproc_out o = p.preprocess(samples, n_samples);
    std::vector<unsigned char> blob;
    put_out(blob, o);
    bool ok = p.compute(o);
    put_out(blob, o);
    *out = malloc(blob.size());
    memcpy(*out, blob.data(), blob.size());
    *out_len = blob.size();
    return ok ? 1 : 0;
}

// mtmd_audio_log_mel_frame over a cache built as the Voxtral stream builds it
void * ref_mel_frame_new(int n_fft, int window, int n_mel, int sample_rate) {
    auto * cache = new mtmd_audio_cache();
    cache->fill_sin_cos_table(n_fft);
    cache->fill_hann_window(window, true);
    cache->fill_mel_filterbank_matrix(n_mel, n_fft, sample_rate);
    return cache;
}
void ref_mel_frame_free(void * c) { delete (mtmd_audio_cache *) c; }
void ref_mel_frame_run(void * c, const float * samples, int n_fft, int n_mel, float * out) {
    mtmd_audio_log_mel_frame(*(const mtmd_audio_cache *) c, samples, n_fft, n_mel, out);
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
