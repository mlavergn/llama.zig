// The reference side of `make mtmd-decode-diff`: upstream's vendored
// stb_image.h and miniaudio.h, compiled unmodified with exactly the defines
// llama.cpp/tools/mtmd/mtmd-helper.cpp gives them, in a C++17 unit as the
// helper is (C++ is what makes stb's `pow(float, float)` call `powf`).
//
// Two departures, both about memory the C leaves uninitialized and the
// port zeroes (src/mtmd/vendor/stb_image.zig, "As mtmd-helper.cpp compiles
// it"), and neither reachable on a well-formed file:
//
// - STBI_MALLOC is calloc. stb_image mallocs the JPEG component planes
//   and a TGA's pixels without filling them, and a corrupt file can leave
//   parts unwritten.
// - MA_MALLOC is calloc too. dr_flac's decoded-samples buffer persists
//   across frames, and a corrupt subframe that fails to decode leaves part
//   of it as it was -- from the start, uninitialized -- in a frame whose
//   CRC can still match by chance.
// - Each load runs on a zeroed stack (`scrub_stack`). `stbi__bmp_load`'s
//   palette is an uninitialized local a corrupt index can read past, and so
//   is the flat HDR loop's `rgbe` when a read comes up short.
//
// `ref_audio_decode` is mtmd-helper.cpp's `audio_helpers::decode_audio_from_buf`
// line for line, reporting which step failed.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define STBI_MALLOC(sz) calloc(1, sz)
#define STBI_REALLOC(p, newsz) realloc(p, newsz)
#define STBI_FREE(p) free(p)
#define MA_MALLOC(sz) calloc(1, (sz))

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

__attribute__((noinline)) static void scrub_stack(void) {
    volatile unsigned char buf[1 << 16];
    memset((void *) buf, 0, sizeof(buf));
}

extern "C" {

unsigned char * ref_stbi_load(const unsigned char * buf, int len, int req_comp, int * x, int * y, int * comp, const char ** reason) {
    stbi__g_failure_reason = NULL;
    *x = *y = *comp = 0;
    scrub_stack();
    unsigned char * data = stbi_load_from_memory(buf, len, x, y, comp, req_comp);
    *reason = stbi__g_failure_reason;
    return data;
}

void ref_stbi_free(unsigned char * data) {
    stbi_image_free(data);
}

// Return: 0 decoded, 1 init failed, 2 length failed, 3 the length exceeds
// `cap` (not decoded), 4 the read failed (`*out` still holds the zeroed
// vector and whatever the read wrote, as upstream's would).
int ref_audio_decode(const unsigned char * buf_in, size_t len, int target_sampler_rate, uint64_t cap, float ** out, uint64_t * n_frames) {
    *out = NULL;
    *n_frames = 0;
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return 1;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return 2;
    }
    *n_frames = frame_count;
    if (frame_count > cap) {
        ma_decoder_uninit(&decoder);
        return 3;
    }

    float * pcm = (float *) calloc(frame_count ? frame_count : 1, sizeof(float));
    *out = pcm;
    result = ma_decoder_read_pcm_frames(&decoder, pcm, frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return 4;
    }

    ma_decoder_uninit(&decoder);
    return 0;
}

void ref_audio_free(float * p) {
    free(p);
}

}

// dr_mp3's kernels, one by one, for the harness's kernel diff: the
// functions are static, and this unit is where they are visible.
extern "C" {
void ref_mp3_dct3_9(float * y) { ma_dr_mp3_L3_dct3_9(y); }
void ref_mp3_imdct36(float * grbuf, float * overlap, const float * window, int nbands) { ma_dr_mp3_L3_imdct36(grbuf, overlap, window, nbands); }
void ref_mp3_imdct_short(float * grbuf, float * overlap, int nbands) { ma_dr_mp3_L3_imdct_short(grbuf, overlap, nbands); }
void ref_mp3_antialias(float * grbuf, int nbands) { ma_dr_mp3_L3_antialias(grbuf, nbands); }
void ref_mp3_dct2(float * grbuf, int n) { ma_dr_mp3d_DCT_II(grbuf, n); }
void ref_mp3_synth_granule(float * qmf_state, float * grbuf, int nbands, int nch, short * pcm, float * lins) { ma_dr_mp3d_synth_granule(qmf_state, grbuf, nbands, nch, pcm, lins); }
float ref_mp3_pow43(int x) { return ma_dr_mp3_L3_pow_43(x); }
float ref_mp3_ldexp_q2(float y, int e) { return ma_dr_mp3_L3_ldexp_q2(y, e); }
}

extern "C" {
// One `ma_dr_mp3dec_decode_frame` over a decoder the caller keeps; the
// decoder is handed back as bytes so the harness can compare its state.
size_t ref_mp3_dec_size(void) { return sizeof(ma_dr_mp3dec); }
int ref_mp3_decode_frame(void * dec, const unsigned char * mp3, int mp3_bytes, short * pcm, int * frame_bytes) {
    ma_dr_mp3dec_frame_info info;
    memset(&info, 0, sizeof(info));
    int n = ma_dr_mp3dec_decode_frame((ma_dr_mp3dec *) dec, mp3, mp3_bytes, pcm, &info);
    *frame_bytes = info.frame_bytes;
    return n;
}
}

extern "C" {
// The scalefactor decoder over the port's own state, which has the C's layout.
void ref_mp3_decode_scalefactors(const unsigned char * hdr, unsigned char * ist_pos, void * bs, const void * gr, float * scf, int ch) {
    ma_dr_mp3_L3_decode_scalefactors(hdr, ist_pos, (ma_dr_mp3_bs *) bs, (const ma_dr_mp3_L3_gr_info *) gr, scf, ch);
}
unsigned ref_mp3_bs_get_bits(void * bs, int n) { return ma_dr_mp3_bs_get_bits((ma_dr_mp3_bs *) bs, n); }
}

extern "C" {
void ref_mp3_synth_pair(short * pcm, int nch, const float * z) { ma_dr_mp3d_synth_pair(pcm, nch, z); }
unsigned long long ref_frame_count_after_resampling(unsigned out, unsigned in, unsigned long long n) { return ma_calculate_frame_count_after_resampling(out, in, n); }
}
