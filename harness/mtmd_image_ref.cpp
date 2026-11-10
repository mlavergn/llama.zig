// The reference side of `make mtmd-image-diff`: upstream's
// `llama.cpp/tools/mtmd/mtmd-image.cpp`, compiled unmodified beside this
// file, driven through one C entry point.
//
// **Not a port.** A harness. Upstream's preprocessors read their hparams
// through `clip_get_hparams(ctx)` and deepseekocr its projector through
// `clip_get_projector_type(ctx)`, both defined in `clip.cpp`. This file
// defines both instead, over its own `clip_ctx`, so the preprocessors run
// without a model -- and defines `g_logger_state`, so their LOG_DBG lines
// can be captured and compared.
//
// Every output is serialized here to bytes; `harness/mtmd_image_diff.zig`
// serializes the port's the same way, and the two must be equal.

#include "mtmd-image.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

struct clip_ctx {
    clip_hparams   hparams;
    projector_type proj;
};

const clip_hparams * clip_get_hparams(const clip_ctx * ctx) {
    return &ctx->hparams;
}

projector_type clip_get_projector_type(const clip_ctx * ctx) {
    return ctx->proj;
}

struct clip_logger_state g_logger_state = { clip_log_callback_default, nullptr };

extern "C" {

// The hparams a preprocessor reads, flat, laid out as the Zig harness's
// `RefHparams`.
struct ref_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t n_merge;
    int32_t image_longest_edge;
    int32_t image_min_pixels;
    int32_t image_max_pixels;
    int32_t image_resize_algo;
    int32_t image_resize_pad;
    uint8_t image_pad_color[4];
    int32_t n_candidates;
    int32_t candidates[64]; // width, height pairs
    int32_t preproc_min_tiles;
    int32_t preproc_max_tiles;
    int32_t preproc_tile_size;
    int32_t image_resize_algo_rf;
    int32_t image_resize_algo_ov;
    int32_t image_pad_rf;
    int32_t image_pad_ov;
    uint8_t image_pad_color_rf[4];
    uint8_t image_pad_color_ov[4];
    float   image_mean[3];
    float   image_std[3];
    int32_t dsv4_max_n_token;
    int32_t dsv4_max_wh_ratio;
    int32_t max_slice_nums; // v0.7.0
    float   preproc_max_pixels_tolerance; // v0.8.0
    int32_t image_round_half_even;        // v0.8.0
    int32_t image_floor_before_align;     // v0.8.0
};

typedef void (*ref_log_fn)(int level, const char * text, void * user_data);

void ref_set_log(ref_log_fn fn, void * user_data) {
    g_logger_state.log_callback           = (ggml_log_callback) fn;
    g_logger_state.log_callback_user_data = user_data;
}

static void put(std::vector<uint8_t> & out, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    out.insert(out.end(), b, b + n);
}

static void put_i32(std::vector<uint8_t> & out, int32_t v) {
    put(out, &v, sizeof(v));
}

// one clip_image_f32: size, flags, anyres, then every float's bits
static void put_image(std::vector<uint8_t> & out, const clip_image_f32 & img) {
    put_i32(out, img.nx());
    put_i32(out, img.ny());
    put_i32(out, img.add_viewsep ? 1 : 0);
    put_i32(out, img.add_newline ? 1 : 0);
    put_i32(out, img.anyres.grid_x);
    put_i32(out, img.anyres.grid_y);
    put_i32(out, img.anyres.orig_nx);
    put_i32(out, img.anyres.orig_ny);
    const uint64_t n = img.is_placeholder() ? 0 : img.get_ro_buf().size();
    put(out, &n, sizeof(n));
    if (n) {
        put(out, img.get_ro_buf().data(), n * sizeof(float));
    }
}

static void fill_hparams(clip_hparams & h, const ref_hparams * hp) {
    h.image_size         = hp->image_size;
    h.patch_size         = hp->patch_size;
    h.n_merge            = hp->n_merge;
    h.image_longest_edge = hp->image_longest_edge;
    h.image_min_pixels   = hp->image_min_pixels;
    h.image_max_pixels   = hp->image_max_pixels;
    h.image_resize_algo  = (resize_algo) hp->image_resize_algo;
    h.image_resize_pad   = (pad_style) hp->image_resize_pad;
    for (int i = 0; i < 3; i++) {
        h.image_pad_color[i]    = hp->image_pad_color[i];
        h.image_pad_color_rf[i] = hp->image_pad_color_rf[i];
        h.image_pad_color_ov[i] = hp->image_pad_color_ov[i];
        h.image_mean[i]         = hp->image_mean[i];
        h.image_std[i]          = hp->image_std[i];
    }
    for (int i = 0; i < hp->n_candidates; i++) {
        h.image_res_candidates.push_back({ hp->candidates[2 * i], hp->candidates[2 * i + 1] });
    }
    h.preproc_min_tiles    = hp->preproc_min_tiles;
    h.preproc_max_tiles    = hp->preproc_max_tiles;
    h.preproc_tile_size    = hp->preproc_tile_size;
    h.image_resize_algo_rf = (resize_algo) hp->image_resize_algo_rf;
    h.image_resize_algo_ov = (resize_algo) hp->image_resize_algo_ov;
    h.image_pad_rf         = (pad_style) hp->image_pad_rf;
    h.image_pad_ov         = (pad_style) hp->image_pad_ov;
    h.dsv4_max_n_token     = hp->dsv4_max_n_token;
    h.dsv4_max_wh_ratio    = hp->dsv4_max_wh_ratio;
    h.max_slice_nums       = hp->max_slice_nums;

    h.preproc_max_pixels_tolerance = hp->preproc_max_pixels_tolerance;
    h.image_round_half_even        = hp->image_round_half_even != 0;
    h.image_floor_before_align     = hp->image_floor_before_align != 0;
}

static std::unique_ptr<mtmd_image_preprocessor> make_preprocessor(int kind, const clip_ctx * ctx) {
    switch (kind) {
        case 0:  return std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx);
        case 1:  return std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx);
        case 2:  return std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx);
        case 3:  return std::make_unique<mtmd_image_preprocessor_longest_edge>(ctx);
        case 4:  return std::make_unique<mtmd_image_preprocessor_minicpmv>(ctx);
        case 5:  return std::make_unique<mtmd_image_preprocessor_lfm2>(ctx);
        case 6:  return std::make_unique<mtmd_image_preprocessor_idefics3>(ctx);
        case 7:  return std::make_unique<mtmd_image_preprocessor_internvl>(ctx);
        case 8:  return std::make_unique<mtmd_image_preprocessor_deepseekocr>(ctx);
        case 9:  return std::make_unique<mtmd_image_preprocessor_step3vl>(ctx);
        case 10: return std::make_unique<mtmd_image_preprocessor_youtuvl>(ctx);
        case 11: return std::make_unique<mtmd_image_preprocessor_granite>(ctx);
        case 12: return std::make_unique<mtmd_image_preprocessor_muse_glimmer>(ctx);
        case 13: return std::make_unique<mtmd_image_preprocessor_deepseek4v>(ctx);
        case 14: return std::make_unique<mtmd_image_preprocessor_glm5v>(ctx);
        case 15: return std::make_unique<mtmd_image_preprocessor_cohere2v>(ctx);
        case 16: return std::make_unique<mtmd_image_preprocessor_lfm3vl>(ctx);
        case 17: return std::make_unique<mtmd_image_preprocessor_deepseek41v>(ctx);
        case 18: return std::make_unique<mtmd_image_preprocessor_minicpm_ocr_1_0>(ctx);
        case 19: return std::make_unique<mtmd_image_preprocessor_qwen3vl>(ctx);
        default: return nullptr;
    }
}

static void put_what(uint8_t ** out, size_t * out_len, const char * what) {
    *out_len = strlen(what);
    *out     = (uint8_t *) malloc(*out_len + 1);
    memcpy(*out, what, *out_len);
}

// Runs preprocessor `kind` (the order of the Zig `Kind` enum) on an RGB
// image -- a placeholder when `rgb` is null -- through `preprocess`, or
// `preprocess_video_frame` when `video`, and writes the serialized output to
// `*out` (free with `ref_free`). Returns 0, or 2 with `what()` in `*out` when
// it threw.
int ref_preprocess(int kind, int proj_type, const ref_hparams * hp, const uint8_t * rgb, int nx, int ny, int video,
                   uint8_t ** out, size_t * out_len) {
    clip_ctx ctx;
    ctx.proj = (projector_type) proj_type;
    fill_hparams(ctx.hparams, hp);

    std::unique_ptr<mtmd_image_preprocessor> pp = make_preprocessor(kind, &ctx);
    if (!pp) {
        return 1;
    }

    clip_image_u8 img;
    img.set_size({ nx, ny }, rgb == nullptr);
    if (rgb) {
        img.cpy_buf(std::vector<uint8_t>(rgb, rgb + (size_t) nx * ny * 3));
    }

    try {
        mtmd_image_preproc_out res = video ? pp->preprocess_video_frame(img) : pp->preprocess(img);

        std::vector<uint8_t> buf;
        put_i32(buf, res.grid_x);
        put_i32(buf, res.grid_y);
        put_i32(buf, res.has_overview() ? 1 : 0);
        put_image(buf, res.overview);
        const uint64_t n = res.entries.size();
        put(buf, &n, sizeof(n));
        for (const auto & e : res.entries) {
            put_image(buf, e);
        }

        *out = (uint8_t *) malloc(buf.size());
        memcpy(*out, buf.data(), buf.size());
        *out_len = buf.size();
    } catch (const std::exception & e) {
        put_what(out, out_len, e.what());
        return 2;
    }
    return 0;
}

// `get_video_frame_size` of preprocessor `kind`.
int ref_video_frame_size(int kind, const ref_hparams * hp, int n_frames, int width, int height, clip_image_size * out) {
    clip_ctx ctx;
    ctx.proj = PROJECTOR_TYPE_UNKNOWN;
    fill_hparams(ctx.hparams, hp);
    return make_preprocessor(kind, &ctx)->get_video_frame_size(n_frames, width, height, *out) ? 1 : 0;
}

// The static helpers upstream exposes for tests/test-mtmd-impl.cpp.
void ref_dyn_target_size(const ref_hparams * hp, int w, int h, clip_image_size * out) {
    clip_hparams hparams;
    fill_hparams(hparams, hp);
    *out = mtmd_image_preprocessor_dyn_size::get_target_size(hparams, { w, h });
}

// Returns 0, or 2 with `what()` in `*what` (free with `ref_free`) when it threw.
int ref_longest_target_size(const ref_hparams * hp, int w, int h, clip_image_size * out, uint8_t ** what, size_t * what_len) {
    clip_hparams hparams;
    fill_hparams(hparams, hp);
    try {
        *out = mtmd_image_preprocessor_longest_edge::get_target_size(hparams, { w, h });
    } catch (const std::exception & e) {
        put_what(what, what_len, e.what());
        return 2;
    }
    return 0;
}

void ref_qwen3vl_smart_resize(int w, int h, int factor, int64_t min_pixels, int64_t max_pixels, int n_frames, int temporal_factor, clip_image_size * out) {
    *out = mtmd_image_preprocessor_qwen3vl::smart_resize(w, h, factor, min_pixels, max_pixels, n_frames, temporal_factor);
}

int ref_lfm2_should_tile(const ref_hparams * hp, int w, int h) {
    clip_hparams hparams;
    fill_hparams(hparams, hp);
    return mtmd_image_preprocessor_lfm2::should_tile(hparams, { w, h }) ? 1 : 0;
}

void ref_lfm2_smart_resize(const ref_hparams * hp, int w, int h, clip_image_size * out) {
    clip_hparams hparams;
    fill_hparams(hparams, hp);
    *out = mtmd_image_preprocessor_lfm2::smart_resize(hparams, { w, h });
}

void ref_lfm2_grid_layout(const ref_hparams * hp, int height, int width, clip_image_size * out) {
    clip_hparams hparams;
    fill_hparams(hparams, hp);
    *out = mtmd_image_preprocessor_lfm2::get_grid_layout(hparams, height, width);
}

void ref_deepseek41v_plan_grid(int w, int h, int p, int r, int max_n_token, int min_pixels, int * best_w, int * best_h) {
    mtmd_image_preprocessor_deepseek41v::plan_grid(w, h, p, r, max_n_token, min_pixels, *best_w, *best_h);
}

// FNV-1a over `img`'s pixels, as tests/test-mtmd-impl.cpp's `hash_pixels`.
static uint32_t hash_pixels(const clip_image_u8 & img, uint32_t hash) {
    for (uint8_t v : img.get_ro_buf()) {
        hash = (hash ^ v) * 16777619u;
    }
    return hash;
}

// `test_image_preprocessor_lfm2_torch_resize`'s measurement: the hash of the
// tiles and the thumbnail, through the constructor that takes the hparams.
uint32_t ref_lfm2_slice_hash(const ref_hparams * hp, const uint8_t * rgb, int nx, int ny) {
    clip_hparams hparams;
    fill_hparams(hparams, hp);
    const mtmd_image_preprocessor_lfm2 preproc(hparams);

    clip_image_u8 img;
    img.set_size({ nx, ny }, false);
    img.cpy_buf(std::vector<uint8_t>(rgb, rgb + (size_t) nx * ny * 3));
    const auto sliced = preproc.slice_image(img, preproc.get_slice_instructions({ nx, ny }));

    uint32_t hash = 2166136261u;
    for (const auto & slice : sliced.slices) {
        hash = hash_pixels(slice, hash);
    }
    return hash_pixels(sliced.overview, hash);
}

void ref_free(uint8_t * p) {
    free(p);
}

} // extern "C"
