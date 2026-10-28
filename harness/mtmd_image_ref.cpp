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

// Runs preprocessor `kind` (the order of the Zig `Kind` enum) on an RGB
// image -- a placeholder when `rgb` is null -- and writes the serialized
// output to `*out` (free with `ref_free`).
int ref_preprocess(int kind, int proj_type, const ref_hparams * hp, const uint8_t * rgb, int nx, int ny,
                   uint8_t ** out, size_t * out_len) {
    clip_ctx ctx;
    clip_hparams & h = ctx.hparams;
    ctx.proj = (projector_type) proj_type;
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

    std::unique_ptr<mtmd_image_preprocessor> pp;
    switch (kind) {
        case 0:  pp = std::make_unique<mtmd_image_preprocessor_llava_uhd>(&ctx); break;
        case 1:  pp = std::make_unique<mtmd_image_preprocessor_fixed_size>(&ctx); break;
        case 2:  pp = std::make_unique<mtmd_image_preprocessor_dyn_size>(&ctx); break;
        case 3:  pp = std::make_unique<mtmd_image_preprocessor_longest_edge>(&ctx); break;
        case 4:  pp = std::make_unique<mtmd_image_preprocessor_minicpmv>(&ctx); break;
        case 5:  pp = std::make_unique<mtmd_image_preprocessor_lfm2>(&ctx); break;
        case 6:  pp = std::make_unique<mtmd_image_preprocessor_idefics3>(&ctx); break;
        case 7:  pp = std::make_unique<mtmd_image_preprocessor_internvl>(&ctx); break;
        case 8:  pp = std::make_unique<mtmd_image_preprocessor_deepseekocr>(&ctx); break;
        case 9:  pp = std::make_unique<mtmd_image_preprocessor_step3vl>(&ctx); break;
        case 10: pp = std::make_unique<mtmd_image_preprocessor_youtuvl>(&ctx); break;
        case 11: pp = std::make_unique<mtmd_image_preprocessor_granite>(&ctx); break;
        case 12: pp = std::make_unique<mtmd_image_preprocessor_muse_glimmer>(&ctx); break;
        case 13: pp = std::make_unique<mtmd_image_preprocessor_deepseek4v>(&ctx); break;
        default: return 1;
    }

    clip_image_u8 img;
    img.set_size({ nx, ny }, rgb == nullptr);
    if (rgb) {
        img.cpy_buf(std::vector<uint8_t>(rgb, rgb + (size_t) nx * ny * 3));
    }

    mtmd_image_preproc_out res = pp->preprocess(img);

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
    return 0;
}

void ref_free(uint8_t * p) {
    free(p);
}

} // extern "C"
