/* The in-process half of `scripts/meta-diff`: the reference meta backend and
 * the port, side by side in one process, asked the same questions.
 *
 * `scripts/meta-diff` compiles `llama.cpp/ggml/src/ggml-backend-meta.cpp`
 * with its eight unmangled exports renamed to `ref_*` on the command line --
 * which renames the C++'s own internal calls to them too, so the renamed
 * object is self-consistent -- and links it beside `src/ggml/backend_meta.zig`
 * and a `libggml.a` with the stock meta object deleted. Everything else in
 * ggml is shared, so the meta backend is the only variable.
 *
 * What the end-to-end half (`harness/meta_dump.c`) cannot reach, this does:
 *
 *   - every device, buffer-type, buffer and backend vtable answer;
 *   - split states libllama never hands out for Qwen3.5: axis 2, a segmented
 *     axis-0 weight, a weight with two segments of different repeat counts,
 *     PARTIAL weights with and without a contributor mask, deliberate zero
 *     slices;
 *   - `memset_tensor` (no Qwen3.5 decode calls it) and `clear`, on every
 *     layout; partial-range `set_tensor`; the async transfer pair;
 *   - graphs built to steer `graph_compute` down its delayed-reduction
 *     branches: `ADD_ID` with a partial bias followed by a `MUL` chain, two
 *     partial branches summed by one `ADD`, and zero-slice devices that need
 *     the `SCALE`-by-zero step before the butterfly;
 *   - one, two, three and four devices, Metal and CPU mixed, so the
 *     butterfly runs with no reduction, one step, a non-power-of-two fold,
 *     and two steps.
 *
 * Both implementations write one line per fact into their own transcript --
 * with GGML_META_DEBUG=1 that includes every SPLIT_STATE the backend derives
 * -- and the transcripts are compared line by line. A transcript is also
 * checked on its own where the answer is known (a set/get round trip must
 * return what was written), so a bug both sides share in that place still
 * fails.
 *
 * Where the shared library calls a meta predicate by name -- `alloc.zig`'s
 * `ggml_backend_buft_is_meta`, `backend.zig`'s `ggml_backend_buffer_is_meta`
 * -- it reaches the port's, which says "no" to a reference object. The
 * harness therefore calls each side's `alloc_ctx_tensors_from_buft`
 * directly, and the end-to-end half, which links one implementation per
 * binary, is what covers those call sites faithfully.
 *
 * Usage: meta_diff [combo ...]   (default: every combo below)
 *   A combo is one letter per device: m = Metal, c = CPU.
 */

#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <execinfo.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* The reference, renamed. */
const char * ref_ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis);
ggml_backend_dev_t ref_ggml_backend_meta_device(ggml_backend_dev_t *, size_t, ggml_backend_meta_get_split_state_t, void *);
bool ref_ggml_backend_buft_is_meta(ggml_backend_buffer_type_t);
bool ref_ggml_backend_buffer_is_meta(ggml_backend_buffer_t);
bool ref_ggml_backend_is_meta(ggml_backend_t);
size_t ref_ggml_backend_meta_n_backends(ggml_backend_t);
ggml_backend_t ref_ggml_backend_meta_simple_backend(ggml_backend_t, size_t);
struct ggml_backend_buffer * ref_ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context *, ggml_backend_buffer_type_t);

typedef struct {
    const char * label;
    const char * (*axis_name)(enum ggml_backend_meta_split_axis);
    ggml_backend_dev_t (*meta_device)(ggml_backend_dev_t *, size_t, ggml_backend_meta_get_split_state_t, void *);
    bool (*buft_is_meta)(ggml_backend_buffer_type_t);
    bool (*buffer_is_meta)(ggml_backend_buffer_t);
    bool (*is_meta)(ggml_backend_t);
    size_t (*n_backends)(ggml_backend_t);
    ggml_backend_t (*simple_backend)(ggml_backend_t, size_t);
    struct ggml_backend_buffer * (*alloc_ctx)(struct ggml_context *, ggml_backend_buffer_type_t);
} impl_t;

static const impl_t IMPL_REF = {
    "ref", ref_ggml_backend_meta_split_axis_name, ref_ggml_backend_meta_device, ref_ggml_backend_buft_is_meta,
    ref_ggml_backend_buffer_is_meta, ref_ggml_backend_is_meta, ref_ggml_backend_meta_n_backends,
    ref_ggml_backend_meta_simple_backend, ref_ggml_backend_meta_alloc_ctx_tensors_from_buft,
};
static const impl_t IMPL_PORT = {
    "port", ggml_backend_meta_split_axis_name, ggml_backend_meta_device, ggml_backend_buft_is_meta,
    ggml_backend_buffer_is_meta, ggml_backend_is_meta, ggml_backend_meta_n_backends,
    ggml_backend_meta_simple_backend, ggml_backend_meta_alloc_ctx_tensors_from_buft,
};

// ---------------------------------------------------------------------------
// Transcript

static FILE * out = NULL;          /* the current implementation's transcript */
static int self_failures = 0;      /* known-answer checks that failed, either side */

static void on_log(enum ggml_log_level level, const char * text, void * ud) {
    (void) level; (void) ud;
    /* Only the meta backend's own derivations: everything else ggml logs
     * (Metal pipeline compiles with addresses) is not about this file. */
    if (out && strstr(text, "SPLIT_STATE")) fprintf(out, "log %s", text);
}

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// ---------------------------------------------------------------------------
// The split-state callback
//
// A weight's name says how to split it. `nd` comes in through the user data,
// as libllama's does.

typedef struct { int nd; int tag; } split_ud;

/* llama-model.cpp's rounding: device j's upper bound is total*(j+1)/nd
 * rounded down to a multiple of g, the last takes the rest. With g larger
 * than total/nd that gives deliberate zero slices. */
static void split_even(int64_t * ne, int stride, int nd, int64_t total, int64_t g) {
    int64_t low = 0;
    for (int j = 0; j < nd - 1; j++) {
        int64_t high = total * (j + 1) / nd;
        high -= high % g;
        ne[j * stride] = high - low;
        low = high;
    }
    ne[(nd - 1) * stride] = total - low;
}

static struct ggml_backend_meta_split_state split_cb(const struct ggml_tensor * t, void * udp) {
    const split_ud * ud = (const split_ud *) udp;
    const int nd = ud->nd;
    struct ggml_backend_meta_split_state ss;
    memset(&ss, 0, sizeof(ss));
    ss.nr[0] = 1;
    ss.n_segments = 1;
    const char * n = t->name;
    const int64_t blck = ggml_blck_size(t->type);
#define PFX(p) (strncmp(n, p, strlen(p)) == 0)
    if (PFX("ax0h_") || PFX("ax1h_")) {
        /* head-aligned: multiples of 32 */
        const int axis = PFX("ax0h_") ? 0 : 1;
        ss.axis = (enum ggml_backend_meta_split_axis) axis;
        split_even(ss.ne, 1, nd, t->ne[axis], 32);
    } else if (PFX("ax0z_") || PFX("ax1z_")) {
        /* device 1 gets nothing; the rest share evenly */
        const int axis = PFX("ax0z_") ? 0 : 1;
        ss.axis = (enum ggml_backend_meta_split_axis) axis;
        if (nd == 1) {
            ss.ne[0] = t->ne[axis];
        } else {
            int64_t tmp[16] = {0};
            split_even(tmp, 1, nd - 1, t->ne[axis], axis == 0 ? blck : 1);
            for (int j = 0, k = 0; j < nd; j++) ss.ne[j] = j == 1 ? 0 : tmp[k++];
        }
    } else if (PFX("ax0nr_")) {
        /* one segment of a third, repeated three times */
        ss.axis = GGML_BACKEND_SPLIT_AXIS_0;
        split_even(ss.ne, 1, nd, t->ne[0] / 3, blck);
        ss.nr[0] = 3;
    } else if (PFX("ax1nr_")) {
        ss.axis = GGML_BACKEND_SPLIT_AXIS_1;
        split_even(ss.ne, 1, nd, t->ne[1] / 4, 1);
        ss.nr[0] = 4;
    } else if (PFX("ax1seg_")) {
        /* a fused QKV: segment 0 (a third) twice, segment 1 (a third) once */
        ss.axis = GGML_BACKEND_SPLIT_AXIS_1;
        split_even(ss.ne + 0,  1, nd, t->ne[1] / 3, 2);
        split_even(ss.ne + nd, 1, nd, t->ne[1] / 3, 2);
        ss.nr[0] = 2;
        ss.nr[1] = 1;
        ss.n_segments = 2;
    } else if (PFX("ax0_") || PFX("ax1_") || PFX("ax2_")) {
        const int axis = n[2] - '0';
        ss.axis = (enum ggml_backend_meta_split_axis) axis;
        split_even(ss.ne, 1, nd, t->ne[axis], axis == 0 ? blck : 1);
    } else if (PFX("partm_")) {
        /* PARTIAL with a contributor mask: even devices contribute */
        ss.axis = GGML_BACKEND_SPLIT_AXIS_PARTIAL;
        for (int j = 0; j < nd; j++) ss.ne[j] = (j % 2 == 0) ? 1 : 0;
    } else if (PFX("part_")) {
        /* PARTIAL, no mask: every device contributes */
        ss.axis = GGML_BACKEND_SPLIT_AXIS_PARTIAL;
    } else {
        ss.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
#undef PFX
    return ss;
}

// ---------------------------------------------------------------------------
// Fixtures

#define E  64   /* embedding */
#define FF 96   /* feed-forward */
#define T  5    /* tokens */
#define HD 32   /* head size */
#define NH 2    /* heads */
#define KV 8    /* cache rows */

/* Deterministic bytes that are valid for the tensor's type: small floats
 * for f32/f16, a small scale and arbitrary quants for q8_0. */
static void fill(const struct ggml_tensor * t, uint8_t * dst, uint32_t seed) {
    const size_t nb = ggml_nbytes(t);
    uint32_t s = seed * 2654435761u + 12345u;
    if (t->type == GGML_TYPE_F32) {
        float * f = (float *) dst;
        for (size_t i = 0; i < nb / 4; i++) { s = s * 1103515245u + 12345u; f[i] = ((int)((s >> 9) % 2001) - 1000) / 4000.0f; }
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t * h = (ggml_fp16_t *) dst;
        for (size_t i = 0; i < nb / 2; i++) { s = s * 1103515245u + 12345u; h[i] = ggml_fp32_to_fp16(((int)((s >> 9) % 2001) - 1000) / 4000.0f); }
    } else if (t->type == GGML_TYPE_Q8_0) {
        for (size_t b = 0; b < nb / 34; b++) {
            uint8_t * blk = dst + b * 34;
            ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + (b % 7) * 0.0005f);
            memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) { s = s * 1103515245u + 12345u; blk[2 + i] = (uint8_t) (s >> 13); }
        }
    } else {
        for (size_t i = 0; i < nb; i++) { s = s * 1103515245u + 12345u; dst[i] = (uint8_t) (s >> 16); }
    }
}

typedef struct {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buf;
    struct ggml_tensor * t[32];
    int n;
} weights_t;

static struct ggml_tensor * W(weights_t * w, const char * name) {
    for (int i = 0; i < w->n; i++) if (strcmp(w->t[i]->name, name) == 0) return w->t[i];
    fprintf(stderr, "no weight %s\n", name);
    abort();
}

static struct ggml_tensor * mk(weights_t * w, const char * name, enum ggml_type type, int nd, int64_t a, int64_t b, int64_t c) {
    int64_t ne[3] = {a, b, c};
    struct ggml_tensor * t = ggml_new_tensor(w->ctx, type, nd, ne);
    ggml_set_name(t, name);
    w->t[w->n++] = t;
    return t;
}

static void build_weights(weights_t * w) {
    struct ggml_init_params ip = { 64 * ggml_tensor_overhead(), NULL, true };
    w->ctx = ggml_init(ip);
    w->n = 0;
    mk(w, "mir_norm",    GGML_TYPE_F32,  1, E, 1, 1);
    mk(w, "mir_scale",   GGML_TYPE_F32,  1, E, 1, 1);
    mk(w, "ax1_up",      GGML_TYPE_Q8_0, 2, E, FF, 1);
    mk(w, "ax1_gate",    GGML_TYPE_F32,  2, E, FF, 1);
    mk(w, "ax0_down",    GGML_TYPE_F32,  2, FF, E, 1);
    mk(w, "ax0_down2",   GGML_TYPE_F32,  2, FF, E, 1);
    mk(w, "ax0_q8",      GGML_TYPE_Q8_0, 2, 128, 6, 1);
    mk(w, "ax1z_up",     GGML_TYPE_F32,  2, E, FF, 1);
    mk(w, "ax0z_down",   GGML_TYPE_F32,  2, FF, E, 1);
    mk(w, "ax1seg_qkv",  GGML_TYPE_F32,  2, E, 3 * 32, 1);
    mk(w, "ax1nr_x",     GGML_TYPE_F32,  2, E, 64, 1);
    mk(w, "ax0nr_y",     GGML_TYPE_F32,  2, 48, 8, 1);
    mk(w, "ax2_t",       GGML_TYPE_F32,  3, 8, 4, 6);
    mk(w, "ax1_3d",      GGML_TYPE_F16,  3, 16, 12, 2);
    mk(w, "partm_bias",  GGML_TYPE_F32,  2, E, 4, 1);
    mk(w, "part_bias",   GGML_TYPE_F32,  1, E, 1, 1);
    mk(w, "ax1h_q",      GGML_TYPE_F32,  2, E, NH * HD, 1);
    mk(w, "ax1h_k",      GGML_TYPE_F32,  2, E, NH * HD, 1);
    mk(w, "ax1h_v",      GGML_TYPE_F32,  2, E, NH * HD, 1);
    mk(w, "ax0h_o",      GGML_TYPE_F32,  2, NH * HD, E, 1);
    mk(w, "ax0h_kcache", GGML_TYPE_F32,  2, NH * HD, KV, 1);
    mk(w, "ax0h_vcache", GGML_TYPE_F32,  2, NH * HD, KV, 1);
    mk(w, "ax0h_emb",     GGML_TYPE_F32,  2, E, 10, 1);
}

// ---------------------------------------------------------------------------
// Device, buffer type and backend answers

static void dump_device(const impl_t * im, ggml_backend_dev_t dev, ggml_backend_dev_t * devs, int nd, split_ud * ud, split_ud * ud2) {
    fprintf(out, "dev name=%s desc=%s type=%d\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), (int) ggml_backend_dev_type(dev));
    size_t free_mem = 0, total = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total);
    fprintf(out, "dev memory total=%zu\n", total);
    struct ggml_backend_dev_props p;
    memset(&p, 0x5a, sizeof(p));
    ggml_backend_dev_get_props(dev, &p);
    fprintf(out, "props name=%s desc=%s type=%d id=%s total=%zu caps=%d%d%d%d%d\n", p.name, p.description, (int) p.type,
            p.device_id ? p.device_id : "(null)", p.memory_total,
            p.caps.async, p.caps.host_buffer, p.caps.buffer_from_host_ptr, p.caps.events, p.caps.mmap_support);
    fprintf(out, "dev reg=%s is_meta(own buft)=%d\n", ggml_backend_dev_backend_reg(dev) ? "set" : "(null)",
            im->buft_is_meta(ggml_backend_dev_buffer_type(dev)));

    /* interning */
    ggml_backend_dev_t again = im->meta_device(devs, (size_t) nd, split_cb, ud);
    ggml_backend_dev_t other = im->meta_device(devs, (size_t) nd, split_cb, ud2);
    fprintf(out, "interned=%d distinct_ud=%d\n", again == dev, other != dev);

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t buft2 = ggml_backend_dev_buffer_type(dev);
    fprintf(out, "buft name=%s align=%zu max=%zu host=%d same=%d dev_is=%d\n", ggml_backend_buft_name(buft),
            ggml_backend_buft_get_alignment(buft), ggml_backend_buft_get_max_size(buft), ggml_backend_buft_is_host(buft),
            buft == buft2, ggml_backend_buft_get_device(buft) == dev);
    ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(dev);
    fprintf(out, "host_buft=%s\n", hb ? ggml_backend_buft_name(hb) : "(null)");

    fprintf(out, "supports_buft own=%d other_ud=%d simple=%d\n",
            ggml_backend_dev_supports_buft(dev, buft),
            ggml_backend_dev_supports_buft(dev, ggml_backend_dev_buffer_type(other)),
            ggml_backend_dev_supports_buft(dev, ggml_backend_dev_buffer_type(devs[0])));
    if (nd > 1) {
        /* a meta device over fewer devices: never compatible */
        ggml_backend_dev_t fewer = im->meta_device(devs, (size_t) nd - 1, split_cb, ud);
        fprintf(out, "supports_buft fewer=%d\n", ggml_backend_dev_supports_buft(dev, ggml_backend_dev_buffer_type(fewer)));
        /* the same devices in another order */
        ggml_backend_dev_t rev[16];
        for (int i = 0; i < nd; i++) rev[i] = devs[nd - 1 - i];
        ggml_backend_dev_t reordered = im->meta_device(rev, (size_t) nd, split_cb, ud);
        fprintf(out, "supports_buft reordered=%d reordered_same=%d\n",
                ggml_backend_dev_supports_buft(dev, ggml_backend_dev_buffer_type(reordered)), reordered == dev);
    }

    /* cross-implementation identity: neither recognises the other's objects */
    fprintf(out, "own predicates: buft=%d\n", im->buft_is_meta(buft));
    fprintf(out, "names:");
    for (int a = 0; a <= 99; a++) {
        if (a <= 3 || a == 10 || a == 11 || a == 98 || a == 99) fprintf(out, " %s", im->axis_name((enum ggml_backend_meta_split_axis) a));
    }
    fprintf(out, "\n");
}

static void dump_alloc_sizes(ggml_backend_buffer_type_t buft, weights_t * w) {
    for (int i = 0; i < w->n; i++) {
        fprintf(out, "alloc_size %s=%zu\n", w->t[i]->name, ggml_backend_buft_get_alloc_size(buft, w->t[i]));
    }
}

// ---------------------------------------------------------------------------
// Transfers

static uint8_t * host_copy[32];

static void check_roundtrip(const char * what, struct ggml_tensor * t, const uint8_t * expect) {
    const size_t nb = ggml_nbytes(t);
    uint8_t * got = malloc(nb);
    ggml_backend_tensor_get(t, got, 0, nb);
    const int ok = expect == NULL || memcmp(got, expect, nb) == 0;
    if (!ok) self_failures++;
    fprintf(out, "%s %s hash=%016llx %s\n", what, t->name, (unsigned long long) fnv1a(got, nb), ok ? "ok" : "MISMATCH");
    free(got);
}

/* True for weights get_tensor can read: anything but PARTIAL. */
static int readable(const struct ggml_tensor * t) {
    return strncmp(t->name, "part", 4) != 0;
}

static void transfers(weights_t * w) {
    /* whole-tensor set, then get must give back exactly that */
    for (int i = 0; i < w->n; i++) {
        struct ggml_tensor * t = w->t[i];
        host_copy[i] = malloc(ggml_nbytes(t));
        fill(t, host_copy[i], (uint32_t) i + 1);
        ggml_backend_tensor_set(t, host_copy[i], 0, ggml_nbytes(t));
    }
    for (int i = 0; i < w->n; i++) {
        if (readable(w->t[i])) check_roundtrip("set/get", w->t[i], host_copy[i]);
    }

    /* row-range sets: axis-0 splits take whole rows of nb[1]; segmented
     * axis-0 takes rows too; axis-1 splits take whole nb[2] slabs. */
    const char * ranged[] = { "ax0_down", "ax0nr_y", "ax0_q8", "ax1_3d", "mir_scale" };
    for (size_t r = 0; r < sizeof(ranged) / sizeof(ranged[0]); r++) {
        struct ggml_tensor * t = W(w, ranged[r]);
        int idx = 0;
        while (w->t[idx] != t) idx++;
        size_t unit, count;
        if (strncmp(t->name, "ax1", 3) == 0) { unit = t->nb[2]; count = 1; }
        else if (strncmp(t->name, "mir", 3) == 0) { unit = t->nb[0]; count = 7; }
        else { unit = t->nb[1]; count = 2; }
        const size_t off = unit * 1;
        uint8_t * patch = malloc(unit * count);
        if (t->type == GGML_TYPE_F32) {
            /* keep it finite: reuse whole floats from the original, rotated */
            for (size_t k = 0; k < unit * count / 4; k++) ((float *) patch)[k] = ((float *) host_copy[idx])[(k * 5 + 1) % (unit * count / 4)];
        } else {
            memcpy(patch, host_copy[idx], unit * count);
            for (size_t k = 0; k + 1 < unit * count; k += 2) { uint8_t x = patch[k]; patch[k] = patch[k + 1]; patch[k + 1] = x; }
            if (t->type == GGML_TYPE_Q8_0) memcpy(patch, host_copy[idx], 2); /* keep the first scale */
        }
        ggml_backend_tensor_set(t, patch, off, unit * count);
        memcpy(host_copy[idx] + off, patch, unit * count);
        free(patch);
        check_roundtrip("range-set/get", t, host_copy[idx]);
    }

    /* memset: on every readable layout, a slab, then restore */
    const char * memsets[] = { "mir_scale", "ax0_down", "ax0nr_y", "ax1nr_x", "ax1seg_qkv", "ax2_t", "ax1_3d", "ax1z_up", "ax0z_down" };
    for (size_t r = 0; r < sizeof(memsets) / sizeof(memsets[0]); r++) {
        struct ggml_tensor * t = W(w, memsets[r]);
        int idx = 0;
        while (w->t[idx] != t) idx++;
        size_t off = 0, size = ggml_nbytes(t);
        if (strncmp(t->name, "ax0", 3) == 0) { off = t->nb[1]; size = t->nb[1] * 2; }
        if (strncmp(t->name, "ax2", 3) == 0) { off = 0; size = t->nb[3]; }
        ggml_backend_tensor_memset(t, 0x3c, off, size);
        check_roundtrip("memset/get", t, NULL);
        ggml_backend_tensor_set(t, host_copy[idx], 0, ggml_nbytes(t));
        check_roundtrip("restore/get", t, host_copy[idx]);
    }
    /* memset of PARTIAL tensors to zero is the one value it allows */
    ggml_backend_tensor_memset(W(w, "part_bias"), 0, 0, ggml_nbytes(W(w, "part_bias")));
    {
        int idx = 0;
        while (strcmp(w->t[idx]->name, "part_bias") != 0) idx++;
        ggml_backend_tensor_set(w->t[idx], host_copy[idx], 0, ggml_nbytes(w->t[idx]));
    }
}

// ---------------------------------------------------------------------------
// Graphs

typedef struct {
    struct ggml_context * ctx;
    struct ggml_cgraph * gf;
    struct ggml_tensor * outs[16];
    int n_outs;
    struct ggml_tensor * ins[16];
    int n_ins;
} graph_t;

static struct ggml_tensor * input(graph_t * g, enum ggml_type type, int n, int64_t a, int64_t b, const char * name) {
    int64_t ne[2] = {a, b};
    struct ggml_tensor * t = ggml_new_tensor(g->ctx, type, n, ne);
    ggml_set_name(t, name);
    ggml_set_input(t);
    g->ins[g->n_ins++] = t;
    return t;
}

static void output(graph_t * g, struct ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);
    ggml_set_output(t);
    g->outs[g->n_outs++] = t;
    ggml_build_forward_expand(g->gf, t);
}

static void graph_begin(graph_t * g) {
    struct ggml_init_params ip = { 512 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    memset(g, 0, sizeof(*g));
    g->ctx = ggml_init(ip);
    g->gf = ggml_new_graph(g->ctx);
}

static struct ggml_tensor * norm(graph_t * g, weights_t * w, struct ggml_tensor * x) {
    return ggml_mul(g->ctx, ggml_rms_norm(g->ctx, x, 1e-6f), W(w, "mir_norm"));
}

/* Tensor-parallel MLP: a column split, a row split, one all-reduce. */
static void graph_mlp(graph_t * g, weights_t * w) {
    graph_begin(g);
    struct ggml_tensor * x = input(g, GGML_TYPE_F32, 2, E, T, "x");
    struct ggml_tensor * n = norm(g, w, x);
    struct ggml_tensor * up = ggml_mul_mat(g->ctx, W(w, "ax1_up"), n);
    struct ggml_tensor * gate = ggml_mul_mat(g->ctx, W(w, "ax1_gate"), n);
    struct ggml_tensor * h = ggml_mul(g->ctx, ggml_silu(g->ctx, gate), up);
    output(g, h, "h");
    struct ggml_tensor * o = ggml_mul_mat(g->ctx, W(w, "ax0_down"), h);
    struct ggml_tensor * r = ggml_add(g->ctx, o, x);
    output(g, r, "r");
    output(g, ggml_soft_max(g->ctx, r), "sm");
    output(g, ggml_scale(g->ctx, ggml_cont(g->ctx, ggml_transpose(g->ctx, h)), 0.5f), "hT");
    output(g, ggml_sum_rows(g->ctx, ggml_cont(g->ctx, ggml_transpose(g->ctx, h))), "hT_sum");
}

/* The MoE shape graph_compute looks for to delay an all-reduce
 * (get_i_delayed_branch, ggml-backend-meta.cpp:2010): a partial expert
 * product, an unrelated mirrored node to skip, ADD_ID with a partial bias,
 * a MUL by a mirrored scale, then one VIEW per used expert and the ADD tree
 * that sums them. The reduction moves from the expert product to the sum.
 * With `zero`, device 1 holds no slice, so the compute flags it loses are
 * carried across the delayed nodes too. */
static void graph_moe_impl(graph_t * g, weights_t * w, int zero) {
    graph_begin(g);
    struct ggml_tensor * x = input(g, GGML_TYPE_F32, 2, E, T, "x");
    struct ggml_tensor * x2 = input(g, GGML_TYPE_F32, 2, E, 3 * T, "x2");
    struct ggml_tensor * ids_in = input(g, GGML_TYPE_I32, 2, 3, T, "ids");
    struct ggml_tensor * n2 = norm(g, w, x2);
    struct ggml_tensor * h = ggml_mul_mat(g->ctx, W(w, zero ? "ax1z_up" : "ax1_gate"), n2);
    struct ggml_tensor * o = ggml_mul_mat(g->ctx, W(w, zero ? "ax0z_down" : "ax0_down"), ggml_reshape_3d(g->ctx, h, FF, 3, T));
    struct ggml_tensor * ids = ggml_cont(g->ctx, ggml_view_2d(g->ctx, ids_in, 3, T, ids_in->nb[1], 0));
    struct ggml_tensor * a = ggml_add_id(g->ctx, o, W(w, "partm_bias"), ids);
    struct ggml_tensor * m = ggml_mul(g->ctx, a, W(w, "mir_scale"));
    struct ggml_tensor * v0 = ggml_view_2d(g->ctx, m, E, T, m->nb[2], 0);
    struct ggml_tensor * v1 = ggml_view_2d(g->ctx, m, E, T, m->nb[2], m->nb[1]);
    struct ggml_tensor * v2 = ggml_view_2d(g->ctx, m, E, T, m->nb[2], 2 * m->nb[1]);
    /* expanded in order, as build_moe_ffn does, so the views are adjacent */
    ggml_build_forward_expand(g->gf, v0);
    ggml_build_forward_expand(g->gf, v1);
    ggml_build_forward_expand(g->gf, v2);
    struct ggml_tensor * sum = ggml_add(g->ctx, ggml_add(g->ctx, v0, v1), v2);
    output(g, ggml_add(g->ctx, sum, x), "r");
}
static void graph_moe(graph_t * g, weights_t * w) { graph_moe_impl(g, w, 0); }
static void graph_moe_zero(graph_t * g, weights_t * w) { graph_moe_impl(g, w, 1); }

/* Never called: a MAP_CUSTOM1 node is only allocated, not computed. */
static void custom1_noop(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * ud) {
    (void) dst; (void) a; (void) ith; (void) nth; (void) ud;
}

/* Two partial branches summed: one all-reduce after the ADD serves both
 * (get_i_delayed's second half). */
static void graph_two_partials(graph_t * g, weights_t * w) {
    graph_begin(g);
    struct ggml_tensor * x = input(g, GGML_TYPE_F32, 2, E, T, "x");
    struct ggml_tensor * n = norm(g, w, x);
    struct ggml_tensor * o1 = ggml_mul_mat(g->ctx, W(w, "ax0_down"), ggml_mul_mat(g->ctx, W(w, "ax1_up"), n));
    struct ggml_tensor * o2 = ggml_mul_mat(g->ctx, W(w, "ax0_down2"), ggml_mul_mat(g->ctx, W(w, "ax1_gate"), n));
    struct ggml_tensor * s = ggml_add(g->ctx, o1, o2);
    output(g, ggml_add(g->ctx, s, x), "r");
}

/* Device 1 holds no slice: its nodes lose the compute flag and the
 * all-reduce zeroes its partial before the butterfly. */
static void graph_zero_slice(graph_t * g, weights_t * w) {
    graph_begin(g);
    struct ggml_tensor * x = input(g, GGML_TYPE_F32, 2, E, T, "x");
    struct ggml_tensor * n = norm(g, w, x);
    struct ggml_tensor * hz = ggml_gelu(g->ctx, ggml_mul_mat(g->ctx, W(w, "ax1z_up"), n));
    struct ggml_tensor * oz = ggml_mul_mat(g->ctx, W(w, "ax0z_down"), hz);
    output(g, ggml_add(g->ctx, oz, x), "r");
}

/* Attention, head-split: rope, a KV cache written with set_rows, permutes,
 * flash attention, and the output projection's all-reduce. Heads are 32
 * wide and the split is head-aligned, so three and four devices get
 * zero-head slices. */
static void graph_attn(graph_t * g, weights_t * w) {
    graph_begin(g);
    struct ggml_tensor * x = input(g, GGML_TYPE_F32, 2, E, T, "x");
    struct ggml_tensor * pos = input(g, GGML_TYPE_I32, 1, T, 1, "pos");
    struct ggml_tensor * idx = input(g, GGML_TYPE_I64, 1, T, 1, "idx");
    struct ggml_tensor * mask = input(g, GGML_TYPE_F16, 2, KV, T, "mask");
    struct ggml_tensor * n = norm(g, w, x);

    struct ggml_tensor * q = ggml_reshape_3d(g->ctx, ggml_mul_mat(g->ctx, W(w, "ax1h_q"), n), HD, NH, T);
    q = ggml_rope(g->ctx, q, pos, HD, 0);
    struct ggml_tensor * k = ggml_mul_mat(g->ctx, W(w, "ax1h_k"), n);
    struct ggml_tensor * v = ggml_mul_mat(g->ctx, W(w, "ax1h_v"), n);
    struct ggml_tensor * kc = ggml_set_rows(g->ctx, W(w, "ax0h_kcache"), k, idx);
    struct ggml_tensor * vc = ggml_set_rows(g->ctx, W(w, "ax0h_vcache"), v, idx);
    struct ggml_tensor * k3 = ggml_permute(g->ctx, ggml_reshape_3d(g->ctx, kc, HD, NH, KV), 0, 2, 1, 3);
    struct ggml_tensor * v3 = ggml_permute(g->ctx, ggml_reshape_3d(g->ctx, vc, HD, NH, KV), 0, 2, 1, 3);
    struct ggml_tensor * qp = ggml_permute(g->ctx, q, 0, 2, 1, 3);
    struct ggml_tensor * fa = ggml_flash_attn_ext(g->ctx, qp, k3, v3, mask, 1.0f / sqrtf((float) HD), 0.0f, 0.0f);
    struct ggml_tensor * a2 = ggml_reshape_2d(g->ctx, fa, NH * HD, T);
    output(g, a2, "attn");
    struct ggml_tensor * o = ggml_mul_mat(g->ctx, W(w, "ax0h_o"), a2);
    output(g, ggml_add(g->ctx, o, x), "r");
}

/* The remaining handlers that a split tensor can go through without the
 * reference aborting: get_rows, views, cpy, pad, concat, glu, argsort, the
 * axis-2 and segmented weights, a broadcast MUL. */
static void graph_misc(graph_t * g, weights_t * w) {
    graph_begin(g);
    struct ggml_tensor * x = input(g, GGML_TYPE_F32, 2, E, T, "x");
    struct ggml_tensor * ids = input(g, GGML_TYPE_I32, 1, T, 1, "ids");
    struct ggml_tensor * bc = input(g, GGML_TYPE_F32, 2, 8, 4, "bc");
    struct ggml_tensor * n = norm(g, w, x);

    struct ggml_tensor * e = ggml_get_rows(g->ctx, W(w, "ax0h_emb"), ids);
    output(g, e, "emb");
    output(g, ggml_cont(g->ctx, ggml_view_2d(g->ctx, e, E, 2, e->nb[1], e->nb[1])), "emb_rows");
    output(g, ggml_cont(g->ctx, ggml_view_2d(g->ctx, e, E / 2, T, e->nb[1], 0)), "emb_half");
    output(g, ggml_cpy(g->ctx, e, ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, E, T)), "emb_f16");
    /* a view whose offset lies inside one split row: the C++ keeps it as is
     * rather than scaling it to the slice (split_internal_offset) */
    struct ggml_tensor * eT = ggml_cont(g->ctx, ggml_transpose(g->ctx, e));
    output(g, ggml_cont(g->ctx, ggml_view_2d(g->ctx, eT, 2, E, eT->nb[1], 2 * sizeof(float))), "embT_cols");
    output(g, ggml_pad(g->ctx, e, 0, 3, 0, 0), "emb_pad");
    output(g, ggml_concat(g->ctx, x, n, 1), "cat");
    output(g, ggml_argsort(g->ctx, ggml_cont(g->ctx, ggml_transpose(g->ctx, e)), GGML_SORT_ORDER_DESC), "argsort");
    output(g, ggml_mul_mat(g->ctx, W(w, "ax1nr_x"), n), "nr_x");
    output(g, ggml_mul(g->ctx, W(w, "ax2_t"), ggml_reshape_3d(g->ctx, bc, 8, 4, 1)), "ax2_bcast");
    output(g, ggml_sum_rows(g->ctx, ggml_scale(g->ctx, W(w, "ax2_t"), 2.0f)), "ax2_sum");
    output(g, ggml_mean(g->ctx, x), "mean");
}

/* A weight with two segments as a mul_mat source. The reference's
 * GGML_META_DEBUG logging asserts every source has one segment
 * (ggml-backend-meta.cpp:1130), so this graph runs on its own allocator
 * whose compute buffer is created with the variable unset. */
static void graph_seg2(graph_t * g, weights_t * w) {
    graph_begin(g);
    struct ggml_tensor * x = input(g, GGML_TYPE_F32, 2, E, T, "x");
    struct ggml_tensor * n = norm(g, w, x);
    output(g, ggml_swiglu(g->ctx, ggml_mul_mat(g->ctx, W(w, "ax1seg_qkv"), n)), "qkv_glu");
}

/* Allocation only, never computed: split states are derived when a tensor
 * is initialised, so allocating a graph runs every op's arm of
 * calculate_split_state without needing a kernel for it. Mostly mirrored
 * inputs through the op switch's generic and per-row arms, plus the arms of
 * the specific handlers a computed graph above does not take. */
static void graph_props(graph_t * g, weights_t * w) {
    graph_begin(g);
    struct ggml_context * c = g->ctx;
    struct ggml_tensor * a  = input(g, GGML_TYPE_F32, 2, 16, 8, "pa");
    struct ggml_tensor * b  = input(g, GGML_TYPE_F32, 2, 16, 8, "pb");
    struct ggml_tensor * v  = input(g, GGML_TYPE_F32, 2, 16, 1, "pv");
    struct ggml_tensor * sq = input(g, GGML_TYPE_F32, 2, 8, 8, "psq");
    struct ggml_tensor * ga = input(g, GGML_TYPE_F32, 2, 16, 4, "pg");
    struct ggml_tensor * id = input(g, GGML_TYPE_I32, 1, 4, 1, "pids");
    struct ggml_tensor * ei = input(g, GGML_TYPE_I32, 1, T, 1, "ids");
    struct ggml_tensor * ts = input(g, GGML_TYPE_F32, 1, 4, 1, "pts");
    struct ggml_tensor * pm = input(g, GGML_TYPE_F32, 2, E, 3, "pm");
    struct ggml_tensor * pt = input(g, GGML_TYPE_F32, 2, T, 6, "ptt");
    struct ggml_tensor * p3 = ggml_new_tensor_3d(c, GGML_TYPE_F32, 8, 3, 6);
    struct ggml_tensor * pk = ggml_new_tensor_3d(c, GGML_TYPE_F32, 3, 4, 8);
    struct ggml_tensor * pw = ggml_new_tensor_3d(c, GGML_TYPE_F32, 16, 8, 8);
    struct ggml_tensor * pr = ggml_new_tensor_2d(c, GGML_TYPE_F32, 16, 7);
    struct ggml_tensor * sx = ggml_new_tensor_3d(c, GGML_TYPE_F32, 7, 16, 1);
    struct ggml_tensor * cc = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 16);
    struct ggml_tensor * fq = ggml_new_tensor_3d(c, GGML_TYPE_F32, 32, T, 2);
    struct ggml_tensor * fk = ggml_new_tensor_3d(c, GGML_TYPE_F32, 32, KV, 2);
    struct ggml_tensor * fv = ggml_new_tensor_3d(c, GGML_TYPE_F32, 32, KV, 2);
    struct ggml_tensor * fm = ggml_new_tensor_2d(c, GGML_TYPE_F16, KV, T);
    struct ggml_tensor * dq = ggml_new_tensor_4d(c, GGML_TYPE_F32, 4, 2, 3, 1);
    struct ggml_tensor * dk = ggml_new_tensor_4d(c, GGML_TYPE_F32, 4, 2, 3, 1);
    struct ggml_tensor * dv = ggml_new_tensor_4d(c, GGML_TYPE_F32, 4, 2, 3, 1);
    struct ggml_tensor * dg = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 2, 3, 1);
    struct ggml_tensor * db = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 2, 3, 1);
    struct ggml_tensor * ds = ggml_new_tensor_4d(c, GGML_TYPE_F32, 4, 4, 2, 1);
    struct ggml_tensor * lq = ggml_new_tensor_3d(c, GGML_TYPE_F32, 8, 2, 3);
    struct ggml_tensor * lk = ggml_new_tensor_3d(c, GGML_TYPE_F32, 8, 1, 5);
    struct ggml_tensor * lw = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2, 3);
    struct ggml_tensor * lm = ggml_new_tensor_2d(c, GGML_TYPE_F16, 5, 3);
    struct ggml_tensor * extra_in[] = { p3, pk, pw, pr, sx, cc, fq, fk, fv, fm, dq, dk, dv, dg, db, ds, lq, lk, lw, lm };
    for (size_t i = 0; i < sizeof(extra_in) / sizeof(extra_in[0]); i++) ggml_set_input(extra_in[i]);

#define P(name, expr) do { struct ggml_tensor * t_ = (expr); ggml_set_name(t_, name); ggml_build_forward_expand(g->gf, t_); } while (0)
    /* the op switch, on mirrored sources */
    P("dup", ggml_dup(c, a));
    P("sub", ggml_sub(c, a, b));
    P("div", ggml_div(c, a, b));
    P("sqr", ggml_sqr(c, a));
    P("sqrt", ggml_sqrt(c, a));
    P("log", ggml_log(c, a));
    P("sin", ggml_sin(c, a));
    P("cos", ggml_cos(c, a));
    P("sum", ggml_sum(c, a));
    P("cumsum", ggml_cumsum(c, a));
    P("argmax", ggml_argmax(c, a));
    P("count_equal", ggml_count_equal(c, a, b));
    P("repeat", ggml_repeat(c, v, a));
    P("repeat_back", ggml_repeat_back(c, a, v));
    P("silu_back", ggml_silu_back(c, a, b));
    P("norm", ggml_norm(c, a, 1e-5f));
    P("group_norm", ggml_group_norm(c, a, 2, 1e-5f));
    P("l2_norm", ggml_l2_norm(c, a, 1e-5f));
    P("rms_norm_back", ggml_rms_norm_back(c, a, b, 1e-5f));
    P("out_prod", ggml_out_prod(c, a, b));
    P("set_1d", ggml_set_1d(c, a, v, 0));
    P("acc", ggml_acc(c, a, v, a->nb[1], a->nb[2], a->nb[3], 0));
    P("permute_mir", ggml_cont(c, ggml_permute(c, a, 1, 0, 2, 3)));
    P("get_rows_back", ggml_get_rows_back(c, ga, id, a));
    P("diag", ggml_diag(c, v));
    P("diag_mask_inf", ggml_diag_mask_inf(c, sq, 0));
    P("diag_mask_zero", ggml_diag_mask_zero(c, sq, 0));
    P("soft_max_back", ggml_soft_max_ext_back(c, a, b, 1.0f, 0.0f));
    P("clamp", ggml_clamp(c, a, -1.0f, 1.0f));
    P("upscale", ggml_upscale(c, a, 2, GGML_SCALE_MODE_NEAREST));
    P("pad_reflect", ggml_pad_reflect_1d(c, a, 1, 1));
    P("roll", ggml_roll(c, a, 1, 0, 0, 0));
    P("timestep", ggml_timestep_embedding(c, ts, 16, 10000));
    P("top_k", ggml_top_k(c, a, 4));
    P("leaky_relu", ggml_leaky_relu(c, a, 0.1f, false));
    P("tri", ggml_tri(c, sq, GGML_TRI_TYPE_LOWER));
    P("fill", ggml_fill(c, a, 1.0f));
    P("pool_1d", ggml_pool_1d(c, a, GGML_OP_POOL_AVG, 2, 2, 0));
    P("pool_2d", ggml_pool_2d(c, a, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0));
    P("conv_t_1d", ggml_conv_transpose_1d(c, pk, a, 1, 0, 1));
    P("win_part", ggml_win_part(c, pw, 4));
    P("get_rel_pos", ggml_get_rel_pos(c, pr, 4, 4));
    P("xent", ggml_cross_entropy_loss(c, a, b));
    P("ssm_conv_mir", ggml_ssm_conv(c, sx, cc));
    P("fa_mir", ggml_flash_attn_ext(c, fq, fk, fv, fm, 0.125f, 0.0f, 0.0f));
    P("gdn_mir", ggml_gated_delta_net(c, dq, dk, dv, dg, db, ds, 1));
    P("lightning_mir", ggml_lightning_indexer(c, lq, lk, lw, lm));
    P("view_mir", ggml_cont(c, ggml_view_2d(c, a, 4, 4, a->nb[1] * 2, 0)));
    P("transpose_mir", ggml_cont(c, ggml_transpose(c, a)));
    P("cpy_mir", ggml_cpy(c, a, ggml_new_tensor_2d(c, GGML_TYPE_F16, 16, 8)));
    P("add1", ggml_add1(c, a, ggml_new_tensor_1d(c, GGML_TYPE_F32, 1)));
    P("custom1", ggml_map_custom1(c, a, custom1_noop, 1, NULL));
    {
        struct ggml_tensor * rpos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
        P("rope_back", ggml_rope_ext_back(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, 8, 2, 4), rpos, NULL, 8, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
    }

    /* the specific handlers' other arms, on split sources */
    struct ggml_tensor * e = ggml_get_rows(c, W(w, "ax0h_emb"), ei);
    P("concat_mir_split", ggml_concat(c, pm, e, 1));
    P("concat_split_mir", ggml_concat(c, e, pm, 1));
    P("concat_split_split", ggml_concat(c, e, e, 1));
    struct ggml_tensor * e2 = ggml_cont(c, ggml_transpose(c, e));
    P("mm_mir_ax1", ggml_mul_mat(c, pt, e2));
    P("mm_ax2_ax2", ggml_mul_mat(c, W(w, "ax2_t"), W(w, "ax2_t")));
    P("mm_ax2_mir", ggml_mul_mat(c, W(w, "ax2_t"), p3));
    P("reshape_rows", ggml_cont(c, ggml_reshape_3d(c, W(w, "ax0z_down"), FF, 1, E)));
    P("reshape_last", ggml_cont(c, ggml_reshape_3d(c, W(w, "ax1_gate"), E, FF, 1)));
    P("transpose_ax2", ggml_cont(c, ggml_transpose(c, W(w, "ax2_t"))));
    P("permute_ax2", ggml_cont(c, ggml_permute(c, W(w, "ax2_t"), 2, 0, 1, 3)));
    P("sum_rows_ax1", ggml_sum_rows(c, e2));
    P("norm_ax1", ggml_norm(c, e2, 1e-5f));
    P("argsort_ax1", ggml_argsort(c, e2, GGML_SORT_ORDER_ASC));
#undef P
}

static void fill_inputs(graph_t * g) {
    for (int i = 0; i < g->n_ins; i++) {
        struct ggml_tensor * t = g->ins[i];
        const size_t nb = ggml_nbytes(t);
        uint8_t * d = malloc(nb);
        if (strcmp(t->name, "ids") == 0) {
            for (int64_t k = 0; k < ggml_nelements(t); k++) ((int32_t *) d)[k] = (int32_t) ((k * 3 + 1) % 4);
        } else if (strcmp(t->name, "pos") == 0) {
            for (int64_t k = 0; k < ggml_nelements(t); k++) ((int32_t *) d)[k] = (int32_t) k + 2;
        } else if (strcmp(t->name, "idx") == 0) {
            for (int64_t k = 0; k < ggml_nelements(t); k++) ((int64_t *) d)[k] = (k * 3 + 2) % KV;
        } else if (strcmp(t->name, "mask") == 0) {
            for (int64_t r = 0; r < t->ne[1]; r++) for (int64_t c = 0; c < t->ne[0]; c++)
                ((ggml_fp16_t *) d)[r * t->ne[0] + c] = ggml_fp32_to_fp16(c <= r + 3 ? 0.0f : -INFINITY);
        } else {
            fill(t, d, 77u + (uint32_t) i);
        }
        ggml_backend_tensor_set(t, d, 0, nb);
        free(d);
    }
}

/* Runs one graph through the meta backend: allocate, fill, compute, read. */
static void run_graph(const char * label, graph_t * g, ggml_backend_t backend, ggml_backend_dev_t dev, ggml_gallocr_t galloc, int repeat_uid) {
    if (backend == NULL) {
        /* allocation only: the SPLIT_STATE log is the observable */
        if (!ggml_gallocr_alloc_graph(galloc, g->gf)) {
            fprintf(out, "%s alloc failed\n", label);
            return;
        }
        for (int i = 0; i < ggml_graph_n_nodes(g->gf); i++) {
            struct ggml_tensor * node = ggml_graph_node(g->gf, i);
            fprintf(out, "%s node %d %s %s [%lld,%lld,%lld,%lld]\n", label, i, ggml_op_desc(node), node->name,
                    (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
        }
        fprintf(out, "%s compute buffer=%zu\n", label, ggml_gallocr_get_buffer_size(galloc, 0));
        return;
    }
    int supported = 1;
    for (int i = 0; i < ggml_graph_n_nodes(g->gf); i++) {
        struct ggml_tensor * node = ggml_graph_node(g->gf, i);
        const int s = ggml_backend_dev_supports_op(dev, node);
        fprintf(out, "%s node %d %s %s supports=%d\n", label, i, ggml_op_desc(node), node->name, s);
        supported &= s;
    }
    if (!supported) {
        fprintf(out, "%s skipped: an op is unsupported\n", label);
        return;
    }
    if (!ggml_gallocr_alloc_graph(galloc, g->gf)) {
        fprintf(out, "%s alloc failed\n", label);
        return;
    }
    fprintf(out, "%s compute buffer=%zu\n", label, ggml_gallocr_get_buffer_size(galloc, 0));
    uint64_t first[16] = {0};
    for (int rep = 0; rep < (repeat_uid ? 2 : 1); rep++) {
        /* every rep: the allocator may reuse an input's memory once its last
         * consumer has run, so inputs do not survive a compute */
        fill_inputs(g);
        /* A defined uid lets the second compute skip the rebuild. */
        if (repeat_uid) g->gf->uid = 4242;
        for (int i = 0; i < g->n_ins; i++) {
            struct ggml_tensor * t = g->ins[i];
            uint8_t * d = malloc(ggml_nbytes(t));
            ggml_backend_tensor_get(t, d, 0, ggml_nbytes(t));
            fprintf(out, "%s rep %d in %s hash=%016llx\n", label, rep, t->name, (unsigned long long) fnv1a(d, ggml_nbytes(t)));
            free(d);
        }
        const enum ggml_status st = ggml_backend_graph_compute(backend, g->gf);
        fprintf(out, "%s rep %d status=%d\n", label, rep, (int) st);
        for (int i = 0; i < g->n_outs; i++) {
            struct ggml_tensor * t = g->outs[i];
            const size_t nb = ggml_nbytes(t);
            uint8_t * d = malloc(nb);
            ggml_backend_tensor_get(t, d, 0, nb);
            const uint64_t h = fnv1a(d, nb);
            /* the second rep reuses the first's subgraphs (same uid, no
             * rebuild) and must compute exactly the same */
            const int same = rep == 0 || h == first[i];
            if (!same) self_failures++;
            if (rep == 0) first[i] = h;
            fprintf(out, "%s rep %d out %s [%lld,%lld,%lld] hash=%016llx%s\n", label, rep, t->name,
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (unsigned long long) h,
                    same ? "" : " CHANGED-ON-REUSE");
            free(d);
        }
    }
}

/* The async pair, on an input (mirrored) and an output of each layout. */
static void run_async(graph_t * g, ggml_backend_t backend) {
    struct ggml_tensor * x = g->ins[0];
    const size_t nb = ggml_nbytes(x);
    uint8_t * d = malloc(nb);
    fill(x, d, 999u);
    ggml_backend_tensor_set_async(backend, x, d, 0, nb);
    ggml_backend_synchronize(backend);
    uint8_t * back = malloc(nb);
    ggml_backend_tensor_get_async(backend, x, back, 0, nb);
    ggml_backend_synchronize(backend);
    const int ok = memcmp(d, back, nb) == 0;
    if (!ok) self_failures++;
    fprintf(out, "async x roundtrip %s\n", ok ? "ok" : "MISMATCH");
    free(d); free(back);
    for (int i = 0; i < g->n_outs; i++) {
        struct ggml_tensor * t = g->outs[i];
        const size_t n = ggml_nbytes(t);
        uint8_t * a = malloc(n);
        uint8_t * s = malloc(n);
        ggml_backend_tensor_get_async(backend, t, a, 0, n);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t, s, 0, n);
        const int same = memcmp(a, s, n) == 0;
        if (!same) self_failures++;
        fprintf(out, "async get %s hash=%016llx %s\n", t->name, (unsigned long long) fnv1a(a, n), same ? "ok" : "MISMATCH");
        /* and write it back asynchronously: set_tensor_async */
        ggml_backend_tensor_set_async(backend, t, s, 0, n);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t, a, 0, n);
        const int same2 = memcmp(a, s, n) == 0;
        if (!same2) self_failures++;
        fprintf(out, "async set %s %s\n", t->name, same2 ? "ok" : "MISMATCH");
        free(a); free(s);
    }
}

// ---------------------------------------------------------------------------
// One implementation over one device combination

static void run_impl(const impl_t * im, const char * combo, FILE * f) {
    out = f;
    ggml_backend_dev_t devs[16];
    int nd = 0;
    for (const char * p = combo; *p; p++) {
        devs[nd++] = ggml_backend_dev_by_type(*p == 'm' ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    static split_ud uds[64][2];
    static int n_uds = 0;
    split_ud * ud = &uds[n_uds][0];
    split_ud * ud2 = &uds[n_uds][1];
    n_uds++;
    ud->nd = nd; ud->tag = 1;
    ud2->nd = nd; ud2->tag = 2;

    fprintf(out, "== combo %s\n", combo);
    ggml_backend_dev_t dev = im->meta_device(devs, (size_t) nd, split_cb, ud);
    dump_device(im, dev, devs, nd, ud, ud2);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    weights_t w;
    build_weights(&w);
    dump_alloc_sizes(buft, &w);
    w.buf = im->alloc_ctx(w.ctx, buft);
    fprintf(out, "weights buffer name=%s size=%zu host=%d is_meta=%d align=%zu\n", ggml_backend_buffer_name(w.buf),
            ggml_backend_buffer_get_size(w.buf), ggml_backend_buffer_is_host(w.buf), im->buffer_is_meta(w.buf),
            ggml_backend_buffer_get_alignment(w.buf));
    for (int i = 0; i < w.n; i++) {
        fprintf(out, "weight %s data=%p buffer_is_weights=%d\n", w.t[i]->name, w.t[i]->data, w.t[i]->buffer == w.buf);
    }
    transfers(&w);

    if (strspn(combo, "c") == strlen(combo)) {
        /* CPU-only: the reference's graph_compute crashes here (its
         * `s_copy_main` FIXME skips views of host-resident leaves), so only
         * the device answers and the transfers are compared. This is where
         * the buffer type is host and the simple buffers are plain memory. */
        fprintf(out, "cpu-only: graphs skipped\n");
        ggml_backend_buffer_free(w.buf);
        ggml_free(w.ctx);
        for (int i = 0; i < w.n; i++) { free(host_copy[i]); host_copy[i] = NULL; }
        fflush(out);
        out = NULL;
        return;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    fprintf(out, "backend name=%s is_meta=%d n=%zu guid=%016llx\n", ggml_backend_name(backend), im->is_meta(backend),
            im->n_backends(backend), (unsigned long long) fnv1a(ggml_backend_guid(backend), 16));
    for (size_t i = 0; i < im->n_backends(backend); i++) {
        fprintf(out, "simple backend %zu %s\n", i, ggml_backend_name(im->simple_backend(backend, i)));
    }
    fprintf(out, "default buft same=%d alignment=%zu\n", ggml_backend_get_default_buffer_type(backend) == buft,
            ggml_backend_get_alignment(backend));

    /* Two backends, because of an upstream use-after-free the sequence must
     * stay clear of: when a graph raises max_nnodes, graph_compute frees its
     * graph context and re-creates only the first n_subgraphs subgraphs
     * (ggml-backend-meta.cpp:2223), so a later graph with more subgraphs
     * runs on a cgraph in freed memory. Run here as mlp, ..., attn, misc,
     * mlp-again on one backend, the port crashed and the reference did not
     * -- heap luck, not a difference. Backend A only ever sees graphs no
     * larger than its first, with at most as many subgraphs; backend B's
     * graph that raises the count (misc) has one subgraph and nothing after
     * it has more. */
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    enum { NG = 7, NA = 5 };
    graph_t gs[NG];
    void (*builders[NG])(graph_t *, weights_t *) = { graph_mlp, graph_moe, graph_moe_zero, graph_two_partials, graph_zero_slice, graph_attn, graph_misc };
    const char * labels[NG] = { "mlp", "moe", "moe_zero", "two_partials", "zero_slice", "attn", "misc" };
    ggml_backend_t backend_b = ggml_backend_dev_init(dev, NULL);
    for (int i = 0; i < NG; i++) {
        builders[i](&gs[i], &w);
        run_graph(labels[i], &gs[i], i < NA ? backend : backend_b, dev, galloc, i == 0);
        if (i == 0) run_async(&gs[0], backend);
        if (i == NG - 2) run_async(&gs[i], backend_b);
    }
    {
        graph_t gp;
        graph_props(&gp, &w);
        ggml_gallocr_t galloc_p = ggml_gallocr_new(buft);
        run_graph("props", &gp, NULL, dev, galloc_p, 0);
        ggml_gallocr_free(galloc_p);
        ggml_free(gp.ctx);
    }
    {
        graph_t g2;
        graph_seg2(&g2, &w);
        unsetenv("GGML_META_DEBUG");
        ggml_gallocr_t galloc2 = ggml_gallocr_new(buft);
        run_graph("seg2", &g2, backend_b, dev, galloc2, 0);
        setenv("GGML_META_DEBUG", "1", 1);
        ggml_gallocr_free(galloc2);
        ggml_free(g2.ctx);
    }
    /* the first graph rebuilt from scratch after the others: a rebuild after
     * several container flips. (Re-running gs[0] itself is not possible: its
     * tensors keep their addresses, so the allocator skips them, and their
     * slices lived in a compute container that has since been reset -- the
     * reference aborts on it exactly as the port does.) */
    graph_t again;
    graph_mlp(&again, &w);
    run_graph("mlp-again", &again, backend, dev, galloc, 0);
    /* the cache after the attention graph wrote it */
    check_roundtrip("after-compute", W(&w, "ax0h_kcache"), NULL);

    ggml_backend_buffer_clear(w.buf, 0);
    for (int i = 0; i < w.n; i++) {
        if (readable(w.t[i])) check_roundtrip("cleared", w.t[i], NULL);
    }

    ggml_gallocr_free(galloc);
    for (int i = 0; i < NG; i++) ggml_free(gs[i].ctx);
    ggml_free(again.ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend);
    ggml_backend_buffer_free(w.buf);
    ggml_free(w.ctx);
    for (int i = 0; i < w.n; i++) { free(host_copy[i]); host_copy[i] = NULL; }
    fflush(out);
    out = NULL;
}

/* A crash in either implementation should say where, not just "139". */
static void on_crash(int sig) {
    void * frames[64];
    const int n = backtrace(frames, 64);
    fprintf(stderr, "meta_diff: signal %d\n", sig);
    backtrace_symbols_fd(frames, n, 2);
    _exit(128 + sig);
}

int main(int argc, char ** argv) {
    signal(SIGSEGV, on_crash);
    signal(SIGBUS, on_crash);
    signal(SIGABRT, on_crash);
    setenv("GGML_META_DEBUG", "1", 1);
    ggml_log_set(on_log, NULL);

    static const char * defaults[] = { "m", "mm", "mc", "cm", "mmm", "mcm", "mmmm", "c", "cc", "ccc" };
    const char ** combos = defaults;
    int n_combos = (int) (sizeof(defaults) / sizeof(defaults[0]));
    if (argc > 1) { combos = (const char **) (argv + 1); n_combos = argc - 1; }

    /* META_DIFF_TRACE=ref|port: run one side only, transcript on stderr as it
     * is written -- for finding where a run aborts. */
    const char * trace = getenv("META_DIFF_TRACE");
    if (trace) {
        for (int ci = 0; ci < n_combos; ci++) {
            run_impl(strcmp(trace, "port") == 0 ? &IMPL_PORT : &IMPL_REF, combos[ci], stderr);
        }
        return 0;
    }

    long checks = 0;
    int failed = 0;
    for (int ci = 0; ci < n_combos; ci++) {
        char * rbuf = NULL; size_t rlen = 0;
        char * pbuf = NULL; size_t plen = 0;
        FILE * rf = open_memstream(&rbuf, &rlen);
        FILE * pf = open_memstream(&pbuf, &plen);
        run_impl(&IMPL_REF, combos[ci], rf);
        run_impl(&IMPL_PORT, combos[ci], pf);
        fclose(rf);
        fclose(pf);

        /* line by line; the weight `data=` pointers are fixed sentinels on
         * both sides, so even they compare */
        char * rs = rbuf; char * ps = pbuf;
        long lines = 0, splits = 0;
        int diverged = 0;
        while (*rs || *ps) {
            char * re = strchr(rs, '\n'); char * pe = strchr(ps, '\n');
            const size_t rl = re ? (size_t) (re - rs) : strlen(rs);
            const size_t pl = pe ? (size_t) (pe - ps) : strlen(ps);
            if (rl != pl || memcmp(rs, ps, rl) != 0) {
                printf("  %-5s DIFFERS at line %ld:\n    ref:  %.*s\n    port: %.*s\n", combos[ci], lines + 1, (int) rl, rs, (int) pl, ps);
                diverged = 1;
                break;
            }
            if (memmem(rs, rl, "SPLIT_STATE", 11)) splits++;
            lines++;
            rs = re ? re + 1 : rs + rl;
            ps = pe ? pe + 1 : ps + pl;
        }
        if (!diverged) {
            printf("  %-5s %6ld lines identical (%ld SPLIT_STATE)\n", combos[ci], lines, splits);
        }
        if (getenv("META_DIFF_OUT")) {
            char path[512];
            snprintf(path, sizeof(path), "%s/ref_%s.txt", getenv("META_DIFF_OUT"), combos[ci]);
            FILE * f = fopen(path, "w"); if (f) { fwrite(rbuf, 1, rlen, f); fclose(f); }
            snprintf(path, sizeof(path), "%s/port_%s.txt", getenv("META_DIFF_OUT"), combos[ci]);
            f = fopen(path, "w"); if (f) { fwrite(pbuf, 1, plen, f); fclose(f); }
        }
        checks += lines;
        failed |= diverged;
        free(rbuf);
        free(pbuf);
    }
    if (self_failures) {
        printf("  %d known-answer checks failed (a set/get round trip did not return what was written)\n", self_failures);
        failed = 1;
    }
    printf("%s: %ld transcript lines compared\n", failed ? "FAIL" : "PASS", checks);
    return failed;
}
