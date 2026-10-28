/*
 * The Vulkan backend's registry and device vtables, asked the same questions
 * on the port and on the reference: `cuda-backend-diff` for Vulkan. **Not a
 * port.** See scripts/vk-backend-diff.
 *
 * One binary links both -- the port through `ggml_backend_vk_*`, the
 * reference through `ref_ggml_backend_vk_*` (upstream's `ggml-vulkan.cpp`
 * with its exports renamed, in `libggml-vulkan-ref.a`) -- but each run asks
 * **one** side, named by `argv[1]`, and prints every answer and every log
 * line. The instance initialises once per process and prints while it does,
 * and some of the C++'s messages go straight to `stderr`; only a process of
 * its own makes all of that attributable to one side. The script runs each
 * side under each environment and diffs the transcripts.
 *
 * Asked so far -- what the port has: the registry's name, device count and
 * proc addresses; per device, name, description, type, props (total memory
 * only: free memory moves under any run), `offload_op` on both sides of
 * `op_offload_min_batch_size` for every arm of `ggml_vk_get_op_batch_size`;
 * the exported description at buffer sizes that truncate; and
 * `vk_conv_block_sizes`. Each later layer adds its questions here.
 *
 * The logical device: per device, the buffer type's name, alignment, max
 * size, `alloc_size` and `is_host`, `supports_buft` against both devices'
 * buffer types and the CPU's; then a buffer, round-tripped through
 * `set_tensor` and `get_tensor` (whole, at an offset, and 2-D), cleared,
 * `memset_tensor` and `cpy_tensor`ed, with every byte read back hashed. The
 * host buffer type is asked the same of device 0. Creating the device runs
 * `ggml_vk_get_device` and the pipeline walk on both sides.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <math.h>

struct conv_block_size { uint32_t K, NPQ, CRS; };

ggml_backend_reg_t ref_ggml_backend_vk_reg(void);
bool ref_ggml_backend_is_vk(ggml_backend_t backend);
static int g_port;
static bool is_vk(ggml_backend_t be) { return g_port ? ggml_backend_is_vk(be) : ref_ggml_backend_is_vk(be); }
int ref_ggml_backend_vk_get_device_count(void);
void ref_ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
void ref_ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);
extern struct conv_block_size ref_vk_conv_block_sizes[4];
extern struct conv_block_size vk_conv_block_sizes[4];

static void log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    printf("log %d %s", (int) level, text);
}

static uint64_t fnv(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* Every byte of a buffer, through a tensor spanning it. */
static uint64_t read_all(ggml_backend_buffer_t buf, struct ggml_tensor * t, uint8_t * scratch, size_t n) {
    memset(scratch, 0xa5, n);
    ggml_backend_tensor_get(t, scratch, 0, n);
    (void) buf;
    return fnv(scratch, n);
}

static void ask_buffers(size_t i, ggml_backend_dev_t dev, ggml_backend_reg_t reg, struct ggml_context * ctx) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    struct ggml_tensor * probe = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 3);
    printf("dev %zu buft name = %s alignment = %zu max_size = %zu alloc_size = %zu is_host = %d\n", i,
           ggml_backend_buft_name(buft), ggml_backend_buft_get_alignment(buft), ggml_backend_buft_get_max_size(buft),
           ggml_backend_buft_get_alloc_size(buft, probe), ggml_backend_buft_is_host(buft));
    for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); j++) {
        printf("dev %zu supports_buft(dev %zu buft) = %d\n", i, j,
               ggml_backend_dev_supports_buft(dev, ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, j))));
    }
    printf("dev %zu supports_buft(cpu) = %d\n", i, ggml_backend_dev_supports_buft(dev, ggml_backend_cpu_buffer_type()));

    const size_t n = 1 << 16;
    uint8_t * src = malloc(n), * scratch = malloc(n);
    for (size_t k = 0; k < n; k++) src[k] = (uint8_t) (k * 131 + 7);

    ggml_backend_buffer_t a = ggml_backend_buft_alloc_buffer(buft, n);
    ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, n);
    printf("dev %zu buffer = %s size = %zu base = %p name = %s is_host = %d\n", i, a ? "yes" : "null",
           a ? ggml_backend_buffer_get_size(a) : 0, a ? ggml_backend_buffer_get_base(a) : NULL,
           a ? ggml_backend_buffer_name(a) : "", a ? ggml_backend_buffer_is_host(a) : 0);
    if (a && b) {
        struct ggml_tensor * ta = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, n);
        struct ggml_tensor * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, n);
        ggml_backend_tensor_alloc(a, ta, ggml_backend_buffer_get_base(a));
        ggml_backend_tensor_alloc(b, tb, ggml_backend_buffer_get_base(b));

        ggml_backend_tensor_set(ta, src, 0, n);
        printf("dev %zu set/get = %016llx\n", i, (unsigned long long) read_all(a, ta, scratch, n));
        ggml_backend_tensor_set(ta, src + 100, 4096, 1000);
        printf("dev %zu set at offset = %016llx\n", i, (unsigned long long) read_all(a, ta, scratch, n));
        memset(scratch, 0, 64);
        ggml_backend_tensor_get(ta, scratch, 4090, 64);
        printf("dev %zu get at offset = %016llx\n", i, (unsigned long long) fnv(scratch, 64));
        ggml_backend_tensor_set_2d(ta, src, 128, 32, 5, 256, 64);
        printf("dev %zu set_2d = %016llx\n", i, (unsigned long long) read_all(a, ta, scratch, n));
        memset(scratch, 0, 5 * 48);
        ggml_backend_tensor_get_2d(ta, scratch, 300, 40, 5, 200, 48);
        printf("dev %zu get_2d = %016llx\n", i, (unsigned long long) fnv(scratch, 5 * 48));
        ggml_backend_tensor_memset(ta, 0x3c, 1000, 2000);
        printf("dev %zu memset_tensor = %016llx\n", i, (unsigned long long) read_all(a, ta, scratch, n));
        ggml_backend_buffer_clear(b, 0x5a);
        printf("dev %zu clear = %016llx\n", i, (unsigned long long) read_all(b, tb, scratch, n));
        ggml_backend_tensor_copy(ta, tb);
        printf("dev %zu cpy_tensor = %016llx\n", i, (unsigned long long) read_all(b, tb, scratch, n));
    }
    if (a) ggml_backend_buffer_free(a);
    if (b) ggml_backend_buffer_free(b);

    if (i == 0) {
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(dev);
        printf("host buft name = %s alignment = %zu max_size = %zu is_host = %d alloc_size = %zu\n",
               ggml_backend_buft_name(host), ggml_backend_buft_get_alignment(host), ggml_backend_buft_get_max_size(host),
               ggml_backend_buft_is_host(host), ggml_backend_buft_get_alloc_size(host, probe));
        ggml_backend_buffer_t h = ggml_backend_buft_alloc_buffer(host, n);
        printf("host buffer = %s size = %zu is_host = %d name = %s\n", h ? "yes" : "null", h ? ggml_backend_buffer_get_size(h) : 0,
               h ? ggml_backend_buffer_is_host(h) : 0, h ? ggml_backend_buffer_name(h) : "");
        if (h) {
            uint8_t * p = ggml_backend_buffer_get_base(h);
            memcpy(p, src, n);
            printf("host buffer readback = %016llx\n", (unsigned long long) fnv(p, n));
            /* A transfer whose host side is pinned memory copies through it
             * instead of the staging buffer, when the device buffer is not
             * mapped. */
            ggml_backend_buffer_t d = ggml_backend_buft_alloc_buffer(buft, n / 2);
            if (d) {
                struct ggml_tensor * td = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, n / 2);
                ggml_backend_tensor_alloc(d, td, ggml_backend_buffer_get_base(d));
                ggml_backend_tensor_set(td, p + 1000, 0, n / 2);
                memset(p, 0, n);
                ggml_backend_tensor_get(td, p + 64, 0, n / 2);
                printf("pinned set/get = %016llx\n", (unsigned long long) fnv(p, n));
                ggml_backend_tensor_set_2d(td, p + 7, 16, 100, 9, 300, 333);
                ggml_backend_tensor_get_2d(td, p + 5, 20, 90, 9, 300, 120);
                printf("pinned 2d = %016llx\n", (unsigned long long) fnv(p, n));
                ggml_backend_buffer_free(d);
            }
            ggml_backend_buffer_free(h);
        }
    }
    free(src);
    free(scratch);
}

/* The backend object: init, the async transfers through a backend
 * (contexts, submission, synchronize), events, and host-pointer buffers. */
static void ask_backend(size_t i, ggml_backend_dev_t dev, struct ggml_context * ctx) {
    ggml_backend_t be = ggml_backend_dev_init(dev, NULL);
    printf("dev %zu backend = %s name = %s is_vk = %d default_buft = %s\n", i, be ? "yes" : "null",
           be ? ggml_backend_name(be) : "", be ? is_vk(be) : 0,
           be ? ggml_backend_buft_name(ggml_backend_get_default_buffer_type(be)) : "");
    if (!be) return;
    const size_t n = 1 << 15;
    uint8_t * src = malloc(n), * dst = malloc(n);
    for (size_t k = 0; k < n; k++) src[k] = (uint8_t) (k * 37 + 3);
    ggml_backend_buffer_t a = ggml_backend_alloc_buffer(be, n);
    ggml_backend_buffer_t b = ggml_backend_alloc_buffer(be, n);
    struct ggml_tensor * ta = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, n);
    struct ggml_tensor * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, n);
    ggml_backend_tensor_alloc(a, ta, ggml_backend_buffer_get_base(a));
    ggml_backend_tensor_alloc(b, tb, ggml_backend_buffer_get_base(b));

    ggml_backend_tensor_set_async(be, ta, src, 0, n);
    ggml_backend_tensor_set_async(be, ta, src + 9, 500, 3000);
    ggml_backend_tensor_set_2d_async(be, ta, src, 64, 20, 7, 100, 30);
    ggml_backend_synchronize(be);
    memset(dst, 0, n);
    ggml_backend_tensor_get_async(be, ta, dst, 0, n);
    ggml_backend_synchronize(be);
    printf("dev %zu async set/get = %016llx\n", i, (unsigned long long) fnv(dst, n));
    memset(dst, 0, n);
    ggml_backend_tensor_get_2d_async(be, ta, dst, 40, 50, 9, 300, 60);
    ggml_backend_synchronize(be);
    printf("dev %zu async get_2d = %016llx\n", i, (unsigned long long) fnv(dst, n));

    /* The same through pinned memory: an async transfer whose host side
     * is the backend's host buffer copies through it, not the staging
     * buffer. */
    ggml_backend_buffer_t hbuf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(dev), n);
    if (hbuf) {
        uint8_t * hp = ggml_backend_buffer_get_base(hbuf);
        memset(hp, 0, n);
        ggml_backend_tensor_get_2d_async(be, ta, hp, 40, 50, 9, 300, 60);
        ggml_backend_synchronize(be);
        printf("dev %zu pinned async get_2d = %016llx\n", i, (unsigned long long) fnv(hp, n));
        for (size_t k = 0; k < n; k++) hp[k] = (uint8_t) (k * 11 + 5);
        ggml_backend_tensor_set_2d_async(be, tb, hp, 32, 70, 11, 200, 90);
        ggml_backend_synchronize(be);
        memset(dst, 0, n);
        ggml_backend_tensor_get(tb, dst, 0, n);
        printf("dev %zu pinned async set_2d = %016llx\n", i, (unsigned long long) fnv(dst, n));
        /* A host tensor in pinned memory copied asynchronously to the device. */
        struct ggml_tensor * th = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, n / 2);
        struct ggml_tensor * tbh = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, n / 2);
        ggml_backend_tensor_alloc(hbuf, th, hp + n / 4);
        ggml_backend_tensor_alloc(b, tbh, (char *) ggml_backend_buffer_get_base(b) + n / 8);
        ggml_backend_tensor_copy_async(be, be, th, tbh);
        ggml_backend_synchronize(be);
        memset(dst, 0, n);
        ggml_backend_tensor_get(tb, dst, 0, n);
        printf("dev %zu pinned cpy_tensor_async = %016llx\n", i, (unsigned long long) fnv(dst, n));
        ggml_backend_buffer_free(hbuf);
    }

    ggml_backend_tensor_copy_async(be, be, ta, tb);
    ggml_backend_synchronize(be);
    memset(dst, 0, n);
    ggml_backend_tensor_get(tb, dst, 0, n);
    printf("dev %zu cpy_tensor_async = %016llx\n", i, (unsigned long long) fnv(dst, n));

    /* An event recorded after a write, waited on, and synchronized. */
    ggml_backend_event_t ev = ggml_backend_event_new(dev);
    printf("dev %zu event = %s\n", i, ev ? "yes" : "null");
    if (ev) {
        ggml_backend_event_synchronize(ev);   /* never recorded: a no-op */
        ggml_backend_tensor_set_async(be, tb, src + 77, 0, 1000);
        ggml_backend_event_record(ev, be);
        ggml_backend_event_wait(be, ev);
        ggml_backend_event_synchronize(ev);
        ggml_backend_event_record(ev, be);
        ggml_backend_event_synchronize(ev);
        ggml_backend_synchronize(be);
        memset(dst, 0, n);
        ggml_backend_tensor_get(tb, dst, 0, n);
        printf("dev %zu after events = %016llx\n", i, (unsigned long long) fnv(dst, n));
        ggml_backend_event_free(ev);
    }

    /* Host memory imported as a device buffer, if the device can. */
    void * host = NULL;
    if (posix_memalign(&host, 65536, 1 << 16) == 0) {
        memcpy(host, src, 1 << 15);
        ggml_backend_buffer_t hb = ggml_backend_dev_buffer_from_host_ptr(dev, host, 1 << 16, 1 << 16);
        printf("dev %zu buffer_from_host_ptr = %s\n", i, hb ? "yes" : "null");
        if (hb) {
            struct ggml_tensor * th = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 1 << 15);
            ggml_backend_tensor_alloc(hb, th, ggml_backend_buffer_get_base(hb));
            memset(dst, 0, n);
            ggml_backend_tensor_get(th, dst, 0, 1 << 15);
            printf("dev %zu host ptr readback = %016llx\n", i, (unsigned long long) fnv(dst, n));
            ggml_backend_buffer_free(hb);
        }
        /* A misaligned pointer is refused. */
        ggml_backend_buffer_t hm = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) host + 64, 4096, 4096);
        printf("dev %zu buffer_from_host_ptr misaligned = %s\n", i, hm ? "yes" : "null");
        if (hm) ggml_backend_buffer_free(hm);
        free(host);
    }

    ggml_backend_buffer_free(a);
    ggml_backend_buffer_free(b);
    ggml_backend_free(be);
    free(src);
    free(dst);
}

static void ask_offload(ggml_backend_dev_t dev, struct ggml_context * ctx) {
    /* One tensor per arm of `ggml_vk_get_op_batch_size`, its batch
     * dimension at 31 and 32 -- either side of the default minimum. */
    for (int64_t n = 31; n <= 32; n++) {
        struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 64, 4);
        struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, n, 4);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n);
        struct ggml_tensor * bid = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 2, n);
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 2, n);
        struct ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        struct ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        struct ggml_tensor * add = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, n);
        struct ggml_tensor * ops[] = {
            ggml_mul_mat(ctx, a, b),
            ggml_mul_mat_id(ctx, a, bid, ids),
            ggml_rope(ctx, x, pos, 64, 0),
            ggml_rope_ext_back(ctx, x, pos, NULL, 64, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
            ggml_get_rows(ctx, table, rows),
            ggml_add(ctx, add, add),
        };
        for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++) {
            printf("offload_op %s n=%lld = %d\n", ggml_op_desc(ops[i]), (long long) n, ggml_backend_dev_offload_op(dev, ops[i]));
        }
    }
}

/* ---------------------------------------------------------------------------
 * graph_optimize. test-backend-ops never calls it -- it computes graphs
 * directly -- and node-diff sees it only through hashes after the reorder.
 * Here both backends reorder the same graphs and the new order is printed
 * as original indices. No allocation: the reorder reads ops, sources and
 * view sources, never data. */

/* The scheduler's allocation-dependency callback (ggml-backend.cpp, since
 * v0.4.0). Vulkan ignores it on both sides; printing each call makes a side
 * that starts using it differ from one that does not. */
static void record_alloc_dep(void * user_data, struct ggml_tensor * tensor, struct ggml_tensor * until) {
    (void) user_data;
    printf("add_alloc_dep %s until %s\n", tensor->name, until->name);
}

static void print_order(size_t i, const char * name, struct ggml_cgraph * g, ggml_backend_t be) {
    const int n = ggml_graph_n_nodes(g);
    struct ggml_tensor ** orig = (struct ggml_tensor **) malloc(sizeof(*orig) * (size_t) n);
    for (int k = 0; k < n; k++) orig[k] = ggml_graph_node(g, k);
    struct ggml_backend_graph_optimize_params params = { record_alloc_dep, NULL };
    be->iface.graph_optimize(be, g, &params);
    printf("dev %zu graph_optimize %s (%d nodes):", i, name, n);
    for (int k = 0; k < n; k++) {
        int at = -1;
        for (int j = 0; j < n; j++) if (orig[j] == ggml_graph_node(g, k)) { at = j; break; }
        printf(" %d", at);
    }
    printf("\n");
    free(orig);
}

/* Two decoder layers: rms_norm*w feeding q/k/v projections, rope on q and
 * k, k written into a cache through view+set_rows, attention, residual
 * add, rms_norm*w, gated FFN, matmul+add+add bias chain, residual add. */
static struct ggml_cgraph * graph_decoder(struct ggml_context * ctx) {
    const int64_t d = 64, nh = 4, hd = 16, nt = 3, ff = 128;
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, nt);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
    struct ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
    for (int l = 0; l < 2; l++) {
        struct ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d));
        struct ggml_tensor * q = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, d), h);
        struct ggml_tensor * k = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, d), h);
        struct ggml_tensor * v = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, d), h);
        /* QK-norm, both norms expanded before either rope, so each rope
         * sits a few nodes past its MUL: the reorder has to find it. */
        q = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, q, hd, nh, nt), 1e-6f), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, k, hd, nh, nt), 1e-6f), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd));
        ggml_build_forward_expand(g, q);
        ggml_build_forward_expand(g, k);
        q = ggml_rope(ctx, q, pos, (int) hd, GGML_ROPE_TYPE_NEOX);
        k = ggml_rope(ctx, k, pos, (int) hd, GGML_ROPE_TYPE_NEOX);
        struct ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, 16);
        struct ggml_tensor * kc = ggml_set_rows(ctx, cache, ggml_view_2d(ctx, k, d, nt, k->nb[2], 0), rows);
        struct ggml_tensor * att = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, kc, d, 16), ggml_reshape_2d(ctx, q, d, nt)), NULL, 0.125f, 0.0f);
        ggml_build_forward_expand(g, att);
        /* A second reader of the cache through its own view: it depends on
         * the set_rows only through their shared view_src. */
        struct ggml_tensor * q2 = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, nt), 0.5f);
        struct ggml_tensor * att2 = ggml_mul_mat(ctx, ggml_view_2d(ctx, cache, d, 16, cache->nb[1], 0), q2);
        struct ggml_tensor * o = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, d), att), v),
                                          ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, d), att2));
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, d), o));
        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d));
        struct ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, ff), h);
        struct ggml_tensor * up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, ff), h);
        struct ggml_tensor * f = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ff, d), ggml_swiglu_split(ctx, gate, up));
        f = ggml_add(ctx, ggml_add(ctx, f, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, nt)), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, nt));
        x = ggml_add(ctx, x, f);
    }
    ggml_build_forward_expand(g, x);
    return g;
}

/* llama's MoE router (soft_max, argsort top-k, get_rows, sum_rows, clamp,
 * div: the early_softmax_norm pattern), expert matmuls with add_id and a
 * scale, a snake chain, and ssm_conv + add + silu -- interleaved with
 * independent work so the reorder has something to move. */
static struct ggml_cgraph * graph_patterns(struct ggml_context * ctx) {
    const int64_t n_exp = 8, k_used = 2, nt = 4, d = 32;
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, nt);
    struct ggml_tensor * side = ggml_sqr(ctx, x);
    struct ggml_tensor * logits = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_exp), x);
    struct ggml_tensor * probs = ggml_soft_max(ctx, logits);
    /* A zero-row GET_ROWS, as a single-token decode of a recurrent model
     * has: ggml_vk_is_empty counts it empty, the reorder's own is_empty
     * does not, and swapping the two moved it among the views. */
    ggml_build_forward_expand(g, ggml_scale(ctx, ggml_get_rows(ctx, x, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 0)), 2.0f));
    struct ggml_tensor * sel = ggml_argsort_top_k(ctx, probs, (int) k_used);
    struct ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_exp, nt), sel);
    w = ggml_reshape_2d(ctx, w, k_used, nt);
    w = ggml_div(ctx, w, ggml_clamp(ctx, ggml_sum_rows(ctx, w), 6.103515625e-5f, INFINITY));
    w = ggml_reshape_3d(ctx, w, 1, k_used, nt);
    side = ggml_add(ctx, side, x);
    struct ggml_tensor * e = ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, d, n_exp), ggml_reshape_3d(ctx, x, d, 1, nt), sel);
    e = ggml_add_id(ctx, e, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_exp), sel);
    e = ggml_mul(ctx, e, w);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nt);
    struct ggml_tensor * ib = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nt);
    struct ggml_tensor * sn = ggml_add(ctx, side, ggml_mul(ctx, ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, side, a))), ib));
    struct ggml_tensor * conv = ggml_ssm_conv(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 + nt, d, 1), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, d));
    conv = ggml_silu(ctx, ggml_add(ctx, conv, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d)));
    ggml_build_forward_expand(g, ggml_add(ctx, ggml_sum_rows(ctx, e), ggml_reshape_3d(ctx, ggml_sum_rows(ctx, sn), 1, 1, nt)));
    ggml_build_forward_expand(g, conv);
    return g;
}

/* A seeded random DAG: each node is an elementwise op, an rms_norm, a view
 * or a reshape of one or two recent tensors, so dependencies reach back a
 * few nodes -- within and beyond the reorder's 20-node window. */
static struct ggml_cgraph * graph_random(struct ggml_context * ctx, uint32_t seed, int count) {
    uint32_t s = seed;
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    enum { POOL = 64 };
    struct ggml_tensor * pool[POOL];
    int np = 0;
    for (int k = 0; k < 4; k++) pool[np++] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
    for (int k = 0; k < count; k++) {
        s = s * 1664525u + 1013904223u;
        const uint32_t r = s >> 8;
        struct ggml_tensor * a = pool[np - 1 - (int) (r % (uint32_t) (np < 12 ? np : 12))];
        struct ggml_tensor * b = pool[np - 1 - (int) ((r >> 8) % (uint32_t) (np < 30 ? np : 30))];
        struct ggml_tensor * t;
        switch ((r >> 16) % 7) {
            case 0:  t = ggml_add(ctx, a, b); break;
            case 1:  t = ggml_mul(ctx, a, b); break;
            case 2:  t = ggml_rms_norm(ctx, a, 1e-6f); break;
            case 3:  t = ggml_view_2d(ctx, a, 16, 4, a->nb[1], 0); break;
            case 4:  t = ggml_reshape_2d(ctx, a, 16, 4); break;
            case 5:  t = ggml_scale(ctx, a, 0.5f); break;
            default: t = ggml_sqr(ctx, a); break;
        }
        if (np == POOL) { memmove(pool, pool + 1, sizeof(pool[0]) * (POOL - 1)); np--; }
        pool[np++] = t;
        if ((r & 7) == 0) ggml_build_forward_expand(g, t);
    }
    ggml_build_forward_expand(g, pool[np - 1]);
    return g;
}

/* A serial chain long enough that nothing past it is in reach, then a
 * snake chain on its last node, then a node that depends on that node
 * too. Only keeping the snake whole stops the reorder from pulling the
 * independent node into the middle of the chain. */
static struct ggml_cgraph * graph_snake_tail(struct ggml_context * ctx) {
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
    struct ggml_tensor * t = x;
    for (int k = 0; k < 24; k++) t = ggml_add(ctx, ggml_sqr(ctx, t), x);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 4);
    struct ggml_tensor * ib = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 4);
    ggml_build_forward_expand(g, ggml_add(ctx, t, ggml_mul(ctx, ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, t, a))), ib)));
    ggml_build_forward_expand(g, ggml_scale(ctx, t, 2.0f));
    return g;
}

static void ask_graph_optimize(size_t i, ggml_backend_dev_t dev) {
    ggml_backend_t be = ggml_backend_dev_init(dev, NULL);
    if (be == NULL) return;
    struct ggml_init_params params = { 4096 * ggml_tensor_overhead() + 8 * ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    print_order(i, "decoder", graph_decoder(ctx), be);
    print_order(i, "patterns", graph_patterns(ctx), be);
    print_order(i, "snake_tail", graph_snake_tail(ctx), be);
    for (uint32_t seed = 1; seed <= 4; seed++) {
        char name[32];
        snprintf(name, sizeof(name), "random%u", seed);
        print_order(i, name, graph_random(ctx, seed, 150), be);
    }
    ggml_free(ctx);
    ggml_backend_free(be);
}

int main(int argc, char ** argv) {
    if (argc != 2 || (strcmp(argv[1], "port") != 0 && strcmp(argv[1], "ref") != 0)) {
        fprintf(stderr, "usage: vk_backend_diff port|ref\n");
        return 2;
    }
    const int port = strcmp(argv[1], "port") == 0;
    g_port = port;
    ggml_log_set(log_cb, NULL);
    setvbuf(stdout, NULL, _IONBF, 0);

    ggml_backend_reg_t reg = port ? ggml_backend_vk_reg() : ref_ggml_backend_vk_reg();
    printf("reg = %s\n", reg ? "yes" : "null");
    if (reg == NULL) {
        return 0;
    }
    printf("reg name = %s\n", ggml_backend_reg_name(reg));
    const size_t n_dev = ggml_backend_reg_dev_count(reg);
    printf("reg dev_count = %zu\n", n_dev);
    static const char * procs[] = { "ggml_backend_get_features", "ggml_backend_split_buffer_type", "no_such_function" };
    for (size_t i = 0; i < sizeof(procs) / sizeof(procs[0]); i++) {
        printf("reg proc %s = %s\n", procs[i], ggml_backend_reg_get_proc_address(reg, procs[i]) ? "yes" : "null");
    }

    const int count = port ? ggml_backend_vk_get_device_count() : ref_ggml_backend_vk_get_device_count();
    printf("get_device_count = %d\n", count);

    struct ggml_init_params params = { 1024 * ggml_tensor_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        printf("dev %zu name = %s\n", i, ggml_backend_dev_name(dev));
        printf("dev %zu description = %s\n", i, ggml_backend_dev_description(dev));
        printf("dev %zu type = %d\n", i, (int) ggml_backend_dev_type(dev));
        struct ggml_backend_dev_props props;
        memset(&props, 0, sizeof(props));
        ggml_backend_dev_get_props(dev, &props);
        printf("dev %zu props name = %s description = %s type = %d device_id = %s total = %zu\n", i,
               props.name, props.description, (int) props.type, props.device_id ? props.device_id : "(null)", props.memory_total);
        printf("dev %zu caps async=%d host_buffer=%d buffer_from_host_ptr=%d events=%d mmap=%d\n", i,
               props.caps.async, props.caps.host_buffer, props.caps.buffer_from_host_ptr, props.caps.events, props.caps.mmap_support);
        size_t free = 0, total = 0;
        if (port) ggml_backend_vk_get_device_memory((int) i, &free, &total);
        else      ref_ggml_backend_vk_get_device_memory((int) i, &free, &total);
        printf("dev %zu get_device_memory total = %zu free<=total = %d\n", i, total, free <= total);

        static const size_t sizes[] = { 1, 2, 7, 12, 256 };
        for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
            char buf[256];
            memset(buf, 'x', sizeof(buf));
            if (port) ggml_backend_vk_get_device_description((int) i, buf, sizes[s]);
            else      ref_ggml_backend_vk_get_device_description((int) i, buf, sizes[s]);
            buf[sizeof(buf) - 1] = 0;
            printf("dev %zu get_device_description(%zu) = [%s]\n", i, sizes[s], buf);
        }
        ask_offload(dev, ctx);
    }
    if (getenv("VK_DIFF_NO_DEVICE") == NULL) {
        for (size_t i = 0; i < n_dev; i++) {
            ask_buffers(i, ggml_backend_reg_dev_get(reg, i), reg, ctx);
            ask_backend(i, ggml_backend_reg_dev_get(reg, i), ctx);
            ask_graph_optimize(i, ggml_backend_reg_dev_get(reg, i));
        }
    }
    ggml_free(ctx);

    const struct conv_block_size * cbs = port ? vk_conv_block_sizes : ref_vk_conv_block_sizes;
    for (int i = 0; i < 4; i++) {
        printf("vk_conv_block_sizes[%d] = %u %u %u\n", i, cbs[i].K, cbs[i].NPQ, cbs[i].CRS);
    }
    return 0;
}
