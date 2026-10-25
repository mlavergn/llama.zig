// The ported BLAS backend (`src/ggml/blas.zig`) against upstream's
// `ggml-blas.cpp`, in one process -- `make blas-diff`.
//
// `scripts/blas-diff` compiles upstream's translation unit with its four
// exports renamed `ref_*` and links it beside a `-Dblas -Dblas-port` libggml,
// so both backends live in this process under different names, over the same
// Accelerate and the same ggml core. Then:
//
// - **The vtables.** Every answer the registration, the device and a backend
//   give: names, description, type, memory, every capability, the procedure
//   address table, the buffer type, `supports_buft`, `buffer_from_host_ptr`,
//   `is_blas` both ways, the GUID.
// - **`supports_op`**, swept: `MUL_MAT` for every type, at sizes straddling
//   the 32 the C++ requires of `ne0`, `ne1` and `ne10`, with F32 and F16
//   `src1`, contiguous and not, with and without the Hadamard hint;
//   `OUT_PROD` over types, ranks and `src1` layouts; the view ops; and ops it
//   must refuse. Both answers, true and false, must occur.
// - **Computation, on bits.** Every type with a `to_float`, broadcast in both
//   batch dimensions, at one thread and at four -- four splits the
//   dequantization across `std::async` tasks, one split per plane -- and
//   `OUT_PROD` with a plain and a transposed `src1`. Each case runs on the
//   reference, then on the port, into a buffer refilled with a sentinel
//   between them; the port's output must equal the reference's bit for bit.
//   A case only counts if the reference overwrote every sentinel and lands
//   near the CPU backend's answer -- so a graph neither side computed cannot
//   pass by agreeing.
//
// Prints one line per difference and a verdict; exits 1 on any difference.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-blas.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ggml_backend_t     ref_ggml_backend_blas_init(void);
bool               ref_ggml_backend_is_blas(ggml_backend_t backend);
void               ref_ggml_backend_blas_set_n_threads(ggml_backend_t backend, int n_threads);
ggml_backend_reg_t ref_ggml_backend_blas_reg(void);

static long checks;
static long failures;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        checks++;                                           \
        if (!(cond)) {                                      \
            failures++;                                     \
            if (failures <= 40) {                           \
                printf("DIFF ");                            \
                printf(__VA_ARGS__);                        \
                printf("\n");                               \
            }                                               \
        }                                                   \
    } while (0)

static int streq(const char * a, const char * b) {
    if (a == NULL || b == NULL) return a == b;
    return strcmp(a, b) == 0;
}

// ---------------------------------------------------------------------------
// The vtables

static void vtables(void) {
    ggml_backend_reg_t rr = ref_ggml_backend_blas_reg();
    ggml_backend_reg_t pr = ggml_backend_blas_reg();

    CHECK(streq(ggml_backend_reg_name(rr), ggml_backend_reg_name(pr)), "reg name: %s vs %s", ggml_backend_reg_name(rr), ggml_backend_reg_name(pr));
    CHECK(ggml_backend_reg_dev_count(rr) == ggml_backend_reg_dev_count(pr), "device count");
    CHECK(rr->api_version == pr->api_version, "api version");

    ggml_backend_dev_t rd = ggml_backend_reg_dev_get(rr, 0);
    ggml_backend_dev_t pd = ggml_backend_reg_dev_get(pr, 0);
    CHECK(rd->reg == rr && pd->reg == pr, "device reg back-pointer");
    CHECK(rd->context == NULL && pd->context == NULL, "device context");

    CHECK(streq(ggml_backend_dev_name(rd), ggml_backend_dev_name(pd)), "device name: %s vs %s", ggml_backend_dev_name(rd), ggml_backend_dev_name(pd));
    CHECK(streq(ggml_backend_dev_description(rd), ggml_backend_dev_description(pd)), "description: %s vs %s", ggml_backend_dev_description(rd), ggml_backend_dev_description(pd));
    CHECK(ggml_backend_dev_type(rd) == ggml_backend_dev_type(pd), "device type");

    size_t rf = 1, rt = 1, pf = 2, pt = 2;
    ggml_backend_dev_memory(rd, &rf, &rt);
    ggml_backend_dev_memory(pd, &pf, &pt);
    CHECK(rf == pf && rt == pt, "memory: %zu/%zu vs %zu/%zu", rf, rt, pf, pt);

    struct ggml_backend_dev_props rp, pp;
    memset(&rp, 0xa5, sizeof(rp));
    memset(&pp, 0x5a, sizeof(pp));
    ggml_backend_dev_get_props(rd, &rp);
    ggml_backend_dev_get_props(pd, &pp);
    CHECK(streq(rp.name, pp.name) && streq(rp.description, pp.description), "props name/description");
    CHECK(rp.type == pp.type, "props type");
    CHECK(rp.memory_free == pp.memory_free && rp.memory_total == pp.memory_total, "props memory");
    CHECK(rp.caps.async == pp.caps.async, "caps.async");
    CHECK(rp.caps.host_buffer == pp.caps.host_buffer, "caps.host_buffer");
    CHECK(rp.caps.buffer_from_host_ptr == pp.caps.buffer_from_host_ptr, "caps.buffer_from_host_ptr");
    CHECK(rp.caps.events == pp.caps.events, "caps.events");
    CHECK(rp.caps.mmap_support == pp.caps.mmap_support, "caps.mmap_support");

    const char * procs[] = { "ggml_backend_set_n_threads", "ggml_backend_get_features", "ggml_backend_dev_get_extra_bufts", "nonsense" };
    for (size_t i = 0; i < sizeof(procs) / sizeof(procs[0]); i++) {
        void * a = ggml_backend_reg_get_proc_address(rr, procs[i]);
        void * b = ggml_backend_reg_get_proc_address(pr, procs[i]);
        CHECK((a == NULL) == (b == NULL), "proc address %s: %p vs %p", procs[i], a, b);
        if (i == 0) {
            CHECK(a == (void *) ref_ggml_backend_blas_set_n_threads, "reference proc address is its own set_n_threads");
            CHECK(b == (void *) ggml_backend_blas_set_n_threads, "port proc address is its own set_n_threads");
        }
    }

    CHECK(ggml_backend_dev_buffer_type(rd) == ggml_backend_cpu_buffer_type(), "reference buffer type is the CPU's");
    CHECK(ggml_backend_dev_buffer_type(pd) == ggml_backend_cpu_buffer_type(), "port buffer type is the CPU's");
    CHECK(rd->iface.get_host_buffer_type == NULL && pd->iface.get_host_buffer_type == NULL, "no host buffer type");
    CHECK((rd->iface.offload_op == NULL) == (pd->iface.offload_op == NULL), "offload_op");
    CHECK((rd->iface.event_new == NULL) == (pd->iface.event_new == NULL), "event_new");

    // supports_buft: the CPU's (host), and every other device's buffer type.
    CHECK(ggml_backend_dev_supports_buft(rd, ggml_backend_cpu_buffer_type()) == ggml_backend_dev_supports_buft(pd, ggml_backend_cpu_buffer_type()), "supports_buft CPU");
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_dev_get(i));
        CHECK(ggml_backend_dev_supports_buft(rd, buft) == ggml_backend_dev_supports_buft(pd, buft), "supports_buft %s", ggml_backend_buft_name(buft));
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_dev_get(i));
        if (host) {
            CHECK(ggml_backend_dev_supports_buft(rd, host) == ggml_backend_dev_supports_buft(pd, host), "supports_buft host %s", ggml_backend_buft_name(host));
        }
    }

    // `ggml_backend_cpu_buffer_from_ptr` asserts the pointer is aligned.
    static float host[1024] __attribute__((aligned(64)));
    ggml_backend_buffer_t rb = ggml_backend_dev_buffer_from_host_ptr(rd, host, sizeof(host), 64);
    ggml_backend_buffer_t pb = ggml_backend_dev_buffer_from_host_ptr(pd, host, sizeof(host), 64);
    CHECK(rb && pb, "buffer_from_host_ptr");
    if (rb && pb) {
        CHECK(ggml_backend_buffer_get_base(rb) == ggml_backend_buffer_get_base(pb), "host buffer base");
        CHECK(ggml_backend_buffer_get_size(rb) == ggml_backend_buffer_get_size(pb), "host buffer size");
        CHECK(ggml_backend_buffer_get_type(rb) == ggml_backend_buffer_get_type(pb), "host buffer type");
        CHECK(ggml_backend_buffer_is_host(rb) == ggml_backend_buffer_is_host(pb), "host buffer is_host");
        ggml_backend_buffer_free(rb);
        ggml_backend_buffer_free(pb);
    }

    ggml_backend_t rbk = ggml_backend_dev_init(rd, NULL);
    ggml_backend_t pbk = ggml_backend_dev_init(pd, "ignored");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(rbk && pbk, "init_backend");
    CHECK(streq(ggml_backend_name(rbk), ggml_backend_name(pbk)), "backend name");
    CHECK(ggml_backend_get_device(rbk) == rd && ggml_backend_get_device(pbk) == pd, "backend device");
    CHECK(memcmp(ggml_backend_guid(rbk), ggml_backend_guid(pbk), sizeof(ggml_guid)) == 0, "guid");
    CHECK(ref_ggml_backend_is_blas(rbk) == ggml_backend_is_blas(pbk), "is_blas own");
    CHECK(ref_ggml_backend_is_blas(pbk) == ggml_backend_is_blas(rbk), "is_blas crosswise");
    CHECK(ref_ggml_backend_is_blas(cpu) == ggml_backend_is_blas(cpu), "is_blas CPU");
    CHECK(ref_ggml_backend_is_blas(NULL) == ggml_backend_is_blas(NULL), "is_blas NULL");
    CHECK(ggml_backend_is_blas(pbk) && !ggml_backend_is_blas(cpu), "is_blas answers at all");
    struct ggml_backend_i ri = rbk->iface, pi = pbk->iface;
    CHECK((ri.synchronize == NULL) == (pi.synchronize == NULL), "synchronize");
    CHECK((ri.graph_plan_create == NULL) == (pi.graph_plan_create == NULL), "graph_plan_create");
    CHECK((ri.set_tensor_async == NULL) == (pi.set_tensor_async == NULL), "set_tensor_async");
    CHECK((ri.graph_optimize == NULL) == (pi.graph_optimize == NULL), "graph_optimize");
    CHECK((ri.event_record == NULL) == (pi.event_record == NULL), "event_record");
    ggml_backend_free(rbk);
    ggml_backend_free(pbk);
    ggml_backend_free(cpu);
}

// ---------------------------------------------------------------------------
// supports_op

static long n_supported;
static long n_refused;

static void ask(struct ggml_tensor * op, const char * what) {
    ggml_backend_dev_t rd = ggml_backend_reg_dev_get(ref_ggml_backend_blas_reg(), 0);
    ggml_backend_dev_t pd = ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0);
    bool a = ggml_backend_dev_supports_op(rd, op);
    bool b = ggml_backend_dev_supports_op(pd, op);
    CHECK(a == b, "supports_op %s (%s, ne %lld %lld %lld %lld): reference %d, port %d", what, ggml_op_desc(op),
          (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3], a, b);
    if (a) n_supported++; else n_refused++;
}

static void supports_ops(void) {
    struct ggml_init_params ip = { 256u * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t sizes[] = { 31, 32, 33 };
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const enum ggml_type type = (enum ggml_type) t;
        if (ggml_type_size(type) == 0 || ggml_blck_size(type) == 0) continue;
        const int64_t blck = ggml_blck_size(type);
        // k: straddles 32 where the block allows, else one and two blocks
        const int64_t ks[3] = { blck == 1 ? 31 : blck, blck == 1 ? 32 : 2 * blck, blck == 1 ? 33 : 3 * blck };
        for (int ik = 0; ik < 3; ik++) {
            for (int i0 = 0; i0 < 3; i0++) {
                for (int i1 = 0; i1 < 3; i1++) {
                    for (int s1 = 0; s1 < 2; s1++) {
                        const int64_t k = ks[ik], m = sizes[i0], n = sizes[i1];
                        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
                        struct ggml_tensor * b = ggml_new_tensor_2d(ctx, s1 ? GGML_TYPE_F16 : GGML_TYPE_F32, k, n);
                        struct ggml_tensor * op = ggml_mul_mat(ctx, a, b);
                        ask(op, ggml_type_name(type));
                        if (s1 == 0 && ik == 1 && i0 == 1 && i1 == 1) {
                            op->op_params[1] = GGML_HINT_SRC0_IS_HADAMARD;
                            ask(op, "hadamard hint");
                            op->op_params[1] = 0;
                        }
                    }
                }
            }
        }
    }

    // Non-contiguous operands, F32 at sizes it would otherwise take.
    {
        struct ggml_tensor * a  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 64, 2);
        struct ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 64, 2);
        struct ggml_tensor * av = ggml_view_2d(ctx, a, 64, 32, a->nb[1] * 2, 0);
        struct ggml_tensor * bp = ggml_permute(ctx, b, 0, 2, 1, 3);
        ask(ggml_mul_mat(ctx, a, b), "contiguous 3d");
        ask(ggml_mul_mat(ctx, av, ggml_view_2d(ctx, b, 64, 64, b->nb[1], 0)), "strided src0");
        ask(ggml_mul_mat(ctx, ggml_view_3d(ctx, a, 64, 64, 2, a->nb[1], a->nb[2], 0), bp), "permuted src1");
        struct ggml_tensor * big = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 64, 3, 2);
        ask(ggml_mul_mat(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 64, 1, 2), big), "broadcast");
    }

    // OUT_PROD: types, ranks, src1 layouts.
    const enum ggml_type op_types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 };
    for (int ta = 0; ta < 3; ta++) {
        for (int tb = 0; tb < 2; tb++) {
            const int64_t k = 64;
            struct ggml_tensor * a = ggml_new_tensor_2d(ctx, op_types[ta], k, 48);
            struct ggml_tensor * b = ggml_new_tensor_2d(ctx, op_types[tb], 40, 48);
            ask(ggml_out_prod(ctx, a, b), "out_prod matrix");
            struct ggml_tensor * bt = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, op_types[tb], 48, 40));
            ask(ggml_out_prod(ctx, a, bt), "out_prod transposed src1");
            struct ggml_tensor * a3 = ggml_new_tensor_3d(ctx, op_types[ta], k, 48, 2);
            struct ggml_tensor * b3 = ggml_new_tensor_3d(ctx, op_types[tb], 40, 48, 2);
            ask(ggml_out_prod(ctx, a3, b3), "out_prod 3d");
            // A matrix src0 broadcast over a 3-d and a 4-d src1, so each
            // operand's own rank check is the one that decides.
            ask(ggml_out_prod(ctx, a, b3), "out_prod 3d src1");
            ask(ggml_out_prod(ctx, a, ggml_new_tensor_4d(ctx, op_types[tb], 40, 48, 1, 2)), "out_prod 4d src1");
            struct ggml_tensor * bs = ggml_view_2d(ctx, ggml_new_tensor_2d(ctx, op_types[tb], 80, 48), 40, 48, 80 * ggml_type_size(op_types[tb]), 0);
            ask(ggml_out_prod(ctx, a, bs), "out_prod strided src1");
        }
    }

    // The view ops it accepts, and ops it must refuse.
    {
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ask(x, "leaf");
        ask(ggml_reshape_1d(ctx, x, 64 * 64), "reshape");
        ask(ggml_view_1d(ctx, x, 32, 0), "view");
        ask(ggml_permute(ctx, x, 1, 0, 2, 3), "permute");
        ask(ggml_transpose(ctx, x), "transpose");
        ask(ggml_add(ctx, x, x), "add");
        ask(ggml_soft_max(ctx, x), "soft_max");
        ask(ggml_cont(ctx, ggml_transpose(ctx, x)), "cont");
        ask(ggml_mul_mat_id(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 64, 4), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 2, 32),
                            ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 32)), "mul_mat_id");
    }

    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// Computation

static uint64_t rng_state = 0x2545F4914F6CDD1DULL;

static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float) ((rng_state >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

static void fill_f32(float * x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = frand();
}

// Fills `t` from random floats, quantized by ggml's own quantizer -- the same
// bytes reach both backends, which is all a comparison needs.
static void fill_tensor(struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    float * f = malloc(n * sizeof(float));
    fill_f32(f, n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(t->data, f, n * sizeof(float));
    } else {
        const int64_t row = t->ne[0];
        float * imatrix = malloc(row * sizeof(float));
        for (int64_t i = 0; i < row; i++) imatrix[i] = 1.0f;
        ggml_quantize_chunk(t->type, f, t->data, 0, n / row, row, imatrix);
        free(imatrix);
    }
    free(f);
}

static long n_computed;

// Runs `out` on `backend` after refilling it with `sentinel`.
static void run(ggml_backend_t backend, struct ggml_cgraph * gf, struct ggml_tensor * out, uint32_t sentinel) {
    uint32_t * d = out->data;
    const int64_t n = ggml_nelements(out);
    for (int64_t i = 0; i < n; i++) d[i] = sentinel;
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    CHECK(st == GGML_STATUS_SUCCESS, "graph_compute status %d", st);
}

// One case: the reference's result must be real, and the port's must equal it.
static void compare(struct ggml_context * ctx, struct ggml_tensor * out, const char * what, int n_threads) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t ref = ref_ggml_backend_blas_init();
    ggml_backend_t port = ggml_backend_blas_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ref_ggml_backend_blas_set_n_threads(ref, n_threads);
    ggml_backend_blas_set_n_threads(port, n_threads);

    const int64_t n = ggml_nelements(out);
    float * want = malloc(n * sizeof(float));
    float * near = malloc(n * sizeof(float));

    run(cpu, gf, out, 0x7fc00001u);
    memcpy(near, out->data, n * sizeof(float));

    run(ref, gf, out, 0x7fc00002u);
    memcpy(want, out->data, n * sizeof(float));
    // Real, not merely equal: every sentinel overwritten, and near the CPU
    // backend by normalized mean squared error -- loose, because the CPU
    // quantizes src1 to 8 bits for a quantized src0 and BLAS does not.
    long unwritten = 0;
    double err = 0.0, norm = 0.0;
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &want[i], 4);
        if (bits == 0x7fc00002u) unwritten++;
        err += ((double) want[i] - near[i]) * ((double) want[i] - near[i]);
        norm += (double) near[i] * near[i];
    }
    CHECK(unwritten == 0, "%s: the reference left %ld of %lld outputs unwritten", what, unwritten, (long long) n);
    CHECK(norm > 0.0 && err / norm < 5e-3, "%s: the reference is far from the CPU backend (NMSE %g)", what, norm > 0.0 ? err / norm : -1.0);

    run(port, gf, out, 0x7fc00003u);
    long differ = 0;
    int64_t first = -1;
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(&want[i], (float *) out->data + i, 4) != 0) {
            if (first < 0) first = i;
            differ++;
        }
    }
    CHECK(differ == 0, "%s, %d threads: %ld of %lld outputs differ, first at %lld (%a vs %a)", what, n_threads, differ, (long long) n,
          (long long) first, first >= 0 ? want[first] : 0.0, first >= 0 ? ((float *) out->data)[first] : 0.0);
    n_computed++;

    free(want);
    free(near);
    ggml_backend_free(ref);
    ggml_backend_free(port);
    ggml_backend_free(cpu);
}

static void mul_mat_case(enum ggml_type type, int64_t k, int64_t m, int64_t n, int64_t ne02, int64_t ne03, int64_t r2, int64_t r3, int n_threads) {
    struct ggml_init_params ip = { 512u * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, type, k, m, ne02, ne03);
    struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, n, ne02 * r2, ne03 * r3);
    fill_tensor(a);
    fill_tensor(b);
    struct ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    char what[128];
    snprintf(what, sizeof(what), "mul_mat %s k=%lld m=%lld n=%lld planes=%lldx%lld bcast=%lldx%lld", ggml_type_name(type),
             (long long) k, (long long) m, (long long) n, (long long) ne02, (long long) ne03, (long long) r2, (long long) r3);
    compare(ctx, out, what, n_threads);
    ggml_free(ctx);
}

static void out_prod_case(int64_t n_, int64_t k, int64_t m, bool transposed) {
    struct ggml_init_params ip = { 64u * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_, k);
    struct ggml_tensor * b;
    if (transposed) {
        struct ggml_tensor * bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
        fill_tensor(bt);
        b = ggml_transpose(ctx, bt);
    } else {
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, k);
        fill_tensor(b);
    }
    fill_tensor(a);
    struct ggml_tensor * out = ggml_out_prod(ctx, a, b);
    char what[96];
    snprintf(what, sizeof(what), "out_prod n=%lld k=%lld m=%lld%s", (long long) n_, (long long) k, (long long) m, transposed ? " transposed src1" : "");
    compare(ctx, out, what, 1);
    ggml_free(ctx);
}

static void computations(void) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const enum ggml_type type = (enum ggml_type) t;
        if (ggml_type_size(type) == 0 || ggml_blck_size(type) == 0) continue;
        if (type != GGML_TYPE_F32 && ggml_get_type_traits(type)->to_float == NULL) continue;
        // Types ggml can quantize to, which is how the inputs are made.
        if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 &&
            ggml_get_type_traits(type)->from_float_ref == NULL && !ggml_quantize_requires_imatrix(type)) {
            continue;
        }
        const int64_t blck = ggml_blck_size(type);
        const int64_t k = blck >= 256 ? 256 : 128;
        // One plane, then broadcast across both batch dimensions; one
        // thread, then four, which splits each plane's dequantization.
        mul_mat_case(type, k, 64, 40, 1, 1, 1, 1, 1);
        mul_mat_case(type, k, 64, 40, 2, 1, 2, 3, 4);
        mul_mat_case(type, k, 96, 33, 1, 2, 1, 1, 4);
    }
    // The dequantization split's own arithmetic: rows not divisible by the
    // thread count, and a k so wide each thread takes one row.
    mul_mat_case(GGML_TYPE_Q4_0, 4096, 37, 32, 1, 1, 1, 1, 4);
    mul_mat_case(GGML_TYPE_Q8_0, 64, 130, 35, 1, 1, 1, 1, 3);

    out_prod_case(48, 64, 40, false);
    out_prod_case(48, 64, 40, true);
    out_prod_case(33, 17, 65, false);
    out_prod_case(33, 17, 65, true);
}

static void quiet(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

int main(void) {
    ggml_log_set(quiet, NULL);
    ggml_cpu_init();

    vtables();
    supports_ops();
    computations();

    printf("blas-diff: %ld checks; supports_op answered %ld true and %ld false; %ld computations\n", checks, n_supported, n_refused, n_computed);
    if (n_supported == 0 || n_refused == 0 || n_computed < 20) {
        printf("FAIL: the sweep did not exercise both answers and every computation\n");
        return 1;
    }
    if (failures) {
        printf("FAIL: %ld differences between the ported and the reference BLAS backend\n", failures);
        return 1;
    }
    printf("PASS: the ported BLAS backend is identical to ggml-blas.cpp\n");
    return 0;
}
