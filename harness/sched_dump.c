// Dump the scheduler's split structure for a few small graphs.
//
// `scripts/sched-diff` builds this twice -- once against the reference C
// libraries and once against the ported Zig -- and diffs the output. It is the
// scheduler's equivalent of `harness/graph_dump.c`, and it exists because the
// five assignment passes in `ggml_backend_sched_split_graph` had no gate:
// breaking one changes which backend runs an op, and `make parity-cli` cannot
// see that, because at --temp 0 the argmax is unchanged whichever backend runs
// it.
//
// Two stub devices make the passes observable. FAST claims only GGML_OP_ADD and
// has its own buffer type; SLOW claims everything. Where a node lands is then a
// direct readout of what the passes decided.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct ggml_backend_buffer_type g_fast_buft;
static struct ggml_backend_buffer_type g_slow_buft;
static struct ggml_backend_device      g_fast_dev;
static struct ggml_backend_device      g_slow_dev;
static struct ggml_backend             g_fast_backend;
static struct ggml_backend             g_slow_backend;

static const char * buft_name(ggml_backend_buffer_type_t buft) {
    return buft == &g_fast_buft ? "FAST" : "SLOW";
}
static size_t buft_alignment(ggml_backend_buffer_type_t buft) { (void) buft; return 32; }
static bool   buft_is_host  (ggml_backend_buffer_type_t buft) { (void) buft; return true; }

static const char * dev_name(ggml_backend_dev_t dev) {
    return dev == &g_fast_dev ? "FAST" : "SLOW";
}
static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t dev) {
    return dev == &g_fast_dev ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
}
static ggml_backend_buffer_type_t dev_buffer_type(ggml_backend_dev_t dev) {
    return dev == &g_fast_dev ? &g_fast_buft : &g_slow_buft;
}
static bool dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (dev == &g_slow_dev) return true;
    return op->op == GGML_OP_ADD;
}
static bool dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == dev_buffer_type(dev);
}
static const char * backend_name(ggml_backend_t b) {
    return b == &g_fast_backend ? "FAST" : "SLOW";
}
static void backend_free(ggml_backend_t b) { (void) b; }

static void stubs_init(void) {
    memset(&g_fast_buft, 0, sizeof(g_fast_buft));
    memset(&g_slow_buft, 0, sizeof(g_slow_buft));
    memset(&g_fast_dev,  0, sizeof(g_fast_dev));
    memset(&g_slow_dev,  0, sizeof(g_slow_dev));
    memset(&g_fast_backend, 0, sizeof(g_fast_backend));
    memset(&g_slow_backend, 0, sizeof(g_slow_backend));

    g_fast_buft.iface.get_name      = buft_name;
    g_fast_buft.iface.get_alignment = buft_alignment;
    g_fast_buft.iface.is_host       = buft_is_host;
    g_fast_buft.device              = &g_fast_dev;
    g_slow_buft.iface               = g_fast_buft.iface;
    g_slow_buft.device              = &g_slow_dev;

    g_fast_dev.iface.get_name        = dev_name;
    g_fast_dev.iface.get_description = dev_name;
    g_fast_dev.iface.get_type        = dev_type;
    g_fast_dev.iface.get_buffer_type = dev_buffer_type;
    g_fast_dev.iface.supports_op     = dev_supports_op;
    g_fast_dev.iface.supports_buft   = dev_supports_buft;
    g_slow_dev.iface                 = g_fast_dev.iface;

    g_fast_backend.iface.get_name = backend_name;
    g_fast_backend.iface.free     = backend_free;
    g_fast_backend.device         = &g_fast_dev;
    g_slow_backend.iface          = g_fast_backend.iface;
    g_slow_backend.device         = &g_slow_dev;
}

static void dump(const char * label, struct ggml_cgraph * graph) {
    stubs_init();
    ggml_backend_t backends[2] = { &g_fast_backend, &g_slow_backend };
    ggml_backend_buffer_type_t bufts[2] = { &g_fast_buft, &g_slow_buft };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 512, false, false);

    ggml_backend_sched_split_graph(sched, graph);

    printf("%s: n_splits=%d\n", label, ggml_backend_sched_get_n_splits(sched));
    // `ggml_cgraph` is opaque in the public header, so walk it with the
    // accessors rather than reaching into the struct.
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor * n = ggml_graph_node(graph, i);
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, n);
        printf("  %s node %2d %-12s -> %s\n", label, i, ggml_op_name(n->op), b ? ggml_backend_name(b) : "NULL");
    }
    ggml_backend_sched_free(sched);
}


// -----------------------------------------------------------------------------
// The copy and realloc callbacks (v0.7.0)
//
// `ggml_backend_sched_set_copy_callback` and `_set_realloc_callback` are what
// libllama's MoE cache uses to copy only the experts a batch selects. Only a
// split on a backend that cannot read host memory copies inputs at all, so
// this stands up a third stub, ACC: a device with its own malloc'd buffer
// type that claims only GGML_OP_ADD and offloads it, and computes ADD in f32
// itself. The CPU backend is the real one, holding the weights. Printed: each
// copy callback's src, dst and what the dst of the split's *non-weight* input
// already holds when a weight is offered (host weights are copied last since
// v0.7.0, so it holds the input's data), each realloc callback's two flags,
// and the result, which shows which half of a weight the callback copied.

static struct ggml_backend_buffer_type g_acc_buft;
static struct ggml_backend_device      g_acc_dev;
static struct ggml_backend             g_acc_backend;

static void   acc_buf_free(ggml_backend_buffer_t b) { free(b->context); }
static void * acc_buf_base(ggml_backend_buffer_t b) { return b->context; }
static void   acc_buf_memset(ggml_backend_buffer_t b, struct ggml_tensor * t, uint8_t v, size_t off, size_t size) {
    (void) b; memset((char *) t->data + off, v, size);
}
static void   acc_buf_set(ggml_backend_buffer_t b, struct ggml_tensor * t, const void * data, size_t off, size_t size) {
    (void) b; memcpy((char *) t->data + off, data, size);
}
static void   acc_buf_get(ggml_backend_buffer_t b, const struct ggml_tensor * t, void * data, size_t off, size_t size) {
    (void) b; memcpy(data, (const char *) t->data + off, size);
}
static void   acc_buf_clear(ggml_backend_buffer_t b, uint8_t v) { memset(b->context, v, b->size); }

static ggml_backend_buffer_t acc_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_buffer_i iface;
    memset(&iface, 0, sizeof(iface));
    iface.free_buffer   = acc_buf_free;
    iface.get_base      = acc_buf_base;
    iface.memset_tensor = acc_buf_memset;
    iface.set_tensor    = acc_buf_set;
    iface.get_tensor    = acc_buf_get;
    iface.clear         = acc_buf_clear;
    // calloc, so bytes no copy wrote read the same on both sides
    return ggml_backend_buffer_init(buft, iface, calloc(1, size + 64), size);
}
static const char * acc_name(void * p) { (void) p; return "ACC"; }
static bool   acc_buft_is_host(ggml_backend_buffer_type_t buft) { (void) buft; return false; }
static enum ggml_backend_dev_type acc_dev_type(ggml_backend_dev_t dev) { (void) dev; return GGML_BACKEND_DEVICE_TYPE_GPU; }
static ggml_backend_buffer_type_t acc_dev_buft(ggml_backend_dev_t dev) { (void) dev; return &g_acc_buft; }
static bool   acc_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) { (void) dev; return op->op == GGML_OP_ADD; }
static bool   acc_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { (void) dev; return buft == &g_acc_buft; }
static bool   acc_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) { (void) dev; (void) op; return true; }

static enum ggml_status acc_graph_compute(ggml_backend_t backend, struct ggml_cgraph * g) {
    (void) backend;
    for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
        struct ggml_tensor * t = ggml_graph_node(g, i);
        if (t->op != GGML_OP_ADD) continue;
        const float * a = t->src[0]->data;
        const float * b = t->src[1]->data;
        float * d = t->data;
        for (int64_t j = 0; j < ggml_nelements(t); j++) d[j] = a[j] + b[j];
    }
    return GGML_STATUS_SUCCESS;
}

static void acc_init(void) {
    memset(&g_acc_buft, 0, sizeof(g_acc_buft));
    memset(&g_acc_dev, 0, sizeof(g_acc_dev));
    memset(&g_acc_backend, 0, sizeof(g_acc_backend));
    g_acc_buft.iface.get_name      = (const char * (*)(ggml_backend_buffer_type_t)) acc_name;
    g_acc_buft.iface.alloc_buffer  = acc_buft_alloc;
    g_acc_buft.iface.get_alignment = buft_alignment;
    g_acc_buft.iface.is_host       = acc_buft_is_host;
    g_acc_buft.device              = &g_acc_dev;
    g_acc_dev.iface.get_name        = (const char * (*)(ggml_backend_dev_t)) acc_name;
    g_acc_dev.iface.get_description = (const char * (*)(ggml_backend_dev_t)) acc_name;
    g_acc_dev.iface.get_type        = acc_dev_type;
    g_acc_dev.iface.get_buffer_type = acc_dev_buft;
    g_acc_dev.iface.supports_op     = acc_supports_op;
    g_acc_dev.iface.supports_buft   = acc_supports_buft;
    g_acc_dev.iface.offload_op      = acc_offload_op;
    g_acc_backend.iface.get_name      = (const char * (*)(ggml_backend_t)) acc_name;
    g_acc_backend.iface.free          = backend_free;
    g_acc_backend.iface.graph_compute = acc_graph_compute;
    g_acc_backend.device              = &g_acc_dev;
}

// What the copy callback is told, and what it does: the n-th call copies the
// first half of the weight itself and returns true when bit n of `mode` is
// set, and declines otherwise, so the scheduler copies the whole weight.
struct copy_ud {
    int calls;
    unsigned mode;
    const struct ggml_tensor * input; // the split's user input, for its copy
};

static bool copy_cb(ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst,
                    struct ggml_cgraph * graph, void * user_data) {
    struct copy_ud * ud = user_data;
    // the split's copy of the user input, named "<backend>#x#<copy>"
    const struct ggml_tensor * in_cpy = NULL;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        const struct ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < 2; s++) {
            if (t->src[s] && strstr(t->src[s]->name, "#x#")) in_cpy = t->src[s];
        }
    }
    float v0 = -1.0f;
    if (in_cpy && in_cpy->data) memcpy(&v0, in_cpy->data, sizeof(v0));
    const bool take = (ud->mode >> ud->calls) & 1;
    printf("  copy_cb #%d backend=%s src=%s dst=%s graph_nodes=%d input_cpy[0]=%g take=%s\n", ud->calls,
           ggml_backend_name(backend), src->name, dst->name, ggml_graph_n_nodes(graph), v0, take ? "half" : "decline");
    ud->calls++;
    if (take) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src) / 2);
    }
    return take;
}

static void realloc_cb(ggml_backend_sched_t sched, struct ggml_cgraph * graph, bool ids_changed, bool size_changed,
                       void * user_data) {
    (void) sched;
    int * n = user_data;
    printf("  realloc_cb #%d backend_ids_changed=%d graph_size_changed=%d nodes=%d\n", (*n)++, ids_changed,
           size_changed, ggml_graph_n_nodes(graph));
}

// y = w0 + x on ACC, then y2 = w1 + y, then n_extra more ADDs of w1, so a
// second run changes the graph's size. `x` is a user input on the CPU; `w0`
// and `w1` are weights in a CPU buffer, so the ADDs are offloaded to ACC and
// both weights become inputs of its split, listed before `x`.
//
// `pin`, when not NULL, is the backend the first ADD is pinned to, which
// changes the assignment without changing the node count.
static void run_callbacks(const char * label, ggml_backend_sched_t sched, struct ggml_tensor * w0,
                          struct ggml_tensor * w1, int n_extra, struct copy_ud * ud, ggml_backend_t pin) {
    struct ggml_init_params p = { 1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(p);
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    struct ggml_tensor * y = ggml_add(ctx, w0, x);
    struct ggml_tensor * y0 = y;
    y = ggml_add(ctx, w1, y);
    for (int i = 0; i < n_extra; i++) y = ggml_add(ctx, w1, y);
    ggml_set_output(y);
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);

    static int run = 0;
    run++;
    ud->input = x;
    printf("%s:\n", label);
    if (pin) ggml_backend_sched_set_tensor_backend(sched, y0, pin);
    if (!ggml_backend_sched_alloc_graph(sched, g)) {
        printf("  alloc failed\n");
    } else {
        float xs[8];
        // a new value every run, so a copy that has not happened yet reads
        // the previous run's and shows
        for (int i = 0; i < 8; i++) xs[i] = 100.0f * run + i;
        ggml_backend_tensor_set(x, xs, 0, sizeof(xs));
        const enum ggml_status st = ggml_backend_sched_graph_compute(sched, g);
        printf("  n_splits=%d status=%d\n", ggml_backend_sched_get_n_splits(sched), st);
        float ys[8];
        ggml_backend_tensor_get(y, ys, 0, sizeof(ys));
        printf("  y =");
        for (int i = 0; i < 8; i++) printf(" %g", ys[i]);
        printf("\n");
    }
    ggml_backend_sched_reset(sched);
    ggml_free(ctx);
}

static void callbacks(void) {
    acc_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();

    struct ggml_init_params pw = { 1024*1024, NULL, true };
    struct ggml_context * ctx_w = ggml_init(pw);
    struct ggml_tensor * w0 = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 8);
    struct ggml_tensor * w1 = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 8);
    ggml_set_name(w0, "w0");
    ggml_set_name(w1, "w1");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, cpu);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    float ws[8];
    for (int i = 0; i < 8; i++) ws[i] = 1.0f + i;
    ggml_backend_tensor_set(w0, ws, 0, sizeof(ws));
    for (int i = 0; i < 8; i++) ws[i] = 10.0f * (i + 1);
    ggml_backend_tensor_set(w1, ws, 0, sizeof(ws));

    ggml_backend_t backends[2] = { &g_acc_backend, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, 512, false, true);

    // no callbacks: the whole weights are copied
    struct copy_ud ud = { 0, 0, NULL };
    int n_realloc = 0;
    run_callbacks("callbacks none", sched, w0, w1, 0, &ud, NULL);

    ggml_backend_sched_set_copy_callback(sched, copy_cb, &ud);
    ggml_backend_sched_set_realloc_callback(sched, realloc_cb, &n_realloc);

    ud.calls = 0; ud.mode = 0x1; // take w0's half, decline w1
    run_callbacks("callbacks first", sched, w0, w1, 0, &ud, NULL);
    ud.calls = 0; ud.mode = 0x2; // decline w0, take w1's half
    run_callbacks("callbacks same", sched, w0, w1, 0, &ud, NULL);
    ud.calls = 0; ud.mode = 0x3; // take both, on a larger graph
    run_callbacks("callbacks larger", sched, w0, w1, 3, &ud, NULL);
    ud.calls = 0; ud.mode = 0x0; // decline both, back to the smaller graph
    run_callbacks("callbacks smaller", sched, w0, w1, 0, &ud, NULL);
    ud.calls = 0; ud.mode = 0x1; // the first ADD pinned to the CPU: same nodes, new assignment
    run_callbacks("callbacks pinned", sched, w0, w1, 0, &ud, cpu);

    ggml_backend_sched_set_copy_callback(sched, NULL, NULL);
    ggml_backend_sched_set_realloc_callback(sched, NULL, NULL);
    run_callbacks("callbacks cleared", sched, w0, w1, 0, &ud, NULL);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(cpu);
}

int main(void) {
    struct ggml_init_params p = { 64*1024*1024, NULL, true };

    {   // add -> mul -> add: the middle op forces a change of backend
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        struct ggml_tensor * t = ggml_add(ctx, a, b);
        t = ggml_mul(ctx, t, b);
        t = ggml_add(ctx, t, b);
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, t);
        dump("mixed", g);
        ggml_free(ctx);
    }
    {   // every op supported by FAST
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        struct ggml_tensor * t = ggml_add(ctx, a, b);
        for (int i = 0; i < 5; i++) t = ggml_add(ctx, t, b);
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, t);
        dump("alladd", g);
        ggml_free(ctx);
    }
    {   // Pass 2 only fires when something is already assigned: it expands an
        // existing assignment to adjacent unassigned nodes, and skips the
        // lowest-priority backend while doing so. `ggml_set_input` makes pass 1
        // pin a tensor to the last backend (assumed CPU), which is exactly the
        // assignment the skip is about.
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        struct ggml_tensor * t = ggml_add(ctx, a, b);
        // Pin an intermediate *node*: pass 2 walks nodes, not leafs, so a
        // flagged leaf would leave it with nothing to expand from. The flag
        // is set by hand: since v0.6.0 `ggml_set_input` asserts the tensor
        // is a leaf (`op == GGML_OP_NONE`).
        t->flags |= GGML_TENSOR_FLAG_INPUT;
        for (int i = 0; i < 3; i++) t = ggml_add(ctx, t, b);
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, t);
        dump("input", g);
        ggml_free(ctx);
    }
    {   // a view, to exercise pass 4's view_src propagation
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        struct ggml_tensor * s = ggml_add(ctx, a, b);
        struct ggml_tensor * v = ggml_view_1d(ctx, s, 4, 0);
        struct ggml_tensor * o = ggml_add(ctx, v, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4));
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, o);
        dump("view", g);
        ggml_free(ctx);
    }
    {   // Pipeline parallelism (`n_copies > 1`): pass 6 gives every input
        // *leaf* its per-copy tensors and lists it in `graph_inputs`, which
        // the graph copy then allocates first -- including an input no node
        // consumes (`x`, expanded on its own). Neither is visible in the
        // assignments, so this allocates on a real CPU backend and prints
        // the buffer size and where each leaf landed.
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        struct ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_set_name(x, "x"); ggml_set_name(a, "a"); ggml_set_name(b, "b"); ggml_set_name(w, "w");
        ggml_set_input(x);
        ggml_set_input(a);
        ggml_set_input(b);
        struct ggml_tensor * t = ggml_add(ctx, b, w);
        t = ggml_mul(ctx, t, a);
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, x);
        ggml_build_forward_expand(g, t);

        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_t backends[1] = { cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 1, 512, true, false);
        bool ok = ggml_backend_sched_alloc_graph(sched, g);
        printf("parallel: alloc=%d n_copies=%d buffer=%zu\n", ok, ggml_backend_sched_get_n_copies(sched),
               ggml_backend_sched_get_buffer_size(sched, cpu));
        struct ggml_tensor * leafs[4] = { x, a, b, w };
        for (int i = 0; i < 4; i++) {
            struct ggml_tensor * l = leafs[i];
            long off = l->data && l->buffer ? (long) ((char *) l->data - (char *) ggml_backend_buffer_get_base(l->buffer)) : -1;
            printf("  parallel leaf %s -> off %ld\n", l->name, off);
        }
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }
    callbacks();
    return 0;
}
