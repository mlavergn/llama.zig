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

#include <stdio.h>
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
        // flagged leaf would leave it with nothing to expand from.
        ggml_set_input(t);
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
    return 0;
}
