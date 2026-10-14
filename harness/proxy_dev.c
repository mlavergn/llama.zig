// A backend device that is the CPU without GGML_OP_GATED_DELTA_NET.
//
// **Not a port.** The harness's way into the paths a backend takes when it
// lacks a fused op. Every device on this machine supports the fused Gated
// Delta Net, so `resolve_fused_ops` always enables it and the unfused
// chunked and autoregressive graphs in `delta-net-base.cpp` never run --
// measured: halving their chunk size passed `context-diff`. A backend
// without the op (the condition the probe exists for) takes them. This
// device reports itself as a GPU, forwards everything to the CPU device,
// and declines that one op, so the fused node lands on the CPU backend, the
// probe sees a device mismatch, and both sides build the unfused graph.
//
// Used by harness/context_diff.zig through scripts/context-diff.

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

static struct ggml_backend_device proxy;
static ggml_backend_dev_t cpu_dev;

static const char * proxy_name(ggml_backend_dev_t dev) {
    (void) dev;
    return "PROXY";
}

static const char * proxy_description(ggml_backend_dev_t dev) {
    (void) dev;
    return "CPU without GATED_DELTA_NET";
}

static enum ggml_backend_dev_type proxy_type(ggml_backend_dev_t dev) {
    (void) dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void proxy_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    cpu_dev->iface.get_props(cpu_dev, props);
    props->name        = proxy_name(dev);
    props->description = proxy_description(dev);
    props->type        = proxy_type(dev);
}

static ggml_backend_t proxy_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = cpu_dev->iface.init_backend(cpu_dev, params);
    if (backend) {
        backend->device = dev;
    }
    return backend;
}

static bool proxy_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void) dev;
    if (op->op == GGML_OP_GATED_DELTA_NET) {
        return false;
    }
    return cpu_dev->iface.supports_op(cpu_dev, op);
}

ggml_backend_dev_t harness_proxy_device(void) {
    if (!cpu_dev) {
        cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        proxy = *cpu_dev;
        proxy.iface.get_name        = proxy_name;
        proxy.iface.get_description = proxy_description;
        proxy.iface.get_type        = proxy_type;
        proxy.iface.get_props       = proxy_props;
        proxy.iface.init_backend    = proxy_init_backend;
        proxy.iface.supports_op     = proxy_supports_op;
    }
    return &proxy;
}
