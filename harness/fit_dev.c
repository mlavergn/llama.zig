// Fake GPUs whose free and total memory the harness chooses, for
// harness/fit_diff.zig.
//
// **Not a port.** `common_fit_params` decides from what each device reports
// free, so on this machine -- one Metal device with far more memory than any
// test model needs -- it never leaves step 1 for a GPU, and it never sees two
// devices at all. These devices report whatever `fit_dev_set` says, so every
// arm is reachable and every run is deterministic.
//
// Each one is the CPU device under another name with three differences that
// matter to the fit: it says it is a GPU; it has a buffer type of its own,
// not host memory and pointing back at it, so the memory breakdown attributes
// what lands there to this device rather than to the host; and its registry
// exposes no extra buffer types, so nothing is placed in the CPU's repack
// buffers on its behalf, and declines a split buffer type. Only `no_alloc`
// loads ever reach it: the buffers it allocates are the CPU's, relabelled,
// and it claims every op, as a GPU that supports everything would. Nothing
// ever computes on it.

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <string.h>

#define FIT_DEV_MAX 4

static struct ggml_backend_device         devs[FIT_DEV_MAX];
static struct ggml_backend_buffer_type    bufts[FIT_DEV_MAX];
static struct ggml_backend_reg            reg;
static char                               names[FIT_DEV_MAX][16];
static char                               buft_names[FIT_DEV_MAX][16];
static size_t                             mem_free[FIT_DEV_MAX];
static size_t                             mem_total[FIT_DEV_MAX];
static enum ggml_backend_dev_type         types[FIT_DEV_MAX];
static ggml_backend_dev_t                 cpu_dev;
static ggml_backend_buffer_type_t         cpu_buft;

static int dev_index(ggml_backend_dev_t dev) {
    return (int) (dev - devs);
}

static int buft_index(ggml_backend_buffer_type_t buft) {
    return (int) (buft - bufts);
}

static const char * dev_name(ggml_backend_dev_t dev) {
    return names[dev_index(dev)];
}

static const char * dev_description(ggml_backend_dev_t dev) {
    (void) dev;
    return "NVIDIA GeForce fake";
}

static void dev_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = mem_free[dev_index(dev)];
    *total = mem_total[dev_index(dev)];
}

static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t dev) {
    return types[dev_index(dev)];
}

static void dev_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    cpu_dev->iface.get_props(cpu_dev, props);
    props->name        = dev_name(dev);
    props->description = dev_description(dev);
    props->type        = dev_type(dev);
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps.async                = false;
    props->caps.host_buffer          = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events               = false;
}

static ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = cpu_dev->iface.init_backend(cpu_dev, params);
    if (backend) {
        backend->device = dev;
    }
    return backend;
}

static ggml_backend_buffer_type_t dev_buffer_type(ggml_backend_dev_t dev) {
    return &bufts[dev_index(dev)];
}

static bool dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void) dev;
    (void) op;
    return true;
}

static bool dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &bufts[dev_index(dev)];
}

static const char * buft_name(ggml_backend_buffer_type_t buft) {
    return buft_names[buft_index(buft)];
}

static ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buf = cpu_buft->iface.alloc_buffer(cpu_buft, size);
    if (buf) {
        buf->buft = buft;
    }
    return buf;
}

static size_t buft_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return ggml_backend_buft_get_alignment(cpu_buft);
}

static bool buft_is_host(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return false;
}

static const char * reg_name(ggml_backend_reg_t r) {
    (void) r;
    return "FITDEV";
}

static size_t reg_device_count(ggml_backend_reg_t r) {
    (void) r;
    return FIT_DEV_MAX;
}

static ggml_backend_dev_t reg_device(ggml_backend_reg_t r, size_t index) {
    (void) r;
    return &devs[index];
}

// A split buffer type the device declines to give, so a ROW split loads
// with the device's own buffer type and reaches the fit's refusal of it --
// without the function at all, the load itself fails first.
static ggml_backend_buffer_type_t split_buffer_type(int main_device, const float * tensor_split) {
    (void) main_device;
    (void) tensor_split;
    return NULL;
}

static void * reg_proc_address(ggml_backend_reg_t r, const char * name) {
    (void) r;
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *) split_buffer_type;
    }
    return NULL;
}

// Device `i`, reporting `free` of `total` bytes, as a `type` (0 GPU, 1 IGPU,
// 2 ACCEL, as `enum ggml_backend_dev_type` numbers them past CPU).
ggml_backend_dev_t fit_dev_set(int i, size_t free, size_t total, int type) {
    if (!cpu_dev) {
        cpu_dev  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        cpu_buft = ggml_backend_dev_buffer_type(cpu_dev);
        reg.api_version            = GGML_BACKEND_API_VERSION;
        reg.iface.get_name         = reg_name;
        reg.iface.get_device_count = reg_device_count;
        reg.iface.get_device       = reg_device;
        reg.iface.get_proc_address = reg_proc_address;
        reg.context                = NULL;
        for (int k = 0; k < FIT_DEV_MAX; k++) {
            snprintf(names[k], sizeof(names[k]), "FAKE%d", k);
            snprintf(buft_names[k], sizeof(buft_names[k]), "FAKE%d_BUF", k);
            devs[k] = *cpu_dev;
            devs[k].reg                        = &reg;
            devs[k].iface.get_name             = dev_name;
            devs[k].iface.get_description      = dev_description;
            devs[k].iface.get_memory           = dev_memory;
            devs[k].iface.get_type             = dev_type;
            devs[k].iface.get_props            = dev_props;
            devs[k].iface.init_backend         = dev_init_backend;
            devs[k].iface.get_buffer_type      = dev_buffer_type;
            devs[k].iface.get_host_buffer_type = NULL;
            devs[k].iface.buffer_from_host_ptr = NULL;
            devs[k].iface.supports_op          = dev_supports_op;
            devs[k].iface.supports_buft        = dev_supports_buft;
            devs[k].iface.offload_op           = NULL;
            devs[k].iface.event_new            = NULL;
            devs[k].iface.event_free           = NULL;
            devs[k].iface.event_synchronize    = NULL;

            bufts[k] = *cpu_buft;
            bufts[k].iface.get_name      = buft_name;
            bufts[k].iface.alloc_buffer  = buft_alloc_buffer;
            bufts[k].iface.get_alignment = buft_alignment;
            bufts[k].iface.is_host       = buft_is_host;
            bufts[k].device              = &devs[k];
        }
    }
    mem_free[i]  = free;
    mem_total[i] = total;
    types[i]     = type == 0 ? GGML_BACKEND_DEVICE_TYPE_GPU : type == 1 ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_ACCEL;
    return &devs[i];
}
