/*
 * The CUDA headers the Zig host layer calls through: the driver API (kernel
 * loading and launch, the VMM pool), the runtime API (streams, events,
 * memory, graphs) and cuBLAS. Translated by `build/llamacpp.zig`'s
 * `cudaCModule` as the `cuda_c` module.
 *
 * **Not a port.** None of these includes a ggml header, so this translation
 * cannot produce a second, incompatible copy of any ggml type.
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
