// Not a port: a link stub for scripts/repack-diff off macOS.
//
// The reference `traits.cpp`, compiled with its exports renamed to `ref_*`,
// still calls the C++-linkage `ggml_backend_cpu_get_extra_buffer_types()` from
// `ref_ggml_cpu_extra_compute_forward` and `ref_ggml_cpu_extra_work_size`. The
// harness calls neither -- it needs `traits.cpp` only for the two base
// classes `repack.cpp` derives from -- and the port has no such symbol: its
// list is a slice in `cpu/cpu_backend.zig`. ld64 dead-strips the two callers
// and never asks; lld reports the reference even from discarded code. This
// definition satisfies it, and aborts if anything ever does call it.

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    std::fprintf(stderr, "repack_diff_stub: ggml_backend_cpu_get_extra_buffer_types called\n");
    std::abort();
}
