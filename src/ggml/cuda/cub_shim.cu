// CUB's device-wide algorithms behind a C ABI, for the Zig host layer.
//
// **Not a port.** PLAN.md, Decision 47: `argsort.cu` and `cumsum.cu` call
// CUB's device-wide sort and scan, whose host-side dispatch -- tuning
// policies, temporary-storage layout, the kernel launches themselves --
// lives in CUB's header templates and is library code, not ggml's. This
// file is the only CUDA host code the port writes in C++. It wraps exactly
// the calls upstream makes and nothing else:
//
// - `llama.cpp/ggml/src/ggml-cuda/argsort.cu` (v0.3.0, `c1d0e7a00`),
//   `argsort_f32_i32_cuda_cub` (argsort.cu:45): `DeviceRadixSort`,
//   `DeviceSegmentedRadixSort` and `DeviceSegmentedSort`, `SortPairs` and
//   `SortPairsDescending`, each called once to size the temporary storage
//   and once to sort.
// - `llama.cpp/ggml/src/ggml-cuda/cumsum.cu` (v0.3.0, `c1d0e7a00`),
//   `cumsum_cub<float>` (cumsum.cu:190): `DeviceScan::InclusiveSum`.
// - `llama.cpp/ggml/src/ggml-cuda/sum.cu` (v0.3.0, `c1d0e7a00`),
//   `sum_f32_cuda` (sum.cu:11), and `llama.cpp/ggml/src/ggml-cuda/mean.cu`
//   (v0.3.0, `c1d0e7a00`), `ggml_cuda_op_mean` (mean.cu:13):
//   `DeviceReduce::Sum`. Both call it with `const float *`, `float *` and an
//   `int64_t` count, so one wrapper is both instantiations.
//
// Each function takes its arguments with **the C++ types upstream's call
// site has** -- `float *` keys passed as both `d_keys_in` and `d_keys_out`,
// `int *` values, `const int` counts, `int *` offsets, the `0, sizeof(float)
// * 8` bit range -- and passes them in upstream's order, so template
// argument deduction picks the same instantiation and the device code is
// the same. Measured, not assumed, with `cuobjdump -sass` on this object
// and on the reference's `argsort.o` and `cumsum.o`: this object's 23
// kernels are exactly the 23 CUB kernels those two carry, by mangled name,
// and every one's SASS -- instruction encodings included -- is identical.
//
// The Zig side keeps everything around the calls: the pool allocations,
// `CUDA_CHECK` where upstream has it, the stream-capture test that chooses
// the segmented radix sort. Compiled by `build/llamacpp.zig`'s
// `addCudaPort` with `nvcc -c` and the reference's flags.

#include <cub/cub.cuh>
#include <cuda_runtime.h>

// argsort.cu builds with this `using`; the calls below spell CUB's names
// as upstream does.
using namespace cub;

// argsort.cu's `STRIDED_ITERATOR_AVAILABLE` needs CCCL >= 3.1. Upstream
// then passes a strided counting iterator as the segment offsets instead of
// an `int *` -- a different instantiation. Fail loudly rather than wrap the
// wrong one.
#if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#    error "CCCL >= 3.1: argsort.cu uses cuda::make_strided_iterator offsets; cub_shim.cu wraps the int * form"
#endif

extern "C" {

// argsort.cu:88, 126 -- `nrows == 1`, ascending.
cudaError_t ggml_cuda_cub_radix_sort_pairs(void * d_temp_storage, size_t * temp_storage_bytes, float * temp_keys,
                                           int * temp_indices, int * dst, const int ncols, cudaStream_t stream) {
    return DeviceRadixSort::SortPairs(d_temp_storage, *temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                                      temp_indices, dst,  // values (indices)
                                      ncols, 0, sizeof(float) * 8, stream);
}

// argsort.cu:106, 141 -- `nrows == 1`, descending.
cudaError_t ggml_cuda_cub_radix_sort_pairs_descending(void * d_temp_storage, size_t * temp_storage_bytes,
                                                      float * temp_keys, int * temp_indices, int * dst,
                                                      const int ncols, cudaStream_t stream) {
    return DeviceRadixSort::SortPairsDescending(d_temp_storage, *temp_storage_bytes, temp_keys,
                                                temp_keys,          // keys (in-place)
                                                temp_indices, dst,  // values (indices)
                                                ncols, 0, sizeof(float) * 8, stream);
}

// argsort.cu:92, 131 -- several rows while the stream is capturing, ascending.
cudaError_t ggml_cuda_cub_segmented_radix_sort_pairs(void * d_temp_storage, size_t * temp_storage_bytes,
                                                     float * temp_keys, int * temp_indices, int * dst,
                                                     const int ncols, const int nrows, int * offset_iterator,
                                                     cudaStream_t stream) {
    return DeviceSegmentedRadixSort::SortPairs(d_temp_storage, *temp_storage_bytes, temp_keys,
                                               temp_keys,                                  // keys (in-place)
                                               temp_indices, dst,                          // values (indices)
                                               ncols * nrows, nrows,                       // num items, num segments
                                               offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream);
}

// argsort.cu:111, 146 -- several rows while the stream is capturing, descending.
cudaError_t ggml_cuda_cub_segmented_radix_sort_pairs_descending(void * d_temp_storage, size_t * temp_storage_bytes,
                                                                float * temp_keys, int * temp_indices, int * dst,
                                                                const int ncols, const int nrows,
                                                                int * offset_iterator, cudaStream_t stream) {
    return DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, *temp_storage_bytes, temp_keys, temp_keys,
                                                         temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                         offset_iterator + 1, 0, sizeof(float) * 8, stream);
}

// argsort.cu:98, 135 -- several rows, not capturing, ascending.
cudaError_t ggml_cuda_cub_segmented_sort_pairs(void * d_temp_storage, size_t * temp_storage_bytes, float * temp_keys,
                                               int * temp_indices, int * dst, const int ncols, const int nrows,
                                               int * offset_iterator, cudaStream_t stream) {
    return DeviceSegmentedSort::SortPairs(d_temp_storage, *temp_storage_bytes, temp_keys,
                                          temp_keys,             // keys (in-place)
                                          temp_indices, dst,     // values (indices)
                                          ncols * nrows, nrows,  // num items, num segments
                                          offset_iterator, offset_iterator + 1, stream);
}

// argsort.cu:115, 150 -- several rows, not capturing, descending.
cudaError_t ggml_cuda_cub_segmented_sort_pairs_descending(void * d_temp_storage, size_t * temp_storage_bytes,
                                                          float * temp_keys, int * temp_indices, int * dst,
                                                          const int ncols, const int nrows, int * offset_iterator,
                                                          cudaStream_t stream) {
    return DeviceSegmentedSort::SortPairsDescending(d_temp_storage, *temp_storage_bytes, temp_keys, temp_keys,
                                                    temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                    offset_iterator + 1, stream);
}

// cumsum.cu:198, 209 -- `cumsum_cub<float>`. Upstream discards the result;
// it is returned here so the caller can discard it the same way.
cudaError_t ggml_cuda_cub_inclusive_sum_f32(void * d_temp_storage, size_t * tmp_size, const float * src, float * dst,
                                            int64_t ne, cudaStream_t stream) {
    return cub::DeviceScan::InclusiveSum(d_temp_storage,  // d_temp_storage (null = just query size)
                                         *tmp_size,       // reference to size (will be set by CUB)
                                         src,             // input pointer
                                         dst,             // output pointer
                                         ne,              // number of elements
                                         stream           // CUDA stream to use
    );
}

// sum.cu:14, 16 and mean.cu:47, 50 -- `DeviceReduce::Sum` over one row.
cudaError_t ggml_cuda_cub_reduce_sum_f32(void * d_temp_storage, size_t * tmp_size, const float * x, float * dst,
                                         const int64_t ne, cudaStream_t stream) {
    return DeviceReduce::Sum(d_temp_storage, *tmp_size, x, dst, ne, stream);
}

}  // extern "C"
