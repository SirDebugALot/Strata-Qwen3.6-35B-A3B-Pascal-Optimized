// Original Q6_K dense GPU projections, in GGUF blocks or lossless row planes selected by DQ8::q6_row_planes.
// No intermediate Q8 requantization or persistent duplicate dense weights.
#pragma once
#include "kernels.cuh"

namespace sq {
#ifdef __CUDACC__
namespace detail {
// Each byte is in 0..63. Adding 96 cannot carry between bytes, and flipping bit7 then represents u-32.
// Keep the old intrinsic as a separately compiled variant; neither path tests the option inside the dot loop.
template <bool FastCenter>
__device__ __forceinline__ int dense_q6_center(uint32_t values) {
    if constexpr (FastCenter) return (int)((values + 0x60606060u) ^ 0x80808080u);
    else return __vsubss4((int)values, 0x20202020);
}
} // namespace detail
#endif
void k_gemv_q6(const DQ8& weight, ActQ activation, float* output, int tokens, cudaStream_t stream);
// Same NT1 arithmetic as the full projection. Invalid rows and non-first duplicates produce -INFINITY.
void k_gather_q6(const DQ8& weight, ActQ activation, const int32_t* ids, const int32_t* first,
                  int slots, float* output, cudaStream_t stream);
// Explicit variant selection for same-process bitwise parity; production caches both opt-ins per process.
void k_test_gemv_q6(const DQ8& weight, ActQ activation, float* output, int tokens, bool transpose,
                    cudaStream_t stream, bool fast_center = false);
void k_test_gather_q6(const DQ8& weight, ActQ activation, const int32_t* ids, const int32_t* first,
                       int slots, float* output, bool transpose, cudaStream_t stream, bool fast_center = false);
} // namespace sq
