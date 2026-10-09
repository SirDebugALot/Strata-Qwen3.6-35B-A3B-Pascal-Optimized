// Original Q6_K dense projections for Pascal SM61 DP4A. Q6 unpacking and scale math follow kernels.cu's
// dot_q6k_c / ggml Q6_K layout; each unpacked word is reused across all tokens in a verification window.
#include "dense_q6.cuh"
#include "common.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdlib>

namespace sq {
namespace {

__device__ __forceinline__ int load_word_2(const uint8_t* bytes) {
    const uint16_t* words = (const uint16_t*)bytes;
    return (int)((uint32_t)words[0] | ((uint32_t)words[1] << 16));
}

template <int NT, bool Transpose, bool RowPlanes, bool FastCenter>
__device__ __forceinline__ void add_q6_group(const uint8_t* row_weights, int group,
                                            const int* activation, const float* scales,
                                            int cols, int activation_group, float (&acc)[NT]) {
    const int half = group >> 2, quadrant = group & 3;
    const int blocks = cols / 256, superblock = activation_group / 8;
    const uint8_t* block = row_weights + (RowPlanes ? 0 : superblock * 210);
    const uint8_t* ql = block + 64 * half + 32 * (quadrant & 1);
    const uint8_t* qh = block + 128 + 32 * half;
    const int low_shift = 4 * (quadrant >> 1), high_shift = 2 * quadrant;
    const int8_t* subscale = RowPlanes ? (const int8_t*)(row_weights + 192 * blocks + 16 * superblock + 2 * group)
                                      : (const int8_t*)(block + 192) + 8 * half + 2 * quadrant;
    int sums[NT][2] = {};
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        // A plane retains the same four source bytes but places neighboring lanes' words contiguously.
        // Each aligned 32-bit load replaces two 16-bit loads required by the original 210-byte blocks.
        const int low_word = RowPlanes ? __ldg((const int*)row_weights + word * blocks * 4 +
                                              superblock * 4 + half * 2 + (quadrant & 1))
                                       : load_word_2(ql + 4 * word);
        const int high_word = RowPlanes ? __ldg((const int*)(row_weights + 128 * blocks) + word * blocks * 2 +
                                               superblock * 2 + half)
                                        : load_word_2(qh + 4 * word);
        const int low = (low_word >> low_shift) & 0x0f0f0f0f;
        const int high = ((high_word >> high_shift) & 0x03030303) << 4;
        const int weights = detail::dense_q6_center<FastCenter>((uint32_t)(low | high));
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int index = Transpose ? t * (cols / 4 + 8) + word * (cols / 32 + 1) + activation_group
                                        : t * (cols / 4) + activation_group * 8 + word;
            const int values = activation[index];
            sums[t][word / 4] = __dp4a(weights, values, sums[t][word / 4]);
        }
    }
    const uint8_t* scale = RowPlanes ? row_weights + 208 * blocks + 2 * superblock : block + 208;
    const float d = __half2float(*(const __half*)scale);
#pragma unroll
    for (int t = 0; t < NT; ++t)
        acc[t] += d * scales[t * (cols / 32) + activation_group] *
                  (float)(subscale[0] * sums[t][0] + subscale[1] * sums[t][1]);
}

template <int NT, int LanesPerRow, bool Gather, bool Transpose, bool RowPlanes, bool FastCenter>
__global__ void __launch_bounds__(256)
dense_q6_kernel(const uint8_t* __restrict__ weights, ActQ activation, float* __restrict__ output,
                 int rows, int cols, const int32_t* ids, const int32_t* first, int slots) {
    extern __shared__ int4 memory[];
    int* aq = (int*)memory;
    const int activation_bytes = cols + (Transpose ? 32 : 0);
    float* ad = (float*)(memory + NT * activation_bytes / 16);
    if constexpr (Transpose) {
        // Word planes turn the dot lanes' stride-eight int32 reads into bank-contiguous reads. One padding
        // group reduces staging-copy conflicts and costs only32 bytes/token; arithmetic below is unchanged.
        const int words = cols / 4, plane = cols / 32 + 1;
        for (int i = threadIdx.x; i < NT * words; i += blockDim.x) {
            const int token = i / words, q = i % words;
            aq[token * (words + 8) + (q & 7) * plane + (q >> 3)] = ((const int*)activation.q)[i];
        }
    } else {
        for (int i = threadIdx.x; i < NT * cols / 16; i += blockDim.x) memory[i] = ((const int4*)activation.q)[i];
    }
    for (int i = threadIdx.x; i < NT * cols / 32; i += blockDim.x) ad[i] = activation.d[i];
    __syncthreads();
    const int slot = blockIdx.x * (256 / LanesPerRow) + threadIdx.x / LanesPerRow;
    const int lane = threadIdx.x % LanesPerRow;
    if (slot >= slots) return;
    const int row = Gather ? ids[slot] : slot;
    if constexpr (Gather) {
        if (row < 0 || row >= rows || first[row] != slot) {
            if (lane == 0) output[slot] = -CUDART_INF_F;
            return;
        }
    }
    const int blocks = cols / 256, groups = cols / 32;
    const int row_bytes = RowPlanes ? ((blocks * 210 + 31) & ~31) : blocks * 210;
    const uint8_t* row_weights = weights + (size_t)row * row_bytes;
    float acc[NT] = {};
    for (int group = lane; group < groups; group += LanesPerRow)
        add_q6_group<NT, Transpose, RowPlanes, FastCenter>(row_weights, group & 7, aq, ad, cols, group, acc);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int offset = LanesPerRow / 2; offset > 0; offset >>= 1)
            acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], offset);
        if (lane == 0) output[(size_t)t * slots + slot] = acc[t];
    }
}

template <int NT, bool Gather, bool Transpose, bool RowPlanes, bool FastCenter>
void launch(const DQ8& weight, ActQ activation, float* output, const int32_t* ids,
              const int32_t* first, int slots, cudaStream_t stream) {
    const size_t shared = (size_t)NT * (weight.cols + (Transpose ? 32 : 0) + (weight.cols / 32) * sizeof(float));
    if (weight.cols == 512)
        dense_q6_kernel<NT, 16, Gather, Transpose, RowPlanes, FastCenter><<<(slots + 15) / 16, 256, shared, stream>>>(
            weight.q6, activation, output, weight.rows, weight.cols, ids, first, slots);
    else
        dense_q6_kernel<NT, 32, Gather, Transpose, RowPlanes, FastCenter><<<(slots + 7) / 8, 256, shared, stream>>>(
            weight.q6, activation, output, weight.rows, weight.cols, ids, first, slots);
}

template <int NT, bool Gather, bool FastCenter>
void launch_layout(const DQ8& weight, ActQ activation, float* output, const int32_t* ids,
                     const int32_t* first, int slots, bool transpose, cudaStream_t stream) {
    if (weight.q6_row_planes) {
        if (transpose) launch<NT, Gather, true, true, FastCenter>(weight, activation, output, ids, first, slots, stream);
        else launch<NT, Gather, false, true, FastCenter>(weight, activation, output, ids, first, slots, stream);
    } else {
        if (transpose) launch<NT, Gather, true, false, FastCenter>(weight, activation, output, ids, first, slots, stream);
        else launch<NT, Gather, false, false, FastCenter>(weight, activation, output, ids, first, slots, stream);
    }
}

template <int NT, bool Gather>
void launch_centering(const DQ8& weight, ActQ activation, float* output, const int32_t* ids,
                       const int32_t* first, int slots, bool transpose, bool fast_center, cudaStream_t stream) {
    if (fast_center) launch_layout<NT, Gather, true>(weight, activation, output, ids, first, slots, transpose, stream);
    else launch_layout<NT, Gather, false>(weight, activation, output, ids, first, slots, transpose, stream);
}

bool transpose_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_DENSE_Q6_TRANSPOSE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

bool fast_center_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_DENSE_Q6_FAST_CENTER");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}
} // namespace

void k_test_gemv_q6(const DQ8& weight, ActQ activation, float* output, int tokens, bool transpose,
                    cudaStream_t stream, bool fast_center) {
    SQ_CHECK(weight.q6 && !weight.qs && !weight.d && weight.rows > 0 && weight.cols > 0 && weight.cols % 256 == 0,
             "invalid native Q6 dense descriptor");
    switch (tokens) {
        case 1: launch_centering<1, false>(weight, activation, output, nullptr, nullptr, weight.rows, transpose, fast_center, stream); break;
        case 2: launch_centering<2, false>(weight, activation, output, nullptr, nullptr, weight.rows, transpose, fast_center, stream); break;
        case 3: launch_centering<3, false>(weight, activation, output, nullptr, nullptr, weight.rows, transpose, fast_center, stream); break;
        case 4: launch_centering<4, false>(weight, activation, output, nullptr, nullptr, weight.rows, transpose, fast_center, stream); break;
        default: die("native Q6 dense step needs 1..4 tokens, got %d", tokens);
    }
}

void k_test_gather_q6(const DQ8& weight, ActQ activation, const int32_t* ids, const int32_t* first,
                       int slots, float* output, bool transpose, cudaStream_t stream, bool fast_center) {
    SQ_CHECK(weight.q6 && !weight.qs && !weight.d && weight.rows > 0 && weight.cols > 0 && weight.cols % 256 == 0 && slots > 0,
             "invalid native Q6 gathered descriptor");
    launch_centering<1, true>(weight, activation, output, ids, first, slots, transpose, fast_center, stream);
}

void k_gemv_q6(const DQ8& weight, ActQ activation, float* output, int tokens, cudaStream_t stream) {
    k_test_gemv_q6(weight, activation, output, tokens, transpose_enabled(), stream, fast_center_enabled());
}

void k_gather_q6(const DQ8& weight, ActQ activation, const int32_t* ids, const int32_t* first,
                  int slots, float* output, cudaStream_t stream) {
    k_test_gather_q6(weight, activation, ids, first, slots, output, transpose_enabled(), stream, fast_center_enabled());
}
} // namespace sq
