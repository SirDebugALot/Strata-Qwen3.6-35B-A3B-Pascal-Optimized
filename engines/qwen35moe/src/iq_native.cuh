// Native packed IQ and low-bit K expert dot products for Pascal sm_61 and newer.
//
// Adapted from Strata 0.1.40.2 src/kernels/cuda/{iq_kernels,native_mmvq}.cu and its llama.cpp
// vecdotq.cuh source at third_party/ggml/VERSION.txt. Copyright (c) 2026 Niko1221
// and the Strata contributors; Copyright (c) 2023-2026 the ggml authors.
// MIT licenses: repository LICENSE and third_party/ggml/LICENSE.
// The block layouts and codebooks below are included unchanged from ggml.
//
// Each lane computes one contiguous group of 32 values, matching this engine's
// ActQ layout (int8 activations and a FLOAT scale per 32, not ggml's half q8_1).
// IQ2_S keeps the exact fractional subscale; rounding an intermediate integer
// division would change the original dequantized weights.
#pragma once

#include "quant.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "../../../third_party/ggml/ggml-common.h"
// ggml declares QK_K as a macro; do not shadow sq::QK_K for following includes.
#undef QK_K

namespace sq::iq_native {

// IQ2_S / IQ3_S blocks have only two-byte alignment (82 / 110 bytes per block).
__device__ __forceinline__ int load_i32_b2(const void* p, int i) {
    const uint16_t* h = static_cast<const uint16_t*>(p) + 2 * i;
    return static_cast<int>(static_cast<uint32_t>(h[0]) | (static_cast<uint32_t>(h[1]) << 16));
}

__device__ __forceinline__ int load_i32_b4(const void* p, int i) {
    return static_cast<const int*>(p)[i];
}

// Optional shared-memory transpose: consecutive lanes own consecutive 32-value groups; their word j lives
// j*AStride words away. AStride=1 is the original contiguous layout and must keep its arithmetic unchanged.
template <int AStride = 1>
__device__ __forceinline__ int load_act(const int8_t* p, int i) {
    return load_i32_b4(p, i * AStride);
}

// Expand four packed nibbles per word into two words of signed codebook values.
// Uses Pascal byte-permutation instructions, with no tensor-core dependency.
__device__ __forceinline__ int2 lookup_iq4(int q4) {
    const uint32_t* table = reinterpret_cast<const uint32_t*>(kvalues_iq4nl);
    const uint32_t selection = 0x32103210u | ((static_cast<uint32_t>(q4) & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table[0], table[1], static_cast<uint32_t>(q4) >> shift);
        const uint32_t high = __byte_perm(table[2], table[3], static_cast<uint32_t>(q4) >> shift);
        tmp[i] = __byte_perm(low, high, selection >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// Signs are one byte per eight weights. The masks replicate each bit to a byte.
__device__ __forceinline__ int sign_low(uint8_t s) {
    return __vcmpne4(((static_cast<uint32_t>(s) & 0x03u) << 7) |
                    ((static_cast<uint32_t>(s) & 0x0cu) << 21), 0u);
}

__device__ __forceinline__ int sign_high(uint8_t s) {
    return __vcmpne4(((static_cast<uint32_t>(s) & 0x30u) << 3) |
                    ((static_cast<uint32_t>(s) & 0xc0u) << 17), 0u);
}

template <int AStride = 1>
__device__ __forceinline__ float dot_iq2_s_32(const uint8_t* block, int group,
                                            const int8_t* aq, float ad) {
    const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(block);
    const int packed_qs = load_i32_b2(b->qs, group);
    const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed_qs);
    const int packed_signs = load_i32_b2(b->qs, 8 + group);
    const uint8_t* signs = reinterpret_cast<const uint8_t*>(&packed_signs);
    const int qh = b->qh[group];
    int sums[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int index = qs[j] | ((qh << (8 - 2 * j)) & 0x300);
        const uint2 grid = reinterpret_cast<const uint2*>(iq2s_grid)[index];
        const int s0 = sign_low(signs[j]), s1 = sign_high(signs[j]);
        const int v0 = __vsub4(grid.x ^ s0, s0), v1 = __vsub4(grid.y ^ s1, s1);
        sums[j / 2] = __dp4a(v0, load_act<AStride>(aq, 2 * j), sums[j / 2]);
        sums[j / 2] = __dp4a(v1, load_act<AStride>(aq, 2 * j + 1), sums[j / 2]);
    }
    const int s0 = 2 * (b->scales[group] & 15) + 1;
    const int s1 = 2 * (b->scales[group] >> 4) + 1;
    return (__half2float(b->d) * ad) * (0.125f * static_cast<float>(s0 * sums[0] + s1 * sums[1]));
}

template <int AStride = 1>
__device__ __forceinline__ float dot_iq3_s_32(const uint8_t* block, int group,
                                            const int8_t* aq, float ad) {
    const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(block);
    const int2 packed_qs = make_int2(load_i32_b2(b->qs, 2 * group), load_i32_b2(b->qs, 2 * group + 1));
    const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed_qs);
    const int packed_signs = load_i32_b2(b->signs, group);
    const uint8_t* signs = reinterpret_cast<const uint8_t*>(&packed_signs);
    const int qh = b->qh[group];
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int index0 = qs[2 * j] | ((qh << (8 - 2 * j)) & 0x100);
        const int index1 = qs[2 * j + 1] | ((qh << (7 - 2 * j)) & 0x100);
        const int s0 = sign_low(signs[j]), s1 = sign_high(signs[j]);
        const int v0 = __vsub4(iq3s_grid[index0] ^ s0, s0);
        const int v1 = __vsub4(iq3s_grid[index1] ^ s1, s1);
        sum = __dp4a(v0, load_act<AStride>(aq, 2 * j), sum);
        sum = __dp4a(v1, load_act<AStride>(aq, 2 * j + 1), sum);
    }
    const int scale = 1 + 2 * ((b->scales[group / 2] >> (4 * (group & 1))) & 15);
    return (__half2float(b->d) * ad) * static_cast<float>(sum * scale);
}

template <int AStride = 1>
__device__ __forceinline__ float dot_iq4_xs_32(const uint8_t* block, int group,
                                             const int8_t* aq, float ad) {
    const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(block);
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int2 v = lookup_iq4(load_i32_b4(b->qs, 4 * group + j));
        sum = __dp4a(v.x, load_act<AStride>(aq, j), sum);
        sum = __dp4a(v.y, load_act<AStride>(aq, j + 4), sum);
    }
    const int scale = ((b->scales_l[group / 2] >> (4 * (group & 1))) & 15) |
                      (((b->scales_h >> (2 * group)) & 3) << 4);
    return (__half2float(b->d) * ad) * static_cast<float>(sum * (scale - 32));
}

// MTP gate/up: original Q2_K, not a requantized surrogate. Each group has two
// independent 16-value scale/min pairs, so retain two activation sums as well.
template <int AStride = 1>
__device__ __forceinline__ float dot_q2_k_32(const uint8_t* block, int group,
                                           const int8_t* aq, float ad) {
    const uint8_t* qs = block + 16 + (group / 4) * 32;
    const int shift = 2 * (group & 3);
    int dots[2] = {0, 0}, sums[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = (load_i32_b2(qs, j) >> shift) & 0x03030303;
        const int a = load_act<AStride>(aq, j);
        dots[j / 4] = __dp4a(q, a, dots[j / 4]);
        sums[j / 4] = __dp4a(0x01010101, a, sums[j / 4]);
    }
    const int s0 = block[2 * group], s1 = block[2 * group + 1];
    const float d = __half2float(*reinterpret_cast<const __half*>(block + 80));
    const float dm = __half2float(*reinterpret_cast<const __half*>(block + 82));
    const int weighted = (s0 & 15) * dots[0] + (s1 & 15) * dots[1];
    const int offset = (s0 >> 4) * sums[0] + (s1 >> 4) * sums[1];
    return ad * (d * static_cast<float>(weighted) - dm * static_cast<float>(offset));
}

__device__ __forceinline__ int q3_k_scale(const uint8_t* scales, int group16) {
    const int low = (scales[group16 % 8] >> (4 * (group16 / 8))) & 15;
    const int high = ((scales[8 + group16 % 4] >> (2 * (group16 / 4))) & 3) << 4;
    return (low | high) - 32;
}

// MTP down: Q3_K. The mask bit means a nonnegative low-two-bit value; an unset
// bit subtracts four. Signed-byte subtraction avoids carries between weights.
template <int AStride = 1>
__device__ __forceinline__ float dot_q3_k_32(const uint8_t* block, int group,
                                           const int8_t* aq, float ad) {
    const uint8_t* qs = block + 32 + (group / 4) * 32;
    const int shift = 2 * (group & 3);
    int sums[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int low = (load_i32_b2(qs, j) >> shift) & 0x03030303;
        const uint32_t inverted = ~static_cast<uint32_t>(load_i32_b2(block, j));
        const int high = static_cast<int>(((inverted >> group) << 2) & 0x04040404u);
        const int q = __vsub4(low, high);
        sums[j / 4] = __dp4a(q, load_act<AStride>(aq, j), sums[j / 4]);
    }
    const int s0 = q3_k_scale(block + 96, 2 * group);
    const int s1 = q3_k_scale(block + 96, 2 * group + 1);
    const float d = __half2float(*reinterpret_cast<const __half*>(block + 108));
    return (d * ad) * static_cast<float>(s0 * sums[0] + s1 * sums[1]);
}

template <uint32_t Type, int AStride = 1>
__device__ __forceinline__ float dot_32(const uint8_t* block, int group, const int8_t* aq, float ad) {
    static_assert(Type == T_IQ2_S || Type == T_IQ3_S || Type == T_IQ4_XS || Type == T_Q2_K || Type == T_Q3_K,
                  "unsupported native packed expert type");
    if constexpr (Type == T_Q2_K) return dot_q2_k_32<AStride>(block, group, aq, ad);
    if constexpr (Type == T_Q3_K) return dot_q3_k_32<AStride>(block, group, aq, ad);
    if constexpr (Type == T_IQ2_S) return dot_iq2_s_32<AStride>(block, group, aq, ad);
    if constexpr (Type == T_IQ3_S) return dot_iq3_s_32<AStride>(block, group, aq, ad);
    return dot_iq4_xs_32<AStride>(block, group, aq, ad);
}

// Reuse the original packed weights and codebook/sign expansion for several tokens. The integer dot
// sequence and final FP32 expression for each token match dot_32. Inactive pointers may be null; an
// expert's active mask is uniform across the warp, so skipped routes perform no activation loads.
template <uint32_t Type, int NT, bool Accumulate = false, int AStride = 1>
__device__ __forceinline__ void dot_32_multi(const uint8_t* block, int group,
                                             const int8_t* const (&aq)[NT], const float (&ad)[NT],
                                             unsigned active, float (&out)[NT]) {
    static_assert(NT >= 2 && NT <= 4, "native expert reuse supports 2..4 tokens");
    static_assert(Type == T_IQ2_S || Type == T_IQ3_S || Type == T_IQ4_XS || Type == T_Q2_K || Type == T_Q3_K,
                  "unsupported native packed expert type");
    if constexpr (!Accumulate) {
#pragma unroll
        for (int t = 0; t < NT; ++t) out[t] = 0.0f;
    }
    if constexpr (Type == T_IQ2_S) {
        const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(block);
        const int packed_qs = load_i32_b2(b->qs, group);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed_qs);
        const int packed_signs = load_i32_b2(b->qs, 8 + group);
        const uint8_t* signs = reinterpret_cast<const uint8_t*>(&packed_signs);
        const int qh = b->qh[group];
        int sums[NT][2] = {};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int index = qs[j] | ((qh << (8 - 2 * j)) & 0x300);
            const uint2 grid = reinterpret_cast<const uint2*>(iq2s_grid)[index];
            const int s0 = sign_low(signs[j]), s1 = sign_high(signs[j]);
            const int v0 = __vsub4(grid.x ^ s0, s0), v1 = __vsub4(grid.y ^ s1, s1);
#pragma unroll
            for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
                sums[t][j / 2] = __dp4a(v0, load_act<AStride>(aq[t], 2 * j), sums[t][j / 2]);
                sums[t][j / 2] = __dp4a(v1, load_act<AStride>(aq[t], 2 * j + 1), sums[t][j / 2]);
            }
        }
        const int s0 = 2 * (b->scales[group] & 15) + 1;
        const int s1 = 2 * (b->scales[group] >> 4) + 1;
        const float d = __half2float(b->d);
#pragma unroll
        for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
            if constexpr (Accumulate)
                out[t] += (d * ad[t]) * (0.125f * static_cast<float>(s0 * sums[t][0] + s1 * sums[t][1]));
            else out[t] = (d * ad[t]) * (0.125f * static_cast<float>(s0 * sums[t][0] + s1 * sums[t][1]));
        }
    } else if constexpr (Type == T_IQ3_S) {
        const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(block);
        const int2 packed_qs = make_int2(load_i32_b2(b->qs, 2 * group), load_i32_b2(b->qs, 2 * group + 1));
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&packed_qs);
        const int packed_signs = load_i32_b2(b->signs, group);
        const uint8_t* signs = reinterpret_cast<const uint8_t*>(&packed_signs);
        const int qh = b->qh[group];
        int sums[NT] = {};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int index0 = qs[2 * j] | ((qh << (8 - 2 * j)) & 0x100);
            const int index1 = qs[2 * j + 1] | ((qh << (7 - 2 * j)) & 0x100);
            const int s0 = sign_low(signs[j]), s1 = sign_high(signs[j]);
            const int v0 = __vsub4(iq3s_grid[index0] ^ s0, s0);
            const int v1 = __vsub4(iq3s_grid[index1] ^ s1, s1);
#pragma unroll
            for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
                sums[t] = __dp4a(v0, load_act<AStride>(aq[t], 2 * j), sums[t]);
                sums[t] = __dp4a(v1, load_act<AStride>(aq[t], 2 * j + 1), sums[t]);
            }
        }
        const int scale = 1 + 2 * ((b->scales[group / 2] >> (4 * (group & 1))) & 15);
        const float d = __half2float(b->d);
#pragma unroll
        for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
            if constexpr (Accumulate) out[t] += (d * ad[t]) * static_cast<float>(sums[t] * scale);
            else out[t] = (d * ad[t]) * static_cast<float>(sums[t] * scale);
        }
    } else if constexpr (Type == T_IQ4_XS) {
        const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(block);
        int sums[NT] = {};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = lookup_iq4(load_i32_b4(b->qs, 4 * group + j));
#pragma unroll
            for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
                sums[t] = __dp4a(v.x, load_act<AStride>(aq[t], j), sums[t]);
                sums[t] = __dp4a(v.y, load_act<AStride>(aq[t], j + 4), sums[t]);
            }
        }
        const int scale = ((b->scales_l[group / 2] >> (4 * (group & 1))) & 15) |
                          (((b->scales_h >> (2 * group)) & 3) << 4);
        const float d = __half2float(b->d);
#pragma unroll
        for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
            if constexpr (Accumulate) out[t] += (d * ad[t]) * static_cast<float>(sums[t] * (scale - 32));
            else out[t] = (d * ad[t]) * static_cast<float>(sums[t] * (scale - 32));
        }
    } else if constexpr (Type == T_Q2_K) {
        const uint8_t* qs = block + 16 + (group / 4) * 32;
        const int shift = 2 * (group & 3);
        int dots[NT][2] = {}, sums[NT][2] = {};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int q = (load_i32_b2(qs, j) >> shift) & 0x03030303;
#pragma unroll
            for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
                const int a = load_act<AStride>(aq[t], j);
                dots[t][j / 4] = __dp4a(q, a, dots[t][j / 4]);
                sums[t][j / 4] = __dp4a(0x01010101, a, sums[t][j / 4]);
            }
        }
        const int s0 = block[2 * group], s1 = block[2 * group + 1];
        const float d = __half2float(*reinterpret_cast<const __half*>(block + 80));
        const float dm = __half2float(*reinterpret_cast<const __half*>(block + 82));
#pragma unroll
        for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
            const int weighted = (s0 & 15) * dots[t][0] + (s1 & 15) * dots[t][1];
            const int offset = (s0 >> 4) * sums[t][0] + (s1 >> 4) * sums[t][1];
            if constexpr (Accumulate)
                out[t] += ad[t] * (d * static_cast<float>(weighted) - dm * static_cast<float>(offset));
            else out[t] = ad[t] * (d * static_cast<float>(weighted) - dm * static_cast<float>(offset));
        }
    } else if constexpr (Type == T_Q3_K) {
        const uint8_t* qs = block + 32 + (group / 4) * 32;
        const int shift = 2 * (group & 3);
        int sums[NT][2] = {};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int low = (load_i32_b2(qs, j) >> shift) & 0x03030303;
            const uint32_t inverted = ~static_cast<uint32_t>(load_i32_b2(block, j));
            const int high = static_cast<int>(((inverted >> group) << 2) & 0x04040404u);
            const int q = __vsub4(low, high);
#pragma unroll
            for (int t = 0; t < NT; ++t) if (active & (1u << t))
                sums[t][j / 4] = __dp4a(q, load_act<AStride>(aq[t], j), sums[t][j / 4]);
        }
        const int s0 = q3_k_scale(block + 96, 2 * group);
        const int s1 = q3_k_scale(block + 96, 2 * group + 1);
        const float d = __half2float(*reinterpret_cast<const __half*>(block + 108));
#pragma unroll
        for (int t = 0; t < NT; ++t) if (active & (1u << t)) {
            if constexpr (Accumulate) out[t] += (d * ad[t]) * static_cast<float>(s0 * sums[t][0] + s1 * sums[t][1]);
            else out[t] = (d * ad[t]) * static_cast<float>(s0 * sums[t][0] + s1 * sums[t][1]);
        }
    }
}

}  // namespace sq::iq_native
