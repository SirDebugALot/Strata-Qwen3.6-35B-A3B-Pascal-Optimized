// Native Q2_K/Q3_K rows for the MTP block. Layout/arithmetic follow ggml
// (MIT, third_party/ggml/LICENSE); the existing Q8K activation ABI is preserved.
#include "cpu_iq.hpp"
#include "common.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>

namespace sq::native_iq {
namespace {
inline int sum4(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4e));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xb1));
    return _mm_cvtsi128_si32(v);
}
inline int dot16(__m128i w, const int8_t* a) {
    const __m256i wi = _mm256_cvtepi8_epi16(w);
    const __m256i ai = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    const __m256i d = _mm256_madd_epi16(wi, ai);
    return sum4(_mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1)));
}
inline float half_at(const uint8_t* p) {
    uint16_t h; std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}
}
float dot_q2k(const void* vw, const Q8K* a, int nb) {
    const auto* w = (const uint8_t*)vw;
    float result = 0.f;
    for (int b = 0; b < nb; ++b, w += 84) {
        int dot = 0, mins = 0;
        for (int h = 0; h < 8; ++h) {
            const auto packed = _mm256_loadu_si256((const __m256i*)(w + 16 + (h / 4) * 32));
            const auto q = _mm256_and_si256(_mm256_srl_epi16(packed, _mm_cvtsi32_si128(2 * (h % 4))), _mm256_set1_epi8(3));
            dot += (w[2*h] & 15) * dot16(_mm256_castsi256_si128(q), a[b].nat + h*32);
            dot += (w[2*h+1] & 15) * dot16(_mm256_extracti128_si256(q, 1), a[b].nat + h*32+16);
            mins += (w[2*h] >> 4) * a[b].bsums[2*h] + (w[2*h+1] >> 4) * a[b].bsums[2*h+1];
        }
        result += a[b].d * (half_at(w+80) * dot - half_at(w+82) * mins);
    }
    return result;
}
float dot_q3k(const void* vw, const Q8K* a, int nb) {
    const auto* w = (const uint8_t*)vw;
    float result = 0.f;
    for (int b = 0; b < nb; ++b, w += 110) {
        uint32_t scales[4]; std::memcpy(scales, w+96, 12);
        const uint32_t tmp = scales[2], m1=0x03030303, m2=0x0f0f0f0f;
        scales[2]=((scales[0]>>4)&m2)|(((tmp>>4)&m1)<<4);
        scales[3]=((scales[1]>>4)&m2)|(((tmp>>6)&m1)<<4);
        scales[0]=(scales[0]&m2)|(((tmp>>0)&m1)<<4);
        scales[1]=(scales[1]&m2)|(((tmp>>2)&m1)<<4);
        const auto* sc = (const uint8_t*)scales;
        const auto high = _mm256_loadu_si256((const __m256i*)w);
        int dot = 0;
        for (int h = 0; h < 8; ++h) {
            const auto packed = _mm256_loadu_si256((const __m256i*)(w + 32 + (h/4)*32));
            const auto lo = _mm256_and_si256(_mm256_srl_epi16(packed, _mm_cvtsi32_si128(2*(h%4))), _mm256_set1_epi8(3));
            const auto neg = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(high, _mm256_set1_epi8((char)(1<<h))), _mm256_setzero_si256()), _mm256_set1_epi8(4));
            const auto q = _mm256_sub_epi8(lo, neg);
            dot += (int(sc[2*h])-32) * dot16(_mm256_castsi256_si128(q), a[b].nat+h*32);
            dot += (int(sc[2*h+1])-32) * dot16(_mm256_extracti128_si256(q,1), a[b].nat+h*32+16);
        }
        result += half_at(w+108) * a[b].d * dot;
    }
    return result;
}

namespace {
// Identical integer chain to dot16, except the weight extension is shared by all active tokens.
inline int dot16_wide(__m256i wi, const int8_t* a) {
    const __m256i ai = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    const __m256i d = _mm256_madd_epi16(wi, ai);
    return sum4(_mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1)));
}

template <int NT>
void dot_q2k_multi(const uint8_t* w, const Q8K* const* a, int nb, float* result) {
    for (int t = 0; t < NT; ++t) result[t] = 0.f;
    for (int b = 0; b < nb; ++b, w += 84) {
        int dot[NT] = {}, mins[NT] = {};
        const float d = half_at(w + 80), dmin = half_at(w + 82);
        for (int h = 0; h < 8; ++h) {
            const auto packed = _mm256_loadu_si256((const __m256i*)(w + 16 + (h / 4) * 32));
            const auto q = _mm256_and_si256(_mm256_srl_epi16(packed, _mm_cvtsi32_si128(2 * (h % 4))), _mm256_set1_epi8(3));
            const auto low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            const auto high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q, 1));
            const int scale0 = w[2*h] & 15, scale1 = w[2*h+1] & 15;
            const int min0 = w[2*h] >> 4, min1 = w[2*h+1] >> 4;
            for (int t = 0; t < NT; ++t) {
                dot[t] += scale0 * dot16_wide(low, a[t][b].nat + h*32);
                dot[t] += scale1 * dot16_wide(high, a[t][b].nat + h*32+16);
                mins[t] += min0 * a[t][b].bsums[2*h] + min1 * a[t][b].bsums[2*h+1];
            }
        }
        // Preserve both the per-superblock FP expression and the outer accumulation order of dot_q2k.
        for (int t = 0; t < NT; ++t)
            result[t] += a[t][b].d * (d * dot[t] - dmin * mins[t]);
    }
}

template <int NT>
void dot_q3k_multi(const uint8_t* w, const Q8K* const* a, int nb, float* result) {
    for (int t = 0; t < NT; ++t) result[t] = 0.f;
    for (int b = 0; b < nb; ++b, w += 110) {
        uint32_t scales[4]; std::memcpy(scales, w+96, 12);
        const uint32_t tmp = scales[2], m1=0x03030303, m2=0x0f0f0f0f;
        scales[2]=((scales[0]>>4)&m2)|(((tmp>>4)&m1)<<4);
        scales[3]=((scales[1]>>4)&m2)|(((tmp>>6)&m1)<<4);
        scales[0]=(scales[0]&m2)|(((tmp>>0)&m1)<<4);
        scales[1]=(scales[1]&m2)|(((tmp>>2)&m1)<<4);
        const auto* sc = (const uint8_t*)scales;
        const auto high_bits = _mm256_loadu_si256((const __m256i*)w);
        const float d = half_at(w + 108);
        int dot[NT] = {};
        for (int h = 0; h < 8; ++h) {
            const auto packed = _mm256_loadu_si256((const __m256i*)(w + 32 + (h/4)*32));
            const auto lo = _mm256_and_si256(_mm256_srl_epi16(packed, _mm_cvtsi32_si128(2*(h%4))), _mm256_set1_epi8(3));
            const auto neg = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(high_bits, _mm256_set1_epi8((char)(1<<h))), _mm256_setzero_si256()), _mm256_set1_epi8(4));
            const auto q = _mm256_sub_epi8(lo, neg);
            const auto low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            const auto high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q, 1));
            const int scale0 = int(sc[2*h])-32, scale1 = int(sc[2*h+1])-32;
            for (int t = 0; t < NT; ++t) {
                dot[t] += scale0 * dot16_wide(low, a[t][b].nat+h*32);
                dot[t] += scale1 * dot16_wide(high, a[t][b].nat+h*32+16);
            }
        }
        for (int t = 0; t < NT; ++t) result[t] += d * a[t][b].d * dot[t];
    }
}

template <uint32_t Type, int NT>
void k23_dot(const uint8_t* weights, const Q8K* const* acts, int blocks, float* result) {
    if constexpr (NT == 1)
        result[0] = Type == T_Q2_K ? dot_q2k(weights, acts[0], blocks) : dot_q3k(weights, acts[0], blocks);
    else if constexpr (Type == T_Q2_K) dot_q2k_multi<NT>(weights, acts, blocks, result);
    else dot_q3k_multi<NT>(weights, acts, blocks, result);
}

template <uint32_t Type, int NT, bool GateUp>
void k23_apply_rows(const uint8_t* weights, size_t row_bytes, size_t up_offset, int cols,
                     const void* const* inputs, float* const* outputs, int r0, int r1) {
    const Q8K* acts[NT];
    for (int t = 0; t < NT; ++t) acts[t] = (const Q8K*)inputs[t];
    float gate[NT], up[NT];
    for (int row = r0; row < r1; ++row) {
        k23_dot<Type, NT>(weights + row * row_bytes, acts, cols / 256, gate);
        if constexpr (GateUp) {
            k23_dot<Type, NT>(weights + up_offset + row * row_bytes, acts, cols / 256, up);
            for (int t = 0; t < NT; ++t) outputs[t][row] = (gate[t] / (1.f + std::exp(-gate[t]))) * up[t];
        } else {
            for (int t = 0; t < NT; ++t) outputs[t][row] = gate[t];
        }
    }
}

template <uint32_t Type, bool GateUp>
void k23_dispatch(int nt, const uint8_t* weights, size_t row_bytes, size_t up_offset, int cols,
                    const void* const* acts, float* const* outputs, int r0, int r1) {
    switch (nt) {
        case 1: k23_apply_rows<Type, 1, GateUp>(weights, row_bytes, up_offset, cols, acts, outputs, r0, r1); break;
        case 2: k23_apply_rows<Type, 2, GateUp>(weights, row_bytes, up_offset, cols, acts, outputs, r0, r1); break;
        case 3: k23_apply_rows<Type, 3, GateUp>(weights, row_bytes, up_offset, cols, acts, outputs, r0, r1); break;
        case 4: k23_apply_rows<Type, 4, GateUp>(weights, row_bytes, up_offset, cols, acts, outputs, r0, r1); break;
        default: die("Q2/Q3 shared rows require 1..4 tokens, got %d", nt);
    }
}
} // namespace

void k23_rows(uint32_t type, const uint8_t* weights, size_t row_bytes, int cols, const void* const* acts,
              int nt, float* const* outputs, int r0, int r1) {
    SQ_CHECK(k23_supported(type) && cols > 0 && cols % 256 == 0 && r0 >= 0 && r1 >= r0, "invalid Q2/Q3 row call");
    if (type == T_Q2_K) k23_dispatch<T_Q2_K, false>(nt, weights, row_bytes, 0, cols, acts, outputs, r0, r1);
    else k23_dispatch<T_Q3_K, false>(nt, weights, row_bytes, 0, cols, acts, outputs, r0, r1);
}

void k23_gu_rows(uint32_t type, const uint8_t* weights, size_t row_bytes, size_t up_offset, int cols,
                 const void* const* acts, int nt, float* const* outputs, int r0, int r1) {
    SQ_CHECK(k23_supported(type) && cols > 0 && cols % 256 == 0 && r0 >= 0 && r1 >= r0, "invalid Q2/Q3 gate/up call");
    if (type == T_Q2_K) k23_dispatch<T_Q2_K, true>(nt, weights, row_bytes, up_offset, cols, acts, outputs, r0, r1);
    else k23_dispatch<T_Q3_K, true>(nt, weights, row_bytes, up_offset, cols, acts, outputs, r0, r1);
}
}
