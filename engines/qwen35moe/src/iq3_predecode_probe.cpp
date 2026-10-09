// Lossless CPU-only IQ3_S predecode experiment, independent of the production dispatcher.
// IQ3 grid is ggml's original MIT-licensed table; see third_party/ggml/LICENSE.
#include "iq3_predecode_probe.hpp"
#include "cpu_moe.hpp"
#include <immintrin.h>
#include <cstring>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../../../third_party/ggml/ggml-common.h"

namespace iq3_probe {
void decode(const uint8_t* source, Block* decoded, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* b = source + i * 110;
        Block& d = decoded[i];
        std::memcpy(&d.d, b, 2);
        for (int h = 0; h < 8; ++h) {
            d.subscales[h] = (int8_t)(2 * ((b[106 + h / 2] >> (4 * (h & 1))) & 15) + 1);
            for (int group = 0; group < 8; ++group) {
                const int index = b[2 + 8 * h + group] | (((b[66 + h] >> group) & 1) << 8);
                const uint32_t grid = iq3s_grid[index];
                for (int lane = 0; lane < 4; ++lane) {
                    const int k = 4 * group + lane;
                    const int magnitude = (grid >> (8 * lane)) & 255;
                    const bool negative = (b[74 + 4 * h + k / 8] >> (k & 7)) & 1;
                    d.coefficients[32 * h + k] = (int8_t)(negative ? -magnitude : magnitude);
                }
            }
        }
    }
}

void decode_nibbles(const uint8_t* source, NibbleBlock* output, size_t blocks) {
    sq::native_iq::decode_iq3_nibbles(source, output, blocks);
}

namespace {
inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline __m256i coefficients(const Block& b, int h) {
    return _mm256_loadu_si256((const __m256i*)(b.coefficients + 32 * h));
}


template<int NT, class WeightBlock>
void rows_impl(const WeightBlock* weights, int nb, const sq::Q8K* const* y, float* const* out, int r0, int r1) {
    for (int row = r0; row < r1; ++row) {
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const WeightBlock& block = weights[(size_t)row * nb + i];
            __m256i acci[NT];
            for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
            for (int h = 0; h < 8; ++h) {
                const __m256i w = coefficients(block, h);
                const __m256i magnitude = _mm256_abs_epi8(w);
                const __m256i scales = _mm256_set1_epi16(block.subscales[h]);
                for (int t = 0; t < NT; ++t) {
                    const __m256i a = _mm256_loadu_si256((const __m256i*)(y[t][i].nat + 32 * h));
                    // Keep native vpsignb behavior at -128. A signed*signed mathematical dot would differ.
                    const __m256i signed_a = _mm256_sign_epi8(a, w);
                    const __m256i pairs = _mm256_maddubs_epi16(magnitude, signed_a);
                    acci[t] = _mm256_add_epi32(acci[t], _mm256_madd_epi16(pairs, scales));
                }
            }
            const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(block.d)));
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][row] = hsum8(accf[t]);
    }
}
}

void rows(const Block* w, int nb, const sq::Q8K* const* y, int nt, float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows_impl<1>(w, nb, y, out, r0, r1); break;
        case 2: rows_impl<2>(w, nb, y, out, r0, r1); break;
        case 3: rows_impl<3>(w, nb, y, out, r0, r1); break;
        case 4: rows_impl<4>(w, nb, y, out, r0, r1); break;
    }
}
void rows_nibbles(const NibbleBlock* w, int nb, const sq::Q8K* const* y, int nt, float* const* out, int r0, int r1) {
    sq::native_iq::iq3_nibble_rows(w, nb, y, nt, out, r0, r1);
}
}
