// Exact lossless IQ3_S nibble path, validated against the original native IQ3 row kernels.
// IQ3 grid is ggml's original MIT-licensed table; see third_party/ggml/LICENSE.
#include "cpu_iq3_nibble.hpp"
#include "cpu_moe.hpp"
#include "common.hpp"
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../../../third_party/ggml/ggml-common.h"
namespace sq::native_iq {
void decode_iq3_nibbles(const uint8_t* source, Iq3NibbleBlock* decoded, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* b = source + i * 110;
        Iq3NibbleBlock& d = decoded[i];
        std::memcpy(&d.d, b, 2);
        for (int h = 0; h < 8; ++h) {
            d.subscales[h] = (int8_t)(2 * ((b[106 + h / 2] >> (4 * (h & 1))) & 15) + 1);
            uint8_t codes[32];
            for (int group = 0; group < 8; ++group) {
                const int index = b[2 + 8 * h + group] | (((b[66 + h] >> group) & 1) << 8);
                const uint32_t grid = iq3s_grid[index];
                for (int lane = 0; lane < 4; ++lane) {
                    const int k = 4 * group + lane;
                    const int magnitude = (grid >> (8 * lane)) & 255;
                    const int negative = (b[74 + 4 * h + k / 8] >> (k & 7)) & 1;
                    codes[k] = (uint8_t)(((magnitude - 1) / 2) | (negative << 3));
                }
            }
            for (int k = 0; k < 16; ++k) d.coefficients[16 * h + k] = codes[k] | (codes[k + 16] << 4);
        }
    }
}

namespace {
inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline __m256i coefficients(const Iq3NibbleBlock& b, int h) {
    const __m128i table = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15);
    const __m128i packed = _mm_loadu_si128((const __m128i*)(b.coefficients + 16 * h));
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i low = _mm_shuffle_epi8(table, _mm_and_si128(packed, mask));
    const __m128i high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
}

template<int NT>
void rows_impl(const Iq3NibbleBlock* weights, int nb, const sq::Q8K* const* y, float* const* out, int r0, int r1) {
    for (int row = r0; row < r1; ++row) {
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const Iq3NibbleBlock& block = weights[(size_t)row * nb + i];
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
void iq3_nibble_rows(const Iq3NibbleBlock* w, int nb, const sq::Q8K* const* y, int nt, float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows_impl<1>(w, nb, y, out, r0, r1); break;
        case 2: rows_impl<2>(w, nb, y, out, r0, r1); break;
        case 3: rows_impl<3>(w, nb, y, out, r0, r1); break;
        case 4: rows_impl<4>(w, nb, y, out, r0, r1); break;
    }
}

bool iq3_nibble_wide_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_CPU_IQ3_WIDE");
        if (!value) return false;
        SQ_CHECK(std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0,
                 "STRATA_CPU_IQ3_WIDE must be 0 or 1");
        return value[0] == '1';
    }();
    return enabled;
}

namespace {
inline __m256i wide_coefficients(const Iq3NibbleBlock& b, int h) {
    const __m128i table = _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i packed = _mm_loadu_si128((const __m128i*)(b.coefficients + 16 * h));
    const __m256i repeated = _mm256_broadcastsi128_si256(packed);
    const __m256i shifts = _mm256_setr_epi32(0,0,0,0,4,4,4,4);
    const __m256i codes = _mm256_and_si256(_mm256_srlv_epi32(repeated, shifts), _mm256_set1_epi8(15));
    return _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(table), codes);
}

// Validated V2 wide+PF2048 specialization. Keep every native integer/FMA/horizontal-sum sequence;
// notably vpsignb at activation -128 is intentional and must not become a signed*signed dot.
template<int NT>
void wide_prefetch_rows_impl(const Iq3NibbleBlock* weights, int nb, const sq::Q8K* const* y,
                             float* const* out, int r0, int r1) {
    for (int row = r0; row < r1; ++row) {
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const size_t index = (size_t)row * nb + i;
            // Nonfaulting x86 hints. Integer addition avoids out-of-object C++ pointer arithmetic.
            const uintptr_t address = reinterpret_cast<uintptr_t>(weights + index);
            _mm_prefetch(reinterpret_cast<const char*>(address + 2048), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(address + 2112), _MM_HINT_T0);
            const Iq3NibbleBlock& block = weights[index];
            __m256i acci[NT];
            for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
            for (int h = 0; h < 8; ++h) {
                const __m256i w = wide_coefficients(block, h);
                const __m256i magnitude = _mm256_abs_epi8(w);
                const __m256i scales = _mm256_set1_epi16(block.subscales[h]);
                for (int t = 0; t < NT; ++t) {
                    const __m256i a = _mm256_loadu_si256((const __m256i*)(y[t][i].nat + 32 * h));
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

void iq3_nibble_rows_wide_prefetch(const Iq3NibbleBlock* w, int nb, const sq::Q8K* const* y,
                                  int nt, float* const* out, int r0, int r1) {
    SQ_CHECK(nb > 0 && r0 >= 0 && r1 >= r0, "invalid IQ3 wide row extent");
    switch (nt) {
        case 1: wide_prefetch_rows_impl<1>(w, nb, y, out, r0, r1); break;
        case 2: wide_prefetch_rows_impl<2>(w, nb, y, out, r0, r1); break;
        case 3: wide_prefetch_rows_impl<3>(w, nb, y, out, r0, r1); break;
        case 4: wide_prefetch_rows_impl<4>(w, nb, y, out, r0, r1); break;
        default: die("IQ3 wide rows require NT1..4");
    }
}
}
