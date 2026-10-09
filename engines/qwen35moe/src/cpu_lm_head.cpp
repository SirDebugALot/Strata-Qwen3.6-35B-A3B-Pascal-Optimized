#include "cpu_lm_head.hpp"

#include <immintrin.h>
#include <cstddef>
#include <stdexcept>

namespace sq {

namespace {

// Sign-extend before multiplying: maddubs would saturate for Q8 weights near +/-128.
inline int dot32_loaded(__m256i w0, __m256i w1, const int8_t* a) {
    const __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    const __m256i a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a + 16)));
    const __m256i v = _mm256_add_epi32(_mm256_madd_epi16(w0, a0), _mm256_madd_epi16(w1, a1));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

inline int dot32(const int8_t* w, const int8_t* a) {
    const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)w));
    const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + 16)));
    return dot32_loaded(w0, w1, a);
}

template <int NT>
void project_multi(const int8_t* wq, const uint16_t* wd, const int8_t* aq, const float* ad,
                   float* out, int cols, int rows, int begin, int end) {
    const int nb = cols / 32;
    for (int row = begin; row < end; ++row) {
        const int8_t* wr = wq + (size_t)row * cols;
        const uint16_t* dr = wd + (size_t)row * nb;
        float sum[NT] = {};
        for (int b = 0; b < nb; ++b) {
            // A verification window revisits the same large head matrix. Read and sign-extend the
            // weights once, then retain T=1's integer dot and FP32 expression for each token.
            const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(wr + b * 32)));
            const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(wr + b * 32 + 16)));
            const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dr[b])));
            for (int t = 0; t < NT; ++t)
                sum[t] += (float)dot32_loaded(w0, w1, aq + (size_t)t * cols + b * 32) * d * ad[(size_t)t * nb + b];
        }
        for (int t = 0; t < NT; ++t) out[(size_t)t * rows + row] = sum[t];
    }
}

}  // namespace

void cpu_lm_head_q8_rows(const int8_t* wq, const uint16_t* wd, const int8_t* aq, const float* ad,
                         float* out, int cols, int begin, int end) {
    const int nb = cols / 32;
    for (int row = begin; row < end; ++row) {
        const int8_t* wr = wq + (size_t)row * cols;
        const uint16_t* dr = wd + (size_t)row * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dr[b])));
            sum += (float)dot32(wr + b * 32, aq + b * 32) * d * ad[b];
        }
        out[row] = sum;
    }
}

void cpu_lm_head_q8_rows_multi(const int8_t* wq, const uint16_t* wd, const int8_t* aq, const float* ad,
                              float* out, int cols, int rows, int begin, int end, int nt) {
    switch (nt) {
        case 1: cpu_lm_head_q8_rows(wq, wd, aq, ad, out, cols, begin, end); break;
        case 2: project_multi<2>(wq, wd, aq, ad, out, cols, rows, begin, end); break;
        case 3: project_multi<3>(wq, wd, aq, ad, out, cols, rows, begin, end); break;
        case 4: project_multi<4>(wq, wd, aq, ad, out, cols, rows, begin, end); break;
        default: throw std::invalid_argument("CPU LM head token count must be 1..4");
    }
}

}  // namespace sq
