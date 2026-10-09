// cpu_avx512.cpp - the AVX-512 VNNI expert kernels (compiled with AVX-512 flags; chosen at run time by cpu_moe.cpp).
//
// Every K-quant dot product uses the same trick: the weight quants are UNSIGNED (Q4_K 0..15, Q5_K 0..31,
// Q6_K 0..63 before its -32 offset) and the activations are signed int8, which is exactly the operand shape of
// VPDPBUSD.  Offsets (Q4_K/Q5_K mins, Q6_K's -32) are applied afterwards from the activation block sums.
#include "cpu_kernels.hpp"

#include <immintrin.h>

#include <cstring>

namespace sq::avx512 {

// ================================================================== quantization
void quantize_q8k(const float* x, Q8K* y, int n) {
    for (int i = 0; i < n / 256; ++i) {
        const float* xb = x + i * 256;
        Q8K& b = y[i];
        __m512 vmax = _mm512_setzero_ps();
        for (int k = 0; k < 16; ++k) vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(xb + 16 * k)));
        const float amax = _mm512_reduce_max_ps(vmax);
        if (amax == 0.0f) {
            std::memset(&b, 0, sizeof(Q8K));
            continue;
        }
        const __m512 id = _mm512_set1_ps(127.0f / amax);
        for (int k = 0; k < 16; ++k) {
            const __m512i iv = _mm512_cvt_roundps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xb + 16 * k), id),
                                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i*)(b.nat + 16 * k), _mm512_cvtsepi32_epi8(iv));
            b.bsums[k] = (int16_t)_mm512_reduce_add_epi32(iv);
        }
        for (int h = 0; h < 2; ++h) {
            const int8_t* s = b.nat + 128 * h;
            int8_t* d = b.pair + 128 * h;
            std::memcpy(d + 0, s + 0, 32);
            std::memcpy(d + 32, s + 64, 32);
            std::memcpy(d + 64, s + 32, 32);
            std::memcpy(d + 96, s + 96, 32);
        }
        b.d = amax / 127.0f;
    }
}

// ================================================================== dot products
namespace {

// fp16 -> f32 with F16C (every AVX2 CPU has it).  quant.hpp's inline fp16_to_f32 is deliberately not used here: an
// inline function compiled in this file gets this file's instruction set, and the linker may keep that copy for the
// whole program.
inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

inline __m512i perm_idx(int a, int b) {
    return _mm512_setr_epi32(a, a, a, a, a, a, a, a, b, b, b, b, b, b, b, b);
}

inline float hsum_epi32_128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
    return (float)_mm_cvtsi128_si32(v);
}

/// Unpacks the 12 packed scale bytes of Q4_K/Q5_K into 8 scales (bytes 0..7) and 8 mins (bytes 8..15).
inline void unpack_scales_k4(const uint8_t* s12, uint32_t utmp[4]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    std::memcpy(utmp, s12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

/// Sum over the 8 sub-blocks of mins[s] * (bsums[2s] + bsums[2s+1]).
inline int32_t min_dot(const uint32_t utmp[4], const int16_t* bsums) {
    const __m128i mins16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(utmp + 2)));
    const __m128i bs = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*)bsums), _mm_loadu_si128((const __m128i*)(bsums + 8)));
    return (int32_t)hsum_epi32_128(_mm_madd_epi16(mins16, bs));
}

}  // namespace

float dot_q4k(const void* wv, const Q8K* a, int nb) {
    const BlockQ4_K* w = (const BlockQ4_K*)wv;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i i02 = perm_idx(0, 2), i13 = perm_idx(1, 3), i46 = perm_idx(4, 6), i57 = perm_idx(5, 7);
    __m512 acc = _mm512_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint32_t utmp[4];
        unpack_scales_k4(w[i].scales, utmp);
        const __m512i sc32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)utmp));
        const float d = h2f(w[i].d) * a[i].d;
        const float dmin = h2f(w[i].dmin) * a[i].d;
        accm += dmin * (float)min_dot(utmp, a[i].bsums);
        const __m512i q0 = _mm512_loadu_si512(w[i].qs), q1 = _mm512_loadu_si512(w[i].qs + 64);
        const __m512i* ap = (const __m512i*)a[i].pair;
        const __m512i z = _mm512_setzero_si512();
        const __m512i p0 = _mm512_dpbusd_epi32(z, _mm512_and_si512(q0, m4), _mm512_load_si512(ap + 0));
        const __m512i p1 = _mm512_dpbusd_epi32(z, _mm512_and_si512(_mm512_srli_epi16(q0, 4), m4), _mm512_load_si512(ap + 1));
        const __m512i p2 = _mm512_dpbusd_epi32(z, _mm512_and_si512(q1, m4), _mm512_load_si512(ap + 2));
        const __m512i p3 = _mm512_dpbusd_epi32(z, _mm512_and_si512(_mm512_srli_epi16(q1, 4), m4), _mm512_load_si512(ap + 3));
        __m512i s = _mm512_mullo_epi32(p0, _mm512_permutexvar_epi32(i02, sc32));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p1, _mm512_permutexvar_epi32(i13, sc32)));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p2, _mm512_permutexvar_epi32(i46, sc32)));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p3, _mm512_permutexvar_epi32(i57, sc32)));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(d), acc);
    }
    return _mm512_reduce_add_ps(acc) - accm;
}

float dot_q5k(const void* wv, const Q8K* a, int nb) {
    const BlockQ5_K* w = (const BlockQ5_K*)wv;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i v16 = _mm512_set1_epi8(16);
    const __m512i i02 = perm_idx(0, 2), i13 = perm_idx(1, 3), i46 = perm_idx(4, 6), i57 = perm_idx(5, 7);
    auto sel = [](int lo, int hi) {
        return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8((char)lo)), _mm256_set1_epi8((char)hi), 1);
    };
    const __m512i s02 = sel(0x01, 0x04), s13 = sel(0x02, 0x08), s46 = sel(0x10, 0x40), s57 = sel(0x20, 0x80);
    __m512 acc = _mm512_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint32_t utmp[4];
        unpack_scales_k4(w[i].scales, utmp);
        const __m512i sc32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)utmp));
        const float d = h2f(w[i].d) * a[i].d;
        const float dmin = h2f(w[i].dmin) * a[i].d;
        accm += dmin * (float)min_dot(utmp, a[i].bsums);
        const __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)w[i].qh));
        const __m512i q0 = _mm512_loadu_si512(w[i].qs), q1 = _mm512_loadu_si512(w[i].qs + 64);
        __m512i l0 = _mm512_and_si512(q0, m4), h0 = _mm512_and_si512(_mm512_srli_epi16(q0, 4), m4);
        __m512i l1 = _mm512_and_si512(q1, m4), h1 = _mm512_and_si512(_mm512_srli_epi16(q1, 4), m4);
        l0 = _mm512_mask_add_epi8(l0, _mm512_test_epi8_mask(qh, s02), l0, v16);
        h0 = _mm512_mask_add_epi8(h0, _mm512_test_epi8_mask(qh, s13), h0, v16);
        l1 = _mm512_mask_add_epi8(l1, _mm512_test_epi8_mask(qh, s46), l1, v16);
        h1 = _mm512_mask_add_epi8(h1, _mm512_test_epi8_mask(qh, s57), h1, v16);
        const __m512i* ap = (const __m512i*)a[i].pair;
        const __m512i z = _mm512_setzero_si512();
        const __m512i p0 = _mm512_dpbusd_epi32(z, l0, _mm512_load_si512(ap + 0));
        const __m512i p1 = _mm512_dpbusd_epi32(z, h0, _mm512_load_si512(ap + 1));
        const __m512i p2 = _mm512_dpbusd_epi32(z, l1, _mm512_load_si512(ap + 2));
        const __m512i p3 = _mm512_dpbusd_epi32(z, h1, _mm512_load_si512(ap + 3));
        __m512i s = _mm512_mullo_epi32(p0, _mm512_permutexvar_epi32(i02, sc32));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p1, _mm512_permutexvar_epi32(i13, sc32)));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p2, _mm512_permutexvar_epi32(i46, sc32)));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p3, _mm512_permutexvar_epi32(i57, sc32)));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(d), acc);
    }
    return _mm512_reduce_add_ps(acc) - accm;
}

float dot_q6k(const void* wv, const Q8K* a, int nb) {
    const BlockQ6_K* w = (const BlockQ6_K*)wv;
    const __m512i m4 = _mm512_set1_epi8(0x0F), m3 = _mm512_set1_epi8(0x03);
    auto cnt = [](int lo, int hi) {
        return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16((short)lo)), _mm256_set1_epi16((short)hi), 1);
    };
    const __m512i c_lo = cnt(0, 2), c_hi = cnt(4, 6);
    // scale index for lane l of the product covering values [64k + 4l, 64k + 4l + 4): (64k + 4l) / 16
    const __m512i ix[4] = {
        _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
        _mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7),
        _mm512_setr_epi32(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11),
        _mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15),
    };
    __m512 acc = _mm512_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const __m128i sc8 = _mm_loadu_si128((const __m128i*)w[i].scales);
        const __m512i sc32 = _mm512_cvtepi8_epi32(sc8);
        const __m256i off = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc8), _mm256_loadu_si256((const __m256i*)a[i].bsums));
        const int32_t isum_off = (int32_t)hsum_epi32_128(_mm_add_epi32(_mm256_castsi256_si128(off), _mm256_extracti128_si256(off, 1)));
        const float d = h2f(w[i].d) * a[i].d;
        accm += d * 32.0f * (float)isum_off;
        __m512i s = _mm512_setzero_si512();
        for (int h = 0; h < 2; ++h) {
            const __m512i L = _mm512_loadu_si512(w[i].ql + 64 * h);
            const __m512i H = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)(w[i].qh + 32 * h)));
            const __m512i vlo = _mm512_or_si512(_mm512_and_si512(L, m4),
                                                _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(H, c_lo), m3), 4));
            const __m512i vhi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(L, 4), m4),
                                                _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(H, c_hi), m3), 4));
            const __m512i z = _mm512_setzero_si512();
            const __m512i plo = _mm512_dpbusd_epi32(z, vlo, _mm512_load_si512((const __m512i*)(a[i].nat + 128 * h)));
            const __m512i phi = _mm512_dpbusd_epi32(z, vhi, _mm512_load_si512((const __m512i*)(a[i].nat + 128 * h + 64)));
            s = _mm512_add_epi32(s, _mm512_mullo_epi32(plo, _mm512_permutexvar_epi32(ix[2 * h + 0], sc32)));
            s = _mm512_add_epi32(s, _mm512_mullo_epi32(phi, _mm512_permutexvar_epi32(ix[2 * h + 1], sc32)));
        }
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(d), acc);
    }
    return _mm512_reduce_add_ps(acc) - accm;
}

}  // namespace sq::avx512
