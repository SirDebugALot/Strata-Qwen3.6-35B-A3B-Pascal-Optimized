// quant.hpp - GGML block layouts and scalar reference dequantizers.
// Block layouts follow ggml-common.h (MIT, ggml authors). Native expert mode keeps
// IQ2_S/IQ3_S/IQ4_XS packed; legacy mode imports unsupported types into Q6_K.
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

namespace sq {

enum GgmlType : uint32_t {
    T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q2_K = 10, T_Q3_K = 11, T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30,
};

constexpr int QK_K = 256;
constexpr int QK8_0 = 32;

#pragma pack(push, 1)
struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };                                         // 34 B / 32
struct BlockQ4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };            // 144 B / 256
struct BlockQ5_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };  // 176 B / 256
struct BlockQ6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };   // 210 B / 256
#pragma pack(pop)
static_assert(sizeof(BlockQ8_0) == 34);
static_assert(sizeof(BlockQ4_K) == 144);
static_assert(sizeof(BlockQ5_K) == 176);
static_assert(sizeof(BlockQ6_K) == 210);

inline const char* type_name(uint32_t t) {
    switch (t) {
        case T_Q2_K: return "Q2_K"; case T_Q3_K: return "Q3_K";
        case T_IQ2_S: return "IQ2_S"; case T_IQ3_S: return "IQ3_S"; case T_IQ4_XS: return "IQ4_XS";
        case T_F32: return "F32"; case T_F16: return "F16"; case T_Q8_0: return "Q8_0";
        case T_Q4_K: return "Q4_K"; case T_Q5_K: return "Q5_K"; case T_Q6_K: return "Q6_K"; case T_BF16: return "BF16";
        default: return "?";
    }
}

/// Bytes of one row of `n` values.
inline int64_t row_bytes(uint32_t t, int64_t n) {
    switch (t) {
        case T_F32: return n * 4;
        case T_F16: case T_BF16: return n * 2;
        case T_Q8_0: return n / 32 * 34;
        case T_Q2_K: return n / 256 * 84;
        case T_Q3_K: case T_IQ3_S: return n / 256 * 110;
        case T_IQ2_S: return n / 256 * 82;
        case T_IQ4_XS: return n / 256 * 136;
        case T_Q4_K: return n / 256 * 144;
        case T_Q5_K: return n / 256 * 176;
        case T_Q6_K: return n / 256 * 210;
        default: return -1;
    }
}

inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {  // subnormal
            int e = -1;
            do { man <<= 1; ++e; } while (!(man & 0x400));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000;
        const int shift = 14 - exp;
        uint32_t r = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    return (uint16_t)(sign | r);
}

/// The 6-bit (scale, min) pair `j` of a Q4_K / Q5_K super-block.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// ------------------------------------------------------------------ scalar reference dequantizers
inline void dequant_q8_0(const void* src, float* y, int64_t n) {
    const BlockQ8_0* b = (const BlockQ8_0*)src;
    for (int64_t i = 0; i < n / 32; ++i) {
        const float d = fp16_to_f32(b[i].d);
        for (int j = 0; j < 32; ++j) y[i * 32 + j] = d * b[i].qs[j];
    }
}

inline void dequant_q4_K(const void* src, float* y, int64_t n) {
    const BlockQ4_K* x = (const BlockQ4_K*)src;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const uint8_t* q = x[i].qs;
        const float d = fp16_to_f32(x[i].d), mn = fp16_to_f32(x[i].dmin);
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, sc, m);
            const float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, x[i].scales, sc, m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

inline void dequant_q5_K(const void* src, float* y, int64_t n) {
    const BlockQ5_K* x = (const BlockQ5_K*)src;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d = fp16_to_f32(x[i].d), mn = fp16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, sc, m);
            const float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, x[i].scales, sc, m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

inline void dequant_q6_K(const void* src, float* y, int64_t n) {
    const BlockQ6_K* x = (const BlockQ6_K*)src;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = fp16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        for (int nn = 0; nn < QK_K; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// Import-only formats and the reference Q6_K quantizer, implemented in quant_import.cpp.
void dequant_import(uint32_t type, const void* src, float* y, int64_t n);
void quantize_row_q6_K(const float* x, void* dst, int64_t n, bool fast = false);

inline void dequant_row(uint32_t type, const void* src, float* y, int64_t n) {
    switch (type) {
        case T_F32: std::memcpy(y, src, n * 4); break;
        case T_F16: for (int64_t i = 0; i < n; ++i) y[i] = fp16_to_f32(((const uint16_t*)src)[i]); break;
        case T_BF16:
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t b = (uint32_t)((const uint16_t*)src)[i] << 16;
                std::memcpy(y + i, &b, 4);
            }
            break;
        case T_Q8_0: dequant_q8_0(src, y, n); break;
        case T_Q4_K: dequant_q4_K(src, y, n); break;
        case T_Q5_K: dequant_q5_K(src, y, n); break;
        case T_Q6_K: dequant_q6_K(src, y, n); break;
        case T_Q2_K: case T_Q3_K: case T_IQ2_S: case T_IQ3_S: case T_IQ4_XS:
            dequant_import(type, src, y, n); break;
        default: break;
    }
}

/// True for the types dequant_row() handles.
inline bool can_dequant(uint32_t type) {
    switch (type) {
        case T_F32: case T_F16: case T_BF16: case T_Q8_0: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_Q2_K: case T_Q3_K: case T_IQ2_S: case T_IQ3_S: case T_IQ4_XS: return true;
        default: return false;
    }
}

/// ggml's reference Q8_0 quantizer (quantize_row_q8_0_ref), into the GPU's split layout: n int8 quants and n/32
/// fp16 scales.
inline void quantize_row_q8_0(const float* x, int8_t* qs, uint16_t* d, int64_t n) {
    for (int64_t i = 0; i < n / 32; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[i * 32 + j]));
        const float dd = amax / 127.0f;
        const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
        d[i] = f32_to_fp16(dd);
        for (int j = 0; j < 32; ++j) qs[i * 32 + j] = (int8_t)std::lround(x[i * 32 + j] * id);
    }
}

}  // namespace sq
