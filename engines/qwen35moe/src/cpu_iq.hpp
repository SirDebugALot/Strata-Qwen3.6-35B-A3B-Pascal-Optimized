#pragma once
#include "cpu_moe.hpp"
namespace sq::native_iq {
float dot_q2k(const void*, const Q8K*, int);
float dot_q3k(const void*, const Q8K*, int);
float dot_iq2s(const void*, const Q8K*, int);
float dot_iq3s(const void*, const Q8K*, int);
float dot_iq4xs(const void*, const Q8K*, int);
inline bool k23_supported(uint32_t type) { return type == T_Q2_K || type == T_Q3_K; }
// Several activation rows reuse each original Q2/Q3 weight unpack. Each token retains the scalar dot order.
void k23_rows(uint32_t type, const uint8_t* weights, size_t row_bytes, int cols, const void* const* acts,
              int nt, float* const* outputs, int r0, int r1);
void k23_gu_rows(uint32_t type, const uint8_t* weights, size_t row_bytes, size_t up_offset, int cols,
                 const void* const* acts, int nt, float* const* outputs, int r0, int r1);
}
