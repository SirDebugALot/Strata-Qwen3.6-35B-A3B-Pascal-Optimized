// CPU projection for an optional host-resident Q8_0 LM head.  AVX2/F16C is required, as for CPU MoE.
#pragma once

#include <cstdint>

namespace sq {

// Weight quants are row-major; weight scales and activation scales are per 32 values.
// Computes rows [begin, end) of one activation, preserving output values outside that range.
void cpu_lm_head_q8_rows(const int8_t* wq, const uint16_t* wd, const int8_t* aq, const float* ad,
                         float* out, int cols, int begin, int end);

// Token-major activations/output, with output stride rows. Reuses each weight block for 1..4 tokens,
// keeping the same per-token accumulation order as cpu_lm_head_q8_rows.
void cpu_lm_head_q8_rows_multi(const int8_t* wq, const uint16_t* wd, const int8_t* aq, const float* ad,
                              float* out, int cols, int rows, int begin, int end, int nt);

}  // namespace sq
