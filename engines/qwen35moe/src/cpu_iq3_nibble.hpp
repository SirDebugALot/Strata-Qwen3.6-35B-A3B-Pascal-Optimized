// Lossless CPU representation of IQ3_S down weights. Original GPU expert blobs are unchanged.
#pragma once
#include <cstddef>
#include <cstdint>
namespace sq { struct Q8K; }
namespace sq::native_iq {
struct Iq3NibbleBlock {
    uint16_t d;
    uint8_t coefficients[128]; // per32 values: first16 low nibble, last16 high nibble
    int8_t subscales[8];
};
static_assert(sizeof(Iq3NibbleBlock) == 138, "lossless IQ3_S nibble layout");
void decode_iq3_nibbles(const uint8_t* source, Iq3NibbleBlock* output, size_t blocks);
void iq3_nibble_rows(const Iq3NibbleBlock* weights, int blocks_per_row, const Q8K* const* activation,
                     int nt, float* const* out, int r0, int r1);
// Default-off, immutable process setting. Production dispatch keeps the original NT1 path.
bool iq3_nibble_wide_enabled();
// Same existing lossless weights; only AVX2 coefficient expansion and PF2048 hints differ.
void iq3_nibble_rows_wide_prefetch(const Iq3NibbleBlock* weights, int blocks_per_row,
                                  const Q8K* const* activation, int nt, float* const* out, int r0, int r1);
}
