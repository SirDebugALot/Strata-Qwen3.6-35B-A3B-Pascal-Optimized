// Standalone CPU research prototype. Not used by the engine or its expert-cache format.
#pragma once
#include <cstddef>
#include <cstdint>
#include "cpu_iq3_nibble.hpp"

namespace sq { struct Q8K; }
namespace iq3_probe {

// Preserve source FP16 d and the exact codebook integers/subscales. No requantization.
// Two-byte alignment needs no per-block padding; coefficient loads are intentionally unaligned.
struct Block {
    uint16_t d;
    int8_t coefficients[256];
    int8_t subscales[8];
};
static_assert(sizeof(Block) == 266, "lossless IQ3_S probe layout");

// IQ3_S magnitudes are exactly {1,3,5,7,9,11,13,15}. Including sign gives16 values, so a nibble
// stores the exact codebook coefficient. Each half packs values0..15 low and16..31 high.
using NibbleBlock = sq::native_iq::Iq3NibbleBlock;

void decode(const uint8_t* packed, Block* decoded, size_t blocks);
void decode_nibbles(const uint8_t* packed, NibbleBlock* decoded, size_t blocks);
// Same lane sums, block FMA order and hsum as native IQ3_S, including activation -128 sign wrapping.
void rows(const Block* weights, int blocks_per_row, const sq::Q8K* const* activation,
          int nt, float* const* out, int r0, int r1);
void rows_nibbles(const NibbleBlock* weights, int blocks_per_row, const sq::Q8K* const* activation,
                  int nt, float* const* out, int r0, int r1);
}
